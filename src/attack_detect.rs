//! Layer III **signal-driven attack detector** — the
//! encoder-side classifier that picks, per granule, whether the
//! granule contains a "transient" event sharp enough to warrant a
//! window-switched short block over the long block. The detection
//! drives the geometry of [`crate::block_type_sm::BlockTypeStateMachine`],
//! which schedules the §C.1.5.2
//! LONG → START → SHORT → STOP → LONG transition window the decoder
//! requires to splice 36-point and 12-point MDCT outputs without an
//! audible discontinuity.
//!
//! # Why a heuristic
//!
//! ISO/IEC 11172-3:1993 leaves the encoder-side block-type decision
//! entirely to the implementation. Annex C §C.1.5 references the
//! informative Annex D psychoacoustic model — but the algorithm
//! itself is non-normative; any encoder is free to choose a
//! window-switching policy that produces a syntactically valid
//! bitstream. This module implements a clean-room **signal-driven**
//! heuristic that needs no psychoacoustic-spreading function, no
//! masking-curve estimation, and no auxiliary tables — only the
//! PCM-domain sum-of-squared samples per granule subframe (the same
//! quantity Layer III's short-block path partitions the granule into:
//! three 192-sample subframes per granule, corresponding to the three
//! 12-point short MDCTs of §2.4.2.7).
//!
//! The heuristic mirrors the well-known principle that **a transient
//! is a localised energy burst on a quiet background**. A subframe
//! whose energy is much larger than the recent running estimate is
//! classified as carrying an attack; the granule is flagged when
//! *any* of its three subframes crosses the threshold.
//!
//! # Method
//!
//! Per granule the PCM run of 576 samples is split into three
//! consecutive 192-sample subframes. Each subframe's energy is the
//! sum of squared sample magnitudes
//!
//! ```text
//! E_k = Σ_{i ∈ subframe_k} pcm[i] · pcm[i]
//! ```
//!
//! The detector maintains a running **ambient-energy estimate**
//! `E_amb` (an exponentially smoothed running mean of the
//! `min`-of-subframe-energies of recent granules — the *floor* of the
//! signal, less perturbed by the same bursts the detector is looking
//! for). On each granule:
//!
//! 1. Compute `[E_0, E_1, E_2]`.
//! 2. Compute the per-subframe **attack ratio**
//!    `r_k = E_k / max(E_amb, ε)`. The `ε` floor (`f64::MIN_POSITIVE`,
//!    promoted to a small positive constant `SILENCE_FLOOR`) prevents
//!    division-by-zero on a leading-silence granule and bounds the
//!    detector's behaviour on pathological inputs.
//! 3. The granule is flagged as carrying an attack iff
//!    `max_k r_k > threshold`.
//! 4. The ambient estimate is updated with a single-pole IIR using a
//!    leakage factor (default [`DEFAULT_AMBIENT_LEAK`] = `0.5`, knob
//!    on [`AttackDetectorParams::leak`]) against `min_k E_k`: this
//!    keeps the ambient slow to rise (so a quiet decay following an
//!    attack doesn't drag the threshold up and miss the *next* attack)
//!    but responsive enough on a steady-state signal to converge to
//!    the correct floor within ≈ 4 granules at the default leak. r164
//!    promoted this from a private constant into a per-instance knob
//!    so callers can tune adaptation rate independently of the attack
//!    threshold (e.g. a slower leak for material with a steady
//!    background and rare transients, a faster leak for material with
//!    a gradually-swelling background).
//!
//! # Threshold guidance
//!
//! A threshold around **`10.0`** (one order of magnitude over
//! ambient) detects loud, sharp transients while leaving sustained
//! tones, steady noise, and slow swells classified as non-attack
//! (long block). Higher thresholds (≥ `30`) reserve short-blocks for
//! only the most extreme bursts; lower thresholds (≤ `3`) make the
//! detector over-aggressive (almost any modulated signal trips). The
//! default in [`AttackDetector::new`] is `10.0`.
//!
//! No external reference implementation was consulted while writing
//! this heuristic. Every constant and every formula in this module
//! is justified by the clean-room reasoning above (energy
//! localisation, threshold semantics, IIR-leakage stability).

use crate::stream_encoder::SAMPLES_PER_GRANULE;

/// Number of subframes per granule for attack-detection purposes —
/// three, matching Layer III's three short-window MDCT partition of a
/// granule (ISO/IEC 11172-3:1993 §2.4.2.7).
pub const SUBFRAMES_PER_GRANULE: usize = 3;

/// Samples per subframe — `SAMPLES_PER_GRANULE / SUBFRAMES_PER_GRANULE`.
pub const SAMPLES_PER_SUBFRAME: usize = SAMPLES_PER_GRANULE / SUBFRAMES_PER_GRANULE;

/// Lower bound on the ambient-energy estimate to prevent
/// divide-by-zero on a leading-silence granule. Slightly larger than
/// `f64::MIN_POSITIVE` so the attack-ratio of a non-silent first
/// granule doesn't explode to ∞: a 192-sample silent subframe with
/// even one ±1/32768 sample contributes ≈ `1 / 2^30`, and we want the
/// ratio for that case to be bounded, not infinite.
pub const SILENCE_FLOOR: f64 = 1.0e-30;

/// Default exponential-leakage factor for the ambient-energy estimate.
/// A value of `0.5` means the estimate moves halfway toward the new
/// floor sample per granule — slow enough to ride a sustained
/// transient train without rising into it, but fast enough to track
/// genuine background-level changes within a handful of granules.
///
/// r164 promoted this from a private constant into the public default
/// for the [`AttackDetectorParams::leak`] knob. Callers tuning the
/// detector's adaptation rate can pass a different value to
/// [`AttackDetector::with_params`]: smaller values (e.g. `0.1`) make
/// the ambient slower to adapt — useful for material where the
/// background is steady and transients arrive irregularly — and larger
/// values (e.g. `0.9`) make it faster to follow a swelling background,
/// at the cost of a stronger tendency to absorb sustained transient
/// trains into the ambient (and thereby miss subsequent attacks of the
/// same magnitude).
pub const DEFAULT_AMBIENT_LEAK: f64 = 0.5;

/// Default attack-detection threshold (subframe-to-ambient ratio that
/// the loudest subframe must exceed for the granule to be flagged).
/// Empirically a 10× ratio separates clear transients from
/// steady-state modulation; see the module doc for tuning guidance.
pub const DEFAULT_ATTACK_THRESHOLD: f64 = 10.0;

/// Tunable parameters for the [`AttackDetector`] — the §2.4.3.4.10
/// window-switching policy knobs that drive the encoder-side block-type
/// scheduler. The two knobs trade off **sensitivity** (how loud a
/// burst has to be, relative to the background, to be flagged) against
/// **adaptation** (how quickly the running ambient catches up to a
/// changing background).
///
/// Both fields are validated by [`AttackDetector::with_params`]:
/// non-finite, non-positive, or out-of-domain values fall back to the
/// corresponding `DEFAULT_*` constant, matching the
/// [`AttackDetector::with_threshold`] coercion contract that already
/// shipped in earlier rounds.
///
/// # Field semantics
///
/// * [`Self::threshold`] — the subframe-to-ambient energy ratio that
///   the loudest subframe in a granule must exceed for the detector to
///   report `true`. Larger = fewer short blocks (more conservative);
///   smaller = more short blocks (more aggressive). Default
///   [`DEFAULT_ATTACK_THRESHOLD`] (`10.0`).
/// * [`Self::leak`] — the per-granule IIR leakage factor against the
///   running ambient. Must lie strictly in `(0, 1)`: `0` would freeze
///   the ambient forever at its seed value and `1` would replace it on
///   every granule (defeating the purpose of running-min smoothing).
///   The new ambient is computed as
///   `ambient ← leak · min_k E_k + (1 − leak) · ambient`.
///   Default [`DEFAULT_AMBIENT_LEAK`] (`0.5`).
///
/// No external reference implementation was consulted while choosing
/// either knob's semantics. The ratio-of-energies threshold and the
/// IIR-leakage adaptation are both consequences of the clean-room
/// reasoning at the top of this module (energy localisation +
/// adapt-to-floor).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttackDetectorParams {
    /// Subframe-to-ambient ratio above which a granule is flagged.
    /// See [`DEFAULT_ATTACK_THRESHOLD`].
    pub threshold: f64,
    /// Per-granule IIR leakage factor for the ambient estimate; must
    /// be in `(0, 1)`. See [`DEFAULT_AMBIENT_LEAK`].
    pub leak: f64,
}

impl AttackDetectorParams {
    /// Construct a parameter pair using both DEFAULT_* values.
    #[must_use]
    pub fn new() -> Self {
        Self {
            threshold: DEFAULT_ATTACK_THRESHOLD,
            leak: DEFAULT_AMBIENT_LEAK,
        }
    }
}

impl Default for AttackDetectorParams {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute the sum-of-squared samples of a slice of PCM.
///
/// ```text
/// E = Σ_i pcm[i] · pcm[i]
/// ```
///
/// The result is non-negative by construction. Returned as `f64` to
/// keep the subsequent ratio arithmetic stable across the
/// 8-orders-of-magnitude dynamic range of typical PCM material.
#[must_use]
pub fn subframe_energy(pcm: &[f32]) -> f64 {
    let mut acc = 0.0f64;
    for &s in pcm {
        let sd = f64::from(s);
        acc += sd * sd;
    }
    acc
}

/// Compute the three subframe energies of one granule's 576-sample
/// PCM run.
#[must_use]
pub fn granule_subframe_energies(pcm: &[f32; SAMPLES_PER_GRANULE]) -> [f64; SUBFRAMES_PER_GRANULE] {
    let mut out = [0.0f64; SUBFRAMES_PER_GRANULE];
    for (k, slot) in out.iter_mut().enumerate() {
        let lo = k * SAMPLES_PER_SUBFRAME;
        let hi = lo + SAMPLES_PER_SUBFRAME;
        *slot = subframe_energy(&pcm[lo..hi]);
    }
    out
}

/// Stateful signal-driven attack detector. Holds a single-pole IIR
/// ambient-energy estimate updated per granule and a configurable
/// attack-threshold ratio. One detector per channel — the ambient
/// floor is content-driven and the two channels of a stereo file can
/// have very different floor levels.
///
/// Two knobs are exposed on construction (see
/// [`AttackDetectorParams`]):
///
/// * `threshold` — the subframe-to-ambient ratio at which the granule
///   is flagged. Default [`DEFAULT_ATTACK_THRESHOLD`] = `10.0`.
/// * `leak` — the IIR adaptation rate of the ambient estimate. Default
///   [`DEFAULT_AMBIENT_LEAK`] = `0.5`. (Promoted from a private
///   constant to a per-instance knob in r164.)
#[derive(Debug, Clone)]
pub struct AttackDetector {
    /// Running ambient-energy estimate (the running `min`-floor of
    /// recent granule subframes; not the running *mean* of all
    /// subframes — that would be biased upward by every transient).
    ambient: f64,
    /// Attack threshold: the granule is flagged iff
    /// `max(E_k) / max(ambient, SILENCE_FLOOR) > threshold`.
    threshold: f64,
    /// Per-granule IIR leakage against `min_k E_k`. The new ambient is
    /// `leak · min_k E_k + (1 − leak) · ambient`. Must be in `(0, 1)`.
    /// Validated by the constructors (`with_threshold`,
    /// `with_params`); pathological caller values fall back to
    /// [`DEFAULT_AMBIENT_LEAK`].
    leak: f64,
}

impl AttackDetector {
    /// Construct a detector with [`AttackDetectorParams::default`].
    #[must_use]
    pub fn new() -> Self {
        Self::with_params(AttackDetectorParams::new())
    }

    /// Construct a detector with a caller-chosen attack-ratio
    /// threshold (subframe-to-ambient ratio) and the default leakage
    /// factor. Larger threshold = more conservative (fewer short
    /// blocks); smaller = more aggressive (more short blocks). See the
    /// module docs for guidance. Equivalent to
    /// `with_params(AttackDetectorParams { threshold, leak:
    /// DEFAULT_AMBIENT_LEAK })`.
    #[must_use]
    pub fn with_threshold(threshold: f64) -> Self {
        Self::with_params(AttackDetectorParams {
            threshold,
            leak: DEFAULT_AMBIENT_LEAK,
        })
    }

    /// Construct a detector with both tuning knobs. Out-of-domain
    /// values are silently coerced to their `DEFAULT_*` counterparts:
    ///
    /// * `params.threshold` ≤ 0 or non-finite → [`DEFAULT_ATTACK_THRESHOLD`].
    /// * `params.leak` outside `(0, 1)` or non-finite → [`DEFAULT_AMBIENT_LEAK`].
    ///
    /// The two knobs are validated independently — supplying a bad
    /// `threshold` does not force a fallback on `leak`, and vice
    /// versa.
    #[must_use]
    pub fn with_params(params: AttackDetectorParams) -> Self {
        let threshold = if params.threshold.is_finite() && params.threshold > 0.0 {
            params.threshold
        } else {
            DEFAULT_ATTACK_THRESHOLD
        };
        // Strictly open interval: `0.0` would freeze the ambient,
        // `1.0` would replace it on every granule.
        let leak = if params.leak.is_finite() && params.leak > 0.0 && params.leak < 1.0 {
            params.leak
        } else {
            DEFAULT_AMBIENT_LEAK
        };
        Self {
            ambient: 0.0,
            threshold,
            leak,
        }
    }

    /// Current attack-ratio threshold (subframe-to-ambient).
    #[must_use]
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Current ambient-estimate IIR leakage factor (`(0, 1)`).
    #[must_use]
    pub fn leak(&self) -> f64 {
        self.leak
    }

    /// The effective tuning parameters this detector was constructed
    /// with, after the [`Self::with_params`] coercion of out-of-domain
    /// values. Useful for debugging / round-tripping config.
    #[must_use]
    pub fn params(&self) -> AttackDetectorParams {
        AttackDetectorParams {
            threshold: self.threshold,
            leak: self.leak,
        }
    }

    /// Current ambient-energy estimate. Mostly useful for testing and
    /// diagnostics; the operational decision is via
    /// [`Self::classify`].
    #[must_use]
    pub fn ambient(&self) -> f64 {
        self.ambient
    }

    /// Reset the detector's state. The ambient estimate is cleared
    /// back to zero so the next granule's `min`-floor seeds it
    /// directly (rather than being smoothed against the previous
    /// stream's floor).
    pub fn reset(&mut self) {
        self.ambient = 0.0;
    }

    /// Classify the supplied granule PCM. Returns `true` when the
    /// loudest subframe's energy exceeds `threshold ×` the running
    /// ambient floor — i.e., the granule is flagged as carrying an
    /// attack — and `false` otherwise.
    ///
    /// **Side effect:** the detector's ambient estimate is updated
    /// in-place. A repeated `classify` call with the same PCM input
    /// will not return the same answer in general, because the
    /// internal estimate has advanced. Callers that want a
    /// non-destructive peek should clone the detector first.
    pub fn classify(&mut self, pcm: &[f32; SAMPLES_PER_GRANULE]) -> bool {
        let energies = granule_subframe_energies(pcm);
        let e_max = energies
            .iter()
            .copied()
            .fold(0.0f64, |a, b| if b > a { b } else { a });
        let e_min = energies
            .iter()
            .copied()
            .fold(f64::INFINITY, |a, b| if b < a { b } else { a });
        let floor = self.ambient.max(SILENCE_FLOOR);
        let ratio = e_max / floor;
        let flagged = ratio > self.threshold;
        // Update the ambient estimate using the granule's *floor*
        // subframe energy: the running min better tracks the true
        // background, since a transient pulls e_max up but typically
        // leaves at least one subframe near the underlying signal
        // level.
        //
        // On the first call (ambient == 0), seed directly with
        // `e_min` so the detector doesn't fire spuriously on the
        // first non-silent granule.
        if self.ambient == 0.0 {
            self.ambient = e_min.max(SILENCE_FLOOR);
        } else {
            self.ambient = self.leak * e_min + (1.0 - self.leak) * self.ambient;
        }
        flagged
    }
}

impl Default for AttackDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 192-sample subframe of zeroes has zero energy.
    #[test]
    fn silent_subframe_has_zero_energy() {
        let s = [0.0f32; SAMPLES_PER_SUBFRAME];
        assert_eq!(subframe_energy(&s), 0.0);
    }

    /// A 192-sample subframe of unit DC has energy 192.
    #[test]
    fn unit_dc_subframe_energy_is_n() {
        let s = [1.0f32; SAMPLES_PER_SUBFRAME];
        assert_eq!(subframe_energy(&s), SAMPLES_PER_SUBFRAME as f64);
    }

    /// A 576-sample granule of unit DC has equal energies in all
    /// three subframes.
    #[test]
    fn dc_granule_subframe_energies_are_uniform() {
        let g = [1.0f32; SAMPLES_PER_GRANULE];
        let e = granule_subframe_energies(&g);
        assert_eq!(e[0], e[1]);
        assert_eq!(e[1], e[2]);
        assert_eq!(e[0], SAMPLES_PER_SUBFRAME as f64);
    }

    /// A pure sine wave (constant-amplitude steady-state) is **not**
    /// flagged as carrying an attack: the three subframes have
    /// closely-matched energies, so e_max ≈ e_min ≈ ambient and the
    /// ratio stays around 1.
    #[test]
    fn pure_sine_not_flagged() {
        let mut det = AttackDetector::new();
        // 440 Hz at 44.1 kHz → ω·t = 2π · 440/44100 · t
        let omega = 2.0 * std::f32::consts::PI * 440.0 / 44100.0;
        // Run a few granules to settle the ambient estimate, then
        // assert no further granule is flagged.
        for gr in 0..8 {
            let mut g = [0.0f32; SAMPLES_PER_GRANULE];
            for (i, slot) in g.iter_mut().enumerate() {
                let t = (gr * SAMPLES_PER_GRANULE + i) as f32;
                *slot = 0.25 * (omega * t).sin();
            }
            let flagged = det.classify(&g);
            // After the first granule the ambient is settled enough
            // that no later granule should fire.
            if gr >= 1 {
                assert!(!flagged, "pure sine flagged at granule {gr}");
            }
        }
    }

    /// A unit step inside an otherwise quiet granule is flagged as an
    /// attack — the late subframes carry orders of magnitude more
    /// energy than the early one.
    #[test]
    fn step_burst_is_flagged() {
        let mut det = AttackDetector::new();
        // First "warm-up" granule: nearly silent (-90 dB white floor)
        // so the detector seeds a tiny ambient. We use a deterministic
        // pseudo-noise so the test is reproducible without rand.
        let mut g0 = [0.0f32; SAMPLES_PER_GRANULE];
        for (i, slot) in g0.iter_mut().enumerate() {
            // Small ±1 LSB-level wobble.
            *slot = if i % 7 == 0 { 1.0e-4 } else { -1.0e-4 };
        }
        let _ = det.classify(&g0);

        // Second granule: silent in the first subframe, then a hard
        // step up to ±0.5 in subframes 1 and 2. Subframe 0's energy
        // is tiny; subframes 1 and 2 are ~48× higher (relative to
        // the seeded ambient).
        let mut g1 = [0.0f32; SAMPLES_PER_GRANULE];
        for (i, slot) in g1.iter_mut().enumerate() {
            *slot = if i < SAMPLES_PER_SUBFRAME {
                if i % 7 == 0 {
                    1.0e-4
                } else {
                    -1.0e-4
                }
            } else if i % 2 == 0 {
                0.5
            } else {
                -0.5
            };
        }
        let flagged = det.classify(&g1);
        assert!(flagged, "step burst was not flagged as attack");
    }

    /// A granule consisting of pure silence is **not** flagged, and
    /// the ambient floor stays bounded (no division-by-zero
    /// blowup).
    #[test]
    fn pure_silence_not_flagged() {
        let mut det = AttackDetector::new();
        let g = [0.0f32; SAMPLES_PER_GRANULE];
        let flagged = det.classify(&g);
        assert!(!flagged);
        // Ambient should be the silence floor (not zero, not NaN, not
        // infinity).
        assert!(det.ambient() >= SILENCE_FLOOR);
        assert!(det.ambient().is_finite());
    }

    /// Once a step burst has passed, the detector's ambient estimate
    /// rises far enough that a subsequent burst of similar magnitude
    /// is **not** flagged (the detector adapts to the new
    /// background level). This is the desired "no false-fire on
    /// sustained modulation" behaviour.
    #[test]
    fn detector_adapts_after_repeated_bursts() {
        let mut det = AttackDetector::new();
        // Five granules of the same loud-everywhere signal: after a
        // few iterations the ambient should rise so subsequent
        // identical granules are not flagged.
        let mut burst = [0.0f32; SAMPLES_PER_GRANULE];
        for (i, slot) in burst.iter_mut().enumerate() {
            *slot = if i % 2 == 0 { 0.5 } else { -0.5 };
        }
        let mut fired_count = 0;
        for _ in 0..5 {
            if det.classify(&burst) {
                fired_count += 1;
            }
        }
        // At most one early granule may have fired before adaptation.
        assert!(
            fired_count <= 1,
            "detector kept firing on steady signal: fired_count={fired_count}"
        );
    }

    /// A genuine attack (a sharp click) after a long quiet run **is**
    /// flagged — the ambient is settled near zero and the click's
    /// energy is orders of magnitude higher. This confirms the
    /// adapt-to-floor case (the inverse of the adapt-after-bursts
    /// case above).
    #[test]
    fn click_after_silence_is_flagged() {
        let mut det = AttackDetector::new();
        let g_silent = [0.0f32; SAMPLES_PER_GRANULE];
        for _ in 0..4 {
            let _ = det.classify(&g_silent);
        }
        // Now a granule with a single big sample at the start of
        // subframe 1.
        let mut g = [0.0f32; SAMPLES_PER_GRANULE];
        g[SAMPLES_PER_SUBFRAME] = 0.9;
        let flagged = det.classify(&g);
        assert!(flagged, "click after silence was not flagged");
    }

    /// `with_threshold` rejects non-finite or non-positive values by
    /// falling back to the default.
    #[test]
    fn invalid_threshold_falls_back_to_default() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -1.0] {
            let det = AttackDetector::with_threshold(bad);
            assert_eq!(det.threshold(), DEFAULT_ATTACK_THRESHOLD);
        }
    }

    /// `reset` clears the ambient back to zero so the next granule
    /// re-seeds the floor from its own subframes.
    #[test]
    fn reset_clears_ambient() {
        let mut det = AttackDetector::new();
        let g = [0.1f32; SAMPLES_PER_GRANULE];
        let _ = det.classify(&g);
        assert!(det.ambient() > 0.0);
        det.reset();
        assert_eq!(det.ambient(), 0.0);
    }

    // r164 — finer §2.4.3.4.10 attack-detector knobs:
    // `AttackDetectorParams` exposes the IIR leakage factor as a
    // per-instance tunable alongside the existing threshold, with
    // identical validation semantics (out-of-domain → DEFAULT_*).

    /// `AttackDetectorParams::default` and `::new` agree, and surface
    /// the DEFAULT_* constants verbatim. This pins the
    /// backwards-compatible default behaviour: every existing caller
    /// goes through `AttackDetector::new` → `with_params(default())`,
    /// so the per-instance leak must equal the pre-r164 hardcoded
    /// `0.5`.
    #[test]
    fn default_params_match_documented_constants() {
        let p = AttackDetectorParams::default();
        assert_eq!(p, AttackDetectorParams::new());
        assert_eq!(p.threshold, DEFAULT_ATTACK_THRESHOLD);
        assert_eq!(p.leak, DEFAULT_AMBIENT_LEAK);
        assert_eq!(DEFAULT_AMBIENT_LEAK, 0.5);
    }

    /// `with_params` round-trips its in-domain inputs verbatim and
    /// exposes them via `threshold()` / `leak()` / `params()`.
    #[test]
    fn with_params_round_trips_in_domain_values() {
        let p = AttackDetectorParams {
            threshold: 7.5,
            leak: 0.25,
        };
        let det = AttackDetector::with_params(p);
        assert_eq!(det.threshold(), 7.5);
        assert_eq!(det.leak(), 0.25);
        assert_eq!(det.params(), p);
        // Ambient seed is always zero on construction; classify
        // re-seeds from the first granule.
        assert_eq!(det.ambient(), 0.0);
    }

    /// `with_params` clamps a bad `threshold` to
    /// `DEFAULT_ATTACK_THRESHOLD` independently of `leak`, and
    /// vice-versa — neither bad knob may force the other to its
    /// default. The two coercions are independent by design so a
    /// caller who provides one good knob keeps it.
    #[test]
    fn with_params_validates_each_knob_independently() {
        // Good threshold, bad leak.
        for bad_leak in [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            0.0,
            -0.1,
            1.0,
            1.5,
        ] {
            let det = AttackDetector::with_params(AttackDetectorParams {
                threshold: 4.0,
                leak: bad_leak,
            });
            assert_eq!(
                det.threshold(),
                4.0,
                "good threshold dropped on bad leak={bad_leak}"
            );
            assert_eq!(
                det.leak(),
                DEFAULT_AMBIENT_LEAK,
                "bad leak={bad_leak} not coerced"
            );
        }
        // Bad threshold, good leak.
        for bad_thr in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -1.0] {
            let det = AttackDetector::with_params(AttackDetectorParams {
                threshold: bad_thr,
                leak: 0.3,
            });
            assert_eq!(
                det.threshold(),
                DEFAULT_ATTACK_THRESHOLD,
                "bad threshold={bad_thr} not coerced"
            );
            assert_eq!(
                det.leak(),
                0.3,
                "good leak dropped on bad threshold={bad_thr}"
            );
        }
    }

    /// `with_threshold` is documented to be equivalent to
    /// `with_params { threshold, leak: DEFAULT_AMBIENT_LEAK }`. The
    /// existing pre-r164 callers go through `with_threshold`, so the
    /// effective leak must match the legacy `0.5`.
    #[test]
    fn with_threshold_uses_default_leak() {
        let det = AttackDetector::with_threshold(5.0);
        assert_eq!(det.leak(), DEFAULT_AMBIENT_LEAK);
        assert_eq!(det.threshold(), 5.0);
    }

    /// A smaller leak makes the ambient slower to adapt to a steady
    /// loud signal — so a detector with `leak = 0.1` keeps flagging
    /// longer than a `leak = 0.9` detector on the same repeated burst.
    /// This validates that the knob is actually wired into
    /// `classify`'s IIR update.
    ///
    /// Construction: warm up both detectors with a quiet granule so
    /// the ambient seeds tiny, then feed an identical sequence of
    /// loud bursts and count flag-fires. The slow-leak detector
    /// **must** fire at least as many times as the fast-leak one
    /// (typically strictly more), because its ambient takes longer to
    /// catch up to the new background level.
    #[test]
    fn slower_leak_keeps_firing_longer_than_faster_leak() {
        fn make(leak: f64) -> AttackDetector {
            AttackDetector::with_params(AttackDetectorParams {
                threshold: DEFAULT_ATTACK_THRESHOLD,
                leak,
            })
        }
        let mut slow = make(0.05); // very slow adaptation
        let mut fast = make(0.95); // very fast adaptation

        // Quiet seed granule (1e-4 magnitude).
        let mut g_quiet = [0.0f32; SAMPLES_PER_GRANULE];
        for (i, slot) in g_quiet.iter_mut().enumerate() {
            *slot = if i % 7 == 0 { 1.0e-4 } else { -1.0e-4 };
        }
        let _ = slow.classify(&g_quiet);
        let _ = fast.classify(&g_quiet);

        // Loud burst (square wave at ±0.5) repeated for several
        // granules. Both detectors should fire on the first burst;
        // the slow-leak detector keeps firing longer because its
        // ambient catches up to the new floor more slowly.
        let mut g_loud = [0.0f32; SAMPLES_PER_GRANULE];
        for (i, slot) in g_loud.iter_mut().enumerate() {
            *slot = if i % 2 == 0 { 0.5 } else { -0.5 };
        }
        let mut slow_fires = 0;
        let mut fast_fires = 0;
        for _ in 0..10 {
            if slow.classify(&g_loud) {
                slow_fires += 1;
            }
            if fast.classify(&g_loud) {
                fast_fires += 1;
            }
        }
        // Both must fire at least once (the first burst). The slow
        // detector must fire >= the fast detector — and on this
        // construction we expect strictly more, with the fast one
        // adapting within ~2 granules.
        assert!(slow_fires >= 1, "slow detector never fired");
        assert!(
            slow_fires >= fast_fires,
            "slow leak fired less than fast leak: slow={slow_fires} fast={fast_fires}"
        );
        // Fast leak adapts within a handful of granules and stops
        // firing — sanity check on the fast-end of the knob.
        assert!(
            fast_fires <= 3,
            "fast leak failed to adapt: fast_fires={fast_fires}"
        );
    }

    /// Boundary check: a `leak` exactly at `0.0` or `1.0` is rejected
    /// (closed-interval values would either freeze or replace the
    /// ambient and defeat the IIR's purpose).
    #[test]
    fn leak_boundary_values_are_rejected() {
        for boundary in [0.0, 1.0] {
            let det = AttackDetector::with_params(AttackDetectorParams {
                threshold: DEFAULT_ATTACK_THRESHOLD,
                leak: boundary,
            });
            assert_eq!(
                det.leak(),
                DEFAULT_AMBIENT_LEAK,
                "leak boundary {boundary} should fall back to default"
            );
        }
    }

    /// `AttackDetector::new` and `with_params(default())` produce
    /// equivalent detectors — same threshold, same leak, same seed
    /// ambient. This pins the no-args constructor as a transparent
    /// alias for the default params, the only documented relationship
    /// between the two constructors.
    #[test]
    fn new_equivalent_to_with_params_default() {
        let a = AttackDetector::new();
        let b = AttackDetector::with_params(AttackDetectorParams::default());
        assert_eq!(a.threshold(), b.threshold());
        assert_eq!(a.leak(), b.leak());
        assert_eq!(a.ambient(), b.ambient());
        assert_eq!(a.params(), b.params());
    }
}
