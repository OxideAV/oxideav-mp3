//! Layer III **mixed-vs-pure-short PCM-domain classifier** — the
//! encoder-side companion to the [`crate::attack_detect::AttackDetector`]
//! that decides, on a granule the attack detector has flagged as
//! transient, whether the transient is broadband (use a *pure-short*
//! block) or band-limited to the upper frequencies (use a *mixed*
//! block: §2.4.3.4.10.3 carves the two lowest subbands long, the rest
//! short).
//!
//! # Why
//!
//! ISO/IEC 11172-3:1993 §2.4.3.4.10.3 defines a mixed block as
//! `block_type = 2` with `mixed_block_flag = 1`: the two lowest
//! subbands (lines 0..36, ≤ ~330 Hz at 44.1 kHz / ≤ ~360 Hz at 48 kHz)
//! are transformed with the 36-point sine window (the long-block
//! transform), and subbands 2..31 with the three 12-point sine
//! windows (the short-block transform). The trade-off is purely a
//! frequency-resolution-vs-time-resolution split: the long-window
//! lower subbands keep their full per-band scalefactor resolution
//! against a steady low-frequency tonal background, while the
//! short-window upper subbands resolve the transient in time without
//! smearing it across the granule.
//!
//! Mixed is **only** appropriate when the low-frequency content of
//! the granule is approximately stationary across the three short
//! subframes — if the low band itself is bursting (e.g. a kick-drum
//! attack that carries energy at < 200 Hz too), the long-window lower
//! subbands smear that low-band transient in time, defeating the
//! purpose. The §C.1.5.2 short-block decision then reverts to the
//! pure-short geometry where every subband resolves the burst in
//! time.
//!
//! # Method
//!
//! Per granule PCM run of [`SAMPLES_PER_GRANULE`] samples, this
//! module:
//!
//! 1. **Low-pass filter** the PCM with a clean-room first-order
//!    moving-average kernel `y[n] = (x[n] + x[n-1]) / 2`. The
//!    transfer function `H(z) = (1 + z^{-1}) / 2` has a frequency
//!    response `|H(e^{jω})| = |cos(ω/2)|` — a smooth roll-off from
//!    `|H| = 1` at DC to `|H| = 0` at the Nyquist frequency, with
//!    `|H| = 1/√2` at the half-Nyquist frequency. At 44.1 kHz the
//!    half-Nyquist is 11.025 kHz; at 48 kHz it is 12 kHz. The
//!    attenuation is gentle but the kernel is enough to suppress the
//!    high-frequency component of a broadband transient so the
//!    subsequent subframe-energy ratio is dominated by the low-band
//!    behaviour the §2.4.3.4.10.3 mixed-block carve-out actually
//!    captures.
//!
//!    A sharper low-pass (e.g. a higher-order FIR scoped to ≤ 360 Hz)
//!    would track the geometric low-band edge more precisely, but the
//!    classifier needs only a *qualitative* yes/no decision — a
//!    high-frequency-only attack vs a broadband attack — and the
//!    one-tap kernel suffices for that distinction without bringing
//!    in tabulated filter coefficients (which would muddy the
//!    clean-room provenance).
//!
//! 2. **Per-subframe energy of the low-passed signal.** The same
//!    [`SUBFRAMES_PER_GRANULE`] partition the attack detector uses
//!    (three subframes per granule, matching the three short-window
//!    MDCTs); compute `E_lp_k = Σ_i lp[i] · lp[i]` per subframe.
//!
//! 3. **Low-band stability ratio.** Compute `r_lp = max_k E_lp_k /
//!    max(min_k E_lp_k, ε)`. If the low band is stationary across the
//!    granule this ratio is ≈ 1 (uniform energies). A broadband
//!    transient lifts at least one subframe's low-band energy well
//!    above the others, lifting the ratio.
//!
//! 4. **Decision.** If `r_lp ≤ threshold` the low band is judged
//!    stable and mixed is appropriate; otherwise the low band is
//!    judged transient and pure-short is appropriate. A default
//!    threshold of [`DEFAULT_MIXED_LOW_BAND_STABILITY`] = `4.0`
//!    keeps mixed reserved for clear cases (a 4× max-to-min low-band
//!    swing covers natural tonal jitter without crossing into burst
//!    territory); the caller can override via
//!    [`MixedClassifier::with_threshold`].
//!
//! # Stateless
//!
//! Unlike the [`crate::attack_detect::AttackDetector`] this
//! classifier has no IIR state — it makes its decision on the
//! current granule's PCM alone, deciding only whether the *shape* of
//! the low-band energy across the three subframes warrants a mixed
//! carve-out. The detector that decided the granule is transient at
//! all (the upstream [`crate::attack_detect::AttackDetector`]) is
//! the one that holds the running ambient.
//!
//! # Clean-room
//!
//! The §2.4.3.4.10.3 mixed-block geometry is the
//! only motivation; the one-tap LP kernel + subframe-energy ratio is
//! a straightforward clean-room realisation. No table is imported.

#[cfg(test)]
use crate::attack_detect::SAMPLES_PER_SUBFRAME;
use crate::attack_detect::{granule_subframe_energies, SILENCE_FLOOR};
use crate::stream_encoder::SAMPLES_PER_GRANULE;

/// Default low-band stability threshold: the maximum allowed ratio
/// `max_k E_lp_k / min_k E_lp_k` for the low-passed PCM's three
/// subframe energies. Below this the low band is judged stable and a
/// mixed block is appropriate; above this the low band is bursting
/// too and pure-short is preferred. A 4× swing is a reasonable
/// upper bound for natural low-band fluctuation: a flat tone trips
/// well under 1.5; a kick-drum onset typically lands above 10.
pub const DEFAULT_MIXED_LOW_BAND_STABILITY: f64 = 4.0;

/// Apply the one-tap moving-average low-pass `y[n] = (x[n] + x[n-1])
/// / 2` to the supplied granule PCM, returning the low-passed signal
/// in-place. `prev_last` is the sample immediately preceding `pcm[0]`
/// (typically the previous granule's last sample); pass `0.0` at
/// stream start. The filter has unity DC gain and a `|cos(ω/2)|`
/// magnitude response.
#[must_use]
pub fn low_pass_granule(
    pcm: &[f32; SAMPLES_PER_GRANULE],
    prev_last: f32,
) -> [f32; SAMPLES_PER_GRANULE] {
    let mut out = [0.0f32; SAMPLES_PER_GRANULE];
    let mut prev = prev_last;
    for (i, &x) in pcm.iter().enumerate() {
        out[i] = 0.5 * (x + prev);
        prev = x;
    }
    out
}

/// Compute the low-band stability ratio of a granule's PCM:
/// `max_k E_lp_k / max(min_k E_lp_k, ε)` over the three
/// [`SAMPLES_PER_SUBFRAME`]-sized subframes of the low-passed signal.
/// Returns 1.0 for a perfectly stationary low band and grows with
/// the imbalance.
#[must_use]
pub fn low_band_stability_ratio(pcm: &[f32; SAMPLES_PER_GRANULE], prev_last: f32) -> f64 {
    let lp = low_pass_granule(pcm, prev_last);
    let energies = granule_subframe_energies(&lp);
    let e_max = energies
        .iter()
        .copied()
        .fold(0.0f64, |a, b| if b > a { b } else { a });
    let e_min = energies
        .iter()
        .copied()
        .fold(f64::INFINITY, |a, b| if b < a { b } else { a });
    // `granule_subframe_energies` always returns three finite
    // non-negative values, so `e_max` and `e_min` are also finite and
    // non-negative; `e_max / max(e_min, ε)` is well-defined.
    e_max / e_min.max(SILENCE_FLOOR)
}

/// Stateful mixed-vs-pure-short classifier. Carries only the
/// previous granule's last PCM sample (so the one-tap LP filter has
/// a continuous history across granules) and the caller-chosen
/// threshold. One instance per channel.
#[derive(Debug, Clone)]
pub struct MixedClassifier {
    /// Last PCM sample from the previous granule, used to seed the
    /// one-tap moving-average LP filter at granule boundaries.
    /// Zero at construction.
    prev_last: f32,
    /// Maximum allowed `max_k E_lp_k / min_k E_lp_k` ratio for a
    /// mixed-block decision. See
    /// [`DEFAULT_MIXED_LOW_BAND_STABILITY`].
    threshold: f64,
}

impl MixedClassifier {
    /// Construct a classifier with the default
    /// [`DEFAULT_MIXED_LOW_BAND_STABILITY`] threshold.
    #[must_use]
    pub fn new() -> Self {
        Self::with_threshold(DEFAULT_MIXED_LOW_BAND_STABILITY)
    }

    /// Construct a classifier with a caller-chosen low-band
    /// stability threshold. Smaller = more conservative (fewer mixed
    /// emissions, more pure-short); larger = more permissive (more
    /// mixed). Non-finite or non-positive values are silently
    /// coerced to the default.
    #[must_use]
    pub fn with_threshold(threshold: f64) -> Self {
        let threshold = if threshold.is_finite() && threshold > 0.0 {
            threshold
        } else {
            DEFAULT_MIXED_LOW_BAND_STABILITY
        };
        Self {
            prev_last: 0.0,
            threshold,
        }
    }

    /// Current low-band stability threshold (max-to-min ratio of the
    /// low-passed subframe energies that the granule must stay under
    /// to be judged mixed-appropriate).
    #[must_use]
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// The last PCM sample of the previous granule (the LP seed for
    /// the next call). Mostly useful for testing.
    #[must_use]
    pub fn prev_last(&self) -> f32 {
        self.prev_last
    }

    /// Reset the classifier's LP-history sample to zero. The
    /// threshold is preserved.
    pub fn reset(&mut self) {
        self.prev_last = 0.0;
    }

    /// Classify the supplied granule PCM. Returns `true` iff the
    /// low-passed signal's per-subframe energy ratio is at or below
    /// the threshold — i.e., the low band is judged stable and a
    /// **mixed** block is appropriate. Returns `false` for
    /// pure-short.
    ///
    /// **Side effect:** updates the LP-history seed to the last
    /// sample of `pcm` so the next call's filter is continuous
    /// across the granule boundary.
    ///
    /// **No upstream gating.** This classifier does not check
    /// whether the granule was flagged transient at all; the caller
    /// (typically [`crate::stream_encoder::Mp3Encoder`]) should only
    /// consult it when the §C.1.5.2 state machine would otherwise
    /// emit a pure-short block. A non-transient granule is on the
    /// long-family path and the mixed/short distinction is moot.
    pub fn classify_mixed(&mut self, pcm: &[f32; SAMPLES_PER_GRANULE]) -> bool {
        let ratio = low_band_stability_ratio(pcm, self.prev_last);
        // Advance the LP-history seed unconditionally.
        self.prev_last = pcm[SAMPLES_PER_GRANULE - 1];
        ratio <= self.threshold
    }
}

impl Default for MixedClassifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A perfectly-silent granule has zero energy in every subframe
    /// of its low-passed signal too. By the `e_max / max(e_min, ε)`
    /// definition the ratio is `0 / ε = 0`, well below threshold —
    /// the classifier returns "mixed appropriate". That is the
    /// degenerate-but-consistent answer: silence has no transient
    /// content of any kind.
    #[test]
    fn silent_granule_is_mixed_appropriate() {
        let pcm = [0.0f32; SAMPLES_PER_GRANULE];
        let r = low_band_stability_ratio(&pcm, 0.0);
        assert_eq!(r, 0.0);
        let mut c = MixedClassifier::new();
        assert!(c.classify_mixed(&pcm));
    }

    /// A steady-state DC granule has equal energies in all three
    /// subframes of the low-passed signal too: the LP filter passes
    /// DC unchanged, every subframe contributes `1·1·N`. The ratio
    /// is exactly 1.0, well under any reasonable threshold → mixed.
    #[test]
    fn dc_granule_low_band_is_stable() {
        let pcm = [1.0f32; SAMPLES_PER_GRANULE];
        // prev_last = 1.0 too so the LP boundary doesn't dent the
        // first sample's energy.
        let r = low_band_stability_ratio(&pcm, 1.0);
        // All subframes equal → ratio = 1.0.
        assert!((r - 1.0).abs() < 1.0e-9, "dc ratio = {r}");
        let mut c = MixedClassifier::new();
        // With prev_last seeded as 1.0 at construction time we'd see
        // 1.0; the default seed is 0.0 so the very first low-passed
        // sample is 0.5 instead of 1.0, dropping subframe-0 energy by
        // a hair. Still well under threshold.
        let _ = c.classify_mixed(&pcm); // step the state once
        let mixed = c.classify_mixed(&pcm); // second call has prev_last = 1.0
        assert!(mixed);
    }

    /// A pure high-frequency burst (alternating ±1.0 sample, the
    /// Nyquist frequency) on top of a tiny low-band DC is fully
    /// attenuated by the one-tap LP filter on the *interior* of the
    /// granule — adjacent samples cancel — so the low-passed signal
    /// is dominated by the DC residual in every subframe and the
    /// ratio stays near 1. The classifier should return
    /// mixed-appropriate. The seed sample at the granule boundary
    /// must be the right one for the boundary cancellation to also
    /// fire (otherwise sample 0 carries a one-sample LP spike from
    /// the seed-edge discontinuity); the same applies to real
    /// signal continuity at granule boundaries — the
    /// `MixedClassifier`'s stateful `prev_last` is exactly the seed
    /// that keeps the filter continuous, so under realistic
    /// stream-encoder use the boundary spike does not arise.
    #[test]
    fn high_frequency_only_attack_is_mixed_appropriate() {
        let mut pcm = [0.0f32; SAMPLES_PER_GRANULE];
        for (i, slot) in pcm.iter_mut().enumerate() {
            let hf = if i % 2 == 0 { 1.0f32 } else { -1.0f32 };
            *slot = 0.01 + hf;
        }
        // prev_last = pcm[last_index_with_polarity_matching_pcm[-1]]:
        // pcm[0] = 1.01, so the immediate predecessor in a continuous
        // ±1 sequence is -0.99 (the *negative* polarity).
        // lp[0] = (1.01 + -0.99) / 2 = 0.01.
        // lp[1] = (-0.99 + 1.01) / 2 = 0.01.
        // All interior samples settle to 0.01 too. The ratio is
        // ≈ 1.0 exactly.
        let r = low_band_stability_ratio(&pcm, 0.01 - 1.0);
        assert!(r < 1.5, "high-frequency-only attack ratio = {r}");

        // Same test via the stateful classifier: we step it once with
        // a "warm-up" granule that lands prev_last on the correct
        // polarity (the granule itself ends at index N-1 where N is
        // even, so pcm[N-1] = 0.01 - 1 = -0.99; that matches the
        // seed we used above).
        let mut c = MixedClassifier::new();
        // Warm-up granule with the same alternating pattern.
        let _ = c.classify_mixed(&pcm);
        // Second call: prev_last is now -0.99 (pcm[N-1]). Classifier
        // sees a stable low band.
        assert!(
            c.classify_mixed(&pcm),
            "high-frequency-only attack judged pure-short (should be mixed)"
        );
    }

    /// At a true cold-start (prev_last = 0) feeding a Nyquist tone
    /// produces a one-sample boundary spike at lp[0], so the ratio
    /// is large and the classifier returns pure-short — the
    /// conservative answer at a discontinuity. This documents the
    /// boundary behaviour explicitly so callers know to seed the
    /// classifier with a warm-up granule (or accept a one-granule
    /// miss at stream start).
    #[test]
    fn cold_start_nyquist_classified_pure_short() {
        let mut pcm = [0.0f32; SAMPLES_PER_GRANULE];
        for (i, slot) in pcm.iter_mut().enumerate() {
            *slot = if i % 2 == 0 { 1.0 } else { -1.0 };
        }
        let mut c = MixedClassifier::new();
        // First call with prev_last == 0.0: boundary spike → ratio
        // is large → classifier says pure-short.
        assert!(!c.classify_mixed(&pcm));
    }

    /// A broadband attack — a sudden jump in low-frequency level on
    /// top of any high-frequency content — lifts the low-passed
    /// signal's late-subframe energy well above the early subframes.
    /// The ratio rises past threshold and the classifier returns
    /// pure-short.
    #[test]
    fn broadband_attack_is_pure_short() {
        // Subframe 0: silent. Subframe 1: silent. Subframe 2: hard
        // DC jump to +0.5 (the LP filter passes this through nearly
        // unchanged; e_2 ≈ 0.5² · 192 = 48). e_0 ≈ e_1 ≈ 0. The
        // ratio is 48 / SILENCE_FLOOR — astronomically high.
        let mut pcm = [0.0f32; SAMPLES_PER_GRANULE];
        for slot in pcm.iter_mut().skip(2 * SAMPLES_PER_SUBFRAME) {
            *slot = 0.5;
        }
        let r = low_band_stability_ratio(&pcm, 0.0);
        assert!(r > DEFAULT_MIXED_LOW_BAND_STABILITY * 10.0, "ratio = {r}");
        let mut c = MixedClassifier::new();
        assert!(
            !c.classify_mixed(&pcm),
            "broadband attack judged mixed (should be pure-short)"
        );
    }

    /// `with_threshold` rejects non-finite / non-positive values and
    /// falls back to the default.
    #[test]
    fn invalid_threshold_falls_back_to_default() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -1.0] {
            let c = MixedClassifier::with_threshold(bad);
            assert_eq!(c.threshold(), DEFAULT_MIXED_LOW_BAND_STABILITY);
        }
    }

    /// `reset` clears the LP-history seed back to zero so the next
    /// granule's first sample is filtered against silence (the same
    /// as the constructor-time initial state).
    #[test]
    fn reset_clears_prev_last() {
        let mut c = MixedClassifier::new();
        let mut pcm = [0.0f32; SAMPLES_PER_GRANULE];
        for (i, slot) in pcm.iter_mut().enumerate() {
            *slot = (i as f32 / SAMPLES_PER_GRANULE as f32) - 0.5;
        }
        let _ = c.classify_mixed(&pcm);
        assert!(c.prev_last() != 0.0);
        c.reset();
        assert_eq!(c.prev_last(), 0.0);
    }

    /// `prev_last` is the very last sample of the granule, so two
    /// successive identical-pcm calls produce identical decisions
    /// only when the granule is shape-stationary; that is the
    /// definitional check on the seed update.
    #[test]
    fn prev_last_tracks_last_sample() {
        let mut c = MixedClassifier::new();
        let mut pcm = [0.0f32; SAMPLES_PER_GRANULE];
        pcm[SAMPLES_PER_GRANULE - 1] = 0.7;
        let _ = c.classify_mixed(&pcm);
        assert!((c.prev_last() - 0.7).abs() < 1.0e-9);
    }

    /// The low-pass filter is unity-DC-gain: an all-DC granule fed
    /// from an all-DC previous granule produces the same all-DC
    /// signal exactly.
    #[test]
    fn low_pass_unity_dc_gain() {
        let pcm = [0.3f32; SAMPLES_PER_GRANULE];
        let lp = low_pass_granule(&pcm, 0.3);
        for v in lp.iter() {
            assert!((v - 0.3).abs() < 1.0e-9);
        }
    }

    /// The low-pass filter nulls the Nyquist frequency: an
    /// alternating ±1 sequence fed from its own continuation
    /// produces all-zero output.
    #[test]
    fn low_pass_nulls_nyquist() {
        let mut pcm = [0.0f32; SAMPLES_PER_GRANULE];
        for (i, slot) in pcm.iter_mut().enumerate() {
            *slot = if i % 2 == 0 { 1.0 } else { -1.0 };
        }
        // prev_last = -1.0 so pcm[0] = 1.0 has the immediate
        // predecessor that cancels it: lp[0] = (1 + -1)/2 = 0.
        let lp = low_pass_granule(&pcm, -1.0);
        for v in lp.iter() {
            assert!(v.abs() < 1.0e-9);
        }
    }
}
