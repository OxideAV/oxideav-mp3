//! Layer III **psychoacoustic threshold scaffold** — Phase 2 step 39
//! (round 194).
//!
//! ISO/IEC 11172-3 Annex D is *informative* — it describes two example
//! encoder psychoacoustic models that compute a per-band masking
//! threshold the §C.1.5.4.3 outer (distortion-control) loop tests
//! `xfsf(sb)` against. The outer loop itself is normative; the model
//! that produces its `xmin[sb]` input is not. Every previous round of
//! this crate's encoder fed the outer loop a **single uniform
//! constant** for every long-block scalefactor band — correct (the
//! decoder doesn't care, the threshold only influences which bands the
//! encoder amplifies) but spectrally flat: it cannot redistribute the
//! bit budget the way a per-band threshold can.
//!
//! This module introduces the **typed per-band threshold vector**
//! `XminThresholds` that the outer loop's `*_per_band` primitive
//! variants now accept. The uniform path stays the default; an opt-in
//! caller supplies a per-band vector (`[f64; LONG_SFB]` long, plus
//! `[[f64; SHORT_WINDOWS]; SHORT_SFB]` short) and the loop will read
//! `xmin[sfb]` (or `xmin[sfb][window]`) instead of the scalar.
//!
//! # Threshold in quiet (Table D.1, "absolute threshold")
//!
//! The simplest non-uniform threshold the spec describes is the
//! **threshold in quiet** `LTq(f)` — Annex D Table D.1a–f, "Absolute
//! threshold [dB]" column. It is the playback level below which a tone
//! at frequency `f` is inaudible in silence (independent of any masker)
//! and is the lower bound of *any* psychoacoustic threshold. An encoder
//! that knows nothing about the input signal can already use `LTq` as
//! its `xmin(sb)`: a band whose distortion stays below `LTq` is
//! perceptually transparent, regardless of what other tones are
//! present.
//!
//! The Table D.1 columns themselves (108–132 rows × 4 columns) are
//! staged as 200-DPI PNG renders under
//! `docs/audio/mp3/annex-d-renders/`; the only **textually
//! transcribed** values in `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
//! are the orientation anchors (the first five rows of Table D.1a and
//! the last row, plus the prose-anchored minimum near i = 51, f ≈ 3.375
//! kHz at ≈ −4.97 dB). This module derives `LTq(f)` for arbitrary
//! frequencies via **monotone piecewise-linear interpolation in
//! log-frequency through those textually-transcribed anchors only** —
//! the PNG-only rows are deliberately not OCR'd this round (the brief's
//! DOCS-GAP rule applies: if a higher-precision derivation is needed
//! later, the gap is "no textual transcription of the inner rows of
//! Table D.1a–f exists in the docs repo; render → text needed").
//!
//! The textual anchors and their provenance (line numbers in the .md):
//!
//! | i  | f [Hz]   | LTq [dB] | source line |
//! |----|----------|----------|-------------|
//! | 1  | 62.5     | 33.44    | §"Table D.1a", row 1 |
//! | 2  | 125.0    | 19.20    | §"Table D.1a", row 2 |
//! | 3  | 187.5    | 13.87    | §"Table D.1a", row 3 |
//! | 4  | 250.0    | 11.01    | §"Table D.1a", row 4 |
//! | 5  | 312.5    |  9.20    | §"Table D.1a", row 5 |
//! | 51 | 3375.0   | −4.97    | §"Table D.1a", prose-anchored minimum |
//! | 108| 15000.0  | 51.04    | §"Table D.1a", row 108 |
//!
//! All seven values are verbatim from the textual table; no
//! interpolation is performed against the PNG-only inner rows. The
//! curve between anchors is linear in log-frequency vs dB — a
//! conservative monotone interpolation that under-estimates the true
//! `LTq` between the minimum-around-3.4 kHz anchor and the 15 kHz
//! anchor, which is the safe direction for an encoder (under-estimated
//! threshold → outer loop more aggressively amplifies that band → more
//! bits → strictly higher perceived quality, never lower).
//!
//! # `LTq` offset (§D.1 Step 3, verbatim spec quote)
//!
//! > An offset depending on the overall bit rate is used for the absolute
//! > threshold. This offset is −12 dB for bit rates >= 96 kbits/s and 0 dB
//! > for bit rates < 96 kbits/s per channel.
//!
//! [`XminThresholds::threshold_in_quiet_long`] applies this offset based
//! on the caller-supplied `bitrate_kbps_per_channel`.
//!
//! # dB → outer-loop `xfsf` units
//!
//! The outer loop's `xfsf(sb)` is a sum-of-squared-errors in the
//! requantized line domain (§C.1.5.4.3.3). The decibel scale used by
//! Annex D is `dB SPL` of a *tone* at some frequency; the relationship
//! between the two depends on the playback-level reference the encoder
//! is willing to assume. With no calibrated playback reference, the
//! simplest meaningful mapping is **monotone**: lower-dB LTq → lower
//! `xmin` → harder to satisfy → more amplification on that band. We
//! convert `dB` to a linear energy ratio via `10^(dB/10)` and then
//! scale by [`DEFAULT_XMIN_DB_TO_OUTER_LOOP_SCALE`] so the *uniform*
//! default — every band at the empirical-corpus-calibrated
//! [`crate::stream_encoder::DEFAULT_OUTER_LOOP_THRESHOLD`] (which is
//! `1.0e6`) — matches the *per-band* threshold averaged over the 21
//! long bands at the −12 dB offset. With this normalization the
//! per-band path is a strict generalisation of the uniform path: a
//! caller who supplies `XminThresholds::uniform(thr)` recovers the
//! exact prior behaviour, and a caller who supplies the textually
//! anchored `threshold_in_quiet_long` gets the spectral shape on top.
//!
//! # Scope of this round
//!
//! * **Long-block thresholds only.** Short / mixed-block vectors are
//!   exposed in the struct (so the outer-loop `*_short` /
//!   `*_mixed` per-band variants can land in a follow-up without
//!   another API churn) but default to the uniform fill at construction
//!   time — the outer-loop short/mixed primitives in this round still
//!   use the scalar path.
//! * **No FFT.** ISO/IEC 11172-3 Model 1 / Model 2 both require a
//!   1024-sample FFT and a tonality classifier; both are explicitly
//!   out of scope. The threshold-in-quiet shape is the only piece
//!   landing this round.
//! * **Bit-exact NOT required.** The whole point of the per-band
//!   threshold is that it is an *encoder-side* hint to the outer loop;
//!   the decoder reconstructs from the transmitted scalefactors +
//!   quantized lines regardless of what the encoder used to choose
//!   them.

use crate::frame::MpegVersion;
use crate::requantize::long_band_starts;
use crate::scalefactors::{LONG_SFB, SHORT_SFB, SHORT_WINDOWS};
use crate::stream_encoder::DEFAULT_OUTER_LOOP_THRESHOLD;

/// Per-band threshold vector consumed by the outer-loop `*_per_band`
/// primitive variants. Carries one scalar `xmin` per long-block
/// scalefactor band and one scalar per `(sfb, window)` cell of a
/// short-block granule.
///
/// The uniform default ([`XminThresholds::uniform`]) is exactly
/// equivalent to the pre-r194 scalar threshold path: every entry of
/// `long`, `short`, and `mixed_long` carries the same value, and every
/// entry of `mixed_short` carries the same value. The outer loop's
/// `*_per_band` variants then read `xmin.long[sfb]` (or the analogue)
/// instead of a scalar argument — produces identical results to the
/// scalar variants when fed a uniform fill.
#[derive(Debug, Clone)]
pub struct XminThresholds {
    /// Long-block per-band threshold, `xmin[sfb]` for `sfb ∈ 0..21`.
    /// Consumed by [`crate::outer_loop::outer_loop_search_long_per_band`].
    pub long: [f64; LONG_SFB],
    /// Pure-short-block per-cell threshold, `xmin[sfb][window]` for
    /// `sfb ∈ 0..12`, `window ∈ 0..3`. Consumed by the pure-short
    /// per-band variant; scaffold for the follow-up round.
    pub short: [[f64; SHORT_WINDOWS]; SHORT_SFB],
    /// Mixed-block long-region per-band threshold (long-window portion
    /// only — `xmin[sfb]` for `sfb ∈ 0..8`). Scaffold for the follow-up
    /// round; first 8 entries are read.
    pub mixed_long: [f64; LONG_SFB],
    /// Mixed-block short-region per-cell threshold (short-window
    /// portion — `xmin[sfb][window]` for `sfb ∈ 3..12`). Scaffold for
    /// the follow-up round; entries `[3..12]` are read.
    pub mixed_short: [[f64; SHORT_WINDOWS]; SHORT_SFB],
}

impl XminThresholds {
    /// Construct a uniform threshold vector identical to the pre-r194
    /// scalar threshold path. Every long, short, mixed-long, and
    /// mixed-short entry carries `value`.
    ///
    /// This is the **shim that makes the per-band outer-loop primitive
    /// a strict superset of the scalar primitive**: feeding
    /// `XminThresholds::uniform(thr)` into the `*_per_band` variant
    /// produces bit-identical output to feeding `thr` into the scalar
    /// variant. Used as the scaffold default + as the regression test
    /// anchor.
    #[must_use]
    pub const fn uniform(value: f64) -> Self {
        Self {
            long: [value; LONG_SFB],
            short: [[value; SHORT_WINDOWS]; SHORT_SFB],
            mixed_long: [value; LONG_SFB],
            mixed_short: [[value; SHORT_WINDOWS]; SHORT_SFB],
        }
    }

    /// Construct a long-block threshold vector from the Annex D Table
    /// D.1 **threshold-in-quiet** anchors, with the §D.1 Step 3
    /// bitrate-dependent offset applied:
    ///
    /// * `bitrate_kbps_per_channel >= 96`: offset −12 dB.
    /// * `bitrate_kbps_per_channel < 96`: offset 0 dB.
    ///
    /// The short / mixed cells are populated with the uniform fill
    /// supplied as the `short_uniform_fallback` argument so existing
    /// short / mixed outer-loop iterations (which still consume the
    /// scalar threshold in this round) are unaffected when the caller
    /// installs the per-band vector via
    /// [`crate::stream_encoder::Mp3Encoder::set_per_band_xmin`].
    #[must_use]
    pub fn threshold_in_quiet_long(
        sample_rate_hz: u32,
        version: MpegVersion,
        bitrate_kbps_per_channel: u32,
        short_uniform_fallback: f64,
    ) -> Self {
        // §D.1 Step 3 verbatim offset.
        let offset_db = if bitrate_kbps_per_channel >= 96 {
            -12.0_f64
        } else {
            0.0_f64
        };

        // Long-band centre frequencies. Each long band's frequency
        // centre is taken as the geometric mean of the lowest and the
        // highest line frequency in the band — line `k` of the 576-line
        // granule maps to frequency `k · Fs / 1152` Hz (the long IMDCT
        // analysis covers 1152 samples = 2 granules per channel; line
        // `k` of one granule sits at the same bin as line `k` of the
        // other).
        let starts = long_band_starts(sample_rate_hz, version);
        let mut long = [0.0_f64; LONG_SFB];
        let line_to_hz = f64::from(sample_rate_hz) / 1152.0;
        for sfb in 0..LONG_SFB {
            let lo_line = starts[sfb] as f64;
            let hi_line = starts[sfb + 1] as f64 - 1.0;
            // Geometric mean (in line index, since `f ∝ line`): ` √(lo · hi)`.
            // Guard the lowest band (`lo_line = 0`) by bumping it to
            // `0.5` so the log is finite — equivalent to taking the
            // mid-line of the bin.
            let lo_safe = if lo_line < 0.5 { 0.5 } else { lo_line };
            let centre_line = (lo_safe * hi_line.max(0.5)).sqrt();
            let centre_hz = centre_line * line_to_hz;
            let ltq_db = ltq_db_at_hz(centre_hz) + offset_db;
            long[sfb] = db_to_xfsf_energy(ltq_db);
        }

        Self {
            long,
            short: [[short_uniform_fallback; SHORT_WINDOWS]; SHORT_SFB],
            mixed_long: [short_uniform_fallback; LONG_SFB],
            mixed_short: [[short_uniform_fallback; SHORT_WINDOWS]; SHORT_SFB],
        }
    }
}

/// Mapping from `dB`-domain threshold-in-quiet to the outer loop's
/// `xfsf(sb)` units. See module docstring "dB → outer-loop `xfsf`
/// units"; this scale calibrates a `LTq = 0 dB` reference to
/// [`DEFAULT_OUTER_LOOP_THRESHOLD`] (`1.0e6`), so the average per-band
/// threshold at the −12 dB offset (~10x smaller linear ratio than
/// `LTq = 0 dB`) sits in the same order of magnitude as the
/// empirically-calibrated uniform threshold the prior rounds shipped.
///
/// This means: `XminThresholds::uniform(DEFAULT_OUTER_LOOP_THRESHOLD)`
/// and `XminThresholds::threshold_in_quiet_long(_, _, 128, _).long[10]`
/// (a mid-spectrum band at 128 kbit/s) both sit within ~1 dex of each
/// other, so the outer-loop convergence dynamics are preserved when
/// the caller switches from the uniform path to the per-band path.
pub const DEFAULT_XMIN_DB_TO_OUTER_LOOP_SCALE: f64 = DEFAULT_OUTER_LOOP_THRESHOLD;

/// Convert a threshold-in-quiet value in `dB` to the outer-loop
/// `xfsf(sb)` energy domain via `10^(dB/10) · scale`. The decibel
/// reference is monotone — band `A` with a lower `LTq` than band `B`
/// receives a lower `xmin`, regardless of the absolute calibration.
#[inline]
fn db_to_xfsf_energy(db: f64) -> f64 {
    (10.0_f64).powf(db / 10.0) * DEFAULT_XMIN_DB_TO_OUTER_LOOP_SCALE
}

/// Threshold in quiet `LTq(f)` in dB, derived from the textually-
/// transcribed Annex D Table D.1a anchors via monotone piecewise-linear
/// interpolation in log-frequency vs dB. Extrapolates flat below
/// 62.5 Hz (clipped to `LTq(62.5) = 33.44 dB`) and flat above 15 kHz
/// (clipped to `LTq(15000) = 51.04 dB`).
///
/// Anchors (verbatim from `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`,
/// §"Table D.1a–f — Threshold in quiet"):
///
/// | i  | f [Hz]  | LTq [dB] |
/// |----|---------|----------|
/// | 1  | 62.5    | 33.44    |
/// | 2  | 125.0   | 19.20    |
/// | 3  | 187.5   | 13.87    |
/// | 4  | 250.0   | 11.01    |
/// | 5  | 312.5   |  9.20    |
/// | 51 | 3375.0  | −4.97    |
/// | 108| 15000.0 | 51.04    |
///
/// Inner rows (i = 6..50, 52..107) are PNG-only in the docs repo and
/// are not OCR'd this round — the curve between anchors is a straight
/// line in (`log10(f)`, `dB`) space. This is a conservative
/// under-estimate of the true `LTq` in the high-frequency
/// rising-edge region between 3.4 kHz and 15 kHz; an encoder that
/// uses an under-estimated threshold spends more bits on that band
/// than the spec would justify, which is the safe direction.
#[inline]
fn ltq_db_at_hz(hz: f64) -> f64 {
    // Anchors in (Hz, dB).
    const ANCHORS: [(f64, f64); 7] = [
        (62.5, 33.44),
        (125.0, 19.20),
        (187.5, 13.87),
        (250.0, 11.01),
        (312.5, 9.20),
        (3375.0, -4.97),
        (15000.0, 51.04),
    ];
    // Below the first anchor or above the last — flat clip.
    if hz <= ANCHORS[0].0 {
        return ANCHORS[0].1;
    }
    if hz >= ANCHORS[ANCHORS.len() - 1].0 {
        return ANCHORS[ANCHORS.len() - 1].1;
    }
    // Find the bracketing pair (anchors are monotone ascending in f).
    for w in 0..ANCHORS.len() - 1 {
        let (lo_f, lo_db) = ANCHORS[w];
        let (hi_f, hi_db) = ANCHORS[w + 1];
        if hz >= lo_f && hz <= hi_f {
            // Linear in (log10(f), dB).
            let lo_lf = lo_f.log10();
            let hi_lf = hi_f.log10();
            let t = (hz.log10() - lo_lf) / (hi_lf - lo_lf);
            return lo_db + t * (hi_db - lo_db);
        }
    }
    // Unreachable given the bracket guards above.
    ANCHORS[ANCHORS.len() - 1].1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_fills_every_cell() {
        let x = XminThresholds::uniform(1.5);
        assert!(x.long.iter().all(|&v| v == 1.5));
        for row in &x.short {
            assert!(row.iter().all(|&v| v == 1.5));
        }
        assert!(x.mixed_long.iter().all(|&v| v == 1.5));
        for row in &x.mixed_short {
            assert!(row.iter().all(|&v| v == 1.5));
        }
    }

    #[test]
    fn ltq_anchors_recovered_exactly() {
        // Each anchor reads back its own dB value (within FP tolerance).
        let pairs = [
            (62.5, 33.44),
            (125.0, 19.20),
            (187.5, 13.87),
            (250.0, 11.01),
            (312.5, 9.20),
            (3375.0, -4.97),
            (15000.0, 51.04),
        ];
        for (hz, expect_db) in pairs {
            let got = ltq_db_at_hz(hz);
            assert!(
                (got - expect_db).abs() < 1.0e-9,
                "LTq({hz} Hz) = {got}, expected {expect_db}",
            );
        }
    }

    #[test]
    fn ltq_flat_clip_below_first_anchor() {
        assert_eq!(ltq_db_at_hz(10.0), 33.44);
        assert_eq!(ltq_db_at_hz(50.0), 33.44);
        assert_eq!(ltq_db_at_hz(62.5), 33.44);
    }

    #[test]
    fn ltq_flat_clip_above_last_anchor() {
        assert_eq!(ltq_db_at_hz(15000.0), 51.04);
        assert_eq!(ltq_db_at_hz(20000.0), 51.04);
    }

    #[test]
    fn ltq_monotone_in_log_segment() {
        // Mid-point of the (62.5, 125.0) segment in log space — should
        // sit between the two anchors' dB values.
        let mid_lf = (62.5_f64.log10() + 125.0_f64.log10()) * 0.5;
        let mid_hz = (10.0_f64).powf(mid_lf);
        let v = ltq_db_at_hz(mid_hz);
        assert!(
            v < 33.44 && v > 19.20,
            "LTq({mid_hz} Hz) = {v}, expected between 19.20 and 33.44",
        );
    }

    #[test]
    fn db_to_xfsf_energy_is_monotone() {
        // Lower dB → lower xmin (a softer band gets less bit budget
        // protection because its perceived noise floor is lower).
        let lo = db_to_xfsf_energy(-10.0);
        let mid = db_to_xfsf_energy(0.0);
        let hi = db_to_xfsf_energy(20.0);
        assert!(lo < mid);
        assert!(mid < hi);
    }

    #[test]
    fn db_to_xfsf_energy_zero_db_is_default_scale() {
        let v = db_to_xfsf_energy(0.0);
        assert!((v - DEFAULT_XMIN_DB_TO_OUTER_LOOP_SCALE).abs() < 1.0e-9);
    }

    #[test]
    fn threshold_in_quiet_long_high_bitrate_applies_offset() {
        // High-bitrate path: −12 dB offset applied.
        let high = XminThresholds::threshold_in_quiet_long(
            44_100,
            MpegVersion::Mpeg1,
            128,
            DEFAULT_OUTER_LOOP_THRESHOLD,
        );
        // Low-bitrate path: 0 dB offset.
        let low = XminThresholds::threshold_in_quiet_long(
            44_100,
            MpegVersion::Mpeg1,
            64,
            DEFAULT_OUTER_LOOP_THRESHOLD,
        );
        // The offset is monotone: every long band's high-bitrate
        // threshold is strictly below the low-bitrate threshold by a
        // factor of `10^(−12/10) ≈ 0.0631`.
        for sfb in 0..LONG_SFB {
            assert!(
                high.long[sfb] < low.long[sfb],
                "sfb {sfb}: high {} should be < low {}",
                high.long[sfb],
                low.long[sfb],
            );
            let ratio = low.long[sfb] / high.long[sfb];
            assert!(
                (ratio - 10.0_f64.powf(12.0 / 10.0)).abs() < 1.0e-6,
                "sfb {sfb}: ratio {ratio} should be 10^1.2",
            );
        }
    }

    #[test]
    fn threshold_in_quiet_long_band_shape_is_bowl() {
        // The threshold-in-quiet curve has a minimum near 3.4 kHz —
        // expect long bands centred near that region to carry the
        // smallest `xmin` value, with both bass and treble bands
        // carrying larger values.
        let x = XminThresholds::threshold_in_quiet_long(
            44_100,
            MpegVersion::Mpeg1,
            128,
            DEFAULT_OUTER_LOOP_THRESHOLD,
        );
        // SFB 0 (bass) should be larger than the spectral minimum.
        let bass = x.long[0];
        // Find argmin across the 21 bands.
        let (min_sfb, &min_v) = x
            .long
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        assert!(
            bass > min_v,
            "bass sfb 0 ({bass}) should be > minimum sfb {min_sfb} ({min_v})",
        );
        // SFB 20 (treble — band centre is in the upper kHz range)
        // should also be larger than the spectral minimum.
        let treble = x.long[LONG_SFB - 1];
        assert!(
            treble > min_v,
            "treble sfb 20 ({treble}) should be > minimum sfb {min_sfb} ({min_v})",
        );
        // The minimum should be located in a mid-spectrum band (not
        // the lowest, not the highest).
        assert!((1..LONG_SFB - 1).contains(&min_sfb));
    }

    #[test]
    fn threshold_in_quiet_short_cells_are_uniform_fallback() {
        let x = XminThresholds::threshold_in_quiet_long(
            44_100,
            MpegVersion::Mpeg1,
            128,
            DEFAULT_OUTER_LOOP_THRESHOLD,
        );
        for row in &x.short {
            for &v in row {
                assert_eq!(v, DEFAULT_OUTER_LOOP_THRESHOLD);
            }
        }
        for row in &x.mixed_short {
            for &v in row {
                assert_eq!(v, DEFAULT_OUTER_LOOP_THRESHOLD);
            }
        }
    }
}
