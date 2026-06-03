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
use crate::requantize::{long_band_starts, short_band_starts};
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
    /// `sfb ∈ 0..12`, `window ∈ 0..3`. Consumed by
    /// [`crate::outer_loop::outer_loop_search_short_per_band`].
    pub short: [[f64; SHORT_WINDOWS]; SHORT_SFB],
    /// Mixed-block long-region per-band threshold (long-window portion
    /// only — `xmin[sfb]` for `sfb ∈ 0..8`). Consumed by
    /// [`crate::outer_loop::outer_loop_search_mixed_per_band`]; first 8
    /// entries are read.
    pub mixed_long: [f64; LONG_SFB],
    /// Mixed-block short-region per-cell threshold (short-window
    /// portion — `xmin[sfb][window]` for `sfb ∈ 3..12`). Consumed by
    /// [`crate::outer_loop::outer_loop_search_mixed_per_band`]; entries
    /// `[3..12]` are read.
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

    /// Construct a pure-short-block per-cell threshold matrix from the
    /// Annex D Table D.1 **threshold-in-quiet** anchors, plus a long-band
    /// vector for any caller that mixes long and short granules through
    /// the same encoder (the long path is identical to
    /// [`Self::threshold_in_quiet_long`]). The §D.1 Step 3 offset
    /// (`−12 dB` for `bitrate_kbps_per_channel >= 96`, `0 dB` otherwise)
    /// applies to both shapes — same bitrate-dependent transparency
    /// reference for both block types.
    ///
    /// Short-band centre-frequency mapping: a pure-short granule splits
    /// the 576-line spectrum into 3 windows × 192 per-window lines.
    /// Per-window line `k` represents the bin at `f = k · Fs / 384` Hz
    /// (the per-window short MDCT is 192-sample / 384-point window;
    /// Nyquist `Fs/2` sits at per-window line `k = 96`, i.e. half the
    /// per-window line count). Each short scalefactor band `sfb` covers
    /// per-window lines `[starts[sfb], starts[sfb+1])`; the band's
    /// centre frequency is the geometric mean of the lowest and highest
    /// per-window line in the band, converted through that `Fs/384`
    /// per-line factor. All three windows of the band share the same
    /// per-cell threshold — the threshold-in-quiet is a *frequency*
    /// property, not a temporal one (Annex D Table D.1 is sampled
    /// against `f` only). The temporal structure of the short window
    /// (which window peaks where) is exactly what `subblock_gain` and
    /// the per-window MDCT already capture; the threshold-in-quiet
    /// per-cell value should mirror the long-block bowl shape and apply
    /// uniformly across the three windows of each band.
    ///
    /// The mixed-block long region (`mixed_long[0..=7]`) and mixed-block
    /// short region (`mixed_short[3..=11][..]`) are populated from the
    /// same anchors using the appropriate frequency mapping for each
    /// sub-region. The mixed-block long region uses the same long-band
    /// centre-frequency formula as `Self::threshold_in_quiet_long` over
    /// the first 8 long bands (the only ones a mixed block carries
    /// long-coded — see [`crate::outer_loop::MIXED_LAST_LONG_SFB`]); the
    /// mixed-block short region uses the same per-window short-band
    /// formula as the pure-short path over `sfb ∈ [3, 12)` (mixed blocks
    /// absorb short `sfb 0..=2` into the long-window portion).
    #[must_use]
    pub fn threshold_in_quiet(
        sample_rate_hz: u32,
        version: MpegVersion,
        bitrate_kbps_per_channel: u32,
    ) -> Self {
        // §D.1 Step 3 verbatim offset.
        let offset_db = if bitrate_kbps_per_channel >= 96 {
            -12.0_f64
        } else {
            0.0_f64
        };
        Self::threshold_in_quiet_with_offset_db(sample_rate_hz, version, offset_db)
    }

    /// As [`Self::threshold_in_quiet`], but the caller supplies the dB
    /// **offset** that would normally be derived from
    /// `bitrate_kbps_per_channel` per ISO/IEC 11172-3:1993 §D.1 Step 3.
    ///
    /// The spec's Step 3 procedure mandates exactly two offsets — `−12
    /// dB` for `bitrate_kbps_per_channel >= 96`, `0 dB` otherwise — and
    /// every caller producing a *spec-conformant* outer-loop threshold
    /// must continue to use [`Self::threshold_in_quiet`]. This
    /// `_with_offset_db` variant exists for callers that need to
    /// **tune** the transparency target (e.g. quality-knob front-ends,
    /// regression-test sweeps over the offset, VBR encoders that pick a
    /// running offset from a recent-bitrate accumulator). The dB scalar
    /// is applied **uniformly across every band, long / short / mixed
    /// alike**, on top of the per-frequency `LTq` shape — i.e. the
    /// curve's bowl is preserved and the whole curve is translated up
    /// or down by `offset_db` dB.
    ///
    /// Conventions for the caller:
    ///
    /// * `offset_db = -12.0` is the spec-default high-bitrate path
    ///   (matches `threshold_in_quiet(_, _, 96)` to within FP).
    /// * `offset_db = 0.0` is the spec-default low-bitrate path
    ///   (matches `threshold_in_quiet(_, _, 64)` to within FP).
    /// * `offset_db < -12.0` tightens the threshold (more bits per
    ///   band, higher SNR vs LTq).
    /// * `offset_db > 0.0` loosens it (fewer bits, lower SNR vs LTq).
    /// * No FP-domain clamping is applied — extreme values (`±200 dB`)
    ///   will produce arithmetic over/underflows in `db_to_xfsf_energy`
    ///   that are the caller's responsibility to avoid; the encoder
    ///   itself doesn't read the absolute magnitude, only the per-band
    ///   ordering.
    #[must_use]
    pub fn threshold_in_quiet_with_offset_db(
        sample_rate_hz: u32,
        version: MpegVersion,
        offset_db: f64,
    ) -> Self {
        // Long bands — same derivation as `threshold_in_quiet_long`.
        let long_starts = long_band_starts(sample_rate_hz, version);
        let mut long = [0.0_f64; LONG_SFB];
        let long_line_to_hz = f64::from(sample_rate_hz) / 1152.0;
        for sfb in 0..LONG_SFB {
            let lo_line = long_starts[sfb] as f64;
            let hi_line = long_starts[sfb + 1] as f64 - 1.0;
            let lo_safe = if lo_line < 0.5 { 0.5 } else { lo_line };
            let centre_line = (lo_safe * hi_line.max(0.5)).sqrt();
            let centre_hz = centre_line * long_line_to_hz;
            let ltq_db = ltq_db_at_hz(centre_hz) + offset_db;
            long[sfb] = db_to_xfsf_energy(ltq_db);
        }

        // Pure-short bands — per-window line `k` → `f = k · Fs / 384` Hz.
        // All three windows of each band share the same per-cell xmin
        // (Annex D Table D.1 is purely a function of frequency; the
        // temporal placement of the three windows is captured by the
        // §2.4.3.4.7.1 `subblock_gain[w]` reconstruction term, not by
        // `LTq(f)`).
        let short_starts = short_band_starts(sample_rate_hz, version);
        let short_line_to_hz = f64::from(sample_rate_hz) / 384.0;
        let mut short = [[0.0_f64; SHORT_WINDOWS]; SHORT_SFB];
        for sfb in 0..SHORT_SFB {
            let lo_line = short_starts[sfb] as f64;
            let hi_line = short_starts[sfb + 1] as f64 - 1.0;
            let lo_safe = if lo_line < 0.5 { 0.5 } else { lo_line };
            let centre_line = (lo_safe * hi_line.max(0.5)).sqrt();
            let centre_hz = centre_line * short_line_to_hz;
            let ltq_db = ltq_db_at_hz(centre_hz) + offset_db;
            let xmin = db_to_xfsf_energy(ltq_db);
            for cell in short[sfb].iter_mut() {
                *cell = xmin;
            }
        }

        // Mixed-block long region: same long-band derivation, but only
        // the first 8 long bands are carried by a mixed granule. Entries
        // 8..21 are populated with the same long-block values (so any
        // out-of-range read returns a sensible value rather than zero —
        // not consumed by the mixed primitive, but kept consistent for
        // simple inspection).
        let mut mixed_long = long;
        for entry in mixed_long.iter_mut().take(8) {
            // Already filled from the `long` clone — no-op; kept as a
            // comment anchor that the first 8 entries are the
            // mixed-block long-region thresholds the outer loop will
            // read once the per-band mixed primitive lands.
            let _ = entry;
        }

        // Mixed-block short region: same per-window short-band
        // derivation, restricted to `sfb ∈ [3, 12)`. Entries `[0, 3)`
        // are populated with the same per-band values (so out-of-range
        // reads return a sensible value rather than zero); the mixed
        // primitive will only read `[3, 12)`.
        let mixed_short = short;

        Self {
            long,
            short,
            mixed_long,
            mixed_short,
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

// =====================================================================
// Annex D Model 1 — §D.1 Step 6 masking-function `vf` and §D.1 Step 7
// global-threshold summation (Phase 2 step 44 / r219).
//
// Spec context (clause D.1, ISO/IEC 11172-3:1993, informative annex):
//
//   The individual masking threshold of a tonal/non-tonal masker is
//       LT_tm[z(j),z(i)] = X_tm[z(j)] + av_tm[z(j)] + vf[z(j),z(i)]   dB
//       LT_nm[z(j),z(i)] = X_nm[z(j)] + av_nm[z(j)] + vf[z(j),z(i)]   dB
//
//   Masking index av (verbatim, clause D.1 Step 6):
//       tonal     : av_tm = -1,525 - 0,275 * z(j) - 4,5   dB
//       non-tonal : av_nm = -1,525 - 0,175 * z(j) - 0,5   dB
//
//   Masking function vf (same for tonal and non-tonal; dz =
//   z(i) - z(j) is the Bark distance from masker j to line i; X is the
//   SPL of the masker in dB):
//       vf = 17 * (dz + 1) - (0,4 * X[z(j)] + 6)   dB     for -3 <= dz < -1
//       vf = (0,4 * X[z(j)] + 6) * dz              dB     for -1 <= dz <  0
//       vf = -17 * dz                              dB     for  0 <= dz <  1
//       vf = -(dz - 1) * (17 - 0,15 * X[z(j)]) - 17 dB    for  1 <= dz <  8
//
//   Outside `-3 <= dz < 8` the masker is ignored (LT set to -inf dB).
//
//   Global masking threshold (clause D.1 Step 7), summing the powers
//   of the m tonal and n non-tonal individual thresholds with the
//   threshold in quiet LTq:
//       LTg(i) = 10 * log10( 10^(LTq(i)/10)
//                          + Sum 10^(LT_tm[z(j),z(i)]/10)
//                          + Sum 10^(LT_nm[z(j),z(i)]/10) )   dB
//
// These primitives are pure functions of the (masker SPL, masker Bark
// position, target Bark position) tuple — they do not consult the
// PNG-only inner rows of Tables D.1 / D.2 / D.3. Steps 1-5 of Model 1
// (1024-sample FFT, SPL conversion, tonality classifier, decimation /
// reorganisation, masker selection) remain blocked on the PNG render
// transcription DOCS-GAP and are not landed this round; this round
// supplies the masker -> masking-threshold half of the model that
// the future Steps 1-5 will eventually drive.
//
// Decimal-comma convention: the spec uses European decimal notation
// (`0,617` = 0.617). Constants below are reproduced with the period
// equivalents (`0.617`) consistent with idiomatic Rust f64 literals;
// no value has been rounded or altered from the spec.
// =====================================================================

/// Classification of a Model 1 masker per ISO/IEC 11172-3:1993 §D.1
/// Step 4 (tonal vs non-tonal). The two carry different masking-index
/// constants — a tonal masker has a deeper masking floor than a
/// non-tonal masker at the same SPL and Bark distance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskerKind {
    /// Tonal masker (Step 4 selection: local maxima of the SPL
    /// spectrum surrounded by clearly lower neighbours). Carries
    /// `av_tm = -1.525 - 0.275 * z(j) - 4.5` dB.
    Tonal,
    /// Non-tonal masker (Step 4: per-critical-band energy sum of all
    /// non-tonal FFT lines, lumped to a single representative SPL +
    /// Bark position). Carries `av_nm = -1.525 - 0.175 * z(j) - 0.5` dB.
    NonTonal,
}

/// A single Model 1 masker carrying its SPL (`X[z(j)]` in dB) and its
/// Bark position (`z(j)`). Produced by §D.1 Step 4 (tonal /
/// non-tonal selection) and consumed by §D.1 Step 6 (individual
/// masking-threshold computation) and §D.1 Step 7 (global-threshold
/// summation).
///
/// This is a pure data carrier — the primitive functions on this
/// module read `spl_db` and `z_bark` directly. The intermediate
/// "tonal / non-tonal" Bark-coordinate transformation done by Step 4
/// is the caller's responsibility (Steps 1-5 of Model 1 are not
/// implemented this round; see the module-level DOCS-GAP note on the
/// PNG-only D.1 / D.2 tables).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Masker {
    /// Tonal / non-tonal classification per §D.1 Step 4.
    pub kind: MaskerKind,
    /// Masker Bark position `z(j)`, in Bark units (0..~26 across the
    /// 32 / 44.1 / 48 kHz audio band).
    pub z_bark: f64,
    /// Masker SPL `X[z(j)]`, in dB.
    pub spl_db: f64,
}

/// Lower bound of the Bark-distance window in which a masker
/// contributes a non-`-inf` individual masking threshold (verbatim
/// §D.1 Step 6: the `vf` piecewise function is defined for
/// `-3 <= dz < 8`).
pub const MASKING_FUNCTION_DZ_LO: f64 = -3.0;

/// Upper bound (exclusive) of the Bark-distance window in which a
/// masker contributes a non-`-inf` individual masking threshold
/// (verbatim §D.1 Step 6).
pub const MASKING_FUNCTION_DZ_HI: f64 = 8.0;

/// §D.1 Step 6 masking index for a **tonal** masker at Bark
/// position `z_j`:
///
/// ```text
/// av_tm = -1.525 - 0.275 * z(j) - 4.5   dB
/// ```
///
/// (Verbatim spec equation. The constant `-4.5` is the tonal /
/// non-tonal differential — the tonal masker's individual masking
/// threshold sits ~4 dB lower for the same SPL + Bark position
/// because a tone is a more efficient masker than noise.)
#[inline]
#[must_use]
pub fn masking_index_tonal(z_j_bark: f64) -> f64 {
    -1.525 - 0.275 * z_j_bark - 4.5
}

/// §D.1 Step 6 masking index for a **non-tonal** masker at Bark
/// position `z_j`:
///
/// ```text
/// av_nm = -1.525 - 0.175 * z(j) - 0.5   dB
/// ```
///
/// (Verbatim spec equation. The slope `-0.175` is lower than the
/// tonal `-0.275`, so the difference between tonal and non-tonal
/// masking indices widens with Bark — a tonal masker high in the
/// spectrum gets a relatively deeper threshold floor than the same
/// SPL non-tonal masker does.)
#[inline]
#[must_use]
pub fn masking_index_non_tonal(z_j_bark: f64) -> f64 {
    -1.525 - 0.175 * z_j_bark - 0.5
}

/// §D.1 Step 6 masking function `vf(dz, X)` (verbatim spec text).
/// `dz = z(i) - z(j)` is the Bark distance from the masker `j` to
/// the target line `i`; `x_db` is the SPL `X[z(j)]` of the masker
/// in dB.
///
/// ```text
/// vf =  17 * (dz + 1) - (0.4 * X + 6)     dB   for -3 <= dz < -1
/// vf =  (0.4 * X + 6) * dz                dB   for -1 <= dz <  0
/// vf = -17 * dz                           dB   for  0 <= dz <  1
/// vf = -(dz - 1) * (17 - 0.15 * X) - 17   dB   for  1 <= dz <  8
/// ```
///
/// Outside `-3 <= dz < 8` the masker is ignored — this function
/// returns `None` (the caller treats the masker as `LT = -inf dB`,
/// i.e. it contributes nothing to the global threshold sum).
///
/// At the boundary `dz = 0` the second and third branches agree
/// (both produce `0` dB), so the line co-located with the masker
/// itself returns the unattenuated masking-index + SPL.
#[inline]
#[must_use]
pub fn masking_function_vf(dz_bark: f64, x_db: f64) -> Option<f64> {
    // Out-of-range guard. The spec uses half-open `[-3, 8)`; preserve
    // that exactly so `dz = 8.0` produces `None`.
    if !(MASKING_FUNCTION_DZ_LO..MASKING_FUNCTION_DZ_HI).contains(&dz_bark) {
        return None;
    }
    let vf = if dz_bark < -1.0 {
        17.0 * (dz_bark + 1.0) - (0.4 * x_db + 6.0)
    } else if dz_bark < 0.0 {
        (0.4 * x_db + 6.0) * dz_bark
    } else if dz_bark < 1.0 {
        -17.0 * dz_bark
    } else {
        -(dz_bark - 1.0) * (17.0 - 0.15 * x_db) - 17.0
    };
    Some(vf)
}

/// §D.1 Step 6 individual masking threshold `LT` (dB) for a single
/// masker at the target Bark line `z(i)`. Combines the masking-index
/// `av` (tonal or non-tonal per `masker.kind`) and the masking
/// function `vf` per the verbatim spec equation:
///
/// ```text
/// LT_tm[z(j), z(i)] = X_tm[z(j)] + av_tm[z(j)] + vf[z(j), z(i)]   dB
/// LT_nm[z(j), z(i)] = X_nm[z(j)] + av_nm[z(j)] + vf[z(j), z(i)]   dB
/// ```
///
/// Returns `None` when the Bark distance `dz = z(i) - z(j)` is
/// outside `[-3, 8)` — the masker contributes nothing to the global
/// threshold at this line.
#[inline]
#[must_use]
pub fn individual_masking_threshold_db(masker: &Masker, z_i_bark: f64) -> Option<f64> {
    let dz = z_i_bark - masker.z_bark;
    let vf = masking_function_vf(dz, masker.spl_db)?;
    let av = match masker.kind {
        MaskerKind::Tonal => masking_index_tonal(masker.z_bark),
        MaskerKind::NonTonal => masking_index_non_tonal(masker.z_bark),
    };
    Some(masker.spl_db + av + vf)
}

/// §D.1 Step 7 global masking threshold `LTg(i)` in dB at the target
/// Bark line `z(i)`, summing the energy contributions of every
/// in-range masker with the threshold-in-quiet `LTq(i)`:
///
/// ```text
/// LTg(i) = 10 * log10( 10^(LTq(i) / 10)
///                    + Sum 10^(LT_tm[z(j), z(i)] / 10)
///                    + Sum 10^(LT_nm[z(j), z(i)] / 10) )   dB
/// ```
///
/// (Verbatim spec equation; tonal and non-tonal contributions enter
/// the sum identically — the per-classification difference lives in
/// the masking-index `av` already folded into each
/// `individual_masking_threshold_db` term.)
///
/// Maskers outside the `[-3, 8)` Bark window (per
/// [`masking_function_vf`]) contribute nothing — they are dropped
/// from the sum, equivalent to `10^(-inf / 10) = 0`. The spec note
/// "for a given i the range of j may be reduced to maskers within
/// -8..+3 Bark of i" is a *symmetric* optimisation across the
/// masker / line pair (a masker at `z(j)` only masks lines in
/// `[z(j) - 3, z(j) + 8)` per `vf`, equivalently a line at `z(i)`
/// is only masked by maskers in `(z(i) - 8, z(i) + 3]`); this
/// function preserves the spec's `vf`-based range exactly without
/// the optional Step 7 short-circuit.
///
/// `ltq_db` is the threshold-in-quiet at `z(i)` — derived externally
/// from [`ltq_db_at_hz`] after converting `z(i)` to a frequency via
/// the Bark / Hz mapping table the caller supplies.
#[must_use]
pub fn global_masking_threshold_db(maskers: &[Masker], z_i_bark: f64, ltq_db: f64) -> f64 {
    // Threshold in quiet contributes 10^(LTq / 10) to the energy sum.
    let mut energy_sum = (10.0_f64).powf(ltq_db / 10.0);
    for masker in maskers {
        if let Some(lt_db) = individual_masking_threshold_db(masker, z_i_bark) {
            energy_sum += (10.0_f64).powf(lt_db / 10.0);
        }
    }
    10.0 * energy_sum.log10()
}

// =====================================================================
// Annex D Model 1 — Table D.2a–f critical-band boundaries
// (Phase 2 step 45 / r224).
//
// Spec context (clause D.1, ISO/IEC 11172-3:1993, informative annex):
//
//   "Step 4: finding tonal/non-tonal components" partitions the FFT
//   spectrum into critical bands defined by Tables D.2a-f. Each row
//   gives a critical-band boundary as (band number, top FFT-line
//   index into Table D.1, top frequency in Hz, top Bark coordinate).
//   A band `k` spans the FFT lines from the previous band's top
//   index + 1 through the current row's `index_fcb` inclusive. The
//   bottom of band 0 is implicitly FFT line 1.
//
// Six tables cover the (Layer, Fs) Cartesian product:
//   D.2a — Layer I, Fs = 32 kHz, 24 bands (no 0..23)
//   D.2b — Layer I, Fs = 44.1 kHz, 25 bands (no 0..24)
//   D.2c — Layer I, Fs = 48 kHz, 26 bands (no 0..25)
//   D.2d — Layer II, Fs = 32 kHz, 25 bands (no 0..24)
//   D.2e — Layer II, Fs = 44.1 kHz, 27 bands (no 0..26)
//   D.2f — Layer II, Fs = 48 kHz, 27 bands (no 0..26)
//
// Verbatim transcription from
// `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md` §"Table
// D.2a-f - Critical band boundaries" (lines 125..314). The docs
// file marks band 17 of D.2e Bark coordinate as `16,11[illegible]`
// (clipped final digit in the PDF render); this module preserves
// that uncertainty by recording the legible prefix as `16.11` and
// documenting the illegibility through the
// `D2E_BAND_17_BARK_IS_ILLEGIBLE` constant (the doc's prose
// estimate of `16,116` is explicitly NOT adopted as fact — the
// implementation reads the legible-only value).
//
// Decimal-comma convention: the spec uses European decimal notation
// (`0,617` = 0.617; `15 000,000` Hz = 15000.0 Hz). Constants below
// are reproduced with the period equivalents and the thin-space
// thousands separator stripped, consistent with idiomatic Rust f64
// literals; no value has been rounded or altered from the spec.
//
// This module surfaces the tables as `&[CriticalBandBoundary]` slices
// dispatched by [`critical_band_boundaries`] on (`Layer`, `Fs`); the
// future Model 1 §D.1 Step 4 (masker selection) will iterate the
// returned slice and place each tonal/non-tonal masker at the band's
// `z_bark` coordinate before feeding it to the already-landed
// `Masker` carrier consumed by [`global_masking_threshold_db`].
// =====================================================================

/// A single row of Annex D Table D.2 (critical-band boundary). The
/// row's fields are the top end of the band: the highest FFT-line
/// index in the band (`index_fcb`, 1-based into Table D.1), the top
/// frequency `frequency_hz` (Hz, the frequency corresponding to that
/// FFT line per Table D.1), and the top Bark coordinate `z_bark` (the
/// critical-band rate `z` corresponding to that frequency per
/// Table D.1).
///
/// A band `k` spans the FFT lines from `boundaries[k - 1].index_fcb + 1`
/// (or `1` for `k = 0`) through `boundaries[k].index_fcb` inclusive,
/// covering the closed Bark interval
/// `[boundaries[k - 1].z_bark, boundaries[k].z_bark]` (the spec
/// boundaries are right-closed in Bark).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CriticalBandBoundary {
    /// Critical-band index (`no` column in the spec table), zero-based
    /// per the spec's `no 0..` numbering.
    pub no: u16,
    /// Top FFT-line index in this critical band (the spec's "index of
    /// F&CB" column — i.e. an index into the matching Table D.1
    /// frequency / critical-band table for this `(Layer, Fs)`).
    pub index_fcb: u16,
    /// Top frequency of this critical band, in Hz (the frequency
    /// corresponding to `index_fcb` per Table D.1).
    pub frequency_hz: f64,
    /// Top critical-band rate of this band, in Bark units `z` (the
    /// Bark coordinate corresponding to `frequency_hz` per Table D.1).
    pub z_bark: f64,
}

impl CriticalBandBoundary {
    /// Construct a boundary row at compile time.
    #[inline]
    #[must_use]
    pub const fn new(no: u16, index_fcb: u16, frequency_hz: f64, z_bark: f64) -> Self {
        Self {
            no,
            index_fcb,
            frequency_hz,
            z_bark,
        }
    }
}

/// Documented illegibility marker: D.2e band 17's Bark coordinate
/// prints as `16,11` with a clipped final digit in the staged PDF
/// render. This module records the legible prefix `16.11` (a strict
/// under-estimate within ±0.01 Bark of the true value, and not the
/// docs file's prose `16,116` estimate, which is explicitly NOT
/// adopted as a verbatim source value).
///
/// Marker set to `true` only for the affected (Layer II, 44.1 kHz)
/// row 17 cell; consumers that need to widen the under-estimate by
/// the `~0.006 Bark` typesetting tolerance can read this marker.
pub const D2E_BAND_17_BARK_IS_ILLEGIBLE: bool = true;

/// Table D.2a — Layer I, Fs = 32 kHz (24 bands, `no` 0..=23).
///
/// Verbatim from `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
/// §"Table D.2a - Layer I, Fs = 32 kHz (24 bands, no 0..23)".
pub const CRITICAL_BANDS_D2A: [CriticalBandBoundary; 24] = [
    CriticalBandBoundary::new(0, 1, 62.500, 0.617),
    CriticalBandBoundary::new(1, 3, 187.500, 1.842),
    CriticalBandBoundary::new(2, 5, 312.500, 3.037),
    CriticalBandBoundary::new(3, 7, 437.500, 4.185),
    CriticalBandBoundary::new(4, 9, 562.500, 5.272),
    CriticalBandBoundary::new(5, 11, 687.500, 6.289),
    CriticalBandBoundary::new(6, 13, 812.500, 7.233),
    CriticalBandBoundary::new(7, 15, 937.500, 8.103),
    CriticalBandBoundary::new(8, 18, 1125.000, 9.275),
    CriticalBandBoundary::new(9, 21, 1312.500, 10.301),
    CriticalBandBoundary::new(10, 24, 1500.000, 11.199),
    CriticalBandBoundary::new(11, 27, 1687.500, 11.988),
    CriticalBandBoundary::new(12, 32, 2000.000, 13.104),
    CriticalBandBoundary::new(13, 37, 2312.500, 14.027),
    CriticalBandBoundary::new(14, 44, 2750.000, 15.087),
    CriticalBandBoundary::new(15, 50, 3250.000, 16.069),
    CriticalBandBoundary::new(16, 55, 3875.000, 17.078),
    CriticalBandBoundary::new(17, 61, 4625.000, 18.089),
    CriticalBandBoundary::new(18, 68, 5500.000, 19.095),
    CriticalBandBoundary::new(19, 74, 6500.000, 20.079),
    CriticalBandBoundary::new(20, 79, 7750.000, 21.098),
    CriticalBandBoundary::new(21, 85, 9250.000, 22.046),
    CriticalBandBoundary::new(22, 94, 11500.000, 23.030),
    CriticalBandBoundary::new(23, 108, 15000.000, 23.923),
];

/// Table D.2b — Layer I, Fs = 44.1 kHz (25 bands, `no` 0..=24).
///
/// Verbatim from `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
/// §"Table D.2b - Layer I, Fs = 44,1 kHz (25 bands, no 0..24)".
pub const CRITICAL_BANDS_D2B: [CriticalBandBoundary; 25] = [
    CriticalBandBoundary::new(0, 1, 86.133, 0.850),
    CriticalBandBoundary::new(1, 2, 172.266, 1.694),
    CriticalBandBoundary::new(2, 3, 258.398, 2.525),
    CriticalBandBoundary::new(3, 5, 430.664, 4.124),
    CriticalBandBoundary::new(4, 6, 516.797, 4.882),
    CriticalBandBoundary::new(5, 8, 689.063, 6.301),
    CriticalBandBoundary::new(6, 9, 775.195, 6.959),
    CriticalBandBoundary::new(7, 11, 947.461, 8.169),
    CriticalBandBoundary::new(8, 13, 1119.727, 9.244),
    CriticalBandBoundary::new(9, 15, 1291.992, 10.195),
    CriticalBandBoundary::new(10, 17, 1464.258, 11.037),
    CriticalBandBoundary::new(11, 20, 1722.656, 12.125),
    CriticalBandBoundary::new(12, 23, 1981.055, 13.042),
    CriticalBandBoundary::new(13, 27, 2325.586, 14.062),
    CriticalBandBoundary::new(14, 32, 2756.250, 15.100),
    CriticalBandBoundary::new(15, 37, 3186.914, 15.955),
    CriticalBandBoundary::new(16, 45, 3875.977, 17.079),
    CriticalBandBoundary::new(17, 50, 4478.906, 17.904),
    CriticalBandBoundary::new(18, 55, 5340.234, 18.922),
    CriticalBandBoundary::new(19, 61, 6373.828, 19.963),
    CriticalBandBoundary::new(20, 68, 7579.688, 20.971),
    CriticalBandBoundary::new(21, 75, 9302.344, 22.074),
    CriticalBandBoundary::new(22, 81, 11369.531, 22.984),
    CriticalBandBoundary::new(23, 93, 15503.906, 24.013),
    CriticalBandBoundary::new(24, 106, 19982.813, 24.573),
];

/// Table D.2c — Layer I, Fs = 48 kHz (26 bands, `no` 0..=25).
///
/// Verbatim from `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
/// §"Table D.2c - Layer I, Fs = 48 kHz (26 bands, no 0..25)".
pub const CRITICAL_BANDS_D2C: [CriticalBandBoundary; 26] = [
    CriticalBandBoundary::new(0, 1, 93.750, 0.925),
    CriticalBandBoundary::new(1, 2, 187.500, 1.842),
    CriticalBandBoundary::new(2, 3, 281.250, 2.742),
    CriticalBandBoundary::new(3, 4, 375.000, 3.618),
    CriticalBandBoundary::new(4, 5, 468.750, 4.463),
    CriticalBandBoundary::new(5, 6, 562.500, 5.272),
    CriticalBandBoundary::new(6, 7, 656.250, 6.041),
    CriticalBandBoundary::new(7, 9, 843.750, 7.457),
    CriticalBandBoundary::new(8, 10, 937.500, 8.103),
    CriticalBandBoundary::new(9, 12, 1125.000, 9.275),
    CriticalBandBoundary::new(10, 14, 1312.500, 10.301),
    CriticalBandBoundary::new(11, 16, 1500.000, 11.199),
    CriticalBandBoundary::new(12, 19, 1781.250, 12.347),
    CriticalBandBoundary::new(13, 21, 1968.750, 13.002),
    CriticalBandBoundary::new(14, 25, 2343.750, 14.111),
    CriticalBandBoundary::new(15, 29, 2718.750, 15.018),
    CriticalBandBoundary::new(16, 35, 3281.250, 16.124),
    CriticalBandBoundary::new(17, 41, 3843.750, 17.032),
    CriticalBandBoundary::new(18, 49, 4687.500, 18.166),
    CriticalBandBoundary::new(19, 53, 5437.500, 19.028),
    CriticalBandBoundary::new(20, 58, 6375.000, 19.964),
    CriticalBandBoundary::new(21, 65, 7687.500, 21.052),
    CriticalBandBoundary::new(22, 73, 9375.000, 22.113),
    CriticalBandBoundary::new(23, 79, 11625.000, 23.072),
    CriticalBandBoundary::new(24, 89, 15375.000, 23.991),
    CriticalBandBoundary::new(25, 102, 20250.000, 24.597),
];

/// Table D.2d — Layer II, Fs = 32 kHz (25 bands, `no` 0..=24).
///
/// Verbatim from `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
/// §"Table D.2d - Layer II, Fs = 32 kHz (25 bands, no 0..24)".
pub const CRITICAL_BANDS_D2D: [CriticalBandBoundary; 25] = [
    CriticalBandBoundary::new(0, 1, 31.250, 0.309),
    CriticalBandBoundary::new(1, 3, 93.750, 0.925),
    CriticalBandBoundary::new(2, 6, 187.500, 1.842),
    CriticalBandBoundary::new(3, 10, 312.500, 3.037),
    CriticalBandBoundary::new(4, 13, 406.250, 3.903),
    CriticalBandBoundary::new(5, 17, 531.250, 5.006),
    CriticalBandBoundary::new(6, 21, 656.250, 6.041),
    CriticalBandBoundary::new(7, 25, 781.250, 7.004),
    CriticalBandBoundary::new(8, 30, 937.500, 8.103),
    CriticalBandBoundary::new(9, 35, 1093.750, 9.090),
    CriticalBandBoundary::new(10, 41, 1281.250, 10.139),
    CriticalBandBoundary::new(11, 47, 1468.750, 11.058),
    CriticalBandBoundary::new(12, 51, 1687.500, 11.988),
    CriticalBandBoundary::new(13, 56, 2000.000, 13.104),
    CriticalBandBoundary::new(14, 61, 2312.500, 14.027),
    CriticalBandBoundary::new(15, 68, 2750.000, 15.087),
    CriticalBandBoundary::new(16, 74, 3250.000, 16.069),
    CriticalBandBoundary::new(17, 79, 3875.000, 17.078),
    CriticalBandBoundary::new(18, 85, 4625.000, 18.089),
    CriticalBandBoundary::new(19, 92, 5500.000, 19.095),
    CriticalBandBoundary::new(20, 98, 6500.000, 20.079),
    CriticalBandBoundary::new(21, 103, 7750.000, 21.098),
    CriticalBandBoundary::new(22, 109, 9250.000, 22.046),
    CriticalBandBoundary::new(23, 118, 11500.000, 23.030),
    CriticalBandBoundary::new(24, 132, 15000.000, 23.923),
];

/// Table D.2e — Layer II, Fs = 44.1 kHz (27 bands, `no` 0..=26).
///
/// Verbatim from `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
/// §"Table D.2e - Layer II, Fs = 44,1 kHz (27 bands, no 0..26)".
///
/// **Illegible cell:** row 17, `z_bark`. The PDF render clips the
/// final digit, printing `16,11` with a fragment of a fourth digit.
/// The docs file marks it `[illegible]`. This array records the
/// legible-only value `16.11`; the docs file's prose estimate
/// (`16,116`) is explicitly NOT adopted. See
/// [`D2E_BAND_17_BARK_IS_ILLEGIBLE`].
pub const CRITICAL_BANDS_D2E: [CriticalBandBoundary; 27] = [
    CriticalBandBoundary::new(0, 1, 43.066, 0.425),
    CriticalBandBoundary::new(1, 2, 86.133, 0.850),
    CriticalBandBoundary::new(2, 3, 129.199, 1.273),
    CriticalBandBoundary::new(3, 5, 215.332, 2.112),
    CriticalBandBoundary::new(4, 7, 301.465, 2.934),
    CriticalBandBoundary::new(5, 10, 430.664, 4.124),
    CriticalBandBoundary::new(6, 13, 559.863, 5.249),
    CriticalBandBoundary::new(7, 16, 689.063, 6.301),
    CriticalBandBoundary::new(8, 19, 818.262, 7.274),
    CriticalBandBoundary::new(9, 22, 947.461, 8.169),
    CriticalBandBoundary::new(10, 26, 1119.727, 9.244),
    CriticalBandBoundary::new(11, 30, 1291.992, 10.195),
    CriticalBandBoundary::new(12, 35, 1507.324, 11.232),
    CriticalBandBoundary::new(13, 40, 1722.656, 12.125),
    CriticalBandBoundary::new(14, 46, 1981.055, 13.042),
    CriticalBandBoundary::new(15, 51, 2325.586, 14.062),
    CriticalBandBoundary::new(16, 56, 2756.250, 15.100),
    // Row 17: docs marks `z_bark` as `16,11[illegible]`. Legible-
    // only value transcribed; see `D2E_BAND_17_BARK_IS_ILLEGIBLE`.
    CriticalBandBoundary::new(17, 62, 3273.047, 16.11),
    CriticalBandBoundary::new(18, 69, 3875.977, 17.079),
    CriticalBandBoundary::new(19, 74, 4478.906, 17.904),
    CriticalBandBoundary::new(20, 79, 5340.234, 18.922),
    CriticalBandBoundary::new(21, 85, 6373.828, 19.963),
    CriticalBandBoundary::new(22, 92, 7579.688, 20.971),
    CriticalBandBoundary::new(23, 99, 9302.344, 22.074),
    CriticalBandBoundary::new(24, 105, 11369.531, 22.984),
    CriticalBandBoundary::new(25, 117, 15503.906, 24.013),
    CriticalBandBoundary::new(26, 130, 19982.813, 24.573),
];

/// Table D.2f — Layer II, Fs = 48 kHz (27 bands, `no` 0..=26).
///
/// Verbatim from `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
/// §"Table D.2f - Layer II, Fs = 48 kHz (27 bands, no 0..26)".
pub const CRITICAL_BANDS_D2F: [CriticalBandBoundary; 27] = [
    CriticalBandBoundary::new(0, 1, 46.875, 0.463),
    CriticalBandBoundary::new(1, 2, 93.750, 0.925),
    CriticalBandBoundary::new(2, 3, 140.625, 1.385),
    CriticalBandBoundary::new(3, 5, 234.375, 2.295),
    CriticalBandBoundary::new(4, 7, 328.125, 3.184),
    CriticalBandBoundary::new(5, 9, 421.875, 4.045),
    CriticalBandBoundary::new(6, 12, 562.500, 5.272),
    CriticalBandBoundary::new(7, 14, 656.250, 6.041),
    CriticalBandBoundary::new(8, 17, 796.875, 7.119),
    CriticalBandBoundary::new(9, 20, 937.500, 8.103),
    CriticalBandBoundary::new(10, 24, 1125.000, 9.275),
    CriticalBandBoundary::new(11, 27, 1265.625, 10.057),
    CriticalBandBoundary::new(12, 32, 1500.000, 11.199),
    CriticalBandBoundary::new(13, 37, 1734.375, 12.170),
    CriticalBandBoundary::new(14, 42, 1968.750, 13.002),
    CriticalBandBoundary::new(15, 49, 2343.750, 14.111),
    CriticalBandBoundary::new(16, 53, 2718.750, 15.018),
    CriticalBandBoundary::new(17, 59, 3281.250, 16.124),
    CriticalBandBoundary::new(18, 65, 3843.750, 17.032),
    CriticalBandBoundary::new(19, 73, 4687.500, 18.166),
    CriticalBandBoundary::new(20, 77, 5437.500, 19.028),
    CriticalBandBoundary::new(21, 82, 6375.000, 19.964),
    CriticalBandBoundary::new(22, 89, 7687.500, 21.052),
    CriticalBandBoundary::new(23, 97, 9375.000, 22.113),
    CriticalBandBoundary::new(24, 103, 11625.000, 23.072),
    CriticalBandBoundary::new(25, 113, 15375.000, 23.991),
    CriticalBandBoundary::new(26, 126, 20250.000, 24.597),
];

/// One of the three Annex D sampling frequencies the critical-band
/// tables are defined for (`32`, `44.1`, `48` kHz). The Annex D
/// dispatch key is the (Layer, Fs) pair; this enum collapses the
/// three integer rates onto a typed key so callers cannot accidentally
/// request a non-existent table for, e.g., 16 kHz (the LSF rates of
/// MPEG-2 lower-sampling-frequency are out of scope — Annex D is
/// MPEG-1 only).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AnnexDSamplingRate {
    /// 32 kHz.
    Hz32000,
    /// 44.1 kHz.
    Hz44100,
    /// 48 kHz.
    Hz48000,
}

impl AnnexDSamplingRate {
    /// Construct from a raw Hz value, returning `None` for any rate
    /// outside the Annex D set (`32000`, `44100`, `48000`).
    #[inline]
    #[must_use]
    pub const fn from_hz(hz: u32) -> Option<Self> {
        match hz {
            32_000 => Some(Self::Hz32000),
            44_100 => Some(Self::Hz44100),
            48_000 => Some(Self::Hz48000),
            _ => None,
        }
    }

    /// Sampling rate in Hz.
    #[inline]
    #[must_use]
    pub const fn as_hz(self) -> u32 {
        match self {
            Self::Hz32000 => 32_000,
            Self::Hz44100 => 44_100,
            Self::Hz48000 => 48_000,
        }
    }
}

/// Return the verbatim Annex D Table D.2 critical-band-boundary slice
/// for `(layer, fs)`. Returns `None` for Layer III (Annex D is
/// defined only for Layer I and Layer II — Layer III's psychoacoustic
/// model selection is described in clause C.1.5.3.2.1 which re-uses
/// the Layer I/II tables with a Layer-III-specific spreading-function
/// override, so a Layer III caller should pass the matching Layer
/// (`LayerI` or `LayerII`) explicitly per the Annex D scope).
#[inline]
#[must_use]
pub fn critical_band_boundaries(
    layer: crate::frame::Layer,
    fs: AnnexDSamplingRate,
) -> Option<&'static [CriticalBandBoundary]> {
    use crate::frame::Layer;
    match (layer, fs) {
        (Layer::LayerI, AnnexDSamplingRate::Hz32000) => Some(&CRITICAL_BANDS_D2A),
        (Layer::LayerI, AnnexDSamplingRate::Hz44100) => Some(&CRITICAL_BANDS_D2B),
        (Layer::LayerI, AnnexDSamplingRate::Hz48000) => Some(&CRITICAL_BANDS_D2C),
        (Layer::LayerII, AnnexDSamplingRate::Hz32000) => Some(&CRITICAL_BANDS_D2D),
        (Layer::LayerII, AnnexDSamplingRate::Hz44100) => Some(&CRITICAL_BANDS_D2E),
        (Layer::LayerII, AnnexDSamplingRate::Hz48000) => Some(&CRITICAL_BANDS_D2F),
        (Layer::LayerIII, _) => None,
    }
}

/// Map an FFT-line index `i` (1-based, into the matching Table D.1
/// frequency / critical-band table) to the critical-band index `no`
/// it falls into. Returns `None` only if `i` is `0` (the spec's
/// indices are 1-based) or exceeds the largest band's `index_fcb`
/// (an FFT line above the audio band of the table).
#[inline]
#[must_use]
pub fn band_of_fft_line(boundaries: &[CriticalBandBoundary], fft_line_index: u16) -> Option<u16> {
    if fft_line_index == 0 {
        return None;
    }
    for b in boundaries {
        if fft_line_index <= b.index_fcb {
            return Some(b.no);
        }
    }
    None
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

    // =====================================================================
    // `threshold_in_quiet` (r197 — short / mixed cells now derived) tests
    // =====================================================================

    #[test]
    fn threshold_in_quiet_long_cells_match_threshold_in_quiet_long() {
        // The long cells produced by `threshold_in_quiet` must match
        // those produced by `threshold_in_quiet_long` at the same
        // bitrate (the long-band derivation is identical; the only
        // difference is that `threshold_in_quiet` also derives the short
        // / mixed cells).
        let both = XminThresholds::threshold_in_quiet(44_100, MpegVersion::Mpeg1, 128);
        let long_only = XminThresholds::threshold_in_quiet_long(
            44_100,
            MpegVersion::Mpeg1,
            128,
            DEFAULT_OUTER_LOOP_THRESHOLD,
        );
        for sfb in 0..LONG_SFB {
            assert!(
                (both.long[sfb] - long_only.long[sfb]).abs() < 1.0e-9,
                "sfb {sfb}: both.long={} long_only.long={}",
                both.long[sfb],
                long_only.long[sfb],
            );
        }
    }

    #[test]
    fn threshold_in_quiet_short_cells_share_per_band_value_across_windows() {
        // The threshold-in-quiet is purely a function of frequency
        // (Annex D Table D.1), so the three windows of each short SFB
        // must carry the same per-cell xmin.
        let x = XminThresholds::threshold_in_quiet(44_100, MpegVersion::Mpeg1, 128);
        for sfb in 0..SHORT_SFB {
            let w0 = x.short[sfb][0];
            for win in 1..SHORT_WINDOWS {
                assert!(
                    (x.short[sfb][win] - w0).abs() < 1.0e-12,
                    "sfb {sfb}: window {win} ({}) differs from window 0 ({w0})",
                    x.short[sfb][win],
                );
            }
        }
    }

    #[test]
    fn threshold_in_quiet_short_high_bitrate_applies_offset() {
        let high = XminThresholds::threshold_in_quiet(44_100, MpegVersion::Mpeg1, 128);
        let low = XminThresholds::threshold_in_quiet(44_100, MpegVersion::Mpeg1, 64);
        // Same monotone offset as the long path: `−12 dB` ⇒ each high
        // entry is `10^(−12/10)` times the low entry (factor ~0.0631).
        for sfb in 0..SHORT_SFB {
            let ratio = low.short[sfb][0] / high.short[sfb][0];
            assert!(
                (ratio - 10.0_f64.powf(12.0 / 10.0)).abs() < 1.0e-6,
                "sfb {sfb}: ratio {ratio} should be 10^1.2",
            );
        }
    }

    #[test]
    fn threshold_in_quiet_short_band_shape_is_bowl() {
        // Same bowl as the long path: a minimum in the mid-spectrum,
        // higher thresholds at the low- and high-band ends. The short-
        // band centre-frequency mapping uses `Fs/384` per per-window
        // line — band 11 at 44.1 kHz covers per-window lines [106, 136),
        // centre ≈ 13.4 kHz, well into the rising treble edge of LTq.
        let x = XminThresholds::threshold_in_quiet(44_100, MpegVersion::Mpeg1, 128);
        let bass = x.short[0][0];
        let treble = x.short[SHORT_SFB - 1][0];
        let (min_sfb, &min_v) = x
            .short
            .iter()
            .enumerate()
            .map(|(sfb, row)| (sfb, &row[0]))
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        assert!(
            bass > min_v,
            "bass sfb 0 ({bass}) should be > minimum sfb {min_sfb} ({min_v})",
        );
        assert!(
            treble > min_v,
            "treble sfb 11 ({treble}) should be > minimum sfb {min_sfb} ({min_v})",
        );
        assert!((1..SHORT_SFB - 1).contains(&min_sfb));
    }

    #[test]
    fn threshold_in_quiet_mixed_short_matches_short() {
        // Mixed-short carries the same per-band values as pure-short
        // (same frequency mapping; mixed blocks just don't read sfb
        // 0..=2). Convenient invariant for callers that swap a
        // pure-short granule for a mixed granule without re-computing
        // the matrix.
        let x = XminThresholds::threshold_in_quiet(44_100, MpegVersion::Mpeg1, 128);
        for sfb in 0..SHORT_SFB {
            for win in 0..SHORT_WINDOWS {
                assert!(
                    (x.mixed_short[sfb][win] - x.short[sfb][win]).abs() < 1.0e-12,
                    "mixed_short[{sfb}][{win}] != short[{sfb}][{win}]",
                );
            }
        }
    }

    // =====================================================================
    // `threshold_in_quiet_with_offset_db` — caller-supplied §D.1 Step 3
    // offset (r213).
    // =====================================================================

    #[test]
    fn threshold_in_quiet_with_offset_db_recovers_spec_high_bitrate_path() {
        // `offset_db = -12.0` (the spec's `bitrate_kbps_per_channel >= 96`
        // path) must reproduce the same vector as
        // `threshold_in_quiet(_, _, 128)` to within FP tolerance, for all
        // cells (long / short / mixed).
        let custom =
            XminThresholds::threshold_in_quiet_with_offset_db(44_100, MpegVersion::Mpeg1, -12.0);
        let spec = XminThresholds::threshold_in_quiet(44_100, MpegVersion::Mpeg1, 128);
        for sfb in 0..LONG_SFB {
            assert!(
                (custom.long[sfb] - spec.long[sfb]).abs() < 1.0e-9,
                "long sfb {sfb}: custom {} vs spec {}",
                custom.long[sfb],
                spec.long[sfb],
            );
            assert!((custom.mixed_long[sfb] - spec.mixed_long[sfb]).abs() < 1.0e-9);
        }
        for sfb in 0..SHORT_SFB {
            for win in 0..SHORT_WINDOWS {
                assert!(
                    (custom.short[sfb][win] - spec.short[sfb][win]).abs() < 1.0e-9,
                    "short [{sfb}][{win}]: custom {} vs spec {}",
                    custom.short[sfb][win],
                    spec.short[sfb][win],
                );
                assert!((custom.mixed_short[sfb][win] - spec.mixed_short[sfb][win]).abs() < 1.0e-9);
            }
        }
    }

    #[test]
    fn threshold_in_quiet_with_offset_db_recovers_spec_low_bitrate_path() {
        // `offset_db = 0.0` (the spec's `bitrate_kbps_per_channel < 96`
        // path) must reproduce the same vector as
        // `threshold_in_quiet(_, _, 64)`.
        let custom =
            XminThresholds::threshold_in_quiet_with_offset_db(44_100, MpegVersion::Mpeg1, 0.0);
        let spec = XminThresholds::threshold_in_quiet(44_100, MpegVersion::Mpeg1, 64);
        for sfb in 0..LONG_SFB {
            assert!(
                (custom.long[sfb] - spec.long[sfb]).abs() < 1.0e-9,
                "long sfb {sfb}: custom {} vs spec {}",
                custom.long[sfb],
                spec.long[sfb],
            );
        }
        for sfb in 0..SHORT_SFB {
            for win in 0..SHORT_WINDOWS {
                assert!(
                    (custom.short[sfb][win] - spec.short[sfb][win]).abs() < 1.0e-9,
                    "short [{sfb}][{win}]: custom {} vs spec {}",
                    custom.short[sfb][win],
                    spec.short[sfb][win],
                );
            }
        }
    }

    #[test]
    fn threshold_in_quiet_with_offset_db_tightens_below_spec_minus12() {
        // An offset stricter than `−12 dB` translates the bowl down
        // (linear ratio `10^(offset/10)`): every cell of the
        // `offset = −24 dB` vector is exactly `10^(−12/10)` times the
        // corresponding cell of the `offset = −12 dB` vector — same
        // bowl shape, shifted floor.
        let spec =
            XminThresholds::threshold_in_quiet_with_offset_db(44_100, MpegVersion::Mpeg1, -12.0);
        let tighter =
            XminThresholds::threshold_in_quiet_with_offset_db(44_100, MpegVersion::Mpeg1, -24.0);
        let expected_ratio = 10.0_f64.powf(-12.0 / 10.0);
        for sfb in 0..LONG_SFB {
            let ratio = tighter.long[sfb] / spec.long[sfb];
            assert!(
                (ratio - expected_ratio).abs() < 1.0e-9,
                "long sfb {sfb}: ratio {ratio} should be {expected_ratio}",
            );
        }
        for sfb in 0..SHORT_SFB {
            let ratio = tighter.short[sfb][0] / spec.short[sfb][0];
            assert!(
                (ratio - expected_ratio).abs() < 1.0e-9,
                "short sfb {sfb}: ratio {ratio} should be {expected_ratio}",
            );
        }
    }

    #[test]
    fn threshold_in_quiet_with_offset_db_loosens_above_zero() {
        // Mirror of the previous test: an offset above `0 dB` translates
        // the bowl up by the same linear ratio.
        let spec =
            XminThresholds::threshold_in_quiet_with_offset_db(44_100, MpegVersion::Mpeg1, 0.0);
        let looser =
            XminThresholds::threshold_in_quiet_with_offset_db(44_100, MpegVersion::Mpeg1, 6.0);
        let expected_ratio = 10.0_f64.powf(6.0 / 10.0);
        for sfb in 0..LONG_SFB {
            let ratio = looser.long[sfb] / spec.long[sfb];
            assert!(
                (ratio - expected_ratio).abs() < 1.0e-6,
                "long sfb {sfb}: ratio {ratio} should be {expected_ratio}",
            );
        }
    }

    #[test]
    fn threshold_in_quiet_with_offset_db_preserves_bowl_shape() {
        // The bowl-vs-bass-vs-treble invariants of the spec-default
        // `threshold_in_quiet` must survive an arbitrary offset (the
        // offset is a uniform dB translation; the per-frequency
        // ordering is unchanged).
        let x =
            XminThresholds::threshold_in_quiet_with_offset_db(44_100, MpegVersion::Mpeg1, -30.0);
        let bass = x.long[0];
        let treble = x.long[LONG_SFB - 1];
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
        assert!(
            treble > min_v,
            "treble sfb 20 ({treble}) should be > minimum sfb {min_sfb} ({min_v})",
        );
        assert!((1..LONG_SFB - 1).contains(&min_sfb));
    }

    // =====================================================================
    // §D.1 Step 6 masking-index / masking-function / individual threshold
    // and §D.1 Step 7 global-threshold tests (Phase 2 step 44 / r219).
    // =====================================================================

    #[test]
    fn masking_index_tonal_recovers_spec_formula() {
        // Verbatim spec: `av_tm = -1.525 - 0.275 * z(j) - 4.5`.
        // Spot-check three Bark positions across the audio band.
        for z_j in [0.0_f64, 5.0, 12.0, 20.0, 25.0] {
            let got = masking_index_tonal(z_j);
            let expected = -1.525 - 0.275 * z_j - 4.5;
            assert!(
                (got - expected).abs() < 1.0e-12,
                "av_tm({z_j}) = {got}, expected {expected}",
            );
        }
    }

    #[test]
    fn masking_index_non_tonal_recovers_spec_formula() {
        // Verbatim spec: `av_nm = -1.525 - 0.175 * z(j) - 0.5`.
        for z_j in [0.0_f64, 5.0, 12.0, 20.0, 25.0] {
            let got = masking_index_non_tonal(z_j);
            let expected = -1.525 - 0.175 * z_j - 0.5;
            assert!(
                (got - expected).abs() < 1.0e-12,
                "av_nm({z_j}) = {got}, expected {expected}",
            );
        }
    }

    #[test]
    fn masking_index_tonal_below_non_tonal_at_same_z() {
        // The tonal constant `-4.5` is deeper than the non-tonal `-0.5`,
        // so for any Bark position the tonal masking-index sits below
        // the non-tonal one (a more efficient masker pushes its
        // threshold floor further below its SPL).
        for z_j in [0.0_f64, 5.0, 12.0, 20.0] {
            let tm = masking_index_tonal(z_j);
            let nm = masking_index_non_tonal(z_j);
            assert!(
                tm < nm,
                "tonal {tm} should be < non-tonal {nm} at z(j) = {z_j}",
            );
        }
    }

    #[test]
    fn masking_function_vf_out_of_range_returns_none() {
        // `vf` is defined on `[-3, 8)`. Outside the masker is ignored.
        assert!(masking_function_vf(-3.0001, 60.0).is_none());
        assert!(masking_function_vf(-10.0, 60.0).is_none());
        assert!(masking_function_vf(8.0, 60.0).is_none());
        assert!(masking_function_vf(8.5, 60.0).is_none());
        // Boundary inside the half-open `[-3, 8)` is in-range.
        assert!(masking_function_vf(-3.0, 60.0).is_some());
        assert!(masking_function_vf(7.999, 60.0).is_some());
    }

    #[test]
    fn masking_function_vf_branch_far_left_lobe() {
        // Branch 1, `-3 <= dz < -1`:
        //   vf = 17 * (dz + 1) - (0.4 * X + 6)
        // At dz = -3 the masker contribution is far below the SPL:
        //   vf(-3, 60) = 17 * (-2) - (0.4 * 60 + 6) = -34 - 30 = -64
        let v = masking_function_vf(-3.0, 60.0).unwrap();
        assert!((v - (-64.0)).abs() < 1.0e-12, "vf(-3, 60) = {v}");
        // dz = -2, X = 80:
        //   vf = 17 * (-1) - (0.4 * 80 + 6) = -17 - 38 = -55
        let v = masking_function_vf(-2.0, 80.0).unwrap();
        assert!((v - (-55.0)).abs() < 1.0e-12, "vf(-2, 80) = {v}");
    }

    #[test]
    fn masking_function_vf_branch_near_left_lobe() {
        // Branch 2, `-1 <= dz < 0`:
        //   vf = (0.4 * X + 6) * dz
        // At dz = -1, X = 60: vf = (24 + 6) * (-1) = -30.
        let v = masking_function_vf(-1.0, 60.0).unwrap();
        assert!((v - (-30.0)).abs() < 1.0e-12, "vf(-1, 60) = {v}");
        // At dz = -0.5, X = 60: vf = 30 * (-0.5) = -15.
        let v = masking_function_vf(-0.5, 60.0).unwrap();
        assert!((v - (-15.0)).abs() < 1.0e-12, "vf(-0.5, 60) = {v}");
    }

    #[test]
    fn masking_function_vf_branch_near_right_lobe() {
        // Branch 3, `0 <= dz < 1`:
        //   vf = -17 * dz
        // At dz = 0: vf = 0.
        let v = masking_function_vf(0.0, 60.0).unwrap();
        assert!(v.abs() < 1.0e-12, "vf(0, 60) = {v}");
        // At dz = 0.5: vf = -8.5.
        let v = masking_function_vf(0.5, 60.0).unwrap();
        assert!((v - (-8.5)).abs() < 1.0e-12, "vf(0.5, 60) = {v}");
        // At dz = 0.999: vf ~= -16.983.
        let v = masking_function_vf(0.999, 60.0).unwrap();
        assert!((v - (-17.0 * 0.999)).abs() < 1.0e-12, "vf(0.999, 60) = {v}");
    }

    #[test]
    fn masking_function_vf_branch_far_right_lobe() {
        // Branch 4, `1 <= dz < 8`:
        //   vf = -(dz - 1) * (17 - 0.15 * X) - 17
        // At dz = 1, X = 60: vf = -0 * (17 - 9) - 17 = -17.
        let v = masking_function_vf(1.0, 60.0).unwrap();
        assert!((v - (-17.0)).abs() < 1.0e-12, "vf(1, 60) = {v}");
        // At dz = 2, X = 60: vf = -1 * (17 - 9) - 17 = -8 - 17 = -25.
        let v = masking_function_vf(2.0, 60.0).unwrap();
        assert!((v - (-25.0)).abs() < 1.0e-12, "vf(2, 60) = {v}");
        // At dz = 5, X = 80: vf = -4 * (17 - 12) - 17 = -20 - 17 = -37.
        let v = masking_function_vf(5.0, 80.0).unwrap();
        assert!((v - (-37.0)).abs() < 1.0e-12, "vf(5, 80) = {v}");
    }

    #[test]
    fn masking_function_vf_continuous_at_dz_zero() {
        // Branches 2 and 3 must agree at `dz = 0`: branch 2 gives
        // `(0.4 * X + 6) * 0 = 0`, branch 3 gives `-17 * 0 = 0`.
        // Continuity at the dz = 0 boundary preserves the spec's
        // implicit "masker SPL is the unattenuated peak" property.
        for x in [40.0_f64, 60.0, 80.0, 100.0] {
            // Approach from the left (branch 2).
            let left = masking_function_vf(-1.0e-12, x).unwrap();
            // Exactly at zero (branch 3, since the if-chain checks
            // `< 0` then `< 1`).
            let at = masking_function_vf(0.0, x).unwrap();
            assert!(left.abs() < 1.0e-9, "left limit at X = {x}: {left}");
            assert!(at.abs() < 1.0e-12, "exactly at zero, X = {x}: {at}");
        }
    }

    #[test]
    fn individual_masking_threshold_db_tonal_at_self_is_spl_plus_av() {
        // At `z(i) = z(j)` (dz = 0) the masking function `vf = 0`, so
        // LT_tm = SPL + av_tm = SPL + (-1.525 - 0.275 * z - 4.5).
        let masker = Masker {
            kind: MaskerKind::Tonal,
            z_bark: 10.0,
            spl_db: 80.0,
        };
        let lt = individual_masking_threshold_db(&masker, 10.0).unwrap();
        let expected = 80.0 + masking_index_tonal(10.0);
        assert!(
            (lt - expected).abs() < 1.0e-12,
            "LT_tm(self) = {lt}, expected {expected}",
        );
    }

    #[test]
    fn individual_masking_threshold_db_non_tonal_at_self_is_spl_plus_av() {
        // Same invariant for non-tonal maskers.
        let masker = Masker {
            kind: MaskerKind::NonTonal,
            z_bark: 10.0,
            spl_db: 80.0,
        };
        let lt = individual_masking_threshold_db(&masker, 10.0).unwrap();
        let expected = 80.0 + masking_index_non_tonal(10.0);
        assert!(
            (lt - expected).abs() < 1.0e-12,
            "LT_nm(self) = {lt}, expected {expected}",
        );
    }

    #[test]
    fn individual_masking_threshold_db_returns_none_outside_window() {
        // Masker at z(j) = 5, target at z(i) = 14 -> dz = 9, outside
        // the `[-3, 8)` `vf` window.
        let masker = Masker {
            kind: MaskerKind::Tonal,
            z_bark: 5.0,
            spl_db: 80.0,
        };
        assert!(individual_masking_threshold_db(&masker, 14.0).is_none());
        // Masker at z(j) = 10, target at z(i) = 6.5 -> dz = -3.5,
        // outside the window on the low side.
        assert!(individual_masking_threshold_db(&masker, 1.4).is_none());
    }

    #[test]
    fn individual_masking_threshold_db_tonal_below_non_tonal_at_same_z() {
        // Same masker position + SPL, same target z(i): the tonal
        // individual threshold sits below the non-tonal one (deeper
        // masking-index `av_tm < av_nm`).
        let z_j = 10.0;
        let spl = 80.0;
        let tm = Masker {
            kind: MaskerKind::Tonal,
            z_bark: z_j,
            spl_db: spl,
        };
        let nm = Masker {
            kind: MaskerKind::NonTonal,
            z_bark: z_j,
            spl_db: spl,
        };
        // Test at several z(i) inside the window.
        for z_i in [9.0_f64, 10.0, 11.5, 14.0, 17.0] {
            let lt_t = individual_masking_threshold_db(&tm, z_i).unwrap();
            let lt_n = individual_masking_threshold_db(&nm, z_i).unwrap();
            assert!(
                lt_t < lt_n,
                "tonal LT {lt_t} should be < non-tonal LT {lt_n} at z(i) = {z_i}",
            );
        }
    }

    #[test]
    fn global_masking_threshold_db_no_maskers_is_ltq() {
        // With zero maskers the energy sum is just 10^(LTq/10), so
        // LTg = LTq exactly.
        let ltg = global_masking_threshold_db(&[], 10.0, -5.0);
        assert!(
            (ltg - (-5.0)).abs() < 1.0e-12,
            "LTg(no maskers) = {ltg}, expected -5.0",
        );
    }

    #[test]
    fn global_masking_threshold_db_distant_masker_drops_to_ltq() {
        // A masker outside the `[-3, 8)` window doesn't contribute,
        // so LTg collapses back to LTq.
        let masker = Masker {
            kind: MaskerKind::Tonal,
            z_bark: 5.0,
            spl_db: 80.0,
        };
        let ltg = global_masking_threshold_db(&[masker], 20.0, 10.0);
        assert!(
            (ltg - 10.0).abs() < 1.0e-12,
            "LTg(distant masker) = {ltg}, expected LTq = 10.0",
        );
    }

    #[test]
    fn global_masking_threshold_db_strong_local_masker_dominates_ltq() {
        // A strong nearby masker should drive LTg far above LTq.
        let masker = Masker {
            kind: MaskerKind::Tonal,
            z_bark: 10.0,
            spl_db: 80.0,
        };
        // At z(i) = z(j): LT = 80 + av_tm(10) ≈ 80 - 6.775 = 73.225 dB,
        // which dwarfs LTq = 0.
        let ltg = global_masking_threshold_db(&[masker], 10.0, 0.0);
        let lt_at_self = individual_masking_threshold_db(&masker, 10.0).unwrap();
        // The masker contribution dominates: LTg ≈ LT_at_self.
        assert!(
            (ltg - lt_at_self).abs() < 1.0,
            "LTg {ltg} should be close to LT_at_self {lt_at_self}",
        );
        assert!(ltg > 0.0, "LTg {ltg} should be > LTq = 0");
    }

    #[test]
    fn global_masking_threshold_db_sums_energies_monotonically() {
        // Two maskers stack: LTg with both is strictly above LTg
        // with either alone (power addition is monotone in number of
        // sources).
        let m1 = Masker {
            kind: MaskerKind::Tonal,
            z_bark: 10.0,
            spl_db: 60.0,
        };
        let m2 = Masker {
            kind: MaskerKind::NonTonal,
            z_bark: 11.0,
            spl_db: 60.0,
        };
        let ltq = -10.0;
        let z_i = 10.5;
        let ltg_m1 = global_masking_threshold_db(&[m1], z_i, ltq);
        let ltg_m2 = global_masking_threshold_db(&[m2], z_i, ltq);
        let ltg_both = global_masking_threshold_db(&[m1, m2], z_i, ltq);
        assert!(
            ltg_both > ltg_m1,
            "LTg both {ltg_both} should be > LTg m1 alone {ltg_m1}",
        );
        assert!(
            ltg_both > ltg_m2,
            "LTg both {ltg_both} should be > LTg m2 alone {ltg_m2}",
        );
    }

    #[test]
    fn global_masking_threshold_db_two_equal_powers_add_three_db() {
        // Two equal-power sources sum to exactly +3.0103 dB above
        // either one (`10 * log10(2)`). Use two co-located masker
        // contributions at z(i) = z(j) and dial LTq far below so it
        // doesn't influence the sum.
        let m = Masker {
            kind: MaskerKind::Tonal,
            z_bark: 10.0,
            spl_db: 80.0,
        };
        let z_i = 10.0;
        let ltq = -200.0; // Effectively zero contribution.
        let single = global_masking_threshold_db(&[m], z_i, ltq);
        let double = global_masking_threshold_db(&[m, m], z_i, ltq);
        let expected = single + 10.0 * 2.0_f64.log10();
        assert!(
            (double - expected).abs() < 1.0e-9,
            "double {double} - single {single} = {} dB, expected +3.0103",
            double - single,
        );
    }

    // -------------------------------------------------------------------
    // Annex D Table D.2a–f critical-band-boundary tests (Phase 2 step 45).
    // -------------------------------------------------------------------

    #[test]
    fn d2a_has_24_bands_numbered_0_through_23() {
        assert_eq!(CRITICAL_BANDS_D2A.len(), 24);
        for (k, b) in CRITICAL_BANDS_D2A.iter().enumerate() {
            assert_eq!(b.no as usize, k, "D.2a row {k} has wrong `no`");
        }
    }

    #[test]
    fn d2b_has_25_bands_numbered_0_through_24() {
        assert_eq!(CRITICAL_BANDS_D2B.len(), 25);
        for (k, b) in CRITICAL_BANDS_D2B.iter().enumerate() {
            assert_eq!(b.no as usize, k, "D.2b row {k} has wrong `no`");
        }
    }

    #[test]
    fn d2c_has_26_bands_numbered_0_through_25() {
        assert_eq!(CRITICAL_BANDS_D2C.len(), 26);
        for (k, b) in CRITICAL_BANDS_D2C.iter().enumerate() {
            assert_eq!(b.no as usize, k, "D.2c row {k} has wrong `no`");
        }
    }

    #[test]
    fn d2d_has_25_bands_numbered_0_through_24() {
        assert_eq!(CRITICAL_BANDS_D2D.len(), 25);
        for (k, b) in CRITICAL_BANDS_D2D.iter().enumerate() {
            assert_eq!(b.no as usize, k, "D.2d row {k} has wrong `no`");
        }
    }

    #[test]
    fn d2e_has_27_bands_numbered_0_through_26() {
        assert_eq!(CRITICAL_BANDS_D2E.len(), 27);
        for (k, b) in CRITICAL_BANDS_D2E.iter().enumerate() {
            assert_eq!(b.no as usize, k, "D.2e row {k} has wrong `no`");
        }
    }

    #[test]
    fn d2f_has_27_bands_numbered_0_through_26() {
        assert_eq!(CRITICAL_BANDS_D2F.len(), 27);
        for (k, b) in CRITICAL_BANDS_D2F.iter().enumerate() {
            assert_eq!(b.no as usize, k, "D.2f row {k} has wrong `no`");
        }
    }

    #[test]
    fn d2a_first_row_matches_docs_anchor() {
        // Spec anchor (D.2a row 0): no = 0, index = 1, freq = 62.500,
        // z = 0.617.
        let row = CRITICAL_BANDS_D2A[0];
        assert_eq!(row.no, 0);
        assert_eq!(row.index_fcb, 1);
        assert!((row.frequency_hz - 62.500).abs() < 1.0e-9);
        assert!((row.z_bark - 0.617).abs() < 1.0e-9);
    }

    #[test]
    fn d2a_last_row_matches_docs_anchor() {
        // Spec anchor (D.2a row 23): no = 23, index = 108, freq =
        // 15000.0, z = 23.923.
        let row = *CRITICAL_BANDS_D2A.last().unwrap();
        assert_eq!(row.no, 23);
        assert_eq!(row.index_fcb, 108);
        assert!((row.frequency_hz - 15000.000).abs() < 1.0e-9);
        assert!((row.z_bark - 23.923).abs() < 1.0e-9);
    }

    #[test]
    fn d2c_first_and_last_rows_match_docs_anchors() {
        // Layer I, 48 kHz: 26 bands, first at (1, 93.750, 0.925),
        // last at (102, 20250.000, 24.597).
        let first = CRITICAL_BANDS_D2C[0];
        assert_eq!((first.no, first.index_fcb), (0, 1));
        assert!((first.frequency_hz - 93.750).abs() < 1.0e-9);
        assert!((first.z_bark - 0.925).abs() < 1.0e-9);
        let last = *CRITICAL_BANDS_D2C.last().unwrap();
        assert_eq!((last.no, last.index_fcb), (25, 102));
        assert!((last.frequency_hz - 20250.000).abs() < 1.0e-9);
        assert!((last.z_bark - 24.597).abs() < 1.0e-9);
    }

    #[test]
    fn d2e_illegible_band_17_preserves_legible_prefix() {
        // Docs marks D.2e row 17 z_bark as `16,11[illegible]`. The
        // legible prefix `16.11` MUST be preserved verbatim; the
        // prose-estimate `16.116` MUST NOT be silently adopted.
        let row = CRITICAL_BANDS_D2E[17];
        assert_eq!(row.no, 17);
        assert_eq!(row.index_fcb, 62);
        assert!((row.frequency_hz - 3273.047).abs() < 1.0e-9);
        // Two-decimal exact: 16.11, not 16.116.
        assert!(
            (row.z_bark - 16.11).abs() < 1.0e-9,
            "D.2e row 17 z_bark = {} (must be the legible prefix 16.11, not the prose estimate 16.116)",
            row.z_bark,
        );
        // And the documented-illegibility marker must agree.
        const { assert!(D2E_BAND_17_BARK_IS_ILLEGIBLE) }
    }

    #[test]
    fn all_tables_monotone_in_frequency_and_bark() {
        // Critical-band boundaries are top-of-band ascending; the
        // spec's tables are monotone in (index_fcb, frequency_hz,
        // z_bark) jointly.
        for (label, table) in [
            ("D.2a", &CRITICAL_BANDS_D2A[..]),
            ("D.2b", &CRITICAL_BANDS_D2B[..]),
            ("D.2c", &CRITICAL_BANDS_D2C[..]),
            ("D.2d", &CRITICAL_BANDS_D2D[..]),
            ("D.2e", &CRITICAL_BANDS_D2E[..]),
            ("D.2f", &CRITICAL_BANDS_D2F[..]),
        ] {
            for w in table.windows(2) {
                let (a, b) = (w[0], w[1]);
                assert!(
                    a.index_fcb < b.index_fcb,
                    "{label}: index_fcb not strictly ascending at bands {} -> {}",
                    a.no,
                    b.no,
                );
                assert!(
                    a.frequency_hz < b.frequency_hz,
                    "{label}: frequency_hz not strictly ascending at bands {} -> {}",
                    a.no,
                    b.no,
                );
                assert!(
                    a.z_bark < b.z_bark,
                    "{label}: z_bark not strictly ascending at bands {} -> {}",
                    a.no,
                    b.no,
                );
            }
        }
    }

    #[test]
    fn annex_d_sampling_rate_round_trips() {
        for &hz in &[32_000u32, 44_100, 48_000] {
            let fs = AnnexDSamplingRate::from_hz(hz).unwrap();
            assert_eq!(fs.as_hz(), hz);
        }
        assert!(AnnexDSamplingRate::from_hz(16_000).is_none());
        assert!(AnnexDSamplingRate::from_hz(22_050).is_none());
        assert!(AnnexDSamplingRate::from_hz(24_000).is_none());
    }

    #[test]
    fn critical_band_boundaries_dispatches_correct_table() {
        use crate::frame::Layer;
        // Six valid combinations dispatch to the six tables.
        let cases = [
            (
                Layer::LayerI,
                AnnexDSamplingRate::Hz32000,
                &CRITICAL_BANDS_D2A[..],
            ),
            (
                Layer::LayerI,
                AnnexDSamplingRate::Hz44100,
                &CRITICAL_BANDS_D2B[..],
            ),
            (
                Layer::LayerI,
                AnnexDSamplingRate::Hz48000,
                &CRITICAL_BANDS_D2C[..],
            ),
            (
                Layer::LayerII,
                AnnexDSamplingRate::Hz32000,
                &CRITICAL_BANDS_D2D[..],
            ),
            (
                Layer::LayerII,
                AnnexDSamplingRate::Hz44100,
                &CRITICAL_BANDS_D2E[..],
            ),
            (
                Layer::LayerII,
                AnnexDSamplingRate::Hz48000,
                &CRITICAL_BANDS_D2F[..],
            ),
        ];
        for (layer, fs, expected) in cases {
            let got = critical_band_boundaries(layer, fs).unwrap();
            // Compare by len and first/last for a cheap structural check.
            assert_eq!(got.len(), expected.len());
            assert_eq!(got.first(), expected.first());
            assert_eq!(got.last(), expected.last());
        }
        // Layer III: returns None for every Fs.
        for fs in [
            AnnexDSamplingRate::Hz32000,
            AnnexDSamplingRate::Hz44100,
            AnnexDSamplingRate::Hz48000,
        ] {
            assert!(critical_band_boundaries(Layer::LayerIII, fs).is_none());
        }
    }

    #[test]
    fn band_of_fft_line_locates_each_band_correctly() {
        // Verify the band-locator against D.2a, where the first band
        // covers line 1, the second covers lines 2..=3, the third
        // covers lines 4..=5, and so on.
        let t = &CRITICAL_BANDS_D2A;
        // 0 is not a valid 1-based index.
        assert_eq!(band_of_fft_line(t, 0), None);
        // Line 1 -> band 0.
        assert_eq!(band_of_fft_line(t, 1), Some(0));
        // Lines 2, 3 -> band 1 (covers 2..=3).
        assert_eq!(band_of_fft_line(t, 2), Some(1));
        assert_eq!(band_of_fft_line(t, 3), Some(1));
        // Lines 4, 5 -> band 2 (covers 4..=5).
        assert_eq!(band_of_fft_line(t, 4), Some(2));
        assert_eq!(band_of_fft_line(t, 5), Some(2));
        // Line 108 (top of last band) -> band 23.
        assert_eq!(band_of_fft_line(t, 108), Some(23));
        // Line 109 (above the audio band) -> None.
        assert_eq!(band_of_fft_line(t, 109), None);
        // Line 999 (way out of range) -> None.
        assert_eq!(band_of_fft_line(t, 999), None);
    }

    #[test]
    fn band_of_fft_line_locates_each_d2e_band_correctly() {
        // D.2e bands cover lines 1, 2, 3, 4..=5, 6..=7, 8..=10,
        // 11..=13, 14..=16, ..., 118..=130. Spot-check a few.
        let t = &CRITICAL_BANDS_D2E;
        assert_eq!(band_of_fft_line(t, 1), Some(0));
        assert_eq!(band_of_fft_line(t, 2), Some(1));
        assert_eq!(band_of_fft_line(t, 3), Some(2));
        assert_eq!(band_of_fft_line(t, 4), Some(3));
        assert_eq!(band_of_fft_line(t, 5), Some(3));
        assert_eq!(band_of_fft_line(t, 6), Some(4));
        assert_eq!(band_of_fft_line(t, 7), Some(4));
        // Top of last band (no = 26): index 130.
        assert_eq!(band_of_fft_line(t, 130), Some(26));
        assert_eq!(band_of_fft_line(t, 131), None);
    }

    #[test]
    fn d2e_band_17_z_bark_is_under_estimate_within_typeset_tolerance() {
        // The legible prefix `16.11` is a strict under-estimate of
        // the true Bark value (the clipped fourth digit cannot drop
        // it below `16.11` because all four digits are visible up to
        // the second-decimal place). The surrounding bands sit at
        // `15.100` (band 16) and `17.079` (band 18); a monotone
        // interpolation must place band 17 inside `(15.100, 17.079)`.
        let b16 = CRITICAL_BANDS_D2E[16];
        let b17 = CRITICAL_BANDS_D2E[17];
        let b18 = CRITICAL_BANDS_D2E[18];
        assert!(
            b16.z_bark < b17.z_bark && b17.z_bark < b18.z_bark,
            "D.2e band 17 z_bark {} must lie strictly between band 16 {} and band 18 {}",
            b17.z_bark,
            b16.z_bark,
            b18.z_bark,
        );
        // And within `0.01` Bark of the documented prefix.
        assert!((b17.z_bark - 16.11).abs() < 1.0e-9);
    }

    #[test]
    fn d2d_first_row_extends_below_d2a_first_row() {
        // Layer II Fs = 32 kHz starts the first critical band at
        // 31.25 Hz (z = 0.309) — the Layer II FFT window is twice as
        // long, so it can resolve a band below Layer I's first band
        // edge of 62.5 Hz. Cross-table sanity check.
        let l1 = CRITICAL_BANDS_D2A[0];
        let l2 = CRITICAL_BANDS_D2D[0];
        assert!(
            l2.frequency_hz < l1.frequency_hz,
            "D.2d first band freq {} should be < D.2a first band freq {}",
            l2.frequency_hz,
            l1.frequency_hz,
        );
    }

    #[test]
    fn d2_band_counts_match_docs_step_4_summary() {
        // From the docs file (clause D.1 prose, "Annex D contents map"):
        //   Layer I  : 23 / 24 / 25  @ 32 / 44.1 / 48 kHz
        //   Layer II : 24 / 26 / 26  @ 32 / 44.1 / 48 kHz
        //
        // The table headers carry the *cell-count* (24/25/26 etc.,
        // numbered no 0..N-1). The docs' "23/24/25" prose counts the
        // band-edge intervals between rows, which is one less than
        // the row count: 24 rows -> 23 intervals. Cross-check that
        // the row count = prose-count + 1 holds for each table.
        assert_eq!(CRITICAL_BANDS_D2A.len(), 23 + 1);
        assert_eq!(CRITICAL_BANDS_D2B.len(), 24 + 1);
        assert_eq!(CRITICAL_BANDS_D2C.len(), 25 + 1);
        assert_eq!(CRITICAL_BANDS_D2D.len(), 24 + 1);
        assert_eq!(CRITICAL_BANDS_D2E.len(), 26 + 1);
        assert_eq!(CRITICAL_BANDS_D2F.len(), 26 + 1);
    }
}
