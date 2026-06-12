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
//! The full Table D.1a–f columns (frequency / critical-band rate /
//! absolute threshold; 102–132 rows per table) are transcribed in
//! this module from the staged 200-DPI PNG renders under
//! `docs/audio/mp3/annex-d-renders/Table-D.1*.png` — see
//! [`MODEL1_THRESHOLD_D1A`]–[`MODEL1_THRESHOLD_D1F`] and the
//! [`model1_threshold_table`] dispatcher further down. Before that
//! transcription landed (r278), the only **textually transcribed**
//! values in `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
//! were the orientation anchors (the first five rows of Table D.1a and
//! the last row, plus the prose-anchored minimum near i = 51, f ≈ 3.375
//! kHz at ≈ −4.97 dB). The [`ltq_db_at_hz`] helper below predates the
//! full tables and still derives `LTq(f)` for arbitrary frequencies via
//! **monotone piecewise-linear interpolation in log-frequency through
//! those textually-transcribed anchors only**; the anchors agree
//! exactly with the corresponding rows of the full tables (pinned by a
//! unit test).
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
/// render. This module records the legible prefix `16.11` (and not
/// the docs file's prose `16,116` estimate, which is explicitly NOT
/// adopted as a verbatim source value).
///
/// **Resolved via Table D.1 (r278):** the D.2 Bark column repeats the
/// `Crit.Band Rate` of the Table D.1 row its `index F&CB` cites, and
/// D.1e row 62 (the row band 17 cites, same frequency 3 273,05 Hz)
/// legibly prints `16,110` — as does D.1b row 38 at the same
/// frequency. The clipped digit is therefore `0`: the stored `16.11`
/// IS the exact value (and the docs prose estimate `16,116` is
/// wrong). The marker stays `true` because the D.2e cell itself
/// remains illegible in its render; the cross-table resolution is
/// pinned by the `table_d1_agrees_with_d2_boundary_rows` unit test.
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
///
/// **Print discrepancies:** rows 17, 20, 24, `z_bark` — the spec
/// prints `17,904` / `20,971` / `24,573` here but `17,905` /
/// `20,972` / `24,574` in the Table D.1b rows they cite (indices 50,
/// 68, 106). See the matching note on [`CRITICAL_BANDS_D2E`].
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
/// (`16,116`) is explicitly NOT adopted. Resolved r278 via Table
/// D.1e row 62 (legibly `16,110` — the clipped digit is `0`, so the
/// stored value is exact). See [`D2E_BAND_17_BARK_IS_ILLEGIBLE`].
///
/// **Print discrepancies:** rows 19, 22, 26, `z_bark`. This table
/// prints `17,904` / `20,971` / `24,573`, but the Table D.1e rows
/// they cite (indices 74, 92, 130) print `17,905` / `20,972` /
/// `24,574`, as do D.1b rows 50, 68, 106 at the same frequencies.
/// D.2b prints the same three lower values at its bands 17, 20, 24,
/// so each side is double-printed and self-consistent — a rounding
/// inconsistency in the printed spec. The verbatim D.2e prints are
/// kept here; the Table D.1 values are what the Step 4 → Bark
/// bridge reads.
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

// =====================================================================
// Annex D Model 1 — Step 4 masker placement + Step 7 nearby-masker
// Bark-window filter (Phase 2 step 46 / r229).
//
// Step 45 (r224) landed the Tables D.2a-f critical-band-boundary
// constants + the `band_of_fft_line` locator that maps an FFT-line
// index to its critical-band index `no`. Step 44 (r219) landed the
// `Masker` data carrier consumed by Steps 6 + 7. The compositional
// gap between those two primitives is "given a critical-band index
// and an SPL in dB, produce a `Masker` placed at the band's `z_bark`
// coordinate." That placement is the verbatim §D.1 Step 4 rule (the
// masker's Bark coordinate is the band's top `z_bark` per Tables
// D.2; the masker's SPL is the band's representative SPL produced
// by Steps 1-3, which remain blocked on the PNG-only Tables D.1 /
// D.3 / D.4 — this round delivers the placement, not the SPL).
//
// Step 7 separately allows a "for a given i the range of j may be
// reduced to maskers within −8…+3 Bark of i" pre-filter. Today's
// `global_masking_threshold_db` already drops out-of-range maskers
// via the `vf` `[-3, 8)` window inside
// `individual_masking_threshold_db`; the spec optimisation is a
// caller-side pre-pass that skips even the `vf` evaluation. We
// expose it as a generic iterator-style predicate
// (`masker_in_step7_window_of_line`) so a caller building a sparse
// per-line `LTg(i)` map can pre-shrink its masker slice once per
// line without re-implementing the Bark arithmetic.
// =====================================================================

/// Place a Model 1 masker at the §D.1 Step 4 critical-band boundary
/// coordinate. Returns a [`Masker`] whose `z_bark` equals
/// `boundaries[band_no as usize].z_bark` (the band's top Bark
/// coordinate per the spec's Tables D.2) and whose `spl_db` is the
/// caller-supplied per-band representative SPL (the value Step 4
/// produces from the Step 1-3 SPL spectrum — `X_tm[z(j)]` for tonal
/// maskers, the energy-sum `X_nm[z(j)]` for non-tonal maskers).
///
/// Returns `None` if `band_no` is out of range for `boundaries`.
///
/// This is a thin composition primitive: it bundles "read the band's
/// Bark coordinate from the Tables D.2 slice you already have" plus
/// "wrap the SPL into a typed [`Masker`]" into one inlined call so
/// the caller (a future Steps 1-5 driver, blocked on Tables D.1 /
/// D.3 / D.4 PNG-only transcription) never has to know about the
/// `CriticalBandBoundary.z_bark` field directly. The placement rule
/// is verbatim spec: the masker sits at the band's top Bark
/// coordinate.
#[inline]
#[must_use]
pub fn masker_at_band(
    boundaries: &[CriticalBandBoundary],
    band_no: u16,
    kind: MaskerKind,
    spl_db: f64,
) -> Option<Masker> {
    let row = boundaries.get(band_no as usize)?;
    Some(Masker {
        kind,
        z_bark: row.z_bark,
        spl_db,
    })
}

/// Lower bound of the §D.1 Step 7 "nearby-masker" Bark window for a
/// target line at `z(i)`. The spec text reads:
///
/// > For a given i the range of j may be reduced to maskers within
/// > −8…+3 Bark of i.
///
/// Equivalently: a masker at `z(j)` is "near" line `i` iff
/// `z(j) ∈ (z(i) - 8, z(i) + 3]`. This constant is the open-bottom
/// `-8` (the lowest `z(j) - z(i)` displacement, exclusive — a masker
/// 8 Bark below the line is the edge of the §D.1 Step 6 `vf`
/// piecewise function's `dz < 8` upper branch, which is right-open).
pub const STEP7_NEARBY_MASKER_DZ_LO_FROM_LINE: f64 = -8.0;

/// Upper bound (inclusive) of the §D.1 Step 7 "nearby-masker" Bark
/// window for a target line at `z(i)`: a masker at `z(j)` is "near"
/// line `i` iff `z(j) ∈ (z(i) - 8, z(i) + 3]`. The `+3` is the
/// highest `z(j) - z(i)` displacement (inclusive — the spec's
/// `vf` lower branch is left-closed at `dz = -3`, so a masker
/// exactly 3 Bark above the line is still in range).
pub const STEP7_NEARBY_MASKER_DZ_HI_FROM_LINE: f64 = 3.0;

/// Predicate for the §D.1 Step 7 "nearby-masker" optimisation:
/// returns `true` iff the masker at `z(j) = masker.z_bark` is inside
/// the spec's `(z(i) - 8, z(i) + 3]` Bark window of the target line
/// at `z_i_bark`.
///
/// Equivalent to "the masker would contribute a finite `vf` term to
/// the global masking threshold at this line" — the §D.1 Step 6 `vf`
/// is defined for `dz = z(i) - z(j) ∈ [-3, 8)`, which is exactly
/// `z(j) ∈ (z(i) - 8, z(i) + 3]`. A caller can use this predicate
/// to pre-shrink the masker slice fed to
/// [`global_masking_threshold_db`] once per line, avoiding the
/// `individual_masking_threshold_db` call (and its branch on `vf`)
/// for every out-of-range masker.
///
/// The two bounds are sourced from the spec text "maskers within
/// −8…+3 Bark of i" (the cited range in §D.1 Step 7) intersected
/// with the §D.1 Step 6 `vf` piecewise function's half-open
/// `[-3, 8)` `dz` window. Pairing the two yields the open-low,
/// closed-high form encoded above; this exactly matches the set of
/// maskers for which `individual_masking_threshold_db` returns
/// `Some`.
#[inline]
#[must_use]
pub fn masker_in_step7_window_of_line(masker: &Masker, z_i_bark: f64) -> bool {
    let dz_from_line = masker.z_bark - z_i_bark;
    dz_from_line > STEP7_NEARBY_MASKER_DZ_LO_FROM_LINE
        && dz_from_line <= STEP7_NEARBY_MASKER_DZ_HI_FROM_LINE
}

// =====================================================================
// Annex D Model 1 — §D.1 Step 5 decimation primitives (Phase 2 step 47).
//
// Step 5 of Model 1 is "decimation of tonal and non-tonal masking
// components" — a two-part sieve that runs between Step 4's masker
// placement (already wired in r229) and Step 6's individual
// masking-threshold calculation (already wired in r219). The spec text
// (PDF page 118 / printed 112, clause D.1 "Step 5") defines two
// sub-procedures:
//
//   (a) Threshold-in-quiet screening — a masker is kept only if its
//       SPL is at or above the threshold-in-quiet LTq at the masker's
//       own frequency. The spec equations are
//
//           X_tm(k) >= LTq(k)         keep tonal masker
//           X_nm(k) >= LTq(k)         keep non-tonal masker
//
//       (verbatim from the spec text). Both masker classes use the
//       same comparison rule.
//
//   (b) Tonal cluster decimation — tonal maskers within a 0.5-Bark
//       sliding window collapse to the loudest member of the cluster
//       (verbatim spec text: "Decimation of two or more tonal
//       components within a distance of less than 0,5 Bark: Keep the
//       component with the highest power, and remove the smaller
//       component(s) from the list of tonal components. For this
//       operation, a sliding window in the critical band domain is
//       used with a width of 0,5 Bark"). The spec applies this only
//       to tonal maskers — non-tonal maskers are already at most one
//       per critical band by Step 4(c) and are passed through
//       unchanged.
//
// Both primitives operate on the typed `&[Masker]` slice produced by
// `masker_at_band` (r229). The LTq value for Step 5(a) is supplied
// by the caller in dB — typically sourced from `ltq_db_at_hz` after
// converting the masker's `z_bark` to its corresponding frequency
// via the Tables D.1 (Frequency / Bark / LTq) mapping the caller
// holds. The 0.5-Bark window constant for Step 5(b) is exposed as a
// named `pub const` (`STEP5_TONAL_DECIMATION_WINDOW_BARK`) for direct
// citation back to the spec text.
//
// The two primitives compose left-to-right (a → b) into a single
// Step 5 sieve; callers that already pre-filter their masker slice
// (e.g. by feeding only above-LTq lines from the Step 4 tonal /
// non-tonal selection) may skip (a) and run (b) directly on the
// tonal slice. Both primitives preserve the caller's masker order
// for in-cluster ties (the first-encountered loudest masker wins),
// which keeps the output stable across repeated calls on the same
// input.
// =====================================================================

/// Spec width of the §D.1 Step 5(b) tonal-decimation sliding window,
/// in Bark units (verbatim spec text: "a sliding window in the
/// critical band domain is used with a width of 0,5 Bark").
///
/// Two tonal maskers separated by **strictly less than** this width
/// are in the same cluster and collapse to the loudest member; the
/// spec phrasing is "less than 0,5 Bark", which this constant
/// captures as the strict upper bound.
pub const STEP5_TONAL_DECIMATION_WINDOW_BARK: f64 = 0.5;

/// §D.1 Step 5(a) threshold-in-quiet screening predicate. Returns
/// `true` iff the masker survives the spec's screening rule
///
/// ```text
/// X_tm(k) >= LTq(k)        tonal masker kept
/// X_nm(k) >= LTq(k)        non-tonal masker kept
/// ```
///
/// `ltq_db` is the threshold-in-quiet **at the masker's own
/// frequency** in dB — sourced by the caller from `ltq_db_at_hz`
/// after converting `masker.z_bark` to a Hertz value via the
/// caller's Tables D.1 (Frequency / Bark / LTq) mapping. The
/// comparison is identical for tonal and non-tonal maskers; the
/// kind tag is preserved on the surviving maskers because Step 6
/// dispatches the masking-index `av` per kind.
#[inline]
#[must_use]
pub fn masker_above_threshold_in_quiet(masker: &Masker, ltq_db: f64) -> bool {
    masker.spl_db >= ltq_db
}

/// §D.1 Step 5(b) tonal-cluster decimation. Returns a new `Vec`
/// containing every input masker with **tonal** maskers in any
/// strictly-less-than-`0.5`-Bark cluster collapsed to the loudest
/// member of the cluster. Non-tonal maskers pass through unchanged
/// (the spec applies this sub-step only to tonal maskers — Step 4(c)
/// already produces at most one non-tonal masker per critical band).
///
/// The output preserves the caller's input order: surviving maskers
/// appear in their original positions in the slice. Tied SPLs in a
/// cluster resolve to the first-encountered masker (input-order
/// stable), so repeated calls on the same input produce identical
/// output.
///
/// Algorithm: a sliding window over the **tonal** subset sorted by
/// `z_bark` accumulates a cluster while consecutive tonal maskers
/// are within `STEP5_TONAL_DECIMATION_WINDOW_BARK` of each other;
/// when the window closes, the cluster's loudest member is kept and
/// the rest are dropped. The non-tonal subset is interleaved back
/// into the output in original-slice order.
///
/// The spec text reads "less than 0,5 Bark", which this
/// implementation encodes as a strict `<` comparison on the Bark
/// difference between consecutive sorted tonal maskers. A pair of
/// tonal maskers separated by exactly `0.5` Bark is therefore
/// **not** in the same cluster and both survive.
#[must_use]
pub fn decimate_tonal_within_half_bark(maskers: &[Masker]) -> Vec<Masker> {
    // Fast path: nothing to do for the empty / singleton input.
    if maskers.len() < 2 {
        return maskers.to_vec();
    }
    // Collect the tonal subset with its original-slice positions so
    // the output ordering survives the sort-and-cluster pass.
    let mut tonal_indices: Vec<usize> = maskers
        .iter()
        .enumerate()
        .filter_map(|(i, m)| match m.kind {
            MaskerKind::Tonal => Some(i),
            MaskerKind::NonTonal => None,
        })
        .collect();
    // Sort tonal indices by z_bark ascending. Ties on z_bark fall
    // back to input order via the original-index secondary key to
    // keep cluster membership deterministic on coincident maskers.
    tonal_indices.sort_by(|&a, &b| {
        maskers[a]
            .z_bark
            .partial_cmp(&maskers[b].z_bark)
            .unwrap_or(core::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });
    // Walk the sorted tonal list, accumulating clusters whose
    // consecutive Bark gap is strictly below the window width.
    // For each cluster, mark all but the loudest member as dropped.
    let mut dropped = vec![false; maskers.len()];
    let mut cluster_start = 0usize; // index into tonal_indices
    while cluster_start < tonal_indices.len() {
        let mut cluster_end = cluster_start + 1; // exclusive
        while cluster_end < tonal_indices.len() {
            let prev = maskers[tonal_indices[cluster_end - 1]].z_bark;
            let cur = maskers[tonal_indices[cluster_end]].z_bark;
            if cur - prev < STEP5_TONAL_DECIMATION_WINDOW_BARK {
                cluster_end += 1;
            } else {
                break;
            }
        }
        // Cluster spans `tonal_indices[cluster_start..cluster_end]`.
        // Keep the loudest member; first-encountered wins on ties so
        // the output is input-order stable.
        let mut keep_pos = cluster_start;
        let mut keep_spl = maskers[tonal_indices[cluster_start]].spl_db;
        for cur in (cluster_start + 1)..cluster_end {
            let spl = maskers[tonal_indices[cur]].spl_db;
            if spl > keep_spl {
                keep_spl = spl;
                keep_pos = cur;
            }
        }
        for cur in cluster_start..cluster_end {
            if cur != keep_pos {
                dropped[tonal_indices[cur]] = true;
            }
        }
        cluster_start = cluster_end;
    }
    // Emit the surviving maskers in original-slice order.
    maskers
        .iter()
        .enumerate()
        .filter_map(|(i, m)| if dropped[i] { None } else { Some(*m) })
        .collect()
}

// =====================================================================
// Annex D Model 2 — §C.1.5.3.2.1 Layer III spreading function (Phase 2
// step 48).
//
// Spec context (clause C.1.5.3.2.1 — "Layer III modifies the
// spreading function", as transcribed in
// docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md from the
// staged ISO/IEC 11172-3:1993 PDF):
//
//   if j >= i : tmpy = 3,0 * (j - i)
//   else      : tmpy = 1,5 * (j - i)
//   Only spreading-function values greater than 1e-6 are used; all
//   others set to zero.
//
// Here `i` is the partition index of the *masker* (the partition
// emitting energy) and `j` is the partition index of the *masked*
// (the partition receiving spread energy). The two branches encode
// an asymmetric Bark-domain spread: a steeper roll-off when
// spreading upward (`j > i`, the high-frequency direction the ear
// masks well) and a gentler roll-off when spreading downward
// (`j < i`, the low-frequency direction the ear masks less). At
// `j == i` the two branches agree (`tmpy = 0`), so the linear
// spreading factor is `10^0 = 1` on the diagonal.
//
// The spec computes `tmpy` in dB and then converts it into a linear
// energy-domain factor `sprdngf = 10^(tmpy/10)`. The final clamp
// "values greater than 1e-6 are used; all others set to zero" is
// stated as a threshold on the linear factor, not on `tmpy`. In
// linear terms `10^(tmpy/10) > 1.0e-6` is equivalent to
// `tmpy > -60 dB` (since `10^(-60/10) = 1.0e-6` exactly); the spec's
// linear-domain phrasing is what this primitive implements (the
// caller may inspect the unclamped dB value via the separate `_db`
// accessor).
// =====================================================================

/// Spec clamp for the Model 2 Layer III spreading-function factor
/// (verbatim spec text: "Only spreading-function values greater than
/// 1e-6 are used; all others set to zero").
///
/// The threshold is a hard lower bound on the *linear* factor
/// `10^(tmpy/10)`; spreading-function values at or below this
/// threshold are clamped to exact zero by the spec procedure.
pub const MODEL2_LAYER3_SPREAD_LINEAR_MIN: f64 = 1.0e-6;

/// §C.1.5.3.2.1 Layer III spreading-function dB value `tmpy(i, j)` —
/// the per-partition asymmetric spread in dB before the linear
/// conversion. `i` is the *masker* partition index, `j` is the
/// *masked* partition index, and both are in Model 2's partition
/// space (the `Index` column of Tables D.3a–c).
///
/// Spec branches (verbatim text):
///
/// ```text
/// j >= i : tmpy = 3.0 * (j - i)
/// j <  i : tmpy = 1.5 * (j - i)
/// ```
///
/// At `i == j` the value is exactly `0.0`. For `j > i` (upward
/// spread, masked partition above masker) `tmpy` is positive and
/// the linear factor grows above unity; for `j < i` (downward
/// spread) `tmpy` is negative and the linear factor falls below
/// unity, eventually triggering the spec's
/// `MODEL2_LAYER3_SPREAD_LINEAR_MIN` clamp.
#[inline]
#[must_use]
pub fn model2_layer3_spread_db(i: i32, j: i32) -> f64 {
    let dj = f64::from(j - i);
    if j >= i {
        // Upward / on-diagonal branch. At j == i this yields 0.0
        // (unity spread on the diagonal).
        3.0 * dj
    } else {
        // Downward branch: `dj` is negative, so the result is
        // negative.
        1.5 * dj
    }
}

/// §C.1.5.3.2.1 Layer III spreading-function **linear** factor
/// `sprdngf(i, j) = 10^(tmpy(i, j) / 10)`, with the spec's
/// "greater-than-`1.0e-6` clamp" applied: factors at or below
/// `MODEL2_LAYER3_SPREAD_LINEAR_MIN` collapse to exact `0.0`.
///
/// The clamped factor is suitable as a Model 2 spreading-matrix
/// entry: zero-valued cells contribute no energy to the partition's
/// masking sum and can be dropped from sparse-matrix multiplies.
///
/// At `i == j` the factor is exactly `1.0`; the spec's `tmpy = 0`
/// gives `10^0 = 1`, which is well above the clamp threshold. For
/// `j > i` the factor grows above unity (the upward branch has a
/// positive `tmpy`); the clamp only takes effect on the downward
/// branch when `j` is sufficiently far below `i` to drive `tmpy`
/// below `-60` dB (i.e. `j - i <= -40` for the `1.5 * (j - i)`
/// branch).
#[inline]
#[must_use]
pub fn model2_layer3_spread_linear(i: i32, j: i32) -> f64 {
    let tmpy_db = model2_layer3_spread_db(i, j);
    let linear = (10.0_f64).powf(tmpy_db / 10.0);
    if linear > MODEL2_LAYER3_SPREAD_LINEAR_MIN {
        linear
    } else {
        0.0
    }
}

// =====================================================================
// Annex D Model 2 — §D.2.3 "The spreading function" + §D.2.4 steps f)
// and g) (Phase 2 step 81 / r279).
//
// Spec context (ISO/IEC 11172-3:1993 Annex D clause D.2.3, printed
// p.129 / PDF p.135, as transcribed in
// docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md — including
// the corrected `10^((x + tmpy)/10)` envelope; an earlier docs-file
// revision dropped the `x` term):
//
//   tmpx = 1,05 * (j - i)
//   x    = 8 * minimum( (tmpx - 0,5)^2 - 2*(tmpx - 0,5), 0 )
//   tmpy = 15,811389 + 7,5*(tmpx + 0,474)
//          - 17,5*(1,0 + (tmpx + 0,474)^2)^0,5
//   if (tmpy < -100) then sprdngf(i,j) = 0
//   else sprdngf(i,j) = 10^((x + tmpy)/10)
//
// Per the D.2.3 prose, `i` is the **Bark value** of the signal being
// spread and `j` the **Bark value** of the band being spread into —
// real-valued Bark coordinates (the `bval` column of Tables D.3a–c),
// NOT integer partition indices. (The §C.1.5.3.2.1 Layer III variant
// above replaces `tmpy` with a piecewise-linear function of the
// partition-index difference; this section is the *base* Model 2
// function used by the §D.2.4 step f) convolution for Layers I/II.)
//
// Structure of the curve: `tmpy` is a hyperbola-like asymmetric
// envelope whose constant `15,811389 = 17,5*sqrt(1 + 0,474^2)
// - 7,5*0,474` makes it exactly 0 dB at `tmpx = 0` (the diagonal),
// falling off at ≈ -10 dB/Bark-unit-of-tmpx upward (`tmpx → +∞`:
// `7,5v - 17,5v`) and ≈ -25 dB downward (`tmpx → -∞`:
// `7,5v + 17,5|v|`). The parabolic correction `x` is non-zero only
// for `0,5 < tmpx < 2,5` (where `(tmpx-0,5)^2 - 2*(tmpx-0,5) < 0`),
// carving up to -8 dB (at `tmpx = 1,5`) out of the near-upward skirt.
// The `tmpy < -100` cutoff is applied to `tmpy` ALONE (before adding
// `x`), exactly as printed.
// =====================================================================

/// §D.2.3 cutoff: when `tmpy` falls below this value (in dB) the
/// spreading-function value is forced to exact zero (spec verbatim:
/// "if (tmpy < -100) then sprdngf(i,j) = 0").
///
/// The comparison is strict (`<`) and tests `tmpy` alone — the
/// parabolic `x` term does not participate in the cutoff decision.
pub const MODEL2_SPRDNGF_TMPY_CUTOFF_DB: f64 = -100.0;

/// §D.2.3 temporary variable `tmpx = 1,05 * (j - i)` — the scaled
/// Bark distance from the masker (`i_bark`, "the Bark value of the
/// signal being spread") to the destination band (`j_bark`, "the
/// Bark value of the band being spread into").
///
/// Positive for upward spread (into higher Bark), negative for
/// downward spread.
#[inline]
#[must_use]
pub fn model2_sprdngf_tmpx(i_bark: f64, j_bark: f64) -> f64 {
    1.05 * (j_bark - i_bark)
}

/// §D.2.3 parabolic correction term
/// `x = 8 * minimum((tmpx - 0,5)^2 - 2*(tmpx - 0,5), 0)` in dB.
///
/// The spec defines "minimum(a,b)" as "a function returning the more
/// negative of a or b", so `x` is never positive. Writing
/// `u = tmpx - 0,5`, the inner expression `u^2 - 2u` is negative
/// exactly for `0 < u < 2`, i.e. the correction is active only on
/// the near-upward skirt `0,5 < tmpx < 2,5`, reaching its most
/// negative value `8 * (1 - 2) = -8` dB at `tmpx = 1,5`.
#[inline]
#[must_use]
pub fn model2_sprdngf_x_db(tmpx: f64) -> f64 {
    let u = tmpx - 0.5;
    8.0 * (u * u - 2.0 * u).min(0.0)
}

/// §D.2.3 envelope term `tmpy = 15,811389 + 7,5*(tmpx + 0,474)
/// - 17,5*(1,0 + (tmpx + 0,474)^2)^0,5` in dB.
///
/// Exactly `0` dB at `tmpx = 0` (the printed constant `15,811389`
/// equals `17,5*sqrt(1 + 0,474^2) - 7,5*0,474` to the printed
/// precision), asymptotically `-10 dB` per unit of `tmpx` upward and
/// `-25 dB` per unit downward — the classic asymmetric Bark-domain
/// masking skirt (upward masking reaches much farther than downward).
#[inline]
#[must_use]
pub fn model2_sprdngf_tmpy_db(tmpx: f64) -> f64 {
    let v = tmpx + 0.474;
    15.811_389 + 7.5 * v - 17.5 * (1.0 + v * v).sqrt()
}

/// §D.2.3 spreading function `sprdngf(i,j)` — the linear
/// energy-domain factor by which energy in a partition with median
/// Bark value `i_bark` spreads into a partition with median Bark
/// value `j_bark`.
///
/// Spec procedure (verbatim order): compute `tmpx`, `x`, `tmpy`;
/// `if (tmpy < -100) then sprdngf(i,j) = 0 else
/// sprdngf(i,j) = 10^((x + tmpy)/10)`.
///
/// On the diagonal (`i_bark == j_bark`) the value is `1.0` to within
/// the rounding of the printed `15,811389` constant (≈ `5e-7`
/// relative). The cutoff zeroes the factor beyond ≈ 4,8 Bark
/// downward and ≈ 10,5 Bark upward.
#[inline]
#[must_use]
pub fn model2_sprdngf(i_bark: f64, j_bark: f64) -> f64 {
    let tmpx = model2_sprdngf_tmpx(i_bark, j_bark);
    let tmpy = model2_sprdngf_tmpy_db(tmpx);
    if tmpy < MODEL2_SPRDNGF_TMPY_CUTOFF_DB {
        0.0
    } else {
        let x = model2_sprdngf_x_db(tmpx);
        (10.0_f64).powf((x + tmpy) / 10.0)
    }
}

// =====================================================================
// Annex D Model 2 — §D.2.1 inputs + §D.2.4 steps a) through e)
// (Phase 2 step 84 / r282): FFT analysis window + complex spectrum,
// magnitude/phase prediction, unpredictability measure, and the
// partition energy/unpredictability sums that feed the step f)
// spreading convolution already landed above.
//
// Spec context (ISO/IEC 11172-3:1993 Annex D, printed pp.128–130 /
// PDF pp.134–136 of docs/audio/mp3/ISO_IEC_11172-3-MP3-1993.pdf):
//
// §D.2.1 "General" — the threshold generation process has three
// inputs: a) the shift length iblen, "where 384<iblen<640", constant
// over any particular application; b) "the newest iblen samples of
// the signal"; c) the sampling rate. One output: "a set of
// Signal-to-Masking Ratios, SMR_n". And: "Before running the model
// initially, the array used to hold the preceding FFT source data
// window and the arrays used to hold r and f should be zeroed to
// provide a known starting point."
//
// §D.2.2 "Comments on notation" — ω indexes the FFT spectral line
// domain: "An index of 1 corresponds to the DC term and an index of
// 513 corresponds to the spectral line at the Nyquist frequency."
//
// §D.2.4 steps (verbatim formulas):
//
//   a) Reconstruct 1 024 samples of the input signal. "iblen new
//      samples are made available at every call to the threshold
//      generator. The threshold generator must store 1 024-iblen
//      samples, and concatenate those samples to accurately
//      reconstruct 1 024 consecutive samples of the input signal,
//      s_i, where i represents the index, 1 <= i <= 1 024 of the
//      current input stream."
//
//   b) Calculate the complex spectrum of the input signal.
//      "First, s_i is windowed by a 1 024 point Hann window, i.e.
//         sw_i = s_i * (0,5 - 0,5*cos(2π(i - 0,5)/1024)).
//      Second, a standard forward FFT of sw_i is calculated.
//      Third, the polar representation of the transform is
//      calculated. r_ω and f_ω represent the magnitude and phase
//      components of the transformed sw_i, respectively."
//
//   c) Calculate a predicted r and f:
//         r̂_ω = 2,0·r_ω(t-1) - r_ω(t-2)
//         f̂_ω = 2,0·f_ω(t-1) - f_ω(t-2)
//      "where t represents the current block number, t-1 indexes the
//      previous block's data, and t-2 indexes the data from the
//      threshold calculation block before that."
//
//   d) Calculate the unpredictability measure c_ω:
//         c_ω = ((r_ω·cos f_ω - r̂_ω·cos f̂_ω)²
//               + (r_ω·sin f_ω - r̂_ω·sin f̂_ω)²)^0,5
//               / (r_ω + abs(r̂_ω))
//
//   e) Calculate the energy and the weighted unpredictability in the
//      threshold calculation partitions:
//         e_b = Σ_{ω=ωlow_b}^{ωhigh_b} r_ω²
//         c_b = Σ_{ω=ωlow_b}^{ωhigh_b} r_ω²·c_ω
// =====================================================================

/// §D.2.4 step a) analysis-window length — "Reconstruct 1 024 samples
/// of the input signal" (and the §D.2.2 line domain ω ∈ 1..=513 is
/// exactly this transform's half-spectrum).
pub const MODEL2_FFT_LEN: usize = 1024;

/// Number of FFT spectral lines in the Model 2 line domain — §D.2.2
/// verbatim: "An index of 1 corresponds to the DC term and an index
/// of 513 corresponds to the spectral line at the Nyquist frequency."
pub const MODEL2_FFT_LINES: usize = 513;

/// §D.2.1 input a) shift-length constraint — verbatim "384<iblen<640"
/// (both bounds strict). `iblen` outside this range requires a
/// different window and/or transform length per the §D.2.1 prose
/// ("Use a different length transform … or … a substantially shorter
/// Hann window"); the standard table set assumes the in-range case.
#[inline]
#[must_use]
pub const fn model2_iblen_in_range(iblen: usize) -> bool {
    384 < iblen && iblen < 640
}

/// §D.2.4 step d) default unpredictability above the partial-
/// calculation limit — verbatim: "By sacrificing performance, this
/// measure can be calculated on only a lower portion of the frequency
/// lines. … The c_ω values above this limit should be set to 0,3."
///
/// (The same prose bounds the limit: "Calculations should be done
/// from DC to at least 3 kHz and preferably to 7kHz. An upper limit
/// of less than 5,5kHz may considerably reduce performance … Best
/// results will be obtained by calculating c_ω up to 20 kHz.")
pub const MODEL2_CW_ABOVE_LIMIT: f64 = 0.3;

/// §D.2.4 step a) — reconstruct the 1 024-sample analysis window from
/// the preceding window and the `iblen` newest input samples.
///
/// `prev_window` is the previous call's 1 024 reconstructed samples
/// (all-zero before the first call, per the §D.2.1 "should be zeroed"
/// initialization); `new_samples` carries the `iblen` newest samples.
/// The output concatenates the most recent `1 024 - iblen` samples of
/// the previous window with the new block, yielding "1 024
/// consecutive samples of the input signal".
///
/// Returns `None` when `prev_window.len() != 1 024` or
/// `new_samples` is empty or longer than 1 024. The §D.2.1
/// `384<iblen<640` constraint is the *caller's* application contract
/// (checkable via [`model2_iblen_in_range`]); the reconstruction
/// itself is well-defined for any `1 <= iblen <= 1 024` and is not
/// artificially narrowed here.
#[must_use]
pub fn model2_step_a_reconstruct(prev_window: &[f64], new_samples: &[f64]) -> Option<Vec<f64>> {
    let iblen = new_samples.len();
    if prev_window.len() != MODEL2_FFT_LEN || iblen == 0 || iblen > MODEL2_FFT_LEN {
        return None;
    }
    let mut out = Vec::with_capacity(MODEL2_FFT_LEN);
    out.extend_from_slice(&prev_window[iblen..]);
    out.extend_from_slice(new_samples);
    Some(out)
}

/// §D.2.4 step b) Hann window coefficient for the **1-based** sample
/// index `i` — verbatim `0,5 - 0,5*cos(2π(i - 0,5)/1024)`.
///
/// Returns `None` outside the spec domain `1 <= i <= 1 024`. The
/// half-sample offset `(i - 0,5)` makes the window exactly symmetric
/// about the block centre (`w(i) = w(1025 - i)`) with no zero-valued
/// endpoint sample. Unlike the Model 1 Step 1 window
/// ([`model1_hann_window`]) there is **no** `sqrt(8/3)` power
/// prefactor — the printed step b) formula is the bare raised cosine,
/// and the implementation-dependent normalization is absorbed
/// downstream by the step l) [`model2_absthr_energy`] conversion
/// ("after considering the FFT normalization actually used").
#[must_use]
pub fn model2_hann_window(i: usize) -> Option<f64> {
    if i == 0 || i > MODEL2_FFT_LEN {
        return None;
    }
    let angle = 2.0 * core::f64::consts::PI * (i as f64 - 0.5) / (MODEL2_FFT_LEN as f64);
    Some(0.5 - 0.5 * angle.cos())
}

/// Polar half-spectrum of one Model 2 analysis block — the step b)
/// `r_ω` (magnitude) and `f_ω` (phase, radians) components for the
/// §D.2.2 line domain ω ∈ 1..=513, with slice index `ω - 1` holding
/// line `ω`. Also the carrier for the step c) predicted spectrum
/// (`r̂_ω` / `f̂_ω`).
#[derive(Debug, Clone, PartialEq)]
pub struct Model2Polar {
    /// Magnitude per FFT line (`r_ω`, index `ω - 1`).
    pub r: Vec<f64>,
    /// Phase per FFT line in radians (`f_ω`, index `ω - 1`).
    pub f: Vec<f64>,
}

impl Model2Polar {
    /// All-zero polar spectrum over `MODEL2_FFT_LINES` lines — the
    /// §D.2.1 initial state ("the arrays used to hold r and f should
    /// be zeroed to provide a known starting point").
    #[must_use]
    pub fn zeroed() -> Self {
        Self {
            r: vec![0.0; MODEL2_FFT_LINES],
            f: vec![0.0; MODEL2_FFT_LINES],
        }
    }
}

/// §D.2.4 step b) — complex spectrum of one reconstructed analysis
/// block, in polar representation.
///
/// Applies the three printed sub-steps in order: the 1 024-point Hann
/// window ([`model2_hann_window`]; `sw_i = s_i · w(i)`), "a standard
/// forward FFT of sw_i" (unnormalized forward transform
/// `Σ_l sw_l · e^(-jωl2π/N)`), and the polar conversion
/// (`r_ω = |X_ω|`, `f_ω = arg X_ω`). The output covers the §D.2.2
/// line domain ω ∈ 1..=513 (DC through Nyquist).
///
/// The spec does not prescribe an FFT normalization; this
/// implementation applies none, and the step l) absolute-threshold
/// conversion ([`model2_absthr_energy`]) takes the resulting
/// reference level as its explicit `half_lsb_sine_level_db`
/// parameter, per the printed "after considering the FFT
/// normalization actually used".
///
/// Returns `None` unless `s.len() == 1 024` — the step a)
/// reconstruction ([`model2_step_a_reconstruct`]) is the only
/// supported producer; no padding or truncation is invented.
#[must_use]
pub fn model2_step_b_spectrum(s: &[f64]) -> Option<Model2Polar> {
    if s.len() != MODEL2_FFT_LEN {
        return None;
    }
    let mut re: Vec<f64> = s
        .iter()
        .enumerate()
        .map(|(l, &sample)| {
            // 0-based slice index `l` is the spec's 1-based `i = l + 1`,
            // always in the window's domain, so the accessor cannot
            // return `None`.
            model2_hann_window(l + 1).unwrap_or(0.0) * sample
        })
        .collect();
    let mut im = vec![0.0_f64; MODEL2_FFT_LEN];
    fft_in_place(&mut re, &mut im);
    let mut r = Vec::with_capacity(MODEL2_FFT_LINES);
    let mut f = Vec::with_capacity(MODEL2_FFT_LINES);
    for k in 0..MODEL2_FFT_LINES {
        r.push(re[k].hypot(im[k]));
        f.push(im[k].atan2(re[k]));
    }
    Some(Model2Polar { r, f })
}

/// §D.2.4 step c) — linear prediction of one magnitude or phase
/// component from the preceding two threshold-calculation blocks:
/// `x̂_ω = 2,0·x_ω(t-1) - x_ω(t-2)`.
#[inline]
#[must_use]
pub fn model2_step_c_predict(prev: f64, prev2: f64) -> f64 {
    2.0 * prev - prev2
}

/// §D.2.4 step c) over the polar half-spectrum — the predicted
/// magnitude `r̂_ω` and phase `f̂_ω` from the previous block (`t-1`)
/// and the block before that (`t-2`).
///
/// Returns `None` when the four input slices do not all share one
/// length. Phase prediction operates on the principal-value phases
/// the step b) polar conversion produces; since the step d)
/// unpredictability measure only consumes `f̂_ω` through `cos`/`sin`,
/// the `2·f(t-1) - f(t-2)` combination is invariant (mod 2π) to the
/// principal-value branch cuts.
#[must_use]
pub fn model2_step_c_predict_polar(prev: &Model2Polar, prev2: &Model2Polar) -> Option<Model2Polar> {
    if prev.r.len() != prev.f.len()
        || prev2.r.len() != prev2.f.len()
        || prev.r.len() != prev2.r.len()
    {
        return None;
    }
    let predict = |a: &[f64], b: &[f64]| -> Vec<f64> {
        a.iter()
            .zip(b.iter())
            .map(|(&p1, &p2)| model2_step_c_predict(p1, p2))
            .collect()
    };
    Some(Model2Polar {
        r: predict(&prev.r, &prev2.r),
        f: predict(&prev.f, &prev2.f),
    })
}

/// §D.2.4 step d) — unpredictability measure for one FFT line:
///
/// ```text
/// c_ω = ((r_ω·cos f_ω - r̂_ω·cos f̂_ω)² + (r_ω·sin f_ω - r̂_ω·sin f̂_ω)²)^0,5
///       / (r_ω + abs(r̂_ω))
/// ```
///
/// The numerator is the Euclidean distance between the actual and
/// predicted complex spectral values; the denominator normalizes by
/// the magnitude sum, bounding `c_ω` to `[0, 1]` over the spec domain
/// (`r_ω ≥ 0`): a perfectly predicted line gives 0, a zero-magnitude
/// prediction against a live line (or opposite-phase prediction of
/// equal magnitude) gives 1. The spec leaves the all-silent
/// `0/0` case (`r_ω = r̂_ω = 0`) undefined; this implementation
/// returns `0.0` there (a silent line predicted silent is perfectly
/// predictable), keeping the downstream step e)/g) chain finite.
#[inline]
#[must_use]
pub fn model2_step_d_cw(r: f64, f: f64, r_hat: f64, f_hat: f64) -> f64 {
    let den = r + r_hat.abs();
    if den == 0.0 {
        return 0.0;
    }
    let dre = r * f.cos() - r_hat * f_hat.cos();
    let dim = r * f.sin() - r_hat * f_hat.sin();
    dre.hypot(dim) / den
}

/// §D.2.4 step d) over the line domain — the unpredictability vector
/// `c_ω` for ω ∈ 1..=lines, with the spec's optional
/// partial-calculation convention.
///
/// `cur` is the step b) spectrum of the current block; `predicted`
/// the step c) prediction. With `compute_through_line = None` every
/// line is computed exactly; with `Some(limit)` only lines
/// `ω <= limit` are computed and every line above the limit is set to
/// [`MODEL2_CW_ABOVE_LIMIT`] (verbatim: "this measure can be
/// calculated on only a lower portion of the frequency lines … The
/// c_ω values above this limit should be set to 0,3"). The prose
/// bounds the sensible limit in frequency terms — at least 3 kHz,
/// preferably 7 kHz and up to 20 kHz — which the caller translates to
/// a line index for its sampling rate.
///
/// Returns `None` when the two spectra do not share one line count
/// (or either is internally inconsistent). Output index `ω - 1` holds
/// line `ω`.
#[must_use]
pub fn model2_step_d_cw_lines(
    cur: &Model2Polar,
    predicted: &Model2Polar,
    compute_through_line: Option<usize>,
) -> Option<Vec<f64>> {
    if cur.r.len() != cur.f.len()
        || predicted.r.len() != predicted.f.len()
        || cur.r.len() != predicted.r.len()
    {
        return None;
    }
    Some(
        (0..cur.r.len())
            .map(|idx| {
                let line = idx + 1;
                match compute_through_line {
                    Some(limit) if line > limit => MODEL2_CW_ABOVE_LIMIT,
                    _ => {
                        model2_step_d_cw(cur.r[idx], cur.f[idx], predicted.r[idx], predicted.f[idx])
                    }
                }
            })
            .collect(),
    )
}

/// §D.2.4 step e) — energy per threshold calculation partition:
/// `e_b = Σ_{ω=ωlow_b}^{ωhigh_b} r_ω²`.
///
/// `r_lines` carries the step b) magnitudes with slice index `ω - 1`
/// holding line `ω`; `partitions` the Table D.3 rows for the
/// sampling rate (pass `model2_partition_table(fs)`). Returns `None`
/// when `partitions` is empty or `r_lines` is too short to cover the
/// last partition's `ωhigh`. One entry per partition, in table order.
#[must_use]
pub fn model2_step_e_eb(r_lines: &[f64], partitions: &[Model2PartitionEntry]) -> Option<Vec<f64>> {
    if partitions.is_empty() || r_lines.len() < partitions.last()?.whigh as usize {
        return None;
    }
    Some(
        partitions
            .iter()
            .map(|e| {
                r_lines[e.wlow as usize - 1..e.whigh as usize]
                    .iter()
                    .map(|&r| r * r)
                    .sum()
            })
            .collect(),
    )
}

/// §D.2.4 step e) — weighted unpredictability per threshold
/// calculation partition: `c_b = Σ_{ω=ωlow_b}^{ωhigh_b} r_ω²·c_ω`.
///
/// `r_lines` / `cw_lines` carry the step b) magnitudes and the step
/// d) unpredictability in the shared line layout (slice index `ω - 1`
/// holds line `ω`). Returns `None` when the two line slices disagree
/// in length, `partitions` is empty, or the slices are too short to
/// cover the last partition's `ωhigh`. One entry per partition, in
/// table order.
#[must_use]
pub fn model2_step_e_cb(
    r_lines: &[f64],
    cw_lines: &[f64],
    partitions: &[Model2PartitionEntry],
) -> Option<Vec<f64>> {
    if r_lines.len() != cw_lines.len()
        || partitions.is_empty()
        || r_lines.len() < partitions.last()?.whigh as usize
    {
        return None;
    }
    Some(
        partitions
            .iter()
            .map(|e| {
                let span = e.wlow as usize - 1..e.whigh as usize;
                r_lines[span.clone()]
                    .iter()
                    .zip(cw_lines[span].iter())
                    .map(|(&r, &c)| r * r * c)
                    .sum()
            })
            .collect(),
    )
}

/// §D.2.4 step f) — convolve a per-partition quantity with the
/// §D.2.3 spreading function:
///
/// ```text
/// out_b = Σ_{bb=1}^{bmax} in_bb * sprdngf(bval_bb, bval_b)
/// ```
///
/// The same reduction serves both printed step-f) convolutions: with
/// `per_partition = e_b` (partition energy, step e) it yields
/// `ecb_b`, and with `per_partition = c_b` (weighted
/// unpredictability, step e) it yields `ct_b`.
///
/// `bval` carries the median Bark value of each calculation
/// partition (the `bval` column of Tables D.3a–c — transcribed in
/// [`MODEL2_PARTITION_D3A`]–[`MODEL2_PARTITION_D3C`]; [`model2_bval`]
/// extracts the column for this function). Returns
/// `None` when the two slices disagree in length; the output has one
/// entry per partition, in slice order.
#[must_use]
pub fn model2_step_f_spread(per_partition: &[f64], bval: &[f64]) -> Option<Vec<f64>> {
    if per_partition.len() != bval.len() {
        return None;
    }
    Some(
        bval.iter()
            .map(|&bval_b| {
                per_partition
                    .iter()
                    .zip(bval.iter())
                    .map(|(&in_bb, &bval_bb)| in_bb * model2_sprdngf(bval_bb, bval_b))
                    .sum()
            })
            .collect(),
    )
}

/// §D.2.4 step f) normalization coefficient
/// `rnorm_b = 1 / Σ_bb sprdngf(bval_bb, bval_b)`.
///
/// "Due to the non-normalized nature of the spreading function,
/// `ecb_b` should be renormalized" — `rnorm_b` is the reciprocal of
/// the spreading-function row sum over all calculation partitions.
/// (The printed summation bound reads `bb=0` while clause D.2.2
/// states "Partition numbering starts at 1" and the step-f)
/// convolutions sum from `bb=1`; with a slice-based API the sum
/// simply runs over every provided partition, which satisfies both
/// readings.) One entry per partition, in slice order.
#[must_use]
pub fn model2_step_f_rnorm(bval: &[f64]) -> Vec<f64> {
    bval.iter()
        .map(|&bval_b| {
            let row_sum: f64 = bval
                .iter()
                .map(|&bval_bb| model2_sprdngf(bval_bb, bval_b))
                .sum();
            1.0 / row_sum
        })
        .collect()
}

/// §D.2.4 step f) renormalized weighted unpredictability
/// `cb_b = ct_b / ecb_b` ("Because `ct_b` is weighted by the signal
/// energy, it must be renormalized to `cb_b`").
///
/// Returns `None` on length mismatch. The spec does not define the
/// quotient for a partition with zero spread energy (`ecb_b = 0`,
/// only possible for an all-zero spectrum, where `ct_b` is zero
/// too); this implementation defines `cb_b = 0` there so the
/// downstream step g) tonality index stays finite.
#[must_use]
pub fn model2_step_f_cb(ct: &[f64], ecb: &[f64]) -> Option<Vec<f64>> {
    if ct.len() != ecb.len() {
        return None;
    }
    Some(
        ct.iter()
            .zip(ecb.iter())
            .map(|(&ct_b, &ecb_b)| if ecb_b == 0.0 { 0.0 } else { ct_b / ecb_b })
            .collect(),
    )
}

/// §D.2.4 step f) normalized energy `en_b = ecb_b * rnorm_b`.
///
/// Returns `None` on length mismatch.
#[must_use]
pub fn model2_step_f_en(ecb: &[f64], rnorm: &[f64]) -> Option<Vec<f64>> {
    if ecb.len() != rnorm.len() {
        return None;
    }
    Some(
        ecb.iter()
            .zip(rnorm.iter())
            .map(|(&ecb_b, &rnorm_b)| ecb_b * rnorm_b)
            .collect(),
    )
}

/// §D.2.4 step g) — tonality index
/// `tb_b = -0,299 - 0,43 * log_e(cb_b)`, limited to `0 ≤ tb_b ≤ 1`
/// (the spec prints the limit as "the range of 0<tb_b<1"; values
/// outside are clamped to the nearer bound).
///
/// `cb` is the step-f) renormalized weighted unpredictability: small
/// `cb` (predictable signal) maps to a tonality index near 1
/// (tone-like), large `cb` (unpredictable signal) to 0 (noise-like).
/// The unclamped expression crosses 1 at `cb = e^(-1,299/0,43)
/// ≈ 0,0488` and 0 at `cb = e^(-0,299/0,43) ≈ 0,4989`. `cb = 0`
/// (e.g. the zero-energy convention of [`model2_step_f_cb`]) clamps
/// to exactly 1. Inputs below 0 are outside the spec domain
/// (`cb_b` is a power-weighted average of the unpredictability
/// measure `c_ω ≥ 0`).
#[inline]
#[must_use]
pub fn model2_step_g_tonality(cb: f64) -> f64 {
    (-0.299 - 0.43 * cb.ln()).clamp(0.0, 1.0)
}

/// §D.2.4 step h) noise-masking-tone value: "`NMT_b = 5,5 dB` for
/// all `b`. `NMT_b` is the value for noise masking tone (in dB) for
/// the partition."
pub const MODEL2_NMT_DB: f64 = 5.5;

/// §D.2.4 step h) — required SNR for one calculation partition:
///
/// ```text
/// SNR_b = maximum(minval_b, tb_b * TMN_b + (1 - tb_b) * NMT_b)
/// ```
///
/// "Where maximum (a,b) is a function returning the least negative
/// of a or b." `tb` is the step g) tonality index (0 = noise-like,
/// 1 = tone-like); the second argument interpolates linearly between
/// the partition's tone-masking-noise offset `TMN_b` (Table D.3
/// `TMN` column, [`Model2PartitionEntry::tmn_db`]) at `tb = 1` and
/// the constant noise-masking-tone offset [`MODEL2_NMT_DB`] at
/// `tb = 0`; `minval_b` (Table D.3 `minval` column,
/// [`Model2PartitionEntry::minval_db`]) is the per-partition lower
/// limit "that controls stereo unmasking effects" (step e) item 5).
/// All quantities are in dB.
#[inline]
#[must_use]
pub fn model2_step_h_snr_db(tb: f64, minval_db: f64, tmn_db: f64) -> f64 {
    minval_db.max(tb * tmn_db + (1.0 - tb) * MODEL2_NMT_DB)
}

/// §D.2.4 step h) over all partitions — required SNR per
/// calculation partition, with `minval_b` / `TMN_b` read from the
/// Table D.3 rows (pass `model2_partition_table(fs)` as
/// `partitions`). `tb` carries the step g) tonality index per
/// partition, in slice order. Returns `None` on length mismatch;
/// one dB entry per partition otherwise.
#[must_use]
pub fn model2_step_h_snr(tb: &[f64], partitions: &[Model2PartitionEntry]) -> Option<Vec<f64>> {
    if tb.len() != partitions.len() {
        return None;
    }
    Some(
        tb.iter()
            .zip(partitions.iter())
            .map(|(&tb_b, e)| model2_step_h_snr_db(tb_b, e.minval_db, e.tmn_db))
            .collect(),
    )
}

/// §D.2.4 step i) — power ratio `bc_b = 10^(-SNR_b / 10)`.
///
/// Converts the step h) required SNR (dB) into the linear energy
/// ratio applied to the normalized partition energy in step j).
/// `SNR_b ≥ 0` dB (every Table D.3 `TMN` ≥ 24,5 dB and
/// `NMT = 5,5 dB`, so the step h) maximum is always positive) maps
/// to `0 < bc_b ≤ 1`, monotone decreasing in `SNR_b`.
#[inline]
#[must_use]
pub fn model2_step_i_bc(snr_db: f64) -> f64 {
    (10.0_f64).powf(-snr_db / 10.0)
}

/// §D.2.4 step j) — actual energy threshold per partition,
/// `nb_b = en_b * bc_b`.
///
/// `en` is the step f) normalized energy ([`model2_step_f_en`]);
/// `bc` is the step i) power ratio per partition. Returns `None` on
/// length mismatch.
#[must_use]
pub fn model2_step_j_nb(en: &[f64], bc: &[f64]) -> Option<Vec<f64>> {
    if en.len() != bc.len() {
        return None;
    }
    Some(
        en.iter()
            .zip(bc.iter())
            .map(|(&en_b, &bc_b)| en_b * bc_b)
            .collect(),
    )
}

/// §D.2.4 step k) — spread the threshold energy over FFT lines,
/// yielding `nb_ω`:
///
/// ```text
/// nb_ω = nb_b / (ωhigh_b - ωlow_b + 1)
/// ```
///
/// where `b` is the calculation partition containing line `ω`. `nb`
/// carries the step j) per-partition threshold energies in slice
/// order; `partitions` carries the matching Table D.3 rows (pass
/// `model2_partition_table(fs)`). The output is indexed by FFT line
/// with slice index `ω - 1` holding line `ω` (the spec's line domain
/// is 1-based), and its length is the last partition's `ωhigh` —
/// with the full Table D.3 slices that is exactly 513, covering the
/// 1024-point-FFT half-spectrum with no gaps (coverage contiguity is
/// a pinned transcription property). Energy is conserved:
/// summing `nb_ω` over a partition's lines recovers `nb_b`.
///
/// Returns `None` when the two slices disagree in length or
/// `partitions` is empty.
#[must_use]
pub fn model2_step_k_nb_lines(nb: &[f64], partitions: &[Model2PartitionEntry]) -> Option<Vec<f64>> {
    if nb.len() != partitions.len() || partitions.is_empty() {
        return None;
    }
    let last_line = partitions.last()?.whigh as usize;
    let mut out = vec![0.0; last_line];
    for (&nb_b, e) in nb.iter().zip(partitions.iter()) {
        let count = f64::from(e.whigh - e.wlow + 1);
        let nb_w = nb_b / count;
        for line in e.wlow..=e.whigh {
            out[line as usize - 1] = nb_w;
        }
    }
    Some(out)
}

/// Convert a Table D.4 `absthr` dB value to the energy domain.
///
/// Step l): "The dB values of `absthr` shown in tables D.4 …
/// are relative to the level that a sine wave of ±½ lsb has in the
/// FFT used for threshold calculation. The dB values must be
/// converted into the energy domain after considering the FFT
/// normalization actually used." The conversion is therefore
/// implementation-dependent: `half_lsb_sine_level_db` is the energy
/// level, in dB under the same convention, that a ±½-lsb sine wave
/// produces in the caller's FFT — i.e. the table's 0-dB reference
/// point. The result is `10^((absthr_db + half_lsb_sine_level_db) / 10)`.
#[inline]
#[must_use]
pub fn model2_absthr_energy(absthr_db: f64, half_lsb_sine_level_db: f64) -> f64 {
    (10.0_f64).powf((absthr_db + half_lsb_sine_level_db) / 10.0)
}

/// §D.2.4 step l) — final energy threshold of audibility for one
/// FFT line: `thr_ω = max(nb_ω, absthr_ω)`.
///
/// `absthr_w` must already be in the energy domain (see
/// [`model2_absthr_energy`] for the documented conversion from the
/// Table D.4 dB prints).
#[inline]
#[must_use]
pub fn model2_step_l_thr(nb_w: f64, absthr_w: f64) -> f64 {
    nb_w.max(absthr_w)
}

/// §D.2.4 step l) over the line domain — elementwise
/// `thr_ω = max(nb_ω, absthr_ω)`. `nb` is the step k) per-line
/// threshold ([`model2_step_k_nb_lines`] output layout: slice index
/// `ω - 1` holds line `ω`); `absthr` is the energy-domain absolute
/// threshold per line in the same layout. Lines the printed Table
/// D.4 leaves uncovered (the D.4a line-58 gap and every line above
/// the table's last covered line — [`model2_absthr_for_line`]
/// returns `None` there) have no absolute-threshold floor; callers
/// represent that as `absthr_ω = 0` so the maximum passes `nb_ω`
/// through unchanged. Returns `None` on length mismatch.
///
/// Step m) (pre-echo control) follows this step for Layer III only —
/// "This step is omitted for Layers I and II."
#[must_use]
pub fn model2_step_l_thr_lines(nb: &[f64], absthr: &[f64]) -> Option<Vec<f64>> {
    if nb.len() != absthr.len() {
        return None;
    }
    Some(
        nb.iter()
            .zip(absthr.iter())
            .map(|(&nb_w, &absthr_w)| model2_step_l_thr(nb_w, absthr_w))
            .collect(),
    )
}

/// §D.2.4 step n) — energy in one Table D.5 coder partition
/// (scalefactor band):
///
/// ```text
/// epart_n = Σ_{ω=ωlow_n}^{ωhigh_n} r_ω²
/// ```
///
/// `r_lines` carries the FFT magnitudes `r_ω` with slice index
/// `ω - 1` holding line `ω`; `span` is the partition's Table D.5
/// descriptor ([`coder_partition_d5_span`] /
/// [`coder_partition_d5_spans`]). Returns `None` when `r_lines` is
/// too short to cover `ωhigh_n`.
#[must_use]
pub fn model2_step_n_epart(r_lines: &[f64], span: CoderPartitionD5Span) -> Option<f64> {
    let lines = r_lines.get(span.omega_low as usize - 1..span.omega_high as usize)?;
    Some(lines.iter().map(|&r| r * r).sum())
}

/// §D.2.4 step n) — noise level in one Table D.5 coder partition.
///
/// If `width_n = 1` (psychoacoustically narrow scalefactor band —
/// "one whose width is less than approximately ⅓ critical band"):
///
/// ```text
/// npart_n = Σ_{ω=ωlow_n}^{ωhigh_n} thr_ω
/// ```
///
/// else (`width_n = 0`, psychoacoustically wide):
///
/// ```text
/// npart_n = minimum(thr_ωlow_n, …, thr_ωhigh_n) * (ωhigh_n - ωlow_n + 1)
/// ```
///
/// "Where, in this case, minimum (a,…,z) is a function returning the
/// smallest **positive** argument of the arguments a…z." With every
/// in-domain `thr_ω > 0` (energies floored by the step l) absolute
/// threshold) the positivity qualifier is a no-op; for the
/// out-of-domain case where no argument is positive this
/// implementation returns `npart_n = 0`. `thr_lines` uses the
/// step k)/l) line layout (slice index `ω - 1` holds line `ω`).
/// Returns `None` when `thr_lines` is too short to cover `ωhigh_n`.
#[must_use]
pub fn model2_step_n_npart(thr_lines: &[f64], span: CoderPartitionD5Span) -> Option<f64> {
    let lines = thr_lines.get(span.omega_low as usize - 1..span.omega_high as usize)?;
    Some(if span.width == 1 {
        lines.iter().sum()
    } else {
        let min_pos = lines
            .iter()
            .copied()
            .filter(|&t| t > 0.0)
            .fold(f64::INFINITY, f64::min);
        if min_pos.is_finite() {
            min_pos * lines.len() as f64
        } else {
            0.0
        }
    })
}

/// §D.2.4 step n) — the signal-to-mask ratio sent to the coder for
/// one partition:
///
/// ```text
/// SMR_n = 10 log10(epart_n / npart_n)
/// ```
///
/// Spec domain is positive energies (`epart_n > 0`, `npart_n > 0`);
/// outside it the IEEE quotient/logarithm conventions apply
/// unmodified.
#[inline]
#[must_use]
pub fn model2_step_n_smr_db(epart: f64, npart: f64) -> f64 {
    10.0 * (epart / npart).log10()
}

/// §D.2.4 step n) over all recoverable Table D.5 coder partitions —
/// the Model 2 output vector of signal-to-mask ratios `SMR_n` for
/// `n ∈ 1..=32`, in ascending partition order.
///
/// `r_lines` carries the FFT magnitudes `r_ω` and `thr_lines` the
/// step l) final energy thresholds `thr_ω`, both with slice index
/// `ω - 1` holding line `ω`. Returns `None` when either slice is too
/// short to cover the last partition's `ωhigh` (513). Each entry is
/// `model2_step_n_smr_db(epart_n, npart_n)` over the partition's
/// inclusive Table D.5 line range (the shared boundary line
/// `ωhigh_n = ωlow_{n+1}` is read by both adjacent partitions, per
/// the inclusive-on-both-ends reading of the printed boundary
/// column).
#[must_use]
pub fn model2_step_n_smr(r_lines: &[f64], thr_lines: &[f64]) -> Option<Vec<f64>> {
    coder_partition_d5_spans()
        .map(|span| {
            let epart = model2_step_n_epart(r_lines, span)?;
            let npart = model2_step_n_npart(thr_lines, span)?;
            Some(model2_step_n_smr_db(epart, npart))
        })
        .collect()
}

/// Persistent state of one Psychoacoustic Model 2 threshold
/// generator — the §D.2.1 "preceding FFT source data window" plus the
/// `t-1` / `t-2` polar spectra the step c) predictor consumes
/// (Phase 2 step 84 / r282).
///
/// §D.2.1 (verbatim): "Before running the model initially, the array
/// used to hold the preceding FFT source data window and the arrays
/// used to hold r and f should be zeroed to provide a known starting
/// point." [`Model2State::new`] performs exactly that zeroing; the
/// first two [`Model2State::smr`] calls therefore predict against
/// zeroed history (maximally unpredictable, `c_ω = 1` on live lines)
/// and converge from the third call on, as the spec procedure
/// dictates.
///
/// One state instance corresponds to one "particular application of
/// the threshold calculation process" — `iblen` and the sampling rate
/// "must remain constant over any particular application" (§D.2.1
/// inputs a) and c)); a Layer III encoder needing two shift lengths
/// runs "two processes, each running with a fixed shift length",
/// i.e. two independent `Model2State` values.
#[derive(Debug, Clone, PartialEq)]
pub struct Model2State {
    /// The preceding 1 024-sample FFT source data window (§D.2.4
    /// step a) reconstruction output of the previous call; all-zero
    /// initially).
    window: Vec<f64>,
    /// Polar spectrum of the previous block (`t-1`; zeroed initially).
    prev: Model2Polar,
    /// Polar spectrum of the block before that (`t-2`; zeroed
    /// initially).
    prev2: Model2Polar,
}

impl Default for Model2State {
    fn default() -> Self {
        Self::new()
    }
}

impl Model2State {
    /// Freshly zeroed threshold-generator state, per the §D.2.1
    /// initialization sentence.
    #[must_use]
    pub fn new() -> Self {
        Self {
            window: vec![0.0; MODEL2_FFT_LEN],
            prev: Model2Polar::zeroed(),
            prev2: Model2Polar::zeroed(),
        }
    }

    /// Run one full §D.2.4 threshold calculation — steps a) through
    /// l) plus n) — over the `iblen` newest input samples, producing
    /// the Model 2 output ("a set of Signal-to-Masking Ratios,
    /// SMR_n") for the 32 Table D.5 coder partitions.
    ///
    /// * `new_samples` — the §D.2.1 input b) block (`iblen`
    ///   samples; the §D.2.1 `384<iblen<640` application contract is
    ///   checkable via [`model2_iblen_in_range`] and the caller keeps
    ///   `iblen` constant across calls).
    /// * `fs` — the §D.2.1 input c) sampling rate, selecting the
    ///   Table D.3 calculation partitions and Table D.4 absolute
    ///   thresholds.
    /// * `half_lsb_sine_level_db` — the step l) Table D.4 0-dB
    ///   reference under this implementation's FFT normalization
    ///   (see [`model2_absthr_energy`]).
    /// * `cw_through_line` — the optional step d)
    ///   partial-calculation limit (see [`model2_step_d_cw_lines`]).
    ///
    /// The walk chains the front-half primitives (steps a)–e)) into
    /// the previously landed back half: f) spreading convolution +
    /// renormalization, g) tonality, h) required SNR, i) power
    /// ratio, j) partition threshold, k) line spread, l)
    /// absolute-threshold floor (Table D.4-uncovered lines have no
    /// floor and pass `nb_ω` through), and n) the SMR reduction over
    /// the Table D.5 spans. Step m) (pre-echo control) "is omitted
    /// for Layers I and II" and is not part of this walk.
    ///
    /// On success the state advances (`t-1 → t-2`, current block
    /// → `t-1`, window replaced) and the 32 `SMR_n` values are
    /// returned in ascending partition order. Returns `None` —
    /// leaving the state untouched — when `new_samples` is empty or
    /// longer than 1 024 (the step a) domain).
    #[must_use]
    pub fn smr(
        &mut self,
        new_samples: &[f64],
        fs: AnnexDSamplingRate,
        half_lsb_sine_level_db: f64,
        cw_through_line: Option<usize>,
    ) -> Option<Vec<f64>> {
        // a) Reconstruct the 1 024-sample analysis window.
        let window = model2_step_a_reconstruct(&self.window, new_samples)?;
        // b) Windowed forward FFT, polar representation.
        let polar = model2_step_b_spectrum(&window)?;
        // c) Predicted r̂/f̂ from the t-1 / t-2 spectra.
        let predicted = model2_step_c_predict_polar(&self.prev, &self.prev2)?;
        // d) Unpredictability measure c_ω.
        let cw = model2_step_d_cw_lines(&polar, &predicted, cw_through_line)?;
        // e) Partition energy e_b and weighted unpredictability c_b.
        let partitions = model2_partition_table(fs);
        let eb = model2_step_e_eb(&polar.r, partitions)?;
        let cb_raw = model2_step_e_cb(&polar.r, &cw, partitions)?;
        // f) Spreading convolution + renormalization.
        let bval = model2_bval(fs);
        let ecb = model2_step_f_spread(&eb, &bval)?;
        let ct = model2_step_f_spread(&cb_raw, &bval)?;
        let cb = model2_step_f_cb(&ct, &ecb)?;
        let rnorm = model2_step_f_rnorm(&bval);
        let en = model2_step_f_en(&ecb, &rnorm)?;
        // g) Tonality index per partition.
        let tb: Vec<f64> = cb.iter().map(|&c| model2_step_g_tonality(c)).collect();
        // h) Required SNR; i) power ratio; j) partition threshold.
        let snr = model2_step_h_snr(&tb, partitions)?;
        let bc: Vec<f64> = snr.iter().map(|&s| model2_step_i_bc(s)).collect();
        let nb = model2_step_j_nb(&en, &bc)?;
        // k) Spread the threshold energy over the FFT lines.
        let nb_lines = model2_step_k_nb_lines(&nb, partitions)?;
        // l) Floor by the Table D.4 absolute threshold (energy
        // domain); uncovered lines have no floor (absthr_ω = 0).
        let absthr: Vec<f64> = (1..=nb_lines.len() as u16)
            .map(|line| {
                model2_absthr_for_line(fs, line)
                    .map_or(0.0, |db| model2_absthr_energy(db, half_lsb_sine_level_db))
            })
            .collect();
        let thr = model2_step_l_thr_lines(&nb_lines, &absthr)?;
        // n) SMR_n over the Table D.5 coder partitions.
        let smr = model2_step_n_smr(&polar.r, &thr)?;
        // Advance the state only after the whole walk succeeded.
        self.prev2 = core::mem::replace(&mut self.prev, polar);
        self.window = window;
        Some(smr)
    }
}

/// One row of Annex D Table D.5 — *Layer I and Layer II coder
/// partition table*.
///
/// Table D.5 enumerates the **coder partitions** of ISO/IEC 11172-3
/// Annex D clause D.2 used by Models 1 and 2 to aggregate
/// per-FFT-line thresholds into the per-partition values fed to the
/// Layer I / Layer II bit-allocation loop. The table prints three
/// columns per row, with the second column doubling as both the
/// previous partition's top line (`ωhigh_n`) and the next
/// partition's first line (`ωlow_{n+1}`):
///
/// 1. `index` — the coder-partition number `n`, 0..=32.
/// 2. `omega_boundary` — the FFT-line index that sits at the
///    partition boundary; for row `n` this is simultaneously
///    `ωlow_{n+1}` (the first line of partition `n+1`) and
///    `ωhigh_n` (the last line of partition `n`).
/// 3. `width` — the `width_n` value the spec table prints
///    against partition `n`. In the table this is 0 for rows
///    0..=12 and 1 for rows 13..=32.
///
/// The fields are transcribed verbatim from the
/// `mp3-annex-d-psychoacoustic-extracts.md` text extract (which is
/// itself a verbatim transcription of the PDF page render of
/// printed p.139, PDF page 145).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoderPartitionD5 {
    /// Coder-partition index `n`. Spec range: 0..=32.
    pub index: u16,
    /// Partition boundary FFT-line index — `ωlow_{n+1}` /
    /// `ωhigh_n` in the spec table. Spec range: 1..=513.
    pub omega_boundary: u16,
    /// `width_n` value the spec table prints for this row. 0 for
    /// rows 0..=12, 1 for rows 13..=32.
    pub width: u16,
}

impl CoderPartitionD5 {
    /// Construct a row at compile time.
    #[inline]
    #[must_use]
    pub const fn new(index: u16, omega_boundary: u16, width: u16) -> Self {
        Self {
            index,
            omega_boundary,
            width,
        }
    }

    /// Read this row's `omega_boundary` under its **`ωhigh_n` role**.
    ///
    /// Annex D Table D.5 prints a single column for the partition
    /// boundary FFT-line index under the dual-role heading
    /// `ωlow_{n+1} / ωhigh_n`. The cell's value is verbatim the
    /// printed integer; the column heading names two distinct spec
    /// roles for it. This accessor returns the value under the
    /// `ωhigh_n` role — the FFT-line index of the upper boundary of
    /// partition `n` (where `n = self.index`). It is a pure rename of
    /// `self.omega_boundary` and performs **no** arithmetic.
    ///
    /// Provenance: column heading `ωlow_{n+1} / ωhigh_n` in
    /// `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
    /// §"Table D.5 - Layer I and Layer II coder partition table".
    #[inline]
    #[must_use]
    pub const fn omega_high(self) -> u16 {
        self.omega_boundary
    }

    /// Read this row's `omega_boundary` under its **`ωlow_{n+1}`
    /// role**.
    ///
    /// The Table D.5 column heading `ωlow_{n+1} / ωhigh_n` names two
    /// distinct spec roles for the row's verbatim printed integer.
    /// This accessor returns the value under the `ωlow_{n+1}` role —
    /// the FFT-line index of the lower boundary of the **next**
    /// partition `n + 1` (where `n = self.index`). It is a pure
    /// rename of `self.omega_boundary` and performs **no** arithmetic.
    ///
    /// Partition `0`'s own lower boundary `ωlow_0` is **not** in
    /// Table D.5 — only `ωlow_n` for `n ∈ 1..=33` is recoverable from
    /// this row set, by reading row `n - 1`'s `omega_low_of_next()`.
    ///
    /// Provenance: column heading `ωlow_{n+1} / ωhigh_n` in
    /// `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
    /// §"Table D.5 - Layer I and Layer II coder partition table".
    #[inline]
    #[must_use]
    pub const fn omega_low_of_next(self) -> u16 {
        self.omega_boundary
    }
}

/// Annex D **Table D.5** — Layer I / Layer II coder partition table.
/// 33 rows, transcribed verbatim from
/// `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
/// §"Table D.5 - Layer I and Layer II coder partition table".
///
/// Each row carries the partition boundary FFT-line index
/// (`ωlow_{n+1} / ωhigh_n` in the spec table) and the spec's
/// `width_n` value. The rows are ordered by ascending partition
/// index `n = 0..=32` and the boundary column is strictly monotonic
/// in `n` by 16 lines per partition.
pub const CODER_PARTITION_TABLE_D5: [CoderPartitionD5; 33] = [
    CoderPartitionD5::new(0, 1, 0),
    CoderPartitionD5::new(1, 17, 0),
    CoderPartitionD5::new(2, 33, 0),
    CoderPartitionD5::new(3, 49, 0),
    CoderPartitionD5::new(4, 65, 0),
    CoderPartitionD5::new(5, 81, 0),
    CoderPartitionD5::new(6, 97, 0),
    CoderPartitionD5::new(7, 113, 0),
    CoderPartitionD5::new(8, 129, 0),
    CoderPartitionD5::new(9, 145, 0),
    CoderPartitionD5::new(10, 161, 0),
    CoderPartitionD5::new(11, 177, 0),
    CoderPartitionD5::new(12, 193, 0),
    CoderPartitionD5::new(13, 209, 1),
    CoderPartitionD5::new(14, 225, 1),
    CoderPartitionD5::new(15, 241, 1),
    CoderPartitionD5::new(16, 257, 1),
    CoderPartitionD5::new(17, 273, 1),
    CoderPartitionD5::new(18, 289, 1),
    CoderPartitionD5::new(19, 305, 1),
    CoderPartitionD5::new(20, 321, 1),
    CoderPartitionD5::new(21, 337, 1),
    CoderPartitionD5::new(22, 353, 1),
    CoderPartitionD5::new(23, 369, 1),
    CoderPartitionD5::new(24, 385, 1),
    CoderPartitionD5::new(25, 401, 1),
    CoderPartitionD5::new(26, 417, 1),
    CoderPartitionD5::new(27, 433, 1),
    CoderPartitionD5::new(28, 449, 1),
    CoderPartitionD5::new(29, 465, 1),
    CoderPartitionD5::new(30, 481, 1),
    CoderPartitionD5::new(31, 497, 1),
    CoderPartitionD5::new(32, 513, 1),
];

/// Coder-partition stride: the difference in the FFT-line boundary
/// column between two consecutive Table D.5 rows. The spec table
/// prints a strictly uniform 16-line stride across all 32 row
/// transitions, so every partition spans 16 FFT lines (the
/// `width_n` column records a separate orthogonal quantity per the
/// spec text and is exposed unchanged on each row).
pub const CODER_PARTITION_D5_STRIDE: u16 = 16;

/// Look up the Table D.5 row for partition index `n`. Returns
/// `None` for any `n` outside the spec range 0..=32.
///
/// This is a direct row accessor — it does **not** interpret the
/// `ωlow_{n+1} / ωhigh_n` column heading's dual role: the row's
/// `omega_boundary` field is the verbatim printed value, and
/// callers that need either of the two boundary roles must apply
/// the spec arithmetic explicitly. See
/// [`coder_partition_d5_omega_high`] and
/// [`coder_partition_d5_omega_low`] for the two role-aware
/// table-level accessors.
#[inline]
#[must_use]
pub fn coder_partition_d5(n: u16) -> Option<CoderPartitionD5> {
    CODER_PARTITION_TABLE_D5.get(n as usize).copied()
}

/// Read the upper FFT-line boundary `ωhigh_n` of coder partition
/// `n` from Table D.5. Returns `None` for any `n` outside the spec
/// range 0..=32.
///
/// Annex D Table D.5 prints a single column for the partition
/// boundary FFT-line index under the dual-role heading
/// `ωlow_{n+1} / ωhigh_n`. The row at index `n` carries the
/// verbatim printed integer in its `omega_boundary` field; under
/// the `ωhigh_n` role that integer is the **upper** boundary of
/// partition `n`. This accessor is a pure column rename — it
/// performs **no** arithmetic and the value matches
/// `coder_partition_d5(n).map(|r| r.omega_boundary)` exactly.
///
/// Provenance: column heading `ωlow_{n+1} / ωhigh_n` in
/// `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
/// §"Table D.5 - Layer I and Layer II coder partition table".
#[inline]
#[must_use]
pub fn coder_partition_d5_omega_high(n: u16) -> Option<u16> {
    coder_partition_d5(n).map(CoderPartitionD5::omega_high)
}

/// Read the lower FFT-line boundary `ωlow_n` of coder partition
/// `n` from Table D.5. Returns `None` for any `n` outside the spec
/// range **1..=33** (NOT 0..=32 — see below).
///
/// Annex D Table D.5 prints `ωlow_{n+1}` at row `n`'s
/// `omega_boundary` cell, so the table covers `ωlow_n` for
/// `n ∈ 1..=33` only — partition 0's own lower boundary `ωlow_0`
/// is **not** present in the table. Inputs `n = 0` and `n > 33`
/// both return `None`; inputs `n ∈ 1..=33` return row `n - 1`'s
/// verbatim `omega_boundary` value (i.e. row `n - 1`'s
/// [`CoderPartitionD5::omega_low_of_next`] reading). This accessor
/// is a pure column rename plus the `n → n - 1` row shift required
/// by the column heading's `ωlow_{n+1}` half — no other arithmetic
/// is performed.
///
/// Provenance: column heading `ωlow_{n+1} / ωhigh_n` in
/// `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
/// §"Table D.5 - Layer I and Layer II coder partition table".
#[inline]
#[must_use]
pub fn coder_partition_d5_omega_low(n: u16) -> Option<u16> {
    if n == 0 {
        return None;
    }
    coder_partition_d5(n - 1).map(CoderPartitionD5::omega_low_of_next)
}

/// Read partition `n`'s **FFT-line span** `[ωlow_n, ωhigh_n]` from
/// Table D.5 as the inclusive `(ωlow_n, ωhigh_n)` tuple. Returns
/// `None` for any `n` outside the spec range **1..=32**.
///
/// Annex D Table D.5 prints a single FFT-line boundary column under
/// the dual-role heading `ωlow_{n+1} / ωhigh_n`. The Phase 2 step 50
/// accessors [`coder_partition_d5_omega_high`] and
/// [`coder_partition_d5_omega_low`] expose each role separately;
/// this accessor composes them into the full inclusive line span of
/// a single partition. The composition rule is verbatim:
///
/// * `ωlow_n` comes from row `n - 1`'s `omega_boundary` (the column
///   heading's `ωlow_{n+1}` role at row `n - 1`).
/// * `ωhigh_n` comes from row `n`'s `omega_boundary` (the column
///   heading's `ωhigh_n` role at row `n`).
///
/// The valid input range is the intersection of the two underlying
/// accessors' ranges: `omega_low` is defined for `n ∈ 1..=33`,
/// `omega_high` for `n ∈ 0..=32`, so a partition's full span is
/// recoverable only for `n ∈ 1..=32`. Two partitions are missing
/// one boundary each:
///
/// * `n = 0` — `ωlow_0` is **not** in Table D.5 (the column heading
///   `ωlow_{n+1}` shifts the lower boundary up by one). Returns
///   `None` verbatim; no default lower boundary is invented.
/// * `n = 33` — `ωhigh_33` is **not** in Table D.5 (the table tops
///   out at row `n = 32` with `ωhigh_32 = 513`). Returns `None`
///   verbatim.
///
/// For every `n ∈ 1..=32` the returned tuple is the inclusive
/// `(lower, upper)` pair: every FFT line index `k ∈ [lower, upper]`
/// is inside partition `n`. By the spec table's uniform 16-line
/// stride (pinned by `CODER_PARTITION_D5_STRIDE`) the span always
/// covers exactly 17 lines (the boundary cells at both ends are
/// inclusive; the open-interval line count is 16).
///
/// This accessor is a pure composition of [`coder_partition_d5_omega_low`]
/// and [`coder_partition_d5_omega_high`] — no arithmetic beyond the
/// `n → n - 1` row shift that the `ωlow_{n+1}` column-heading half
/// already encodes inside `coder_partition_d5_omega_low`.
///
/// Provenance: column heading `ωlow_{n+1} / ωhigh_n` in
/// `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
/// §"Table D.5 - Layer I and Layer II coder partition table".
#[inline]
#[must_use]
pub fn coder_partition_d5_line_range(n: u16) -> Option<(u16, u16)> {
    let low = coder_partition_d5_omega_low(n)?;
    let high = coder_partition_d5_omega_high(n)?;
    Some((low, high))
}

/// Read the `width_n` value of coder partition `n` from Table D.5.
/// Returns `None` for any `n` outside the spec range 0..=32.
///
/// Annex D Table D.5 prints three columns per row: the partition
/// index `n`, the dual-role partition-boundary FFT-line cell
/// `ωlow_{n+1} / ωhigh_n`, and a third column `width_n`. The
/// previously-landed accessors expose the first two columns; this
/// accessor exposes the third. The verbatim transcribed values are:
///
/// * rows `n ∈ 0..=12` — `width_n = 0`;
/// * rows `n ∈ 13..=32` — `width_n = 1`.
///
/// The column is structurally orthogonal to the boundary column
/// (the row `n = 13` transition from 0 to 1 does NOT coincide with
/// any other discontinuity in the table) and is exposed here as a
/// pure rename of `CoderPartitionD5::width` — no arithmetic and no
/// interpretation. This accessor matches
/// `coder_partition_d5(n).map(|r| r.width)` exactly.
///
/// Provenance: the `width_n` column in
/// `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
/// §"Table D.5 - Layer I and Layer II coder partition table".
#[inline]
#[must_use]
pub fn coder_partition_d5_width(n: u16) -> Option<u16> {
    coder_partition_d5(n).map(|r| r.width)
}

/// A composed per-partition descriptor for Annex D Table D.5 — the
/// three verbatim columns of coder partition `n` reassembled into a
/// single record indexable by partition number, with the
/// dual-role boundary column already resolved into the two
/// distinct spec roles.
///
/// The descriptor carries:
///
/// 1. `index` — partition number `n`.
/// 2. `omega_low` — FFT-line index `ωlow_n` (lower boundary, inclusive),
///    read from row `n - 1`'s boundary cell under the column heading's
///    `ωlow_{n+1}` role.
/// 3. `omega_high` — FFT-line index `ωhigh_n` (upper boundary,
///    inclusive), read from row `n`'s boundary cell under the column
///    heading's `ωhigh_n` role.
/// 4. `width` — the `width_n` value the spec table prints against
///    partition `n` (0 for `n ∈ 1..=12`, 1 for `n ∈ 13..=32`; the
///    `n = 0` row's `width_n = 0` cell is not surfaced through this
///    descriptor — see range restriction below).
///
/// The descriptor exists for `n ∈ 1..=32` only — the same range as
/// [`coder_partition_d5_line_range`]: partition 0's `ωlow_0` is not
/// in Table D.5 (the column heading's `ωlow_{n+1}` shift removes it)
/// and partition 33's `ωhigh_33` is not in Table D.5 (the table tops
/// out at row 32 with `ωhigh_32 = 513`). The descriptor inherits
/// these two boundary-table gaps verbatim and never invents a
/// synthetic lower or upper boundary at either edge.
///
/// Provenance: the Table D.5 row at index `n` (for `width`) and at
/// indices `n - 1` and `n` (for the dual-role boundary column) in
/// `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
/// §"Table D.5 - Layer I and Layer II coder partition table".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoderPartitionD5Span {
    /// Partition number `n`. Spec range under this descriptor: 1..=32.
    pub index: u16,
    /// Lower FFT-line boundary `ωlow_n` (inclusive). Read from row
    /// `n - 1`'s `omega_boundary` cell under the column heading's
    /// `ωlow_{n+1}` role.
    pub omega_low: u16,
    /// Upper FFT-line boundary `ωhigh_n` (inclusive). Read from row
    /// `n`'s `omega_boundary` cell under the column heading's
    /// `ωhigh_n` role.
    pub omega_high: u16,
    /// `width_n` value the spec table prints for partition `n`. 0 for
    /// `n ∈ 1..=12`; 1 for `n ∈ 13..=32`.
    pub width: u16,
}

/// Compose Annex D Table D.5 partition `n`'s FFT-line range with its
/// `width_n` value into a single per-partition descriptor. Returns
/// `None` for any `n` outside the spec range **1..=32**.
///
/// The downstream Model 1 / Model 2 partition-threshold reduction
/// iterates Table D.5 row by row and, for each in-range partition,
/// reads three pieces of data: the lower FFT-line boundary `ωlow_n`,
/// the upper FFT-line boundary `ωhigh_n`, and the `width_n` value
/// the table prints against the row. The Phase 2 step 51 accessor
/// [`coder_partition_d5_line_range`] exposed the line-range pair and
/// the Phase 2 step 52 accessor [`coder_partition_d5_width`] exposed
/// the `width_n` value; this step 53 accessor composes the two into
/// the single descriptor that the per-partition reduction loop
/// consumes per iteration.
///
/// The composition is **pure** — no arithmetic beyond what the
/// underlying accessors already perform:
///
/// * `omega_low` is `coder_partition_d5_omega_low(n)` (the `n → n - 1`
///   row shift that the `ωlow_{n+1}` column-heading half encodes,
///   then a column rename — both inherited verbatim from step 50);
/// * `omega_high` is `coder_partition_d5_omega_high(n)` (a column
///   rename of row `n`'s `omega_boundary` cell — inherited from
///   step 50);
/// * `width` is `coder_partition_d5_width(n)` (a rename of row `n`'s
///   `width` field — inherited from step 52).
///
/// The descriptor's valid range is the **intersection** of the
/// line-range accessor's range (`n ∈ 1..=32`) and the width
/// accessor's range (`n ∈ 0..=32`) — i.e. `n ∈ 1..=32`. Partition
/// 0's row carries a valid `width_n = 0` and a valid `ωhigh_0 = 1`,
/// but no `ωlow_0`, so the descriptor at `n = 0` returns `None`
/// verbatim. Partition 33 returns `None` because neither
/// `ωhigh_33` nor a row for `width_33` is in Table D.5.
///
/// Provenance: only Table D.5 in
/// `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
/// §"Table D.5 - Layer I and Layer II coder partition table" is
/// consulted; the underlying step-50, step-51 and step-52 accessors
/// already cite the same source.
#[inline]
#[must_use]
pub fn coder_partition_d5_span(n: u16) -> Option<CoderPartitionD5Span> {
    let (omega_low, omega_high) = coder_partition_d5_line_range(n)?;
    let width = coder_partition_d5_width(n)?;
    Some(CoderPartitionD5Span {
        index: n,
        omega_low,
        omega_high,
        width,
    })
}

/// Inclusive-line membership predicate over Annex D Table D.5 partition
/// `n` — does FFT-line index `omega` fall within partition `n`'s
/// inclusive boundary range `[ωlow_n, ωhigh_n]`?
///
/// The downstream Model 1 / Model 2 partition-threshold reduction needs
/// to bin per-FFT-line energies into the per-partition accumulators that
/// Table D.5 defines. The Phase 2 step 53 (r252) descriptor
/// [`coder_partition_d5_span`] exposes the inclusive boundary pair
/// `(ωlow_n, ωhigh_n)` for partition `n`; this step 54 accessor lifts
/// the obvious membership test on that pair to a named predicate so
/// callers don't repeat the inequality at every binning site (and so
/// the range-rejection behaviour at the two boundary-table gaps stays
/// in one place).
///
/// The predicate evaluates `omega_low <= omega && omega <= omega_high`
/// over the descriptor `[ωlow_n, ωhigh_n]` returned by
/// [`coder_partition_d5_span`], reflecting the spec's reading of the
/// boundary column heading `ωlow_{n+1} / ωhigh_n` as *inclusive* on
/// both ends. The tiling property already pinned by
/// `coder_partition_d5_span_tiles_the_band` is `ωhigh_n =
/// ωlow_{n+1}`, so the shared boundary line lies in **both**
/// partitions `n` and `n + 1` under the inclusive-on-both-ends
/// reading; the caller's downstream reduction handles the shared
/// boundary as the spec prescribes (typically by reading partition
/// `n` first up through `ωhigh_n` then partition `n + 1` from the
/// same `ωlow_{n+1} = ωhigh_n` line — both readings are sample-exact
/// against the spec table).
///
/// The accessor returns `Some(bool)` for any `n ∈ 1..=32` (the same
/// range as the descriptor itself) and **`None`** for any `n` outside
/// that range. The two edge cases inherit from the descriptor:
///
/// * `n = 0` — partition 0's lower boundary `ωlow_0` is not in
///   Table D.5; without a `ωlow_n`, the membership test is undefined.
///   `None` rather than a synthetic answer.
/// * `n = 33` — neither row 33's boundary nor its `width_n` cell
///   exists in Table D.5. `None`.
///
/// The `omega` argument is **not** range-checked against the
/// table-wide FFT-line domain `[1, 513]`. A caller passing an
/// out-of-band value (e.g. `omega = 0` or `omega = 1024`) gets a
/// well-defined `false` answer for every in-range `n`, exactly as the
/// inequality on the descriptor's `[ωlow_n, ωhigh_n]` dictates — the
/// predicate is a pure boolean over the descriptor and does not
/// re-invent the table-wide line domain.
///
/// The predicate is **pure**: it is `coder_partition_d5_span(n).map(
/// |s| s.omega_low <= omega && omega <= s.omega_high)` exactly. No
/// arithmetic beyond the inequality on the descriptor's pre-computed
/// boundaries is introduced.
///
/// Provenance: only the Phase 2 step 53 descriptor
/// [`coder_partition_d5_span`] and its underlying Table D.5
/// transcription in
/// `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
/// §"Table D.5 - Layer I and Layer II coder partition table" are
/// consulted; the inclusive-on-both-ends boundary reading is the
/// spec's, pinned by Phase 2 step 50 (r249).
#[inline]
#[must_use]
pub fn partition_n_contains_line(n: u16, omega: u16) -> Option<bool> {
    let span = coder_partition_d5_span(n)?;
    Some(span.omega_low <= omega && omega <= span.omega_high)
}

/// Row-order iterator over Annex D Table D.5's recoverable partition
/// descriptors — yields `CoderPartitionD5Span` for every `n ∈ 1..=32`
/// in ascending order, with no gaps and no repetition.
///
/// The downstream Model 1 / Model 2 partition-threshold reduction walks
/// Table D.5 row by row, accumulating per-partition values across the
/// in-range FFT lines. Phase 2 step 53 (r252) composed each partition
/// `n`'s three Table D.5 columns into a single
/// [`CoderPartitionD5Span`] descriptor; Phase 2 step 54 (r253) added
/// the `partition_n_contains_line(n, ω)` inclusive-line membership
/// predicate on that descriptor. This step 55 helper closes the loop:
/// the reduction now reads as
///
/// ```text
///     for span in coder_partition_d5_spans() {
///         // bin every FFT line ω with partition_n_contains_line(span.index, ω)
///         …
///     }
/// ```
///
/// matching the spec table's row-order presentation without
/// open-coding the `1..=32` range or the descriptor lookup at every
/// reduction site.
///
/// The iterator emits **exactly 32** descriptors — one per recoverable
/// partition. The two boundary-table-gap edges that
/// [`coder_partition_d5_span`] returns `None` for (`n = 0` and
/// `n = 33`) are **not** emitted: a row-order walk of Table D.5 sees
/// the same boundary-table gaps the descriptor sees, so emitting either
/// edge would force the caller to filter back to `1..=32` immediately.
/// The iterator is the descriptor's natural row-walk and nothing else.
///
/// Implementation: a `Range<u16>` over `1..=32` mapped through
/// [`coder_partition_d5_span`]. The `.unwrap()` inside the map is
/// **infallible** for the iterated range — every `n ∈ 1..=32` is a
/// recoverable Table D.5 row by construction (pinned by Phase 2 step
/// 53 tests). The returned iterator is
/// `ExactSizeIterator + DoubleEndedIterator + Clone` via the
/// `Range`'s trait passthrough, but its public surface is kept generic
/// (`impl Iterator<Item = CoderPartitionD5Span>`) so future
/// implementation changes don't break consumers.
///
/// Provenance: only the Phase 2 step 53 descriptor
/// [`coder_partition_d5_span`] and its underlying Table D.5
/// transcription in
/// `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
/// §"Table D.5 - Layer I and Layer II coder partition table" are
/// consulted; the row-order walk is the spec table's own ordering.
#[inline]
pub fn coder_partition_d5_spans() -> impl Iterator<Item = CoderPartitionD5Span> {
    // `1..=32` matches the descriptor's recoverable range exactly;
    // every `coder_partition_d5_span(n)` is `Some(_)` for that range
    // by step 53 construction so the `.unwrap()` is infallible.
    (1_u16..=32).map(|n| coder_partition_d5_span(n).expect("n ∈ 1..=32 is recoverable"))
}

/// Inverse lookup over Annex D Table D.5 — given an FFT line `omega`,
/// return the index `n` of the **lowest** partition whose inclusive
/// boundary range `[ωlow_n, ωhigh_n]` contains it.
///
/// Phase 2 step 53 (r252) composed each partition's three Table D.5
/// columns into a [`CoderPartitionD5Span`] descriptor with the
/// inclusive boundary pair `(ωlow_n, ωhigh_n)`. Phase 2 step 54 (r253)
/// lifted the membership inequality to the named predicate
/// [`partition_n_contains_line`]. Phase 2 step 55 (r254) added a
/// row-order iterator [`coder_partition_d5_spans`] over the
/// recoverable descriptors. This step 56 accessor closes the inverse
/// direction: instead of asking "is line `ω` in partition `n`?", the
/// Model 1 / Model 2 reduction asks "given line `ω`, which partition
/// `n` does it land in?" — the natural question when walking the
/// FFT-line domain and binning each line into its partition.
///
/// The accessor returns `Some(n)` with `n ∈ 1..=32` for any
/// `omega ∈ [1, 513]` (the in-band FFT-line domain Table D.5 covers)
/// and `None` for any `omega` outside that band.
///
/// **Shared-boundary disambiguation.** Phase 2 step 50 (r249) pinned
/// the column-heading `ωlow_{n+1} / ωhigh_n` dual reading, and Phase 2
/// step 54 (r253) pinned the inclusive-on-both-ends boundary
/// semantics: every shared boundary line is a member of **both**
/// partition `n` (as its `ωhigh_n`) and partition `n + 1` (as its
/// `ωlow_{n+1}`). When the caller asks the inverse question, this
/// accessor returns the **lower** index `n` — the first ascending-row
/// match — matching both the spec table's row-order presentation
/// (the boundary cell is printed on row `n`'s line, not on row
/// `n + 1`'s) and the row-order iterator's ascending walk pinned by
/// Phase 2 step 55. The downstream reduction is free to walk
/// partitions in either direction; the "lowest partition first"
/// convention is the unique deterministic choice that does not
/// double-count the boundary lines.
///
/// The accessor is **pure** with respect to the spec — it is exactly
/// `coder_partition_d5_spans().find(|s| s.omega_low <= omega &&
/// omega <= s.omega_high).map(|s| s.index)`. No arithmetic beyond the
/// inequality on each descriptor's pre-computed boundaries is
/// introduced; no division, no modulus, no bit-tricks. The row-order
/// iterator + first-match composition is the spec table's own
/// row-by-row search, hoisted into a single accessor so the reduction
/// doesn't re-implement it at every site.
///
/// Complexity is `O(32)` worst case (a linear scan of the table). For
/// a Model 1 / Model 2 reduction sweeping all 513 FFT lines this is
/// `O(513 × 32) ≈ 16 K` boundary comparisons — well below any
/// performance threshold worth complicating the accessor over. A
/// stride-based `O(1)` variant is theoretically derivable from the
/// mostly-16-wide partition stride (Phase 2 step 49 pinned
/// `CODER_PARTITION_D5_STRIDE = 16`), but the table's first row
/// (`width = 0`, single-line partition 0 with `ωhigh_0` absent) and
/// the dual-role `ωlow_{n+1} / ωhigh_n` boundary column would force
/// the closed form to encode three special cases against the simple
/// linear scan's one (the in/out-of-band check). This accessor stays
/// with the spec-faithful row-walk; the `O(1)` variant can be added
/// later as a separate accessor if a profile ever demands it.
///
/// Provenance: only the Phase 2 step 55 iterator
/// [`coder_partition_d5_spans`] and (through it) the Phase 2 step 53
/// descriptor [`coder_partition_d5_span`] and its underlying Table D.5
/// transcription in
/// `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
/// §"Table D.5 - Layer I and Layer II coder partition table" are
/// consulted; the lowest-index-first convention is the spec table's
/// row-order presentation, already pinned by Phase 2 steps 50 and 55.
#[inline]
#[must_use]
pub fn first_partition_containing_line(omega: u16) -> Option<u16> {
    coder_partition_d5_spans()
        .find(|s| s.omega_low <= omega && omega <= s.omega_high)
        .map(|s| s.index)
}

/// Inclusive FFT-line iterator over Annex D Table D.5 partition `n` —
/// yields every `omega ∈ [ωlow_n, ωhigh_n]` in ascending order.
///
/// Phase 2 step 51 (r250) exposed each partition's `(ωlow_n, ωhigh_n)`
/// boundary pair via [`coder_partition_d5_line_range`]. Phase 2 step 53
/// (r252) composed those boundaries with the `width_n` value into the
/// [`CoderPartitionD5Span`] descriptor. Phase 2 step 54 (r253) lifted
/// the membership inequality on that pair to the named predicate
/// [`partition_n_contains_line`]; Phase 2 step 55 (r254) added the
/// row-order iterator [`coder_partition_d5_spans`] over the recoverable
/// descriptors; Phase 2 step 56 (r255) closed the inverse lookup with
/// [`first_partition_containing_line`]. This step 57 accessor closes
/// the per-partition FFT-line walk: given partition `n`, yield every
/// in-range `omega` so the downstream Model 1 / Model 2 reduction can
/// write
///
/// ```text
///     for span in coder_partition_d5_spans() {
///         let acc = coder_partition_d5_omega_iter(span.index)
///             .expect("span.index ∈ 1..=32")
///             .map(|omega| per_line_value(omega))
///             .sum::<f64>();
///         …
///     }
/// ```
///
/// matching the spec's per-partition sum-over-lines pattern (clause D.1
/// Step 7's `Σ_{j ∈ partition}` form) without open-coding either the
/// `ωlow_n..=ωhigh_n` range or the lookup at every reduction site.
///
/// Returns `Some(ωlow_n..=ωhigh_n)` (a [`core::ops::RangeInclusive`])
/// for any `n ∈ 1..=32` — the same recoverable range as the descriptor
/// itself — and **`None`** for any `n` outside that range. The two
/// edge cases inherit from the descriptor:
///
/// * `n = 0` — partition 0's lower boundary `ωlow_0` is not in
///   Table D.5; without a `ωlow_n`, the range is undefined. `None`.
/// * `n = 33` — neither row 33's boundary nor its `width_n` cell
///   exists in Table D.5. `None`.
///
/// **Boundary semantics.** The iterator is **inclusive on both ends**,
/// matching the dual-role `ωlow_{n+1} / ωhigh_n` reading Phase 2
/// step 50 (r249) pinned and the inclusive-on-both-ends membership
/// predicate Phase 2 step 54 (r253) named. Two consecutive partitions
/// `n` and `n + 1` therefore both emit the shared boundary line
/// `ω = ωhigh_n = ωlow_{n+1}`; a caller that wants to bin every
/// FFT line into exactly one partition (no double-counting) should
/// use [`first_partition_containing_line`] (which deterministically
/// assigns each shared boundary line to the *lower* partition).
/// A caller that wants the spec's per-partition sum-over-lines (where
/// the shared boundary line *does* contribute to both partitions'
/// reductions per the Annex D Step 7 `Σ_{j ∈ partition}` reading) uses
/// this iterator directly.
///
/// Implementation: `coder_partition_d5_line_range(n).map(|(lo, hi)|
/// lo..=hi)` — a pure composition of the step-51 range accessor and
/// `RangeInclusive::new`, with no arithmetic introduced. Complexity is
/// `O(ωhigh_n − ωlow_n + 1)`; for the spec's mostly-16-wide partitions
/// this is `O(16)` per partition, `O(513)` across the whole table.
///
/// Provenance: only the Phase 2 step 51 accessor
/// [`coder_partition_d5_line_range`] and its underlying Table D.5
/// transcription in
/// `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
/// §"Table D.5 - Layer I and Layer II coder partition table" are
/// consulted; the inclusive-on-both-ends reading is the spec's,
/// pinned by Phase 2 step 50 (r249) and step 54 (r253).
#[inline]
#[must_use]
pub fn coder_partition_d5_omega_iter(n: u16) -> Option<core::ops::RangeInclusive<u16>> {
    let (low, high) = coder_partition_d5_line_range(n)?;
    Some(low..=high)
}

// =====================================================================
// Annex D Model 1 — §D.1 Step 8 per-partition LTg minimum reduction
// (Phase 2 step 58 / r257).
//
// Spec context (clause D.1, ISO/IEC 11172-3:1993, informative annex):
//
//   Step 7 (already landed at Phase 2 step 44 / r219 as
//   `global_masking_threshold_db`) produces the per-FFT-line global
//   masking threshold `LTg(i)` (dB) by summing the energy contributions
//   of every in-range tonal/non-tonal masker with the threshold in
//   quiet `LTq(i)`.
//
//   The Layer I / Layer II coder partition table (Table D.5, transcribed
//   at Phase 2 step 49 / r248) groups the 513 FFT lines into 32 coder
//   partitions `n ∈ 1..=32` (partition `0` is a single-line degenerate
//   carrying `ωlow_0` only and is not used as a reduction target — see
//   the §D.1 Step 4 critical-band-boundary reading at Phase 2 step 50).
//
//   Step 8 reduces the per-line `LTg(i)` over each coder partition by
//   taking the minimum:
//
//       LTmin_n = min_{ω ∈ [ωlow_n, ωhigh_n]} LTg(ω)   dB
//
//   The minimum is the most-conservative per-partition perceptual
//   threshold — a single FFT line dipping below the partition's average
//   threshold pulls the whole partition's bit-allocation budget down to
//   that line's level (the encoder cannot afford to leak any
//   sub-threshold noise into the partition without becoming audible).
//   This is the value the Layer I / Layer II bit-allocation loop (the
//   Layer III analogue is the outer-loop SNR budget) consumes per
//   partition.
//
// Composition rather than introduction: this step is a strict
// composition of the Phase 2 step 57 per-partition FFT-line iterator
// `coder_partition_d5_omega_iter` (which yields every `ω ∈ [ωlow_n,
// ωhigh_n]` in ascending order) with the caller-supplied `LTg(ω)`
// callback. No spec arithmetic is introduced — only the per-line
// minimum fold over the recoverable line range. The Step 7 LTg
// callback itself is provided by the caller (typically a closure
// closing over the static masker list + threshold-in-quiet curve);
// this accessor stays pure with respect to the masker selection
// pipeline (Steps 1-5), which remain blocked on the PNG-only Table
// D.1 / D.2 / D.3 transcription gap. Once Steps 1-5 land the
// concrete `LTg(ω)` closure will be the one produced by Step 7's
// `global_masking_threshold_db` applied per line.
//
// Boundary semantics: the underlying step 57 iterator is inclusive
// on both ends and emits the shared boundary line `ωhigh_n =
// ωlow_{n+1}` to both adjacent partitions `n` and `n + 1`. That
// matches the spec's per-partition reduction reading (a shared
// boundary line legitimately enters both partitions' minimums),
// and `coder_partition_d5_ltg_min` inherits the semantics
// unchanged — a sharp dip located exactly on a shared boundary
// reduces the LTmin of both adjacent partitions, which is the
// conservative-bit-allocation reading the spec intends.
//
// Shared-boundary disambiguation: a caller that wants every FFT
// line to enter exactly one partition's minimum (single-assignment
// binning, no shared-boundary double-influence) uses the step 56
// inverse accessor `first_partition_containing_line` to bin per
// line, then folds per partition. This module exposes both
// reductions because the spec text is silent on which reading is
// "correct" — both are defensible, and the choice depends on the
// downstream bit-allocation loop's preference. The step 58 default
// (this accessor) matches the per-partition `Σ_{ω ∈ partition}`
// composition pattern Phase 2 step 57 (r256) wired into Step 7's
// own sum-over-lines.
// =====================================================================

/// §D.1 Step 8 per-partition minimum global masking threshold
/// `LTmin_n` (dB) for the Layer I / Layer II coder partition
/// `n ∈ 1..=32`. Reduces the caller-supplied per-FFT-line global
/// threshold `ltg_per_line(ω)` (from Step 7's
/// [`global_masking_threshold_db`], applied per line) over every
/// `ω ∈ [ωlow_n, ωhigh_n]` by taking the minimum:
///
/// ```text
/// LTmin_n = min_{ω ∈ [ωlow_n, ωhigh_n]} LTg(ω)   dB
/// ```
///
/// The reduction is the spec's most-conservative per-partition
/// threshold reading — a single FFT line dipping below the partition's
/// average threshold pulls the whole partition's bit-allocation budget
/// down to that line's level (the encoder cannot afford to leak
/// sub-threshold noise into the partition without becoming audible).
/// This is the value the Layer I / Layer II bit-allocation loop
/// consumes per partition (Layer III's outer-loop SNR budget is the
/// analogue).
///
/// Returns:
///
/// * `Some(LTmin_n)` for any `n ∈ 1..=32` — the inclusive minimum of
///   `ltg_per_line(ω)` over every `ω ∈ [ωlow_n, ωhigh_n]` (table-
///   wide bounds [`coder_partition_d5_line_range`] exposes the
///   `(ωlow_n, ωhigh_n)` pair). The minimum is taken in IEEE-754 f64
///   total-order semantics through `f64::min`, so any `NaN` value
///   passed through by the caller propagates per the standard rules
///   — the caller is responsible for ensuring `ltg_per_line` is
///   finite (the Step 7 `global_masking_threshold_db` guarantees this
///   for any non-empty masker list with `LTq(i)` finite, the only
///   reachable path in practice).
/// * `None` for any `n` outside `1..=32` — the two edge cases inherit
///   from [`coder_partition_d5_omega_iter`]:
///   * `n = 0` — partition 0's lower boundary `ωlow_0` is not in
///     Table D.5; without a `ωlow_n`, the reduction range is
///     undefined.
///   * `n = 33` — neither row 33's boundary nor its `width_n` cell
///     exists in Table D.5; the reduction range is undefined.
///
/// **Boundary semantics.** The reduction is inclusive on both ends,
/// matching the per-partition sum-over-lines pattern Phase 2 step 57
/// (r256) wired into Step 7's own `Σ_{ω ∈ partition}` form. Two
/// consecutive partitions `n` and `n + 1` therefore both consider the
/// shared boundary line `ω = ωhigh_n = ωlow_{n+1}` in their minimum
/// — a sharp dip located exactly on a shared boundary reduces both
/// adjacent partitions' `LTmin`, which is the conservative-bit-
/// allocation reading the spec intends. A caller that wants every
/// FFT line to enter exactly one partition's minimum (single-
/// assignment binning, no shared-boundary double-influence) uses
/// [`first_partition_containing_line`] to bin per line, then folds
/// per partition outside this accessor.
///
/// **Implementation.** A pure composition of
/// [`coder_partition_d5_omega_iter`] (Phase 2 step 57) and
/// `Iterator::map ∘ Iterator::fold(f64::INFINITY, f64::min)`. No spec
/// arithmetic is introduced — only the per-line minimum fold over the
/// recoverable line range. The `f64::INFINITY` seed pairs with
/// `f64::min` to produce the per-partition minimum for any partition
/// with at least one line (every `n ∈ 1..=32` has `width_n ≥ 0` and
/// produces at least one inclusive line per
/// [`coder_partition_d5_omega_iter`]'s non-empty range guarantee).
///
/// Complexity is `O(ωhigh_n − ωlow_n + 1)` per partition, dominated by
/// the caller's `ltg_per_line` cost per line. For the spec's mostly-
/// 16-wide partitions this is `O(16)` per partition, `O(513)` across
/// the whole table (matching the per-partition iterator's own
/// complexity).
///
/// Provenance: only the Phase 2 step 57 per-partition iterator
/// [`coder_partition_d5_omega_iter`] (and through it the Phase 2 step
/// 51 line-range accessor and its underlying Table D.5 transcription
/// in `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
/// §"Table D.5 - Layer I and Layer II coder partition table") and
/// the Step 7 `LTg(ω)` reading (Phase 2 step 44 /
/// [`global_masking_threshold_db`]) are consulted. The minimum-
/// reduction reading is the spec's per Annex D Step 8 (informative
/// Model 1 reduction); no external implementation was read.
#[inline]
#[must_use]
pub fn coder_partition_d5_ltg_min<F>(n: u16, ltg_per_line: F) -> Option<f64>
where
    F: Fn(u16) -> f64,
{
    let range = coder_partition_d5_omega_iter(n)?;
    Some(range.map(ltg_per_line).fold(f64::INFINITY, f64::min))
}

// =====================================================================
// Annex D Model 1 — §D.1 Step 8 row-order LTmin vector over Table D.5
// (Phase 2 step 59 / r258).
//
// Spec context (clause D.1, ISO/IEC 11172-3:1993, informative annex):
//
//   Step 8 (Phase 2 step 58 / r257) reduces the per-FFT-line global
//   masking threshold `LTg(ω)` (dB) over a single coder partition
//   `n ∈ 1..=32` by taking the minimum
//
//       LTmin_n = min_{ω ∈ [ωlow_n, ωhigh_n]} LTg(ω)   dB
//
//   The Layer I / Layer II bit-allocation loop consumes the **full
//   vector** `[LTmin_1, LTmin_2, …, LTmin_32]` per frame, walking the
//   32 coder partitions in row order (the spec table's
//   ascending-`n` presentation, pinned at row-order by Phase 2
//   step 55 / r254's `coder_partition_d5_spans`). The Layer III
//   outer-loop SNR-budget analogue consumes the same per-partition
//   vector.
//
// Composition rather than introduction: this step is a strict
// composition of the Phase 2 step 55 row-order partition iterator
// `coder_partition_d5_spans` (which yields every recoverable
// `n ∈ 1..=32` in ascending order) with the Phase 2 step 58
// per-partition reducer `coder_partition_d5_ltg_min`. No new spec
// arithmetic is introduced — only the broadcast of step 58's single-
// partition reduction across all 32 recoverable partitions. The
// `LTg(ω)` callback is the caller's, keeping this accessor pure
// with respect to the masker selection pipeline (Steps 1-5), which
// remain blocked on the PNG-only Table D.1 / D.2 / D.3 transcription
// gap.
//
// The output is a 32-element `[f64; 32]` indexed 0-based, with
// element `i` holding `LTmin_{i + 1}` (the spec's 1-based `n` in
// 0-based array form). This matches the spec's row-order
// presentation of Table D.5 (partition 0 is the degenerate
// `width_n = 0` single-line row excluded from the reduction
// targets — see Phase 2 step 58's `None`-on-`n = 0` clause) and is
// the natural index the downstream Layer I / Layer II bit-allocation
// loop walks against.
//
// Boundary semantics inherit from Phase 2 step 58 unchanged — the
// reduction is inclusive on both ends, so a sharp dip on a shared
// boundary `ω = ωhigh_n = ωlow_{n+1}` enters both adjacent
// partitions' `LTmin`. A caller that wants single-assignment binning
// uses the step 56 inverse accessor `first_partition_containing_line`
// to bin each FFT line into exactly one partition before folding.
// =====================================================================

/// §D.1 Step 8 row-order minimum global masking threshold vector
/// `[LTmin_1, LTmin_2, …, LTmin_32]` (dB) for every Layer I /
/// Layer II coder partition `n ∈ 1..=32`. Element `i` of the
/// returned `[f64; 32]` holds `LTmin_{i + 1}` (the spec's 1-based
/// `n` in 0-based array form):
///
/// ```text
/// LTmin_n = min_{ω ∈ [ωlow_n, ωhigh_n]} LTg(ω)   dB
/// ```
///
/// The vector is the per-frame input the Layer I / Layer II bit-
/// allocation loop consumes (Layer III's outer-loop SNR-budget
/// analogue is the same per-partition vector). Each element is the
/// most-conservative per-partition threshold — a single FFT line
/// dipping below the partition's average threshold pulls the whole
/// partition's `LTmin` down to that line's level (the encoder cannot
/// afford to leak sub-threshold noise into the partition without
/// becoming audible).
///
/// **Index convention.** 0-based on the returned slice;
/// `out[i] = LTmin_{i + 1}`. The spec's 1-based partition index
/// `n ∈ 1..=32` maps to array index `i = n - 1 ∈ 0..=31`. Partition
/// 0 (the degenerate single-line `width_n = 0` row carrying `ωlow_0`
/// only) is excluded from the vector — Phase 2 step 58 returns
/// `None` for `n = 0` because the reduction range is undefined
/// without a `ωlow_n` boundary in Table D.5. The downstream bit-
/// allocation loop walks partitions `1..=32` and does not consult
/// partition 0, matching the spec's coder-partition usage.
///
/// **Composition.** A pure broadcast of Phase 2 step 58's per-
/// partition reducer [`coder_partition_d5_ltg_min`] across the
/// Phase 2 step 55 row-order iterator [`coder_partition_d5_spans`].
/// No spec arithmetic is introduced — only the broadcast of step
/// 58's single-partition reduction across all 32 recoverable
/// partitions, which is the row-order vector form the Layer I /
/// Layer II bit-allocation loop consumes per frame. The `LTg(ω)`
/// callback is the caller's, typically a closure closing over the
/// static masker list + threshold-in-quiet curve — keeping this
/// accessor pure with respect to the masker selection pipeline
/// (Steps 1-5), which remain blocked on the PNG-only Table D.1 /
/// D.2 / D.3 transcription gap. Once Steps 1-5 land the concrete
/// `LTg(ω)` closure will be the one produced by Step 7's
/// `global_masking_threshold_db` applied per line.
///
/// **Boundary semantics.** Inherits Phase 2 step 58's inclusive-on-
/// both-ends reduction semantics unchanged: a sharp dip on a shared
/// boundary `ω = ωhigh_n = ωlow_{n+1}` enters **both** adjacent
/// partitions' `LTmin` (the conservative-bit-allocation reading the
/// spec intends, where two adjacent partitions both see the shared
/// boundary line). A caller that wants every FFT line to enter
/// exactly one partition's reduction (single-assignment binning) uses
/// [`first_partition_containing_line`] to bin per line before
/// folding outside this accessor.
///
/// **Implementation.** A pure composition of
/// [`coder_partition_d5_spans`] (Phase 2 step 55) and
/// [`coder_partition_d5_ltg_min`] (Phase 2 step 58): for each
/// recoverable span the function calls the step 58 reducer with
/// the caller's `ltg_per_line` callback. The `.expect("…")` on the
/// step 58 result is infallible by construction — every span
/// emitted by [`coder_partition_d5_spans`] has `index ∈ 1..=32`,
/// the exact range step 58 returns `Some(_)` over. Complexity is
/// `O(513)` per frame total — `Σ_{n=1..=32} (ωhigh_n − ωlow_n + 1)`
/// summed over the table — dominated by the caller's `ltg_per_line`
/// cost.
///
/// Provenance: only the Phase 2 step 58 per-partition reducer
/// [`coder_partition_d5_ltg_min`] and the Phase 2 step 55 row-order
/// iterator [`coder_partition_d5_spans`] (and through them the
/// underlying Table D.5 transcription in
/// `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
/// §"Table D.5 - Layer I and Layer II coder partition table") are
/// consulted. The row-order broadcast reading is the spec's per
/// Annex D Step 8 (informative Model 1 reduction); no external
/// implementation was read.
#[must_use]
pub fn coder_partition_d5_ltg_min_row_order<F>(ltg_per_line: F) -> [f64; 32]
where
    F: Fn(u16) -> f64,
{
    let mut out = [f64::INFINITY; 32];
    for span in coder_partition_d5_spans() {
        // `span.index ∈ 1..=32` by step 55 construction; the step 58
        // reducer is `Some(_)` exactly over that range so the
        // `.expect` is infallible.
        let i = (span.index - 1) as usize;
        out[i] = coder_partition_d5_ltg_min(span.index, &ltg_per_line)
            .expect("n ∈ 1..=32 is recoverable by step 58 / Table D.5");
    }
    out
}

// =====================================================================
// Annex D Model 1 — §D.1 Step 8 row-order width vector over Table D.5
// (Phase 2 step 60 / r259).
//
// Spec context (clause D.1, ISO/IEC 11172-3:1993, informative annex):
//
//   Table D.5 prints three columns per row — the partition index `n`,
//   the dual-role partition-boundary FFT-line cell `ωlow_{n+1} / ωhigh_n`,
//   and the `width_n` column. Phase 2 step 52 (r251) exposed the
//   `width_n` value of a single partition `n ∈ 1..=32` via
//   `coder_partition_d5_width`. The verbatim transcribed values from
//   the §"Table D.5 - Layer I and Layer II coder partition table"
//   render are:
//
//       n ∈ 1..=12 → width_n = 0
//       n ∈ 13..=32 → width_n = 1
//
//   The Layer I / Layer II bit-allocation loop consumes the **full
//   vector** `[width_1, width_2, …, width_32]` per frame alongside the
//   row-order LTmin vector landed by Phase 2 step 59 (r258), walking
//   the 32 coder partitions in row order (the spec table's
//   ascending-`n` presentation, pinned at iteration order by Phase 2
//   step 55 / r254's `coder_partition_d5_spans`). The width vector
//   pairs each `LTmin_n` element with its partition's `width_n`
//   value, and the downstream bit-allocation step pairs the two
//   columns per row.
//
// Composition rather than introduction: this step is a strict
// composition of the Phase 2 step 55 row-order partition iterator
// `coder_partition_d5_spans` (which yields every recoverable
// `n ∈ 1..=32` in ascending order) with the Phase 2 step 52 per-
// partition `width_n` accessor `coder_partition_d5_width`. No new
// spec arithmetic is introduced — only the broadcast of step 52's
// single-partition lookup across all 32 recoverable partitions. The
// output is a **pure constant** of Table D.5 (the `width_n` column
// has no run-time inputs) — unlike step 59's `LTmin` vector, which
// closes over a caller-supplied `LTg(ω)` callback, the width vector
// is fully determined by the static table.
//
// The output is a 32-element `[u16; 32]` indexed 0-based, with
// element `i` holding `width_{i + 1}` (the spec's 1-based `n` in
// 0-based array form). This matches the spec's row-order
// presentation of Table D.5 and the same index convention pinned by
// step 59. Partition 0 (the degenerate single-line `width_n = 0` row
// carrying `ωlow_0` only) is excluded from the vector for index
// consistency with step 59 — the downstream bit-allocation loop
// walks partitions `1..=32` and does not consult partition 0,
// matching the spec's coder-partition usage.
// =====================================================================

/// §D.1 Step 8 row-order `width_n` vector
/// `[width_1, width_2, …, width_32]` for every Layer I / Layer II
/// coder partition `n ∈ 1..=32`. Element `i` of the returned
/// `[u16; 32]` holds `width_{i + 1}` (the spec's 1-based `n` in
/// 0-based array form):
///
/// ```text
/// out[0]  = width_1  = 0
/// out[1]  = width_2  = 0
/// …
/// out[11] = width_12 = 0
/// out[12] = width_13 = 1
/// …
/// out[31] = width_32 = 1
/// ```
///
/// The vector is the static per-frame input the Layer I / Layer II
/// bit-allocation loop consumes alongside the row-order LTmin vector
/// landed by Phase 2 step 59 (r258), pairing each `LTmin_n` element
/// with its partition's `width_n` value at the same array index.
///
/// **Index convention.** 0-based on the returned slice;
/// `out[i] = width_{i + 1}`. The spec's 1-based partition index
/// `n ∈ 1..=32` maps to array index `i = n - 1 ∈ 0..=31`. Partition
/// 0 (the degenerate single-line `width_n = 0` row carrying `ωlow_0`
/// only) is excluded from the vector for index consistency with
/// Phase 2 step 59 (r258)'s LTmin vector. The downstream bit-
/// allocation loop walks partitions `1..=32` and does not consult
/// partition 0, matching the spec's coder-partition usage.
///
/// **Composition.** A pure broadcast of Phase 2 step 52's per-
/// partition `width_n` accessor [`coder_partition_d5_width`] across
/// the Phase 2 step 55 row-order iterator [`coder_partition_d5_spans`].
/// No spec arithmetic is introduced — only the broadcast of step
/// 52's single-partition lookup across all 32 recoverable
/// partitions, which is the row-order vector form the Layer I /
/// Layer II bit-allocation loop pairs with the step 59 LTmin vector
/// per frame. Unlike step 59, this accessor has no run-time inputs:
/// the `width_n` column is a static property of Table D.5, so the
/// returned vector is the same `[u16; 32]` on every call.
///
/// **Constant values.** Per the Table D.5 transcription the vector
/// is exactly twelve zeros followed by twenty ones:
///
/// ```text
/// [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
///  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
/// ```
///
/// The single 0 → 1 transition lies between array indices 11 and 12
/// (partitions 12 and 13), pinned by Phase 2 step 52's transcription
/// rule "rows 0..=12 have width 0; rows 13..=32 have width 1" and
/// the step 55 row-order iterator's ascending-`n` ordering.
///
/// **Implementation.** A pure composition of
/// [`coder_partition_d5_spans`] (Phase 2 step 55) and
/// [`coder_partition_d5_width`] (Phase 2 step 52): for each
/// recoverable span the function calls the step 52 lookup with the
/// span's `index`. The `.expect("…")` on the step 52 result is
/// infallible by construction — every span emitted by
/// [`coder_partition_d5_spans`] has `index ∈ 1..=32`, the exact
/// range step 52 returns `Some(_)` over. Complexity is `O(32)` per
/// call — one lookup per span — with no per-line work (the
/// `width_n` column does not span FFT lines).
///
/// Provenance: only the Phase 2 step 52 per-partition `width_n`
/// accessor [`coder_partition_d5_width`] and the Phase 2 step 55
/// row-order iterator [`coder_partition_d5_spans`] (and through them
/// the underlying Table D.5 transcription in
/// `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
/// §"Table D.5 - Layer I and Layer II coder partition table") are
/// consulted. The row-order broadcast reading is the spec's per
/// Annex D Step 8 (informative Model 1 reduction) row-by-row
/// presentation; no external implementation was read.
#[must_use]
pub fn coder_partition_d5_width_row_order() -> [u16; 32] {
    let mut out = [0u16; 32];
    for span in coder_partition_d5_spans() {
        // `span.index ∈ 1..=32` by step 55 construction; the step 52
        // accessor is `Some(_)` exactly over that range so the
        // `.expect` is infallible.
        let i = (span.index - 1) as usize;
        out[i] = coder_partition_d5_width(span.index)
            .expect("n ∈ 1..=32 is recoverable by step 52 / Table D.5");
    }
    out
}

// =====================================================================
// Annex D Model 1 — §D.1 Step 8 paired (LTmin_n, width_n) row-order
// vector over Table D.5 (Phase 2 step 61 / r260).
//
// Spec context (clause D.1, ISO/IEC 11172-3:1993, informative annex):
//
//   Step 8 produces, per coder partition n ∈ 1..=32, a per-partition
//   minimum global masking threshold LTmin_n (dB) and reads the
//   partition's width column width_n (0 for n ∈ 1..=12; 1 for
//   n ∈ 13..=32). The Layer I / Layer II bit-allocation loop walks the
//   32 partitions in row order (the spec table's ascending-n
//   presentation, pinned at iteration order by Phase 2 step 55 /
//   r254's `coder_partition_d5_spans`) and at every row consumes
//   **both** columns paired — the LTmin_n value drives the per-
//   partition target threshold, the width_n column flags whether the
//   partition spans more than one Layer I / Layer II coder partition
//   row (width_n = 1) or carries a single boundary row (width_n = 0).
//
//   Phase 2 step 59 (r258) exposed the row-order LTmin vector
//   `[LTmin_1, …, LTmin_32]`. Phase 2 step 60 (r259) exposed the row-
//   order width vector `[width_1, …, width_32]`. The bit-allocation
//   loop pairs the two at the call site as the per-row tuple
//   `(LTmin_n, width_n)`.
//
// Composition rather than introduction: this step is a strict
// composition of the Phase 2 step 59 row-order LTmin reducer
// `coder_partition_d5_ltg_min_row_order` and the Phase 2 step 60
// row-order width vector `coder_partition_d5_width_row_order`. It
// pairs the two columns at the same array index without introducing
// any spec arithmetic — the LTmin column closes over the caller's
// `LTg(ω)` callback (the run-time-dependent half of the input pair)
// and the width column is the static Table D.5 column. The output is
// the per-frame paired input the Layer I / Layer II bit-allocation
// loop consumes in lockstep.
//
// The output is a 32-element `[CoderPartitionD5Reduction; 32]` indexed
// 0-based, with element `i` holding the `(LTmin_{i + 1}, width_{i + 1})`
// pair (the spec's 1-based `n` in 0-based array form). This matches
// the spec's row-order presentation of Table D.5 and the same index
// convention pinned by steps 59 and 60. Partition 0 (the degenerate
// single-line `width_n = 0` row carrying `ωlow_0` only) is excluded
// from the vector — Phase 2 step 58 returns `None` for `n = 0`
// because the reduction range is undefined without a `ωlow_n`
// boundary in Table D.5. The downstream bit-allocation loop walks
// partitions `1..=32` and does not consult partition 0, matching the
// spec's coder-partition usage.
//
// Boundary semantics inherit from Phase 2 step 59 unchanged — the
// `LTmin_n` reduction is inclusive on both ends so a sharp dip on a
// shared boundary `ω = ωhigh_n = ωlow_{n+1}` enters both adjacent
// partitions' `LTmin`. The width column has no boundary semantics
// (it is a static per-row table value).
// =====================================================================

/// A single row of the Layer I / Layer II coder-partition bit-
/// allocation input: the per-partition minimum global masking
/// threshold `LTmin_n` (dB) paired with the partition's `width_n`
/// column (0 for `n ∈ 1..=12`; 1 for `n ∈ 13..=32`) at the same row
/// index in row-order Table D.5 presentation.
///
/// Produced by [`coder_partition_d5_reduction_row_order`] (Phase 2
/// step 61 / r260).
///
/// **Field semantics.** `ltmin_db` carries the inclusive minimum of
/// `LTg(ω)` (dB) over the partition's FFT-line range, as defined by
/// Phase 2 step 58's per-partition reducer
/// [`coder_partition_d5_ltg_min`]. `width_n` is the static Table D.5
/// column value read by Phase 2 step 52's per-partition accessor
/// [`coder_partition_d5_width`]. The two columns are orthogonal: the
/// LTmin column closes over the caller's `LTg(ω)` callback (run-
/// time-dependent), the width column is a pure constant of the
/// table.
#[derive(Clone, Copy, Debug)]
pub struct CoderPartitionD5Reduction {
    /// Per-partition minimum global masking threshold `LTmin_n` (dB),
    /// as reduced by Phase 2 step 58's [`coder_partition_d5_ltg_min`]
    /// applied to the caller's `LTg(ω)` callback over the partition's
    /// FFT-line range.
    pub ltmin_db: f64,
    /// Partition's `width_n` column from Table D.5 — `0` for
    /// `n ∈ 1..=12`, `1` for `n ∈ 13..=32`. The same value Phase 2
    /// step 52's [`coder_partition_d5_width`] returns for the
    /// partition.
    pub width_n: u16,
}

/// §D.1 Step 8 paired row-order `[(LTmin_n, width_n), …]` vector for
/// every Layer I / Layer II coder partition `n ∈ 1..=32`. Element
/// `i` of the returned `[CoderPartitionD5Reduction; 32]` holds the
/// `(LTmin_{i + 1}, width_{i + 1})` pair (the spec's 1-based `n` in
/// 0-based array form):
///
/// ```text
/// out[i].ltmin_db = min_{ω ∈ [ωlow_{i+1}, ωhigh_{i+1}]} LTg(ω)   dB
/// out[i].width_n  = width_{i + 1}   (∈ {0, 1})
/// ```
///
/// The vector is the per-frame paired input the Layer I / Layer II
/// bit-allocation loop consumes — at every row the loop reads both
/// columns together as the partition's "per-row brief" (target
/// threshold + width flag).
///
/// **Index convention.** 0-based on the returned slice; element `i`
/// holds `(LTmin_{i + 1}, width_{i + 1})`. The spec's 1-based partition
/// index `n ∈ 1..=32` maps to array index `i = n - 1 ∈ 0..=31`.
/// Partition 0 (the degenerate single-line `width_n = 0` row carrying
/// `ωlow_0` only) is excluded from the vector for index consistency
/// with Phase 2 steps 59 and 60. The downstream bit-allocation loop
/// walks partitions `1..=32` and does not consult partition 0,
/// matching the spec's coder-partition usage.
///
/// **Composition.** A pure index-aligned zip of Phase 2 step 59's
/// row-order LTmin vector [`coder_partition_d5_ltg_min_row_order`]
/// (closed over the caller's `LTg(ω)` callback) with Phase 2 step
/// 60's row-order width vector [`coder_partition_d5_width_row_order`].
/// No spec arithmetic is introduced — only the per-row pairing of
/// the two existing row-order columns at the same array index, which
/// is exactly the per-row input the Layer I / Layer II bit-
/// allocation loop reads.
///
/// **Caller cost.** The `LTg(ω)` callback is invoked exactly as many
/// times as Phase 2 step 59 invokes it (one call per FFT line in
/// `Σ_{n=1..=32} (ωhigh_n − ωlow_n + 1)` summed over the table);
/// the width vector adds no callback invocations.
///
/// **Boundary semantics.** Inherits Phase 2 step 59's inclusive-on-
/// both-ends reduction semantics unchanged for the LTmin column — a
/// sharp dip on a shared boundary `ω = ωhigh_n = ωlow_{n+1}` enters
/// **both** adjacent partitions' `LTmin`. The width column has no
/// boundary semantics (it is a static per-row table value).
///
/// **Implementation.** Calls [`coder_partition_d5_ltg_min_row_order`]
/// once (folding the caller's callback over every recoverable
/// partition's FFT-line range) and [`coder_partition_d5_width_row_order`]
/// once (the static width column), then zips the two into the paired
/// output at the same array index. The zip is a strict-composition
/// pairing — neither column influences the other's computation.
///
/// Provenance: only the Phase 2 step 59 row-order LTmin reducer
/// [`coder_partition_d5_ltg_min_row_order`] and the Phase 2 step 60
/// row-order width vector [`coder_partition_d5_width_row_order`]
/// (and through them the Phase 2 step 58 per-partition reducer
/// [`coder_partition_d5_ltg_min`], the Phase 2 step 52 per-partition
/// width accessor [`coder_partition_d5_width`], and the underlying
/// Table D.5 transcription in
/// `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
/// §"Table D.5 - Layer I and Layer II coder partition table") are
/// consulted. The paired-row-order reading is the spec's per Annex D
/// Step 8 (informative Model 1 reduction) row-by-row presentation;
/// no external implementation was read.
#[must_use]
pub fn coder_partition_d5_reduction_row_order<F>(ltg_per_line: F) -> [CoderPartitionD5Reduction; 32]
where
    F: Fn(u16) -> f64,
{
    let ltmin = coder_partition_d5_ltg_min_row_order(ltg_per_line);
    let widths = coder_partition_d5_width_row_order();
    let mut out = [CoderPartitionD5Reduction {
        ltmin_db: f64::INFINITY,
        width_n: 0,
    }; 32];
    for i in 0..32 {
        out[i] = CoderPartitionD5Reduction {
            ltmin_db: ltmin[i],
            width_n: widths[i],
        };
    }
    out
}

// =====================================================================
// Annex D Model 1 — §D.1 Step 8 width-gated split of the row-order
// paired vector over Table D.5 (Phase 2 step 62 / r261).
//
// Spec context (clause D.2, ISO/IEC 11172-3:1993, informative annex):
//
//   Table D.5's `width_n` column is binary: `width_n = 0` for
//   `n ∈ 1..=12` and `width_n = 1` for `n ∈ 13..=32`. The two halves
//   of the table are not interchangeable from the Layer I / Layer II
//   bit-allocation loop's perspective — the spec table prints the
//   `width_n` column precisely so the bit-allocation step can branch
//   on it per row. The lower block (`width_n = 0`, partitions 1..=12)
//   covers the FFT-line range `ω ∈ 1..=193` at single-line stride,
//   while the upper block (`width_n = 1`, partitions 13..=32) covers
//   `ω ∈ 194..=513` at the wider stride implied by the table's
//   boundary cadence. The bit-allocation loop reads the lower-block
//   `LTmin_n` values as the narrow-band per-partition target and the
//   upper-block `LTmin_n` values as the wide-band target.
//
//   Phase 2 step 61 (r260) exposed the row-order paired vector
//   `[(LTmin_n, width_n); 32]`. The next narrow step is to surface
//   the **width-gated split** of that vector: the contiguous prefix
//   of rows with `width_n = 0` (the lower block) and the contiguous
//   suffix with `width_n = 1` (the upper block). The split is fully
//   determined by Table D.5's static `width_n` column and is the
//   row-order partitioning the bit-allocation loop branches on.
//
// Composition rather than introduction: this step is a strict
// composition of the Phase 2 step 61 row-order paired-vector
// accessor `coder_partition_d5_reduction_row_order` with the static
// Table D.5 `width_n` transcription (twelve zeros followed by
// twenty ones — pinned by Phase 2 step 60's row-order width vector
// `coder_partition_d5_width_row_order`). No spec arithmetic is
// introduced — only the partitioning of the 32-row paired vector
// into its two width-bands at the same array indices the paired
// vector already uses. The split point is constant (12, by the
// width column's single 0 → 1 transition between array indices 11
// and 12 / partitions 12 and 13).
//
// The split is exposed as a struct holding two by-value subarrays
// (12 elements for the lower block, 20 elements for the upper
// block), each preserving the row-order ordering of the paired
// vector. Both subarrays carry the same `CoderPartitionD5Reduction`
// element type as Phase 2 step 61 — the split is a re-presentation
// of the same per-row data, not a transformation. The split also
// pins the spec's width-column invariant per side: every element of
// the lower subarray has `width_n = 0`; every element of the upper
// subarray has `width_n = 1`. The split is fully determined by the
// static width column and so the slice lengths (12 / 20) are
// constant across calls; only the `LTmin_n` column varies with the
// caller's `LTg(ω)` callback.
// =====================================================================

/// A width-gated view of the §D.1 Step 8 row-order paired vector,
/// split into the two contiguous halves the Layer I / Layer II bit-
/// allocation loop branches on per the `width_n` column of Table D.5.
///
/// Produced by [`coder_partition_d5_reduction_row_order_by_width`]
/// (Phase 2 step 62 / r261).
///
/// **Field semantics.** `narrow_band` carries the contiguous prefix
/// of rows with `width_n = 0` (partitions `n ∈ 1..=12`, the lower
/// FFT-line block). `wide_band` carries the contiguous suffix of
/// rows with `width_n = 1` (partitions `n ∈ 13..=32`, the upper FFT-
/// line block). The two subarrays preserve the row-order ordering of
/// the paired vector — element `i` of `narrow_band` holds the same
/// `(LTmin_{i + 1}, width_{i + 1} = 0)` pair Phase 2 step 61's row
/// `i` carries; element `j` of `wide_band` holds the same
/// `(LTmin_{j + 13}, width_{j + 13} = 1)` pair Phase 2 step 61's row
/// `j + 12` carries.
///
/// **Width invariant.** Every element of `narrow_band` has
/// `width_n = 0`; every element of `wide_band` has `width_n = 1`. The
/// invariant is structural — pinned at construction by the split
/// point (12) coming from Phase 2 step 60's row-order width vector.
#[derive(Clone, Copy, Debug)]
pub struct CoderPartitionD5ReductionByWidth {
    /// The contiguous prefix of rows with `width_n = 0` (partitions
    /// `n ∈ 1..=12`, the lower FFT-line block). Twelve elements in
    /// row order; element `i` holds the `(LTmin_{i + 1}, width = 0)`
    /// pair.
    pub narrow_band: [CoderPartitionD5Reduction; 12],
    /// The contiguous suffix of rows with `width_n = 1` (partitions
    /// `n ∈ 13..=32`, the upper FFT-line block). Twenty elements in
    /// row order; element `j` holds the `(LTmin_{j + 13}, width = 1)`
    /// pair.
    pub wide_band: [CoderPartitionD5Reduction; 20],
}

/// §D.1 Step 8 width-gated split of the row-order paired vector for
/// every Layer I / Layer II coder partition `n ∈ 1..=32`. Returns a
/// [`CoderPartitionD5ReductionByWidth`] holding:
///
/// ```text
/// narrow_band[i] = (LTmin_{i + 1},  width_{i + 1} = 0)   for i ∈ 0..=11
/// wide_band[j]   = (LTmin_{j + 13}, width_{j + 13} = 1)  for j ∈ 0..=19
/// ```
///
/// The split is the row-order partitioning the Layer I / Layer II
/// bit-allocation loop branches on per the `width_n` column — the
/// lower (narrow) block drives the single-line per-partition target,
/// the upper (wide) block drives the multi-line per-partition target.
///
/// **Index convention.** 0-based on each subarray independently.
/// `narrow_band[i]` holds partition `i + 1`; `wide_band[j]` holds
/// partition `j + 13`. The split point (12) is constant — it is the
/// single 0 → 1 transition in Table D.5's `width_n` column, pinned
/// by Phase 2 step 60's row-order width vector.
///
/// **Composition.** A pure split of Phase 2 step 61's row-order
/// paired vector [`coder_partition_d5_reduction_row_order`] at the
/// constant index 12 (the width column's single 0 → 1 transition,
/// pinned by Phase 2 step 60's row-order width vector). No spec
/// arithmetic is introduced — only the re-presentation of the 32-row
/// paired vector as two width-gated subarrays. The `LTg(ω)` callback
/// is invoked exactly as many times as Phase 2 step 61 invokes it
/// (one call per FFT line in `Σ_{n=1..=32} (ωhigh_n − ωlow_n + 1)`).
///
/// **Width invariant.** Every element of the returned
/// `narrow_band` has `width_n = 0`; every element of the returned
/// `wide_band` has `width_n = 1`. The invariant is structural — the
/// split point and the static `width_n` column together pin it.
///
/// **Boundary semantics.** Inherits Phase 2 step 61's inclusive-on-
/// both-ends reduction semantics unchanged for the `LTmin_n` column.
/// The split itself has no boundary semantics — the lower-block /
/// upper-block boundary in Table D.5 is at partition 12 / 13 (a
/// row-level partition-index split, not an FFT-line split), distinct
/// from the per-partition `ωhigh_n = ωlow_{n+1}` boundary that step
/// 58's per-partition reducer reads.
///
/// **Implementation.** Calls
/// [`coder_partition_d5_reduction_row_order`] once (folding the
/// caller's callback over every recoverable partition's FFT-line
/// range) and copies the first 12 elements into `narrow_band` and
/// the last 20 elements into `wide_band`. The split is structurally
/// pinned at the same index 12 the static width-column transition
/// lives at — Phase 2 step 60's row-order width vector matches the
/// pattern `[0; 12]` followed by `[1; 20]` exactly.
///
/// Provenance: only the Phase 2 step 61 row-order paired-vector
/// accessor [`coder_partition_d5_reduction_row_order`] (and through
/// it the Phase 2 step 59 / 60 row-order LTmin and width vectors,
/// the Phase 2 step 58 per-partition reducer, the Phase 2 step 52
/// per-partition width accessor, and the underlying Table D.5
/// transcription in
/// `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
/// §"Table D.5 - Layer I and Layer II coder partition table") is
/// consulted. The width-gated split reading is the spec's per the
/// Table D.5 `width_n` column's role as a per-row Layer I / Layer II
/// bit-allocation branch flag (Annex D informative Model 1
/// reduction); no external implementation was read.
#[must_use]
pub fn coder_partition_d5_reduction_row_order_by_width<F>(
    ltg_per_line: F,
) -> CoderPartitionD5ReductionByWidth
where
    F: Fn(u16) -> f64,
{
    let paired = coder_partition_d5_reduction_row_order(ltg_per_line);
    let mut narrow = [CoderPartitionD5Reduction {
        ltmin_db: f64::INFINITY,
        width_n: 0,
    }; 12];
    let mut wide = [CoderPartitionD5Reduction {
        ltmin_db: f64::INFINITY,
        width_n: 0,
    }; 20];
    narrow.copy_from_slice(&paired[..12]);
    wide.copy_from_slice(&paired[12..]);
    CoderPartitionD5ReductionByWidth {
        narrow_band: narrow,
        wide_band: wide,
    }
}

// =====================================================================
// Annex D Model 1 — §D.1 Step 8 width-gated `LTmin_n` column
// projection over Table D.5 (Phase 2 step 63 / r262).
//
// Spec context (clause D.1, ISO/IEC 11172-3:1993, informative annex):
//
//   Step 8 produces, per coder partition n ∈ 1..=32, a per-partition
//   minimum global masking threshold `LTmin_n` (dB). Phase 2 step 62
//   (r261) exposed the width-gated split of the row-order paired
//   `(LTmin_n, width_n)` vector as the two contiguous subarrays the
//   Layer I / Layer II bit-allocation loop branches on per the
//   `width_n` column of Table D.5. Some downstream consumers (e.g. a
//   subsequent dB→linear conversion that operates per width band, or
//   a width-block bit-target-budget summation that consumes only the
//   threshold column) do not need the `width_n` field at the call
//   site once the call site has already chosen which subarray to
//   walk — `width_n` is implicit in the choice of `narrow_band`
//   versus `wide_band`.
//
// Composition rather than introduction: this step is a strict
// projection of Phase 2 step 62's `CoderPartitionD5ReductionByWidth`
// onto the `ltmin_db` field of each subarray. It introduces no spec
// arithmetic — every output cell is a copy of a cell in the step 62
// struct at the same array index. The `LTg(ω)` callback is invoked
// exactly as many times as Phase 2 step 62 invokes it (one call per
// FFT line in `Σ_{n=1..=32} (ωhigh_n − ωlow_n + 1)`); the projection
// itself adds no callback evaluations.
//
// The output is a `CoderPartitionD5LtminDbByWidth` carrying:
//
//   narrow_band[i] = LTmin_{i + 1}   for i ∈ 0..=11  (width_n = 0)
//   wide_band[j]   = LTmin_{j + 13}  for j ∈ 0..=19  (width_n = 1)
//
// The 0-based index convention on each subarray matches Phase 2
// step 62's exactly. Partition 0 is excluded for the same reason
// step 62 (and through it steps 58/59/61) excludes it — the
// reduction range is undefined without an `ωlow_n` boundary in
// Table D.5.
//
// Boundary semantics inherit from Phase 2 step 62 unchanged: a sharp
// dip on a shared boundary `ω = ωhigh_n = ωlow_{n+1}` enters both
// adjacent partitions' `LTmin` and therefore both columns. The
// projection has no boundary semantics of its own — it is a pure
// field-selection operation that reads exactly one field per row.
// =====================================================================

/// A width-gated view of the §D.1 Step 8 row-order `LTmin_n` column,
/// split into the two contiguous halves the Layer I / Layer II bit-
/// allocation loop branches on per the `width_n` column of Table D.5
/// with the (implicit-in-subarray-choice) `width_n` field elided.
///
/// Produced by [`coder_partition_d5_ltmin_db_row_order_by_width`]
/// (Phase 2 step 63 / r262).
///
/// **Field semantics.** `narrow_band` carries the contiguous prefix
/// of `LTmin_n` values (dB) for the rows with `width_n = 0`
/// (partitions `n ∈ 1..=12`, the lower FFT-line block). `wide_band`
/// carries the contiguous suffix of `LTmin_n` values (dB) for the
/// rows with `width_n = 1` (partitions `n ∈ 13..=32`, the upper FFT-
/// line block). The two subarrays preserve the row-order ordering of
/// the paired vector — element `i` of `narrow_band` carries
/// `LTmin_{i + 1}`; element `j` of `wide_band` carries
/// `LTmin_{j + 13}`.
///
/// **Width invariant.** The width column is implicit in the choice
/// of subarray — every cell of `narrow_band` corresponds to a
/// partition with `width_n = 0`; every cell of `wide_band`
/// corresponds to a partition with `width_n = 1`. The invariant is
/// structural — pinned at construction by the split point (12)
/// coming from Phase 2 step 60's row-order width vector and
/// preserved verbatim from Phase 2 step 62's `narrow_band` /
/// `wide_band` lengths.
#[derive(Clone, Copy, Debug)]
pub struct CoderPartitionD5LtminDbByWidth {
    /// Per-partition minimum global masking threshold `LTmin_n` (dB)
    /// for the contiguous prefix of rows with `width_n = 0`
    /// (partitions `n ∈ 1..=12`, the lower FFT-line block). Twelve
    /// elements in row order; element `i` carries `LTmin_{i + 1}`.
    pub narrow_band: [f64; 12],
    /// Per-partition minimum global masking threshold `LTmin_n` (dB)
    /// for the contiguous suffix of rows with `width_n = 1`
    /// (partitions `n ∈ 13..=32`, the upper FFT-line block). Twenty
    /// elements in row order; element `j` carries `LTmin_{j + 13}`.
    pub wide_band: [f64; 20],
}

/// §D.1 Step 8 width-gated `LTmin_n` column projection over Table
/// D.5. Returns a [`CoderPartitionD5LtminDbByWidth`] holding:
///
/// ```text
/// narrow_band[i] = LTmin_{i + 1}   for i ∈ 0..=11  (width_n = 0)
/// wide_band[j]   = LTmin_{j + 13}  for j ∈ 0..=19  (width_n = 1)
/// ```
///
/// This is the width-gated split of the per-frame `LTmin_n` column
/// the Layer I / Layer II bit-allocation loop reads when the call
/// site has already chosen which width subarray to walk — the
/// `width_n` value is implicit in the choice of `narrow_band` versus
/// `wide_band` and is therefore elided.
///
/// **Index convention.** 0-based on each subarray independently.
/// `narrow_band[i]` carries `LTmin_{i + 1}`; `wide_band[j]` carries
/// `LTmin_{j + 13}`. The split point (12) is constant — it is the
/// single 0 → 1 transition in Table D.5's `width_n` column, pinned
/// by Phase 2 step 60's row-order width vector and preserved
/// verbatim from Phase 2 step 62's split.
///
/// **Composition.** A pure projection of Phase 2 step 62's
/// [`coder_partition_d5_reduction_row_order_by_width`] onto the
/// `ltmin_db` field of each subarray. No spec arithmetic is
/// introduced — only the field-selection re-presentation of the
/// width-gated paired vector as two width-gated column vectors. The
/// `LTg(ω)` callback is invoked exactly as many times as Phase 2
/// step 62 invokes it (one call per FFT line in
/// `Σ_{n=1..=32} (ωhigh_n − ωlow_n + 1)`).
///
/// **Width invariant.** The width column is implicit in the choice
/// of subarray — every cell of the returned `narrow_band`
/// corresponds to a partition with `width_n = 0`; every cell of the
/// returned `wide_band` corresponds to a partition with
/// `width_n = 1`. The invariant is structural — pinned at the same
/// split point (12) Phase 2 step 62 pins.
///
/// **Boundary semantics.** Inherits Phase 2 step 62's (and through
/// it step 61 / 58 / 59's) inclusive-on-both-ends `LTmin_n`
/// reduction semantics unchanged. The projection has no boundary
/// semantics of its own — it is a pure field-selection operation
/// reading exactly one field per row.
///
/// **Implementation.** Calls
/// [`coder_partition_d5_reduction_row_order_by_width`] once (folding
/// the caller's callback over every recoverable partition's FFT-line
/// range) and reads the `ltmin_db` field of each cell into the
/// matching output subarray. The projection is structurally pinned
/// at the same index 12 the static width-column transition lives at
/// — Phase 2 step 60's row-order width vector matches the pattern
/// `[0; 12]` followed by `[1; 20]` exactly and Phase 2 step 62
/// inherits the split point verbatim.
///
/// Provenance: only the Phase 2 step 62 width-gated paired-vector
/// accessor [`coder_partition_d5_reduction_row_order_by_width`] (and
/// through it the Phase 2 step 61 row-order paired vector, the Phase
/// 2 step 59 / 60 row-order LTmin and width vectors, the Phase 2
/// step 58 per-partition reducer, the Phase 2 step 52 per-partition
/// width accessor, and the underlying Table D.5 transcription in
/// `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
/// §"Table D.5 - Layer I and Layer II coder partition table") is
/// consulted. The width-gated column-projection reading is the
/// spec's per the Table D.5 `width_n` column's role as a per-row
/// Layer I / Layer II bit-allocation branch flag (Annex D
/// informative Model 1 reduction); no external implementation was
/// read.
#[must_use]
pub fn coder_partition_d5_ltmin_db_row_order_by_width<F>(
    ltg_per_line: F,
) -> CoderPartitionD5LtminDbByWidth
where
    F: Fn(u16) -> f64,
{
    let split = coder_partition_d5_reduction_row_order_by_width(ltg_per_line);
    let mut narrow = [f64::INFINITY; 12];
    let mut wide = [f64::INFINITY; 20];
    for (i, cell) in split.narrow_band.iter().enumerate() {
        narrow[i] = cell.ltmin_db;
    }
    for (j, cell) in split.wide_band.iter().enumerate() {
        wide[j] = cell.ltmin_db;
    }
    CoderPartitionD5LtminDbByWidth {
        narrow_band: narrow,
        wide_band: wide,
    }
}

// ---------------------------------------------------------------------------
// Annex D Model 1 — §D.1 Step 8 width-gated `LTmin_n` column projection
// converted to linear energy `10^(LTmin_n / 10)` over Table D.5
// (Phase 2 step 64 / r263).
//
// Step 8 (Phase 2 steps 58 / 59 / 60 / 61 / 62 / 63) produces, per
// coder partition `n ∈ 1..=32`, the per-partition minimum global
// masking threshold `LTmin_n` (dB) and width column flag
// `width_n ∈ {0, 1}`. Phase 2 step 63 (r262) projected the row-order
// paired `(LTmin_n, width_n)` vector — already pre-split at the
// single width-column 0 → 1 transition (between rows 12 and 13) —
// onto the `ltmin_db` field of each subarray:
//
//   narrow_band[i] = LTmin_{i + 1}   (dB)  for i ∈ 0..=11  (width_n = 0)
//   wide_band[j]   = LTmin_{j + 13}  (dB)  for j ∈ 0..=19  (width_n = 1)
//
// Several Step 9 / Step 10 / outer-loop consumers do not read the
// per-band masking threshold in dB — they read it in the linear
// energy domain `10^(LTmin_n / 10)`. The dB → linear conversion is a
// strict mathematical primitive: it is the same `10^(·/10)`
// transformation `db_to_xfsf_energy` already uses (line 411), the
// same Step 7 `Σ 10^(LTtm/10)` global-threshold summer uses (lines
// 702 / 705), and the same Model 2 Layer III spread linearisation
// uses (line 1492). It introduces no new spec arithmetic.
//
// Step 64 exposes that conversion as a free function returning a new
// `CoderPartitionD5LtminLinearByWidth` carrying the same row-order
// subarray split (12 narrow + 20 wide) but with each cell holding a
// non-negative linear energy value `10^(dB/10)` instead of a dB
// value. Like step 63, the function is a pure projection of step
// 62's struct — it invokes the caller's `LTg(ω)` callback exactly
// the same number of times as step 62 (and through it steps 61 / 60
// / 59 / 58) invokes it (one call per FFT line in
// `Σ_{n=1..=32} (ωhigh_n − ωlow_n + 1)`), and applies one
// `(10.0_f64).powf(db / 10.0)` per output cell.
// ---------------------------------------------------------------------------

/// Per-partition minimum global masking threshold `LTmin_n` over
/// Table D.5 in **linear energy** (`10^(LTmin_n / 10)`), split by
/// the `width_n` column.
///
/// Produced by [`coder_partition_d5_ltmin_linear_row_order_by_width`]
/// (the linear-energy projection of Phase 2 step 63's
/// [`coder_partition_d5_ltmin_db_row_order_by_width`]).
///
/// The two subarrays carry the partition's minimum global masking
/// threshold per row of Table D.5, but in the linear energy domain
/// rather than the dB domain. They preserve step 63's row-order
/// indexing and the same split point (12) Table D.5's `width_n`
/// column pins.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoderPartitionD5LtminLinearByWidth {
    /// Per-partition minimum global masking threshold `LTmin_n`
    /// converted to linear energy `10^(LTmin_n / 10)` for the
    /// contiguous prefix of rows with `width_n = 0` (partitions
    /// `n ∈ 1..=12`, the lower FFT-line block). Twelve elements in
    /// row order; element `i` carries `10^(LTmin_{i + 1} / 10)`.
    /// Every cell is strictly positive (the conversion preserves the
    /// non-negativity of `10^x`); `INFINITY` if step 62's reduction
    /// over that partition's FFT-line range yielded `INFINITY` (an
    /// empty partition or an all-`INFINITY` callback).
    pub narrow_band: [f64; 12],
    /// Per-partition minimum global masking threshold `LTmin_n`
    /// converted to linear energy `10^(LTmin_n / 10)` for the
    /// contiguous suffix of rows with `width_n = 1` (partitions
    /// `n ∈ 13..=32`, the upper FFT-line block). Twenty elements in
    /// row order; element `j` carries `10^(LTmin_{j + 13} / 10)`.
    /// Every cell is strictly positive; `INFINITY` under the same
    /// degenerate condition described for `narrow_band`.
    pub wide_band: [f64; 20],
}

/// §D.1 Step 8 width-gated `LTmin_n` column projection over Table
/// D.5 converted to **linear energy** (`10^(LTmin_n / 10)`). Returns
/// a [`CoderPartitionD5LtminLinearByWidth`] holding:
///
/// ```text
/// narrow_band[i] = 10^(LTmin_{i + 1}  / 10)   for i ∈ 0..=11  (width_n = 0)
/// wide_band[j]   = 10^(LTmin_{j + 13} / 10)   for j ∈ 0..=19  (width_n = 1)
/// ```
///
/// This is the linear-energy presentation of Phase 2 step 63's
/// width-gated per-band `LTmin_n` (dB) column. The dB → linear
/// conversion is the same monotone transformation
/// `db_to_xfsf_energy` already applies to the threshold-in-quiet
/// curve, the same `Σ 10^(LTtm/10)` Step 7 global-threshold summer
/// applies to per-line masker dB contributions, and the same Model 2
/// Layer III spread linearisation applies — it introduces no new
/// spec arithmetic.
///
/// **Index convention.** 0-based on each subarray independently.
/// `narrow_band[i]` carries `10^(LTmin_{i + 1} / 10)`;
/// `wide_band[j]` carries `10^(LTmin_{j + 13} / 10)`. The split
/// point (12) is constant — preserved verbatim from Phase 2 step 63
/// (and through it step 62 / 60).
///
/// **Composition.** A pure linearisation of Phase 2 step 63's
/// [`coder_partition_d5_ltmin_db_row_order_by_width`]. The `LTg(ω)`
/// callback is invoked exactly as many times as step 63 invokes it
/// (one call per FFT line in
/// `Σ_{n=1..=32} (ωhigh_n − ωlow_n + 1)`); each output cell is
/// `(10.0_f64).powf(input_cell_db / 10.0)`.
///
/// **Monotonicity.** The conversion is strictly monotone in dB —
/// `a_db < b_db ⇔ 10^(a_db / 10) < 10^(b_db / 10)` — so ordering is
/// preserved cell-wise. A partition whose `LTmin_n` in dB is strictly
/// less than another partition's `LTmin_n` in dB will have a strictly
/// smaller linear-energy cell at the corresponding row index, and
/// vice versa.
///
/// **Non-negativity.** Every output cell is strictly positive (the
/// conversion `10^x` is strictly positive for every finite real
/// `x`). The only way an output cell can be `INFINITY` is if the
/// corresponding step 63 cell's dB value is `INFINITY` (which
/// happens only under a degenerate callback that returns `INFINITY`
/// for every FFT line in some partition's range).
///
/// **Width invariant.** The width column is implicit in the choice
/// of subarray — every cell of the returned `narrow_band`
/// corresponds to a partition with `width_n = 0`; every cell of the
/// returned `wide_band` corresponds to a partition with
/// `width_n = 1`. The invariant is structural — pinned at the same
/// split point (12) Phase 2 step 63 pins.
///
/// **Boundary semantics.** Inherits Phase 2 step 63's (and through
/// it step 62 / 61 / 58 / 59's) inclusive-on-both-ends `LTmin_n`
/// reduction semantics unchanged. The linearisation has no boundary
/// semantics of its own — it is a pure cell-wise transformation
/// reading exactly one input cell per output cell.
///
/// **Implementation.** Calls
/// [`coder_partition_d5_ltmin_db_row_order_by_width`] once (folding
/// the caller's callback over every recoverable partition's FFT-line
/// range) and applies `(10.0_f64).powf(db / 10.0)` to each cell of
/// each subarray into the matching output subarray. The conversion
/// is structurally pinned at the same index 12 the static
/// width-column transition lives at — Phase 2 step 60's row-order
/// width vector matches the pattern `[0; 12]` followed by `[1; 20]`
/// exactly and Phase 2 step 63 inherits the split point verbatim.
///
/// Provenance: only the Phase 2 step 63 width-gated `LTmin_n` (dB)
/// column accessor [`coder_partition_d5_ltmin_db_row_order_by_width`]
/// (and through it the Phase 2 step 62 width-gated paired-vector
/// accessor, the Phase 2 step 61 row-order paired vector, the Phase
/// 2 step 59 / 60 row-order LTmin and width vectors, the Phase 2
/// step 58 per-partition reducer, the Phase 2 step 52 per-partition
/// width accessor, and the underlying Table D.5 transcription in
/// `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
/// §"Table D.5 - Layer I and Layer II coder partition table") is
/// consulted. The `10^(x / 10)` dB → linear conversion is the
/// in-tree convention used by `db_to_xfsf_energy` (line 411), the
/// Step 7 `global_masking_threshold_db` summer (lines 702 / 705),
/// and the Model 2 Layer III spread linearisation (line 1492); no
/// external implementation was read.
#[must_use]
pub fn coder_partition_d5_ltmin_linear_row_order_by_width<F>(
    ltg_per_line: F,
) -> CoderPartitionD5LtminLinearByWidth
where
    F: Fn(u16) -> f64,
{
    let db = coder_partition_d5_ltmin_db_row_order_by_width(ltg_per_line);
    let mut narrow = [f64::INFINITY; 12];
    let mut wide = [f64::INFINITY; 20];
    for (i, &cell_db) in db.narrow_band.iter().enumerate() {
        narrow[i] = (10.0_f64).powf(cell_db / 10.0);
    }
    for (j, &cell_db) in db.wide_band.iter().enumerate() {
        wide[j] = (10.0_f64).powf(cell_db / 10.0);
    }
    CoderPartitionD5LtminLinearByWidth {
        narrow_band: narrow,
        wide_band: wide,
    }
}

// ---------------------------------------------------------------------------
// Annex D Model 1 — §D.1 Step 8 width-gated `log2(LTmin_lin_n)` column
// projection over Table D.5 (Phase 2 step 65 / r264).
//
// Step 8 (Phase 2 steps 58 / 59 / 60 / 61 / 62 / 63 / 64) produces, per
// coder partition `n ∈ 1..=32`, the per-partition minimum global
// masking threshold `LTmin_n` in two presentations — dB (step 63) and
// linear energy `10^(LTmin_n / 10)` (step 64) — already split at the
// single width-column 0 → 1 transition (between rows 12 and 13):
//
//   narrow_band[i] = 10^(LTmin_{i + 1}  / 10)   for i ∈ 0..=11  (width_n = 0)
//   wide_band[j]   = 10^(LTmin_{j + 13} / 10)   for j ∈ 0..=19  (width_n = 1)
//
// Several Step 9 / Step 10 / outer-loop consumers do not read the
// per-band linear-energy threshold directly — they read its base-2
// logarithm `log2(10^(LTmin_n / 10))`. The base-2 log of a linear-
// energy threshold is the natural per-band bit-budget proxy in the
// Layer I/II bit-allocation loop: a linear-energy ratio expressed in
// "bits" maps one factor of two in masking power to a single bit of
// allocated dynamic range. The `log2` transformation is a strict
// mathematical primitive — `f64::log2` — and introduces no new spec
// arithmetic.
//
// Step 65 exposes that conversion as a free function returning a new
// `CoderPartitionD5LtminLog2ByWidth` carrying the same row-order
// subarray split (12 narrow + 20 wide) but with each cell holding
// `log2(linear_n) = log2(10^(LTmin_n / 10)) = LTmin_n · log2(10) / 10`
// instead of a linear-energy value. Like step 64, the function is a
// pure projection of step 62's struct — it invokes the caller's
// `LTg(ω)` callback exactly the same number of times as step 64 (and
// through it steps 63 / 62 / 61 / 60 / 59 / 58) invokes it (one call
// per FFT line in `Σ_{n=1..=32} (ωhigh_n − ωlow_n + 1)`), and applies
// one `f64::log2` per output cell of step 64's linear projection.
// ---------------------------------------------------------------------------

/// Per-partition minimum global masking threshold `LTmin_n` over
/// Table D.5 in **`log2` of linear energy** (`log2(10^(LTmin_n / 10))`),
/// split by the `width_n` column.
///
/// Produced by [`coder_partition_d5_ltmin_log2_row_order_by_width`]
/// (the `log2` projection of Phase 2 step 64's
/// [`coder_partition_d5_ltmin_linear_row_order_by_width`]).
///
/// The two subarrays carry the partition's minimum global masking
/// threshold per row of Table D.5, expressed as the base-2 logarithm
/// of the linear-energy presentation. They preserve step 64's
/// row-order indexing and the same split point (12) Table D.5's
/// `width_n` column pins.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoderPartitionD5LtminLog2ByWidth {
    /// `log2(linear_n)` for the contiguous prefix of rows with
    /// `width_n = 0` (partitions `n ∈ 1..=12`, the lower FFT-line
    /// block). Twelve elements in row order; element `i` carries
    /// `log2(10^(LTmin_{i + 1} / 10))`. Cells are finite when step 64's
    /// matching cell is finite and strictly positive (always the case
    /// for any callback returning a finite dB value at every FFT line
    /// in the partition's range); `+INFINITY` only when step 64's
    /// matching cell is `+INFINITY` (the degenerate all-`INFINITY`
    /// callback).
    pub narrow_band: [f64; 12],
    /// `log2(linear_n)` for the contiguous suffix of rows with
    /// `width_n = 1` (partitions `n ∈ 13..=32`, the upper FFT-line
    /// block). Twenty elements in row order; element `j` carries
    /// `log2(10^(LTmin_{j + 13} / 10))`. Same finiteness convention as
    /// `narrow_band`.
    pub wide_band: [f64; 20],
}

/// §D.1 Step 8 width-gated `LTmin_n` column projection over Table D.5
/// converted to **`log2` of linear energy** (`log2(10^(LTmin_n / 10))`).
/// Returns a [`CoderPartitionD5LtminLog2ByWidth`] holding:
///
/// ```text
/// narrow_band[i] = log2(10^(LTmin_{i + 1}  / 10))   for i ∈ 0..=11  (width_n = 0)
/// wide_band[j]   = log2(10^(LTmin_{j + 13} / 10))   for j ∈ 0..=19  (width_n = 1)
/// ```
///
/// This is the base-2 logarithm presentation of Phase 2 step 64's
/// width-gated per-band linear-energy column. The `log2` conversion is
/// the natural per-band bit-budget proxy in the Layer I/II
/// bit-allocation loop: every factor-of-two change in linear masking
/// energy corresponds to exactly one unit on the `log2` axis, which is
/// the unit Step 9's signal-to-mask ratio and the outer loop's per-band
/// bit-demand summation both work in. It introduces no new spec
/// arithmetic — `log2` is a pure `f64::log2` primitive.
///
/// **Index convention.** 0-based on each subarray independently.
/// `narrow_band[i]` carries `log2(10^(LTmin_{i + 1}  / 10))`;
/// `wide_band[j]`   carries `log2(10^(LTmin_{j + 13} / 10))`. The split
/// point (12) is constant — preserved verbatim from Phase 2 step 64
/// (and through it step 63 / 62 / 60).
///
/// **Composition.** A pure logarithmisation of Phase 2 step 64's
/// [`coder_partition_d5_ltmin_linear_row_order_by_width`]. The `LTg(ω)`
/// callback is invoked exactly as many times as step 64 invokes it
/// (one call per FFT line in
/// `Σ_{n=1..=32} (ωhigh_n − ωlow_n + 1)`); each output cell is
/// `input_cell_linear.log2()`.
///
/// **Identity with the dB column.** `log2` of step 64's linear cell
/// equals `LTmin_n · log2(10) / 10` — a strictly proportional rescaling
/// of step 63's dB column by the constant `log2(10) / 10 ≈ 0.33219`.
/// This makes the `log2` view equivalent in information content to
/// the dB view (both preserve the same ordering and same per-band
/// magnitudes up to a constant); the `log2` view is what consumers
/// need when their downstream summation is in linear-energy bit units
/// (a power-of-two quantisation step), and the dB view is what
/// consumers need when their downstream summation is in dB
/// (a power-of-ten quantisation step). Both are pure projections of
/// step 62.
///
/// **Monotonicity.** `log2` is strictly monotone on the positive
/// reals — `a < b ⇔ log2(a) < log2(b)` for `0 < a, b < ∞` — so the
/// cell-wise ordering is preserved. A partition whose `LTmin_n` is
/// strictly less than another partition's `LTmin_n` (in any of the
/// dB, linear, or log2 presentations) will have a strictly smaller
/// `log2` cell at the corresponding row index, and vice versa.
///
/// **Sign convention.** Unlike the linear-energy presentation
/// (`> 0` always) and the dB presentation (sign unconstrained), the
/// `log2` cells can be of either sign: `log2(x) < 0 ⇔ x < 1`, i.e.
/// every cell whose linear-energy threshold is strictly less than
/// unit energy (one in the integer pulse-code modulation grid) sits
/// below zero on the `log2` axis. A `0` cell corresponds to unit
/// linear energy (`10^0 = 1`), which corresponds to a `0 dB` LTmin.
///
/// **Width invariant.** The width column is implicit in the choice
/// of subarray — every cell of the returned `narrow_band` corresponds
/// to a partition with `width_n = 0`; every cell of the returned
/// `wide_band` corresponds to a partition with `width_n = 1`. The
/// invariant is structural — pinned at the same split point (12)
/// Phase 2 step 64 pins.
///
/// **Boundary semantics.** Inherits Phase 2 step 64's (and through
/// it step 63 / 62 / 61 / 58 / 59's) inclusive-on-both-ends `LTmin_n`
/// reduction semantics unchanged. The logarithmisation has no
/// boundary semantics of its own — it is a pure cell-wise
/// transformation reading exactly one input cell per output cell.
///
/// **Implementation.** Calls
/// [`coder_partition_d5_ltmin_linear_row_order_by_width`] once (folding
/// the caller's callback over every recoverable partition's FFT-line
/// range) and applies `f64::log2` to each cell of each subarray into
/// the matching output subarray. The conversion is structurally
/// pinned at the same index 12 the static width-column transition
/// lives at — Phase 2 step 60's row-order width vector matches the
/// pattern `[0; 12]` followed by `[1; 20]` exactly and Phase 2 step 64
/// inherits the split point verbatim.
///
/// Provenance: only the Phase 2 step 64 width-gated linear-energy
/// `LTmin_n` accessor [`coder_partition_d5_ltmin_linear_row_order_by_width`]
/// (and through it the Phase 2 step 63 width-gated dB column accessor,
/// the Phase 2 step 62 width-gated paired-vector accessor, the Phase
/// 2 step 61 row-order paired vector, the Phase 2 step 59 / 60
/// row-order LTmin and width vectors, the Phase 2 step 58
/// per-partition reducer, the Phase 2 step 52 per-partition width
/// accessor, and the underlying Table D.5 transcription in
/// `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
/// §"Table D.5 - Layer I and Layer II coder partition table") is
/// consulted. The `f64::log2` primitive is the in-tree standard
/// library call; no external implementation was read.
#[must_use]
pub fn coder_partition_d5_ltmin_log2_row_order_by_width<F>(
    ltg_per_line: F,
) -> CoderPartitionD5LtminLog2ByWidth
where
    F: Fn(u16) -> f64,
{
    let lin = coder_partition_d5_ltmin_linear_row_order_by_width(ltg_per_line);
    let mut narrow = [f64::INFINITY; 12];
    let mut wide = [f64::INFINITY; 20];
    for (i, &cell_lin) in lin.narrow_band.iter().enumerate() {
        narrow[i] = cell_lin.log2();
    }
    for (j, &cell_lin) in lin.wide_band.iter().enumerate() {
        wide[j] = cell_lin.log2();
    }
    CoderPartitionD5LtminLog2ByWidth {
        narrow_band: narrow,
        wide_band: wide,
    }
}

// ---------------------------------------------------------------------------
// Annex D Model 1 — §D.1 Step 8 width-gated wide-band signed bit-budget
// reduction `Σ_{n=1..=32} width_n · log2(LTmin_lin_n)` over Table D.5
// (Phase 2 step 66 / r265).
//
// Step 8 (Phase 2 steps 58 / 59 / 60 / 61 / 62 / 63 / 64 / 65)
// produces, per coder partition `n ∈ 1..=32`, the per-partition
// minimum global masking threshold `LTmin_n` already projected onto
// the `log2`-of-linear-energy presentation `log2(10^(LTmin_n / 10))`
// and split at the single width-column 0 → 1 transition (between
// rows 12 and 13):
//
//   narrow_band[i] = log2(10^(LTmin_{i + 1}  / 10))   for i ∈ 0..=11  (width_n = 0)
//   wide_band[j]   = log2(10^(LTmin_{j + 13} / 10))   for j ∈ 0..=19  (width_n = 1)
//
// Several Step 9 / Step 10 consumers do not read the per-band
// `log2(LTmin_lin_n)` column cell-by-cell — they read its weighted
// total `Σ_{n=1..=32} width_n · log2_n` where `width_n` is the row's
// width column flag. Because `width_n = 0` for every narrow row and
// `width_n = 1` for every wide row (a structural invariant of
// Table D.5 verified by Phase 2 step 60 and inherited by step 65),
// the weighted total collapses to the unweighted sum of step 65's
// `wide_band` subarray — a 20-element strict reduction that
// introduces no new spec arithmetic beyond `+`.
//
// Step 66 exposes that reduction as a free function returning a
// single `f64`. Like step 65, the function is a pure projection of
// step 62's struct — it invokes the caller's `LTg(ω)` callback
// exactly the same number of times as step 65 (and through it steps
// 64 / 63 / 62 / 61 / 60 / 59 / 58) invokes it (one call per FFT
// line in `Σ_{n=1..=32} (ωhigh_n − ωlow_n + 1)`), and applies one
// addition per wide-band output cell of step 65's `log2` projection.
// ---------------------------------------------------------------------------

/// §D.1 Step 8 width-gated wide-band signed bit-budget reduction
/// `Σ_{n=1..=32} width_n · log2(LTmin_lin_n)` over Table D.5. Returns
/// the scalar `f64` total:
///
/// ```text
/// total = Σ_{n=1..=32} width_n · log2(10^(LTmin_n / 10))
///       = Σ_{j=0..=19} wide_band[j]                  (since width_n = 0 for n ∈ 1..=12)
/// ```
///
/// This is the wide-block signed bit-budget total of Phase 2 step
/// 65's width-gated per-band `log2(LTmin_lin_n)` column. Because
/// Table D.5's `width_n` column is `0` for every row in `1..=12`
/// and `1` for every row in `13..=32` (Phase 2 step 60 verifies the
/// shape and step 65 inherits the split point), the weighted total
/// reduces algebraically to the unweighted sum of step 65's
/// `wide_band` subarray. The reduction introduces no new spec
/// arithmetic — it is pure addition over 20 cells.
///
/// **Composition.** A pure reduction of Phase 2 step 65's
/// [`coder_partition_d5_ltmin_log2_row_order_by_width`]. The `LTg(ω)`
/// callback is invoked exactly as many times as step 65 invokes it
/// (one call per FFT line in
/// `Σ_{n=1..=32} (ωhigh_n − ωlow_n + 1)`); the output total is
/// `wide_band.iter().sum::<f64>()`.
///
/// **Sign semantics.** The "signed" qualifier reflects that
/// `log2(linear)` is unbounded below as `linear → 0`. A partition
/// whose `LTmin_n` is well below 0 dB linearises to a value strictly
/// less than `1.0`, whose `log2` is strictly negative; conversely a
/// partition whose `LTmin_n` exceeds 0 dB contributes a strictly
/// positive cell. The total accumulates both signs without clipping.
///
/// **Width invariant.** The reduction reads only step 65's
/// `wide_band` subarray (20 cells). Narrow-band cells contribute
/// `0 · log2_n = 0` per the width column and are deliberately
/// skipped — the reduction is structurally identical to
/// `Σ_{n=1..=32} width_n · log2_n` but avoids 12 redundant
/// multiplications by zero. The optimisation is pinned at the same
/// split index (12) Phase 2 step 60's row-order width vector
/// transitions at and Phase 2 step 65 inherits.
///
/// **Boundary semantics.** Inherits Phase 2 step 65's (and through
/// it step 64 / 63 / 62 / 61 / 58 / 59's) inclusive-on-both-ends
/// `LTmin_n` reduction semantics unchanged. The wide-band sum has no
/// boundary semantics of its own — it is a pure cell-wise addition
/// reading every cell of `wide_band` exactly once.
///
/// **Finiteness.** Finite when every wide cell of step 65 is finite
/// (which holds whenever the callback returns a finite real for
/// every FFT line in partitions `13..=32`). If any wide cell is
/// `INFINITY` (a degenerate callback returning `INFINITY` for every
/// FFT line in some wide partition's range, lifted through step 64's
/// `10^x` linearisation and step 65's `log2`), the total is
/// `INFINITY`. If any wide cell is `-INFINITY` (a callback returning
/// `-INFINITY` for every FFT line in some wide partition's range,
/// driving step 64's `10^x` to `0.0` whose `log2` is `-INFINITY`),
/// the total is `-INFINITY`.
///
/// **Determinism.** A pure function of the callback: invoking the
/// reduction twice with the same callback returns the same `f64`.
///
/// **Implementation.** Calls
/// [`coder_partition_d5_ltmin_log2_row_order_by_width`] once
/// (folding the caller's callback over every recoverable partition's
/// FFT-line range) and sums the resulting `wide_band` subarray. The
/// reduction is structurally pinned at the same index 12 the static
/// width-column transition lives at — Phase 2 step 60's row-order
/// width vector matches the pattern `[0; 12]` followed by `[1; 20]`
/// exactly and Phase 2 step 65 inherits the split point verbatim.
///
/// Provenance: only the Phase 2 step 65 width-gated `log2(LTmin_lin_n)`
/// column accessor [`coder_partition_d5_ltmin_log2_row_order_by_width`]
/// (and through it the Phase 2 step 64 width-gated `LTmin_n` linear
/// accessor, the Phase 2 step 63 width-gated `LTmin_n` dB accessor,
/// the Phase 2 step 62 width-gated paired-vector accessor, the
/// Phase 2 step 61 row-order paired vector, the Phase 2 step 59 / 60
/// row-order LTmin and width vectors, the Phase 2 step 58
/// per-partition reducer, the Phase 2 step 52 per-partition width
/// accessor, and the underlying Table D.5 transcription in
/// `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
/// §"Table D.5 - Layer I and Layer II coder partition table") is
/// consulted. The reduction is plain `f64` addition; no external
/// implementation was read.
#[must_use]
pub fn coder_partition_d5_ltmin_log2_wide_band_bit_budget_total<F>(ltg_per_line: F) -> f64
where
    F: Fn(u16) -> f64,
{
    let split = coder_partition_d5_ltmin_log2_row_order_by_width(ltg_per_line);
    let mut total = 0.0_f64;
    for &cell in &split.wide_band {
        total += cell;
    }
    total
}

// ---------------------------------------------------------------------------
// Annex D Model 1 — §D.1 Step 8 width-gated narrow-band complementary
// `Σ_{n=1..=32} (1 − width_n) · log2(LTmin_lin_n)` reduction over
// Table D.5 (Phase 2 step 67 / r266).
//
// Step 8 (Phase 2 steps 58 / 59 / 60 / 61 / 62 / 63 / 64 / 65 / 66)
// produces, per coder partition `n ∈ 1..=32`, the per-partition
// minimum global masking threshold `LTmin_n` projected onto the
// `log2`-of-linear-energy presentation `log2(10^(LTmin_n / 10))` and
// split at the single width-column 0 → 1 transition (between rows
// 12 and 13):
//
//   narrow_band[i] = log2(10^(LTmin_{i + 1}  / 10))   for i ∈ 0..=11  (width_n = 0)
//   wide_band[j]   = log2(10^(LTmin_{j + 13} / 10))   for j ∈ 0..=19  (width_n = 1)
//
// Phase 2 step 66 (r265) exposed the wide-band weighted total
// `Σ_n width_n · log2_n`, which algebraically collapses onto the
// unweighted sum of `wide_band` because every wide row carries
// `width_n = 1` and every narrow row carries `width_n = 0`.
//
// Several Step 9 / Step 10 consumers also read the *complementary*
// total `Σ_n (1 − width_n) · log2_n`, the narrow-block companion of
// step 66. Under the same Table D.5 width-column invariant (Phase 2
// step 60), the complementary weighted total collapses
// algebraically onto the unweighted sum of step 65's `narrow_band`
// subarray — a 12-element strict reduction that introduces no new
// spec arithmetic beyond `+`.
//
// Step 67 exposes that complementary reduction as a free function
// returning a single `f64`. Like step 66, it is a pure projection of
// step 65's struct — it invokes the caller's `LTg(ω)` callback
// exactly the same number of times as step 65 (and through it steps
// 64 / 63 / 62 / 61 / 60 / 59 / 58) invokes it (one call per FFT
// line in `Σ_{n=1..=32} (ωhigh_n − ωlow_n + 1)`), and applies one
// addition per narrow-band output cell of step 65's `log2`
// projection.
//
// The pair (step 66, step 67) reconstructs the full row-order
// `Σ_n log2_n` exactly: the unweighted total equals
// `wide_total + narrow_total` by the partition `[narrow_band]`/
// `[wide_band]` of step 65's output without re-summation. The two
// reductions together carry the full Table D.5 row-order signed
// bit-budget budget without losing the width-column split.
// ---------------------------------------------------------------------------

/// §D.1 Step 8 width-gated narrow-band complementary signed bit-budget
/// reduction `Σ_{n=1..=32} (1 − width_n) · log2(LTmin_lin_n)` over
/// Table D.5. Returns the scalar `f64` total:
///
/// ```text
/// total = Σ_{n=1..=32} (1 − width_n) · log2(10^(LTmin_n / 10))
///       = Σ_{i=0..=11} narrow_band[i]                  (since width_n = 1 for n ∈ 13..=32)
/// ```
///
/// This is the narrow-block complementary signed bit-budget total of
/// Phase 2 step 65's width-gated per-band `log2(LTmin_lin_n)` column.
/// Because Table D.5's `width_n` column is `0` for every row in
/// `1..=12` and `1` for every row in `13..=32` (Phase 2 step 60
/// verifies the shape and step 65 inherits the split point), the
/// complementary weighted total reduces algebraically to the
/// unweighted sum of step 65's `narrow_band` subarray. The reduction
/// introduces no new spec arithmetic — it is pure addition over 12
/// cells.
///
/// **Composition.** A pure reduction of Phase 2 step 65's
/// [`coder_partition_d5_ltmin_log2_row_order_by_width`]. The `LTg(ω)`
/// callback is invoked exactly as many times as step 65 invokes it
/// (one call per FFT line in
/// `Σ_{n=1..=32} (ωhigh_n − ωlow_n + 1)`); the output total is
/// `narrow_band.iter().sum::<f64>()`.
///
/// **Pairing with step 66.** The pair `(narrow_total, wide_total)`
/// partitions the full row-order `Σ_n log2_n` exactly: by the
/// disjointness of step 65's `narrow_band` / `wide_band` fields, the
/// unweighted total of the 32-row vector equals
/// `narrow_total + wide_total` without re-summation. The two
/// reductions together carry the full Table D.5 row-order signed
/// bit-budget budget without losing the width-column split.
///
/// **Sign semantics.** The "signed" qualifier reflects that
/// `log2(linear)` is unbounded below as `linear → 0`. A partition
/// whose `LTmin_n` is well below 0 dB linearises to a value strictly
/// less than `1.0`, whose `log2` is strictly negative; conversely a
/// partition whose `LTmin_n` exceeds 0 dB contributes a strictly
/// positive cell. The total accumulates both signs without clipping.
///
/// **Width invariant.** The reduction reads only step 65's
/// `narrow_band` subarray (12 cells). Wide-band cells contribute
/// `(1 − 1) · log2_n = 0` per the width column and are deliberately
/// skipped — the reduction is structurally identical to
/// `Σ_{n=1..=32} (1 − width_n) · log2_n` but avoids 20 redundant
/// multiplications by zero. The optimisation is pinned at the same
/// split index (12) Phase 2 step 60's row-order width vector
/// transitions at and Phase 2 step 65 inherits.
///
/// **Boundary semantics.** Inherits Phase 2 step 65's (and through
/// it step 64 / 63 / 62 / 61 / 58 / 59's) inclusive-on-both-ends
/// `LTmin_n` reduction semantics unchanged. The narrow-band sum has
/// no boundary semantics of its own — it is a pure cell-wise
/// addition reading every cell of `narrow_band` exactly once.
///
/// **Finiteness.** Finite when every narrow cell of step 65 is
/// finite (which holds whenever the callback returns a finite real
/// for every FFT line in partitions `1..=12`). If any narrow cell is
/// `INFINITY` (a degenerate callback returning `INFINITY` for every
/// FFT line in some narrow partition's range, lifted through step
/// 64's `10^x` linearisation and step 65's `log2`), the total is
/// `INFINITY`. If any narrow cell is `-INFINITY` (a callback
/// returning `-INFINITY` for every FFT line in some narrow
/// partition's range, driving step 64's `10^x` to `0.0` whose `log2`
/// is `-INFINITY`), the total is `-INFINITY`.
///
/// **Determinism.** A pure function of the callback: invoking the
/// reduction twice with the same callback returns the same `f64`.
///
/// **Implementation.** Calls
/// [`coder_partition_d5_ltmin_log2_row_order_by_width`] once
/// (folding the caller's callback over every recoverable partition's
/// FFT-line range) and sums the resulting `narrow_band` subarray.
/// The reduction is structurally pinned at the same index 12 the
/// static width-column transition lives at — Phase 2 step 60's
/// row-order width vector matches the pattern `[0; 12]` followed by
/// `[1; 20]` exactly and Phase 2 step 65 inherits the split point
/// verbatim.
///
/// Provenance: only the Phase 2 step 65 width-gated `log2(LTmin_lin_n)`
/// column accessor [`coder_partition_d5_ltmin_log2_row_order_by_width`]
/// (and through it the Phase 2 step 64 width-gated `LTmin_n` linear
/// accessor, the Phase 2 step 63 width-gated `LTmin_n` dB accessor,
/// the Phase 2 step 62 width-gated paired-vector accessor, the
/// Phase 2 step 61 row-order paired vector, the Phase 2 step 59 / 60
/// row-order LTmin and width vectors, the Phase 2 step 58
/// per-partition reducer, the Phase 2 step 52 per-partition width
/// accessor, and the underlying Table D.5 transcription in
/// `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
/// §"Table D.5 - Layer I and Layer II coder partition table") is
/// consulted. The reduction is plain `f64` addition; no external
/// implementation was read.
#[must_use]
pub fn coder_partition_d5_ltmin_log2_narrow_band_bit_budget_total<F>(ltg_per_line: F) -> f64
where
    F: Fn(u16) -> f64,
{
    let split = coder_partition_d5_ltmin_log2_row_order_by_width(ltg_per_line);
    let mut total = 0.0_f64;
    for &cell in &split.narrow_band {
        total += cell;
    }
    total
}

// ---------------------------------------------------------------------------
// Annex D Model 1 — §D.1 Step 8 width-gated paired `(narrow_total,
// wide_total)` signed bit-budget reduction over Table D.5 with a single
// step-65 invocation (Phase 2 step 68 / r267).
//
// Phase 2 step 66 (r265) exposed the wide-band weighted total
// `Σ_n width_n · log2_n` (collapsing onto `Σ wide_band`) and Phase 2
// step 67 (r266) exposed the complementary narrow-band weighted total
// `Σ_n (1 − width_n) · log2_n` (collapsing onto `Σ narrow_band`). The
// two reductions partition the full row-order `Σ_n log2_n` exactly.
//
// Several Step 9 / Step 10 consumers read *both* totals together — the
// narrow-block companion and the wide-block companion of the same
// width-gated `log2(LTmin_lin_n)` column. Calling step 66 and step 67
// back-to-back invokes the caller's `LTg(ω)` callback *twice* over the
// full `Σ_{n=1..=32} (ωhigh_n − ωlow_n + 1)` FFT-line range, because
// each total independently re-derives step 65's split struct. For an
// FFT-line callback whose evaluation is non-trivial (the realistic
// Step 8 case, where `LTg(ω)` is itself a per-line global-masking
// reduction), that doubles the per-line work for no algebraic gain.
//
// Step 68 fuses the two reductions: it invokes step 65 *once*, then
// sums the `narrow_band` and `wide_band` subarrays of the single
// returned struct independently, returning the pair
// `(narrow_total, wide_total)`. The callback fan-out is exactly half
// of the back-to-back step 67 + step 66 pairing — one pass over the
// FFT-line range instead of two — while the two scalars are
// bit-identical to the standalone step 67 / step 66 results because
// each is the same cell-wise sum over the same subarray. No new spec
// arithmetic is introduced beyond `+`.
// ---------------------------------------------------------------------------

/// §D.1 Step 8 width-gated paired `(narrow_total, wide_total)` signed
/// bit-budget reduction over Table D.5, computed with a single step-65
/// invocation. Returns the tuple `(narrow_total, wide_total)`:
///
/// ```text
/// narrow_total = Σ_{n=1..=32} (1 − width_n) · log2(10^(LTmin_n / 10))
///              = Σ_{i=0..=11} narrow_band[i]      (since width_n = 1 for n ∈ 13..=32)
/// wide_total   = Σ_{n=1..=32}      width_n  · log2(10^(LTmin_n / 10))
///              = Σ_{j=0..=19} wide_band[j]        (since width_n = 0 for n ∈ 1..=12)
/// ```
///
/// This is the fused presentation of Phase 2 step 67's narrow-band
/// complementary total
/// ([`coder_partition_d5_ltmin_log2_narrow_band_bit_budget_total`]) and
/// Phase 2 step 66's wide-band total
/// ([`coder_partition_d5_ltmin_log2_wide_band_bit_budget_total`]). The
/// two scalars are bit-identical to the standalone step 67 / step 66
/// results — each is the same cell-wise sum over the same subarray of
/// step 65's split struct — but they are produced from a **single**
/// call to [`coder_partition_d5_ltmin_log2_row_order_by_width`] rather
/// than two.
///
/// **Tuple order.** The pair is `(narrow_total, wide_total)` — narrow
/// first, matching the partition order of step 65's struct
/// (`narrow_band` field declared before `wide_band`) and the
/// `(1 − width_n)` / `width_n` companion order. The first element is
/// step 67's value; the second is step 66's value.
///
/// **Callback fan-out.** The `LTg(ω)` callback is invoked exactly the
/// same number of times step 65 invokes it (one call per FFT line in
/// `Σ_{n=1..=32} (ωhigh_n − ωlow_n + 1)`) — **half** the fan-out of
/// calling [`coder_partition_d5_ltmin_log2_narrow_band_bit_budget_total`]
/// and [`coder_partition_d5_ltmin_log2_wide_band_bit_budget_total`]
/// back-to-back, which would re-walk the FFT-line range twice. For a
/// non-trivial per-line callback this halves the per-line work.
///
/// **Pairing identity.** By the disjointness of step 65's
/// `narrow_band` / `wide_band` fields, `narrow_total + wide_total`
/// equals the unweighted full row-order `Σ_n log2_n` exactly — the
/// pair partitions the row-order signed bit-budget budget without
/// losing the width-column split (the same identity Phase 2 step 67
/// documents against the step 66 / 67 pair, now produced in one pass).
///
/// **Sign semantics.** Each total inherits step 66 / 67's "signed"
/// qualifier: `log2(linear)` is unbounded below as `linear → 0`, so a
/// partition below 0 dB contributes a strictly negative cell and one
/// above 0 dB a strictly positive cell. Both totals accumulate both
/// signs without clipping.
///
/// **Width invariant.** `narrow_total` reads only step 65's
/// `narrow_band` subarray (12 cells, all `width_n = 0`); `wide_total`
/// reads only the `wide_band` subarray (20 cells, all `width_n = 1`).
/// The split is structurally pinned at index 12 — Phase 2 step 60's
/// row-order width vector matches `[0; 12]` followed by `[1; 20]`
/// exactly and Phase 2 step 65 inherits the split point verbatim. The
/// fused accessor reads the two subarrays of the single struct without
/// re-deriving the split.
///
/// **Boundary semantics.** Inherits Phase 2 step 65's (and through it
/// step 64 / 63 / 62 / 61 / 58 / 59's) inclusive-on-both-ends
/// `LTmin_n` reduction semantics unchanged. Neither sum has boundary
/// semantics of its own — each is a pure cell-wise addition reading
/// every cell of its subarray exactly once.
///
/// **Finiteness.** `narrow_total` is finite when every narrow cell of
/// step 65 is finite; `wide_total` when every wide cell is finite. A
/// degenerate callback driving any cell to `±INFINITY` (per step
/// 64's `10^x` / step 65's `log2` lift) propagates that infinity into
/// the corresponding total independently — an infinity in the narrow
/// block does not contaminate `wide_total` and vice versa.
///
/// **Determinism.** A pure function of the callback: invoking the
/// fused reduction twice with the same callback returns the same pair.
///
/// **Implementation.** Calls
/// [`coder_partition_d5_ltmin_log2_row_order_by_width`] **once**
/// (folding the caller's callback over every recoverable partition's
/// FFT-line range) and sums the `narrow_band` and `wide_band`
/// subarrays of the single returned struct independently. The
/// reduction is structurally pinned at the same index 12 the static
/// width-column transition lives at — Phase 2 step 60's row-order
/// width vector matches the pattern `[0; 12]` followed by `[1; 20]`
/// exactly and Phase 2 step 65 inherits the split point verbatim.
///
/// Provenance: only the Phase 2 step 65 width-gated `log2(LTmin_lin_n)`
/// column accessor [`coder_partition_d5_ltmin_log2_row_order_by_width`]
/// (and through it the Phase 2 step 64 width-gated `LTmin_n` linear
/// accessor, the Phase 2 step 63 width-gated `LTmin_n` dB accessor,
/// the Phase 2 step 62 width-gated paired-vector accessor, the
/// Phase 2 step 61 row-order paired vector, the Phase 2 step 59 / 60
/// row-order LTmin and width vectors, the Phase 2 step 58
/// per-partition reducer, the Phase 2 step 52 per-partition width
/// accessor, and the underlying Table D.5 transcription in
/// `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
/// §"Table D.5 - Layer I and Layer II coder partition table") is
/// consulted. The reduction is plain `f64` addition; no external
/// implementation was read.
#[must_use]
pub fn coder_partition_d5_ltmin_log2_paired_bit_budget_totals<F>(ltg_per_line: F) -> (f64, f64)
where
    F: Fn(u16) -> f64,
{
    let split = coder_partition_d5_ltmin_log2_row_order_by_width(ltg_per_line);
    let mut narrow_total = 0.0_f64;
    for &cell in &split.narrow_band {
        narrow_total += cell;
    }
    let mut wide_total = 0.0_f64;
    for &cell in &split.wide_band {
        wide_total += cell;
    }
    (narrow_total, wide_total)
}

// =====================================================================
// Annex D Model 1 — §D.1 Step 9 signal-to-mask-ratio over Table D.5
// (Phase 2 step 69 / r268).
//
// Spec context (clause D.1, ISO/IEC 11172-3:1993, informative annex,
// printed p.115 — "Step 9: Calculation of the signal-to-mask-ratio"):
//
//   The signal-to-mask ratio
//
//       SMR_sb(n) = Lsb(n) − LTmin(n)   dB
//
//   is computed for every subband n.
//
// The two operands come from earlier §D.1 steps:
//
//   * `Lsb(n)` is the Step 2 sound pressure level (printed p.110:
//     `Lsb(n) = MAX[X(k), 20·log10(scf_max(n)·32768) − 10]` dB,
//     reading the FFT power density spectrum `X(k)` and the
//     transmitted scalefactor maximum). Steps 1–2 (FFT analysis +
//     SPL determination) are not yet landed — they sit behind the
//     PNG-only Tables D.1 / D.2 transcription gap blocking the
//     masker-selection pipeline — so `Lsb(n)` enters as a
//     caller-supplied per-partition callback, the same dependency-
//     injection pattern Phase 2 steps 58–68 use for `LTg(ω)`.
//   * `LTmin(n)` is the Step 8 minimum masking threshold (printed
//     p.114: `LTmin(n) = MIN[ LTg(i) ]` over `f(i)` in subband `n`),
//     already produced per Table D.5 coder partition by the Phase 2
//     step 58–63 chain in its width-gated dB presentation
//     (`coder_partition_d5_ltmin_db_row_order_by_width`).
//
// Composition rather than introduction: the only new spec arithmetic
// is the Step 9 subtraction itself — one `Lsb(n) − LTmin(n)` per
// row of Table D.5. The presentation mirrors the step 63 width-gated
// split (12 narrow + 20 wide) the downstream Layer I / Layer II
// bit-allocation loop branches on: the SMR is the loop's per-band
// input (the bit-allocation iteration assigns bits to the subband
// with the worst mask-to-noise ratio, which is seeded from this
// per-band SMR vector).
// =====================================================================

/// Per-partition §D.1 Step 9 signal-to-mask ratio `SMR_n` (dB) over
/// Table D.5, split by the `width_n` column.
///
/// Produced by [`coder_partition_d5_smr_db_row_order_by_width`]
/// (Phase 2 step 69 / r268).
///
/// **Field semantics.** `narrow_band` carries the contiguous prefix
/// of `SMR_n` values (dB) for the rows with `width_n = 0`
/// (partitions `n ∈ 1..=12`, the lower FFT-line block); element `i`
/// carries `SMR_{i + 1}`. `wide_band` carries the contiguous suffix
/// for the rows with `width_n = 1` (partitions `n ∈ 13..=32`, the
/// upper FFT-line block); element `j` carries `SMR_{j + 13}`. The
/// split point (12) is the single 0 → 1 transition in Table D.5's
/// `width_n` column, pinned by Phase 2 step 60's row-order width
/// vector and inherited verbatim from Phase 2 step 63's split.
///
/// **Sign semantics.** Positive `SMR_n` means the partition's signal
/// level sits above its minimum masking threshold (audible content
/// needing coding bits); negative means the partition is fully
/// masked (the threshold exceeds the signal, so quantization noise
/// up to the threshold is inaudible). Both signs are preserved
/// without clipping.
#[derive(Clone, Copy, Debug)]
pub struct CoderPartitionD5SmrByWidth {
    /// Per-partition signal-to-mask ratio `SMR_n` (dB) for the
    /// contiguous prefix of rows with `width_n = 0` (partitions
    /// `n ∈ 1..=12`). Twelve elements in row order; element `i`
    /// carries `SMR_{i + 1}`.
    pub narrow_band: [f64; 12],
    /// Per-partition signal-to-mask ratio `SMR_n` (dB) for the
    /// contiguous suffix of rows with `width_n = 1` (partitions
    /// `n ∈ 13..=32`). Twenty elements in row order; element `j`
    /// carries `SMR_{j + 13}`.
    pub wide_band: [f64; 20],
}

/// §D.1 Step 9 width-gated signal-to-mask-ratio computation over
/// Table D.5 (ISO/IEC 11172-3:1993 Annex D, printed p.115):
///
/// ```text
/// SMR_n = Lsb(n) − LTmin_n   dB        for n ∈ 1..=32
///
/// narrow_band[i] = SMR_{i + 1}    for i ∈ 0..=11  (width_n = 0)
/// wide_band[j]   = SMR_{j + 13}   for j ∈ 0..=19  (width_n = 1)
/// ```
///
/// `Lsb(n)` is the §D.1 Step 2 sound pressure level of partition
/// `n`, supplied by the caller as the `lsb_per_partition` callback
/// (Steps 1–2 are not yet landed; they remain behind the PNG-only
/// Tables D.1 / D.2 transcription gap). `LTmin_n` is the §D.1
/// Step 8 minimum masking threshold (printed p.114:
/// `LTmin(n) = MIN[LTg(i)]` over the partition's FFT lines), derived
/// here from the caller's `ltg_per_line` callback through the Phase 2
/// step 63 width-gated dB accessor.
///
/// **Index convention.** `lsb_per_partition` receives the spec's
/// 1-based partition index `n ∈ 1..=32`, invoked exactly once per
/// partition in ascending row order (`1, 2, …, 32`). The returned
/// subarrays are 0-based: `narrow_band[i]` carries `SMR_{i + 1}`;
/// `wide_band[j]` carries `SMR_{j + 13}`. Partition 0 (the
/// degenerate single-line `width_n = 0` row) is excluded, matching
/// the step 58–63 chain's recoverable range.
///
/// **Composition.** Calls Phase 2 step 63's
/// [`coder_partition_d5_ltmin_db_row_order_by_width`] once (folding
/// `ltg_per_line` over every recoverable partition's FFT-line range —
/// the callback fan-out is exactly one step-63 pass, one call per
/// FFT line in `Σ_{n=1..=32} (ωhigh_n − ωlow_n + 1)`), then applies
/// the Step 9 subtraction `Lsb(n) − LTmin_n` cell-wise. The
/// subtraction is the only new spec arithmetic this step introduces.
///
/// **Sign semantics.** Positive cells mark partitions whose signal
/// exceeds the masking threshold (the bit-allocation loop must spend
/// bits there); negative cells mark fully-masked partitions. A
/// callback pair driving `Lsb(n)` below `LTmin_n` produces a strictly
/// negative cell; above, strictly positive. No clipping.
///
/// **Boundary semantics.** Inherits Phase 2 step 63's (and through
/// it step 62 / 61 / 59 / 58's) inclusive-on-both-ends `LTmin_n`
/// reduction semantics unchanged: a sharp `LTg` dip on a shared
/// boundary line enters both adjacent partitions' `LTmin` and hence
/// raises both partitions' SMR. The Step 9 subtraction itself has no
/// boundary semantics — it is a pure per-row operation.
///
/// **Determinism.** A pure function of the two callbacks: invoking
/// twice with the same pure callbacks returns identical cells.
///
/// Provenance: the Step 9 formula `SMR_sb(n) = Lsb(n) − LTmin(n)` dB
/// is transcribed from ISO/IEC 11172-3:1993 Annex D §D.1 Step 9
/// (printed p.115, `docs/audio/mp3/ISO_IEC_11172-3-MP3-1993.pdf`);
/// the `LTmin_n` operand comes from the Phase 2 step 63 width-gated
/// dB accessor [`coder_partition_d5_ltmin_db_row_order_by_width`]
/// (and through it the cascade down to the Table D.5 transcription in
/// `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
/// §"Table D.5 — Layer I and Layer II coder partition table"). No
/// external implementation was read.
#[must_use]
pub fn coder_partition_d5_smr_db_row_order_by_width<L, F>(
    lsb_per_partition: L,
    ltg_per_line: F,
) -> CoderPartitionD5SmrByWidth
where
    L: Fn(u16) -> f64,
    F: Fn(u16) -> f64,
{
    let ltmin = coder_partition_d5_ltmin_db_row_order_by_width(ltg_per_line);
    let mut narrow = [0.0_f64; 12];
    let mut wide = [0.0_f64; 20];
    for (i, &ltmin_db) in ltmin.narrow_band.iter().enumerate() {
        // narrow rows carry partitions n = i + 1 (spec 1-based).
        narrow[i] = lsb_per_partition(i as u16 + 1) - ltmin_db;
    }
    for (j, &ltmin_db) in ltmin.wide_band.iter().enumerate() {
        // wide rows carry partitions n = j + 13 (spec 1-based).
        wide[j] = lsb_per_partition(j as u16 + 13) - ltmin_db;
    }
    CoderPartitionD5SmrByWidth {
        narrow_band: narrow,
        wide_band: wide,
    }
}

// =====================================================================
// Annex D Model 1 — §D.1 Step 9 row-order signal-to-mask-ratio vector
// over Table D.5 (Phase 2 step 70 / r269).
//
// Spec context (clause D.1, ISO/IEC 11172-3:1993, informative annex,
// printed p.115 — "Step 9: Calculation of the signal-to-mask-ratio"):
//
//   SMR_sb(n) = Lsb(n) − LTmin(n)   dB     computed for every
//   subband n.
//
// Phase 2 step 69 (r268) landed the Step 9 subtraction in the
// width-gated split presentation (12 narrow + 20 wide subarrays).
// The Layer I / Layer II bit-allocation loop, however, walks the 32
// coder partitions of Table D.5 **in row order**, pairing each
// partition's `SMR_n` with the same row's `width_n` flag (Phase 2
// step 60's `[u16; 32]` vector) and `LTmin_n` value (Phase 2 step
// 59's `[f64; 32]` vector) at the same array index — the same
// row-order vector form steps 59 / 60 / 61 expose for the Step 8
// columns. This step supplies the missing row-order presentation of
// the Step 9 output: a single `[f64; 32]` with element `i` carrying
// `SMR_{i + 1}`.
//
// Composition rather than introduction: the `LTmin_n` operand comes
// from one Phase 2 step 59 pass
// (`coder_partition_d5_ltg_min_row_order`); the Step 9 subtraction
// `Lsb(n) − LTmin_n` per row is the same arithmetic step 69
// introduced — no new spec arithmetic appears here. Because Phase 2
// step 63's by-width `LTmin_n` cells are index-preserving copies of
// the step 59 row-order vector (via the step 61 / 62 chain), the
// returned vector is bit-identical to step 69's split read back in
// row order: `out[0..12] == narrow_band`, `out[12..32] == wide_band`.
// =====================================================================

/// §D.1 Step 9 row-order signal-to-mask-ratio vector
/// `[SMR_1, SMR_2, …, SMR_32]` (dB) over Table D.5
/// (ISO/IEC 11172-3:1993 Annex D, printed p.115):
///
/// ```text
/// out[i] = SMR_{i + 1} = Lsb(i + 1) − LTmin_{i + 1}   dB
///                                          for i ∈ 0..=31
/// ```
///
/// The row-order vector is the per-frame form the Layer I / Layer II
/// bit-allocation loop consumes: it pairs `out[i]` with the same
/// row's `width_{i + 1}` flag ([`coder_partition_d5_width_row_order`],
/// Phase 2 step 60) and `LTmin_{i + 1}` value
/// ([`coder_partition_d5_ltg_min_row_order`], Phase 2 step 59) at the
/// same array index, then iterates bit assignment on the partition
/// with the worst mask-to-noise ratio seeded from this SMR vector.
///
/// `Lsb(n)` is the §D.1 Step 2 sound pressure level of partition `n`
/// (printed p.110: `Lsb(n) = MAX[X(k), 20·log10(scf_max(n)·32768) −
/// 10]` dB), supplied by the caller as the `lsb_per_partition`
/// callback — Steps 1–2 (FFT analysis + SPL determination) remain
/// behind the PNG-only Tables D.1 / D.2 transcription gap, the same
/// dependency-injection pattern Phase 2 steps 58–69 use for
/// `LTg(ω)`. `LTmin_n` is the §D.1 Step 8 minimum masking threshold
/// (printed p.114: `LTmin(n) = MIN[LTg(i)]` over the partition's FFT
/// lines), derived here from the caller's `ltg_per_line` callback
/// through the Phase 2 step 59 row-order reducer.
///
/// **Index convention.** `lsb_per_partition` receives the spec's
/// 1-based partition index `n ∈ 1..=32`, invoked exactly once per
/// partition in ascending row order (`1, 2, …, 32`). The returned
/// slice is 0-based: `out[i]` carries `SMR_{i + 1}`. Partition 0
/// (the degenerate single-line `width_n = 0` row) is excluded,
/// matching the step 58–69 chain's recoverable range.
///
/// **Composition.** Calls Phase 2 step 59's
/// [`coder_partition_d5_ltg_min_row_order`] once (folding
/// `ltg_per_line` over every recoverable partition's FFT-line range —
/// the callback fan-out is exactly one step-59 pass, one call per FFT
/// line in `Σ_{n=1..=32} (ωhigh_n − ωlow_n + 1)`), then applies the
/// Step 9 subtraction `Lsb(n) − LTmin_n` per row. No new spec
/// arithmetic beyond the step 69 subtraction is introduced.
///
/// **Bit-identity with the step 69 split.** Phase 2 step 63's
/// by-width `LTmin_n` cells are index-preserving copies of the step
/// 59 row-order vector (through the step 61 / 62 chain), so this
/// vector equals Phase 2 step 69's
/// [`coder_partition_d5_smr_db_row_order_by_width`] output read back
/// in row order bit-for-bit: `out[i] == narrow_band[i]` for
/// `i ∈ 0..=11` and `out[j + 12] == wide_band[j]` for `j ∈ 0..=19`.
///
/// **Sign semantics.** Positive cells mark partitions whose signal
/// exceeds the masking threshold (the bit-allocation loop must spend
/// bits there); negative cells mark fully-masked partitions. No
/// clipping.
///
/// **Boundary semantics.** Inherits Phase 2 step 59's (and through it
/// step 58's) inclusive-on-both-ends `LTmin_n` reduction semantics
/// unchanged: a sharp `LTg` dip on a shared boundary line enters both
/// adjacent partitions' `LTmin` and hence raises both partitions'
/// SMR. The Step 9 subtraction itself has no boundary semantics — it
/// is a pure per-row operation.
///
/// **Determinism.** A pure function of the two callbacks: invoking
/// twice with the same pure callbacks returns identical cells.
///
/// Provenance: the Step 9 formula `SMR_sb(n) = Lsb(n) − LTmin(n)` dB
/// is transcribed from ISO/IEC 11172-3:1993 Annex D §D.1 Step 9
/// (printed p.115, `docs/audio/mp3/ISO_IEC_11172-3-MP3-1993.pdf`);
/// the `LTmin_n` operand comes from the Phase 2 step 59 row-order
/// reducer [`coder_partition_d5_ltg_min_row_order`] (and through it
/// the cascade down to the Table D.5 transcription in
/// `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
/// §"Table D.5 — Layer I and Layer II coder partition table"). No
/// external implementation was read.
#[must_use]
pub fn coder_partition_d5_smr_db_row_order<L, F>(lsb_per_partition: L, ltg_per_line: F) -> [f64; 32]
where
    L: Fn(u16) -> f64,
    F: Fn(u16) -> f64,
{
    let ltmin = coder_partition_d5_ltg_min_row_order(ltg_per_line);
    let mut out = [0.0_f64; 32];
    for (i, &ltmin_db) in ltmin.iter().enumerate() {
        // Row i carries partition n = i + 1 (spec 1-based).
        out[i] = lsb_per_partition(i as u16 + 1) - ltmin_db;
    }
    out
}

// =====================================================================
// Annex D Model 1 — §D.1 Step 9 paired `(SMR_n, width_n)` row-order
// vector over Table D.5 (Phase 2 step 71 / r270).
//
// Spec context (ISO/IEC 11172-3:1993 Annex D §D.1 Step 9, printed
// p.115; Table D.5 — Layer I and Layer II coder partition table):
//
//   Phase 2 step 70 (r269) exposed the bare row-order signal-to-mask-
//   ratio vector `[SMR_1, …, SMR_32]` (dB). The Layer I / Layer II
//   bit-allocation loop walks the 32 coder partitions in row order and
//   at every row consumes **both** the partition's SMR value (the
//   per-partition signal-to-mask ratio that seeds its mask-to-noise
//   iteration) **and** its `width_n` column flag (whether the
//   partition spans one or more than one Layer I / Layer II coder
//   partition row, driving single-line vs multi-line per-partition
//   bit targeting). The two columns are read paired, in lockstep, the
//   same way the §D.1 Step 8 paired `(LTmin_n, width_n)` vector (Phase
//   2 step 61 / r260) is read.
//
// This step is the SMR analogue of step 61: a pure index-aligned zip
// of step 70's row-order SMR vector
// (`coder_partition_d5_smr_db_row_order`, closed over the caller's two
// callbacks) with step 60's static row-order width vector
// (`coder_partition_d5_width_row_order`). No spec arithmetic is
// introduced beyond the step 70 subtraction already present in the SMR
// column — only the per-row pairing of the two existing row-order
// columns at the same array index, which is exactly the per-row input
// the bit-allocation loop reads.
// =====================================================================

/// A single row of the Layer I / Layer II coder-partition bit-
/// allocation input in §D.1 Step 9 form: the per-partition signal-to-
/// mask ratio `SMR_n` (dB) paired with the partition's `width_n`
/// column (0 for `n ∈ 1..=12`; 1 for `n ∈ 13..=32`) at the same row
/// index in row-order Table D.5 presentation.
///
/// Produced by [`coder_partition_d5_smr_row_order`] (Phase 2 step 71 /
/// r270). The SMR analogue of Phase 2 step 61's
/// [`CoderPartitionD5Reduction`] (which pairs the §D.1 Step 8
/// `LTmin_n` column with `width_n`).
///
/// **Field semantics.** `smr_db` carries the §D.1 Step 9 signal-to-
/// mask ratio `SMR_n = Lsb(n) − LTmin_n` (dB), as computed by Phase 2
/// step 70's [`coder_partition_d5_smr_db_row_order`] from the caller's
/// two callbacks. `width_n` is the static Table D.5 column value read
/// by Phase 2 step 52's per-partition accessor
/// [`coder_partition_d5_width`]. The two columns are orthogonal: the
/// SMR column closes over the caller's `Lsb(n)` and `LTg(ω)` callbacks
/// (run-time-dependent), the width column is a pure constant of the
/// table.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoderPartitionD5Smr {
    /// Per-partition signal-to-mask ratio `SMR_n = Lsb(n) − LTmin_n`
    /// (dB), as computed by Phase 2 step 70's
    /// [`coder_partition_d5_smr_db_row_order`]. Positive = audible
    /// content needing bits; negative = fully masked. No clipping.
    pub smr_db: f64,
    /// Partition's `width_n` column from Table D.5 — `0` for
    /// `n ∈ 1..=12`, `1` for `n ∈ 13..=32`. The same value Phase 2
    /// step 52's [`coder_partition_d5_width`] returns for the
    /// partition.
    pub width_n: u16,
}

/// §D.1 Step 9 paired row-order `[(SMR_n, width_n), …]` vector for
/// every Layer I / Layer II coder partition `n ∈ 1..=32`. Element `i`
/// of the returned `[CoderPartitionD5Smr; 32]` holds the
/// `(SMR_{i + 1}, width_{i + 1})` pair (the spec's 1-based `n` in
/// 0-based array form):
///
/// ```text
/// out[i].smr_db  = Lsb(i + 1) − LTmin_{i + 1}   dB
/// out[i].width_n = width_{i + 1}   (∈ {0, 1})
/// ```
///
/// The vector is the per-frame paired input the Layer I / Layer II
/// bit-allocation loop consumes — at every row the loop reads both
/// columns together as the partition's "per-row brief" (signal-to-mask
/// ratio + width flag), then iterates bit assignment on the partition
/// with the worst mask-to-noise ratio. It is the §D.1 Step 9 analogue
/// of Phase 2 step 61's paired §D.1 Step 8
/// `(LTmin_n, width_n)` vector [`coder_partition_d5_reduction_row_order`].
///
/// **Index convention.** 0-based on the returned slice; element `i`
/// holds `(SMR_{i + 1}, width_{i + 1})`. The spec's 1-based partition
/// index `n ∈ 1..=32` maps to array index `i = n - 1 ∈ 0..=31`.
/// Partition 0 (the degenerate single-line `width_n = 0` row) is
/// excluded, matching the step 58–70 chain's recoverable range.
///
/// **Callbacks.** `lsb_per_partition` is the §D.1 Step 2 sound
/// pressure level `Lsb(n)` (printed p.110), supplied per partition —
/// Steps 1–2 (FFT analysis + SPL determination) remain behind the
/// PNG-only Tables D.1 / D.2 transcription gap, the same dependency-
/// injection pattern Phase 2 steps 58–70 use. `ltg_per_line` is the
/// §D.1 Step 7 per-FFT-line global masking threshold `LTg(ω)`, reduced
/// to each partition's `LTmin_n` through the Phase 2 step 59 row-order
/// reducer.
///
/// **Composition.** A pure index-aligned zip of Phase 2 step 70's
/// row-order SMR vector [`coder_partition_d5_smr_db_row_order`] (closed
/// over both caller callbacks) with Phase 2 step 60's row-order width
/// vector [`coder_partition_d5_width_row_order`]. No spec arithmetic is
/// introduced beyond the step 70 subtraction already in the SMR column
/// — only the per-row pairing of the two existing row-order columns at
/// the same array index.
///
/// **Caller cost.** `lsb_per_partition` is invoked exactly once per
/// partition `n ∈ 1..=32` in ascending row order; the `ltg_per_line`
/// callback is invoked exactly as many times as Phase 2 step 70 (= one
/// step-59 pass, one call per FFT line in
/// `Σ_{n=1..=32} (ωhigh_n − ωlow_n + 1)`). The width vector adds no
/// callback invocations.
///
/// **Bit-identity with step 70.** The `smr_db` column equals Phase 2
/// step 70's [`coder_partition_d5_smr_db_row_order`] output cell-for-
/// cell (`out[i].smr_db == step70[i]`), and the `width_n` column equals
/// Phase 2 step 60's [`coder_partition_d5_width_row_order`] cell-for-
/// cell.
///
/// **Boundary / sign semantics.** Inherits Phase 2 step 70's unchanged:
/// a sharp `LTg` dip on a shared boundary line raises both adjacent
/// partitions' SMR; positive cells mark partitions whose signal exceeds
/// the masking threshold, negative cells fully-masked partitions, no
/// clipping. The width column has no boundary semantics (it is a static
/// per-row table value).
///
/// **Determinism.** A pure function of the two callbacks: invoking
/// twice with the same pure callbacks returns identical cells.
///
/// Provenance: only the Phase 2 step 70 row-order SMR vector
/// [`coder_partition_d5_smr_db_row_order`] and the Phase 2 step 60
/// row-order width vector [`coder_partition_d5_width_row_order`] (and
/// through them the §D.1 Step 9 formula transcribed from
/// ISO/IEC 11172-3:1993 Annex D §D.1 Step 9 printed p.115 in
/// `docs/audio/mp3/ISO_IEC_11172-3-MP3-1993.pdf`, the Phase 2 step 52
/// per-partition width accessor [`coder_partition_d5_width`], and the
/// Table D.5 transcription in
/// `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
/// §"Table D.5 - Layer I and Layer II coder partition table") are
/// consulted. The
/// paired-row-order reading is the spec's per Annex D §D.1 Step 9
/// (informative Model 1) row-by-row presentation; no external
/// implementation was read.
#[must_use]
pub fn coder_partition_d5_smr_row_order<L, F>(
    lsb_per_partition: L,
    ltg_per_line: F,
) -> [CoderPartitionD5Smr; 32]
where
    L: Fn(u16) -> f64,
    F: Fn(u16) -> f64,
{
    let smr = coder_partition_d5_smr_db_row_order(lsb_per_partition, ltg_per_line);
    let widths = coder_partition_d5_width_row_order();
    let mut out = [CoderPartitionD5Smr {
        smr_db: 0.0,
        width_n: 0,
    }; 32];
    for i in 0..32 {
        out[i] = CoderPartitionD5Smr {
            smr_db: smr[i],
            width_n: widths[i],
        };
    }
    out
}

// =====================================================================
// Annex C §C.1.5.2.7 "Bit allocation" — per-partition mask-to-noise
// ratio `MNR_n = SNR_n − SMR_n` row-order vector over Table D.5
// (Phase 2 step 72 / r271).
//
// Spec context (ISO/IEC 11172-3:1993 Annex C §C.1.5.2.7 "Bit
// allocation", printed p.73):
//
//   "The allocation procedure is an iterative procedure where, in each
//    iteration step the number of levels of the subband that has the
//    greatest benefit is increased. First the mask-to-noise ratio
//    'MNR' for each subband is calculated by subtracting from the
//    signal-to-noise-ratio 'SNR' the signal-to-mask-ratio 'SMR':
//        MNR = SNR − SMR
//    The signal-to-noise-ratio can be found in table C.5 'Layer II
//    Signal-to-Noise Ratios'. The signal-to-mask-ratio is the output
//    of the psychoacoustic model."
//
// Phase 2 step 71 (r270) exposed the §D.1 Step 9 paired `(SMR_n,
// width_n)` row-order vector — the "output of the psychoacoustic
// model" the §C.1.5.2.7 procedure consumes. This step takes the very
// first arithmetic of the bit-allocation iterative loop — the
// per-subband `MNR_n = SNR_n − SMR_n` initialisation, computed once
// per partition before the iterative level-bumping begins — and
// presents it as a row-order vector seeded directly off the step-71
// paired SMR vector.
//
// The `SNR_n` term is the Table C.5 "Layer II Signal-to-Noise Ratios"
// column, supplied per partition through a caller callback (exactly
// the dependency-injection pattern Phase 2 steps 58–71 use for the
// §D.1 Step 2 `Lsb(n)` term). Table C.5 lives behind the same
// numeric-table transcription gap as Tables D.1 / D.2, so the
// quantization-step-count → SNR mapping is injected, not transcribed.
//
// No spec arithmetic is introduced beyond the verbatim `SNR − SMR`
// subtraction above; the SMR column is bit-identical to step 71's and
// the `width_n` column passes through unchanged.
// =====================================================================

/// A single row of the Layer I / Layer II bit-allocation iterative
/// procedure in §C.1.5.2.7 initial form: the per-partition mask-to-
/// noise ratio `MNR_n = SNR_n − SMR_n` (dB) paired with the
/// partition's `width_n` column at the same row index in row-order
/// Table D.5 presentation.
///
/// Produced by [`coder_partition_d5_mnr_row_order`] (Phase 2 step 72 /
/// r271). The per-iteration successor of Phase 2 step 71's
/// [`CoderPartitionD5Smr`] (which carries the bare §D.1 Step 9 `SMR_n`
/// column): the §C.1.5.2.7 loop subtracts the supplied `SNR_n` from
/// each `SMR_n` to obtain the `MNR_n` it then minimises over.
///
/// **Field semantics.** `mnr_db` carries the §C.1.5.2.7 mask-to-noise
/// ratio `MNR_n = SNR_n − SMR_n` (dB), where `SMR_n` is the Phase 2
/// step 71 psychoacoustic-model output and `SNR_n` is the caller-
/// supplied Table C.5 value for the partition. The `smr_db` column is
/// preserved verbatim from step 71 (the loop re-uses it across
/// iterations), and `width_n` is the static Table D.5 column.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoderPartitionD5Mnr {
    /// Per-partition mask-to-noise ratio `MNR_n = SNR_n − SMR_n` (dB),
    /// per the §C.1.5.2.7 verbatim definition. A **larger** `MNR_n`
    /// means more margin above the masking threshold; the iterative
    /// procedure minimises the total by bumping the subband with the
    /// **smallest** `MNR_n` first ("the subband that has the greatest
    /// benefit"). No clipping.
    pub mnr_db: f64,
    /// The §D.1 Step 9 signal-to-mask ratio `SMR_n` (dB) preserved
    /// verbatim from Phase 2 step 71's
    /// [`CoderPartitionD5Smr::smr_db`] — the psychoacoustic-model
    /// output the loop re-reads each iteration.
    pub smr_db: f64,
    /// Partition's `width_n` column from Table D.5 — `0` for
    /// `n ∈ 1..=12`, `1` for `n ∈ 13..=32`. Identical to Phase 2
    /// step 71's [`CoderPartitionD5Smr::width_n`].
    pub width_n: u16,
}

/// §C.1.5.2.7 mask-to-noise-ratio row-order
/// `[(MNR_n, SMR_n, width_n), …]` vector for every Layer I / Layer II
/// coder partition `n ∈ 1..=32`. Element `i` of the returned
/// `[CoderPartitionD5Mnr; 32]` holds the partition `n = i + 1` triple
/// (the spec's 1-based `n` in 0-based array form):
///
/// ```text
/// out[i].mnr_db  = SNR_{i + 1} − SMR_{i + 1}   dB
/// out[i].smr_db  = SMR_{i + 1}                 dB
/// out[i].width_n = width_{i + 1}   (∈ {0, 1})
/// ```
///
/// This is the very first arithmetic of the §C.1.5.2.7 bit-allocation
/// iterative procedure — the per-subband `MNR` initialisation computed
/// once before the loop's level-bumping begins. The iterative loop
/// then repeatedly "determines the minimal MNR of all subbands" and
/// increases the quantization accuracy of that subband; this primitive
/// supplies the loop's row-order starting `MNR_n` (and re-presents the
/// `SMR_n` it derives from, since the loop re-uses `SMR_n` whenever a
/// subband's `SNR_n` advances to the next quantization-table entry).
/// It is the per-iteration successor of Phase 2 step 71's paired
/// §D.1 Step 9 `(SMR_n, width_n)` vector
/// [`coder_partition_d5_smr_row_order`].
///
/// **Index convention.** 0-based on the returned slice; element `i`
/// holds the partition-`(i + 1)` triple. The spec's 1-based partition
/// index `n ∈ 1..=32` maps to array index `i = n - 1 ∈ 0..=31`.
///
/// **Callbacks.** `snr_per_partition` is the Table C.5 "Layer II
/// Signal-to-Noise Ratios" value `SNR_n` (dB) for the partition's
/// current quantization-table entry, supplied per partition — Table
/// C.5 lives behind the same numeric-table transcription gap as
/// Tables D.1 / D.2, so the value is injected, the same dependency-
/// injection pattern Phase 2 steps 58–71 use for the §D.1 Step 2
/// `Lsb(n)` term. `lsb_per_partition` and `ltg_per_line` are forwarded
/// unchanged to Phase 2 step 71's
/// [`coder_partition_d5_smr_row_order`] to obtain the `SMR_n` /
/// `width_n` columns.
///
/// **Composition.** `out[i].smr_db` / `out[i].width_n` come verbatim
/// from Phase 2 step 71's paired SMR vector (closed over
/// `lsb_per_partition` / `ltg_per_line`); `out[i].mnr_db` is the single
/// §C.1.5.2.7 subtraction `snr_per_partition(i + 1) − smr_db`. No spec
/// arithmetic is introduced beyond that verbatim subtraction.
///
/// **Caller cost.** `snr_per_partition` is invoked exactly once per
/// partition `n ∈ 1..=32` in ascending row order; `lsb_per_partition`
/// and `ltg_per_line` are invoked exactly as many times as Phase 2
/// step 71 (= one step-70 pass).
///
/// **SMR-column identity with step 71.** `out[i].smr_db` equals Phase
/// 2 step 71's [`coder_partition_d5_smr_row_order`] `smr_db` cell-for-
/// cell, and `out[i].width_n` its `width_n` cell-for-cell.
///
/// **Sign semantics.** Per the §C.1.5.2.7 definition `MNR = SNR − SMR`:
/// a subband with a high `SMR_n` (audible signal needing protection)
/// and a low `SNR_n` (coarse current quantization) yields a small —
/// possibly negative — `MNR_n`, marking it as the subband "that has the
/// greatest benefit" from a finer quantization step. No clipping.
///
/// **Determinism.** A pure function of the three callbacks: invoking
/// twice with the same pure callbacks returns identical cells.
///
/// Provenance: only the Phase 2 step 71 paired SMR vector
/// [`coder_partition_d5_smr_row_order`] (and through it the §D.1
/// Step 9 formula and the Table D.5 transcription in
/// `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`) and the
/// §C.1.5.2.7 verbatim `MNR = SNR − SMR` definition transcribed from
/// ISO/IEC 11172-3:1993 Annex C §C.1.5.2.7 "Bit allocation" printed
/// p.73 in `docs/audio/mp3/ISO_IEC_11172-3-MP3-1993.pdf` are consulted.
/// The Table C.5 `SNR_n` term is caller-injected (the table is behind
/// the numeric-table transcription gap); no external implementation
/// was read.
#[must_use]
pub fn coder_partition_d5_mnr_row_order<S, L, F>(
    snr_per_partition: S,
    lsb_per_partition: L,
    ltg_per_line: F,
) -> [CoderPartitionD5Mnr; 32]
where
    S: Fn(u16) -> f64,
    L: Fn(u16) -> f64,
    F: Fn(u16) -> f64,
{
    let smr = coder_partition_d5_smr_row_order(lsb_per_partition, ltg_per_line);
    let mut out = [CoderPartitionD5Mnr {
        mnr_db: 0.0,
        smr_db: 0.0,
        width_n: 0,
    }; 32];
    for (i, row) in smr.iter().enumerate() {
        let n = (i + 1) as u16;
        out[i] = CoderPartitionD5Mnr {
            mnr_db: snr_per_partition(n) - row.smr_db,
            smr_db: row.smr_db,
            width_n: row.width_n,
        };
    }
    out
}

/// The §C.1.5.2.7 "subband with the minimal MNR" — the single coder
/// partition the bit-allocation loop selects at the head of each
/// iteration to receive the next-higher quantization-accuracy entry.
///
/// Produced by [`coder_partition_d5_min_mnr`] (Phase 2 step 73 / r272).
/// The per-iteration successor of Phase 2 step 72's
/// [`CoderPartitionD5Mnr`] vector: where step 72 computes the row-order
/// `MNR_n = SNR_n − SMR_n` column, this step performs the loop's very
/// first iteration action — "Determination of the minimal MNR of all
/// subbands" (verbatim, printed p.71) — reducing the 32-row vector to
/// the one partition "that has the greatest benefit".
///
/// **Field semantics.** `partition_n` is the spec's 1-based partition
/// index `n ∈ 1..=32` of the selected subband (the array index plus
/// one); `mnr_db`, `smr_db`, and `width_n` are that partition's
/// [`CoderPartitionD5Mnr`] columns carried through verbatim.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoderPartitionD5MinMnr {
    /// 1-based partition index `n ∈ 1..=32` of the subband holding the
    /// minimal `MNR_n` — the "subband with the minimal MNR" the
    /// §C.1.5.2.7 loop increases the quantization accuracy of next.
    /// Maps to array index `partition_n - 1` on the step-72 vector.
    pub partition_n: u16,
    /// The selected partition's mask-to-noise ratio `MNR_n` (dB) — the
    /// minimum over all 32 subbands. Carried verbatim from the step-72
    /// [`CoderPartitionD5Mnr::mnr_db`] cell.
    pub mnr_db: f64,
    /// The selected partition's §D.1 Step 9 signal-to-mask ratio
    /// `SMR_n` (dB), carried verbatim from
    /// [`CoderPartitionD5Mnr::smr_db`].
    pub smr_db: f64,
    /// The selected partition's `width_n` Table D.5 column flag,
    /// carried verbatim from [`CoderPartitionD5Mnr::width_n`].
    pub width_n: u16,
}

/// §C.1.5.2.7 "Determination of the minimal MNR of all subbands" — the
/// argmin selection at the head of every Layer I / Layer II
/// bit-allocation iteration.
///
/// Given the Phase 2 step 72 row-order MNR vector
/// (`[CoderPartitionD5Mnr; 32]`), returns the single
/// [`CoderPartitionD5MinMnr`] for the subband with the **smallest**
/// `mnr_db` — the partition "that has the greatest benefit", which the
/// loop then promotes to the next-higher quantization-accuracy entry.
/// This is the first iteration action of the iterative allocation
/// procedure, performed once per loop pass after the step-72
/// `MNR = SNR − SMR` initialisation.
///
/// **Selection rule.** A row-order scan over `n ∈ 1..=32` keeping the
/// running minimum; the first (lowest-index) partition achieving the
/// minimum wins. The spec selects "the subband with the minimal MNR"
/// (singular), so a deterministic tie-break is required — equal-MNR
/// subbands resolve to the lowest partition index, the order in which
/// the §C.1.5.2.7 loop walks Table D.5.
///
/// **Index convention.** `partition_n` is 1-based (`n ∈ 1..=32`),
/// matching the spec; it equals `i + 1` for the winning array index
/// `i ∈ 0..=31` of the step-72 vector.
///
/// **Column pass-through.** `mnr_db`, `smr_db`, and `width_n` are
/// copied verbatim from the selected [`CoderPartitionD5Mnr`] row; no
/// arithmetic is performed beyond the `<` comparisons of the scan.
///
/// **NaN handling.** A `NaN` `mnr_db` never compares `<` the running
/// minimum, so it is skipped; an all-`NaN` vector therefore retains the
/// `n = 1` seed. (Step 72's `MNR = SNR − SMR` produces `NaN` only when a
/// caller injects a `NaN` `SNR_n`/`SMR_n`; well-formed inputs are
/// finite.)
///
/// **Determinism.** A pure function of the input vector: the same
/// `[CoderPartitionD5Mnr; 32]` always yields the same selection.
///
/// Provenance: only the §C.1.5.2.7 "Determination of the minimal MNR of
/// all subbands" loop step transcribed from ISO/IEC 11172-3:1993 Annex
/// C §C.1.5.2.7 "Bit allocation" (printed p.71) in
/// `docs/audio/mp3/ISO_IEC_11172-3-MP3-1993.pdf` and the Phase 2 step 72
/// [`coder_partition_d5_mnr_row_order`] vector it consumes are read; no
/// external implementation was consulted.
#[must_use]
pub fn coder_partition_d5_min_mnr(mnr: &[CoderPartitionD5Mnr; 32]) -> CoderPartitionD5MinMnr {
    let mut best = 0usize;
    for (i, row) in mnr.iter().enumerate().skip(1) {
        if row.mnr_db < mnr[best].mnr_db {
            best = i;
        }
    }
    CoderPartitionD5MinMnr {
        partition_n: (best + 1) as u16,
        mnr_db: mnr[best].mnr_db,
        smr_db: mnr[best].smr_db,
        width_n: mnr[best].width_n,
    }
}

/// The outcome of the §C.1.5.2.7 "increase the accuracy of the
/// quantization of the subband with the minimal MNR" loop action — the
/// per-subband allocation-entry advance produced by
/// [`bit_allocation_promote_entry`] (Phase 2 step 74 / r273).
///
/// The §C.1.5.2.7 iteration, after step 73 selects the subband "that has
/// the greatest benefit" (the minimal-MNR partition), increases that
/// subband's quantization accuracy "by using the next higher entry in
/// the relevant table B.2, *Layer II Possible Quantization per
/// subband*". This struct reports the post-advance allocation-table
/// entry index for the selected subband and whether the advance was
/// actually applied.
///
/// **Field semantics.** `subband` echoes the 0-based subband index the
/// advance targeted (the array index of the selected partition);
/// `entry` is the resulting Table B.2 column entry index after the
/// promotion (`prev_entry + 1` on a successful advance, otherwise
/// `prev_entry` unchanged); `advanced` is `true` iff a next-higher entry
/// existed and was selected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BitAllocPromotion {
    /// 0-based subband index whose Table B.2 entry the loop tried to
    /// advance — the array slot of the §C.1.5.2.7 minimal-MNR subband.
    pub subband: u16,
    /// The subband's resulting Table B.2 *Layer II Possible Quantization
    /// per subband* column entry index after this loop action: one
    /// higher than the prior entry on a successful advance, or the prior
    /// entry unchanged when the subband was already at its top entry.
    pub entry: u16,
    /// `true` iff a next-higher entry was available and selected; `false`
    /// when the subband already held the highest entry of its B.2 column
    /// (`entry_count - 1`), so no finer quantization could be assigned.
    pub advanced: bool,
}

/// §C.1.5.2.7 "The accuracy of the quantization of the subband with the
/// minimal MNR is increased by using the next higher entry in the
/// relevant table B.2" — the second action of every Layer I / Layer II
/// bit-allocation iteration.
///
/// Phase 2 step 73 ([`coder_partition_d5_min_mnr`]) selected the subband
/// "that has the greatest benefit" — the minimal-MNR partition. This
/// step performs the loop's next verbatim action: it advances that
/// subband's quantization accuracy to the **next-higher entry** of its
/// Table B.2 *Layer II Possible Quantization per subband* column, so the
/// following loop step can recompute the subband's MNR at the finer
/// quantization.
///
/// **Parameters.** `subband` is the 0-based index of the selected
/// subband (the array slot of the step-73 minimal-MNR partition, i.e.
/// `CoderPartitionD5MinMnr::partition_n - 1`); `prev_entry` is that
/// subband's current Table B.2 column entry index before this action;
/// `entry_count` is the number of entries in that subband's B.2 column
/// (the count of possible quantizations the column permits). The B.2
/// column lengths are caller-supplied: Table B.2 lives behind the same
/// numeric-table transcription gap as Tables C.5 / D.1 / D.2, so the
/// per-subband entry count is injected, the dependency-injection pattern
/// the surrounding Phase 2 steps use.
///
/// **Advance rule.** When `prev_entry + 1 < entry_count` a next-higher
/// entry exists, so `entry = prev_entry + 1` and `advanced = true`. When
/// `prev_entry` is already the top entry (`prev_entry + 1 >= entry_count`)
/// no finer quantization can be assigned, so `entry = prev_entry`
/// unchanged and `advanced = false` — the loop must then leave this
/// subband and reselect (a later §C.1.5.2.7 step). An `entry_count` of
/// zero (a subband with no possible quantization) likewise yields no
/// advance with `entry = prev_entry`.
///
/// **No spec arithmetic beyond the increment.** The only computation is
/// the `prev_entry + 1` index step and the bound comparison against
/// `entry_count`; the B.2 entry *values* (step counts / bit costs) are
/// not consulted here — only the column's entry index is advanced.
///
/// **Determinism.** A pure function of its three arguments.
///
/// Provenance: only the §C.1.5.2.7 "The accuracy of the quantization of
/// the subband with the minimal MNR is increased by using the next
/// higher entry in the relevant table B.2" loop step transcribed from
/// ISO/IEC 11172-3:1993 Annex C §C.1.5.2.7 "Bit allocation" (printed
/// p.71) in `docs/audio/mp3/ISO_IEC_11172-3-MP3-1993.pdf`, and the
/// Phase 2 step 73 [`coder_partition_d5_min_mnr`] selection it consumes,
/// are read. The Table B.2 column length is caller-injected (the table
/// is behind the numeric-table transcription gap); no external
/// implementation was consulted.
#[must_use]
pub fn bit_allocation_promote_entry(
    subband: u16,
    prev_entry: u16,
    entry_count: u16,
) -> BitAllocPromotion {
    let next = prev_entry.saturating_add(1);
    if next < entry_count {
        BitAllocPromotion {
            subband,
            entry: next,
            advanced: true,
        }
    } else {
        BitAllocPromotion {
            subband,
            entry: prev_entry,
            advanced: false,
        }
    }
}

/// The outcome of the §C.1.5.2.7 "The new MNR of this subband is
/// calculated" loop action — the post-promotion mask-to-noise ratio of
/// the subband whose quantization accuracy was just increased, produced
/// by [`bit_allocation_recompute_mnr`] (Phase 2 step 75 / r274).
///
/// The §C.1.5.2.7 iteration, after step 73 selects the minimal-MNR
/// subband and step 74 advances its Table B.2 entry to the next-higher
/// quantization accuracy, recomputes that subband's MNR with the verbatim
/// definition `MNR = SNR − SMR`. The finer quantization carries a larger
/// `SNR_n` (its Table C.5 *Layer II Signal-to-Noise Ratios* value for the
/// advanced entry), while the `SMR_n` — the psychoacoustic-model output —
/// is unchanged; the new `MNR_n` is therefore larger than the prior one,
/// removing this subband from the "greatest benefit" position so a later
/// iteration can reselect.
///
/// **Field semantics.** `subband` echoes the 0-based subband index whose
/// MNR was recomputed; `entry` is the Table B.2 column entry index the
/// recomputed `SNR_n` corresponds to (the post-promotion entry from
/// step 74); `mnr_db` is the recomputed `MNR_n = SNR_n − SMR_n` (dB);
/// `smr_db` is the carried-through psychoacoustic-model `SMR_n` (dB).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoderPartitionD5RecomputedMnr {
    /// 0-based subband index whose `MNR_n` was recomputed after the
    /// step-74 next-higher-entry promotion — the array slot of the
    /// §C.1.5.2.7 minimal-MNR subband.
    pub subband: u16,
    /// The Table B.2 *Layer II Possible Quantization per subband* column
    /// entry index the recomputed `SNR_n` corresponds to: the post-
    /// promotion entry reported by step 74's [`BitAllocPromotion::entry`].
    pub entry: u16,
    /// The recomputed mask-to-noise ratio `MNR_n = SNR_n − SMR_n` (dB)
    /// after the promotion, where `SNR_n` is the Table C.5 value for the
    /// advanced entry and `SMR_n` is the unchanged psychoacoustic-model
    /// output. Larger than the pre-promotion `MNR_n` for a monotone
    /// Table C.5 column (finer quantization ⇒ higher `SNR_n`). No
    /// clipping.
    pub mnr_db: f64,
    /// The §D.1 Step 9 signal-to-mask ratio `SMR_n` (dB) carried verbatim
    /// from the selected subband — the psychoacoustic-model output the
    /// loop re-uses unchanged across iterations.
    pub smr_db: f64,
}

/// §C.1.5.2.7 "The new MNR of this subband is calculated" — the third
/// action of every Layer I / Layer II bit-allocation iteration.
///
/// Phase 2 step 73 ([`coder_partition_d5_min_mnr`]) selected the subband
/// "that has the greatest benefit"; Phase 2 step 74
/// ([`bit_allocation_promote_entry`]) advanced that subband's Table B.2
/// entry to the next-higher quantization accuracy. This step performs the
/// loop's next verbatim action: it recomputes the subband's mask-to-noise
/// ratio at the finer quantization using the §C.1.5.2.7 definition
/// `MNR = SNR − SMR`. The `SNR_n` is the Table C.5 *Layer II
/// Signal-to-Noise Ratios* value for the **advanced** entry; the `SMR_n`
/// is unchanged from the selected subband (the psychoacoustic-model
/// output the loop re-reads each iteration). The result removes the
/// subband from its minimal-MNR position so a subsequent iteration's
/// step-73 selection can move on.
///
/// **Parameters.** `promotion` is the step-74
/// [`BitAllocPromotion`] for the selected subband (supplying its 0-based
/// `subband` index and the post-promotion `entry`); `smr_db` is that
/// subband's §D.1 Step 9 `SMR_n` (dB), carried verbatim from the step-73
/// [`CoderPartitionD5MinMnr::smr_db`] selection (the loop re-uses the
/// psychoacoustic-model output unchanged); `snr_for_entry` returns the
/// Table C.5 `SNR_n` (dB) for the promotion's post-advance entry. Table
/// C.5 lives behind the same numeric-table transcription gap as Tables
/// B.2 / D.1 / D.2, so the `SNR_n` value is injected — the dependency-
/// injection pattern the surrounding Phase 2 steps use.
///
/// **Recompute rule.** `mnr_db = snr_for_entry(promotion.entry) − smr_db`,
/// the single verbatim §C.1.5.2.7 subtraction. The `entry` carried in the
/// result is `promotion.entry` (so a saturated step-74 promotion that did
/// not advance recomputes the MNR at the held entry — an idempotent re-
/// evaluation, since the `SNR_n` for the unchanged entry is the same).
/// No spec arithmetic is introduced beyond the `SNR − SMR` subtraction.
///
/// **Determinism.** A pure function of its arguments and the injected
/// `SNR_n` callback.
///
/// Provenance: only the §C.1.5.2.7 "The new MNR of this subband is
/// calculated" loop step transcribed from ISO/IEC 11172-3:1993 Annex C
/// §C.1.5.2.7 "Bit allocation" (printed p.71) in
/// `docs/audio/mp3/ISO_IEC_11172-3-MP3-1993.pdf`, the verbatim
/// `MNR = SNR − SMR` definition from the same clause, and the Phase 2
/// step 74 [`bit_allocation_promote_entry`] result it consumes are read.
/// The Table C.5 `SNR_n` term is caller-injected (the table is behind the
/// numeric-table transcription gap); no external implementation was
/// consulted.
#[must_use]
pub fn bit_allocation_recompute_mnr<S>(
    promotion: BitAllocPromotion,
    smr_db: f64,
    snr_for_entry: S,
) -> CoderPartitionD5RecomputedMnr
where
    S: Fn(u16) -> f64,
{
    CoderPartitionD5RecomputedMnr {
        subband: promotion.subband,
        entry: promotion.entry,
        mnr_db: snr_for_entry(promotion.entry) - smr_db,
        smr_db,
    }
}

/// The running bit-budget accumulators of the §C.1.5.2.7 Layer II
/// bit-allocation loop — the `bspl` / `bsel` / `bscf` totals and the
/// derived available-data-bits `adb` after the §C.1.5.2.7 step-4 update
/// produced by [`bit_allocation_budget_update`] (Phase 2 step 76 / r275).
///
/// The §C.1.5.2.7 iteration's fourth action reads, verbatim: "bspl is
/// updated according to the additional number of bits required. If a
/// non-zero number of bits is assigned to a subband for the first time,
/// bsel has to be updated, and bscf has to be updated according to the
/// number of scalefactors required for this subband. Then adb is
/// calculated again using the formula:
/// `adb = cb - (bhdr + bcrc + bbal + bsel + bscf + bspl + banc)`."
/// This struct carries the three loop-mutated accumulators plus the
/// recomputed `adb` after one such update.
///
/// **Field semantics.** `bspl` is the running total of bits assigned to
/// the subband **samples** after adding this iteration's additional
/// sample bits; `bsel` is the running total of bits for the
/// scalefactor-selection information (scfsi); `bscf` is the running total
/// of bits for the transmitted **scalefactors**; `first_time` is `true`
/// iff this iteration assigned a non-zero number of bits to the selected
/// subband for the first time (the condition under which `bsel` and
/// `bscf` are grown); `adb` is the recomputed available-data-bits left
/// for samples and scalefactors after the §C.1.5.2.7 `adb` formula.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BitAllocBudget {
    /// Running total of bits assigned to the subband **samples** (`bspl`)
    /// after this iteration's additional sample bits were added.
    pub bspl: u32,
    /// Running total of bits for the scalefactor-**selection** information
    /// (`bsel`, the scfsi field) after this iteration's update. Grown only
    /// when a non-zero number of bits is assigned to the subband for the
    /// first time.
    pub bsel: u32,
    /// Running total of bits for the transmitted **scalefactors**
    /// (`bscf`) after this iteration's update. Grown only when a non-zero
    /// number of bits is assigned to the subband for the first time.
    pub bscf: u32,
    /// `true` iff this iteration assigned a non-zero number of bits to the
    /// selected subband for the **first time** — the §C.1.5.2.7 condition
    /// under which `bsel` and `bscf` are updated.
    pub first_time: bool,
    /// Recomputed available-data-bits `adb = cb - (bhdr + bcrc + bbal +
    /// bsel + bscf + bspl + banc)` after this iteration's accumulator
    /// update. Saturates at zero (never negative) — the §C.1.5.2.7
    /// termination test compares it against the next possible increase.
    pub adb: u32,
}

/// The fixed per-frame overhead bit counts of the §C.1.5.2.7 `adb`
/// formula — the terms subtracted from the total available bits `cb`
/// that do **not** change across the bit-allocation loop's iterations.
///
/// These are the `bhdr` (header), `bcrc` (CRC checkword), `bbal` (bit
/// allocation field), `banc` (ancillary data) terms of
/// `adb = cb - (bhdr + bcrc + bbal + bsel + bscf + bspl + banc)`. They
/// are caller-supplied because their values depend on the frame's
/// configuration (header is 32 bits, CRC is 16 bits only when used, the
/// bit-allocation field width follows Table B.2 per the chosen layout,
/// and ancillary data is application-defined) rather than on any single
/// numeric table behind the transcription gap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BitAllocOverhead {
    /// Total available bits for the frame (`cb`).
    pub cb: u32,
    /// Header bits (`bhdr`, 32 for a Layer II frame).
    pub bhdr: u32,
    /// CRC-checkword bits (`bcrc`, 16 when CRC protection is used, else
    /// 0).
    pub bcrc: u32,
    /// Bit-allocation-field bits (`bbal`).
    pub bbal: u32,
    /// Ancillary-data bits (`banc`).
    pub banc: u32,
}

/// §C.1.5.2.7 step 4 — "bspl is updated according to the additional
/// number of bits required … Then adb is calculated again" — the fourth
/// and final action of every Layer II bit-allocation iteration.
///
/// Phase 2 step 73 selected the minimal-MNR subband, step 74
/// ([`bit_allocation_promote_entry`]) advanced its Table B.2 entry, and
/// step 75 ([`bit_allocation_recompute_mnr`]) recomputed its MNR. This
/// step closes the iteration by folding the promotion's bit cost into the
/// running budget and recomputing the available-data-bits `adb`.
///
/// **Parameters.** `prev` is the running budget before this iteration
/// (`bspl` / `bsel` / `bscf` accumulators; its `first_time` / `adb`
/// fields are ignored on input); `extra_sample_bits` is the additional
/// number of bits the promotion requires for this subband's samples (the
/// difference between the Table B.4 sample-bit cost at the new and old
/// Table B.2 entries — caller-supplied because Table B.4 is behind the
/// numeric-table transcription gap); `first_time` is `true` iff this
/// promotion assigned a non-zero number of bits to the subband for the
/// **first time** (i.e. the subband moved off its zero-bit entry);
/// `sel_bits` and `scf_bits` are the bits this subband then contributes
/// to `bsel` (scfsi) and `bscf` (scalefactors) respectively — added only
/// on a `first_time` promotion; `overhead` carries the fixed `cb` / `bhdr`
/// / `bcrc` / `bbal` / `banc` terms of the `adb` formula.
///
/// **Update rule (verbatim §C.1.5.2.7).**
/// `bspl += extra_sample_bits`; and when `first_time` is set,
/// `bsel += sel_bits` and `bscf += scf_bits`. Then
/// `adb = cb - (bhdr + bcrc + bbal + bsel + bscf + bspl + banc)`,
/// saturating at zero. When `first_time` is `false` the `bsel` / `bscf`
/// totals are carried through unchanged (the subband already held a
/// non-zero allocation, so its scalefactor / selection bits are already
/// in the running totals).
///
/// **No spec arithmetic beyond the additions and the formula.** The
/// per-entry sample-bit and scalefactor-bit costs are caller-injected
/// (Tables B.2 / B.4 are behind the numeric-table transcription gap); the
/// only computation here is the accumulator additions and the verbatim
/// `adb` subtraction.
///
/// **Determinism.** A pure function of its arguments.
///
/// Provenance: only the §C.1.5.2.7 "bspl is updated … Then adb is
/// calculated again" loop step and its `adb = cb - (bhdr + bcrc + bbal +
/// bsel + bscf + bspl + banc)` formula transcribed from
/// ISO/IEC 11172-3:1993 Annex C §C.1.5.2.7 "Bit allocation" (printed
/// p.74) in `docs/audio/mp3/ISO_IEC_11172-3-MP3-1993.pdf`, and the
/// Phase 2 step 74 [`bit_allocation_promote_entry`] result it follows,
/// are read. The Table B.2 / B.4 per-entry bit costs are caller-injected
/// (behind the numeric-table transcription gap); no external
/// implementation was consulted.
#[must_use]
pub fn bit_allocation_budget_update(
    prev: BitAllocBudget,
    extra_sample_bits: u32,
    first_time: bool,
    sel_bits: u32,
    scf_bits: u32,
    overhead: BitAllocOverhead,
) -> BitAllocBudget {
    let bspl = prev.bspl.saturating_add(extra_sample_bits);
    let (bsel, bscf) = if first_time {
        (
            prev.bsel.saturating_add(sel_bits),
            prev.bscf.saturating_add(scf_bits),
        )
    } else {
        (prev.bsel, prev.bscf)
    };
    let used = overhead
        .bhdr
        .saturating_add(overhead.bcrc)
        .saturating_add(overhead.bbal)
        .saturating_add(bsel)
        .saturating_add(bscf)
        .saturating_add(bspl)
        .saturating_add(overhead.banc);
    let adb = overhead.cb.saturating_sub(used);
    BitAllocBudget {
        bspl,
        bsel,
        bscf,
        first_time,
        adb,
    }
}

/// §C.1.5.2.7 loop-continuation test — "The iterative procedure is
/// repeated as long as adb is not less than any possible increase of
/// bspl, bsel and bscf within one loop." (Phase 2 step 76 / r275.)
///
/// After the step-4 [`bit_allocation_budget_update`] recomputes `adb`,
/// the loop re-runs (step 73 reselects the minimal-MNR subband, step 74
/// promotes it, …) only while the remaining `adb` can still pay for the
/// **largest possible** single-iteration increase of the three loop
/// accumulators. `max_possible_increase` is that worst-case one-loop bit
/// cost — the maximum over all subbands of the additional `bspl` bits a
/// next-entry promotion would cost, plus the `bsel` + `bscf` bits a
/// first-time allocation of that subband would add (caller-supplied; the
/// per-entry costs live behind the Tables B.2 / B.4 transcription gap).
///
/// Returns `true` iff `adb >= max_possible_increase` — i.e. the loop
/// should iterate again. When the largest possible increase no longer
/// fits the remaining `adb`, the iteration terminates (`false`). A
/// `max_possible_increase` of zero (no subband can be promoted further —
/// every subband is at its top Table B.2 entry) returns `true` only while
/// `adb >= 0`, which `u32` always satisfies; callers detect the
/// no-promotable-subband terminal condition from step 74's
/// [`BitAllocPromotion::advanced`] flag, not from this predicate.
///
/// **Determinism.** A pure comparison of its two arguments.
///
/// Provenance: only the §C.1.5.2.7 "The iterative procedure is repeated
/// as long as adb is not less than any possible increase of bspl, bsel
/// and bscf within one loop" termination sentence transcribed from
/// ISO/IEC 11172-3:1993 Annex C §C.1.5.2.7 "Bit allocation" (printed
/// p.74) in `docs/audio/mp3/ISO_IEC_11172-3-MP3-1993.pdf` is read. The
/// worst-case one-loop increase is caller-supplied (its per-entry bit
/// costs are behind the numeric-table transcription gap); no external
/// implementation was consulted.
#[must_use]
pub fn bit_allocation_should_iterate(adb: u32, max_possible_increase: u32) -> bool {
    adb >= max_possible_increase
}

/// §D.1 Step 1 FFT transform length for **Layer I** — 512 samples.
///
/// Annex D Model 1 Step 1 "FFT Analysis" (printed p.110) technical
/// data: "transform length — Layer I: 512 samples". Frequency
/// resolution is `sampling_frequency / 512`.
pub const MODEL1_FFT_LEN_LAYER1: usize = 512;

/// §D.1 Step 1 FFT transform length for **Layer II** — 1 024 samples.
///
/// Annex D Model 1 Step 1 "FFT Analysis" (printed p.110) technical
/// data: "transform length — Layer II: 1 024 samples". Frequency
/// resolution is `sampling_frequency / 1024`. The D.1 preamble notes
/// "the model can be adapted to Layer III"; the Layer III adaptation
/// keeps this 1 024-sample length (its half-spectrum lines `k ∈
/// 0..=512` are exactly the 1-based ω ∈ 1..=513 lines that the Table
/// D.5 coder-partition accessors above consume).
pub const MODEL1_FFT_LEN_LAYER2: usize = 1024;

/// §D.1 Step 1 sound-pressure-level reference — 96 dB.
///
/// Verbatim (printed p.110): "A normalization to the reference level
/// of 96 dB SPL (Sound Pressure Level) has to be done in such a way
/// that the maximum value corresponds to 96 dB."
pub const MODEL1_SPL_REFERENCE_DB: f64 = 96.0;

/// §D.1 Step 1 Hann window coefficient `h(i)` (Phase 2 step 77 /
/// r276).
///
/// Verbatim formula (printed p.110):
///
/// ```text
/// h(i) = sqrt(8/3) * 0,5 * {1 - cos[2 * π * (i)/N]}      0 <= i <= N-1
/// ```
///
/// Returns `None` for `n == 0` or any `i` outside the spec's
/// `0 <= i <= N-1` domain; no clamping or periodic extension is
/// invented. The `sqrt(8/3)` prefactor makes the window
/// **unit-power**: `Σ h(i)² = N` exactly in exact arithmetic (the
/// `0,5²·(1-cos)²` expansion averages to `3/8` over a full period and
/// `8/3 · 3/8 = 1`), so windowing does not bias the power-density
/// estimate of [`model1_power_density_spectrum`].
///
/// **Determinism.** Pure function of `(i, n)`.
///
/// Provenance: only the Step 1 "Hann window, h(i)" formula line
/// transcribed above from ISO/IEC 11172-3:1993 Annex D §D.1 Step 1
/// "FFT Analysis" (printed p.110) in
/// `docs/audio/mp3/ISO_IEC_11172-3-MP3-1993.pdf` is read; no external
/// implementation was consulted.
#[must_use]
pub fn model1_hann_window(i: usize, n: usize) -> Option<f64> {
    if n == 0 || i >= n {
        return None;
    }
    let sqrt_8_3 = (8.0_f64 / 3.0).sqrt();
    let angle = 2.0 * core::f64::consts::PI * (i as f64) / (n as f64);
    Some(sqrt_8_3 * 0.5 * (1.0 - angle.cos()))
}

/// In-place iterative radix-2 decimation-in-time FFT over split
/// real/imaginary slices. Private helper for
/// [`model1_power_density_spectrum`]; `re.len()` must be a power of
/// two and equal `im.len()` (callers guarantee it — the public entry
/// point only accepts the two spec transform lengths 512 / 1 024).
/// Standard-mathematics Cooley-Tukey butterflies; nothing here is
/// codec-specific.
fn fft_in_place(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    debug_assert!(n.is_power_of_two());
    debug_assert_eq!(n, im.len());
    // Bit-reversal permutation.
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    // Butterfly passes.
    let mut len = 2usize;
    while len <= n {
        let half = len / 2;
        let ang = -2.0 * core::f64::consts::PI / (len as f64);
        let (step_re, step_im) = (ang.cos(), ang.sin());
        let mut base = 0usize;
        while base < n {
            let (mut w_re, mut w_im) = (1.0_f64, 0.0_f64);
            for k in 0..half {
                let (u_re, u_im) = (re[base + k], im[base + k]);
                let (t_re, t_im) = (re[base + k + half], im[base + k + half]);
                let (v_re, v_im) = (t_re * w_re - t_im * w_im, t_re * w_im + t_im * w_re);
                re[base + k] = u_re + v_re;
                im[base + k] = u_im + v_im;
                re[base + k + half] = u_re - v_re;
                im[base + k + half] = u_im - v_im;
                let next_re = w_re * step_re - w_im * step_im;
                w_im = w_re * step_im + w_im * step_re;
                w_re = next_re;
            }
            base += len;
        }
        len <<= 1;
    }
}

/// §D.1 Step 1 power-density spectrum `X(k)` of one analysis block
/// (Phase 2 step 77 / r276).
///
/// Verbatim formula (printed p.110):
///
/// ```text
/// X(k) = 10 * log10 | (1/N) Σ_{l=0}^{N-1} h(l) * s(l) * e^(-j*k*l*2*π/N) |²  dB
///                                                            k = 0...N/2,
/// ```
///
/// where `s(l)` is the input signal and `h(l)` the
/// [`model1_hann_window`] coefficient. The masking threshold "is
/// derived from an estimate of the power density spectrum that is
/// calculated by a 512-point FFT for Layer I, or by a 1 024-point FFT
/// for Layer II" (Layer III adapts the 1 024-point variant); any
/// other input length returns `None` verbatim — no padding or
/// truncation is invented. The output carries the spec's `k = 0...N/2`
/// inclusive half-spectrum: `N/2 + 1` lines (513 for the 1 024-sample
/// block — matching the 1-based ω ∈ 1..=513 Table D.5 convention via
/// `k = ω - 1`).
///
/// An all-zero (silent) block yields `10·log10(0) = -∞` dB lines;
/// `f64::NEG_INFINITY` is returned unmodified (the spec expresses the
/// spectrum in dB with no floor; [`model1_normalize_to_96db_spl`]
/// refuses to normalize a spectrum with no finite maximum).
///
/// The Step 1 PCM **window-placement** rules (the 256-sample analysis
/// subband-filter delay compensation and the ±64-sample Hann/frame
/// alignment shifts) are caller responsibilities: this primitive
/// transforms exactly the block it is handed.
///
/// **Determinism.** Pure function of the input block; the in-place
/// radix-2 FFT introduces only standard floating-point rounding
/// (cross-checked against a direct DFT evaluation in the tests).
///
/// Provenance: only the Step 1 "power density spectrum X(k)" formula,
/// the "windowed by a Hann window" sentence, the transform-length
/// technical-data lines, and the `k = 0...N/2` index range transcribed
/// from ISO/IEC 11172-3:1993 Annex D §D.1 Step 1 "FFT Analysis"
/// (printed p.110) in `docs/audio/mp3/ISO_IEC_11172-3-MP3-1993.pdf`
/// are read; no external implementation was consulted.
#[must_use]
pub fn model1_power_density_spectrum(s: &[f64]) -> Option<Vec<f64>> {
    let n = s.len();
    if n != MODEL1_FFT_LEN_LAYER1 && n != MODEL1_FFT_LEN_LAYER2 {
        return None;
    }
    let mut re: Vec<f64> = s
        .iter()
        .enumerate()
        .map(|(l, &sample)| {
            // `l < n` always holds here, so the window accessor cannot
            // return `None`.
            model1_hann_window(l, n).unwrap_or(0.0) * sample
        })
        .collect();
    let mut im = vec![0.0_f64; n];
    fft_in_place(&mut re, &mut im);
    let inv_n = 1.0 / (n as f64);
    Some(
        (0..=n / 2)
            .map(|k| {
                let r = re[k] * inv_n;
                let i = im[k] * inv_n;
                10.0 * (r * r + i * i).log10()
            })
            .collect(),
    )
}

/// §D.1 Step 1 normalization of a dB spectrum to the 96 dB SPL
/// reference (Phase 2 step 77 / r276).
///
/// Verbatim (printed p.110): "A normalization to the reference level
/// of 96 dB SPL (Sound Pressure Level) has to be done in such a way
/// that the maximum value corresponds to 96 dB."
///
/// Adds the constant offset `96 − max(x)` dB to every line in place
/// and returns the applied offset. The maximum is taken over all
/// supplied lines; relative line-to-line differences are preserved
/// exactly (a single shared addend). Returns `None` — leaving the
/// slice untouched — when no finite maximum exists (empty slice, or
/// an all-`-∞` silent-block spectrum), since no finite offset can
/// place a non-finite maximum at 96 dB.
///
/// **Determinism.** Pure in-place affine shift.
///
/// Provenance: only the Step 1 normalization sentence quoted above
/// from ISO/IEC 11172-3:1993 Annex D §D.1 Step 1 "FFT Analysis"
/// (printed p.110) in `docs/audio/mp3/ISO_IEC_11172-3-MP3-1993.pdf`
/// is read; no external implementation was consulted.
#[must_use]
pub fn model1_normalize_to_96db_spl(x: &mut [f64]) -> Option<f64> {
    let max = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        return None;
    }
    let offset = MODEL1_SPL_REFERENCE_DB - max;
    for line in x.iter_mut() {
        *line += offset;
    }
    Some(offset)
}

/// §D.1 Step 2 full-scale factor — 32 768.
///
/// The verbatim `Lsb(n)` formula's scalefactor term multiplies
/// `scf_max(n)` by `32 768` before taking `20·log` (printed p.110).
pub const MODEL1_STEP2_FULL_SCALE: f64 = 32768.0;

/// §D.1 Step 2 peak-to-RMS correction — 10 dB.
///
/// Verbatim (printed p.110): "The '-10 dB' term corrects for the
/// difference between peak and RMS level."
pub const MODEL1_STEP2_PEAK_RMS_CORRECTION_DB: f64 = 10.0;

/// §D.1 Step 2 scalefactor SPL term `20·log(scf_max(n)·32 768) − 10`
/// dB (Phase 2 step 78 / r276).
///
/// The second argument of the verbatim Step 2 maximum (printed
/// p.110):
///
/// ```text
/// Lsb(n) = MAX[ X(k), 20*log(scf_max(n)*32 768)-10 ]  dB
///               X(k) in subband n
/// ```
///
/// "The expression scf_max(n) is in Layer I the scalefactor, and in
/// Layer II the maximum of the three scalefactors of subband n within
/// a frame." The caller supplies that per-subband maximum scalefactor
/// value; this primitive evaluates only the formula term. The log is
/// the dB-convention base-10 logarithm. No domain clamping is
/// invented: a non-positive `scf_max` propagates the IEEE `log10`
/// result (`-∞` at zero, NaN below) — spec scalefactors are positive.
///
/// **Determinism.** Pure function of `scf_max`.
///
/// Provenance: only the Step 2 `Lsb(n)` formula and the scf_max /
/// "-10 dB" explanatory sentences transcribed from ISO/IEC
/// 11172-3:1993 Annex D §D.1 Step 2 "Determination of the sound
/// pressure level" (printed p.110) in
/// `docs/audio/mp3/ISO_IEC_11172-3-MP3-1993.pdf` are read; no
/// external implementation was consulted.
#[must_use]
pub fn model1_step2_scf_term_db(scf_max: f64) -> f64 {
    20.0 * (scf_max * MODEL1_STEP2_FULL_SCALE).log10() - MODEL1_STEP2_PEAK_RMS_CORRECTION_DB
}

/// §D.1 Step 2 sound pressure level `Lsb(n)` (Phase 2 step 78 /
/// r276).
///
/// The verbatim outer maximum (printed p.110):
///
/// ```text
/// Lsb(n) = MAX[ X(k), 20*log(scf_max(n)*32 768)-10 ]  dB
/// ```
///
/// `x_subband_db` is the caller-determined spectral argument — either
/// "the sound pressure level of the spectral line with index k of the
/// FFT with the maximum amplitude in the frequency range corresponding
/// to subband n" ([`model1_step2_subband_max_line_db`]) or, for the
/// spec's alternative method ("offers a potential for better encoder
/// performance, but this technique has not been subjected to a formal
/// audio quality test", printed p.110–111), the alternative
/// sound-pressure level `Xspl(n)` ([`model1_step2_subband_xspl_db`]).
/// Both methods share this identical outer MAX with the
/// [`model1_step2_scf_term_db`] scalefactor term.
///
/// **Determinism.** Pure two-argument maximum.
///
/// Provenance: only the two Step 2 `Lsb(n)` formula lines (primary
/// and alternative method) transcribed from ISO/IEC 11172-3:1993
/// Annex D §D.1 Step 2 (printed p.110–111) in
/// `docs/audio/mp3/ISO_IEC_11172-3-MP3-1993.pdf` are read; no
/// external implementation was consulted.
#[must_use]
pub fn model1_step2_lsb_db(x_subband_db: f64, scf_max: f64) -> f64 {
    x_subband_db.max(model1_step2_scf_term_db(scf_max))
}

/// §D.1 Step 2 alternative sound pressure level `Xspl` over a set of
/// spectral lines (Phase 2 step 78 / r276).
///
/// Verbatim (printed p.111):
///
/// ```text
/// Xspl(n) = 10*log10( Σ_k 10^(X(k)/10) ) dB        k in subband n
/// ```
///
/// The caller selects the lines ("k in subband n" — see
/// [`model1_step2_subband_xspl_db`] for the Table D.5-driven subband
/// selection); this primitive evaluates the dB-domain power sum over
/// exactly the lines it is handed. `-∞` dB (silent) lines contribute
/// zero linear power; an empty (or all-silent) selection yields
/// `10·log10(0) = -∞` dB.
///
/// **Determinism.** Pure fold over the input slice.
///
/// Provenance: only the Step 2 alternative-method `Xspl(n)` formula
/// transcribed from ISO/IEC 11172-3:1993 Annex D §D.1 Step 2 (printed
/// p.111) in `docs/audio/mp3/ISO_IEC_11172-3-MP3-1993.pdf` is read;
/// no external implementation was consulted.
#[must_use]
pub fn model1_step2_xspl_db(lines_db: &[f64]) -> f64 {
    let linear_sum: f64 = lines_db.iter().map(|&db| 10.0_f64.powf(db / 10.0)).sum();
    10.0 * linear_sum.log10()
}

/// §D.1 Step 2 maximum spectral line of subband `n` over the Table
/// D.5 line span (Phase 2 step 78 / r276).
///
/// Selects "the spectral line … of the FFT with the maximum amplitude
/// in the frequency range corresponding to subband n" (printed p.110)
/// from a step-77 [`model1_power_density_spectrum`] half-spectrum.
/// The "frequency range corresponding to subband n" is read from
/// Table D.5: partition `n ∈ 1..=32` spans the inclusive 1-based FFT
/// lines `[ωlow_n, ωhigh_n]` ([`coder_partition_d5_line_range`]),
/// mapped onto the spectrum vector via `k = ω − 1`. Adjacent spans
/// share their boundary cell (the table's dual-role `ωlow_{n+1} /
/// ωhigh_n` column) — harmless under a maximum. Returns `None` for
/// `n` outside `1..=32` or when `x` is not the 513-line (1 024-sample
/// FFT) half-spectrum the D.5 ω-indexing addresses; no alternative
/// line mapping is invented.
///
/// **Determinism.** Pure table lookup + slice maximum.
///
/// Provenance: the Step 2 maximum-amplitude sentence (printed p.110)
/// in `docs/audio/mp3/ISO_IEC_11172-3-MP3-1993.pdf` plus the Table
/// D.5 spans already transcribed in
/// `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`; no
/// external implementation was consulted.
#[must_use]
pub fn model1_step2_subband_max_line_db(x: &[f64], n: u16) -> Option<f64> {
    if x.len() != MODEL1_FFT_LEN_LAYER2 / 2 + 1 {
        return None;
    }
    let (low, high) = coder_partition_d5_line_range(n)?;
    Some(
        x[(low as usize - 1)..=(high as usize - 1)]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max),
    )
}

/// §D.1 Step 2 alternative sound pressure level `Xspl(n)` of subband
/// `n` over the Table D.5 line span (Phase 2 step 78 / r276).
///
/// Composes [`model1_step2_xspl_db`] with the same Table D.5 "k in
/// subband n" line selection as [`model1_step2_subband_max_line_db`]:
/// partition `n ∈ 1..=32`'s inclusive 1-based span `[ωlow_n, ωhigh_n]`
/// mapped via `k = ω − 1` onto a 513-line step-77 half-spectrum
/// (`None` otherwise). The dual-role boundary cell shared by adjacent
/// spans is part of both subbands' sums, exactly as the D.5 column
/// prints it; no exclusive re-partitioning is invented.
///
/// **Determinism.** Pure table lookup + dB-domain power sum.
///
/// Provenance: the Step 2 alternative-method `Xspl(n)` formula
/// (printed p.111) in `docs/audio/mp3/ISO_IEC_11172-3-MP3-1993.pdf`
/// plus the Table D.5 spans already transcribed in
/// `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`; no
/// external implementation was consulted.
#[must_use]
pub fn model1_step2_subband_xspl_db(x: &[f64], n: u16) -> Option<f64> {
    if x.len() != MODEL1_FFT_LEN_LAYER2 / 2 + 1 {
        return None;
    }
    let (low, high) = coder_partition_d5_line_range(n)?;
    Some(model1_step2_xspl_db(
        &x[(low as usize - 1)..=(high as usize - 1)],
    ))
}

// =====================================================================
// Annex D Model 1 — §D.1 Step 4 "Finding of tonal and non-tonal
// components" (Phase 2 step 79 / r277).
//
// Step 4 classifies the step-77 SPL spectrum `X(k)` into discrete
// tonal maskers (sinusoid-like local maxima) and one non-tonal
// (noise) masker per critical band. The spec text (printed
// p.111–112) defines three operations:
//
//   (a) Labelling of local maxima —
//       "A spectral line X(k) is labelled as a local maximum if
//        X(k) > X(k-1) and X(k) >= X(k+1)"
//
//   (b) Listing of tonal components and calculation of the sound
//       pressure level — "A local maximum is put in the list of
//       tonal components if X(k) - X(k+j) >= 7 dB", with the offset
//       set j chosen by layer and `k` range (transcribed verbatim in
//       `model1_step4_tonal_check_offsets`). A listed component
//       carries the index k, the SPL
//       `X_tm(k) = 10·log10(10^(X(k-1)/10) + 10^(X(k)/10) +
//       10^(X(k+1)/10))` dB, and the tonal flag. "Next, all spectral
//       lines within the examined frequency range are set to -∞ dB."
//
//   (c) Listing of non-tonal components and calculation of the
//       power — "Within each critical band, the power of the
//       spectral lines (remaining after the tonal components have
//       been zeroed) are summed to form the sound pressure level of
//       the new non-tonal component X_nm(k) corresponding to that
//       critical band", listed at the "index number k of the
//       spectral line nearest to the geometric mean of the critical
//       band" with the non-tonal flag. The critical bands are the
//       Tables D.2a–f boundaries already transcribed above.
//
// Reading notes (documented choices, all from the spec text alone):
//
// * Operation (a) labels the local maxima of the spectrum *before*
//   any zeroing (it is a separate, earlier operation in the spec's
//   enumerated list), so this implementation evaluates every (a)/(b)
//   decision — including the 7 dB examinations and the three-line
//   SPL sums — against a snapshot of the input spectrum, and applies
//   the "set to -∞ dB" zeroing only as the feed into operation (c).
//   Sequential within-pass zeroing would manufacture order-dependent
//   maxima that operation (a) never labelled. Two genuine tonal
//   components close enough to share examined ranges both list here;
//   §D.1 Step 5(b)'s 0,5-Bark decimation is the spec's dedup stage.
// * The multi-valued `j` condition is read as "for every j in the
//   set" — a single passing offset cannot discriminate a sinusoid
//   from noise, and the spec prints the set as a single condition
//   over the whole list.
// * "All spectral lines within the examined frequency range" is the
//   contiguous run `k − j_max ..= k + j_max` (the df window around
//   the maximum whose half-width the j set encodes), which includes
//   the component's own three SPL lines.
// * The spec prints per-sampling-rate `df` values in Hz but a single
//   layer-wide `j` table in line units; the `j` table is the
//   operative listing rule ("where j is chosen according to") and is
//   what this implementation transcribes.
// * The critical-band line spans for operation (c) follow the
//   crate's established Tables D.2 reading (`band_of_fft_line`):
//   each row is the inclusive *top* of its band (the docs file
//   glosses `index F&CB` as "the top FFT line of this band"), so
//   band `no` spans raw lines `(top(no−1), top(no)]` with band 0
//   starting at line 1 (DC line 0 is in no band). Raw-line tops are
//   recovered from each row's exact `frequency [Hz]` column via
//   `k = round(f · N / Fs)` — the D.2 frequencies are exact
//   line-center multiples of `Fs/N`, whereas the `index F&CB` column
//   indexes the *subsampled* Table D.1 domain and cannot address the
//   full-resolution spectrum that Step 4(c) sums.
// * "Nearest to the geometric mean of the critical band" is
//   evaluated over the band's own line span: `round(sqrt(k_first ·
//   k_last))` (frequency is proportional to line index, so this is
//   the frequency-domain geometric mean of the band's first/last
//   line centers, and it always lands inside the summed span).
// =====================================================================

/// §D.1 Step 4(b) tonal-component SPL margin — 7 dB.
///
/// Verbatim (printed p.112): "A local maximum is put in the list of
/// tonal components if X(k) - X(k+j) >= 7 dB".
pub const MODEL1_STEP4_TONAL_DELTA_DB: f64 = 7.0;

/// §D.1 Step 4(b) examined-neighbour offsets, shared first range
/// (`2 < k < 63`, both layers): `j = -2, +2`.
const MODEL1_STEP4_J_NEAR: [i32; 2] = [-2, 2];

/// §D.1 Step 4(b) offsets, shared second range (`63 <= k < 127`,
/// both layers): `j = -3, -2, +2, +3`.
const MODEL1_STEP4_J_MID: [i32; 4] = [-3, -2, 2, 3];

/// §D.1 Step 4(b) offsets, third range (`127 <= k <= 250` Layer I /
/// `127 <= k < 255` Layer II): `j = -6, …, -2, +2, …, +6`.
const MODEL1_STEP4_J_FAR: [i32; 10] = [-6, -5, -4, -3, -2, 2, 3, 4, 5, 6];

/// §D.1 Step 4(b) offsets, Layer II top range (`255 <= k <= 500`):
/// `j = -12, …, -2, +2, …, +12`.
const MODEL1_STEP4_J_TOP_LAYER2: [i32; 22] = [
    -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
];

/// §D.1 Step 4(b) examined-neighbour offset set `j` for a candidate
/// local maximum at spectral line `k` (Phase 2 step 79 / r277).
///
/// Verbatim (printed p.112), "where j is chosen according to":
///
/// ```text
/// Layer I:
///   j = -2, +2                          for   2 <  k <  63
///   j = -3, -2, +2, +3                  for  63 <= k < 127
///   j = -6, …, -2, +2, …, +6            for 127 <= k <= 250
///
/// Layer II:
///   j = -2, +2                          for   2 <  k <  63
///   j = -3, -2, +2, +3                  for  63 <= k < 127
///   j = -6, …, -2, +2, …, +6            for 127 <= k < 255
///   j = -12, …, -2, +2, …, +12          for 255 <= k <= 500
/// ```
///
/// Returns `None` for any `k` outside the listed ranges (`k <= 2`,
/// `k > 250` Layer I, `k > 500` Layer II) — the spec defines no
/// examination there, so no line outside the table can be listed as
/// tonal — and `None` for Layer III (the D.1 preamble adapts the
/// Layer II 1 024-point model to Layer III; a Layer III caller passes
/// `LayerII` explicitly, mirroring [`critical_band_boundaries`]).
///
/// **Determinism.** Pure range dispatch onto `'static` tables.
///
/// Provenance: only the Step 4(b) `j` listing transcribed above from
/// ISO/IEC 11172-3:1993 Annex D §D.1 Step 4 (printed p.112) in
/// `docs/audio/mp3/ISO_IEC_11172-3-MP3-1993.pdf` is read; no external
/// implementation was consulted.
#[must_use]
pub fn model1_step4_tonal_check_offsets(
    layer: crate::frame::Layer,
    k: usize,
) -> Option<&'static [i32]> {
    use crate::frame::Layer;
    match layer {
        Layer::LayerI => match k {
            3..=62 => Some(&MODEL1_STEP4_J_NEAR),
            63..=126 => Some(&MODEL1_STEP4_J_MID),
            127..=250 => Some(&MODEL1_STEP4_J_FAR),
            _ => None,
        },
        Layer::LayerII => match k {
            3..=62 => Some(&MODEL1_STEP4_J_NEAR),
            63..=126 => Some(&MODEL1_STEP4_J_MID),
            127..=254 => Some(&MODEL1_STEP4_J_FAR),
            255..=500 => Some(&MODEL1_STEP4_J_TOP_LAYER2),
            _ => None,
        },
        Layer::LayerIII => None,
    }
}

/// §D.1 Step 4(a) local-maximum label for spectral line `k` (Phase 2
/// step 79 / r277).
///
/// Verbatim (printed p.112): "A spectral line X(k) is labelled as a
/// local maximum if X(k) > X(k-1) and X(k) >= X(k+1)" — strict
/// against the lower neighbour, non-strict against the upper.
/// Returns `None` when either neighbour does not exist (`k == 0` or
/// `k + 1 >= x.len()`); the spec formula is undefined there and no
/// boundary extension is invented.
///
/// **Determinism.** Pure two-comparison predicate.
///
/// Provenance: only the Step 4(a) labelling rule quoted above from
/// ISO/IEC 11172-3:1993 Annex D §D.1 Step 4 (printed p.112) in
/// `docs/audio/mp3/ISO_IEC_11172-3-MP3-1993.pdf` is read; no external
/// implementation was consulted.
#[inline]
#[must_use]
pub fn model1_step4_is_local_maximum(x: &[f64], k: usize) -> Option<bool> {
    if k == 0 || k + 1 >= x.len() {
        return None;
    }
    Some(x[k] > x[k - 1] && x[k] >= x[k + 1])
}

/// §D.1 Step 4(b) tonality test for spectral line `k` (Phase 2 step
/// 79 / r277).
///
/// `Some(true)` iff line `k` is a Step 4(a) local maximum **and**
/// `X(k) − X(k+j) >= 7 dB` holds for *every* `j` in the
/// [`model1_step4_tonal_check_offsets`] set for `(layer, k)` (the
/// multi-valued condition is read conjunctively — a single passing
/// offset cannot discriminate a sinusoid from noise). Returns `None`
/// when the spec defines no examination at `k` (no `j` set — see the
/// offsets accessor) or when any examined line `k + j` falls outside
/// the supplied spectrum; both spec transform lengths keep every
/// listed `k` range fully in bounds, so `None` from the bounds check
/// only arises on truncated input.
///
/// `-∞` dB (silent / already-zeroed) lines behave per IEEE
/// arithmetic: a `-∞` line is never a local maximum (`-∞ > -∞` is
/// false) and `-∞ − -∞ = NaN` fails the `>= 7` comparison.
///
/// **Determinism.** Pure predicate over the slice.
///
/// Provenance: only the Step 4(a)/(b) rules quoted in
/// [`model1_step4_is_local_maximum`] / the offsets accessor from
/// ISO/IEC 11172-3:1993 Annex D §D.1 Step 4 (printed p.112) in
/// `docs/audio/mp3/ISO_IEC_11172-3-MP3-1993.pdf` are read; no
/// external implementation was consulted.
#[must_use]
pub fn model1_step4_is_tonal(x: &[f64], layer: crate::frame::Layer, k: usize) -> Option<bool> {
    let offsets = model1_step4_tonal_check_offsets(layer, k)?;
    let local_max = model1_step4_is_local_maximum(x, k)?;
    // `j_max` is the largest |j| (the sets are symmetric); every
    // examined line must exist.
    let j_max = offsets.last().copied().unwrap_or(0) as usize;
    if k < j_max || k + j_max >= x.len() {
        return None;
    }
    Some(
        local_max
            && offsets.iter().all(|&j| {
                let neighbour = (k as i64 + i64::from(j)) as usize;
                x[k] - x[neighbour] >= MODEL1_STEP4_TONAL_DELTA_DB
            }),
    )
}

/// §D.1 Step 4(b) tonal-component sound pressure level `X_tm(k)`
/// (Phase 2 step 79 / r277).
///
/// Verbatim (printed p.112): a listed tonal component carries
///
/// ```text
/// X_tm(k) = 10 * log10( 10^(X(k-1)/10) + 10^(X(k)/10) + 10^(X(k+1)/10) )  dB
/// ```
///
/// — the dB-domain power sum of the maximum and its two immediate
/// neighbours (the sinusoid's Hann-window leakage lines). Returns
/// `None` when either neighbour does not exist. Implemented as
/// [`model1_step2_xspl_db`] over the three-line window (it is the
/// same power-sum formula).
///
/// **Determinism.** Pure three-term dB-domain sum.
///
/// Provenance: only the Step 4(b) `X_tm(k)` formula quoted above from
/// ISO/IEC 11172-3:1993 Annex D §D.1 Step 4 (printed p.112) in
/// `docs/audio/mp3/ISO_IEC_11172-3-MP3-1993.pdf` is read; no external
/// implementation was consulted.
#[inline]
#[must_use]
pub fn model1_step4_tonal_spl_db(x: &[f64], k: usize) -> Option<f64> {
    if k == 0 || k + 1 >= x.len() {
        return None;
    }
    Some(model1_step2_xspl_db(&x[k - 1..=k + 1]))
}

/// One §D.1 Step 4 masking component: the listed parameters are the
/// spectral-line index `k`, the sound pressure level, and the
/// tonal / non-tonal flag (verbatim the three list entries of
/// operations (b) and (c), printed p.112).
///
/// `k` is the 0-based step-77 spectrum index (the spec's FFT bin
/// `k = 0…N/2`). For tonal components it is the local-maximum line;
/// for non-tonal components it is "the spectral line nearest to the
/// geometric mean of the critical band". The Bark-coordinate mapping
/// (`z(k)` via Tables D.1) that Steps 5–7's [`Masker`] carrier needs
/// remains blocked on the PNG-only Tables D.1 transcription, so this
/// carrier stays in the line-index domain.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Model1Step4Component {
    /// 0-based spectral-line index `k` of the component.
    pub k: u16,
    /// Sound pressure level of the component in dB (`X_tm(k)` for
    /// tonal, `X_nm(k)` for non-tonal).
    pub spl_db: f64,
    /// Tonal / non-tonal flag.
    pub kind: MaskerKind,
}

/// §D.1 Step 4(a)+(b) tonal-component extraction (Phase 2 step 79 /
/// r277).
///
/// Scans the spectrum in increasing `k` over the spec's examined
/// ranges, lists every line that passes [`model1_step4_is_tonal`] as
/// a [`Model1Step4Component`] with its [`model1_step4_tonal_spl_db`]
/// SPL and the tonal flag, and then applies the verbatim "Next, all
/// spectral lines within the examined frequency range are set to
/// -∞ dB" zeroing to `x` — the contiguous run `k − j_max ..= k +
/// j_max` around each listed maximum — leaving `x` as the residual
/// spectrum that operation (c) ([`model1_step4_non_tonal_components`])
/// sums.
///
/// All (a)/(b) decisions (local-maximum labels, 7 dB examinations,
/// three-line SPL sums) are evaluated against a snapshot of the input
/// spectrum: operation (a) labels the maxima of the spectrum before
/// any zeroing, so within-pass zeroing must not manufacture or
/// destroy candidates (see the module-level reading notes). Returns
/// `None` unless `x` is exactly the layer's step-77 half-spectrum
/// (257 lines for Layer I's 512-point FFT, 513 for Layer II's
/// 1 024-point; Layer III → `None`, pass `LayerII` per the D.1
/// preamble's adaptation).
///
/// **Determinism.** Pure function of the input block (plus the
/// in-place residue write-back).
///
/// Provenance: only the §D.1 Step 4 operations (a)/(b) text (printed
/// p.112) in `docs/audio/mp3/ISO_IEC_11172-3-MP3-1993.pdf` is read;
/// no external implementation was consulted.
#[must_use]
pub fn model1_step4_extract_tonal(
    x: &mut [f64],
    layer: crate::frame::Layer,
) -> Option<Vec<Model1Step4Component>> {
    use crate::frame::Layer;
    let expected = match layer {
        Layer::LayerI => MODEL1_FFT_LEN_LAYER1 / 2 + 1,
        Layer::LayerII => MODEL1_FFT_LEN_LAYER2 / 2 + 1,
        Layer::LayerIII => return None,
    };
    if x.len() != expected {
        return None;
    }
    let snapshot = x.to_vec();
    let mut components = Vec::new();
    for k in 3..x.len() - 1 {
        let Some(offsets) = model1_step4_tonal_check_offsets(layer, k) else {
            continue;
        };
        if model1_step4_is_tonal(&snapshot, layer, k) != Some(true) {
            continue;
        }
        // The neighbours exist for every examined k, so the SPL
        // accessor cannot return `None` here.
        let spl_db = model1_step4_tonal_spl_db(&snapshot, k)?;
        components.push(Model1Step4Component {
            k: k as u16,
            spl_db,
            kind: MaskerKind::Tonal,
        });
        let j_max = offsets.last().copied().unwrap_or(0) as usize;
        for line in &mut x[k - j_max..=k + j_max] {
            *line = f64::NEG_INFINITY;
        }
    }
    Some(components)
}

/// Raw-spectral-line span of one §D.1 Step 4(c) critical band: band
/// `no` covers the inclusive 0-based step-77 lines `k_first ..=
/// k_last`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Model1Step4BandSpan {
    /// Critical-band number (the Tables D.2 `no` column).
    pub no: u16,
    /// First 0-based spectral line of the band (inclusive).
    pub k_first: u16,
    /// Last 0-based spectral line of the band (inclusive; the band's
    /// top boundary line per the Tables D.2 `frequency [Hz]` column).
    pub k_last: u16,
}

/// §D.1 Step 4(c) critical-band spans in raw step-77 line units
/// (Phase 2 step 79 / r277).
///
/// Maps the Tables D.2a–f boundary rows for `(layer, fs)` onto the
/// full-resolution spectrum: each row's exact `frequency [Hz]` column
/// is converted to its 0-based line index via `k = round(f · N / Fs)`
/// (`N` = 512 Layer I / 1 024 Layer II; the D.2 frequencies are exact
/// line-center multiples of `Fs/N`), each row is the inclusive top of
/// its band per the crate's established Tables D.2 reading
/// ([`band_of_fft_line`]), band 0 starts at line 1, and DC line 0
/// belongs to no band. The row `index F&CB` column is *not* used
/// here: it indexes the subsampled Table D.1 domain, which cannot
/// address the full-resolution lines that Step 4(c) sums. Returns
/// `None` exactly when [`critical_band_boundaries`] does (Layer III).
///
/// **Determinism.** Pure table transformation.
///
/// Provenance: the Tables D.2a–f rows already transcribed in
/// `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md` plus the
/// §D.1 Step 4(c) prose (printed p.112) in
/// `docs/audio/mp3/ISO_IEC_11172-3-MP3-1993.pdf`; no external
/// implementation was consulted.
#[must_use]
pub fn model1_step4_band_line_spans(
    layer: crate::frame::Layer,
    fs: AnnexDSamplingRate,
) -> Option<Vec<Model1Step4BandSpan>> {
    use crate::frame::Layer;
    let boundaries = critical_band_boundaries(layer, fs)?;
    let fft_len = match layer {
        Layer::LayerI => MODEL1_FFT_LEN_LAYER1,
        Layer::LayerII => MODEL1_FFT_LEN_LAYER2,
        // `critical_band_boundaries` already returned `None`.
        Layer::LayerIII => return None,
    };
    let fs_hz = f64::from(fs.as_hz());
    let mut prev_top = 0u16;
    Some(
        boundaries
            .iter()
            .map(|b| {
                let top = (b.frequency_hz * (fft_len as f64) / fs_hz).round() as u16;
                let span = Model1Step4BandSpan {
                    no: b.no,
                    k_first: prev_top + 1,
                    k_last: top,
                };
                prev_top = top;
                span
            })
            .collect(),
    )
}

/// §D.1 Step 4(c) non-tonal components from the tonal-zeroed residual
/// spectrum (Phase 2 step 79 / r277).
///
/// For each [`model1_step4_band_line_spans`] critical band the power
/// of the residual lines is summed in the dB domain (the same
/// [`model1_step2_xspl_db`] power sum — verbatim "the power of the
/// spectral lines (remaining after the tonal components have been
/// zeroed) are summed to form the sound pressure level of the new
/// non-tonal component"), listed at the "index number k of the
/// spectral line nearest to the geometric mean of the critical band"
/// — evaluated over the band's own line span as
/// `round(sqrt(k_first · k_last))`, which is the frequency-domain
/// geometric mean of the band's first/last line centers (frequency ∝
/// line index) and always lands inside the summed span — with the
/// non-tonal flag. A band whose lines were all zeroed by Step 4(b)
/// yields a `-∞` dB component verbatim (zero linear power; Step 5(a)
/// screens it out against LTq). Returns `None` unless `x` is exactly
/// the layer's step-77 half-spectrum length (Layer III → `None`).
///
/// **Determinism.** Pure per-band fold.
///
/// Provenance: only the §D.1 Step 4(c) prose (printed p.112) in
/// `docs/audio/mp3/ISO_IEC_11172-3-MP3-1993.pdf` plus the in-repo
/// Tables D.2 transcription are read; no external implementation was
/// consulted.
#[must_use]
pub fn model1_step4_non_tonal_components(
    x: &[f64],
    layer: crate::frame::Layer,
    fs: AnnexDSamplingRate,
) -> Option<Vec<Model1Step4Component>> {
    use crate::frame::Layer;
    let expected = match layer {
        Layer::LayerI => MODEL1_FFT_LEN_LAYER1 / 2 + 1,
        Layer::LayerII => MODEL1_FFT_LEN_LAYER2 / 2 + 1,
        Layer::LayerIII => return None,
    };
    if x.len() != expected {
        return None;
    }
    let spans = model1_step4_band_line_spans(layer, fs)?;
    Some(
        spans
            .iter()
            .map(|s| {
                let spl_db = model1_step2_xspl_db(&x[s.k_first as usize..=s.k_last as usize]);
                let gm = (f64::from(s.k_first) * f64::from(s.k_last)).sqrt().round() as u16;
                Model1Step4Component {
                    k: gm,
                    spl_db,
                    kind: MaskerKind::NonTonal,
                }
            })
            .collect(),
    )
}

/// §D.1 Step 4 end-to-end classification: tonal list + non-tonal list
/// (Phase 2 step 79 / r277).
///
/// Composes the three Step 4 operations on a copy of the input
/// spectrum: [`model1_step4_extract_tonal`] lists the tonal
/// components and zeroes their examined ranges in the copy, then
/// [`model1_step4_non_tonal_components`] sums the residue per
/// critical band. Returns `(tonal, non_tonal)`, or `None` under the
/// same `(layer, fs, length)` validation as the two stages.
///
/// **Determinism.** Pure function of the input spectrum.
///
/// Provenance: composition of the Step 4 primitives above; no
/// additional spec material and no external implementation consulted.
#[must_use]
pub fn model1_step4_components(
    x: &[f64],
    layer: crate::frame::Layer,
    fs: AnnexDSamplingRate,
) -> Option<(Vec<Model1Step4Component>, Vec<Model1Step4Component>)> {
    let mut residue = x.to_vec();
    let tonal = model1_step4_extract_tonal(&mut residue, layer)?;
    let non_tonal = model1_step4_non_tonal_components(&residue, layer, fs)?;
    Some((tonal, non_tonal))
}

// =====================================================================
// Annex D Model 1 — Tables D.1a–f (frequencies, critical band rates,
// absolute thresholds) + the Step 4 → Bark bridge and the end-to-end
// §D.1 Step 5 sieve (Phase 2 step 80 / r278).
//
// Spec context (ISO/IEC 11172-3:1993, informative Annex D): the six
// "Table D.1x -- Frequencies, critical band rates and absolute
// threshold" pages tabulate, per (Layer, Fs) pair, the subsampled
// frequency grid the model works on. Each row carries four columns:
//
//   Index Number i | Frequency [Hz] | Crit.Band Rate [z] | Absolute
//                                                          Thresh. [dB]
//
//   D.1a — Layer I,  Fs = 32 kHz   (108 rows, printed p.116)
//   D.1b — Layer I,  Fs = 44,1 kHz (106 rows, printed p.117)
//   D.1c — Layer I,  Fs = 48 kHz   (102 rows, printed p.118)
//   D.1d — Layer II, Fs = 32 kHz   (132 rows, printed p.119)
//   D.1e — Layer II, Fs = 44,1 kHz (130 rows, printed p.120)
//   D.1f — Layer II, Fs = 48 kHz   (126 rows, printed p.121)
//
// Transcribed from the staged page renders
// `docs/audio/mp3/annex-d-renders/Table-D.1{a,b,c,d,e,f}-*.png`
// (read at high magnification in cropped strips; every previously
// ambiguous cell re-read in a dedicated zoom). Decimal commas are
// reproduced as periods and thin-space thousands separators stripped,
// the same convention as the Tables D.2 constants above.
//
// The index column is implicit (row position + 1). The printed
// frequency column reveals the table's subsampling structure — the
// rows sit on the FFT line grid `Fs/N` (`N` = 512 Layer I / 1 024
// Layer II) at:
//
//   rows  1..=48 : lines   1..=48  (every line)
//   rows 49..=72 : lines  50..=96  (every 2nd line)
//   rows 73..=96 : lines 100..=192 (every 4th line; Layer I tables
//                  continue this region to their last row)
//   rows 97..    : lines 200..     (every 8th line; Layer II only)
//
// which [`model1_d1_line_for_index`] encodes and a unit test verifies
// against the printed frequency column of all 704 rows.
//
// Redundancy in the printed tables (used as transcription
// cross-checks, all pinned by unit tests):
//   - a Layer II table prints exactly the same frequency / z / LTq
//     values as the Layer I table at the same Fs wherever the two
//     grids coincide (Layer I line L = Layer II line 2L); rows 49..
//     of D.1d/e/f equal rows 25.. of D.1a/b/c verbatim.
//   - the Tables D.2 `index F&CB` column indexes into these tables:
//     every D.2 boundary row's frequency / Bark pair equals the
//     Table D.1 row it cites. This resolves the documented D.2e
//     band-17 illegible Bark digit: D.1e row 62 legibly prints
//     `16,110` (so the docs file's prose estimate of `16,116` is
//     wrong, and the stored legible-prefix `16.11` is in fact the
//     exact value). A systematic print difference also surfaces:
//     at 44,1 kHz the spec's D.2 tables print a Bark value exactly
//     0,001 below the D.1 tables' at three frequencies —
//     4 478,9 Hz (D.2b band 17 / D.2e band 19: `17,904` vs D.1b row
//     50 / D.1e row 74: `17,905`), 7 579,7 Hz (D.2b band 20 / D.2e
//     band 22: `20,971` vs `20,972`) and 19 982,8 Hz (D.2b band 24 /
//     D.2e band 26: `24,573` vs `24,574`). Each side is
//     double-printed and self-consistent (D.2b = D.2e, D.1b = D.1e),
//     so this is a rounding inconsistency in the printed spec, not a
//     transcription error; both verbatim prints are kept, the bridge
//     below reads the D.1 values, and the
//     `table_d1_agrees_with_d2_boundary_rows` test pins the exact
//     six-cell exception list.
//
// With the z column in hand, the §D.1 Step 4 component lists (line
// index k + SPL, r277) can finally be lifted into the Bark-domain
// [`Masker`] carrier consumed by the already-landed Step 5(a)/(b)
// primitives (r229) and the Step 6/7 threshold evaluators (r219):
// `model1_masker_from_component` does the per-component lift via the
// nearest Table D.1 row, and `model1_step5_components` composes
// bridge + Step 5(a) threshold-in-quiet screening (against the same
// row's Absolute Thresh. column) + Step 5(b) 0,5-Bark tonal
// decimation into the spec's full Step 5 sieve.
//
// Provenance: only the six PNG renders named above plus the §D.1
// Step 5 prose (printed p.112) already quoted on the Step 5
// primitives; no external implementation was consulted.
// =====================================================================

/// One row of Annex D Table D.1 (frequencies, critical band rates and
/// absolute threshold). The row's 1-based `Index Number i` column is
/// implicit (slice position + 1); [`model1_d1_line_for_index`] maps it
/// to the raw FFT-line index the step-77 spectrum uses.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Model1ThresholdEntry {
    /// `Frequency [Hz]` column (the FFT line-center frequency, printed
    /// to two decimals).
    pub frequency_hz: f64,
    /// `Crit.Band Rate [z]` column — the critical-band rate (Bark
    /// coordinate) of this frequency.
    pub z_bark: f64,
    /// `Absolute Thresh. [dB]` column — the threshold in quiet
    /// `LTq` at this frequency, in dB SPL.
    pub ltq_db: f64,
}

impl Model1ThresholdEntry {
    /// Construct a table row at compile time.
    #[inline]
    #[must_use]
    pub const fn new(frequency_hz: f64, z_bark: f64, ltq_db: f64) -> Self {
        Self {
            frequency_hz,
            z_bark,
            ltq_db,
        }
    }
}

/// Table D.1a — Layer I, Fs = 32 kHz (108 rows, printed p.116).
///
/// Verbatim from
/// `docs/audio/mp3/annex-d-renders/Table-D.1a-threshold-in-quiet-LayerI-32kHz-p116.png`.
// The 6,28 dB threshold at 500 Hz is the spec's printed table value,
// not an approximation of a mathematical constant.
#[allow(clippy::approx_constant)]
pub const MODEL1_THRESHOLD_D1A: [Model1ThresholdEntry; 108] = [
    Model1ThresholdEntry::new(62.50, 0.617, 33.44),
    Model1ThresholdEntry::new(125.00, 1.232, 19.20),
    Model1ThresholdEntry::new(187.50, 1.842, 13.87),
    Model1ThresholdEntry::new(250.00, 2.445, 11.01),
    Model1ThresholdEntry::new(312.50, 3.037, 9.20),
    Model1ThresholdEntry::new(375.00, 3.618, 7.94),
    Model1ThresholdEntry::new(437.50, 4.185, 7.00),
    Model1ThresholdEntry::new(500.00, 4.736, 6.28),
    Model1ThresholdEntry::new(562.50, 5.272, 5.70),
    Model1ThresholdEntry::new(625.00, 5.789, 5.21),
    Model1ThresholdEntry::new(687.50, 6.289, 4.80),
    Model1ThresholdEntry::new(750.00, 6.770, 4.45),
    Model1ThresholdEntry::new(812.50, 7.233, 4.14),
    Model1ThresholdEntry::new(875.00, 7.677, 3.86),
    Model1ThresholdEntry::new(937.50, 8.103, 3.61),
    Model1ThresholdEntry::new(1000.00, 8.511, 3.37),
    Model1ThresholdEntry::new(1062.50, 8.901, 3.15),
    Model1ThresholdEntry::new(1125.00, 9.275, 2.93),
    Model1ThresholdEntry::new(1187.50, 9.632, 2.73),
    Model1ThresholdEntry::new(1250.00, 9.974, 2.53),
    Model1ThresholdEntry::new(1312.50, 10.301, 2.32),
    Model1ThresholdEntry::new(1375.00, 10.614, 2.12),
    Model1ThresholdEntry::new(1437.50, 10.913, 1.92),
    Model1ThresholdEntry::new(1500.00, 11.199, 1.71),
    Model1ThresholdEntry::new(1562.50, 11.474, 1.49),
    Model1ThresholdEntry::new(1625.00, 11.736, 1.27),
    Model1ThresholdEntry::new(1687.50, 11.988, 1.04),
    Model1ThresholdEntry::new(1750.00, 12.230, 0.80),
    Model1ThresholdEntry::new(1812.50, 12.461, 0.55),
    Model1ThresholdEntry::new(1875.00, 12.684, 0.29),
    Model1ThresholdEntry::new(1937.50, 12.898, 0.02),
    Model1ThresholdEntry::new(2000.00, 13.104, -0.25),
    Model1ThresholdEntry::new(2062.50, 13.302, -0.54),
    Model1ThresholdEntry::new(2125.00, 13.493, -0.83),
    Model1ThresholdEntry::new(2187.50, 13.678, -1.12),
    Model1ThresholdEntry::new(2250.00, 13.855, -1.43),
    Model1ThresholdEntry::new(2312.50, 14.027, -1.73),
    Model1ThresholdEntry::new(2375.00, 14.193, -2.04),
    Model1ThresholdEntry::new(2437.50, 14.354, -2.34),
    Model1ThresholdEntry::new(2500.00, 14.509, -2.64),
    Model1ThresholdEntry::new(2562.50, 14.660, -2.93),
    Model1ThresholdEntry::new(2625.00, 14.807, -3.22),
    Model1ThresholdEntry::new(2687.50, 14.949, -3.49),
    Model1ThresholdEntry::new(2750.00, 15.087, -3.74),
    Model1ThresholdEntry::new(2812.50, 15.221, -3.98),
    Model1ThresholdEntry::new(2875.00, 15.351, -4.20),
    Model1ThresholdEntry::new(2937.50, 15.478, -4.40),
    Model1ThresholdEntry::new(3000.00, 15.602, -4.57),
    Model1ThresholdEntry::new(3125.00, 15.841, -4.82),
    Model1ThresholdEntry::new(3250.00, 16.069, -4.96),
    Model1ThresholdEntry::new(3375.00, 16.287, -4.97),
    Model1ThresholdEntry::new(3500.00, 16.496, -4.86),
    Model1ThresholdEntry::new(3625.00, 16.697, -4.63),
    Model1ThresholdEntry::new(3750.00, 16.891, -4.29),
    Model1ThresholdEntry::new(3875.00, 17.078, -3.87),
    Model1ThresholdEntry::new(4000.00, 17.259, -3.39),
    Model1ThresholdEntry::new(4125.00, 17.434, -2.86),
    Model1ThresholdEntry::new(4250.00, 17.605, -2.31),
    Model1ThresholdEntry::new(4375.00, 17.770, -1.77),
    Model1ThresholdEntry::new(4500.00, 17.932, -1.24),
    Model1ThresholdEntry::new(4625.00, 18.089, -0.74),
    Model1ThresholdEntry::new(4750.00, 18.242, -0.29),
    Model1ThresholdEntry::new(4875.00, 18.392, 0.12),
    Model1ThresholdEntry::new(5000.00, 18.539, 0.48),
    Model1ThresholdEntry::new(5125.00, 18.682, 0.79),
    Model1ThresholdEntry::new(5250.00, 18.823, 1.06),
    Model1ThresholdEntry::new(5375.00, 18.960, 1.29),
    Model1ThresholdEntry::new(5500.00, 19.095, 1.49),
    Model1ThresholdEntry::new(5625.00, 19.226, 1.66),
    Model1ThresholdEntry::new(5750.00, 19.356, 1.81),
    Model1ThresholdEntry::new(5875.00, 19.482, 1.95),
    Model1ThresholdEntry::new(6000.00, 19.606, 2.08),
    Model1ThresholdEntry::new(6250.00, 19.847, 2.33),
    Model1ThresholdEntry::new(6500.00, 20.079, 2.59),
    Model1ThresholdEntry::new(6750.00, 20.300, 2.86),
    Model1ThresholdEntry::new(7000.00, 20.513, 3.17),
    Model1ThresholdEntry::new(7250.00, 20.717, 3.51),
    Model1ThresholdEntry::new(7500.00, 20.912, 3.89),
    Model1ThresholdEntry::new(7750.00, 21.098, 4.31),
    Model1ThresholdEntry::new(8000.00, 21.275, 4.79),
    Model1ThresholdEntry::new(8250.00, 21.445, 5.31),
    Model1ThresholdEntry::new(8500.00, 21.606, 5.88),
    Model1ThresholdEntry::new(8750.00, 21.760, 6.50),
    Model1ThresholdEntry::new(9000.00, 21.906, 7.19),
    Model1ThresholdEntry::new(9250.00, 22.046, 7.93),
    Model1ThresholdEntry::new(9500.00, 22.178, 8.75),
    Model1ThresholdEntry::new(9750.00, 22.304, 9.63),
    Model1ThresholdEntry::new(10000.00, 22.424, 10.58),
    Model1ThresholdEntry::new(10250.00, 22.538, 11.60),
    Model1ThresholdEntry::new(10500.00, 22.646, 12.71),
    Model1ThresholdEntry::new(10750.00, 22.749, 13.90),
    Model1ThresholdEntry::new(11000.00, 22.847, 15.18),
    Model1ThresholdEntry::new(11250.00, 22.941, 16.54),
    Model1ThresholdEntry::new(11500.00, 23.030, 18.01),
    Model1ThresholdEntry::new(11750.00, 23.114, 19.57),
    Model1ThresholdEntry::new(12000.00, 23.195, 21.23),
    Model1ThresholdEntry::new(12250.00, 23.272, 23.01),
    Model1ThresholdEntry::new(12500.00, 23.345, 24.90),
    Model1ThresholdEntry::new(12750.00, 23.415, 26.90),
    Model1ThresholdEntry::new(13000.00, 23.482, 29.03),
    Model1ThresholdEntry::new(13250.00, 23.546, 31.28),
    Model1ThresholdEntry::new(13500.00, 23.607, 33.67),
    Model1ThresholdEntry::new(13750.00, 23.666, 36.19),
    Model1ThresholdEntry::new(14000.00, 23.722, 38.86),
    Model1ThresholdEntry::new(14250.00, 23.775, 41.67),
    Model1ThresholdEntry::new(14500.00, 23.827, 44.63),
    Model1ThresholdEntry::new(14750.00, 23.876, 47.76),
    Model1ThresholdEntry::new(15000.00, 23.923, 51.04),
];

/// Table D.1b — Layer I, Fs = 44,1 kHz (106 rows, printed p.117).
///
/// Verbatim from
/// `docs/audio/mp3/annex-d-renders/Table-D.1b-threshold-in-quiet-LayerI-44k1Hz-p117.png`.
pub const MODEL1_THRESHOLD_D1B: [Model1ThresholdEntry; 106] = [
    Model1ThresholdEntry::new(86.13, 0.850, 25.87),
    Model1ThresholdEntry::new(172.27, 1.694, 14.85),
    Model1ThresholdEntry::new(258.40, 2.525, 10.72),
    Model1ThresholdEntry::new(344.53, 3.337, 8.50),
    Model1ThresholdEntry::new(430.66, 4.124, 7.10),
    Model1ThresholdEntry::new(516.80, 4.882, 6.11),
    Model1ThresholdEntry::new(602.93, 5.608, 5.37),
    Model1ThresholdEntry::new(689.06, 6.301, 4.79),
    Model1ThresholdEntry::new(775.20, 6.959, 4.32),
    Model1ThresholdEntry::new(861.33, 7.581, 3.92),
    Model1ThresholdEntry::new(947.46, 8.169, 3.57),
    Model1ThresholdEntry::new(1033.59, 8.723, 3.25),
    Model1ThresholdEntry::new(1119.73, 9.244, 2.95),
    Model1ThresholdEntry::new(1205.86, 9.734, 2.67),
    Model1ThresholdEntry::new(1291.99, 10.195, 2.39),
    Model1ThresholdEntry::new(1378.13, 10.629, 2.11),
    Model1ThresholdEntry::new(1464.26, 11.037, 1.83),
    Model1ThresholdEntry::new(1550.39, 11.421, 1.53),
    Model1ThresholdEntry::new(1636.52, 11.783, 1.23),
    Model1ThresholdEntry::new(1722.66, 12.125, 0.90),
    Model1ThresholdEntry::new(1808.79, 12.448, 0.56),
    Model1ThresholdEntry::new(1894.92, 12.753, 0.21),
    Model1ThresholdEntry::new(1981.05, 13.042, -0.17),
    Model1ThresholdEntry::new(2067.19, 13.317, -0.56),
    Model1ThresholdEntry::new(2153.32, 13.578, -0.96),
    Model1ThresholdEntry::new(2239.45, 13.826, -1.38),
    Model1ThresholdEntry::new(2325.59, 14.062, -1.79),
    Model1ThresholdEntry::new(2411.72, 14.288, -2.21),
    Model1ThresholdEntry::new(2497.85, 14.504, -2.63),
    Model1ThresholdEntry::new(2583.98, 14.711, -3.03),
    Model1ThresholdEntry::new(2670.12, 14.909, -3.41),
    Model1ThresholdEntry::new(2756.25, 15.100, -3.77),
    Model1ThresholdEntry::new(2842.38, 15.284, -4.09),
    Model1ThresholdEntry::new(2928.52, 15.460, -4.37),
    Model1ThresholdEntry::new(3014.65, 15.631, -4.60),
    Model1ThresholdEntry::new(3100.78, 15.796, -4.78),
    Model1ThresholdEntry::new(3186.91, 15.955, -4.91),
    Model1ThresholdEntry::new(3273.05, 16.110, -4.97),
    Model1ThresholdEntry::new(3359.18, 16.260, -4.98),
    Model1ThresholdEntry::new(3445.31, 16.406, -4.92),
    Model1ThresholdEntry::new(3531.45, 16.547, -4.81),
    Model1ThresholdEntry::new(3617.58, 16.685, -4.65),
    Model1ThresholdEntry::new(3703.71, 16.820, -4.43),
    Model1ThresholdEntry::new(3789.84, 16.951, -4.17),
    Model1ThresholdEntry::new(3875.98, 17.079, -3.87),
    Model1ThresholdEntry::new(3962.11, 17.205, -3.54),
    Model1ThresholdEntry::new(4048.24, 17.327, -3.19),
    Model1ThresholdEntry::new(4134.38, 17.447, -2.82),
    Model1ThresholdEntry::new(4306.64, 17.680, -2.06),
    Model1ThresholdEntry::new(4478.91, 17.905, -1.32),
    Model1ThresholdEntry::new(4651.17, 18.121, -0.64),
    Model1ThresholdEntry::new(4823.44, 18.331, -0.04),
    Model1ThresholdEntry::new(4995.70, 18.534, 0.47),
    Model1ThresholdEntry::new(5167.97, 18.731, 0.89),
    Model1ThresholdEntry::new(5340.23, 18.922, 1.23),
    Model1ThresholdEntry::new(5512.50, 19.108, 1.51),
    Model1ThresholdEntry::new(5684.77, 19.289, 1.74),
    Model1ThresholdEntry::new(5857.03, 19.464, 1.93),
    Model1ThresholdEntry::new(6029.30, 19.635, 2.11),
    Model1ThresholdEntry::new(6201.56, 19.801, 2.28),
    Model1ThresholdEntry::new(6373.83, 19.963, 2.46),
    Model1ThresholdEntry::new(6546.09, 20.120, 2.63),
    Model1ThresholdEntry::new(6718.36, 20.273, 2.82),
    Model1ThresholdEntry::new(6890.63, 20.421, 3.03),
    Model1ThresholdEntry::new(7062.89, 20.565, 3.25),
    Model1ThresholdEntry::new(7235.16, 20.705, 3.49),
    Model1ThresholdEntry::new(7407.42, 20.840, 3.74),
    Model1ThresholdEntry::new(7579.69, 20.972, 4.02),
    Model1ThresholdEntry::new(7751.95, 21.099, 4.32),
    Model1ThresholdEntry::new(7924.22, 21.222, 4.64),
    Model1ThresholdEntry::new(8096.48, 21.342, 4.98),
    Model1ThresholdEntry::new(8268.75, 21.457, 5.35),
    Model1ThresholdEntry::new(8613.28, 21.677, 6.15),
    Model1ThresholdEntry::new(8957.81, 21.882, 7.07),
    Model1ThresholdEntry::new(9302.34, 22.074, 8.10),
    Model1ThresholdEntry::new(9646.88, 22.253, 9.25),
    Model1ThresholdEntry::new(9991.41, 22.420, 10.54),
    Model1ThresholdEntry::new(10335.94, 22.576, 11.97),
    Model1ThresholdEntry::new(10680.47, 22.721, 13.56),
    Model1ThresholdEntry::new(11025.00, 22.857, 15.31),
    Model1ThresholdEntry::new(11369.53, 22.984, 17.23),
    Model1ThresholdEntry::new(11714.06, 23.102, 19.34),
    Model1ThresholdEntry::new(12058.59, 23.213, 21.64),
    Model1ThresholdEntry::new(12403.13, 23.317, 24.15),
    Model1ThresholdEntry::new(12747.66, 23.415, 26.88),
    Model1ThresholdEntry::new(13092.19, 23.506, 29.84),
    Model1ThresholdEntry::new(13436.72, 23.592, 33.05),
    Model1ThresholdEntry::new(13781.25, 23.673, 36.52),
    Model1ThresholdEntry::new(14125.78, 23.749, 40.25),
    Model1ThresholdEntry::new(14470.31, 23.821, 44.27),
    Model1ThresholdEntry::new(14814.84, 23.888, 48.59),
    Model1ThresholdEntry::new(15159.38, 23.952, 53.22),
    Model1ThresholdEntry::new(15503.91, 24.013, 58.18),
    Model1ThresholdEntry::new(15848.44, 24.070, 63.49),
    Model1ThresholdEntry::new(16192.97, 24.125, 68.00),
    Model1ThresholdEntry::new(16537.50, 24.176, 68.00),
    Model1ThresholdEntry::new(16882.03, 24.225, 68.00),
    Model1ThresholdEntry::new(17226.56, 24.271, 68.00),
    Model1ThresholdEntry::new(17571.09, 24.316, 68.00),
    Model1ThresholdEntry::new(17915.63, 24.358, 68.00),
    Model1ThresholdEntry::new(18260.16, 24.398, 68.00),
    Model1ThresholdEntry::new(18604.69, 24.436, 68.00),
    Model1ThresholdEntry::new(18949.22, 24.473, 68.00),
    Model1ThresholdEntry::new(19293.75, 24.508, 68.00),
    Model1ThresholdEntry::new(19638.28, 24.542, 68.00),
    Model1ThresholdEntry::new(19982.81, 24.574, 68.00),
];

/// Table D.1c — Layer I, Fs = 48 kHz (102 rows, printed p.118).
///
/// Verbatim from
/// `docs/audio/mp3/annex-d-renders/Table-D.1c-threshold-in-quiet-LayerI-48kHz-p118.png`.
pub const MODEL1_THRESHOLD_D1C: [Model1ThresholdEntry; 102] = [
    Model1ThresholdEntry::new(93.75, 0.925, 24.17),
    Model1ThresholdEntry::new(187.50, 1.842, 13.87),
    Model1ThresholdEntry::new(281.25, 2.742, 10.01),
    Model1ThresholdEntry::new(375.00, 3.618, 7.94),
    Model1ThresholdEntry::new(468.75, 4.463, 6.62),
    Model1ThresholdEntry::new(562.50, 5.272, 5.70),
    Model1ThresholdEntry::new(656.25, 6.041, 5.00),
    Model1ThresholdEntry::new(750.00, 6.770, 4.45),
    Model1ThresholdEntry::new(843.75, 7.457, 4.00),
    Model1ThresholdEntry::new(937.50, 8.103, 3.61),
    Model1ThresholdEntry::new(1031.25, 8.708, 3.26),
    Model1ThresholdEntry::new(1125.00, 9.275, 2.93),
    Model1ThresholdEntry::new(1218.75, 9.805, 2.63),
    Model1ThresholdEntry::new(1312.50, 10.301, 2.32),
    Model1ThresholdEntry::new(1406.25, 10.765, 2.02),
    Model1ThresholdEntry::new(1500.00, 11.199, 1.71),
    Model1ThresholdEntry::new(1593.75, 11.606, 1.38),
    Model1ThresholdEntry::new(1687.50, 11.988, 1.04),
    Model1ThresholdEntry::new(1781.25, 12.347, 0.67),
    Model1ThresholdEntry::new(1875.00, 12.684, 0.29),
    Model1ThresholdEntry::new(1968.75, 13.002, -0.11),
    Model1ThresholdEntry::new(2062.50, 13.302, -0.54),
    Model1ThresholdEntry::new(2156.25, 13.586, -0.97),
    Model1ThresholdEntry::new(2250.00, 13.855, -1.43),
    Model1ThresholdEntry::new(2343.75, 14.111, -1.88),
    Model1ThresholdEntry::new(2437.50, 14.354, -2.34),
    Model1ThresholdEntry::new(2531.25, 14.585, -2.79),
    Model1ThresholdEntry::new(2625.00, 14.807, -3.22),
    Model1ThresholdEntry::new(2718.75, 15.018, -3.62),
    Model1ThresholdEntry::new(2812.50, 15.221, -3.98),
    Model1ThresholdEntry::new(2906.25, 15.415, -4.30),
    Model1ThresholdEntry::new(3000.00, 15.602, -4.57),
    Model1ThresholdEntry::new(3093.75, 15.783, -4.77),
    Model1ThresholdEntry::new(3187.50, 15.956, -4.91),
    Model1ThresholdEntry::new(3281.25, 16.124, -4.98),
    Model1ThresholdEntry::new(3375.00, 16.287, -4.97),
    Model1ThresholdEntry::new(3468.75, 16.445, -4.90),
    Model1ThresholdEntry::new(3562.50, 16.598, -4.76),
    Model1ThresholdEntry::new(3656.25, 16.746, -4.55),
    Model1ThresholdEntry::new(3750.00, 16.891, -4.29),
    Model1ThresholdEntry::new(3843.75, 17.032, -3.99),
    Model1ThresholdEntry::new(3937.50, 17.169, -3.64),
    Model1ThresholdEntry::new(4031.25, 17.303, -3.26),
    Model1ThresholdEntry::new(4125.00, 17.434, -2.86),
    Model1ThresholdEntry::new(4218.75, 17.563, -2.45),
    Model1ThresholdEntry::new(4312.50, 17.688, -2.04),
    Model1ThresholdEntry::new(4406.25, 17.811, -1.63),
    Model1ThresholdEntry::new(4500.00, 17.932, -1.24),
    Model1ThresholdEntry::new(4687.50, 18.166, -0.51),
    Model1ThresholdEntry::new(4875.00, 18.392, 0.12),
    Model1ThresholdEntry::new(5062.50, 18.611, 0.64),
    Model1ThresholdEntry::new(5250.00, 18.823, 1.06),
    Model1ThresholdEntry::new(5437.50, 19.028, 1.39),
    Model1ThresholdEntry::new(5625.00, 19.226, 1.66),
    Model1ThresholdEntry::new(5812.50, 19.419, 1.88),
    Model1ThresholdEntry::new(6000.00, 19.606, 2.08),
    Model1ThresholdEntry::new(6187.50, 19.788, 2.27),
    Model1ThresholdEntry::new(6375.00, 19.964, 2.46),
    Model1ThresholdEntry::new(6562.50, 20.135, 2.65),
    Model1ThresholdEntry::new(6750.00, 20.300, 2.86),
    Model1ThresholdEntry::new(6937.50, 20.461, 3.09),
    Model1ThresholdEntry::new(7125.00, 20.616, 3.33),
    Model1ThresholdEntry::new(7312.50, 20.766, 3.60),
    Model1ThresholdEntry::new(7500.00, 20.912, 3.89),
    Model1ThresholdEntry::new(7687.50, 21.052, 4.20),
    Model1ThresholdEntry::new(7875.00, 21.188, 4.54),
    Model1ThresholdEntry::new(8062.50, 21.318, 4.91),
    Model1ThresholdEntry::new(8250.00, 21.445, 5.31),
    Model1ThresholdEntry::new(8437.50, 21.567, 5.73),
    Model1ThresholdEntry::new(8625.00, 21.684, 6.18),
    Model1ThresholdEntry::new(8812.50, 21.797, 6.67),
    Model1ThresholdEntry::new(9000.00, 21.906, 7.19),
    Model1ThresholdEntry::new(9375.00, 22.113, 8.33),
    Model1ThresholdEntry::new(9750.00, 22.304, 9.63),
    Model1ThresholdEntry::new(10125.00, 22.482, 11.08),
    Model1ThresholdEntry::new(10500.00, 22.646, 12.71),
    Model1ThresholdEntry::new(10875.00, 22.799, 14.53),
    Model1ThresholdEntry::new(11250.00, 22.941, 16.54),
    Model1ThresholdEntry::new(11625.00, 23.072, 18.77),
    Model1ThresholdEntry::new(12000.00, 23.195, 21.23),
    Model1ThresholdEntry::new(12375.00, 23.309, 23.94),
    Model1ThresholdEntry::new(12750.00, 23.415, 26.90),
    Model1ThresholdEntry::new(13125.00, 23.515, 30.14),
    Model1ThresholdEntry::new(13500.00, 23.607, 33.67),
    Model1ThresholdEntry::new(13875.00, 23.694, 37.51),
    Model1ThresholdEntry::new(14250.00, 23.775, 41.67),
    Model1ThresholdEntry::new(14625.00, 23.852, 46.17),
    Model1ThresholdEntry::new(15000.00, 23.923, 51.04),
    Model1ThresholdEntry::new(15375.00, 23.991, 56.29),
    Model1ThresholdEntry::new(15750.00, 24.054, 61.94),
    Model1ThresholdEntry::new(16125.00, 24.114, 68.00),
    Model1ThresholdEntry::new(16500.00, 24.171, 68.00),
    Model1ThresholdEntry::new(16875.00, 24.224, 68.00),
    Model1ThresholdEntry::new(17250.00, 24.275, 68.00),
    Model1ThresholdEntry::new(17625.00, 24.322, 68.00),
    Model1ThresholdEntry::new(18000.00, 24.368, 68.00),
    Model1ThresholdEntry::new(18375.00, 24.411, 68.00),
    Model1ThresholdEntry::new(18750.00, 24.452, 68.00),
    Model1ThresholdEntry::new(19125.00, 24.491, 68.00),
    Model1ThresholdEntry::new(19500.00, 24.528, 68.00),
    Model1ThresholdEntry::new(19875.00, 24.564, 68.00),
    Model1ThresholdEntry::new(20250.00, 24.597, 68.00),
];

/// Table D.1d — Layer II, Fs = 32 kHz (132 rows, printed p.119).
///
/// Verbatim from
/// `docs/audio/mp3/annex-d-renders/Table-D.1d-threshold-in-quiet-LayerII-32kHz-p119.png`.
///
/// Rows 49.. print the same frequency / z / LTq values as the
/// matching Layer I table's rows 25.. (the two line grids coincide
/// there) — verified on the renders and pinned by a unit test.
// The 6,28 dB threshold at 500 Hz is the spec's printed table value,
// not an approximation of a mathematical constant.
#[allow(clippy::approx_constant)]
pub const MODEL1_THRESHOLD_D1D: [Model1ThresholdEntry; 132] = [
    Model1ThresholdEntry::new(31.25, 0.309, 58.23),
    Model1ThresholdEntry::new(62.50, 0.617, 33.44),
    Model1ThresholdEntry::new(93.75, 0.925, 24.17),
    Model1ThresholdEntry::new(125.00, 1.232, 19.20),
    Model1ThresholdEntry::new(156.25, 1.538, 16.05),
    Model1ThresholdEntry::new(187.50, 1.842, 13.87),
    Model1ThresholdEntry::new(218.75, 2.145, 12.26),
    Model1ThresholdEntry::new(250.00, 2.445, 11.01),
    Model1ThresholdEntry::new(281.25, 2.742, 10.01),
    Model1ThresholdEntry::new(312.50, 3.037, 9.20),
    Model1ThresholdEntry::new(343.75, 3.329, 8.52),
    Model1ThresholdEntry::new(375.00, 3.618, 7.94),
    Model1ThresholdEntry::new(406.25, 3.903, 7.44),
    Model1ThresholdEntry::new(437.50, 4.185, 7.00),
    Model1ThresholdEntry::new(468.75, 4.463, 6.62),
    Model1ThresholdEntry::new(500.00, 4.736, 6.28),
    Model1ThresholdEntry::new(531.25, 5.006, 5.97),
    Model1ThresholdEntry::new(562.50, 5.272, 5.70),
    Model1ThresholdEntry::new(593.75, 5.533, 5.44),
    Model1ThresholdEntry::new(625.00, 5.789, 5.21),
    Model1ThresholdEntry::new(656.25, 6.041, 5.00),
    Model1ThresholdEntry::new(687.50, 6.289, 4.80),
    Model1ThresholdEntry::new(718.75, 6.532, 4.62),
    Model1ThresholdEntry::new(750.00, 6.770, 4.45),
    Model1ThresholdEntry::new(781.25, 7.004, 4.29),
    Model1ThresholdEntry::new(812.50, 7.233, 4.14),
    Model1ThresholdEntry::new(843.75, 7.457, 4.00),
    Model1ThresholdEntry::new(875.00, 7.677, 3.86),
    Model1ThresholdEntry::new(906.25, 7.892, 3.73),
    Model1ThresholdEntry::new(937.50, 8.103, 3.61),
    Model1ThresholdEntry::new(968.75, 8.309, 3.49),
    Model1ThresholdEntry::new(1000.00, 8.511, 3.37),
    Model1ThresholdEntry::new(1031.25, 8.708, 3.26),
    Model1ThresholdEntry::new(1062.50, 8.901, 3.15),
    Model1ThresholdEntry::new(1093.75, 9.090, 3.04),
    Model1ThresholdEntry::new(1125.00, 9.275, 2.93),
    Model1ThresholdEntry::new(1156.25, 9.456, 2.83),
    Model1ThresholdEntry::new(1187.50, 9.632, 2.73),
    Model1ThresholdEntry::new(1218.75, 9.805, 2.63),
    Model1ThresholdEntry::new(1250.00, 9.974, 2.53),
    Model1ThresholdEntry::new(1281.25, 10.139, 2.42),
    Model1ThresholdEntry::new(1312.50, 10.301, 2.32),
    Model1ThresholdEntry::new(1343.75, 10.459, 2.22),
    Model1ThresholdEntry::new(1375.00, 10.614, 2.12),
    Model1ThresholdEntry::new(1406.25, 10.765, 2.02),
    Model1ThresholdEntry::new(1437.50, 10.913, 1.92),
    Model1ThresholdEntry::new(1468.75, 11.058, 1.81),
    Model1ThresholdEntry::new(1500.00, 11.199, 1.71),
    Model1ThresholdEntry::new(1562.50, 11.474, 1.49),
    Model1ThresholdEntry::new(1625.00, 11.736, 1.27),
    Model1ThresholdEntry::new(1687.50, 11.988, 1.04),
    Model1ThresholdEntry::new(1750.00, 12.230, 0.80),
    Model1ThresholdEntry::new(1812.50, 12.461, 0.55),
    Model1ThresholdEntry::new(1875.00, 12.684, 0.29),
    Model1ThresholdEntry::new(1937.50, 12.898, 0.02),
    Model1ThresholdEntry::new(2000.00, 13.104, -0.25),
    Model1ThresholdEntry::new(2062.50, 13.302, -0.54),
    Model1ThresholdEntry::new(2125.00, 13.493, -0.83),
    Model1ThresholdEntry::new(2187.50, 13.678, -1.12),
    Model1ThresholdEntry::new(2250.00, 13.855, -1.43),
    Model1ThresholdEntry::new(2312.50, 14.027, -1.73),
    Model1ThresholdEntry::new(2375.00, 14.193, -2.04),
    Model1ThresholdEntry::new(2437.50, 14.354, -2.34),
    Model1ThresholdEntry::new(2500.00, 14.509, -2.64),
    Model1ThresholdEntry::new(2562.50, 14.660, -2.93),
    Model1ThresholdEntry::new(2625.00, 14.807, -3.22),
    Model1ThresholdEntry::new(2687.50, 14.949, -3.49),
    Model1ThresholdEntry::new(2750.00, 15.087, -3.74),
    Model1ThresholdEntry::new(2812.50, 15.221, -3.98),
    Model1ThresholdEntry::new(2875.00, 15.351, -4.20),
    Model1ThresholdEntry::new(2937.50, 15.478, -4.40),
    Model1ThresholdEntry::new(3000.00, 15.602, -4.57),
    Model1ThresholdEntry::new(3125.00, 15.841, -4.82),
    Model1ThresholdEntry::new(3250.00, 16.069, -4.96),
    Model1ThresholdEntry::new(3375.00, 16.287, -4.97),
    Model1ThresholdEntry::new(3500.00, 16.496, -4.86),
    Model1ThresholdEntry::new(3625.00, 16.697, -4.63),
    Model1ThresholdEntry::new(3750.00, 16.891, -4.29),
    Model1ThresholdEntry::new(3875.00, 17.078, -3.87),
    Model1ThresholdEntry::new(4000.00, 17.259, -3.39),
    Model1ThresholdEntry::new(4125.00, 17.434, -2.86),
    Model1ThresholdEntry::new(4250.00, 17.605, -2.31),
    Model1ThresholdEntry::new(4375.00, 17.770, -1.77),
    Model1ThresholdEntry::new(4500.00, 17.932, -1.24),
    Model1ThresholdEntry::new(4625.00, 18.089, -0.74),
    Model1ThresholdEntry::new(4750.00, 18.242, -0.29),
    Model1ThresholdEntry::new(4875.00, 18.392, 0.12),
    Model1ThresholdEntry::new(5000.00, 18.539, 0.48),
    Model1ThresholdEntry::new(5125.00, 18.682, 0.79),
    Model1ThresholdEntry::new(5250.00, 18.823, 1.06),
    Model1ThresholdEntry::new(5375.00, 18.960, 1.29),
    Model1ThresholdEntry::new(5500.00, 19.095, 1.49),
    Model1ThresholdEntry::new(5625.00, 19.226, 1.66),
    Model1ThresholdEntry::new(5750.00, 19.356, 1.81),
    Model1ThresholdEntry::new(5875.00, 19.482, 1.95),
    Model1ThresholdEntry::new(6000.00, 19.606, 2.08),
    Model1ThresholdEntry::new(6250.00, 19.847, 2.33),
    Model1ThresholdEntry::new(6500.00, 20.079, 2.59),
    Model1ThresholdEntry::new(6750.00, 20.300, 2.86),
    Model1ThresholdEntry::new(7000.00, 20.513, 3.17),
    Model1ThresholdEntry::new(7250.00, 20.717, 3.51),
    Model1ThresholdEntry::new(7500.00, 20.912, 3.89),
    Model1ThresholdEntry::new(7750.00, 21.098, 4.31),
    Model1ThresholdEntry::new(8000.00, 21.275, 4.79),
    Model1ThresholdEntry::new(8250.00, 21.445, 5.31),
    Model1ThresholdEntry::new(8500.00, 21.606, 5.88),
    Model1ThresholdEntry::new(8750.00, 21.760, 6.50),
    Model1ThresholdEntry::new(9000.00, 21.906, 7.19),
    Model1ThresholdEntry::new(9250.00, 22.046, 7.93),
    Model1ThresholdEntry::new(9500.00, 22.178, 8.75),
    Model1ThresholdEntry::new(9750.00, 22.304, 9.63),
    Model1ThresholdEntry::new(10000.00, 22.424, 10.58),
    Model1ThresholdEntry::new(10250.00, 22.538, 11.60),
    Model1ThresholdEntry::new(10500.00, 22.646, 12.71),
    Model1ThresholdEntry::new(10750.00, 22.749, 13.90),
    Model1ThresholdEntry::new(11000.00, 22.847, 15.18),
    Model1ThresholdEntry::new(11250.00, 22.941, 16.54),
    Model1ThresholdEntry::new(11500.00, 23.030, 18.01),
    Model1ThresholdEntry::new(11750.00, 23.114, 19.57),
    Model1ThresholdEntry::new(12000.00, 23.195, 21.23),
    Model1ThresholdEntry::new(12250.00, 23.272, 23.01),
    Model1ThresholdEntry::new(12500.00, 23.345, 24.90),
    Model1ThresholdEntry::new(12750.00, 23.415, 26.90),
    Model1ThresholdEntry::new(13000.00, 23.482, 29.03),
    Model1ThresholdEntry::new(13250.00, 23.546, 31.28),
    Model1ThresholdEntry::new(13500.00, 23.607, 33.67),
    Model1ThresholdEntry::new(13750.00, 23.666, 36.19),
    Model1ThresholdEntry::new(14000.00, 23.722, 38.86),
    Model1ThresholdEntry::new(14250.00, 23.775, 41.67),
    Model1ThresholdEntry::new(14500.00, 23.827, 44.63),
    Model1ThresholdEntry::new(14750.00, 23.876, 47.76),
    Model1ThresholdEntry::new(15000.00, 23.923, 51.04),
];

/// Table D.1e — Layer II, Fs = 44,1 kHz (130 rows, printed p.120).
///
/// Verbatim from
/// `docs/audio/mp3/annex-d-renders/Table-D.1e-threshold-in-quiet-LayerII-44k1Hz-p120.png`.
///
/// Rows 49.. print the same frequency / z / LTq values as the
/// matching Layer I table's rows 25.. (the two line grids coincide
/// there) — verified on the renders and pinned by a unit test.
pub const MODEL1_THRESHOLD_D1E: [Model1ThresholdEntry; 130] = [
    Model1ThresholdEntry::new(43.07, 0.425, 45.05),
    Model1ThresholdEntry::new(86.13, 0.850, 25.87),
    Model1ThresholdEntry::new(129.20, 1.273, 18.70),
    Model1ThresholdEntry::new(172.27, 1.694, 14.85),
    Model1ThresholdEntry::new(215.33, 2.112, 12.41),
    Model1ThresholdEntry::new(258.40, 2.525, 10.72),
    Model1ThresholdEntry::new(301.46, 2.934, 9.47),
    Model1ThresholdEntry::new(344.53, 3.337, 8.50),
    Model1ThresholdEntry::new(387.60, 3.733, 7.73),
    Model1ThresholdEntry::new(430.66, 4.124, 7.10),
    Model1ThresholdEntry::new(473.73, 4.507, 6.56),
    Model1ThresholdEntry::new(516.80, 4.882, 6.11),
    Model1ThresholdEntry::new(559.86, 5.249, 5.72),
    Model1ThresholdEntry::new(602.93, 5.608, 5.37),
    Model1ThresholdEntry::new(646.00, 5.959, 5.07),
    Model1ThresholdEntry::new(689.06, 6.301, 4.79),
    Model1ThresholdEntry::new(732.13, 6.634, 4.55),
    Model1ThresholdEntry::new(775.20, 6.959, 4.32),
    Model1ThresholdEntry::new(818.26, 7.274, 4.11),
    Model1ThresholdEntry::new(861.33, 7.581, 3.92),
    Model1ThresholdEntry::new(904.39, 7.879, 3.74),
    Model1ThresholdEntry::new(947.46, 8.169, 3.57),
    Model1ThresholdEntry::new(990.53, 8.450, 3.40),
    Model1ThresholdEntry::new(1033.59, 8.723, 3.25),
    Model1ThresholdEntry::new(1076.66, 8.987, 3.10),
    Model1ThresholdEntry::new(1119.73, 9.244, 2.95),
    Model1ThresholdEntry::new(1162.79, 9.493, 2.81),
    Model1ThresholdEntry::new(1205.86, 9.734, 2.67),
    Model1ThresholdEntry::new(1248.93, 9.968, 2.53),
    Model1ThresholdEntry::new(1291.99, 10.195, 2.39),
    Model1ThresholdEntry::new(1335.06, 10.416, 2.25),
    Model1ThresholdEntry::new(1378.13, 10.629, 2.11),
    Model1ThresholdEntry::new(1421.19, 10.836, 1.97),
    Model1ThresholdEntry::new(1464.26, 11.037, 1.83),
    Model1ThresholdEntry::new(1507.32, 11.232, 1.68),
    Model1ThresholdEntry::new(1550.39, 11.421, 1.53),
    Model1ThresholdEntry::new(1593.46, 11.605, 1.38),
    Model1ThresholdEntry::new(1636.52, 11.783, 1.23),
    Model1ThresholdEntry::new(1679.59, 11.957, 1.07),
    Model1ThresholdEntry::new(1722.66, 12.125, 0.90),
    Model1ThresholdEntry::new(1765.72, 12.289, 0.74),
    Model1ThresholdEntry::new(1808.79, 12.448, 0.56),
    Model1ThresholdEntry::new(1851.86, 12.603, 0.39),
    Model1ThresholdEntry::new(1894.92, 12.753, 0.21),
    Model1ThresholdEntry::new(1937.99, 12.900, 0.02),
    Model1ThresholdEntry::new(1981.05, 13.042, -0.17),
    Model1ThresholdEntry::new(2024.12, 13.181, -0.36),
    Model1ThresholdEntry::new(2067.19, 13.317, -0.56),
    Model1ThresholdEntry::new(2153.32, 13.578, -0.96),
    Model1ThresholdEntry::new(2239.45, 13.826, -1.38),
    Model1ThresholdEntry::new(2325.59, 14.062, -1.79),
    Model1ThresholdEntry::new(2411.72, 14.288, -2.21),
    Model1ThresholdEntry::new(2497.85, 14.504, -2.63),
    Model1ThresholdEntry::new(2583.98, 14.711, -3.03),
    Model1ThresholdEntry::new(2670.12, 14.909, -3.41),
    Model1ThresholdEntry::new(2756.25, 15.100, -3.77),
    Model1ThresholdEntry::new(2842.38, 15.284, -4.09),
    Model1ThresholdEntry::new(2928.52, 15.460, -4.37),
    Model1ThresholdEntry::new(3014.65, 15.631, -4.60),
    Model1ThresholdEntry::new(3100.78, 15.796, -4.78),
    Model1ThresholdEntry::new(3186.91, 15.955, -4.91),
    Model1ThresholdEntry::new(3273.05, 16.110, -4.97),
    Model1ThresholdEntry::new(3359.18, 16.260, -4.98),
    Model1ThresholdEntry::new(3445.31, 16.406, -4.92),
    Model1ThresholdEntry::new(3531.45, 16.547, -4.81),
    Model1ThresholdEntry::new(3617.58, 16.685, -4.65),
    Model1ThresholdEntry::new(3703.71, 16.820, -4.43),
    Model1ThresholdEntry::new(3789.84, 16.951, -4.17),
    Model1ThresholdEntry::new(3875.98, 17.079, -3.87),
    Model1ThresholdEntry::new(3962.11, 17.205, -3.54),
    Model1ThresholdEntry::new(4048.24, 17.327, -3.19),
    Model1ThresholdEntry::new(4134.38, 17.447, -2.82),
    Model1ThresholdEntry::new(4306.64, 17.680, -2.06),
    Model1ThresholdEntry::new(4478.91, 17.905, -1.32),
    Model1ThresholdEntry::new(4651.17, 18.121, -0.64),
    Model1ThresholdEntry::new(4823.44, 18.331, -0.04),
    Model1ThresholdEntry::new(4995.70, 18.534, 0.47),
    Model1ThresholdEntry::new(5167.97, 18.731, 0.89),
    Model1ThresholdEntry::new(5340.23, 18.922, 1.23),
    Model1ThresholdEntry::new(5512.50, 19.108, 1.51),
    Model1ThresholdEntry::new(5684.77, 19.289, 1.74),
    Model1ThresholdEntry::new(5857.03, 19.464, 1.93),
    Model1ThresholdEntry::new(6029.30, 19.635, 2.11),
    Model1ThresholdEntry::new(6201.56, 19.801, 2.28),
    Model1ThresholdEntry::new(6373.83, 19.963, 2.46),
    Model1ThresholdEntry::new(6546.09, 20.120, 2.63),
    Model1ThresholdEntry::new(6718.36, 20.273, 2.82),
    Model1ThresholdEntry::new(6890.63, 20.421, 3.03),
    Model1ThresholdEntry::new(7062.89, 20.565, 3.25),
    Model1ThresholdEntry::new(7235.16, 20.705, 3.49),
    Model1ThresholdEntry::new(7407.42, 20.840, 3.74),
    Model1ThresholdEntry::new(7579.69, 20.972, 4.02),
    Model1ThresholdEntry::new(7751.95, 21.099, 4.32),
    Model1ThresholdEntry::new(7924.22, 21.222, 4.64),
    Model1ThresholdEntry::new(8096.48, 21.342, 4.98),
    Model1ThresholdEntry::new(8268.75, 21.457, 5.35),
    Model1ThresholdEntry::new(8613.28, 21.677, 6.15),
    Model1ThresholdEntry::new(8957.81, 21.882, 7.07),
    Model1ThresholdEntry::new(9302.34, 22.074, 8.10),
    Model1ThresholdEntry::new(9646.88, 22.253, 9.25),
    Model1ThresholdEntry::new(9991.41, 22.420, 10.54),
    Model1ThresholdEntry::new(10335.94, 22.576, 11.97),
    Model1ThresholdEntry::new(10680.47, 22.721, 13.56),
    Model1ThresholdEntry::new(11025.00, 22.857, 15.31),
    Model1ThresholdEntry::new(11369.53, 22.984, 17.23),
    Model1ThresholdEntry::new(11714.06, 23.102, 19.34),
    Model1ThresholdEntry::new(12058.59, 23.213, 21.64),
    Model1ThresholdEntry::new(12403.13, 23.317, 24.15),
    Model1ThresholdEntry::new(12747.66, 23.415, 26.88),
    Model1ThresholdEntry::new(13092.19, 23.506, 29.84),
    Model1ThresholdEntry::new(13436.72, 23.592, 33.05),
    Model1ThresholdEntry::new(13781.25, 23.673, 36.52),
    Model1ThresholdEntry::new(14125.78, 23.749, 40.25),
    Model1ThresholdEntry::new(14470.31, 23.821, 44.27),
    Model1ThresholdEntry::new(14814.84, 23.888, 48.59),
    Model1ThresholdEntry::new(15159.38, 23.952, 53.22),
    Model1ThresholdEntry::new(15503.91, 24.013, 58.18),
    Model1ThresholdEntry::new(15848.44, 24.070, 63.49),
    Model1ThresholdEntry::new(16192.97, 24.125, 68.00),
    Model1ThresholdEntry::new(16537.50, 24.176, 68.00),
    Model1ThresholdEntry::new(16882.03, 24.225, 68.00),
    Model1ThresholdEntry::new(17226.56, 24.271, 68.00),
    Model1ThresholdEntry::new(17571.09, 24.316, 68.00),
    Model1ThresholdEntry::new(17915.63, 24.358, 68.00),
    Model1ThresholdEntry::new(18260.16, 24.398, 68.00),
    Model1ThresholdEntry::new(18604.69, 24.436, 68.00),
    Model1ThresholdEntry::new(18949.22, 24.473, 68.00),
    Model1ThresholdEntry::new(19293.75, 24.508, 68.00),
    Model1ThresholdEntry::new(19638.28, 24.542, 68.00),
    Model1ThresholdEntry::new(19982.81, 24.574, 68.00),
];

/// Table D.1f — Layer II, Fs = 48 kHz (126 rows, printed p.121).
///
/// Verbatim from
/// `docs/audio/mp3/annex-d-renders/Table-D.1f-threshold-in-quiet-LayerII-48kHz-p121.png`.
///
/// Rows 49.. print the same frequency / z / LTq values as the
/// matching Layer I table's rows 25.. (the two line grids coincide
/// there) — verified on the renders and pinned by a unit test.
pub const MODEL1_THRESHOLD_D1F: [Model1ThresholdEntry; 126] = [
    Model1ThresholdEntry::new(46.88, 0.463, 42.10),
    Model1ThresholdEntry::new(93.75, 0.925, 24.17),
    Model1ThresholdEntry::new(140.63, 1.385, 17.47),
    Model1ThresholdEntry::new(187.50, 1.842, 13.87),
    Model1ThresholdEntry::new(234.38, 2.295, 11.60),
    Model1ThresholdEntry::new(281.25, 2.742, 10.01),
    Model1ThresholdEntry::new(328.13, 3.184, 8.84),
    Model1ThresholdEntry::new(375.00, 3.618, 7.94),
    Model1ThresholdEntry::new(421.88, 4.045, 7.22),
    Model1ThresholdEntry::new(468.75, 4.463, 6.62),
    Model1ThresholdEntry::new(515.63, 4.872, 6.12),
    Model1ThresholdEntry::new(562.50, 5.272, 5.70),
    Model1ThresholdEntry::new(609.38, 5.661, 5.33),
    Model1ThresholdEntry::new(656.25, 6.041, 5.00),
    Model1ThresholdEntry::new(703.13, 6.411, 4.71),
    Model1ThresholdEntry::new(750.00, 6.770, 4.45),
    Model1ThresholdEntry::new(796.88, 7.119, 4.21),
    Model1ThresholdEntry::new(843.75, 7.457, 4.00),
    Model1ThresholdEntry::new(890.63, 7.785, 3.79),
    Model1ThresholdEntry::new(937.50, 8.103, 3.61),
    Model1ThresholdEntry::new(984.38, 8.410, 3.43),
    Model1ThresholdEntry::new(1031.25, 8.708, 3.26),
    Model1ThresholdEntry::new(1078.13, 8.996, 3.09),
    Model1ThresholdEntry::new(1125.00, 9.275, 2.93),
    Model1ThresholdEntry::new(1171.88, 9.544, 2.78),
    Model1ThresholdEntry::new(1218.75, 9.805, 2.63),
    Model1ThresholdEntry::new(1265.63, 10.057, 2.47),
    Model1ThresholdEntry::new(1312.50, 10.301, 2.32),
    Model1ThresholdEntry::new(1359.38, 10.537, 2.17),
    Model1ThresholdEntry::new(1406.25, 10.765, 2.02),
    Model1ThresholdEntry::new(1453.13, 10.986, 1.86),
    Model1ThresholdEntry::new(1500.00, 11.199, 1.71),
    Model1ThresholdEntry::new(1546.88, 11.406, 1.55),
    Model1ThresholdEntry::new(1593.75, 11.606, 1.38),
    Model1ThresholdEntry::new(1640.63, 11.800, 1.21),
    Model1ThresholdEntry::new(1687.50, 11.988, 1.04),
    Model1ThresholdEntry::new(1734.38, 12.170, 0.86),
    Model1ThresholdEntry::new(1781.25, 12.347, 0.67),
    Model1ThresholdEntry::new(1828.13, 12.518, 0.49),
    Model1ThresholdEntry::new(1875.00, 12.684, 0.29),
    Model1ThresholdEntry::new(1921.88, 12.845, 0.09),
    Model1ThresholdEntry::new(1968.75, 13.002, -0.11),
    Model1ThresholdEntry::new(2015.63, 13.154, -0.32),
    Model1ThresholdEntry::new(2062.50, 13.302, -0.54),
    Model1ThresholdEntry::new(2109.38, 13.446, -0.75),
    Model1ThresholdEntry::new(2156.25, 13.586, -0.97),
    Model1ThresholdEntry::new(2203.13, 13.723, -1.20),
    Model1ThresholdEntry::new(2250.00, 13.855, -1.43),
    Model1ThresholdEntry::new(2343.75, 14.111, -1.88),
    Model1ThresholdEntry::new(2437.50, 14.354, -2.34),
    Model1ThresholdEntry::new(2531.25, 14.585, -2.79),
    Model1ThresholdEntry::new(2625.00, 14.807, -3.22),
    Model1ThresholdEntry::new(2718.75, 15.018, -3.62),
    Model1ThresholdEntry::new(2812.50, 15.221, -3.98),
    Model1ThresholdEntry::new(2906.25, 15.415, -4.30),
    Model1ThresholdEntry::new(3000.00, 15.602, -4.57),
    Model1ThresholdEntry::new(3093.75, 15.783, -4.77),
    Model1ThresholdEntry::new(3187.50, 15.956, -4.91),
    Model1ThresholdEntry::new(3281.25, 16.124, -4.98),
    Model1ThresholdEntry::new(3375.00, 16.287, -4.97),
    Model1ThresholdEntry::new(3468.75, 16.445, -4.90),
    Model1ThresholdEntry::new(3562.50, 16.598, -4.76),
    Model1ThresholdEntry::new(3656.25, 16.746, -4.55),
    Model1ThresholdEntry::new(3750.00, 16.891, -4.29),
    Model1ThresholdEntry::new(3843.75, 17.032, -3.99),
    Model1ThresholdEntry::new(3937.50, 17.169, -3.64),
    Model1ThresholdEntry::new(4031.25, 17.303, -3.26),
    Model1ThresholdEntry::new(4125.00, 17.434, -2.86),
    Model1ThresholdEntry::new(4218.75, 17.563, -2.45),
    Model1ThresholdEntry::new(4312.50, 17.688, -2.04),
    Model1ThresholdEntry::new(4406.25, 17.811, -1.63),
    Model1ThresholdEntry::new(4500.00, 17.932, -1.24),
    Model1ThresholdEntry::new(4687.50, 18.166, -0.51),
    Model1ThresholdEntry::new(4875.00, 18.392, 0.12),
    Model1ThresholdEntry::new(5062.50, 18.611, 0.64),
    Model1ThresholdEntry::new(5250.00, 18.823, 1.06),
    Model1ThresholdEntry::new(5437.50, 19.028, 1.39),
    Model1ThresholdEntry::new(5625.00, 19.226, 1.66),
    Model1ThresholdEntry::new(5812.50, 19.419, 1.88),
    Model1ThresholdEntry::new(6000.00, 19.606, 2.08),
    Model1ThresholdEntry::new(6187.50, 19.788, 2.27),
    Model1ThresholdEntry::new(6375.00, 19.964, 2.46),
    Model1ThresholdEntry::new(6562.50, 20.135, 2.65),
    Model1ThresholdEntry::new(6750.00, 20.300, 2.86),
    Model1ThresholdEntry::new(6937.50, 20.461, 3.09),
    Model1ThresholdEntry::new(7125.00, 20.616, 3.33),
    Model1ThresholdEntry::new(7312.50, 20.766, 3.60),
    Model1ThresholdEntry::new(7500.00, 20.912, 3.89),
    Model1ThresholdEntry::new(7687.50, 21.052, 4.20),
    Model1ThresholdEntry::new(7875.00, 21.188, 4.54),
    Model1ThresholdEntry::new(8062.50, 21.318, 4.91),
    Model1ThresholdEntry::new(8250.00, 21.445, 5.31),
    Model1ThresholdEntry::new(8437.50, 21.567, 5.73),
    Model1ThresholdEntry::new(8625.00, 21.684, 6.18),
    Model1ThresholdEntry::new(8812.50, 21.797, 6.67),
    Model1ThresholdEntry::new(9000.00, 21.906, 7.19),
    Model1ThresholdEntry::new(9375.00, 22.113, 8.33),
    Model1ThresholdEntry::new(9750.00, 22.304, 9.63),
    Model1ThresholdEntry::new(10125.00, 22.482, 11.08),
    Model1ThresholdEntry::new(10500.00, 22.646, 12.71),
    Model1ThresholdEntry::new(10875.00, 22.799, 14.53),
    Model1ThresholdEntry::new(11250.00, 22.941, 16.54),
    Model1ThresholdEntry::new(11625.00, 23.072, 18.77),
    Model1ThresholdEntry::new(12000.00, 23.195, 21.23),
    Model1ThresholdEntry::new(12375.00, 23.309, 23.94),
    Model1ThresholdEntry::new(12750.00, 23.415, 26.90),
    Model1ThresholdEntry::new(13125.00, 23.515, 30.14),
    Model1ThresholdEntry::new(13500.00, 23.607, 33.67),
    Model1ThresholdEntry::new(13875.00, 23.694, 37.51),
    Model1ThresholdEntry::new(14250.00, 23.775, 41.67),
    Model1ThresholdEntry::new(14625.00, 23.852, 46.17),
    Model1ThresholdEntry::new(15000.00, 23.923, 51.04),
    Model1ThresholdEntry::new(15375.00, 23.991, 56.29),
    Model1ThresholdEntry::new(15750.00, 24.054, 61.94),
    Model1ThresholdEntry::new(16125.00, 24.114, 68.00),
    Model1ThresholdEntry::new(16500.00, 24.171, 68.00),
    Model1ThresholdEntry::new(16875.00, 24.224, 68.00),
    Model1ThresholdEntry::new(17250.00, 24.275, 68.00),
    Model1ThresholdEntry::new(17625.00, 24.322, 68.00),
    Model1ThresholdEntry::new(18000.00, 24.368, 68.00),
    Model1ThresholdEntry::new(18375.00, 24.411, 68.00),
    Model1ThresholdEntry::new(18750.00, 24.452, 68.00),
    Model1ThresholdEntry::new(19125.00, 24.491, 68.00),
    Model1ThresholdEntry::new(19500.00, 24.528, 68.00),
    Model1ThresholdEntry::new(19875.00, 24.564, 68.00),
    Model1ThresholdEntry::new(20250.00, 24.597, 68.00),
];

/// Return the verbatim Annex D Table D.1 slice for `(layer, fs)`.
/// Returns `None` for Layer III under the same convention as
/// [`critical_band_boundaries`]: Annex D defines its tables for
/// Layer I and Layer II only, and a Layer III caller (clause
/// C.1.5.3.2.1 re-uses the Layer I/II model) passes the matching
/// Layer explicitly.
#[inline]
#[must_use]
pub fn model1_threshold_table(
    layer: crate::frame::Layer,
    fs: AnnexDSamplingRate,
) -> Option<&'static [Model1ThresholdEntry]> {
    use crate::frame::Layer;
    match (layer, fs) {
        (Layer::LayerI, AnnexDSamplingRate::Hz32000) => Some(&MODEL1_THRESHOLD_D1A),
        (Layer::LayerI, AnnexDSamplingRate::Hz44100) => Some(&MODEL1_THRESHOLD_D1B),
        (Layer::LayerI, AnnexDSamplingRate::Hz48000) => Some(&MODEL1_THRESHOLD_D1C),
        (Layer::LayerII, AnnexDSamplingRate::Hz32000) => Some(&MODEL1_THRESHOLD_D1D),
        (Layer::LayerII, AnnexDSamplingRate::Hz44100) => Some(&MODEL1_THRESHOLD_D1E),
        (Layer::LayerII, AnnexDSamplingRate::Hz48000) => Some(&MODEL1_THRESHOLD_D1F),
        (Layer::LayerIII, _) => None,
    }
}

/// Map a 1-based Table D.1 `Index Number i` to the raw FFT-line index
/// (the step-77 spectrum index `k`) the row sits on, per the table's
/// printed frequency column: rows 1..=48 are lines 1..=48, rows
/// 49..=72 are every 2nd line (50..=96), rows 73..=96 every 4th line
/// (100..=192; Layer I tables continue this region to their end), and
/// Layer II rows 97.. every 8th line (200..). Returns `None` for
/// `index == 0`, an index past the table's last row, or a Layer III
/// dispatch.
#[inline]
#[must_use]
pub fn model1_d1_line_for_index(
    layer: crate::frame::Layer,
    fs: AnnexDSamplingRate,
    index: u16,
) -> Option<u16> {
    use crate::frame::Layer;
    let table = model1_threshold_table(layer, fs)?;
    if index == 0 || (index as usize) > table.len() {
        return None;
    }
    Some(match index {
        1..=48 => index,
        49..=72 => 2 * index - 48,
        // Layer I stays in the 4-line region to its last row; Layer II
        // switches to the 8-line region at row 97.
        _ if matches!(layer, Layer::LayerI) || index <= 96 => 4 * index - 192,
        _ => 8 * index - 576,
    })
}

/// Map a raw FFT-line index `k` (0-based step-77 spectrum index) to
/// the 1-based Table D.1 `Index Number i` of the **nearest** tabulated
/// line. In the subsampled regions an unlisted line maps to the
/// closest listed neighbour; an exact mid-point tie resolves to the
/// **lower** index (the spec subsamples without prescribing a rounding
/// rule for off-grid lines — the tie-down choice is documented here
/// and pinned by a unit test; the z error of either choice is bounded
/// by half the local Bark step). Returns `None` for `k == 0` (DC; the
/// tables are 1-based), for `k` above the table's last tabulated line
/// (no extrapolation past the audio band the spec tabulates), and for
/// a Layer III dispatch.
#[must_use]
pub fn model1_d1_index_for_line(
    layer: crate::frame::Layer,
    fs: AnnexDSamplingRate,
    k: u16,
) -> Option<u16> {
    let table = model1_threshold_table(layer, fs)?;
    let len = table.len() as u16;
    let line = |i: u16| model1_d1_line_for_index(layer, fs, i).expect("index in 1..=len");
    if k == 0 || k > line(len) {
        return None;
    }
    // Binary search for the greatest index whose line is <= k (exists:
    // line(1) = 1 <= k).
    let (mut lo, mut hi) = (1u16, len);
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if line(mid) <= k {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    // Nearest of floor / ceil; ties resolve down.
    if lo < len && (line(lo + 1) - k) < (k - line(lo)) {
        lo += 1;
    }
    Some(lo)
}

/// Convenience lookup: the Table D.1 row nearest to the raw FFT line
/// `k`, under the [`model1_d1_index_for_line`] mapping (same `None`
/// conditions).
#[inline]
#[must_use]
pub fn model1_d1_entry_for_line(
    layer: crate::frame::Layer,
    fs: AnnexDSamplingRate,
    k: u16,
) -> Option<&'static Model1ThresholdEntry> {
    let table = model1_threshold_table(layer, fs)?;
    let index = model1_d1_index_for_line(layer, fs, k)?;
    Some(&table[index as usize - 1])
}

/// Lift one §D.1 Step 4 component (line index `k` + SPL + kind, r277)
/// into the Bark-domain [`Masker`] carrier consumed by the Step 5
/// primitives and the Step 6/7 threshold evaluators: the masker's
/// `z_bark` is the `Crit.Band Rate` column of the Table D.1 row
/// nearest to `k`, and its SPL and tonal/non-tonal kind pass through
/// unchanged. Returns `None` when `k` has no Table D.1 row (DC, or
/// above the table's last tabulated line) or for a Layer III dispatch.
#[inline]
#[must_use]
pub fn model1_masker_from_component(
    component: &Model1Step4Component,
    layer: crate::frame::Layer,
    fs: AnnexDSamplingRate,
) -> Option<Masker> {
    let entry = model1_d1_entry_for_line(layer, fs, component.k)?;
    Some(Masker {
        kind: component.kind,
        z_bark: entry.z_bark,
        spl_db: component.spl_db,
    })
}

/// §D.1 Step 5 end-to-end sieve over the Step 4 component lists
/// (Phase 2 step 80 / r278): lifts every tonal and non-tonal component
/// onto the Table D.1 Bark grid ([`model1_masker_from_component`]),
/// applies the Step 5(a) threshold-in-quiet screen against the same
/// row's `Absolute Thresh.` column (`X_tm(k) >= LTq(k)` /
/// `X_nm(k) >= LTq(k)`, via [`masker_above_threshold_in_quiet`]), and
/// collapses tonal clusters within the 0,5-Bark sliding window
/// ([`decimate_tonal_within_half_bark`], Step 5(b)). The returned
/// maskers are ready for the Step 6/7 evaluators
/// ([`individual_masking_threshold_db`] /
/// [`global_masking_threshold_db`]).
///
/// Components whose line lies above the table's last tabulated line
/// are dropped: the spec tabulates neither a Bark coordinate nor an
/// LTq there, and the Tables D.2 critical bands (whose top line equals
/// the Table D.1 top line) end at the same point, so such components
/// sit outside the audio band the model evaluates. Output order is
/// the tonal list followed by the non-tonal list (each in input
/// order), the order Step 5(b) preserves.
///
/// Returns `None` only for a Layer III dispatch (no Annex D tables).
///
/// **Determinism.** Pure function of the component lists.
///
/// Provenance: composition of the Step 5(a)/(b) primitives (spec text
/// printed p.112) with the Tables D.1 transcription above; no
/// external implementation was consulted.
#[must_use]
pub fn model1_step5_components(
    tonal: &[Model1Step4Component],
    non_tonal: &[Model1Step4Component],
    layer: crate::frame::Layer,
    fs: AnnexDSamplingRate,
) -> Option<Vec<Masker>> {
    // Distinguish "no table" (None) from "components dropped" (Some).
    model1_threshold_table(layer, fs)?;
    let screened: Vec<Masker> = tonal
        .iter()
        .chain(non_tonal.iter())
        .filter_map(|c| {
            let entry = model1_d1_entry_for_line(layer, fs, c.k)?;
            let masker = model1_masker_from_component(c, layer, fs)?;
            masker_above_threshold_in_quiet(&masker, entry.ltq_db).then_some(masker)
        })
        .collect();
    Some(decimate_tonal_within_half_bark(&screened))
}

// =====================================================================
// Annex D Model 2 — Tables D.3a–c "Calculation partition table" and
// Tables D.4a–c "Absolute threshold table" (Phase 2 step 82 / r280).
//
// Spec context (ISO/IEC 11172-3:1993 Annex D clause D.2; tables
// printed pp.133–138 / PDF pp.139–144). Transcribed verbatim from the
// staged page renders
// `docs/audio/mp3/annex-d-renders/Table-D.3{a,b,c}-calc-partition-*.png`
// and `Table-D.4{a,b,c}-absolute-threshold-*.png` (the docs extracts
// file marks these renders as the authoritative reading; its prose
// "32 kHz has 63 partitions" note is an erratum — the printed D.3a
// ends at Index 49 with ωhigh = 513, i.e. full coverage of the
// 1024-point-FFT half-spectrum, and likewise D.3b at 57 / D.3c at 58).
//
// Tables D.3 (one per sampling rate, common to all Layers) define the
// Model 2 *calculation partitions*: for each 1-based partition Index,
// the first / last 1-based FFT line (ωlow / ωhigh), the partition's
// median Bark value `bval` (the spreading-function coordinate used by
// the §D.2.4 step f) convolution above), the minimum
// masking-spread value `minval` [dB] and the tone-masking-noise
// offset `TMN` [dB] (both consumed by later §D.2.4 steps).
//
// Tables D.4 (one per sampling rate) tabulate the Model 2 absolute
// threshold (threshold in quiet) per FFT-line *range* — columns
// `index [line] lower / higher` and `absthr [dB]`, with the printed
// page note: "A value of 0 dB represents a level in the absolute
// threshold calculation of 96 dB below the energy of a sine wave of
// amplitude +-32 760."
//
// Printed-table quirks kept verbatim (each pinned by a unit test):
//   - D.4a prints the row pair `57 | 57` followed by `59 | 60`: FFT
//     line 58 is not covered by any printed row. The row's absthr
//     (0,55 dB) equals Table D.1d's LTq at *line 58* (1 812,50 Hz),
//     so the `higher = 57` cell is almost certainly a misprint for
//     58, but the transcription keeps the printed value and
//     [`model2_absthr_for_line`] returns `None` for line 58 at
//     32 kHz.
//   - D.4c prints a single 4-line group `329 | 332` inside the
//     otherwise-8-line tail region (…, `321 | 328`, `329 | 332`,
//     `333 | 340`, …); coverage stays contiguous but the tail rows
//     after it sit 4 lines off the Table D.1f 8-line grid.
//   - D.4a's last row (`473 | 480`) prints `51,03` dB where Tables
//     D.1a/D.1d print `51,04` at the same 15 000 Hz line — a
//     rounding inconsistency in the printed spec (same shape as the
//     documented D.1/D.2 0,001-Bark print differences); both verbatim
//     prints are kept.
//
// Redundancy used as a transcription cross-check (pinned by a unit
// test): wherever a D.4 row's `higher` line is also tabulated by the
// Layer II Table D.1 at the same sampling rate (D.1d/e/f share the
// 1024-point FFT line grid), the row's `absthr` equals that row's
// `Absolute Thresh.` column. Exceptions, all printed-spec
// inconsistencies pinned by the test: the D.4a 51,03 / 51,04 print
// difference above, plus a systematic 44,1 kHz divergence — 14
// shared lines print exactly 0,01 dB lower in D.4b than in D.1e
// (both sides re-verified on the renders, e.g. lines 51..=52 print
// -1,37 in D.4b vs -1,38 in D.1e row 50), and D.4b's top-of-band
// saturation plateau prints 69,13 dB where D.1e (and D.4c / D.1f)
// clamp at 68,00 dB.
//
// Provenance: only the six staged PNG renders named above (and the
// docs extracts file for column semantics); no external
// implementation was consulted.
// =====================================================================

/// One row of Annex D Table D.3 (calculation partition table). The
/// 1-based `Index` column is implicit (slice position + 1).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Model2PartitionEntry {
    /// `ωlow` column — first 1-based FFT line of the partition.
    pub wlow: u16,
    /// `ωhigh` column — last 1-based FFT line of the partition.
    pub whigh: u16,
    /// `bval` column — median Bark value of the partition (the
    /// spreading-function coordinate of [`model2_step_f_spread`]).
    pub bval: f64,
    /// `minval` column — minimum masking-spread value in dB.
    pub minval_db: f64,
    /// `TMN` column — tone-masking-noise offset in dB.
    pub tmn_db: f64,
}

impl Model2PartitionEntry {
    /// Construct a table row at compile time.
    #[inline]
    #[must_use]
    pub const fn new(wlow: u16, whigh: u16, bval: f64, minval_db: f64, tmn_db: f64) -> Self {
        Self {
            wlow,
            whigh,
            bval,
            minval_db,
            tmn_db,
        }
    }
}

/// Table D.3a — calculation partition table, Fs = 32 kHz (49
/// partitions, printed p.133).
///
/// Verbatim from
/// `docs/audio/mp3/annex-d-renders/Table-D.3a-calc-partition-32kHz-p133.png`.
pub const MODEL2_PARTITION_D3A: [Model2PartitionEntry; 49] = [
    Model2PartitionEntry::new(1, 1, 0.00, 0.0, 24.5),
    Model2PartitionEntry::new(2, 4, 0.63, 0.0, 24.5),
    Model2PartitionEntry::new(5, 7, 1.56, 20.0, 24.5),
    Model2PartitionEntry::new(8, 10, 2.50, 20.0, 24.5),
    Model2PartitionEntry::new(11, 13, 3.44, 20.0, 24.5),
    Model2PartitionEntry::new(14, 16, 4.34, 20.0, 24.5),
    Model2PartitionEntry::new(17, 19, 5.17, 20.0, 24.5),
    Model2PartitionEntry::new(20, 22, 5.94, 20.0, 24.5),
    Model2PartitionEntry::new(23, 25, 6.63, 17.0, 24.5),
    Model2PartitionEntry::new(26, 28, 7.28, 15.0, 24.5),
    Model2PartitionEntry::new(29, 31, 7.90, 15.0, 24.5),
    Model2PartitionEntry::new(32, 34, 8.50, 10.0, 24.5),
    Model2PartitionEntry::new(35, 37, 9.06, 7.0, 24.5),
    Model2PartitionEntry::new(38, 41, 9.65, 7.0, 24.5),
    Model2PartitionEntry::new(42, 45, 10.28, 4.4, 24.8),
    Model2PartitionEntry::new(46, 49, 10.87, 4.4, 25.4),
    Model2PartitionEntry::new(50, 53, 11.41, 4.5, 25.9),
    Model2PartitionEntry::new(54, 57, 11.92, 4.5, 26.4),
    Model2PartitionEntry::new(58, 61, 12.39, 4.5, 26.9),
    Model2PartitionEntry::new(62, 65, 12.83, 4.5, 27.3),
    Model2PartitionEntry::new(66, 70, 13.29, 4.5, 27.8),
    Model2PartitionEntry::new(71, 75, 13.78, 4.5, 28.3),
    Model2PartitionEntry::new(76, 81, 14.27, 4.5, 28.8),
    Model2PartitionEntry::new(82, 87, 14.76, 4.5, 29.3),
    Model2PartitionEntry::new(88, 93, 15.22, 4.5, 29.7),
    Model2PartitionEntry::new(94, 99, 15.63, 4.5, 30.1),
    Model2PartitionEntry::new(100, 106, 16.06, 4.5, 30.6),
    Model2PartitionEntry::new(107, 113, 16.47, 4.5, 31.0),
    Model2PartitionEntry::new(114, 120, 16.86, 4.5, 31.4),
    Model2PartitionEntry::new(121, 129, 17.25, 4.5, 31.8),
    Model2PartitionEntry::new(130, 138, 17.65, 4.5, 32.2),
    Model2PartitionEntry::new(139, 148, 18.05, 4.5, 32.5),
    Model2PartitionEntry::new(149, 159, 18.42, 4.5, 32.9),
    Model2PartitionEntry::new(160, 170, 18.81, 4.5, 33.3),
    Model2PartitionEntry::new(171, 183, 19.18, 4.5, 33.7),
    Model2PartitionEntry::new(184, 196, 19.55, 4.5, 34.1),
    Model2PartitionEntry::new(197, 210, 19.93, 4.5, 34.4),
    Model2PartitionEntry::new(211, 225, 20.29, 4.5, 34.8),
    Model2PartitionEntry::new(226, 240, 20.65, 4.5, 35.2),
    Model2PartitionEntry::new(241, 258, 21.02, 4.5, 35.5),
    Model2PartitionEntry::new(259, 279, 21.38, 4.5, 35.9),
    Model2PartitionEntry::new(280, 300, 21.74, 4.5, 36.2),
    Model2PartitionEntry::new(301, 326, 22.10, 4.5, 36.6),
    Model2PartitionEntry::new(327, 354, 22.44, 4.5, 36.9),
    Model2PartitionEntry::new(355, 382, 22.79, 4.5, 37.3),
    Model2PartitionEntry::new(383, 420, 23.14, 4.5, 37.6),
    Model2PartitionEntry::new(421, 458, 23.49, 4.5, 38.0),
    Model2PartitionEntry::new(459, 496, 23.83, 4.5, 38.3),
    Model2PartitionEntry::new(497, 513, 24.07, 4.5, 38.6),
];

/// Table D.3b — calculation partition table, Fs = 44,1 kHz (57
/// partitions, printed p.134).
///
/// Verbatim from
/// `docs/audio/mp3/annex-d-renders/Table-D.3b-calc-partition-44k1Hz-p134.png`.
pub const MODEL2_PARTITION_D3B: [Model2PartitionEntry; 57] = [
    Model2PartitionEntry::new(1, 1, 0.00, 0.0, 24.5),
    Model2PartitionEntry::new(2, 2, 0.43, 0.0, 24.5),
    Model2PartitionEntry::new(3, 3, 0.86, 0.0, 24.5),
    Model2PartitionEntry::new(4, 4, 1.29, 20.0, 24.5),
    Model2PartitionEntry::new(5, 5, 1.72, 20.0, 24.5),
    Model2PartitionEntry::new(6, 6, 2.15, 20.0, 24.5),
    Model2PartitionEntry::new(7, 7, 2.58, 20.0, 24.5),
    Model2PartitionEntry::new(8, 8, 3.01, 20.0, 24.5),
    Model2PartitionEntry::new(9, 9, 3.45, 20.0, 24.5),
    Model2PartitionEntry::new(10, 10, 3.88, 20.0, 24.5),
    Model2PartitionEntry::new(11, 11, 4.28, 20.0, 24.5),
    Model2PartitionEntry::new(12, 12, 4.67, 20.0, 24.5),
    Model2PartitionEntry::new(13, 13, 5.06, 20.0, 24.5),
    Model2PartitionEntry::new(14, 14, 5.42, 20.0, 24.5),
    Model2PartitionEntry::new(15, 15, 5.77, 20.0, 24.5),
    Model2PartitionEntry::new(16, 16, 6.11, 17.0, 24.5),
    Model2PartitionEntry::new(17, 19, 6.73, 17.0, 24.5),
    Model2PartitionEntry::new(20, 22, 7.61, 15.0, 24.5),
    Model2PartitionEntry::new(23, 25, 8.44, 10.0, 24.5),
    Model2PartitionEntry::new(26, 28, 9.21, 7.0, 24.5),
    Model2PartitionEntry::new(29, 31, 9.88, 7.0, 24.5),
    Model2PartitionEntry::new(32, 34, 10.51, 4.4, 25.0),
    Model2PartitionEntry::new(35, 37, 11.11, 4.5, 25.6),
    Model2PartitionEntry::new(38, 40, 11.65, 4.5, 26.2),
    Model2PartitionEntry::new(41, 44, 12.24, 4.5, 26.7),
    Model2PartitionEntry::new(45, 48, 12.85, 4.5, 27.4),
    Model2PartitionEntry::new(49, 52, 13.41, 4.5, 27.9),
    Model2PartitionEntry::new(53, 56, 13.94, 4.5, 28.4),
    Model2PartitionEntry::new(57, 60, 14.42, 4.5, 28.9),
    Model2PartitionEntry::new(61, 64, 14.86, 4.5, 29.4),
    Model2PartitionEntry::new(65, 69, 15.32, 4.5, 29.8),
    Model2PartitionEntry::new(70, 74, 15.79, 4.5, 30.3),
    Model2PartitionEntry::new(75, 80, 16.26, 4.5, 30.8),
    Model2PartitionEntry::new(81, 86, 16.73, 4.5, 31.2),
    Model2PartitionEntry::new(87, 93, 17.19, 4.5, 31.7),
    Model2PartitionEntry::new(94, 100, 17.62, 4.5, 32.1),
    Model2PartitionEntry::new(101, 108, 18.05, 4.5, 32.5),
    Model2PartitionEntry::new(109, 116, 18.45, 4.5, 32.9),
    Model2PartitionEntry::new(117, 124, 18.83, 4.5, 33.3),
    Model2PartitionEntry::new(125, 134, 19.21, 4.5, 33.7),
    Model2PartitionEntry::new(135, 144, 19.60, 4.5, 34.1),
    Model2PartitionEntry::new(145, 155, 20.00, 4.5, 34.5),
    Model2PartitionEntry::new(156, 166, 20.38, 4.5, 34.9),
    Model2PartitionEntry::new(167, 177, 20.74, 4.5, 35.2),
    Model2PartitionEntry::new(178, 192, 21.12, 4.5, 35.6),
    Model2PartitionEntry::new(193, 207, 21.48, 4.5, 36.0),
    Model2PartitionEntry::new(208, 222, 21.84, 4.5, 36.3),
    Model2PartitionEntry::new(223, 243, 22.20, 4.5, 36.7),
    Model2PartitionEntry::new(244, 264, 22.56, 4.5, 37.1),
    Model2PartitionEntry::new(265, 286, 22.91, 4.5, 37.4),
    Model2PartitionEntry::new(287, 314, 23.26, 4.5, 37.8),
    Model2PartitionEntry::new(315, 342, 23.60, 4.5, 38.1),
    Model2PartitionEntry::new(343, 371, 23.95, 4.5, 38.4),
    Model2PartitionEntry::new(372, 401, 24.30, 4.5, 38.8),
    Model2PartitionEntry::new(402, 431, 24.65, 4.5, 39.1),
    Model2PartitionEntry::new(432, 469, 25.00, 4.5, 39.5),
    Model2PartitionEntry::new(470, 513, 25.33, 3.5, 39.8),
];

/// Table D.3c — calculation partition table, Fs = 48 kHz (58
/// partitions, printed p.135).
///
/// Verbatim from
/// `docs/audio/mp3/annex-d-renders/Table-D.3c-calc-partition-48kHz-p135.png`.
pub const MODEL2_PARTITION_D3C: [Model2PartitionEntry; 58] = [
    Model2PartitionEntry::new(1, 1, 0.00, 0.0, 24.5),
    Model2PartitionEntry::new(2, 2, 0.47, 0.0, 24.5),
    Model2PartitionEntry::new(3, 3, 0.94, 0.0, 24.5),
    Model2PartitionEntry::new(4, 4, 1.41, 20.0, 24.5),
    Model2PartitionEntry::new(5, 5, 1.88, 20.0, 24.5),
    Model2PartitionEntry::new(6, 6, 2.34, 20.0, 24.5),
    Model2PartitionEntry::new(7, 7, 2.81, 20.0, 24.5),
    Model2PartitionEntry::new(8, 8, 3.28, 20.0, 24.5),
    Model2PartitionEntry::new(9, 9, 3.75, 20.0, 24.5),
    Model2PartitionEntry::new(10, 10, 4.20, 20.0, 24.5),
    Model2PartitionEntry::new(11, 11, 4.63, 20.0, 24.5),
    Model2PartitionEntry::new(12, 12, 5.05, 20.0, 24.5),
    Model2PartitionEntry::new(13, 13, 5.44, 20.0, 24.5),
    Model2PartitionEntry::new(14, 14, 5.83, 20.0, 24.5),
    Model2PartitionEntry::new(15, 15, 6.19, 20.0, 24.5),
    Model2PartitionEntry::new(16, 16, 6.52, 17.0, 24.5),
    Model2PartitionEntry::new(17, 17, 6.86, 17.0, 24.5),
    Model2PartitionEntry::new(18, 20, 7.49, 15.0, 24.5),
    Model2PartitionEntry::new(21, 23, 8.40, 10.0, 24.5),
    Model2PartitionEntry::new(24, 26, 9.24, 7.0, 24.5),
    Model2PartitionEntry::new(27, 29, 9.97, 7.0, 24.5),
    Model2PartitionEntry::new(30, 32, 10.65, 4.4, 25.1),
    Model2PartitionEntry::new(33, 35, 11.28, 4.5, 25.8),
    Model2PartitionEntry::new(36, 38, 11.86, 4.5, 26.4),
    Model2PartitionEntry::new(39, 41, 12.39, 4.5, 26.9),
    Model2PartitionEntry::new(42, 45, 12.96, 4.5, 27.5),
    Model2PartitionEntry::new(46, 49, 13.56, 4.5, 28.1),
    Model2PartitionEntry::new(50, 53, 14.12, 4.5, 28.6),
    Model2PartitionEntry::new(54, 57, 14.62, 4.5, 29.1),
    Model2PartitionEntry::new(58, 62, 15.14, 4.5, 29.6),
    Model2PartitionEntry::new(63, 67, 15.67, 4.5, 30.2),
    Model2PartitionEntry::new(68, 72, 16.15, 4.5, 30.7),
    Model2PartitionEntry::new(73, 77, 16.58, 4.5, 31.1),
    Model2PartitionEntry::new(78, 83, 17.02, 4.5, 31.5),
    Model2PartitionEntry::new(84, 89, 17.44, 4.5, 31.9),
    Model2PartitionEntry::new(90, 95, 17.84, 4.5, 32.3),
    Model2PartitionEntry::new(96, 103, 18.24, 4.5, 32.7),
    Model2PartitionEntry::new(104, 111, 18.66, 4.5, 33.2),
    Model2PartitionEntry::new(112, 120, 19.07, 4.5, 33.6),
    Model2PartitionEntry::new(121, 129, 19.47, 4.5, 34.0),
    Model2PartitionEntry::new(130, 138, 19.85, 4.5, 34.3),
    Model2PartitionEntry::new(139, 149, 20.23, 4.5, 34.7),
    Model2PartitionEntry::new(150, 160, 20.63, 4.5, 35.1),
    Model2PartitionEntry::new(161, 173, 21.02, 4.5, 35.5),
    Model2PartitionEntry::new(174, 187, 21.40, 4.5, 35.9),
    Model2PartitionEntry::new(188, 201, 21.76, 4.5, 36.3),
    Model2PartitionEntry::new(202, 219, 22.12, 4.5, 36.6),
    Model2PartitionEntry::new(220, 238, 22.47, 4.5, 37.0),
    Model2PartitionEntry::new(239, 257, 22.83, 4.5, 37.3),
    Model2PartitionEntry::new(258, 283, 23.18, 4.5, 37.7),
    Model2PartitionEntry::new(284, 309, 23.53, 4.5, 38.0),
    Model2PartitionEntry::new(310, 335, 23.88, 4.5, 38.4),
    Model2PartitionEntry::new(336, 363, 24.23, 4.5, 38.7),
    Model2PartitionEntry::new(364, 391, 24.58, 4.5, 39.1),
    Model2PartitionEntry::new(392, 423, 24.93, 4.5, 39.4),
    Model2PartitionEntry::new(424, 465, 25.27, 4.5, 39.8),
    Model2PartitionEntry::new(466, 507, 25.61, 3.5, 40.1),
    Model2PartitionEntry::new(508, 513, 25.81, 3.5, 40.3),
];

/// Return the verbatim Annex D Table D.3 calculation-partition slice
/// for `fs`. Unlike the Model 1 tables, the Model 2 partition tables
/// are common to all Layers, so there is no Layer dimension.
#[inline]
#[must_use]
pub fn model2_partition_table(fs: AnnexDSamplingRate) -> &'static [Model2PartitionEntry] {
    match fs {
        AnnexDSamplingRate::Hz32000 => &MODEL2_PARTITION_D3A,
        AnnexDSamplingRate::Hz44100 => &MODEL2_PARTITION_D3B,
        AnnexDSamplingRate::Hz48000 => &MODEL2_PARTITION_D3C,
    }
}

/// The `bval` column of the Table D.3 partition table for `fs`, in
/// partition order — the spreading-function coordinate vector the
/// §D.2.4 step f) reductions ([`model2_step_f_spread`] /
/// [`model2_step_f_rnorm`]) take as `bval`.
#[must_use]
pub fn model2_bval(fs: AnnexDSamplingRate) -> Vec<f64> {
    model2_partition_table(fs).iter().map(|e| e.bval).collect()
}

/// Map a 1-based FFT line (1..=513) to the 1-based Table D.3
/// calculation-partition `Index` containing it. Returns `None` for
/// `line == 0` or `line > 513` (the tables cover the 1024-point-FFT
/// half-spectrum exactly: every table's first ωlow is 1 and last
/// ωhigh is 513, with contiguous coverage in between).
#[must_use]
pub fn model2_partition_index_for_line(fs: AnnexDSamplingRate, line: u16) -> Option<u16> {
    let table = model2_partition_table(fs);
    table
        .iter()
        .position(|e| e.wlow <= line && line <= e.whigh)
        .map(|p| p as u16 + 1)
}

/// One row of Annex D Table D.4 (absolute threshold table): the
/// threshold in quiet `absthr` for the 1-based FFT lines
/// `lower..=higher`.
///
/// Per the printed page note, "A value of 0 dB represents a level in
/// the absolute threshold calculation of 96 dB below the energy of a
/// sine wave of amplitude +-32 760."
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Model2AbsThrEntry {
    /// `index [line] lower` column — first 1-based FFT line.
    pub lower: u16,
    /// `index [line] higher` column — last 1-based FFT line.
    pub higher: u16,
    /// `absthr [dB]` column.
    pub absthr_db: f64,
}

impl Model2AbsThrEntry {
    /// Construct a table row at compile time.
    #[inline]
    #[must_use]
    pub const fn new(lower: u16, higher: u16, absthr_db: f64) -> Self {
        Self {
            lower,
            higher,
            absthr_db,
        }
    }
}

/// Table D.4a — absolute threshold table, Fs = 32 kHz (132 rows,
/// lines 1..=480 except the printed line-58 gap; printed p.136).
///
/// Verbatim from
/// `docs/audio/mp3/annex-d-renders/Table-D.4a-absolute-threshold-32kHz-p136.png`
/// — including the printed `57 | 57` row (see the section comment) and
/// the printed `51,03` final value.
// The 6,28 dB threshold at line 16 (500 Hz) is the spec's printed
// table value, not an approximation of a mathematical constant.
#[allow(clippy::approx_constant)]
pub const MODEL2_ABSTHR_D4A: [Model2AbsThrEntry; 132] = [
    Model2AbsThrEntry::new(1, 1, 58.23),
    Model2AbsThrEntry::new(2, 2, 33.44),
    Model2AbsThrEntry::new(3, 3, 24.17),
    Model2AbsThrEntry::new(4, 4, 19.20),
    Model2AbsThrEntry::new(5, 5, 16.05),
    Model2AbsThrEntry::new(6, 6, 13.87),
    Model2AbsThrEntry::new(7, 7, 12.26),
    Model2AbsThrEntry::new(8, 8, 11.01),
    Model2AbsThrEntry::new(9, 9, 10.01),
    Model2AbsThrEntry::new(10, 10, 9.20),
    Model2AbsThrEntry::new(11, 11, 8.52),
    Model2AbsThrEntry::new(12, 12, 7.94),
    Model2AbsThrEntry::new(13, 13, 7.44),
    Model2AbsThrEntry::new(14, 14, 7.00),
    Model2AbsThrEntry::new(15, 15, 6.62),
    Model2AbsThrEntry::new(16, 16, 6.28),
    Model2AbsThrEntry::new(17, 17, 5.97),
    Model2AbsThrEntry::new(18, 18, 5.70),
    Model2AbsThrEntry::new(19, 19, 5.44),
    Model2AbsThrEntry::new(20, 20, 5.21),
    Model2AbsThrEntry::new(21, 21, 5.00),
    Model2AbsThrEntry::new(22, 22, 4.80),
    Model2AbsThrEntry::new(23, 23, 4.62),
    Model2AbsThrEntry::new(24, 24, 4.45),
    Model2AbsThrEntry::new(25, 25, 4.29),
    Model2AbsThrEntry::new(26, 26, 4.14),
    Model2AbsThrEntry::new(27, 27, 4.00),
    Model2AbsThrEntry::new(28, 28, 3.86),
    Model2AbsThrEntry::new(29, 29, 3.73),
    Model2AbsThrEntry::new(30, 30, 3.61),
    Model2AbsThrEntry::new(31, 31, 3.49),
    Model2AbsThrEntry::new(32, 32, 3.37),
    Model2AbsThrEntry::new(33, 33, 3.26),
    Model2AbsThrEntry::new(34, 34, 3.15),
    Model2AbsThrEntry::new(35, 35, 3.04),
    Model2AbsThrEntry::new(36, 36, 2.93),
    Model2AbsThrEntry::new(37, 37, 2.83),
    Model2AbsThrEntry::new(38, 38, 2.73),
    Model2AbsThrEntry::new(39, 39, 2.63),
    Model2AbsThrEntry::new(40, 40, 2.53),
    Model2AbsThrEntry::new(41, 41, 2.42),
    Model2AbsThrEntry::new(42, 42, 2.32),
    Model2AbsThrEntry::new(43, 43, 2.22),
    Model2AbsThrEntry::new(44, 44, 2.12),
    Model2AbsThrEntry::new(45, 45, 2.02),
    Model2AbsThrEntry::new(46, 46, 1.92),
    Model2AbsThrEntry::new(47, 47, 1.81),
    Model2AbsThrEntry::new(48, 48, 1.71),
    Model2AbsThrEntry::new(49, 50, 1.49),
    Model2AbsThrEntry::new(51, 52, 1.27),
    Model2AbsThrEntry::new(53, 54, 1.04),
    Model2AbsThrEntry::new(55, 56, 0.80),
    Model2AbsThrEntry::new(57, 57, 0.55),
    Model2AbsThrEntry::new(59, 60, 0.29),
    Model2AbsThrEntry::new(61, 62, 0.02),
    Model2AbsThrEntry::new(63, 64, -0.25),
    Model2AbsThrEntry::new(65, 66, -0.54),
    Model2AbsThrEntry::new(67, 68, -0.83),
    Model2AbsThrEntry::new(69, 70, -1.12),
    Model2AbsThrEntry::new(71, 72, -1.43),
    Model2AbsThrEntry::new(73, 74, -1.73),
    Model2AbsThrEntry::new(75, 76, -2.04),
    Model2AbsThrEntry::new(77, 78, -2.34),
    Model2AbsThrEntry::new(79, 80, -2.64),
    Model2AbsThrEntry::new(81, 82, -2.93),
    Model2AbsThrEntry::new(83, 84, -3.22),
    Model2AbsThrEntry::new(85, 86, -3.49),
    Model2AbsThrEntry::new(87, 88, -3.74),
    Model2AbsThrEntry::new(89, 90, -3.98),
    Model2AbsThrEntry::new(91, 92, -4.20),
    Model2AbsThrEntry::new(93, 94, -4.40),
    Model2AbsThrEntry::new(95, 96, -4.57),
    Model2AbsThrEntry::new(97, 100, -4.82),
    Model2AbsThrEntry::new(101, 104, -4.96),
    Model2AbsThrEntry::new(105, 108, -4.97),
    Model2AbsThrEntry::new(109, 112, -4.86),
    Model2AbsThrEntry::new(113, 116, -4.63),
    Model2AbsThrEntry::new(117, 120, -4.29),
    Model2AbsThrEntry::new(121, 124, -3.87),
    Model2AbsThrEntry::new(125, 128, -3.39),
    Model2AbsThrEntry::new(129, 132, -2.86),
    Model2AbsThrEntry::new(133, 136, -2.31),
    Model2AbsThrEntry::new(137, 140, -1.77),
    Model2AbsThrEntry::new(141, 144, -1.24),
    Model2AbsThrEntry::new(145, 148, -0.74),
    Model2AbsThrEntry::new(149, 152, -0.29),
    Model2AbsThrEntry::new(153, 156, 0.12),
    Model2AbsThrEntry::new(157, 160, 0.48),
    Model2AbsThrEntry::new(161, 164, 0.79),
    Model2AbsThrEntry::new(165, 168, 1.06),
    Model2AbsThrEntry::new(169, 172, 1.29),
    Model2AbsThrEntry::new(173, 176, 1.49),
    Model2AbsThrEntry::new(177, 180, 1.66),
    Model2AbsThrEntry::new(181, 184, 1.81),
    Model2AbsThrEntry::new(185, 188, 1.95),
    Model2AbsThrEntry::new(189, 192, 2.08),
    Model2AbsThrEntry::new(193, 200, 2.33),
    Model2AbsThrEntry::new(201, 208, 2.59),
    Model2AbsThrEntry::new(209, 216, 2.86),
    Model2AbsThrEntry::new(217, 224, 3.17),
    Model2AbsThrEntry::new(225, 232, 3.51),
    Model2AbsThrEntry::new(233, 240, 3.89),
    Model2AbsThrEntry::new(241, 248, 4.31),
    Model2AbsThrEntry::new(249, 256, 4.79),
    Model2AbsThrEntry::new(257, 264, 5.31),
    Model2AbsThrEntry::new(265, 272, 5.88),
    Model2AbsThrEntry::new(273, 280, 6.50),
    Model2AbsThrEntry::new(281, 288, 7.19),
    Model2AbsThrEntry::new(289, 296, 7.93),
    Model2AbsThrEntry::new(297, 304, 8.75),
    Model2AbsThrEntry::new(305, 312, 9.63),
    Model2AbsThrEntry::new(313, 320, 10.58),
    Model2AbsThrEntry::new(321, 328, 11.60),
    Model2AbsThrEntry::new(329, 336, 12.71),
    Model2AbsThrEntry::new(337, 344, 13.90),
    Model2AbsThrEntry::new(345, 352, 15.18),
    Model2AbsThrEntry::new(353, 360, 16.54),
    Model2AbsThrEntry::new(361, 368, 18.01),
    Model2AbsThrEntry::new(369, 376, 19.57),
    Model2AbsThrEntry::new(377, 384, 21.23),
    Model2AbsThrEntry::new(385, 392, 23.01),
    Model2AbsThrEntry::new(393, 400, 24.90),
    Model2AbsThrEntry::new(401, 408, 26.90),
    Model2AbsThrEntry::new(409, 416, 29.03),
    Model2AbsThrEntry::new(417, 424, 31.28),
    Model2AbsThrEntry::new(425, 432, 33.67),
    Model2AbsThrEntry::new(433, 440, 36.19),
    Model2AbsThrEntry::new(441, 448, 38.86),
    Model2AbsThrEntry::new(449, 456, 41.67),
    Model2AbsThrEntry::new(457, 464, 44.63),
    Model2AbsThrEntry::new(465, 472, 47.76),
    Model2AbsThrEntry::new(473, 480, 51.03),
];

/// Table D.4b — absolute threshold table, Fs = 44,1 kHz (130 rows,
/// lines 1..=464; printed p.137).
///
/// Verbatim from
/// `docs/audio/mp3/annex-d-renders/Table-D.4b-absolute-threshold-44k1Hz-p137.png`.
pub const MODEL2_ABSTHR_D4B: [Model2AbsThrEntry; 130] = [
    Model2AbsThrEntry::new(1, 1, 45.05),
    Model2AbsThrEntry::new(2, 2, 25.87),
    Model2AbsThrEntry::new(3, 3, 18.70),
    Model2AbsThrEntry::new(4, 4, 14.85),
    Model2AbsThrEntry::new(5, 5, 12.41),
    Model2AbsThrEntry::new(6, 6, 10.72),
    Model2AbsThrEntry::new(7, 7, 9.47),
    Model2AbsThrEntry::new(8, 8, 8.50),
    Model2AbsThrEntry::new(9, 9, 7.73),
    Model2AbsThrEntry::new(10, 10, 7.10),
    Model2AbsThrEntry::new(11, 11, 6.56),
    Model2AbsThrEntry::new(12, 12, 6.11),
    Model2AbsThrEntry::new(13, 13, 5.72),
    Model2AbsThrEntry::new(14, 14, 5.37),
    Model2AbsThrEntry::new(15, 15, 5.07),
    Model2AbsThrEntry::new(16, 16, 4.79),
    Model2AbsThrEntry::new(17, 17, 4.55),
    Model2AbsThrEntry::new(18, 18, 4.32),
    Model2AbsThrEntry::new(19, 19, 4.11),
    Model2AbsThrEntry::new(20, 20, 3.92),
    Model2AbsThrEntry::new(21, 21, 3.74),
    Model2AbsThrEntry::new(22, 22, 3.57),
    Model2AbsThrEntry::new(23, 23, 3.40),
    Model2AbsThrEntry::new(24, 24, 3.25),
    Model2AbsThrEntry::new(25, 25, 3.10),
    Model2AbsThrEntry::new(26, 26, 2.95),
    Model2AbsThrEntry::new(27, 27, 2.81),
    Model2AbsThrEntry::new(28, 28, 2.67),
    Model2AbsThrEntry::new(29, 29, 2.53),
    Model2AbsThrEntry::new(30, 30, 2.39),
    Model2AbsThrEntry::new(31, 31, 2.25),
    Model2AbsThrEntry::new(32, 32, 2.11),
    Model2AbsThrEntry::new(33, 33, 1.97),
    Model2AbsThrEntry::new(34, 34, 1.83),
    Model2AbsThrEntry::new(35, 35, 1.68),
    Model2AbsThrEntry::new(36, 36, 1.53),
    Model2AbsThrEntry::new(37, 37, 1.38),
    Model2AbsThrEntry::new(38, 38, 1.23),
    Model2AbsThrEntry::new(39, 39, 1.07),
    Model2AbsThrEntry::new(40, 40, 0.90),
    Model2AbsThrEntry::new(41, 41, 0.74),
    Model2AbsThrEntry::new(42, 42, 0.56),
    Model2AbsThrEntry::new(43, 43, 0.39),
    Model2AbsThrEntry::new(44, 44, 0.21),
    Model2AbsThrEntry::new(45, 45, 0.02),
    Model2AbsThrEntry::new(46, 46, -0.17),
    Model2AbsThrEntry::new(47, 47, -0.36),
    Model2AbsThrEntry::new(48, 48, -0.56),
    Model2AbsThrEntry::new(49, 50, -0.96),
    Model2AbsThrEntry::new(51, 52, -1.37),
    Model2AbsThrEntry::new(53, 54, -1.79),
    Model2AbsThrEntry::new(55, 56, -2.21),
    Model2AbsThrEntry::new(57, 58, -2.63),
    Model2AbsThrEntry::new(59, 60, -3.03),
    Model2AbsThrEntry::new(61, 62, -3.41),
    Model2AbsThrEntry::new(63, 64, -3.77),
    Model2AbsThrEntry::new(65, 66, -4.09),
    Model2AbsThrEntry::new(67, 68, -4.37),
    Model2AbsThrEntry::new(69, 70, -4.60),
    Model2AbsThrEntry::new(71, 72, -4.78),
    Model2AbsThrEntry::new(73, 74, -4.91),
    Model2AbsThrEntry::new(75, 76, -4.97),
    Model2AbsThrEntry::new(77, 78, -4.98),
    Model2AbsThrEntry::new(79, 80, -4.92),
    Model2AbsThrEntry::new(81, 82, -4.81),
    Model2AbsThrEntry::new(83, 84, -4.65),
    Model2AbsThrEntry::new(85, 86, -4.43),
    Model2AbsThrEntry::new(87, 88, -4.17),
    Model2AbsThrEntry::new(89, 90, -3.87),
    Model2AbsThrEntry::new(91, 92, -3.54),
    Model2AbsThrEntry::new(93, 94, -3.19),
    Model2AbsThrEntry::new(95, 96, -2.82),
    Model2AbsThrEntry::new(97, 100, -2.06),
    Model2AbsThrEntry::new(101, 104, -1.33),
    Model2AbsThrEntry::new(105, 108, -0.64),
    Model2AbsThrEntry::new(109, 112, -0.04),
    Model2AbsThrEntry::new(113, 116, 0.47),
    Model2AbsThrEntry::new(117, 120, 0.89),
    Model2AbsThrEntry::new(121, 124, 1.23),
    Model2AbsThrEntry::new(125, 128, 1.51),
    Model2AbsThrEntry::new(129, 132, 1.74),
    Model2AbsThrEntry::new(133, 136, 1.93),
    Model2AbsThrEntry::new(137, 140, 2.11),
    Model2AbsThrEntry::new(141, 144, 2.28),
    Model2AbsThrEntry::new(145, 148, 2.45),
    Model2AbsThrEntry::new(149, 152, 2.63),
    Model2AbsThrEntry::new(153, 156, 2.82),
    Model2AbsThrEntry::new(157, 160, 3.03),
    Model2AbsThrEntry::new(161, 164, 3.25),
    Model2AbsThrEntry::new(165, 168, 3.49),
    Model2AbsThrEntry::new(169, 172, 3.74),
    Model2AbsThrEntry::new(173, 176, 4.02),
    Model2AbsThrEntry::new(177, 180, 4.32),
    Model2AbsThrEntry::new(181, 184, 4.64),
    Model2AbsThrEntry::new(185, 188, 4.98),
    Model2AbsThrEntry::new(189, 192, 5.35),
    Model2AbsThrEntry::new(193, 200, 6.15),
    Model2AbsThrEntry::new(201, 208, 7.07),
    Model2AbsThrEntry::new(209, 216, 8.10),
    Model2AbsThrEntry::new(217, 224, 9.25),
    Model2AbsThrEntry::new(225, 232, 10.54),
    Model2AbsThrEntry::new(233, 240, 11.97),
    Model2AbsThrEntry::new(241, 248, 13.56),
    Model2AbsThrEntry::new(249, 256, 15.30),
    Model2AbsThrEntry::new(257, 264, 17.23),
    Model2AbsThrEntry::new(265, 272, 19.33),
    Model2AbsThrEntry::new(273, 280, 21.64),
    Model2AbsThrEntry::new(281, 288, 24.15),
    Model2AbsThrEntry::new(289, 296, 26.88),
    Model2AbsThrEntry::new(297, 304, 29.84),
    Model2AbsThrEntry::new(305, 312, 33.04),
    Model2AbsThrEntry::new(313, 320, 36.51),
    Model2AbsThrEntry::new(321, 328, 40.24),
    Model2AbsThrEntry::new(329, 336, 44.26),
    Model2AbsThrEntry::new(337, 344, 48.58),
    Model2AbsThrEntry::new(345, 352, 53.21),
    Model2AbsThrEntry::new(353, 360, 58.17),
    Model2AbsThrEntry::new(361, 368, 63.48),
    Model2AbsThrEntry::new(369, 376, 69.13),
    Model2AbsThrEntry::new(377, 384, 69.13),
    Model2AbsThrEntry::new(385, 392, 69.13),
    Model2AbsThrEntry::new(393, 400, 69.13),
    Model2AbsThrEntry::new(401, 408, 69.13),
    Model2AbsThrEntry::new(409, 416, 69.13),
    Model2AbsThrEntry::new(417, 424, 69.13),
    Model2AbsThrEntry::new(425, 432, 69.13),
    Model2AbsThrEntry::new(433, 440, 69.13),
    Model2AbsThrEntry::new(441, 448, 69.13),
    Model2AbsThrEntry::new(449, 456, 69.13),
    Model2AbsThrEntry::new(457, 464, 69.13),
];

/// Table D.4c — absolute threshold table, Fs = 48 kHz (126 rows,
/// lines 1..=428; printed p.138).
///
/// Verbatim from
/// `docs/audio/mp3/annex-d-renders/Table-D.4c-absolute-threshold-48kHz-p138.png`
/// — including the printed 4-line group `329 | 332` (see the section
/// comment).
pub const MODEL2_ABSTHR_D4C: [Model2AbsThrEntry; 126] = [
    Model2AbsThrEntry::new(1, 1, 42.10),
    Model2AbsThrEntry::new(2, 2, 24.17),
    Model2AbsThrEntry::new(3, 3, 17.47),
    Model2AbsThrEntry::new(4, 4, 13.87),
    Model2AbsThrEntry::new(5, 5, 11.60),
    Model2AbsThrEntry::new(6, 6, 10.01),
    Model2AbsThrEntry::new(7, 7, 8.84),
    Model2AbsThrEntry::new(8, 8, 7.94),
    Model2AbsThrEntry::new(9, 9, 7.22),
    Model2AbsThrEntry::new(10, 10, 6.62),
    Model2AbsThrEntry::new(11, 11, 6.12),
    Model2AbsThrEntry::new(12, 12, 5.70),
    Model2AbsThrEntry::new(13, 13, 5.33),
    Model2AbsThrEntry::new(14, 14, 5.00),
    Model2AbsThrEntry::new(15, 15, 4.71),
    Model2AbsThrEntry::new(16, 16, 4.45),
    Model2AbsThrEntry::new(17, 17, 4.21),
    Model2AbsThrEntry::new(18, 18, 4.00),
    Model2AbsThrEntry::new(19, 19, 3.79),
    Model2AbsThrEntry::new(20, 20, 3.61),
    Model2AbsThrEntry::new(21, 21, 3.43),
    Model2AbsThrEntry::new(22, 22, 3.26),
    Model2AbsThrEntry::new(23, 23, 3.09),
    Model2AbsThrEntry::new(24, 24, 2.93),
    Model2AbsThrEntry::new(25, 25, 2.78),
    Model2AbsThrEntry::new(26, 26, 2.63),
    Model2AbsThrEntry::new(27, 27, 2.47),
    Model2AbsThrEntry::new(28, 28, 2.32),
    Model2AbsThrEntry::new(29, 29, 2.17),
    Model2AbsThrEntry::new(30, 30, 2.02),
    Model2AbsThrEntry::new(31, 31, 1.86),
    Model2AbsThrEntry::new(32, 32, 1.71),
    Model2AbsThrEntry::new(33, 33, 1.55),
    Model2AbsThrEntry::new(34, 34, 1.38),
    Model2AbsThrEntry::new(35, 35, 1.21),
    Model2AbsThrEntry::new(36, 36, 1.04),
    Model2AbsThrEntry::new(37, 37, 0.86),
    Model2AbsThrEntry::new(38, 38, 0.67),
    Model2AbsThrEntry::new(39, 39, 0.49),
    Model2AbsThrEntry::new(40, 40, 0.29),
    Model2AbsThrEntry::new(41, 41, 0.09),
    Model2AbsThrEntry::new(42, 42, -0.11),
    Model2AbsThrEntry::new(43, 43, -0.32),
    Model2AbsThrEntry::new(44, 44, -0.54),
    Model2AbsThrEntry::new(45, 45, -0.75),
    Model2AbsThrEntry::new(46, 46, -0.97),
    Model2AbsThrEntry::new(47, 47, -1.20),
    Model2AbsThrEntry::new(48, 48, -1.43),
    Model2AbsThrEntry::new(49, 50, -1.88),
    Model2AbsThrEntry::new(51, 52, -2.34),
    Model2AbsThrEntry::new(53, 54, -2.79),
    Model2AbsThrEntry::new(55, 56, -3.22),
    Model2AbsThrEntry::new(57, 58, -3.62),
    Model2AbsThrEntry::new(59, 60, -3.98),
    Model2AbsThrEntry::new(61, 62, -4.30),
    Model2AbsThrEntry::new(63, 64, -4.57),
    Model2AbsThrEntry::new(65, 66, -4.77),
    Model2AbsThrEntry::new(67, 68, -4.91),
    Model2AbsThrEntry::new(69, 70, -4.98),
    Model2AbsThrEntry::new(71, 72, -4.97),
    Model2AbsThrEntry::new(73, 74, -4.90),
    Model2AbsThrEntry::new(75, 76, -4.76),
    Model2AbsThrEntry::new(77, 78, -4.55),
    Model2AbsThrEntry::new(79, 80, -4.29),
    Model2AbsThrEntry::new(81, 82, -3.99),
    Model2AbsThrEntry::new(83, 84, -3.64),
    Model2AbsThrEntry::new(85, 86, -3.26),
    Model2AbsThrEntry::new(87, 88, -2.86),
    Model2AbsThrEntry::new(89, 90, -2.45),
    Model2AbsThrEntry::new(91, 92, -2.04),
    Model2AbsThrEntry::new(93, 94, -1.63),
    Model2AbsThrEntry::new(95, 96, -1.24),
    Model2AbsThrEntry::new(97, 100, -0.51),
    Model2AbsThrEntry::new(101, 104, 0.12),
    Model2AbsThrEntry::new(105, 108, 0.64),
    Model2AbsThrEntry::new(109, 112, 1.06),
    Model2AbsThrEntry::new(113, 116, 1.39),
    Model2AbsThrEntry::new(117, 120, 1.66),
    Model2AbsThrEntry::new(121, 124, 1.88),
    Model2AbsThrEntry::new(125, 128, 2.08),
    Model2AbsThrEntry::new(129, 132, 2.27),
    Model2AbsThrEntry::new(133, 136, 2.46),
    Model2AbsThrEntry::new(137, 140, 2.65),
    Model2AbsThrEntry::new(141, 144, 2.86),
    Model2AbsThrEntry::new(145, 148, 3.09),
    Model2AbsThrEntry::new(149, 152, 3.33),
    Model2AbsThrEntry::new(153, 156, 3.60),
    Model2AbsThrEntry::new(157, 160, 3.89),
    Model2AbsThrEntry::new(161, 164, 4.20),
    Model2AbsThrEntry::new(165, 168, 4.54),
    Model2AbsThrEntry::new(169, 172, 4.91),
    Model2AbsThrEntry::new(173, 176, 5.31),
    Model2AbsThrEntry::new(177, 180, 5.73),
    Model2AbsThrEntry::new(181, 184, 6.18),
    Model2AbsThrEntry::new(185, 188, 6.67),
    Model2AbsThrEntry::new(189, 192, 7.19),
    Model2AbsThrEntry::new(193, 200, 8.33),
    Model2AbsThrEntry::new(201, 208, 9.63),
    Model2AbsThrEntry::new(209, 216, 11.08),
    Model2AbsThrEntry::new(217, 224, 12.71),
    Model2AbsThrEntry::new(225, 232, 14.53),
    Model2AbsThrEntry::new(233, 240, 16.54),
    Model2AbsThrEntry::new(241, 248, 18.77),
    Model2AbsThrEntry::new(249, 256, 21.23),
    Model2AbsThrEntry::new(257, 264, 23.94),
    Model2AbsThrEntry::new(265, 272, 26.90),
    Model2AbsThrEntry::new(273, 280, 30.14),
    Model2AbsThrEntry::new(281, 288, 33.67),
    Model2AbsThrEntry::new(289, 296, 37.51),
    Model2AbsThrEntry::new(297, 304, 41.67),
    Model2AbsThrEntry::new(305, 312, 46.17),
    Model2AbsThrEntry::new(313, 320, 51.04),
    Model2AbsThrEntry::new(321, 328, 56.29),
    Model2AbsThrEntry::new(329, 332, 61.94),
    Model2AbsThrEntry::new(333, 340, 68.00),
    Model2AbsThrEntry::new(341, 348, 68.00),
    Model2AbsThrEntry::new(349, 356, 68.00),
    Model2AbsThrEntry::new(357, 364, 68.00),
    Model2AbsThrEntry::new(365, 372, 68.00),
    Model2AbsThrEntry::new(373, 380, 68.00),
    Model2AbsThrEntry::new(381, 388, 68.00),
    Model2AbsThrEntry::new(389, 396, 68.00),
    Model2AbsThrEntry::new(397, 404, 68.00),
    Model2AbsThrEntry::new(405, 412, 68.00),
    Model2AbsThrEntry::new(413, 420, 68.00),
    Model2AbsThrEntry::new(421, 428, 68.00),
];

/// Return the verbatim Annex D Table D.4 absolute-threshold slice for
/// `fs` (common to all Layers, like the Table D.3 partition tables).
#[inline]
#[must_use]
pub fn model2_absthr_table(fs: AnnexDSamplingRate) -> &'static [Model2AbsThrEntry] {
    match fs {
        AnnexDSamplingRate::Hz32000 => &MODEL2_ABSTHR_D4A,
        AnnexDSamplingRate::Hz44100 => &MODEL2_ABSTHR_D4B,
        AnnexDSamplingRate::Hz48000 => &MODEL2_ABSTHR_D4C,
    }
}

/// The Table D.4 absolute threshold (dB, in the printed 0-dB
/// convention — 96 dB below a +-32 760 sine) for the 1-based FFT line
/// `line`, or `None` where the printed table has no covering row:
/// `line == 0`, `line` above the table's last covered line (480 /
/// 464 / 428 at 32 / 44,1 / 48 kHz — the tables stop short of line
/// 513), and the printed D.4a line-58 gap at 32 kHz.
#[must_use]
pub fn model2_absthr_for_line(fs: AnnexDSamplingRate, line: u16) -> Option<f64> {
    model2_absthr_table(fs)
        .iter()
        .find(|e| e.lower <= line && line <= e.higher)
        .map(|e| e.absthr_db)
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

    // ---- Phase 2 step 46 / r229 — §D.1 Step 4 placement + Step 7
    //      Bark-window range pre-filter primitives.

    #[test]
    fn masker_at_band_uses_band_top_z_bark_for_first_band() {
        // D.2a band 0 has z_bark = 0,617 per the docs file (row 1 of
        // the §D.1 Step 4 table for Layer I, 32 kHz). A tonal masker
        // placed at band 0 with SPL = 60 dB sits at z = 0.617 and
        // carries the caller-supplied SPL verbatim.
        let m = masker_at_band(&CRITICAL_BANDS_D2A, 0, MaskerKind::Tonal, 60.0)
            .expect("band 0 is in range");
        assert_eq!(m.kind, MaskerKind::Tonal);
        assert!((m.z_bark - 0.617).abs() < 1.0e-9);
        assert!((m.spl_db - 60.0).abs() < 1.0e-9);
    }

    #[test]
    fn masker_at_band_uses_band_top_z_bark_for_last_band() {
        // D.2a band 23 has z_bark = 23,923 per the docs file (last row
        // of the §D.1 Step 4 table for Layer I, 32 kHz; the top
        // critical-band boundary inside the 32 kHz audio band, at FFT
        // line 108 / 15 000 Hz). A non-tonal masker placed at band
        // 23 with SPL = 45 dB sits at z = 23.923.
        let m = masker_at_band(&CRITICAL_BANDS_D2A, 23, MaskerKind::NonTonal, 45.0)
            .expect("band 23 is in range");
        assert_eq!(m.kind, MaskerKind::NonTonal);
        assert!((m.z_bark - 23.923).abs() < 1.0e-9);
        assert!((m.spl_db - 45.0).abs() < 1.0e-9);
    }

    #[test]
    fn masker_at_band_out_of_range_returns_none() {
        // D.2a has 24 bands (no 0..23). Band 24 is out of range.
        assert!(masker_at_band(&CRITICAL_BANDS_D2A, 24, MaskerKind::Tonal, 60.0).is_none());
        // And an obviously-too-large index also returns None.
        assert!(masker_at_band(&CRITICAL_BANDS_D2A, 999, MaskerKind::NonTonal, 30.0).is_none());
    }

    #[test]
    fn masker_at_band_dispatches_per_table() {
        // The same band index draws from different tables based on
        // (Layer, Fs). Band 0 of D.2d (Layer II, 32 kHz) sits below
        // band 0 of D.2a (Layer I, 32 kHz): Layer II's longer FFT
        // resolves a lower band edge. Placement reads the matching
        // table's z_bark verbatim.
        let l1 = masker_at_band(&CRITICAL_BANDS_D2A, 0, MaskerKind::Tonal, 60.0).unwrap();
        let l2 = masker_at_band(&CRITICAL_BANDS_D2D, 0, MaskerKind::Tonal, 60.0).unwrap();
        assert!(l2.z_bark < l1.z_bark);
    }

    #[test]
    fn masker_at_band_then_individual_threshold_reproduces_self_spl_plus_av() {
        // Composition smoke test: place a tonal masker at D.2a band 5
        // with SPL = 70 dB, then evaluate the individual masking
        // threshold at the masker's own z_bark. The result must equal
        // SPL + av_tm(z) per the §D.1 Step 6 spec equation
        // (vf(0, X) = 0).
        let m = masker_at_band(&CRITICAL_BANDS_D2A, 5, MaskerKind::Tonal, 70.0).unwrap();
        let lt = individual_masking_threshold_db(&m, m.z_bark).unwrap();
        let av = masking_index_tonal(m.z_bark);
        assert!((lt - (70.0 + av)).abs() < 1.0e-9);
    }

    #[test]
    fn masker_at_band_then_global_threshold_at_band_top_exceeds_ltq() {
        // A loud local masker placed at a band's top z_bark should
        // dominate the global threshold sum at that line: the result
        // is strictly above the threshold-in-quiet floor.
        let m = masker_at_band(&CRITICAL_BANDS_D2C, 10, MaskerKind::NonTonal, 80.0).unwrap();
        let ltq_db = -4.97;
        let ltg = global_masking_threshold_db(&[m], m.z_bark, ltq_db);
        assert!(
            ltg > ltq_db + 30.0,
            "LTg = {ltg} should be >> LTq = {ltq_db}"
        );
    }

    #[test]
    fn step7_window_constants_match_spec_text() {
        // Verbatim spec text: "for a given i the range of j may be
        // reduced to maskers within −8…+3 Bark of i". The constants
        // expose those two numbers as `dz = z(j) - z(i)` bounds: the
        // low end is open at -8 (the vf branch is right-open at
        // dz < 8 → z(j) > z(i) - 8), the high end is closed at +3
        // (the vf branch is left-closed at dz >= -3 → z(j) <=
        // z(i) + 3).
        assert!((STEP7_NEARBY_MASKER_DZ_LO_FROM_LINE - (-8.0)).abs() < 1.0e-12);
        assert!((STEP7_NEARBY_MASKER_DZ_HI_FROM_LINE - 3.0).abs() < 1.0e-12);
    }

    #[test]
    fn step7_window_includes_in_range_masker() {
        // Masker at z_j = 5 Bark, target line at z_i = 5 Bark
        // (dz_from_line = 0): in range.
        let m = Masker {
            kind: MaskerKind::Tonal,
            z_bark: 5.0,
            spl_db: 60.0,
        };
        assert!(masker_in_step7_window_of_line(&m, 5.0));
        // Masker 2 Bark above the line (dz_from_line = +2): in
        // range.
        assert!(masker_in_step7_window_of_line(&m, 3.0));
        // Masker 5 Bark below the line (dz_from_line = -5): in
        // range.
        assert!(masker_in_step7_window_of_line(&m, 10.0));
    }

    #[test]
    fn step7_window_high_edge_inclusive() {
        // Masker exactly 3 Bark above the target line
        // (dz_from_line = +3): inclusive — the §D.1 Step 6 `vf` lower
        // branch is left-closed at dz = -3, so this masker still
        // contributes (vf returns Some).
        let m = Masker {
            kind: MaskerKind::Tonal,
            z_bark: 8.0,
            spl_db: 60.0,
        };
        assert!(masker_in_step7_window_of_line(&m, 5.0));
        // Confirm `vf` agrees: the corresponding dz = -3 returns
        // Some (the `vf` predicate returns Some on the closed left
        // edge).
        let dz = 5.0_f64 - 8.0;
        assert_eq!(dz, -3.0);
        assert!(masking_function_vf(dz, 60.0).is_some());
    }

    #[test]
    fn step7_window_high_edge_just_above_excluded() {
        // Masker `dz_from_line = +3.0001`: out of range.
        let m = Masker {
            kind: MaskerKind::Tonal,
            z_bark: 8.0001,
            spl_db: 60.0,
        };
        assert!(!masker_in_step7_window_of_line(&m, 5.0));
    }

    #[test]
    fn step7_window_low_edge_exclusive() {
        // Masker exactly 8 Bark below the target line
        // (dz_from_line = -8): exclusive — the §D.1 Step 6 `vf`
        // upper branch is right-open at dz < 8, so this masker is
        // ignored.
        let m = Masker {
            kind: MaskerKind::Tonal,
            z_bark: 5.0,
            spl_db: 60.0,
        };
        // dz_from_line = 5 - 13 = -8 → exclusive.
        assert!(!masker_in_step7_window_of_line(&m, 13.0));
        // Confirm vf agrees: dz = +8 (z(i) - z(j) = 13 - 5 = 8)
        // returns None.
        let dz = 13.0_f64 - 5.0;
        assert_eq!(dz, 8.0);
        assert!(masking_function_vf(dz, 60.0).is_none());
    }

    #[test]
    fn step7_window_low_edge_just_above_included() {
        // Masker `dz_from_line = -7.999`: in range. The masker sits
        // 7.999 Bark below the line, just inside the open low edge.
        let m = Masker {
            kind: MaskerKind::Tonal,
            z_bark: 5.0,
            spl_db: 60.0,
        };
        assert!(masker_in_step7_window_of_line(&m, 12.999));
    }

    #[test]
    fn step7_window_matches_individual_threshold_some_set() {
        // Spec invariant — the §D.1 Step 7 nearby-masker window
        // exactly equals the set of maskers for which
        // `individual_masking_threshold_db` returns `Some`.
        // Spot-check: iterate a 0.25-Bark sweep of masker positions
        // around a fixed target line and verify the predicate
        // agrees with the `vf` `Some/None` outcome.
        let z_i = 10.0;
        let mut sweep_z = -2.0;
        while sweep_z < 20.0 {
            let m = Masker {
                kind: MaskerKind::NonTonal,
                z_bark: sweep_z,
                spl_db: 50.0,
            };
            let predicate = masker_in_step7_window_of_line(&m, z_i);
            let lt = individual_masking_threshold_db(&m, z_i);
            assert_eq!(
                predicate,
                lt.is_some(),
                "predicate disagrees at sweep_z = {sweep_z}: pred = {predicate}, lt = {lt:?}",
            );
            sweep_z += 0.25;
        }
    }

    #[test]
    fn step7_window_pre_filter_preserves_global_threshold_value() {
        // Functional invariant — pre-filtering the masker slice with
        // the §D.1 Step 7 window predicate produces the same
        // `LTg(i)` value as feeding the full slice to
        // `global_masking_threshold_db`. (The optimisation is
        // strictly mechanical: filtered-out maskers would have
        // contributed `vf = None` anyway, dropping them from the
        // energy sum.)
        let z_i = 8.0;
        let ltq_db = -3.0;
        let maskers = [
            // In-range tonal masker.
            Masker {
                kind: MaskerKind::Tonal,
                z_bark: 7.5,
                spl_db: 65.0,
            },
            // In-range non-tonal masker.
            Masker {
                kind: MaskerKind::NonTonal,
                z_bark: 9.5,
                spl_db: 55.0,
            },
            // Out-of-range masker far below the line.
            Masker {
                kind: MaskerKind::Tonal,
                z_bark: -5.0,
                spl_db: 90.0,
            },
            // Out-of-range masker far above the line.
            Masker {
                kind: MaskerKind::NonTonal,
                z_bark: 20.0,
                spl_db: 90.0,
            },
        ];
        let full = global_masking_threshold_db(&maskers, z_i, ltq_db);
        let filtered: Vec<Masker> = maskers
            .iter()
            .copied()
            .filter(|m| masker_in_step7_window_of_line(m, z_i))
            .collect();
        // Two maskers survive the pre-filter.
        assert_eq!(filtered.len(), 2);
        let after = global_masking_threshold_db(&filtered, z_i, ltq_db);
        assert!(
            (full - after).abs() < 1.0e-12,
            "pre-filter changed LTg: full = {full}, after = {after}",
        );
    }

    // ---- Phase 2 step 47 — §D.1 Step 5 decimation primitives.

    #[test]
    fn step5a_keeps_masker_above_ltq() {
        // Masker SPL strictly above LTq → kept (tonal).
        let m_tonal = Masker {
            kind: MaskerKind::Tonal,
            z_bark: 5.0,
            spl_db: 40.0,
        };
        assert!(masker_above_threshold_in_quiet(&m_tonal, 20.0));
        // Same for a non-tonal masker — the spec rule is identical
        // for both classes.
        let m_nt = Masker {
            kind: MaskerKind::NonTonal,
            z_bark: 5.0,
            spl_db: 40.0,
        };
        assert!(masker_above_threshold_in_quiet(&m_nt, 20.0));
    }

    #[test]
    fn step5a_drops_masker_below_ltq() {
        // Masker SPL strictly below LTq → dropped.
        let m_tonal = Masker {
            kind: MaskerKind::Tonal,
            z_bark: 5.0,
            spl_db: 10.0,
        };
        assert!(!masker_above_threshold_in_quiet(&m_tonal, 20.0));
        let m_nt = Masker {
            kind: MaskerKind::NonTonal,
            z_bark: 5.0,
            spl_db: -5.0,
        };
        assert!(!masker_above_threshold_in_quiet(&m_nt, 0.0));
    }

    #[test]
    fn step5a_keeps_masker_at_ltq_inclusive() {
        // The spec uses `>=`, so a masker exactly at LTq survives.
        let m = Masker {
            kind: MaskerKind::Tonal,
            z_bark: 5.0,
            spl_db: 20.0,
        };
        assert!(masker_above_threshold_in_quiet(&m, 20.0));
    }

    #[test]
    fn step5b_window_constant_is_half_bark() {
        // Verbatim spec text: "a sliding window in the critical band
        // domain is used with a width of 0,5 Bark".
        assert!((STEP5_TONAL_DECIMATION_WINDOW_BARK - 0.5).abs() < 1.0e-12);
    }

    #[test]
    fn step5b_empty_input_returns_empty() {
        let out = decimate_tonal_within_half_bark(&[]);
        assert!(out.is_empty());
    }

    #[test]
    fn step5b_singleton_passes_through() {
        let m = Masker {
            kind: MaskerKind::Tonal,
            z_bark: 5.0,
            spl_db: 60.0,
        };
        let out = decimate_tonal_within_half_bark(&[m]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], m);
    }

    #[test]
    fn step5b_pair_within_window_keeps_loudest() {
        // Two tonal maskers 0.3 Bark apart — strictly inside the
        // 0.5-Bark window. The louder one survives.
        let quiet = Masker {
            kind: MaskerKind::Tonal,
            z_bark: 5.0,
            spl_db: 40.0,
        };
        let loud = Masker {
            kind: MaskerKind::Tonal,
            z_bark: 5.3,
            spl_db: 70.0,
        };
        let out = decimate_tonal_within_half_bark(&[quiet, loud]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], loud);
    }

    #[test]
    fn step5b_pair_at_exact_half_bark_both_survive() {
        // The spec text reads "less than 0,5 Bark", so exactly
        // 0.5 Bark apart is OUTSIDE the cluster — both survive.
        let a = Masker {
            kind: MaskerKind::Tonal,
            z_bark: 5.0,
            spl_db: 40.0,
        };
        let b = Masker {
            kind: MaskerKind::Tonal,
            z_bark: 5.5,
            spl_db: 70.0,
        };
        let out = decimate_tonal_within_half_bark(&[a, b]);
        assert_eq!(out.len(), 2);
        // Output preserves input order.
        assert_eq!(out[0], a);
        assert_eq!(out[1], b);
    }

    #[test]
    fn step5b_pair_outside_window_both_survive() {
        // Two tonal maskers 1.0 Bark apart — well outside the window.
        let a = Masker {
            kind: MaskerKind::Tonal,
            z_bark: 5.0,
            spl_db: 60.0,
        };
        let b = Masker {
            kind: MaskerKind::Tonal,
            z_bark: 6.0,
            spl_db: 50.0,
        };
        let out = decimate_tonal_within_half_bark(&[a, b]);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn step5b_non_tonal_passes_through_unchanged() {
        // Non-tonal maskers are not subject to Step 5(b) at all — even
        // a tight 0.1-Bark non-tonal cluster survives intact (the spec
        // applies decimation only to tonal maskers because Step 4(c)
        // already produces at most one non-tonal masker per critical
        // band).
        let a = Masker {
            kind: MaskerKind::NonTonal,
            z_bark: 5.0,
            spl_db: 40.0,
        };
        let b = Masker {
            kind: MaskerKind::NonTonal,
            z_bark: 5.1,
            spl_db: 70.0,
        };
        let out = decimate_tonal_within_half_bark(&[a, b]);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0], a);
        assert_eq!(out[1], b);
    }

    #[test]
    fn step5b_cluster_of_three_keeps_only_loudest() {
        // Three tonal maskers in a tight cluster: 5.0, 5.2, 5.4 Bark
        // (consecutive gaps 0.2 each, total span 0.4 — fully inside
        // the 0.5-Bark window). The single loudest survives.
        let q1 = Masker {
            kind: MaskerKind::Tonal,
            z_bark: 5.0,
            spl_db: 40.0,
        };
        let loud = Masker {
            kind: MaskerKind::Tonal,
            z_bark: 5.2,
            spl_db: 80.0,
        };
        let q2 = Masker {
            kind: MaskerKind::Tonal,
            z_bark: 5.4,
            spl_db: 55.0,
        };
        let out = decimate_tonal_within_half_bark(&[q1, loud, q2]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], loud);
    }

    #[test]
    fn step5b_two_separate_clusters_each_collapse_independently() {
        // Cluster A: 5.0 / 5.2 (loudest at 5.2). Gap to cluster B:
        // 5.2 → 6.0 = 0.8 Bark (outside the window). Cluster B:
        // 6.0 / 6.1 (loudest at 6.1). Result: two surviving maskers.
        let a1 = Masker {
            kind: MaskerKind::Tonal,
            z_bark: 5.0,
            spl_db: 50.0,
        };
        let a2_loud = Masker {
            kind: MaskerKind::Tonal,
            z_bark: 5.2,
            spl_db: 75.0,
        };
        let b1 = Masker {
            kind: MaskerKind::Tonal,
            z_bark: 6.0,
            spl_db: 60.0,
        };
        let b2_loud = Masker {
            kind: MaskerKind::Tonal,
            z_bark: 6.1,
            spl_db: 65.0,
        };
        let out = decimate_tonal_within_half_bark(&[a1, a2_loud, b1, b2_loud]);
        assert_eq!(out.len(), 2);
        // Output preserves original-slice ordering.
        assert_eq!(out[0], a2_loud);
        assert_eq!(out[1], b2_loud);
    }

    #[test]
    fn step5b_ties_resolve_to_first_encountered() {
        // Two tonal maskers in-cluster with identical SPLs: the
        // first-encountered (lower-z_bark) wins so the output is
        // stable across repeated calls.
        let first = Masker {
            kind: MaskerKind::Tonal,
            z_bark: 5.0,
            spl_db: 60.0,
        };
        let second = Masker {
            kind: MaskerKind::Tonal,
            z_bark: 5.3,
            spl_db: 60.0,
        };
        let out = decimate_tonal_within_half_bark(&[first, second]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], first);
    }

    #[test]
    fn step5b_unsorted_input_still_clusters_correctly() {
        // The caller may pass maskers in arbitrary order (Step 4
        // emits tonal then non-tonal lists, which Step 5(b) doesn't
        // necessarily receive sorted by z_bark). Verify a deliberately
        // shuffled cluster still collapses to its loudest member.
        let loud = Masker {
            kind: MaskerKind::Tonal,
            z_bark: 5.2,
            spl_db: 80.0,
        };
        let quiet_a = Masker {
            kind: MaskerKind::Tonal,
            z_bark: 5.0,
            spl_db: 40.0,
        };
        let quiet_b = Masker {
            kind: MaskerKind::Tonal,
            z_bark: 5.4,
            spl_db: 50.0,
        };
        // Input order: loud, quiet_a, quiet_b (unsorted by z_bark).
        let out = decimate_tonal_within_half_bark(&[loud, quiet_a, quiet_b]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], loud);
    }

    #[test]
    fn step5b_mixed_tonal_and_non_tonal_preserves_non_tonal_in_place() {
        // Cluster of tonal maskers around 5.0 Bark, two non-tonal
        // maskers at 3.0 and 7.0 (well outside the tonal cluster).
        // The non-tonal maskers must survive at their original
        // positions in the output slice.
        let nt_low = Masker {
            kind: MaskerKind::NonTonal,
            z_bark: 3.0,
            spl_db: 30.0,
        };
        let t_quiet = Masker {
            kind: MaskerKind::Tonal,
            z_bark: 5.0,
            spl_db: 40.0,
        };
        let t_loud = Masker {
            kind: MaskerKind::Tonal,
            z_bark: 5.2,
            spl_db: 70.0,
        };
        let nt_high = Masker {
            kind: MaskerKind::NonTonal,
            z_bark: 7.0,
            spl_db: 35.0,
        };
        let out = decimate_tonal_within_half_bark(&[nt_low, t_quiet, t_loud, nt_high]);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0], nt_low);
        assert_eq!(out[1], t_loud);
        assert_eq!(out[2], nt_high);
    }

    #[test]
    fn step5_composes_a_then_b() {
        // Compositional invariant — Step 5(a) screening followed by
        // Step 5(b) tonal decimation reproduces the spec's full
        // Step 5 sieve. Source slice contains:
        //   - tonal masker below LTq (Step 5(a) drops),
        //   - tonal cluster with one loud + one quiet member,
        //     both above LTq (Step 5(b) collapses to loud one),
        //   - lone tonal masker above LTq + outside any cluster
        //     (both steps pass through),
        //   - non-tonal masker above LTq (both steps pass through).
        let ltq_db = 20.0;
        let quiet_below = Masker {
            kind: MaskerKind::Tonal,
            z_bark: 4.0,
            spl_db: 10.0,
        }; // dropped by (a)
        let cluster_quiet = Masker {
            kind: MaskerKind::Tonal,
            z_bark: 5.0,
            spl_db: 40.0,
        };
        let cluster_loud = Masker {
            kind: MaskerKind::Tonal,
            z_bark: 5.3,
            spl_db: 70.0,
        };
        let isolated = Masker {
            kind: MaskerKind::Tonal,
            z_bark: 8.0,
            spl_db: 60.0,
        };
        let non_tonal = Masker {
            kind: MaskerKind::NonTonal,
            z_bark: 10.0,
            spl_db: 50.0,
        };
        let src = [
            quiet_below,
            cluster_quiet,
            cluster_loud,
            isolated,
            non_tonal,
        ];
        let after_a: Vec<Masker> = src
            .iter()
            .copied()
            .filter(|m| masker_above_threshold_in_quiet(m, ltq_db))
            .collect();
        // Step 5(a) drops the below-LTq masker; 4 survive.
        assert_eq!(after_a.len(), 4);
        let after_b = decimate_tonal_within_half_bark(&after_a);
        // Step 5(b) collapses the cluster; 3 survive total.
        assert_eq!(after_b.len(), 3);
        assert_eq!(after_b[0], cluster_loud);
        assert_eq!(after_b[1], isolated);
        assert_eq!(after_b[2], non_tonal);
    }

    #[test]
    fn step5_then_step7_feeds_global_threshold_consistently() {
        // End-to-end compositional smoke: after Step 5(a) + Step 5(b),
        // feeding the surviving slice through Step 7's
        // `global_masking_threshold_db` produces an LTg(i) that is
        // strictly above the floor LTq for a target line near the
        // surviving loud masker, and matches what we'd get if we
        // had hand-decimated the input.
        let z_i = 5.3;
        let ltq_db = 0.0;
        // Cluster: dropped + loud (both above LTq). Plus an
        // out-of-band below-LTq masker that Step 5(a) discards.
        let drop_a = Masker {
            kind: MaskerKind::Tonal,
            z_bark: 5.0,
            spl_db: 40.0,
        };
        let keep = Masker {
            kind: MaskerKind::Tonal,
            z_bark: 5.3,
            spl_db: 75.0,
        };
        let below_ltq = Masker {
            kind: MaskerKind::Tonal,
            z_bark: 8.0,
            spl_db: -10.0,
        };
        let src = [drop_a, keep, below_ltq];
        let after_a: Vec<Masker> = src
            .iter()
            .copied()
            .filter(|m| masker_above_threshold_in_quiet(m, ltq_db))
            .collect();
        assert_eq!(after_a.len(), 2);
        let after_b = decimate_tonal_within_half_bark(&after_a);
        assert_eq!(after_b.len(), 1);
        assert_eq!(after_b[0], keep);
        let ltg = global_masking_threshold_db(&after_b, z_i, ltq_db);
        // The loud masker is at the target line itself, so
        // LTg = X + av_tm(z) (in dB, after the energy sum
        // dominates LTq). The result is well above LTq.
        assert!(ltg > ltq_db + 30.0, "LTg = {ltg} should be >> LTq");
        // And it must equal the value obtained by manually keeping
        // only `keep`.
        let direct = global_masking_threshold_db(&[keep], z_i, ltq_db);
        assert!((ltg - direct).abs() < 1.0e-12);
    }

    // ---- Phase 2 step 48 — §C.1.5.3.2.1 Model 2 Layer III
    // spreading function primitives.

    #[test]
    fn model2_layer3_spread_db_on_diagonal_is_zero() {
        // Spec: both branches collapse at j == i (the `j >= i`
        // branch evaluates `3.0 * 0 = 0`). Spot-check a handful of
        // partition indices spanning Model 2's index range (1…63
        // for Fs = 32 kHz per Table D.3a).
        for i in [1, 5, 20, 40, 63] {
            assert_eq!(model2_layer3_spread_db(i, i), 0.0);
        }
    }

    #[test]
    fn model2_layer3_spread_db_upward_branch_matches_spec() {
        // Spec verbatim: `j >= i : tmpy = 3.0 * (j - i)`. Verify a
        // 1-Bark step (j = i + 1), a 5-step jump, and a 20-step
        // jump.
        let i = 10;
        assert_eq!(model2_layer3_spread_db(i, i + 1), 3.0);
        assert_eq!(model2_layer3_spread_db(i, i + 5), 15.0);
        assert_eq!(model2_layer3_spread_db(i, i + 20), 60.0);
    }

    #[test]
    fn model2_layer3_spread_db_downward_branch_matches_spec() {
        // Spec verbatim: `j < i : tmpy = 1.5 * (j - i)`. `j - i` is
        // negative on this branch so `tmpy` is negative.
        let i = 30;
        assert_eq!(model2_layer3_spread_db(i, i - 1), -1.5);
        assert_eq!(model2_layer3_spread_db(i, i - 4), -6.0);
        assert_eq!(model2_layer3_spread_db(i, i - 20), -30.0);
    }

    #[test]
    fn model2_layer3_spread_linear_diagonal_is_unity() {
        // `tmpy = 0` → `10^0 = 1`. Well above the 1e-6 clamp, so
        // the diagonal entry is exactly 1.0.
        for i in [1, 7, 25, 50, 63] {
            assert_eq!(model2_layer3_spread_linear(i, i), 1.0);
        }
    }

    #[test]
    fn model2_layer3_spread_linear_upward_is_above_unity() {
        // Upward branch carries a positive `tmpy`, so the linear
        // factor strictly exceeds 1.0 for any `j > i`.
        let i = 10;
        let f1 = model2_layer3_spread_linear(i, i + 1);
        let f3 = model2_layer3_spread_linear(i, i + 3);
        assert!(f1 > 1.0, "j = i + 1 factor must exceed unity, got {f1}");
        assert!(f3 > f1, "factor must grow with distance: {f3} vs {f1}");
        // tmpy(i, i+1) = +3 dB → 10^0.3 ≈ 1.9953.
        assert!((f1 - 1.995_262_314_968_88).abs() < 1.0e-9);
    }

    #[test]
    fn model2_layer3_spread_linear_downward_is_below_unity() {
        // Downward branch carries a negative `tmpy`, so the linear
        // factor is strictly below 1.0 for any `j < i` (until the
        // 1e-6 clamp kicks in at very large distances).
        let i = 30;
        let f1 = model2_layer3_spread_linear(i, i - 1);
        let f3 = model2_layer3_spread_linear(i, i - 3);
        assert!(f1 < 1.0, "j = i - 1 factor must be below unity, got {f1}");
        assert!(f3 < f1, "factor must shrink with distance: {f3} vs {f1}");
        assert!(f3 > 0.0, "factor stays positive away from clamp regime");
        // tmpy(i, i-1) = -1.5 dB → 10^(-0.15) ≈ 0.7079.
        assert!((f1 - 0.707_945_784_384_138).abs() < 1.0e-9);
    }

    #[test]
    fn model2_layer3_spread_linear_clamp_kicks_in_at_minus_60_db() {
        // Spec clamp: `> 1.0e-6` survives; `<= 1.0e-6` becomes 0.
        // `1.0e-6` in dB is exactly -60 dB. On the downward branch
        // `tmpy = 1.5 * (j - i)`, so `j - i = -40` gives `tmpy =
        // -60` dB → linear factor `10^(-6)` which is exactly the
        // clamp threshold (not strictly greater) → clamped to 0.
        // `j - i = -39` gives `tmpy = -58.5` dB → linear factor
        // ~`1.4e-6`, just above the clamp → survives.
        let i = 50;
        // Above the clamp threshold (survives).
        let survives = model2_layer3_spread_linear(i, i - 39);
        assert!(
            survives > MODEL2_LAYER3_SPREAD_LINEAR_MIN,
            "j - i = -39 must survive clamp, got {survives}",
        );
        // At the clamp threshold (collapses to exact zero per the
        // strict `>` comparison in the spec text "values *greater
        // than* 1e-6 are used").
        let at_threshold = model2_layer3_spread_linear(i, i - 40);
        assert_eq!(at_threshold, 0.0);
        // Below the threshold (clamped).
        let below = model2_layer3_spread_linear(i, i - 50);
        assert_eq!(below, 0.0);
    }

    #[test]
    fn model2_layer3_spread_constants_match_spec_text() {
        // Sanity-check the spec text "Only spreading-function
        // values greater than 1e-6 are used; all others set to
        // zero" — the constant is exactly the `1e-6` figure as
        // typeset in the spec.
        assert_eq!(MODEL2_LAYER3_SPREAD_LINEAR_MIN, 1.0e-6);
    }

    #[test]
    fn model2_layer3_spread_branches_agree_at_diagonal() {
        // Continuity check: the spec's `j >= i` branch with `j = i`
        // yields exactly the same value as a hypothetical
        // `1.5 * (j - i)` evaluation at `j = i` (both are zero).
        // This anchors the diagonal as the boundary between the two
        // branches and ensures the `>=` half-open interval choice
        // produces the same physical value the `<` branch would
        // give if extended.
        let i = 25;
        let upward_at_diag = model2_layer3_spread_db(i, i);
        let downward_extension = 1.5 * f64::from(0);
        assert_eq!(upward_at_diag, downward_extension);
        assert_eq!(upward_at_diag, 0.0);
    }

    // ---- Table D.5 — Layer I/II coder partition table -------------

    #[test]
    fn coder_partition_d5_has_33_rows() {
        // Spec: Index n = 0..=32 — 33 rows in Table D.5.
        assert_eq!(CODER_PARTITION_TABLE_D5.len(), 33);
    }

    #[test]
    fn coder_partition_d5_indices_are_contiguous_zero_based() {
        // Every row's `index` field equals its position in the array.
        for (pos, row) in CODER_PARTITION_TABLE_D5.iter().enumerate() {
            assert_eq!(
                u16::try_from(pos).unwrap(),
                row.index,
                "row at array position {pos} carries index = {}",
                row.index,
            );
        }
    }

    #[test]
    fn coder_partition_d5_anchor_rows_match_spec() {
        // Verbatim cross-check against the docs file: first row, the
        // last row of the width-0 block, the first row of the width-1
        // block, and the last row.
        assert_eq!(
            CODER_PARTITION_TABLE_D5[0],
            CoderPartitionD5::new(0, 1, 0),
            "row 0 (ωlow_1 = 1, width_0 = 0)",
        );
        assert_eq!(
            CODER_PARTITION_TABLE_D5[12],
            CoderPartitionD5::new(12, 193, 0),
            "row 12 (last width-0 row, ωlow_13 = 193)",
        );
        assert_eq!(
            CODER_PARTITION_TABLE_D5[13],
            CoderPartitionD5::new(13, 209, 1),
            "row 13 (first width-1 row, ωlow_14 = 209)",
        );
        assert_eq!(
            CODER_PARTITION_TABLE_D5[32],
            CoderPartitionD5::new(32, 513, 1),
            "row 32 (last row, ωlow_33 = 513, width_32 = 1)",
        );
    }

    #[test]
    fn coder_partition_d5_omega_boundary_is_strictly_monotonic() {
        // The spec table's ω column is strictly increasing across all
        // 33 rows — boundary lines never repeat or reverse.
        for pair in CODER_PARTITION_TABLE_D5.windows(2) {
            assert!(
                pair[0].omega_boundary < pair[1].omega_boundary,
                "ω monotonicity broken at index {}: {} -> {}",
                pair[0].index,
                pair[0].omega_boundary,
                pair[1].omega_boundary,
            );
        }
    }

    #[test]
    fn coder_partition_d5_stride_is_uniform_16_lines() {
        // The spec table prints a uniform 16-line stride for all 32
        // row transitions (1 -> 17, 17 -> 33, …, 497 -> 513). The
        // module exposes the stride as a `pub const`; this test
        // pins both the constant value AND the uniformity over the
        // whole table.
        assert_eq!(CODER_PARTITION_D5_STRIDE, 16);
        for pair in CODER_PARTITION_TABLE_D5.windows(2) {
            assert_eq!(
                pair[1].omega_boundary - pair[0].omega_boundary,
                CODER_PARTITION_D5_STRIDE,
                "non-uniform stride at index {}",
                pair[0].index,
            );
        }
    }

    #[test]
    fn coder_partition_d5_width_field_partitions_at_row_13() {
        // Rows 0..=12 carry width = 0; rows 13..=32 carry width = 1.
        // The split is exactly at row 13 (the docs file's first
        // width = 1 row) — there is no transitional row.
        for row in &CODER_PARTITION_TABLE_D5[0..=12] {
            assert_eq!(
                row.width, 0,
                "row {} expected width = 0 in lower block",
                row.index,
            );
        }
        for row in &CODER_PARTITION_TABLE_D5[13..=32] {
            assert_eq!(
                row.width, 1,
                "row {} expected width = 1 in upper block",
                row.index,
            );
        }
    }

    #[test]
    fn coder_partition_d5_lookup_returns_each_row_by_index() {
        // `coder_partition_d5(n)` is a thin row accessor — every
        // in-range index round-trips to the indexed array row.
        for row in &CODER_PARTITION_TABLE_D5 {
            assert_eq!(coder_partition_d5(row.index), Some(*row));
        }
    }

    #[test]
    fn coder_partition_d5_lookup_rejects_out_of_range_indices() {
        // Spec range is 0..=32; index 33 and above are not defined.
        assert_eq!(coder_partition_d5(33), None);
        assert_eq!(coder_partition_d5(64), None);
        assert_eq!(coder_partition_d5(u16::MAX), None);
    }

    #[test]
    fn coder_partition_d5_first_row_omega_is_one_based() {
        // The spec's FFT-line column is 1-based: row 0 prints ω = 1,
        // matching the docs file's prose "lower index ωlow_{n+1}".
        // This is a structural pin — a 0-based transcription error
        // would shift every row by one.
        assert_eq!(CODER_PARTITION_TABLE_D5[0].omega_boundary, 1);
    }

    #[test]
    fn coder_partition_d5_top_partition_boundary_is_513() {
        // The top of the table is ω = 513 at row n = 32 — the 1-based
        // FFT-line index of the last analysed line in Models 1 / 2's
        // 1024-sample FFT half-spectrum (lines 1..=513).
        assert_eq!(CODER_PARTITION_TABLE_D5[32].omega_boundary, 513);
        // And the total span of the 32 strides equals (513 - 1).
        let span = CODER_PARTITION_TABLE_D5[32].omega_boundary
            - CODER_PARTITION_TABLE_D5[0].omega_boundary;
        assert_eq!(span, 32 * CODER_PARTITION_D5_STRIDE);
    }

    // ---- Table D.5 — dual-role accessors --------------------------

    #[test]
    fn coder_partition_d5_omega_high_method_renames_omega_boundary() {
        // Per-row method: `omega_high()` is a pure rename of the
        // verbatim `omega_boundary` field with no arithmetic.
        for row in &CODER_PARTITION_TABLE_D5 {
            assert_eq!(row.omega_high(), row.omega_boundary);
        }
    }

    #[test]
    fn coder_partition_d5_omega_low_of_next_method_renames_omega_boundary() {
        // Per-row method: `omega_low_of_next()` is a pure rename of
        // the verbatim `omega_boundary` field with no arithmetic.
        for row in &CODER_PARTITION_TABLE_D5 {
            assert_eq!(row.omega_low_of_next(), row.omega_boundary);
        }
    }

    #[test]
    fn coder_partition_d5_dual_role_methods_return_same_value() {
        // The two role-aware methods carry distinct spec names but
        // expose the same printed integer — the column heading
        // `ωlow_{n+1} / ωhigh_n` literally aliases the cell.
        for row in &CODER_PARTITION_TABLE_D5 {
            assert_eq!(row.omega_high(), row.omega_low_of_next());
        }
    }

    #[test]
    fn coder_partition_d5_omega_high_table_accessor_anchor_rows() {
        // Spec-anchored values: row 0 → ωhigh_0 = 1; row 12 →
        // ωhigh_12 = 193; row 13 → ωhigh_13 = 209; row 32 →
        // ωhigh_32 = 513.
        assert_eq!(coder_partition_d5_omega_high(0), Some(1));
        assert_eq!(coder_partition_d5_omega_high(12), Some(193));
        assert_eq!(coder_partition_d5_omega_high(13), Some(209));
        assert_eq!(coder_partition_d5_omega_high(32), Some(513));
    }

    #[test]
    fn coder_partition_d5_omega_high_table_accessor_matches_omega_boundary_for_all_rows() {
        // The role-aware table accessor must equal the verbatim
        // `omega_boundary` for every in-range index — pure rename.
        for row in &CODER_PARTITION_TABLE_D5 {
            assert_eq!(
                coder_partition_d5_omega_high(row.index),
                Some(row.omega_boundary),
            );
        }
    }

    #[test]
    fn coder_partition_d5_omega_high_table_accessor_rejects_out_of_range() {
        // Spec range is 0..=32; index 33 and above return None.
        assert_eq!(coder_partition_d5_omega_high(33), None);
        assert_eq!(coder_partition_d5_omega_high(64), None);
        assert_eq!(coder_partition_d5_omega_high(u16::MAX), None);
    }

    #[test]
    fn coder_partition_d5_omega_low_table_accessor_anchor_rows() {
        // Spec-anchored values: ωlow_1 = 1 (row 0); ωlow_13 = 193
        // (row 12); ωlow_14 = 209 (row 13); ωlow_33 = 513 (row 32).
        assert_eq!(coder_partition_d5_omega_low(1), Some(1));
        assert_eq!(coder_partition_d5_omega_low(13), Some(193));
        assert_eq!(coder_partition_d5_omega_low(14), Some(209));
        assert_eq!(coder_partition_d5_omega_low(33), Some(513));
    }

    #[test]
    fn coder_partition_d5_omega_low_partition_zero_is_not_in_table() {
        // ωlow_0 — partition 0's own lower boundary — is NOT in
        // Table D.5. The column heading `ωlow_{n+1} / ωhigh_n`
        // shifts row n's value to ωlow_{n+1}, so the table covers
        // ωlow_n only for n ∈ 1..=33. Input n = 0 returns None
        // verbatim — no DOCS-GAP assumption is inserted.
        assert_eq!(coder_partition_d5_omega_low(0), None);
    }

    #[test]
    fn coder_partition_d5_omega_low_rejects_out_of_range() {
        // The table covers ωlow_n only for n ∈ 1..=33; n ≥ 34
        // returns None.
        assert_eq!(coder_partition_d5_omega_low(34), None);
        assert_eq!(coder_partition_d5_omega_low(64), None);
        assert_eq!(coder_partition_d5_omega_low(u16::MAX), None);
    }

    #[test]
    fn coder_partition_d5_omega_low_n_plus_1_equals_omega_high_n() {
        // The column heading `ωlow_{n+1} / ωhigh_n` says these two
        // roles share the same printed integer per row, so for every
        // partition n ∈ 0..=32 the dual-role identity must hold:
        // ωlow_{n+1} == ωhigh_n.
        for n in 0_u16..=32 {
            assert_eq!(
                coder_partition_d5_omega_low(n + 1),
                coder_partition_d5_omega_high(n),
                "ωlow_{} != ωhigh_{}",
                n + 1,
                n,
            );
        }
    }

    // ---- Table D.5 — partition FFT-line range accessor ------------

    #[test]
    fn coder_partition_d5_line_range_anchor_rows() {
        // Spec-anchored partition spans, derived by composing the two
        // dual-role accessors at the four anchor partitions exercised
        // by the step-50 tests:
        //
        //  n = 1:  (ωlow_1, ωhigh_1)   = (1,   17)
        //  n = 13: (ωlow_13, ωhigh_13) = (193, 209)
        //  n = 14: (ωlow_14, ωhigh_14) = (209, 225)
        //  n = 32: (ωlow_32, ωhigh_32) = (497, 513)
        assert_eq!(coder_partition_d5_line_range(1), Some((1, 17)));
        assert_eq!(coder_partition_d5_line_range(13), Some((193, 209)));
        assert_eq!(coder_partition_d5_line_range(14), Some((209, 225)));
        assert_eq!(coder_partition_d5_line_range(32), Some((497, 513)));
    }

    #[test]
    fn coder_partition_d5_line_range_partition_zero_missing_low_boundary() {
        // ωlow_0 is NOT in Table D.5 (the column heading shifts the
        // lower boundary up by one), so the full span of partition 0
        // is not recoverable. The accessor returns None verbatim —
        // no synthetic lower boundary is invented.
        assert_eq!(coder_partition_d5_line_range(0), None);
    }

    #[test]
    fn coder_partition_d5_line_range_partition_thirty_three_missing_high_boundary() {
        // Although ωlow_33 = 513 is present (as row 32's value under
        // its `ωlow_{n+1}` role), ωhigh_33 is NOT — the table tops
        // out at row n = 32. Partition 33's upper boundary is not
        // recoverable from Table D.5 alone, so the accessor returns
        // None verbatim.
        assert_eq!(coder_partition_d5_line_range(33), None);
    }

    #[test]
    fn coder_partition_d5_line_range_rejects_out_of_range() {
        // Indices well above the spec range trivially return None on
        // both component accessors.
        assert_eq!(coder_partition_d5_line_range(34), None);
        assert_eq!(coder_partition_d5_line_range(64), None);
        assert_eq!(coder_partition_d5_line_range(u16::MAX), None);
    }

    #[test]
    fn coder_partition_d5_line_range_low_le_high_for_all_in_range() {
        // For every recoverable partition the lower boundary sits at
        // or below the upper boundary (it sits strictly below in
        // practice; the equality case is allowed by the accessor's
        // contract but doesn't occur in Table D.5).
        for n in 1_u16..=32 {
            let (low, high) = coder_partition_d5_line_range(n).unwrap_or_else(|| {
                panic!("expected Some span for n = {n}");
            });
            assert!(
                low <= high,
                "partition {n}: ωlow = {low} must be ≤ ωhigh = {high}",
            );
        }
    }

    #[test]
    fn coder_partition_d5_line_range_strict_inequality_for_all_in_range() {
        // Stronger structural pin: every span is non-degenerate
        // (low < high). Table D.5's stride is 16 FFT lines per
        // partition and the boundary cells are inclusive at both
        // ends, so a recoverable partition always covers more than
        // a single line.
        for n in 1_u16..=32 {
            let (low, high) = coder_partition_d5_line_range(n).unwrap();
            assert!(
                low < high,
                "partition {n}: span {low}..={high} should be non-degenerate",
            );
        }
    }

    #[test]
    fn coder_partition_d5_line_range_composes_omega_low_and_omega_high() {
        // The accessor is a pure composition of the two step-50
        // dual-role accessors. Pin that contract at every recoverable
        // partition: the returned tuple is exactly
        // (omega_low(n), omega_high(n)) — no rearrangement.
        for n in 1_u16..=32 {
            let low = coder_partition_d5_omega_low(n).unwrap();
            let high = coder_partition_d5_omega_high(n).unwrap();
            assert_eq!(
                coder_partition_d5_line_range(n),
                Some((low, high)),
                "partition {n}: span must compose verbatim",
            );
        }
    }

    #[test]
    fn coder_partition_d5_line_range_uses_stride_plus_one_lines() {
        // The spec table's uniform 16-line stride implies an
        // inclusive span width of `stride + 1` lines per partition
        // (both endpoints are inclusive). Pin this across all 32
        // recoverable partitions as a structural check on the
        // composition.
        for n in 1_u16..=32 {
            let (low, high) = coder_partition_d5_line_range(n).unwrap();
            // The `high - low` open span equals one stride; the
            // inclusive span is `stride + 1` lines (here 17).
            assert_eq!(
                high - low,
                CODER_PARTITION_D5_STRIDE,
                "partition {n}: open span must equal stride",
            );
        }
    }

    #[test]
    fn coder_partition_d5_line_range_partitions_tile_fft_line_band_two_to_513() {
        // The 32 recoverable partition spans tile the FFT-line band
        // `[2, 513]` with adjacent partitions sharing a single
        // boundary line — partition n's ωhigh equals partition n+1's
        // ωlow (the dual-role identity at the table level). Pin both
        // halves: the band's lower edge is partition 1's ωlow (= 1)
        // shifted up by one (since partition 1's `ωlow_1 = 1` is the
        // top edge of partition 0's missing span); and adjacent
        // spans share a boundary line.
        let mut prev_high = None;
        for n in 1_u16..=32 {
            let (low, high) = coder_partition_d5_line_range(n).unwrap();
            if let Some(p) = prev_high {
                assert_eq!(
                    low, p,
                    "partition {n}: ωlow {low} must equal previous partition's ωhigh {p}",
                );
            }
            prev_high = Some(high);
        }
        // The top of the band is partition 32's ωhigh = 513.
        assert_eq!(prev_high, Some(513));
        // And the bottom recoverable line is partition 1's ωlow = 1.
        assert_eq!(coder_partition_d5_line_range(1).map(|t| t.0), Some(1));
    }

    // ---- Table D.5 — width_n accessor -----------------------------

    #[test]
    fn coder_partition_d5_width_anchor_rows() {
        // Spec-anchored values at the four reference rows used by the
        // earlier step's tests: row 0 → 0; row 12 (last width-0 row)
        // → 0; row 13 (first width-1 row) → 1; row 32 (last row) → 1.
        assert_eq!(coder_partition_d5_width(0), Some(0));
        assert_eq!(coder_partition_d5_width(12), Some(0));
        assert_eq!(coder_partition_d5_width(13), Some(1));
        assert_eq!(coder_partition_d5_width(32), Some(1));
    }

    #[test]
    fn coder_partition_d5_width_matches_row_field_for_every_in_range_index() {
        // The accessor is a pure rename of the row's `width` field.
        // Pin that contract across every in-range partition.
        for row in &CODER_PARTITION_TABLE_D5 {
            assert_eq!(
                coder_partition_d5_width(row.index),
                Some(row.width),
                "row {} expected width = {}",
                row.index,
                row.width,
            );
        }
    }

    #[test]
    fn coder_partition_d5_width_is_zero_for_lower_block_one_for_upper_block() {
        // The spec table's `width_n` column is exactly two values:
        // 0 for the lower block (rows 0..=12) and 1 for the upper
        // block (rows 13..=32). The split is a step function at
        // row 13 — no transitional row. Pin this at the table-level
        // accessor (the row-field version is pinned separately
        // above).
        for n in 0_u16..=12 {
            assert_eq!(
                coder_partition_d5_width(n),
                Some(0),
                "partition {n}: expected width = 0 in lower block",
            );
        }
        for n in 13_u16..=32 {
            assert_eq!(
                coder_partition_d5_width(n),
                Some(1),
                "partition {n}: expected width = 1 in upper block",
            );
        }
    }

    #[test]
    fn coder_partition_d5_width_rejects_out_of_range() {
        // Spec range is 0..=32; index 33 and above return None.
        assert_eq!(coder_partition_d5_width(33), None);
        assert_eq!(coder_partition_d5_width(64), None);
        assert_eq!(coder_partition_d5_width(u16::MAX), None);
    }

    #[test]
    fn coder_partition_d5_width_range_is_exactly_zero_or_one() {
        // The spec table never prints a `width_n` other than 0 or 1.
        // Pin this across every in-range partition.
        for n in 0_u16..=32 {
            let w = coder_partition_d5_width(n).unwrap();
            assert!(w == 0 || w == 1, "partition {n}: width {w} must be 0 or 1",);
        }
    }

    #[test]
    fn coder_partition_d5_width_transition_is_a_single_step_at_row_thirteen() {
        // Stronger structural pin: across the 32 consecutive partition
        // pairs `(n, n+1)` for `n ∈ 0..=31`, the `width_n` column
        // changes value at exactly one place — between rows 12 and 13.
        // Anywhere else the value is unchanged. The split is a step
        // function, not a ramp.
        let mut transitions = 0_u16;
        for n in 0_u16..=31 {
            let cur = coder_partition_d5_width(n).unwrap();
            let nxt = coder_partition_d5_width(n + 1).unwrap();
            if cur != nxt {
                transitions += 1;
                // The single transition must be at the 12 → 13 step,
                // going from 0 to 1.
                assert_eq!(n, 12, "unexpected transition at partition {n}");
                assert_eq!(cur, 0, "transition must rise from 0");
                assert_eq!(nxt, 1, "transition must rise to 1");
            }
        }
        assert_eq!(transitions, 1, "expected exactly one width transition");
    }

    #[test]
    fn coder_partition_d5_width_is_orthogonal_to_omega_boundary() {
        // The `width_n` column is structurally orthogonal to the
        // boundary column — the boundary column is strictly monotonic
        // in `n` with a uniform 16-line stride (pinned elsewhere),
        // whereas `width_n` is constant within each of the two blocks.
        // Pin the constant-within-block half here: for every consecutive
        // pair inside one block, width is unchanged even as the
        // boundary cell advances by `CODER_PARTITION_D5_STRIDE`.
        for n in 0_u16..=11 {
            assert_eq!(coder_partition_d5_width(n), coder_partition_d5_width(n + 1));
        }
        for n in 13_u16..=31 {
            assert_eq!(coder_partition_d5_width(n), coder_partition_d5_width(n + 1));
        }
    }

    // ---- Table D.5 — composed partition descriptor ----------------

    #[test]
    fn coder_partition_d5_span_anchor_rows() {
        // Spec-anchored values at four reference partitions:
        //   n = 1  → (ωlow=1,   ωhigh=17,  width=0)  (lower block edge)
        //   n = 12 → (ωlow=177, ωhigh=193, width=0)  (last width-0)
        //   n = 13 → (ωlow=193, ωhigh=209, width=1)  (first width-1)
        //   n = 32 → (ωlow=497, ωhigh=513, width=1)  (table top edge)
        assert_eq!(
            coder_partition_d5_span(1),
            Some(CoderPartitionD5Span {
                index: 1,
                omega_low: 1,
                omega_high: 17,
                width: 0,
            }),
        );
        assert_eq!(
            coder_partition_d5_span(12),
            Some(CoderPartitionD5Span {
                index: 12,
                omega_low: 177,
                omega_high: 193,
                width: 0,
            }),
        );
        assert_eq!(
            coder_partition_d5_span(13),
            Some(CoderPartitionD5Span {
                index: 13,
                omega_low: 193,
                omega_high: 209,
                width: 1,
            }),
        );
        assert_eq!(
            coder_partition_d5_span(32),
            Some(CoderPartitionD5Span {
                index: 32,
                omega_low: 497,
                omega_high: 513,
                width: 1,
            }),
        );
    }

    #[test]
    fn coder_partition_d5_span_rejects_edges_and_out_of_range() {
        // Partition 0's row carries valid `width_n` and `ωhigh_0` cells
        // but no `ωlow_0`; partition 33 has no row at all. Both must
        // return `None`, matching the line-range accessor's range
        // restriction. Out-of-range inputs above 33 also return None.
        assert_eq!(coder_partition_d5_span(0), None);
        assert_eq!(coder_partition_d5_span(33), None);
        assert_eq!(coder_partition_d5_span(34), None);
        assert_eq!(coder_partition_d5_span(64), None);
        assert_eq!(coder_partition_d5_span(u16::MAX), None);
    }

    #[test]
    fn coder_partition_d5_span_composes_underlying_accessors_for_every_in_range_index() {
        // The descriptor is a pure composition — `omega_low`/`omega_high`
        // come from `line_range`, `width` from `width`. Pin that
        // composition across every in-range partition.
        for n in 1_u16..=32 {
            let span = coder_partition_d5_span(n).unwrap();
            let (low, high) = coder_partition_d5_line_range(n).unwrap();
            let width = coder_partition_d5_width(n).unwrap();
            assert_eq!(span.index, n);
            assert_eq!(span.omega_low, low);
            assert_eq!(span.omega_high, high);
            assert_eq!(span.width, width);
        }
    }

    #[test]
    fn coder_partition_d5_span_inclusive_span_is_17_lines_everywhere() {
        // The boundary column advances by a uniform 16-line stride
        // (pinned by `CODER_PARTITION_D5_STRIDE` elsewhere), so every
        // partition's inclusive line span covers exactly 17 lines
        // (`high - low + 1`). Pin that on the composed descriptor too —
        // the composition must not drop or shift either boundary.
        for n in 1_u16..=32 {
            let span = coder_partition_d5_span(n).unwrap();
            let inclusive_len = span.omega_high - span.omega_low + 1;
            assert_eq!(
                inclusive_len,
                CODER_PARTITION_D5_STRIDE + 1,
                "partition {n}: inclusive line span {} expected {}",
                inclusive_len,
                CODER_PARTITION_D5_STRIDE + 1,
            );
        }
    }

    #[test]
    fn coder_partition_d5_span_width_block_structure_is_preserved() {
        // The composition must not perturb the `width_n` block
        // structure: width = 0 across `n ∈ 1..=12` and width = 1
        // across `n ∈ 13..=32`. Note: this descriptor's range starts
        // at 1 (not 0), so the lower-block test starts at 1 — the
        // row-0 `width_n = 0` cell is unreachable through this
        // descriptor.
        for n in 1_u16..=12 {
            let span = coder_partition_d5_span(n).unwrap();
            assert_eq!(span.width, 0, "partition {n}: expected width 0");
        }
        for n in 13_u16..=32 {
            let span = coder_partition_d5_span(n).unwrap();
            assert_eq!(span.width, 1, "partition {n}: expected width 1");
        }
    }

    #[test]
    fn coder_partition_d5_span_tiles_the_band() {
        // The composed descriptor's boundaries must tile: partition
        // `n`'s upper boundary equals partition `n + 1`'s lower
        // boundary, for every adjacent in-range pair. This is a
        // structural consequence of the dual-role boundary column
        // (one printed integer covers both `ωhigh_n` and
        // `ωlow_{n+1}`); the composition preserves it.
        for n in 1_u16..=31 {
            let cur = coder_partition_d5_span(n).unwrap();
            let nxt = coder_partition_d5_span(n + 1).unwrap();
            assert_eq!(
                cur.omega_high, nxt.omega_low,
                "partition {n} tile gap: high {} != next low {}",
                cur.omega_high, nxt.omega_low,
            );
        }
    }

    #[test]
    fn coder_partition_d5_span_index_field_matches_input() {
        // The descriptor's `index` field must echo the input partition
        // number verbatim — no off-by-one and no row-shift bleeding
        // through from the `omega_low` computation.
        for n in 1_u16..=32 {
            let span = coder_partition_d5_span(n).unwrap();
            assert_eq!(
                span.index, n,
                "partition {n}: descriptor index {} should match input",
                span.index,
            );
        }
    }

    #[test]
    fn coder_partition_d5_span_low_is_strictly_less_than_high() {
        // Every recoverable partition has a non-degenerate FFT-line
        // span: `omega_low < omega_high` (the inclusive 17-line span
        // pinned above implies it but pin it explicitly too).
        for n in 1_u16..=32 {
            let span = coder_partition_d5_span(n).unwrap();
            assert!(
                span.omega_low < span.omega_high,
                "partition {n}: low {} should be < high {}",
                span.omega_low,
                span.omega_high,
            );
        }
    }

    // =====================================================================
    // Phase 2 step 54 (r253) — Table D.5 inclusive-line membership
    // predicate `partition_n_contains_line(n, ω)`.
    // =====================================================================

    #[test]
    fn partition_n_contains_line_inclusive_at_both_boundaries() {
        // Spec-anchor rows: the predicate is true at both endpoints of
        // the inclusive boundary range `[ωlow_n, ωhigh_n]` — pinning
        // the inclusive-on-both-ends reading the descriptor inherits
        // from Phase 2 step 50.
        for n in 1_u16..=32 {
            let span = coder_partition_d5_span(n).unwrap();
            assert_eq!(
                partition_n_contains_line(n, span.omega_low),
                Some(true),
                "partition {n}: ωlow_{n} = {} should be inside",
                span.omega_low,
            );
            assert_eq!(
                partition_n_contains_line(n, span.omega_high),
                Some(true),
                "partition {n}: ωhigh_{n} = {} should be inside",
                span.omega_high,
            );
        }
    }

    #[test]
    fn partition_n_contains_line_rejects_just_outside_each_boundary() {
        // The line one step below `ωlow_n` is outside; the line one
        // step above `ωhigh_n` is outside. Pin both directly so a
        // future off-by-one in the inequality is caught immediately.
        for n in 1_u16..=32 {
            let span = coder_partition_d5_span(n).unwrap();
            // Below the lower boundary. For `n = 1`, `ωlow_1 = 1`, so
            // we probe `omega = 0` — still representable in `u16`. For
            // n > 1, ωlow_n > 1 so the predecessor is well-defined too.
            assert!(span.omega_low >= 1, "partition {n}: ωlow precondition");
            assert_eq!(
                partition_n_contains_line(n, span.omega_low - 1),
                Some(false),
                "partition {n}: line just below ωlow_{n} should be outside",
            );
            // Above the upper boundary (no overflow risk in `u16`).
            assert_eq!(
                partition_n_contains_line(n, span.omega_high + 1),
                Some(false),
                "partition {n}: line just above ωhigh_{n} should be outside",
            );
        }
    }

    #[test]
    fn partition_n_contains_line_anchor_lines() {
        // Spec-anchored anchor evaluations from Table D.5 (read via the
        // step 50 dual-role accessors — every value here is verbatim a
        // boundary cell read of row `n - 1` or row `n`):
        //
        //   partition 1: ωlow_1 = 1 (row 0), ωhigh_1 = 17 (row 1) →
        //                contains {1, 17}; excludes 18.
        //   partition 12: ωlow_12 = 177 (row 11), ωhigh_12 = 193 (row 12)
        //                 → contains 185 (interior midpoint).
        //   partition 13: ωlow_13 = 193 (row 12), ωhigh_13 = 209
        //                 (row 13) → contains 200 (interior); 193 is
        //                 the shared boundary line (also in partition
        //                 12) — pinned separately by the tiling test.
        //   partition 32: ωlow_32 = 497 (row 31), ωhigh_32 = 513
        //                 (row 32) → contains 513; excludes 514.
        assert_eq!(partition_n_contains_line(1, 1), Some(true));
        assert_eq!(partition_n_contains_line(1, 17), Some(true));
        assert_eq!(partition_n_contains_line(1, 18), Some(false));

        assert_eq!(partition_n_contains_line(12, 185), Some(true));
        assert_eq!(partition_n_contains_line(12, 177), Some(true));
        assert_eq!(partition_n_contains_line(12, 176), Some(false));

        assert_eq!(partition_n_contains_line(13, 200), Some(true));
        assert_eq!(partition_n_contains_line(13, 209), Some(true));
        assert_eq!(partition_n_contains_line(13, 210), Some(false));

        assert_eq!(partition_n_contains_line(32, 497), Some(true));
        assert_eq!(partition_n_contains_line(32, 513), Some(true));
        assert_eq!(partition_n_contains_line(32, 514), Some(false));
    }

    #[test]
    fn partition_n_contains_line_rejects_partition_index_edges_and_out_of_range() {
        // Inherits the descriptor's range — `n = 0` and `n = 33` are
        // both boundary-table gaps; `n ∈ {34, 64, u16::MAX}` are out of
        // range. The predicate returns `None` for any `omega` at those
        // partition indices; sweep a few representative `omega` values
        // to confirm the answer doesn't depend on the line argument at
        // an unrecoverable partition index.
        for &omega in &[0_u16, 1, 2, 100, 256, 513, 514, 1024, u16::MAX] {
            assert_eq!(partition_n_contains_line(0, omega), None);
            assert_eq!(partition_n_contains_line(33, omega), None);
            assert_eq!(partition_n_contains_line(34, omega), None);
            assert_eq!(partition_n_contains_line(64, omega), None);
            assert_eq!(partition_n_contains_line(u16::MAX, omega), None);
        }
    }

    #[test]
    fn partition_n_contains_line_every_in_band_line_belongs_to_exactly_one_partition() {
        // Tiling property at the line level: across the FFT-line range
        // covered by the table — `[ωlow_1, ωhigh_32] = [2, 513]` — every
        // line index is contained by exactly two partitions for
        // boundary lines and exactly one for interior lines, by
        // construction of the inclusive-on-both-ends reading where
        // `ωhigh_n = ωlow_{n+1}` (the step 53 tiling test pinned the
        // boundary equality directly).
        //
        // Pin both: boundary lines (every `ωhigh_n` for `n ∈ 1..=31`)
        // belong to partitions `n` *and* `n + 1`; interior lines
        // (every line in `(ωlow_n, ωhigh_n)`) belong to exactly
        // partition `n`.
        for n in 1_u16..=31 {
            let span = coder_partition_d5_span(n).unwrap();
            // Boundary line `ωhigh_n` is in partition n.
            assert_eq!(partition_n_contains_line(n, span.omega_high), Some(true));
            // And also in partition n+1 (since `ωlow_{n+1} = ωhigh_n`).
            assert_eq!(
                partition_n_contains_line(n + 1, span.omega_high),
                Some(true),
                "boundary line {} should be in partition {}",
                span.omega_high,
                n + 1,
            );
            // An interior line (the midpoint of the inclusive span) is
            // in partition n only.
            let mid = span.omega_low + (span.omega_high - span.omega_low) / 2;
            assert_eq!(partition_n_contains_line(n, mid), Some(true));
            if n >= 2 {
                assert_eq!(partition_n_contains_line(n - 1, mid), Some(false));
            }
            if n <= 31 {
                assert_eq!(partition_n_contains_line(n + 1, mid), Some(false));
            }
        }
    }

    #[test]
    fn partition_n_contains_line_matches_descriptor_inequality_for_every_in_range_pair() {
        // Pure-composition pin: across every recoverable partition `n`
        // and every FFT-line `ω` in `0..=520` (slightly past the
        // table-wide upper bound `ωhigh_32 = 513` to exercise the
        // out-of-band false branch too), the predicate value equals
        // the inequality `s.omega_low <= ω && ω <= s.omega_high` on
        // the step 53 descriptor `s`. No drift between the two paths.
        for n in 1_u16..=32 {
            let span = coder_partition_d5_span(n).unwrap();
            for omega in 0_u16..=520 {
                let by_predicate = partition_n_contains_line(n, omega).unwrap();
                let by_descriptor = span.omega_low <= omega && omega <= span.omega_high;
                assert_eq!(
                    by_predicate, by_descriptor,
                    "partition {n}, omega {omega}: predicate {by_predicate} != descriptor {by_descriptor}",
                );
            }
        }
    }

    #[test]
    fn partition_n_contains_line_out_of_band_omega_is_false_at_every_in_range_partition() {
        // The table-wide FFT-line domain is `[1, 513]` — partition 1's
        // `ωlow_1 = 2` lower bound and partition 32's `ωhigh_32 = 513`
        // upper bound. Calling the predicate with `omega = 0` (below
        // every partition's `ωlow_n`) is `false` at every in-range
        // `n`; calling with `omega = 514` (above every partition's
        // `ωhigh_n`) is `false` at every in-range `n`. The predicate
        // does not range-check `omega` against the table-wide line
        // domain — it just evaluates the inequality.
        for n in 1_u16..=32 {
            assert_eq!(partition_n_contains_line(n, 0), Some(false));
            assert_eq!(partition_n_contains_line(n, 514), Some(false));
            assert_eq!(partition_n_contains_line(n, 1024), Some(false));
            assert_eq!(partition_n_contains_line(n, u16::MAX), Some(false));
        }
    }

    // =====================================================================
    // Phase 2 step 55 (r254) — Table D.5 row-order iteration helper
    // `coder_partition_d5_spans()`.
    // =====================================================================

    #[test]
    fn coder_partition_d5_spans_yields_thirty_two_descriptors() {
        // The iterator emits exactly one descriptor per recoverable
        // Table D.5 row — `n ∈ 1..=32` is 32 partitions.
        assert_eq!(coder_partition_d5_spans().count(), 32);
    }

    #[test]
    fn coder_partition_d5_spans_yields_row_order() {
        // Pin the row-order property: the iterator's `index` field
        // sequence is exactly `1, 2, …, 32` — no gaps, no repetition,
        // no reordering.
        let indices: Vec<u16> = coder_partition_d5_spans().map(|s| s.index).collect();
        let expected: Vec<u16> = (1_u16..=32).collect();
        assert_eq!(indices, expected);
    }

    #[test]
    fn coder_partition_d5_spans_each_descriptor_matches_table_lookup() {
        // Every yielded descriptor equals the corresponding
        // `coder_partition_d5_span(n)` — the iterator is a pure
        // row-walk of the step 53 accessor and does not invent any
        // new descriptor field values.
        for span in coder_partition_d5_spans() {
            assert_eq!(
                Some(span),
                coder_partition_d5_span(span.index),
                "iterator descriptor at index {} disagrees with table lookup",
                span.index,
            );
        }
    }

    #[test]
    fn coder_partition_d5_spans_skips_boundary_table_gaps() {
        // The two `None`-returning descriptor edges (`n = 0`, `n = 33`)
        // are not emitted — a row-order walk sees the same boundary-
        // table gaps the descriptor does.
        let indices: Vec<u16> = coder_partition_d5_spans().map(|s| s.index).collect();
        assert!(!indices.contains(&0), "iterator should skip n = 0");
        assert!(!indices.contains(&33), "iterator should skip n = 33");
    }

    #[test]
    fn coder_partition_d5_spans_tiles_the_full_band() {
        // The iterator's descriptors collectively cover every FFT line
        // index in the table-wide band `[ωlow_1, ωhigh_32] = [1, 513]`
        // (with shared boundary lines belonging to two consecutive
        // partitions, per the inclusive-on-both-ends reading already
        // pinned by `coder_partition_d5_span_tiles_the_band`).
        //
        // Composed with the step 54 predicate
        // `partition_n_contains_line`, the iterator walks every line
        // in the band exactly as the downstream partition-threshold
        // reduction will.
        let spans: Vec<CoderPartitionD5Span> = coder_partition_d5_spans().collect();
        let first = spans.first().expect("iterator yields at least one span");
        let last = spans.last().expect("iterator yields at least one span");
        assert_eq!(first.omega_low, 1, "first partition's ωlow should be 1");
        assert_eq!(last.omega_high, 513, "last partition's ωhigh should be 513");
        // Adjacent-row tiling identity: `ωhigh_n = ωlow_{n+1}` for
        // every consecutive pair the iterator yields.
        for pair in spans.windows(2) {
            let (a, b) = (&pair[0], &pair[1]);
            assert_eq!(
                a.omega_high, b.omega_low,
                "tiling broken between partition {} and partition {}",
                a.index, b.index,
            );
        }
    }

    #[test]
    fn coder_partition_d5_spans_pairs_with_membership_predicate() {
        // Pin the spec-read pairing pattern: for each yielded span,
        // `partition_n_contains_line(span.index, ω)` agrees with the
        // descriptor's inequality at every line in `0..=520` (sweeping
        // past the upper band edge to exercise the out-of-band branch).
        // This is the pattern the downstream reduction will use; pin
        // the agreement directly so a future drift between the
        // iterator and the predicate fails the build.
        for span in coder_partition_d5_spans() {
            for omega in 0_u16..=520 {
                let by_predicate = partition_n_contains_line(span.index, omega).unwrap();
                let by_descriptor = span.omega_low <= omega && omega <= span.omega_high;
                assert_eq!(
                    by_predicate, by_descriptor,
                    "iterator/predicate disagree at (n = {}, ω = {})",
                    span.index, omega,
                );
            }
        }
    }

    #[test]
    fn coder_partition_d5_spans_is_clone_and_repeatable() {
        // The iterator is cheap to clone (a `Range::Map` over a
        // pure-function map) and yields identical sequences on each
        // walk — pinning the "may be re-iterated" property the
        // downstream reduction relies on for multi-pass walks.
        let first_walk: Vec<CoderPartitionD5Span> = coder_partition_d5_spans().collect();
        let second_walk: Vec<CoderPartitionD5Span> = coder_partition_d5_spans().collect();
        assert_eq!(first_walk, second_walk);
        assert_eq!(first_walk.len(), 32);
    }

    // Phase 2 step 56 (r255) — Table D.5 inverse line→partition
    // lookup. The accessor `first_partition_containing_line(ω)` returns
    // `Some(n)` for the lowest partition whose inclusive boundary
    // range `[ωlow_n, ωhigh_n]` contains `ω`, and `None` for any
    // out-of-band `ω`.

    #[test]
    fn first_partition_returns_none_below_band() {
        // ω = 0 is below the table-wide lower edge ωlow_1 = 1; no
        // partition contains it. Pin the out-of-band None branch on
        // the low side.
        assert_eq!(first_partition_containing_line(0), None);
    }

    #[test]
    fn first_partition_returns_none_above_band() {
        // ω = 514 is one line above the table-wide upper edge
        // ωhigh_32 = 513; no partition contains it. Pin the
        // out-of-band None branch on the high side.
        assert_eq!(first_partition_containing_line(514), None);
        // A clearly far-above-band value reports None too — the
        // accessor doesn't accidentally clamp into the table.
        assert_eq!(first_partition_containing_line(10_000), None);
        // u16::MAX exercises the saturating upper edge.
        assert_eq!(first_partition_containing_line(u16::MAX), None);
    }

    #[test]
    fn first_partition_at_table_wide_lower_edge_is_partition_one() {
        // ω = 1 is partition 1's ωlow_1 (the table-wide lower edge).
        // Partition 1 is the unique container — partition 0 is gone
        // (the descriptor returns None for n = 0, pinned by step 53)
        // and partition 2 starts at ωlow_2 = 17.
        assert_eq!(first_partition_containing_line(1), Some(1));
    }

    #[test]
    fn first_partition_at_table_wide_upper_edge_is_partition_thirty_two() {
        // ω = 513 is partition 32's ωhigh_32 (the table-wide upper
        // edge). Partition 32 is the unique container — there is no
        // partition 33 by the boundary-table gap pinned by step 53.
        assert_eq!(first_partition_containing_line(513), Some(32));
    }

    #[test]
    fn first_partition_at_shared_boundary_picks_lower_index() {
        // Every shared boundary line ω = ωhigh_n = ωlow_{n+1} is a
        // member of BOTH partition n and partition n + 1 under the
        // inclusive-on-both-ends reading pinned by Phase 2 step 54.
        // The inverse accessor breaks the tie in favour of the LOWER
        // partition n — the spec table's row-order presentation prints
        // the boundary cell on row n's line, not on row n + 1's.
        //
        // Test every shared boundary in the table directly: for each
        // n ∈ 1..=31, the value ωhigh_n is a shared boundary and the
        // accessor must return n (not n + 1).
        for n in 1_u16..=31 {
            let high = coder_partition_d5_omega_high(n).expect("step 51 accessor recovers ωhigh_n");
            let next_low =
                coder_partition_d5_omega_low(n + 1).expect("step 51 accessor recovers ωlow_{n+1}");
            assert_eq!(
                high, next_low,
                "step 50 dual-role identity should hold at partition boundary n = {n}",
            );
            assert_eq!(
                first_partition_containing_line(high),
                Some(n),
                "shared boundary ω = {high} should pick lower partition n = {n}, not {}",
                n + 1,
            );
        }
    }

    #[test]
    fn first_partition_at_strict_interior_lines_matches_step_53_descriptor() {
        // For every n ∈ 1..=32 the strictly-interior line
        // ω = ωlow_n + 1 (exists whenever ωhigh_n > ωlow_n, i.e. the
        // partition is not degenerate) is unambiguously in partition
        // n and nowhere else. Pin the accessor against the step 53
        // descriptor directly on those interior lines.
        for span in coder_partition_d5_spans() {
            if span.omega_high > span.omega_low {
                let interior = span.omega_low + 1;
                // The interior line is strictly inside the partition
                // (interior > ωlow, and interior ≤ ωhigh since
                // ωhigh ≥ ωlow + 1 by the strict inequality above),
                // so the accessor must return exactly span.index.
                assert_eq!(
                    first_partition_containing_line(interior),
                    Some(span.index),
                    "interior ω = {interior} of partition {} should map to {}",
                    span.index,
                    span.index,
                );
            }
        }
    }

    #[test]
    fn first_partition_walks_the_full_band_with_no_gaps() {
        // Sweep every ω ∈ [1, 513] across the table-wide FFT-line
        // domain: the accessor must return Some(n) at every line —
        // no gaps in coverage. Pin the table-wide coverage property
        // already verified at the descriptor level by Phase 2 step 55
        // (`…_tiles_the_full_band`), now exposed through the inverse
        // accessor.
        for omega in 1_u16..=513 {
            assert!(
                first_partition_containing_line(omega).is_some(),
                "in-band ω = {omega} should map to some partition",
            );
        }
    }

    #[test]
    fn first_partition_n_agrees_with_step_54_membership_predicate() {
        // Pin the agreement between the inverse accessor and the
        // step 54 membership predicate: if
        // `first_partition_containing_line(ω) = Some(n)`, then
        // `partition_n_contains_line(n, ω) = Some(true)`. The
        // inverse accessor returns a partition that genuinely
        // contains ω under step 54's reading.
        //
        // Sweep slightly past the upper band edge so the out-of-band
        // None branch is exercised; that branch propagates through
        // both accessors consistently (no membership claim is made
        // when no partition contains ω).
        for omega in 0_u16..=520 {
            if let Some(n) = first_partition_containing_line(omega) {
                assert_eq!(
                    partition_n_contains_line(n, omega),
                    Some(true),
                    "first-partition {n} for ω = {omega} should also be a member by step 54",
                );
            }
        }
    }

    #[test]
    fn first_partition_n_is_the_minimum_of_all_containing_partitions() {
        // Pin the "lowest partition first" semantics directly: for
        // every in-band ω, the inverse accessor's answer is the
        // minimum n across all partitions that contain ω under the
        // step 54 membership predicate.
        //
        // Computed from first principles: collect every partition n
        // that contains ω, then assert the inverse accessor returns
        // the min. At shared boundary lines the containing-set has
        // exactly two elements {n, n + 1} and the min is n; at
        // interior lines the containing-set is the singleton {n} and
        // the min is trivially that n.
        for omega in 1_u16..=513 {
            let containing: Vec<u16> = (1_u16..=32)
                .filter(|&n| partition_n_contains_line(n, omega) == Some(true))
                .collect();
            let expected = *containing
                .iter()
                .min()
                .expect("every in-band ω lies in some partition");
            assert_eq!(
                first_partition_containing_line(omega),
                Some(expected),
                "ω = {omega} should map to min partition; containing set = {containing:?}",
            );
        }
    }

    // Phase 2 step 57 (r256) — Table D.5 per-partition FFT-line
    // iterator. The accessor `coder_partition_d5_omega_iter(n)` returns
    // `Some(ωlow_n..=ωhigh_n)` for any `n ∈ 1..=32` and `None` for any
    // n outside that range. The iterator is the per-partition omega
    // walk the Annex D Step 7 partition reduction binds its sum across.

    #[test]
    fn coder_partition_d5_omega_iter_returns_none_for_partition_zero() {
        // Partition 0's lower boundary ωlow_0 is not in Table D.5, so
        // the descriptor returns None for n = 0; the iterator inherits
        // this exactly.
        assert!(coder_partition_d5_omega_iter(0).is_none());
    }

    #[test]
    fn coder_partition_d5_omega_iter_returns_none_for_partition_thirty_three() {
        // Row 33's boundary and its width_n cell are not in Table D.5,
        // so the descriptor returns None for n = 33; the iterator
        // inherits this exactly.
        assert!(coder_partition_d5_omega_iter(33).is_none());
    }

    #[test]
    fn coder_partition_d5_omega_iter_returns_none_for_far_out_of_range_indices() {
        // Two values clearly outside the table's recoverable range —
        // a clearly-large index and the u16 ceiling — both report None,
        // matching the descriptor's None on n outside 1..=32.
        assert!(coder_partition_d5_omega_iter(100).is_none());
        assert!(coder_partition_d5_omega_iter(u16::MAX).is_none());
    }

    #[test]
    fn coder_partition_d5_omega_iter_partition_one_starts_at_table_wide_lower_edge() {
        // Partition 1 spans ωlow_1 = 1 .. ωhigh_1 = 17 (inclusive).
        // The iterator yields exactly that inclusive range.
        let iter = coder_partition_d5_omega_iter(1).expect("partition 1 is recoverable");
        let lines: Vec<u16> = iter.collect();
        // The first emitted line is the table-wide lower edge.
        assert_eq!(lines.first(), Some(&1));
        // The last emitted line is partition 1's ωhigh_1.
        let high = coder_partition_d5_omega_high(1).expect("step 51 recovers ωhigh_1");
        assert_eq!(lines.last(), Some(&high));
    }

    #[test]
    fn coder_partition_d5_omega_iter_partition_thirty_two_ends_at_table_wide_upper_edge() {
        // Partition 32 spans ωlow_32 .. ωhigh_32 = 513 (inclusive).
        // The iterator yields exactly that inclusive range, and its
        // last emitted line is the table-wide upper edge ω = 513.
        let iter = coder_partition_d5_omega_iter(32).expect("partition 32 is recoverable");
        let lines: Vec<u16> = iter.collect();
        let low = coder_partition_d5_omega_low(32).expect("step 51 recovers ωlow_32");
        assert_eq!(lines.first(), Some(&low));
        assert_eq!(lines.last(), Some(&513));
    }

    #[test]
    fn coder_partition_d5_omega_iter_matches_step_51_line_range_for_every_partition() {
        // For every recoverable n ∈ 1..=32 the iterator's
        // (first, last) endpoints must equal (ωlow_n, ωhigh_n) exactly,
        // and the emitted-line count must equal ωhigh_n - ωlow_n + 1
        // (inclusive-on-both-ends arithmetic). Pin both invariants
        // against the step 51 line-range accessor directly.
        for n in 1_u16..=32 {
            let (low, high) =
                coder_partition_d5_line_range(n).expect("step 51 recovers line range");
            let lines: Vec<u16> = coder_partition_d5_omega_iter(n)
                .expect("step 57 iterator is Some for n ∈ 1..=32")
                .collect();
            assert_eq!(
                lines.first(),
                Some(&low),
                "partition {n}: first emitted line = ωlow_{n} = {low}",
            );
            assert_eq!(
                lines.last(),
                Some(&high),
                "partition {n}: last emitted line = ωhigh_{n} = {high}",
            );
            let expected_count = usize::from(high - low + 1);
            assert_eq!(
                lines.len(),
                expected_count,
                "partition {n}: line count = ωhigh_{n} - ωlow_{n} + 1 = {expected_count}",
            );
        }
    }

    #[test]
    fn coder_partition_d5_omega_iter_emits_ascending_lines_with_no_gaps() {
        // Within each partition the iterator is a plain ascending walk
        // from ωlow_n through ωhigh_n with stride 1 — no gaps, no
        // duplicates, strictly increasing.
        for n in 1_u16..=32 {
            let lines: Vec<u16> = coder_partition_d5_omega_iter(n)
                .expect("step 57 iterator is Some for n ∈ 1..=32")
                .collect();
            for window in lines.windows(2) {
                assert_eq!(
                    window[1],
                    window[0] + 1,
                    "partition {n}: consecutive lines must differ by 1, got {} then {}",
                    window[0],
                    window[1],
                );
            }
        }
    }

    #[test]
    fn coder_partition_d5_omega_iter_every_emitted_line_passes_membership_predicate() {
        // Pin agreement with the step 54 membership predicate:
        // every line the iterator emits for partition n must satisfy
        // `partition_n_contains_line(n, ω) = Some(true)`. The two
        // accessors are duals — the iterator yields lines, the
        // predicate tests lines — and they must agree on every
        // (n, ω) pair the iterator produces.
        for n in 1_u16..=32 {
            for omega in coder_partition_d5_omega_iter(n).expect("recoverable n") {
                assert_eq!(
                    partition_n_contains_line(n, omega),
                    Some(true),
                    "iterator emitted ω = {omega} for partition {n}, but membership predicate disagrees",
                );
            }
        }
    }

    #[test]
    fn coder_partition_d5_omega_iter_shared_boundary_line_is_in_both_partitions() {
        // The inclusive-on-both-ends reading pinned by step 50 / step 54
        // means every shared boundary line ω = ωhigh_n = ωlow_{n+1}
        // appears in BOTH partition n's and partition (n+1)'s iterator.
        // Pin this directly at every shared boundary in the table.
        for n in 1_u16..=31 {
            let boundary = coder_partition_d5_omega_high(n).expect("step 51 recovers ωhigh_n");
            let n_lines: Vec<u16> = coder_partition_d5_omega_iter(n)
                .expect("partition n is recoverable")
                .collect();
            let next_lines: Vec<u16> = coder_partition_d5_omega_iter(n + 1)
                .expect("partition n+1 is recoverable")
                .collect();
            assert!(
                n_lines.contains(&boundary),
                "partition {n}: iterator should emit shared boundary ω = {boundary}",
            );
            assert!(
                next_lines.contains(&boundary),
                "partition {}: iterator should also emit shared boundary ω = {boundary}",
                n + 1,
            );
        }
    }

    #[test]
    fn coder_partition_d5_omega_iter_union_covers_table_wide_band() {
        // Union of every partition's emitted lines (1..=32) covers the
        // table-wide FFT-line band [1, 513] with no gaps. Boundary
        // lines may be emitted twice (once per neighbouring partition
        // under the inclusive-on-both-ends reading); the SET of emitted
        // lines must equal the closed interval {1, 2, …, 513} exactly.
        let mut seen = std::collections::BTreeSet::new();
        for n in 1_u16..=32 {
            for omega in coder_partition_d5_omega_iter(n).expect("recoverable n") {
                seen.insert(omega);
            }
        }
        let expected: std::collections::BTreeSet<u16> = (1_u16..=513).collect();
        assert_eq!(seen, expected, "union of partition iterators ≠ [1, 513]");
    }

    #[test]
    fn coder_partition_d5_omega_iter_total_line_count_with_boundary_double_counting() {
        // With the inclusive-on-both-ends reading every interior
        // shared boundary line (n ∈ 1..=31 with ωhigh_n = ωlow_{n+1})
        // gets counted twice when summing partition lengths. Total
        // emitted-line count across n ∈ 1..=32 must equal the band
        // size (513) plus the 31 shared boundaries that are double-
        // counted — i.e. 513 + 31 = 544.
        let total: usize = (1_u16..=32)
            .map(|n| {
                coder_partition_d5_omega_iter(n)
                    .expect("recoverable n")
                    .count()
            })
            .sum();
        assert_eq!(
            total, 544,
            "513 in-band lines + 31 shared boundaries double-counted = 544",
        );
    }

    #[test]
    fn coder_partition_d5_omega_iter_supports_step_seven_per_partition_fold() {
        // End-to-end smoke pin: the step 57 iterator composes naturally
        // with `coder_partition_d5_spans` (step 55) into the spec's
        // per-partition sum-over-lines pattern from clause D.1 Step 7.
        // Use a trivial per-line value (the line index itself, as f64)
        // and assert each partition's sum equals the arithmetic-series
        // closed form sum_{ω=ωlow_n}^{ωhigh_n} ω = (ωlow_n + ωhigh_n)
        // * (ωhigh_n - ωlow_n + 1) / 2. This pins the composition path
        // the downstream Step 8 partition-threshold reduction will use.
        for span in coder_partition_d5_spans() {
            let sum: f64 = coder_partition_d5_omega_iter(span.index)
                .expect("span.index ∈ 1..=32 is recoverable")
                .map(f64::from)
                .sum();
            let n_terms = f64::from(span.omega_high - span.omega_low + 1);
            let endpoints = f64::from(span.omega_low) + f64::from(span.omega_high);
            let expected = endpoints * n_terms * 0.5;
            assert!(
                (sum - expected).abs() < 1.0e-9,
                "partition {}: sum-over-ω = {sum}, expected closed-form {expected}",
                span.index,
            );
        }
    }

    #[test]
    fn coder_partition_d5_ltg_min_returns_none_for_partition_zero() {
        // Partition 0 has no `ωlow_0` in Table D.5, so the step 57
        // iterator is `None` — the step 58 reduction inherits that.
        let v = coder_partition_d5_ltg_min(0, |_| 0.0);
        assert!(v.is_none(), "partition 0 must return None, got {v:?}");
    }

    #[test]
    fn coder_partition_d5_ltg_min_returns_none_for_partition_thirty_three() {
        // Row 33 isn't in Table D.5; the reduction range is undefined.
        let v = coder_partition_d5_ltg_min(33, |_| 0.0);
        assert!(v.is_none(), "partition 33 must return None, got {v:?}");
    }

    #[test]
    fn coder_partition_d5_ltg_min_returns_none_for_far_out_of_range_indices() {
        // Same boundary semantics as the underlying step 57 iterator:
        // any `n` outside `1..=32` is None.
        for n in [100_u16, 1_000, u16::MAX] {
            let v = coder_partition_d5_ltg_min(n, |_| 0.0);
            assert!(v.is_none(), "n = {n} must return None, got {v:?}");
        }
    }

    #[test]
    fn coder_partition_d5_ltg_min_constant_ltg_returns_that_constant() {
        // If LTg(ω) is the same constant C at every line, the per-
        // partition minimum is C for every partition.
        const C: f64 = -3.5;
        for span in coder_partition_d5_spans() {
            let v = coder_partition_d5_ltg_min(span.index, |_| C)
                .expect("span.index ∈ 1..=32 is recoverable");
            assert!(
                (v - C).abs() < 1.0e-12,
                "partition {}: LTmin = {v}, expected constant {C}",
                span.index,
            );
        }
    }

    #[test]
    fn coder_partition_d5_ltg_min_identity_ltg_returns_omega_low() {
        // If LTg(ω) = ω as f64, the per-partition minimum is exactly
        // ωlow_n (the lowest line in the partition's inclusive range).
        for span in coder_partition_d5_spans() {
            let v = coder_partition_d5_ltg_min(span.index, f64::from)
                .expect("span.index ∈ 1..=32 is recoverable");
            let expected = f64::from(span.omega_low);
            assert!(
                (v - expected).abs() < 1.0e-12,
                "partition {}: LTmin = {v}, expected ωlow_n = {expected}",
                span.index,
            );
        }
    }

    #[test]
    fn coder_partition_d5_ltg_min_negative_identity_returns_omega_high() {
        // If LTg(ω) = -ω as f64, the per-partition minimum is exactly
        // -ωhigh_n (the highest line in the partition's inclusive range
        // produces the most-negative value).
        for span in coder_partition_d5_spans() {
            let v = coder_partition_d5_ltg_min(span.index, |omega| -f64::from(omega))
                .expect("span.index ∈ 1..=32 is recoverable");
            let expected = -f64::from(span.omega_high);
            assert!(
                (v - expected).abs() < 1.0e-12,
                "partition {}: LTmin = {v}, expected -ωhigh_n = {expected}",
                span.index,
            );
        }
    }

    #[test]
    fn coder_partition_d5_ltg_min_single_dip_pulls_partition_minimum() {
        // The conservative-bit-allocation reading: a single FFT line
        // dipping below the partition's average threshold pulls the
        // whole partition's LTmin down to that line's level. Place a
        // -100 dB dip at the middle line of each partition and assert
        // the partition's LTmin is exactly -100 dB (not the +0 dB
        // baseline of every other line).
        const DIP_DB: f64 = -100.0;
        for span in coder_partition_d5_spans() {
            let middle = (span.omega_low + span.omega_high) / 2;
            let v =
                coder_partition_d5_ltg_min(
                    span.index,
                    |omega| if omega == middle { DIP_DB } else { 0.0 },
                )
                .expect("span.index ∈ 1..=32 is recoverable");
            assert!(
                (v - DIP_DB).abs() < 1.0e-12,
                "partition {} with dip at ω={middle}: LTmin = {v}, expected {DIP_DB}",
                span.index,
            );
        }
    }

    #[test]
    fn coder_partition_d5_ltg_min_matches_explicit_per_line_fold() {
        // Cross-check: the accessor returns the same value as an
        // explicit fold over the step 57 iterator with the same
        // callback. This pins the accessor as a strict composition of
        // step 57 + min-fold, no spec arithmetic introduced.
        // Use a deterministic non-trivial per-line value: ω * 0.7 - 13.
        let ltg = |omega: u16| f64::from(omega) * 0.7 - 13.0;
        for span in coder_partition_d5_spans() {
            let via_accessor = coder_partition_d5_ltg_min(span.index, ltg)
                .expect("span.index ∈ 1..=32 is recoverable");
            let via_explicit = coder_partition_d5_omega_iter(span.index)
                .expect("span.index ∈ 1..=32 is recoverable")
                .map(ltg)
                .fold(f64::INFINITY, f64::min);
            assert!(
                (via_accessor - via_explicit).abs() < 1.0e-12,
                "partition {}: accessor {via_accessor} != explicit fold {via_explicit}",
                span.index,
            );
        }
    }

    #[test]
    fn coder_partition_d5_ltg_min_shared_boundary_pulls_both_neighbours() {
        // Boundary-semantics pin: the step 57 iterator emits the
        // shared boundary `ωhigh_n = ωlow_{n+1}` to both adjacent
        // partitions. Place a single dip at every shared boundary
        // line and verify partition n AND partition n+1 both record
        // the dip as their LTmin.
        const DIP_DB: f64 = -50.0;
        for span_n in coder_partition_d5_spans().filter(|s| s.index < 32) {
            let shared = span_n.omega_high; // shared with partition n+1
            let f = |omega: u16| if omega == shared { DIP_DB } else { 0.0 };
            let ltmin_n = coder_partition_d5_ltg_min(span_n.index, f)
                .expect("span_n.index ∈ 1..=32 is recoverable");
            let ltmin_n_plus_1 = coder_partition_d5_ltg_min(span_n.index + 1, f)
                .expect("span_n.index + 1 ∈ 1..=32 is recoverable");
            assert!(
                (ltmin_n - DIP_DB).abs() < 1.0e-12,
                "partition {} (shared with {}+1 at ω={shared}): LTmin = {ltmin_n}, expected {DIP_DB}",
                span_n.index,
                span_n.index,
            );
            assert!(
                (ltmin_n_plus_1 - DIP_DB).abs() < 1.0e-12,
                "partition {} (shared with {}-1 at ω={shared}): LTmin = {ltmin_n_plus_1}, expected {DIP_DB}",
                span_n.index + 1,
                span_n.index + 1,
            );
        }
    }

    #[test]
    fn coder_partition_d5_ltg_min_composes_with_step_seven_global_threshold() {
        // End-to-end composition pin: feed the step 58 reducer the
        // actual Step 7 `global_masking_threshold_db` value at every
        // FFT line and assert it returns the minimum of those values
        // across the partition. The line-to-Hz mapping for the per-
        // partition coder-table (Table D.5) isn't pinned by this
        // accessor — Step 7 is being exercised here as the black-box
        // dB callback; the per-line frequency mapping is the caller's
        // responsibility (Step 1's FFT-bin → Hz mapping, currently
        // gated on the PNG-only Table D.1 transcription gap).
        //
        // Use a synthetic mapping: line ω → Hz = ω * 50 (a stand-in
        // for the FFT-bin-to-Hz mapping until Step 1 lands). Place
        // one tonal masker at z = 5 Bark with SPL = 60 dB and verify
        // every partition's LTmin matches the minimum of Step 7's
        // own LTg values at the partition's FFT lines.
        let masker = Masker {
            kind: MaskerKind::Tonal,
            z_bark: 5.0,
            spl_db: 60.0,
        };
        let maskers = [masker];
        // Synthetic z(ω) mapping: ω → ω * 0.05 Bark. The mapping is a
        // stand-in for the Table D.1 lookup; what matters here is that
        // it's a monotonically-increasing total function from FFT
        // line → Bark, exercising the masking-function piecewise
        // branches across the partition.
        let z_of = |omega: u16| f64::from(omega) * 0.05;
        let ltg = |omega: u16| {
            let z_i = z_of(omega);
            // Use a flat LTq of -5 dB (stand-in until Step 1 lands the
            // line → Hz mapping that feeds `ltq_db_at_hz`).
            global_masking_threshold_db(&maskers, z_i, -5.0)
        };
        for span in coder_partition_d5_spans() {
            let via_step58 = coder_partition_d5_ltg_min(span.index, ltg)
                .expect("span.index ∈ 1..=32 is recoverable");
            // Spec-faithful explicit fold via step 57's iterator.
            let via_explicit = coder_partition_d5_omega_iter(span.index)
                .expect("span.index ∈ 1..=32 is recoverable")
                .map(ltg)
                .fold(f64::INFINITY, f64::min);
            assert!(
                (via_step58 - via_explicit).abs() < 1.0e-9,
                "partition {}: step-58 reduction {via_step58} != explicit fold {via_explicit}",
                span.index,
            );
            // Sanity: every partition is finite (no NaN/inf escape).
            assert!(
                via_step58.is_finite(),
                "partition {}: LTmin = {via_step58} is not finite",
                span.index,
            );
        }
    }

    // ---------- Phase 2 step 59 / r258 — row-order LTmin vector ----------

    #[test]
    fn coder_partition_d5_ltg_min_row_order_constant_callback_fills_every_cell() {
        // A constant LTg ≡ C callback must produce [C; 32]: every
        // partition's minimum over a flat dB curve is the constant.
        const C: f64 = -12.5;
        let v = coder_partition_d5_ltg_min_row_order(|_| C);
        assert_eq!(v.len(), 32);
        for (i, x) in v.iter().enumerate() {
            assert!(
                (x - C).abs() < 1.0e-9,
                "index {i} (partition {}): got {x}, expected {C}",
                i + 1,
            );
        }
    }

    #[test]
    fn coder_partition_d5_ltg_min_row_order_index_zero_holds_partition_one() {
        // Pin the 0-based-array / 1-based-partition convention:
        // out[0] = LTmin_1 (partition 1's reduction, not partition 0's).
        // Use an identity LTg(ω) = ω — partition 1's minimum is
        // ωlow_1 (table-wide lower edge ω = 1) and partition 32's
        // minimum is ωlow_32. Verify against the per-partition
        // accessor for every row.
        let v = coder_partition_d5_ltg_min_row_order(f64::from);
        for (i, &got) in v.iter().enumerate() {
            let n = (i + 1) as u16;
            let expected =
                coder_partition_d5_ltg_min(n, f64::from).expect("n ∈ 1..=32 is recoverable");
            assert!(
                (got - expected).abs() < 1.0e-9,
                "index {i} (partition {n}): row-order vec = {got}, per-partition = {expected}",
            );
        }
        // Spot-check: out[0] is partition 1's reduction, which is
        // ωlow_1 = 1 for the identity callback (table-wide lower
        // edge pinned at step 51).
        let (lo_1, _hi_1) = coder_partition_d5_line_range(1).unwrap();
        assert!(
            (v[0] - f64::from(lo_1)).abs() < 1.0e-9,
            "out[0] = {} should equal ωlow_1 = {lo_1} under identity LTg",
            v[0],
        );
    }

    #[test]
    fn coder_partition_d5_ltg_min_row_order_matches_per_partition_for_arbitrary_callback() {
        // Strict-composition pin: the row-order vector must agree
        // element-by-element with a manual loop calling the step 58
        // per-partition reducer for n ∈ 1..=32. Use a non-trivial
        // callback to exercise the broadcast under realistic dB
        // values.
        let ltg = |omega: u16| f64::from(omega) * 0.7 - 13.0;
        let v = coder_partition_d5_ltg_min_row_order(ltg);
        for n in 1_u16..=32 {
            let expected = coder_partition_d5_ltg_min(n, ltg).expect("n ∈ 1..=32 is recoverable");
            let got = v[(n - 1) as usize];
            assert!(
                (got - expected).abs() < 1.0e-9,
                "partition {n}: row-order = {got}, per-partition = {expected}",
            );
        }
    }

    #[test]
    fn coder_partition_d5_ltg_min_row_order_returns_exactly_thirty_two_elements() {
        // The row-order vector covers partitions 1..=32 (Phase 2 step
        // 58 returns None for n = 0 and n = 33); the array length is
        // exactly 32.
        let v = coder_partition_d5_ltg_min_row_order(|_| 0.0);
        assert_eq!(v.len(), 32);
    }

    #[test]
    fn coder_partition_d5_ltg_min_row_order_all_cells_finite_for_finite_callback() {
        // The f64::INFINITY initialisation must be replaced in every
        // cell — a finite LTg callback should produce 32 finite values
        // (no inf leak from the seed).
        let v = coder_partition_d5_ltg_min_row_order(|omega| -f64::from(omega));
        for (i, x) in v.iter().enumerate() {
            assert!(
                x.is_finite(),
                "index {i} (partition {}): got non-finite {x}",
                i + 1,
            );
        }
    }

    #[test]
    fn coder_partition_d5_ltg_min_row_order_single_dip_only_affects_target_partition() {
        // Single-assignment binning regression pin: a -100 dB dip at
        // the *middle* line of partition `target` (not on a shared
        // boundary) must pull down only the target partition's LTmin
        // in the row-order vector and leave every other partition at
        // the baseline.
        const BASELINE: f64 = 5.0;
        const DIP: f64 = -100.0;
        // Use partition 5 (a typical mid-table partition); its line
        // range is (ωlow_5, ωhigh_5) = (65, 80) from Table D.5.
        let target: u16 = 5;
        let (lo, hi) = coder_partition_d5_line_range(target).unwrap();
        let middle = lo + (hi - lo) / 2;
        // Ensure middle is interior (not on either inclusive end) so
        // no shared-boundary spill into neighbouring partitions.
        assert!(middle > lo && middle < hi, "middle line must be interior");
        let v = coder_partition_d5_ltg_min_row_order(
            |omega| if omega == middle { DIP } else { BASELINE },
        );
        for (i, &got) in v.iter().enumerate() {
            let n = (i + 1) as u16;
            let expected = if n == target { DIP } else { BASELINE };
            assert!(
                (got - expected).abs() < 1.0e-9,
                "partition {n}: got {got}, expected {expected}",
            );
        }
    }

    #[test]
    fn coder_partition_d5_ltg_min_row_order_shared_boundary_dip_pulls_both_neighbours() {
        // Shared-boundary semantics pin: a -50 dB dip placed exactly
        // on a shared boundary line ωhigh_n = ωlow_{n+1} must pull
        // both partitions n and n+1 to -50 dB in the row-order vector
        // (the conservative-bit-allocation reading inherited from
        // step 58). Use the boundary between partitions 5 and 6:
        // ωhigh_5 = ωlow_6.
        const BASELINE: f64 = 10.0;
        const DIP: f64 = -50.0;
        let target: u16 = 5;
        let (_lo_n, hi_n) = coder_partition_d5_line_range(target).unwrap();
        let (lo_n1, _hi_n1) = coder_partition_d5_line_range(target + 1).unwrap();
        // The dual-role ωlow_{n+1} / ωhigh_n column means the shared
        // boundary line is precisely hi_n = lo_n1.
        assert_eq!(hi_n, lo_n1, "shared-boundary precondition");
        let v = coder_partition_d5_ltg_min_row_order(
            |omega| if omega == hi_n { DIP } else { BASELINE },
        );
        // Both adjacent partitions register the dip.
        assert!(
            (v[(target - 1) as usize] - DIP).abs() < 1.0e-9,
            "partition {target}: got {}, expected {DIP}",
            v[(target - 1) as usize],
        );
        assert!(
            (v[target as usize] - DIP).abs() < 1.0e-9,
            "partition {}: got {}, expected {DIP}",
            target + 1,
            v[target as usize],
        );
        // Non-adjacent partitions stay at baseline.
        for (i, &got) in v.iter().enumerate() {
            let n = (i + 1) as u16;
            if n == target || n == target + 1 {
                continue;
            }
            assert!(
                (got - BASELINE).abs() < 1.0e-9,
                "partition {n}: got {got}, expected baseline {BASELINE}",
            );
        }
    }

    #[test]
    fn coder_partition_d5_ltg_min_row_order_composes_with_step_seven_global_threshold() {
        // End-to-end composition pin: feed the row-order vector
        // builder the Step 7 `global_masking_threshold_db` value at
        // every FFT line (with one tonal masker at z = 5 Bark, SPL =
        // 60 dB and a synthetic z(ω) = ω · 0.05 Bark stand-in until
        // Step 1's FFT-bin → Hz table lands). Every cell of the
        // row-order vector must agree with the explicit per-line
        // fold via step 57's iterator.
        let masker = Masker {
            kind: MaskerKind::Tonal,
            z_bark: 5.0,
            spl_db: 60.0,
        };
        let maskers = [masker];
        let z_of = |omega: u16| f64::from(omega) * 0.05;
        let ltg = |omega: u16| {
            let z_i = z_of(omega);
            global_masking_threshold_db(&maskers, z_i, -5.0)
        };
        let v = coder_partition_d5_ltg_min_row_order(ltg);
        for n in 1_u16..=32 {
            let expected = coder_partition_d5_omega_iter(n)
                .expect("n ∈ 1..=32 is recoverable")
                .map(ltg)
                .fold(f64::INFINITY, f64::min);
            let got = v[(n - 1) as usize];
            assert!(
                (got - expected).abs() < 1.0e-9,
                "partition {n}: row-order = {got}, explicit fold = {expected}",
            );
            assert!(
                got.is_finite(),
                "partition {n}: LTmin = {got} is not finite"
            );
        }
    }

    #[test]
    fn coder_partition_d5_ltg_min_row_order_walks_partitions_in_ascending_order() {
        // The row-order vector reflects ascending-`n` order: feed an
        // identity callback and verify out[i] is monotonically non-
        // decreasing across i ∈ 0..31 (every ωlow_n grows with n by
        // the strictly-monotonic boundary pinned at step 50).
        let v = coder_partition_d5_ltg_min_row_order(f64::from);
        for w in v.windows(2) {
            assert!(
                w[0] <= w[1],
                "row-order should be non-decreasing under identity LTg: {} > {}",
                w[0],
                w[1],
            );
        }
    }

    #[test]
    fn coder_partition_d5_ltg_min_row_order_endpoints_match_table_d5_edges() {
        // Pin the table-wide edges via the identity callback:
        // out[0]  = ωlow_1  (partition 1, table-wide lower edge)
        // out[31] = ωlow_32 (partition 32, the last recoverable
        //                    partition's lower line)
        // The minimum over an inclusive range under f64::from is the
        // range's lower endpoint.
        let v = coder_partition_d5_ltg_min_row_order(f64::from);
        let (lo_1, _) = coder_partition_d5_line_range(1).unwrap();
        let (lo_32, _) = coder_partition_d5_line_range(32).unwrap();
        assert!(
            (v[0] - f64::from(lo_1)).abs() < 1.0e-9,
            "out[0] = {} should equal ωlow_1 = {lo_1}",
            v[0],
        );
        assert!(
            (v[31] - f64::from(lo_32)).abs() < 1.0e-9,
            "out[31] = {} should equal ωlow_32 = {lo_32}",
            v[31],
        );
    }

    #[test]
    fn coder_partition_d5_ltg_min_row_order_negative_identity_returns_omega_high_per_row() {
        // Negative-identity LTg(ω) = -ω callback: each partition's
        // minimum is the *most negative* line, which is -ωhigh_n
        // (the inclusive upper endpoint). Verify per-row in the
        // 0-based vector matches -ωhigh_{i+1}.
        let v = coder_partition_d5_ltg_min_row_order(|omega| -f64::from(omega));
        for n in 1_u16..=32 {
            let (_lo, hi) = coder_partition_d5_line_range(n).unwrap();
            let got = v[(n - 1) as usize];
            let expected = -f64::from(hi);
            assert!(
                (got - expected).abs() < 1.0e-9,
                "partition {n}: got {got}, expected -ωhigh_n = {expected}",
            );
        }
    }

    // ---------- Phase 2 step 60 / r259 — row-order width vector ----------

    #[test]
    fn coder_partition_d5_width_row_order_returns_exactly_thirty_two_elements() {
        // Vector size is pinned by the Table D.5 transcription: 32
        // recoverable partitions n ∈ 1..=32 (partition 0 excluded).
        let v = coder_partition_d5_width_row_order();
        assert_eq!(v.len(), 32);
    }

    #[test]
    fn coder_partition_d5_width_row_order_lower_block_is_zero() {
        // Spec rule: rows n ∈ 1..=12 carry width_n = 0. Array indices
        // 0..=11 hold partitions 1..=12 in 0-based form.
        let v = coder_partition_d5_width_row_order();
        for (i, &w) in v.iter().enumerate().take(12) {
            assert_eq!(
                w,
                0,
                "array index {i} (partition n = {}) should be 0",
                i + 1,
            );
        }
    }

    #[test]
    fn coder_partition_d5_width_row_order_upper_block_is_one() {
        // Spec rule: rows n ∈ 13..=32 carry width_n = 1. Array indices
        // 12..=31 hold partitions 13..=32 in 0-based form.
        let v = coder_partition_d5_width_row_order();
        for (i, &w) in v.iter().enumerate().skip(12) {
            assert_eq!(
                w,
                1,
                "array index {i} (partition n = {}) should be 1",
                i + 1,
            );
        }
    }

    #[test]
    fn coder_partition_d5_width_row_order_transition_is_a_single_step_at_index_twelve() {
        // The 0 → 1 transition is a single step at array index 12
        // (partition 13). No partition holds an intermediate value
        // (the column is binary 0/1).
        let v = coder_partition_d5_width_row_order();
        assert_eq!(v[11], 0, "array index 11 (partition 12) should be 0");
        assert_eq!(v[12], 1, "array index 12 (partition 13) should be 1");
    }

    #[test]
    fn coder_partition_d5_width_row_order_every_cell_is_zero_or_one() {
        // The width_n column carries only 0 or 1 (binary per
        // transcription). Verify no cell holds any other value.
        let v = coder_partition_d5_width_row_order();
        for (i, &w) in v.iter().enumerate() {
            assert!(
                w == 0 || w == 1,
                "array index {i} carries non-binary width {w}",
            );
        }
    }

    #[test]
    fn coder_partition_d5_width_row_order_matches_per_partition_lookup() {
        // Strict-composition cross-check: each cell equals the step
        // 52 per-partition accessor `coder_partition_d5_width(n)`
        // applied at n = i + 1. The row-order vector is exactly that
        // broadcast.
        let v = coder_partition_d5_width_row_order();
        for n in 1_u16..=32 {
            let direct = coder_partition_d5_width(n).expect("n ∈ 1..=32 is recoverable by step 52");
            let from_row = v[(n - 1) as usize];
            assert_eq!(
                from_row, direct,
                "partition {n}: row-order vector {from_row} vs direct lookup {direct}",
            );
        }
    }

    #[test]
    fn coder_partition_d5_width_row_order_matches_full_table_literal() {
        // Pin the exact 32-element vector: twelve zeros followed by
        // twenty ones. Any future change to Table D.5's width_n column
        // would surface here as a literal mismatch (independent of the
        // step 52 / step 55 underlying accessors).
        let v = coder_partition_d5_width_row_order();
        let expected: [u16; 32] = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // partitions 1..=12
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 13..=32
        ];
        assert_eq!(v, expected);
    }

    #[test]
    fn coder_partition_d5_width_row_order_endpoints_match_table_d5_edges() {
        // Table-wide endpoint pin: array index 0 holds partition 1's
        // width (= 0, lower-block edge); array index 31 holds
        // partition 32's width (= 1, upper-block edge).
        let v = coder_partition_d5_width_row_order();
        assert_eq!(v[0], 0, "array index 0 (partition 1) should be 0");
        assert_eq!(v[31], 1, "array index 31 (partition 32) should be 1");
    }

    #[test]
    fn coder_partition_d5_width_row_order_sum_matches_upper_block_count() {
        // Sum of every cell equals the number of partitions in the
        // upper block (each contributing 1; the lower block
        // contributes 0). The upper block is n ∈ 13..=32 — twenty
        // partitions — so the sum is 20.
        let v = coder_partition_d5_width_row_order();
        let total: u32 = v.iter().map(|&w| u32::from(w)).sum();
        assert_eq!(total, 20);
    }

    #[test]
    fn coder_partition_d5_width_row_order_is_idempotent_across_calls() {
        // The function has no run-time inputs and reads only Table
        // D.5's static width_n column. Every call returns the same
        // vector — verify two back-to-back calls agree.
        let a = coder_partition_d5_width_row_order();
        let b = coder_partition_d5_width_row_order();
        assert_eq!(a, b);
    }

    #[test]
    fn coder_partition_d5_width_row_order_is_non_decreasing() {
        // The width_n column rises monotonically from 0 (rows 1..=12)
        // to 1 (rows 13..=32) with a single step. Verify the vector
        // is non-decreasing in 0-based array order.
        let v = coder_partition_d5_width_row_order();
        for i in 1_usize..32 {
            assert!(
                v[i] >= v[i - 1],
                "non-monotone at index {i}: v[{}] = {} > v[{}] = {}",
                i - 1,
                v[i - 1],
                i,
                v[i],
            );
        }
    }

    #[test]
    fn coder_partition_d5_width_row_order_walks_partitions_in_ascending_order() {
        // The row-order iterator visits every recoverable partition
        // exactly once in ascending n order. Verify by reconstructing
        // the vector via a manual ascending walk and comparing it to
        // the function's output.
        let v = coder_partition_d5_width_row_order();
        let mut manual = [0u16; 32];
        for n in 1_u16..=32 {
            manual[(n - 1) as usize] = coder_partition_d5_width(n).unwrap();
        }
        assert_eq!(v, manual);
    }

    // ---------- Phase 2 step 61 / r260 — paired (LTmin, width) row-order vector ----------

    #[test]
    fn coder_partition_d5_reduction_row_order_returns_exactly_thirty_two_pairs() {
        // The paired vector pins the same 32-element length as steps
        // 59 and 60 — partition 0 is excluded, partitions 1..=32 are
        // each represented exactly once.
        let v = coder_partition_d5_reduction_row_order(|_| 0.0);
        assert_eq!(v.len(), 32);
    }

    #[test]
    fn coder_partition_d5_reduction_row_order_constant_ltg_fills_ltmin_with_constant() {
        // For a constant LTg(ω) = c the per-partition min is c for
        // every partition (Phase 2 step 58 inherits this). Verify the
        // paired vector's ltmin_db column carries c at every row.
        let c = -7.25_f64;
        let v = coder_partition_d5_reduction_row_order(|_| c);
        for (i, pair) in v.iter().enumerate() {
            assert!(
                (pair.ltmin_db - c).abs() < 1.0e-12,
                "row {i}: ltmin_db {} != constant {c}",
                pair.ltmin_db,
            );
        }
    }

    #[test]
    fn coder_partition_d5_reduction_row_order_ltmin_column_matches_step_59() {
        // Strict-composition cross-check: the paired vector's
        // ltmin_db column must equal Phase 2 step 59's row-order LTmin
        // vector for the same callback (the paired accessor is a pure
        // zip — neither column influences the other's computation).
        // Use a non-trivial line-dependent callback to exercise the
        // per-partition reduction.
        let cb = |omega: u16| f64::from(omega).sin();
        let paired = coder_partition_d5_reduction_row_order(cb);
        let step59 = coder_partition_d5_ltg_min_row_order(cb);
        for i in 0..32 {
            assert!(
                (paired[i].ltmin_db - step59[i]).abs() < 1.0e-12,
                "row {i}: paired ltmin_db {} != step 59 {}",
                paired[i].ltmin_db,
                step59[i],
            );
        }
    }

    #[test]
    fn coder_partition_d5_reduction_row_order_width_column_matches_step_60() {
        // The paired vector's width_n column must equal Phase 2 step
        // 60's row-order width vector independent of the callback
        // (the width column is a static Table D.5 column — no run-
        // time inputs). Verify under two different callbacks.
        let widths = coder_partition_d5_width_row_order();
        let a = coder_partition_d5_reduction_row_order(|_| 0.0);
        let b = coder_partition_d5_reduction_row_order(f64::from);
        for i in 0..32 {
            assert_eq!(a[i].width_n, widths[i]);
            assert_eq!(b[i].width_n, widths[i]);
        }
    }

    #[test]
    fn coder_partition_d5_reduction_row_order_width_invariant_across_callbacks() {
        // Structural orthogonality: the width column is fully
        // determined by the static Table D.5 column and does not
        // depend on the caller's `LTg(ω)`. Verify directly that two
        // callbacks produce identical width columns.
        let a = coder_partition_d5_reduction_row_order(|_| 0.0);
        let b = coder_partition_d5_reduction_row_order(|omega| f64::from(omega) * 3.0 - 1.0);
        for i in 0..32 {
            assert_eq!(
                a[i].width_n, b[i].width_n,
                "row {i}: width column varied across callbacks ({} vs {})",
                a[i].width_n, b[i].width_n,
            );
        }
    }

    #[test]
    fn coder_partition_d5_reduction_row_order_width_matches_full_table_literal() {
        // Pin the width column of the paired vector against the
        // verbatim Table D.5 literal — twelve zeros followed by
        // twenty ones. Any future change to Table D.5's width_n
        // column would surface here independently of step 60.
        let v = coder_partition_d5_reduction_row_order(|_| 0.0);
        let widths: [u16; 32] = core::array::from_fn(|i| v[i].width_n);
        let expected: [u16; 32] = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // partitions 1..=12
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 13..=32
        ];
        assert_eq!(widths, expected);
    }

    #[test]
    fn coder_partition_d5_reduction_row_order_identity_ltg_returns_omega_low_per_row() {
        // For LTg(ω) = ω the per-partition minimum lies at ωlow_n
        // (the inclusive lower endpoint of the partition's FFT-line
        // range). Verify the paired vector's ltmin_db column equals
        // f64::from(ωlow_{i+1}) per row.
        let v = coder_partition_d5_reduction_row_order(f64::from);
        for n in 1_u16..=32 {
            let (lo, _hi) = coder_partition_d5_line_range(n).unwrap();
            let got = v[(n - 1) as usize].ltmin_db;
            let expected = f64::from(lo);
            assert!(
                (got - expected).abs() < 1.0e-9,
                "partition {n}: paired ltmin_db {got} != ωlow_n = {expected}",
            );
        }
    }

    #[test]
    fn coder_partition_d5_reduction_row_order_negative_identity_returns_omega_high_per_row() {
        // For LTg(ω) = -ω the per-partition minimum is the *most
        // negative* line, which is -ωhigh_n (inclusive upper
        // endpoint). Verify the paired vector's ltmin_db column
        // matches -ωhigh_{i+1} per row.
        let v = coder_partition_d5_reduction_row_order(|omega| -f64::from(omega));
        for n in 1_u16..=32 {
            let (_lo, hi) = coder_partition_d5_line_range(n).unwrap();
            let got = v[(n - 1) as usize].ltmin_db;
            let expected = -f64::from(hi);
            assert!(
                (got - expected).abs() < 1.0e-9,
                "partition {n}: paired ltmin_db {got} != -ωhigh_n = {expected}",
            );
        }
    }

    #[test]
    fn coder_partition_d5_reduction_row_order_transition_pair_at_index_twelve() {
        // The width column transitions 0 → 1 between array indices 11
        // and 12 (partitions 12 and 13). Pin the boundary in the
        // paired vector: out[11].width_n = 0, out[12].width_n = 1.
        let v = coder_partition_d5_reduction_row_order(|_| 0.0);
        assert_eq!(v[11].width_n, 0, "row 11 (partition 12) width should be 0");
        assert_eq!(v[12].width_n, 1, "row 12 (partition 13) width should be 1");
    }

    #[test]
    fn coder_partition_d5_reduction_row_order_endpoints_match_table_d5_edges() {
        // Table-wide endpoint pin: array index 0 holds partition 1,
        // index 31 holds partition 32. width_n endpoints are 0 and 1
        // respectively (Table D.5 transcription); ltmin_db carries
        // the constant under a constant callback.
        let v = coder_partition_d5_reduction_row_order(|_| -3.5);
        assert_eq!(v[0].width_n, 0, "row 0 (partition 1) width should be 0");
        assert_eq!(v[31].width_n, 1, "row 31 (partition 32) width should be 1");
        assert!((v[0].ltmin_db - -3.5).abs() < 1.0e-12);
        assert!((v[31].ltmin_db - -3.5).abs() < 1.0e-12);
    }

    #[test]
    fn coder_partition_d5_reduction_row_order_is_idempotent_for_pure_callback() {
        // A pure callback closing over no mutable state produces the
        // same paired vector on every call. Verify back-to-back
        // invocations agree column-by-column.
        let a = coder_partition_d5_reduction_row_order(|omega| f64::from(omega).cos());
        let b = coder_partition_d5_reduction_row_order(|omega| f64::from(omega).cos());
        for i in 0..32 {
            assert!((a[i].ltmin_db - b[i].ltmin_db).abs() < 1.0e-12);
            assert_eq!(a[i].width_n, b[i].width_n);
        }
    }

    #[test]
    fn coder_partition_d5_reduction_row_order_single_dip_only_affects_target_partition() {
        // A dip at a strict-interior line of partition n pulls only
        // that partition's ltmin_db down (no neighbour shares the
        // line, so no neighbour is affected). Verify by dipping a
        // strict-interior line of partition 5 and checking only row 4
        // (1-based n = 5 → 0-based index 4) drops while every other
        // row stays at the constant baseline. Width column is
        // untouched throughout.
        let target_n: u16 = 5;
        let (lo, hi) = coder_partition_d5_line_range(target_n).unwrap();
        // Pick a strict-interior line; partition 5 spans more than
        // one line so lo + 1 is interior and != hi for any plausible
        // width_n configuration in Table D.5.
        assert!(
            lo + 1 < hi,
            "partition {target_n} must have an interior line for the test"
        );
        let dip_line = lo + 1;
        let baseline = 0.0;
        let dip = -100.0;
        let v =
            coder_partition_d5_reduction_row_order(
                |omega| {
                    if omega == dip_line {
                        dip
                    } else {
                        baseline
                    }
                },
            );
        for n in 1_u16..=32 {
            let i = (n - 1) as usize;
            let expected = if n == target_n { dip } else { baseline };
            assert!(
                (v[i].ltmin_db - expected).abs() < 1.0e-12,
                "partition {n}: ltmin_db {} != expected {expected}",
                v[i].ltmin_db,
            );
        }
        // Width column unchanged by the dip.
        let widths_dipped: [u16; 32] = core::array::from_fn(|i| v[i].width_n);
        let widths_static = coder_partition_d5_width_row_order();
        assert_eq!(widths_dipped, widths_static);
    }

    #[test]
    fn coder_partition_d5_reduction_row_order_pairs_at_every_row_with_step_59_and_step_60() {
        // Strict-composition pin: at every row the paired entry is
        // exactly the index-aligned zip of step 59's LTmin vector
        // with step 60's width vector. Walk every row and verify
        // both columns individually agree with the underlying single-
        // column accessors.
        let cb = |omega: u16| (f64::from(omega) - 256.0) * 0.5;
        let paired = coder_partition_d5_reduction_row_order(cb);
        let ltmin = coder_partition_d5_ltg_min_row_order(cb);
        let widths = coder_partition_d5_width_row_order();
        for i in 0..32 {
            assert!(
                (paired[i].ltmin_db - ltmin[i]).abs() < 1.0e-12,
                "row {i}: paired ltmin_db {} != step 59 {}",
                paired[i].ltmin_db,
                ltmin[i],
            );
            assert_eq!(
                paired[i].width_n, widths[i],
                "row {i}: paired width_n {} != step 60 {}",
                paired[i].width_n, widths[i],
            );
        }
    }

    // ---------- Phase 2 step 62 / r261 — width-gated split of paired vector ----------

    #[test]
    fn coder_partition_d5_reduction_row_order_by_width_lengths_are_twelve_and_twenty() {
        // Spec rule: the width column is twelve zeros followed by
        // twenty ones. The split mirrors that pattern — the narrow
        // band carries twelve partitions, the wide band twenty.
        let split = coder_partition_d5_reduction_row_order_by_width(|_| 0.0);
        assert_eq!(split.narrow_band.len(), 12);
        assert_eq!(split.wide_band.len(), 20);
    }

    #[test]
    fn coder_partition_d5_reduction_row_order_by_width_narrow_carries_width_zero() {
        // The narrow band is the contiguous prefix of rows with
        // `width_n = 0`. Verify every element of the narrow subarray
        // has `width_n == 0`.
        let split = coder_partition_d5_reduction_row_order_by_width(|_| 0.0);
        for (i, pair) in split.narrow_band.iter().enumerate() {
            assert_eq!(
                pair.width_n,
                0,
                "narrow band index {i} (partition {}) should carry width_n = 0",
                i + 1,
            );
        }
    }

    #[test]
    fn coder_partition_d5_reduction_row_order_by_width_wide_carries_width_one() {
        // The wide band is the contiguous suffix of rows with
        // `width_n = 1`. Verify every element of the wide subarray
        // has `width_n == 1`.
        let split = coder_partition_d5_reduction_row_order_by_width(|_| 0.0);
        for (j, pair) in split.wide_band.iter().enumerate() {
            assert_eq!(
                pair.width_n,
                1,
                "wide band index {j} (partition {}) should carry width_n = 1",
                j + 13,
            );
        }
    }

    #[test]
    fn coder_partition_d5_reduction_row_order_by_width_narrow_ltmin_matches_paired_prefix() {
        // Strict-composition cross-check: the narrow band's
        // `ltmin_db` column must equal the first 12 entries of step
        // 61's paired vector for the same callback.
        let cb = |omega: u16| f64::from(omega).sin();
        let split = coder_partition_d5_reduction_row_order_by_width(cb);
        let paired = coder_partition_d5_reduction_row_order(cb);
        for (i, p) in paired.iter().enumerate().take(12) {
            assert!(
                (split.narrow_band[i].ltmin_db - p.ltmin_db).abs() < 1.0e-12,
                "narrow band index {i}: ltmin_db {} != paired[{i}] {}",
                split.narrow_band[i].ltmin_db,
                p.ltmin_db,
            );
        }
    }

    #[test]
    fn coder_partition_d5_reduction_row_order_by_width_wide_ltmin_matches_paired_suffix() {
        // Strict-composition cross-check: the wide band's `ltmin_db`
        // column must equal the last 20 entries of step 61's paired
        // vector for the same callback.
        let cb = |omega: u16| f64::from(omega).cos();
        let split = coder_partition_d5_reduction_row_order_by_width(cb);
        let paired = coder_partition_d5_reduction_row_order(cb);
        for j in 0..20 {
            assert!(
                (split.wide_band[j].ltmin_db - paired[j + 12].ltmin_db).abs() < 1.0e-12,
                "wide band index {j}: ltmin_db {} != paired[{}] {}",
                split.wide_band[j].ltmin_db,
                j + 12,
                paired[j + 12].ltmin_db,
            );
        }
    }

    #[test]
    fn coder_partition_d5_reduction_row_order_by_width_concatenates_to_paired_vector() {
        // Round-trip pin: concatenating the narrow band and the wide
        // band reconstructs the full paired vector exactly. The split
        // is structurally reversible — it re-presents the same data,
        // it does not transform it.
        let cb = |omega: u16| (f64::from(omega) - 100.0) * 0.25;
        let split = coder_partition_d5_reduction_row_order_by_width(cb);
        let paired = coder_partition_d5_reduction_row_order(cb);
        for (i, p) in paired.iter().enumerate().take(12) {
            assert!((split.narrow_band[i].ltmin_db - p.ltmin_db).abs() < 1.0e-12);
            assert_eq!(split.narrow_band[i].width_n, p.width_n);
        }
        for (j, p) in paired.iter().enumerate().skip(12) {
            let k = j - 12;
            assert!((split.wide_band[k].ltmin_db - p.ltmin_db).abs() < 1.0e-12);
            assert_eq!(split.wide_band[k].width_n, p.width_n);
        }
    }

    #[test]
    fn coder_partition_d5_reduction_row_order_by_width_constant_callback_carries_constant() {
        // For a constant LTg(ω) = c the per-partition min is c
        // everywhere (Phase 2 step 58 inherits this). Verify both
        // subarrays of the split carry the constant in `ltmin_db`.
        let c = -11.5_f64;
        let split = coder_partition_d5_reduction_row_order_by_width(|_| c);
        for pair in &split.narrow_band {
            assert!((pair.ltmin_db - c).abs() < 1.0e-12);
        }
        for pair in &split.wide_band {
            assert!((pair.ltmin_db - c).abs() < 1.0e-12);
        }
    }

    #[test]
    fn coder_partition_d5_reduction_row_order_by_width_endpoints_match_table_d5_edges() {
        // Endpoint pin: narrow band's first element holds partition
        // 1, narrow band's last element holds partition 12; wide
        // band's first element holds partition 13, wide band's last
        // element holds partition 32. Verify under a constant
        // callback so the LTmin column carries the constant.
        let split = coder_partition_d5_reduction_row_order_by_width(|_| -2.0);
        assert_eq!(split.narrow_band[0].width_n, 0);
        assert_eq!(split.narrow_band[11].width_n, 0);
        assert_eq!(split.wide_band[0].width_n, 1);
        assert_eq!(split.wide_band[19].width_n, 1);
        assert!((split.narrow_band[0].ltmin_db - -2.0).abs() < 1.0e-12);
        assert!((split.narrow_band[11].ltmin_db - -2.0).abs() < 1.0e-12);
        assert!((split.wide_band[0].ltmin_db - -2.0).abs() < 1.0e-12);
        assert!((split.wide_band[19].ltmin_db - -2.0).abs() < 1.0e-12);
    }

    #[test]
    fn coder_partition_d5_reduction_row_order_by_width_split_point_is_at_partition_thirteen() {
        // The split point is fixed at 12 (the width column's single
        // 0 → 1 transition between partitions 12 and 13). Pin the
        // structural boundary: the narrow band ends at partition 12;
        // the wide band starts at partition 13.
        let split = coder_partition_d5_reduction_row_order_by_width(f64::from);
        // Partition 12 → 0-based narrow_band index 11
        let (lo_12, _hi_12) = coder_partition_d5_line_range(12).unwrap();
        assert!(
            (split.narrow_band[11].ltmin_db - f64::from(lo_12)).abs() < 1.0e-9,
            "narrow tail: expected partition 12 ωlow = {}, got {}",
            lo_12,
            split.narrow_band[11].ltmin_db,
        );
        // Partition 13 → 0-based wide_band index 0
        let (lo_13, _hi_13) = coder_partition_d5_line_range(13).unwrap();
        assert!(
            (split.wide_band[0].ltmin_db - f64::from(lo_13)).abs() < 1.0e-9,
            "wide head: expected partition 13 ωlow = {}, got {}",
            lo_13,
            split.wide_band[0].ltmin_db,
        );
    }

    #[test]
    fn coder_partition_d5_reduction_row_order_by_width_widths_invariant_across_callbacks() {
        // Structural orthogonality: the split point and the width
        // values are fully determined by the static Table D.5 column
        // and do not depend on the caller's `LTg(ω)`. Verify two
        // different callbacks produce identical width columns on
        // both subarrays.
        let a = coder_partition_d5_reduction_row_order_by_width(|_| 0.0);
        let b =
            coder_partition_d5_reduction_row_order_by_width(|omega| f64::from(omega) * 3.0 - 1.0);
        for i in 0..12 {
            assert_eq!(a.narrow_band[i].width_n, b.narrow_band[i].width_n);
        }
        for j in 0..20 {
            assert_eq!(a.wide_band[j].width_n, b.wide_band[j].width_n);
        }
    }

    #[test]
    fn coder_partition_d5_reduction_row_order_by_width_is_idempotent_for_pure_callback() {
        // A pure callback closing over no mutable state produces the
        // same split on every call. Verify back-to-back invocations
        // agree column-by-column on both subarrays.
        let a = coder_partition_d5_reduction_row_order_by_width(|omega| f64::from(omega).cos());
        let b = coder_partition_d5_reduction_row_order_by_width(|omega| f64::from(omega).cos());
        for i in 0..12 {
            assert!((a.narrow_band[i].ltmin_db - b.narrow_band[i].ltmin_db).abs() < 1.0e-12);
            assert_eq!(a.narrow_band[i].width_n, b.narrow_band[i].width_n);
        }
        for j in 0..20 {
            assert!((a.wide_band[j].ltmin_db - b.wide_band[j].ltmin_db).abs() < 1.0e-12);
            assert_eq!(a.wide_band[j].width_n, b.wide_band[j].width_n);
        }
    }

    #[test]
    fn coder_partition_d5_reduction_row_order_by_width_total_length_matches_paired_vector() {
        // The total number of partitions across both subarrays
        // matches Phase 2 step 61's 32-element paired vector (12 + 20
        // = 32). The split exhausts the paired vector — no row is
        // dropped, no row is duplicated.
        let split = coder_partition_d5_reduction_row_order_by_width(|_| 0.0);
        assert_eq!(split.narrow_band.len() + split.wide_band.len(), 32);
    }

    #[test]
    fn coder_partition_d5_reduction_row_order_by_width_dip_in_narrow_only_affects_narrow() {
        // A dip at a strict-interior line of a partition in the
        // narrow band (n ∈ 1..=12) pulls only that narrow-band row's
        // `ltmin_db` down. The wide band is untouched. Verify by
        // dipping a strict-interior line of partition 5 (narrow
        // band, 0-based index 4) and checking the wide band stays at
        // the baseline everywhere.
        let target_n: u16 = 5;
        let (lo, hi) = coder_partition_d5_line_range(target_n).unwrap();
        assert!(
            lo + 1 < hi,
            "partition {target_n} must have an interior line for the test"
        );
        let dip_line = lo + 1;
        let baseline = 0.0;
        let dip = -100.0;
        let split = coder_partition_d5_reduction_row_order_by_width(|omega| {
            if omega == dip_line {
                dip
            } else {
                baseline
            }
        });
        // Narrow band: only index 4 (partition 5) drops.
        for (i, pair) in split.narrow_band.iter().enumerate() {
            let expected = if i + 1 == target_n as usize {
                dip
            } else {
                baseline
            };
            assert!(
                (pair.ltmin_db - expected).abs() < 1.0e-12,
                "narrow band index {i}: ltmin_db {} != expected {expected}",
                pair.ltmin_db,
            );
        }
        // Wide band: every element at the baseline.
        for (j, pair) in split.wide_band.iter().enumerate() {
            assert!(
                (pair.ltmin_db - baseline).abs() < 1.0e-12,
                "wide band index {j}: ltmin_db {} != baseline {baseline}",
                pair.ltmin_db,
            );
        }
    }

    #[test]
    fn coder_partition_d5_reduction_row_order_by_width_dip_in_wide_only_affects_wide() {
        // Dual of the previous test: a dip at a strict-interior line
        // of a partition in the wide band (n ∈ 13..=32) pulls only
        // that wide-band row's `ltmin_db` down. The narrow band is
        // untouched.
        let target_n: u16 = 20;
        let (lo, hi) = coder_partition_d5_line_range(target_n).unwrap();
        assert!(
            lo + 1 < hi,
            "partition {target_n} must have an interior line for the test"
        );
        let dip_line = lo + 1;
        let baseline = 0.0;
        let dip = -100.0;
        let split = coder_partition_d5_reduction_row_order_by_width(|omega| {
            if omega == dip_line {
                dip
            } else {
                baseline
            }
        });
        // Narrow band: every element at the baseline.
        for (i, pair) in split.narrow_band.iter().enumerate() {
            assert!(
                (pair.ltmin_db - baseline).abs() < 1.0e-12,
                "narrow band index {i}: ltmin_db {} != baseline {baseline}",
                pair.ltmin_db,
            );
        }
        // Wide band: only index (target_n - 13) drops.
        let expected_index = (target_n as usize) - 13;
        for (j, pair) in split.wide_band.iter().enumerate() {
            let expected = if j == expected_index { dip } else { baseline };
            assert!(
                (pair.ltmin_db - expected).abs() < 1.0e-12,
                "wide band index {j}: ltmin_db {} != expected {expected}",
                pair.ltmin_db,
            );
        }
    }

    // -----------------------------------------------------------------
    // Phase 2 step 63 (r262): §D.1 Step 8 width-gated `LTmin_n` column
    // projection over Table D.5. Strict projection of step 62's
    // width-gated paired vector onto the `ltmin_db` field of each
    // subarray.
    // -----------------------------------------------------------------

    #[test]
    fn coder_partition_d5_ltmin_db_row_order_by_width_lengths_are_twelve_and_twenty() {
        // Narrow band is 12 rows (partitions 1..=12, width = 0); wide
        // band is 20 rows (partitions 13..=32, width = 1). Match step 62.
        let split = coder_partition_d5_ltmin_db_row_order_by_width(|_| 0.0);
        assert_eq!(split.narrow_band.len(), 12);
        assert_eq!(split.wide_band.len(), 20);
    }

    #[test]
    fn coder_partition_d5_ltmin_db_row_order_by_width_total_length_matches_paired_vector() {
        // 12 + 20 = 32 — the total number of recoverable partitions.
        let split = coder_partition_d5_ltmin_db_row_order_by_width(|_| 0.0);
        assert_eq!(split.narrow_band.len() + split.wide_band.len(), 32);
    }

    #[test]
    fn coder_partition_d5_ltmin_db_row_order_by_width_constant_callback_carries_constant() {
        // A constant `LTg(ω)` callback reduces to itself in every
        // partition (min of a constant is the constant) — every cell
        // in both subarrays equals the constant.
        let c = -3.5_f64;
        let split = coder_partition_d5_ltmin_db_row_order_by_width(|_| c);
        for (i, v) in split.narrow_band.iter().enumerate() {
            assert!(
                (v - c).abs() < 1.0e-12,
                "narrow band index {i}: ltmin_db {v} != constant {c}",
            );
        }
        for (j, v) in split.wide_band.iter().enumerate() {
            assert!(
                (v - c).abs() < 1.0e-12,
                "wide band index {j}: ltmin_db {v} != constant {c}",
            );
        }
    }

    #[test]
    fn coder_partition_d5_ltmin_db_row_order_by_width_narrow_matches_step_62_narrow_field() {
        // Strict-projection cross-check: every narrow cell equals the
        // matching step 62 narrow cell's `ltmin_db` field at the same
        // array index, under a non-trivial line-dependent callback.
        let cb = |omega: u16| -> f64 { (f64::from(omega) * 0.011).sin() };
        let split = coder_partition_d5_ltmin_db_row_order_by_width(cb);
        let paired_split = coder_partition_d5_reduction_row_order_by_width(cb);
        for i in 0..12 {
            assert!(
                (split.narrow_band[i] - paired_split.narrow_band[i].ltmin_db).abs() < 1.0e-12,
                "narrow band index {i}: projected {} != step 62 ltmin_db {}",
                split.narrow_band[i],
                paired_split.narrow_band[i].ltmin_db,
            );
        }
    }

    #[test]
    fn coder_partition_d5_ltmin_db_row_order_by_width_wide_matches_step_62_wide_field() {
        // Strict-projection cross-check on the wide subarray.
        let cb = |omega: u16| -> f64 { (f64::from(omega) * 0.013).cos() };
        let split = coder_partition_d5_ltmin_db_row_order_by_width(cb);
        let paired_split = coder_partition_d5_reduction_row_order_by_width(cb);
        for j in 0..20 {
            assert!(
                (split.wide_band[j] - paired_split.wide_band[j].ltmin_db).abs() < 1.0e-12,
                "wide band index {j}: projected {} != step 62 ltmin_db {}",
                split.wide_band[j],
                paired_split.wide_band[j].ltmin_db,
            );
        }
    }

    #[test]
    fn coder_partition_d5_ltmin_db_row_order_by_width_narrow_matches_paired_prefix() {
        // The narrow subarray's projected ltmin_db column matches the
        // first 12 entries of step 61's row-order paired vector's
        // `ltmin_db` column.
        let cb = |omega: u16| -> f64 { f64::from(omega) * 0.5 - 2.0 };
        let split = coder_partition_d5_ltmin_db_row_order_by_width(cb);
        let paired = coder_partition_d5_reduction_row_order(cb);
        for (i, paired_cell) in paired.iter().enumerate().take(12) {
            assert!(
                (split.narrow_band[i] - paired_cell.ltmin_db).abs() < 1.0e-12,
                "narrow band index {i}: projected {} != step 61 prefix ltmin_db {}",
                split.narrow_band[i],
                paired_cell.ltmin_db,
            );
        }
    }

    #[test]
    fn coder_partition_d5_ltmin_db_row_order_by_width_wide_matches_paired_suffix() {
        // The wide subarray's projected ltmin_db column matches the
        // last 20 entries of step 61's row-order paired vector's
        // `ltmin_db` column.
        let cb = |omega: u16| -> f64 { -f64::from(omega) * 0.25 + 1.0 };
        let split = coder_partition_d5_ltmin_db_row_order_by_width(cb);
        let paired = coder_partition_d5_reduction_row_order(cb);
        for j in 0..20 {
            assert!(
                (split.wide_band[j] - paired[j + 12].ltmin_db).abs() < 1.0e-12,
                "wide band index {j}: projected {} != step 61 suffix ltmin_db {}",
                split.wide_band[j],
                paired[j + 12].ltmin_db,
            );
        }
    }

    #[test]
    fn coder_partition_d5_ltmin_db_row_order_by_width_concatenates_to_step_59_vector() {
        // narrow_band ++ wide_band equals the 32-element step 59
        // row-order LTmin vector index-by-index.
        let cb = |omega: u16| -> f64 { (f64::from(omega) * 0.017).tan() };
        let split = coder_partition_d5_ltmin_db_row_order_by_width(cb);
        let row_order = coder_partition_d5_ltg_min_row_order(cb);
        for (i, &row) in row_order.iter().enumerate().take(12) {
            assert!(
                (split.narrow_band[i] - row).abs() < 1.0e-12,
                "narrow band index {i}: projected {} != step 59 row {row}",
                split.narrow_band[i],
            );
        }
        for (j, &row) in row_order.iter().enumerate().skip(12) {
            let idx = j - 12;
            assert!(
                (split.wide_band[idx] - row).abs() < 1.0e-12,
                "wide band index {idx}: projected {} != step 59 row {row}",
                split.wide_band[idx],
            );
        }
    }

    #[test]
    fn coder_partition_d5_ltmin_db_row_order_by_width_endpoints_match_table_d5_edges() {
        // Endpoint pins: narrow_band[0] is partition 1 (lowest narrow
        // row), narrow_band[11] is partition 12 (highest narrow row),
        // wide_band[0] is partition 13 (lowest wide row),
        // wide_band[19] is partition 32 (highest wide row).
        let cb = |omega: u16| -> f64 { f64::from(omega) };
        let split = coder_partition_d5_ltmin_db_row_order_by_width(cb);
        // identity callback under inclusive-min reduces to ωlow_n per row.
        let (lo_1, _) = coder_partition_d5_line_range(1).unwrap();
        let (lo_12, _) = coder_partition_d5_line_range(12).unwrap();
        let (lo_13, _) = coder_partition_d5_line_range(13).unwrap();
        let (lo_32, _) = coder_partition_d5_line_range(32).unwrap();
        assert!((split.narrow_band[0] - f64::from(lo_1)).abs() < 1.0e-12);
        assert!((split.narrow_band[11] - f64::from(lo_12)).abs() < 1.0e-12);
        assert!((split.wide_band[0] - f64::from(lo_13)).abs() < 1.0e-12);
        assert!((split.wide_band[19] - f64::from(lo_32)).abs() < 1.0e-12);
    }

    #[test]
    fn coder_partition_d5_ltmin_db_row_order_by_width_split_point_is_at_partition_thirteen() {
        // Split point pin: narrow_band[11] reduces over partition 12's
        // FFT-line range; wide_band[0] reduces over partition 13's
        // FFT-line range. The two rows are adjacent in row-order Table
        // D.5 — the width-column 0 → 1 transition lives between them.
        let cb = |omega: u16| -> f64 { f64::from(omega) };
        let split = coder_partition_d5_ltmin_db_row_order_by_width(cb);
        let (lo_12, _) = coder_partition_d5_line_range(12).unwrap();
        let (lo_13, _) = coder_partition_d5_line_range(13).unwrap();
        assert!(
            (split.narrow_band[11] - f64::from(lo_12)).abs() < 1.0e-12,
            "narrow_band[11] (partition 12) should reduce to ωlow_12 = {lo_12}",
        );
        assert!(
            (split.wide_band[0] - f64::from(lo_13)).abs() < 1.0e-12,
            "wide_band[0] (partition 13) should reduce to ωlow_13 = {lo_13}",
        );
    }

    #[test]
    fn coder_partition_d5_ltmin_db_row_order_by_width_is_idempotent_for_pure_callback() {
        // Pure callbacks produce the same column projection on
        // back-to-back calls.
        let cb = |omega: u16| -> f64 { f64::from(omega).cos() };
        let a = coder_partition_d5_ltmin_db_row_order_by_width(cb);
        let b = coder_partition_d5_ltmin_db_row_order_by_width(cb);
        for i in 0..12 {
            assert!((a.narrow_band[i] - b.narrow_band[i]).abs() < 1.0e-12);
        }
        for j in 0..20 {
            assert!((a.wide_band[j] - b.wide_band[j]).abs() < 1.0e-12);
        }
    }

    #[test]
    fn coder_partition_d5_ltmin_db_row_order_by_width_dip_in_narrow_only_affects_narrow() {
        // Single-line dip in partition 6 (a strict-interior narrow
        // partition): only narrow_band[5] drops. wide_band is
        // untouched. Strict cross-band isolation, projection variant.
        let baseline = 5.0_f64;
        let dip = -25.0_f64;
        let target_n: u16 = 6;
        let (lo, hi) = coder_partition_d5_line_range(target_n).unwrap();
        let interior = lo + (hi - lo) / 2;
        assert!(interior > lo && interior < hi);
        let split = coder_partition_d5_ltmin_db_row_order_by_width(|omega| {
            if omega == interior {
                dip
            } else {
                baseline
            }
        });
        let expected_index = (target_n as usize) - 1;
        for (i, v) in split.narrow_band.iter().enumerate() {
            let expected = if i == expected_index { dip } else { baseline };
            assert!(
                (v - expected).abs() < 1.0e-12,
                "narrow band index {i}: projected {v} != expected {expected}",
            );
        }
        for (j, v) in split.wide_band.iter().enumerate() {
            assert!(
                (v - baseline).abs() < 1.0e-12,
                "wide band index {j}: projected {v} != baseline {baseline}",
            );
        }
    }

    #[test]
    fn coder_partition_d5_ltmin_db_row_order_by_width_dip_in_wide_only_affects_wide() {
        // Single-line dip in partition 22 (a strict-interior wide
        // partition): only wide_band[22 - 13] = wide_band[9] drops.
        // narrow_band is untouched. Dual cross-band isolation.
        let baseline = 5.0_f64;
        let dip = -25.0_f64;
        let target_n: u16 = 22;
        let (lo, hi) = coder_partition_d5_line_range(target_n).unwrap();
        let interior = lo + (hi - lo) / 2;
        assert!(interior > lo && interior < hi);
        let split = coder_partition_d5_ltmin_db_row_order_by_width(|omega| {
            if omega == interior {
                dip
            } else {
                baseline
            }
        });
        for (i, v) in split.narrow_band.iter().enumerate() {
            assert!(
                (v - baseline).abs() < 1.0e-12,
                "narrow band index {i}: projected {v} != baseline {baseline}",
            );
        }
        let expected_index = (target_n as usize) - 13;
        for (j, v) in split.wide_band.iter().enumerate() {
            let expected = if j == expected_index { dip } else { baseline };
            assert!(
                (v - expected).abs() < 1.0e-12,
                "wide band index {j}: projected {v} != expected {expected}",
            );
        }
    }

    #[test]
    fn coder_partition_d5_ltmin_db_row_order_by_width_split_yields_a_partition_of_step_59() {
        // Composition pin: for every row-order callback, concatenating
        // the projected narrow and wide subarrays equals step 59's
        // row-order vector — i.e. the step 63 projection is a
        // partition of step 59's output.
        let cb = |omega: u16| -> f64 { (f64::from(omega) * 0.019).sin().abs() };
        let split = coder_partition_d5_ltmin_db_row_order_by_width(cb);
        let row_order = coder_partition_d5_ltg_min_row_order(cb);
        let mut concat = [0.0_f64; 32];
        concat[..12].copy_from_slice(&split.narrow_band);
        concat[12..].copy_from_slice(&split.wide_band);
        for i in 0..32 {
            assert!(
                (concat[i] - row_order[i]).abs() < 1.0e-12,
                "concatenation index {i}: {} != step 59 row {}",
                concat[i],
                row_order[i],
            );
        }
    }

    #[test]
    fn coder_partition_d5_ltmin_db_row_order_by_width_independent_of_width_column() {
        // Sanity: the projection is purely on the ltmin_db field —
        // its values do not depend on the width column at all (the
        // width column is implicit in the choice of subarray). Two
        // hypothetical width-column orderings (had they applied)
        // would still produce the same projected columns because the
        // projection reads only ltmin_db and the underlying step 58
        // / 59 reduction is width-column-blind. Verified here by
        // checking equality against a non-trivial recomputation
        // through the paired step 61 + manual prefix/suffix split,
        // skipping step 62 entirely.
        let cb = |omega: u16| -> f64 { (f64::from(omega) * 0.023).cos() * 4.0 - 1.0 };
        let split = coder_partition_d5_ltmin_db_row_order_by_width(cb);
        let paired = coder_partition_d5_reduction_row_order(cb);
        for (i, paired_cell) in paired.iter().enumerate().take(12) {
            assert!((split.narrow_band[i] - paired_cell.ltmin_db).abs() < 1.0e-12);
        }
        for (j, paired_cell) in paired.iter().enumerate().skip(12) {
            let idx = j - 12;
            assert!((split.wide_band[idx] - paired_cell.ltmin_db).abs() < 1.0e-12);
        }
    }

    // ---- Phase 2 step 64 / r263 — width-gated `LTmin_n` linear-energy
    // projection over Table D.5 (linearisation of step 63's per-band dB
    // column).
    #[test]
    fn coder_partition_d5_ltmin_linear_row_order_by_width_lengths_are_twelve_and_twenty() {
        // Structural pin: 12 narrow + 20 wide cells, matching step 63.
        let split = coder_partition_d5_ltmin_linear_row_order_by_width(|_| 0.0);
        assert_eq!(split.narrow_band.len(), 12);
        assert_eq!(split.wide_band.len(), 20);
    }

    #[test]
    fn coder_partition_d5_ltmin_linear_row_order_by_width_zero_db_is_unit_energy() {
        // 10^(0/10) = 1.0 in every cell when the callback returns 0 dB
        // for every FFT line — every partition's minimum reduces to 0
        // dB, which linearises to unit energy.
        let split = coder_partition_d5_ltmin_linear_row_order_by_width(|_| 0.0);
        for &v in &split.narrow_band {
            assert!((v - 1.0).abs() < 1.0e-12, "narrow cell {v} != 1.0");
        }
        for &v in &split.wide_band {
            assert!((v - 1.0).abs() < 1.0e-12, "wide cell {v} != 1.0");
        }
    }

    #[test]
    fn coder_partition_d5_ltmin_linear_row_order_by_width_strictly_positive() {
        // Linearisation preserves non-negativity: 10^x > 0 for every
        // finite real x. A callback returning −∞ < db_value < +∞
        // produces strictly positive cells.
        let cb = |omega: u16| -> f64 { -50.0 + f64::from(omega) * 0.1 };
        let split = coder_partition_d5_ltmin_linear_row_order_by_width(cb);
        for &v in &split.narrow_band {
            assert!(v > 0.0, "narrow cell {v} is not strictly positive");
            assert!(v.is_finite(), "narrow cell {v} is not finite");
        }
        for &v in &split.wide_band {
            assert!(v > 0.0, "wide cell {v} is not strictly positive");
            assert!(v.is_finite(), "wide cell {v} is not finite");
        }
    }

    #[test]
    fn coder_partition_d5_ltmin_linear_row_order_by_width_matches_step63_pow10_div10() {
        // Cell-wise relation: linear[i] = 10^(db[i] / 10) exactly,
        // reusing the same callback for both projections.
        let cb = |omega: u16| -> f64 { f64::from(omega) * 0.5 - 2.0 };
        let lin = coder_partition_d5_ltmin_linear_row_order_by_width(cb);
        let db = coder_partition_d5_ltmin_db_row_order_by_width(cb);
        for (i, (&l, &d)) in lin
            .narrow_band
            .iter()
            .zip(db.narrow_band.iter())
            .enumerate()
        {
            let expect = (10.0_f64).powf(d / 10.0);
            assert!(
                (l - expect).abs() < 1.0e-12,
                "narrow {i}: lin {l} != 10^({d}/10) = {expect}",
            );
        }
        for (j, (&l, &d)) in lin.wide_band.iter().zip(db.wide_band.iter()).enumerate() {
            let expect = (10.0_f64).powf(d / 10.0);
            assert!(
                (l - expect).abs() < 1.0e-12,
                "wide {j}: lin {l} != 10^({d}/10) = {expect}",
            );
        }
    }

    #[test]
    fn coder_partition_d5_ltmin_linear_row_order_by_width_ten_db_is_factor_ten() {
        // Spot pin: a uniform 10 dB callback linearises to 10.0
        // (since `LTmin_n = min(10 dB) = 10 dB` for every n,
        // 10^(10/10) = 10.0).
        let split = coder_partition_d5_ltmin_linear_row_order_by_width(|_| 10.0);
        for &v in &split.narrow_band {
            assert!((v - 10.0).abs() < 1.0e-12, "narrow cell {v} != 10.0");
        }
        for &v in &split.wide_band {
            assert!((v - 10.0).abs() < 1.0e-12, "wide cell {v} != 10.0");
        }
    }

    #[test]
    fn coder_partition_d5_ltmin_linear_row_order_by_width_minus_ten_db_is_one_tenth() {
        // Spot pin: a uniform −10 dB callback linearises to 0.1
        // (10^(−10/10) = 0.1).
        let split = coder_partition_d5_ltmin_linear_row_order_by_width(|_| -10.0);
        for &v in &split.narrow_band {
            assert!((v - 0.1).abs() < 1.0e-12, "narrow cell {v} != 0.1");
        }
        for &v in &split.wide_band {
            assert!((v - 0.1).abs() < 1.0e-12, "wide cell {v} != 0.1");
        }
    }

    #[test]
    fn coder_partition_d5_ltmin_linear_row_order_by_width_monotone_in_db_per_cell() {
        // Strict monotonicity: a callback that everywhere returns
        // (cb_a - 1.0) dB produces a linear cell that is strictly
        // smaller (in fact, 10^(−1/10) ≈ 0.794 of) the linear cell of
        // a callback that returns cb_a dB. Because every partition's
        // min is shifted by exactly −1 dB, the linear ratio is
        // identical at every cell.
        let cb_hi = |omega: u16| -> f64 { f64::from(omega) * 0.25 + 3.0 };
        let cb_lo = |omega: u16| -> f64 { f64::from(omega) * 0.25 + 2.0 };
        let hi = coder_partition_d5_ltmin_linear_row_order_by_width(cb_hi);
        let lo = coder_partition_d5_ltmin_linear_row_order_by_width(cb_lo);
        let ratio = (10.0_f64).powf(-1.0 / 10.0);
        for (i, (&h, &l)) in hi.narrow_band.iter().zip(lo.narrow_band.iter()).enumerate() {
            assert!(l < h, "narrow {i}: lo {l} not < hi {h}");
            let r = l / h;
            assert!(
                (r - ratio).abs() < 1.0e-12,
                "narrow {i}: ratio {r} != {ratio}",
            );
        }
        for (j, (&h, &l)) in hi.wide_band.iter().zip(lo.wide_band.iter()).enumerate() {
            assert!(l < h, "wide {j}: lo {l} not < hi {h}");
            let r = l / h;
            assert!(
                (r - ratio).abs() < 1.0e-12,
                "wide {j}: ratio {r} != {ratio}",
            );
        }
    }

    #[test]
    fn coder_partition_d5_ltmin_linear_row_order_by_width_is_idempotent_for_pure_callback() {
        // Pure callback → same linear projection on repeated invocation.
        let cb = |omega: u16| -> f64 { f64::from(omega).sin() * 6.0 + 0.5 };
        let a = coder_partition_d5_ltmin_linear_row_order_by_width(cb);
        let b = coder_partition_d5_ltmin_linear_row_order_by_width(cb);
        for i in 0..12 {
            assert!((a.narrow_band[i] - b.narrow_band[i]).abs() < 1.0e-12);
        }
        for j in 0..20 {
            assert!((a.wide_band[j] - b.wide_band[j]).abs() < 1.0e-12);
        }
    }

    #[test]
    fn coder_partition_d5_ltmin_linear_row_order_by_width_dip_in_narrow_only_affects_narrow() {
        // A −20 dB dip in a single FFT line that lives inside the
        // narrow block (line ω = 50 lives in partition 3, narrow row
        // index 2) lowers exactly one narrow-band cell relative to the
        // baseline; wide-band cells are unchanged.
        let baseline = coder_partition_d5_ltmin_linear_row_order_by_width(|_| 0.0);
        let cb = |omega: u16| -> f64 {
            if omega == 50 {
                -20.0
            } else {
                0.0
            }
        };
        let split = coder_partition_d5_ltmin_linear_row_order_by_width(cb);
        for j in 0..20 {
            assert!(
                (split.wide_band[j] - baseline.wide_band[j]).abs() < 1.0e-12,
                "wide band cell {j} should be unaffected by a narrow-line dip",
            );
        }
        let mut narrow_dipped = 0u32;
        for i in 0..12 {
            if (split.narrow_band[i] - baseline.narrow_band[i]).abs() > 1.0e-12 {
                narrow_dipped += 1;
                assert!(
                    split.narrow_band[i] < baseline.narrow_band[i],
                    "narrow band cell {i} should have dipped",
                );
            }
        }
        assert_eq!(
            narrow_dipped, 1,
            "exactly one narrow band cell should dip from a single-line callback",
        );
    }

    #[test]
    fn coder_partition_d5_ltmin_linear_row_order_by_width_dip_in_wide_only_affects_wide() {
        // A −20 dB dip in a single FFT line that lives inside the
        // wide block (line ω = 300 lives in partition 18, wide row
        // index 5) lowers exactly one wide-band cell relative to the
        // baseline; narrow-band cells are unchanged.
        let baseline = coder_partition_d5_ltmin_linear_row_order_by_width(|_| 0.0);
        let cb = |omega: u16| -> f64 {
            if omega == 300 {
                -20.0
            } else {
                0.0
            }
        };
        let split = coder_partition_d5_ltmin_linear_row_order_by_width(cb);
        for i in 0..12 {
            assert!(
                (split.narrow_band[i] - baseline.narrow_band[i]).abs() < 1.0e-12,
                "narrow band cell {i} should be unaffected by a wide-line dip",
            );
        }
        let mut wide_dipped = 0u32;
        for j in 0..20 {
            if (split.wide_band[j] - baseline.wide_band[j]).abs() > 1.0e-12 {
                wide_dipped += 1;
                assert!(
                    split.wide_band[j] < baseline.wide_band[j],
                    "wide band cell {j} should have dipped",
                );
            }
        }
        assert_eq!(
            wide_dipped, 1,
            "exactly one wide band cell should dip from a single-line callback",
        );
    }

    #[test]
    fn coder_partition_d5_ltmin_linear_row_order_by_width_split_yields_a_partition_of_step59() {
        // narrow_band ++ wide_band, when log-mapped back to dB, equals
        // step 59's row-order LTmin vector index-by-index.
        let cb = |omega: u16| -> f64 { f64::from(omega) * 0.03 - 1.5 };
        let split = coder_partition_d5_ltmin_linear_row_order_by_width(cb);
        let row_order = coder_partition_d5_ltg_min_row_order(cb);
        for (i, &row_db) in row_order.iter().enumerate().take(12) {
            let recovered_db = 10.0 * split.narrow_band[i].log10();
            assert!(
                (recovered_db - row_db).abs() < 1.0e-9,
                "narrow band index {i}: recovered {recovered_db} dB != step 59 {row_db} dB",
            );
        }
        for (j, &row_db) in row_order.iter().enumerate().skip(12) {
            let idx = j - 12;
            let recovered_db = 10.0 * split.wide_band[idx].log10();
            assert!(
                (recovered_db - row_db).abs() < 1.0e-9,
                "wide band index {idx}: recovered {recovered_db} dB != step 59 {row_db} dB",
            );
        }
    }

    // ---- Phase 2 step 65 / r264 — width-gated `log2(LTmin_lin_n)` column
    // projection over Table D.5 (logarithmisation of step 64's per-band
    // linear-energy column).
    #[test]
    fn coder_partition_d5_ltmin_log2_row_order_by_width_lengths_are_twelve_and_twenty() {
        // Structural pin: 12 narrow + 20 wide cells, matching step 64.
        let split = coder_partition_d5_ltmin_log2_row_order_by_width(|_| 0.0);
        assert_eq!(split.narrow_band.len(), 12);
        assert_eq!(split.wide_band.len(), 20);
    }

    #[test]
    fn coder_partition_d5_ltmin_log2_row_order_by_width_zero_db_is_zero_log2() {
        // 0 dB → linear 1.0 → log2(1.0) = 0.0 in every cell.
        let split = coder_partition_d5_ltmin_log2_row_order_by_width(|_| 0.0);
        for &v in &split.narrow_band {
            assert!(v.abs() < 1.0e-12, "narrow cell {v} != 0.0");
        }
        for &v in &split.wide_band {
            assert!(v.abs() < 1.0e-12, "wide cell {v} != 0.0");
        }
    }

    #[test]
    fn coder_partition_d5_ltmin_log2_row_order_by_width_finite_for_finite_callback() {
        // Every output cell is finite when the callback returns finite
        // dB at every FFT line (step 64's positivity guarantees
        // `log2(strictly_positive) ∈ ℝ`).
        let cb = |omega: u16| -> f64 { -25.0 + f64::from(omega) * 0.07 };
        let split = coder_partition_d5_ltmin_log2_row_order_by_width(cb);
        for &v in &split.narrow_band {
            assert!(v.is_finite(), "narrow cell {v} is not finite");
        }
        for &v in &split.wide_band {
            assert!(v.is_finite(), "wide cell {v} is not finite");
        }
    }

    #[test]
    fn coder_partition_d5_ltmin_log2_row_order_by_width_matches_step64_log2() {
        // Cell-wise relation: log2[i] = log2(linear[i]) exactly,
        // reusing the same callback for both projections.
        let cb = |omega: u16| -> f64 { f64::from(omega) * 0.3 - 1.5 };
        let log2 = coder_partition_d5_ltmin_log2_row_order_by_width(cb);
        let lin = coder_partition_d5_ltmin_linear_row_order_by_width(cb);
        for (i, (&l2, &li)) in log2
            .narrow_band
            .iter()
            .zip(lin.narrow_band.iter())
            .enumerate()
        {
            let expect = li.log2();
            assert!(
                (l2 - expect).abs() < 1.0e-12,
                "narrow {i}: log2 {l2} != log2(lin {li}) = {expect}",
            );
        }
        for (j, (&l2, &li)) in log2.wide_band.iter().zip(lin.wide_band.iter()).enumerate() {
            let expect = li.log2();
            assert!(
                (l2 - expect).abs() < 1.0e-12,
                "wide {j}: log2 {l2} != log2(lin {li}) = {expect}",
            );
        }
    }

    #[test]
    fn coder_partition_d5_ltmin_log2_row_order_by_width_three_db_is_pin() {
        // Spot pin: a uniform +3 dB callback produces `log2(10^0.3)` in
        // every cell. log10(2) ≈ 0.30103, so log2(10^0.3) =
        // 0.3 / log10(2) ≈ 0.9966.
        let split = coder_partition_d5_ltmin_log2_row_order_by_width(|_| 3.0);
        let expect = 3.0_f64 / 10.0 / 2.0_f64.log10();
        for &v in &split.narrow_band {
            assert!((v - expect).abs() < 1.0e-12, "narrow cell {v} != {expect}");
        }
        for &v in &split.wide_band {
            assert!((v - expect).abs() < 1.0e-12, "wide cell {v} != {expect}");
        }
    }

    #[test]
    fn coder_partition_d5_ltmin_log2_row_order_by_width_minus_three_db_is_sign_flipped() {
        // log2 is odd-symmetric around log2(1.0) = 0: a uniform −3 dB
        // callback produces a cell equal to the negative of the uniform
        // +3 dB cell (both sit at `linear = 10^(±0.3) = 2.0^(±x)` where
        // `x = 0.3 / log10(2)`).
        let hi = coder_partition_d5_ltmin_log2_row_order_by_width(|_| 3.0);
        let lo = coder_partition_d5_ltmin_log2_row_order_by_width(|_| -3.0);
        for (i, (&h, &l)) in hi.narrow_band.iter().zip(lo.narrow_band.iter()).enumerate() {
            assert!((h + l).abs() < 1.0e-12, "narrow {i}: hi {h} + lo {l} != 0");
        }
        for (j, (&h, &l)) in hi.wide_band.iter().zip(lo.wide_band.iter()).enumerate() {
            assert!((h + l).abs() < 1.0e-12, "wide {j}: hi {h} + lo {l} != 0");
        }
    }

    #[test]
    fn coder_partition_d5_ltmin_log2_row_order_by_width_is_proportional_to_step63_db() {
        // Identity: every output cell = step 63's matching dB cell
        // multiplied by `log2(10) / 10` (the dB → log2-linear constant).
        let cb = |omega: u16| -> f64 { f64::from(omega) * 0.42 + 1.7 };
        let log2 = coder_partition_d5_ltmin_log2_row_order_by_width(cb);
        let db = coder_partition_d5_ltmin_db_row_order_by_width(cb);
        let k = 10.0_f64.log2() / 10.0;
        for (i, (&l2, &d)) in log2
            .narrow_band
            .iter()
            .zip(db.narrow_band.iter())
            .enumerate()
        {
            let expect = d * k;
            assert!(
                (l2 - expect).abs() < 1.0e-12,
                "narrow {i}: log2 {l2} != {d} dB × {k} = {expect}",
            );
        }
        for (j, (&l2, &d)) in log2.wide_band.iter().zip(db.wide_band.iter()).enumerate() {
            let expect = d * k;
            assert!(
                (l2 - expect).abs() < 1.0e-12,
                "wide {j}: log2 {l2} != {d} dB × {k} = {expect}",
            );
        }
    }

    #[test]
    fn coder_partition_d5_ltmin_log2_row_order_by_width_monotone_in_db_per_cell() {
        // Strict monotonicity: a callback that everywhere returns
        // (cb_a − 1.0) dB produces a log2 cell that is strictly smaller
        // (in fact, shifted by exactly −log2(10)/10 ≈ −0.33219 from)
        // the log2 cell of a callback returning cb_a dB. Because every
        // partition's min is shifted by exactly −1 dB, the log2 shift
        // is identical at every cell.
        let cb_hi = |omega: u16| -> f64 { f64::from(omega) * 0.5 + 3.0 };
        let cb_lo = |omega: u16| -> f64 { f64::from(omega) * 0.5 + 2.0 };
        let hi = coder_partition_d5_ltmin_log2_row_order_by_width(cb_hi);
        let lo = coder_partition_d5_ltmin_log2_row_order_by_width(cb_lo);
        let shift = -10.0_f64.log2() / 10.0;
        for (i, (&h, &l)) in hi.narrow_band.iter().zip(lo.narrow_band.iter()).enumerate() {
            assert!(l < h, "narrow {i}: lo {l} not < hi {h}");
            let diff = l - h;
            assert!(
                (diff - shift).abs() < 1.0e-12,
                "narrow {i}: diff {diff} != {shift}",
            );
        }
        for (j, (&h, &l)) in hi.wide_band.iter().zip(lo.wide_band.iter()).enumerate() {
            assert!(l < h, "wide {j}: lo {l} not < hi {h}");
            let diff = l - h;
            assert!(
                (diff - shift).abs() < 1.0e-12,
                "wide {j}: diff {diff} != {shift}",
            );
        }
    }

    #[test]
    fn coder_partition_d5_ltmin_log2_row_order_by_width_is_idempotent_for_pure_callback() {
        // Pure callback → same log2 projection on repeated invocation.
        let cb = |omega: u16| -> f64 { f64::from(omega).cos() * 5.0 - 0.25 };
        let a = coder_partition_d5_ltmin_log2_row_order_by_width(cb);
        let b = coder_partition_d5_ltmin_log2_row_order_by_width(cb);
        for i in 0..12 {
            assert!((a.narrow_band[i] - b.narrow_band[i]).abs() < 1.0e-12);
        }
        for j in 0..20 {
            assert!((a.wide_band[j] - b.wide_band[j]).abs() < 1.0e-12);
        }
    }

    #[test]
    fn coder_partition_d5_ltmin_log2_row_order_by_width_dip_in_narrow_only_affects_narrow() {
        // A −20 dB dip in a single FFT line that lives inside the
        // narrow block (line ω = 50 lives in partition 3, narrow row
        // index 2) lowers exactly one narrow-band cell relative to the
        // baseline; wide-band cells are unchanged.
        let baseline = coder_partition_d5_ltmin_log2_row_order_by_width(|_| 0.0);
        let cb = |omega: u16| -> f64 {
            if omega == 50 {
                -20.0
            } else {
                0.0
            }
        };
        let split = coder_partition_d5_ltmin_log2_row_order_by_width(cb);
        for j in 0..20 {
            assert!(
                (split.wide_band[j] - baseline.wide_band[j]).abs() < 1.0e-12,
                "wide band cell {j} should be unaffected by a narrow-line dip",
            );
        }
        let mut narrow_dipped = 0u32;
        for i in 0..12 {
            if (split.narrow_band[i] - baseline.narrow_band[i]).abs() > 1.0e-12 {
                narrow_dipped += 1;
                assert!(
                    split.narrow_band[i] < baseline.narrow_band[i],
                    "narrow band cell {i} should have dipped",
                );
            }
        }
        assert_eq!(
            narrow_dipped, 1,
            "exactly one narrow band cell should dip from a single-line callback",
        );
    }

    #[test]
    fn coder_partition_d5_ltmin_log2_row_order_by_width_dip_in_wide_only_affects_wide() {
        // A −20 dB dip in a single FFT line that lives inside the
        // wide block (line ω = 300 lives in partition 18, wide row
        // index 5) lowers exactly one wide-band cell relative to the
        // baseline; narrow-band cells are unchanged.
        let baseline = coder_partition_d5_ltmin_log2_row_order_by_width(|_| 0.0);
        let cb = |omega: u16| -> f64 {
            if omega == 300 {
                -20.0
            } else {
                0.0
            }
        };
        let split = coder_partition_d5_ltmin_log2_row_order_by_width(cb);
        for i in 0..12 {
            assert!(
                (split.narrow_band[i] - baseline.narrow_band[i]).abs() < 1.0e-12,
                "narrow band cell {i} should be unaffected by a wide-line dip",
            );
        }
        let mut wide_dipped = 0u32;
        for j in 0..20 {
            if (split.wide_band[j] - baseline.wide_band[j]).abs() > 1.0e-12 {
                wide_dipped += 1;
                assert!(
                    split.wide_band[j] < baseline.wide_band[j],
                    "wide band cell {j} should have dipped",
                );
            }
        }
        assert_eq!(
            wide_dipped, 1,
            "exactly one wide band cell should dip from a single-line callback",
        );
    }

    #[test]
    fn coder_partition_d5_ltmin_log2_row_order_by_width_split_yields_a_partition_of_step59() {
        // narrow_band ++ wide_band, when mapped back to dB
        // (`10 · log10(2^cell) = cell · 10 · log10(2)`), equals
        // step 59's row-order LTmin vector index-by-index.
        let cb = |omega: u16| -> f64 { f64::from(omega) * 0.04 - 1.0 };
        let split = coder_partition_d5_ltmin_log2_row_order_by_width(cb);
        let row_order = coder_partition_d5_ltg_min_row_order(cb);
        let k = 10.0 * 2.0_f64.log10();
        for (i, &row_db) in row_order.iter().enumerate().take(12) {
            let recovered_db = split.narrow_band[i] * k;
            assert!(
                (recovered_db - row_db).abs() < 1.0e-9,
                "narrow band index {i}: recovered {recovered_db} dB != step 59 {row_db} dB",
            );
        }
        for (j, &row_db) in row_order.iter().enumerate().skip(12) {
            let idx = j - 12;
            let recovered_db = split.wide_band[idx] * k;
            assert!(
                (recovered_db - row_db).abs() < 1.0e-9,
                "wide band index {idx}: recovered {recovered_db} dB != step 59 {row_db} dB",
            );
        }
    }

    // ---- Phase 2 step 66 / r265 — width-gated wide-band signed
    // bit-budget reduction `Σ_n width_n · log2(LTmin_lin_n)` over
    // Table D.5 (algebraic collapse of the weighted total onto the
    // unweighted sum of step 65's `wide_band` subarray).
    #[test]
    fn coder_partition_d5_ltmin_log2_wide_band_bit_budget_total_zero_db_is_zero() {
        // 0 dB everywhere → linear 1.0 in every cell → log2(1.0) = 0
        // in every wide cell → total sum = 0.0 exactly.
        let total = coder_partition_d5_ltmin_log2_wide_band_bit_budget_total(|_| 0.0);
        assert!(total.abs() < 1.0e-12, "total {total} != 0.0");
    }

    #[test]
    fn coder_partition_d5_ltmin_log2_wide_band_bit_budget_total_finite_for_finite_callback() {
        // Finite real callback → finite total (each of the 20 wide
        // cells is finite, and the sum of 20 finite f64s is finite
        // outside catastrophic-cancellation overflow which a tame
        // bounded ramp cannot trigger).
        let cb = |omega: u16| -> f64 { -30.0 + f64::from(omega) * 0.05 };
        let total = coder_partition_d5_ltmin_log2_wide_band_bit_budget_total(cb);
        assert!(total.is_finite(), "total {total} is not finite");
    }

    #[test]
    fn coder_partition_d5_ltmin_log2_wide_band_bit_budget_total_matches_wide_band_sum() {
        // Algebraic identity: the reduction equals the unweighted sum
        // of step 65's `wide_band` subarray (a 20-cell straight sum).
        let cb = |omega: u16| -> f64 { f64::from(omega) * 0.25 - 4.5 };
        let total = coder_partition_d5_ltmin_log2_wide_band_bit_budget_total(cb);
        let split = coder_partition_d5_ltmin_log2_row_order_by_width(cb);
        let expect: f64 = split.wide_band.iter().sum();
        assert!(
            (total - expect).abs() < 1.0e-12,
            "total {total} != Σ wide_band {expect}",
        );
    }

    #[test]
    fn coder_partition_d5_ltmin_log2_wide_band_bit_budget_total_ignores_narrow_band_contributions()
    {
        // The reduction reads only `wide_band`. Verify by perturbing
        // narrow callbacks (ω = 50 lives in partition 3, a narrow
        // row) and confirming the total is unchanged from the
        // baseline that returns 0 dB everywhere.
        let baseline = coder_partition_d5_ltmin_log2_wide_band_bit_budget_total(|_| 0.0);
        let perturb_narrow = |omega: u16| -> f64 {
            if omega == 50 || omega == 30 || omega == 100 {
                -25.0
            } else {
                0.0
            }
        };
        let perturbed = coder_partition_d5_ltmin_log2_wide_band_bit_budget_total(perturb_narrow);
        assert!(
            (perturbed - baseline).abs() < 1.0e-12,
            "narrow-only perturbation changed total: baseline {baseline} → {perturbed}",
        );
    }

    #[test]
    fn coder_partition_d5_ltmin_log2_wide_band_bit_budget_total_three_db_is_pin() {
        // Spot pin: a uniform +3 dB callback drives every wide cell
        // to `log2(10^0.3) = 0.3 / log10(2)`. The total is exactly
        // 20 × that value.
        let total = coder_partition_d5_ltmin_log2_wide_band_bit_budget_total(|_| 3.0);
        let per_cell = 3.0_f64 / 10.0 / 2.0_f64.log10();
        let expect = 20.0 * per_cell;
        assert!(
            (total - expect).abs() < 1.0e-12,
            "total {total} != 20 × {per_cell} = {expect}",
        );
    }

    #[test]
    fn coder_partition_d5_ltmin_log2_wide_band_bit_budget_total_minus_three_db_is_sign_flipped() {
        // log2-of-10^(x/10) is odd around 0 dB. The total at uniform
        // −3 dB equals the additive inverse of the total at +3 dB
        // (each of the 20 wide cells flips sign identically).
        let hi = coder_partition_d5_ltmin_log2_wide_band_bit_budget_total(|_| 3.0);
        let lo = coder_partition_d5_ltmin_log2_wide_band_bit_budget_total(|_| -3.0);
        assert!(
            (hi + lo).abs() < 1.0e-12,
            "hi {hi} + lo {lo} != 0 (odd-symmetry broken)",
        );
    }

    #[test]
    fn coder_partition_d5_ltmin_log2_wide_band_bit_budget_total_uniform_db_scales_linearly() {
        // Linearity in dB: the per-cell mapping `db → log2(10^(db/10))`
        // is `db · log2(10) / 10`, a linear scaling. A uniform
        // callback at `2 · k` dB produces exactly twice the total of
        // a uniform callback at `k` dB.
        let one = coder_partition_d5_ltmin_log2_wide_band_bit_budget_total(|_| 5.0);
        let two = coder_partition_d5_ltmin_log2_wide_band_bit_budget_total(|_| 10.0);
        assert!(
            (two - 2.0 * one).abs() < 1.0e-12,
            "uniform 10 dB total {two} != 2 × uniform 5 dB total {one}",
        );
    }

    #[test]
    fn coder_partition_d5_ltmin_log2_wide_band_bit_budget_total_is_idempotent_for_pure_callback() {
        // Pure callback → identical total on repeated invocation.
        let cb = |omega: u16| -> f64 { f64::from(omega).sin() * 4.0 + 0.7 };
        let a = coder_partition_d5_ltmin_log2_wide_band_bit_budget_total(cb);
        let b = coder_partition_d5_ltmin_log2_wide_band_bit_budget_total(cb);
        assert!(
            (a - b).abs() < 1.0e-12,
            "non-deterministic total: {a} vs {b}"
        );
    }

    #[test]
    fn coder_partition_d5_ltmin_log2_wide_band_bit_budget_total_matches_weighted_full_sum_over_row_order(
    ) {
        // Width-gating algebraic identity: the reduction equals the
        // weighted sum `Σ_{n=1..=32} width_n · log2_n` taken over the
        // full row-order log2 vector — narrow rows weighted by 0,
        // wide rows weighted by 1. Reuse step 60's row-order width
        // vector and step 65's row-order log2 columns (concatenated)
        // and confirm equality with the direct reduction.
        let cb = |omega: u16| -> f64 { f64::from(omega) * 0.10 + 0.5 };
        let total = coder_partition_d5_ltmin_log2_wide_band_bit_budget_total(cb);
        let widths = coder_partition_d5_width_row_order();
        let row_order_db = coder_partition_d5_ltg_min_row_order(cb);
        let k = 10.0_f64.log2() / 10.0;
        let mut weighted = 0.0_f64;
        for (i, &db) in row_order_db.iter().enumerate() {
            weighted += f64::from(widths[i]) * (db * k);
        }
        assert!(
            (total - weighted).abs() < 1.0e-9,
            "wide-band sum {total} != weighted full sum {weighted}",
        );
    }

    #[test]
    fn coder_partition_d5_ltmin_log2_wide_band_bit_budget_total_dip_in_wide_only_lowers_total() {
        // A −20 dB dip in a single FFT line that lives inside the
        // wide block (line ω = 300 lives in partition 18) lowers the
        // total relative to the baseline. The dip drives exactly one
        // wide cell strictly negative (the rest stay at 0), so the
        // total drops by that cell's magnitude.
        let baseline = coder_partition_d5_ltmin_log2_wide_band_bit_budget_total(|_| 0.0);
        let dipped = coder_partition_d5_ltmin_log2_wide_band_bit_budget_total(|omega: u16| {
            if omega == 300 {
                -20.0
            } else {
                0.0
            }
        });
        assert!(
            dipped < baseline,
            "dipped total {dipped} should be strictly below baseline {baseline}",
        );
        // Expected magnitude: one cell dropped to log2(10^(−20/10)) =
        // −20 / 10 × log2(10) = −2 · log2(10).
        let expected_drop = 2.0 * 10.0_f64.log2();
        let actual_drop = baseline - dipped;
        assert!(
            (actual_drop - expected_drop).abs() < 1.0e-9,
            "actual drop {actual_drop} != expected −2·log2(10) = {expected_drop}",
        );
    }

    #[test]
    fn coder_partition_d5_ltmin_log2_wide_band_bit_budget_total_proportional_to_step63_wide_sum() {
        // Equivalence to step 63's wide-band dB sum scaled by the
        // dB → log2-linear constant `log2(10) / 10`. This pins the
        // reduction to the dB presentation without invoking step 65
        // a second time.
        let cb = |omega: u16| -> f64 { f64::from(omega) * 0.13 - 1.1 };
        let total = coder_partition_d5_ltmin_log2_wide_band_bit_budget_total(cb);
        let db_split = coder_partition_d5_ltmin_db_row_order_by_width(cb);
        let db_wide_sum: f64 = db_split.wide_band.iter().sum();
        let k = 10.0_f64.log2() / 10.0;
        let expect = db_wide_sum * k;
        assert!(
            (total - expect).abs() < 1.0e-9,
            "total {total} != db wide sum {db_wide_sum} × {k} = {expect}",
        );
    }

    // ---- Phase 2 step 67 / r266 — width-gated narrow-band complementary
    // signed bit-budget reduction `Σ_n (1 − width_n) · log2(LTmin_lin_n)`
    // over Table D.5 (algebraic collapse of the complementary weighted
    // total onto the unweighted sum of step 65's `narrow_band` subarray).
    #[test]
    fn coder_partition_d5_ltmin_log2_narrow_band_bit_budget_total_zero_db_is_zero() {
        // 0 dB everywhere → linear 1.0 in every cell → log2(1.0) = 0
        // in every narrow cell → total sum = 0.0 exactly.
        let total = coder_partition_d5_ltmin_log2_narrow_band_bit_budget_total(|_| 0.0);
        assert!(total.abs() < 1.0e-12, "total {total} != 0.0");
    }

    #[test]
    fn coder_partition_d5_ltmin_log2_narrow_band_bit_budget_total_finite_for_finite_callback() {
        // Finite real callback → finite total (each of the 12 narrow
        // cells is finite, and the sum of 12 finite f64s is finite
        // outside catastrophic-cancellation overflow which a tame
        // bounded ramp cannot trigger).
        let cb = |omega: u16| -> f64 { -30.0 + f64::from(omega) * 0.05 };
        let total = coder_partition_d5_ltmin_log2_narrow_band_bit_budget_total(cb);
        assert!(total.is_finite(), "total {total} is not finite");
    }

    #[test]
    fn coder_partition_d5_ltmin_log2_narrow_band_bit_budget_total_matches_narrow_band_sum() {
        // Algebraic identity: the reduction equals the unweighted sum
        // of step 65's `narrow_band` subarray (a 12-cell straight sum).
        let cb = |omega: u16| -> f64 { f64::from(omega) * 0.25 - 4.5 };
        let total = coder_partition_d5_ltmin_log2_narrow_band_bit_budget_total(cb);
        let split = coder_partition_d5_ltmin_log2_row_order_by_width(cb);
        let expect: f64 = split.narrow_band.iter().sum();
        assert!(
            (total - expect).abs() < 1.0e-12,
            "total {total} != Σ narrow_band {expect}",
        );
    }

    #[test]
    fn coder_partition_d5_ltmin_log2_narrow_band_bit_budget_total_ignores_wide_band_contributions()
    {
        // The reduction reads only `narrow_band`. Verify by perturbing
        // wide callbacks (ω = 300 lives in partition 18, a wide row;
        // ω = 450 lives in partition 27; ω = 500 lives in partition 31)
        // and confirming the total is unchanged from the baseline that
        // returns 0 dB everywhere.
        let baseline = coder_partition_d5_ltmin_log2_narrow_band_bit_budget_total(|_| 0.0);
        let perturb_wide = |omega: u16| -> f64 {
            if omega == 300 || omega == 450 || omega == 500 {
                -25.0
            } else {
                0.0
            }
        };
        let perturbed = coder_partition_d5_ltmin_log2_narrow_band_bit_budget_total(perturb_wide);
        assert!(
            (perturbed - baseline).abs() < 1.0e-12,
            "wide-only perturbation changed total: baseline {baseline} → {perturbed}",
        );
    }

    #[test]
    fn coder_partition_d5_ltmin_log2_narrow_band_bit_budget_total_three_db_is_pin() {
        // Spot pin: a uniform +3 dB callback drives every narrow cell
        // to `log2(10^0.3) = 0.3 / log10(2)`. The total is exactly
        // 12 × that value.
        let total = coder_partition_d5_ltmin_log2_narrow_band_bit_budget_total(|_| 3.0);
        let per_cell = 3.0_f64 / 10.0 / 2.0_f64.log10();
        let expect = 12.0 * per_cell;
        assert!(
            (total - expect).abs() < 1.0e-12,
            "total {total} != 12 × {per_cell} = {expect}",
        );
    }

    #[test]
    fn coder_partition_d5_ltmin_log2_narrow_band_bit_budget_total_minus_three_db_is_sign_flipped() {
        // log2-of-10^(x/10) is odd around 0 dB. The total at uniform
        // −3 dB equals the additive inverse of the total at +3 dB
        // (each of the 12 narrow cells flips sign identically).
        let hi = coder_partition_d5_ltmin_log2_narrow_band_bit_budget_total(|_| 3.0);
        let lo = coder_partition_d5_ltmin_log2_narrow_band_bit_budget_total(|_| -3.0);
        assert!(
            (hi + lo).abs() < 1.0e-12,
            "hi {hi} + lo {lo} != 0 (odd-symmetry broken)",
        );
    }

    #[test]
    fn coder_partition_d5_ltmin_log2_narrow_band_bit_budget_total_uniform_db_scales_linearly() {
        // Linearity in dB: the per-cell mapping `db → log2(10^(db/10))`
        // is `db · log2(10) / 10`, a linear scaling. A uniform
        // callback at `2 · k` dB produces exactly twice the total of
        // a uniform callback at `k` dB.
        let one = coder_partition_d5_ltmin_log2_narrow_band_bit_budget_total(|_| 5.0);
        let two = coder_partition_d5_ltmin_log2_narrow_band_bit_budget_total(|_| 10.0);
        assert!(
            (two - 2.0 * one).abs() < 1.0e-12,
            "uniform 10 dB total {two} != 2 × uniform 5 dB total {one}",
        );
    }

    #[test]
    fn coder_partition_d5_ltmin_log2_narrow_band_bit_budget_total_is_idempotent_for_pure_callback()
    {
        // Pure callback → identical total on repeated invocation.
        let cb = |omega: u16| -> f64 { f64::from(omega).sin() * 4.0 + 0.7 };
        let a = coder_partition_d5_ltmin_log2_narrow_band_bit_budget_total(cb);
        let b = coder_partition_d5_ltmin_log2_narrow_band_bit_budget_total(cb);
        assert!(
            (a - b).abs() < 1.0e-12,
            "non-deterministic total: {a} vs {b}"
        );
    }

    #[test]
    fn coder_partition_d5_ltmin_log2_narrow_band_bit_budget_total_matches_complementary_full_sum_over_row_order(
    ) {
        // Complementary width-gating algebraic identity: the reduction
        // equals the complementary weighted sum
        // `Σ_{n=1..=32} (1 − width_n) · log2_n` taken over the full
        // row-order log2 vector — narrow rows weighted by 1, wide
        // rows weighted by 0. Reuse step 60's row-order width vector
        // and step 65's row-order log2 columns (concatenated) and
        // confirm equality with the direct reduction.
        let cb = |omega: u16| -> f64 { f64::from(omega) * 0.10 + 0.5 };
        let total = coder_partition_d5_ltmin_log2_narrow_band_bit_budget_total(cb);
        let widths = coder_partition_d5_width_row_order();
        let row_order_db = coder_partition_d5_ltg_min_row_order(cb);
        let k = 10.0_f64.log2() / 10.0;
        let mut weighted = 0.0_f64;
        for (i, &db) in row_order_db.iter().enumerate() {
            // 1 − width_n complement; widths is [0; 12]+[1; 20] so the
            // complement reads as `1 − w` for each row.
            let comp = 1.0_f64 - f64::from(widths[i]);
            weighted += comp * (db * k);
        }
        assert!(
            (total - weighted).abs() < 1.0e-9,
            "narrow-band sum {total} != complementary full sum {weighted}",
        );
    }

    #[test]
    fn coder_partition_d5_ltmin_log2_narrow_band_bit_budget_total_dip_in_narrow_only_lowers_total()
    {
        // A −20 dB dip in a single FFT line that lives inside the
        // narrow block (line ω = 50 lives in partition 3) lowers the
        // total relative to the baseline. The dip drives exactly one
        // narrow cell strictly negative (the rest stay at 0), so the
        // total drops by that cell's magnitude.
        let baseline = coder_partition_d5_ltmin_log2_narrow_band_bit_budget_total(|_| 0.0);
        let dipped = coder_partition_d5_ltmin_log2_narrow_band_bit_budget_total(|omega: u16| {
            if omega == 50 {
                -20.0
            } else {
                0.0
            }
        });
        assert!(
            dipped < baseline,
            "dipped total {dipped} should be strictly below baseline {baseline}",
        );
        // Expected magnitude: one cell dropped to log2(10^(−20/10)) =
        // −20 / 10 × log2(10) = −2 · log2(10).
        let expected_drop = 2.0 * 10.0_f64.log2();
        let actual_drop = baseline - dipped;
        assert!(
            (actual_drop - expected_drop).abs() < 1.0e-9,
            "actual drop {actual_drop} != expected −2·log2(10) = {expected_drop}",
        );
    }

    #[test]
    fn coder_partition_d5_ltmin_log2_narrow_band_bit_budget_total_proportional_to_step63_narrow_sum(
    ) {
        // Equivalence to step 63's narrow-band dB sum scaled by the
        // dB → log2-linear constant `log2(10) / 10`. This pins the
        // reduction to the dB presentation without invoking step 65
        // a second time.
        let cb = |omega: u16| -> f64 { f64::from(omega) * 0.13 - 1.1 };
        let total = coder_partition_d5_ltmin_log2_narrow_band_bit_budget_total(cb);
        let db_split = coder_partition_d5_ltmin_db_row_order_by_width(cb);
        let db_narrow_sum: f64 = db_split.narrow_band.iter().sum();
        let k = 10.0_f64.log2() / 10.0;
        let expect = db_narrow_sum * k;
        assert!(
            (total - expect).abs() < 1.0e-9,
            "total {total} != db narrow sum {db_narrow_sum} × {k} = {expect}",
        );
    }

    #[test]
    fn coder_partition_d5_ltmin_log2_narrow_band_plus_wide_band_recovers_full_row_order_sum() {
        // Pairing identity: by the disjointness of step 65's
        // `narrow_band` / `wide_band` fields, the unweighted total of
        // the full 32-row log2 vector equals
        // `narrow_total + wide_total` exactly. This pins the
        // step 66 / step 67 pair as a partition of the row-order
        // signed bit-budget budget.
        let cb = |omega: u16| -> f64 { f64::from(omega) * 0.07 - 2.3 };
        let narrow = coder_partition_d5_ltmin_log2_narrow_band_bit_budget_total(cb);
        let wide = coder_partition_d5_ltmin_log2_wide_band_bit_budget_total(cb);
        let split = coder_partition_d5_ltmin_log2_row_order_by_width(cb);
        let full: f64 = split.narrow_band.iter().sum::<f64>() + split.wide_band.iter().sum::<f64>();
        assert!(
            (narrow + wide - full).abs() < 1.0e-12,
            "narrow {narrow} + wide {wide} != full row-order sum {full}",
        );
    }

    // -- Phase 2 step 68 (r267): fused paired (narrow_total, wide_total) ----

    #[test]
    fn coder_partition_d5_ltmin_log2_paired_totals_zero_db_is_zero_pair() {
        // 0 dB → linear 1.0 → log2 0.0 in every cell, so both totals
        // are exactly 0.0.
        let (narrow, wide) = coder_partition_d5_ltmin_log2_paired_bit_budget_totals(|_| 0.0);
        assert_eq!(narrow, 0.0);
        assert_eq!(wide, 0.0);
    }

    #[test]
    fn coder_partition_d5_ltmin_log2_paired_totals_match_standalone_steps_67_and_66() {
        // The fused pair is bit-identical to the standalone step 67
        // (narrow) and step 66 (wide) results for the same callback.
        let cb = |omega: u16| -> f64 { f64::from(omega) * 0.13 - 5.5 };
        let (narrow, wide) = coder_partition_d5_ltmin_log2_paired_bit_budget_totals(cb);
        let narrow_std = coder_partition_d5_ltmin_log2_narrow_band_bit_budget_total(cb);
        let wide_std = coder_partition_d5_ltmin_log2_wide_band_bit_budget_total(cb);
        assert_eq!(narrow, narrow_std);
        assert_eq!(wide, wide_std);
    }

    #[test]
    fn coder_partition_d5_ltmin_log2_paired_totals_invoke_callback_exactly_once_over_range() {
        // Callback fan-out is half the back-to-back step 67 + step 66
        // pairing: the fused accessor walks the FFT-line range once,
        // so its call count equals a single step-65 invocation, which
        // is exactly half the count of calling narrow + wide
        // standalone.
        use core::cell::Cell;
        let fused_calls = Cell::new(0usize);
        let _ = coder_partition_d5_ltmin_log2_paired_bit_budget_totals(|omega: u16| {
            fused_calls.set(fused_calls.get() + 1);
            f64::from(omega) * 0.01
        });

        let split_calls = Cell::new(0usize);
        let _ = coder_partition_d5_ltmin_log2_row_order_by_width(|omega: u16| {
            split_calls.set(split_calls.get() + 1);
            f64::from(omega) * 0.01
        });

        // The two standalone reductions are counted with separate
        // counters (each closure moves into exactly one call); their
        // sum is the back-to-back fan-out.
        let narrow_calls = Cell::new(0usize);
        let _ = coder_partition_d5_ltmin_log2_narrow_band_bit_budget_total(|omega: u16| {
            narrow_calls.set(narrow_calls.get() + 1);
            f64::from(omega) * 0.01
        });
        let wide_calls = Cell::new(0usize);
        let _ = coder_partition_d5_ltmin_log2_wide_band_bit_budget_total(|omega: u16| {
            wide_calls.set(wide_calls.get() + 1);
            f64::from(omega) * 0.01
        });
        let standalone_total = narrow_calls.get() + wide_calls.get();

        assert_eq!(
            fused_calls.get(),
            split_calls.get(),
            "fused accessor must invoke the callback exactly as many times as one step-65 pass",
        );
        assert_eq!(
            standalone_total,
            2 * fused_calls.get(),
            "back-to-back step 67 + step 66 must double the fused fan-out",
        );
    }

    #[test]
    fn coder_partition_d5_ltmin_log2_paired_totals_sum_recovers_full_row_order() {
        // Pairing identity through the fused accessor: narrow + wide
        // equals the unweighted full row-order sum exactly.
        let cb = |omega: u16| -> f64 { f64::from(omega) * 0.07 - 2.3 };
        let (narrow, wide) = coder_partition_d5_ltmin_log2_paired_bit_budget_totals(cb);
        let split = coder_partition_d5_ltmin_log2_row_order_by_width(cb);
        let full: f64 = split.narrow_band.iter().sum::<f64>() + split.wide_band.iter().sum::<f64>();
        assert!(
            (narrow + wide - full).abs() < 1.0e-12,
            "narrow {narrow} + wide {wide} != full row-order sum {full}",
        );
    }

    #[test]
    fn coder_partition_d5_ltmin_log2_paired_totals_blocks_are_independent() {
        // A perturbation confined to wide partitions (13..=32) moves
        // only `wide_total`; `narrow_total` is unchanged, and vice
        // versa. Confirms the two sums read disjoint subarrays.
        let baseline = coder_partition_d5_ltmin_log2_paired_bit_budget_totals(|_| 0.0);

        // `LTmin_n` is the inclusive *minimum* of `LTg(ω)` over the
        // partition's FFT-line range, so a perturbation only moves a
        // total when it drops a line *below* the baseline 0.0 — raising
        // a line above 0.0 leaves the min (and the total) unchanged.
        //
        // Wide partitions are 13..=32, spanning FFT lines in
        // `[209, 513]` (row 13's `[193, 209]` through row 32's
        // `[497, 513]`). omega = 400 is interior to a wide partition.
        let wide_perturbed =
            coder_partition_d5_ltmin_log2_paired_bit_budget_totals(|omega: u16| {
                if omega == 400 {
                    -4.0
                } else {
                    0.0
                }
            });
        assert_eq!(
            wide_perturbed.0, baseline.0,
            "narrow_total must be unaffected by a wide-only perturbation",
        );
        assert!(
            wide_perturbed.1 != baseline.1,
            "wide_total must move under a wide-only perturbation",
        );

        // Narrow partitions are 1..=12, spanning FFT lines in
        // `[1, 193]`. omega = 100 is interior to partition 7
        // (`[97, 113]`, width 0).
        let narrow_perturbed =
            coder_partition_d5_ltmin_log2_paired_bit_budget_totals(|omega: u16| {
                if omega == 100 {
                    -4.0
                } else {
                    0.0
                }
            });
        assert!(
            narrow_perturbed.0 != baseline.0,
            "narrow_total must move under a narrow-only perturbation",
        );
        assert_eq!(
            narrow_perturbed.1, baseline.1,
            "wide_total must be unaffected by a narrow-only perturbation",
        );
    }

    #[test]
    fn coder_partition_d5_ltmin_log2_paired_totals_is_idempotent_for_pure_callback() {
        let cb = |omega: u16| -> f64 { (f64::from(omega) * 0.001).sin() };
        let a = coder_partition_d5_ltmin_log2_paired_bit_budget_totals(cb);
        let b = coder_partition_d5_ltmin_log2_paired_bit_budget_totals(cb);
        assert_eq!(a, b);
    }

    // ---- Phase 2 step 69 / r268 — §D.1 Step 9 width-gated
    // signal-to-mask-ratio `SMR_n = Lsb(n) − LTmin_n` (dB) over
    // Table D.5 (printed p.115).
    #[test]
    fn coder_partition_d5_smr_zero_callbacks_yield_zero_everywhere() {
        // Lsb(n) = 0 dB and LTg(ω) = 0 dB → LTmin_n = 0 in every
        // partition → SMR_n = 0 − 0 = 0.0 exactly in all 32 cells.
        let smr = coder_partition_d5_smr_db_row_order_by_width(|_| 0.0, |_| 0.0);
        assert!(smr.narrow_band.iter().all(|&v| v == 0.0));
        assert!(smr.wide_band.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn coder_partition_d5_smr_uniform_pin() {
        // Lsb(n) = 96 dB everywhere, LTg(ω) = 20 dB everywhere →
        // LTmin_n = 20 → SMR_n = 96 − 20 = 76.0 dB exactly per cell.
        let smr = coder_partition_d5_smr_db_row_order_by_width(|_| 96.0, |_| 20.0);
        assert!(smr.narrow_band.iter().all(|&v| v == 76.0));
        assert!(smr.wide_band.iter().all(|&v| v == 76.0));
    }

    #[test]
    fn coder_partition_d5_smr_matches_step63_difference_cell_wise() {
        // Strict-composition cross-check: every cell equals
        // `lsb(n) − step63_ltmin_db(n)` reconstructed independently
        // under non-trivial callbacks on both inputs.
        let lsb = |n: u16| -> f64 { f64::from(n) * 1.75 + 40.0 };
        let ltg = |omega: u16| -> f64 { (f64::from(omega) * 0.01).sin() * 12.0 + 30.0 };
        let smr = coder_partition_d5_smr_db_row_order_by_width(lsb, ltg);
        let ltmin = coder_partition_d5_ltmin_db_row_order_by_width(ltg);
        for i in 0..12 {
            let expect = lsb(i as u16 + 1) - ltmin.narrow_band[i];
            assert!(
                (smr.narrow_band[i] - expect).abs() < 1.0e-12,
                "narrow[{i}] = {} != {expect}",
                smr.narrow_band[i],
            );
        }
        for j in 0..20 {
            let expect = lsb(j as u16 + 13) - ltmin.wide_band[j];
            assert!(
                (smr.wide_band[j] - expect).abs() < 1.0e-12,
                "wide[{j}] = {} != {expect}",
                smr.wide_band[j],
            );
        }
    }

    #[test]
    fn coder_partition_d5_smr_partition_index_mapping_row_order() {
        // Lsb(n) = n with a flat 0-dB threshold pins the subarray →
        // partition mapping: narrow_band[i] = i + 1 (partitions
        // 1..=12), wide_band[j] = j + 13 (partitions 13..=32).
        let smr = coder_partition_d5_smr_db_row_order_by_width(f64::from, |_| 0.0);
        for (i, &v) in smr.narrow_band.iter().enumerate() {
            assert_eq!(v, (i + 1) as f64, "narrow[{i}]");
        }
        for (j, &v) in smr.wide_band.iter().enumerate() {
            assert_eq!(v, (j + 13) as f64, "wide[{j}]");
        }
    }

    #[test]
    fn coder_partition_d5_smr_sign_semantics() {
        // Lsb below the threshold → strictly negative SMR (fully
        // masked); above → strictly positive (audible content).
        let masked = coder_partition_d5_smr_db_row_order_by_width(|_| 10.0, |_| 25.0);
        assert!(masked.narrow_band.iter().all(|&v| v == -15.0));
        assert!(masked.wide_band.iter().all(|&v| v == -15.0));
        let audible = coder_partition_d5_smr_db_row_order_by_width(|_| 60.0, |_| 25.0);
        assert!(audible.narrow_band.iter().all(|&v| v == 35.0));
        assert!(audible.wide_band.iter().all(|&v| v == 35.0));
    }

    #[test]
    fn coder_partition_d5_smr_lsb_fanout_once_per_partition_ascending() {
        // The Lsb(n) callback is invoked exactly once per partition
        // n ∈ 1..=32 in ascending row order; the LTg(ω) callback
        // fan-out equals exactly one step-63 pass (one call per FFT
        // line in Σ_{n=1..=32} (ωhigh_n − ωlow_n + 1)).
        use core::cell::{Cell, RefCell};
        let seen = RefCell::new(Vec::new());
        let ltg_calls = Cell::new(0_u32);
        let _ = coder_partition_d5_smr_db_row_order_by_width(
            |n: u16| {
                seen.borrow_mut().push(n);
                0.0
            },
            |_omega: u16| {
                ltg_calls.set(ltg_calls.get() + 1);
                0.0
            },
        );
        let expected: Vec<u16> = (1..=32).collect();
        assert_eq!(*seen.borrow(), expected, "Lsb fan-out / order");
        // One step-63 pass: count it independently.
        let direct = Cell::new(0_u32);
        let _ = coder_partition_d5_ltmin_db_row_order_by_width(|_omega: u16| {
            direct.set(direct.get() + 1);
            0.0
        });
        assert_eq!(ltg_calls.get(), direct.get(), "LTg fan-out");
    }

    #[test]
    fn coder_partition_d5_smr_ltg_dip_raises_exactly_one_interior_partition() {
        // A −30 dB LTg dip at the interior line ω = 300 (not a shared
        // 16k + 1 boundary) lowers exactly one partition's LTmin by
        // 30 dB, raising that partition's SMR by exactly +30 dB; all
        // other cells are unchanged. The affected partition is found
        // via the step 56 inverse lookup.
        let n = first_partition_containing_line(300).expect("ω = 300 is in-table");
        assert!((13..=32).contains(&n), "ω = 300 lives in the wide block");
        let baseline = coder_partition_d5_smr_db_row_order_by_width(|_| 50.0, |_| 0.0);
        let dipped = coder_partition_d5_smr_db_row_order_by_width(
            |_| 50.0,
            |omega: u16| if omega == 300 { -30.0 } else { 0.0 },
        );
        for i in 0..12 {
            assert_eq!(
                dipped.narrow_band[i], baseline.narrow_band[i],
                "narrow[{i}]"
            );
        }
        for j in 0..20 {
            let expect = if j as u16 + 13 == n {
                baseline.wide_band[j] + 30.0
            } else {
                baseline.wide_band[j]
            };
            assert_eq!(dipped.wide_band[j], expect, "wide[{j}]");
        }
    }

    #[test]
    fn coder_partition_d5_smr_is_idempotent_for_pure_callbacks() {
        let lsb = |n: u16| -> f64 { f64::from(n) * 0.5 + 30.0 };
        let ltg = |omega: u16| -> f64 { (f64::from(omega) * 0.002).cos() * 8.0 };
        let a = coder_partition_d5_smr_db_row_order_by_width(lsb, ltg);
        let b = coder_partition_d5_smr_db_row_order_by_width(lsb, ltg);
        assert_eq!(a.narrow_band, b.narrow_band);
        assert_eq!(a.wide_band, b.wide_band);
    }

    // ---- Phase 2 step 70 / r269 — §D.1 Step 9 row-order
    // signal-to-mask-ratio vector `[SMR_1 … SMR_32]` (dB) over
    // Table D.5 (printed p.115).
    #[test]
    fn coder_partition_d5_smr_row_order_zero_callbacks_yield_zero_everywhere() {
        // Lsb(n) = 0 dB and LTg(ω) = 0 dB → LTmin_n = 0 in every
        // partition → SMR_n = 0 − 0 = 0.0 exactly in all 32 rows.
        let smr = coder_partition_d5_smr_db_row_order(|_| 0.0, |_| 0.0);
        assert!(smr.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn coder_partition_d5_smr_row_order_uniform_pin() {
        // Lsb(n) = 96 dB everywhere, LTg(ω) = 20 dB everywhere →
        // LTmin_n = 20 → SMR_n = 96 − 20 = 76.0 dB exactly per row.
        let smr = coder_partition_d5_smr_db_row_order(|_| 96.0, |_| 20.0);
        assert!(smr.iter().all(|&v| v == 76.0));
    }

    #[test]
    fn coder_partition_d5_smr_row_order_matches_step59_difference_cell_wise() {
        // Strict-composition cross-check: every row equals
        // `lsb(n) − step59_ltmin[n − 1]` reconstructed independently
        // under non-trivial callbacks on both inputs.
        let lsb = |n: u16| -> f64 { f64::from(n) * 1.75 + 40.0 };
        let ltg = |omega: u16| -> f64 { (f64::from(omega) * 0.01).sin() * 12.0 + 30.0 };
        let smr = coder_partition_d5_smr_db_row_order(lsb, ltg);
        let ltmin = coder_partition_d5_ltg_min_row_order(ltg);
        for i in 0..32 {
            let expect = lsb(i as u16 + 1) - ltmin[i];
            assert!(
                (smr[i] - expect).abs() < 1.0e-12,
                "out[{i}] = {} != {expect}",
                smr[i],
            );
        }
    }

    #[test]
    fn coder_partition_d5_smr_row_order_partition_index_mapping() {
        // Lsb(n) = n with a flat 0-dB threshold pins the row →
        // partition mapping: out[i] = SMR_{i + 1} = i + 1.
        let smr = coder_partition_d5_smr_db_row_order(f64::from, |_| 0.0);
        for (i, &v) in smr.iter().enumerate() {
            assert_eq!(v, (i + 1) as f64, "out[{i}]");
        }
    }

    #[test]
    fn coder_partition_d5_smr_row_order_sign_semantics() {
        // Lsb below the threshold → strictly negative SMR (fully
        // masked); above → strictly positive (audible content).
        let masked = coder_partition_d5_smr_db_row_order(|_| 10.0, |_| 25.0);
        assert!(masked.iter().all(|&v| v == -15.0));
        let audible = coder_partition_d5_smr_db_row_order(|_| 60.0, |_| 25.0);
        assert!(audible.iter().all(|&v| v == 35.0));
    }

    #[test]
    fn coder_partition_d5_smr_row_order_lsb_fanout_once_per_partition_ascending() {
        // The Lsb(n) callback is invoked exactly once per partition
        // n ∈ 1..=32 in ascending row order; the LTg(ω) callback
        // fan-out equals exactly one step-59 pass (one call per FFT
        // line in Σ_{n=1..=32} (ωhigh_n − ωlow_n + 1)).
        use core::cell::{Cell, RefCell};
        let seen = RefCell::new(Vec::new());
        let ltg_calls = Cell::new(0_u32);
        let _ = coder_partition_d5_smr_db_row_order(
            |n: u16| {
                seen.borrow_mut().push(n);
                0.0
            },
            |_omega: u16| {
                ltg_calls.set(ltg_calls.get() + 1);
                0.0
            },
        );
        let expected: Vec<u16> = (1..=32).collect();
        assert_eq!(*seen.borrow(), expected, "Lsb fan-out / order");
        // One step-59 pass: count it independently.
        let direct = Cell::new(0_u32);
        let _ = coder_partition_d5_ltg_min_row_order(|_omega: u16| {
            direct.set(direct.get() + 1);
            0.0
        });
        assert_eq!(ltg_calls.get(), direct.get(), "LTg fan-out");
    }

    #[test]
    fn coder_partition_d5_smr_row_order_bit_identical_to_step69_split() {
        // Step 63's by-width LTmin cells are index-preserving copies
        // of the step 59 row-order vector, so the row-order SMR must
        // equal the step 69 split read back in row order bit-for-bit
        // under non-trivial callbacks: out[0..12] == narrow_band,
        // out[12..32] == wide_band (exact ==, not approximate).
        let lsb = |n: u16| -> f64 { f64::from(n) * 2.25 + 17.0 };
        let ltg = |omega: u16| -> f64 { (f64::from(omega) * 0.007).cos() * 9.0 + 41.0 };
        let row = coder_partition_d5_smr_db_row_order(lsb, ltg);
        let split = coder_partition_d5_smr_db_row_order_by_width(lsb, ltg);
        assert_eq!(&row[..12], &split.narrow_band[..], "narrow block");
        assert_eq!(&row[12..], &split.wide_band[..], "wide block");
    }

    #[test]
    fn coder_partition_d5_smr_row_order_ltg_dip_raises_exactly_one_interior_partition() {
        // A −30 dB LTg dip at the interior line ω = 300 (not a shared
        // 16k + 1 boundary) lowers exactly one partition's LTmin by
        // 30 dB, raising that row's SMR by exactly +30 dB; all other
        // rows are unchanged. The affected partition is found via the
        // step 56 inverse lookup.
        let n = first_partition_containing_line(300).expect("ω = 300 is in-table");
        let baseline = coder_partition_d5_smr_db_row_order(|_| 50.0, |_| 0.0);
        let dipped = coder_partition_d5_smr_db_row_order(
            |_| 50.0,
            |omega: u16| if omega == 300 { -30.0 } else { 0.0 },
        );
        for i in 0..32 {
            let expect = if i as u16 + 1 == n {
                baseline[i] + 30.0
            } else {
                baseline[i]
            };
            assert_eq!(dipped[i], expect, "out[{i}]");
        }
    }

    #[test]
    fn coder_partition_d5_smr_row_order_is_idempotent_for_pure_callbacks() {
        let lsb = |n: u16| -> f64 { f64::from(n) * 0.5 + 30.0 };
        let ltg = |omega: u16| -> f64 { (f64::from(omega) * 0.002).cos() * 8.0 };
        let a = coder_partition_d5_smr_db_row_order(lsb, ltg);
        let b = coder_partition_d5_smr_db_row_order(lsb, ltg);
        assert_eq!(a, b);
    }

    // ---- Phase 2 step 71 / r270 — §D.1 Step 9 paired
    // `(SMR_n, width_n)` row-order vector over Table D.5 (printed
    // p.115). The SMR analogue of step 61's paired
    // `(LTmin_n, width_n)` vector.
    #[test]
    fn coder_partition_d5_smr_pair_returns_exactly_thirty_two_pairs() {
        let v = coder_partition_d5_smr_row_order(|_| 0.0, |_| 0.0);
        assert_eq!(v.len(), 32);
    }

    #[test]
    fn coder_partition_d5_smr_pair_zero_callbacks_yield_zero_smr_everywhere() {
        // Lsb(n) = 0 dB and LTg(ω) = 0 dB → SMR_n = 0 − 0 = 0.0 in
        // every row; the width column is the static table value.
        let v = coder_partition_d5_smr_row_order(|_| 0.0, |_| 0.0);
        assert!(v.iter().all(|r| r.smr_db == 0.0));
    }

    #[test]
    fn coder_partition_d5_smr_pair_uniform_pin() {
        // Lsb(n) = 96, LTg(ω) = 20 → SMR_n = 76.0 dB exactly per row.
        let v = coder_partition_d5_smr_row_order(|_| 96.0, |_| 20.0);
        assert!(v.iter().all(|r| r.smr_db == 76.0));
    }

    #[test]
    fn coder_partition_d5_smr_pair_smr_column_matches_step70_cell_wise() {
        // Strict-composition cross-check: the smr_db column equals the
        // step-70 bare row-order SMR vector cell-for-cell under
        // non-trivial callbacks (exact ==, not approximate — pure
        // re-presentation, no arithmetic introduced).
        let lsb = |n: u16| -> f64 { f64::from(n) * 1.75 + 40.0 };
        let ltg = |omega: u16| -> f64 { (f64::from(omega) * 0.01).sin() * 12.0 + 30.0 };
        let v = coder_partition_d5_smr_row_order(lsb, ltg);
        let smr = coder_partition_d5_smr_db_row_order(lsb, ltg);
        for i in 0..32 {
            assert_eq!(v[i].smr_db, smr[i], "smr_db[{i}]");
        }
    }

    #[test]
    fn coder_partition_d5_smr_pair_width_column_matches_step60() {
        // The width column equals the step-60 static row-order width
        // vector cell-for-cell, regardless of the callbacks.
        let widths = coder_partition_d5_width_row_order();
        let a = coder_partition_d5_smr_row_order(|_| 0.0, |_| 0.0);
        let b = coder_partition_d5_smr_row_order(f64::from, |omega| f64::from(omega) * 0.3);
        for i in 0..32 {
            assert_eq!(a[i].width_n, widths[i], "a.width_n[{i}]");
            assert_eq!(b[i].width_n, widths[i], "b.width_n[{i}]");
        }
    }

    #[test]
    fn coder_partition_d5_smr_pair_width_column_is_twelve_zeros_then_twenty_ones() {
        // Table D.5 literal: width_n = 0 for n ∈ 1..=12 (array indices
        // 0..=11), width_n = 1 for n ∈ 13..=32 (array indices 12..=31).
        let v = coder_partition_d5_smr_row_order(|_| 0.0, |_| 0.0);
        for r in &v[..12] {
            assert_eq!(r.width_n, 0);
        }
        for r in &v[12..] {
            assert_eq!(r.width_n, 1);
        }
    }

    #[test]
    fn coder_partition_d5_smr_pair_partition_index_mapping() {
        // Lsb(n) = n with a flat 0-dB threshold pins the row →
        // partition mapping: out[i].smr_db = SMR_{i + 1} = i + 1.
        let v = coder_partition_d5_smr_row_order(f64::from, |_| 0.0);
        for (i, r) in v.iter().enumerate() {
            assert_eq!(r.smr_db, (i + 1) as f64, "smr_db[{i}]");
        }
    }

    #[test]
    fn coder_partition_d5_smr_pair_sign_semantics() {
        // Lsb below threshold → strictly negative SMR (fully masked);
        // above → strictly positive (audible). Width unaffected.
        let masked = coder_partition_d5_smr_row_order(|_| 10.0, |_| 25.0);
        assert!(masked.iter().all(|r| r.smr_db == -15.0));
        let audible = coder_partition_d5_smr_row_order(|_| 60.0, |_| 25.0);
        assert!(audible.iter().all(|r| r.smr_db == 35.0));
    }

    #[test]
    fn coder_partition_d5_smr_pair_matches_paired_step61_pattern() {
        // The pairing pattern is identical to step 61's
        // (LTmin_n, width_n) vector: same width column at every row,
        // and the smr_db column is `lsb(n) − step61.ltmin_db` per row.
        let lsb = |n: u16| -> f64 { f64::from(n) * 2.25 + 17.0 };
        let ltg = |omega: u16| -> f64 { (f64::from(omega) * 0.007).cos() * 9.0 + 41.0 };
        let smr_pair = coder_partition_d5_smr_row_order(lsb, ltg);
        let ltmin_pair = coder_partition_d5_reduction_row_order(ltg);
        for i in 0..32 {
            assert_eq!(smr_pair[i].width_n, ltmin_pair[i].width_n, "width_n[{i}]");
            let expect = lsb(i as u16 + 1) - ltmin_pair[i].ltmin_db;
            assert!(
                (smr_pair[i].smr_db - expect).abs() < 1.0e-12,
                "smr_db[{i}] = {} != {expect}",
                smr_pair[i].smr_db,
            );
        }
    }

    #[test]
    fn coder_partition_d5_smr_pair_lsb_fanout_once_per_partition_ascending() {
        // Lsb(n) invoked once per partition n ∈ 1..=32 ascending; the
        // LTg(ω) fan-out equals exactly one step-59 pass.
        use core::cell::{Cell, RefCell};
        let seen = RefCell::new(Vec::new());
        let ltg_calls = Cell::new(0_u32);
        let _ = coder_partition_d5_smr_row_order(
            |n: u16| {
                seen.borrow_mut().push(n);
                0.0
            },
            |_omega: u16| {
                ltg_calls.set(ltg_calls.get() + 1);
                0.0
            },
        );
        let expected: Vec<u16> = (1..=32).collect();
        assert_eq!(*seen.borrow(), expected, "Lsb fan-out / order");
        let direct = Cell::new(0_u32);
        let _ = coder_partition_d5_ltg_min_row_order(|_omega: u16| {
            direct.set(direct.get() + 1);
            0.0
        });
        assert_eq!(ltg_calls.get(), direct.get(), "LTg fan-out");
    }

    #[test]
    fn coder_partition_d5_smr_pair_ltg_dip_raises_exactly_one_interior_partition() {
        // A −30 dB LTg dip at interior line ω = 300 raises exactly one
        // partition's smr_db by +30 dB; widths and all other rows are
        // unchanged.
        let n = first_partition_containing_line(300).expect("ω = 300 is in-table");
        let baseline = coder_partition_d5_smr_row_order(|_| 50.0, |_| 0.0);
        let dipped = coder_partition_d5_smr_row_order(
            |_| 50.0,
            |omega: u16| if omega == 300 { -30.0 } else { 0.0 },
        );
        for i in 0..32 {
            let expect = if i as u16 + 1 == n {
                baseline[i].smr_db + 30.0
            } else {
                baseline[i].smr_db
            };
            assert_eq!(dipped[i].smr_db, expect, "smr_db[{i}]");
            assert_eq!(dipped[i].width_n, baseline[i].width_n, "width_n[{i}]");
        }
    }

    #[test]
    fn coder_partition_d5_smr_pair_is_idempotent_for_pure_callbacks() {
        let lsb = |n: u16| -> f64 { f64::from(n) * 0.5 + 30.0 };
        let ltg = |omega: u16| -> f64 { (f64::from(omega) * 0.002).cos() * 8.0 };
        let a = coder_partition_d5_smr_row_order(lsb, ltg);
        let b = coder_partition_d5_smr_row_order(lsb, ltg);
        assert_eq!(a, b);
    }

    // ---- Phase 2 step 72 / r271 — §C.1.5.2.7 "Bit allocation"
    // per-partition mask-to-noise ratio `MNR_n = SNR_n − SMR_n`
    // row-order vector over Table D.5 (printed p.73). The per-iteration
    // successor of step 71's paired `(SMR_n, width_n)` vector.
    #[test]
    fn coder_partition_d5_mnr_returns_exactly_thirty_two_rows() {
        let v = coder_partition_d5_mnr_row_order(|_| 0.0, |_| 0.0, |_| 0.0);
        assert_eq!(v.len(), 32);
    }

    #[test]
    fn coder_partition_d5_mnr_zero_callbacks_yield_zero_everywhere() {
        // SNR(n) = 0, SMR_n = 0 − 0 = 0 → MNR_n = 0 − 0 = 0 in every
        // row; width is the static table value.
        let v = coder_partition_d5_mnr_row_order(|_| 0.0, |_| 0.0, |_| 0.0);
        assert!(v.iter().all(|r| r.mnr_db == 0.0 && r.smr_db == 0.0));
    }

    #[test]
    fn coder_partition_d5_mnr_uniform_pin() {
        // SNR(n) = 30, Lsb(n) = 96, LTg(ω) = 20 → SMR_n = 76.0,
        // MNR_n = 30 − 76 = −46.0 dB exactly per row.
        let v = coder_partition_d5_mnr_row_order(|_| 30.0, |_| 96.0, |_| 20.0);
        assert!(v.iter().all(|r| r.smr_db == 76.0 && r.mnr_db == -46.0));
    }

    #[test]
    fn coder_partition_d5_mnr_is_snr_minus_smr_cell_wise() {
        // The verbatim §C.1.5.2.7 definition MNR = SNR − SMR holds at
        // every row against the step-71 SMR column directly.
        let snr = |n: u16| -> f64 { f64::from(n) * 1.5 + 12.0 };
        let lsb = |n: u16| -> f64 { f64::from(n) * 1.75 + 40.0 };
        let ltg = |omega: u16| -> f64 { (f64::from(omega) * 0.01).sin() * 12.0 + 30.0 };
        let mnr = coder_partition_d5_mnr_row_order(snr, lsb, ltg);
        let smr = coder_partition_d5_smr_row_order(lsb, ltg);
        for i in 0..32 {
            let n = i as u16 + 1;
            assert_eq!(mnr[i].smr_db, smr[i].smr_db, "smr_db[{i}]");
            assert!(
                (mnr[i].mnr_db - (snr(n) - smr[i].smr_db)).abs() < 1.0e-12,
                "mnr_db[{i}]",
            );
        }
    }

    #[test]
    fn coder_partition_d5_mnr_smr_and_width_columns_pass_through_step71() {
        // The smr_db / width_n columns are bit-identical to step 71's
        // paired SMR vector under non-trivial callbacks, regardless of
        // the SNR callback.
        let lsb = |n: u16| -> f64 { f64::from(n) * 2.25 + 17.0 };
        let ltg = |omega: u16| -> f64 { (f64::from(omega) * 0.007).cos() * 9.0 + 41.0 };
        let step71 = coder_partition_d5_smr_row_order(lsb, ltg);
        let mnr = coder_partition_d5_mnr_row_order(|n| f64::from(n) * 3.1, lsb, ltg);
        for i in 0..32 {
            assert_eq!(mnr[i].smr_db, step71[i].smr_db, "smr_db[{i}]");
            assert_eq!(mnr[i].width_n, step71[i].width_n, "width_n[{i}]");
        }
    }

    #[test]
    fn coder_partition_d5_mnr_width_column_is_twelve_zeros_then_twenty_ones() {
        // Table D.5 literal width pattern survives unchanged.
        let v = coder_partition_d5_mnr_row_order(|_| 0.0, |_| 0.0, |_| 0.0);
        for r in &v[..12] {
            assert_eq!(r.width_n, 0);
        }
        for r in &v[12..] {
            assert_eq!(r.width_n, 1);
        }
    }

    #[test]
    fn coder_partition_d5_mnr_partition_index_mapping() {
        // SNR(n) = n, Lsb(n) = 0, LTg = 0 → SMR_n = 0 →
        // MNR_n = n − 0 = n = i + 1.
        let v = coder_partition_d5_mnr_row_order(f64::from, |_| 0.0, |_| 0.0);
        for (i, r) in v.iter().enumerate() {
            assert_eq!(r.mnr_db, (i + 1) as f64, "mnr_db[{i}]");
        }
    }

    #[test]
    fn coder_partition_d5_mnr_minimum_marks_greatest_benefit_subband() {
        // §C.1.5.2.7: the loop bumps "the subband that has the greatest
        // benefit" = the minimal MNR. With a flat SNR a single raised
        // SMR partition becomes the unique minimum-MNR row.
        let n = first_partition_containing_line(300).expect("ω = 300 is in-table");
        // A −30 dB LTg dip at line ω = 300 lowers that partition's
        // LTmin, raising its SMR by +30 dB; with a flat SNR that raises
        // its MNR-deficit, so MNR = SNR − SMR is at its smallest there.
        let v = coder_partition_d5_mnr_row_order(
            |_| 20.0,
            |_| 50.0,
            |omega: u16| if omega == 300 { -30.0 } else { 0.0 },
        );
        // The dipped (raised-SMR) partition has the smallest (most
        // negative) MNR — it is the unique argmin.
        let argmin = v
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.mnr_db.partial_cmp(&b.1.mnr_db).unwrap())
            .map(|(i, _)| i as u16 + 1)
            .unwrap();
        assert_eq!(argmin, n);
    }

    #[test]
    fn coder_partition_d5_mnr_snr_fanout_once_per_partition_ascending() {
        // SNR(n) invoked once per partition n ∈ 1..=32 ascending.
        use core::cell::RefCell;
        let seen = RefCell::new(Vec::new());
        let _ = coder_partition_d5_mnr_row_order(
            |n: u16| {
                seen.borrow_mut().push(n);
                0.0
            },
            |_| 0.0,
            |_| 0.0,
        );
        let expected: Vec<u16> = (1..=32).collect();
        assert_eq!(*seen.borrow(), expected);
    }

    #[test]
    fn coder_partition_d5_mnr_sign_semantics() {
        // High SMR + low SNR → small/negative MNR (needs bits);
        // low SMR + high SNR → large positive MNR (already protected).
        let needs = coder_partition_d5_mnr_row_order(|_| 5.0, |_| 60.0, |_| 25.0);
        assert!(needs.iter().all(|r| r.smr_db == 35.0 && r.mnr_db == -30.0));
        let protected = coder_partition_d5_mnr_row_order(|_| 80.0, |_| 10.0, |_| 25.0);
        assert!(protected
            .iter()
            .all(|r| r.smr_db == -15.0 && r.mnr_db == 95.0));
    }

    #[test]
    fn coder_partition_d5_mnr_is_idempotent_for_pure_callbacks() {
        let snr = |n: u16| -> f64 { f64::from(n) * 0.9 + 11.0 };
        let lsb = |n: u16| -> f64 { f64::from(n) * 0.5 + 30.0 };
        let ltg = |omega: u16| -> f64 { (f64::from(omega) * 0.002).cos() * 8.0 };
        let a = coder_partition_d5_mnr_row_order(snr, lsb, ltg);
        let b = coder_partition_d5_mnr_row_order(snr, lsb, ltg);
        assert_eq!(a, b);
    }

    // ----- Phase 2 step 73 (r272): §C.1.5.2.7 minimal-MNR selection -----

    #[test]
    fn coder_partition_d5_min_mnr_selects_unique_minimum() {
        // A −30 dB LTg dip at ω = 300 raises one partition's SMR by +30 dB,
        // making its MNR = SNR − SMR the unique smallest (most negative).
        // §C.1.5.2.7: the loop selects "the subband with the minimal MNR".
        let n = first_partition_containing_line(300).expect("ω = 300 is in-table");
        let mnr = coder_partition_d5_mnr_row_order(
            |_| 20.0,
            |_| 50.0,
            |omega: u16| if omega == 300 { -30.0 } else { 0.0 },
        );
        let sel = coder_partition_d5_min_mnr(&mnr);
        assert_eq!(sel.partition_n, n);
        // The selection echoes the winning step-72 row verbatim.
        assert_eq!(sel.mnr_db, mnr[(n - 1) as usize].mnr_db);
        assert_eq!(sel.smr_db, mnr[(n - 1) as usize].smr_db);
        assert_eq!(sel.width_n, mnr[(n - 1) as usize].width_n);
        // The dipped partition's MNR is genuinely below the flat rest.
        assert!(sel.mnr_db < mnr[0].mnr_db);
    }

    #[test]
    fn coder_partition_d5_min_mnr_partition_index_mapping() {
        // SNR(n) = n with flat SMR makes MNR strictly ascending in n, so
        // partition n = 1 (array index 0) holds the unique minimum.
        let mnr = coder_partition_d5_mnr_row_order(f64::from, |_| 0.0, |_| 0.0);
        let sel = coder_partition_d5_min_mnr(&mnr);
        assert_eq!(sel.partition_n, 1);
        assert_eq!(sel.mnr_db, 1.0); // SNR(1) − 0 = 1
    }

    #[test]
    fn coder_partition_d5_min_mnr_high_index_minimum() {
        // SNR(n) = 100 − n makes MNR strictly descending in n, so the
        // last partition n = 32 (array index 31) holds the minimum.
        let mnr = coder_partition_d5_mnr_row_order(|n| 100.0 - f64::from(n), |_| 0.0, |_| 0.0);
        let sel = coder_partition_d5_min_mnr(&mnr);
        assert_eq!(sel.partition_n, 32);
        assert_eq!(sel.mnr_db, 68.0); // 100 − 32 = 68
    }

    #[test]
    fn coder_partition_d5_min_mnr_ties_resolve_to_lowest_index() {
        // All-equal MNR: the spec selects "the" subband, so the
        // deterministic tie-break is the lowest partition index — the
        // row-order scan keeps the first occurrence.
        let mnr = coder_partition_d5_mnr_row_order(|_| 7.0, |_| 7.0, |_| 0.0);
        // smr_db is 7 − 0 = 7 (uniform), mnr_db is 7 − 7 = 0 (uniform).
        assert!(mnr.iter().all(|r| r.mnr_db == 0.0));
        let sel = coder_partition_d5_min_mnr(&mnr);
        assert_eq!(sel.partition_n, 1);
        assert_eq!(sel.mnr_db, 0.0);
    }

    #[test]
    fn coder_partition_d5_min_mnr_carries_width_and_smr_of_winner() {
        // Pick an interior partition in the wide band (width_n = 1) as the
        // unique minimum and confirm both pass-through columns.
        let n = first_partition_containing_line(300).expect("ω = 300 is in-table");
        assert!(n >= 13, "ω = 300 falls in a width_n = 1 partition");
        let mnr = coder_partition_d5_mnr_row_order(
            |_| 0.0,
            |_| 40.0,
            |omega: u16| if omega == 300 { -30.0 } else { 0.0 },
        );
        let sel = coder_partition_d5_min_mnr(&mnr);
        assert_eq!(sel.partition_n, n);
        assert_eq!(sel.width_n, 1);
        assert_eq!(sel.smr_db, mnr[(n - 1) as usize].smr_db);
    }

    #[test]
    fn coder_partition_d5_min_mnr_negative_minimum_is_selected() {
        // A negative MNR (signal needs more bits than current quantization
        // provides) is a valid — and the most urgent — selection.
        let mut mnr = coder_partition_d5_mnr_row_order(|_| 10.0, |_| 0.0, |_| 0.0);
        // All rows currently mnr_db = 10; depress partition n = 20.
        mnr[19].mnr_db = -5.0;
        let sel = coder_partition_d5_min_mnr(&mnr);
        assert_eq!(sel.partition_n, 20);
        assert_eq!(sel.mnr_db, -5.0);
    }

    #[test]
    fn coder_partition_d5_min_mnr_nan_rows_are_skipped() {
        // A NaN MNR never compares `<` the running minimum, so it is
        // skipped; the finite minimum still wins.
        let mut mnr = coder_partition_d5_mnr_row_order(|_| 10.0, |_| 0.0, |_| 0.0);
        mnr[5].mnr_db = f64::NAN;
        mnr[18].mnr_db = 2.0;
        let sel = coder_partition_d5_min_mnr(&mnr);
        assert_eq!(sel.partition_n, 19);
        assert_eq!(sel.mnr_db, 2.0);
    }

    #[test]
    fn coder_partition_d5_min_mnr_is_idempotent() {
        let snr = |n: u16| -> f64 { (f64::from(n) * 1.7).sin() * 10.0 + 30.0 };
        let lsb = |n: u16| -> f64 { f64::from(n) * 0.5 + 30.0 };
        let ltg = |omega: u16| -> f64 { (f64::from(omega) * 0.002).cos() * 8.0 };
        let mnr = coder_partition_d5_mnr_row_order(snr, lsb, ltg);
        assert_eq!(
            coder_partition_d5_min_mnr(&mnr),
            coder_partition_d5_min_mnr(&mnr)
        );
    }

    #[test]
    fn coder_partition_d5_min_mnr_matches_brute_force_argmin() {
        // Cross-check the row-order scan against an independent argmin over
        // the same 32-row vector for an arbitrary non-degenerate input.
        let snr = |n: u16| -> f64 { (f64::from(n) * 2.3).cos() * 13.0 + 25.0 };
        let lsb = |n: u16| -> f64 { f64::from(n) * 0.37 + 28.0 };
        let ltg = |omega: u16| -> f64 { (f64::from(omega) * 0.0017).sin() * 6.0 };
        let mnr = coder_partition_d5_mnr_row_order(snr, lsb, ltg);
        let expected = mnr
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.mnr_db.partial_cmp(&b.1.mnr_db).unwrap())
            .map(|(i, _)| i as u16 + 1)
            .unwrap();
        assert_eq!(coder_partition_d5_min_mnr(&mnr).partition_n, expected);
    }

    // ---- Phase 2 step 74: §C.1.5.2.7 next-higher-entry promotion -----

    #[test]
    fn promote_entry_advances_to_next_higher() {
        // A subband mid-column advances by exactly one entry index.
        let got = bit_allocation_promote_entry(3, 2, 16);
        assert_eq!(
            got,
            BitAllocPromotion {
                subband: 3,
                entry: 3,
                advanced: true,
            }
        );
    }

    #[test]
    fn promote_entry_from_bottom_entry() {
        // Entry 0 (coarsest) advances to entry 1 when the column has room.
        let got = bit_allocation_promote_entry(0, 0, 4);
        assert_eq!(got.entry, 1);
        assert!(got.advanced);
    }

    #[test]
    fn promote_entry_saturates_at_top_entry() {
        // Already at the highest entry (entry_count - 1): no advance, entry
        // held unchanged, advanced = false.
        let got = bit_allocation_promote_entry(7, 15, 16);
        assert_eq!(
            got,
            BitAllocPromotion {
                subband: 7,
                entry: 15,
                advanced: false,
            }
        );
    }

    #[test]
    fn promote_entry_single_entry_column_never_advances() {
        // A column with exactly one entry can never be promoted.
        let got = bit_allocation_promote_entry(11, 0, 1);
        assert_eq!(got.entry, 0);
        assert!(!got.advanced);
    }

    #[test]
    fn promote_entry_zero_entry_count_never_advances() {
        // A subband with no possible quantization (entry_count = 0) holds
        // its prior entry and reports no advance.
        let got = bit_allocation_promote_entry(20, 0, 0);
        assert_eq!(got.entry, 0);
        assert!(!got.advanced);
    }

    #[test]
    fn promote_entry_echoes_subband_index() {
        // The targeted subband index is carried through verbatim in both
        // the advancing and the saturated cases.
        assert_eq!(bit_allocation_promote_entry(29, 0, 8).subband, 29);
        assert_eq!(bit_allocation_promote_entry(29, 7, 8).subband, 29);
    }

    #[test]
    fn promote_entry_last_room_advances_then_next_saturates() {
        // From the penultimate entry the advance lands on the top entry and
        // succeeds; a second call from there saturates.
        let first = bit_allocation_promote_entry(5, 14, 16);
        assert_eq!(first.entry, 15);
        assert!(first.advanced);
        let second = bit_allocation_promote_entry(5, first.entry, 16);
        assert_eq!(second.entry, 15);
        assert!(!second.advanced);
    }

    #[test]
    fn promote_entry_repeated_walk_climbs_one_per_call() {
        // Iterating the action over a fresh column climbs entry indices one
        // at a time up to (entry_count - 1), then stops.
        let entry_count = 6u16;
        let mut entry = 0u16;
        let mut steps = 0u16;
        loop {
            let p = bit_allocation_promote_entry(2, entry, entry_count);
            if !p.advanced {
                break;
            }
            assert_eq!(p.entry, entry + 1);
            entry = p.entry;
            steps += 1;
        }
        assert_eq!(entry, entry_count - 1);
        assert_eq!(steps, entry_count - 1);
    }

    #[test]
    fn promote_entry_is_deterministic() {
        // A pure function: same arguments yield identical results.
        assert_eq!(
            bit_allocation_promote_entry(4, 9, 27),
            bit_allocation_promote_entry(4, 9, 27)
        );
    }

    // ---- Phase 2 step 75: §C.1.5.2.7 "new MNR is calculated" ----------

    #[test]
    fn recompute_mnr_is_snr_for_advanced_entry_minus_smr() {
        // After step 74 advances entry 2 -> 3, the new MNR uses the
        // Table C.5 SNR for entry 3 minus the unchanged SMR.
        let promotion = bit_allocation_promote_entry(3, 2, 16);
        assert!(promotion.advanced);
        assert_eq!(promotion.entry, 3);
        // SNR table modelled as a monotone column: SNR(entry) grows.
        let snr = |entry: u16| 6.0 * f64::from(entry);
        let got = bit_allocation_recompute_mnr(promotion, 10.0, snr);
        assert_eq!(
            got,
            CoderPartitionD5RecomputedMnr {
                subband: 3,
                entry: 3,
                mnr_db: 6.0 * 3.0 - 10.0,
                smr_db: 10.0,
            }
        );
    }

    #[test]
    fn recompute_mnr_carries_smr_through_verbatim() {
        // The SMR field echoes the supplied psychoacoustic-model output
        // unchanged regardless of the SNR column.
        let promotion = bit_allocation_promote_entry(7, 0, 4);
        let got = bit_allocation_recompute_mnr(promotion, -3.5, |_| 42.0);
        assert_eq!(got.smr_db, -3.5);
        assert_eq!(got.mnr_db, 42.0 - (-3.5));
    }

    #[test]
    fn recompute_mnr_echoes_subband_and_post_promotion_entry() {
        // subband / entry fields mirror the step-74 promotion exactly.
        let promotion = bit_allocation_promote_entry(29, 5, 8);
        let got = bit_allocation_recompute_mnr(promotion, 0.0, f64::from);
        assert_eq!(got.subband, 29);
        assert_eq!(got.entry, promotion.entry);
    }

    #[test]
    fn recompute_mnr_raises_mnr_for_monotone_snr_column() {
        // The finer quantization (higher entry) yields a strictly larger
        // MNR than the pre-promotion entry for a monotone SNR column —
        // exactly why the loop drops this subband from "greatest benefit".
        let smr = 20.0;
        let snr = |entry: u16| 4.0 * f64::from(entry);
        let prev_entry = 2u16;
        let prev_mnr = snr(prev_entry) - smr;
        let promotion = bit_allocation_promote_entry(1, prev_entry, 10);
        let got = bit_allocation_recompute_mnr(promotion, smr, snr);
        assert!(got.mnr_db > prev_mnr);
    }

    #[test]
    fn recompute_mnr_on_saturated_promotion_holds_entry() {
        // A step-74 promotion that could not advance (top entry) recomputes
        // the MNR at the held entry: an idempotent re-evaluation since the
        // SNR for the unchanged entry is identical.
        let promotion = bit_allocation_promote_entry(4, 15, 16);
        assert!(!promotion.advanced);
        let snr = |entry: u16| 5.0 * f64::from(entry);
        let got = bit_allocation_recompute_mnr(promotion, 1.0, snr);
        assert_eq!(got.entry, 15);
        assert_eq!(got.mnr_db, 5.0 * 15.0 - 1.0);
    }

    #[test]
    fn recompute_mnr_invokes_snr_callback_for_the_post_entry_only() {
        // The SNR callback is consulted exactly once, for the post-promotion
        // entry index.
        use core::cell::Cell;
        let calls = Cell::new(0u32);
        let seen = Cell::new(u16::MAX);
        let promotion = bit_allocation_promote_entry(2, 3, 9);
        let _ = bit_allocation_recompute_mnr(promotion, 0.0, |e| {
            calls.set(calls.get() + 1);
            seen.set(e);
            0.0
        });
        assert_eq!(calls.get(), 1);
        assert_eq!(seen.get(), promotion.entry);
    }

    #[test]
    fn recompute_mnr_matches_step72_definition_cell_wise() {
        // The recomputed MNR uses the identical MNR = SNR - SMR definition
        // as the step-72 initialisation, just with the advanced entry's SNR.
        let smr = 33.3;
        let snr = |entry: u16| 100.0 - 7.0 * f64::from(entry);
        let promotion = bit_allocation_promote_entry(6, 1, 12);
        let got = bit_allocation_recompute_mnr(promotion, smr, snr);
        assert!((got.mnr_db - (snr(promotion.entry) - smr)).abs() < 1e-12);
    }

    #[test]
    fn recompute_mnr_is_deterministic() {
        // A pure function of its arguments and the (pure) SNR callback.
        let promotion = bit_allocation_promote_entry(8, 4, 20);
        let snr = |e: u16| 2.5 * f64::from(e) + 1.0;
        assert_eq!(
            bit_allocation_recompute_mnr(promotion, 9.0, snr),
            bit_allocation_recompute_mnr(promotion, 9.0, snr)
        );
    }

    #[test]
    fn recompute_mnr_negative_smr_increases_mnr() {
        // A negative SMR (mask below 0 dB reference) lifts the MNR by its
        // magnitude, consistent with MNR = SNR - SMR.
        let promotion = bit_allocation_promote_entry(0, 0, 4);
        let got = bit_allocation_recompute_mnr(promotion, -12.0, |_| 8.0);
        assert_eq!(got.mnr_db, 8.0 + 12.0);
    }

    // --- Phase 2 step 76 (r275): §C.1.5.2.7 step-4 budget update + ------
    // --- iterate/terminate test --------------------------------------

    fn overhead_fixture() -> BitAllocOverhead {
        // A representative Layer II overhead: 32-bit header, no CRC, a
        // 64-bit allocation field, no ancillary data, 1000-bit frame.
        BitAllocOverhead {
            cb: 1000,
            bhdr: 32,
            bcrc: 0,
            bbal: 64,
            banc: 0,
        }
    }

    #[test]
    fn budget_update_adds_sample_bits_to_bspl() {
        // A non-first-time promotion only grows bspl; bsel/bscf are
        // carried through unchanged.
        let prev = BitAllocBudget {
            bspl: 100,
            bsel: 8,
            bscf: 36,
            first_time: false,
            adb: 0,
        };
        let got = bit_allocation_budget_update(prev, 40, false, 999, 999, overhead_fixture());
        assert_eq!(got.bspl, 140);
        assert_eq!(got.bsel, 8); // unchanged: not a first-time allocation
        assert_eq!(got.bscf, 36); // unchanged
        assert!(!got.first_time);
    }

    #[test]
    fn budget_update_first_time_grows_bsel_and_bscf() {
        // A first-time allocation grows all three accumulators.
        let prev = BitAllocBudget {
            bspl: 0,
            bsel: 0,
            bscf: 0,
            first_time: false,
            adb: 0,
        };
        let got = bit_allocation_budget_update(prev, 30, true, 2, 18, overhead_fixture());
        assert_eq!(got.bspl, 30);
        assert_eq!(got.bsel, 2);
        assert_eq!(got.bscf, 18);
        assert!(got.first_time);
    }

    #[test]
    fn budget_update_recomputes_adb_per_spec_formula() {
        // adb = cb - (bhdr + bcrc + bbal + bsel + bscf + bspl + banc).
        let prev = BitAllocBudget {
            bspl: 50,
            bsel: 4,
            bscf: 20,
            first_time: false,
            adb: 0,
        };
        let oh = overhead_fixture();
        let got = bit_allocation_budget_update(prev, 10, false, 0, 0, oh);
        // bspl=60, bsel=4, bscf=20; used = 32+0+64+4+20+60+0 = 180.
        let expect = oh.cb - (oh.bhdr + oh.bcrc + oh.bbal + 4 + 20 + 60 + oh.banc);
        assert_eq!(got.adb, expect);
        assert_eq!(got.adb, 1000 - 180);
    }

    #[test]
    fn budget_update_first_time_adb_includes_new_sel_scf() {
        // On a first-time promotion the recomputed adb reflects the freshly
        // added bsel and bscf, not just bspl.
        let prev = BitAllocBudget {
            bspl: 0,
            bsel: 0,
            bscf: 0,
            first_time: false,
            adb: 0,
        };
        let oh = overhead_fixture();
        let got = bit_allocation_budget_update(prev, 30, true, 2, 18, oh);
        // used = 32 + 0 + 64 + 2 + 18 + 30 + 0 = 146.
        assert_eq!(got.adb, 1000 - 146);
    }

    #[test]
    fn budget_update_adb_saturates_at_zero_on_overcommit() {
        // When the running totals exceed cb, adb floors at zero rather than
        // wrapping (u32 saturation).
        let prev = BitAllocBudget {
            bspl: 900,
            bsel: 8,
            bscf: 36,
            first_time: false,
            adb: 0,
        };
        let got = bit_allocation_budget_update(prev, 200, false, 0, 0, overhead_fixture());
        assert_eq!(got.adb, 0);
    }

    #[test]
    fn budget_update_zero_extra_bits_is_idempotent_on_accumulators() {
        // A no-op promotion (saturated step-74) adds zero bits; the
        // accumulators are unchanged and adb is recomputed from them.
        let prev = BitAllocBudget {
            bspl: 120,
            bsel: 8,
            bscf: 36,
            first_time: false,
            adb: 0,
        };
        let got = bit_allocation_budget_update(prev, 0, false, 5, 5, overhead_fixture());
        assert_eq!((got.bspl, got.bsel, got.bscf), (120, 8, 36));
    }

    #[test]
    fn budget_update_is_deterministic() {
        let prev = BitAllocBudget {
            bspl: 10,
            bsel: 2,
            bscf: 6,
            first_time: false,
            adb: 0,
        };
        let oh = overhead_fixture();
        assert_eq!(
            bit_allocation_budget_update(prev, 7, true, 2, 12, oh),
            bit_allocation_budget_update(prev, 7, true, 2, 12, oh)
        );
    }

    #[test]
    fn budget_update_threads_across_iterations() {
        // Chain three iterations: the budget accumulates and adb shrinks
        // monotonically as bits are spent.
        let oh = overhead_fixture();
        let mut b = BitAllocBudget {
            bspl: 0,
            bsel: 0,
            bscf: 0,
            first_time: false,
            adb: oh.cb,
        };
        let mut prev_adb = oh.cb;
        for _ in 0..3 {
            b = bit_allocation_budget_update(b, 50, true, 2, 18, oh);
            assert!(b.adb < prev_adb, "adb must shrink as bits are spent");
            prev_adb = b.adb;
        }
        // After three first-time allocations: bspl=150, bsel=6, bscf=54.
        assert_eq!((b.bspl, b.bsel, b.bscf), (150, 6, 54));
    }

    #[test]
    fn should_iterate_true_while_adb_covers_largest_increase() {
        // adb >= max increase ⇒ loop continues.
        assert!(bit_allocation_should_iterate(100, 40));
        assert!(bit_allocation_should_iterate(40, 40)); // not-less-than ⇒ boundary continues
    }

    #[test]
    fn should_iterate_false_when_adb_below_largest_increase() {
        // adb < max increase ⇒ loop terminates.
        assert!(!bit_allocation_should_iterate(39, 40));
        assert!(!bit_allocation_should_iterate(0, 1));
    }

    #[test]
    fn should_iterate_zero_increase_always_continues() {
        // A zero worst-case increase (no promotable subband) trivially
        // satisfies the >= test; the terminal condition is detected from
        // step 74's `advanced` flag instead.
        assert!(bit_allocation_should_iterate(0, 0));
        assert!(bit_allocation_should_iterate(500, 0));
    }

    #[test]
    fn budget_then_iterate_terminates_when_budget_exhausted() {
        // End-to-end: a step-4 update that drives adb below the next
        // possible increase makes should_iterate report termination.
        let oh = overhead_fixture();
        let prev = BitAllocBudget {
            bspl: 800,
            bsel: 8,
            bscf: 36,
            first_time: false,
            adb: 0,
        };
        // used so far before this add = 32+64+8+36+800 = 940; adb = 60.
        let got = bit_allocation_budget_update(prev, 30, false, 0, 0, oh);
        // bspl=830 ⇒ used=970 ⇒ adb=30.
        assert_eq!(got.adb, 30);
        // Next possible increase needs 50 bits ⇒ cannot continue.
        assert!(!bit_allocation_should_iterate(got.adb, 50));
        // A cheaper next increase (20 bits) would still fit.
        assert!(bit_allocation_should_iterate(got.adb, 20));
    }

    // ----- Phase 2 step 77: §D.1 Step 1 FFT analysis -----

    /// Deterministic pseudo-random sample stream for FFT cross-checks
    /// (simple LCG; no external randomness).
    fn lcg_samples(len: usize, mut state: u64) -> Vec<f64> {
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                // Map the top 32 bits to (-1, 1).
                ((state >> 32) as f64 / 2147483648.0) - 1.0
            })
            .collect()
    }

    /// Direct (naive) evaluation of the Step 1 power-density formula —
    /// independent of the radix-2 FFT under test.
    fn direct_power_density(s: &[f64]) -> Vec<f64> {
        let n = s.len();
        let inv_n = 1.0 / n as f64;
        (0..=n / 2)
            .map(|k| {
                let mut re = 0.0_f64;
                let mut im = 0.0_f64;
                for (l, &sample) in s.iter().enumerate() {
                    let w = model1_hann_window(l, n).unwrap() * sample;
                    let ang = -2.0 * core::f64::consts::PI * (k as f64) * (l as f64) / (n as f64);
                    re += w * ang.cos();
                    im += w * ang.sin();
                }
                let (r, i) = (re * inv_n, im * inv_n);
                10.0 * (r * r + i * i).log10()
            })
            .collect()
    }

    #[test]
    fn hann_window_domain_and_endpoints() {
        // Out-of-domain rejections: n = 0 and i >= n.
        assert_eq!(model1_hann_window(0, 0), None);
        assert_eq!(model1_hann_window(1024, 1024), None);
        assert_eq!(model1_hann_window(usize::MAX, 1024), None);
        // h(0) = sqrt(8/3)·0,5·(1 − cos 0) = 0.
        assert_eq!(model1_hann_window(0, 1024), Some(0.0));
        // Midpoint h(N/2): cos(π) = −1 ⇒ h = sqrt(8/3).
        let mid = model1_hann_window(512, 1024).unwrap();
        assert!((mid - (8.0_f64 / 3.0).sqrt()).abs() < 1e-15);
    }

    #[test]
    fn hann_window_symmetry_and_unit_power() {
        let n = MODEL1_FFT_LEN_LAYER2;
        // h(i) = h(N − i) for 1 <= i <= N−1 (cos is even around 0/2π).
        for i in 1..n {
            let a = model1_hann_window(i, n).unwrap();
            let b = model1_hann_window(n - i, n).unwrap();
            assert!((a - b).abs() < 1e-12, "asymmetry at i={i}");
        }
        // Unit power: Σ h(i)² = N (the sqrt(8/3) prefactor's purpose).
        let power: f64 = (0..n)
            .map(|i| {
                let h = model1_hann_window(i, n).unwrap();
                h * h
            })
            .sum();
        assert!((power - n as f64).abs() < 1e-6, "Σh² = {power}");
    }

    #[test]
    fn power_density_rejects_non_spec_lengths() {
        assert!(model1_power_density_spectrum(&[]).is_none());
        assert!(model1_power_density_spectrum(&[0.0; 100]).is_none());
        assert!(model1_power_density_spectrum(&[0.0; 576]).is_none());
        assert!(model1_power_density_spectrum(&[0.0; 2048]).is_none());
    }

    #[test]
    fn power_density_output_is_inclusive_half_spectrum() {
        // k = 0...N/2 ⇒ N/2 + 1 lines: 257 (Layer I) / 513 (Layer II).
        let x1 = model1_power_density_spectrum(&vec![0.5; MODEL1_FFT_LEN_LAYER1]).unwrap();
        assert_eq!(x1.len(), MODEL1_FFT_LEN_LAYER1 / 2 + 1);
        let x2 = model1_power_density_spectrum(&vec![0.5; MODEL1_FFT_LEN_LAYER2]).unwrap();
        assert_eq!(x2.len(), MODEL1_FFT_LEN_LAYER2 / 2 + 1);
    }

    #[test]
    fn power_density_dc_anchor() {
        // Constant signal s(l) = 1: only the DC coefficient survives.
        // (1/N)·Σ h(l) = sqrt(8/3)·0,5 (the cos term sums to zero over a
        // full period) ⇒ power = (8/3)·0,25 = 2/3 ⇒ 10·log10(2/3) dB.
        let n = MODEL1_FFT_LEN_LAYER2;
        let x = model1_power_density_spectrum(&vec![1.0; n]).unwrap();
        let expect_dc = 10.0 * (2.0_f64 / 3.0).log10();
        assert!((x[0] - expect_dc).abs() < 1e-9, "X(0) = {}", x[0]);
        // Every non-adjacent line is numerically negligible.
        for (k, &line) in x.iter().enumerate().skip(2) {
            assert!(line < -250.0, "leak at k={k}: {line} dB");
        }
    }

    #[test]
    fn power_density_pure_tone_anchors() {
        // Unit-amplitude sine at exact bin k0: the Hann-windowed DFT
        // concentrates power on k0 and k0±1.
        //   |X(k0)|   = sqrt(8/3)/4  ⇒ power = 1/6
        //   |X(k0±1)| = sqrt(8/3)/8  ⇒ power = 1/24  (−10·log10 4 dB)
        let n = MODEL1_FFT_LEN_LAYER2;
        let k0 = 100usize;
        let s: Vec<f64> = (0..n)
            .map(|l| (2.0 * core::f64::consts::PI * (k0 as f64) * (l as f64) / (n as f64)).sin())
            .collect();
        let x = model1_power_density_spectrum(&s).unwrap();
        let peak_expect = 10.0 * (1.0_f64 / 6.0).log10();
        let side_expect = 10.0 * (1.0_f64 / 24.0).log10();
        assert!((x[k0] - peak_expect).abs() < 1e-9, "X(k0) = {}", x[k0]);
        assert!((x[k0 - 1] - side_expect).abs() < 1e-9);
        assert!((x[k0 + 1] - side_expect).abs() < 1e-9);
        // Hann leakage stops after ±1: everything else is negligible.
        for (k, &line) in x.iter().enumerate() {
            if k + 2 <= k0 || k >= k0 + 2 {
                assert!(line < -250.0, "leak at k={k}: {line} dB");
            }
        }
        // The ±1 lines sit exactly 10·log10(4) ≈ 6,02 dB below the peak.
        assert!(((x[k0] - x[k0 + 1]) - 10.0 * 4.0_f64.log10()).abs() < 1e-9);
    }

    #[test]
    fn power_density_matches_direct_dft_evaluation() {
        // Cross-check the radix-2 FFT against an independent direct
        // evaluation of the verbatim Step 1 formula on a deterministic
        // broadband block (both transform lengths).
        for &n in &[MODEL1_FFT_LEN_LAYER1, MODEL1_FFT_LEN_LAYER2] {
            let s = lcg_samples(n, 0x0D15_EA5E_u64);
            let fast = model1_power_density_spectrum(&s).unwrap();
            let slow = direct_power_density(&s);
            assert_eq!(fast.len(), slow.len());
            for (k, (a, b)) in fast.iter().zip(slow.iter()).enumerate() {
                assert!((a - b).abs() < 1e-8, "n={n} k={k}: fft {a} vs dft {b}");
            }
        }
    }

    #[test]
    fn power_density_silent_block_is_negative_infinity() {
        let x = model1_power_density_spectrum(&vec![0.0; MODEL1_FFT_LEN_LAYER2]).unwrap();
        assert!(x.iter().all(|&v| v == f64::NEG_INFINITY));
    }

    #[test]
    fn normalize_pins_maximum_to_96db_and_preserves_deltas() {
        let mut x = vec![-30.0, -7.5, 12.25, 3.0, f64::NEG_INFINITY];
        let offset = model1_normalize_to_96db_spl(&mut x).unwrap();
        assert_eq!(offset, MODEL1_SPL_REFERENCE_DB - 12.25);
        let max = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert_eq!(max, MODEL1_SPL_REFERENCE_DB);
        // Relative differences preserved (single shared addend).
        assert_eq!(x[2] - x[0], 12.25 - (-30.0));
        assert_eq!(x[2] - x[1], 12.25 - (-7.5));
        // −∞ stays −∞ (an affine shift cannot lift true silence).
        assert_eq!(x[4], f64::NEG_INFINITY);
    }

    #[test]
    fn normalize_already_at_reference_is_identity() {
        let mut x = vec![96.0, 50.0, -10.0];
        let offset = model1_normalize_to_96db_spl(&mut x).unwrap();
        assert_eq!(offset, 0.0);
        assert_eq!(x, vec![96.0, 50.0, -10.0]);
    }

    #[test]
    fn normalize_refuses_spectra_without_finite_maximum() {
        let mut empty: Vec<f64> = vec![];
        assert_eq!(model1_normalize_to_96db_spl(&mut empty), None);
        let mut silent = vec![f64::NEG_INFINITY; 5];
        assert_eq!(model1_normalize_to_96db_spl(&mut silent), None);
        // Slice untouched on refusal.
        assert!(silent.iter().all(|&v| v == f64::NEG_INFINITY));
    }

    #[test]
    fn step1_pipeline_tone_normalizes_to_reference() {
        // End-to-end Step 1: window+FFT a pure tone, then normalize —
        // the tonal peak line lands exactly at 96 dB SPL.
        let n = MODEL1_FFT_LEN_LAYER2;
        let k0 = 64usize;
        let s: Vec<f64> = (0..n)
            .map(|l| {
                0.25 * (2.0 * core::f64::consts::PI * (k0 as f64) * (l as f64) / (n as f64)).sin()
            })
            .collect();
        let mut x = model1_power_density_spectrum(&s).unwrap();
        let offset = model1_normalize_to_96db_spl(&mut x).unwrap();
        assert!(offset.is_finite());
        assert_eq!(x[k0], MODEL1_SPL_REFERENCE_DB);
        // The ±1 Hann sidelines keep their −6,02 dB relation post-shift.
        assert!(((x[k0] - x[k0 + 1]) - 10.0 * 4.0_f64.log10()).abs() < 1e-9);
    }

    // ----- Phase 2 step 78: §D.1 Step 2 sound pressure level -----

    #[test]
    fn step2_scf_term_anchors() {
        // scf_max = 1: 20·log10(32 768) − 10 = 20·15·log10(2) − 10.
        let unit = model1_step2_scf_term_db(1.0);
        let expect = 20.0 * 32768.0_f64.log10() - 10.0;
        assert!((unit - expect).abs() < 1e-12);
        assert!((unit - 80.30899869919435).abs() < 1e-9);
        // scf_max = 1/32 768 makes the log term vanish: exactly −10 dB.
        let tiny = model1_step2_scf_term_db(1.0 / 32768.0);
        assert!((tiny - (-MODEL1_STEP2_PEAK_RMS_CORRECTION_DB)).abs() < 1e-12);
        // Doubling scf_max adds 20·log10(2) ≈ 6,0206 dB.
        let doubled = model1_step2_scf_term_db(2.0);
        assert!(((doubled - unit) - 20.0 * 2.0_f64.log10()).abs() < 1e-12);
    }

    #[test]
    fn step2_lsb_is_outer_maximum() {
        let scf_term = model1_step2_scf_term_db(0.5);
        // Spectral argument dominates…
        assert_eq!(model1_step2_lsb_db(scf_term + 5.0, 0.5), scf_term + 5.0);
        // …or the scalefactor term dominates…
        assert_eq!(model1_step2_lsb_db(scf_term - 5.0, 0.5), scf_term);
        // …and a silent subband falls back to the scf term entirely.
        assert_eq!(model1_step2_lsb_db(f64::NEG_INFINITY, 0.5), scf_term);
    }

    #[test]
    fn step2_xspl_power_sum_anchors() {
        // A single line is returned unchanged.
        assert!((model1_step2_xspl_db(&[-7.5]) - (-7.5)).abs() < 1e-12);
        // Two equal-power lines: +10·log10(2) ≈ +3,0103 dB.
        let two = model1_step2_xspl_db(&[20.0, 20.0]);
        assert!(((two - 20.0) - 10.0 * 2.0_f64.log10()).abs() < 1e-12);
        // −∞ (silent) lines contribute zero linear power.
        let with_silence = model1_step2_xspl_db(&[20.0, f64::NEG_INFINITY]);
        assert!((with_silence - 20.0).abs() < 1e-12);
        // Empty / all-silent selections collapse to −∞.
        assert_eq!(model1_step2_xspl_db(&[]), f64::NEG_INFINITY);
        assert_eq!(
            model1_step2_xspl_db(&[f64::NEG_INFINITY; 4]),
            f64::NEG_INFINITY
        );
    }

    #[test]
    fn step2_subband_accessors_reject_bad_inputs() {
        let x = vec![0.0; MODEL1_FFT_LEN_LAYER2 / 2 + 1];
        // Partition index outside 1..=32.
        assert!(model1_step2_subband_max_line_db(&x, 0).is_none());
        assert!(model1_step2_subband_max_line_db(&x, 33).is_none());
        assert!(model1_step2_subband_xspl_db(&x, 0).is_none());
        assert!(model1_step2_subband_xspl_db(&x, 33).is_none());
        // Not the 513-line 1 024-sample half-spectrum.
        let short = vec![0.0; MODEL1_FFT_LEN_LAYER1 / 2 + 1];
        assert!(model1_step2_subband_max_line_db(&short, 1).is_none());
        assert!(model1_step2_subband_xspl_db(&short, 1).is_none());
    }

    #[test]
    fn step2_subband_max_finds_planted_line_via_d5_span() {
        // Plant a single loud line inside every partition's exclusive
        // interior and confirm the D.5-driven max recovers it.
        for n in 1..=32u16 {
            let (low, high) = coder_partition_d5_line_range(n).unwrap();
            let mut x = vec![-300.0; MODEL1_FFT_LEN_LAYER2 / 2 + 1];
            // Interior line (not a shared boundary cell).
            let k = (low as usize - 1) + (high as usize - low as usize) / 2;
            x[k] = -3.25;
            assert_eq!(model1_step2_subband_max_line_db(&x, n), Some(-3.25));
        }
    }

    #[test]
    fn step2_subband_spans_share_boundary_cells() {
        // The D.5 dual-role ωlow_{n+1}/ωhigh_n cell belongs to both
        // adjacent subbands: a loud boundary line is seen by both maxima.
        let n = 7u16;
        let (_, high) = coder_partition_d5_line_range(n).unwrap();
        let mut x = vec![-300.0; MODEL1_FFT_LEN_LAYER2 / 2 + 1];
        x[high as usize - 1] = 1.5;
        assert_eq!(model1_step2_subband_max_line_db(&x, n), Some(1.5));
        assert_eq!(model1_step2_subband_max_line_db(&x, n + 1), Some(1.5));
    }

    #[test]
    fn step2_xspl_dominates_max_line_on_broadband_spectrum() {
        // Xspl(n) sums power over the whole span, so it is never below
        // the span's single maximum line — checked on a real step-77
        // spectrum across every partition.
        let s = lcg_samples(MODEL1_FFT_LEN_LAYER2, 0xFEED_F00D_u64);
        let x = model1_power_density_spectrum(&s).unwrap();
        for n in 1..=32u16 {
            let max_line = model1_step2_subband_max_line_db(&x, n).unwrap();
            let xspl = model1_step2_subband_xspl_db(&x, n).unwrap();
            assert!(
                xspl >= max_line - 1e-12,
                "n={n}: Xspl {xspl} < max line {max_line}"
            );
        }
    }

    #[test]
    fn step2_lsb_end_to_end_from_step1_spectrum() {
        // Full Step 1 → Step 2 chain: tone at bin 100 (inside partition
        // 7 per the D.5 16-line stride), normalized to 96 dB, then Lsb.
        let n_fft = MODEL1_FFT_LEN_LAYER2;
        let k0 = 100usize;
        let s: Vec<f64> = (0..n_fft)
            .map(|l| {
                (2.0 * core::f64::consts::PI * (k0 as f64) * (l as f64) / (n_fft as f64)).sin()
            })
            .collect();
        let mut x = model1_power_density_spectrum(&s).unwrap();
        model1_normalize_to_96db_spl(&mut x).unwrap();
        // Line k = 100 is ω = 101 ∈ [97, 113] = partition 7's span.
        let part = first_partition_containing_line(101).unwrap();
        assert_eq!(part, 7);
        let max_line = model1_step2_subband_max_line_db(&x, part).unwrap();
        assert_eq!(max_line, MODEL1_SPL_REFERENCE_DB);
        // A small scalefactor leaves the spectral argument dominant…
        let lsb = model1_step2_lsb_db(max_line, 1.0 / 32768.0);
        assert_eq!(lsb, MODEL1_SPL_REFERENCE_DB);
        // …a full-scale scalefactor pushes Lsb to the scf term instead
        // (96 < 20·log10(32 768·10) − 10 would be false; use a huge one).
        let huge_scf_term = model1_step2_scf_term_db(100.0);
        assert!(huge_scf_term > MODEL1_SPL_REFERENCE_DB);
        assert_eq!(model1_step2_lsb_db(max_line, 100.0), huge_scf_term);
    }

    // ---- Phase 2 step 79 / r277 — §D.1 Step 4 tonal / non-tonal
    //      classification.

    /// Flat spectrum at `level` dB with the Layer II half-spectrum
    /// length (513 lines).
    fn flat_spectrum_l2(level: f64) -> Vec<f64> {
        vec![level; MODEL1_FFT_LEN_LAYER2 / 2 + 1]
    }

    #[test]
    fn step4_offsets_layer1_ranges() {
        use crate::frame::Layer;
        // Below the first examined range (k <= 2) — no j set.
        assert!(model1_step4_tonal_check_offsets(Layer::LayerI, 0).is_none());
        assert!(model1_step4_tonal_check_offsets(Layer::LayerI, 2).is_none());
        // 2 < k < 63: j = -2, +2.
        assert_eq!(
            model1_step4_tonal_check_offsets(Layer::LayerI, 3),
            Some(&[-2, 2][..])
        );
        assert_eq!(
            model1_step4_tonal_check_offsets(Layer::LayerI, 62),
            Some(&[-2, 2][..])
        );
        // 63 <= k < 127: j = -3, -2, +2, +3.
        assert_eq!(
            model1_step4_tonal_check_offsets(Layer::LayerI, 63),
            Some(&[-3, -2, 2, 3][..])
        );
        assert_eq!(
            model1_step4_tonal_check_offsets(Layer::LayerI, 126),
            Some(&[-3, -2, 2, 3][..])
        );
        // 127 <= k <= 250: j = -6..-2, +2..+6 (10 offsets, no ±1, no 0).
        let far = model1_step4_tonal_check_offsets(Layer::LayerI, 127).unwrap();
        assert_eq!(far.len(), 10);
        assert!(!far.contains(&-1) && !far.contains(&0) && !far.contains(&1));
        assert_eq!(far.first(), Some(&-6));
        assert_eq!(far.last(), Some(&6));
        assert_eq!(
            model1_step4_tonal_check_offsets(Layer::LayerI, 250),
            Some(far)
        );
        // Layer I's table ends at k = 250 inclusive.
        assert!(model1_step4_tonal_check_offsets(Layer::LayerI, 251).is_none());
    }

    #[test]
    fn step4_offsets_layer2_ranges() {
        use crate::frame::Layer;
        // The first three ranges match Layer I…
        assert_eq!(
            model1_step4_tonal_check_offsets(Layer::LayerII, 3),
            model1_step4_tonal_check_offsets(Layer::LayerI, 3)
        );
        assert_eq!(
            model1_step4_tonal_check_offsets(Layer::LayerII, 126),
            model1_step4_tonal_check_offsets(Layer::LayerI, 126)
        );
        // …but the third range is right-open at 255 for Layer II.
        assert_eq!(
            model1_step4_tonal_check_offsets(Layer::LayerII, 254).map(<[i32]>::len),
            Some(10)
        );
        // 255 <= k <= 500: j = -12..-2, +2..+12 (22 offsets).
        let top = model1_step4_tonal_check_offsets(Layer::LayerII, 255).unwrap();
        assert_eq!(top.len(), 22);
        assert_eq!(top.first(), Some(&-12));
        assert_eq!(top.last(), Some(&12));
        assert!(!top.contains(&-1) && !top.contains(&0) && !top.contains(&1));
        assert_eq!(
            model1_step4_tonal_check_offsets(Layer::LayerII, 500),
            Some(top)
        );
        assert!(model1_step4_tonal_check_offsets(Layer::LayerII, 501).is_none());
        // Annex D defines the j table for Layers I/II only.
        assert!(model1_step4_tonal_check_offsets(Layer::LayerIII, 100).is_none());
    }

    #[test]
    fn step4_local_maximum_strict_low_nonstrict_high() {
        // X(k) > X(k-1) is strict; X(k) >= X(k+1) is non-strict: a
        // plateau's *first* line is the labelled maximum.
        let x = [0.0, 2.0, 2.0, 1.0, 1.0];
        assert_eq!(model1_step4_is_local_maximum(&x, 1), Some(true)); // 2 > 0, 2 >= 2
        assert_eq!(model1_step4_is_local_maximum(&x, 2), Some(false)); // 2 > 2 fails
        assert_eq!(model1_step4_is_local_maximum(&x, 3), Some(false)); // 1 > 2 fails
                                                                       // Edges have no neighbour — the spec formula is undefined.
        assert!(model1_step4_is_local_maximum(&x, 0).is_none());
        assert!(model1_step4_is_local_maximum(&x, 4).is_none());
    }

    #[test]
    fn step4_is_tonal_seven_db_margin_inclusive() {
        use crate::frame::Layer;
        // k = 20 sits in the first range (j = ±2). A 7.0 dB margin
        // passes (>= is inclusive); 6.9 dB fails.
        let mut x = flat_spectrum_l2(0.0);
        x[20] = 7.0;
        assert_eq!(model1_step4_is_tonal(&x, Layer::LayerII, 20), Some(true));
        x[18] = 0.1; // X(20) - X(18) = 6.9 < 7
        assert_eq!(model1_step4_is_tonal(&x, Layer::LayerII, 20), Some(false));
    }

    #[test]
    fn step4_is_tonal_requires_every_offset() {
        use crate::frame::Layer;
        // k = 300 is in the Layer II top range (j up to ±12). One
        // near-level line at k + 12 defeats the whole conjunction.
        let mut x = flat_spectrum_l2(-10.0);
        x[300] = 50.0;
        x[312] = 44.0; // 50 - 44 = 6 < 7
        assert_eq!(model1_step4_is_tonal(&x, Layer::LayerII, 300), Some(false));
        x[312] = -10.0;
        assert_eq!(model1_step4_is_tonal(&x, Layer::LayerII, 300), Some(true));
    }

    #[test]
    fn step4_is_tonal_none_outside_examined_ranges() {
        use crate::frame::Layer;
        let x = flat_spectrum_l2(0.0);
        assert!(model1_step4_is_tonal(&x, Layer::LayerII, 2).is_none());
        assert!(model1_step4_is_tonal(&x, Layer::LayerII, 501).is_none());
        assert!(model1_step4_is_tonal(&x, Layer::LayerI, 251).is_none());
        assert!(model1_step4_is_tonal(&x, Layer::LayerIII, 100).is_none());
    }

    #[test]
    fn step4_tonal_spl_three_line_power_sum() {
        // Three equal lines at L dB sum to L + 10·log10(3), and the
        // formula is exactly the Step 2 Xspl power sum over the
        // three-line window.
        let x = [f64::NEG_INFINITY, 40.0, 40.0, 40.0, f64::NEG_INFINITY];
        let spl = model1_step4_tonal_spl_db(&x, 2).unwrap();
        assert!((spl - (40.0 + 10.0 * 3.0_f64.log10())).abs() < 1e-12);
        assert_eq!(spl, model1_step2_xspl_db(&x[1..=3]));
        // Neighbourless edges are undefined.
        assert!(model1_step4_tonal_spl_db(&x, 0).is_none());
        assert!(model1_step4_tonal_spl_db(&x, 4).is_none());
    }

    #[test]
    fn step4_extract_tonal_lists_and_zeroes_examined_range() {
        use crate::frame::Layer;
        // One sinusoid-like peak at k = 100 (mid range, j_max = 3)
        // over a -20 dB floor, with ±1 leakage lines at 10 dB.
        let mut x = flat_spectrum_l2(-20.0);
        x[100] = 30.0;
        x[99] = 10.0;
        x[101] = 10.0;
        let tonal = model1_step4_extract_tonal(&mut x, Layer::LayerII).unwrap();
        assert_eq!(tonal.len(), 1);
        assert_eq!(tonal[0].k, 100);
        assert_eq!(tonal[0].kind, MaskerKind::Tonal);
        let expected_spl = 10.0 * (10.0_f64.powf(3.0) + 2.0 * 10.0_f64.powf(1.0)).log10();
        assert!((tonal[0].spl_db - expected_spl).abs() < 1e-12);
        // The examined range k ± 3 is zeroed to -∞; the floor outside
        // is untouched.
        for (line, &v) in x.iter().enumerate().take(103 + 1).skip(97) {
            assert_eq!(v, f64::NEG_INFINITY, "line {line} must be -∞");
        }
        assert_eq!(x[96], -20.0);
        assert_eq!(x[104], -20.0);
    }

    #[test]
    fn step4_extract_tonal_decisions_use_the_original_spectrum() {
        use crate::frame::Layer;
        // Operation (a) labels maxima before any zeroing. The blocker
        // line at k = 102 (inside the k = 100 peak's examined range)
        // keeps k = 105 non-tonal: 25 - 20 = 5 < 7 dB. A sequential
        // in-pass zeroing would have erased the blocker first and
        // wrongly listed k = 105.
        let mut x = flat_spectrum_l2(-20.0);
        x[100] = 30.0;
        x[102] = 20.0;
        x[105] = 25.0;
        let tonal = model1_step4_extract_tonal(&mut x, Layer::LayerII).unwrap();
        assert_eq!(tonal.len(), 1);
        assert_eq!(tonal[0].k, 100);
    }

    #[test]
    fn step4_extract_tonal_rejects_wrong_length_and_layer3() {
        use crate::frame::Layer;
        let mut short = vec![0.0; MODEL1_FFT_LEN_LAYER1 / 2 + 1];
        assert!(model1_step4_extract_tonal(&mut short, Layer::LayerII).is_none());
        assert!(model1_step4_extract_tonal(&mut short, Layer::LayerI).is_some());
        let mut long = flat_spectrum_l2(0.0);
        assert!(model1_step4_extract_tonal(&mut long, Layer::LayerI).is_none());
        assert!(model1_step4_extract_tonal(&mut long, Layer::LayerIII).is_none());
    }

    #[test]
    fn step4_band_line_spans_tile_contiguously_from_line_1() {
        use crate::frame::Layer;
        // For every (layer, fs) table: band 0 starts at line 1, spans
        // tile with no gaps or overlaps, and the top band ends at the
        // table's last boundary line (frequency / line-spacing).
        let cases = [
            (Layer::LayerI, AnnexDSamplingRate::Hz32000, 240u16),
            (Layer::LayerI, AnnexDSamplingRate::Hz44100, 232),
            (Layer::LayerI, AnnexDSamplingRate::Hz48000, 216),
            (Layer::LayerII, AnnexDSamplingRate::Hz32000, 480),
            (Layer::LayerII, AnnexDSamplingRate::Hz44100, 464),
            (Layer::LayerII, AnnexDSamplingRate::Hz48000, 432),
        ];
        for (layer, fs, last_top) in cases {
            let spans = model1_step4_band_line_spans(layer, fs).unwrap();
            let table = critical_band_boundaries(layer, fs).unwrap();
            assert_eq!(spans.len(), table.len());
            assert_eq!(spans[0].k_first, 1, "{layer:?}/{fs:?}");
            let mut prev_last = 0u16;
            for s in &spans {
                assert_eq!(s.k_first, prev_last + 1, "{layer:?}/{fs:?} band {}", s.no);
                assert!(s.k_last >= s.k_first, "{layer:?}/{fs:?} band {}", s.no);
                prev_last = s.k_last;
            }
            assert_eq!(prev_last, last_top, "{layer:?}/{fs:?} top line");
        }
        assert!(
            model1_step4_band_line_spans(Layer::LayerIII, AnnexDSamplingRate::Hz44100).is_none()
        );
    }

    #[test]
    fn step4_band_line_spans_d2d_anchor_rows() {
        use crate::frame::Layer;
        // D.2d (Layer II, 32 kHz; line spacing 31,25 Hz): band 0 tops
        // at 31,25 Hz = line 1, band 1 at 93,75 Hz = line 3, band 2 at
        // 187,5 Hz = line 6; band 22 tops at 9 250 Hz = line 296, so
        // band 23 (top 11 500 Hz = line 368) spans 297..=368 and band
        // 24 (top 15 000 Hz = line 480) spans 369..=480.
        let spans =
            model1_step4_band_line_spans(Layer::LayerII, AnnexDSamplingRate::Hz32000).unwrap();
        assert_eq!((spans[0].k_first, spans[0].k_last), (1, 1));
        assert_eq!((spans[1].k_first, spans[1].k_last), (2, 3));
        assert_eq!((spans[2].k_first, spans[2].k_last), (4, 6));
        assert_eq!((spans[23].k_first, spans[23].k_last), (297, 368));
        assert_eq!((spans[24].k_first, spans[24].k_last), (369, 480));
    }

    #[test]
    fn step4_non_tonal_flat_spectrum_per_band_power() {
        use crate::frame::Layer;
        // On a flat 0 dB residue every band's non-tonal SPL is
        // 10·log10(width) and the listed line is round(sqrt(first·last)),
        // inside the band.
        let x = flat_spectrum_l2(0.0);
        let spans =
            model1_step4_band_line_spans(Layer::LayerII, AnnexDSamplingRate::Hz32000).unwrap();
        let nt = model1_step4_non_tonal_components(&x, Layer::LayerII, AnnexDSamplingRate::Hz32000)
            .unwrap();
        assert_eq!(nt.len(), spans.len());
        for (c, s) in nt.iter().zip(&spans) {
            assert_eq!(c.kind, MaskerKind::NonTonal);
            let width = f64::from(s.k_last - s.k_first + 1);
            assert!(
                (c.spl_db - 10.0 * width.log10()).abs() < 1e-12,
                "band {}: spl {} vs width {}",
                s.no,
                c.spl_db,
                width
            );
            let gm = (f64::from(s.k_first) * f64::from(s.k_last)).sqrt().round() as u16;
            assert_eq!(c.k, gm);
            assert!(c.k >= s.k_first && c.k <= s.k_last, "band {}", s.no);
        }
        // Anchor the geometric-mean line of the widest D.2d band:
        // round(sqrt(369·480)) = round(420,86) = 421.
        assert_eq!(nt[24].k, 421);
    }

    #[test]
    fn step4_non_tonal_geometric_mean_inside_span_all_tables() {
        use crate::frame::Layer;
        for layer in [Layer::LayerI, Layer::LayerII] {
            let len = match layer {
                Layer::LayerI => MODEL1_FFT_LEN_LAYER1 / 2 + 1,
                _ => MODEL1_FFT_LEN_LAYER2 / 2 + 1,
            };
            let x = vec![-30.0; len];
            for fs in [
                AnnexDSamplingRate::Hz32000,
                AnnexDSamplingRate::Hz44100,
                AnnexDSamplingRate::Hz48000,
            ] {
                let spans = model1_step4_band_line_spans(layer, fs).unwrap();
                let nt = model1_step4_non_tonal_components(&x, layer, fs).unwrap();
                for (c, s) in nt.iter().zip(&spans) {
                    assert!(
                        c.k >= s.k_first && c.k <= s.k_last,
                        "{layer:?}/{fs:?} band {}: line {} outside [{}, {}]",
                        s.no,
                        c.k,
                        s.k_first,
                        s.k_last
                    );
                }
            }
        }
    }

    #[test]
    fn step4_components_end_to_end_peak_plus_floor() {
        use crate::frame::Layer;
        // Peak at k = 100 over a -30 dB floor (Layer II, 44,1 kHz).
        // The tonal list carries the peak; the band containing it
        // (D.2e band 19 spans lines 91..=104) loses the zeroed peak
        // energy from its non-tonal sum.
        let mut x = flat_spectrum_l2(-30.0);
        x[100] = 40.0;
        x[99] = 20.0;
        x[101] = 20.0;
        let (tonal, non_tonal) =
            model1_step4_components(&x, Layer::LayerII, AnnexDSamplingRate::Hz44100).unwrap();
        assert_eq!(tonal.len(), 1);
        assert_eq!(tonal[0].k, 100);
        // 27 critical bands per Table D.2e.
        assert_eq!(non_tonal.len(), CRITICAL_BANDS_D2E.len());
        // The unzeroed spectrum would put the peak's power into the
        // band's non-tonal sum; the residue must come in lower.
        let unzeroed =
            model1_step4_non_tonal_components(&x, Layer::LayerII, AnnexDSamplingRate::Hz44100)
                .unwrap();
        let spans =
            model1_step4_band_line_spans(Layer::LayerII, AnnexDSamplingRate::Hz44100).unwrap();
        let band = spans
            .iter()
            .position(|s| s.k_first <= 100 && 100 <= s.k_last)
            .unwrap();
        assert_eq!(spans[band].no, 19);
        assert!(non_tonal[band].spl_db < unzeroed[band].spl_db - 30.0);
        // Bands away from the peak are untouched by the zeroing.
        assert_eq!(non_tonal[5].spl_db, unzeroed[5].spl_db);
    }

    #[test]
    fn step4_components_from_step1_spectrum_detects_pure_tone() {
        use crate::frame::Layer;
        // Full Step 1 → normalize → Step 4 chain: a pure sine at bin
        // 100 yields exactly one above-floor tonal component at
        // k = 100 whose three-line SPL exceeds the 96 dB peak line by
        // the two ±1 Hann sidelines (each exactly 6,02 dB down:
        // +10·log10(1,5) ≈ 1,76 dB).
        let n_fft = MODEL1_FFT_LEN_LAYER2;
        let s: Vec<f64> = (0..n_fft)
            .map(|l| (2.0 * core::f64::consts::PI * 100.0 * (l as f64) / (n_fft as f64)).sin())
            .collect();
        let mut x = model1_power_density_spectrum(&s).unwrap();
        model1_normalize_to_96db_spl(&mut x).unwrap();
        let (tonal, non_tonal) =
            model1_step4_components(&x, Layer::LayerII, AnnexDSamplingRate::Hz44100).unwrap();
        let loud: Vec<_> = tonal.iter().filter(|c| c.spl_db > 0.0).collect();
        assert_eq!(loud.len(), 1);
        assert_eq!(loud[0].k, 100);
        let expected = MODEL1_SPL_REFERENCE_DB + 10.0 * 1.5_f64.log10();
        assert!((loud[0].spl_db - expected).abs() < 1e-6);
        assert_eq!(non_tonal.len(), CRITICAL_BANDS_D2E.len());
        // Every non-tonal component is at/below the numerical noise
        // floor — the tone's energy left the residue with the zeroing.
        for c in &non_tonal {
            assert!(c.spl_db < 0.0, "band line {}: {}", c.k, c.spl_db);
        }
    }

    #[test]
    fn step4_components_silent_spectrum_yields_no_tonal_and_silent_bands() {
        use crate::frame::Layer;
        let x = flat_spectrum_l2(f64::NEG_INFINITY);
        let (tonal, non_tonal) =
            model1_step4_components(&x, Layer::LayerII, AnnexDSamplingRate::Hz48000).unwrap();
        assert!(tonal.is_empty());
        assert_eq!(non_tonal.len(), CRITICAL_BANDS_D2F.len());
        for c in &non_tonal {
            assert_eq!(c.spl_db, f64::NEG_INFINITY);
            assert_eq!(c.kind, MaskerKind::NonTonal);
        }
    }

    // ---- Phase 2 step 80 — Tables D.1a–f + Step 4 → Bark bridge +
    // end-to-end §D.1 Step 5 sieve.

    /// All six (table, layer, fs, FFT length) dispatch cases.
    fn d1_cases() -> [(
        &'static [Model1ThresholdEntry],
        crate::frame::Layer,
        AnnexDSamplingRate,
        usize,
    ); 6] {
        use crate::frame::Layer;
        [
            (
                &MODEL1_THRESHOLD_D1A,
                Layer::LayerI,
                AnnexDSamplingRate::Hz32000,
                512,
            ),
            (
                &MODEL1_THRESHOLD_D1B,
                Layer::LayerI,
                AnnexDSamplingRate::Hz44100,
                512,
            ),
            (
                &MODEL1_THRESHOLD_D1C,
                Layer::LayerI,
                AnnexDSamplingRate::Hz48000,
                512,
            ),
            (
                &MODEL1_THRESHOLD_D1D,
                Layer::LayerII,
                AnnexDSamplingRate::Hz32000,
                1024,
            ),
            (
                &MODEL1_THRESHOLD_D1E,
                Layer::LayerII,
                AnnexDSamplingRate::Hz44100,
                1024,
            ),
            (
                &MODEL1_THRESHOLD_D1F,
                Layer::LayerII,
                AnnexDSamplingRate::Hz48000,
                1024,
            ),
        ]
    }

    #[test]
    fn table_d1_lengths_and_dispatch() {
        use crate::frame::Layer;
        let expect = [108usize, 106, 102, 132, 130, 126];
        for ((table, layer, fs, _), len) in d1_cases().into_iter().zip(expect) {
            assert_eq!(table.len(), len, "{layer:?}/{fs:?}");
            let dispatched = model1_threshold_table(layer, fs).unwrap();
            assert_eq!(dispatched, table);
        }
        assert!(model1_threshold_table(Layer::LayerIII, AnnexDSamplingRate::Hz44100).is_none());
    }

    #[test]
    fn table_d1_frequency_column_matches_line_grid() {
        // Every printed frequency is the line-center frequency of the
        // raw FFT line `model1_d1_line_for_index` assigns to the row
        // (within the 2-decimal print rounding) — this pins both the
        // frequency-column transcription and the index → line map for
        // all 704 rows.
        for (table, layer, fs, n_fft) in d1_cases() {
            let delta_f = f64::from(fs.as_hz()) / (n_fft as f64);
            for (pos, entry) in table.iter().enumerate() {
                let i = pos as u16 + 1;
                let line = model1_d1_line_for_index(layer, fs, i).unwrap();
                let grid = f64::from(line) * delta_f;
                assert!(
                    (entry.frequency_hz - grid).abs() < 0.006,
                    "{layer:?}/{fs:?} row {i}: {} vs line {line} = {grid}",
                    entry.frequency_hz,
                );
            }
            // One past the end (and index 0) refuse.
            assert!(model1_d1_line_for_index(layer, fs, 0).is_none());
            assert!(model1_d1_line_for_index(layer, fs, table.len() as u16 + 1).is_none());
        }
    }

    #[test]
    fn table_d1_bark_column_strictly_increasing() {
        for (table, layer, fs, _) in d1_cases() {
            let mut prev = f64::NEG_INFINITY;
            for (pos, entry) in table.iter().enumerate() {
                assert!(
                    entry.z_bark > prev,
                    "{layer:?}/{fs:?} row {}: z {} after {}",
                    pos + 1,
                    entry.z_bark,
                    prev
                );
                prev = entry.z_bark;
                // LTq stays within the printed range (minimum −4,98 dB
                // near 3,3 kHz; 68,00 dB cap at the top of the band).
                assert!((-4.99..=68.0).contains(&entry.ltq_db), "{layer:?}/{fs:?}");
            }
        }
    }

    #[test]
    fn table_d1_layer2_embeds_layer1_rows() {
        use crate::frame::Layer;
        // At the same Fs, Layer I line L and Layer II line 2L are the
        // same frequency, and the spec prints identical z / LTq for
        // them. Every Layer I row therefore reappears in the Layer II
        // table (rows 49.. of D.1d/e/f are rows 25.. of D.1a/b/c).
        for fs in [
            AnnexDSamplingRate::Hz32000,
            AnnexDSamplingRate::Hz44100,
            AnnexDSamplingRate::Hz48000,
        ] {
            let t1 = model1_threshold_table(Layer::LayerI, fs).unwrap();
            let t2 = model1_threshold_table(Layer::LayerII, fs).unwrap();
            for (pos, e1) in t1.iter().enumerate() {
                let i1 = pos as u16 + 1;
                let line2 = 2 * model1_d1_line_for_index(Layer::LayerI, fs, i1).unwrap();
                let i2 = model1_d1_index_for_line(Layer::LayerII, fs, line2).unwrap();
                assert_eq!(
                    model1_d1_line_for_index(Layer::LayerII, fs, i2).unwrap(),
                    line2,
                    "{fs:?} L1 row {i1}: Layer II line {line2} not on the grid"
                );
                let e2 = &t2[i2 as usize - 1];
                assert!(
                    (e1.frequency_hz - e2.frequency_hz).abs() < 0.005,
                    "{fs:?} row {i1}"
                );
                assert_eq!(e1.z_bark, e2.z_bark, "{fs:?} L1 row {i1} → L2 row {i2}");
                assert_eq!(e1.ltq_db, e2.ltq_db, "{fs:?} L1 row {i1} → L2 row {i2}");
            }
        }
    }

    #[test]
    fn table_d1_agrees_with_d2_boundary_rows() {
        // The Tables D.2 `index F&CB` column indexes into Table D.1:
        // every boundary row's frequency / Bark pair equals the cited
        // Table D.1 row. This includes D.2e band 17, whose clipped
        // `16,11[.]` print D.1e row 62 resolves to exactly 16,110 =
        // the stored 16.11. Exceptions: the six 44,1 kHz cells where
        // the printed spec's D.2 tables sit exactly 0,001 Bark below
        // the (double-printed, self-consistent) D.1 tables — see the
        // print-discrepancy notes on `CRITICAL_BANDS_D2B` /
        // `CRITICAL_BANDS_D2E`.
        let exceptions = [
            // (fs, band no per layer pair, D.2 print, D.1 print)
            (17u16, 19u16, 17.904, 17.905),
            (20, 22, 20.971, 20.972),
            (24, 26, 24.573, 24.574),
        ];
        use crate::frame::Layer;
        for (table, layer, fs, _) in d1_cases() {
            let boundaries = critical_band_boundaries(layer, fs).unwrap();
            for b in boundaries {
                let entry = &table[b.index_fcb as usize - 1];
                // D.2 prints 3 decimals, D.1 prints 2: the two
                // roundings can differ by up to 0,005 Hz.
                assert!(
                    (b.frequency_hz - entry.frequency_hz).abs() < 0.0051,
                    "{layer:?}/{fs:?} band {}: D.2 {} vs D.1 {}",
                    b.no,
                    b.frequency_hz,
                    entry.frequency_hz,
                );
                let exception = fs == AnnexDSamplingRate::Hz44100
                    && exceptions.iter().any(|&(no1, no2, d2, d1)| {
                        let no = match layer {
                            Layer::LayerI => no1,
                            _ => no2,
                        };
                        b.no == no && b.z_bark == d2 && entry.z_bark == d1
                    });
                assert!(
                    exception || b.z_bark == entry.z_bark,
                    "{layer:?}/{fs:?} band {} (D.1 index {}): D.2 z {} vs D.1 z {}",
                    b.no,
                    b.index_fcb,
                    b.z_bark,
                    entry.z_bark,
                );
            }
        }
    }

    #[test]
    fn table_d1_matches_textual_ltq_anchors() {
        // The seven textually-transcribed Table D.1a anchors that
        // `ltq_db_at_hz` interpolates through are rows 1–5, 51 and
        // 108 of the full table.
        for (i, hz, ltq) in [
            (1u16, 62.5, 33.44),
            (2, 125.0, 19.20),
            (3, 187.5, 13.87),
            (4, 250.0, 11.01),
            (5, 312.5, 9.20),
            (51, 3375.0, -4.97),
            (108, 15000.0, 51.04),
        ] {
            let entry = &MODEL1_THRESHOLD_D1A[i as usize - 1];
            assert_eq!(entry.frequency_hz, hz);
            assert_eq!(entry.ltq_db, ltq);
            assert!((ltq_db_at_hz(hz) - ltq).abs() < 1.0e-9);
        }
    }

    #[test]
    fn table_d1_index_for_line_round_trip_and_nearest() {
        use crate::frame::Layer;
        // Round-trip: every tabulated line maps back to its own row.
        for (table, layer, fs, _) in d1_cases() {
            for i in 1..=table.len() as u16 {
                let line = model1_d1_line_for_index(layer, fs, i).unwrap();
                assert_eq!(
                    model1_d1_index_for_line(layer, fs, line),
                    Some(i),
                    "{layer:?}/{fs:?} row {i} line {line}"
                );
            }
        }
        // Nearest-with-tie-down on unlisted lines (Layer II tables;
        // listed lines …48, 50, 52, …, 96, 100, 104, …, 192, 200,
        // 208, …).
        let (l2, fs) = (Layer::LayerII, AnnexDSamplingRate::Hz44100);
        assert_eq!(model1_d1_index_for_line(l2, fs, 51), Some(49)); // tie → down
        assert_eq!(model1_d1_index_for_line(l2, fs, 53), Some(50)); // tie → down
        assert_eq!(model1_d1_index_for_line(l2, fs, 97), Some(72)); // 96 at 1 vs 100 at 3
        assert_eq!(model1_d1_index_for_line(l2, fs, 98), Some(72)); // tie → down
        assert_eq!(model1_d1_index_for_line(l2, fs, 99), Some(73)); // 100 at 1 vs 96 at 3
        assert_eq!(model1_d1_index_for_line(l2, fs, 203), Some(97)); // 200 at 3 vs 208 at 5
        assert_eq!(model1_d1_index_for_line(l2, fs, 205), Some(98)); // 208 at 3 vs 200 at 5
                                                                     // Bounds: DC and above the last tabulated line refuse.
        assert_eq!(model1_d1_index_for_line(l2, fs, 0), None);
        assert_eq!(model1_d1_index_for_line(l2, fs, 464), Some(130));
        assert_eq!(model1_d1_index_for_line(l2, fs, 465), None);
        let (l1, fs32) = (Layer::LayerI, AnnexDSamplingRate::Hz32000);
        assert_eq!(model1_d1_index_for_line(l1, fs32, 240), Some(108));
        assert_eq!(model1_d1_index_for_line(l1, fs32, 241), None);
        assert_eq!(model1_d1_index_for_line(Layer::LayerIII, fs, 100), None);
    }

    #[test]
    fn step5_masker_bridge_places_components_on_table_d1_bark() {
        use crate::frame::Layer;
        let (l2, fs) = (Layer::LayerII, AnnexDSamplingRate::Hz44100);
        // Tonal component at line 100 → D.1e row 73 (4 306,64 Hz,
        // z = 17,680).
        let tonal = Model1Step4Component {
            k: 100,
            spl_db: 60.0,
            kind: MaskerKind::Tonal,
        };
        let m = model1_masker_from_component(&tonal, l2, fs).unwrap();
        assert_eq!(m.kind, MaskerKind::Tonal);
        assert_eq!(m.z_bark, 17.680);
        assert_eq!(m.spl_db, 60.0);
        // Non-tonal component at line 50 → D.1e row 49 (2 153,32 Hz,
        // z = 13,578); kind passes through.
        let non_tonal = Model1Step4Component {
            k: 50,
            spl_db: 40.0,
            kind: MaskerKind::NonTonal,
        };
        let m = model1_masker_from_component(&non_tonal, l2, fs).unwrap();
        assert_eq!(m.kind, MaskerKind::NonTonal);
        assert_eq!(m.z_bark, 13.578);
        // Above the table's last tabulated line (464) and Layer III:
        // no placement.
        let high = Model1Step4Component {
            k: 470,
            spl_db: 60.0,
            kind: MaskerKind::Tonal,
        };
        assert!(model1_masker_from_component(&high, l2, fs).is_none());
        assert!(model1_masker_from_component(&tonal, Layer::LayerIII, fs).is_none());
    }

    #[test]
    fn step5_components_screen_and_decimate_end_to_end() {
        use crate::frame::Layer;
        let (l2, fs) = (Layer::LayerII, AnnexDSamplingRate::Hz44100);
        let t = |k: u16, spl_db: f64| Model1Step4Component {
            k,
            spl_db,
            kind: MaskerKind::Tonal,
        };
        let n = |k: u16, spl_db: f64| Model1Step4Component {
            k,
            spl_db,
            kind: MaskerKind::NonTonal,
        };
        let tonal = [
            t(100, 60.0), // kept: 60 ≥ LTq(row 73) = −2,06
            t(101, 50.0), // same nearest row (73) → same z → Step 5(b)
            // cluster with the 60 dB masker → dropped
            t(24, -10.0), // Step 5(a): −10 < LTq(row 24) = 3,25 → dropped
        ];
        let non_tonal = [
            n(50, 40.0),  // kept: 40 ≥ LTq(row 49) = −0,96
            n(470, 20.0), // above the last tabulated line → dropped
        ];
        let maskers = model1_step5_components(&tonal, &non_tonal, l2, fs).unwrap();
        assert_eq!(maskers.len(), 2);
        assert_eq!(maskers[0].kind, MaskerKind::Tonal);
        assert_eq!(maskers[0].z_bark, 17.680);
        assert_eq!(maskers[0].spl_db, 60.0);
        assert_eq!(maskers[1].kind, MaskerKind::NonTonal);
        assert_eq!(maskers[1].z_bark, 13.578);
        assert_eq!(maskers[1].spl_db, 40.0);
        // Layer III: no Annex D tables.
        assert!(model1_step5_components(&tonal, &non_tonal, Layer::LayerIII, fs).is_none());
    }

    #[test]
    fn step5_chain_from_step1_pure_tone_through_global_threshold() {
        use crate::frame::Layer;
        // Full Step 1 → normalize → Step 4 → Step 5 chain on a pure
        // sine at bin 100 (Layer II, 44,1 kHz): exactly one masker
        // survives the sieve — the tonal component at line 100, on
        // the D.1e row-73 Bark coordinate — and feeds Step 6/7.
        let n_fft = MODEL1_FFT_LEN_LAYER2;
        let s: Vec<f64> = (0..n_fft)
            .map(|l| (2.0 * core::f64::consts::PI * 100.0 * (l as f64) / (n_fft as f64)).sin())
            .collect();
        let mut x = model1_power_density_spectrum(&s).unwrap();
        model1_normalize_to_96db_spl(&mut x).unwrap();
        let (tonal, non_tonal) =
            model1_step4_components(&x, Layer::LayerII, AnnexDSamplingRate::Hz44100).unwrap();
        let maskers = model1_step5_components(
            &tonal,
            &non_tonal,
            Layer::LayerII,
            AnnexDSamplingRate::Hz44100,
        )
        .unwrap();
        assert_eq!(maskers.len(), 1, "only the tone survives Step 5");
        assert_eq!(maskers[0].kind, MaskerKind::Tonal);
        assert_eq!(maskers[0].z_bark, 17.680);
        let expected_spl = MODEL1_SPL_REFERENCE_DB + 10.0 * 1.5_f64.log10();
        assert!((maskers[0].spl_db - expected_spl).abs() < 1e-6);
        // Step 6/7: the global threshold next to the masker is
        // dominated by its individual threshold (LTq term is tiny).
        let z_i = maskers[0].z_bark + 0.32;
        let ltq_db = -0.04; // D.1e row 76 neighbourhood
        let ltg = global_masking_threshold_db(&maskers, z_i, ltq_db);
        let lt = individual_masking_threshold_db(&maskers[0], z_i).unwrap();
        assert!(ltg >= lt && ltg < lt + 0.01, "ltg {ltg} vs lt {lt}");
    }

    // ---- Phase 2 step 81 / r279 — §D.2.3 base Model 2 spreading
    // function + §D.2.4 step f) convolution/renormalization + step g)
    // tonality index.

    /// The `bval` column of the first 20 rows of Table D.3a (Fs =
    /// 32 kHz calculation partition table) — the text anchor
    /// transcribed in
    /// `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`. A
    /// realistic, strictly-increasing median-Bark fixture for the
    /// step-f) reductions (the full 63-row table stays a PNG
    /// transcription gap).
    const D3A_BVAL_ANCHOR_20: [f64; 20] = [
        0.00, 0.63, 1.56, 2.50, 3.44, 4.34, 5.17, 5.94, 6.63, 7.28, 7.90, 8.50, 9.06, 9.65, 10.28,
        10.87, 11.41, 11.92, 12.39, 12.83,
    ];

    #[test]
    fn model2_sprdngf_diagonal_is_unity_within_printed_constant_rounding() {
        // At i == j: tmpx = 0, u = -0.5 gives u^2 - 2u = 1.25 > 0 so
        // x = 0, and tmpy = 15.811389 + 7.5*0.474
        // - 17.5*sqrt(1 + 0.474^2) — the printed constant 15,811389
        // equals 17,5*sqrt(1 + 0,474^2) - 7,5*0,474 to its printed
        // precision, so tmpy ≈ 0 and sprdngf ≈ 10^0 = 1. The
        // residual is the rounding of the 6-decimal constant
        // (≈ 5e-7 relative).
        for z in [0.0, 0.63, 5.17, 12.83, 24.5] {
            let f = model2_sprdngf(z, z);
            assert!((f - 1.0).abs() < 1.0e-5, "sprdngf({z},{z}) = {f}");
            assert_eq!(model2_sprdngf_x_db(model2_sprdngf_tmpx(z, z)), 0.0);
        }
    }

    #[test]
    fn model2_sprdngf_x_parabola_active_only_on_near_upward_skirt() {
        // x = 8 * min((tmpx-0.5)^2 - 2*(tmpx-0.5), 0): with
        // u = tmpx - 0.5 the inner expression u^2 - 2u is negative
        // exactly for 0 < u < 2, i.e. 0.5 < tmpx < 2.5.
        assert_eq!(model2_sprdngf_x_db(0.5), 0.0);
        assert_eq!(model2_sprdngf_x_db(2.5), 0.0);
        assert_eq!(model2_sprdngf_x_db(0.0), 0.0);
        assert_eq!(model2_sprdngf_x_db(-3.0), 0.0);
        assert_eq!(model2_sprdngf_x_db(4.0), 0.0);
        // Most negative at tmpx = 1.5 (u = 1): 8 * (1 - 2) = -8 dB.
        assert_eq!(model2_sprdngf_x_db(1.5), -8.0);
        for tmpx in [0.6, 1.0, 2.0, 2.4] {
            let x = model2_sprdngf_x_db(tmpx);
            assert!((-8.0..0.0).contains(&x), "x({tmpx}) = {x}");
        }
    }

    #[test]
    fn model2_sprdngf_matches_hand_substituted_one_bark_up() {
        // j - i = 1 Bark upward: tmpx = 1.05. Substituting into the
        // printed formulas (independent re-derivation):
        //   u = 0.55, x = 8*(0.3025 - 1.1) = -6.38
        //   v = 1.524, tmpy = 15.811389 + 11.43 - 17.5*sqrt(3.322576)
        //   sprdngf = 10^((x + tmpy)/10)
        let x = 8.0 * (0.55_f64 * 0.55 - 2.0 * 0.55);
        assert!((x + 6.38).abs() < 1.0e-12);
        let tmpy = 15.811_389 + 7.5 * 1.524 - 17.5 * (1.0 + 1.524_f64 * 1.524).sqrt();
        let expected = (10.0_f64).powf((x + tmpy) / 10.0);
        let got = model2_sprdngf(10.0, 11.0);
        assert!((got - expected).abs() < 1.0e-12, "{got} vs {expected}");
        // And the components agree with the helpers.
        assert!((model2_sprdngf_tmpy_db(1.05) - tmpy).abs() < 1.0e-12);
        assert!((model2_sprdngf_x_db(1.05) - x).abs() < 1.0e-12);
    }

    #[test]
    fn model2_sprdngf_tmpy_cutoff_zeroes_far_spreads() {
        // 6 Bark downward: tmpx = -6.3, v = -5.826 gives
        // tmpy = 15.811389 - 43.695 - 17.5*sqrt(34.942276) ≈ -131.3,
        // below the -100 dB cutoff → exact 0.
        assert!(model2_sprdngf_tmpy_db(-6.3) < MODEL2_SPRDNGF_TMPY_CUTOFF_DB);
        assert_eq!(model2_sprdngf(10.0, 4.0), 0.0);
        // 12 Bark upward: tmpx = 12.6, tmpy ≈ -115.6 → exact 0.
        assert!(model2_sprdngf_tmpy_db(12.6) < MODEL2_SPRDNGF_TMPY_CUTOFF_DB);
        assert_eq!(model2_sprdngf(4.0, 16.0), 0.0);
        // 4 Bark downward: tmpx = -4.2, tmpy ≈ -79.6 → survives.
        assert!(model2_sprdngf_tmpy_db(-4.2) > MODEL2_SPRDNGF_TMPY_CUTOFF_DB);
        assert!(model2_sprdngf(10.0, 6.0) > 0.0);
    }

    #[test]
    fn model2_sprdngf_upward_reach_exceeds_downward_reach() {
        // The asymmetric skirt: ~-10 dB per tmpx upward vs ~-25 dB
        // downward. At 5 Bark distance the upward factor survives
        // (tmpy(5.25) ≈ -42.9) while the downward factor is already
        // cut off (tmpy(-5.25) ≈ -105.4 < -100).
        let z = 12.0;
        assert!(model2_sprdngf(z, z + 5.0) > 0.0);
        assert_eq!(model2_sprdngf(z, z - 5.0), 0.0);
        // Within the surviving range the factor decays monotonically
        // with distance in both directions.
        let up1 = model2_sprdngf(z, z + 1.0);
        let up3 = model2_sprdngf(z, z + 3.0);
        let dn1 = model2_sprdngf(z, z - 1.0);
        let dn3 = model2_sprdngf(z, z - 3.0);
        assert!(up1 < 1.0 && up3 < up1, "up: {up1} {up3}");
        assert!(dn1 < 1.0 && dn3 < dn1, "down: {dn1} {dn3}");
        // And at equal distance the downward factor is the smaller
        // one (steeper downward skirt).
        assert!(dn3 < up3, "asymmetry: {dn3} vs {up3}");
    }

    #[test]
    fn model2_step_f_spread_impulse_recovers_sprdngf_row() {
        // A unit impulse in partition 5 (index 4): the convolution
        // out_b = Σ_bb in_bb * sprdngf(bval_bb, bval_b) collapses to
        // the single term sprdngf(bval_4, bval_b).
        let bval = &D3A_BVAL_ANCHOR_20;
        let mut e = [0.0; 20];
        e[4] = 1.0;
        let ecb = model2_step_f_spread(&e, bval).unwrap();
        assert_eq!(ecb.len(), 20);
        for (b, &got) in ecb.iter().enumerate() {
            let expected = model2_sprdngf(bval[4], bval[b]);
            assert!(
                (got - expected).abs() < 1.0e-12,
                "b={b}: {got} vs {expected}"
            );
        }
        // The masker's own partition holds the largest entry.
        let peak = ecb.iter().cloned().fold(f64::MIN, f64::max);
        assert!((ecb[4] - peak).abs() < 1.0e-12);
    }

    #[test]
    fn model2_step_f_length_mismatches_return_none() {
        let bval = &D3A_BVAL_ANCHOR_20;
        assert!(model2_step_f_spread(&[1.0; 19], bval).is_none());
        assert!(model2_step_f_cb(&[1.0; 3], &[1.0; 4]).is_none());
        assert!(model2_step_f_en(&[1.0; 4], &[1.0; 3]).is_none());
    }

    #[test]
    fn model2_step_f_rnorm_normalizes_uniform_energy() {
        // With e_b = 1 everywhere, ecb_b is exactly the
        // spreading-function row sum, so en_b = ecb_b * rnorm_b must
        // come back to 1 in every partition — the spec's stated
        // purpose of rnorm ("due to the non-normalized nature of the
        // spreading function").
        let bval = &D3A_BVAL_ANCHOR_20;
        let ones = [1.0; 20];
        let ecb = model2_step_f_spread(&ones, bval).unwrap();
        let rnorm = model2_step_f_rnorm(bval);
        assert!(rnorm.iter().all(|&r| r > 0.0 && r <= 1.0));
        let en = model2_step_f_en(&ecb, &rnorm).unwrap();
        for (b, &en_b) in en.iter().enumerate() {
            assert!((en_b - 1.0).abs() < 1.0e-12, "en[{b}] = {en_b}");
        }
    }

    #[test]
    fn model2_step_f_cb_recovers_constant_unpredictability() {
        // If every FFT line carries the same unpredictability
        // c_ω = 0.3, then step e) gives c_b = 0.3 * e_b in every
        // partition, both convolutions scale identically, and the
        // renormalization cb_b = ct_b / ecb_b recovers 0.3 exactly.
        let bval = &D3A_BVAL_ANCHOR_20;
        let e: Vec<f64> = (0..20).map(|b| 1.0 + 0.5 * b as f64).collect();
        let c: Vec<f64> = e.iter().map(|&eb| 0.3 * eb).collect();
        let ecb = model2_step_f_spread(&e, bval).unwrap();
        let ct = model2_step_f_spread(&c, bval).unwrap();
        let cb = model2_step_f_cb(&ct, &ecb).unwrap();
        for (b, &cb_b) in cb.iter().enumerate() {
            assert!((cb_b - 0.3).abs() < 1.0e-12, "cb[{b}] = {cb_b}");
        }
    }

    #[test]
    fn model2_step_f_cb_zero_energy_partition_yields_zero() {
        // All-zero spectrum: ecb_b = ct_b = 0 everywhere; the
        // documented convention defines cb_b = 0 (instead of 0/0).
        let bval = &D3A_BVAL_ANCHOR_20;
        let zeros = [0.0; 20];
        let ecb = model2_step_f_spread(&zeros, bval).unwrap();
        let cb = model2_step_f_cb(&ecb, &ecb).unwrap();
        assert!(cb.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn model2_step_g_tonality_clamps_at_both_ends() {
        // cb = 1: ln(1) = 0 → unclamped tb = -0.299 → clamps to 0
        // (fully noise-like).
        assert_eq!(model2_step_g_tonality(1.0), 0.0);
        // Unclamped tb crosses 0 at cb = e^(-0.299/0.43) ≈ 0.4989;
        // anything above that clamps to 0.
        assert_eq!(model2_step_g_tonality(0.6), 0.0);
        // cb = 0 (the zero-energy convention): ln → -∞, unclamped tb
        // → +∞ → clamps to 1 (fully tone-like).
        assert_eq!(model2_step_g_tonality(0.0), 1.0);
        // Unclamped tb crosses 1 at cb = e^(-1.299/0.43) ≈ 0.0488;
        // anything below that clamps to 1.
        assert_eq!(model2_step_g_tonality(0.01), 1.0);
    }

    #[test]
    fn model2_step_g_tonality_matches_formula_and_is_monotone() {
        // Interior point: tb = -0.299 - 0.43*ln(cb), re-derived
        // inline from the printed step g) formula.
        for cb in [0.05, 0.1, 0.2, 0.3, 0.45] {
            let expected = -0.299 - 0.43 * f64::ln(cb);
            let got = model2_step_g_tonality(cb);
            assert!(
                (got - expected).abs() < 1.0e-12,
                "cb={cb}: {got} vs {expected}"
            );
            assert!((0.0..=1.0).contains(&got));
        }
        // Monotone non-increasing in cb: more unpredictability →
        // less tonal.
        let t1 = model2_step_g_tonality(0.06);
        let t2 = model2_step_g_tonality(0.2);
        let t3 = model2_step_g_tonality(0.45);
        assert!(t1 > t2 && t2 > t3, "{t1} {t2} {t3}");
    }

    // ----- §D.2.4 steps h)–l) + n) (step 83 / r281) -----

    #[test]
    fn model2_step_h_interpolates_tmn_to_nmt() {
        // Fully tonal (tb = 1): the interpolation collapses to TMN.
        assert_eq!(model2_step_h_snr_db(1.0, 0.0, 24.5), 24.5);
        // Fully noise-like (tb = 0): collapses to NMT = 5,5 dB.
        assert_eq!(model2_step_h_snr_db(0.0, 0.0, 24.5), MODEL2_NMT_DB);
        // Interior tb: linear interpolation, re-derived inline from
        // the printed step h) formula.
        for tb in [0.25, 0.5, 0.75] {
            let expected = tb * 24.5 + (1.0 - tb) * 5.5;
            let got = model2_step_h_snr_db(tb, 0.0, 24.5);
            assert!((got - expected).abs() < 1.0e-12, "tb={tb}: {got}");
        }
    }

    #[test]
    fn model2_step_h_minval_is_a_lower_limit() {
        // Table D.3a row 3 prints minval = 20,0 / TMN = 24,5. At
        // tb = 0 the interpolated value (5,5 dB) sits below minval,
        // so the maximum returns minval.
        assert_eq!(model2_step_h_snr_db(0.0, 20.0, 24.5), 20.0);
        // At tb = 1 the interpolated value (24,5 dB) exceeds minval.
        assert_eq!(model2_step_h_snr_db(1.0, 20.0, 24.5), 24.5);
    }

    #[test]
    fn model2_step_h_slice_reads_d3_columns() {
        // Drive the slice form over the first three Table D.3a rows
        // (minval 0 / 0 / 20, TMN 24,5 throughout) with tb = 0
        // everywhere: rows 1–2 give NMT = 5,5; row 3's minval floor
        // gives 20.
        let partitions = &MODEL2_PARTITION_D3A[..3];
        let snr = model2_step_h_snr(&[0.0; 3], partitions).unwrap();
        assert_eq!(snr, vec![5.5, 5.5, 20.0]);
        // Length mismatch.
        assert!(model2_step_h_snr(&[0.0; 4], partitions).is_none());
    }

    #[test]
    fn model2_step_i_power_ratio() {
        // bc = 10^(-SNR/10): 0 dB → 1; 10 dB → 0,1; 20 dB → 0,01.
        assert!((model2_step_i_bc(0.0) - 1.0).abs() < 1.0e-15);
        assert!((model2_step_i_bc(10.0) - 0.1).abs() < 1.0e-15);
        assert!((model2_step_i_bc(20.0) - 0.01).abs() < 1.0e-15);
        // Monotone decreasing in SNR, and within (0, 1] for the
        // non-negative-SNR spec domain.
        assert!(model2_step_i_bc(5.5) > model2_step_i_bc(24.5));
        assert!(model2_step_i_bc(24.5) > 0.0);
    }

    #[test]
    fn model2_step_j_threshold_energy() {
        let nb = model2_step_j_nb(&[2.0, 3.0, 0.5], &[0.1, 0.01, 1.0]).unwrap();
        assert_eq!(nb, vec![0.2, 0.03, 0.5]);
        assert!(model2_step_j_nb(&[1.0; 3], &[1.0; 4]).is_none());
    }

    #[test]
    fn model2_step_k_spreads_and_conserves_energy() {
        let partitions = model2_partition_table(AnnexDSamplingRate::Hz32000);
        let nb: Vec<f64> = (0..partitions.len()).map(|b| 1.0 + b as f64).collect();
        let lines = model2_step_k_nb_lines(&nb, partitions).unwrap();
        // The 32 kHz table covers lines 1..=513 exactly.
        assert_eq!(lines.len(), 513);
        // Partition 1 (row index 0) is the single line 1: nb_ω = nb_b.
        assert_eq!(lines[0], nb[0]);
        // Partition 2 spans lines 2..=4 (3 lines): each gets nb_b/3.
        for w in 2..=4 {
            assert!((lines[w - 1] - nb[1] / 3.0).abs() < 1.0e-15, "line {w}");
        }
        // Energy conservation: the per-line values sum back to Σ nb_b
        // (coverage is contiguous and non-overlapping).
        let total: f64 = lines.iter().sum();
        let expected: f64 = nb.iter().sum();
        assert!((total - expected).abs() < 1.0e-9, "{total} vs {expected}");
        // Length mismatch / empty partitions.
        assert!(model2_step_k_nb_lines(&nb[..3], partitions).is_none());
        assert!(model2_step_k_nb_lines(&[], &[]).is_none());
    }

    #[test]
    fn model2_step_l_floors_at_absolute_threshold() {
        // Scalar max in both orders.
        assert_eq!(model2_step_l_thr(2.0, 1.0), 2.0);
        assert_eq!(model2_step_l_thr(1.0, 2.0), 2.0);
        // dB → energy conversion: with the table's 0-dB reference at
        // calibration level L, absthr_db = 0 maps to 10^(L/10).
        assert!((model2_absthr_energy(0.0, 0.0) - 1.0).abs() < 1.0e-15);
        assert!((model2_absthr_energy(10.0, 0.0) - 10.0).abs() < 1.0e-15);
        assert!((model2_absthr_energy(0.0, 20.0) - 100.0).abs() < 1.0e-12);
        // Elementwise line form; an uncovered line's absthr_ω = 0
        // convention passes nb_ω through.
        let thr = model2_step_l_thr_lines(&[1.0, 0.5, 3.0], &[2.0, 0.0, 1.0]).unwrap();
        assert_eq!(thr, vec![2.0, 0.5, 3.0]);
        assert!(model2_step_l_thr_lines(&[1.0; 2], &[1.0; 3]).is_none());
    }

    #[test]
    fn model2_step_n_epart_sums_squared_magnitudes() {
        // Uniform r_ω = 1: epart_n is the partition's inclusive line
        // count. Span 1 runs ωlow=1..=ωhigh=17 → 17 lines.
        let r = vec![1.0; 513];
        let span1 = coder_partition_d5_span(1).unwrap();
        assert_eq!(model2_step_n_epart(&r, span1).unwrap(), 17.0);
        // Non-uniform: squares, not magnitudes.
        let mut r2 = vec![0.0; 513];
        r2[0] = 3.0; // line 1
        r2[16] = 2.0; // line 17
        assert_eq!(model2_step_n_epart(&r2, span1).unwrap(), 13.0);
        // Slice too short to cover ωhigh.
        assert!(model2_step_n_epart(&r[..16], span1).is_none());
    }

    #[test]
    fn model2_step_n_npart_width_split() {
        // width = 0 (wide, spans 1..=12): smallest positive × count.
        let span1 = coder_partition_d5_span(1).unwrap();
        assert_eq!(span1.width, 0);
        let mut thr = vec![2.0; 513];
        thr[4] = 0.5; // line 5, inside span 1
        let npart = model2_step_n_npart(&thr, span1).unwrap();
        assert!((npart - 0.5 * 17.0).abs() < 1.0e-12);
        // The "smallest positive" qualifier skips a zero entry.
        thr[6] = 0.0; // line 7
        let npart = model2_step_n_npart(&thr, span1).unwrap();
        assert!((npart - 0.5 * 17.0).abs() < 1.0e-12);
        // No positive argument at all → documented 0 convention.
        let zeros = vec![0.0; 513];
        assert_eq!(model2_step_n_npart(&zeros, span1).unwrap(), 0.0);
        // width = 1 (narrow, spans 13..=32): plain sum.
        let span13 = coder_partition_d5_span(13).unwrap();
        assert_eq!(span13.width, 1);
        let thr2 = vec![0.25; 513];
        let lines = f64::from(span13.omega_high - span13.omega_low + 1);
        let npart = model2_step_n_npart(&thr2, span13).unwrap();
        assert!((npart - 0.25 * lines).abs() < 1.0e-12);
        // Slice too short.
        assert!(model2_step_n_npart(&thr2[..200], span13).is_none());
    }

    #[test]
    fn model2_step_n_smr_full_vector() {
        // SMR_n = 10 log10(epart/npart): equal energies → 0 dB; a
        // 10× signal-over-threshold → 10 dB.
        assert!((model2_step_n_smr_db(1.0, 1.0)).abs() < 1.0e-15);
        assert!((model2_step_n_smr_db(10.0, 1.0) - 10.0).abs() < 1.0e-12);
        // Full driver: r_ω = 1 (epart = line count) and thr_ω = 0,1
        // (npart = 0,1 × line count for both width readings, since
        // thr is uniform) → SMR_n = 10 dB in every partition.
        let r = vec![1.0; 513];
        let thr = vec![0.1; 513];
        let smr = model2_step_n_smr(&r, &thr).unwrap();
        assert_eq!(smr.len(), 32);
        for (n, &s) in smr.iter().enumerate() {
            assert!((s - 10.0).abs() < 1.0e-9, "n={}: {s}", n + 1);
        }
        // Either slice too short to reach ωhigh_32 = 513 → None.
        assert!(model2_step_n_smr(&r[..512], &thr).is_none());
        assert!(model2_step_n_smr(&r, &thr[..512]).is_none());
    }

    #[test]
    fn model2_steps_h_to_l_chain_to_line_thresholds() {
        // End-to-end h)→i)→j)→k)→l) over the full 32 kHz tables with
        // a uniform normalized energy en_b = 1 and tb_b = 0: the
        // chain must give nb_ω = 10^(-SNR_b/10) / lines_b on every
        // line of partition b, then floor at the (energy-domain)
        // absolute threshold.
        let fs = AnnexDSamplingRate::Hz32000;
        let partitions = model2_partition_table(fs);
        let nparts = partitions.len();
        let tb = vec![0.0; nparts];
        let snr = model2_step_h_snr(&tb, partitions).unwrap();
        // Every Table D.3 minval ≤ 20 dB and TMN ≥ 24,5 dB keep the
        // required SNR within sane dB bounds.
        assert!(snr.iter().all(|&s| (5.5..=20.0).contains(&s)));
        let bc: Vec<f64> = snr.iter().map(|&s| model2_step_i_bc(s)).collect();
        let en = vec![1.0; nparts];
        let nb = model2_step_j_nb(&en, &bc).unwrap();
        let nb_lines = model2_step_k_nb_lines(&nb, partitions).unwrap();
        assert_eq!(nb_lines.len(), 513);
        // Spot-check partition 1 (single line 1, minval 0, tb 0):
        // nb_1 = 10^(-0,55) on line 1.
        let expected = (10.0_f64).powf(-0.55);
        assert!((nb_lines[0] - expected).abs() < 1.0e-12);
        // Step l) with the Table D.4a dB prints converted at a 0 dB
        // calibration level; uncovered lines (58 and 481..=513) get
        // no floor.
        let absthr: Vec<f64> = (1..=513)
            .map(|w| model2_absthr_for_line(fs, w).map_or(0.0, |db| model2_absthr_energy(db, 0.0)))
            .collect();
        let thr = model2_step_l_thr_lines(&nb_lines, &absthr).unwrap();
        // Line 1's D.4a print is 58,23 dB → energy ≈ 10^5,823 — far
        // above nb_1, so the floor wins there.
        assert!((thr[0] - (10.0_f64).powf(5.823)).abs() < 1.0e-6 * thr[0]);
        // Every final threshold is ≥ both inputs.
        for w in 0..513 {
            assert!(
                thr[w] >= nb_lines[w] && thr[w] >= absthr[w],
                "line {}",
                w + 1
            );
        }
    }

    // ----- Tables D.3a–c / D.4a–c transcription (step 82 / r280) -----

    const ALL_RATES: [AnnexDSamplingRate; 3] = [
        AnnexDSamplingRate::Hz32000,
        AnnexDSamplingRate::Hz44100,
        AnnexDSamplingRate::Hz48000,
    ];

    #[test]
    fn table_d3_lengths_and_full_contiguous_line_coverage() {
        // Printed partition counts (the docs extracts file's "63
        // partitions at 32 kHz" prose is an erratum; the renders are
        // authoritative and end at 49 / 57 / 58).
        assert_eq!(MODEL2_PARTITION_D3A.len(), 49);
        assert_eq!(MODEL2_PARTITION_D3B.len(), 57);
        assert_eq!(MODEL2_PARTITION_D3C.len(), 58);
        for fs in ALL_RATES {
            let t = model2_partition_table(fs);
            // Exact half-spectrum coverage: 1..=513, contiguous.
            assert_eq!(t[0].wlow, 1, "{fs:?}");
            assert_eq!(t[t.len() - 1].whigh, 513, "{fs:?}");
            for (i, w) in t.windows(2).enumerate() {
                assert!(w[0].wlow <= w[0].whigh, "{fs:?} index {}", i + 1);
                assert_eq!(
                    w[1].wlow,
                    w[0].whigh + 1,
                    "{fs:?} index {} -> {}",
                    i + 1,
                    i + 2
                );
            }
        }
    }

    #[test]
    fn table_d3_columns_are_well_formed() {
        for fs in ALL_RATES {
            let t = model2_partition_table(fs);
            // bval starts at 0,00 and increases strictly.
            assert_eq!(t[0].bval, 0.00, "{fs:?}");
            for (i, w) in t.windows(2).enumerate() {
                assert!(w[0].bval < w[1].bval, "{fs:?} bval index {}", i + 1);
                // TMN is non-decreasing down the table.
                assert!(w[0].tmn_db <= w[1].tmn_db, "{fs:?} TMN index {}", i + 1);
            }
            // minval only takes the printed value set.
            for (i, e) in t.iter().enumerate() {
                assert!(
                    [0.0, 4.4, 4.5, 3.5, 7.0, 10.0, 15.0, 17.0, 20.0].contains(&e.minval_db),
                    "{fs:?} minval index {}: {}",
                    i + 1,
                    e.minval_db
                );
            }
            // TMN spans the printed range.
            assert_eq!(t[0].tmn_db, 24.5, "{fs:?}");
            assert!(t[t.len() - 1].tmn_db <= 40.3, "{fs:?}");
        }
    }

    #[test]
    fn table_d3_spot_rows_match_renders() {
        // D.3a anchors (also printed as text in the docs extracts
        // file, rows 1..=20): row 15 is the first non-24,5 TMN.
        let r15 = &MODEL2_PARTITION_D3A[14];
        assert_eq!((r15.wlow, r15.whigh), (42, 45));
        assert_eq!((r15.bval, r15.minval_db, r15.tmn_db), (10.28, 4.4, 24.8));
        // Last rows of each table.
        let a49 = &MODEL2_PARTITION_D3A[48];
        assert_eq!((a49.wlow, a49.whigh, a49.bval), (497, 513, 24.07));
        assert_eq!((a49.minval_db, a49.tmn_db), (4.5, 38.6));
        let b57 = &MODEL2_PARTITION_D3B[56];
        assert_eq!((b57.wlow, b57.whigh, b57.bval), (470, 513, 25.33));
        assert_eq!((b57.minval_db, b57.tmn_db), (3.5, 39.8));
        let c58 = &MODEL2_PARTITION_D3C[57];
        assert_eq!((c58.wlow, c58.whigh, c58.bval), (508, 513, 25.81));
        assert_eq!((c58.minval_db, c58.tmn_db), (3.5, 40.3));
        // Single-line head regions: D.3b lines 1..=16 / D.3c 1..=17
        // are one-line partitions; D.3a only line 1.
        assert!(MODEL2_PARTITION_D3B[..16].iter().all(|e| e.wlow == e.whigh));
        assert_eq!(MODEL2_PARTITION_D3B[16].whigh, 19);
        assert!(MODEL2_PARTITION_D3C[..17].iter().all(|e| e.wlow == e.whigh));
        assert_eq!(MODEL2_PARTITION_D3C[17].whigh, 20);
    }

    #[test]
    fn table_d3_bval_consistent_with_d1_bark_column() {
        // Redundancy check: a partition's `bval` tracks the Bark
        // range its FFT lines span per the Layer II Tables D.1 (same
        // 1024-point FFT line grid). The two tables use different
        // Bark conventions at the band edges (D.3 prints `bval =
        // 0,00` for partition 1 where D.1 prints the line-center
        // Bark, 0,309 at 32 kHz), so the guard allows half a Bark of
        // slack — still tight enough to catch a digit-level
        // transcription error in any `bval` cell.
        use crate::frame::Layer;
        for fs in ALL_RATES {
            let d1 = model1_threshold_table(Layer::LayerII, fs).unwrap();
            let last_line = model1_d1_line_for_index(Layer::LayerII, fs, d1.len() as u16).unwrap();
            for (i, e) in model2_partition_table(fs).iter().enumerate() {
                if e.whigh > last_line {
                    continue; // partition extends past the D.1 grid
                }
                let z_lo = model1_d1_entry_for_line(Layer::LayerII, fs, e.wlow)
                    .unwrap()
                    .z_bark;
                let z_hi = model1_d1_entry_for_line(Layer::LayerII, fs, e.whigh)
                    .unwrap()
                    .z_bark;
                assert!(
                    z_lo - 0.5 <= e.bval && e.bval <= z_hi + 0.5,
                    "{fs:?} partition {}: bval {} outside [{z_lo}, {z_hi}] ± 0,5",
                    i + 1,
                    e.bval
                );
            }
        }
    }

    #[test]
    fn model2_bval_extracts_partition_bval_column() {
        for fs in ALL_RATES {
            let bval = model2_bval(fs);
            let t = model2_partition_table(fs);
            assert_eq!(bval.len(), t.len());
            assert!(bval.iter().zip(t.iter()).all(|(&b, e)| b == e.bval));
            // And it feeds the step-f) reductions directly.
            assert!(model2_step_f_spread(&vec![1.0; bval.len()], &bval).is_some());
        }
    }

    #[test]
    fn model2_partition_index_for_line_covers_and_bounds() {
        for fs in ALL_RATES {
            let t = model2_partition_table(fs);
            assert_eq!(model2_partition_index_for_line(fs, 0), None, "{fs:?}");
            assert_eq!(model2_partition_index_for_line(fs, 514), None, "{fs:?}");
            assert_eq!(model2_partition_index_for_line(fs, 1), Some(1), "{fs:?}");
            assert_eq!(
                model2_partition_index_for_line(fs, 513),
                Some(t.len() as u16),
                "{fs:?}"
            );
            // Every line maps to the partition whose [wlow, whigh]
            // contains it.
            for (i, e) in t.iter().enumerate() {
                for line in [e.wlow, e.whigh] {
                    assert_eq!(
                        model2_partition_index_for_line(fs, line),
                        Some(i as u16 + 1),
                        "{fs:?} line {line}"
                    );
                }
            }
        }
    }

    #[test]
    fn table_d4_lengths_and_coverage_with_printed_quirks() {
        assert_eq!(MODEL2_ABSTHR_D4A.len(), 132);
        assert_eq!(MODEL2_ABSTHR_D4B.len(), 130);
        assert_eq!(MODEL2_ABSTHR_D4C.len(), 126);
        // Last covered line per rate (the tables stop short of 513).
        for (fs, last) in [
            (AnnexDSamplingRate::Hz32000, 480),
            (AnnexDSamplingRate::Hz44100, 464),
            (AnnexDSamplingRate::Hz48000, 428),
        ] {
            let t = model2_absthr_table(fs);
            assert_eq!(t[0].lower, 1, "{fs:?}");
            assert_eq!(t[t.len() - 1].higher, last, "{fs:?}");
            for (i, w) in t.windows(2).enumerate() {
                assert!(w[0].lower <= w[0].higher, "{fs:?} row {}", i + 1);
                // Contiguous coverage — except the printed D.4a
                // `57 | 57` → `59 | 60` gap (line 58 uncovered).
                if fs == AnnexDSamplingRate::Hz32000 && w[0].higher == 57 && w[0].lower == 57 {
                    assert_eq!(w[1].lower, 59, "the printed line-58 gap");
                    continue;
                }
                assert_eq!(
                    w[1].lower,
                    w[0].higher + 1,
                    "{fs:?} row {} -> {}",
                    i + 1,
                    i + 2
                );
            }
        }
        // The D.4c reduced 4-line group inside the 8-line tail.
        let quirk = MODEL2_ABSTHR_D4C
            .iter()
            .find(|e| e.lower == 329)
            .expect("printed row 329");
        assert_eq!((quirk.higher, quirk.absthr_db), (332, 61.94));
    }

    #[test]
    fn model2_absthr_for_line_lookup_and_printed_gap() {
        for fs in ALL_RATES {
            assert_eq!(model2_absthr_for_line(fs, 0), None, "{fs:?}");
            assert_eq!(model2_absthr_for_line(fs, 481), None, "32k past end");
        }
        assert_eq!(
            model2_absthr_for_line(AnnexDSamplingRate::Hz32000, 1),
            Some(58.23)
        );
        assert_eq!(
            model2_absthr_for_line(AnnexDSamplingRate::Hz44100, 1),
            Some(45.05)
        );
        assert_eq!(
            model2_absthr_for_line(AnnexDSamplingRate::Hz48000, 1),
            Some(42.10)
        );
        // Range lookups land on the covering row.
        assert_eq!(
            model2_absthr_for_line(AnnexDSamplingRate::Hz32000, 99),
            Some(-4.82)
        );
        assert_eq!(
            model2_absthr_for_line(AnnexDSamplingRate::Hz44100, 464),
            Some(69.13)
        );
        assert_eq!(
            model2_absthr_for_line(AnnexDSamplingRate::Hz48000, 330),
            Some(61.94)
        );
        // The printed D.4a gap: line 58 has no row at 32 kHz (the
        // `57 | 57` row's 0,55 dB matches Table D.1d's LTq at line
        // 58 — see the next test — so `higher = 57` is almost
        // certainly a misprint, but the verbatim print rules).
        assert_eq!(
            model2_absthr_for_line(AnnexDSamplingRate::Hz32000, 58),
            None
        );
        assert_eq!(
            model2_absthr_for_line(AnnexDSamplingRate::Hz32000, 57),
            Some(0.55)
        );
        assert_eq!(
            model2_absthr_for_line(AnnexDSamplingRate::Hz44100, 58),
            Some(-2.63)
        );
    }

    #[test]
    fn table_d4_agrees_with_layer2_d1_ltq_on_shared_lines() {
        // Redundancy check: the Layer II Tables D.1 tabulate the
        // threshold in quiet on the same 1024-point FFT line grid.
        // Wherever a D.4 row's `higher` line is D.1-tabulated, the
        // printed values agree — exceptions below are printed-spec
        // rounding inconsistencies, not transcription errors.
        use crate::frame::Layer;
        let mut mismatches = Vec::new();
        for fs in ALL_RATES {
            let d1 = model1_threshold_table(Layer::LayerII, fs).unwrap();
            let mut ltq_by_line = std::collections::HashMap::new();
            for i in 1..=d1.len() as u16 {
                let line = model1_d1_line_for_index(Layer::LayerII, fs, i).unwrap();
                ltq_by_line.insert(line, d1[i as usize - 1].ltq_db);
            }
            for e in model2_absthr_table(fs) {
                let Some(&ltq) = ltq_by_line.get(&e.higher) else {
                    continue;
                };
                if e.absthr_db != ltq {
                    mismatches.push((fs, e.higher, e.absthr_db, ltq));
                }
            }
            // The D.4a `57 | 57` quirk row: its absthr equals the
            // D.1d LTq at line *58* (the line the printed pair
            // skips).
            if fs == AnnexDSamplingRate::Hz32000 {
                assert_eq!(ltq_by_line[&58], 0.55);
            }
        }
        // Pinned printed-spec inconsistencies (sample cells of both
        // sides re-verified at 300–400 % zoom on the renders; each
        // side is legible). At 32 / 48 kHz the two tables agree on
        // every shared line except D.4a's last row (51,03 vs 51,04
        // at 15 000 Hz). At 44,1 kHz the printed D.4b systematically
        // disagrees with the printed D.1e in two ways — evidently
        // the 44,1 kHz absolute-threshold table was generated /
        // rounded separately when the spec was typeset:
        //   - 14 shared lines print exactly 0,01 dB lower in D.4b
        //     (e.g. lines 51..=52: -1,37 vs D.1e row 50's -1,38 at
        //     2 239,45 Hz);
        //   - the top-of-band saturation plateau prints 69,13 dB in
        //     D.4b (lines 369..=464) where D.1e clamps at 68,00 dB
        //     (and D.4c/D.1f both use 68,00).
        let mut expected = vec![
            (AnnexDSamplingRate::Hz32000, 480, 51.03, 51.04),
            (AnnexDSamplingRate::Hz44100, 52, -1.37, -1.38),
            (AnnexDSamplingRate::Hz44100, 104, -1.33, -1.32),
            (AnnexDSamplingRate::Hz44100, 148, 2.45, 2.46),
            (AnnexDSamplingRate::Hz44100, 256, 15.30, 15.31),
            (AnnexDSamplingRate::Hz44100, 272, 19.33, 19.34),
            (AnnexDSamplingRate::Hz44100, 312, 33.04, 33.05),
            (AnnexDSamplingRate::Hz44100, 320, 36.51, 36.52),
            (AnnexDSamplingRate::Hz44100, 328, 40.24, 40.25),
            (AnnexDSamplingRate::Hz44100, 336, 44.26, 44.27),
            (AnnexDSamplingRate::Hz44100, 344, 48.58, 48.59),
            (AnnexDSamplingRate::Hz44100, 352, 53.21, 53.22),
            (AnnexDSamplingRate::Hz44100, 360, 58.17, 58.18),
            (AnnexDSamplingRate::Hz44100, 368, 63.48, 63.49),
        ];
        expected.extend(
            (376..=464)
                .step_by(8)
                .map(|l| (AnnexDSamplingRate::Hz44100, l, 69.13, 68.00)),
        );
        assert_eq!(mismatches, expected);
    }

    // ----- §D.2.1 + §D.2.4 steps a)–e) front half (Phase 2 step 84) -----

    #[test]
    fn model2_iblen_range_bounds_are_strict() {
        // §D.2.1 a) verbatim "384<iblen<640": both bounds excluded.
        assert!(!model2_iblen_in_range(384));
        assert!(model2_iblen_in_range(385));
        assert!(model2_iblen_in_range(576)); // the Layer III granule shift
        assert!(model2_iblen_in_range(639));
        assert!(!model2_iblen_in_range(640));
        assert!(!model2_iblen_in_range(0));
    }

    #[test]
    fn model2_step_a_reconstructs_consecutive_samples() {
        let prev: Vec<f64> = (0..1024).map(f64::from).collect();
        let new: Vec<f64> = (0..576).map(|i| f64::from(2000 + i)).collect();
        let out = model2_step_a_reconstruct(&prev, &new).unwrap();
        assert_eq!(out.len(), MODEL2_FFT_LEN);
        // The most recent 1024-576 = 448 samples of the previous
        // window come first…
        assert_eq!(out[..448], prev[576..]);
        // …then the iblen new samples.
        assert_eq!(out[448..], new[..]);
    }

    #[test]
    fn model2_step_a_rejects_out_of_domain_lengths() {
        let prev = vec![0.0; 1024];
        assert!(model2_step_a_reconstruct(&prev, &[]).is_none());
        assert!(model2_step_a_reconstruct(&prev, &vec![0.0; 1025]).is_none());
        assert!(model2_step_a_reconstruct(&vec![0.0; 1023], &vec![0.0; 576]).is_none());
        // Full-window replacement (iblen = 1024) is within the
        // primitive's domain even though it is outside the §D.2.1
        // standard-table range.
        assert!(model2_step_a_reconstruct(&prev, &vec![1.0; 1024]).is_some());
    }

    #[test]
    fn model2_hann_window_domain_symmetry_and_values() {
        assert!(model2_hann_window(0).is_none());
        assert!(model2_hann_window(1025).is_none());
        // Half-sample-offset symmetry: w(i) = w(1025 - i).
        for i in [1usize, 2, 100, 512] {
            let a = model2_hann_window(i).unwrap();
            let b = model2_hann_window(1025 - i).unwrap();
            assert!(
                (a - b).abs() < 1.0e-15,
                "w({i}) = {a} vs w({}) = {b}",
                1025 - i
            );
        }
        // No zero endpoint (the (i - 0,5) offset keeps w(1) > 0) and
        // no sample reaches the raised cosine's supremum of 1.
        let w1 = model2_hann_window(1).unwrap();
        assert!(w1 > 0.0 && w1 < 1.0e-4);
        let wmax = (1..=1024)
            .map(|i| model2_hann_window(i).unwrap())
            .fold(0.0, f64::max);
        assert!(wmax < 1.0 && wmax > 0.999_99);
        // Spot value: w(1) = 0,5 - 0,5·cos(2π·0,5/1024).
        let expect = 0.5 - 0.5 * (core::f64::consts::PI / 1024.0).cos();
        assert!((w1 - expect).abs() < 1.0e-15);
        // Unlike the Model 1 Step 1 window there is no sqrt(8/3)
        // power prefactor: Σ w(i)² is strictly below N (= what the
        // unit-power Model 1 window sums to).
        let power: f64 = (1..=1024)
            .map(|i| model2_hann_window(i).unwrap().powi(2))
            .sum();
        assert!((power - 384.0).abs() < 1.0e-9); // N·3/8 for the bare raised cosine
    }

    #[test]
    fn model2_step_b_rejects_wrong_lengths() {
        assert!(model2_step_b_spectrum(&[]).is_none());
        assert!(model2_step_b_spectrum(&vec![0.0; 512]).is_none());
        assert!(model2_step_b_spectrum(&vec![0.0; 1025]).is_none());
    }

    #[test]
    fn model2_step_b_dc_block_concentrates_at_line_1() {
        // A constant block transforms to the window's own sum at the
        // DC line (ω = 1): Σ w(i) = N/2 = 512 exactly for the bare
        // raised cosine (the cos terms cancel by half-sample
        // symmetry).
        let polar = model2_step_b_spectrum(&vec![1.0; 1024]).unwrap();
        assert_eq!(polar.r.len(), MODEL2_FFT_LINES);
        assert_eq!(polar.f.len(), MODEL2_FFT_LINES);
        assert!((polar.r[0] - 512.0).abs() < 1.0e-9, "r_1 = {}", polar.r[0]);
        // The Hann window's spectral leakage is confined to ±1 line:
        // from line 4 on the magnitude is negligible next to DC.
        for (idx, &r) in polar.r.iter().enumerate().skip(3) {
            assert!(r < 1.0e-9 * 512.0, "line {}: r = {r}", idx + 1);
        }
    }

    #[test]
    fn model2_step_b_bin_exact_sine_peaks_at_its_line() {
        // s_i = cos(2π·64·(i-1)/1024) sits exactly on FFT bin 64
        // (line ω = 65); the windowed magnitude there is N/4 = 256
        // (half the DC response, the cosine splitting its energy
        // between ±64).
        let s: Vec<f64> = (0..1024)
            .map(|n| (2.0 * core::f64::consts::PI * 64.0 * n as f64 / 1024.0).cos())
            .collect();
        let polar = model2_step_b_spectrum(&s).unwrap();
        assert!(
            (polar.r[64] - 256.0).abs() < 1.0e-8,
            "r_65 = {}",
            polar.r[64]
        );
        // Hann leakage: ±1 lines carry half the peak; beyond ±2 lines
        // nothing.
        assert!((polar.r[63] - 128.0).abs() < 1.0e-8);
        assert!((polar.r[65] - 128.0).abs() < 1.0e-8);
        assert!(polar.r[60] < 1.0e-6);
        assert!(polar.r[70] < 1.0e-6);
    }

    #[test]
    fn model2_step_c_prediction_is_linear_extrapolation() {
        assert_eq!(model2_step_c_predict(3.0, 1.0), 5.0);
        assert_eq!(model2_step_c_predict(0.0, 0.0), 0.0);
        let prev = Model2Polar {
            r: vec![3.0, 2.0],
            f: vec![0.5, -0.25],
        };
        let prev2 = Model2Polar {
            r: vec![1.0, 2.0],
            f: vec![0.25, -0.5],
        };
        let p = model2_step_c_predict_polar(&prev, &prev2).unwrap();
        assert_eq!(p.r, vec![5.0, 2.0]);
        assert_eq!(p.f, vec![0.75, 0.0]);
        // Length mismatches are rejected.
        assert!(model2_step_c_predict_polar(&prev, &Model2Polar::zeroed()).is_none());
    }

    #[test]
    fn model2_step_d_cw_endpoints() {
        // Perfect prediction → 0.
        assert_eq!(model2_step_d_cw(2.0, 0.7, 2.0, 0.7), 0.0);
        // Zero-magnitude prediction of a live line → 1.
        assert!((model2_step_d_cw(5.0, 1.2, 0.0, 0.4) - 1.0).abs() < 1.0e-15);
        // Opposite-phase, equal-magnitude prediction → 1 (the
        // measure's maximum).
        let c = model2_step_d_cw(1.0, 0.0, 1.0, core::f64::consts::PI);
        assert!((c - 1.0).abs() < 1.0e-15);
        // All-silent 0/0 convention → 0.
        assert_eq!(model2_step_d_cw(0.0, 0.0, 0.0, 0.0), 0.0);
    }

    #[test]
    fn model2_step_d_cw_lines_partial_limit_sets_0_3() {
        let cur = Model2Polar {
            r: vec![1.0; 8],
            f: vec![0.0; 8],
        };
        let pred = Model2Polar {
            r: vec![1.0; 8],
            f: vec![0.0; 8],
        };
        let full = model2_step_d_cw_lines(&cur, &pred, None).unwrap();
        assert_eq!(full, vec![0.0; 8]);
        let partial = model2_step_d_cw_lines(&cur, &pred, Some(3)).unwrap();
        assert_eq!(&partial[..3], &[0.0; 3]);
        assert_eq!(&partial[3..], &[MODEL2_CW_ABOVE_LIMIT; 5]);
        // A limit at or past the line count is the full calculation.
        assert_eq!(model2_step_d_cw_lines(&cur, &pred, Some(8)).unwrap(), full);
        // Mismatched spectra are rejected.
        assert!(model2_step_d_cw_lines(&cur, &Model2Polar::zeroed(), None).is_none());
    }

    #[test]
    fn model2_step_e_partitions_conserve_energy() {
        // The Table D.3 partitions tile lines 1..=513, so the e_b sum
        // over partitions equals the total line energy — at every
        // sampling rate.
        let r: Vec<f64> = (0..MODEL2_FFT_LINES)
            .map(|i| ((i * 37 + 11) % 101) as f64 / 100.0)
            .collect();
        let total: f64 = r.iter().map(|&x| x * x).sum();
        for fs in ALL_RATES {
            let parts = model2_partition_table(fs);
            let eb = model2_step_e_eb(&r, parts).unwrap();
            assert_eq!(eb.len(), parts.len());
            let sum: f64 = eb.iter().sum();
            assert!(
                (sum - total).abs() < 1.0e-9 * total,
                "{fs:?}: {sum} vs {total}"
            );
            // With c_ω ≡ 1 the weighted unpredictability c_b equals e_b.
            let cb = model2_step_e_cb(&r, &vec![1.0; r.len()], parts).unwrap();
            for (a, b) in cb.iter().zip(eb.iter()) {
                assert!((a - b).abs() < 1.0e-12);
            }
        }
    }

    #[test]
    fn model2_step_e_rejects_short_or_mismatched_slices() {
        let parts = model2_partition_table(AnnexDSamplingRate::Hz32000);
        assert!(model2_step_e_eb(&vec![0.0; 512], parts).is_none());
        assert!(model2_step_e_eb(&[], parts).is_none());
        assert!(model2_step_e_eb(&vec![0.0; 513], &[]).is_none());
        assert!(model2_step_e_cb(&vec![0.0; 513], &vec![0.0; 512], parts).is_none());
        assert!(model2_step_e_cb(&vec![0.0; 512], &vec![0.0; 512], parts).is_none());
    }

    #[test]
    fn model2_state_walk_matches_manual_primitive_chain() {
        // The full Model2State walk reproduces, value for value, a
        // by-hand chain of the step primitives — pinning the bridge
        // between the r282 front half and the r279–r281 back half.
        let fs = AnnexDSamplingRate::Hz44100;
        let iblen = 576usize;
        let half_lsb_db = -20.0;
        let block: Vec<f64> = (0..iblen)
            .map(|n| (2.0 * core::f64::consts::PI * n as f64 / 32.0).sin() * 0.25)
            .collect();

        let mut state = Model2State::new();
        // Warm up two calls so t-1 / t-2 hold real spectra.
        state.smr(&block, fs, half_lsb_db, None).unwrap();
        state.smr(&block, fs, half_lsb_db, None).unwrap();

        // Manual replay of call 3 from a mirror of the state's inputs.
        let mut window = vec![0.0; MODEL2_FFT_LEN];
        let (mut r1, mut f1) = (vec![0.0; 513], vec![0.0; 513]);
        let (mut r2, mut f2) = (vec![0.0; 513], vec![0.0; 513]);
        for _ in 0..2 {
            window = model2_step_a_reconstruct(&window, &block).unwrap();
            let p = model2_step_b_spectrum(&window).unwrap();
            (r2, f2) = (r1, f1);
            (r1, f1) = (p.r, p.f);
        }
        let window3 = model2_step_a_reconstruct(&window, &block).unwrap();
        let polar = model2_step_b_spectrum(&window3).unwrap();
        let predicted = model2_step_c_predict_polar(
            &Model2Polar { r: r1, f: f1 },
            &Model2Polar { r: r2, f: f2 },
        )
        .unwrap();
        let cw = model2_step_d_cw_lines(&polar, &predicted, None).unwrap();
        let parts = model2_partition_table(fs);
        let eb = model2_step_e_eb(&polar.r, parts).unwrap();
        let cb_raw = model2_step_e_cb(&polar.r, &cw, parts).unwrap();
        let bval = model2_bval(fs);
        let ecb = model2_step_f_spread(&eb, &bval).unwrap();
        let ct = model2_step_f_spread(&cb_raw, &bval).unwrap();
        let cb = model2_step_f_cb(&ct, &ecb).unwrap();
        let rnorm = model2_step_f_rnorm(&bval);
        let en = model2_step_f_en(&ecb, &rnorm).unwrap();
        let tb: Vec<f64> = cb.iter().map(|&c| model2_step_g_tonality(c)).collect();
        let snr = model2_step_h_snr(&tb, parts).unwrap();
        let bc: Vec<f64> = snr.iter().map(|&s| model2_step_i_bc(s)).collect();
        let nb = model2_step_j_nb(&en, &bc).unwrap();
        let nb_lines = model2_step_k_nb_lines(&nb, parts).unwrap();
        let absthr: Vec<f64> = (1..=nb_lines.len() as u16)
            .map(|line| {
                model2_absthr_for_line(fs, line)
                    .map_or(0.0, |db| model2_absthr_energy(db, half_lsb_db))
            })
            .collect();
        let thr = model2_step_l_thr_lines(&nb_lines, &absthr).unwrap();
        let expect = model2_step_n_smr(&polar.r, &thr).unwrap();

        let got = state.smr(&block, fs, half_lsb_db, None).unwrap();
        assert_eq!(got, expect);
    }

    #[test]
    fn model2_state_steady_sine_is_tonal_and_smr_positive_at_the_tone() {
        // A bin-exact sinusoid whose period divides the shift length
        // produces identical spectra every block: from the third call
        // the step c) prediction is exact, c_ω ≈ 0 at the live lines,
        // tonality ≈ 1, and the SMR at the tone's coder partition is
        // strongly positive (signal well above its masking
        // threshold). iblen = 512 keeps every 1 024-window identical;
        // bin 64 (line 65, period 16 samples) divides 512.
        let fs = AnnexDSamplingRate::Hz44100;
        let mut state = Model2State::new();
        let mut smr = Vec::new();
        for call in 0..4 {
            let base = call * 512;
            let block: Vec<f64> = (0..512)
                .map(|n| {
                    (2.0 * core::f64::consts::PI * 64.0 * (base + n) as f64 / 1024.0).cos() * 1000.0
                })
                .collect();
            smr = state.smr(&block, fs, -96.0, None).unwrap();
            assert_eq!(smr.len(), 32);
        }
        let tone_partition = first_partition_containing_line(65).unwrap();
        let tone_smr = smr[tone_partition as usize - 1];
        assert!(
            tone_smr.is_finite() && tone_smr > 10.0,
            "SMR at partition {tone_partition} = {tone_smr}"
        );
    }

    #[test]
    fn model2_state_rejects_bad_blocks_and_stays_usable() {
        let fs = AnnexDSamplingRate::Hz32000;
        let mut state = Model2State::new();
        let before = state.clone();
        assert!(state.smr(&[], fs, 0.0, None).is_none());
        assert!(state.smr(&vec![0.0; 1025], fs, 0.0, None).is_none());
        // A failed call leaves the state untouched…
        assert_eq!(state, before);
        // …and a silent valid block still walks the whole chain
        // (SMR_n is outside the spec's positive-energy domain there;
        // the walk itself must not fail). A silent block over a
        // zeroed state advances to a bit-identical zeroed state —
        // the §D.2.1 known starting point is a fixed point of
        // silence.
        let out = state.smr(&vec![0.0; 576], fs, 0.0, None).unwrap();
        assert_eq!(out.len(), 32);
        assert_eq!(state, before);
        // A live block does perturb the state.
        let out = state.smr(&vec![0.5; 576], fs, 0.0, None).unwrap();
        assert_eq!(out.len(), 32);
        assert_ne!(state, before);
    }
}
