//! Layer III **requantization** — the §2.4.3.4.7 stage that converts the
//! 576 quantized integer frequency lines `is[576]` (from the Huffman
//! decode) into 576 float frequency lines `xr[576]` for one
//! granule-channel.
//!
//! The non-uniform quantizer of Layer III uses a power law: for each
//! Huffman-decoded value `is_i`, `|is_i|^(4/3)` is computed and then
//! scaled by the global gain, the per-subblock gain (short blocks only),
//! and the per-scalefactor-band scalefactors (with the optional preflag
//! high-frequency-amplification table). Every numeric constant in this
//! file was transcribed by hand from ISO/IEC 11172-3:1993 §2.4.3.4.7.1
//! (the requantization formula, PDF p.34–35), Table B.6 (the preemphasis
//! `pretab`, PDF p.53), and Table B.8 (the scalefactor-band start
//! indices, PDF p.62–64).
//!
//! # The two formulas (§2.4.3.4.7.1)
//!
//! For **long blocks** the input to the synthesis filterbank at index
//! `i` is
//!
//! ```text
//! xr_i = sign(is_i) * |is_i|^(4/3)
//!      * 2^( (1/4) * (global_gain[gr] - 210) )
//!      * 2^( -(scalefac_multiplier
//!              * (scalefac_l[gr][ch][sfb] + preflag[gr] * pretab[sfb])) )
//! ```
//!
//! For **short blocks** (`block_type == 2`) the per-window
//! `subblock_gain` offset enters the global-gain term and the
//! scalefactor is the per-window `scalefac_s`:
//!
//! ```text
//! xr_i = sign(is_i) * |is_i|^(4/3)
//!      * 2^( (1/4) * (global_gain[gr] - 210 - 8*subblock_gain[window][gr]) )
//!      * 2^( -(scalefac_multiplier * scalefac_s[gr][ch][sfb][window]) )
//! ```
//!
//! The constant `210` is a system constant that scales the decoder
//! output into the PCM range `[-1.0, +1.0]` (§2.4.3.4.7.1). The
//! `scalefac_multiplier` is `0.5` when `scalefac_scale == 0` and `1.0`
//! when `scalefac_scale == 1` (§2.4.2.7, "scalefac_scale" table). The
//! preflag amplification is never applied to short blocks (§2.4.2.7,
//! "preflag").
//!
//! # Line ordering
//!
//! The requantizer operates on `is[]` in its native Huffman/bitstream
//! order, i.e. `xr[scf_band][window][freqline]` for short blocks
//! (§2.4.3.4.8 states the reorder into subband order happens *after*
//! requantization, so this stage leaves short-block lines interleaved
//! by `(sfb, window, freqline)`). Each line index is mapped to its
//! `(sfb, window)` via the Table B.8 band boundaries to look up the
//! right scalefactor.
//!
//! The §2.4.3.4.9 stereo processing (MS / intensity) and the
//! §2.4.3.4.8 short-block reorder are later stages and are out of scope
//! here.

use crate::frame::MpegVersion;
use crate::scalefactors::ScaleFactors;
use crate::side_info::{BlockType, GranuleChannel};

/// Number of frequency lines produced per granule-channel.
pub const NUM_LINES: usize = 576;

/// The system constant subtracted from `global_gain` in the
/// requantization exponent (§2.4.3.4.7.1: "The constant 210 in this
/// formula is needed to scale the output appropriately.").
const GAIN_BIAS: i32 = 210;

/// Table B.6 — Layer III preemphasis (`pretab[sfb]`) for the 21
/// long-block scalefactor bands (ISO/IEC 11172-3:1993 p.53). Added to
/// the long-block scalefactors when `preflag` is set.
pub const PRETAB: [u8; 21] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 2,
];

/// The `scalefac_multiplier` selected by `scalefac_scale` (§2.4.2.7):
/// `0.5` for `scalefac_scale == 0`, `1.0` for `scalefac_scale == 1`.
#[must_use]
pub fn scalefac_multiplier(scalefac_scale: bool) -> f32 {
    if scalefac_scale {
        1.0
    } else {
        0.5
    }
}

/// Long-block scalefactor-band *start* indices for the active sampling
/// rate (Table B.8, "index of start" column). Entry `i` is the first
/// line of band `i`; entry 21 is one past the last line of band 20 so a
/// band's line range is `starts[sfb]..starts[sfb + 1]`.
///
/// Shared with the §2.4.3.4.9 [`crate::stereo`] stage, which maps the
/// same long-block band layout when applying per-band MS / intensity
/// processing.
pub(crate) fn long_band_starts(sample_rate_hz: u32, version: MpegVersion) -> &'static [usize; 22] {
    // MPEG-1 rates: ISO/IEC 11172-3 Table B.8a/b/c. MPEG-2 LSF rates
    // (16 / 22.05 / 24 kHz): ISO/IEC 13818-3:1997 Table B.2 ("Layer III
    // scalefactor bands", long blocks).
    //
    // MPEG-2.5 rates (8 / 11.025 / 12 kHz) — `docs/audio/mp3/`
    // `mpeg2.5-scalefactor-bands.md` (#147/#151): the 11.025 / 12 kHz
    // tables are *byte-identical* to the 13818-3 22.05 / 24 kHz LSF long
    // tables (the half-rate sibling reuse, fully grounded in the in-repo
    // 13818-3 PDF), so they share the LSF constants below. 8 kHz is a
    // distinct Fraunhofer-defined table whose top long bands collapse to
    // width 2 (`LONG_STARTS_MPEG25_8`).
    let _ = version;
    match sample_rate_hz {
        16000 | 22050 | 11025 => &LONG_STARTS_LSF_16_22,
        24000 | 12000 => &LONG_STARTS_LSF_24,
        32000 => &LONG_STARTS_32,
        48000 => &LONG_STARTS_48,
        8000 => &LONG_STARTS_MPEG25_8,
        _ => &LONG_STARTS_44, // 44100 and any default.
    }
}

/// Short-block scalefactor-band *start* indices (per window) for the
/// active sampling rate (Table B.8, short-block "index of start"). Entry
/// `i` is the first per-window line of band `i`; entry 12 is one past
/// band 11 so a band's per-window width is `starts[sfb + 1] - starts[sfb]`.
///
/// Shared with the §2.4.3.4.8 [`crate::reorder`] stage, which maps the
/// same native band layout into subband order.
pub(crate) fn short_band_starts(sample_rate_hz: u32, version: MpegVersion) -> &'static [usize; 13] {
    // Same per-rate provenance split as [`long_band_starts`]: MPEG-1
    // rates from 11172-3 Table B.8, LSF rates from 13818-3 Table B.2
    // (short blocks). MPEG-2.5 (`mpeg2.5-scalefactor-bands.md`): the
    // 11.025 / 12 kHz short tables are byte-identical to the 22.05 / 24
    // kHz LSF short tables; 8 kHz is the distinct Fraunhofer short table
    // (`SHORT_STARTS_MPEG25_8`).
    let _ = version;
    match sample_rate_hz {
        16000 => &SHORT_STARTS_LSF_16,
        22050 | 11025 => &SHORT_STARTS_LSF_22,
        24000 | 12000 => &SHORT_STARTS_LSF_24,
        32000 => &SHORT_STARTS_32,
        48000 => &SHORT_STARTS_48,
        8000 => &SHORT_STARTS_MPEG25_8,
        _ => &SHORT_STARTS_44,
    }
}

/// Table B.8a (32 kHz) long-block band start indices + end+1.
const LONG_STARTS_32: [usize; 22] = [
    0, 4, 8, 12, 16, 20, 24, 30, 36, 44, 54, 66, 82, 102, 126, 156, 194, 240, 296, 364, 448, 550,
];
/// Table B.8b (44.1 kHz) long-block band start indices + end+1.
const LONG_STARTS_44: [usize; 22] = [
    0, 4, 8, 12, 16, 20, 24, 30, 36, 44, 52, 62, 74, 90, 110, 134, 162, 196, 238, 288, 342, 418,
];
/// Table B.8c (48 kHz) long-block band start indices + end+1.
const LONG_STARTS_48: [usize; 22] = [
    0, 4, 8, 12, 16, 20, 24, 30, 36, 42, 50, 60, 72, 88, 106, 128, 156, 190, 230, 276, 330, 384,
];

/// ISO/IEC 13818-3:1997 Table B.2 — 16 kHz and 22,05 kHz long-block
/// band start indices + end+1. The spec prints the two rates' long
/// tables with identical "width of band" columns (6×6, 8, 10, 12, 14,
/// 16, 20, 24, 28, 32, 38, 46, 52, 60, 68, 58, 54), so they share one
/// constant; band 21 spans `starts[21]..576`.
const LONG_STARTS_LSF_16_22: [usize; 22] = [
    0, 6, 12, 18, 24, 30, 36, 44, 54, 66, 80, 96, 116, 140, 168, 200, 238, 284, 336, 396, 464, 522,
];
/// ISO/IEC 13818-3:1997 Table B.2 — 24 kHz long-block band start
/// indices + end+1 (widths 6×6, 8, 10, 12, 14, 16, 18, 22, 26, 32, 38,
/// 46, 54, 62, 70, 76, 36); band 21 spans `starts[21]..576`.
const LONG_STARTS_LSF_24: [usize; 22] = [
    0, 6, 12, 18, 24, 30, 36, 44, 54, 66, 80, 96, 114, 136, 162, 194, 232, 278, 332, 394, 464, 540,
];

/// MPEG-2.5 8 kHz long-block band start indices + end+1
/// (`docs/audio/mp3/mpeg2.5-scalefactor-bands.md`, "8 kHz, long blocks").
/// Widths 12×6, 16, 20, 24, 28, 32, 40, 48, 56, 64, 76, 90, then five
/// width-2 filler bands (17–21): at 8 kHz the 4 kHz Nyquist leaves no
/// energy above line 565, so the top long bands collapse to width 2.
/// Band 21 spans `starts[21]..576`. This is a distinct Fraunhofer table
/// with no 13818-3 sibling (the 16 kHz LSF long table is *not* its
/// source — see the doc's "8 kHz provenance" section).
const LONG_STARTS_MPEG25_8: [usize; 22] = [
    0, 12, 24, 36, 48, 60, 72, 88, 108, 132, 160, 192, 232, 280, 336, 400, 476, 566, 568, 570, 572,
    574,
];

/// Table B.8a (32 kHz) short-block per-window band start indices + end+1.
const SHORT_STARTS_32: [usize; 13] = [0, 4, 8, 12, 16, 22, 30, 42, 58, 78, 104, 138, 180];
/// Table B.8b (44.1 kHz) short-block per-window band start indices + end+1.
const SHORT_STARTS_44: [usize; 13] = [0, 4, 8, 12, 16, 22, 30, 40, 52, 66, 84, 106, 136];
/// Table B.8c (48 kHz) short-block per-window band start indices + end+1.
const SHORT_STARTS_48: [usize; 13] = [0, 4, 8, 12, 16, 22, 28, 38, 50, 64, 80, 100, 126];

/// ISO/IEC 13818-3:1997 Table B.2 — 16 kHz short-block per-window band
/// start indices + end+1 (widths 4, 4, 4, 6, 8, 10, 12, 14, 18, 24, 30,
/// 40, 18); band 12 spans `starts[12]..192` per window.
const SHORT_STARTS_LSF_16: [usize; 13] = [0, 4, 8, 12, 18, 26, 36, 48, 62, 80, 104, 134, 174];
/// ISO/IEC 13818-3:1997 Table B.2 — 22,05 kHz short-block per-window
/// band start indices + end+1 (widths 4, 4, 4, 6, 6, 8, 10, 14, 18, 26,
/// 32, 42, 18); band 12 spans `starts[12]..192` per window.
const SHORT_STARTS_LSF_22: [usize; 13] = [0, 4, 8, 12, 18, 24, 32, 42, 56, 74, 100, 132, 174];
/// ISO/IEC 13818-3:1997 Table B.2 — 24 kHz short-block per-window band
/// start indices + end+1 (widths 4, 4, 4, 6, 8, 10, 12, 14, 18, 24, 32,
/// 44, 12); band 12 spans `starts[12]..192` per window. The start
/// indices are accumulated from the "width of band" column — the
/// authoritative one (the rendered "index of start/end" columns for
/// this rate carry off-by-one typography).
const SHORT_STARTS_LSF_24: [usize; 13] = [0, 4, 8, 12, 18, 26, 36, 48, 62, 80, 104, 136, 180];

/// MPEG-2.5 8 kHz short-block per-window band start indices + end+1
/// (`docs/audio/mp3/mpeg2.5-scalefactor-bands.md`, "8 kHz, short
/// blocks"). Widths 8, 8, 8, 12, 16, 20, 24, 28, 36, then three width-2
/// filler bands (9–11) mirroring the long-block collapse, with band 12
/// (width 26) sweeping the residual lines; band 12 spans
/// `starts[12]..192` per window. Distinct Fraunhofer table, no 13818-3
/// sibling.
const SHORT_STARTS_MPEG25_8: [usize; 13] = [0, 8, 16, 24, 36, 52, 72, 96, 124, 160, 162, 164, 166];

/// In a mixed block the two lowest polyphase subbands (36 lines) are
/// coded as a long block; the remainder uses short windows (§2.4.2.7,
/// "mixed_block_flag"). 36 lines correspond to long-block scalefactor
/// bands 0..8 (whose widths sum to 36 at every sampling rate) and short
/// scalefactor bands 3..12 (which begin at per-window line 12, i.e.
/// interleaved line 36).
const MIXED_LONG_LINES: usize = 36;
/// The first short scalefactor band used in a mixed block (bands 0..3
/// are absorbed by the long-window portion).
const MIXED_FIRST_SHORT_SFB: usize = 3;

/// `2^(x/4)` for an integer-quarter exponent. Implemented via `exp2`
/// (`f32::powi`-free) so the long-block gain term `2^((global_gain-210)/4)`
/// and the short-block subblock term are evaluated directly from the
/// integer numerator without a lookup table.
fn pow2_quarter(quarter_numerator: i32) -> f32 {
    (quarter_numerator as f32 * 0.25).exp2()
}

/// `|is|^(4/3)` with the sign of `is` reapplied. `is == 0` maps to
/// `0.0` exactly.
fn signed_pow43(is: i32) -> f32 {
    if is == 0 {
        return 0.0;
    }
    let mag = (is.unsigned_abs() as f32).powf(4.0 / 3.0);
    if is < 0 {
        -mag
    } else {
        mag
    }
}

/// Requantize one granule-channel's 576 quantized frequency lines into
/// float frequency lines `xr[576]` per ISO/IEC 11172-3:1993
/// §2.4.3.4.7.1.
///
/// * `is` is the `[i32; 576]` quantized-line buffer from
///   [`crate::huffman::decode_huffman`].
/// * `gc` carries `global_gain`, `subblock_gain`, `scalefac_scale`,
///   `block_type`, `window_switching_flag`, and `mixed_block_flag`.
/// * `sf` carries the decoded `scalefac_l` / `scalefac_s` and the
///   effective `preflag` (transmitted for MPEG-1, derived for LSF — both
///   already resolved by [`crate::scalefactors::decode_scalefactors`]).
/// * `sample_rate_hz` / `version` select the Table B.8 band layout.
///
/// Returns the populated `[f32; 576]` `xr` buffer. Short-block lines are
/// left in their native `(sfb, window, freqline)` interleave; the
/// §2.4.3.4.8 reorder is a later stage.
#[must_use]
pub fn requantize(
    is: &[i32; NUM_LINES],
    gc: &GranuleChannel,
    sf: &ScaleFactors,
    sample_rate_hz: u32,
    version: MpegVersion,
) -> [f32; NUM_LINES] {
    let mut xr = [0.0f32; NUM_LINES];
    let mult = scalefac_multiplier(gc.scalefac_scale);
    let global = i32::from(gc.global_gain);
    let is_short = gc.window_switching_flag && gc.block_type == BlockType::Short;

    if !is_short {
        requantize_long_range(
            &mut xr,
            is,
            sf,
            0,
            NUM_LINES,
            global,
            mult,
            sample_rate_hz,
            version,
        );
        return xr;
    }

    // Short / window-switched block. The long-block gain term has no
    // subblock_gain; the short term subtracts 8*subblock_gain[window].
    if gc.mixed_block_flag {
        // Lowest 36 lines are a long block (§2.4.2.7), then short.
        requantize_long_range(
            &mut xr,
            is,
            sf,
            0,
            MIXED_LONG_LINES,
            global,
            mult,
            sample_rate_hz,
            version,
        );
        requantize_short_range(
            &mut xr,
            is,
            gc,
            sf,
            MIXED_FIRST_SHORT_SFB,
            global,
            mult,
            sample_rate_hz,
            version,
        );
    } else {
        requantize_short_range(
            &mut xr,
            is,
            gc,
            sf,
            0,
            global,
            mult,
            sample_rate_hz,
            version,
        );
    }

    xr
}

/// Requantize the line range `[lo, hi)` as long-block data, mapping each
/// line to its long scalefactor band (§2.4.3.4.7.1 long formula).
#[allow(clippy::too_many_arguments)]
fn requantize_long_range(
    xr: &mut [f32; NUM_LINES],
    is: &[i32; NUM_LINES],
    sf: &ScaleFactors,
    lo: usize,
    hi: usize,
    global: i32,
    mult: f32,
    sample_rate_hz: u32,
    version: MpegVersion,
) {
    let starts = long_band_starts(sample_rate_hz, version);
    // Long-block global gain term: 2^((global_gain - 210)/4).
    let gain = pow2_quarter(global - GAIN_BIAS);

    let mut sfb = 0usize;
    for i in lo..hi {
        // Advance the band cursor so `i` lies in [starts[sfb], starts[sfb+1]).
        while sfb + 1 < starts.len() && i >= starts[sfb + 1] {
            sfb += 1;
        }
        let scalefac = if sfb < 21 {
            let pre = if sf.preflag {
                u32::from(PRETAB[sfb])
            } else {
                0
            };
            u32::from(sf.long[sfb]) + pre
        } else {
            // Lines above the highest band have scalefactor zero
            // (§2.4.2.7: "the scale factor for frequency lines above the
            // highest line ... is zero").
            0
        };
        let sf_term = (-(mult * scalefac as f32)).exp2();
        xr[i] = signed_pow43(is[i]) * gain * sf_term;
    }
}

/// Requantize short-block data starting at short scalefactor band
/// `first_sfb` (§2.4.3.4.7.1 short formula). Lines are laid out
/// `(sfb, window, freqline)`: short band `sfb` with per-window
/// `[start, start+width)` occupies interleaved lines
/// `[3*start, 3*(start+width))`, window 0's `width` lines, then window 1,
/// then window 2.
#[allow(clippy::too_many_arguments)]
fn requantize_short_range(
    xr: &mut [f32; NUM_LINES],
    is: &[i32; NUM_LINES],
    gc: &GranuleChannel,
    sf: &ScaleFactors,
    first_sfb: usize,
    global: i32,
    mult: f32,
    sample_rate_hz: u32,
    version: MpegVersion,
) {
    let starts = short_band_starts(sample_rate_hz, version);
    // Per-window subblock-gain term factored out: precompute the gain
    // 2^((global_gain - 210 - 8*subblock_gain[window])/4) for each window.
    let win_gain = [
        pow2_quarter(global - GAIN_BIAS - 8 * i32::from(gc.subblock_gain[0])),
        pow2_quarter(global - GAIN_BIAS - 8 * i32::from(gc.subblock_gain[1])),
        pow2_quarter(global - GAIN_BIAS - 8 * i32::from(gc.subblock_gain[2])),
    ];

    for sfb in first_sfb..12 {
        let win_start = starts[sfb];
        let win_width = starts[sfb + 1] - starts[sfb];
        for (win, &gain) in win_gain.iter().enumerate() {
            let sf_term = (-(mult * f32::from(sf.short[sfb][win]))).exp2();
            let factor = gain * sf_term;
            // Interleaved base: bands below sfb contributed 3*win_start
            // lines; window `win` of this band starts after `win` blocks
            // of `win_width`.
            let base = 3 * win_start + win * win_width;
            for k in 0..win_width {
                let i = base + k;
                if i < NUM_LINES {
                    xr[i] = signed_pow43(is[i]) * factor;
                }
            }
        }
    }
}

#[cfg(test)]
include!("requantize_tests.rs");
