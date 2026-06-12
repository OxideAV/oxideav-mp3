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
    // MPEG-1 (ISO/IEC 11172-3 Table B.8a/b/c). LSF (13818-3) reuses
    // these long-block layouts for the band→scalefactor mapping this
    // round; the LSF-specific band tables are deferred (see report).
    let _ = version;
    match sample_rate_hz {
        32000 | 16000 | 8000 => &LONG_STARTS_32,
        48000 | 24000 | 12000 => &LONG_STARTS_48,
        _ => &LONG_STARTS_44, // 44100, 22050, 11025 and any default.
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
    let _ = version;
    match sample_rate_hz {
        32000 | 16000 | 8000 => &SHORT_STARTS_32,
        48000 | 24000 | 12000 => &SHORT_STARTS_48,
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

/// Table B.8a (32 kHz) short-block per-window band start indices + end+1.
const SHORT_STARTS_32: [usize; 13] = [0, 4, 8, 12, 16, 22, 30, 42, 58, 78, 104, 138, 180];
/// Table B.8b (44.1 kHz) short-block per-window band start indices + end+1.
const SHORT_STARTS_44: [usize; 13] = [0, 4, 8, 12, 16, 22, 30, 40, 52, 66, 84, 106, 136];
/// Table B.8c (48 kHz) short-block per-window band start indices + end+1.
const SHORT_STARTS_48: [usize; 13] = [0, 4, 8, 12, 16, 22, 28, 38, 50, 64, 80, 100, 126];

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
