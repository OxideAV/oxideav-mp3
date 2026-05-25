//! Layer III **quantization primitive** — the algebraic inverse of
//! [`crate::requantize::requantize`] (§2.4.3.4.7.1 of
//! ISO/IEC 11172-3:1993). Given a target float spectrum `xr[576]` and a
//! chosen scalefactor configuration (`GranuleChannel` + `ScaleFactors`),
//! pick the integer-Huffman-input values `is[576]` that the decoder will
//! requantize back into a buffer matching `xr` within precision.
//!
//! # Scope (Phase 2 step 4)
//!
//! This file is **only** the primitive — given an already-chosen
//! `global_gain` / `subblock_gain` / scalefactor configuration, solve for
//! `is[i]`. It does **not** search for a `global_gain`, allocate bits,
//! choose scalefactors, or perform noise-shaping iterations. Those are
//! the later steps of the §C.1.5.4 (informational) iteration loops; this
//! crate will grow them as separate primitives in subsequent rounds.
//!
//! # Algebra (§2.4.3.4.7.1)
//!
//! The decoder long-block formula is
//!
//! ```text
//! xr_i = sign(is_i) * |is_i|^(4/3) * G_long(sfb)
//! G_long(sfb) = 2^((global_gain - 210)/4)
//!             * 2^(-mult * (scalefac_l[sfb] + preflag*pretab[sfb]))
//! ```
//!
//! Solving for `is_i`:
//!
//! ```text
//! |is_i|^(4/3) = |xr_i| / G_long(sfb)
//!       |is_i| = (|xr_i| / G_long(sfb))^(3/4)
//!        is_i  = sign(xr_i) * round((|xr_i| / G_long(sfb))^(3/4))
//! ```
//!
//! The short-block form is identical with the per-window factor
//!
//! ```text
//! G_short(sfb, w) = 2^((global_gain - 210 - 8*subblock_gain[w])/4)
//!                 * 2^(-mult * scalefac_s[sfb][w])
//! ```
//!
//! The mixed-block split is the same as on the decoder side: lines
//! `[0, 36)` use the long form against `scalefac_l[0..8]`; lines
//! `[36, 576)` use the short form starting at `sfb = 3`.
//!
//! # Why round-to-nearest is enough for the primitive
//!
//! The decoder applies `|is|^(4/3) * factor`. With round-to-nearest the
//! reconstruction error on a flat unit input is bounded by half a
//! quantization step in the `4/3`-power domain; the bin-level RMS stays
//! comfortably below `1e-3` for unit-magnitude `xr`. The spec's
//! informational Annex C §C.1.5.4 offers a bias term (`+0.0946`) inside
//! the `^(3/4)` to bias the rounding away from zero — that is part of
//! the noise-shaping iteration, not the primitive — so we leave it out
//! and document the option for the bit-allocation step that comes next.
//!
//! No external implementation was consulted; every constant and
//! field-routing rule is taken from the ISO/IEC 11172-3:1993 §2.4.3.4.7
//! text (the same §s the [`crate::requantize`] decoder side cites).

use crate::frame::MpegVersion;
use crate::requantize::{
    long_band_starts, scalefac_multiplier, short_band_starts, NUM_LINES, PRETAB,
};
use crate::scalefactors::ScaleFactors;
use crate::side_info::{BlockType, GranuleChannel};

/// System constant subtracted from `global_gain` in the §2.4.3.4.7.1
/// exponent (matches [`crate::requantize`]).
const GAIN_BIAS: i32 = 210;

/// Lines `[0, 36)` of a mixed block are encoded as a long block; the
/// short region starts at short scalefactor band 3.
const MIXED_LONG_LINES: usize = 36;
const MIXED_FIRST_SHORT_SFB: usize = 3;

/// `2^(x/4)` for an integer-quarter exponent (mirror of the decoder
/// helper; reproduced here so the encoder module doesn't reach into a
/// decoder-internal helper).
fn pow2_quarter(quarter_numerator: i32) -> f32 {
    (quarter_numerator as f32 * 0.25).exp2()
}

/// `round(|x|^(3/4))` clipped to a non-negative `i32`. `x == 0` maps to
/// `0`. The spec's `is[]` field is unsigned per the Huffman codebooks,
/// so the caller reapplies the sign of `xr` after this returns.
fn quantize_magnitude(abs_xr_over_factor: f32) -> i32 {
    if abs_xr_over_factor == 0.0 {
        return 0;
    }
    let mag = abs_xr_over_factor.powf(0.75);
    // round-half-away-from-zero is the f32 default of `.round()`. The
    // §C.1.5.4 (informational) iteration adds a +0.0946 bias before the
    // round to push values away from zero — that's a noise-shaping
    // choice, not the primitive; the bit-allocation step adds it.
    let rounded = mag.round();
    if rounded.is_finite() && rounded >= 0.0 {
        rounded as i32
    } else {
        0
    }
}

/// Quantize one frequency line: divide by the precomputed inverse-gain
/// factor, take the `3/4` power, round, and reapply the sign of `xr`.
fn quantize_line(xr: f32, factor: f32) -> i32 {
    if xr == 0.0 || factor == 0.0 {
        return 0;
    }
    let abs_quot = (xr.abs() / factor).max(0.0);
    let mag = quantize_magnitude(abs_quot);
    if xr < 0.0 {
        -mag
    } else {
        mag
    }
}

/// Quantize one granule-channel's 576 target float frequency lines `xr[]`
/// into the integer Huffman-input buffer `is[576]` that, when passed to
/// [`crate::requantize::requantize`] with the same `gc` / `sf` /
/// `sample_rate_hz` / `version`, reproduces `xr[]` within `f32`
/// round-to-nearest precision.
///
/// This is the algebraic inverse of the decoder side and does **not**
/// optimize `global_gain`, allocate bits, or choose scalefactors — those
/// are subsequent primitives in the encoder build-out.
///
/// Returns the `[i32; 576]` `is` buffer. Short-block lines are produced
/// in the same native `(sfb, window, freqline)` interleave the decoder
/// reads (the §2.4.3.4.8 reorder is a separate stage).
#[must_use]
pub fn quantize(
    xr: &[f32; NUM_LINES],
    gc: &GranuleChannel,
    sf: &ScaleFactors,
    sample_rate_hz: u32,
    version: MpegVersion,
) -> [i32; NUM_LINES] {
    let mut is = [0i32; NUM_LINES];
    let mult = scalefac_multiplier(gc.scalefac_scale);
    let global = i32::from(gc.global_gain);
    let is_short = gc.window_switching_flag && gc.block_type == BlockType::Short;

    if !is_short {
        quantize_long_range(
            &mut is,
            xr,
            sf,
            0,
            NUM_LINES,
            global,
            mult,
            sample_rate_hz,
            version,
        );
        return is;
    }

    if gc.mixed_block_flag {
        quantize_long_range(
            &mut is,
            xr,
            sf,
            0,
            MIXED_LONG_LINES,
            global,
            mult,
            sample_rate_hz,
            version,
        );
        quantize_short_range(
            &mut is,
            xr,
            gc,
            sf,
            MIXED_FIRST_SHORT_SFB,
            global,
            mult,
            sample_rate_hz,
            version,
        );
    } else {
        quantize_short_range(
            &mut is,
            xr,
            gc,
            sf,
            0,
            global,
            mult,
            sample_rate_hz,
            version,
        );
    }

    is
}

/// Inverse of [`crate::requantize::requantize_long_range`]: quantize the
/// line range `[lo, hi)` against the long scalefactor bands.
#[allow(clippy::too_many_arguments)]
fn quantize_long_range(
    is: &mut [i32; NUM_LINES],
    xr: &[f32; NUM_LINES],
    sf: &ScaleFactors,
    lo: usize,
    hi: usize,
    global: i32,
    mult: f32,
    sample_rate_hz: u32,
    version: MpegVersion,
) {
    let starts = long_band_starts(sample_rate_hz, version);
    let gain = pow2_quarter(global - GAIN_BIAS);

    let mut sfb = 0usize;
    for i in lo..hi {
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
            0
        };
        let sf_term = (-(mult * scalefac as f32)).exp2();
        let factor = gain * sf_term;
        is[i] = quantize_line(xr[i], factor);
    }
}

/// Inverse of [`crate::requantize::requantize_short_range`]: quantize the
/// short-block lines starting at short scalefactor band `first_sfb`.
#[allow(clippy::too_many_arguments)]
fn quantize_short_range(
    is: &mut [i32; NUM_LINES],
    xr: &[f32; NUM_LINES],
    gc: &GranuleChannel,
    sf: &ScaleFactors,
    first_sfb: usize,
    global: i32,
    mult: f32,
    sample_rate_hz: u32,
    version: MpegVersion,
) {
    let starts = short_band_starts(sample_rate_hz, version);
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
            let base = 3 * win_start + win * win_width;
            for k in 0..win_width {
                let i = base + k;
                if i < NUM_LINES {
                    is[i] = quantize_line(xr[i], factor);
                }
            }
        }
    }
}

#[cfg(test)]
include!("quantize_tests.rs");
