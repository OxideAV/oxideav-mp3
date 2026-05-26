//! Layer III **stereo processing** — the §2.4.3.4.9 stage that runs after
//! §2.4.3.4.8 short-block reordering and before alias reduction / the
//! IMDCT.
//!
//! Joint-stereo Layer III carries two channels of one granule jointly:
//! the *left* channel position holds either the left signal (`L`), the
//! mid/sum signal (`M`), or the intensity magnitude; the *right* channel
//! position holds either the right signal (`R`), the side/difference
//! signal (`S`), or — in intensity bands — the stereo *position*
//! (transmitted as scalefactors, not spectral data). The two
//! `mode_extension` bits in the frame header (ISO/IEC 11172-3:1993
//! §2.4.2.3) select which methods are active:
//!
//! ```text
//! mode_extension   intensity_stereo   ms_stereo
//!      '00'             off               off
//!      '01'             on                off
//!      '10'             off               on
//!      '11'             on                on
//! ```
//!
//! This module reconstructs `L`/`R` from `M`/`S` and/or the intensity
//! positions, in place, per the §2.4.3.4.9 equations. It is only invoked
//! for joint-stereo frames; ordinary stereo / dual-channel / mono frames
//! pass each channel through unchanged (the caller skips this stage).
//!
//! # MS stereo (§2.4.3.4.9.2)
//!
//! In MS-stereo mode the normalized mid/side channels `M_i`/`S_i` are
//! transmitted instead of `L_i`/`R_i`; the decoder reconstructs
//!
//! ```text
//! L_i = (M_i + S_i) / sqrt(2)
//! R_i = (M_i - S_i) / sqrt(2)
//! ```
//!
//! with `M_i` in the left channel and `S_i` in the right (§2.4.3.4.9.2).
//! When MS-stereo is enabled but intensity stereo is not, the *entire*
//! spectrum is decoded in MS-stereo. When both are enabled, MS applies
//! only up to the intensity bound — the scalefactor band of the last
//! non-zero right-channel line — and intensity applies above it
//! (§2.4.3.4.9.1).
//!
//! # Intensity stereo (§2.4.3.4.9.3)
//!
//! Above the intensity bound the right channel carries no spectral data
//! ("zero-part"); its per-band scalefactor doubles as a stereo position
//! `is_pos[sfb]`. For each intensity-coded band the §2.4.3.4.9.3 steps
//! reconstruct `L`/`R` from the left-channel magnitude and the position.
//!
//! **MPEG-1** (ISO/IEC 11172-3 §2.4.3.4.9.3):
//!
//! ```text
//! is_pos == 7            -> illegal: band is not intensity-coded
//! is_ratio = tan(is_pos * pi / 12)
//! L_i = L_i * is_ratio / (1 + is_ratio)
//! R_i = L_i * 1       / (1 + is_ratio)   (L_i is the pre-step-4 value)
//! ```
//!
//! **MPEG-2 / MPEG-2.5 LSF** (ISO/IEC 13818-3 §2.4.3.2, which replaces
//! steps 4 and 5): the position uses a power-law factor `i0` selected by
//! `intensity_scale`, and the maximum position value (`is_pos == 7`,
//! illegal-position marker) means the band is not intensity-coded.
//!
//! ```text
//! i0 = 1/sqrt(2)         if intensity_scale == 1
//! i0 = 1/sqrt(sqrt(2))   otherwise
//! is_pos == 0            -> kl = 1, kr = 1
//! is_pos odd             -> kl = i0^((is_pos+1)/2), kr = 1
//! is_pos even (> 0)      -> kl = 1,                 kr = i0^(is_pos/2)
//! R_i = L_i * kr
//! L_i = L_i * kl
//! ```
//!
//! An illegal intensity position (or a band that is intensity-flagged but
//! whose position marks "not intensity"; ISO/IEC 13818-3 §2.4.3.2) is
//! decoded by the MS equations if MS-stereo is enabled, else the two
//! channels are left independent.
//!
//! # The intensity bound
//!
//! §2.4.3.4.9.1 / §2.4.3.4.9.3: the last scalefactor band that is *not*
//! intensity-coded is the one in which the last non-zero right-channel
//! frequency line occurs; every higher band is intensity-coded. ISO/IEC
//! 13818-3 §2.4.3.2 adds that for short blocks (`block_type == 2`) the
//! bound is computed *per window* — each short window has its own zero
//! part — so intensity decoding is applied per window. This module
//! derives the bound directly from the reordered right-channel `xr`
//! (the position of its last non-zero line).
//!
//! Band boundaries are read from the same Table B.8 columns as the
//! requantize / reorder stages via [`crate::requantize`].

use crate::frame::{ModeExtension, MpegVersion};
use crate::requantize::{long_band_starts, short_band_starts, NUM_LINES};
use crate::scalefactors::ScaleFactors;
use crate::side_info::{BlockType, GranuleChannel};

/// The intensity-position value that marks a scalefactor band as *not*
/// intensity-coded (ISO/IEC 11172-3 §2.4.3.4.9.3: "An intensity stereo
/// position of 7 … indicates that this scalefactor band is not decoded
/// as intensity stereo."). The same value is the illegal-position marker
/// for ISO/IEC 13818-3 §2.4.3.2.
const IS_POS_ILLEGAL: u8 = 7;

/// Number of short-block scalefactor bands per window (`scalefac_s[..12]`).
const SHORT_SFB: usize = 12;

/// In a mixed block the lowest 36 lines are a long window; the short
/// region begins at short scalefactor band 3. Matches `requantize` /
/// `reorder`.
const MIXED_FIRST_SHORT_SFB: usize = 3;

/// `1/sqrt(2)`, the MS-stereo normalization constant (§2.4.3.4.9.2) and
/// the LSF `intensity_scale == 1` base factor (ISO/IEC 13818-3 §2.4.3.2).
const INV_SQRT2: f32 = std::f32::consts::FRAC_1_SQRT_2;

/// Process one joint-stereo granule's two channels in place per ISO/IEC
/// 11172-3:1993 §2.4.3.4.9 (with the ISO/IEC 13818-3 §2.4.3.2 intensity
/// modifications for LSF frames).
///
/// * `left` is the reordered left-channel `xr[576]` (carrying `M` and/or
///   the intensity magnitude); it is rewritten to the reconstructed `L`.
/// * `right` is the reordered right-channel `xr[576]` (carrying `S` and/or
///   the zero-part above the intensity bound); it is rewritten to `R`.
/// * `right_sf` supplies the right channel's per-band scalefactors, which
///   double as `is_pos[sfb]` in intensity bands, plus `intensity_scale`
///   (LSF only).
/// * `right_gc` supplies the right channel's block type / window-switching
///   / mixed flags, selecting the long vs short (per-window) band layout
///   for the intensity bound.
/// * `mode_extension` carries the two §2.4.2.3 method-enable bits.
/// * `sample_rate_hz` / `version` select the Table B.8 band layout and
///   the MPEG-1 vs LSF intensity formula.
///
/// In MS-stereo mode both channels of a granule must share the same block
/// type (§2.4.3.4.9); the caller guarantees this from the side info.
pub fn process_stereo(
    left: &mut [f32; NUM_LINES],
    right: &mut [f32; NUM_LINES],
    right_sf: &ScaleFactors,
    right_gc: &GranuleChannel,
    mode_extension: ModeExtension,
    sample_rate_hz: u32,
    version: MpegVersion,
) {
    let ms = mode_extension.ms_stereo;
    let intensity = mode_extension.intensity_stereo;
    if !ms && !intensity {
        // mode_extension '00': joint-stereo frame with neither method
        // active (§2.4.2.3). Both channels are already L/R.
        return;
    }

    if !intensity {
        // MS only: the entire spectrum is MS-coded (§2.4.3.4.9.1).
        apply_ms_range(left, right, 0, NUM_LINES);
        return;
    }

    // Intensity is enabled (with or without MS). Process per band so each
    // band can be classified as below-bound (MS or pass-through) or
    // intensity-coded, using the per-window short-block layout when the
    // right channel is a short block.
    let is_short = right_gc.window_switching_flag && right_gc.block_type == BlockType::Short;
    if is_short {
        process_short(left, right, right_sf, right_gc, ms, sample_rate_hz, version);
    } else {
        process_long(left, right, right_sf, ms, sample_rate_hz, version);
    }
}

/// MS matrix on the half-open line range `[lo, hi)` (§2.4.3.4.9.2):
/// `L = (M + S)/sqrt(2)`, `R = (M - S)/sqrt(2)`.
fn apply_ms_range(left: &mut [f32; NUM_LINES], right: &mut [f32; NUM_LINES], lo: usize, hi: usize) {
    for i in lo..hi {
        let m = left[i];
        let s = right[i];
        left[i] = (m + s) * INV_SQRT2;
        right[i] = (m - s) * INV_SQRT2;
    }
}

/// Long-block (or non-window-switched) intensity / MS processing.
///
/// The intensity bound is the band after the one holding the last
/// non-zero right-channel line (§2.4.3.4.9.1). Bands below the bound are
/// MS-coded (if MS enabled) or pass through; bands at/above the bound are
/// intensity-coded.
fn process_long(
    left: &mut [f32; NUM_LINES],
    right: &mut [f32; NUM_LINES],
    right_sf: &ScaleFactors,
    ms: bool,
    sample_rate_hz: u32,
    version: MpegVersion,
) {
    let starts = long_band_starts(sample_rate_hz, version);
    let bound_sfb = long_intensity_bound(right, starts);

    for sfb in 0..21 {
        let lo = starts[sfb];
        let hi = starts[sfb + 1];
        if sfb < bound_sfb {
            if ms {
                apply_ms_range(left, right, lo, hi);
            }
            // else: below-bound band passes through (already L/R).
        } else {
            let is_pos = right_sf.long[sfb];
            apply_intensity_band(
                left,
                right,
                lo,
                hi,
                is_pos,
                right_sf.intensity_scale,
                ms,
                version,
            );
        }
    }
}

/// Short-block intensity / MS processing. The intensity bound is computed
/// per window (ISO/IEC 13818-3 §2.4.3.2: "the calculation of the
/// intensity bound is applied to the values of each short window").
///
/// At this point `xr` is in subband order (post-reorder): the three
/// windows of one frequency line are adjacent, so short scalefactor band
/// `sfb` with per-window `[s, s+w)` occupies interleaved lines
/// `3*s + 3*k + win` for `k in 0..w`, `win in 0..3`. For a mixed block the
/// long region (lines 0..36) is handled with the long-band layout.
fn process_short(
    left: &mut [f32; NUM_LINES],
    right: &mut [f32; NUM_LINES],
    right_sf: &ScaleFactors,
    right_gc: &GranuleChannel,
    ms: bool,
    sample_rate_hz: u32,
    version: MpegVersion,
) {
    let short_starts = short_band_starts(sample_rate_hz, version);
    let first_short_sfb = if right_gc.mixed_block_flag {
        // Lowest 36 lines are a long window; process them with the long
        // layout (they are below the intensity bound for any real stream,
        // but MS still applies if enabled).
        let long_starts = long_band_starts(sample_rate_hz, version);
        let bound_sfb = long_intensity_bound_range(right, long_starts, 0, 36);
        for sfb in 0..8 {
            let lo = long_starts[sfb];
            let hi = long_starts[sfb + 1];
            if sfb < bound_sfb {
                if ms {
                    apply_ms_range(left, right, lo, hi);
                }
            } else {
                let is_pos = right_sf.long[sfb];
                apply_intensity_band(
                    left,
                    right,
                    lo,
                    hi,
                    is_pos,
                    right_sf.intensity_scale,
                    ms,
                    version,
                );
            }
        }
        MIXED_FIRST_SHORT_SFB
    } else {
        0
    };

    // Per-window intensity bound: for each window, find the highest short
    // band that holds a non-zero right-channel line.
    let bound = [
        short_intensity_bound(right, short_starts, 0, first_short_sfb),
        short_intensity_bound(right, short_starts, 1, first_short_sfb),
        short_intensity_bound(right, short_starts, 2, first_short_sfb),
    ];

    for sfb in first_short_sfb..SHORT_SFB {
        let s = short_starts[sfb];
        let w = short_starts[sfb + 1] - short_starts[sfb];
        let base = 3 * s;
        for (win, &bound_sfb) in bound.iter().enumerate() {
            if sfb < bound_sfb {
                if ms {
                    for k in 0..w {
                        let i = base + 3 * k + win;
                        if i < NUM_LINES {
                            let m = left[i];
                            let sv = right[i];
                            left[i] = (m + sv) * INV_SQRT2;
                            right[i] = (m - sv) * INV_SQRT2;
                        }
                    }
                }
            } else {
                let is_pos = right_sf.short[sfb][win];
                let (kl, kr, intensity_ok) =
                    intensity_factors(is_pos, right_sf.intensity_scale, version);
                for k in 0..w {
                    let i = base + 3 * k + win;
                    if i >= NUM_LINES {
                        continue;
                    }
                    if intensity_ok {
                        let l = left[i];
                        left[i] = l * kl;
                        right[i] = l * kr;
                    } else if ms {
                        let m = left[i];
                        let sv = right[i];
                        left[i] = (m + sv) * INV_SQRT2;
                        right[i] = (m - sv) * INV_SQRT2;
                    }
                    // else: illegal is_pos, MS off — channels independent.
                }
            }
        }
    }
}

/// The first intensity-coded long scalefactor band: one past the band
/// holding the last non-zero right-channel line over the whole spectrum
/// (§2.4.3.4.9.1). Returns `21` when the right channel is non-zero up to
/// the top band (no intensity region).
fn long_intensity_bound(right: &[f32; NUM_LINES], starts: &[usize; 22]) -> usize {
    long_intensity_bound_range(right, starts, 0, NUM_LINES)
}

/// As [`long_intensity_bound`] but restricted to lines `[lo, hi)` (used
/// for the long region of a mixed block).
fn long_intensity_bound_range(
    right: &[f32; NUM_LINES],
    starts: &[usize; 22],
    lo: usize,
    hi: usize,
) -> usize {
    let mut last_nonzero: Option<usize> = None;
    for (i, &v) in right.iter().enumerate().take(hi).skip(lo) {
        if v != 0.0 {
            last_nonzero = Some(i);
        }
    }
    match last_nonzero {
        None => 0, // entire range zero -> intensity from the first band.
        Some(line) => {
            // Band holding `line`, plus one (that band is the last
            // non-intensity band).
            for sfb in 0..21 {
                if line < starts[sfb + 1] {
                    return sfb + 1;
                }
            }
            21
        }
    }
}

/// The first intensity-coded short scalefactor band for `window`,
/// computed from that window's last non-zero right-channel line (ISO/IEC
/// 13818-3 §2.4.3.2 per-window bound). Bands below `first_sfb` (the
/// long region of a mixed block) are not considered.
fn short_intensity_bound(
    right: &[f32; NUM_LINES],
    starts: &[usize; 13],
    window: usize,
    first_sfb: usize,
) -> usize {
    let mut last_sfb: Option<usize> = None;
    for sfb in first_sfb..SHORT_SFB {
        let s = starts[sfb];
        let w = starts[sfb + 1] - starts[sfb];
        let base = 3 * s;
        for k in 0..w {
            let i = base + 3 * k + window;
            if i < NUM_LINES && right[i] != 0.0 {
                last_sfb = Some(sfb);
            }
        }
    }
    match last_sfb {
        None => first_sfb,
        Some(sfb) => sfb + 1,
    }
}

/// Apply intensity (or its MS / pass-through fallback) to a contiguous
/// long-block line range `[lo, hi)`.
#[allow(clippy::too_many_arguments)]
fn apply_intensity_band(
    left: &mut [f32; NUM_LINES],
    right: &mut [f32; NUM_LINES],
    lo: usize,
    hi: usize,
    is_pos: u8,
    intensity_scale: bool,
    ms: bool,
    version: MpegVersion,
) {
    let (kl, kr, intensity_ok) = intensity_factors(is_pos, intensity_scale, version);
    for i in lo..hi {
        if intensity_ok {
            let l = left[i];
            left[i] = l * kl;
            right[i] = l * kr;
        } else if ms {
            let m = left[i];
            let s = right[i];
            left[i] = (m + s) * INV_SQRT2;
            right[i] = (m - s) * INV_SQRT2;
        }
        // else: illegal is_pos, MS off — left/right kept independent.
    }
}

/// Compute the intensity-stereo per-channel gains `(kl, kr, ok)` for a
/// band whose right-channel scalefactor is `is_pos`. `ok == false` means
/// the position is illegal (band not intensity-coded), so the caller
/// applies the MS / independent fallback instead.
///
/// * **MPEG-1** (ISO/IEC 11172-3 §2.4.3.4.9.3): `is_ratio = tan(is_pos *
///   pi/12)`; `kl = is_ratio / (1 + is_ratio)`, `kr = 1 / (1 + is_ratio)`
///   so that `L = L*kl`, `R = L*kr` reproduce steps 4 and 5 (`R` uses the
///   pre-step-4 `L`).
/// * **LSF** (ISO/IEC 13818-3 §2.4.3.2): power-law factor selected by
///   `intensity_scale`.
fn intensity_factors(is_pos: u8, intensity_scale: bool, version: MpegVersion) -> (f32, f32, bool) {
    if is_pos == IS_POS_ILLEGAL {
        return (0.0, 0.0, false);
    }
    match version {
        MpegVersion::Mpeg1 => {
            let is_ratio = ((f32::from(is_pos)) * std::f32::consts::PI / 12.0).tan();
            let denom = 1.0 + is_ratio;
            let kl = is_ratio / denom;
            let kr = 1.0 / denom;
            (kl, kr, true)
        }
        // MPEG-2 and MPEG-2.5 share the §13818-3 LSF intensity factors.
        MpegVersion::Mpeg2 | MpegVersion::Mpeg25 => {
            // ISO/IEC 13818-3 §2.4.3.2 step 4/5 replacement.
            // i0 = 1/sqrt(2) for intensity_scale==1, else 1/sqrt(sqrt(2)).
            let i0 = if intensity_scale {
                INV_SQRT2
            } else {
                INV_SQRT2.sqrt()
            };
            if is_pos == 0 {
                (1.0, 1.0, true)
            } else if is_pos % 2 == 1 {
                // is_pos odd: (is_pos + 1) / 2 == is_pos.div_ceil(2).
                let kl = i0.powi(i32::from(is_pos.div_ceil(2)));
                (kl, 1.0, true)
            } else {
                let kr = i0.powi(i32::from(is_pos / 2));
                (1.0, kr, true)
            }
        }
    }
}

#[cfg(test)]
include!("stereo_tests.rs");
