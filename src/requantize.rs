//! MPEG-1 Layer III requantisation, reorder, antialias.
//!
//! Spec: ISO/IEC 11172-3 §2.4.3.4.
//!
//! The requantisation formula for a long block is:
//!
//!   xr[i] = sign(is[i]) * |is[i]|^(4/3)
//!         * 2^( (global_gain - 210) / 4.0 )
//!         * 2^( -scalefac_l[sfb] * (1 + scalefac_scale) * shift )
//!
//! where `shift` is 0.5 normally or `preflag`-adjusted. The short-block
//! form adds a subblock_gain term per-window.
//!
//! After requantisation short-block coefficients are reordered from
//! subband/scalefactor-band layout into window layout.
//!
//! Antialias: 8-tap butterfly across subband boundaries, applied only to
//! long blocks or the long portion of mixed blocks.

use crate::scalefactor::ScaleFactors;
use crate::sfband::{sfband_long, sfband_short};
use crate::sideinfo::GranuleChannel;

/// Preflag pretab - additional scalefactor bias applied when preflag=1.
/// ISO Table 3-B.6.
const PRETAB: [u8; 22] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 2, 0,
];

/// Per-sample requantisation. `is[i]` are the 576 integer coefficients
/// decoded from Huffman / count1 regions. `xr[i]` are the dequantised
/// float coefficients ready for stereo processing, antialias, IMDCT.
pub fn requantize_granule(
    is_: &[i32; 576],
    xr: &mut [f32; 576],
    gc: &GranuleChannel,
    sf: &ScaleFactors,
    sample_rate: u32,
) {
    let global_gain = gc.global_gain as i32;
    let base_scale = f32_pow2((global_gain - 210) as f32 / 4.0);
    let scale_shift = if gc.scalefac_scale { 1.0 } else { 0.5 };

    // Branch on block type for sfb layout.
    if gc.window_switching_flag && gc.block_type == 2 {
        // Short or mixed block.
        let long_sfb_count = if gc.mixed_block_flag { 8 } else { 0 };
        let long_bounds = sfband_long(sample_rate);

        // Long portion (first `long_sfb_count` sfbs) — identical to long
        // handling.
        let long_end = long_bounds[long_sfb_count] as usize;
        for sfb in 0..long_sfb_count {
            let lo = long_bounds[sfb] as usize;
            let hi = long_bounds[sfb + 1] as usize;
            let pre = if gc.preflag { PRETAB[sfb] as i32 } else { 0 };
            let sf_exp = -scale_shift * (sf.l[sfb] as f32 + pre as f32);
            let s = base_scale * f32_pow2(sf_exp);
            for i in lo..hi {
                xr[i] = requant_sample(is_[i], s);
            }
        }

        // Short portion. Per ISO 11172-3 §2.4.3.4, short-block gain is:
        //   xr = sign(is) * |is|^(4/3)
        //        * 2^(0.25 * (global_gain - 210 - 8 * subblock_gain[w]))
        //        * 2^(-(1+scalefac_scale)/2 * scalefac_s[sfb][w])
        // The subblock_gain term applies inside the global-gain exponent
        // (factor 0.25 * 8 = 2.0 per unit), NOT also outside it.
        let short_bounds = sfband_short(sample_rate);
        let start_sfb = if gc.mixed_block_flag { 3 } else { 0 };
        let mut pos = long_end;
        for sfb in start_sfb..13 {
            let sfb_width = (short_bounds[sfb + 1] - short_bounds[sfb]) as usize;
            for win in 0..3 {
                let sbgain = gc.subblock_gain[win] as i32;
                let sf_exp = -scale_shift * sf.s[sfb][win] as f32 - 2.0 * sbgain as f32;
                let s = base_scale * f32_pow2(sf_exp);
                for _i in 0..sfb_width {
                    if pos >= 576 {
                        return;
                    }
                    xr[pos] = requant_sample(is_[pos], s);
                    pos += 1;
                }
            }
        }
    } else {
        // Pure long block.
        let bounds = sfband_long(sample_rate);
        for sfb in 0..21 {
            let lo = bounds[sfb] as usize;
            let hi = bounds[sfb + 1] as usize;
            let pre = if gc.preflag { PRETAB[sfb] as i32 } else { 0 };
            let sf_exp = -scale_shift * (sf.l[sfb] as f32 + pre as f32);
            let s = base_scale * f32_pow2(sf_exp);
            for i in lo..hi.min(576) {
                xr[i] = requant_sample(is_[i], s);
            }
        }
    }
}

fn requant_sample(v: i32, scale: f32) -> f32 {
    if v == 0 {
        return 0.0;
    }
    let mag = (v.unsigned_abs() as f32).powf(4.0 / 3.0);
    let val = mag * scale;
    if v < 0 {
        -val
    } else {
        val
    }
}

#[inline]
fn f32_pow2(e: f32) -> f32 {
    (e * std::f32::consts::LN_2).exp()
}

// --------------- Reorder for short blocks ---------------

/// Reorder short-block coefficients from window-major (sfb→win→freq, the
/// layout produced by `requantize_granule`) to interleaved-by-window
/// (sfb→freq→win) layout that the IMDCT short path expects. ISO 11172-3
/// §2.4.3.4.10.5.
///
/// For pure short blocks, sfb 0..13 are reordered. For mixed blocks, only
/// sfb 3..13 of the short region are reordered (sfb 0..7 are long and stay
/// in subbands 0..1).
pub fn reorder_short(xr: &mut [f32; 576], gc: &GranuleChannel, sample_rate: u32) {
    if !(gc.window_switching_flag && gc.block_type == 2) {
        return;
    }
    let short_bounds = sfband_short(sample_rate);
    let (start_sfb, region_start) = if gc.mixed_block_flag {
        // Mixed: long part covers sfb 0..7 (long sfbs), so the short
        // region starts at long-sfb-index 8 in the granule layout.
        // For 44.1k that's offset 36.
        (3, sfband_long(sample_rate)[8] as usize)
    } else {
        (0, 0)
    };
    let mut buf = [0.0f32; 576];
    let mut pos = region_start;
    let mut sfb = start_sfb;
    while pos < 576 && sfb < 13 {
        let width = (short_bounds[sfb + 1] - short_bounds[sfb]) as usize;
        // Source: window-major. xr[pos..pos + 3*width] holds win 0 (width),
        // win 1 (width), win 2 (width).
        // Destination: interleaved. For each freq j in 0..width and window
        // w in 0..3, dest is at offset 3*j + w within this sfb's region.
        for w in 0..3 {
            for j in 0..width {
                let src = pos + w * width + j;
                let dst = pos + 3 * j + w;
                if src < 576 && dst < 576 {
                    buf[dst] = xr[src];
                }
            }
        }
        pos += 3 * width;
        sfb += 1;
    }
    // Copy back the reordered region.
    xr[region_start..576].copy_from_slice(&buf[region_start..576]);
}

// --------------- Antialias ---------------

/// Antialias butterfly coefficients (ISO Table 3-B.9). c_s = cos, c_a = -sin.
#[rustfmt::skip]
const CS: [f32; 8] = [
    0.857_492_92, 0.881_742_0,  0.949_628_64, 0.983_314_6,
    0.995_517_8,  0.999_160_8,  0.999_899_2,  0.999_993_04,
];
#[rustfmt::skip]
const CA: [f32; 8] = [
   -0.514_495_76, -0.471_731_97, -0.313_377_46, -0.181_913_2,
   -0.094_574_19, -0.040_965_58, -0.014_197_132,-0.003_699_975,
];

/// Apply the 8-tap antialias butterfly across the 18-sample boundaries
/// of the first 18 subbands (except when block_type == 2, in which case
/// antialias is applied only to the long subbands of a mixed block).
pub fn antialias(xr: &mut [f32; 576], gc: &GranuleChannel) {
    let max_subband = if gc.window_switching_flag && gc.block_type == 2 {
        if gc.mixed_block_flag {
            2 // only long part of mixed block: subbands 0 and 1
        } else {
            0 // pure short block: no antialias
        }
    } else {
        32
    };
    for sb in 1..max_subband.min(32) {
        let base = 18 * sb;
        for i in 0..8 {
            let up = base - 1 - i;
            let dn = base + i;
            let a = xr[up];
            let b = xr[dn];
            xr[up] = a * CS[i] - b * CA[i];
            xr[dn] = b * CS[i] + a * CA[i];
        }
    }
}

// --------------- MS / Intensity Stereo ---------------

/// Apply MS stereo (rotate by 1/sqrt(2)) to a stereo pair of coefficients.
/// Condition: mode_extension bit 0x2 is set.
pub fn ms_stereo(xr_l: &mut [f32; 576], xr_r: &mut [f32; 576]) {
    ms_stereo_range(xr_l, xr_r, 0, 576);
}

/// Apply MS stereo only on samples in `[lo, hi)`. Used when IS and MS
/// coexist — MS covers the below-IS-bound region, IS covers above.
pub fn ms_stereo_range(
    xr_l: &mut [f32; 576],
    xr_r: &mut [f32; 576],
    lo: usize,
    hi: usize,
) {
    let inv_sqrt2 = 1.0 / 2.0_f32.sqrt();
    let end = hi.min(576);
    for i in lo..end {
        let m = xr_l[i];
        let s = xr_r[i];
        xr_l[i] = (m + s) * inv_sqrt2;
        xr_r[i] = (m - s) * inv_sqrt2;
    }
}

/// MPEG-1 intensity-stereo `is_ratio = tan(is_pos * PI / 12)` lookup
/// (ISO/IEC 11172-3 §2.4.3.4.9.3). `is_pos` ranges 0..=6; `is_pos == 7`
/// signals the band is NOT intensity-coded (the R scalefactor uses
/// that value as a sentinel). The final L / R gains are:
///   L_out = L * is_ratio / (1 + is_ratio)
///   R_out = L * 1        / (1 + is_ratio)
/// Precomputed as (k_l, k_r) pairs.
#[rustfmt::skip]
pub(crate) const IS_RATIO_MPEG1: [(f32, f32); 7] = {
    // tan(k * pi/12) for k = 0..6: 0, 0.2679..., 0.5773..., 1.0, 1.7320..., 3.7320..., inf
    // Pre-compute split factors; for is_pos=6 the tangent is infinite
    // (tan(pi/2) undefined) — in that case L channel is 0 and R gets
    // the full signal, i.e. (k_l, k_r) = (0, 1).
    [
        (0.0,            1.0),              // is_pos = 0 -> L = L*0, R = L*1 (all to R)
        (0.211_324_87,   0.788_675_13),     // is_pos = 1 -> tan(pi/12)
        (0.366_025_4,    0.633_974_6),
        (0.5,            0.5),              // is_pos = 3 -> equal split
        (0.633_974_6,    0.366_025_4),
        (0.788_675_13,   0.211_324_87),
        (1.0,            0.0),              // is_pos = 6 -> all to L
    ]
};

/// Intensity-stereo processing for a joint-stereo granule.
///
/// # Overview
/// Intensity stereo is an alternative to M/S that encodes a single L-channel
/// spectrum plus, per scalefactor-band, an "intensity position" (`is_pos`)
/// describing the L/R energy split for that band. It is used for highly
/// correlated high-frequency content and coexists with M/S within one
/// frame: below the intensity-bound (`is_bound`), regular (or M/S) stereo
/// applies; at or above `is_bound`, the R channel is reconstructed from L.
///
/// # Spec
/// * MPEG-1: ISO/IEC 11172-3 §2.4.3.4.9.3. `is_pos` = R-channel scalefactor
///   in that band; valid range 0..=6; `is_pos == 7` marks the band as
///   NOT intensity-coded (R stays zero). Ratio: `is_ratio = tan(is_pos * pi/12)`.
/// * MPEG-2 / MPEG-2.5: ISO/IEC 13818-3 §2.4.3.2. Finer-grained
///   `is_pos` (0..=30) and selectable `intensity_scale` — not implemented
///   in this crate yet; MPEG-2 IS falls back to leaving R at zero above
///   `is_bound`, which matches what the encoder wrote for that band.
///
/// # Algorithm (MPEG-1)
/// 1. Determine `is_bound_sfb` = the lowest sfb ≥ all nonzero R coefficients —
///    i.e. the first sfb where every R coeff in that sfb and above is zero.
///    (The encoder guarantees R is zero at/above this bound.)
/// 2. For each sfb ≥ `is_bound_sfb`:
///    * Read `is_pos = sf_r.l[sfb]` (long) or `sf_r.s[sfb][win]` (short).
///    * If `is_pos != 7` and `is_pos <= 6`: for each coefficient `i` in that
///      band, set `R[i] = L[i] * k_r` and `L[i] = L[i] * k_l` where
///      `(k_l, k_r)` come from [`IS_RATIO_MPEG1`].
///    * If `is_pos == 7`: leave R = 0 (the band is NOT intensity-coded;
///      R was simply silenced by the encoder).
/// 3. Below `is_bound_sfb` stereo is handled elsewhere (M/S if enabled,
///    otherwise L/R).
///
/// # Interaction with M/S
/// When both mode_extension bits are set, M/S applies in sfbs `[0, is_bound)`
/// and IS applies in `[is_bound, end)`. We apply IS first here, then the
/// caller runs M/S only on the below-bound region (or skips it; the simple
/// approach used by minimp3 applies M/S to the whole granule because
/// the IS-band R is already zero pre-IS — but that's only correct up
/// through step 1; we run the split explicitly to avoid the M/S rotation
/// stamping on the IS-processed R coeffs).
pub fn intensity_stereo_mpeg1(
    xr_l: &mut [f32; 576],
    xr_r: &mut [f32; 576],
    sf_r: &crate::scalefactor::ScaleFactors,
    gc_r: &GranuleChannel,
    sample_rate: u32,
    is_bound_sfb: usize,
) {
    if gc_r.window_switching_flag && gc_r.block_type == 2 {
        // Short or mixed block. Short sfbs × 3 windows.
        let short_bounds = sfband_short(sample_rate);
        let (long_end_sfb, long_bounds_end_sample) = if gc_r.mixed_block_flag {
            // Mixed: long region sfb 0..8 (subbands 0,1 for 44.1 kHz), then short.
            (8usize, sfband_long(sample_rate)[8] as usize)
        } else {
            (0usize, 0usize)
        };
        // Long portion (mixed only): treat sfb 0..long_end_sfb as long sfbs.
        let long_bounds = sfband_long(sample_rate);
        for sfb in 0..long_end_sfb {
            if sfb < is_bound_sfb {
                continue;
            }
            let lo = long_bounds[sfb] as usize;
            let hi = long_bounds[sfb + 1] as usize;
            let is_pos = sf_r.l[sfb] as usize;
            apply_is_band(xr_l, xr_r, lo, hi, is_pos);
        }
        // Short region. For each short sfb, 3 windows each of width w.
        let short_start_sfb = if gc_r.mixed_block_flag { 3 } else { 0 };
        let mut pos = long_bounds_end_sample;
        for sfb in short_start_sfb..13 {
            let w = (short_bounds[sfb + 1] - short_bounds[sfb]) as usize;
            for win in 0..3 {
                // For short blocks, the IS bound is tracked per window.
                // Spec: the bound is sfb-based, so we apply IS to all short
                // sfbs >= is_bound_sfb regardless of window. Simpler and
                // matches common decoder behaviour.
                if sfb < is_bound_sfb {
                    pos += w;
                    continue;
                }
                let is_pos = sf_r.s[sfb][win] as usize;
                apply_is_band(xr_l, xr_r, pos, pos + w, is_pos);
                pos += w;
            }
        }
    } else {
        // Long block — sfb 0..21, with long sfb bounds.
        let long_bounds = sfband_long(sample_rate);
        for sfb in 0..21 {
            if sfb < is_bound_sfb {
                continue;
            }
            let lo = long_bounds[sfb] as usize;
            let hi = long_bounds[sfb + 1] as usize;
            let is_pos = sf_r.l[sfb] as usize;
            apply_is_band(xr_l, xr_r, lo, hi, is_pos);
        }
    }
}

/// Apply the MPEG-1 IS coupling to the samples in `[lo, hi)`.
/// `is_pos == 7` is the "not intensity-coded" sentinel — leave R = 0.
fn apply_is_band(xr_l: &mut [f32; 576], xr_r: &mut [f32; 576], lo: usize, hi: usize, is_pos: usize) {
    if is_pos >= 7 {
        // Not IS-coded: R stays zero (encoder silenced it). L unchanged.
        // Defensively set R to 0 in this band.
        for i in lo..hi.min(576) {
            xr_r[i] = 0.0;
        }
        return;
    }
    let (k_l, k_r) = IS_RATIO_MPEG1[is_pos];
    for i in lo..hi.min(576) {
        let l = xr_l[i];
        xr_l[i] = l * k_l;
        xr_r[i] = l * k_r;
    }
}

/// Convert the sfb-based intensity-stereo bound into a sample-space
/// boundary for use with [`ms_stereo_range`]. For long blocks this is
/// simply `long_bounds[bound]`. For short / mixed blocks we return the
/// sample index where that sfb starts in the coefficient layout (post-
/// reorder, i.e. interleaved-by-window for short).
pub fn ms_boundary_sample(
    gc_r: &GranuleChannel,
    sample_rate: u32,
    is_bound_sfb: usize,
) -> usize {
    if gc_r.window_switching_flag && gc_r.block_type == 2 {
        let short_bounds = sfband_short(sample_rate);
        let long_bounds = sfband_long(sample_rate);
        let (long_end_sfb, long_end_sample) = if gc_r.mixed_block_flag {
            (8usize, long_bounds[8] as usize)
        } else {
            (0usize, 0usize)
        };
        if is_bound_sfb < long_end_sfb {
            return long_bounds[is_bound_sfb.min(22)] as usize;
        }
        // Short sfbs contribute 3 * width samples each to the reordered
        // layout. The first `is_bound_sfb - start_sfb` short sfbs go into
        // the MS region.
        let short_start_sfb = if gc_r.mixed_block_flag { 3 } else { 0 };
        let mut boundary = long_end_sample;
        let mut sfb = short_start_sfb;
        while sfb < is_bound_sfb.min(13) {
            let w = (short_bounds[sfb + 1] - short_bounds[sfb]) as usize;
            boundary += 3 * w;
            sfb += 1;
        }
        boundary.min(576)
    } else {
        let long_bounds = sfband_long(sample_rate);
        long_bounds[is_bound_sfb.min(22)] as usize
    }
}

/// Find the intensity-stereo boundary for a long-block granule: the
/// lowest sfb such that every R coefficient at or above that sfb's
/// starting sample is zero. This is the standard "is_bound" used to
/// partition M/S (below) from IS (at/above) when both modes are active.
///
/// For short / mixed blocks, we use a conservative approximation: the
/// rightmost sfb with any nonzero R coefficient, applied as an sfb
/// index across all three windows. This matches the common pattern
/// from minimp3 and libmad within a few sfbs and is correct for
/// encoder outputs where R is flat-zero above the IS bound.
pub fn find_is_bound_sfb(
    is_r: &[i32; 576],
    gc_r: &GranuleChannel,
    sample_rate: u32,
) -> usize {
    if gc_r.window_switching_flag && gc_r.block_type == 2 {
        // Short / mixed. Walk sfb from end down; the first sfb with a
        // nonzero R coeff in ANY window defines the bound (bound_sfb + 1).
        let short_bounds = sfband_short(sample_rate);
        let long_bounds = sfband_long(sample_rate);
        let (long_end_sfb, pos_start) = if gc_r.mixed_block_flag {
            (8usize, long_bounds[8] as usize)
        } else {
            (0usize, 0usize)
        };
        // Scan long portion (mixed) for any nonzero.
        let mut bound = 0usize;
        for sfb in 0..long_end_sfb {
            let lo = long_bounds[sfb] as usize;
            let hi = long_bounds[sfb + 1] as usize;
            if is_r[lo..hi.min(576)].iter().any(|&v| v != 0) {
                bound = sfb + 1;
            }
        }
        // Scan short region per-sfb × 3 windows.
        let short_start_sfb = if gc_r.mixed_block_flag { 3 } else { 0 };
        let mut sfb = short_start_sfb;
        let mut pos = pos_start;
        while sfb < 13 {
            let w = (short_bounds[sfb + 1] - short_bounds[sfb]) as usize;
            let region_end = pos + 3 * w;
            if region_end > 576 {
                break;
            }
            if is_r[pos..region_end].iter().any(|&v| v != 0) {
                bound = sfb.max(long_end_sfb) + 1;
            }
            pos = region_end;
            sfb += 1;
        }
        bound.min(13)
    } else {
        // Long block. Walk sfb 0..21 and find the rightmost nonzero.
        let long_bounds = sfband_long(sample_rate);
        let mut bound = 0usize;
        for sfb in 0..21 {
            let lo = long_bounds[sfb] as usize;
            let hi = long_bounds[sfb + 1] as usize;
            if is_r[lo..hi.min(576)].iter().any(|&v| v != 0) {
                bound = sfb + 1;
            }
        }
        bound.min(21)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requant_zero_stays_zero() {
        let is_ = [0i32; 576];
        let mut xr = [0.0f32; 576];
        let gc = GranuleChannel::default();
        let sf = ScaleFactors::default();
        requantize_granule(&is_, &mut xr, &gc, &sf, 44100);
        assert!(xr.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn ms_stereo_roundtrip() {
        let mut l = [0.0f32; 576];
        let mut r = [0.0f32; 576];
        l[0] = 0.7;
        r[0] = 0.3;
        // MS from L/R produces (L+R)/√2, (L-R)/√2.
        ms_stereo(&mut l, &mut r);
        let inv_sqrt2 = 1.0 / 2.0_f32.sqrt();
        assert!((l[0] - (0.7 + 0.3) * inv_sqrt2).abs() < 1e-5);
        assert!((r[0] - (0.7 - 0.3) * inv_sqrt2).abs() < 1e-5);
    }

    #[test]
    fn antialias_identity_on_zero_input() {
        let mut xr = [0.0f32; 576];
        let gc = GranuleChannel::default();
        antialias(&mut xr, &gc);
        assert!(xr.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn is_find_bound_long_block_all_zero_r_returns_zero() {
        let is_r = [0i32; 576];
        let gc = GranuleChannel::default();
        // All zero R → IS bound = 0 (no R content anywhere → everything IS-coded).
        assert_eq!(find_is_bound_sfb(&is_r, &gc, 44_100), 0);
    }

    #[test]
    fn is_find_bound_long_block_detects_rightmost_nonzero() {
        let mut is_r = [0i32; 576];
        // SFB_LONG_44100[5] = 20 .. SFB_LONG_44100[6] = 24. Place a
        // nonzero at index 22 (in sfb 5).
        is_r[22] = 1;
        let gc = GranuleChannel::default();
        let bound = find_is_bound_sfb(&is_r, &gc, 44_100);
        assert_eq!(bound, 6, "bound should be the sfb AFTER the last nonzero");
    }

    #[test]
    fn is_band_pos_3_equal_split() {
        // is_pos = 3 → (k_l, k_r) = (0.5, 0.5).
        let mut l = [0.0f32; 576];
        let mut r = [0.0f32; 576];
        l[200] = 2.0;
        r[200] = -7.0; // should be overwritten by IS
        let (k_l, k_r) = IS_RATIO_MPEG1[3];
        assert!((k_l - 0.5).abs() < 1e-6);
        assert!((k_r - 0.5).abs() < 1e-6);
        apply_is_band(&mut l, &mut r, 200, 201, 3);
        assert!((l[200] - 1.0).abs() < 1e-6);
        assert!((r[200] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn is_band_pos_7_zeroes_r() {
        // is_pos = 7 → "not IS coded" sentinel. L untouched, R forced 0.
        let mut l = [0.0f32; 576];
        let mut r = [0.0f32; 576];
        l[50] = 1.5;
        r[50] = 99.0; // should be wiped
        apply_is_band(&mut l, &mut r, 50, 51, 7);
        assert_eq!(l[50], 1.5);
        assert_eq!(r[50], 0.0);
    }

    #[test]
    fn is_long_block_applies_only_above_bound() {
        // Set up a long-block granule at 44.1 kHz, R all zero above sfb 5
        // (so `find_is_bound_sfb` returns 6 if we seed one nonzero in sfb 5).
        let mut xr_l = [0.0f32; 576];
        let mut xr_r = [0.0f32; 576];
        // L has energy at index 400 (well above sfb 6 start = 24).
        xr_l[400] = 0.8;
        xr_r[400] = 0.1; // pre-IS garbage — IS should overwrite.
        let gc = GranuleChannel::default();
        let mut sf = ScaleFactors::default();
        // For long block, set sf.l[sfb] (= is_pos) = 3 for all sfbs ≥ 6.
        for sfb in 6..21 {
            sf.l[sfb] = 3;
        }
        intensity_stereo_mpeg1(&mut xr_l, &mut xr_r, &sf, &gc, 44_100, 6);
        // At is_pos=3 → (0.5, 0.5): L[400] = 0.8 * 0.5 = 0.4, R[400] = 0.8 * 0.5 = 0.4.
        assert!((xr_l[400] - 0.4).abs() < 1e-6);
        assert!((xr_r[400] - 0.4).abs() < 1e-6);
    }

    #[test]
    fn ms_boundary_sample_long_block() {
        let gc = GranuleChannel::default();
        // bound sfb 6 at 44.1k → SFB_LONG_44100[6] = 24.
        assert_eq!(ms_boundary_sample(&gc, 44_100, 6), 24);
        assert_eq!(ms_boundary_sample(&gc, 44_100, 0), 0);
        // Out-of-range sfb clamps to the end of the table.
        assert!(ms_boundary_sample(&gc, 44_100, 30) <= 576);
    }
}
