//! Lightweight per-scalefactor-band masking model used by the VBR
//! encoder.
//!
//! This is **not** a full ISO 11172-3 §C.1 (psychoacoustic model 1)
//! implementation — that needs an FFT-based tonality detector, the
//! spreading function over Bark partitions, the absolute hearing
//! threshold table, and a separate analysis window pre-FFT. We do
//! something simpler that's still spec-grounded enough to drive a
//! VBR quantizer:
//!
//! 1. Take the 576-coefficient MDCT granule the encoder already
//!    produced.
//! 2. Split it across the 22 long-block scalefactor bands defined
//!    in [`crate::sfband`] (ISO 11172-3 Table 3-B.8).
//! 3. For each band, compute the signal energy E_b = sum xr[i]^2 and
//!    the "mask threshold" T_b = E_b / mask_ratio(quality), where the
//!    `mask_ratio` is a quality-dependent attenuation (higher
//!    quality = lower allowed noise = mask sits further below the
//!    band energy).
//! 4. The encoder's quantizer estimates per-band quantization noise
//!    N_b ~ (band_width) * step_size^2 / 12 — uniform-quantizer
//!    noise approximation. The granule is "masked" when N_b <= T_b
//!    for every band; equivalently, log10(N_b/T_b) <= 0.
//!
//! The "mask_ratio" for VBR quality 0..=9 is borrowed in spirit from
//! the V0..V9 LAME convention without ever reading their code: V0
//! (highest quality, smallest noise budget) maps to a large mask
//! ratio (~30 dB SNR target per band), V9 to a small ratio (~6 dB).
//!
//! This is enough to (a) drop quantizer effort on near-silent bands
//! and (b) push more bits into bands where the signal energy is
//! large. It is NOT enough to model post-mask, simultaneous-mask,
//! or absolute-threshold-of-hearing effects — but those would be
//! quality refinements on top of this same iteration loop.

use crate::sfband::sfband_long;

/// Per-granule masking estimate for the 22 long-block sfb bands.
#[derive(Clone, Debug)]
pub struct GranuleMask {
    /// Band energy E_b = sum_{i in band} xr[i]^2.
    pub energy: [f32; 22],
    /// Band masking threshold T_b. Quantization noise N_b that exceeds
    /// T_b in any band is audible per the (simplified) model.
    pub threshold: [f32; 22],
    /// Number of MDCT coefficients in each band — used by the noise
    /// estimator to compute total band noise from the per-coefficient
    /// step.
    pub width: [u16; 22],
    /// Starting offset of each band into the 576-coefficient granule.
    pub start: [u16; 22],
}

impl GranuleMask {
    /// Build a mask estimate for `xr` at `sample_rate`. `mask_ratio`
    /// (>= 1.0) is the per-band SNR target as a *linear* ratio: the
    /// allowed quantization noise per band is E_b / mask_ratio.
    /// Higher mask_ratio = stricter target = more bits.
    pub fn analyze(xr: &[f32; 576], sample_rate: u32, mask_ratio: f32) -> Self {
        let sfb = sfband_long(sample_rate);
        // We only need 22 bands (the last entry is the sentinel).
        let mut energy = [0.0f32; 22];
        let mut width = [0u16; 22];
        let mut start = [0u16; 22];
        for b in 0..22 {
            let s = sfb[b] as usize;
            let e = sfb[b + 1] as usize;
            let e = e.min(576);
            let s = s.min(e);
            let mut acc = 0.0f32;
            for i in s..e {
                acc += xr[i] * xr[i];
            }
            energy[b] = acc;
            width[b] = (e - s) as u16;
            start[b] = s as u16;
        }
        let mut threshold = [0.0f32; 22];
        // Floor to keep the mask above pure-silence noise — without it
        // a near-silent band has T_b = 0 and the iterator can never
        // satisfy it.
        let global_floor = energy.iter().copied().fold(0.0f32, f32::max) * 1.0e-7 + 1.0e-12;
        for b in 0..22 {
            threshold[b] = (energy[b] / mask_ratio.max(1.0)).max(global_floor);
        }
        Self {
            energy,
            threshold,
            width,
            start,
        }
    }

    /// Estimate per-band noise for a uniform quantizer with step `q`
    /// applied to MDCT coefficients (i.e. the inverse of the
    /// `2^((global_gain - 210)/4)` factor in the encoder). Uniform
    /// quantizer noise variance per coefficient = step^2 / 12.
    pub fn estimate_noise(&self, q: f32) -> [f32; 22] {
        let var = q * q / 12.0;
        let mut n = [0.0f32; 22];
        for b in 0..22 {
            n[b] = var * self.width[b] as f32;
        }
        n
    }

    /// Worst-case (largest) value of log10(noise / threshold) across
    /// bands with non-trivial energy. Returns f32::NEG_INFINITY when
    /// every band is silent. A return value <= 0 means the granule
    /// is masked at the current step.
    pub fn worst_nmr_db(&self, q: f32) -> f32 {
        let noise = self.estimate_noise(q);
        let mut worst = f32::NEG_INFINITY;
        for b in 0..22 {
            // Skip the truly-empty bands (no energy → no perceptual
            // contribution).
            if self.energy[b] <= 1.0e-20 {
                continue;
            }
            let r = noise[b] / self.threshold[b];
            let db = 10.0 * (r.max(1.0e-30)).log10();
            if db > worst {
                worst = db;
            }
        }
        worst
    }
}

/// Map a 0..=9 VBR quality scalar to a per-band SNR target (linear
/// energy ratio). Quality 0 = best (~30 dB SNR target); quality 9 =
/// smallest files (~6 dB SNR target). This is a smooth interpolation
/// across the V0..V9 range without copying any specific external
/// encoder's table.
pub fn vbr_quality_to_mask_ratio(q: u8) -> f32 {
    let q = q.min(9) as f32;
    // SNR_db: q=0 → 30 dB, q=9 → 6 dB.
    let snr_db = 30.0 - (q / 9.0) * 24.0;
    10.0_f32.powf(snr_db / 10.0)
}

/// Convert a global_gain value to its quantizer step in MDCT-coefficient
/// units. Mirrors the inverse of the encoder's
/// `is = (|xr| / step)^(3/4)` step relationship at the linearisation
/// point (small-signal limit). step = 2^((global_gain - 210)/4).
pub fn global_gain_to_step(global_gain: u8) -> f32 {
    let exp = ((global_gain as i32) - 210) as f32 / 4.0;
    2.0_f32.powf(exp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silence_has_zero_energy_and_floor_threshold() {
        let xr = [0.0f32; 576];
        let m = GranuleMask::analyze(&xr, 44_100, vbr_quality_to_mask_ratio(2));
        for b in 0..22 {
            assert_eq!(m.energy[b], 0.0);
            assert!(m.threshold[b] > 0.0); // floor kicks in
        }
        // worst_nmr_db should be NEG_INFINITY (no band has energy).
        assert_eq!(m.worst_nmr_db(1.0), f32::NEG_INFINITY);
    }

    #[test]
    fn mask_ratio_is_monotonic_in_quality() {
        let r0 = vbr_quality_to_mask_ratio(0);
        let r5 = vbr_quality_to_mask_ratio(5);
        let r9 = vbr_quality_to_mask_ratio(9);
        assert!(r0 > r5 && r5 > r9);
    }

    #[test]
    fn nonzero_signal_has_band_energy() {
        let mut xr = [0.0f32; 576];
        // Drop a single coefficient at index 100 — should land in
        // band ~10ish at 44.1k.
        xr[100] = 0.5;
        let m = GranuleMask::analyze(&xr, 44_100, vbr_quality_to_mask_ratio(2));
        let e: f32 = m.energy.iter().sum();
        assert!(e > 0.0);
    }

    #[test]
    fn finer_quantizer_means_lower_noise_db() {
        let mut xr = [0.0f32; 576];
        for i in 0..200 {
            xr[i] = 0.1;
        }
        let m = GranuleMask::analyze(&xr, 44_100, vbr_quality_to_mask_ratio(5));
        // Larger step (higher gain) → larger NMR.
        let nmr_low = m.worst_nmr_db(global_gain_to_step(120));
        let nmr_high = m.worst_nmr_db(global_gain_to_step(180));
        assert!(
            nmr_high > nmr_low,
            "expected higher gain to yield higher NMR; got nmr_low={nmr_low} nmr_high={nmr_high}"
        );
    }

    #[test]
    fn global_gain_step_is_doubling() {
        let s1 = global_gain_to_step(210);
        let s2 = global_gain_to_step(214);
        // 4 gain units = 1 step doubling.
        assert!((s2 / s1 - 2.0).abs() < 1e-5);
    }
}
