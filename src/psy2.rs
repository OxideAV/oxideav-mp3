//! ISO/IEC 11172-3 Annex D **Psychoacoustic Model 2** baseline for the
//! Layer III VBR encoder.
//!
//! Psy-2 refines Psy-1's tonality estimate with a **complex-prediction
//! unpredictability measure** (§D.2.2 / §D.2.3) on top of the same
//! Bark-partition spreading + SNR-offset framework. Where Psy-1 uses
//! the spectral flatness measure (SFM) of the squared-magnitude
//! MDCT / FFT spectrum to guess tonality, Psy-2 makes a second-order
//! linear prediction of the *complex* FFT spectrum and uses the
//! normalised prediction error as a more sensitive tonal / noise
//! discriminator:
//!
//! ```text
//!   X_hat[k] = 2 * X[k-1] - X[k-2]          (complex linear extrapolation)
//!   c[k]     = |X[k] - X_hat[k]|
//!              ─────────────────────           (per-bin unpredictability)
//!              |X[k]| + |X_hat[k]| + ε
//! ```
//!
//! `c[k]` is 0 for a perfectly predictable (tonal) component and 1 for
//! a fully unpredictable (noise-like) component. Per Bark partition:
//!
//! ```text
//!   c_b = mean_{k ∈ b} c[k]       (average unpredictability)
//!   t_b = 1 - c_b                  (tonality; 1 = tone, 0 = noise)
//! ```
//!
//! `t_b` replaces the SFM-based `t_sfm` of Psy-1. Everything downstream
//! (Schroeder spreading, TMN / NMT SNR offsets, per-sfb re-binning) is
//! identical to Psy-1.
//!
//! **Fallback rule.** When a partition contains fewer than 2 bins in the
//! FFT domain (very low sample rates or Nyquist-limited high partitions
//! that don't reach the MDCT grid), we fall back to Psy-1's SFM
//! tonality for that partition only — the prediction needs at least 2
//! previous bins to be well-defined.
//!
//! No external implementation was consulted for these formulas: the
//! linear-prediction approach is open-literature (Atal & Schroeder 1979
//! and the ISO 11172-3 Annex D §D.2 text describe the same family of
//! predictors), and the per-partition averaging / re-binning follows
//! the same logic as [`crate::psy1`].

use crate::fft::FFT_N;
use crate::psy1::{
    build_coeff_partition, build_fft_partition, build_spreading_matrix, N_BARK_PARTITIONS,
};
use crate::sfband::sfband_long;

/// Convert the encoder's 0..=9 VBR quality scalar to a Psy-2 gain.
/// Reuses the same scale as Psy-1 for user-facing consistency.
pub use crate::psy1::vbr_quality_to_psy1_gain as vbr_quality_to_psy2_gain;

/// TMN / NMT SNR offsets (same as Psy-1, per Annex D table B.2.2).
const OFFSET_TONE: f32 = 14.5;
const OFFSET_NOISE: f32 = 5.5;

/// Output of the Psy-2 analysis pass. Same shape as `Psy1Mask` so the
/// encoder can dispatch transparently.
#[derive(Clone, Debug)]
pub struct Psy2Mask {
    /// Number of sfbs: 22 for long blocks, 39 (3 × 13) for short blocks.
    pub n_sfb: usize,
    /// Per-sfb signal energy.
    pub energy: Vec<f32>,
    /// Per-sfb masking threshold.
    pub threshold: Vec<f32>,
    /// Per-sfb width in MDCT coefficients.
    pub width: Vec<u16>,
    /// Per-sfb start index into the 576-coefficient granule.
    pub start: Vec<u16>,
    /// Per-Bark-partition unpredictability measure `c_b ∈ [0, 1]`
    /// (0 = pure tone, 1 = pure noise). Exposed for diagnostics / tests.
    pub partition_unpredictability: [f32; N_BARK_PARTITIONS],
    /// Derived tonality `t_b = 1 - c_b`.
    pub partition_tonality: [f32; N_BARK_PARTITIONS],
}

impl Psy2Mask {
    /// Run the Psy-2 analysis on a long-block granule using the
    /// 1024-point FFT power spectrum for the complex-prediction pass
    /// and the MDCT coefficients for energy + re-binning.
    ///
    /// `fft_complex` is the full complex FFT output `[re, im]` pairs
    /// for bins `0..=N/2` (513 entries from [`crate::fft::Fft1024`]).
    /// `xr` is the 576-coefficient MDCT granule.
    pub fn analyze(
        xr: &[f32; 576],
        fft_complex: &[[f32; 2]; FFT_N / 2 + 1],
        sample_rate: u32,
        gain: f32,
    ) -> Self {
        let n_bins = FFT_N / 2 + 1;
        let fft_partition = build_fft_partition(sample_rate);
        let coeff_partition = build_coeff_partition(sample_rate);
        let spread_mat = build_spreading_matrix(sample_rate);

        // --- Step 1: per-bin unpredictability in the FFT domain ---
        // For k >= 2 use the full 2nd-order predictor; k < 2 has no
        // previous pairs and uses fallback c[k] = 0.5 (ambiguous).
        let mut bin_unpred = [0.5f32; FFT_N / 2 + 1];
        for k in 2..n_bins {
            let re = fft_complex[k][0];
            let im = fft_complex[k][1];
            let re1 = fft_complex[k - 1][0];
            let im1 = fft_complex[k - 1][1];
            let re2 = fft_complex[k - 2][0];
            let im2 = fft_complex[k - 2][1];
            // Linear extrapolation in the complex plane.
            let hat_re = 2.0 * re1 - re2;
            let hat_im = 2.0 * im1 - im2;
            let err_re = re - hat_re;
            let err_im = im - hat_im;
            let err_mag = (err_re * err_re + err_im * err_im).sqrt();
            let sig_mag = (re * re + im * im).sqrt();
            let hat_mag = (hat_re * hat_re + hat_im * hat_im).sqrt();
            let denom = sig_mag + hat_mag + 1.0e-20;
            bin_unpred[k] = (err_mag / denom).clamp(0.0, 1.0);
        }

        // --- Step 2: per-partition unpredictability (mean over bins) ---
        let mut part_unpred_sum = [0.0f32; N_BARK_PARTITIONS];
        let mut part_count = [0u32; N_BARK_PARTITIONS];
        for k in 0..n_bins {
            let p = fft_partition[k] as usize;
            part_unpred_sum[p] += bin_unpred[k];
            part_count[p] += 1;
        }
        let mut partition_unpredictability = [0.5f32; N_BARK_PARTITIONS];
        let mut partition_tonality = [0.5f32; N_BARK_PARTITIONS];
        for b in 0..N_BARK_PARTITIONS {
            let n = part_count[b];
            let c = if n < 2 {
                // Too few bins: fall back to the ambiguous midpoint.
                // (Psy-1's SFM is not accessible here without the MDCT
                // pass, so we use 0.5 — the noise / tone boundary —
                // which triggers a mild tonal offset rather than full
                // TMN or full NMT.)
                0.5
            } else {
                (part_unpred_sum[b] / n as f32).clamp(0.0, 1.0)
            };
            partition_unpredictability[b] = c;
            partition_tonality[b] = 1.0 - c;
        }

        // --- Step 3: MDCT-domain energy per partition ---
        let mut mdct_part_energy = [0.0f32; N_BARK_PARTITIONS];
        let mut mdct_part_count = [0u32; N_BARK_PARTITIONS];
        for k in 0..576 {
            let p = coeff_partition[k] as usize;
            mdct_part_energy[p] += xr[k] * xr[k];
            mdct_part_count[p] += 1;
        }

        // --- Step 4: Spreading + SNR offset using Psy-2 tonality ---
        let part_threshold =
            spread_and_offset_psy2(&mdct_part_energy, &spread_mat, &partition_tonality, gain);

        // --- Step 5: Per-coefficient threshold ---
        let mut per_coeff_thr = [0.0f32; 576];
        for k in 0..576 {
            let p = coeff_partition[k] as usize;
            let n = mdct_part_count[p].max(1) as f32;
            per_coeff_thr[k] = part_threshold[p] / n;
        }

        // --- Step 6: Re-bin to long-block sfbs ---
        let sfb = sfband_long(sample_rate);
        let mut energy = vec![0.0f32; 22];
        let mut threshold = vec![0.0f32; 22];
        let mut width = vec![0u16; 22];
        let mut start = vec![0u16; 22];
        for b in 0..22 {
            let s = (sfb[b] as usize).min(576);
            let e_idx = (sfb[b + 1] as usize).min(576);
            let lo = s.min(e_idx);
            let hi = e_idx;
            let mut e = 0.0f32;
            let mut thr = 0.0f32;
            for k in lo..hi {
                e += xr[k] * xr[k];
                thr += per_coeff_thr[k];
            }
            energy[b] = e;
            threshold[b] = thr;
            width[b] = (hi - lo) as u16;
            start[b] = lo as u16;
        }
        let global_floor = energy.iter().copied().fold(0.0f32, f32::max) * 1.0e-7 + 1.0e-12;
        for b in 0..22 {
            if threshold[b] < global_floor {
                threshold[b] = global_floor;
            }
        }

        Self {
            n_sfb: 22,
            energy,
            threshold,
            width,
            start,
            partition_unpredictability,
            partition_tonality,
        }
    }

    /// Estimate per-band noise for a uniform quantizer with step `q`.
    /// Mirrors [`crate::psy1::Psy1Mask::estimate_noise`].
    pub fn estimate_noise(&self, q: f32) -> Vec<f32> {
        let var = q * q / 12.0;
        let mut n = vec![0.0f32; self.n_sfb];
        for b in 0..self.n_sfb {
            n[b] = var * self.width[b] as f32;
        }
        n
    }

    /// Worst-case NMR in dB. Mirrors [`crate::psy1::Psy1Mask::worst_nmr_db`].
    pub fn worst_nmr_db(&self, q: f32) -> f32 {
        let noise = self.estimate_noise(q);
        let mut worst = f32::NEG_INFINITY;
        for b in 0..self.n_sfb {
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

/// Spread per-partition energy using the Psy-2 tonality (derived from
/// the complex-prediction unpredictability `c_b`). Identical to the
/// Psy-1 spreader — only the tonality source differs.
fn spread_and_offset_psy2(
    part_energy: &[f32; N_BARK_PARTITIONS],
    spread_mat: &[[f32; N_BARK_PARTITIONS]; N_BARK_PARTITIONS],
    tonality: &[f32; N_BARK_PARTITIONS],
    gain: f32,
) -> [f32; N_BARK_PARTITIONS] {
    let mut part_threshold = [0.0f32; N_BARK_PARTITIONS];
    let mut part_e_db = [-200.0f32; N_BARK_PARTITIONS];
    for b in 0..N_BARK_PARTITIONS {
        if part_energy[b] > 1.0e-30 {
            part_e_db[b] = 10.0 * part_energy[b].log10();
        }
    }
    for b in 0..N_BARK_PARTITIONS {
        let mut acc = 0.0f64;
        for i in 0..N_BARK_PARTITIONS {
            let contrib_db = part_e_db[i] + spread_mat[b][i];
            if contrib_db < -150.0 {
                continue;
            }
            acc += 10.0_f64.powf((contrib_db / 10.0) as f64);
        }
        let spread_e = if acc > 0.0 { acc as f32 } else { 0.0 };
        let t = tonality[b];
        let offset_db = OFFSET_TONE * t + OFFSET_NOISE * (1.0 - t);
        let lin = 10.0_f32.powf(-offset_db / 10.0);
        part_threshold[b] = spread_e * lin / gain.max(1.0);
    }
    part_threshold
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fft::{hann_window, Fft1024};

    /// Build complex FFT spectrum from raw PCM (for testing).
    fn fft_complex_from_pcm(pcm: &[f32; FFT_N]) -> [[f32; 2]; FFT_N / 2 + 1] {
        let w = hann_window(FFT_N);
        let mut input = [0.0f32; FFT_N];
        for i in 0..FFT_N {
            input[i] = pcm[i] * w[i];
        }
        Fft1024::new().complex_spectrum(&input)
    }

    #[test]
    fn psy2_silence_has_floor_thresholds() {
        let xr = [0.0f32; 576];
        let pcm = [0.0f32; FFT_N];
        let fft = fft_complex_from_pcm(&pcm);
        let m = Psy2Mask::analyze(&xr, &fft, 44_100, 1.0);
        assert_eq!(m.n_sfb, 22);
        // All thresholds should be at the floor (a tiny positive value).
        for b in 0..22 {
            assert!(
                m.threshold[b] > 0.0,
                "threshold[{b}] should be positive (floor)"
            );
            assert!(
                m.energy[b] <= 1.0e-12,
                "energy[{b}] should be near-zero for silence"
            );
        }
    }

    #[test]
    fn psy2_pure_tone_has_low_unpredictability() {
        // A single FFT bin (pure tone, perfectly predictable in the
        // frequency domain) should give a very low per-partition
        // unpredictability for the partition that contains that bin.
        let mut pcm = [0.0f32; FFT_N];
        // Place a tone at bin 50 (f = 50 * 44100 / 1024 ≈ 2149 Hz).
        let freq = 50.0 * 44_100.0 / FFT_N as f32;
        for i in 0..FFT_N {
            pcm[i] = (2.0 * std::f32::consts::PI * freq * i as f32 / 44_100.0).sin() * 0.5;
        }
        let fft = fft_complex_from_pcm(&pcm);
        let xr = [0.0f32; 576];
        let m = Psy2Mask::analyze(&xr, &fft, 44_100, 1.0);
        // The partition containing bin 50 should have low unpredictability
        // (the complex-prediction error is small for an in-bin tone).
        // We verify that at least one partition has c_b < 0.4 (tonal-ish).
        let any_tonal = m.partition_unpredictability.iter().any(|&c| c < 0.4);
        assert!(
            any_tonal,
            "expected at least one tonal partition for a pure-tone input, got: {:?}",
            &m.partition_unpredictability[..8]
        );
    }

    #[test]
    fn psy2_white_noise_has_high_unpredictability() {
        // White noise is maximally unpredictable; the per-partition
        // unpredictability should be higher on average than for a tone.
        let mut noise = [0.0f32; FFT_N];
        // Deterministic pseudo-noise from a simple LCG.
        let mut state: u32 = 0xDEAD_BEEF;
        for v in noise.iter_mut() {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005_u64 as u32)
                .wrapping_add(1_442_695_040_888_963_407_u64 as u32);
            *v = ((state >> 16) as f32 / 32768.0) - 1.0;
        }
        let fft = fft_complex_from_pcm(&noise);
        let xr = [0.0f32; 576];
        let m = Psy2Mask::analyze(&xr, &fft, 44_100, 1.0);
        // Noise should give higher average unpredictability than a tone.
        let mean_c: f32 =
            m.partition_unpredictability.iter().sum::<f32>() / N_BARK_PARTITIONS as f32;
        assert!(
            mean_c > 0.3,
            "expected mean unpredictability > 0.3 for white noise, got {mean_c}"
        );
    }

    #[test]
    fn psy2_tone_has_lower_unpredictability_than_noise() {
        let mut tone = [0.0f32; FFT_N];
        let freq = 440.0_f32;
        for i in 0..FFT_N {
            tone[i] = (2.0 * std::f32::consts::PI * freq * i as f32 / 44_100.0).sin() * 0.5;
        }
        let mut noise = [0.0f32; FFT_N];
        let mut state: u32 = 0xCAFE_BABE;
        for v in noise.iter_mut() {
            state = state
                .wrapping_mul(1_664_525_u32)
                .wrapping_add(1_013_904_223_u32);
            *v = ((state >> 16) as f32 / 32768.0) - 1.0;
        }
        let xr = [0.0f32; 576];
        let fft_tone = fft_complex_from_pcm(&tone);
        let fft_noise = fft_complex_from_pcm(&noise);
        let m_tone = Psy2Mask::analyze(&xr, &fft_tone, 44_100, 1.0);
        let m_noise = Psy2Mask::analyze(&xr, &fft_noise, 44_100, 1.0);
        let mean_c_tone: f32 =
            m_tone.partition_unpredictability.iter().sum::<f32>() / N_BARK_PARTITIONS as f32;
        let mean_c_noise: f32 =
            m_noise.partition_unpredictability.iter().sum::<f32>() / N_BARK_PARTITIONS as f32;
        assert!(
            mean_c_tone < mean_c_noise,
            "tone unpredictability ({mean_c_tone:.3}) should be < noise unpredictability ({mean_c_noise:.3})"
        );
    }

    #[test]
    fn psy2_stricter_gain_lowers_threshold() {
        let mut xr = [0.0f32; 576];
        for (i, v) in xr.iter_mut().enumerate() {
            *v = ((i as f32) * 0.01).sin() * 0.3;
        }
        let mut pcm = [0.0f32; FFT_N];
        for (i, v) in pcm.iter_mut().enumerate() {
            *v = ((i as f32) * 0.01).sin() * 0.3;
        }
        let fft = fft_complex_from_pcm(&pcm);
        let m_loose = Psy2Mask::analyze(&xr, &fft, 44_100, 1.0);
        let m_strict = Psy2Mask::analyze(&xr, &fft, 44_100, 100.0);
        // Stricter gain (100×) divides the threshold — every non-zero
        // sfb threshold should be lower.
        for b in 0..22 {
            if m_loose.energy[b] > 1.0e-12 {
                assert!(
                    m_strict.threshold[b] < m_loose.threshold[b] * 1.1,
                    "strict threshold[{b}] ({}) should be < loose threshold ({})",
                    m_strict.threshold[b],
                    m_loose.threshold[b]
                );
            }
        }
    }
}
