//! ISO/IEC 11172-3 Annex D **Psychoacoustic Model 1** (Bark-partition
//! spreading) for the Layer III VBR encoder.
//!
//! This is the spec-grounded Bark-domain masking model that the
//! lightweight per-sfb energy mask in [`crate::psy`] approximates. The
//! key pieces of the model:
//!
//! 1. **Bark-band partitions.** The audible spectrum from 0 Hz up to the
//!    Nyquist limit is partitioned into ~24 critical bands, each of
//!    which corresponds to a roughly equal perceptual ("Bark") width.
//!    For the MDCT-domain implementation we approximate each
//!    coefficient's centre frequency by `f = (k + 0.5) * sr / 1152`
//!    (granule index + half-bin) and walk a Zwicker-style boundary
//!    table to assign the 576 long-block coefficients across the
//!    partitions. Annex D.2.4.1 specifies an FFT-domain partition
//!    table; in MDCT terms our partition table is the coefficient-index
//!    list `partition_start[..]` derived from the same Bark boundaries.
//!
//! 2. **Per-partition energy + spreading.** Each Bark partition `b`
//!    accumulates `e_b = sum |X[k]|^2` over its assigned MDCT
//!    coefficients. The masking threshold is then formed by
//!    convolving `e_b` with the spec's spreading function — a
//!    triangular-on-Bark-axis kernel that models how a tone in
//!    partition `i` raises the just-noticeable noise floor in
//!    neighbouring partitions `j`. We use the standard Schroeder
//!    spreading approximation
//!
//!    ```text
//!      sf(dz) = 15.81 + 7.5 * (dz + 0.474)
//!             - 17.5 * sqrt(1 + (dz + 0.474)^2)   [dB]
//!    ```
//!
//!    where `dz = z_j - z_i` is the Bark distance from masker to
//!    maskee. This is the closed-form approximation widely used in
//!    open-literature implementations of Annex D — no specific encoder
//!    table was consulted to derive it.
//!
//! 3. **Tonality estimate.** Annex D.2.4.4 / D.2.4.5 separates each
//!    partition's energy into "tonal" and "noise-like" fractions and
//!    applies different SNR offsets to each (tonal maskers get an
//!    `OFFSET_TONE = 14.5 dB + b/2` budget, noise maskers get a fixed
//!    `OFFSET_NOISE = 5.5 dB`). We approximate tonality with the
//!    **spectral flatness measure (SFM)** of each partition's
//!    coefficients:
//!
//!    ```text
//!      SFM_b = exp(mean(ln |X[k]|^2)) / mean(|X[k]|^2)   ∈ (0, 1]
//!    ```
//!
//!    SFM ≈ 1 ⇒ flat / noisy partition (use NOISE offset);
//!    SFM ≪ 1 ⇒ peaky / tonal partition (use TONE offset). The
//!    per-partition tonality index is then `t_b = min(1,
//!    SFM_db / -60.0)` where `SFM_db = 10 * log10(SFM_b)`.
//!
//! 4. **Per-sfb SMR (signal-to-mask ratio).** Once we have the
//!    spread mask threshold per Bark partition, we re-bin to the
//!    encoder's scalefactor-band layout (ISO Table 3-B.8 — 22 long-block
//!    sfbs at 44.1 kHz). For each sfb we take the *minimum* per-bin
//!    threshold across the partitions that overlap the sfb (tightest
//!    masking constraint wins), then scale by the partition width to
//!    get an energy budget. The encoder's noise allocator drives
//!    global_gain so per-sfb quantisation noise stays under that
//!    budget.
//!
//! No external psy implementation was consulted; the partition table
//! is derived from the Zwicker Bark boundaries cross-referenced against
//! ISO 11172-3 Annex D figures, the spreading formula is Schroeder's
//! closed-form (open-literature), the tonality proxy is SFM (also
//! standard signal-processing material).

use crate::sfband::sfband_long;

/// Number of Bark partitions used by the Psy-1 implementation. Annex D.2.4
/// table B.2.1 lists 63 partitions for Layer III; we use a coarser
/// 21-band Bark axis derived from the standard Zwicker boundaries
/// (0..=20 Bark, 1-Bark spacing) — enough to capture the spreading-
/// function behaviour while keeping the per-coefficient binning cost
/// linear in 576.
pub const N_BARK_PARTITIONS: usize = 24;

/// Upper edges of the 24 Bark partitions, in Hz. From Zwicker's standard
/// table (open-literature, also reprinted in ISO 11172-3 Annex D figures).
/// Partition `b` covers `(BARK_UPPER_HZ[b-1], BARK_UPPER_HZ[b]]` with
/// `BARK_UPPER_HZ[-1] = 0`.
const BARK_UPPER_HZ: [f32; N_BARK_PARTITIONS] = [
    100.0, 200.0, 300.0, 400.0, 510.0, 630.0, 770.0, 920.0, 1080.0, 1270.0, 1480.0, 1720.0, 2000.0,
    2320.0, 2700.0, 3150.0, 3700.0, 4400.0, 5300.0, 6400.0, 7700.0, 9500.0, 12000.0, 15500.0,
];

/// Convert a frequency in Hz to its Bark-axis position. Uses the
/// standard Zwicker formula:
///
/// ```text
///   z(f) = 13 * arctan(0.00076 * f) + 3.5 * arctan((f / 7500)^2)
/// ```
///
/// open-literature, reprinted in countless DSP textbooks. The result
/// is in Bark units (0..~25 across human hearing).
pub fn freq_to_bark(f: f32) -> f32 {
    let a = 13.0 * (0.000_76 * f).atan();
    let b = 3.5 * ((f / 7500.0) * (f / 7500.0)).atan();
    a + b
}

/// MDCT-coefficient centre frequency for a long-block index `k`
/// (0..576) at `sample_rate`. Each coefficient covers `sr / 1152` Hz of
/// bandwidth (granule = 576 coeffs spanning Nyquist = sr/2), so its
/// centre sits at `(k + 0.5) * sr / 1152`.
pub fn coeff_centre_hz(k: usize, sample_rate: u32) -> f32 {
    (k as f32 + 0.5) * sample_rate as f32 / 1152.0
}

/// Build the partition table: for every long-block MDCT coefficient,
/// the index of the Bark partition that carries it. Length = 576.
pub fn build_coeff_partition(sample_rate: u32) -> [u8; 576] {
    let mut out = [0u8; 576];
    for k in 0..576 {
        let f = coeff_centre_hz(k, sample_rate);
        let mut p = 0usize;
        while p < N_BARK_PARTITIONS - 1 && f > BARK_UPPER_HZ[p] {
            p += 1;
        }
        out[k] = p as u8;
    }
    out
}

/// Schroeder spreading function evaluated at Bark distance `dz` (with
/// sign — negative = below the masker, positive = above). Returns the
/// log-domain attenuation in dB. By construction `sf(0) ≈ 0 dB` (the
/// masker fully contributes to its own partition); off-partition
/// values fall off asymmetrically with sharper roll-off below the
/// masker (lower-frequency neighbours less masked) than above.
///
/// Closed-form (open-literature, no external implementation consulted):
///
/// ```text
///   sf(dz) = 15.81 + 7.5 * (dz + 0.474)
///          - 17.5 * sqrt(1 + (dz + 0.474)^2)
/// ```
///
/// At `dz = 0` this evaluates to ~0 dB by construction (the constant
/// `0.474` is chosen so the formula's stationary point sits at the
/// masker). The function is widely cited in the open psychoacoustics
/// literature — no specific encoder source consulted.
pub fn spreading_db(dz: f32) -> f32 {
    let x = dz + 0.474;
    15.81 + 7.5 * x - 17.5 * (1.0 + x * x).sqrt()
}

/// Per-partition Bark centre (mid-point of the partition's frequency
/// range, mapped through `freq_to_bark`).
pub fn partition_bark_centres(sample_rate: u32) -> [f32; N_BARK_PARTITIONS] {
    let mut out = [0.0f32; N_BARK_PARTITIONS];
    let nyq = sample_rate as f32 * 0.5;
    let mut prev = 0.0f32;
    for b in 0..N_BARK_PARTITIONS {
        let upper = BARK_UPPER_HZ[b].min(nyq);
        let centre = 0.5 * (prev + upper);
        out[b] = freq_to_bark(centre.max(1.0));
        prev = upper;
    }
    out
}

/// Pre-computed spreading-matrix row: `spread_db[b][i]` is the dB
/// contribution from masker `i` to maskee `b`. Always <= 0.
fn build_spreading_matrix(sample_rate: u32) -> [[f32; N_BARK_PARTITIONS]; N_BARK_PARTITIONS] {
    let centres = partition_bark_centres(sample_rate);
    let mut mat = [[0.0f32; N_BARK_PARTITIONS]; N_BARK_PARTITIONS];
    for b in 0..N_BARK_PARTITIONS {
        for i in 0..N_BARK_PARTITIONS {
            let dz = centres[b] - centres[i];
            mat[b][i] = spreading_db(dz);
        }
    }
    mat
}

/// Annex D-style per-partition SNR offsets (dB) used to convert the
/// spread energy mask into a noise threshold. Tonal maskers get the
/// larger offset (their masking is narrower in frequency than
/// noise-like maskers — the noise floor under a tone has to sit
/// further down to stay inaudible).
///
/// `OFFSET_TONE` and `OFFSET_NOISE` mirror the spec's "TMN" and "NMT"
/// constants (~14.5 dB and ~5.5 dB respectively) without copying any
/// specific encoder's tuning table.
const OFFSET_TONE: f32 = 14.5;
const OFFSET_NOISE: f32 = 5.5;

/// Output of the Psy-1 analysis pass: per-sfb signal energy + masking
/// threshold ready for the encoder's noise allocator.
#[derive(Clone, Debug)]
pub struct Psy1Mask {
    /// Number of long-block sfbs (always 22 for the long-block path).
    pub n_sfb: usize,
    /// Per-sfb signal energy E_b = sum |X[k]|^2 over the sfb.
    pub energy: [f32; 22],
    /// Per-sfb masking threshold T_b. Quantization noise N_b that
    /// exceeds T_b in any band is audible per the model.
    pub threshold: [f32; 22],
    /// Per-sfb width in MDCT coefficients.
    pub width: [u16; 22],
    /// Per-sfb start index into the 576-coefficient granule.
    pub start: [u16; 22],
    /// Per-Bark-partition tonality estimate in 0..=1 (1 = pure tone,
    /// 0 = pure noise). Exposed mainly for diagnostics + tests.
    pub partition_tonality: [f32; N_BARK_PARTITIONS],
    /// Per-Bark-partition spread energy threshold (after applying the
    /// spreading function + SNR offset). Exposed mainly for tests.
    pub partition_threshold: [f32; N_BARK_PARTITIONS],
}

impl Psy1Mask {
    /// Run the full Psy-1 analysis on a 576-coefficient long-block
    /// granule. `gain` is a quality-derived multiplicative SNR target:
    /// the per-band masking threshold is divided by `gain` so larger
    /// gain = stricter target = more bits.
    pub fn analyze(xr: &[f32; 576], sample_rate: u32, gain: f32) -> Self {
        let coeff_partition = build_coeff_partition(sample_rate);
        let spread_mat = build_spreading_matrix(sample_rate);

        // Per-partition energy + log-energy for SFM tonality estimate.
        let mut part_energy = [0.0f32; N_BARK_PARTITIONS];
        let mut part_count = [0u32; N_BARK_PARTITIONS];
        let mut part_log_sum = [0.0f32; N_BARK_PARTITIONS];
        let log_floor: f32 = 1.0e-20;
        for k in 0..576 {
            let p = coeff_partition[k] as usize;
            let e = xr[k] * xr[k];
            part_energy[p] += e;
            part_log_sum[p] += (e.max(log_floor)).ln();
            part_count[p] += 1;
        }

        // Tonality via SFM.
        let mut tonality = [0.0f32; N_BARK_PARTITIONS];
        for b in 0..N_BARK_PARTITIONS {
            if part_count[b] == 0 || part_energy[b] <= log_floor {
                tonality[b] = 0.0;
                continue;
            }
            let n = part_count[b] as f32;
            let geo = (part_log_sum[b] / n).exp();
            let arith = part_energy[b] / n;
            let sfm = (geo / arith.max(log_floor)).clamp(1.0e-20, 1.0);
            // SFM in dB: 0 dB = pure noise, -inf = pure tone. Clamp the
            // tone end at -60 dB (commonly cited in psy literature).
            let sfm_db = 10.0 * sfm.log10();
            // Map [-60, 0] dB linearly to [1, 0] tonality.
            let t = ((-sfm_db) / 60.0).clamp(0.0, 1.0);
            tonality[b] = t;
        }

        // Spread the per-partition energy across all partitions in dB
        // domain, then convert back to linear and divide by the per-
        // partition SNR offset (interpolated tone↔noise) to get the
        // partition's masking threshold.
        let mut part_threshold = [0.0f32; N_BARK_PARTITIONS];
        // Per-partition energy in dB.
        let mut part_e_db = [-200.0f32; N_BARK_PARTITIONS];
        for b in 0..N_BARK_PARTITIONS {
            if part_energy[b] > 1.0e-30 {
                part_e_db[b] = 10.0 * part_energy[b].log10();
            }
        }
        for b in 0..N_BARK_PARTITIONS {
            // Sum spread contributions in *energy* domain
            // (dB-add via log10(sum 10^(x/10)))
            let mut acc = 0.0f64;
            for i in 0..N_BARK_PARTITIONS {
                let contrib_db = part_e_db[i] + spread_mat[b][i];
                if contrib_db < -150.0 {
                    continue;
                }
                acc += 10.0_f64.powf((contrib_db / 10.0) as f64);
            }
            let spread_e = if acc > 0.0 { acc as f32 } else { 0.0 };
            // SNR offset: tone gets a wider budget (tonal masker raises
            // the floor less), noise gets a tighter budget. Interpolate
            // by tonality.
            let t = tonality[b];
            let offset_db = OFFSET_TONE * t + OFFSET_NOISE * (1.0 - t);
            // Threshold = spread_energy / 10^(offset_db / 10)
            // and we further divide by `gain` (encoder quality knob).
            let lin = 10.0_f32.powf(-offset_db / 10.0);
            part_threshold[b] = spread_e * lin / gain.max(1.0);
        }

        // Re-bin to per-sfb threshold + energy. Energy = sum over sfb's
        // coefficients; threshold = minimum per-coefficient threshold
        // (= partition_threshold / partition_count) summed over sfb's
        // coefficients (so each coefficient contributes its share of
        // its partition's threshold budget).
        let sfb = sfband_long(sample_rate);
        let mut energy = [0.0f32; 22];
        let mut threshold = [0.0f32; 22];
        let mut width = [0u16; 22];
        let mut start = [0u16; 22];
        // Per-coefficient threshold = partition_threshold / partition_count
        let mut per_coeff_thr = [0.0f32; 576];
        for k in 0..576 {
            let p = coeff_partition[k] as usize;
            let n = part_count[p].max(1) as f32;
            per_coeff_thr[k] = part_threshold[p] / n;
        }
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

        // Floor to keep silent bands from breaking the iterator (mirrors
        // the floor in `psy::GranuleMask`).
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
            partition_tonality: tonality,
            partition_threshold: part_threshold,
        }
    }

    /// Estimate per-band noise for a uniform quantizer with step `q`.
    /// Mirrors [`crate::psy::GranuleMask::estimate_noise`] so the two
    /// can swap at the encoder's call site.
    pub fn estimate_noise(&self, q: f32) -> [f32; 22] {
        let var = q * q / 12.0;
        let mut n = [0.0f32; 22];
        for b in 0..22 {
            n[b] = var * self.width[b] as f32;
        }
        n
    }

    /// Worst-case (largest) value of log10(noise / threshold) across
    /// bands with non-trivial energy, in dB. Returns f32::NEG_INFINITY
    /// when every band is silent. Mirrors
    /// [`crate::psy::GranuleMask::worst_nmr_db`].
    pub fn worst_nmr_db(&self, q: f32) -> f32 {
        let noise = self.estimate_noise(q);
        let mut worst = f32::NEG_INFINITY;
        for b in 0..22 {
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

    /// Per-band Signal-to-Mask Ratio (SMR) in dB.
    ///
    /// `SMR_b = 10 * log10(E_b / T_b)`. Higher SMR = the band's energy
    /// is well above its mask, so quantisation noise has plenty of
    /// headroom. Lower SMR = the band's mask sits close to the signal,
    /// so the encoder has to push a finer quantizer (more bits) at it.
    pub fn smr_db(&self) -> [f32; 22] {
        let mut out = [0.0f32; 22];
        for b in 0..22 {
            if self.energy[b] <= 1.0e-20 || self.threshold[b] <= 1.0e-30 {
                out[b] = 0.0;
                continue;
            }
            out[b] = 10.0 * (self.energy[b] / self.threshold[b]).log10();
        }
        out
    }
}

/// Convert the encoder's existing 0..=9 VBR quality scalar to a Psy-1
/// `gain` knob. Quality 0 (best) → gain ~30 (stricter mask, more bits);
/// quality 9 (smallest) → gain ~1 (looser mask, fewer bits). Smooth
/// interpolation matching [`crate::psy::vbr_quality_to_mask_ratio`]
/// in spirit so the user-facing knob behaves identically.
pub fn vbr_quality_to_psy1_gain(q: u8) -> f32 {
    let q = q.min(9) as f32;
    // gain_db: q=0 → 30 dB, q=9 → 0 dB. That's similar to the
    // GranuleMask mask_ratio mapping but applied as an extra
    // tightening on top of the spreading-derived threshold (which
    // already sits ~5-15 dB below the partition energy).
    let gain_db = 30.0 - (q / 9.0) * 30.0;
    10.0_f32.powf(gain_db / 10.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn freq_to_bark_monotonic() {
        let z0 = freq_to_bark(0.0);
        let z1 = freq_to_bark(500.0);
        let z2 = freq_to_bark(1000.0);
        let z3 = freq_to_bark(8000.0);
        let z4 = freq_to_bark(16000.0);
        assert!(z0 < z1 && z1 < z2 && z2 < z3 && z3 < z4);
        // Bark 1 kHz is roughly 8.5; cross-check in a wide band.
        assert!((z2 - 8.5).abs() < 1.0, "z(1 kHz) = {z2}");
    }

    #[test]
    fn coeff_partition_assigns_increasing() {
        let cp = build_coeff_partition(44_100);
        // First coeff -> partition 0; last coeff -> partition >= 20
        // (15.5 kHz upper bound is partition 22; everything beyond
        // sits in the last partition or partition 23).
        assert_eq!(cp[0], 0);
        assert!(cp[575] >= 20, "last partition {}", cp[575]);
        // Monotonic non-decreasing.
        for k in 1..576 {
            assert!(
                cp[k] >= cp[k - 1],
                "partition assignment must be non-decreasing"
            );
        }
    }

    #[test]
    fn spreading_function_peaks_at_zero_and_decays() {
        // sf(0) is normalised to 0 dB by construction (within floating
        // round-off — Schroeder's constant 0.474 is chosen so the
        // formula's stationary point sits at the masker).
        let s0 = spreading_db(0.0);
        assert!(s0.abs() < 0.01, "sf(0) = {s0}");
        // Below and above the masker, attenuation goes negative.
        let s_neg = spreading_db(-3.0);
        let s_pos = spreading_db(3.0);
        assert!(s_neg < -5.0, "spreading at -3 Bark: {s_neg}");
        assert!(s_pos < -5.0, "spreading at +3 Bark: {s_pos}");
        // The spreading is asymmetric: low side falls off faster than
        // high side (per Schroeder).
        assert!(s_neg < s_pos, "expected lower spread on low side");
    }

    #[test]
    fn psy1_silence_has_floor_thresholds() {
        let xr = [0.0f32; 576];
        let m = Psy1Mask::analyze(&xr, 44_100, 1.0);
        for b in 0..22 {
            assert_eq!(m.energy[b], 0.0);
            assert!(m.threshold[b] > 0.0); // floor protects iterator
        }
        assert_eq!(m.worst_nmr_db(1.0), f32::NEG_INFINITY);
    }

    #[test]
    fn psy1_pure_tone_partition_is_tonal() {
        // A pure-tone MDCT (single coefficient hot, neighbours cold)
        // should land in a single Bark partition with SFM ≈ very small
        // ⇒ tonality ≈ 1.
        let mut xr = [0.0f32; 576];
        xr[100] = 1.0;
        let m = Psy1Mask::analyze(&xr, 44_100, 1.0);
        let cp = build_coeff_partition(44_100);
        let p = cp[100] as usize;
        assert!(
            m.partition_tonality[p] > 0.9,
            "expected high tonality for pure tone in partition {p}, got {}",
            m.partition_tonality[p]
        );
    }

    #[test]
    fn psy1_white_noise_partition_is_noisy() {
        // Pseudo-random uniform "noise" across all coefficients should
        // produce SFM ≈ 1 ⇒ tonality ≈ 0 in occupied partitions.
        let mut xr = [0.0f32; 576];
        // Deterministic LCG so the test is reproducible.
        let mut s: u32 = 0x1234_5678;
        for k in 0..576 {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345);
            let v = ((s >> 8) & 0xFFFF) as f32 / 65535.0 - 0.5;
            xr[k] = v;
        }
        let m = Psy1Mask::analyze(&xr, 44_100, 1.0);
        // Most populated partitions should be flagged as noise (low
        // tonality). Allow a few outliers.
        let mut low = 0usize;
        let mut occupied = 0usize;
        for b in 0..N_BARK_PARTITIONS {
            // Skip empty partitions
            if m.partition_threshold[b] <= 0.0 {
                continue;
            }
            occupied += 1;
            if m.partition_tonality[b] < 0.4 {
                low += 1;
            }
        }
        assert!(
            low * 2 > occupied,
            "expected most populated partitions to be noisy, low={low} occupied={occupied}"
        );
    }

    #[test]
    fn psy1_smr_higher_for_signal_band() {
        // Hot band gets high SMR (signal way above mask), silent band
        // gets near-zero SMR.
        let mut xr = [0.0f32; 576];
        for k in 100..150 {
            xr[k] = 0.5;
        }
        let m = Psy1Mask::analyze(&xr, 44_100, 1.0);
        let smr = m.smr_db();
        // Find the sfb that contains coefficient 100.
        let sfb = sfband_long(44_100);
        let mut hot_sfb = 0usize;
        for b in 0..22 {
            if (sfb[b] as usize) <= 100 && (sfb[b + 1] as usize) > 100 {
                hot_sfb = b;
                break;
            }
        }
        let cold_sfb = 21; // last sfb is empty here
        assert!(
            smr[hot_sfb] > 0.0,
            "expected positive SMR on hot sfb {hot_sfb}: {}",
            smr[hot_sfb]
        );
        assert_eq!(smr[cold_sfb], 0.0, "expected zero SMR on silent sfb");
    }

    #[test]
    fn psy1_quality_knob_monotonic() {
        let g0 = vbr_quality_to_psy1_gain(0);
        let g5 = vbr_quality_to_psy1_gain(5);
        let g9 = vbr_quality_to_psy1_gain(9);
        assert!(g0 > g5 && g5 > g9, "{g0} > {g5} > {g9}");
    }

    #[test]
    fn psy1_finer_quantizer_means_lower_nmr() {
        // Same sanity check as psy::tests::finer_quantizer_means_lower_noise_db.
        let mut xr = [0.0f32; 576];
        for i in 0..200 {
            xr[i] = 0.1;
        }
        let m = Psy1Mask::analyze(&xr, 44_100, 1.0);
        let nmr_low = m.worst_nmr_db(crate::psy::global_gain_to_step(120));
        let nmr_high = m.worst_nmr_db(crate::psy::global_gain_to_step(180));
        assert!(
            nmr_high > nmr_low,
            "expected higher gain to yield higher NMR; got nmr_low={nmr_low} nmr_high={nmr_high}"
        );
    }

    #[test]
    fn psy1_partition_count_matches_constant() {
        // Cross-check that N_BARK_PARTITIONS matches the BARK_UPPER_HZ
        // table length (caught compile-time but this asserts runtime).
        assert_eq!(N_BARK_PARTITIONS, BARK_UPPER_HZ.len());
    }

    #[test]
    fn psy1_spreading_raises_neighbour_threshold() {
        // A loud tone in one Bark partition should raise the masking
        // threshold of the neighbouring partition compared to silence.
        let mut xr_hot = [0.0f32; 576];
        for k in 100..110 {
            xr_hot[k] = 0.5;
        }
        let m_hot = Psy1Mask::analyze(&xr_hot, 44_100, 1.0);

        let xr_silence = [0.0f32; 576];
        let m_silence = Psy1Mask::analyze(&xr_silence, 44_100, 1.0);

        // Find a partition near (but not containing) the hot coefficients.
        let cp = build_coeff_partition(44_100);
        let hot_part = cp[105] as usize;
        // Neighbouring partition (one above the hot one).
        let neighbour = (hot_part + 1).min(N_BARK_PARTITIONS - 1);
        if neighbour == hot_part {
            return; // edge case — last partition; skip
        }
        // The neighbour's per-partition threshold should be higher under
        // spreading from the hot partition.
        let t_hot = m_hot.partition_threshold[neighbour];
        let t_silence = m_silence.partition_threshold[neighbour];
        assert!(
            t_hot > t_silence,
            "spreading should raise neighbour threshold (hot={t_hot}, silence={t_silence})"
        );
    }
}
