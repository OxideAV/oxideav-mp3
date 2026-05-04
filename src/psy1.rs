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
//!    coefficient's centre frequency by `f = (k + 0.5) * sr / N_eq`
//!    (granule index + half-bin) and walk a Zwicker-style boundary
//!    table to assign the coefficients across the partitions.
//!    `N_eq = 1152` for long blocks (576 coefficients spanning Nyquist
//!    = sr/2) and `N_eq = 384` for each of the three short-block
//!    192-coefficient windows. Annex D.2.4.1 specifies an FFT-domain
//!    partition table; in MDCT terms our partition table is the
//!    coefficient-index list `partition_start[..]` derived from the
//!    same Bark boundaries.
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
//!    `OFFSET_NOISE = 5.5 dB`). We approximate tonality by combining
//!    two independent estimators and taking the maximum:
//!
//!    a. **Spectral flatness measure (SFM)** of each partition's
//!       coefficients:
//!
//!       ```text
//!         SFM_b = exp(mean(ln |X[k]|^2)) / mean(|X[k]|^2)   ∈ (0, 1]
//!       ```
//!
//!       SFM ≈ 1 ⇒ flat / noisy partition (use NOISE offset);
//!       SFM ≪ 1 ⇒ peaky / tonal partition (use TONE offset). The
//!       per-partition SFM tonality index is
//!       `t_sfm = min(1, SFM_db / -60.0)` where
//!       `SFM_db = 10 * log10(SFM_b)`.
//!
//!    b. **Peak-detection tonality** — Annex D.2.4.4 wording. Walk
//!       the partition's spectrum and find local maxima `|X[k]|^2 >
//!       |X[k-1]|^2` and `> |X[k+1]|^2`. The peak ratio
//!       `p_b = max_peak_energy / mean_partition_energy` indicates
//!       how concentrated the partition's energy is; mapped to
//!       `[0, 1]` over the conventional `peak_ratio_db ∈ [0, 20]`
//!       range. A pure tone gives a single peak with `peak / mean
//!       ≈ partition_count` (very high), white noise gives many
//!       small peaks with `peak / mean ≈ 1` (low).
//!
//!    The final per-partition tonality is `t_b = max(t_sfm, t_peak)`
//!    so a partition flagged as tonal by *either* estimator gets the
//!    wider TMN budget.
//!
//! 4. **Per-sfb SMR (signal-to-mask ratio).** Once we have the
//!    spread mask threshold per Bark partition, we re-bin to the
//!    encoder's scalefactor-band layout (ISO Table 3-B.8 — 22 long-block
//!    sfbs at 44.1 kHz; 13 short-block sfbs per window × 3 windows for
//!    short blocks). For each sfb we take the per-bin threshold across
//!    the partitions that overlap the sfb (tightest masking constraint
//!    wins), then scale by the partition width to get an energy budget.
//!    The encoder's noise allocator drives global_gain so per-sfb
//!    quantisation noise stays under that budget.
//!
//! No external psy implementation was consulted; the partition table
//! is derived from the Zwicker Bark boundaries cross-referenced against
//! ISO 11172-3 Annex D figures, the spreading formula is Schroeder's
//! closed-form (open-literature), the SFM tonality proxy is standard
//! signal-processing material, and the peak-detection variant follows
//! the spec wording (local maxima in the squared-magnitude spectrum).

use crate::sfband::{sfband_long, sfband_short};

/// Number of Bark partitions used by the Psy-1 implementation. Annex D.2.4
/// table B.2.1 lists 63 partitions for Layer III; we use a coarser
/// 24-band Bark axis derived from the standard Zwicker boundaries
/// — enough to capture the spreading-function behaviour while keeping
/// the per-coefficient binning cost linear in the granule width.
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

/// Short-block MDCT-coefficient centre frequency. Each short-block
/// window is 192 coefficients spanning Nyquist (sr/2), so coefficient
/// `k` (0..192) sits at `(k + 0.5) * sr / 384`.
pub fn coeff_centre_hz_short(k: usize, sample_rate: u32) -> f32 {
    (k as f32 + 0.5) * sample_rate as f32 / 384.0
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

/// Build the per-window partition table for short blocks. Each short
/// window has 192 coefficients with twice the bin width of the long
/// path, so a single Bark partition covers ~half as many bins (and the
/// upper Bark partitions can carry zero bins at low sample rates).
pub fn build_coeff_partition_short(sample_rate: u32) -> [u8; 192] {
    let mut out = [0u8; 192];
    for k in 0..192 {
        let f = coeff_centre_hz_short(k, sample_rate);
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
/// threshold ready for the encoder's noise allocator. Long-block masks
/// have 22 entries; short-block masks have 39 entries (3 windows × 13
/// sfbs). Field types are `Vec` so the same struct can carry either
/// shape.
#[derive(Clone, Debug)]
pub struct Psy1Mask {
    /// Number of sfbs covered by `energy` / `threshold` / `width` /
    /// `start`. 22 for long blocks, 39 (3 × 13) for short blocks.
    pub n_sfb: usize,
    /// Per-sfb signal energy E_b = sum |X[k]|^2 over the sfb.
    pub energy: Vec<f32>,
    /// Per-sfb masking threshold T_b. Quantization noise N_b that
    /// exceeds T_b in any band is audible per the model.
    pub threshold: Vec<f32>,
    /// Per-sfb width in MDCT coefficients.
    pub width: Vec<u16>,
    /// Per-sfb start index into the 576-coefficient granule.
    pub start: Vec<u16>,
    /// Per-Bark-partition SFM-derived tonality estimate in 0..=1
    /// (1 = pure tone, 0 = pure noise). For short blocks this is the
    /// average across windows. Exposed mainly for diagnostics + tests.
    pub partition_tonality: [f32; N_BARK_PARTITIONS],
    /// Per-Bark-partition peak-detector tonality estimate in 0..=1
    /// (1 = single sharp local maximum dominates the partition,
    /// 0 = many comparable peaks). Annex D.2.4.4 wording. For short
    /// blocks this is the average across windows.
    pub partition_peak_tonality: [f32; N_BARK_PARTITIONS],
    /// Per-Bark-partition spread energy threshold (after applying the
    /// spreading function + SNR offset). Exposed mainly for tests.
    /// For short blocks this is the sum across windows (so a partition
    /// that's hot in any window has a meaningful threshold).
    pub partition_threshold: [f32; N_BARK_PARTITIONS],
}

impl Psy1Mask {
    /// Run the full Psy-1 analysis on a 576-coefficient long-block
    /// granule. `gain` is a quality-derived multiplicative SNR target:
    /// the per-band masking threshold is divided by `gain` so larger
    /// gain = stricter target = more bits.
    pub fn analyze(xr: &[f32; 576], sample_rate: u32, gain: f32) -> Self {
        let coeff_partition = build_coeff_partition(sample_rate);
        let (per_coeff_thr, partition_threshold, partition_tonality, partition_peak_tonality) =
            partition_pass(xr, &coeff_partition, sample_rate, gain);

        // Re-bin to per-sfb threshold + energy for the long-block sfb
        // layout (22 bands).
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

        // Floor to keep silent bands from breaking the iterator.
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
            partition_tonality,
            partition_peak_tonality,
            partition_threshold,
        }
    }

    /// Run the Psy-1 analysis for a **short-block granule** (block
    /// type = 2). Short blocks pack three independent 192-coefficient
    /// MDCT windows into the same 576-coefficient granule, in
    /// sfb-window-major order: `xr[0..192]` is window 0,
    /// `xr[192..384]` is window 1, `xr[384..576]` is window 2. Each
    /// window is processed independently — its own per-Bark
    /// partition energy, spreading + tonality, and threshold — and
    /// then the per-window 13 short-block sfbs are concatenated into
    /// 39 entries on the output mask. The encoder's noise allocator
    /// sees one bigger energy/threshold vector and naturally drives
    /// global_gain to mask the worst window/sfb combo.
    ///
    /// This is the spec-mandated path for transient content: each
    /// 192-coefficient window has tighter time localisation (4 ms at
    /// 44.1 kHz vs 13 ms for the long block), so the spreader sees
    /// the burst's local spectrum rather than smearing it across the
    /// full granule. Without this, a short-block granule was driven
    /// by the long-block sfb energy, which over-estimated low-band
    /// masking under a high-frequency click.
    pub fn analyze_short(xr: &[f32; 576], sample_rate: u32, gain: f32) -> Self {
        let coeff_partition = build_coeff_partition_short(sample_rate);
        let sfb_short = sfband_short(sample_rate);

        let mut energy = vec![0.0f32; 3 * 13];
        let mut threshold = vec![0.0f32; 3 * 13];
        let mut width = vec![0u16; 3 * 13];
        let mut start = vec![0u16; 3 * 13];
        let mut tonality_sum = [0.0f32; N_BARK_PARTITIONS];
        let mut peak_tonality_sum = [0.0f32; N_BARK_PARTITIONS];
        let mut tonality_count = [0u32; N_BARK_PARTITIONS];
        let mut part_threshold_sum = [0.0f32; N_BARK_PARTITIONS];

        for w in 0..3 {
            let off = w * 192;
            let mut win = [0.0f32; 192];
            win.copy_from_slice(&xr[off..off + 192]);
            let (per_coeff_thr_192, part_thr, part_ton, part_peak_ton) =
                partition_pass_short(&win, &coeff_partition, sample_rate, gain);

            // Walk the 13 short-block sfbs for this window. The output
            // sfb index is `w * 13 + b` (window-major then sfb).
            for b in 0..13 {
                let s = sfb_short[b] as usize;
                let e_idx = (sfb_short[b + 1] as usize).min(192);
                let lo = s.min(e_idx);
                let hi = e_idx;
                let mut e = 0.0f32;
                let mut thr = 0.0f32;
                for k in lo..hi {
                    e += win[k] * win[k];
                    thr += per_coeff_thr_192[k];
                }
                let out_idx = w * 13 + b;
                energy[out_idx] = e;
                threshold[out_idx] = thr;
                width[out_idx] = (hi - lo) as u16;
                start[out_idx] = (off + lo) as u16;
            }

            for p in 0..N_BARK_PARTITIONS {
                if part_thr[p] > 0.0 {
                    part_threshold_sum[p] += part_thr[p];
                    tonality_sum[p] += part_ton[p];
                    peak_tonality_sum[p] += part_peak_ton[p];
                    tonality_count[p] += 1;
                }
            }
        }

        // Per-sfb floor across the whole granule to keep silent bands
        // from breaking the iterator.
        let global_floor = energy.iter().copied().fold(0.0f32, f32::max) * 1.0e-7 + 1.0e-12;
        for b in 0..energy.len() {
            if threshold[b] < global_floor {
                threshold[b] = global_floor;
            }
        }

        // Diagnostic per-partition averages across windows.
        let mut partition_tonality = [0.0f32; N_BARK_PARTITIONS];
        let mut partition_peak_tonality = [0.0f32; N_BARK_PARTITIONS];
        for p in 0..N_BARK_PARTITIONS {
            if tonality_count[p] > 0 {
                let n = tonality_count[p] as f32;
                partition_tonality[p] = tonality_sum[p] / n;
                partition_peak_tonality[p] = peak_tonality_sum[p] / n;
            }
        }

        Self {
            n_sfb: 3 * 13,
            energy,
            threshold,
            width,
            start,
            partition_tonality,
            partition_peak_tonality,
            partition_threshold: part_threshold_sum,
        }
    }

    /// Estimate per-band noise for a uniform quantizer with step `q`.
    /// Mirrors [`crate::psy::GranuleMask::estimate_noise`] so the two
    /// can swap at the encoder's call site.
    pub fn estimate_noise(&self, q: f32) -> Vec<f32> {
        let var = q * q / 12.0;
        let mut n = vec![0.0f32; self.n_sfb];
        for b in 0..self.n_sfb {
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

    /// Per-band Signal-to-Mask Ratio (SMR) in dB.
    ///
    /// `SMR_b = 10 * log10(E_b / T_b)`. Higher SMR = the band's energy
    /// is well above its mask, so quantisation noise has plenty of
    /// headroom. Lower SMR = the band's mask sits close to the signal,
    /// so the encoder has to push a finer quantizer (more bits) at it.
    pub fn smr_db(&self) -> Vec<f32> {
        let mut out = vec![0.0f32; self.n_sfb];
        for b in 0..self.n_sfb {
            if self.energy[b] <= 1.0e-20 || self.threshold[b] <= 1.0e-30 {
                out[b] = 0.0;
                continue;
            }
            out[b] = 10.0 * (self.energy[b] / self.threshold[b]).log10();
        }
        out
    }
}

/// Shared per-partition pass: walks every coefficient, accumulates
/// per-Bark-partition energy + log-energy + peak detection, runs the
/// Schroeder spreader, and returns:
///
/// * `per_coeff_thr` — per-coefficient threshold (partition_threshold
///   divided by partition_count, so each coefficient carries its share
///   of the partition's masking budget).
/// * `partition_threshold` — per-partition spread+offset threshold.
/// * `partition_tonality` — SFM-derived tonality.
/// * `partition_peak_tonality` — peak-detector tonality.
fn partition_pass(
    xr: &[f32; 576],
    coeff_partition: &[u8; 576],
    sample_rate: u32,
    gain: f32,
) -> (
    [f32; 576],
    [f32; N_BARK_PARTITIONS],
    [f32; N_BARK_PARTITIONS],
    [f32; N_BARK_PARTITIONS],
) {
    let spread_mat = build_spreading_matrix(sample_rate);

    let mut part_energy = [0.0f32; N_BARK_PARTITIONS];
    let mut part_count = [0u32; N_BARK_PARTITIONS];
    let mut part_log_sum = [0.0f32; N_BARK_PARTITIONS];
    let mut part_peak = [0.0f32; N_BARK_PARTITIONS];
    let log_floor: f32 = 1.0e-20;
    let energy_at = |k: usize| -> f32 { xr[k] * xr[k] };
    for k in 0..576 {
        let p = coeff_partition[k] as usize;
        let e = energy_at(k);
        part_energy[p] += e;
        part_log_sum[p] += (e.max(log_floor)).ln();
        part_count[p] += 1;
        // Local-maximum peak detector: `e > both neighbours`. End
        // coefficients (k=0, k=575) treat the missing neighbour as
        // zero — they qualify as a peak whenever the existing
        // neighbour is smaller.
        let prev = if k == 0 { 0.0 } else { energy_at(k - 1) };
        let next = if k == 575 { 0.0 } else { energy_at(k + 1) };
        if e > prev && e > next && e > part_peak[p] {
            part_peak[p] = e;
        }
    }

    let (tonality, peak_tonality) =
        compute_tonalities(&part_energy, &part_log_sum, &part_count, &part_peak);
    let part_threshold = spread_and_offset(&part_energy, &spread_mat, &tonality, gain);

    // Per-coefficient threshold = partition_threshold / partition_count.
    let mut per_coeff_thr = [0.0f32; 576];
    for k in 0..576 {
        let p = coeff_partition[k] as usize;
        let n = part_count[p].max(1) as f32;
        per_coeff_thr[k] = part_threshold[p] / n;
    }
    (per_coeff_thr, part_threshold, tonality, peak_tonality)
}

/// Per-partition pass for one short-block window (192 coeffs). Same
/// shape as [`partition_pass`] but on the per-window 192-coefficient
/// span and partition table.
fn partition_pass_short(
    win: &[f32; 192],
    coeff_partition: &[u8; 192],
    sample_rate: u32,
    gain: f32,
) -> (
    [f32; 192],
    [f32; N_BARK_PARTITIONS],
    [f32; N_BARK_PARTITIONS],
    [f32; N_BARK_PARTITIONS],
) {
    let spread_mat = build_spreading_matrix(sample_rate);

    let mut part_energy = [0.0f32; N_BARK_PARTITIONS];
    let mut part_count = [0u32; N_BARK_PARTITIONS];
    let mut part_log_sum = [0.0f32; N_BARK_PARTITIONS];
    let mut part_peak = [0.0f32; N_BARK_PARTITIONS];
    let log_floor: f32 = 1.0e-20;
    let energy_at = |k: usize| -> f32 { win[k] * win[k] };
    for k in 0..192 {
        let p = coeff_partition[k] as usize;
        let e = energy_at(k);
        part_energy[p] += e;
        part_log_sum[p] += (e.max(log_floor)).ln();
        part_count[p] += 1;
        let prev = if k == 0 { 0.0 } else { energy_at(k - 1) };
        let next = if k == 191 { 0.0 } else { energy_at(k + 1) };
        if e > prev && e > next && e > part_peak[p] {
            part_peak[p] = e;
        }
    }

    let (tonality, peak_tonality) =
        compute_tonalities(&part_energy, &part_log_sum, &part_count, &part_peak);
    let part_threshold = spread_and_offset(&part_energy, &spread_mat, &tonality, gain);

    let mut per_coeff_thr = [0.0f32; 192];
    for k in 0..192 {
        let p = coeff_partition[k] as usize;
        let n = part_count[p].max(1) as f32;
        per_coeff_thr[k] = part_threshold[p] / n;
    }
    (per_coeff_thr, part_threshold, tonality, peak_tonality)
}

/// Compute the SFM-based and peak-based tonality estimates per
/// partition; return the *combined* tonality (max of the two) along
/// with the peak-only estimate for diagnostics. Both estimates are in
/// `[0, 1]` (1 = pure tone, 0 = pure noise).
fn compute_tonalities(
    part_energy: &[f32; N_BARK_PARTITIONS],
    part_log_sum: &[f32; N_BARK_PARTITIONS],
    part_count: &[u32; N_BARK_PARTITIONS],
    part_peak: &[f32; N_BARK_PARTITIONS],
) -> ([f32; N_BARK_PARTITIONS], [f32; N_BARK_PARTITIONS]) {
    let log_floor: f32 = 1.0e-20;
    let mut combined = [0.0f32; N_BARK_PARTITIONS];
    let mut peak = [0.0f32; N_BARK_PARTITIONS];
    for b in 0..N_BARK_PARTITIONS {
        if part_count[b] == 0 || part_energy[b] <= log_floor {
            continue;
        }
        let n = part_count[b] as f32;
        let arith = part_energy[b] / n;
        // SFM: geometric / arithmetic mean of |X|^2.
        let geo = (part_log_sum[b] / n).exp();
        let sfm = (geo / arith.max(log_floor)).clamp(1.0e-20, 1.0);
        let sfm_db = 10.0 * sfm.log10();
        let t_sfm = ((-sfm_db) / 60.0).clamp(0.0, 1.0);
        // Peak-detection: ratio of largest local-max energy to mean
        // partition energy. A pure tone gives `peak / mean ≈ N`
        // (where N is partition_count), white noise gives ~1.
        // Convert to dB and map [0, 20] dB → [0, 1].
        let peak_ratio = if arith > 0.0 {
            part_peak[b] / arith
        } else {
            1.0
        };
        let peak_ratio_db = 10.0 * peak_ratio.max(1.0).log10();
        let t_peak = (peak_ratio_db / 20.0).clamp(0.0, 1.0);
        combined[b] = t_sfm.max(t_peak);
        peak[b] = t_peak;
    }
    (combined, peak)
}

/// Spread per-partition energy across all partitions in dB domain,
/// convert back to linear, and apply the tonality-interpolated SNR
/// offset. Final per-partition threshold is divided by `gain` (the
/// encoder quality knob — larger = tighter mask = more bits).
fn spread_and_offset(
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
    fn coeff_partition_short_assigns_increasing() {
        let cp = build_coeff_partition_short(44_100);
        assert_eq!(cp[0], 0);
        // Short-block bins are 6× wider than long-block bins
        // (192 vs 1152 in the divisor), so the last short-block
        // coefficient lands in the highest Bark partition.
        assert!(cp[191] >= 20, "last short partition {}", cp[191]);
        for k in 1..192 {
            assert!(
                cp[k] >= cp[k - 1],
                "short-block partition must be non-decreasing"
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
        assert_eq!(m.n_sfb, 22);
        for b in 0..22 {
            assert_eq!(m.energy[b], 0.0);
            assert!(m.threshold[b] > 0.0); // floor protects iterator
        }
        assert_eq!(m.worst_nmr_db(1.0), f32::NEG_INFINITY);
    }

    #[test]
    fn psy1_short_silence_has_floor_thresholds() {
        let xr = [0.0f32; 576];
        let m = Psy1Mask::analyze_short(&xr, 44_100, 1.0);
        assert_eq!(m.n_sfb, 39);
        for b in 0..39 {
            assert_eq!(m.energy[b], 0.0);
            assert!(m.threshold[b] > 0.0); // floor protects iterator
        }
        assert_eq!(m.worst_nmr_db(1.0), f32::NEG_INFINITY);
    }

    #[test]
    fn psy1_pure_tone_partition_is_tonal() {
        // A pure-tone MDCT (single coefficient hot, neighbours cold)
        // should land in a single Bark partition with very low SFM
        // and a very large peak/mean ratio ⇒ tonality ≈ 1.
        let mut xr = [0.0f32; 576];
        xr[100] = 1.0;
        let m = Psy1Mask::analyze(&xr, 44_100, 1.0);
        let cp = build_coeff_partition(44_100);
        let p = cp[100] as usize;
        assert!(
            m.partition_tonality[p] > 0.9,
            "expected high combined tonality for pure tone in partition {p}, got {}",
            m.partition_tonality[p]
        );
        assert!(
            m.partition_peak_tonality[p] > 0.5,
            "expected high peak tonality for pure tone in partition {p}, got {}",
            m.partition_peak_tonality[p]
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
        // tonality). Allow a few outliers — both SFM and peak
        // detection can spike on small partitions.
        let mut low = 0usize;
        let mut occupied = 0usize;
        for b in 0..N_BARK_PARTITIONS {
            // Skip empty partitions
            if m.partition_threshold[b] <= 0.0 {
                continue;
            }
            occupied += 1;
            if m.partition_tonality[b] < 0.6 {
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
    fn psy1_short_finer_quantizer_means_lower_nmr() {
        // Same sanity check on the short-block path.
        let mut xr = [0.0f32; 576];
        for w in 0..3 {
            for i in 0..100 {
                xr[w * 192 + i] = 0.1;
            }
        }
        let m = Psy1Mask::analyze_short(&xr, 44_100, 1.0);
        let nmr_low = m.worst_nmr_db(crate::psy::global_gain_to_step(120));
        let nmr_high = m.worst_nmr_db(crate::psy::global_gain_to_step(180));
        assert!(
            nmr_high > nmr_low,
            "expected higher gain to yield higher NMR on short blocks; got nmr_low={nmr_low} nmr_high={nmr_high}"
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

    #[test]
    fn psy1_short_independent_per_window() {
        // Build a granule whose three short windows carry very
        // different content: window 0 = silent, window 1 = a hot
        // mid-band tone, window 2 = a hot HF tone. The per-window
        // sfb energies in the resulting mask should mirror those
        // shapes (not be smeared across all 3 windows).
        let mut xr = [0.0f32; 576];
        // Window 1 mid-band hit
        xr[192 + 50] = 1.0;
        // Window 2 HF hit
        xr[384 + 150] = 1.0;
        let m = Psy1Mask::analyze_short(&xr, 44_100, 1.0);
        // Window 0 sfb energies all zero
        for b in 0..13 {
            assert_eq!(
                m.energy[b], 0.0,
                "expected window 0 sfb {b} silent, got {}",
                m.energy[b]
            );
        }
        // Window 1 should have non-zero energy in the sfb that
        // contains coeff 50 (low-mid). Window 2 should have non-zero
        // energy in the sfb that contains coeff 150 (HF).
        let any_w1: f32 = (13..26).map(|b| m.energy[b]).sum();
        let any_w2: f32 = (26..39).map(|b| m.energy[b]).sum();
        assert!(any_w1 > 0.0, "window 1 should carry energy");
        assert!(any_w2 > 0.0, "window 2 should carry energy");
    }

    #[test]
    fn psy1_peak_tonality_high_for_pure_tone() {
        // Pure tone: peak_ratio = N (very large) → peak tonality ≈ 1.
        let mut xr = [0.0f32; 576];
        xr[100] = 1.0;
        let m = Psy1Mask::analyze(&xr, 44_100, 1.0);
        let cp = build_coeff_partition(44_100);
        let p = cp[100] as usize;
        assert!(
            m.partition_peak_tonality[p] > 0.5,
            "pure-tone peak tonality should be > 0.5, got {}",
            m.partition_peak_tonality[p]
        );
    }

    #[test]
    fn psy1_peak_tonality_low_for_flat_partition() {
        // Constant-amplitude partition has no local maxima → peak/mean
        // ≈ 1 → peak tonality 0.
        let mut xr = [0.0f32; 576];
        for k in 0..576 {
            xr[k] = 0.1;
        }
        let m = Psy1Mask::analyze(&xr, 44_100, 1.0);
        // Pick a populated middle partition.
        let p = 12usize;
        assert!(
            m.partition_peak_tonality[p] < 0.3,
            "flat-partition peak tonality should be < 0.3, got {}",
            m.partition_peak_tonality[p]
        );
    }
}
