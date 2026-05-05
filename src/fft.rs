//! Small clean-room radix-2 Cooley-Tukey FFT used by the
//! ISO/IEC 11172-3 Annex D §D.2.4.1 PCM-domain pre-analysis path of
//! the Layer III Psy-1 encoder.
//!
//! The Psy-1 implementation in [`crate::psy1`] runs on MDCT output,
//! which captures in-bin tones cleanly but smears tones that fall
//! between bins (the MDCT projects onto cosines that aren't a strict
//! tone basis). The spec-mandated path is to compute a 1024-point FFT
//! over the **raw PCM** window (Hann-windowed) and use that as a
//! parallel masking input — between-bin tones land cleanly on
//! neighbouring FFT bins where the partition spreader can pick them
//! up.
//!
//! This module is deliberately small: a 1024-point complex FFT, an
//! iterative bit-reversal, a precomputed twiddle table, and the
//! Hann analysis window. No external implementation was consulted —
//! the algorithm is the textbook radix-2 Cooley-Tukey decimation-in-
//! time formulation taught in every introductory DSP class.
//!
//! ## Reference
//!
//! - Cooley, J. W. & Tukey, J. W. (1965). "An Algorithm for the
//!   Machine Calculation of Complex Fourier Series."
//!   *Mathematics of Computation* 19, pp. 297-301.
//! - ISO/IEC 11172-3 Annex D §D.2.4.1 — FFT analysis for the
//!   psychoacoustic model. The standard Hann window is
//!   `w[n] = 0.5 * (1 - cos(2*pi*n / (N-1)))` for `n = 0..N`.

/// FFT length used by the Layer III psychoacoustic pre-analysis. The
/// spec calls for a 1024-point window covering one analysis frame
/// worth of PCM — long enough to resolve tones near the lowest
/// scale-factor band centre frequency (~30 Hz at 44.1 kHz) with
/// usable accuracy.
pub const FFT_N: usize = 1024;

/// Number of stages in the radix-2 decomposition. `log2(FFT_N) = 10`
/// for `FFT_N = 1024`.
const FFT_LOG2: u32 = 10;

/// Build a Hann analysis window of length `N` matching ISO/IEC 11172-3
/// Annex D §D.2.4.1: `w[n] = 0.5 * (1 - cos(2*pi*n / (N-1)))` for
/// `n = 0..N`. Endpoints sit at `0`, the centre sits at `1`.
pub fn hann_window(n: usize) -> Vec<f32> {
    let two_pi = 2.0 * std::f32::consts::PI;
    let denom = (n.saturating_sub(1)).max(1) as f32;
    (0..n)
        .map(|i| 0.5 * (1.0 - (two_pi * i as f32 / denom).cos()))
        .collect()
}

/// Pre-computed twiddle factors `e^(-2*pi*i*k/N)` for `k = 0..N/2`.
/// Stored as interleaved `(real, imag)` pairs so the inner butterfly
/// is one slice index per access (the Rust optimiser reliably hoists
/// these into registers).
fn build_twiddles(n: usize) -> Vec<(f32, f32)> {
    let two_pi = 2.0 * std::f32::consts::PI;
    let denom = n as f32;
    (0..n / 2)
        .map(|k| {
            let a = -two_pi * k as f32 / denom;
            (a.cos(), a.sin())
        })
        .collect()
}

/// Build the bit-reversal permutation table for an `n`-point FFT.
/// `out[i]` is the bit-reversed value of `i` over `log2(n)` bits.
fn build_bitrev(n: usize) -> Vec<usize> {
    let bits = n.trailing_zeros();
    (0..n)
        .map(|i| {
            let mut v = i;
            let mut r = 0usize;
            for _ in 0..bits {
                r = (r << 1) | (v & 1);
                v >>= 1;
            }
            r
        })
        .collect()
}

/// 1024-point radix-2 Cooley-Tukey decimation-in-time FFT plan.
/// Pre-allocates the twiddle + bit-reversal tables once so the
/// per-frame transform is allocation-free.
pub struct Fft1024 {
    twiddles: Vec<(f32, f32)>,
    bitrev: Vec<usize>,
}

impl Default for Fft1024 {
    fn default() -> Self {
        Self::new()
    }
}

impl Fft1024 {
    /// Build a fresh plan. Only call this once per encoder instance.
    pub fn new() -> Self {
        Self {
            twiddles: build_twiddles(FFT_N),
            bitrev: build_bitrev(FFT_N),
        }
    }

    /// Core FFT transform — writes the complex output into `re` and `im`.
    /// Caller must have pre-bit-reversed the input data.
    fn fft_inplace(re: &mut [f32; FFT_N], im: &mut [f32; FFT_N], twiddles: &[(f32, f32)]) {
        for stage in 1..=FFT_LOG2 {
            let m = 1usize << stage;
            let m_half = m >> 1;
            let stride = FFT_N / m;
            let mut k = 0usize;
            while k < FFT_N {
                for j in 0..m_half {
                    let twi = twiddles[j * stride];
                    let i_top = k + j;
                    let i_bot = i_top + m_half;
                    let bot_re = re[i_bot] * twi.0 - im[i_bot] * twi.1;
                    let bot_im = re[i_bot] * twi.1 + im[i_bot] * twi.0;
                    let top_re = re[i_top];
                    let top_im = im[i_top];
                    re[i_top] = top_re + bot_re;
                    im[i_top] = top_im + bot_im;
                    re[i_bot] = top_re - bot_re;
                    im[i_bot] = top_im - bot_im;
                }
                k += m;
            }
        }
    }

    /// Compute the magnitude-squared spectrum `|X[k]|^2` of a real
    /// 1024-sample input window, returning the first half (513 bins,
    /// k = 0..=N/2). The caller is expected to have already applied
    /// the Hann window to the input.
    ///
    /// The magnitude-squared form is exactly what the Bark-partition
    /// spreader consumes — no need to take square roots.
    pub fn power_spectrum(&self, input: &[f32; FFT_N]) -> [f32; FFT_N / 2 + 1] {
        // Bit-reverse + load real input into complex buffer.
        let mut re = [0.0f32; FFT_N];
        let mut im = [0.0f32; FFT_N];
        for i in 0..FFT_N {
            re[self.bitrev[i]] = input[i];
        }
        Self::fft_inplace(&mut re, &mut im, &self.twiddles);
        // Return |X[k]|^2 for k = 0..=N/2.
        let mut out = [0.0f32; FFT_N / 2 + 1];
        for k in 0..=FFT_N / 2 {
            out[k] = re[k] * re[k] + im[k] * im[k];
        }
        out
    }

    /// Compute the full complex spectrum for k = 0..=N/2, returning the
    /// `[re, im]` pairs. Used by the Psy-2 complex-prediction path which
    /// needs the complex-valued FFT output (not just power) to compute
    /// the second-order predictor error.
    pub fn complex_spectrum(&self, input: &[f32; FFT_N]) -> [[f32; 2]; FFT_N / 2 + 1] {
        let mut re = [0.0f32; FFT_N];
        let mut im = [0.0f32; FFT_N];
        for i in 0..FFT_N {
            re[self.bitrev[i]] = input[i];
        }
        Self::fft_inplace(&mut re, &mut im, &self.twiddles);
        let mut out = [[0.0f32; 2]; FFT_N / 2 + 1];
        for k in 0..=FFT_N / 2 {
            out[k] = [re[k], im[k]];
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hann_endpoints_zero_centre_one() {
        let w = hann_window(1024);
        assert!(w[0].abs() < 1.0e-6, "Hann at n=0 should be 0");
        assert!(w[1023].abs() < 1.0e-6, "Hann at n=N-1 should be 0");
        // Centre bin sits at 0.5*(1-cos(pi)) = 1.
        assert!(
            (w[511] - 1.0).abs() < 1.0e-2,
            "Hann at centre should be ~1, got {}",
            w[511]
        );
    }

    #[test]
    fn fft_dc_input_concentrates_energy_at_bin_zero() {
        // Constant input: all energy in DC bin (k=0); every other
        // bin should be zero (±float round-off).
        let plan = Fft1024::new();
        let input = [1.0f32; FFT_N];
        let p = plan.power_spectrum(&input);
        // DC magnitude = N (sum of input), squared = N^2.
        let expect_dc = (FFT_N * FFT_N) as f32;
        assert!(
            (p[0] - expect_dc).abs() / expect_dc < 1.0e-4,
            "DC bin should be ~N^2 = {expect_dc}, got {}",
            p[0]
        );
        for k in 1..p.len() {
            assert!(
                p[k] / expect_dc < 1.0e-6,
                "non-DC bin {k} should be ~0, got {}",
                p[k]
            );
        }
    }

    #[test]
    fn fft_pure_tone_concentrates_at_correct_bin() {
        // Pure sinusoid at k=64/N normalised frequency lands in bin 64.
        let plan = Fft1024::new();
        let mut input = [0.0f32; FFT_N];
        let freq_bin = 64usize;
        let two_pi = 2.0 * std::f32::consts::PI;
        for n in 0..FFT_N {
            input[n] = (two_pi * freq_bin as f32 * n as f32 / FFT_N as f32).cos();
        }
        let p = plan.power_spectrum(&input);
        // Find the peak; should be at bin 64 (or its mirror at N-64,
        // but we only return 0..=N/2).
        let (max_bin, _) = p
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        assert_eq!(
            max_bin, freq_bin,
            "expected peak at bin {freq_bin}, got {max_bin}"
        );
    }

    #[test]
    fn fft_parseval_holds_for_random_input() {
        // Parseval's theorem: sum |x[n]|^2 == (1/N) * sum |X[k]|^2
        // for the unnormalised forward FFT. Equivalently
        // N * sum_n x^2 == sum_k |X[k]|^2 with full-spectrum sum (we
        // store half-spectrum + DC, so accumulate symmetric bins).
        let plan = Fft1024::new();
        let mut input = [0.0f32; FFT_N];
        let mut s: u32 = 0xDEAD_BEEF;
        for n in 0..FFT_N {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345);
            input[n] = (((s >> 8) & 0xFFFF) as f32 / 65535.0) - 0.5;
        }
        let p = plan.power_spectrum(&input);
        // Time-domain energy * N
        let time_energy: f64 = input.iter().map(|&x| (x as f64) * (x as f64)).sum();
        let target = time_energy * FFT_N as f64;
        // Frequency-domain: DC and Nyquist appear once, every other
        // bin contributes its mirror as well.
        let mut freq_energy: f64 = p[0] as f64 + p[FFT_N / 2] as f64;
        for k in 1..FFT_N / 2 {
            freq_energy += 2.0 * p[k] as f64;
        }
        // Allow 0.1% tolerance for f32 round-off across 10 stages of
        // butterflies.
        let rel = (freq_energy - target).abs() / target.max(1.0e-12);
        assert!(rel < 1.0e-3, "Parseval mismatch: rel={rel}");
    }

    #[test]
    fn fft_two_close_tones_are_resolved() {
        // Two tones at adjacent FFT bins should produce two distinct
        // peaks (the very thing the FFT pre-analysis catches that the
        // MDCT-domain pass smears together when between-bin).
        let plan = Fft1024::new();
        let mut input = [0.0f32; FFT_N];
        let two_pi = 2.0 * std::f32::consts::PI;
        for n in 0..FFT_N {
            let t1 = (two_pi * 100.0 * n as f32 / FFT_N as f32).cos();
            let t2 = (two_pi * 105.0 * n as f32 / FFT_N as f32).cos();
            input[n] = 0.5 * (t1 + t2);
        }
        let p = plan.power_spectrum(&input);
        // Both bins should be dominant relative to noise floor.
        let b100 = p[100];
        let b105 = p[105];
        let b150 = p[150]; // unrelated bin = quiet
        assert!(
            b100 > 100.0 * b150 && b105 > 100.0 * b150,
            "tones at 100 + 105 should dominate ({b100}, {b105}) vs noise {b150}"
        );
    }
}
