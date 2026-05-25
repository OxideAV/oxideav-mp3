//! Layer III **forward MDCT** — the §2.4.3.4.10.2 analysis transform that
//! is the encoder-side companion of [`crate::imdct::imdct`].
//!
//! This module begins the Layer III **encoder Phase 2** analysis
//! filterbank. It contains the forward (analysis) MDCT primitive only:
//!
//! * 36-point forward MDCT (long blocks), and
//! * 12-point forward MDCT (each of the three short sub-blocks).
//!
//! It does **not** include analysis windowing, the forward overlap
//! split, the psychoacoustic model, scalefactor estimation, bit
//! allocation, or Huffman encode. Those follow in subsequent Phase 2
//! rounds.
//!
//! # Definition (ISO/IEC 11172-3:1993 §2.4.3.4.10.2)
//!
//! The §2.4.3.4.10.2 IMDCT is given (in the spec's pseudo code, and
//! implemented by [`crate::imdct::imdct`]) by
//!
//! ```text
//! x[i] = sum over k=0..n/2-1 of
//!            X[k] · cos( (pi / (2n)) · (2i + 1 + n/2) · (2k + 1) )
//!        for i = 0 .. n-1
//! ```
//!
//! The forward MDCT — the analysis half of the same cosine-modulated
//! lapped transform — applies the **same** cosine kernel transposed,
//! i.e. summed over the `n` time samples for each of the `n/2` output
//! bins:
//!
//! ```text
//! X[k] = sum over i=0..n-1 of
//!            x[i] · cos( (pi / (2n)) · (2i + 1 + n/2) · (2k + 1) )
//!        for k = 0 .. n/2-1
//! ```
//!
//! With this normalisation (matching the spec's IMDCT) the analysis
//! transform is the **left inverse** of the synthesis transform on the
//! `n/2`-dimensional bin space:
//!
//! ```text
//! MDCT( IMDCT(X) )[k] = (n/2) · X[k]
//! ```
//!
//! which is the cleanest spec-derivable round-trip check (the
//! time-domain TDAC round-trip requires analysis windowing + a forward
//! overlap split, which are a separate Phase 2 piece).
//!
//! All numeric constants in this module are derived from the
//! §2.4.3.4.10.2 formulas only.

use core::f64::consts::PI;

/// Forward MDCT (§2.4.3.4.10.2 analysis): transform `n` time samples
/// `xn` into `n/2` frequency bins.
///
/// `xn` must hold exactly `n` values; the function returns `n / 2`
/// output bins. `n` is intended to be `36` (long block) or `12` (each
/// short sub-block); the implementation does not enforce either, so a
/// caller computing the formula for other even `n` gets the same shape.
///
/// Computation is in `f64`; callers downcast to `f32` after windowing.
///
/// # Inverse relation
///
/// With the spec normalisation, applying this transform after
/// [`crate::imdct::imdct`] recovers the input bins scaled by `n/2`:
///
/// ```text
/// mdct( imdct(X), n )[k] = (n/2) · X[k]
/// ```
///
/// (so an encoder pipeline that wants the round-trip identity divides
/// the analysis output by `n/2` — or absorbs the factor into the
/// quantiser scale.)
#[must_use]
pub fn mdct(xn: &[f64], n: usize) -> Vec<f64> {
    debug_assert_eq!(xn.len(), n, "mdct: xn must have n entries");
    let half = n / 2;
    let nn = n as f64;
    // pi / (2n) — the kernel's per-(i,k) outer factor, matching the
    // §2.4.3.4.10.2 IMDCT exactly.
    let scale = PI / (2.0 * nn);
    let mut out = vec![0.0f64; half];
    for (k, ok) in out.iter_mut().enumerate() {
        // (2k + 1): the per-output (per-bin) phase factor.
        let b = (2 * k + 1) as f64;
        let mut acc = 0.0f64;
        for (i, &x) in xn.iter().enumerate() {
            // (2i + 1 + n/2): the per-time-sample phase offset, the same
            // expression the IMDCT uses for its output-side phase.
            let a = (2 * i + 1 + half) as f64;
            acc += x * (scale * a * b).cos();
        }
        *ok = acc;
    }
    out
}

#[cfg(test)]
// The tests recompute every reference value from the §2.4.3.4.10.2 spec
// formula in their natural `for k in 0..n/2 { X[k] = f(k) }` shape; the
// index variable is part of the formula (e.g. `X[k] = sum_i x[i]·cos((π/
// (2n))·(2i+1+n/2)·(2k+1))`), not just a buffer iterator, so keeping the
// explicit range makes the test text mirror the spec more faithfully than
// iterator chains.
#[allow(clippy::needless_range_loop)]
mod tests_inner {
    use super::*;
    use crate::imdct::imdct;

    const EPS_F64: f64 = 1e-10;

    // -- §2.4.3.4.10.2 analysis MDCT, n = 12 --

    #[test]
    fn mdct_n12_impulse_closed_form_first_sample() {
        // x[0] = 1, x[1..12] = 0 ⇒ X[k] = cos((π/24)·(0·2+1+6)·(2k+1))
        //                              = cos((π/24)·7·(2k+1)).
        let mut xn = [0.0f64; 12];
        xn[0] = 1.0;
        let xk = mdct(&xn, 12);
        assert_eq!(xk.len(), 6);
        for (k, &v) in xk.iter().enumerate() {
            let expect = (PI / 24.0 * 7.0 * (2 * k + 1) as f64).cos();
            assert!(
                (v - expect).abs() < EPS_F64,
                "n=12 x[0]=1 X[{k}]={v} expected {expect}"
            );
        }
    }

    #[test]
    fn mdct_n12_impulse_last_sample() {
        // x[11] = 1, rest 0 ⇒ X[k] = cos((π/24)·(23+6)·(2k+1))
        //                          = cos((π/24)·29·(2k+1)).
        // Note: 2·11+1+6 = 29.
        let mut xn = [0.0f64; 12];
        xn[11] = 1.0;
        let xk = mdct(&xn, 12);
        for (k, &v) in xk.iter().enumerate() {
            let expect = (PI / 24.0 * 29.0 * (2 * k + 1) as f64).cos();
            assert!(
                (v - expect).abs() < EPS_F64,
                "n=12 x[11]=1 X[{k}]={v} expected {expect}"
            );
        }
    }

    // -- §2.4.3.4.10.2 analysis MDCT, n = 36 --

    #[test]
    fn mdct_n36_impulse_closed_form() {
        // x[5] = 1, rest 0 ⇒ X[k] = cos((π/72)·(2·5+1+18)·(2k+1))
        //                         = cos((π/72)·29·(2k+1)).
        let mut xn = [0.0f64; 36];
        xn[5] = 1.0;
        let xk = mdct(&xn, 36);
        assert_eq!(xk.len(), 18);
        for (k, &v) in xk.iter().enumerate() {
            let expect = (PI / 72.0 * 29.0 * (2 * k + 1) as f64).cos();
            assert!(
                (v - expect).abs() < EPS_F64,
                "n=36 x[5]=1 X[{k}]={v} expected {expect}"
            );
        }
    }

    #[test]
    fn mdct_n36_constant_one_bin_zero_spot_checks() {
        // x[i] = 1 for all i = 0..36. The MDCT of a DC signal across the
        // window is *not* a single bin (MDCT is not a DFT); each bin is
        // sum_i cos((π/72)·(2i+1+18)·(2k+1)) which evaluates exactly to
        // 0 for many k by cosine symmetry. Spot-check two bins against
        // direct evaluation.
        let xn = [1.0f64; 36];
        let xk = mdct(&xn, 36);
        for (k, &v) in xk.iter().enumerate() {
            let b = (2 * k + 1) as f64;
            let mut expect = 0.0f64;
            for i in 0..36 {
                let a = (2 * i + 1 + 18) as f64;
                expect += (PI / 72.0 * a * b).cos();
            }
            assert!(
                (v - expect).abs() < EPS_F64 * 36.0,
                "n=36 ones X[{k}]={v} expected {expect}"
            );
        }
    }

    // -- Linearity of the analysis transform --

    #[test]
    fn mdct_n12_linearity() {
        let mut x = [0.0f64; 12];
        let mut y = [0.0f64; 12];
        for i in 0..12 {
            x[i] = (i as f64).sin();
            y[i] = (i as f64 + 0.3).cos();
        }
        let a = 0.6;
        let b = -1.4;
        let mut z = [0.0f64; 12];
        for i in 0..12 {
            z[i] = a * x[i] + b * y[i];
        }
        let xk = mdct(&x, 12);
        let yk = mdct(&y, 12);
        let zk = mdct(&z, 12);
        for k in 0..6 {
            let expect = a * xk[k] + b * yk[k];
            assert!(
                (zk[k] - expect).abs() < EPS_F64,
                "n=12 linearity k={k}: {} vs {}",
                zk[k],
                expect
            );
        }
    }

    #[test]
    fn mdct_n36_linearity() {
        let mut x = [0.0f64; 36];
        let mut y = [0.0f64; 36];
        for i in 0..36 {
            x[i] = (i as f64 * 0.31).sin();
            y[i] = (i as f64 * 0.17 + 1.1).cos();
        }
        let a = 0.9;
        let b = 0.4;
        let mut z = [0.0f64; 36];
        for i in 0..36 {
            z[i] = a * x[i] + b * y[i];
        }
        let xk = mdct(&x, 36);
        let yk = mdct(&y, 36);
        let zk = mdct(&z, 36);
        for k in 0..18 {
            let expect = a * xk[k] + b * yk[k];
            assert!(
                (zk[k] - expect).abs() < EPS_F64,
                "n=36 linearity k={k}: {} vs {}",
                zk[k],
                expect
            );
        }
    }

    // -- Round-trip through the IMDCT already shipped by [`crate::imdct`] --
    //
    // The §2.4.3.4.10.2 IMDCT is the over-determined map from `n/2` bins
    // to `n` time samples; the forward MDCT is its left inverse on the
    // `n/2`-dimensional bin space:
    //
    //   MDCT( IMDCT(X) )[k] = (n/2) · X[k]
    //
    // This is the cleanest "encode → bit-exact decode round-trip" check
    // at the primitive level — exact perfect reconstruction on tones in
    // the frequency-domain bin space, no windowing / overlap needed.
    // The time-domain TDAC round-trip (which does require windowing +
    // overlap) is a separate Phase 2 piece.

    #[test]
    fn mdct_imdct_roundtrip_n12_tone() {
        // A single-bin "tone" in the frequency domain.
        for k0 in 0..6 {
            let mut bins = [0.0f64; 6];
            bins[k0] = 1.0;
            let time = imdct(&bins, 12);
            let recov = mdct(&time, 12);
            for k in 0..6 {
                let expect = if k == k0 { 6.0 } else { 0.0 }; // n/2 = 6
                assert!(
                    (recov[k] - expect).abs() < EPS_F64,
                    "n=12 tone k0={k0} recov[{k}]={} expected {expect}",
                    recov[k]
                );
            }
        }
    }

    #[test]
    fn mdct_imdct_roundtrip_n36_tone() {
        for k0 in 0..18 {
            let mut bins = [0.0f64; 18];
            bins[k0] = 1.0;
            let time = imdct(&bins, 36);
            let recov = mdct(&time, 36);
            for k in 0..18 {
                let expect = if k == k0 { 18.0 } else { 0.0 }; // n/2 = 18
                assert!(
                    (recov[k] - expect).abs() < EPS_F64 * 18.0,
                    "n=36 tone k0={k0} recov[{k}]={} expected {expect}",
                    recov[k]
                );
            }
        }
    }

    #[test]
    fn mdct_imdct_roundtrip_n36_mixed_spectrum() {
        // Arbitrary frequency-domain content (mixture of "tones") should
        // round-trip to (n/2 = 18) × itself within numerical noise.
        let mut bins = [0.0f64; 18];
        for k in 0..18 {
            bins[k] = ((k as f64) * 0.41).sin() + 0.5 * ((k as f64) * 1.7 - 0.3).cos();
        }
        let time = imdct(&bins, 36);
        let recov = mdct(&time, 36);
        for k in 0..18 {
            let expect = 18.0 * bins[k];
            assert!(
                (recov[k] - expect).abs() < EPS_F64 * 32.0,
                "n=36 mixed k={k}: {} vs {}",
                recov[k],
                expect
            );
        }
    }

    #[test]
    fn mdct_imdct_roundtrip_n12_mixed_spectrum() {
        let mut bins = [0.0f64; 6];
        for k in 0..6 {
            bins[k] = (k as f64 + 0.7).cos() - 0.3 * ((k as f64) * 2.1).sin();
        }
        let time = imdct(&bins, 12);
        let recov = mdct(&time, 12);
        for k in 0..6 {
            let expect = 6.0 * bins[k];
            assert!(
                (recov[k] - expect).abs() < EPS_F64 * 8.0,
                "n=12 mixed k={k}: {} vs {}",
                recov[k],
                expect
            );
        }
    }

    // -- Output length contract --

    #[test]
    fn mdct_output_length_is_half_n() {
        assert_eq!(mdct(&[0.0f64; 12], 12).len(), 6);
        assert_eq!(mdct(&[0.0f64; 36], 36).len(), 18);
    }

    // -- Direct spec-sum recomputation on an arbitrary input --
    //
    // The cleanest "the implementation evaluates the §2.4.3.4.10.2
    // analysis sum" check: pick an arbitrary input and re-evaluate the
    // formula independently in the test, then compare bin-for-bin.

    #[test]
    fn mdct_n12_matches_spec_sum_on_arbitrary_input() {
        let xn = [
            0.5f64, -0.2, 0.7, 0.1, -0.4, 0.9, 0.3, -0.6, 0.15, 0.8, -0.25, 0.55,
        ];
        let xk = mdct(&xn, 12);
        for (k, &v) in xk.iter().enumerate() {
            let b = (2 * k + 1) as f64;
            let mut expect = 0.0f64;
            for i in 0..12 {
                let a = (2 * i + 1 + 6) as f64;
                expect += xn[i] * (PI / 24.0 * a * b).cos();
            }
            assert!(
                (v - expect).abs() < EPS_F64 * 12.0,
                "spec-sum n=12 X[{k}]={v} expected {expect}"
            );
        }
    }
}
