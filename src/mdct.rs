//! Layer III **forward MDCT + analysis windowing + forward overlap
//! split** — the §2.4.3.4.10.2 analysis transform and the analysis-side
//! mirror of §2.4.3.4.10.3 / §2.4.3.4.10.4 that together form the
//! encoder-side companion of [`crate::imdct::imdct_granule`].
//!
//! This module covers the Layer III **encoder Phase 2** analysis
//! filterbank up to (and including) the forward overlap split. It
//! provides:
//!
//! * 36-point forward MDCT (long blocks) and 12-point forward MDCT
//!   (each of the three short sub-blocks): [`mdct`].
//! * Analysis windowing for the four block types — the mirror of the
//!   §2.4.3.4.10.3 synthesis windows: [`analysis_long_window`],
//!   [`analysis_short_window`], [`window_long_family_analysis`],
//!   [`window_short_analysis`].
//! * Forward overlap split — the analysis-side mirror of the
//!   §2.4.3.4.10.4 overlap-add: [`MdctState`], [`forward_overlap`].
//!
//! It does **not** include the psychoacoustic model, scalefactor
//! estimation, bit allocation, or Huffman encode. Those follow in
//! subsequent Phase 2 rounds.
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
//! which is the cleanest spec-derivable round-trip check at the
//! transform-primitive level. With this round's added analysis
//! windowing + forward overlap split, time-domain perfect
//! reconstruction (TDAC) is also demonstrable on a long-block stream
//! of two successive granules: feed a window-pair through the full
//! analysis chain (forward overlap → window → MDCT) and back through
//! the synthesis chain (IMDCT → window → overlap-add) and the middle
//! granule is recovered scaled by `n/2 · (window²-energy-sum)`.
//!
//! All numeric constants in this module are derived from the
//! §2.4.3.4.10.2 / §2.4.3.4.10.3 / §2.4.3.4.10.4 formulas only.

use crate::side_info::BlockType;
use core::f64::consts::PI;

/// Windowed-sample count for a long block (§2.4.3.4.10.2, `n = 36`).
pub const LONG_N: usize = 36;

/// Windowed-sample count for one short sub-block (§2.4.3.4.10.2,
/// `n = 12`).
pub const SHORT_N: usize = 12;

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

// ----- Analysis windowing (encoder mirror of §2.4.3.4.10.3) -----
//
// The MP3 forward filterbank is the lapped-transform pair
//
//     forward  : analysis-window → forward MDCT
//     inverse  : inverse MDCT → synthesis-window → overlap-add
//
// Time-domain aliasing cancellation (TDAC) of a windowed lapped MDCT
// requires the *same* window on both halves (the analysis window
// equals the synthesis window, the Princen-Bradley condition). The
// §2.4.3.4.10.3 spec only lists the synthesis-side window because the
// standard only describes decode; the encoder uses the identical
// windows on the forward path.

/// Long-block (`n = 36`) analysis window value at position `i`
/// (§2.4.3.4.10.3 a, mirrored on the encoder side):
/// `sin( (pi/36)·(i + 1/2) )`.
///
/// Identical to the synthesis-side window — the analysis and
/// synthesis windows of a lapped MDCT codec are the same window for
/// time-domain aliasing cancellation (Princen-Bradley).
#[must_use]
pub fn analysis_long_window(i: usize) -> f64 {
    let arg = (PI / 36.0) * (i as f64 + 0.5);
    arg.sin()
}

/// Short-block sub-window value at position `i = 0..12`
/// (§2.4.3.4.10.3 d, mirrored on the encoder side):
/// `sin( (pi/12)·(i + 1/2) )`.
#[must_use]
pub fn analysis_short_window(i: usize) -> f64 {
    let arg = (PI / 12.0) * (i as f64 + 0.5);
    arg.sin()
}

/// Apply the long-family analysis window (block_type 0 / 1 / 3) to the
/// 36 forward-overlap input samples `xn[0..36]`, producing the 36
/// windowed samples that feed the 36-point forward MDCT.
///
/// The block_type partitioning mirrors the §2.4.3.4.10.3 synthesis
/// table exactly:
///
/// * **`Long`** (block_type 0): `out[i] = xn[i] · sin((pi/36)(i+1/2))`
///   over all 36.
/// * **`Start`** (block_type 1): long half-window over `i = 0..17`,
///   pass-through over `i = 18..23`, short half-window
///   `sin((pi/12)(i - 18 + 1/2))` over `i = 24..29`, zero over
///   `i = 30..35`.
/// * **`End`** (block_type 3): zero over `i = 0..5`, short half-window
///   `sin((pi/12)(i - 6 + 1/2))` over `i = 6..11`, pass-through over
///   `i = 12..17`, long half-window over `i = 18..35`.
///
/// `Short` is **not** handled by this function — it has three 12-point
/// sub-blocks; use [`window_short_analysis`] for that path.
#[must_use]
pub fn window_long_family_analysis(xn: &[f64; LONG_N], block_type: BlockType) -> [f64; LONG_N] {
    let mut out = [0.0f64; LONG_N];
    match block_type {
        BlockType::Start => {
            for (i, oi) in out.iter_mut().enumerate() {
                *oi = match i {
                    0..=17 => xn[i] * analysis_long_window(i),
                    18..=23 => xn[i],
                    24..=29 => {
                        let arg = (PI / 12.0) * ((i - 18) as f64 + 0.5);
                        xn[i] * arg.sin()
                    }
                    _ => 0.0, // 30..=35
                };
            }
        }
        BlockType::End => {
            for (i, oi) in out.iter_mut().enumerate() {
                *oi = match i {
                    0..=5 => 0.0,
                    6..=11 => {
                        let arg = (PI / 12.0) * ((i - 6) as f64 + 0.5);
                        xn[i] * arg.sin()
                    }
                    12..=17 => xn[i],
                    _ => xn[i] * analysis_long_window(i), // 18..=35
                };
            }
        }
        _ => {
            // BlockType::Long (block_type 0) — and any other long-family
            // dispatch — use the plain sine window across all 36 samples.
            for (i, oi) in out.iter_mut().enumerate() {
                *oi = xn[i] * analysis_long_window(i);
            }
        }
    }
    out
}

/// Split a 36-sample forward-overlap input frame `xn[0..36]` into the
/// **three 12-sample short sub-blocks** that feed the three
/// independent 12-point forward MDCTs, each pre-multiplied by the
/// short analysis window.
///
/// The split is the analysis-side inverse of the §2.4.3.4.10.3 d
/// synthesis concatenation table
///
/// ```text
/// z[i] = 0                       i = 0..5
///        y0[i-6]                 i = 6..11
///        y0[i-6] + y1[i-12]      i = 12..17
///        y1[i-12] + y2[i-18]     i = 18..23
///        y2[i-18]                i = 24..29
///        0                       i = 30..35
/// ```
///
/// The synthesis table sources `y_j` from disjoint 6-sample regions —
/// `y_0` from i=6..17, `y_1` from i=12..23, `y_2` from i=18..29 — so
/// the analysis-side extraction takes the matching 12-sample windows
/// from `xn`:
///
/// * `xj_in[j][k] = xn[6 + 6·j + k]` for `j = 0..3`, `k = 0..12`.
///
/// (`j = 0` covers i = 6..17, `j = 1` covers i = 12..23, `j = 2` covers
/// i = 18..29 — exactly the three sub-block source spans the synthesis
/// table reads back out, including the half-overlap between adjacent
/// sub-blocks.)
///
/// Each extracted 12-sample sub-block is then multiplied entrywise by
/// the §2.4.3.4.10.3 d short window `sin((pi/12)(k + 1/2))` ready for
/// the 12-point [`mdct`].
#[must_use]
pub fn window_short_analysis(xn: &[f64; LONG_N]) -> [[f64; SHORT_N]; 3] {
    let mut out = [[0.0f64; SHORT_N]; 3];
    for (j, oj) in out.iter_mut().enumerate() {
        for (k, ok) in oj.iter_mut().enumerate() {
            let src_i = 6 + 6 * j + k;
            *ok = xn[src_i] * analysis_short_window(k);
        }
    }
    out
}

// ----- Forward overlap split (encoder mirror of §2.4.3.4.10.4) -----
//
// The §2.4.3.4.10.4 synthesis overlap-add is
//
//     result[i] = z[i] + s_prev[i]   for i = 0..17
//     s_next[i] = z[i + 18]          for i = 0..17
//
// where `z[0..36]` is the windowed IMDCT output of the current
// granule and `s_prev[0..18]` is the saved second half from the
// previous granule.
//
// The analysis-side mirror starts from the 18 new subband-time samples
// of the current granule and assembles the 36-sample forward MDCT
// input frame by *prepending* the previous granule's 18 samples
// (saved across the call) and *appending* the current granule's 18:
//
//     xn[i]      = saved_prev[i]    for i = 0..17
//     xn[i + 18] = current[i]       for i = 0..17
//     saved_prev (after) = current
//
// That is exactly the 50%-overlap lapped-transform window source the
// forward MDCT consumes (with subsequent windowing per block type),
// and is the structural analog of the synthesis-side TDAC overlap-add.

/// Per-subband forward-overlap state for the encoder analysis chain
/// (the analysis mirror of [`crate::imdct::ImdctState`]).
///
/// `saved[i]` is the previous granule's 18 subband-time samples,
/// retained verbatim across calls to assemble the next 36-sample
/// forward-MDCT input frame.
#[derive(Debug, Clone)]
pub struct MdctState {
    saved: [f64; LONG_N / 2],
}

impl Default for MdctState {
    fn default() -> Self {
        MdctState {
            saved: [0.0; LONG_N / 2],
        }
    }
}

impl MdctState {
    /// A fresh all-zero forward-overlap state (stream-start state).
    ///
    /// Symmetric with [`crate::imdct::ImdctState::new`] — the
    /// §2.4.3.4.10.4 overlap-add starts with an all-zero saved second
    /// half on the synthesis side, so the analysis side starts the
    /// same way.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The previously-saved 18 subband-time samples (read-only
    /// inspector for tests + callers that want to peek without
    /// stepping the state).
    #[must_use]
    pub fn saved(&self) -> [f64; LONG_N / 2] {
        self.saved
    }

    /// Build a state pre-populated with `saved` as its previous-granule
    /// memory. Equivalent to `Self::new()` followed by manually setting
    /// `saved` — exposed because [`crate::short_block::forward_short_mdct_subband`]
    /// updates the per-subband memory atomically at the end of its
    /// MDCT chain rather than mid-call, and needs a constructor that
    /// takes the new value directly.
    #[must_use]
    pub fn from_saved(saved: [f64; LONG_N / 2]) -> Self {
        MdctState { saved }
    }
}

/// Assemble the 36-sample forward-MDCT input frame for one
/// subband-granule from the 18 new subband-time samples `current` and
/// the per-subband [`MdctState`] (which holds the previous granule's
/// 18 samples).
///
/// Updates `state.saved` to `current` in place so the next call sees
/// the matching previous half.
///
/// The returned `xn[0..36]` is the unwindowed forward-MDCT input;
/// pass it through [`window_long_family_analysis`] (long / start /
/// stop blocks) or [`window_short_analysis`] (short blocks) before
/// the [`mdct`] step.
#[must_use]
pub fn forward_overlap(current: &[f64; LONG_N / 2], state: &mut MdctState) -> [f64; LONG_N] {
    let mut xn = [0.0f64; LONG_N];
    for (i, slot) in xn.iter_mut().enumerate().take(LONG_N / 2) {
        *slot = state.saved[i];
    }
    for (i, slot) in xn.iter_mut().enumerate().skip(LONG_N / 2) {
        *slot = current[i - LONG_N / 2];
    }
    state.saved = *current;
    xn
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

    // ----- Analysis windowing (§2.4.3.4.10.3 a/b/c/d, encoder mirror) -----

    #[test]
    fn analysis_long_window_matches_synthesis_long_window_byte_for_byte() {
        // The encoder analysis window equals the decoder synthesis
        // window (Princen-Bradley TDAC condition). Cross-check the
        // primitive against re-evaluating sin((pi/36)(i+1/2)) directly.
        for i in 0..36 {
            let expect = (PI / 36.0 * (i as f64 + 0.5)).sin();
            let got = analysis_long_window(i);
            assert!(
                (got - expect).abs() < EPS_F64,
                "analysis_long_window({i}) = {got}, expected {expect}",
            );
        }
        // Spec sentinels: w(0) = sin(pi/72), w(17) = sin(35pi/72) = sin(pi/2 - pi/72)
        // and the table is symmetric around i=17.5 / i=35-i.
        for i in 0..18 {
            let lo = analysis_long_window(i);
            let hi = analysis_long_window(35 - i);
            assert!(
                (lo - hi).abs() < EPS_F64,
                "long window symmetric: w[{i}]={lo} vs w[{}]={hi}",
                35 - i
            );
        }
        // sum_i w² == 18 (the half-sum identity of the sine window over
        // n=36, easy spec-derivable cross-check).
        let energy: f64 = (0..36).map(|i| analysis_long_window(i).powi(2)).sum();
        assert!(
            (energy - 18.0).abs() < EPS_F64 * 32.0,
            "long-window Σw² = {energy} expected 18",
        );
    }

    #[test]
    fn analysis_short_window_matches_synthesis_short_window_byte_for_byte() {
        for i in 0..12 {
            let expect = (PI / 12.0 * (i as f64 + 0.5)).sin();
            let got = analysis_short_window(i);
            assert!(
                (got - expect).abs() < EPS_F64,
                "analysis_short_window({i}) = {got}, expected {expect}",
            );
        }
        for i in 0..6 {
            let lo = analysis_short_window(i);
            let hi = analysis_short_window(11 - i);
            assert!(
                (lo - hi).abs() < EPS_F64,
                "short window symmetric: w[{i}]={lo} vs w[{}]={hi}",
                11 - i
            );
        }
        // sum_i w² == 6 over n=12.
        let energy: f64 = (0..12).map(|i| analysis_short_window(i).powi(2)).sum();
        assert!(
            (energy - 6.0).abs() < EPS_F64 * 16.0,
            "short-window Σw² = {energy} expected 6",
        );
    }

    #[test]
    fn window_long_family_analysis_long_block_is_plain_sine_window() {
        // Constant input ones → out[i] = sin((pi/36)(i+1/2)).
        let xn = [1.0f64; 36];
        let out = window_long_family_analysis(&xn, BlockType::Long);
        for (i, &v) in out.iter().enumerate() {
            let expect = analysis_long_window(i);
            assert!(
                (v - expect).abs() < EPS_F64,
                "long block out[{i}] = {v}, expected {expect}",
            );
        }
    }

    #[test]
    fn window_long_family_analysis_start_block_matches_spec_partition() {
        // §2.4.3.4.10.3 b: long-half 0..17, pass-through 18..23, short-
        // half 24..29, zero 30..35. Use a non-trivial input so each
        // region's behaviour is visible.
        let mut xn = [0.0f64; 36];
        for (i, slot) in xn.iter_mut().enumerate() {
            *slot = (i as f64 + 1.0) * 0.1;
        }
        let out = window_long_family_analysis(&xn, BlockType::Start);
        for (i, &v) in out.iter().enumerate() {
            let expect = match i {
                0..=17 => xn[i] * analysis_long_window(i),
                18..=23 => xn[i],
                24..=29 => xn[i] * (PI / 12.0 * ((i - 18) as f64 + 0.5)).sin(),
                _ => 0.0,
            };
            assert!(
                (v - expect).abs() < EPS_F64,
                "start block out[{i}] = {v}, expected {expect}",
            );
        }
    }

    #[test]
    fn window_long_family_analysis_end_block_matches_spec_partition() {
        // §2.4.3.4.10.3 c: zero 0..5, short-half 6..11, pass-through
        // 12..17, long-half 18..35.
        let mut xn = [0.0f64; 36];
        for (i, slot) in xn.iter_mut().enumerate() {
            *slot = (i as f64 + 1.0) * 0.1;
        }
        let out = window_long_family_analysis(&xn, BlockType::End);
        for (i, &v) in out.iter().enumerate() {
            let expect = match i {
                0..=5 => 0.0,
                6..=11 => xn[i] * (PI / 12.0 * ((i - 6) as f64 + 0.5)).sin(),
                12..=17 => xn[i],
                _ => xn[i] * analysis_long_window(i),
            };
            assert!(
                (v - expect).abs() < EPS_F64,
                "end block out[{i}] = {v}, expected {expect}",
            );
        }
    }

    #[test]
    fn window_long_family_analysis_start_then_end_complementary_zero_regions() {
        // The start block zeros out i=30..35 and the end block zeros
        // out i=0..5 — exactly the regions where the *other* block has
        // its pass-through+short-half coverage. The two block shapes
        // tile each other's footprint, which is what makes
        // start→short→stop transitions seamless in the spec.
        let xn = [1.0f64; 36];
        let s = window_long_family_analysis(&xn, BlockType::Start);
        let e = window_long_family_analysis(&xn, BlockType::End);
        for i in 30..36 {
            assert_eq!(s[i], 0.0, "start block must be zero in tail i={i}");
            assert!(e[i] != 0.0, "end block must be non-zero in tail i={i}");
        }
        for i in 0..6 {
            assert!(s[i] != 0.0, "start block must be non-zero in head i={i}");
            assert_eq!(e[i], 0.0, "end block must be zero in head i={i}");
        }
    }

    #[test]
    fn window_short_analysis_extracts_three_sub_blocks_with_correct_overlap() {
        // The synthesis-side concatenation reads y_0 from i=6..17,
        // y_1 from i=12..23, y_2 from i=18..29 (with the half-overlap
        // sums in the middle bands). The analysis-side extraction must
        // sample the *same* spans: xj_in[j][k] = xn[6 + 6j + k].
        let mut xn = [0.0f64; 36];
        for (i, slot) in xn.iter_mut().enumerate() {
            *slot = i as f64 + 1.0; // 1..36, distinct per position
        }
        let sub = window_short_analysis(&xn);

        for j in 0..3 {
            for k in 0..12 {
                let src_i = 6 + 6 * j + k;
                let expect = xn[src_i] * analysis_short_window(k);
                let got = sub[j][k];
                assert!(
                    (got - expect).abs() < EPS_F64,
                    "sub[{j}][{k}] = {got}, expected {expect} (src_i={src_i})",
                );
            }
        }

        // Half-overlap structural check: sub-block 1's first 6 inputs
        // overlap with sub-block 0's last 6 (i = 12..17), and sub-block
        // 2's first 6 overlap with sub-block 1's last 6 (i = 18..23) —
        // these are the same xn positions that the synthesis side adds
        // y_0+y_1 and y_1+y_2 in §2.4.3.4.10.3 d.
        for k in 0..6 {
            // sub[0][6+k] is xn[12+k] · w(6+k); sub[1][k] is xn[12+k] · w(k).
            assert!(
                (sub[0][6 + k] / analysis_short_window(6 + k) - (12 + k + 1) as f64).abs()
                    < EPS_F64
            );
            assert!((sub[1][k] / analysis_short_window(k) - (12 + k + 1) as f64).abs() < EPS_F64);
            assert!(
                (sub[1][6 + k] / analysis_short_window(6 + k) - (18 + k + 1) as f64).abs()
                    < EPS_F64
            );
            assert!((sub[2][k] / analysis_short_window(k) - (18 + k + 1) as f64).abs() < EPS_F64);
        }
    }

    // ----- Forward overlap split (§2.4.3.4.10.4, encoder mirror) -----

    #[test]
    fn mdct_state_default_is_all_zero() {
        let s = MdctState::new();
        for &v in s.saved().iter() {
            assert_eq!(v, 0.0);
        }
        // Default also.
        let d = MdctState::default();
        for &v in d.saved().iter() {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn forward_overlap_first_granule_head_is_zero_tail_is_input() {
        // Stream-start state: saved = 0. xn[0..18] = 0, xn[18..36] = current.
        let mut state = MdctState::new();
        let mut current = [0.0f64; 18];
        for (i, slot) in current.iter_mut().enumerate() {
            *slot = (i as f64 + 1.0) * 0.5;
        }
        let xn = forward_overlap(&current, &mut state);
        for i in 0..18 {
            assert_eq!(xn[i], 0.0, "first granule head must be zero at i={i}");
        }
        for i in 0..18 {
            assert_eq!(
                xn[18 + i],
                current[i],
                "first granule tail must equal current at i={i}",
            );
        }
        // State must now hold current.
        assert_eq!(state.saved(), current);
    }

    #[test]
    fn forward_overlap_second_granule_head_is_prev_tail_is_current() {
        // Run granule 0 then granule 1; granule 1's frame must be
        // [granule0; granule1] concatenated.
        let mut state = MdctState::new();
        let mut g0 = [0.0f64; 18];
        let mut g1 = [0.0f64; 18];
        for i in 0..18 {
            g0[i] = i as f64 * 0.25;
            g1[i] = (i as f64 + 100.0) * 0.25;
        }
        let _frame0 = forward_overlap(&g0, &mut state);
        let frame1 = forward_overlap(&g1, &mut state);
        for i in 0..18 {
            assert_eq!(frame1[i], g0[i], "granule 1 head i={i}");
            assert_eq!(frame1[18 + i], g1[i], "granule 1 tail i={i}");
        }
        assert_eq!(state.saved(), g1);
    }

    #[test]
    fn forward_overlap_then_imdct_state_zero_input_zero_output() {
        // Zero-current → zero frame at stream start (and zero state
        // after), the analysis mirror of imdct's zero-input-zero-output
        // sanity check.
        let mut state = MdctState::new();
        let zero = [0.0f64; 18];
        let frame = forward_overlap(&zero, &mut state);
        for &v in frame.iter() {
            assert_eq!(v, 0.0);
        }
        for &v in state.saved().iter() {
            assert_eq!(v, 0.0);
        }
    }

    // ----- End-to-end Princen-Bradley check (long block) -----
    //
    // With the analysis-window primitives + forward overlap split in
    // place we can show TDAC: feed two successive granules through the
    // forward chain, IMDCT + synthesis-window + overlap-add through
    // the synthesis chain, and recover the *interior* granule scaled
    // by `n/2` (= 18 for long blocks). Cross-fade aliasing terms cancel
    // between adjacent granules — the Princen-Bradley identity that
    // makes the lapped MDCT a critically-sampled exact-reconstruction
    // transform.
    //
    // This is the strongest single test on the new analysis chain: it
    // requires every analysis primitive (window + overlap split + MDCT)
    // and every synthesis primitive (IMDCT + window + overlap-add) to
    // line up exactly.
    #[test]
    fn analysis_synthesis_long_block_tdac_recovery() {
        use crate::imdct::imdct;
        // Three successive granules of subband-time input. The
        // forward-overlap split + window + MDCT then IMDCT + window +
        // overlap-add reconstructs granule g (the middle one) scaled
        // by n/2 = 18.
        let mut g = [[0.0f64; 18]; 3];
        for j in 0..3 {
            for i in 0..18 {
                g[j][i] =
                    ((j * 17 + i) as f64 * 0.31).sin() + 0.3 * ((i as f64) * 0.7 + j as f64).cos();
            }
        }

        // -- Forward chain over granule 0 then 1 then 2 --
        let mut mdct_state = MdctState::new();
        // Frame 0 covers (prev=0, g[0]); frame 1 covers (g[0], g[1]);
        // frame 2 covers (g[1], g[2]).
        let mut bins = [[0.0f64; 18]; 3];
        for j in 0..3 {
            let frame = forward_overlap(&g[j], &mut mdct_state);
            let windowed = window_long_family_analysis(&frame, BlockType::Long);
            let xk = mdct(&windowed, 36);
            bins[j].copy_from_slice(&xk);
        }

        // -- Synthesis chain: IMDCT → window → overlap-add --
        // The overlap-add adds frame j's first-half to frame (j-1)'s
        // second-half. The middle granule's recovered samples are
        // (synthesis-window² + ...)·(n/2)·g[1] when TDAC holds.
        let mut sec_half_prev = [0.0f64; 18]; // saved second half
        let mut recovered_middle = [0.0f64; 18];
        for j in 0..3 {
            let time = imdct(&bins[j], 36);
            // Apply synthesis (= analysis) window.
            let mut z = [0.0f64; 36];
            for i in 0..36 {
                z[i] = time[i] * analysis_long_window(i);
            }
            // First half = z[0..18] + sec_half_prev[0..18].
            let mut first_half = [0.0f64; 18];
            for i in 0..18 {
                first_half[i] = z[i] + sec_half_prev[i];
            }
            // Save new second half.
            sec_half_prev.copy_from_slice(&z[18..36]);
            // The "current" granule emitted at step j is first_half;
            // its expected source is g[j-1] (with g[-1] := 0). We want
            // the *middle* recovered granule (j=2 emits g[1]).
            if j == 2 {
                recovered_middle = first_half;
            }
        }

        // With the spec normalisation (analysis: no scaling, synthesis:
        // no scaling), the *bin-space* round-trip already shows
        // `mdct(imdct(X)) = (n/2)·X` — the orthogonal cosine kernel
        // contributes a factor of `n/2` over n/2 bins, but the
        // *time-space* round-trip `imdct(mdct(x))` carries only `n/4`
        // (each of the two cosine sums in the product-to-sum expansion
        // contributes `n/4` to the Dirichlet-like delta — see the
        // closed-form check in `mdct_imdct_time_roundtrip_factor_is_n_quarter`
        // below). Combined with the Princen-Bradley TDAC sum
        // `w(i)² + w(i+n/2)² = 1` for the long sine window, the
        // two-frame overlap recovers `g[1]` scaled by `n/4 = 9`.
        for i in 0..18 {
            let expect = 9.0 * g[1][i];
            assert!(
                (recovered_middle[i] - expect).abs() < EPS_F64 * 64.0,
                "TDAC middle recovery [{i}] = {} expected {expect}",
                recovered_middle[i],
            );
        }
    }

    // The time-space round-trip factor — checked here independently
    // from the TDAC test above so the `n/4` scaling is documented as
    // a standalone fact about the spec's analysis ≡ synthesis cosine
    // kernel, not just buried inside the TDAC chain. (The bin-space
    // factor is `n/2`, established in
    // `mdct_imdct_roundtrip_n12_tone` / `mdct_imdct_roundtrip_n36_tone`.)
    //
    // For the spec MDCT/IMDCT pair `imdct(mdct(x))[i]` evaluates to
    //
    //     (n/4) · ( x[i] - x[n/2-1-i] )   for i in [0, n/2),
    //     (n/4) · ( x[i] + x[3n/2-1-i] ) for i in [n/2, n),
    //
    // — the time-domain aliasing structure that Princen-Bradley
    // TDAC cancels via the adjacent-frame overlap-add.
    #[test]
    fn mdct_imdct_time_roundtrip_factor_is_n_quarter_for_n12_impulse() {
        let mut x = [0.0f64; 12];
        x[0] = 1.0; // δ at i=0
        let xk = mdct(&x, 12);
        let out = imdct(&xk, 12);
        // i = 0: (n/4)·(x[0] - x[5]) = 3·(1 - 0) = 3
        assert!(
            (out[0] - 3.0).abs() < EPS_F64,
            "imdct(mdct(δ_0))[0] = {} expected n/4 = 3",
            out[0],
        );
        // i = 5: (n/4)·(x[5] - x[0]) = 3·(0 - 1) = -3
        assert!(
            (out[5] - (-3.0)).abs() < EPS_F64,
            "imdct(mdct(δ_0))[5] = {} expected -n/4 = -3",
            out[5],
        );
        // i = 6: (n/4)·(x[6] + x[11]) = 3·0 = 0 (since x = δ_0)
        assert!(
            out[6].abs() < EPS_F64,
            "imdct(mdct(δ_0))[6] = {} expected 0",
            out[6],
        );
        // i = 11: (n/4)·(x[11] + x[6]) = 3·0 = 0
        assert!(
            out[11].abs() < EPS_F64,
            "imdct(mdct(δ_0))[11] = {} expected 0",
            out[11],
        );
    }
}
