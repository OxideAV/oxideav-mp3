//! Layer III **IMDCT, windowing, and overlap-add** — the
//! §2.4.3.4.10.2 / §2.4.3.4.10.3 / §2.4.3.4.10.4 / §2.4.3.4.10.5 core of
//! the synthesis filterbank, run per polyphase subband after alias
//! reduction.
//!
//! This module turns the 576 alias-reduced frequency lines `xr[576]` of
//! one granule-channel — already grouped 18-per-subband by the reorder
//! stage — into the 32×18 block of subband-domain *time* samples that
//! feed the polyphase synthesis filterbank (a later stage).
//!
//! Each polyphase subband is processed independently: its 18 frequency
//! lines are run through the per-subband IMDCT, windowed according to the
//! granule's block type, overlapped with the second half of the same
//! subband's previous-granule output, and finally sign-flipped on the
//! odd time samples of odd subbands to compensate for the polyphase
//! filterbank's frequency inversion.
//!
//! # IMDCT (§2.4.3.4.10.2)
//!
//! With `n` the number of windowed samples (`n = 36` for long blocks,
//! `n = 12` for short blocks; a short block transforms each of its three
//! 6-line sub-blocks separately), `n/2` input values `X[k]` map to `n`
//! output values `x[i]`:
//!
//! ```text
//! x[i] = sum over k=0..n/2-1 of
//!            X[k] · cos( (pi / (2n)) · (2i + 1 + n/2) · (2k + 1) )
//!        for i = 0 .. n-1
//! ```
//!
//! # Windowing (§2.4.3.4.10.3)
//!
//! The 36 (long-block) or 3×12 (short-block) IMDCT outputs are windowed
//! into a 36-value vector `z[i]` whose shape depends on `block_type`:
//!
//! * **block_type 0** (normal): `z[i] = x[i] · sin( (pi/36)·(i + 1/2) )`,
//!   `i = 0..35`.
//! * **block_type 1** (start): the sine half-window over `i = 0..17`,
//!   pass-through over `i = 18..23`, the short half-window
//!   `sin( (pi/12)·(i - 18 + 1/2) )` over `i = 24..29`, zero over
//!   `i = 30..35`.
//! * **block_type 3** (stop): zero over `i = 0..5`, the short half-window
//!   `sin( (pi/12)·(i - 6 + 1/2) )` over `i = 6..11`, pass-through over
//!   `i = 12..17`, the long half-window over `i = 18..35`.
//! * **block_type 2** (short): each of the three 12-sample IMDCT outputs
//!   is windowed with `sin( (pi/12)·(i + 1/2) )`, `i = 0..11`, then the
//!   three windowed sub-blocks are overlapped and concatenated into the
//!   36-value `z` per the §2.4.3.4.10.3 short-block table.
//!
//! # Overlap-add (§2.4.3.4.10.4)
//!
//! The first 18 windowed values overlap-add with the previous granule's
//! saved second half; the actual block's second 18 values are saved for
//! the next granule:
//!
//! ```text
//! result[i] = z[i] + s_prev[i]   for i = 0..17
//! s_next[i] = z[i + 18]          for i = 0..17
//! ```
//!
//! # Frequency inversion (§2.4.3.4.10.5)
//!
//! Across the 18 time samples of each of the 32 subbands, every odd time
//! sample of every odd subband is multiplied by −1.

use crate::side_info::{BlockType, GranuleChannel};

/// Number of frequency lines in a granule-channel (32 subbands × 18).
pub const NUM_LINES: usize = 576;

/// Number of polyphase subbands per granule.
pub const NUM_SUBBANDS: usize = 32;

/// Time samples produced per subband per granule (after overlap-add).
pub const SAMPLES_PER_SUBBAND: usize = 18;

/// Windowed-sample count for a long block (§2.4.3.4.10.2, `n = 36`).
const LONG_N: usize = 36;

/// Windowed-sample count for one short sub-block (§2.4.3.4.10.2,
/// `n = 12`).
const SHORT_N: usize = 12;

/// Per-channel IMDCT overlap state: the second half of each subband's
/// windowed output, saved from the previous granule and added to the
/// first half of the current granule's output (§2.4.3.4.10.4).
///
/// All 32 subbands start zeroed (the spec initialises the overlap store
/// to zero at stream start), so a fresh [`ImdctState::default`] is the
/// correct decoder start state.
#[derive(Debug, Clone)]
pub struct ImdctState {
    /// `overlap[sb][i]` is the second-half value `z[i+18]` of subband
    /// `sb` from the previous granule, `i = 0..18`.
    overlap: [[f32; SAMPLES_PER_SUBBAND]; NUM_SUBBANDS],
}

impl Default for ImdctState {
    fn default() -> Self {
        ImdctState {
            overlap: [[0.0; SAMPLES_PER_SUBBAND]; NUM_SUBBANDS],
        }
    }
}

impl ImdctState {
    /// A fresh all-zero overlap store (stream-start state).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The saved second-half overlap of subband `sb` (the values that
    /// will be added to the first half of the next granule's output).
    /// Returns a zero array for out-of-range `sb`.
    #[must_use]
    pub fn overlap(&self, sb: usize) -> [f32; SAMPLES_PER_SUBBAND] {
        if sb < NUM_SUBBANDS {
            self.overlap[sb]
        } else {
            [0.0; SAMPLES_PER_SUBBAND]
        }
    }
}

/// Compute one §2.4.3.4.10.2 IMDCT cosine coefficient
/// `cos( (pi/(2n)) · (2i + 1 + n/2) · (2k + 1) )` for output index `i`,
/// input index `k`, transform size `n`.
///
/// The argument is assembled in the *exact* same evaluation order the
/// naive per-sample loop used (`scale * a * b` where `scale = pi/(2n)`,
/// `a = 2i+1+n/2`, `b = 2k+1`), so a precomputed table of these values
/// carries the bit-identical `f64` that the inline `.cos()` produced.
#[must_use]
fn imdct_cos(n: usize, i: usize, k: usize) -> f64 {
    let half = n / 2;
    let scale = core::f64::consts::PI / (2.0 * n as f64);
    let a = (2 * i + 1 + half) as f64;
    let b = (2 * k + 1) as f64;
    (scale * a * b).cos()
}

/// Precomputed IMDCT cosine matrix for the long transform (`n = 36`):
/// `LONG_COS[i][k] = imdct_cos(36, i, k)`, `i = 0..36`, `k = 0..18`.
///
/// The transform coefficients depend only on the constant index pair
/// `(i, k)` and never on the input data, so every `cos()` is evaluated
/// exactly once at first use and the inner IMDCT loop becomes a plain
/// table lookup. Each entry holds the identical `f64` bit pattern the
/// inline cosine produced, and the products are summed in the same
/// `k = 0..18` order, so the transform result is bit-for-bit identical
/// to evaluating the cosine per sample.
static LONG_COS: std::sync::LazyLock<[[f64; LONG_N / 2]; LONG_N]> =
    std::sync::LazyLock::new(|| {
        let mut m = [[0.0f64; LONG_N / 2]; LONG_N];
        for (i, row) in m.iter_mut().enumerate() {
            for (k, slot) in row.iter_mut().enumerate() {
                *slot = imdct_cos(LONG_N, i, k);
            }
        }
        m
    });

/// Precomputed IMDCT cosine matrix for the short transform (`n = 12`):
/// `SHORT_COS[i][k] = imdct_cos(12, i, k)`, `i = 0..12`, `k = 0..6`.
/// Same bit-identical / same-summation-order guarantee as [`LONG_COS`].
static SHORT_COS: std::sync::LazyLock<[[f64; SHORT_N / 2]; SHORT_N]> =
    std::sync::LazyLock::new(|| {
        let mut m = [[0.0f64; SHORT_N / 2]; SHORT_N];
        for (i, row) in m.iter_mut().enumerate() {
            for (k, slot) in row.iter_mut().enumerate() {
                *slot = imdct_cos(SHORT_N, i, k);
            }
        }
        m
    });

/// The §2.4.3.4.10.2 IMDCT: transform `n/2` input lines `xk` (`n` = 36 or
/// 12) into `n` output samples.
///
/// `xk` must hold exactly `n / 2` values. Returns `n` outputs.
/// Computation is in `f64`; callers downcast to `f32` after windowing.
///
/// The two transform sizes the codec actually uses (36 and 12) look the
/// cosine coefficients up from the precomputed [`LONG_COS`] / [`SHORT_COS`]
/// tables — the products are accumulated in the identical `k` order, so
/// the output is bit-for-bit identical to evaluating each `.cos()` inline.
/// Any other `n` (only reachable from tests) falls back to the direct
/// per-sample cosine evaluation.
#[must_use]
pub fn imdct(xk: &[f64], n: usize) -> Vec<f64> {
    debug_assert_eq!(xk.len(), n / 2, "imdct: xk must have n/2 entries");
    let mut out = vec![0.0f64; n];
    match n {
        LONG_N => {
            let cos = &*LONG_COS;
            for (o, c_row) in out.iter_mut().zip(cos.iter()) {
                let mut acc = 0.0f64;
                for (&ck, &x) in c_row.iter().zip(xk.iter()) {
                    acc += x * ck;
                }
                *o = acc;
            }
        }
        SHORT_N => {
            let cos = &*SHORT_COS;
            for (o, c_row) in out.iter_mut().zip(cos.iter()) {
                let mut acc = 0.0f64;
                for (&ck, &x) in c_row.iter().zip(xk.iter()) {
                    acc += x * ck;
                }
                *o = acc;
            }
        }
        _ => {
            for (i, o) in out.iter_mut().enumerate() {
                let mut acc = 0.0f64;
                for (k, &x) in xk.iter().enumerate() {
                    acc += x * imdct_cos(n, i, k);
                }
                *o = acc;
            }
        }
    }
    out
}

/// Allocation-free long (`n = 36`) IMDCT into a stack array.
///
/// Bit-identical to `imdct(xk, 36)` — same [`LONG_COS`] table, same
/// `k = 0..18` accumulation order — but with no `Vec` allocation for the
/// input gather or the output, so the per-subband IMDCT of a long-family
/// granule runs entirely on the stack.
#[must_use]
fn imdct_long(xk: &[f64; LONG_N / 2]) -> [f64; LONG_N] {
    let cos = &*LONG_COS;
    let mut out = [0.0f64; LONG_N];
    for (o, c_row) in out.iter_mut().zip(cos.iter()) {
        let mut acc = 0.0f64;
        for (&ck, &x) in c_row.iter().zip(xk.iter()) {
            acc += x * ck;
        }
        *o = acc;
    }
    out
}

/// Allocation-free short (`n = 12`) IMDCT into a stack array.
///
/// Bit-identical to `imdct(xk, 12)` — same [`SHORT_COS`] table, same
/// `k = 0..6` accumulation order.
#[must_use]
fn imdct_short(xk: &[f64; SHORT_N / 2]) -> [f64; SHORT_N] {
    let cos = &*SHORT_COS;
    let mut out = [0.0f64; SHORT_N];
    for (o, c_row) in out.iter_mut().zip(cos.iter()) {
        let mut acc = 0.0f64;
        for (&ck, &x) in c_row.iter().zip(xk.iter()) {
            acc += x * ck;
        }
        *o = acc;
    }
    out
}

/// Precomputed long-block window table:
/// `LONG_WINDOW[i] = sin( (pi/36)·(i + 1/2) )`, `i = 0..36`
/// (§2.4.3.4.10.3 a). Each entry is the identical `f64` the inline
/// `.sin()` produced from the same argument, so windowing is bit-exact.
static LONG_WINDOW: std::sync::LazyLock<[f64; LONG_N]> = std::sync::LazyLock::new(|| {
    let mut w = [0.0f64; LONG_N];
    for (i, slot) in w.iter_mut().enumerate() {
        *slot = (core::f64::consts::PI / 36.0) * (i as f64 + 0.5);
        *slot = slot.sin();
    }
    w
});

/// Precomputed short-window table:
/// `SHORT_WINDOW[i] = sin( (pi/12)·(i + 1/2) )`, `i = 0..12`
/// (§2.4.3.4.10.3 d). Bit-identical to the inline `.sin()`.
static SHORT_WINDOW: std::sync::LazyLock<[f64; SHORT_N]> = std::sync::LazyLock::new(|| {
    let mut w = [0.0f64; SHORT_N];
    for (i, slot) in w.iter_mut().enumerate() {
        *slot = (core::f64::consts::PI / 12.0) * (i as f64 + 0.5);
        *slot = slot.sin();
    }
    w
});

/// Long-block (`n = 36`) window value at position `i` (§2.4.3.4.10.3 a):
/// `sin( (pi/36)·(i + 1/2) )`. Reads the precomputed [`LONG_WINDOW`]
/// table (bit-identical to the inline `.sin()` for every `i`).
fn long_window(i: usize) -> f64 {
    LONG_WINDOW[i]
}

/// Short-window value at sub-block position `i = 0..12`
/// (§2.4.3.4.10.3 d): `sin( (pi/12)·(i + 1/2) )`. Reads the precomputed
/// [`SHORT_WINDOW`] table (bit-identical to the inline `.sin()`).
fn short_window(i: usize) -> f64 {
    SHORT_WINDOW[i]
}

/// Window the 36 IMDCT outputs of a non-short block into `z[0..36]` per
/// the block type (`Long`, `Start`, or `End`/stop). Short blocks are
/// handled by [`window_short`].
fn window_long_family(x: &[f64], block_type: BlockType) -> [f64; LONG_N] {
    debug_assert_eq!(x.len(), LONG_N);
    let mut z = [0.0f64; LONG_N];
    match block_type {
        BlockType::Start => {
            // §2.4.3.4.10.3 b (block_type 1, start block).
            for (i, zi) in z.iter_mut().enumerate() {
                *zi = match i {
                    0..=17 => x[i] * long_window(i),
                    18..=23 => x[i],
                    // sin( (pi/12)·(i - 18 + 1/2) ) = short_window(i - 18);
                    // the argument is identical to the SHORT_WINDOW table's,
                    // so this stays bit-exact.
                    24..=29 => x[i] * short_window(i - 18),
                    // 30..=35
                    _ => 0.0,
                };
            }
        }
        BlockType::End => {
            // §2.4.3.4.10.3 c (block_type 3, stop block).
            for (i, zi) in z.iter_mut().enumerate() {
                *zi = match i {
                    0..=5 => 0.0,
                    // sin( (pi/12)·(i - 6 + 1/2) ) = short_window(i - 6);
                    // identical argument to the SHORT_WINDOW table, bit-exact.
                    6..=11 => x[i] * short_window(i - 6),
                    12..=17 => x[i],
                    // 18..=35
                    _ => x[i] * long_window(i),
                };
            }
        }
        // BlockType::Long (block_type 0, normal window) — and any other
        // long-family caller — uses the plain sine window over all 36.
        _ => {
            for (i, zi) in z.iter_mut().enumerate() {
                *zi = x[i] * long_window(i);
            }
        }
    }
    z
}

/// Window and concatenate the three short sub-blocks of a short
/// (`block_type 2`) granule into a 36-value `z` (§2.4.3.4.10.3 d).
///
/// `sub[j]` is the 12-sample IMDCT output of short window `j = 0..3`.
/// Each is windowed with [`short_window`], then the three windowed
/// 12-vectors `y[j]` are overlapped and concatenated per the spec table:
///
/// ```text
/// z[i] = 0                       i = 0..5
///        y0[i-6]                 i = 6..11
///        y0[i-6] + y1[i-12]      i = 12..17
///        y1[i-12] + y2[i-18]     i = 18..23
///        y2[i-18]                i = 24..29
///        0                       i = 30..35
/// ```
fn window_short(sub: &[[f64; SHORT_N]; 3]) -> [f64; LONG_N] {
    // Window each short sub-block.
    let mut y = [[0.0f64; SHORT_N]; 3];
    for j in 0..3 {
        for i in 0..SHORT_N {
            y[j][i] = sub[j][i] * short_window(i);
        }
    }
    let mut z = [0.0f64; LONG_N];
    for (i, zi) in z.iter_mut().enumerate() {
        *zi = match i {
            0..=5 => 0.0,
            6..=11 => y[0][i - 6],
            12..=17 => y[0][i - 6] + y[1][i - 12],
            18..=23 => y[1][i - 12] + y[2][i - 18],
            24..=29 => y[2][i - 18],
            // 30..=35
            _ => 0.0,
        };
    }
    z
}

/// Produce the 36-value windowed block `z` for one subband's 18
/// frequency lines, dispatching on the granule's block type.
///
/// `lines` is the 18-line slice for this subband (already alias-reduced
/// and in subband order). Long-family blocks run a single 36-point
/// IMDCT; short blocks run three 12-point IMDCTs (the 18 lines are the
/// three 6-line windows interleaved by the reorder stage, so
/// `lines[3·k + j]` is frequency line `k` of window `j`).
fn windowed_block(lines: &[f32], gc: &GranuleChannel, subband: usize) -> [f64; LONG_N] {
    let is_short = gc.window_switching_flag && gc.block_type == BlockType::Short;

    // §2.4.2.7 `mixed_block_flag`: "If window_switching_flag==1, then
    // the mixed_block_flag indicates whether lower frequency polyphase
    // filter subbands are coded using normal window type. … the
    // frequency lines corresponding to the two lowest frequency
    // polyphase subbands are transformed with normal window
    // (block_type==0), while the remaining 30 subbands are transformed
    // as block_type[gr][ch]." The flag applies to EVERY window-switched
    // block type — Start (1) and End (3) carry it across a mixed
    // transition so the low subbands stay normal-windowed through the
    // whole burst (the §2.4.3.4 overlap-add only cancels between
    // complementary window halves; a start/end window tail against a
    // normal-window head in subbands 0..2 would leave uncancelled
    // aliasing).
    let mixed_low = gc.window_switching_flag && gc.mixed_block_flag && subband < 2;
    let use_long = !is_short || mixed_low;

    if use_long {
        // Single 36-point IMDCT over the 18 lines (n/2 = 18 = subband
        // size); n = 36. The gather + transform run entirely on the
        // stack (no per-subband Vec allocation) and are bit-identical to
        // `imdct(&xk, 36)`.
        let mut xk = [0.0f64; LONG_N / 2];
        for (slot, &v) in xk.iter_mut().zip(lines.iter()) {
            *slot = f64::from(v);
        }
        let x = imdct_long(&xk);
        // A mixed granule's two lowest subbands use the normal window
        // (block_type 0) regardless of the transmitted block type;
        // a non-mixed long/start/end block uses its own.
        let bt = if mixed_low {
            BlockType::Long
        } else {
            gc.block_type
        };
        window_long_family(&x, bt)
    } else {
        // Short block: three independent 12-point IMDCTs, each over the
        // 6 frequency lines of one window. After reorder, the 18 lines
        // are laid out window-interleaved: line `3·k + j` is frequency
        // `k` (0..6) of window `j` (0..3).
        let mut sub = [[0.0f64; SHORT_N]; 3];
        for (j, sj) in sub.iter_mut().enumerate() {
            let mut xk = [0.0f64; SHORT_N / 2]; // 6 input lines
            for (k, xkk) in xk.iter_mut().enumerate() {
                xkk_assign(xkk, lines, 3 * k + j);
            }
            *sj = imdct12(&xk);
        }
        window_short(&sub)
    }
}

/// Assign `lines[idx]` (as `f64`) into `*dst`, treating an out-of-range
/// index as zero. Kept as a small helper so the short-block gather loop
/// stays free of bounds noise.
fn xkk_assign(dst: &mut f64, lines: &[f32], idx: usize) {
    *dst = if idx < lines.len() {
        f64::from(lines[idx])
    } else {
        0.0
    };
}

/// A fixed-size 12-point IMDCT (`n = 12`) returning a `[f64; 12]`, used
/// for the three short sub-blocks. Delegates to the allocation-free
/// [`imdct_short`] (bit-identical to `imdct(xk, 12)`).
fn imdct12(xk: &[f64; SHORT_N / 2]) -> [f64; SHORT_N] {
    imdct_short(xk)
}

/// Run the full §2.4.3.4.10 IMDCT → windowing → overlap-add →
/// frequency-inversion pipeline for one granule-channel.
///
/// * `xr` is the alias-reduced `[f32; 576]` in subband order (output of
///   [`crate::alias::alias_reduce`]).
/// * `gc` supplies the block type (and `mixed_block_flag` /
///   `window_switching_flag`) that selects the window shapes.
/// * `state` carries the per-subband overlap store across granules; it is
///   read for the previous granule's second half and updated in place
///   with this granule's second half.
///
/// Returns the 32×18 subband-domain time samples (`out[sb][t]`,
/// `t = 0..18`) ready for the polyphase synthesis filterbank, with the
/// §2.4.3.4.10.5 frequency inversion already applied.
#[must_use]
pub fn imdct_granule(
    xr: &[f32; NUM_LINES],
    gc: &GranuleChannel,
    state: &mut ImdctState,
) -> [[f32; SAMPLES_PER_SUBBAND]; NUM_SUBBANDS] {
    let mut out = [[0.0f32; SAMPLES_PER_SUBBAND]; NUM_SUBBANDS];

    for (sb, out_sb) in out.iter_mut().enumerate() {
        let base = sb * SAMPLES_PER_SUBBAND;
        let lines = &xr[base..base + SAMPLES_PER_SUBBAND];

        // IMDCT + windowing → 36-value z.
        let z = windowed_block(lines, gc, sb);

        // §2.4.3.4.10.4 overlap-add: result[i] = z[i] + s_prev[i]; save
        // s_next[i] = z[i+18].
        let prev = state.overlap[sb];
        for (i, slot) in out_sb.iter_mut().enumerate() {
            *slot = z[i] as f32 + prev[i];
        }
        for (i, slot) in state.overlap[sb].iter_mut().enumerate() {
            *slot = z[i + SAMPLES_PER_SUBBAND] as f32;
        }
    }

    // §2.4.3.4.10.5 frequency inversion: every odd time sample of every
    // odd subband is negated.
    for out_sb in out.iter_mut().skip(1).step_by(2) {
        for t in (1..SAMPLES_PER_SUBBAND).step_by(2) {
            out_sb[t] = -out_sb[t];
        }
    }

    out
}

#[cfg(test)]
// The tests recompute every reference value from the §2.4.3.4.10 spec
// formulas in their natural `for i in 0..N { z[i] = f(i) }` shape; the
// index variable is part of the formula (e.g. `z[i] = x[i] · sin((π/36)·
// (i + 1/2))`), not just a buffer iterator, so keeping the explicit range
// makes the test text mirror the spec more faithfully than iterator chains.
#[allow(clippy::needless_range_loop)]
mod tests_inner {
    use super::*;
    include!("imdct_tests.rs");
}
