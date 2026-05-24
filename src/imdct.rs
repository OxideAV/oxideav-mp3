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

/// The §2.4.3.4.10.2 IMDCT: transform `n/2` input lines `xk` (`n` = 36 or
/// 12) into `n` output samples.
///
/// `xk` must hold exactly `n / 2` values. Returns `n` outputs.
/// Computation is in `f64`; callers downcast to `f32` after windowing.
#[must_use]
pub fn imdct(xk: &[f64], n: usize) -> Vec<f64> {
    debug_assert_eq!(xk.len(), n / 2, "imdct: xk must have n/2 entries");
    let half = n / 2;
    let nn = n as f64;
    // pi / (2n) factored out of the cosine argument.
    let scale = core::f64::consts::PI / (2.0 * nn);
    let mut out = vec![0.0f64; n];
    for (i, o) in out.iter_mut().enumerate() {
        // (2i + 1 + n/2): the per-output phase offset.
        let a = (2 * i + 1 + half) as f64;
        let mut acc = 0.0f64;
        for (k, &x) in xk.iter().enumerate() {
            let b = (2 * k + 1) as f64;
            acc += x * (scale * a * b).cos();
        }
        *o = acc;
    }
    out
}

/// Long-block (`n = 36`) window value at position `i` (§2.4.3.4.10.3 a):
/// `sin( (pi/36)·(i + 1/2) )`.
fn long_window(i: usize) -> f64 {
    let arg = (core::f64::consts::PI / 36.0) * (i as f64 + 0.5);
    arg.sin()
}

/// Short-window value at sub-block position `i = 0..12`
/// (§2.4.3.4.10.3 d): `sin( (pi/12)·(i + 1/2) )`.
fn short_window(i: usize) -> f64 {
    let arg = (core::f64::consts::PI / 12.0) * (i as f64 + 0.5);
    arg.sin()
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
                    // sin( (pi/12)·(i - 18 + 1/2) )
                    24..=29 => {
                        let arg = (core::f64::consts::PI / 12.0) * ((i - 18) as f64 + 0.5);
                        x[i] * arg.sin()
                    }
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
                    // sin( (pi/12)·(i - 6 + 1/2) )
                    6..=11 => {
                        let arg = (core::f64::consts::PI / 12.0) * ((i - 6) as f64 + 0.5);
                        x[i] * arg.sin()
                    }
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

    // A mixed block codes its two lowest subbands (0 and 1) with the long
    // window and the rest as short (§2.4.2.7). For those two subbands the
    // long-family path (normal window) is used.
    let use_long = !is_short || (gc.mixed_block_flag && subband < 2);

    if use_long {
        // Single 36-point IMDCT over the 18 lines (n/2 = 18 = subband
        // size); n = 36.
        let xk: Vec<f64> = lines.iter().map(|&v| f64::from(v)).collect();
        let x = imdct(&xk, LONG_N);
        // A mixed block's long subbands use the normal window
        // (block_type 0); a pure long/start/end block uses its own.
        let bt = if is_short {
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
/// for the three short sub-blocks. Wraps [`imdct`] with a stack array.
fn imdct12(xk: &[f64; SHORT_N / 2]) -> [f64; SHORT_N] {
    let v = imdct(xk, SHORT_N);
    let mut out = [0.0f64; SHORT_N];
    out.copy_from_slice(&v);
    out
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
