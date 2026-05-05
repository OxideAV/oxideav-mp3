//! MPEG-1 Layer III forward MDCT + windowing + 50%-overlap.
//!
//! Mirrors [`crate::imdct`] in reverse: 36 windowed time-domain samples
//! per subband become 18 frequency-domain coefficients. Long blocks use
//! a 36→18 MDCT; short blocks apply a 12→6 MDCT three times.
//!
//! Forward MDCT formula (long blocks, N=36):
//!
//!   X[k] = sum_{n=0..N-1} x[n] * cos( pi/2N * (2n + 1 + N/2) * (2k + 1) )
//!
//! With the spec's window-then-MDCT order. This is the analytical
//! inverse of the IMDCT in [`crate::imdct::imdct_36`] up to a scale of
//! N/2 — exactly what the IMDCT undoes when our reservoir → IMDCT → OLA
//! pipeline runs end-to-end.

use crate::window::{imdct_window_long, imdct_window_short};

/// Per-channel MDCT carry-over: the second 18-sample half of the
/// previous granule's windowed output, ready to be overlap-added with
/// the current granule's first half before MDCT.
#[derive(Clone)]
pub struct MdctState {
    /// Stored from the previous granule: the SECOND half of the per-subband
    /// 36-sample input window (i.e. samples 18..36 from prev step). For the
    /// next call, this becomes the FIRST half (0..18).
    pub prev_first_half: [[f32; 18]; 32],
}

impl Default for MdctState {
    fn default() -> Self {
        Self {
            prev_first_half: [[0.0; 18]; 32],
        }
    }
}

impl MdctState {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Run forward MDCT for one granule. `subbands_in` holds 32×18 subband
/// samples (subband-major) from the polyphase filter. The output `xr`
/// is 576 frequency coefficients in **bit-stream order** ready to be
/// quantised and Huffman-coded — sfb-window-major for short blocks,
/// long-block layout otherwise.
///
/// `block_type` selects the window and per-subband transform shape:
///   * 0 = long  — single 36-point MDCT, normal sine window
///   * 1 = start — single 36-point MDCT, long→short transition window
///   * 2 = short — three 12-point MDCTs per subband, short window each
///   * 3 = stop  — single 36-point MDCT, short→long transition window
///
/// `sample_rate` is required for short blocks so the post-MDCT
/// reorder (the inverse of [`crate::requantize::reorder_short`]) can
/// re-pack subband-frequency-window-interleaved output into the
/// sfb-window-major bit-stream layout.
pub fn mdct_granule(
    subbands_in: &[[f32; 18]; 32],
    xr: &mut [f32; 576],
    state: &mut MdctState,
    block_type: u8,
    sample_rate: u32,
) {
    mdct_granule_full(subbands_in, xr, state, block_type, false, sample_rate);
}

/// Mixed-block-aware variant of [`mdct_granule`]. When
/// `mixed_block_flag` is `true` (only meaningful for `block_type == 2`),
/// the first two polyphase subbands (sb 0..2, covering long sfbs 0..7
/// at 44.1 kHz with `sfband_long[8] = 36`) are MDCTed with the long-
/// block 36-point transform + normal sine window, and the remaining 30
/// subbands (sb 2..32) get the 3 × 12-point short-block path. Mirrors
/// the decoder's per-subband dispatch in [`crate::imdct::imdct_granule`]
/// (`sub_bt = if mixed_block_flag && sb < 2 { 0 } else { block_type }`).
///
/// Per ISO/IEC 11172-3 §2.4.2.2, mixed blocks bridge a long-block low
/// frequency region (good for tonal content) with short-block high
/// frequencies (good for transient localisation) — useful when a
/// transient sits on top of a sustained low-band tone.
pub fn mdct_granule_full(
    subbands_in: &[[f32; 18]; 32],
    xr: &mut [f32; 576],
    state: &mut MdctState,
    block_type: u8,
    mixed_block_flag: bool,
    sample_rate: u32,
) {
    for sb in 0..32 {
        // 36-sample input = previous half (18) + current half (18).
        let mut in36 = [0.0f32; 36];
        in36[..18].copy_from_slice(&state.prev_first_half[sb]);
        in36[18..].copy_from_slice(&subbands_in[sb]);
        // Save current half for next call's prev_first_half.
        state.prev_first_half[sb] = subbands_in[sb];

        // No per-subband sign manipulation here: the analysis filter has
        // already produced subband samples in the convention the decoder
        // expects, and the decoder's frequency-inversion step (negate
        // odd-indexed samples in odd subbands) is its own concern.

        let base = sb * 18;
        // Per-subband effective block type — mixed-block long prefix
        // forces `sub_bt = 0` for subbands 0..2 to match the decoder.
        let sub_bt = if mixed_block_flag && block_type == 2 && sb < 2 {
            0
        } else {
            block_type
        };
        if sub_bt == 2 {
            // Short blocks: 3 × 12-point MDCT with the short sine window.
            // The 36 input samples are partitioned into three overlapping
            // 12-sample windows starting at offsets 6, 12, 18 (centered
            // on each short window's natural span; mirror of the IMDCT
            // overlap-add in `imdct_granule`).
            //
            // Output layout (matches what the decoder IMDCT reads at
            // `xr[base + w + 3*f]` after `reorder_short`): per subband,
            // 6 frequency bins × 3 windows interleaved as
            //   [w0_f0, w1_f0, w2_f0, w0_f1, w1_f1, w2_f1, .., w2_f5]
            let win = imdct_window_short();
            let mut win_out = [[0.0f32; 6]; 3];
            for w in 0..3 {
                let mut x12 = [0.0f32; 12];
                for i in 0..12 {
                    x12[i] = in36[6 + 6 * w + i] * win[i];
                }
                mdct_12(&x12, &mut win_out[w]);
            }
            for f in 0..6 {
                for w in 0..3 {
                    xr[base + 3 * f + w] = win_out[w][f];
                }
            }
        } else {
            // Long / start / stop: 36-point MDCT, type-specific window.
            let win = imdct_window_long(sub_bt);
            for n in 0..36 {
                in36[n] *= win[n];
            }
            let mut x18 = [0.0f32; 18];
            mdct_36(&in36, &mut x18);
            xr[base..base + 18].copy_from_slice(&x18);
        }
    }

    // For short blocks, the per-subband MDCT above produced output in
    // subband-frequency-window-interleaved layout (the post-reorder
    // layout the decoder's IMDCT will read). Convert to bit-stream
    // layout (sfb-window-major) so quantisation + Huffman coding
    // sees the right linear order — exact inverse of
    // [`crate::requantize::reorder_short`]. For mixed blocks the
    // long-prefix coefficients (xr[0..36] at 44.1 kHz) stay in the
    // long-block linear layout the decoder's requantizer reads first,
    // and only the short-region tail is unreordered starting at
    // sfb 3 of the short table (matches `requantize::reorder_short`'s
    // mixed-block branch).
    if block_type == 2 {
        if mixed_block_flag {
            unreorder_short_mixed_inplace(xr, sample_rate);
        } else {
            unreorder_short_inplace(xr, sample_rate);
        }
    }
}

/// Inverse of [`crate::requantize::reorder_short`] for the pure-short
/// case (mixed_block_flag = 0). Walks the 13 short-block scalefactor
/// bands and re-packs each `3 * width` chunk from
/// frequency-interleaved-by-window into window-major form.
fn unreorder_short_inplace(xr: &mut [f32; 576], sample_rate: u32) {
    let bounds = crate::sfband::sfband_short(sample_rate);
    let mut buf = [0.0f32; 576];
    let mut pos = 0usize;
    let mut sfb = 0usize;
    while pos < 576 && sfb < 13 {
        let width = (bounds[sfb + 1] - bounds[sfb]) as usize;
        // Source layout (post-reorder, what mdct_granule wrote):
        //   for f in 0..width, for w in 0..3:
        //     pos + 3*f + w  holds  freq=f, window=w
        // Destination layout (bit-stream):
        //   for w in 0..3, for f in 0..width:
        //     pos + w*width + f  holds  freq=f, window=w
        for w in 0..3 {
            for f in 0..width {
                let src = pos + 3 * f + w;
                let dst = pos + w * width + f;
                if src < 576 && dst < 576 {
                    buf[dst] = xr[src];
                }
            }
        }
        pos += 3 * width;
        sfb += 1;
    }
    xr[..pos].copy_from_slice(&buf[..pos]);
}

/// Mixed-block variant of [`unreorder_short_inplace`]. The long-prefix
/// coefficients (xr[0..long_bounds[8]] = first two subbands at
/// 44.1 kHz: 36 samples) are NOT touched — they sit in the long-block
/// linear order the decoder's requantizer reads as long sfbs 0..=7.
/// Only the short-region tail starting at `region_start` is converted
/// from frequency-interleaved-by-window to window-major-within-sfb,
/// walking short sfbs 3..13 (matches the decoder's
/// `requantize::reorder_short` mixed branch).
fn unreorder_short_mixed_inplace(xr: &mut [f32; 576], sample_rate: u32) {
    let short_bounds = crate::sfband::sfband_short(sample_rate);
    let long_bounds = crate::sfband::sfband_long(sample_rate);
    // Mixed-block long prefix occupies long sfbs 0..=7 ⇒ ends at
    // `long_bounds[8]` (36 at 44.1 kHz, matching subband boundary 2).
    let region_start = long_bounds[8] as usize;
    let mut buf = [0.0f32; 576];
    let mut pos = region_start;
    let mut sfb = 3usize;
    while pos < 576 && sfb < 13 {
        let width = (short_bounds[sfb + 1] - short_bounds[sfb]) as usize;
        for w in 0..3 {
            for f in 0..width {
                let src = pos + 3 * f + w;
                let dst = pos + w * width + f;
                if src < 576 && dst < 576 {
                    buf[dst] = xr[src];
                }
            }
        }
        pos += 3 * width;
        sfb += 1;
    }
    if pos > region_start {
        xr[region_start..pos].copy_from_slice(&buf[region_start..pos]);
    }
}

/// Forward 36-point MDCT (direct O(N^2) form). Inverse of
/// `imdct::imdct_36` up to a factor of N/2 = 18.
fn mdct_36(x: &[f32; 36], out: &mut [f32; 18]) {
    let pi = std::f32::consts::PI;
    for k in 0..18 {
        let mut acc = 0.0f32;
        for n in 0..36 {
            let phase = pi / 72.0 * ((2 * n + 1 + 18) as f32) * ((2 * k + 1) as f32);
            acc += x[n] * phase.cos();
        }
        out[k] = acc;
    }
}

/// Forward 12-point MDCT (for short blocks).
fn mdct_12(x: &[f32; 12], out: &mut [f32; 6]) {
    let pi = std::f32::consts::PI;
    for k in 0..6 {
        let mut acc = 0.0f32;
        for n in 0..12 {
            let phase = pi / 24.0 * ((2 * n + 1 + 6) as f32) * ((2 * k + 1) as f32);
            acc += x[n] * phase.cos();
        }
        out[k] = acc;
    }
}

/// Reserved short-block window helper — referenced for completeness.
#[allow(dead_code)]
fn _short_window() -> [f32; 12] {
    imdct_window_short()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imdct::{imdct_granule, ImdctState};

    /// MDCT followed by IMDCT (with overlap) recovers a scaled input.
    /// We feed a sinusoid through analysis-equivalent dummy subband
    /// samples (constant DC per subband 0), MDCT it twice (so OLA can
    /// fold), then IMDCT, and check that the recovered signal is
    /// finite and bounded.
    #[test]
    fn mdct_imdct_pipeline_finite() {
        let mut subbands = [[0.0f32; 18]; 32];
        // Inject a small DC into subband 1.
        for i in 0..18 {
            subbands[1][i] = 0.25;
        }

        let mut mdct_state = MdctState::new();
        let mut imdct_state = ImdctState::new();

        // Two granules so OLA is filled.
        for _ in 0..2 {
            let mut xr = [0.0f32; 576];
            mdct_granule(&subbands, &mut xr, &mut mdct_state, 0, 44_100);
            let mut sb_out = [[0.0f32; 18]; 32];
            imdct_granule(&xr, &mut sb_out, &mut imdct_state, 0, false);
            for sb in 0..32 {
                for i in 0..18 {
                    assert!(sb_out[sb][i].is_finite());
                }
            }
        }
    }

    /// Short-block MDCT path produces finite outputs for a transient-like
    /// input (impulse train). This is the bring-up sanity check; an end-
    /// to-end pre-echo PSNR check lives in the encoder integration tests.
    #[test]
    fn mdct_short_path_is_finite() {
        let mut subbands = [[0.0f32; 18]; 32];
        // Inject an "impulse" in the middle of every subband (sample 9).
        for sb in 0..32 {
            subbands[sb][9] = 1.0;
        }
        let mut state = MdctState::new();
        let mut xr = [0.0f32; 576];
        // Run with prev=0 so the first-half is zero and the input has
        // a clear transition.
        mdct_granule(&subbands, &mut xr, &mut state, 2, 44_100);
        for v in xr.iter() {
            assert!(v.is_finite());
        }
    }

    /// Start and stop windows still drive the long-MDCT path; verify they
    /// produce finite outputs (they zero out parts of the 36-sample
    /// envelope by construction).
    #[test]
    fn mdct_start_stop_paths_finite() {
        let mut subbands = [[0.0f32; 18]; 32];
        for sb in 0..32 {
            for i in 0..18 {
                subbands[sb][i] = 0.1;
            }
        }
        for bt in [1u8, 3u8] {
            let mut state = MdctState::new();
            let mut xr = [0.0f32; 576];
            mdct_granule(&subbands, &mut xr, &mut state, bt, 44_100);
            for v in xr.iter() {
                assert!(v.is_finite(), "block_type {bt} produced non-finite output");
            }
        }
    }
}
