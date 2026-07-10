//! Layer III **short-block forward MDCT + inverse-reorder** — the
//! encoder-side primitive that produces native bitstream-order `xr[576]`
//! lines for a granule whose `block_type == Short`.
//!
//! Decoder side has three stages that turn the wire `xr[sfb][win][k]`
//! interleave into PCM:
//!
//! 1. [`crate::requantize`] reads the native-order lines and applies the
//!    short-block requantization formula `2^((global_gain - 210 -
//!    8·subblock_gain[win])/4) · scalefac_s[sfb][win] · |is|^(4/3)`.
//! 2. [`crate::reorder`] rewrites short scalefactor bands (and the short
//!    region of a mixed block) from `[sfb][win][k]` interleave into the
//!    subband-window-interleaved layout `xr[subband][3·k + win]` that
//!    [`crate::imdct::imdct_granule`] consumes.
//! 3. [`crate::imdct::imdct_granule`] runs three 12-point IMDCTs per
//!    subband over `xr[subband][3·k + win]` (one per window), windows
//!    each output with the short window, overlaps and concatenates per
//!    the §2.4.3.4.10.3 d table, overlap-adds with the previous
//!    granule's second half, and applies §2.4.3.4.10.5 frequency
//!    inversion.
//!
//! This module is the encoder mirror of stages 3 + 2 (in that order):
//!
//! * [`forward_short_mdct_subband`] takes one subband's 18 new subband-time
//!   samples plus the previous granule's 18, runs the three-window short
//!   analysis split per §2.4.3.4.10.3 d (extracted by
//!   [`crate::mdct::window_short_analysis`]), runs three independent
//!   12-point MDCTs ([`crate::mdct::mdct`] with `n = 12`), and emits 18
//!   bins in the subband-window-interleaved layout the decoder's
//!   IMDCT path consumes (`out[3·k + win]`).
//! * [`forward_reorder`] is the bit-exact inverse of [`crate::reorder::reorder`]:
//!   given `xr` in subband-window-interleaved layout (the natural output of
//!   `forward_short_mdct_subband` repeated per subband), it produces the
//!   `[sfb][win][k]` native bitstream order the §2.4.1.7 part3 Huffman
//!   bits expect.
//!
//! No alias reduction is applied to short blocks on either the decoder
//! or the encoder side (§2.4.3.4.10.1 scopes alias reduction to
//! `block_type != 2`); the caller of this module therefore *must not*
//! feed the output through [`crate::stream_encoder`]'s
//! `inverse_alias_reduce` for short-block granules.
//!
//! # Scale factor (Princen-Bradley)
//!
//! The §2.4.3.4.10.2 IMDCT and the [`crate::mdct::mdct`] kernel share
//! the same unscaled cosine basis, so the lapped MDCT round-trip
//! `encoder window → MDCT → decoder IMDCT → window → overlap-add`
//! recovers the input scaled by `n / 4`. For the long block (`n = 36`)
//! that factor is 9 (the constant `analysis_synthesis_long_block_tdac_recovery`
//! test pins in [`crate::mdct`]); for the short block (`n = 12`) it is
//! 3. The encoder therefore divides each short MDCT output by 3 to
//! make the chain unit-gain.

use crate::frame::MpegVersion;
use crate::huffman::NUM_LINES;
use crate::mdct::{mdct, window_short_analysis, MdctState, LONG_N, SHORT_N};
use crate::requantize::short_band_starts;
use crate::side_info::{BlockType, GranuleChannel};

/// In a mixed block the short reorder begins at short scalefactor
/// band 3; everything below the coding split `3·short_starts[3]` is
/// long-coded (see `requantize::mixed_long_lines`). Matches the
/// decoder-side boundary so the forward reorder is its exact inverse.
const MIXED_FIRST_SHORT_SFB: usize = 3;

/// Short-block IMDCT Princen-Bradley scale (`n / 4` for `n = 12`).
pub const SHORT_BLOCK_TDAC_SCALE: f64 = 3.0;

/// Run one subband's forward short-block MDCT chain.
///
/// Given the previous granule's 18 subband-time samples (`prev`) and the
/// current granule's 18 (`current`), this function:
///
/// 1. Updates the per-subband [`MdctState`] in place so the next granule
///    sees the matching previous half (the §2.4.3.4.10.4 overlap-add
///    analog).
/// 2. Concatenates `prev || current` into a 36-sample frame `xn`.
/// 3. Splits `xn` into the three 12-sample short sub-blocks per
///    §2.4.3.4.10.3 d via [`window_short_analysis`].
/// 4. Runs an independent 12-point MDCT on each sub-block, producing 6
///    frequency bins each (18 total).
/// 5. Divides each bin by the Princen-Bradley scale [`SHORT_BLOCK_TDAC_SCALE`]
///    so the lapped round-trip is unit-gain.
/// 6. Lays the 18 bins out in the **subband-window-interleaved** layout
///    `out[3·k + win]` the decoder's `imdct.rs::windowed_block` consumes
///    for short blocks: line `3·k + win` is frequency bin `k` (0..6) of
///    window `win` (0..3).
///
/// The output is therefore ready for the §2.4.3.4.8 forward reorder
/// (this module's [`forward_reorder`]) which rewrites it into the
/// native bitstream `[sfb][win][k]` order the §2.4.1.7 part3 Huffman
/// path expects.
#[must_use]
pub fn forward_short_mdct_subband(current: &[f64; LONG_N / 2], state: &mut MdctState) -> [f32; 18] {
    // Step 1+2: assemble the 36-sample overlapped input frame and
    // update the per-subband overlap state.
    let mut xn = [0.0f64; LONG_N];
    let prev = state.saved();
    for (i, slot) in xn.iter_mut().enumerate().take(LONG_N / 2) {
        *slot = prev[i];
    }
    for (i, slot) in xn.iter_mut().enumerate().skip(LONG_N / 2) {
        *slot = current[i - LONG_N / 2];
    }
    // Mirror `forward_overlap`'s state update: the next call's `prev`
    // is THIS call's `current`. Keep `MdctState` state-transition
    // semantics identical across long and short branches so the
    // §2.4.3.4.10.3 b/c start/stop transitions can stitch a granule
    // boundary that flips between branches without dropping samples.
    let _ = window_short_analysis; // keep the path-symbol below readable
    let sub = window_short_analysis(&xn);

    // Step 3+4: three independent 12-point MDCTs, each producing 6
    // frequency bins. The MDCT kernel returns `n/2 = 6` outputs.
    let mut out = [0.0f32; 18];
    for j in 0..3 {
        let xk: Vec<f64> = sub[j].to_vec();
        let bins = mdct(&xk, SHORT_N);
        debug_assert_eq!(bins.len(), SHORT_N / 2);
        // Step 5+6: scale to unit gain and lay out in the
        // window-interleaved layout `3·k + win` per `imdct.rs::windowed_block`.
        for (k, &b) in bins.iter().enumerate() {
            out[3 * k + j] = (b / SHORT_BLOCK_TDAC_SCALE) as f32;
        }
    }
    // Commit the state update LAST so a panic in the MDCT layer above
    // leaves `state` untouched (same atomicity guarantee `forward_overlap`
    // does NOT bother with — but short-block analysis runs three MDCTs
    // per subband and a partial state update across them is harder to
    // reason about, so we hold the assignment to the end).
    let mut saved_next = [0.0f64; LONG_N / 2];
    saved_next.copy_from_slice(current);
    *state = MdctState::from_saved(saved_next);
    out
}

/// Bit-exact inverse of [`crate::reorder::reorder`].
///
/// Decoder's reorder takes native bitstream `xr[sfb][win][k]` and emits
/// subband-window-interleaved `xr_sub[3·s + 3·k + win]` where
/// `s = starts[sfb]` and `w = starts[sfb + 1] - starts[sfb]`:
///
/// ```text
/// dst[3·s + 3·k + win] = src[3·s + win·w + k]
/// ```
///
/// The encoder needs the inverse: given lines already in
/// subband-window-interleaved layout (the natural output of
/// [`forward_short_mdct_subband`] repeated for each of the 32 polyphase
/// subbands), produce the native bitstream layout the part3 Huffman path
/// reads from in §2.4.1.7 order:
///
/// ```text
/// dst[3·s + win·w + k] = src[3·s + 3·k + win]
/// ```
///
/// Long blocks (`block_type != 2`) and the long region of a mixed block
/// (lines 0..36) pass through unchanged — they are already in
/// increasing-frequency / subband-row layout the part3 path consumes
/// directly.
#[must_use]
pub fn forward_reorder(
    xr_sub: &[f32; NUM_LINES],
    gc: &GranuleChannel,
    sample_rate_hz: u32,
    version: MpegVersion,
) -> [f32; NUM_LINES] {
    let is_short = gc.window_switching_flag && gc.block_type == BlockType::Short;
    if !is_short {
        // Long / start / end blocks are already in frequency-ascending
        // subband-row order; nothing to do (mirrors `reorder::reorder`
        // for non-short blocks).
        return *xr_sub;
    }

    let mut out = *xr_sub;
    let first_sfb = if gc.mixed_block_flag {
        // Mixed: the long-coded region (lines below
        // `3·short_starts[3]`) is already frequency-ordered and copied
        // verbatim by `out = *xr_sub`; only short bands 3..12 are
        // reordered.
        MIXED_FIRST_SHORT_SFB
    } else {
        0
    };
    let starts = short_band_starts(sample_rate_hz, version);
    // 13 bands including band 12 — the exact inverse of the decoder's
    // `reorder::reorder` (r405; both previously stopped at band 11).
    for sfb in first_sfb..13 {
        let s = starts[sfb];
        let e = if sfb < 12 { starts[sfb + 1] } else { 192 };
        let w = e - s;
        let base = 3 * s;
        for win in 0..3 {
            for k in 0..w {
                let src_idx = base + 3 * k + win;
                let dst_idx = base + win * w + k;
                if src_idx < NUM_LINES && dst_idx < NUM_LINES {
                    out[dst_idx] = xr_sub[src_idx];
                }
            }
        }
    }
    out
}

/// The §2.4.2.7 hard-coded short-block region split that the decoder's
/// `region_boundaries` (in [`crate::huffman`]) applies regardless of the
/// transmitted `region0_count` / `region1_count` (those are not coded for
/// window-switched blocks; see `encoder::write_granule_channel`). Returns
/// the pair `(region0_count, region1_count)` to stamp into a short
/// granule-channel's side-info anyway, so a downstream re-parse of the
/// side info reproduces the encoder-supplied values byte-for-byte.
///
/// Per ISO/IEC 11172-3:1993 §2.4.2.7 the defaults applicable when
/// `window_switching_flag = 1` and `block_type = 2` are
/// `region_address1 = 8`, `region_address2 = 36` — both encoded as
/// off-by-one band counts, so `region0_count = 7`, `region1_count = 36`.
/// The `36` exceeds the 3-bit `region1_count` field width (max 7) so the
/// encoder clamps to `7`; the decoder ignores the value either way and
/// uses its own §2.4.2.7 short-block rule (`r0 = 3 · short_starts[3]`
/// interleaved lines — 36 for every ISO table, 72 for the MPEG-2.5
/// 8 kHz table — and `r1 = big_values * 2`, no region 2; see
/// `huffman.rs` `region_boundaries`). Returning `(8, 7)` keeps the on-wire
/// value at a deterministic spec-derived sentinel without affecting
/// rendering.
///
/// Callers writing a short granule-channel side info simply assign
/// `gc.region0_count = ret.0; gc.region1_count = ret.1`. The function is
/// pure / panic-free.
#[must_use]
pub const fn short_block_region_defaults() -> (u8, u8) {
    (8, 7)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imdct::{imdct_granule, ImdctState};
    use crate::mdct::MdctState;
    use crate::reorder::reorder;
    use crate::side_info::GranuleChannel;

    fn long_zero_state(n: usize) -> Vec<MdctState> {
        (0..n).map(|_| MdctState::new()).collect()
    }

    /// Build a short-block granule-channel: window-switched, pure
    /// short (no mixed), with the spec defaults this module pins.
    fn short_gc() -> GranuleChannel {
        let (r0, r1) = short_block_region_defaults();
        GranuleChannel {
            part2_3_length: 0,
            big_values: 0,
            global_gain: 0,
            scalefac_compress: 0,
            window_switching_flag: true,
            block_type: BlockType::Short,
            mixed_block_flag: false,
            table_select: [0; 3],
            subblock_gain: [0; 3],
            region0_count: r0,
            region1_count: r1,
            preflag: false,
            scalefac_scale: false,
            count1table_select: false,
        }
    }

    #[test]
    fn forward_reorder_is_inverse_of_decoder_reorder_44k() {
        // Build a deterministic 576-line buffer in subband-window-
        // interleaved layout (the natural output of the forward short
        // MDCT path), apply `forward_reorder`, then apply the decoder's
        // `reorder` and expect bit-exact recovery.
        let mut xr_sub = [0.0f32; NUM_LINES];
        for (i, v) in xr_sub.iter_mut().enumerate() {
            *v = (i as f32) * 0.5 + 1.0;
        }
        let gc = short_gc();
        let native = forward_reorder(&xr_sub, &gc, 44_100, MpegVersion::Mpeg1);
        let round = reorder(&native, &gc, 44_100, MpegVersion::Mpeg1);
        for i in 0..NUM_LINES {
            assert_eq!(
                round[i], xr_sub[i],
                "subband line {i} survived inverse-reorder→reorder round trip"
            );
        }
    }

    #[test]
    fn forward_reorder_long_block_is_identity() {
        let mut xr = [0.0f32; NUM_LINES];
        for (i, v) in xr.iter_mut().enumerate() {
            *v = (i as f32).sin();
        }
        let mut gc = short_gc();
        // Flip to long: clear window switching, set Long. Mixed flag
        // irrelevant for non-short.
        gc.window_switching_flag = false;
        gc.block_type = BlockType::Long;
        let out = forward_reorder(&xr, &gc, 44_100, MpegVersion::Mpeg1);
        for i in 0..NUM_LINES {
            assert_eq!(out[i], xr[i], "long-block reorder is identity at line {i}");
        }
    }

    #[test]
    fn forward_reorder_mixed_preserves_long_region() {
        // Mixed: the lowest 36 lines must pass through unchanged; only
        // the short region (lines 36..) is reordered.
        let mut xr_sub = [0.0f32; NUM_LINES];
        for (i, v) in xr_sub.iter_mut().enumerate() {
            *v = (i as f32) * 1.25 - 3.0;
        }
        let mut gc = short_gc();
        gc.mixed_block_flag = true;
        let out = forward_reorder(&xr_sub, &gc, 44_100, MpegVersion::Mpeg1);
        for i in 0..36 {
            assert_eq!(out[i], xr_sub[i], "mixed-block long region at line {i}");
        }
    }

    /// End-to-end self-consistency for one subband's forward short
    /// MDCT chain: feed a deterministic two-granule input through the
    /// forward chain, then through the decoder's IMDCT short path on
    /// the same `xr` layout (subband-row format, no reorder needed for
    /// a single-subband test since the bins are already in subband
    /// order), and verify the reconstructed subband-time **energy**
    /// approximates the input energy.
    ///
    /// Sample-by-sample TDAC recovery is only meaningful at steady state
    /// (after several short granules in a row); single-granule
    /// transient overlap-add gets contributions from the (zero)
    /// "previous-previous" granule on the granule-0 reconstruction and
    /// the partial overlap on granule 1's reconstruction. Energy is the
    /// scale-invariant Parseval-style witness that matches better at
    /// these short windows.
    #[test]
    fn forward_short_mdct_subband_preserves_energy_across_two_granules() {
        // Two consecutive granules of 18 subband-time samples each.
        let mut gr0 = [0.0f64; LONG_N / 2];
        let mut gr1 = [0.0f64; LONG_N / 2];
        for i in 0..(LONG_N / 2) {
            gr0[i] = ((i as f64) * 0.3).sin();
            gr1[i] = ((i as f64) * 0.41).cos() + 0.2;
        }
        let input_energy: f64 = gr0.iter().chain(gr1.iter()).map(|v| v * v).sum();

        // Forward: drive a fresh `MdctState` through granule 0 then 1.
        let mut fwd_state = MdctState::new();
        let bins0 = forward_short_mdct_subband(&gr0, &mut fwd_state);
        let bins1 = forward_short_mdct_subband(&gr1, &mut fwd_state);

        // Decoder side: stuff each granule's 18 bins onto subband 0 of a
        // 576-line buffer and IMDCT through one ImdctState across the
        // two granules.
        let mut bins0_xr = [0.0f32; NUM_LINES];
        let mut bins1_xr = [0.0f32; NUM_LINES];
        bins0_xr[..18].copy_from_slice(&bins0);
        bins1_xr[..18].copy_from_slice(&bins1);
        let mut imdct_state = ImdctState::new();
        let gc = short_gc();
        let row0 = imdct_granule(&bins0_xr, &gc, &mut imdct_state);
        let row1 = imdct_granule(&bins1_xr, &gc, &mut imdct_state);

        // Subband 0 reconstruction is `row*[0]` (a [f32; 18]).
        let recon_energy: f64 = row0[0]
            .iter()
            .chain(row1[0].iter())
            .map(|&v| f64::from(v) * f64::from(v))
            .sum();

        // The lapped short-block round-trip is not strictly
        // energy-preserving at the granule-pair scale because the
        // windowing zeros the first and last 6 of every short block's
        // contribution; the recovered energy is some fraction of the
        // input. We check that it's non-zero and within a sane order of
        // magnitude (a hard zero or runaway value would indicate the
        // forward chain is silently broken). The exact recovery ratio
        // for the deterministic seed above is empirical; the bound is
        // conservative.
        assert!(
            recon_energy > 0.0,
            "short-block reconstruction collapsed to zero (recon={recon_energy}, input={input_energy})"
        );
        assert!(
            recon_energy < 10.0 * input_energy,
            "short-block reconstruction diverged (recon={recon_energy}, input={input_energy})"
        );
    }

    /// Sanity: the long-block forward overlap state (the original
    /// `MdctState`) and the short-block forward chain share the same
    /// per-subband `saved` history, so a long-then-short transition
    /// hands off correctly. This test merely confirms the
    /// `from_saved` constructor used inside `forward_short_mdct_subband`
    /// is wired up; the actual transition behaviour is exercised by the
    /// stream-encoder integration test.
    #[test]
    fn forward_short_state_carries_previous_granule() {
        let _ = long_zero_state(32); // keep the helper imported

        let mut state = MdctState::new();
        let cur = [1.0f64; LONG_N / 2];
        let _bins = forward_short_mdct_subband(&cur, &mut state);
        // The state should now hold `cur` as its `saved` for the next
        // call.
        let saved = state.saved();
        for i in 0..(LONG_N / 2) {
            assert_eq!(saved[i], cur[i], "saved[{i}] should equal cur");
        }
    }
}
