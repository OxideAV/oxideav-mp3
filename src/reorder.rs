//! Layer III **short-block reordering** — the §2.4.3.4.8 stage that runs
//! after §2.4.3.4.7 requantization and before alias reduction / the
//! IMDCT.
//!
//! After requantization the short-block (and mixed-block short-region)
//! frequency lines are still laid out in the *native* Huffman/bitstream
//! interleave `xr[scf_band][window][freqline]` (ISO/IEC 11172-3:1993
//! §2.4.2.7, "The order of the Huffman data depends on the block_type …
//! If block_type==2 the Huffman encoded data is given for successive
//! scalefactor bands … within each scalefactor band, the data is given
//! for successive time windows … within each window, the quantized
//! values are arranged in order of increasing frequency"). §2.4.3.4.8
//! requires this be reordered into **subband order**
//! `xr[subband[window][freqline]]` prior to the IMDCT.
//!
//! # What the reorder does (§2.4.3.4.8)
//!
//! For a granule whose `block_type == 2` (short windows), the rescaled
//! data `xr[scf_band][window][freqline]` "shall be reordered in subband
//! order, `xr[subband][window][freqline]`". Concretely: within a short
//! scalefactor band whose per-window start is `s` and per-window width is
//! `w` (from the Table B.8 short-block band-start column), requantize
//! laid the band out as three consecutive runs of `w` lines —
//! `[win0: s..s+w][win1: s..s+w][win2: s..s+w]` — occupying interleaved
//! positions `3·s .. 3·(s+w)`. The reorder rewrites that same span so
//! the three windows of one frequency line sit adjacent:
//!
//! ```text
//! out[3·s + 3·k + win] = in[3·s + win·w + k]   for k in 0..w, win in 0..3
//! ```
//!
//! Concatenated across bands this yields a buffer whose every 18
//! consecutive lines form one polyphase subband (6 frequency lines × 3
//! windows), which is exactly the unit the §2.4.3.4.10 synthesis
//! filterbank consumes ("the frequency lines … are fed into the IMDCT
//! matrix, each 18 into one transform block"; §2.4.2.7: short blocks
//! produce "three transforms … producing 12 output values each" from 6
//! input lines per window).
//!
//! # Long and mixed blocks
//!
//! Long blocks (`block_type != 2`) are already in increasing-frequency
//! order (§2.4.2.7) and pass through unchanged. A **mixed** block
//! (`block_type == 2 && mixed_block_flag`) codes the lowest 36 lines as a
//! long window and the remainder as short windows (§2.4.2.7); only the
//! short region (short scalefactor bands 3..12, interleaved lines from
//! `3·12 = 36` upward) is reordered, the long region 0..36 passes
//! through.
//!
//! Every band boundary used here is read from the same Table B.8 columns
//! as the requantize stage; this module re-uses those start tables via
//! [`crate::requantize`].

use crate::frame::MpegVersion;
use crate::requantize::{short_band_starts, NUM_LINES};
use crate::side_info::{BlockType, GranuleChannel};

/// In a mixed block the two lowest polyphase subbands (36 lines,
/// interleaved line `3·12 = 36`) are coded long; the short reorder begins
/// at short scalefactor band 3 (per-window line 12). Matches
/// `requantize`'s `MIXED_LONG_LINES` / `MIXED_FIRST_SHORT_SFB`.
const MIXED_FIRST_SHORT_SFB: usize = 3;

/// Reorder the requantized frequency lines of one granule-channel from
/// the native short-block interleave into subband order per ISO/IEC
/// 11172-3:1993 §2.4.3.4.8.
///
/// * `xr` is the `[f32; 576]` buffer produced by
///   [`crate::requantize::requantize`], with short-block lines still in
///   `(sfb, window, freqline)` interleave.
/// * `gc` supplies `window_switching_flag`, `block_type`, and
///   `mixed_block_flag` to decide whether (and which portion) to reorder.
/// * `sample_rate_hz` / `version` select the Table B.8 short-block band
///   layout (identical column used by the requantize stage).
///
/// Returns a fresh `[f32; 576]` in subband order. Long blocks
/// (`block_type != 2`) are copied through unchanged; mixed blocks copy
/// their long region (lines 0..36) and reorder only the short region.
#[must_use]
pub fn reorder(
    xr: &[f32; NUM_LINES],
    gc: &GranuleChannel,
    sample_rate_hz: u32,
    version: MpegVersion,
) -> [f32; NUM_LINES] {
    let is_short = gc.window_switching_flag && gc.block_type == BlockType::Short;
    if !is_short {
        // Long / start / end blocks are already in increasing-frequency
        // order (§2.4.2.7); nothing to do.
        return *xr;
    }

    let mut out = *xr;
    let first_sfb = if gc.mixed_block_flag {
        // The long region (lines 0..36) is already frequency-ordered and
        // copied verbatim by the `out = *xr` initialisation above; only
        // short bands 3..12 are reordered.
        MIXED_FIRST_SHORT_SFB
    } else {
        0
    };
    reorder_short_region(&mut out, xr, first_sfb, sample_rate_hz, version);
    out
}

/// Reorder short scalefactor bands `[first_sfb, 12)` of `src` into `dst`,
/// mapping native `(sfb, window, freqline)` interleave to subband order
/// (§2.4.3.4.8). `dst` must already hold a copy of any pass-through
/// (long / mixed-long) region; this function only writes the short span.
fn reorder_short_region(
    dst: &mut [f32; NUM_LINES],
    src: &[f32; NUM_LINES],
    first_sfb: usize,
    sample_rate_hz: u32,
    version: MpegVersion,
) {
    let starts = short_band_starts(sample_rate_hz, version);
    // 13 bands: band 12 (per-window lines `starts[12]..192`, no
    // transmitted scalefactor) is short-window data like any other and
    // must be reordered from the native `(sfb, window, freqline)`
    // interleave too (r405; the loop previously stopped at band 11,
    // leaving band 12 in native order).
    for sfb in first_sfb..13 {
        let s = starts[sfb];
        let e = if sfb < 12 { starts[sfb + 1] } else { 192 };
        let w = e - s;
        // Native span of this band: 3 runs of `w` lines starting at 3·s.
        let base = 3 * s;
        for win in 0..3 {
            for k in 0..w {
                let src_idx = base + win * w + k;
                let dst_idx = base + 3 * k + win;
                if src_idx < NUM_LINES && dst_idx < NUM_LINES {
                    dst[dst_idx] = src[src_idx];
                }
            }
        }
    }
}

#[cfg(test)]
include!("reorder_tests.rs");
