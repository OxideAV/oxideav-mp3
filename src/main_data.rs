//! Layer III **main-data assembly** — composing the per-granule,
//! per-channel scalefactors (part2) and Huffman codewords (part3) into a
//! single contiguous main-data block, plus the bit-reservoir back-pointer
//! `main_data_begin`.
//!
//! This module is the forward (encoder-side) counterpart to the decode
//! split across [`crate::scalefactors`] (part2) and [`crate::huffman`]
//! (part3). The ISO/IEC 11172-3:1993 §2.4.1.7 `main_data()` syntax places
//! the two parts **interleaved per granule/channel**:
//!
//! ```text
//! main_data() {
//!   for (gr=0; gr<2; gr++) for (ch=0; ch<nch; ch++) {
//!       <scalefactors[gr][ch]>      // part2
//!       Huffmancodebits()           // part3
//!   }
//!   <ancillary bits>
//! }
//! ```
//!
//! Each `(gr, ch)`'s part2 + part3 bit count is its `part2_3_length`
//! side-info field. [`assemble_main_data`] walks that loop, emitting the
//! scalefactors of each granule/channel (via
//! [`crate::scalefactors::write_mpeg1_granule_channel`] /
//! [`crate::scalefactors::write_lsf_channel`]) directly followed by the
//! Huffman payload (via [`crate::huffman::emit_huffman`]) into one shared
//! [`MainDataWriter`], with no byte alignment between fields. It records
//! each granule/channel's `part2_3_length` back into the supplied
//! [`SideInfo`], so the side info and the main-data block agree.
//!
//! # The bit reservoir (`main_data_begin`)
//!
//! Layer III main data does not, in general, start immediately after its
//! own frame's side information: it can begin up to `main_data_begin`
//! bytes *earlier*, drawing on bits a previous frame under-used (the bit
//! reservoir, §2.4.2.7 / Figure A.7.a). This module implements the
//! simplest legal schedule — **no reservoir** — in which every frame is
//! self-contained: `main_data_begin == 0`, the entire assembled block
//! lives between this frame's side info and the next sync word, and the
//! decoder reads it without consulting any earlier frame. Real reservoir
//! scheduling (carrying spare bits forward to relax a busy granule's rate
//! limit) is a later round; here `main_data_begin` is always `0`.

use crate::frame::{Mp3FrameHeader, MpegVersion};
use crate::huffman::{emit_huffman, encoder_region_boundaries, HuffmanEncodeError, NUM_LINES};
use crate::scalefactors::{
    is_intensity_stereo, write_lsf_channel, write_mpeg1_granule_channel, FrameScaleFactors,
    MainDataWriter, ScaleFactors,
};
use crate::side_info::SideInfo;

/// The Huffman-partition data of one granule/channel needed to emit its
/// part3 `huffmancodebits()` payload: the granule's 576 quantized lines
/// plus the partition split.
///
/// The big-values **region split** and the codebook selections are *not*
/// carried here — the assembler derives the `(region0_end, region1_end)`
/// boundaries from the granule's side-info `region0_count` /
/// `region1_count` (via [`crate::huffman::encoder_region_boundaries`]),
/// and reads `table_select` / `count1table_select` from the side info,
/// exactly as [`crate::huffman::decode_huffman`] does on read-back. That
/// guarantees the emitted region/table assignment matches the decoder's
/// derivation bit-for-bit, so the caller cannot desync the two.
#[derive(Debug, Clone)]
pub struct GranuleChannelData {
    /// The 576 quantized frequency lines (`is[]`), with signs.
    pub is: [i32; NUM_LINES],
    /// Big-values pair count (lines `0..big_pairs*2`). Must equal the
    /// granule/channel's side-info `big_values`.
    pub big_pairs: usize,
    /// count1 quadruple count (lines `big_pairs*2 .. big_pairs*2+quads*4`).
    pub count1_quads: usize,
}

/// A fully assembled main-data block plus its bit-reservoir back-pointer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssembledMainData {
    /// The contiguous main-data bytes (trailing partial byte zero-padded
    /// to a byte boundary). With `main_data_begin == 0` this block starts
    /// immediately after the frame's side information.
    pub bytes: Vec<u8>,
    /// The §2.4.2.7 bit-reservoir back-pointer. Always `0` in this
    /// no-reservoir schedule (every frame is self-contained).
    pub main_data_begin: u16,
    /// Per-granule, per-channel `part2_3_length` (part2 scalefactor bits +
    /// part3 Huffman bits), indexed `[granule][channel]`. Cells outside
    /// the live granule/channel range are `0`.
    pub part2_3_lengths: [[u16; 2]; 2],
    /// Total payload bit length across every granule/channel (the sum of
    /// the live `part2_3_lengths`), excluding the trailing byte pad.
    pub total_bits: usize,
}

/// Assemble the Layer III main-data block for one frame from known
/// scalefactors and quantized `is[]` data, for the **no-reservoir**
/// (`main_data_begin == 0`) schedule.
///
/// `header` selects the MPEG-1 vs LSF layout and (for the LSF
/// intensity-stereo right channel) the channel mode. `side_info` supplies
/// every per-granule/channel side-info field (`scalefac_compress`,
/// `block_type`, `window_switching_flag`, `mixed_block_flag`, `scfsi`,
/// the Huffman selectors, …) and is **mutated** so each live
/// granule/channel's `part2_3_length` is set to the emitted bit count and
/// `main_data_begin` is set to `0`. `scalefactors` holds the part2 values
/// to emit; `gc_data` holds, indexed `[granule][channel]`, the part3
/// Huffman inputs.
///
/// The output [`AssembledMainData::bytes`] is the contiguous part2+part3
/// payload in §2.4.1.7 `main_data()` order, byte-padded at the end. It
/// reads back through a single [`crate::scalefactors::MainDataReader`]
/// per the same loop: each granule/channel's scalefactors then its
/// Huffman data, recovering the original scalefactors and `is[]`
/// bit-exactly.
///
/// # Errors
///
/// Propagates [`HuffmanEncodeError`] from [`crate::huffman::emit_huffman`]
/// if a granule/channel's `is[]` is not codable by its `table_select`.
pub fn assemble_main_data(
    header: &Mp3FrameHeader,
    side_info: &mut SideInfo,
    scalefactors: &FrameScaleFactors,
    gc_data: &[[GranuleChannelData; 2]; 2],
) -> Result<AssembledMainData, HuffmanEncodeError> {
    let nch = side_info.channels as usize;
    let ngr = side_info.granule_count as usize;
    let intensity = is_intensity_stereo(header);

    let mut w = MainDataWriter::new();
    let mut part2_3_lengths = [[0u16; 2]; 2];

    for gr in 0..ngr {
        for ch in 0..nch {
            let before = w.bit_pos();

            // ---- part2: scalefactors ----
            let sf: &ScaleFactors = &scalefactors.granules[gr][ch];
            match header.version {
                MpegVersion::Mpeg1 => {
                    write_mpeg1_granule_channel(
                        &mut w,
                        &side_info.granules[gr][ch],
                        sf,
                        &side_info.scfsi[ch],
                        gr,
                    );
                }
                MpegVersion::Mpeg2 => {
                    let is_intensity_right = intensity && ch == 1;
                    write_lsf_channel(&mut w, &side_info.granules[gr][ch], sf, is_intensity_right);
                }
            }

            // ---- part3: Huffman codewords ----
            // Derive the region split + table selectors from the side
            // info, exactly as `decode_huffman` will on read-back, so the
            // per-line region/table assignment cannot desync.
            let d = &gc_data[gr][ch];
            let gc = &side_info.granules[gr][ch];
            let region_ends =
                encoder_region_boundaries(gc, d.big_pairs, header.sample_rate_hz, header.version);
            emit_huffman(
                &mut w,
                &d.is,
                d.big_pairs,
                region_ends,
                gc.table_select,
                d.count1_quads,
                gc.count1table_select,
            )?;

            let part2_3 = w.bit_pos() - before;
            part2_3_lengths[gr][ch] = part2_3 as u16;
            side_info.granules[gr][ch].part2_3_length = part2_3 as u16;
        }
    }

    // No reservoir: this frame is self-contained.
    side_info.main_data_begin = 0;

    let total_bits = w.bit_pos();
    Ok(AssembledMainData {
        bytes: w.finish(),
        main_data_begin: 0,
        part2_3_lengths,
        total_bits,
    })
}

#[cfg(test)]
include!("main_data_tests.rs");
