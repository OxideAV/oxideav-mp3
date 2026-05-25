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
//! limit) lives below in [`schedule_reservoir`] /
//! [`ReservoirScheduler`].
//!
//! # Cross-frame bit-reservoir scheduling
//!
//! When a sequence of frames is encoded with fixed per-frame slot sizes
//! (the §2.4.2.3 padding-bit ladder), each frame's `slot_bytes`
//! (`frame_len - 4 - CRC? - side_info_bytes`) is constant in CBR. A busy
//! frame may produce more assembled main-data bytes than its own slot can
//! hold, while a previous quiet frame may have left bytes unused. The
//! §2.4.2.7 "bit reservoir" is the rolling tail of those unused bytes:
//! the busy frame's main_data is emitted into the prior frame(s)' unused
//! tail(s), and its `main_data_begin` is set to point that many bytes
//! back from the start of its own slot.
//!
//! The encoder-side scheduler enforces the §2.4.2.7 invariants
//!
//! 1. `R_i ≥ 0` for every frame (cumulative slot bytes ≥ cumulative
//!    main-data bytes at every step) — otherwise frame i's main_data
//!    would have to start *after* its slot, which `main_data_begin`
//!    (a non-negative back-pointer) cannot express.
//! 2. `R_i ≤ 511` (MPEG-1, 9 bits) or `R_i ≤ 255` (MPEG-2 / MPEG-2.5
//!    LSF, 8 bits) — the on-wire field's bit width.
//!
//! With `L_i` the main-data byte length of frame i and `S_i` its slot
//! byte length, the reservoir evolves as `R_{i+1} = R_i + S_i - L_i`,
//! and `main_data_begin_i = R_i` (the start of frame i's main_data sits
//! `R_i` bytes before the start of its slot, in the rolling main-data
//! concatenation).
//!
//! Crucially, a frame's main_data extends *backward* into earlier
//! frames' slots — never *forward* into later slots. A busy frame
//! `L_i > S_i` is schedulable only when earlier quiet frames have left
//! enough unused tail in the rolling concatenation: it requires
//! `R_i ≥ L_i - S_i`, i.e. enough prior reservoir to absorb the overflow.
//! The scheduler buffers all frames before assigning slots so a quiet
//! frame 0 can reserve its tail bytes for a busy frame 1 that follows
//! (and so on); single-pass append-as-you-go is insufficient.
//!
//! On the wire: slot i contains bytes `MD[Σⱼ<ᵢ Sⱼ : Σⱼ<ᵢ Sⱼ + S_i]`
//! where `MD` is the concatenation of every frame's main_data in order.
//! On read-back, [`crate::scalefactors::Reservoir`] keeps the trailing
//! 512 bytes of slot history; `Reservoir::assemble(main_data_begin_i,
//! slot_i)` returns the `main_data_begin_i` prior bytes + this slot's
//! bytes — exactly the run that contains frame i's main_data starting
//! at offset 0 and extending through the byte at offset `L_i - 1`.

use crate::frame::{Mp3FrameHeader, MpegVersion};
use crate::huffman::{emit_huffman, encoder_region_boundaries, HuffmanEncodeError, NUM_LINES};
use crate::scalefactors::{
    is_intensity_stereo, write_lsf_channel, write_mpeg1_granule_channel, FrameScaleFactors,
    MainDataWriter, ScaleFactors,
};
use crate::side_info::SideInfo;

/// The §2.4.2.7 bit-reservoir limit in bytes: `main_data_begin` is 9 bits
/// in MPEG-1 (ISO/IEC 11172-3 §2.4.1.7) and 8 bits in MPEG-2 / MPEG-2.5
/// LSF (ISO/IEC 13818-3 §2.4.1.7).
pub const RESERVOIR_MAX_MPEG1: usize = 511;
/// LSF reservoir limit — 8-bit `main_data_begin` ⇒ 255 bytes.
pub const RESERVOIR_MAX_LSF: usize = 255;

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

/// One frame's contribution to the bit-reservoir schedule: the assembled
/// main-data byte stream plus the per-frame main-data slot capacity in
/// bytes (`frame_len - 4 - CRC? - side_info_bytes`).
///
/// The scheduler treats `main_data` as opaque — it does not look inside —
/// and only requires `slot_bytes` to be ≥ the side-info-implied minimum
/// of zero. Free-format frames have no fixed slot and are out of scope.
#[derive(Debug, Clone)]
pub struct ReservoirFrame<'a> {
    /// This frame's assembled main-data bytes (typically the
    /// [`AssembledMainData::bytes`] of [`assemble_main_data`]).
    pub main_data: &'a [u8],
    /// The per-frame main-data slot size in bytes — the number of bytes
    /// between this frame's side info and the next frame's sync word.
    /// Computed as `header.frame_len() - 4 - (2 if CRC else 0) -
    /// side_info_bytes`.
    pub slot_bytes: usize,
    /// `true` for MPEG-2 / MPEG-2.5 LSF frames (8-bit `main_data_begin`,
    /// 255-byte reservoir cap); `false` for MPEG-1 (9-bit, 511-byte).
    pub lsf: bool,
}

/// Per-frame schedule output: the bytes to write into the frame's
/// main-data slot plus the `main_data_begin` value to stamp into its
/// side info.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduledFrame {
    /// Exactly `slot_bytes` bytes — what goes into this frame's
    /// main-data area on the wire. The leading bytes (up to
    /// `main_data_begin` of the *next* frame) may be the tail of a
    /// previous frame's overflow; the trailing bytes (up to a future
    /// frame's `main_data_begin`) may be unused and form part of the
    /// rolling reservoir.
    pub slot: Vec<u8>,
    /// The `main_data_begin` to stamp into this frame's side info: the
    /// number of bytes back from the start of this slot to the start of
    /// this frame's main_data in the rolling concatenation. `0` when the
    /// frame's main_data fits in its own slot and starts at the slot's
    /// first byte.
    pub main_data_begin: u16,
}

/// Errors produced by the reservoir scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReservoirError {
    /// A frame's main-data length plus the running reservoir would
    /// overflow the §2.4.2.7 9-bit (MPEG-1, 511 bytes) or 8-bit
    /// (LSF, 255 bytes) `main_data_begin` cap on the next frame. The
    /// payload of frame `frame_index` is too big to fit the schedule:
    /// the bitstream's per-frame budget has to be raised (a higher
    /// bitrate, or padding-bit selection) before re-trying.
    ReservoirOverflow {
        /// Index (0-based) into the input frame slice where the overflow
        /// first occurs.
        frame_index: usize,
        /// The reservoir size that would have resulted, in bytes.
        attempted: usize,
        /// The reservoir cap for this frame's layout (511 or 255).
        cap: usize,
    },
    /// The running main-data total exceeds the running slot total — the
    /// next frame's main_data would have to start *after* its slot
    /// begins, which `main_data_begin` (a non-negative offset) cannot
    /// express. The bitstream is unschedulable: the busy frame at
    /// `frame_index` exceeded the combined prior reservoir + own slot
    /// budget.
    SlotUnderflow {
        /// Index (0-based) at which the underflow first occurs.
        frame_index: usize,
        /// Main-data bytes emitted by `frames[frame_index]`.
        main_data_len: usize,
        /// Slot bytes available to `frames[frame_index]`.
        slot_bytes: usize,
        /// Reservoir bytes carried in from prior frames (≥ 0).
        prior_reservoir: usize,
    },
}

impl core::fmt::Display for ReservoirError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ReservoirError::ReservoirOverflow {
                frame_index,
                attempted,
                cap,
            } => write!(
                f,
                "bit reservoir overflows after frame {frame_index}: \
                 would carry {attempted} bytes, cap is {cap} (§2.4.2.7)"
            ),
            ReservoirError::SlotUnderflow {
                frame_index,
                main_data_len,
                slot_bytes,
                prior_reservoir,
            } => write!(
                f,
                "frame {frame_index} main_data ({main_data_len} B) exceeds \
                 slot ({slot_bytes} B) + prior reservoir ({prior_reservoir} B): \
                 main_data_begin cannot point forward"
            ),
        }
    }
}

impl std::error::Error for ReservoirError {}

/// Schedule an entire frame sequence onto the §2.4.2.7 bit reservoir.
///
/// All frames are required up front because the on-wire layout —
/// slot i = `MD[Σⱼ<ᵢ Sⱼ : Σⱼ<ᵢ Sⱼ + S_i]` over the rolling main-data
/// concatenation `MD` — requires the encoder to know *future* frames'
/// main-data before deciding what to put in *this* frame's slot tail.
/// A quiet frame that precedes a busy one writes the busy frame's
/// overflow tail into its own slot's unused bytes; a single-pass
/// append-as-you-go scheduler can't do that.
///
/// The function:
///
/// 1. Validates the §2.4.2.7 schedulability invariants: at every
///    prefix `i`, `Σⱼ<ᵢ Lⱼ ≤ Σⱼ<ᵢ Sⱼ` (reservoir non-negative) and
///    `R_i = Σⱼ<ᵢ Sⱼ − Σⱼ<ᵢ Lⱼ ≤ cap` (within the on-wire bit width).
///    Otherwise returns [`ReservoirError::SlotUnderflow`] or
///    [`ReservoirError::ReservoirOverflow`] respectively.
/// 2. Builds the rolling main-data concatenation `MD`, zero-padding the
///    final tail so the last frame's slot can be sliced out even if its
///    main_data ends mid-slot.
/// 3. Carves `MD` into per-frame slots, computes each frame's
///    `main_data_begin_i = Σⱼ<ᵢ Sⱼ − Σⱼ<ᵢ Lⱼ`, stamps it into
///    `side_infos[i].main_data_begin`, and returns the per-frame
///    [`ScheduledFrame`]s.
///
/// The decoder-side [`crate::scalefactors::Reservoir`] is the inverse:
/// fed slot-i then asked `assemble(main_data_begin_i, slot_i)` it
/// returns the run that begins exactly at frame i's main_data start
/// byte.
///
/// # Errors
///
/// * [`ReservoirError::SlotUnderflow`] if at some prefix the cumulative
///   main-data has overrun the cumulative slot bytes (the busy frame
///   needs more headroom than prior frames could leave).
/// * [`ReservoirError::ReservoirOverflow`] if at some prefix the
///   reservoir size would exceed the §2.4.2.7 9-bit (MPEG-1) or 8-bit
///   (LSF) cap.
///
/// # Panics
///
/// Panics if `frames.len() != side_infos.len()`.
pub fn schedule_reservoir(
    frames: &[ReservoirFrame<'_>],
    side_infos: &mut [SideInfo],
) -> Result<Vec<ScheduledFrame>, ReservoirError> {
    assert_eq!(
        frames.len(),
        side_infos.len(),
        "schedule_reservoir: frames and side_infos must be parallel"
    );

    // Pass 1: validate the cumulative invariants and compute each
    // frame's main_data_begin (R_i) up front.
    let mut cum_l: usize = 0;
    let mut cum_s: usize = 0;
    let mut mdbs: Vec<u16> = Vec::with_capacity(frames.len());
    for (i, f) in frames.iter().enumerate() {
        // R_i = cum_S - cum_L (the reservoir going INTO frame i).
        if cum_l > cum_s {
            return Err(ReservoirError::SlotUnderflow {
                frame_index: i,
                main_data_len: f.main_data.len(),
                slot_bytes: f.slot_bytes,
                prior_reservoir: 0, // already underflowed before frame i
            });
        }
        let r_i = cum_s - cum_l;
        let cap = if f.lsf {
            RESERVOIR_MAX_LSF
        } else {
            RESERVOIR_MAX_MPEG1
        };
        if r_i > cap {
            return Err(ReservoirError::ReservoirOverflow {
                frame_index: i,
                attempted: r_i,
                cap,
            });
        }
        // Schedulability for THIS frame: L_i ≤ R_i + S_i.
        if f.main_data.len() > r_i + f.slot_bytes {
            return Err(ReservoirError::SlotUnderflow {
                frame_index: i,
                main_data_len: f.main_data.len(),
                slot_bytes: f.slot_bytes,
                prior_reservoir: r_i,
            });
        }
        mdbs.push(r_i as u16);
        cum_l += f.main_data.len();
        cum_s += f.slot_bytes;
    }

    // Pass 2: build the rolling main-data concatenation, zero-padding
    // the tail to the total slot length so the final slot can be sliced.
    let total_slot: usize = frames.iter().map(|f| f.slot_bytes).sum();
    let mut md = Vec::with_capacity(total_slot.max(cum_l));
    for f in frames {
        md.extend_from_slice(f.main_data);
    }
    if md.len() < total_slot {
        md.resize(total_slot, 0);
    }

    // Pass 3: carve into per-frame slots, stamp side info, and emit.
    let mut out: Vec<ScheduledFrame> = Vec::with_capacity(frames.len());
    let mut slot_start = 0usize;
    for (i, f) in frames.iter().enumerate() {
        let slot_end = slot_start + f.slot_bytes;
        let slot = md[slot_start..slot_end].to_vec();
        side_infos[i].main_data_begin = mdbs[i];
        out.push(ScheduledFrame {
            slot,
            main_data_begin: mdbs[i],
        });
        slot_start = slot_end;
    }
    Ok(out)
}

/// Stateful step-by-step variant of [`schedule_reservoir`] for callers
/// that prefer to schedule incrementally — e.g. when the frame sequence
/// is being produced by a generator. The semantics are identical to the
/// one-shot function: scheduling is done in two passes (compute every
/// frame's `main_data_begin` up front, then emit slots) so callers must
/// still feed every frame via [`ReservoirScheduler::push_frame`] before
/// observing the final slot bytes; the per-frame return from
/// `push_frame` carries the up-front-computed `main_data_begin` and a
/// pending slot index, and [`ReservoirScheduler::finish`] then emits the
/// concrete slot bytes once the rolling concatenation is complete.
///
/// In this round the type exists primarily as documentation and as a
/// future-proofing handle; the one-shot [`schedule_reservoir`] is the
/// expected entry point.
#[derive(Debug, Default)]
pub struct ReservoirScheduler {
    frames: Vec<OwnedReservoirFrame>,
}

#[derive(Debug)]
struct OwnedReservoirFrame {
    main_data: Vec<u8>,
    slot_bytes: usize,
    lsf: bool,
}

impl ReservoirScheduler {
    /// A new, empty scheduler.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of frames queued so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// `true` when no frames have been queued.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Queue one frame for later scheduling. The frame's main_data is
    /// copied internally.
    pub fn push(&mut self, frame: &ReservoirFrame<'_>) {
        self.frames.push(OwnedReservoirFrame {
            main_data: frame.main_data.to_vec(),
            slot_bytes: frame.slot_bytes,
            lsf: frame.lsf,
        });
    }

    /// Finalise: schedule every queued frame, stamping each side info's
    /// `main_data_begin` and returning the per-frame slot outputs. See
    /// [`schedule_reservoir`] for the contract.
    ///
    /// # Errors
    ///
    /// Propagates the [`ReservoirError`] from the underlying
    /// [`schedule_reservoir`] call.
    ///
    /// # Panics
    ///
    /// Panics if `side_infos.len() != self.len()`.
    pub fn finish(
        self,
        side_infos: &mut [SideInfo],
    ) -> Result<Vec<ScheduledFrame>, ReservoirError> {
        let frames: Vec<ReservoirFrame<'_>> = self
            .frames
            .iter()
            .map(|f| ReservoirFrame {
                main_data: &f.main_data,
                slot_bytes: f.slot_bytes,
                lsf: f.lsf,
            })
            .collect();
        schedule_reservoir(&frames, side_infos)
    }
}

#[cfg(test)]
include!("main_data_tests.rs");
