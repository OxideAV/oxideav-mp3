//! # oxideav-mp3
//!
//! **Status:** clean-room rebuild in progress (reset 2026-05-24).
//!
//! The prior implementation was retired under the workspace clean-room
//! policy: several of its data tables and decode-loop structures were
//! documented as having been consulted from external reference
//! implementations (their source, not the ISO/IEC specification),
//! which violates the clean-room provenance requirement regardless of
//! those references' licensing. The crate is being re-implemented from
//! scratch against ISO/IEC 11172-3:1993 and ISO/IEC 13818-3:1997
//! (numeric tables read only from those standards).
//!
//! ## What is implemented
//!
//! The [`frame`] module provides the MPEG audio **framing** layer:
//! the four-byte frame-header parser ([`frame::parse_header`] →
//! [`frame::Mp3FrameHeader`]), per-frame byte-length computation
//! including the padding slot, and a self-delimiting
//! [`frame::FrameWalker`] that iterates frames over a byte buffer with
//! mid-stream resynchronisation on bad sync.
//!
//! The [`side_info`] module parses the Layer III **side-information**
//! block for both layouts: MPEG-1 (ISO/IEC 11172-3 §2.4.1.7 /
//! §2.4.2.7) and MPEG-2 / MPEG-2.5 lower-sampling-frequency (ISO/IEC
//! 13818-3 §2.4.1.7 / §2.4.2.7). [`side_info::parse_side_info`] →
//! [`side_info::SideInfo`] dispatches on the header's
//! [`MpegVersion`], covering `main_data_begin`,
//! `private_bits`, MPEG-1 `scfsi`, and the full per-granule-per-channel
//! [`side_info::GranuleChannel`] record for both the long-block and
//! window-switching branches. The LSF form has one granule, an 8-bit
//! `main_data_begin`, a 9-bit `scalefac_compress`, and no `scfsi`.
//!
//! The [`scalefactors`] module implements the Layer III **scalefactor
//! decode** stage — the main-data step between side-information parsing
//! and Huffman decode. It models the main-data bit reservoir
//! ([`scalefactors::Reservoir`] / [`scalefactors::MainDataReader`]) and
//! reads the per-granule-per-channel scalefactors via
//! [`scalefactors::decode_scalefactors`] for both MPEG-1 (ISO/IEC
//! 11172-3 §2.4.2.7, with `slen1`/`slen2` from `scalefac_compress` and
//! `scfsi` reuse across granules) and MPEG-2 / MPEG-2.5 LSF (ISO/IEC
//! 13818-3 §2.4.3.4, deriving `slen1..slen4` + `nr_of_sfb` + `preflag`
//! + `intensity_scale` from the 9-bit `scalefac_compress`).
//!
//! The [`huffman`] module decodes the Layer III main-data
//! **Huffman** stage — the `huffmancodebits()` syntax of ISO/IEC
//! 11172-3:1993 §2.4.1.7 (`Huffmancodebits()` on p.18) with the
//! semantics of §2.4.2.7 (p.26–28). [`huffman::decode_huffman`]
//! produces the 576 quantized frequency lines `is[0..576]` of one
//! granule-channel from the three-region big-values partition
//! (region boundaries derived from `region0_count` / `region1_count`
//! and Table 3-B.8 long-block band-start indices, with codebook
//! selection per `table_select` over **all** Table 3-B.7 entries
//! 0..=31 — the small/medium tables 0..=13, the large 16×16 tables 15,
//! 16 and 24, and the linbits aliases 17..=23 (table 16 codes) and
//! 25..=31 (table 24 codes); tables 4 and 14 are "not used"), followed
//! by the count1 quadruple partition (table A or B per
//! `count1table_select`) decoded until the granule's part-3 bit
//! budget is exhausted; the remaining lines are zero-filled.
//!
//! The [`requantize`] module implements the Layer III
//! **requantization** stage — ISO/IEC 11172-3:1993 §2.4.3.4.7.1 — which
//! turns the 576 quantized integer lines `is[576]` of one
//! granule-channel into 576 float frequency lines `xr[576]`.
//! [`requantize::requantize`] applies the power-law `|is|^(4/3)`, the
//! `global_gain`/`subblock_gain` exponential, and the per-scalefactor-band
//! scalefactor (with the [`requantize::PRETAB`] preemphasis table from
//! Annex B.6 when `preflag` is set), covering the long-block formula,
//! the short-block per-window form with `subblock_gain`, the mixed-block
//! split (long bands 0..8 / lines 0..36, then short bands 3..12), and
//! the LSF variant (which shares the same §2.4.3.4 formula).
//!
//! The [`reorder`] module implements the Layer III **short-block
//! reordering** stage — ISO/IEC 11172-3:1993 §2.4.3.4.8 — which rewrites
//! the requantized short-block lines from their native
//! `(scf_band, window, freqline)` Huffman interleave into subband order
//! `xr[subband][window][freqline]`, so each consecutive run of 18 lines
//! forms one polyphase subband (6 frequency lines × 3 windows) for the
//! IMDCT. [`reorder::reorder`] reorders pure short blocks and the short
//! region of mixed blocks (short bands 3..12, lines 36..) while leaving
//! long blocks and the mixed-block long region (lines 0..36) unchanged.
//!
//! The [`stereo`] module implements the Layer III **stereo processing**
//! stage — ISO/IEC 11172-3:1993 §2.4.3.4.9 (with the ISO/IEC 13818-3:1997
//! §2.4.3.2 intensity modifications for MPEG-2 / MPEG-2.5 LSF) — which
//! reconstructs the left/right channels of a joint-stereo granule from the
//! transmitted mid/side and intensity-position representations.
//! [`stereo::process_stereo`] applies the MS matrix
//! (`L = (M+S)/√2`, `R = (M-S)/√2`) and/or intensity stereo (per-band
//! `is_pos` taken from the right channel's scalefactors) per the
//! `mode_extension` header bits, deriving the intensity bound from the
//! last non-zero right-channel line (per window for short blocks) and
//! covering the MPEG-1 `tan(is_pos·π/12)` formula plus the LSF power-law
//! `i0` factors selected by `intensity_scale`.
//!
//! The [`alias`] module implements the Layer III **alias reduction**
//! stage — ISO/IEC 11172-3:1993 §2.4.3.4.10.1 — the eight-butterfly
//! decorrelation across each subband boundary that precedes the IMDCT.
//! [`alias::alias_reduce`] applies the §2.4.3.4.10.1 pseudo code over the
//! 31 subband boundaries of a granule-channel's reordered `xr[576]`,
//! using the butterfly coefficients `cs[i] = 1/√(1+c[i]²)` and
//! `ca[i] = c[i]/√(1+c[i]²)` derived from the Table 3-B.9 raw
//! coefficients ([`alias::ALIAS_C`]); granules with `block_type == 2`
//! (short or mixed) pass through unchanged per the spec's literal
//! `block_type`-only test.
//!
//! The [`imdct`] module implements the Layer III **IMDCT, windowing,
//! overlap-add and frequency inversion** — ISO/IEC 11172-3:1993
//! §2.4.3.4.10.2 / §2.4.3.4.10.3 / §2.4.3.4.10.4 / §2.4.3.4.10.5 — the
//! per-subband transform stack that runs after alias reduction and
//! produces the 32×18 subband-domain time samples consumed by the
//! polyphase synthesis filterbank (a later stage). [`imdct::imdct_granule`]
//! runs the 36-point or three-12-point IMDCT, applies the
//! [`side_info::BlockType`]-specific window
//! (normal / start / short(3×) / stop, including a mixed block's two
//! lowest long subbands), overlap-adds the saved second half of the
//! previous granule via [`imdct::ImdctState`], saves the new second
//! half, and negates every odd time sample of every odd subband.
//!
//! The [`synth`] module implements the **polyphase synthesis subband
//! filterbank** — ISO/IEC 11172-3:1993 §2.4.3.2 / Figure A.2 — the last
//! decode stage. [`synth::synth_granule`] consumes one granule-channel's
//! 32×18 subband-time block (the output of [`imdct::imdct_granule`]) and
//! emits 576 PCM samples per granule per channel, running 18 sequential
//! [`synth::synth_row`] passes over the 1024-value [`synth::SynthState`]
//! shift register, the 64×32 matrixing
//! `N[i,k] = cos((16+i)·(2k+1)·π/64)`, the [`synth::D_TABLE`]
//! 512-coefficient window, and the 16-tap summation.
//!
//! ## What is not implemented yet
//!
//! No frame-driver / decoder API and no encoder. The PCM-producing
//! pipeline (Huffman → requantize → reorder → stereo → alias → IMDCT →
//! synthesis) is now complete end-to-end at the granule level: feed an
//! [`huffman::decode_huffman`]-produced `[i32; 576]` through the stack
//! and out comes a `[f32; 576]` PCM run. The Huffman stage covers
//! **all** Table 3-B.7 codebooks (0..=31 minus the unused 4 and 14).
//! [`register`] is a no-op until a [`Decoder`]/[`Demuxer`] is wired up,
//! so the public decode/encode surface still returns
//! [`Error::NotImplemented`].
//!
//! [`Decoder`]: oxideav_core::Decoder
//! [`Demuxer`]: oxideav_core::Demuxer

#![warn(missing_debug_implementations)]

pub mod alias;
pub mod frame;
pub mod huffman;
pub mod imdct;
pub mod reorder;
pub mod requantize;
pub mod scalefactors;
pub mod side_info;
pub mod stereo;
pub mod synth;

pub use alias::{alias_ca, alias_cs, alias_reduce, ALIAS_C};
pub use frame::{
    parse_header, ChannelMode, Emphasis, Frame, FrameWalker, HeaderError, Layer, ModeExtension,
    Mp3FrameHeader, MpegVersion,
};
pub use huffman::{decode_huffman, HuffmanError, NUM_LINES};
pub use imdct::{imdct_granule, ImdctState, SAMPLES_PER_SUBBAND};
pub use reorder::reorder;
pub use requantize::{requantize, scalefac_multiplier, PRETAB};
pub use scalefactors::{
    decode_scalefactors, lsf_scale_params, FrameScaleFactors, LsfScaleParams, MainDataReader,
    Reservoir, ScaleFactorError, ScaleFactors, LONG_SFB, MPEG1_SLEN, SHORT_SFB, SHORT_WINDOWS,
};
pub use side_info::{
    parse_side_info, BlockType, GranuleChannel, SideInfo, SideInfoError, GRANULES, GRANULES_LSF,
    SIDE_INFO_BYTES_LSF_MONO, SIDE_INFO_BYTES_LSF_STEREO, SIDE_INFO_BYTES_MONO,
    SIDE_INFO_BYTES_STEREO,
};
pub use stereo::process_stereo;
pub use synth::{n_coefficient, synth_granule, synth_row, SynthState, D_TABLE, PCM_PER_GRANULE};

use oxideav_core::RuntimeContext;

/// Crate-local error type. Until the clean-room rebuild lands every
/// public API path returns [`Error::NotImplemented`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// The crate has been reset to a scaffold pending clean-room
    /// rebuild; no decoder or encoder functionality is wired up yet.
    NotImplemented,
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "oxideav-mp3: orphan-rebuild scaffold — no codec wired up"
        )
    }
}

impl std::error::Error for Error {}

/// No-op codec registration — the orphan-rebuild scaffold registers
/// nothing into the runtime context.
pub fn register(_ctx: &mut RuntimeContext) {}

oxideav_core::register!("mp3", register);
