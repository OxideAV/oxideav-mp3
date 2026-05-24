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
//! ## What is not implemented yet
//!
//! No Layer III main-data decode (scalefactor reader, Huffman,
//! requantise, IMDCT, synthesis filterbank) and no encoder.
//! [`register`] is a no-op until a [`Decoder`]/[`Demuxer`] is wired up,
//! so the public decode/encode surface still returns
//! [`Error::NotImplemented`].
//!
//! [`Decoder`]: oxideav_core::Decoder
//! [`Demuxer`]: oxideav_core::Demuxer

#![warn(missing_debug_implementations)]

pub mod frame;
pub mod side_info;

pub use frame::{
    parse_header, ChannelMode, Emphasis, Frame, FrameWalker, HeaderError, Layer, ModeExtension,
    Mp3FrameHeader, MpegVersion,
};
pub use side_info::{
    parse_side_info, BlockType, GranuleChannel, SideInfo, SideInfoError, GRANULES, GRANULES_LSF,
    SIDE_INFO_BYTES_LSF_MONO, SIDE_INFO_BYTES_LSF_STEREO, SIDE_INFO_BYTES_MONO,
    SIDE_INFO_BYTES_STEREO,
};

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
