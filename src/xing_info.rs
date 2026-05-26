//! Xing / Info VBR information-frame **emission** — the encode-side
//! inverse of [`crate::demuxer::parse_xing_info`].
//!
//! ## What this module does
//!
//! Builds the byte run that goes immediately after the Layer III
//! side-info bytes of an MPEG audio frame's payload, encoding up to
//! four optional fields described by `flags`:
//!
//! | bit | field    | width  | meaning                                            |
//! | --- | -------- | ------ | -------------------------------------------------- |
//! |  0  | frames   | BE u32 | Total MPEG audio frame count following this frame. |
//! |  1  | bytes    | BE u32 | Total compressed-audio byte count following.       |
//! |  2  | toc      | 100 B  | Per-percentile byte-offset lookup table.           |
//! |  3  | quality  | BE u32 | Encoder quality indicator (0..=100).               |
//!
//! Fields are emitted in increasing bit order — exactly the order
//! [`crate::demuxer::parse_xing_info`] consumes them on the decoder
//! side, so `parse_xing_info(write_xing_info(spec)) == spec` for any
//! `spec` whose `flags` enumerate only the four bits listed above
//! (higher flag bits are preserved on the wire but not interpreted).
//!
//! ## Layout (verified against
//! `docs/audio/mp3/fixtures/layer3-with-xing-vbri-tag/input.mp3` +
//! `trace.txt` and the symmetric `parse_xing_info`)
//!
//! ```text
//! +0..+3   "Xing" or "Info" magic (selects [`XingTagId`])
//! +4..+7   flags (BE u32; low four bits enumerated)
//! +8..+11  frames     (BE u32, present iff flags & 0x1)
//! +12..+15 bytes      (BE u32, present iff flags & 0x2)
//! +16..+115 toc[100]  (raw bytes, present iff flags & 0x4)
//! +116..+119 quality  (BE u32, present iff flags & 0x8)
//! ```
//!
//! Successive fields collapse: when `flags & 0x1 == 0` the
//! `bytes` field (if present) starts at +8 rather than +12, and so on.
//!
//! ## Carrier-frame placement
//!
//! The Xing / Info tag is **not** an MP3 frame in its own right. It is
//! the leading payload of an otherwise-silent MPEG audio frame —
//! header + side-info + main-data-slot whose main-data-slot begins with
//! the Xing/Info magic. Decoders that ignore the tag still see a valid
//! Layer III frame and (because the silent side-info has every
//! `part2_3_length == 0`, every `big_values == 0`) reconstruct silence
//! from it. Decoders that recognise the tag (including
//! [`crate::Mp3Demuxer`]) consume the leading frame as metadata and
//! emit packets starting at the next audio frame.
//!
//! This module ships [`build_info_frame`] as the helper that bakes a
//! [`XingTagSpec`] into a complete on-wire CBR carrier frame
//! (header + side-info + Xing/Info magic + flagged fields + zero pad
//! out to [`crate::Mp3FrameHeader::frame_len`]).

use crate::demuxer::{XingTag, XingTagId};
use crate::encoder::{encode_silent_frame, EncodeError};
use crate::frame::Mp3FrameHeader;

/// Author-side specification of a Xing / Info VBR information-frame
/// payload.
///
/// `flags` is the inclusive OR of the four [`flag_bit`] enumerated
/// values; each present field's [`Option`] must be `Some(_)` when its
/// flag bit is set, else [`build_xing_info_payload`] returns
/// [`XingEmitError::FlagFieldMismatch`].
#[derive(Debug, Clone)]
pub struct XingTagSpec {
    /// Tag id: `Xing` for true-VBR streams, `Info` for CBR streams
    /// carrying the same structure for VBR-aware seekers.
    pub id: XingTagId,
    /// The full flag word (low four bits enumerated). Higher bits are
    /// written verbatim but never trigger additional field emission;
    /// fields they would describe are not part of this module's scope.
    pub flags: u32,
    /// Total MPEG audio frame count following the info frame, written
    /// when `flags & flag_bit::FRAMES` is set.
    pub frames: Option<u32>,
    /// Total compressed-audio byte count following the info frame,
    /// written when `flags & flag_bit::BYTES` is set.
    pub bytes: Option<u32>,
    /// 100-entry seek table, written when `flags & flag_bit::TOC` is
    /// set. Entry `i` holds `floor(256 * file_offset / bytes)` for the
    /// playback position `i / 100` of the audio region.
    pub toc: Option<[u8; 100]>,
    /// Encoder quality indicator (0..=100), written when
    /// `flags & flag_bit::QUALITY` is set.
    pub quality: Option<u32>,
}

/// Enumerated low-four-bit flag positions used by [`XingTagSpec`].
pub mod flag_bit {
    /// `frames` field present.
    pub const FRAMES: u32 = 0x1;
    /// `bytes` field present.
    pub const BYTES: u32 = 0x2;
    /// `toc[100]` field present.
    pub const TOC: u32 = 0x4;
    /// `quality` field present.
    pub const QUALITY: u32 = 0x8;
    /// Inclusive OR of all four enumerated flag bits.
    pub const ALL_FOUR: u32 = FRAMES | BYTES | TOC | QUALITY;
}

/// Errors returned by [`build_xing_info_payload`] and
/// [`build_info_frame`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XingEmitError {
    /// A flag bit is set in `flags` but the corresponding
    /// [`XingTagSpec`] field is `None`, or vice versa. The wire layout
    /// would be ambiguous, so the build is rejected.
    FlagFieldMismatch,
    /// The carrier frame's main-data slot is too small to hold the
    /// fully-flagged Xing payload. Raise the bitrate or drop optional
    /// fields and retry. (At 128 kbit/s mono 44.1 kHz the slot is
    /// 396 bytes — well above the 120-byte max payload.)
    PayloadTooLarge,
    /// Propagated from [`encode_silent_frame`] when the supplied header
    /// is not a Layer III CBR header.
    Header(EncodeError),
}

impl core::fmt::Display for XingEmitError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            XingEmitError::FlagFieldMismatch => {
                f.write_str("Xing/Info flag bit set but field missing (or vice versa)")
            }
            XingEmitError::PayloadTooLarge => {
                f.write_str("Xing/Info payload exceeds main-data slot")
            }
            XingEmitError::Header(e) => write!(f, "header build: {e}"),
        }
    }
}

impl std::error::Error for XingEmitError {}

impl From<EncodeError> for XingEmitError {
    fn from(e: EncodeError) -> Self {
        XingEmitError::Header(e)
    }
}

impl XingTagSpec {
    /// Convert into the [`XingTag`] struct returned by
    /// [`crate::demuxer::parse_xing_info`]. Used by the round-trip
    /// tests to compare the decoded-side struct against the
    /// emitted-side spec.
    #[must_use]
    pub fn as_xing_tag(&self) -> XingTag {
        XingTag {
            id: self.id,
            flags: self.flags,
            frames: self.frames,
            bytes: self.bytes,
            toc: self.toc,
            quality: self.quality,
        }
    }
}

/// Maximum payload size produced by [`build_xing_info_payload`] when
/// every enumerated flag bit (`FRAMES | BYTES | TOC | QUALITY`) is
/// set: 4 magic + 4 flags + 4 frames + 4 bytes + 100 toc + 4 quality =
/// 120 bytes.
pub const MAX_PAYLOAD_BYTES: usize = 4 + 4 + 4 + 4 + 100 + 4;

/// Build the Xing / Info payload run from `spec`.
///
/// The returned bytes hold the magic + flag word + every flagged
/// optional field, in increasing flag-bit order. The bytes are
/// positioned by the caller — typically immediately after the Layer III
/// side-info bytes of an MPEG audio frame, where
/// [`crate::demuxer::parse_xing_info`] expects them.
///
/// # Errors
///
/// * [`XingEmitError::FlagFieldMismatch`] when a flag bit is set
///   without its matching field, or a field is supplied without its
///   flag bit (the wire encoding would be ambiguous in either case).
pub fn build_xing_info_payload(spec: &XingTagSpec) -> Result<Vec<u8>, XingEmitError> {
    // Cross-validate flag bits against carried `Option` fields.
    let has = |bit: u32| spec.flags & bit != 0;
    if has(flag_bit::FRAMES) != spec.frames.is_some()
        || has(flag_bit::BYTES) != spec.bytes.is_some()
        || has(flag_bit::TOC) != spec.toc.is_some()
        || has(flag_bit::QUALITY) != spec.quality.is_some()
    {
        return Err(XingEmitError::FlagFieldMismatch);
    }

    let mut out: Vec<u8> = Vec::with_capacity(MAX_PAYLOAD_BYTES);
    out.extend_from_slice(magic_bytes(spec.id));
    out.extend_from_slice(&spec.flags.to_be_bytes());
    if let Some(n) = spec.frames {
        out.extend_from_slice(&n.to_be_bytes());
    }
    if let Some(n) = spec.bytes {
        out.extend_from_slice(&n.to_be_bytes());
    }
    if let Some(toc) = spec.toc {
        out.extend_from_slice(&toc);
    }
    if let Some(q) = spec.quality {
        out.extend_from_slice(&q.to_be_bytes());
    }
    Ok(out)
}

/// Map a [`XingTagId`] to its four-byte ASCII magic prefix.
fn magic_bytes(id: XingTagId) -> &'static [u8; 4] {
    match id {
        XingTagId::Xing => b"Xing",
        XingTagId::Info => b"Info",
    }
}

/// Build a complete on-wire **carrier** info frame for the Xing / Info
/// `spec` and a target `header` describing the carrier frame's
/// envelope.
///
/// The carrier frame is a silent Layer III frame
/// ([`encode_silent_frame`]) — header + side-info + zero-filled main
/// data — into which the Xing / Info payload is patched at the start
/// of the main-data slot (the position [`crate::demuxer::parse_xing_info`]
/// reads from). The returned byte sequence has length
/// `header.frame_len()` and is a complete, self-delimiting MP3 frame
/// suitable for direct emission as the **first** frame of a CBR
/// stream.
///
/// The carrier frame is itself counted by neither the Xing `frames`
/// field nor the `bytes` field — both refer to the audio frames that
/// **follow** it. Callers driving a top-level encoder over a
/// known-length audio run can therefore set
/// `frames = audio_frame_count` and `bytes = audio_byte_count` without
/// adjusting for the carrier frame's contribution. The carrier frame's
/// own size is `header.frame_len()` and lives outside both totals.
///
/// # Errors
///
/// * [`XingEmitError::Header`] propagated from [`encode_silent_frame`]
///   when `header` is not a CBR Layer III header.
/// * [`XingEmitError::FlagFieldMismatch`] from
///   [`build_xing_info_payload`].
/// * [`XingEmitError::PayloadTooLarge`] when the payload does not fit
///   inside the carrier frame's main-data slot (4 header bytes + N
///   side-info bytes consumed; the rest is the slot).
pub fn build_info_frame(
    header: &Mp3FrameHeader,
    spec: &XingTagSpec,
) -> Result<Vec<u8>, XingEmitError> {
    // Start from a silent carrier frame: header + side_info + zero
    // main-data slot, length == header.frame_len().
    let mut frame = encode_silent_frame(header)?;
    // Patch the Xing/Info payload over the leading bytes of the
    // main-data slot. The silent encode forces no CRC, so the slot
    // starts at offset 4 (header) + side_info bytes.
    let payload = build_xing_info_payload(spec)?;
    let nch = header.channel_count();
    let lsf = header.version == crate::frame::MpegVersion::Mpeg2;
    let si_bytes = match (lsf, nch == 1) {
        (false, true) => crate::side_info::SIDE_INFO_BYTES_MONO,
        (false, false) => crate::side_info::SIDE_INFO_BYTES_STEREO,
        (true, true) => crate::side_info::SIDE_INFO_BYTES_LSF_MONO,
        (true, false) => crate::side_info::SIDE_INFO_BYTES_LSF_STEREO,
    };
    let slot_start = 4 + si_bytes;
    let slot_end = frame.len();
    if slot_start + payload.len() > slot_end {
        return Err(XingEmitError::PayloadTooLarge);
    }
    frame[slot_start..slot_start + payload.len()].copy_from_slice(&payload);
    Ok(frame)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::demuxer::{parse_xing_info, side_info_len};
    use crate::frame::{parse_header, ChannelMode};
    use crate::make_silent_header;

    fn spec_all_fields() -> XingTagSpec {
        XingTagSpec {
            id: XingTagId::Xing,
            flags: flag_bit::ALL_FOUR,
            frames: Some(32),
            bytes: Some(6114),
            toc: Some([7u8; 100]),
            quality: Some(0),
        }
    }

    #[test]
    fn payload_layout_all_four_fields() {
        let spec = spec_all_fields();
        let payload = build_xing_info_payload(&spec).expect("build payload");
        // 4 magic + 4 flags + 4 frames + 4 bytes + 100 toc + 4 quality.
        assert_eq!(payload.len(), MAX_PAYLOAD_BYTES);
        assert_eq!(&payload[..4], b"Xing");
        assert_eq!(&payload[4..8], &0x0000_000f_u32.to_be_bytes());
        assert_eq!(&payload[8..12], &32u32.to_be_bytes());
        assert_eq!(&payload[12..16], &6114u32.to_be_bytes());
        assert_eq!(&payload[16..116], &[7u8; 100][..]);
        assert_eq!(&payload[116..120], &0u32.to_be_bytes());
    }

    #[test]
    fn payload_layout_info_magic() {
        let mut spec = spec_all_fields();
        spec.id = XingTagId::Info;
        let payload = build_xing_info_payload(&spec).unwrap();
        assert_eq!(&payload[..4], b"Info");
    }

    #[test]
    fn payload_layout_frames_only() {
        let spec = XingTagSpec {
            id: XingTagId::Xing,
            flags: flag_bit::FRAMES,
            frames: Some(0x1234_5678),
            bytes: None,
            toc: None,
            quality: None,
        };
        let payload = build_xing_info_payload(&spec).unwrap();
        // 4 magic + 4 flags + 4 frames = 12 bytes.
        assert_eq!(payload.len(), 12);
        assert_eq!(&payload[8..12], &0x1234_5678u32.to_be_bytes());
    }

    #[test]
    fn payload_layout_frames_and_bytes() {
        let spec = XingTagSpec {
            id: XingTagId::Xing,
            flags: flag_bit::FRAMES | flag_bit::BYTES,
            frames: Some(100),
            bytes: Some(200_000),
            toc: None,
            quality: None,
        };
        let payload = build_xing_info_payload(&spec).unwrap();
        // 4 + 4 + 4 + 4 = 16 bytes; bytes field at +12, not +16.
        assert_eq!(payload.len(), 16);
        assert_eq!(&payload[8..12], &100u32.to_be_bytes());
        assert_eq!(&payload[12..16], &200_000u32.to_be_bytes());
    }

    #[test]
    fn payload_flag_field_mismatch_rejected_flag_without_field() {
        // Flag bit set but field is None → reject.
        let spec = XingTagSpec {
            id: XingTagId::Xing,
            flags: flag_bit::FRAMES,
            frames: None,
            bytes: None,
            toc: None,
            quality: None,
        };
        assert!(matches!(
            build_xing_info_payload(&spec),
            Err(XingEmitError::FlagFieldMismatch)
        ));
    }

    #[test]
    fn payload_flag_field_mismatch_rejected_field_without_flag() {
        // Field present but flag bit not set → reject.
        let spec = XingTagSpec {
            id: XingTagId::Xing,
            flags: 0,
            frames: Some(7),
            bytes: None,
            toc: None,
            quality: None,
        };
        assert!(matches!(
            build_xing_info_payload(&spec),
            Err(XingEmitError::FlagFieldMismatch)
        ));
    }

    #[test]
    fn carrier_frame_is_complete_mp3_frame_at_expected_size() {
        let hdr = make_silent_header(128, 44_100, ChannelMode::SingleChannel).unwrap();
        let want_len = hdr.frame_len().unwrap();
        let frame = build_info_frame(&hdr, &spec_all_fields()).expect("build info frame");
        assert_eq!(frame.len(), want_len);
        // Header round-trips.
        let hdr_back = parse_header(&frame[..4]).unwrap();
        assert_eq!(hdr_back.layer, hdr.layer);
        assert_eq!(hdr_back.version, hdr.version);
        assert_eq!(hdr_back.bitrate_kbps, hdr.bitrate_kbps);
        assert_eq!(hdr_back.sample_rate_hz, hdr.sample_rate_hz);
        assert!(!hdr_back.crc_protected);
    }

    #[test]
    fn carrier_frame_parse_xing_info_recovers_spec() {
        // The full round-trip: emit a carrier frame, hand its payload
        // to the symmetric parser, recover the spec field-for-field.
        // This is the wall against any drift between writer and reader.
        let hdr = make_silent_header(128, 44_100, ChannelMode::SingleChannel).unwrap();
        let frame = build_info_frame(&hdr, &spec_all_fields()).unwrap();
        let si_bytes = side_info_len(hdr.version, hdr.channel_count());
        // parse_xing_info expects the full frame payload (header
        // included), and uses side_info_bytes to skip to the magic.
        let tag = parse_xing_info(&frame, si_bytes).expect("parse_xing_info");
        let spec = spec_all_fields();
        assert_eq!(tag.id, spec.id);
        assert_eq!(tag.flags, spec.flags);
        assert_eq!(tag.frames, spec.frames);
        assert_eq!(tag.bytes, spec.bytes);
        assert_eq!(
            tag.toc.as_ref().map(|t| &t[..]),
            spec.toc.as_ref().map(|t| &t[..])
        );
        assert_eq!(tag.quality, spec.quality);
    }

    #[test]
    fn carrier_frame_roundtrip_info_magic() {
        // Same as above but with the "Info" magic.
        let hdr = make_silent_header(128, 44_100, ChannelMode::SingleChannel).unwrap();
        let mut spec = spec_all_fields();
        spec.id = XingTagId::Info;
        let frame = build_info_frame(&hdr, &spec).unwrap();
        let si_bytes = side_info_len(hdr.version, hdr.channel_count());
        let tag = parse_xing_info(&frame, si_bytes).expect("parse_xing_info");
        assert_eq!(tag.id, XingTagId::Info);
        assert_eq!(tag.frames, Some(32));
    }

    #[test]
    fn carrier_frame_roundtrip_frames_only_spec() {
        // Sparse spec: only `frames` set. parse_xing_info must report
        // the same: frames=Some(_), bytes=toc=quality=None.
        let hdr = make_silent_header(128, 44_100, ChannelMode::SingleChannel).unwrap();
        let spec = XingTagSpec {
            id: XingTagId::Xing,
            flags: flag_bit::FRAMES,
            frames: Some(99),
            bytes: None,
            toc: None,
            quality: None,
        };
        let frame = build_info_frame(&hdr, &spec).unwrap();
        let si_bytes = side_info_len(hdr.version, hdr.channel_count());
        let tag = parse_xing_info(&frame, si_bytes).expect("parse_xing_info");
        assert_eq!(tag.flags, flag_bit::FRAMES);
        assert_eq!(tag.frames, Some(99));
        assert!(tag.bytes.is_none());
        assert!(tag.toc.is_none());
        assert!(tag.quality.is_none());
    }

    #[test]
    fn carrier_frame_silent_main_data_decodes_to_zero_audio() {
        // The carrier frame's side-info has every part2_3_length=0 and
        // big_values=0, so beyond the Xing payload bytes the slot is
        // simply zero-filled; the decoder's Huffman stage reads 0 bits
        // per granule-channel and emits all-zero spectral lines. End-
        // to-end this is the silent-frame guarantee
        // `encode_silent_frame` already documents — the Xing patch
        // doesn't disturb it because the decoder skips the entire
        // main-data region (zero part2_3_length).
        let hdr = make_silent_header(128, 44_100, ChannelMode::SingleChannel).unwrap();
        let frame = build_info_frame(&hdr, &spec_all_fields()).unwrap();
        // Sanity: side-info reconstructs to all-zero granules.
        let si_bytes = side_info_len(hdr.version, hdr.channel_count());
        let si = crate::side_info::parse_side_info(&hdr, &frame[4..4 + si_bytes]).unwrap();
        for gr in 0..si.granule_count as usize {
            for ch in 0..si.channels as usize {
                assert_eq!(si.granules[gr][ch].part2_3_length, 0);
                assert_eq!(si.granules[gr][ch].big_values, 0);
            }
        }
    }

    #[test]
    fn payload_too_large_at_8kbps_mono_8khz_short_slot() {
        // The smallest MPEG-1 frame is 32 kbit/s mono 32 kHz at
        // 144 * 32_000 / 32_000 = 144 bytes; slot = 144 - 4 - 17 = 123.
        // The maximum Xing payload (120 B) fits inside that. So construct
        // a contrived case: build a fake "header" whose frame_len would
        // be just below 4 + 17 + 120 = 141. The smallest MPEG-1 mono
        // ladder bitrate is 32 kbit/s @ 48 kHz = 96 bytes — way too
        // small. Use that.
        let hdr = make_silent_header(32, 48_000, ChannelMode::SingleChannel).unwrap();
        let want_len = hdr.frame_len().unwrap();
        // 144 * 32_000 / 48_000 = 96; slot = 96 - 4 - 17 = 75 < 120.
        assert!(want_len < 4 + 17 + MAX_PAYLOAD_BYTES);
        let res = build_info_frame(&hdr, &spec_all_fields());
        assert!(matches!(res, Err(XingEmitError::PayloadTooLarge)));
    }

    #[test]
    fn payload_fits_smallest_fully_flagged_slot() {
        // Sanity check the other direction: a typical
        // 128 kbit/s mono / 44.1 kHz frame easily holds the max-size
        // payload. 144 * 128_000 / 44_100 = 417; slot = 417 - 4 - 17 = 396.
        let hdr = make_silent_header(128, 44_100, ChannelMode::SingleChannel).unwrap();
        let frame = build_info_frame(&hdr, &spec_all_fields()).unwrap();
        // Trailing slot bytes after the 120-B payload should all be zero
        // (silent main-data; the Xing patch does not modify the tail).
        let si_bytes = side_info_len(hdr.version, hdr.channel_count());
        let slot_start = 4 + si_bytes;
        let payload_end = slot_start + MAX_PAYLOAD_BYTES;
        for &b in &frame[payload_end..] {
            assert_eq!(b, 0, "non-zero byte in silent tail");
        }
    }
}
