//! MPEG audio frame framing: header parsing, per-frame byte-length
//! computation, and a self-delimiting frame walker with mid-stream
//! resynchronisation.
//!
//! This module implements *only* the framing layer described in
//! ISO/IEC 11172-3:1993 §2.4.1.3 / §2.4.2.3 (the four-byte header
//! common to all layers) together with the lower-sampling-frequency
//! redefinitions of ISO/IEC 13818-3:1997 §2.4.2.3. It does **not**
//! decode any audio: there is no side-info parser, no Huffman stage,
//! no requantisation, no IMDCT, and no synthesis filterbank here.
//!
//! All numeric constants in this file (the syncword value, the
//! bitrate ladders, the sampling-frequency tables, the samples-per-
//! frame counts, and the slot-length formula) are transcribed from
//! the two ISO/IEC specifications named above and from no other
//! source.
//!
//! # Header layout (ISO/IEC 11172-3 §2.4.1.3)
//!
//! The header is 32 bits, big-endian on the wire:
//!
//! | field              | bits |
//! | ------------------ | ---- |
//! | syncword           |  12  |
//! | ID                 |   1  |
//! | layer              |   2  |
//! | protection_bit     |   1  |
//! | bitrate_index      |   4  |
//! | sampling_frequency |   2  |
//! | padding_bit        |   1  |
//! | private_bit        |   1  |
//! | mode               |   2  |
//! | mode_extension     |   2  |
//! | copyright          |   1  |
//! | original/copy      |   1  |
//! | emphasis           |   2  |

/// MPEG audio version, distinguished by the header `ID` bit.
///
/// Per ISO/IEC 11172-3 §2.4.2.3 the `ID` bit equals `'1'` for MPEG-1
/// audio. Per ISO/IEC 13818-3 §2.4.2.3 the same bit equals `'0'` for
/// the extension to lower sampling frequencies (MPEG-2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MpegVersion {
    /// `ID == 1`: ISO/IEC 11172-3 (MPEG-1) sampling frequencies
    /// (32 / 44.1 / 48 kHz).
    Mpeg1,
    /// `ID == 0`: ISO/IEC 13818-3 (MPEG-2) lower sampling
    /// frequencies (16 / 22.05 / 24 kHz).
    Mpeg2,
}

/// MPEG audio layer, from the 2-bit `layer` field.
///
/// Encoding per ISO/IEC 11172-3 §2.4.2.3: `'11'` = Layer I,
/// `'10'` = Layer II, `'01'` = Layer III, `'00'` = reserved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Layer {
    /// `layer == '11'`.
    LayerI,
    /// `layer == '10'`.
    LayerII,
    /// `layer == '01'`.
    LayerIII,
}

/// Channel mode, from the 2-bit `mode` field (ISO/IEC 11172-3
/// §2.4.2.3): `'00'` = stereo, `'01'` = joint_stereo,
/// `'10'` = dual_channel, `'11'` = single_channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChannelMode {
    /// `mode == '00'`.
    Stereo,
    /// `mode == '01'` (intensity_stereo and/or ms_stereo).
    JointStereo,
    /// `mode == '10'`.
    DualChannel,
    /// `mode == '11'`.
    SingleChannel,
}

impl ChannelMode {
    /// Number of audio channels carried by this mode.
    ///
    /// Single-channel mode carries one channel; every other mode
    /// carries two. This matches the `nch` parameter of ISO/IEC
    /// 11172-3 §2.4.2.7 (1 for mono, 2 for all other modes).
    #[must_use]
    pub fn channel_count(self) -> u8 {
        match self {
            ChannelMode::SingleChannel => 1,
            _ => 2,
        }
    }
}

/// The 2-bit `mode_extension` field, meaningful only in
/// joint_stereo mode.
///
/// For Layer III the bits select which joint-stereo coding methods
/// are applied (ISO/IEC 11172-3 §2.4.2.3): bit pattern `i` `s`,
/// where `i` enables intensity_stereo and `s` enables ms_stereo.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModeExtension {
    /// Layer III: intensity_stereo on (`'01'` or `'11'`).
    /// Layer I/II: indicates the intensity-stereo subband bound
    /// instead; see [`ModeExtension::raw`].
    pub intensity_stereo: bool,
    /// Layer III: ms_stereo on (`'10'` or `'11'`).
    pub ms_stereo: bool,
    /// The raw 2-bit field value, preserved verbatim so Layer I/II
    /// callers can recover the intensity-stereo `bound` (`'00'`..`'11'`
    /// map to bound 4 / 8 / 12 / 16 per ISO/IEC 11172-3 §2.4.2.3).
    pub raw: u8,
}

/// De-emphasis selection, from the 2-bit `emphasis` field
/// (ISO/IEC 11172-3 §2.4.2.3): `'00'` = none, `'01'` = 50/15 µs,
/// `'10'` = reserved, `'11'` = CCITT J.17.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Emphasis {
    /// `emphasis == '00'`.
    None,
    /// `emphasis == '01'`.
    FiftyFifteenMicroseconds,
    /// `emphasis == '10'` (reserved by the standard).
    Reserved,
    /// `emphasis == '11'`.
    CcittJ17,
}

/// A parsed MPEG audio frame header.
///
/// Constructed by [`parse_header`] from the four header bytes.
/// Carries both the raw field values and the values they decode to
/// via the standard's tables (bitrate in kbit/s, sample rate in Hz,
/// frame length in bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Mp3FrameHeader {
    /// MPEG version, from the `ID` bit.
    pub version: MpegVersion,
    /// Audio layer, from the `layer` field.
    pub layer: Layer,
    /// `protection_bit`: `true` means a 16-bit CRC follows the
    /// header (ISO/IEC 11172-3 §2.4.2.3 defines `'0'` = redundancy
    /// added; this flag reports "CRC present").
    pub crc_protected: bool,
    /// Raw 4-bit `bitrate_index` (1..=14; 0 = free format).
    pub bitrate_index: u8,
    /// Decoded bitrate in kbit/s, or `None` for free format
    /// (`bitrate_index == 0`).
    pub bitrate_kbps: Option<u32>,
    /// Raw 2-bit `sampling_frequency` field.
    pub sampling_frequency_index: u8,
    /// Decoded sampling frequency in Hz.
    pub sample_rate_hz: u32,
    /// `padding_bit`: `true` adds one extra slot to the frame
    /// (ISO/IEC 11172-3 §2.4.2.3).
    pub padding: bool,
    /// `private_bit`, preserved verbatim.
    pub private_bit: bool,
    /// Channel mode, from the `mode` field.
    pub mode: ChannelMode,
    /// `mode_extension` field (meaningful in joint_stereo mode).
    pub mode_extension: ModeExtension,
    /// `copyright` bit: `true` if the stream is copyright protected.
    pub copyright: bool,
    /// `original/copy` bit: `true` if the stream is an original.
    pub original: bool,
    /// De-emphasis selection.
    pub emphasis: Emphasis,
}

impl Mp3FrameHeader {
    /// Number of audio channels (1 for single_channel, else 2).
    #[must_use]
    pub fn channel_count(&self) -> u8 {
        self.mode.channel_count()
    }

    /// Number of PCM samples per channel produced by one frame.
    ///
    /// ISO/IEC 11172-3 §2.4.2.1: Layer I frames carry 384 samples,
    /// Layer II and Layer III frames carry 1152. ISO/IEC 13818-3
    /// §2.4.2.1 redefines a Layer III lower-sampling-rate frame to
    /// carry only 576 samples.
    #[must_use]
    pub fn samples_per_frame(&self) -> u32 {
        match (self.layer, self.version) {
            (Layer::LayerI, _) => 384,
            (Layer::LayerII, _) => 1152,
            (Layer::LayerIII, MpegVersion::Mpeg1) => 1152,
            (Layer::LayerIII, MpegVersion::Mpeg2) => 576,
        }
    }

    /// Total frame length in bytes, including the header, the
    /// optional CRC, the audio data, and the padding slot if
    /// `padding` is set. Returns `None` for free-format frames
    /// (`bitrate_index == 0`), whose length is not derivable from
    /// the header alone.
    ///
    /// Derivation. A "slot" is the framing granularity from
    /// ISO/IEC 11172-3 §2.4.2.1: four bytes in Layer I, one byte in
    /// Layers II and III. The §2.4.2.3 padding method gives the
    /// per-frame slot remainder as
    /// `(12 * bitrate) % sampling_frequency` for Layer I and
    /// `(frame_size/8 * bitrate) % sampling_frequency` otherwise,
    /// where `frame_size` is 384 (Layer I) or 1152 (Layer II/III)
    /// samples. The unpadded slot count is therefore the integer
    /// quotient `frame_size/8 * bitrate / sampling_frequency`,
    /// scaled by the per-slot byte width:
    ///
    /// * Layer I:   `(12  * bitrate / sampling_frequency + pad) * 4` bytes
    /// * Layer II:  `144 * bitrate / sampling_frequency + pad`       bytes
    /// * Layer III (MPEG-1, 1152 samples):
    ///   `144 * bitrate / sampling_frequency + pad`                  bytes
    /// * Layer III (MPEG-2, 576 samples):
    ///   `72  * bitrate / sampling_frequency + pad`                  bytes
    ///
    /// Here `bitrate` is in kbit/s and `sampling_frequency` in kHz;
    /// the factor-of-1000 cancels (e.g. Layer III MPEG-1 reduces to
    /// `1152 * bitrate / (8 * sampling_frequency)` = `144 *
    /// bitrate / sampling_frequency`).
    #[must_use]
    pub fn frame_len(&self) -> Option<usize> {
        let bitrate = self.bitrate_kbps?; // None => free format
        let sr = self.sample_rate_hz;
        // Work in Hz / bit-per-second to keep the integer division
        // exact and avoid kHz/kbps unit confusion.
        let bitrate_bps = bitrate * 1000;
        let pad = u32::from(self.padding);
        let len = match self.layer {
            Layer::LayerI => {
                // 12 slots of 4 bytes; quotient counts 4-byte slots.
                let slots = (12 * bitrate_bps) / sr + pad;
                slots * 4
            }
            Layer::LayerII => (144 * bitrate_bps) / sr + pad,
            Layer::LayerIII => {
                let coeff = match self.version {
                    MpegVersion::Mpeg1 => 144, // 1152 samples / 8
                    MpegVersion::Mpeg2 => 72,  // 576 samples / 8
                };
                (coeff * bitrate_bps) / sr + pad
            }
        };
        Some(len as usize)
    }

    /// `true` when the header advertises free format
    /// (`bitrate_index == 0`), for which [`frame_len`] cannot
    /// compute a length.
    ///
    /// [`frame_len`]: Mp3FrameHeader::frame_len
    #[must_use]
    pub fn is_free_format(&self) -> bool {
        self.bitrate_index == 0
    }
}

/// Error returned when four candidate bytes do not form a valid
/// MPEG audio frame header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeaderError {
    /// Fewer than four bytes were available.
    TooShort,
    /// The leading 12 bits were not the `'1111 1111 1111'` syncword
    /// (ISO/IEC 11172-3 §2.4.2.3).
    BadSync,
    /// The `layer` field held the reserved value `'00'`.
    ReservedLayer,
    /// The `bitrate_index` field held the forbidden value `'1111'`.
    ForbiddenBitrate,
    /// The `sampling_frequency` field held the reserved value `'11'`.
    ReservedSampleRate,
}

impl core::fmt::Display for HeaderError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let msg = match self {
            HeaderError::TooShort => "header too short (need 4 bytes)",
            HeaderError::BadSync => "missing 12-bit frame syncword",
            HeaderError::ReservedLayer => "reserved layer value '00'",
            HeaderError::ForbiddenBitrate => "forbidden bitrate_index '1111'",
            HeaderError::ReservedSampleRate => "reserved sampling_frequency '11'",
        };
        f.write_str(msg)
    }
}

impl std::error::Error for HeaderError {}

// --- Bitrate ladders (kbit/s) -------------------------------------
//
// Index 0 is free format and index 15 is forbidden; both are handled
// before indexing, so these arrays hold the 14 listed rates at
// positions 1..=14 with sentinel 0 at the ends.

/// MPEG-1 Layer I bitrate ladder (ISO/IEC 11172-3 §2.4.2.3).
const BITRATE_V1_L1: [u32; 16] = [
    0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 0,
];
/// MPEG-1 Layer II bitrate ladder (ISO/IEC 11172-3 §2.4.2.3).
const BITRATE_V1_L2: [u32; 16] = [
    0, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 0,
];
/// MPEG-1 Layer III bitrate ladder (ISO/IEC 11172-3 §2.4.2.3).
const BITRATE_V1_L3: [u32; 16] = [
    0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0,
];
/// MPEG-2 (LSF) Layer I bitrate ladder (ISO/IEC 13818-3 §2.4.2.3).
const BITRATE_V2_L1: [u32; 16] = [
    0, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256, 0,
];
/// MPEG-2 (LSF) Layer II and Layer III bitrate ladder, shared
/// (ISO/IEC 13818-3 §2.4.2.3 — "same for Layer II and III").
const BITRATE_V2_L23: [u32; 16] = [
    0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0,
];

// --- Sampling-frequency tables (Hz) -------------------------------

/// MPEG-1 sampling frequencies (ISO/IEC 11172-3 §2.4.2.3):
/// `'00'` = 44.1, `'01'` = 48, `'10'` = 32 kHz; `'11'` reserved.
const SAMPLE_RATE_V1: [u32; 3] = [44_100, 48_000, 32_000];
/// MPEG-2 LSF sampling frequencies (ISO/IEC 13818-3 §2.4.2.3):
/// `'00'` = 22.05, `'01'` = 24, `'10'` = 16 kHz; `'11'` reserved.
const SAMPLE_RATE_V2: [u32; 3] = [22_050, 24_000, 16_000];

/// Parse the four bytes of an MPEG audio frame header.
///
/// Validates the syncword, rejects the reserved/forbidden field
/// values (`layer == '00'`, `bitrate_index == '1111'`,
/// `sampling_frequency == '11'`), and decodes every field per
/// ISO/IEC 11172-3 §2.4.2.3 (MPEG-1) and ISO/IEC 13818-3 §2.4.2.3
/// (MPEG-2 lower sampling frequencies).
///
/// Free-format frames (`bitrate_index == 0`) are accepted; their
/// [`Mp3FrameHeader::bitrate_kbps`] is `None`.
///
/// # Errors
///
/// Returns [`HeaderError`] if the slice is shorter than four bytes
/// or any field holds a reserved/forbidden value.
pub fn parse_header(bytes: &[u8]) -> Result<Mp3FrameHeader, HeaderError> {
    if bytes.len() < 4 {
        return Err(HeaderError::TooShort);
    }
    let raw = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);

    // syncword: top 12 bits must be all ones (ISO/IEC 11172-3 §2.4.2.3).
    if (raw >> 20) != 0xFFF {
        return Err(HeaderError::BadSync);
    }

    // ID: 1 bit. '1' = MPEG-1 (11172-3), '0' = MPEG-2 LSF (13818-3).
    let version = if (raw >> 19) & 1 == 1 {
        MpegVersion::Mpeg1
    } else {
        MpegVersion::Mpeg2
    };

    // layer: 2 bits, '11'=I '10'=II '01'=III '00'=reserved.
    let layer = match (raw >> 17) & 0b11 {
        0b11 => Layer::LayerI,
        0b10 => Layer::LayerII,
        0b01 => Layer::LayerIII,
        _ => return Err(HeaderError::ReservedLayer),
    };

    // protection_bit: '0' means CRC present (redundancy added).
    let crc_protected = (raw >> 16) & 1 == 0;

    // bitrate_index: 4 bits.
    let bitrate_index = ((raw >> 12) & 0b1111) as u8;
    if bitrate_index == 0b1111 {
        return Err(HeaderError::ForbiddenBitrate);
    }
    let bitrate_kbps = if bitrate_index == 0 {
        None // free format
    } else {
        let ladder = match (version, layer) {
            (MpegVersion::Mpeg1, Layer::LayerI) => &BITRATE_V1_L1,
            (MpegVersion::Mpeg1, Layer::LayerII) => &BITRATE_V1_L2,
            (MpegVersion::Mpeg1, Layer::LayerIII) => &BITRATE_V1_L3,
            (MpegVersion::Mpeg2, Layer::LayerI) => &BITRATE_V2_L1,
            (MpegVersion::Mpeg2, Layer::LayerII | Layer::LayerIII) => &BITRATE_V2_L23,
        };
        Some(ladder[bitrate_index as usize])
    };

    // sampling_frequency: 2 bits, '11' reserved.
    let sampling_frequency_index = ((raw >> 10) & 0b11) as u8;
    if sampling_frequency_index == 0b11 {
        return Err(HeaderError::ReservedSampleRate);
    }
    let sample_rate_hz = match version {
        MpegVersion::Mpeg1 => SAMPLE_RATE_V1[sampling_frequency_index as usize],
        MpegVersion::Mpeg2 => SAMPLE_RATE_V2[sampling_frequency_index as usize],
    };

    let padding = (raw >> 9) & 1 == 1;
    let private_bit = (raw >> 8) & 1 == 1;

    // mode: 2 bits.
    let mode = match (raw >> 6) & 0b11 {
        0b00 => ChannelMode::Stereo,
        0b01 => ChannelMode::JointStereo,
        0b10 => ChannelMode::DualChannel,
        _ => ChannelMode::SingleChannel,
    };

    // mode_extension: 2 bits.
    let me_raw = ((raw >> 4) & 0b11) as u8;
    let mode_extension = ModeExtension {
        // Layer III: low bit = intensity_stereo, high bit = ms_stereo.
        intensity_stereo: me_raw & 0b01 != 0,
        ms_stereo: me_raw & 0b10 != 0,
        raw: me_raw,
    };

    let copyright = (raw >> 3) & 1 == 1;
    let original = (raw >> 2) & 1 == 1;

    let emphasis = match raw & 0b11 {
        0b00 => Emphasis::None,
        0b01 => Emphasis::FiftyFifteenMicroseconds,
        0b10 => Emphasis::Reserved,
        _ => Emphasis::CcittJ17,
    };

    Ok(Mp3FrameHeader {
        version,
        layer,
        crc_protected,
        bitrate_index,
        bitrate_kbps,
        sampling_frequency_index,
        sample_rate_hz,
        padding,
        private_bit,
        mode,
        mode_extension,
        copyright,
        original,
        emphasis,
    })
}

/// One frame located by [`FrameWalker`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Frame<'a> {
    /// The parsed header for this frame.
    pub header: Mp3FrameHeader,
    /// Offset of the frame's first byte (the syncword) within the
    /// buffer the walker was constructed over.
    pub offset: usize,
    /// The frame's bytes, from the syncword through the last byte of
    /// audio data (length == [`Mp3FrameHeader::frame_len`]).
    pub data: &'a [u8],
}

/// Iterates self-delimiting MPEG audio frames over a byte buffer.
///
/// The walker scans for the 12-bit syncword, parses a candidate
/// header, computes the frame length from the header
/// ([`Mp3FrameHeader::frame_len`]), and yields a [`Frame`] spanning
/// exactly that many bytes. On a bad/invalid sync — or when a
/// candidate frame would run past the end of the buffer — it
/// advances one byte and resynchronises, so a corrupt or truncated
/// region in the middle of a stream does not abort the walk.
///
/// Free-format frames (no length derivable from the header) and
/// frames whose computed length exceeds the remaining buffer are
/// skipped via the same one-byte resync, since the framing layer
/// alone cannot bound them.
#[derive(Debug, Clone)]
pub struct FrameWalker<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> FrameWalker<'a> {
    /// Create a walker over `buf`, starting at offset 0.
    #[must_use]
    pub fn new(buf: &'a [u8]) -> Self {
        FrameWalker { buf, pos: 0 }
    }

    /// Current scan offset within the buffer.
    #[must_use]
    pub fn position(&self) -> usize {
        self.pos
    }
}

impl<'a> Iterator for FrameWalker<'a> {
    type Item = Frame<'a>;

    fn next(&mut self) -> Option<Frame<'a>> {
        // Need at least a full 4-byte header to test a sync point.
        while self.pos + 4 <= self.buf.len() {
            // Cheap pre-filter: first byte and top nibble of the
            // second byte must be the start of the 12-bit syncword.
            if self.buf[self.pos] != 0xFF || (self.buf[self.pos + 1] & 0xE0) != 0xE0 {
                self.pos += 1;
                continue;
            }
            match parse_header(&self.buf[self.pos..]) {
                Ok(header) => match header.frame_len() {
                    Some(len) if len >= 4 && self.pos + len <= self.buf.len() => {
                        let frame = Frame {
                            header,
                            offset: self.pos,
                            data: &self.buf[self.pos..self.pos + len],
                        };
                        self.pos += len;
                        return Some(frame);
                    }
                    // Free format, degenerate length, or a frame that
                    // would overrun the buffer: resync one byte on.
                    _ => {
                        self.pos += 1;
                    }
                },
                Err(_) => {
                    self.pos += 1;
                }
            }
        }
        // Park at the end so position() reflects a fully consumed scan.
        self.pos = self.buf.len();
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Assemble a 4-byte header from the individual fields. Bit
    /// positions mirror ISO/IEC 11172-3 §2.4.1.3 exactly so the
    /// tests construct their inputs straight from the spec layout
    /// rather than from any pre-baked byte pattern.
    #[allow(clippy::too_many_arguments)]
    fn build_header(
        id: u32,
        layer: u32,
        protection: u32,
        bitrate_index: u32,
        sampling_frequency: u32,
        padding: u32,
        private: u32,
        mode: u32,
        mode_ext: u32,
        copyright: u32,
        original: u32,
        emphasis: u32,
    ) -> [u8; 4] {
        let raw: u32 = (0xFFF << 20)
            | (id << 19)
            | (layer << 17)
            | (protection << 16)
            | (bitrate_index << 12)
            | (sampling_frequency << 10)
            | (padding << 9)
            | (private << 8)
            | (mode << 6)
            | (mode_ext << 4)
            | (copyright << 3)
            | (original << 2)
            | emphasis;
        raw.to_be_bytes()
    }

    #[test]
    fn parses_mpeg1_layer3_128k_44100_stereo() {
        // ID=1 (MPEG-1), layer='01' (III), protection='1' (no CRC),
        // bitrate_index='1001' (128 kbps for V1 L3),
        // sampling_frequency='00' (44.1 kHz), no padding, stereo.
        let h = build_header(1, 0b01, 1, 0b1001, 0b00, 0, 0, 0b00, 0, 0, 1, 0);
        let hdr = parse_header(&h).expect("valid header");
        assert_eq!(hdr.version, MpegVersion::Mpeg1);
        assert_eq!(hdr.layer, Layer::LayerIII);
        assert!(!hdr.crc_protected);
        assert_eq!(hdr.bitrate_kbps, Some(128));
        assert_eq!(hdr.sample_rate_hz, 44_100);
        assert_eq!(hdr.mode, ChannelMode::Stereo);
        assert_eq!(hdr.channel_count(), 2);
        assert_eq!(hdr.samples_per_frame(), 1152);
        // 144 * 128000 / 44100 = 417 (no padding).
        assert_eq!(hdr.frame_len(), Some(417));
    }

    #[test]
    fn padding_adds_one_byte_for_layer3() {
        // Same as above but padding_bit set: 417 + 1 = 418.
        let h = build_header(1, 0b01, 1, 0b1001, 0b00, 1, 0, 0b00, 0, 0, 1, 0);
        let hdr = parse_header(&h).unwrap();
        assert!(hdr.padding);
        assert_eq!(hdr.frame_len(), Some(418));
    }

    #[test]
    fn frame_len_320k_48000() {
        // bitrate_index='1110' = 320 kbps (V1 L3), 48 kHz ('01').
        // 144 * 320000 / 48000 = 960.
        let h = build_header(1, 0b01, 1, 0b1110, 0b01, 0, 0, 0b00, 0, 0, 1, 0);
        let hdr = parse_header(&h).unwrap();
        assert_eq!(hdr.bitrate_kbps, Some(320));
        assert_eq!(hdr.sample_rate_hz, 48_000);
        assert_eq!(hdr.frame_len(), Some(960));
    }

    #[test]
    fn frame_len_32k_32000() {
        // bitrate_index='0001' = 32 kbps (V1 L3), 32 kHz ('10').
        // 144 * 32000 / 32000 = 144.
        let h = build_header(1, 0b01, 1, 0b0001, 0b10, 0, 0, 0b11, 0, 0, 1, 0);
        let hdr = parse_header(&h).unwrap();
        assert_eq!(hdr.bitrate_kbps, Some(32));
        assert_eq!(hdr.sample_rate_hz, 32_000);
        assert_eq!(hdr.frame_len(), Some(144));
    }

    #[test]
    fn parses_mpeg2_lsf_layer3() {
        // ID=0 (MPEG-2 LSF), layer III, bitrate_index='1000'
        // => 64 kbps (V2 L2/L3 shared ladder),
        // sampling_frequency='00' => 22.05 kHz.
        let h = build_header(0, 0b01, 1, 0b1000, 0b00, 0, 0, 0b00, 0, 0, 1, 0);
        let hdr = parse_header(&h).unwrap();
        assert_eq!(hdr.version, MpegVersion::Mpeg2);
        assert_eq!(hdr.layer, Layer::LayerIII);
        assert_eq!(hdr.bitrate_kbps, Some(64));
        assert_eq!(hdr.sample_rate_hz, 22_050);
        // MPEG-2 Layer III frame: 576 samples.
        assert_eq!(hdr.samples_per_frame(), 576);
        // 72 * 64000 / 22050 = 208 (integer division).
        assert_eq!(hdr.frame_len(), Some(208));
    }

    #[test]
    fn mpeg2_lsf_sample_rate_table() {
        // '00'=22.05, '01'=24, '10'=16 kHz for ID=0.
        for (idx, expect) in [(0b00, 22_050), (0b01, 24_000), (0b10, 16_000)] {
            let h = build_header(0, 0b01, 1, 0b1000, idx, 0, 0, 0b11, 0, 0, 1, 0);
            assert_eq!(parse_header(&h).unwrap().sample_rate_hz, expect);
        }
    }

    #[test]
    fn mpeg1_sample_rate_table() {
        // '00'=44.1, '01'=48, '10'=32 kHz for ID=1.
        for (idx, expect) in [(0b00, 44_100), (0b01, 48_000), (0b10, 32_000)] {
            let h = build_header(1, 0b01, 1, 0b1001, idx, 0, 0, 0b11, 0, 0, 1, 0);
            assert_eq!(parse_header(&h).unwrap().sample_rate_hz, expect);
        }
    }

    #[test]
    fn layer1_and_layer2_frame_lengths() {
        // Layer I, MPEG-1, bitrate_index='0100' => 128 kbps (V1 L1),
        // 44.1 kHz: (12 * 128000 / 44100) * 4 = 34 * 4 = 136.
        let l1 = build_header(1, 0b11, 1, 0b0100, 0b00, 0, 0, 0b00, 0, 0, 1, 0);
        let h1 = parse_header(&l1).unwrap();
        assert_eq!(h1.layer, Layer::LayerI);
        assert_eq!(h1.samples_per_frame(), 384);
        assert_eq!(h1.frame_len(), Some(136));

        // Layer II, MPEG-1, bitrate_index='1000' => 128 kbps (V1 L2),
        // 44.1 kHz: 144 * 128000 / 44100 = 417.
        let l2 = build_header(1, 0b10, 1, 0b1000, 0b00, 0, 0, 0b00, 0, 0, 1, 0);
        let h2 = parse_header(&l2).unwrap();
        assert_eq!(h2.layer, Layer::LayerII);
        assert_eq!(h2.samples_per_frame(), 1152);
        assert_eq!(h2.frame_len(), Some(417));
    }

    #[test]
    fn all_field_decoding() {
        // Exercise every enum branch in one header:
        // ID=1, Layer II, protection='0' (CRC present),
        // bitrate_index 5, sampling_frequency '01' (48k),
        // padding=1, private=1, mode='01' (joint),
        // mode_ext='11', copyright=1, original=0, emphasis='11'.
        let h = build_header(1, 0b10, 0, 0b0101, 0b01, 1, 1, 0b01, 0b11, 1, 0, 0b11);
        let hdr = parse_header(&h).unwrap();
        assert!(hdr.crc_protected);
        assert!(hdr.padding);
        assert!(hdr.private_bit);
        assert_eq!(hdr.mode, ChannelMode::JointStereo);
        assert!(hdr.mode_extension.intensity_stereo);
        assert!(hdr.mode_extension.ms_stereo);
        assert_eq!(hdr.mode_extension.raw, 0b11);
        assert!(hdr.copyright);
        assert!(!hdr.original);
        assert_eq!(hdr.emphasis, Emphasis::CcittJ17);
    }

    #[test]
    fn mode_decoding_all_four() {
        for (bits, expect) in [
            (0b00, ChannelMode::Stereo),
            (0b01, ChannelMode::JointStereo),
            (0b10, ChannelMode::DualChannel),
            (0b11, ChannelMode::SingleChannel),
        ] {
            let h = build_header(1, 0b01, 1, 0b1001, 0b00, 0, 0, bits, 0, 0, 1, 0);
            assert_eq!(parse_header(&h).unwrap().mode, expect);
        }
        // single_channel -> 1 channel, all others -> 2.
        let mono = build_header(1, 0b01, 1, 0b1001, 0b00, 0, 0, 0b11, 0, 0, 1, 0);
        assert_eq!(parse_header(&mono).unwrap().channel_count(), 1);
    }

    #[test]
    fn layer_decoding_all_three() {
        for (bits, expect) in [
            (0b11, Layer::LayerI),
            (0b10, Layer::LayerII),
            (0b01, Layer::LayerIII),
        ] {
            // Use Layer I bitrate index valid in all ladders.
            let h = build_header(1, bits, 1, 0b0100, 0b00, 0, 0, 0b00, 0, 0, 1, 0);
            assert_eq!(parse_header(&h).unwrap().layer, expect);
        }
    }

    #[test]
    fn free_format_has_no_length() {
        // bitrate_index = 0 => free format.
        let h = build_header(1, 0b01, 1, 0b0000, 0b00, 0, 0, 0b00, 0, 0, 1, 0);
        let hdr = parse_header(&h).unwrap();
        assert!(hdr.is_free_format());
        assert_eq!(hdr.bitrate_kbps, None);
        assert_eq!(hdr.frame_len(), None);
    }

    #[test]
    fn rejects_too_short() {
        assert_eq!(
            parse_header(&[0xFF, 0xFB, 0x90]),
            Err(HeaderError::TooShort)
        );
    }

    #[test]
    fn rejects_bad_sync() {
        // Top 12 bits not all ones.
        assert_eq!(
            parse_header(&[0xFF, 0x1B, 0x90, 0x00]),
            Err(HeaderError::BadSync)
        );
    }

    #[test]
    fn rejects_reserved_layer() {
        let h = build_header(1, 0b00, 1, 0b1001, 0b00, 0, 0, 0b00, 0, 0, 1, 0);
        assert_eq!(parse_header(&h), Err(HeaderError::ReservedLayer));
    }

    #[test]
    fn rejects_forbidden_bitrate() {
        let h = build_header(1, 0b01, 1, 0b1111, 0b00, 0, 0, 0b00, 0, 0, 1, 0);
        assert_eq!(parse_header(&h), Err(HeaderError::ForbiddenBitrate));
    }

    #[test]
    fn rejects_reserved_sample_rate() {
        let h = build_header(1, 0b01, 1, 0b1001, 0b11, 0, 0, 0b00, 0, 0, 1, 0);
        assert_eq!(parse_header(&h), Err(HeaderError::ReservedSampleRate));
    }

    #[test]
    fn walker_iterates_consecutive_frames() {
        // Two identical 417-byte MPEG-1 L3 128k/44.1k frames back to back.
        let hb = build_header(1, 0b01, 1, 0b1001, 0b00, 0, 0, 0b00, 0, 0, 1, 0);
        let len = parse_header(&hb).unwrap().frame_len().unwrap();
        assert_eq!(len, 417);
        let mut buf = Vec::new();
        for _ in 0..2 {
            buf.extend_from_slice(&hb);
            buf.extend(std::iter::repeat_n(0u8, len - 4));
        }
        let frames: Vec<_> = FrameWalker::new(&buf).collect();
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].offset, 0);
        assert_eq!(frames[0].data.len(), 417);
        assert_eq!(frames[1].offset, 417);
        assert_eq!(frames[1].data.len(), 417);
    }

    #[test]
    fn walker_resyncs_over_garbage_prefix() {
        // Junk, then a valid frame.
        let hb = build_header(1, 0b01, 1, 0b1001, 0b00, 0, 0, 0b00, 0, 0, 1, 0);
        let len = parse_header(&hb).unwrap().frame_len().unwrap();
        let mut buf = vec![0x13, 0x37, 0xFF, 0x00, 0xAB]; // 5 junk bytes
        let frame_start = buf.len();
        buf.extend_from_slice(&hb);
        buf.extend(std::iter::repeat_n(0u8, len - 4));
        let frames: Vec<_> = FrameWalker::new(&buf).collect();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].offset, frame_start);
    }

    #[test]
    fn walker_resyncs_mid_stream() {
        // Frame, garbage, frame: a corrupt region between two good
        // frames must not abort the walk.
        let hb = build_header(1, 0b01, 1, 0b1001, 0b00, 0, 0, 0b00, 0, 0, 1, 0);
        let len = parse_header(&hb).unwrap().frame_len().unwrap();
        let mut buf = Vec::new();
        buf.extend_from_slice(&hb);
        buf.extend(std::iter::repeat_n(0u8, len - 4));
        // Corrupt gap that is not a valid sync.
        buf.extend_from_slice(&[0x00, 0x11, 0x22, 0x33, 0x44, 0x55]);
        let second_start = buf.len();
        buf.extend_from_slice(&hb);
        buf.extend(std::iter::repeat_n(0u8, len - 4));
        let frames: Vec<_> = FrameWalker::new(&buf).collect();
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].offset, 0);
        assert_eq!(frames[1].offset, second_start);
    }

    #[test]
    fn walker_skips_truncated_trailing_frame() {
        // A header whose computed length overruns the buffer must be
        // resynced past, not yielded with a short slice.
        let hb = build_header(1, 0b01, 1, 0b1001, 0b00, 0, 0, 0b00, 0, 0, 1, 0);
        // Only the 4 header bytes present; frame claims 417 bytes.
        let buf = hb.to_vec();
        let frames: Vec<_> = FrameWalker::new(&buf).collect();
        assert!(frames.is_empty());
    }

    #[test]
    fn walker_empty_and_short_inputs() {
        assert!(FrameWalker::new(&[]).next().is_none());
        assert!(FrameWalker::new(&[0xFF, 0xFB]).next().is_none());
    }
}
