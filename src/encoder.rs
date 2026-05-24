//! Layer III **bitstream-formatting** encoder, Phase 1.
//!
//! This module writes the framing half of an MP3 (MPEG-1/2 Audio Layer
//! III) frame: the four-byte frame header (ISO/IEC 11172-3:1993
//! §2.4.1.3 / §2.4.2.3, with the lower-sampling-frequency redefinitions
//! of ISO/IEC 13818-3:1997 §2.4.2.3) and the Layer III side-information
//! block (ISO/IEC 11172-3 §2.4.1.7 for MPEG-1, ISO/IEC 13818-3 §2.4.1.7
//! for MPEG-2 / MPEG-2.5 LSF). Each writer is the exact byte-for-byte
//! inverse of the corresponding parser in [`crate::frame`] /
//! [`crate::side_info`] — feeding a writer's output back through the
//! parser reproduces the input struct.
//!
//! It also provides a **silent-frame** encode path
//! ([`encode_silent_frame`]): an all-zero-quantization Layer III frame
//! whose header + side-info are structurally valid and whose main data
//! is empty (every granule-channel carries `part2_3_length == 0`,
//! `big_values == 0`, so no scalefactors and no Huffman bits are coded).
//! Such a frame is a complete, decodable MP3 frame that reconstructs to
//! silence — it round-trips through this crate's own [`crate::frame`] +
//! [`crate::side_info`] parsers and is accepted by black-box external
//! decoders (e.g. `ffmpeg`, `mpg123`).
//!
//! # What this module does **not** do (Phase 1 scope)
//!
//! There is no psychoacoustic model, no MDCT analysis filterbank, no bit
//! allocation, no scalefactor estimation, and no Huffman *encoding* of
//! non-zero spectral lines. Those require the forward (analysis) signal
//! path and are a later round. Phase 1 is purely the bitstream layout:
//! valid framing + side-info field packing + a zero-data frame.
//!
//! All numeric constants and field orders are transcribed from the two
//! ISO/IEC specifications named above and from no other source.

use crate::frame::{ChannelMode, Emphasis, Layer, ModeExtension, Mp3FrameHeader, MpegVersion};
use crate::side_info::{
    BlockType, GranuleChannel, SideInfo, GRANULES, GRANULES_LSF, SIDE_INFO_BYTES_LSF_MONO,
    SIDE_INFO_BYTES_LSF_STEREO, SIDE_INFO_BYTES_MONO, SIDE_INFO_BYTES_STEREO,
};

/// A most-significant-bit-first bit writer over a growing byte buffer.
///
/// The MPEG audio bitstream is written MSB-first (ISO/IEC 11172-3
/// §2.4.1: "the most significant bit … is transmitted first"). This is
/// the exact inverse of the `BitReader` used by the side-info parser:
/// bits accumulate into the current byte from bit 7 downward, and a
/// completed byte is flushed to the buffer. [`BitWriter::finish`]
/// zero-pads the trailing partial byte (if any) to a byte boundary —
/// the MPEG-1 / LSF side-info blocks are byte-aligned by construction,
/// so for those there is never a partial byte to pad.
#[derive(Debug, Default)]
struct BitWriter {
    bytes: Vec<u8>,
    /// The partially-filled current byte (bits packed from MSB down).
    cur: u8,
    /// Number of bits already written into `cur` (0..=7).
    nbits: u8,
}

impl BitWriter {
    /// Pre-size the output buffer to `cap` bytes.
    fn with_capacity(cap: usize) -> Self {
        BitWriter {
            bytes: Vec::with_capacity(cap),
            cur: 0,
            nbits: 0,
        }
    }

    /// Write the low `n` bits of `value` (0 ≤ `n` ≤ 32), MSB-first.
    fn write(&mut self, value: u32, n: u32) {
        debug_assert!(n <= 32);
        for i in (0..n).rev() {
            let bit = ((value >> i) & 1) as u8;
            self.cur = (self.cur << 1) | bit;
            self.nbits += 1;
            if self.nbits == 8 {
                self.bytes.push(self.cur);
                self.cur = 0;
                self.nbits = 0;
            }
        }
    }

    /// Write a single bit.
    fn write_bool(&mut self, b: bool) {
        self.write(u32::from(b), 1);
    }

    /// Flush the trailing partial byte (zero-padded to a byte boundary)
    /// and return the packed bytes.
    fn finish(mut self) -> Vec<u8> {
        if self.nbits > 0 {
            self.cur <<= 8 - self.nbits;
            self.bytes.push(self.cur);
            self.cur = 0;
            self.nbits = 0;
        }
        self.bytes
    }
}

/// Map a [`MpegVersion`] to its header `ID` bit (ISO/IEC 11172-3
/// §2.4.2.3: `'1'` = MPEG-1; ISO/IEC 13818-3 §2.4.2.3: `'0'` = MPEG-2
/// lower sampling frequencies).
fn id_bit(version: MpegVersion) -> u32 {
    match version {
        MpegVersion::Mpeg1 => 1,
        MpegVersion::Mpeg2 => 0,
    }
}

/// Map a [`Layer`] to its 2-bit `layer` field (ISO/IEC 11172-3
/// §2.4.2.3: `'11'` = I, `'10'` = II, `'01'` = III).
fn layer_bits(layer: Layer) -> u32 {
    match layer {
        Layer::LayerI => 0b11,
        Layer::LayerII => 0b10,
        Layer::LayerIII => 0b01,
    }
}

/// Map a [`ChannelMode`] to its 2-bit `mode` field (ISO/IEC 11172-3
/// §2.4.2.3).
fn mode_bits(mode: ChannelMode) -> u32 {
    match mode {
        ChannelMode::Stereo => 0b00,
        ChannelMode::JointStereo => 0b01,
        ChannelMode::DualChannel => 0b10,
        ChannelMode::SingleChannel => 0b11,
    }
}

/// Map an [`Emphasis`] to its 2-bit `emphasis` field (ISO/IEC 11172-3
/// §2.4.2.3).
fn emphasis_bits(emphasis: Emphasis) -> u32 {
    match emphasis {
        Emphasis::None => 0b00,
        Emphasis::FiftyFifteenMicroseconds => 0b01,
        Emphasis::Reserved => 0b10,
        Emphasis::CcittJ17 => 0b11,
    }
}

/// Write the four bytes of an MPEG audio frame header.
///
/// This is the exact inverse of [`crate::frame::parse_header`]: it packs
/// every field of `header` at the §2.4.1.3 bit position the parser reads
/// it from, so `parse_header(&write_header(&h)) == Ok(h)` for any header
/// that round-trips (i.e. whose decoded `bitrate_kbps` / `sample_rate_hz`
/// are consistent with its raw `bitrate_index` / `sampling_frequency_index`,
/// which is always the case for a header produced by the parser).
///
/// The raw index fields `bitrate_index` and `sampling_frequency_index`
/// are written verbatim — the encoder is responsible for setting them to
/// indices consistent with the desired bitrate / sample rate; the
/// decoded `bitrate_kbps` / `sample_rate_hz` fields are *not* re-derived
/// here.
///
/// The `protection_bit` is the inverse sense of
/// [`Mp3FrameHeader::crc_protected`]: the wire bit is `'0'` when a CRC is
/// present (ISO/IEC 11172-3 §2.4.2.3 "redundancy added"), so
/// `crc_protected == true` writes `protection_bit = 0`.
#[must_use]
pub fn write_header(header: &Mp3FrameHeader) -> [u8; 4] {
    let protection = u32::from(!header.crc_protected);
    let raw: u32 = (0xFFF << 20)
        | (id_bit(header.version) << 19)
        | (layer_bits(header.layer) << 17)
        | (protection << 16)
        | (u32::from(header.bitrate_index) << 12)
        | (u32::from(header.sampling_frequency_index) << 10)
        | (u32::from(header.padding) << 9)
        | (u32::from(header.private_bit) << 8)
        | (mode_bits(header.mode) << 6)
        | (u32::from(header.mode_extension.raw) << 4)
        | (u32::from(header.copyright) << 3)
        | (u32::from(header.original) << 2)
        | emphasis_bits(header.emphasis);
    raw.to_be_bytes()
}

/// Write one per-granule-per-channel side-info record (ISO/IEC 11172-3
/// §2.4.1.7 inner body; ISO/IEC 13818-3 §2.4.1.7 for LSF).
///
/// Field order and bit widths exactly mirror
/// [`crate::side_info::parse_side_info`]'s `read_granule_channel`:
/// `part2_3_length(12)`, `big_values(9)`, `global_gain(8)`,
/// `scalefac_compress(4 MPEG-1 / 9 LSF)`, `window_switching_flag(1)`,
/// then the window branch (22 bits either way), then `preflag(1, MPEG-1
/// only)`, `scalefac_scale(1)`, `count1table_select(1)`.
fn write_granule_channel(w: &mut BitWriter, gc: &GranuleChannel, lsf: bool) {
    w.write(u32::from(gc.part2_3_length), 12);
    w.write(u32::from(gc.big_values), 9);
    w.write(u32::from(gc.global_gain), 8);
    // scalefac_compress: 4 bits in MPEG-1, 9 bits in LSF.
    if lsf {
        w.write(u32::from(gc.scalefac_compress), 9);
    } else {
        w.write(u32::from(gc.scalefac_compress), 4);
    }
    w.write_bool(gc.window_switching_flag);

    if gc.window_switching_flag {
        // Window-switched branch: block_type(2), mixed_block_flag(1),
        // 2×table_select(5), 3×subblock_gain(3) = 22 bits.
        w.write(block_type_bits(gc.block_type), 2);
        w.write_bool(gc.mixed_block_flag);
        for &ts in gc.table_select.iter().take(2) {
            w.write(u32::from(ts), 5);
        }
        for &sg in &gc.subblock_gain {
            w.write(u32::from(sg), 3);
        }
        // region0_count / region1_count are NOT transmitted in this
        // branch — they carry §2.4.2.7 defaults the decoder reconstructs.
    } else {
        // Long branch: 3×table_select(5), region0_count(4),
        // region1_count(3) = 22 bits.
        for &ts in &gc.table_select {
            w.write(u32::from(ts), 5);
        }
        w.write(u32::from(gc.region0_count), 4);
        w.write(u32::from(gc.region1_count), 3);
    }

    // preflag is a transmitted bit in MPEG-1 only; LSF derives it from
    // scalefac_compress at decode time (no bit on the wire).
    if !lsf {
        w.write_bool(gc.preflag);
    }
    w.write_bool(gc.scalefac_scale);
    w.write_bool(gc.count1table_select);
}

/// Map a [`BlockType`] to its 2-bit `block_type` field (ISO/IEC 11172-3
/// §2.4.2.7): `0` long, `1` start, `2` short, `3` end.
fn block_type_bits(bt: BlockType) -> u32 {
    match bt {
        BlockType::Long => 0,
        BlockType::Start => 1,
        BlockType::Short => 2,
        BlockType::End => 3,
    }
}

/// Write the Layer III side-information block described by `si`.
///
/// Exact inverse of [`crate::side_info::parse_side_info`]: the returned
/// bytes, fed back through that parser with a matching header, reproduce
/// `si`. Dispatches on [`SideInfo::lsf`]: MPEG-1 writes the ISO/IEC
/// 11172-3 §2.4.1.7 layout (9-bit `main_data_begin`, per-channel `scfsi`,
/// two granules); MPEG-2 / MPEG-2.5 LSF writes the ISO/IEC 13818-3
/// §2.4.1.7 layout (8-bit `main_data_begin`, no `scfsi`, one granule).
///
/// The output is byte-aligned and exactly [`SideInfo::byte_len`] bytes.
#[must_use]
pub fn write_side_info(si: &SideInfo) -> Vec<u8> {
    let nch = si.channels as usize;
    let mono = si.channels == 1;
    let mut w = BitWriter::with_capacity(si.byte_len());

    if si.lsf {
        // main_data_begin (8 bits in LSF).
        w.write(u32::from(si.main_data_begin), 8);
        // private_bits: 1 bit mono, 2 bits otherwise; no scfsi in LSF.
        if mono {
            w.write(u32::from(si.private_bits), 1);
        } else {
            w.write(u32::from(si.private_bits), 2);
        }
        // One granule, nch channels.
        for ch in 0..nch {
            write_granule_channel(&mut w, &si.granules[0][ch], true);
        }
    } else {
        // main_data_begin (9 bits in MPEG-1).
        w.write(u32::from(si.main_data_begin), 9);
        // private_bits: 5 bits mono, 3 bits otherwise.
        if mono {
            w.write(u32::from(si.private_bits), 5);
        } else {
            w.write(u32::from(si.private_bits), 3);
        }
        // scfsi[ch][band]: 4 one-bit flags per channel.
        for ch in 0..nch {
            for band in 0..4 {
                w.write_bool(si.scfsi[ch][band]);
            }
        }
        // Two granules, nch channels each.
        for gr in 0..GRANULES {
            for ch in 0..nch {
                write_granule_channel(&mut w, &si.granules[gr][ch], false);
            }
        }
    }

    w.finish()
}

/// The all-zero / long-block granule-channel record used for a silent
/// frame: no main-data bits (`part2_3_length == 0`, `big_values == 0`),
/// a long window, and the §2.4.2.7 defaults the decoder expects for an
/// empty granule.
const SILENT_GRANULE_CHANNEL: GranuleChannel = GranuleChannel {
    part2_3_length: 0,
    big_values: 0,
    global_gain: 0,
    scalefac_compress: 0,
    window_switching_flag: false,
    block_type: BlockType::Long,
    mixed_block_flag: false,
    table_select: [0; 3],
    subblock_gain: [0; 3],
    region0_count: 0,
    region1_count: 0,
    preflag: false,
    scalefac_scale: false,
    count1table_select: false,
};

/// Errors returned by the Phase 1 encode path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodeError {
    /// The requested layer is not Layer III. The Phase 1 encoder writes
    /// Layer III frames only.
    NotLayer3,
    /// The header is free-format (`bitrate_index == 0`), whose frame
    /// length is not derivable from the header alone, so a self-contained
    /// silent frame cannot be sized.
    FreeFormat,
    /// The header's `bitrate_index` / `sampling_frequency_index` /
    /// channel-mode combination yields a frame too small to hold the
    /// header plus side-information block (only possible at the lowest
    /// bitrates with the widest side info).
    FrameTooSmall,
}

impl core::fmt::Display for EncodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let msg = match self {
            EncodeError::NotLayer3 => "Phase 1 encoder writes Layer III frames only",
            EncodeError::FreeFormat => "free-format header has no derivable frame length",
            EncodeError::FrameTooSmall => "frame length too small for header + side info",
        };
        f.write_str(msg)
    }
}

impl std::error::Error for EncodeError {}

/// Build a [`SideInfo`] for a silent frame matching `header`.
///
/// Every granule-channel is the all-zero long-block record (no
/// main-data bits: `part2_3_length == 0`, `big_values == 0`),
/// `main_data_begin == 0` (main data — empty — starts immediately after
/// the side info), all `scfsi` clear, and all `private_bits` zero.
/// Granule / channel counts follow the header version and mode.
#[must_use]
pub fn silent_side_info(header: &Mp3FrameHeader) -> SideInfo {
    let nch = header.channel_count();
    let lsf = header.version == MpegVersion::Mpeg2;
    let granule_count = if lsf { GRANULES_LSF } else { GRANULES } as u8;

    SideInfo {
        main_data_begin: 0,
        private_bits: 0,
        scfsi: [[false; 4]; 2],
        granules: [[SILENT_GRANULE_CHANNEL; 2]; GRANULES],
        channels: nch,
        granule_count,
        lsf,
    }
}

/// The byte length of a Layer III side-info block for `version` /
/// `channels` (ISO/IEC 11172-3 §2.4.1.7 / ISO/IEC 13818-3 §2.4.1.7).
fn side_info_bytes(version: MpegVersion, channels: u8) -> usize {
    match (version, channels == 1) {
        (MpegVersion::Mpeg1, true) => SIDE_INFO_BYTES_MONO,
        (MpegVersion::Mpeg1, false) => SIDE_INFO_BYTES_STEREO,
        (MpegVersion::Mpeg2, true) => SIDE_INFO_BYTES_LSF_MONO,
        (MpegVersion::Mpeg2, false) => SIDE_INFO_BYTES_LSF_STEREO,
    }
}

/// Encode one **silent** Layer III frame for the given `header`.
///
/// Produces a complete, self-delimiting MP3 frame:
///
/// 1. the four header bytes ([`write_header`]),
/// 2. the side-information block ([`write_side_info`] over
///    [`silent_side_info`]) — *with no CRC*: this path always emits a
///    frame with the protection bit set (no CRC), so the caller's
///    `header.crc_protected` is forced `false` in the written header,
/// 3. the main-data region zero-filled out to [`Mp3FrameHeader::frame_len`].
///
/// Because every granule-channel carries `part2_3_length == 0` and
/// `big_values == 0`, there are no scalefactor bits and no Huffman bits;
/// the entire main-data region is unused (all zero). A conformant
/// decoder reconstructs all-zero spectral lines → silence.
///
/// The returned `Vec<u8>` length equals `header.frame_len()`.
///
/// # Errors
///
/// * [`EncodeError::NotLayer3`] if `header.layer != Layer::LayerIII`.
/// * [`EncodeError::FreeFormat`] if `header.bitrate_index == 0`.
/// * [`EncodeError::FrameTooSmall`] if the derived frame length cannot
///   hold the 4-byte header plus the side-info block.
pub fn encode_silent_frame(header: &Mp3FrameHeader) -> Result<Vec<u8>, EncodeError> {
    if header.layer != Layer::LayerIII {
        return Err(EncodeError::NotLayer3);
    }
    if header.is_free_format() {
        return Err(EncodeError::FreeFormat);
    }
    let frame_len = header.frame_len().ok_or(EncodeError::FreeFormat)?;

    // Force the no-CRC form for the silent path: a CRC would otherwise
    // need a computed 16-bit checksum and would shift the side info.
    let mut hdr = *header;
    hdr.crc_protected = false;

    let nch = hdr.channel_count();
    let si_bytes = side_info_bytes(hdr.version, nch);

    if frame_len < 4 + si_bytes {
        return Err(EncodeError::FrameTooSmall);
    }

    let mut out = Vec::with_capacity(frame_len);
    out.extend_from_slice(&write_header(&hdr));

    let si = silent_side_info(&hdr);
    let si_bytes_written = write_side_info(&si);
    debug_assert_eq!(si_bytes_written.len(), si_bytes);
    out.extend_from_slice(&si_bytes_written);

    // Zero-fill the remaining main-data region.
    out.resize(frame_len, 0);
    Ok(out)
}

/// Build an [`Mp3FrameHeader`] for a CBR Layer III silent stream from a
/// bitrate / sample rate / channel mode.
///
/// Resolves `bitrate_kbps` and `sample_rate_hz` to their raw header
/// indices, infers the [`MpegVersion`] from the sample rate (MPEG-1
/// rates 32 / 44.1 / 48 kHz vs MPEG-2 LSF 16 / 22.05 / 24 kHz), and
/// fills the remaining header fields with the standard CBR defaults
/// (no CRC, no padding, private bit clear, mode_extension `'00'`,
/// copyright clear, original set, no emphasis).
///
/// # Errors
///
/// Returns [`EncodeError::FreeFormat`] when `bitrate_kbps` /
/// `sample_rate_hz` / `mode` do not map to a valid Layer III header
/// index (the requested bitrate is not on the layer's ladder for the
/// inferred version, or the sample rate is not a recognised MPEG-1 / LSF
/// rate). The error name is reused as the catch-all "no valid index"
/// signal for this convenience constructor.
pub fn make_silent_header(
    bitrate_kbps: u32,
    sample_rate_hz: u32,
    mode: ChannelMode,
) -> Result<Mp3FrameHeader, EncodeError> {
    let (version, sf_index) = sample_rate_index(sample_rate_hz).ok_or(EncodeError::FreeFormat)?;
    let bitrate_index =
        layer3_bitrate_index(version, bitrate_kbps).ok_or(EncodeError::FreeFormat)?;

    Ok(Mp3FrameHeader {
        version,
        layer: Layer::LayerIII,
        crc_protected: false,
        bitrate_index,
        bitrate_kbps: Some(bitrate_kbps),
        sampling_frequency_index: sf_index,
        sample_rate_hz,
        padding: false,
        private_bit: false,
        mode,
        mode_extension: ModeExtension {
            intensity_stereo: false,
            ms_stereo: false,
            raw: 0,
        },
        copyright: false,
        original: true,
        emphasis: Emphasis::None,
    })
}

/// Resolve a sample rate (Hz) to its `(version, sampling_frequency_index)`
/// per ISO/IEC 11172-3 §2.4.2.3 (MPEG-1) and ISO/IEC 13818-3 §2.4.2.3
/// (MPEG-2 LSF). Returns `None` for an unrecognised rate.
fn sample_rate_index(rate: u32) -> Option<(MpegVersion, u8)> {
    match rate {
        44_100 => Some((MpegVersion::Mpeg1, 0)),
        48_000 => Some((MpegVersion::Mpeg1, 1)),
        32_000 => Some((MpegVersion::Mpeg1, 2)),
        22_050 => Some((MpegVersion::Mpeg2, 0)),
        24_000 => Some((MpegVersion::Mpeg2, 1)),
        16_000 => Some((MpegVersion::Mpeg2, 2)),
        _ => None,
    }
}

/// Resolve a Layer III bitrate (kbit/s) to its 4-bit `bitrate_index` for
/// the given `version` per ISO/IEC 11172-3 §2.4.2.3 (MPEG-1 Layer III
/// ladder) and ISO/IEC 13818-3 §2.4.2.3 (MPEG-2 LSF Layer II/III ladder).
/// Returns `None` if the bitrate is not on the layer's ladder.
fn layer3_bitrate_index(version: MpegVersion, kbps: u32) -> Option<u8> {
    // The two ladders, indices 1..=14 (0 = free, 15 = forbidden).
    const V1_L3: [u32; 16] = [
        0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0,
    ];
    const V2_L23: [u32; 16] = [
        0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0,
    ];
    let ladder = match version {
        MpegVersion::Mpeg1 => &V1_L3,
        MpegVersion::Mpeg2 => &V2_L23,
    };
    ladder
        .iter()
        .enumerate()
        .skip(1)
        .take(14)
        .find(|&(_, &b)| b == kbps)
        .map(|(i, _)| i as u8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::parse_header;
    use crate::side_info::parse_side_info;

    #[test]
    fn header_writer_is_parse_inverse_mpeg1() {
        // MPEG-1 Layer III 128 kbps / 44.1 kHz stereo, no CRC.
        let h = make_silent_header(128, 44_100, ChannelMode::Stereo).unwrap();
        let bytes = write_header(&h);
        let parsed = parse_header(&bytes).unwrap();
        assert_eq!(parsed, h);
    }

    #[test]
    fn header_writer_is_parse_inverse_lsf() {
        // MPEG-2 LSF Layer III 64 kbps / 22.05 kHz mono.
        let h = make_silent_header(64, 22_050, ChannelMode::SingleChannel).unwrap();
        let bytes = write_header(&h);
        let parsed = parse_header(&bytes).unwrap();
        assert_eq!(parsed, h);
        assert_eq!(parsed.version, MpegVersion::Mpeg2);
    }

    #[test]
    fn header_writer_preserves_all_fields() {
        // Construct a header by hand exercising every non-default field,
        // write it, parse it back, and confirm equality. crc_protected
        // is exercised separately because the silent path forces it off.
        let h = Mp3FrameHeader {
            version: MpegVersion::Mpeg1,
            layer: Layer::LayerIII,
            crc_protected: true,
            bitrate_index: 5,
            bitrate_kbps: Some(64),
            sampling_frequency_index: 1,
            sample_rate_hz: 48_000,
            padding: true,
            private_bit: true,
            mode: ChannelMode::JointStereo,
            mode_extension: ModeExtension {
                intensity_stereo: true,
                ms_stereo: true,
                raw: 0b11,
            },
            copyright: true,
            original: false,
            emphasis: Emphasis::CcittJ17,
        };
        let parsed = parse_header(&write_header(&h)).unwrap();
        assert_eq!(parsed, h);
    }

    #[test]
    fn header_writer_protection_bit_sense() {
        // crc_protected == true must write protection_bit '0'.
        let mut h = make_silent_header(128, 44_100, ChannelMode::Stereo).unwrap();
        h.crc_protected = true;
        let bytes = write_header(&h);
        // bit 16 of the 32-bit big-endian header == bit 0 of byte 1's
        // low nibble... protection is raw>>16 & 1; with crc_protected it
        // is 0.
        let raw = u32::from_be_bytes(bytes);
        assert_eq!((raw >> 16) & 1, 0);
        h.crc_protected = false;
        let raw = u32::from_be_bytes(write_header(&h));
        assert_eq!((raw >> 16) & 1, 1);
    }

    #[test]
    fn side_info_writer_is_parse_inverse_mpeg1_stereo() {
        let h = make_silent_header(128, 44_100, ChannelMode::Stereo).unwrap();
        let si = silent_side_info(&h);
        let bytes = write_side_info(&si);
        assert_eq!(bytes.len(), SIDE_INFO_BYTES_STEREO);
        let parsed = parse_side_info(&h, &bytes).unwrap();
        assert_eq!(parsed, si);
    }

    #[test]
    fn side_info_writer_is_parse_inverse_mpeg1_mono() {
        let h = make_silent_header(128, 44_100, ChannelMode::SingleChannel).unwrap();
        let si = silent_side_info(&h);
        let bytes = write_side_info(&si);
        assert_eq!(bytes.len(), SIDE_INFO_BYTES_MONO);
        let parsed = parse_side_info(&h, &bytes).unwrap();
        assert_eq!(parsed, si);
    }

    #[test]
    fn side_info_writer_is_parse_inverse_lsf_stereo() {
        let h = make_silent_header(64, 22_050, ChannelMode::Stereo).unwrap();
        let si = silent_side_info(&h);
        let bytes = write_side_info(&si);
        assert_eq!(bytes.len(), SIDE_INFO_BYTES_LSF_STEREO);
        let parsed = parse_side_info(&h, &bytes).unwrap();
        assert_eq!(parsed, si);
    }

    #[test]
    fn side_info_writer_is_parse_inverse_lsf_mono() {
        let h = make_silent_header(64, 22_050, ChannelMode::SingleChannel).unwrap();
        let si = silent_side_info(&h);
        let bytes = write_side_info(&si);
        assert_eq!(bytes.len(), SIDE_INFO_BYTES_LSF_MONO);
        let parsed = parse_side_info(&h, &bytes).unwrap();
        assert_eq!(parsed, si);
    }

    #[test]
    fn side_info_writer_roundtrips_nonzero_fields() {
        // Populate a side info with assorted non-default field values
        // (long branch + window-switched branch) and confirm the writer
        // is the parser's inverse for both branches.
        let h = make_silent_header(192, 48_000, ChannelMode::Stereo).unwrap();
        let mut si = silent_side_info(&h);
        si.main_data_begin = 257;
        si.private_bits = 0b101;
        si.scfsi = [[true, false, true, false], [false, true, false, true]];
        // gr0 ch0: long block with non-zero fields.
        si.granules[0][0] = GranuleChannel {
            part2_3_length: 1234,
            big_values: 200,
            global_gain: 210,
            scalefac_compress: 9,
            window_switching_flag: false,
            block_type: BlockType::Long,
            mixed_block_flag: false,
            table_select: [10, 5, 16],
            subblock_gain: [0; 3],
            region0_count: 7,
            region1_count: 3,
            preflag: true,
            scalefac_scale: true,
            count1table_select: true,
        };
        // gr1 ch1: short window-switched block.
        si.granules[1][1] = GranuleChannel {
            part2_3_length: 500,
            big_values: 50,
            global_gain: 180,
            scalefac_compress: 4,
            window_switching_flag: true,
            block_type: BlockType::Short,
            mixed_block_flag: false,
            table_select: [12, 7, 0],
            subblock_gain: [1, 2, 3],
            // §2.4.2.7 defaults for short non-mixed: region0=8, region1=63.
            region0_count: 8,
            region1_count: 63,
            preflag: false,
            scalefac_scale: false,
            count1table_select: true,
        };
        let bytes = write_side_info(&si);
        let parsed = parse_side_info(&h, &bytes).unwrap();
        assert_eq!(parsed, si);
    }

    #[test]
    fn silent_frame_length_matches_header() {
        let h = make_silent_header(128, 44_100, ChannelMode::Stereo).unwrap();
        let frame = encode_silent_frame(&h).unwrap();
        assert_eq!(frame.len(), h.frame_len().unwrap());
        // 144 * 128000 / 44100 = 417 bytes.
        assert_eq!(frame.len(), 417);
    }

    #[test]
    fn silent_frame_roundtrips_through_parsers() {
        for (br, sr, mode) in [
            (128, 44_100, ChannelMode::Stereo),
            (320, 48_000, ChannelMode::Stereo),
            (32, 32_000, ChannelMode::SingleChannel),
            (64, 22_050, ChannelMode::SingleChannel),
            (160, 24_000, ChannelMode::Stereo),
        ] {
            let h = make_silent_header(br, sr, mode).unwrap();
            let frame = encode_silent_frame(&h).unwrap();
            // Header round-trip.
            let hdr = parse_header(&frame[..4]).unwrap();
            assert_eq!(hdr.bitrate_kbps, Some(br));
            assert_eq!(hdr.sample_rate_hz, sr);
            assert!(!hdr.crc_protected);
            // Side info round-trip (no CRC, so side info starts at byte 4).
            let si = parse_side_info(&hdr, &frame[4..]).unwrap();
            assert_eq!(si.main_data_begin, 0);
            for gr in 0..si.granule_count as usize {
                for ch in 0..si.channels as usize {
                    assert_eq!(si.granules[gr][ch].part2_3_length, 0);
                    assert_eq!(si.granules[gr][ch].big_values, 0);
                }
            }
        }
    }

    #[test]
    fn silent_frame_walker_finds_one_frame() {
        use crate::frame::FrameWalker;
        let h = make_silent_header(128, 44_100, ChannelMode::Stereo).unwrap();
        let frame = encode_silent_frame(&h).unwrap();
        let frames: Vec<_> = FrameWalker::new(&frame).collect();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].data.len(), frame.len());
    }

    #[test]
    fn encode_rejects_free_format() {
        let mut h = make_silent_header(128, 44_100, ChannelMode::Stereo).unwrap();
        h.bitrate_index = 0;
        h.bitrate_kbps = None;
        assert_eq!(encode_silent_frame(&h), Err(EncodeError::FreeFormat));
    }

    #[test]
    fn encode_rejects_non_layer3() {
        let mut h = make_silent_header(128, 44_100, ChannelMode::Stereo).unwrap();
        h.layer = Layer::LayerII;
        assert_eq!(encode_silent_frame(&h), Err(EncodeError::NotLayer3));
    }

    #[test]
    fn make_silent_header_rejects_bad_bitrate() {
        // 130 kbps is not on the MPEG-1 Layer III ladder.
        assert_eq!(
            make_silent_header(130, 44_100, ChannelMode::Stereo),
            Err(EncodeError::FreeFormat)
        );
    }

    #[test]
    fn make_silent_header_rejects_bad_sample_rate() {
        assert_eq!(
            make_silent_header(128, 11_025, ChannelMode::Stereo),
            Err(EncodeError::FreeFormat)
        );
    }

    #[test]
    fn make_silent_header_infers_version() {
        let v1 = make_silent_header(128, 44_100, ChannelMode::Stereo).unwrap();
        assert_eq!(v1.version, MpegVersion::Mpeg1);
        let v2 = make_silent_header(64, 16_000, ChannelMode::Stereo).unwrap();
        assert_eq!(v2.version, MpegVersion::Mpeg2);
    }

    #[test]
    fn lsf_silent_frame_has_one_granule() {
        let h = make_silent_header(64, 22_050, ChannelMode::Stereo).unwrap();
        let si = silent_side_info(&h);
        assert_eq!(si.granule_count, GRANULES_LSF as u8);
        assert!(si.lsf);
    }
}
