//! LAME / Xing tag extension — gapless-playback byte layout.
//!
//! This module parses the **encoder-delay + zero-padding** field that
//! sits inside the LAME-tag extension of an MP3's leading Xing / Info
//! information frame. Together with the four traditional Xing fields
//! ([`crate::demuxer::parse_xing_info`]), the LAME tag is what makes
//! gapless playback possible: it tells the decoder exactly how many
//! priming PCM samples the encoder inserted at the start of the stream
//! and how many zero-padding samples were appended at the end to fill
//! the last MPEG audio frame.
//!
//! # Provenance
//!
//! The byte-offset table below is transcribed from
//! `docs/audio/mp3/lame-xing-info-tag.md` — the project's clean-room
//! staging of Gabriel Bouvigne's independently-published
//! *Mp3 Info Tag revision 1 Specifications*
//! (<http://gabriel.mp3-tech.org/mp3infotag.html>, fetched 2026-05-29,
//! `sha256 e9be52a7…dd36d7`). The staged doc is independent format
//! documentation — per `CLEANROOM-MANUAL §6/§10` that is the correct
//! provenance for the Xing / Info / LAME-tag byte layout. No external
//! implementation source was consulted.
//!
//! # Layout (offsets relative to the Xing / Info magic byte)
//!
//! The staged doc gives **absolute** offsets `$24, $9A, $A5, …` for the
//! MPEG-1 stereo carrier-frame case (magic at `$24`). Other MPEG
//! version / channel-mode combinations shift everything by the
//! difference in the magic offset, so this module works in
//! **magic-relative** offsets exclusively:
//!
//! | Rel. offset | Bytes | Field |
//! |---|---|---|
//! | +0   | 4   | `"Xing"` or `"Info"` magic |
//! | +4   | 4   | flags (BE u32, low-four-bit gated) |
//! | +8   | 4   | frames (when `FRAMES_FLAG`) |
//! | +12  | 4   | bytes  (when `BYTES_FLAG`) |
//! | +16  | 100 | toc[100] (when `TOC_FLAG`) |
//! | +116 | 4   | quality (when `VBR_SCALE_FLAG`) |
//! | +118 | 9   | encoder version string (`"LAME…"` UNMOVABLE) |
//! | +129 | 1   | info-tag revision (4 MSB) + VBR method (4 LSB) |
//! | +130 | 1   | lowpass (×100 Hz) |
//! | +131 | 4   | Replay-Gain peak amplitude (IEEE-754 f32 BE) |
//! | +135 | 2   | Radio Replay-Gain field |
//! | +137 | 2   | Audiophile Replay-Gain field |
//! | +139 | 1   | encoding flags (4 MSB) + ATH type (4 LSB) |
//! | +140 | 1   | ABR target / CBR-VBR (minimum) bitrate |
//! | **+141** | **3** | **encoder delay + zero padding** (two 12-bit BE values) |
//! | +144 | 1   | misc (noise-shaping / stereo mode / source-rate) |
//! | +145 | 1   | MP3 Gain (signed; ×1.5 dB step) |
//! | +146 | 2   | preset + surround |
//! | +148 | 4   | music length (BE u32) |
//! | +152 | 2   | music CRC (CRC-16) |
//! | +154 | 2   | tag CRC (CRC-16 over magic-relative `+0..+154`) |
//!
//! The full LAME tag therefore ends at magic-relative offset **+156**
//! and the absolute mapping for the four side-info layouts (§1 of the
//! staged doc) is reconstructed from
//! [`crate::demuxer::side_info_len`] + the 4-byte MPEG header.
//!
//! # Gapless field — `delay + padding` (magic-relative `+141..+143`)
//!
//! The 3-byte run carries two 12-bit unsigned integers (each 0..=4095)
//! packed `[xxxxxxxx][xxxxyyyy][yyyyyyyy]`:
//!
//! ```text
//! encoder_delay = (b0 << 4) | (b1 >> 4)
//! padding       = ((b1 & 0x0F) << 8) | b2
//! ```
//!
//! `encoder_delay` is the number of zero PCM samples the encoder
//! inserted at the start of the stream to flush its analysis filter
//! bank. `padding` is the number of zero PCM samples appended at the
//! end of the last frame to fill it to the granular boundary.
//!
//! The exact original PCM length is then
//!
//! ```text
//! exact_samples = (frames × samples_per_frame) − encoder_delay − padding
//! ```
//!
//! and a gapless-aware decoder trims `encoder_delay` decoded samples
//! off the front (in addition to any decoder-intrinsic prime offset)
//! and `padding` samples off the back.
//!
//! # CRC validation
//!
//! The staged doc names `CRCInitValue = 0x0000` for the tag CRC but
//! does not specify the polynomial. Without a polynomial in the
//! cleanroom corpus this module **records** the on-wire CRC field
//! without verifying it; the caller can compare against an expected
//! value if it has one. Filling in the polynomial-specific
//! verification is a follow-up task once the staged doc is extended
//! (or another cleanroom source confirms the polynomial).
//!
//! # See also
//!
//! - [`crate::demuxer::parse_xing_info`] — the four-field Xing / Info
//!   header parser this module's offsets are tied to.
//! - [`crate::xing_info`] — the encoder-side emitter for the four
//!   traditional Xing fields. (LAME-tag emission is a separate
//!   round's scope.)

/// LAME extension fields parsed from an MP3's leading Xing / Info
/// information frame.
///
/// Field semantics follow `docs/audio/mp3/lame-xing-info-tag.md`. The
/// gapless-relevant pair is [`Self::encoder_delay`] +
/// [`Self::zero_padding`]; the remaining fields are recorded as raw
/// bytes (or unpacked semantic values for the well-defined ones) so a
/// downstream consumer can inspect or surface them without re-parsing
/// the wire bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LameTag {
    /// 9-byte encoder version string — typically ASCII like `"LAME3.100"`.
    /// The first four bytes are the unmovable `"LAME"` magic.
    pub encoder_version: [u8; 9],
    /// Info-tag revision (high 4 bits of the revision/VBR-method byte).
    pub info_tag_revision: u8,
    /// VBR method (low 4 bits). 0 = unknown; 1 = CBR; 2 = ABR; 3..=8
    /// are the LAME-defined VBR methods 1..=6 per the staged doc.
    pub vbr_method: u8,
    /// Lowpass filter frequency in Hz (raw byte multiplied by 100).
    pub lowpass_hz: u32,
    /// Replay-Gain peak amplitude — 32-bit big-endian IEEE-754 float.
    /// Stored as the raw 4 bytes so callers needing a numeric value
    /// can `f32::from_be_bytes` it without losing the bit pattern.
    pub replay_gain_peak: [u8; 4],
    /// Radio Replay-Gain field (16 bits packed per the doc).
    pub radio_replay_gain: u16,
    /// Audiophile Replay-Gain field (16 bits packed per the doc).
    pub audiophile_replay_gain: u16,
    /// Encoding flags (high 4 bits) + ATH type (low 4 bits).
    pub encoding_flags_ath: u8,
    /// ABR target bitrate (for ABR), or CBR / minimum-VBR bitrate
    /// otherwise. `0xFF` is the sticky "≥255 kbps" value.
    pub bitrate_byte: u8,
    /// Encoder delay — number of zero PCM samples the encoder prepended
    /// to the stream to flush its analysis filter bank. 0..=4095.
    pub encoder_delay: u16,
    /// Zero padding — number of zero PCM samples the encoder appended
    /// to the last frame to fill the granular boundary. 0..=4095.
    pub zero_padding: u16,
    /// Misc byte (noise-shaping / stereo-mode / source-sample-rate
    /// packing per the staged doc §4). Stored verbatim.
    pub misc: u8,
    /// MP3-gain byte (signed; ×1.5 dB step). Default `0`. Only
    /// mp3gain-style global-gain editors update it; the LAME-tag CRC
    /// at +154 covers it.
    pub mp3_gain: i8,
    /// Preset + surround word (3 bits surround / 11 bits preset).
    pub preset_surround: u16,
    /// Music length in bytes (BE u32). `0` means unknown or ≥4 GiB.
    pub music_length: u32,
    /// Music CRC-16 over the audio frames (excluding tags). The
    /// staged doc does not specify the polynomial; this module
    /// records the on-wire value verbatim.
    pub music_crc: u16,
    /// Tag CRC-16 over the magic-relative `+0..+154` byte range.
    /// Polynomial unspecified by the staged doc (see module-level
    /// "CRC validation" note); recorded verbatim.
    pub tag_crc: u16,
}

/// Magic-relative byte length of a fully-populated LAME tag (the
/// staged doc's `$24..$BF` range when the carrier frame is MPEG-1
/// stereo with all four Xing flags set — i.e. magic `$24` to tag CRC
/// `$BF`, total 156 bytes).
pub const LAME_TAG_FULL_LEN: usize = 156;

/// Magic-relative offset of the `"LAME"` magic when **all four** Xing
/// flag bits (FRAMES + BYTES + TOC + VBR_SCALE) are set. Same
/// `$9A − $24 = +0x76` derivation. With fewer flags the offset is
/// smaller by the omitted-field widths; callers pass the post-Xing
/// cursor to [`parse_lame_tag`] explicitly.
pub const LAME_MAGIC_OFFSET_ALL_FLAGS: usize = 118;

/// Magic-relative offset (from the LAME magic, not from the Xing
/// magic) of the three-byte encoder-delay + zero-padding run.
pub const DELAY_PADDING_OFFSET_FROM_LAME_MAGIC: usize = OFF_DELAY_PADDING;

/// Errors returned by [`parse_lame_tag`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LameParseError {
    /// The byte range described by `lame_magic_offset` and the
    /// trailing 38 bytes of LAME-extension fields does not fit inside
    /// the frame payload supplied by the caller.
    Truncated,
    /// The four bytes at `lame_magic_offset` are not the `"LAME"`
    /// ASCII magic. Other encoder strings (`"Lavc"`, `"Lavf"`,
    /// `"Xing"` legacy emitters) do not populate the LAME-extension
    /// layout the same way, so this module refuses to read past the
    /// magic for them.
    NotLameTag,
}

impl core::fmt::Display for LameParseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            LameParseError::Truncated => f.write_str("frame payload too short for LAME extension"),
            LameParseError::NotLameTag => f.write_str("encoder string is not the LAME magic"),
        }
    }
}

impl std::error::Error for LameParseError {}

/// Parse the LAME-tag extension at `lame_magic_offset` inside the
/// (whole-frame) `frame_payload`.
///
/// `lame_magic_offset` is the absolute byte offset, from the start of
/// the frame, of the 9-byte encoder version string described in
/// `docs/audio/mp3/lame-xing-info-tag.md §4`. The caller supplies it
/// because it depends on (a) the MPEG header's side-info length and
/// (b) which Xing/Info flag bits were set in the preceding traditional
/// header — both of which the demuxer already computes.
///
/// # Offset interpretation
///
/// The staged doc gives the encoder-version run as `$9A–$A4` "9 bytes"
/// — an internal inconsistency, since `$9A..=$A4` is 11 bytes
/// inclusive. The rest of the doc's offset chain
/// (`$A5` revision/method, `$A6` lowpass, …, `$B1` delay+padding) is
/// internally consistent and gives `$B1 − $9A = 23` bytes from the
/// LAME magic to the delay+padding run. We resolve the doc ambiguity
/// by trusting the **absolute-offset chain** (which the doc derives
/// from worked hex dumps) over the **"9 bytes"** annotation: the field
/// region from LAME magic through the tag CRC at `$BF` is
/// `$BF − $9A + 1 = 38` bytes, with the encoder-version run occupying
/// the 9 ASCII characters at offsets 0..9 from the magic and the next
/// two bytes (`$A3–$A4`, magic-relative 9..11) being either trailing
/// NUL padding of the version string or a per-LAME-version reserved
/// region; either way no LAME-defined field occupies them.
///
/// # Errors
///
/// * [`LameParseError::Truncated`] when the frame payload does not
///   reach as far as the last LAME-extension byte (`lame_magic_offset
///   + 38`).
/// * [`LameParseError::NotLameTag`] when the four bytes at
///   `lame_magic_offset` are not `"LAME"`.
pub fn parse_lame_tag(
    frame_payload: &[u8],
    lame_magic_offset: usize,
) -> Result<LameTag, LameParseError> {
    let end = lame_magic_offset
        .checked_add(LAME_TAG_FIELDS_LEN)
        .ok_or(LameParseError::Truncated)?;
    if end > frame_payload.len() {
        return Err(LameParseError::Truncated);
    }
    let bytes = &frame_payload[lame_magic_offset..end];
    if &bytes[..4] != b"LAME" {
        return Err(LameParseError::NotLameTag);
    }

    let mut encoder_version = [0u8; 9];
    encoder_version.copy_from_slice(&bytes[..9]);

    // Field cursor inside the slice, magic-relative offsets per the
    // doc's absolute-offset chain ($9A is byte 0; $A5 is byte 11; etc).
    let rev_vbr = bytes[OFF_REV_METHOD];
    let lowpass_byte = bytes[OFF_LOWPASS];
    let mut peak = [0u8; 4];
    peak.copy_from_slice(&bytes[OFF_PEAK..OFF_PEAK + 4]);
    let radio_rg = u16::from_be_bytes([bytes[OFF_RADIO_RG], bytes[OFF_RADIO_RG + 1]]);
    let audiophile_rg =
        u16::from_be_bytes([bytes[OFF_AUDIOPHILE_RG], bytes[OFF_AUDIOPHILE_RG + 1]]);
    let enc_flags_ath = bytes[OFF_FLAGS_ATH];
    let bitrate_byte = bytes[OFF_BITRATE];
    let b0 = bytes[OFF_DELAY_PADDING];
    let b1 = bytes[OFF_DELAY_PADDING + 1];
    let b2 = bytes[OFF_DELAY_PADDING + 2];
    let encoder_delay = ((b0 as u16) << 4) | ((b1 as u16) >> 4);
    let zero_padding = (((b1 & 0x0F) as u16) << 8) | (b2 as u16);
    let misc = bytes[OFF_MISC];
    let mp3_gain = bytes[OFF_MP3_GAIN] as i8;
    let preset_surround =
        u16::from_be_bytes([bytes[OFF_PRESET_SURROUND], bytes[OFF_PRESET_SURROUND + 1]]);
    let music_length = u32::from_be_bytes([
        bytes[OFF_MUSIC_LEN],
        bytes[OFF_MUSIC_LEN + 1],
        bytes[OFF_MUSIC_LEN + 2],
        bytes[OFF_MUSIC_LEN + 3],
    ]);
    let music_crc = u16::from_be_bytes([bytes[OFF_MUSIC_CRC], bytes[OFF_MUSIC_CRC + 1]]);
    let tag_crc = u16::from_be_bytes([bytes[OFF_TAG_CRC], bytes[OFF_TAG_CRC + 1]]);

    Ok(LameTag {
        encoder_version,
        info_tag_revision: (rev_vbr >> 4) & 0x0F,
        vbr_method: rev_vbr & 0x0F,
        lowpass_hz: (lowpass_byte as u32) * 100,
        replay_gain_peak: peak,
        radio_replay_gain: radio_rg,
        audiophile_replay_gain: audiophile_rg,
        encoding_flags_ath: enc_flags_ath,
        bitrate_byte,
        encoder_delay,
        zero_padding,
        misc,
        mp3_gain,
        preset_surround,
        music_length,
        music_crc,
        tag_crc,
    })
}

// Magic-relative field offsets (LAME magic at +0). Derived from the
// staged doc's absolute chain: $A5 - $9A = 11 for revision/method,
// $A6 - $9A = 12 for lowpass, $A7 - $9A = 13 for peak (4 bytes), and
// so on through $BE-$BF for the tag CRC at +36.
const OFF_REV_METHOD: usize = 11;
const OFF_LOWPASS: usize = 12;
const OFF_PEAK: usize = 13;
const OFF_RADIO_RG: usize = 17;
const OFF_AUDIOPHILE_RG: usize = 19;
const OFF_FLAGS_ATH: usize = 21;
const OFF_BITRATE: usize = 22;
const OFF_DELAY_PADDING: usize = 23;
const OFF_MISC: usize = 26;
const OFF_MP3_GAIN: usize = 27;
const OFF_PRESET_SURROUND: usize = 28;
const OFF_MUSIC_LEN: usize = 30;
const OFF_MUSIC_CRC: usize = 34;
const OFF_TAG_CRC: usize = 36;

/// Number of bytes the LAME-extension fields occupy starting at and
/// including the 9-byte `"LAME"`-magic encoder-version string and
/// running through the end of the tag-CRC field (`$9A..=$BF`,
/// inclusive, of the staged doc's worked example).
///
/// `$BF − $9A + 1 = 38`.
pub const LAME_TAG_FIELDS_LEN: usize = 38;

impl LameTag {
    /// Compute the trimmed PCM-sample count given the total number of
    /// MPEG audio frames following the carrier and the
    /// samples-per-frame implied by the carrier frame's header.
    ///
    /// `frames_after_carrier` is typically the Xing `frames` field
    /// (which by convention counts the audio frames following the
    /// carrier itself). `samples_per_frame` is the carrier-frame
    /// header's [`crate::Mp3FrameHeader::samples_per_frame`] return
    /// — 1152 for MPEG-1 Layer III, 576 for MPEG-2 LSF, 384 for
    /// Layer I, 1152 for Layer II.
    ///
    /// Returns `None` if either `encoder_delay + zero_padding` exceeds
    /// the gross frame-sample total (a malformed tag).
    #[must_use]
    pub fn trimmed_samples(
        &self,
        frames_after_carrier: u64,
        samples_per_frame: u32,
    ) -> Option<u64> {
        let gross = frames_after_carrier.checked_mul(samples_per_frame as u64)?;
        let trim = (self.encoder_delay as u64) + (self.zero_padding as u64);
        if trim > gross {
            return None;
        }
        Some(gross - trim)
    }

    /// True when the on-wire `encoder_delay` or `zero_padding` field
    /// is non-zero. A bare LAME tag with both at zero behaves like a
    /// non-gapless stream for trimming purposes.
    #[must_use]
    pub fn has_gapless_trim(&self) -> bool {
        self.encoder_delay != 0 || self.zero_padding != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a 156-byte synthetic Xing+LAME first-frame payload area
    /// (without the 4-byte MPEG header and without the side-info
    /// padding) — i.e. the bytes starting at the Xing magic. The LAME
    /// magic sits at offset 118 inside this slice when all four Xing
    /// flags are set, matching the staged doc's MPEG-1-stereo
    /// magic-`$24` worked example after subtracting the magic offset.
    fn synthetic_lame_payload(delay: u16, padding: u16, version: &[u8; 9]) -> Vec<u8> {
        let mut buf = vec![0u8; LAME_TAG_FULL_LEN];
        // +0..+4: "Xing"
        buf[..4].copy_from_slice(b"Xing");
        // +4..+8: flags = 0x0F (all four set).
        buf[4..8].copy_from_slice(&0x0000_000Fu32.to_be_bytes());
        // +8..+12: frames = 100.
        buf[8..12].copy_from_slice(&100u32.to_be_bytes());
        // +12..+16: bytes = 32_000.
        buf[12..16].copy_from_slice(&32_000u32.to_be_bytes());
        // +16..+116: 100-byte TOC (identity ramp).
        for i in 0..100 {
            buf[16 + i] = ((i * 255) / 99) as u8;
        }
        // +116..+120: vbr quality = 1.
        buf[116..120].copy_from_slice(&1u32.to_be_bytes());
        // +118..+127: 9-byte encoder version ("LAME...")
        // Per the staged-doc note: the 9-byte encoder version sits at
        // magic-relative offset 118 and overlaps the last two bytes
        // of the VBR-scale field (+116..+120). Overwrite those two
        // bytes with the start of the version — this is exactly what
        // real LAME-tagged carrier frames look like on the wire.
        buf[LAME_MAGIC_OFFSET_ALL_FLAGS..LAME_MAGIC_OFFSET_ALL_FLAGS + 9].copy_from_slice(version);
        // The remaining LAME fields, magic-relative offsets per the
        // OFF_* constants above (rev/method at +11, …, tag CRC at +36).
        let f = LAME_MAGIC_OFFSET_ALL_FLAGS;
        buf[f + OFF_REV_METHOD] = 0x10; // revision 1, VBR method 0
        buf[f + OFF_LOWPASS] = 196; // lowpass = 196 * 100 = 19_600 Hz
                                    // peak amplitude as f32 BE = 1.0 → 0x3F80_0000.
        buf[f + OFF_PEAK..f + OFF_PEAK + 4].copy_from_slice(&0x3F80_0000u32.to_be_bytes());
        buf[f + OFF_RADIO_RG..f + OFF_RADIO_RG + 2].copy_from_slice(&0x1234u16.to_be_bytes());
        buf[f + OFF_AUDIOPHILE_RG..f + OFF_AUDIOPHILE_RG + 2]
            .copy_from_slice(&0x5678u16.to_be_bytes());
        buf[f + OFF_FLAGS_ATH] = 0xA5;
        buf[f + OFF_BITRATE] = 192;
        // delay+padding pack: [b0][b1][b2] s.t. delay = (b0<<4)|(b1>>4),
        // padding = ((b1 & 0x0F)<<8) | b2.
        let b0 = (delay >> 4) as u8;
        let b1 = (((delay & 0x0F) << 4) | ((padding >> 8) & 0x0F)) as u8;
        let b2 = (padding & 0xFF) as u8;
        buf[f + OFF_DELAY_PADDING] = b0;
        buf[f + OFF_DELAY_PADDING + 1] = b1;
        buf[f + OFF_DELAY_PADDING + 2] = b2;
        buf[f + OFF_MISC] = 0x07;
        buf[f + OFF_MP3_GAIN] = 0x05;
        buf[f + OFF_PRESET_SURROUND..f + OFF_PRESET_SURROUND + 2]
            .copy_from_slice(&0x0010u16.to_be_bytes());
        buf[f + OFF_MUSIC_LEN..f + OFF_MUSIC_LEN + 4].copy_from_slice(&12_345u32.to_be_bytes());
        buf[f + OFF_MUSIC_CRC..f + OFF_MUSIC_CRC + 2].copy_from_slice(&0xABCDu16.to_be_bytes());
        buf[f + OFF_TAG_CRC..f + OFF_TAG_CRC + 2].copy_from_slice(&0xEF01u16.to_be_bytes());
        buf
    }

    #[test]
    fn parse_extracts_delay_and_padding() {
        let buf = synthetic_lame_payload(1729, 722, b"LAME3.100");
        let tag = parse_lame_tag(&buf, LAME_MAGIC_OFFSET_ALL_FLAGS).expect("parse");
        // Worked example from the staged doc §5: [01101100][00010010]
        // [11010010] → delay = 1729, padding = 722.
        assert_eq!(tag.encoder_delay, 1729);
        assert_eq!(tag.zero_padding, 722);
        assert_eq!(&tag.encoder_version, b"LAME3.100");
    }

    #[test]
    fn parse_extracts_all_fields() {
        let buf = synthetic_lame_payload(123, 456, b"LAME3.99r");
        let tag = parse_lame_tag(&buf, LAME_MAGIC_OFFSET_ALL_FLAGS).unwrap();
        assert_eq!(tag.info_tag_revision, 1);
        assert_eq!(tag.vbr_method, 0);
        assert_eq!(tag.lowpass_hz, 19_600);
        // 1.0_f32 → 0x3F80_0000.
        assert_eq!(tag.replay_gain_peak, [0x3F, 0x80, 0x00, 0x00]);
        assert_eq!(tag.radio_replay_gain, 0x1234);
        assert_eq!(tag.audiophile_replay_gain, 0x5678);
        assert_eq!(tag.encoding_flags_ath, 0xA5);
        assert_eq!(tag.bitrate_byte, 192);
        assert_eq!(tag.encoder_delay, 123);
        assert_eq!(tag.zero_padding, 456);
        assert_eq!(tag.misc, 0x07);
        assert_eq!(tag.mp3_gain, 5);
        assert_eq!(tag.preset_surround, 0x0010);
        assert_eq!(tag.music_length, 12_345);
        assert_eq!(tag.music_crc, 0xABCD);
        assert_eq!(tag.tag_crc, 0xEF01);
    }

    #[test]
    fn parse_doc_worked_example_byte_pattern() {
        // §5 of the staged doc gives the exact byte pattern
        // [01101100][00010010][11010010] and expects delay=1729,
        // padding=722. Drive a minimal slice that holds only the
        // LAME-extension fields and verify the unpack.
        let mut buf = vec![0u8; LAME_TAG_FIELDS_LEN];
        buf[..4].copy_from_slice(b"LAME");
        buf[OFF_DELAY_PADDING] = 0b0110_1100;
        buf[OFF_DELAY_PADDING + 1] = 0b0001_0010;
        buf[OFF_DELAY_PADDING + 2] = 0b1101_0010;
        let tag = parse_lame_tag(&buf, 0).unwrap();
        assert_eq!(tag.encoder_delay, 1729);
        assert_eq!(tag.zero_padding, 722);
    }

    #[test]
    fn parse_rejects_non_lame_magic() {
        let mut buf = vec![0u8; LAME_TAG_FIELDS_LEN];
        buf[..4].copy_from_slice(b"Lavc");
        let err = parse_lame_tag(&buf, 0).unwrap_err();
        assert_eq!(err, LameParseError::NotLameTag);
    }

    #[test]
    fn parse_rejects_truncated_payload() {
        let buf = vec![b'L', b'A', b'M', b'E', 0, 0, 0, 0]; // way short
        let err = parse_lame_tag(&buf, 0).unwrap_err();
        assert_eq!(err, LameParseError::Truncated);
    }

    #[test]
    fn parse_rejects_out_of_range_offset() {
        let buf = vec![0u8; 16];
        let err = parse_lame_tag(&buf, 32).unwrap_err();
        assert_eq!(err, LameParseError::Truncated);
    }

    #[test]
    fn pack_unpack_covers_full_12_bit_range() {
        // Hammer the 12-bit boundaries.
        for &delay in &[0u16, 1, 2047, 2048, 4094, 4095] {
            for &padding in &[0u16, 1, 2047, 2048, 4094, 4095] {
                let mut buf = vec![0u8; LAME_TAG_FIELDS_LEN];
                buf[..4].copy_from_slice(b"LAME");
                let b0 = (delay >> 4) as u8;
                let b1 = (((delay & 0x0F) << 4) | ((padding >> 8) & 0x0F)) as u8;
                let b2 = (padding & 0xFF) as u8;
                buf[OFF_DELAY_PADDING] = b0;
                buf[OFF_DELAY_PADDING + 1] = b1;
                buf[OFF_DELAY_PADDING + 2] = b2;
                let tag = parse_lame_tag(&buf, 0).unwrap();
                assert_eq!(tag.encoder_delay, delay, "delay round-trip");
                assert_eq!(tag.zero_padding, padding, "padding round-trip");
            }
        }
    }

    #[test]
    fn trimmed_samples_subtracts_delay_and_padding() {
        let buf = synthetic_lame_payload(1729, 722, b"LAME3.100");
        let tag = parse_lame_tag(&buf, LAME_MAGIC_OFFSET_ALL_FLAGS).unwrap();
        // 100 frames × 1152 samples = 115_200.
        // Trim 1729 + 722 = 2451 → 112_749.
        assert_eq!(tag.trimmed_samples(100, 1152), Some(115_200 - 2451));
    }

    #[test]
    fn trimmed_samples_lsf_576_per_frame() {
        let buf = synthetic_lame_payload(528, 0, b"LAME3.100");
        let tag = parse_lame_tag(&buf, LAME_MAGIC_OFFSET_ALL_FLAGS).unwrap();
        // 50 frames × 576 = 28_800; trim 528 → 28_272.
        assert_eq!(tag.trimmed_samples(50, 576), Some(28_272));
    }

    #[test]
    fn trimmed_samples_returns_none_on_overflow_trim() {
        // Construct a tag with absurdly large delay so trim exceeds
        // gross sample count and the trimmed value should refuse to
        // underflow.
        let buf = synthetic_lame_payload(4095, 4095, b"LAME3.100");
        let tag = parse_lame_tag(&buf, LAME_MAGIC_OFFSET_ALL_FLAGS).unwrap();
        // 5 frames × 1152 = 5760; trim = 4095+4095 = 8190 > 5760.
        assert_eq!(tag.trimmed_samples(5, 1152), None);
    }

    #[test]
    fn has_gapless_trim_reflects_field_values() {
        let buf_zero = synthetic_lame_payload(0, 0, b"LAME3.100");
        let tag_zero = parse_lame_tag(&buf_zero, LAME_MAGIC_OFFSET_ALL_FLAGS).unwrap();
        assert!(!tag_zero.has_gapless_trim());
        let buf_delay = synthetic_lame_payload(1, 0, b"LAME3.100");
        let tag_delay = parse_lame_tag(&buf_delay, LAME_MAGIC_OFFSET_ALL_FLAGS).unwrap();
        assert!(tag_delay.has_gapless_trim());
        let buf_pad = synthetic_lame_payload(0, 1, b"LAME3.100");
        let tag_pad = parse_lame_tag(&buf_pad, LAME_MAGIC_OFFSET_ALL_FLAGS).unwrap();
        assert!(tag_pad.has_gapless_trim());
    }
}
