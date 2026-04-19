//! MPEG-1 / MPEG-2 LSF Layer III side information parser.
//!
//! Side information immediately follows the 4-byte frame header (and
//! optional 2-byte CRC). It directs main-data decoding: Huffman table
//! selection, scalefactor encoding, window-switching, and bit-reservoir
//! offset.
//!
//! This module handles **MPEG-1 Layer III** (2 granules, 32/17 bytes)
//! and **MPEG-2 LSF Layer III** (1 granule, 17/9 bytes). MPEG-2.5 uses
//! the same side-info shape as MPEG-2 LSF but is not wired up yet.
//!
//! Layout (MPEG-1, stereo = 32 bytes, mono = 17 bytes), bits read MSB-first:
//!
//! ```text
//!   main_data_begin    : 9   (byte offset back into the bit reservoir)
//!   private_bits       : 3 (mono) or 5 (stereo)
//!   scfsi[ch][group]   : ch * 4 bits (1 per scalefactor-group-4)
//!   --- then 2 granules, each with per-channel block: ---
//!     part2_3_length   : 12 — total main-data bits for this gr+ch
//!     big_values       : 9  — non-zero coeff count in "bigvalues" region
//!     global_gain      : 8
//!     scalefac_compress: 4  — indexes slen1/slen2 table
//!     windows_switching_flag : 1
//!     if windows_switching_flag:
//!         block_type       : 2  — 0 normal, 1 start, 2 short, 3 stop
//!         mixed_block_flag : 1
//!         table_select[0..2] : 2 * 5
//!         subblock_gain[0..3]: 3 * 3
//!     else:
//!         table_select[0..3] : 3 * 5
//!         region0_count    : 4
//!         region1_count    : 3
//!     preflag           : 1
//!     scalefac_scale    : 1
//!     count1table_select: 1
//! ```
//!
//! Layout (MPEG-2 LSF, stereo = 17 bytes, mono = 9 bytes):
//!
//! ```text
//!   main_data_begin    : 8   (one bit narrower than MPEG-1)
//!   private_bits       : 1 (mono) or 2 (stereo)
//!   --- no scfsi (MPEG-2 LSF has only one granule) ---
//!   --- one granule, per-channel block: ---
//!     part2_3_length   : 12
//!     big_values       : 9
//!     global_gain      : 8
//!     scalefac_compress: 9   (MPEG-2: indirect slen derivation)
//!     window_switching_flag : 1
//!     if wsf:  block_type(2) mixed_block_flag(1) 2*table_select(5) 3*subblock_gain(3)
//!     else:    3*table_select(5) region0_count(4) region1_count(3)
//!     scalefac_scale    : 1
//!     count1table_select: 1
//!   --- NO preflag bit in MPEG-2 side info (derived from scalefac_compress) ---
//! ```

use oxideav_core::{Error, Result};

use crate::frame::{ChannelMode, FrameHeader, MpegVersion};
use oxideav_core::bits::BitReader;

/// Per-granule, per-channel side info.
#[derive(Clone, Copy, Debug, Default)]
pub struct GranuleChannel {
    pub part2_3_length: u16,
    pub big_values: u16,
    pub global_gain: u8,
    /// MPEG-1: 4-bit scalefac_compress field from side info.
    pub scalefac_compress: u8,
    /// MPEG-2 LSF: 9-bit scalefac_compress. Ignored (left 0) for MPEG-1.
    pub scalefac_compress_9: u16,
    pub window_switching_flag: bool,
    /// 0 = long (normal), 1 = start, 2 = short, 3 = stop.
    pub block_type: u8,
    pub mixed_block_flag: bool,
    pub table_select: [u8; 3],
    pub subblock_gain: [u8; 3],
    pub region0_count: u8,
    pub region1_count: u8,
    pub preflag: bool,
    pub scalefac_scale: bool,
    pub count1table_select: bool,
}

/// Decoded side-information block for one frame.
///
/// For MPEG-1 both `granules[0]` and `granules[1]` are populated (two
/// granules per frame). For MPEG-2 LSF only `granules[0]` is used — the
/// MPEG-2 layout has a single granule per frame, and `num_granules` is
/// 1 in that case.
#[derive(Clone, Debug)]
pub struct SideInfo {
    /// Offset (in bytes) into the bit reservoir — main data for this frame
    /// starts `main_data_begin` bytes BEFORE the side-info block end.
    pub main_data_begin: u16,
    /// Number of channels (1 or 2).
    pub channels: u8,
    /// Number of granules in the frame — 2 for MPEG-1, 1 for MPEG-2/2.5.
    pub num_granules: u8,
    /// scfsi[ch][group] — group is 0..4 corresponding to sfb 0-5, 6-10,
    /// 11-15, 16-20. Only valid for MPEG-1 (MPEG-2 LSF has no scfsi since
    /// there's only one granule per frame).
    pub scfsi: [[bool; 4]; 2],
    /// gr[0..2][ch], channels used per `channels` field above. For MPEG-2,
    /// only row `[0][..]` is valid.
    pub granules: [[GranuleChannel; 2]; 2],
}

impl SideInfo {
    /// Parse MPEG-1 Layer III side info from the start of `bytes`.
    /// Consumes exactly `17` (mono) or `32` (stereo) bytes.
    pub fn parse_mpeg1(header: &FrameHeader, bytes: &[u8]) -> Result<Self> {
        if header.version != MpegVersion::Mpeg1 {
            return Err(Error::unsupported(
                "MP3 side info: MPEG-2/2.5 not yet supported",
            ));
        }
        let channels = header.channel_mode.channel_count() as usize;
        let needed = header.side_info_bytes();
        if bytes.len() < needed {
            return Err(Error::NeedMore);
        }
        let mut br = BitReader::new(&bytes[..needed]);

        let main_data_begin = br.read_u32(9)? as u16;
        let private_bits = if channels == 1 { 5 } else { 3 };
        let _ = br.read_u32(private_bits)?;

        let mut scfsi = [[false; 4]; 2];
        for scfsi_row in scfsi.iter_mut().take(channels) {
            for v in scfsi_row.iter_mut() {
                *v = br.read_bit()?;
            }
        }

        let mut granules = [[GranuleChannel::default(); 2]; 2];
        for gr in 0..2 {
            for ch in 0..channels {
                granules[gr][ch] = parse_granule_channel(&mut br)?;
            }
        }

        // The ChannelMode field `channel_mode_count` in `FrameHeader` could still
        // be StereoMode which yields 2 channels; if mono but somehow mismatched,
        // return error.
        if header.channel_mode == ChannelMode::Mono && channels != 1 {
            return Err(Error::invalid("MP3 side info: channel mismatch"));
        }

        Ok(SideInfo {
            main_data_begin,
            channels: channels as u8,
            num_granules: 2,
            scfsi,
            granules,
        })
    }

    /// Parse MPEG-2 LSF Layer III side info from the start of `bytes`.
    /// Consumes exactly `9` (mono) or `17` (stereo) bytes.
    ///
    /// Differences from MPEG-1 (see module docs):
    /// - `main_data_begin` is 8 bits (not 9).
    /// - Private bits follow: 1 (mono) or 2 (stereo).
    /// - No `scfsi` — there is only one granule per MPEG-2 frame.
    /// - One granule; `scalefac_compress` is 9 bits instead of 4; there is
    ///   no transmitted `preflag` bit (preflag is derived from
    ///   `scalefac_compress >= 500`, see `scalefactor::decode_mpeg2`).
    pub fn parse_mpeg2(header: &FrameHeader, bytes: &[u8]) -> Result<Self> {
        if header.version == MpegVersion::Mpeg1 {
            return Err(Error::invalid(
                "MP3 side info: parse_mpeg2 called on MPEG-1 header",
            ));
        }
        let channels = header.channel_mode.channel_count() as usize;
        let needed = header.side_info_bytes();
        if bytes.len() < needed {
            return Err(Error::NeedMore);
        }
        let mut br = BitReader::new(&bytes[..needed]);

        let main_data_begin = br.read_u32(8)? as u16;
        // Private bits: 1 for mono, 2 for stereo.
        let private_bits = if channels == 1 { 1 } else { 2 };
        let _ = br.read_u32(private_bits)?;

        let mut granules = [[GranuleChannel::default(); 2]; 2];
        for ch in 0..channels {
            granules[0][ch] = parse_granule_channel_mpeg2(&mut br)?;
        }

        if header.channel_mode == ChannelMode::Mono && channels != 1 {
            return Err(Error::invalid("MP3 side info: channel mismatch"));
        }

        Ok(SideInfo {
            main_data_begin,
            channels: channels as u8,
            num_granules: 1,
            scfsi: [[false; 4]; 2],
            granules,
        })
    }
}

fn parse_granule_channel(br: &mut BitReader<'_>) -> Result<GranuleChannel> {
    let part2_3_length = br.read_u32(12)? as u16;
    let big_values = br.read_u32(9)? as u16;
    let global_gain = br.read_u32(8)? as u8;
    let scalefac_compress = br.read_u32(4)? as u8;
    let window_switching_flag = br.read_bit()?;

    let mut gc = GranuleChannel {
        part2_3_length,
        big_values,
        global_gain,
        scalefac_compress,
        scalefac_compress_9: 0,
        window_switching_flag,
        block_type: 0,
        mixed_block_flag: false,
        table_select: [0; 3],
        subblock_gain: [0; 3],
        region0_count: 0,
        region1_count: 0,
        preflag: false,
        scalefac_scale: false,
        count1table_select: false,
    };

    if window_switching_flag {
        gc.block_type = br.read_u32(2)? as u8;
        gc.mixed_block_flag = br.read_bit()?;
        for i in 0..2 {
            gc.table_select[i] = br.read_u32(5)? as u8;
        }
        for i in 0..3 {
            gc.subblock_gain[i] = br.read_u32(3)? as u8;
        }
        // Per ISO/IEC 11172-3 §2.4.2.7 Table: region0_count and region1_count
        // are NOT sent when window_switching_flag is set.
        // Implicit region0_count = 8 (long blocks) or 7 (short blocks?),
        // see Annex. We use 8/36 scheme for long / 9 for short (see requant).
        // These fields being 0 here is fine — they'll be re-derived from
        // block_type during requantisation.
        if gc.block_type == 2 && !gc.mixed_block_flag {
            gc.region0_count = 8; // spec: when short blocks, 36 samples long
        } else {
            gc.region0_count = 7;
        }
        gc.region1_count = 36; // always 36 for switching
    } else {
        for i in 0..3 {
            gc.table_select[i] = br.read_u32(5)? as u8;
        }
        gc.region0_count = br.read_u32(4)? as u8;
        gc.region1_count = br.read_u32(3)? as u8;
    }

    gc.preflag = br.read_bit()?;
    gc.scalefac_scale = br.read_bit()?;
    gc.count1table_select = br.read_bit()?;

    Ok(gc)
}

/// Parse one MPEG-2 LSF granule+channel block from the bitstream.
/// MPEG-2 LSF differences (see ISO/IEC 13818-3 §2.4.2.7):
///   - `scalefac_compress` is 9 bits (vs 4 for MPEG-1).
///   - No `preflag` bit — derived later from `scalefac_compress >= 500`.
fn parse_granule_channel_mpeg2(br: &mut BitReader<'_>) -> Result<GranuleChannel> {
    let part2_3_length = br.read_u32(12)? as u16;
    let big_values = br.read_u32(9)? as u16;
    let global_gain = br.read_u32(8)? as u8;
    let scalefac_compress = br.read_u32(9)? as u16;
    let window_switching_flag = br.read_bit()?;

    // Stash scalefac_compress in the same u8 field used for MPEG-1 won't
    // work — use `scalefac_compress_9` below for MPEG-2 scalefactor decode.
    let mut gc = GranuleChannel {
        part2_3_length,
        big_values,
        global_gain,
        scalefac_compress: 0,
        scalefac_compress_9: scalefac_compress,
        window_switching_flag,
        block_type: 0,
        mixed_block_flag: false,
        table_select: [0; 3],
        subblock_gain: [0; 3],
        region0_count: 0,
        region1_count: 0,
        preflag: false,
        scalefac_scale: false,
        count1table_select: false,
    };

    if window_switching_flag {
        gc.block_type = br.read_u32(2)? as u8;
        gc.mixed_block_flag = br.read_bit()?;
        for i in 0..2 {
            gc.table_select[i] = br.read_u32(5)? as u8;
        }
        for i in 0..3 {
            gc.subblock_gain[i] = br.read_u32(3)? as u8;
        }
        if gc.block_type == 2 && !gc.mixed_block_flag {
            gc.region0_count = 8;
        } else {
            gc.region0_count = 7;
        }
        gc.region1_count = 36;
    } else {
        for i in 0..3 {
            gc.table_select[i] = br.read_u32(5)? as u8;
        }
        gc.region0_count = br.read_u32(4)? as u8;
        gc.region1_count = br.read_u32(3)? as u8;
    }

    // MPEG-2: no transmitted preflag; derive from scalefac_compress.
    gc.preflag = scalefac_compress >= 500;
    gc.scalefac_scale = br.read_bit()?;
    gc.count1table_select = br.read_bit()?;

    Ok(gc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{parse_frame_header, parse_frame_header_u32};

    #[test]
    fn parses_mpeg1_mono_side_info_length() {
        // MPEG-1 L3 mono 128k / 48kHz: FF FB 94 C0 (mode 11 = mono).
        let hdr = parse_frame_header(&[0xFF, 0xFB, 0x94, 0xC0]).unwrap();
        assert_eq!(hdr.side_info_bytes(), 17);

        let zeros = vec![0u8; 17];
        let si = SideInfo::parse_mpeg1(&hdr, &zeros).unwrap();
        assert_eq!(si.main_data_begin, 0);
        assert_eq!(si.channels, 1);
    }

    #[test]
    fn rejects_short_buffer() {
        let hdr = parse_frame_header_u32(0xFFFB9000).unwrap();
        let zeros = vec![0u8; 10]; // need 32 for stereo
        let err = SideInfo::parse_mpeg1(&hdr, &zeros).unwrap_err();
        assert!(matches!(err, Error::NeedMore));
    }
}
