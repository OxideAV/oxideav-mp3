//! MPEG-1 Layer III **side-information** parser.
//!
//! This module parses the side-information block that immediately
//! follows the 4-byte frame header (and the optional 16-bit CRC) in an
//! MPEG-1 Layer III audio frame, per ISO/IEC 11172-3:1993 §2.4.1.7
//! (`audio_data()` syntax) with the field semantics of §2.4.2.7.
//!
//! It implements *only* the side-info parse: every field from
//! `main_data_begin` through `count1table_select`. It does **not**
//! decode any main data — there is no scalefactor reader, no Huffman
//! stage, no requantisation, no IMDCT, and no synthesis filterbank
//! here. The [`SideInfo`] this module produces is the input those
//! later stages will consume in a future round.
//!
//! # Scope: MPEG-1 only
//!
//! The §2.4.1.7 layout this module parses is the MPEG-1 form with
//! **two granules** per frame, a 9-bit `main_data_begin`, and a
//! per-channel `scfsi`. ISO/IEC 13818-3:1997 §2.4.1.7 redefines the
//! lower-sampling-frequency (MPEG-2 / MPEG-2.5) side-info to a
//! **single** granule, an 8-bit `main_data_begin`, and no `scfsi`
//! field; that variant is out of scope here and rejected by
//! [`parse_side_info`] with [`SideInfoError::NotMpeg1`].
//!
//! # Byte layout (ISO/IEC 11172-3 §2.4.1.7)
//!
//! The MPEG-1 side-information block is byte-aligned and a fixed size:
//! **17 bytes** for single-channel (mono) mode and **32 bytes** for
//! every other (two-channel) mode. The bit accounting, taken directly
//! from the §2.4.1.7 syntax:
//!
//! | field                       | bits (mono / stereo)        |
//! | --------------------------- | --------------------------- |
//! | `main_data_begin`           | 9                           |
//! | `private_bits`              | 5 (mono) / 3 (stereo)       |
//! | `scfsi[ch][band]`           | nch × 4                     |
//! | per granule × channel (×2·nch): each 59 bits …             |
//! | ` part2_3_length`           | 12                          |
//! | ` big_values`               | 9                           |
//! | ` global_gain`              | 8                           |
//! | ` scalefac_compress`        | 4                           |
//! | ` window_switching_flag`    | 1                           |
//! | ` window branch`            | 22 (either branch)          |
//! | ` preflag`                  | 1                           |
//! | ` scalefac_scale`           | 1                           |
//! | ` count1table_select`       | 1                           |
//!
//! Both branches of `window_switching_flag` occupy exactly 22 bits —
//! `block_type` (2) + `mixed_block_flag` (1) + 2×`table_select` (5) +
//! 3×`subblock_gain` (3) = 22 when set, and 3×`table_select` (5) +
//! `region0_count` (4) + `region1_count` (3) = 22 when clear — so the
//! per-granule-per-channel record is a constant 59 bits regardless of
//! window switching. The totals are therefore `9 + 5 + 1·4 + 2·1·59 =
//! 136` bits = 17 bytes (mono) and `9 + 3 + 2·4 + 2·2·59 = 256` bits
//! = 32 bytes (stereo).

use crate::frame::{ChannelMode, Layer, Mp3FrameHeader, MpegVersion};

/// Number of granules in an MPEG-1 Layer III frame (ISO/IEC 11172-3
/// §2.4.1.7: `for (gr=0; gr<2; gr++)`).
pub const GRANULES: usize = 2;

/// Side-information size in bytes for single-channel (mono) MPEG-1
/// Layer III (ISO/IEC 11172-3 §2.4.1.7).
pub const SIDE_INFO_BYTES_MONO: usize = 17;

/// Side-information size in bytes for two-channel MPEG-1 Layer III
/// (ISO/IEC 11172-3 §2.4.1.7).
pub const SIDE_INFO_BYTES_STEREO: usize = 32;

/// The window type carried by `block_type[gr][ch]` (ISO/IEC 11172-3
/// §2.4.2.7).
///
/// When `window_switching_flag` is `0`, `block_type` is defined to be
/// zero (the normal/long window) and is represented here by
/// [`BlockType::Long`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlockType {
    /// `block_type == 0`: the normal (long) window. The §2.4.2.7
    /// table lists `0` as "reserved" for the *window-switched* branch,
    /// but §2.4.2.7 also states that when `window_switching_flag` is
    /// not set, `block_type` is zero — the ordinary long block.
    Long,
    /// `block_type == 1`: start block (long→short transition window).
    Start,
    /// `block_type == 2`: three short windows.
    Short,
    /// `block_type == 3`: end block (short→long transition window).
    End,
}

impl BlockType {
    /// `true` for [`BlockType::Short`] (the only block type that uses
    /// `subblock_gain` and disables `preflag`, per §2.4.2.7).
    #[must_use]
    pub fn is_short(self) -> bool {
        matches!(self, BlockType::Short)
    }
}

/// Per-granule, per-channel side information (ISO/IEC 11172-3
/// §2.4.1.7, inner `for (gr) for (ch)` body; semantics §2.4.2.7).
///
/// The window-switching branch and the long branch are unified by the
/// `window_switching_flag` discriminator. When `window_switching_flag`
/// is `false`, `block_type` is [`BlockType::Long`], `mixed_block_flag`
/// is `false`, `subblock_gain` is all zero, and `table_select` holds
/// three entries with `region0_count` / `region1_count` populated.
/// When `true`, `table_select` holds two entries (the third is `0`),
/// `subblock_gain` is populated, and `region0_count` /
/// `region1_count` carry the §2.4.2.7 default values implied by the
/// block type (region0_count = 7 for type 1/3 and for type 2 with
/// `mixed_block_flag`; 8 for type 2 without; region1_count = 63).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GranuleChannel {
    /// `part2_3_length` (12 bits): number of main-data bits used for
    /// the scalefactors plus the Huffman-coded data of this granule.
    pub part2_3_length: u16,
    /// `big_values` (9 bits): the number of value *pairs* in the
    /// big-values partition (constrained so `big_values*2 <= 576`).
    pub big_values: u16,
    /// `global_gain` (8 bits): the logarithmically quantized global
    /// quantizer step size.
    pub global_gain: u8,
    /// `scalefac_compress` (4 bits): selects the `slen1`/`slen2`
    /// scalefactor bit widths via the §2.4.2.7 table.
    pub scalefac_compress: u8,
    /// `window_switching_flag` (1 bit): when set, the block uses a
    /// window other than the normal (long) type.
    pub window_switching_flag: bool,
    /// `block_type` (2 bits when window-switched; defined as `0`
    /// otherwise → [`BlockType::Long`]).
    pub block_type: BlockType,
    /// `mixed_block_flag` (1 bit, window-switched branch only): when
    /// set, the two lowest subbands use the long window while the rest
    /// use `block_type`.
    pub mixed_block_flag: bool,
    /// `table_select[region]` (5 bits each): Huffman table index per
    /// big-values region. Three entries in the long branch; two in the
    /// window-switched branch (index 2 is left `0`).
    pub table_select: [u8; 3],
    /// `subblock_gain[window]` (3 bits each, window-switched branch
    /// only): per-short-window gain offset from `global_gain`. All
    /// zero in the long branch.
    pub subblock_gain: [u8; 3],
    /// `region0_count` (4 bits in the long branch; a §2.4.2.7 default
    /// in the window-switched branch).
    pub region0_count: u8,
    /// `region1_count` (3 bits in the long branch; `63` by default in
    /// the window-switched branch).
    pub region1_count: u8,
    /// `preflag` (1 bit): high-frequency amplification shortcut. Never
    /// set for short blocks (§2.4.2.7).
    pub preflag: bool,
    /// `scalefac_scale` (1 bit): selects the scalefactor logarithmic
    /// step (0 → multiplier 0.5, 1 → multiplier 1).
    pub scalefac_scale: bool,
    /// `count1table_select` (1 bit): selects Huffman table A (`0`) or
    /// B (`1`) for the count1 (quadruple) region.
    pub count1table_select: bool,
}

/// A fully parsed MPEG-1 Layer III side-information block (ISO/IEC
/// 11172-3 §2.4.1.7).
///
/// Indexed `[granule][channel]`. For mono there is one channel; for
/// every other mode there are two.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SideInfo {
    /// `main_data_begin` (9 bits): the negative byte offset, from the
    /// first byte of this frame's sync word, at which this frame's
    /// main data begins (the "bit reservoir" back-pointer). `0` means
    /// the main data starts immediately after the side information
    /// (§2.4.2.7).
    pub main_data_begin: u16,
    /// `private_bits`: 5 bits in single-channel mode, 3 bits
    /// otherwise. Reserved for private use; preserved verbatim in the
    /// low bits of this field (§2.4.2.7).
    pub private_bits: u8,
    /// `scfsi[ch][scfsi_band]` (1 bit each, 4 bands per channel):
    /// scalefactor selection information. `scfsi[ch][band] == true`
    /// means the granule-0 scalefactors of that band are reused for
    /// granule 1 (§2.4.2.7). Indexed `[channel][band]`; the second
    /// channel's entries are all `false` in mono.
    pub scfsi: [[bool; 4]; 2],
    /// Per-granule, per-channel records, indexed `[granule][channel]`.
    /// The second channel of each granule is a default-filled record
    /// in mono mode; consult [`SideInfo::channels`] for the live
    /// channel count.
    pub granules: [[GranuleChannel; 2]; GRANULES],
    /// Number of channels actually present (1 for mono, 2 otherwise).
    pub channels: u8,
}

impl SideInfo {
    /// The side-information block size in bytes for this channel count
    /// (17 for mono, 32 for stereo), per ISO/IEC 11172-3 §2.4.1.7.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        if self.channels == 1 {
            SIDE_INFO_BYTES_MONO
        } else {
            SIDE_INFO_BYTES_STEREO
        }
    }
}

/// Errors returned by [`parse_side_info`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SideInfoError {
    /// The supplied slice was shorter than the side-info block this
    /// header implies (17 mono / 32 stereo bytes).
    TooShort,
    /// The header is not MPEG-1 Layer III. This parser implements only
    /// the §2.4.1.7 two-granule MPEG-1 layout; MPEG-2 / MPEG-2.5
    /// lower-sampling-frequency side-info (ISO/IEC 13818-3, one
    /// granule, 8-bit `main_data_begin`, no `scfsi`) is out of scope.
    NotMpeg1,
}

impl core::fmt::Display for SideInfoError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let msg = match self {
            SideInfoError::TooShort => "side-info slice shorter than the header implies",
            SideInfoError::NotMpeg1 => "side-info parser handles only MPEG-1 Layer III",
        };
        f.write_str(msg)
    }
}

impl std::error::Error for SideInfoError {}

/// A big-endian, MSB-first bit reader over a byte slice.
///
/// The MPEG audio bitstream is read most-significant-bit-first
/// (ISO/IEC 11172-3 §2.4.1, "the most significant bit … is
/// transmitted first"). This reader walks `bytes` one bit at a time
/// from bit 7 of byte 0. All reads in this module stay within the
/// side-info block whose length the caller has already validated, so
/// the reader saturates (returns zero bits) rather than panicking past
/// the end — a defensive choice; the parse never relies on it for a
/// correctly sized input.
struct BitReader<'a> {
    bytes: &'a [u8],
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        BitReader { bytes, bit_pos: 0 }
    }

    /// Read `n` bits (0 ≤ `n` ≤ 32) MSB-first as an unsigned integer.
    fn read(&mut self, n: u32) -> u32 {
        let mut value: u32 = 0;
        for _ in 0..n {
            let byte_idx = self.bit_pos >> 3;
            let bit = if byte_idx < self.bytes.len() {
                let shift = 7 - (self.bit_pos & 7);
                (self.bytes[byte_idx] >> shift) & 1
            } else {
                0
            };
            value = (value << 1) | u32::from(bit);
            self.bit_pos += 1;
        }
        value
    }

    /// Read a single bit as a `bool`.
    fn read_bool(&mut self) -> bool {
        self.read(1) == 1
    }
}

/// Map the 2-bit `block_type` field to a [`BlockType`].
fn block_type_from_bits(bits: u32) -> BlockType {
    match bits {
        1 => BlockType::Start,
        2 => BlockType::Short,
        3 => BlockType::End,
        // 0 is the normal/long window (and the only value used when
        // window_switching_flag is clear).
        _ => BlockType::Long,
    }
}

/// Parse the MPEG-1 Layer III side-information block that follows the
/// frame header (and its optional CRC) for the frame described by
/// `header`.
///
/// `data` must point at the first byte of the side-information block —
/// i.e. the caller has already consumed the 4 header bytes and, when
/// [`Mp3FrameHeader::crc_protected`] is set, the 2 CRC bytes. This
/// function reads exactly [`SideInfo::byte_len`] bytes (17 mono / 32
/// stereo) per ISO/IEC 11172-3 §2.4.1.7.
///
/// # Errors
///
/// * [`SideInfoError::NotMpeg1`] if `header` is not MPEG-1 Layer III.
/// * [`SideInfoError::TooShort`] if `data` is shorter than the
///   side-info block the header implies.
pub fn parse_side_info(header: &Mp3FrameHeader, data: &[u8]) -> Result<SideInfo, SideInfoError> {
    if header.version != MpegVersion::Mpeg1 || header.layer != Layer::LayerIII {
        return Err(SideInfoError::NotMpeg1);
    }

    let mono = header.mode == ChannelMode::SingleChannel;
    let nch: usize = if mono { 1 } else { 2 };
    let needed = if mono {
        SIDE_INFO_BYTES_MONO
    } else {
        SIDE_INFO_BYTES_STEREO
    };
    if data.len() < needed {
        return Err(SideInfoError::TooShort);
    }

    let mut r = BitReader::new(&data[..needed]);

    // main_data_begin (9 bits).
    let main_data_begin = r.read(9) as u16;

    // private_bits: 5 bits for single_channel, 3 bits otherwise
    // (ISO/IEC 11172-3 §2.4.1.7).
    let private_bits = if mono {
        r.read(5) as u8
    } else {
        r.read(3) as u8
    };

    // scfsi[ch][scfsi_band]: 4 one-bit flags per channel.
    let mut scfsi = [[false; 4]; 2];
    for ch in scfsi.iter_mut().take(nch) {
        for band in ch.iter_mut() {
            *band = r.read_bool();
        }
    }

    // Per-granule, per-channel records. Default-filled so the unused
    // mono channel is a deterministic zero record.
    let default_gc = GranuleChannel {
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
    let mut granules = [[default_gc; 2]; GRANULES];

    for granule in &mut granules {
        for chan in granule.iter_mut().take(nch) {
            *chan = read_granule_channel(&mut r);
        }
    }

    Ok(SideInfo {
        main_data_begin,
        private_bits,
        scfsi,
        granules,
        channels: nch as u8,
    })
}

/// Read one `for (gr) for (ch)` body of §2.4.1.7.
fn read_granule_channel(r: &mut BitReader<'_>) -> GranuleChannel {
    let part2_3_length = r.read(12) as u16;
    let big_values = r.read(9) as u16;
    let global_gain = r.read(8) as u8;
    let scalefac_compress = r.read(4) as u8;
    let window_switching_flag = r.read_bool();

    let mut table_select = [0u8; 3];
    let mut subblock_gain = [0u8; 3];
    let block_type;
    let mixed_block_flag;
    let region0_count;
    let region1_count;

    if window_switching_flag {
        block_type = block_type_from_bits(r.read(2));
        mixed_block_flag = r.read_bool();
        // Two table_select entries; the third region is absent.
        for ts in table_select.iter_mut().take(2) {
            *ts = r.read(5) as u8;
        }
        // Three subblock_gain entries (one per short window).
        for sg in subblock_gain.iter_mut() {
            *sg = r.read(3) as u8;
        }
        // §2.4.2.7 default region partitioning for window-switched
        // blocks: region0_count is 7 for start/end blocks and for
        // short blocks with mixed_block_flag; 8 for short blocks
        // without it. region1_count is 63 (all remaining big-values
        // in region 1).
        region0_count = if block_type == BlockType::Short && !mixed_block_flag {
            8
        } else {
            7
        };
        region1_count = 63;
    } else {
        // Normal (long) window: block_type is defined to be zero.
        block_type = BlockType::Long;
        mixed_block_flag = false;
        for ts in table_select.iter_mut() {
            *ts = r.read(5) as u8;
        }
        region0_count = r.read(4) as u8;
        region1_count = r.read(3) as u8;
    }

    let preflag = r.read_bool();
    let scalefac_scale = r.read_bool();
    let count1table_select = r.read_bool();

    GranuleChannel {
        part2_3_length,
        big_values,
        global_gain,
        scalefac_compress,
        window_switching_flag,
        block_type,
        mixed_block_flag,
        table_select,
        subblock_gain,
        region0_count,
        region1_count,
        preflag,
        scalefac_scale,
        count1table_select,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::parse_header;

    /// Assemble a 4-byte MPEG-1 Layer III header for the given channel
    /// mode and CRC flag, mirroring ISO/IEC 11172-3 §2.4.1.3 bit
    /// positions. bitrate_index 9 (128 kbps), sample-rate index 0
    /// (44.1 kHz).
    fn mpeg1_l3_header(mode: u32, protection: u32) -> [u8; 4] {
        // Fields whose value is zero (padding, private_bit,
        // mode_extension, copyright, emphasis) are simply omitted from
        // the OR chain — their contribution is nil.
        let raw: u32 = (0xFFF << 20)  // syncword
            | (1 << 19)               // ID = MPEG-1
            | (0b01 << 17)            // layer III
            | (protection << 16)      // protection_bit
            | (0b1001 << 12)          // bitrate_index = 128 kbps
            // sampling_frequency = 44.1 kHz ('00')
            | (mode << 6)             // mode
            | (1 << 2); // original
        raw.to_be_bytes()
    }

    /// A minimal MSB-first bit *writer* so the tests construct
    /// side-info blocks straight from the §2.4.1.7 field layout rather
    /// than from any pre-baked byte pattern lifted from an external
    /// source.
    struct BitWriter {
        bits: Vec<u8>,
    }

    impl BitWriter {
        fn new() -> Self {
            BitWriter { bits: Vec::new() }
        }
        fn put(&mut self, value: u32, n: u32) {
            for i in (0..n).rev() {
                self.bits.push(((value >> i) & 1) as u8);
            }
        }
        fn put_bool(&mut self, b: bool) {
            self.bits.push(u8::from(b));
        }
        /// Pad to a byte boundary with zeros and pack to bytes.
        fn finish(mut self) -> Vec<u8> {
            while self.bits.len() % 8 != 0 {
                self.bits.push(0);
            }
            self.bits
                .chunks(8)
                .map(|c| c.iter().fold(0u8, |acc, &b| (acc << 1) | b))
                .collect()
        }
    }

    /// Append a long-block (window_switching_flag == 0)
    /// granule-channel record with the supplied field values.
    #[allow(clippy::too_many_arguments)]
    fn put_long_granule(
        w: &mut BitWriter,
        part2_3_length: u32,
        big_values: u32,
        global_gain: u32,
        scalefac_compress: u32,
        table_select: [u32; 3],
        region0_count: u32,
        region1_count: u32,
        preflag: bool,
        scalefac_scale: bool,
        count1table_select: bool,
    ) {
        w.put(part2_3_length, 12);
        w.put(big_values, 9);
        w.put(global_gain, 8);
        w.put(scalefac_compress, 4);
        w.put_bool(false); // window_switching_flag = 0
        for ts in table_select {
            w.put(ts, 5);
        }
        w.put(region0_count, 4);
        w.put(region1_count, 3);
        w.put_bool(preflag);
        w.put_bool(scalefac_scale);
        w.put_bool(count1table_select);
    }

    /// Append a window-switched (window_switching_flag == 1)
    /// granule-channel record.
    #[allow(clippy::too_many_arguments)]
    fn put_switched_granule(
        w: &mut BitWriter,
        part2_3_length: u32,
        big_values: u32,
        global_gain: u32,
        scalefac_compress: u32,
        block_type: u32,
        mixed_block_flag: bool,
        table_select: [u32; 2],
        subblock_gain: [u32; 3],
        preflag: bool,
        scalefac_scale: bool,
        count1table_select: bool,
    ) {
        w.put(part2_3_length, 12);
        w.put(big_values, 9);
        w.put(global_gain, 8);
        w.put(scalefac_compress, 4);
        w.put_bool(true); // window_switching_flag = 1
        w.put(block_type, 2);
        w.put_bool(mixed_block_flag);
        for ts in table_select {
            w.put(ts, 5);
        }
        for sg in subblock_gain {
            w.put(sg, 3);
        }
        w.put_bool(preflag);
        w.put_bool(scalefac_scale);
        w.put_bool(count1table_select);
    }

    #[test]
    fn mono_long_block_roundtrip() {
        let hdr = parse_header(&mpeg1_l3_header(0b11, 1)).unwrap(); // single_channel
        assert_eq!(hdr.channel_count(), 1);

        let mut w = BitWriter::new();
        w.put(123, 9); // main_data_begin
        w.put(0b10101, 5); // private_bits (5 bits in mono)
                           // scfsi[0][0..4]
        w.put_bool(true);
        w.put_bool(false);
        w.put_bool(true);
        w.put_bool(false);
        // granule 0 channel 0
        put_long_granule(
            &mut w,
            400,
            150,
            210,
            9,
            [10, 20, 30],
            7,
            3,
            true,
            false,
            true,
        );
        // granule 1 channel 0
        put_long_granule(
            &mut w,
            350,
            120,
            205,
            5,
            [1, 2, 3],
            4,
            2,
            false,
            true,
            false,
        );
        let bytes = w.finish();
        assert_eq!(bytes.len(), SIDE_INFO_BYTES_MONO);

        let si = parse_side_info(&hdr, &bytes).unwrap();
        assert_eq!(si.channels, 1);
        assert_eq!(si.byte_len(), 17);
        assert_eq!(si.main_data_begin, 123);
        assert_eq!(si.private_bits, 0b10101);
        assert_eq!(si.scfsi[0], [true, false, true, false]);
        assert_eq!(si.scfsi[1], [false; 4]); // unused channel

        let g0 = &si.granules[0][0];
        assert_eq!(g0.part2_3_length, 400);
        assert_eq!(g0.big_values, 150);
        assert_eq!(g0.global_gain, 210);
        assert_eq!(g0.scalefac_compress, 9);
        assert!(!g0.window_switching_flag);
        assert_eq!(g0.block_type, BlockType::Long);
        assert!(!g0.mixed_block_flag);
        assert_eq!(g0.table_select, [10, 20, 30]);
        assert_eq!(g0.subblock_gain, [0, 0, 0]);
        assert_eq!(g0.region0_count, 7);
        assert_eq!(g0.region1_count, 3);
        assert!(g0.preflag);
        assert!(!g0.scalefac_scale);
        assert!(g0.count1table_select);

        let g1 = &si.granules[1][0];
        assert_eq!(g1.part2_3_length, 350);
        assert_eq!(g1.big_values, 120);
        assert_eq!(g1.global_gain, 205);
        assert_eq!(g1.scalefac_compress, 5);
        assert_eq!(g1.table_select, [1, 2, 3]);
        assert_eq!(g1.region0_count, 4);
        assert_eq!(g1.region1_count, 2);
        assert!(!g1.preflag);
        assert!(g1.scalefac_scale);
        assert!(!g1.count1table_select);
    }

    #[test]
    fn stereo_long_block_all_channels() {
        let hdr = parse_header(&mpeg1_l3_header(0b00, 1)).unwrap(); // stereo
        assert_eq!(hdr.channel_count(), 2);

        let mut w = BitWriter::new();
        w.put(0x1FF, 9); // main_data_begin = 511 (max 9-bit)
        w.put(0b101, 3); // private_bits (3 bits in stereo)
                         // scfsi[ch][band] for two channels
        for band in 0..4 {
            w.put_bool(band % 2 == 0); // ch0: T,F,T,F
        }
        for band in 0..4 {
            w.put_bool(band % 2 == 1); // ch1: F,T,F,T
        }
        // 2 granules × 2 channels, long blocks with distinct values.
        let mut tag = 0u32;
        for _gr in 0..2 {
            for _ch in 0..2 {
                put_long_granule(
                    &mut w,
                    100 + tag,
                    10 + tag,
                    50 + tag,
                    tag % 16,
                    [tag % 32, (tag + 1) % 32, (tag + 2) % 32],
                    tag % 16,
                    tag % 8,
                    tag % 2 == 0,
                    tag % 3 == 0,
                    tag % 5 == 0,
                );
                tag += 1;
            }
        }
        let bytes = w.finish();
        assert_eq!(bytes.len(), SIDE_INFO_BYTES_STEREO);

        let si = parse_side_info(&hdr, &bytes).unwrap();
        assert_eq!(si.channels, 2);
        assert_eq!(si.byte_len(), 32);
        assert_eq!(si.main_data_begin, 511);
        assert_eq!(si.private_bits, 0b101);
        assert_eq!(si.scfsi[0], [true, false, true, false]);
        assert_eq!(si.scfsi[1], [false, true, false, true]);

        let mut tag = 0u32;
        for gr in 0..2 {
            for ch in 0..2 {
                let g = &si.granules[gr][ch];
                assert_eq!(g.part2_3_length, (100 + tag) as u16, "p23 gr{gr} ch{ch}");
                assert_eq!(g.big_values, (10 + tag) as u16);
                assert_eq!(g.global_gain, (50 + tag) as u8);
                assert_eq!(g.scalefac_compress, (tag % 16) as u8);
                assert_eq!(
                    g.table_select,
                    [
                        (tag % 32) as u8,
                        ((tag + 1) % 32) as u8,
                        ((tag + 2) % 32) as u8
                    ]
                );
                assert_eq!(g.region0_count, (tag % 16) as u8);
                assert_eq!(g.region1_count, (tag % 8) as u8);
                assert_eq!(g.preflag, tag % 2 == 0);
                assert_eq!(g.scalefac_scale, tag % 3 == 0);
                assert_eq!(g.count1table_select, tag % 5 == 0);
                assert!(!g.window_switching_flag);
                tag += 1;
            }
        }
    }

    #[test]
    fn window_switched_short_block_no_mixed() {
        let hdr = parse_header(&mpeg1_l3_header(0b11, 1)).unwrap(); // mono

        let mut w = BitWriter::new();
        w.put(0, 9); // main_data_begin
        w.put(0, 5); // private_bits
        for _ in 0..4 {
            w.put_bool(false); // scfsi all 0 (required for short blocks)
        }
        // granule 0: short block (block_type=2), mixed=0.
        put_switched_granule(
            &mut w,
            500,
            0, // big_values often 0 for short, but value is arbitrary here
            190,
            12,
            2,     // block_type = short
            false, // mixed_block_flag
            [15, 7],
            [1, 2, 3],
            false, // preflag never set for short, but bit is still read
            true,
            false,
        );
        // granule 1: start block (block_type=1).
        put_switched_granule(
            &mut w,
            480,
            5,
            188,
            3,
            1, // block_type = start
            false,
            [8, 9],
            [4, 5, 6],
            false,
            false,
            true,
        );
        let bytes = w.finish();
        assert_eq!(bytes.len(), SIDE_INFO_BYTES_MONO);

        let si = parse_side_info(&hdr, &bytes).unwrap();
        let g0 = &si.granules[0][0];
        assert!(g0.window_switching_flag);
        assert_eq!(g0.block_type, BlockType::Short);
        assert!(!g0.mixed_block_flag);
        assert_eq!(g0.table_select, [15, 7, 0]); // third entry absent
        assert_eq!(g0.subblock_gain, [1, 2, 3]);
        // §2.4.2.7 default: short + !mixed => region0_count = 8.
        assert_eq!(g0.region0_count, 8);
        assert_eq!(g0.region1_count, 63);
        assert_eq!(g0.scalefac_compress, 12);
        assert!(g0.scalefac_scale);
        assert!(!g0.count1table_select);

        let g1 = &si.granules[1][0];
        assert!(g1.window_switching_flag);
        assert_eq!(g1.block_type, BlockType::Start);
        assert_eq!(g1.subblock_gain, [4, 5, 6]);
        // start block => region0_count = 7.
        assert_eq!(g1.region0_count, 7);
        assert_eq!(g1.region1_count, 63);
        assert!(g1.count1table_select);
    }

    #[test]
    fn window_switched_short_block_mixed() {
        let hdr = parse_header(&mpeg1_l3_header(0b11, 1)).unwrap(); // mono

        let mut w = BitWriter::new();
        w.put(7, 9);
        w.put(0, 5);
        for _ in 0..4 {
            w.put_bool(false);
        }
        // Both granules short + mixed.
        put_switched_granule(
            &mut w,
            300,
            12,
            200,
            0,
            2,
            true,
            [3, 4],
            [7, 0, 1],
            false,
            false,
            false,
        );
        put_switched_granule(
            &mut w,
            290,
            8,
            198,
            1,
            2,
            true,
            [5, 6],
            [0, 2, 3],
            false,
            true,
            true,
        );
        let bytes = w.finish();

        let si = parse_side_info(&hdr, &bytes).unwrap();
        let g0 = &si.granules[0][0];
        assert!(g0.window_switching_flag);
        assert_eq!(g0.block_type, BlockType::Short);
        assert!(g0.mixed_block_flag);
        // §2.4.2.7 default: short + mixed => region0_count = 7.
        assert_eq!(g0.region0_count, 7);
        assert_eq!(g0.region1_count, 63);
        assert_eq!(g0.subblock_gain, [7, 0, 1]);
    }

    #[test]
    fn all_block_types_decode() {
        let hdr = parse_header(&mpeg1_l3_header(0b11, 1)).unwrap();
        for (bits, expect) in [
            (1u32, BlockType::Start),
            (2, BlockType::Short),
            (3, BlockType::End),
        ] {
            let mut w = BitWriter::new();
            w.put(0, 9);
            w.put(0, 5);
            for _ in 0..4 {
                w.put_bool(false);
            }
            put_switched_granule(
                &mut w,
                0,
                0,
                0,
                0,
                bits,
                false,
                [0, 0],
                [0, 0, 0],
                false,
                false,
                false,
            );
            put_switched_granule(
                &mut w,
                0,
                0,
                0,
                0,
                bits,
                false,
                [0, 0],
                [0, 0, 0],
                false,
                false,
                false,
            );
            let bytes = w.finish();
            let si = parse_side_info(&hdr, &bytes).unwrap();
            assert_eq!(si.granules[0][0].block_type, expect);
        }
    }

    #[test]
    fn rejects_too_short() {
        let hdr = parse_header(&mpeg1_l3_header(0b00, 1)).unwrap(); // stereo => 32 bytes
        let buf = vec![0u8; SIDE_INFO_BYTES_STEREO - 1];
        assert_eq!(parse_side_info(&hdr, &buf), Err(SideInfoError::TooShort));
        // 31 bytes is enough for mono but not stereo.
        let mono = parse_header(&mpeg1_l3_header(0b11, 1)).unwrap();
        let buf17 = vec![0u8; SIDE_INFO_BYTES_MONO];
        assert!(parse_side_info(&mono, &buf17).is_ok());
        let buf16 = vec![0u8; SIDE_INFO_BYTES_MONO - 1];
        assert_eq!(parse_side_info(&mono, &buf16), Err(SideInfoError::TooShort));
    }

    #[test]
    fn rejects_non_mpeg1() {
        // MPEG-2 LSF Layer III header (ID=0): out of scope here.
        // ID bit, sampling_frequency, and mode are all zero, so they
        // are omitted from the OR chain.
        let raw: u32 = (0xFFF << 20)
            | (0b01 << 17) // layer III
            | (1 << 16)
            | (0b1000 << 12); // bitrate_index
        let bytes = raw.to_be_bytes();
        let hdr = parse_header(&bytes).unwrap();
        assert_eq!(hdr.version, MpegVersion::Mpeg2);
        let buf = vec![0u8; 32];
        assert_eq!(parse_side_info(&hdr, &buf), Err(SideInfoError::NotMpeg1));
    }

    #[test]
    fn rejects_layer1_and_layer2() {
        // Layer I and Layer II MPEG-1 headers must be rejected: this
        // parser is Layer III only.
        for layer in [0b11u32 /* I */, 0b10 /* II */] {
            // sampling_frequency '00' is omitted from the OR chain.
            let raw: u32 = (0xFFF << 20)
                | (1 << 19) // MPEG-1
                | (layer << 17)
                | (1 << 16)
                | (0b1000 << 12); // bitrate_index
            let bytes = raw.to_be_bytes();
            let hdr = parse_header(&bytes).unwrap();
            let buf = vec![0u8; 32];
            assert_eq!(parse_side_info(&hdr, &buf), Err(SideInfoError::NotMpeg1));
        }
    }

    #[test]
    fn extra_trailing_bytes_are_ignored() {
        // A buffer longer than the side-info block parses identically
        // (the parser reads exactly byte_len bytes).
        let hdr = parse_header(&mpeg1_l3_header(0b11, 1)).unwrap();
        let mut w = BitWriter::new();
        w.put(42, 9);
        w.put(0b11111, 5);
        for _ in 0..4 {
            w.put_bool(false);
        }
        put_long_granule(&mut w, 1, 1, 1, 1, [1, 1, 1], 1, 1, false, false, false);
        put_long_granule(&mut w, 2, 2, 2, 2, [2, 2, 2], 2, 2, false, false, false);
        let mut bytes = w.finish();
        let exact = parse_side_info(&hdr, &bytes).unwrap();
        // Append main-data garbage; parse must be unaffected.
        bytes.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);
        let with_tail = parse_side_info(&hdr, &bytes).unwrap();
        assert_eq!(exact, with_tail);
        assert_eq!(exact.main_data_begin, 42);
        assert_eq!(exact.private_bits, 0b11111);
    }

    #[test]
    fn max_field_values_pack_and_unpack() {
        // Saturate every field to its maximum to catch bit-width
        // mistakes (a too-wide read would steal a neighbour's bits).
        let hdr = parse_header(&mpeg1_l3_header(0b11, 1)).unwrap();
        let mut w = BitWriter::new();
        w.put(0x1FF, 9); // main_data_begin max
        w.put(0x1F, 5); // private_bits max (5 bits)
        for _ in 0..4 {
            w.put_bool(true);
        }
        put_long_granule(
            &mut w,
            0xFFF,              // part2_3_length max (12 bits)
            0x1FF,              // big_values max (9 bits)
            0xFF,               // global_gain max (8 bits)
            0xF,                // scalefac_compress max (4 bits)
            [0x1F, 0x1F, 0x1F], // table_select max (5 bits)
            0xF,                // region0_count max (4 bits)
            0x7,                // region1_count max (3 bits)
            true,
            true,
            true,
        );
        put_long_granule(&mut w, 0, 0, 0, 0, [0, 0, 0], 0, 0, false, false, false);
        let bytes = w.finish();
        let si = parse_side_info(&hdr, &bytes).unwrap();
        assert_eq!(si.main_data_begin, 0x1FF);
        assert_eq!(si.private_bits, 0x1F);
        assert_eq!(si.scfsi[0], [true; 4]);
        let g = &si.granules[0][0];
        assert_eq!(g.part2_3_length, 0xFFF);
        assert_eq!(g.big_values, 0x1FF);
        assert_eq!(g.global_gain, 0xFF);
        assert_eq!(g.scalefac_compress, 0xF);
        assert_eq!(g.table_select, [0x1F, 0x1F, 0x1F]);
        assert_eq!(g.region0_count, 0xF);
        assert_eq!(g.region1_count, 0x7);
        // Second granule stayed at zero — no bit bleed across records.
        assert_eq!(si.granules[1][0].part2_3_length, 0);
        assert_eq!(si.granules[1][0].global_gain, 0);
    }
}
