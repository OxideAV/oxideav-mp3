//! MPEG-1 and MPEG-2 (LSF) Layer III **side-information** parser.
//!
//! This module parses the side-information block that immediately
//! follows the 4-byte frame header (and the optional 16-bit CRC) in a
//! Layer III audio frame, for both the MPEG-1 layout of ISO/IEC
//! 11172-3:1993 §2.4.1.7 (`audio_data()` syntax, semantics §2.4.2.7)
//! and the lower-sampling-frequency layout of ISO/IEC 13818-3:1997
//! §2.4.1.7 (semantics §2.4.2.7).
//!
//! It implements *only* the side-info parse: every field from
//! `main_data_begin` through `count1table_select`. It does **not**
//! decode any main data — there is no scalefactor reader, no Huffman
//! stage, no requantisation, no IMDCT, and no synthesis filterbank
//! here. The [`SideInfo`] this module produces is the input those
//! later stages will consume in a future round.
//!
//! # Two layouts: MPEG-1 vs MPEG-2 LSF
//!
//! [`parse_side_info`] dispatches on the header's [`MpegVersion`]:
//!
//! * **MPEG-1** ([`MpegVersion::Mpeg1`]): the §2.4.1.7 form of ISO/IEC
//!   11172-3 with **two granules** per frame, a 9-bit
//!   `main_data_begin`, a 4-bit `scalefac_compress`, a per-channel
//!   `scfsi`, and a transmitted `preflag` bit.
//! * **MPEG-2 / MPEG-2.5 LSF** ([`MpegVersion::Mpeg2`]): the §2.4.1.7
//!   form of ISO/IEC 13818-3, with a **single** granule, an 8-bit
//!   `main_data_begin`, a 9-bit `scalefac_compress`, **no** `scfsi`
//!   field, and **no** transmitted `preflag` (per §2.4.2.7 the LSF
//!   `preflag` is derived from `scalefac_compress` at
//!   scalefactor-decode time, not carried in the side info; this parser
//!   leaves [`GranuleChannel::preflag`] `false` for LSF).
//!
//! # MPEG-1 byte layout (ISO/IEC 11172-3 §2.4.1.7)
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
//!
//! # MPEG-2 LSF byte layout (ISO/IEC 13818-3 §2.4.1.7)
//!
//! The LSF block drops `scfsi` and the second granule, narrows
//! `main_data_begin` to 8 bits, and widens `scalefac_compress` to 9
//! bits while dropping the transmitted `preflag`. The window branch is
//! unchanged (22 bits). So the per-channel record is `12 + 9 + 8 + 9 +
//! 1 + 22 + 1 + 1 = 63` bits, and the totals are:
//!
//! | field                       | bits (mono / stereo)        |
//! | --------------------------- | --------------------------- |
//! | `main_data_begin`           | 8                           |
//! | `private_bits`              | 1 (mono) / 2 (stereo)       |
//! | per channel (×nch): each 63 bits …                        |
//! | ` part2_3_length`           | 12                          |
//! | ` big_values`               | 9                           |
//! | ` global_gain`              | 8                           |
//! | ` scalefac_compress`        | 9                           |
//! | ` window_switching_flag`    | 1                           |
//! | ` window branch`            | 22 (either branch)          |
//! | ` scalefac_scale`           | 1                           |
//! | ` count1table_select`       | 1                           |
//!
//! `8 + 1 + 1·63 = 72` bits = 9 bytes (mono) and `8 + 2 + 2·63 = 136`
//! bits = 17 bytes (stereo). These match the §2.4.2.4 LSF CRC bit
//! ranges, which cover "bits 0..71 of audio_data for single channel
//! mode, bits 0..135 of audio_data for other modes".

use crate::frame::{ChannelMode, Layer, Mp3FrameHeader, MpegVersion};

/// Number of granules in an MPEG-1 Layer III frame (ISO/IEC 11172-3
/// §2.4.1.7: `for (gr=0; gr<2; gr++)`).
pub const GRANULES: usize = 2;

/// Number of granules in an MPEG-2 / MPEG-2.5 LSF Layer III frame
/// (ISO/IEC 13818-3 §2.4.1.7 / §2.4.3.2: "a Layer III frame contains
/// only one granule … the variable 'gr' does not exist any more").
pub const GRANULES_LSF: usize = 1;

/// Side-information size in bytes for single-channel (mono) MPEG-1
/// Layer III (ISO/IEC 11172-3 §2.4.1.7).
pub const SIDE_INFO_BYTES_MONO: usize = 17;

/// Side-information size in bytes for two-channel MPEG-1 Layer III
/// (ISO/IEC 11172-3 §2.4.1.7).
pub const SIDE_INFO_BYTES_STEREO: usize = 32;

/// Side-information size in bytes for single-channel (mono) MPEG-2 /
/// MPEG-2.5 LSF Layer III (ISO/IEC 13818-3 §2.4.1.7): `8 + 1 + 63 =
/// 72` bits = 9 bytes (matches the §2.4.2.4 "bits 0..71 of
/// audio_data" CRC range for single-channel mode).
pub const SIDE_INFO_BYTES_LSF_MONO: usize = 9;

/// Side-information size in bytes for two-channel MPEG-2 / MPEG-2.5
/// LSF Layer III (ISO/IEC 13818-3 §2.4.1.7): `8 + 2 + 2·63 = 136`
/// bits = 17 bytes (matches the §2.4.2.4 "bits 0..135 of audio_data"
/// CRC range for other modes).
pub const SIDE_INFO_BYTES_LSF_STEREO: usize = 17;

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
    /// `scalefac_compress`: selects the scalefactor bit widths.
    ///
    /// **MPEG-1** (ISO/IEC 11172-3 §2.4.2.7): 4 bits, indexing the
    /// `slen1`/`slen2` table (values 0..15).
    ///
    /// **MPEG-2 / MPEG-2.5 LSF** (ISO/IEC 13818-3 §2.4.2.7): 9 bits
    /// (values 0..511); the §2.4.2.7 LSF procedure derives `slen1`..
    /// `slen4`, the `nr_of_sfb*` partition sizes, and `preflag` from
    /// this value at scalefactor-decode time (not in scope this round).
    pub scalefac_compress: u16,
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
    /// `preflag`: high-frequency amplification shortcut.
    ///
    /// **MPEG-1** (ISO/IEC 11172-3 §2.4.2.7): a transmitted 1-bit
    /// side-info field; never set for short blocks.
    ///
    /// **MPEG-2 / MPEG-2.5 LSF** (ISO/IEC 13818-3 §2.4.2.7): *not*
    /// transmitted — `preflag` is derived from `scalefac_compress`
    /// during scalefactor decode (set only when `500 <=
    /// scalefac_compress < 512` in the non-intensity case). This parser
    /// leaves it `false` for LSF frames; the eventual scalefactor
    /// stage computes the real value.
    pub preflag: bool,
    /// `scalefac_scale` (1 bit): selects the scalefactor logarithmic
    /// step (0 → multiplier 0.5, 1 → multiplier 1).
    pub scalefac_scale: bool,
    /// `count1table_select` (1 bit): selects Huffman table A (`0`) or
    /// B (`1`) for the count1 (quadruple) region.
    pub count1table_select: bool,
}

/// A fully parsed Layer III side-information block.
///
/// Covers both the MPEG-1 layout (ISO/IEC 11172-3 §2.4.1.7, two
/// granules, `scfsi` present) and the MPEG-2 / MPEG-2.5 LSF layout
/// (ISO/IEC 13818-3 §2.4.1.7, one granule, no `scfsi`). Consult
/// [`SideInfo::granule_count`] and [`SideInfo::channels`] for the live
/// dimensions; cells outside those bounds are deterministic defaults.
///
/// Indexed `[granule][channel]`. For mono there is one channel; for
/// every other mode there are two. For LSF there is one granule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SideInfo {
    /// `main_data_begin`: the negative byte offset, from the first
    /// byte of this frame's sync word, at which this frame's main data
    /// begins (the "bit reservoir" back-pointer). `0` means the main
    /// data starts immediately after the side information (§2.4.2.7).
    /// 9 bits in MPEG-1, 8 bits in MPEG-2 / MPEG-2.5 LSF.
    pub main_data_begin: u16,
    /// `private_bits`: reserved for private use; preserved verbatim in
    /// the low bits of this field (§2.4.2.7). MPEG-1: 5 bits in
    /// single-channel mode, 3 bits otherwise. MPEG-2 / MPEG-2.5 LSF:
    /// 1 bit in single-channel mode, 2 bits otherwise.
    pub private_bits: u8,
    /// `scfsi[ch][scfsi_band]` (1 bit each, 4 bands per channel):
    /// scalefactor selection information. `scfsi[ch][band] == true`
    /// means the granule-0 scalefactors of that band are reused for
    /// granule 1 (§2.4.2.7). Indexed `[channel][band]`; the second
    /// channel's entries are all `false` in mono.
    ///
    /// **MPEG-2 / MPEG-2.5 LSF has no `scfsi` field** (there is only
    /// one granule, so there is nothing to share into a second one);
    /// for LSF frames this is left all-`false`.
    pub scfsi: [[bool; 4]; 2],
    /// Per-granule, per-channel records, indexed `[granule][channel]`.
    /// The second channel of each granule is a default-filled record
    /// in mono mode; the second granule is default-filled for LSF.
    /// Consult [`SideInfo::granule_count`] and [`SideInfo::channels`]
    /// for the live counts.
    pub granules: [[GranuleChannel; 2]; GRANULES],
    /// Number of channels actually present (1 for mono, 2 otherwise).
    pub channels: u8,
    /// Number of granules actually present: 2 for MPEG-1 (ISO/IEC
    /// 11172-3 §2.4.1.7), 1 for MPEG-2 / MPEG-2.5 LSF (ISO/IEC
    /// 13818-3 §2.4.1.7 / §2.4.3.2). Records at `granules[1]` are
    /// default-filled when this is 1.
    pub granule_count: u8,
    /// `true` when this side info was parsed under the ISO/IEC 13818-3
    /// lower-sampling-frequency (MPEG-2 / MPEG-2.5) layout, `false`
    /// for the ISO/IEC 11172-3 (MPEG-1) layout. Disambiguates the
    /// `main_data_begin` / `scalefac_compress` bit widths and tells
    /// the (future) scalefactor stage which §2.4.2.7 procedure applies.
    pub lsf: bool,
}

impl SideInfo {
    /// The side-information block size in bytes for this channel count
    /// and layout: MPEG-1 → 17 mono / 32 stereo (ISO/IEC 11172-3
    /// §2.4.1.7); MPEG-2 / MPEG-2.5 LSF → 9 mono / 17 stereo (ISO/IEC
    /// 13818-3 §2.4.1.7).
    #[must_use]
    pub fn byte_len(&self) -> usize {
        match (self.lsf, self.channels == 1) {
            (false, true) => SIDE_INFO_BYTES_MONO,
            (false, false) => SIDE_INFO_BYTES_STEREO,
            (true, true) => SIDE_INFO_BYTES_LSF_MONO,
            (true, false) => SIDE_INFO_BYTES_LSF_STEREO,
        }
    }
}

/// Errors returned by [`parse_side_info`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SideInfoError {
    /// The supplied slice was shorter than the side-info block this
    /// header implies (MPEG-1: 17 mono / 32 stereo; MPEG-2 / MPEG-2.5
    /// LSF: 9 mono / 17 stereo bytes).
    TooShort,
    /// The header is not Layer III. The side-info layout parsed here is
    /// Layer III only — Layer I and Layer II carry their side data in a
    /// different form (ISO/IEC 11172-3 §2.4.1.5 / §2.4.1.6).
    NotLayer3,
}

impl core::fmt::Display for SideInfoError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let msg = match self {
            SideInfoError::TooShort => "side-info slice shorter than the header implies",
            SideInfoError::NotLayer3 => "side-info parser handles only Layer III",
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

/// The deterministic all-zero / long-block [`GranuleChannel`] record
/// used to fill cells outside the live `[granule][channel]` bounds.
const DEFAULT_GRANULE_CHANNEL: GranuleChannel = GranuleChannel {
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

/// Parse the Layer III side-information block that follows the frame
/// header (and its optional CRC) for the frame described by `header`.
///
/// Dispatches on [`Mp3FrameHeader::version`]: MPEG-1 frames use the
/// ISO/IEC 11172-3 §2.4.1.7 layout (two granules, 9-bit
/// `main_data_begin`, 4-bit `scalefac_compress`, per-channel `scfsi`,
/// transmitted `preflag`); MPEG-2 / MPEG-2.5 LSF frames use the ISO/IEC
/// 13818-3 §2.4.1.7 layout (one granule, 8-bit `main_data_begin`, 9-bit
/// `scalefac_compress`, no `scfsi`, no transmitted `preflag`).
///
/// `data` must point at the first byte of the side-information block —
/// i.e. the caller has already consumed the 4 header bytes and, when
/// [`Mp3FrameHeader::crc_protected`] is set, the 2 CRC bytes. This
/// function reads exactly [`SideInfo::byte_len`] bytes.
///
/// # Errors
///
/// * [`SideInfoError::NotLayer3`] if `header` is not Layer III.
/// * [`SideInfoError::TooShort`] if `data` is shorter than the
///   side-info block the header implies.
pub fn parse_side_info(header: &Mp3FrameHeader, data: &[u8]) -> Result<SideInfo, SideInfoError> {
    if header.layer != Layer::LayerIII {
        return Err(SideInfoError::NotLayer3);
    }

    let mono = header.mode == ChannelMode::SingleChannel;
    let nch: usize = if mono { 1 } else { 2 };

    match header.version {
        MpegVersion::Mpeg1 => parse_mpeg1(data, mono, nch),
        MpegVersion::Mpeg2 => parse_lsf(data, mono, nch),
    }
}

/// Parse the MPEG-1 two-granule side-info layout (ISO/IEC 11172-3
/// §2.4.1.7).
fn parse_mpeg1(data: &[u8], mono: bool, nch: usize) -> Result<SideInfo, SideInfoError> {
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

    // Two granules, each with `nch` channel records.
    let mut granules = [[DEFAULT_GRANULE_CHANNEL; 2]; GRANULES];
    for granule in &mut granules {
        for chan in granule.iter_mut().take(nch) {
            *chan = read_granule_channel(&mut r, false);
        }
    }

    Ok(SideInfo {
        main_data_begin,
        private_bits,
        scfsi,
        granules,
        channels: nch as u8,
        granule_count: GRANULES as u8,
        lsf: false,
    })
}

/// Parse the MPEG-2 / MPEG-2.5 LSF single-granule side-info layout
/// (ISO/IEC 13818-3 §2.4.1.7).
fn parse_lsf(data: &[u8], mono: bool, nch: usize) -> Result<SideInfo, SideInfoError> {
    let needed = if mono {
        SIDE_INFO_BYTES_LSF_MONO
    } else {
        SIDE_INFO_BYTES_LSF_STEREO
    };
    if data.len() < needed {
        return Err(SideInfoError::TooShort);
    }

    let mut r = BitReader::new(&data[..needed]);

    // main_data_begin (8 bits in LSF; ISO/IEC 13818-3 §2.4.1.7).
    let main_data_begin = r.read(8) as u16;

    // private_bits: 1 bit for single_channel, 2 bits otherwise
    // (ISO/IEC 13818-3 §2.4.1.7). There is no scfsi field in LSF.
    let private_bits = if mono {
        r.read(1) as u8
    } else {
        r.read(2) as u8
    };

    // One granule with `nch` channel records.
    let mut granules = [[DEFAULT_GRANULE_CHANNEL; 2]; GRANULES];
    for chan in granules[0].iter_mut().take(nch) {
        *chan = read_granule_channel(&mut r, true);
    }

    Ok(SideInfo {
        main_data_begin,
        private_bits,
        scfsi: [[false; 4]; 2],
        granules,
        channels: nch as u8,
        granule_count: GRANULES_LSF as u8,
        lsf: true,
    })
}

/// Read one per-channel side-info record.
///
/// `lsf == false` reads the ISO/IEC 11172-3 §2.4.1.7 MPEG-1 body
/// (4-bit `scalefac_compress`, a transmitted `preflag` bit). `lsf ==
/// true` reads the ISO/IEC 13818-3 §2.4.1.7 LSF body (9-bit
/// `scalefac_compress`, no transmitted `preflag` — left `false` here;
/// §2.4.2.7 derives it from `scalefac_compress` during scalefactor
/// decode). The window-switching branch is identical in both.
fn read_granule_channel(r: &mut BitReader<'_>, lsf: bool) -> GranuleChannel {
    let part2_3_length = r.read(12) as u16;
    let big_values = r.read(9) as u16;
    let global_gain = r.read(8) as u8;
    // scalefac_compress: 4 bits in MPEG-1, 9 bits in LSF.
    let scalefac_compress = if lsf {
        r.read(9) as u16
    } else {
        r.read(4) as u16
    };
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

    // preflag is a transmitted bit in MPEG-1 only. In LSF it is not
    // carried in the side info (ISO/IEC 13818-3 §2.4.2.7 derives it
    // from scalefac_compress), so no bit is consumed and it stays
    // `false` here.
    let preflag = if lsf { false } else { r.read_bool() };
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

    /// Assemble a 4-byte MPEG-2 (LSF) Layer III header for the given
    /// channel mode and CRC flag, mirroring the ISO/IEC 13818-3
    /// §2.4.2.3 field positions (the bit layout is shared with ISO/IEC
    /// 11172-3 §2.4.1.3; only the `ID` bit differs). bitrate_index 9,
    /// sample-rate index 0 (16 kHz).
    fn lsf_l3_header(mode: u32, protection: u32) -> [u8; 4] {
        // ID = 0 (extension to lower sampling frequencies) so the bit
        // is omitted from the OR chain. sampling_frequency '00' (the
        // first MPEG-2 entry, 16 kHz) and the other zero fields are
        // likewise omitted.
        let raw: u32 = (0xFFF << 20)  // syncword
            | (0b01 << 17)            // layer III
            | (protection << 16)      // protection_bit
            | (0b1001 << 12)          // bitrate_index
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

    /// Append an LSF long-block (window_switching_flag == 0) per-channel
    /// record per ISO/IEC 13818-3 §2.4.1.7: 9-bit `scalefac_compress`,
    /// no transmitted `preflag` bit.
    #[allow(clippy::too_many_arguments)]
    fn put_lsf_long_channel(
        w: &mut BitWriter,
        part2_3_length: u32,
        big_values: u32,
        global_gain: u32,
        scalefac_compress: u32,
        table_select: [u32; 3],
        region0_count: u32,
        region1_count: u32,
        scalefac_scale: bool,
        count1table_select: bool,
    ) {
        w.put(part2_3_length, 12);
        w.put(big_values, 9);
        w.put(global_gain, 8);
        w.put(scalefac_compress, 9); // 9 bits in LSF
        w.put_bool(false); // window_switching_flag = 0
        for ts in table_select {
            w.put(ts, 5);
        }
        w.put(region0_count, 4);
        w.put(region1_count, 3);
        // no preflag bit in LSF
        w.put_bool(scalefac_scale);
        w.put_bool(count1table_select);
    }

    /// Append an LSF window-switched (window_switching_flag == 1)
    /// per-channel record per ISO/IEC 13818-3 §2.4.1.7.
    #[allow(clippy::too_many_arguments)]
    fn put_lsf_switched_channel(
        w: &mut BitWriter,
        part2_3_length: u32,
        big_values: u32,
        global_gain: u32,
        scalefac_compress: u32,
        block_type: u32,
        mixed_block_flag: bool,
        table_select: [u32; 2],
        subblock_gain: [u32; 3],
        scalefac_scale: bool,
        count1table_select: bool,
    ) {
        w.put(part2_3_length, 12);
        w.put(big_values, 9);
        w.put(global_gain, 8);
        w.put(scalefac_compress, 9); // 9 bits in LSF
        w.put_bool(true); // window_switching_flag = 1
        w.put(block_type, 2);
        w.put_bool(mixed_block_flag);
        for ts in table_select {
            w.put(ts, 5);
        }
        for sg in subblock_gain {
            w.put(sg, 3);
        }
        // no preflag bit in LSF
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
                assert_eq!(g.scalefac_compress, (tag % 16) as u16);
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
    fn mpeg1_sets_layout_markers() {
        // An MPEG-1 parse reports the two-granule, non-LSF layout.
        let hdr = parse_header(&mpeg1_l3_header(0b11, 1)).unwrap();
        let buf = vec![0u8; SIDE_INFO_BYTES_MONO];
        let si = parse_side_info(&hdr, &buf).unwrap();
        assert!(!si.lsf);
        assert_eq!(si.granule_count, 2);
        assert_eq!(si.channels, 1);
        assert_eq!(si.byte_len(), SIDE_INFO_BYTES_MONO);
    }

    #[test]
    fn rejects_layer1_and_layer2() {
        // Layer I and Layer II headers must be rejected: this parser
        // handles the Layer III side-info layout only. Checked for both
        // MPEG-1 (ID=1) and MPEG-2 LSF (ID=0) variants.
        for id in [1u32 /* MPEG-1 */, 0 /* MPEG-2 LSF */] {
            for layer in [0b11u32 /* I */, 0b10 /* II */] {
                // sampling_frequency '00' is omitted from the OR chain.
                let raw: u32 =
                    (0xFFF << 20) | (id << 19) | (layer << 17) | (1 << 16) | (0b1000 << 12); // bitrate_index
                let bytes = raw.to_be_bytes();
                let hdr = parse_header(&bytes).unwrap();
                let buf = vec![0u8; 32];
                assert_eq!(parse_side_info(&hdr, &buf), Err(SideInfoError::NotLayer3));
            }
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

    // -- MPEG-2 / MPEG-2.5 LSF side-info (ISO/IEC 13818-3 §2.4.1.7) --

    #[test]
    fn lsf_mono_long_block() {
        let hdr = parse_header(&lsf_l3_header(0b11, 1)).unwrap(); // single_channel
        assert_eq!(hdr.version, MpegVersion::Mpeg2);
        assert_eq!(hdr.channel_count(), 1);

        let mut w = BitWriter::new();
        w.put(200, 8); // main_data_begin (8 bits in LSF)
        w.put_bool(true); // private_bits (1 bit in mono LSF)
                          // no scfsi field in LSF
                          // single granule, channel 0
        put_lsf_long_channel(
            &mut w,
            511,         // part2_3_length
            150,         // big_values
            205,         // global_gain
            411,         // scalefac_compress (9 bits, > 4-bit range)
            [12, 24, 5], // table_select
            6,           // region0_count
            4,           // region1_count
            true,        // scalefac_scale
            false,       // count1table_select
        );
        let bytes = w.finish();
        assert_eq!(bytes.len(), SIDE_INFO_BYTES_LSF_MONO); // 9 bytes

        let si = parse_side_info(&hdr, &bytes).unwrap();
        assert!(si.lsf);
        assert_eq!(si.granule_count, 1);
        assert_eq!(si.channels, 1);
        assert_eq!(si.byte_len(), 9);
        assert_eq!(si.main_data_begin, 200);
        assert_eq!(si.private_bits, 1);
        assert_eq!(si.scfsi, [[false; 4]; 2]); // no scfsi in LSF

        let g = &si.granules[0][0];
        assert_eq!(g.part2_3_length, 511);
        assert_eq!(g.big_values, 150);
        assert_eq!(g.global_gain, 205);
        assert_eq!(g.scalefac_compress, 411);
        assert!(!g.window_switching_flag);
        assert_eq!(g.block_type, BlockType::Long);
        assert_eq!(g.table_select, [12, 24, 5]);
        assert_eq!(g.region0_count, 6);
        assert_eq!(g.region1_count, 4);
        // preflag is never read from LSF side info.
        assert!(!g.preflag);
        assert!(g.scalefac_scale);
        assert!(!g.count1table_select);

        // Second granule is the deterministic default (LSF has one).
        assert_eq!(si.granules[1][0], DEFAULT_GRANULE_CHANNEL);
    }

    #[test]
    fn lsf_stereo_long_block_both_channels() {
        let hdr = parse_header(&lsf_l3_header(0b00, 1)).unwrap(); // stereo
        assert_eq!(hdr.version, MpegVersion::Mpeg2);
        assert_eq!(hdr.channel_count(), 2);

        let mut w = BitWriter::new();
        w.put(0xFF, 8); // main_data_begin = 255 (max 8-bit)
        w.put(0b10, 2); // private_bits (2 bits in stereo LSF)
                        // single granule, two channels, distinct values
        put_lsf_long_channel(&mut w, 100, 20, 60, 300, [1, 2, 3], 5, 2, false, true);
        put_lsf_long_channel(&mut w, 250, 40, 90, 130, [4, 5, 6], 9, 6, true, false);
        let bytes = w.finish();
        assert_eq!(bytes.len(), SIDE_INFO_BYTES_LSF_STEREO); // 17 bytes

        let si = parse_side_info(&hdr, &bytes).unwrap();
        assert!(si.lsf);
        assert_eq!(si.granule_count, 1);
        assert_eq!(si.channels, 2);
        assert_eq!(si.byte_len(), 17);
        assert_eq!(si.main_data_begin, 255);
        assert_eq!(si.private_bits, 0b10);

        let c0 = &si.granules[0][0];
        assert_eq!(c0.part2_3_length, 100);
        assert_eq!(c0.big_values, 20);
        assert_eq!(c0.global_gain, 60);
        assert_eq!(c0.scalefac_compress, 300);
        assert_eq!(c0.table_select, [1, 2, 3]);
        assert_eq!(c0.region0_count, 5);
        assert_eq!(c0.region1_count, 2);
        assert!(!c0.scalefac_scale);
        assert!(c0.count1table_select);

        let c1 = &si.granules[0][1];
        assert_eq!(c1.part2_3_length, 250);
        assert_eq!(c1.big_values, 40);
        assert_eq!(c1.global_gain, 90);
        assert_eq!(c1.scalefac_compress, 130);
        assert_eq!(c1.table_select, [4, 5, 6]);
        assert_eq!(c1.region0_count, 9);
        assert_eq!(c1.region1_count, 6);
        assert!(c1.scalefac_scale);
        assert!(!c1.count1table_select);
    }

    #[test]
    fn lsf_mono_window_switched_short() {
        let hdr = parse_header(&lsf_l3_header(0b11, 1)).unwrap(); // mono

        let mut w = BitWriter::new();
        w.put(0, 8); // main_data_begin
        w.put_bool(false); // private_bits
        put_lsf_switched_channel(
            &mut w,
            480,       // part2_3_length
            7,         // big_values
            188,       // global_gain
            500,       // scalefac_compress (in the preflag=1 LSF range)
            2,         // block_type = short
            false,     // mixed_block_flag
            [21, 9],   // table_select
            [3, 4, 5], // subblock_gain
            true,      // scalefac_scale
            true,      // count1table_select
        );
        let bytes = w.finish();
        assert_eq!(bytes.len(), SIDE_INFO_BYTES_LSF_MONO);

        let si = parse_side_info(&hdr, &bytes).unwrap();
        assert!(si.lsf);
        let g = &si.granules[0][0];
        assert!(g.window_switching_flag);
        assert_eq!(g.block_type, BlockType::Short);
        assert!(!g.mixed_block_flag);
        assert_eq!(g.table_select, [21, 9, 0]); // third entry absent
        assert_eq!(g.subblock_gain, [3, 4, 5]);
        // §2.4.2.7 default: short + !mixed => region0_count = 8.
        assert_eq!(g.region0_count, 8);
        assert_eq!(g.region1_count, 63);
        assert_eq!(g.scalefac_compress, 500);
        // preflag is not transmitted in LSF, regardless of
        // scalefac_compress; it stays false until scalefactor decode.
        assert!(!g.preflag);
        assert!(g.scalefac_scale);
        assert!(g.count1table_select);
    }

    #[test]
    fn lsf_max_field_values_pack_and_unpack() {
        // Saturate every field to its maximum to catch LSF bit-width
        // mistakes (a too-wide read would steal a neighbour's bits).
        let hdr = parse_header(&lsf_l3_header(0b11, 1)).unwrap(); // mono
        let mut w = BitWriter::new();
        w.put(0xFF, 8); // main_data_begin max (8 bits)
        w.put_bool(true); // private_bits max (1 bit)
        put_lsf_long_channel(
            &mut w,
            0xFFF,              // part2_3_length max (12 bits)
            0x1FF,              // big_values max (9 bits)
            0xFF,               // global_gain max (8 bits)
            0x1FF,              // scalefac_compress max (9 bits)
            [0x1F, 0x1F, 0x1F], // table_select max (5 bits)
            0xF,                // region0_count max (4 bits)
            0x7,                // region1_count max (3 bits)
            true,
            true,
        );
        let bytes = w.finish();
        assert_eq!(bytes.len(), SIDE_INFO_BYTES_LSF_MONO);
        let si = parse_side_info(&hdr, &bytes).unwrap();
        assert_eq!(si.main_data_begin, 0xFF);
        assert_eq!(si.private_bits, 1);
        let g = &si.granules[0][0];
        assert_eq!(g.part2_3_length, 0xFFF);
        assert_eq!(g.big_values, 0x1FF);
        assert_eq!(g.global_gain, 0xFF);
        assert_eq!(g.scalefac_compress, 0x1FF);
        assert_eq!(g.table_select, [0x1F, 0x1F, 0x1F]);
        assert_eq!(g.region0_count, 0xF);
        assert_eq!(g.region1_count, 0x7);
        assert!(g.scalefac_scale);
        assert!(g.count1table_select);
    }

    #[test]
    fn lsf_rejects_too_short() {
        // LSF stereo needs 17 bytes; LSF mono needs 9.
        let stereo = parse_header(&lsf_l3_header(0b00, 1)).unwrap();
        let buf16 = vec![0u8; SIDE_INFO_BYTES_LSF_STEREO - 1];
        assert_eq!(
            parse_side_info(&stereo, &buf16),
            Err(SideInfoError::TooShort)
        );
        let mono = parse_header(&lsf_l3_header(0b11, 1)).unwrap();
        let buf9 = vec![0u8; SIDE_INFO_BYTES_LSF_MONO];
        assert!(parse_side_info(&mono, &buf9).is_ok());
        let buf8 = vec![0u8; SIDE_INFO_BYTES_LSF_MONO - 1];
        assert_eq!(parse_side_info(&mono, &buf8), Err(SideInfoError::TooShort));
        // A 16-byte buffer is enough for LSF mono but not LSF stereo.
        let buf16b = vec![0u8; 16];
        assert!(parse_side_info(&mono, &buf16b).is_ok());
        assert_eq!(
            parse_side_info(&stereo, &buf16b),
            Err(SideInfoError::TooShort)
        );
    }

    #[test]
    fn lsf_extra_trailing_bytes_ignored() {
        // An LSF buffer longer than its block parses identically.
        let hdr = parse_header(&lsf_l3_header(0b11, 1)).unwrap();
        let mut w = BitWriter::new();
        w.put(33, 8);
        w.put_bool(true);
        put_lsf_long_channel(&mut w, 5, 5, 5, 5, [5, 5, 5], 5, 5, false, false);
        let mut bytes = w.finish();
        let exact = parse_side_info(&hdr, &bytes).unwrap();
        bytes.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);
        let with_tail = parse_side_info(&hdr, &bytes).unwrap();
        assert_eq!(exact, with_tail);
        assert_eq!(exact.main_data_begin, 33);
        assert_eq!(exact.granules[0][0].scalefac_compress, 5);
    }
}
