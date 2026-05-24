//! Layer III **scalefactor decode** — the main-data stage between
//! side-information parsing and Huffman decoding.
//!
//! This module covers exactly one decode stage of ISO/IEC 11172-3:1993
//! / ISO/IEC 13818-3:1997 Layer III: reading the per-granule,
//! per-channel scalefactors from the main-data bit reservoir, given a
//! [`SideInfo`] block already parsed by [`crate::side_info`]. It does
//! **not** perform Huffman decode of the big-values / count1 region, nor
//! requantisation, nor the IMDCT — those are later stages.
//!
//! # The bit reservoir
//!
//! Layer III main data does not generally follow the header + side info
//! of its own frame: it begins at a negative byte offset
//! `main_data_begin` measured back from the first byte *after* this
//! frame's side information (ISO/IEC 11172-3 §2.4.1.7 / §2.4.2.7, and
//! the §2.4.2.7 figure A.7.a). The decoder therefore maintains a
//! rolling buffer of the *main-data* bytes of recent frames — the bit
//! reservoir — and, for each frame, seeks `main_data_begin` bytes back
//! into it before reading. [`Reservoir`] models that buffer;
//! [`MainDataReader`] is an MSB-first bit reader over a contiguous
//! main-data byte run.
//!
//! # MPEG-1 scalefactor decode (ISO/IEC 11172-3 §2.4.2.7 / §2.4.3.4.5)
//!
//! The bit widths `slen1` / `slen2` are looked up from the 4-bit
//! `scalefac_compress` via [`MPEG1_SLEN`]. For a long block the 21
//! scalefactor bands split `[0,10]`→`slen1`, `[11,20]`→`slen2`; for a
//! short (`block_type==2`) block the read is per-window. `scfsi[ch][g]`
//! lets granule 1 reuse granule 0's scalefactors for the band group
//! `g` (long blocks only; short blocks always have `scfsi==0`).
//!
//! # MPEG-2 / MPEG-2.5 LSF scalefactor decode (ISO/IEC 13818-3 §2.4.3.4)
//!
//! The LSF form replaces the §2.4.3.4.5 "Scalefactors" paragraph with a
//! 4-partition scheme: `slen1..slen4`, `nr_of_sfb1..nr_of_sfb4`, and
//! `preflag` are all *derived* from the 9-bit `scalefac_compress` (and,
//! for the right channel of an intensity-stereo frame, from the
//! `int_scalefac_compress = scalefac_compress >> 1`). There is one
//! granule, so there is no `scfsi` reuse.

use crate::frame::{ChannelMode, Mp3FrameHeader, MpegVersion};
use crate::side_info::{BlockType, GranuleChannel, SideInfo};

/// Number of long-block scalefactor bands in a Layer III granule
/// (ISO/IEC 11172-3 §2.4.2.7: `scalefac_l[gr][ch][0..21]`).
pub const LONG_SFB: usize = 21;

/// Number of short-block scalefactor bands per window (ISO/IEC 11172-3
/// §2.4.2.7: `scalefac_s[gr][ch][0..12][window]`).
pub const SHORT_SFB: usize = 12;

/// Number of short windows in a `block_type==2` granule (ISO/IEC
/// 11172-3 §2.4.2.7: `for (window=0; window<3; window++)`).
pub const SHORT_WINDOWS: usize = 3;

/// MPEG-1 `scalefac_compress` → `(slen1, slen2)` table (ISO/IEC
/// 11172-3 §2.4.2.7).
///
/// `scalefac_compress` is a 4-bit side-info field (0..=15). `slen1` is
/// the per-scalefactor bit width for the low band group, `slen2` for
/// the high band group; the exact band ranges depend on the block type
/// (see [`mpeg1_long_band_slen`] / the short-block read in
/// [`read_mpeg1_granule_channel`]).
pub const MPEG1_SLEN: [(u8, u8); 16] = [
    (0, 0), // 0
    (0, 1), // 1
    (0, 2), // 2
    (0, 3), // 3
    (3, 0), // 4
    (1, 1), // 5
    (1, 2), // 6
    (1, 3), // 7
    (2, 1), // 8
    (2, 2), // 9
    (2, 3), // 10
    (3, 1), // 11
    (3, 2), // 12
    (3, 3), // 13
    (4, 2), // 14
    (4, 3), // 15
];

/// Per-granule, per-channel decoded scalefactors.
///
/// For a long block, [`ScaleFactors::long`] holds the 21 long-block
/// scalefactors (`scalefac_l`) and [`ScaleFactors::short`] is all zero.
/// For a short (`block_type==2`) block, [`ScaleFactors::short`] holds
/// the per-window short-block scalefactors (`scalefac_s`); a *mixed*
/// short block additionally populates the low entries of
/// [`ScaleFactors::long`] (the long-window portion). Bands above the
/// transmitted range are zero (ISO/IEC 11172-3 §2.4.2.7: "the scale
/// factor for frequency lines above the highest line … is zero").
///
/// [`ScaleFactors::preflag`] is the *effective* preflag for this
/// granule/channel: the transmitted side-info bit for MPEG-1, or the
/// value derived from `scalefac_compress` for LSF.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScaleFactors {
    /// `scalefac_l[sfb]` for `sfb` in `0..21`. Zero where not
    /// transmitted (short blocks beyond the mixed long-window portion,
    /// or bands above the transmitted range).
    pub long: [u8; LONG_SFB],
    /// `scalefac_s[sfb][window]` for `sfb` in `0..12`, `window` in
    /// `0..3`. All zero for long blocks.
    pub short: [[u8; SHORT_WINDOWS]; SHORT_SFB],
    /// The effective high-frequency-amplification preflag for this
    /// granule/channel (transmitted for MPEG-1, derived from
    /// `scalefac_compress` for LSF). Never set for short blocks in
    /// MPEG-1 (§2.4.2.7).
    pub preflag: bool,
    /// The intensity-stereo scaling selector, derived from the right
    /// channel's `scalefac_compress` in an LSF intensity-stereo frame
    /// (`intensity_scale = scalefac_compress % 2`, ISO/IEC 13818-3
    /// §2.4.3.4). `false` for MPEG-1 and for non-intensity LSF
    /// channels.
    pub intensity_scale: bool,
}

impl Default for ScaleFactors {
    fn default() -> Self {
        ScaleFactors {
            long: [0; LONG_SFB],
            short: [[0; SHORT_WINDOWS]; SHORT_SFB],
            preflag: false,
            intensity_scale: false,
        }
    }
}

/// All decoded scalefactors for one frame, indexed `[granule][channel]`.
///
/// Cells outside [`SideInfo::granule_count`] × [`SideInfo::channels`]
/// are left at their [`ScaleFactors::default`] (all-zero) value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameScaleFactors {
    /// Per-granule, per-channel scalefactors. `[granule][channel]`.
    pub granules: [[ScaleFactors; 2]; 2],
    /// Number of granules actually decoded (mirrors
    /// [`SideInfo::granule_count`]).
    pub granule_count: u8,
    /// Number of channels actually decoded (mirrors
    /// [`SideInfo::channels`]).
    pub channels: u8,
}

/// Errors returned by the scalefactor decode stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleFactorError {
    /// The main-data bit reservoir did not hold enough bytes to satisfy
    /// a `main_data_begin` back-pointer, i.e. the referenced bytes were
    /// never seen (a malformed or mid-stream-started bitstream).
    ReservoirUnderflow,
    /// The contiguous main-data run was exhausted before the scalefactor
    /// fields for every granule/channel had been read (`part2_length`
    /// exceeds the bytes available).
    OutOfData,
}

impl core::fmt::Display for ScaleFactorError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let msg = match self {
            ScaleFactorError::ReservoirUnderflow => {
                "main_data_begin points before the start of the bit reservoir"
            }
            ScaleFactorError::OutOfData => {
                "main-data run exhausted before all scalefactors were read"
            }
        };
        f.write_str(msg)
    }
}

impl std::error::Error for ScaleFactorError {}

/// LSF (MPEG-2 / MPEG-2.5) scalefactor partitioning derived from
/// `scalefac_compress` (ISO/IEC 13818-3 §2.4.3.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LsfScaleParams {
    /// Per-partition scalefactor bit widths.
    pub slen: [u8; 4],
    /// Per-partition scalefactor-band counts.
    pub nr_of_sfb: [u8; 4],
    /// Derived high-frequency-amplification preflag.
    pub preflag: bool,
    /// Derived intensity-stereo scaling selector (only meaningful for
    /// the right channel of an intensity-stereo frame).
    pub intensity_scale: bool,
}

/// A rolling main-data byte reservoir (ISO/IEC 11172-3 §2.4.2.7).
///
/// Layer III frames place their main data at a negative byte offset
/// `main_data_begin` measured from the first byte *after* the side
/// info. The decoder feeds each frame's main-data bytes here in stream
/// order; [`Reservoir::assemble`] then returns the contiguous run that
/// a frame with a given `main_data_begin` should read, namely the last
/// `main_data_begin` bytes already buffered followed by this frame's own
/// main-data bytes.
#[derive(Debug, Default, Clone)]
pub struct Reservoir {
    buf: Vec<u8>,
}

impl Reservoir {
    /// A new, empty reservoir.
    #[must_use]
    pub fn new() -> Self {
        Reservoir { buf: Vec::new() }
    }

    /// Number of bytes currently buffered.
    #[must_use]
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// `true` when the reservoir holds no bytes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Discard all buffered bytes (e.g. after a seek/resync).
    pub fn clear(&mut self) {
        self.buf.clear();
    }

    /// Append one frame's main-data bytes to the reservoir, then return
    /// the contiguous run this same frame should decode from: the
    /// `main_data_begin` bytes that preceded `frame_main_data`, followed
    /// by `frame_main_data` itself.
    ///
    /// The reservoir is trimmed afterwards so it never grows without
    /// bound: only the most recent bytes that any future frame could
    /// reference are retained. ISO/IEC 11172-3 §2.4.2.7 bounds
    /// `main_data_begin` to 9 bits (≤ 511) for MPEG-1 and 8 bits
    /// (≤ 255) for LSF, so retaining 512 trailing bytes is always
    /// sufficient.
    ///
    /// # Errors
    ///
    /// [`ScaleFactorError::ReservoirUnderflow`] if `main_data_begin`
    /// exceeds the bytes buffered *before* this frame (the referenced
    /// history was never seen).
    pub fn assemble(
        &mut self,
        main_data_begin: usize,
        frame_main_data: &[u8],
    ) -> Result<Vec<u8>, ScaleFactorError> {
        if main_data_begin > self.buf.len() {
            // Still append so the stream can recover on the next frame,
            // but this frame cannot be decoded.
            self.push_and_trim(frame_main_data);
            return Err(ScaleFactorError::ReservoirUnderflow);
        }
        let carry_start = self.buf.len() - main_data_begin;
        let mut run = Vec::with_capacity(main_data_begin + frame_main_data.len());
        run.extend_from_slice(&self.buf[carry_start..]);
        run.extend_from_slice(frame_main_data);
        self.push_and_trim(frame_main_data);
        Ok(run)
    }

    fn push_and_trim(&mut self, frame_main_data: &[u8]) {
        self.buf.extend_from_slice(frame_main_data);
        // Retain only the trailing window a future main_data_begin could
        // reference (max 511 for MPEG-1, 255 for LSF).
        const MAX_BACK: usize = 512;
        if self.buf.len() > MAX_BACK {
            let drop = self.buf.len() - MAX_BACK;
            self.buf.drain(..drop);
        }
    }
}

/// An MSB-first bit reader over a contiguous main-data byte run.
///
/// ISO/IEC 11172-3 §2.4.1: the most significant bit of each byte is
/// transmitted first. Reads past the end return zero bits and set the
/// exhausted flag, which callers check via [`MainDataReader::exhausted`]
/// to surface [`ScaleFactorError::OutOfData`].
#[derive(Debug)]
pub struct MainDataReader<'a> {
    bytes: &'a [u8],
    bit_pos: usize,
    exhausted: bool,
}

impl<'a> MainDataReader<'a> {
    /// A new reader positioned at bit 0.
    #[must_use]
    pub fn new(bytes: &'a [u8]) -> Self {
        MainDataReader {
            bytes,
            bit_pos: 0,
            exhausted: false,
        }
    }

    /// The current bit position (number of bits consumed).
    #[must_use]
    pub fn bit_pos(&self) -> usize {
        self.bit_pos
    }

    /// `true` if a read ran past the end of the byte run.
    #[must_use]
    pub fn exhausted(&self) -> bool {
        self.exhausted
    }

    /// Read `n` bits (`0 ≤ n ≤ 32`) MSB-first as an unsigned integer.
    ///
    /// Reading zero bits returns `0`. Reading past the end yields the
    /// bits available with the remainder zero-filled and sets
    /// [`MainDataReader::exhausted`].
    pub fn read(&mut self, n: u32) -> u32 {
        let mut value: u32 = 0;
        for _ in 0..n {
            let byte_idx = self.bit_pos >> 3;
            let bit = if byte_idx < self.bytes.len() {
                let shift = 7 - (self.bit_pos & 7);
                (self.bytes[byte_idx] >> shift) & 1
            } else {
                self.exhausted = true;
                0
            };
            value = (value << 1) | u32::from(bit);
            self.bit_pos += 1;
        }
        value
    }
}

/// Derive the LSF `(slen1..4, nr_of_sfb1..4, preflag, intensity_scale)`
/// from the 9-bit `scalefac_compress` (ISO/IEC 13818-3 §2.4.3.4).
///
/// `block_type` and `mixed_block_flag` select which row of the
/// per-range `nr_of_sfb` table applies. `is_intensity_right` is `true`
/// for the right channel (`ch==1`) of an intensity-stereo frame
/// (`mode_extension` `'01'` or `'11'`); that branch decodes against
/// `int_scalefac_compress = scalefac_compress >> 1` and a different
/// table per §2.4.3.4.
#[must_use]
pub fn lsf_scale_params(
    scalefac_compress: u16,
    block_type: BlockType,
    mixed_block_flag: bool,
    is_intensity_right: bool,
) -> LsfScaleParams {
    // Row selector into the per-range nr_of_sfb tables:
    //   0 => block_type in {0,1,3}  ('00','01','11')
    //   1 => block_type==2 (short), mixed_block_flag==0  ('10', mbf 0)
    //   2 => block_type==2 (short), mixed_block_flag==1  ('10', mbf 1)
    let row = if block_type == BlockType::Short {
        if mixed_block_flag {
            2
        } else {
            1
        }
    } else {
        0
    };

    if !is_intensity_right {
        let sc = u32::from(scalefac_compress);
        if sc < 400 {
            let slen = [
                ((sc >> 4) / 5) as u8,
                ((sc >> 4) % 5) as u8,
                ((sc % 16) >> 2) as u8,
                (sc % 4) as u8,
            ];
            // Table for scalefac_compress < 400.
            let nr = [[6, 5, 5, 5], [9, 9, 9, 9], [6, 9, 9, 9]][row];
            LsfScaleParams {
                slen,
                nr_of_sfb: nr,
                preflag: false,
                intensity_scale: false,
            }
        } else if sc < 500 {
            let v = sc - 400;
            let slen = [((v >> 2) / 5) as u8, ((v >> 2) % 5) as u8, (v % 4) as u8, 0];
            let nr = [[6, 5, 7, 3], [9, 9, 12, 6], [6, 9, 12, 6]][row];
            LsfScaleParams {
                slen,
                nr_of_sfb: nr,
                preflag: false,
                intensity_scale: false,
            }
        } else {
            // 500 <= scalefac_compress < 512
            let v = sc - 500;
            let slen = [(v / 3) as u8, (v % 3) as u8, 0, 0];
            let nr = [[11, 10, 0, 0], [18, 18, 0, 0], [15, 18, 0, 0]][row];
            LsfScaleParams {
                slen,
                nr_of_sfb: nr,
                preflag: true,
                intensity_scale: false,
            }
        }
    } else {
        let intensity_scale = (scalefac_compress % 2) == 1;
        let isc = u32::from(scalefac_compress >> 1);
        if isc < 180 {
            let slen = [
                (isc / 36) as u8,
                ((isc % 36) / 6) as u8,
                ((isc % 36) % 6) as u8,
                0,
            ];
            let nr = [[7, 7, 7, 0], [12, 12, 12, 0], [6, 15, 12, 0]][row];
            LsfScaleParams {
                slen,
                nr_of_sfb: nr,
                preflag: false,
                intensity_scale,
            }
        } else if isc < 244 {
            let v = isc - 180;
            let slen = [
                ((v % 64) >> 4) as u8,
                ((v % 16) >> 2) as u8,
                (v % 4) as u8,
                0,
            ];
            let nr = [[6, 6, 6, 3], [12, 9, 9, 6], [6, 12, 9, 6]][row];
            LsfScaleParams {
                slen,
                nr_of_sfb: nr,
                preflag: false,
                intensity_scale,
            }
        } else {
            // 244 <= int_scalefac_compress <= 255
            let v = isc - 244;
            let slen = [(v / 3) as u8, (v % 3) as u8, 0, 0];
            let nr = [[8, 8, 5, 0], [15, 12, 9, 0], [6, 18, 9, 0]][row];
            LsfScaleParams {
                slen,
                nr_of_sfb: nr,
                preflag: false,
                intensity_scale,
            }
        }
    }
}

/// The slen used for a given MPEG-1 long-block scalefactor band:
/// `slen1` for `sfb` in `0..=10`, `slen2` for `sfb` in `11..=20`
/// (ISO/IEC 11172-3 §2.4.2.7).
#[must_use]
pub fn mpeg1_long_band_slen(sfb: usize, slen1: u8, slen2: u8) -> u8 {
    if sfb <= 10 {
        slen1
    } else {
        slen2
    }
}

/// Read the MPEG-1 scalefactors for one granule/channel from `r`,
/// honouring `scfsi` reuse from `prev` (granule 0's scalefactors) when
/// `gr == 1`.
///
/// `gc` is the granule/channel side info. `scfsi` is this channel's
/// 4-band selection-information array. `prev` is granule 0's decoded
/// scalefactors for this channel, used to fill the reused bands when
/// `gr == 1` and the corresponding `scfsi` bit is set; pass `None` for
/// `gr == 0`.
fn read_mpeg1_granule_channel(
    r: &mut MainDataReader<'_>,
    gc: &GranuleChannel,
    scfsi: &[bool; 4],
    gr: usize,
    prev: Option<&ScaleFactors>,
) -> ScaleFactors {
    let (slen1, slen2) = MPEG1_SLEN[(gc.scalefac_compress & 0xF) as usize];
    let mut sf = ScaleFactors {
        preflag: gc.preflag,
        ..ScaleFactors::default()
    };

    let short = gc.window_switching_flag && gc.block_type == BlockType::Short;

    if short {
        if gc.mixed_block_flag {
            // Mixed: long-window scalefactors sfb 0..8 at slen1, then
            // short-window scalefactors sfb 3..12 (sfb 3..6 at slen1,
            // sfb 6..12 at slen2). (ISO/IEC 11172-3 §2.4.2.7 main_data.)
            for band in sf.long.iter_mut().take(8) {
                *band = r.read(u32::from(slen1)) as u8;
            }
            for sfb in 3..12 {
                let slen = if sfb < 6 { slen1 } else { slen2 };
                for win in 0..SHORT_WINDOWS {
                    sf.short[sfb][win] = r.read(u32::from(slen)) as u8;
                }
            }
        } else {
            // Pure short: sfb 0..6 at slen1, sfb 6..12 at slen2, each
            // over three windows.
            for sfb in 0..SHORT_SFB {
                let slen = if sfb < 6 { slen1 } else { slen2 };
                for win in 0..SHORT_WINDOWS {
                    sf.short[sfb][win] = r.read(u32::from(slen)) as u8;
                }
            }
        }
        // preflag is never set for short blocks (§2.4.2.7).
        sf.preflag = false;
    } else {
        // Long block: four scfsi band groups. Band group g is reused
        // from granule 0 when gr==1 && scfsi[g]; otherwise it is read.
        // Groups: [0,6), [6,11), [11,16), [16,21); the first two use
        // slen1, the last two slen2 (ISO/IEC 11172-3 §2.4.2.7).
        const GROUPS: [(usize, usize); 4] = [(0, 6), (6, 11), (11, 16), (16, 21)];
        for (g, &(lo, hi)) in GROUPS.iter().enumerate() {
            let reuse = gr == 1 && scfsi[g];
            for sfb in lo..hi {
                if reuse {
                    sf.long[sfb] = prev.map_or(0, |p| p.long[sfb]);
                } else {
                    let slen = mpeg1_long_band_slen(sfb, slen1, slen2);
                    sf.long[sfb] = r.read(u32::from(slen)) as u8;
                }
            }
        }
    }

    sf
}

/// Read the LSF (MPEG-2 / MPEG-2.5) scalefactors for one channel from
/// `r` (ISO/IEC 13818-3 §2.4.3.4). There is one granule, so no `scfsi`.
fn read_lsf_channel(
    r: &mut MainDataReader<'_>,
    gc: &GranuleChannel,
    is_intensity_right: bool,
) -> ScaleFactors {
    let params = lsf_scale_params(
        gc.scalefac_compress,
        gc.block_type,
        gc.mixed_block_flag,
        is_intensity_right,
    );
    let mut sf = ScaleFactors {
        preflag: params.preflag,
        intensity_scale: params.intensity_scale,
        ..ScaleFactors::default()
    };

    let short = gc.window_switching_flag && gc.block_type == BlockType::Short;

    // The four partitions are read consecutively. Each partition p
    // contributes nr_of_sfb[p] scalefactor entries of width slen[p]
    // bits. When slen[p]==0 the entries are zero with no bits read
    // (§2.4.3.4: "scalefactors of these bands must be set to zero").
    if short {
        if gc.mixed_block_flag {
            // Mixed LSF: the read order is scalefac_l sfb 0..6 (long
            // window) followed by scalefac_s sfb 3..12 over three
            // windows (ISO/IEC 13818-3 §2.4.1.7 main_data). The four
            // partitions span this sequence in order.
            let mut entry = LsfMixedShortWriter::new(&mut sf);
            read_partitions(r, &params, &mut entry);
        } else {
            // Pure short LSF: scalefac_s sfb 0..12 over three windows,
            // in (sfb,window) order.
            let mut entry = LsfShortWriter::new(&mut sf);
            read_partitions(r, &params, &mut entry);
        }
    } else {
        // Long LSF: scalefac_l sfb 0..21 in order.
        let mut entry = LsfLongWriter::new(&mut sf);
        read_partitions(r, &params, &mut entry);
    }

    sf
}

/// A sink that places successive scalefactor values into the right slot
/// of a [`ScaleFactors`] as the LSF four-partition reader walks them.
trait ScaleFactorSink {
    /// Store one scalefactor value at the next position. Extra values
    /// beyond the layout's capacity are ignored (the partition totals
    /// always match the layout for well-formed `scalefac_compress`).
    fn put(&mut self, value: u8);
}

/// Read the four LSF partitions into `sink`: for each partition `p`,
/// `nr_of_sfb[p]` values of `slen[p]` bits each (zero-width partitions
/// emit zeros without consuming bits).
fn read_partitions(
    r: &mut MainDataReader<'_>,
    params: &LsfScaleParams,
    sink: &mut dyn ScaleFactorSink,
) {
    for p in 0..4 {
        let slen = params.slen[p];
        let count = params.nr_of_sfb[p];
        for _ in 0..count {
            let v = if slen == 0 {
                0
            } else {
                r.read(u32::from(slen)) as u8
            };
            sink.put(v);
        }
    }
}

/// Sink for LSF long blocks: fills `long[0..21]` in order.
struct LsfLongWriter<'a> {
    sf: &'a mut ScaleFactors,
    idx: usize,
}
impl<'a> LsfLongWriter<'a> {
    fn new(sf: &'a mut ScaleFactors) -> Self {
        LsfLongWriter { sf, idx: 0 }
    }
}
impl ScaleFactorSink for LsfLongWriter<'_> {
    fn put(&mut self, value: u8) {
        if self.idx < LONG_SFB {
            self.sf.long[self.idx] = value;
        }
        self.idx += 1;
    }
}

/// Sink for pure-short LSF blocks: fills `short[sfb][window]` for
/// `sfb` 0..12, `window` 0..3, in `(sfb, window)` order.
struct LsfShortWriter<'a> {
    sf: &'a mut ScaleFactors,
    idx: usize,
}
impl<'a> LsfShortWriter<'a> {
    fn new(sf: &'a mut ScaleFactors) -> Self {
        LsfShortWriter { sf, idx: 0 }
    }
}
impl ScaleFactorSink for LsfShortWriter<'_> {
    fn put(&mut self, value: u8) {
        let sfb = self.idx / SHORT_WINDOWS;
        let win = self.idx % SHORT_WINDOWS;
        if sfb < SHORT_SFB {
            self.sf.short[sfb][win] = value;
        }
        self.idx += 1;
    }
}

/// Sink for mixed-short LSF blocks: first 6 values go to `long[0..6]`
/// (long-window portion), then the remainder fills `short[sfb][window]`
/// for `sfb` 3..12, `window` 0..3 in `(sfb, window)` order (ISO/IEC
/// 13818-3 §2.4.1.7 main_data read order).
struct LsfMixedShortWriter<'a> {
    sf: &'a mut ScaleFactors,
    idx: usize,
}
impl<'a> LsfMixedShortWriter<'a> {
    const LONG_PART: usize = 6;
    fn new(sf: &'a mut ScaleFactors) -> Self {
        LsfMixedShortWriter { sf, idx: 0 }
    }
}
impl ScaleFactorSink for LsfMixedShortWriter<'_> {
    fn put(&mut self, value: u8) {
        if self.idx < Self::LONG_PART {
            self.sf.long[self.idx] = value;
        } else {
            let off = self.idx - Self::LONG_PART;
            let sfb = 3 + off / SHORT_WINDOWS;
            let win = off % SHORT_WINDOWS;
            if sfb < SHORT_SFB {
                self.sf.short[sfb][win] = value;
            }
        }
        self.idx += 1;
    }
}

/// Decode all Layer III scalefactors for one frame from a contiguous
/// main-data byte run.
///
/// `header` provides the version (MPEG-1 vs LSF) and the channel
/// `mode` / `mode_extension` (needed for the LSF intensity-stereo
/// right-channel branch). `side_info` provides every per-granule,
/// per-channel field, including `scfsi` for MPEG-1 reuse. `main_data`
/// is the contiguous run produced by [`Reservoir::assemble`] (or, for a
/// self-contained frame with `main_data_begin == 0`, simply the frame's
/// own main-data bytes).
///
/// The returned [`FrameScaleFactors`] holds the decoded scalefactors,
/// indexed `[granule][channel]`. The reader's bit position after the
/// scalefactors is *not* returned here — the Huffman stage (a later
/// round) re-reads from `main_data` using each granule's
/// `part2_3_length` to bound the combined scalefactor + Huffman region.
///
/// # Errors
///
/// [`ScaleFactorError::OutOfData`] if `main_data` was exhausted before
/// all scalefactor fields were read.
pub fn decode_scalefactors(
    header: &Mp3FrameHeader,
    side_info: &SideInfo,
    main_data: &[u8],
) -> Result<FrameScaleFactors, ScaleFactorError> {
    let mut r = MainDataReader::new(main_data);
    let mut out = FrameScaleFactors {
        granules: [[ScaleFactors::default(); 2]; 2],
        granule_count: side_info.granule_count,
        channels: side_info.channels,
    };

    let nch = side_info.channels as usize;
    let ngr = side_info.granule_count as usize;
    let intensity = is_intensity_stereo(header);

    match header.version {
        MpegVersion::Mpeg1 => {
            for gr in 0..ngr {
                for ch in 0..nch {
                    let gc = &side_info.granules[gr][ch];
                    let prev = if gr == 1 {
                        Some(out.granules[0][ch])
                    } else {
                        None
                    };
                    let sf = read_mpeg1_granule_channel(
                        &mut r,
                        gc,
                        &side_info.scfsi[ch],
                        gr,
                        prev.as_ref(),
                    );
                    out.granules[gr][ch] = sf;
                }
            }
        }
        MpegVersion::Mpeg2 => {
            // LSF: exactly one granule.
            for ch in 0..nch {
                let gc = &side_info.granules[0][ch];
                let is_intensity_right = intensity && ch == 1;
                let sf = read_lsf_channel(&mut r, gc, is_intensity_right);
                out.granules[0][ch] = sf;
            }
        }
    }

    if r.exhausted() {
        return Err(ScaleFactorError::OutOfData);
    }

    Ok(out)
}

/// `true` when the frame uses intensity stereo, i.e. joint-stereo mode
/// with the intensity bit of `mode_extension` set (`'01'` or `'11'`).
/// Only then does the LSF right-channel `int_scalefac_compress` branch
/// apply (ISO/IEC 13818-3 §2.4.3.4).
fn is_intensity_stereo(header: &Mp3FrameHeader) -> bool {
    header.mode == ChannelMode::JointStereo && header.mode_extension.intensity_stereo
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::parse_header;
    use crate::side_info::parse_side_info;

    // --- Bit writer mirroring the side_info tests, so every fixture is
    //     constructed from the §2.4.1.7 / §2.4.3.4 field layout rather
    //     than a pre-baked byte pattern from any external source. ---

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

    // Header builders (sample-rate/bitrate values irrelevant to the
    // scalefactor stage; chosen valid).
    fn mpeg1_header(mode: u32) -> [u8; 4] {
        let raw: u32 = (0xFFF << 20)
            | (1 << 19) // ID = MPEG-1
            | (0b01 << 17) // layer III
            | (1 << 16) // no CRC
            | (0b1001 << 12) // bitrate 128k
            | (mode << 6)
            | (1 << 2);
        raw.to_be_bytes()
    }
    fn lsf_header(mode: u32, mode_ext: u32) -> [u8; 4] {
        let raw: u32 = (0xFFF << 20)
            | (0b01 << 17) // layer III, ID=0 (LSF)
            | (1 << 16)
            | (0b1001 << 12)
            | (mode << 6)
            | (mode_ext << 4)
            | (1 << 2);
        raw.to_be_bytes()
    }

    // ---- bit reservoir ----

    #[test]
    fn reservoir_self_contained_frame() {
        // main_data_begin == 0: the run is exactly this frame's bytes.
        let mut res = Reservoir::new();
        let frame = [1u8, 2, 3, 4];
        let run = res.assemble(0, &frame).unwrap();
        assert_eq!(run, frame);
        assert_eq!(res.len(), 4);
    }

    #[test]
    fn reservoir_back_reference() {
        let mut res = Reservoir::new();
        // Frame A contributes 5 bytes, no back-reference.
        res.assemble(0, &[10, 11, 12, 13, 14]).unwrap();
        // Frame B references 3 bytes back, then its own 2 bytes.
        let run = res.assemble(3, &[20, 21]).unwrap();
        assert_eq!(run, vec![12, 13, 14, 20, 21]);
    }

    #[test]
    fn reservoir_underflow() {
        let mut res = Reservoir::new();
        res.assemble(0, &[1, 2]).unwrap();
        // Reference 5 bytes back but only 2 are buffered.
        assert_eq!(
            res.assemble(5, &[3]),
            Err(ScaleFactorError::ReservoirUnderflow)
        );
        // The 3rd frame's byte was still appended for recovery.
        assert_eq!(res.len(), 3);
    }

    #[test]
    fn reservoir_trims_to_window() {
        let mut res = Reservoir::new();
        // Push 1000 bytes across frames; only the trailing 512 retained.
        for _ in 0..10 {
            res.assemble(0, &[0u8; 100]).unwrap();
        }
        assert_eq!(res.len(), 512);
    }

    // ---- MainDataReader ----

    #[test]
    fn reader_msb_first() {
        let mut r = MainDataReader::new(&[0b1011_0010, 0b1100_0000]);
        assert_eq!(r.read(1), 1);
        assert_eq!(r.read(3), 0b011);
        assert_eq!(r.read(4), 0b0010);
        assert_eq!(r.read(2), 0b11);
        assert!(!r.exhausted());
    }

    #[test]
    fn reader_exhaustion() {
        let mut r = MainDataReader::new(&[0xFF]);
        assert_eq!(r.read(8), 0xFF);
        assert!(!r.exhausted());
        let _ = r.read(1);
        assert!(r.exhausted());
    }

    // ---- MPEG-1 slen table ----

    #[test]
    fn mpeg1_slen_table_matches_spec() {
        // ISO/IEC 11172-3 §2.4.2.7 scalefac_compress table.
        assert_eq!(MPEG1_SLEN[0], (0, 0));
        assert_eq!(MPEG1_SLEN[4], (3, 0));
        assert_eq!(MPEG1_SLEN[5], (1, 1));
        assert_eq!(MPEG1_SLEN[13], (3, 3));
        assert_eq!(MPEG1_SLEN[14], (4, 2));
        assert_eq!(MPEG1_SLEN[15], (4, 3));
    }

    // ---- MPEG-1 long block, no scfsi ----

    /// Build an MPEG-1 mono side-info block (long blocks) with the given
    /// scalefac_compress per granule and scfsi pattern, returning the
    /// parsed SideInfo plus header. Helper for the scalefactor tests.
    #[allow(clippy::too_many_arguments)]
    fn mpeg1_mono_long_sideinfo(
        sc0: u32,
        sc1: u32,
        scfsi: [bool; 4],
    ) -> (Mp3FrameHeader, SideInfo) {
        let hdr = parse_header(&mpeg1_header(0b11)).unwrap();
        let mut w = BitWriter::new();
        w.put(0, 9); // main_data_begin
        w.put(0, 5); // private_bits (mono)
        for b in scfsi {
            w.put_bool(b);
        }
        for sc in [sc0, sc1] {
            w.put(200, 12); // part2_3_length
            w.put(50, 9); // big_values
            w.put(180, 8); // global_gain
            w.put(sc, 4); // scalefac_compress
            w.put_bool(false); // window_switching_flag = 0 (long)
            for _ in 0..3 {
                w.put(0, 5); // table_select
            }
            w.put(0, 4); // region0_count
            w.put(0, 3); // region1_count
            w.put_bool(false); // preflag
            w.put_bool(false); // scalefac_scale
            w.put_bool(false); // count1table_select
        }
        let bytes = w.finish();
        let si = parse_side_info(&hdr, &bytes).unwrap();
        (hdr, si)
    }

    #[test]
    fn mpeg1_long_no_scfsi_reads_both_granules() {
        // scalefac_compress = 5 => (slen1, slen2) = (1, 1): every band
        // is 1 bit wide. 21 bands/granule × 2 granules = 42 bits.
        let (hdr, si) = mpeg1_mono_long_sideinfo(5, 5, [false; 4]);
        // Build main data: write 21 scalefactors for gr0 then gr1,
        // each 1 bit. Use an alternating pattern so band index is
        // recoverable.
        let mut w = BitWriter::new();
        for _gr in 0..2 {
            for sfb in 0..21u32 {
                w.put(sfb % 2, 1); // 0,1,0,1,...
            }
        }
        let md = w.finish();
        let fsf = decode_scalefactors(&hdr, &si, &md).unwrap();
        assert_eq!(fsf.granule_count, 2);
        assert_eq!(fsf.channels, 1);
        for gr in 0..2 {
            for sfb in 0..21 {
                assert_eq!(
                    fsf.granules[gr][0].long[sfb],
                    (sfb % 2) as u8,
                    "gr{gr} sfb{sfb}"
                );
            }
        }
    }

    #[test]
    fn mpeg1_long_scfsi_reuse_granule1() {
        // scfsi reuse: granule 1 reuses granule 0's scalefactors for
        // the bands whose scfsi bit is set. Set scfsi groups 0 and 2.
        // scalefac_compress = 5 => (slen1, slen2) = (1,1).
        let scfsi = [true, false, true, false];
        let (hdr, si) = mpeg1_mono_long_sideinfo(5, 5, scfsi);

        // gr0: all 21 bands = 1. gr1: only the *non-reused* bands are
        // transmitted, value 0. Reused groups are 0 ([0,6)) and 2
        // ([11,16)); transmitted groups are 1 ([6,11)) and 3 ([16,21)).
        let mut w = BitWriter::new();
        // gr0: 21 ones.
        for _ in 0..21 {
            w.put(1, 1);
        }
        // gr1: group 1 (5 bands) + group 3 (5 bands) = 10 transmitted
        // zeros; groups 0 and 2 not transmitted (reused).
        for _ in 0..10 {
            w.put(0, 1);
        }
        let md = w.finish();
        let fsf = decode_scalefactors(&hdr, &si, &md).unwrap();

        // gr0 all ones.
        for sfb in 0..21 {
            assert_eq!(fsf.granules[0][0].long[sfb], 1, "gr0 sfb{sfb}");
        }
        // gr1: reused groups (0..6 and 11..16) keep gr0's value (1);
        // transmitted groups (6..11 and 16..21) are 0.
        for sfb in 0..21 {
            let reused = (0..6).contains(&sfb) || (11..16).contains(&sfb);
            let expect = if reused { 1 } else { 0 };
            assert_eq!(fsf.granules[1][0].long[sfb], expect, "gr1 sfb{sfb}");
        }
    }

    #[test]
    fn mpeg1_long_slen_widths_distinguish_band_groups() {
        // scalefac_compress = 7 => (slen1, slen2) = (1, 3). slen1 bands
        // (0..11) are 1 bit; slen2 bands (11..21) are 3 bits.
        let (hdr, si) = mpeg1_mono_long_sideinfo(7, 7, [false; 4]);
        let mut w = BitWriter::new();
        for _gr in 0..2 {
            // 11 slen1 bands, value 1 each.
            for _ in 0..11 {
                w.put(1, 1);
            }
            // 10 slen2 bands, value 5 each (3-bit).
            for _ in 0..10 {
                w.put(5, 3);
            }
        }
        let md = w.finish();
        let fsf = decode_scalefactors(&hdr, &si, &md).unwrap();
        for gr in 0..2 {
            for sfb in 0..11 {
                assert_eq!(fsf.granules[gr][0].long[sfb], 1, "gr{gr} sfb{sfb} slen1");
            }
            for sfb in 11..21 {
                assert_eq!(fsf.granules[gr][0].long[sfb], 5, "gr{gr} sfb{sfb} slen2");
            }
        }
    }

    // ---- MPEG-1 short block ----

    #[test]
    fn mpeg1_short_block_no_mixed() {
        // Pure short block (block_type=2, mixed=0): sfb 0..6 at slen1,
        // sfb 6..12 at slen2, each over 3 windows. scalefac_compress=7
        // => (slen1, slen2) = (1, 3).
        let hdr = parse_header(&mpeg1_header(0b11)).unwrap();
        let mut w = BitWriter::new();
        w.put(0, 9); // main_data_begin
        w.put(0, 5); // private_bits
        for _ in 0..4 {
            w.put_bool(false); // scfsi (must be 0 for short)
        }
        // gr0: short. gr1: short, identical fields.
        for _ in 0..2 {
            w.put(300, 12); // part2_3_length
            w.put(0, 9); // big_values
            w.put(190, 8); // global_gain
            w.put(7, 4); // scalefac_compress => (1,3)
            w.put_bool(true); // window_switching_flag
            w.put(2, 2); // block_type = short
            w.put_bool(false); // mixed_block_flag
            for _ in 0..2 {
                w.put(0, 5); // table_select (2 in short branch)
            }
            for _ in 0..3 {
                w.put(0, 3); // subblock_gain
            }
            w.put_bool(false); // preflag
            w.put_bool(false); // scalefac_scale
            w.put_bool(false); // count1table_select
        }
        let si = parse_side_info(&hdr, &w.finish()).unwrap();
        assert!(si.granules[0][0].window_switching_flag);
        assert_eq!(si.granules[0][0].block_type, BlockType::Short);

        // main data: per granule, sfb 0..6 (1-bit) × 3 win, sfb 6..12
        // (3-bit) × 3 win. Use value = window+1 in slen1 part, value=4
        // in slen2 part (encodable in both 1 and 3 bits? 4 needs 3 bits;
        // for slen1=1 use win&1).
        let mut w2 = BitWriter::new();
        for _gr in 0..2 {
            for _sfb in 0..6 {
                for win in 0..3u32 {
                    w2.put(win & 1, 1); // 0,1,0
                }
            }
            for _sfb in 6..12 {
                for win in 0..3u32 {
                    w2.put(win + 2, 3); // 2,3,4
                }
            }
        }
        let md = w2.finish();
        let fsf = decode_scalefactors(&hdr, &si, &md).unwrap();
        for gr in 0..2 {
            let g = &fsf.granules[gr][0];
            // long array all zero for pure short.
            assert_eq!(g.long, [0u8; LONG_SFB]);
            for sfb in 0..6 {
                assert_eq!(g.short[sfb], [0, 1, 0], "gr{gr} sfb{sfb} slen1");
            }
            for sfb in 6..12 {
                assert_eq!(g.short[sfb], [2, 3, 4], "gr{gr} sfb{sfb} slen2");
            }
            assert!(!g.preflag, "short blocks never set preflag");
        }
    }

    #[test]
    fn mpeg1_short_block_mixed() {
        // Mixed short block: scalefac_l sfb 0..8 at slen1, then
        // scalefac_s sfb 3..12 (sfb 3..6 at slen1, sfb 6..12 at slen2)
        // over 3 windows. scalefac_compress=5 => (slen1, slen2)=(1,1).
        let hdr = parse_header(&mpeg1_header(0b11)).unwrap();
        let mut w = BitWriter::new();
        w.put(0, 9);
        w.put(0, 5);
        for _ in 0..4 {
            w.put_bool(false);
        }
        for _ in 0..2 {
            w.put(300, 12);
            w.put(0, 9);
            w.put(190, 8);
            w.put(5, 4); // (slen1,slen2)=(1,1)
            w.put_bool(true);
            w.put(2, 2); // short
            w.put_bool(true); // mixed
            for _ in 0..2 {
                w.put(0, 5);
            }
            for _ in 0..3 {
                w.put(0, 3);
            }
            w.put_bool(false);
            w.put_bool(false);
            w.put_bool(false);
        }
        let si = parse_side_info(&hdr, &w.finish()).unwrap();
        assert!(si.granules[0][0].mixed_block_flag);

        // main data per granule: 8 long-window scalefactors (1-bit),
        // then 9 short bands (3..12) × 3 windows (1-bit each) = 8 + 27
        // = 35 entries. Use 1 for the long part, alternating for short.
        let mut w2 = BitWriter::new();
        for _gr in 0..2 {
            for _ in 0..8 {
                w2.put(1, 1); // long-window scalefactors
            }
            for sfb in 3..12u32 {
                for win in 0..3u32 {
                    w2.put((sfb + win) & 1, 1);
                }
            }
        }
        let md = w2.finish();
        let fsf = decode_scalefactors(&hdr, &si, &md).unwrap();
        for gr in 0..2 {
            let g = &fsf.granules[gr][0];
            // First 8 long bands are 1; bands 8..21 stay 0.
            for sfb in 0..8 {
                assert_eq!(g.long[sfb], 1, "gr{gr} long sfb{sfb}");
            }
            for sfb in 8..21 {
                assert_eq!(g.long[sfb], 0, "gr{gr} long sfb{sfb} untouched");
            }
            // short bands 0..3 untouched (mixed uses long there).
            for sfb in 0..3 {
                assert_eq!(g.short[sfb], [0, 0, 0], "gr{gr} short sfb{sfb}");
            }
            for sfb in 3..12 {
                for win in 0..3 {
                    let expect = ((sfb as u32 + win as u32) & 1) as u8;
                    assert_eq!(g.short[sfb][win], expect, "gr{gr} short sfb{sfb} win{win}");
                }
            }
        }
    }

    // ---- LSF scalefactor parameter derivation ----

    #[test]
    fn lsf_params_range0_long() {
        // scalefac_compress < 400, long block, non-intensity.
        // sc = 0 => all slen 0, nr_of_sfb = [6,5,5,5], preflag 0.
        let p = lsf_scale_params(0, BlockType::Long, false, false);
        assert_eq!(p.slen, [0, 0, 0, 0]);
        assert_eq!(p.nr_of_sfb, [6, 5, 5, 5]);
        assert!(!p.preflag);
        // sc = 0x1FF? out of range0. Use sc that exercises divisions:
        // sc = 4*16 + 5 = 69 => sc>>4 = 4 => slen1 = 4/5 = 0,
        // slen2 = 4%5 = 4; sc%16 = 5 => slen3 = 5>>2 = 1; slen4 = 5%4=1.
        let p2 = lsf_scale_params(69, BlockType::Long, false, false);
        assert_eq!(p2.slen, [0, 4, 1, 1]);
        assert_eq!(p2.nr_of_sfb, [6, 5, 5, 5]);
    }

    #[test]
    fn lsf_params_range0_short_rows() {
        // Short !mixed => row 1 => nr_of_sfb [9,9,9,9].
        let p = lsf_scale_params(0, BlockType::Short, false, false);
        assert_eq!(p.nr_of_sfb, [9, 9, 9, 9]);
        // Short mixed => row 2 => [6,9,9,9].
        let p = lsf_scale_params(0, BlockType::Short, true, false);
        assert_eq!(p.nr_of_sfb, [6, 9, 9, 9]);
    }

    #[test]
    fn lsf_params_range1() {
        // 400 <= sc < 500. sc = 400 => v=0 => all slen 0;
        // nr_of_sfb long = [6,5,7,3], preflag 0.
        let p = lsf_scale_params(400, BlockType::Long, false, false);
        assert_eq!(p.slen, [0, 0, 0, 0]);
        assert_eq!(p.nr_of_sfb, [6, 5, 7, 3]);
        assert!(!p.preflag);
        // sc=499 => v=99 => v>>2=24 => slen1=24/5=4, slen2=24%5=4;
        // v%4 = 3 => slen3=3.
        let p2 = lsf_scale_params(499, BlockType::Long, false, false);
        assert_eq!(p2.slen, [4, 4, 3, 0]);
    }

    #[test]
    fn lsf_params_range2_sets_preflag() {
        // 500 <= sc < 512 => preflag 1, slen3=slen4=0.
        // sc=500 => v=0 => slen [0,0,0,0]; nr_of_sfb long [11,10,0,0].
        let p = lsf_scale_params(500, BlockType::Long, false, false);
        assert_eq!(p.slen, [0, 0, 0, 0]);
        assert_eq!(p.nr_of_sfb, [11, 10, 0, 0]);
        assert!(p.preflag);
        // Short !mixed => [18,18,0,0]; short mixed => [15,18,0,0].
        assert_eq!(
            lsf_scale_params(500, BlockType::Short, false, false).nr_of_sfb,
            [18, 18, 0, 0]
        );
        assert_eq!(
            lsf_scale_params(500, BlockType::Short, true, false).nr_of_sfb,
            [15, 18, 0, 0]
        );
        // sc=511 => v=11 => slen1=11/3=3, slen2=11%3=2.
        let p2 = lsf_scale_params(511, BlockType::Long, false, false);
        assert_eq!(p2.slen, [3, 2, 0, 0]);
        assert!(p2.preflag);
    }

    #[test]
    fn lsf_params_intensity_right_channel() {
        // Intensity right channel uses int_scalefac_compress = sc>>1
        // and intensity_scale = sc%2.
        // sc=361 => intensity_scale=1, isc=180 => range [180,244).
        // v = isc-180 = 0 => slen all 0; nr_of_sfb long [6,6,6,3].
        let p = lsf_scale_params(361, BlockType::Long, false, true);
        assert!(p.intensity_scale);
        assert_eq!(p.slen, [0, 0, 0, 0]);
        assert_eq!(p.nr_of_sfb, [6, 6, 6, 3]);
        // sc=0 => intensity_scale=0, isc=0 => range <180.
        // slen all 0; nr_of_sfb long [7,7,7,0].
        let p2 = lsf_scale_params(0, BlockType::Long, false, true);
        assert!(!p2.intensity_scale);
        assert_eq!(p2.nr_of_sfb, [7, 7, 7, 0]);
        // sc=489 => isc=244 => intensity_scale=1; range [244,255].
        // v=0 => slen 0; nr_of_sfb long [8,8,5,0].
        let p3 = lsf_scale_params(489, BlockType::Long, false, true);
        assert!(p3.intensity_scale);
        assert_eq!(p3.nr_of_sfb, [8, 8, 5, 0]);
    }

    // ---- LSF scalefactor read ----

    /// Build an LSF mono side-info block (long block) with a given
    /// scalefac_compress, parse it, and return header + SideInfo.
    fn lsf_mono_long_sideinfo(sc: u32) -> (Mp3FrameHeader, SideInfo) {
        let hdr = parse_header(&lsf_header(0b11, 0)).unwrap();
        let mut w = BitWriter::new();
        w.put(0, 8); // main_data_begin (8 bit LSF)
        w.put_bool(false); // private_bits (1 bit mono)
        w.put(200, 12); // part2_3_length
        w.put(50, 9); // big_values
        w.put(180, 8); // global_gain
        w.put(sc, 9); // scalefac_compress (9 bit)
        w.put_bool(false); // window_switching_flag
        for _ in 0..3 {
            w.put(0, 5);
        }
        w.put(0, 4);
        w.put(0, 3);
        // no preflag bit in LSF
        w.put_bool(false); // scalefac_scale
        w.put_bool(false); // count1table_select
        let si = parse_side_info(&hdr, &w.finish()).unwrap();
        (hdr, si)
    }

    #[test]
    fn lsf_long_reads_four_partitions() {
        // scalefac_compress chosen so all four slens are non-zero and
        // distinct: range0, sc such that slen = [s1,s2,s3,s4].
        // sc = (sc>>4)*16 + (sc%16). Want slen1=1 => sc>>4 in 5..9.
        // slen1=(sc>>4)/5, slen2=(sc>>4)%5, slen3=(sc%16)>>2, slen4=sc%4.
        // Pick sc>>4 = 6 => slen1=1, slen2=1. sc%16 = 0b1110 = 14 =>
        // slen3 = 14>>2 = 3, slen4 = 14%4 = 2. So sc = 6*16+14 = 110.
        let (hdr, si) = lsf_mono_long_sideinfo(110);
        let p = lsf_scale_params(110, BlockType::Long, false, false);
        assert_eq!(p.slen, [1, 1, 3, 2]);
        assert_eq!(p.nr_of_sfb, [6, 5, 5, 5]); // sums to 21

        // main data: partition1 6 bands @1bit (=1), partition2 5 @1bit
        // (=0), partition3 5 @3bit (=5), partition4 5 @2bit (=3).
        let mut w = BitWriter::new();
        for _ in 0..6 {
            w.put(1, 1);
        }
        for _ in 0..5 {
            w.put(0, 1);
        }
        for _ in 0..5 {
            w.put(5, 3);
        }
        for _ in 0..5 {
            w.put(3, 2);
        }
        let md = w.finish();
        let fsf = decode_scalefactors(&hdr, &si, &md).unwrap();
        assert_eq!(fsf.granule_count, 1);
        let g = &fsf.granules[0][0];
        for sfb in 0..6 {
            assert_eq!(g.long[sfb], 1, "p1 sfb{sfb}");
        }
        for sfb in 6..11 {
            assert_eq!(g.long[sfb], 0, "p2 sfb{sfb}");
        }
        for sfb in 11..16 {
            assert_eq!(g.long[sfb], 5, "p3 sfb{sfb}");
        }
        for sfb in 16..21 {
            assert_eq!(g.long[sfb], 3, "p4 sfb{sfb}");
        }
        assert!(!g.preflag);
    }

    #[test]
    fn lsf_long_preflag_range() {
        // sc in [500,512) sets preflag and uses [11,10,0,0] partitions.
        // sc=511 => slen [3,2,0,0].
        let (hdr, si) = lsf_mono_long_sideinfo(511);
        let mut w = BitWriter::new();
        // partition1: 11 bands @ 3 bits (=7); partition2: 10 @ 2 bits
        // (=2). slen3,slen4 zero => no bits, no bands.
        for _ in 0..11 {
            w.put(7, 3);
        }
        for _ in 0..10 {
            w.put(2, 2);
        }
        let md = w.finish();
        let fsf = decode_scalefactors(&hdr, &si, &md).unwrap();
        let g = &fsf.granules[0][0];
        assert!(g.preflag);
        for sfb in 0..11 {
            assert_eq!(g.long[sfb], 7, "p1 sfb{sfb}");
        }
        for sfb in 11..21 {
            assert_eq!(g.long[sfb], 2, "p2 sfb{sfb}");
        }
    }

    #[test]
    fn lsf_intensity_right_channel_decode() {
        // Joint-stereo intensity LSF: right channel (ch==1) decodes via
        // int_scalefac_compress. Left channel uses the normal path.
        // mode_extension '01' => intensity_stereo on.
        let hdr = parse_header(&lsf_header(0b01, 0b01)).unwrap();
        assert_eq!(hdr.mode, ChannelMode::JointStereo);
        assert!(hdr.mode_extension.intensity_stereo);

        // Left channel sc = 0 (range0 long: slen all 0, nr [6,5,5,5]).
        // Right channel sc = 0 => intensity path isc=0 => nr [7,7,7,0],
        // slen all 0, intensity_scale=0.
        let mut w = BitWriter::new();
        w.put(0, 8); // main_data_begin
        w.put(0, 2); // private_bits (2 bit stereo)
        for _ch in 0..2 {
            w.put(200, 12);
            w.put(50, 9);
            w.put(180, 8);
            w.put(0, 9); // scalefac_compress = 0
            w.put_bool(false);
            for _ in 0..3 {
                w.put(0, 5);
            }
            w.put(0, 4);
            w.put(0, 3);
            w.put_bool(false);
            w.put_bool(false);
        }
        let si = parse_side_info(&hdr, &w.finish()).unwrap();
        // With slen all zero, no main-data bits are consumed for either
        // channel; an empty (but non-exhausted-on-read) buffer works.
        let md = [0u8; 4];
        let fsf = decode_scalefactors(&hdr, &si, &md).unwrap();
        assert_eq!(fsf.channels, 2);
        // Both channels all-zero scalefactors; right channel
        // intensity_scale derived = false (sc%2==0).
        assert_eq!(fsf.granules[0][0].long, [0u8; LONG_SFB]);
        assert_eq!(fsf.granules[0][1].long, [0u8; LONG_SFB]);
        assert!(!fsf.granules[0][1].intensity_scale);

        // Now flip right-channel sc to an odd value in the intensity
        // range to exercise intensity_scale = 1.
        let mut w2 = BitWriter::new();
        w2.put(0, 8);
        w2.put(0, 2);
        // left ch sc=0
        w2.put(200, 12);
        w2.put(50, 9);
        w2.put(180, 8);
        w2.put(0, 9);
        w2.put_bool(false);
        for _ in 0..3 {
            w2.put(0, 5);
        }
        w2.put(0, 4);
        w2.put(0, 3);
        w2.put_bool(false);
        w2.put_bool(false);
        // right ch sc=361 => intensity_scale=1, isc=180, slen all 0.
        w2.put(200, 12);
        w2.put(50, 9);
        w2.put(180, 8);
        w2.put(361, 9);
        w2.put_bool(false);
        for _ in 0..3 {
            w2.put(0, 5);
        }
        w2.put(0, 4);
        w2.put(0, 3);
        w2.put_bool(false);
        w2.put_bool(false);
        let si2 = parse_side_info(&hdr, &w2.finish()).unwrap();
        let fsf2 = decode_scalefactors(&hdr, &si2, &md).unwrap();
        assert!(fsf2.granules[0][1].intensity_scale);
    }

    #[test]
    fn lsf_short_pure_block() {
        // Pure short LSF: scalefac_s sfb 0..12 × 3 windows = 36 entries.
        // Range0 short => nr_of_sfb [9,9,9,9] (=36). slen from sc.
        // sc = 6*16 + 14 = 110 => slen [1,1,3,2].
        let hdr = parse_header(&lsf_header(0b11, 0)).unwrap();
        let mut w = BitWriter::new();
        w.put(0, 8);
        w.put_bool(false);
        w.put(300, 12);
        w.put(0, 9);
        w.put(190, 8);
        w.put(110, 9); // sc => slen [1,1,3,2]
        w.put_bool(true); // window_switching
        w.put(2, 2); // short
        w.put_bool(false); // not mixed
        for _ in 0..2 {
            w.put(0, 5);
        }
        for _ in 0..3 {
            w.put(0, 3);
        }
        w.put_bool(false);
        w.put_bool(false);
        let si = parse_side_info(&hdr, &w.finish()).unwrap();
        let p = lsf_scale_params(110, BlockType::Short, false, false);
        assert_eq!(p.nr_of_sfb, [9, 9, 9, 9]);
        assert_eq!(p.slen, [1, 1, 3, 2]);

        // main data: 9 entries @1, 9 @1(=0), 9 @3(=5), 9 @2(=3). These
        // map (sfb,window) in order: entries 0..36 => sfb=e/3, win=e%3.
        let mut w2 = BitWriter::new();
        for _ in 0..9 {
            w2.put(1, 1);
        }
        for _ in 0..9 {
            w2.put(0, 1);
        }
        for _ in 0..9 {
            w2.put(5, 3);
        }
        for _ in 0..9 {
            w2.put(3, 2);
        }
        let md = w2.finish();
        let fsf = decode_scalefactors(&hdr, &si, &md).unwrap();
        let g = &fsf.granules[0][0];
        // long all zero for pure short.
        assert_eq!(g.long, [0u8; LONG_SFB]);
        // Verify a few representative (sfb,window) positions.
        for e in 0..36usize {
            let sfb = e / 3;
            let win = e % 3;
            let expect = match e {
                0..=8 => 1,
                9..=17 => 0,
                18..=26 => 5,
                _ => 3,
            };
            assert_eq!(g.short[sfb][win], expect, "entry{e} sfb{sfb} win{win}");
        }
    }

    #[test]
    fn out_of_data_error() {
        // A long-block MPEG-1 frame whose main data is too short to hold
        // all scalefactors must error. scalefac_compress=13 => (3,3):
        // 21*3 + 21*3 = 126 bits = 16 bytes needed; supply 1 byte.
        let (hdr, si) = mpeg1_mono_long_sideinfo(13, 13, [false; 4]);
        let md = [0xFFu8; 1];
        assert_eq!(
            decode_scalefactors(&hdr, &si, &md),
            Err(ScaleFactorError::OutOfData)
        );
    }

    #[test]
    fn lsf_part2_length_matches_formula() {
        // Cross-check: part2_length = sum(nr_of_sfb[i]*slen[i]) should
        // equal the bits consumed by the reader for a long block.
        // sc=110 long => slen [1,1,3,2], nr [6,5,5,5].
        let p = lsf_scale_params(110, BlockType::Long, false, false);
        let part2: usize = (0..4)
            .map(|i| p.nr_of_sfb[i] as usize * p.slen[i] as usize)
            .sum();
        // = 6*1 + 5*1 + 5*3 + 5*2 per the §2.4.3.4 part2_length formula.
        assert_eq!(part2, 36);

        let (hdr, si) = lsf_mono_long_sideinfo(110);
        let mut w = BitWriter::new();
        for _ in 0..36 {
            w.put(1, 1);
        }
        let md = w.finish();
        let mut r = MainDataReader::new(&md);
        let _ = read_lsf_channel(&mut r, &si.granules[0][0], false);
        assert_eq!(r.bit_pos(), part2);
        let _ = (hdr, si);
    }

    #[test]
    fn mpeg1_part2_length_matches_formula() {
        // Long block: part2_length = 11*slen1 + 10*slen2.
        // sc=7 => (1,3) => 11*1 + 10*3 = 41 bits.
        let (hdr, si) = mpeg1_mono_long_sideinfo(7, 7, [false; 4]);
        let mut w = BitWriter::new();
        for _ in 0..11 {
            w.put(1, 1);
        }
        for _ in 0..10 {
            w.put(5, 3);
        }
        let md = w.finish();
        let mut r = MainDataReader::new(&md);
        let _ = read_mpeg1_granule_channel(&mut r, &si.granules[0][0], &si.scfsi[0], 0, None);
        // = 11*slen1 + 10*slen2 = 11*1 + 10*3 per the §2.4.3.4.5 formula.
        assert_eq!(r.bit_pos(), 41);
        let _ = hdr;
    }
}
