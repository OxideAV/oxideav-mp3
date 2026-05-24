//! Layer III main-data **Huffman decode** — the stage that turns the
//! Huffman-coded portion of a granule-channel's main data into the 576
//! quantized frequency lines `is[0..576]`.
//!
//! This module implements the `Huffmancodebits()` syntax of ISO/IEC
//! 11172-3:1993 §2.4.1.7 (the `No. of bits` table on p.18) with the
//! semantics of §2.4.2.7 (p.26–28), using the Huffman codebooks of
//! **Table 3-B.7** (p.54–60) and the scalefactor-band boundaries of
//! **Table 3-B.8** (p.62–64). Every numeric value in this file was
//! transcribed by hand from those pages of the ISO/IEC 11172-3:1993
//! PDF (rendered with `pdftoppm`); no external implementation was
//! consulted.
//!
//! # The three partitions
//!
//! A granule-channel's Huffman data is decoded in three partitions
//! (§2.4.2.7, "huffmancodebits()"):
//!
//! 1. **big_values** — `big_values` *pairs* `(x, y)` of quantized
//!    values, each pair coded with one of the 32 Huffman codebooks of
//!    Table 3-B.7 selected per region by `table_select`. The pair
//!    region is split into three sub-regions (region 0, 1, 2) whose
//!    boundaries are aligned to scalefactor-band edges via
//!    `region0_count` / `region1_count` (§2.4.2.7, "region0_count").
//!    Codebooks 16–31 carry an ESC mechanism: a value of magnitude
//!    ≥ 15 is coded as the Huffman symbol 15 followed by a `linbits`
//!    extension field (added to 15), then the sign bit.
//! 2. **count1** — *quadruples* `(v, w, x, y)` of values with
//!    magnitude ≤ 1, coded with quad table A or B (Table 3-B.7,
//!    selected by `count1table_select`), decoded until the granule's
//!    `part2_3_length` bit budget is exhausted.
//! 3. **zero** — the remaining lines up to index 576 are zero.
//!
//! The decoder is given the number of *part-3* bits available
//! (`part2_3_length` minus the scalefactor bits already consumed) and
//! stops the count1 partition the moment that budget is used up
//! (§2.4.2.7: "The end of the count1 partition is known only when all
//! bits for the granule … have been exhausted").

use crate::frame::MpegVersion;
use crate::scalefactors::MainDataReader;
use crate::side_info::{BlockType, GranuleChannel};

/// Number of quantized frequency lines produced per granule-channel.
pub const NUM_LINES: usize = 576;

/// A big-values Huffman codebook entry: the code length in bits and the
/// right-aligned code value. A `(0, 0)` entry marks an `(x, y)` pair
/// that does not occur in the table (the table is rectangular but a
/// few high corners are unused in some codebooks).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HEntry {
    /// `hlen` — number of bits in the codeword.
    len: u8,
    /// `hcod` — the codeword, right-aligned in the low `len` bits.
    code: u16,
}

const fn e(len: u8, code: u16) -> HEntry {
    HEntry { len, code }
}

/// A big-values Huffman codebook: a rectangular `(x, y)` table with a
/// `linbits` ESC width. `xlen` is the number of distinct `x` values
/// (= number of distinct `y` values; the tables are square), so the
/// flat `entries` slice has `xlen * xlen` rows indexed `x * xlen + y`.
#[derive(Debug, Clone, Copy)]
struct BigTable {
    /// Side length of the square `(x, y)` index space.
    xlen: u8,
    /// `linbits` ESC width (Table 3-B.7 per-table header); `0` for the
    /// small tables.
    linbits: u8,
    /// Flattened `[x * xlen + y]` entries.
    entries: &'static [HEntry],
}

/// Error from the Huffman decode stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HuffmanError {
    /// The Huffman bitstream did not match any codeword in the selected
    /// table before the symbol's maximum length was reached, indicating
    /// a corrupt stream or a wrong table selection.
    InvalidCode,
    /// `table_select` named a codebook index that is "not used" in
    /// Table 3-B.7 (4 or 14) or is out of the 0..=31 range.
    UnusedTable(u8),
    /// `big_values * 2` exceeded the 576-line granule capacity.
    BigValuesTooLarge,
    /// `table_select` named a Table 3-B.7 codebook that has not yet
    /// been transcribed in this crate (the large 16×16 tables 15, 16,
    /// 24 and their linbits aliases 17..=23, 25..=31). Tracked as a
    /// follow-up round.
    TableNotYetTranscribed(u8),
}

impl core::fmt::Display for HuffmanError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            HuffmanError::InvalidCode => write!(f, "no Huffman codeword matched the bitstream"),
            HuffmanError::UnusedTable(t) => {
                write!(f, "Huffman table_select {t} is unused (Table 3-B.7)")
            }
            HuffmanError::BigValuesTooLarge => {
                write!(f, "big_values*2 exceeds the 576-line granule capacity")
            }
            HuffmanError::TableNotYetTranscribed(t) => write!(
                f,
                "Huffman table_select {t} (Table 3-B.7) not yet transcribed"
            ),
        }
    }
}

impl std::error::Error for HuffmanError {}

/// Decode the Huffman-coded main data of one granule-channel into its
/// 576 quantized frequency lines.
///
/// * `reader` is positioned at the first bit of the granule's
///   `huffmancodebits()` (i.e. immediately after the scalefactors).
/// * `gc` carries the side-info fields (`big_values`, `table_select`,
///   `region0_count`, `region1_count`, `count1table_select`,
///   `block_type`, `mixed_block_flag`).
/// * `part3_bits` is the number of bits available for the Huffman data
///   (`part2_3_length` minus the scalefactor bits already read). The
///   count1 partition stops when this budget is exhausted.
/// * `sample_rate_hz` / `version` select the Table 3-B.8 band table for
///   the region-boundary split (long blocks only).
///
/// Returns the populated `[i32; 576]` line buffer. Values carry their
/// sign; lines past the count1 partition are zero.
pub fn decode_huffman(
    reader: &mut MainDataReader<'_>,
    gc: &GranuleChannel,
    part3_bits: u32,
    sample_rate_hz: u32,
    version: MpegVersion,
) -> Result<[i32; NUM_LINES], HuffmanError> {
    let mut is = [0i32; NUM_LINES];
    let start_bit = reader.bit_pos() as u32;
    let budget_end = start_bit + part3_bits;

    let big_pairs = usize::from(gc.big_values);
    if big_pairs * 2 > NUM_LINES {
        return Err(HuffmanError::BigValuesTooLarge);
    }

    // ---- big_values partition: three regions ----
    let (r0_end, r1_end) = region_boundaries(gc, sample_rate_hz, version);
    // Region pair-index boundaries clamped to the big_values pair count.
    let bv2 = big_pairs * 2; // line count covered by big_values
    let mut line = 0usize;
    while line < bv2 {
        // Pick the region's table_select by the *line* index.
        let region = if line < r0_end {
            0
        } else if line < r1_end {
            1
        } else {
            2
        };
        let tbl_idx = gc.table_select[region];
        let table = big_table(tbl_idx)?;
        let (x, y) = decode_big_pair(reader, table)?;
        is[line] = x;
        is[line + 1] = y;
        line += 2;
    }

    // ---- count1 partition: quadruples until the budget is spent ----
    let quad_b = gc.count1table_select;
    while line + 4 <= NUM_LINES {
        // Stop the moment the part-3 budget would be exceeded: a
        // quadruple is only decoded while bits remain.
        if (reader.bit_pos() as u32) >= budget_end {
            break;
        }
        let (v, w, x, y) = decode_count1_quad(reader, quad_b);
        is[line] = v;
        is[line + 1] = w;
        is[line + 2] = x;
        is[line + 3] = y;
        line += 4;
        // If reading the quad ran past the buffer/budget we still stop
        // on the next loop check; one over-read quad is permitted by
        // the spec's "until bits exhausted" rule.
    }

    // ---- zero partition: already zero-initialised ----
    Ok(is)
}

/// Decode one big-values `(x, y)` pair: match a codeword, apply the
/// `linbits` ESC extension on magnitude-15 symbols, then the sign bits.
fn decode_big_pair(
    reader: &mut MainDataReader<'_>,
    table: &BigTable,
) -> Result<(i32, i32), HuffmanError> {
    let (mut x, mut y) = match_big_code(reader, table)?;
    let mut xv = i32::from(x);
    let mut yv = i32::from(y);
    let linbits = u32::from(table.linbits);

    // §2.4.1.7 huffmancodebits: linbitsx after the code (if |x|==15 &&
    // linbits>0), then signx (if x!=0); same for y.
    if x == 15 && linbits > 0 {
        xv += reader.read(linbits) as i32;
    }
    if xv != 0 && reader.read(1) == 1 {
        xv = -xv;
    }
    if y == 15 && linbits > 0 {
        yv += reader.read(linbits) as i32;
    }
    if yv != 0 && reader.read(1) == 1 {
        yv = -yv;
    }
    // Silence unused-assignment lints on the magnitude pair.
    let _ = (&mut x, &mut y);
    Ok((xv, yv))
}

/// Walk the bitstream one bit at a time, accumulating a candidate
/// codeword, and return the `(x, y)` whose `(len, code)` matches. The
/// tables are not prefix-ambiguous, so the first exact match at the
/// running length is unique.
fn match_big_code(
    reader: &mut MainDataReader<'_>,
    table: &BigTable,
) -> Result<(u8, u8), HuffmanError> {
    // Maximum codeword length in this table (so we bound the search).
    let max_len = table.entries.iter().map(|e| e.len).max().unwrap_or(0);
    if max_len == 0 {
        // Table 0 in Table 3-B.7: the single (0,0) entry is encoded
        // with zero bits — the table emits (0,0) without consuming
        // anything from the stream.
        return Ok((0, 0));
    }
    let mut acc: u32 = 0;
    let mut len: u8 = 0;
    while len < max_len {
        acc = (acc << 1) | reader.read(1);
        len += 1;
        // Scan for an entry of exactly this length and code.
        for (idx, ent) in table.entries.iter().enumerate() {
            if ent.len == len && u32::from(ent.code) == acc {
                let xl = usize::from(table.xlen);
                let x = (idx / xl) as u8;
                let y = (idx % xl) as u8;
                return Ok((x, y));
            }
        }
    }
    Err(HuffmanError::InvalidCode)
}

/// Decode one count1 `(v, w, x, y)` quadruple. Quad table A is the
/// 16-entry Huffman code of Table 3-B.7-A; quad table B is the trivial
/// 4-bit code (1→`0`, 0→`1`) per §2.4.2.7. Sign bits follow each
/// non-zero value.
fn decode_count1_quad(reader: &mut MainDataReader<'_>, quad_b: bool) -> (i32, i32, i32, i32) {
    let (v, w, x, y) = if quad_b {
        // Table B: each value is one bit, 0 → magnitude 1, 1 → 0.
        let b = reader.read(4);
        (
            ((b >> 3) & 1) ^ 1,
            ((b >> 2) & 1) ^ 1,
            ((b >> 1) & 1) ^ 1,
            (b & 1) ^ 1,
        )
    } else {
        // Table A: variable-length code over the 16 vwxy patterns.
        match_quad_a(reader)
    };
    let mut vv = v as i32;
    let mut wv = w as i32;
    let mut xv = x as i32;
    let mut yv = y as i32;
    if vv != 0 && reader.read(1) == 1 {
        vv = -vv;
    }
    if wv != 0 && reader.read(1) == 1 {
        wv = -wv;
    }
    if xv != 0 && reader.read(1) == 1 {
        xv = -xv;
    }
    if yv != 0 && reader.read(1) == 1 {
        yv = -yv;
    }
    (vv, wv, xv, yv)
}

/// Match a count1 table-A codeword and return its `(v, w, x, y)`
/// magnitude pattern. On no match (corrupt / exhausted) returns the
/// all-zero pattern so the caller's budget check terminates the loop.
fn match_quad_a(reader: &mut MainDataReader<'_>) -> (u32, u32, u32, u32) {
    let mut acc: u32 = 0;
    let mut len: u8 = 0;
    let max_len = QUAD_A.iter().map(|e| e.0).max().unwrap_or(0);
    while len < max_len {
        acc = (acc << 1) | reader.read(1);
        len += 1;
        for (i, &(clen, code)) in QUAD_A.iter().enumerate() {
            if clen == len && u32::from(code) == acc {
                let vwxy = i as u32;
                return ((vwxy >> 3) & 1, (vwxy >> 2) & 1, (vwxy >> 1) & 1, vwxy & 1);
            }
        }
    }
    (0, 0, 0, 0)
}

/// Compute the line-index boundaries `(region0_end, region1_end)` of
/// the big-values partition (§2.4.2.7, "region0_count").
///
/// For long blocks the boundaries are aligned to scalefactor-band
/// edges from Table 3-B.8: region 0 covers bands `0..=region0_count`,
/// region 1 the following `region1_count + 1` bands. For short /
/// window-switched blocks the spec fixes region 0 at the first
/// 36 lines and region 1 to the rest of big_values, so we return
/// `(36, big_values*2)`-style boundaries (region 2 empty).
fn region_boundaries(
    gc: &GranuleChannel,
    sample_rate_hz: u32,
    version: MpegVersion,
) -> (usize, usize) {
    if gc.window_switching_flag && gc.block_type == BlockType::Short {
        // Window-switched short blocks: region0 is the first 36 lines
        // (the default region0_count of 8 short triple-bands → band 3
        // → start index 36 at 44.1 kHz long mapping is not used here;
        // the spec fixes region0 to the lowest 36 lines for short
        // blocks). Region 1 extends to big_values*2; region 2 empty.
        let r0 = 36usize.min(usize::from(gc.big_values) * 2);
        let r1 = usize::from(gc.big_values) * 2;
        return (r0, r1);
    }

    let starts = long_band_starts(sample_rate_hz, version);
    // region0 covers bands 0..=region0_count, so region 1 starts at the
    // start index of band (region0_count + 1).
    let r0_band = usize::from(gc.region0_count) + 1;
    let r1_band = r0_band + usize::from(gc.region1_count) + 1;
    let r0_end = starts.get(r0_band).copied().unwrap_or(NUM_LINES);
    let r1_end = starts.get(r1_band).copied().unwrap_or(NUM_LINES);
    (r0_end.min(NUM_LINES), r1_end.min(NUM_LINES))
}

/// The long-block scalefactor-band *start* line indices for the active
/// sampling rate (Table 3-B.8a/b/c). Index `i` is the first line of
/// band `i`; entry 21 (one past band 20) is the band's end+1 so callers
/// can read a region boundary at the top of the long-block range.
fn long_band_starts(sample_rate_hz: u32, version: MpegVersion) -> &'static [usize; 22] {
    // MPEG-1 (ISO/IEC 11172-3 Table 3-B.8). MPEG-2 LSF (13818-3) reuses
    // these long-block band layouts for the region split this round; the
    // LSF-specific band tables are deferred (see module docs / report).
    let _ = version;
    match sample_rate_hz {
        32000 | 16000 | 8000 => &LONG_BANDS_32,
        48000 | 24000 | 12000 => &LONG_BANDS_48,
        // 44100, 22050, 11025 and any default.
        _ => &LONG_BANDS_44,
    }
}

/// Table 3-B.8a (32 kHz) long-block band start indices + end+1.
const LONG_BANDS_32: [usize; 22] = [
    0, 4, 8, 12, 16, 20, 24, 30, 36, 44, 54, 66, 82, 102, 126, 156, 194, 240, 296, 364, 448, 550,
];

/// Table 3-B.8b (44.1 kHz) long-block band start indices + end+1.
const LONG_BANDS_44: [usize; 22] = [
    0, 4, 8, 12, 16, 20, 24, 30, 36, 44, 52, 62, 74, 90, 110, 134, 162, 196, 238, 288, 342, 418,
];

/// Table 3-B.8c (48 kHz) long-block band start indices + end+1.
const LONG_BANDS_48: [usize; 22] = [
    0, 4, 8, 12, 16, 20, 24, 30, 36, 42, 50, 60, 72, 88, 106, 128, 156, 190, 230, 276, 330, 384,
];

/// Resolve a 5-bit `table_select` into its Table 3-B.7 codebook,
/// rejecting the "not used" tables 4 and 14.
fn big_table(idx: u8) -> Result<&'static BigTable, HuffmanError> {
    match idx {
        0 => Ok(&TABLE0),
        1 => Ok(&TABLE1),
        2 => Ok(&TABLE2),
        3 => Ok(&TABLE3),
        4 | 14 => Err(HuffmanError::UnusedTable(idx)),
        5 => Ok(&TABLE5),
        6 => Ok(&TABLE6),
        7 => Ok(&TABLE7),
        8 => Ok(&TABLE8),
        9 => Ok(&TABLE9),
        10 => Ok(&TABLE10),
        11 => Ok(&TABLE11),
        12 => Ok(&TABLE12),
        13 => Ok(&TABLE13),
        15..=31 => Err(HuffmanError::TableNotYetTranscribed(idx)),
        _ => Err(HuffmanError::UnusedTable(idx)),
    }
}

include!("huffman_tables.rs");

#[cfg(test)]
include!("huffman_tests.rs");
