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
//! PDF (rendered with `pdftoppm`).
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
use crate::scalefactors::{MainDataReader, MainDataWriter};
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
    /// **Deprecated / no longer produced.** Earlier rounds returned this
    /// for the large 16×16 codebooks 15, 16, 24 and their linbits aliases
    /// 17..=23 / 25..=31 before they were transcribed. All of Table 3-B.7
    /// (0..=31 minus the unused 4 / 14) is now implemented, so the decoder
    /// never constructs this variant; it is retained only for API
    /// stability.
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
        let (x, y) = decode_big_pair(reader, tbl_idx, table)?;
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

/// A direct-mapped **canonical prefix decode table** for one big-values
/// codebook.
///
/// Layer III Huffman codebooks are prefix codes, so the next `bits`
/// (`= max_len`, the table's longest codeword) uniquely identify the
/// codeword: every stream prefix that starts with codeword `c` of length
/// `len` lands in the `2^(bits-len)` contiguous slots whose top `len`
/// bits are `c`. `slots[prefix]` therefore records `(x, y, len)` for the
/// codeword that is a prefix of `prefix`; a `len == 0` slot is one no
/// codeword covers (a corrupt-stream prefix), matching the old scan's
/// `InvalidCode`.
///
/// Decoding then peeks `bits`, indexes once, and consumes exactly `len`
/// bits — O(1) versus the former O(entries × max_len) linear scan, and
/// bit-for-bit identical: the `(x, y)` returned and the bits consumed are
/// the same prefix code as before.
struct FastTable {
    /// Prefix width used to index `slots` (the table's `max_len`).
    bits: u8,
    /// `2^bits` slots of `(x, y, len)`; `len == 0` marks an unmatched
    /// prefix (invalid codeword).
    slots: Vec<(u8, u8, u8)>,
}

impl FastTable {
    /// Build the canonical prefix table from a codebook's flat entries.
    fn build(table: &BigTable) -> Self {
        let max_len = table.entries.iter().map(|e| e.len).max().unwrap_or(0);
        // `max_len == 0` is only Table 0's single zero-bit (0,0) entry;
        // a 1-slot table with a zero-length code decodes (0,0) consuming
        // nothing, handled by the caller before peeking.
        let bits = max_len;
        let size = 1usize << bits;
        let mut slots = vec![(0u8, 0u8, 0u8); size.max(1)];
        let xl = usize::from(table.xlen);
        for (idx, ent) in table.entries.iter().enumerate() {
            if ent.len == 0 {
                continue;
            }
            let x = (idx / xl) as u8;
            let y = (idx % xl) as u8;
            // The code occupies the top `ent.len` bits; fill every
            // lower-bit completion so any prefix starting with this code
            // resolves to it.
            let shift = bits - ent.len;
            let base = usize::from(ent.code) << shift;
            for slot in slots.iter_mut().skip(base).take(1usize << shift) {
                *slot = (x, y, ent.len);
            }
        }
        FastTable { bits, slots }
    }
}

/// Per-codebook-index (`0..32`) canonical prefix decode tables, built
/// once at first use. Indices 4 and 14 are unused (Table 3-B.7) and get
/// an empty placeholder that is never consulted (the caller resolves
/// `big_table` first, which rejects them).
static FAST_TABLES: std::sync::LazyLock<Vec<Option<FastTable>>> = std::sync::LazyLock::new(|| {
    (0u8..32)
        .map(|idx| big_table(idx).ok().map(FastTable::build))
        .collect()
});

/// Decode one big-values codeword through the canonical prefix table for
/// codebook `tbl_idx`, returning the `(x, y)` magnitude pair. Bit-exact
/// with [`match_big_code`]: same code, same bits consumed.
fn match_big_code_fast(
    reader: &mut MainDataReader<'_>,
    tbl_idx: u8,
    table: &BigTable,
) -> Result<(u8, u8), HuffmanError> {
    let fast = match FAST_TABLES
        .get(usize::from(tbl_idx))
        .and_then(|t| t.as_ref())
    {
        Some(f) => f,
        // Should be unreachable (caller resolved big_table already), but
        // fall back to the scanning matcher rather than panic.
        None => return match_big_code(reader, table),
    };
    if fast.bits == 0 {
        // Table 0: the single zero-bit (0,0) entry consumes nothing.
        return Ok((0, 0));
    }
    let prefix = reader.peek(u32::from(fast.bits)) as usize;
    let (x, y, len) = fast.slots[prefix];
    if len == 0 {
        // No codeword covers this prefix: consume the full max_len (as
        // the scanning matcher did before erroring) and report the same
        // InvalidCode.
        let _ = reader.read(u32::from(fast.bits));
        return Err(HuffmanError::InvalidCode);
    }
    // Consume exactly the codeword length.
    let _ = reader.read(u32::from(len));
    Ok((x, y))
}

/// Decode one big-values `(x, y)` pair: match a codeword, apply the
/// `linbits` ESC extension on magnitude-15 symbols, then the sign bits.
fn decode_big_pair(
    reader: &mut MainDataReader<'_>,
    tbl_idx: u8,
    table: &BigTable,
) -> Result<(i32, i32), HuffmanError> {
    let (mut x, mut y) = match_big_code_fast(reader, tbl_idx, table)?;
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
/// region 1 the following `region1_count + 1` bands. For
/// window-switched short blocks the §2.4.2.7 defaults apply
/// (`region0_count = 8`, `region1_count = 63`) with short-block band
/// counting: "in the case of short blocks, each scale factor band is
/// counted three times, once for each short window", so the nine
/// region-0 window-bands are short scalefactor bands 0..=2 across the
/// three windows and region 0 ends at interleaved line
/// `3 · short_starts[3]`; region 1 covers the rest of big_values
/// (region 2 empty).
fn region_boundaries(
    gc: &GranuleChannel,
    sample_rate_hz: u32,
    version: MpegVersion,
) -> (usize, usize) {
    if gc.window_switching_flag && gc.block_type == BlockType::Short {
        // Window-switched short blocks. Pure short: the §2.4.2.7
        // default `region0_count = 8` counts nine window-bands = short
        // sfb 0..=2 × 3 windows, i.e. region 0 ends at interleaved
        // line `3 · short_starts[3]`. Every short band table in
        // ISO/IEC 11172-3 / 13818-3 has `short_starts[3] = 12`
        // (region 0 = 36 lines — the value this branch used to
        // hardcode), but the MPEG-2.5 8 kHz Fraunhofer table has
        // `short_starts[3] = 24` (region 0 = 72 lines): the
        // band-relative form is required for deployed-decoder
        // agreement at 8 kHz (r405 observer-trace; the hardcoded 36
        // measured NCC ≈ 0.05 against two independent black-box
        // validators, while the band-relative boundary decodes in the
        // float-rounding regime).
        //
        // Mixed short blocks keep the 36-line boundary: §2.4.2.7 sets
        // `region0_count = 7` for them, and the mixed band sequence
        // opens with the long bands of the 36-line long region, so
        // the eight-band region 0 ends at the long/short split line.
        let r0_lines = if gc.mixed_block_flag {
            36usize
        } else {
            3 * short_band_starts(sample_rate_hz, version)[3]
        };
        let r0 = r0_lines.min(usize::from(gc.big_values) * 2);
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
/// sampling rate. Index `i` is the first line of band `i`; entry 21
/// (one past band 20) is the band's end+1 so callers can read a region
/// boundary at the top of the long-block range.
///
/// Delegates to [`crate::requantize::long_band_starts`] — the single
/// in-crate transcription of ISO/IEC 11172-3 Table 3-B.8 (MPEG-1
/// rates) and ISO/IEC 13818-3:1997 Table B.2 (MPEG-2 LSF rates) — so
/// the §2.4.2.7 region split and the §2.4.3.4.7 requantizer can never
/// disagree on band boundaries.
fn long_band_starts(sample_rate_hz: u32, version: MpegVersion) -> &'static [usize; 22] {
    crate::requantize::long_band_starts(sample_rate_hz, version)
}

/// The short-block per-window scalefactor-band *start* line indices
/// for the active sampling rate. Delegates to
/// [`crate::requantize::short_band_starts`] — the single in-crate
/// transcription — so the §2.4.2.7 short-block region-0 boundary and
/// the §2.4.3.4.7 requantizer can never disagree on band boundaries.
fn short_band_starts(sample_rate_hz: u32, version: MpegVersion) -> &'static [usize; 13] {
    crate::requantize::short_band_starts(sample_rate_hz, version)
}

/// Maximum unsigned magnitude that big-values codebook `idx` can encode
/// **without truncation** — i.e. the largest `|is_i|` whose round-trip
/// through `emit_big_pair` / `decode_big_pair` is bit-exact.
///
/// For the small tables 0..=15 the linbits ESC field is absent
/// (`table.linbits == 0`); a value of magnitude `≥ 15` cannot be coded at
/// all without that escape, and a magnitude `< 15` is coded directly by
/// the Huffman symbol whose index equals the magnitude. The reach is
/// therefore `xlen - 1` (= 1 for table 1, 2 for tables 2/3, 5 for tables
/// 7/8/9, 7 for tables 10/11/12, and 15 for tables 13/15). Table 0 codes
/// only the all-zero pair, so its reach is `0`.
///
/// For the large ESC tables 16..=31 the linbits field carries `(|is| -
/// 15)` in `table.linbits` bits, so the reach is `15 + (2^linbits - 1)`
/// (= 16, 18, 22, 30, 78, 270, 1038, 8206 for tables 16..=23; same
/// progression with a shift for tables 24..=31). Tables 4 and 14 are
/// "not used" in §B.7 and return `0`.
///
/// The encoder's table chooser ([`choose_best_table_for_region`]) uses
/// this to drop codebooks whose reach is less than the actual `max|is|`
/// in the range — otherwise the chooser could pick e.g. table 16
/// (`linbits=1`, reach 16) for a range with `|is| = 100`, and
/// `emit_big_pair` would silently truncate the value to its low
/// `linbits` bits at emission time.
#[must_use]
pub fn big_table_reach(idx: u8) -> u32 {
    match big_table(idx) {
        Ok(table) => {
            let xlen = u32::from(table.xlen);
            let linbits = u32::from(table.linbits);
            if linbits == 0 {
                // Small table: reach = xlen - 1 (index 0..=xlen-1).
                xlen.saturating_sub(1)
            } else {
                // ESC table: reach = 15 + (2^linbits - 1).
                15 + ((1u32 << linbits) - 1)
            }
        }
        Err(_) => 0,
    }
}

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
        15 => Ok(&TABLE15),
        16 => Ok(&TABLE16),
        17 => Ok(&TABLE17),
        18 => Ok(&TABLE18),
        19 => Ok(&TABLE19),
        20 => Ok(&TABLE20),
        21 => Ok(&TABLE21),
        22 => Ok(&TABLE22),
        23 => Ok(&TABLE23),
        24 => Ok(&TABLE24),
        25 => Ok(&TABLE25),
        26 => Ok(&TABLE26),
        27 => Ok(&TABLE27),
        28 => Ok(&TABLE28),
        29 => Ok(&TABLE29),
        30 => Ok(&TABLE30),
        31 => Ok(&TABLE31),
        _ => Err(HuffmanError::UnusedTable(idx)),
    }
}

// =====================================================================
// Forward Huffman **bit count** — the §C.1.5.4.4.5 / §C.1.5.4.4.8 count
// the encoder's inner iteration loop needs to decide whether a quantized
// `is[]` fits the available bit budget. This is the exact inverse of the
// decode path above: it reports how many bits `decode_huffman` would have
// consumed for a given `is[]`, region split and table selection.
//
// The decoder's `BigTable.len` / `QUAD_A.0` store the **codeword length
// only** (no sign or `linbits` bits), whereas the spec's `bitz` /
// `countltable` length tables "have to include the number of bits
// necessary to encode the sign bits" (§C.1.5.4.4.5, §C.1.5.4.4.8 notes).
// We therefore add the sign bits (one per non-zero value) and the
// `linbits` ESC field explicitly, exactly mirroring `decode_big_pair` /
// `decode_count1_quad`, so the count is bit-for-bit identical to what a
// round-trip through `decode_huffman` would read.
// =====================================================================

/// Cost in bits of coding one big-values `(x, y)` pair with `table`, per
/// §C.1.5.4.4.8: the Huffman codeword length `bitz[min(15,|x|)][min(15,|y|)]`,
/// plus one `linbits` ESC field for each component whose magnitude is
/// `≥ 15` (`s(ix - 15)`), plus one sign bit per non-zero component.
///
/// Returns `None` if the `(min(15,|x|), min(15,|y|))` cell is an unused
/// corner (`len == 0` with both indices non-zero) of the chosen table —
/// i.e. the pair cannot be coded by that codebook. Table 0's single
/// `(0, 0)` zero-length entry is the one legitimate `len == 0` cell and
/// costs `0` bits (no sign, no codeword) for an all-zero pair.
fn big_pair_bits(table: &BigTable, x: i32, y: i32) -> Option<usize> {
    let xl = usize::from(table.xlen);
    let ax = x.unsigned_abs();
    let ay = y.unsigned_abs();
    // Huffman symbol index is the magnitude clamped to 15 (the ESC code).
    let xi = ax.min(15) as usize;
    let yi = ay.min(15) as usize;
    if xi >= xl || yi >= xl {
        // Magnitude exceeds this codebook's index range without an ESC
        // path (small tables, linbits == 0) — not codable here.
        return None;
    }
    let ent = table.entries[xi * xl + yi];
    if ent.len == 0 && (xi != 0 || yi != 0) {
        // Unused rectangular corner: this pair is not in the codebook.
        return None;
    }
    let linbits = usize::from(table.linbits);
    let mut bits = usize::from(ent.len);
    // §C.1.5.4.4.8 step function s(ix - 15): a linbits ESC field is
    // appended whenever the *original* magnitude reaches 15.
    if ax >= 15 {
        bits += linbits;
    }
    if ay >= 15 {
        bits += linbits;
    }
    // Sign bits: one per non-zero value (mirrors decode_big_pair).
    if x != 0 {
        bits += 1;
    }
    if y != 0 {
        bits += 1;
    }
    Some(bits)
}

/// Cost in bits of coding all pairs of `is[start..end]` (a half-open line
/// range, `end - start` even) with codebook `table_idx`, per
/// §C.1.5.4.4.8. Returns `None` if any pair is not codable by the chosen
/// table (magnitude out of range / unused corner) or if `table_idx` is an
/// unused/out-of-range codebook.
fn region_bits_with_table(
    is: &[i32; NUM_LINES],
    start: usize,
    end: usize,
    table_idx: u8,
) -> Option<usize> {
    let table = big_table(table_idx).ok()?;
    let mut bits = 0usize;
    let mut k = start;
    // Pairs step by two; need both k and k+1 inside the range.
    while k + 1 < end && k + 1 < NUM_LINES {
        bits += big_pair_bits(table, is[k], is[k + 1])?;
        k += 2;
    }
    Some(bits)
}

/// The set of Table 3-B.7 big-values codebook indices that may be chosen
/// for a region, in ascending order. Tables 4 and 14 are "not used"
/// (§B.7) and are omitted; every other index 0..=31 is selectable.
pub const SELECTABLE_BIG_TABLES: [u8; 30] = [
    0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
    28, 29, 30, 31,
];

/// §C.1.5.4.4.7 — choose the big-values codebook that codes the line
/// range `is[start..end]` in the **fewest** bits, per §C.1.5.4.4.8.
///
/// Mirrors the spec's "trying all of these tables" strategy: every
/// selectable codebook (0..=31 minus the unused 4 / 14) is costed and the
/// minimum-bit table is returned as `(table_select, bits)`. An empty
/// range (`start >= end`) is coded by table 0 at `0` bits.
///
/// **Linbits-reach filter (encoder correctness, #1106).** A candidate
/// codebook is silently dropped if its [`big_table_reach`] is less than
/// the range's `max|is|` — without this filter, the chooser could pick
/// e.g. table 16 (`linbits=1`, reach 16) for a range with `|is| = 100`,
/// and `emit_big_pair` would write only the low `linbits` bits of
/// `|is| - 15` (here `85 & 0x1 = 1`), silently emitting a `16` instead
/// of `100` at decode time. The corner-only `xlen` check inside
/// `region_bits_with_table` catches small-table overflow (any magnitude
/// `≥ xlen` is rejected as not-codable) but **does not** catch ESC-table
/// overflow, because the Huffman symbol is clamped to 15 before the
/// codebook lookup. The reach test is the encoder-side correctness
/// guarantee that the decode round-trip is bit-exact.
///
/// Returns `None` only if **no** codebook is in-reach for the range —
/// impossible in practice because table 23 has reach 8206 (its
/// `linbits=13` field plus the 15-anchor), well past the §C.1.5.4.4.2
/// magnitude clamp of 8191, so this is reserved for a corrupt input
/// range or a programming error.
#[must_use]
pub fn choose_best_table_for_region(
    is: &[i32; NUM_LINES],
    start: usize,
    end: usize,
) -> Option<(u8, usize)> {
    if start >= end {
        // An empty region costs nothing and is nominally table 0.
        return Some((0, 0));
    }
    // §C.1.5.4.4.8 — find the range's peak magnitude so we can filter
    // codebooks whose linbits cannot represent it.
    let end_clamped = end.min(NUM_LINES);
    let max_mag = is[start..end_clamped]
        .iter()
        .map(|v| v.unsigned_abs())
        .max()
        .unwrap_or(0);

    let mut best: Option<(u8, usize)> = None;
    for &idx in SELECTABLE_BIG_TABLES.iter() {
        // Reach filter: drop codebooks that would truncate the largest
        // magnitude in the range. Table 0 has reach 0 and is selectable
        // only for an all-zero range; this falls out naturally.
        if big_table_reach(idx) < max_mag {
            continue;
        }
        if let Some(bits) = region_bits_with_table(is, start, end, idx) {
            match best {
                Some((_, b)) if bits >= b => {}
                _ => best = Some((idx, bits)),
            }
        }
    }
    best
}

/// Cost in bits of one count1 quadruple `(v, w, x, y)` (each magnitude
/// `≤ 1`) under quad table A or B, per §C.1.5.4.4.5. The codeword length
/// (Table 3-B.7-A for `table_b == false`, the trivial 4-bit code for
/// `table_b == true`) plus one sign bit per non-zero value (the length
/// tables "include the number of bits necessary to encode the sign
/// bits"). Indexing mirrors `decode_count1_quad` exactly: the quad
/// pattern is `(|v|<<3)|(|w|<<2)|(|x|<<1)|(|y|)`.
fn count1_quad_bits(v: i32, w: i32, x: i32, y: i32, table_b: bool) -> usize {
    let nz = |c: i32| usize::from(c != 0);
    let signs = nz(v) + nz(w) + nz(x) + nz(y);
    if table_b {
        // Table B: a flat 4-bit code (one bit per value) + sign bits.
        4 + signs
    } else {
        let pat = (nz(v) << 3) | (nz(w) << 2) | (nz(x) << 1) | nz(y);
        usize::from(QUAD_A[pat].0) + signs
    }
}

/// §C.1.5.4.4.5 — bits to code the count1 partition `is[start..end]`
/// (a half-open line range, `end - start` a multiple of 4) with the
/// chosen quad table. Mirrors the `bitsum_tableX` sum: one Huffman code
/// word per quadruple, sign bits included.
#[must_use]
pub fn count1_bits(is: &[i32; NUM_LINES], start: usize, end: usize, table_b: bool) -> usize {
    let mut bits = 0usize;
    let mut k = start;
    while k + 4 <= end && k + 4 <= NUM_LINES {
        bits += count1_quad_bits(is[k], is[k + 1], is[k + 2], is[k + 3], table_b);
        k += 4;
    }
    bits
}

/// §C.1.5.4.4.5 — bits for the count1 partition under the **better** of
/// the two quad tables, `min(bitsum_tableA, bitsum_tableB)`. Returns
/// `(count1table_select, bits)` where `count1table_select` is `false` for
/// table A or `true` for table B (the §2.4.2.7 `count1table_select`
/// field semantics: 0 → A, 1 → B).
#[must_use]
pub fn choose_best_count1_table(is: &[i32; NUM_LINES], start: usize, end: usize) -> (bool, usize) {
    let bits_a = count1_bits(is, start, end, false);
    let bits_b = count1_bits(is, start, end, true);
    if bits_b < bits_a {
        (true, bits_b)
    } else {
        (false, bits_a)
    }
}

/// Big-values / count1 partition split of a quantized `is[]`, derived the
/// way the §C.1.5.4.4.3 / §C.1.5.4.4.4 run-length steps prescribe:
///
/// * **`big_pairs`** — pairs in the big-values partition. Following
///   §C.1.5.4.4.3 ("run length of zeros at the upper end") and
///   §C.1.5.4.4.4 ("run length of values `≤ 1` … following the rzero
///   pairs"), the trailing all-zero pairs are dropped, then the trailing
///   run of `≤ 1`-magnitude quadruples is assigned to count1; everything
///   below that is big-values. `big_pairs * 2` is the big-values line
///   count.
/// * **`count1_quads`** — count1 quadruples between the big-values
///   partition and the trailing zero run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PartitionSplit {
    /// Number of `(x, y)` pairs in the big-values partition.
    pub big_pairs: usize,
    /// Number of `(v, w, x, y)` quadruples in the count1 partition.
    pub count1_quads: usize,
}

/// Compute the §C.1.5.4.4.3 / .4 partition split of `is[]`: trailing zero
/// pairs are dropped (r_zero), the trailing run of `≤ 1`-magnitude
/// quadruples becomes count1, and everything below it is big-values (whole
/// pairs). No non-zero line is ever dropped: the count1 upper edge is
/// rounded **up** to a quadruple boundary so any leftover trailing lines
/// are captured by the count1 run (they are `≤ 1` by construction since
/// they sit above the last `≥ 2`-magnitude line) rather than discarded.
#[must_use]
pub fn partition_split(is: &[i32; NUM_LINES]) -> PartitionSplit {
    // §C.1.5.4.4.3: r_zero — locate the last non-zero line.
    let mut last_nonzero: isize = -1;
    for (i, &v) in is.iter().enumerate() {
        if v != 0 {
            last_nonzero = i as isize;
        }
    }
    if last_nonzero < 0 {
        return PartitionSplit {
            big_pairs: 0,
            count1_quads: 0,
        };
    }
    let nonzero_lines = (last_nonzero + 1) as usize;

    // §C.1.5.4.4.4: count1 — the trailing run of quadruples whose four
    // magnitudes are all ≤ 1, scanning *down* from the end of the
    // non-zero region. Round the upper edge UP to a multiple of 4 so the
    // final partial quad (whose trailing lines are zero, hence ≤ 1) is
    // included — this keeps every non-zero line inside a partition.
    let mut count1_end = nonzero_lines.div_ceil(4) * 4;
    if count1_end > NUM_LINES {
        count1_end = NUM_LINES;
    }
    let mut count1_start = count1_end;
    let mut q = count1_end;
    while q >= 4 {
        let s = q - 4;
        if is[s..s + 4].iter().all(|&v| v.abs() <= 1) {
            count1_start = s;
            q -= 4;
        } else {
            break;
        }
    }

    let count1_quads = (count1_end - count1_start) / 4;
    // Big-values covers lines 0..count1_start, as whole pairs. count1_start
    // is a multiple of 4 (hence even), so this is exact.
    let big_pairs = count1_start / 2;
    PartitionSplit {
        big_pairs,
        count1_quads,
    }
}

/// Region line-index boundaries for an encoder-side big-values count:
/// the public wrapper over the decoder's `region_boundaries`, clamped to
/// the big-values line count `big_pairs * 2`.
///
/// Returns `(region0_end, region1_end)` line indices; region 2 runs from
/// `region1_end` to `big_pairs * 2`.
#[must_use]
pub fn encoder_region_boundaries(
    gc: &GranuleChannel,
    big_pairs: usize,
    sample_rate_hz: u32,
    version: MpegVersion,
) -> (usize, usize) {
    let bv2 = big_pairs * 2;
    let (r0, r1) = region_boundaries(gc, sample_rate_hz, version);
    (r0.min(bv2), r1.min(bv2))
}

/// **Exact** §C.1.5.4.4.5 + §C.1.5.4.4.8 Huffman bit count for a quantized
/// `is[]`, with explicit region boundaries and table selections — the
/// `count_bits` the inner iteration loop uses to test the rate budget.
///
/// Inputs:
/// * `is` — the 576 quantized lines.
/// * `big_pairs` — big-values pair count (lines `0..big_pairs*2`).
/// * `region_ends` — `(region0_end, region1_end)` line indices splitting
///   the big-values partition into three sub-regions (§C.1.5.4.4.6).
///   Region 0 is `0..region0_end`, region 1 `region0_end..region1_end`,
///   region 2 `region1_end..big_pairs*2`. Each is clamped to the
///   big-values range.
/// * `table_select` — the three big-values codebook indices (one per
///   sub-region).
/// * `count1_quads` — count1 quadruple count (lines
///   `big_pairs*2 .. big_pairs*2 + count1_quads*4`).
/// * `count1table_b` — `false` for quad table A, `true` for table B.
///
/// Returns the total big-values + count1 bit count (the zero partition
/// costs nothing), or `None` if a big-values pair is not codable by the
/// table chosen for its region (magnitude out of range / unused corner).
#[must_use]
pub fn count_huffman_bits(
    is: &[i32; NUM_LINES],
    big_pairs: usize,
    region_ends: (usize, usize),
    table_select: [u8; 3],
    count1_quads: usize,
    count1table_b: bool,
) -> Option<usize> {
    let bv2 = (big_pairs * 2).min(NUM_LINES);
    let r0 = region_ends.0.min(bv2);
    let r1 = region_ends.1.max(r0).min(bv2);

    let mut bits = 0usize;
    // Three big-values sub-regions (§C.1.5.4.4.8).
    bits += region_bits_with_table(is, 0, r0, table_select[0])?;
    bits += region_bits_with_table(is, r0, r1, table_select[1])?;
    bits += region_bits_with_table(is, r1, bv2, table_select[2])?;

    // count1 partition (§C.1.5.4.4.5).
    let c1_start = bv2;
    let c1_end = (c1_start + count1_quads * 4).min(NUM_LINES);
    bits += count1_bits(is, c1_start, c1_end, count1table_b);

    Some(bits)
}

// =====================================================================
// Forward Huffman **bit emission** — the §2.4.1.7 `huffmancodebits()`
// encoder counterpart to `decode_huffman` above. Given a quantized
// `is[]`, the region split, and the chosen table-selects (from
// `choose_best_table_for_region` / `choose_best_count1_table`), this
// writes the actual codewords + linbits ESC fields + sign bits, in the
// exact order `decode_huffman` reads them, into a bit buffer. The result
// is byte-aligned and reads back through `MainDataReader` bit-for-bit, so
// `encode_huffman` → `decode_huffman` recovers the original `is[]`.
//
// The emitted bit length equals the §C.1.5.4.4.5 / .8 `count_huffman_bits`
// for the same inputs: the writer emits exactly the codeword
// (`HEntry.len` / `QUAD_A.0` bits), one `linbits` field per magnitude-≥-15
// component, and one sign bit per non-zero value — the same three terms
// `big_pair_bits` / `count1_quad_bits` sum. The emitter writes into a
// shared [`MainDataWriter`] (the inverse of `MainDataReader`), so the
// Huffman (part3) payload can be appended directly after a granule's
// scalefactor (part2) bits with no intervening byte alignment — the
// §2.4.1.7 `main_data()` layout.

/// Write the sign bit for a non-zero value: `1` if negative, `0` if
/// positive (mirrors `decode_big_pair`'s `read(1) == 1` negation).
fn write_sign(w: &mut MainDataWriter, v: i32) {
    w.write(u32::from(v < 0), 1);
}

/// Error from the Huffman **encode** stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HuffmanEncodeError {
    /// A big-values `(x, y)` pair could not be coded by the `table_select`
    /// chosen for its region: the magnitude is out of the codebook's index
    /// range (small table, no `linbits` escape) or lands on an unused
    /// rectangular corner. Carries the region's `table_select` index.
    PairNotCodable(u8),
    /// `table_select` named a codebook index that is "not used" in
    /// Table 3-B.7 (4 or 14) or is out of the 0..=31 range.
    UnusedTable(u8),
    /// `big_pairs * 2` exceeded the 576-line granule capacity.
    BigValuesTooLarge,
}

impl core::fmt::Display for HuffmanEncodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            HuffmanEncodeError::PairNotCodable(t) => {
                write!(f, "big-values pair not codable by table_select {t}")
            }
            HuffmanEncodeError::UnusedTable(t) => {
                write!(f, "Huffman table_select {t} is unused (Table 3-B.7)")
            }
            HuffmanEncodeError::BigValuesTooLarge => {
                write!(f, "big_pairs*2 exceeds the 576-line granule capacity")
            }
        }
    }
}

impl std::error::Error for HuffmanEncodeError {}

/// The Huffman main-data payload of one granule-channel: the packed
/// codeword bytes plus the exact number of payload bits written (the
/// trailing byte-pad of [`Mp3HuffmanData::bytes`] is excluded from
/// `bit_len`). `bit_len` equals the matching [`count_huffman_bits`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mp3HuffmanData {
    /// Byte-aligned codeword payload (last byte zero-padded). Readable by
    /// [`MainDataReader`] bit-for-bit.
    pub bytes: Vec<u8>,
    /// Number of payload bits (excluding the trailing byte-pad).
    pub bit_len: usize,
}

/// Emit one big-values `(x, y)` pair into `w` with `table`, the exact
/// inverse of [`decode_big_pair`]: the Huffman codeword for the
/// magnitude-clamped `(min(15,|x|), min(15,|y|))` cell, then — when a
/// component's *original* magnitude is `≥ 15` and `linbits > 0` — the
/// `linbits` ESC field carrying `|v| - 15`, then the sign bit (`1` for
/// negative) for each non-zero component. The §2.4.1.7 order is
/// codeword → linbits_x → sign_x → linbits_y → sign_y.
fn emit_big_pair(
    w: &mut MainDataWriter,
    table: &BigTable,
    x: i32,
    y: i32,
    table_idx: u8,
) -> Result<(), HuffmanEncodeError> {
    let xl = usize::from(table.xlen);
    let ax = x.unsigned_abs();
    let ay = y.unsigned_abs();
    let xi = ax.min(15) as usize;
    let yi = ay.min(15) as usize;
    if xi >= xl || yi >= xl {
        return Err(HuffmanEncodeError::PairNotCodable(table_idx));
    }
    let ent = table.entries[xi * xl + yi];
    if ent.len == 0 && (xi != 0 || yi != 0) {
        return Err(HuffmanEncodeError::PairNotCodable(table_idx));
    }
    let linbits = u32::from(table.linbits);
    // Codeword (zero-length entry for an all-zero pair writes nothing).
    w.write(u32::from(ent.code), u32::from(ent.len));
    // linbits ESC for x, then sign for x.
    if ax >= 15 && linbits > 0 {
        w.write(ax - 15, linbits);
    }
    if x != 0 {
        write_sign(w, x);
    }
    // linbits ESC for y, then sign for y.
    if ay >= 15 && linbits > 0 {
        w.write(ay - 15, linbits);
    }
    if y != 0 {
        write_sign(w, y);
    }
    Ok(())
}

/// Emit the big-values pairs of `is[start..end]` with codebook
/// `table_idx`, the inverse of the region loop in `decode_huffman`.
fn emit_region(
    w: &mut MainDataWriter,
    is: &[i32; NUM_LINES],
    start: usize,
    end: usize,
    table_idx: u8,
) -> Result<(), HuffmanEncodeError> {
    let table = big_table(table_idx).map_err(|_| HuffmanEncodeError::UnusedTable(table_idx))?;
    let mut k = start;
    while k + 1 < end && k + 1 < NUM_LINES {
        emit_big_pair(w, table, is[k], is[k + 1], table_idx)?;
        k += 2;
    }
    Ok(())
}

/// Emit one count1 `(v, w, x, y)` quadruple into `bw`, the exact inverse
/// of [`decode_count1_quad`]. Quad table B writes the trivial 4-bit code
/// (`0` → magnitude 1, `1` → 0, MSB = v); table A writes the variable
/// `QUAD_A` codeword for the `(|v|<<3)|(|w|<<2)|(|x|<<1)|(|y|)` pattern.
/// Sign bits (`1` for negative) follow each non-zero value in v, w, x, y
/// order.
fn emit_count1_quad(bw: &mut MainDataWriter, v: i32, w: i32, x: i32, y: i32, table_b: bool) {
    let nz = |c: i32| u32::from(c != 0);
    if table_b {
        // Table B: one bit per value, 0 → magnitude 1, 1 → 0. The
        // decoder reads 4 bits MSB-first and XORs with 1, so the wire bit
        // is `0` for a non-zero magnitude.
        let bits = ((nz(v) ^ 1) << 3) | ((nz(w) ^ 1) << 2) | ((nz(x) ^ 1) << 1) | (nz(y) ^ 1);
        bw.write(bits, 4);
    } else {
        let pat = ((nz(v) << 3) | (nz(w) << 2) | (nz(x) << 1) | nz(y)) as usize;
        let (clen, code) = QUAD_A[pat];
        bw.write(u32::from(code), u32::from(clen));
    }
    if v != 0 {
        write_sign(bw, v);
    }
    if w != 0 {
        write_sign(bw, w);
    }
    if x != 0 {
        write_sign(bw, x);
    }
    if y != 0 {
        write_sign(bw, y);
    }
}

/// **Emit** the §2.4.1.7 `huffmancodebits()` payload for one
/// granule-channel — the forward counterpart to [`decode_huffman`] and
/// the byte-producing sibling of [`count_huffman_bits`].
///
/// Inputs mirror [`count_huffman_bits`] exactly:
/// * `is` — the 576 quantized lines.
/// * `big_pairs` — big-values pair count (lines `0..big_pairs*2`).
/// * `region_ends` — `(region0_end, region1_end)` line indices splitting
///   the big-values partition into three sub-regions. Region 0 is
///   `0..region0_end`, region 1 `region0_end..region1_end`, region 2
///   `region1_end..big_pairs*2`. Each is clamped to the big-values range.
/// * `table_select` — the three big-values codebook indices.
/// * `count1_quads` — count1 quadruple count.
/// * `count1table_b` — `false` for quad table A, `true` for table B.
///
/// Returns the byte-aligned codeword payload and its exact payload bit
/// length (which equals `count_huffman_bits` of the same inputs). Errors
/// if a big-values pair is not codable by the table chosen for its
/// region, or if a `table_select` names an unused/out-of-range codebook.
///
/// The codewords are written in `decode_huffman`'s read order, so
/// `decode_huffman(MainDataReader::new(&data.bytes), gc, data.bit_len, …)`
/// recovers `is[0 .. big_pairs*2 + count1_quads*4]` bit-exactly (the
/// remaining lines decode as the zero partition).
pub fn encode_huffman(
    is: &[i32; NUM_LINES],
    big_pairs: usize,
    region_ends: (usize, usize),
    table_select: [u8; 3],
    count1_quads: usize,
    count1table_b: bool,
) -> Result<Mp3HuffmanData, HuffmanEncodeError> {
    let mut w = MainDataWriter::new();
    let bit_len = emit_huffman(
        &mut w,
        is,
        big_pairs,
        region_ends,
        table_select,
        count1_quads,
        count1table_b,
    )?;
    Ok(Mp3HuffmanData {
        bytes: w.finish(),
        bit_len,
    })
}

/// Emit the §2.4.1.7 `huffmancodebits()` (part3) payload of one
/// granule-channel into a shared [`MainDataWriter`], returning the number
/// of bits written.
///
/// Unlike [`encode_huffman`] (which wraps a fresh writer and byte-aligns
/// the result), this appends the codewords directly to `w` at its current
/// bit position, so a caller can place a granule's part2 scalefactors
/// immediately before the part3 Huffman data with no intervening byte
/// alignment — the contiguous `main_data()` layout. The inputs and the
/// returned bit count are identical to [`encode_huffman`] /
/// [`count_huffman_bits`].
///
/// # Errors
///
/// [`HuffmanEncodeError::BigValuesTooLarge`] if `big_pairs * 2` exceeds
/// the 576-line granule capacity; [`HuffmanEncodeError::PairNotCodable`]
/// if a big-values pair cannot be coded by its region's table;
/// [`HuffmanEncodeError::UnusedTable`] if a `table_select` names an
/// unused/out-of-range codebook.
pub fn emit_huffman(
    w: &mut MainDataWriter,
    is: &[i32; NUM_LINES],
    big_pairs: usize,
    region_ends: (usize, usize),
    table_select: [u8; 3],
    count1_quads: usize,
    count1table_b: bool,
) -> Result<usize, HuffmanEncodeError> {
    if big_pairs * 2 > NUM_LINES {
        return Err(HuffmanEncodeError::BigValuesTooLarge);
    }
    let bv2 = (big_pairs * 2).min(NUM_LINES);
    let r0 = region_ends.0.min(bv2);
    let r1 = region_ends.1.max(r0).min(bv2);

    let start = w.bit_pos();
    // Three big-values sub-regions, in line order (§2.4.1.7).
    emit_region(w, is, 0, r0, table_select[0])?;
    emit_region(w, is, r0, r1, table_select[1])?;
    emit_region(w, is, r1, bv2, table_select[2])?;

    // count1 partition: whole quadruples above the big-values lines.
    let c1_start = bv2;
    let c1_end = (c1_start + count1_quads * 4).min(NUM_LINES);
    let mut k = c1_start;
    while k + 4 <= c1_end {
        emit_count1_quad(w, is[k], is[k + 1], is[k + 2], is[k + 3], count1table_b);
        k += 4;
    }

    Ok(w.bit_pos() - start)
}

include!("huffman_tables.rs");

#[cfg(test)]
include!("huffman_tests.rs");
