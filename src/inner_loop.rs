//! Layer III **inner-loop global-gain search** — the rate-control loop
//! of ISO/IEC 11172-3:1993 Annex C §C.1.5.4.4 (informational). This
//! wraps the §2.4.3.4.7 [`crate::quantize::quantize`] primitive in the
//! `global_gain` step-size search the spec's inner iteration loop
//! performs.
//!
//! # Scope (Phase 2 step 5)
//!
//! Given a target magnitude spectrum `xr[576]` and an already-chosen
//! scalefactor configuration (the `GranuleChannel` minus its
//! `global_gain`, plus the `ScaleFactors`), find the **smallest**
//! `global_gain` — i.e. the finest quantization, the largest output
//! magnitudes — whose quantized `is[576]` still satisfies a constraint.
//! Two forms are provided:
//!
//! * **Magnitude clamp** ([`search_magnitude_clamp`]) — §C.1.5.4.4.2
//!   ("Test of the maximum of the quantized values"): the largest
//!   allowed quantized value is bounded. The §2.4.1.7 `big_values`
//!   definition states *"The maximum absolute value in this range is
//!   constrained to 8191"*, so the constraint here is `max|is| ≤ 8191`
//!   ([`BIG_VALUES_LIMIT`]). The spec increases the quantizer step size
//!   (`qquant = qquant + 1`, i.e. raises `global_gain`) until the
//!   maximum is within range; we binary-search the threshold instead.
//!
//! * **Bit budget** ([`search_bit_budget`]) — §C.1.5.4.4
//!   ("increases the quantizer step size until the output vector can be
//!   coded with the available number of bits"): the smallest
//!   `global_gain` whose **exact** §C.1.5.4.4.5 + §C.1.5.4.4.8 Huffman
//!   bit count ([`exact_bit_count`]) fits a supplied budget. The count
//!   partitions `is[]` (§C.1.5.4.4.3 / .4), SUBDIVIDEs the big-values
//!   range into three sub-regions (§C.1.5.4.4.6), picks the minimum-bit
//!   codebook per region and the better count1 quad table
//!   (§C.1.5.4.4.7), then sums Table 3-B.7 codeword lengths plus the
//!   `linbits` ESC fields and the sign bits. The legacy
//!   [`coarse_bit_estimate`] placeholder of the r133 search is retained
//!   only for reference / comparison.
//!
//! This file does **not** implement the psychoacoustic model, the
//! §C.1.5.4.3 outer (distortion-control) loop, or scalefactor
//! estimation. It searches the one scalar (`global_gain`) the inner loop
//! varies and now computes the exact Huffman count that gates it.
//!
//! # Search strategy: bisection vs. linear scan
//!
//! The §2.4.3.4.7.1 requantization gain is
//! `2^((global_gain - 210)/4) · …`, so the per-line quantizer
//! `|is_i| = round((|xr_i| / factor)^(3/4))` is **monotone
//! non-increasing** in `global_gain`: a larger `global_gain` multiplies
//! the divisor, shrinking every `|is_i|`. Therefore `max_i |is_i|` (and
//! any non-negative *weighting* of the magnitudes, such as the coarse
//! estimate) is monotone non-increasing in `global_gain`, so the
//! magnitude-clamp predicate is a step function over `[0, 255]` and the
//! smallest satisfying gain is found by **binary search**.
//!
//! The **exact** Huffman count is the exception: Huffman codeword lengths
//! are not monotone in magnitude and the minimum-bit codebook per region
//! changes as the values shrink, so a coarser quantization can cost a few
//! more bits than a finer one. The exact-count budget search therefore
//! uses the spec's own upward `qquant = qquant + 1` step (§C.1.5.4.4),
//! returning the first (smallest) gain whose count fits, rather than a
//! bisection. See [`search_linear`].
//!
//! Every rule is taken from the ISO/IEC 11172-3:1993 §C.1.5.4 /
//! §2.4.1.7 / §2.4.3.4.7 text.

use crate::frame::MpegVersion;
use crate::huffman::{
    choose_best_count1_table, choose_best_table_for_region, count_huffman_bits, partition_split,
    PartitionSplit,
};
use crate::quantize::quantize;
use crate::requantize::NUM_LINES;
use crate::scalefactors::ScaleFactors;
use crate::side_info::{BlockType, GranuleChannel};

/// The maximum absolute quantized value the big-values partition may
/// carry, from the §2.4.1.7 `big_values` definition: *"The maximum
/// absolute value in this range is constrained to 8191."*
pub const BIG_VALUES_LIMIT: i32 = 8191;

/// Inclusive lower bound of the 8-bit `global_gain` side-info field.
pub const GAIN_MIN: u8 = u8::MIN;
/// Inclusive upper bound of the 8-bit `global_gain` side-info field.
pub const GAIN_MAX: u8 = u8::MAX;

/// Outcome of an inner-loop `global_gain` search.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InnerLoopResult {
    /// The chosen `global_gain` — the smallest gain whose `is[]`
    /// satisfies the search constraint, or [`GAIN_MAX`] when no gain in
    /// `[GAIN_MIN, GAIN_MAX]` does (in which case `satisfied` is false
    /// and `is` / `max_abs` are the values at [`GAIN_MAX`], the coarsest
    /// available quantization).
    pub global_gain: u8,
    /// The quantized Huffman-input buffer at the chosen `global_gain`.
    pub is: [i32; NUM_LINES],
    /// `max_i |is[i]|` for the returned `is` (`0` for an all-zero
    /// buffer).
    pub max_abs: i32,
    /// Whether the chosen `global_gain` actually satisfies the
    /// constraint. False only when even the coarsest quantization
    /// ([`GAIN_MAX`]) cannot meet it.
    pub satisfied: bool,
}

/// `max_i |is[i]|` over the 576 lines (`0` for an all-zero buffer).
#[must_use]
pub fn max_abs(is: &[i32; NUM_LINES]) -> i32 {
    is.iter().map(|&v| v.abs()).max().unwrap_or(0)
}

/// A deliberately coarse stand-in for the §C.1.5.4.4.5 / §C.1.5.4.4.8
/// Huffman bit count: it approximates the cost of one granule-channel
/// as the sum over all non-zero lines of `bits(|is_i|) + 1` (a linear
/// PCM word length plus one sign bit), where `bits(n)` is the number of
/// bits to hold `n` (`0` for `n == 0`, `1` for `n == 1`, …). This is an
/// order-of-magnitude proxy that is **not** the exact codebook-length
/// sum; the real §C.1.5.4.4 count now lives in [`exact_bit_count`] and
/// gates [`search_bit_budget`]. `coarse_bit_estimate` is retained only
/// for reference / comparison (its monotonicity in `global_gain` is a
/// useful contrast with the exact count's non-monotonicity).
#[must_use]
pub fn coarse_bit_estimate(is: &[i32; NUM_LINES]) -> u64 {
    let mut bits = 0u64;
    for &v in is.iter() {
        let mag = v.unsigned_abs();
        if mag != 0 {
            // bit length of `mag` plus one sign bit.
            bits += u64::from(u32::BITS - mag.leading_zeros()) + 1;
        }
    }
    bits
}

/// A SUBDIVIDE of the big-values partition into three sub-regions
/// (§C.1.5.4.4.6) plus the chosen best codebook per region, plus the
/// count1 partition's best quad table — the full table/region decision
/// the inner loop makes before counting bits, returned alongside the
/// **exact** §C.1.5.4.4.5 + §C.1.5.4.4.8 bit total.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExactBitCount {
    /// The §C.1.5.4.4.3 / .4 big-values / count1 partition split.
    pub split: PartitionSplit,
    /// Big-values sub-region end line indices `(region0_end, region1_end)`;
    /// region 2 runs to `split.big_pairs * 2`.
    pub region_ends: (usize, usize),
    /// Best big-values codebook per sub-region (§C.1.5.4.4.7).
    pub table_select: [u8; 3],
    /// Best count1 quad table: `false` for table A, `true` for table B.
    pub count1table_b: bool,
    /// Total exact Huffman bit count (big-values + count1; zero partition
    /// is free).
    pub bits: usize,
}

/// SUBDIVIDE the big-values pair count into three sub-region **line**
/// boundaries, per the §C.1.5.4.4.6 "simple strategy" the spec offers
/// (assign ~1/3 of the range to region 0 and ~1/4 to region 2). For
/// block-split (short) blocks the spec uses only two sub-regions with
/// region 2 empty; we mirror that by collapsing region 2.
///
/// Returns `(region0_end, region1_end)` line indices within `0..bv2`
/// where `bv2 = big_pairs * 2`.
fn subdivide(gc: &GranuleChannel, big_pairs: usize) -> (usize, usize) {
    let bv2 = big_pairs * 2;
    if bv2 == 0 {
        return (0, 0);
    }
    if gc.window_switching_flag && gc.block_type == BlockType::Short {
        // Two sub-regions only (§C.1.5.4.4.6): split the big-values pairs
        // in two, region 2 empty.
        let r0_pairs = big_pairs / 2;
        let r0 = (r0_pairs * 2).min(bv2);
        return (r0, bv2);
    }
    // ~1/3 to region 0, ~1/4 to region 2 (so ~1/3, ~5/12, ~1/4). Work in
    // whole pairs and align region ends to a pair boundary.
    let r0_pairs = big_pairs / 3;
    let r2_pairs = big_pairs / 4;
    let r1_pairs = big_pairs.saturating_sub(r0_pairs + r2_pairs);
    let r0 = (r0_pairs * 2).min(bv2);
    let r1 = ((r0_pairs + r1_pairs) * 2).min(bv2);
    (r0, r1)
}

/// A band-aligned §C.1.5.4.4.6 SUBDIVIDE of the big-values range for a
/// long-family granule: the region boundaries fall on **scalefactor-band
/// edges** (the spec's "SUBDIVIDE splits the *scalefactor bands*
/// corresponding to these values into three groups") and the chosen
/// `region0_count` / `region1_count` are the band counts the side-info
/// fields carry (`region0_count + 1` bands in region 0,
/// `region1_count + 1` in region 1, the rest in region 2 — §2.4.2.7).
///
/// This is the boundary form the decoder's `region_boundaries` can
/// actually reconstruct: the decoder derives the region ends solely from
/// the band-start table and the transmitted `region0_count` /
/// `region1_count`, so a boundary chosen mid-band (as the simpler
/// [`subdivide`] pair-thirds heuristic may produce) is unrepresentable on
/// the wire. [`SubdivideBands`] therefore lets the inner-loop bit
/// estimate count exactly the partition the encoder will emit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubdivideBands {
    /// Big-values sub-region end line indices `(region0_end, region1_end)`;
    /// region 2 runs to `big_pairs * 2`. Both are scalefactor-band-aligned.
    pub region_ends: (usize, usize),
    /// `region0_count` side-info value (4-bit field, `0..=15`): region 0
    /// covers bands `0..=region0_count`.
    pub region0_count: u8,
    /// `region1_count` side-info value (3-bit field, `0..=7`): region 1
    /// covers the following `region1_count + 1` bands.
    pub region1_count: u8,
}

/// Band-aligned §C.1.5.4.4.6 SUBDIVIDE for a **long-family** granule
/// (`block_type ∈ {Long, Start, End}`): choose region boundaries on
/// scalefactor-band edges, assigning ~1/3 of the big-values range to
/// region 0 and ~1/4 to region 2 — the "very simple" split strategy the
/// spec offers, but snapped to band edges (the only boundaries the
/// decoder can reproduce) and clamped to the 4-bit `region0_count` /
/// 3-bit `region1_count` field widths.
///
/// `big_pairs` is the §C.1.5.4.4.3/.4 big-values pair count (so the
/// big-values line span is `bv2 = big_pairs * 2`). For a short / mixed
/// block this primitive does not apply — those use the two-subregion
/// blocksplit rule (§C.1.5.4.4.6, region1_count default) handled by
/// [`crate::short_block::short_block_region_defaults`]; callers gate on
/// the block type before calling.
///
/// The returned `region_ends` are clamped to `bv2`, so an
/// already-band-aligned `bv2` reproduces exactly.
// The band-walking loops use `b` as both the loop variable and the index
// into `starts` (what the decoder's `region_boundaries` reads), so the
// band ↔ start-index relationship the spec uses stays explicit.
#[allow(clippy::needless_range_loop)]
#[must_use]
pub fn subdivide_bands(
    sample_rate_hz: u32,
    version: MpegVersion,
    big_pairs: usize,
) -> SubdivideBands {
    let bv2 = big_pairs * 2;
    if bv2 == 0 {
        return SubdivideBands {
            region_ends: (0, 0),
            region0_count: 0,
            region1_count: 0,
        };
    }
    let starts = crate::requantize::long_band_starts(sample_rate_hz, version);
    // §C.1.5.4.4.6 "very simple" strategy: ~1/3 of the band span to
    // region 0, ~1/4 to region 2 (so region 1 takes the ~5/12 middle).
    let third = bv2 / 3;
    let three_quarters = (bv2 * 3) / 4;

    // region 0 covers bands 0..=region0_count, so its end line is
    // starts[region0_count + 1]. Walk band edges up to the one at or just
    // below `third`; clamp region0_count to the 4-bit field max (15).
    let mut r0_count: u8 = 0;
    for b in 1..=21usize {
        if starts[b] <= third {
            r0_count = (b - 1) as u8;
        } else {
            break;
        }
    }
    r0_count = r0_count.min(15);
    let r0_band = usize::from(r0_count) + 1;
    let r0_end = starts
        .get(r0_band)
        .copied()
        .unwrap_or(NUM_LINES)
        .min(NUM_LINES);

    // region 1 covers the next region1_count+1 bands; walk band edges up
    // to the one at or just below `three_quarters`; clamp to 3-bit max (7).
    let mut r1_count: u8 = 0;
    for b in (r0_band + 1)..=21usize {
        if starts[b] <= three_quarters {
            r1_count = (b - r0_band - 1) as u8;
        } else {
            break;
        }
    }
    r1_count = r1_count.min(7);
    let r1_band = r0_band + usize::from(r1_count) + 1;
    let r1_end = starts
        .get(r1_band)
        .copied()
        .unwrap_or(NUM_LINES)
        .min(NUM_LINES);

    SubdivideBands {
        region_ends: (r0_end.min(bv2), r1_end.min(bv2).max(r0_end.min(bv2))),
        region0_count: r0_count,
        region1_count: r1_count,
    }
}

/// Band-aligned variant of [`exact_bit_count`]: identical exact
/// §C.1.5.4.4.5 + §C.1.5.4.4.8 Huffman bit total, but the big-values
/// SUBDIVIDE boundaries fall on scalefactor-band edges via
/// [`subdivide_bands`] (long-family blocks) so the count matches the
/// partition the encoder will actually emit on the wire. For short /
/// mixed blocks it falls back to the block-type-steered pair-split
/// [`subdivide`] (those carry the §C.1.5.4.4.6 two-subregion blocksplit
/// defaults the decoder ignores). The returned [`ExactBitCount`] also
/// surfaces the chosen `region0_count` / `region1_count` via its
/// `region_ends`; pair this with [`subdivide_bands`] when the side-info
/// field values are needed.
///
/// This does not replace [`exact_bit_count`] — the default inner-loop
/// search keeps the simpler heuristic so emitted bytes are unchanged
/// unless a caller opts into the band-aligned estimate.
#[must_use]
pub fn exact_bit_count_band_aligned(
    is: &[i32; NUM_LINES],
    gc: &GranuleChannel,
    sample_rate_hz: u32,
    version: MpegVersion,
) -> Option<ExactBitCount> {
    let split = partition_split(is);
    let bv2 = split.big_pairs * 2;

    // Long-family blocks get band-aligned boundaries; short / mixed keep
    // the block-type-steered two-subregion pair split.
    let region_ends = if gc.window_switching_flag && gc.block_type == BlockType::Short {
        subdivide(gc, split.big_pairs)
    } else {
        subdivide_bands(sample_rate_hz, version, split.big_pairs).region_ends
    };

    let (t0, _) = choose_best_table_for_region(is, 0, region_ends.0)?;
    let (t1, _) = choose_best_table_for_region(is, region_ends.0, region_ends.1)?;
    let (t2, _) = choose_best_table_for_region(is, region_ends.1, bv2)?;
    let table_select = [t0, t1, t2];

    let c1_start = bv2;
    let c1_end = c1_start + split.count1_quads * 4;
    let (count1table_b, _) = choose_best_count1_table(is, c1_start, c1_end);

    let bits = count_huffman_bits(
        is,
        split.big_pairs,
        region_ends,
        table_select,
        split.count1_quads,
        count1table_b,
    )?;

    Some(ExactBitCount {
        split,
        region_ends,
        table_select,
        count1table_b,
        bits,
    })
}

/// **Exact** §C.1.5.4.4.5 + §C.1.5.4.4.8 Huffman bit count for a
/// quantized `is[]`: partition it (§C.1.5.4.4.3 / .4), SUBDIVIDE the
/// big-values range (§C.1.5.4.4.6), choose the minimum-bit codebook per
/// sub-region and the better count1 quad table (§C.1.5.4.4.7), and sum
/// the codeword lengths via [`count_huffman_bits`]. This is the real
/// `count_bits` the rate-control loop tests — it replaces
/// [`coarse_bit_estimate`].
///
/// `gc` supplies only the block-type / window-switching flags that steer
/// SUBDIVIDE; its `region0_count` / `region1_count` are **not** used (the
/// region split is chosen here). Returns `None` only on a corrupt range
/// no codebook can code (see [`choose_best_table_for_region`]).
#[must_use]
pub fn exact_bit_count(is: &[i32; NUM_LINES], gc: &GranuleChannel) -> Option<ExactBitCount> {
    let split = partition_split(is);
    let bv2 = split.big_pairs * 2;
    let region_ends = subdivide(gc, split.big_pairs);

    // Best codebook per big-values sub-region (§C.1.5.4.4.7 / .8).
    let (t0, _) = choose_best_table_for_region(is, 0, region_ends.0)?;
    let (t1, _) = choose_best_table_for_region(is, region_ends.0, region_ends.1)?;
    let (t2, _) = choose_best_table_for_region(is, region_ends.1, bv2)?;
    let table_select = [t0, t1, t2];

    // Best count1 quad table (§C.1.5.4.4.5).
    let c1_start = bv2;
    let c1_end = c1_start + split.count1_quads * 4;
    let (count1table_b, _) = choose_best_count1_table(is, c1_start, c1_end);

    let bits = count_huffman_bits(
        is,
        split.big_pairs,
        region_ends,
        table_select,
        split.count1_quads,
        count1table_b,
    )?;

    Some(ExactBitCount {
        split,
        region_ends,
        table_select,
        count1table_b,
        bits,
    })
}

/// Quantize `xr` at the given `global_gain` (everything else in
/// `gc_template` / `sf` held fixed) and return the resulting `is[]`.
fn quantize_at(
    xr: &[f32; NUM_LINES],
    gc_template: &GranuleChannel,
    sf: &ScaleFactors,
    sample_rate_hz: u32,
    version: MpegVersion,
    global_gain: u8,
) -> [i32; NUM_LINES] {
    let mut gc = *gc_template;
    gc.global_gain = global_gain;
    quantize(xr, &gc, sf, sample_rate_hz, version)
}

/// Generic binary search for the smallest `global_gain` in
/// `[GAIN_MIN, GAIN_MAX]` whose quantized `is[]` satisfies `predicate`.
///
/// `predicate(is)` must be monotone in `global_gain`: false for gains
/// below some threshold and true at and above it. The magnitude-clamp
/// predicate is (because `|is_i|` is monotone non-increasing in
/// `global_gain`); the exact-bit-budget predicate is **not**, so it uses
/// [`search_linear`] instead. Returns the threshold gain and its `is[]`;
/// if no gain satisfies the predicate the result carries [`GAIN_MAX`]
/// with `satisfied == false`.
fn search<F>(
    xr: &[f32; NUM_LINES],
    gc_template: &GranuleChannel,
    sf: &ScaleFactors,
    sample_rate_hz: u32,
    version: MpegVersion,
    predicate: F,
) -> InnerLoopResult
where
    F: Fn(&[i32; NUM_LINES]) -> bool,
{
    let quant = |gain: u8| quantize_at(xr, gc_template, sf, sample_rate_hz, version, gain);

    // Standard lower-bound binary search over the inclusive integer
    // range [GAIN_MIN, GAIN_MAX]: find the smallest `gain` with
    // predicate(quant(gain)) == true. `lo` is a candidate, `hi` is one
    // past the searched range, kept as u16 to avoid u8 overflow.
    let mut lo: u16 = u16::from(GAIN_MIN);
    let mut hi: u16 = u16::from(GAIN_MAX) + 1;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let is_mid = quant(mid as u8);
        if predicate(&is_mid) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }

    if lo > u16::from(GAIN_MAX) {
        // No gain satisfies the predicate; report the coarsest one.
        let is = quant(GAIN_MAX);
        let m = max_abs(&is);
        InnerLoopResult {
            global_gain: GAIN_MAX,
            is,
            max_abs: m,
            satisfied: false,
        }
    } else {
        let gain = lo as u8;
        let is = quant(gain);
        let m = max_abs(&is);
        InnerLoopResult {
            global_gain: gain,
            is,
            max_abs: m,
            satisfied: true,
        }
    }
}

/// Linear upward scan for the smallest `global_gain` in
/// `[GAIN_MIN, GAIN_MAX]` whose quantized `is[]` satisfies `predicate`,
/// for a predicate that is **not** monotone in `global_gain`.
///
/// The exact §C.1.5.4.4 Huffman bit count is *not* monotone in the gain:
/// raising the gain shrinks every `|is_i|`, but Huffman codeword lengths
/// are not monotone in magnitude and the optimal codebook per region
/// changes, so a coarser quantization can occasionally cost a few more
/// bits than a finer one. The spec's inner loop therefore does not binary
/// search — it steps `qquant = qquant + 1` (§C.1.5.4.4) and stops at the
/// first gain whose count fits. This helper mirrors that: it scans gains
/// upward and returns the first satisfying one (the smallest gain that
/// fits), making no monotonicity assumption.
fn search_linear<F>(
    xr: &[f32; NUM_LINES],
    gc_template: &GranuleChannel,
    sf: &ScaleFactors,
    sample_rate_hz: u32,
    version: MpegVersion,
    predicate: F,
) -> InnerLoopResult
where
    F: Fn(&[i32; NUM_LINES]) -> bool,
{
    let quant = |gain: u8| quantize_at(xr, gc_template, sf, sample_rate_hz, version, gain);
    for g in u16::from(GAIN_MIN)..=u16::from(GAIN_MAX) {
        let is = quant(g as u8);
        if predicate(&is) {
            let m = max_abs(&is);
            return InnerLoopResult {
                global_gain: g as u8,
                is,
                max_abs: m,
                satisfied: true,
            };
        }
    }
    // No gain satisfies the predicate; report the coarsest one.
    let is = quant(GAIN_MAX);
    let m = max_abs(&is);
    InnerLoopResult {
        global_gain: GAIN_MAX,
        is,
        max_abs: m,
        satisfied: false,
    }
}

/// §C.1.5.4.4.2 magnitude-clamp inner loop: find the smallest
/// `global_gain` whose quantized `is[]` keeps `max|is| ≤`
/// [`BIG_VALUES_LIMIT`].
///
/// `gc_template.global_gain` is ignored — it is the field being
/// searched; every other field (`scalefac_scale`, `subblock_gain`,
/// `block_type`, `preflag`, …) and `sf` are held fixed. The returned
/// `is[]` is the finest available quantization (smallest gain) that
/// still fits the 8191 limit. When even [`GAIN_MIN`] fits (a small-
/// magnitude `xr`), [`GAIN_MIN`] is returned.
///
/// Raising `global_gain` only divides by `2^((global_gain - 210)/4)`,
/// so for a fixed scalefactor config there is a finite largest
/// amplitude the coarsest gain ([`GAIN_MAX`]) can bring under the
/// limit. A target louder than that reach cannot be clamped by
/// `global_gain` alone — the §C.1.5.4.3 outer loop would raise
/// scalefactors / `subblock_gain` instead, which is out of scope here —
/// so the result reports `satisfied == false` and carries the
/// [`GAIN_MAX`] fallback (`is[]` and `max_abs` at the coarsest gain).
#[must_use]
pub fn search_magnitude_clamp(
    xr: &[f32; NUM_LINES],
    gc_template: &GranuleChannel,
    sf: &ScaleFactors,
    sample_rate_hz: u32,
    version: MpegVersion,
) -> InnerLoopResult {
    search(xr, gc_template, sf, sample_rate_hz, version, |is| {
        max_abs(is) <= BIG_VALUES_LIMIT
    })
}

/// §C.1.5.4.4 bit-budget inner loop: find the smallest `global_gain`
/// whose **exact** §C.1.5.4.4.5 + §C.1.5.4.4.8 Huffman bit count
/// ([`exact_bit_count`]) of the quantized `is[]` is `≤ budget`.
///
/// As with [`search_magnitude_clamp`], `gc_template.global_gain` is the
/// searched field and all other fields plus `sf` are held fixed.
/// `gc_template`'s block-type / window-switching flags steer the
/// §C.1.5.4.4.6 SUBDIVIDE; the codebook per sub-region and the count1
/// quad table are chosen to minimise bits (§C.1.5.4.4.7). If no gain in
/// `[GAIN_MIN, GAIN_MAX]` fits the budget the result carries
/// `satisfied == false`; in practice a budget `≥ 0` is always met by the
/// all-zero quantization at the coarsest gain (cost `0`).
///
/// This is the exact codebook-length count, not the
/// [`coarse_bit_estimate`] placeholder of the r133 search. Unlike the
/// coarse estimate, the exact count is **not** monotone in `global_gain`
/// (Huffman codeword lengths are not monotone in magnitude and the
/// best codebook per region shifts as values shrink), so this search uses
/// the spec's upward `qquant + 1` scan ([`search_linear`]) rather than a
/// binary search, returning the smallest gain whose count fits.
#[must_use]
pub fn search_bit_budget(
    xr: &[f32; NUM_LINES],
    gc_template: &GranuleChannel,
    sf: &ScaleFactors,
    sample_rate_hz: u32,
    version: MpegVersion,
    budget: u64,
) -> InnerLoopResult {
    search_linear(xr, gc_template, sf, sample_rate_hz, version, |is| {
        // A range no codebook can code (corrupt input) is treated as
        // over-budget so the search raises the gain.
        exact_bit_count(is, gc_template).is_some_and(|c| c.bits as u64 <= budget)
    })
}

/// §C.1.5.4.4 bit-budget inner loop, gated on the **band-aligned**
/// SUBDIVIDE bit count: find the smallest `global_gain` whose exact
/// §C.1.5.4.4.5 + §C.1.5.4.4.8 Huffman bit total over the *wire-
/// representable* big-values partition ([`exact_bit_count_band_aligned`])
/// is `≤ budget`.
///
/// This differs from [`search_bit_budget`] only in which SUBDIVIDE the bit
/// count is taken against. [`search_bit_budget`] uses the default
/// pair-thirds heuristic ([`subdivide`]), whose region boundaries may land
/// mid-band — a partition the decoder's `region_boundaries` cannot
/// reconstruct, so the count it gates on is for a split the encoder can
/// never emit. This variant uses [`subdivide_bands`] for a long-family
/// granule (`block_type ∈ {Long, Start, End}`), snapping the same
/// §C.1.5.4.4.6 "~1/3 to region 0, ~1/4 to region 2" strategy to
/// scalefactor-band edges and the 4-bit / 3-bit `region0_count` /
/// `region1_count` field widths — the exact boundaries the decoder
/// reproduces. The bit count therefore matches the part2_3 length the
/// encoder will actually write, so the gain it picks fits the real wire
/// budget rather than an unrepresentable approximation of it. Short /
/// mixed blocks share the two-subregion blocksplit path, so for those this
/// is identical to [`search_bit_budget`].
///
/// As with [`search_bit_budget`], `gc_template.global_gain` is the
/// searched field and all other fields plus `sf` are held fixed; the count
/// is **not** monotone in the gain (Huffman codeword lengths are not
/// monotone in magnitude — see [`search_linear`]), so this uses the spec's
/// upward `qquant + 1` scan, returning the smallest gain whose count fits.
/// A budget `≥ 0` is always met by the all-zero quantization at the
/// coarsest gain (cost `0`).
#[must_use]
pub fn search_bit_budget_band_aligned(
    xr: &[f32; NUM_LINES],
    gc_template: &GranuleChannel,
    sf: &ScaleFactors,
    sample_rate_hz: u32,
    version: MpegVersion,
    budget: u64,
) -> InnerLoopResult {
    search_linear(xr, gc_template, sf, sample_rate_hz, version, |is| {
        // A range no codebook can code (corrupt input) is treated as
        // over-budget so the search raises the gain.
        exact_bit_count_band_aligned(is, gc_template, sample_rate_hz, version)
            .is_some_and(|c| c.bits as u64 <= budget)
    })
}

#[cfg(test)]
include!("inner_loop_tests.rs");
