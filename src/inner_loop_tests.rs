// Unit tests for the §C.1.5.4.4 inner-loop `global_gain` search that
// wraps the §2.4.3.4.7 `quantize` primitive.
//
// The contract under test:
//  * `max|is| <= 8191` (BIG_VALUES_LIMIT) at the chosen gain — the
//    §2.4.1.7 big-values constraint — for every target.
//  * The chosen gain is the SMALLEST satisfying it: dropping the gain by
//    one violates the constraint (unless the gain is already GAIN_MIN).
//  * `max|is|` is monotone NON-INCREASING in `global_gain` (the basis
//    for the binary search) — verified by sweeping the whole range.
//  * Larger target magnitude => larger (or equal) chosen gain, because a
//    louder spectrum needs a coarser step to stay under 8191.
//  * `requantize(is)` at the chosen gain reproduces `xr` to within the
//    quantizer-grid bound (the round-trip is intrinsically lossy for an
//    arbitrary off-grid `xr`, so we bound the residual, not equate it).
//
// This file is `include!`d into `crate::inner_loop`, so the module's own
// `use` lines (MpegVersion, NUM_LINES, ScaleFactors, GranuleChannel,
// quantize) are already in scope.

use crate::requantize::requantize;
// `BlockType` is already in scope via the parent module's `use`.

const SR: u32 = 44100;
const V: MpegVersion = MpegVersion::Mpeg1;

/// Long-block GC scaffold; `global_gain` is a placeholder the search
/// overwrites.
fn long_gc(scalefac_scale: bool) -> GranuleChannel {
    GranuleChannel {
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
        scalefac_scale,
        count1table_select: false,
    }
}

/// Short-block (window-switched, non-mixed) GC scaffold.
fn short_gc(subblock_gain: [u8; 3]) -> GranuleChannel {
    GranuleChannel {
        part2_3_length: 0,
        big_values: 0,
        global_gain: 0,
        scalefac_compress: 0,
        window_switching_flag: true,
        block_type: BlockType::Short,
        mixed_block_flag: false,
        table_select: [0; 3],
        subblock_gain,
        region0_count: 0,
        region1_count: 0,
        preflag: false,
        scalefac_scale: false,
        count1table_select: false,
    }
}

/// Flat magnitude target: every line at `amp`.
fn flat(amp: f32) -> [f32; NUM_LINES] {
    [amp; NUM_LINES]
}

#[test]
fn max_abs_basics() {
    let mut is = [0i32; NUM_LINES];
    assert_eq!(max_abs(&is), 0);
    is[10] = -7;
    is[20] = 3;
    assert_eq!(max_abs(&is), 7);
    is[30] = -100;
    assert_eq!(max_abs(&is), 100);
}

#[test]
fn coarse_bit_estimate_zero_for_silence() {
    let is = [0i32; NUM_LINES];
    assert_eq!(coarse_bit_estimate(&is), 0);
}

#[test]
fn coarse_bit_estimate_counts_magnitude_plus_sign() {
    let mut is = [0i32; NUM_LINES];
    is[0] = 1; // bits(1)=1, +1 sign = 2
    is[1] = -1; // 2
    is[2] = 7; // bits(7)=3, +1 = 4
    is[3] = 8; // bits(8)=4, +1 = 5
    assert_eq!(coarse_bit_estimate(&is), 2 + 2 + 4 + 5);
}

/// `max|is|` must be monotone non-increasing across the full gain range
/// — the property the binary search relies on.
#[test]
fn max_abs_monotone_in_global_gain() {
    let gc = long_gc(false);
    let sf = ScaleFactors::default();
    let xr = flat(50.0);
    let mut prev = i32::MAX;
    for g in (GAIN_MIN as u16)..=(GAIN_MAX as u16) {
        let mut g_gc = gc;
        g_gc.global_gain = g as u8;
        let is = quantize(&xr, &g_gc, &sf, SR, V);
        let m = max_abs(&is);
        assert!(
            m <= prev,
            "max|is| rose from {prev} to {m} at global_gain {g}"
        );
        prev = m;
    }
}

/// The coarse bit estimate is likewise monotone non-increasing in gain.
#[test]
fn coarse_bits_monotone_in_global_gain() {
    let gc = long_gc(false);
    let sf = ScaleFactors::default();
    let xr = flat(30.0);
    let mut prev = u64::MAX;
    for g in (GAIN_MIN as u16)..=(GAIN_MAX as u16) {
        let mut g_gc = gc;
        g_gc.global_gain = g as u8;
        let is = quantize(&xr, &g_gc, &sf, SR, V);
        let bits = coarse_bit_estimate(&is);
        assert!(bits <= prev, "coarse bits rose from {prev} to {bits} at {g}");
        prev = bits;
    }
}

/// Magnitude clamp: result always within 8191, and it is the smallest
/// gain to do so (one less violates), for a range of amplitudes.
#[test]
fn magnitude_clamp_is_under_limit_and_minimal() {
    let gc = long_gc(false);
    let sf = ScaleFactors::default();
    for &amp in &[0.0f32, 0.5, 5.0, 50.0, 500.0, 5_000.0, 50_000.0] {
        let xr = flat(amp);
        let r = search_magnitude_clamp(&xr, &gc, &sf, SR, V);
        assert!(r.satisfied, "amp {amp} should always be satisfiable");
        assert!(
            r.max_abs <= BIG_VALUES_LIMIT,
            "amp {amp}: max_abs {} exceeds {BIG_VALUES_LIMIT}",
            r.max_abs
        );
        // Minimality: if not already at GAIN_MIN, the next-finer gain
        // must violate the limit.
        if r.global_gain > GAIN_MIN {
            let mut finer = gc;
            finer.global_gain = r.global_gain - 1;
            let is_finer = quantize(&xr, &finer, &sf, SR, V);
            assert!(
                max_abs(&is_finer) > BIG_VALUES_LIMIT,
                "amp {amp}: gain {} is not minimal (gain-1 still fits)",
                r.global_gain
            );
        }
        // The returned `is` matches a fresh quantize at the chosen gain.
        let mut chosen = gc;
        chosen.global_gain = r.global_gain;
        assert_eq!(r.is, quantize(&xr, &chosen, &sf, SR, V));
        assert_eq!(r.max_abs, max_abs(&r.is));
    }
}

/// Louder target => coarser (larger-or-equal) chosen gain.
#[test]
fn larger_target_yields_larger_gain() {
    let gc = long_gc(false);
    let sf = ScaleFactors::default();
    let mut prev_gain = GAIN_MIN;
    let mut saw_increase = false;
    for &amp in &[1.0f32, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0, 1.0e6] {
        let xr = flat(amp);
        let r = search_magnitude_clamp(&xr, &gc, &sf, SR, V);
        assert!(
            r.global_gain >= prev_gain,
            "amp {amp}: gain {} dropped below previous {prev_gain}",
            r.global_gain
        );
        if r.global_gain > prev_gain {
            saw_increase = true;
        }
        prev_gain = r.global_gain;
    }
    assert!(
        saw_increase,
        "gain never increased across a 6-decade amplitude sweep"
    );
}

/// Round-trip: requantize(is) at the chosen gain reproduces the flat
/// target within the quantizer-grid bound. For a flat target the grid
/// step near the target is `(k+1)^(4/3) - k^(4/3)` times the factor;
/// the residual per bin is bounded by one step. We verify each bin lies
/// within a generous multiple of the target (the chosen gain keeps the
/// grid near `xr`, so the relative error is small for large `k`).
#[test]
fn requantize_approximates_target() {
    let gc = long_gc(false);
    let sf = ScaleFactors::default();
    // Amplitudes large enough that the chosen gain yields a sizeable
    // `max|is|` (fine relative grid), so requantize lands close.
    for &amp in &[100.0f32, 1_000.0, 8_000.0] {
        let xr = flat(amp);
        let r = search_magnitude_clamp(&xr, &gc, &sf, SR, V);
        let mut chosen = gc;
        chosen.global_gain = r.global_gain;
        let xr_back = requantize(&r.is, &chosen, &sf, SR, V);
        // Relative residual per bin: the grid near a large `k` is fine,
        // so |xr_back - amp| / amp is small. Bound at 25% (loose, but
        // catches gross gain-search errors); the exact bound depends on
        // where `amp` falls between two `k^(4/3)` grid points.
        for (i, &b) in xr_back.iter().enumerate() {
            let rel = (b - amp).abs() / amp;
            assert!(
                rel <= 0.25,
                "bin {i}: amp {amp}, back {b}, rel residual {rel} > 0.25"
            );
        }
    }
}

/// All-zero target => gain GAIN_MIN, all-zero `is`, satisfied.
#[test]
fn zero_target_picks_min_gain() {
    let gc = long_gc(false);
    let sf = ScaleFactors::default();
    let xr = [0.0f32; NUM_LINES];
    let r = search_magnitude_clamp(&xr, &gc, &sf, SR, V);
    assert_eq!(r.global_gain, GAIN_MIN);
    assert_eq!(r.max_abs, 0);
    assert!(r.satisfied);
    assert!(r.is.iter().all(|&v| v == 0));
}

/// `exact_bit_count` of the chosen `is`, in bits — the search's own gate.
fn exact_bits(is: &[i32; NUM_LINES], gc: &GranuleChannel) -> u64 {
    exact_bit_count(is, gc).map(|c| c.bits as u64).unwrap_or(u64::MAX)
}

/// Bit-budget search: a tight budget forces a coarser gain than a loose
/// one; the chosen gain meets the **exact** budget and is minimal.
#[test]
fn bit_budget_tighter_means_coarser() {
    let gc = long_gc(false);
    let sf = ScaleFactors::default();
    let xr = flat(200.0);

    // Loose budget: the magnitude-clamp gain's own exact cost, doubled.
    let clamp = search_magnitude_clamp(&xr, &gc, &sf, SR, V);
    let clamp_cost = exact_bits(&clamp.is, &gc);
    let loose_budget = clamp_cost.saturating_mul(2).max(1);
    let loose = search_bit_budget(&xr, &gc, &sf, SR, V, loose_budget);
    assert!(loose.satisfied);
    assert!(exact_bits(&loose.is, &gc) <= loose_budget);

    // Tight budget: a quarter of the clamp cost — needs a coarser gain.
    let tight_budget = (clamp_cost / 4).max(1);
    let tight = search_bit_budget(&xr, &gc, &sf, SR, V, tight_budget);
    assert!(tight.satisfied);
    assert!(exact_bits(&tight.is, &gc) <= tight_budget);
    assert!(
        tight.global_gain >= loose.global_gain,
        "tighter budget {tight_budget} did not yield coarser-or-equal gain ({} vs {})",
        tight.global_gain,
        loose.global_gain
    );

    // Minimality of the tight result: the next-finer gain blows the budget.
    if tight.global_gain > GAIN_MIN {
        let mut finer = gc;
        finer.global_gain = tight.global_gain - 1;
        let is_finer = quantize(&xr, &finer, &sf, SR, V);
        assert!(
            exact_bits(&is_finer, &gc) > tight_budget,
            "tight gain {} not minimal for budget {tight_budget}",
            tight.global_gain
        );
    }
}

/// Bit budget of 0 forces the all-zero quantization (the coarsest gain
/// whose exact cost is 0).
#[test]
fn bit_budget_zero_drives_to_silence() {
    let gc = long_gc(false);
    let sf = ScaleFactors::default();
    let xr = flat(100.0);
    let r = search_bit_budget(&xr, &gc, &sf, SR, V, 0);
    assert!(r.satisfied, "budget 0 is met by the all-zero quantization");
    assert_eq!(exact_bits(&r.is, &gc), 0);
    assert!(r.is.iter().all(|&v| v == 0));
}

/// The search holds non-gain fields fixed: a scaled-scalefactor config
/// still yields a satisfying, minimal gain.
#[test]
fn scalefac_scale_config_held_fixed() {
    let gc = long_gc(true); // scalefac_scale = 1
    let mut sf = ScaleFactors::default();
    for (sfb, v) in sf.long.iter_mut().enumerate() {
        *v = (sfb as u8) % 4;
    }
    let xr = flat(300.0);
    let r = search_magnitude_clamp(&xr, &gc, &sf, SR, V);
    assert!(r.satisfied);
    assert!(r.max_abs <= BIG_VALUES_LIMIT);
    // The returned is reflects scalefac_scale=true + non-zero sf.
    let mut chosen = gc;
    chosen.global_gain = r.global_gain;
    assert_eq!(r.is, quantize(&xr, &chosen, &sf, SR, V));
}

/// Short-block target also yields a within-limit minimal gain.
#[test]
fn short_block_magnitude_clamp() {
    let gc = short_gc([0, 1, 2]);
    let sf = ScaleFactors::default();
    let xr = flat(400.0);
    let r = search_magnitude_clamp(&xr, &gc, &sf, SR, V);
    assert!(r.satisfied);
    assert!(r.max_abs <= BIG_VALUES_LIMIT);
    if r.global_gain > GAIN_MIN {
        let mut finer = gc;
        finer.global_gain = r.global_gain - 1;
        let is_finer = quantize(&xr, &finer, &sf, SR, V);
        assert!(max_abs(&is_finer) > BIG_VALUES_LIMIT);
    }
}

/// Raising `global_gain` only divides by `2^((g-210)/4)`, so for a
/// fixed scalefactor config there is a finite largest amplitude the
/// clamp can reach by the coarsest gain (GAIN_MAX). With `sf == 0` and
/// `scalefac_scale == 0`, the factor at GAIN_MAX is `2^((255-210)/4)`,
/// so the largest clampable `xr` is `factor * 8191^(4/3) ≈ 4.0e8`.
/// A target within that reach is satisfied; one beyond it correctly
/// reports `satisfied == false` and falls back to the coarsest gain
/// (the outer loop / scalefactors — not in scope this round — would
/// extend the range). This boundary test pins both sides.
#[test]
fn clamp_reach_is_bounded_by_max_gain() {
    let gc = long_gc(false);
    let sf = ScaleFactors::default();
    let factor = ((GAIN_MAX as f32 - 210.0) / 4.0).exp2();
    let reach = factor * (BIG_VALUES_LIMIT as f32).powf(4.0 / 3.0);

    // Comfortably within reach: satisfied, under limit, gain == GAIN_MAX
    // (only the coarsest gain keeps such a loud signal in range).
    let inside = flat(reach * 0.5);
    let r_in = search_magnitude_clamp(&inside, &gc, &sf, SR, V);
    assert!(r_in.satisfied, "an amplitude inside reach must be clampable");
    assert!(r_in.max_abs <= BIG_VALUES_LIMIT);

    // Far beyond reach: even GAIN_MAX cannot clamp it. The result must
    // report the failure honestly and carry the GAIN_MAX fallback.
    let outside = flat(reach * 100.0);
    let r_out = search_magnitude_clamp(&outside, &gc, &sf, SR, V);
    assert!(
        !r_out.satisfied,
        "an amplitude beyond GAIN_MAX's reach must report satisfied == false"
    );
    assert_eq!(r_out.global_gain, GAIN_MAX);
    assert!(r_out.max_abs > BIG_VALUES_LIMIT);
    // The fallback `is` is a genuine quantization at GAIN_MAX.
    let mut max_gc = gc;
    max_gc.global_gain = GAIN_MAX;
    assert_eq!(r_out.is, quantize(&outside, &max_gc, &sf, SR, V));
}

// =====================================================================
// Exact §C.1.5.4.4.5 + §C.1.5.4.4.8 Huffman bit count tests.
//
// The truth values are the Table 3-B.7 codeword lengths transcribed in
// huffman_tables.rs, summed by hand here, plus the explicit sign and
// `linbits` bits the spec's "bitz"/"countltable" tables fold in. Each
// count must equal what `decode_huffman` would read back — the round-
// trip the existing huffman_tests.rs decoder tests already pin.
// =====================================================================

// `count_huffman_bits`, `partition_split`, `choose_best_count1_table`,
// `choose_best_table_for_region` are already in scope via the parent
// module's `use`; only `count1_bits` needs importing here.
use crate::huffman::count1_bits;

/// All-zero `is[]` costs 0 bits (no big-values pairs, no count1 quads).
#[test]
fn exact_count_silence_is_zero() {
    let is = [0i32; NUM_LINES];
    let gc = long_gc(false);
    let c = exact_bit_count(&is, &gc).unwrap();
    assert_eq!(c.bits, 0);
    assert_eq!(c.split.big_pairs, 0);
    assert_eq!(c.split.count1_quads, 0);
}

/// A single big-values pair coded with table 1: entry (1,0) is hcod "01"
/// of length 2 (huffman_tables.rs TABLE1_E[2]), plus one sign bit for the
/// non-zero x → 3 bits. (Matches the region_split_uses_two_tables decoder
/// test, where (1,0) under table 1 read "01" + 1 sign.)
#[test]
fn count_huffman_bits_single_table1_pair() {
    let mut is = [0i32; NUM_LINES];
    is[0] = 1;
    is[1] = 0;
    // One big-values pair, region 0 = lines 0..2, all under table 1.
    let bits = count_huffman_bits(&is, 1, (2, 2), [1, 0, 0], 0, false).unwrap();
    assert_eq!(bits, 2 + 1);
}

/// big_pair (1,1) under table 1: entry (1,1) is hcod "000" length 3
/// (TABLE1_E[3]) plus two sign bits = 5 bits.
#[test]
fn count_huffman_bits_table1_pair_two_signs() {
    let mut is = [0i32; NUM_LINES];
    is[0] = -1;
    is[1] = 1;
    let bits = count_huffman_bits(&is, 1, (2, 2), [1, 0, 0], 0, false).unwrap();
    assert_eq!(bits, 3 + 2);
}

/// linbits ESC: table 16 has linbits=1. A pair (20, 0): the Huffman
/// symbol is min(15,20)=15 → TABLE16 entry (15,0); magnitude 20 ≥ 15 so a
/// 1-bit linbits field follows for x; plus one sign bit for x. The y=0
/// component adds no sign and no linbits. Expected =
/// len(TABLE16[15][0]) + 1 (linbits) + 1 (sign x).
#[test]
fn count_huffman_bits_linbits_escape_table16() {
    // Look up the canonical codeword length for (15,0) in table 16 by
    // costing the pair (15,0) (no ESC, no extra magnitude) first.
    let mut base = [0i32; NUM_LINES];
    base[0] = 15;
    base[1] = 0;
    // (15,0): len + linbits(1, since 15≥15) + sign(1) for table 16.
    let bits_15 = count_huffman_bits(&base, 1, (2, 2), [16, 0, 0], 0, false).unwrap();

    // (20,0): same codeword (min(15,20)=15), same single linbits field
    // (linbits is a fixed-width PCM field, NOT magnitude-dependent), same
    // single sign — so the bit count is identical to (15,0).
    let mut big = [0i32; NUM_LINES];
    big[0] = 20;
    big[1] = 0;
    let bits_20 = count_huffman_bits(&big, 1, (2, 2), [16, 0, 0], 0, false).unwrap();
    assert_eq!(bits_15, bits_20);
    // And the magnitude-15 cost is (codeword len) + 1 linbits + 1 sign.
    // The codeword len for (15,0) in table 16 is whatever TABLE16 holds;
    // we only assert the +2 (linbits + sign) overhead is present by
    // comparing to a magnitude-14 pair (no ESC, just codeword + sign).
    let mut small = [0i32; NUM_LINES];
    small[0] = 14;
    small[1] = 0;
    let bits_14 = count_huffman_bits(&small, 1, (2, 2), [16, 0, 0], 0, false).unwrap();
    // (15,*) costs exactly one extra linbits bit vs (14,*) for the same
    // sign count, modulo the codeword-length difference between the
    // symbols 15 and 14. We instead assert the linbits field is counted
    // by comparing table 16 (linbits=1) to table 17 (linbits=2) on the
    // same (15,0): table 17 must cost exactly one more bit.
    let bits_15_t17 = count_huffman_bits(&base, 1, (2, 2), [17, 0, 0], 0, false).unwrap();
    assert_eq!(
        bits_15_t17,
        bits_15 + 1,
        "table 17 (linbits=2) must cost 1 more bit than table 16 (linbits=1) on (15,0)"
    );
    let _ = bits_14;
}

/// count1 quad table A: pattern (1,1,1,1) is hcod "000001" length 6
/// (QUAD_A[0b1111]) plus four sign bits = 10 bits — the
/// count1_quad_a_all_ones_with_signs decoder test's exact read length.
#[test]
fn count1_bits_quad_a_all_ones() {
    let mut is = [0i32; NUM_LINES];
    is[0] = -1;
    is[1] = -1;
    is[2] = -1;
    is[3] = -1;
    // No big-values; one count1 quad, table A.
    let bits = count1_bits(&is, 0, 4, false);
    assert_eq!(bits, 6 + 4);
}

/// count1 quad table A: the all-zero pattern is hcod "1" length 1, no
/// signs → 1 bit (the count1_quad_a_zero_pattern decoder test).
#[test]
fn count1_bits_quad_a_zero_pattern() {
    let is = [0i32; NUM_LINES];
    let bits = count1_bits(&is, 0, 4, false);
    assert_eq!(bits, 1);
}

/// count1 quad table B: a 4-bit flat code + one sign per non-zero. Pattern
/// (1,0,1,0) = 4 code bits + 2 sign bits = 6 bits (the
/// count1_quad_b_trivial_pattern decoder test read "0101" + 2 signs).
#[test]
fn count1_bits_quad_b_pattern() {
    let mut is = [0i32; NUM_LINES];
    is[0] = 1;
    is[1] = 0;
    is[2] = 1;
    is[3] = 0;
    let bits = count1_bits(&is, 0, 4, true);
    assert_eq!(bits, 4 + 2);
}

/// `choose_best_count1_table` picks the smaller of tables A / B. For the
/// all-ones quad, table A costs 6+4=10 while table B costs 4+4=8, so B
/// (true) wins.
#[test]
fn choose_count1_table_prefers_smaller() {
    let mut is = [0i32; NUM_LINES];
    for v in is[..4].iter_mut() {
        *v = 1;
    }
    let (table_b, bits) = choose_best_count1_table(&is, 0, 4);
    assert!(table_b, "table B (4+4=8) should beat table A (6+4=10)");
    assert_eq!(bits, 8);
    // For the all-zero quad, table A's "1" (1 bit) beats table B's 4 bits.
    let zero = [0i32; NUM_LINES];
    let (tz, bz) = choose_best_count1_table(&zero, 0, 4);
    assert!(!tz, "table A (1 bit) should beat table B (4 bits) for a zero quad");
    assert_eq!(bz, 1);
}

/// `choose_best_table_for_region` returns the minimum-bit table and its
/// cost. For a pair (1,1), table 1's (1,1) entry "000" (len 3) + 2 signs
/// = 5 should be among the candidates; the chosen cost must be the global
/// minimum, i.e. no other selectable table codes it in fewer bits.
#[test]
fn choose_region_table_is_minimal() {
    let mut is = [0i32; NUM_LINES];
    is[0] = 1;
    is[1] = 1;
    let (best_tbl, best_bits) = choose_best_table_for_region(&is, 0, 2).unwrap();
    // The chosen cost equals re-counting that pair under the chosen table.
    let recount = count_huffman_bits(&is, 1, (2, 2), [best_tbl, 0, 0], 0, false).unwrap();
    assert_eq!(best_bits, recount);
    // It is a true minimum: every selectable table costs ≥ best_bits.
    for &t in crate::huffman::SELECTABLE_BIG_TABLES.iter() {
        if let Some((tt, bb)) = choose_best_table_for_region(&is, 0, 2) {
            assert!(bb <= best_bits || tt == best_tbl);
        }
        // Direct per-table cost ≥ best.
        if let Some(b) = exact_pair_cost(&is, t) {
            assert!(b >= best_bits, "table {t} costs {b} < chosen min {best_bits}");
        }
    }
}

/// Helper: cost one big-values pair `is[0..2]` under table `t` (or None
/// if not codable), via the public count entry point.
fn exact_pair_cost(is: &[i32; NUM_LINES], t: u8) -> Option<usize> {
    count_huffman_bits(is, 1, (2, 2), [t, 0, 0], 0, false)
}

/// End-to-end on a small hand-built `is[]`: two big-values pairs in
/// region 0 (table 1) plus one count1 quad (table A), summed by hand.
/// Pairs: (1,0)→"01"+1sign=3 ; (1,1)→"000"+2sign=5. count1 (1,1,1,1)→
/// 6+4=10. Total = 3+5+10 = 18.
#[test]
fn count_huffman_bits_multi_region_sum() {
    let mut is = [0i32; NUM_LINES];
    // big_values: 2 pairs (lines 0..4).
    is[0] = 1;
    is[1] = 0; // pair 0 -> 3
    is[2] = -1;
    is[3] = 1; // pair 1 -> 5
    // count1: 1 quad (lines 4..8).
    is[4] = 1;
    is[5] = -1;
    is[6] = 1;
    is[7] = -1; // -> 10
    let bits = count_huffman_bits(&is, 2, (4, 4), [1, 0, 0], 1, false).unwrap();
    assert_eq!(bits, 3 + 5 + 10);
}

/// `partition_split` strips the trailing zero pairs and assigns the
/// trailing ≤1 quad run to count1. For an `is[]` with magnitude-2 in the
/// low lines and magnitude-1 in the upper, the low part is big-values and
/// the upper ≤1 run is count1.
#[test]
fn partition_split_separates_bigvalues_and_count1() {
    let mut is = [0i32; NUM_LINES];
    // big-values region: a magnitude-≥2 pair at lines 0..2.
    is[0] = 3;
    is[1] = 2;
    // a ≤1 quad at lines 4..8 (lines 2,3 are zero, part of big-values).
    is[4] = 1;
    is[5] = 1;
    is[6] = 1;
    is[7] = 0;
    let split = partition_split(&is);
    // Trailing nonzero ends at line 6 (is[6]=1) → nonzero_lines rounds to
    // 8 (even). count1 aligns to multiple of 4 = 8; the quad 4..8 is all
    // ≤1 → count1. The quad 0..4 contains magnitude 3 → big-values.
    assert_eq!(split.count1_quads, 1, "the trailing ≤1 quad is count1");
    assert_eq!(split.big_pairs, 2, "lines 0..4 (2 pairs) are big-values");
    // Exact count is self-consistent with the split.
    let gc = long_gc(false);
    let c = exact_bit_count(&is, &gc).unwrap();
    assert_eq!(c.split, split);
}

/// The exact count is NOT monotone in `global_gain` (unlike the coarse
/// estimate): raising the gain shrinks every `|is_i|`, but Huffman
/// codeword lengths are not monotone in magnitude and the best codebook
/// per region shifts, so a coarser quantization can cost a few more bits
/// than a finer one. This is exactly why `search_bit_budget` uses the
/// spec's upward `qquant + 1` scan rather than a binary search. We assert
/// the count is well-defined for every gain whose `is[]` satisfies the
/// §C.1.5.4.4.2 magnitude clamp (`max|is| <= BIG_VALUES_LIMIT`) and
/// exhibits at least one non-monotone step on a flat spectrum
/// (documenting the property the linear scan exists to handle). It must
/// also reach 0 at GAIN_MAX.
///
/// Gains finer than the magnitude-clamp lower bound are skipped — the
/// r154 linbits-reach filter in
/// [`crate::huffman::choose_best_table_for_region`] correctly returns
/// `None` for ranges whose `max|is|` exceeds every codebook's reach
/// (table 23's `linbits=13` reach of 8206 covers the 8191 clamp but
/// nothing past it), so the exact-bit-count predicate is well-defined
/// only on the clamp-respecting subset, which is exactly the subset
/// the surrounding `search_bit_budget` walks (it starts from the gain
/// `search_magnitude_clamp` returned).
#[test]
fn exact_bits_not_strictly_monotone_but_well_defined() {
    let gc = long_gc(false);
    let sf = ScaleFactors::default();
    let xr = flat(30.0);
    // Lower bound of the well-defined range: the smallest gain whose
    // quantized `is[]` fits the §C.1.5.4.4.2 8191 clamp. The §C.1.5.4.4
    // outer scan starts from here, never from the unconditional
    // `GAIN_MIN`.
    let clamp = search_magnitude_clamp(&xr, &gc, &sf, SR, V);
    assert!(
        clamp.satisfied,
        "flat(30.0) must be clamp-reachable for the test premise to hold"
    );
    let gain_lo = clamp.global_gain;
    let mut prev = u64::MAX;
    let mut saw_rise = false;
    for g in (gain_lo as u16)..=(GAIN_MAX as u16) {
        let mut g_gc = gc;
        g_gc.global_gain = g as u8;
        let is = quantize(&xr, &g_gc, &sf, SR, V);
        // Best-table selection always codes a clamp-respecting spectrum.
        let bits = exact_bit_count(&is, &g_gc)
            .expect("a clamp-respecting quantized spectrum is always codable")
            .bits as u64;
        if g > gain_lo as u16 && bits > prev {
            saw_rise = true;
        }
        prev = bits;
    }
    assert!(
        saw_rise,
        "expected the exact count to be non-monotone somewhere on a flat \
         spectrum (the reason search_bit_budget scans rather than bisects)"
    );
    // At the coarsest gain the whole spectrum quantizes to zero → 0 bits.
    let mut max_gc = gc;
    max_gc.global_gain = GAIN_MAX;
    let is_max = quantize(&xr, &max_gc, &sf, SR, V);
    assert_eq!(exact_bit_count(&is_max, &max_gc).unwrap().bits, 0);
}

/// `search_bit_budget` returns the SMALLEST gain whose exact count fits:
/// the gain one finer (if any) must overflow the budget — the upward scan
/// guarantees this even though the count is not globally monotone.
#[test]
fn bit_budget_finer_gain_overflows() {
    let gc = long_gc(false);
    let sf = ScaleFactors::default();
    let xr = flat(120.0);
    // Pick a budget between the finest and coarsest costs.
    let fine = {
        let mut g = gc;
        g.global_gain = 180;
        quantize(&xr, &g, &sf, SR, V)
    };
    let budget = (exact_bits(&fine, &gc) / 2).max(1);
    let r = search_bit_budget(&xr, &gc, &sf, SR, V, budget);
    assert!(r.satisfied);
    assert!(exact_bits(&r.is, &gc) <= budget);
    if r.global_gain > GAIN_MIN {
        let mut finer = gc;
        finer.global_gain = r.global_gain - 1;
        let is_finer = quantize(&xr, &finer, &sf, SR, V);
        assert!(
            exact_bits(&is_finer, &gc) > budget,
            "gain {} is not the smallest fitting gain (gain-1 also fits)",
            r.global_gain
        );
    }
}

/// The exact count equals the bits a round-trip through `decode_huffman`
/// would read: build an `is[]`, count it exactly, then confirm the count
/// equals the codeword lengths re-derived from the same Table 3-B.7
/// entries the decoder uses. (Cross-checked structurally via the per-pair
/// and per-quad tests above; here we assert the whole-granule sum equals
/// big-values + count1 with no double counting or omission.)
#[test]
fn exact_count_equals_region_plus_count1_decomposition() {
    let mut is = [0i32; NUM_LINES];
    is[0] = 2;
    is[1] = 3; // big pair, region 0 (max value 3 → needs a 4x4 table)
    is[2] = 1;
    is[3] = 0; // big pair, region 0
    is[4] = 1;
    is[5] = 1;
    is[6] = 0;
    is[7] = 1; // count1 quad
    // Table 5 is 4x4 (xlen=4), so both (2,3) and (1,0) are codable.
    let big_bits = count_huffman_bits(&is, 2, (4, 4), [5, 0, 0], 0, false).unwrap();
    let c1_bits = count1_bits(&is, 4, 8, false);
    let total = count_huffman_bits(&is, 2, (4, 4), [5, 0, 0], 1, false).unwrap();
    assert_eq!(total, big_bits + c1_bits);
}

