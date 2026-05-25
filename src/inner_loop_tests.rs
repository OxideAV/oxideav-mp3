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
use crate::side_info::BlockType;

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

/// Bit-budget search: a tight budget forces a coarser gain than a loose
/// one; the chosen gain meets the budget and is minimal.
#[test]
fn bit_budget_tighter_means_coarser() {
    let gc = long_gc(false);
    let sf = ScaleFactors::default();
    let xr = flat(200.0);

    // Loose budget: the magnitude-clamp gain's own cost, doubled — easily met.
    let clamp = search_magnitude_clamp(&xr, &gc, &sf, SR, V);
    let loose_budget = coarse_bit_estimate(&clamp.is).saturating_mul(2).max(1);
    let loose = search_bit_budget(&xr, &gc, &sf, SR, V, loose_budget);
    assert!(loose.satisfied);
    assert!(coarse_bit_estimate(&loose.is) <= loose_budget);

    // Tight budget: a quarter of the clamp cost — needs a coarser gain.
    let tight_budget = (coarse_bit_estimate(&clamp.is) / 4).max(1);
    let tight = search_bit_budget(&xr, &gc, &sf, SR, V, tight_budget);
    assert!(tight.satisfied);
    assert!(coarse_bit_estimate(&tight.is) <= tight_budget);
    assert!(
        tight.global_gain >= loose.global_gain,
        "tighter budget {tight_budget} did not yield coarser-or-equal gain ({} vs {})",
        tight.global_gain,
        loose.global_gain
    );

    // Minimality of the tight result.
    if tight.global_gain > GAIN_MIN {
        let mut finer = gc;
        finer.global_gain = tight.global_gain - 1;
        let is_finer = quantize(&xr, &finer, &sf, SR, V);
        assert!(
            coarse_bit_estimate(&is_finer) > tight_budget,
            "tight gain {} not minimal for budget {tight_budget}",
            tight.global_gain
        );
    }
}

/// Bit budget of 0 forces the all-zero quantization (the coarsest gain
/// whose cost is 0).
#[test]
fn bit_budget_zero_drives_to_silence() {
    let gc = long_gc(false);
    let sf = ScaleFactors::default();
    let xr = flat(100.0);
    let r = search_bit_budget(&xr, &gc, &sf, SR, V, 0);
    assert!(r.satisfied, "budget 0 is met by the all-zero quantization");
    assert_eq!(coarse_bit_estimate(&r.is), 0);
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
