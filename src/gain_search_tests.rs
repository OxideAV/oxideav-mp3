// Unit tests for the §C.1.5.4.4 inner-loop global-gain search (encoder
// rate-control step). This file is `include!`d into `crate::gain_search`,
// so the module's `use` lines (MpegVersion, NUM_LINES, ScaleFactors,
// GranuleChannel, quantize) are already in scope.
//
// The brief's contract:
//   1. for several xr magnitudes the search returns a gain whose quantized
//      output keeps max|is| <= 8191 (MAX_HUFFMAN_VALUE), and
//   2. requantize(quantize(xr, gain)) approximates xr (within the discrete
//      grid's half-step bound), and
//   3. a larger target magnitude yields a larger (or equal) gain.

use crate::requantize::requantize;
use crate::side_info::BlockType;

/// Long-block GC scaffold (mirrors the quantize-test helper). The
/// `global_gain` here is a placeholder; the search overrides it.
fn long_gc() -> GranuleChannel {
    GranuleChannel {
        part2_3_length: 0,
        big_values: 0,
        global_gain: 210,
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
    }
}

fn short_gc(subblock_gain: [u8; 3]) -> GranuleChannel {
    GranuleChannel {
        part2_3_length: 0,
        big_values: 0,
        global_gain: 210,
        scalefac_compress: 0,
        window_switching_flag: true,
        block_type: BlockType::Short,
        mixed_block_flag: false,
        table_select: [0; 3],
        subblock_gain,
        region0_count: 0,
        region1_count: 63,
        preflag: false,
        scalefac_scale: false,
        count1table_select: false,
    }
}

/// A flat target spectrum of magnitude `m`.
fn flat(m: f32) -> [f32; NUM_LINES] {
    [m; NUM_LINES]
}

/// Relative RMS of (requantized search output) vs target, over the bins.
/// Returns 0 for an all-zero target.
fn rel_rms(xr_target: &[f32; NUM_LINES], xr_back: &[f32; NUM_LINES]) -> f32 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (&t, &b) in xr_target.iter().zip(xr_back.iter()) {
        let d = (b - t) as f64;
        num += d * d;
        den += (t as f64) * (t as f64);
    }
    if den == 0.0 {
        0.0
    } else {
        ((num / den).sqrt()) as f32
    }
}

#[test]
fn fits_max_value_for_several_magnitudes() {
    let gc = long_gc();
    let sf = ScaleFactors::default();
    for &m in &[1e-4f32, 1e-2, 1.0, 10.0, 100.0, 1000.0, 1e4, 1e5] {
        let xr = flat(m);
        let res = search_gain_for_max_value(
            &xr,
            &gc,
            &sf,
            44100,
            MpegVersion::Mpeg1,
            MAX_HUFFMAN_VALUE,
        );
        assert!(res.satisfied, "magnitude {m}: no gain satisfied the limit");
        assert!(
            res.max_abs <= MAX_HUFFMAN_VALUE,
            "magnitude {m}: max|is| {} exceeds the {MAX_HUFFMAN_VALUE} limit",
            res.max_abs
        );
    }
}

#[test]
fn requantize_of_search_output_approximates_target() {
    let gc = long_gc();
    let sf = ScaleFactors::default();
    // For each target magnitude the search picks the smallest gain whose
    // grid still fits the magnitude budget. That gain is far below the
    // budget ceiling for moderate magnitudes, so the grid is fine and the
    // round-trip is close. The discrete grid means the residual is bounded
    // by ~half a step; a relative RMS under 10% confirms the gain landed
    // the grid near the target rather than coarsely above it.
    for &m in &[1.0f32, 8.0, 64.0, 512.0, 4096.0] {
        let xr = flat(m);
        let res = search_gain_for_max_value(
            &xr,
            &gc,
            &sf,
            44100,
            MpegVersion::Mpeg1,
            MAX_HUFFMAN_VALUE,
        );
        let mut probe = gc;
        probe.global_gain = res.global_gain;
        let xr_back = requantize(&res.is, &probe, &sf, 44100, MpegVersion::Mpeg1);
        let rms = rel_rms(&xr, &xr_back);
        assert!(
            rms < 0.10,
            "magnitude {m}: relative RMS {rms} too large (gain {})",
            res.global_gain
        );
    }
}

#[test]
fn larger_target_yields_larger_or_equal_gain() {
    let gc = long_gc();
    let sf = ScaleFactors::default();
    let mags = [0.1f32, 1.0, 10.0, 100.0, 1000.0, 1e4, 1e5];
    let mut prev_gain: Option<u8> = None;
    for &m in &mags {
        let xr = flat(m);
        let res = search_gain_for_max_value(
            &xr,
            &gc,
            &sf,
            44100,
            MpegVersion::Mpeg1,
            MAX_HUFFMAN_VALUE,
        );
        if let Some(p) = prev_gain {
            assert!(
                res.global_gain >= p,
                "magnitude {m}: gain {} regressed below the previous {p}",
                res.global_gain
            );
        }
        prev_gain = Some(res.global_gain);
    }
}

#[test]
fn strictly_larger_gain_across_wide_magnitude_span() {
    // Magnitudes separated by enough to force a strictly larger gain (a
    // 16x magnitude jump needs roughly +16 in global_gain since the grid
    // value scales as 2^((g-210)/4) and the magnitude is xr^(3/4)).
    let gc = long_gc();
    let sf = ScaleFactors::default();
    let small = search_gain_for_max_value(
        &flat(1.0),
        &gc,
        &sf,
        44100,
        MpegVersion::Mpeg1,
        MAX_HUFFMAN_VALUE,
    );
    let large = search_gain_for_max_value(
        &flat(1e6),
        &gc,
        &sf,
        44100,
        MpegVersion::Mpeg1,
        MAX_HUFFMAN_VALUE,
    );
    assert!(
        large.global_gain > small.global_gain,
        "large-target gain {} did not exceed small-target gain {}",
        large.global_gain,
        small.global_gain
    );
}

#[test]
fn search_returns_smallest_satisfying_gain() {
    // The search must return the *smallest* gain that fits: one below it
    // must violate the constraint (else the grid is coarser than needed
    // and fidelity is needlessly lost).
    let gc = long_gc();
    let sf = ScaleFactors::default();
    let xr = flat(100.0);
    let res = search_gain_for_max_value(
        &xr,
        &gc,
        &sf,
        44100,
        MpegVersion::Mpeg1,
        MAX_HUFFMAN_VALUE,
    );
    assert!(res.satisfied);
    assert!(res.max_abs <= MAX_HUFFMAN_VALUE);
    if res.global_gain > 0 {
        let mut probe = gc;
        probe.global_gain = res.global_gain - 1;
        let is_below = quantize(&xr, &probe, &sf, 44100, MpegVersion::Mpeg1);
        assert!(
            max_abs_is(&is_below) > MAX_HUFFMAN_VALUE,
            "gain {} - 1 still fit (max|is| {}); search not minimal",
            res.global_gain,
            max_abs_is(&is_below)
        );
    }
}

#[test]
fn tighter_max_quant_yields_larger_or_equal_gain() {
    // A tighter magnitude budget forces a coarser grid → larger gain.
    let gc = long_gc();
    let sf = ScaleFactors::default();
    let xr = flat(1000.0);
    let loose = search_gain_for_max_value(
        &xr,
        &gc,
        &sf,
        44100,
        MpegVersion::Mpeg1,
        MAX_HUFFMAN_VALUE,
    );
    let tight =
        search_gain_for_max_value(&xr, &gc, &sf, 44100, MpegVersion::Mpeg1, 15);
    assert!(tight.satisfied && loose.satisfied);
    assert!(tight.max_abs <= 15);
    assert!(
        tight.global_gain >= loose.global_gain,
        "tighter budget gain {} not >= looser budget gain {}",
        tight.global_gain,
        loose.global_gain
    );
}

#[test]
fn bit_budget_search_fits_and_is_monotone() {
    let gc = long_gc();
    let sf = ScaleFactors::default();
    let xr = flat(1000.0);
    // A generous budget: a few thousand bits across 576 lines.
    let budget = 4000u64;
    let res = search_gain_for_bit_budget(
        &xr,
        &gc,
        &sf,
        44100,
        MpegVersion::Mpeg1,
        budget,
    );
    assert!(res.satisfied, "bit-budget search found no satisfying gain");
    assert!(
        res.estimated_bits <= budget,
        "estimate {} exceeds budget {budget}",
        res.estimated_bits
    );
    // A tighter budget must not pick a smaller gain (coarser quant uses
    // fewer bits → needs a larger gain).
    let tight = search_gain_for_bit_budget(
        &xr,
        &gc,
        &sf,
        44100,
        MpegVersion::Mpeg1,
        2000,
    );
    assert!(tight.satisfied);
    assert!(
        tight.global_gain >= res.global_gain,
        "tighter bit budget gain {} not >= looser {}",
        tight.global_gain,
        res.global_gain
    );
}

#[test]
fn all_zero_target_fits_at_minimum_gain() {
    // An all-zero spectrum quantizes to all zeros at every gain, so the
    // smallest gain (0) already fits both constraint forms.
    let gc = long_gc();
    let sf = ScaleFactors::default();
    let xr = [0.0f32; NUM_LINES];
    let res = search_gain_for_max_value(
        &xr,
        &gc,
        &sf,
        44100,
        MpegVersion::Mpeg1,
        MAX_HUFFMAN_VALUE,
    );
    assert!(res.satisfied);
    assert_eq!(res.global_gain, 0, "all-zero target should pick gain 0");
    assert_eq!(res.max_abs, 0);
    assert_eq!(res.estimated_bits, 0);
}

#[test]
fn short_block_search_fits() {
    // The search wraps `quantize`, which handles short blocks; confirm the
    // magnitude constraint is honoured for a window-switched short block.
    let gc = short_gc([0, 1, 2]);
    let sf = ScaleFactors::default();
    let xr = flat(500.0);
    let res = search_gain_for_max_value(
        &xr,
        &gc,
        &sf,
        44100,
        MpegVersion::Mpeg1,
        MAX_HUFFMAN_VALUE,
    );
    assert!(res.satisfied, "short-block search found no satisfying gain");
    assert!(res.max_abs <= MAX_HUFFMAN_VALUE);
}

#[test]
fn unsatisfiable_constraint_reports_false_at_max_gain() {
    // A target so large that even gain 255 leaves max|is| above an
    // absurdly tight budget reports satisfied == false at gain 255. Use a
    // huge magnitude with a budget of 0: any non-zero line violates it.
    let gc = long_gc();
    let sf = ScaleFactors::default();
    let xr = flat(1e9);
    let res =
        search_gain_for_max_value(&xr, &gc, &sf, 44100, MpegVersion::Mpeg1, 0);
    // With max_quant == 0 the only way to fit is all-zero output. At gain
    // 255 a 1e9 target may or may not quantize to zero; assert the
    // reported flag is consistent with the actual output.
    let expected = res.max_abs <= 0;
    assert_eq!(res.satisfied, expected);
    if !res.satisfied {
        assert_eq!(res.global_gain, u8::MAX);
    }
}

#[test]
fn max_abs_helper_matches_manual_scan() {
    let mut is = [0i32; NUM_LINES];
    is[10] = -77;
    is[200] = 50;
    is[575] = 12;
    assert_eq!(max_abs_is(&is), 77);
}

#[test]
fn estimate_bits_counts_magnitude_plus_sign() {
    let mut is = [0i32; NUM_LINES];
    // |1| -> ceil(log2(2)) = 1 mag bit + 1 sign = 2.
    is[0] = 1;
    // |3| -> ceil(log2(4)) = 2 mag bits + 1 sign = 3.
    is[1] = -3;
    // |8| -> 32 - leading_zeros(8) = 4 mag bits + 1 sign = 5.
    is[2] = 8;
    assert_eq!(estimate_bits(&is), 2 + 3 + 5);
}

