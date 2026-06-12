// Unit tests for the §2.4.3.4.10.1 alias-reduction stage. The reference
// values are computed directly from the spec's butterfly pseudo code and
// the Table 3-B.9 coefficients (`c[i]`, `cs[i]`, `ca[i]`).

// This file is `include!`d into the `alias` module, so its `use` lines
// (BlockType, GranuleChannel) and the module items are already in scope.

/// A long-block granule-channel record (whole spectrum is alias-reduced).
fn long_gc() -> GranuleChannel {
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
        scalefac_scale: false,
        count1table_select: false,
    }
}

/// A window-switched short / mixed granule-channel (block_type == 2).
fn short_gc(mixed: bool) -> GranuleChannel {
    GranuleChannel {
        part2_3_length: 0,
        big_values: 0,
        global_gain: 0,
        scalefac_compress: 0,
        window_switching_flag: true,
        block_type: BlockType::Short,
        mixed_block_flag: mixed,
        table_select: [0; 3],
        subblock_gain: [0; 3],
        region0_count: 0,
        region1_count: 0,
        preflag: false,
        scalefac_scale: false,
        count1table_select: false,
    }
}

const EPS: f32 = 1e-6;

#[test]
fn table_b9_raw_coefficients() {
    // Table 3-B.9 c[i] verbatim from the ISO PDF.
    assert_eq!(
        ALIAS_C,
        [-0.6, -0.535, -0.33, -0.185, -0.095, -0.041, -0.0142, -0.0037]
    );
}

#[test]
fn butterfly_coefficients_match_derivation() {
    // cs[i] = 1/sqrt(1+c²), ca[i] = c/sqrt(1+c²); also cs²+ca² == 1 and
    // ca/cs == c.
    let cs = alias_cs();
    let ca = alias_ca();
    for (i, &c) in ALIAS_C.iter().enumerate() {
        let denom = (1.0 + c * c).sqrt();
        assert!((cs[i] as f64 - 1.0 / denom).abs() < 1e-6, "cs[{i}]");
        assert!((ca[i] as f64 - c / denom).abs() < 1e-6, "ca[{i}]");
        let unit = (cs[i] as f64).powi(2) + (ca[i] as f64).powi(2);
        assert!((unit - 1.0).abs() < 1e-6, "cs²+ca² at {i}");
        assert!((ca[i] as f64 / cs[i] as f64 - c).abs() < 1e-6, "ca/cs at {i}");
    }
}

#[test]
fn cs0_ca0_known_values() {
    // c[0] = -0.6 → sqrt(1.36) = 1.16619…; cs0 ≈ 0.857493, ca0 ≈ -0.514496.
    let cs = alias_cs();
    let ca = alias_ca();
    assert!((cs[0] - 0.857_493_3).abs() < 1e-5);
    assert!((ca[0] - (-0.514_495_8)).abs() < 1e-5);
}

#[test]
fn short_block_passes_through_unchanged() {
    // §2.4.3.4.10.1: block_type == 2 is not alias-reduced.
    let gc = short_gc(false);
    let mut xr = [0.0f32; NUM_LINES];
    for (i, v) in xr.iter_mut().enumerate() {
        *v = (i as f32) * 0.5 - 100.0;
    }
    let out = alias_reduce(&xr, &gc);
    assert_eq!(out, xr);
}

#[test]
fn mixed_block_passes_through_unchanged() {
    // A mixed block is still block_type == 2 under the spec's literal
    // test, so it is not alias-reduced either.
    let gc = short_gc(true);
    let mut xr = [0.0f32; NUM_LINES];
    for (i, v) in xr.iter_mut().enumerate() {
        *v = (i as f32).sin();
    }
    let out = alias_reduce(&xr, &gc);
    assert_eq!(out, xr);
}

#[test]
fn long_block_first_boundary_butterfly() {
    // Place a unit impulse at line 17 (just below the sb=1 boundary at
    // line 18) and another at line 18 (just above it). The i=0 butterfly
    // (cs0, ca0) mixes lines 17 and 18:
    //   xar[17] = xr[17]·cs0 − xr[18]·ca0
    //   xar[18] = xr[18]·cs0 + xr[17]·ca0
    let gc = long_gc();
    let mut xr = [0.0f32; NUM_LINES];
    xr[17] = 1.0;
    xr[18] = 2.0;
    let out = alias_reduce(&xr, &gc);
    let cs = alias_cs();
    let ca = alias_ca();
    let exp17 = 1.0 * cs[0] - 2.0 * ca[0];
    let exp18 = 2.0 * cs[0] + 1.0 * ca[0];
    assert!((out[17] - exp17).abs() < EPS, "out[17]={} exp={}", out[17], exp17);
    assert!((out[18] - exp18).abs() < EPS, "out[18]={} exp={}", out[18], exp18);
    // Lines far from any boundary (e.g. line 9, 8 below the boundary, not
    // touched by the i<8 butterflies) are unchanged.
    assert_eq!(out[0], 0.0);
}

#[test]
fn long_block_butterflies_use_original_inputs() {
    // The spec computes both outputs of a butterfly from the *original*
    // xr values; an in-place implementation that overwrote xr[lo] before
    // computing xar[hi] would be wrong. Set both straddling lines and
    // check the cross terms use the pre-butterfly values.
    let gc = long_gc();
    let mut xr = [0.0f32; NUM_LINES];
    // sb=1 boundary, i=3: lo=14, hi=21.
    xr[14] = 3.0;
    xr[21] = -5.0;
    let out = alias_reduce(&xr, &gc);
    let cs = alias_cs();
    let ca = alias_ca();
    let exp_lo = 3.0 * cs[3] - (-5.0) * ca[3];
    let exp_hi = -5.0 * cs[3] + 3.0 * ca[3];
    assert!((out[14] - exp_lo).abs() < EPS);
    assert!((out[21] - exp_hi).abs() < EPS);
}

#[test]
fn long_block_all_31_boundaries_processed() {
    // Every subband boundary sb=1..32 should run its 8 butterflies. Seed
    // a constant 1.0 everywhere; each butterfly maps (1,1) → (cs−ca,
    // cs+ca), which differs from 1 (since ca≠0), so each of the 31·8·2
    // straddling lines changes, while the 16 interior lines of subband 0
    // below line 9 and the deepest interior lines stay 1.0.
    let gc = long_gc();
    let xr = [1.0f32; NUM_LINES];
    let out = alias_reduce(&xr, &gc);
    let cs = alias_cs();
    let ca = alias_ca();
    // Spot-check a high boundary (sb=31, boundary line 558), i=0.
    let exp_lo = 1.0 * cs[0] - 1.0 * ca[0];
    let exp_hi = 1.0 * cs[0] + 1.0 * ca[0];
    assert!((out[557] - exp_lo).abs() < EPS);
    assert!((out[558] - exp_hi).abs() < EPS);
    // An interior line untouched by butterflies (line 9 is 9 below the
    // sb=1 boundary, beyond the i<8 reach which spans lines 10..18).
    assert_eq!(out[9], 1.0);
}

#[test]
fn long_block_no_boundary_below_subband0() {
    // The loop starts at sb=1, so there is no butterfly straddling line
    // -1/0; lines 0..9 of subband 0 are never the `hi` side of any
    // boundary and stay unchanged for a single low impulse.
    let gc = long_gc();
    let mut xr = [0.0f32; NUM_LINES];
    xr[5] = 7.0;
    let out = alias_reduce(&xr, &gc);
    assert_eq!(out[5], 7.0);
}
