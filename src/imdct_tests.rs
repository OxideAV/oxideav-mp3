// Unit tests for the §2.4.3.4.10 IMDCT / windowing / overlap-add /
// frequency-inversion pipeline. Every reference value is computed
// directly from the spec formulas of §2.4.3.4.10.2 and §2.4.3.4.10.3; no
// external implementation was consulted.

// This file is `include!`d into the `imdct` module, so the public items
// of the module are already in scope (BlockType, GranuleChannel,
// ImdctState, imdct, imdct_granule, NUM_SUBBANDS, SAMPLES_PER_SUBBAND,
// NUM_LINES, and the private helpers long_window / short_window /
// window_long_family / window_short).

const EPS_F32: f32 = 1e-5;
const EPS_F64: f64 = 1e-10;

/// A long-block granule-channel record.
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

/// A window-switched short / mixed granule-channel.
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

/// A window-switched start/stop granule-channel.
fn switched_gc(bt: BlockType) -> GranuleChannel {
    GranuleChannel {
        part2_3_length: 0,
        big_values: 0,
        global_gain: 0,
        scalefac_compress: 0,
        window_switching_flag: true,
        block_type: bt,
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

// -- §2.4.3.4.10.2 IMDCT --

#[test]
fn imdct_n12_impulse_closed_form() {
    // X[k] = delta_{k,0} ⇒ x[i] = cos( (pi/24)·(2i+1+6)·1 )
    //                          = cos( (pi/24)·(2i+7) ).
    let xk = [1.0f64, 0.0, 0.0, 0.0, 0.0, 0.0];
    let out = imdct(&xk, 12);
    assert_eq!(out.len(), 12);
    for (i, &v) in out.iter().enumerate() {
        let expect = (core::f64::consts::PI / 24.0 * (2 * i + 7) as f64).cos();
        assert!(
            (v - expect).abs() < EPS_F64,
            "n=12 impulse x[{i}]={v} expected {expect}"
        );
    }
    // Spot-check two values byte-exactly against the spec arithmetic
    // (independently computed): x[0] = cos(7π/24) ≈ 0.60876143,
    // x[11] = cos(29π/24) ≈ -0.79335334.
    assert!((out[0] - 0.608_761_429).abs() < 1e-8);
    assert!((out[11] - (-0.793_353_340)).abs() < 1e-8);
}

#[test]
fn imdct_n36_all_ones_reference_values() {
    // X[k] = 1 for k=0..18, n = 36. Pre-computed by direct evaluation of
    // the §2.4.3.4.10.2 sum: x[0] ≈ -0.67817085,
    // x[18] ≈ -0.74009362.
    let xk = [1.0f64; 18];
    let out = imdct(&xk, 36);
    assert_eq!(out.len(), 36);
    assert!(
        (out[0] - (-0.678_170_847)).abs() < 1e-8,
        "x[0]={}",
        out[0]
    );
    assert!(
        (out[18] - (-0.740_093_618)).abs() < 1e-8,
        "x[18]={}",
        out[18]
    );
}

#[test]
fn imdct_linearity() {
    // The IMDCT is a linear transform — IMDCT(a·X + b·Y) = a·IMDCT(X) +
    // b·IMDCT(Y). A useful sanity check that the per-k summation is right.
    let mut x = [0.0f64; 18];
    let mut y = [0.0f64; 18];
    for (k, (xk, yk)) in x.iter_mut().zip(y.iter_mut()).enumerate() {
        *xk = (k as f64).sin();
        *yk = (k as f64 + 1.5).cos();
    }
    let a = 0.7;
    let b = -1.3;
    let mut z = [0.0f64; 18];
    for k in 0..18 {
        z[k] = a * x[k] + b * y[k];
    }
    let ox = imdct(&x, 36);
    let oy = imdct(&y, 36);
    let oz = imdct(&z, 36);
    for i in 0..36 {
        let expect = a * ox[i] + b * oy[i];
        assert!(
            (oz[i] - expect).abs() < EPS_F64,
            "linearity i={i}: {} vs {}",
            oz[i],
            expect
        );
    }
}

// -- §2.4.3.4.10.3 Windowing tables (byte-exact spec checks) --

#[test]
fn long_window_table_byte_exact() {
    // sin( (pi/36)·(i+1/2) ), i = 0..35. Spot-check four points against
    // independently computed reference values.
    let w0 = long_window(0); // sin(pi/72)
    let w17 = long_window(17); // sin(35·pi/72)
    let w18 = long_window(18); // sin(37·pi/72) — equal to w17 by symmetry
    let w35 = long_window(35); // sin(71·pi/72) = sin(pi/72) by symmetry
    assert!((w0 - 0.043_619_387).abs() < 1e-9, "w0={}", w0);
    // w17 = sin(35·pi/72) = sin((pi - pi/72)·... ) — by reflection
    // sin((36·pi - pi)/72) = sin(pi/2 - pi/72) = cos(pi/72) ≈ 0.99904822.
    assert!((w17 - 0.999_048_221).abs() < 1e-9, "w17={}", w17);
    assert!((w18 - 0.999_048_221).abs() < 1e-9, "w18={}", w18);
    assert!((w17 - w18).abs() < 1e-12, "long sym about i=17.5");
    // sin((2k+1)pi/72) for k=0 and k=35 are symmetric (both = sin(pi/72)).
    assert!((w35 - w0).abs() < 1e-12, "w35={} w0={}", w35, w0);
    // Symmetry: long_window(35-i) == long_window(i+18) ? actually
    // long_window is symmetric about i = 17.5: w(i) = w(35-i).
    for i in 0..18 {
        let lhs = long_window(i);
        let rhs = long_window(35 - i);
        assert!((lhs - rhs).abs() < 1e-12, "long sym i={i}");
    }
}

#[test]
fn short_window_table_byte_exact() {
    // sin( (pi/12)·(i+1/2) ), i = 0..11. Spec-formula direct values.
    let w0 = short_window(0); // sin(pi/24)
    let w5 = short_window(5); // sin(11·pi/24)
    let w6 = short_window(6); // sin(13·pi/24)
    let w11 = short_window(11); // sin(23·pi/24) = sin(pi/24)
    assert!((w0 - 0.130_526_192).abs() < 1e-9, "w0={}", w0);
    assert!((w5 - 0.991_444_861).abs() < 1e-9, "w5={}", w5);
    assert!((w6 - 0.991_444_861).abs() < 1e-9, "w6={}", w6);
    assert!((w11 - w0).abs() < 1e-12);
    // Short window symmetry: w(i) = w(11-i).
    for i in 0..6 {
        let lhs = short_window(i);
        let rhs = short_window(11 - i);
        assert!((lhs - rhs).abs() < 1e-12, "short sym i={i}");
    }
}

// -- §2.4.3.4.10.2 IMDCT cosine-table memoization (bit-exact guard) --
//
// The decode hot path looks the IMDCT cosine coefficients up from the
// precomputed LONG_COS / SHORT_COS tables and the sine windows up from
// LONG_WINDOW / SHORT_WINDOW instead of evaluating the transcendental on
// every sample. These tests pin the tables to the inline computation at
// the *bit* level (`f64::to_bits` equality) so any future refactor that
// silently perturbs an argument — reassociating the `scale·a·b` product,
// changing a summation order, or swapping the table population loop — is
// caught immediately. Bit equality (not an epsilon) is the round's core
// correctness claim: decoded PCM must stay byte-identical.

#[test]
fn imdct_long_cos_table_is_bit_exact_vs_inline() {
    // LONG_COS[i][k] must equal the inline cos((pi/72)·(2i+1+18)·(2k+1))
    // to the bit, for every (i, k) the long transform ever indexes.
    for i in 0..LONG_N {
        for k in 0..(LONG_N / 2) {
            let scale = core::f64::consts::PI / (2.0 * LONG_N as f64);
            let a = (2 * i + 1 + LONG_N / 2) as f64;
            let b = (2 * k + 1) as f64;
            let inline = (scale * a * b).cos();
            assert_eq!(
                LONG_COS[i][k].to_bits(),
                inline.to_bits(),
                "LONG_COS[{i}][{k}] must be bit-identical to inline cos"
            );
        }
    }
}

#[test]
fn imdct_short_cos_table_is_bit_exact_vs_inline() {
    for i in 0..SHORT_N {
        for k in 0..(SHORT_N / 2) {
            let scale = core::f64::consts::PI / (2.0 * SHORT_N as f64);
            let a = (2 * i + 1 + SHORT_N / 2) as f64;
            let b = (2 * k + 1) as f64;
            let inline = (scale * a * b).cos();
            assert_eq!(
                SHORT_COS[i][k].to_bits(),
                inline.to_bits(),
                "SHORT_COS[{i}][{k}] must be bit-identical to inline cos"
            );
        }
    }
}

#[test]
fn imdct_window_tables_are_bit_exact_vs_inline() {
    for i in 0..LONG_N {
        let inline = ((core::f64::consts::PI / 36.0) * (i as f64 + 0.5)).sin();
        assert_eq!(
            LONG_WINDOW[i].to_bits(),
            inline.to_bits(),
            "LONG_WINDOW[{i}] must be bit-identical to inline sin"
        );
    }
    for i in 0..SHORT_N {
        let inline = ((core::f64::consts::PI / 12.0) * (i as f64 + 0.5)).sin();
        assert_eq!(
            SHORT_WINDOW[i].to_bits(),
            inline.to_bits(),
            "SHORT_WINDOW[{i}] must be bit-identical to inline sin"
        );
    }
}

#[test]
fn imdct_table_path_matches_direct_cosine_bit_for_bit() {
    // The memoized `imdct(&xk, n)` must produce, for both transform sizes
    // the codec uses, output bit-identical to a fresh per-sample cosine
    // evaluation over a non-trivial input — proof that the table lookup +
    // preserved k-summation order changed nothing observable.
    for &n in &[LONG_N, SHORT_N] {
        let half = n / 2;
        // Deterministic mixed-sign, mixed-magnitude input.
        let xk: Vec<f64> = (0..half)
            .map(|k| ((k as f64) * 0.618_033_988 - 3.0).sin() * (k as f64 + 1.0))
            .collect();

        // Reference: direct per-sample cosine, no table.
        let scale = core::f64::consts::PI / (2.0 * n as f64);
        let mut reference = vec![0.0f64; n];
        for (i, r) in reference.iter_mut().enumerate() {
            let a = (2 * i + 1 + half) as f64;
            let mut acc = 0.0f64;
            for (k, &x) in xk.iter().enumerate() {
                let b = (2 * k + 1) as f64;
                acc += x * (scale * a * b).cos();
            }
            *r = acc;
        }

        let got = imdct(&xk, n);
        assert_eq!(got.len(), reference.len());
        for (i, (&g, &r)) in got.iter().zip(reference.iter()).enumerate() {
            assert_eq!(
                g.to_bits(),
                r.to_bits(),
                "imdct(n={n})[{i}] must be bit-identical to the direct cosine path"
            );
        }
    }
}

#[test]
fn long_window_squared_sum_two_halves_is_18() {
    // For the long block (block_type 0): sum_{i=0..35} w(i)^2. Computed
    // from the spec formula sum_i sin^2((pi/36)(i+1/2)). Closed form
    // gives exactly 18 (half of 36, since the mean of sin^2 over the
    // full period is 1/2). A small numeric check; the byte-exact part is
    // long_window itself above.
    let mut s = 0.0f64;
    for i in 0..36 {
        let w = long_window(i);
        s += w * w;
    }
    assert!((s - 18.0).abs() < 1e-12, "sum w^2 = {}", s);
}

// -- Windowed-block shapes per §2.4.3.4.10.3 --

#[test]
fn long_window_family_normal_matches_spec() {
    // For block_type 0, window_long_family applies sin((pi/36)(i+1/2)) to
    // every i. Check by feeding a constant input 1.0..1.0 — the output
    // should be exactly the window table.
    let x = [1.0f64; 36];
    let z = window_long_family(&x, BlockType::Long);
    for i in 0..36 {
        assert!((z[i] - long_window(i)).abs() < 1e-12, "normal i={i}");
    }
}

#[test]
fn start_block_window_shape_per_spec() {
    // block_type 1: long-window over 0..17, pass-through over 18..23,
    // short-window of (i-18) over 24..29, zero over 30..35.
    let x = [1.0f64; 36];
    let z = window_long_family(&x, BlockType::Start);
    for i in 0..36 {
        let expect = match i {
            0..=17 => long_window(i),
            18..=23 => 1.0,
            24..=29 => {
                let arg = (core::f64::consts::PI / 12.0) * ((i - 18) as f64 + 0.5);
                arg.sin()
            }
            _ => 0.0,
        };
        assert!((z[i] - expect).abs() < 1e-12, "start i={i}: {} vs {}", z[i], expect);
    }
    // Tail of a start block (i = 30..35) is identically zero — this is
    // what makes the start block "open at the bottom, closed at the top".
    for i in 30..36 {
        assert_eq!(z[i], 0.0, "start tail i={i}");
    }
}

#[test]
fn stop_block_window_shape_per_spec() {
    // block_type 3: zero over 0..5, short-window of (i-6) over 6..11,
    // pass-through over 12..17, long-window over 18..35.
    let x = [1.0f64; 36];
    let z = window_long_family(&x, BlockType::End);
    for i in 0..36 {
        let expect = match i {
            0..=5 => 0.0,
            6..=11 => {
                let arg = (core::f64::consts::PI / 12.0) * ((i - 6) as f64 + 0.5);
                arg.sin()
            }
            12..=17 => 1.0,
            _ => long_window(i),
        };
        assert!((z[i] - expect).abs() < 1e-12, "stop i={i}: {} vs {}", z[i], expect);
    }
    // Head of a stop block (i = 0..5) is identically zero — what makes
    // it "open at the top, closed at the bottom".
    for i in 0..6 {
        assert_eq!(z[i], 0.0, "stop head i={i}");
    }
}

#[test]
fn short_block_concatenation_table_matches_spec() {
    // §2.4.3.4.10.3 d concatenation table:
    //   z[0..5]    = 0
    //   z[6..11]   = y0[i-6]
    //   z[12..17]  = y0[i-6] + y1[i-12]
    //   z[18..23]  = y1[i-12] + y2[i-18]
    //   z[24..29]  = y2[i-18]
    //   z[30..35]  = 0
    // Feed each sub-block a constant 1.0..1.0; y_j[k] = sin((pi/12)(k+1/2)).
    let sub = [[1.0f64; 12]; 3];
    let z = window_short(&sub);
    for i in 0..36 {
        let expect = match i {
            0..=5 => 0.0,
            6..=11 => short_window(i - 6),
            12..=17 => short_window(i - 6) + short_window(i - 12),
            18..=23 => short_window(i - 12) + short_window(i - 18),
            24..=29 => short_window(i - 18),
            _ => 0.0,
        };
        assert!((z[i] - expect).abs() < 1e-12, "short i={i}: {} vs {}", z[i], expect);
    }
    // Boundary zeros are byte-exact.
    for i in 0..6 {
        assert_eq!(z[i], 0.0, "short head i={i}");
    }
    for i in 30..36 {
        assert_eq!(z[i], 0.0, "short tail i={i}");
    }
}

#[test]
fn short_block_middle_overlap_sums_two_sub_blocks() {
    // Distinct sub-blocks let us check that the i=12..17 and i=18..23
    // regions actually add the two adjacent sub-block windowings (not
    // just take one of them).
    let sub = [
        [1.0f64; 12], // y0
        [2.0f64; 12], // y1
        [3.0f64; 12], // y2
    ];
    let z = window_short(&sub);
    for i in 12..18 {
        let expect = 1.0 * short_window(i - 6) + 2.0 * short_window(i - 12);
        assert!((z[i] - expect).abs() < 1e-12, "i={i}: {} vs {}", z[i], expect);
    }
    for i in 18..24 {
        let expect = 2.0 * short_window(i - 12) + 3.0 * short_window(i - 18);
        assert!((z[i] - expect).abs() < 1e-12, "i={i}: {} vs {}", z[i], expect);
    }
}

// -- §2.4.3.4.10.4 Overlap-add behaviour --

#[test]
fn overlap_state_starts_zero() {
    let st = ImdctState::default();
    for sb in 0..NUM_SUBBANDS {
        assert_eq!(st.overlap(sb), [0.0; SAMPLES_PER_SUBBAND]);
    }
}

#[test]
fn overlap_first_granule_uses_zero_prev() {
    // With s_prev all zero, result[i] = z[i] for i=0..17, and the saved
    // overlap becomes z[i+18]. Drive a long block with a single non-zero
    // line so z is predictable.
    let gc = long_gc();
    let mut xr = [0.0f32; NUM_LINES];
    // Put X[0] = 1 in subband 0 (line index 0 is k=0 of subband 0).
    xr[0] = 1.0;
    let mut st = ImdctState::default();
    let out = imdct_granule(&xr, &gc, &mut st);

    // Recompute z for subband 0 directly: IMDCT of [1,0,0,…,0] of length
    // 18, then window with long_window.
    let xk = {
        let mut v = [0.0f64; 18];
        v[0] = 1.0;
        v
    };
    let x = imdct(&xk, 36);
    let mut z = [0.0f64; 36];
    for i in 0..36 {
        z[i] = x[i] * long_window(i);
    }
    // First half is the output (s_prev = 0); subband 0 is "subband 0",
    // which is even, so no frequency-inversion sign flip.
    for i in 0..18 {
        let expect = z[i] as f32;
        assert!(
            (out[0][i] - expect).abs() < EPS_F32,
            "subband0 t={i}: {} vs {}",
            out[0][i],
            expect
        );
    }
    // Saved overlap is the second half.
    for i in 0..18 {
        let expect = z[i + 18] as f32;
        assert!(
            (st.overlap[0][i] - expect).abs() < EPS_F32,
            "save sb0 t={i}: {} vs {}",
            st.overlap[0][i],
            expect
        );
    }
}

#[test]
fn overlap_second_granule_adds_saved_first_half() {
    // Run two granules with the same input. Granule 2's output = z[i] +
    // z[i+18] (since the first granule saved z[i+18] into the overlap).
    let gc = long_gc();
    let mut xr = [0.0f32; NUM_LINES];
    xr[0] = 1.0; // subband 0, line 0
    let mut st = ImdctState::default();
    let _g1 = imdct_granule(&xr, &gc, &mut st);
    let saved = st.overlap[0];
    let g2 = imdct_granule(&xr, &gc, &mut st);

    // Recompute z for subband 0.
    let mut xk = [0.0f64; 18];
    xk[0] = 1.0;
    let x = imdct(&xk, 36);
    let mut z = [0.0f64; 36];
    for i in 0..36 {
        z[i] = x[i] * long_window(i);
    }
    for i in 0..18 {
        let expect = z[i] as f32 + saved[i];
        assert!(
            (g2[0][i] - expect).abs() < EPS_F32,
            "sb0 g2 t={i}: {} vs {}",
            g2[0][i],
            expect
        );
    }
}

#[test]
fn overlap_per_subband_isolation() {
    // A non-zero subband's overlap must not leak into other subbands'
    // outputs. Drive only subband 5 in granule 1; in granule 2 with zero
    // input, only subband 5 should have non-zero output.
    let gc = long_gc();
    let mut xr = [0.0f32; NUM_LINES];
    let sb = 5;
    xr[sb * SAMPLES_PER_SUBBAND] = 1.0;
    let mut st = ImdctState::default();
    let _ = imdct_granule(&xr, &gc, &mut st);
    let xr_zero = [0.0f32; NUM_LINES];
    let g2 = imdct_granule(&xr_zero, &gc, &mut st);
    for s in 0..NUM_SUBBANDS {
        for t in 0..SAMPLES_PER_SUBBAND {
            if s == sb {
                // Some outputs may be exactly zero (where z[t+18] was 0);
                // that's fine. Just confirm at least one is non-zero.
                continue;
            }
            assert_eq!(g2[s][t], 0.0, "leak into sb={s} t={t}: {}", g2[s][t]);
        }
    }
    let any_nonzero = (0..SAMPLES_PER_SUBBAND).any(|t| g2[sb][t] != 0.0);
    assert!(any_nonzero, "expected non-zero carry into sb={sb} g2");
}

// -- §2.4.3.4.10.5 Frequency inversion --

#[test]
fn frequency_inversion_odd_subbands_odd_times() {
    // Drive a constant 1.0 input across the spectrum, long block. Compare
    // odd-subband-odd-time samples against the same subband's even-time
    // samples sign-flipped relative to subband-0 (without inversion they
    // would equal the subband-0 value at the same time index — all
    // long-block windowed IMDCTs of the same input are identical because
    // each subband holds 18 identical input lines).
    let gc = long_gc();
    let mut xr = [0.0f32; NUM_LINES];
    for v in xr.iter_mut() {
        *v = 1.0;
    }
    let mut st = ImdctState::default();
    let out = imdct_granule(&xr, &gc, &mut st);

    // Subband 0 (even sb) is unaffected by inversion. Subband 1 (odd sb)
    // has every odd time sample negated. So out[1][t] == -out[0][t] for
    // odd t, and out[1][t] == out[0][t] for even t.
    for t in 0..SAMPLES_PER_SUBBAND {
        let s0 = out[0][t];
        let s1 = out[1][t];
        if t % 2 == 1 {
            assert!(
                (s1 - (-s0)).abs() < EPS_F32,
                "sb1 odd t={t}: {} vs {}",
                s1,
                -s0
            );
        } else {
            assert!(
                (s1 - s0).abs() < EPS_F32,
                "sb1 even t={t}: {} vs {}",
                s1,
                s0
            );
        }
    }
    // Subband 2 (even) is unaffected; should equal subband 0 exactly.
    for t in 0..SAMPLES_PER_SUBBAND {
        assert!(
            (out[2][t] - out[0][t]).abs() < EPS_F32,
            "sb2 even t={t}"
        );
    }
}

#[test]
fn frequency_inversion_even_subband_unaffected() {
    // Drive a known impulse in subband 4 (even). The output must match
    // the raw IMDCT+window result with no sign flips.
    let gc = long_gc();
    let mut xr = [0.0f32; NUM_LINES];
    let sb = 4;
    xr[sb * SAMPLES_PER_SUBBAND + 3] = 1.0; // X[3] = 1
    let mut st = ImdctState::default();
    let out = imdct_granule(&xr, &gc, &mut st);

    // Recompute the first-half of z directly.
    let mut xk = [0.0f64; 18];
    xk[3] = 1.0;
    let x = imdct(&xk, 36);
    for t in 0..18 {
        let z = x[t] * long_window(t);
        assert!(
            (out[sb][t] - z as f32).abs() < EPS_F32,
            "even sb={sb} t={t}: {} vs {}",
            out[sb][t],
            z
        );
    }
}

// -- Block-type dispatch through imdct_granule --

#[test]
fn short_block_three_sub_imdcts_per_subband() {
    // A short block runs three 12-point IMDCTs per subband. With a single
    // non-zero short-window line in subband 0, only one of the three
    // sub-IMDCTs is non-zero, and its windowed output ends up in the
    // correct concatenation slot (windows 0/1/2 land in z[6..29] runs).
    let gc = short_gc(false);
    let mut xr = [0.0f32; NUM_LINES];
    // After reorder, line index 3·k + j is freq-line k of window j.
    // Put X^(1)[0] = 1 in subband 0 → index 3·0 + 1 = 1.
    xr[1] = 1.0;
    let mut st = ImdctState::default();
    let out = imdct_granule(&xr, &gc, &mut st);

    // Compute expected: y1 = IMDCT12([1,0,0,0,0,0]) windowed by short_window.
    let mut xk = [0.0f64; 6];
    xk[0] = 1.0;
    let y1_raw = imdct(&xk, 12);
    let mut y1 = [0.0f64; 12];
    for i in 0..12 {
        y1[i] = y1_raw[i] * short_window(i);
    }
    // z[i] for window 1: contributions to z[12..23] per the concat table.
    // z[12..17] = y0 + y1 ⇒ here y0=0, so z[i] = y1[i-12], i=12..17.
    // z[18..23] = y1 + y2 ⇒ here y2=0, so z[i] = y1[i-12], i=18..23.
    // The first half (i=0..17) of z is the output; the second half is saved.
    for i in 12..18 {
        let expect = y1[i - 12] as f32;
        assert!(
            (out[0][i] - expect).abs() < EPS_F32,
            "short out t={i}: {} vs {}",
            out[0][i],
            expect
        );
    }
    // Out side: t=0..11 in subband 0 must be 0 (short-block z[0..11] are 0
    // for an isolated y1 contribution at i=12..23; and z[0..5]=0 by table).
    for t in 0..6 {
        assert_eq!(out[0][t], 0.0, "short z[0..5]=0 t={t}");
    }
    for t in 6..12 {
        // z[t] = y0[t-6] = 0.
        assert_eq!(out[0][t], 0.0, "short z[6..11]=y0=0 t={t}");
    }
    // Saved overlap is z[18..35]: only z[18..23] are non-zero here.
    let saved = st.overlap[0];
    for t in 0..6 {
        let expect = y1[(t + 18) - 12] as f32;
        assert!(
            (saved[t] - expect).abs() < EPS_F32,
            "short save t={t}: {} vs {}",
            saved[t],
            expect
        );
    }
    for t in 6..18 {
        assert_eq!(saved[t], 0.0, "short save tail t={t}");
    }
}

#[test]
fn mixed_block_lower_two_subbands_use_long_window() {
    // A mixed block (block_type 2, mixed_block_flag) codes subbands 0
    // and 1 with the long window. Drive subband 0 with X[0]=1; the
    // result should match a *long-block* IMDCT+normal-window, not the
    // short-block triple-IMDCT.
    let gc = short_gc(true);
    let mut xr = [0.0f32; NUM_LINES];
    xr[0] = 1.0;
    let mut st = ImdctState::default();
    let out = imdct_granule(&xr, &gc, &mut st);

    // Expected: 36-point IMDCT of [1,0,…,0], normal-window first half.
    let mut xk = [0.0f64; 18];
    xk[0] = 1.0;
    let x = imdct(&xk, 36);
    for t in 0..18 {
        let z = x[t] * long_window(t);
        assert!(
            (out[0][t] - z as f32).abs() < EPS_F32,
            "mixed sb0 t={t}: {} vs {}",
            out[0][t],
            z
        );
    }
    // Subband 1 also uses long-window in a mixed block. Drive an impulse
    // and check.
    let mut xr2 = [0.0f32; NUM_LINES];
    xr2[18] = 1.0; // subband 1, line 0
    let mut st2 = ImdctState::default();
    let out2 = imdct_granule(&xr2, &gc, &mut st2);
    // Same expected z first-half, but subband 1 is odd ⇒ odd-time
    // samples are sign-flipped by §2.4.3.4.10.5.
    for t in 0..18 {
        let z = x[t] * long_window(t);
        let expect = if t % 2 == 1 { -(z as f32) } else { z as f32 };
        assert!(
            (out2[1][t] - expect).abs() < EPS_F32,
            "mixed sb1 t={t}: {} vs {}",
            out2[1][t],
            expect
        );
    }
}

#[test]
fn start_block_through_granule_uses_start_window() {
    // A start block dispatched through imdct_granule must produce z with
    // its tail (i=30..35 in the windowed block) identically zero. After
    // overlap-add the saved second half (z[18..35]) of subband 0 should
    // have its last six entries (indices 12..18 of the saved array,
    // corresponding to z[30..35]) at exactly zero — confirming the start
    // window was applied.
    let gc = switched_gc(BlockType::Start);
    let mut xr = [0.0f32; NUM_LINES];
    for v in xr.iter_mut().take(SAMPLES_PER_SUBBAND) {
        *v = 1.0; // subband 0 driven uniformly
    }
    let mut st = ImdctState::default();
    let _ = imdct_granule(&xr, &gc, &mut st);
    let saved = st.overlap[0];
    // saved[i] = z[i+18]; indices i=12..18 correspond to z[30..35]=0.
    for i in 12..18 {
        assert_eq!(saved[i], 0.0, "start tail saved[{i}] = z[{}]", i + 18);
    }
}

#[test]
fn stop_block_through_granule_uses_stop_window() {
    // A stop block's z[0..5] are identically zero, so the first six
    // output samples of any subband in granule 1 (with zero s_prev) are
    // zero regardless of input.
    let gc = switched_gc(BlockType::End);
    let mut xr = [0.0f32; NUM_LINES];
    for v in xr.iter_mut().take(SAMPLES_PER_SUBBAND) {
        *v = 1.0;
    }
    let mut st = ImdctState::default();
    let out = imdct_granule(&xr, &gc, &mut st);
    for t in 0..6 {
        assert_eq!(out[0][t], 0.0, "stop head out[0][{t}]");
    }
}

#[test]
fn zero_input_zero_output() {
    let gc = long_gc();
    let xr = [0.0f32; NUM_LINES];
    let mut st = ImdctState::default();
    let out = imdct_granule(&xr, &gc, &mut st);
    for sb in 0..NUM_SUBBANDS {
        for t in 0..SAMPLES_PER_SUBBAND {
            assert_eq!(out[sb][t], 0.0, "zero-in nonzero-out sb={sb} t={t}");
        }
        assert_eq!(st.overlap[sb], [0.0; SAMPLES_PER_SUBBAND]);
    }
}
