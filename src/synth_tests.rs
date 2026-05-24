// Unit tests for the §2.4.3.2 / Figure A.2 polyphase synthesis subband
// filterbank. Every reference value is derived directly from the spec
// formulas of §2.4.3.2.2 (the `N[i,k]` matrixing coefficient formula),
// Figure A.2 (the per-step pseudo code), and Annex B Table B.3 (the
// `D[i]` window coefficients). No external implementation was consulted.

// This file is `include!`d into the `synth` module, so the public items
// (SynthState, synth_row, synth_granule, n_coefficient, D_TABLE,
// NUM_SUBBANDS, SAMPLES_PER_SUBBAND, PCM_PER_GRANULE, V_LEN, U_LEN) are
// already in scope.

const EPS_F64: f64 = 1e-12;
const EPS_F32: f32 = 1e-5;

// ----- D[] table cross-checks -----

#[test]
fn d_table_length_is_512() {
    assert_eq!(D_TABLE.len(), U_LEN);
    assert_eq!(U_LEN, 512);
}

#[test]
fn d_table_boundary_values_match_spec() {
    // Spot-check the table boundaries directly from the rendered Table
    // B.3 pages (`docs/audio/mp3/annex-b-renders/Table-B.3-coefficients-
    // Di-p5{6,7,8}.png`):
    //   D[0]    =  0.000000000     (origin)
    //   D[1]    = -0.000015259     (smallest non-zero magnitude)
    //   D[255]  = -1.144287109     (negative peak, last before flip)
    //   D[256]  =  1.144989014     (positive peak, the global maximum)
    //   D[257]  =  1.144287109     (mirror of D[255], opposite sign)
    //   D[511]  =  0.000015259     (tail)
    assert_eq!(D_TABLE[0], 0.000000000);
    assert_eq!(D_TABLE[1], -0.000015259);
    assert_eq!(D_TABLE[255], -1.144287109);
    assert_eq!(D_TABLE[256], 1.144989014);
    assert_eq!(D_TABLE[257], 1.144287109);
    assert_eq!(D_TABLE[511], 0.000015259);
}

#[test]
fn d_table_d256_is_global_maximum() {
    // The spec design has D[256] as the unique global maximum (1.144989014).
    let mut max = f64::NEG_INFINITY;
    let mut argmax = 0usize;
    for (i, &v) in D_TABLE.iter().enumerate() {
        if v > max {
            max = v;
            argmax = i;
        }
    }
    assert_eq!(argmax, 256);
    assert_eq!(max, 1.144989014);
}

#[test]
fn d_table_d255_is_global_minimum() {
    // The spec design has D[255] = D[257] negated, with D[255] being the
    // unique global minimum at -1.144287109.
    let mut min = f64::INFINITY;
    let mut argmin = 0usize;
    for (i, &v) in D_TABLE.iter().enumerate() {
        if v < min {
            min = v;
            argmin = i;
        }
    }
    assert_eq!(argmin, 255);
    assert_eq!(min, -1.144287109);
}

#[test]
fn d_table_mirror_symmetry_pairs_match_table_values() {
    // The Table B.3 listing shows several |D[256-i]| ~= |D[256+i]| pairs
    // (the spec's prototype filter is mirror-symmetric around index 256
    // with sign flips at certain stripes). Verify a handful of these
    // directly against the printed numbers:
    //   D[64]   =  0.003250122   D[448]  =  0.003250122
    //   D[128]  =  0.031082153   D[384]  =  0.031082153
    //   D[192]  =  0.100311279   D[320]  =  0.100311279
    assert_eq!(D_TABLE[64], D_TABLE[448]);
    assert_eq!(D_TABLE[128], D_TABLE[384]);
    assert_eq!(D_TABLE[192], D_TABLE[320]);
}

// ----- N[i,k] formula cross-checks -----

#[test]
fn n_coefficient_matches_spec_formula() {
    // Spot-check N[i,k] = cos((16+i)·(2k+1)·π/64) at boundary values.
    // i = 0, k = 0: cos(16·1·π/64) = cos(π/4) = √2/2.
    let nv = n_coefficient(0, 0);
    assert!((nv - core::f64::consts::FRAC_1_SQRT_2).abs() < EPS_F64);

    // i = 0, k = 31: cos(16·63·π/64) = cos(63π/4) = cos(63π/4 - 16π) =
    //   cos(63π/4 - 64π/4) = cos(-π/4) = √2/2.
    let nv2 = n_coefficient(0, 31);
    assert!((nv2 - core::f64::consts::FRAC_1_SQRT_2).abs() < EPS_F64);

    // i = 48, k = 0: cos((16+48)·π/64) = cos(π) = -1.
    let nv3 = n_coefficient(48, 0);
    assert!((nv3 - (-1.0)).abs() < EPS_F64);

    // i = 16, k = 0: cos((16+16)·π/64) = cos(π/2) = 0.
    let nv4 = n_coefficient(16, 0);
    assert!(nv4.abs() < EPS_F64);
}

// ----- SynthState start state -----

#[test]
fn synth_state_default_is_zero() {
    let s = SynthState::default();
    for i in 0..V_LEN {
        assert_eq!(s.v(i), 0.0);
    }
}

#[test]
fn synth_state_v_out_of_range_returns_zero() {
    let s = SynthState::default();
    assert_eq!(s.v(V_LEN), 0.0);
    assert_eq!(s.v(V_LEN + 1024), 0.0);
}

// ----- Zero input → zero output -----

#[test]
fn zero_input_produces_zero_output() {
    let mut state = SynthState::new();
    let s = [0.0f64; NUM_SUBBANDS];
    for _ in 0..32 {
        let out = synth_row(&s, &mut state);
        for &v in out.iter() {
            assert_eq!(v, 0.0);
        }
    }
    // The shift register also remains all-zero.
    for i in 0..V_LEN {
        assert_eq!(state.v(i), 0.0);
    }
}

// ----- Hand-computed known vector: single subband impulse -----

#[test]
fn impulse_in_subband_0_produces_d_times_n_column() {
    // First-iteration sanity check derived directly from Figure A.2.
    //
    // Start state: V = 0. Input: S[0] = 1, S[1..32] = 0.
    //
    // Step 1 (Shift):   V[64..1024] still zero (the shift moves zeros
    //                   into zeros), V[0..64] is overwritten next.
    // Step 2 (Matrix):  V[i] = sum_k N[i,k]·S[k] = N[i,0]
    //                       = cos((16+i)·π/64)   for i = 0..64.
    // Step 3 (Build U): U[64i+j]    = V[128i+j]
    //                   U[64i+32+j] = V[128i+96+j]
    //                   Since V is non-zero only at indices 0..64:
    //                     i=0: U[j]      = V[j]            for j=0..32
    //                          U[32+j]   = V[96+j] = 0     for j=0..32
    //                     i>=1: all U slots read from V at indices >=128
    //                          and so are zero.
    //                   ⇒ U[0..32] = V[0..32], U[32..512] = 0.
    // Step 4/5:         S_out[j] = sum_{i=0..16} U[j+32i]·D[j+32i]
    //                            = U[j]·D[j]   (only i=0 survives)
    //                            = N[j,0] · D[j]
    //                            = cos((16+j)·π/64) · D[j].
    let mut state = SynthState::new();
    let mut s = [0.0f64; NUM_SUBBANDS];
    s[0] = 1.0;
    let out = synth_row(&s, &mut state);

    for j in 0..NUM_SUBBANDS {
        let expected = n_coefficient(j, 0) * D_TABLE[j];
        assert!(
            (out[j] - expected).abs() < EPS_F64,
            "j={j}: got {} expected {}",
            out[j],
            expected
        );
    }

    // Also verify V[i] = N[i,0] for i = 0..64 after the matrix step.
    for i in 0..64 {
        let expected = n_coefficient(i, 0);
        assert!(
            (state.v(i) - expected).abs() < EPS_F64,
            "V[{i}] = {} expected {}",
            state.v(i),
            expected
        );
    }
    // And V[64..1024] is still zero (the shift moved zeros into zeros
    // and the matrix only wrote 0..64).
    for i in 64..V_LEN {
        assert_eq!(state.v(i), 0.0, "V[{i}] should still be zero");
    }
}

// ----- Hand-computed known vector: single subband impulse in subband k>0 -----

#[test]
fn impulse_in_subband_5_after_first_iteration_matches_formula() {
    let mut state = SynthState::new();
    let mut s = [0.0f64; NUM_SUBBANDS];
    s[5] = 1.0;
    let out = synth_row(&s, &mut state);

    // Same derivation: S_out[j] = N[j,5] · D[j] for j = 0..32.
    for j in 0..NUM_SUBBANDS {
        let expected = n_coefficient(j, 5) * D_TABLE[j];
        assert!(
            (out[j] - expected).abs() < EPS_F64,
            "j={j}: got {} expected {}",
            out[j],
            expected
        );
    }
}

// ----- Linearity -----

#[test]
fn synth_row_is_linear_in_input() {
    // The synthesis filter is linear: synth(a·S1 + b·S2) = a·synth(S1) +
    // b·synth(S2). With a zero start state, applying the filter to two
    // separate impulses then to their sum should match.
    let mut s1 = [0.0f64; NUM_SUBBANDS];
    s1[2] = 1.0;
    let mut s2 = [0.0f64; NUM_SUBBANDS];
    s2[7] = 1.0;
    let a = 0.375;
    let b = -1.25;
    let mut combined = [0.0f64; NUM_SUBBANDS];
    for i in 0..NUM_SUBBANDS {
        combined[i] = a * s1[i] + b * s2[i];
    }

    let mut st1 = SynthState::new();
    let out1 = synth_row(&s1, &mut st1);
    let mut st2 = SynthState::new();
    let out2 = synth_row(&s2, &mut st2);
    let mut stc = SynthState::new();
    let outc = synth_row(&combined, &mut stc);

    for j in 0..NUM_SUBBANDS {
        let expected = a * out1[j] + b * out2[j];
        assert!(
            (outc[j] - expected).abs() < EPS_F64,
            "j={j}: linearity violated, got {} expected {}",
            outc[j],
            expected
        );
    }
}

// ----- Shift register propagates impulse across iterations -----

#[test]
fn shift_register_propagates_v_across_iterations() {
    // After one iteration with S[0]=1: V[0..64] = N[i,0], V[64..]=0.
    // After a second iteration with S = 0: the shift moves V[0..960]
    // up into V[64..1024], so V[64..128] becomes the previous V[0..64].
    let mut state = SynthState::new();
    let mut s = [0.0f64; NUM_SUBBANDS];
    s[0] = 1.0;
    let _ = synth_row(&s, &mut state);

    let snapshot: [f64; 64] = std::array::from_fn(|i| state.v(i));

    let zero = [0.0f64; NUM_SUBBANDS];
    let _ = synth_row(&zero, &mut state);

    // V[64..128] now equals the snapshot (the shift moved V[0..64] up).
    for i in 0..64 {
        assert!(
            (state.v(64 + i) - snapshot[i]).abs() < EPS_F64,
            "V[{}] = {} did not pick up shifted snapshot[{i}] = {}",
            64 + i,
            state.v(64 + i),
            snapshot[i]
        );
    }
    // V[0..64] is the matrix output with S=0, so V[0..64] is all zero.
    for i in 0..64 {
        assert_eq!(state.v(i), 0.0, "V[{i}] should be 0 (matrixing S=0)");
    }
}

// ----- synth_granule wraps 18 synth_row invocations and shapes output -----

#[test]
fn synth_granule_zero_input_yields_zero_pcm() {
    let mut state = SynthState::new();
    let zero_block = [[0.0f32; SAMPLES_PER_SUBBAND]; NUM_SUBBANDS];
    let pcm = synth_granule(&zero_block, &mut state);
    assert_eq!(pcm.len(), PCM_PER_GRANULE);
    assert_eq!(PCM_PER_GRANULE, 576);
    for &v in pcm.iter() {
        assert_eq!(v, 0.0);
    }
}

#[test]
fn synth_granule_first_row_matches_synth_row() {
    // Place an impulse in subband 0, time-row 0; zero everywhere else.
    // Then synth_granule's first 32 outputs (time-row 0) must match
    // synth_row applied to that single row.
    let mut block = [[0.0f32; SAMPLES_PER_SUBBAND]; NUM_SUBBANDS];
    block[0][0] = 1.0;

    let mut state_g = SynthState::new();
    let pcm = synth_granule(&block, &mut state_g);

    let mut state_r = SynthState::new();
    let mut row = [0.0f64; NUM_SUBBANDS];
    row[0] = 1.0;
    let out_row = synth_row(&row, &mut state_r);

    for j in 0..NUM_SUBBANDS {
        let g = pcm[j] as f64;
        let r = out_row[j];
        assert!(
            (g - r).abs() < f64::from(EPS_F32),
            "j={j}: granule[{j}] = {g} vs synth_row[{j}] = {r}",
        );
    }
    // The next 17 rows should be the granule output for input rows of
    // all zeros, but with the shift register holding the propagated V
    // from row 0. These values are non-zero in general but the test
    // verifies the simpler row-0 equivalence.
}

// ----- End-to-end synthetic sine through imdct + synth (no encode) -----

#[test]
fn end_to_end_zero_imdct_to_synth_yields_zero_pcm() {
    // A clean-room end-to-end sanity check: pass an all-zero alias-reduced
    // xr buffer through imdct_granule, then synth_granule, with both
    // states starting at zero. The output must be 576 PCM zeros (the
    // entire pipeline is linear and the input is zero).
    use crate::imdct::{imdct_granule, ImdctState};

    let xr = [0.0f32; crate::imdct::NUM_LINES];
    let gc = crate::side_info::GranuleChannel {
        part2_3_length: 0,
        big_values: 0,
        global_gain: 0,
        scalefac_compress: 0,
        window_switching_flag: false,
        block_type: crate::side_info::BlockType::Long,
        mixed_block_flag: false,
        table_select: [0; 3],
        subblock_gain: [0; 3],
        region0_count: 0,
        region1_count: 0,
        preflag: false,
        scalefac_scale: false,
        count1table_select: false,
    };

    let mut istate = ImdctState::new();
    let subband_time = imdct_granule(&xr, &gc, &mut istate);
    let mut sstate = SynthState::new();
    let pcm = synth_granule(&subband_time, &mut sstate);

    for &v in pcm.iter() {
        assert_eq!(v, 0.0);
    }
}

#[test]
fn end_to_end_synthetic_dc_frequency_in_imdct_produces_finite_pcm() {
    // A clean-room synthetic-frame end-to-end smoke test: build an
    // alias-reduced xr buffer with a single DC frequency line in
    // subband 0 (xr[0] = 1.0, rest 0), run it through imdct_granule and
    // synth_granule from cold start, and verify the resulting 576 PCM
    // samples are all finite (no NaN / no infinity). The point is to
    // exercise the full §2.4.3.4.10 → §2.4.3.2 pipeline on a non-zero
    // input where every numeric step has to actually fire, not just
    // pass zeros through.
    use crate::imdct::{imdct_granule, ImdctState};

    let mut xr = [0.0f32; crate::imdct::NUM_LINES];
    xr[0] = 1.0;
    let gc = crate::side_info::GranuleChannel {
        part2_3_length: 0,
        big_values: 0,
        global_gain: 0,
        scalefac_compress: 0,
        window_switching_flag: false,
        block_type: crate::side_info::BlockType::Long,
        mixed_block_flag: false,
        table_select: [0; 3],
        subblock_gain: [0; 3],
        region0_count: 0,
        region1_count: 0,
        preflag: false,
        scalefac_scale: false,
        count1table_select: false,
    };

    let mut istate = ImdctState::new();
    let subband_time = imdct_granule(&xr, &gc, &mut istate);
    let mut sstate = SynthState::new();
    let pcm = synth_granule(&subband_time, &mut sstate);

    let mut any_nonzero = false;
    for &v in pcm.iter() {
        assert!(v.is_finite(), "non-finite PCM sample {v}");
        if v.abs() > 1e-6 {
            any_nonzero = true;
        }
    }
    // At least some PCM samples must be non-zero — the input was not
    // null and the filter chain has gain at DC in subband 0.
    assert!(any_nonzero, "all PCM samples were zero; pipeline silent");
}

// ----- D[] table sanity: tail values match Table B.3 exactly -----

#[test]
fn d_table_tail_values_match_table_b3() {
    // The last eight entries of Table B.3 are all 0.000015259 (the
    // pattern reflects the prototype filter's near-zero tail).
    for i in 504..512 {
        // D[504..506] = 0.000030518 in the table, then D[506..512] =
        // 0.000015259. Verify both bands match.
        if i < 506 {
            assert_eq!(D_TABLE[i], 0.000030518, "D[{i}]");
        } else {
            assert_eq!(D_TABLE[i], 0.000015259, "D[{i}]");
        }
    }
}

#[test]
fn d_table_first_values_match_table_b3() {
    // D[0] = 0, D[1..7] = -0.000015259 (six entries), D[7..11] =
    // -0.000030518 (four entries) — verbatim from Table B.3.
    assert_eq!(D_TABLE[0], 0.0);
    for i in 1..7 {
        assert_eq!(D_TABLE[i], -0.000015259, "D[{i}]");
    }
    for i in 7..11 {
        assert_eq!(D_TABLE[i], -0.000030518, "D[{i}]");
    }
}
