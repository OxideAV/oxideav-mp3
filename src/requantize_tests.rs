// Unit tests for the §2.4.3.4.7 requantization stage. Constructed from
// the ISO/IEC 11172-3:1993 §2.4.3.4.7.1 formulas and the Annex B tables
// (B.6 pretab, B.8 scalefactor bands). Float comparisons use a relative
// epsilon because the power-law term is computed in `f32`.

// This file is `include!`d into the `requantize` module, so the module's
// own `use` lines (MpegVersion, ScaleFactors, BlockType, GranuleChannel)
// are already in scope.

/// A long-block granule-channel record with the given global gain /
/// scalefac_scale / preflag; every other field zeroed.
fn long_gc(global_gain: u8, scalefac_scale: bool) -> GranuleChannel {
    GranuleChannel {
        part2_3_length: 0,
        big_values: 0,
        global_gain,
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

/// A short-block (window-switched, non-mixed) granule-channel record.
fn short_gc(global_gain: u8, subblock_gain: [u8; 3], scalefac_scale: bool) -> GranuleChannel {
    GranuleChannel {
        part2_3_length: 0,
        big_values: 0,
        global_gain,
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
        count1table_select: scalefac_scale,
    }
}

fn approx(a: f32, b: f32) {
    let tol = 1e-4 * b.abs().max(1.0);
    assert!(
        (a - b).abs() <= tol,
        "expected {b}, got {a} (tol {tol})"
    );
}

#[test]
fn pretab_matches_table_b6() {
    // Table B.6 — Layer III preemphasis (pretab), ISO/IEC 11172-3 p.53.
    assert_eq!(
        PRETAB,
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 2]
    );
    assert_eq!(PRETAB.len(), 21);
    // The amplification only kicks in at band 11 and peaks at 3.
    assert_eq!(PRETAB[10], 0);
    assert_eq!(PRETAB[11], 1);
    assert_eq!(*PRETAB.iter().max().unwrap(), 3);
}

#[test]
fn scalefac_multiplier_table() {
    // §2.4.2.7: scalefac_scale 0 -> 0.5, 1 -> 1.0.
    assert_eq!(scalefac_multiplier(false), 0.5);
    assert_eq!(scalefac_multiplier(true), 1.0);
}

#[test]
fn long_block_unit_gain_identity() {
    // global_gain == 210 makes the gain term 2^0 == 1; all scalefactors
    // zero -> xr_i = sign(is_i)*|is_i|^(4/3).
    let gc = long_gc(210, false);
    let sf = ScaleFactors::default();
    let mut is = [0i32; NUM_LINES];
    is[0] = 1; // 1^(4/3) = 1
    is[1] = -8; // 8^(4/3) = 16, negated
    is[2] = 27; // 27^(4/3) = 81
    is[3] = 0; // exact zero
    let xr = requantize(&is, &gc, &sf, 44100, MpegVersion::Mpeg1);
    approx(xr[0], 1.0);
    approx(xr[1], -16.0);
    approx(xr[2], 81.0);
    assert_eq!(xr[3], 0.0);
}

#[test]
fn long_block_global_gain_step() {
    // Increasing global_gain by 4 multiplies the gain term by 2^1 = 2.
    let sf = ScaleFactors::default();
    let mut is = [0i32; NUM_LINES];
    is[0] = 1;
    let base = requantize(&is, &long_gc(210, false), &sf, 44100, MpegVersion::Mpeg1);
    let up4 = requantize(&is, &long_gc(214, false), &sf, 44100, MpegVersion::Mpeg1);
    approx(up4[0], base[0] * 2.0);
    // global_gain 206 -> 2^((206-210)/4) = 2^-1 = 0.5.
    let down4 = requantize(&is, &long_gc(206, false), &sf, 44100, MpegVersion::Mpeg1);
    approx(down4[0], base[0] * 0.5);
}

#[test]
fn long_block_scalefactor_term() {
    // scalefac_l[0] = 2, scalefac_scale = 0 -> multiplier 0.5 ->
    // 2^(-(0.5*2)) = 2^-1 = 0.5. global_gain 210 -> gain 1.
    let gc = long_gc(210, false);
    let mut sf = ScaleFactors::default();
    sf.long[0] = 2;
    let mut is = [0i32; NUM_LINES];
    is[0] = 1; // band 0 (lines 0..3 at 44.1k)
    is[4] = 1; // band 1 (no scalefactor) -> unchanged
    let xr = requantize(&is, &gc, &sf, 44100, MpegVersion::Mpeg1);
    approx(xr[0], 0.5);
    approx(xr[4], 1.0);
}

#[test]
fn long_block_scalefac_scale_one_doubles_exponent() {
    // scalefac_scale = 1 -> multiplier 1.0; scalefac_l[0]=2 ->
    // 2^(-(1.0*2)) = 2^-2 = 0.25.
    let gc = long_gc(210, true);
    let mut sf = ScaleFactors::default();
    sf.long[0] = 2;
    let mut is = [0i32; NUM_LINES];
    is[0] = 1;
    let xr = requantize(&is, &gc, &sf, 44100, MpegVersion::Mpeg1);
    approx(xr[0], 0.25);
}

#[test]
fn long_block_preflag_adds_pretab() {
    // preflag adds pretab[sfb] to the scalefactor. Band 11 has
    // pretab=1; with scalefac_l[11]=0, scalefac_scale=0 ->
    // 2^(-(0.5*(0+1))) = 2^-0.5.
    let gc = long_gc(210, false);
    let sf = ScaleFactors {
        preflag: true,
        ..ScaleFactors::default()
    };
    // Band 11 at 44.1k starts at line 62.
    let mut is = [0i32; NUM_LINES];
    is[62] = 1;
    // Band 0 (pretab 0) unaffected.
    is[0] = 1;
    let xr = requantize(&is, &gc, &sf, 44100, MpegVersion::Mpeg1);
    approx(xr[62], 2f32.powf(-0.5));
    approx(xr[0], 1.0);
}

#[test]
fn long_block_preflag_off_ignores_pretab() {
    // Without preflag, pretab is not applied even on band 11.
    let gc = long_gc(210, false);
    let sf = ScaleFactors::default(); // preflag false
    let mut is = [0i32; NUM_LINES];
    is[62] = 1; // band 11 at 44.1k
    let xr = requantize(&is, &gc, &sf, 44100, MpegVersion::Mpeg1);
    approx(xr[62], 1.0);
}

#[test]
fn short_block_subblock_gain_per_window() {
    // Non-mixed short block. subblock_gain[1] = 1 lowers window 1 by
    // 2^(-8/4) = 2^-2 = 0.25 relative to window 0. global_gain 210.
    // 44.1k short band 0 has per-window width 4; interleaved layout is
    // window0[0..4], window1[4..8], window2[8..12].
    let gc = short_gc(210, [0, 1, 0], false);
    let sf = ScaleFactors::default();
    let mut is = [0i32; NUM_LINES];
    is[0] = 1; // band 0, window 0, freqline 0
    is[4] = 1; // band 0, window 1, freqline 0
    is[8] = 1; // band 0, window 2, freqline 0
    let xr = requantize(&is, &gc, &sf, 44100, MpegVersion::Mpeg1);
    approx(xr[0], 1.0); // 2^((210-210-0)/4) = 1
    approx(xr[4], 0.25); // window 1 dropped by 2^-2
    approx(xr[8], 1.0); // window 2 unchanged
}

#[test]
fn short_block_scalefactor_per_window() {
    // Per-window scalefac_s applies independently. Band 0 window 0 gets
    // scalefac 2 (mult 0.5 -> 2^-1 = 0.5); window 2 gets scalefac 4
    // (2^-2 = 0.25). global_gain 210, subblock_gain 0.
    let gc = short_gc(210, [0, 0, 0], false);
    let mut sf = ScaleFactors::default();
    sf.short[0][0] = 2;
    sf.short[0][2] = 4;
    let mut is = [0i32; NUM_LINES];
    is[0] = 1; // win 0
    is[4] = 1; // win 1
    is[8] = 1; // win 2
    let xr = requantize(&is, &gc, &sf, 44100, MpegVersion::Mpeg1);
    approx(xr[0], 0.5);
    approx(xr[4], 1.0);
    approx(xr[8], 0.25);
}

#[test]
fn short_block_band_one_interleave() {
    // Band 1 at 44.1k: per-window start 4, width 4 -> interleaved lines
    // 12..24 (win0 12..16, win1 16..20, win2 20..24). Verify a value in
    // band 1, window 2 picks up scalefac_s[1][2].
    let gc = short_gc(210, [0, 0, 0], false);
    let mut sf = ScaleFactors::default();
    sf.short[1][2] = 2; // 2^-1 = 0.5
    let mut is = [0i32; NUM_LINES];
    is[20] = 1; // band 1, window 2, freqline 0
    let xr = requantize(&is, &gc, &sf, 44100, MpegVersion::Mpeg1);
    approx(xr[20], 0.5);
}

#[test]
fn mixed_block_long_then_short() {
    // Mixed block: lines 0..36 are long, the rest short. A value in the
    // long portion uses scalefac_l; a value in the short portion uses
    // scalefac_s with the per-window subblock_gain.
    let mut gc = short_gc(210, [0, 0, 0], false);
    gc.mixed_block_flag = true;
    let mut sf = ScaleFactors::default();
    sf.long[0] = 2; // long band 0, mult 0.5 -> 0.5
    sf.short[3][0] = 2; // first short band used in mixed mode
    let mut is = [0i32; NUM_LINES];
    is[0] = 1; // long band 0
    // Short band 3 at 44.1k: per-window start 12, so interleaved base
    // 3*12 = 36, window 0 freqline 0.
    is[36] = 1;
    let xr = requantize(&is, &gc, &sf, 44100, MpegVersion::Mpeg1);
    approx(xr[0], 0.5); // long formula with scalefac_l[0]=2
    approx(xr[36], 0.5); // short formula with scalefac_s[3][0]=2
}

#[test]
fn mixed_block_short_portion_skips_low_bands() {
    // In a mixed block the lowest 36 lines are long; short bands 0..3
    // are absorbed there, so requantize_short_range starts at sfb 3 and
    // must not overwrite the long-coded lines 0..36.
    let mut gc = short_gc(210, [0, 0, 0], false);
    gc.mixed_block_flag = true;
    let mut sf = ScaleFactors::default();
    // Give long band 0 a non-trivial scalefactor and a value; if the
    // short pass mistakenly touched it, the value would change.
    sf.long[0] = 4; // mult 0.5 -> 2^-2 = 0.25
    let mut is = [0i32; NUM_LINES];
    is[0] = 1;
    let xr = requantize(&is, &gc, &sf, 44100, MpegVersion::Mpeg1);
    approx(xr[0], 0.25);
}

#[test]
fn lsf_uses_same_formula() {
    // LSF (MPEG-2) requantization uses the identical §2.4.3.4.7.1
    // formula (ISO/IEC 13818-3 defers to ISO/IEC 11172-3 2.4.3.4). A
    // long LSF block at 24 kHz with global_gain 210, scalefac_l[0]=2,
    // scalefac_scale=0 -> 2^-1 = 0.5.
    let gc = long_gc(210, false);
    let mut sf = ScaleFactors::default();
    sf.long[0] = 2;
    let mut is = [0i32; NUM_LINES];
    is[0] = 1;
    let xr = requantize(&is, &gc, &sf, 24000, MpegVersion::Mpeg2);
    approx(xr[0], 0.5);
}

#[test]
fn lsf_preflag_amplifies_high_bands() {
    // LSF derives preflag from scalefac_compress; here we exercise the
    // resolved effective preflag (ScaleFactors::preflag) on a band with
    // pretab > 0. Band 17 has pretab 3; scalefac_scale 0 ->
    // 2^(-(0.5*3)) = 2^-1.5.
    let gc = long_gc(210, false);
    let sf = ScaleFactors {
        preflag: true,
        ..ScaleFactors::default()
    };
    // Band 17 at 24 kHz starts at line 278 (ISO/IEC 13818-3:1997
    // Table B.2, 24 kHz long blocks: "index of start" 278).
    let mut is = [0i32; NUM_LINES];
    is[278] = 1;
    let xr = requantize(&is, &gc, &sf, 24000, MpegVersion::Mpeg2);
    approx(xr[278], 2f32.powf(-1.5));
}

#[test]
fn sign_is_preserved_through_power_law() {
    let gc = long_gc(210, false);
    let sf = ScaleFactors::default();
    let mut is = [0i32; NUM_LINES];
    is[0] = -1;
    is[1] = -27;
    is[2] = 27;
    let xr = requantize(&is, &gc, &sf, 44100, MpegVersion::Mpeg1);
    approx(xr[0], -1.0);
    approx(xr[1], -81.0);
    approx(xr[2], 81.0);
}

#[test]
fn all_zero_input_yields_all_zero_output() {
    let gc = long_gc(123, true);
    let sf = ScaleFactors::default();
    let is = [0i32; NUM_LINES];
    let xr = requantize(&is, &gc, &sf, 48000, MpegVersion::Mpeg1);
    assert!(xr.iter().all(|&v| v == 0.0));
}

#[test]
fn band_boundaries_select_correct_scalefactor_44k() {
    // At 44.1k, band 1 starts at line 4 and band 2 at line 8. A value at
    // line 4 must use scalefac_l[1], at line 7 still band 1, at line 8
    // band 2.
    let gc = long_gc(210, false);
    let mut sf = ScaleFactors::default();
    sf.long[1] = 2; // 2^-1 = 0.5
    sf.long[2] = 4; // 2^-2 = 0.25
    let mut is = [0i32; NUM_LINES];
    is[3] = 1; // band 0 (no sf)
    is[4] = 1; // band 1
    is[7] = 1; // band 1
    is[8] = 1; // band 2
    let xr = requantize(&is, &gc, &sf, 44100, MpegVersion::Mpeg1);
    approx(xr[3], 1.0);
    approx(xr[4], 0.5);
    approx(xr[7], 0.5);
    approx(xr[8], 0.25);
}

#[test]
fn power_law_magnitude_8191_finite() {
    // Largest magnitude reachable via the 16x16 ESC tables (15 + linbits)
    // is well below 8191; check a large value stays finite and positive.
    let gc = long_gc(255, true);
    let sf = ScaleFactors::default();
    let mut is = [0i32; NUM_LINES];
    is[0] = 8191;
    let xr = requantize(&is, &gc, &sf, 44100, MpegVersion::Mpeg1);
    assert!(xr[0].is_finite());
    assert!(xr[0] > 0.0);
    // 8191^(4/3) is the magnitude; gain 2^((255-210)/4).
    let expected = (8191f32).powf(4.0 / 3.0) * (45.0 / 4.0f32).exp2();
    approx(xr[0], expected);
}

// --- MPEG-2.5 (8 / 11.025 / 12 kHz) scalefactor-band tables ---
// docs/audio/mp3/mpeg2.5-scalefactor-bands.md (#147/#151). The 11.025 /
// 12 kHz tables are byte-identical to the 22.05 / 24 kHz LSF tables; the
// 8 kHz table is the distinct Fraunhofer table with collapsed top bands.

/// Every long table has 22 entries forming a contiguous, strictly
/// non-decreasing partition of the 576-line granule, with band 21
/// spanning `starts[21]..576` (cross-check column of the doc).
fn assert_long_invariants(starts: &[usize; 22]) {
    assert_eq!(starts[0], 0, "band 0 starts at line 0");
    for w in starts.windows(2) {
        assert!(w[0] < w[1], "long bands must be strictly increasing");
    }
    assert!(starts[21] < NUM_LINES, "band 21 spans starts[21]..576");
}

/// Every short table has 13 per-window entries partitioning the 192-line
/// window, band 12 spanning `starts[12]..192`.
fn assert_short_invariants(starts: &[usize; 13]) {
    assert_eq!(starts[0], 0);
    for w in starts.windows(2) {
        assert!(w[0] < w[1], "short bands must be strictly increasing");
    }
    assert!(starts[12] < 192, "band 12 spans starts[12]..192 per window");
}

#[test]
fn mpeg25_8khz_long_table_exact() {
    // docs §"8 kHz, long blocks": widths 12×6, 16,20,24,28,32,40,48,56,
    // 64,76,90, then five width-2 filler bands. Σ widths = 576.
    let starts = long_band_starts(8000, MpegVersion::Mpeg25);
    assert_eq!(
        *starts,
        [
            0, 12, 24, 36, 48, 60, 72, 88, 108, 132, 160, 192, 232, 280, 336, 400, 476, 566, 568,
            570, 572, 574,
        ]
    );
    assert_long_invariants(starts);
    // Top long bands 17..21 collapse to width 2 (the 8 kHz signature).
    for sfb in 17..21 {
        assert_eq!(starts[sfb + 1] - starts[sfb], 2, "band {sfb} width 2");
    }
    assert_eq!(NUM_LINES - starts[21], 2, "band 21 width 2");
}

#[test]
fn mpeg25_8khz_short_table_exact() {
    // docs §"8 kHz, short blocks": widths 8,8,8,12,16,20,24,28,36, three
    // width-2 fillers, then band 12 sweeping the residual 26 lines.
    let starts = short_band_starts(8000, MpegVersion::Mpeg25);
    assert_eq!(*starts, [0, 8, 16, 24, 36, 52, 72, 96, 124, 160, 162, 164, 166]);
    assert_short_invariants(starts);
    for sfb in 9..12 {
        assert_eq!(starts[sfb + 1] - starts[sfb], 2, "band {sfb} width 2");
    }
    assert_eq!(192 - starts[12], 26, "band 12 width 26");
}

#[test]
fn mpeg25_11025_reuses_lsf_22050() {
    // docs key result: 11.025 kHz tables are byte-identical to the
    // in-repo 13818-3 22.05 kHz LSF tables (long + short).
    assert_eq!(
        long_band_starts(11025, MpegVersion::Mpeg25),
        long_band_starts(22050, MpegVersion::Mpeg2),
        "11.025 kHz long == 22.05 kHz LSF long"
    );
    assert_eq!(
        short_band_starts(11025, MpegVersion::Mpeg25),
        short_band_starts(22050, MpegVersion::Mpeg2),
        "11.025 kHz short == 22.05 kHz LSF short"
    );
}

#[test]
fn mpeg25_12000_reuses_lsf_24000() {
    // docs key result: 12 kHz tables are byte-identical to the in-repo
    // 13818-3 24 kHz LSF tables (long + short).
    assert_eq!(
        long_band_starts(12000, MpegVersion::Mpeg25),
        long_band_starts(24000, MpegVersion::Mpeg2),
        "12 kHz long == 24 kHz LSF long"
    );
    assert_eq!(
        short_band_starts(12000, MpegVersion::Mpeg25),
        short_band_starts(24000, MpegVersion::Mpeg2),
        "12 kHz short == 24 kHz LSF short"
    );
}

#[test]
fn mpeg25_all_rates_satisfy_band_invariants() {
    for &rate in &[8000u32, 11025, 12000] {
        assert_long_invariants(long_band_starts(rate, MpegVersion::Mpeg25));
        assert_short_invariants(short_band_starts(rate, MpegVersion::Mpeg25));
    }
}

#[test]
fn mpeg25_8khz_no_longer_aliases_mpeg1_32k() {
    // Regression: before #147/#151 the 8 kHz rate borrowed the 32 kHz
    // MPEG-1 layout as a placeholder. It must now resolve to the distinct
    // Fraunhofer 8 kHz table.
    assert_ne!(
        long_band_starts(8000, MpegVersion::Mpeg25),
        long_band_starts(32000, MpegVersion::Mpeg1),
        "8 kHz long must not reuse the 32 kHz placeholder"
    );
    assert_ne!(
        short_band_starts(8000, MpegVersion::Mpeg25),
        short_band_starts(32000, MpegVersion::Mpeg1),
        "8 kHz short must not reuse the 32 kHz placeholder"
    );
}

#[test]
fn mpeg25_8khz_band_boundaries_select_correct_scalefactor() {
    // At 8 kHz, long band 0 spans lines 0..12, band 1 12..24, band 2
    // 24..36. A line in band 1 must pick scalefac_l[1], band 2 [2].
    let gc = long_gc(210, false);
    let mut sf = ScaleFactors::default();
    sf.long[1] = 2; // 2^-1 = 0.5
    sf.long[2] = 4; // 2^-2 = 0.25
    let mut is = [0i32; NUM_LINES];
    is[0] = 1; // band 0 (no sf)
    is[12] = 1; // band 1
    is[23] = 1; // band 1 (last line)
    is[24] = 1; // band 2
    let xr = requantize(&is, &gc, &sf, 8000, MpegVersion::Mpeg25);
    approx(xr[0], 1.0);
    approx(xr[12], 0.5);
    approx(xr[23], 0.5);
    approx(xr[24], 0.25);
}
