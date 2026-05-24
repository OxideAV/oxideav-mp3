// Unit tests for the §2.4.3.4.9 stereo-processing stage. Patterns are
// derived from ISO/IEC 11172-3:1993 §2.4.3.4.9 (the MS matrix and the
// intensity-stereo steps) and ISO/IEC 13818-3:1997 §2.4.3.2 (the LSF
// intensity-stereo step 4/5 replacement), plus the §2.4.2.3
// mode_extension table and the Table B.8 band-start indices. No external
// implementation was consulted.

// This file is `include!`d into the `stereo` module, so its `use` lines
// (ModeExtension, MpegVersion, ScaleFactors, GranuleChannel, BlockType,
// NUM_LINES, band-start helpers) are already in scope.

/// A `mode_extension` with the two Layer III method bits set explicitly.
fn mode_ext(intensity: bool, ms: bool) -> ModeExtension {
    let raw = (u8::from(intensity) << 1) | u8::from(ms);
    ModeExtension {
        intensity_stereo: intensity,
        ms_stereo: ms,
        raw,
    }
}

/// Right-channel scalefactors with a single long-band `is_pos` set.
fn sf_long(sfb: usize, pos: u8) -> ScaleFactors {
    let mut long = [0u8; crate::scalefactors::LONG_SFB];
    long[sfb] = pos;
    ScaleFactors {
        long,
        ..ScaleFactors::default()
    }
}

/// A long-block right-channel record (no window switching).
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

/// A short-block right-channel record (window-switched, non-mixed).
fn short_gc() -> GranuleChannel {
    let mut gc = long_gc();
    gc.window_switching_flag = true;
    gc.block_type = BlockType::Short;
    gc
}

const SQRT2: f32 = std::f32::consts::SQRT_2;
const EPS: f32 = 1e-5;

fn approx(a: f32, b: f32) -> bool {
    (a - b).abs() <= EPS * (1.0 + a.abs() + b.abs())
}

// ---------------------------------------------------------------------
// mode_extension '00' — neither method active.
// ---------------------------------------------------------------------

#[test]
fn neither_method_passes_through() {
    // §2.4.2.3 '00': joint-stereo but no MS / no intensity. Both channels
    // must be left untouched (they are already L/R).
    let mut left = [0.0f32; NUM_LINES];
    let mut right = [0.0f32; NUM_LINES];
    for i in 0..NUM_LINES {
        left[i] = i as f32;
        right[i] = -(i as f32);
    }
    let l0 = left;
    let r0 = right;
    let sf = ScaleFactors::default();
    process_stereo(
        &mut left,
        &mut right,
        &sf,
        &long_gc(),
        mode_ext(false, false),
        44100,
        MpegVersion::Mpeg1,
    );
    assert_eq!(left, l0);
    assert_eq!(right, r0);
}

// ---------------------------------------------------------------------
// MS stereo (§2.4.3.4.9.2).
// ---------------------------------------------------------------------

#[test]
fn ms_only_whole_spectrum() {
    // mode_extension '10': MS on, intensity off. The ENTIRE spectrum is
    // MS-decoded: L = (M+S)/sqrt2, R = (M-S)/sqrt2 (§2.4.3.4.9.1/.2).
    let mut left = [0.0f32; NUM_LINES]; // M
    let mut right = [0.0f32; NUM_LINES]; // S
    for i in 0..NUM_LINES {
        left[i] = 2.0 + i as f32;
        right[i] = 1.0 - i as f32;
    }
    let m = left;
    let s = right;
    let sf = ScaleFactors::default();
    process_stereo(
        &mut left,
        &mut right,
        &sf,
        &long_gc(),
        mode_ext(false, true),
        44100,
        MpegVersion::Mpeg1,
    );
    for i in 0..NUM_LINES {
        assert!(approx(left[i], (m[i] + s[i]) / SQRT2), "L[{i}]");
        assert!(approx(right[i], (m[i] - s[i]) / SQRT2), "R[{i}]");
    }
}

#[test]
fn ms_inverse_is_invertible() {
    // A unit M/S pair: M=1, S=0 -> L=R=1/sqrt2; M=0, S=1 -> L=1/sqrt2,
    // R=-1/sqrt2. Confirms the matrix orientation matches §2.4.3.4.9.2.
    let mut left = [0.0f32; NUM_LINES];
    let mut right = [0.0f32; NUM_LINES];
    left[0] = 1.0;
    right[0] = 0.0;
    left[1] = 0.0;
    right[1] = 1.0;
    let sf = ScaleFactors::default();
    process_stereo(
        &mut left,
        &mut right,
        &sf,
        &long_gc(),
        mode_ext(false, true),
        44100,
        MpegVersion::Mpeg1,
    );
    let inv = 1.0 / SQRT2;
    assert!(approx(left[0], inv));
    assert!(approx(right[0], inv));
    assert!(approx(left[1], inv));
    assert!(approx(right[1], -inv));
}

// ---------------------------------------------------------------------
// Intensity stereo, MPEG-1 (§2.4.3.4.9.3).
// ---------------------------------------------------------------------

#[test]
fn intensity_only_mpeg1_long() {
    // mode_extension '01': intensity on, MS off. Right channel non-zero
    // only in band 0 (line 0), so the intensity bound is band 1: bands
    // 1..21 are intensity-coded using is_pos = right scalefactor.
    let mut left = [0.0f32; NUM_LINES];
    let mut right = [0.0f32; NUM_LINES];
    // Below-bound band 0: pass-through (MS off) -> unchanged.
    left[0] = 5.0;
    right[0] = 3.0;
    // Intensity magnitude in band 1 (line 4 at 44.1k) carried by left.
    let starts = long_band_starts(44100, MpegVersion::Mpeg1);
    let b1 = starts[1]; // first line of band 1
    left[b1] = 8.0;
    right[b1] = 0.0; // zero-part

    // is_pos for band 1 = 3 -> is_ratio = tan(3*pi/12) = tan(pi/4) = 1.
    let sf = sf_long(1, 3);

    process_stereo(
        &mut left,
        &mut right,
        &sf,
        &long_gc(),
        mode_ext(true, false),
        44100,
        MpegVersion::Mpeg1,
    );

    // Band 0 below bound, MS off -> unchanged.
    assert!(approx(left[0], 5.0));
    assert!(approx(right[0], 3.0));
    // Band 1 intensity, is_ratio=1: kl = 1/(1+1)=0.5, kr = 1/(1+1)=0.5.
    assert!(approx(left[b1], 8.0 * 0.5));
    assert!(approx(right[b1], 8.0 * 0.5));
}

#[test]
fn intensity_mpeg1_is_pos_zero_full_pan_left() {
    // is_pos = 0 -> is_ratio = tan(0) = 0 -> kl = 0/(1+0) = 0,
    // kr = 1/(1+0) = 1: full pan to the right channel (L=0, R=L_orig).
    let mut left = [0.0f32; NUM_LINES];
    let mut right = [0.0f32; NUM_LINES];
    let starts = long_band_starts(44100, MpegVersion::Mpeg1);
    let b1 = starts[1];
    left[b1] = 9.0;
    let sf = sf_long(1, 0);
    process_stereo(
        &mut left,
        &mut right,
        &sf,
        &long_gc(),
        mode_ext(true, false),
        44100,
        MpegVersion::Mpeg1,
    );
    assert!(approx(left[b1], 0.0));
    assert!(approx(right[b1], 9.0));
}

#[test]
fn intensity_mpeg1_illegal_is_pos_with_ms() {
    // is_pos = 7 is illegal: the band is NOT intensity-coded. With MS
    // enabled (mode_extension '11') that band falls back to the MS matrix.
    let mut left = [0.0f32; NUM_LINES];
    let mut right = [0.0f32; NUM_LINES];
    let starts = long_band_starts(44100, MpegVersion::Mpeg1);
    let b1 = starts[1];
    // Right channel zero from band 1 up -> bound = band 1.
    left[0] = 1.0;
    right[0] = 1.0; // band 0 below bound
    left[b1] = 4.0;
    right[b1] = 2.0; // present even though "intensity band": illegal pos.
    let sf = sf_long(1, 7); // illegal
    process_stereo(
        &mut left,
        &mut right,
        &sf,
        &long_gc(),
        mode_ext(true, true), // MS also on
        44100,
        MpegVersion::Mpeg1,
    );
    // Band 0 below bound + MS on -> MS matrix.
    assert!(approx(left[0], (1.0 + 1.0) / SQRT2));
    assert!(approx(right[0], (1.0 - 1.0) / SQRT2));
    // Band 1 illegal is_pos, MS on -> MS matrix on M=4, S=2.
    assert!(approx(left[b1], (4.0 + 2.0) / SQRT2));
    assert!(approx(right[b1], (4.0 - 2.0) / SQRT2));
}

#[test]
fn intensity_mpeg1_illegal_is_pos_no_ms_independent() {
    // is_pos = 7 illegal AND MS off -> the two channels stay independent
    // (unchanged) in that band.
    let mut left = [0.0f32; NUM_LINES];
    let mut right = [0.0f32; NUM_LINES];
    let starts = long_band_starts(44100, MpegVersion::Mpeg1);
    let b1 = starts[1];
    left[b1] = 4.0;
    right[b1] = 2.0;
    let sf = sf_long(1, 7);
    process_stereo(
        &mut left,
        &mut right,
        &sf,
        &long_gc(),
        mode_ext(true, false),
        44100,
        MpegVersion::Mpeg1,
    );
    assert!(approx(left[b1], 4.0));
    assert!(approx(right[b1], 2.0));
}

// ---------------------------------------------------------------------
// Both methods (§2.4.3.4.9.1): MS below bound, intensity above.
// ---------------------------------------------------------------------

#[test]
fn both_ms_below_intensity_above() {
    // mode_extension '11'. Right channel non-zero through band 2; bound is
    // band 3. Bands 0..3 -> MS; bands 3..21 -> intensity.
    let starts = long_band_starts(44100, MpegVersion::Mpeg1);
    let mut left = [0.0f32; NUM_LINES];
    let mut right = [0.0f32; NUM_LINES];
    // Below-bound MS-coded content in bands 0..3 (last nonzero in band 2).
    let last_b2 = starts[3] - 1; // last line of band 2
    left[0] = 6.0;
    right[0] = 2.0;
    left[last_b2] = 10.0;
    right[last_b2] = -4.0;
    // Intensity magnitude in band 3.
    let b3 = starts[3];
    left[b3] = 12.0;
    right[b3] = 0.0;
    let sf = sf_long(3, 6); // is_ratio = tan(6*pi/12) = tan(pi/2) -> huge.

    process_stereo(
        &mut left,
        &mut right,
        &sf,
        &long_gc(),
        mode_ext(true, true),
        44100,
        MpegVersion::Mpeg1,
    );

    // Band 0 (below bound) MS:
    assert!(approx(left[0], (6.0 + 2.0) / SQRT2));
    assert!(approx(right[0], (6.0 - 2.0) / SQRT2));
    // Band 2 (below bound) MS:
    assert!(approx(left[last_b2], (10.0 + -4.0) / SQRT2));
    assert!(approx(right[last_b2], (10.0 - -4.0) / SQRT2));
    // Band 3 intensity: is_pos=6 -> is_ratio very large, kl -> 1, kr -> 0:
    // nearly full pan to left.
    let is_ratio = (6.0 * std::f32::consts::PI / 12.0).tan();
    let kl = is_ratio / (1.0 + is_ratio);
    let kr = 1.0 / (1.0 + is_ratio);
    assert!(approx(left[b3], 12.0 * kl));
    assert!(approx(right[b3], 12.0 * kr));
}

// ---------------------------------------------------------------------
// LSF intensity stereo (ISO/IEC 13818-3 §2.4.3.2).
// ---------------------------------------------------------------------

#[test]
fn intensity_lsf_pos_zero_unity() {
    // LSF is_pos == 0 -> kl = kr = 1 (both channels = L_orig).
    let mut left = [0.0f32; NUM_LINES];
    let mut right = [0.0f32; NUM_LINES];
    let starts = long_band_starts(24000, MpegVersion::Mpeg2);
    let b1 = starts[1];
    left[b1] = 7.0;
    let sf = sf_long(1, 0);
    process_stereo(
        &mut left,
        &mut right,
        &sf,
        &long_gc(),
        mode_ext(true, false),
        24000,
        MpegVersion::Mpeg2,
    );
    assert!(approx(left[b1], 7.0));
    assert!(approx(right[b1], 7.0));
}

#[test]
fn intensity_lsf_odd_even_scale1() {
    // LSF, intensity_scale == 1 -> i0 = 1/sqrt2.
    // is_pos odd (=1): kl = i0^((1+1)/2) = i0^1 = 1/sqrt2, kr = 1.
    // is_pos even (=2): kl = 1, kr = i0^(2/2) = i0^1 = 1/sqrt2.
    let i0 = 1.0 / SQRT2;
    let starts = long_band_starts(24000, MpegVersion::Mpeg2);
    let b1 = starts[1];
    let b2 = starts[2];

    let mut left = [0.0f32; NUM_LINES];
    let mut right = [0.0f32; NUM_LINES];
    left[b1] = 4.0; // band 1: is_pos odd
    left[b2] = 6.0; // band 2: is_pos even
    let mut long = [0u8; crate::scalefactors::LONG_SFB];
    long[1] = 1;
    long[2] = 2;
    let sf = ScaleFactors {
        long,
        intensity_scale: true,
        ..ScaleFactors::default()
    };
    process_stereo(
        &mut left,
        &mut right,
        &sf,
        &long_gc(),
        mode_ext(true, false),
        24000,
        MpegVersion::Mpeg2,
    );
    // Band 1 odd: L = L*i0, R = L*1.
    assert!(approx(left[b1], 4.0 * i0));
    assert!(approx(right[b1], 4.0));
    // Band 2 even: L = L*1, R = L*i0.
    assert!(approx(left[b2], 6.0));
    assert!(approx(right[b2], 6.0 * i0));
}

#[test]
fn intensity_lsf_scale0_quarter_root() {
    // intensity_scale == 0 -> i0 = 1/sqrt(sqrt(2)) = 2^(-1/4).
    // is_pos = 2 (even) -> kr = i0^(2/2) = i0.
    let i0 = (1.0f32 / SQRT2).sqrt();
    let starts = long_band_starts(24000, MpegVersion::Mpeg2);
    let b1 = starts[1];
    let mut left = [0.0f32; NUM_LINES];
    let mut right = [0.0f32; NUM_LINES];
    left[b1] = 8.0;
    // intensity_scale: false is the default; set band 1's is_pos = 2.
    let sf = sf_long(1, 2);
    process_stereo(
        &mut left,
        &mut right,
        &sf,
        &long_gc(),
        mode_ext(true, false),
        24000,
        MpegVersion::Mpeg2,
    );
    assert!(approx(left[b1], 8.0));
    assert!(approx(right[b1], 8.0 * i0));
}

// ---------------------------------------------------------------------
// Short blocks: per-window intensity bound (ISO/IEC 13818-3 §2.4.3.2).
// ---------------------------------------------------------------------

#[test]
fn short_block_ms_only_whole_spectrum() {
    // MS only on a short block: whole spectrum MS (the short path is not
    // even taken for MS-only — process_stereo applies MS to all 576 lines
    // before dispatching on block type). Verify the matrix all the same.
    let mut left = [0.0f32; NUM_LINES];
    let mut right = [0.0f32; NUM_LINES];
    for i in 0..NUM_LINES {
        left[i] = (i as f32) * 0.5;
        right[i] = (i as f32) * 0.25;
    }
    let m = left;
    let s = right;
    let sf = ScaleFactors::default();
    process_stereo(
        &mut left,
        &mut right,
        &sf,
        &short_gc(),
        mode_ext(false, true),
        44100,
        MpegVersion::Mpeg1,
    );
    for i in 0..NUM_LINES {
        assert!(approx(left[i], (m[i] + s[i]) / SQRT2));
        assert!(approx(right[i], (m[i] - s[i]) / SQRT2));
    }
}

#[test]
fn short_block_intensity_per_window() {
    // Short block, intensity only. Reordered (subband-order) layout: short
    // band sfb with per-window [s, s+w) occupies interleaved lines
    // 3*s + 3*k + win. Put a non-zero right line only in window 0 of band
    // 0; windows 1 and 2 are entirely zero in the right channel, so their
    // intensity bound is band 0 (everything is intensity-coded). Window 0
    // is non-zero through band 0, so its bound is band 1.
    let starts = short_band_starts(44100, MpegVersion::Mpeg1);
    let mut left = [0.0f32; NUM_LINES];
    let mut right = [0.0f32; NUM_LINES];

    // Window 0, band 0, freqline 0 (interleaved line 0): right non-zero so
    // band 0 win 0 is below the bound (pass-through; MS off).
    left[0] = 5.0;
    right[0] = 2.0;

    // Window 1, band 0, freqline 0 -> interleaved line 3*0 + 3*0 + 1 = 1.
    // Right is zero everywhere in window 1, so band 0 is intensity-coded.
    left[1] = 10.0;
    // Window 2, band 1, freqline 0 -> 3*starts[1] + 0 + 2.
    let b1 = starts[1];
    let w2_b1 = 3 * b1 + 2;
    left[w2_b1] = 12.0;

    // is_pos for band 0 = 3 (is_ratio=1 -> kl=kr=0.5) across windows.
    let mut short = [[0u8; crate::scalefactors::SHORT_WINDOWS]; crate::scalefactors::SHORT_SFB];
    short[0][1] = 3; // window 1, band 0
    short[1][2] = 0; // window 2, band 1 -> kl=0, kr=1
    let sf = ScaleFactors {
        short,
        ..ScaleFactors::default()
    };

    process_stereo(
        &mut left,
        &mut right,
        &sf,
        &short_gc(),
        mode_ext(true, false),
        44100,
        MpegVersion::Mpeg1,
    );

    // Window 0 band 0 below its own bound, MS off -> unchanged.
    assert!(approx(left[0], 5.0));
    assert!(approx(right[0], 2.0));
    // Window 1 band 0 intensity, is_pos=3 -> kl=kr=0.5.
    assert!(approx(left[1], 10.0 * 0.5));
    assert!(approx(right[1], 10.0 * 0.5));
    // Window 2 band 1 intensity, is_pos=0 -> kl=0, kr=1.
    assert!(approx(left[w2_b1], 0.0));
    assert!(approx(right[w2_b1], 12.0));
}
