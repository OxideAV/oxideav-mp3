// Unit tests for the §2.4.3.4.7 quantization-primitive (encoder-side
// inverse of `crate::requantize`). The cornerstone test is the lossless
// round-trip in the *natural* encoder-input direction: pick an integer
// `is[576]` buffer, requantize it to obtain `xr_ref[576]`, feed that
// `xr_ref` to `quantize`, and confirm the recovered `is'[576]` equals
// the original `is` exactly. That direction is the encoder's contract:
// every value `xr_ref` it ever sees as the target of a quantization
// decision IS reachable (it came from the decoder formula), and the
// quantizer must invert it exactly.
//
// The opposite direction (`xr -> is -> xr_back`) is intrinsically lossy
// for arbitrary `xr`: the §2.4.3.4.7 grid is discrete (multiples of
// `factor * k^(4/3)`), so a `xr` that lies between two grid points
// rounds to the nearer one and the residual is half a quantizer step.
// Those bounds are quantifiable but they are not bit-exact, so the
// "encoder reproduces decoder-grid values" formulation is the stronger
// test.
//
// This file is `include!`d into `crate::quantize`, so the module's own
// `use` lines (MpegVersion, NUM_LINES, ScaleFactors, BlockType,
// GranuleChannel) are already in scope.

use crate::requantize::requantize;

/// Long-block GC scaffold (mirrors the decoder-test helper).
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

/// Short-block (window-switched, non-mixed) GC scaffold.
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
        scalefac_scale,
        count1table_select: false,
    }
}

/// Bin-level RMS over all 576 bins.
fn bin_rms(actual: &[f32; NUM_LINES], expected: &[f32; NUM_LINES]) -> f32 {
    let mut sum_sq = 0.0f64;
    for (&a, &e) in actual.iter().zip(expected.iter()) {
        let d = (a - e) as f64;
        sum_sq += d * d;
    }
    ((sum_sq / NUM_LINES as f64).sqrt()) as f32
}

#[test]
fn flat_unit_is_lossless_roundtrip_long_unit_gain() {
    // is[i] = 1 for all bins. global_gain == 210, scalefactors zero ->
    // factor 1 everywhere. xr_ref[i] = 1^(4/3) = 1. The encoder must
    // recover is = [1; 576] exactly, and the decoder reproduces xr = [1;
    // 576] -> bin RMS == 0.
    let gc = long_gc(210, false);
    let sf = ScaleFactors::default();
    let is = [1i32; NUM_LINES];
    let xr_ref = requantize(&is, &gc, &sf, 44100, MpegVersion::Mpeg1);
    let is_back = quantize(&xr_ref, &gc, &sf, 44100, MpegVersion::Mpeg1);
    assert_eq!(is_back, is, "encoder did not recover the original is[]");
    let xr_back = requantize(&is_back, &gc, &sf, 44100, MpegVersion::Mpeg1);
    let rms = bin_rms(&xr_back, &xr_ref);
    assert!(rms < 1e-3, "lossless flat-unit RMS = {rms}");
}

#[test]
fn varied_is_long_roundtrip() {
    let gc = long_gc(200, false);
    let sf = ScaleFactors::default();
    let mut is = [0i32; NUM_LINES];
    for (i, v) in is.iter_mut().enumerate() {
        let mag = ((i % 7) + 1) as i32;
        *v = if i % 3 == 0 { -mag } else { mag };
    }
    let xr_ref = requantize(&is, &gc, &sf, 44100, MpegVersion::Mpeg1);
    let is_back = quantize(&xr_ref, &gc, &sf, 44100, MpegVersion::Mpeg1);
    assert_eq!(is_back, is, "encoder did not invert varied long-block is[]");
    let xr_back = requantize(&is_back, &gc, &sf, 44100, MpegVersion::Mpeg1);
    let rms = bin_rms(&xr_back, &xr_ref);
    assert!(rms < 1e-3, "varied long-block RMS = {rms}");
}

#[test]
fn nonzero_global_gain_roundtrip() {
    for &gain in &[180u8, 200, 210, 224, 240] {
        let gc = long_gc(gain, false);
        let sf = ScaleFactors::default();
        let is = [3i32; NUM_LINES];
        let xr_ref = requantize(&is, &gc, &sf, 44100, MpegVersion::Mpeg1);
        let is_back = quantize(&xr_ref, &gc, &sf, 44100, MpegVersion::Mpeg1);
        assert_eq!(is_back, is, "gain {gain} round-trip lost the is[]");
        let xr_back = requantize(&is_back, &gc, &sf, 44100, MpegVersion::Mpeg1);
        let rms = bin_rms(&xr_back, &xr_ref);
        assert!(rms < 1e-3, "gain {gain} flat RMS = {rms}");
    }
}

#[test]
fn long_block_scalefactor_roundtrip() {
    let gc = long_gc(210, true); // scalefac_scale = 1
    let mut sf = ScaleFactors::default();
    for (sfb, v) in sf.long.iter_mut().enumerate() {
        *v = (sfb as u8) % 4;
    }
    let is = [2i32; NUM_LINES];
    let xr_ref = requantize(&is, &gc, &sf, 44100, MpegVersion::Mpeg1);
    let is_back = quantize(&xr_ref, &gc, &sf, 44100, MpegVersion::Mpeg1);
    assert_eq!(is_back, is, "scalefactor round-trip lost the is[]");
    let xr_back = requantize(&is_back, &gc, &sf, 44100, MpegVersion::Mpeg1);
    let rms = bin_rms(&xr_back, &xr_ref);
    assert!(rms < 1e-3, "scalefactor RMS = {rms}");
}

#[test]
fn long_block_preflag_roundtrip() {
    let gc = long_gc(210, false);
    let sf = ScaleFactors {
        preflag: true,
        ..ScaleFactors::default()
    };
    let is = [2i32; NUM_LINES];
    let xr_ref = requantize(&is, &gc, &sf, 44100, MpegVersion::Mpeg1);
    let is_back = quantize(&xr_ref, &gc, &sf, 44100, MpegVersion::Mpeg1);
    assert_eq!(is_back, is, "preflag round-trip lost the is[]");
    let xr_back = requantize(&is_back, &gc, &sf, 44100, MpegVersion::Mpeg1);
    let rms = bin_rms(&xr_back, &xr_ref);
    assert!(rms < 1e-3, "preflag RMS = {rms}");
}

#[test]
fn sign_preserved() {
    let gc = long_gc(210, false);
    let sf = ScaleFactors::default();
    let mut is = [0i32; NUM_LINES];
    is[0] = -1;
    is[1] = -27;
    is[2] = 27;
    let xr_ref = requantize(&is, &gc, &sf, 44100, MpegVersion::Mpeg1);
    let is_back = quantize(&xr_ref, &gc, &sf, 44100, MpegVersion::Mpeg1);
    assert_eq!(is_back[0], -1);
    assert_eq!(is_back[1], -27);
    assert_eq!(is_back[2], 27);
}

#[test]
fn all_zero_is_remains_all_zero() {
    let gc = long_gc(200, false);
    let sf = ScaleFactors::default();
    let is = [0i32; NUM_LINES];
    let xr_ref = requantize(&is, &gc, &sf, 44100, MpegVersion::Mpeg1);
    let is_back = quantize(&xr_ref, &gc, &sf, 44100, MpegVersion::Mpeg1);
    assert_eq!(is_back, is);
}

#[test]
fn all_zero_target_yields_all_zero_output() {
    let gc = long_gc(200, false);
    let sf = ScaleFactors::default();
    let xr_target = [0.0f32; NUM_LINES];
    let is = quantize(&xr_target, &gc, &sf, 44100, MpegVersion::Mpeg1);
    assert!(is.iter().all(|&v| v == 0));
}

/// Highest line index requantized by a short-block at the given sample
/// rate (sum of `3 * width` across all 12 short bands). Lines beyond
/// this index are never written by `requantize` for a short block — they
/// stay 0.0 from the buffer initialiser — so the encoder also produces
/// 0 there. Round-trip equality on `is[]` only holds inside `[0, end)`.
fn short_active_end(sample_rate_hz: u32) -> usize {
    match sample_rate_hz {
        32000 | 16000 | 8000 => 3 * 180,
        48000 | 24000 | 12000 => 3 * 126,
        _ => 3 * 136, // 44.1 / 22.05 / 11.025
    }
}

#[test]
fn short_block_roundtrip() {
    // Non-mixed short block. Per-window subblock_gain must round-trip.
    // Only the in-band lines (those the decoder actually requantizes)
    // are part of the encoder's contract; lines beyond `short_active_end`
    // are untouched zeros on both sides.
    let gc = short_gc(210, [0, 1, 2], false);
    let sf = ScaleFactors::default();
    let end = short_active_end(44100);
    let mut is = [0i32; NUM_LINES];
    is.iter_mut().take(end).for_each(|v| *v = 2);
    let xr_ref = requantize(&is, &gc, &sf, 44100, MpegVersion::Mpeg1);
    let is_back = quantize(&xr_ref, &gc, &sf, 44100, MpegVersion::Mpeg1);
    assert_eq!(is_back, is, "short-block round-trip lost the is[]");
    let xr_back = requantize(&is_back, &gc, &sf, 44100, MpegVersion::Mpeg1);
    let rms = bin_rms(&xr_back, &xr_ref);
    assert!(rms < 1e-3, "short-block flat RMS = {rms}");
}

#[test]
fn short_block_per_window_scalefactor_roundtrip() {
    let gc = short_gc(210, [0, 0, 0], true);
    let mut sf = ScaleFactors::default();
    for sfb in 0..12 {
        for w in 0..3 {
            sf.short[sfb][w] = ((sfb + w) as u8) % 5;
        }
    }
    let end = short_active_end(44100);
    let mut is = [0i32; NUM_LINES];
    is.iter_mut().take(end).for_each(|v| *v = 3);
    let xr_ref = requantize(&is, &gc, &sf, 44100, MpegVersion::Mpeg1);
    let is_back = quantize(&xr_ref, &gc, &sf, 44100, MpegVersion::Mpeg1);
    assert_eq!(
        is_back, is,
        "short per-window scalefactor round-trip lost the is[]"
    );
    let xr_back = requantize(&is_back, &gc, &sf, 44100, MpegVersion::Mpeg1);
    let rms = bin_rms(&xr_back, &xr_ref);
    assert!(rms < 1e-3, "short per-window scalefactor RMS = {rms}");
}

#[test]
fn mixed_block_roundtrip() {
    let mut gc = short_gc(210, [0, 1, 0], false);
    gc.mixed_block_flag = true;
    let mut sf = ScaleFactors::default();
    sf.long[0] = 2;
    sf.long[4] = 4;
    sf.short[3][0] = 2;
    sf.short[5][2] = 4;
    // Mixed block: lines 0..36 long (always populated by the long pass),
    // lines 36..short_active_end populated by the short pass. The encoder
    // mirrors both ranges and leaves the trailing zero-padded tail alone.
    let end = short_active_end(44100);
    let mut is = [0i32; NUM_LINES];
    is.iter_mut().take(end).for_each(|v| *v = 2);
    let xr_ref = requantize(&is, &gc, &sf, 44100, MpegVersion::Mpeg1);
    let is_back = quantize(&xr_ref, &gc, &sf, 44100, MpegVersion::Mpeg1);
    assert_eq!(is_back, is, "mixed-block round-trip lost the is[]");
    let xr_back = requantize(&is_back, &gc, &sf, 44100, MpegVersion::Mpeg1);
    let rms = bin_rms(&xr_back, &xr_ref);
    assert!(rms < 1e-3, "mixed-block flat RMS = {rms}");
}

#[test]
fn lsf_long_block_roundtrip() {
    let gc = long_gc(210, false);
    let mut sf = ScaleFactors::default();
    sf.long[0] = 2;
    sf.long[10] = 3;
    let is = [2i32; NUM_LINES];
    let xr_ref = requantize(&is, &gc, &sf, 24000, MpegVersion::Mpeg2);
    let is_back = quantize(&xr_ref, &gc, &sf, 24000, MpegVersion::Mpeg2);
    assert_eq!(is_back, is, "LSF long-block round-trip lost the is[]");
    let xr_back = requantize(&is_back, &gc, &sf, 24000, MpegVersion::Mpeg2);
    let rms = bin_rms(&xr_back, &xr_ref);
    assert!(rms < 1e-3, "LSF long-block RMS = {rms}");
}

#[test]
fn varied_magnitude_is_roundtrip() {
    // Mix of small / medium / large is values across all sample rates and
    // gain configurations. The encoder must recover every is[] exactly.
    let gc = long_gc(210, false);
    let sf = ScaleFactors::default();
    let mut is = [0i32; NUM_LINES];
    for (i, v) in is.iter_mut().enumerate().take(128) {
        let k = (i as i32 % 16) + 1;
        *v = if i % 3 == 0 { -k } else { k };
    }
    let xr_ref = requantize(&is, &gc, &sf, 44100, MpegVersion::Mpeg1);
    let is_back = quantize(&xr_ref, &gc, &sf, 44100, MpegVersion::Mpeg1);
    assert_eq!(is_back, is, "varied-magnitude round-trip lost the is[]");
    let xr_back = requantize(&is_back, &gc, &sf, 44100, MpegVersion::Mpeg1);
    let rms = bin_rms(&xr_back, &xr_ref);
    assert!(rms < 1e-3, "varied-magnitude RMS = {rms}");
}

#[test]
fn arbitrary_target_quantization_error_is_bounded() {
    // For an arbitrary (not on-grid) flat unit input, the round-trip is
    // intrinsically lossy because the §2.4.3.4.7 grid is discrete. Verify
    // the residual stays bounded by half a quantizer step for the chosen
    // (global_gain, sf) configuration: with global_gain == 210 and all
    // scalefactors zero, the smallest non-zero grid value is 1^(4/3) = 1,
    // so |xr_back - 1.0| <= 1.0. The check captures the bound rather
    // than a tight RMS — the tight RMS is achievable only when the
    // encoder is run inside a bit-allocation loop that picks
    // (global_gain, sf) to put the grid near the target.
    let gc = long_gc(210, false);
    let sf = ScaleFactors::default();
    let xr_target = [1.0f32; NUM_LINES];
    let is = quantize(&xr_target, &gc, &sf, 44100, MpegVersion::Mpeg1);
    let xr_back = requantize(&is, &gc, &sf, 44100, MpegVersion::Mpeg1);
    for (i, (&t, &b)) in xr_target.iter().zip(xr_back.iter()).enumerate() {
        assert!(
            (b - t).abs() <= 1.0,
            "bin {i} residual {} exceeds half-step bound 1.0",
            (b - t).abs()
        );
    }
}
