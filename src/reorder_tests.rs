// Unit tests for the §2.4.3.4.8 short-block reordering stage. Patterns
// are derived from ISO/IEC 11172-3:1993 §2.4.3.4.8 (the reorder
// requirement), §2.4.2.7 (the native Huffman `(sfb, window, freqline)`
// interleave), and the Table B.8 short-block band-start indices. No
// external implementation was consulted.

// This file is `include!`d into the `reorder` module, so its `use` lines
// (MpegVersion, BlockType, GranuleChannel) are already in scope.

/// A short-block (window-switched, non-mixed) granule-channel record.
/// Only the fields the reorder reads (`window_switching_flag`,
/// `block_type`, `mixed_block_flag`) matter; everything else is zeroed.
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

/// A long-block granule-channel record (the pass-through case).
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

#[test]
fn long_block_passes_through_unchanged() {
    // block_type != 2 is already in increasing-frequency order; reorder
    // must be the identity.
    let gc = long_gc();
    let mut xr = [0.0f32; NUM_LINES];
    for (i, v) in xr.iter_mut().enumerate() {
        *v = i as f32;
    }
    let out = reorder(&xr, &gc, 44100, MpegVersion::Mpeg1);
    assert_eq!(out, xr);
}

#[test]
fn band0_44k_three_windows_interleaved() {
    // 44.1k short band 0: per-window start 0, width 4. Native interleave
    // occupies lines 0..12 as [win0:0..4][win1:4..8][win2:8..12]; reorder
    // must rewrite to (freqline, window) order:
    //   out[3*k + win] = in[win*4 + k].
    let gc = short_gc(false);
    let mut xr = [0.0f32; NUM_LINES];
    // Tag each native line so the permutation is unambiguous: encode
    // (window, freqline) as 10*win + freqline + 1.
    for win in 0..3 {
        for k in 0..4 {
            xr[win * 4 + k] = (10 * win + k + 1) as f32;
        }
    }
    let out = reorder(&xr, &gc, 44100, MpegVersion::Mpeg1);
    // After reorder, line 3*k+win must carry the native (win, k) tag.
    for win in 0..3 {
        for k in 0..4 {
            let expected = (10 * win + k + 1) as f32;
            assert_eq!(
                out[3 * k + win],
                expected,
                "freqline {k} window {win} mislanded"
            );
        }
    }
}

#[test]
fn first_subband_is_six_freqlines_by_three_windows() {
    // The first 18 reordered lines must form one polyphase subband: 6
    // frequency lines (per-window freqlines 0..6) × 3 windows, in
    // (freqline, window) order. At 44.1k that spans short bands 0 (start
    // 0, width 4) and 1 (start 4, width 4): per-window freqlines 0..8,
    // but only 0..6 land in subband 0.
    let gc = short_gc(false);
    let mut xr = [0.0f32; NUM_LINES];
    // Band 0 native lines 0..12, band 1 native lines 12..24.
    // Encode native (band, win, k) -> a unique value.
    // Band 0: win*4 + k in 0..12. Band 1: 12 + win*4 + k in 12..24.
    for (i, v) in xr.iter_mut().enumerate().take(24) {
        *v = (i + 1) as f32;
    }
    let out = reorder(&xr, &gc, 44100, MpegVersion::Mpeg1);
    // Subband 0 = out[0..18]. Per-window freqline f in 0..6, window w:
    //   f in 0..4 -> band 0 (start 0): native = w*4 + f.
    //   f in 4..6 -> band 1 (start 4): native = 12 + w*4 + (f-4).
    for f in 0..6usize {
        for w in 0..3usize {
            let native = if f < 4 {
                w * 4 + f
            } else {
                12 + w * 4 + (f - 4)
            };
            assert_eq!(
                out[3 * f + w],
                (native + 1) as f32,
                "subband0 freqline {f} window {w}"
            );
        }
    }
}

#[test]
fn reorder_is_a_permutation_pure_short_44k() {
    // The reorder must be a bijection over the short span 0..576: every
    // input value appears exactly once in the output (no drop, no dup).
    let gc = short_gc(false);
    let mut xr = [0.0f32; NUM_LINES];
    for (i, v) in xr.iter_mut().enumerate() {
        *v = (i + 1) as f32; // distinct non-zero tags
    }
    let out = reorder(&xr, &gc, 44100, MpegVersion::Mpeg1);
    let mut sorted_in: Vec<i32> = xr.iter().map(|&v| v as i32).collect();
    let mut sorted_out: Vec<i32> = out.iter().map(|&v| v as i32).collect();
    sorted_in.sort_unstable();
    sorted_out.sort_unstable();
    assert_eq!(sorted_in, sorted_out, "reorder dropped or duplicated a line");
}

#[test]
fn band_width_change_at_44k() {
    // 44.1k short band 6 has per-window start 30, width 10 (start[7]=40).
    // Native span 3*30=90 .. 3*40=120 laid out as three runs of 10.
    // Verify the (freqline, window) reshuffle for a wider band.
    let gc = short_gc(false);
    let mut xr = [0.0f32; NUM_LINES];
    let base = 3 * 30; // 90
    let w = 10;
    for win in 0..3 {
        for k in 0..w {
            xr[base + win * w + k] = (100 * win + k + 1) as f32;
        }
    }
    let out = reorder(&xr, &gc, 44100, MpegVersion::Mpeg1);
    for win in 0..3 {
        for k in 0..w {
            assert_eq!(
                out[base + 3 * k + win],
                (100 * win + k + 1) as f32,
                "band6 freqline {k} window {win}"
            );
        }
    }
}

#[test]
fn mixed_block_long_region_untouched() {
    // In a mixed block the lowest 36 lines are a long window in
    // increasing-frequency order and must survive the reorder verbatim.
    let gc = short_gc(true);
    let mut xr = [0.0f32; NUM_LINES];
    for (i, v) in xr.iter_mut().enumerate().take(36) {
        *v = (i + 1) as f32;
    }
    let out = reorder(&xr, &gc, 44100, MpegVersion::Mpeg1);
    for (i, &v) in out.iter().enumerate().take(36) {
        assert_eq!(v, (i + 1) as f32, "mixed long line {i} changed");
    }
}

#[test]
fn mixed_block_short_region_reordered_from_band3() {
    // The short region of a mixed block starts at short band 3 (per-window
    // start 12, interleaved line 3*12 = 36). Band 3 at 44.1k: start 12,
    // width 4 (start[4]=16). Native span 36..48 = three runs of 4.
    let gc = short_gc(true);
    let mut xr = [0.0f32; NUM_LINES];
    let base = 3 * 12; // 36
    let w = 4;
    for win in 0..3 {
        for k in 0..w {
            xr[base + win * w + k] = (10 * win + k + 1) as f32;
        }
    }
    let out = reorder(&xr, &gc, 44100, MpegVersion::Mpeg1);
    for win in 0..3 {
        for k in 0..w {
            assert_eq!(
                out[base + 3 * k + win],
                (10 * win + k + 1) as f32,
                "mixed short band3 freqline {k} window {win}"
            );
        }
    }
}

#[test]
fn mixed_block_does_not_reorder_below_band3() {
    // Bands 0..3 in a mixed block belong to the long region. The short
    // reorder must not touch interleaved lines below 36. Seed a value in
    // the would-be native slot of short band 0 (line 0..) and confirm the
    // long pass-through left it where it was rather than reshuffling it.
    let gc = short_gc(true);
    let mut xr = [0.0f32; NUM_LINES];
    // Distinctive marker at line 1 and line 4 (would move if band 0/1 of
    // the short layout were reordered).
    xr[1] = 11.0;
    xr[4] = 44.0;
    let out = reorder(&xr, &gc, 44100, MpegVersion::Mpeg1);
    assert_eq!(out[1], 11.0);
    assert_eq!(out[4], 44.0);
}

#[test]
fn reorder_permutation_48k_and_32k() {
    // The permutation must be a bijection at the other two MPEG-1 band
    // layouts as well (48k and 32k use different short-band widths).
    for sr in [48000u32, 32000] {
        let gc = short_gc(false);
        let mut xr = [0.0f32; NUM_LINES];
        for (i, v) in xr.iter_mut().enumerate() {
            *v = (i + 1) as f32;
        }
        let out = reorder(&xr, &gc, sr, MpegVersion::Mpeg1);
        let mut a: Vec<i32> = xr.iter().map(|&v| v as i32).collect();
        let mut b: Vec<i32> = out.iter().map(|&v| v as i32).collect();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b, "non-bijective reorder at {sr} Hz");
    }
}

#[test]
fn highest_short_band_top_lines_preserved() {
    // 44.1k short bands top out at per-window line 136 (start[12]); 3*136
    // = 408 interleaved lines are short data. Lines 408..576 are never
    // populated by the short coder (they sit above the highest band) and
    // must pass through untouched.
    let gc = short_gc(false);
    let mut xr = [0.0f32; NUM_LINES];
    xr[408] = 7.0;
    xr[575] = 9.0;
    let out = reorder(&xr, &gc, 44100, MpegVersion::Mpeg1);
    assert_eq!(out[408], 7.0);
    assert_eq!(out[575], 9.0);
}

#[test]
fn start_and_end_blocks_pass_through() {
    // block_type 1 (start) and 3 (end) are long-windowed for filterbank
    // purposes and must not be reordered. They are only reached when
    // window_switching_flag is set but block_type != Short.
    for bt in [BlockType::Start, BlockType::End] {
        let mut gc = long_gc();
        gc.window_switching_flag = true;
        gc.block_type = bt;
        let mut xr = [0.0f32; NUM_LINES];
        for (i, v) in xr.iter_mut().enumerate() {
            *v = i as f32;
        }
        let out = reorder(&xr, &gc, 44100, MpegVersion::Mpeg1);
        assert_eq!(out, xr, "{bt:?} block was reordered");
    }
}
