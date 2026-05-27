//! Integration test for the Phase 2 step-26 **signal-driven
//! auto block-type** encode path
//! (round 156 — [`oxideav_mp3::Mp3Encoder::enable_auto_block_type`]).
//!
//! Validates the end-to-end §C.1.5.2 LONG → START → SHORT → STOP → LONG
//! transition state machine on the encode side: with the auto toggle
//! enabled the encoder runs the [`oxideav_mp3::AttackDetector`] +
//! [`oxideav_mp3::BlockTypeStateMachine`] pair per granule, chooses
//! the appropriate window family + MDCT path, and writes a side-info
//! record whose `window_switching_flag` / `block_type` reflect the
//! scheduler's decision per granule.
//!
//! The test:
//!
//! 1. Sanity-checks the API surface (mono-only restriction, mutual
//!    exclusion with the testing toggles, enable / disable round-trip).
//!    As of r158 the outer-loop combination is *accepted* (Short
//!    granules dispatch onto `outer_loop_search_short`); as of r160
//!    Start/End granules ALSO route into the outer loop via the
//!    long-family primitive `outer_loop_search_long` (previously they
//!    fell back to the fixed-gain inner-loop path). Every block-type
//!    the auto scheduler emits now runs the §C.1.5.4.3 distortion
//!    control loop.
//! 2. Confirms that auto-block-type **default off** is the identity
//!    transform: every granule of a sustained-tone stream still
//!    emits a long block.
//! 3. Confirms that with auto on AND a sustained sine input (no
//!    transients), the scheduler emits long blocks throughout (the
//!    detector correctly classifies the steady signal as
//!    non-attack).
//! 4. Confirms that with auto on AND a "click + silence + click"
//!    transient PCM, the scheduler emits at least one
//!    Start → Short pair somewhere in the output (a witness that
//!    the §C.1.5.2 transition geometry has been engaged).
//! 5. Confirms that the auto-encoded stream is structurally valid
//!    (frames parse, side-info round-trips, demuxer accepts it).
//!
//! The test uses ONLY this crate; no external library is invoked.

use std::io::Cursor;

use oxideav_core::Demuxer;
use oxideav_mp3::{
    parse_header, parse_side_info, BlockType, ChannelMode, FrameWalker, Mp3Demuxer, Mp3Encoder,
    DEFAULT_ATTACK_THRESHOLD,
};

const SR: u32 = 44_100;
const BR: u32 = 192;

/// Synthesise an `n`-sample mono `i16` sine tone of `freq_hz` at
/// `sample_rate_hz`, peak amplitude `amp` ∈ `[0, 1]`.
fn sine_pcm(n: usize, freq_hz: f32, sample_rate_hz: f32, amp: f32) -> Vec<i16> {
    use std::f32::consts::PI;
    let two_pi = 2.0 * PI;
    let mut out = Vec::with_capacity(n);
    let scale = amp * (i16::MAX as f32);
    for i in 0..n {
        let t = i as f32 / sample_rate_hz;
        let s = (two_pi * freq_hz * t).sin() * scale;
        out.push(s.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16);
    }
    out
}

/// Synthesise a "click train" stimulus: alternating runs of silence
/// and full-scale impulses. The bursts are chosen so the
/// AttackDetector's running ambient stays low (silence between
/// clicks) so each click reliably trips the detector.
fn click_train_pcm(total_samples: usize, click_period: usize) -> Vec<i16> {
    let mut out = vec![0i16; total_samples];
    let mut pos = click_period;
    while pos + 64 < total_samples {
        // A 64-sample full-scale rectangular burst followed by silence.
        for j in 0..64 {
            out[pos + j] = if j % 2 == 0 { 30_000 } else { -30_000 };
        }
        pos += click_period;
    }
    out
}

#[test]
fn auto_block_type_rejected_on_ms_stereo_encoder() {
    // r162: the rejection is now scoped to MS-stereo joint modes
    // (§2.4.3.4.9 requires both channels of an MS-stereo granule to
    // share `block_type`; the cross-channel-MS agreement wiring is
    // deferred). Independent stereo is accepted in the
    // `auto_block_type_accepted_on_independent_stereo` test below.
    let mut enc = Mp3Encoder::new_joint_stereo_ms(BR, SR).expect("MS-stereo encoder build");
    assert!(!enc.auto_block_type_enabled());
    let err = enc
        .enable_auto_block_type(DEFAULT_ATTACK_THRESHOLD)
        .expect_err("MS-stereo + auto should be rejected");
    let msg = format!("{err}");
    assert!(!msg.is_empty(), "error must have a message");
    assert!(
        !enc.auto_block_type_enabled(),
        "flag must stay off after error"
    );
}

#[test]
fn auto_block_type_rejected_on_ms_auto_stereo_encoder() {
    let mut enc = Mp3Encoder::new_joint_stereo_auto(BR, SR).expect("MS-auto encoder build");
    assert!(!enc.auto_block_type_enabled());
    let err = enc
        .enable_auto_block_type(DEFAULT_ATTACK_THRESHOLD)
        .expect_err("MS-auto + auto should be rejected");
    assert!(!format!("{err}").is_empty(), "error must have a message");
    assert!(!enc.auto_block_type_enabled(), "flag must stay off");
}

#[test]
fn auto_block_type_accepted_on_independent_stereo() {
    // r162: independent stereo (no MS coupling) accepts the auto
    // toggle. Each channel runs its own detector + scheduler so the
    // per-channel side-info carries an independent §C.1.5.2 transition
    // sequence.
    let mut enc = Mp3Encoder::new(BR, SR, ChannelMode::Stereo).expect("stereo encoder build");
    enc.enable_auto_block_type(DEFAULT_ATTACK_THRESHOLD)
        .expect("auto accepted on independent stereo");
    assert!(enc.auto_block_type_enabled());
    assert_eq!(
        enc.auto_block_type_threshold(),
        Some(DEFAULT_ATTACK_THRESHOLD)
    );

    let mut enc2 =
        Mp3Encoder::new(BR, SR, ChannelMode::DualChannel).expect("dual-channel encoder build");
    enc2.enable_auto_block_type(DEFAULT_ATTACK_THRESHOLD)
        .expect("auto accepted on dual-channel");
    assert!(enc2.auto_block_type_enabled());
}

#[test]
fn auto_block_type_default_off() {
    // Fresh encoder: auto should be off, threshold None, force-toggles
    // off.
    let enc = Mp3Encoder::new(BR, SR, ChannelMode::SingleChannel).expect("mono encoder build");
    assert!(!enc.auto_block_type_enabled());
    assert_eq!(enc.auto_block_type_threshold(), None);
    assert!(!enc.force_short_blocks_enabled());
    assert!(!enc.force_mixed_blocks_enabled());
}

#[test]
fn auto_block_type_enable_disable_round_trip() {
    let mut enc = Mp3Encoder::new(BR, SR, ChannelMode::SingleChannel).expect("mono encoder build");
    enc.enable_auto_block_type(DEFAULT_ATTACK_THRESHOLD)
        .expect("enable auto");
    assert!(enc.auto_block_type_enabled());
    assert_eq!(
        enc.auto_block_type_threshold(),
        Some(DEFAULT_ATTACK_THRESHOLD)
    );
    enc.disable_auto_block_type();
    assert!(!enc.auto_block_type_enabled());
    assert_eq!(enc.auto_block_type_threshold(), None);
}

#[test]
fn auto_block_type_mutually_exclusive_with_force_short() {
    let mut enc = Mp3Encoder::new(BR, SR, ChannelMode::SingleChannel).expect("mono encoder build");
    enc.enable_auto_block_type(DEFAULT_ATTACK_THRESHOLD)
        .expect("enable auto");
    assert!(enc.auto_block_type_enabled());
    // Enabling force-short clears auto.
    enc.force_short_blocks_for_testing(true)
        .expect("force_short toggle");
    assert!(enc.force_short_blocks_enabled());
    assert!(
        !enc.auto_block_type_enabled(),
        "auto must clear when force-short turns on"
    );
}

#[test]
fn auto_block_type_mutually_exclusive_with_force_mixed() {
    let mut enc = Mp3Encoder::new(BR, SR, ChannelMode::SingleChannel).expect("mono encoder build");
    enc.enable_auto_block_type(DEFAULT_ATTACK_THRESHOLD)
        .expect("enable auto");
    assert!(enc.auto_block_type_enabled());
    enc.force_mixed_blocks_for_testing(true)
        .expect("force_mixed toggle");
    assert!(enc.force_mixed_blocks_enabled());
    assert!(
        !enc.auto_block_type_enabled(),
        "auto must clear when force-mixed turns on"
    );
}

#[test]
fn auto_block_type_combines_with_outer_loop_and_roundtrips() {
    // r158 unblocked: `outer_loop_search_short` is now wired into the
    // encoder so the §C.1.5.4.3 distortion-control loop can run on the
    // auto-block-type Short granules. r160 follow-up: Start/End
    // transition skeletons now also route into the outer loop via the
    // long-family primitive `outer_loop_search_long` (no more fixed-gain
    // fallback for any block-type the auto scheduler emits).
    //
    // The combined configuration must:
    //   * accept `enable_auto_block_type` on top of `new_with_outer_loop`
    //     (the r156 rejection is removed);
    //   * produce a structurally valid MP3 stream on a click-train
    //     stimulus that engages the Short geometry;
    //   * keep the §2.4.2.7 invariants on every short granule
    //     (`preflag == false`; transmitted `subblock_gain` values fit the
    //     3-bit field).
    let mut enc = Mp3Encoder::new_with_outer_loop(
        BR,
        SR,
        ChannelMode::SingleChannel,
        /*uniform_threshold=*/ 1.0e-6,
    )
    .expect("outer-loop encoder build");
    assert!(!enc.auto_block_type_enabled());
    enc.enable_auto_block_type(DEFAULT_ATTACK_THRESHOLD)
        .expect("auto + outer-loop combination now accepted");
    assert!(
        enc.auto_block_type_enabled(),
        "flag must be set after enable"
    );

    // Drive a click-train PCM that exercises the auto scheduler's
    // LONG → START → SHORT → END → LONG sequence (so the outer-loop
    // dispatch visits all four block-type cases at least once).
    let n = SR as usize;
    let pcm = click_train_pcm(n, /*click_period=*/ 6600);
    enc.push_samples(&pcm).expect("push pcm");
    let mut bytes = Vec::new();
    let _ = enc.finish(&mut bytes).expect("encoder finish");

    let mut frames = 0usize;
    let mut saw_short = false;
    for frame in FrameWalker::new(&bytes) {
        frames += 1;
        let hdr = parse_header(&frame.data[..4]).expect("header");
        let si = parse_side_info(&hdr, &frame.data[4..]).expect("side_info");
        for gr in 0..si.granule_count as usize {
            for ch in 0..si.channels as usize {
                let gc = &si.granules[gr][ch];
                // §2.4.2.7 invariant: short blocks never set preflag.
                if gc.window_switching_flag && gc.block_type == BlockType::Short {
                    assert!(
                        !gc.preflag,
                        "short block emitted with preflag (violates §2.4.2.7)"
                    );
                    saw_short = true;
                }
                // 3-bit field check on subblock_gain.
                for &sg in &gc.subblock_gain {
                    assert!(sg <= 7, "subblock_gain {sg} exceeds 3-bit field");
                }
            }
        }
    }
    assert!(frames > 0, "no frames emitted in auto + outer-loop stream");
    assert!(
        saw_short,
        "click-train auto + outer-loop did not engage any Short granule \
         (outer-loop dispatch on Short would not have been exercised)"
    );

    // Demuxer must still accept the resulting byte stream end-to-end.
    let cursor = Cursor::new(bytes);
    let mut demux = Mp3Demuxer::open(Box::new(cursor)).expect("demuxer open");
    let mut packets = 0usize;
    loop {
        match demux.next_packet() {
            Ok(_pkt) => {
                packets += 1;
                if packets > 2_000 {
                    break;
                }
            }
            Err(oxideav_core::Error::Eof) => break,
            Err(e) => panic!("demux next_packet: {e}"),
        }
    }
    assert!(
        packets > 0,
        "demuxer accepted no packets from auto + outer-loop stream"
    );
}

#[test]
fn auto_block_type_on_pure_sine_stays_long() {
    // Steady-state tone: detector should never fire, scheduler
    // emits Long throughout.
    let n = SR as usize / 2; // 500 ms
    let pcm = sine_pcm(n, 440.0, SR as f32, 0.5);
    let mut enc = Mp3Encoder::new(BR, SR, ChannelMode::SingleChannel).expect("mono encoder build");
    enc.enable_auto_block_type(DEFAULT_ATTACK_THRESHOLD)
        .expect("enable auto");
    enc.push_samples(&pcm).expect("push pcm");
    let mut bytes = Vec::new();
    let _ = enc.finish(&mut bytes).expect("encoder finish");

    let mut frames = 0usize;
    let mut long_only = true;
    let mut any_gc = false;
    for frame in FrameWalker::new(&bytes) {
        frames += 1;
        let hdr = parse_header(&frame.data[..4]).expect("header");
        let si = parse_side_info(&hdr, &frame.data[4..]).expect("side_info");
        for gr in 0..si.granule_count as usize {
            for ch in 0..si.channels as usize {
                any_gc = true;
                let gc = &si.granules[gr][ch];
                if gc.window_switching_flag || gc.block_type != BlockType::Long {
                    long_only = false;
                }
            }
        }
    }
    assert!(frames > 0, "no frames emitted in pure-sine auto stream");
    assert!(any_gc, "no granule-channels visited");
    assert!(
        long_only,
        "auto on a pure sine emitted a non-long block (detector over-fired)"
    );
}

#[allow(clippy::needless_range_loop)]
#[test]
fn auto_block_type_on_transient_burst_engages_start_short_sequence() {
    // Click-train stimulus: the detector should fire on each burst,
    // the scheduler should emit at least one Start → Short pair.
    //
    // Period is chosen so successive bursts are separated by several
    // granules of silence (so the ambient resets between clicks and
    // each click reliably trips the threshold).
    let n = SR as usize; // 1 s of audio
                         // Click every ~150 ms (~6600 samples, well over one granule).
    let pcm = click_train_pcm(n, /*click_period=*/ 6600);
    let mut enc = Mp3Encoder::new(BR, SR, ChannelMode::SingleChannel).expect("mono encoder build");
    enc.enable_auto_block_type(DEFAULT_ATTACK_THRESHOLD)
        .expect("enable auto");
    enc.push_samples(&pcm).expect("push pcm");
    let mut bytes = Vec::new();
    let _ = enc.finish(&mut bytes).expect("encoder finish");

    let mut frames = 0usize;
    let mut block_type_counts = [0usize; 4]; // Long, Start, Short, End
    let mut saw_start = false;
    let mut saw_short = false;
    let mut saw_end = false;
    // Track per-channel previous block-type so we can verify
    // geometry of the emitted sequence.
    let mut prev_bt: Vec<Option<BlockType>> = vec![None; 1];
    for frame in FrameWalker::new(&bytes) {
        frames += 1;
        let hdr = parse_header(&frame.data[..4]).expect("header");
        let si = parse_side_info(&hdr, &frame.data[4..]).expect("side_info");
        for gr in 0..si.granule_count as usize {
            for ch in 0..si.channels as usize {
                let gc = &si.granules[gr][ch];
                let bt = gc.block_type;
                match bt {
                    BlockType::Long => block_type_counts[0] += 1,
                    BlockType::Start => {
                        block_type_counts[1] += 1;
                        saw_start = true;
                    }
                    BlockType::Short => {
                        block_type_counts[2] += 1;
                        saw_short = true;
                    }
                    BlockType::End => {
                        block_type_counts[3] += 1;
                        saw_end = true;
                    }
                }
                // §C.1.5.2 transition geometry checks: invalid
                // transitions should never appear on the wire.
                if let Some(prev) = prev_bt[ch] {
                    let valid = match (prev, bt) {
                        (BlockType::Long, BlockType::Long) => true,
                        (BlockType::Long, BlockType::Start) => true,
                        (BlockType::Start, BlockType::Short) => true,
                        (BlockType::Short, BlockType::Short) => true,
                        (BlockType::Short, BlockType::End) => true,
                        (BlockType::End, BlockType::Long) => true,
                        // The only Start→Short pair is from a granule
                        // freshly entering the burst window; cycling
                        // back through other pairs is invalid.
                        _ => false,
                    };
                    assert!(valid, "invalid §C.1.5.2 transition {:?} → {:?}", prev, bt);
                }
                prev_bt[ch] = Some(bt);
            }
        }
    }
    assert!(
        frames > 0,
        "no frames emitted in transient-burst auto stream"
    );
    // The transient input should have triggered the burst geometry at
    // least once: Start, Short, End all present.
    assert!(
        saw_start,
        "no Start block emitted for click-train input; \
         block-type counts (Long, Start, Short, End): {:?}",
        block_type_counts
    );
    assert!(
        saw_short,
        "no Short block emitted for click-train input; counts: {:?}",
        block_type_counts
    );
    assert!(
        saw_end,
        "no End (Stop) block emitted for click-train input; counts: {:?}",
        block_type_counts
    );
}

#[test]
fn auto_block_type_stream_is_demuxer_accepted() {
    // The stream the auto path produces must remain a structurally
    // valid MP3 — the demuxer's frame walker should accept it and
    // emit one Packet per audio frame.
    let n = SR as usize / 2;
    let pcm = click_train_pcm(n, /*click_period=*/ 6600);
    let mut enc = Mp3Encoder::new(BR, SR, ChannelMode::SingleChannel).expect("mono encoder build");
    enc.enable_auto_block_type(DEFAULT_ATTACK_THRESHOLD)
        .expect("enable auto");
    enc.push_samples(&pcm).expect("push pcm");
    let mut bytes = Vec::new();
    let _ = enc.finish(&mut bytes).expect("encoder finish");

    let cursor = Cursor::new(bytes);
    let mut demux = Mp3Demuxer::open(Box::new(cursor)).expect("demuxer open");
    // Walk every emitted packet; we just need confirmation that the
    // stream parses without surprises.
    let mut packets = 0usize;
    loop {
        match demux.next_packet() {
            Ok(_pkt) => {
                packets += 1;
                if packets > 2_000 {
                    break;
                }
            }
            Err(oxideav_core::Error::Eof) => break,
            Err(e) => panic!("demux next_packet: {e}"),
        }
    }
    assert!(packets > 0, "demuxer accepted no packets from auto stream");
}

#[test]
fn auto_block_type_on_independent_stereo_runs_per_channel_scheduler() {
    // r162: on independent stereo every channel runs an independent
    // detector + scheduler. Drive the left channel with a click train
    // and the right channel with a sustained sine — the encoder must:
    //   1. emit valid frames (demuxer round-trip),
    //   2. emit at least one Start AND/OR Short granule somewhere in
    //      the left-channel column (witness the click-driven left
    //      detector engaged the §C.1.5.2 transition),
    //   3. emit Long throughout the right-channel column (witness the
    //      right detector saw no attack).
    // The second + third assertions together are what makes this an
    // *independent-per-channel* witness rather than a generic "auto
    // works on stereo" smoke test.
    use std::f32::consts::PI;
    let n = SR as usize;
    let click_period = 6_600usize;
    let mut pcm: Vec<i16> = Vec::with_capacity(n * 2);
    let scale = 0.5 * (i16::MAX as f32);
    // Build the click pattern for left, sine for right, interleaved.
    let mut next_click_pos = click_period;
    for i in 0..n {
        // Left channel: silence except for a 64-sample full-scale
        // alternating burst at click_period intervals.
        let l_burst = if i >= next_click_pos && i < next_click_pos + 64 {
            let s = i - next_click_pos;
            let v = if s % 2 == 0 { 30_000i16 } else { -30_000i16 };
            if s + 1 == 64 {
                next_click_pos += click_period;
            }
            v
        } else {
            0i16
        };
        // Right channel: sustained 440 Hz sine.
        let t = i as f32 / SR as f32;
        let r = (2.0 * PI * 440.0 * t).sin() * scale;
        pcm.push(l_burst);
        pcm.push(r.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16);
    }

    let mut enc = Mp3Encoder::new(BR, SR, ChannelMode::Stereo).expect("stereo encoder build");
    enc.enable_auto_block_type(DEFAULT_ATTACK_THRESHOLD)
        .expect("auto accepted on independent stereo");
    enc.push_samples(&pcm).expect("push stereo pcm");
    let mut bytes: Vec<u8> = Vec::new();
    let _ = enc.finish(&mut bytes).expect("encoder finish");

    let mut frames = 0usize;
    let mut left_non_long = 0usize;
    let mut right_long = 0usize;
    let mut right_total = 0usize;
    for frame in FrameWalker::new(&bytes) {
        frames += 1;
        let hdr = parse_header(&frame.data[..4]).expect("header parse");
        assert_eq!(hdr.channel_count(), 2);
        let si = parse_side_info(&hdr, &frame.data[4..]).expect("side_info parse");
        for gr in 0..si.granule_count as usize {
            // Channel 0 (left): driven by click train, expect Start / Short.
            let left = &si.granules[gr][0];
            if left.window_switching_flag && left.block_type != BlockType::Long {
                left_non_long += 1;
            }
            // Channel 1 (right): driven by sustained sine, expect Long.
            let right = &si.granules[gr][1];
            right_total += 1;
            if !right.window_switching_flag && right.block_type == BlockType::Long {
                right_long += 1;
            }
        }
    }
    assert!(frames > 0, "stereo auto emitted no frames");
    assert!(
        left_non_long > 0,
        "left-channel click train never engaged a non-Long granule \
         (per-channel scheduler may not be hooked up)"
    );
    assert!(
        right_total > 0 && right_long == right_total,
        "right-channel sine should stay Long throughout (was {right_long}/{right_total})"
    );

    let mut demux =
        Mp3Demuxer::open(Box::new(Cursor::new(bytes.clone()))).expect("stereo demuxer open");
    let mut packets = 0usize;
    loop {
        match demux.next_packet() {
            Ok(_pkt) => packets += 1,
            Err(oxideav_core::Error::Eof) => break,
            Err(e) => panic!("stereo demuxer next_packet: {e}"),
        }
    }
    assert!(packets > 0, "stereo auto demuxer accepted no packets");
}

#[test]
fn auto_block_type_outer_loop_dispatches_start_end_through_long_primitive() {
    // r160 wiring: with both auto-block-type AND the outer loop
    // enabled, the Start / End transition skeletons no longer fall
    // back to the fixed-gain inner-loop-only path — they route into
    // `outer_loop_search_long` (the long-family primitive). The
    // observable on the wire is `scalefac_compress = 15` (the
    // `OUTER_LOOP_SCALEFAC_COMPRESS` constant): the fixed-gain path
    // writes `scalefac_compress = 0` (zero scalefactors), and any
    // outer-loop path seeds 15 so the chosen per-band scalefactors
    // can be transmitted at slen1 = 4 / slen2 = 3. This is the same
    // distinguishability signature used by the r159 mixed-block
    // wiring test in `mixed_block_encoder_roundtrip.rs`.
    let mut enc = Mp3Encoder::new_with_outer_loop(
        BR,
        SR,
        ChannelMode::SingleChannel,
        /*uniform_threshold=*/ 1.0e-6,
    )
    .expect("outer-loop encoder build");
    enc.enable_auto_block_type(DEFAULT_ATTACK_THRESHOLD)
        .expect("auto + outer-loop accepted (r158)");

    // Click-train PCM exercises the LONG → START → SHORT → END → LONG
    // sequence so the dispatcher visits Start and End at least once.
    let n = SR as usize;
    let pcm = click_train_pcm(n, /*click_period=*/ 6600);
    enc.push_samples(&pcm).expect("push pcm");
    let mut bytes = Vec::new();
    let _ = enc.finish(&mut bytes).expect("encoder finish");

    let mut start_granules = 0usize;
    let mut end_granules = 0usize;
    let mut start_with_outer_loop = 0usize;
    let mut end_with_outer_loop = 0usize;
    for frame in FrameWalker::new(&bytes) {
        let hdr = parse_header(&frame.data[..4]).expect("header");
        let si = parse_side_info(&hdr, &frame.data[4..]).expect("side_info");
        for gr in 0..si.granule_count as usize {
            for ch in 0..si.channels as usize {
                let gc = &si.granules[gr][ch];
                match gc.block_type {
                    BlockType::Start => {
                        assert!(
                            gc.window_switching_flag,
                            "Start block must carry window_switching_flag",
                        );
                        start_granules += 1;
                        if gc.scalefac_compress == 15 {
                            start_with_outer_loop += 1;
                        }
                    }
                    BlockType::End => {
                        assert!(
                            gc.window_switching_flag,
                            "End block must carry window_switching_flag",
                        );
                        end_granules += 1;
                        if gc.scalefac_compress == 15 {
                            end_with_outer_loop += 1;
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    assert!(
        start_granules > 0,
        "click-train auto+outer-loop did not engage Start at all"
    );
    assert!(
        end_granules > 0,
        "click-train auto+outer-loop did not engage End at all"
    );
    // The dispatch contract: every Start / End granule must run the
    // outer loop (no fixed-gain fallback this round).
    assert_eq!(
        start_with_outer_loop,
        start_granules,
        "{}/{} Start granules failed the outer-loop wire signature \
         (scalefac_compress == 15); the fixed-gain fallback path \
         was unexpectedly taken",
        start_granules - start_with_outer_loop,
        start_granules,
    );
    assert_eq!(
        end_with_outer_loop,
        end_granules,
        "{}/{} End granules failed the outer-loop wire signature \
         (scalefac_compress == 15); the fixed-gain fallback path \
         was unexpectedly taken",
        end_granules - end_with_outer_loop,
        end_granules,
    );
}

#[test]
fn auto_block_type_outer_loop_start_end_stream_is_demuxer_accepted() {
    // End-to-end roundtrip: with Start / End now running through the
    // outer loop, the assembled bytestream must remain a structurally
    // valid MP3 (the demuxer reads every frame back without surprise).
    // This guards against a hypothetical regression where the
    // long-family primitive's chosen scalefactors don't round-trip
    // through the part2 writer / scalefactor reader.
    let mut enc = Mp3Encoder::new_with_outer_loop(
        BR,
        SR,
        ChannelMode::SingleChannel,
        /*uniform_threshold=*/ 1.0e-6,
    )
    .expect("outer-loop encoder build");
    enc.enable_auto_block_type(DEFAULT_ATTACK_THRESHOLD)
        .expect("auto + outer-loop accepted (r158)");

    let n = SR as usize;
    let pcm = click_train_pcm(n, /*click_period=*/ 6600);
    enc.push_samples(&pcm).expect("push pcm");
    let mut bytes = Vec::new();
    let _ = enc.finish(&mut bytes).expect("encoder finish");

    let cursor = Cursor::new(bytes);
    let mut demux = Mp3Demuxer::open(Box::new(cursor)).expect("demuxer open");
    let mut packets = 0usize;
    loop {
        match demux.next_packet() {
            Ok(_pkt) => {
                packets += 1;
                if packets > 4_000 {
                    break;
                }
            }
            Err(oxideav_core::Error::Eof) => break,
            Err(e) => panic!("demux next_packet: {e}"),
        }
    }
    assert!(
        packets > 0,
        "demuxer accepted no packets from r160 transition-skeleton outer-loop stream",
    );
}
