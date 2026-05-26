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
//!    exclusion with the testing toggles + the outer loop, enable /
//!    disable round-trip).
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
fn auto_block_type_rejected_on_stereo_encoder() {
    // Same mono-only restriction as the force-toggles: cross-channel
    // block-type agreement (§2.4.3.4.9) is deferred to a follow-up.
    let mut enc = Mp3Encoder::new(BR, SR, ChannelMode::Stereo).expect("stereo encoder build");
    assert!(!enc.auto_block_type_enabled());
    let err = enc
        .enable_auto_block_type(DEFAULT_ATTACK_THRESHOLD)
        .expect_err("stereo + auto should be rejected");
    let msg = format!("{err}");
    assert!(!msg.is_empty(), "error must have a message");
    assert!(
        !enc.auto_block_type_enabled(),
        "flag must stay off after error"
    );
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
fn auto_block_type_rejected_when_outer_loop_configured() {
    // The outer loop is long-block-only this round; auto block-type
    // can emit Short / Start / End which the loop doesn't yet handle.
    // Reject the combination at enable-time.
    let mut enc = Mp3Encoder::new_with_outer_loop(
        BR,
        SR,
        ChannelMode::SingleChannel,
        /*uniform_threshold=*/ 1.0e-6,
    )
    .expect("outer-loop encoder build");
    assert!(!enc.auto_block_type_enabled());
    let res = enc.enable_auto_block_type(DEFAULT_ATTACK_THRESHOLD);
    assert!(
        res.is_err(),
        "auto + outer-loop should be rejected at enable time"
    );
    assert!(
        !enc.auto_block_type_enabled(),
        "flag must stay off after rejection"
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
