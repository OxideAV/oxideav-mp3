//! Integration test for the Phase 2 step-22 forward short-block encode
//! path (round 151 — `Mp3Encoder::force_short_blocks_for_testing`).
//!
//! Validates the end-to-end §2.4.3.4.10.2 IMDCT short-block path on the
//! encode side: the encoder runs three independent 12-point MDCTs per
//! polyphase subband (via
//! [`oxideav_mp3::short_block::forward_short_mdct_subband`]), lays the
//! bins out in the bitstream-native `[sfb][win][k]` interleave (via
//! [`oxideav_mp3::short_block::forward_reorder`]), and writes a side-info
//! record with `window_switching_flag = 1`, `block_type = 2`,
//! `mixed_block_flag = 0`, and the spec-default short-block region
//! sentinels (per
//! [`oxideav_mp3::short_block::short_block_region_defaults`]).
//!
//! The test:
//!
//! 1. Synthesises one second of a 440 Hz sine tone (mono, 44.1 kHz,
//!    `i16` PCM).
//! 2. Encodes the same input two ways: with `force_short_blocks` off
//!    (the long-block baseline; every previous round) and on (the new
//!    short-block forward path).
//! 3. Parses both streams and asserts every emitted granule-channel
//!    carries the expected `window_switching_flag` / `block_type` for
//!    its branch — confirming the side-info wiring is honoured on the
//!    wire.
//! 4. Decodes the short-block stream through the crate's own
//!    primitives (`huffman → requantize → reorder → alias → imdct →
//!    synth`) and confirms the recovered PCM has finite energy and is
//!    not pure silence (a smoke-level witness that the chain
//!    end-to-end works; rigorous PSNR comparison against the long
//!    baseline is deferred to the follow-up round that adds the
//!    block-type auto-decision heuristic).
//!
//! The test uses ONLY this crate; no external library is invoked.

use std::io::Cursor;

use oxideav_core::Demuxer;
use oxideav_mp3::{
    alias_reduce, decode_huffman, decode_scalefactors, imdct_granule, parse_header,
    parse_side_info, reorder, requantize, synth_granule, BlockType, ChannelMode, FrameWalker,
    ImdctState, MainDataReader, Mp3Demuxer, Mp3Encoder, Reservoir, SynthState, PCM_PER_GRANULE,
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

fn encode(force_short: bool) -> Vec<u8> {
    let n = SR as usize / 4; // 250 ms is plenty for several frames
    let pcm = sine_pcm(n, 440.0, SR as f32, 0.5);
    let mut enc = Mp3Encoder::new(BR, SR, ChannelMode::SingleChannel).expect("encoder build");
    enc.force_short_blocks_for_testing(force_short)
        .expect("force_short toggle");
    enc.push_samples(&pcm).expect("push pcm");
    let mut out: Vec<u8> = Vec::new();
    let _bytes = enc.finish(&mut out).expect("encoder finish");
    out
}

#[test]
fn force_short_blocks_rejected_on_stereo_encoder() {
    // The toggle is intentionally mono-only this round; multi-channel
    // short-block encode needs the §2.4.3.4.9 cross-channel
    // block-type-agreement wiring deferred to a follow-up.
    let mut enc = Mp3Encoder::new(BR, SR, ChannelMode::Stereo).expect("stereo encoder build");
    assert!(!enc.force_short_blocks_enabled());
    let err = enc
        .force_short_blocks_for_testing(true)
        .expect_err("stereo + force-short should be rejected");
    let msg = format!("{err}");
    assert!(
        msg.contains("Mp3Encoder::new_joint_stereo_ms")
            || msg.contains("Stereo")
            || msg.contains("JointStereo"),
        "unexpected error message: {msg}"
    );
    assert!(!enc.force_short_blocks_enabled(), "flag must stay off");
}

#[test]
fn long_block_baseline_carries_only_long_granules() {
    // Default encoder (no force-short): every granule must be coded as
    // a long block with window_switching = 0.
    let bytes = encode(/*force_short=*/ false);
    let mut frames = 0usize;
    let mut long_only = true;
    for frame in FrameWalker::new(&bytes) {
        frames += 1;
        let hdr = parse_header(&frame.data[..4]).expect("header");
        let si = parse_side_info(&hdr, &frame.data[4..]).expect("side_info");
        for gr in 0..si.granule_count as usize {
            for ch in 0..si.channels as usize {
                let gc = &si.granules[gr][ch];
                if gc.window_switching_flag || gc.block_type != BlockType::Long {
                    long_only = false;
                }
            }
        }
    }
    assert!(frames > 0, "no frames emitted in long baseline");
    assert!(long_only, "long-block baseline carried a non-long granule");
}

#[test]
fn force_short_blocks_writes_short_block_side_info() {
    // With force-short on, every granule-channel's side-info must
    // carry window_switching_flag = 1, block_type = Short,
    // mixed_block_flag = 0.
    let bytes = encode(/*force_short=*/ true);
    let mut frames = 0usize;
    let mut short_only = true;
    let mut any = false;
    for frame in FrameWalker::new(&bytes) {
        frames += 1;
        let hdr = parse_header(&frame.data[..4]).expect("header");
        let si = parse_side_info(&hdr, &frame.data[4..]).expect("side_info");
        for gr in 0..si.granule_count as usize {
            for ch in 0..si.channels as usize {
                any = true;
                let gc = &si.granules[gr][ch];
                if !gc.window_switching_flag
                    || gc.block_type != BlockType::Short
                    || gc.mixed_block_flag
                {
                    short_only = false;
                }
                // subblock_gain defaults to all-zero in this round.
                for &g in &gc.subblock_gain {
                    assert_eq!(g, 0, "subblock_gain default should be 0");
                }
            }
        }
    }
    assert!(frames > 0, "force-short emitted no frames");
    assert!(any, "force-short emitted frames but no granules?!");
    assert!(short_only, "force-short stream carried a non-short granule");
}

#[test]
fn force_short_stream_decodes_to_finite_non_silent_pcm() {
    // The force-short stream must round-trip through the crate's own
    // decode primitives, producing PCM that is finite and not silent.
    // (A rigorous PSNR comparison versus the long baseline would
    // require the §C.1.5 attack-detection heuristic the encoder does
    // not yet expose; that comparison is deferred to the round that
    // lands the auto-decision logic.)
    let bytes = encode(/*force_short=*/ true);
    let recon = decode_mp3_mono_short_aware(&bytes);
    assert!(!recon.is_empty(), "decoded PCM was empty");
    let energy: f64 = recon
        .iter()
        .map(|&v| f64::from(v) * f64::from(v))
        .sum::<f64>()
        / recon.len() as f64;
    assert!(
        energy.is_finite() && energy > 0.0,
        "decoded PCM had zero or non-finite energy ({energy})"
    );
    // The decoded waveform should also be bounded — a runaway IMDCT
    // would saturate the `i16` clamp on every sample. We expect at
    // least *some* near-zero crossings (a pure-sine reconstruction
    // crosses zero at twice the tone frequency).
    let mut zero_crossings = 0usize;
    for w in recon.windows(2) {
        if (w[0] >= 0) != (w[1] >= 0) {
            zero_crossings += 1;
        }
    }
    assert!(
        zero_crossings > 10,
        "no audible zero crossings: {zero_crossings} (decode chain probably broken)"
    );
}

#[test]
fn force_short_stream_passes_demuxer() {
    // The Mp3Demuxer (which parses headers + side-info on every frame)
    // must accept every force-short frame.
    let bytes = encode(/*force_short=*/ true);
    let mut demux = Mp3Demuxer::open(Box::new(Cursor::new(bytes.clone()))).expect("demuxer open");
    let mut frame_count = 0usize;
    loop {
        match demux.next_packet() {
            Ok(_pkt) => frame_count += 1,
            Err(oxideav_core::Error::Eof) => break,
            Err(e) => panic!("demuxer next_packet: {e}"),
        }
    }
    assert!(
        frame_count > 0,
        "demuxer surfaced zero frames on force-short stream"
    );
}

/// Decode an MP3 byte stream into mono `i16` PCM using the crate's own
/// primitives, with the §2.4.3.4.8 reorder stage in the pipeline for
/// short-block granules. Mirror of the helper in
/// `tests/stream_encoder_roundtrip.rs::decode_mp3_mono` extended to
/// call `reorder` before `alias_reduce` (the long-only baseline test
/// skipped reorder because long blocks pass through unchanged).
fn decode_mp3_mono_short_aware(bytes: &[u8]) -> Vec<i16> {
    let mut reservoir = Reservoir::new();
    let mut synth_state = SynthState::new();
    let mut imdct_state = ImdctState::new();
    let mut out_pcm: Vec<i16> = Vec::new();

    for frame in FrameWalker::new(bytes) {
        let hdr = parse_header(&frame.data[..4]).expect("header parse");
        let si_start = 4;
        let si = parse_side_info(&hdr, &frame.data[si_start..]).expect("side_info parse");
        let si_len = si.byte_len();
        let main_data_slot = &frame.data[si_start + si_len..];
        let run = reservoir
            .assemble(usize::from(si.main_data_begin), main_data_slot)
            .expect("reservoir assemble");
        let fsf = decode_scalefactors(&hdr, &si, &run).expect("scalefactors decode");

        let mut bit_cursor = 0usize;
        for gr in 0..si.granule_count as usize {
            for ch in 0..si.channels as usize {
                let gc = &si.granules[gr][ch];
                let mut r = MainDataReader::new(&run);
                let mut left = bit_cursor;
                while left >= 32 {
                    let _ = r.read(32);
                    left -= 32;
                }
                if left > 0 {
                    let _ = r.read(left as u32);
                }
                let part3_bits = u32::from(gc.part2_3_length);
                let is = decode_huffman(&mut r, gc, part3_bits, hdr.sample_rate_hz, hdr.version)
                    .expect("huffman");
                let sf = &fsf.granules[gr][ch];
                let xr = requantize(&is, gc, sf, hdr.sample_rate_hz, hdr.version);
                // §2.4.3.4.8 reorder must run for short / mixed
                // granules; long granules pass through unchanged.
                let xr_ord = reorder(&xr, gc, hdr.sample_rate_hz, hdr.version);
                let xar = alias_reduce(&xr_ord, gc);
                let subband_time = imdct_granule(&xar, gc, &mut imdct_state);
                let pcm_f32 = synth_granule(&subband_time, &mut synth_state);
                for &p in pcm_f32.iter().take(PCM_PER_GRANULE) {
                    let v = p * f32::from(i16::MAX);
                    out_pcm.push(v.clamp(i16::MIN as f32, i16::MAX as f32) as i16);
                }
                bit_cursor += gc.part2_3_length as usize;
            }
        }
    }
    out_pcm
}
