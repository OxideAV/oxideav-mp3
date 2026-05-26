//! Integration test for the Phase 2 step-23 forward mixed-block encode
//! path (round 152 — `Mp3Encoder::force_mixed_blocks_for_testing`).
//!
//! Validates the end-to-end §2.4.2.7 mixed-block encode path:
//!
//! 1. Subbands 0 and 1 (lines 0..36) are coded with the long-family
//!    forward MDCT — `forward_overlap → window_long_family_analysis(Long)
//!    → 36-pt mdct → ÷9` — exactly the same primitives the long-block
//!    branch uses.
//! 2. Subbands 2..31 (lines 36..576) are coded with the short-block
//!    forward MDCT — `forward_short_mdct_subband` — producing 18 bins
//!    each in subband-window-interleaved layout.
//! 3. `forward_reorder` is invoked with a mixed `GranuleChannel` so the
//!    long region passes through and only short SFB 3..12 is rewritten
//!    into native bitstream `[sfb][win][k]` order.
//! 4. No inverse alias reduction (the decoder's `alias_reduce` is a
//!    pass-through for `block_type == Short`, mixed or not).
//! 5. Side info carries `window_switching_flag = 1`, `block_type = 2`,
//!    `mixed_block_flag = 1`.
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

fn encode(force_mixed: bool) -> Vec<u8> {
    let n = SR as usize / 4; // 250 ms is plenty for several frames
    let pcm = sine_pcm(n, 440.0, SR as f32, 0.5);
    let mut enc = Mp3Encoder::new(BR, SR, ChannelMode::SingleChannel).expect("encoder build");
    enc.force_mixed_blocks_for_testing(force_mixed)
        .expect("force_mixed toggle");
    enc.push_samples(&pcm).expect("push pcm");
    let mut out: Vec<u8> = Vec::new();
    let _bytes = enc.finish(&mut out).expect("encoder finish");
    out
}

#[test]
fn force_mixed_blocks_rejected_on_stereo_encoder() {
    // The toggle is intentionally mono-only this round; multi-channel
    // mixed-block encode needs the §2.4.3.4.9 cross-channel
    // block-type-agreement wiring deferred to a follow-up.
    let mut enc = Mp3Encoder::new(BR, SR, ChannelMode::Stereo).expect("stereo encoder build");
    assert!(!enc.force_mixed_blocks_enabled());
    let err = enc
        .force_mixed_blocks_for_testing(true)
        .expect_err("stereo + force-mixed should be rejected");
    let msg = format!("{err}");
    assert!(
        msg.contains("Mp3Encoder::new_joint_stereo_ms")
            || msg.contains("Stereo")
            || msg.contains("JointStereo"),
        "unexpected error message: {msg}"
    );
    assert!(!enc.force_mixed_blocks_enabled(), "flag must stay off");
}

#[test]
fn force_mixed_clears_force_short_and_vice_versa() {
    // The two flags are mutually exclusive (a granule is long, short,
    // or mixed — not two-at-once). Enabling one should clear the other.
    let mut enc = Mp3Encoder::new(BR, SR, ChannelMode::SingleChannel).expect("mono encoder");
    enc.force_short_blocks_for_testing(true)
        .expect("force_short on");
    assert!(enc.force_short_blocks_enabled());
    assert!(!enc.force_mixed_blocks_enabled());
    enc.force_mixed_blocks_for_testing(true)
        .expect("force_mixed on");
    assert!(enc.force_mixed_blocks_enabled());
    assert!(
        !enc.force_short_blocks_enabled(),
        "force_short must be cleared when force_mixed is enabled"
    );
    enc.force_short_blocks_for_testing(true)
        .expect("force_short on again");
    assert!(enc.force_short_blocks_enabled());
    assert!(
        !enc.force_mixed_blocks_enabled(),
        "force_mixed must be cleared when force_short is re-enabled"
    );
}

#[test]
fn force_mixed_blocks_writes_mixed_block_side_info() {
    // With force-mixed on, every granule-channel's side-info must
    // carry window_switching_flag = 1, block_type = Short,
    // mixed_block_flag = 1.
    let bytes = encode(/*force_mixed=*/ true);
    let mut frames = 0usize;
    let mut any_granule = false;
    let mut mixed_only = true;
    for frame in FrameWalker::new(&bytes) {
        frames += 1;
        let hdr = parse_header(&frame.data[..4]).expect("header");
        let si = parse_side_info(&hdr, &frame.data[4..]).expect("side_info");
        for gr in 0..si.granule_count as usize {
            for ch in 0..si.channels as usize {
                any_granule = true;
                let gc = &si.granules[gr][ch];
                if !gc.window_switching_flag
                    || gc.block_type != BlockType::Short
                    || !gc.mixed_block_flag
                {
                    mixed_only = false;
                }
                // subblock_gain defaults to all-zero in this round.
                for &g in &gc.subblock_gain {
                    assert_eq!(g, 0, "subblock_gain default should be 0");
                }
            }
        }
    }
    assert!(frames > 0, "force-mixed emitted no frames");
    assert!(any_granule, "force-mixed emitted frames but no granules?!");
    assert!(mixed_only, "force-mixed stream carried a non-mixed granule");
}

#[test]
fn force_mixed_stream_decodes_to_finite_non_silent_pcm() {
    // The force-mixed stream must round-trip through the crate's own
    // decode primitives, producing PCM that is finite and not silent.
    // Bit-exact PSNR comparison against the long baseline depends on
    // the §C.1.5 attack-detection heuristic that picks block-types per
    // granule from signal energy; the deterministic force-mixed toggle
    // is a building-block primitive on top of which that heuristic
    // will land in a follow-up round.
    let bytes = encode(/*force_mixed=*/ true);
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
fn force_mixed_stream_passes_demuxer() {
    // The Mp3Demuxer (which parses headers + side-info on every frame)
    // must accept every force-mixed frame.
    let bytes = encode(/*force_mixed=*/ true);
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
        "demuxer surfaced zero frames on force-mixed stream"
    );
}

#[test]
fn force_mixed_default_off() {
    let enc = Mp3Encoder::new(BR, SR, ChannelMode::SingleChannel).expect("mono encoder");
    assert!(
        !enc.force_mixed_blocks_enabled(),
        "force_mixed must default to false"
    );
}

/// Decode an MP3 byte stream into mono `i16` PCM using the crate's own
/// primitives, with the §2.4.3.4.8 reorder stage in the pipeline for
/// short and mixed granules. Identical to the helper in
/// `tests/short_block_encoder_roundtrip.rs::decode_mp3_mono_short_aware`;
/// duplicated here because integration tests live in independent
/// compilation units and the test helpers are not exported from the
/// crate's public API.
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
