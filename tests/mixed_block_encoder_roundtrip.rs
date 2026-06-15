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
    parse_side_info, pcm_f32_to_i16, reorder, requantize, synth_granule, BlockType, ChannelMode,
    FrameWalker, ImdctState, MainDataReader, Mp3Demuxer, Mp3Encoder, Reservoir, SynthState,
    PCM_PER_GRANULE,
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
fn force_mixed_blocks_accepted_on_ms_stereo_encoder() {
    // r163 (§2.4.3.4.9 cross-channel-MS block-type agreement): the
    // force-mixed toggle now accepts MS-stereo joint modes. Both
    // channels of every granule emit the same
    // `(BlockType::Short, mixed_block_flag = true)`, satisfying the
    // §2.4.3.4.9 agreement trivially.
    let mut enc = Mp3Encoder::new_joint_stereo_ms(BR, SR).expect("MS-stereo encoder build");
    assert!(!enc.force_mixed_blocks_enabled());
    enc.force_mixed_blocks_for_testing(true)
        .expect("force-mixed toggle accepted on MS-stereo (r163)");
    assert!(enc.force_mixed_blocks_enabled());
}

#[test]
fn force_mixed_blocks_accepted_on_ms_auto_stereo_encoder() {
    // r163: the MS/LR auto picker also accepts force-mixed.
    let mut enc = Mp3Encoder::new_joint_stereo_auto(BR, SR).expect("MS-auto encoder build");
    assert!(!enc.force_mixed_blocks_enabled());
    enc.force_mixed_blocks_for_testing(true)
        .expect("force-mixed toggle accepted on MS-auto (r163)");
    assert!(enc.force_mixed_blocks_enabled());
}

#[test]
fn force_mixed_blocks_accepted_on_independent_stereo_encoder() {
    // r162: independent stereo is per-channel side-info verbatim
    // (§2.4.1.7 / §2.4.2.7) — both channels can independently take
    // the forced mixed block.
    let mut enc = Mp3Encoder::new(BR, SR, ChannelMode::Stereo).expect("stereo encoder build");
    enc.force_mixed_blocks_for_testing(true)
        .expect("force-mixed on independent stereo");
    assert!(enc.force_mixed_blocks_enabled());

    let mut enc2 =
        Mp3Encoder::new(BR, SR, ChannelMode::DualChannel).expect("dual-channel encoder build");
    enc2.force_mixed_blocks_for_testing(true)
        .expect("force-mixed on dual-channel");
    assert!(enc2.force_mixed_blocks_enabled());
}

#[test]
fn force_mixed_blocks_on_independent_stereo_writes_mixed_side_info_both_channels() {
    // Round-trip: encode 250 ms of stereo (different per-channel
    // tones) with force-mixed and confirm every emitted
    // granule-channel carries `window_switching_flag = 1`,
    // `block_type = Short`, `mixed_block_flag = 1`. Also confirm the
    // bytes round-trip through Mp3Demuxer (wire-layout sanity).
    use std::f32::consts::PI;
    let n = SR as usize / 4;
    let mut pcm = Vec::with_capacity(n * 2);
    let scale = 0.5 * (i16::MAX as f32);
    for i in 0..n {
        let t = i as f32 / SR as f32;
        let l = (2.0 * PI * 220.0 * t).sin() * scale;
        let r = (2.0 * PI * 440.0 * t).sin() * scale;
        pcm.push(l.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16);
        pcm.push(r.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16);
    }
    let mut enc = Mp3Encoder::new(BR, SR, ChannelMode::Stereo).expect("stereo encoder build");
    enc.force_mixed_blocks_for_testing(true)
        .expect("force-mixed accepted on independent stereo");
    enc.push_samples(&pcm).expect("push stereo pcm");
    let mut bytes: Vec<u8> = Vec::new();
    let _ = enc.finish(&mut bytes).expect("encoder finish");

    let mut frames = 0usize;
    let mut total_gcs = 0usize;
    let mut mixed_only = true;
    for frame in FrameWalker::new(&bytes) {
        frames += 1;
        let hdr = parse_header(&frame.data[..4]).expect("header parse");
        assert_eq!(
            hdr.channel_count(),
            2,
            "stereo encoder must emit 2-channel frames"
        );
        let si = parse_side_info(&hdr, &frame.data[4..]).expect("side_info parse");
        for gr in 0..si.granule_count as usize {
            for ch in 0..si.channels as usize {
                total_gcs += 1;
                let gc = &si.granules[gr][ch];
                if !gc.window_switching_flag
                    || gc.block_type != BlockType::Short
                    || !gc.mixed_block_flag
                {
                    mixed_only = false;
                }
            }
        }
    }
    assert!(frames > 0, "stereo force-mixed emitted no frames");
    assert!(
        total_gcs >= 4,
        "stereo force-mixed emitted < 4 gc (was {total_gcs})"
    );
    assert!(mixed_only, "stereo force-mixed emitted a non-mixed gc");

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
    assert!(
        packets > 0,
        "stereo demuxer accepted no force-mixed packets"
    );
}

#[test]
fn force_mixed_blocks_on_ms_stereo_writes_agreed_mixed_side_info() {
    // r163 (§2.4.3.4.9 cross-channel-MS agreement): with MS-stereo +
    // force-mixed, every (gr, ch) tile carries the same
    // `(BlockType::Short, mixed_block_flag = true)` pair. The frame
    // header still emits joint stereo + MS on.
    use std::f32::consts::PI;
    let n = SR as usize / 4;
    let mut pcm = Vec::with_capacity(n * 2);
    let scale = 0.5 * (i16::MAX as f32);
    for i in 0..n {
        let t = i as f32 / SR as f32;
        let l = (2.0 * PI * 220.0 * t).sin() * scale;
        let r = (2.0 * PI * 440.0 * t).sin() * scale;
        pcm.push(l.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16);
        pcm.push(r.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16);
    }
    let mut enc = Mp3Encoder::new_joint_stereo_ms(BR, SR).expect("MS-stereo encoder build");
    enc.force_mixed_blocks_for_testing(true)
        .expect("force-mixed accepted on MS-stereo (r163)");
    enc.push_samples(&pcm).expect("push stereo pcm");
    let mut bytes: Vec<u8> = Vec::new();
    let _ = enc.finish(&mut bytes).expect("encoder finish");

    let mut frames = 0usize;
    let mut total_gcs = 0usize;
    let mut mixed_only = true;
    let mut all_agreed = true;
    for frame in FrameWalker::new(&bytes) {
        frames += 1;
        let hdr = parse_header(&frame.data[..4]).expect("header parse");
        assert_eq!(hdr.channel_count(), 2);
        let si = parse_side_info(&hdr, &frame.data[4..]).expect("side_info parse");
        for gr in 0..si.granule_count as usize {
            let gc_l = &si.granules[gr][0];
            let gc_r = &si.granules[gr][1];
            total_gcs += 2;
            for gc in [gc_l, gc_r] {
                if !gc.window_switching_flag
                    || gc.block_type != BlockType::Short
                    || !gc.mixed_block_flag
                {
                    mixed_only = false;
                }
            }
            if gc_l.block_type != gc_r.block_type
                || gc_l.window_switching_flag != gc_r.window_switching_flag
                || gc_l.mixed_block_flag != gc_r.mixed_block_flag
            {
                all_agreed = false;
            }
        }
    }
    assert!(frames > 0, "MS-stereo + force-mixed emitted no frames");
    assert!(
        total_gcs >= 4,
        "MS-stereo + force-mixed emitted < 4 gc (was {total_gcs})"
    );
    assert!(mixed_only, "MS-stereo + force-mixed emitted a non-mixed gc");
    assert!(
        all_agreed,
        "MS-stereo + force-mixed emitted disagreeing per-channel side-info"
    );

    let mut demux =
        Mp3Demuxer::open(Box::new(Cursor::new(bytes.clone()))).expect("MS-stereo demuxer open");
    let mut packets = 0usize;
    loop {
        match demux.next_packet() {
            Ok(_pkt) => packets += 1,
            Err(oxideav_core::Error::Eof) => break,
            Err(e) => panic!("MS-stereo demuxer next_packet: {e}"),
        }
    }
    assert!(
        packets > 0,
        "MS-stereo demuxer accepted no force-mixed packets"
    );
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
                    out_pcm.push(pcm_f32_to_i16(p));
                }
                bit_cursor += gc.part2_3_length as usize;
            }
        }
    }
    out_pcm
}

/// Encode the same sine PCM with the §C.1.5.4.3 outer (distortion-
/// control) loop wired through the mixed-block forward MDCT path
/// (Phase 2 step 29). The encoder is built via
/// `Mp3Encoder::new_with_outer_loop` + `force_mixed_blocks_for_testing(true)`
/// so every granule lands on the mixed branch and the dispatcher
/// routes the (gr, ch) pair onto `outer_loop_search_mixed`.
fn encode_outer_loop_mixed(uniform_threshold: f64) -> Vec<u8> {
    let n = SR as usize / 4; // 250 ms
    let pcm = sine_pcm(n, 440.0, SR as f32, 0.5);
    let mut enc =
        Mp3Encoder::new_with_outer_loop(BR, SR, ChannelMode::SingleChannel, uniform_threshold)
            .expect("encoder build");
    enc.force_mixed_blocks_for_testing(true)
        .expect("force_mixed toggle");
    enc.push_samples(&pcm).expect("push pcm");
    let mut out: Vec<u8> = Vec::new();
    let _bytes = enc.finish(&mut out).expect("encoder finish");
    out
}

#[test]
fn force_mixed_plus_outer_loop_writes_scalefac_compress_15() {
    // Phase 2 step 29: the `force_mixed_blocks_for_testing(true)` toggle
    // composed with `Mp3Encoder::new_with_outer_loop` MUST run the
    // mixed-block outer loop on every granule-channel. The wire
    // signature of the new primitive (vs the previous round's
    // fixed-gain fallback) is `scalefac_compress = 15` on every
    // assembled granule — the outer loop seeds slen1 = 4 / slen2 = 3 so
    // the §C.1.5.4.3.6 caps are 15 / 7 and the chosen scalefactors fit
    // in the part2 field. Before r159 the dispatcher fell back to
    // `scalefac_compress = 0` on mixed; the new code path is provably
    // exercised only when the assembled granules show `scalefac_compress = 15`.
    let bytes = encode_outer_loop_mixed(oxideav_mp3::stream_encoder::DEFAULT_OUTER_LOOP_THRESHOLD);
    let mut frames = 0usize;
    let mut any_granule = false;
    let mut all_outer_loop_compress = true;
    for frame in FrameWalker::new(&bytes) {
        frames += 1;
        let hdr = parse_header(&frame.data[..4]).expect("header");
        let si = parse_side_info(&hdr, &frame.data[4..]).expect("side_info");
        for gr in 0..si.granule_count as usize {
            for ch in 0..si.channels as usize {
                any_granule = true;
                let gc = &si.granules[gr][ch];
                // The mixed-block branch must still wear
                // window_switching + Short + mixed_block_flag = 1.
                assert!(gc.window_switching_flag);
                assert_eq!(gc.block_type, BlockType::Short);
                assert!(
                    gc.mixed_block_flag,
                    "force-mixed must keep mixed_block_flag"
                );
                if gc.scalefac_compress != 15 {
                    all_outer_loop_compress = false;
                }
            }
        }
    }
    assert!(frames > 0);
    assert!(any_granule);
    assert!(
        all_outer_loop_compress,
        "mixed-block outer-loop dispatch must emit scalefac_compress = 15 \
         (the r159 primitive's part2 layout). Without this every (gr, ch) \
         is still on the r158 fixed-gain fallback.",
    );
}

#[test]
fn force_mixed_plus_outer_loop_subblock_gain_bounded() {
    // The mixed outer loop reuses the pure-short loop's bounded
    // §2.4.2.7 subblock_gain search (3-bit field, range [0, 7]). On a
    // gentle 440 Hz fixture nothing should escalate subblock_gain off
    // zero, but more importantly: regardless of fixture, the field must
    // ALWAYS stay within [0, 7] — the wire is 3 bits and a value > 7
    // would be a clean-room invariant violation.
    let bytes = encode_outer_loop_mixed(oxideav_mp3::stream_encoder::DEFAULT_OUTER_LOOP_THRESHOLD);
    for frame in FrameWalker::new(&bytes) {
        let hdr = parse_header(&frame.data[..4]).expect("header");
        let si = parse_side_info(&hdr, &frame.data[4..]).expect("side_info");
        for gr in 0..si.granule_count as usize {
            for ch in 0..si.channels as usize {
                let gc = &si.granules[gr][ch];
                for (w, &sg) in gc.subblock_gain.iter().enumerate() {
                    assert!(
                        sg <= 7,
                        "subblock_gain[{w}] = {sg} exceeds §2.4.2.7 3-bit field range",
                    );
                }
            }
        }
    }
}

#[test]
fn force_mixed_plus_outer_loop_roundtrips_to_finite_non_silent_pcm() {
    // The outer-loop mixed encode must round-trip through the crate's
    // own decode chain producing finite, non-silent PCM. This is the
    // direct analogue of `force_mixed_stream_decodes_to_finite_non_silent_pcm`
    // (r158 fixed-gain) for the new r159 distortion-control path.
    let bytes = encode_outer_loop_mixed(oxideav_mp3::stream_encoder::DEFAULT_OUTER_LOOP_THRESHOLD);
    let recon = decode_mp3_mono_short_aware(&bytes);
    assert!(!recon.is_empty(), "decoded PCM was empty");
    let energy: f64 = recon
        .iter()
        .map(|&v| f64::from(v) * f64::from(v))
        .sum::<f64>()
        / recon.len() as f64;
    assert!(
        energy.is_finite() && energy > 0.0,
        "decoded PCM had zero or non-finite energy ({energy})",
    );
    // Bounded waveform sanity check (same heuristic the r158 mixed test
    // applies): at least some zero crossings — a pure 440 Hz sine
    // crosses zero ~880 times per second.
    let mut zero_crossings = 0usize;
    for w in recon.windows(2) {
        if (w[0] >= 0) != (w[1] >= 0) {
            zero_crossings += 1;
        }
    }
    assert!(
        zero_crossings > 10,
        "no audible zero crossings: {zero_crossings} (decode chain broken)",
    );
}

#[test]
fn force_mixed_plus_outer_loop_demuxer_accepts_stream() {
    // The Mp3Demuxer (which parses headers + side-info on every frame)
    // must accept every force-mixed-outer-loop frame. The new
    // dispatcher path changes part2_3_length values; this guards
    // against a frame-length / side-info mismatch breaking the
    // demuxer's per-frame walk.
    let bytes = encode_outer_loop_mixed(oxideav_mp3::stream_encoder::DEFAULT_OUTER_LOOP_THRESHOLD);
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
        "demuxer surfaced zero frames on force-mixed-outer-loop stream",
    );
}
