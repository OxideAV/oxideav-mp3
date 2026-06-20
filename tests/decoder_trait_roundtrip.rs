//! Integration test for the r141 `oxideav_core::Decoder` trait
//! wiring on top of the existing Layer III decode chain.
//!
//! Symmetric counterpart to `encoder_trait_roundtrip.rs`. Where that
//! test drives the encoder through the framework trait and decodes via
//! the direct primitives, this test does the reverse: it encodes via
//! the direct `Mp3Encoder`, slices the resulting byte stream into
//! per-MP3-frame packets, drives them through the **registered**
//! `oxideav_core::Decoder` trait API (resolved via
//! `RuntimeContext::codecs::first_decoder`), and asserts the trait-
//! driven PCM is byte-exact identical to what the direct decode chain
//! produces on the same input bytes.
//!
//! Validation:
//!
//! 1. The runtime-context registry installed by `oxideav_mp3::register`
//!    exposes a working `oxideav_core::Decoder` factory.
//! 2. `send_packet` / `receive_frame` produce one `AudioFrame` per
//!    inbound MP3 frame, with the per-frame sample count `1152` and
//!    monotonic PTS lifted verbatim from the inbound packet.
//! 3. The trait-driven PCM byte stream equals the direct-chain PCM
//!    byte stream sample-for-sample (the round-mandate's bit-exact
//!    requirement).

use std::f32::consts::PI;

use oxideav_core::{
    CodecId, CodecParameters, Decoder, Error, Frame, Packet, RuntimeContext, SampleFormat, TimeBase,
};
use oxideav_mp3::{
    alias_reduce, decode_huffman, decode_scalefactors, imdct_granule, parse_header,
    parse_side_info, pcm_f32_to_i16, requantize, synth_granule, ChannelMode, FrameWalker,
    ImdctState, MainDataReader, Mp3Encoder, Reservoir, SynthState, PCM_PER_GRANULE,
    SAMPLES_PER_FRAME_MPEG1,
};

fn sine_pcm(n: usize, freq_hz: f32, sr: f32, amp: f32) -> Vec<i16> {
    let two_pi = 2.0 * PI;
    let scale = amp * (i16::MAX as f32);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f32 / sr;
        let s = (two_pi * freq_hz * t).sin() * scale;
        out.push(s.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16);
    }
    out
}

fn encode_to_mp3(pcm: &[i16], sample_rate: u32, bitrate_kbps: u32) -> Vec<u8> {
    let mut enc = Mp3Encoder::new(bitrate_kbps, sample_rate, ChannelMode::SingleChannel)
        .expect("Mp3Encoder build");
    enc.push_samples(pcm).expect("push_samples");
    let mut bytes: Vec<u8> = Vec::new();
    enc.finish(&mut bytes).expect("finish");
    bytes
}

fn mp3_to_packets(bytes: &[u8], sample_rate: u32) -> Vec<Packet> {
    let tb = TimeBase::new(1, i64::from(sample_rate));
    let mut out = Vec::new();
    let mut pts: i64 = 0;
    for f in FrameWalker::new(bytes) {
        let mut pkt = Packet::new(0, tb, f.data.to_vec());
        pkt.pts = Some(pts);
        pkt.duration = Some(SAMPLES_PER_FRAME_MPEG1 as i64);
        out.push(pkt);
        pts += SAMPLES_PER_FRAME_MPEG1 as i64;
    }
    out
}

/// Reference decode through the direct primitives (the same chain used
/// inside the trait wrapper, so a byte-exact match is the success
/// criterion for the trait adaptor).
fn decode_direct(bytes: &[u8]) -> Vec<i16> {
    let mut reservoir = Reservoir::new();
    let mut synth_state = SynthState::new();
    let mut imdct_state = ImdctState::new();
    let mut out_pcm: Vec<i16> = Vec::new();
    for frame in FrameWalker::new(bytes) {
        let hdr = parse_header(&frame.data[..4]).expect("header parse");
        let si_start = 4 + if hdr.crc_protected { 2 } else { 0 };
        let si = parse_side_info(&hdr, &frame.data[si_start..]).expect("side_info parse");
        let si_len = si.byte_len();
        let main_slot = &frame.data[si_start + si_len..frame.data.len()];
        let run = reservoir
            .assemble(usize::from(si.main_data_begin), main_slot)
            .expect("reservoir assemble");
        let fsf = decode_scalefactors(&hdr, &si, &run).expect("scalefactors decode");
        let mut bit_cursor = 0usize;
        for gr in 0..si.granule_count as usize {
            for ch in 0..si.channels as usize {
                let gc = &si.granules[gr][ch];
                let mut r = MainDataReader::new(&run);
                let mut left = bit_cursor + fsf.part2_bits[gr][ch] as usize;
                while left >= 32 {
                    let _ = r.read(32);
                    left -= 32;
                }
                if left > 0 {
                    let _ = r.read(left as u32);
                }
                let part3_bits =
                    u32::from(gc.part2_3_length).saturating_sub(fsf.part2_bits[gr][ch]);
                let is = decode_huffman(&mut r, gc, part3_bits, hdr.sample_rate_hz, hdr.version)
                    .expect("huffman");
                let sf = &fsf.granules[gr][ch];
                let xr = requantize(&is, gc, sf, hdr.sample_rate_hz, hdr.version);
                let xar = alias_reduce(&xr, gc);
                let st = imdct_granule(&xar, gc, &mut imdct_state);
                let pcm_f32 = synth_granule(&st, &mut synth_state);
                for &p in pcm_f32.iter().take(PCM_PER_GRANULE) {
                    out_pcm.push(pcm_f32_to_i16(p));
                }
                bit_cursor += gc.part2_3_length as usize;
            }
        }
    }
    out_pcm
}

fn make_decoder_via_registry(sample_rate: u32) -> Box<dyn Decoder> {
    let mut ctx = RuntimeContext::default();
    oxideav_mp3::register(&mut ctx);
    let mut params = CodecParameters::audio(CodecId::new("mp3"));
    params.sample_rate = Some(sample_rate);
    params.channels = Some(1);
    params.sample_format = Some(SampleFormat::S16);
    ctx.codecs
        .first_decoder(&params)
        .expect("decoder factory present after register()")
}

#[test]
fn registry_decoder_emits_audio_frames_with_monotonic_pts() {
    // 250 ms of sine → encode → slice into packets → drive the
    // registry-installed decoder. Confirm each emitted AudioFrame
    // carries 1152 samples per channel (MPEG-1 Layer III) and a
    // monotonically-increasing PTS.
    const SR: u32 = 44_100;
    let n = (SR as usize) / 4;
    let pcm = sine_pcm(n, 440.0, SR as f32, 0.5);
    let wire = encode_to_mp3(&pcm, SR, 128);
    assert!(wire.len() > 100, "encoded too short");

    let mut dec = make_decoder_via_registry(SR);
    let mut emitted: Vec<Frame> = Vec::new();
    for pkt in mp3_to_packets(&wire, SR) {
        dec.send_packet(&pkt).expect("send_packet");
        loop {
            match dec.receive_frame() {
                Ok(f) => emitted.push(f),
                Err(Error::NeedMore) => break,
                Err(e) => panic!("receive_frame: {e}"),
            }
        }
    }
    dec.flush().expect("flush");

    assert!(!emitted.is_empty(), "decoder emitted no frames");
    let mut prev_pts = i64::MIN;
    for f in &emitted {
        let Frame::Audio(a) = f else {
            panic!("non-audio frame from MP3 decoder: {f:?}");
        };
        assert_eq!(
            a.samples, SAMPLES_PER_FRAME_MPEG1 as u32,
            "expected {} samples/frame, got {}",
            SAMPLES_PER_FRAME_MPEG1, a.samples
        );
        assert_eq!(
            a.data.len(),
            1,
            "mono S16 frame should have exactly 1 plane",
        );
        // Interleaved/planar S16 mono → 2 bytes per sample.
        assert_eq!(
            a.data[0].len(),
            (a.samples as usize) * 2,
            "S16 plane length mismatch",
        );
        let pts = a.pts.expect("PTS stamped from inbound packet");
        assert!(pts > prev_pts, "PTS not monotonic ({prev_pts} >= {pts})");
        prev_pts = pts;
    }
}

#[test]
fn registry_decoder_byte_exact_against_direct_chain() {
    // The core round-mandate check: drive a synthetic MP3 byte stream
    // through the registered Decoder trait API, recover i16 PCM
    // frames, byte-exact match to what the direct decode chain
    // produces on the same input bytes.
    const SR: u32 = 44_100;
    let n = (SR as usize) / 2; // 500 ms
    let pcm = sine_pcm(n, 440.0, SR as f32, 0.5);
    let wire = encode_to_mp3(&pcm, SR, 128);

    // Direct-chain reference.
    let direct = decode_direct(&wire);

    // Trait-driven equivalent.
    let mut dec = make_decoder_via_registry(SR);
    let mut trait_out: Vec<i16> = Vec::new();
    for pkt in mp3_to_packets(&wire, SR) {
        dec.send_packet(&pkt).expect("send_packet");
        loop {
            match dec.receive_frame() {
                Ok(Frame::Audio(a)) => {
                    for chunk in a.data[0].chunks_exact(2) {
                        trait_out.push(i16::from_le_bytes([chunk[0], chunk[1]]));
                    }
                }
                Ok(other) => panic!("non-audio frame: {other:?}"),
                Err(Error::NeedMore) => break,
                Err(e) => panic!("receive_frame: {e}"),
            }
        }
    }
    dec.flush().expect("flush");

    assert_eq!(
        trait_out.len(),
        direct.len(),
        "trait-driven sample count {} != direct-chain {}",
        trait_out.len(),
        direct.len()
    );
    let mismatches = trait_out
        .iter()
        .zip(direct.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        mismatches, 0,
        "trait-driven decode differs from direct chain at {mismatches} samples",
    );
}

#[test]
fn registry_installs_both_encoder_and_decoder_factories() {
    let mut ctx = RuntimeContext::default();
    oxideav_mp3::register(&mut ctx);
    let id = CodecId::new("mp3");
    assert!(ctx.codecs.has_decoder(&id), "decoder factory not installed",);
    assert!(ctx.codecs.has_encoder(&id), "encoder factory not installed",);
}
