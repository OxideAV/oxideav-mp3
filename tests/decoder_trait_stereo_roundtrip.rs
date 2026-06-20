//! Integration test for the round-177 stereo widening of the
//! `oxideav_core::Decoder` trait wrapper (`Mp3CoreDecoder`).
//!
//! The trait wrapper now accepts both mono and stereo MPEG-1 Layer III
//! frames; this test exercises the two stereo paths the framework
//! routinely sees on encoded streams:
//!
//! 1. **Independent stereo** (`ChannelMode::Stereo`, `mode = '00'`,
//!    `mode_extension = '00'`) — both channels carry their own L/R
//!    coefficients; the §2.4.3.4.9 stereo processing stage is a
//!    pass-through, and per-channel L/R are recovered directly by the
//!    per-channel `imdct_state` + `synth_state` pipeline.
//! 2. **Joint MS stereo** (`ChannelMode::JointStereo`,
//!    `mode_extension = '10'`) — channel 0 carries `M = (L+R)/√2`,
//!    channel 1 carries `S = (L-R)/√2`; the §2.4.3.4.9.2 inverse
//!    matrix runs between requantize and alias reduction inside the
//!    trait wrapper.
//!
//! For both modes:
//!
//! * Each emitted `AudioFrame` is **planar** (one plane per channel),
//!   with the per-plane interleaved S16 byte runs equal in length and
//!   in sample-count to `granule_count × PCM_PER_GRANULE`.
//! * The trait-driven L and R PCM byte streams equal the direct-chain
//!   stereo decode (the same chain used in
//!   `stereo_encoder_roundtrip.rs`, with per-channel `imdct_state` /
//!   `synth_state` and per-granule `process_stereo` for joint
//!   granules) **sample-for-sample**.

use std::f32::consts::PI;

use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, Packet, RuntimeContext, SampleFormat, TimeBase,
};
use oxideav_mp3::{
    alias_reduce, decode_huffman, decode_scalefactors, imdct_granule, parse_header,
    parse_side_info, pcm_f32_to_i16, process_stereo, requantize, synth_granule, ChannelMode,
    FrameWalker, ImdctState, MainDataReader, Mp3Encoder, Reservoir, SynthState, PCM_PER_GRANULE,
    SAMPLES_PER_FRAME_MPEG1,
};

const SR: u32 = 44_100;
const BITRATE: u32 = 192;

/// Interleaved `[L0, R0, …]` stereo `i16` PCM from two independent
/// sine tones at `freq_l` / `freq_r`.
fn stereo_sine_pcm(n: usize, freq_l: f32, freq_r: f32, sr: f32, amp: f32) -> Vec<i16> {
    let two_pi = 2.0 * PI;
    let mut out = Vec::with_capacity(n * 2);
    let scale = amp * (i16::MAX as f32);
    for i in 0..n {
        let t = i as f32 / sr;
        let l = (two_pi * freq_l * t).sin() * scale;
        let r = (two_pi * freq_r * t).sin() * scale;
        out.push(l.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16);
        out.push(r.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16);
    }
    out
}

fn encode_independent_stereo(pcm: &[i16]) -> Vec<u8> {
    let mut enc = Mp3Encoder::new(BITRATE, SR, ChannelMode::Stereo).expect("encoder");
    enc.push_samples(pcm).expect("push_samples");
    let mut bytes: Vec<u8> = Vec::new();
    enc.finish(&mut bytes).expect("finish");
    bytes
}

fn encode_joint_ms_stereo(pcm: &[i16]) -> Vec<u8> {
    let mut enc = Mp3Encoder::new_joint_stereo_ms(BITRATE, SR).expect("encoder");
    enc.push_samples(pcm).expect("push_samples");
    let mut bytes: Vec<u8> = Vec::new();
    enc.finish(&mut bytes).expect("finish");
    bytes
}

/// Reference decode chain — per-channel `imdct_state` / `synth_state`
/// and per-granule `process_stereo` on joint granules. Returns
/// `(left_pcm, right_pcm)`.
fn decode_stereo_direct(bytes: &[u8]) -> (Vec<i16>, Vec<i16>) {
    let mut reservoir = Reservoir::new();
    let mut synth: [SynthState; 2] = [SynthState::new(), SynthState::new()];
    let mut imdct: [ImdctState; 2] = [ImdctState::new(), ImdctState::new()];
    let mut out_l: Vec<i16> = Vec::new();
    let mut out_r: Vec<i16> = Vec::new();
    for frame in FrameWalker::new(bytes) {
        let hdr = parse_header(&frame.data[..4]).expect("header");
        let si_start = 4 + if hdr.crc_protected { 2 } else { 0 };
        let si = parse_side_info(&hdr, &frame.data[si_start..]).expect("side_info");
        let si_len = si.byte_len();
        let main_slot = &frame.data[si_start + si_len..];
        let run = reservoir
            .assemble(usize::from(si.main_data_begin), main_slot)
            .expect("reservoir");
        let fsf = decode_scalefactors(&hdr, &si, &run).expect("scalefactors");
        let nch = si.channels as usize;
        let mut bit_cursor = 0usize;
        for gr in 0..si.granule_count as usize {
            // Per-granule first pass: huffman + requantize.
            let mut xr_per_ch: Vec<[f32; 576]> = (0..nch).map(|_| [0.0; 576]).collect();
            for (ch, xr_slot) in xr_per_ch.iter_mut().enumerate() {
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
                *xr_slot = requantize(&is, gc, sf, hdr.sample_rate_hz, hdr.version);
                bit_cursor += gc.part2_3_length as usize;
            }
            if nch == 2 && hdr.mode == ChannelMode::JointStereo {
                let (l_split, r_split) = xr_per_ch.split_at_mut(1);
                let right_sf = &fsf.granules[gr][1];
                let right_gc = &si.granules[gr][1];
                process_stereo(
                    &mut l_split[0],
                    &mut r_split[0],
                    right_sf,
                    right_gc,
                    hdr.mode_extension,
                    hdr.sample_rate_hz,
                    hdr.version,
                );
            }
            for (ch, xr_ch) in xr_per_ch.iter().enumerate() {
                let gc = &si.granules[gr][ch];
                let xar = alias_reduce(xr_ch, gc);
                let subband_time = imdct_granule(&xar, gc, &mut imdct[ch]);
                let pcm_f32 = synth_granule(&subband_time, &mut synth[ch]);
                let sink = if ch == 0 { &mut out_l } else { &mut out_r };
                for &p in pcm_f32.iter().take(PCM_PER_GRANULE) {
                    sink.push(pcm_f32_to_i16(p));
                }
            }
        }
    }
    (out_l, out_r)
}

fn mp3_to_packets(bytes: &[u8]) -> Vec<Packet> {
    let tb = TimeBase::new(1, i64::from(SR));
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

fn build_stereo_decoder_params() -> CodecParameters {
    let mut p = CodecParameters::audio(CodecId::new("mp3"));
    p.sample_rate = Some(SR);
    p.channels = Some(2);
    p.sample_format = Some(SampleFormat::S16);
    p
}

/// Drive the supplied stereo MP3 byte stream through the registered
/// trait decoder and pull per-channel `i16` PCM out of the planar
/// `AudioFrame`s.
fn trait_decode_stereo(bytes: &[u8]) -> (Vec<i16>, Vec<i16>) {
    // Use the freestanding `make_decoder` so this test does not depend
    // on the runtime context registry already carrying our codec; the
    // registry-based path is exercised by the mono integration test.
    let mut dec = oxideav_mp3::make_decoder(&build_stereo_decoder_params()).expect("make_decoder");
    let mut out_l: Vec<i16> = Vec::new();
    let mut out_r: Vec<i16> = Vec::new();
    for pkt in mp3_to_packets(bytes) {
        dec.send_packet(&pkt).expect("send_packet");
        loop {
            match dec.receive_frame() {
                Ok(Frame::Audio(a)) => {
                    assert_eq!(
                        a.data.len(),
                        2,
                        "stereo trait decoder must emit two planes per frame"
                    );
                    for chunk in a.data[0].chunks_exact(2) {
                        out_l.push(i16::from_le_bytes([chunk[0], chunk[1]]));
                    }
                    for chunk in a.data[1].chunks_exact(2) {
                        out_r.push(i16::from_le_bytes([chunk[0], chunk[1]]));
                    }
                    // Per-frame sample count invariant.
                    assert_eq!(
                        a.samples as usize,
                        2 * PCM_PER_GRANULE,
                        "MPEG-1 Layer III frame must carry 1152 per-channel samples"
                    );
                    assert_eq!(
                        a.data[0].len(),
                        a.data[1].len(),
                        "planes must be equal length"
                    );
                }
                Ok(other) => panic!("non-audio frame: {other:?}"),
                Err(Error::NeedMore) => break,
                Err(e) => panic!("receive_frame: {e}"),
            }
        }
    }
    dec.flush().expect("flush");
    loop {
        match dec.receive_frame() {
            Ok(Frame::Audio(a)) => {
                for chunk in a.data[0].chunks_exact(2) {
                    out_l.push(i16::from_le_bytes([chunk[0], chunk[1]]));
                }
                for chunk in a.data[1].chunks_exact(2) {
                    out_r.push(i16::from_le_bytes([chunk[0], chunk[1]]));
                }
            }
            Ok(other) => panic!("non-audio frame on flush: {other:?}"),
            Err(Error::Eof) => break,
            Err(Error::NeedMore) => break,
            Err(e) => panic!("post-flush receive_frame: {e}"),
        }
    }
    (out_l, out_r)
}

#[test]
fn trait_decode_independent_stereo_matches_direct_chain_byte_exact() {
    // 250 ms of distinct sines per channel (L = 440 Hz, R = 880 Hz).
    let n = SR as usize / 4;
    let pcm = stereo_sine_pcm(n, 440.0, 880.0, SR as f32, 0.5);
    let wire = encode_independent_stereo(&pcm);
    assert!(wire.len() > 200, "stereo encoded stream too small");

    let (l_direct, r_direct) = decode_stereo_direct(&wire);
    let (l_trait, r_trait) = trait_decode_stereo(&wire);

    assert_eq!(l_trait.len(), l_direct.len(), "L sample count mismatch");
    assert_eq!(r_trait.len(), r_direct.len(), "R sample count mismatch");
    let mism_l = l_trait
        .iter()
        .zip(l_direct.iter())
        .filter(|(a, b)| a != b)
        .count();
    let mism_r = r_trait
        .iter()
        .zip(r_direct.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        mism_l, 0,
        "trait-driven independent-stereo L diverges from direct chain in {mism_l} samples"
    );
    assert_eq!(
        mism_r, 0,
        "trait-driven independent-stereo R diverges from direct chain in {mism_r} samples"
    );
}

#[test]
fn trait_decode_joint_ms_stereo_matches_direct_chain_byte_exact() {
    // Pan-asymmetric input maximally exercises the MS rotation: a
    // pure mono signal in L (silent R) becomes M = L/√2, S = L/√2,
    // so the joint-MS path moves real energy onto channel 1 — the
    // §2.4.3.4.9.2 inverse must run inside the trait wrapper to
    // recover (L=L, R=0).
    let n = SR as usize / 4;
    let mut pcm = stereo_sine_pcm(n, 440.0, 440.0, SR as f32, 0.5);
    // Force R = 0 (mono-on-L panning).
    for i in 0..n {
        pcm[2 * i + 1] = 0;
    }
    let wire = encode_joint_ms_stereo(&pcm);
    assert!(wire.len() > 200, "joint-MS encoded stream too small");

    // Confirm the first frame's wire header carries the joint-stereo +
    // MS bits — this is the only way to know the test exercises the
    // §2.4.3.4.9.2 inverse path inside the wrapper rather than the
    // pass-through independent path.
    let first = FrameWalker::new(&wire).next().expect("at least one frame");
    let first_hdr = parse_header(&first.data[..4]).expect("first header");
    assert_eq!(first_hdr.mode, ChannelMode::JointStereo);
    assert!(
        first_hdr.mode_extension.ms_stereo,
        "joint-MS encoder must set ms_stereo bit"
    );
    assert!(
        !first_hdr.mode_extension.intensity_stereo,
        "joint-MS encoder does not set intensity_stereo bit"
    );

    let (l_direct, r_direct) = decode_stereo_direct(&wire);
    let (l_trait, r_trait) = trait_decode_stereo(&wire);

    assert_eq!(l_trait.len(), l_direct.len(), "L sample count mismatch");
    assert_eq!(r_trait.len(), r_direct.len(), "R sample count mismatch");
    let mism_l = l_trait
        .iter()
        .zip(l_direct.iter())
        .filter(|(a, b)| a != b)
        .count();
    let mism_r = r_trait
        .iter()
        .zip(r_direct.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        mism_l, 0,
        "trait-driven joint-MS L diverges from direct chain in {mism_l} samples"
    );
    assert_eq!(
        mism_r, 0,
        "trait-driven joint-MS R diverges from direct chain in {mism_r} samples"
    );
}

#[test]
fn trait_decode_stereo_emits_planar_audioframes_with_correct_sample_count() {
    let n = SR as usize / 8; // 125 ms
    let pcm = stereo_sine_pcm(n, 220.0, 660.0, SR as f32, 0.4);
    let wire = encode_independent_stereo(&pcm);

    let mut dec = oxideav_mp3::make_decoder(&build_stereo_decoder_params()).expect("make_decoder");
    let mut frames_seen = 0usize;
    for pkt in mp3_to_packets(&wire) {
        dec.send_packet(&pkt).expect("send_packet");
        while let Ok(Frame::Audio(a)) = dec.receive_frame() {
            assert_eq!(a.data.len(), 2, "stereo plane count");
            assert_eq!(a.samples as usize, 2 * PCM_PER_GRANULE);
            // 2 bytes per S16 sample on each plane.
            assert_eq!(a.data[0].len(), 2 * a.samples as usize);
            assert_eq!(a.data[1].len(), 2 * a.samples as usize);
            frames_seen += 1;
        }
    }
    assert!(frames_seen > 0, "no frames decoded");
}

#[test]
fn registry_built_decoder_handles_stereo_packets() {
    // Confirm `oxideav_mp3::register` installs a decoder factory that
    // accepts stereo channel params and drives the same byte-exact
    // path as `make_decoder`.
    let pcm = stereo_sine_pcm(SR as usize / 8, 440.0, 880.0, SR as f32, 0.4);
    let wire = encode_independent_stereo(&pcm);
    let (l_direct, r_direct) = decode_stereo_direct(&wire);

    let mut ctx = RuntimeContext::new();
    oxideav_mp3::register(&mut ctx);
    let mut dec = ctx
        .codecs
        .first_decoder(&build_stereo_decoder_params())
        .expect("registered decoder");
    let mut out_l: Vec<i16> = Vec::new();
    let mut out_r: Vec<i16> = Vec::new();
    for pkt in mp3_to_packets(&wire) {
        dec.send_packet(&pkt).expect("send_packet");
        while let Ok(Frame::Audio(a)) = dec.receive_frame() {
            assert_eq!(a.data.len(), 2);
            for chunk in a.data[0].chunks_exact(2) {
                out_l.push(i16::from_le_bytes([chunk[0], chunk[1]]));
            }
            for chunk in a.data[1].chunks_exact(2) {
                out_r.push(i16::from_le_bytes([chunk[0], chunk[1]]));
            }
        }
    }
    dec.flush().expect("flush");
    while let Ok(Frame::Audio(a)) = dec.receive_frame() {
        for chunk in a.data[0].chunks_exact(2) {
            out_l.push(i16::from_le_bytes([chunk[0], chunk[1]]));
        }
        for chunk in a.data[1].chunks_exact(2) {
            out_r.push(i16::from_le_bytes([chunk[0], chunk[1]]));
        }
    }
    assert_eq!(out_l, l_direct);
    assert_eq!(out_r, r_direct);
}
