//! Integration test for Phase 2 step 12 — the `oxideav_core::Encoder`
//! trait wiring on top of the `Mp3Encoder` stream encoder.
//!
//! Unlike `stream_encoder_roundtrip.rs` (which calls the direct
//! `Mp3Encoder::push_samples` + `finish` API), this test drives the
//! encoder **exclusively through the framework's `Encoder` trait** —
//! `send_frame` / `receive_packet` / `flush` — to confirm the
//! adaptation layer is correct end-to-end. It then re-assembles the
//! emitted packets into a contiguous MP3 byte stream, runs the byte
//! stream through `Mp3Demuxer`, and decodes it via the same
//! crate-local decode primitives the direct-API test uses
//! (`decode_huffman` → `requantize` → `alias_reduce` → `imdct_granule`
//! → `synth_granule`).
//!
//! Validation:
//!
//! 1. The registered encoder factory builds a working boxed
//!    `oxideav_core::Encoder`.
//! 2. Each `receive_packet` returns one complete MP3 frame (header +
//!    side-info + main-data slot) starting with the 0xFFF sync.
//! 3. The re-assembled stream demuxes cleanly with `Mp3Demuxer`.
//! 4. PSNR of the trait-driven round-trip is ≥ the direct-API
//!    baseline (≈ 80 dB on a 440 Hz 0.5-amplitude sine after the
//!    1057-sample group delay).

use std::f32::consts::PI;
use std::io::Cursor;

use oxideav_core::{
    AudioFrame, CodecId, CodecParameters, Demuxer, Encoder, Error, Frame, RuntimeContext,
    SampleFormat,
};
use oxideav_mp3::{
    alias_reduce, decode_huffman, decode_scalefactors, imdct_granule, parse_header,
    parse_side_info, pcm_f32_to_i16, requantize, synth_granule, FrameWalker, ImdctState,
    MainDataReader, Mp3Demuxer, Reservoir, SynthState, PCM_PER_GRANULE, SAMPLES_PER_FRAME_MPEG1,
};

/// Synthesise `n` mono `i16` samples of a sine tone.
fn sine_pcm(n: usize, freq_hz: f32, sample_rate_hz: f32, amp: f32) -> Vec<i16> {
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

/// PSNR (dB) of `recon` against `original`. Equal lengths required.
fn psnr(original: &[i16], recon: &[i16]) -> f32 {
    assert_eq!(original.len(), recon.len(), "psnr length mismatch");
    let mut sse = 0.0f64;
    for (a, b) in original.iter().zip(recon.iter()) {
        let d = f64::from(*a) - f64::from(*b);
        sse += d * d;
    }
    let mse = sse / original.len() as f64;
    if mse == 0.0 {
        return f32::INFINITY;
    }
    let max = f64::from(i16::MAX);
    (10.0 * (max * max / mse).log10()) as f32
}

/// Decode an MP3 byte stream into mono `i16` PCM using the crate's
/// own decode primitives. Identical chain to the direct-API test;
/// kept locally so this test stays self-contained.
fn decode_mp3_mono(bytes: &[u8]) -> Vec<i16> {
    let mut reservoir = Reservoir::new();
    let mut synth_state = SynthState::new();
    let mut imdct_state = ImdctState::new();
    let mut out_pcm: Vec<i16> = Vec::new();
    for frame in FrameWalker::new(bytes) {
        let hdr = parse_header(&frame.data[..4]).expect("header parse");
        let si_start = 4; // no CRC
        let si = parse_side_info(&hdr, &frame.data[si_start..]).expect("side_info parse");
        let si_len = si.byte_len();
        let main_slot = &frame.data[si_start + si_len..];
        let run = reservoir
            .assemble(usize::from(si.main_data_begin), main_slot)
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
                let xar = alias_reduce(&xr, gc);
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

/// Build a boxed encoder via the runtime-context registry installed by
/// `oxideav_mp3::register`. This exercises the full registration +
/// factory-lookup path, not just the direct factory call.
fn make_encoder_via_registry(sample_rate: u32, bit_rate_bps: u64) -> Box<dyn Encoder> {
    let mut ctx = RuntimeContext::default();
    oxideav_mp3::register(&mut ctx);
    let mut params = CodecParameters::audio(CodecId::new("mp3"));
    params.sample_rate = Some(sample_rate);
    params.channels = Some(1);
    params.sample_format = Some(SampleFormat::S16);
    params.bit_rate = Some(bit_rate_bps);
    ctx.codecs
        .first_encoder(&params)
        .expect("encoder factory present after register()")
}

#[test]
fn registry_encoder_emits_valid_mp3_frames() {
    // Smoke-level: 200 ms of sine, drive via the trait API, confirm
    // every received packet is a valid MP3 frame.
    let sr = 44_100u32;
    let n = (sr as usize) / 5; // 200 ms
    let pcm = sine_pcm(n, 440.0, sr as f32, 0.5);
    let mut enc = make_encoder_via_registry(sr, 128_000);

    // Send the PCM in two batches to exercise multi-frame buffering.
    let half = pcm.len() / 2;
    for slice in [&pcm[..half], &pcm[half..]] {
        let mut bytes = Vec::with_capacity(slice.len() * 2);
        for s in slice {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        let f = AudioFrame {
            samples: slice.len() as u32,
            pts: None,
            data: vec![bytes],
        };
        enc.send_frame(&Frame::Audio(f)).expect("send_frame");
        // Before flush, no packets are available.
        assert!(matches!(enc.receive_packet(), Err(Error::NeedMore)));
    }

    enc.flush().expect("flush");

    let mut packets = Vec::new();
    loop {
        match enc.receive_packet() {
            Ok(p) => packets.push(p),
            Err(Error::Eof) => break,
            Err(e) => panic!("unexpected error during drain: {e}"),
        }
    }
    // 200 ms / (1152 / 44100) ≈ 7.66 frames. The encoder rounds up
    // for the tail.
    assert!(
        (7..=9).contains(&packets.len()),
        "unexpected packet count {}",
        packets.len()
    );
    // Every packet starts with the MP3 sync (0xFFF) and stamps a
    // monotonic PTS.
    let mut prev = i64::MIN;
    for pkt in &packets {
        assert!(pkt.data.len() >= 4, "packet too short");
        assert_eq!(pkt.data[0], 0xFF);
        assert_eq!(pkt.data[1] & 0xE0, 0xE0);
        let pts = pkt.pts.expect("PTS stamped");
        assert!(pts > prev, "PTS not monotonic ({prev} >= {pts})");
        prev = pts;
        assert_eq!(pkt.duration, Some(SAMPLES_PER_FRAME_MPEG1 as i64));
        assert!(pkt.flags.keyframe);
    }
}

#[test]
fn registry_encoder_self_decode_psnr() {
    // The trait-driven equivalent of the direct-API
    // `sine_tone_one_second_self_decode_psnr` test. Drives the
    // registered `oxideav_core::Encoder`, re-assembles its packets
    // into a contiguous byte stream, demuxes via `Mp3Demuxer`, and
    // decodes via the crate's own primitives. Asserts PSNR > 20 dB
    // matches the direct-API baseline (typical ~80 dB).
    const SR: u32 = 44_100;
    const BR: u64 = 128_000;
    let n = SR as usize; // 1 second
    let pcm = sine_pcm(n, 440.0, SR as f32, 0.5);

    let mut enc = make_encoder_via_registry(SR, BR);
    // Push the whole PCM in one frame.
    let mut bytes = Vec::with_capacity(pcm.len() * 2);
    for s in &pcm {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    let f = AudioFrame {
        samples: pcm.len() as u32,
        pts: None,
        data: vec![bytes],
    };
    enc.send_frame(&Frame::Audio(f)).expect("send_frame");
    enc.flush().expect("flush");

    // Drain packets, re-concatenate into the on-wire stream.
    let mut wire: Vec<u8> = Vec::new();
    let mut pkt_count = 0usize;
    loop {
        match enc.receive_packet() {
            Ok(p) => {
                wire.extend_from_slice(&p.data);
                pkt_count += 1;
            }
            Err(Error::Eof) => break,
            Err(e) => panic!("drain error: {e}"),
        }
    }
    assert!(
        (38..=40).contains(&pkt_count),
        "packet count out of range: {pkt_count}"
    );
    assert!(wire.len() > 1000, "wire stream too small: {}", wire.len());

    // Demux to confirm the byte stream is a valid MP3 file.
    let mut demux = Mp3Demuxer::open(Box::new(Cursor::new(wire.clone()))).expect("demuxer open");
    let mut demuxed = 0usize;
    loop {
        match demux.next_packet() {
            Ok(_) => demuxed += 1,
            Err(Error::Eof) => break,
            Err(e) => panic!("demuxer error: {e}"),
        }
    }
    assert_eq!(demuxed, pkt_count, "demuxer frame count mismatch");

    // Self-decode via the crate's primitives and compute PSNR
    // against the input with the 1057-sample group-delay alignment
    // documented in `stream_encoder_roundtrip.rs`.
    let recon = decode_mp3_mono(&wire);
    let warmup = 4 * 1152;
    let total_delay = 1057usize;
    assert!(
        recon.len() > warmup + total_delay,
        "recon too short: {}",
        recon.len()
    );
    let head_recon = warmup + total_delay;
    let cmp_len = recon
        .len()
        .saturating_sub(head_recon)
        .min(pcm.len() - warmup);
    let recon_cmp = &recon[head_recon..head_recon + cmp_len];
    let pcm_cmp = &pcm[warmup..warmup + cmp_len];
    let p = psnr(pcm_cmp, recon_cmp);
    eprintln!(
        "trait-driven self-decode PSNR = {p} dB (n={} samples, delay={total_delay})",
        cmp_len
    );
    // The direct-API path achieves ~80–86 dB; we require strictly
    // > 20 dB to match the existing baseline assertion. A regression
    // dropping below this would point at the trait wrapper rather
    // than the underlying encoder.
    assert!(p > 20.0, "trait-driven PSNR too low: {p} dB");
}
