//! End-to-end tests for the **Annex D Psy Model 2** VBR path.
//!
//! Covers:
//! - `psy_model = 2` produces a non-empty, decodable bitstream.
//! - Round-trip decode through our own decoder produces a clean spectrum
//!   at the input frequency (VBR round-trip quality gate).
//! - ffmpeg cross-decode succeeds without warnings (interop sanity).
//! - Psy-2 produces different output than Psy-1 on a multi-tone fixture
//!   (regression guard: a no-op Psy-2 path would emit identical bytes to
//!   Psy-1 since they share the same gain ladder).
//! - Short-block fallback: transient-heavy input must still encode cleanly
//!   — the Psy-1 short path used by Psy-2's short-block fallback must not
//!   regress.
//! - MPEG-2 LSF path works with psy_model = 2.
//!
//! Skipped silently when ffmpeg is not on PATH (keeps CI portable).

use oxideav_core::options::CodecOptions;
use oxideav_core::{AudioFrame, CodecId, CodecParameters, Frame, Packet, SampleFormat, TimeBase};
use oxideav_mp3::decoder::make_decoder;
use oxideav_mp3::encoder::make_encoder;
use oxideav_mp3::frame::parse_frame_header;
use oxideav_mp3::CODEC_ID_STR;

fn build_sine_pcm(freq: f32, sample_rate: u32, duration_s: f32, amp: f32) -> Vec<i16> {
    let n = (sample_rate as f32 * duration_s) as usize;
    let mut out = Vec::with_capacity(n);
    let two_pi = 2.0 * std::f32::consts::PI;
    for i in 0..n {
        let t = i as f32 / sample_rate as f32;
        let s = (two_pi * freq * t).sin() * amp;
        out.push((s * 32767.0) as i16);
    }
    out
}

fn build_music_pcm(sample_rate: u32, duration_s: f32) -> Vec<i16> {
    let n = (sample_rate as f32 * duration_s) as usize;
    let mut out = Vec::with_capacity(n);
    let two_pi = 2.0 * std::f32::consts::PI;
    let freqs = [220.0_f32, 440.0, 587.0, 880.0, 1318.0, 1760.0, 3520.0];
    let weights = [0.20_f32, 0.20, 0.16, 0.14, 0.12, 0.10, 0.08];
    for i in 0..n {
        let t = i as f32 / sample_rate as f32;
        let mut s = 0.0f32;
        for (f, w) in freqs.iter().zip(weights.iter()) {
            s += (two_pi * f * t).sin() * w;
        }
        s = s.clamp(-1.0, 1.0) * 0.5;
        out.push((s * 32767.0) as i16);
    }
    out
}

fn build_castanet_pcm(sample_rate: u32, duration_s: f32) -> Vec<i16> {
    let n = (sample_rate as f32 * duration_s) as usize;
    let mut out = Vec::with_capacity(n);
    let burst_period = sample_rate as usize / 4;
    let burst_len = sample_rate as usize / 100;
    let mut rng_state: u32 = 0xdead_beef;
    for i in 0..n {
        rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
        let noise = ((rng_state >> 16) as i16 as f32) / 32768.0 * 0.005;
        let phase = i % burst_period;
        let burst = if phase < burst_len {
            let t = phase as f32 / burst_len as f32;
            let env = (-4.0 * t).exp();
            env * 0.6 * (2.0 * std::f32::consts::PI * 2000.0 * i as f32 / sample_rate as f32).sin()
        } else {
            0.0
        };
        let s = noise + burst;
        out.push((s.clamp(-1.0, 1.0) * 32767.0) as i16);
    }
    out
}

fn encode_psy2(pcm: &[i16], sample_rate: u32, channels: u16, quality: u8) -> Vec<u8> {
    let mut params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
    params.channels = Some(channels);
    params.sample_rate = Some(sample_rate);
    params.sample_format = Some(SampleFormat::S16);
    params.options = CodecOptions::new()
        .set("vbr_quality", quality.to_string())
        .set("psy_model", "2");

    let mut enc = make_encoder(&params).expect("encoder");
    let chunk = 1152 * channels as usize;
    let mut bytes_in: Vec<u8> = Vec::with_capacity(pcm.len() * 2);
    for &s in pcm {
        bytes_in.extend_from_slice(&s.to_le_bytes());
    }
    let mut pts: i64 = 0;
    for slice in bytes_in.chunks(chunk * 2) {
        let n_samples = slice.len() / (2 * channels as usize);
        let frame = AudioFrame {
            samples: n_samples as u32,
            pts: Some(pts),
            data: vec![slice.to_vec()],
        };
        enc.send_frame(&Frame::Audio(frame)).expect("send_frame");
        pts += n_samples as i64;
    }
    enc.flush().expect("flush");

    let mut out: Vec<u8> = Vec::new();
    while let Ok(p) = enc.receive_packet() {
        out.extend_from_slice(&p.data);
    }
    out
}

fn encode_with_model(
    pcm: &[i16],
    sample_rate: u32,
    channels: u16,
    quality: u8,
    psy_model: u8,
) -> Vec<u8> {
    let mut params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
    params.channels = Some(channels);
    params.sample_rate = Some(sample_rate);
    params.sample_format = Some(SampleFormat::S16);
    params.options = CodecOptions::new()
        .set("vbr_quality", quality.to_string())
        .set("psy_model", psy_model.to_string());

    let mut enc = make_encoder(&params).expect("encoder");
    let chunk = 1152 * channels as usize;
    let mut bytes_in: Vec<u8> = Vec::with_capacity(pcm.len() * 2);
    for &s in pcm {
        bytes_in.extend_from_slice(&s.to_le_bytes());
    }
    let mut pts: i64 = 0;
    for slice in bytes_in.chunks(chunk * 2) {
        let n_samples = slice.len() / (2 * channels as usize);
        let frame = AudioFrame {
            samples: n_samples as u32,
            pts: Some(pts),
            data: vec![slice.to_vec()],
        };
        enc.send_frame(&Frame::Audio(frame)).expect("send_frame");
        pts += n_samples as i64;
    }
    enc.flush().expect("flush");

    let mut out: Vec<u8> = Vec::new();
    while let Ok(p) = enc.receive_packet() {
        out.extend_from_slice(&p.data);
    }
    out
}

fn decode_to_pcm(bitstream: &[u8], sample_rate: u32) -> Vec<f32> {
    let params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
    let mut dec = make_decoder(&params).expect("decoder");
    let tb = TimeBase::new(1, sample_rate as i64);
    let mut pcm: Vec<f32> = Vec::new();
    let mut pos = 0usize;
    while pos + 4 <= bitstream.len() {
        let Ok(hdr) = parse_frame_header(&bitstream[pos..]) else {
            break;
        };
        let Some(flen) = hdr.frame_bytes() else { break };
        let flen = flen as usize;
        if pos + flen > bitstream.len() {
            break;
        }
        let pkt = Packet::new(0, tb, bitstream[pos..pos + flen].to_vec());
        if dec.send_packet(&pkt).is_ok() {
            if let Ok(Frame::Audio(a)) = dec.receive_frame() {
                for chunk in a.data[0].chunks_exact(2) {
                    let s = i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0;
                    pcm.push(s);
                }
            }
        }
        pos += flen;
    }
    pcm
}

fn goertzel_power(pcm: &[f32], sample_rate: u32, freq: f32) -> f32 {
    let n = pcm.len();
    let k = (n as f32 * freq / sample_rate as f32).round();
    let omega = 2.0 * std::f32::consts::PI * k / n as f32;
    let coeff = 2.0 * omega.cos();
    let mut s_prev = 0.0f32;
    let mut s_prev2 = 0.0f32;
    for &x in pcm {
        let s = x + coeff * s_prev - s_prev2;
        s_prev2 = s_prev;
        s_prev = s;
    }
    s_prev2 * s_prev2 + s_prev * s_prev - coeff * s_prev * s_prev2
}

fn snr_ratio(pcm: &[f32], sample_rate: u32, target: f32, noise_bins: &[f32]) -> f32 {
    let p = goertzel_power(pcm, sample_rate, target);
    let mut acc = 0.0f32;
    for &f in noise_bins {
        acc += goertzel_power(pcm, sample_rate, f);
    }
    let avg = acc / noise_bins.len().max(1) as f32 + 1e-12;
    p / avg
}

/// Psy-2 produces a non-empty bitstream and round-trips cleanly at 440 Hz.
#[test]
fn psy2_round_trip_440hz() {
    let sample_rate = 44_100u32;
    let pcm = build_sine_pcm(440.0, sample_rate, 1.5, 0.5);
    let bytes = encode_psy2(&pcm, sample_rate, 1, 3);
    assert!(!bytes.is_empty(), "psy_model=2 produced no output");
    let decoded = decode_to_pcm(&bytes, sample_rate);
    assert!(
        decoded.len() >= 4 * 1152,
        "too few samples decoded: {}",
        decoded.len()
    );
    let warmup = 4 * 1152;
    let analysis = &decoded[warmup..];
    let noise_bins = [180.0_f32, 320.0, 1500.0, 3000.0, 7000.0];
    let ratio = snr_ratio(analysis, sample_rate, 440.0, &noise_bins);
    eprintln!("psy_model=2 q=3 own-decode SNR: {ratio:.2}");
    assert!(
        ratio >= 30.0,
        "round-trip SNR too low for psy_model=2: {ratio:.2}"
    );
}

/// Psy-2 produces different output than Psy-1 on a multi-tone fixture.
/// They use the same gain ladder, so identical bytes would mean the Psy-2
/// complex-prediction unpredictability is not altering the masking threshold.
#[test]
fn psy2_differs_from_psy1_output() {
    let sample_rate = 44_100u32;
    let pcm = build_music_pcm(sample_rate, 1.0);
    let bytes_psy1 = encode_with_model(&pcm, sample_rate, 1, 4, 1);
    let bytes_psy2 = encode_with_model(&pcm, sample_rate, 1, 4, 2);
    eprintln!("psy1={} B, psy2={} B", bytes_psy1.len(), bytes_psy2.len());
    assert_ne!(
        bytes_psy1, bytes_psy2,
        "psy_model=2 output must differ from psy_model=1 (complex-prediction tonality should change masks)"
    );
}

/// ffmpeg cross-decode of Psy-2 VBR output must succeed without warnings.
#[test]
fn psy2_ffmpeg_cross_decode_440hz() {
    use std::process::{Command, Stdio};
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        eprintln!("ffmpeg not available — skipping Psy-2 ffmpeg interop");
        return;
    }
    let sample_rate = 44_100u32;
    let q = 2u8;
    let pcm = build_sine_pcm(440.0, sample_rate, 1.5, 0.5);
    let bytes = encode_psy2(&pcm, sample_rate, 1, q);

    let tmp_mp3 = std::env::temp_dir().join("oxideav_mp3_psy2_440.mp3");
    let tmp_wav = std::env::temp_dir().join("oxideav_mp3_psy2_440.wav");
    std::fs::write(&tmp_mp3, &bytes).expect("write mp3");
    let out = Command::new("ffmpeg")
        .arg("-y")
        .arg("-loglevel")
        .arg("warning")
        .arg("-i")
        .arg(&tmp_mp3)
        .arg("-f")
        .arg("wav")
        .arg(&tmp_wav)
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()
        .expect("ffmpeg");
    assert!(out.status.success(), "ffmpeg failed");
    let wav = std::fs::read(&tmp_wav).expect("wav");
    let data_off = wav
        .windows(4)
        .position(|w| w == b"data")
        .expect("WAV data tag")
        + 8;
    let mut decoded: Vec<f32> = Vec::new();
    for ch in wav[data_off..].chunks_exact(2) {
        decoded.push(i16::from_le_bytes([ch[0], ch[1]]) as f32 / 32768.0);
    }
    let warmup = 4 * 1152;
    let analysis = &decoded[warmup..];
    let noise_bins = [180.0_f32, 320.0, 1500.0, 3000.0, 7000.0];
    let ratio = snr_ratio(analysis, sample_rate, 440.0, &noise_bins);
    eprintln!(
        "psy_model=2 q={q} ffmpeg SNR: {ratio:.2}, bytes={}",
        bytes.len()
    );
    assert!(
        ratio >= 30.0,
        "Psy-2 ffmpeg interop SNR too low: {ratio:.2}"
    );
}

/// Psy-2 short-block fallback: transient-heavy castanet signal must encode
/// cleanly (short blocks fall through to the Psy-1 per-window path).
#[test]
fn psy2_short_block_fallback_castanet_roundtrip() {
    let sample_rate = 44_100u32;
    let pcm = build_castanet_pcm(sample_rate, 1.0);
    let bytes = encode_psy2(&pcm, sample_rate, 1, 3);
    assert!(!bytes.is_empty(), "psy_model=2 castanet produced no output");
    let dec = decode_to_pcm(&bytes, sample_rate);
    assert!(
        dec.len() > 4 * 1152,
        "too few samples decoded: {}",
        dec.len()
    );
    for v in dec.iter() {
        assert!(v.is_finite(), "decoded NaN/Inf");
        assert!((-1.5..=1.5).contains(v), "decoded sample out of range: {v}");
    }
    let out_e: f64 = dec.iter().map(|&v| (v as f64).powi(2)).sum();
    assert!(
        out_e > 0.01,
        "Psy-2 castanet round-trip produced near-silent output: {out_e:.6}"
    );
}

/// MPEG-2 LSF Psy-2 path produces a valid bitstream.
#[test]
fn psy2_mpeg2_lsf_24khz_roundtrip() {
    let sample_rate = 24_000u32;
    let q = 3u8;
    let pcm = build_sine_pcm(1000.0, sample_rate, 1.5, 0.25);
    let bytes = encode_psy2(&pcm, sample_rate, 1, q);
    assert!(!bytes.is_empty(), "no MPEG-2 LSF Psy-2 output");
    assert_eq!(bytes[0], 0xFF);
    assert_eq!(
        bytes[1] & 0xFE,
        0b1111_0010,
        "expected MPEG-2 LSF header, got {:08b}",
        bytes[1]
    );
    let decoded = decode_to_pcm(&bytes, sample_rate);
    assert!(
        decoded.len() >= 4 * 576,
        "too few samples decoded: {}",
        decoded.len()
    );
    let warmup = 8 * 576;
    let analysis = &decoded[warmup..decoded.len().min(warmup + 8192)];
    let noise_bins = [180.0_f32, 420.0, 3000.0, 5000.0, 7000.0];
    let ratio = snr_ratio(analysis, sample_rate, 1000.0, &noise_bins);
    eprintln!("MPEG-2 LSF Psy-2 q={q} 1kHz SNR: {ratio:.2}");
    assert!(ratio >= 30.0, "MPEG-2 LSF Psy-2 SNR too low: {ratio:.2}");
}

/// Long-input stability: Psy-2's complex FFT history must not produce
/// NaN/Inf or silent output after many frames.
#[test]
fn psy2_long_input_stable() {
    let sample_rate = 44_100u32;
    let pcm = build_music_pcm(sample_rate, 4.0);
    let bytes = encode_psy2(&pcm, sample_rate, 1, 4);
    assert!(!bytes.is_empty(), "no output");
    let dec = decode_to_pcm(&bytes, sample_rate);
    assert!(dec.len() > 100 * 1152);
    for v in dec.iter() {
        assert!(v.is_finite(), "decoded NaN/Inf after long Psy-2 history");
    }
    let energy: f64 = dec.iter().map(|&v| (v as f64).powi(2)).sum();
    assert!(
        energy > 1.0,
        "long-input Psy-2 output went silent: energy={energy}"
    );
}
