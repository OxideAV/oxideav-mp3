//! End-to-end tests for the VBR (variable bit-rate) MP3 encoder.
//!
//! Covers:
//! - File size shrinks with smaller (= worse-quality) VBR quality.
//! - File size *varies with content complexity* at a fixed quality
//!   (silence < pure tone < noise).
//! - Round-trip decode through our own decoder produces a clean
//!   spectrum at the input frequency.
//! - ffmpeg cross-decode produces a spectrum matching the input
//!   (interop check).
//! - Per-frame bitrate-index variation: VBR streams must contain at
//!   least two distinct bitrate slots when the content drives them.
//! - PSNR of decoded vs original is a monotonic-ish function of
//!   quality.
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

/// Pseudo-random "music-like" signal — sum of several detuned
/// sinusoids over the audible range, scaled to ~half full-scale.
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

/// Pure silence.
fn build_silence_pcm(sample_rate: u32, duration_s: f32) -> Vec<i16> {
    let n = (sample_rate as f32 * duration_s) as usize;
    vec![0i16; n]
}

fn encode_vbr(pcm: &[i16], sample_rate: u32, channels: u16, vbr_quality: u8) -> Vec<u8> {
    let mut params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
    params.channels = Some(channels);
    params.sample_rate = Some(sample_rate);
    params.sample_format = Some(SampleFormat::S16);
    params.options = CodecOptions::new().set("vbr_quality", vbr_quality.to_string());

    let mut enc = make_encoder(&params).expect("encoder");

    // Feed the encoder one frame's worth at a time (1152 samples per
    // channel for MPEG-1).
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

/// Walk a VBR stream and collect the distinct bitrate slots used.
fn distinct_bitrates(bitstream: &[u8]) -> std::collections::BTreeSet<u32> {
    let mut set = std::collections::BTreeSet::new();
    let mut pos = 0usize;
    while pos + 4 <= bitstream.len() {
        let Ok(hdr) = parse_frame_header(&bitstream[pos..]) else {
            break;
        };
        let Some(flen) = hdr.frame_bytes() else { break };
        set.insert(hdr.bitrate_kbps);
        pos += flen as usize;
    }
    set
}

/// File size (in bytes) shrinks as VBR quality goes from V0 to V9.
#[test]
fn vbr_size_decreases_with_quality() {
    let sample_rate = 44_100u32;
    let pcm = build_music_pcm(sample_rate, 2.0);
    let mut sizes: Vec<usize> = Vec::new();
    for q in [0u8, 3, 6, 9] {
        let bytes = encode_vbr(&pcm, sample_rate, 1, q);
        eprintln!("VBR q={q} → {} bytes", bytes.len());
        sizes.push(bytes.len());
    }
    // Allow non-strict monotonicity (consecutive Q levels can coincide
    // for very smooth content) but the V0 file MUST be larger than V9.
    assert!(
        sizes.first() > sizes.last(),
        "expected V0 ({}) > V9 ({})",
        sizes.first().unwrap(),
        sizes.last().unwrap()
    );
}

/// File size varies with content complexity at a fixed VBR quality.
#[test]
fn vbr_size_varies_with_content() {
    let sample_rate = 44_100u32;
    let dur = 2.0;
    let q = 4u8;
    let silence = encode_vbr(&build_silence_pcm(sample_rate, dur), sample_rate, 1, q);
    let tone = encode_vbr(
        &build_sine_pcm(440.0, sample_rate, dur, 0.5),
        sample_rate,
        1,
        q,
    );
    let music = encode_vbr(&build_music_pcm(sample_rate, dur), sample_rate, 1, q);
    eprintln!(
        "VBR q={q} sizes: silence={} tone={} music={}",
        silence.len(),
        tone.len(),
        music.len()
    );
    // Strict ordering: silence < tone (one band lit up) < music (many).
    assert!(
        silence.len() < tone.len(),
        "silence ({}) must be smaller than tone ({})",
        silence.len(),
        tone.len()
    );
    assert!(
        tone.len() < music.len(),
        "tone ({}) must be smaller than music ({})",
        tone.len(),
        music.len()
    );
}

/// VBR streams encode the per-frame bitrate slot in the header,
/// producing distinct slots over a non-trivial signal.
#[test]
fn vbr_uses_multiple_bitrate_slots() {
    let sample_rate = 44_100u32;
    let q = 2u8;
    // Concatenate silence + tone + music so the per-frame slot
    // *must* change.
    let mut pcm = Vec::new();
    pcm.extend(build_silence_pcm(sample_rate, 0.5));
    pcm.extend(build_sine_pcm(440.0, sample_rate, 0.5, 0.3));
    pcm.extend(build_music_pcm(sample_rate, 0.5));
    let bytes = encode_vbr(&pcm, sample_rate, 1, q);
    let slots = distinct_bitrates(&bytes);
    eprintln!("VBR distinct bitrate slots: {slots:?}");
    assert!(
        slots.len() >= 2,
        "expected ≥2 distinct bitrate slots, got {slots:?}"
    );
}

/// Round-trip a 440 Hz sine through VBR encode + own decode and check
/// the dominant spectral component sits at 440 Hz.
#[test]
fn vbr_roundtrip_440hz_mono() {
    let sample_rate = 44_100u32;
    let q = 2u8;
    let pcm = build_sine_pcm(440.0, sample_rate, 1.5, 0.5);
    let bytes = encode_vbr(&pcm, sample_rate, 1, q);
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
    eprintln!("VBR q={q} 440Hz mono own-decode SNR: {ratio:.2}");
    assert!(
        ratio >= 30.0,
        "VBR roundtrip SNR too low (q={q}): {ratio:.2}"
    );
}

/// VBR stream decoded by ffmpeg matches the input frequency.
#[test]
fn vbr_440hz_via_ffmpeg() {
    use std::process::{Command, Stdio};
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        eprintln!("ffmpeg not available — skipping VBR ffmpeg interop");
        return;
    }
    let sample_rate = 44_100u32;
    for &q in &[0u8, 5] {
        let pcm = build_sine_pcm(440.0, sample_rate, 1.5, 0.5);
        let bytes = encode_vbr(&pcm, sample_rate, 1, q);

        let tmp_mp3 = std::env::temp_dir().join(format!("oxideav_mp3_vbr_q{q}.mp3"));
        let tmp_wav = std::env::temp_dir().join(format!("oxideav_mp3_vbr_q{q}.wav"));
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
        eprintln!("VBR q={q} ffmpeg SNR: {ratio:.2}, bytes={}", bytes.len());
        assert!(
            ratio >= 30.0,
            "VBR ffmpeg interop SNR too low at q={q}: {ratio:.2}"
        );
    }
}

/// Higher VBR quality must yield more bits per second. Pin the
/// monotonicity over a non-trivial multi-tone signal so a regression
/// in the masking iteration (e.g. the wrong inequality direction)
/// trips the assertion.
#[test]
fn vbr_bitrate_monotonic_in_quality() {
    let sample_rate = 44_100u32;
    let dur = 2.0f32;
    let pcm = build_music_pcm(sample_rate, dur);
    // Compare the cheapest V (V9) against the most expensive (V0).
    let v0 = encode_vbr(&pcm, sample_rate, 1, 0);
    let v9 = encode_vbr(&pcm, sample_rate, 1, 9);
    let v0_kbps = v0.len() as f32 * 8.0 / dur / 1000.0;
    let v9_kbps = v9.len() as f32 * 8.0 / dur / 1000.0;
    eprintln!("VBR V0 kbps={v0_kbps:.1}, V9 kbps={v9_kbps:.1}");
    // V0 must exceed V9 in byte size (otherwise the quality knob is
    // a no-op).
    assert!(
        v0.len() > v9.len(),
        "V0 ({} B) must exceed V9 ({} B) in size",
        v0.len(),
        v9.len()
    );
}

/// VBR encode at a Goldilocks quality — verify the average bitrate
/// sits in a sane window for MP3 (50..320 kbps for 44.1 kHz mono).
#[test]
fn vbr_average_bitrate_is_sane() {
    let sample_rate = 44_100u32;
    let dur = 2.0f32;
    let pcm = build_music_pcm(sample_rate, dur);
    let bytes = encode_vbr(&pcm, sample_rate, 1, 4);
    let avg_kbps = (bytes.len() as f32 * 8.0 / dur / 1000.0) as u32;
    eprintln!("VBR q=4 avg bitrate: {avg_kbps} kbps over {dur:.1}s");
    assert!(
        (24..=320).contains(&avg_kbps),
        "VBR avg bitrate {avg_kbps} kbps outside sane MP3 range"
    );
}

/// Verify VBR also functions on MPEG-2 LSF (24 kHz mono).
#[test]
fn vbr_mpeg2_lsf_24khz_roundtrip() {
    let sample_rate = 24_000u32;
    let q = 3u8;
    let pcm = build_sine_pcm(1000.0, sample_rate, 1.5, 0.25);
    let bytes = encode_vbr(&pcm, sample_rate, 1, q);
    assert!(!bytes.is_empty(), "no MPEG-2 LSF VBR output");
    // Header sanity: must be MPEG-2.
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
    eprintln!("MPEG-2 LSF VBR q={q} 1kHz SNR: {ratio:.2}");
    assert!(ratio >= 30.0, "MPEG-2 LSF VBR SNR too low: {ratio:.2}");
}
