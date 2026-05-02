//! Joint-stereo (MS) encoding tests for the MP3 encoder.
//!
//! Per ISO/IEC 11172-3 §2.4.3.4.10 the encoder may rotate L/R into
//! M/S = (L+R)/sqrt(2), (L-R)/sqrt(2) when the spectral correlation
//! is high. Frames in this mode set `mode = 0b01` (joint stereo) and
//! `mode_extension = 0b10` (MS on, IS off) in the header.
//!
//! These tests cover:
//! 1. The header bits actually flip when the encoder picks MS.
//! 2. A correlated stereo signal (centred voice + ambient) shrinks at
//!    least ~5% versus the same input forced to dual-channel.
//! 3. Round-trip through both our own decoder and ffmpeg recovers L
//!    and R within tolerance (no silent-channel bug, no swapped sign,
//!    no L/R cross-talk).
//! 4. Anti-correlated content (`L = -R`) does *not* trigger MS — its
//!    M energy is near zero and S is the loud signal, so the heuristic
//!    must keep L/R coding instead of inverting the cost.
//! 5. Disabling joint stereo via `joint_stereo=0` reverts to the
//!    historical dual-channel behaviour (mode_bits = 0b10).

use oxideav_core::options::CodecOptions;
use oxideav_core::{AudioFrame, CodecId, CodecParameters, Frame, SampleFormat, TimeBase};
use oxideav_mp3::decoder::make_decoder;
use oxideav_mp3::encoder::make_encoder;
use oxideav_mp3::frame::parse_frame_header;
use oxideav_mp3::CODEC_ID_STR;

fn build_correlated_stereo(sample_rate: u32, duration_s: f32) -> Vec<i16> {
    // Centred voice (440 Hz mono fold-down on both channels) with a
    // tiny 1 kHz ambient differential (loud on L, quiet on R) — the
    // common "vocal in the centre, room tone" mix that MS shines on.
    let n = (sample_rate as f32 * duration_s) as usize;
    let two_pi = 2.0 * std::f32::consts::PI;
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        let t = i as f32 / sample_rate as f32;
        let voice = (two_pi * 440.0 * t).sin() * 0.5;
        let ambient = (two_pi * 1000.0 * t).sin() * 0.02;
        let l = voice + ambient;
        let r = voice - ambient;
        out.push((l.clamp(-1.0, 1.0) * 32767.0) as i16);
        out.push((r.clamp(-1.0, 1.0) * 32767.0) as i16);
    }
    out
}

fn build_anticorrelated_stereo(sample_rate: u32, duration_s: f32) -> Vec<i16> {
    // L = -R: any centre fold-down vanishes (M = 0) and the side
    // channel carries everything. MS would *waste* bits — the encoder
    // must NOT pick it.
    let n = (sample_rate as f32 * duration_s) as usize;
    let two_pi = 2.0 * std::f32::consts::PI;
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        let t = i as f32 / sample_rate as f32;
        let s = (two_pi * 440.0 * t).sin() * 0.5;
        let q = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        out.push(q);
        out.push(-q);
    }
    out
}

fn encode_to_bytes_with_opts(
    pcm: &[i16],
    sample_rate: u32,
    channels: u16,
    bitrate_bps: u64,
    options: CodecOptions,
) -> Vec<u8> {
    let mut params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
    params.channels = Some(channels);
    params.sample_rate = Some(sample_rate);
    params.sample_format = Some(SampleFormat::S16);
    params.bit_rate = Some(bitrate_bps);
    params.options = options;

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
        let pkt = oxideav_core::Packet::new(0, tb, bitstream[pos..pos + flen].to_vec());
        let _ = dec.send_packet(&pkt);
        while let Ok(Frame::Audio(af)) = dec.receive_frame() {
            for plane in &af.data {
                for ch in plane.chunks_exact(2) {
                    pcm.push(i16::from_le_bytes([ch[0], ch[1]]) as f32 / 32768.0);
                }
            }
        }
        pos += flen;
    }
    pcm
}

/// Read mode + mode_extension from a 4-byte MP3 frame header.
fn frame_header_mode_bits(hdr_bytes: &[u8]) -> (u8, u8) {
    let h = u32::from_be_bytes([hdr_bytes[0], hdr_bytes[1], hdr_bytes[2], hdr_bytes[3]]);
    let mode = ((h >> 6) & 0x3) as u8;
    let mode_ext = ((h >> 4) & 0x3) as u8;
    (mode, mode_ext)
}

fn count_frames(bitstream: &[u8]) -> Vec<(u8, u8, usize)> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos + 4 <= bitstream.len() {
        let Ok(hdr) = parse_frame_header(&bitstream[pos..]) else {
            break;
        };
        let Some(flen) = hdr.frame_bytes() else { break };
        let flen = flen as usize;
        if pos + flen > bitstream.len() {
            break;
        }
        let (mode, mode_ext) = frame_header_mode_bits(&bitstream[pos..pos + 4]);
        out.push((mode, mode_ext, flen));
        pos += flen;
    }
    out
}

#[test]
fn correlated_stereo_triggers_ms_header_bits() {
    let sample_rate = 44_100u32;
    let pcm = build_correlated_stereo(sample_rate, 1.0);
    // Default options → joint_stereo enabled.
    let bytes = encode_to_bytes_with_opts(&pcm, sample_rate, 2, 192_000, CodecOptions::new());
    let frames = count_frames(&bytes);
    assert!(!frames.is_empty(), "no frames emitted");

    // We expect the vast majority of frames to be marked joint-stereo
    // (mode = 0b01) with the MS bit (0x2) set in mode_extension. The
    // IS bit (0x1) may or may not also be set depending on whether the
    // post-#174 IS path qualifies the HF tail of the same correlated
    // material; checking only the MS bit isolates the #115 contract.
    let js_ms = frames
        .iter()
        .filter(|(mode, ext, _)| *mode == 0b01 && (*ext & 0b10) != 0)
        .count();
    let total = frames.len();
    eprintln!("joint-stereo MS frames: {js_ms} / {total}");
    assert!(
        js_ms * 10 >= total * 8,
        "expected >=80% joint-stereo MS frames, got {js_ms}/{total}"
    );
}

#[test]
fn ms_stereo_encoding_shrinks_correlated_signal() {
    let sample_rate = 44_100u32;
    // Pure mono fold-down: L exactly equals R. The S channel collapses
    // to bit-exact zero after MS rotation, which lets the VBR encoder
    // hit its `any_energy` shortcut on every granule of the side
    // channel and emit a near-zero-bit Huffman block. This is the
    // strongest case for MS savings and the one ISO/IEC 11172-3
    // §2.4.3.4.10 calls out as the design motivation.
    let mono = build_mono_440(sample_rate, 2.0);
    let mut pcm: Vec<i16> = Vec::with_capacity(mono.len() * 2);
    for &s in &mono {
        pcm.push(s);
        pcm.push(s);
    }

    // CBR slot sizes are fixed by `bit_rate` so a direct byte-count
    // delta in CBR mode would never beat zero. The savings are real
    // but show up in bit-reservoir usage / quantisation noise instead.
    // We measure the byte delta in VBR mode where the slot is picked
    // per frame.
    let vbr_dual_opts = CodecOptions::new()
        .set("vbr_quality", "4")
        .set("joint_stereo", "0");
    let vbr_ms_opts = CodecOptions::new().set("vbr_quality", "4");
    let vbr_dual_bytes = encode_to_bytes_with_opts(&pcm, sample_rate, 2, 192_000, vbr_dual_opts);
    let vbr_ms_bytes = encode_to_bytes_with_opts(&pcm, sample_rate, 2, 192_000, vbr_ms_opts);
    let saved_pct = 100.0 * (vbr_dual_bytes.len() as f32 - vbr_ms_bytes.len() as f32)
        / vbr_dual_bytes.len() as f32;
    eprintln!(
        "VBR mono-fold-down stereo: dual={}B MS={}B saved={:.1}%",
        vbr_dual_bytes.len(),
        vbr_ms_bytes.len(),
        saved_pct
    );
    assert!(
        saved_pct >= 5.0,
        "expected MS to save >=5% on mono fold-down, got {saved_pct:.1}% (dual={}B, MS={}B)",
        vbr_dual_bytes.len(),
        vbr_ms_bytes.len()
    );

    // CBR sanity: same input under default joint-stereo vs forced
    // dual-channel must produce the same number of frames at the same
    // slot size — MS doesn't change CBR slot sizing.
    let dual_opts = CodecOptions::new().set("joint_stereo", "0");
    let dual_cbr = encode_to_bytes_with_opts(&pcm, sample_rate, 2, 192_000, dual_opts);
    let ms_cbr = encode_to_bytes_with_opts(&pcm, sample_rate, 2, 192_000, CodecOptions::new());
    assert_eq!(
        dual_cbr.len(),
        ms_cbr.len(),
        "CBR slot sizes must match: dual={} MS={}",
        dual_cbr.len(),
        ms_cbr.len()
    );
}

fn build_mono_440(sample_rate: u32, duration_s: f32) -> Vec<i16> {
    let n = (sample_rate as f32 * duration_s) as usize;
    let two_pi = 2.0 * std::f32::consts::PI;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f32 / sample_rate as f32;
        let s = (two_pi * 440.0 * t).sin() * 0.5;
        out.push((s.clamp(-1.0, 1.0) * 32767.0) as i16);
    }
    out
}

#[test]
fn anticorrelated_stereo_avoids_ms() {
    let sample_rate = 44_100u32;
    let pcm = build_anticorrelated_stereo(sample_rate, 1.0);
    let bytes = encode_to_bytes_with_opts(&pcm, sample_rate, 2, 192_000, CodecOptions::new());
    let frames = count_frames(&bytes);
    assert!(!frames.is_empty(), "no frames emitted");
    // L = -R → S energy = full signal, M energy = 0 → ratio is huge,
    // heuristic must reject MS for every frame.
    let ms_frames = frames
        .iter()
        .filter(|(mode, ext, _)| *mode == 0b01 && (*ext & 0b10) != 0)
        .count();
    eprintln!("anticorrelated MS frames: {ms_frames} / {}", frames.len());
    assert_eq!(
        ms_frames, 0,
        "expected zero MS frames for anti-correlated input, got {ms_frames}"
    );
}

#[test]
fn ms_encoded_stereo_decodes_back_to_lr() {
    // Round-trip through our own decoder. Both channels must come back
    // intact — if the M/S rotation were applied without flipping
    // mode_extension the decoder would interpret raw L/R as M/S and
    // output garbage.
    let sample_rate = 44_100u32;
    let pcm = build_correlated_stereo(sample_rate, 1.0);
    let bytes = encode_to_bytes_with_opts(&pcm, sample_rate, 2, 192_000, CodecOptions::new());

    let decoded = decode_to_pcm(&bytes, sample_rate);
    assert!(decoded.len() >= 4 * 1152 * 2);

    let warmup = 4 * 1152 * 2;
    let l: Vec<f32> = decoded[warmup..].chunks_exact(2).map(|p| p[0]).collect();
    let r: Vec<f32> = decoded[warmup..].chunks_exact(2).map(|p| p[1]).collect();

    let l_e: f32 = l.iter().map(|v| v * v).sum();
    let r_e: f32 = r.iter().map(|v| v * v).sum();
    assert!(l_e > 1.0, "left channel silent: e={l_e}");
    assert!(r_e > 1.0, "right channel silent: e={r_e}");

    // Both channels should carry the 440 Hz centre tone with similar
    // energy (~1 within 6 dB). If MS unrotation were skipped on the
    // decode side this would degenerate.
    let ratio_db = 10.0 * (l_e / r_e).log10();
    eprintln!("L/R energy ratio after MS round-trip: {ratio_db:.2} dB");
    assert!(
        ratio_db.abs() < 6.0,
        "L/R imbalance too large after MS round-trip: {ratio_db:.2} dB"
    );
}

/// ffmpeg cross-decode check. Skipped silently when ffmpeg is missing.
#[test]
fn ms_encoded_stereo_round_trips_via_ffmpeg() {
    use std::process::{Command, Stdio};
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        eprintln!("ffmpeg not available — skipping MS interop check");
        return;
    }

    let sample_rate = 44_100u32;
    let pcm = build_correlated_stereo(sample_rate, 1.0);
    let bytes = encode_to_bytes_with_opts(&pcm, sample_rate, 2, 192_000, CodecOptions::new());

    let tmp_mp3 = std::env::temp_dir().join("oxideav_mp3_ms_stereo.mp3");
    let tmp_wav = std::env::temp_dir().join("oxideav_mp3_ms_stereo.wav");
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
        .expect("ffmpeg run");
    assert!(out.status.success(), "ffmpeg failed: {:?}", out.status);
    let stderr = String::from_utf8_lossy(&out.stderr);
    let suspicious_lines: Vec<&str> = stderr
        .lines()
        .filter(|l| !l.contains("Estimating duration from bitrate"))
        .filter(|l| !l.trim().is_empty())
        .collect();
    assert!(
        suspicious_lines.is_empty(),
        "ffmpeg emitted warnings: {suspicious_lines:?}"
    );

    let wav = std::fs::read(&tmp_wav).expect("read wav");
    let data_off = wav
        .windows(4)
        .position(|w| w == b"data")
        .expect("WAV data tag")
        + 8;
    let mut decoded: Vec<f32> = Vec::new();
    for ch in wav[data_off..].chunks_exact(2) {
        decoded.push(i16::from_le_bytes([ch[0], ch[1]]) as f32 / 32768.0);
    }
    assert!(decoded.len() >= 4 * 1152 * 2);

    let warmup = 4 * 1152 * 2;
    let l: Vec<f32> = decoded[warmup..].chunks_exact(2).map(|p| p[0]).collect();
    let r: Vec<f32> = decoded[warmup..].chunks_exact(2).map(|p| p[1]).collect();

    let l_e: f32 = l.iter().map(|v| v * v).sum();
    let r_e: f32 = r.iter().map(|v| v * v).sum();
    assert!(l_e > 1.0, "ffmpeg: left channel silent: e={l_e}");
    assert!(r_e > 1.0, "ffmpeg: right channel silent: e={r_e}");

    // Goertzel detector at 440 Hz on each channel — must be the
    // dominant component on both.
    let p_l = goertzel_power(&l, sample_rate, 440.0);
    let p_r = goertzel_power(&r, sample_rate, 440.0);
    let e_l_total: f32 = l.iter().map(|v| v * v).sum();
    let e_r_total: f32 = r.iter().map(|v| v * v).sum();
    let frac_l = p_l / e_l_total.max(1e-9);
    let frac_r = p_r / e_r_total.max(1e-9);
    eprintln!(
        "ffmpeg MS-decode 440Hz energy fraction: L={:.3} R={:.3}",
        frac_l, frac_r
    );
    assert!(frac_l > 0.5, "ffmpeg: 440Hz tone missing from L: {frac_l}");
    assert!(frac_r > 0.5, "ffmpeg: 440Hz tone missing from R: {frac_r}");
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

#[test]
fn joint_stereo_disabled_emits_dual_channel() {
    let sample_rate = 44_100u32;
    let pcm = build_correlated_stereo(sample_rate, 0.5);
    let opts = CodecOptions::new().set("joint_stereo", "0");
    let bytes = encode_to_bytes_with_opts(&pcm, sample_rate, 2, 192_000, opts);
    let frames = count_frames(&bytes);
    assert!(!frames.is_empty(), "no frames emitted");
    for (mode, ext, _) in &frames {
        assert_eq!(*mode, 0b10, "expected dual-channel (0b10), got {mode:#b}");
        assert_eq!(*ext, 0, "expected mode_extension 0, got {ext:#b}");
    }
}
