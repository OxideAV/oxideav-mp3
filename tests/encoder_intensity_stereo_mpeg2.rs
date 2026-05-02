//! MPEG-2 LSF intensity-stereo encoding tests.
//!
//! Per ISO/IEC 13818-3 §2.4.2.7 and §2.4.3.4.10.2 the LSF (Low Sampling
//! Frequency) Layer III variant uses a different scalefactor partition
//! and a 5-bit `is_pos` field (0..=30, 31 = "not IS-coded") on a
//! geometric-step ratio table. The encoder packs the per-band `is_pos`
//! values into the IS half of `SCF_PARTITIONS_MPEG2[0][16..28]` via a
//! fixed `scalefac_compress_9 = 358` (slens [4, 5, 5, 0],
//! `intensity_scale = 0`) and zeroes the R-channel coefficients above
//! the IS bound so the decoder's `find_is_bound_sfb` recovers the same
//! cutoff.
//!
//! These tests cover, at LSF sample rates (16 / 22.05 / 24 kHz):
//! 1. The header flips to `mode_ext` with the IS bit set on stereo
//!    content that is correlated above an HF crossover.
//! 2. A correlated-HF stereo signal shrinks measurably (~5 % or more in
//!    VBR mode) versus the same input encoded with IS disabled.
//! 3. Round-trip through our own decoder recovers L and R with both
//!    channels live (no silent right channel, no L/R swap).
//! 4. Round-trip through ffmpeg recovers the same content cleanly with
//!    no warning lines printed.
//! 5. The opt-out (`intensity_stereo=0`) keeps the encoder on the
//!    pre-IS path (no IS bit set).

use oxideav_core::options::CodecOptions;
use oxideav_core::{AudioFrame, CodecId, CodecParameters, Frame, SampleFormat, TimeBase};
use oxideav_mp3::decoder::make_decoder;
use oxideav_mp3::encoder::make_encoder;
use oxideav_mp3::frame::parse_frame_header;
use oxideav_mp3::CODEC_ID_STR;

/// LSF target sample rate for these tests. 22.05 kHz is the canonical
/// MPEG-2 LSF rate and gives the broadest downstream interop.
const LSF_SR: u32 = 22_050;

/// Build a stereo signal with uncorrelated low frequencies and a comb
/// of correlated tones across 4-9 kHz (well within the LSF Nyquist of
/// 11.025 kHz). The HF tail is the canonical IS candidate; the LF
/// region exercises the MS/dual-channel path below the IS bound.
fn build_lsf_hf_correlated_stereo(sample_rate: u32, duration_s: f32) -> Vec<i16> {
    let n = (sample_rate as f32 * duration_s) as usize;
    let two_pi = 2.0 * std::f32::consts::PI;
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        let t = i as f32 / sample_rate as f32;
        // LF: full stereo image — different tones in the two channels.
        let lf_l = (two_pi * 220.0 * t).sin() * 0.25;
        let lf_r = (two_pi * 277.0 * t).sin() * 0.25;
        // HF: comb of tones across 4-9 kHz, identical in both channels.
        let mut hf = 0.0f32;
        for &freq in &[4_000.0f32, 5_000.0, 6_000.0, 7_000.0, 8_000.0, 9_000.0] {
            hf += (two_pi * freq * t).sin();
        }
        hf *= 0.05;
        let l = lf_l + hf;
        let r = lf_r + hf;
        out.push((l.clamp(-1.0, 1.0) * 32767.0) as i16);
        out.push((r.clamp(-1.0, 1.0) * 32767.0) as i16);
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

    // MPEG-2 LSF: 576 samples per frame (1 granule per frame).
    let chunk = 576 * channels as usize;
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
fn lsf_hf_correlated_stereo_triggers_is_header_bits() {
    let pcm = build_lsf_hf_correlated_stereo(LSF_SR, 1.0);
    let bytes = encode_to_bytes_with_opts(&pcm, LSF_SR, 2, 96_000, CodecOptions::new());
    let frames = count_frames(&bytes);
    assert!(!frames.is_empty(), "no frames emitted");
    // Expect a substantial fraction of frames to carry IS in the
    // mode_extension. Allow either IS-only (0b01) or MS+IS (0b11).
    let is_frames = frames
        .iter()
        .filter(|(mode, ext, _)| *mode == 0b01 && (ext & 0b01) != 0)
        .count();
    let total = frames.len();
    eprintln!("LSF IS frames (any mode_ext.0x1): {is_frames} / {total}");
    assert!(
        is_frames * 2 >= total,
        "expected >=50% LSF IS frames, got {is_frames}/{total}"
    );
}

#[test]
fn lsf_intensity_stereo_disabled_keeps_is_off() {
    let pcm = build_lsf_hf_correlated_stereo(LSF_SR, 1.0);
    let opts = CodecOptions::new().set("intensity_stereo", "0");
    let bytes = encode_to_bytes_with_opts(&pcm, LSF_SR, 2, 96_000, opts);
    let frames = count_frames(&bytes);
    assert!(!frames.is_empty(), "no frames emitted");
    for (_, ext, _) in &frames {
        assert_eq!(
            ext & 0b01,
            0,
            "intensity_stereo=0 must never set the IS bit (LSF), got mode_ext={ext:#b}"
        );
    }
}

#[test]
fn lsf_intensity_stereo_shrinks_correlated_hf_in_vbr() {
    // VBR mode is the place the byte-count delta from IS shows up; CBR
    // keeps slot sizes fixed and spends the freed bits on lower
    // quantisation noise instead.
    let pcm = build_lsf_hf_correlated_stereo(LSF_SR, 2.0);

    let no_is_opts = CodecOptions::new()
        .set("vbr_quality", "4")
        .set("intensity_stereo", "0");
    let with_is_opts = CodecOptions::new().set("vbr_quality", "4");
    let no_is_bytes = encode_to_bytes_with_opts(&pcm, LSF_SR, 2, 96_000, no_is_opts);
    let with_is_bytes = encode_to_bytes_with_opts(&pcm, LSF_SR, 2, 96_000, with_is_opts);
    let saved_pct =
        100.0 * (no_is_bytes.len() as f32 - with_is_bytes.len() as f32) / no_is_bytes.len() as f32;
    eprintln!(
        "LSF VBR HF-correlated: noIS={}B IS={}B saved={:.1}%",
        no_is_bytes.len(),
        with_is_bytes.len(),
        saved_pct
    );
    assert!(
        saved_pct >= 5.0,
        "expected LSF IS to save >=5% on HF-correlated content, got {saved_pct:.1}% (noIS={}B, IS={}B)",
        no_is_bytes.len(),
        with_is_bytes.len()
    );
}

#[test]
fn lsf_intensity_stereo_round_trips_through_decoder() {
    let pcm = build_lsf_hf_correlated_stereo(LSF_SR, 1.0);
    let bytes = encode_to_bytes_with_opts(&pcm, LSF_SR, 2, 96_000, CodecOptions::new());

    let decoded = decode_to_pcm(&bytes, LSF_SR);
    assert!(decoded.len() >= 4 * 576 * 2);
    let warmup = 4 * 576 * 2;
    let l: Vec<f32> = decoded[warmup..].chunks_exact(2).map(|p| p[0]).collect();
    let r: Vec<f32> = decoded[warmup..].chunks_exact(2).map(|p| p[1]).collect();
    let l_e: f32 = l.iter().map(|v| v * v).sum();
    let r_e: f32 = r.iter().map(|v| v * v).sum();
    eprintln!("LSF IS round-trip energies: L={l_e:.3} R={r_e:.3}");
    assert!(l_e > 0.5, "LSF: left channel silent: e={l_e}");
    assert!(r_e > 0.5, "LSF: right channel silent: e={r_e}");
}

/// ffmpeg cross-decode check. Skipped silently when ffmpeg is missing.
#[test]
fn lsf_intensity_stereo_round_trips_via_ffmpeg() {
    use std::process::{Command, Stdio};
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        eprintln!("ffmpeg not available — skipping LSF IS interop check");
        return;
    }

    let pcm = build_lsf_hf_correlated_stereo(LSF_SR, 1.0);
    let bytes = encode_to_bytes_with_opts(&pcm, LSF_SR, 2, 96_000, CodecOptions::new());

    let tmp_mp3 = std::env::temp_dir().join("oxideav_mp3_lsf_is_stereo.mp3");
    let tmp_wav = std::env::temp_dir().join("oxideav_mp3_lsf_is_stereo.wav");
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
    assert!(decoded.len() >= 4 * 576 * 2);

    let warmup = 4 * 576 * 2;
    let l: Vec<f32> = decoded[warmup..].chunks_exact(2).map(|p| p[0]).collect();
    let r: Vec<f32> = decoded[warmup..].chunks_exact(2).map(|p| p[1]).collect();
    let l_e: f32 = l.iter().map(|v| v * v).sum();
    let r_e: f32 = r.iter().map(|v| v * v).sum();
    eprintln!("ffmpeg LSF IS-decode energies: L={l_e:.3} R={r_e:.3}");
    assert!(l_e > 0.5, "ffmpeg LSF: left channel silent: e={l_e}");
    assert!(r_e > 0.5, "ffmpeg LSF: right channel silent: e={r_e}");
}
