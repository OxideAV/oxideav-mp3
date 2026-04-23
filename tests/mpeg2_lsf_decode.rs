//! MPEG-2 LSF decode tests.
//!
//! Hand-crafts minimal MPEG-2 Layer III (LSF) frames and verifies the
//! decoder produces the expected 576-sample-per-granule output. MPEG-2
//! LSF frames have:
//! - `version_id` = 0b10 in the frame header
//! - 1 granule per frame → 576 samples per channel per frame
//! - 8-bit `main_data_begin` and 9-bit `scalefac_compress` in side info
//! - sample rates 16/22.05/24 kHz
//!
//! Reference: ISO/IEC 13818-3 §2.4.

use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};
use oxideav_mp3::decoder::make_decoder;
use oxideav_mp3::frame::{parse_frame_header, ChannelMode, Layer, MpegVersion};
use oxideav_mp3::CODEC_ID_STR;

/// Build a single MPEG-2 LSF Layer III frame at 24 kHz, 64 kbit/s,
/// stereo, with an all-zero payload (silence).
///
/// Header layout (see `frame.rs` module docs):
/// byte 0 = 0xFF (sync 11111111)
/// byte 1 = 0xF3 (sync 111 | ver 10 (MPEG-2) | layer 01 (LIII) | noCRC 1)
/// byte 2 = 0x84 (bitrate 1000 (=64k) | sr 01 (=24k) | pad 0 | priv 0)
/// byte 3 = 0x00 (mode 00 stereo | modext 00 | copy 0 | orig 0 | emph 00)
///
/// Frame length = 72 * br / sr + pad = 72 * 64 / 24 + 0 = 192 bytes.
/// Header (4) + side info (17 for MPEG-2 stereo) = 21 bytes. Main data = 171.
fn build_mpeg2_lsf_silent_frame_24k_stereo() -> Vec<u8> {
    let mut frame = vec![0u8; 192];
    frame[0] = 0xFF;
    frame[1] = 0xF3;
    frame[2] = 0x84;
    frame[3] = 0x00;
    // Everything else (side info + main data) is left zero — which
    // represents: main_data_begin=0, private_bits=0, and a granule
    // where part2_3_length=big_values=global_gain=scalefac_compress=0
    // and all flags are zero. Global gain 0 means the final scale
    // factor 2^((0-210)/4) ~= 2^-52.5, i.e. numerical silence.
    frame
}

#[test]
fn mpeg2_lsf_header_parses_as_mpeg2_layer3() {
    let frame = build_mpeg2_lsf_silent_frame_24k_stereo();
    let hdr = parse_frame_header(&frame[..4]).expect("parse");
    assert_eq!(hdr.version, MpegVersion::Mpeg2);
    assert_eq!(hdr.layer, Layer::Layer3);
    assert_eq!(hdr.sample_rate, 24_000);
    assert_eq!(hdr.bitrate_kbps, 64);
    assert_eq!(hdr.channel_mode, ChannelMode::Stereo);
    assert_eq!(hdr.samples_per_frame(), 576);
    assert_eq!(hdr.side_info_bytes(), 17);
    assert_eq!(hdr.frame_bytes(), Some(192));
}

#[test]
fn mpeg2_lsf_decode_silent_frame_produces_576_samples() {
    let frame = build_mpeg2_lsf_silent_frame_24k_stereo();
    let params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
    let mut dec = make_decoder(&params).expect("decoder");
    let tb = TimeBase::new(1, 24_000);
    let pkt = Packet::new(0, tb, frame.clone());
    dec.send_packet(&pkt).expect("send_packet");
    let out = dec.receive_frame().expect("receive_frame");
    let audio = match out {
        Frame::Audio(a) => a,
        _ => panic!("expected AudioFrame"),
    };
    assert_eq!(audio.sample_rate, 24_000);
    assert_eq!(audio.channels, 2);
    // MPEG-2 LSF: 576 samples per frame (1 granule × 576).
    assert_eq!(audio.samples, 576);
    // S16 stereo → 576 * 2 channels * 2 bytes = 2304 bytes.
    assert_eq!(audio.data[0].len(), 576 * 2 * 2);
    // All samples should be silence (within a tight tolerance —
    // global_gain=0 yields ~2^-52 scale, which is well below s16 LSB).
    let max_abs = audio.data[0]
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]).unsigned_abs())
        .max()
        .unwrap_or(0);
    assert_eq!(
        max_abs, 0,
        "expected pure silence from MPEG-2 LSF silent frame, got max_abs={max_abs}"
    );
}

/// ffmpeg-generated MPEG-2 LSF round-trip. Skips cleanly when ffmpeg is
/// unavailable (CI environments without the binary). Produces a 440 Hz
/// mono sine at 24 kHz / 64 kbit/s and decodes it with our decoder;
/// confirms the dominant frequency is near 440 Hz.
#[test]
fn mpeg2_lsf_ffmpeg_encoded_440hz_24k_mono_decodes() {
    const FFMPEG: &str = "/usr/bin/ffmpeg";
    if !std::path::Path::new(FFMPEG).exists() {
        eprintln!("skipping: {FFMPEG} not found");
        return;
    }
    let tmp = std::env::temp_dir().join("oxideav_mp3_mpeg2lsf_440.mp3");
    let status = std::process::Command::new(FFMPEG)
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=24000:duration=1.0",
            "-ac",
            "1",
            "-ar",
            "24000",
            "-b:a",
            "64k",
            "-f",
            "mp3",
            tmp.to_str().unwrap(),
        ])
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg encode failed");

    let bytes = std::fs::read(&tmp).expect("read mp3");

    // Skip ID3v2 if present.
    let mut pos = 0usize;
    if bytes.len() >= 10 && &bytes[0..3] == b"ID3" {
        let sz = ((bytes[6] as usize) << 21)
            | ((bytes[7] as usize) << 14)
            | ((bytes[8] as usize) << 7)
            | (bytes[9] as usize);
        pos = 10 + sz;
    }

    let params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
    let mut dec = make_decoder(&params).expect("decoder");
    let tb = TimeBase::new(1, 24_000);
    let mut pcm: Vec<f32> = Vec::new();
    let mut saw_mpeg2 = false;
    while pos + 4 <= bytes.len() {
        let Ok(hdr) = parse_frame_header(&bytes[pos..]) else {
            pos += 1;
            continue;
        };
        if hdr.version == MpegVersion::Mpeg2 {
            saw_mpeg2 = true;
        }
        let Some(flen) = hdr.frame_bytes() else { break };
        let flen = flen as usize;
        if pos + flen > bytes.len() {
            break;
        }
        let pkt = Packet::new(0, tb, bytes[pos..pos + flen].to_vec());
        if dec.send_packet(&pkt).is_ok() {
            if let Ok(Frame::Audio(a)) = dec.receive_frame() {
                for chunk in a.data[0].chunks_exact(2) {
                    pcm.push(i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0);
                }
            }
        }
        pos += flen;
    }
    assert!(saw_mpeg2, "expected at least one MPEG-2 LSF frame");
    assert!(
        pcm.len() > 4 * 576,
        "too few samples decoded: {}",
        pcm.len()
    );

    // Skip warm-up frames. Run a Goertzel filter at 440 Hz and at
    // 200/800/2000 Hz noise bins; dominant-bin ratio should be clearly >1.
    let warmup = 4 * 576;
    let analysis = &pcm[warmup..];
    let sr = 24_000f32;
    let g = |freq: f32| -> f32 {
        let n = analysis.len();
        let k = (n as f32 * freq / sr).round();
        let omega = 2.0 * std::f32::consts::PI * k / n as f32;
        let coeff = 2.0 * omega.cos();
        let mut sp = 0.0f32;
        let mut sp2 = 0.0f32;
        for &x in analysis {
            let s = x + coeff * sp - sp2;
            sp2 = sp;
            sp = s;
        }
        sp2 * sp2 + sp * sp - coeff * sp * sp2
    };
    let p440 = g(440.0);
    let noise = (g(200.0) + g(800.0) + g(2000.0)) / 3.0 + 1e-12;
    let ratio = p440 / noise;
    eprintln!("MPEG-2 LSF 440Hz/noise ratio: {ratio:.2}");
    assert!(
        ratio > 20.0,
        "440Hz too weak in decoded MPEG-2 LSF: {ratio:.2}"
    );
}

#[test]
fn mpeg2_lsf_decode_silent_frame_mono_22050() {
    // MPEG-2 LSF, 22.05 kHz, 32 kbit/s, mono.
    // Frame length = 72 * 32 / 22.05 ≈ 104 bytes. Actually integer: 72 * 32000 / 22050 = 104.
    // Header(4) + side-info(9 mono) = 13 bytes. Main data = 91.
    let mut frame = vec![0u8; 104];
    // byte 0: 0xFF
    // byte 1: sync(111) ver(10) layer(01) noCRC(1) = 11110011 = 0xF3
    // byte 2: bitrate(0100=32k index 4) sr(00=22.05) pad(0) priv(0) = 01000000 = 0x40
    // byte 3: mode(11 mono) modext(00) copy(0) orig(0) emph(00) = 11000000 = 0xC0
    frame[0] = 0xFF;
    frame[1] = 0xF3;
    frame[2] = 0x40;
    frame[3] = 0xC0;

    let hdr = parse_frame_header(&frame[..4]).expect("parse");
    assert_eq!(hdr.version, MpegVersion::Mpeg2);
    assert_eq!(hdr.sample_rate, 22_050);
    assert_eq!(hdr.bitrate_kbps, 32);
    assert_eq!(hdr.channel_mode, ChannelMode::Mono);
    assert_eq!(hdr.side_info_bytes(), 9);
    assert_eq!(hdr.frame_bytes(), Some(104));

    let params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
    let mut dec = make_decoder(&params).expect("decoder");
    let tb = TimeBase::new(1, 22_050);
    let pkt = Packet::new(0, tb, frame);
    dec.send_packet(&pkt).expect("send_packet");
    let out = dec.receive_frame().expect("receive_frame");
    let audio = match out {
        Frame::Audio(a) => a,
        _ => panic!("expected AudioFrame"),
    };
    assert_eq!(audio.samples, 576);
    assert_eq!(audio.channels, 1);
    // Mono: 576 * 1 * 2 bytes.
    assert_eq!(audio.data[0].len(), 576 * 2);
    let max_abs = audio.data[0]
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]).unsigned_abs())
        .max()
        .unwrap_or(0);
    assert_eq!(max_abs, 0);
}

/// MPEG-2.5 (unofficial Fraunhofer low-sample-rate extension) header +
/// silent-frame smoke test at 12 kHz / 32 kbps / stereo.
///
/// Header:
/// byte 0 = 0xFF
/// byte 1 = sync(111) ver(00 = MPEG-2.5) layer(01 = L3) noCRC(1) = 0xE3
/// byte 2 = bitrate(0100 = 32 kbps idx 4) sr(01 = 12000) pad(0) priv(0) = 0x44
/// byte 3 = mode(00 stereo) modext(00) copy(0) orig(0) emph(00) = 0x00
///
/// Frame length = 72 * 32 / 12 = 192 bytes. Header(4) + side-info(17) = 21. Main data = 171.
#[test]
fn mpeg25_header_parses_and_decode_silent_frame_12k_stereo() {
    let mut frame = vec![0u8; 192];
    frame[0] = 0xFF;
    frame[1] = 0xE3;
    frame[2] = 0x44;
    frame[3] = 0x00;

    let hdr = parse_frame_header(&frame[..4]).expect("parse");
    assert_eq!(hdr.version, MpegVersion::Mpeg25);
    assert_eq!(hdr.layer, Layer::Layer3);
    assert_eq!(hdr.sample_rate, 12_000);
    assert_eq!(hdr.bitrate_kbps, 32);
    assert_eq!(hdr.channel_mode, ChannelMode::Stereo);
    // MPEG-2.5 inherits MPEG-2 LSF side-info shape (single granule,
    // 17 stereo / 9 mono bytes).
    assert_eq!(hdr.side_info_bytes(), 17);
    assert_eq!(hdr.samples_per_frame(), 576);
    assert_eq!(hdr.frame_bytes(), Some(192));

    let params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
    let mut dec = make_decoder(&params).expect("decoder");
    let tb = TimeBase::new(1, 12_000);
    let pkt = Packet::new(0, tb, frame);
    dec.send_packet(&pkt).expect("send_packet");
    let out = dec.receive_frame().expect("receive_frame");
    let audio = match out {
        Frame::Audio(a) => a,
        _ => panic!("expected AudioFrame"),
    };
    assert_eq!(audio.sample_rate, 12_000);
    assert_eq!(audio.channels, 2);
    assert_eq!(audio.samples, 576);
    assert_eq!(audio.data[0].len(), 576 * 2 * 2);
    let max_abs = audio.data[0]
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]).unsigned_abs())
        .max()
        .unwrap_or(0);
    assert_eq!(
        max_abs, 0,
        "expected pure silence from MPEG-2.5 silent frame, got max_abs={max_abs}"
    );
}

/// MPEG-2.5 8 kHz mono silent-frame smoke test. Exercises the distinct
/// 8 kHz sfband partition added by the MPEG-2.5 low-sample-rate annex.
///
/// Header:
/// byte 1 = sync(111) ver(00) layer(01) noCRC(1) = 0xE3
/// byte 2 = bitrate(0010 = 16 kbps idx 2) sr(10 = 8000) pad(0) priv(0) = 0x28
/// byte 3 = mode(11 mono) modext(00) copy(0) orig(0) emph(00) = 0xC0
///
/// Frame length = 72 * 16 / 8 = 144 bytes. Header(4) + side-info(9 mono) = 13. Main data = 131.
#[test]
fn mpeg25_decode_silent_frame_8k_mono() {
    let mut frame = vec![0u8; 144];
    frame[0] = 0xFF;
    frame[1] = 0xE3;
    frame[2] = 0x28;
    frame[3] = 0xC0;

    let hdr = parse_frame_header(&frame[..4]).expect("parse");
    assert_eq!(hdr.version, MpegVersion::Mpeg25);
    assert_eq!(hdr.sample_rate, 8_000);
    assert_eq!(hdr.bitrate_kbps, 16);
    assert_eq!(hdr.channel_mode, ChannelMode::Mono);
    assert_eq!(hdr.side_info_bytes(), 9);
    assert_eq!(hdr.frame_bytes(), Some(144));

    let params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
    let mut dec = make_decoder(&params).expect("decoder");
    let tb = TimeBase::new(1, 8_000);
    let pkt = Packet::new(0, tb, frame);
    dec.send_packet(&pkt).expect("send_packet");
    let out = dec.receive_frame().expect("receive_frame");
    let audio = match out {
        Frame::Audio(a) => a,
        _ => panic!("expected AudioFrame"),
    };
    assert_eq!(audio.sample_rate, 8_000);
    assert_eq!(audio.channels, 1);
    assert_eq!(audio.samples, 576);
    assert_eq!(audio.data[0].len(), 576 * 2);
    let max_abs = audio.data[0]
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]).unsigned_abs())
        .max()
        .unwrap_or(0);
    assert_eq!(max_abs, 0);
}

/// End-to-end test for MPEG-2 LSF intensity stereo. ffmpeg (libmp3lame)
/// is asked to encode a stereo signal where the two channels differ by
/// amplitude at 16 kHz / 32 kbit/s / joint-stereo — a configuration that
/// strongly encourages the encoder to switch to MPEG-2 IS coding on high
/// frequencies. The decoded R channel must carry non-zero audio content
/// (before the MPEG-2 IS landing, the R channel was zero above the IS
/// bound because the is_pos path was a no-op).
#[test]
fn mpeg2_lsf_joint_stereo_ffmpeg_encoded_440hz_16k_decodes_r_channel() {
    const FFMPEG: &str = "/usr/bin/ffmpeg";
    if !std::path::Path::new(FFMPEG).exists() {
        eprintln!("skipping: {FFMPEG} not found");
        return;
    }
    let tmp = std::env::temp_dir().join("oxideav_mp3_mpeg2lsf_js_16k.mp3");
    // Dual-tone stereo input: L gets a 440 Hz sine, R a 523.25 Hz sine
    // (different frequency so the decoder really has to reconstruct R
    // rather than copy L). Merged with lavfi amerge.
    let status = std::process::Command::new(FFMPEG)
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=16000:duration=1.5",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=523.25:sample_rate=16000:duration=1.5",
            "-filter_complex",
            "[0:a][1:a]amerge=inputs=2,volume=0.6[a]",
            "-map",
            "[a]",
            "-ac",
            "2",
            "-ar",
            "16000",
            "-b:a",
            "32k",
            "-joint_stereo",
            "1",
            "-f",
            "mp3",
            tmp.to_str().unwrap(),
        ])
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg encode failed");

    let bytes = std::fs::read(&tmp).expect("read mp3");

    // Skip ID3v2 prefix if any.
    let mut pos = 0usize;
    if bytes.len() >= 10 && &bytes[0..3] == b"ID3" {
        let sz = ((bytes[6] as usize) << 21)
            | ((bytes[7] as usize) << 14)
            | ((bytes[8] as usize) << 7)
            | (bytes[9] as usize);
        pos = 10 + sz;
    }

    let params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
    let mut dec = make_decoder(&params).expect("decoder");
    let tb = TimeBase::new(1, 16_000);
    let mut pcm_l: Vec<f32> = Vec::new();
    let mut pcm_r: Vec<f32> = Vec::new();
    let mut saw_mpeg2 = false;
    let mut saw_js = false;
    let mut saw_is_mode = false;
    while pos + 4 <= bytes.len() {
        let Ok(hdr) = parse_frame_header(&bytes[pos..]) else {
            pos += 1;
            continue;
        };
        if hdr.version == MpegVersion::Mpeg2 {
            saw_mpeg2 = true;
        }
        if hdr.channel_mode == ChannelMode::JointStereo {
            saw_js = true;
            if hdr.mode_extension & 0x1 != 0 {
                saw_is_mode = true;
            }
        }
        let Some(flen) = hdr.frame_bytes() else { break };
        let flen = flen as usize;
        if pos + flen > bytes.len() {
            break;
        }
        let pkt = Packet::new(0, tb, bytes[pos..pos + flen].to_vec());
        if dec.send_packet(&pkt).is_ok() {
            if let Ok(Frame::Audio(a)) = dec.receive_frame() {
                for chunk in a.data[0].chunks_exact(4) {
                    pcm_l.push(
                        i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0,
                    );
                    pcm_r.push(
                        i16::from_le_bytes([chunk[2], chunk[3]]) as f32 / 32768.0,
                    );
                }
            }
        }
        pos += flen;
    }
    assert!(saw_mpeg2, "expected at least one MPEG-2 LSF frame");
    assert!(saw_js, "expected joint-stereo mode");
    eprintln!(
        "MPEG-2 LSF JS test: saw_mpeg2={saw_mpeg2}, saw_js={saw_js}, saw_is_mode={saw_is_mode}"
    );

    assert!(
        pcm_l.len() > 4 * 576,
        "too few samples decoded: {}",
        pcm_l.len()
    );
    assert_eq!(pcm_l.len(), pcm_r.len());

    // Skip warm-up.
    let warmup = 4 * 576;
    let l_tail = &pcm_l[warmup..];
    let r_tail = &pcm_r[warmup..];

    let rms = |x: &[f32]| {
        (x.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / x.len() as f64).sqrt()
    };
    let rms_l = rms(l_tail);
    let rms_r = rms(r_tail);
    eprintln!("MPEG-2 LSF JS decoded RMS: L={rms_l:.5}, R={rms_r:.5}");

    // L channel must have meaningful audio. R channel must also be
    // non-zero (regression check: before MPEG-2 IS landed, R could be
    // near-silent above the IS bound).
    assert!(rms_l > 0.01, "L channel too quiet, RMS={rms_l}");
    assert!(
        rms_r > 0.001,
        "R channel near-silent — MPEG-2 LSF IS not applied? RMS={rms_r}"
    );
}
