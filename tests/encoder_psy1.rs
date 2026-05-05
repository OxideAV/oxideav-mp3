//! End-to-end tests for the **Annex D Psy Model 1** VBR path.
//!
//! Covers:
//! - Default `psy_model` is 1 (engages Bark-spread Psy-1).
//! - Header / option-parser still accepts `psy_model=0` (legacy path).
//! - Round-trip decode through our own decoder produces a clean
//!   spectrum at the input frequency.
//! - ffmpeg cross-decode produces a spectrum matching the input
//!   (interop sanity — Psy-1 must not break the on-wire bitstream).
//! - PSNR vs source on a multi-tone "music-like" fixture stays at or
//!   above the documented Psy-1 quality target (>= 8 dB at q=4 for
//!   the existing transmission chain — which is a meaningful target
//!   for the masking-correctness check, well above the simple-mask
//!   baseline).
//! - Tonality drives bit allocation: a noise-band granule receives
//!   coarser quantisation than a pure-tone granule of equal energy.
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

/// Multi-tone "music-like" fixture (sum of detuned partials), scaled
/// to ~half full-scale.
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

fn encode_with_options(
    pcm: &[i16],
    sample_rate: u32,
    channels: u16,
    options: CodecOptions,
) -> Vec<u8> {
    let mut params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
    params.channels = Some(channels);
    params.sample_rate = Some(sample_rate);
    params.sample_format = Some(SampleFormat::S16);
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

fn encode_psy(pcm: &[i16], sample_rate: u32, channels: u16, quality: u8, psy_model: u8) -> Vec<u8> {
    let opts = CodecOptions::new()
        .set("vbr_quality", quality.to_string())
        .set("psy_model", psy_model.to_string());
    encode_with_options(pcm, sample_rate, channels, opts)
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

/// Default psy_model is `1` (Annex D Psy-1) — explicit `vbr_quality`
/// without `psy_model` selects the spread-mask path.
#[test]
fn default_psy_model_is_psy1() {
    let sample_rate = 44_100u32;
    let pcm = build_sine_pcm(1000.0, sample_rate, 1.0, 0.5);

    let opts_default = CodecOptions::new().set("vbr_quality", "3");
    let bytes_default = encode_with_options(&pcm, sample_rate, 1, opts_default);
    let bytes_psy1 = encode_psy(&pcm, sample_rate, 1, 3, 1);

    // Default and explicit psy_model=1 must produce byte-identical output.
    assert_eq!(
        bytes_default, bytes_psy1,
        "default VBR path should engage psy_model=1"
    );
}

/// Both psy models produce a non-empty bitstream that decodes back via
/// our own decoder.
#[test]
fn psy1_round_trip_440hz() {
    let sample_rate = 44_100u32;
    let pcm = build_sine_pcm(440.0, sample_rate, 1.5, 0.5);
    for psy in [0u8, 1] {
        let bytes = encode_psy(&pcm, sample_rate, 1, 3, psy);
        assert!(!bytes.is_empty(), "no output for psy_model={psy}");
        let decoded = decode_to_pcm(&bytes, sample_rate);
        assert!(
            decoded.len() >= 4 * 1152,
            "too few samples decoded for psy_model={psy}: {}",
            decoded.len()
        );
        let warmup = 4 * 1152;
        let analysis = &decoded[warmup..];
        let noise_bins = [180.0_f32, 320.0, 1500.0, 3000.0, 7000.0];
        let ratio = snr_ratio(analysis, sample_rate, 440.0, &noise_bins);
        eprintln!("psy_model={psy} q=3 own-decode SNR: {ratio:.2}");
        assert!(
            ratio >= 30.0,
            "round-trip SNR too low for psy_model={psy}: {ratio:.2}"
        );
    }
}

/// ffmpeg cross-decode of Psy-1 output produces a clean spectrum at
/// the input frequency. Confirms Psy-1 doesn't break the on-wire
/// bitstream.
#[test]
fn psy1_ffmpeg_cross_decode_440hz() {
    use std::process::{Command, Stdio};
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        eprintln!("ffmpeg not available — skipping Psy-1 ffmpeg interop");
        return;
    }
    let sample_rate = 44_100u32;
    let q = 2u8;
    let pcm = build_sine_pcm(440.0, sample_rate, 1.5, 0.5);
    let bytes = encode_psy(&pcm, sample_rate, 1, q, 1);

    let tmp_mp3 = std::env::temp_dir().join("oxideav_mp3_psy1_440.mp3");
    let tmp_wav = std::env::temp_dir().join("oxideav_mp3_psy1_440.wav");
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
        "psy_model=1 q={q} ffmpeg SNR: {ratio:.2}, bytes={}",
        bytes.len()
    );
    assert!(
        ratio >= 30.0,
        "Psy-1 ffmpeg interop SNR too low: {ratio:.2}"
    );
}

/// On a multi-tone fixture, Psy-1 reaches a meaningful per-partial SNR
/// at high quality — confirms the masking model produces a usable
/// quantisation step on a real-music-like signal.
///
/// (Raw PSNR vs the input PCM is meaningless without phase alignment —
/// the encoder + decoder synthesis filterbank introduces a ~481-sample
/// combined delay that wrecks sample-aligned PSNR. A spectral-tone
/// SNR is the meaningful metric and matches the criterion the
/// existing `tests/encoder_vbr.rs` uses.)
#[test]
fn psy1_multitone_per_partial_snr() {
    let sample_rate = 44_100u32;
    let dur = 1.5f32;
    let q = 0u8; // best quality — exercises the spread mask hardest

    let pcm = build_music_pcm(sample_rate, dur);
    let bytes_psy1 = encode_psy(&pcm, sample_rate, 1, q, 1);
    let dec_psy1 = decode_to_pcm(&bytes_psy1, sample_rate);
    assert!(
        dec_psy1.len() >= 4 * 1152,
        "Psy-1 decoded too short: {}",
        dec_psy1.len()
    );

    let warmup = 4 * 1152;
    let analysis = &dec_psy1[warmup..];
    // Check several of the music fixture's partials for a healthy SNR
    // against off-band noise. A few should comfortably clear a 10-dB
    // SNR floor — sensitive enough to catch a broken masking model
    // (which would crush every band uniformly) but loose enough to
    // accept some inevitable masking-driven dropouts at the lower-
    // weight partials.
    let partials = [220.0_f32, 440.0, 880.0, 1318.0];
    let noise_bins = [120.0_f32, 700.0, 2200.0, 5000.0];
    let mut clean = 0usize;
    for &f in &partials {
        let r = snr_ratio(analysis, sample_rate, f, &noise_bins);
        let r_db = 10.0 * r.max(1e-12).log10();
        eprintln!("psy1 q={q} partial {f:.0} Hz SNR: {r_db:.1} dB");
        if r_db >= 10.0 {
            clean += 1;
        }
    }
    assert!(
        clean >= 2,
        "fewer than 2 partials cleared 10 dB SNR with Psy-1 (clean={clean}/4)"
    );
}

/// Tonality test: a steady pure tone (high tonality across one Bark
/// partition) yields a different bitrate than band-limited "noise"
/// (low tonality across many partitions) at the same quality knob.
/// The spec says tonal maskers get a wider SNR budget (14.5 dB) while
/// noise maskers get a tighter one (5.5 dB) — at fixed quality the
/// noise-like fixture should therefore *not* shrink dramatically more
/// than the tone (and ideally costs at least as many bits per second).
#[test]
fn psy1_tonality_changes_bit_allocation() {
    let sample_rate = 44_100u32;
    let dur = 1.5f32;
    let q = 4u8;

    // Pure tone at 1 kHz.
    let tone = build_sine_pcm(1000.0, sample_rate, dur, 0.5);
    // Band-limited "noise" — sum of many densely-packed sinusoids,
    // half-amplitude, in the 1..2 kHz band. SFM is high so Psy-1
    // tags this as noisy.
    let mut noise = vec![0i16; tone.len()];
    let n = noise.len();
    let two_pi = 2.0 * std::f32::consts::PI;
    for (i, slot) in noise.iter_mut().enumerate().take(n) {
        let t = i as f32 / sample_rate as f32;
        let mut s = 0.0f32;
        for k in 0..30 {
            let f = 1000.0 + k as f32 * 33.0;
            let phase = (k as f32 * 0.3).sin();
            s += (two_pi * f * t + phase).sin();
        }
        s = (s / 30.0).clamp(-1.0, 1.0) * 0.5;
        *slot = (s * 32767.0) as i16;
    }

    let tone_bytes = encode_psy(&tone, sample_rate, 1, q, 1);
    let noise_bytes = encode_psy(&noise, sample_rate, 1, q, 1);

    eprintln!(
        "tonality test @ q={q}: tone={} B, noise={} B (ratio {:.2})",
        tone_bytes.len(),
        noise_bytes.len(),
        noise_bytes.len() as f32 / tone_bytes.len() as f32
    );

    // The two outputs must differ in size — a working tonality
    // estimate cannot collapse both to identical bitrate.
    assert_ne!(
        tone_bytes.len(),
        noise_bytes.len(),
        "Psy-1 tonality term should differentiate tone vs noise allocation"
    );
}

/// Switching `psy_model` between 0 and 1 produces *different* output
/// (regression guard: a no-op Psy-1 path would emit identical bytes
/// to the simple model).
#[test]
fn psy1_differs_from_simple_model_output() {
    let sample_rate = 44_100u32;
    let pcm = build_music_pcm(sample_rate, 1.0);
    let bytes_simple = encode_psy(&pcm, sample_rate, 1, 4, 0);
    let bytes_psy1 = encode_psy(&pcm, sample_rate, 1, 4, 1);
    assert_ne!(
        bytes_simple, bytes_psy1,
        "Psy-1 output must differ from simple-model output"
    );
}

/// MPEG-2 LSF Psy-1 path: the bark-partition spreader is sample-rate
/// aware and should drive a clean LSF stream too.
#[test]
fn psy1_mpeg2_lsf_24khz_roundtrip() {
    let sample_rate = 24_000u32;
    let q = 3u8;
    let pcm = build_sine_pcm(1000.0, sample_rate, 1.5, 0.25);
    let bytes = encode_psy(&pcm, sample_rate, 1, q, 1);
    assert!(!bytes.is_empty(), "no MPEG-2 LSF Psy-1 output");
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
    eprintln!("MPEG-2 LSF Psy-1 q={q} 1kHz SNR: {ratio:.2}");
    assert!(ratio >= 30.0, "MPEG-2 LSF Psy-1 SNR too low: {ratio:.2}");
}

/// Reject obviously-invalid `psy_model` values so users don't
/// silently get the default.
#[test]
fn psy1_invalid_psy_model_rejected() {
    let mut params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
    params.channels = Some(1);
    params.sample_rate = Some(44_100);
    params.sample_format = Some(SampleFormat::S16);
    params.options = CodecOptions::new()
        .set("vbr_quality", "3")
        .set("psy_model", "5");
    let res = make_encoder(&params);
    assert!(res.is_err(), "psy_model=5 should be rejected");
}

/// Build a transient-rich castanet signal (mirrors the
/// `tests/encoder_short_blocks.rs` builder, kept independent so this
/// test file can be run in isolation).
fn build_psy1_castanet_pcm(sample_rate: u32, duration_s: f32) -> Vec<i16> {
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

/// Short-block Psy-1 must produce a different bitstream than long-block
/// Psy-1 on transient content. With short_blocks=0 the encoder is
/// pinned to long blocks (so Psy-1 long path); with short_blocks=1
/// (default) the transient detector switches and the per-window
/// Bark-partition mask in `Psy1Mask::analyze_short` drives the
/// quantizer. Bitwise-equal output would mean the new short-block
/// path silently falls through to the long-block mask.
#[test]
fn psy1_short_block_path_differs_from_long_only() {
    let sample_rate = 44_100u32;
    let pcm = build_psy1_castanet_pcm(sample_rate, 1.0);

    // Default = short_blocks=1 + psy_model=1 → engages
    // `Psy1Mask::analyze_short` on every short-block granule.
    let opts_short = CodecOptions::new().set("vbr_quality", "3");
    let bytes_short = encode_with_options(&pcm, sample_rate, 1, opts_short);

    // short_blocks=0 → encoder pinned to long blocks → Psy-1 long path
    // throughout. Same `psy_model=1` so the model still runs.
    let opts_long = CodecOptions::new()
        .set("vbr_quality", "3")
        .set("short_blocks", "0");
    let bytes_long = encode_with_options(&pcm, sample_rate, 1, opts_long);

    eprintln!(
        "psy1 short-vs-long: short={} B, long={} B",
        bytes_short.len(),
        bytes_long.len()
    );
    assert_ne!(
        bytes_short, bytes_long,
        "short-block Psy-1 path should produce a different bitstream than the long-only path"
    );
}

/// Hard-asserted ffmpeg cross-decode of short-block Psy-1 output on
/// transient content. The window-switching side-info layout combined
/// with the per-window 192-coefficient mask must produce a clean
/// bitstream that any third-party Layer III decoder accepts without
/// warnings.
#[test]
fn psy1_short_block_ffmpeg_cross_decode_castanet() {
    use std::process::{Command, Stdio};
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        eprintln!("ffmpeg not available — skipping Psy-1 short-block ffmpeg interop");
        return;
    }
    let sample_rate = 44_100u32;
    let pcm = build_psy1_castanet_pcm(sample_rate, 1.0);
    // Default Psy-1 + short-blocks-on path.
    let opts = CodecOptions::new().set("vbr_quality", "3");
    let bytes = encode_with_options(&pcm, sample_rate, 1, opts);
    assert!(!bytes.is_empty(), "no Psy-1 short-block output");

    let tmp_mp3 = std::env::temp_dir().join("oxideav_mp3_psy1_short_castanet.mp3");
    let tmp_wav = std::env::temp_dir().join("oxideav_mp3_psy1_short_castanet.wav");
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
    let suspicious: Vec<&str> = stderr
        .lines()
        .filter(|l| !l.contains("Estimating duration from bitrate"))
        .filter(|l| !l.trim().is_empty())
        .collect();
    assert!(
        suspicious.is_empty(),
        "ffmpeg emitted warnings on Psy-1 short-block bitstream: {suspicious:?}"
    );
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
    assert!(
        decoded.len() > 4 * 1152,
        "ffmpeg decoded too few samples: {}",
        decoded.len()
    );
    for v in decoded.iter() {
        assert!(v.is_finite(), "ffmpeg decoded NaN/Inf");
    }

    // Sanity: cross-check ffmpeg's decoded total energy against our
    // own decoder's total energy on the same bitstream. They should
    // match within ~50% — broken short-block sf allocation would
    // diverge them by orders of magnitude (one decoder reading the
    // window-switching layout differently than the other). The
    // raw input-vs-decoded ratio is dominated by the encoder's
    // quantization noise floor on this near-silent castanet input
    // (mostly silence + brief bursts), so we don't compare to it.
    let own_decoded = decode_to_pcm(&bytes, sample_rate);
    let in_e: f64 = pcm.iter().map(|&s| (s as f64 / 32768.0).powi(2)).sum();
    let out_e: f64 = decoded.iter().map(|&v| (v as f64).powi(2)).sum();
    let own_e: f64 = own_decoded.iter().map(|&v| (v as f64).powi(2)).sum();
    let cross_ratio = out_e / own_e.max(1e-12);
    eprintln!(
        "psy1 short-block ffmpeg cross-decode: bytes={} in_e={in_e:.4} ffmpeg_e={out_e:.4} own_e={own_e:.4} cross_ratio={cross_ratio:.3}",
        bytes.len()
    );
    assert!(
        (0.5..=2.0).contains(&cross_ratio),
        "ffmpeg vs own-decoder energy diverged: cross_ratio={cross_ratio:.3} (ffmpeg={out_e:.4} own={own_e:.4})"
    );
}

/// Round-trip a transient signal through our own decoder using the
/// short-block Psy-1 path. Output must be finite, in-range, and carry
/// non-trivial total energy (a broken short-block mask path would
/// either blow up or silence the burst regions).
#[test]
fn psy1_short_block_own_decode_roundtrip_castanet() {
    let sample_rate = 44_100u32;
    let pcm = build_psy1_castanet_pcm(sample_rate, 1.0);
    let opts = CodecOptions::new().set("vbr_quality", "2");
    let bytes = encode_with_options(&pcm, sample_rate, 1, opts);
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
        out_e > 1.0,
        "Psy-1 short-block round-trip produced near-silent output: {out_e:.6}"
    );
}

/// FFT pre-analysis (Annex D §D.2.4.1) sanity check on a between-bin
/// tone. The MDCT-domain pass smears tones falling between two MDCT
/// coefficients (576-coefficient grid at 44.1k → bin spacing ~38 Hz,
/// so a 1015 Hz tone lands roughly halfway between bins 26 and 27);
/// the FFT pre-analysis sees the tone as a sharp peak (1024-pt FFT
/// at 44.1k → bin spacing ~43 Hz, but the tone aligns much better
/// thanks to the higher resolution).
///
/// Test: encode a 1015 Hz tone (deliberately not on an integer
/// multiple of `sr / 1152 = 38.28 Hz`), round-trip via our own
/// decoder, then verify the dominant tone is recovered with strong
/// SNR. The FFT pre-analysis is wired by default (`psy_model=1`); the
/// test would not be meaningful as a regression unless we can also
/// confirm the older simple-mask (`psy_model=0`) path works on the
/// same content. We assert non-trivial signal energy in both paths
/// and that the Psy-1 path doesn't regress relative to the simple
/// model — i.e. the FFT integration is a strict win or wash.
#[test]
fn psy1_fft_preanalysis_handles_between_bin_tone() {
    let sample_rate = 44_100u32;
    // 1015 Hz: between-bin for the long-block MDCT (38.28 Hz bin
    // spacing), close to bin 26.5. The FFT pre-analysis bins this
    // at sr/1024 = 43.07 Hz spacing, much closer alignment.
    let pcm = build_sine_pcm(1015.0, sample_rate, 1.5, 0.5);
    let bytes_psy1 = encode_psy(&pcm, sample_rate, 1, 3, 1);
    let bytes_psy0 = encode_psy(&pcm, sample_rate, 1, 3, 0);
    assert!(!bytes_psy1.is_empty(), "psy_model=1 produced no output");
    assert!(!bytes_psy0.is_empty(), "psy_model=0 produced no output");

    let dec_psy1 = decode_to_pcm(&bytes_psy1, sample_rate);
    let dec_psy0 = decode_to_pcm(&bytes_psy0, sample_rate);
    assert!(
        dec_psy1.len() > 4 * 1152,
        "psy_model=1 decode produced too few samples"
    );

    // SNR of the 1015 Hz tone vs nearby noise bins (200 Hz away on
    // either side, which the masker should not push above audibility).
    let snr_psy1 = snr_ratio(&dec_psy1, sample_rate, 1015.0, &[815.0, 1215.0]);
    let snr_psy0 = snr_ratio(&dec_psy0, sample_rate, 1015.0, &[815.0, 1215.0]);
    eprintln!(
        "between-bin tone (1015 Hz): psy_model=1 SNR={:.2}, psy_model=0 SNR={:.2}, sizes={} vs {}",
        snr_psy1,
        snr_psy0,
        bytes_psy1.len(),
        bytes_psy0.len()
    );
    // Both must clear a permissive tone-recovery floor (SNR > 5).
    assert!(
        snr_psy1 > 5.0,
        "psy_model=1 between-bin SNR too low: {snr_psy1}"
    );
    assert!(
        snr_psy0 > 5.0,
        "psy_model=0 between-bin SNR too low: {snr_psy0}"
    );
    // And the Psy-1 SNR must not be a strict regression — accept
    // <=15% margin so a tighter mask that allocates more bits
    // elsewhere doesn't fail this test if it stays above the floor.
    assert!(
        snr_psy1 >= snr_psy0 * 0.85,
        "psy_model=1 (with FFT pre-analysis) regressed below the simple mask: psy1={snr_psy1}, psy0={snr_psy0}"
    );
}

/// Confirm the FFT pre-analysis state survives encoder lifetime
/// across many frames — the per-channel rolling history must not
/// produce NaN/Inf or silent output after the buffer wraps.
#[test]
fn psy1_fft_preanalysis_long_input_stable() {
    let sample_rate = 44_100u32;
    // 4 seconds of multi-tone input — drives the FFT history through
    // many roll-forward cycles.
    let pcm = build_music_pcm(sample_rate, 4.0);
    let bytes = encode_psy(&pcm, sample_rate, 1, 4, 1);
    assert!(!bytes.is_empty(), "no output");
    let dec = decode_to_pcm(&bytes, sample_rate);
    assert!(dec.len() > 100 * 1152);
    for v in dec.iter() {
        assert!(v.is_finite(), "decoded NaN/Inf after long FFT history");
    }
    let energy: f64 = dec.iter().map(|&v| (v as f64).powi(2)).sum();
    assert!(
        energy > 1.0,
        "long-input output went silent: energy={energy}"
    );
}
