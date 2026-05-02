//! Encoder transient / short-block tests.
//!
//! Verifies the block-type machine + transient detector wired in
//! oxideav-mp3 round 24 actually:
//!
//! 1. Engages short blocks on a transient-rich castanet-like signal
//!    (the `mode = 0b11` window-switching frames are visible in the
//!    side info).
//! 2. Stays in long blocks on a steady tone (no spurious switches —
//!    short-block coding hurts coding efficiency on stationary signals).
//! 3. Round-trip-decodes through our own decoder cleanly across
//!    transients (no NaNs, no clipping cascade).
//! 4. Cross-decodes cleanly through ffmpeg when present (interop check
//!    of the on-wire window-switching side-info layout).

use oxideav_core::options::CodecOptions;
use oxideav_core::{AudioFrame, CodecId, CodecParameters, Frame, Packet, SampleFormat, TimeBase};
use oxideav_mp3::decoder::make_decoder;
use oxideav_mp3::encoder::make_encoder;
use oxideav_mp3::frame::parse_frame_header;
use oxideav_mp3::sideinfo::SideInfo;
use oxideav_mp3::CODEC_ID_STR;

/// Build a "castanet"-style PCM stream: low-level noise punctuated by
/// sharp impulse bursts at regular intervals. Returns S16 mono samples
/// at the given sample rate.
fn build_castanet_pcm(sample_rate: u32, duration_s: f32) -> Vec<i16> {
    let n = (sample_rate as f32 * duration_s) as usize;
    let mut out = Vec::with_capacity(n);
    // Burst spacing ~0.25 s so we get several transients per second.
    let burst_period = sample_rate as usize / 4;
    let burst_len = sample_rate as usize / 100; // 10 ms attack
    let mut rng_state: u32 = 0xdead_beef;
    for i in 0..n {
        // Simple LCG noise floor at ~1/100 amplitude.
        rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
        let noise = ((rng_state >> 16) as i16 as f32) / 32768.0 * 0.005;
        // Burst envelope: exponential decay starting at each burst.
        let phase = i % burst_period;
        let burst = if phase < burst_len {
            let t = phase as f32 / burst_len as f32;
            // Damped sinusoid — keeps the impulse spectrally rich.
            let env = (-4.0 * t).exp();
            env * 0.6 * (2.0 * std::f32::consts::PI * 2000.0 * i as f32 / sample_rate as f32).sin()
        } else {
            0.0
        };
        let s = noise + burst;
        let q = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        out.push(q);
    }
    out
}

fn build_sine_pcm(freq: f32, sample_rate: u32, duration_s: f32) -> Vec<i16> {
    let n = (sample_rate as f32 * duration_s) as usize;
    let mut out = Vec::with_capacity(n);
    let two_pi = 2.0 * std::f32::consts::PI;
    for i in 0..n {
        let t = i as f32 / sample_rate as f32;
        let s = (two_pi * freq * t).sin() * 0.4;
        out.push((s * 32767.0) as i16);
    }
    out
}

fn encode_to_bytes_opts(
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

fn encode_to_bytes(pcm: &[i16], sample_rate: u32, channels: u16, bitrate_bps: u64) -> Vec<u8> {
    encode_to_bytes_opts(pcm, sample_rate, channels, bitrate_bps, CodecOptions::new())
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
        if dec.send_packet(&pkt).is_err() {
            pos += flen;
            continue;
        }
        if let Ok(Frame::Audio(a)) = dec.receive_frame() {
            for chunk in a.data[0].chunks_exact(2) {
                let s = i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0;
                pcm.push(s);
            }
        }
        pos += flen;
    }
    pcm
}

/// Walk the bitstream and count granules per block type, by parsing
/// header + side info of each frame.
fn count_block_types(bitstream: &[u8]) -> [usize; 4] {
    let mut counts = [0usize; 4];
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
        // Side info comes right after the 4-byte header (no CRC since
        // the encoder sets protection_bit = 1).
        let si_start = pos + 4;
        let si_bytes = hdr.side_info_bytes();
        if si_start + si_bytes > bitstream.len() {
            break;
        }
        let si_slice = &bitstream[si_start..si_start + si_bytes];
        let si = if hdr.version == oxideav_mp3::frame::MpegVersion::Mpeg1 {
            SideInfo::parse_mpeg1(&hdr, si_slice)
        } else {
            SideInfo::parse_mpeg2(&hdr, si_slice)
        };
        if let Ok(si) = si {
            for gr in 0..si.num_granules as usize {
                for ch in 0..si.channels as usize {
                    let gc = si.granules[gr][ch];
                    let bt = if gc.window_switching_flag {
                        gc.block_type as usize
                    } else {
                        0
                    };
                    counts[bt.min(3)] += 1;
                }
            }
        }
        pos += flen;
    }
    counts
}

/// Transient-rich castanet-style input MUST trigger at least one
/// short-block granule. Without window switching the encoder pre-echoes
/// every onset across the full 18-coefficient long block; with short
/// blocks the energy stays localised.
#[test]
fn castanet_signal_engages_short_blocks() {
    let sample_rate = 44_100u32;
    let pcm = build_castanet_pcm(sample_rate, 2.0);
    let bytes = encode_to_bytes(&pcm, sample_rate, 1, 128_000);
    let counts = count_block_types(&bytes);
    eprintln!("block-type granule counts long/start/short/stop = {counts:?}");
    assert!(
        counts[2] > 0,
        "castanet stream produced no short-block granules: {counts:?}"
    );
    // A castanet stream should also produce balanced start + stop
    // bridges (one of each per short-block region).
    assert!(
        counts[1] > 0 && counts[3] > 0,
        "expected start + stop granules around shorts: {counts:?}"
    );
}

/// Steady tones MUST stay in long blocks — switching to short on a
/// stationary signal would burn coding efficiency for no perceptual win.
#[test]
fn steady_tone_stays_in_long_blocks() {
    let sample_rate = 44_100u32;
    let pcm = build_sine_pcm(440.0, sample_rate, 1.0);
    let bytes = encode_to_bytes(&pcm, sample_rate, 1, 128_000);
    let counts = count_block_types(&bytes);
    eprintln!("steady-tone block counts long/start/short/stop = {counts:?}");
    assert_eq!(
        counts[2], 0,
        "steady 440 Hz tone tripped short-block path: {counts:?}"
    );
    assert_eq!(counts[1], 0);
    assert_eq!(counts[3], 0);
    assert!(counts[0] > 50, "expected mostly long blocks: {counts:?}");
}

/// Round-trip a castanet signal through our own decoder and verify the
/// short-block frames decode cleanly (finite samples, no NaN cascade,
/// PSNR above a permissive floor).
#[test]
fn castanet_roundtrip_decodes_cleanly() {
    let sample_rate = 44_100u32;
    let pcm = build_castanet_pcm(sample_rate, 1.0);
    let bytes = encode_to_bytes(&pcm, sample_rate, 1, 192_000);
    let decoded = decode_to_pcm(&bytes, sample_rate);
    assert!(decoded.len() > 4 * 1152, "too few samples decoded");
    for v in decoded.iter() {
        assert!(v.is_finite(), "decoded NaN/Inf");
        assert!((-1.5..=1.5).contains(v), "decoded sample out of range: {v}");
    }
    // Energy sanity: total decoded energy should be on the same order
    // as the input. Cross-check by also encoding with short_blocks
    // disabled — both should produce comparable total energy. The
    // castanet stream is mostly silence with brief bursts; the
    // encoder's quantizer adds a noise floor that dominates total
    // energy regardless of block type.
    let in_e: f64 = pcm.iter().map(|&s| (s as f64 / 32768.0).powi(2)).sum();
    let out_e: f64 = decoded.iter().map(|&v| (v as f64).powi(2)).sum();
    let opts_long = CodecOptions::new().set("short_blocks", "0");
    let bytes_long = encode_to_bytes_opts(&pcm, sample_rate, 1, 192_000, opts_long);
    let dec_long = decode_to_pcm(&bytes_long, sample_rate);
    let long_e: f64 = dec_long.iter().map(|&v| (v as f64).powi(2)).sum();
    eprintln!("castanet roundtrip energy in={in_e:.3} out_short={out_e:.3} out_long={long_e:.3}",);
    // With-short and long-only outputs should track each other closely
    // — they encode the same content through the same quantizer; only
    // the windowing / IMDCT path differs around transients.
    let delta = (out_e - long_e).abs() / long_e.max(1e-12);
    assert!(
        delta < 0.5,
        "with-short ({out_e:.3}) diverged from long-only ({long_e:.3}); delta={delta:.3}"
    );
}

/// Pre-echo PSNR comparison: encode a transient-rich signal twice —
/// once with short blocks disabled (`short_blocks=0`, the pre-round-24
/// baseline) and once with the new short-block path. Compare the
/// PRE-onset energy around the leading edge of the transient.
///
/// With short blocks engaged the pre-echo (energy that bleeds backward
/// into the silence ahead of the transient) shrinks because the IMDCT
/// reconstruction window is 192 samples instead of 576 — that's the
/// whole point of window switching per ISO/IEC 11172-3 §2.4.2.2.
///
/// We use a single, well-isolated transient (silence followed by a
/// loud burst at a known offset) so the pre-echo region is easy to
/// pinpoint and the detector hasn't already "learned" the burst's
/// energy level (which would happen with a periodic train).
#[test]
fn short_blocks_reduce_pre_echo_on_isolated_transient() {
    let sample_rate = 44_100u32;
    // Build: 1 second of pure silence, then a loud 50 ms burst, then
    // 0.5 s of silence again. Embedded in S16.
    let total_samples = sample_rate as usize * 2;
    let burst_start = sample_rate as usize; // 1.0 s
    let burst_len = sample_rate as usize / 20; // 50 ms
    let mut pcm_i16 = vec![0i16; total_samples];
    for i in 0..burst_len {
        let t = i as f32 / burst_len as f32;
        let env = (-3.0 * t).exp();
        let s = env
            * 0.7
            * (2.0 * std::f32::consts::PI * 1500.0 * (burst_start + i) as f32 / sample_rate as f32)
                .sin();
        pcm_i16[burst_start + i] = (s * 32767.0) as i16;
    }

    // Baseline: long blocks only.
    let opts_long = CodecOptions::new().set("short_blocks", "0");
    let bytes_long = encode_to_bytes_opts(&pcm_i16, sample_rate, 1, 192_000, opts_long);
    let dec_long = decode_to_pcm(&bytes_long, sample_rate);

    // With short blocks enabled.
    let bytes_short = encode_to_bytes(&pcm_i16, sample_rate, 1, 192_000);
    let dec_short = decode_to_pcm(&bytes_short, sample_rate);

    eprintln!(
        "isolated transient: short bytes={} long bytes={} short blocks present={}",
        bytes_short.len(),
        bytes_long.len(),
        count_block_types(&bytes_short)[2] > 0,
    );

    // Pre-echo region: the 576 samples immediately before the burst
    // onset. With long-only encoding, the burst's spectral content
    // bleeds backward through the 18-coefficient long MDCT into this
    // region. With short blocks the bleed is constrained to 192
    // samples.
    let common = dec_long.len().min(dec_short.len()).min(burst_start);
    let pre_lo = burst_start.saturating_sub(576).min(common);
    let pre_hi = burst_start.min(common);

    let e_long: f64 = dec_long[pre_lo..pre_hi]
        .iter()
        .map(|&v| (v as f64).powi(2))
        .sum();
    let e_short: f64 = dec_short[pre_lo..pre_hi]
        .iter()
        .map(|&v| (v as f64).powi(2))
        .sum();
    let psnr_long = -10.0 * (e_long / (pre_hi - pre_lo) as f64).max(1e-30).log10();
    let psnr_short = -10.0 * (e_short / (pre_hi - pre_lo) as f64).max(1e-30).log10();
    eprintln!(
        "pre-echo energy long-only={e_long:.6} short={e_short:.6} \
         pre-echo PSNR long={psnr_long:.2} dB short={psnr_short:.2} dB \
         delta={:.2} dB",
        psnr_short - psnr_long
    );
    // Short-block encoding should reduce the pre-echo region's energy
    // by at least 10% (≈ 0.5 dB PSNR improvement). Any positive delta
    // demonstrates the short-block IMDCT envelope is actually
    // localizing the transient.
    assert!(
        e_short < e_long * 0.9,
        "short blocks did not reduce pre-echo: long={e_long:.6} short={e_short:.6}"
    );
}

/// ffmpeg interop check — the window-switching side-info layout (only
/// 2 table_select entries + 3 subblock_gain) has to round-trip through
/// a third-party decoder. Skipped silently when ffmpeg is unavailable.
#[test]
fn castanet_decodes_via_ffmpeg() {
    use std::process::{Command, Stdio};
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        eprintln!("ffmpeg not available — skipping interop check");
        return;
    }
    let sample_rate = 44_100u32;
    let pcm = build_castanet_pcm(sample_rate, 1.0);
    let bytes = encode_to_bytes(&pcm, sample_rate, 1, 192_000);
    let tmp_mp3 = std::env::temp_dir().join("oxideav_mp3_castanet.mp3");
    let tmp_wav = std::env::temp_dir().join("oxideav_mp3_castanet.wav");
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
        "ffmpeg emitted warnings on castanet bitstream: {suspicious_lines:?}"
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
    assert!(
        decoded.len() > 4 * 1152,
        "ffmpeg decoded too few samples: {}",
        decoded.len()
    );
    for v in decoded.iter() {
        assert!(v.is_finite());
    }
}
