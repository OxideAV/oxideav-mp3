//! **Black-box validator decode sweep** of the encoder's output.
//!
//! The crate's encode→self-decode round-trips prove internal
//! consistency, but a stream that only *this* crate can decode would
//! still pass them. This sweep closes that loop: it encodes PCM at
//! every supported sample rate — MPEG-1 (32 / 44.1 / 48 kHz), MPEG-2
//! LSF (16 / 22.05 / 24 kHz), and all three MPEG-2.5 rates (8 /
//! 11.025 / 12 kHz) — hands each produced stream to an *external*
//! black-box decoder binary, and checks:
//!
//! 1. the validator accepts the stream and reports the exact sample
//!    rate and channel count we encoded (this pins the header
//!    version/sample-rate-index dispatch, including the MPEG-2.5
//!    `id`-bit layout, against what deployed decoders expect);
//! 2. the validator's PCM matches this crate's own decode of the same
//!    bytes in the float-rounding regime (aligned by peak normalized
//!    cross-correlation) — i.e. an independent decoder reads back the
//!    very waveform we think we wrote, at every rate, both channel
//!    layouts, long *and* forced-short blocks.
//!
//! ## Running it
//!
//! The validator is injected by environment variable so the test stays
//! toolchain-agnostic and CI-safe (skips with a log line when unset):
//!
//! ```text
//! OXIDEAV_MP3_VALIDATOR_DECODE='<shell command with {IN} and {OUT}>'
//! ```
//!
//! The command is run through `sh -c` after `{IN}` is replaced with the
//! path of an `.mp3` file and `{OUT}` with the path of the RIFF/WAVE
//! 16-bit integer PCM file the command must produce. Only the binary's
//! CLI contract is relied on; its internals stay a black box.

use std::env;
use std::fs;
use std::io::Cursor;
use std::path::Path;
use std::process::Command;

use oxideav_core::{
    CodecId, CodecParameters, Decoder, Demuxer, Error, Frame, RuntimeContext, SampleFormat,
};
use oxideav_mp3::{ChannelMode, Mp3Demuxer, Mp3Encoder};

/// One sweep case: label, sample rate, bitrate, stereo?, forced-short?,
/// wideband-noise input? (noise exercises every scalefactor band —
/// narrowband tones cannot distinguish band tables that differ only in
/// their upper bands, which is exactly how the MPEG-2.5 low-rate short
/// tables differ from their 22.05 / 24 kHz LSF lookalikes).
struct Case {
    label: &'static str,
    sample_rate: u32,
    bitrate_kbps: u32,
    stereo: bool,
    force_short: bool,
    noise: bool,
    /// Force every granule onto the §2.4.2.7 mixed-block path. Only
    /// rates where both deployed validators agree on mixed geometry
    /// are listed (8 kHz mixed is refused by the encoder; MPEG-2.5
    /// 11.025 / 12 kHz mixed decodes bit-faithfully on the primary
    /// deployed decoder but a second deployed decoder renders it
    /// differently, so those stay out of the committed matrix).
    force_mixed: bool,
    /// Arm the §C.1.5.2 signal-driven auto block-type scheduler and
    /// feed an attack-train signal, so the stream carries
    /// LONG / START / SHORT / STOP transitions.
    auto: bool,
    /// Arm the mixed-promotion auto scheduler
    /// (`enable_auto_block_type_with_mixed`) and feed a low-band-DC +
    /// Nyquist-click stimulus, so the stream carries **mixed bursts**
    /// — `Start` / `Short+mixed` / `End` transition sequences whose
    /// flanking granules carry the §2.4.2.7 `mixed_block_flag` (r408:
    /// before the flanking-flag fix these sequences left uncancelled
    /// low-subband aliasing that an independent deployed decoder
    /// rendered differently, nrmse ≈ 3e-2..8e-2).
    auto_mixed: bool,
}

const CASES: &[Case] = &[
    // MPEG-1
    Case {
        label: "mpeg1-32000-mono-64",
        sample_rate: 32_000,
        bitrate_kbps: 64,
        stereo: false,
        force_short: false,
        noise: false,
        force_mixed: false,
        auto: false,
        auto_mixed: false,
    },
    Case {
        label: "mpeg1-44100-mono-128",
        sample_rate: 44_100,
        bitrate_kbps: 128,
        stereo: false,
        force_short: false,
        noise: false,
        force_mixed: false,
        auto: false,
        auto_mixed: false,
    },
    Case {
        label: "mpeg1-44100-stereo-192",
        sample_rate: 44_100,
        bitrate_kbps: 192,
        stereo: true,
        force_short: false,
        noise: false,
        force_mixed: false,
        auto: false,
        auto_mixed: false,
    },
    Case {
        label: "mpeg1-48000-stereo-128",
        sample_rate: 48_000,
        bitrate_kbps: 128,
        stereo: true,
        force_short: false,
        noise: false,
        force_mixed: false,
        auto: false,
        auto_mixed: false,
    },
    // MPEG-2 LSF
    Case {
        label: "mpeg2-16000-mono-32",
        sample_rate: 16_000,
        bitrate_kbps: 32,
        stereo: false,
        force_short: false,
        noise: false,
        force_mixed: false,
        auto: false,
        auto_mixed: false,
    },
    Case {
        label: "mpeg2-22050-mono-48",
        sample_rate: 22_050,
        bitrate_kbps: 48,
        stereo: false,
        force_short: false,
        noise: false,
        force_mixed: false,
        auto: false,
        auto_mixed: false,
    },
    Case {
        label: "mpeg2-24000-stereo-64",
        sample_rate: 24_000,
        bitrate_kbps: 64,
        stereo: true,
        force_short: false,
        noise: false,
        force_mixed: false,
        auto: false,
        auto_mixed: false,
    },
    Case {
        label: "mpeg2-16000-mono-32-short",
        sample_rate: 16_000,
        bitrate_kbps: 32,
        stereo: false,
        force_short: true,
        noise: false,
        force_mixed: false,
        auto: false,
        auto_mixed: false,
    },
    Case {
        label: "mpeg2-22050-mono-48-short",
        sample_rate: 22_050,
        bitrate_kbps: 48,
        stereo: false,
        force_short: true,
        noise: false,
        force_mixed: false,
        auto: false,
        auto_mixed: false,
    },
    Case {
        label: "mpeg2-24000-mono-48-short",
        sample_rate: 24_000,
        bitrate_kbps: 48,
        stereo: false,
        force_short: true,
        noise: false,
        force_mixed: false,
        auto: false,
        auto_mixed: false,
    },
    Case {
        label: "mpeg1-44100-mono-128-short",
        sample_rate: 44_100,
        bitrate_kbps: 128,
        stereo: false,
        force_short: true,
        noise: false,
        force_mixed: false,
        auto: false,
        auto_mixed: false,
    },
    // MPEG-2.5 — the Fraunhofer low-rate extension (the encode path
    // whose band tables come from mpeg2.5-scalefactor-bands.md).
    Case {
        label: "mpeg25-8000-mono-32",
        sample_rate: 8_000,
        bitrate_kbps: 32,
        stereo: false,
        force_short: false,
        noise: false,
        force_mixed: false,
        auto: false,
        auto_mixed: false,
    },
    Case {
        label: "mpeg25-8000-mono-32-short",
        sample_rate: 8_000,
        bitrate_kbps: 32,
        stereo: false,
        force_short: true,
        noise: false,
        force_mixed: false,
        auto: false,
        auto_mixed: false,
    },
    Case {
        label: "mpeg25-8000-stereo-64",
        sample_rate: 8_000,
        bitrate_kbps: 64,
        stereo: true,
        force_short: false,
        noise: false,
        force_mixed: false,
        auto: false,
        auto_mixed: false,
    },
    Case {
        label: "mpeg25-11025-mono-32",
        sample_rate: 11_025,
        bitrate_kbps: 32,
        stereo: false,
        force_short: false,
        noise: false,
        force_mixed: false,
        auto: false,
        auto_mixed: false,
    },
    Case {
        label: "mpeg25-11025-stereo-64",
        sample_rate: 11_025,
        bitrate_kbps: 64,
        stereo: true,
        force_short: false,
        noise: false,
        force_mixed: false,
        auto: false,
        auto_mixed: false,
    },
    Case {
        label: "mpeg25-11025-mono-32-short",
        sample_rate: 11_025,
        bitrate_kbps: 32,
        stereo: false,
        force_short: true,
        noise: false,
        force_mixed: false,
        auto: false,
        auto_mixed: false,
    },
    Case {
        label: "mpeg25-12000-mono-40",
        sample_rate: 12_000,
        bitrate_kbps: 40,
        stereo: false,
        force_short: false,
        noise: false,
        force_mixed: false,
        auto: false,
        auto_mixed: false,
    },
    Case {
        label: "mpeg25-12000-stereo-64",
        sample_rate: 12_000,
        bitrate_kbps: 64,
        stereo: true,
        force_short: false,
        noise: false,
        force_mixed: false,
        auto: false,
        auto_mixed: false,
    },
    Case {
        label: "mpeg25-12000-mono-40-short",
        sample_rate: 12_000,
        bitrate_kbps: 40,
        stereo: false,
        force_short: true,
        noise: false,
        force_mixed: false,
        auto: false,
        auto_mixed: false,
    },
    // Wideband-noise forced-short cases: every band of the short
    // tables carries energy, pinning the full band layout (including
    // the upper bands and the band-12 tail) at each low rate.
    Case {
        label: "mpeg25-8000-noise-short",
        sample_rate: 8_000,
        bitrate_kbps: 32,
        stereo: false,
        force_short: true,
        noise: true,
        force_mixed: false,
        auto: false,
        auto_mixed: false,
    },
    Case {
        label: "mpeg25-11025-noise-short",
        sample_rate: 11_025,
        bitrate_kbps: 32,
        stereo: false,
        force_short: true,
        noise: true,
        force_mixed: false,
        auto: false,
        auto_mixed: false,
    },
    Case {
        label: "mpeg25-12000-noise-short",
        sample_rate: 12_000,
        bitrate_kbps: 40,
        stereo: false,
        force_short: true,
        noise: true,
        force_mixed: false,
        auto: false,
        auto_mixed: false,
    },
    Case {
        label: "mpeg2-22050-noise-short",
        sample_rate: 22_050,
        bitrate_kbps: 48,
        stereo: false,
        force_short: true,
        noise: true,
        force_mixed: false,
        auto: false,
        auto_mixed: false,
    },
    Case {
        label: "mpeg1-44100-noise-short",
        sample_rate: 44_100,
        bitrate_kbps: 128,
        stereo: false,
        force_short: true,
        noise: true,
        force_mixed: false,
        auto: false,
        auto_mixed: false,
    },
    // Mixed-block cases (the §2.4.2.7 carve-out: 36-line long region +
    // short remainder, single sb == 1 alias butterfly).
    Case {
        label: "mpeg1-44100-mixed",
        sample_rate: 44_100,
        bitrate_kbps: 128,
        stereo: false,
        force_short: false,
        noise: false,
        force_mixed: true,
        auto: false,
        auto_mixed: false,
    },
    Case {
        label: "mpeg1-32000-mixed",
        sample_rate: 32_000,
        bitrate_kbps: 64,
        stereo: false,
        force_short: false,
        noise: false,
        force_mixed: true,
        auto: false,
        auto_mixed: false,
    },
    Case {
        label: "mpeg2-22050-mixed",
        sample_rate: 22_050,
        bitrate_kbps: 48,
        stereo: false,
        force_short: false,
        noise: false,
        force_mixed: true,
        auto: false,
        auto_mixed: false,
    },
    // Auto block-type cases: attack-train input drives the scheduler
    // through LONG → START → SHORT → STOP cycles, so the stream
    // carries window-switched transition granules (whose region
    // counts / third codebook are not on the wire).
    Case {
        label: "mpeg1-44100-auto",
        sample_rate: 44_100,
        bitrate_kbps: 128,
        stereo: false,
        force_short: false,
        noise: false,
        force_mixed: false,
        auto: true,
        auto_mixed: false,
    },
    Case {
        label: "mpeg2-22050-auto",
        sample_rate: 22_050,
        bitrate_kbps: 48,
        stereo: false,
        force_short: false,
        noise: false,
        force_mixed: false,
        auto: true,
        auto_mixed: false,
    },
    Case {
        label: "mpeg25-12000-auto",
        sample_rate: 12_000,
        bitrate_kbps: 40,
        stereo: false,
        force_short: false,
        noise: false,
        force_mixed: false,
        auto: true,
        auto_mixed: false,
    },
    Case {
        label: "mpeg25-8000-auto",
        sample_rate: 8_000,
        bitrate_kbps: 32,
        stereo: false,
        force_short: false,
        noise: false,
        force_mixed: false,
        auto: true,
        auto_mixed: false,
    },
    // Auto block-type WITH mixed promotion: low-band-DC + Nyquist-click
    // stimulus drives Start / Short+mixed / End bursts, so the stream
    // carries mixed granules INSIDE window transitions — at the MPEG-1
    // rates the flanking Start / End granules carry the §2.4.2.7
    // mixed_block_flag (r408). At the LSF rates the scheduler demotes
    // mixed bursts to pure-short (deployed decoders split 2-2 on the
    // flagged-flank wire combination, r408) — the 22.05 kHz case below
    // pins that the demoted stream decodes float-perfectly everywhere.
    // (MPEG-2.5 mixed stays out of the committed matrix — deployed-
    // decoder grey zone; 8 kHz mixed is refused by the encoder.)
    Case {
        label: "mpeg1-44100-automixed",
        sample_rate: 44_100,
        bitrate_kbps: 128,
        stereo: false,
        force_short: false,
        noise: false,
        force_mixed: false,
        auto: false,
        auto_mixed: true,
    },
    Case {
        label: "mpeg1-32000-automixed",
        sample_rate: 32_000,
        bitrate_kbps: 64,
        stereo: false,
        force_short: false,
        noise: false,
        force_mixed: false,
        auto: false,
        auto_mixed: true,
    },
    Case {
        label: "mpeg2-22050-automixed",
        sample_rate: 22_050,
        bitrate_kbps: 48,
        stereo: false,
        force_short: false,
        noise: false,
        force_mixed: false,
        auto: false,
        auto_mixed: true,
    },
];

/// Steady-state normalized RMS error bound for validator-vs-own decode
/// of the same bytes. Measured 2026-07 across all cases after the
/// r405 MPEG-2.5 table + short-band-12 fixes and the r408
/// mixed-transition flanking-flag fix: worst ≈ 8e-5 (float-rounding
/// differences between two independent decoders plus the two 16-bit
/// output quantizations); the bound sits an order of magnitude above.
/// Before the fixes the bad configurations measured nrmse 0.4–1.4
/// (r405 band tables) and 3e-2..8e-2 (r408 mixed transitions on an
/// independent deployed decoder) — this sweep is the test family that
/// caught them.
const NRMSE_BOUND: f64 = 1e-3;

/// Low-band 50 Hz carrier + Nyquist-alternation click bursts: the
/// low band of every granule is stationary (the mixed classifier's
/// stability condition) while the clicks fire the attack detector, so
/// the mixed-promotion auto scheduler emits Start / Short+mixed / End
/// bursts. Same stimulus family as
/// `tests/auto_block_type_mixed_roundtrip.rs`.
fn lf_dc_with_hf_click_pcm(n: usize, sample_rate: u32) -> Vec<i16> {
    let sr = f64::from(sample_rate);
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64 / sr;
        out.push((0.05 * (two_pi * 50.0 * t).sin() * f64::from(i16::MAX)) as i16);
    }
    let period = (sr * 0.15) as usize;
    let mut pos = period;
    while pos + 64 < n {
        for j in 0..64 {
            let hf = if j % 2 == 0 { 30_000i32 } else { -30_000i32 };
            let v = i32::from(out[pos + j]).saturating_add(hf);
            out[pos + j] = v.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
        }
        pos += period;
    }
    out
}

/// Tone with periodic loud bursts (attacks) to drive the auto
/// block-type scheduler through START / SHORT / STOP transitions.
fn attack_pcm(n: usize, sample_rate: u32) -> Vec<i16> {
    let sr = f64::from(sample_rate);
    let two_pi = 2.0 * std::f64::consts::PI;
    let period = (sr * 0.09) as usize; // burst every 90 ms
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64 / sr;
        let mut v = 0.08 * (two_pi * 330.0 * t).sin();
        let ph = i % period.max(1);
        if ph < (sr * 0.02) as usize {
            let env = 1.0 - ph as f64 / (sr * 0.02);
            v += env * 0.7 * (two_pi * 900.0 * t).sin();
        }
        out.push((v * f64::from(i16::MAX)) as i16);
    }
    out
}

/// Deterministic wideband noise (LCG), flat-ish spectrum.
fn noise_pcm(n: usize, stereo: bool) -> Vec<i16> {
    let mut state = 0x1234_5678u32;
    let nch = if stereo { 2 } else { 1 };
    let mut out = Vec::with_capacity(n * nch);
    for _ in 0..n * nch {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let v = f64::from((state >> 16) as i32 - 32768) / 32768.0;
        out.push((v * 0.35 * f64::from(i16::MAX)) as i16);
    }
    out
}

/// Synthesise `n` samples of a two-tone test signal (per channel a
/// different tone pair so a swapped-channel bug cannot cancel out).
/// Tones sit well inside the 4 kHz Nyquist of the lowest rate.
fn tone_pcm(n: usize, sample_rate: u32, stereo: bool) -> Vec<i16> {
    let sr = sample_rate as f64;
    let two_pi = 2.0 * std::f64::consts::PI;
    let nch = if stereo { 2 } else { 1 };
    let mut out = Vec::with_capacity(n * nch);
    for i in 0..n {
        let t = i as f64 / sr;
        let l = 0.35 * (two_pi * 440.0 * t).sin() + 0.15 * (two_pi * 1330.0 * t).sin();
        out.push((l * f64::from(i16::MAX)) as i16);
        if stereo {
            let r = 0.35 * (two_pi * 554.0 * t).sin() + 0.15 * (two_pi * 990.0 * t).sin();
            out.push((r * f64::from(i16::MAX)) as i16);
        }
    }
    out
}

/// Encode PCM to a complete MP3 byte stream.
fn encode(case: &Case, pcm: &[i16]) -> Vec<u8> {
    let mode = if case.stereo {
        ChannelMode::Stereo
    } else {
        ChannelMode::SingleChannel
    };
    let mut enc =
        Mp3Encoder::new(case.bitrate_kbps, case.sample_rate, mode).expect("Mp3Encoder build");
    if case.force_short {
        enc.force_short_blocks_for_testing(true)
            .expect("force short blocks");
    }
    if case.force_mixed {
        enc.force_mixed_blocks_for_testing(true)
            .expect("force mixed blocks");
    }
    if case.auto {
        enc.enable_auto_block_type(2.0).expect("auto block type");
    }
    if case.auto_mixed {
        // Relaxed low-band stability (8.0) as in
        // `tests/auto_block_type_mixed_roundtrip.rs`, so the Nyquist
        // clicks on top of the 50 Hz carrier still classify as
        // low-band-stable and promote to mixed.
        enc.enable_auto_block_type_with_mixed(2.0, 8.0)
            .expect("auto block type with mixed");
    }
    enc.push_samples(pcm).expect("push_samples");
    let mut bytes = Vec::new();
    enc.finish(&mut bytes).expect("finish");
    bytes
}

/// Decode an MP3 byte stream through the production `Mp3Demuxer` +
/// registered `Decoder` chain. Returns per-channel f32 planes.
fn own_decode(bytes: &[u8], channels: u16, sample_rate: u32) -> Vec<Vec<f32>> {
    let mut demux = Mp3Demuxer::open(Box::new(Cursor::new(bytes.to_vec()))).expect("demux open");
    let mut ctx = RuntimeContext::default();
    oxideav_mp3::register(&mut ctx);
    let mut params = CodecParameters::audio(CodecId::new("mp3"));
    params.sample_rate = Some(sample_rate);
    params.channels = Some(channels);
    params.sample_format = Some(SampleFormat::S16);
    let mut dec: Box<dyn Decoder> = ctx
        .codecs
        .first_decoder(&params)
        .expect("decoder factory present");
    let mut out: Vec<Vec<f32>> = vec![Vec::new(); channels as usize];
    loop {
        let pkt = match demux.next_packet() {
            Ok(p) => p,
            Err(Error::Eof) => break,
            Err(e) => panic!("next_packet: {e}"),
        };
        dec.send_packet(&pkt).expect("send_packet");
        loop {
            match dec.receive_frame() {
                Ok(Frame::Audio(a)) => {
                    assert_eq!(a.data.len(), channels as usize, "plane count");
                    for (ch, plane) in a.data.iter().enumerate() {
                        for chunk in plane.chunks_exact(2) {
                            let v = i16::from_le_bytes([chunk[0], chunk[1]]);
                            out[ch].push(f32::from(v) / 32768.0);
                        }
                    }
                }
                Ok(other) => panic!("non-audio frame: {other:?}"),
                Err(Error::NeedMore) => break,
                Err(e) => panic!("receive_frame: {e}"),
            }
        }
    }
    dec.flush().expect("flush");
    out
}

/// Run the injected validator command on `mp3_path`, producing
/// `wav_path`. Panics with the command's stderr on failure.
fn run_validator(template: &str, mp3_path: &Path, wav_path: &Path) {
    let cmd = template
        .replace("{IN}", &mp3_path.display().to_string())
        .replace("{OUT}", &wav_path.display().to_string());
    let output = Command::new("sh")
        .arg("-c")
        .arg(&cmd)
        .output()
        .expect("spawn validator");
    assert!(
        output.status.success() && wav_path.exists(),
        "validator failed on {}:\n{}",
        mp3_path.display(),
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Minimal RIFF/WAVE reader for 16-bit LE integer PCM.
/// Returns `(channels, sample_rate, interleaved f32 in [-1, 1])`.
fn read_wav_s16(path: &Path) -> (u16, u32, Vec<f32>) {
    let bytes = fs::read(path).expect("read wav");
    assert!(bytes.len() > 44 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WAVE");
    let mut pos = 12usize;
    let mut channels = 0u16;
    let mut sample_rate = 0u32;
    let mut data: Vec<f32> = Vec::new();
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let len = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let body = &bytes[pos + 8..(pos + 8 + len).min(bytes.len())];
        match id {
            b"fmt " => {
                let fmt_tag = u16::from_le_bytes(body[0..2].try_into().unwrap());
                assert_eq!(fmt_tag, 1, "validator must emit integer PCM");
                channels = u16::from_le_bytes(body[2..4].try_into().unwrap());
                sample_rate = u32::from_le_bytes(body[4..8].try_into().unwrap());
                let bits = u16::from_le_bytes(body[14..16].try_into().unwrap());
                assert_eq!(bits, 16, "validator must emit 16-bit PCM");
            }
            b"data" => {
                assert!(channels > 0, "fmt chunk must precede data");
                for chunk in body.chunks_exact(2) {
                    let v = i16::from_le_bytes([chunk[0], chunk[1]]);
                    data.push(f32::from(v) / 32768.0);
                }
            }
            _ => {}
        }
        pos += 8 + len + (len & 1);
    }
    (channels, sample_rate, data)
}

/// Split interleaved PCM into per-channel planes.
fn deinterleave(interleaved: &[f32], channels: usize) -> Vec<Vec<f32>> {
    let mut planes = vec![Vec::with_capacity(interleaved.len() / channels); channels];
    for (i, &v) in interleaved.iter().enumerate() {
        planes[i % channels].push(v);
    }
    planes
}

/// Best-correlation lag of `b` relative to `a` plus the normalized RMS
/// error at that lag over the steady-state region, summed across
/// channels. Same convention as `tests/corpus_reference_pcm.rs`.
fn align_and_error(
    a: &[Vec<f32>],
    b: &[Vec<f32>],
    max_lag: isize,
    win: usize,
) -> (isize, f64, f64) {
    assert_eq!(a.len(), b.len(), "channel count mismatch");
    let anchor = 3 * 576usize;
    let (pa_, pb_) = (&a[0], &b[0]);
    let mut best = (0isize, f64::NEG_INFINITY);
    for lag in -max_lag..=max_lag {
        let mut dot = 0f64;
        let mut pa = 0f64;
        let mut pb = 0f64;
        let mut n = 0usize;
        for k in 0..win {
            let i = anchor + k;
            let j = i as isize + lag;
            if j < 0 || j as usize >= pb_.len() || i >= pa_.len() {
                continue;
            }
            let (x, y) = (f64::from(pa_[i]), f64::from(pb_[j as usize]));
            dot += x * y;
            pa += x * x;
            pb += y * y;
            n += 1;
        }
        if n > 1000 {
            let ncc = dot / (pa.sqrt() * pb.sqrt()).max(1e-30);
            if ncc > best.1 {
                best = (lag, ncc);
            }
        }
    }
    let (lag, peak_ncc) = best;
    let mut err = 0f64;
    let mut ref_pow = 0f64;
    let mut n = 0usize;
    for ch in 0..a.len() {
        let steady = 2 * 576..a[ch].len().saturating_sub(2 * 576);
        for i in steady {
            let j = i as isize + lag;
            if j < 0 || j as usize >= b[ch].len() {
                continue;
            }
            let d = f64::from(a[ch][i]) - f64::from(b[ch][j as usize]);
            err += d * d;
            ref_pow += f64::from(a[ch][i]) * f64::from(a[ch][i]);
            n += 1;
        }
    }
    assert!(n > 1000, "overlap too small after alignment");
    (lag, (err / ref_pow.max(1e-30)).sqrt(), peak_ncc)
}

#[test]
fn validator_decodes_encoder_output_at_every_rate() {
    let Ok(template) = env::var("OXIDEAV_MP3_VALIDATOR_DECODE") else {
        eprintln!(
            "skip: OXIDEAV_MP3_VALIDATOR_DECODE not set \
             (external black-box validator sweep runs locally only)"
        );
        return;
    };
    let work = env::temp_dir().join(format!(
        "oxideav-mp3-validator-sweep-{}",
        std::process::id()
    ));
    fs::create_dir_all(&work).expect("create work dir");

    let mut failures: Vec<String> = Vec::new();
    for case in CASES {
        let channels: u16 = if case.stereo { 2 } else { 1 };
        // ~1.2 s of audio — enough frames for reservoir traffic and a
        // steady-state interior at every rate.
        let n = (case.sample_rate as usize * 12) / 10;
        let pcm = if case.noise {
            noise_pcm(n, case.stereo)
        } else if case.auto {
            attack_pcm(n, case.sample_rate)
        } else if case.auto_mixed {
            lf_dc_with_hf_click_pcm(n, case.sample_rate)
        } else {
            tone_pcm(n, case.sample_rate, case.stereo)
        };
        let mp3 = encode(case, &pcm);
        assert!(!mp3.is_empty(), "{}: empty encode", case.label);

        let mp3_path = work.join(format!("{}.mp3", case.label));
        let wav_path = work.join(format!("{}.wav", case.label));
        fs::write(&mp3_path, &mp3).expect("write mp3");
        run_validator(&template, &mp3_path, &wav_path);

        let (v_ch, v_rate, v_pcm) = read_wav_s16(&wav_path);
        if v_ch != channels {
            failures.push(format!(
                "{}: validator reports {} channels, encoded {}",
                case.label, v_ch, channels
            ));
            continue;
        }
        if v_rate != case.sample_rate {
            failures.push(format!(
                "{}: validator reports {} Hz, encoded {} Hz",
                case.label, v_rate, case.sample_rate
            ));
            continue;
        }

        let ours = own_decode(&mp3, channels, case.sample_rate);
        let val = deinterleave(&v_pcm, channels as usize);
        let (lag, nrmse, ncc) = align_and_error(&val, &ours, 6 * 576, 8 * 576);
        eprintln!(
            "{:<28} val={:>6} ours={:>6} lag={lag:>5} ncc={ncc:.5} nrmse={nrmse:.3e}",
            case.label,
            val[0].len(),
            ours[0].len(),
        );
        if ncc <= 0.999 {
            failures.push(format!("{}: alignment peak NCC {ncc:.5}", case.label));
        }
        if nrmse >= NRMSE_BOUND {
            failures.push(format!(
                "{}: validator-vs-own nrmse {nrmse:.3e} >= {NRMSE_BOUND:.0e}",
                case.label
            ));
        }
    }
    let _ = fs::remove_dir_all(&work);
    assert!(
        failures.is_empty(),
        "validator decode sweep failures:\n{}",
        failures.join("\n")
    );
}
