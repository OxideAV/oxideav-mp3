//! Live black-box matrix, direction **foreign → ours**.
//!
//! A black-box Layer III encoder (`lame`) writes a wide matrix of
//! streams into a temp dir; every stream is decoded by this crate
//! (demuxer + registry decoder) and by two black-box decoders
//! (`mpg123`, `ffmpeg`), and the PCM is compared sample by sample.
//!
//! Matrix axes (see [`build_cases`]):
//!
//! * every sampling frequency — MPEG-1 48 / 44.1 / 32 kHz, MPEG-2 LSF
//!   24 / 22.05 / 16 kHz, MPEG-2.5 12 / 11.025 / 8 kHz;
//! * the full bitrate ladder of each version, CBR, in every channel
//!   mode the encoder offers: joint stereo (adaptive MS), plain stereo,
//!   forced MS, dual channel, mono;
//! * free-format streams (below, between and above the ladder);
//! * VBR (two rate-control routines, quality 0..9) and ABR;
//! * CRC protection, reservoir disabled, strict buffer constraints,
//!   emphasis / copyright / original header bits, amplified
//!   (clipping) input, no information frame, forced information frame,
//!   ID3v1 / ID3v2 / padded ID3v2 tags;
//! * transient material throughout (short-block granules, reservoir
//!   swings, `scalefac_scale` / `preflag` at the low rates);
//! * tiny inputs (1 sample .. just over one frame) and a long stream;
//! * damaged files: truncated tail, truncated head, junk in front,
//!   two different streams concatenated.
//!
//! Judgement. ISO/IEC 11172-3 specifies the decoder output exactly up
//! to arithmetic precision, so on an intact stream this crate's 16-bit
//! output is compared with each black-box decoder's **float** output:
//! every sample within [`MAX_LSB`] LSB (2^-14 of full scale) and an
//! RMS difference of at most [`MAX_RMS_LSB`] LSB — the rounding of our
//! own 16-bit output (1/sqrt(12) LSB) plus the same again for
//! arithmetic differences, in quadrature. That is tighter than the
//! "<= 2 LSB" compare the staged conformance record
//! (`iso-13818-4-audio-conformance.md`) uses for its anchors. The
//! sample count must equal the ungapped black-box count exactly.
//! Damaged files have no specified output for the damaged frames;
//! there the test requires no error, a sample count within two frames,
//! and agreement on the undamaged interior.
//!
//! When the two black-box decoders disagree **with each other** on a
//! cell, the cell is reported as a *split* and passes if this crate
//! agrees with one of them; a cell one black-box decoder cannot read
//! at all (free format through `ffmpeg`) is judged on the other.
//!
//! The test skips (with a log line) when `lame` or `mpg123` is not
//! installed; the `ffmpeg` comparison is added when it is present.
//! `OXIDEAV_MP3_MATRIX_VERBOSE=1` prints one line per cell.

use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use oxideav_core::{CodecId, CodecParameters, Decoder, Demuxer, Error, Frame};
use oxideav_mp3::Mp3Demuxer;
use oxideav_mp3::codec_decoder::make_decoder;

/// Largest per-sample difference accepted on an intact stream.
const MAX_LSB: f64 = 2.0;
/// Largest RMS difference (in 16-bit LSB) accepted on an intact stream.
const MAX_RMS_LSB: f64 = 0.41;

fn have(tool: &str) -> bool {
    Command::new(tool)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok()
        || Command::new(tool)
            .arg("-version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok()
}

/// Deterministic test material: tones, a noise bed, and sharp clicks
/// every ~130 ms (attacks), different in the two channels so that
/// stereo decisions matter. Interleaved 16-bit.
fn material(rate: u32, channels: usize, samples: usize, seed: u32) -> Vec<i16> {
    let mut state = 0x1234_5678u32 ^ seed.wrapping_mul(0x9E37_79B9);
    let mut noise = move || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((state >> 8) as f64 / (1u32 << 24) as f64) * 2.0 - 1.0
    };
    let r = rate as f64;
    let click_period = (0.13 * r) as usize;
    let mut out = Vec::with_capacity(samples * channels);
    let mut env = [0.0f64; 2];
    for n in 0..samples {
        let t = n as f64 / r;
        let tau = std::f64::consts::TAU;
        let nz = noise();
        let nz2 = noise();
        // Quiet stretch in the middle third exercises low-level coding.
        let level = if n > samples / 3 && n < samples / 2 {
            0.08
        } else {
            1.0
        };
        for (ch, e) in env.iter_mut().enumerate().take(channels) {
            if click_period > 0 && (n + ch * click_period / 2).is_multiple_of(click_period) {
                *e = 1.0;
            }
            let f1 = if ch == 0 { 440.0 } else { 554.4 };
            let f2 = (0.11 + 0.05 * ch as f64) * r;
            let f3 = 0.31 * r;
            let tone = 0.22 * (tau * f1 * t).sin()
                + 0.12 * (tau * f2 * t).sin()
                + 0.05 * (tau * f3 * t + ch as f64).sin();
            let bed = 0.04 * if ch == 0 { nz } else { 0.5 * nz + 0.5 * nz2 };
            let click = 0.45 * *e * if ch == 0 { nz2 } else { nz };
            *e *= 0.985f64.powf(44_100.0 / r);
            let v = level * (tone + bed) + click;
            out.push((v.clamp(-1.0, 1.0) * 30_000.0) as i16);
        }
    }
    out
}

fn write_wav(path: &Path, rate: u32, channels: usize, pcm: &[i16]) {
    let data_len = (pcm.len() * 2) as u32;
    let mut b = Vec::with_capacity(44 + pcm.len() * 2);
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&(36 + data_len).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&(channels as u16).to_le_bytes());
    b.extend_from_slice(&rate.to_le_bytes());
    b.extend_from_slice(&(rate * channels as u32 * 2).to_le_bytes());
    b.extend_from_slice(&((channels * 2) as u16).to_le_bytes());
    b.extend_from_slice(&16u16.to_le_bytes());
    b.extend_from_slice(b"data");
    b.extend_from_slice(&data_len.to_le_bytes());
    for s in pcm {
        b.extend_from_slice(&s.to_le_bytes());
    }
    std::fs::write(path, b).expect("write wav");
}

/// This crate's decode of a whole file: interleaved PCM, channel
/// count of the first audio frame, or the error text.
fn ours(bytes: &[u8]) -> Result<(Vec<i16>, usize), String> {
    let mut d = Mp3Demuxer::open(Box::new(Cursor::new(bytes.to_vec())))
        .map_err(|e| format!("open: {e}"))?;
    let mut dec: Box<dyn Decoder> = make_decoder(&CodecParameters::audio(CodecId::new("mp3")))
        .map_err(|e| format!("make_decoder: {e}"))?;
    let mut pcm = Vec::new();
    let mut channels = 0usize;
    let drain =
        |dec: &mut dyn Decoder, pcm: &mut Vec<i16>, channels: &mut usize| -> Result<(), String> {
            loop {
                match dec.receive_frame() {
                    Ok(Frame::Audio(af)) => {
                        let planes: Vec<Vec<i16>> = af
                            .data
                            .iter()
                            .map(|pl| {
                                pl.as_chunks::<2>()
                                    .0
                                    .iter()
                                    .map(|c| i16::from_le_bytes(*c))
                                    .collect()
                            })
                            .collect();
                        if *channels == 0 {
                            *channels = planes.len();
                        }
                        // A mid-stream layout change keeps the first
                        // layout's interleave width (mono is doubled,
                        // stereo keeps its first plane) — the same
                        // thing the comparison decoders are asked for.
                        let n = planes[0].len();
                        let width = *channels;
                        pcm.extend(
                            (0..n * width)
                                .map(|k| planes[(k % width).min(planes.len() - 1)][k / width]),
                        );
                    }
                    Ok(_) => {}
                    Err(Error::NeedMore) | Err(Error::Eof) => return Ok(()),
                    Err(e) => return Err(format!("receive_frame: {e}")),
                }
            }
        };
    loop {
        match d.next_packet() {
            Ok(p) => {
                dec.send_packet(&p)
                    .map_err(|e| format!("send_packet: {e}"))?;
                drain(dec.as_mut(), &mut pcm, &mut channels)?;
            }
            Err(Error::Eof) => break,
            Err(e) => return Err(format!("next_packet: {e}")),
        }
    }
    dec.flush().map_err(|e| format!("flush: {e}"))?;
    drain(dec.as_mut(), &mut pcm, &mut channels)?;
    Ok((pcm, channels))
}

/// Float PCM (little-endian `f32`, full scale 1.0) in 16-bit LSB.
fn f32_lsb(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| (f32::from_le_bytes(*c) * 32768.0).clamp(-32768.0, 32767.0))
        .collect()
}

/// `mpg123` without gapless trimming: every decoded frame, float.
fn decode_mpg123(mp3: &Path, channels: usize) -> Result<Vec<f32>, String> {
    let mut cmd = Command::new("mpg123");
    cmd.args(["-q", "--no-gapless", "-e", "f32", "-s"]);
    cmd.arg(if channels == 1 { "--mono" } else { "--stereo" });
    let out = cmd
        .arg(mp3)
        .stderr(Stdio::null())
        .output()
        .map_err(|e| e.to_string())?;
    Ok(f32_lsb(&out.stdout))
}

fn decode_ffmpeg(mp3: &Path, channels: usize) -> Result<Vec<f32>, String> {
    let out = Command::new("ffmpeg")
        .args(["-v", "quiet", "-nostdin", "-i"])
        .arg(mp3)
        .args(["-ac", &channels.to_string(), "-f", "f32le", "-"])
        .stderr(Stdio::null())
        .output()
        .map_err(|e| e.to_string())?;
    Ok(f32_lsb(&out.stdout))
}

/// Compare `a` (ours) against `b` (reference) with `b[i] = a[i + lag]`
/// over frames `[from, to)` of `b` (interleaved, `ch` channels).
/// Returns (max abs diff, rms diff, rms limit that applies) or None
/// when the range is empty.
fn diff<A: Copy + Into<f64>>(
    a: &[A],
    b: &[f32],
    ch: usize,
    lag: isize,
    from: usize,
    to: usize,
) -> Option<(f64, f64, f64)> {
    let mut max = 0f64;
    let mut sq = 0f64;
    let mut n = 0usize;
    for i in from..to {
        let j = i as isize + lag;
        if j < 0 {
            continue;
        }
        let (ai, bi) = (j as usize * ch, i * ch);
        if ai + ch > a.len() || bi + ch > b.len() {
            break;
        }
        for c in 0..ch {
            let d: f64 = a[ai + c].into() - f64::from(b[bi + c]);
            max = max.max(d.abs());
            sq += d * d;
            n += 1;
        }
    }
    // An RMS over a handful of samples is not a statistic: below
    // 1000 samples only the maximum is judged (the RMS limit is
    // lifted to the maximum limit).
    let rms = (sq / n.max(1) as f64).sqrt();
    let limit = if n >= 1000 { MAX_RMS_LSB } else { MAX_LSB };
    (n > 0).then_some((max, rms, limit))
}

/// Best lag (in frames) aligning the reference to our decode, searched
/// over ±`span` on a window in the middle of the first `upto` frames
/// of the reference.
fn best_lag<A: Copy + Into<f64>>(a: &[A], b: &[f32], ch: usize, span: isize, upto: usize) -> isize {
    let frames = (b.len() / ch).min(upto);
    let w = frames.min(1500);
    let from = (frames - w) / 2;
    let mut best = (f64::MAX, 0isize);
    for lag in -span..=span {
        if let Some((_, rms, _)) = diff(a, b, ch, lag, from, from + w)
            && rms < best.0
        {
            best = (rms, lag);
        }
    }
    best.1
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    /// Intact stream: exact length, full-range comparison.
    Intact,
    /// Damaged stream: interior comparison only.
    Damaged,
}

struct Case {
    label: String,
    rate: u32,
    channels: usize,
    samples: usize,
    args: Vec<String>,
    kind: Kind,
    /// Post-processing of the encoded bytes (damage injection).
    mangle: Mangle,
    /// A second stream to hand to `mangle` (concatenation), as
    /// (rate, channels, args).
    second: Option<(u32, usize, Vec<String>)>,
    /// The second black-box decoder resamples after a sampling
    /// frequency switch; such cells are judged on the first alone.
    no_ffmpeg: bool,
}

/// Damage injection: (first stream, second stream) -> file bytes.
type Mangle = fn(Vec<u8>, &[u8]) -> Vec<u8>;

fn keep(a: Vec<u8>, _: &[u8]) -> Vec<u8> {
    a
}

const RATES: [u32; 9] = [
    48_000, 44_100, 32_000, 24_000, 22_050, 16_000, 12_000, 11_025, 8_000,
];

fn ladder(rate: u32) -> &'static [u32] {
    if rate >= 32_000 {
        &[
            32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320,
        ]
    } else if rate >= 16_000 {
        &[8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160]
    } else {
        // The black-box encoder offers 8..64 at the MPEG-2.5 rates.
        &[8, 16, 24, 32, 40, 48, 56, 64]
    }
}

fn khz(rate: u32) -> String {
    format!("{}", f64::from(rate) / 1000.0)
}

fn sv(a: &[&str]) -> Vec<String> {
    a.iter().map(|s| s.to_string()).collect()
}

fn build_cases() -> Vec<Case> {
    let mut v = Vec::new();
    let mut add = |label: String, rate: u32, channels: usize, secs: f64, args: Vec<String>| {
        let free = args.iter().any(|a| a == "--freeformat");
        v.push(Case {
            label,
            rate,
            channels,
            samples: (secs * f64::from(rate)) as usize,
            args,
            kind: Kind::Intact,
            mangle: keep,
            second: None,
            // The second black-box decoder does not read free format.
            no_ffmpeg: free,
        });
    };
    // A. CBR: every rate x full ladder x every channel mode.
    for rate in RATES {
        for &br in ladder(rate) {
            for mode in ["j", "s", "f", "d", "m"] {
                let ch = if mode == "m" { 1 } else { 2 };
                let mut a = sv(&["--cbr", "-b", &br.to_string(), "-m", mode]);
                a.extend(sv(&["--resample", &khz(rate)]));
                add(format!("cbr-{rate}-{br}k-{mode}"), rate, ch, 1.3, a);
            }
        }
    }
    // B. Free format.
    for rate in RATES {
        let rates: &[u32] = if rate >= 32_000 {
            &[20, 100, 300, 420, 600]
        } else if rate >= 16_000 {
            &[12, 100, 200, 300]
        } else {
            &[12, 36, 100, 150]
        };
        for &br in rates {
            for mode in ["j", "m"] {
                let ch = if mode == "m" { 1 } else { 2 };
                let mut a = sv(&["--freeformat", "-b", &br.to_string(), "-m", mode]);
                a.extend(sv(&["--resample", &khz(rate)]));
                add(format!("free-{rate}-{br}k-{mode}"), rate, ch, 1.3, a);
            }
        }
    }
    // C. VBR (both routines) and ABR.
    for rate in RATES {
        for q in [0, 2, 4, 6, 9] {
            for routine in ["--vbr-new", "--vbr-old"] {
                for mode in ["j", "m"] {
                    let ch = if mode == "m" { 1 } else { 2 };
                    let mut a = sv(&[routine, "-V", &q.to_string(), "-m", mode]);
                    a.extend(sv(&["--resample", &khz(rate)]));
                    add(format!("vbr-{rate}-V{q}{routine}-{mode}"), rate, ch, 1.3, a);
                }
            }
        }
        let l = ladder(rate);
        for abr in [l[1], l[l.len() / 2], l[l.len() - 2]] {
            let mut a = sv(&["--abr", &abr.to_string()]);
            a.extend(sv(&["--resample", &khz(rate)]));
            add(format!("abr-{rate}-{abr}k"), rate, 2, 1.3, a);
        }
    }
    // D. Stream options at one rate per MPEG version.
    let flags: &[&[&str]] = &[
        &["-p"],
        &["--nores"],
        &["--strictly-enforce-ISO"],
        &["--buffer-constraint", "strict"],
        &["--buffer-constraint", "maximum"],
        &["-t"],
        &["-q", "0"],
        &["-q", "9"],
        &["-c", "-o"],
        &["-e", "5"],
        &["-e", "c"],
        &["--scale", "3"],
        &["--scale", "0.01"],
        &["--lowpass", "3"],
        &["--tt", "title", "--ta", "artist", "--id3v1-only"],
        &["--tt", "title", "--ta", "artist", "--id3v2-only"],
        &["--tt", "title", "--add-id3v2", "--pad-id3v2-size", "3000"],
        &["--tt", "title", "--add-id3v2", "--id3v2-utf16"],
        &["-p", "--nores", "-m", "s"],
        &["-p", "-m", "m"],
        &["-p", "-V", "3"],
        &["-V", "5", "-t"],
        &["-V", "2", "-b", "128", "-F"],
        &["-V", "8", "-B", "64"],
        &["-V", "4", "-T"],
    ];
    for (rate, br) in [(44_100u32, 128u32), (22_050, 64), (11_025, 32)] {
        for f in flags {
            let ch = if f.contains(&"m") && f.contains(&"-m") && !f.contains(&"s") {
                1
            } else {
                2
            };
            let mut a = if f.contains(&"-V") {
                Vec::new()
            } else {
                sv(&["-b", &br.to_string()])
            };
            a.extend(sv(f));
            a.extend(sv(&["--resample", &khz(rate)]));
            add(format!("opt-{rate}-{}", f.join("")), rate, ch, 1.3, a);
        }
    }
    // E. Tiny inputs and one long stream.
    for rate in [44_100u32, 22_050, 8_000] {
        for n in [1usize, 100, 576, 1152, 1153, 2305] {
            for mode in ["j", "m"] {
                let ch = if mode == "m" { 1 } else { 2 };
                let mut a = sv(&["-m", mode, "--resample", &khz(rate)]);
                a.extend(sv(&["-b", if rate > 30_000 { "128" } else { "32" }]));
                add(
                    format!("tiny-{rate}-{n}-{mode}"),
                    rate,
                    ch,
                    n as f64 / f64::from(rate),
                    a,
                );
            }
        }
    }
    add(
        "long-44100-V4".into(),
        44_100,
        2,
        40.0,
        sv(&["-V", "4", "--resample", "44.1"]),
    );
    add(
        "long-44100-cbr128".into(),
        44_100,
        2,
        40.0,
        sv(&["-b", "128", "--resample", "44.1"]),
    );
    // F. Damaged files.
    let cut_tail_1: Mangle = |mut a, _| {
        a.truncate(a.len() - 1);
        a
    };
    let cut_tail_half: Mangle = |mut a, _| {
        a.truncate(a.len() - 200);
        a
    };
    let cut_head: Mangle = |a, _| a[1500..].to_vec();
    let junk_head: Mangle = |a, _| {
        let mut j: Vec<u8> = (0..777u32).map(|i| (i * 37 % 251) as u8).collect();
        j.extend_from_slice(&a);
        j
    };
    let concat: Mangle = |mut a, b| {
        a.extend_from_slice(b);
        a
    };
    let damage: [(&str, Mangle); 4] = [
        ("cut-tail-1", cut_tail_1),
        ("cut-tail-200", cut_tail_half),
        ("cut-head-1500", cut_head),
        ("junk-head", junk_head),
    ];
    let mut dmg = Vec::new();
    for (rate, br) in [(44_100u32, "128"), (22_050, "64"), (8_000, "24")] {
        for (name, f) in damage {
            for tagarg in ["-t", "-T"] {
                dmg.push(Case {
                    label: format!("damaged-{rate}-{name}{tagarg}"),
                    rate,
                    channels: 2,
                    samples: (1.6 * f64::from(rate)) as usize,
                    args: sv(&["-b", br, tagarg, "--resample", &khz(rate)]),
                    kind: Kind::Damaged,
                    mangle: f,
                    second: None,
                    no_ffmpeg: false,
                });
            }
        }
    }
    for (r1, r2) in [
        (44_100u32, 44_100u32),
        (44_100, 22_050),
        (48_000, 8_000),
        (16_000, 32_000),
    ] {
        for tagarg in ["-t", "-T"] {
            dmg.push(Case {
                label: format!("concat-{r1}-{r2}{tagarg}"),
                rate: r1,
                channels: 2,
                samples: (1.2 * f64::from(r1)) as usize,
                args: sv(&["-V", "5", tagarg, "--resample", &khz(r1)]),
                kind: Kind::Damaged,
                mangle: concat,
                second: Some((r2, 2, sv(&["-V", "5", tagarg, "--resample", &khz(r2)]))),
                no_ffmpeg: r1 != r2,
            });
        }
    }
    v.extend(dmg);
    v
}

fn encode(
    dir: &Path,
    idx: usize,
    tag: &str,
    rate: u32,
    ch: usize,
    n: usize,
    args: &[String],
) -> Result<Vec<u8>, String> {
    let wav = dir.join(format!("{idx}{tag}.wav"));
    let mp3 = dir.join(format!("{idx}{tag}.enc.mp3"));
    write_wav(&wav, rate, ch, &material(rate, ch, n, idx as u32));
    let st = Command::new("lame")
        .arg("--quiet")
        .args(args)
        .arg(&wav)
        .arg(&mp3)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map_err(|e| e.to_string())?;
    let _ = std::fs::remove_file(&wav);
    if !st.success() {
        return Err("encoder refused".into());
    }
    let b = std::fs::read(&mp3).map_err(|e| e.to_string())?;
    let _ = std::fs::remove_file(&mp3);
    Ok(b)
}

enum Verdict {
    Pass(String),
    /// The two black-box decoders disagree with each other; this
    /// crate agrees with one of them.
    Split(String),
    /// No two decoders agree: the black-box decoders differ from each
    /// other and from this crate, so the cell has no reference.
    Contested(String),
    /// The black-box encoder does not make this cell.
    NotMade,
    Fail(String),
}

/// Compare decode `a` with reference decode `b` under the rules of
/// `kind`. `exact` demands equal length and zero lag.
fn compare<A: Copy + Into<f64>>(
    a: &[A],
    b: &[f32],
    ch: usize,
    c: &Case,
    exact: bool,
) -> Result<String, String> {
    let (af, bf) = (a.len() / ch, b.len() / ch);
    let spf = if c.rate >= 32_000 { 1152 } else { 576 };
    if af == 0 && bf == 0 {
        return Ok("empty".into());
    }
    if af == 0 || bf == 0 {
        return Err(format!("length {af} vs {bf}"));
    }
    match c.kind {
        Kind::Intact => {
            if exact && af != bf {
                return Err(format!("length {af} vs {bf}"));
            }
            if af.abs_diff(bf) > 3 * 1152 {
                return Err(format!("length {af} vs {bf}"));
            }
            // A decoder that trims its own start delay is aligned by
            // search.
            let lag = if exact {
                0
            } else {
                best_lag(a, b, ch, 2 * 1152 + 600, usize::MAX)
            };
            let Some((max, rms, rms_limit)) = diff(a, b, ch, lag, 0, bf) else {
                return Err("no overlap".into());
            };
            if max > MAX_LSB || rms > rms_limit {
                return Err(format!("max {max:.2} LSB rms {rms:.3} lag {lag}"));
            }
            Ok(format!("{max:.2}/{rms:.2}"))
        }
        Kind::Damaged => {
            if af.abs_diff(bf) > 2 * spf + 1152 + 600 {
                return Err(format!("length {af} vs {bf}"));
            }
            // Interior of the first segment: skip four frames at each
            // end (the reservoir of a cut stream needs that long to
            // refill, and the tail may be cut).
            let seg = c.samples.min(bf);
            let lag = best_lag(a, b, ch, 3 * 1152 + 600, seg);
            let from = 4 * spf;
            let to = seg.saturating_sub(5 * spf + 1500);
            let Some((max, rms, rms_limit)) = diff(a, b, ch, lag, from, to) else {
                return Err("no interior overlap".into());
            };
            if max > MAX_LSB || rms > rms_limit {
                return Err(format!("interior max {max:.2} LSB rms {rms:.3} lag {lag}"));
            }
            Ok(format!("{max:.2}/{rms:.2}@{lag}({af}/{bf})"))
        }
    }
}

fn run_case(dir: &Path, idx: usize, c: &Case, with_ffmpeg: bool) -> Verdict {
    let first = match encode(dir, idx, "a", c.rate, c.channels, c.samples, &c.args) {
        Ok(b) => b,
        Err(_) => return Verdict::NotMade,
    };
    let second = match &c.second {
        Some((r, ch, args)) => {
            match encode(dir, idx, "b", *r, *ch, (1.2 * f64::from(*r)) as usize, args) {
                Ok(b) => b,
                Err(_) => return Verdict::NotMade,
            }
        }
        None => Vec::new(),
    };
    let bytes = (c.mangle)(first, &second);
    let mp3 = dir.join(format!("{idx}.mp3"));
    std::fs::write(&mp3, &bytes).expect("write mp3");
    let res = (|| -> Result<Verdict, String> {
        let (a, ch) = match std::panic::catch_unwind(|| ours(&bytes)) {
            Ok(r) => r?,
            Err(_) => return Err("PANIC in this crate's decode".into()),
        };
        let ch = ch.max(c.channels);
        // The gapless figures of an untouched stream that carries the
        // information frame give back the input length exactly. (The
        // black-box encoder leaves the frame out where an audio frame
        // of the stream is too small to hold it.)
        if c.kind == Kind::Intact {
            let d = Mp3Demuxer::open(Box::new(Cursor::new(bytes.clone())))
                .map_err(|e| format!("open: {e}"))?;
            if d.xing().is_some() && d.trimmed_duration_samples() != Some(c.samples as i64) {
                return Err(format!(
                    "gapless length {:?} for {} input samples (delay {:?}, padding {:?})",
                    d.trimmed_duration_samples(),
                    c.samples,
                    d.encoder_delay_samples(),
                    d.zero_padding_samples()
                ));
            }
        }
        let m = decode_mpg123(&mp3, ch)?;
        let vm = compare(&a, &m, ch, c, c.kind == Kind::Intact).map_err(|e| format!("mpg123: {e}"));
        if !with_ffmpeg || c.no_ffmpeg {
            return vm.map(|n| Verdict::Pass(format!(" mpg123:{n}")));
        }
        let f = decode_ffmpeg(&mp3, ch)?;
        let vf = compare(&a, &f, ch, c, false).map_err(|e| format!("ffmpeg: {e}"));
        match (vm, vf) {
            (Ok(n1), Ok(n2)) => Ok(Verdict::Pass(format!(" mpg123:{n1} ffmpeg:{n2}"))),
            (Err(e1), Err(e2)) => Err(format!("{e1}; {e2}")),
            (Ok(n), Err(e)) | (Err(e), Ok(n)) => {
                // One disagrees: a split if the two references also
                // disagree with each other.
                match compare(&m, &f, ch, c, false) {
                    Ok(_) => Err(e),
                    Err(between) => Ok(Verdict::Split(format!(
                        " agrees with one ({n}); other: {e}; references differ: {between}"
                    ))),
                }
            }
        }
    })();
    match res {
        Ok(v) => {
            let _ = std::fs::remove_file(&mp3);
            v
        }
        Err(e) => Verdict::Fail(e),
    }
}

#[test]
fn foreign_streams_decode_like_the_black_box_decoders() {
    if !have("lame") || !have("mpg123") {
        eprintln!("skipping: `lame` and `mpg123` are required for the black-box matrix");
        return;
    }
    let with_ffmpeg = have("ffmpeg");
    let verbose = std::env::var_os("OXIDEAV_MP3_MATRIX_VERBOSE").is_some();
    let only = std::env::var("OXIDEAV_MP3_MATRIX_ONLY").ok();
    let dir: PathBuf =
        std::env::temp_dir().join(format!("oxideav-mp3-matrix-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let cases: Vec<Case> = build_cases()
        .into_iter()
        .filter(|c| only.as_ref().is_none_or(|o| c.label.contains(o.as_str())))
        .collect();
    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<(usize, Verdict)>> = Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..4 {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= cases.len() {
                        break;
                    }
                    let v = run_case(&dir, i, &cases[i], with_ffmpeg);
                    results.lock().unwrap().push((i, v));
                }
            });
        }
    });
    let mut results = results.into_inner().unwrap();
    results.sort_by_key(|r| r.0);
    let (mut pass, mut split, mut not_made) = (0usize, 0usize, 0usize);
    let mut fails = Vec::new();
    for (i, v) in &results {
        match v {
            Verdict::Pass(n) => {
                pass += 1;
                if verbose {
                    eprintln!("ok   {}{n}", cases[*i].label);
                }
            }
            Verdict::Split(n) => {
                split += 1;
                eprintln!("split {}{n}", cases[*i].label);
            }
            Verdict::Contested(n) => fails.push(format!("{} [{}.mp3]: {n}", cases[*i].label, i)),
            Verdict::NotMade => {
                not_made += 1;
                if verbose {
                    eprintln!("n/a  {}", cases[*i].label);
                }
            }
            Verdict::Fail(e) => fails.push(format!("{} [{}.mp3]: {e}", cases[*i].label, i)),
        }
    }
    eprintln!(
        "black-box matrix (foreign -> ours): {} cells, {} not produced by the encoder, {} compared: {} agree with every black-box decoder, {} agree with one where the two differ, {} fail (ffmpeg: {})",
        cases.len(),
        not_made,
        pass + split + fails.len(),
        pass,
        split,
        fails.len(),
        with_ffmpeg
    );
    if fails.is_empty() {
        let _ = std::fs::remove_dir_all(&dir);
    } else {
        for f in &fails {
            eprintln!("FAIL {f}");
        }
        panic!(
            "{} of {} cells disagree; streams kept in {}",
            fails.len(),
            pass + split + fails.len(),
            dir.display()
        );
    }
}

// ---------------------------------------------------------------------
// Direction ours -> foreign.
// ---------------------------------------------------------------------

use oxideav_mp3::{ChannelMode, Mp3Encoder, QualityPreset};

/// One encoder cell: a label and how to build the encoder.
struct EncCell {
    label: String,
    rate: u32,
    channels: usize,
    build: Box<dyn Fn() -> Option<Mp3Encoder> + Send + Sync>,
}

fn enc_cells() -> Vec<EncCell> {
    let mut v: Vec<EncCell> = Vec::new();
    let mut add = |label: String,
                   rate: u32,
                   channels: usize,
                   build: Box<dyn Fn() -> Option<Mp3Encoder> + Send + Sync>| {
        v.push(EncCell {
            label,
            rate,
            channels,
            build,
        });
    };
    // A. Every sampling frequency x the full ladder x every channel
    //    mode, default configuration.
    for rate in RATES {
        let full: &[u32] = if rate >= 32_000 {
            &[
                32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320,
            ]
        } else {
            &[8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160]
        };
        for &br in full {
            add(
                format!("enc-{rate}-{br}k-mono"),
                rate,
                1,
                Box::new(move || Mp3Encoder::new(br, rate, ChannelMode::SingleChannel).ok()),
            );
            add(
                format!("enc-{rate}-{br}k-stereo"),
                rate,
                2,
                Box::new(move || Mp3Encoder::new(br, rate, ChannelMode::Stereo).ok()),
            );
            add(
                format!("enc-{rate}-{br}k-dual"),
                rate,
                2,
                Box::new(move || Mp3Encoder::new(br, rate, ChannelMode::DualChannel).ok()),
            );
            add(
                format!("enc-{rate}-{br}k-ms"),
                rate,
                2,
                Box::new(move || Mp3Encoder::new_joint_stereo_ms(br, rate).ok()),
            );
            add(
                format!("enc-{rate}-{br}k-joint-auto"),
                rate,
                2,
                Box::new(move || Mp3Encoder::new_joint_stereo_auto(br, rate).ok()),
            );
        }
    }
    // B. Every encoder option, at one rate per MPEG version (and all
    //    three MPEG-2.5 rates, whose band tables differ), mono and
    //    stereo.
    type Opt = (&'static str, fn(&mut Mp3Encoder) -> bool);
    let opts: [Opt; 7] = [
        ("crc", |e| {
            e.with_protection_bit(true);
            true
        }),
        ("short", |e| e.force_short_blocks_for_testing(true).is_ok()),
        ("mixed", |e| e.force_mixed_blocks_for_testing(true).is_ok()),
        ("auto", |e| e.enable_auto_block_type(4.0).is_ok()),
        ("auto-mixed", |e| {
            e.enable_auto_block_type_with_mixed(4.0, 0.5).is_ok()
        }),
        ("scfsi", |e| {
            e.enable_scfsi_reuse();
            true
        }),
        ("crc+auto", |e| {
            e.with_protection_bit(true);
            e.enable_auto_block_type(4.0).is_ok()
        }),
    ];
    for (rate, br) in [
        (44_100u32, 128u32),
        (48_000, 192),
        (22_050, 64),
        (12_000, 32),
        (11_025, 32),
        (8_000, 24),
    ] {
        for (name, apply) in opts {
            for (mname, mode, ch) in [
                ("mono", ChannelMode::SingleChannel, 1usize),
                ("stereo", ChannelMode::Stereo, 2),
            ] {
                add(
                    format!("enc-{rate}-{br}k-{mname}-{name}"),
                    rate,
                    ch,
                    Box::new(move || {
                        let mut e = Mp3Encoder::new(br, rate, mode).ok()?;
                        apply(&mut e).then_some(e)
                    }),
                );
            }
            // Joint stereo with the same option.
            add(
                format!("enc-{rate}-{br}k-joint-auto-{name}"),
                rate,
                2,
                Box::new(move || {
                    let mut e = Mp3Encoder::new_joint_stereo_auto(br, rate).ok()?;
                    apply(&mut e).then_some(e)
                }),
            );
        }
        // Psychoacoustic constructors.
        for (mname, mode, ch) in [
            ("mono", ChannelMode::SingleChannel, 1usize),
            ("stereo", ChannelMode::Stereo, 2),
        ] {
            add(
                format!("enc-{rate}-{br}k-{mname}-tiq"),
                rate,
                ch,
                Box::new(move || Mp3Encoder::new_with_threshold_in_quiet(br, rate, mode).ok()),
            );
            add(
                format!("enc-{rate}-{br}k-{mname}-outer"),
                rate,
                ch,
                Box::new(move || Mp3Encoder::new_with_outer_loop(br, rate, mode, 1e-6).ok()),
            );
            for (pname, preset) in [
                ("fast", QualityPreset::Fast),
                ("standard", QualityPreset::Standard),
                ("high", QualityPreset::High),
                ("transparent", QualityPreset::Transparent),
            ] {
                add(
                    format!("enc-{rate}-{br}k-{mname}-new-preset-{pname}"),
                    rate,
                    ch,
                    Box::new(move || {
                        Mp3Encoder::new_with_quality_preset(br, rate, mode, preset).ok()
                    }),
                );
            }
            add(
                format!("enc-{rate}-{br}k-{mname}-model2-blocks"),
                rate,
                ch,
                Box::new(move || {
                    let mut e =
                        Mp3Encoder::new_with_quality_preset(br, rate, mode, QualityPreset::High)
                            .ok()?;
                    e.enable_auto_block_type_model2().is_ok().then_some(e)
                }),
            );
            // VBR over the whole ladder below the constructor rate.
            add(
                format!("enc-{rate}-{br}k-{mname}-vbr"),
                rate,
                ch,
                Box::new(move || {
                    let mut e = Mp3Encoder::new(br, rate, mode).ok()?;
                    let min = if rate >= 32_000 { 32 } else { 8 };
                    e.enable_vbr(min, br).is_ok().then_some(e)
                }),
            );
        }
        // Intensity stereo, alone and with MS, fixed and adaptive.
        for start in [4usize, 12, 18] {
            add(
                format!("enc-{rate}-{br}k-is{start}"),
                rate,
                2,
                Box::new(move || Mp3Encoder::new_joint_stereo_is(br, rate, start).ok()),
            );
            add(
                format!("enc-{rate}-{br}k-ms-is{start}"),
                rate,
                2,
                Box::new(move || Mp3Encoder::new_joint_stereo_ms_is(br, rate, start).ok()),
            );
            add(
                format!("enc-{rate}-{br}k-auto-is{start}"),
                rate,
                2,
                Box::new(move || Mp3Encoder::new_joint_stereo_auto_is(br, rate, start).ok()),
            );
            add(
                format!("enc-{rate}-{br}k-auto-is-adaptive{start}"),
                rate,
                2,
                Box::new(move || {
                    Mp3Encoder::new_joint_stereo_auto_is_adaptive(br, rate, start).ok()
                }),
            );
            for (name, apply) in [opts[1], opts[3]] {
                add(
                    format!("enc-{rate}-{br}k-ms-is{start}-{name}"),
                    rate,
                    2,
                    Box::new(move || {
                        let mut e = Mp3Encoder::new_joint_stereo_ms_is(br, rate, start).ok()?;
                        apply(&mut e).then_some(e)
                    }),
                );
            }
        }
    }
    v
}

/// A black-box decoder's gapless float decode and whatever it printed
/// on stderr at warning level.
fn gapless_decode(tool: &str, mp3: &Path, channels: usize) -> (Vec<f32>, String) {
    let out = if tool == "mpg123" {
        Command::new("mpg123")
            .args(["-q", "-e", "f32", "-s"])
            .arg(if channels == 1 { "--mono" } else { "--stereo" })
            .arg(mp3)
            .output()
    } else {
        Command::new("ffmpeg")
            .args(["-v", "warning", "-nostdin", "-i"])
            .arg(mp3)
            .args(["-f", "f32le", "-"])
            .output()
    }
    .expect("run decoder");
    (
        f32_lsb(&out.stdout),
        String::from_utf8_lossy(&out.stderr).trim().to_string(),
    )
}

fn run_enc_cell(dir: &Path, idx: usize, c: &EncCell, with_ffmpeg: bool) -> Verdict {
    let Some(mut enc) = (c.build)() else {
        return Verdict::NotMade;
    };
    // An odd length, so the last frame is partly padding.
    let n = c.rate as usize + 1234 + idx % 577;
    let pcm = material(c.rate, c.channels, n, idx as u32);
    enc.enable_info_frame();
    let mut bytes = Vec::new();
    if let Err(e) = enc.push_samples(&pcm).and_then(|()| enc.finish(&mut bytes)) {
        return Verdict::Fail(format!("encode: {e}"));
    }
    let mp3 = dir.join(format!("enc{idx}.mp3"));
    std::fs::write(&mp3, &bytes).expect("write mp3");
    let res = (|| -> Result<String, String> {
        // Our own reading of the stream.
        let d = Mp3Demuxer::open(Box::new(Cursor::new(bytes.clone())))
            .map_err(|e| format!("own demuxer: {e}"))?;
        if d.trimmed_duration_samples() != Some(n as i64) {
            return Err(format!(
                "own demuxer: gapless length {:?}, pushed {n}",
                d.trimmed_duration_samples()
            ));
        }
        let delay = d.encoder_delay_samples().unwrap_or(0) as usize + 529;
        let (a, ch) = ours(&bytes)?;
        if ch != c.channels {
            return Err(format!("own decoder: {ch} channels"));
        }
        if a.len() / ch < delay + n {
            return Err(format!(
                "stream ends before the last input sample ({})",
                a.len() / ch
            ));
        }
        let ours_trimmed = &a[delay * ch..(delay + n) * ch];
        let mut tools = vec!["mpg123"];
        if with_ffmpeg {
            tools.push("ffmpeg");
        }
        let mut decodes: Vec<(&str, Vec<f32>)> = Vec::new();
        let mut verdicts: Vec<Result<String, String>> = Vec::new();
        for tool in tools {
            let (b, stderr) = gapless_decode(tool, &mp3, ch);
            // One black-box decoder states that it does not implement
            // mixed blocks at 8 kHz; such a cell is judged on the other.
            if stderr.contains("is not implemented") {
                continue;
            }
            if !stderr.is_empty() {
                return Err(format!("{tool} complains: {stderr}"));
            }
            if b.len() / ch != n {
                return Err(format!(
                    "{tool}: gapless length {} for {n} pushed",
                    b.len() / ch
                ));
            }
            let Some((max, rms, limit)) = diff(ours_trimmed, &b, ch, 0, 0, n) else {
                return Err(format!("{tool}: nothing to compare"));
            };
            verdicts.push(if max > MAX_LSB || rms > limit {
                Err(format!("{tool}: max {max:.2} LSB rms {rms:.3}"))
            } else {
                Ok(format!(" {tool}:{max:.2}/{rms:.2}"))
            });
            decodes.push((tool, b));
        }
        let agree = verdicts.iter().filter(|v| v.is_ok()).count();
        let mut note: String = verdicts.iter().flatten().cloned().collect();
        if agree < verdicts.len() {
            let errs: Vec<String> = verdicts.iter().filter_map(|v| v.clone().err()).collect();
            // A cell passes as a split when the black-box decoders
            // disagree with each other and this crate agrees with one.
            let between = (decodes.len() == 2)
                .then(|| diff(&decodes[0].1, &decodes[1].1, ch, 0, 0, n))
                .flatten()
                .is_some_and(|(max, _, _)| max > MAX_LSB);
            if !between {
                return Err(errs.join("; "));
            }
            note = if agree == 0 {
                format!("CONTESTED {}", errs.join("; "))
            } else {
                format!("SPLIT{note}; {}", errs.join("; "))
            };
        }
        // The decode is the input, in place: the error against the
        // pushed PCM is smallest at zero lag.
        let input: Vec<f32> = pcm.iter().map(|&s| f32::from(s)).collect();
        let err_at = |lag: isize| diff(ours_trimmed, &input, ch, lag, 8, n - 8).map(|d| d.1);
        let e0 = err_at(0).unwrap_or(f64::MAX);
        for lag in [-2isize, -1, 1, 2] {
            if err_at(lag).is_some_and(|e| e < e0) {
                return Err(format!("decode is offset from the input by {lag}"));
            }
        }
        Ok(note)
    })();
    match res {
        Ok(n) => {
            let _ = std::fs::remove_file(&mp3);
            if n.starts_with("SPLIT") {
                Verdict::Split(n)
            } else if n.starts_with("CONTESTED") {
                Verdict::Contested(n)
            } else {
                Verdict::Pass(n)
            }
        }
        Err(e) => Verdict::Fail(e),
    }
}

#[test]
fn own_streams_decode_on_the_black_box_decoders() {
    if !have("mpg123") {
        eprintln!("skipping: `mpg123` is required for the black-box matrix");
        return;
    }
    let with_ffmpeg = have("ffmpeg");
    let verbose = std::env::var_os("OXIDEAV_MP3_MATRIX_VERBOSE").is_some();
    let only = std::env::var("OXIDEAV_MP3_MATRIX_ONLY").ok();
    let dir: PathBuf =
        std::env::temp_dir().join(format!("oxideav-mp3-matrix-enc-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let cells: Vec<EncCell> = enc_cells()
        .into_iter()
        .filter(|c| only.as_ref().is_none_or(|o| c.label.contains(o.as_str())))
        .collect();
    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<(usize, Verdict)>> = Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..4 {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= cells.len() {
                        break;
                    }
                    let v = run_enc_cell(&dir, i, &cells[i], with_ffmpeg);
                    results.lock().unwrap().push((i, v));
                }
            });
        }
    });
    let mut results = results.into_inner().unwrap();
    results.sort_by_key(|r| r.0);
    let (mut pass, mut split, mut contested, mut not_made) = (0usize, 0usize, 0usize, 0usize);
    let mut fails = Vec::new();
    for (i, v) in &results {
        match v {
            Verdict::Pass(n) => {
                pass += 1;
                if verbose {
                    eprintln!("ok   {}{n}", cells[*i].label);
                }
            }
            Verdict::Split(n) => {
                split += 1;
                if verbose {
                    eprintln!("split {} {n}", cells[*i].label);
                }
            }
            Verdict::Contested(n) => {
                contested += 1;
                eprintln!("contested {} {n}", cells[*i].label);
            }
            Verdict::NotMade => {
                not_made += 1;
                if verbose {
                    eprintln!("n/a  {}", cells[*i].label);
                }
            }
            Verdict::Fail(e) => fails.push(format!("{} [enc{}.mp3]: {e}", cells[*i].label, i)),
        }
    }
    eprintln!(
        "black-box matrix (ours -> foreign): {} cells, {} configurations the encoder does not offer, {} encoded: {} accepted without complaint, gapless-exact and sample-equal on every black-box decoder, {} where the black-box decoders differ from each other and this crate equals one, {} where no two decoders agree, {} fail (ffmpeg: {})",
        cells.len(),
        not_made,
        pass + split + contested + fails.len(),
        pass,
        split,
        contested,
        fails.len(),
        with_ffmpeg
    );
    if fails.is_empty() {
        let _ = std::fs::remove_dir_all(&dir);
    } else {
        for f in &fails {
            eprintln!("FAIL {f}");
        }
        panic!(
            "{} of {} cells fail; streams kept in {}",
            fails.len(),
            pass + split + contested + fails.len(),
            dir.display()
        );
    }
}
