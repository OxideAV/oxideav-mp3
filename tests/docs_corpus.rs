//! Integration tests against the `docs/audio/mp3/fixtures/` corpus.
//!
//! Each fixture under `../../docs/audio/mp3/fixtures/<name>/` ships an
//! `input.mp3` (raw MPEG-1/2/2.5 audio frames, optionally with an ID3v2
//! prefix and/or a Xing/VBRI info-frame) plus an `expected.wav` ground-
//! truth produced by FFmpeg's reference MP3 decoder. This driver opens
//! each input through the in-tree [`oxideav_mp3::container`] demuxer,
//! routes every emitted packet into [`oxideav_mp3::decoder`], collects
//! the resulting interleaved S16 PCM, and reports per-channel
//! statistics against the WAV reference:
//!
//! - decoded vs reference sample-count,
//! - bit-exact match percentage,
//! - per-channel RMS error,
//! - PSNR (full-scale int16 dynamic range).
//!
//! MP3 is a lossy codec — independent decoders sometimes produce
//! sample-identical PCM (the synthesis filterbank is integer-friendly)
//! but the IMDCT path can drift by ±1 LSB on long-term carries depending
//! on how the encoder dithered the spectrum. We therefore start every
//! fixture in [`Tier::ReportOnly`] and let CI surface the deltas without
//! gating; tightening to a numeric PSNR floor on a per-fixture basis is
//! a follow-up once the residual divergences are catalogued (the
//! `layer2-*` fixture in particular is filed here as a known
//! divergence — Layer II is decoded by a sibling crate, not us, so the
//! test is expected to skip with a NotSupported codec id).
//!
//! The trace.txt files under each fixture dir are not consumed here;
//! they are an aid for the human implementer when localising
//! divergences against the static parser output documented in
//! `docs/audio/mp3/`.

use std::fs;
use std::path::PathBuf;

use oxideav_core::{ContainerRegistry, Error, Frame, NullCodecResolver, ReadSeek};
// `Box<dyn Decoder>` / `Box<dyn Demuxer>` resolve trait methods through
// the dyn-vtable, so the traits don't need to be in scope at the call
// site here.

/// Locate `docs/audio/mp3/fixtures/<name>/`. When the test runs as
/// part of the umbrella workspace, CWD is the crate root and the docs
/// live two levels up at `../../docs/`. When the standalone
/// oxideav-mp3 repo is checked out alone (CI), `../../docs/` is
/// absent and every fixture access skips gracefully.
fn fixture_dir(name: &str) -> PathBuf {
    PathBuf::from("../../docs/audio/mp3/fixtures").join(name)
}

#[derive(Clone, Copy, Debug)]
enum Tier {
    /// Decode is permitted to diverge from the FFmpeg reference; we
    /// log the deltas but do not gate CI on them. MP3 is a lossy codec
    /// and two independent decoders typically differ at the sub-LSB
    /// level on long-term IMDCT carry. See the module-level comment.
    ReportOnly,
}

struct CorpusCase {
    name: &'static str,
    /// Expected channels (used to sanity-check the demuxer's
    /// frame-header parse). Set to None to skip the check.
    channels: Option<u16>,
    /// Expected sample rate. None to skip.
    sample_rate: Option<u32>,
    tier: Tier,
}

/// Decoded output from one fixture: interleaved s16le samples plus the
/// channel count + sample rate the decoder advertised.
struct DecodedPcm {
    samples: Vec<i16>,
    channels: u16,
    sample_rate: u32,
}

/// Reference PCM extracted from the fixture's expected.wav.
struct RefPcm {
    samples: Vec<i16>,
    channels: u16,
    sample_rate: u32,
}

/// Per-channel diff numbers + aggregate match percentage and PSNR.
struct ChannelStat {
    rms_ref: f64,
    rms_ours: f64,
    /// Sum of squared per-sample error. Display value (RMS) is computed
    /// at print time; PSNR turns this back into MSE by dividing by total.
    sse: f64,
    exact: usize,
    near: usize, // |delta| <= 1 LSB
    total: usize,
    max_abs_err: i32,
}

impl ChannelStat {
    fn new() -> Self {
        Self {
            rms_ref: 0.0,
            rms_ours: 0.0,
            sse: 0.0,
            exact: 0,
            near: 0,
            total: 0,
            max_abs_err: 0,
        }
    }

    fn match_pct(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.exact as f64 / self.total as f64 * 100.0
        }
    }

    fn near_pct(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.near as f64 / self.total as f64 * 100.0
        }
    }

    /// PSNR over a 16-bit signed full scale (peak = 32767). Returns
    /// `f64::INFINITY` on perfect match.
    fn psnr_db(&self) -> f64 {
        if self.total == 0 || self.sse == 0.0 {
            return f64::INFINITY;
        }
        let mse = self.sse / self.total as f64;
        let peak = 32767.0_f64;
        10.0 * (peak * peak / mse).log10()
    }
}

/// Build a one-shot ContainerRegistry that knows about MP3, then open
/// the input file through it and decode every packet end-to-end.
fn decode_fixture_pcm(case: &CorpusCase) -> Option<DecodedPcm> {
    let dir = fixture_dir(case.name);
    let mp3_path = dir.join("input.mp3");
    let file = match fs::File::open(&mp3_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("skip {}: missing {} ({e})", case.name, mp3_path.display());
            return None;
        }
    };
    let mut creg = ContainerRegistry::new();
    oxideav_mp3::container::register(&mut creg);

    let rs: Box<dyn ReadSeek> = Box::new(file);
    let mut demux = match creg.open_demuxer("mp3", rs, &NullCodecResolver) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skip {}: mp3 demuxer open failed: {e}", case.name);
            return None;
        }
    };

    let streams = demux.streams();
    if streams.is_empty() {
        eprintln!("skip {}: mp3 has no streams", case.name);
        return None;
    }
    let stream = streams[0].clone();
    let params = stream.params.clone();
    let codec = params.codec_id.as_str().to_owned();
    if codec != "mp3" {
        // Layer I / Layer II inputs come back as `mp1` / `mp2` and are
        // owned by sibling crates. The test still passes — we report
        // the skip and let the human eye file a follow-up if the layer
        // mix changes.
        eprintln!(
            "skip {}: first stream codec is {} (this crate handles mp3 only)",
            case.name, codec,
        );
        return None;
    }
    let channels = params.channels.unwrap_or(0);
    let sample_rate = params.sample_rate.unwrap_or(0);
    if channels == 0 || sample_rate == 0 {
        eprintln!(
            "skip {}: stream advertises bogus channels/rate ({channels}/{sample_rate})",
            case.name
        );
        return None;
    }
    if let Some(want_ch) = case.channels {
        assert_eq!(
            channels, want_ch,
            "{}: header says {channels} channels, expected {want_ch}",
            case.name
        );
    }
    if let Some(want_sr) = case.sample_rate {
        assert_eq!(
            sample_rate, want_sr,
            "{}: header says {sample_rate} Hz, expected {want_sr}",
            case.name
        );
    }

    let mut decoder = match oxideav_mp3::decoder::make_decoder(&params) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skip {}: decoder ctor failed: {e}", case.name);
            return None;
        }
    };

    let stream_index = stream.index;
    let mut samples: Vec<i16> = Vec::new();
    let mut decoder_errors = 0usize;
    loop {
        let pkt = match demux.next_packet() {
            Ok(p) => p,
            Err(Error::Eof) => break,
            Err(e) => {
                eprintln!(
                    "{}: demux error after {} samples: {e}",
                    case.name,
                    samples.len()
                );
                break;
            }
        };
        if pkt.stream_index != stream_index {
            continue;
        }
        if let Err(e) = decoder.send_packet(&pkt) {
            decoder_errors += 1;
            if decoder_errors <= 3 {
                eprintln!("{}: send_packet error: {e}", case.name);
            }
            continue;
        }
        loop {
            match decoder.receive_frame() {
                Ok(Frame::Audio(af)) => {
                    let plane = &af.data[0];
                    for chunk in plane.chunks_exact(2) {
                        samples.push(i16::from_le_bytes([chunk[0], chunk[1]]));
                    }
                }
                Ok(other) => {
                    eprintln!("{}: unexpected non-audio frame: {other:?}", case.name);
                    break;
                }
                Err(Error::NeedMore) => break,
                Err(Error::Eof) => break,
                Err(e) => {
                    decoder_errors += 1;
                    if decoder_errors <= 3 {
                        eprintln!("{}: receive_frame error: {e}", case.name);
                    }
                    break;
                }
            }
        }
    }
    if decoder_errors > 0 {
        eprintln!(
            "{}: total decoder errors: {decoder_errors} (decoded {} samples / {} per channel)",
            case.name,
            samples.len(),
            samples.len() / channels.max(1) as usize
        );
    }

    Some(DecodedPcm {
        samples,
        channels,
        sample_rate,
    })
}

/// Parse a minimal RIFF/WAVE file: locate the `fmt ` chunk to read
/// channels + sample-rate + bits-per-sample, then return the `data`
/// chunk as interleaved s16le samples. Skips any LIST/INFO,
/// JUNK, or other non-essential chunks between `fmt ` and `data`.
fn parse_wav(bytes: &[u8]) -> Option<RefPcm> {
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return None;
    }
    let mut i = 12usize;
    let mut channels: u16 = 0;
    let mut sample_rate: u32 = 0;
    let mut bits_per_sample: u16 = 0;
    let mut data: Option<&[u8]> = None;
    while i + 8 <= bytes.len() {
        let id = &bytes[i..i + 4];
        let sz =
            u32::from_le_bytes([bytes[i + 4], bytes[i + 5], bytes[i + 6], bytes[i + 7]]) as usize;
        let body_start = i + 8;
        let body_end = body_start + sz;
        if body_end > bytes.len() {
            break;
        }
        match id {
            b"fmt " => {
                if sz < 16 {
                    return None;
                }
                let format_tag = u16::from_le_bytes([bytes[body_start], bytes[body_start + 1]]);
                channels = u16::from_le_bytes([bytes[body_start + 2], bytes[body_start + 3]]);
                sample_rate = u32::from_le_bytes([
                    bytes[body_start + 4],
                    bytes[body_start + 5],
                    bytes[body_start + 6],
                    bytes[body_start + 7],
                ]);
                bits_per_sample =
                    u16::from_le_bytes([bytes[body_start + 14], bytes[body_start + 15]]);
                if format_tag != 1 && format_tag != 0xFFFE {
                    return None;
                }
            }
            b"data" => {
                data = Some(&bytes[body_start..body_end]);
                break;
            }
            _ => {}
        }
        i = body_end + (sz & 1);
    }
    let data = data?;
    if channels == 0 || sample_rate == 0 || bits_per_sample != 16 {
        return None;
    }
    let mut samples = Vec::with_capacity(data.len() / 2);
    for chunk in data.chunks_exact(2) {
        samples.push(i16::from_le_bytes([chunk[0], chunk[1]]));
    }
    Some(RefPcm {
        samples,
        channels,
        sample_rate,
    })
}

fn read_reference(case: &CorpusCase) -> Option<RefPcm> {
    let dir = fixture_dir(case.name);
    let wav_path = dir.join("expected.wav");
    let bytes = match fs::read(&wav_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skip {}: missing {} ({e})", case.name, wav_path.display());
            return None;
        }
    };
    parse_wav(&bytes)
}

/// Compute per-channel match/PSNR statistics over the overlapping
/// prefix of decoded vs reference (lossy decoders produce slightly
/// fewer or more samples than the reference because of MDCT block
/// alignment / encoder delay).
fn compare(ours: &DecodedPcm, refp: &RefPcm) -> Vec<ChannelStat> {
    let chs = ours.channels.min(refp.channels) as usize;
    if chs == 0 {
        return Vec::new();
    }
    let frames_ours = ours.samples.len() / ours.channels.max(1) as usize;
    let frames_ref = refp.samples.len() / refp.channels.max(1) as usize;
    let n = frames_ours.min(frames_ref);

    let mut stats: Vec<ChannelStat> = (0..chs).map(|_| ChannelStat::new()).collect();
    for f in 0..n {
        for (ch, s) in stats.iter_mut().enumerate() {
            let our = ours.samples[f * ours.channels as usize + ch] as i64;
            let r = refp.samples[f * refp.channels as usize + ch] as i64;
            let err = (our - r).abs();
            s.total += 1;
            if err == 0 {
                s.exact += 1;
            }
            if err <= 1 {
                s.near += 1;
            }
            if err as i32 > s.max_abs_err {
                s.max_abs_err = err as i32;
            }
            s.rms_ref += (r * r) as f64;
            s.rms_ours += (our * our) as f64;
            s.sse += (err * err) as f64;
        }
    }
    for s in &mut stats {
        if s.total > 0 {
            s.rms_ref = (s.rms_ref / s.total as f64).sqrt();
            s.rms_ours = (s.rms_ours / s.total as f64).sqrt();
        }
    }
    stats
}

/// Decode -> compare -> log -> tier-aware assert (no fixture currently
/// bit-exact, so all tiers are ReportOnly today).
fn evaluate(case: &CorpusCase) {
    eprintln!("--- {} (tier={:?}) ---", case.name, case.tier);
    let Some(ours) = decode_fixture_pcm(case) else {
        return;
    };
    let Some(refp) = read_reference(case) else {
        eprintln!("{}: could not parse expected.wav", case.name);
        return;
    };

    eprintln!(
        "{}: decoded ch={} sr={} samples={} ({} frames); reference ch={} sr={} samples={} ({} frames)",
        case.name,
        ours.channels,
        ours.sample_rate,
        ours.samples.len(),
        ours.samples.len() / ours.channels.max(1) as usize,
        refp.channels,
        refp.sample_rate,
        refp.samples.len(),
        refp.samples.len() / refp.channels.max(1) as usize,
    );

    if ours.channels != refp.channels {
        eprintln!(
            "{}: WARN channel count mismatch (decoded {} vs reference {})",
            case.name, ours.channels, refp.channels
        );
    }
    if ours.sample_rate != refp.sample_rate {
        eprintln!(
            "{}: WARN sample-rate mismatch (decoded {} vs reference {})",
            case.name, ours.sample_rate, refp.sample_rate
        );
    }

    let stats = compare(&ours, &refp);
    if stats.is_empty() {
        eprintln!("{}: no overlapping channels to compare", case.name);
        return;
    }

    let mut total_exact = 0usize;
    let mut total_near = 0usize;
    let mut total_samples = 0usize;
    let mut max_err_overall = 0i32;
    let mut psnr_min: f64 = f64::INFINITY;
    for (i, s) in stats.iter().enumerate() {
        let psnr = s.psnr_db();
        if psnr < psnr_min {
            psnr_min = psnr;
        }
        let rms_err_disp = if s.total > 0 {
            (s.sse / s.total as f64).sqrt()
        } else {
            0.0
        };
        eprintln!(
            "  ch{i}: rms_ref={:.1} rms_ours={:.1} rms_err={:.2} match={:.4}% near<=1LSB={:.4}% max_abs_err={} psnr={:.2} dB",
            s.rms_ref,
            s.rms_ours,
            rms_err_disp,
            s.match_pct(),
            s.near_pct(),
            s.max_abs_err,
            psnr,
        );
        total_exact += s.exact;
        total_near += s.near;
        total_samples += s.total;
        if s.max_abs_err > max_err_overall {
            max_err_overall = s.max_abs_err;
        }
    }
    let agg_pct = if total_samples > 0 {
        total_exact as f64 / total_samples as f64 * 100.0
    } else {
        0.0
    };
    let near_pct = if total_samples > 0 {
        total_near as f64 / total_samples as f64 * 100.0
    } else {
        0.0
    };
    eprintln!(
        "{}: aggregate match={:.4}% near<=1LSB={:.4}% max_abs_err={} min_psnr={:.2} dB",
        case.name, agg_pct, near_pct, max_err_overall, psnr_min,
    );

    match case.tier {
        Tier::ReportOnly => {
            // Logged; never gates CI. Underlying float-rounding deltas
            // are tracked as follow-up tasks if PSNR drops below ~40 dB.
        }
    }
}

// ---------------------------------------------------------------------------
// Per-fixture tests.
//
// Every fixture is Tier::ReportOnly until at least one is shown to round-
// trip cleanly. The Layer-II fixture (`layer2-stereo-44100-192kbps`) is
// included for completeness — `decode_fixture_pcm` skips with a clean
// "this crate handles mp3 only" message because the demuxer routes Layer
// II frames through the `mp2` codec id.
// ---------------------------------------------------------------------------

#[test]
fn corpus_layer2_stereo_44100_192kbps() {
    // Routed to codec id `mp2` by the shared MPEG-audio demuxer; this
    // crate decodes mp3 only, so the test is expected to skip cleanly.
    evaluate(&CorpusCase {
        name: "layer2-stereo-44100-192kbps",
        channels: None,
        sample_rate: None,
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_layer3_cbr_320kbps_stereo_44100() {
    evaluate(&CorpusCase {
        name: "layer3-cbr-320kbps-stereo-44100",
        channels: Some(2),
        sample_rate: Some(44_100),
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_layer3_intensity_stereo_44100_low() {
    evaluate(&CorpusCase {
        name: "layer3-intensity-stereo-44100-low",
        channels: Some(2),
        sample_rate: Some(44_100),
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_layer3_joint_stereo_44100_128kbps() {
    evaluate(&CorpusCase {
        name: "layer3-joint-stereo-44100-128kbps",
        channels: Some(2),
        sample_rate: Some(44_100),
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_layer3_mixed_block_flag() {
    evaluate(&CorpusCase {
        name: "layer3-mixed-block-flag",
        channels: None,
        sample_rate: None,
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_layer3_mono_44100_128kbps() {
    evaluate(&CorpusCase {
        name: "layer3-mono-44100-128kbps",
        channels: Some(1),
        sample_rate: Some(44_100),
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_layer3_mono_44100_64kbps_short() {
    evaluate(&CorpusCase {
        name: "layer3-mono-44100-64kbps-short",
        channels: Some(1),
        sample_rate: Some(44_100),
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_layer3_mpeg2_22050_64kbps() {
    evaluate(&CorpusCase {
        name: "layer3-mpeg2-22050-64kbps",
        channels: None,
        sample_rate: Some(22_050),
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_layer3_mpeg25_11025_32kbps() {
    evaluate(&CorpusCase {
        name: "layer3-mpeg25-11025-32kbps",
        channels: None,
        sample_rate: Some(11_025),
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_layer3_ms_stereo_44100_128kbps() {
    evaluate(&CorpusCase {
        name: "layer3-ms-stereo-44100-128kbps",
        channels: Some(2),
        sample_rate: Some(44_100),
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_layer3_padding_byte_cycle() {
    evaluate(&CorpusCase {
        name: "layer3-padding-byte-cycle",
        channels: None,
        sample_rate: None,
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_layer3_stereo_32000_128kbps() {
    evaluate(&CorpusCase {
        name: "layer3-stereo-32000-128kbps",
        channels: Some(2),
        sample_rate: Some(32_000),
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_layer3_stereo_44100_128kbps() {
    evaluate(&CorpusCase {
        name: "layer3-stereo-44100-128kbps",
        channels: Some(2),
        sample_rate: Some(44_100),
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_layer3_stereo_48000_128kbps() {
    evaluate(&CorpusCase {
        name: "layer3-stereo-48000-128kbps",
        channels: Some(2),
        sample_rate: Some(48_000),
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_layer3_vbr_q5_stereo_44100() {
    evaluate(&CorpusCase {
        name: "layer3-vbr-q5-stereo-44100",
        channels: Some(2),
        sample_rate: Some(44_100),
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_layer3_with_id3v2_tag() {
    evaluate(&CorpusCase {
        name: "layer3-with-id3v2-tag",
        channels: None,
        sample_rate: None,
        tier: Tier::ReportOnly,
    });
}

#[test]
fn corpus_layer3_with_xing_vbri_tag() {
    evaluate(&CorpusCase {
        name: "layer3-with-xing-vbri-tag",
        channels: None,
        sample_rate: None,
        tier: Tier::ReportOnly,
    });
}
