//! Corpus-wide **differential decode sweep**: every Layer III fixture
//! under `docs/audio/mp3/fixtures/` is decoded through the *production*
//! chain — `Mp3Demuxer` → registered `oxideav_core::Decoder` — and the
//! PCM is compared against the fixture's `expected.wav` (the black-box
//! reference decode staged with the corpus).
//!
//! `tests/lsf_reference_pcm.rs` and `tests/mpeg25_reference_pcm.rs`
//! prove reference-tracking for two single fixtures (22.05 kHz LSF and
//! 11.025 kHz MPEG-2.5) through the *direct* decode primitives. This
//! test widens the net to the whole corpus — MPEG-1 mono/stereo at
//! 32 / 44.1 / 48 kHz, CBR 320, VBR, joint stereo (MS and intensity),
//! short/mixed blocks, ID3v2 / Xing / VBRI frontmatter, the padding
//! byte cycle, MPEG-2 LSF, and MPEG-2.5 — and runs the demuxer +
//! trait-decoder path a real player would use.
//!
//! For each fixture the two decodes are aligned by peak normalized
//! cross-correlation (the two toolchains trim codec delay differently)
//! and compared over the steady-state interior:
//!
//! * **peak NCC** must be ≈ 1 (same waveform, not a chance alignment);
//! * **normalized RMS error** must sit in the float-rounding regime for
//!   a correct band layout / requantizer / stereo path. A wrong
//!   scalefactor band table, a broken stereo mode, or a reservoir bug
//!   shows up as nrmse orders of magnitude above the bound.
//!
//! The Layer II fixture is excluded by design: this crate decodes
//! Layer III only (the demuxer walks Layer II framing, but there is no
//! Layer II decoder to differ against).
//!
//! Skips (with a log line) when the workspace `docs/` tree is absent
//! (standalone-crate CI checkout), matching `tests/docs_corpus.rs`.

use std::fs;
use std::path::PathBuf;

use oxideav_core::{
    CodecId, CodecParameters, Decoder, Demuxer, Error, Frame, RuntimeContext, SampleFormat,
};
use oxideav_mp3::Mp3Demuxer;

/// Per-fixture expectations: name, channel count, sample rate.
///
/// Every fixture must meet the same steady-state normalized-RMS-error
/// bound [`NRMSE_BOUND`]: measured 2026-07, all 16 fixtures track
/// their reference at nrmse ≤ 1.6e-5 (the float-rounding regime — the
/// two decoders agree to within ±1 LSB of the 16-bit output on every
/// path: independent/MS/intensity stereo, short and mixed blocks, CBR
/// / VBR / padding cycle, MPEG-1 / MPEG-2 LSF / MPEG-2.5), all at the
/// canonical 1105-sample codec-delay lag with alignment NCC = 1.0000.
/// The bound sits one order of magnitude above the worst measurement;
/// a wrong scalefactor-band table, a broken stereo mode, or a
/// reservoir bug shows up orders of magnitude above it.
const FIXTURES: &[(&str, u16, u32)] = &[
    ("layer3-cbr-320kbps-stereo-44100", 2, 44_100),
    ("layer3-intensity-stereo-44100-low", 2, 44_100),
    ("layer3-joint-stereo-44100-128kbps", 2, 44_100),
    ("layer3-mixed-block-flag", 2, 44_100),
    ("layer3-mono-44100-128kbps", 1, 44_100),
    ("layer3-mono-44100-64kbps-short", 1, 44_100),
    ("layer3-mpeg2-22050-64kbps", 2, 22_050),
    ("layer3-mpeg25-11025-32kbps", 1, 11_025),
    ("layer3-ms-stereo-44100-128kbps", 2, 44_100),
    ("layer3-padding-byte-cycle", 1, 44_100),
    ("layer3-stereo-32000-128kbps", 2, 32_000),
    ("layer3-stereo-44100-128kbps", 2, 44_100),
    ("layer3-stereo-48000-128kbps", 2, 48_000),
    ("layer3-vbr-q5-stereo-44100", 2, 44_100),
    ("layer3-with-id3v2-tag", 2, 44_100),
    ("layer3-with-xing-vbri-tag", 2, 44_100),
];

/// Steady-state normalized RMS error bound — see [`FIXTURES`].
const NRMSE_BOUND: f64 = 2e-4;

fn corpus_root() -> Option<PathBuf> {
    let p = PathBuf::from("../../docs/audio/mp3/fixtures");
    if p.is_dir() {
        Some(p)
    } else {
        None
    }
}

/// Decode `input.mp3` through the production path: `Mp3Demuxer` (ID3v2
/// skip, Xing/Info carrier consumption, frame walk) feeding the
/// registered `Decoder`. Returns per-channel f32 PCM planes in [-1, 1]
/// (the decoder's `AudioFrame` convention is planar: `data[0]` = L,
/// `data[1]` = R).
fn production_decode(path: &std::path::Path, channels: u16, sample_rate: u32) -> Vec<Vec<f32>> {
    let file = fs::File::open(path).expect("open input.mp3");
    let mut demux = Mp3Demuxer::open(Box::new(file)).expect("demuxer open");

    let mut ctx = RuntimeContext::default();
    oxideav_mp3::register(&mut ctx);
    let mut params = CodecParameters::audio(CodecId::new("mp3"));
    params.sample_rate = Some(sample_rate);
    params.channels = Some(channels);
    params.sample_format = Some(SampleFormat::S16);
    let mut dec: Box<dyn Decoder> = ctx
        .codecs
        .first_decoder(&params)
        .expect("decoder factory present after register()");

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
                    assert_eq!(
                        a.data.len(),
                        channels as usize,
                        "decoder plane count != channel count"
                    );
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

/// Minimal RIFF/WAVE reader for the 16-bit LE PCM `expected.wav`.
/// Returns `(channels, sample_rate, interleaved f32 in [-1, 1])`.
fn read_wav_s16(path: &std::path::Path) -> (u16, u32, Vec<f32>) {
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
                assert_eq!(fmt_tag, 1, "expected.wav must be integer PCM");
                channels = u16::from_le_bytes(body[2..4].try_into().unwrap());
                sample_rate = u32::from_le_bytes(body[4..8].try_into().unwrap());
                let bits = u16::from_le_bytes(body[14..16].try_into().unwrap());
                assert_eq!(bits, 16, "expected.wav must be 16-bit");
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

/// Best-correlation lag of `b` relative to `a` (searched on the given
/// channel plane) plus the normalized RMS error over the steady-state
/// region, summed across all channel planes at that single shared lag.
/// Mirrors `align_and_error` in `tests/mpeg25_reference_pcm.rs`,
/// generalized to multi-channel.
fn align_and_error(
    a: &[Vec<f32>],
    b: &[Vec<f32>],
    max_lag: isize,
    win: usize,
) -> (isize, f64, f64) {
    assert_eq!(a.len(), b.len(), "channel count mismatch");
    let anchor = 3 * 576usize;
    // Pick the alignment channel: the plane with the most energy in the
    // correlation window (an intensity-coded side channel can be almost
    // silent and would produce a noisy correlation peak).
    let pick = (0..a.len())
        .max_by(|&x, &y| {
            let e = |ch: usize| -> f64 {
                a[ch]
                    .iter()
                    .skip(anchor)
                    .take(win)
                    .map(|&v| f64::from(v) * f64::from(v))
                    .sum()
            };
            e(x).partial_cmp(&e(y)).unwrap()
        })
        .unwrap();
    let (pa_, pb_) = (&a[pick], &b[pick]);
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
fn every_layer3_fixture_tracks_its_reference_pcm() {
    let Some(root) = corpus_root() else {
        eprintln!("skip: docs corpus absent (standalone-crate CI checkout)");
        return;
    };
    let mut failures: Vec<String> = Vec::new();
    for &(name, channels, rate) in FIXTURES {
        let dir = root.join(name);
        assert!(
            dir.join("input.mp3").exists() && dir.join("expected.wav").exists(),
            "{name}: fixture files missing — corpus layout changed?"
        );
        let b = production_decode(&dir.join("input.mp3"), channels, rate);
        let (ref_ch, ref_rate, ref_pcm) = read_wav_s16(&dir.join("expected.wav"));
        assert_eq!(ref_ch, channels, "{name}: reference channel count");
        assert_eq!(ref_rate, rate, "{name}: reference sample rate");
        assert!(
            !b[0].is_empty(),
            "{name}: production decode produced no PCM"
        );

        let a = deinterleave(&ref_pcm, channels as usize);
        let max_lag = 6 * 576isize;
        let win = 8 * 576usize;
        let (lag, nrmse, ncc) = align_and_error(&a, &b, max_lag, win);
        eprintln!(
            "{name:<40} ref={:>6} ours={:>6} lag={lag:>5} ncc={ncc:.5} nrmse={nrmse:.3e}",
            a[0].len(),
            b[0].len(),
        );
        if ncc <= 0.999 {
            failures.push(format!("{name}: alignment peak NCC {ncc:.5} <= 0.999"));
        }
        if nrmse >= NRMSE_BOUND {
            failures.push(format!(
                "{name}: nrmse {nrmse:.3e} >= bound {NRMSE_BOUND:.0e}"
            ));
        }
        if lag != 1105 {
            failures.push(format!(
                "{name}: alignment lag {lag} != canonical 1105-sample codec delay"
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "corpus differential decode failures:\n{}",
        failures.join("\n")
    );
}
