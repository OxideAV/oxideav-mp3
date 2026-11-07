//! The demuxer on streams whose frame headers change mid-stream.
//!
//! Fixtures (`tests/fixtures/splice/gen.sh`): short single-shape MP3
//! files encoded with the black-box `ffmpeg` CLI, one per
//! sampling frequency (MPEG-1 48 / 44.1 / 32 kHz, MPEG-2 LSF 24 / 22.05
//! / 16 kHz, MPEG-2.5 12 / 11.025 / 8 kHz) plus channel-mode, bitrate
//! and VBR variants and three fully-tagged files (ID3v2 + Info / Xing
//! frame + ID3v1 trailer). The tests byte-concatenate them the way
//! `cat a.mp3 b.mp3` does and check, against oracles that do not go
//! through the demuxer:
//!
//! * the frame count of each segment — what ffprobe counted
//!   (`manifest.txt`) and what an independent header walk over the
//!   segment file alone finds;
//! * the packet bytes — for tag-free segments the packets concatenate
//!   back to the input byte for byte, so every frame came out once and
//!   nothing else did;
//! * the timestamps — continuous (`pts + duration` = next `pts`),
//!   strictly increasing, in the first segment's `1 / f` time base, and
//!   summing to the exact running time
//!   `Σ frames · samples_per_frame / f_segment`;
//! * the PCM — the decode of a spliced stream through demuxer + decoder
//!   equals the standalone decodes of its segments, and the decoder
//!   raises exactly one `Frame::StreamParams` per output-shape change.

use std::collections::HashMap;
use std::io::Cursor;

use oxideav_core::{CodecId, CodecParameters, Decoder, Demuxer, Error, Frame, Packet};
use oxideav_mp3::codec_decoder::make_decoder;
use oxideav_mp3::frame::FrameWalker;
use oxideav_mp3::{Mp3Demuxer, parse_header};

const FIX: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/splice");

/// The nine sampling-frequency segments, stereo.
const RATES: [&str; 9] = [
    "raw_48000.mp3",
    "raw_44100.mp3",
    "raw_32000.mp3",
    "raw_24000.mp3",
    "raw_22050.mp3",
    "raw_16000.mp3",
    "raw_12000.mp3",
    "raw_11025.mp3",
    "raw_8000.mp3",
];

fn file(name: &str) -> Vec<u8> {
    std::fs::read(format!("{FIX}/{name}")).expect("fixture readable")
}

/// `name -> (ffprobe packet count, sampling frequency, channels)`.
fn manifest() -> HashMap<String, (usize, u32, u16)> {
    let text = std::fs::read_to_string(format!("{FIX}/manifest.txt")).expect("manifest");
    text.lines()
        .map(|l| {
            let f: Vec<&str> = l.split_whitespace().collect();
            // ffprobe prints sample_rate, channels, nb_read_packets.
            (
                f[0].to_string(),
                (
                    f[3].parse().unwrap(),
                    f[1].parse().unwrap(),
                    f[2].parse().unwrap(),
                ),
            )
        })
        .collect()
}

/// Independent header walk over one tag-free segment file: frame
/// count, samples per frame and sampling frequency.
fn header_walk(bytes: &[u8]) -> (usize, u32, u32) {
    let frames: Vec<_> = FrameWalker::new(bytes).collect();
    let h = frames[0].header;
    // The walker consumed the whole file: the segment is frames only.
    let covered: usize = frames.iter().map(|f| f.data.len()).sum();
    assert_eq!(covered, bytes.len(), "segment is frames only");
    (frames.len(), h.samples_per_frame(), h.sample_rate_hz)
}

fn open(bytes: Vec<u8>) -> Mp3Demuxer {
    Mp3Demuxer::open(Box::new(Cursor::new(bytes))).expect("open")
}

fn packets(d: &mut Mp3Demuxer) -> Vec<Packet> {
    let mut out = Vec::new();
    loop {
        match d.next_packet() {
            Ok(p) => out.push(p),
            Err(Error::Eof) => break,
            Err(e) => panic!("next_packet: {e}"),
        }
    }
    assert!(matches!(d.next_packet(), Err(Error::Eof)));
    out
}

/// Timestamps are continuous, strictly increasing from zero, and end
/// at the declared duration.
fn check_timeline(d: &Mp3Demuxer, pkts: &[Packet], what: &str) {
    let mut expect = 0i64;
    for (i, p) in pkts.iter().enumerate() {
        let pts = p.pts.unwrap();
        let dur = p.duration.unwrap();
        assert_eq!(pts, expect, "{what}: packet {i} pts continuous");
        assert_eq!(p.dts, Some(pts));
        assert!(dur > 0, "{what}: packet {i} duration positive");
        assert_eq!(p.time_base, d.streams()[0].time_base);
        expect = pts + dur;
    }
    assert_eq!(
        d.streams()[0].duration,
        Some(expect),
        "{what}: duration = Σ packet durations"
    );
}

/// Exact running time of `segments` (`(frames, samples_per_frame,
/// rate)`), floored to `1 / base_rate` units.
fn exact_duration(segments: &[(usize, u32, u32)], base_rate: u32) -> i64 {
    // Σ frames·spf/rate as a rational over the common denominator.
    let l: u128 = 14_112_000;
    let ticks: u128 = segments
        .iter()
        .map(|&(n, spf, r)| n as u128 * spf as u128 * (l / r as u128))
        .sum();
    (ticks * base_rate as u128 / l) as i64
}

/// One `receive_frame` result.
enum Out {
    Event(u32, u16),
    Audio(Option<i64>, Vec<Vec<i16>>),
}

fn decode(pkts: &[Packet]) -> Vec<Out> {
    let mut dec: Box<dyn Decoder> =
        make_decoder(&CodecParameters::audio(CodecId::new("mp3"))).expect("decoder");
    let mut out = Vec::new();
    let drain = |dec: &mut dyn Decoder, out: &mut Vec<Out>| {
        loop {
            match dec.receive_frame() {
                Ok(Frame::StreamParams(p)) => {
                    out.push(Out::Event(p.sample_rate.unwrap(), p.channels.unwrap()))
                }
                Ok(Frame::Audio(af)) => out.push(Out::Audio(
                    af.pts,
                    af.data
                        .iter()
                        .map(|pl| {
                            pl.as_chunks::<2>()
                                .0
                                .iter()
                                .map(|c| i16::from_le_bytes(*c))
                                .collect()
                        })
                        .collect(),
                )),
                Ok(other) => panic!("unexpected {other:?}"),
                Err(Error::NeedMore) | Err(Error::Eof) => break,
                Err(e) => panic!("decode: {e}"),
            }
        }
    };
    for p in pkts {
        dec.send_packet(p).expect("send_packet");
        drain(dec.as_mut(), &mut out);
    }
    dec.flush().unwrap();
    drain(dec.as_mut(), &mut out);
    out
}

/// PCM of a decode, one entry per audio frame: (pts, planes).
fn audio(out: &[Out]) -> Vec<(Option<i64>, Vec<Vec<i16>>)> {
    out.iter()
        .filter_map(|o| match o {
            Out::Audio(pts, p) => Some((*pts, p.clone())),
            Out::Event(..) => None,
        })
        .collect()
}

fn events(out: &[Out]) -> Vec<(u32, u16)> {
    out.iter()
        .filter_map(|o| match o {
            Out::Event(r, c) => Some((*r, *c)),
            Out::Audio(..) => None,
        })
        .collect()
}

/// Standalone decode of one segment through demuxer + decoder.
fn standalone(bytes: &[u8]) -> Vec<Vec<Vec<i16>>> {
    let mut d = open(bytes.to_vec());
    let out = decode(&packets(&mut d));
    assert!(events(&out).is_empty(), "a constant segment has no event");
    audio(&out).into_iter().map(|(_, p)| p).collect()
}

/// `(frames, samples_per_frame, sampling frequency)` of one segment.
type Walk = (usize, u32, u32);

/// Concatenate segment files and check the framing of the spliced
/// stream: frame counts, bytes, timeline, duration, switch count.
/// Returns the packets and the segments' bytes and header walks.
fn check_framing(names: &[&str]) -> (Vec<Packet>, Vec<Vec<u8>>, Vec<Walk>) {
    let man = manifest();
    let segs: Vec<Vec<u8>> = names.iter().map(|n| file(n)).collect();
    let walks: Vec<(usize, u32, u32)> = segs.iter().map(|b| header_walk(b)).collect();
    for (i, n) in names.iter().enumerate() {
        let (count, rate, _) = man[*n];
        assert_eq!(walks[i].0, count, "{n}: header walk vs ffprobe count");
        assert_eq!(walks[i].2, rate, "{n}: rate");
    }
    let what = names.join("+");
    let joined: Vec<u8> = segs.concat();
    let mut d = open(joined.clone());
    let pkts = packets(&mut d);

    // Every frame exactly once, nothing else.
    let total: usize = walks.iter().map(|w| w.0).sum();
    assert_eq!(pkts.len(), total, "{what}: packet count");
    assert_eq!(d.audio_frame_count(), total as u64);
    let rebuilt: Vec<u8> = pkts.iter().flat_map(|p| p.data.iter().copied()).collect();
    assert!(
        rebuilt == joined,
        "{what}: packets re-concatenate to the input"
    );

    // Timeline in the first segment's time base, exact overall.
    let base = walks[0].2;
    assert_eq!(d.streams()[0].time_base.0.den, i64::from(base), "{what}");
    check_timeline(&d, &pkts, &what);
    assert_eq!(
        d.streams()[0].duration,
        Some(exact_duration(&walks, base)),
        "{what}: exact duration"
    );
    // A framing switch: version / layer / sampling frequency change.
    let keys: Vec<_> = segs
        .iter()
        .map(|b| {
            let h = parse_header(b).unwrap();
            (h.version, h.layer, h.sample_rate_hz)
        })
        .collect();
    let switches = keys.windows(2).filter(|w| w[0] != w[1]).count();
    assert_eq!(d.framing_switch_count(), switches as u64, "{what}");
    (pkts, segs, walks)
}

/// [`check_framing`], then decode: one `StreamParams` event per
/// output-shape change, and each segment's PCM equal to its standalone
/// decode — from its first frame after a shape change (the decoder
/// starts a fresh segment there), from its second frame on at a
/// boundary the decoder does not see as a change (its filterbank
/// memory carries across).
fn check_splice(names: &[&str]) {
    let man = manifest();
    let what = names.join("+");
    let (pkts, segs, walks) = check_framing(names);
    let total = pkts.len();

    // Decode: one event per output-shape change, PCM per segment.
    let shapes: Vec<(u32, u16)> = names.iter().map(|n| (man[*n].1, man[*n].2)).collect();
    let out = decode(&pkts);
    let want_events: Vec<(u32, u16)> = shapes
        .windows(2)
        .filter(|w| w[0] != w[1])
        .map(|w| w[1])
        .collect();
    assert_eq!(events(&out), want_events, "{what}: StreamParams events");
    let got = audio(&out);
    assert_eq!(got.len(), total, "{what}: one audio frame per packet");
    for (i, (pts, _)) in got.iter().enumerate() {
        assert_eq!(
            *pts, pkts[i].pts,
            "{what}: frame {i} carries its packet pts"
        );
    }
    let mut at = 0;
    for (i, seg) in segs.iter().enumerate() {
        let alone = standalone(seg);
        assert_eq!(alone.len(), walks[i].0);
        let fresh = i == 0 || shapes[i - 1] != shapes[i];
        let from = if fresh { 0 } else { 1 };
        for (k, frame) in alone.iter().enumerate().skip(from) {
            assert!(
                got[at + k].1 == *frame,
                "{what}: segment {i} ({}) frame {k} PCM differs from its standalone decode",
                names[i]
            );
        }
        at += alone.len();
    }
}

/// Every ordered pair of sampling frequencies — MPEG-1 ↔ MPEG-2 LSF ↔
/// MPEG-2.5 and within each version — splices cleanly.
#[test]
fn every_sample_rate_pair() {
    for a in RATES {
        for b in RATES {
            if a != b {
                check_splice(&[a, b]);
            }
        }
    }
}

/// One stream visiting all nine sampling frequencies, twice, with
/// mono segments in between: crosses the seek-index stride several
/// times.
#[test]
fn chain_through_every_rate() {
    let mut names: Vec<&str> = Vec::new();
    for _ in 0..2 {
        names.extend_from_slice(&RATES);
        names.push("raw_8000_mono.mp3");
        names.push("raw_22050_mono.mp3");
        names.push("raw_44100_mono.mp3");
    }
    check_splice(&names);
}

/// Channel mode (mono / independent stereo / joint stereo) and bitrate
/// (CBR 64 / 128 kbit/s, VBR) switches at one sampling frequency: no
/// framing switch, every frame out.
#[test]
fn channel_mode_and_bitrate_switches() {
    check_splice(&[
        "raw_44100.mp3",
        "raw_44100_mono.mp3",
        "raw_44100_lr.mp3",
        "raw_44100_128k.mp3",
        "raw_44100_vbr.mp3",
        "raw_44100_mono.mp3",
        "raw_44100.mp3",
    ]);
    check_splice(&["raw_22050.mp3", "raw_22050_mono.mp3", "raw_8000_mono.mp3"]);
    check_splice(&["raw_44100_vbr.mp3", "raw_32000.mp3", "raw_44100_vbr.mp3"]);
}

/// Layer III ↔ Layer II switches (MPEG-1 and MPEG-2 LSF Layer II,
/// CRC-protected), including one at an unchanged sampling frequency:
/// framed completely (Layer II is not decoded by this crate).
#[test]
fn layer_switches() {
    check_framing(&[
        "raw_48000.mp3",
        "raw_l2_48000.mp2",
        "raw_8000.mp3",
        "raw_l2_22050.mp2",
        "raw_22050.mp3",
        "raw_l2_48000.mp2",
    ]);
    for a in RATES {
        check_framing(&[a, "raw_l2_48000.mp2", a]);
        check_framing(&["raw_l2_22050.mp2", a, "raw_l2_22050.mp2"]);
    }
}

/// The round-482 failure: 44.1 kHz followed by 32 kHz or 22.05 kHz.
#[test]
fn rate_drop_after_44100_keeps_every_frame() {
    for b in ["raw_32000.mp3", "raw_22050.mp3"] {
        let joined = [file("raw_44100.mp3"), file(b)].concat();
        let mut d = open(joined);
        let pkts = packets(&mut d);
        let man = manifest();
        assert_eq!(pkts.len(), man["raw_44100.mp3"].0 + man[b].0, "{b}");
        let out = decode(&pkts);
        assert_eq!(events(&out), vec![(man[b].1, 2)], "{b}");
    }
}

/// Rewrite every frame's `bitrate_index` to 0 (free format). A CBR
/// stream already has a constant unpadded frame body, so the result is
/// a valid free-format stream decoding to the same PCM.
fn to_free_format(bytes: &[u8]) -> Vec<u8> {
    let mut out = bytes.to_vec();
    let mut pos = 0;
    for f in FrameWalker::new(bytes) {
        assert_eq!(f.offset, pos);
        out[pos + 2] &= 0x0F;
        pos += f.data.len();
    }
    out
}

/// Free format: a free-format segment followed by another at the same
/// sampling frequency but a different fixed bitrate (the measured slot
/// count has to be re-measured at the boundary), then a rate switch, a
/// table-bitrate segment, and free format again.
#[test]
fn free_format_segments() {
    let man = manifest();
    let parts: Vec<(&str, bool)> = vec![
        ("raw_44100.mp3", true),
        ("raw_44100_128k.mp3", true),
        ("raw_22050.mp3", true),
        ("raw_48000.mp3", false),
        ("raw_8000.mp3", true),
        ("raw_44100.mp3", true),
    ];
    let segs: Vec<Vec<u8>> = parts
        .iter()
        .map(|(n, free)| {
            let b = file(n);
            if *free { to_free_format(&b) } else { b }
        })
        .collect();
    let joined = segs.concat();
    let mut d = open(joined.clone());
    assert!(parse_header(&joined[..4]).unwrap().is_free_format());
    let pkts = packets(&mut d);
    let total: usize = parts.iter().map(|(n, _)| man[*n].0).sum();
    assert_eq!(pkts.len(), total, "free-format splice packet count");
    let rebuilt: Vec<u8> = pkts.iter().flat_map(|p| p.data.iter().copied()).collect();
    assert!(rebuilt == joined, "packets re-concatenate to the input");
    check_timeline(&d, &pkts, "free format");

    // Same PCM as the table-bitrate original splice.
    let table: Vec<u8> = parts.iter().flat_map(|(n, _)| file(n)).collect();
    let mut dt = open(table);
    let want = audio(&decode(&packets(&mut dt)));
    let got_out = decode(&pkts);
    let got = audio(&got_out);
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(g.1 == w.1, "free-format frame {i} PCM differs");
    }
    assert_eq!(events(&got_out).len(), 4, "44.1→22.05→48→8→44.1 kHz");
}

/// Fully tagged files back to back: the second and third files' ID3v2
/// tags, Info / Xing frames and ID3v1 trailers sit mid-stream and are
/// stepped over; the packets are exactly each file's own packets.
#[test]
fn tagged_files_concatenated() {
    let man = manifest();
    let names = [
        "tagged_44100.mp3",
        "tagged_32000_vbr.mp3",
        "tagged_22050_mono.mp3",
        "tagged_44100.mp3",
    ];
    let mut want: Vec<Vec<u8>> = Vec::new();
    for n in names {
        let mut d = open(file(n));
        assert!(d.xing().is_some(), "{n} carries an Info/Xing frame");
        assert!(d.tags().id3v2_present && d.tags().id3v1_present, "{n}");
        let p = packets(&mut d);
        assert_eq!(p.len(), man[n].0, "{n}: demuxer vs ffprobe count");
        want.extend(p.into_iter().map(|p| p.data));
    }
    let joined: Vec<u8> = names.iter().flat_map(|n| file(n)).collect();
    let mut d = open(joined);
    // The first file's Info frame describes the first file only; the
    // duration comes from the header walk instead.
    let first = open(file(names[0]));
    assert_eq!(d.xing().unwrap().frames, first.xing().unwrap().frames);
    let pkts = packets(&mut d);
    let got: Vec<Vec<u8>> = pkts.iter().map(|p| p.data.clone()).collect();
    assert_eq!(got.len(), want.len(), "tagged splice packet count");
    assert!(got == want, "tagged splice packets are each file's packets");
    check_timeline(&d, &pkts, "tagged");
    let out = decode(&pkts);
    assert_eq!(
        events(&out),
        vec![(32_000, 2), (22_050, 1), (44_100, 2)],
        "tagged splice events"
    );
    assert!(d.duration_micros().unwrap() > first.duration_micros().unwrap() * 3);
}

/// Deterministic junk dense in syncword-like bytes and well-formed
/// but mis-spaced frame headers.
fn junk(len: usize, seed: u32) -> Vec<u8> {
    let mut s = seed;
    let mut out = Vec::with_capacity(len);
    // A few real headers from the fixtures (44.1 kHz, 32 kHz, 8 kHz).
    let headers: Vec<[u8; 4]> = RATES
        .iter()
        .map(|n| {
            let b = file(n);
            [b[0], b[1], b[2], b[3]]
        })
        .collect();
    while out.len() < len {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        match s >> 29 {
            0 | 1 => out.push(0xFF),
            2 => out.extend_from_slice(&headers[(s >> 8) as usize % headers.len()]),
            _ => out.push((s >> 16) as u8),
        }
    }
    out.truncate(len);
    out
}

/// Junk between segments (a damaged splice): the resync after the
/// switch finds the real next frame and no false sync inside the junk.
#[test]
fn resync_through_junk_between_switching_segments() {
    for (k, (a, b)) in [
        ("raw_44100.mp3", "raw_8000.mp3"),
        ("raw_8000.mp3", "raw_48000.mp3"),
        ("raw_22050.mp3", "raw_32000.mp3"),
        ("raw_44100_vbr.mp3", "raw_11025.mp3"),
    ]
    .into_iter()
    .enumerate()
    {
        for (j, len) in [1usize, 3, 37, 400, 2000].into_iter().enumerate() {
            let (fa, fb) = (file(a), file(b));
            let mut joined = fa.clone();
            joined.extend(junk(len, (k * 10 + j) as u32 + 1));
            joined.extend_from_slice(&fb);
            let mut d = open(joined);
            let pkts = packets(&mut d);
            let want: Vec<u8> = [fa, fb].concat();
            let got: Vec<u8> = pkts.iter().flat_map(|p| p.data.iter().copied()).collect();
            assert!(
                got == want,
                "{a} + {len} junk bytes + {b}: exactly the real frames"
            );
            check_timeline(&d, &pkts, "junk");
        }
    }
}

/// Seeking into a spliced stream lands on the frame containing the
/// target (at or before it), returns the PTS that frame is then emitted
/// with, and continues exactly like the linear walk.
#[test]
fn seek_across_switches() {
    let names: Vec<&str> = RATES
        .iter()
        .copied()
        .chain([
            "raw_44100_mono.mp3",
            "raw_44100_vbr.mp3",
            "raw_8000_mono.mp3",
        ])
        .collect();
    let joined: Vec<u8> = names.iter().flat_map(|n| file(n)).collect();
    let mut lin = open(joined.clone());
    let all = packets(&mut lin);
    let mut d = open(joined);
    let end = d.streams()[0].duration.unwrap();
    let mut targets: Vec<i64> = Vec::new();
    for p in &all {
        let (pts, dur) = (p.pts.unwrap(), p.duration.unwrap());
        targets.extend([pts, pts + dur / 2, pts + dur - 1]);
    }
    // Out of order on purpose: backward and forward seeks.
    targets.reverse();
    let third = targets.len() / 3;
    targets.rotate_left(third);
    for t in targets {
        let landed = d.seek_to(0, t).expect("seek");
        let idx = all
            .iter()
            .rposition(|p| p.pts.unwrap() <= t)
            .expect("a frame at or before");
        assert_eq!(landed, all[idx].pts.unwrap(), "seek to {t}");
        for want in &all[idx..(idx + 3).min(all.len())] {
            let got = d.next_packet().expect("packet after seek");
            assert_eq!(got.pts, want.pts, "after seek to {t}");
            assert!(got.data == want.data, "after seek to {t}");
        }
    }
    // Past the end: lands at the end, then EOF.
    assert_eq!(d.seek_to(0, end + 10_000).unwrap(), end);
    assert!(matches!(d.next_packet(), Err(Error::Eof)));
    // Back to the start.
    assert_eq!(d.seek_to(0, 0).unwrap(), 0);
    assert!(d.next_packet().unwrap().data == all[0].data);
}

/// A cut splice: the first file's last frame is truncated and the next
/// file starts inside its span. The cut frame is dropped; every whole
/// frame of both files comes out — unless the cut frame's nominal end
/// happens to land exactly on a frame start of the next file, which no
/// header lookahead can tell from an intact frame: then the cut frame
/// (with the next file's first frames inside it) is emitted as one
/// packet and framing continues from that frame start.
#[test]
fn truncated_frame_before_switch() {
    let mut coincident = 0;
    for (a, b) in [
        ("raw_44100.mp3", "raw_8000.mp3"),
        ("raw_8000.mp3", "raw_48000.mp3"),
        ("raw_44100_vbr.mp3", "raw_44100_mono.mp3"),
        ("raw_22050.mp3", "raw_22050.mp3"),
        ("raw_l2_48000.mp2", "raw_24000.mp3"),
    ] {
        let (fa, fb) = (file(a), file(b));
        let frames_a: Vec<&[u8]> = FrameWalker::new(&fa).map(|f| f.data).collect();
        let starts_b: Vec<usize> = FrameWalker::new(&fb).map(|f| f.offset).collect();
        let last = frames_a.last().unwrap().len();
        for cut in [1usize, 17, last / 2, last - 5] {
            let kept = fa.len() - cut;
            let mut joined = fa[..kept].to_vec();
            joined.extend_from_slice(&fb);
            let mut d = open(joined.clone());
            let pkts = packets(&mut d);
            let got: Vec<Vec<u8>> = pkts.iter().map(|p| p.data.clone()).collect();
            let cut_frame_at = kept - (last - cut);
            let nominal_end_in_b = cut_frame_at + last - kept;
            let mut want: Vec<Vec<u8>> = frames_a[..frames_a.len() - 1]
                .iter()
                .map(|f| f.to_vec())
                .collect();
            let from = if let Some(k) = starts_b.iter().position(|&o| o == nominal_end_in_b) {
                coincident += 1;
                want.push(joined[cut_frame_at..cut_frame_at + last].to_vec());
                k
            } else {
                0
            };
            want.extend(FrameWalker::new(&fb).skip(from).map(|f| f.data.to_vec()));
            assert!(got == want, "{a} cut by {cut} + {b}: whole frames only");
            check_timeline(&d, &pkts, "cut");
        }
    }
    // The fixtures hit the coincidence twice (e.g. a 44.1 kHz VBR frame
    // cut in half before 64 kbit/s mono frames about half its length).
    assert_eq!(coincident, 2, "coincident cuts");
}
