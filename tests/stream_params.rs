//! `Frame::StreamParams` (oxideav-core 0.1.43): the registry decoder
//! announces a mid-stream change of its output shape — the sample rate
//! and the channel count / layout `output_params()` describes — with one
//! in-band event, returned by `receive_frame` after every frame of the
//! old shape and immediately before the first frame of the new one.
//!
//! Fixtures: single-shape black-box ffmpeg / libmp3lame encodes
//! (`tests/fixtures/stream_params/gen.sh`), byte-concatenated here into
//! chained streams the way `cat a.mp3 b.mp3` produces them.
//!
//! What is a change (§2.4.2.3 header fields): `sampling_frequency`
//! under the version `ID` (so MPEG-1 ↔ MPEG-2 LSF ↔ MPEG-2.5 switches
//! are changes through their rate), and the layout `mode` implies
//! (mono / stereo pair / dual-channel programmes). What is not:
//! `bitrate_index` (VBR, free format `0`), and `stereo` ↔
//! `joint_stereo` (both decode to an L, R pair).

use oxideav_core::{
    ChannelLayout, CodecId, CodecParameters, Decoder, Error, Frame, OutputParamsWatch, Packet,
    SampleFormat, TimeBase,
};
use oxideav_mp3::codec_decoder::make_decoder;
use oxideav_mp3::frame::FrameWalker;

const FIX: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/stream_params");

fn file(name: &str) -> Vec<u8> {
    std::fs::read(format!("{FIX}/{name}")).expect("fixture readable")
}

fn frames_of(data: &[u8]) -> Vec<Vec<u8>> {
    FrameWalker::new(data).map(|f| f.data.to_vec()).collect()
}

fn decoder() -> Box<dyn Decoder> {
    make_decoder(&CodecParameters::audio(CodecId::new("mp3"))).expect("decoder")
}

/// One thing `receive_frame` returned.
#[derive(Debug)]
enum Out {
    /// A `StreamParams` event: its payload, plus `output_params()` and
    /// the generation at the moment it was returned.
    Event(Box<CodecParameters>, CodecParameters, u64),
    /// An audio frame: its planes (i16 per channel), and
    /// `output_params()` / the generation when it was returned.
    Audio(Vec<Vec<i16>>, CodecParameters, u64),
}

fn planes(af: &oxideav_core::AudioFrame) -> Vec<Vec<i16>> {
    af.data
        .iter()
        .map(|p| {
            p.as_chunks::<2>()
                .0
                .iter()
                .map(|c| i16::from_le_bytes(*c))
                .collect()
        })
        .collect()
}

/// Drain `receive_frame` until `NeedMore` / `Eof`.
fn drain(dec: &mut dyn Decoder, out: &mut Vec<Out>) {
    loop {
        match dec.receive_frame() {
            Ok(Frame::StreamParams(p)) => out.push(Out::Event(
                p,
                dec.output_params().expect("described"),
                dec.output_params_generation(),
            )),
            Ok(Frame::Audio(af)) => out.push(Out::Audio(
                planes(&af),
                dec.output_params().expect("described"),
                dec.output_params_generation(),
            )),
            Ok(other) => panic!("unexpected {other:?}"),
            Err(Error::NeedMore) | Err(Error::Eof) => break,
            Err(e) => panic!("decode: {e}"),
        }
    }
}

fn send(dec: &mut dyn Decoder, pkt: &[u8]) {
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 1), pkt.to_vec()))
        .expect("packet");
}

/// Decode `pkts`, draining after every packet (`interleaved`) or only
/// after `flush` (every frame queued first).
fn decode(pkts: &[Vec<u8>], interleaved: bool) -> Vec<Out> {
    let mut dec = decoder();
    let mut out = Vec::new();
    for p in pkts {
        send(dec.as_mut(), p);
        if interleaved {
            drain(dec.as_mut(), &mut out);
        }
    }
    dec.flush().unwrap();
    drain(dec.as_mut(), &mut out);
    out
}

/// Just the audio of a standalone decode of one fixture.
fn standalone(name: &str) -> Vec<Vec<Vec<i16>>> {
    decode(&frames_of(&file(name)), true)
        .into_iter()
        .map(|o| match o {
            Out::Audio(p, ..) => p,
            Out::Event(..) => panic!("{name}: a constant stream has no event"),
        })
        .collect()
}

fn shape(p: &CodecParameters) -> (Option<u32>, Option<u16>, Option<ChannelLayout>) {
    (p.sample_rate, p.channels, p.channel_layout)
}

/// Checks the event rules on `out` and returns the audio split into
/// segments at the events.
fn check_and_split(out: &[Out]) -> Vec<Vec<Vec<Vec<i16>>>> {
    let mut segments: Vec<Vec<Vec<Vec<i16>>>> = vec![Vec::new()];
    let mut gen_now = 0u64;
    let mut prev_shape = None;
    let mut after_event = false;
    for (i, o) in out.iter().enumerate() {
        match o {
            Out::Event(payload, now, g) => {
                assert!(!after_event, "#{i}: two events in a row");
                assert!(prev_shape.is_some(), "#{i}: no event for the first shape");
                assert_eq!(
                    format!("{payload:?}"),
                    format!("{now:?}"),
                    "#{i}: payload == output_params()"
                );
                assert_eq!(*g, gen_now.wrapping_add(1), "#{i}: one bump per event");
                assert_ne!(Some(shape(payload)), prev_shape, "#{i}: a real change");
                assert_eq!(payload.sample_format, Some(SampleFormat::S16P));
                gen_now = *g;
                after_event = true;
                segments.push(Vec::new());
            }
            Out::Audio(planes, now, g) => {
                assert_eq!(*g, gen_now, "#{i}: the generation moves only at an event");
                assert_eq!(
                    planes.len(),
                    usize::from(now.channels.unwrap()),
                    "#{i}: the frame has the announced channel count"
                );
                if let Some(prev) = prev_shape
                    && !after_event
                {
                    assert_eq!(shape(now), prev, "#{i}: shape moved without an event");
                }
                prev_shape = Some(shape(now));
                after_event = false;
                segments.last_mut().unwrap().push(planes.clone());
            }
        }
    }
    segments
}

#[test]
fn constant_streams_have_no_event() {
    for name in [
        "joint_44100.mp3",
        "stereo_44100.mp3",
        "mono_44100.mp3",
        "mono_22050.mp3",
        "vbr_48000.mp3",
    ] {
        let out = decode(&frames_of(&file(name)), true);
        assert!(out.iter().all(|o| matches!(o, Out::Audio(..))), "{name}");
        assert!(out.len() >= 8, "{name}");
    }
}

/// A VBR stream changes `bitrate_index` frame to frame; `output_params()`
/// does not describe the bit rate, so it is no shape change.
#[test]
fn vbr_bitrate_changes_are_not_a_shape_change() {
    let pkts = frames_of(&file("vbr_48000.mp3"));
    let mut indices: Vec<u8> = pkts.iter().map(|p| p[2] >> 4).collect();
    indices.sort_unstable();
    indices.dedup();
    assert!(indices.len() >= 3, "fixture is VBR: {indices:?}");
    let out = decode(&pkts, true);
    let segments = check_and_split(&out);
    assert_eq!(segments.len(), 1);
    for o in &out {
        let Out::Audio(_, p, g) = o else {
            unreachable!()
        };
        assert_eq!(*g, 0);
        assert_eq!(p.bit_rate, None, "the bit rate is not part of the shape");
        assert_eq!(
            shape(p),
            (Some(48_000), Some(2), Some(ChannelLayout::Stereo))
        );
    }
}

/// Free format (`bitrate_index == 0`, §2.4.2.3) mixed with CBR frames of
/// the same rate and mode: the frames are the same bytes apart from the
/// header nibble, so the PCM is identical and there is no event.
#[test]
fn free_format_frames_are_not_a_shape_change() {
    let mut pkts = frames_of(&file("joint_44100.mp3"));
    for p in pkts.iter_mut().skip(1).step_by(2) {
        p[2] &= 0x0F;
    }
    assert_eq!(pkts[1][2] >> 4, 0, "free format");
    let out = decode(&pkts, true);
    let segments = check_and_split(&out);
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0], standalone("joint_44100.mp3"));
}

/// `stereo` (`00`) ↔ `joint_stereo` (`01`) keeps the L, R pair.
#[test]
fn stereo_joint_stereo_switch_is_not_a_change() {
    let mut cat = file("joint_44100.mp3");
    cat.extend(file("stereo_44100.mp3"));
    cat.extend(file("joint_44100.mp3"));
    let pkts = frames_of(&cat);
    let modes: Vec<u8> = pkts.iter().map(|p| p[3] >> 6).collect();
    assert!(modes.contains(&0) && modes.contains(&1), "{modes:?}");
    let out = decode(&pkts, true);
    assert_eq!(check_and_split(&out).len(), 1);
}

/// The chained fixture order and the shape each segment decodes to.
const CHAIN: &[(&str, u32, u16)] = &[
    ("joint_44100.mp3", 44_100, 2),
    // mono ↔ stereo at the same rate
    ("mono_44100.mp3", 44_100, 1),
    // MPEG-1 → MPEG-2 LSF
    ("mono_22050.mp3", 22_050, 1),
    // MPEG-2 LSF → MPEG-1, mono → stereo, VBR
    ("vbr_48000.mp3", 48_000, 2),
    // rate change, stereo kept (joint → independent stereo)
    ("stereo_44100.mp3", 44_100, 2),
];

fn chain_packets() -> Vec<Vec<u8>> {
    let mut cat = Vec::new();
    for (name, ..) in CHAIN {
        cat.extend(file(name));
    }
    frames_of(&cat)
}

/// Concatenated files: one event per boundary, placed after every frame
/// of the old file and before the first of the new; each segment decodes
/// to exactly what the file decodes to on its own (the old shape's
/// carry-over state does not leak into the new one).
#[test]
fn concatenated_files_announce_each_change_once() {
    for interleaved in [true, false] {
        let out = decode(&chain_packets(), interleaved);
        let segments = check_and_split(&out);
        assert_eq!(segments.len(), CHAIN.len(), "interleaved={interleaved}");
        let events: Vec<&CodecParameters> = out
            .iter()
            .filter_map(|o| match o {
                Out::Event(p, ..) => Some(&**p),
                _ => None,
            })
            .collect();
        for (k, ((name, rate, ch), seg)) in CHAIN.iter().zip(&segments).enumerate() {
            assert_eq!(
                *seg,
                standalone(name),
                "segment {k} ({name}) == its standalone decode, interleaved={interleaved}"
            );
            if k > 0 {
                let e = events[k - 1];
                assert_eq!((e.sample_rate, e.channels), (Some(*rate), Some(*ch)));
            }
        }
        let Some(Out::Audio(_, _, g)) = out.last() else {
            panic!()
        };
        assert_eq!(*g, (CHAIN.len() - 1) as u64);
    }
}

/// Dual channel (`mode == 10`) decodes to two independent programmes,
/// which `output_params()` reports as `DiscreteN(2)`, not `Stereo`, so a
/// stereo ↔ dual-channel switch changes the description and is
/// announced. (Stereo and dual channel share the side-info layout and
/// decode identically, so rewriting the header `mode` of a stereo
/// fixture yields a valid dual-channel stream.)
#[test]
fn stereo_dual_channel_switch_follows_the_layout() {
    let stereo = frames_of(&file("stereo_44100.mp3"));
    assert!(stereo.iter().all(|p| p[3] >> 6 == 0 && p[1] & 1 == 1));
    let dual: Vec<Vec<u8>> = stereo
        .iter()
        .map(|p| {
            let mut p = p.clone();
            p[3] = (p[3] & 0x3F) | (0b10 << 6);
            p
        })
        .collect();
    let mut pkts = stereo.clone();
    pkts.extend(dual);
    let out = decode(&pkts, true);
    let segments = check_and_split(&out);
    assert_eq!(segments.len(), 2);
    let ev = out
        .iter()
        .find_map(|o| match o {
            Out::Event(p, ..) => Some(p),
            _ => None,
        })
        .unwrap();
    assert_eq!(ev.channel_layout, Some(ChannelLayout::DiscreteN(2)));
    assert_eq!(ev.channels, Some(2));
    assert_eq!(segments[1], segments[0], "same PCM, new layout");
}

/// `reset()` (a seek): a different shape after it is announced before the
/// first frame; the same shape is not.
#[test]
fn reset_announces_a_new_shape_only() {
    let mut dec = decoder();
    let mut out = Vec::new();
    for p in frames_of(&file("mono_22050.mp3")) {
        send(dec.as_mut(), &p);
        drain(dec.as_mut(), &mut out);
    }
    // Seek inside the same shape: no event.
    dec.reset().unwrap();
    let mono = frames_of(&file("mono_22050.mp3"));
    for p in &mono[3..] {
        send(dec.as_mut(), p);
        drain(dec.as_mut(), &mut out);
    }
    // Seek into a stereo 44.1 kHz segment, sending before draining.
    send(dec.as_mut(), &mono[4]);
    dec.reset().unwrap();
    for p in frames_of(&file("joint_44100.mp3")) {
        send(dec.as_mut(), &p);
    }
    dec.flush().unwrap();
    drain(dec.as_mut(), &mut out);
    let segments = check_and_split(&out);
    assert_eq!(segments.len(), 2);
    assert_eq!(segments[1], standalone("joint_44100.mp3"));
    assert_eq!(dec.output_params_generation(), 1);
}

/// The consumer helpers agree: `observe_frame` on the events and
/// `observe` on the counter each report every change once.
#[test]
fn consumer_watch_sees_each_change_once() {
    let mut dec = decoder();
    let mut watch = OutputParamsWatch::default();
    let mut stream = CodecParameters::audio(CodecId::new("mp3"));
    let mut changes = 0;
    let mut events = 0;
    for p in chain_packets() {
        send(dec.as_mut(), &p);
        loop {
            match dec.receive_frame() {
                Ok(f) => {
                    if f.is_stream_params() {
                        events += 1;
                        assert!(watch.observe_frame(&f, &mut stream));
                        changes += 1;
                    } else if watch.observe(dec.as_ref(), &mut stream) {
                        changes += 1;
                    }
                }
                Err(Error::NeedMore) => break,
                Err(e) => panic!("{e}"),
            }
        }
    }
    assert_eq!(events, CHAIN.len() - 1);
    // The first shape is resolved through `observe`; each later change
    // through its event only.
    assert_eq!(changes, CHAIN.len());
    assert_eq!(stream.sample_rate, Some(44_100));
}
