//! `Decoder::output_params` / `output_params_generation` (oxideav-core
//! 0.1.42): the registry decoder states the decoded PCM's sample rate,
//! channel count / layout and sample format at the stream level as
//! soon as it has parsed a frame header, and bumps the generation
//! exactly at the first frame of a new shape (a chained stream
//! switching MPEG version, rate or mode).
//!
//! Fixtures come from black-box ffmpeg / libmp3lame
//! (`tests/fixtures/output_params/gen.sh`, which also prints the
//! `ffprobe -show_streams` values pinned below). ffprobe's `sample_fmt`
//! is ffmpeg's own decoder's; this decoder emits planar `S16P` (one
//! little-endian `i16` plane per channel), checked against the frames.

use oxideav_core::{
    ChannelLayout, CodecId, CodecParameters, Decoder, Error, Frame, Packet, SampleFormat, TimeBase,
};
use oxideav_mp3::codec_decoder::make_decoder;
use oxideav_mp3::frame::FrameWalker;

const FIX: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/output_params");

fn packets(name: &str) -> Vec<Vec<u8>> {
    let data = std::fs::read(format!("{FIX}/{name}")).expect("fixture readable");
    FrameWalker::new(&data).map(|f| f.data.to_vec()).collect()
}

/// `(file, sample_rate, channels, samples per channel per frame)` —
/// ffprobe: `sample_rate` / `channels` / `channel_layout` (mono /
/// stereo) as listed; Layer III frames carry 1152 samples (MPEG-1) or
/// 576 (MPEG-2 LSF / MPEG-2.5) per channel.
const FIXTURES: &[(&str, u32, u16, u32)] = &[
    ("mpeg1_44100_stereo.mp3", 44_100, 2, 1152),
    ("mpeg1_32000_mono.mp3", 32_000, 1, 1152),
    ("mpeg2_22050_stereo.mp3", 22_050, 2, 576),
    ("mpeg2_16000_mono.mp3", 16_000, 1, 576),
    ("mpeg25_11025_mono.mp3", 11_025, 1, 576),
    ("mpeg25_8000_stereo.mp3", 8_000, 2, 576),
];

fn decoder() -> Box<dyn Decoder> {
    // Deliberately wrong container hints: the answer comes from the
    // frame headers, never from these.
    let mut p = CodecParameters::audio(CodecId::new("mp3"));
    p.sample_rate = Some(48_000);
    p.channels = Some(1);
    make_decoder(&p).expect("decoder")
}

fn layout_of(channels: u16) -> ChannelLayout {
    if channels == 1 {
        ChannelLayout::Mono
    } else {
        ChannelLayout::Stereo
    }
}

/// Send every packet, draining after each; returns `(generation,
/// params)` per returned frame, checking each frame against the
/// description and that a sent-but-not-returned frame never moves it.
fn decode_all(dec: &mut dyn Decoder, pkts: &[Vec<u8>]) -> Vec<(u64, CodecParameters)> {
    let mut out = Vec::new();
    decode_into(dec, pkts, &mut out);
    out
}

/// [`decode_all`] appending to `out` (which holds what the same
/// decoder returned before).
fn decode_into(dec: &mut dyn Decoder, pkts: &[Vec<u8>], out: &mut Vec<(u64, CodecParameters)>) {
    for (i, data) in pkts.iter().enumerate() {
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 1), data.clone()).with_pts(i as i64))
            .expect("packet");
        let before = dec.output_params().expect("known once a header is parsed");
        if let Some((g, prev)) = out.last() {
            assert_eq!(dec.output_params_generation(), *g);
            assert_eq!(format!("{before:?}"), format!("{prev:?}"));
        }
        loop {
            match dec.receive_frame() {
                Ok(Frame::Audio(af)) => {
                    let p = dec.output_params().expect("described after a frame");
                    let ch = p.channels.unwrap() as usize;
                    assert_eq!(af.data.len(), ch, "one plane per channel");
                    for plane in &af.data {
                        assert_eq!(plane.len(), af.samples as usize * 2, "S16 plane");
                    }
                    if out.is_empty() {
                        assert_eq!(format!("{before:?}"), format!("{p:?}"));
                    }
                    out.push((dec.output_params_generation(), p));
                }
                Ok(_) => panic!("audio only"),
                Err(Error::NeedMore) => break,
                Err(e) => panic!("decode: {e}"),
            }
        }
    }
}

#[test]
fn every_mpeg_version_and_mode_matches_ffprobe() {
    for &(name, rate, channels, _) in FIXTURES {
        let pkts = packets(name);
        assert!(pkts.len() >= 4, "{name}");
        let mut dec = decoder();
        assert!(
            dec.output_params().is_none(),
            "{name}: no header parsed yet"
        );
        // The first header states the shape before any frame returns.
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 1), pkts[0].clone()))
            .unwrap();
        let early = dec.output_params().expect("from the first header");
        assert_eq!(early.sample_rate, Some(rate), "{name}");
        assert_eq!(early.channels, Some(channels), "{name}");
        assert_eq!(early.channel_layout, Some(layout_of(channels)), "{name}");
        assert_eq!(early.sample_format, Some(SampleFormat::S16P), "{name}");

        let mut dec = decoder();
        let seen = decode_all(dec.as_mut(), &pkts);
        assert!(!seen.is_empty(), "{name}");
        for (g, p) in &seen {
            assert_eq!(*g, 0, "{name}: fixed shape");
            assert_eq!(
                (p.sample_rate, p.channels, p.channel_layout, p.sample_format),
                (
                    Some(rate),
                    Some(channels),
                    Some(layout_of(channels)),
                    Some(SampleFormat::S16P)
                ),
                "{name}"
            );
        }
    }
}

/// Samples per frame follow the version (checked on the frames).
#[test]
fn frame_lengths_follow_the_version() {
    for &(name, _, _, spf) in FIXTURES {
        let mut dec = decoder();
        for data in packets(name) {
            dec.send_packet(&Packet::new(0, TimeBase::new(1, 1), data))
                .unwrap();
            while let Ok(Frame::Audio(af)) = dec.receive_frame() {
                assert_eq!(af.samples, spf, "{name}");
            }
        }
    }
}

/// Chained streams switching MPEG version, rate and mode: the
/// generation bumps exactly at the first frame of each new shape (a
/// return to an earlier shape is a change too), and survives `reset`.
#[test]
fn chained_streams_bump_the_generation_at_the_first_new_frame() {
    let order = [0usize, 2, 4, 1, 1, 5, 0];
    let mut pkts = Vec::new();
    let mut expect_shapes = Vec::new();
    for &k in &order {
        let p = packets(FIXTURES[k].0);
        pkts.push(p);
        expect_shapes.push((FIXTURES[k].1, FIXTURES[k].2));
    }
    let mut dec = decoder();
    let mut gen_expected = 0u64;
    let mut last: Option<(u32, u16)> = None;
    let mut all = Vec::new();
    for (seg, (p, &shape)) in pkts.iter().zip(&expect_shapes).enumerate() {
        let start = all.len();
        decode_into(dec.as_mut(), p, &mut all);
        let seen = &all[start..];
        assert!(!seen.is_empty(), "segment {seg}");
        if last.is_some_and(|l| l != shape) {
            gen_expected += 1;
        }
        last = Some(shape);
        for (g, params) in seen {
            assert_eq!(*g, gen_expected, "segment {seg}");
            assert_eq!(
                (params.sample_rate.unwrap(), params.channels.unwrap()),
                shape
            );
        }
    }
    assert_eq!(gen_expected, 5, "0→2→4→1 (=1) →5→0: five changes");

    // A seek: reset keeps the answer until the next frame returns.
    let before = format!("{:?}", dec.output_params().unwrap());
    dec.reset().unwrap();
    assert_eq!(format!("{:?}", dec.output_params().unwrap()), before);
    let start = all.len();
    decode_into(dec.as_mut(), &packets(FIXTURES[3].0), &mut all);
    assert_eq!(all[start].0, gen_expected + 1);
    assert_eq!(all[start].1.sample_rate, Some(16_000));
}
