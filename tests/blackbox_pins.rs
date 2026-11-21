//! Small committed pins for the defects the live black-box matrix
//! (`tests/blackbox_matrix.rs`) exposed. Every stream here is built
//! in the test from header fields; nothing depends on an installed
//! tool.

use std::io::Cursor;

use oxideav_core::{CodecId, CodecParameters, Decoder, Demuxer, Error, Frame, Packet};
use oxideav_mp3::codec_decoder::make_decoder;
use oxideav_mp3::frame::FrameWalker;
use oxideav_mp3::side_info::parse_side_info;
use oxideav_mp3::{Mp3Demuxer, parse_header};

/// A digitally silent Layer III frame: `header`, zero side information
/// and zero main data, `len` bytes long.
fn silent_frame(header: [u8; 4], len: usize) -> Vec<u8> {
    let mut f = vec![0u8; len];
    f[..4].copy_from_slice(&header);
    f
}

fn packets(d: &mut Mp3Demuxer) -> Vec<Packet> {
    let mut out = Vec::new();
    loop {
        match d.next_packet() {
            Ok(p) => out.push(p),
            Err(Error::Eof) => return out,
            Err(e) => panic!("next_packet: {e}"),
        }
    }
}

/// Samples per channel of every audio frame the decoder returns.
fn decode_lengths(pkts: &[Packet]) -> Vec<usize> {
    let mut dec: Box<dyn Decoder> =
        make_decoder(&CodecParameters::audio(CodecId::new("mp3"))).expect("decoder");
    let mut out = Vec::new();
    let mut drain = |dec: &mut dyn Decoder| {
        loop {
            match dec.receive_frame() {
                Ok(Frame::Audio(af)) => out.push(af.data[0].len() / 2),
                Ok(_) => {}
                Err(Error::NeedMore) | Err(Error::Eof) => break,
                Err(e) => panic!("decode: {e}"),
            }
        }
    };
    for p in pkts {
        dec.send_packet(p).expect("send_packet");
        drain(dec.as_mut());
    }
    dec.flush().unwrap();
    drain(dec.as_mut());
    out
}

/// MPEG-1 Layer III, 44.1 kHz, 128 kbit/s, joint stereo: 417 bytes
/// unpadded. `FF FB` = no CRC, `FF FA` = CRC-protected.
const H_128: [u8; 4] = [0xFF, 0xFB, 0x90, 0x64];
const H_128_CRC: [u8; 4] = [0xFF, 0xFA, 0x90, 0x64];

/// A CRC-protected information frame keeps the "Info" / "Xing" magic
/// at the offset `lame-xing-info-tag.md` §1 tabulates from the frame
/// start (36 for MPEG-1 stereo) — not two bytes later. Such a frame
/// was taken for audio: one frame too many, 1152 samples of offset.
#[test]
fn info_frame_in_a_crc_protected_stream_is_not_audio() {
    for magic in [b"Info", b"Xing"] {
        let mut info = silent_frame(H_128_CRC, 417);
        info[36..40].copy_from_slice(magic);
        info[43] = 0x01; // flags: frames field present
        info[47] = 5; // frames = 5
        let mut bytes = info;
        for _ in 0..5 {
            bytes.extend(silent_frame(H_128, 417));
        }
        let mut d = Mp3Demuxer::open(Box::new(Cursor::new(bytes))).expect("open");
        let tag = d.xing().expect("information frame recognised");
        assert_eq!(tag.frames, Some(5));
        assert_eq!(d.audio_frame_count(), 5);
        assert_eq!(d.first_audio_frame_offset(), 417);
        assert_eq!(packets(&mut d).len(), 5);
    }
}

/// The same frame with the magic after the CRC word and the whole
/// side information (offset 38) is recognised as well.
#[test]
fn info_frame_after_the_crc_word_is_recognised_too() {
    let mut info = silent_frame(H_128_CRC, 417);
    info[38..42].copy_from_slice(b"Info");
    let mut bytes = info;
    for _ in 0..4 {
        bytes.extend(silent_frame(H_128, 417));
    }
    let mut d = Mp3Demuxer::open(Box::new(Cursor::new(bytes))).expect("open");
    assert!(d.xing().is_some());
    assert_eq!(d.audio_frame_count(), 4);
    assert_eq!(packets(&mut d).len(), 4);
}

/// Free-format Layer III above the bitrate a decoder is required to
/// support (320 kbit/s MPEG-1, 160 kbit/s at the lower sampling
/// frequencies) is still delimited by its headers; such files failed
/// to open ("no MPEG audio frame sync").
#[test]
fn free_format_above_the_mandatory_ceiling_opens_and_decodes() {
    // (header, frame bytes, samples per frame, sampling frequency)
    let cells: [([u8; 4], usize, usize, u64); 3] = [
        // MPEG-1 44.1 kHz stereo, 1371-byte frames = 419.9 kbit/s.
        ([0xFF, 0xFB, 0x00, 0x64], 1371, 1152, 44_100),
        // MPEG-1 48 kHz stereo, 1920-byte frames = 640 kbit/s.
        ([0xFF, 0xFB, 0x04, 0x64], 1920, 1152, 48_000),
        // MPEG-2 22.05 kHz stereo, 979-byte frames = 299.8 kbit/s.
        ([0xFF, 0xF3, 0x00, 0x64], 979, 576, 22_050),
    ];
    for (header, len, spf, rate) in cells {
        let h = parse_header(&header).expect("header");
        assert!(h.is_free_format());
        assert_eq!(u64::from(h.sample_rate_hz), rate);
        let mut bytes = Vec::new();
        for _ in 0..6 {
            bytes.extend(silent_frame(header, len));
        }
        let mut d = Mp3Demuxer::open(Box::new(Cursor::new(bytes)))
            .unwrap_or_else(|e| panic!("{len}-byte free-format frames: {e}"));
        assert_eq!(d.audio_frame_count(), 6);
        let pkts = packets(&mut d);
        assert_eq!(pkts.len(), 6);
        assert!(pkts.iter().all(|p| p.data.len() == len));
        assert_eq!(decode_lengths(&pkts), vec![spf; 6]);
    }
}

/// A stream entered at a frame whose `main_data_begin` points at main
/// data that was never received (cut head, seek): that frame cannot be
/// decoded, but it still occupies its samples on the timeline. It was
/// dropped, which shifted every later sample one frame early relative
/// to the container's timestamps and to both black-box decoders.
#[test]
fn a_frame_with_missing_reservoir_data_is_rendered_as_silence() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/splice/raw_44100_128k.mp3"
    );
    let bytes = std::fs::read(path).expect("fixture");
    // First frame that borrows main data from its predecessors.
    let mut offset = 0usize;
    let mut cut = None;
    for f in FrameWalker::new(&bytes) {
        let si = parse_side_info(&f.header, &f.data[4..]).expect("side info");
        if si.main_data_begin > 0 {
            cut = Some(offset);
            break;
        }
        offset += f.data.len();
    }
    let cut = cut.expect("the fixture uses the bit reservoir");
    let tail = bytes[cut..].to_vec();
    let frames = FrameWalker::new(&tail).count();
    let mut d = Mp3Demuxer::open(Box::new(Cursor::new(tail))).expect("open");
    let pkts = packets(&mut d);
    assert_eq!(pkts.len(), frames);
    let lens = decode_lengths(&pkts);
    assert_eq!(lens, vec![1152; frames], "one audio frame per packet");

    // The undecodable first frame is silence.
    let mut dec = make_decoder(&CodecParameters::audio(CodecId::new("mp3"))).unwrap();
    dec.send_packet(&pkts[0]).unwrap();
    match dec.receive_frame() {
        Ok(Frame::Audio(af)) => {
            assert_eq!(af.pts, pkts[0].pts);
            assert_eq!(af.data.len(), 2);
            assert!(af.data.iter().all(|p| p.iter().all(|&b| b == 0)));
        }
        other => panic!("expected a silent audio frame, got {other:?}"),
    }
}

// ---------------------------------------------------------------------
// Encoder pins (direction ours -> foreign).
// ---------------------------------------------------------------------

use oxideav_mp3::{ChannelMode, Mp3Encoder};

/// Deterministic wide-band noise, mono.
fn noise(n: usize) -> Vec<i16> {
    let mut s = 12_345u32;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((s >> 16) as i16) / 4
        })
        .collect()
}

fn encode(mut enc: Mp3Encoder, pcm: &[i16]) -> Vec<u8> {
    enc.push_samples(pcm).expect("push");
    let mut out = Vec::new();
    enc.finish(&mut out).expect("finish");
    out
}

/// `part2_3_length` is a 12-bit field. At 8 kHz mono 64 kbit/s a frame
/// holds 4608 bits for its single granule-channel; the quantizer spent
/// past 4095 and the written length wrapped (e.g. 4474 -> 378), which
/// two black-box decoders rendered as silence / noise.
#[test]
fn part2_3_length_never_exceeds_its_twelve_bits() {
    for (kbps, rate) in [(64u32, 8_000u32), (56, 8_000), (160, 24_000), (320, 48_000)] {
        let enc = Mp3Encoder::new(kbps, rate, ChannelMode::SingleChannel).expect("encoder");
        let out = encode(enc, &noise(rate as usize));
        // Every frame's main data must hold what its side info claims:
        // walk the stream with a running reservoir.
        let mut reservoir = 0usize;
        let mut frames = 0;
        for f in FrameWalker::new(&out) {
            let si = parse_side_info(&f.header, &f.data[4..]).expect("side info");
            let side = si.byte_len();
            let claimed: usize = (0..si.granule_count as usize)
                .map(|gr| usize::from(si.granules[gr][0].part2_3_length))
                .sum();
            let available = (usize::from(si.main_data_begin) + f.data.len() - 4 - side) * 8;
            assert!(usize::from(si.main_data_begin) <= reservoir);
            assert!(
                claimed <= available,
                "{kbps} kbit/s at {rate} Hz: {claimed} bits claimed, {available} present"
            );
            reservoir = (reservoir + f.data.len() - 4 - side).min(511);
            frames += 1;
        }
        assert!(frames > 10);
        // And the decode is the noise, not silence: its energy is
        // within 3 dB of the input's.
        let mut d = Mp3Demuxer::open(Box::new(Cursor::new(out))).expect("open");
        let pkts = packets(&mut d);
        let mut dec = make_decoder(&CodecParameters::audio(CodecId::new("mp3"))).unwrap();
        let mut energy = 0f64;
        let mut n = 0usize;
        for p in &pkts[4..pkts.len() - 4] {
            dec.send_packet(p).unwrap();
            if let Ok(Frame::Audio(af)) = dec.receive_frame() {
                for c in af.data[0].as_chunks::<2>().0 {
                    let v = f64::from(i16::from_le_bytes(*c));
                    energy += v * v;
                    n += 1;
                }
            }
        }
        let input: f64 = noise(1000)
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum::<f64>()
            / 1000.0;
        let ratio = energy / n as f64 / input;
        assert!(
            (0.5..2.0).contains(&ratio),
            "{kbps} kbit/s at {rate} Hz: decoded / input energy = {ratio}"
        );
    }
}

/// The end of the input used to be lost: a sample entering the encoder
/// leaves a decoder 1057 samples later, and the stream stopped at the
/// frame boundary after the last input sample. The stream now runs
/// past `input + 1057`, and the information frame's delay / padding
/// give the input length back exactly.
#[test]
fn the_stream_covers_the_last_input_sample_and_declares_its_length() {
    for (kbps, rate, spf) in [
        (128u32, 44_100u32, 1152usize),
        (64, 22_050, 576),
        (32, 8_000, 576),
    ] {
        // Lengths around the frame boundaries, where the old stream
        // ended exactly at the input.
        for n in [spf * 8, spf * 8 + 1, spf * 8 - 1, spf * 8 + spf / 2, 1, 100] {
            let mut pcm = vec![0i16; n];
            // A burst in the last samples.
            for (i, v) in pcm.iter_mut().rev().take(8).enumerate() {
                *v = if i % 2 == 0 { 20_000 } else { -20_000 };
            }
            let mut enc = Mp3Encoder::new(kbps, rate, ChannelMode::SingleChannel).unwrap();
            enc.enable_info_frame();
            let out = encode(enc, &pcm);
            let mut d = Mp3Demuxer::open(Box::new(Cursor::new(out))).expect("open");
            assert_eq!(d.encoder_delay_samples(), Some(528));
            assert_eq!(
                d.trimmed_duration_samples(),
                Some(n as i64),
                "{rate} Hz, {n} samples"
            );
            let frames = d.audio_frame_count() as usize;
            assert_eq!(frames, (n + 1057).div_ceil(spf), "{rate} Hz, {n} samples");
            // The burst is in the decode, at `position + 1057`.
            let pkts = packets(&mut d);
            let mut dec = make_decoder(&CodecParameters::audio(CodecId::new("mp3"))).unwrap();
            let mut decoded: Vec<i16> = Vec::new();
            for p in &pkts {
                dec.send_packet(p).unwrap();
                while let Ok(Frame::Audio(af)) = dec.receive_frame() {
                    decoded.extend(
                        af.data[0]
                            .as_chunks::<2>()
                            .0
                            .iter()
                            .map(|c| i16::from_le_bytes(*c)),
                    );
                }
            }
            assert_eq!(decoded.len(), frames * spf);
            if n >= 8 {
                let at = n - 8 + 1057;
                let peak = decoded[at..at + 8]
                    .iter()
                    .map(|v| v.unsigned_abs())
                    .max()
                    .unwrap();
                assert!(
                    peak > 2_000,
                    "{rate} Hz, {n} samples: the final burst is missing (peak {peak})"
                );
            }
        }
    }
}
