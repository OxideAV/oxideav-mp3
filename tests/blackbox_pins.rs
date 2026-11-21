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
