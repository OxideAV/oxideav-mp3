//! Free-format MPEG audio stream demuxing
//! (ISO/IEC 11172-3 §2.4.2.3 final paragraph).
//!
//! When `bitrate_index == 0` the frame header alone does not encode
//! the frame's byte size; the decoder must measure the inter-sync
//! distance to the next matching header and cache that value
//! (free-format streams are required by the spec to use a constant
//! frame size). This test hand-crafts a tiny free-format file
//! (FFmpeg's libmp3lame wrapper rejects `-b:a 0` so we can't borrow a
//! real fixture from ffmpeg) and walks it through the demuxer.
//!
//! Layout (44.1 kHz / mono / Layer III / free-format / fixed
//! per-frame size of 500 bytes):
//!
//! ```text
//!   byte 0 = 0xFF
//!   byte 1 = 11111011 = 0xFB (sync 111 | ver 11 MPEG-1 | layer 01 LIII | noCRC 1)
//!   byte 2 = 00000000 = 0x00 (bitrate_index 0000 free | sr_index 00 44100 | pad 0 | priv 0)
//!   byte 3 = 11000000 = 0xC0 (mode 11 mono | ext 00 | copy 0 | orig 0 | emph 00)
//!   bytes 4..500 = zero-padded body (decoder treats as silence)
//! ```

use std::io::Cursor;

use oxideav_core::ContainerRegistry;
use oxideav_mp3::frame::{parse_frame_header, MpegVersion};

const FREE_FORMAT_FRAME_BYTES: usize = 500;

/// Build a single free-format Layer III frame at 44.1 kHz / mono with
/// the chosen constant size. Body is zero-padded — the body's content
/// does not influence the demuxer (which only walks frame boundaries),
/// only the inter-sync distance does.
fn build_free_format_frame() -> Vec<u8> {
    let mut f = vec![0u8; FREE_FORMAT_FRAME_BYTES];
    f[0] = 0xFF;
    f[1] = 0xFB;
    f[2] = 0x00;
    f[3] = 0xC0;
    f
}

#[test]
fn free_format_header_parses_but_frame_bytes_is_none() {
    // Sanity check: the frame parser exposes free-format (bitrate_index=0)
    // by returning a header with `bitrate_kbps == 0` and `frame_bytes()
    // == None`. Container code branches on that signal to decide whether
    // to measure the inter-sync distance.
    let frame = build_free_format_frame();
    let hdr = parse_frame_header(&frame[..4]).expect("parse free-format header");
    assert_eq!(hdr.version, MpegVersion::Mpeg1);
    assert_eq!(hdr.bitrate_index, 0, "expected free-format index 0");
    assert_eq!(hdr.bitrate_kbps, 0, "expected free-format bitrate 0");
    assert_eq!(hdr.sample_rate, 44_100);
    assert!(
        hdr.frame_bytes().is_none(),
        "free-format must NOT have a header-derivable frame size"
    );
}

#[test]
fn free_format_demuxer_walks_frames_with_measured_size() {
    // Build a 5-frame free-format stream. The demuxer must:
    //   1. Detect bitrate_index=0 on the first header.
    //   2. Measure the distance to the second sync — 500 bytes.
    //   3. Emit each subsequent packet at exactly 500-byte boundaries.
    // The measurement happens once at open(); per-frame the value is
    // reused (constant-size invariant).
    const N_FRAMES: usize = 5;
    let mut file = Vec::with_capacity(FREE_FORMAT_FRAME_BYTES * N_FRAMES);
    for _ in 0..N_FRAMES {
        file.extend_from_slice(&build_free_format_frame());
    }

    let mut reg = ContainerRegistry::new();
    oxideav_mp3::register_containers(&mut reg);

    let cursor: Box<dyn oxideav_core::ReadSeek> = Box::new(Cursor::new(file));
    let mut demuxer = reg
        .open_demuxer("mp3", cursor, &oxideav_core::NullCodecResolver)
        .expect("open free-format mp3 demuxer");

    let stream = &demuxer.streams()[0];
    assert_eq!(stream.params.sample_rate, Some(44_100));
    assert_eq!(stream.params.channels, Some(1));
    assert_eq!(stream.params.codec_id.as_str(), "mp3");

    let mut count = 0usize;
    loop {
        match demuxer.next_packet() {
            Ok(pkt) => {
                assert_eq!(
                    pkt.data.len(),
                    FREE_FORMAT_FRAME_BYTES,
                    "free-format frame {count} has wrong measured size: {} (expected {FREE_FORMAT_FRAME_BYTES})",
                    pkt.data.len()
                );
                // Each packet starts with the 4-byte free-format header.
                assert_eq!(pkt.data[0], 0xFF);
                assert_eq!(pkt.data[1], 0xFB);
                assert_eq!(pkt.data[2], 0x00);
                assert_eq!(pkt.data[3], 0xC0);
                count += 1;
            }
            Err(oxideav_core::Error::Eof) => break,
            Err(e) => panic!("unexpected demuxer error after {count} frames: {e:?}"),
        }
    }
    // MPEG-1 Layer III emits 1152 samples per frame regardless of bit
    // rate; PTS should advance by that amount per packet. We assert
    // packet count == frames written; with no Xing tag this is simply
    // the full input.
    assert_eq!(
        count, N_FRAMES,
        "expected {N_FRAMES} packets from free-format stream, got {count}"
    );
}

#[test]
fn free_format_rejects_when_no_second_sync() {
    // A free-format stream with only ONE frame (and no trailing sync)
    // cannot have its frame size measured. The demuxer should reject
    // open() rather than silently emit a malformed packet.
    let file = build_free_format_frame();
    let mut reg = ContainerRegistry::new();
    oxideav_mp3::register_containers(&mut reg);
    let cursor: Box<dyn oxideav_core::ReadSeek> = Box::new(Cursor::new(file));
    // No second sync → measure_free_format_size returns Unsupported.
    let res = reg.open_demuxer("mp3", cursor, &oxideav_core::NullCodecResolver);
    let err = match res {
        Ok(_) => panic!("expected open failure on un-measurable free-format stream"),
        Err(e) => e,
    };
    let msg = format!("{err:?}");
    assert!(
        msg.contains("free-format") || msg.contains("Unsupported"),
        "expected free-format / Unsupported error, got: {msg}"
    );
}
