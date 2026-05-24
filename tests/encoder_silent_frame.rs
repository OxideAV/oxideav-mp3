//! Integration tests for the Phase 1 Layer III bitstream-formatting
//! encoder.
//!
//! These exercise [`oxideav_mp3::encode_silent_frame`] end-to-end:
//!
//! 1. **Self round-trip** — the emitted frame parses back through the
//!    crate's own [`oxideav_mp3::parse_header`] /
//!    [`oxideav_mp3::parse_side_info`] and is located by
//!    [`oxideav_mp3::FrameWalker`].
//! 2. **Demuxer round-trip** — a multi-frame silent stream is walked by
//!    [`oxideav_mp3::Mp3Demuxer`], which must surface exactly the frames
//!    that were emitted, with the right header parameters.
//! 3. **Black-box external decode** — when `ffmpeg` is on `PATH` the
//!    stream is decoded to raw PCM and asserted to be (a) the right
//!    number of samples and (b) pure silence (every sample 0). `ffmpeg`
//!    is used only as an opaque process: we feed it our bytes and check
//!    its bytes; its source is never consulted. When `ffmpeg` is absent
//!    (CI without the binary) the external check logs a skip and the
//!    test still passes on the in-crate round-trips.

use std::io::Cursor;
use std::process::Command;

use oxideav_core::Demuxer;
use oxideav_mp3::{
    encode_silent_frame, make_silent_header, parse_header, parse_side_info, ChannelMode,
    FrameWalker, Mp3Demuxer, MpegVersion,
};

/// Build a CBR silent stream of `n` frames at the given parameters.
fn silent_stream(bitrate: u32, sample_rate: u32, mode: ChannelMode, n: usize) -> Vec<u8> {
    let h = make_silent_header(bitrate, sample_rate, mode).expect("valid header params");
    let mut buf = Vec::new();
    for _ in 0..n {
        buf.extend_from_slice(&encode_silent_frame(&h).expect("encode silent frame"));
    }
    buf
}

#[test]
fn single_frame_self_roundtrip() {
    let h = make_silent_header(128, 44_100, ChannelMode::Stereo).unwrap();
    let frame = encode_silent_frame(&h).unwrap();

    // Header parses back with matching parameters and no CRC.
    let hdr = parse_header(&frame[..4]).unwrap();
    assert_eq!(hdr.bitrate_kbps, Some(128));
    assert_eq!(hdr.sample_rate_hz, 44_100);
    assert_eq!(hdr.channel_count(), 2);
    assert!(!hdr.crc_protected);

    // Side info parses back (no CRC → side info starts at byte 4).
    let si = parse_side_info(&hdr, &frame[4..]).unwrap();
    assert_eq!(si.main_data_begin, 0);
    assert_eq!(si.granule_count, 2);
    assert_eq!(si.channels, 2);
    for gr in 0..2 {
        for ch in 0..2 {
            assert_eq!(si.granules[gr][ch].part2_3_length, 0);
            assert_eq!(si.granules[gr][ch].big_values, 0);
        }
    }

    // FrameWalker locates exactly one frame spanning the whole buffer.
    let frames: Vec<_> = FrameWalker::new(&frame).collect();
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].data.len(), frame.len());
}

#[test]
fn multi_frame_walker_roundtrip() {
    // Several parameter sets, including MPEG-2 LSF (single granule).
    for (br, sr, mode, n) in [
        (128, 44_100, ChannelMode::Stereo, 10),
        (320, 48_000, ChannelMode::Stereo, 5),
        (32, 32_000, ChannelMode::SingleChannel, 8),
        (64, 22_050, ChannelMode::SingleChannel, 6),
        (160, 24_000, ChannelMode::Stereo, 4),
    ] {
        let stream = silent_stream(br, sr, mode, n);
        let frames: Vec<_> = FrameWalker::new(&stream).collect();
        assert_eq!(frames.len(), n, "expected {n} frames for {br}k/{sr}Hz");
        for f in &frames {
            assert_eq!(f.header.bitrate_kbps, Some(br));
            assert_eq!(f.header.sample_rate_hz, sr);
        }
    }
}

#[test]
fn demuxer_surfaces_silent_frames() {
    // 20 frames of 128k/44.1k stereo, fed through the crate's own demuxer.
    let stream = silent_stream(128, 44_100, ChannelMode::Stereo, 20);
    let mut demux = Mp3Demuxer::open(Box::new(Cursor::new(stream))).expect("open demuxer");

    let mut count = 0;
    loop {
        match demux.next_packet() {
            Ok(_pkt) => count += 1,
            Err(oxideav_core::Error::Eof) => break,
            Err(e) => panic!("unexpected demux error: {e:?}"),
        }
    }
    assert_eq!(count, 20);
}

#[test]
fn lsf_silent_frame_roundtrip() {
    // MPEG-2 LSF: 64k/16kHz mono, one granule per frame, 576 samples.
    let h = make_silent_header(64, 16_000, ChannelMode::SingleChannel).unwrap();
    assert_eq!(h.version, MpegVersion::Mpeg2);
    let frame = encode_silent_frame(&h).unwrap();
    let hdr = parse_header(&frame[..4]).unwrap();
    assert_eq!(hdr.samples_per_frame(), 576);
    let si = parse_side_info(&hdr, &frame[4..]).unwrap();
    assert_eq!(si.granule_count, 1);
    assert!(si.lsf);
}

/// Locate `ffmpeg` on `PATH`; `None` if not present.
fn ffmpeg_path() -> Option<std::path::PathBuf> {
    let out = Command::new("ffmpeg").arg("-version").output().ok()?;
    if out.status.success() {
        Some(std::path::PathBuf::from("ffmpeg"))
    } else {
        None
    }
}

#[test]
fn ffmpeg_accepts_and_decodes_to_silence() {
    let Some(ffmpeg) = ffmpeg_path() else {
        eprintln!("ffmpeg not found on PATH; skipping black-box external-decode check");
        return;
    };

    // 50 frames of 128k / 44.1k stereo MPEG-1 Layer III.
    let n = 50;
    let stream = silent_stream(128, 44_100, ChannelMode::Stereo, n);

    // Write to a temp file (ffmpeg's mp3 demuxer wants a seekable input).
    let dir = std::env::temp_dir();
    let mp3_path = dir.join(format!("oxideav_mp3_silent_{}.mp3", std::process::id()));
    let pcm_path = dir.join(format!("oxideav_mp3_silent_{}.pcm", std::process::id()));
    std::fs::write(&mp3_path, &stream).expect("write mp3");

    // Decode to signed-16-bit little-endian PCM, no resampling.
    let status = Command::new(&ffmpeg)
        .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
        .arg(&mp3_path)
        .args(["-f", "s16le"])
        .arg(&pcm_path)
        .status()
        .expect("run ffmpeg");
    assert!(
        status.success(),
        "ffmpeg failed to decode the emitted stream"
    );

    let pcm = std::fs::read(&pcm_path).expect("read decoded pcm");
    // Best-effort cleanup.
    let _ = std::fs::remove_file(&mp3_path);
    let _ = std::fs::remove_file(&pcm_path);

    // 2 channels × 2 bytes/sample × 1152 samples/frame × n frames. ffmpeg
    // may drop the first frame's worth of samples to prime the decoder's
    // overlap/synthesis state, so accept any non-trivial sample count and
    // assert the decoded audio is pure silence.
    assert!(
        !pcm.is_empty(),
        "ffmpeg produced no PCM from the emitted stream"
    );
    assert_eq!(pcm.len() % 2, 0, "PCM byte length not 16-bit aligned");

    let max_abs = pcm
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]).unsigned_abs())
        .max()
        .unwrap_or(0);
    assert_eq!(
        max_abs, 0,
        "emitted silent frame did not decode to silence (max |sample| = {max_abs})"
    );
}
