//! End-to-end integration tests for the **true-VBR** encoder path.
//!
//! The encoder is configured at a high constructor bitrate, then
//! [`Mp3Encoder::enable_vbr`] narrows the per-frame `bitrate_index`
//! choice to a `[min_kbps, max_kbps]` window. Every emitted audio
//! frame's header carries the chosen ladder index; the on-wire size of
//! that frame is `144 · bitrate / sample_rate (+ pad)` — varying across
//! the stream depending on per-frame content complexity.
//!
//! These tests validate:
//!
//! 1. Frames in a silence-heavy stream pick the **minimum** ladder
//!    index (their main-data fits in the smallest slot).
//! 2. Mixed content lands a mix of bitrate indices — the stream is
//!    truly VBR, not effectively-CBR-at-min.
//! 3. The crate's own demuxer / decoder consume the VBR stream without
//!    error, recovering one packet per audio frame.
//! 4. With Xing emission enabled and `flag_bit::TOC` set, the encoder
//!    auto-populates the seek table from per-frame cumulative offsets;
//!    the TOC's last entry approaches 255 (= 256 · last_frame_start /
//!    total_bytes ≈ 256 · (N-1)/N).
//! 5. Misconfigured VBR (min > max, off-ladder, max > ctor) errors at
//!    `enable_vbr` time, not at flush.

use std::io::Cursor;

use oxideav_core::Demuxer;
use oxideav_mp3::{
    demuxer::{parse_xing_info, side_info_len, Mp3Demuxer, XingTagId},
    frame::{parse_header, ChannelMode, FrameWalker, Layer},
    xing_flag_bit, Mp3Encoder, StreamEncodeError, XingTagSpec,
};

const SR: u32 = 44_100;
const SAMPLES_PER_FRAME: usize = 1152;

/// Mono sine i16 PCM.
fn sine_pcm(n: usize, freq_hz: f32, sr: f32, amp: f32) -> Vec<i16> {
    let two_pi = 2.0 * std::f32::consts::PI;
    let scale = amp * (i16::MAX as f32);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f32 / sr;
        let s = (two_pi * freq_hz * t).sin() * scale;
        out.push(s.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16);
    }
    out
}

/// Concatenate two PCM regions: a quiet (low-amp) tone and a louder /
/// higher-frequency tone. The two halves stress the encoder
/// differently and exercise the VBR per-frame index selection.
fn mixed_content_pcm(quiet_frames: usize, loud_frames: usize) -> Vec<i16> {
    let mut out = sine_pcm(SAMPLES_PER_FRAME * quiet_frames, 220.0, SR as f32, 0.02);
    out.extend(sine_pcm(
        SAMPLES_PER_FRAME * loud_frames,
        2_000.0,
        SR as f32,
        0.85,
    ));
    out
}

#[test]
fn enable_vbr_off_ladder_min_rejected() {
    let mut enc = Mp3Encoder::new(320, SR, ChannelMode::SingleChannel).unwrap();
    // 35 kbps is not on the §2.4.2.3 V1/L3 ladder (between 32 and 40).
    let err = enc.enable_vbr(35, 128).unwrap_err();
    assert!(matches!(err, StreamEncodeError::InvalidVbrConfig));
}

#[test]
fn enable_vbr_off_ladder_max_rejected() {
    let mut enc = Mp3Encoder::new(320, SR, ChannelMode::SingleChannel).unwrap();
    // 200 kbps is not on the ladder (192 and 224 surround it).
    let err = enc.enable_vbr(32, 200).unwrap_err();
    assert!(matches!(err, StreamEncodeError::InvalidVbrConfig));
}

#[test]
fn enable_vbr_min_greater_than_max_rejected() {
    let mut enc = Mp3Encoder::new(320, SR, ChannelMode::SingleChannel).unwrap();
    let err = enc.enable_vbr(192, 64).unwrap_err();
    assert!(matches!(err, StreamEncodeError::InvalidVbrConfig));
}

#[test]
fn enable_vbr_max_above_ctor_rejected() {
    // Constructor at 128, max 192 exceeds it → reject (the inner-loop
    // budget would not cover the chosen max).
    let mut enc = Mp3Encoder::new(128, SR, ChannelMode::SingleChannel).unwrap();
    let err = enc.enable_vbr(64, 192).unwrap_err();
    assert!(matches!(err, StreamEncodeError::InvalidVbrConfig));
}

#[test]
fn enable_vbr_min_equals_max_succeeds() {
    let mut enc = Mp3Encoder::new(320, SR, ChannelMode::SingleChannel).unwrap();
    enc.enable_vbr(128, 128).expect("min==max valid");
}

#[test]
fn vbr_silence_stream_picks_min_index() {
    // A pure-silence stream's per-frame main_data is tiny (header
    // bookkeeping + a near-zero side-info); with a VBR window of
    // [32, 320] every audio frame should land on the 32 kbps (min)
    // bucket. Anything stronger than dead silence will quickly fill
    // the quantizer's high-resolution bins because the smallest
    // `global_gain` the §C.1.5.4.4.2 magnitude-clamp accepts for
    // non-trivial input is already large enough to keep many
    // coefficients non-zero — so the "quiet but non-zero" floor for
    // landing on min-index is rather low.
    let pcm = vec![0i16; SAMPLES_PER_FRAME * 8];
    let mut enc = Mp3Encoder::new(320, SR, ChannelMode::SingleChannel).unwrap();
    enc.enable_vbr(32, 320).expect("vbr enable");
    enc.push_samples(&pcm).unwrap();
    let mut bytes: Vec<u8> = Vec::new();
    enc.finish(&mut bytes).unwrap();

    let frames: Vec<&[u8]> = FrameWalker::new(&bytes).map(|f| f.data).collect();
    assert!(
        frames.len() >= 8,
        "expected ≥8 frames, got {}",
        frames.len()
    );
    // Every audio frame's header bitrate should equal 32 kbps.
    for f in &frames {
        let hdr = parse_header(&f[..4]).expect("header parse");
        assert_eq!(hdr.layer, Layer::LayerIII);
        assert_eq!(
            hdr.bitrate_kbps,
            Some(32),
            "silent frame did not land on min-index 32 kbps"
        );
    }
}

#[test]
fn vbr_emits_mixed_bitrate_indices_for_mixed_content() {
    // A two-half stream (quiet then loud) should produce a MIX of
    // bitrate indices: the loud-frames region needs more bits to keep
    // distortion under control than the quiet region. With a window
    // of [32, 320] we expect to see >= 2 distinct bitrates in the
    // output. The constructor uses 320 so the analysis budget covers
    // every member of the window.
    let pcm = mixed_content_pcm(/*quiet=*/ 10, /*loud=*/ 10);
    let mut enc = Mp3Encoder::new(320, SR, ChannelMode::SingleChannel).unwrap();
    enc.enable_vbr(32, 320).expect("vbr enable");
    enc.push_samples(&pcm).unwrap();
    let mut bytes: Vec<u8> = Vec::new();
    enc.finish(&mut bytes).unwrap();

    let frames: Vec<&[u8]> = FrameWalker::new(&bytes).map(|f| f.data).collect();
    assert!(
        frames.len() >= 20,
        "expected ≥20 frames, got {}",
        frames.len()
    );

    // Collect unique bitrate values across the audio region.
    let mut seen: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
    for f in &frames {
        let hdr = parse_header(&f[..4]).expect("header parse");
        seen.insert(hdr.bitrate_kbps.expect("non-free format"));
    }
    assert!(
        seen.len() >= 2,
        "expected VBR to emit ≥2 distinct bitrates over mixed content; saw {:?}",
        seen
    );
}

#[test]
fn vbr_stream_walks_with_frame_walker() {
    // FrameWalker tracks per-header `frame_len`, so a stream with
    // varying per-frame bitrates must still parse cleanly: each
    // `next()` returns one whole frame and advances to the next.
    let pcm = mixed_content_pcm(5, 5);
    let mut enc = Mp3Encoder::new(320, SR, ChannelMode::SingleChannel).unwrap();
    enc.enable_vbr(64, 192).expect("vbr enable");
    enc.push_samples(&pcm).unwrap();
    let mut bytes: Vec<u8> = Vec::new();
    let written = enc.finish(&mut bytes).unwrap();
    assert_eq!(bytes.len(), written);

    let mut total = 0usize;
    let mut frame_count = 0usize;
    for f in FrameWalker::new(&bytes) {
        let hdr = parse_header(&f.data[..4]).unwrap();
        let want_len = hdr.frame_len().unwrap();
        assert_eq!(f.data.len(), want_len, "walker frame length mismatch");
        total += f.data.len();
        frame_count += 1;
    }
    assert_eq!(total, written, "walker did not consume entire stream");
    assert!(
        frame_count >= 10,
        "expected at least 10 frames, got {frame_count}"
    );
}

#[test]
fn vbr_demuxer_consumes_stream_without_error() {
    // Drive a VBR stream through the crate's own `Mp3Demuxer`. The
    // demuxer's per-frame walk doesn't assume CBR; each
    // `next_packet()` should return one packet per audio frame until
    // EOF. We don't compare PCM here — the underlying decoder chain
    // is already covered by `stream_encoder_roundtrip.rs` — we just
    // confirm the demux side accepts a varying-bitrate stream.
    let pcm = mixed_content_pcm(8, 8);
    let mut enc = Mp3Encoder::new(320, SR, ChannelMode::SingleChannel).unwrap();
    enc.enable_vbr(64, 256).expect("vbr enable");
    enc.push_samples(&pcm).unwrap();
    let mut bytes: Vec<u8> = Vec::new();
    enc.finish(&mut bytes).unwrap();

    let cursor: Box<dyn oxideav_core::ReadSeek> = Box::new(Cursor::new(bytes));
    let mut demux = Mp3Demuxer::open(cursor).expect("Mp3Demuxer::open");
    let mut pkt_count = 0usize;
    loop {
        match demux.next_packet() {
            Ok(_pkt) => pkt_count += 1,
            Err(oxideav_core::Error::Eof) => break,
            Err(e) => panic!("demuxer error on VBR stream: {e}"),
        }
    }
    assert!(
        pkt_count >= 16,
        "expected ≥16 audio packets from demuxer, got {pkt_count}"
    );
}

#[test]
fn vbr_with_xing_toc_auto_fills_seek_table() {
    // Emit a VBR stream with a Xing template that flags FRAMES |
    // BYTES | TOC, all three Option fields None. The encoder fills
    // every flagged field at finish time. The TOC must:
    //
    // * have toc[0] == 0 (start of audio region maps to byte 0);
    // * be monotonically non-decreasing (frame starts only advance);
    // * approach 255 at the tail (last frame start ≈ total_bytes -
    //   one_frame; for N=20 frames toc[99] is around 256 · 19/20 = 243).
    let pcm = mixed_content_pcm(10, 10);
    let mut enc = Mp3Encoder::new(320, SR, ChannelMode::SingleChannel).unwrap();
    enc.enable_vbr(64, 192).expect("vbr enable");
    enc.enable_xing_info(XingTagSpec {
        id: XingTagId::Xing,
        flags: xing_flag_bit::FRAMES | xing_flag_bit::BYTES | xing_flag_bit::TOC,
        frames: None,
        bytes: None,
        toc: None,
        quality: None,
    });
    enc.push_samples(&pcm).unwrap();
    let mut bytes: Vec<u8> = Vec::new();
    enc.finish(&mut bytes).unwrap();

    // First frame is the Xing carrier.
    let first = FrameWalker::new(&bytes).next().expect("carrier");
    let hdr = parse_header(&first.data[..4]).unwrap();
    let si_bytes = side_info_len(hdr.version, hdr.channel_count());
    let tag = parse_xing_info(first.data, si_bytes).expect("xing tag");
    assert_eq!(tag.id, XingTagId::Xing);
    assert_eq!(
        tag.flags,
        xing_flag_bit::FRAMES | xing_flag_bit::BYTES | xing_flag_bit::TOC
    );
    let toc = tag.toc.expect("auto-filled TOC");
    assert_eq!(toc[0], 0, "TOC[0] must map to start of audio region");
    // Monotonic non-decreasing.
    for w in toc.windows(2) {
        assert!(w[1] >= w[0], "TOC not monotone: {} > {}", w[0], w[1]);
    }
    // Tail entry is close to 255: with ≥20 audio frames the last
    // frame's start offset is ≥ (N-1)/N of total bytes, so toc[99] is
    // 256 · (N-1)/N ≈ 243 for N=20 — comfortably above 200.
    assert!(
        toc[99] >= 200,
        "TOC tail entry {} unexpectedly small for stream with {} frames",
        toc[99],
        tag.frames.unwrap()
    );
}

#[test]
fn vbr_xing_toc_bytes_field_matches_audio_region() {
    // Cross-check: the BYTES field the encoder fills in is the byte
    // total of the audio region (post-carrier). The TOC scale uses
    // the same total. Walking the post-carrier frames must agree with
    // the BYTES field exactly.
    let pcm = mixed_content_pcm(8, 8);
    let mut enc = Mp3Encoder::new(256, SR, ChannelMode::SingleChannel).unwrap();
    enc.enable_vbr(64, 192).expect("vbr enable");
    enc.enable_xing_info(XingTagSpec {
        id: XingTagId::Xing,
        flags: xing_flag_bit::FRAMES | xing_flag_bit::BYTES | xing_flag_bit::TOC,
        frames: None,
        bytes: None,
        toc: None,
        quality: None,
    });
    enc.push_samples(&pcm).unwrap();
    let mut bytes: Vec<u8> = Vec::new();
    enc.finish(&mut bytes).unwrap();

    let mut walker = FrameWalker::new(&bytes);
    let carrier = walker.next().expect("carrier");
    let hdr = parse_header(&carrier.data[..4]).unwrap();
    let si_bytes = side_info_len(hdr.version, hdr.channel_count());
    let tag = parse_xing_info(carrier.data, si_bytes).expect("xing");

    let mut audio_bytes: u64 = 0;
    let mut audio_frames: u32 = 0;
    for f in walker {
        audio_bytes += f.data.len() as u64;
        audio_frames += 1;
    }
    assert_eq!(tag.frames.unwrap(), audio_frames);
    assert_eq!(tag.bytes.unwrap() as u64, audio_bytes);
}

#[test]
fn vbr_xing_toc_seeks_via_demuxer() {
    // The demuxer's seek path uses the Xing TOC. Drive a VBR stream
    // through `Mp3Demuxer::open` and verify a TOC-based seek to
    // ~50% playback lands on a real audio-frame boundary, reports the
    // frame's *exact* PTS, and that the next packet carries the same
    // PTS and the stream stays monotone from there.
    let pcm = mixed_content_pcm(10, 10); // ~520 ms of audio.
    let mut enc = Mp3Encoder::new(320, SR, ChannelMode::SingleChannel).unwrap();
    enc.enable_vbr(64, 192).expect("vbr enable");
    enc.enable_xing_info(XingTagSpec {
        id: XingTagId::Xing,
        flags: xing_flag_bit::FRAMES | xing_flag_bit::BYTES | xing_flag_bit::TOC,
        frames: None,
        bytes: None,
        toc: None,
        quality: None,
    });
    enc.push_samples(&pcm).unwrap();
    let mut bytes: Vec<u8> = Vec::new();
    enc.finish(&mut bytes).unwrap();

    let cursor: Box<dyn oxideav_core::ReadSeek> = Box::new(Cursor::new(bytes));
    let mut demux = Mp3Demuxer::open(cursor).expect("Mp3Demuxer::open");
    let xing = demux.xing().expect("demuxer reports xing tag").clone();
    assert!(xing.toc.is_some(), "TOC missing from demuxer view");
    assert_eq!(xing.flags & xing_flag_bit::TOC, xing_flag_bit::TOC);

    let total = demux.streams()[0]
        .duration
        .expect("VBR duration from Xing frame count");
    let spf = 1152i64; // MPEG-1 Layer III samples per frame.

    // Seek to ~50% of playback.
    let landed = demux.seek_to(0, total / 2).expect("TOC seek");
    // The landed PTS is a whole-frame multiple (the demuxer snapped to
    // a real frame boundary and re-derived its exact PTS), and it sits
    // strictly inside the stream — not clamped to the first or last.
    assert_eq!(landed % spf, 0, "landed PTS not frame-aligned: {landed}");
    assert!(landed > 0, "TOC seek collapsed to the start");
    assert!(landed < total, "TOC seek overran the duration");

    // The next packet carries exactly the landed PTS and the stream
    // continues monotonically.
    let pkt = demux.next_packet().expect("packet after TOC seek");
    assert_eq!(pkt.pts, Some(landed));
    let pkt2 = demux.next_packet().expect("second packet after TOC seek");
    assert_eq!(pkt2.pts, Some(landed + spf));
}

#[test]
fn vbr_with_only_min_index_matches_cbr_at_that_index() {
    // Degenerate VBR window [128, 128] — every frame is at 128 kbps.
    // The stream should structurally match CBR at 128 kbps from the
    // user-facing perspective (per-frame bitrate constant; reservoir
    // already trivial in our zero-pad schedule).
    let pcm = sine_pcm(SAMPLES_PER_FRAME * 5, 440.0, SR as f32, 0.5);

    let mut enc_vbr = Mp3Encoder::new(128, SR, ChannelMode::SingleChannel).unwrap();
    enc_vbr.enable_vbr(128, 128).unwrap();
    enc_vbr.push_samples(&pcm).unwrap();
    let mut vbr_bytes: Vec<u8> = Vec::new();
    enc_vbr.finish(&mut vbr_bytes).unwrap();

    // CBR baseline: same constructor, no VBR enable.
    let mut enc_cbr = Mp3Encoder::new(128, SR, ChannelMode::SingleChannel).unwrap();
    enc_cbr.push_samples(&pcm).unwrap();
    let mut cbr_bytes: Vec<u8> = Vec::new();
    enc_cbr.finish(&mut cbr_bytes).unwrap();

    // Both streams should produce frames at 128 kbps. They may differ
    // by ±1 frame's padding-byte allocation since the VBR pad logic
    // only fires when `unpadded_slot < need` while CBR uses the
    // Bresenham ladder. But every frame in either stream must have
    // header bitrate == 128.
    for f in FrameWalker::new(&vbr_bytes) {
        let hdr = parse_header(&f.data[..4]).unwrap();
        assert_eq!(hdr.bitrate_kbps, Some(128));
    }
    for f in FrameWalker::new(&cbr_bytes) {
        let hdr = parse_header(&f.data[..4]).unwrap();
        assert_eq!(hdr.bitrate_kbps, Some(128));
    }
}
