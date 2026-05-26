//! End-to-end integration tests for the Xing / Info VBR
//! information-frame **emit** path on the top-level [`Mp3Encoder`].
//!
//! The encoder is given an opt-in Xing template via
//! [`Mp3Encoder::enable_xing_info`]; the resulting byte stream is read
//! back through (a) the crate's own
//! [`oxideav_mp3::demuxer::parse_xing_info`] applied to the first
//! frame, and (b) the crate's own [`oxideav_mp3::Mp3Demuxer`] open
//! path which performs the same detection at first-frame discovery
//! time. Both should recover the same fields the encoder wrote.

use std::io::Cursor;

use oxideav_mp3::{
    demuxer::{parse_xing_info, side_info_len, Mp3Demuxer, XingTagId},
    frame::{parse_header, ChannelMode, FrameWalker, Layer},
    xing_flag_bit, Mp3Encoder, XingTagSpec,
};

const SR: u32 = 44_100;
const BITRATE_KBPS: u32 = 128;
const SAMPLES_PER_FRAME: usize = 1152;
const BASE_FRAME_LEN_128_44: usize = 417; // 144 * 128_000 / 44_100 = 417 (unpadded).

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

/// Build a fresh CBR mono encoder with Xing emission enabled.
fn fresh_encoder_with_xing(template: XingTagSpec) -> Mp3Encoder {
    let mut enc = Mp3Encoder::new(BITRATE_KBPS, SR, ChannelMode::SingleChannel).unwrap();
    enc.enable_xing_info(template);
    enc
}

#[test]
fn carrier_frame_is_first_in_stream_and_carries_xing_magic() {
    // Encode 5 audio frames worth of PCM with an "Xing"-tagged
    // template that flags only frames + bytes (encoder fills them in).
    let n = SAMPLES_PER_FRAME * 5;
    let pcm = sine_pcm(n, 440.0, SR as f32, 0.5);
    let mut enc = fresh_encoder_with_xing(XingTagSpec {
        id: XingTagId::Xing,
        flags: xing_flag_bit::FRAMES | xing_flag_bit::BYTES,
        frames: None,
        bytes: None,
        toc: None,
        quality: None,
    });
    enc.push_samples(&pcm).unwrap();
    let mut bytes: Vec<u8> = Vec::new();
    enc.finish(&mut bytes).unwrap();

    // The first 4 bytes must be a Layer III header.
    let hdr = parse_header(&bytes[..4]).expect("first frame header");
    assert_eq!(hdr.layer, Layer::LayerIII);
    assert_eq!(hdr.bitrate_kbps, Some(BITRATE_KBPS));
    assert_eq!(hdr.sample_rate_hz, SR);
    // Carrier is unpadded → exactly 417 bytes.
    assert_eq!(hdr.frame_len(), Some(BASE_FRAME_LEN_128_44));

    // The Xing magic must sit right after the side-info bytes.
    let si_bytes = side_info_len(hdr.version, hdr.channel_count());
    let magic_offset = 4 + si_bytes;
    assert_eq!(&bytes[magic_offset..magic_offset + 4], b"Xing");
}

#[test]
fn carrier_frame_parse_xing_info_recovers_writer_intent() {
    let n = SAMPLES_PER_FRAME * 7;
    let pcm = sine_pcm(n, 440.0, SR as f32, 0.5);
    let mut enc = fresh_encoder_with_xing(XingTagSpec {
        id: XingTagId::Info,
        flags: xing_flag_bit::FRAMES | xing_flag_bit::BYTES,
        frames: None,
        bytes: None,
        toc: None,
        quality: None,
    });
    enc.push_samples(&pcm).unwrap();
    let mut bytes: Vec<u8> = Vec::new();
    enc.finish(&mut bytes).unwrap();

    // Walk frames, grab the first one's full payload.
    let first = FrameWalker::new(&bytes).next().expect("first frame");
    let hdr = parse_header(&first.data[..4]).unwrap();
    let si_bytes = side_info_len(hdr.version, hdr.channel_count());
    let tag = parse_xing_info(first.data, si_bytes).expect("xing tag present");

    assert_eq!(tag.id, XingTagId::Info);
    // Flag bits exactly the two we set on the template.
    assert_eq!(tag.flags, xing_flag_bit::FRAMES | xing_flag_bit::BYTES);
    // The encoder filled in frames / bytes from the post-encode totals.
    // 7 audio frames followed the carrier; bytes is the on-wire byte
    // count of those 7 audio frames (header + side_info + slot each).
    let audio_frame_count = tag.frames.expect("frames field");
    let audio_byte_count = tag.bytes.expect("bytes field");
    assert!(audio_frame_count >= 7);
    // Reasonable lower bound: 7 base frames at 417 bytes each.
    assert!(
        audio_byte_count as usize >= 7 * BASE_FRAME_LEN_128_44,
        "bytes field {audio_byte_count} below 7 base frames"
    );
    // And no toc / quality fields (we didn't flag them).
    assert!(tag.toc.is_none());
    assert!(tag.quality.is_none());
}

#[test]
fn xing_audio_total_bytes_matches_actual_audio_region_length() {
    // The `bytes` field the encoder fills in must equal the on-wire
    // byte length of every audio frame that follows the carrier —
    // not just a count of N base-length frames (some frames may carry
    // the CBR padding byte). Verify the encoder's accounting agrees
    // with byte-counting via FrameWalker.
    let n = SAMPLES_PER_FRAME * 20; // ~520 ms — pad cycle long enough to land padded frames.
    let pcm = sine_pcm(n, 440.0, SR as f32, 0.5);
    let mut enc = fresh_encoder_with_xing(XingTagSpec {
        id: XingTagId::Info,
        flags: xing_flag_bit::FRAMES | xing_flag_bit::BYTES,
        frames: None,
        bytes: None,
        toc: None,
        quality: None,
    });
    enc.push_samples(&pcm).unwrap();
    let mut bytes: Vec<u8> = Vec::new();
    enc.finish(&mut bytes).unwrap();

    // First frame is the Xing carrier. Walk the rest and sum bytes.
    let mut walker = FrameWalker::new(&bytes);
    let carrier = walker.next().expect("carrier frame");
    assert_eq!(carrier.data.len(), BASE_FRAME_LEN_128_44);
    let mut audio_bytes: u64 = 0;
    let mut audio_frames: u32 = 0;
    for f in walker {
        audio_bytes += f.data.len() as u64;
        audio_frames += 1;
    }

    let hdr = parse_header(&carrier.data[..4]).unwrap();
    let si_bytes = side_info_len(hdr.version, hdr.channel_count());
    let tag = parse_xing_info(carrier.data, si_bytes).expect("xing tag");
    assert_eq!(
        tag.frames.unwrap(),
        audio_frames,
        "frames field {} != walker count {}",
        tag.frames.unwrap(),
        audio_frames
    );
    assert_eq!(
        tag.bytes.unwrap() as u64,
        audio_bytes,
        "bytes field {} != audio-region byte count {}",
        tag.bytes.unwrap(),
        audio_bytes
    );
}

#[test]
fn pre_filled_xing_template_fields_are_written_verbatim() {
    // Template carries explicit frames + bytes; the encoder must NOT
    // overwrite them with post-encode totals.
    let pcm = sine_pcm(SAMPLES_PER_FRAME * 3, 440.0, SR as f32, 0.5);
    let toc_in: [u8; 100] = std::array::from_fn(|i| i as u8);
    let mut enc = fresh_encoder_with_xing(XingTagSpec {
        id: XingTagId::Xing,
        flags: xing_flag_bit::ALL_FOUR,
        frames: Some(0xDEAD),
        bytes: Some(0xBEEF),
        toc: Some(toc_in),
        quality: Some(50),
    });
    enc.push_samples(&pcm).unwrap();
    let mut bytes: Vec<u8> = Vec::new();
    enc.finish(&mut bytes).unwrap();

    let first = FrameWalker::new(&bytes).next().unwrap();
    let hdr = parse_header(&first.data[..4]).unwrap();
    let si_bytes = side_info_len(hdr.version, hdr.channel_count());
    let tag = parse_xing_info(first.data, si_bytes).unwrap();
    assert_eq!(tag.frames, Some(0xDEAD));
    assert_eq!(tag.bytes, Some(0xBEEF));
    assert_eq!(tag.toc.unwrap()[..], toc_in[..]);
    assert_eq!(tag.quality, Some(50));
}

#[test]
fn no_xing_template_emits_audio_only_bytestream() {
    // Without `enable_xing_info`, the output must match the prior
    // behaviour: first frame is audio, no Xing magic anywhere in the
    // leading frame's main-data slot.
    let pcm = sine_pcm(SAMPLES_PER_FRAME * 4, 440.0, SR as f32, 0.5);
    let mut enc = Mp3Encoder::new(BITRATE_KBPS, SR, ChannelMode::SingleChannel).unwrap();
    enc.push_samples(&pcm).unwrap();
    let mut bytes: Vec<u8> = Vec::new();
    enc.finish(&mut bytes).unwrap();

    let first = FrameWalker::new(&bytes).next().unwrap();
    let hdr = parse_header(&first.data[..4]).unwrap();
    let si_bytes = side_info_len(hdr.version, hdr.channel_count());
    let tag = parse_xing_info(first.data, si_bytes);
    assert!(
        tag.is_none(),
        "unexpected Xing tag in unflagged-encoder output"
    );
}

#[test]
fn mp3_demuxer_consumes_carrier_frame_at_open() {
    // The crate's own `Mp3Demuxer::open` reads the first MPEG audio
    // frame, detects the Xing/Info tag immediately after the
    // side-info bytes, and skips the carrier (next_packet returns
    // audio frames). Verify the demuxer reports the same Xing tag the
    // encoder wrote.
    let n = SAMPLES_PER_FRAME * 6;
    let pcm = sine_pcm(n, 440.0, SR as f32, 0.5);
    let mut enc = fresh_encoder_with_xing(XingTagSpec {
        id: XingTagId::Info,
        flags: xing_flag_bit::FRAMES | xing_flag_bit::BYTES,
        frames: None,
        bytes: None,
        toc: None,
        quality: None,
    });
    enc.push_samples(&pcm).unwrap();
    let mut bytes: Vec<u8> = Vec::new();
    enc.finish(&mut bytes).unwrap();

    // Open through `Mp3Demuxer::open` so we can inspect the `xing()`
    // accessor; the demuxer's first-frame discovery detects and skips
    // the Xing/Info tag exactly as it would for any real file.
    let cursor: Box<dyn oxideav_core::ReadSeek> = Box::new(Cursor::new(bytes));
    let demux = Mp3Demuxer::open(cursor).expect("Mp3Demuxer::open");
    let tag = demux.xing().expect("demuxer detected Xing tag");
    assert_eq!(tag.id, XingTagId::Info);
    assert_eq!(tag.flags, xing_flag_bit::FRAMES | xing_flag_bit::BYTES);
    assert!(tag.frames.is_some());
    assert!(tag.bytes.is_some());
}

#[test]
fn carrier_decode_to_silence_does_not_leak_into_audio() {
    // A black-box conformance check: an MP3 decoder that ignores the
    // Xing tag still gets a valid silent leading frame, so the audio
    // begins at the SECOND frame as far as a tag-blind decoder is
    // concerned. The crate's own decode chain does NOT skip the
    // carrier — that's the demuxer's job — but if a caller decodes
    // the carrier frame's main_data directly through the decode chain,
    // every granule's `part2_3_length` reads as 0, so no spectral data
    // is consumed and the output is silence.
    let pcm = sine_pcm(SAMPLES_PER_FRAME * 3, 440.0, SR as f32, 0.5);
    let mut enc = fresh_encoder_with_xing(XingTagSpec {
        id: XingTagId::Xing,
        flags: xing_flag_bit::FRAMES,
        frames: None,
        bytes: None,
        toc: None,
        quality: None,
    });
    enc.push_samples(&pcm).unwrap();
    let mut bytes: Vec<u8> = Vec::new();
    enc.finish(&mut bytes).unwrap();
    let first = FrameWalker::new(&bytes).next().unwrap();
    let hdr = parse_header(&first.data[..4]).unwrap();
    let si_bytes = side_info_len(hdr.version, hdr.channel_count());
    let si = oxideav_mp3::parse_side_info(&hdr, &first.data[4..4 + si_bytes]).unwrap();
    for gr in 0..si.granule_count as usize {
        for ch in 0..si.channels as usize {
            assert_eq!(
                si.granules[gr][ch].part2_3_length, 0,
                "carrier frame side-info granule {gr}/{ch} has non-zero part2_3_length"
            );
            assert_eq!(si.granules[gr][ch].big_values, 0);
        }
    }
}
