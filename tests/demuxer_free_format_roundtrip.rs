//! End-to-end **free-format** (`bitrate_index == 0`) integration test:
//! demux a free-format MP3 stream through [`Mp3Demuxer`] and decode the
//! emitted packets through the registered `oxideav_core::Decoder`,
//! checking the result is byte-exact with the same stream decoded as
//! CBR.
//!
//! ## Why this is distinct from `decoder_trait_free_format_roundtrip`
//!
//! That sibling test slices frames with `FrameWalker` and forces the
//! `bitrate_index` to `0` per packet — it exercises the *decoder*'s
//! free-format length handling but never opens the bytes through the
//! container demuxer. This test drives the **demuxer** path: it builds a
//! contiguous on-disk free-format byte stream and relies on
//! [`Mp3Demuxer`] to (a) accept a free-format first frame, (b) measure
//! the constant unpadded frame length from the distance between the
//! first two syncs (ISO/IEC 11172-3 §2.4.1.3 fixes the free-format
//! bitrate, so each frame is that length plus its own padding slot), and
//! (c) walk every frame to EOF. The packets it yields are then decoded.
//!
//! ## Construction
//!
//! The crate's encoder only emits table bitrates. To get a genuine
//! free-format stream we encode CBR, then rewrite each frame's 4-bit
//! `bitrate_index` (the high nibble of header byte 2) to `0`, leaving
//! every other byte — including the padding bit and the whole side-info
//! and main-data slot — untouched. A CBR stream already has a *constant*
//! frame body for a fixed bitrate, exactly the free-format invariant, so
//! the rewritten stream is a valid free-format stream whose frames
//! decode to precisely the CBR PCM.

use oxideav_core::{
    CodecId, CodecParameters, Decoder, Demuxer, Error, Frame, Packet, RuntimeContext, SampleFormat,
};
use oxideav_mp3::{parse_header, ChannelMode, Mp3Demuxer, Mp3Encoder};

use std::f32::consts::PI;
use std::io::Cursor;

fn sine_pcm(n: usize, freq_hz: f32, sr: f32, amp: f32) -> Vec<i16> {
    let two_pi = 2.0 * PI;
    let scale = amp * (i16::MAX as f32);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f32 / sr;
        let s = (two_pi * freq_hz * t).sin() * scale;
        out.push(s.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16);
    }
    out
}

fn encode_cbr(pcm: &[i16], sample_rate: u32, bitrate_kbps: u32, mode: ChannelMode) -> Vec<u8> {
    let mut enc = Mp3Encoder::new(bitrate_kbps, sample_rate, mode).expect("Mp3Encoder build");
    enc.push_samples(pcm).expect("push_samples");
    let mut bytes: Vec<u8> = Vec::new();
    enc.finish(&mut bytes).expect("finish");
    bytes
}

/// Rewrite **every** frame's `bitrate_index` to `0` (free format) in a
/// contiguous CBR byte stream, in place, walking frame by frame using
/// the header-derivable CBR length. Returns the rewritten stream.
fn whole_stream_to_free_format(cbr: &[u8]) -> Vec<u8> {
    let mut out = cbr.to_vec();
    let mut pos = 0usize;
    while pos + 4 <= out.len() {
        // Find a sync at pos.
        if out[pos] != 0xFF || (out[pos + 1] & 0xE0) != 0xE0 {
            pos += 1;
            continue;
        }
        let hdr = match parse_header(&out[pos..pos + 4]) {
            Ok(h) => h,
            Err(_) => {
                pos += 1;
                continue;
            }
        };
        let len = match hdr.frame_len() {
            Some(l) => l,
            None => {
                pos += 1;
                continue;
            }
        };
        if pos + len > out.len() {
            break;
        }
        // Clear the bitrate_index nibble (high nibble of byte 2).
        out[pos + 2] &= 0x0F;
        pos += len;
    }
    out
}

fn make_decoder(sample_rate: u32, channels: u16) -> Box<dyn Decoder> {
    let mut ctx = RuntimeContext::default();
    oxideav_mp3::register(&mut ctx);
    let mut params = CodecParameters::audio(CodecId::new("mp3"));
    params.sample_rate = Some(sample_rate);
    params.channels = Some(channels);
    params.sample_format = Some(SampleFormat::S16);
    ctx.codecs
        .first_decoder(&params)
        .expect("decoder factory present after register()")
}

fn decode_packets(packets: &[Packet], sample_rate: u32, channels: u16) -> Vec<i16> {
    let mut dec = make_decoder(sample_rate, channels);
    let mut out: Vec<i16> = Vec::new();
    for pkt in packets {
        dec.send_packet(pkt).expect("send_packet");
        loop {
            match dec.receive_frame() {
                Ok(Frame::Audio(a)) => {
                    for plane in &a.data {
                        for chunk in plane.chunks_exact(2) {
                            out.push(i16::from_le_bytes([chunk[0], chunk[1]]));
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

/// Collect every packet a demuxer yields until EOF.
fn drain_demuxer(mut d: Mp3Demuxer) -> Vec<Packet> {
    let mut out = Vec::new();
    loop {
        match d.next_packet() {
            Ok(p) => out.push(p),
            Err(Error::Eof) => break,
            Err(e) => panic!("demuxer next_packet: {e}"),
        }
    }
    out
}

/// The demuxer accepts a free-format mono stream, measures the constant
/// frame length, and walks every frame; decoding the result is
/// byte-exact with the CBR original.
#[test]
fn demux_free_format_mono_byte_exact_against_cbr() {
    const SR: u32 = 44_100;
    let n = SR as usize / 2; // 500 ms
    let pcm = sine_pcm(n, 440.0, SR as f32, 0.45);
    let cbr = encode_cbr(&pcm, SR, 128, ChannelMode::SingleChannel);

    // Reference packets straight off the CBR demuxer.
    let cbr_d = Mp3Demuxer::open(Box::new(Cursor::new(cbr.clone()))).expect("open CBR");
    assert_eq!(cbr_d.streams()[0].params.sample_rate, Some(SR));
    let cbr_pkts = drain_demuxer(cbr_d);
    assert!(cbr_pkts.len() > 4, "need several frames");
    let cbr_pcm = decode_packets(&cbr_pkts, SR, 1);
    assert!(!cbr_pcm.is_empty());

    // Free-format stream: same bytes, bitrate_index forced to 0.
    let ff = whole_stream_to_free_format(&cbr);
    let ff_d = Mp3Demuxer::open(Box::new(Cursor::new(ff))).expect("open free-format");
    let ff_pkts = drain_demuxer(ff_d);

    // The demuxer must recover exactly the same frame boundaries: same
    // packet count and, frame by frame, the same payload length and PTS.
    assert_eq!(
        ff_pkts.len(),
        cbr_pkts.len(),
        "free-format demux emitted a different frame count than CBR"
    );
    for (i, (ff_p, cbr_p)) in ff_pkts.iter().zip(cbr_pkts.iter()).enumerate() {
        assert_eq!(
            ff_p.data.len(),
            cbr_p.data.len(),
            "frame {i}: free-format length differs from CBR"
        );
        assert_eq!(ff_p.pts, cbr_p.pts, "frame {i}: PTS differs");
        // Every emitted free-format packet must genuinely be free format.
        let h = parse_header(&ff_p.data[..4]).expect("ff header");
        assert!(h.is_free_format(), "frame {i} not free format");
    }

    let ff_pcm = decode_packets(&ff_pkts, SR, 1);
    assert_eq!(ff_pcm.len(), cbr_pcm.len(), "sample count mismatch");
    assert_eq!(
        ff_pcm, cbr_pcm,
        "free-format demux+decode must be byte-exact with CBR"
    );
}

/// Same end-to-end check for a stereo stream, exercising the per-channel
/// decode state under the demuxer's free-format length path.
#[test]
fn demux_free_format_stereo_byte_exact_against_cbr() {
    const SR: u32 = 44_100;
    let n = SR as usize / 4; // 250 ms
    let left = sine_pcm(n, 440.0, SR as f32, 0.4);
    let right = sine_pcm(n, 660.0, SR as f32, 0.4);
    let mut inter = Vec::with_capacity(n * 2);
    for i in 0..n {
        inter.push(left[i]);
        inter.push(right[i]);
    }
    let cbr = encode_cbr(&inter, SR, 192, ChannelMode::Stereo);

    let cbr_d = Mp3Demuxer::open(Box::new(Cursor::new(cbr.clone()))).expect("open CBR stereo");
    let cbr_pkts = drain_demuxer(cbr_d);
    assert!(cbr_pkts.len() > 2);
    let cbr_pcm = decode_packets(&cbr_pkts, SR, 2);

    let ff = whole_stream_to_free_format(&cbr);
    let ff_d = Mp3Demuxer::open(Box::new(Cursor::new(ff))).expect("open free-format stereo");
    let ff_pkts = drain_demuxer(ff_d);

    assert_eq!(ff_pkts.len(), cbr_pkts.len(), "stereo frame count mismatch");
    let ff_pcm = decode_packets(&ff_pkts, SR, 2);
    assert_eq!(ff_pcm.len(), cbr_pcm.len(), "stereo sample count mismatch");
    assert_eq!(
        ff_pcm, cbr_pcm,
        "stereo free-format demux+decode must be byte-exact with CBR"
    );
}

/// The demuxer reports a derived constant bitrate and a finite duration
/// for a free-format stream (it has no table bitrate, so both come from
/// the measured frame length, not the header).
#[test]
fn demux_free_format_reports_derived_bitrate_and_duration() {
    const SR: u32 = 48_000; // no 44.1k padding churn; every frame is one length
    let n = SR as usize / 2;
    let pcm = sine_pcm(n, 330.0, SR as f32, 0.4);
    let cbr = encode_cbr(&pcm, SR, 128, ChannelMode::SingleChannel);
    let ff = whole_stream_to_free_format(&cbr);

    let d = Mp3Demuxer::open(Box::new(Cursor::new(ff))).expect("open free-format 48k");
    let info = &d.streams()[0];
    assert_eq!(info.params.sample_rate, Some(SR));
    // 128 kbit/s at 48 kHz: 144 * 128000 / 48000 = 384-byte frames, all
    // unpadded, so the derived rate recovers ~128 kbit/s.
    let br = info.params.bit_rate.expect("free-format derived bitrate");
    assert_eq!(br, 384 * 8 * SR as u64 / 1152);
    assert_eq!(br, 128_000);
    let dur = info.duration.expect("free-format duration");
    assert!(dur > 0, "duration must be positive");
}
