//! Integration test for **free-format** (`bitrate_index == 0`) Layer III
//! decode through the registered `oxideav_core::Decoder` trait wiring.
//!
//! ## What "free format" means (ISO/IEC 11172-3 §2.4.2.3)
//!
//! A normal MPEG audio frame advertises a `bitrate_index ∈ 1..=14` in
//! its 4-byte header, and the §2.4.2.3 formula derives a fixed byte
//! length from that index + the sample rate + padding. A **free-format**
//! frame sets `bitrate_index == 0`: the encoder chose a constant frame
//! length the bitstream itself does not encode, and the framing layer
//! recovers it as the distance from one syncword to the next. The header
//! therefore yields *no* derivable length (`frame_len()` returns `None`).
//!
//! The trait contract is that each inbound `Packet.data` holds exactly
//! one complete MP3 frame (header + optional CRC + side-info + main-data
//! slot), so the authoritative free-format length is simply the packet
//! length. The whole downstream decode (side-info → Huffman →
//! requantize → IMDCT → synthesis) is driven by `part2_3_length` from the
//! side-info, never by the advertised bitrate, so a free-format frame
//! decodes through the identical chain once we know where its main-data
//! slot ends.
//!
//! ## How this test builds a free-format stream
//!
//! The crate's own encoder always emits a fixed `bitrate_index`. To
//! exercise the free-format path without a foreign fixture, we encode a
//! CBR stream, slice it into per-frame packets, and rewrite each frame's
//! 4-bit `bitrate_index` header field to `0` **without changing any
//! other byte**. The resulting packet is bit-for-bit identical in its
//! side-info and main-data slot, so a correct free-format decode MUST
//! produce exactly the PCM the original CBR frame produced — the
//! bitrate field is purely advisory for framing and is ignored by the
//! sample-recovery math. That byte-exact equality is the success
//! criterion.
//!
//! The `bitrate_index` occupies bits 12..16 of the big-endian header
//! word, i.e. the **high nibble of header byte 2**. Clearing that nibble
//! sets free format; the forbidden value `0b1111` is never produced by
//! our encoder, so there is no risk of accidentally hitting it.

use oxideav_core::{
    CodecId, CodecParameters, Decoder, Error, Frame, Packet, RuntimeContext, SampleFormat, TimeBase,
};
use oxideav_mp3::{parse_header, ChannelMode, FrameWalker, Mp3Encoder, SAMPLES_PER_FRAME_MPEG1};

use std::f32::consts::PI;

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

fn encode_to_mp3(pcm: &[i16], sample_rate: u32, bitrate_kbps: u32) -> Vec<u8> {
    let mut enc = Mp3Encoder::new(bitrate_kbps, sample_rate, ChannelMode::SingleChannel)
        .expect("Mp3Encoder build");
    enc.push_samples(pcm).expect("push_samples");
    let mut bytes: Vec<u8> = Vec::new();
    enc.finish(&mut bytes).expect("finish");
    bytes
}

fn make_decoder_via_registry(sample_rate: u32, channels: u16) -> Box<dyn Decoder> {
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

/// Slice a CBR byte stream into one `Packet` per MP3 frame, each
/// carrying exactly that frame's bytes (as the demuxer / framer would).
fn cbr_packets(bytes: &[u8], sample_rate: u32) -> Vec<Packet> {
    let tb = TimeBase::new(1, i64::from(sample_rate));
    let mut out = Vec::new();
    let mut pts: i64 = 0;
    for f in FrameWalker::new(bytes) {
        let mut pkt = Packet::new(0, tb, f.data.to_vec());
        pkt.pts = Some(pts);
        pkt.duration = Some(SAMPLES_PER_FRAME_MPEG1 as i64);
        out.push(pkt);
        pts += SAMPLES_PER_FRAME_MPEG1 as i64;
    }
    out
}

/// Rewrite a single MP3 frame's 4-bit `bitrate_index` to `0`
/// (free format), changing nothing else. The field lives in the high
/// nibble of header byte 2.
fn to_free_format(frame: &[u8]) -> Vec<u8> {
    let mut out = frame.to_vec();
    out[2] &= 0x0F;
    out
}

/// Drive a packet stream through the registered trait decoder and
/// flatten all returned audio into a single interleaved i16 vector.
fn decode_via_trait(packets: &[Packet], sample_rate: u32, channels: u16) -> Vec<i16> {
    let mut dec = make_decoder_via_registry(sample_rate, channels);
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
                Ok(other) => panic!("non-audio frame from MP3 decoder: {other:?}"),
                Err(Error::NeedMore) => break,
                Err(e) => panic!("receive_frame: {e}"),
            }
        }
    }
    dec.flush().expect("flush");
    out
}

/// A `bitrate_index == 0` header parses as free format and yields no
/// derivable frame length — the precondition the new decode path relies
/// on.
#[test]
fn free_format_header_has_no_derivable_length() {
    const SR: u32 = 44_100;
    let pcm = sine_pcm(SR as usize / 8, 440.0, SR as f32, 0.5);
    let wire = encode_to_mp3(&pcm, SR, 128);
    let first = FrameWalker::new(&wire).next().expect("at least one frame");

    // Sanity: the CBR frame DOES have a derivable length.
    let cbr_hdr = parse_header(&first.data[..4]).expect("cbr header");
    assert!(
        cbr_hdr.frame_len().is_some(),
        "CBR frame should have a header-derivable length"
    );
    assert_eq!(cbr_hdr.bitrate_index, 9, "128 kbps V1 L3 == index 9");

    // After flipping the bitrate field, the header is free format.
    let ff = to_free_format(first.data);
    let ff_hdr = parse_header(&ff[..4]).expect("free-format header parses");
    assert_eq!(ff_hdr.bitrate_index, 0, "free format");
    assert!(ff_hdr.is_free_format(), "is_free_format() true");
    assert!(
        ff_hdr.frame_len().is_none(),
        "free-format header must not yield a derivable frame length"
    );
    // Everything else about the header is unchanged.
    assert_eq!(ff_hdr.version, cbr_hdr.version);
    assert_eq!(ff_hdr.layer, cbr_hdr.layer);
    assert_eq!(ff_hdr.sample_rate_hz, cbr_hdr.sample_rate_hz);
    assert_eq!(ff_hdr.channel_count(), cbr_hdr.channel_count());
}

/// The headline check: a free-format frame stream decodes to exactly the
/// PCM the equivalent CBR frame stream decodes to, because the bitrate
/// field never participates in sample recovery — only the framing.
#[test]
fn free_format_decode_byte_exact_against_cbr() {
    const SR: u32 = 44_100;
    let n = SR as usize / 2; // 500 ms
    let pcm = sine_pcm(n, 440.0, SR as f32, 0.45);
    let wire = encode_to_mp3(&pcm, SR, 128);

    // Reference: decode the unmodified CBR stream through the trait.
    let cbr_pkts = cbr_packets(&wire, SR);
    assert!(cbr_pkts.len() > 4, "need several frames to be meaningful");
    let cbr_pcm = decode_via_trait(&cbr_pkts, SR, 1);
    assert!(!cbr_pcm.is_empty(), "CBR decode produced no PCM");

    // Free-format equivalent: same packets, bitrate_index forced to 0.
    let ff_pkts: Vec<Packet> = cbr_pkts
        .iter()
        .map(|p| {
            let mut np = Packet::new(0, p.time_base, to_free_format(&p.data));
            np.pts = p.pts;
            np.duration = p.duration;
            np
        })
        .collect();
    // Confirm every packet is genuinely free format now.
    for p in &ff_pkts {
        let h = parse_header(&p.data[..4]).expect("ff header");
        assert!(h.is_free_format(), "packet not free format");
    }
    let ff_pcm = decode_via_trait(&ff_pkts, SR, 1);

    assert_eq!(
        ff_pcm.len(),
        cbr_pcm.len(),
        "free-format decode produced a different sample count than CBR"
    );
    assert_eq!(
        ff_pcm, cbr_pcm,
        "free-format decode must be byte-exact with CBR (bitrate field is framing-only)"
    );
}

/// Free-format decode also works for a stereo stream, exercising the
/// per-channel state arrays under the new length path.
#[test]
fn free_format_decode_stereo_byte_exact() {
    const SR: u32 = 44_100;
    let n = SR as usize / 4; // 250 ms
    let left = sine_pcm(n, 440.0, SR as f32, 0.4);
    let right = sine_pcm(n, 660.0, SR as f32, 0.4);
    // Interleave L/R for the stereo encoder.
    let mut inter = Vec::with_capacity(n * 2);
    for i in 0..n {
        inter.push(left[i]);
        inter.push(right[i]);
    }
    let mut enc = Mp3Encoder::new(192, SR, ChannelMode::Stereo).expect("stereo Mp3Encoder build");
    enc.push_samples(&inter).expect("push_samples");
    let mut wire: Vec<u8> = Vec::new();
    enc.finish(&mut wire).expect("finish");

    let cbr_pkts = cbr_packets(&wire, SR);
    assert!(cbr_pkts.len() > 2, "need several stereo frames");
    let cbr_pcm = decode_via_trait(&cbr_pkts, SR, 2);

    let ff_pkts: Vec<Packet> = cbr_pkts
        .iter()
        .map(|p| {
            let mut np = Packet::new(0, p.time_base, to_free_format(&p.data));
            np.pts = p.pts;
            np.duration = p.duration;
            np
        })
        .collect();
    let ff_pcm = decode_via_trait(&ff_pkts, SR, 2);

    assert_eq!(ff_pcm.len(), cbr_pcm.len(), "stereo sample count mismatch");
    assert_eq!(
        ff_pcm, cbr_pcm,
        "stereo free-format decode must be byte-exact with CBR"
    );
}

/// A free-format header with no payload past the 4-byte sync is
/// rejected (there is no main-data slot to decode), rather than
/// panicking or producing silence.
#[test]
fn free_format_bare_sync_is_rejected() {
    const SR: u32 = 44_100;
    let pcm = sine_pcm(SR as usize / 8, 440.0, SR as f32, 0.5);
    let wire = encode_to_mp3(&pcm, SR, 128);
    let first = FrameWalker::new(&wire).next().expect("a frame");
    let hdr4 = to_free_format(&first.data[..4]);

    let mut dec = make_decoder_via_registry(SR, 1);
    let pkt = Packet::new(0, TimeBase::new(1, i64::from(SR)), hdr4);
    let err = dec
        .send_packet(&pkt)
        .expect_err("bare free-format sync rejected");
    assert!(
        matches!(err, Error::InvalidData(_)),
        "expected InvalidData for a payload-less free-format frame, got {err:?}"
    );
}
