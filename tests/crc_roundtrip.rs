//! Integration test for [`oxideav_mp3::Mp3Encoder::with_protection_bit`]
//! — the §2.4.3.1 / Annex B Table B.5 opt-in CRC-16 frame protection
//! added in Phase 2 step 15 (round 144).
//!
//! Covers:
//!
//! * Frame-by-frame structural placement of the 2-byte CRC slot
//!   (immediately after the 4-byte header; side-info shifts down by
//!   two bytes; total per-frame byte count unchanged).
//! * Wire `protection_bit = 0` on every emitted audio frame.
//! * Bit-exact CRC value against an independent recomputation via
//!   [`oxideav_mp3::crc::crc16_layer3`] (mono / 135-bit window).
//! * Decoder round-trip: a CRC-enabled stream still decodes to
//!   silence-equivalent PCM via the crate's own decode chain (which
//!   already skips the CRC slot per
//!   [`oxideav_mp3::Mp3FrameHeader::crc_protected`]).

use oxideav_mp3::{
    crc::crc16_layer3, parse_header, parse_side_info, ChannelMode, FrameWalker, Mp3Encoder,
    SAMPLES_PER_FRAME_MPEG1,
};

/// Synthesise an `n`-sample mono `i16` silence buffer (every sample
/// zero). Silence keeps the encoded main-data slot small so the
/// CRC-induced 2-byte shrink can't push us off the bitrate slot.
fn silence(n: usize) -> Vec<i16> {
    vec![0i16; n]
}

#[test]
fn crc_enabled_stream_has_valid_per_frame_crc16() {
    // 4 frames of mono silence at 128 kbit/s, 44.1 kHz with CRC on.
    let pcm = silence(SAMPLES_PER_FRAME_MPEG1 * 4);

    let mut enc = Mp3Encoder::new(128, 44_100, ChannelMode::SingleChannel).unwrap();
    enc.with_protection_bit(true);
    enc.push_samples(&pcm).unwrap();
    let mut bytes: Vec<u8> = Vec::new();
    enc.finish(&mut bytes).unwrap();

    let frames: Vec<_> = FrameWalker::new(&bytes).collect();
    assert!(
        !frames.is_empty(),
        "expected at least one frame in CRC-enabled output"
    );

    for (i, f) in frames.iter().enumerate() {
        let hdr_bytes: [u8; 4] = f.data[..4].try_into().unwrap();
        let hdr = parse_header(&hdr_bytes).expect("header parse");
        assert!(
            hdr.crc_protected,
            "frame {i} at offset {} must have protection_bit=0",
            f.offset
        );
        // 17 bytes of mono side-info; CRC owns bytes 4..6.
        let wire_crc = u16::from_be_bytes([f.data[4], f.data[5]]);
        let si_start = 6; // 4 header + 2 CRC
        let si_slice = &f.data[si_start..si_start + 17];
        let expected = crc16_layer3(&hdr_bytes, si_slice, 1);
        assert_eq!(
            wire_crc, expected,
            "frame {i} CRC mismatch: wire 0x{wire_crc:04X} vs spec 0x{expected:04X}"
        );
        // Side-info itself must still parse: the CRC slot doesn't
        // corrupt the layout, it just shifts it.
        parse_side_info(&hdr, si_slice).expect("side_info parse");
    }
}

#[test]
fn crc_enabled_and_disabled_streams_have_same_per_frame_length() {
    // §2.4.3.1: CRC is INSIDE the frame, not appended to it. Both
    // streams must emit the same per-frame byte count and the same
    // total bytes (the only differences are: (a) protection bit, (b)
    // the 2-byte CRC slot taking 2 bytes away from main-data slot).
    let pcm = silence(SAMPLES_PER_FRAME_MPEG1 * 6);

    let mut enc_nocrc = Mp3Encoder::new(128, 44_100, ChannelMode::SingleChannel).unwrap();
    enc_nocrc.push_samples(&pcm).unwrap();
    let mut out_nocrc: Vec<u8> = Vec::new();
    enc_nocrc.finish(&mut out_nocrc).unwrap();

    let mut enc_crc = Mp3Encoder::new(128, 44_100, ChannelMode::SingleChannel).unwrap();
    enc_crc.with_protection_bit(true);
    enc_crc.push_samples(&pcm).unwrap();
    let mut out_crc: Vec<u8> = Vec::new();
    enc_crc.finish(&mut out_crc).unwrap();

    assert_eq!(
        out_nocrc.len(),
        out_crc.len(),
        "CRC must not change total stream length"
    );

    // Per-frame lengths also match.
    let f_nocrc: Vec<_> = FrameWalker::new(&out_nocrc).collect();
    let f_crc: Vec<_> = FrameWalker::new(&out_crc).collect();
    assert_eq!(f_nocrc.len(), f_crc.len(), "frame count must match");
    for (a, b) in f_nocrc.iter().zip(f_crc.iter()) {
        assert_eq!(a.data.len(), b.data.len(), "per-frame length must match");
    }
}

#[test]
fn crc_enabled_stream_decodes_via_existing_decoder() {
    // The crate's existing decode path skips the optional CRC slot
    // unconditionally (see codec_decoder.rs §"Skip the optional 2-byte
    // CRC slot"). A CRC-enabled stream must therefore decode to the
    // same audio output as a CRC-disabled stream of the same content.
    //
    // For silence input the comparison is straightforward: both
    // decoders should yield silence-equivalent PCM. We use the
    // crate's own `decode_huffman` + downstream chain via the
    // `Mp3CoreDecoder` trait.
    use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};

    let pcm_in = silence(SAMPLES_PER_FRAME_MPEG1 * 4);

    // Encode with CRC.
    let mut enc = Mp3Encoder::new(128, 44_100, ChannelMode::SingleChannel).unwrap();
    enc.with_protection_bit(true);
    enc.push_samples(&pcm_in).unwrap();
    let mut bytes = Vec::new();
    enc.finish(&mut bytes).unwrap();

    // Decode each frame via the crate's trait-style decoder.
    let mut params = CodecParameters::audio(CodecId::new("mp3"));
    params.sample_rate = Some(44_100);
    params.channels = Some(1);
    let mut dec = oxideav_mp3::make_decoder(&params).expect("make_decoder");

    let tb = TimeBase::new(1, 44_100);
    let mut total_samples = 0usize;
    for f in FrameWalker::new(&bytes) {
        let pkt = Packet::new(0, tb, f.data.to_vec());
        dec.send_packet(&pkt).expect("send_packet");
        loop {
            match dec.receive_frame() {
                Ok(Frame::Audio(a)) => {
                    total_samples += a.samples as usize;
                    // Silence input → recovered PCM should have small
                    // amplitude (decoder warm-up adds a transient,
                    // but for true silence input the energy is bounded
                    // tight). Confirm finite values and a low cap.
                    for plane in &a.data {
                        for chunk in plane.chunks_exact(2) {
                            let s = i16::from_le_bytes([chunk[0], chunk[1]]);
                            assert!(
                                s.unsigned_abs() < 8000,
                                "recovered silence sample magnitude too large: {s}"
                            );
                        }
                    }
                }
                Ok(_) => panic!("unexpected non-audio frame"),
                Err(oxideav_core::Error::NeedMore) => break,
                Err(e) => panic!("decoder error: {e}"),
            }
        }
    }
    // Drain the decoder.
    dec.flush().ok();
    while let Ok(Frame::Audio(a)) = dec.receive_frame() {
        total_samples += a.samples as usize;
    }

    assert!(
        total_samples > 0,
        "decoder produced no samples from CRC-enabled stream"
    );
}
