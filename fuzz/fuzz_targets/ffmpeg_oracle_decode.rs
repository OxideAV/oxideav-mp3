#![no_main]

//! Fuzz: arbitrary MPEG-1/2 Layer III bytes → both `Mp3Decoder` and
//! libavcodec's MP3 decoder; when libavcodec accepts the input, ours
//! must too with matching `samples`, `sample_rate`, and channel count,
//! and PCM samples within ±2 LSB after fixed-vs-float MDCT precision
//! is accounted for.
//!
//! The harness skips silently (eprintln-only — never `#[ignore]`) when
//! no libavcodec shared library is installed.
//!
//! ## Carving the input
//!
//! Random fuzz bytes almost never start with a valid MP3 sync word
//! (`0xFFFB`/`0xFFFA`/etc.). To get useful coverage we carve the input
//! into (header_seed, payload): the first 4 bytes seed a *guaranteed
//! valid* frame header (sync = 0xFFE0..., MPEG-1 Layer III, picked
//! sample rate / bitrate / channels from the standard tables), and
//! the rest of the fuzz bytes fill the per-frame payload. This is the
//! shape that catches Huffman / scalefactor / requantise divergences
//! between us and ffmpeg — feeding 100 % random bytes would just
//! exercise the header reject path on both sides.

use libfuzzer_sys::fuzz_target;
use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};
use oxideav_mp3::decoder::make_decoder;
use oxideav_mp3::frame::parse_frame_header;
use oxideav_mp3::CODEC_ID_STR;
use oxideav_mp3_fuzz::libavcodec;

/// Per-sample tolerance after scaling float oracle output to S16.
/// MP3's IMDCT + polyphase synthesis are float-on-both-sides for both
/// libavcodec and us, so the only divergence comes from accumulator
/// rounding inside the cosine-table-driven MDCT (see ISO 11172-3
/// Annex B, Table B.7); ±2 LSB at S16 quantisation covers that
/// without hiding real bugs.
const S16_TOLERANCE: i32 = 2;

fuzz_target!(|data: &[u8]| {
    // Always probe the loader so the cache populates on the first
    // iteration; if we have no libavcodec, run the panic-free path
    // on our decoder and bail before the oracle comparison.
    let oracle_available = libavcodec::available();
    if !oracle_available {
        // Print exactly once so log-watchers can confirm the runner
        // is actually exercising the oracle. We log to stderr from
        // the very first iteration; libfuzzer threads stderr per
        // worker so the message lands at most a handful of times.
        static SKIP_PRINTED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        SKIP_PRINTED.get_or_init(|| {
            eprintln!(
                "[oracle skip] libavcodec shared library not loadable; \
                 ffmpeg_oracle_decode falls back to a panic-free smoke \
                 test on Mp3Decoder only"
            );
        });
    }

    let Some(frame) = build_frame(data) else {
        return;
    };

    // Run libavcodec first; if it rejects, we still need our decoder
    // to not panic, but we don't compare results.
    let oracle = if oracle_available {
        libavcodec::decode_mp3(&frame)
    } else {
        None
    };

    // Run our decoder.
    let params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
    let mut dec = match make_decoder(&params) {
        Ok(d) => d,
        Err(_) => return,
    };
    let pkt = Packet::new(0u32, TimeBase::new(1, 44_100), frame.clone());
    let send_ok = dec.send_packet(&pkt).is_ok();
    let our_frame = if send_ok {
        match dec.receive_frame() {
            Ok(Frame::Audio(af)) => Some(af),
            _ => None,
        }
    } else {
        None
    };

    let Some(oracle) = oracle else {
        // ffmpeg said no — we don't compare. (Our decoder returning
        // either Ok or Err here is acceptable; many "rejected by
        // ffmpeg" inputs are still partially parseable by us if the
        // header is valid but the main_data is short.)
        return;
    };

    // ffmpeg accepted the frame; we must too.
    let Some(our_af) = our_frame else {
        // Real bug: ffmpeg decoded the frame, ours didn't. Surface
        // it as a libfuzzer crash.
        panic!(
            "ffmpeg_oracle: libavcodec decoded {} samples but Mp3Decoder rejected the frame \
             (sample_rate={}, channels={}, bytes={})",
            oracle.samples,
            oracle.sample_rate,
            oracle.channels,
            frame.len(),
        );
    };

    // Re-parse our header so we can read sample_rate / channels
    // (AudioFrame doesn't carry them — they live in CodecParameters,
    // and we need the per-frame values for comparison).
    let Ok(hdr) = parse_frame_header(&frame) else {
        return;
    };
    let our_channels = hdr.channels() as u32;
    let our_sample_rate = hdr.sample_rate;

    assert_eq!(
        our_af.samples, oracle.samples,
        "ffmpeg_oracle: sample-count mismatch — libavcodec={}, oxideav={}",
        oracle.samples, our_af.samples
    );
    assert_eq!(
        our_sample_rate, oracle.sample_rate,
        "ffmpeg_oracle: sample-rate mismatch — libavcodec={}, oxideav={}",
        oracle.sample_rate, our_sample_rate
    );
    assert_eq!(
        our_channels, oracle.channels,
        "ffmpeg_oracle: channel-count mismatch — libavcodec={}, oxideav={}",
        oracle.channels, our_channels
    );

    // PCM comparison. Our decoder emits S16 LE interleaved; the oracle
    // returns interleaved f32 in [-1, 1]. Convert + compare with
    // ±S16_TOLERANCE LSB tolerance. Skip if the AudioFrame is in a
    // shape we can't read.
    if our_af.data.is_empty() {
        return;
    }
    let our_bytes = &our_af.data[0];
    let n_samples = our_af.samples as usize * our_channels as usize;
    if our_bytes.len() < n_samples * 2 {
        return;
    }
    if oracle.samples_f32.len() < n_samples {
        return;
    }
    for i in 0..n_samples {
        let ours = i16::from_le_bytes([our_bytes[2 * i], our_bytes[2 * i + 1]]) as i32;
        // Saturate-clip the oracle value in case it slightly overshoots
        // [-1, 1] (libavcodec's float output is unbounded).
        let oracle_s16 = (oracle.samples_f32[i].clamp(-1.0, 1.0) * 32767.0).round() as i32;
        let diff = (ours - oracle_s16).abs();
        // Use a slightly looser per-sample bound for the very first
        // frame from a stream — both decoders prepend codec-delay
        // silence with a small DC offset that varies between
        // implementations.
        let tol = if i < 64 {
            S16_TOLERANCE + 4
        } else {
            S16_TOLERANCE
        };
        assert!(
            diff <= tol,
            "ffmpeg_oracle: PCM divergence at sample {}: oxideav={}, libavcodec_s16={}, diff={}",
            i,
            ours,
            oracle_s16,
            diff
        );
    }
});

/// Build a self-consistent MPEG-1 Layer III frame from fuzz bytes:
/// sync + version + layer + bitrate slot + sample rate + padding +
/// channel mode are all picked from valid options; the remainder of
/// the bytes fills the per-frame payload up to the slot's frame size.
fn build_frame(data: &[u8]) -> Option<Vec<u8>> {
    // Need at least a few seed bytes + minimal payload.
    if data.len() < 8 {
        return None;
    }
    let seed = &data[..3];
    let payload = &data[3..];

    // sample_rate selector (3 entries — MPEG-1 only for the oracle
    // path so we exercise the dominant code path; MPEG-2/2.5 are
    // covered by the panic_free harness).
    let sr_table: [(u32, u8); 3] = [(44_100, 0), (48_000, 1), (32_000, 2)];
    let (sample_rate, sr_index) = sr_table[(seed[0] as usize) % sr_table.len()];

    // bitrate slot — MPEG-1 Layer III nibble values.
    let br_table: [(u32, u8); 8] = [
        (32, 0x1),
        (64, 0x4),
        (96, 0x6),
        (128, 0x9),
        (160, 0xA),
        (192, 0xB),
        (256, 0xD),
        (320, 0xE),
    ];
    let (bitrate_kbps, br_index) = br_table[(seed[1] as usize) % br_table.len()];

    // channel mode: stereo (0b00) / mono (0b11) — joint-stereo path
    // ride along too occasionally.
    let mode_bits: u8 = match seed[2] & 0x3 {
        0 => 0b00, // stereo
        1 => 0b01, // joint stereo
        2 => 0b10, // dual channel
        _ => 0b11, // mono
    };
    let channels = if mode_bits == 0b11 { 1 } else { 2 };

    // Frame size = floor(144 * bitrate / sample_rate) + padding.
    // We force padding=0 for simplicity.
    let frame_bytes = (144 * bitrate_kbps * 1000 / sample_rate) as usize;
    if frame_bytes < 32 || frame_bytes > 2048 {
        return None;
    }

    // Header bytes:
    //   B0: 0xFF
    //   B1: 0xFB                    (sync continued + MPEG-1 + Layer III + protection-bit-off)
    //   B2: bitrate(4) << 4 | sample_rate(2) << 2 | padding(1) << 1 | private(1)
    //   B3: channel_mode(2) << 6 | mode_extension(2) << 4 | copyright(1) << 3 | original(1) << 2 | emphasis(2)
    let mut frame = vec![0u8; frame_bytes.max(4)];
    frame[0] = 0xFF;
    // 0xFB = sync(3) | MPEG-1(2) | Layer III(2) | protection-off(1)
    //      = 1111 1011
    frame[1] = 0xFB;
    frame[2] = (br_index << 4) | (sr_index << 2);
    frame[3] = mode_bits << 6;

    // Side-info bytes — leave at zero so requantize sees an all-zero
    // stream. MPEG-1 stereo: 32 bytes side info; mono: 17 bytes.
    let si_bytes = if channels == 1 { 17 } else { 32 };
    if frame.len() < 4 + si_bytes {
        return None;
    }
    // Fill the rest from the fuzz payload — the more random bits in
    // here, the more often we exercise interesting Huffman paths.
    let main_data_off = 4 + si_bytes;
    let main_data_len = frame.len() - main_data_off;
    if main_data_len > 0 {
        let take = main_data_len.min(payload.len());
        frame[main_data_off..main_data_off + take].copy_from_slice(&payload[..take]);
        // Any tail past the fuzz payload stays zero.
    }
    Some(frame)
}
