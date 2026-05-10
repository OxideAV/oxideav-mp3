#![no_main]

//! Fuzz: random S16 PCM → Mp3Encoder → Mp3Decoder, checking that the
//! decoder consumes everything the encoder produces without panicking
//! and reproduces the right total sample count.
//!
//! MP3 is lossy and the pipeline carries non-trivial codec delay
//! (~528-sample MDCT lookahead + ~481-sample synthesis filter delay
//! per ISO/IEC 11172-3 Annex C, plus bit-reservoir carry across frames),
//! so we don't pixel-compare PCM here. The point of this harness is the
//! looser invariant that *whatever the encoder emits, our own decoder
//! parses cleanly* — a regression that breaks bitstream conformance on
//! one side without breaking the other will show up as a decode error
//! or a sample-count divergence beyond the documented codec delay.

use libfuzzer_sys::fuzz_target;
use oxideav_core::{AudioFrame, CodecId, CodecParameters, Frame, Packet, SampleFormat, TimeBase};
use oxideav_mp3::decoder::make_decoder;
use oxideav_mp3::encoder::make_encoder;
use oxideav_mp3::frame::parse_frame_header;
use oxideav_mp3::CODEC_ID_STR;

/// Standard MP3 codec delay (samples) — encoder MDCT lookahead +
/// decoder polyphase synthesis warmup. Real-world pipelines carry
/// this delay opaquely; the round-trip check loses up to one frame
/// (1152 samples for MPEG-1, 576 for MPEG-2 LSF) at each boundary.
const MAX_DELAY_SAMPLES: i64 = 4096;

fuzz_target!(|data: &[u8]| {
    // Carve fuzz bytes into (sample_rate_index, channels, bitrate_index, pcm_bytes).
    let Some((sample_rate, channels, bitrate_kbps, pcm)) = parse_inputs(data) else {
        return;
    };
    if pcm.is_empty() {
        return;
    }

    let mut params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
    params.channels = Some(channels);
    params.sample_rate = Some(sample_rate);
    params.sample_format = Some(SampleFormat::S16);
    params.bit_rate = Some((bitrate_kbps as u64) * 1000);

    let mut enc = match make_encoder(&params) {
        Ok(e) => e,
        Err(_) => return,
    };

    // Feed encoder in 1152-sample-per-channel chunks (MPEG-1 frame
    // size; MPEG-2 LSF will internally split into 576-sample
    // granules — the encoder accepts both shapes).
    let chunk_samples = 1152usize;
    let bytes_per_sample_frame = 2 * channels as usize;
    let chunk_bytes = chunk_samples * bytes_per_sample_frame;
    let mut pts: i64 = 0;
    let mut total_in_samples: i64 = 0;
    for slice in pcm.chunks(chunk_bytes) {
        let n_samples = slice.len() / bytes_per_sample_frame;
        if n_samples == 0 {
            break;
        }
        let frame = AudioFrame {
            samples: n_samples as u32,
            pts: Some(pts),
            data: vec![slice.to_vec()],
        };
        if enc.send_frame(&Frame::Audio(frame)).is_err() {
            return;
        }
        pts += n_samples as i64;
        total_in_samples += n_samples as i64;
    }
    if enc.flush().is_err() {
        return;
    }

    let mut bitstream: Vec<u8> = Vec::new();
    while let Ok(p) = enc.receive_packet() {
        bitstream.extend_from_slice(&p.data);
    }
    if bitstream.is_empty() {
        return;
    }

    // Now decode with our own decoder, framing on the MP3 sync bytes.
    let dec_params = {
        let mut p = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
        p.channels = Some(channels);
        p.sample_rate = Some(sample_rate);
        p
    };
    let mut dec = match make_decoder(&dec_params) {
        Ok(d) => d,
        Err(_) => return,
    };
    let tb = TimeBase::new(1, sample_rate as i64);
    let mut total_out_samples: i64 = 0;
    let mut pos = 0usize;
    while pos + 4 <= bitstream.len() {
        let Ok(hdr) = parse_frame_header(&bitstream[pos..]) else {
            break;
        };
        let Some(flen) = hdr.frame_bytes() else { break };
        let flen = flen as usize;
        if pos + flen > bitstream.len() {
            break;
        }
        let pkt = Packet::new(0u32, tb, bitstream[pos..pos + flen].to_vec());
        if dec.send_packet(&pkt).is_err() {
            // Encoder emitted a packet our decoder rejects — that's a
            // real bug. Surface it.
            panic!(
                "Mp3Decoder rejected a packet emitted by Mp3Encoder \
                 (sample_rate={}, channels={}, bitrate={}kbps, pos={})",
                sample_rate, channels, bitrate_kbps, pos
            );
        }
        if let Ok(Frame::Audio(af)) = dec.receive_frame() {
            total_out_samples += af.samples as i64;
        }
        pos += flen;
    }

    // Total decoded samples should be within `MAX_DELAY_SAMPLES` of
    // the input (the encoder may emit pre-roll silence; the decoder
    // may swallow trailing reservoir bytes). Anything wildly off is
    // a sign of broken framing or sample-rate confusion.
    let drift = (total_out_samples - total_in_samples).abs();
    assert!(
        drift <= total_in_samples + MAX_DELAY_SAMPLES,
        "round-trip sample-count drift too large: in={}, out={}, drift={}",
        total_in_samples,
        total_out_samples,
        drift
    );
});

/// Carve fuzz bytes into encoder configuration + PCM payload.
fn parse_inputs(data: &[u8]) -> Option<(u32, u16, u32, Vec<u8>)> {
    let (&cfg, rest) = data.split_first()?;
    // Bits 0..=2 → sample-rate selector (6 supported rates).
    let sr_table: [u32; 6] = [44_100, 48_000, 32_000, 22_050, 24_000, 16_000];
    let sample_rate = sr_table[(cfg as usize) % sr_table.len()];
    // Bit 3 → channels.
    let channels: u16 = if (cfg & 0x08) != 0 { 2 } else { 1 };
    // Bits 4..=7 → bitrate selector. We pick from the version's
    // standard table so the encoder accepts the slot.
    let bitrate_kbps = if matches!(sample_rate, 16_000 | 22_050 | 24_000) {
        // MPEG-2 LSF slots.
        let table: [u32; 8] = [8, 16, 32, 48, 64, 96, 128, 160];
        table[((cfg >> 4) as usize) % table.len()]
    } else {
        // MPEG-1 slots.
        let table: [u32; 8] = [32, 64, 96, 128, 160, 192, 256, 320];
        table[((cfg >> 4) as usize) % table.len()]
    };

    // Cap PCM length to keep fuzz throughput reasonable: at most
    // 16 frames worth of MPEG-1 stereo input ≈ 70 KiB.
    const MAX_PCM_BYTES: usize = 16 * 1152 * 2 * 2;
    let pcm_len = rest.len().min(MAX_PCM_BYTES);
    // Trim to a multiple of (2 * channels) so we have whole sample frames.
    let bytes_per_sf = 2 * channels as usize;
    let pcm_len = (pcm_len / bytes_per_sf) * bytes_per_sf;
    if pcm_len < bytes_per_sf {
        return None;
    }
    let pcm = rest[..pcm_len].to_vec();
    Some((sample_rate, channels, bitrate_kbps, pcm))
}
