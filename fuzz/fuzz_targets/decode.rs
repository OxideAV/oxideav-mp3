#![no_main]

//! Drive attacker-supplied bytes through `Decoder::send_packet` on a
//! fresh MPEG-1 / MPEG-2 LSF Audio Layer III decoder.
//!
//! Round 289 depth-mode lane: a panic-freedom fuzzer over the Layer
//! III decoder's attacker surface. The contract under test is purely
//! that every public entry point on the registered `Decoder` trait
//! object returns a `Result` (never panics, never integer-overflows
//! in a debug build, never indexes out of bounds) on arbitrary input,
//! across a multi-packet stream so the cross-packet bit-reservoir
//! state machine is exercised — not just a single cold-start frame.
//!
//! ## Why structurally-valid headers
//!
//! A free-form byte fuzzer almost never produces the 11-bit frame
//! sync (ISO/IEC 11172-3 §2.4.2.3), so it spends all its energy on the
//! `BadSync` rejection and never reaches side-info parse, Huffman
//! decode, requantisation, alias reduction, IMDCT, or the synthesis
//! filterbank. This target instead *constructs* a valid 4-byte header
//! whose every field (version, layer, bitrate index, sample-rate
//! index, channel mode, mode extension, CRC flag, padding) is
//! attacker-chosen, then fills the remainder of the
//! header-implied frame length with attacker bytes for the optional
//! CRC slot, the side-info block, and the main-data slot. The deep
//! decode chain is therefore reached on essentially every iteration,
//! with the bytes past the header fully attacker-controlled.
//!
//! ## Cross-packet state
//!
//! Layer III carries part of a frame's main data in *previous* frames'
//! slots (the bit reservoir, §2.4.3.4). `main_data_begin` in the
//! side-info points backwards into the reservoir; the IMDCT and the
//! polyphase synthesis filterbank both carry overlap state across
//! granules. A single cold-start packet can never reach the
//! reservoir-satisfied branch where a frame actually decodes to PCM,
//! nor the overlap carry-over. This target stitches several crafted
//! frames into one decoder session so those transitions are reachable,
//! and injects a mid-stream `reset()` + `flush()` so the
//! state-machine transitions through both calls.
//!
//! ## Surfaces driven on every iteration
//!
//! 1. `oxideav_mp3::register_codecs` → `make_decoder` factory surface.
//! 2. `Decoder::send_packet` across a fuzzer-chosen sequence of
//!    crafted frames (valid header + attacker body), plus raw-byte
//!    packets so the `< 4 bytes` and short-frame rejections are hit.
//! 3. `Decoder::receive_frame` after each `send_packet` to drain
//!    `pending_frames`.
//! 4. `Decoder::reset` + `Decoder::flush` mid-stream, then
//!    `receive_frame` until `Eof`.
//!
//! Output PCM is discarded — the fuzzer only cares about *return*,
//! never sample-correctness (that is the integration tests' job).
//!
//! Spec basis: ISO/IEC 11172-3 §2.4.1.7 (side information),
//! §2.4.2.3 (frame header / sync), §2.4.2.7 (Huffman code-bit
//! decoding), §2.4.3.4 (requantisation + filterbank).

use libfuzzer_sys::fuzz_target;
use oxideav_core::{CodecId, CodecParameters, CodecRegistry, Decoder, Packet, Rational, TimeBase};

const SAMPLE_RATE: u32 = 44_100;
const MAX_PACKETS_PER_ITER: usize = 12;
/// Bound the crafted-frame body so a malicious bitrate/sample-rate
/// combination can't pin a worker on a multi-kilobyte allocation per
/// frame. The largest legal MPEG-1 Layer III frame is ~1440 bytes
/// (320 kbit/s @ 32 kHz); this cap covers it with margin.
const MAX_FRAME_BYTES: usize = 2048;

/// Assemble a structurally-valid 4-byte Layer III header from
/// attacker-chosen field bits, returning the four header bytes.
///
/// The field choices are constrained to the *accepted* ranges so the
/// header actually parses (the decoder only reaches the deep chain on
/// a parsing header); within those ranges every field is
/// attacker-controlled.
///
/// `v` selects MPEG version, `sr` the sample-rate index, `br` the
/// bitrate index (1..=14, never 0 free-format or 15 forbidden),
/// `mode` the channel mode, and the remaining control bits drive the
/// CRC, padding, mode-extension, and private/copyright/original
/// flags. Layer is fixed to III (the decoder rejects I/II).
fn build_header(v: u8, sr: u8, br: u8, mode: u8, ext: u8, flags: u8) -> [u8; 4] {
    // 11-bit sync (bits 31..21 all ones).
    let mut raw: u32 = 0x7FF << 21;
    // Version: 0b11 MPEG-1, 0b10 MPEG-2, 0b00 MPEG-2.5 (0b01 reserved).
    let ver_bits = match v % 3 {
        0 => 0b11u32,
        1 => 0b10,
        _ => 0b00,
    };
    raw |= ver_bits << 19;
    // Layer III = 0b01.
    raw |= 0b01u32 << 17;
    // protection_bit: '0' = CRC present.
    let crc = flags & 0b0000_0001;
    raw |= u32::from(crc ^ 1) << 16; // store '0' when CRC present
                                     // bitrate index 1..=14 (avoid 0 free-format and 15 forbidden).
    let br_idx = (br % 14) + 1;
    raw |= u32::from(br_idx) << 12;
    // sample-rate index 0..=2 (3 is reserved/rejected).
    let sr_idx = sr % 3;
    raw |= u32::from(sr_idx) << 10;
    // padding bit.
    raw |= u32::from((flags >> 1) & 1) << 9;
    // private bit.
    raw |= u32::from((flags >> 2) & 1) << 8;
    // mode 0..=3.
    raw |= u32::from(mode & 0b11) << 6;
    // mode extension 0..=3.
    raw |= u32::from(ext & 0b11) << 4;
    // copyright / original / emphasis (emphasis 0b00 = none).
    raw |= u32::from((flags >> 3) & 1) << 3;
    raw |= u32::from((flags >> 4) & 1) << 2;
    raw.to_be_bytes()
}

fn build_decoder() -> Option<Box<dyn Decoder>> {
    let mut reg = CodecRegistry::new();
    oxideav_mp3::register_codecs(&mut reg);
    let id = CodecId::new(oxideav_mp3::demuxer::CODEC_ID_STR);
    let mut params = CodecParameters::audio(id);
    params.channels = Some(2);
    params.sample_rate = Some(SAMPLE_RATE);
    reg.first_decoder(&params).ok()
}

fuzz_target!(|data: &[u8]| {
    // A 0/1-byte buffer still exercises the `< 4 bytes` rejection on a
    // cold decoder.
    let mut dec = match build_decoder() {
        Some(d) => d,
        None => return,
    };
    let tb = TimeBase(Rational::new(1, SAMPLE_RATE as i64));

    if data.is_empty() {
        let pkt = Packet::new(0, tb, Vec::new());
        let _ = dec.send_packet(&pkt);
        return;
    }

    let packet_count = ((data[0] as usize) % MAX_PACKETS_PER_ITER) + 1;
    // Deterministic mid-stream reset / flush hooks (derived from
    // data[0] so libfuzzer's minimiser can still shrink reliably).
    let reset_at = if packet_count >= 3 {
        (data[0] as usize) % packet_count
    } else {
        usize::MAX
    };
    let flush_at = if packet_count >= 2 {
        ((data[0] >> 4) as usize) % packet_count
    } else {
        usize::MAX
    };

    let mut cursor = 1usize;
    for i in 0..packet_count {
        if cursor >= data.len() {
            break;
        }
        // One control byte selects how this packet is shaped.
        let ctl = data[cursor];
        cursor += 1;

        let body: Vec<u8> = if ctl & 0b1000_0000 != 0 {
            // Raw-byte packet: a short, attacker-chosen slice fed
            // verbatim so the `< 4 bytes` / `BadSync` / short-frame
            // rejections are reached with no constructed header.
            let want = (ctl as usize & 0x3f).min(data.len().saturating_sub(cursor));
            let slice = data[cursor..cursor + want].to_vec();
            cursor += want;
            slice
        } else {
            // Crafted frame: valid header drawn from the next 6 bytes,
            // body filled from the bytes after.
            let need_for_header = 6usize;
            if data.len() - cursor < need_for_header {
                // Not enough to craft a header; feed the tail raw.
                let slice = data[cursor..].to_vec();
                cursor = data.len();
                slice
            } else {
                let v = data[cursor];
                let sr = data[cursor + 1];
                let br = data[cursor + 2];
                let mode = data[cursor + 3];
                let ext = data[cursor + 4];
                let flags = data[cursor + 5];
                cursor += need_for_header;
                let hdr = build_header(v, sr, br, mode, ext, flags);

                // Re-derive the frame length the decoder will compute so
                // the body is the right size to satisfy its length check
                // (otherwise the short-frame rejection fires first and
                // the deep chain is never reached). Fall back to a fixed
                // length on the rare parse miss.
                let frame_len = oxideav_mp3::frame::parse_header(&hdr)
                    .ok()
                    .and_then(|h| h.frame_len())
                    .unwrap_or(417)
                    .min(MAX_FRAME_BYTES);
                let mut frame = vec![0u8; frame_len];
                frame[..4].copy_from_slice(&hdr);
                // Fill the rest from attacker bytes, walking the buffer.
                for (j, slot) in frame.iter_mut().enumerate().skip(4) {
                    if let Some(&b) = data.get(cursor + ((j - 4) % data.len().max(1))) {
                        *slot = b;
                    }
                }
                // Advance the cursor by a stride so successive frames
                // draw fresh body bytes.
                cursor = (cursor + 16).min(data.len());
                frame
            }
        };

        let pkt = Packet::new(0, tb, body);
        // send_packet may legitimately Err on a truncated tail,
        // unsupported header (MPEG-2.5 / free-format), or short frame —
        // that's the contract. Output frames are drained immediately to
        // keep `pending_frames` bounded.
        if dec.send_packet(&pkt).is_ok() {
            let _ = dec.receive_frame();
        }

        if i == flush_at {
            let _ = dec.flush();
            for _ in 0..4 {
                if dec.receive_frame().is_err() {
                    break;
                }
            }
        }
        if i == reset_at {
            let _ = dec.reset();
        }
    }

    // Final flush + drain, regardless of where the loop exited. A
    // forced double-flush exercises the idempotent post-flush `Eof`
    // path.
    let _ = dec.flush();
    let _ = dec.flush();
    for _ in 0..4 {
        if dec.receive_frame().is_err() {
            break;
        }
    }
});
