#![no_main]

//! Drive attacker-supplied bytes through the [`oxideav_mp3::Mp3Demuxer`]
//! container surface: `open` (ID3v2 skip, first-frame probe, Xing /
//! Info / VBRI and LAME-tag parsing, free-format frame-length
//! measurement), `next_packet` iteration (frame walk, resync,
//! reservoir-aware keyframe flag), duration estimation, and
//! `seek_to` (Xing-TOC percentile path, CBR proportional path,
//! `pts_at_cursor` frame recount).
//!
//! r405 depth lane. The contract under test is panic-freedom: every
//! entry point returns a `Result` (never panics, never
//! integer-overflows in a debug build, never indexes out of bounds) on
//! arbitrary input.
//!
//! ## Input shaping
//!
//! Half the corpus energy would be wasted on `open()` rejecting inputs
//! with no frame sync at all, so the harness plants a structurally
//! valid attacker-parameterised first frame header (11-bit sync +
//! attacker-chosen version / layer / bitrate / rate / padding /
//! channel-mode bits) at the start of the stream body when the first
//! input byte's high bit is set; otherwise the bytes run raw (probing
//! the resync / reject paths, including a leading pseudo-ID3v2 tag
//! when the input happens to start with "ID3"). The remaining input
//! bytes are the stream body — frame payloads, subsequent headers,
//! Xing/VBRI candidates, and truncation garbage are all
//! attacker-controlled.
//!
//! ## Seek schedule
//!
//! After draining up to a bounded number of packets, the harness
//! issues `seek_to` calls derived from input bytes (front, interior
//! percentiles, far-past-EOF) and drains a few more packets after each
//! — covering the TOC snap, the proportional estimate, the resync
//! walk, and the post-seek PTS re-derivation, plus the
//! monotonicity-relevant interior state they rebuild.

use libfuzzer_sys::fuzz_target;
use oxideav_core::{Demuxer, Error};
use oxideav_mp3::Mp3Demuxer;
use std::io::Cursor;

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }
    let (ctrl, body) = data.split_at(1);
    let ctrl = ctrl[0];

    // Assemble the stream: optionally plant a valid-sync first header
    // so open() reaches the deep parse paths on most iterations.
    let mut stream = Vec::with_capacity(body.len() + 4);
    if ctrl & 0x80 != 0 && body.len() >= 3 {
        // 11-bit sync; the remaining 21 header bits come from the
        // first three body bytes (version/layer/protection from
        // byte 0 low bits, bitrate/rate/padding/private from byte 1,
        // mode/extension/copyright/original/emphasis from byte 2).
        stream.push(0xFF);
        stream.push(0xE0 | (body[0] & 0x1F));
        stream.push(body[1]);
        stream.push(body[2]);
        stream.extend_from_slice(&body[3..]);
    } else {
        stream.extend_from_slice(body);
    }

    let Ok(mut demux) = Mp3Demuxer::open(Box::new(Cursor::new(stream))) else {
        return;
    };

    // Metadata surfaces must be callable without panicking.
    let _ = demux.streams();
    let _ = demux.duration_micros();
    let _ = demux.trimmed_duration_samples();
    let _ = demux.xing();
    let _ = demux.lame();
    let _ = demux.tags();
    let _ = demux.is_vbr();

    // Bounded packet drain.
    let mut drained = 0usize;
    loop {
        match demux.next_packet() {
            Ok(_) => {
                drained += 1;
                if drained >= 64 {
                    break;
                }
            }
            Err(Error::Eof) => break,
            Err(_) => break,
        }
    }

    // Seek schedule: front, two attacker-chosen interior points, and
    // far past EOF; drain a few packets after each.
    let dur = demux.duration_micros().unwrap_or(0).max(0);
    let targets = [
        0i64,
        (dur / 4) * i64::from(ctrl & 0x03),
        i64::from(u16::from_le_bytes([
            *stream_byte(data, 1),
            *stream_byte(data, 2),
        ])) * 1000,
        i64::MAX / 2,
    ];
    for &t in &targets {
        // Convert micros → sample-based pts the demuxer expects
        // (seek_to takes a pts in the stream time base; the demuxer's
        // packets carry sample-count pts, so reuse the raw value —
        // the contract under test is panic-freedom, not accuracy).
        let _ = demux.seek_to(0, t);
        for _ in 0..8 {
            match demux.next_packet() {
                Ok(_) => {}
                Err(_) => break,
            }
        }
    }
});

/// Bounded byte accessor (avoids repeated `get` boilerplate above).
fn stream_byte(data: &[u8], idx: usize) -> &u8 {
    data.get(idx).unwrap_or(&0)
}
