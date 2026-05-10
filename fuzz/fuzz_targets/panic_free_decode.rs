#![no_main]

//! Fuzz: feed arbitrary bytes to `Mp3Decoder::send_packet` /
//! `receive_frame` and assert the decoder never panics. Decode failures
//! are expected (most random inputs are not valid MP3); the contract
//! is that they bubble out as `Result::Err`, never as a process abort.
//!
//! The harness allocates a fresh decoder per fuzz iteration and feeds
//! the input as a single packet — that's the same shape the real demux
//! pipeline uses (one MP3 frame per packet). Inputs longer than 16 KiB
//! are truncated to bound per-iteration work.

use libfuzzer_sys::fuzz_target;
use oxideav_core::{CodecId, CodecParameters, Packet, TimeBase};
use oxideav_mp3::decoder::make_decoder;
use oxideav_mp3::CODEC_ID_STR;

const MAX_INPUT_BYTES: usize = 16 * 1024;

fuzz_target!(|data: &[u8]| {
    let bytes = if data.len() > MAX_INPUT_BYTES {
        &data[..MAX_INPUT_BYTES]
    } else {
        data
    };

    let params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
    let mut dec = match make_decoder(&params) {
        Ok(d) => d,
        Err(_) => return,
    };
    let pkt = Packet::new(0u32, TimeBase::new(1, 44_100), bytes.to_vec());
    // Either send_packet fails (bad header) or receive_frame fails
    // (bad payload) — both are fine. The assertion is implicit:
    // libfuzzer treats a panic / abort / timeout as a finding.
    let _ = dec.send_packet(&pkt);
    let _ = dec.receive_frame();
});
