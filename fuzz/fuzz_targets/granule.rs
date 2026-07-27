#![no_main]

//! Structured adversarial fuzzer for the Layer III **per-granule
//! decode primitives**, below the `Decoder` trait surface.
//!
//! Round 289 FUZZ-depth lane. The `decode` target drives the public
//! `Decoder` trait, but the deepest decode primitives are reached
//! there only once the bit-reservoir `main_data_begin` lookback is
//! satisfiable — a state a free-form fuzzer hits only by luck. This
//! target *constructs* a structurally-valid header + side-info block
//! and then drives the deep per-granule chain directly on attacker
//! bytes, so the dequant / overlap / filterbank math is stressed on
//! every iteration regardless of reservoir state.
//!
//! ## Surfaces driven on every iteration
//!
//! 1. `frame::parse_header` on a constructed 4-byte header (valid
//!    sync, attacker-chosen version / sample-rate / bitrate / mode).
//! 2. `side_info::parse_side_info` on attacker bytes shaped as the
//!    header-implied side-info block (ISO/IEC 11172-3 §2.4.1.7 /
//!    ISO/IEC 13818-3 §2.4.1.7) — yields real `GranuleChannel`
//!    instances whose `part2_3_length`, `big_values`, `block_type`,
//!    `region*_count`, `table_select`, `subblock_gain`, etc. are all
//!    attacker-controlled within the parser's accepted ranges.
//! 3. `scalefactors::decode_scalefactors` on the attacker main-data
//!    slot (§2.4.2.7 scalefactor bit-unpacking).
//! 4. `huffman::decode_huffman` per granule/channel — big_values
//!    region split + count1 quadruple decode (§2.4.2.7), driven by a
//!    `MainDataReader` over the attacker slot with the granule's
//!    attacker `part2_3_length` bit budget.
//! 5. `requantize::requantize` — power-3/4 dequant + scalefactor
//!    application + global/subblock gain (§2.4.3.4).
//! 6. `alias::alias_reduce` — §2.4.3.4.10.1 alias butterflies (long /
//!    mixed only).
//! 7. `imdct::imdct_granule` — long / short / start / stop window
//!    overlap-add (§2.4.3.4.10.2), with carried overlap state.
//! 8. `synth::synth_granule` — 32-band polyphase synthesis filterbank
//!    (§2.4.3.4.11), with carried shift-register state.
//!
//! The contract under test is panic-freedom: every primitive returns
//! a value or an `Err`, never panics, overflows in debug, or indexes
//! out of bounds, on any attacker-shaped granule.
//!
//! ## Fuzz input layout
//!
//! ```text
//!   byte 0      — control:
//!                   bits 0-1 → MPEG version (MPEG-1 / MPEG-2 LSF /
//!                              MPEG-2.5 — all three decode)
//!                   bits 2-3 → sample-rate index (0..=2)
//!                   bits 4-7 → channel mode + body stride seed
//!   byte 1      — bitrate index seed (mapped to 1..=14)
//!   byte 2      — mode / mode-extension seed
//!   bytes 3..   — raw bytes used verbatim for the side-info block and
//!                 the main-data slot.
//! ```
//!
//! Output is discarded — the fuzzer only cares about *return*.
//!
//! Spec basis: ISO/IEC 11172-3 §2.4.1.7 (side info), §2.4.2.7
//! (Huffman and scalefactors), §2.4.3.4 (requant, alias, IMDCT,
//! filterbank); ISO/IEC 13818-3 §2.4.1.7 / §2.4.2.7 (LSF layout).

use libfuzzer_sys::fuzz_target;
use oxideav_mp3::alias::alias_reduce;
use oxideav_mp3::frame::{parse_header, MpegVersion};
use oxideav_mp3::huffman::decode_huffman;
use oxideav_mp3::imdct::{imdct_granule, ImdctState};
use oxideav_mp3::requantize::requantize;
use oxideav_mp3::scalefactors::{decode_scalefactors, MainDataReader};
use oxideav_mp3::side_info::parse_side_info;
use oxideav_mp3::synth::{synth_granule, SynthState};

/// Build a structurally-valid 4-byte Layer III header. All three
/// decodable versions are covered — MPEG-1, MPEG-2 LSF, and (since
/// r432, the MPEG-2.5 tables having landed in the meantime) the
/// MPEG-2.5 extension, whose 8 / 11.025 / 12 kHz scalefactor-band
/// dispatch (`requantize::LONG_STARTS_MPEG25_8` and friends) is
/// exactly the deep-chain surface this target exists to stress.
///
/// Note the MPEG-2.5 header narrows the syncword to 11 bits and
/// repurposes bit 20 as the extension discriminator (see
/// `frame::MpegVersion`): writing `0b00` into bits 20..19 below
/// produces exactly that shape.
fn build_header(ctl: u8, br: u8, modesel: u8) -> [u8; 4] {
    let mut raw: u32 = 0x7FF << 21;
    // MPEG-1 / MPEG-2 LSF / MPEG-2.5 (two seed values map to 2.5 so
    // the low-rate tables get a fair share of the energy).
    let ver_bits = match ctl & 0b11 {
        0 => 0b11u32,
        1 => 0b10,
        _ => 0b00,
    };
    raw |= ver_bits << 19;
    raw |= 0b01u32 << 17; // Layer III
    raw |= 1 << 16; // protection_bit '1' = no CRC (keep the slot layout simple)
    let br_idx = (br % 14) + 1; // 1..=14
    raw |= u32::from(br_idx) << 12;
    let sr_idx = (ctl >> 2) & 0b11;
    let sr_idx = sr_idx % 3; // 0..=2 (3 reserved)
    raw |= u32::from(sr_idx) << 10;
    raw |= u32::from((ctl >> 4) & 1) << 9; // padding
    let mode = modesel & 0b11;
    raw |= u32::from(mode) << 6;
    let ext = (modesel >> 2) & 0b11;
    raw |= u32::from(ext) << 4;
    raw.to_be_bytes()
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        // Still exercise the header + side-info parsers on a tiny
        // buffer.
        let _ = parse_header(data);
        return;
    }
    let ctl = data[0];
    let br = data[1];
    let modesel = data[2];
    let body = &data[3..];

    let hdr_bytes = build_header(ctl, br, modesel);
    let hdr = match parse_header(&hdr_bytes) {
        Ok(h) => h,
        Err(_) => return,
    };

    // Parse the side-info from attacker bytes. The side-info block sits
    // right after the header (no CRC, since we cleared protection); we
    // feed the attacker body verbatim. A parse failure (short buffer,
    // out-of-range field) is a valid outcome — just return.
    let si = match parse_side_info(&hdr, body) {
        Ok(si) => si,
        Err(_) => return,
    };
    let si_len = si.byte_len();
    if body.len() <= si_len {
        // No main-data slot left after side-info — the parsers above
        // were still exercised.
        return;
    }
    let main_slot = &body[si_len..];

    // §2.4.2.7 scalefactor decode on the attacker main-data slot.
    let fsf = match decode_scalefactors(&hdr, &si, main_slot) {
        Ok(f) => f,
        Err(_) => return,
    };

    // Carried filterbank state across granules/channels, exactly as the
    // real decoder threads it.
    let mut imdct_state = [ImdctState::new(), ImdctState::new()];
    let mut synth_state = [SynthState::new(), SynthState::new()];

    let nch = si.channels as usize;
    let lsf = hdr.version != MpegVersion::Mpeg1;
    let _ = lsf;

    for gr in 0..si.granule_count as usize {
        for ch in 0..nch {
            let gc = &si.granules[gr][ch];
            let sf = &fsf.granules[gr][ch];

            // Huffman decode over the attacker slot with this granule's
            // attacker bit budget. A fresh reader per (gr, ch) keeps the
            // budget exercise independent; the real decoder advances a
            // shared cursor, but for panic-freedom an independent read
            // is strictly more adversarial (the slot is re-read from the
            // top with a possibly-huge part2_3_length budget).
            let mut reader = MainDataReader::new(main_slot);
            let part3_bits = u32::from(gc.part2_3_length);
            let is = match decode_huffman(
                &mut reader,
                gc,
                part3_bits,
                hdr.sample_rate_hz,
                hdr.version,
            ) {
                Ok(is) => is,
                Err(_) => continue,
            };

            // §2.4.3.4 requantise → alias-reduce → IMDCT → synthesis.
            let xr = requantize(&is, gc, sf, hdr.sample_rate_hz, hdr.version);
            let xar = alias_reduce(&xr, gc);
            let ch_state = ch.min(1);
            let subband_time = imdct_granule(&xar, gc, &mut imdct_state[ch_state]);
            let _pcm = synth_granule(&subband_time, &mut synth_state[ch_state]);
        }
    }
});
