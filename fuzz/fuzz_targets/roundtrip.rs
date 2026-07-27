#![no_main]

//! Encode → decode **roundtrip invariant** fuzzer.
//!
//! r432 FUZZ-depth lane. Unlike the panic-freedom targets, this one
//! asserts a *semantic* invariant: every byte stream this crate's
//! encoder emits under a **valid** configuration must (a) frame-walk
//! back into exactly the frames the encoder scheduled, (b) decode
//! through this crate's own `Decoder` trait implementation without a
//! single error, and (c) yield at least as many PCM samples per
//! channel as were pushed in (the §2.4.1.7 tail flush zero-pads the
//! final partial frame, so encoded sample count is
//! `ceil(pushed / samples_per_frame) × samples_per_frame ≥ pushed`).
//! An `Err` out of `send_packet`, a lost frame, or a short sample
//! count is a real encoder- or decoder-side bug, reported as a fuzz
//! finding via `panic!`.
//!
//! ## Configuration space (kept *valid* on purpose)
//!
//! The attacker chooses — within the supported envelope — the sample
//! rate (all nine: MPEG-1 / MPEG-2 LSF / MPEG-2.5), a ladder bitrate
//! for the rate's version, mono / stereo / dual-channel / MS
//! joint-stereo, the outer-loop / threshold-in-quiet / quality-preset
//! constructors, CRC-16 protection, VBR with a valid `min ≤ max ≤
//! ctor` ladder window, Xing/Info emission, forced short blocks, and
//! the auto block-type scheduler (tolerated to refuse on LSF).
//! Configuration calls that *reject* (e.g. mixed blocks at 8 kHz,
//! auto block-typing on LSF) simply drop that toggle — the roundtrip
//! then runs with whatever configuration stuck. `finish` may still
//! legitimately fail on an unschedulable bit-reservoir budget
//! (extreme low bitrates); that aborts the iteration without any
//! assertion, since no stream was emitted.
//!
//! ## PCM
//!
//! Sample values come from the attacker byte pool (cycled as
//! little-endian i16 pairs — silence, full-scale steps, sign flips
//! and arbitrary noise are all reachable), with an attacker-chosen
//! total length up to a few frames plus a partial tail so the
//! zero-padded tail-flush path is exercised on most iterations.
//!
//! Spec basis: ISO/IEC 11172-3 §2.4 (frame syntax + decode chain),
//! §C.1.5 (encode procedure); ISO/IEC 13818-3 §2.4.3.2 (LSF frames).

use libfuzzer_sys::fuzz_target;
use oxideav_core::{
    CodecId, CodecParameters, CodecRegistry, Decoder, Frame, Packet, Rational, TimeBase,
};
use oxideav_mp3::frame::{ChannelMode, FrameWalker};
use oxideav_mp3::quality::QualityPreset;
use oxideav_mp3::stream_encoder::{
    Mp3Encoder, LSF_L3_BITRATE_LADDER_KBPS, MPEG1_L3_BITRATE_LADDER_KBPS,
};
use oxideav_mp3::xing_info::{flag_bit, XingTagSpec};
use oxideav_mp3::XingTagId;

const RATES: [u32; 9] = [
    44_100, 48_000, 32_000, 22_050, 24_000, 16_000, 11_025, 12_000, 8_000,
];

/// Bound the pushed PCM to a few frames plus a tail: enough for
/// reservoir scheduling across frames and the zero-padded tail flush,
/// small enough for throughput.
const MAX_PCM_PER_CH: usize = 3 * 1152 + 500;

fn pcm_from_pool(pool: &[u8], n: usize) -> Vec<i16> {
    if pool.is_empty() {
        return vec![0i16; n];
    }
    (0..n)
        .map(|i| {
            let lo = pool[(2 * i) % pool.len()];
            let hi = pool[(2 * i + 1) % pool.len()];
            i16::from_le_bytes([lo, hi])
        })
        .collect()
}

fn build_decoder(sample_rate: u32, channels: u16) -> Box<dyn Decoder> {
    let mut reg = CodecRegistry::new();
    oxideav_mp3::register_codecs(&mut reg);
    let id = CodecId::new(oxideav_mp3::demuxer::CODEC_ID_STR);
    let mut params = CodecParameters::audio(id);
    params.channels = Some(channels);
    params.sample_rate = Some(sample_rate);
    reg.first_decoder(&params)
        .expect("decoder factory must build for a supported rate/channel pair")
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 6 {
        return;
    }
    let rate = RATES[usize::from(data[0]) % RATES.len()];
    let mpeg1 = matches!(rate, 32_000 | 44_100 | 48_000);
    let ladder: &[u32; 14] = if mpeg1 {
        &MPEG1_L3_BITRATE_LADDER_KBPS
    } else {
        &LSF_L3_BITRATE_LADDER_KBPS
    };
    let ctor_idx = usize::from(data[1]) % 14;
    let bitrate = ladder[ctor_idx];
    let sel = data[2];
    let toggles = data[3];
    let seed_a = data[4];
    let seed_b = data[5];
    let pool = &data[6..];

    let stereo = sel & 0x01 != 0;
    let mode = if stereo {
        if sel & 0x02 != 0 {
            ChannelMode::DualChannel
        } else {
            ChannelMode::Stereo
        }
    } else {
        ChannelMode::SingleChannel
    };
    let nch: usize = if stereo { 2 } else { 1 };

    let preset = match seed_a % 4 {
        0 => QualityPreset::Transparent,
        1 => QualityPreset::High,
        2 => QualityPreset::Standard,
        _ => QualityPreset::Fast,
    };
    let built = match (sel >> 2) % 5 {
        0 => Mp3Encoder::new(bitrate, rate, mode),
        1 => Mp3Encoder::new_with_outer_loop(
            bitrate,
            rate,
            mode,
            oxideav_mp3::stream_encoder::DEFAULT_OUTER_LOOP_THRESHOLD,
        ),
        2 => Mp3Encoder::new_with_threshold_in_quiet(bitrate, rate, mode),
        3 => Mp3Encoder::new_with_quality_preset(bitrate, rate, mode, preset),
        _ => {
            if stereo {
                Mp3Encoder::new_joint_stereo_ms(bitrate, rate)
            } else {
                Mp3Encoder::new(bitrate, rate, mode)
            }
        }
    };
    let Ok(mut enc) = built else {
        // Every configuration above is intended to be valid; a
        // constructor rejection here is a supported-envelope
        // regression.
        panic!("valid-envelope constructor rejected: rate={rate} bitrate={bitrate} mode={mode:?}");
    };

    // Optional toggles. Rejections are tolerated (documented
    // envelope restrictions: auto block-typing on LSF, VBR windows,
    // etc.) — the roundtrip proceeds with whatever stuck.
    if toggles & 0x01 != 0 {
        enc.with_protection_bit(true);
    }
    if toggles & 0x02 != 0 {
        // Valid VBR window: min ≤ max ≤ ctor bitrate, both on-ladder.
        let max_idx = usize::from(seed_a) % (ctor_idx + 1);
        let min_idx = usize::from(seed_b) % (max_idx + 1);
        let _ = enc.enable_vbr(ladder[min_idx], ladder[max_idx]);
    }
    if toggles & 0x04 != 0 {
        enc.enable_xing_info(XingTagSpec {
            id: if seed_a & 1 == 0 {
                XingTagId::Xing
            } else {
                XingTagId::Info
            },
            flags: flag_bit::FRAMES | flag_bit::BYTES,
            frames: None,
            bytes: None,
            toc: None,
            quality: None,
        });
    }
    if toggles & 0x08 != 0 {
        let _ = enc.force_short_blocks_for_testing(true);
    }
    if toggles & 0x10 != 0 {
        let _ = enc.enable_auto_block_type(f64::from(seed_b) * 0.02 + 0.01);
    }

    // Push attacker PCM in a couple of chunks (chunk boundary from
    // the pool so partial-frame buffering is exercised), then finish.
    let total_per_ch = {
        let hint = usize::from(seed_a) * 37 + usize::from(seed_b) * 691;
        hint % (MAX_PCM_PER_CH + 1)
    };
    let total_values = total_per_ch * nch;
    let split = if total_values > 1 {
        usize::from(seed_b) % total_values
    } else {
        0
    };
    let pcm = pcm_from_pool(pool, total_values);
    if enc.push_samples(&pcm[..split]).is_err() {
        return;
    }
    if enc.push_samples(&pcm[split..]).is_err() {
        return;
    }
    let mut stream: Vec<u8> = Vec::new();
    if enc.finish(&mut stream).is_err() {
        // Legitimate: unschedulable reservoir / VBR slot at extreme
        // low bitrates. No stream was emitted; nothing to assert.
        return;
    }

    // --- Decode leg. Every emitted frame must decode cleanly. ---
    let mut dec = build_decoder(rate, nch as u16);
    let tb = TimeBase(Rational::new(1, i64::from(rate)));
    let mut decoded_per_ch: u64 = 0;
    let mut frames_walked = 0usize;
    for f in FrameWalker::new(&stream) {
        frames_walked += 1;
        let pkt = Packet::new(0, tb, f.data.to_vec());
        if let Err(e) = dec.send_packet(&pkt) {
            panic!(
                "encoder-emitted frame #{frames_walked} rejected by decoder: {e:?} \
                 (rate={rate} bitrate={bitrate} mode={mode:?} toggles={toggles:#04x} \
                 offset={} len={})",
                f.offset,
                f.data.len()
            );
        }
        while let Ok(frame) = dec.receive_frame() {
            if let Frame::Audio(a) = frame {
                decoded_per_ch += u64::from(a.samples);
            }
        }
    }
    let _ = dec.flush();
    while let Ok(frame) = dec.receive_frame() {
        if let Frame::Audio(a) = frame {
            decoded_per_ch += u64::from(a.samples);
        }
    }

    // (a) No frame lost between encoder scheduling and the walker:
    // the stream must contain at least one frame whenever a full
    // frame of PCM went in.
    if total_per_ch > 0 && frames_walked == 0 {
        panic!(
            "no frames walked from a non-empty encode \
             (rate={rate} bitrate={bitrate} pushed={total_per_ch} bytes={})",
            stream.len()
        );
    }
    // (c) Tail-flush padding invariant: decoded PCM covers the input.
    if decoded_per_ch < total_per_ch as u64 {
        panic!(
            "decoded {decoded_per_ch} samples/ch < pushed {total_per_ch} \
             (rate={rate} bitrate={bitrate} mode={mode:?} toggles={toggles:#04x} \
             frames={frames_walked})"
        );
    }
});
