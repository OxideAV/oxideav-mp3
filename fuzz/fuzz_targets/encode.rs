#![no_main]

//! Adversarial fuzzer for the Layer III **encode loop**: hostile PCM
//! geometries, extreme bitrate / sample-rate / mode configurations,
//! and every opt-in encoder toggle, driven through both the
//! `oxideav_core::Encoder` trait surface and the direct
//! [`oxideav_mp3::Mp3Encoder`] stream API.
//!
//! r432 FUZZ-depth lane. The decode surface has had libFuzzer
//! coverage since r289 (`decode`, `granule`) and the demuxer since
//! r405 (`demux`); this target closes the encode-side gap. The
//! contract under test is panic-freedom: every constructor, toggle,
//! push, and flush returns a value or an `Err` — never panics, never
//! overflows in a debug build, never indexes out of bounds — on any
//! configuration/PCM combination an attacker can request.
//!
//! ## Lane A — trait surface (`CodecRegistry` → `Encoder`)
//!
//! `CodecParameters` built from raw attacker numbers (sample_rate,
//! channels, bit_rate all unconstrained) exercise the
//! `make_encoder` validation ladder; when construction succeeds,
//! `send_frame` is driven with hostile [`AudioFrame`] geometries —
//! wrong plane counts, byte lengths inconsistent with the advertised
//! `samples`, zero-sample frames, non-audio frames are unreachable
//! here but plane/length mismatch is the `frame_to_i16` validation
//! surface — plus a mid-stream `flush` so the post-flush
//! `send_frame` rejection and the idempotent double-flush are hit.
//!
//! ## Lane B — direct stream API (`Mp3Encoder`)
//!
//! An attacker-selected constructor (plain CBR, outer-loop,
//! threshold-in-quiet ± offset, quality preset, and the five
//! joint-stereo variants) at an attacker-selected rate (the nine
//! §2.4.2.3 / LSF / MPEG-2.5 rates, or a hostile off-table rate),
//! bitrate (often off-ladder), and channel mode (all four, including
//! the rejected `JointStereo` request on the plain constructor).
//! On top: the full toggle surface — `enable_vbr` with attacker
//! min/max, CRC protection, Xing/Info emission with attacker flag
//! words, forced short / mixed blocks, energy- and Model-2-driven
//! auto block-typing with non-finite thresholds (NaN / ±inf),
//! intensity-stereo arming with attacker band bounds. PCM is then
//! pushed in attacker-shaped chunks (length 0, odd lengths on
//! stereo interleave, single samples, multi-frame bursts) and the
//! stream is finished into a byte sink. Every `Err` at any stage is
//! a valid outcome; the harness simply stops the lane.
//!
//! Spec basis: ISO/IEC 11172-3 §2.4.2.3 (header field ladders),
//! §C.1.5 (encode procedure, inner/outer loops, block switching),
//! §2.4.3.4.9 (MS / intensity joint stereo); ISO/IEC 13818-3
//! §2.4.3.2 (LSF single-granule frames).

use libfuzzer_sys::fuzz_target;
use oxideav_core::{AudioFrame, CodecId, CodecParameters, CodecRegistry, Frame, SampleFormat};
use oxideav_mp3::frame::ChannelMode;
use oxideav_mp3::quality::QualityPreset;
use oxideav_mp3::stream_encoder::Mp3Encoder;
use oxideav_mp3::xing_info::{flag_bit, XingTagSpec};
use oxideav_mp3::XingTagId;

/// Every sample rate the encoder claims to support: MPEG-1, MPEG-2
/// LSF, MPEG-2.5.
const RATES: [u32; 9] = [
    44_100, 48_000, 32_000, 22_050, 24_000, 16_000, 11_025, 12_000, 8_000,
];

/// Cap on the total number of i16 PCM values pushed per iteration.
/// Four MPEG-1 stereo frames (4 × 1152 × 2) plus a partial tail —
/// enough to assemble several frames (reservoir scheduling, block-type
/// state machine transitions) while keeping iterations fast.
const MAX_PCM_VALUES: usize = 4 * 1152 * 2 + 777;

/// Synthesize `n` i16 samples by cycling the attacker byte pool as
/// little-endian pairs. Decouples the reachable PCM volume from the
/// fuzz-input length so frame assembly is reached even on small
/// inputs, while every sample value stays attacker-controlled.
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

/// Hostile f64 threshold values, including the non-finite ones the
/// comparison-driven auto paths must tolerate.
fn hostile_f64(seed: u8) -> f64 {
    match seed % 8 {
        0 => f64::NAN,
        1 => f64::INFINITY,
        2 => f64::NEG_INFINITY,
        3 => -1.0,
        4 => 0.0,
        5 => f64::MIN_POSITIVE,
        6 => 1.0e300,
        _ => f64::from(seed) * 0.37,
    }
}

fn lane_trait(bytes: &[u8]) {
    if bytes.len() < 6 {
        return;
    }
    let mut reg = CodecRegistry::new();
    oxideav_mp3::register_codecs(&mut reg);
    let id = CodecId::new(oxideav_mp3::demuxer::CODEC_ID_STR);
    let mut params = CodecParameters::audio(id);
    // Raw attacker numbers — the validation ladder is the surface.
    params.sample_rate = Some(u32::from(u16::from_le_bytes([bytes[0], bytes[1]])));
    params.channels = Some(u16::from(bytes[2] % 5));
    params.bit_rate = Some(u64::from(bytes[3]) * 4_000);
    params.sample_format = Some(SampleFormat::S16);
    let Ok(mut enc) = reg.first_encoder(&params) else {
        return;
    };

    let channels = usize::from(bytes[2] % 5).max(1);
    let pool = &bytes[6..];
    for step in 0..6u8 {
        let ctl = *pool.get(usize::from(step)).unwrap_or(&step);
        // Attacker-shaped AudioFrame: samples count, plane count, and
        // per-plane byte length all independently chosen so the
        // consistency checks (1 plane, samples × channels × 2 bytes)
        // are exercised from every direction.
        let samples = u32::from(ctl & 0x3F) * 37;
        let plane_count = usize::from(ctl >> 6); // 0..=3 planes
        let correct_len = samples as usize * channels * 2;
        let plane_len = if ctl & 0x20 != 0 {
            correct_len
        } else {
            // Deliberately inconsistent length.
            (correct_len)
                .wrapping_add(usize::from(ctl))
                .wrapping_sub(16)
                % (correct_len + 64)
        };
        let planes: Vec<Vec<u8>> = (0..plane_count)
            .map(|p| {
                let mut v = vec![0u8; plane_len];
                for (i, slot) in v.iter_mut().enumerate() {
                    *slot = pool
                        .get((usize::from(step) + 1 + i + p) % pool.len().max(1))
                        .copied()
                        .unwrap_or(0);
                }
                v
            })
            .collect();
        let frame = Frame::Audio(AudioFrame {
            samples,
            pts: Some(i64::from(step) * 1152),
            data: planes,
        });
        let _ = enc.send_frame(&frame);
        while enc.receive_packet().is_ok() {}
        if step == 3 && ctl & 0x10 != 0 {
            // Mid-stream flush: subsequent send_frame must reject
            // cleanly, receive_packet must drain to Eof.
            let _ = enc.flush();
        }
    }
    let _ = enc.flush();
    let _ = enc.flush();
    while enc.receive_packet().is_ok() {}
}

#[allow(clippy::too_many_lines)]
fn lane_direct(bytes: &[u8]) {
    if bytes.len() < 8 {
        return;
    }
    let ctor_sel = bytes[0] % 12;
    let rate = if bytes[1] & 0x80 != 0 {
        // Hostile off-table rate (0 and huge values included).
        u32::from(bytes[1] & 0x7F) * 997
    } else {
        RATES[usize::from(bytes[1]) % RATES.len()]
    };
    let bitrate = if bytes[2] & 0x80 != 0 {
        // Off-ladder / extreme bitrate request.
        u32::from(bytes[2] & 0x7F) * 4
    } else {
        // On one of the two ladders often enough to construct.
        let ladders = [
            &oxideav_mp3::stream_encoder::MPEG1_L3_BITRATE_LADDER_KBPS,
            &oxideav_mp3::stream_encoder::LSF_L3_BITRATE_LADDER_KBPS,
        ];
        ladders[usize::from(bytes[2] >> 6) % 2][usize::from(bytes[2]) % 14]
    };
    let mode = match bytes[3] & 0b11 {
        0 => ChannelMode::SingleChannel,
        1 => ChannelMode::Stereo,
        2 => ChannelMode::DualChannel,
        _ => ChannelMode::JointStereo, // rejected by the plain ctor
    };
    let toggles = bytes[4];
    let seed_a = bytes[5];
    let seed_b = bytes[6];
    let preset = match bytes[7] % 4 {
        0 => QualityPreset::Transparent,
        1 => QualityPreset::High,
        2 => QualityPreset::Standard,
        _ => QualityPreset::Fast,
    };

    let built = match ctor_sel {
        0 => Mp3Encoder::new(bitrate, rate, mode),
        1 => Mp3Encoder::new_with_outer_loop(bitrate, rate, mode, hostile_f64(seed_a)),
        2 => Mp3Encoder::new_with_threshold_in_quiet(bitrate, rate, mode),
        3 => {
            Mp3Encoder::new_with_threshold_in_quiet_offset(bitrate, rate, mode, hostile_f64(seed_a))
        }
        4 => Mp3Encoder::new_with_quality_preset(bitrate, rate, mode, preset),
        5 => Mp3Encoder::new_joint_stereo_ms(bitrate, rate),
        6 => Mp3Encoder::new_joint_stereo_auto(bitrate, rate),
        7 => Mp3Encoder::new_joint_stereo_is(bitrate, rate, usize::from(seed_a)),
        8 => Mp3Encoder::new_joint_stereo_ms_is(bitrate, rate, usize::from(seed_a)),
        9 => Mp3Encoder::new_joint_stereo_auto_is(bitrate, rate, usize::from(seed_a)),
        10 => Mp3Encoder::new_joint_stereo_auto_is_adaptive(bitrate, rate, usize::from(seed_a)),
        _ => Mp3Encoder::new(bitrate, rate, mode),
    };
    let Ok(mut enc) = built else {
        return;
    };

    // Toggle surface — every Err is a valid outcome; keep going with
    // whatever stuck so toggle *combinations* are exercised too.
    if toggles & 0x01 != 0 {
        let min = u32::from(seed_a) * 2;
        let max = u32::from(seed_b) * 2;
        let _ = enc.enable_vbr(min, max);
    }
    if toggles & 0x02 != 0 {
        enc.with_protection_bit(true);
    }
    if toggles & 0x04 != 0 {
        enc.enable_xing_info(XingTagSpec {
            id: if seed_a & 1 == 0 {
                XingTagId::Xing
            } else {
                XingTagId::Info
            },
            // Attacker flag word; FRAMES | BYTES auto-fill at flush,
            // higher bits written verbatim.
            flags: u32::from(seed_b) | flag_bit::FRAMES,
            frames: None,
            bytes: None,
            toc: if seed_b & 0x04 != 0 {
                Some([seed_a; 100])
            } else {
                None
            },
            quality: if seed_b & 0x08 != 0 {
                Some(u32::from(seed_a))
            } else {
                None
            },
        });
    }
    if toggles & 0x08 != 0 {
        let _ = enc.force_short_blocks_for_testing(seed_a & 1 == 0);
    }
    if toggles & 0x10 != 0 {
        let _ = enc.force_mixed_blocks_for_testing(seed_a & 2 == 0);
    }
    if toggles & 0x20 != 0 {
        let _ = if seed_a & 4 == 0 {
            enc.enable_auto_block_type(hostile_f64(seed_b))
        } else {
            enc.enable_auto_block_type_with_mixed(hostile_f64(seed_b), hostile_f64(seed_a))
        };
    }
    if toggles & 0x40 != 0 {
        let _ = enc.enable_auto_block_type_model2();
    }
    if toggles & 0x80 != 0 {
        let _ = enc.enable_intensity_stereo(usize::from(seed_b));
    }

    // Hostile PCM geometry: attacker-shaped chunk schedule over a
    // bounded total. Chunk lengths include 0, 1 (odd on stereo
    // interleave), and multi-frame bursts.
    let pool = &bytes[8..];
    let mut remaining = {
        let hint = usize::from(seed_a) * usize::from(seed_b) * 8;
        hint % (MAX_PCM_VALUES + 1)
    };
    let mut chunk_sel = 0usize;
    while remaining > 0 {
        let sel = *pool.get(chunk_sel % pool.len().max(1)).unwrap_or(&17);
        chunk_sel += 1;
        let chunk = match sel % 5 {
            0 => 0usize,
            1 => 1,
            2 => usize::from(sel) * 3,
            3 => 1152,
            _ => 2 * 1152 + usize::from(sel),
        }
        .min(remaining);
        let pcm = pcm_from_pool(pool, chunk);
        if enc.push_samples(&pcm).is_err() {
            return;
        }
        if chunk == 0 {
            // Zero-length chunks make no progress; bail out of the
            // schedule rather than spinning.
            remaining = remaining.saturating_sub(1);
        } else {
            remaining -= chunk;
        }
    }
    let mut sink: Vec<u8> = Vec::new();
    let _ = enc.finish(&mut sink);
}

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }
    let (lane, rest) = data.split_at(1);
    if lane[0] & 0x80 != 0 {
        lane_trait(rest);
    } else {
        lane_direct(rest);
    }
});
