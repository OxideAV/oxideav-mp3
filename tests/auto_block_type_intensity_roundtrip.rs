//! Integration tests for the round-307 **auto-block-type-scheduled +
//! MS-joint + intensity-stereo** encoder path (ISO/IEC 11172-3:1993 +
//! ISO/IEC 13818-3:1997 §2.4.3.2 per-window bound, §2.4.3.4.9 channel
//! agreement, §C.1.5.2 transition scheduler).
//!
//! r303 / r305 / r306 wired the §2.4.3.4.9.3 per-window short intensity
//! bound on the **force-short** intensity paths (every granule pinned to
//! a pure short block). r307 lifts the `IntensityShortBlocksUnsupported`
//! rejection on the signal-driven
//! [`oxideav_mp3::Mp3Encoder::enable_auto_block_type`] scheduler when MS
//! joint stereo is armed: the §C.1.5.2 state machine emits a *mix* of
//! Long / Start / Short / End granules within one stream, and the
//! intensity coupling is now chosen **per granule** —
//!
//!   * a granule the scheduler emitted as a **pure short** block takes
//!     the §2.4.3.4.9.3 per-window short coupling (Pass 1.45 short branch
//!     + Pass 1.5 per-window MS region);
//!   * Long / Start / End granules take the long-block band-walk coupling.
//!
//! MS-joint stereo ([`oxideav_mp3::Mp3Encoder::new_joint_stereo_ms_is`])
//! mirrors one shared scheduler emission across both channels of every
//! granule, so the §2.4.3.4.9 "both channels share the same block_type /
//! window_switching_flag / mixed_block_flag" agreement that intensity
//! coupling needs (it folds each granule's `(L, R)` band-by-band) holds
//! by construction.
//!
//! r308 lifts the remaining rejection on the **intensity-only** path
//! ([`oxideav_mp3::Mp3Encoder::new_joint_stereo_is`] +
//! `enable_auto_block_type`, no MS). Arming intensity coupling now forces
//! the SAME channel-agreement OR-fold in the scheduler walk
//! (`channel_agreement_active = MS-joint OR intensity-armed`): the
//! per-channel attack flags are OR-folded into one shared (channel-0)
//! scheduler and its emission is mirrored across both channels, so L/R
//! block types stay consistent even without MS — the per-band fold
//! geometry is therefore well-defined. The mixed-promotion auto variant
//! and the Model-2-driven auto path under intensity stay rejected.
//!
//! Covered end-to-end with in-tree primitives only:
//!
//! 1. **API acceptance** — `enable_auto_block_type` succeeds on an
//!    MS+intensity encoder AND on an intensity-only encoder; the
//!    mixed-promotion + Model-2 auto variants under intensity stay
//!    rejected.
//! 2. **Mixed block-type stream** — a transient stereo stimulus drives
//!    the scheduler to emit both long-family AND pure-short granules in
//!    the same stream, each carrying `mode = '01'`, the intensity bit set,
//!    and (for short granules) per-window scalefactor positions in range.
//! 3. **Self-decode fidelity** — a hard-left high (intensity-region) tone
//!    reconstructs left-leaning through a spec-order self-decode.
//! 4. **Bit-exact stability** — same PCM in ⇒ same bytes out.

use std::f32::consts::PI;

use oxideav_mp3::{
    alias_reduce, decode_huffman, decode_scalefactors, imdct_granule, parse_header,
    parse_side_info, pcm_f32_to_i16, process_stereo, reorder, requantize, synth_granule, BlockType,
    ChannelMode, FrameWalker, ImdctState, MainDataReader, Mp3Encoder, Reservoir, StreamEncodeError,
    SynthState, DEFAULT_ATTACK_THRESHOLD, NUM_LINES, PCM_PER_GRANULE,
};

const SR: u32 = 44_100;
const BR: u32 = 192;
const START_SFB: usize = 8;
const HIGH_HZ: f32 = 8_000.0;

/// Part2 (scalefactor) bit count of one MPEG-1 granule channel, branching
/// on block type so the self-decoder can skip past the scalefactors of
/// either a long-family or a pure-short granule.
fn mpeg1_part2_bits(scalefac_compress: u16, short: bool) -> usize {
    const MPEG1_SLEN: [(u8, u8); 16] = [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (3, 0),
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 1),
        (2, 2),
        (2, 3),
        (3, 1),
        (3, 2),
        (3, 3),
        (4, 2),
        (4, 3),
    ];
    let (slen1, slen2) = MPEG1_SLEN[(scalefac_compress & 0xF) as usize];
    if short {
        // 12 short SFB × 3 windows, slen1 for 0..6, slen2 for 6..12.
        let mut bits = 0usize;
        for sfb in 0..12usize {
            let slen = if sfb < 6 { slen1 } else { slen2 };
            bits += (slen as usize) * 3;
        }
        bits
    } else {
        // 21 long bands grouped 11·slen1 + 10·slen2.
        11 * (slen1 as usize) + 10 * (slen2 as usize)
    }
}

/// Stereo PCM: a sustained low tone (mild L>R pan) with periodic
/// full-scale transient bursts and a hard-left high tone for the
/// intensity region. The bursts trip the attack detector so the
/// §C.1.5.2 scheduler engages Start → Short → Stop around them while
/// keeping long blocks on the steady stretches — producing a stream that
/// mixes long-family and pure-short granules.
fn transient_stereo_pcm(n: usize) -> Vec<i16> {
    let two_pi = 2.0 * PI;
    let scale = f32::from(i16::MAX);
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        let t = i as f32 / SR as f32;
        // Quiet steady tones so the attack detector's running ambient
        // stays low: a below-bound 440 Hz tone with a mild L>R pan and a
        // hard-left high (intensity-region) tone.
        let low = (two_pi * 440.0 * t).sin();
        let high = (two_pi * HIGH_HZ * t).sin();
        let mut l = (low * 0.07 + high * 0.09) * scale;
        let mut r = (low * 0.05) * scale;
        // Transient burst every ~9216 samples (8 granules): a 64-sample
        // near-full-scale rectangular impulse that towers over the quiet
        // floor (subframe-vs-ambient ratio far past the 10× threshold).
        if i % 9216 < 64 {
            let imp = if i % 2 == 0 { 0.92 } else { -0.92 } * scale;
            l += imp;
            r += imp;
        }
        out.push(l.round().clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16);
        out.push(r.round().clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16);
    }
    out
}

fn goertzel_power(samples: &[i16], freq_hz: f32) -> f64 {
    let w = 2.0 * std::f64::consts::PI * f64::from(freq_hz) / f64::from(SR);
    let coeff = 2.0 * w.cos();
    let (mut s1, mut s2) = (0.0f64, 0.0f64);
    for &x in samples {
        let s0 = f64::from(x) + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    s1 * s1 + s2 * s2 - coeff * s1 * s2
}

fn encode_auto_ms_is(pcm: &[i16]) -> Vec<u8> {
    let mut enc =
        Mp3Encoder::new_joint_stereo_ms_is(BR, SR, START_SFB).expect("MS+intensity encoder");
    enc.enable_auto_block_type(DEFAULT_ATTACK_THRESHOLD)
        .expect("auto block-type on MS + intensity encoder (r307)");
    assert!(
        enc.intensity_stereo_enabled() && enc.ms_stereo_enabled() && enc.auto_block_type_enabled()
    );
    enc.push_samples(pcm).expect("push pcm");
    let mut out = Vec::new();
    enc.finish(&mut out).expect("finish");
    out
}

/// Intensity-only (no MS) auto-block-type encode (r308). `mode = '01'`,
/// `mode_extension` low bit set (intensity on) but the MS bit clear, so
/// `mode_extension.raw == 0b01`. The auto scheduler still mirrors its
/// channel-0 emission across both channels because intensity coupling
/// forces the §2.4.3.4.9 agreement OR-fold.
fn encode_auto_is_only(pcm: &[i16]) -> Vec<u8> {
    let mut enc = Mp3Encoder::new_joint_stereo_is(BR, SR, START_SFB).expect("intensity encoder");
    enc.enable_auto_block_type(DEFAULT_ATTACK_THRESHOLD)
        .expect("auto block-type on intensity-only encoder (r308)");
    assert!(
        enc.intensity_stereo_enabled() && !enc.ms_stereo_enabled() && enc.auto_block_type_enabled()
    );
    enc.push_samples(pcm).expect("push pcm");
    let mut out = Vec::new();
    enc.finish(&mut out).expect("finish");
    out
}

/// Spec-order stereo self-decode handling a per-granule mix of
/// long-family and pure-short blocks (the §2.4.3.4.8 reorder runs only on
/// the short granules, driven by the parsed side-info).
// The per-(gr, ch) loop variable doubles as the side-info / scalefactor /
// xr-pair subscript, mirroring the §2.4.1.7 main_data() walk.
#[allow(clippy::needless_range_loop)]
fn decode_mp3_stereo(bytes: &[u8]) -> (Vec<i16>, Vec<i16>) {
    let mut reservoir = Reservoir::new();
    let mut synth = [SynthState::new(), SynthState::new()];
    let mut imdct = [ImdctState::new(), ImdctState::new()];
    let mut out_l = Vec::new();
    let mut out_r = Vec::new();

    for frame in FrameWalker::new(bytes) {
        let hdr = parse_header(&frame.data[..4]).expect("header");
        let si = parse_side_info(&hdr, &frame.data[4..]).expect("si");
        let run = reservoir
            .assemble(
                usize::from(si.main_data_begin),
                &frame.data[4 + si.byte_len()..],
            )
            .expect("assemble");
        let fsf = decode_scalefactors(&hdr, &si, &run).expect("scalefactors");

        let mut bit_cursor = 0usize;
        for gr in 0..si.granule_count as usize {
            let mut xr_pair = [[0.0f32; NUM_LINES]; 2];
            for ch in 0..2usize {
                let gc = &si.granules[gr][ch];
                let short = gc.window_switching_flag
                    && gc.block_type == BlockType::Short
                    && !gc.mixed_block_flag;
                let part2_bits = mpeg1_part2_bits(gc.scalefac_compress, short);
                let mut r = MainDataReader::new(&run);
                let mut skip = bit_cursor + part2_bits;
                while skip >= 32 {
                    let _ = r.read(32);
                    skip -= 32;
                }
                if skip > 0 {
                    let _ = r.read(skip as u32);
                }
                let part3 = u32::from(gc.part2_3_length).saturating_sub(part2_bits as u32);
                let is = decode_huffman(&mut r, gc, part3, SR, hdr.version).expect("huffman");
                let sf = &fsf.granules[gr][ch];
                xr_pair[ch] = requantize(&is, gc, sf, SR, hdr.version);
                bit_cursor += gc.part2_3_length as usize;
            }

            // §2.4.3.4.8 reorder BEFORE the stereo stage (short granules
            // only; `reorder` is the identity for long-family blocks).
            for (ch, xr) in xr_pair.iter_mut().enumerate() {
                *xr = reorder(xr, &si.granules[gr][ch], SR, hdr.version);
            }

            if matches!(hdr.mode, ChannelMode::JointStereo) {
                let (l, r) = xr_pair.split_at_mut(1);
                process_stereo(
                    &mut l[0],
                    &mut r[0],
                    &fsf.granules[gr][1],
                    &si.granules[gr][1],
                    hdr.mode_extension,
                    SR,
                    hdr.version,
                );
            }

            for ch in 0..2usize {
                let gc = &si.granules[gr][ch];
                let xar = alias_reduce(&xr_pair[ch], gc);
                let st = imdct_granule(&xar, gc, &mut imdct[ch]);
                let pcm = synth_granule(&st, &mut synth[ch]);
                let sink = if ch == 0 { &mut out_l } else { &mut out_r };
                for &p in pcm.iter().take(PCM_PER_GRANULE) {
                    sink.push(pcm_f32_to_i16(p));
                }
            }
        }
    }
    (out_l, out_r)
}

#[test]
fn auto_ms_intensity_api_acceptance() {
    // Accepted on the MS+intensity encoder (r307).
    let mut ms = Mp3Encoder::new_joint_stereo_ms_is(BR, SR, START_SFB).expect("ms+is");
    assert!(ms.enable_auto_block_type(DEFAULT_ATTACK_THRESHOLD).is_ok());
    assert!(ms.auto_block_type_enabled());

    // Accepted on the auto-MS (picker) + intensity encoder too.
    let mut auto = Mp3Encoder::new_joint_stereo_auto_is(BR, SR, START_SFB).expect("auto+is");
    assert!(auto
        .enable_auto_block_type(DEFAULT_ATTACK_THRESHOLD)
        .is_ok());

    // Accepted on the intensity-only path too (r308): arming intensity
    // coupling forces the §2.4.3.4.9 channel-agreement OR-fold in the
    // scheduler walk (channel-0's emission mirrored across both
    // channels), so L/R block types stay consistent and the per-window /
    // long intensity coupling is well-defined.
    let mut is_only = Mp3Encoder::new_joint_stereo_is(BR, SR, START_SFB).expect("is-only");
    assert!(is_only
        .enable_auto_block_type(DEFAULT_ATTACK_THRESHOLD)
        .is_ok());
    assert!(is_only.auto_block_type_enabled() && is_only.intensity_stereo_enabled());

    // Mixed promotion stays rejected on the intensity-only path too (the
    // §2.4.3.4.10.3 carve-out bound is unwired regardless of MS).
    let mut is_only_mixed = Mp3Encoder::new_joint_stereo_is(BR, SR, START_SFB).expect("is-only");
    assert!(matches!(
        is_only_mixed.enable_auto_block_type_with_mixed(DEFAULT_ATTACK_THRESHOLD, 4.0),
        Err(StreamEncodeError::IntensityShortBlocksUnsupported)
    ));

    // Mixed promotion under intensity stays rejected (the mixed
    // carve-out bound is not wired).
    let mut ms2 = Mp3Encoder::new_joint_stereo_ms_is(BR, SR, START_SFB).expect("ms+is");
    assert!(matches!(
        ms2.enable_auto_block_type_with_mixed(DEFAULT_ATTACK_THRESHOLD, 4.0),
        Err(StreamEncodeError::IntensityShortBlocksUnsupported)
    ));
}

#[test]
fn auto_ms_intensity_stream_mixes_block_types() {
    // A two-second transient stream should drive the scheduler into both
    // long-family AND pure-short granules, each carrying the joint /
    // intensity header.
    let pcm = transient_stereo_pcm(SR as usize * 2);
    let out = encode_auto_ms_is(&pcm);

    let mut frames = 0usize;
    let mut saw_long = false;
    let mut saw_short = false;
    for frame in FrameWalker::new(&out) {
        let hdr = parse_header(&frame.data[..4]).expect("header");
        assert!(matches!(hdr.mode, ChannelMode::JointStereo));
        assert!(hdr.mode_extension.intensity_stereo, "intensity bit set");
        assert!(
            matches!(hdr.mode_extension.raw, 0b01 | 0b11),
            "mode_extension is '01' or '11', got {:#04b}",
            hdr.mode_extension.raw
        );
        let si = parse_side_info(&hdr, &frame.data[4..]).expect("si");
        for gr in 0..si.granule_count as usize {
            // §2.4.3.4.9 channel agreement: both channels share geometry.
            let g0 = &si.granules[gr][0];
            let g1 = &si.granules[gr][1];
            assert_eq!(g0.block_type, g1.block_type, "channels share block_type");
            assert_eq!(
                g0.window_switching_flag, g1.window_switching_flag,
                "channels share window_switching_flag"
            );
            assert_eq!(
                g0.mixed_block_flag, g1.mixed_block_flag,
                "channels share mixed_block_flag"
            );
            assert!(!g0.mixed_block_flag, "auto path emits no mixed blocks");
            if g0.block_type == BlockType::Short && g0.window_switching_flag {
                saw_short = true;
            } else if g0.block_type == BlockType::Long {
                saw_long = true;
            }
        }
        frames += 1;
    }
    assert!(frames > 0);
    assert!(
        saw_long,
        "scheduler must keep long blocks on steady stretches"
    );
    assert!(
        saw_short,
        "transient bursts must drive at least one short granule"
    );
}

#[test]
fn auto_ms_intensity_short_positions_in_range() {
    let pcm = transient_stereo_pcm(SR as usize * 2);
    let out = encode_auto_ms_is(&pcm);

    let mut reservoir = Reservoir::new();
    let mut checked_short = false;
    for (idx, frame) in FrameWalker::new(&out).enumerate() {
        let hdr = parse_header(&frame.data[..4]).expect("header");
        let si = parse_side_info(&hdr, &frame.data[4..]).expect("si");
        let run = reservoir
            .assemble(
                usize::from(si.main_data_begin),
                &frame.data[4 + si.byte_len()..],
            )
            .expect("assemble");
        if idx < 4 {
            continue; // filterbank warm-up
        }
        let fsf = decode_scalefactors(&hdr, &si, &run).expect("scalefactors");
        for gr in 0..si.granule_count as usize {
            let gc = &si.granules[gr][1];
            let sf = &fsf.granules[gr][1];
            if gc.window_switching_flag && gc.block_type == BlockType::Short && !gc.mixed_block_flag
            {
                checked_short = true;
                for sfb in 0..12 {
                    for win in 0..3 {
                        assert!(
                            sf.short[sfb][win] <= 7,
                            "frame {idx} gr {gr}: short pos {} out of range",
                            sf.short[sfb][win]
                        );
                    }
                }
            } else {
                for sfb in 0..21 {
                    assert!(
                        sf.long[sfb] <= 7,
                        "frame {idx} gr {gr}: long pos {} out of range",
                        sf.long[sfb]
                    );
                }
            }
        }
    }
    assert!(
        checked_short,
        "no short granule appeared to validate per-window positions"
    );
}

#[test]
fn auto_ms_intensity_self_decode_left_leaning() {
    let n = SR as usize * 2;
    let pcm = transient_stereo_pcm(n);
    let out = encode_auto_ms_is(&pcm);

    let (recon_l, recon_r) = decode_mp3_stereo(&out);
    let head = 4 * 1152 + 1057;
    assert!(recon_l.len() > head + 8192, "not enough steady PCM");
    let seg_l = &recon_l[head..head + 8192];
    let seg_r = &recon_r[head..head + 8192];

    // Hard-left high (intensity-region) tone reconstructs left-leaning.
    let hi_l = goertzel_power(seg_l, HIGH_HZ);
    let hi_r = goertzel_power(seg_r, HIGH_HZ);
    assert!(hi_l > 0.0, "high tone vanished on the left");
    let hi_ratio = (hi_l / hi_r.max(1.0)).sqrt();
    eprintln!("auto+ms+intensity 8 kHz reconstructed |L|/|R| = {hi_ratio:.3}");
    assert!(
        hi_ratio > 1.8,
        "coupled hard-left high tone should reconstruct left-leaning, got {hi_ratio}"
    );
}

#[test]
fn auto_ms_intensity_encode_is_bit_exact() {
    let pcm = transient_stereo_pcm(SR as usize);
    let a = encode_auto_ms_is(&pcm);
    let b = encode_auto_ms_is(&pcm);
    assert_eq!(a, b, "two encodes of the same PCM must be byte-identical");
}

// ---- r308: intensity-only (no MS) auto block-type ----

#[test]
fn auto_is_only_stream_mixes_block_types_and_agrees() {
    // The intensity-only auto path mixes long-family AND pure-short
    // granules; the §2.4.3.4.9 agreement is forced by the
    // intensity-armed OR-fold even though MS is OFF, so every granule's
    // channels share geometry and the header carries '01' (intensity on,
    // MS off).
    let pcm = transient_stereo_pcm(SR as usize * 2);
    let out = encode_auto_is_only(&pcm);

    let mut frames = 0usize;
    let mut saw_long = false;
    let mut saw_short = false;
    for frame in FrameWalker::new(&out) {
        let hdr = parse_header(&frame.data[..4]).expect("header");
        assert!(matches!(hdr.mode, ChannelMode::JointStereo));
        assert!(hdr.mode_extension.intensity_stereo, "intensity bit set");
        assert!(
            !hdr.mode_extension.ms_stereo,
            "MS bit clear (intensity-only)"
        );
        assert_eq!(
            hdr.mode_extension.raw, 0b01,
            "mode_extension '01' (intensity only), got {:#04b}",
            hdr.mode_extension.raw
        );
        let si = parse_side_info(&hdr, &frame.data[4..]).expect("si");
        for gr in 0..si.granule_count as usize {
            let g0 = &si.granules[gr][0];
            let g1 = &si.granules[gr][1];
            assert_eq!(g0.block_type, g1.block_type, "channels share block_type");
            assert_eq!(
                g0.window_switching_flag, g1.window_switching_flag,
                "channels share window_switching_flag"
            );
            assert_eq!(
                g0.mixed_block_flag, g1.mixed_block_flag,
                "channels share mixed_block_flag"
            );
            assert!(!g0.mixed_block_flag, "auto path emits no mixed blocks");
            if g0.block_type == BlockType::Short && g0.window_switching_flag {
                saw_short = true;
            } else if g0.block_type == BlockType::Long {
                saw_long = true;
            }
        }
        frames += 1;
    }
    assert!(frames > 0);
    assert!(
        saw_long,
        "scheduler must keep long blocks on steady stretches"
    );
    assert!(
        saw_short,
        "transient bursts must drive at least one short granule"
    );
}

#[test]
fn auto_is_only_short_positions_in_range() {
    let pcm = transient_stereo_pcm(SR as usize * 2);
    let out = encode_auto_is_only(&pcm);

    let mut reservoir = Reservoir::new();
    let mut checked_short = false;
    for (idx, frame) in FrameWalker::new(&out).enumerate() {
        let hdr = parse_header(&frame.data[..4]).expect("header");
        let si = parse_side_info(&hdr, &frame.data[4..]).expect("si");
        let run = reservoir
            .assemble(
                usize::from(si.main_data_begin),
                &frame.data[4 + si.byte_len()..],
            )
            .expect("assemble");
        if idx < 4 {
            continue; // filterbank warm-up
        }
        let fsf = decode_scalefactors(&hdr, &si, &run).expect("scalefactors");
        for gr in 0..si.granule_count as usize {
            let gc = &si.granules[gr][1];
            let sf = &fsf.granules[gr][1];
            if gc.window_switching_flag && gc.block_type == BlockType::Short && !gc.mixed_block_flag
            {
                checked_short = true;
                for sfb in 0..12 {
                    for win in 0..3 {
                        assert!(
                            sf.short[sfb][win] <= 7,
                            "frame {idx} gr {gr}: short pos {} out of range",
                            sf.short[sfb][win]
                        );
                    }
                }
            } else {
                for sfb in 0..21 {
                    assert!(
                        sf.long[sfb] <= 7,
                        "frame {idx} gr {gr}: long pos {} out of range",
                        sf.long[sfb]
                    );
                }
            }
        }
    }
    assert!(
        checked_short,
        "no short granule appeared to validate per-window positions"
    );
}

#[test]
fn auto_is_only_self_decode_left_leaning() {
    let n = SR as usize * 2;
    let pcm = transient_stereo_pcm(n);
    let out = encode_auto_is_only(&pcm);

    let (recon_l, recon_r) = decode_mp3_stereo(&out);
    let head = 4 * 1152 + 1057;
    assert!(recon_l.len() > head + 8192, "not enough steady PCM");
    let seg_l = &recon_l[head..head + 8192];
    let seg_r = &recon_r[head..head + 8192];

    let hi_l = goertzel_power(seg_l, HIGH_HZ);
    let hi_r = goertzel_power(seg_r, HIGH_HZ);
    assert!(hi_l > 0.0, "high tone vanished on the left");
    let hi_ratio = (hi_l / hi_r.max(1.0)).sqrt();
    eprintln!("auto+intensity-only 8 kHz reconstructed |L|/|R| = {hi_ratio:.3}");
    assert!(
        hi_ratio > 1.8,
        "coupled hard-left high tone should reconstruct left-leaning, got {hi_ratio}"
    );
}

#[test]
fn auto_is_only_encode_is_bit_exact() {
    let pcm = transient_stereo_pcm(SR as usize);
    let a = encode_auto_is_only(&pcm);
    let b = encode_auto_is_only(&pcm);
    assert_eq!(a, b, "two encodes of the same PCM must be byte-identical");
}
