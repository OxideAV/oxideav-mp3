//! Integration tests for the round-311 §2.4.3.4.10.3 **mixed-block
//! intensity-stereo** encoder path (ISO/IEC 11172-3:1993 +
//! ISO/IEC 13818-3:1997 §2.4.3.2 per-window bound).
//!
//! r302/r303/r305-r308 wired intensity coupling for long blocks and
//! pure-short blocks (fixed/adaptive/MS/auto). r311 adds the mixed-block
//! carve-out on the *intensity-only* (non-MS) force-mixed path
//! ([`oxideav_mp3::Mp3Encoder::force_mixed_blocks_for_testing`] on a
//! [`oxideav_mp3::Mp3Encoder::new_joint_stereo_is`] encoder).
//!
//! A mixed block (§2.4.3.4.10.3) transforms its lowest 2 polyphase
//! subbands (long bands 0..=7, lines 0..36) with a normal long window and
//! the upper 30 subbands (short bands 3..12) as short blocks. Intensity
//! coupling therefore runs in two regions: the long region couples on the
//! long-band walk (positions on `scalefac_l[sfb]`) and the short region
//! couples per window (positions on `scalefac_s[sfb][win]`) — exactly the
//! two regions the decoder's `process_stereo` rebuilds for a
//! `mixed_block_flag` granule.
//!
//! Covered end-to-end with in-tree primitives only:
//!
//! 1. **Header + side-info** — every audio frame carries `mode = '01'`
//!    (joint stereo), `mode_extension = '01'` (intensity only), and every
//!    granule-channel is a mixed block (`window_switching_flag = 1`,
//!    `block_type = Short`, `mixed_block_flag = 1`).
//! 2. **Wire-level positions** — the right channel's `scalefac_l[sfb]`
//!    (long region) and `scalefac_s[sfb][win]` (short region) carry legal
//!    stereo positions (`is_pos ∈ 0..=6`) or the illegal marker `7`
//!    (Annex G.2 c) in their respective coupled regions.
//! 3. **Positional fidelity** — after a spec-order self-decode
//!    (huffman → requantize → reorder → process_stereo → alias → imdct →
//!    synth), a hard-left high-frequency intensity tone reconstructs
//!    left-leaning.
//! 4. **Bit-exact stability** — same PCM in ⇒ same bytes out.

use std::f32::consts::PI;

use oxideav_mp3::{
    alias_reduce, decode_huffman, decode_scalefactors, imdct_granule, parse_header,
    parse_side_info, pcm_f32_to_i16, process_stereo, reorder, requantize, synth_granule, BlockType,
    ChannelMode, FrameWalker, ImdctState, MainDataReader, Mp3Encoder, Reservoir, SynthState,
    NUM_LINES, PCM_PER_GRANULE,
};

const SR: u32 = 44_100;
const BR: u32 = 192;
const START_SFB: usize = 8;

/// Intensity-region probe tone (well above the START_SFB bound).
const HIGH_HZ: f32 = 8_000.0;

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

/// Part2 (scalefactor) bit count of one MPEG-1 **mixed-block** granule
/// channel under `scalefac_compress` (slen1, slen2): 8 long bands carry
/// slen1 bits each, then short bands 3..12 carry slen1 (bands 3..6) /
/// slen2 (bands 6..12) per window over 3 windows. With
/// `scalefac_compress = 0` the count is zero; the intensity right channel
/// carries `scalefac_compress = 15` (slen1 = 4, slen2 = 3 ⇒
/// 8·4 + 3·3·4 + 6·3·3 = 122 bits).
fn mpeg1_mixed_part2_bits(scalefac_compress: u16) -> usize {
    let (slen1, slen2) = MPEG1_SLEN[(scalefac_compress & 0xF) as usize];
    let mut bits = 0usize;
    // 8 long bands (0..=7), slen1 each.
    bits += 8 * slen1 as usize;
    // Short bands 3..12, three windows; slen1 for 3..6, slen2 for 6..12.
    for sfb in 3..12usize {
        let slen = if sfb < 6 { slen1 } else { slen2 };
        bits += (slen as usize) * 3;
    }
    bits
}

fn two_tone_stereo_pcm(n: usize, hard_left_high: bool) -> Vec<i16> {
    let two_pi = 2.0 * PI;
    let scale = f32::from(i16::MAX);
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        let t = i as f32 / SR as f32;
        let low = (two_pi * 440.0 * t).sin();
        let high = (two_pi * HIGH_HZ * t).sin();
        let (hl, hr) = if hard_left_high {
            (high * 0.45, 0.0)
        } else {
            (high * 0.45, high * 0.15)
        };
        let l = (low * 0.35 + hl) * scale;
        let r = (low * 0.25 + hr) * scale;
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

fn encode_mixed_is(pcm: &[i16]) -> Vec<u8> {
    let mut enc = Mp3Encoder::new_joint_stereo_is(BR, SR, START_SFB).expect("intensity encoder");
    enc.force_mixed_blocks_for_testing(true)
        .expect("force-mixed on intensity-only encoder (r311)");
    assert!(enc.intensity_stereo_enabled() && enc.force_mixed_blocks_enabled());
    enc.push_samples(pcm).expect("push pcm");
    let mut out = Vec::new();
    enc.finish(&mut out).expect("finish");
    out
}

/// Spec-order stereo self-decode (with the §2.4.3.4.8 reorder that the
/// mixed-block path requires before the §2.4.3.4.9 stereo stage).
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
                let part2_bits = mpeg1_mixed_part2_bits(gc.scalefac_compress);
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

            // §2.4.3.4.8 reorder BEFORE the stereo stage (short region of
            // the mixed block; the long region passes through).
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
fn mixed_intensity_header_and_mixed_side_info() {
    let pcm = two_tone_stereo_pcm(SR as usize / 2, false);
    let out = encode_mixed_is(&pcm);

    let mut frames = 0usize;
    for frame in FrameWalker::new(&out) {
        let hdr = parse_header(&frame.data[..4]).expect("header");
        assert!(matches!(hdr.mode, ChannelMode::JointStereo));
        assert_eq!(hdr.mode_extension.raw, 0b01, "intensity-only header");
        assert!(hdr.mode_extension.intensity_stereo && !hdr.mode_extension.ms_stereo);
        let si = parse_side_info(&hdr, &frame.data[4..]).expect("si");
        for gr in 0..si.granule_count as usize {
            for ch in 0..si.channels as usize {
                let gc = &si.granules[gr][ch];
                assert!(
                    gc.window_switching_flag
                        && gc.block_type == BlockType::Short
                        && gc.mixed_block_flag,
                    "every granule-channel must be a mixed block"
                );
            }
        }
        frames += 1;
    }
    assert!(frames > 0);
}

#[test]
fn mixed_intensity_right_channel_carries_long_and_short_positions() {
    let pcm = two_tone_stereo_pcm(SR as usize / 2, true);
    let out = encode_mixed_is(&pcm);

    let mut reservoir = Reservoir::new();
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
            let sf = &fsf.granules[gr][1];
            // Long region (bands 0..8): legal positions or marker 7.
            for sfb in 0..8 {
                assert!(
                    sf.long[sfb] <= 7,
                    "frame {idx} gr {gr}: long sfb {sfb} pos {} out of range",
                    sf.long[sfb]
                );
            }
            // Short region (bands 3..12, per window): legal or marker 7.
            for sfb in 3..12 {
                for win in 0..3 {
                    assert!(
                        sf.short[sfb][win] <= 7,
                        "frame {idx} gr {gr}: short sfb {sfb} win {win} pos {} out of range",
                        sf.short[sfb][win]
                    );
                }
            }
        }
        if idx >= 8 {
            break;
        }
    }
}

#[test]
fn mixed_intensity_self_decode_preserves_left_image() {
    let n = SR as usize;
    let pcm = two_tone_stereo_pcm(n, true); // hard-left high tone
    let out = encode_mixed_is(&pcm);

    let (recon_l, recon_r) = decode_mp3_stereo(&out);
    let head = 4 * 1152 + 1057;
    assert!(recon_l.len() > head + 8192, "not enough steady PCM");
    let seg_l = &recon_l[head..head + 8192];
    let seg_r = &recon_r[head..head + 8192];

    let hi_l = goertzel_power(seg_l, HIGH_HZ);
    let hi_r = goertzel_power(seg_r, HIGH_HZ);
    assert!(hi_l > 0.0, "high tone vanished on the left");
    // Hard-left high tone coupled with is_pos = 6 reconstructs strongly
    // left-leaning: |L| >> |R| in the intensity region.
    let ratio = (hi_l / hi_r.max(1.0)).sqrt();
    eprintln!("mixed-intensity 8 kHz reconstructed |L|/|R| = {ratio:.3}");
    assert!(
        ratio > 2.0,
        "coupled hard-left high tone should reconstruct left-leaning, got {ratio}"
    );
}

#[test]
fn mixed_intensity_encode_is_bit_exact() {
    let pcm = two_tone_stereo_pcm(SR as usize / 4, false);
    let a = encode_mixed_is(&pcm);
    let b = encode_mixed_is(&pcm);
    assert_eq!(a, b, "two encodes of the same PCM must be byte-identical");
}
