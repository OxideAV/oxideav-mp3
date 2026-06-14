//! Integration tests for the round-284 §2.4.3.4.9.3 **intensity-stereo**
//! encoder path (ISO/IEC 11172-3:1993).
//!
//! Covered end-to-end, using only in-tree primitives plus optional
//! black-box CLI validators:
//!
//! 1. **Header + wire bits** — [`oxideav_mp3::Mp3Encoder::new_joint_stereo_is`]
//!    emits `mode = '01'` (joint stereo) with `mode_extension = '01'`
//!    (intensity_stereo on, ms_stereo off) per §2.4.2.3 on every audio
//!    frame; [`oxideav_mp3::Mp3Encoder::new_joint_stereo_ms_is`] emits
//!    `'11'` (both methods).
//! 2. **Wire-level scalefactor layout** — the right channel of an
//!    intensity frame carries per-band stereo positions (`is_pos ∈
//!    0..=6`) in the intensity region, the illegal-position marker `7`
//!    on the all-zero bands below the bound (Annex G.2 c), and zero
//!    spectral data above the bound (the §2.4.3.4.9.1 zero-part the
//!    decoder derives the bound from).
//! 3. **Positional / band-energy fidelity** — after a full self-decode
//!    (huffman → requantize → process_stereo → alias → imdct → synth),
//!    the reconstructed amplitude ratio of an intensity-coded tone
//!    matches the `tan(is_pos·π/12)` grid angle nearest the original
//!    pan, and a below-bound tone keeps its true L/R ratio.
//! 4. **Bit-exact stability** — encoding the same PCM twice produces
//!    identical bytes, and re-decoding the same stream twice produces
//!    identical PCM.
//! 5. **Per-frame auto picker** —
//!    [`oxideav_mp3::Mp3Encoder::new_joint_stereo_auto_is`] flips
//!    between `'11'` and `'01'` with the below-bound correlation.
//! 6. **Black-box cross-decode** — when `ffmpeg` and/or `mpg123` are on
//!    `PATH`, the emitted intensity stream is decoded by invoking the
//!    binary (its bytes only) and
//!    the recovered PCM must preserve the left-leaning stereo image.

use std::f32::consts::PI;
use std::process::Command;

use oxideav_mp3::{
    alias_reduce, decode_huffman, decode_scalefactors, imdct_granule, parse_header,
    parse_side_info, process_stereo, requantize, synth_granule, ChannelMode, FrameWalker,
    ImdctState, MainDataReader, Mp3Encoder, Reservoir, SynthState, MPEG1_SLEN, NUM_LINES,
    PCM_PER_GRANULE,
};

/// Part2 (scalefactor) bit count of one MPEG-1 **long-block** granule
/// channel, honouring §2.4.2.7 scfsi reuse. Bands 0..=10 carry `slen1`
/// bits and bands 11..=20 carry `slen2` bits; for granule 1, any of the
/// four scfsi_band groups (`{0..6}`, `{6..11}`, `{11..16}`, `{16..21}`)
/// whose `scfsi` flag is set is *not* retransmitted, so its bands
/// contribute zero bits. The intensity streams carry
/// `scalefac_compress = 15` on the right channel (the positions need ≥ 3
/// bits per band), so — unlike the zero-part2 MS-test streams — the
/// Huffman reader must skip past the scalefactor bits and the part3
/// budget must exclude them.
///
/// As of r301 the encoder auto-arms scfsi reuse, so an intensity frame's
/// long-block channels may share a group between granules; this counter
/// must therefore match the scfsi-aware part2 the encoder actually wrote
/// or the part3 skip desyncs.
fn mpeg1_long_part2_bits(scalefac_compress: u16, scfsi: &[bool; 4], gr: usize) -> usize {
    let (slen1, slen2) = MPEG1_SLEN[(scalefac_compress & 0xF) as usize];
    // (group bands, slen index): groups span bands {0..6}, {6..11},
    // {11..16}, {16..21}; slen1 covers bands 0..=10, slen2 bands 11..=20.
    const GROUPS: [(usize, usize); 4] = [(0, 6), (6, 11), (11, 16), (16, 21)];
    let mut bits = 0usize;
    for (g, &(lo, hi)) in GROUPS.iter().enumerate() {
        if gr == 1 && scfsi[g] {
            continue; // reused from granule 0 — not on the wire.
        }
        for sfb in lo..hi {
            let slen = if sfb <= 10 { slen1 } else { slen2 };
            bits += slen as usize;
        }
    }
    bits
}

const SR: u32 = 44_100;
const BR: u32 = 192;

/// First intensity-coded long scalefactor band used by these tests.
/// At 44.1 kHz the long-band layout starts band 11 at line 62
/// (≈ 2.37 kHz), leaving the 440 Hz probe tone far below the bound and
/// the 6 kHz probe tone far above it.
const START_SFB: usize = 11;

/// Below-bound probe tone (band 2 at 44.1 kHz).
const LOW_HZ: f32 = 440.0;
/// Intensity-region probe tone (line ≈ 157, long band 15 at 44.1 kHz).
const HIGH_HZ: f32 = 6_000.0;

/// Pan gains. The high tone's energy ratio is (0.48/0.12)² = 16, i.e.
/// an amplitude ratio of 4, which the Annex G.2 c) formula
/// `NINT((12/π)·arctan(4)) = NINT(5.06)` maps to `is_pos = 5`; the
/// decoder's §2.4.3.4.9.3 step 3 then reconstructs the ratio as
/// `tan(5π/12) ≈ 3.732`.
const LOW_PAN: (f32, f32) = (0.35, 0.25);
const HIGH_PAN: (f32, f32) = (0.48, 0.12);
const EXPECTED_HIGH_IS_POS: u8 = 5;
const EXPECTED_HIGH_RATIO: f32 = 3.732;

/// Interleaved `[L0, R0, …]` two-tone stereo PCM: a below-bound tone
/// and an intensity-region tone, each with its own pan.
fn two_tone_stereo_pcm(n: usize) -> Vec<i16> {
    let two_pi = 2.0 * PI;
    let scale = f32::from(i16::MAX);
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        let t = i as f32 / SR as f32;
        let low = (two_pi * LOW_HZ * t).sin();
        let high = (two_pi * HIGH_HZ * t).sin();
        let l = (low * LOW_PAN.0 + high * HIGH_PAN.0) * scale;
        let r = (low * LOW_PAN.1 + high * HIGH_PAN.1) * scale;
        out.push(l.round().clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16);
        out.push(r.round().clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16);
    }
    out
}

/// Goertzel power of `samples` at `freq_hz` (relative units — only
/// ratios between channels are asserted).
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

/// Decode a joint-stereo MP3 byte stream into two per-channel `i16` PCM
/// vectors `(left, right)` with the crate's own primitives. The
/// §2.4.3.4.9 stage ([`process_stereo`]) runs between requantize and
/// alias reduction and reads the frame's `mode_extension` bits, the
/// right channel's scalefactors (the intensity positions) and side
/// info — i.e. the full intensity / MS decode path.
// The per-(gr, ch) loops mirror the §2.4.1.7 main_data() walk; the loop
// variable doubles as the side-info / scalefactor / xr_pair subscript.
#[allow(clippy::needless_range_loop)]
fn decode_mp3_stereo(bytes: &[u8]) -> (Vec<i16>, Vec<i16>) {
    let mut reservoir = Reservoir::new();
    let mut synth: Vec<SynthState> = vec![SynthState::new(), SynthState::new()];
    let mut imdct: Vec<ImdctState> = vec![ImdctState::new(), ImdctState::new()];
    let mut out_l: Vec<i16> = Vec::new();
    let mut out_r: Vec<i16> = Vec::new();

    for frame in FrameWalker::new(bytes) {
        let hdr = parse_header(&frame.data[..4]).expect("header parse");
        let si = parse_side_info(&hdr, &frame.data[4..]).expect("side_info parse");
        assert_eq!(si.channels, 2, "expected stereo side info");
        let si_len = si.byte_len();
        let main_data_slot = &frame.data[4 + si_len..];
        let run = reservoir
            .assemble(usize::from(si.main_data_begin), main_data_slot)
            .expect("reservoir assemble");
        let fsf = decode_scalefactors(&hdr, &si, &run).expect("scalefactors decode");

        let mut bit_cursor = 0usize;
        for gr in 0..si.granule_count as usize {
            let mut xr_pair: [[f32; NUM_LINES]; 2] = [[0.0; NUM_LINES]; 2];
            for ch in 0..si.channels as usize {
                let gc = &si.granules[gr][ch];
                // Position the reader at the first huffmancodebits()
                // bit: past the previous tiles plus this tile's part2.
                let part2_bits = mpeg1_long_part2_bits(gc.scalefac_compress, &si.scfsi[ch], gr);
                let mut r = MainDataReader::new(&run);
                let mut left = bit_cursor + part2_bits;
                while left >= 32 {
                    let _ = r.read(32);
                    left -= 32;
                }
                if left > 0 {
                    let _ = r.read(left as u32);
                }
                let part3_bits = u32::from(gc.part2_3_length).saturating_sub(part2_bits as u32);
                let is = decode_huffman(&mut r, gc, part3_bits, hdr.sample_rate_hz, hdr.version)
                    .expect("huffman");
                let sf = &fsf.granules[gr][ch];
                xr_pair[ch] = requantize(&is, gc, sf, hdr.sample_rate_hz, hdr.version);
                bit_cursor += gc.part2_3_length as usize;
            }

            if matches!(hdr.mode, ChannelMode::JointStereo) {
                let (left_slice, right_slice) = xr_pair.split_at_mut(1);
                process_stereo(
                    &mut left_slice[0],
                    &mut right_slice[0],
                    &fsf.granules[gr][1],
                    &si.granules[gr][1],
                    hdr.mode_extension,
                    hdr.sample_rate_hz,
                    hdr.version,
                );
            }

            for ch in 0..si.channels as usize {
                let gc = &si.granules[gr][ch];
                let xar = alias_reduce(&xr_pair[ch], gc);
                let subband_time = imdct_granule(&xar, gc, &mut imdct[ch]);
                let pcm_f32 = synth_granule(&subband_time, &mut synth[ch]);
                let sink = if ch == 0 { &mut out_l } else { &mut out_r };
                for &p in pcm_f32.iter().take(PCM_PER_GRANULE) {
                    let v = p * f32::from(i16::MAX);
                    sink.push(v.clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16);
                }
            }
        }
    }
    (out_l, out_r)
}

/// PSNR (dB) of `recon` against `original`.
fn psnr(original: &[i16], recon: &[i16]) -> f32 {
    assert_eq!(original.len(), recon.len(), "psnr length mismatch");
    let mut sse = 0.0f64;
    for (a, b) in original.iter().zip(recon.iter()) {
        let d = f64::from(*a) - f64::from(*b);
        sse += d * d;
    }
    let mse = sse / original.len() as f64;
    if mse == 0.0 {
        return f32::INFINITY;
    }
    let max = f64::from(i16::MAX);
    (10.0 * (max * max / mse).log10()) as f32
}

fn deinterleave(interleaved: &[i16]) -> (Vec<i16>, Vec<i16>) {
    let n = interleaved.len() / 2;
    let mut l = Vec::with_capacity(n);
    let mut r = Vec::with_capacity(n);
    for pair in interleaved.chunks_exact(2) {
        l.push(pair[0]);
        r.push(pair[1]);
    }
    (l, r)
}

fn encode_is_stream(pcm: &[i16]) -> Vec<u8> {
    let mut enc = Mp3Encoder::new_joint_stereo_is(BR, SR, START_SFB).expect("intensity encoder");
    assert!(enc.intensity_stereo_enabled());
    enc.push_samples(pcm).expect("push pcm");
    let mut out = Vec::new();
    enc.finish(&mut out).expect("finish");
    out
}

/// 44.1 kHz MPEG-1 long-band starts (Table B.8), used by the wire-level
/// assertions to map lines to scalefactor bands.
const LONG_STARTS_44: [usize; 22] = [
    0, 4, 8, 12, 16, 20, 24, 30, 36, 44, 52, 62, 74, 90, 110, 134, 162, 196, 238, 288, 342, 418,
];

#[test]
fn intensity_header_and_wire_bits() {
    let pcm = two_tone_stereo_pcm(SR as usize / 2);
    let out = encode_is_stream(&pcm);

    let mut frame_count = 0usize;
    for frame in FrameWalker::new(&out) {
        let hdr = parse_header(&frame.data[..4]).expect("header parse");
        assert_eq!(hdr.channel_count(), 2);
        assert!(
            matches!(hdr.mode, ChannelMode::JointStereo),
            "expected JointStereo mode, got {:?}",
            hdr.mode
        );
        // mode_extension '01': intensity_stereo on, ms_stereo off.
        assert!(
            hdr.mode_extension.intensity_stereo && !hdr.mode_extension.ms_stereo,
            "expected mode_extension '01' (intensity only), got raw=0b{:02b}",
            hdr.mode_extension.raw
        );
        assert_eq!(hdr.mode_extension.raw, 0b01);
        // Wire bits: byte 3 bits 7..6 = '01' (joint stereo);
        // bits 5..4 = '01' (mode_ext intensity only).
        assert_eq!((frame.data[3] >> 6) & 0b11, 0b01, "wire mode bits");
        assert_eq!((frame.data[3] >> 4) & 0b11, 0b01, "wire mode_ext bits");
        let si = parse_side_info(&hdr, &frame.data[4..]).expect("si parse");
        assert_eq!(si.byte_len(), 32, "expected MPEG-1 stereo SI = 32 B");
        frame_count += 1;
    }
    assert!(
        (18..=20).contains(&frame_count),
        "frame count out of range: {frame_count}"
    );
}

/// Wire-level §2.4.3.4.9.3 layout of a steady-state frame: the right
/// channel's intensity-region scalefactors are legal positions (with
/// the probe band carrying the Annex G.2 c) derived value), its
/// spectral data above the bound is the all-zero zero-part, and the
/// zero bands below the bound carry the marker 7.
#[test]
fn intensity_right_channel_scalefactors_carry_positions() {
    let pcm = two_tone_stereo_pcm(SR as usize / 2);
    let out = encode_is_stream(&pcm);

    let mut reservoir = Reservoir::new();
    for (idx, frame) in FrameWalker::new(&out).enumerate() {
        let hdr = parse_header(&frame.data[..4]).expect("header parse");
        let si = parse_side_info(&hdr, &frame.data[4..]).expect("si parse");
        let run = reservoir
            .assemble(
                usize::from(si.main_data_begin),
                &frame.data[4 + si.byte_len()..],
            )
            .expect("reservoir assemble");
        if idx < 4 {
            continue; // skip filterbank warm-up frames
        }
        let fsf = decode_scalefactors(&hdr, &si, &run).expect("scalefactors decode");

        // The §2.4.1.7 main_data walk is continuous across the frame's
        // (gr, ch) tiles; keep one running bit cursor.
        let mut bit_cursor = 0usize;
        for gr in 0..si.granule_count as usize {
            // Recover the right channel's quantized spectrum to locate
            // the decoder-visible zero-part.
            let mut right_is = [0i32; NUM_LINES];
            for ch in 0..2usize {
                let gc = &si.granules[gr][ch];
                let part2_bits = mpeg1_long_part2_bits(gc.scalefac_compress, &si.scfsi[ch], gr);
                let mut r = MainDataReader::new(&run);
                let mut skip = bit_cursor + part2_bits;
                while skip >= 32 {
                    let _ = r.read(32);
                    skip -= 32;
                }
                if skip > 0 {
                    let _ = r.read(skip as u32);
                }
                let part3_bits = u32::from(gc.part2_3_length).saturating_sub(part2_bits as u32);
                let is = decode_huffman(&mut r, gc, part3_bits, SR, hdr.version).expect("huffman");
                if ch == 1 {
                    right_is = is;
                }
                bit_cursor += gc.part2_3_length as usize;
            }

            // (a) Zero-part: no right-channel spectral data at or above
            // the intensity bound.
            let bound_line = LONG_STARTS_44[START_SFB];
            assert!(
                right_is[bound_line..].iter().all(|&v| v == 0),
                "frame {idx} gr {gr}: right channel has spectral data above the intensity bound"
            );

            let sf = &fsf.granules[gr][1];
            // (b) Intensity region: legal positions only (7 = illegal
            // marker must not appear at/above the bound), with the
            // 6 kHz probe band carrying the derived position.
            for sfb in START_SFB..21 {
                assert!(
                    sf.long[sfb] <= 6,
                    "frame {idx} gr {gr}: sfb {sfb} carries illegal is_pos {}",
                    sf.long[sfb]
                );
            }
            assert_eq!(
                sf.long[15], EXPECTED_HIGH_IS_POS,
                "frame {idx} gr {gr}: 6 kHz band position"
            );

            // (c) Below-bound zero tail: every band after the last
            // non-zero right-channel line and before the bound carries
            // the Annex G.2 c) marker 7 so it is not intensity-decoded.
            let last_nz = right_is[..bound_line].iter().rposition(|&v| v != 0);
            let zero_tail_from = match last_nz {
                None => 0,
                Some(line) => {
                    (0..21)
                        .find(|&sfb| line < LONG_STARTS_44[sfb + 1])
                        .unwrap_or(20)
                        + 1
                }
            };
            // The range may be empty when the right channel's quantized
            // content reaches the band just below the bound; the
            // high-tone-only scenario below guarantees a non-empty
            // marker run.
            for sfb in zero_tail_from..START_SFB {
                assert_eq!(
                    sf.long[sfb], 7,
                    "frame {idx} gr {gr}: zero band {sfb} below the bound must carry the marker 7"
                );
            }
        }
        if idx >= 8 {
            break; // a handful of steady-state frames is plenty
        }
    }
}

/// Annex G.2 c) marker run: with a **hard-left** probe (right channel
/// all-zero PCM) the right channel's zero-part spans the whole
/// spectrum, so every band below the chosen bound must carry the
/// illegal-position marker `7` — a decoder deriving the bound from the
/// zero-part (§2.4.3.4.9.1) would otherwise intensity-decode them with
/// a bogus position — and every intensity band derives the `E_R → 0`
/// limit position `6` (all-left).
#[test]
fn intensity_marker_seven_fills_silent_below_bound_bands() {
    let two_pi = 2.0 * PI;
    let scale = f32::from(i16::MAX);
    let n = SR as usize / 2;
    let mut pcm = Vec::with_capacity(n * 2);
    for i in 0..n {
        let t = i as f32 / SR as f32;
        let low = (two_pi * LOW_HZ * t).sin();
        let high = (two_pi * HIGH_HZ * t).sin();
        pcm.push(((low * 0.35 + high * 0.48) * scale).round() as i16);
        pcm.push(0i16); // hard-left: the right channel is exactly silent
    }
    let out = encode_is_stream(&pcm);

    let mut reservoir = Reservoir::new();
    for (idx, frame) in FrameWalker::new(&out).enumerate() {
        let hdr = parse_header(&frame.data[..4]).expect("header parse");
        let si = parse_side_info(&hdr, &frame.data[4..]).expect("si parse");
        let run = reservoir
            .assemble(
                usize::from(si.main_data_begin),
                &frame.data[4 + si.byte_len()..],
            )
            .expect("reservoir assemble");
        if idx < 4 {
            continue;
        }
        let fsf = decode_scalefactors(&hdr, &si, &run).expect("scalefactors decode");

        let mut bit_cursor = 0usize;
        for gr in 0..si.granule_count as usize {
            let mut right_is = [0i32; NUM_LINES];
            for ch in 0..2usize {
                let gc = &si.granules[gr][ch];
                let part2_bits = mpeg1_long_part2_bits(gc.scalefac_compress, &si.scfsi[ch], gr);
                let mut r = MainDataReader::new(&run);
                let mut skip = bit_cursor + part2_bits;
                while skip >= 32 {
                    let _ = r.read(32);
                    skip -= 32;
                }
                if skip > 0 {
                    let _ = r.read(skip as u32);
                }
                let part3_bits = u32::from(gc.part2_3_length).saturating_sub(part2_bits as u32);
                let is = decode_huffman(&mut r, gc, part3_bits, SR, hdr.version).expect("huffman");
                if ch == 1 {
                    right_is = is;
                }
                bit_cursor += gc.part2_3_length as usize;
            }

            // The hard-left probe leaves the right channel with no
            // spectral data anywhere: the zero-part spans the whole
            // spectrum and the marker run covers every band below the
            // bound.
            assert!(
                right_is.iter().all(|&v| v == 0),
                "frame {idx} gr {gr}: hard-left probe leaked right-channel spectral data"
            );
            let sf = &fsf.granules[gr][1];
            for sfb in 0..START_SFB {
                assert_eq!(
                    sf.long[sfb], 7,
                    "frame {idx} gr {gr}: silent band {sfb} below the bound must carry the marker 7"
                );
            }
            // Every intensity band derives the E_R → 0 limit position
            // 6 (all-left), including the tone band.
            for sfb in START_SFB..21 {
                assert_eq!(
                    sf.long[sfb], 6,
                    "frame {idx} gr {gr}: hard-left intensity band {sfb} position"
                );
            }
        }
        if idx >= 8 {
            break;
        }
    }
}

/// Band-energy / positional fidelity after a full self-decode: the
/// intensity-coded 6 kHz tone reconstructs at the `tan(is_pos·π/12)`
/// grid angle, the below-bound 440 Hz tone keeps its true pan, and the
/// per-channel PSNR stays at parity with an independent-stereo encode
/// of the same PCM at the same bitrate (the fixed-gain quantizer — not
/// the intensity coupling — sets the absolute noise floor on this
/// dense two-tone probe).
#[test]
fn intensity_self_decode_positional_fidelity_and_psnr() {
    let n = SR as usize;
    let pcm = two_tone_stereo_pcm(n);
    let out = encode_is_stream(&pcm);

    let (recon_l, recon_r) = decode_mp3_stereo(&out);
    let (in_l, in_r) = deinterleave(&pcm);

    // Encoder + decoder chain group delay (481-sample polyphase
    // prototype + 576-sample lapped MDCT) plus filterbank warm-up.
    let total_delay = 1057usize;
    let warmup = 4 * 1152;
    let head = warmup + total_delay;
    assert!(recon_l.len() > head + 8192);

    // Positional fidelity of the intensity-coded tone: the decoder
    // rebuilds L/R from the combined magnitude and is_pos = 5, so the
    // reconstructed amplitude ratio at 6 kHz must sit at the grid angle
    // tan(5π/12) ≈ 3.732 (the original pan ratio was 4.0).
    let seg_l = &recon_l[head..head + 8192];
    let seg_r = &recon_r[head..head + 8192];
    let high_ratio = (goertzel_power(seg_l, HIGH_HZ) / goertzel_power(seg_r, HIGH_HZ)).sqrt();
    eprintln!(
        "intensity 6 kHz reconstructed |L|/|R| = {high_ratio:.3} (grid {EXPECTED_HIGH_RATIO})"
    );
    assert!(
        (f64::from(EXPECTED_HIGH_RATIO) * 0.75..=f64::from(EXPECTED_HIGH_RATIO) * 1.33)
            .contains(&high_ratio),
        "intensity positional ratio off the tan(is_pos·π/12) grid: {high_ratio}"
    );

    // Below-bound fidelity: the 440 Hz tone is coded as plain L/R and
    // must keep its true 0.35/0.25 = 1.4 pan ratio.
    let low_ratio = (goertzel_power(seg_l, LOW_HZ) / goertzel_power(seg_r, LOW_HZ)).sqrt();
    eprintln!("below-bound 440 Hz reconstructed |L|/|R| = {low_ratio:.3} (true 1.4)");
    assert!(
        (1.15..=1.7).contains(&low_ratio),
        "below-bound pan ratio drifted: {low_ratio}"
    );

    // Whole-signal per-channel PSNR: parity with the independent
    // ChannelMode::Stereo encode of the same PCM at the same bitrate
    // (within 1.5 dB per channel), plus an absolute sanity floor. The
    // right channel typically *gains* PSNR under intensity — its
    // high-band content is rebuilt from the (better-quantized)
    // combined magnitude instead of its own low-amplitude coding.
    let mut baseline_enc = Mp3Encoder::new(BR, SR, ChannelMode::Stereo).expect("baseline encoder");
    baseline_enc.push_samples(&pcm).expect("push baseline");
    let mut baseline_out = Vec::new();
    baseline_enc
        .finish(&mut baseline_out)
        .expect("finish baseline");
    let (base_l, base_r) = decode_mp3_stereo(&baseline_out);

    let cmp_len = recon_l.len().saturating_sub(head).min(in_l.len() - warmup);
    let pl = psnr(
        &in_l[warmup..warmup + cmp_len],
        &recon_l[head..head + cmp_len],
    );
    let pr = psnr(
        &in_r[warmup..warmup + cmp_len],
        &recon_r[head..head + cmp_len],
    );
    let bl = psnr(
        &in_l[warmup..warmup + cmp_len],
        &base_l[head..head + cmp_len],
    );
    let br = psnr(
        &in_r[warmup..warmup + cmp_len],
        &base_r[head..head + cmp_len],
    );
    eprintln!(
        "intensity self-decode PSNR: L = {pl} dB (baseline {bl}), R = {pr} dB (baseline {br})"
    );
    assert!(pl > 12.0, "L-channel PSNR below sanity floor: {pl} dB");
    assert!(pr > 18.0, "R-channel PSNR below sanity floor: {pr} dB");
    assert!(
        pl >= bl - 1.5,
        "L-channel PSNR regressed vs independent stereo: {pl} dB vs {bl} dB"
    );
    assert!(
        pr >= br - 1.5,
        "R-channel PSNR regressed vs independent stereo: {pr} dB vs {br} dB"
    );
}

/// `new_joint_stereo_ms_is`: every frame carries `mode_extension = '11'`,
/// the below-bound spectrum is MS-coded (and recovered), the intensity
/// region keeps positional fidelity.
#[test]
fn ms_plus_intensity_mode_extension_11_roundtrip() {
    let n = SR as usize;
    let pcm = two_tone_stereo_pcm(n);
    let mut enc = Mp3Encoder::new_joint_stereo_ms_is(BR, SR, START_SFB).expect("ms+is encoder");
    assert!(enc.intensity_stereo_enabled() && enc.ms_stereo_enabled());
    enc.push_samples(&pcm).expect("push");
    let mut out = Vec::new();
    enc.finish(&mut out).expect("finish");

    for frame in FrameWalker::new(&out) {
        let hdr = parse_header(&frame.data[..4]).expect("header parse");
        assert!(matches!(hdr.mode, ChannelMode::JointStereo));
        assert_eq!(
            hdr.mode_extension.raw, 0b11,
            "expected mode_extension '11' (MS + intensity)"
        );
        assert!(hdr.mode_extension.intensity_stereo && hdr.mode_extension.ms_stereo);
    }

    let (recon_l, recon_r) = decode_mp3_stereo(&out);
    let (in_l, in_r) = deinterleave(&pcm);
    let head = 4 * 1152 + 1057;
    assert!(recon_l.len() > head + 8192);

    let seg_l = &recon_l[head..head + 8192];
    let seg_r = &recon_r[head..head + 8192];
    let high_ratio = (goertzel_power(seg_l, HIGH_HZ) / goertzel_power(seg_r, HIGH_HZ)).sqrt();
    assert!(
        (f64::from(EXPECTED_HIGH_RATIO) * 0.75..=f64::from(EXPECTED_HIGH_RATIO) * 1.33)
            .contains(&high_ratio),
        "MS+intensity positional ratio off-grid: {high_ratio}"
    );
    let low_ratio = (goertzel_power(seg_l, LOW_HZ) / goertzel_power(seg_r, LOW_HZ)).sqrt();
    assert!(
        (1.15..=1.7).contains(&low_ratio),
        "MS-coded below-bound pan ratio drifted: {low_ratio}"
    );

    // PSNR parity with the full-spectrum MS encode of the same PCM
    // (within 1.5 dB per channel) plus an absolute sanity floor — the
    // fixed-gain quantizer sets the absolute noise floor on this dense
    // two-tone probe, for the MS-only reference exactly as for the
    // MS + intensity path under test.
    let mut ms_enc = Mp3Encoder::new_joint_stereo_ms(BR, SR).expect("ms-only encoder");
    ms_enc.push_samples(&pcm).expect("push ms baseline");
    let mut ms_out = Vec::new();
    ms_enc.finish(&mut ms_out).expect("finish ms baseline");
    let (base_l, base_r) = decode_mp3_stereo(&ms_out);

    let warmup = 4 * 1152;
    let cmp_len = recon_l.len().saturating_sub(head).min(in_l.len() - warmup);
    let pl = psnr(
        &in_l[warmup..warmup + cmp_len],
        &recon_l[head..head + cmp_len],
    );
    let pr = psnr(
        &in_r[warmup..warmup + cmp_len],
        &recon_r[head..head + cmp_len],
    );
    let bl = psnr(
        &in_l[warmup..warmup + cmp_len],
        &base_l[head..head + cmp_len],
    );
    let br = psnr(
        &in_r[warmup..warmup + cmp_len],
        &base_r[head..head + cmp_len],
    );
    eprintln!(
        "MS+intensity self-decode PSNR: L = {pl} dB (MS-only {bl}), R = {pr} dB (MS-only {br})"
    );
    assert!(pl > 12.0, "L-channel PSNR below sanity floor: {pl} dB");
    assert!(pr > 14.0, "R-channel PSNR below sanity floor: {pr} dB");
    assert!(
        pl >= bl - 1.5,
        "L-channel PSNR regressed vs MS-only: {pl} dB vs {bl} dB"
    );
    assert!(
        pr >= br - 1.5,
        "R-channel PSNR regressed vs MS-only: {pr} dB vs {br} dB"
    );
}

/// The auto picker keeps the intensity bit set on both outcomes:
/// correlated below-bound content elects `'11'` (MS + intensity),
/// anti-phase below-bound content elects `'01'` (intensity only).
#[test]
fn auto_intensity_picker_flips_between_11_and_01() {
    let two_pi = 2.0 * PI;
    let scale = f32::from(i16::MAX);
    let n = SAMPLES_PER_FRAME * 8;

    // Correlated: identical 440 Hz on both channels + intensity-region
    // tone. The below-bound side-energy fraction is ~0 → MS elected.
    let mut correlated = Vec::with_capacity(n * 2);
    // Anti-phase: L = -R at 440 Hz. Below-bound side-energy fraction is
    // ~1 → MS rejected, intensity stays on.
    let mut anti = Vec::with_capacity(n * 2);
    for i in 0..n {
        let t = i as f32 / SR as f32;
        let low = (two_pi * LOW_HZ * t).sin() * 0.4;
        let high = (two_pi * HIGH_HZ * t).sin();
        let high_l = high * 0.3;
        let high_r = high * 0.1;
        correlated.push(((low + high_l) * scale).round() as i16);
        correlated.push(((low + high_r) * scale).round() as i16);
        anti.push(((low + high_l) * scale).round() as i16);
        anti.push(((-low + high_r) * scale).round() as i16);
    }

    for (label, pcm, expect_raw) in [
        ("correlated", &correlated, 0b11u8),
        ("anti-phase", &anti, 0b01u8),
    ] {
        let mut enc =
            Mp3Encoder::new_joint_stereo_auto_is(BR, SR, START_SFB).expect("auto+is encoder");
        enc.push_samples(pcm).expect("push");
        let mut out = Vec::new();
        enc.finish(&mut out).expect("finish");
        for (idx, frame) in FrameWalker::new(&out).enumerate() {
            let hdr = parse_header(&frame.data[..4]).expect("header parse");
            assert!(matches!(hdr.mode, ChannelMode::JointStereo));
            assert!(
                hdr.mode_extension.intensity_stereo,
                "{label}: intensity bit must stay set on every frame"
            );
            if idx < 2 {
                continue; // cold-start frames carry filterbank warm-up
            }
            assert_eq!(
                hdr.mode_extension.raw, expect_raw,
                "{label}: frame {idx} picked the wrong mode_extension"
            );
        }
    }
}

const SAMPLES_PER_FRAME: usize = 1152;

/// Bit-exact stability: same PCM in → same bytes out, and the same
/// bytes re-decode to identical PCM.
#[test]
fn intensity_encode_and_redecode_are_bit_exact() {
    let pcm = two_tone_stereo_pcm(SR as usize / 4);
    let a = encode_is_stream(&pcm);
    let b = encode_is_stream(&pcm);
    assert_eq!(a, b, "two encodes of the same PCM must be byte-identical");

    let (l1, r1) = decode_mp3_stereo(&a);
    let (l2, r2) = decode_mp3_stereo(&a);
    assert_eq!(l1, l2, "re-decode must be bit-exact (left)");
    assert_eq!(r1, r2, "re-decode must be bit-exact (right)");
}

/// Locate a CLI decoder on `PATH`; `None` if not present. The binary
/// is invoked as a black box on the emitted bytes only.
fn binary_on_path(name: &str, probe_arg: &str) -> Option<std::path::PathBuf> {
    let out = Command::new(name).arg(probe_arg).output().ok()?;
    if out.status.success() {
        Some(std::path::PathBuf::from(name))
    } else {
        None
    }
}

/// Shared assertions for an externally-decoded interleaved s16 PCM
/// buffer: non-trivial length, audible signal, and the left-leaning
/// stereo image of both probe tones preserved through the intensity
/// coupling.
fn assert_external_pcm_preserves_image(pcm_bytes: &[u8], label: &str) {
    assert!(
        pcm_bytes.len() > 4 * SAMPLES_PER_FRAME,
        "{label} produced too little PCM ({} B)",
        pcm_bytes.len()
    );
    let samples: Vec<i16> = pcm_bytes
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]))
        .collect();
    let (l, r) = deinterleave(&samples);
    // Skip the decoder's warm-up/delay region, keep a steady segment.
    let head = 4 * SAMPLES_PER_FRAME;
    assert!(l.len() > head + 8192, "{label}: not enough steady PCM");
    let seg_l = &l[head..head + 8192];
    let seg_r = &r[head..head + 8192];
    let peak = seg_l
        .iter()
        .chain(seg_r)
        .map(|s| s.unsigned_abs())
        .max()
        .unwrap();
    assert!(
        peak > 1000,
        "{label}: decoded signal implausibly quiet ({peak})"
    );

    // The intensity-coded 6 kHz tone must come back left-leaning near
    // the tan(5π/12) grid angle; tolerate generous decoder variation.
    let high_ratio = (goertzel_power(seg_l, HIGH_HZ) / goertzel_power(seg_r, HIGH_HZ)).sqrt();
    eprintln!("{label}: external 6 kHz |L|/|R| = {high_ratio:.3}");
    assert!(
        (2.0..=6.0).contains(&high_ratio),
        "{label}: intensity image lost in external decode (ratio {high_ratio})"
    );
    // The below-bound 440 Hz tone keeps its mild 1.4 pan.
    let low_ratio = (goertzel_power(seg_l, LOW_HZ) / goertzel_power(seg_r, LOW_HZ)).sqrt();
    eprintln!("{label}: external 440 Hz |L|/|R| = {low_ratio:.3}");
    assert!(
        (1.1..=1.8).contains(&low_ratio),
        "{label}: below-bound pan drifted in external decode (ratio {low_ratio})"
    );
}

#[test]
fn external_decoders_accept_intensity_stream() {
    let pcm = two_tone_stereo_pcm(SR as usize);
    let out = encode_is_stream(&pcm);

    let dir = std::env::temp_dir();
    let mp3_path = dir.join(format!("oxideav_mp3_is_{}.mp3", std::process::id()));
    std::fs::write(&mp3_path, &out).expect("write mp3");

    let mut ran_any = false;
    if let Some(ffmpeg) = binary_on_path("ffmpeg", "-version") {
        let pcm_path = dir.join(format!("oxideav_mp3_is_{}.pcm", std::process::id()));
        let status = Command::new(&ffmpeg)
            .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
            .arg(&mp3_path)
            .args(["-f", "s16le"])
            .arg(&pcm_path)
            .status()
            .expect("run ffmpeg");
        assert!(
            status.success(),
            "ffmpeg failed to decode the intensity stream"
        );
        let decoded = std::fs::read(&pcm_path).expect("read decoded pcm");
        let _ = std::fs::remove_file(&pcm_path);
        assert_external_pcm_preserves_image(&decoded, "ffmpeg");
        ran_any = true;
    }
    if let Some(mpg123) = binary_on_path("mpg123", "--version") {
        // `-s` writes raw 16-bit native-endian interleaved PCM to
        // stdout (little-endian on every CI target this crate builds).
        let output = Command::new(&mpg123)
            .args(["-q", "-s"])
            .arg(&mp3_path)
            .output()
            .expect("run mpg123");
        assert!(
            output.status.success(),
            "mpg123 failed to decode the intensity stream"
        );
        assert_external_pcm_preserves_image(&output.stdout, "mpg123");
        ran_any = true;
    }
    let _ = std::fs::remove_file(&mp3_path);
    if !ran_any {
        eprintln!("neither ffmpeg nor mpg123 on PATH; skipping black-box cross-decode");
    }
}

/// Stereo PCM where the below-bound 440 Hz tone is panned (true L/R
/// information) and the intensity-region 6 kHz tone has the requested
/// high-band stereo character: `mono` ⇒ identical on both channels (low
/// side-energy fraction, so the adaptive chooser couples the high tail),
/// otherwise anti-phase `L = -R` (side-energy fraction ≈ 1, so the
/// chooser leaves it independent).
fn adaptive_probe_pcm(n: usize, high_mono: bool) -> Vec<i16> {
    let two_pi = 2.0 * PI;
    let scale = f32::from(i16::MAX);
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        let t = i as f32 / SR as f32;
        let low = (two_pi * LOW_HZ * t).sin();
        let high = (two_pi * HIGH_HZ * t).sin();
        // Below-bound pan kept identical to the fixed-bound tests.
        let l_low = low * LOW_PAN.0;
        let r_low = low * LOW_PAN.1;
        let (l_high, r_high) = if high_mono {
            (high * 0.3, high * 0.3)
        } else {
            (high * 0.3, -high * 0.3)
        };
        let l = (l_low + l_high) * scale;
        let r = (r_low + r_high) * scale;
        out.push(l.round().clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16);
        out.push(r.round().clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16);
    }
    out
}

/// Adaptive intensity bound (round 302). With a near-mono high tail the
/// per-granule chooser couples the high bands: the right channel's
/// 6 kHz energy is reconstructed from the transmitted left magnitude
/// and the §2.4.3.4.9.3 stereo position, so its decoded amplitude
/// approaches the left channel's. With an anti-phase high tail the
/// chooser raises the bound (couples nothing up there), so the right
/// channel keeps its real, opposite-phase 6 kHz tone. In both cases the
/// below-bound 440 Hz pan must survive, the bytes must be deterministic,
/// and every frame stays under one `mode = '01'` / `mode_extension =
/// '01'` header.
#[test]
fn adaptive_intensity_bound_couples_low_stereo_high_tail() {
    let n = SR as usize;

    // --- near-mono high tail: high band qualifies and is coupled ---
    let pcm_mono = adaptive_probe_pcm(n, true);
    let mut enc =
        Mp3Encoder::new_joint_stereo_auto_is_adaptive(BR, SR, START_SFB).expect("adaptive encoder");
    assert!(enc.intensity_stereo_enabled());
    assert_eq!(enc.intensity_auto_threshold(), Some(0.25));
    enc.push_samples(&pcm_mono).expect("push mono-tail");
    let mut out_mono = Vec::new();
    enc.finish(&mut out_mono).expect("finish mono-tail");

    // Determinism: encoding the same PCM twice is byte-identical.
    let mut enc2 =
        Mp3Encoder::new_joint_stereo_auto_is_adaptive(BR, SR, START_SFB).expect("adaptive encoder");
    enc2.push_samples(&pcm_mono).expect("push mono-tail 2");
    let mut out_mono2 = Vec::new();
    enc2.finish(&mut out_mono2).expect("finish mono-tail 2");
    assert_eq!(out_mono, out_mono2, "adaptive encode is not deterministic");

    for frame in FrameWalker::new(&out_mono) {
        let hdr = parse_header(&frame.data[..4]).expect("header parse");
        assert!(matches!(hdr.mode, ChannelMode::JointStereo));
        assert_eq!(
            hdr.mode_extension.raw, 0b01,
            "expected intensity-only header"
        );
    }

    let (recon_l, recon_r) = decode_mp3_stereo(&out_mono);
    let head = 4 * 1152 + 1057;
    assert!(recon_l.len() > head + 8192);
    let seg_l = &recon_l[head..head + 8192];
    let seg_r = &recon_r[head..head + 8192];
    // Coupled high band: the reconstructed right-channel 6 kHz energy is
    // driven by the transmitted left magnitude and the derived stereo
    // position, so it is non-trivial (the band is *not* a zero-part on
    // the right after decode) and the channels are close in level (a
    // near-mono original → is_pos ≈ 3 → balanced reconstruction).
    let hi_l = goertzel_power(seg_l, HIGH_HZ);
    let hi_r = goertzel_power(seg_r, HIGH_HZ);
    assert!(
        hi_r > 0.0 && hi_l > 0.0,
        "high tone vanished after coupling"
    );
    let coupled_ratio = (hi_l / hi_r).sqrt();
    assert!(
        (0.4..=2.5).contains(&coupled_ratio),
        "coupled near-mono high tone should reconstruct roughly balanced, got {coupled_ratio}"
    );
    // Below-bound 440 Hz pan preserved (true L/R, untouched by coupling).
    let low_ratio = (goertzel_power(seg_l, LOW_HZ) / goertzel_power(seg_r, LOW_HZ)).sqrt();
    assert!(
        (1.15..=1.7).contains(&low_ratio),
        "below-bound pan drifted under adaptive coupling: {low_ratio}"
    );

    // --- anti-phase high tail: chooser leaves it independent ---
    let pcm_anti = adaptive_probe_pcm(n, false);
    let mut enc =
        Mp3Encoder::new_joint_stereo_auto_is_adaptive(BR, SR, START_SFB).expect("adaptive encoder");
    enc.push_samples(&pcm_anti).expect("push anti-tail");
    let mut out_anti = Vec::new();
    enc.finish(&mut out_anti).expect("finish anti-tail");

    let (recon_l_a, recon_r_a) = decode_mp3_stereo(&out_anti);
    let seg_l_a = &recon_l_a[head..head + 8192];
    let seg_r_a = &recon_r_a[head..head + 8192];
    // Not coupled: the right channel keeps its real anti-phase 6 kHz
    // tone, so both channels carry comparable high-tone energy (an
    // intensity-coupled band would have zeroed the right channel and
    // forced a left-leaning reconstruction instead).
    let hi_l_a = goertzel_power(seg_l_a, HIGH_HZ);
    let hi_r_a = goertzel_power(seg_r_a, HIGH_HZ);
    let anti_ratio = (hi_l_a / hi_r_a).sqrt();
    assert!(
        (0.5..=2.0).contains(&anti_ratio),
        "anti-phase high tone should stay independent (balanced level), got {anti_ratio}"
    );
}
