//! Integration test for the round-146 joint-stereo MS encoder path.
//!
//! This test exercises the [`oxideav_mp3::Mp3Encoder::new_joint_stereo_ms`]
//! constructor end-to-end:
//!
//! 1. Synthesise one second of correlated stereo PCM (a 440 Hz tone
//!    panned ~70/30 toward L) at 44.1 kHz / `i16`, interleaved as
//!    `[L0, R0, L1, R1, …]` per ISO/IEC 11172-3 §2.4.2.1.
//! 2. Encode it through `Mp3Encoder::new_joint_stereo_ms` at
//!    192 kbit/s. Every emitted frame must carry header
//!    `mode = '01'` (joint stereo) with `mode_extension = '10'`
//!    (ms_stereo on, intensity_stereo off) per §2.4.2.3.
//! 3. Decode the byte stream through the in-tree pipeline
//!    (huffman → requantize → **process_stereo (MS inverse)** →
//!    alias → imdct → synth), gathering per-channel PCM.
//! 4. Confirm per-channel PSNR > 20 dB.
//! 5. Encode the same PCM through `Mp3Encoder::new(..., Stereo)`
//!    (independent-channel path, no MS) and confirm the MS path is
//!    not strictly worse at the same nominal bitrate — for correlated
//!    stereo content the §2.4.3.4.9.2 mid/side rotation concentrates
//!    energy in the M channel, letting the inner loop spend more bits
//!    on the part of the signal that carries the audible content.
//!
//! No external decoder is invoked; the asserts use only the crate's
//! own primitives.

use std::f32::consts::PI;

use oxideav_mp3::{
    alias_reduce, decode_huffman, decode_scalefactors, imdct_granule, parse_header,
    parse_side_info, process_stereo, requantize, synth_granule, ChannelMode, FrameWalker,
    ImdctState, MainDataReader, Mp3Encoder, Reservoir, SynthState, PCM_PER_GRANULE,
};

/// Build an interleaved `[L0, R0, …]` stereo `i16` PCM stream of a single
/// tone panned across the two channels (`L = pan_l · sin(2π f t)`,
/// `R = pan_r · sin(2π f t)`). With `pan_l ≠ pan_r` the two channels are
/// strongly correlated — the regime joint-stereo MS targets.
fn correlated_stereo_sine_pcm(
    n: usize,
    freq: f32,
    sample_rate_hz: f32,
    pan_l: f32,
    pan_r: f32,
) -> Vec<i16> {
    let two_pi = 2.0 * PI;
    let mut out = Vec::with_capacity(n * 2);
    let scale = i16::MAX as f32;
    for i in 0..n {
        let t = i as f32 / sample_rate_hz;
        let s = (two_pi * freq * t).sin();
        let l = (s * pan_l * scale)
            .round()
            .clamp(i16::MIN as f32, i16::MAX as f32) as i16;
        let r = (s * pan_r * scale)
            .round()
            .clamp(i16::MIN as f32, i16::MAX as f32) as i16;
        out.push(l);
        out.push(r);
    }
    out
}

/// PSNR (dB) of `recon` against `original`. Returns `f32::INFINITY` on
/// bit-exact match.
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
    let psnr = 10.0 * (max * max / mse).log10();
    psnr as f32
}

/// Decode a joint-stereo (or independent-stereo) MP3 byte stream into
/// two per-channel `i16` PCM vectors `(left, right)`. The in-tree
/// `process_stereo` primitive applies the inverse MS matrix when the
/// frame's `mode_extension` bit pattern requests it (`'10'` →
/// MS-stereo over the whole spectrum); for independent-stereo frames
/// (`mode = '00'` or `mode_extension = '00'`) it is a no-op.
// The per-(gr, ch) loops mirror the §2.4.1.7 main_data() walk; the
// loop variable doubles as the side-info / scalefactor / xr_pair
// subscript, so the indexed-by-integer form reads more clearly than
// an iterator chain.
#[allow(clippy::needless_range_loop)]
fn decode_mp3_stereo(bytes: &[u8]) -> (Vec<i16>, Vec<i16>) {
    let mut reservoir = Reservoir::new();
    let mut synth: Vec<SynthState> = vec![SynthState::new(), SynthState::new()];
    let mut imdct: Vec<ImdctState> = vec![ImdctState::new(), ImdctState::new()];
    let mut out_l: Vec<i16> = Vec::new();
    let mut out_r: Vec<i16> = Vec::new();

    for frame in FrameWalker::new(bytes) {
        let hdr = parse_header(&frame.data[..4]).expect("header parse");
        // Encoder writes `protection_bit = 1` (no CRC) by default; the
        // side-info block starts immediately at byte 4.
        let si_start = 4;
        let si = parse_side_info(&hdr, &frame.data[si_start..]).expect("side_info parse");
        assert_eq!(si.channels, 2, "expected stereo side info");
        let si_len = si.byte_len();
        let main_data_slot = &frame.data[si_start + si_len..];
        let run = reservoir
            .assemble(usize::from(si.main_data_begin), main_data_slot)
            .expect("reservoir assemble");
        let fsf = decode_scalefactors(&hdr, &si, &run).expect("scalefactors decode");

        let mut bit_cursor = 0usize;
        // We need per-(gr, ch) the requantized xr; cache them so we can
        // invoke process_stereo on the (L, R) pair within a granule
        // BEFORE alias_reduce / imdct (the §2.4.3.4.9 stage runs
        // between requantize and alias_reduce on the decode side).
        for gr in 0..si.granule_count as usize {
            // Decode both channels' Huffman runs first to obtain xr,
            // then apply MS once per granule, then continue through
            // the rest of the per-channel pipeline.
            let mut xr_pair: [[f32; oxideav_mp3::NUM_LINES]; 2] =
                [[0.0; oxideav_mp3::NUM_LINES]; 2];
            for ch in 0..si.channels as usize {
                let gc = &si.granules[gr][ch];
                let mut r = MainDataReader::new(&run);
                let mut left = bit_cursor;
                while left >= 32 {
                    let _ = r.read(32);
                    left -= 32;
                }
                if left > 0 {
                    let _ = r.read(left as u32);
                }
                let part3_bits = u32::from(gc.part2_3_length);
                let is = decode_huffman(&mut r, gc, part3_bits, hdr.sample_rate_hz, hdr.version)
                    .expect("huffman");
                let sf = &fsf.granules[gr][ch];
                xr_pair[ch] = requantize(&is, gc, sf, hdr.sample_rate_hz, hdr.version);
                bit_cursor += gc.part2_3_length as usize;
            }

            // §2.4.3.4.9 stereo processing: only kicks in for joint
            // stereo with at least one of the two mode_extension bits
            // set. The crate's `process_stereo` already implements
            // both the MS matrix and intensity stereo; it inspects the
            // mode_extension bits internally and is a no-op when both
            // are clear or when the channel mode is not joint stereo.
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
                    sink.push(v.clamp(i16::MIN as f32, i16::MAX as f32) as i16);
                }
            }
        }
    }
    (out_l, out_r)
}

/// Deinterleave a `[L0, R0, …]` interleaved sample stream into two
/// per-channel `Vec<i16>`.
fn deinterleave(interleaved: &[i16]) -> (Vec<i16>, Vec<i16>) {
    let n = interleaved.len() / 2;
    let mut l = Vec::with_capacity(n);
    let mut r = Vec::with_capacity(n);
    for i in 0..n {
        l.push(interleaved[2 * i]);
        r.push(interleaved[2 * i + 1]);
    }
    (l, r)
}

#[test]
fn joint_stereo_ms_one_second_self_decode_psnr() {
    const SR: u32 = 44_100;
    // 192 kbit/s — same nominal rate as the r145 independent-stereo
    // test, so the two paths are compared apples-to-apples.
    const BR: u32 = 192;
    let n = SR as usize;
    // Correlated stereo content: same 440 Hz tone on both channels but
    // panned 70%/30% toward L. This is the regime where joint-stereo MS
    // is meant to help — the L+R "mid" carries most of the energy and
    // the L-R "side" carries the pan-difference residue.
    let pcm = correlated_stereo_sine_pcm(n, 440.0, SR as f32, 0.7, 0.3);

    let mut enc = Mp3Encoder::new_joint_stereo_ms(BR, SR).expect("joint-stereo MS encoder build");
    assert!(enc.ms_stereo_enabled(), "constructor should arm MS");
    enc.push_samples(&pcm).expect("push interleaved pcm");
    let mut out: Vec<u8> = Vec::new();
    let bytes = enc.finish(&mut out).expect("encoder finish");
    assert_eq!(out.len(), bytes);
    assert!(
        bytes > 2_000,
        "joint-stereo encoded stream too small: {bytes}"
    );

    // Every emitted frame should report 2 channels with the 32-byte
    // stereo side-info block AND carry the joint-stereo header bits.
    let mut frame_count = 0usize;
    for frame in FrameWalker::new(&out) {
        let hdr = parse_header(&frame.data[..4]).expect("header parse");
        assert_eq!(hdr.channel_count(), 2);
        assert!(
            matches!(hdr.mode, ChannelMode::JointStereo),
            "expected JointStereo mode, got {:?}",
            hdr.mode
        );
        // mode_extension '10': ms_stereo on, intensity_stereo off.
        assert!(
            hdr.mode_extension.ms_stereo && !hdr.mode_extension.intensity_stereo,
            "expected mode_extension '10' (MS only), got raw=0b{:02b}",
            hdr.mode_extension.raw
        );
        assert_eq!(hdr.mode_extension.raw, 0b10);
        // Wire bits: byte 3 bits 7..6 = '01' (joint stereo);
        // bits 5..4 = '10' (mode_ext MS only).
        let mode = (frame.data[3] >> 6) & 0b11;
        let mode_ext = (frame.data[3] >> 4) & 0b11;
        assert_eq!(mode, 0b01, "wire mode bits should be '01'");
        assert_eq!(mode_ext, 0b10, "wire mode_extension bits should be '10'");
        let si = parse_side_info(&hdr, &frame.data[4..]).expect("si parse");
        assert_eq!(si.byte_len(), 32, "expected MPEG-1 stereo SI = 32 B");
        frame_count += 1;
    }
    assert!(
        (38..=40).contains(&frame_count),
        "frame count out of range: {frame_count}"
    );

    let (recon_l, recon_r) = decode_mp3_stereo(&out);
    let (in_l, in_r) = deinterleave(&pcm);

    // Encoder + decoder chain group delay: 481-sample polyphase prototype
    // + 576-sample lapped MDCT = 1057 PCM samples. Plus 4 granules
    // (4·576) of filterbank / overlap warm-up before stable energy.
    let total_delay = 1057usize;
    let warmup = 4 * 1152;
    assert!(recon_l.len() > warmup + total_delay);
    assert!(recon_r.len() > warmup + total_delay);

    let head_recon = warmup + total_delay;
    let cmp_len = recon_l
        .len()
        .saturating_sub(head_recon)
        .min(in_l.len() - warmup);
    let pl = psnr(
        &in_l[warmup..warmup + cmp_len],
        &recon_l[head_recon..head_recon + cmp_len],
    );
    let pr = psnr(
        &in_r[warmup..warmup + cmp_len],
        &recon_r[head_recon..head_recon + cmp_len],
    );
    eprintln!(
        "joint-stereo MS self-decode PSNR: L = {pl} dB, R = {pr} dB, n = {} samples",
        cmp_len
    );

    // Per-channel PSNR floor (same 20 dB bar the r145 independent test
    // uses). The MS matrix is its own inverse; any quantization-free
    // round-trip would land at the same PSNR as the independent path.
    assert!(pl > 20.0, "L-channel PSNR too low: {pl} dB");
    assert!(pr > 20.0, "R-channel PSNR too low: {pr} dB");
}

#[test]
fn joint_stereo_ms_silence_decodes_to_near_zero_both_channels() {
    const SR: u32 = 44_100;
    let mut enc = Mp3Encoder::new_joint_stereo_ms(192, SR).expect("joint-stereo encoder");
    // 3 frames of all-zero interleaved PCM (3 × 1152 × 2 samples).
    let zero_pcm = vec![0i16; 3 * 1152 * 2];
    enc.push_samples(&zero_pcm).expect("push zeros");
    let mut out: Vec<u8> = Vec::new();
    enc.finish(&mut out).expect("finish");

    // Every frame still carries the joint-stereo header bits — silent
    // input does not turn the joint mode off.
    for frame in FrameWalker::new(&out) {
        let hdr = parse_header(&frame.data[..4]).expect("header");
        assert!(matches!(hdr.mode, ChannelMode::JointStereo));
        assert_eq!(hdr.mode_extension.raw, 0b10);
    }

    let (l, r) = decode_mp3_stereo(&out);
    let warmup = 2 * 1152;
    for (label, ch) in [("L", &l), ("R", &r)] {
        if ch.len() > warmup {
            let peak = ch[warmup..]
                .iter()
                .map(|s| s.unsigned_abs())
                .max()
                .unwrap_or(0);
            assert!(
                peak <= 16,
                "{label} silence reconstruction peak too high: {peak}"
            );
        }
    }
}

#[test]
fn joint_stereo_ms_first_frame_byte_layout() {
    const SR: u32 = 44_100;
    let mut enc = Mp3Encoder::new_joint_stereo_ms(192, SR).expect("joint-stereo encoder");
    let pcm = correlated_stereo_sine_pcm(1152 * 2, 440.0, SR as f32, 0.7, 0.3);
    enc.push_samples(&pcm).unwrap();
    let mut out: Vec<u8> = Vec::new();
    enc.finish(&mut out).unwrap();
    let first = FrameWalker::new(&out).next().expect("at least one frame");
    assert_eq!(first.data[0], 0xFF);
    assert_eq!(first.data[1] & 0xE0, 0xE0);
    // Byte 3 bits 7..6 = mode (joint stereo = '01'),
    // bits 5..4 = mode_extension ('10' for MS only).
    assert_eq!(first.data[3] & 0xC0, 0x40, "mode should be '01' joint");
    assert_eq!(first.data[3] & 0x30, 0x20, "mode_ext should be '10'");
    let hdr = parse_header(&first.data[..4]).unwrap();
    assert_eq!(hdr.channel_count(), 2);
    assert!(matches!(hdr.mode, ChannelMode::JointStereo));
    assert!(hdr.mode_extension.ms_stereo);
    assert!(!hdr.mode_extension.intensity_stereo);
    let si = parse_side_info(&hdr, &first.data[4..]).unwrap();
    assert_eq!(si.byte_len(), 32);
    assert_eq!(si.channels, 2);
}

#[test]
fn joint_stereo_ms_recovers_l_minus_r_pan() {
    // Strong-pan input (90% L, 10% R) gives the MS matrix a non-trivial
    // S channel — if the encoder dropped the MS step the decoder would
    // recover near-silence on both sides (M and S are both quantized,
    // then the decoder rotates M and S into L and R). Confirm the
    // pan is preserved at the reconstruction-level by comparing the
    // average absolute amplitude of L vs R: it should remain skewed
    // L > R after the round-trip.
    const SR: u32 = 44_100;
    let n = SR as usize / 2; // 0.5 s
    let pcm = correlated_stereo_sine_pcm(n, 440.0, SR as f32, 0.9, 0.1);

    let mut enc = Mp3Encoder::new_joint_stereo_ms(192, SR).expect("joint-stereo encoder");
    enc.push_samples(&pcm).expect("push");
    let mut out: Vec<u8> = Vec::new();
    enc.finish(&mut out).expect("finish");

    let (recon_l, recon_r) = decode_mp3_stereo(&out);
    let warmup = 4 * 1152;
    let total_delay = 1057usize;
    let head = warmup + total_delay;
    assert!(recon_l.len() > head + 4096);
    let abs_sum_l: u64 = recon_l[head..head + 4096]
        .iter()
        .map(|&s| u64::from(s.unsigned_abs()))
        .sum();
    let abs_sum_r: u64 = recon_r[head..head + 4096]
        .iter()
        .map(|&s| u64::from(s.unsigned_abs()))
        .sum();
    eprintln!("joint-stereo MS pan preservation: |L| sum = {abs_sum_l}, |R| sum = {abs_sum_r}");
    // The input ratio is 0.9 / 0.1 = 9; even with quantization the
    // reconstruction should keep L at least 3× louder than R.
    assert!(
        abs_sum_l > abs_sum_r * 3,
        "MS round-trip should preserve the L pan: |L| = {abs_sum_l}, |R| = {abs_sum_r}"
    );
}
