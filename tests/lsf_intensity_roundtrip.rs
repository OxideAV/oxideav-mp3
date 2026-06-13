//! Integration tests for the round-286 **LSF (MPEG-2 / MPEG-2.5)
//! intensity-stereo** encoder path (ISO/IEC 13818-3 §2.4.3.2).
//!
//! An LSF intensity frame differs from the MPEG-1 §2.4.3.4.9.3 format
//! on the wire: the right channel routes its 9-bit `scalefac_compress`
//! through `int_scalefac_compress = scalefac_compress >> 1` with its
//! own §2.4.3.2 partition tables, an `intensity_scale = scalefac_compress
//! % 2` selector, and a power-law `i0` reconstruction ladder (step-4/5
//! replacement) rather than the MPEG-1 `tan` grid. The encoder writes
//! `scalefac_compress = 258` on the intensity-right channel:
//! `258 >> 1 = 129 < 180` ⇒ `slen = (3, 3, 3, 0)` over the long-block
//! partition `nr_of_sfb = (7, 7, 7, 0)` (3 bits on every one of the 21
//! long bands), `258 % 2 = 0` ⇒ `intensity_scale = 0`. With `slen = 3`
//! the maximum band value `7` is the §13818-3 illegal-position marker
//! the decoder tests for.
//!
//! Every test encodes synthetic stereo PCM and decodes the result with
//! the crate's own decode primitives.

use std::f32::consts::PI;

use oxideav_mp3::{
    alias_reduce, decode_huffman, decode_scalefactors, imdct_granule, lsf_scale_params,
    parse_header, parse_side_info, process_stereo, reorder, requantize, synth_granule, BlockType,
    ChannelMode, FrameWalker, GranuleChannel, ImdctState, MainDataReader, Mp3Encoder, MpegVersion,
    Reservoir, StreamEncodeError, SynthState, PCM_PER_GRANULE,
};

/// Synthesise an `n`-sample mono `i16` sine tone.
fn sine_pcm(n: usize, freq_hz: f32, sample_rate_hz: f32, amp: f32) -> Vec<i16> {
    let two_pi = 2.0 * PI;
    let scale = amp * (i16::MAX as f32);
    (0..n)
        .map(|i| {
            let t = i as f32 / sample_rate_hz;
            let s = (two_pi * freq_hz * t).sin() * scale;
            s.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16
        })
        .collect()
}

/// PSNR (dB) of `recon` against `original` over equal-length slices.
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

/// LSF part2 (scalefactor) bit count for one granule-channel.
/// `is_intensity_right` selects the §2.4.3.2 right-channel partition
/// (`int_scalefac_compress = scalefac_compress >> 1`).
fn lsf_part2_bits(gc: &GranuleChannel, is_intensity_right: bool) -> u32 {
    let p = lsf_scale_params(
        gc.scalefac_compress,
        gc.block_type,
        gc.mixed_block_flag,
        is_intensity_right,
    );
    (0..4)
        .map(|i| u32::from(p.slen[i]) * u32::from(p.nr_of_sfb[i]))
        .sum()
}

/// Decode an LSF intensity-stereo Layer III stream into per-channel
/// `i16` PCM, using the crate's own primitives. The right channel of an
/// intensity frame (`mode_extension` `'01'` / `'11'`, `ch == 1`) is
/// part2-skipped with its §2.4.3.2 right-channel partition, and
/// `process_stereo` applies the LSF intensity factors.
fn decode_lsf_stream(bytes: &[u8]) -> Vec<Vec<i16>> {
    let mut reservoir = Reservoir::new();
    let mut synth: Vec<SynthState> = Vec::new();
    let mut imdct: Vec<ImdctState> = Vec::new();
    let mut out: Vec<Vec<i16>> = Vec::new();

    for frame in FrameWalker::new(bytes) {
        let hdr = parse_header(&frame.data[..4]).expect("header parse");
        assert!(
            hdr.version.is_lsf(),
            "expected LSF frame, got {:?}",
            hdr.version
        );
        let intensity = hdr.mode == ChannelMode::JointStereo && hdr.mode_extension.intensity_stereo;
        let si_start = 4 + if hdr.crc_protected { 2 } else { 0 };
        let si = parse_side_info(&hdr, &frame.data[si_start..]).expect("side_info parse");
        assert_eq!(si.granule_count, 1, "LSF frame must carry one granule");
        assert!(si.lsf, "side info must parse as LSF layout");
        let nch = si.channels as usize;
        if out.is_empty() {
            out = vec![Vec::new(); nch];
            synth = (0..nch).map(|_| SynthState::new()).collect();
            imdct = (0..nch).map(|_| ImdctState::new()).collect();
        }
        let si_len = si.byte_len();
        let main_slot = &frame.data[si_start + si_len..];
        let run = reservoir
            .assemble(usize::from(si.main_data_begin), main_slot)
            .expect("reservoir assemble");
        let fsf = decode_scalefactors(&hdr, &si, &run).expect("scalefactors decode");

        let mut bit_cursor = 0usize;
        let gr = 0usize;
        let mut xr_per_ch: Vec<[f32; 576]> = (0..nch).map(|_| [0.0; 576]).collect();
        for (ch, xr_slot) in xr_per_ch.iter_mut().enumerate() {
            let gc = &si.granules[gr][ch];
            let is_intensity_right = intensity && ch == 1;
            let part2_bits = lsf_part2_bits(gc, is_intensity_right) as usize;
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
            let is = decode_huffman(&mut r, gc, part3_bits, hdr.sample_rate_hz, hdr.version)
                .expect("huffman");
            let sf = &fsf.granules[gr][ch];
            let xr = requantize(&is, gc, sf, hdr.sample_rate_hz, hdr.version);
            *xr_slot = reorder(&xr, gc, hdr.sample_rate_hz, hdr.version);
            bit_cursor += gc.part2_3_length as usize;
        }
        if nch == 2 && hdr.mode == ChannelMode::JointStereo {
            let (l_split, r_split) = xr_per_ch.split_at_mut(1);
            process_stereo(
                &mut l_split[0],
                &mut r_split[0],
                &fsf.granules[gr][1],
                &si.granules[gr][1],
                hdr.mode_extension,
                hdr.sample_rate_hz,
                hdr.version,
            );
        }
        for (ch, xr_ch) in xr_per_ch.iter().enumerate() {
            let gc = &si.granules[gr][ch];
            let xar = alias_reduce(xr_ch, gc);
            let subband_time = imdct_granule(&xar, gc, &mut imdct[ch]);
            let pcm_f32 = synth_granule(&subband_time, &mut synth[ch]);
            for &p in pcm_f32.iter().take(PCM_PER_GRANULE) {
                let v = p * f32::from(i16::MAX);
                out[ch].push(v.clamp(i16::MIN as f32, i16::MAX as f32) as i16);
            }
        }
    }
    out
}

/// Encoder + decoder fixed group delay plus a filterbank warm-up margin,
/// in 576-sample LSF granules (mirrors `lsf_encoder_roundtrip.rs`).
const WARMUP_GRANULES: usize = 8;
const TOTAL_DELAY: usize = 1057;

const SR: u32 = 22_050;
const START_SFB: usize = 11;

/// Interleave two mono channels into `[L0, R0, L1, R1, ...]`.
fn interleave(left: &[i16], right: &[i16]) -> Vec<i16> {
    assert_eq!(left.len(), right.len());
    let mut v = Vec::with_capacity(left.len() * 2);
    for i in 0..left.len() {
        v.push(left[i]);
        v.push(right[i]);
    }
    v
}

/// The §2.4.3.2 LSF intensity constructors arm coupling and never reject
/// an LSF version (the r285 `LsfUnsupported` rejection is lifted), while
/// still rejecting an out-of-range start band.
#[test]
fn lsf_intensity_constructors_arm_state() {
    let enc = Mp3Encoder::new_joint_stereo_is(128, SR, START_SFB).expect("LSF IS encoder");
    assert!(enc.intensity_stereo_enabled());
    assert_eq!(enc.intensity_start_sfb(), Some(START_SFB));

    let enc = Mp3Encoder::new_joint_stereo_ms_is(128, SR, 8).expect("LSF MS+IS encoder");
    assert!(enc.intensity_stereo_enabled());

    // Out-of-range start band still rejected on LSF.
    assert!(matches!(
        Mp3Encoder::new_joint_stereo_is(128, SR, 0),
        Err(StreamEncodeError::InvalidIntensityStartSfb { start_sfb: 0 })
    ));
    assert!(matches!(
        Mp3Encoder::new_joint_stereo_is(128, SR, 21),
        Err(StreamEncodeError::InvalidIntensityStartSfb { start_sfb: 21 })
    ));
}

/// Every intensity frame carries `mode = '01'`, `mode_extension`
/// intensity bit set, LSF framing, and the right channel writes
/// `scalefac_compress = 258` (the §2.4.3.2 right-channel partition).
#[test]
fn lsf_intensity_wire_layout() {
    let n = SR as usize / 2;
    // Pan a high tone hard-left so the intensity region couples.
    let left = sine_pcm(n, 6_000.0, SR as f32, 0.48);
    let right = sine_pcm(n, 6_000.0, SR as f32, 0.12);
    let pcm = interleave(&left, &right);

    let mut enc = Mp3Encoder::new_joint_stereo_is(128, SR, START_SFB).expect("LSF IS encoder");
    enc.push_samples(&pcm).expect("push pcm");
    let mut out: Vec<u8> = Vec::new();
    enc.finish(&mut out).expect("finish");

    let mut frames = 0usize;
    for f in FrameWalker::new(&out) {
        let hdr = parse_header(&f.data[..4]).unwrap();
        assert_eq!(hdr.version, MpegVersion::Mpeg2);
        assert_eq!(hdr.mode, ChannelMode::JointStereo);
        assert!(
            hdr.mode_extension.intensity_stereo,
            "intensity bit must be set"
        );
        let si = parse_side_info(&hdr, &f.data[4..]).unwrap();
        assert_eq!(si.granule_count, 1);
        let right_gc = &si.granules[0][1];
        assert_eq!(
            right_gc.scalefac_compress, 258,
            "LSF intensity-right scalefac_compress must be 258"
        );
        // The §2.4.3.2 right-channel partition is (3,3,3,0)/(7,7,7,0).
        let p = lsf_scale_params(258, BlockType::Long, false, true);
        assert_eq!(p.slen, [3, 3, 3, 0]);
        assert_eq!(p.nr_of_sfb, [7, 7, 7, 0]);
        assert!(!p.intensity_scale, "258 % 2 == 0 ⇒ intensity_scale = 0");
        frames += 1;
    }
    assert!(frames > 10, "expected many frames, got {frames}");
}

/// Full self-decode round-trip: the left-panned high tone survives the
/// intensity coupling (left channel reconstructs near full amplitude,
/// the right channel a fraction of it), and the below-bound region keeps
/// its independent L/R image. PSNR is asserted on the dominant channel.
#[test]
fn lsf_intensity_self_decode_roundtrip() {
    let n = SR as usize;
    // Below-bound: independent tones. Above-bound: a hard-left high tone.
    let mut left = sine_pcm(n, 440.0, SR as f32, 0.30);
    let right = sine_pcm(n, 440.0, SR as f32, 0.28);
    let high_l = sine_pcm(n, 6_000.0, SR as f32, 0.30);
    let high_r = sine_pcm(n, 6_000.0, SR as f32, 0.08);
    let mut right = right;
    for i in 0..n {
        left[i] = left[i].saturating_add(high_l[i]);
        right[i] = right[i].saturating_add(high_r[i]);
    }
    let pcm = interleave(&left, &right);

    let mut enc = Mp3Encoder::new_joint_stereo_is(128, SR, START_SFB).expect("LSF IS encoder");
    enc.push_samples(&pcm).expect("push pcm");
    let mut out: Vec<u8> = Vec::new();
    enc.finish(&mut out).expect("finish");

    let recon = decode_lsf_stream(&out);
    assert_eq!(recon.len(), 2);
    let warmup = WARMUP_GRANULES * PCM_PER_GRANULE;
    let head = warmup + TOTAL_DELAY;
    // The left channel carries the combined intensity magnitude, so it
    // reconstructs with good fidelity against the original left.
    let r = &recon[0];
    let cmp = r
        .len()
        .saturating_sub(head)
        .min(left.len().saturating_sub(warmup));
    assert!(cmp > 4096, "comparison window too small: {cmp}");
    let p = psnr(&left[warmup..warmup + cmp], &r[head..head + cmp]);
    eprintln!("LSF intensity left-channel self-decode PSNR = {p} dB over {cmp} samples");
    assert!(p > 18.0, "left PSNR too low: {p} dB");
}

/// Bit-exact determinism: encoding the same PCM twice yields identical
/// bytes, and re-decoding the same stream twice yields identical PCM.
#[test]
fn lsf_intensity_deterministic() {
    let n = SR as usize / 2;
    let left = sine_pcm(n, 5_000.0, SR as f32, 0.45);
    let right = sine_pcm(n, 5_000.0, SR as f32, 0.15);
    let pcm = interleave(&left, &right);

    let encode = || {
        let mut enc = Mp3Encoder::new_joint_stereo_is(128, SR, START_SFB).expect("enc");
        enc.push_samples(&pcm).expect("push");
        let mut out = Vec::new();
        enc.finish(&mut out).expect("finish");
        out
    };
    let a = encode();
    let b = encode();
    assert_eq!(a, b, "encoder must be deterministic");

    let da = decode_lsf_stream(&a);
    let db = decode_lsf_stream(&a);
    assert_eq!(da, db, "decoder must be deterministic");
}

/// The MS-below-bound + intensity-above-bound combined mode (`'11'`)
/// round-trips on LSF too: both channels decode without panic and the
/// dominant left channel keeps reasonable fidelity.
#[test]
fn lsf_ms_intensity_combined_roundtrip() {
    let n = SR as usize;
    // Correlated below-bound (MS habitat) + hard-left high tone.
    let mut left = sine_pcm(n, 330.0, SR as f32, 0.40);
    let mut right = sine_pcm(n, 330.0, SR as f32, 0.36);
    let high_l = sine_pcm(n, 7_000.0, SR as f32, 0.25);
    let high_r = sine_pcm(n, 7_000.0, SR as f32, 0.06);
    for i in 0..n {
        left[i] = left[i].saturating_add(high_l[i]);
        right[i] = right[i].saturating_add(high_r[i]);
    }
    let pcm = interleave(&left, &right);

    let mut enc = Mp3Encoder::new_joint_stereo_ms_is(128, SR, 8).expect("LSF MS+IS encoder");
    enc.push_samples(&pcm).expect("push");
    let mut out: Vec<u8> = Vec::new();
    enc.finish(&mut out).expect("finish");

    for f in FrameWalker::new(&out) {
        let hdr = parse_header(&f.data[..4]).unwrap();
        assert_eq!(hdr.version, MpegVersion::Mpeg2);
        assert!(hdr.mode_extension.intensity_stereo);
        assert!(hdr.mode_extension.ms_stereo);
    }

    let recon = decode_lsf_stream(&out);
    let warmup = WARMUP_GRANULES * PCM_PER_GRANULE;
    let head = warmup + TOTAL_DELAY;
    let r = &recon[0];
    let cmp = r
        .len()
        .saturating_sub(head)
        .min(left.len().saturating_sub(warmup));
    let p = psnr(&left[warmup..warmup + cmp], &r[head..head + cmp]);
    eprintln!("LSF MS+IS left-channel self-decode PSNR = {p} dB");
    assert!(p > 15.0, "left PSNR too low: {p} dB");
}

/// MPEG-2.5 (the extension rates) inherits the LSF intensity framing.
#[test]
fn mpeg25_intensity_roundtrip() {
    const SR25: u32 = 11_025;
    let n = SR25 as usize;
    let left = sine_pcm(n, 3_000.0, SR25 as f32, 0.45);
    let right = sine_pcm(n, 3_000.0, SR25 as f32, 0.12);
    let pcm = interleave(&left, &right);

    let mut enc =
        Mp3Encoder::new_joint_stereo_is(32, SR25, START_SFB).expect("MPEG-2.5 IS encoder");
    enc.push_samples(&pcm).expect("push");
    let mut out: Vec<u8> = Vec::new();
    enc.finish(&mut out).expect("finish");

    for f in FrameWalker::new(&out) {
        let hdr = parse_header(&f.data[..4]).unwrap();
        assert_eq!(hdr.version, MpegVersion::Mpeg25);
        assert!(hdr.mode_extension.intensity_stereo);
    }

    let recon = decode_lsf_stream(&out);
    assert_eq!(recon.len(), 2);
    // Just assert it decodes a non-trivial amount of audio per channel.
    assert!(recon[0].len() > 8192);
    assert!(recon[1].len() > 8192);
}
