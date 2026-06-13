//! Integration tests for the r287 LSF widening of the §C.1.5.2 auto
//! block-type scheduler ([`oxideav_mp3::Mp3Encoder::enable_auto_block_type`]).
//!
//! Through r286 the auto block-type frame walk in
//! `assemble_frame_with_lookahead` was hard-shaped to the MPEG-1
//! two-granule geometry: it classified granule 0, granule 1 and one
//! lookahead granule per channel, then stepped the
//! [`oxideav_mp3`] block-type state machine exactly twice per frame.
//! An LSF (MPEG-2 16 / 22.05 / 24 kHz, MPEG-2.5 8 / 11.025 / 12 kHz)
//! frame carries a **single** 576-sample granule (ISO/IEC 13818-3
//! §2.4.3.2), so the encoder rejected `enable_auto_block_type` with
//! `LsfUnsupported`.
//!
//! r287 generalised the walk over `ngr ∈ {1, 2}`: granule `g` is fed
//! `(attack[g], attack[g + 1])`, where `attack[ngr]` is the next
//! frame's leading granule (the §C.1.5.2 lookahead). On LSF that is a
//! single scheduler step per frame with the next frame's granule as the
//! lookahead; on MPEG-1 it reproduces the prior two-step walk verbatim.
//! The §2.4.3.4.10.3 window-switching geometry is identical across
//! versions — only the per-frame granule count differs.
//!
//! These tests drive synthetic LSF PCM through the encoder with auto
//! block-type on and decode the result with the crate's **own** decode
//! primitives only (no external library).

use std::f32::consts::PI;

use oxideav_mp3::{
    alias_reduce, decode_huffman, decode_scalefactors, imdct_granule, lsf_scale_params,
    parse_header, parse_side_info, process_stereo, reorder, requantize, synth_granule, BlockType,
    ChannelMode, FrameWalker, GranuleChannel, ImdctState, MainDataReader, Mp3Encoder, Reservoir,
    SynthState, DEFAULT_ATTACK_THRESHOLD, PCM_PER_GRANULE,
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

/// "Click train": silence punctuated by short full-scale bursts. Each
/// burst trips the [`AttackDetector`](oxideav_mp3) so the §C.1.5.2
/// scheduler fires its LONG → START → SHORT → END → LONG sequence.
fn click_train_pcm(total_samples: usize, click_period: usize) -> Vec<i16> {
    let mut out = vec![0i16; total_samples];
    let mut pos = click_period;
    while pos + 64 < total_samples {
        for j in 0..64 {
            out[pos + j] = if j % 2 == 0 { 30_000 } else { -30_000 };
        }
        pos += click_period;
    }
    out
}

/// LSF part2 (scalefactor) bit count for one granule-channel, derived
/// from `scalefac_compress` per ISO/IEC 13818-3 §2.4.3.2.
fn lsf_part2_bits(gc: &GranuleChannel) -> u32 {
    let p = lsf_scale_params(
        gc.scalefac_compress,
        gc.block_type,
        gc.mixed_block_flag,
        false,
    );
    (0..4)
        .map(|i| u32::from(p.slen[i]) * u32::from(p.nr_of_sfb[i]))
        .sum()
}

/// Decode an LSF stream with crate-internal primitives only, returning
/// per-channel reconstructed i16 PCM.
fn decode_lsf_stream(bytes: &[u8]) -> Vec<Vec<i16>> {
    let mut reservoir = Reservoir::new();
    let mut synth: Vec<SynthState> = Vec::new();
    let mut imdct: Vec<ImdctState> = Vec::new();
    let mut out: Vec<Vec<i16>> = Vec::new();

    for frame in FrameWalker::new(bytes) {
        let hdr = parse_header(&frame.data[..4]).expect("header parse");
        assert!(hdr.version.is_lsf(), "expected an LSF frame");
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
            let part2_bits = lsf_part2_bits(gc) as usize;
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

/// Walk an LSF stream's side-info and report `(frames, saw_short)`,
/// asserting the §2.4.2.7 short-block invariants on every emission.
fn walk_block_types(bytes: &[u8]) -> (usize, bool) {
    let mut frames = 0usize;
    let mut saw_short = false;
    for frame in FrameWalker::new(bytes) {
        frames += 1;
        let hdr = parse_header(&frame.data[..4]).expect("header");
        assert!(hdr.version.is_lsf());
        let si = parse_side_info(&hdr, &frame.data[4..]).expect("side_info");
        assert_eq!(si.granule_count, 1, "LSF frame carries one granule");
        for ch in 0..si.channels as usize {
            let gc = &si.granules[0][ch];
            if gc.window_switching_flag && gc.block_type == BlockType::Short {
                assert!(
                    !gc.preflag,
                    "short block emitted with preflag (violates §2.4.2.7)"
                );
                saw_short = true;
            }
            for &sg in &gc.subblock_gain {
                assert!(sg <= 7, "subblock_gain {sg} exceeds 3-bit field");
            }
        }
    }
    (frames, saw_short)
}

#[test]
fn lsf_auto_block_type_on_pure_sine_stays_long() {
    // Steady tone: detector never fires, scheduler emits Long for every
    // single-granule LSF frame.
    const SR: u32 = 22_050;
    let n = SR as usize / 2; // 500 ms
    let pcm = sine_pcm(n, 440.0, SR as f32, 0.5);
    let mut enc = Mp3Encoder::new(64, SR, ChannelMode::SingleChannel).expect("encoder build");
    enc.enable_auto_block_type(DEFAULT_ATTACK_THRESHOLD)
        .expect("enable auto on LSF (r287)");
    enc.push_samples(&pcm).expect("push pcm");
    let mut bytes = Vec::new();
    let _ = enc.finish(&mut bytes).expect("finish");

    for frame in FrameWalker::new(&bytes) {
        let hdr = parse_header(&frame.data[..4]).expect("header");
        let si = parse_side_info(&hdr, &frame.data[4..]).expect("side_info");
        let gc = &si.granules[0][0];
        assert!(
            !gc.window_switching_flag && gc.block_type == BlockType::Long,
            "steady sine produced a window-switched granule"
        );
    }
}

#[test]
fn lsf_auto_block_type_on_click_train_engages_short() {
    // Click train: each burst trips the detector so the single-granule
    // LSF walk fires LONG → START → SHORT → END → LONG. Verify a Short
    // granule appears, the §2.4.2.7 invariants hold, and the stream
    // self-decodes through the crate's own decode primitives.
    const SR: u32 = 22_050;
    let n = SR as usize; // 1 s
    let pcm = click_train_pcm(n, /*click_period=*/ 3300);
    let mut enc = Mp3Encoder::new(64, SR, ChannelMode::SingleChannel).expect("encoder build");
    enc.enable_auto_block_type(DEFAULT_ATTACK_THRESHOLD)
        .expect("enable auto on LSF (r287)");
    enc.push_samples(&pcm).expect("push pcm");
    let mut bytes = Vec::new();
    let _ = enc.finish(&mut bytes).expect("finish");

    let (frames, saw_short) = walk_block_types(&bytes);
    assert!(frames > 0, "no frames emitted");
    assert!(
        saw_short,
        "LSF click-train auto block-type engaged no Short granule"
    );

    // Self-decode must succeed end-to-end.
    let recon = decode_lsf_stream(&bytes);
    assert_eq!(recon.len(), 1);
    assert!(!recon[0].is_empty(), "decode produced no samples");
}

#[test]
fn lsf_auto_block_type_with_mixed_click_train_engages_short() {
    // The mixed-promotion variant must also accept LSF and produce a
    // self-decodable stream that engages the Short geometry.
    const SR: u32 = 24_000;
    let n = SR as usize; // 1 s
    let pcm = click_train_pcm(n, /*click_period=*/ 3300);
    let mut enc = Mp3Encoder::new(64, SR, ChannelMode::SingleChannel).expect("encoder build");
    enc.enable_auto_block_type_with_mixed(DEFAULT_ATTACK_THRESHOLD, 0.5)
        .expect("enable auto+mixed on LSF (r287)");
    assert!(enc.auto_block_type_mixed_enabled());
    enc.push_samples(&pcm).expect("push pcm");
    let mut bytes = Vec::new();
    let _ = enc.finish(&mut bytes).expect("finish");

    let (frames, saw_short) = walk_block_types(&bytes);
    assert!(frames > 0);
    assert!(saw_short, "auto+mixed engaged no Short granule on LSF");
    let recon = decode_lsf_stream(&bytes);
    assert!(!recon[0].is_empty());
}

#[test]
fn lsf_auto_block_type_independent_stereo_roundtrips() {
    // Independent (non-MS) LSF stereo: each channel runs its own
    // detector + scheduler, so the per-channel side-info can carry
    // distinct §C.1.5.2 sequences. Drive a transient on one channel and
    // a steady tone on the other and confirm the stream self-decodes.
    const SR: u32 = 22_050;
    let n = SR as usize; // 1 s
    let left = click_train_pcm(n, 3300);
    let right = sine_pcm(n, 660.0, SR as f32, 0.45);
    let mut interleaved = Vec::with_capacity(n * 2);
    for i in 0..n {
        interleaved.push(left[i]);
        interleaved.push(right[i]);
    }
    let mut enc = Mp3Encoder::new(128, SR, ChannelMode::Stereo).expect("encoder build");
    enc.enable_auto_block_type(DEFAULT_ATTACK_THRESHOLD)
        .expect("enable auto on LSF stereo (r287)");
    enc.push_samples(&interleaved).expect("push pcm");
    let mut bytes = Vec::new();
    let _ = enc.finish(&mut bytes).expect("finish");

    let (frames, saw_short) = walk_block_types(&bytes);
    assert!(frames > 0);
    assert!(
        saw_short,
        "transient channel engaged no Short granule on LSF stereo"
    );
    let recon = decode_lsf_stream(&bytes);
    assert_eq!(recon.len(), 2, "stereo decodes to two channels");
    assert!(!recon[0].is_empty() && !recon[1].is_empty());
}

#[test]
fn lsf_auto_block_type_ms_stereo_agrees_per_granule() {
    // MS-stereo LSF: both channels of a granule must share window
    // geometry (§2.4.3.4.9). The shared scheduler folds per-channel
    // attack flags via OR before stepping, so an attack on either
    // channel switches both. Verify agreement and self-decode.
    const SR: u32 = 22_050;
    let n = SR as usize; // 1 s
    let left = click_train_pcm(n, 3300);
    let right = sine_pcm(n, 550.0, SR as f32, 0.4);
    let mut interleaved = Vec::with_capacity(n * 2);
    for i in 0..n {
        interleaved.push(left[i]);
        interleaved.push(right[i]);
    }
    let mut enc = Mp3Encoder::new_joint_stereo_ms(128, SR).expect("MS encoder build");
    enc.enable_auto_block_type(DEFAULT_ATTACK_THRESHOLD)
        .expect("enable auto on LSF MS (r287)");
    enc.push_samples(&interleaved).expect("push pcm");
    let mut bytes = Vec::new();
    let _ = enc.finish(&mut bytes).expect("finish");

    let mut saw_short = false;
    for frame in FrameWalker::new(&bytes) {
        let hdr = parse_header(&frame.data[..4]).expect("header");
        let si = parse_side_info(&hdr, &frame.data[4..]).expect("side_info");
        let gc0 = &si.granules[0][0];
        let gc1 = &si.granules[0][1];
        assert_eq!(
            gc0.block_type, gc1.block_type,
            "MS-stereo granule channels disagree on block_type"
        );
        assert_eq!(
            gc0.window_switching_flag, gc1.window_switching_flag,
            "MS-stereo granule channels disagree on window_switching_flag"
        );
        if gc0.block_type == BlockType::Short {
            saw_short = true;
        }
    }
    assert!(saw_short, "MS-stereo LSF engaged no Short granule");
    let recon = decode_lsf_stream(&bytes);
    assert_eq!(recon.len(), 2);
    assert!(!recon[0].is_empty() && !recon[1].is_empty());
}
