//! Integration tests for the r194 (Phase 2 step 39) **per-band
//! psychoacoustic threshold** plumbing — [`oxideav_mp3::XminThresholds`]
//! + [`oxideav_mp3::Mp3Encoder::set_per_band_xmin`].
//!
//! Five demonstrations:
//!
//! 1. **Per-band uniform is bit-equivalent to scalar uniform** — the
//!    `*_per_band` outer-loop primitive is a strict generalisation of
//!    the scalar primitive. Installing
//!    `XminThresholds::uniform(thr)` on an encoder built with
//!    `Mp3Encoder::new_with_outer_loop(thr)` must produce a
//!    bit-identical byte stream to the path without
//!    `set_per_band_xmin`.
//!
//! 2. **Threshold-in-quiet long stream round-trips** — installing
//!    `XminThresholds::threshold_in_quiet_long(SR, version, 128, _)`
//!    on a long-block encode produces a parseable Layer III byte
//!    stream that self-decodes through the crate's own decode chain
//!    at finite PSNR > 20 dB.
//!
//! 3. **Per-band threshold changes amplification pattern vs uniform**
//!    — at the same scalar threshold + same fixture, installing the
//!    LTq vector produces a different byte stream than the uniform
//!    path. Witnesses that the per-band threshold actually propagates
//!    into the encoder's §C.1.5.4.3 distortion-control loop.
//!
//! 4. **Per-band path rejects misuse** — calling `set_per_band_xmin`
//!    on an encoder constructed via `Mp3Encoder::new` (no outer loop)
//!    returns `StreamEncodeError::PerBandXminWithoutOuterLoop`.
//!
//! 5. **Silence + single-tone round-trips** — basic sanity that the
//!    per-band path produces a finite, decodable stream on the trivial
//!    inputs every encoder must handle.
//!
//! No external library is invoked — the decode harness reuses the
//! crate's own `FrameWalker` + `decode_huffman` + `requantize` +
//! `imdct_granule` + `synth_granule` chain. The harness is a near
//! copy of `outer_loop_roundtrip.rs`'s `decode_mp3_mono` — kept inline
//! rather than re-exported because it is test-only scaffolding.

use std::f32::consts::PI;

use oxideav_mp3::{
    alias_reduce, decode_huffman, imdct_granule, parse_header, parse_side_info, requantize,
    synth_granule, BlockType, ChannelMode, FrameWalker, ImdctState, MainDataReader, Mp3Encoder,
    MpegVersion, Reservoir, ScaleFactors, StreamEncodeError, SynthState, XminThresholds,
    MPEG1_SLEN, PCM_PER_GRANULE,
};

const SR: u32 = 44_100;
const BR: u32 = 128;
const PCM_LEN_SAMPLES: usize = SR as usize; // 1 second

fn multi_tone_pcm(n: usize, freqs_hz: &[f32], per_tone_amp: f32) -> Vec<i16> {
    let two_pi = 2.0 * PI;
    let mut out = Vec::with_capacity(n);
    let scale = i16::MAX as f32;
    for i in 0..n {
        let t = i as f32 / (SR as f32);
        let mut s = 0.0f32;
        for &f in freqs_hz {
            s += (two_pi * f * t).sin();
        }
        let v = (s * per_tone_amp * scale)
            .round()
            .clamp(i16::MIN as f32, i16::MAX as f32);
        out.push(v as i16);
    }
    out
}

fn sine_pcm(n: usize, freq_hz: f32, amp: f32) -> Vec<i16> {
    let two_pi = 2.0 * PI;
    let mut out = Vec::with_capacity(n);
    let scale = (i16::MAX as f32) * amp;
    for i in 0..n {
        let t = i as f32 / (SR as f32);
        let v = ((two_pi * freq_hz * t).sin() * scale)
            .round()
            .clamp(i16::MIN as f32, i16::MAX as f32);
        out.push(v as i16);
    }
    out
}

fn psnr(original: &[i16], recon: &[i16]) -> f64 {
    let n = original.len().min(recon.len());
    if n == 0 {
        return f64::NEG_INFINITY;
    }
    let mut sse = 0.0_f64;
    for i in 0..n {
        let d = f64::from(original[i]) - f64::from(recon[i]);
        sse += d * d;
    }
    let mse = sse / n as f64;
    if mse == 0.0 {
        return f64::INFINITY;
    }
    let peak = f64::from(i16::MAX);
    10.0 * (peak * peak / mse).log10()
}

/// Single-pass walk of the spec layout in `outer_loop_roundtrip.rs`,
/// inlined here because it is test-only scaffolding for the decode
/// harness below.
fn read_mpeg1_part2(
    r: &mut MainDataReader,
    gc: &oxideav_mp3::GranuleChannel,
    scfsi: &[bool; 4],
    gr: usize,
    prev: Option<&ScaleFactors>,
) -> ScaleFactors {
    let (slen1, slen2) = MPEG1_SLEN[(gc.scalefac_compress & 0xF) as usize];
    let mut sf = ScaleFactors {
        preflag: gc.preflag,
        ..ScaleFactors::default()
    };
    let short = gc.window_switching_flag && gc.block_type == BlockType::Short;
    if short {
        if gc.mixed_block_flag {
            for band in sf.long.iter_mut().take(8) {
                *band = r.read(u32::from(slen1)) as u8;
            }
            for sfb in 3..12 {
                let slen = if sfb < 6 { slen1 } else { slen2 };
                for w in 0..3 {
                    sf.short[sfb][w] = r.read(u32::from(slen)) as u8;
                }
            }
        } else {
            for sfb in 0..12 {
                let slen = if sfb < 6 { slen1 } else { slen2 };
                for w in 0..3 {
                    sf.short[sfb][w] = r.read(u32::from(slen)) as u8;
                }
            }
        }
        sf.preflag = false;
    } else {
        const GROUPS: [(usize, usize); 4] = [(0, 6), (6, 11), (11, 16), (16, 21)];
        for (g, &(lo, hi)) in GROUPS.iter().enumerate() {
            let reuse = gr == 1 && scfsi[g];
            for sfb in lo..hi {
                if reuse {
                    sf.long[sfb] = prev.map_or(0, |p| p.long[sfb]);
                } else {
                    let slen = if sfb <= 10 { slen1 } else { slen2 };
                    sf.long[sfb] = r.read(u32::from(slen)) as u8;
                }
            }
        }
    }
    sf
}

fn decode_mp3_mono(bytes: &[u8]) -> Vec<i16> {
    let mut reservoir = Reservoir::new();
    let mut synth_state = SynthState::new();
    let mut imdct_state = ImdctState::new();
    let mut out_pcm: Vec<i16> = Vec::new();

    for frame in FrameWalker::new(bytes) {
        let hdr = parse_header(&frame.data[..4]).expect("header parse");
        let si = parse_side_info(&hdr, &frame.data[4..]).expect("side_info parse");
        let si_len = si.byte_len();
        let main_data_slot = &frame.data[4 + si_len..];
        let run = reservoir
            .assemble(usize::from(si.main_data_begin), main_data_slot)
            .expect("reservoir assemble");

        let mut r = MainDataReader::new(&run);
        let mut prev_g0_sf: [ScaleFactors; 2] = [ScaleFactors::default(); 2];

        #[allow(clippy::needless_range_loop)]
        for gr in 0..si.granule_count as usize {
            #[allow(clippy::needless_range_loop)]
            for ch in 0..si.channels as usize {
                let gc = &si.granules[gr][ch];
                let gc_start = r.bit_pos();
                let sf = match hdr.version {
                    MpegVersion::Mpeg1 => {
                        let prev = if gr == 1 { Some(&prev_g0_sf[ch]) } else { None };
                        let s = read_mpeg1_part2(&mut r, gc, &si.scfsi[ch], gr, prev);
                        if gr == 0 {
                            prev_g0_sf[ch] = s;
                        }
                        s
                    }
                    MpegVersion::Mpeg2 | MpegVersion::Mpeg25 => {
                        unreachable!("LSF / MPEG-2.5 outer-loop encode not in scope this round")
                    }
                };
                let part2_bits = r.bit_pos() - gc_start;
                let part3_bits = u32::from(gc.part2_3_length).saturating_sub(part2_bits as u32);
                let is = decode_huffman(&mut r, gc, part3_bits, hdr.sample_rate_hz, hdr.version)
                    .expect("huffman");
                let want = gc_start + gc.part2_3_length as usize;
                while r.bit_pos() < want {
                    let n = (want - r.bit_pos()).min(32);
                    let _ = r.read(n as u32);
                }

                let xr = requantize(&is, gc, &sf, hdr.sample_rate_hz, hdr.version);
                let xar = alias_reduce(&xr, gc);
                let subband_time = imdct_granule(&xar, gc, &mut imdct_state);
                let pcm_f32 = synth_granule(&subband_time, &mut synth_state);
                for &p in pcm_f32.iter().take(PCM_PER_GRANULE) {
                    let v = p * f32::from(i16::MAX);
                    out_pcm.push(v.clamp(i16::MIN as f32, i16::MAX as f32) as i16);
                }
            }
        }
    }
    out_pcm
}

fn encode<F>(make_encoder: F, pcm: &[i16]) -> Vec<u8>
where
    F: FnOnce() -> Mp3Encoder,
{
    let mut enc = make_encoder();
    enc.push_samples(pcm).expect("push pcm");
    let mut out: Vec<u8> = Vec::new();
    let n = enc.finish(&mut out).expect("encoder finish");
    assert_eq!(out.len(), n);
    out
}

fn aligned_psnr(pcm_in: &[i16], pcm_out: &[i16]) -> f64 {
    let warmup = 4 * 1152;
    let total_delay = 1057usize;
    if pcm_out.len() <= warmup + total_delay {
        panic!("recon too short: {}", pcm_out.len());
    }
    let head_recon = warmup + total_delay;
    let cmp_len = pcm_out
        .len()
        .saturating_sub(head_recon)
        .min(pcm_in.len() - warmup);
    let recon_cmp = &pcm_out[head_recon..head_recon + cmp_len];
    let pcm_cmp = &pcm_in[warmup..warmup + cmp_len];
    psnr(pcm_cmp, recon_cmp)
}

/// **Backward-compat anchor** — installing `XminThresholds::uniform(thr)`
/// on an encoder built with `Mp3Encoder::new_with_outer_loop(thr)` must
/// produce a bit-identical byte stream to the path without
/// `set_per_band_xmin`.
#[test]
fn per_band_uniform_is_bit_identical_to_scalar_uniform() {
    let pcm = multi_tone_pcm(
        PCM_LEN_SAMPLES,
        &[110.0, 440.0, 880.0, 1760.0, 3520.0, 7040.0],
        0.12,
    );
    let thr = 5.0e-8;

    let bytes_scalar = encode(
        || {
            Mp3Encoder::new_with_outer_loop(BR, SR, ChannelMode::SingleChannel, thr)
                .expect("encoder build")
        },
        &pcm,
    );

    let bytes_per_band = encode(
        || {
            let mut enc = Mp3Encoder::new_with_outer_loop(BR, SR, ChannelMode::SingleChannel, thr)
                .expect("encoder build");
            enc.set_per_band_xmin(XminThresholds::uniform(thr))
                .expect("install uniform per-band");
            assert!(enc.per_band_xmin_enabled());
            enc
        },
        &pcm,
    );

    assert_eq!(
        bytes_scalar.len(),
        bytes_per_band.len(),
        "per-band uniform produced different stream length than scalar uniform",
    );
    assert_eq!(
        bytes_scalar, bytes_per_band,
        "per-band uniform fill must be bit-identical to scalar uniform threshold path",
    );
}

/// **Threshold-in-quiet long stream round-trips** through the crate's
/// own decode chain at finite PSNR > 20 dB.
#[test]
fn per_band_threshold_in_quiet_long_self_decode_psnr_above_floor() {
    let pcm = multi_tone_pcm(
        PCM_LEN_SAMPLES,
        &[110.0, 440.0, 880.0, 1760.0, 3520.0, 7040.0],
        0.12,
    );

    let bytes = encode(
        || {
            let mut enc = Mp3Encoder::new_with_outer_loop(
                BR,
                SR,
                ChannelMode::SingleChannel,
                oxideav_mp3::stream_encoder::DEFAULT_OUTER_LOOP_THRESHOLD,
            )
            .expect("encoder build");
            enc.set_per_band_xmin(XminThresholds::threshold_in_quiet_long(
                SR,
                MpegVersion::Mpeg1,
                BR, // per-channel bit rate; mono so equals stream bit rate
                oxideav_mp3::stream_encoder::DEFAULT_OUTER_LOOP_THRESHOLD,
            ))
            .expect("install ltq per-band");
            enc
        },
        &pcm,
    );
    assert!(
        bytes.len() > 1000,
        "encoded stream too small: {}",
        bytes.len()
    );
    let recon = decode_mp3_mono(&bytes);
    let p = aligned_psnr(&pcm, &recon);
    eprintln!("per-band LTq self-decode PSNR = {p:.2} dB");
    assert!(
        p.is_finite() && p > 20.0,
        "per-band LTq PSNR too low: {p} dB",
    );
}

/// **Spectral shape is observable** — at the same scalar threshold +
/// same fixture, installing the per-band LTq vector produces a
/// different byte stream than the uniform path. Witnesses that the
/// per-band threshold propagates into the encoder's outer-loop
/// decision.
#[test]
fn per_band_threshold_changes_byte_stream_vs_uniform() {
    let pcm = multi_tone_pcm(
        PCM_LEN_SAMPLES,
        &[110.0, 440.0, 880.0, 1760.0, 3520.0, 7040.0],
        0.12,
    );

    // Use a TIGHT scalar threshold so the uniform-path outer loop
    // actually fires the amplification step (otherwise both paths
    // short-circuit at iteration 1 with no amplification and produce
    // identical output). 5e-8 sits between the loudest bands'
    // baseline distortion and the median for this fixture — same
    // value `outer_loop_strictly_higher_psnr_than_fixed_gain_on_multi_tone`
    // uses in `outer_loop_roundtrip.rs`.
    let scalar_thr = 5.0e-8;

    let bytes_uniform = encode(
        || {
            Mp3Encoder::new_with_outer_loop(BR, SR, ChannelMode::SingleChannel, scalar_thr)
                .expect("encoder build")
        },
        &pcm,
    );

    // Per-band path: construct with a tight scalar fallback (so the
    // dispatch into the outer-loop branch fires) and override the
    // long-band vector with the LTq curve. The LTq long bands carry
    // *spectrally-shaped* thresholds — some bands much tighter than
    // the scalar `5e-8`, others much looser — so the amplification
    // pattern per granule diverges from the uniform path.
    let bytes_per_band = encode(
        || {
            let mut enc =
                Mp3Encoder::new_with_outer_loop(BR, SR, ChannelMode::SingleChannel, scalar_thr)
                    .expect("encoder build");
            // LTq at 128 kbit/s (≥ 96 kbit/s ⇒ −12 dB offset). Use
            // the scalar fallback for short/mixed since those branches
            // still consume the scalar threshold this round.
            enc.set_per_band_xmin(XminThresholds::threshold_in_quiet_long(
                SR,
                MpegVersion::Mpeg1,
                BR,
                scalar_thr,
            ))
            .expect("install ltq per-band");
            enc
        },
        &pcm,
    );

    assert_ne!(
        bytes_uniform, bytes_per_band,
        "per-band LTq path produced the SAME bytes as uniform path on a spectrally-rich \
         fixture under a tight scalar threshold — the per-band threshold isn't propagating \
         into the outer-loop decision",
    );
}

/// **API rejection** — installing the per-band vector without first
/// enabling the outer loop is a user error and must be rejected at API
/// time.
#[test]
fn set_per_band_xmin_without_outer_loop_is_rejected() {
    let mut enc = Mp3Encoder::new(BR, SR, ChannelMode::SingleChannel).expect("encoder build");
    let err = enc
        .set_per_band_xmin(XminThresholds::uniform(1.0e6))
        .expect_err("install without outer loop must fail");
    assert!(matches!(
        err,
        StreamEncodeError::PerBandXminWithoutOuterLoop
    ));
    assert!(!enc.per_band_xmin_enabled());
}

/// **Silence round-trips** — all-zero PCM under the per-band LTq
/// vector still decodes to near-silence.
#[test]
fn per_band_silence_decodes_to_near_zero() {
    let pcm = vec![0i16; 3 * 1152];
    let bytes = encode(
        || {
            let mut enc = Mp3Encoder::new_with_outer_loop(
                BR,
                SR,
                ChannelMode::SingleChannel,
                oxideav_mp3::stream_encoder::DEFAULT_OUTER_LOOP_THRESHOLD,
            )
            .expect("encoder build");
            enc.set_per_band_xmin(XminThresholds::threshold_in_quiet_long(
                SR,
                MpegVersion::Mpeg1,
                BR,
                oxideav_mp3::stream_encoder::DEFAULT_OUTER_LOOP_THRESHOLD,
            ))
            .expect("install ltq per-band");
            enc
        },
        &pcm,
    );
    let recon = decode_mp3_mono(&bytes);
    let warmup = 2 * 1152;
    if recon.len() > warmup {
        let peak = recon[warmup..]
            .iter()
            .map(|s| s.unsigned_abs())
            .max()
            .unwrap_or(0);
        assert!(peak <= 16, "per-band LTq silence peak too high: {peak}");
    }
}

/// **Sine self-decode** — a single 440 Hz tone under the per-band LTq
/// threshold round-trips at finite PSNR > 20 dB.
#[test]
fn per_band_sine_self_decode_psnr_above_floor() {
    let pcm = sine_pcm(PCM_LEN_SAMPLES, 440.0, 0.5);
    let bytes = encode(
        || {
            let mut enc = Mp3Encoder::new_with_outer_loop(
                BR,
                SR,
                ChannelMode::SingleChannel,
                oxideav_mp3::stream_encoder::DEFAULT_OUTER_LOOP_THRESHOLD,
            )
            .expect("encoder build");
            enc.set_per_band_xmin(XminThresholds::threshold_in_quiet_long(
                SR,
                MpegVersion::Mpeg1,
                BR,
                oxideav_mp3::stream_encoder::DEFAULT_OUTER_LOOP_THRESHOLD,
            ))
            .expect("install ltq per-band");
            enc
        },
        &pcm,
    );
    assert!(
        bytes.len() > 1000,
        "encoded stream too small: {}",
        bytes.len()
    );
    let recon = decode_mp3_mono(&bytes);
    let p = aligned_psnr(&pcm, &recon);
    eprintln!("per-band LTq sine self-decode PSNR = {p:.2} dB");
    assert!(p > 20.0, "per-band LTq sine PSNR too low: {p} dB");
}

// =========================================================================
// r197 step 40 — per-band short-block tests
// =========================================================================

/// **Backward-compat anchor (short-block)** — installing
/// `XminThresholds::uniform(thr)` while every granule is force-shorted
/// must produce a bit-identical byte stream to the scalar uniform path.
/// Witnesses the r197 `outer_loop_search_short` scalar shim is
/// bit-for-bit equivalent to the per-band primitive under a uniform
/// matrix.
#[test]
fn per_band_short_uniform_is_bit_identical_to_scalar_uniform() {
    let pcm = multi_tone_pcm(
        PCM_LEN_SAMPLES,
        &[110.0, 440.0, 880.0, 1760.0, 3520.0, 7040.0],
        0.12,
    );
    let thr = 5.0e-8;

    let bytes_scalar = encode(
        || {
            let mut enc = Mp3Encoder::new_with_outer_loop(BR, SR, ChannelMode::SingleChannel, thr)
                .expect("encoder build");
            enc.force_short_blocks_for_testing(true)
                .expect("enable force-short");
            enc
        },
        &pcm,
    );

    let bytes_per_band = encode(
        || {
            let mut enc = Mp3Encoder::new_with_outer_loop(BR, SR, ChannelMode::SingleChannel, thr)
                .expect("encoder build");
            enc.force_short_blocks_for_testing(true)
                .expect("enable force-short");
            enc.set_per_band_xmin(XminThresholds::uniform(thr))
                .expect("install uniform per-band");
            assert!(enc.per_band_xmin_enabled());
            enc
        },
        &pcm,
    );

    assert_eq!(
        bytes_scalar.len(),
        bytes_per_band.len(),
        "short-block per-band uniform produced different stream length than scalar uniform",
    );
    assert_eq!(
        bytes_scalar, bytes_per_band,
        "short-block per-band uniform fill must be bit-identical to scalar uniform threshold path",
    );
}

/// **Threshold-in-quiet short-block round-trips to non-silent finite PCM**
/// — a force-shorted granule under the per-band LTq matrix decodes
/// through the crate's own decode chain to finite, non-silent PCM.
/// Witnesses the new `XminThresholds::threshold_in_quiet` constructor's
/// short cells propagate into the encoder's pure-short outer loop and
/// produce a parseable byte stream. (Force-short on a steady multi-tone
/// fixture inherently has lower PSNR than the long-block path —
/// short-block windows are designed for transients, not sustained
/// tones; mirrors `force_short_stream_decodes_to_finite_non_silent_pcm`
/// in `short_block_encoder_roundtrip.rs`.)
#[test]
fn per_band_threshold_in_quiet_short_self_decode_is_finite_non_silent() {
    let pcm = multi_tone_pcm(
        PCM_LEN_SAMPLES,
        &[110.0, 440.0, 880.0, 1760.0, 3520.0, 7040.0],
        0.12,
    );

    let bytes = encode(
        || {
            let mut enc = Mp3Encoder::new_with_outer_loop(
                BR,
                SR,
                ChannelMode::SingleChannel,
                oxideav_mp3::stream_encoder::DEFAULT_OUTER_LOOP_THRESHOLD,
            )
            .expect("encoder build");
            enc.force_short_blocks_for_testing(true)
                .expect("enable force-short");
            enc.set_per_band_xmin(XminThresholds::threshold_in_quiet(
                SR,
                MpegVersion::Mpeg1,
                BR,
            ))
            .expect("install ltq per-band");
            enc
        },
        &pcm,
    );
    assert!(
        bytes.len() > 1000,
        "encoded stream too small: {}",
        bytes.len()
    );
    let recon = decode_mp3_mono(&bytes);
    assert!(!recon.is_empty(), "decoded PCM was empty");
    let energy: f64 = recon
        .iter()
        .map(|&v| f64::from(v) * f64::from(v))
        .sum::<f64>()
        / recon.len() as f64;
    assert!(
        energy.is_finite() && energy > 0.0,
        "decoded PCM had zero or non-finite energy ({energy})",
    );
    let p = aligned_psnr(&pcm, &recon);
    eprintln!(
        "per-band short-block LTq self-decode PSNR = {p:.2} dB (finite, non-silent — \
         short-block paths inherently lose more on sustained tones than long blocks)",
    );
    assert!(p.is_finite() && p > 0.0, "PSNR was non-finite or negative");
}

/// **Per-band short threshold changes byte stream vs uniform** — at the
/// same scalar threshold + same fixture under force-short, installing
/// the LTq matrix produces a different byte stream than the uniform
/// path. Witnesses that the per-cell threshold propagates into the
/// pure-short outer-loop decision.
#[test]
fn per_band_short_threshold_changes_byte_stream_vs_uniform() {
    let pcm = multi_tone_pcm(
        PCM_LEN_SAMPLES,
        &[110.0, 440.0, 880.0, 1760.0, 3520.0, 7040.0],
        0.12,
    );
    // Tight scalar threshold — same value as the long-block divergence
    // test; ensures the uniform path actually fires the amp step.
    let scalar_thr = 5.0e-8;

    let bytes_uniform = encode(
        || {
            let mut enc =
                Mp3Encoder::new_with_outer_loop(BR, SR, ChannelMode::SingleChannel, scalar_thr)
                    .expect("encoder build");
            enc.force_short_blocks_for_testing(true)
                .expect("enable force-short");
            enc
        },
        &pcm,
    );

    let bytes_per_band = encode(
        || {
            let mut enc =
                Mp3Encoder::new_with_outer_loop(BR, SR, ChannelMode::SingleChannel, scalar_thr)
                    .expect("encoder build");
            enc.force_short_blocks_for_testing(true)
                .expect("enable force-short");
            enc.set_per_band_xmin(XminThresholds::threshold_in_quiet(
                SR,
                MpegVersion::Mpeg1,
                BR,
            ))
            .expect("install ltq per-band");
            enc
        },
        &pcm,
    );

    assert_ne!(
        bytes_uniform, bytes_per_band,
        "short-block per-band LTq path produced the SAME bytes as uniform path on a \
         spectrally-rich fixture under a tight scalar threshold — the per-cell threshold \
         isn't propagating into the pure-short outer-loop decision",
    );
}
