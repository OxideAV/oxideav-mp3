//! Integration tests for the r355 **named psychoacoustic quality
//! preset** — [`oxideav_mp3::QualityPreset`] +
//! [`oxideav_mp3::Mp3Encoder::with_quality_preset`] and the
//! registry-path [`oxideav_mp3::make_encoder_quality_preset`] factory.
//!
//! Demonstrations:
//!
//! 1. **Every preset round-trips** — each of the four presets, applied
//!    to a 44.1 kHz (staged Annex D rate) long-block encode, produces a
//!    parseable Layer III byte stream that self-decodes through the
//!    crate's own decode chain at a finite PSNR above a floor.
//! 2. **Annex D rates run the signal-dependent path** — at 44.1 kHz the
//!    Model-2-bearing presets report
//!    `quality_preset_is_signal_dependent() == true`; `Fast` (no Model 2)
//!    reports `false`.
//! 3. **Non-Annex-D rates fall back gracefully** — at a MPEG-2 LSF rate
//!    (24 kHz) a preset still applies (no error) but reports
//!    `is_signal_dependent() == false`, having installed the signal-
//!    independent threshold-in-quiet vector translated by the preset
//!    offset.
//! 4. **Preset offset orders the installed threshold** — on the signal-
//!    independent fallback path the tightest preset (`Transparent`,
//!    -24 dB) installs a strictly smaller per-band `xmin` than `Standard`
//!    (0 dB), which is in turn strictly smaller than `Fast` (+6 dB),
//!    witnessing that the §D.1 Step 3 offset reaches the installed
//!    threshold. (Asserted on the vector, not the bytes: at the
//!    calibrated default threshold scale the §C.1.5.4.3 outer loop is
//!    rate-bound for typical CBR content, so a uniform dB translation of
//!    the whole bowl — preserving every per-band ratio — leaves the bit
//!    allocation, and thus the bytes, unchanged.)
//! 5. **Misuse is rejected** — applying a preset to an encoder built via
//!    `Mp3Encoder::new` (no outer loop) returns
//!    `StreamEncodeError::PerBandXminWithoutOuterLoop`.
//! 6. **Registry path works** — `make_encoder_quality_preset` builds a
//!    working boxed `oxideav_core::Encoder` whose flushed stream
//!    self-decodes.
//!
//! The decode harness reuses the crate's own `FrameWalker` +
//! `decode_huffman` + `requantize` + `imdct_granule` + `synth_granule`
//! chain (a near copy of `per_band_xmin_roundtrip.rs`'s `decode_mp3_mono`,
//! kept inline because it is test-only scaffolding). No external library
//! is invoked.

use std::f32::consts::PI;

use oxideav_core::{AudioFrame, CodecId, CodecParameters, Error, Frame, SampleFormat};
use oxideav_mp3::{
    alias_reduce, decode_huffman, imdct_granule, make_encoder_quality_preset, parse_header,
    parse_side_info, pcm_f32_to_i16, requantize, synth_granule, BlockType, ChannelMode,
    FrameWalker, ImdctState, MainDataReader, Mp3Encoder, MpegVersion, QualityPreset, Reservoir,
    ScaleFactors, StreamEncodeError, SynthState, MPEG1_SLEN, PCM_PER_GRANULE,
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

/// Single-pass MPEG-1 part-2 scalefactor walk (inlined test scaffolding).
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
    } else {
        // Long block: four scfsi_band groups {0..5},{6..10},{11..15},{16..20}.
        let groups: [(usize, usize); 4] = [(0, 6), (6, 11), (11, 16), (16, 21)];
        for (g, &(lo, hi)) in groups.iter().enumerate() {
            let reuse = gr == 1 && scfsi[g];
            for sfb in lo..hi {
                if reuse {
                    sf.long[sfb] = prev.map(|p| p.long[sfb]).unwrap_or(0);
                } else {
                    let slen = if sfb < 11 { slen1 } else { slen2 };
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
                        unreachable!("LSF / MPEG-2.5 decode not exercised by this harness")
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
                    out_pcm.push(pcm_f32_to_i16(p));
                }
            }
        }
    }
    out_pcm
}

fn encode_with_preset(preset: QualityPreset, pcm: &[i16]) -> (Vec<u8>, bool) {
    let mut enc = Mp3Encoder::new_with_outer_loop(
        BR,
        SR,
        ChannelMode::SingleChannel,
        oxideav_mp3::stream_encoder::DEFAULT_OUTER_LOOP_THRESHOLD,
    )
    .expect("outer-loop encoder");
    enc.with_quality_preset(preset).expect("apply preset");
    assert_eq!(enc.quality_preset(), Some(preset));
    let signal_dependent = enc.quality_preset_is_signal_dependent();
    enc.push_samples(pcm).expect("push pcm");
    let mut out: Vec<u8> = Vec::new();
    let n = enc.finish(&mut out).expect("encoder finish");
    assert_eq!(out.len(), n);
    (out, signal_dependent)
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

#[test]
fn every_preset_round_trips_at_annex_d_rate() {
    let pcm = multi_tone_pcm(
        PCM_LEN_SAMPLES,
        &[110.0, 440.0, 880.0, 1760.0, 3520.0, 7040.0],
        0.10,
    );
    for preset in [
        QualityPreset::Transparent,
        QualityPreset::High,
        QualityPreset::Standard,
        QualityPreset::Fast,
    ] {
        let (bytes, _) = encode_with_preset(preset, &pcm);
        assert!(!bytes.is_empty(), "{preset:?} produced no bytes");
        let recon = decode_mp3_mono(&bytes);
        let p = aligned_psnr(&pcm, &recon);
        assert!(
            p.is_finite() && p > 15.0,
            "{preset:?} self-decode PSNR {p:.2} dB below floor"
        );
    }
}

#[test]
fn annex_d_rate_arms_model2_for_model2_presets() {
    let pcm = multi_tone_pcm(PCM_LEN_SAMPLES, &[440.0, 3520.0], 0.15);
    // Model-2-bearing presets → signal-dependent path at 44.1 kHz.
    for preset in [
        QualityPreset::Transparent,
        QualityPreset::High,
        QualityPreset::Standard,
    ] {
        let (_, signal_dependent) = encode_with_preset(preset, &pcm);
        assert!(
            signal_dependent,
            "{preset:?} should run the Model 2 path at a staged Annex D rate"
        );
    }
    // Fast skips Model 2 entirely.
    let (_, fast_sd) = encode_with_preset(QualityPreset::Fast, &pcm);
    assert!(!fast_sd, "Fast must not arm Model 2");
}

#[test]
fn non_annex_d_rate_falls_back_to_signal_independent() {
    // 24 kHz is a MPEG-2 LSF rate — no staged Annex D calculation-
    // partition table, so a Model-2-bearing preset must fall back to the
    // signal-independent per-band threshold-in-quiet vector rather than
    // erroring.
    let lsf_rate = 24_000u32;
    let mut enc = Mp3Encoder::new_with_outer_loop(
        BR,
        lsf_rate,
        ChannelMode::SingleChannel,
        oxideav_mp3::stream_encoder::DEFAULT_OUTER_LOOP_THRESHOLD,
    )
    .expect("LSF outer-loop encoder");
    enc.with_quality_preset(QualityPreset::High)
        .expect("preset applies at LSF rate");
    assert_eq!(enc.quality_preset(), Some(QualityPreset::High));
    assert!(
        !enc.quality_preset_is_signal_dependent(),
        "Model 2 has no staged tables at 24 kHz; preset must fall back"
    );
    // The fallback installed a per-band vector but did not arm Model 2.
    assert!(!enc.model2_psychoacoustics_enabled());

    // The encoder still produces a stream.
    let pcm = multi_tone_pcm(8 * 1152, &[440.0, 1760.0], 0.12);
    enc.push_samples(&pcm).expect("push pcm");
    let mut out: Vec<u8> = Vec::new();
    let n = enc.finish(&mut out).expect("LSF finish");
    assert_eq!(out.len(), n);
    assert!(!out.is_empty(), "LSF preset stream empty");
}

/// The preset's §D.1 Step 3 offset must reach the **installed per-band
/// threshold** on the signal-independent fallback path: a tighter preset
/// installs a strictly smaller `xmin` (asks for higher SNR ⇒ more bits)
/// at every long band than a looser preset.
///
/// This is asserted on the threshold *vector* (the contract the preset
/// directly controls), not on the encoded bytes: at the calibrated
/// default threshold scale the §C.1.5.4.3 outer loop is rate-bound for
/// typical CBR content, so a uniform dB translation of the whole bowl —
/// which preserves every per-band ratio — leaves the bit allocation (and
/// thus the bytes) unchanged. The vector ordering is the scale-
/// independent witness that the offset propagated. A non-Annex-D rate is
/// used so every preset takes the static fallback (no per-granule Model 2
/// overwrite), leaving a stable vector to inspect.
#[test]
fn preset_offset_orders_installed_threshold_on_fallback() {
    let lsf_rate = 24_000u32;
    let build = |preset: QualityPreset| {
        let mut enc = Mp3Encoder::new_with_outer_loop(
            BR,
            lsf_rate,
            ChannelMode::SingleChannel,
            oxideav_mp3::stream_encoder::DEFAULT_OUTER_LOOP_THRESHOLD,
        )
        .expect("LSF outer-loop encoder");
        enc.with_quality_preset(preset).expect("apply preset");
        enc
    };
    let transparent = build(QualityPreset::Transparent); // -24 dB
    let standard = build(QualityPreset::Standard); // 0 dB
    let fast = build(QualityPreset::Fast); // +6 dB

    let xt = transparent
        .installed_per_band_xmin()
        .expect("Transparent installs a static vector at the LSF fallback");
    let xs = standard
        .installed_per_band_xmin()
        .expect("Standard installs a static vector");
    let xf = fast
        .installed_per_band_xmin()
        .expect("Fast installs a static vector");

    for sfb in 0..xt.long.len() {
        assert!(
            xt.long[sfb] < xs.long[sfb],
            "long sfb {sfb}: Transparent (-24 dB) {} must be tighter than Standard (0 dB) {}",
            xt.long[sfb],
            xs.long[sfb],
        );
        assert!(
            xs.long[sfb] < xf.long[sfb],
            "long sfb {sfb}: Standard (0 dB) {} must be tighter than Fast (+6 dB) {}",
            xs.long[sfb],
            xf.long[sfb],
        );
    }
}

/// The one-call `new_with_quality_preset` constructor produces an
/// encoder equivalent to `new_with_outer_loop` + `with_quality_preset`:
/// the preset is recorded, the signal-dependent path is armed at the
/// Annex D rate, and the stream self-decodes.
#[test]
fn one_call_constructor_round_trips() {
    let pcm = multi_tone_pcm(PCM_LEN_SAMPLES, &[440.0, 1760.0, 5280.0], 0.10);
    let mut enc = Mp3Encoder::new_with_quality_preset(
        BR,
        SR,
        ChannelMode::SingleChannel,
        QualityPreset::High,
    )
    .expect("one-call build");
    assert_eq!(enc.quality_preset(), Some(QualityPreset::High));
    assert!(
        enc.quality_preset_is_signal_dependent(),
        "High at 44.1 kHz should arm Model 2"
    );
    enc.push_samples(&pcm).expect("push pcm");
    let mut out: Vec<u8> = Vec::new();
    let n = enc.finish(&mut out).expect("finish");
    assert_eq!(out.len(), n);
    let recon = decode_mp3_mono(&out);
    let p = aligned_psnr(&pcm, &recon);
    assert!(
        p.is_finite() && p > 15.0,
        "one-call constructor PSNR {p:.2} dB below floor"
    );
}

#[test]
fn preset_without_outer_loop_is_rejected() {
    let mut enc = Mp3Encoder::new(BR, SR, ChannelMode::SingleChannel).expect("plain encoder");
    let err = enc.with_quality_preset(QualityPreset::High).unwrap_err();
    assert!(matches!(
        err,
        StreamEncodeError::PerBandXminWithoutOuterLoop
    ));
    assert_eq!(enc.quality_preset(), None);
}

#[test]
fn registry_factory_round_trips() {
    let pcm = multi_tone_pcm(PCM_LEN_SAMPLES, &[440.0, 1760.0, 7040.0], 0.12);
    let mut params = CodecParameters::audio(CodecId::new("mp3"));
    params.sample_rate = Some(SR);
    params.channels = Some(1);
    params.bit_rate = Some(u64::from(BR) * 1000);
    params.sample_format = Some(SampleFormat::S16);

    let mut enc =
        make_encoder_quality_preset(&params, QualityPreset::High).expect("build via factory");

    // Drive through the Encoder trait: one frame in, flush, drain packets.
    let mut bytes = Vec::with_capacity(pcm.len() * 2);
    for s in &pcm {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    let frame = AudioFrame {
        samples: pcm.len() as u32,
        pts: None,
        data: vec![bytes],
    };
    enc.send_frame(&Frame::Audio(frame)).expect("send_frame");
    enc.flush().expect("flush");
    let mut stream: Vec<u8> = Vec::new();
    loop {
        match enc.receive_packet() {
            Ok(pkt) => stream.extend_from_slice(&pkt.data),
            Err(Error::Eof) => break,
            Err(Error::NeedMore) => break,
            Err(e) => panic!("unexpected drain error: {e}"),
        }
    }
    assert!(!stream.is_empty(), "factory produced no bytes");
    let recon = decode_mp3_mono(&stream);
    let p = aligned_psnr(&pcm, &recon);
    assert!(
        p.is_finite() && p > 15.0,
        "factory self-decode PSNR {p:.2} dB below floor"
    );
}
