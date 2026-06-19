//! Integration tests for the r285 MPEG-2 LSF / MPEG-2.5 widening of
//! the PCM → MP3 stream encoder ([`oxideav_mp3::Mp3Encoder`]).
//!
//! Per ISO/IEC 13818-3 §2.4.3.2 an LSF Layer III frame carries **one**
//! 576-sample granule (`slots_per_frame` constant 72 instead of 144),
//! the §2.4.1.7 LSF side-info layout (8-bit `main_data_begin`, no
//! `scfsi`, 9-bit `scalefac_compress`, no transmitted `preflag`), the
//! LSF §2.4.2.3 bitrate ladder, and the Table B.2 scalefactor bands.
//! The MPEG-2.5 rates (8 / 11.025 / 12 kHz) inherit the same framing
//! (`docs/audio/mp3/MPEG-2.5-GAP.md`).
//!
//! Every test encodes synthetic PCM and decodes the result with the
//! crate's **own** decode primitives only (header / side-info parse,
//! scalefactor + Huffman decode, requantize, alias, IMDCT, synthesis)
//! — no external library is invoked.

use std::f32::consts::PI;

use oxideav_mp3::{
    alias_reduce, crc16_layer3_lsf, decode_huffman, decode_scalefactors, imdct_granule,
    lsf_scale_params, parse_header, parse_side_info, pcm_f32_to_i16, process_stereo, reorder,
    requantize, synth_granule, BlockType, ChannelMode, FrameWalker, GranuleChannel, ImdctState,
    MainDataReader, Mp3Encoder, MpegVersion, Reservoir, StreamEncodeError, SynthState,
    XminThresholds, PCM_PER_GRANULE,
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

/// LSF part2 (scalefactor) bit count for one granule-channel, derived
/// from `scalefac_compress` per ISO/IEC 13818-3 §2.4.3.2: the sum of
/// `slen[p] · nr_of_sfb[p]` over the four partitions. The encoder
/// never emits intensity-coded right channels at LSF (rejected at
/// constructor time), so the non-intensity branch always applies.
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

/// Decode an LSF (or MPEG-2.5) Layer III stream produced by the
/// encoder under test into per-channel `i16` PCM, using the crate's
/// own primitives. Handles CRC frames (skips the 2-byte word), LSF
/// part2 skipping, and joint-stereo MS frames via `process_stereo`.
fn decode_lsf_stream(bytes: &[u8]) -> Vec<Vec<i16>> {
    let mut reservoir = Reservoir::new();
    let mut synth: Vec<SynthState> = Vec::new();
    let mut imdct: Vec<ImdctState> = Vec::new();
    let mut out: Vec<Vec<i16>> = Vec::new();

    for frame in FrameWalker::new(bytes) {
        let hdr = parse_header(&frame.data[..4]).expect("header parse");
        assert!(
            hdr.version.is_lsf(),
            "expected an LSF frame, got {:?}",
            hdr.version
        );
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
            // §2.4.3.4.8 reorder (identity for long blocks, required
            // for short / mixed granules).
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
                out[ch].push(pcm_f32_to_i16(p));
            }
        }
    }
    out
}

/// Encoder + decoder fixed group delay (481 polyphase + 576 MDCT) plus
/// a filterbank warm-up margin, in 576-sample LSF granules.
const WARMUP_GRANULES: usize = 8;
const TOTAL_DELAY: usize = 1057;

/// Encode `pcm` (mono) and assert the self-decode PSNR over the
/// steady-state region exceeds `min_psnr`.
fn assert_mono_roundtrip(enc: Mp3Encoder, pcm: &[i16], min_psnr: f32) -> Vec<u8> {
    let mut enc = enc;
    enc.push_samples(pcm).expect("push pcm");
    let mut out: Vec<u8> = Vec::new();
    let n = enc.finish(&mut out).expect("encoder finish");
    assert_eq!(out.len(), n);
    let recon = decode_lsf_stream(&out);
    assert_eq!(recon.len(), 1, "mono stream must decode to one channel");
    let recon = &recon[0];
    let warmup = WARMUP_GRANULES * PCM_PER_GRANULE;
    let head = warmup + TOTAL_DELAY;
    assert!(recon.len() > head, "recon too short: {}", recon.len());
    let cmp = recon
        .len()
        .saturating_sub(head)
        .min(pcm.len().saturating_sub(warmup));
    let p = psnr(&pcm[warmup..warmup + cmp], &recon[head..head + cmp]);
    eprintln!("LSF mono self-decode PSNR = {p} dB over {cmp} samples");
    assert!(p > min_psnr, "PSNR too low: {p} dB (need > {min_psnr})");
    out
}

#[test]
fn lsf_mono_22050_sine_roundtrip() {
    const SR: u32 = 22_050;
    let pcm = sine_pcm(SR as usize, 440.0, SR as f32, 0.5); // 1 s
    let enc = Mp3Encoder::new(64, SR, ChannelMode::SingleChannel).expect("encoder build");
    let out = assert_mono_roundtrip(enc, &pcm, 20.0);

    // Framing checks: every frame is MPEG-2, one granule, LSF side
    // info, on the LSF ladder, 9-byte mono side info.
    let mut frames = 0usize;
    for f in FrameWalker::new(&out) {
        let hdr = parse_header(&f.data[..4]).unwrap();
        assert_eq!(hdr.version, MpegVersion::Mpeg2);
        assert_eq!(hdr.samples_per_frame(), 576);
        assert_eq!(hdr.bitrate_kbps, Some(64));
        let si = parse_side_info(&hdr, &f.data[4..]).unwrap();
        assert!(si.lsf);
        assert_eq!(si.granule_count, 1);
        assert_eq!(si.byte_len(), 9);
        frames += 1;
    }
    // 1 s @ 22.05 kHz = 22050 / 576 ≈ 38.3 frames (+1 tail round-up).
    assert!(
        (38..=40).contains(&frames),
        "frame count out of range: {frames}"
    );
}

#[test]
fn lsf_mono_24000_and_16000_roundtrip() {
    for (sr, br) in [(24_000u32, 64u32), (16_000, 48)] {
        let pcm = sine_pcm(sr as usize / 2, 330.0, sr as f32, 0.5); // 0.5 s
        let enc = Mp3Encoder::new(br, sr, ChannelMode::SingleChannel).expect("encoder build");
        let out = assert_mono_roundtrip(enc, &pcm, 20.0);
        let first = FrameWalker::new(&out).next().expect("one frame");
        let hdr = parse_header(&first.data[..4]).unwrap();
        assert_eq!(hdr.version, MpegVersion::Mpeg2);
        assert_eq!(hdr.sample_rate_hz, sr);
    }
}

#[test]
fn mpeg25_mono_11025_roundtrip() {
    // MPEG-2.5 11.025 kHz: the LSF framing on the extension rate. The
    // scalefactor-band tables are the real grounded tables now
    // (`mpeg2.5-scalefactor-bands.md`, #147/#151): 11.025 kHz reuses
    // the in-repo ISO/IEC 13818-3 22.05 kHz LSF long+short tables
    // verbatim, threaded through `long_band_starts` / `short_band_starts`
    // by the shared band-boundary functions. The self-decode round-trip
    // confirms the encoder and decoder agree on that band layout.
    const SR: u32 = 11_025;
    let pcm = sine_pcm(SR as usize, 220.0, SR as f32, 0.5); // 1 s
    let enc = Mp3Encoder::new(32, SR, ChannelMode::SingleChannel).expect("encoder build");
    let out = assert_mono_roundtrip(enc, &pcm, 20.0);
    let first = FrameWalker::new(&out).next().expect("one frame");
    let hdr = parse_header(&first.data[..4]).unwrap();
    assert_eq!(hdr.version, MpegVersion::Mpeg25);
    assert_eq!(hdr.samples_per_frame(), 576);
}

#[test]
fn mpeg25_mono_12000_roundtrip() {
    // MPEG-2.5 12 kHz: reuses the in-repo ISO/IEC 13818-3 24 kHz LSF
    // long+short scalefactor-band tables verbatim
    // (`mpeg2.5-scalefactor-bands.md`). Same band-aligned encode path as
    // 11.025 kHz, distinct table shape (24 kHz LSF vs 22.05 kHz LSF).
    const SR: u32 = 12_000;
    let pcm = sine_pcm(SR as usize, 300.0, SR as f32, 0.5);
    let enc = Mp3Encoder::new(32, SR, ChannelMode::SingleChannel).expect("encoder build");
    let out = assert_mono_roundtrip(enc, &pcm, 20.0);
    let first = FrameWalker::new(&out).next().expect("one frame");
    let hdr = parse_header(&first.data[..4]).unwrap();
    assert_eq!(hdr.version, MpegVersion::Mpeg25);
    assert_eq!(hdr.sample_rate_hz, 12_000);
    assert_eq!(hdr.samples_per_frame(), 576);
}

#[test]
fn mpeg25_mono_8000_roundtrip() {
    // MPEG-2.5 8 kHz: the genuinely distinct Fraunhofer table — its top
    // long bands (sfb 17..21) collapse to width 2 at the 4 kHz Nyquist
    // (`mpeg2.5-scalefactor-bands.md`, `LONG_STARTS_MPEG25_8`), and the
    // short table bands 9..11 are width-2 fillers. This round-trip
    // exercises the encoder's quantization / inner+outer loops over a
    // band layout that exists at no other rate, so a band-misalignment
    // (e.g. silently falling back to a 16 kHz LSF table) would break it.
    // Use a tone well below the 4 kHz Nyquist so the transmitted bands
    // carry real energy.
    const SR: u32 = 8_000;
    let pcm = sine_pcm(SR as usize, 1000.0, SR as f32, 0.5);
    let enc = Mp3Encoder::new(32, SR, ChannelMode::SingleChannel).expect("encoder build");
    let out = assert_mono_roundtrip(enc, &pcm, 20.0);
    let first = FrameWalker::new(&out).next().expect("one frame");
    let hdr = parse_header(&first.data[..4]).unwrap();
    assert_eq!(hdr.version, MpegVersion::Mpeg25);
    assert_eq!(hdr.sample_rate_hz, 8_000);
    assert_eq!(hdr.samples_per_frame(), 576);
}

#[test]
fn mpeg25_mono_8000_short_blocks_band_aligned_roundtrip() {
    // 8 kHz forced short blocks: drives the distinct
    // `SHORT_STARTS_MPEG25_8` short-band layout (bands 9..11 width-2
    // fillers, band 12 the residual sweep) through the reorder + outer
    // loop on the one rate whose short table has no LSF sibling. A
    // self-consistent encode↔decode here proves the encoder's short
    // reorder and the decoder's reorder both index the same 8 kHz short
    // table.
    const SR: u32 = 8_000;
    // Need > WARMUP_GRANULES granules + TOTAL_DELAY samples of recon to
    // reach steady state; at 8 kHz that is ~5665 samples, so encode a
    // full ~1.25 s.
    let pcm = sine_pcm(SR as usize * 5 / 4, 1200.0, SR as f32, 0.5);
    let mut enc = Mp3Encoder::new(32, SR, ChannelMode::SingleChannel).expect("encoder build");
    enc.force_short_blocks_for_testing(true)
        .expect("force short blocks");
    assert!(enc.force_short_blocks_enabled());
    let out = assert_mono_roundtrip(enc, &pcm, 18.0);
    let first = FrameWalker::new(&out).next().expect("one frame");
    let hdr = parse_header(&first.data[..4]).unwrap();
    assert_eq!(hdr.version, MpegVersion::Mpeg25);
    assert_eq!(hdr.sample_rate_hz, 8_000);
}

#[test]
fn mpeg25_threshold_in_quiet_psychoacoustic_roundtrip_all_rates() {
    // The milestone wiring: the threshold-in-quiet psychoacoustic path
    // (`XminThresholds::threshold_in_quiet` → the §C.1.5.4.3
    // distortion-control outer loop) must produce a band-aligned,
    // self-decodable MP3 stream at every MPEG-2.5 rate. The per-band
    // threshold is built over the rate's `long_band_starts` /
    // `short_band_starts`, so a band-table regression at 8 kHz (the
    // distinct Fraunhofer table) would mis-shape `xmin` and surface here.
    for sr in [8_000u32, 11_025, 12_000] {
        let pcm = sine_pcm(sr as usize, 700.0, sr as f32, 0.5);
        let enc = Mp3Encoder::new_with_threshold_in_quiet(32, sr, ChannelMode::SingleChannel)
            .expect("threshold-in-quiet MPEG-2.5 encoder build");
        let out = assert_mono_roundtrip(enc, &pcm, 18.0);
        let first = FrameWalker::new(&out).next().expect("one frame");
        let hdr = parse_header(&first.data[..4]).unwrap();
        assert_eq!(hdr.version, MpegVersion::Mpeg25);
        assert_eq!(hdr.sample_rate_hz, sr);
    }
}

#[test]
fn mpeg25_threshold_in_quiet_band_vector_is_band_aligned() {
    // Directly inspect the per-band threshold vector the encoder
    // installs at each MPEG-2.5 rate. The threshold-in-quiet bowl is a
    // function of each band's centre frequency, which is derived from
    // that rate's scalefactor-band start indices. The vector must be
    // (a) all-finite-positive over the 21 transmitted long bands and
    // (b) genuinely non-uniform (a real bowl, not a degenerate
    // constant) — the latter is the witness that the band partitioning
    // actually shaped the threshold rather than collapsing to a single
    // value. The −12 dB high-bitrate offset is irrelevant here; 32 kbps
    // mono → 0 dB offset, the low-rate transparency reference.
    for sr in [8_000u32, 11_025, 12_000] {
        let xmin = XminThresholds::threshold_in_quiet(sr, MpegVersion::Mpeg25, 32);
        // 21 transmitted long bands (band 21 is the fixed non-transmitted
        // filler); all must be finite and strictly positive.
        let mut min = f64::INFINITY;
        let mut max = 0.0_f64;
        for &v in xmin.long.iter().take(21) {
            assert!(
                v.is_finite() && v > 0.0,
                "non-positive xmin at {sr} Hz: {v}"
            );
            min = min.min(v);
            max = max.max(v);
        }
        // Real spectral shape: the loudest-vs-quietest band ratio must be
        // appreciable. (At MPEG-2.5 the audio band spans only a few kHz,
        // but the threshold-in-quiet bowl still varies by orders of
        // magnitude between the mid-band minimum and the band edges.)
        assert!(
            max / min > 4.0,
            "xmin vector at {sr} Hz is too flat (max/min = {}), band shaping did not apply",
            max / min
        );
    }
}

#[test]
fn lsf_stereo_22050_independent_roundtrip() {
    const SR: u32 = 22_050;
    let n = SR as usize / 2; // 0.5 s
    let left = sine_pcm(n, 440.0, SR as f32, 0.45);
    let right = sine_pcm(n, 660.0, SR as f32, 0.45);
    let mut interleaved = Vec::with_capacity(n * 2);
    for i in 0..n {
        interleaved.push(left[i]);
        interleaved.push(right[i]);
    }
    let mut enc = Mp3Encoder::new(128, SR, ChannelMode::Stereo).expect("encoder build");
    enc.push_samples(&interleaved).expect("push pcm");
    let mut out: Vec<u8> = Vec::new();
    enc.finish(&mut out).expect("encoder finish");

    // 17-byte stereo LSF side info on every frame.
    for f in FrameWalker::new(&out) {
        let hdr = parse_header(&f.data[..4]).unwrap();
        let si = parse_side_info(&hdr, &f.data[4..]).unwrap();
        assert_eq!(si.byte_len(), 17);
    }

    let recon = decode_lsf_stream(&out);
    assert_eq!(recon.len(), 2);
    let warmup = WARMUP_GRANULES * PCM_PER_GRANULE;
    let head = warmup + TOTAL_DELAY;
    for (ch, src) in [(0usize, &left), (1, &right)] {
        let r = &recon[ch];
        let cmp = r
            .len()
            .saturating_sub(head)
            .min(src.len().saturating_sub(warmup));
        let p = psnr(&src[warmup..warmup + cmp], &r[head..head + cmp]);
        eprintln!("LSF stereo ch{ch} self-decode PSNR = {p} dB");
        assert!(p > 20.0, "ch{ch} PSNR too low: {p} dB");
    }
}

#[test]
fn lsf_ms_joint_stereo_roundtrip() {
    const SR: u32 = 22_050;
    let n = SR as usize / 2;
    // Strongly-correlated channels: same tone, slightly different
    // amplitude — the MS picker's natural habitat.
    let left = sine_pcm(n, 440.0, SR as f32, 0.5);
    let right = sine_pcm(n, 440.0, SR as f32, 0.4);
    let mut interleaved = Vec::with_capacity(n * 2);
    for i in 0..n {
        interleaved.push(left[i]);
        interleaved.push(right[i]);
    }
    let mut enc = Mp3Encoder::new_joint_stereo_ms(128, SR).expect("MS encoder build");
    enc.push_samples(&interleaved).expect("push pcm");
    let mut out: Vec<u8> = Vec::new();
    enc.finish(&mut out).expect("encoder finish");

    // Every frame: joint-stereo mode, MS mode_extension, LSF framing.
    for f in FrameWalker::new(&out) {
        let hdr = parse_header(&f.data[..4]).unwrap();
        assert_eq!(hdr.mode, ChannelMode::JointStereo);
        assert!(hdr.mode_extension.ms_stereo);
        assert_eq!(hdr.version, MpegVersion::Mpeg2);
    }

    let recon = decode_lsf_stream(&out);
    let warmup = WARMUP_GRANULES * PCM_PER_GRANULE;
    let head = warmup + TOTAL_DELAY;
    for (ch, src) in [(0usize, &left), (1, &right)] {
        let r = &recon[ch];
        let cmp = r
            .len()
            .saturating_sub(head)
            .min(src.len().saturating_sub(warmup));
        let p = psnr(&src[warmup..warmup + cmp], &r[head..head + cmp]);
        eprintln!("LSF MS ch{ch} self-decode PSNR = {p} dB");
        assert!(p > 20.0, "ch{ch} PSNR too low: {p} dB");
    }
}

#[test]
fn lsf_outer_loop_writes_scalefac_compress_399() {
    const SR: u32 = 22_050;
    let pcm = sine_pcm(SR as usize / 2, 880.0, SR as f32, 0.5);
    let enc = Mp3Encoder::new_with_outer_loop(
        64,
        SR,
        ChannelMode::SingleChannel,
        oxideav_mp3::stream_encoder::DEFAULT_OUTER_LOOP_THRESHOLD,
    )
    .expect("outer-loop encoder build");
    let out = assert_mono_roundtrip(enc, &pcm, 20.0);

    // Every non-empty granule carries the LSF outer-loop
    // scalefac_compress 399 (slen (4,4,3,3) / partition (6,5,5,5) per
    // ISO/IEC 13818-3 §2.4.3.2) — never the MPEG-1 value 15, whose
    // LSF decode would partition completely differently.
    let mut saw_399 = false;
    for f in FrameWalker::new(&out) {
        let hdr = parse_header(&f.data[..4]).unwrap();
        let si = parse_side_info(&hdr, &f.data[4..]).unwrap();
        let gc = &si.granules[0][0];
        if gc.part2_3_length > 0 {
            assert_eq!(
                gc.scalefac_compress, 399,
                "LSF outer-loop granule must carry scalefac_compress 399"
            );
            saw_399 = true;
        }
    }
    assert!(saw_399, "no outer-loop granule observed");
}

#[test]
fn lsf_force_short_blocks_roundtrip() {
    const SR: u32 = 22_050;
    let pcm = sine_pcm(SR as usize / 2, 550.0, SR as f32, 0.5);
    let mut enc = Mp3Encoder::new(64, SR, ChannelMode::SingleChannel).expect("encoder build");
    enc.force_short_blocks_for_testing(true)
        .expect("force short blocks");
    assert!(enc.force_short_blocks_enabled());
    enc.push_samples(&pcm).expect("push pcm");
    let mut out: Vec<u8> = Vec::new();
    enc.finish(&mut out).expect("encoder finish");

    // Every granule is window-switched Short; decode exercises the
    // 13818-3 Table B.2 short band tables + the LSF short-block
    // scalefactor partition.
    for f in FrameWalker::new(&out) {
        let hdr = parse_header(&f.data[..4]).unwrap();
        let si = parse_side_info(&hdr, &f.data[4..]).unwrap();
        let gc = &si.granules[0][0];
        assert!(gc.window_switching_flag);
        assert_eq!(gc.block_type, BlockType::Short);
    }
    let recon = decode_lsf_stream(&out);
    assert!(!recon[0].is_empty());
    // Short-window coding of a steady tone is lossier than long-block
    // coding; assert a sane floor rather than transparency.
    let warmup = WARMUP_GRANULES * PCM_PER_GRANULE;
    let head = warmup + TOTAL_DELAY;
    let r = &recon[0];
    let cmp = r
        .len()
        .saturating_sub(head)
        .min(pcm.len().saturating_sub(warmup));
    let p = psnr(&pcm[warmup..warmup + cmp], &r[head..head + cmp]);
    eprintln!("LSF forced-short self-decode PSNR = {p} dB");
    assert!(p > 10.0, "short-block PSNR too low: {p} dB");
}

#[test]
fn lsf_crc_frames_carry_valid_checksum() {
    const SR: u32 = 22_050;
    let pcm = sine_pcm(SR as usize / 4, 440.0, SR as f32, 0.5);
    let mut enc = Mp3Encoder::new(64, SR, ChannelMode::SingleChannel).expect("encoder build");
    enc.with_protection_bit(true);
    enc.push_samples(&pcm).expect("push pcm");
    let mut out: Vec<u8> = Vec::new();
    enc.finish(&mut out).expect("encoder finish");

    let mut frames = 0usize;
    for f in FrameWalker::new(&out) {
        let hdr_bytes: [u8; 4] = f.data[..4].try_into().unwrap();
        let hdr = parse_header(&hdr_bytes).unwrap();
        assert!(hdr.crc_protected);
        let wire_crc = u16::from_be_bytes([f.data[4], f.data[5]]);
        // LSF mono side info = 9 bytes starting after the CRC word.
        let si_slice = &f.data[6..6 + 9];
        let expected = crc16_layer3_lsf(&hdr_bytes, si_slice, 1);
        assert_eq!(wire_crc, expected, "frame at {} CRC mismatch", f.offset);
        frames += 1;
    }
    assert!(frames > 0);
    // The CRC-bearing stream still decodes (the decoder skips the
    // 2-byte word via `hdr.crc_protected`).
    let recon = decode_lsf_stream(&out);
    assert!(!recon[0].is_empty());
}

#[test]
fn lsf_vbr_picks_indices_on_lsf_ladder() {
    const SR: u32 = 22_050;
    let pcm = sine_pcm(SR as usize / 2, 440.0, SR as f32, 0.5);
    let mut enc = Mp3Encoder::new(64, SR, ChannelMode::SingleChannel).expect("encoder build");
    enc.enable_vbr(16, 64).expect("VBR config on LSF ladder");
    enc.push_samples(&pcm).expect("push pcm");
    let mut out: Vec<u8> = Vec::new();
    enc.finish(&mut out).expect("encoder finish");

    for f in FrameWalker::new(&out) {
        let hdr = parse_header(&f.data[..4]).unwrap();
        let kbps = hdr.bitrate_kbps.expect("non-free-format");
        assert!(
            (16..=64).contains(&kbps),
            "VBR frame bitrate {kbps} outside [16, 64]"
        );
        // 8 kbit/s is on the LSF ladder but below min; 80+ above max.
    }
    let recon = decode_lsf_stream(&out);
    assert!(!recon[0].is_empty());

    // MPEG-1-only ladder values must be rejected on the LSF ladder.
    let mut enc2 = Mp3Encoder::new(64, SR, ChannelMode::SingleChannel).unwrap();
    assert!(matches!(
        enc2.enable_vbr(16, 320),
        Err(StreamEncodeError::InvalidVbrConfig)
    ));
}

#[test]
fn lsf_intensity_constructors_build() {
    const SR: u32 = 22_050;
    // Intensity stereo on LSF is supported as of r286 (the 13818-3
    // §2.4.3.2 `int_scalefac_compress` right-channel format lands in
    // `tests/lsf_intensity_roundtrip.rs`); the three intensity
    // constructors now build cleanly on an LSF rate.
    assert!(Mp3Encoder::new_joint_stereo_is(64, SR, 7).is_ok());
    assert!(Mp3Encoder::new_joint_stereo_ms_is(64, SR, 7).is_ok());
    assert!(Mp3Encoder::new_joint_stereo_auto_is(64, SR, 7).is_ok());
}

#[test]
fn lsf_auto_block_type_accepted() {
    // As of r287 the §C.1.5.2 auto block-type scheduler is version
    // agnostic: the frame walk steps the state machine once per LSF
    // frame (single 576-sample granule, ISO/IEC 13818-3) instead of
    // twice (MPEG-1 two-granule geometry). Enabling it on an LSF
    // encoder now succeeds rather than returning `LsfUnsupported`.
    const SR: u32 = 22_050;
    let mut enc = Mp3Encoder::new(64, SR, ChannelMode::SingleChannel).unwrap();
    assert!(enc.enable_auto_block_type(2.0).is_ok());
    assert!(enc.auto_block_type_enabled());
    // The with-mixed variant is equally available.
    let mut enc2 = Mp3Encoder::new(64, SR, ChannelMode::SingleChannel).unwrap();
    assert!(enc2.enable_auto_block_type_with_mixed(2.0, 0.5).is_ok());
    assert!(enc2.auto_block_type_enabled());
}
