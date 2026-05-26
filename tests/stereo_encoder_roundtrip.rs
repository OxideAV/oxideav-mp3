//! Integration test for the round-145 stereo encoder path.
//!
//! This test exercises the [`oxideav_mp3::Mp3Encoder`] stereo encode
//! path end-to-end:
//!
//! 1. Synthesise one second of two **distinct** sine tones (440 Hz on
//!    L, 880 Hz on R) at 44.1 kHz / `i16` PCM, interleaved as
//!    `[L0, R0, L1, R1, …]` per ISO/IEC 11172-3 §2.4.2.1 stereo
//!    sample order.
//! 2. Encode them into a CBR 192 kbit/s MPEG-1 Layer III stereo byte
//!    stream via [`oxideav_mp3::Mp3Encoder`] with
//!    [`oxideav_mp3::ChannelMode::Stereo`] (`mode = '00'`,
//!    independent-channel coding, `mode_extension = '00'`).
//! 3. Confirm every frame walks via [`oxideav_mp3::FrameWalker`] and
//!    every header reports `channel_count() == 2` with side-info size
//!    32 bytes per §2.4.1.7 stereo layout.
//! 4. Run the in-tree decode pipeline (huffman → requantize → alias →
//!    imdct → synth) on each (gr, ch) and gather **separate** L / R
//!    PCM streams.
//! 5. Assert per-channel PSNR > 20 dB and per-channel **fundamental
//!    energy concentration** in the correct half of the spectrum
//!    (L should carry the 440 Hz tone, R the 880 Hz tone — no
//!    cross-channel leakage). No external decoder is invoked.

use std::f32::consts::PI;

use oxideav_mp3::{
    alias_reduce, decode_huffman, decode_scalefactors, imdct_granule, parse_header,
    parse_side_info, requantize, synth_granule, ChannelMode, FrameWalker, ImdctState,
    MainDataReader, Mp3Encoder, Reservoir, SynthState, PCM_PER_GRANULE,
};

/// Build an interleaved `[L0, R0, L1, R1, …]` stereo `i16` PCM stream
/// from two independent sine tones.
fn stereo_sine_pcm(n: usize, freq_l: f32, freq_r: f32, sample_rate_hz: f32, amp: f32) -> Vec<i16> {
    let two_pi = 2.0 * PI;
    let mut out = Vec::with_capacity(n * 2);
    let scale = amp * (i16::MAX as f32);
    for i in 0..n {
        let t = i as f32 / sample_rate_hz;
        let l = (two_pi * freq_l * t).sin() * scale;
        let r = (two_pi * freq_r * t).sin() * scale;
        out.push(l.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16);
        out.push(r.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16);
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

/// Decode a stereo MP3 byte stream into two per-channel `i16` PCM
/// vectors `(left, right)`. Mirrors the mono decoder used by
/// `tests/stream_encoder_roundtrip.rs` but routes each granule's two
/// channels into the correct output channel; per-channel `SynthState`
/// + `ImdctState` are kept independent (one per channel).
///
/// Uses only the crate's own primitives — no external decoder.
fn decode_mp3_stereo(bytes: &[u8]) -> (Vec<i16>, Vec<i16>) {
    let mut reservoir = Reservoir::new();
    // Two channels → two independent synth + imdct states.
    let mut synth: Vec<SynthState> = vec![SynthState::new(), SynthState::new()];
    let mut imdct: Vec<ImdctState> = vec![ImdctState::new(), ImdctState::new()];
    let mut out_l: Vec<i16> = Vec::new();
    let mut out_r: Vec<i16> = Vec::new();

    for frame in FrameWalker::new(bytes) {
        let hdr = parse_header(&frame.data[..4]).expect("header parse");
        // The encoder writes `protection_bit = 1` (no CRC) by default;
        // side_info starts immediately at byte 4.
        let si_start = 4;
        let si = parse_side_info(&hdr, &frame.data[si_start..]).expect("side_info parse");
        assert_eq!(si.channels, 2, "expected stereo side info");
        let si_len = si.byte_len();
        let main_data_slot = &frame.data[si_start + si_len..];
        let run = reservoir
            .assemble(usize::from(si.main_data_begin), main_data_slot)
            .expect("reservoir assemble");
        let fsf = decode_scalefactors(&hdr, &si, &run).expect("scalefactors decode");

        // §2.4.1.7 `main_data()` order: gr, then ch — Huffman bits
        // concatenated; our encoder writes `scalefac_compress = 0` so
        // part2 is zero bits per granule-channel and `part2_3_length`
        // is entirely part3. Walk by a running bit cursor.
        let mut bit_cursor = 0usize;
        for gr in 0..si.granule_count as usize {
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
                let xr = requantize(&is, gc, sf, hdr.sample_rate_hz, hdr.version);
                let xar = alias_reduce(&xr, gc);
                let subband_time = imdct_granule(&xar, gc, &mut imdct[ch]);
                let pcm_f32 = synth_granule(&subband_time, &mut synth[ch]);
                let sink = if ch == 0 { &mut out_l } else { &mut out_r };
                for &p in pcm_f32.iter().take(PCM_PER_GRANULE) {
                    let v = p * f32::from(i16::MAX);
                    sink.push(v.clamp(i16::MIN as f32, i16::MAX as f32) as i16);
                }
                bit_cursor += gc.part2_3_length as usize;
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

/// Naive single-bin DFT magnitude at `freq_hz`, in `[0, 1]` after
/// normalising by the input's peak amplitude. Used as a cheap
/// per-channel frequency-content sanity check (L should carry the
/// L-channel tone, R should carry the R-channel tone — no
/// cross-channel leakage). Not a precise spectral estimate; just a
/// "does the dominant energy land at the right frequency" probe.
fn bin_magnitude(samples: &[i16], freq_hz: f32, sample_rate_hz: f32) -> f64 {
    let n = samples.len();
    if n == 0 {
        return 0.0;
    }
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut re = 0.0f64;
    let mut im = 0.0f64;
    for (i, &s) in samples.iter().enumerate() {
        let t = i as f64 / sample_rate_hz as f64;
        let phase = two_pi * f64::from(freq_hz) * t;
        re += f64::from(s) * phase.cos();
        im += f64::from(s) * phase.sin();
    }
    let mag = (re * re + im * im).sqrt();
    mag / (n as f64 * f64::from(i16::MAX))
}

#[test]
fn stereo_one_second_self_decode_per_channel_psnr() {
    const SR: u32 = 44_100;
    // 192 kbit/s is the standard reference rate for stereo content on
    // the §2.4.2.3 ladder (96 kbit/s per channel after the 50/50
    // budget split the encoder applies). The mono test at 128 used
    // 128 kbit/s for one channel; doubling to 192 gives stereo a
    // comparable per-channel bit budget.
    const BR: u32 = 192;
    let n = SR as usize; // one second
    let pcm = stereo_sine_pcm(n, 440.0, 880.0, SR as f32, 0.5);

    // Encode through the direct stream API in stereo mode.
    let mut enc = Mp3Encoder::new(BR, SR, ChannelMode::Stereo).expect("stereo encoder build");
    enc.push_samples(&pcm).expect("push interleaved pcm");
    let mut out: Vec<u8> = Vec::new();
    let bytes = enc.finish(&mut out).expect("encoder finish");
    assert_eq!(out.len(), bytes);
    assert!(bytes > 2_000, "stereo encoded stream too small: {bytes}");

    // Every frame should report 2 channels with the 32-byte stereo
    // side-info block.
    let mut frame_count = 0usize;
    for frame in FrameWalker::new(&out) {
        let hdr = parse_header(&frame.data[..4]).expect("header parse");
        assert_eq!(hdr.channel_count(), 2);
        // mode '00' (stereo) → mode_extension is irrelevant; verify
        // the wire bits.
        let mode = (frame.data[3] >> 6) & 0b11;
        assert_eq!(mode, 0b00, "expected stereo mode '00', got '{mode:02b}'");
        let si = parse_side_info(&hdr, &frame.data[4..]).expect("si parse");
        assert_eq!(si.byte_len(), 32, "expected MPEG-1 stereo SI = 32 B");
        frame_count += 1;
    }
    assert!(
        (38..=40).contains(&frame_count),
        "frame count out of range: {frame_count}"
    );

    // Per-channel decode via our own primitives.
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
        "stereo self-decode per-channel PSNR: L = {pl} dB (440 Hz), R = {pr} dB (880 Hz), n = {} samples",
        cmp_len
    );

    // Per-channel PSNR floor matches the mono test's >20 dB bar
    // (without a psy model the absolute level is bounded; the round's
    // goal is to demonstrate per-channel correctness, not bit budget
    // efficiency).
    assert!(pl > 20.0, "L-channel PSNR too low: {pl} dB");
    assert!(pr > 20.0, "R-channel PSNR too low: {pr} dB");

    // Frequency-content sanity check: L should be dominated by the
    // 440 Hz tone, R by the 880 Hz tone, with neither carrying the
    // other channel's tone above noise.
    let l_at_440 = bin_magnitude(&recon_l[head_recon..head_recon + cmp_len], 440.0, SR as f32);
    let l_at_880 = bin_magnitude(&recon_l[head_recon..head_recon + cmp_len], 880.0, SR as f32);
    let r_at_440 = bin_magnitude(&recon_r[head_recon..head_recon + cmp_len], 440.0, SR as f32);
    let r_at_880 = bin_magnitude(&recon_r[head_recon..head_recon + cmp_len], 880.0, SR as f32);
    eprintln!("L bin mag: 440 Hz = {l_at_440:.4}, 880 Hz = {l_at_880:.4}");
    eprintln!("R bin mag: 440 Hz = {r_at_440:.4}, 880 Hz = {r_at_880:.4}");
    assert!(
        l_at_440 > l_at_880 * 4.0,
        "L channel cross-talk at 880 Hz too high (440={l_at_440}, 880={l_at_880})"
    );
    assert!(
        r_at_880 > r_at_440 * 4.0,
        "R channel cross-talk at 440 Hz too high (440={r_at_440}, 880={r_at_880})"
    );
}

#[test]
fn stereo_silence_decodes_to_near_zero_both_channels() {
    const SR: u32 = 44_100;
    let mut enc = Mp3Encoder::new(192, SR, ChannelMode::Stereo).expect("stereo encoder");
    // 3 frames of all-zero interleaved PCM (3 × 1152 × 2 samples).
    let zero_pcm = vec![0i16; 3 * 1152 * 2];
    enc.push_samples(&zero_pcm).expect("push zeros");
    let mut out: Vec<u8> = Vec::new();
    enc.finish(&mut out).expect("finish");

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
fn stereo_mono_first_frame_byte_layout() {
    // The first emitted frame from a stereo encoder must have:
    //  * sync `0xFFF` in the top 12 bits,
    //  * mode `'00'` (stereo) in byte 3 bits 7..6,
    //  * channel_count == 2 → side info 32 bytes (MPEG-1 §2.4.1.7).
    const SR: u32 = 44_100;
    let mut enc = Mp3Encoder::new(192, SR, ChannelMode::Stereo).expect("stereo encoder");
    let pcm = stereo_sine_pcm(1152 * 2, 440.0, 880.0, SR as f32, 0.5);
    enc.push_samples(&pcm).unwrap();
    let mut out: Vec<u8> = Vec::new();
    enc.finish(&mut out).unwrap();
    let first = FrameWalker::new(&out).next().expect("at least one frame");
    assert_eq!(first.data[0], 0xFF);
    assert_eq!(first.data[1] & 0xE0, 0xE0);
    assert_eq!(first.data[3] & 0xC0, 0x00, "mode should be '00' stereo");
    let hdr = parse_header(&first.data[..4]).unwrap();
    assert_eq!(hdr.channel_count(), 2);
    let si = parse_side_info(&hdr, &first.data[4..]).unwrap();
    assert_eq!(si.byte_len(), 32);
    assert_eq!(si.channels, 2);
}
