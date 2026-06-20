//! MPEG-2 LSF decode accuracy against the staged reference PCM.
//!
//! `docs/audio/mp3/fixtures/layer3-mpeg2-22050-64kbps/` carries both
//! the encoded stream (`input.mp3`, produced by the docs
//! collaborator's reference toolchain) and the black-box reference
//! decode (`expected.wav`, the toolchain's PCM output — consumed here
//! as opaque fixture data). With the ISO/IEC 13818-3:1997 Table B.2
//! scalefactor-band tables wired into the requantizer, this crate's
//! own decode of the LSF stream must land close to that reference:
//! the two decoders differ only in float rounding and synthesis
//! windowing arithmetic, not in band layout.
//!
//! The test decodes `input.mp3` through the direct chain, aligns the
//! result against `expected.wav` by exhaustive lag search (different
//! decoders trim different amounts of codec delay), and asserts the
//! normalized RMS error over the overlap is small. Before the Table
//! B.2 tables landed, the requantizer applied the MPEG-1 44.1 kHz
//! band layout at 22.05 kHz, which mis-scaled every line above the
//! first band-boundary divergence and pushed this metric to gross
//! error; with the correct tables it sits in the float-rounding
//! regime.
//!
//! Skips (with a log line) when the workspace `docs/` tree is absent
//! (standalone-crate CI checkout), matching `tests/docs_corpus.rs`.

use std::fs;
use std::path::PathBuf;

use oxideav_mp3::{
    alias_reduce, decode_huffman, decode_scalefactors, imdct_granule, parse_header,
    parse_side_info, process_stereo, requantize, synth_granule, ChannelMode, FrameWalker,
    ImdctState, MainDataReader, Reservoir, SynthState, PCM_PER_GRANULE,
};

fn fixture_dir() -> Option<PathBuf> {
    let p = PathBuf::from("../../docs/audio/mp3/fixtures/layer3-mpeg2-22050-64kbps");
    if p.join("input.mp3").exists() && p.join("expected.wav").exists() {
        Some(p)
    } else {
        None
    }
}

/// Strip a leading ID3v2 tag (synchsafe size) if present.
fn strip_id3v2(bytes: &[u8]) -> &[u8] {
    if bytes.len() >= 10 && &bytes[..3] == b"ID3" {
        let size = ((u32::from(bytes[6]) & 0x7F) << 21)
            | ((u32::from(bytes[7]) & 0x7F) << 14)
            | ((u32::from(bytes[8]) & 0x7F) << 7)
            | (u32::from(bytes[9]) & 0x7F);
        let total = 10 + size as usize;
        if total <= bytes.len() {
            return &bytes[total..];
        }
    }
    bytes
}

/// Direct-chain stereo decode (same wedge as
/// `tests/decoder_trait_lsf_roundtrip.rs`), returning f32 PCM in
/// [-1, 1] per channel.
fn decode_direct_f32(bytes: &[u8]) -> (Vec<f32>, Vec<f32>) {
    let mut reservoir = Reservoir::new();
    let mut synth: [SynthState; 2] = [SynthState::new(), SynthState::new()];
    let mut imdct: [ImdctState; 2] = [ImdctState::new(), ImdctState::new()];
    let mut out_l: Vec<f32> = Vec::new();
    let mut out_r: Vec<f32> = Vec::new();
    for (idx, frame) in FrameWalker::new(bytes).enumerate() {
        let hdr = parse_header(&frame.data[..4]).expect("header");
        let si_start = 4 + if hdr.crc_protected { 2 } else { 0 };
        // Skip the Xing/Info carrier frame: it is not audio, and its
        // slot bytes must not enter the bit reservoir (the reference
        // decoder skips it the same way — see the demuxer's
        // first-frame Xing path).
        if idx == 0 {
            let si_bytes = oxideav_mp3::side_info_len(hdr.version, hdr.channel_count());
            if oxideav_mp3::parse_xing_info(frame.data, si_bytes).is_some() {
                continue;
            }
        }
        let si = parse_side_info(&hdr, &frame.data[si_start..]).expect("side_info");
        let si_len = si.byte_len();
        let main_slot = &frame.data[si_start + si_len..];
        let run = match reservoir.assemble(usize::from(si.main_data_begin), main_slot) {
            Ok(r) => r,
            Err(_) => continue,
        };
        let fsf = decode_scalefactors(&hdr, &si, &run).expect("scalefactors");
        let nch = si.channels as usize;
        let mut bit_cursor = 0usize;
        for gr in 0..si.granule_count as usize {
            let mut xr_per_ch: Vec<[f32; 576]> = (0..nch).map(|_| [0.0; 576]).collect();
            for (ch, xr_slot) in xr_per_ch.iter_mut().enumerate() {
                let gc = &si.granules[gr][ch];
                let mut r = MainDataReader::new(&run);
                // Skip part-2 (scalefactor) bits so the reader lands on
                // the part-3 Huffman codewords; budget the remainder.
                let mut left = bit_cursor + fsf.part2_bits[gr][ch] as usize;
                while left >= 32 {
                    let _ = r.read(32);
                    left -= 32;
                }
                if left > 0 {
                    let _ = r.read(left as u32);
                }
                let part3_bits =
                    u32::from(gc.part2_3_length).saturating_sub(fsf.part2_bits[gr][ch]);
                let is = decode_huffman(&mut r, gc, part3_bits, hdr.sample_rate_hz, hdr.version)
                    .expect("huffman");
                let sf = &fsf.granules[gr][ch];
                *xr_slot = requantize(&is, gc, sf, hdr.sample_rate_hz, hdr.version);
                bit_cursor += gc.part2_3_length as usize;
            }
            if nch == 2 && hdr.mode == ChannelMode::JointStereo {
                let (l_split, r_split) = xr_per_ch.split_at_mut(1);
                let right_sf = &fsf.granules[gr][1];
                let right_gc = &si.granules[gr][1];
                process_stereo(
                    &mut l_split[0],
                    &mut r_split[0],
                    right_sf,
                    right_gc,
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
                let sink = if ch == 0 { &mut out_l } else { &mut out_r };
                sink.extend_from_slice(&pcm_f32[..PCM_PER_GRANULE]);
            }
        }
    }
    (out_l, out_r)
}

/// Minimal RIFF/WAVE reader for the 16-bit LE PCM `expected.wav`
/// fixtures: walks the chunk list, returns (channels, sample_rate,
/// deinterleaved f32 channel data).
fn read_wav_s16(path: &std::path::Path) -> (u16, u32, Vec<Vec<f32>>) {
    let bytes = fs::read(path).expect("read wav");
    assert!(bytes.len() > 44 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WAVE");
    let mut pos = 12usize;
    let mut channels = 0u16;
    let mut sample_rate = 0u32;
    let mut data: Vec<Vec<f32>> = Vec::new();
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let len = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let body = &bytes[pos + 8..(pos + 8 + len).min(bytes.len())];
        match id {
            b"fmt " => {
                let fmt_tag = u16::from_le_bytes(body[0..2].try_into().unwrap());
                assert_eq!(fmt_tag, 1, "expected.wav must be integer PCM");
                channels = u16::from_le_bytes(body[2..4].try_into().unwrap());
                sample_rate = u32::from_le_bytes(body[4..8].try_into().unwrap());
                let bits = u16::from_le_bytes(body[14..16].try_into().unwrap());
                assert_eq!(bits, 16, "expected.wav must be 16-bit");
            }
            b"data" => {
                assert!(channels > 0, "fmt chunk must precede data");
                data = vec![Vec::new(); channels as usize];
                for (i, chunk) in body.chunks_exact(2).enumerate() {
                    let v = i16::from_le_bytes([chunk[0], chunk[1]]);
                    data[i % channels as usize].push(f32::from(v) / 32768.0);
                }
            }
            _ => {}
        }
        pos += 8 + len + (len & 1);
    }
    (channels, sample_rate, data)
}

/// Best-correlation lag of `b` relative to `a` over `±max_lag`,
/// evaluated on up to `win` overlapping samples anchored at `a[anchor..]`
/// (a mid-file anchor avoids the codec warm-up transient), plus the
/// normalized RMS error at that lag over the steady-state region —
/// `a` with two 576-sample frames shaved off each end. The shaved
/// frames are where the two decoders legitimately differ: the head
/// carries the reference's gapless lead-in trim against our full
/// warm-up output, and the tail carries its end-padding trim. In
/// between, the decodes must track sample-for-sample.
fn align_and_error(a: &[f32], b: &[f32], max_lag: isize, win: usize) -> (isize, f64, f64) {
    let anchor = 4 * 576usize;
    let mut best = (0isize, f64::NEG_INFINITY);
    for lag in -max_lag..=max_lag {
        let mut dot = 0f64;
        let mut pa = 0f64;
        let mut pb = 0f64;
        let mut n = 0usize;
        for k in 0..win {
            let i = anchor + k;
            let j = i as isize + lag;
            if j < 0 || j as usize >= b.len() || i >= a.len() {
                continue;
            }
            let (x, y) = (f64::from(a[i]), f64::from(b[j as usize]));
            dot += x * y;
            pa += x * x;
            pb += y * y;
            n += 1;
        }
        if n > 1000 {
            let ncc = dot / (pa.sqrt() * pb.sqrt()).max(1e-30);
            if ncc > best.1 {
                best = (lag, ncc);
            }
        }
    }
    let (lag, peak_ncc) = best;
    let mut err = 0f64;
    let mut ref_pow = 0f64;
    let mut n = 0usize;
    let steady = 2 * 576..a.len().saturating_sub(2 * 576);
    for i in steady {
        let j = i as isize + lag;
        if j < 0 || j as usize >= b.len() {
            continue;
        }
        let d = f64::from(a[i]) - f64::from(b[j as usize]);
        err += d * d;
        ref_pow += f64::from(a[i]) * f64::from(a[i]);
        n += 1;
    }
    assert!(n > 1000, "overlap too small after alignment");
    (lag, (err / ref_pow.max(1e-30)).sqrt(), peak_ncc)
}

#[test]
fn lsf_fixture_decode_tracks_reference_pcm() {
    let Some(dir) = fixture_dir() else {
        eprintln!("skip: layer3-mpeg2-22050-64kbps fixture absent (standalone-crate CI checkout)");
        return;
    };
    let mp3 = fs::read(dir.join("input.mp3")).expect("read input.mp3");
    let wire = strip_id3v2(&mp3);
    let (our_l, our_r) = decode_direct_f32(wire);
    assert!(our_l.len() > 10_000, "decode produced too few samples");

    let (channels, rate, ref_pcm) = read_wav_s16(&dir.join("expected.wav"));
    assert_eq!(channels, 2);
    assert_eq!(rate, 22_050);

    // ±6 LSF frames of lag absorbs any combination of codec-delay
    // trimming and info-frame handling between the two decoders.
    let max_lag = 6 * 576isize;
    let (lag_l, nrmse_l, ncc_l) = align_and_error(&ref_pcm[0], &our_l, max_lag, 16 * 576);
    let (lag_r, nrmse_r, ncc_r) = align_and_error(&ref_pcm[1], &our_r, max_lag, 16 * 576);
    eprintln!(
        "LSF fixture vs reference: ref={} ours={} samples; lag L={lag_l} R={lag_r}, \
         peak ncc L={ncc_l:.4} R={ncc_r:.4}, \
         steady-state normalized RMS error L={nrmse_l:.6} R={nrmse_r:.6}",
        ref_pcm[0].len(),
        our_l.len(),
    );
    // Both channels must align at the same lag (the canonical 1105 =
    // 576 encoder-delay + 529 synthesis-delay offset) and track the
    // reference essentially sample-exactly over the steady-state
    // region (measured 0.0000 normalized RMS error on every interior
    // frame with the Table B.2 tables; the pre-Table-B.2 placeholder
    // layout mis-scaled every line above the first band-boundary
    // divergence and could not reach this bound).
    assert_eq!(lag_l, lag_r, "channels aligned at different lags");
    assert!(
        ncc_l > 0.999 && ncc_r > 0.999,
        "alignment failed to lock: ncc L={ncc_l:.4} R={ncc_r:.4}"
    );
    assert!(
        nrmse_l < 0.01 && nrmse_r < 0.01,
        "LSF decode diverges from reference PCM: L={nrmse_l:.6} R={nrmse_r:.6}"
    );
}
