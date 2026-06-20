//! MPEG-2.5 (low-sample-rate Fraunhofer extension) decode accuracy
//! against the staged reference PCM.
//!
//! `docs/audio/mp3/fixtures/layer3-mpeg25-11025-32kbps/` carries both
//! the encoded stream (`input.mp3`, a 32 kbps / 11.025 kHz **mono**
//! MPEG-2.5 Layer III stream produced by the docs collaborator's
//! reference toolchain — its `trace.txt` records every frame header as
//! `version=MPEG-2.5 / sample_rate=11025 / channel_mode=mono`) and the
//! black-box reference decode (`expected.wav`, the toolchain's PCM
//! output, consumed here as opaque fixture data).
//!
//! MPEG-2.5 is the Fraunhofer-IIS low-sample-rate extension
//! (`docs/audio/mp3/MPEG-2.5-GAP.md` for attribution). It reuses the
//! ISO/IEC 13818-3 LSF framing on the half-rate sample rates; for
//! 11.025 kHz the scalefactor-band layout is *byte-identical* to the
//! in-repo 13818-3 22.05 kHz LSF Table B.2 (long + short — fully
//! grounded, `docs/audio/mp3/mpeg2.5-scalefactor-bands.md`, #147/#151).
//! With those tables wired into the requantizer, this crate's own
//! decode of the 11.025 kHz MPEG-2.5 stream must track that reference
//! sample-for-sample over the steady-state region: the two decoders
//! differ only in float rounding and codec-delay trimming, not in band
//! layout. This is the MPEG-2.5 sibling of
//! `tests/lsf_reference_pcm.rs` (which proves the same property for the
//! 22.05 kHz LSF stream).
//!
//! Skips (with a log line) when the workspace `docs/` tree is absent
//! (standalone-crate CI checkout), matching `tests/docs_corpus.rs`.

use std::fs;
use std::path::PathBuf;

use oxideav_core::{CodecId, CodecParameters, Error, Frame, Packet, SampleFormat, TimeBase};
use oxideav_mp3::{
    alias_reduce, decode_huffman, decode_scalefactors, imdct_granule, parse_header,
    parse_side_info, reorder, requantize, synth_granule, FrameWalker, ImdctState, MainDataReader,
    MpegVersion, Reservoir, SynthState, PCM_PER_GRANULE,
};

fn fixture_dir() -> Option<PathBuf> {
    let p = PathBuf::from("../../docs/audio/mp3/fixtures/layer3-mpeg25-11025-32kbps");
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

/// Direct-chain **mono** decode (the MPEG-2.5 fixture is single-channel,
/// so the stereo split / `process_stereo` path of `lsf_reference_pcm.rs`
/// is not exercised here). Returns f32 PCM in [-1, 1].
fn decode_mono_f32(bytes: &[u8]) -> Vec<f32> {
    let mut reservoir = Reservoir::new();
    let mut synth = SynthState::new();
    let mut imdct = ImdctState::new();
    let mut out: Vec<f32> = Vec::new();
    for (idx, frame) in FrameWalker::new(bytes).enumerate() {
        let hdr = parse_header(&frame.data[..4]).expect("header");
        let si_start = 4 + if hdr.crc_protected { 2 } else { 0 };
        let si = parse_side_info(&hdr, &frame.data[si_start..]).expect("side_info");
        let si_len = si.byte_len();
        let main_slot = &frame.data[si_start + si_len..];
        // The Xing/Info carrier frame produces no audio, but its
        // main-data slot bytes MUST still enter the bit reservoir: a
        // later frame's `main_data_begin` can reach back through the
        // carrier's slot. (Dropping the carrier's slot from the
        // reservoir mis-aligns every back-referencing frame — the cause
        // of the per-frame phase drift seen before this fix.)
        let is_info = idx == 0 && {
            let si_bytes = oxideav_mp3::side_info_len(hdr.version, hdr.channel_count());
            oxideav_mp3::parse_xing_info(frame.data, si_bytes).is_some()
        };
        let run = match reservoir.assemble(usize::from(si.main_data_begin), main_slot) {
            Ok(r) => r,
            Err(_) => continue,
        };
        if is_info {
            // Carrier slot is now in the reservoir; emit no PCM for it.
            continue;
        }
        let fsf = decode_scalefactors(&hdr, &si, &run).expect("scalefactors");
        let mut bit_cursor = 0usize;
        for gr in 0..si.granule_count as usize {
            let gc = &si.granules[gr][0];
            let mut r = MainDataReader::new(&run);
            // Skip part-2 (scalefactor) bits so the reader lands on the
            // part-3 Huffman codewords; cap the Huffman budget at the
            // remaining bits of `part2_3_length`.
            let mut left = bit_cursor + fsf.part2_bits[gr][0] as usize;
            while left >= 32 {
                let _ = r.read(32);
                left -= 32;
            }
            if left > 0 {
                let _ = r.read(left as u32);
            }
            let part3_bits = u32::from(gc.part2_3_length).saturating_sub(fsf.part2_bits[gr][0]);
            let is = decode_huffman(&mut r, gc, part3_bits, hdr.sample_rate_hz, hdr.version)
                .expect("huffman");
            let sf = &fsf.granules[gr][0];
            let xr = requantize(&is, gc, sf, hdr.sample_rate_hz, hdr.version);
            // §2.4.3.4.8 reorder: short/mixed granules must be rewritten
            // into subband-window order before alias reduction + IMDCT;
            // long granules pass through unchanged.
            let xr = reorder(&xr, gc, hdr.sample_rate_hz, hdr.version);
            bit_cursor += gc.part2_3_length as usize;
            let xar = alias_reduce(&xr, gc);
            let subband_time = imdct_granule(&xar, gc, &mut imdct);
            let pcm_f32 = synth_granule(&subband_time, &mut synth);
            out.extend_from_slice(&pcm_f32[..PCM_PER_GRANULE]);
        }
    }
    out
}

/// Minimal RIFF/WAVE reader for the 16-bit LE mono PCM `expected.wav`.
fn read_wav_s16_mono(path: &std::path::Path) -> (u16, u32, Vec<f32>) {
    let bytes = fs::read(path).expect("read wav");
    assert!(bytes.len() > 44 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WAVE");
    let mut pos = 12usize;
    let mut channels = 0u16;
    let mut sample_rate = 0u32;
    let mut data: Vec<f32> = Vec::new();
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
                for chunk in body.chunks_exact(2) {
                    let v = i16::from_le_bytes([chunk[0], chunk[1]]);
                    data.push(f32::from(v) / 32768.0);
                }
            }
            _ => {}
        }
        pos += 8 + len + (len & 1);
    }
    (channels, sample_rate, data)
}

/// Best-correlation lag of `b` relative to `a` plus the normalized RMS
/// error over the steady-state region. Mirrors `align_and_error` in
/// `tests/lsf_reference_pcm.rs`, adapted for a shorter (18-frame)
/// fixture: the anchor + correlation window shrink to fit.
fn align_and_error(a: &[f32], b: &[f32], max_lag: isize, win: usize) -> (isize, f64, f64) {
    let anchor = 3 * 576usize;
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
fn mpeg25_11025_fixture_decode_tracks_reference_pcm() {
    let Some(dir) = fixture_dir() else {
        eprintln!("skip: layer3-mpeg25-11025-32kbps fixture absent (standalone-crate CI checkout)");
        return;
    };
    let mp3 = fs::read(dir.join("input.mp3")).expect("read input.mp3");
    let wire = strip_id3v2(&mp3);

    // First walked frame: pin the MPEG-2.5 path. The test would
    // silently pass on a hypothetical LSF/MPEG-1 fixture, so assert the
    // version + rate + mono channel count match the trace.
    let first = FrameWalker::new(wire).next().expect("at least one frame");
    let first_hdr = parse_header(&first.data[..4]).expect("first header parses");
    assert_eq!(
        first_hdr.version,
        MpegVersion::Mpeg25,
        "fixture's first audio frame must be MPEG-2.5 \
         (trace.txt says version=MPEG-2.5, sample_rate=11025)"
    );
    assert_eq!(first_hdr.sample_rate_hz, 11_025);
    assert_eq!(first_hdr.channel_count(), 1);

    let our = decode_mono_f32(wire);
    assert!(our.len() > 4_000, "decode produced too few samples");

    let (channels, rate, ref_pcm) = read_wav_s16_mono(&dir.join("expected.wav"));
    assert_eq!(channels, 1, "MPEG-2.5 fixture reference is mono");
    assert_eq!(rate, 11_025);

    // ±6 frames of lag absorbs any combination of codec-delay trimming
    // and info-frame handling between the two decoders.
    let max_lag = 6 * 576isize;
    let win = 8 * 576;
    let (lag, nrmse, ncc) = align_and_error(&ref_pcm, &our, max_lag, win);
    eprintln!(
        "MPEG-2.5 11025 fixture vs reference: ref={} ours={} samples; \
         lag={lag}, peak ncc={ncc:.4}, steady-state normalized RMS error={nrmse:.6}",
        ref_pcm.len(),
        our.len(),
    );
    // The 11.025 kHz scalefactor-band layout is byte-identical to the
    // in-repo 13818-3 22.05 kHz LSF Table B.2, so the decode locks onto
    // the reference at the canonical 1105-sample codec delay and tracks
    // it essentially sample-exactly over the steady-state interior
    // (measured nrmse ≈ 1e-4 — the float-rounding regime). Before the
    // part-2/part-3 split fix that this test exercises, frames carrying
    // a non-zero `slen` partition (scalefac_compress ≠ 0) had
    // `decode_huffman` mis-read scalefactor bits as Huffman codewords,
    // pushing this metric to ~0.77.
    assert_eq!(lag, 1105, "decode must lock at the canonical codec delay");
    assert!(ncc > 0.999, "alignment failed to lock: ncc={ncc:.4}");
    assert!(
        nrmse < 0.005,
        "MPEG-2.5 11025 decode diverges from reference PCM: nrmse={nrmse:.6}"
    );
}

/// Slice the contiguous MP3 byte stream into per-frame packets.
fn mp3_to_packets(bytes: &[u8]) -> Vec<Packet> {
    let mut out = Vec::new();
    let mut pts: i64 = 0;
    for f in FrameWalker::new(bytes) {
        let hdr = parse_header(&f.data[..4]).expect("walker yields parseable headers");
        let tb = TimeBase::new(1, i64::from(hdr.sample_rate_hz));
        let mut pkt = Packet::new(0, tb, f.data.to_vec());
        pkt.pts = Some(pts);
        pkt.duration = Some(PCM_PER_GRANULE as i64); // one LSF granule
        out.push(pkt);
        pts += PCM_PER_GRANULE as i64;
    }
    out
}

/// Drive the MPEG-2.5 mono stream through the registered trait
/// [`oxideav_core::Decoder`] (the production `Mp3CoreDecoder` path) and
/// recover the mono `i16` PCM.
fn trait_decode_mono(bytes: &[u8]) -> Vec<i16> {
    let mut params = CodecParameters::audio(CodecId::new("mp3"));
    params.sample_rate = Some(11_025);
    params.channels = Some(1);
    params.sample_format = Some(SampleFormat::S16);
    let mut dec = oxideav_mp3::make_decoder(&params).expect("make_decoder");
    let mut out: Vec<i16> = Vec::new();
    let drain = |a: &oxideav_core::AudioFrame, out: &mut Vec<i16>| {
        for chunk in a.data[0].chunks_exact(2) {
            out.push(i16::from_le_bytes([chunk[0], chunk[1]]));
        }
    };
    for pkt in mp3_to_packets(bytes) {
        dec.send_packet(&pkt).expect("send_packet");
        loop {
            match dec.receive_frame() {
                Ok(Frame::Audio(a)) => drain(&a, &mut out),
                Ok(other) => panic!("non-audio frame: {other:?}"),
                Err(Error::NeedMore) => break,
                Err(e) => panic!("receive_frame: {e}"),
            }
        }
    }
    dec.flush().expect("flush");
    loop {
        match dec.receive_frame() {
            Ok(Frame::Audio(a)) => drain(&a, &mut out),
            Ok(other) => panic!("non-audio frame on flush: {other:?}"),
            Err(Error::Eof | Error::NeedMore) => break,
            Err(e) => panic!("post-flush receive_frame: {e}"),
        }
    }
    out
}

/// The production trait decoder must reconstruct the same PCM the direct
/// decode chain does — byte-exact — proving the part-2/part-3 split fix
/// is wired into `Mp3CoreDecoder`, not just the test harness.
#[test]
fn mpeg25_11025_trait_decoder_matches_direct_chain() {
    let Some(dir) = fixture_dir() else {
        eprintln!("skip: layer3-mpeg25-11025-32kbps fixture absent (standalone-crate CI checkout)");
        return;
    };
    let mp3 = fs::read(dir.join("input.mp3")).expect("read input.mp3");
    let wire = strip_id3v2(&mp3);

    let direct_f32 = decode_mono_f32(wire);
    let direct: Vec<i16> = direct_f32
        .iter()
        .map(|&p| oxideav_mp3::pcm_f32_to_i16(p))
        .collect();
    let traited = trait_decode_mono(wire);

    // The production trait decoder emits the Xing/Info carrier frame as
    // one (near-silent) granule of audio, whereas the direct chain here
    // skips it; trim that leading granule before comparing. After the
    // trim the two paths must agree byte-exactly — same scalefactor /
    // Huffman / requantize / IMDCT / synthesis arithmetic, including the
    // part-2/part-3 split fix wired into `Mp3CoreDecoder`.
    assert_eq!(
        traited.len(),
        direct.len() + PCM_PER_GRANULE,
        "trait decoder sample count {} != direct {} + one info granule",
        traited.len(),
        direct.len()
    );
    let traited_audio = &traited[PCM_PER_GRANULE..];
    let mism = traited_audio
        .iter()
        .zip(direct.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        mism, 0,
        "trait decoder diverges from direct chain in {mism} samples"
    );
}
