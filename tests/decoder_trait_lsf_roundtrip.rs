//! Integration test for the r183 MPEG-2 LSF widening of the
//! `oxideav_core::Decoder` trait wrapper (`Mp3CoreDecoder`).
//!
//! The wrapper now accepts MPEG-1 Layer III frames (mono and stereo)
//! and MPEG-2 LSF Layer III frames (mono and stereo). This test drives
//! the LSF widening end-to-end through a staged on-disk fixture:
//!
//! * `docs/audio/mp3/fixtures/layer3-mpeg2-22050-64kbps/input.mp3`
//!   is a 64 kbps / 22.05 kHz Layer III stereo stream encoded by the
//!   docs collaborator's reference toolchain (recorded in
//!   `trace.txt`, with the first audio frame header carrying
//!   `version=MPEG-2 / sample_rate=22050 / channel_mode=stereo`).
//!
//! For this stream the test asserts:
//!
//! 1. The first walked frame's header parses as `MpegVersion::Mpeg2`
//!    (LSF, ISO/IEC 13818-3 §2.4.2.3) with two channels and 576
//!    samples per channel per frame (`granule_count == 1`,
//!    `PCM_PER_GRANULE == 576`).
//! 2. Walking every frame through `Mp3CoreDecoder` (the trait
//!    wrapper) produces the same per-channel `i16` PCM byte stream
//!    as the existing direct-chain decode (parse_header /
//!    parse_side_info / decode_scalefactors / decode_huffman /
//!    requantize / process_stereo / alias_reduce / imdct_granule /
//!    synth_granule) reading the exact same wire bytes. This is the
//!    same "byte-exact vs direct chain" wedge r177 used to prove
//!    the MPEG-1 stereo widening, applied to LSF.
//! 3. Each emitted `AudioFrame` is planar (two planes of equal
//!    length) and carries exactly `PCM_PER_GRANULE = 576` samples
//!    per channel (one LSF granule = 576 PCM samples vs MPEG-1's
//!    two granules × 576 = 1152).
//!
//! The fixture is part of the workspace's `docs/audio/mp3/fixtures/`
//! corpus, which standalone-crate CI checkouts do not include; in
//! that case the test logs a skip via `eprintln!` and returns
//! success, matching the pattern already established in
//! `tests/docs_corpus.rs`.

use std::fs;
use std::path::PathBuf;

use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, Packet, RuntimeContext, SampleFormat, TimeBase,
};
use oxideav_mp3::{
    alias_reduce, decode_huffman, decode_scalefactors, imdct_granule, parse_header,
    parse_side_info, pcm_f32_to_i16, process_stereo, reorder, requantize, synth_granule,
    ChannelMode, FrameWalker, ImdctState, MainDataReader, MpegVersion, Reservoir, SynthState,
    PCM_PER_GRANULE,
};

/// Locate the LSF fixture relative to the crate root. Returns `None`
/// when the workspace `docs/` tree is absent (standalone-crate CI).
fn lsf_fixture_path() -> Option<PathBuf> {
    let p = PathBuf::from("../../docs/audio/mp3/fixtures/layer3-mpeg2-22050-64kbps/input.mp3");
    if p.exists() {
        Some(p)
    } else {
        None
    }
}

/// Reference per-channel direct decode. Mirrors `decode_stereo_direct`
/// from `decoder_trait_stereo_roundtrip.rs` but version-aware: the
/// per-granule loop honours `si.granule_count` (1 on LSF, 2 on MPEG-1)
/// and `nch` is taken from the parsed side-info.
fn decode_stereo_direct(bytes: &[u8]) -> (Vec<i16>, Vec<i16>) {
    let mut reservoir = Reservoir::new();
    let mut synth: [SynthState; 2] = [SynthState::new(), SynthState::new()];
    let mut imdct: [ImdctState; 2] = [ImdctState::new(), ImdctState::new()];
    let mut out_l: Vec<i16> = Vec::new();
    let mut out_r: Vec<i16> = Vec::new();
    for frame in FrameWalker::new(bytes) {
        let hdr = parse_header(&frame.data[..4]).expect("header");
        let si_start = 4 + if hdr.crc_protected { 2 } else { 0 };
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
                let xr = requantize(&is, gc, sf, hdr.sample_rate_hz, hdr.version);
                // §2.4.3.4.8 reorder must run before §2.4.3.4.9 stereo and
                // the §2.4.3.4.10 IMDCT for short / mixed granules; long
                // granules pass through unchanged.
                *xr_slot = reorder(&xr, gc, hdr.sample_rate_hz, hdr.version);
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
                for &p in pcm_f32.iter().take(PCM_PER_GRANULE) {
                    sink.push(pcm_f32_to_i16(p));
                }
            }
        }
    }
    (out_l, out_r)
}

/// Slice the contiguous MP3 byte stream into per-frame packets so they
/// can be fed to the trait `Decoder` one at a time. Each packet's PTS
/// advances by the on-the-wire per-frame sample count (576 on LSF, 1152
/// on MPEG-1) so the test logs a meaningful timeline.
fn mp3_to_packets(bytes: &[u8]) -> Vec<Packet> {
    let mut out = Vec::new();
    let mut pts: i64 = 0;
    for f in FrameWalker::new(bytes) {
        let hdr = parse_header(&f.data[..4]).expect("frame walker yields parseable headers");
        let sr = i64::from(hdr.sample_rate_hz);
        let tb = TimeBase::new(1, sr);
        let mut pkt = Packet::new(0, tb, f.data.to_vec());
        pkt.pts = Some(pts);
        let samples_per_frame: i64 = match hdr.version {
            MpegVersion::Mpeg1 => 1152,
            MpegVersion::Mpeg2 | MpegVersion::Mpeg25 => 576,
        };
        pkt.duration = Some(samples_per_frame);
        out.push(pkt);
        pts += samples_per_frame;
    }
    out
}

fn build_lsf_stereo_params() -> CodecParameters {
    let mut p = CodecParameters::audio(CodecId::new("mp3"));
    p.sample_rate = Some(22_050);
    p.channels = Some(2);
    p.sample_format = Some(SampleFormat::S16);
    p
}

/// Drive the supplied LSF stereo MP3 byte stream through the registered
/// trait decoder. Returns `(left_pcm, right_pcm)` recovered from the
/// planar `AudioFrame`s.
fn trait_decode_lsf_stereo(bytes: &[u8]) -> (Vec<i16>, Vec<i16>) {
    let mut dec = oxideav_mp3::make_decoder(&build_lsf_stereo_params()).expect("make_decoder");
    let mut out_l: Vec<i16> = Vec::new();
    let mut out_r: Vec<i16> = Vec::new();
    for pkt in mp3_to_packets(bytes) {
        dec.send_packet(&pkt).expect("send_packet");
        loop {
            match dec.receive_frame() {
                Ok(Frame::Audio(a)) => {
                    assert_eq!(
                        a.data.len(),
                        2,
                        "stereo LSF trait decoder must emit two planes per frame"
                    );
                    // LSF Layer III: granule_count = 1 → 576 PCM
                    // samples per channel per frame (vs MPEG-1's 1152).
                    assert_eq!(
                        a.samples as usize, PCM_PER_GRANULE,
                        "MPEG-2 LSF Layer III frame must carry 576 per-channel samples"
                    );
                    assert_eq!(
                        a.data[0].len(),
                        a.data[1].len(),
                        "planes must be equal length"
                    );
                    assert_eq!(
                        a.data[0].len(),
                        2 * (a.samples as usize),
                        "plane bytes = 2 × sample count (S16 LE)"
                    );
                    for chunk in a.data[0].chunks_exact(2) {
                        out_l.push(i16::from_le_bytes([chunk[0], chunk[1]]));
                    }
                    for chunk in a.data[1].chunks_exact(2) {
                        out_r.push(i16::from_le_bytes([chunk[0], chunk[1]]));
                    }
                }
                Ok(other) => panic!("non-audio frame: {other:?}"),
                Err(Error::NeedMore) => break,
                Err(e) => panic!("receive_frame: {e}"),
            }
        }
    }
    dec.flush().expect("flush");
    loop {
        match dec.receive_frame() {
            Ok(Frame::Audio(a)) => {
                for chunk in a.data[0].chunks_exact(2) {
                    out_l.push(i16::from_le_bytes([chunk[0], chunk[1]]));
                }
                for chunk in a.data[1].chunks_exact(2) {
                    out_r.push(i16::from_le_bytes([chunk[0], chunk[1]]));
                }
            }
            Ok(other) => panic!("non-audio frame on flush: {other:?}"),
            Err(Error::Eof) => break,
            Err(Error::NeedMore) => break,
            Err(e) => panic!("post-flush receive_frame: {e}"),
        }
    }
    (out_l, out_r)
}

#[test]
fn trait_decode_lsf_stereo_fixture_matches_direct_chain_byte_exact() {
    let Some(path) = lsf_fixture_path() else {
        eprintln!("skip: layer3-mpeg2-22050-64kbps fixture absent (standalone-crate CI checkout)");
        return;
    };
    let bytes = fs::read(&path).expect("read LSF fixture");
    // Strip the ID3v2 frontmatter if present so the FrameWalker
    // starts at the first audio frame. The fixture carries an
    // ID3v2.4 head per trace.txt (`ID3V2_HEADER ... total=44`).
    let mut wire = bytes.as_slice();
    if wire.len() >= 10 && &wire[..3] == b"ID3" {
        // ID3v2 size: 4 bytes, 7 bits each (synchsafe).
        let size = ((u32::from(wire[6]) & 0x7F) << 21)
            | ((u32::from(wire[7]) & 0x7F) << 14)
            | ((u32::from(wire[8]) & 0x7F) << 7)
            | (u32::from(wire[9]) & 0x7F);
        let total = 10 + size as usize;
        wire = &wire[total..];
    }

    // First walked frame: confirm we're really exercising the LSF
    // path. (The test would silently pass on a hypothetical MPEG-1
    // fixture; pin the version + channel count here.)
    let first = FrameWalker::new(wire).next().expect("at least one frame");
    let first_hdr = parse_header(&first.data[..4]).expect("first header parses");
    assert_eq!(
        first_hdr.version,
        MpegVersion::Mpeg2,
        "fixture's first audio frame must be MPEG-2 LSF \
         (trace.txt says version=MPEG-2, sample_rate=22050)"
    );
    assert_eq!(first_hdr.sample_rate_hz, 22_050);
    assert_eq!(first_hdr.channel_count(), 2);

    let (l_direct, r_direct) = decode_stereo_direct(wire);
    let (l_trait, r_trait) = trait_decode_lsf_stereo(wire);

    assert_eq!(
        l_trait.len(),
        l_direct.len(),
        "L sample count mismatch: trait={} direct={}",
        l_trait.len(),
        l_direct.len()
    );
    assert_eq!(
        r_trait.len(),
        r_direct.len(),
        "R sample count mismatch: trait={} direct={}",
        r_trait.len(),
        r_direct.len()
    );
    let mism_l = l_trait
        .iter()
        .zip(l_direct.iter())
        .filter(|(a, b)| a != b)
        .count();
    let mism_r = r_trait
        .iter()
        .zip(r_direct.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        mism_l, 0,
        "trait-driven LSF stereo L diverges from direct chain in {mism_l} samples"
    );
    assert_eq!(
        mism_r, 0,
        "trait-driven LSF stereo R diverges from direct chain in {mism_r} samples"
    );
    // Sanity: per the trace.txt the fixture has 33 frames (1 Xing
    // info-frame + 32 audio frames after our ID3v2 strip, walker
    // walks all 33 since it does not synthesise the info-frame
    // skip). Each audio frame contributes 576 PCM samples per
    // channel under LSF, so at minimum we expect on the order of
    // thousands of samples, not zero.
    assert!(
        l_trait.len() > 1024,
        "LSF decode produced suspiciously few samples ({})",
        l_trait.len()
    );
}

#[test]
fn registry_built_decoder_handles_lsf_stereo_packets() {
    let Some(path) = lsf_fixture_path() else {
        eprintln!("skip: layer3-mpeg2-22050-64kbps fixture absent (standalone-crate CI checkout)");
        return;
    };
    let bytes = fs::read(&path).expect("read LSF fixture");
    let mut wire = bytes.as_slice();
    if wire.len() >= 10 && &wire[..3] == b"ID3" {
        let size = ((u32::from(wire[6]) & 0x7F) << 21)
            | ((u32::from(wire[7]) & 0x7F) << 14)
            | ((u32::from(wire[8]) & 0x7F) << 7)
            | (u32::from(wire[9]) & 0x7F);
        let total = 10 + size as usize;
        wire = &wire[total..];
    }

    let (l_direct, r_direct) = decode_stereo_direct(wire);

    let mut ctx = RuntimeContext::new();
    oxideav_mp3::register(&mut ctx);
    let mut dec = ctx
        .codecs
        .first_decoder(&build_lsf_stereo_params())
        .expect("registered decoder");
    let mut out_l: Vec<i16> = Vec::new();
    let mut out_r: Vec<i16> = Vec::new();
    for pkt in mp3_to_packets(wire) {
        dec.send_packet(&pkt).expect("send_packet");
        while let Ok(Frame::Audio(a)) = dec.receive_frame() {
            assert_eq!(a.data.len(), 2);
            assert_eq!(a.samples as usize, PCM_PER_GRANULE);
            for chunk in a.data[0].chunks_exact(2) {
                out_l.push(i16::from_le_bytes([chunk[0], chunk[1]]));
            }
            for chunk in a.data[1].chunks_exact(2) {
                out_r.push(i16::from_le_bytes([chunk[0], chunk[1]]));
            }
        }
    }
    dec.flush().expect("flush");
    while let Ok(Frame::Audio(a)) = dec.receive_frame() {
        for chunk in a.data[0].chunks_exact(2) {
            out_l.push(i16::from_le_bytes([chunk[0], chunk[1]]));
        }
        for chunk in a.data[1].chunks_exact(2) {
            out_r.push(i16::from_le_bytes([chunk[0], chunk[1]]));
        }
    }
    assert_eq!(out_l, l_direct);
    assert_eq!(out_r, r_direct);
}
