//! **r408 — foreign MPEG-2.5 8 kHz mixed-block decode.**
//!
//! When these tests were written the encoder still refused to *emit*
//! mixed blocks at 8 kHz (deployed decoders split 3-1 on the window
//! geometry of the 8 kHz mixed carve-out; r440 later grounded the
//! 36-line window split in the §2.4.2.7 subband-count text and
//! lifted the refusal), but the decoder must render a *foreign*
//! 8 kHz mixed stream sensibly either way. Through r407 it did not:
//! the mixed
//! decode used a fixed 36-line long-coded region and started the
//! short walk at short scalefactor band 3 (per-window line 24 = wire
//! line 72 at the 8 kHz Fraunhofer tables), silently zeroing wire
//! lines 36..72 of every granule.
//!
//! The r408 observer probes (single-nonzero-line streams built with
//! this crate's own low-level frame assembly, decoded by four
//! independent black-box validator binaries) resolved the de-facto
//! layout:
//!
//! * **coding split = 72 lines** at 8 kHz: all four validators
//!   requantize wire lines 36..72 with the transmitted **long**
//!   scalefactor bands 3..5 (the 8 kHz long table's six lowest bands
//!   span exactly 0..72 = `3 · short_starts[3]`) and the long gain
//!   formula — `subblock_gain` does not apply there;
//! * **window split stays 36 lines** (§2.4.2.7's two lowest polyphase
//!   subbands) on three of the four validators: the long-coded lines
//!   36..72 pass through the reorder unchanged and are consumed by
//!   the short IMDCT of subbands 2..3 in its native `[3·k + win]`
//!   interleave — exactly what this crate's pipeline produces once
//!   the requantizer covers the range (the fourth validator
//!   long-windows the whole 72-line region; minority reading);
//! * the mixed Huffman **region-0 boundary is a deployed grey zone**
//!   (36 refuted by all four; then 72 / 96 / >100 three ways at
//!   8 kHz and a 2-2 split between 36 and 48 at 22.05 kHz), so the
//!   decoder uses the band-relative `3 · short_starts[3]` (the
//!   primary validator's reading) and the encoder emits mixed
//!   granules with `table_select[0] == table_select[1]`, which makes
//!   every boundary interpretation consume identical bits.
//!
//! These tests pin the fixed behaviour using only this crate (the
//! validator comparisons ran offline; their numeric findings are in
//! the assertions below).

use std::io::Cursor;

use oxideav_core::{
    CodecId, CodecParameters, Decoder, Demuxer, Error, Frame, RuntimeContext, SampleFormat,
};
use oxideav_mp3::{
    assemble_main_data, make_silent_header, silent_side_info, write_header, write_side_info,
    BlockType, ChannelMode, FrameScaleFactors, GranuleChannelData, Mp3Demuxer, ScaleFactors,
};

/// Decode a stream through the production demuxer + trait decoder.
fn own_decode(bytes: &[u8], sample_rate: u32) -> Vec<f32> {
    let mut demux = Mp3Demuxer::open(Box::new(Cursor::new(bytes.to_vec()))).expect("demux open");
    let mut ctx = RuntimeContext::default();
    oxideav_mp3::register(&mut ctx);
    let mut params = CodecParameters::audio(CodecId::new("mp3"));
    params.sample_rate = Some(sample_rate);
    params.channels = Some(1);
    params.sample_format = Some(SampleFormat::S16);
    let mut dec: Box<dyn Decoder> = ctx.codecs.first_decoder(&params).expect("decoder");
    let mut out: Vec<f32> = Vec::new();
    loop {
        let pkt = match demux.next_packet() {
            Ok(p) => p,
            Err(Error::Eof) => break,
            Err(e) => panic!("next_packet: {e}"),
        };
        dec.send_packet(&pkt).expect("send_packet");
        loop {
            match dec.receive_frame() {
                Ok(Frame::Audio(a)) => {
                    for chunk in a.data[0].chunks_exact(2) {
                        let v = i16::from_le_bytes([chunk[0], chunk[1]]);
                        out.push(f32::from(v) / 32768.0);
                    }
                }
                Ok(other) => panic!("non-audio frame: {other:?}"),
                Err(Error::NeedMore) => break,
                Err(e) => panic!("receive_frame: {e}"),
            }
        }
    }
    dec.flush().expect("flush");
    out
}

/// Build a foreign-style 8 kHz mono mixed-block stream: every frame
/// carries `is[line] = value` and nothing else, with the given long
/// scalefactors (LSF mixed layout: `scalefac_l` 0..6) and
/// `subblock_gain`, repeated `n_frames` times.
#[allow(clippy::too_many_arguments)]
fn probe_stream(
    line: usize,
    value: i32,
    global_gain: u8,
    scalefac_compress: u16,
    scalefac_scale: bool,
    long_sf: [u8; 6],
    subblock_gain: [u8; 3],
    n_frames: usize,
) -> Vec<u8> {
    let header = make_silent_header(64, 8_000, ChannelMode::SingleChannel).expect("header");
    let mut si = silent_side_info(&header);
    {
        let gc = &mut si.granules[0][0];
        gc.window_switching_flag = true;
        gc.block_type = BlockType::Short;
        gc.mixed_block_flag = true;
        gc.global_gain = global_gain;
        gc.scalefac_compress = scalefac_compress;
        gc.scalefac_scale = scalefac_scale;
        // Same codebook for both big-values regions so the decode is
        // independent of any region-boundary interpretation.
        gc.table_select = [13, 13, 0];
        gc.subblock_gain = subblock_gain;
        gc.big_values = (line / 2 + 1) as u16;
    }
    let mut sf0 = ScaleFactors::default();
    sf0.long[..6].copy_from_slice(&long_sf);
    let sf = FrameScaleFactors {
        granules: [
            [sf0, ScaleFactors::default()],
            [ScaleFactors::default(), ScaleFactors::default()],
        ],
        granule_count: 1,
        channels: 1,
        part2_bits: [[0; 2]; 2],
    };
    let empty = GranuleChannelData {
        is: [0; 576],
        big_pairs: 0,
        count1_quads: 0,
    };
    let mut gcd = [[empty.clone(), empty.clone()], [empty.clone(), empty]];
    gcd[0][0].is[line] = value;
    gcd[0][0].big_pairs = line / 2 + 1;

    let asm = assemble_main_data(&header, &mut si, &sf, &gcd).expect("assemble");
    let frame_len = header.frame_len().expect("frame len");
    let mut frame = Vec::with_capacity(frame_len);
    frame.extend_from_slice(&write_header(&header));
    frame.extend_from_slice(&write_side_info(&si));
    frame.extend_from_slice(&asm.bytes);
    assert!(frame.len() <= frame_len, "probe frame overflow");
    frame.resize(frame_len, 0);
    frame.repeat(n_frames)
}

fn rms(x: &[f32]) -> f64 {
    (x.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / x.len().max(1) as f64).sqrt()
}

/// The r407 defect: wire lines 36..72 of a foreign 8 kHz mixed
/// granule rendered silent. They must now carry energy — the
/// short-window level measured on the majority validators (≈ 0.0706
/// for a unit line at `global_gain = 200`), distinctly below the
/// long-window level of the true long region (≈ 0.1226).
#[test]
fn foreign_8k_mixed_lines_36_to_72_render_nonsilent() {
    // Controls: long region (24), short region (80).
    let long_rms = rms(&own_decode(
        &probe_stream(24, 1, 200, 0, false, [0; 6], [0; 3], 24),
        8_000,
    ));
    assert!(
        (0.09..0.16).contains(&long_rms),
        "long-region control rms {long_rms}"
    );
    for line in [36usize, 40, 44, 48, 56, 64] {
        let bytes = probe_stream(line, 1, 200, 0, false, [0; 6], [0; 3], 24);
        let pcm = own_decode(&bytes, 8_000);
        let r = rms(&pcm);
        assert!(
            (0.05..0.09).contains(&r),
            "wire line {line}: rms {r} (was silent before r408; expected the \
             short-window single-line level ≈ 0.07)"
        );
    }
    // Short-region control renders at the same short-window level.
    let short_rms = rms(&own_decode(
        &probe_stream(80, 1, 200, 0, false, [0; 6], [0; 3], 24),
        8_000,
    ));
    assert!(
        (0.05..0.09).contains(&short_rms),
        "short-region control rms {short_rms}"
    );
}

/// Lines 36..72 are long-coded: the transmitted LSF long scalefactor
/// bands 3 (36..48) and 5 (60..72) attenuate them by exactly
/// `2^-scalefac` (scalefac_scale = 1), and a band-3 scalefactor does
/// not touch a band-5 line. Measured identically on all four
/// black-box validators (r408).
#[test]
fn foreign_8k_mixed_lines_36_to_72_use_long_scalefactor_bands() {
    // scalefac_compress = 160: slen1 = 2 (six long scalefactors of
    // two bits each), slen2..4 = 0.
    let base_40 = rms(&own_decode(
        &probe_stream(40, 1, 200, 160, true, [0; 6], [0; 3], 24),
        8_000,
    ));
    let att_40 = rms(&own_decode(
        &probe_stream(40, 1, 200, 160, true, [0, 0, 0, 3, 0, 0], [0; 3], 24),
        8_000,
    ));
    let ratio = att_40 / base_40;
    assert!(
        (ratio - 0.125).abs() < 0.01,
        "long band 3 must attenuate wire line 40 by 2^-3 (got ratio {ratio})"
    );
    // Cross-band control: band 3 must not touch line 64 (band 5).
    let base_64 = rms(&own_decode(
        &probe_stream(64, 1, 200, 160, true, [0; 6], [0; 3], 24),
        8_000,
    ));
    let cross_64 = rms(&own_decode(
        &probe_stream(64, 1, 200, 160, true, [0, 0, 0, 3, 0, 0], [0; 3], 24),
        8_000,
    ));
    assert!(
        (cross_64 / base_64 - 1.0).abs() < 0.01,
        "long band 3 must not touch wire line 64"
    );
    let att_64 = rms(&own_decode(
        &probe_stream(64, 1, 200, 160, true, [0, 0, 0, 0, 0, 3], [0; 3], 24),
        8_000,
    ));
    assert!(
        (att_64 / base_64 - 0.125).abs() < 0.01,
        "long band 5 must attenuate wire line 64 by 2^-3"
    );
}

/// `subblock_gain` applies only to the short-coded region (wire
/// lines 72..576 at 8 kHz): it must not touch the long-coded lines
/// 36..72, and it must attenuate its own window of short band 3.
/// Measured identically on all four black-box validators (r408).
#[test]
fn foreign_8k_mixed_subblock_gain_scope() {
    for line in [40usize, 58] {
        let base = rms(&own_decode(
            &probe_stream(line, 1, 200, 0, false, [0; 6], [0; 3], 24),
            8_000,
        ));
        let sbg = rms(&own_decode(
            &probe_stream(line, 1, 200, 0, false, [0; 6], [0, 7, 0], 24),
            8_000,
        ));
        assert!(
            (sbg / base - 1.0).abs() < 0.01,
            "subblock_gain must not touch long-coded wire line {line}"
        );
    }
    // Short band 3 spans per-window 24..36 (wire 72..108; win1 =
    // 84..96): subblock_gain[1] = 2 attenuates it by 2^-4.
    let base = rms(&own_decode(
        &probe_stream(84, 1, 210, 0, false, [0; 6], [0; 3], 24),
        8_000,
    ));
    let sbg = rms(&own_decode(
        &probe_stream(84, 1, 210, 0, false, [0; 6], [0, 2, 0], 24),
        8_000,
    ));
    assert!(
        (sbg / base - 0.0625).abs() < 0.005,
        "subblock_gain[1] must attenuate short band 3 window 1 by 2^-4 \
         (got ratio {})",
        sbg / base
    );
}
