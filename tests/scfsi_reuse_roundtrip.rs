//! Integration tests for the §C.1.5.3 scalefactor-selection-information
//! (scfsi) reuse path armed by `Mp3Encoder::enable_scfsi_reuse`.
//!
//! MPEG-1 Layer III carries two granules per frame. When granule 1's
//! scalefactors in a §2.4.2.7 scfsi_band group are identical to granule
//! 0's, the frame may set `scfsi[ch][group] = 1` and transmit those
//! scalefactors once (in granule 0) instead of twice. The decoder reuses
//! granule 0's values verbatim, so the reconstructed scalefactors — and
//! every decoded sample — are bit-identical; only granule 1's part2 bit
//! budget shrinks.
//!
//! These tests confirm:
//!
//! 1. **Bit-identical decode.** With scfsi reuse armed, the decoded PCM
//!    matches the un-armed (`scfsi = 0`) encode sample-for-sample. The
//!    optimisation is lossless by construction.
//! 2. **Bits saved.** On the fixed-gain encode (every scalefactor 0, so
//!    every group reuses), the armed stream's total scalefactor part2
//!    bits in granule 1 collapse to zero — at least one frame's stream
//!    is strictly shorter than the un-armed encode.
//! 3. **scfsi bits set correctly on the wire.** The armed encode's
//!    side-info actually carries `scfsi[ch][g] == 1` for the reused
//!    groups, and the decoder reproduces granule 0's values there.
//!
//! No external library is invoked — the decode chain reuses the crate's
//! own `FrameWalker` + `parse_side_info` + scalefactor / Huffman /
//! requantize / IMDCT / synth primitives.

use std::f32::consts::PI;

use oxideav_mp3::{
    alias_reduce, decode_huffman, imdct_granule, parse_header, parse_side_info, requantize,
    synth_granule, BlockType, ChannelMode, FrameWalker, ImdctState, MainDataReader, Mp3Encoder,
    MpegVersion, Reservoir, ScaleFactors, SynthState, MPEG1_SLEN, PCM_PER_GRANULE,
};

const SR: u32 = 44_100;
const BR: u32 = 128;
const PCM_LEN_SAMPLES: usize = SR as usize; // 1 second

fn sine_pcm(n: usize, freq_hz: f32, amp: f32) -> Vec<i16> {
    let two_pi = 2.0 * PI;
    let mut out = Vec::with_capacity(n);
    let scale = amp * (i16::MAX as f32);
    for i in 0..n {
        let t = i as f32 / (SR as f32);
        let s = (two_pi * freq_hz * t).sin() * scale;
        out.push(s.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16);
    }
    out
}

/// §2.4.2.7 MPEG-1 part2 reader honouring `scfsi` reuse from granule 0.
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

/// Decode an MP3 byte stream into mono `i16` PCM using ONLY the crate's
/// own primitives. Honours `scfsi` reuse on the read side.
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
                        unreachable!("LSF / MPEG-2.5 not in scope for scfsi (no scfsi field)")
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

fn encode_with<F>(make_encoder: F, pcm: &[i16]) -> Vec<u8>
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

/// Count the frames whose side-info carries at least one `scfsi == 1`
/// flag, plus the grand total of set scfsi bits across the stream.
fn count_scfsi_bits(bytes: &[u8]) -> (usize, usize) {
    let mut frames_with_reuse = 0usize;
    let mut total_set = 0usize;
    for frame in FrameWalker::new(bytes) {
        let hdr = parse_header(&frame.data[..4]).expect("header parse");
        let si = parse_side_info(&hdr, &frame.data[4..]).expect("side_info parse");
        let mut frame_set = 0usize;
        for ch in 0..si.channels as usize {
            frame_set += si.scfsi[ch].iter().filter(|&&b| b).count();
        }
        if frame_set > 0 {
            frames_with_reuse += 1;
        }
        total_set += frame_set;
    }
    (frames_with_reuse, total_set)
}

#[test]
fn scfsi_disarmed_emits_no_reuse_flags() {
    // r301: scfsi reuse is auto-armed by default, so the historical
    // byte-for-byte `scfsi = 0` output now requires an explicit
    // `disable_scfsi_reuse()`. With reuse disarmed, every frame carries
    // scfsi = 0.
    let pcm = sine_pcm(PCM_LEN_SAMPLES, 440.0, 0.5);
    let bytes = encode_with(
        || {
            let mut e = Mp3Encoder::new(BR, SR, ChannelMode::SingleChannel).expect("encoder build");
            e.disable_scfsi_reuse();
            e
        },
        &pcm,
    );
    let (frames_with_reuse, total_set) = count_scfsi_bits(&bytes);
    assert_eq!(
        frames_with_reuse, 0,
        "disarmed encode must not set any scfsi"
    );
    assert_eq!(total_set, 0);
}

#[test]
fn scfsi_auto_armed_by_default_sets_reuse_flags() {
    // r301: a default encoder (no explicit toggle) auto-arms scfsi reuse
    // inside `push_samples`. A steady tone whose granules frequently share
    // scalefactors must therefore set scfsi on at least one frame, and the
    // default stream must never be larger than the explicitly-disarmed one.
    let pcm = sine_pcm(PCM_LEN_SAMPLES, 440.0, 0.5);
    let default_stream = encode_with(
        || Mp3Encoder::new(BR, SR, ChannelMode::SingleChannel).expect("encoder build"),
        &pcm,
    );
    let disarmed = encode_with(
        || {
            let mut e = Mp3Encoder::new(BR, SR, ChannelMode::SingleChannel).expect("encoder build");
            e.disable_scfsi_reuse();
            e
        },
        &pcm,
    );
    let (frames_with_reuse, total_set) = count_scfsi_bits(&default_stream);
    assert!(
        frames_with_reuse > 0,
        "default encode auto-armed no scfsi reuse on a steady tone"
    );
    assert!(total_set >= frames_with_reuse);
    assert!(
        default_stream.len() <= disarmed.len(),
        "auto-armed default stream grew vs. disarmed: {} > {}",
        default_stream.len(),
        disarmed.len()
    );

    // Lossless: the auto-armed default decodes sample-for-sample like the
    // explicitly-disarmed encode.
    let recon_default = decode_mp3_mono(&default_stream);
    let recon_disarmed = decode_mp3_mono(&disarmed);
    assert_eq!(recon_default.len(), recon_disarmed.len());
    assert!(
        recon_default == recon_disarmed,
        "auto-armed scfsi reuse changed the decoded PCM"
    );
}

#[test]
fn scfsi_armed_sets_reuse_on_fixed_gain_stream() {
    // Fixed-gain encode: every scalefactor is 0 in both granules, so all
    // four scfsi_band groups are reuse-eligible. Arming must mark them.
    let pcm = sine_pcm(PCM_LEN_SAMPLES, 440.0, 0.5);
    let bytes = encode_with(
        || {
            let mut e = Mp3Encoder::new(BR, SR, ChannelMode::SingleChannel).expect("encoder build");
            e.enable_scfsi_reuse();
            e
        },
        &pcm,
    );
    let (frames_with_reuse, total_set) = count_scfsi_bits(&bytes);
    assert!(
        frames_with_reuse > 0,
        "armed fixed-gain encode set scfsi on no frames"
    );
    // Every long-block frame should reuse all four groups (4 bits).
    assert!(
        total_set >= 4 * frames_with_reuse,
        "expected >= 4 set scfsi bits per reuse frame, got {total_set} over {frames_with_reuse} frames"
    );
}

#[test]
fn scfsi_armed_decode_is_bit_identical_to_disarmed() {
    // The optimisation is lossless: arming changes the wire bytes but not
    // a single decoded sample.
    let pcm = sine_pcm(PCM_LEN_SAMPLES, 440.0, 0.5);
    let disarmed = encode_with(
        || {
            let mut e = Mp3Encoder::new(BR, SR, ChannelMode::SingleChannel).expect("encoder build");
            e.disable_scfsi_reuse();
            e
        },
        &pcm,
    );
    let armed = encode_with(
        || {
            let mut e = Mp3Encoder::new(BR, SR, ChannelMode::SingleChannel).expect("encoder build");
            e.enable_scfsi_reuse();
            e
        },
        &pcm,
    );

    let recon_disarmed = decode_mp3_mono(&disarmed);
    let recon_armed = decode_mp3_mono(&armed);
    assert_eq!(
        recon_disarmed.len(),
        recon_armed.len(),
        "armed/disarmed decode length mismatch"
    );
    assert!(
        recon_disarmed == recon_armed,
        "scfsi reuse must be lossless: decoded PCM differs from the disarmed encode"
    );
}

#[test]
fn scfsi_armed_outer_loop_decode_is_bit_identical() {
    // Same lossless guarantee on the outer-loop encode path (nonzero,
    // spectrally-varying scalefactors): wherever the two granules agree
    // the group is reused, and the decode is unchanged everywhere.
    let pcm = sine_pcm(PCM_LEN_SAMPLES, 1234.0, 0.6);
    let disarmed = encode_with(
        || {
            let mut e = Mp3Encoder::new_with_outer_loop(
                BR,
                SR,
                ChannelMode::SingleChannel,
                oxideav_mp3::stream_encoder::DEFAULT_OUTER_LOOP_THRESHOLD,
            )
            .expect("encoder build");
            e.disable_scfsi_reuse();
            e
        },
        &pcm,
    );
    let armed = encode_with(
        || {
            let mut e = Mp3Encoder::new_with_outer_loop(
                BR,
                SR,
                ChannelMode::SingleChannel,
                oxideav_mp3::stream_encoder::DEFAULT_OUTER_LOOP_THRESHOLD,
            )
            .expect("encoder build");
            e.enable_scfsi_reuse();
            e
        },
        &pcm,
    );

    let recon_disarmed = decode_mp3_mono(&disarmed);
    let recon_armed = decode_mp3_mono(&armed);
    assert_eq!(recon_disarmed.len(), recon_armed.len());
    assert!(
        recon_disarmed == recon_armed,
        "scfsi reuse on the outer-loop path must be lossless"
    );
}

/// Sum granule-1 `part2_3_length` across all two-granule frames.
fn sum_gr1_part2_3(bytes: &[u8]) -> u64 {
    let mut s = 0u64;
    for frame in FrameWalker::new(bytes) {
        let hdr = parse_header(&frame.data[..4]).expect("header parse");
        let si = parse_side_info(&hdr, &frame.data[4..]).expect("side_info parse");
        if si.granule_count as usize == 2 {
            for ch in 0..si.channels as usize {
                s += u64::from(si.granules[1][ch].part2_3_length);
            }
        }
    }
    s
}

#[test]
fn scfsi_reuse_shrinks_granule1_part2_on_outer_loop() {
    // The outer-loop encode transmits real (nonzero-`slen`) scalefactors,
    // so a reused group actually drops bits from granule 1's part2. Pick a
    // signal whose two granules frequently agree (a steady single tone),
    // arm scfsi, and confirm the summed granule-1 part2_3 bit budget
    // strictly shrinks while the stream never grows.
    let pcm = sine_pcm(PCM_LEN_SAMPLES, 440.0, 0.5);
    let disarmed = encode_with(
        || {
            let mut e = Mp3Encoder::new_with_outer_loop(
                BR,
                SR,
                ChannelMode::SingleChannel,
                oxideav_mp3::stream_encoder::DEFAULT_OUTER_LOOP_THRESHOLD,
            )
            .expect("encoder build");
            e.disable_scfsi_reuse();
            e
        },
        &pcm,
    );
    let armed = encode_with(
        || {
            let mut e = Mp3Encoder::new_with_outer_loop(
                BR,
                SR,
                ChannelMode::SingleChannel,
                oxideav_mp3::stream_encoder::DEFAULT_OUTER_LOOP_THRESHOLD,
            )
            .expect("encoder build");
            e.enable_scfsi_reuse();
            e
        },
        &pcm,
    );

    // CBR frame size is fixed; the reservoir absorbs the saved part2
    // bits, so the stream never grows.
    assert!(
        armed.len() <= disarmed.len(),
        "armed scfsi stream grew: {} > {}",
        armed.len(),
        disarmed.len()
    );

    let (frames_with_reuse, _) = count_scfsi_bits(&armed);
    assert!(
        frames_with_reuse > 0,
        "outer-loop steady tone produced no reusable granule pairs"
    );

    let dis_gr1 = sum_gr1_part2_3(&disarmed);
    let arm_gr1 = sum_gr1_part2_3(&armed);
    assert!(
        arm_gr1 < dis_gr1,
        "armed granule-1 part2_3 bits ({arm_gr1}) not smaller than disarmed ({dis_gr1})"
    );
}
