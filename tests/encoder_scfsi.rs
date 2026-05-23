//! End-to-end tests for per-band scalefactor noise shaping and scfsi
//! scalefactor reuse on the MPEG-1 Layer III encoder.
//!
//! Background (ISO/IEC 11172-3):
//! - §2.4.3.4 + Table 3-B.32: a non-IS long-block granule may carry a
//!   per-band `scalefac_l[sfb]` section (`scalefac_compress = 15` ⇒
//!   slen1 = 4 / slen2 = 3) that shapes quantization noise toward the
//!   per-band masking threshold. Each step halves that band's quantizer
//!   step on decode.
//! - §2.4.2.7 "scfsi": when both granules of a channel are long blocks,
//!   each of the four scfsi groups (sfb 0-5 / 6-10 / 11-15 / 16-20)
//!   whose granule-1 scalefactors equal granule 0's can be reused —
//!   granule 1 omits them and the decoder copies granule 0's values.
//!
//! The encoder engages shaping on the Psy-1 VBR path (the mask-driven
//! noise-allocation loop) and scfsi reuse during frame assembly. These
//! tests confirm: (a) shaped granules and scfsi reuse actually fire on
//! stationary/correlated content, (b) the output decodes cleanly through
//! our own decoder, and (c) ffmpeg cross-decodes the bitstream without
//! error and recovers the dominant spectrum. ffmpeg tests are skipped
//! silently when ffmpeg is absent.

use std::io::Write;
use std::process::Command;

use oxideav_core::options::CodecOptions;
use oxideav_core::{AudioFrame, CodecId, CodecParameters, Frame, Packet, SampleFormat, TimeBase};
use oxideav_mp3::decoder::make_decoder;
use oxideav_mp3::encoder::make_encoder;
use oxideav_mp3::frame::parse_frame_header;
use oxideav_mp3::sideinfo::SideInfo;
use oxideav_mp3::CODEC_ID_STR;

/// Stationary multi-tone stereo signal with identical L/R channels —
/// the kind of content where the two granules of a frame are nearly
/// identical, maximising scfsi reuse opportunity.
fn build_stationary_stereo(sr: u32, dur: f32) -> Vec<i16> {
    let n = (sr as f32 * dur) as usize;
    let two_pi = 2.0 * std::f32::consts::PI;
    let freqs = [220.0_f32, 440.0, 587.0, 880.0, 1318.0];
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        let t = i as f32 / sr as f32;
        let mut s = 0.0f32;
        for f in freqs {
            s += (two_pi * f * t).sin() * 0.18;
        }
        let v = (s.clamp(-1.0, 1.0) * 0.5 * 32767.0) as i16;
        out.push(v);
        out.push(v);
    }
    out
}

fn encode_vbr(pcm: &[i16], sr: u32, channels: u16, quality: u8) -> Vec<u8> {
    let mut params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
    params.channels = Some(channels);
    params.sample_rate = Some(sr);
    params.sample_format = Some(SampleFormat::S16);
    params.options = CodecOptions::new()
        .set("vbr_quality", quality.to_string())
        .set("psy_model", "1");
    let mut enc = make_encoder(&params).expect("encoder");

    let mut bytes_in: Vec<u8> = Vec::with_capacity(pcm.len() * 2);
    for &s in pcm {
        bytes_in.extend_from_slice(&s.to_le_bytes());
    }
    let chunk = 1152 * channels as usize * 2;
    let mut pts = 0i64;
    for slice in bytes_in.chunks(chunk) {
        let ns = slice.len() / (2 * channels as usize);
        enc.send_frame(&Frame::Audio(AudioFrame {
            samples: ns as u32,
            pts: Some(pts),
            data: vec![slice.to_vec()],
        }))
        .expect("send_frame");
        pts += ns as i64;
    }
    enc.flush().expect("flush");
    let mut out = Vec::new();
    while let Ok(p) = enc.receive_packet() {
        out.extend_from_slice(&p.data);
    }
    out
}

/// Walk frames and tally how many granules used the shaped-scalefactor
/// section (`scalefac_compress = 15`) and how many scfsi groups were set.
fn tally_shaping(bitstream: &[u8]) -> (usize, usize, usize) {
    let mut pos = 0usize;
    let (mut frames, mut shaped, mut scfsi_set) = (0usize, 0usize, 0usize);
    while pos + 4 <= bitstream.len() {
        let Ok(hdr) = parse_frame_header(&bitstream[pos..]) else {
            break;
        };
        let Some(flen) = hdr.frame_bytes() else { break };
        let flen = flen as usize;
        if pos + flen > bitstream.len() {
            break;
        }
        if let Ok(si) = SideInfo::parse_mpeg1(&hdr, &bitstream[pos + 4..]) {
            for ch in 0..2 {
                for grp in 0..4 {
                    if si.scfsi[ch][grp] {
                        scfsi_set += 1;
                    }
                }
            }
            for gr in 0..2 {
                for ch in 0..2 {
                    if si.granules[gr][ch].scalefac_compress == 15 {
                        shaped += 1;
                    }
                }
            }
        }
        frames += 1;
        pos += flen;
    }
    (frames, shaped, scfsi_set)
}

fn decode_to_pcm(bitstream: &[u8], sr: u32, channels: usize) -> Vec<f32> {
    let params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
    let mut dec = make_decoder(&params).expect("decoder");
    let tb = TimeBase::new(1, sr as i64);
    let mut pcm: Vec<f32> = Vec::new();
    let mut pos = 0usize;
    while pos + 4 <= bitstream.len() {
        let Ok(hdr) = parse_frame_header(&bitstream[pos..]) else {
            break;
        };
        let Some(flen) = hdr.frame_bytes() else { break };
        let flen = flen as usize;
        if pos + flen > bitstream.len() {
            break;
        }
        let pkt = Packet::new(0, tb, bitstream[pos..pos + flen].to_vec());
        if dec.send_packet(&pkt).is_ok() {
            if let Ok(Frame::Audio(a)) = dec.receive_frame() {
                for chunk in a.data[0].chunks_exact(2) {
                    pcm.push(i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0);
                }
            }
        }
        pos += flen;
    }
    let _ = channels;
    pcm
}

fn goertzel_power(pcm: &[f32], sr: u32, freq: f32) -> f32 {
    let n = pcm.len();
    let k = (n as f32 * freq / sr as f32).round();
    let omega = 2.0 * std::f32::consts::PI * k / n as f32;
    let coeff = 2.0 * omega.cos();
    let (mut s1, mut s2) = (0.0f32, 0.0f32);
    for &x in pcm {
        let s = x + coeff * s1 - s2;
        s2 = s1;
        s1 = s;
    }
    s2 * s2 + s1 * s1 - coeff * s1 * s2
}

/// Stationary correlated stereo at high quality (q=2) engages per-band
/// scalefactor shaping on many granules and scfsi reuse across the two
/// granules of a frame.
#[test]
fn shaping_and_scfsi_fire_on_stationary_stereo() {
    let sr = 44_100u32;
    let pcm = build_stationary_stereo(sr, 1.5);
    let bytes = encode_vbr(&pcm, sr, 2, 2);
    let (frames, shaped, scfsi_set) = tally_shaping(&bytes);
    eprintln!("frames={frames} shaped_granules={shaped} scfsi_groups_set={scfsi_set}");
    assert!(frames > 10, "too few frames: {frames}");
    assert!(
        shaped > 0,
        "expected the Psy-1 noise allocator to engage per-band shaping"
    );
    assert!(
        scfsi_set > 0,
        "expected scfsi reuse on stationary correlated stereo"
    );
}

/// The shaped + scfsi bitstream decodes cleanly through our own decoder
/// and reproduces the dominant 440 Hz partial.
#[test]
fn shaped_scfsi_own_decode_roundtrip() {
    let sr = 44_100u32;
    let pcm = build_stationary_stereo(sr, 1.5);
    let bytes = encode_vbr(&pcm, sr, 2, 2);
    let decoded = decode_to_pcm(&bytes, sr, 2);
    assert!(decoded.len() >= 8 * 1152 * 2, "too few samples decoded");
    // De-interleave left channel.
    let left: Vec<f32> = decoded.chunks_exact(2).map(|c| c[0]).collect();
    let warm = 6 * 1152;
    let an = &left[warm..];
    let p440 = goertzel_power(an, sr, 440.0);
    let pn = goertzel_power(an, sr, 3000.0) + goertzel_power(an, sr, 7000.0) + 1e-12;
    let ratio = p440 / pn;
    eprintln!("own-decode 440Hz/noise ratio = {ratio:.2}");
    assert!(
        ratio > 10.0,
        "weak tone after shaped+scfsi roundtrip: {ratio}"
    );
}

/// ffmpeg cross-decodes the shaped + scfsi bitstream without error and
/// recovers the dominant spectrum — confirming wire correctness of the
/// per-band scalefactor section and the scfsi reuse signalling.
#[test]
fn shaped_scfsi_ffmpeg_cross_decode() {
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        eprintln!("ffmpeg not available — skipping interop check");
        return;
    }
    let sr = 44_100u32;
    let pcm = build_stationary_stereo(sr, 1.5);
    let bytes = encode_vbr(&pcm, sr, 2, 2);
    // Confirm shaping + scfsi actually engaged so this really exercises
    // the new code path rather than silently passing on a flat stream.
    let (_, shaped, scfsi_set) = tally_shaping(&bytes);
    assert!(
        shaped > 0 && scfsi_set > 0,
        "test stream did not engage shaping/scfsi"
    );

    let tmp = std::env::temp_dir().join("oxideav_mp3_scfsi_test.mp3");
    std::fs::File::create(&tmp)
        .unwrap()
        .write_all(&bytes)
        .unwrap();
    let res = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(&tmp)
        .args(["-f", "f32le", "-ac", "2", "-ar", "44100", "-"])
        .output()
        .expect("ffmpeg run");
    let stderr = String::from_utf8_lossy(&res.stderr);
    assert!(
        res.status.success(),
        "ffmpeg failed decoding shaped+scfsi stream: {stderr}"
    );
    // ffmpeg must not emit hard errors (e.g. "invalid new backstep",
    // scalefactor overruns). Warnings about header parsing of the raw
    // stream tail are tolerated; hard decode errors are not.
    let suspicious: Vec<&str> = stderr
        .lines()
        .filter(|l| {
            let l = l.to_lowercase();
            l.contains("backstep") || l.contains("scalefac") || l.contains("error while decoding")
        })
        .collect();
    assert!(
        suspicious.is_empty(),
        "ffmpeg decode errors: {suspicious:?}"
    );

    let samples: Vec<f32> = res
        .stdout
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let left: Vec<f32> = samples.chunks_exact(2).map(|c| c[0]).collect();
    let warm = 6 * 1152;
    if left.len() <= warm {
        return;
    }
    let an = &left[warm..];
    let p440 = goertzel_power(an, sr, 440.0);
    let pn = goertzel_power(an, sr, 3000.0) + goertzel_power(an, sr, 7000.0) + 1e-12;
    let ratio = p440 / pn;
    eprintln!("ffmpeg 440Hz/noise ratio = {ratio:.2}");
    assert!(ratio > 10.0, "ffmpeg-decoded tone too weak: {ratio}");
}
