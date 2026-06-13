// Parallel-array index loops are idiomatic in codec / bench code; skip
// the lint (mirrors the crate root's allow list).
#![allow(clippy::needless_range_loop)]

//! Per-stage micro-benchmarks for the MPEG-1 Layer III decode hot path
//! (Round 290, depth-mode). The companion `decode` bench times the
//! whole-stream cost; this one isolates each stage so the ranked
//! hotspot map in `BENCHMARKS.md` can attribute time to:
//!
//!   1. **side_info_parse** — `parse_side_info` over a frame's
//!      side-info bytes (bit-unpack of the per-granule fields).
//!   2. **scalefactors** — `decode_scalefactors` over a frame's
//!      assembled main-data run.
//!   3. **huffman** — `decode_huffman` per granule-channel (big-values
//!      region descent + count1 partition).
//!   4. **requantize** — `requantize` per granule (global-gain /
//!      scalefactor power + the `x^(4/3)` dequant curve).
//!   5. **alias** — `alias_reduce` per granule (8 butterflies × 31
//!      subband boundaries, long blocks only).
//!   6. **imdct** — `imdct_granule` per granule (32 subbands ×
//!      IMDCT + windowing + overlap-add).
//!   7. **synth** — `synth_granule` per granule (the polyphase
//!      synthesis filterbank — 18 rows × matrixing + windowing).
//!
//! All seven run over the **same** realistic batch of granule inputs:
//! a 0.5 s mixed mono source (sine carrier + xorshift noise) is encoded
//! by the crate's own production encoder, then the decode chain is run
//! once in setup to capture every per-granule intermediate
//! (`is`/`xr`/`xar`/`subband_time`) and side-info needed to replay each
//! stage standalone. The batch therefore contains the real long /
//! short / mixed block mix the encoder produced; nothing is timed in
//! setup and no fixture file is committed.
//!
//! Run with:
//!     cargo bench -p oxideav-mp3 --bench decode_stages

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use oxideav_mp3::frame::{ChannelMode, FrameWalker, Mp3FrameHeader, MpegVersion};
use oxideav_mp3::imdct::{ImdctState, SAMPLES_PER_SUBBAND};
use oxideav_mp3::scalefactors::{decode_scalefactors, MainDataReader, Reservoir, ScaleFactors};
use oxideav_mp3::side_info::{parse_side_info, GranuleChannel};
use oxideav_mp3::stream_encoder::Mp3Encoder;
use oxideav_mp3::synth::SynthState;
use oxideav_mp3::{
    alias_reduce, decode_huffman, imdct_granule, parse_header, requantize, synth_granule,
};

const NUM_LINES: usize = 576;
const NUM_SUBBANDS: usize = 32;

fn xorshift32(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

/// Mono mixed source: tonal carrier + loud xorshift noise so the
/// encoder produces a representative long / short / mixed granule mix
/// with wide Huffman spectra.
fn mixed_pcm(n: usize, sr: f32, amp: f32) -> Vec<i16> {
    let two_pi = 2.0 * std::f32::consts::PI;
    let carrier = amp * 0.4 * (i16::MAX as f32);
    let noise_amp = amp * 0.55 * (i16::MAX as f32);
    let mut state: u32 = 0x9E37_79B9;
    (0..n)
        .map(|i| {
            let t = i as f32 / sr;
            let c = (two_pi * 440.0 * t).sin() * carrier;
            let raw = (xorshift32(&mut state) as i32 >> 16) as f32 / 32768.0;
            let s = c + raw * noise_amp;
            s.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16
        })
        .collect()
}

fn encode_to_mp3(pcm: &[i16], sample_rate: u32, bitrate_kbps: u32) -> Vec<u8> {
    let mut enc = Mp3Encoder::new(bitrate_kbps, sample_rate, ChannelMode::SingleChannel)
        .expect("Mp3Encoder build");
    enc.push_samples(pcm).expect("push_samples");
    let mut bytes: Vec<u8> = Vec::new();
    enc.finish(&mut bytes).expect("finish");
    bytes
}

/// One frame's parse inputs: header + the raw side-info byte slice, so
/// the `side_info_parse` stage can be replayed without re-walking the
/// stream.
struct FrameSi {
    hdr: Mp3FrameHeader,
    si_bytes: Vec<u8>,
}

/// One frame's scalefactor inputs: header, parsed side-info, and the
/// assembled main-data run.
struct FrameSf {
    hdr: Mp3FrameHeader,
    si: oxideav_mp3::side_info::SideInfo,
    run: Vec<u8>,
}

/// One granule-channel's captured intermediates for the per-granule
/// stages (huffman / requantize / alias / imdct / synth).
struct GranuleCtx {
    gc: GranuleChannel,
    sf: ScaleFactors,
    sample_rate_hz: u32,
    version: MpegVersion,
    /// Assembled main-data run + bit offset of this granule's part3.
    run: Vec<u8>,
    bit_cursor: usize,
    /// Captured `is` (post-Huffman integer lines).
    is: [i32; NUM_LINES],
    /// Captured `xr` (post-requantize).
    xr: [f32; NUM_LINES],
    /// Captured `xar` (post-alias).
    xar: [f32; NUM_LINES],
    /// Captured subband-time output of the IMDCT stage.
    subband_time: [[f32; SAMPLES_PER_SUBBAND]; NUM_SUBBANDS],
}

struct Corpus {
    frames_si: Vec<FrameSi>,
    frames_sf: Vec<FrameSf>,
    granules: Vec<GranuleCtx>,
}

/// Run the decode chain once over a freshly-encoded stream, capturing
/// every per-stage input. None of this is timed.
fn build_corpus() -> Corpus {
    let sr = 44_100u32;
    let n = sr as usize / 2; // 0.5 s
    let pcm = mixed_pcm(n, sr as f32, 0.7);
    let bytes = encode_to_mp3(&pcm, sr, 192);

    let mut reservoir = Reservoir::new();
    let mut synth_state = SynthState::new();
    let mut imdct_state = ImdctState::new();

    let mut frames_si = Vec::new();
    let mut frames_sf = Vec::new();
    let mut granules = Vec::new();

    for frame in FrameWalker::new(&bytes) {
        let hdr = match parse_header(&frame.data[..4]) {
            Ok(h) => h,
            Err(_) => continue,
        };
        let si_start = 4 + if hdr.crc_protected { 2 } else { 0 };
        let si_bytes = frame.data[si_start..].to_vec();
        let si = match parse_side_info(&hdr, &si_bytes) {
            Ok(s) => s,
            Err(_) => continue,
        };
        frames_si.push(FrameSi {
            hdr,
            si_bytes: si_bytes.clone(),
        });

        let si_len = si.byte_len();
        let main_slot = &frame.data[si_start + si_len..frame.data.len()];
        let run = match reservoir.assemble(usize::from(si.main_data_begin), main_slot) {
            Ok(r) => r,
            Err(_) => continue,
        };
        let fsf = decode_scalefactors(&hdr, &si, &run).expect("scalefactors");
        frames_sf.push(FrameSf {
            hdr,
            si: si.clone(),
            run: run.clone(),
        });

        let mut bit_cursor = 0usize;
        for gr in 0..si.granule_count as usize {
            for ch in 0..si.channels as usize {
                let gc = si.granules[gr][ch];
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
                let is = decode_huffman(&mut r, &gc, part3_bits, hdr.sample_rate_hz, hdr.version)
                    .expect("huffman");
                let sf = fsf.granules[gr][ch];
                let xr = requantize(&is, &gc, &sf, hdr.sample_rate_hz, hdr.version);
                let xar = alias_reduce(&xr, &gc);
                let subband_time = imdct_granule(&xar, &gc, &mut imdct_state);
                let _ = synth_granule(&subband_time, &mut synth_state);

                granules.push(GranuleCtx {
                    gc,
                    sf,
                    sample_rate_hz: hdr.sample_rate_hz,
                    version: hdr.version,
                    run: run.clone(),
                    bit_cursor,
                    is,
                    xr,
                    xar,
                    subband_time,
                });
                bit_cursor += si.granules[gr][ch].part2_3_length as usize;
            }
        }
    }

    Corpus {
        frames_si,
        frames_sf,
        granules,
    }
}

fn seek_reader<'a>(run: &'a [u8], bit_cursor: usize) -> MainDataReader<'a> {
    let mut r = MainDataReader::new(run);
    let mut left = bit_cursor;
    while left >= 32 {
        let _ = r.read(32);
        left -= 32;
    }
    if left > 0 {
        let _ = r.read(left as u32);
    }
    r
}

fn bench_stages(c: &mut Criterion) {
    let corpus = build_corpus();
    let ngr = corpus.granules.len() as u64;
    let nsi = corpus.frames_si.len() as u64;
    let nsf = corpus.frames_sf.len() as u64;

    // 1. side_info_parse — per frame.
    {
        let mut g = c.benchmark_group("stage_side_info_parse");
        g.throughput(Throughput::Elements(nsi));
        g.bench_function(BenchmarkId::from_parameter("per_frame_batch"), |b| {
            b.iter(|| {
                let mut acc = 0usize;
                for f in &corpus.frames_si {
                    let si = parse_side_info(&f.hdr, criterion::black_box(&f.si_bytes))
                        .expect("side_info");
                    acc += si.byte_len();
                }
                criterion::black_box(acc);
            });
        });
        g.finish();
    }

    // 2. scalefactors — per frame.
    {
        let mut g = c.benchmark_group("stage_scalefactors");
        g.throughput(Throughput::Elements(nsf));
        g.bench_function(BenchmarkId::from_parameter("per_frame_batch"), |b| {
            b.iter(|| {
                let mut acc = 0u32;
                for f in &corpus.frames_sf {
                    let fsf =
                        decode_scalefactors(&f.hdr, &f.si, criterion::black_box(&f.run)).unwrap();
                    acc = acc.wrapping_add(u32::from(fsf.granule_count));
                }
                criterion::black_box(acc);
            });
        });
        g.finish();
    }

    // 3. huffman — per granule-channel.
    {
        let mut g = c.benchmark_group("stage_huffman");
        g.throughput(Throughput::Elements(ngr));
        g.bench_function(BenchmarkId::from_parameter("per_granule_batch"), |b| {
            b.iter(|| {
                let mut acc = 0i64;
                for gx in &corpus.granules {
                    let mut r = seek_reader(&gx.run, gx.bit_cursor);
                    let part3_bits = u32::from(gx.gc.part2_3_length);
                    let is =
                        decode_huffman(&mut r, &gx.gc, part3_bits, gx.sample_rate_hz, gx.version)
                            .unwrap();
                    acc = acc.wrapping_add(is[0] as i64);
                }
                criterion::black_box(acc);
            });
        });
        g.finish();
    }

    // 4. requantize — per granule.
    {
        let mut g = c.benchmark_group("stage_requantize");
        g.throughput(Throughput::Elements(ngr));
        g.bench_function(BenchmarkId::from_parameter("per_granule_batch"), |b| {
            b.iter(|| {
                let mut acc = 0.0f32;
                for gx in &corpus.granules {
                    let xr = requantize(
                        criterion::black_box(&gx.is),
                        &gx.gc,
                        &gx.sf,
                        gx.sample_rate_hz,
                        gx.version,
                    );
                    acc += xr[0];
                }
                criterion::black_box(acc);
            });
        });
        g.finish();
    }

    // 5. alias — per granule.
    {
        let mut g = c.benchmark_group("stage_alias");
        g.throughput(Throughput::Elements(ngr));
        g.bench_function(BenchmarkId::from_parameter("per_granule_batch"), |b| {
            b.iter(|| {
                let mut acc = 0.0f32;
                for gx in &corpus.granules {
                    let xar = alias_reduce(criterion::black_box(&gx.xr), &gx.gc);
                    acc += xar[0];
                }
                criterion::black_box(acc);
            });
        });
        g.finish();
    }

    // 6. imdct — per granule.
    {
        let mut g = c.benchmark_group("stage_imdct");
        g.throughput(Throughput::Elements(ngr));
        g.bench_function(BenchmarkId::from_parameter("per_granule_batch"), |b| {
            b.iter(|| {
                // Fresh state per pass so overlap-add starts identically;
                // the per-granule IMDCT cost is what we measure.
                let mut state = ImdctState::new();
                let mut acc = 0.0f32;
                for gx in &corpus.granules {
                    let st = imdct_granule(criterion::black_box(&gx.xar), &gx.gc, &mut state);
                    acc += st[0][0];
                }
                criterion::black_box(acc);
            });
        });
        g.finish();
    }

    // 7. synth — per granule.
    {
        let mut g = c.benchmark_group("stage_synth");
        g.throughput(Throughput::Elements(ngr));
        g.bench_function(BenchmarkId::from_parameter("per_granule_batch"), |b| {
            b.iter(|| {
                let mut state = SynthState::new();
                let mut acc = 0.0f32;
                for gx in &corpus.granules {
                    let pcm = synth_granule(criterion::black_box(&gx.subband_time), &mut state);
                    acc += pcm[0];
                }
                criterion::black_box(acc);
            });
        });
        g.finish();
    }
}

criterion_group!(benches, bench_stages);
criterion_main!(benches);
