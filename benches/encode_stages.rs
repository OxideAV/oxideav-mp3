// Parallel-array index loops are idiomatic in codec / bench code; skip
// the lint (mirrors the crate root's allow list).
#![allow(clippy::needless_range_loop)]

//! Per-stage micro-benchmarks for the MPEG-1 Layer III **encode** hot
//! path (Round 398, depth-mode). The companion `encode` bench times the
//! whole-stream cost; this one isolates the front-half analysis stages
//! so the ranked hotspot map in `BENCHMARKS.md` can attribute encode
//! time to:
//!
//!   1. **filterbank** — `analyze_granule`: the polyphase analysis
//!      subband filterbank (576 PCM samples → 32 subbands × 18 time
//!      samples), the mirror of the decode-side synthesis filterbank.
//!   2. **mdct_long** — the long-block forward transform the encoder
//!      runs per subband: the encoder-side frequency inversion of the
//!      odd subbands, then for each of the 32 subbands
//!      `forward_overlap` (36-sample overlapped block) →
//!      `window_long_family_analysis` (long analysis window) →
//!      `mdct` (36-point forward MDCT → 18 bins). This is exactly the
//!      sequence the production encoder uses for a long granule.
//!   3. **inner_loop** — `search_bit_budget`: the §C.1.5.4.4 inner rate
//!      loop that finds the smallest `global_gain` whose exact Huffman
//!      bit count fits a per-granule budget. Each candidate gain
//!      re-quantizes all 576 lines and re-counts their Huffman bits, so
//!      this is the search that dominates the whole-stream
//!      input-shape sensitivity seen in the `encode` bench.
//!
//! All three run over the **same** realistic batch of long granules: a
//! 0.5 s mixed mono source (sine carrier + xorshift noise) is split into
//! 576-sample granules; a setup pass runs the real analysis + long-block
//! MDCT chain once to capture each granule's `subband_time` (input to
//! stage 2) and `xr` (input to stage 3). Nothing is timed in setup and
//! no fixture file is committed.
//!
//! The stateful stages (`analyze_granule` carries filterbank history;
//! `forward_overlap` carries per-subband MDCT overlap) get a fresh state
//! at the start of every timed iteration and stream the whole batch
//! through it, exactly as a real encode would.
//!
//! Run with:
//!     cargo bench -p oxideav-mp3 --bench encode_stages

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use oxideav_mp3::analysis::{analyze_granule, AnalysisState};
use oxideav_mp3::frame::MpegVersion;
use oxideav_mp3::inner_loop::search_bit_budget;
use oxideav_mp3::mdct::{forward_overlap, mdct, window_long_family_analysis, MdctState};
use oxideav_mp3::scalefactors::ScaleFactors;
use oxideav_mp3::side_info::{BlockType, GranuleChannel};

/// Samples per granule = 576 (18 per subband × 32 subbands).
const PCM_PER_GRANULE: usize = 576;
const NUM_SUBBANDS: usize = 32;
const SAMPLES_PER_SUBBAND: usize = 18;
const NUM_LINES: usize = 576;
/// Long-block forward-MDCT length (two granule half-blocks overlapped).
const LONG_N: usize = 36;

const SR: u32 = 44_100;
const VERSION: MpegVersion = MpegVersion::Mpeg1;

/// Representative per-granule Huffman bit budget for 128 kbps CBR mono
/// at 44.1 kHz: ≈ 3340 main-data bits per 1152-sample frame across two
/// granules, less side-info / scalefactors — the exact value only shifts
/// how many gain steps the search walks, not the shape of the cost.
const GRANULE_BIT_BUDGET: u64 = 1500;

fn xorshift32(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

/// Mono mixed source: tonal carrier + loud xorshift noise so the encode
/// front end sees a representative wide spectrum (many nonzero Huffman
/// lines) rather than the degenerate all-zero case.
fn mixed_pcm(n: usize, sr: f32, amp: f32) -> Vec<f32> {
    let two_pi = 2.0 * std::f32::consts::PI;
    let carrier = amp * 0.4;
    let noise_amp = amp * 0.55;
    let mut state: u32 = 0x1234_5678;
    (0..n)
        .map(|i| {
            let t = i as f32 / sr;
            let c = (two_pi * 220.0 * t).sin() * carrier;
            let raw = (xorshift32(&mut state) as i32 >> 16) as f32 / 32768.0;
            (c + raw * noise_amp).clamp(-1.0, 1.0)
        })
        .collect()
}

/// A long-block `GranuleChannel` scaffold; `global_gain` is a
/// placeholder the inner-loop search overwrites.
fn long_gc() -> GranuleChannel {
    GranuleChannel {
        part2_3_length: 0,
        big_values: 0,
        global_gain: 0,
        scalefac_compress: 0,
        window_switching_flag: false,
        block_type: BlockType::Long,
        mixed_block_flag: false,
        table_select: [0; 3],
        subblock_gain: [0; 3],
        region0_count: 0,
        region1_count: 0,
        preflag: false,
        scalefac_scale: false,
        count1table_select: false,
    }
}

/// The encoder's long-block frequency inversion: negate the odd time
/// samples of every odd subband (§ encoder analysis, mirror of the
/// decode-side inversion).
fn freq_invert(subband_time: &mut [[f32; SAMPLES_PER_SUBBAND]; NUM_SUBBANDS]) {
    for sb_row in subband_time.iter_mut().skip(1).step_by(2) {
        for t in (1..SAMPLES_PER_SUBBAND).step_by(2) {
            sb_row[t] = -sb_row[t];
        }
    }
}

/// One long-granule forward MDCT: frequency inversion + per-subband
/// `forward_overlap → window_long_family_analysis → mdct`, exactly the
/// sequence the production encoder runs. Returns the 576-line `xr`.
fn mdct_long_granule(
    subband_time: &[[f32; SAMPLES_PER_SUBBAND]; NUM_SUBBANDS],
    mdct_states: &mut [MdctState; NUM_SUBBANDS],
) -> [f32; NUM_LINES] {
    let mut inv = *subband_time;
    freq_invert(&mut inv);
    let mut xr = [0.0f32; NUM_LINES];
    for sb in 0..NUM_SUBBANDS {
        let mut current = [0.0f64; LONG_N / 2];
        for (t, slot) in current.iter_mut().enumerate() {
            *slot = f64::from(inv[sb][t]);
        }
        let frame36 = forward_overlap(&current, &mut mdct_states[sb]);
        let windowed = window_long_family_analysis(&frame36, BlockType::Long);
        let bins = mdct(&windowed, LONG_N);
        for (k, &b) in bins.iter().enumerate() {
            xr[sb * SAMPLES_PER_SUBBAND + k] = b as f32;
        }
    }
    xr
}

fn fresh_mdct_states() -> [MdctState; NUM_SUBBANDS] {
    std::array::from_fn(|_| MdctState::new())
}

/// A captured per-granule batch: the source PCM plus every intermediate
/// each stage replays from, computed once in setup.
struct Batch {
    /// Per-granule PCM (input to stage 1, `filterbank`).
    pcm: Vec<[f32; PCM_PER_GRANULE]>,
    /// Per-granule subband-time (input to stage 2, `mdct_long`).
    subband_time: Vec<[[f32; SAMPLES_PER_SUBBAND]; NUM_SUBBANDS]>,
    /// Per-granule `xr` (input to stage 3, `inner_loop`).
    xr: Vec<[f32; NUM_LINES]>,
}

fn capture_batch() -> Batch {
    let n = (SR as usize) / 2; // 0.5 s
    let src = mixed_pcm(n, SR as f32, 0.7);
    let granules = src.len() / PCM_PER_GRANULE;

    let mut pcm = Vec::with_capacity(granules);
    let mut subband_time = Vec::with_capacity(granules);
    let mut xr = Vec::with_capacity(granules);

    let mut ana = AnalysisState::new();
    let mut mdct_states = fresh_mdct_states();

    for g in 0..granules {
        let mut gr = [0.0f32; PCM_PER_GRANULE];
        gr.copy_from_slice(&src[g * PCM_PER_GRANULE..(g + 1) * PCM_PER_GRANULE]);
        let st = analyze_granule(&gr, &mut ana);
        let x = mdct_long_granule(&st, &mut mdct_states);
        pcm.push(gr);
        subband_time.push(st);
        xr.push(x);
    }

    Batch {
        pcm,
        subband_time,
        xr,
    }
}

fn bench_filterbank(c: &mut Criterion, batch: &Batch) {
    let mut g = c.benchmark_group("encode_stage_filterbank");
    g.throughput(Throughput::Elements(batch.pcm.len() as u64));
    g.bench_function(BenchmarkId::new("analyze_granule", "batch"), |b| {
        b.iter(|| {
            // Fresh filterbank history per iteration; stream the batch.
            let mut ana = AnalysisState::new();
            for gr in &batch.pcm {
                let st = analyze_granule(criterion::black_box(gr), &mut ana);
                criterion::black_box(st[0][0]);
            }
        });
    });
    g.finish();
}

fn bench_mdct_long(c: &mut Criterion, batch: &Batch) {
    let mut g = c.benchmark_group("encode_stage_mdct_long");
    g.throughput(Throughput::Elements(batch.subband_time.len() as u64));
    g.bench_function(
        BenchmarkId::new("forward_overlap+window+mdct", "batch"),
        |b| {
            b.iter(|| {
                let mut states = fresh_mdct_states();
                for st in &batch.subband_time {
                    let xr = mdct_long_granule(criterion::black_box(st), &mut states);
                    criterion::black_box(xr[0]);
                }
            });
        },
    );
    g.finish();
}

fn bench_inner_loop(c: &mut Criterion, batch: &Batch) {
    let gc = long_gc();
    let sf = ScaleFactors::default();
    let mut g = c.benchmark_group("encode_stage_inner_loop");
    g.throughput(Throughput::Elements(batch.xr.len() as u64));
    g.bench_function(BenchmarkId::new("search_bit_budget", "batch"), |b| {
        b.iter(|| {
            for xr in &batch.xr {
                let r = search_bit_budget(
                    criterion::black_box(xr),
                    &gc,
                    &sf,
                    SR,
                    VERSION,
                    GRANULE_BIT_BUDGET,
                );
                criterion::black_box(r.global_gain);
            }
        });
    });
    g.finish();
}

fn stages(c: &mut Criterion) {
    let batch = capture_batch();
    bench_filterbank(c, &batch);
    bench_mdct_long(c, &batch);
    bench_inner_loop(c, &batch);
}

criterion_group!(benches, stages);
criterion_main!(benches);
