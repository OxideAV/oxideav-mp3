// Parallel-array index loops are idiomatic in codec / bench code; skip
// the lint (mirrors the crate root's allow list).
#![allow(clippy::needless_range_loop)]

//! Criterion benchmarks for the MPEG-1 Audio Layer III **decode** hot
//! path (Round 290, depth-mode).
//!
//! Every scenario synthesises its PCM in a setup step, runs it through
//! the crate's own production [`oxideav_mp3::stream_encoder::Mp3Encoder`]
//! to obtain a real CBR Layer III byte stream (so the decoder sees a
//! realistic mix of long / short / mixed granules, populated bit
//! reservoir, region-split Huffman tables and count1 partitions), then
//! times only the decode side. No fixture files are committed — all
//! input bytes come from a deterministic in-bench source.
//!
//! Two whole-stream timing shapes are measured per scenario:
//!
//!   - **trait**: drive the bytes through the registered
//!     [`oxideav_mp3::Mp3CoreDecoder`] (the `Decoder` trait object the
//!     codec registry hands out) — `send_packet` -> `receive_frame`
//!     per MP3 frame, mirroring how a container feeds the codec.
//!   - **direct**: the same byte stream through the bare decode chain
//!     (header -> side-info -> reservoir assemble -> scalefactors ->
//!     Huffman -> requantize -> alias -> IMDCT -> synthesis) with no
//!     trait dispatch or per-frame `AudioFrame` allocation. This is the
//!     baseline the per-stage `decode_stages` bench drills into.
//!
//! Scenarios cover the input shapes that route through different
//! decode branches:
//!
//!   - **tone_mono_44k1_500ms**: 0.5 s 440 Hz sine, mono, 44.1 kHz,
//!     128 kbps. Steady-state long blocks dominate; the reservoir is
//!     exercised by the encoder's bit budgeting.
//!   - **noise_mono_44k1_500ms**: 0.5 s of band-limited xorshift noise,
//!     mono, 44.1 kHz, 128 kbps. Wide spectra push more nonzero
//!     big-values lines and longer count1 runs through the Huffman
//!     decoder than a pure tone, and the encoder's attack detector
//!     trips short / mixed blocks (a different IMDCT window path).
//!   - **sweep_mono_48k_500ms**: 0.5 s logarithmic sweep, mono, 48 kHz,
//!     192 kbps. Moving spectral peak keeps every region's Huffman
//!     table in play across granules; the higher bitrate widens
//!     `part2_3_length` so the per-granule Huffman budget loop runs
//!     longer.
//!   - **tone_mono_32k_500ms**: 0.5 s 300 Hz sine, mono, 32 kHz,
//!     96 kbps — the lowest MPEG-1 sample rate, longest per-sample
//!     duration, distinct scalefactor-band table.
//!
//! Run with:
//!     cargo bench -p oxideav-mp3 --bench decode

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use oxideav_core::{CodecId, CodecParameters, Frame, Packet, SampleFormat, TimeBase};

use oxideav_mp3::codec_decoder::make_decoder;
use oxideav_mp3::frame::{ChannelMode, FrameWalker};
use oxideav_mp3::imdct::ImdctState;
use oxideav_mp3::scalefactors::{decode_scalefactors, MainDataReader, Reservoir};
use oxideav_mp3::side_info::parse_side_info;
use oxideav_mp3::stream_encoder::{Mp3Encoder, SAMPLES_PER_FRAME_MPEG1};
use oxideav_mp3::synth::{SynthState, PCM_PER_GRANULE};
use oxideav_mp3::{
    alias_reduce, decode_huffman, imdct_granule, parse_header, requantize, synth_granule,
};

fn xorshift32(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

/// Mono sine PCM in i16.
fn sine_pcm(n: usize, freq_hz: f32, sr: f32, amp: f32) -> Vec<i16> {
    let two_pi = 2.0 * std::f32::consts::PI;
    let scale = amp * (i16::MAX as f32);
    (0..n)
        .map(|i| {
            let t = i as f32 / sr;
            let s = (two_pi * freq_hz * t).sin() * scale;
            s.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16
        })
        .collect()
}

/// Mono band-limited xorshift noise: a slow tonal carrier plus a loud
/// pseudo-random term so the encoder keeps wide spectra (more nonzero
/// Huffman lines / longer count1 runs) and routinely trips its attack
/// detector into short / mixed granules.
fn noise_pcm(n: usize, sr: f32, amp: f32) -> Vec<i16> {
    let two_pi = 2.0 * std::f32::consts::PI;
    let carrier = amp * 0.4 * (i16::MAX as f32);
    let noise_amp = amp * 0.55 * (i16::MAX as f32);
    let mut state: u32 = 0x1234_5678;
    (0..n)
        .map(|i| {
            let t = i as f32 / sr;
            let c = (two_pi * 220.0 * t).sin() * carrier;
            let raw = (xorshift32(&mut state) as i32 >> 16) as f32 / 32768.0;
            let s = c + raw * noise_amp;
            s.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16
        })
        .collect()
}

/// Mono logarithmic frequency sweep so the spectral peak walks across
/// the bands granule-to-granule, keeping every region's Huffman table
/// and both block types in play over the run.
fn sweep_pcm(n: usize, f0: f32, f1: f32, sr: f32, amp: f32) -> Vec<i16> {
    let two_pi = 2.0 * std::f32::consts::PI;
    let scale = amp * (i16::MAX as f32);
    let dur = n as f32 / sr;
    let k = (f1 / f0).ln() / dur;
    (0..n)
        .map(|i| {
            let t = i as f32 / sr;
            // Instantaneous-phase integral of an exponential chirp.
            let phase = two_pi * f0 * ((k * t).exp() - 1.0) / k;
            let s = phase.sin() * scale;
            s.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16
        })
        .collect()
}

/// Encode mono PCM to a complete CBR Layer III byte stream with the
/// crate's own production encoder.
fn encode_to_mp3(pcm: &[i16], sample_rate: u32, bitrate_kbps: u32) -> Vec<u8> {
    let mut enc = Mp3Encoder::new(bitrate_kbps, sample_rate, ChannelMode::SingleChannel)
        .expect("Mp3Encoder build");
    enc.push_samples(pcm).expect("push_samples");
    let mut bytes: Vec<u8> = Vec::new();
    enc.finish(&mut bytes).expect("finish");
    bytes
}

/// Slice a contiguous MP3 byte stream into per-frame packets via the
/// crate's own `FrameWalker`, mirroring what a real demuxer emits.
fn mp3_to_packets(bytes: &[u8], sample_rate: u32) -> Vec<Packet> {
    let tb = TimeBase::new(1, i64::from(sample_rate));
    let mut out = Vec::new();
    let mut pts: i64 = 0;
    for f in FrameWalker::new(bytes) {
        let mut pkt = Packet::new(0, tb, f.data.to_vec());
        pkt.pts = Some(pts);
        out.push(pkt);
        pts += SAMPLES_PER_FRAME_MPEG1 as i64;
    }
    out
}

/// Whole-stream decode through the registered `Decoder` trait object.
fn run_trait_decode(params: &CodecParameters, packets: &[Packet]) -> usize {
    let mut dec = make_decoder(params).expect("make_decoder");
    let mut total = 0usize;
    for pkt in packets {
        dec.send_packet(pkt).expect("send_packet");
        loop {
            match dec.receive_frame() {
                Ok(Frame::Audio(a)) => total += a.data[0].len(),
                Ok(_) => {}
                Err(_) => break,
            }
        }
    }
    total
}

/// Whole-stream decode through the bare per-stage chain (no trait
/// dispatch, no per-frame `AudioFrame` allocation).
fn run_direct_decode(bytes: &[u8]) -> usize {
    let mut reservoir = Reservoir::new();
    let mut synth_state = SynthState::new();
    let mut imdct_state = ImdctState::new();
    let mut total = 0usize;
    for frame in FrameWalker::new(bytes) {
        let hdr = match parse_header(&frame.data[..4]) {
            Ok(h) => h,
            Err(_) => continue,
        };
        let si_start = 4 + if hdr.crc_protected { 2 } else { 0 };
        let si = match parse_side_info(&hdr, &frame.data[si_start..]) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let si_len = si.byte_len();
        let main_slot = &frame.data[si_start + si_len..frame.data.len()];
        let run = match reservoir.assemble(usize::from(si.main_data_begin), main_slot) {
            Ok(r) => r,
            Err(_) => continue,
        };
        let fsf = decode_scalefactors(&hdr, &si, &run).expect("scalefactors");
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
                let st = imdct_granule(&xar, gc, &mut imdct_state);
                let pcm_f32 = synth_granule(&st, &mut synth_state);
                total += pcm_f32.iter().take(PCM_PER_GRANULE).count();
                bit_cursor += gc.part2_3_length as usize;
            }
        }
    }
    total
}

fn decoder_params(sample_rate: u32) -> CodecParameters {
    let mut p = CodecParameters::audio(CodecId::new("mp3"));
    p.sample_rate = Some(sample_rate);
    p.channels = Some(1);
    p.sample_format = Some(SampleFormat::S16);
    p
}

/// Register both the `trait` and `direct` whole-stream timings for one
/// pre-encoded byte stream under a single benchmark group.
fn bench_stream(c: &mut Criterion, name: &str, sample_rate: u32, pcm_samples: usize, bytes: &[u8]) {
    let packets = mp3_to_packets(bytes, sample_rate);
    let params = decoder_params(sample_rate);
    let mut g = c.benchmark_group(name);
    g.throughput(Throughput::Elements(pcm_samples as u64));
    g.bench_function(
        BenchmarkId::new("trait", "send_packet/receive_frame"),
        |b| {
            b.iter(|| {
                let n = run_trait_decode(
                    criterion::black_box(&params),
                    criterion::black_box(&packets),
                );
                criterion::black_box(n);
            });
        },
    );
    g.bench_function(BenchmarkId::new("direct", "bare_chain"), |b| {
        b.iter(|| {
            let n = run_direct_decode(criterion::black_box(bytes));
            criterion::black_box(n);
        });
    });
    g.finish();
}

fn bench_tone_mono_44k1(c: &mut Criterion) {
    let sr = 44_100u32;
    let n = (sr as usize) / 2; // 0.5 s
    let pcm = sine_pcm(n, 440.0, sr as f32, 0.6);
    let bytes = encode_to_mp3(&pcm, sr, 128);
    bench_stream(c, "decode_tone_mono_44k1_500ms", sr, n, &bytes);
}

fn bench_noise_mono_44k1(c: &mut Criterion) {
    let sr = 44_100u32;
    let n = (sr as usize) / 2;
    let pcm = noise_pcm(n, sr as f32, 0.7);
    let bytes = encode_to_mp3(&pcm, sr, 128);
    bench_stream(c, "decode_noise_mono_44k1_500ms", sr, n, &bytes);
}

fn bench_sweep_mono_48k(c: &mut Criterion) {
    let sr = 48_000u32;
    let n = (sr as usize) / 2;
    let pcm = sweep_pcm(n, 200.0, 16_000.0, sr as f32, 0.6);
    let bytes = encode_to_mp3(&pcm, sr, 192);
    bench_stream(c, "decode_sweep_mono_48k_500ms", sr, n, &bytes);
}

fn bench_tone_mono_32k(c: &mut Criterion) {
    let sr = 32_000u32;
    let n = (sr as usize) / 2;
    let pcm = sine_pcm(n, 300.0, sr as f32, 0.6);
    let bytes = encode_to_mp3(&pcm, sr, 96);
    bench_stream(c, "decode_tone_mono_32k_500ms", sr, n, &bytes);
}

criterion_group!(
    benches,
    bench_tone_mono_44k1,
    bench_noise_mono_44k1,
    bench_sweep_mono_48k,
    bench_tone_mono_32k,
);
criterion_main!(benches);
