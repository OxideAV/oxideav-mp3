// Parallel-array index loops are idiomatic in codec / bench code; skip
// the lint (mirrors the crate root's allow list).
#![allow(clippy::needless_range_loop)]

//! Criterion benchmarks for the MPEG-1 Audio Layer III **encode** hot
//! path (Round 398, depth-mode).
//!
//! The companion `decode` / `decode_stages` benches time the analysis-
//! free decode side; this one times the encoder — the crate's heaviest
//! code path (the analysis polyphase filterbank, the forward MDCT with
//! long / short / mixed windowing, the psychoacoustic threshold model,
//! the nested inner rate loop and outer distortion loop, Huffman table
//! selection + code emission, and side-info / main-data / reservoir
//! assembly).
//!
//! Every scenario synthesises its input PCM in a setup step (nothing is
//! timed in setup, no fixture file is committed) and then times only
//! the encode of that PCM into a complete CBR Layer III byte stream.
//! Two whole-stream timing shapes are measured per scenario:
//!
//!   - **direct**: drive the raw i16 PCM through the bare
//!     [`oxideav_mp3::stream_encoder::Mp3Encoder`]
//!     (`new` -> `push_samples` -> `finish`) — the whole encode chain
//!     with no trait dispatch and no per-frame `AudioFrame` allocation.
//!   - **trait**: the same PCM through the registered
//!     [`oxideav_core::Encoder`] trait object the codec registry hands
//!     out — `send_frame` (with i16-LE `AudioFrame` bytes) then `flush`
//!     and drain `receive_packet`, mirroring how a muxer feeds the
//!     codec.
//!
//! Scenarios cover the input shapes that route through different encode
//! branches:
//!
//!   - **tone_mono_44k1_500ms**: 0.5 s 440 Hz sine, mono, 44.1 kHz,
//!     128 kbps. Steady-state long blocks; the inner loop settles on a
//!     narrow global-gain range and the reservoir fills gently.
//!   - **noise_mono_44k1_500ms**: 0.5 s of band-limited xorshift noise,
//!     mono, 44.1 kHz, 128 kbps. Wide spectra push more nonzero
//!     big-values lines and longer count1 runs through Huffman table
//!     selection, and the attack detector trips short / mixed blocks
//!     (the short-window MDCT + reorder path).
//!   - **sweep_mono_48k_500ms**: 0.5 s logarithmic sweep, mono, 48 kHz,
//!     192 kbps. The moving spectral peak keeps every region's Huffman
//!     table and the per-band distortion decision in play across
//!     granules; the higher bitrate widens the per-granule bit budget
//!     so the inner loop iterates over a larger `part2_3_length`.
//!   - **tone_mono_32k_500ms**: 0.5 s 300 Hz sine, mono, 32 kHz,
//!     96 kbps — the lowest MPEG-1 sample rate, distinct scalefactor-
//!     band table, tighter per-frame bit budget.
//!   - **mixed_stereo_44k1_500ms**: 0.5 s of an independent left tone
//!     and right noise, stereo, 44.1 kHz, 192 kbps. Both channels run
//!     the full per-channel encode; the noise channel exercises the
//!     short/mixed path while the tone channel stays long, so the
//!     reservoir juggles unequal per-granule demands.
//!
//! Run with:
//!     cargo bench -p oxideav-mp3 --bench encode

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use oxideav_core::{
    AudioFrame, CodecId, CodecParameters, Error, Frame, RuntimeContext, SampleFormat,
};

use oxideav_mp3::frame::ChannelMode;
use oxideav_mp3::stream_encoder::Mp3Encoder;

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

/// Interleaved LR stereo PCM: a pure tone in the left channel and
/// carrier+noise in the right, so the two channels take different
/// block-type / Huffman branches within the same frame.
fn stereo_mixed_pcm(n_per_ch: usize, sr: f32, amp: f32) -> Vec<i16> {
    let left = sine_pcm(n_per_ch, 440.0, sr, amp);
    let right = noise_pcm(n_per_ch, sr, amp);
    let mut out = Vec::with_capacity(n_per_ch * 2);
    for i in 0..n_per_ch {
        out.push(left[i]);
        out.push(right[i]);
    }
    out
}

/// Encode interleaved i16 PCM to a complete CBR Layer III byte stream
/// through the bare production encoder.
fn run_direct_encode(pcm: &[i16], sample_rate: u32, bitrate_kbps: u32, mode: ChannelMode) -> usize {
    let mut enc = Mp3Encoder::new(bitrate_kbps, sample_rate, mode).expect("Mp3Encoder build");
    enc.push_samples(pcm).expect("push_samples");
    let mut bytes: Vec<u8> = Vec::new();
    enc.finish(&mut bytes).expect("finish")
}

/// Build encode-side `CodecParameters` for the registry factory.
fn encoder_params(sample_rate: u32, channels: u16, bit_rate_bps: u64) -> CodecParameters {
    let mut p = CodecParameters::audio(CodecId::new("mp3"));
    p.sample_rate = Some(sample_rate);
    p.channels = Some(channels);
    p.sample_format = Some(SampleFormat::S16);
    p.bit_rate = Some(bit_rate_bps);
    p
}

/// Encode interleaved i16 PCM through the registered `Encoder` trait
/// object: one `send_frame` carrying the whole clip as i16-LE bytes,
/// then `flush` and drain every emitted `Packet`.
fn run_trait_encode(params: &CodecParameters, pcm: &[i16], channels: u16) -> usize {
    let mut ctx = RuntimeContext::default();
    oxideav_mp3::register(&mut ctx);
    let mut enc = ctx
        .codecs
        .first_encoder(params)
        .expect("encoder factory present after register()");

    let mut bytes = Vec::with_capacity(pcm.len() * 2);
    for s in pcm {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    let samples_per_ch = (pcm.len() / channels as usize) as u32;
    let f = AudioFrame {
        samples: samples_per_ch,
        pts: None,
        data: vec![bytes],
    };
    enc.send_frame(&Frame::Audio(f)).expect("send_frame");
    enc.flush().expect("flush");

    let mut total = 0usize;
    loop {
        match enc.receive_packet() {
            Ok(p) => total += p.data.len(),
            Err(Error::Eof) => break,
            Err(e) => panic!("unexpected error draining encoder: {e}"),
        }
    }
    total
}

/// Register the `direct` and `trait` whole-stream encode timings for
/// one PCM clip under a single benchmark group.
fn bench_clip(
    c: &mut Criterion,
    name: &str,
    sample_rate: u32,
    bitrate_kbps: u32,
    mode: ChannelMode,
    pcm: &[i16],
) {
    let channels = u16::from(mode.channel_count());
    let samples_per_ch = pcm.len() / channels as usize;
    let params = encoder_params(sample_rate, channels, u64::from(bitrate_kbps) * 1000);

    let mut g = c.benchmark_group(name);
    g.throughput(Throughput::Elements(samples_per_ch as u64));
    g.bench_function(BenchmarkId::new("direct", "push_samples/finish"), |b| {
        b.iter(|| {
            let n = run_direct_encode(criterion::black_box(pcm), sample_rate, bitrate_kbps, mode);
            criterion::black_box(n);
        });
    });
    g.bench_function(BenchmarkId::new("trait", "send_frame/flush"), |b| {
        b.iter(|| {
            let n = run_trait_encode(
                criterion::black_box(&params),
                criterion::black_box(pcm),
                channels,
            );
            criterion::black_box(n);
        });
    });
    g.finish();
}

fn bench_tone_mono_44k1(c: &mut Criterion) {
    let sr = 44_100u32;
    let n = (sr as usize) / 2; // 0.5 s
    let pcm = sine_pcm(n, 440.0, sr as f32, 0.6);
    bench_clip(
        c,
        "encode_tone_mono_44k1_500ms",
        sr,
        128,
        ChannelMode::SingleChannel,
        &pcm,
    );
}

fn bench_noise_mono_44k1(c: &mut Criterion) {
    let sr = 44_100u32;
    let n = (sr as usize) / 2;
    let pcm = noise_pcm(n, sr as f32, 0.7);
    bench_clip(
        c,
        "encode_noise_mono_44k1_500ms",
        sr,
        128,
        ChannelMode::SingleChannel,
        &pcm,
    );
}

fn bench_sweep_mono_48k(c: &mut Criterion) {
    let sr = 48_000u32;
    let n = (sr as usize) / 2;
    let pcm = sweep_pcm(n, 200.0, 16_000.0, sr as f32, 0.6);
    bench_clip(
        c,
        "encode_sweep_mono_48k_500ms",
        sr,
        192,
        ChannelMode::SingleChannel,
        &pcm,
    );
}

fn bench_tone_mono_32k(c: &mut Criterion) {
    let sr = 32_000u32;
    let n = (sr as usize) / 2;
    let pcm = sine_pcm(n, 300.0, sr as f32, 0.6);
    bench_clip(
        c,
        "encode_tone_mono_32k_500ms",
        sr,
        96,
        ChannelMode::SingleChannel,
        &pcm,
    );
}

fn bench_mixed_stereo_44k1(c: &mut Criterion) {
    let sr = 44_100u32;
    let n = (sr as usize) / 2; // 0.5 s per channel
    let pcm = stereo_mixed_pcm(n, sr as f32, 0.6);
    bench_clip(
        c,
        "encode_mixed_stereo_44k1_500ms",
        sr,
        192,
        ChannelMode::Stereo,
        &pcm,
    );
}

criterion_group!(
    benches,
    bench_tone_mono_44k1,
    bench_noise_mono_44k1,
    bench_sweep_mono_48k,
    bench_tone_mono_32k,
    bench_mixed_stereo_44k1,
);
criterion_main!(benches);
