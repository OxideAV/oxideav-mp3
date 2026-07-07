# oxideav-mp3 — Benchmarks

Criterion benchmarks for the MPEG-1 Audio Layer III **decode** hot
path, landed in Round 290 (depth-mode), and the **encode** hot path,
added in Round 398 (depth-mode). Both rounds are measurement-only —
**no behaviour change**: decoded PCM is byte-identical to the pre-round
decoder, and the encoded byte streams are byte-identical to the
pre-round encoder.

```
cargo bench -p oxideav-mp3 --bench decode          # decode whole-stream
cargo bench -p oxideav-mp3 --bench decode_stages   # decode per-stage
cargo bench -p oxideav-mp3 --bench encode          # encode whole-stream
cargo bench -p oxideav-mp3 --bench encode_stages   # encode per-stage
```

The decode half is documented first; the [encoder
benchmarks](#encoder-benchmarks-round-398) section follows.

Every benchmark is self-contained: each scenario synthesises its input
PCM from a deterministic in-bench source (sine / log-sweep / xorshift
noise), runs it through the crate's own production
`stream_encoder::Mp3Encoder` to obtain a real CBR Layer III byte stream
(so the decoder sees a realistic mix of long / short / mixed granules, a
populated bit reservoir, region-split big-values Huffman tables and
count1 partitions), and only the **decode** side is timed. No fixture
files are committed.

## Harnesses

### `decode` — whole-stream

Times the full Layer III decode of a pre-encoded mono stream two ways:

* **trait** — through the registered `Mp3CoreDecoder` (`Decoder` trait
  object the codec registry hands out): `send_packet` → `receive_frame`
  per MP3 frame, as a container would feed the codec.
* **direct** — the same bytes through the bare per-stage chain (header →
  side-info → reservoir assemble → scalefactors → Huffman → requantize →
  alias → IMDCT → synthesis), with no trait dispatch and no per-frame
  `AudioFrame` allocation.

Scenarios: `tone_mono_44k1_500ms` (440 Hz sine, 128 kbps),
`noise_mono_44k1_500ms` (carrier + noise, 128 kbps),
`sweep_mono_48k_500ms` (200 Hz→16 kHz log sweep, 192 kbps),
`tone_mono_32k_500ms` (300 Hz sine, 96 kbps).

### `decode_stages` — per-stage

Isolates each decode stage over the **same** captured batch. A 0.5 s
mixed mono source (carrier + xorshift noise) at 44.1 kHz / 192 kbps is
encoded, then the decode chain is run once in setup to capture every
per-granule intermediate (`is` / `xr` / `xar` / `subband_time`) plus the
side-info and assembled main-data runs. Each stage then replays
standalone over the batch. The batch is **20 frames / 40 granules** of
the real long / short / mixed block mix the encoder produced.

## Results

Measured on the development host (Apple Silicon, `--measurement-time 3`).
Absolute numbers are host-relative; the **ranking and shares** are the
durable deliverable.

### Whole-stream (`decode`, median time for 0.5 s of audio)

| scenario                      | trait    | direct   | trait overhead |
| ----------------------------- | -------- | -------- | -------------- |
| `tone_mono_44k1_500ms`        | 6.42 ms  | 6.41 ms  | within noise   |
| `noise_mono_44k1_500ms`       | 6.22 ms  | 6.18 ms  | within noise   |
| `sweep_mono_48k_500ms`        | 7.16 ms  | 7.20 ms  | within noise   |
| `tone_mono_32k_500ms`         | 4.48 ms  | ~4.47 ms | within noise   |

Throughput ≈ **3.4 M PCM samples/s** at 44.1 kHz (≈77× real-time mono).
The `Decoder` trait wrapper (`send_packet` / `receive_frame` dispatch +
`AudioFrame` allocation + LE-byte packing) costs nothing measurable over
the bare chain — the per-stage DSP dominates entirely.

### Per-stage (`decode_stages`, median over the 20-frame / 40-granule batch)

| stage             | unit       | batch time | per unit    |
| ----------------- | ---------- | ---------- | ----------- |
| `synth`           | granule    | 4.02 ms    | 100.6 µs    |
| `imdct`           | granule    | 2.01 ms    | 50.4 µs     |
| `huffman`         | granule    | 366 µs     | 9.16 µs     |
| `requantize`      | granule    | 72.7 µs    | 1.82 µs     |
| `scalefactors`    | frame      | 5.85 µs    | 293 ns      |
| `alias`           | granule    | 2.34 µs    | 58.5 ns     |
| `side_info_parse` | frame      | 790 ns     | 39.5 ns     |

## Ranked hotspot map

Per-granule decode cost (per-frame stages amortised over 2 granules):

| rank | stage             | per granule | share of decode |
| ---- | ----------------- | ----------- | --------------- |
| 1    | **synth** (filterbank) | 100.6 µs | **≈ 62 %**   |
| 2    | **imdct**         | 50.4 µs     | **≈ 31 %**      |
| 3    | **huffman**       | 9.16 µs     | ≈ 5.6 %         |
| 4    | requantize        | 1.82 µs     | ≈ 1.1 %         |
| 5    | scalefactors      | 0.15 µs     | ≈ 0.1 %         |
| 6    | alias             | 0.059 µs    | ≈ 0.04 %        |
| 7    | side_info_parse   | 0.02 µs     | ≈ 0.01 %        |

**The polyphase synthesis filterbank and the IMDCT together account for
~93 % of decode time.** Everything in the bitstream-parse / entropy /
requantization front half — Huffman, requantize, scalefactors, alias,
side-info — sums to under 7 %.

### Observations for a future optimisation round (no change made here)

* **`synth` (rank 1).** `synth_granule` gathers each subband row into an
  `f64` working buffer and runs `synth_row` (32-band matrixing + the
  512-tap window) 18 times per granule. The `f64` accumulation path and
  the per-row `[f64; 32]` gather are the obvious first A/B targets; the
  matrixing is the classic candidate for a fast DCT-IV / split-radix
  decomposition.
* **`imdct` (rank 2).** `imdct_granule` runs one IMDCT + windowing +
  overlap-add per subband (32×). The per-subband windowed-block call and
  the long/short window selection branch are where time lives.
* **`huffman` (rank 3).** Distant third, but the only entropy-side
  stage with material cost; the big-values region descent dominates over
  the count1 partition. Input-shape-sensitive (wider spectra → more
  nonzero lines), so the `noise` / `sweep` whole-stream scenarios are
  the right A/B inputs for any table-lookup restructuring.
* Stages 4–7 are noise-floor; optimising them cannot move whole-stream
  decode and they are listed only for completeness.

These are **suggestions for a later round**, not work done here. This
round's deliverable is the harness above and this ranking.

## Encoder benchmarks (Round 398)

The `encode` bench times the whole PCM → Layer III encode — the crate's
heaviest path (analysis polyphase filterbank, forward MDCT with
long/short/mixed windowing, the psychoacoustic threshold model, the
nested inner rate loop and outer distortion loop, Huffman table
selection + code emission, and side-info / main-data / reservoir
assembly). Each scenario synthesises its input PCM in setup (nothing is
timed in setup, no fixture committed) and is timed two ways:

* **direct** — the bare `Mp3Encoder` (`new` → `push_samples` →
  `finish`), the whole encode chain with no trait dispatch.
* **trait** — the same PCM through the registered `Encoder` trait
  object (`send_frame` with i16-LE `AudioFrame` bytes → `flush` → drain
  `receive_packet`), as a muxer would drive it.

### Whole-stream (`encode`, median time for 0.5 s of audio)

| scenario                        | trait     | direct    | trait overhead |
| ------------------------------- | --------- | --------- | -------------- |
| `tone_mono_44k1_500ms`          | 33.1 ms   | 35.1 ms   | within noise   |
| `noise_mono_44k1_500ms`         | 51.5 ms   | 52.8 ms   | within noise   |
| `sweep_mono_48k_500ms`          | 36.3 ms   | 36.9 ms   | within noise   |
| `tone_mono_32k_500ms`           | 22.5 ms   | 22.6 ms   | within noise   |
| `mixed_stereo_44k1_500ms`       | 84.0 ms   | 84.2 ms   | within noise   |

Measured on the same host as the decode tables above (medians; the
±spread is a few %). As on the decode side, the `Encoder` trait wrapper
(`send_frame` / `flush` dispatch + `AudioFrame` handling + LE-byte
packing) costs nothing measurable over the bare chain — the DSP and the
rate/distortion loops dominate entirely.

### Observations

* **Encode is ~5× the cost of decode.** A mono 44.1 kHz tone decodes in
  ~6.4 ms (table above) but encodes in ~33 ms of the same 0.5 s clip:
  ≈ 668 K PCM samples/s, ≈ 15× real-time mono. The asymmetry is the
  analysis-plus-search cost the decoder never pays — the psychoacoustic
  threshold model and the two nested quantization loops (each inner
  iteration re-counts every granule's Huffman bits) run on the encode
  side only.
* **Input shape dominates.** Wide-spectrum noise (52.8 ms) costs ≈ 1.5×
  the steady tone (35.1 ms) at the same rate: more nonzero big-values
  lines lengthen every inner-loop Huffman re-count, and the attack
  detector trips short / mixed granules whose per-window MDCT + reorder
  is heavier than a single long block. The log sweep sits between the
  two. The 32 kHz tone is the cheapest per clip — fewer samples per
  0.5 s and a tighter per-frame budget that the inner loop settles
  quickly.
* **Stereo ≈ 2× mono, as expected** — each channel runs the full
  independent per-channel encode; the 84 ms mixed-stereo clip is
  essentially the tone-channel cost plus the noise-channel cost.

### Per-stage (`encode_stages`, median over a 38-granule batch)

Isolates the encode front-half stages over the **same** captured batch
(0.5 s mixed mono → 38 long granules; each granule's `subband_time` and
`xr` captured once in setup). The stateful stages get a fresh state per
iteration and stream the whole batch, as a real encode would.

| stage        | unit    | batch time | per granule |
| ------------ | ------- | ---------- | ----------- |
| `inner_loop` | granule | 41.4 ms    | **1.09 ms** |
| `filterbank` | granule | 3.70 ms    | 97.3 µs     |
| `mdct_long`  | granule | 1.78 ms    | 47.0 µs     |

`inner_loop` here is the bare `search_bit_budget` at a representative
128 kbps per-granule budget; the whole-stream encode additionally runs
the outer distortion loop *around* it, so the real rate/distortion
search cost is even larger than the figure above.

### Ranked encode hotspot map

Per-granule cost of the measured front-half stages:

| rank | stage        | per granule | vs. next |
| ---- | ------------ | ----------- | -------- |
| 1    | **inner_loop** (rate search) | 1.09 ms | **≈ 11×** |
| 2    | filterbank   | 97.3 µs     | ≈ 2×      |
| 3    | mdct_long    | 47.0 µs     | —         |

**The inner rate loop dominates encode by an order of magnitude.** Each
candidate `global_gain` the search walks re-quantizes all 576 lines and
re-counts their Huffman bits; a wide-spectrum granule (more nonzero
lines) both lengthens every count and, at a tight budget, forces the
search across more gains — exactly the input-shape sensitivity the
whole-stream table shows (noise ≈ 1.5× a tone). This is the first place
any future encode-speed round should look: the front-end DSP
(filterbank + MDCT, ≈ 144 µs/granule combined) is under 15 % of the
inner-loop cost, and the psychoacoustic model + outer loop wrap *around*
this search rather than replacing it.

Observations only — **no behaviour change was made this round**; the
encoded byte streams are byte-identical to before. The harness plus this
ranking are the deliverable; the outer-loop / psychoacoustic / Huffman-
select stages and a short-block MDCT variant are the natural next depth
steps.
