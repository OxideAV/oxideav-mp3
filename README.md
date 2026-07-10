# oxideav-mp3

[![CI](https://github.com/OxideAV/oxideav-mp3/actions/workflows/ci.yml/badge.svg)](https://github.com/OxideAV/oxideav-mp3/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/oxideav-mp3.svg)](https://crates.io/crates/oxideav-mp3) [![docs.rs](https://docs.rs/oxideav-mp3/badge.svg)](https://docs.rs/oxideav-mp3) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A pure-Rust **MPEG-1 / MPEG-2 Audio Layer III** (MP3) codec for the
[oxideav](https://github.com/OxideAV/oxideav-workspace) framework —
decoder, CBR/VBR encoder, and a stream demuxer.

## Status

Clean-room implementation. Every numeric constant is transcribed from
ISO/IEC 11172-3:1993 and ISO/IEC 13818-3:1997, the staged clean-room
docs, and observer-trace measurements of black-box decoder binaries —
and from no other source. The decoder produces PCM end-to-end for
MPEG-1, MPEG-2 LSF, and all three MPEG-2.5 rates (8 / 11.025 /
12 kHz), mono and stereo; a corpus-wide differential sweep tracks the
staged reference PCM for all 16 fixtures in the float-rounding regime
(normalized RMS error ≤ 1.6e-5). The encoder produces valid CBR and
VBR MP3 streams whose decode is verified against external black-box
validator binaries across a 32-case matrix (every rate, mono/stereo,
long / short / mixed / auto block types, tone and wideband noise —
all ≤ 8e-5, `tests/validator_decode_sweep.rs`).

## Decoder

The full Layer III decode pipeline is implemented:

- **Framing** (§2.4.1.3 / §2.4.2.3): `parse_header` →
  `Mp3FrameHeader` with version / layer / bitrate / sampling-frequency
  ladders, padding-aware frame length, and a self-resynchronising
  `FrameWalker`.
- **Side information** (§2.4.1.7 / §2.4.2.7): both the MPEG-1 (two
  granules, `scfsi`) and MPEG-2 / 2.5 LSF (single granule, widened
  `scalefac_compress`, derived `preflag`) layouts.
- **Main-data stages**: scalefactor decode, Huffman decode (big-values
  regions + count1), requantization, reordering of short blocks,
  stereo processing (MS and intensity), the IMDCT with the four block
  types (long / start / short / stop, including mixed blocks), and the
  polyphase synthesis filterbank.
- **Bit reservoir** (§2.4.2.7): `main_data_begin` back-references
  across frames.
- **CRC-16** frame protection (§2.4.3.1).

Output is interleaved PCM through the `oxideav_core::Decoder` trait. The
final float-to-`i16` conversion follows §2.4.3.4.7 exactly: the decoder
output range is `[-1.0, +1.0]`, so samples scale by the MSB weight
`2^15 = 32768` and round to the nearest integer with half-integer values
rounded away from zero (§2.3 "Nearest integer operator"). The decode
path is validated by PSNR self-round-trip tests and against staged
reference fixtures.

## Encoder

A clean-room encoder built from the §C.1.5 / §2.4 encode procedure:
forward MDCT and analysis windowing, the polyphase analysis subband
filterbank, the §2.4.3.4.7 quantizer, the §C.1.5.4.4 inner-loop
`global_gain` search with exact Huffman bit counting, the §C.1.5.4.3
outer distortion-control loop (with `scalefac_scale` escalation and
preemphasis decision), §2.4.1.7 main-data assembly, and §2.4.2.7
cross-frame bit-reservoir scheduling. Additional capabilities:

- Independent stereo and joint-stereo MS encode, with a per-frame
  auto MS/LR picker.
- Forward short-block and mixed-block MDCT paths, plus a signal-driven
  attack detector and the `LONG → START → SHORT → STOP → LONG` block-type
  state machine (opt-in auto block typing).
- True-VBR per-frame bitrate, opt-in Xing / Info VBR information-frame
  emission with auto-filled TOC, and opt-in CRC-16 protection.
- MPEG-2.5 frame header writing and sample-rate dispatch, with the
  low-rate (8 / 11.025 / 12 kHz) Layer III scalefactor-band tables wired
  through the shared band-boundary functions, so quantization, the
  inner/outer loops, reorder, stereo, and the psychoacoustic
  threshold-in-quiet path all use the correct band layout at the
  MPEG-2.5 rates. The tables carry the **deployed de-facto layout**,
  measured per spectral line by r405 observer-trace probes against two
  independent black-box decoder binaries: 11.025 / 12 kHz reuse the
  ISO/IEC 13818-3 **16 kHz LSF table pair** (long + short) — not the
  per-rate half-rate siblings hypothesised in
  `mpeg2.5-scalefactor-bands.md` — and 8 kHz uses the distinct
  Fraunhofer tables from that doc (read back verbatim by the probe).
  The §2.4.2.7 short-block region-0 boundary is band-relative
  (`3·short_starts[3]` — 72 lines at 8 kHz), and mixed blocks are
  refused at 8 kHz (the 8 kHz short table has no boundary at the
  36-line long/short split; deployed decoders disagree there).
  Encoder output at all three rates — long, short, auto block-type,
  tone and wideband noise, mono and stereo — decodes on both external
  validators in the float-rounding regime.

The encoder is reachable through the `oxideav_core::Encoder` trait and
several direct `make_encoder*` factory variants.

## Demuxer

The `demuxer` module provides a streaming MP3 container reader: ID3v2
skip, frame iteration, and parsing of the `Xing` / `Info` VBR header
and the trailing gapless-playback extension (encoder delay / padding),
so `trimmed_duration_samples()` reports the gapless sample count.
Registered as a container alongside the codec.

**Seeking** is frame-accurate. The Xing-TOC path (VBR) and the
proportional bitrate path (CBR / free-format) position the cursor on a
byte estimate, snap it forward to the next real frame syncword, then
re-derive the **exact** PTS of the frame they landed on by counting
whole frames from the first audio frame — so `seek_to` returns a
whole-frame-aligned timestamp that matches the PTS later stamped on the
next packet and keeps the stream strictly monotone (no leaked estimate).
Each emitted packet's **keyframe flag** reflects the §2.4.2.7 bit
reservoir: a Layer III frame is a random-access point only when its
`main_data_begin` back-pointer is zero (it borrows no main data from
earlier frames); frames with a non-zero back-pointer are flagged
non-keyframe so a seeker doesn't treat them as safe entry points.
Layers I/II carry no reservoir, so every frame is a keyframe.

**Free-format streams** (`bitrate_index == 0`) are demuxed end-to-end.
The standard (§2.4.1.3) fixes a free-format stream's bitrate but omits
it from the header bitrate table, so the per-frame length is not
derivable from the header. At open time the demuxer measures the
constant **unpadded** frame body once — the byte distance between the
first two frame syncs that agree on `(version, layer, sample_rate)`,
minus the first frame's own padding slot — and stores it; every
subsequent frame's length is that base plus its own padding bit
(`Mp3FrameHeader::frame_len_free_format`). An effective constant
bitrate (`base · 8 · sample_rate / samples_per_frame`) is derived so
the duration estimate and the CBR/proportional seek path work on
free-format input too. A single free-format frame (no second sync to
measure against) is rejected. End-to-end mono + stereo demux→decode is
byte-exact with the same stream demuxed as CBR
(`tests/demuxer_free_format_roundtrip.rs`).

## MPEG-2.5 decode

The `Decoder` trait wrapper accepts MPEG-1, MPEG-2 LSF, and all three
MPEG-2.5 extension rates. The 11.025 / 12 kHz rates reuse the in-repo
ISO/IEC 13818-3 22.05 / 24 kHz LSF scalefactor-band tables verbatim
(fully grounded per `mpeg2.5-scalefactor-bands.md`, #147/#151). The
**8 kHz** rate uses the distinct Fraunhofer 8 kHz table — its top long
bands collapse to width 2 at the 4 kHz Nyquist — transcribed into
`requantize::{LONG,SHORT}_STARTS_MPEG25_8` from the staged doc's "8 kHz,
long/short blocks" tables (published-factual; satisfies the Table-B.2
structural invariants: Σ = 576 long / 192 short, contiguous, 22/13
bands). All three rates decode through the same chain as MPEG-2 LSF,
byte-exact with the direct decode chain. The 11.025 kHz path is
validated end-to-end against the staged `layer3-mpeg25-11025-32kbps`
reference PCM (`tests/mpeg25_reference_pcm.rs`): the decode locks at the
canonical 1105-sample codec delay with steady-state normalized RMS error
≈ 1e-4, and the production trait decoder reproduces the direct chain
byte-exact. A dedicated 8 kHz observer-trace fixture would further
corroborate the published-factual 8 kHz boundaries (`MPEG-2.5-GAP.md`).

This validation surfaced and fixed a latent decoder bug shared by all
versions: the per-granule decode loop handed the *full* `part2_3_length`
to `decode_huffman` while the reader still sat on the part-2 scalefactor
bits, so frames with a non-zero `slen` partition (scalefac_compress ≠ 0)
mis-read scalefactors as Huffman codewords. `decode_scalefactors` now
records the part-2 bit length (`FrameScaleFactors::part2_bits`); the
decode path skips it and budgets Huffman at `part2_3_length −
part2_bits`.

## Quality presets

A named psychoacoustic **quality knob** bundles the §C.1.5 / Annex D
perceptual toggles into one level. `Mp3Encoder::with_quality_preset`
(and the registry-path `make_encoder_quality_preset` factory) take a
`QualityPreset` — `Transparent` / `High` / `Standard` / `Fast`, ordered
by the §D.1 Step 3 threshold offset (`-24` / `-12` / `0` / `+6` dB;
`High` and `Standard` mirror the spec's own `>= 96` / `< 96` kbit/s-per-
channel branches exactly). Each preset lowers to:

- a §D.1 Step 3 threshold offset on the per-band threshold-in-quiet bowl;
- optionally the §C.1.5.3.2.1 **Model 2** per-band masking analysis;
- optionally the §C.1.5.2 Model-2-driven **block-type** scheduler.

The knob is **rate-graceful**: at the three staged Annex D rates (32 /
44.1 / 48 kHz) the richer presets arm the full signal-dependent Model 2
path (the preset offset translates the Model 2 geometric-mean anchor via
`XminThresholds::from_layer3_granule_with_offset_db`, so the preset's
*level* reaches the signal-dependent path while its content-driven *shape*
is preserved). At the MPEG-2 LSF / MPEG-2.5 rates (no staged calculation-
partition tables) a preset falls back to the signal-independent per-band
threshold-in-quiet vector translated by the preset offset, so a preset is
usable at every supported rate. `quality_preset()` /
`quality_preset_is_signal_dependent()` / `installed_per_band_xmin()`
surface which path was taken.

## Not yet supported

- Annex D psychoacoustic shaping remains **opt-in** under the perceptual
  presets above; the bare-constructor default quantization is still
  rate/distortion-driven. A perceptual preset applied by default at
  construction time (rather than as an explicit `with_quality_preset`
  call) is a possible future ergonomic refinement.

## Robustness

A `cargo-fuzz` harness drives attacker-controlled bytes through the
attack surfaces for panic-freedom: `decode` (multi-packet decoder
sessions with crafted valid-sync headers, reservoir / reset / flush
state transitions), `granule` (the per-granule decode primitives), and
`demux` (container open with ID3v2 / Xing / VBRI / LAME parsing,
packet iteration, and the TOC / proportional seek paths).

## License

MIT — see [LICENSE](./LICENSE).
