# oxideav-mp3

[![CI](https://github.com/OxideAV/oxideav-mp3/actions/workflows/ci.yml/badge.svg)](https://github.com/OxideAV/oxideav-mp3/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/oxideav-mp3.svg)](https://crates.io/crates/oxideav-mp3) [![docs.rs](https://docs.rs/oxideav-mp3/badge.svg)](https://docs.rs/oxideav-mp3) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A pure-Rust **MPEG-1 / MPEG-2 / MPEG-2.5 Audio Layer III** (MP3) codec
for the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework — decoder, CBR / VBR encoder, and the MPEG audio
elementary-stream demuxer / muxer for `.mp3`, `.mp2` and `.mp1` files
(the demuxer labels a Layer I / II stream `mp1` / `mp2` for the
decoders of those layers; this crate decodes Layer III).

Clean-room implementation: every numeric constant is transcribed from
ISO/IEC 11172-3:1993 and ISO/IEC 13818-3:1997, the staged clean-room
docs, and input / output measurements of black-box binaries — and from
no other source.

## What is verified against outside tools

`tests/blackbox_matrix.rs` generates its streams in a temp directory
and skips when the tools (`lame`, `mpg123`, `ffmpeg`) are not
installed.

| Direction | Cells | Result |
|---|---|---|
| Streams of a black-box encoder -> this decoder | 970 | 969 equal to both black-box decoders, 1 where those two differ from each other (this decoder equals one) |
| This encoder -> two black-box decoders | 960 configurations, 952 offered | 933 accepted without complaint, gapless-exact and sample-equal on both; 19 where the two decoders differ from each other (this crate equals one); 0 unreferenced |
| Registry encoder + `.mp3` muxer -> two black-box decoders | 18 | 18 accepted without complaint, sample-equal |

**Foreign streams.** Every sampling frequency (48 / 44.1 / 32, 24 /
22.05 / 16, 12 / 11.025 / 8 kHz) x the full bitrate ladder x joint
stereo / stereo / forced MS / dual channel / mono; free format below,
between and above the ladder (to 640 kbit/s); two VBR routines at
quality 0..9 and ABR; CRC protection, reservoir off, strict buffer
constraints, emphasis / copyright / original bits, clipping input,
with and without the information frame, ID3v1 / ID3v2 / padded ID3v2;
1-sample to 2305-sample inputs and 40 s streams; truncated tails,
truncated heads, junk prefixes and two streams concatenated (same and
different sampling frequency). Judged against the black-box decoders'
**float** output: every sample of this decoder's 16-bit output within
2 LSB and the RMS difference at most 0.41 LSB (measured: maximum 0.5 to
0.56 LSB, RMS 0.25 to 0.29 LSB — the rounding of the 16-bit output
itself), exact sample count, and the gapless figures of every tagged
stream equal to the encoder's input length. The one split cell is
8 kbit/s forced MS at 24 kHz, where one black-box decoder renders
granules with `big_values == 0` differently from the other and from
this decoder.

**Own streams.** Every sampling frequency x full ladder x mono /
stereo / dual channel / MS / adaptive joint stereo, and at six rates
every option: CRC, forced short and mixed blocks, both block-switching
schedulers, scfsi reuse, threshold-in-quiet and outer-loop
constructors, the four quality presets, VBR, and intensity stereo
alone / with MS / adaptive, with long, short and switched blocks. Each
stream must draw nothing on a black-box decoder's stderr, decode
gapless to exactly the pushed sample count, equal this crate's own
decode within 2 LSB, and sit at zero lag against the input. The 19
split cells are all intensity stereo at the lower sampling
frequencies, where the two black-box decoders disagree with each other
(this crate equals one of them on every cell; at 48 / 44.1 kHz all
intensity cells are equal on both). The 8 configurations not offered
are the Model-2 block scheduler at the lower sampling frequencies (no
Annex D tables there). One black-box decoder states that
it does not implement mixed blocks at 8 kHz; those cells are judged on
the other.

**Never verified against an outside stream:** intensity-stereo and
mixed-block streams from a *foreign* encoder (the black-box encoders
here write neither; mixed blocks are covered by one staged fixture),
`scalefac_scale` / `preflag` beyond what the black-box encoder happens
to emit, and VBRI headers (no layout is staged — not implemented).

Defects this matrix found and this round fixed: an information frame
in a CRC-protected stream decoded as audio; free format above
320 kbit/s refused; frames with missing reservoir data dropped instead
of rendered silent; the gapless extension never found in real files
(read two bytes early); the encoder dropping the last frame of input,
overflowing the 12-bit `part2_3_length` at 8 kHz mono 64 kbit/s, and
spending a fifth of its bits on dense spectra; intensity stereo
leaving the band above the last scalefactor left-only, and using a
constant illegal position at the lower sampling frequencies.

## Speed

One thread, 60 s of 44.1 kHz stereo (pink noise + swept tone), CPU
time, median of three runs on a shared Apple-silicon host
(`examples/mp3_to_pcm.rs`, `examples/wav_to_mp3.rs`):

| | 128 kbit/s | 320 kbit/s |
|---|---|---|
| Decode | 0.13 s — 460x realtime | 0.16 s — 375x realtime |
| Encode, joint stereo, before this round | 5.21 s — 11.5x | 4.77 s — 12.6x |
| Encode, joint stereo, now | 1.38 s — 43x | 1.26 s — 48x |

The encoder gain is output-invariant (same bytes as before on these
clips at the commit that made it): the gain search starts at the first
gain a lower bound on the Huffman bit count allows, each trial
quantizes with one multiply per line from a cached 3/4 power, codebook
costing is branch-free, and `push_samples` no longer moves its whole
buffer once per frame. Plain stereo encodes at 53x, a quality preset
(outer loop) at 20x. The decoder spends 36 % of its time in the exact
64x32 `f64` synthesis matrixing; a fast transform there would change
the last bit of some samples and was not done. `BENCHMARKS.md` holds
the per-stage Criterion tables of earlier rounds.

## Encoder quality

Segmental critical-band SNR (`examples/band_snr.rs`: 1024-sample
frames, 25 critical bands, per-cell SNR clamped to -10..50 dB and
averaged — not a listening test) / plain SNR, in dB, 10 s clips at
44.1 kHz stereo, each decoded by a black-box decoder:

| Material | kbit/s | Black-box encoder | This encoder (joint) | Before this round |
|---|---|---|---|---|
| Two sawtooth tones + sine | 64 | 7.8 / 16.0 | 7.6 / 15.8 | 3.7 / 11.4 |
| | 128 | 11.5 / 19.0 | 12.3 / 19.9 | 3.7 / 11.4 |
| | 192 | 15.1 / 22.2 | 16.7 / 24.3 | 3.7 / 11.4 |
| Decaying noise bursts | 64 | 12.9 / 3.3 | 11.6 / 1.2 | 10.0 / 1.2 |
| | 128 | 16.2 / 5.7 | 12.9 / 3.4 | 11.6 / 3.4 |
| | 192 | 20.4 / 8.1 | 17.2 / 9.8 | 13.7 / 7.8 |
| Synthesised speech | 64 | 13.9 / 18.2 | 9.9 / 19.1 | 5.0 / 12.5 |
| | 128 | 21.6 / 25.6 | 18.9 / 28.9 | 7.1 / 13.2 |
| | 192 | 27.9 / 30.3 | 27.0 / 37.2 | 13.9 / 15.9 |

The "before" column is the rate-loop defect (the figure did not move
with the bitrate). What remains: on transients the default encoder
uses long blocks only — arming `enable_auto_block_type` raises the
noise-burst row at 128 kbit/s to 13.7 / 4.5 and is neutral on the
other two — and it does not move bits between frames by demand, which
is most of the remaining gap on bursts; on speech it maximises
waveform SNR (higher than the black-box encoder's) where a perceptual
allocation trades SNR for evenness across bands. The four quality
presets produce identical output on these clips at these bitrates.

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
  across frames. A frame whose back-reference points at data that was
  never received (cut head, seek, damaged splice) is rendered as
  silence and keeps its place on the timeline.
- **Intensity stereo** (§2.4.3.4.9.3, ISO/IEC 13818-3 §2.4.3.2): per
  window for short blocks; at the lower sampling frequencies the
  illegal position is the largest value of the band's `slen` bits. The
  lines above the last scalefactor band take the last transmitted
  position (see the module doc of `stereo` for what the standard
  leaves open there and what was measured).
- **CRC-16** frame protection (§2.4.3.1).

Output is planar `S16P` PCM (one little-endian `i16` plane per channel)
through the `oxideav_core::Decoder` trait. The
final float-to-`i16` conversion follows §2.4.3.4.7 exactly: the decoder
output range is `[-1.0, +1.0]`, so samples scale by the MSB weight
`2^15 = 32768` and round to the nearest integer with half-integer values
rounded away from zero (§2.3 "Nearest integer operator"). The decode
path is validated by PSNR self-round-trip tests and against staged
reference fixtures.

### Stream shape and mid-stream changes

`output_params()` reports the sample rate, channel count / layout and
`S16P` from the frame headers. Every MPEG audio frame carries its own
header, so a chained stream (`cat a.mp3 b.mp3`, an MPEG-1 ↔ MPEG-2 LSF
switch, mono ↔ stereo) can change that shape at any frame. The decoder
then returns one `Frame::StreamParams` event from `receive_frame` after
every frame of the old shape and immediately before the first frame of
the new one; its payload is the new `output_params()` and
`output_params_generation()` has already moved. A new shape starts a
fresh decode segment (bit reservoir, IMDCT overlap and synthesis memory
cleared), so each segment decodes exactly as the file would on its own.
The same holds after `reset()` and while draining after `flush()`.

A change is a different sampling frequency or channel layout: mono,
stereo / joint stereo (`Stereo`), or dual channel (`DiscreteN(2)`).
A bit-rate change (VBR), a free-format frame, or a `stereo` ↔
`joint_stereo` switch is not a change, because `output_params()` does
not describe it. The encoder rejects the event with `Unsupported`: its
rate and mode are fixed at construction.

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
  state machine (opt-in auto block typing). Mixed bursts are
  burst-coherent (r408): the flanking `Start` / `End` granules carry
  the §2.4.2.7 `mixed_block_flag` so the two lowest polyphase subbands
  stay normal-windowed across the whole burst and the §2.4.3.4
  low-subband overlap-add cancels exactly — verified float-perfect on
  four independent deployed black-box decoders at the MPEG-1 rates. At
  the LSF / MPEG-2.5 rates deployed decoders split 2-2 on that
  (conformant) wire combination, so the auto scheduler demotes mixed
  bursts to pure-short there; steady force-mixed streams are
  available at every rate (8 kHz included since r440).
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
  (`3·short_starts[3]` — 72 lines at 8 kHz). Mixed blocks **encode at
  8 kHz too** (r440): the window split is grounded strictly in the
  staged text chain — ISO/IEC 11172-3:1993 §2.4.2.7 fixes the mixed
  carve-out at "the frequency lines corresponding to the two lowest
  frequency polyphase subbands … while the remaining 30 subbands"
  (a fixed subband count, independent of any scalefactor table);
  ISO/IEC 13818-3:1997 §2.4.2.7 inherits that semantics verbatim
  ("with the exception of a different definition of
  scalefac_compress") and its §2.4.3.2 LSF-differences list never
  touches window geometry; and the staged low-rate-extension
  disclosure states the bit-stream structure is otherwise "unchanged
  with respect to the Standard ISO/IEC 13818-3" (only the sync word
  and the 8 kHz scalefactor bandwidth table differ). So the window
  split is 36 lines at every rate, while the coding split is the
  band-relative 72 the r408 observer probes found unanimous across
  all four deployed validators (wire lines 36..72 requantize with
  the transmitted long scalefactor bands 3..5 — the 8 kHz long
  table's six lowest bands span exactly `0..3·short_starts[3] =
  72`). Measured r440: three of the four deployed validators decode
  8 kHz force-mixed streams in the float-rounding regime (≤ 7e-6);
  the fourth long-windows the whole 72-line region — a minority
  reading contradicting the §2.4.2.7 subband-count text — and
  renders subbands 2..3 (250–500 Hz) of a mixed granule differently
  (nrmse ≈ 1.2e-3 with tone energy adjacent to the contested
  subbands, 4.1e-4 with it ≥ 2 subbands away). The **decoder** renders
  foreign 8 kHz mixed granules per the same spec/majority reading
  (long-coded to 72, two long-windowed subbands; before r408 lines
  36..72 were left silent). The mixed Huffman region-0 boundary is a
  deployed grey zone (validators split three ways at 8 kHz and 2-2
  at the LSF rates), so every mixed granule this encoder emits uses
  one codebook for both big-values regions — every boundary
  interpretation consumes identical bits. Encoder output at all
  three rates — long, short, mixed, auto block-type, tone and
  wideband noise, mono and stereo — decodes on the external
  validators in the float-rounding regime.

The encoder is reachable through the `oxideav_core::Encoder` trait and
several direct `make_encoder*` factory variants.

**Delay and gapless.** A sample entering the encoder leaves a decoder
1057 samples later (`END_TO_END_DELAY`). `finish` flushes that delay,
so the whole input is decodable, and `Mp3Encoder::enable_info_frame`
leads the stream with an Info / Xing frame — audio frame count, stream
byte count, 100-entry seek table and the gapless extension (delay 528,
exact end padding, both checksums) — from which readers recover exactly
the pushed samples (verified on two black-box decoders in every cell
of the matrix). The registry path (`make_encoder` packets through the
muxer) writes the Info / Xing frame without the gapless extension:
the delay and padding have no carrier between an `Encoder` and a
`Muxer` in the published `oxideav-core`.

**Buffering.** `Mp3Encoder` assembles frames as PCM arrives but writes
the stream at `finish` (the bit reservoir is scheduled over the whole
stream), and the registry encoder therefore returns its packets at
`flush`. Memory grows with the stream length.

## Demuxer

The `demuxer` module reads MPEG audio elementary streams of all three
layers (`.mp3`, `.mp2`, `.mp1`; MPEG-1, MPEG-2 LSF and MPEG-2.5
headers). The stream is announced under the codec of the layer its
frames carry:

| Frames | Codec id | Codec tag (`wFormatTag`) | Samples per frame |
|--------|----------|--------------------------|-------------------|
| Layer I | `mp1` | 0x0050 | 384 |
| Layer II | `mp2` | 0x0050 | 1152 |
| Layer III | `mp3` | 0x0055 | 1152 (MPEG-1), 576 (MPEG-2 LSF / MPEG-2.5) |

Only Layer III is decoded by this crate; the registry resolves `mp1` /
`mp2` to the decoders registered under those ids. Probe score, frame
count, duration, timestamps and seeking follow the layer; every
Layer I / II packet is a keyframe. `tests/layer_labels.rs` writes 36
Layer II files with a black-box encoder (six sampling frequencies ×
mono / stereo × three bit rates) plus six Layer III files, and checks
codec, frame count and duration of each against a black-box reader:
42 of 42 agree. No available tool writes Layer I, which is checked on
hand-built streams only.

**A change of layer mid-file is a change of codec and ends the
stream**: the packets, frame count, duration and seek range stop at
the first frame of another layer — its frames are never emitted under
the first layer's codec id — and `Mp3Demuxer::layer_change_offset()`
reports where it sits.

For Layer III it provides ID3v2
skip, frame iteration, and parsing of the `Xing` / `Info` VBR header
and the gapless-playback extension behind it (encoder delay / padding),
so `trimmed_duration_samples()` reports the gapless sample count —
equal to the encoder's input length on every tagged stream of the
black-box matrix. The information frame is recognised in CRC-protected
streams too (its magic stays at the offset tabulated from the frame
start). Registered as a container alongside the codec. The decoder
does not trim: it returns every frame, and the demuxer reports the
figures.

**Mid-stream header changes.** Within one layer a stream may switch sampling frequency,
MPEG version (MPEG-1 / MPEG-2 LSF / MPEG-2.5), channel mode, bitrate or
free-format slot count at any frame boundary — `cat a.mp3 b.mp3` does
exactly that, and also leaves the later files' ID3v2 tags, ID3v1
trailers and Xing / Info frames in the middle. Every audio frame comes
out once; tags and info frames are stepped over. A header exactly where
the previous frame ended that keeps the version / layer / sampling
frequency is trusted; one that switches them must be followed by a
header with its own key; a candidate found by scanning (junk, a damaged
splice) must chain three more correctly-spaced headers, so a resync
does not lock onto a false sync inside main data. A cut splice (the
first file's last frame truncated, the next file starting inside it) is
detected when nothing plausible sits where the frame should end but a
fully chained frame starts inside it: the cut frame is dropped. A
free-format segment
re-measures its slot count at a switch, bounded by the §2.4.2.3
free-format bitrate ceilings.

**Time base and duration.** The stream declares `1 / f`, `f` the first
audio frame's sampling frequency, so a stream that never switches gets
plain sample-count timestamps. Across a switch the running time is kept
exactly in units of 1 / 14 112 000 s (the least common multiple of the
nine MPEG audio sampling frequencies) and each packet's `pts` is that
time floored to the declared base: `pts + duration` is always the next
`pts`, timestamps strictly increase, and nothing drifts. `open()` walks
every frame header once, so `streams()[0].duration` is the exact sum of
the packet durations (`audio_frame_count()` and `framing_switch_count()`
report what the walk found). The decoder announces each new output
shape with `Frame::StreamParams`.

**Seeking** is frame-accurate, VBR or CBR, before or after any switch:
`seek_to` uses a sparse index from the open-time walk (every 32nd audio
frame) and steps to the frame containing the requested time — at or
before it — returning the PTS that frame is then emitted with.
`tests/demuxer_splice.rs` checks all 72 ordered pairs of sampling
frequencies, Layer III ↔ Layer II splices (the stream ends at the
layer change), Layer II rate switches, channel-mode /
bitrate / VBR / free-format switches, fully tagged files back to back,
junk between switching segments, cut splices, and seeks to every packet, against ffprobe frame counts, an independent header walk
and per-segment standalone decodes.
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
bitrate (`base · 8 · sample_rate / samples_per_frame`) is reported as
the stream's `bit_rate`. A single free-format frame (no second sync to
measure against) is rejected. End-to-end mono + stereo demux→decode is
byte-exact with the same stream demuxed as CBR
(`tests/demuxer_free_format_roundtrip.rs`).

## MPEG-2.5 decode

The `Decoder` trait wrapper accepts MPEG-1, MPEG-2 LSF, and all three
MPEG-2.5 extension rates. The 11.025 / 12 kHz rates both load the
in-repo ISO/IEC 13818-3 **16 kHz** LSF scalefactor-band table pair
(long + short) — the deployed de-facto mapping measured per spectral
line by the r405 observer-trace
(`mpeg2.5-scalefactor-bands-CORRIGENDUM-r405.md`, which refuted the
half-rate-sibling hypothesis of `mpeg2.5-scalefactor-bands.md`). The
**8 kHz** rate uses the distinct Fraunhofer 8 kHz table — its top long
bands collapse to width 2 at the 4 kHz Nyquist — transcribed into
`requantize::{LONG,SHORT}_STARTS_MPEG25_8` from the staged doc's "8 kHz,
long/short blocks" tables (confirmed verbatim per-line by the same
r405 trace; satisfies the Table-B.2 structural invariants: Σ = 576
long / 192 short, contiguous, 22/13 bands). All three rates decode
through the same chain as MPEG-2 LSF, byte-exact with the direct
decode chain. The 11.025 kHz path is validated end-to-end against the
staged `layer3-mpeg25-11025-32kbps` reference PCM
(`tests/mpeg25_reference_pcm.rs`): the decode locks at the canonical
1105-sample codec delay with steady-state normalized RMS error ≈ 1e-4,
and the production trait decoder reproduces the direct chain
byte-exact.

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

- **VBRI seek headers**: no layout is staged in the clean-room docs.
- **Gapless figures on the registry path** (see "Delay and gapless").
- **Streaming encode**: packets appear at `flush`.
- **Demand-driven bit reservoir** and a perceptual model by default:
  Annex D psychoacoustic shaping is opt-in under the presets; the
  bare-constructor quantization is rate / distortion driven.
- **Intensity stereo at the lower sampling frequencies** decodes
  identically on only one of the two black-box decoders in 19 of the
  matrix cells (those decoders differ from each other there).
- Layers I and II are framed by the demuxer and announced as `mp1` /
  `mp2`, but not decoded here, and the ISO/IEC 13818-3 multichannel extension
  is not decoded.

## Robustness

A `cargo-fuzz` harness covers both directions of the codec. Four
panic-freedom targets: `decode` (multi-packet decoder sessions with
crafted valid-sync headers across all three MPEG versions and
free-format framing, reservoir / reset / flush state transitions),
`granule` (the per-granule decode primitives, including the MPEG-2.5
low-rate scalefactor-band dispatch), `demux` (container open with
ID3v2 / Xing / LAME parsing, the open-time header walk, packet
iteration with resync, and the indexed seek path), and `encode` (hostile PCM geometries,
off-ladder/off-table configurations, and the full toggle surface —
VBR, CRC, Xing, forced/auto block types with non-finite thresholds,
joint-stereo variants — over both the trait path and the direct
stream API). A fifth target, `roundtrip`, asserts a semantic
invariant: every stream the encoder emits under a valid configuration
must frame-walk back losslessly, decode through this crate's own
decoder without error, and cover the pushed PCM sample count.

## License

MIT — see [LICENSE](./LICENSE).
