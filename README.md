# oxideav-mp3

A pure-Rust **MPEG-1 / MPEG-2 Audio Layer III** (MP3) codec for the
[oxideav](https://github.com/OxideAV/oxideav-workspace) framework —
decoder, CBR/VBR encoder, and a stream demuxer.

## Status

Clean-room implementation. Every numeric constant is transcribed from
ISO/IEC 11172-3:1993 and ISO/IEC 13818-3:1997 and from no other source.
The decoder produces PCM end-to-end for MPEG-1 and MPEG-2 LSF (mono and
stereo); MPEG-2.5 framing is parsed but full decode through the trait
wrapper is pending. The encoder produces valid CBR and VBR MP3 streams.

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
decode path is validated by PSNR self-round-trip tests and against
staged reference fixtures.

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
- MPEG-2.5 frame header writing and sample-rate dispatch.

The encoder is reachable through the `oxideav_core::Encoder` trait and
several direct `make_encoder*` factory variants.

## Demuxer

The `demuxer` module provides a streaming MP3 container reader: ID3v2
skip, frame iteration, and parsing of the `Xing` / `Info` VBR header
and the trailing gapless-playback extension (encoder delay / padding),
so `trimmed_duration_samples()` reports the gapless sample count.
Registered as a container alongside the codec.

## Not yet supported

- MPEG-2.5 decode through the `Decoder` trait wrapper (the header guard
  accepts MPEG-1 and MPEG-2 LSF and rejects MPEG-2.5), pending the
  low-rate (8 / 11.025 / 12 kHz) scalefactor-band and Huffman-table
  mappings.
- A full Annex D psychoacoustic model driving encoder quality (the
  current quantization is rate/distortion-driven).

## Robustness

A `cargo-fuzz` harness (`decode` and `granule` targets) drives
attacker-controlled bytes through the decode surface for panic-freedom.

## License

MIT — see [LICENSE](./LICENSE).
