# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Short-block window switching on transients per ISO/IEC 11172-3
  §2.4.2.2 / Annex C. A new `block_type` module hosts the
  `TransientDetector` (per-channel energy-ratio detector that splits
  every 576-sample granule into three 192-sample sub-frames and
  flags any sub-frame whose energy exceeds the smoothed long-term
  average by more than 4×) and the `BlockTypeMachine` (per-channel
  state machine that bridges the legal long → start → short → stop
  sequence using one granule of PCM lookahead). The MDCT now
  branches on block type: long / start / stop drive the existing
  36-point long MDCT with the type's window envelope; short uses
  three 12-point MDCTs per subband, followed by the inverse-reorder
  step that mirrors the decoder's `reorder_short`. Side-info emit
  switches to the window-switching layout (2 × `table_select` + 3 ×
  `subblock_gain`, no region counts) for non-long granules. On an
  isolated 50 ms transient at 44.1 kHz / 192 kbps the pre-onset
  PSNR jumps from 54 dB (long-only) to >120 dB with short blocks
  engaged. Opt out with `short_blocks=0`. New tests: 8 unit tests
  in `block_type::tests`, 2 mdct unit tests, and 5 end-to-end tests
  in `tests/encoder_short_blocks.rs` covering block-type detection,
  steady-tone false-positive rejection, isolated-transient pre-echo
  PSNR delta, ffmpeg cross-decode, and our-decoder round-trip
  energy parity.
- VBR (variable bit-rate) encoding mode. Opt in by setting
  `CodecParameters::options` to `CodecOptions::new().set("vbr_quality",
  "0".."9")` (0 = highest quality / largest files, 9 = smallest). A
  new `psy` module computes a per-scalefactor-band energy + masking
  estimate inspired by ISO 11172-3 §C.1 (psychoacoustic model 1) and
  drives the per-granule global-gain selection so the worst-band
  noise-to-mask ratio stays under the quality target. The per-frame
  bitrate slot is then chosen from the standard table to fit the
  resulting main-data byte count, yielding files that shrink for
  silent / pure-tone content and grow for spectrally-rich content at
  the same quality knob.
- 8 new VBR end-to-end tests (`tests/encoder_vbr.rs`) plus 5 unit
  tests in `psy::tests`.
- Joint-stereo (M/S) encoding per ISO/IEC 11172-3 §2.4.3.4.10. The
  encoder picks per-frame between M/S and dual-channel from the
  side-vs-mid energy ratio across both granules — when the side
  channel carries less than 30% of the mid energy (correlated stereo:
  centred voice + ambient, mono fold-down) the granules are rotated
  into `M = (L+R)/sqrt(2)`, `S = (L-R)/sqrt(2)`, the header switches
  to `mode = 0b01` (joint stereo) and `mode_extension = 0b10` (M/S
  on, IS off). Anti-correlated and true-stereo content stays in
  dual-channel. Opt out with `joint_stereo=0`. On a centred-voice
  stereo fixture this saves ~10% bytes in VBR mode at the same
  quality knob; CBR mode keeps slot sizes fixed and spends the
  freed bits on lower noise instead. New tests:
  `tests/encoder_joint_stereo.rs` (6 cases — header bits, byte-size
  delta, anti-correlation rejection, own-decoder round-trip, ffmpeg
  cross-decode, opt-out).

## [0.0.5](https://github.com/OxideAV/oxideav-mp3/compare/v0.0.4...v0.0.5) - 2026-04-25

### Other

- drop oxideav-codec/oxideav-container shims, import from oxideav-core
- implement MPEG-2 / MPEG-2.5 intensity stereo
- add NOTICES for minimp3 references
- add ffmpeg-based intensity-stereo end-to-end test
- implement MPEG-1 intensity stereo decode
- enable MPEG-2.5 decode (8 / 11.025 / 12 kHz)
- bump oxideav-container dep to "0.1"
- drop Cargo.lock — this crate is a library
- bump oxideav-core / oxideav-codec dep examples to "0.1"
- bump to oxideav-core 0.1.1 + codec 0.1.1
- migrate register() to CodecInfo builder
- bump oxideav-core + oxideav-codec deps to "0.1"
- thread &dyn CodecResolver through open()

## [0.0.4](https://github.com/OxideAV/oxideav-mp3/compare/v0.0.3...v0.0.4) - 2026-04-19

### Other

- claim WAVEFORMATEX tag via oxideav-codec CodecTag registry
- claim WAVEFORMATEX tag via oxideav-codec CodecTag registry
- bump oxideav-container + oxideav-id3 to 0.0.4
- drop duplicate oxideav-codec entry
- bump oxideav-core to 0.0.5
- migrate to oxideav_core::bits shared BitReader / BitWriter
- refresh README + module docstrings to match shipped decoder/encoder
- add MPEG-2 LSF encoder — 16/22.05/24 kHz, 576-sample single-granule frames
