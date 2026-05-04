# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- ISO/IEC 11172-3 **Annex D Psychoacoustic Model 1** (Bark-partition
  spreading) for the VBR encoder, in a new `psy1` module. The model
  partitions the long-block MDCT spectrum across 24 Bark-axis
  partitions (Zwicker boundaries, mapped from each coefficient's
  centre frequency `(k + 0.5) * sr / 1152`), accumulates per-partition
  energy, and convolves the result with the Schroeder closed-form
  spreading function `sf(dz) = 15.81 + 7.5*(dz+0.474) - 17.5*sqrt(1 +
  (dz+0.474)^2)` (open-literature, no specific encoder consulted).
  Tonality is estimated via the spectral flatness measure
  `SFM = exp(mean(ln |X|^2)) / mean(|X|^2)` per partition and mapped
  to `[0, 1]` over the conventional 0..-60 dB SFM range; tonal
  partitions get the wider TMN ~14.5 dB SNR offset, noise partitions
  get the tighter NMT ~5.5 dB. The spread per-partition energy
  threshold is re-binned to the 22 long-block scalefactor bands as
  the per-sfb mask threshold, which feeds a new
  `encode_granule_vbr_psy1` outer loop that follows the spec's
  §C.1.5.4.4 iterate-until-stable-or-N-iter recommendation
  (`MAX_ITER = 8`). Engaged by default in VBR mode (new
  `psy_model = 1` option, default `1`); the previous lightweight
  per-sfb mask is still reachable via `psy_model = 0`. Psy-1 changes
  bit allocation to favour bands with no nearby tonal masker and
  lets neighbouring tones partially relax each other's mask — the
  spread mask plus the tone-vs-noise SNR split give the encoder
  meaningful spectrally-aware budget control without copying any
  external encoder's tuning table. New tests:
  `tests/encoder_psy1.rs` (8 cases — default-model selection, VBR
  round-trip across both models, ffmpeg cross-decode, multi-tone
  per-partial SNR, tonality-driven byte delta, Psy-1 ≠ simple-model
  byte regression guard, MPEG-2 LSF round-trip, invalid-option
  rejection). `psy1::tests` adds 11 unit cases (Bark-axis
  monotonicity, partition assignment, spreading-function shape +
  asymmetry, silence/tone/noise tonality limits, SMR sign across a
  hot vs silent band, neighbour-threshold spreading, quality-knob
  monotonicity, finer-quantizer NMR sanity, partition count).

## [0.0.6](https://github.com/OxideAV/oxideav-mp3/compare/v0.0.5...v0.0.6) - 2026-05-02

### Other

- migrate to centralized OxideAV/.github reusable workflows
- add MPEG-2 LSF intensity-stereo encoding
- add intensity-stereo encoding per ISO 11172-3 §2.4.3.4.10.2
- add short-block window switching on transients
- add joint-stereo (M/S) encoding per ISO 11172-3 §2.4.3.4.10
- add VBR mode with per-band masking model
- adopt slim VideoFrame/AudioFrame shape
- pin release-plz to patch-only bumps

### Added

- Intensity-stereo (IS) encoding for MPEG-2 LSF (Low Sampling Frequency)
  Layer III long-block stereo granules per ISO/IEC 13818-3 §2.4.2.7 and
  §2.4.3.4.10.2. Extends the MPEG-1 IS picker / rewriter to the LSF
  variant: the per-band `is_pos` is now a 5-bit field (0..=30 with 31 =
  "not IS-coded" sentinel) selected from the geometric ratio table at
  `intensity_scale = 0` (mirror of `is_factors_mpeg2(.., 0)` — step
  0.5^n where `n = (is_pos+1)/2`). The same safety-floor / coherence /
  pan-imbalance rules from the MPEG-1 path drive bound selection. The
  L spectrum is rewritten to the louder channel's waveform per band
  (or the (L+R)/2 average when `is_pos = 0`), R is forced to zero so
  `find_is_bound_sfb` recovers the same bound. R-channel scalefactors
  pack into the IS half of `SCF_PARTITIONS_MPEG2[0][16..28]` via a
  fixed `scalefac_compress_9 = 358` ⇒ slen = [4, 5, 5, 0],
  `nr_of_sfb = [7, 7, 7, 0]`, total 7×4 + 7×5 + 7×5 = 98 scalefactor
  bits per IS-active R channel. The slen-4 group covers sfb 0..6 (kept
  out of IS by the encoder's safety floor at sfb 7), so its lower
  resolution is harmless. Frame-level `mode_extension` semantics are
  unchanged from the MPEG-1 path (bit 0x1 = IS on). Opt out with
  `intensity_stereo=0`. On a 22.05 kHz stereo fixture with uncorrelated
  LF and a 4-9 kHz correlated HF tail this saves ~27.9% bytes in VBR
  mode versus IS-disabled (well above the 5-8% target); ffmpeg
  cross-decode is clean. Short / start / stop / mixed LSF blocks stay
  out of IS. New tests: `tests/encoder_intensity_stereo_mpeg2.rs`
  (5 cases — header bits, opt-out, VBR byte delta, own-decoder
  round-trip, ffmpeg cross-decode).
- Intensity-stereo (IS) encoding per ISO/IEC 11172-3 §2.4.3.4.10.2 for
  MPEG-1 long-block stereo granules. The encoder picks a per-granule
  scalefactor-band bound by walking sfbs from the top down: a band
  qualifies for IS when its combined L/R energy is below a 40-dB-down
  noise floor relative to the loudest band, when the absolute L/R
  cross-correlation `|<L,R>| / sqrt(eL * eR) >= 0.85`, or when the
  pan imbalance `|eL - eR| / (eL + eR) >= 0.85` (hard-pan shortcut).
  The bound is clamped to a sfb 7 floor so the audible bands stay in
  raw L/R / MS coding. For each IS-coded band the per-channel
  `is_pos ∈ {0..=6}` is chosen by best fit on the `er / (eL + eR)`
  energy fraction; sentinel `is_pos = 7` covers silent bands. The L
  spectrum is rewritten to a magnitude-preserving surrogate
  `sign(L) * sqrt(L^2 + R^2)` and R is forced to zero so the decoder
  side's `find_is_bound_sfb` recovers the same bound. R-channel
  scalefactors are emitted with `scalefac_compress = 13`
  (slen1 = slen2 = 3) so all 21 long-block sfbs carry the 3-bit
  `is_pos`. Frame-level `mode_extension` bit 0x1 flips on whenever
  any granule uses IS — combined with #115 MS the wire pattern can
  be 0b01 (IS only) or 0b11 (MS + IS). Opt out with
  `intensity_stereo=0`. On a stereo fixture with uncorrelated LF and
  a correlated HF tail this saves ~18.8% bytes in VBR mode at the
  same quality knob; CBR keeps slot sizes fixed and spends the freed
  bits on lower noise. Short / start / stop blocks and MPEG-2 LSF
  stay on the pre-#174 path — they have separate sf partitions that
  the encoder doesn't yet emit. New tests:
  `tests/encoder_intensity_stereo.rs` (7 cases — header bits, opt-out,
  VBR byte delta, own-decoder round-trip, ffmpeg cross-decode,
  MS-on-with-IS regression check, short-block-on-with-IS regression
  check). Lib tests gain 4 cases for `pick_is_bound_long` and
  `apply_is_rewrite_long` covering the safety floor, the silence
  shortcut, the coherence break, and the R-zeroing invariant.
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
