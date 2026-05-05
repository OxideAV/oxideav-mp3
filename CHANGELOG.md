# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1](https://github.com/OxideAV/oxideav-mp3/compare/v0.1.0...v0.1.1) - 2026-05-05

### Other

- install ffmpeg via reusable workflow's extra_packages_* inputs
- add mixed-block window-switching + simple-mask FFT lift

## [0.0.7](https://github.com/OxideAV/oxideav-mp3/compare/v0.0.6...v0.0.7) - 2026-05-05

### Other

- add Annex D §D.2.4.1 FFT pre-analysis + per-window subblock_gain
- fix short-block analyzer to walk encoder's sfb-major layout
- add per-window Psy-1 short-block path + peak-detection tonality
- add ISO/IEC 11172-3 Annex D Psy Model 1 (Bark-spread + tonality)
- wire docs/audio/mp3/fixtures/ corpus into integration suite
- replace never-match regex with semver_check = false

### Added

- **Mixed-block window-switching encoder path.** ISO/IEC 11172-3
  §2.4.2.2 lets a short-block granule keep its low-frequency prefix
  (sfb 0..=7 of the long table — 36 coefficients = subbands 0..2 at
  44.1 kHz) coded as a long block while the high-frequency tail
  (sfb 3..=12 of the short table × 3 windows) gets the short-block
  3 × 12-point MDCT. This bridges a sustained low-frequency tone
  (which would lose spectral compaction under pure-short coding) with
  a transient riding on top — typical of a drum hit on a sustained
  bass note. Five wired-up pieces:
  * `mdct::mdct_granule_full` plus `mdct::unreorder_short_mixed_inplace`
    drive the per-subband MDCT dispatch (subbands 0..2 use the 36-pt
    long MDCT with normal sine window; subbands 2..32 use the
    3 × 12-pt short MDCT) and the post-MDCT reorder for the short tail
    only — exact inverse of `requantize::reorder_short`'s mixed
    branch in the decoder.
  * `psy1::Psy1Mask::analyze_mixed` runs the standard long-block
    Bark-partition spreader on `xr[0..36]` and the per-window
    192-coefficient short-block spreader on the high tail; output
    mask carries 38 entries (8 long + 10 short × 3 windows). The
    encoder's noise allocator walks all 38 and budgets bits for the
    worst (sfb, window) pair across both the long prefix and the
    short tail.
  * `block_type::should_use_mixed_block` decides per-(granule,
    channel) when to flip the flag: 3-tap linear-phase LP/HP
    filter pair splits the granule's PCM at fs/4, then per-sub-frame
    energy peak-to-mean (LF) and peak-to-min (HF) ratios across the
    three sub-frames separate "sustained LF + impulse HF" from
    uniform-spectrum transients. Conservative thresholds
    (HF ratio > 8×, LF ratio < 2×) — the picker prefers pure-short
    over a misfire (long-window LF region under a wrong mixed call
    leaves the LF transient-smeared, worse than pure-short).
  * `encoder::quantize_and_encode_full` plus
    `encoder::build_mixed_coeff_window_map` apply the per-window
    quantizer scale only to the short tail; the long-prefix
    coefficients use the granule's `global_gain` directly (per
    decoder's `requantize.rs` `long_sfb_count` branch — long-prefix
    sfbs don't carry `subblock_gain`).
  * `encoder::GranuleEncoded` gains a `mixed_block_flag` field; the
    MPEG-1 and MPEG-2 LSF window-switching side-info tail emit
    writes the actual flag (previously hard-coded to 0).
  Six new tests (3 unit cases in `block_type::tests` for the mixed
  picker; 3 integration cases in `tests/encoder_short_blocks.rs`
  including a hard-asserted ffmpeg cross-decode):
  `mixed_picker_silence_returns_false`,
  `mixed_picker_pure_burst_returns_false`,
  `mixed_picker_sustained_low_with_burst_returns_true`,
  `mixed_fixture_engages_mixed_blocks`,
  `mixed_blocks_roundtrip_decode_cleanly`,
  `mixed_blocks_decode_via_ffmpeg`. The ffmpeg cross-decode asserts
  the mixed-block bitstream layout (long-prefix scalefactors at
  sfb 0..=7 of the long table + short-tail scalefactors at sfb 3..=12
  × 3 windows of the short table) round-trips cleanly through ffmpeg
  (no warnings, no garbage samples; ffmpeg's decoded energy matches
  our own decoder within 1× on the bass+burst fixture).
- **FFT pre-analysis lift on the simple-mask VBR path.** Previously
  the 1024-point FFT pre-analysis (Annex D §D.2.4.1) was only
  consumed by `psy_model = 1`. New `psy::GranuleMask::analyze_with_fft`
  applies a 3 dB tonality lift to per-sfb mask thresholds wherever
  the FFT spotted a between-bin tone — partitions whose SFM-derived
  tonality exceeds zero get the affected sfb's threshold tightened
  by `10^(-3·t/10)`, where `t ∈ [0, 1]` is the partition tonality.
  The encoder dispatches long-block granules through the new
  function under `psy_model = 0` (short / start / stop stay on the
  base mask — the FFT's 1024-pt span matches the long-block 576-coeff
  granule, not the short-block 192 window). The lift is conservative
  (3 dB vs the full Annex D 9 dB delta) because the simple model
  doesn't have a Bark spreader to re-balance the lift against
  neighbouring noise partitions. Two new unit cases in `psy::tests`:
  `fft_lift_tightens_threshold_on_tonal_input` (single-bin tone in
  the FFT spectrum tightens at least one sfb threshold);
  `fft_lift_silent_fft_matches_base` (silent FFT input leaves every
  sfb threshold unchanged — graceful degradation for silent-prefix
  granules).
- **FFT-domain pre-analysis for Psy-1 long blocks.** New `fft.rs`
  module with a clean-room 1024-point radix-2 Cooley-Tukey FFT plan
  + Hann window builder. The encoder maintains a per-channel rolling
  448-sample PCM history so each long-block granule's FFT input is
  `[history (448), current granule (576)]` Hann-windowed; the
  resulting one-sided power spectrum (513 bins) feeds a parallel
  Bark-partition spreader (`fft_partition_pass` + `build_fft_partition`
  in `psy1`) and the per-partition tonality estimator. The new
  `Psy1Mask::analyze_with_fft` constructor fuses the FFT-domain
  tonality into the MDCT-domain pass via `max(mdct_tonality,
  fft_tonality)` and re-runs the Schroeder spreader with the boosted
  tonality — partitions where the FFT spotted a between-bin tone the
  MDCT smeared get the tonal SNR offset (TMN ~14.5 dB) instead of
  the noise offset (NMT ~5.5 dB), tightening the per-band threshold
  by ~9 dB. ISO/IEC 11172-3 Annex D §D.2.4.1 spec-mandates this
  parallel FFT path; previously the encoder ran tonality detection
  on MDCT output only, which catches in-bin tones cleanly but
  smears tones falling between two MDCT coefficients (the MDCT
  projects onto cosines that aren't a strict tone basis). Long /
  start / stop blocks dispatch through `analyze_with_fft`; short
  blocks stay on the per-window 192-coefficient path
  (`analyze_short`) — the 1024-point FFT spans the long-block
  granule's 576 coefficients, not the short window's 192. Five new
  unit cases (`fft_partition_assigns_increasing`,
  `psy1_with_fft_silence_has_floor_thresholds`,
  `psy1_with_fft_matches_mdct_only_when_fft_is_silent`,
  `psy1_with_fft_tightens_threshold_on_tonal_partition`,
  `psy1_with_fft_quality_knob_monotonic`) plus four FFT-only cases
  (`fft_dc_input_concentrates_energy_at_bin_zero`,
  `fft_pure_tone_concentrates_at_correct_bin`,
  `fft_parseval_holds_for_random_input`,
  `fft_two_close_tones_are_resolved`,
  `hann_endpoints_zero_centre_one`). Two integration cases in
  `tests/encoder_psy1.rs` cover the wired-up encoder path:
  `psy1_fft_preanalysis_handles_between_bin_tone` (1015 Hz tone
  deliberately mid-bin for the long-block 38.28 Hz MDCT grid;
  asserts SNR > 5 and no regression vs the simple-mask baseline)
  and `psy1_fft_preanalysis_long_input_stable` (4-second
  multi-tone fixture exercises the rolling history across many
  roll-forward cycles, asserts non-silent + finite output).
- **Per-window subblock_gain for short blocks.** The encoder now
  picks a per-window `subblock_gain[w]` triple from the Psy-1
  short-block per-window energies. Per ISO/IEC 11172-3 §2.4.3.4,
  `subblock_gain[w]` attenuates the dequantized coefficients of
  window `w` by `2^(-2 * sbgain)` — the encoder side compensates by
  scaling its quantizer step for that window by `2^(1.5 * sbgain)`
  (the `3/4` factor falls out of the encoder's `xr^(3/4)` mapping),
  giving the loud (post-attack) window more dynamic range so its
  high-magnitude transient coefficients fit inside the Huffman
  tables without coarsening the quieter sibling windows. Heuristic
  picker (`pick_subblock_gain_short`) maps the per-window energy
  ratio to `subblock_gain` with a `0.25 * log2(E[w] / E_min)` slope
  (clamped to 0..7) — quieter windows stay at `0`; the loudest
  window gets up to `7` units of dynamic-range extension.
  `quantize_and_encode_full` plus `build_short_coeff_window_map`
  apply the per-window scale based on the encoder's pure-short-
  block sfb-major then window-major-within-sfb layout. The MPEG-1
  and MPEG-2 LSF window-switching side-info tails now emit the
  chosen `subblock_gain[w]` triple instead of `[0; 3]` (3 bits per
  window); the previously hard-coded zeros are gone. Six new
  encoder unit cases:
  `subblock_gain_zero_for_equal_window_energies`,
  `subblock_gain_biases_loud_window_higher`,
  `subblock_gain_clamps_to_seven`,
  `subblock_gain_silence_returns_zeros`,
  `subblock_gain_long_block_mask_returns_zeros`,
  `quantize_short_with_subblock_gain_uses_more_bits_for_loud_window`,
  `build_short_window_map_matches_layout`. The existing
  short-block ffmpeg cross-decode test
  (`psy1_short_block_ffmpeg_cross_decode_castanet`) confirms ffmpeg
  accepts the bitstream with non-zero subblock_gain on the per-
  channel transient frames. Pre-echo PSNR delta on the isolated-
  transient fixture rose to >245 dB (vs the long-only baseline)
  through the combination of short-block window switching + the
  new subblock_gain pre-emphasis.
- **Short-block path for Psy-1.** New `Psy1Mask::analyze_short`
  constructor runs the Bark-partition spreader independently on each
  of the three 192-coefficient short-block windows packed into a
  granule (sfb-window-major: window 0 = `xr[0..192]`, window 1 =
  `xr[192..384]`, window 2 = `xr[384..576]`). Per-window centre
  frequency is `(k + 0.5) * sr / 384` (twice the long-block bin
  width); each window builds its own per-Bark-partition energy +
  log-energy accumulator, runs the same Schroeder spreader, applies
  tone-vs-noise SNR offsets, and re-bins to that window's 13 short-
  block sfbs. The output mask carries 39 entries (3 × 13) and the
  encoder's `worst_nmr_db` walks all of them — so a short-block
  granule whose burst energy lives in window 1 only allocates bits
  to that window's sfbs without smearing the threshold across the
  silent sibling windows. The encoder dispatches based on
  `BlockType::Short` so long / start / stop granules keep using the
  long-block 576-coefficient path. Most impactful on transient
  content (drums, plosives, attacks) — confirmed by an ffmpeg
  cross-decode + own-decoder round-trip on a castanet-style
  fixture, both clean. The Psy-1 long → short bitstreams differ on
  transient input (regression guard via `assert_ne!` on the encoded
  bytes); on this castanet fixture the per-decoder energy ratio
  agrees within 50 % between ffmpeg and our own decoder, confirming
  the on-wire window-switching layout is correct end-to-end.
- **Peak-detection tonality estimator.** `Psy1Mask` now carries a
  `partition_peak_tonality` field alongside the existing SFM-based
  `partition_tonality`. The peak detector walks each partition's
  squared-magnitude spectrum, finds local maxima
  (`|X[k]|^2 > |X[k-1]|^2` and `> |X[k+1]|^2`), and computes
  `peak_ratio = max_peak_energy / mean_partition_energy`. A pure
  tone gives `peak_ratio ≈ partition_count` (very large); white
  noise gives `peak_ratio ≈ 1`. Mapped to `[0, 1]` over `[0, 20]
  dB`. The combined per-partition tonality used by the spreader is
  now `t = max(t_sfm, t_peak)` so a partition flagged as tonal by
  *either* estimator gets the wider TMN ~14.5 dB SNR budget. This
  matches the Annex D.2.4.4 wording on local-maximum detection
  more closely than the previous SFM-only estimate (which is
  open-literature equivalent but doesn't model the spec's
  peak-finder). Two new unit cases:
  `psy1_peak_tonality_high_for_pure_tone` and
  `psy1_peak_tonality_low_for_flat_partition`.
- **Refactor:** `Psy1Mask::energy / threshold / width / start` are
  now `Vec<f32>` / `Vec<u16>` (length 22 for long, 39 for short)
  rather than `[_; 22]`. The encoder consumes them via `.iter()` /
  `.len()` so the change is API-compatible at the call site; the
  `worst_nmr_db` / `estimate_noise` / `smr_db` helpers iterate up
  to `n_sfb` (new field). Three new integration tests in
  `tests/encoder_psy1.rs` cover the short-block path:
  `psy1_short_block_path_differs_from_long_only` (regression guard
  vs long-only Psy-1), `psy1_short_block_ffmpeg_cross_decode_castanet`
  (ffmpeg interop on transient content, ffmpeg-vs-own-decoder
  energy parity check), and `psy1_short_block_own_decode_roundtrip_castanet`
  (own-decoder finite-output sanity). Three new `psy1::tests`
  cases cover the short-block analyzer
  (`coeff_partition_short_assigns_increasing`,
  `psy1_short_silence_has_floor_thresholds`,
  `psy1_short_finer_quantizer_means_lower_nmr`,
  `psy1_short_independent_per_window`).
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
