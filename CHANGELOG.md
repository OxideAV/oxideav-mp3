# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the crate adheres
to [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- analysis / mdct: **precomputed kernel tables + vectorizable
  interchanged loops for the encoder front-end DSP** (r409, bench
  axis). The analysis filterbank re-evaluated the §C.1.3 matrixing
  cosine `M[i,k] = cos((2i+1)(k−16)π/64)` inline — 2048 `cos()` calls
  per 32-sample row — and the forward MDCT re-evaluated its
  §2.4.3.4.10.2 kernel cosine per (bin, sample) pair; both kernels are
  now precomputed once (identical expressions, identical `f64` bit
  patterns) and the matrixing / partial-calculation / transform loops
  run with the reduction index outermost, so every output accumulates
  its products in the identical ascending order (bit-exact per output)
  over consecutive vectorizable lanes. The analysis window functions
  get the same table treatment (the Start/End short half-window
  arguments are exactly `analysis_short_window(i−18)` / `(i−6)`).
  Pinned by `analyze_row_interchanged_matrixing_matches_reference`
  (200 streamed rows vs the inline-cosine per-output reference,
  outputs and `X[]` state compared by bit pattern) and
  `mdct_fast_paths_match_generic_form` (n = 36 / 12 table paths vs the
  retained generic inline-cosine form, plus window-table bit checks).
  Encoded streams are byte-identical. `encode_stage_filterbank` drops
  95.6% (3.82 ms → 168 µs) and `encode_stage_mdct_long` 89.6%
  (1.86 ms → 193 µs). Cumulative r409 whole-stream **encode**: tone
  −90%, noise −69%, sweep −87%, mixed stereo −76% (mono 44.1 kHz tone
  now ≈ 3.3 ms per 0.5 s clip ≈ 150× real-time); whole-stream
  **decode** is −36…−39% from the synth/imdct work (≈ 125× real-time
  mono at 44.1 kHz).

### Changed

- imdct: **vectorizable k-outer transforms** (r409, bench axis). The
  stack IMDCTs (`n = 36` long, `n = 12` short) ran one output at a time
  (dependent-chain sums); they now run with `k` outermost over
  transposed cosine tables, so each output accumulator receives its
  products in the identical ascending-`k` order (bit-exact per output)
  while the inner loop walks 36 / 12 consecutive coefficients with a
  broadcast input line — vectorizable across the independent output
  lanes. The public `imdct()` keeps the straightforward per-output form
  and serves as the reference: `interchanged_imdct_matches_reference`
  pins both stack transforms against it by bit pattern over random
  wide-dynamic-range inputs. Decoded PCM is bit-for-bit unchanged;
  `stage_imdct` drops 39% (137.2 µs → 84.0 µs per 40-granule batch).

### Changed

- synth: **vectorizable matrixing + windowed sum, and a ring-buffer
  shift register** (r409, bench axis). The Figure A.2 matrixing ran one
  output at a time (64 independent 32-term dependent-chain sums); it now
  runs with `k` outermost over a transposed coefficient table, so each
  accumulator still receives its products in the identical ascending-`k`
  order (bit-exact per output) while the inner loop walks 64 consecutive
  coefficients with a broadcast `S[k]` — a form the compiler vectorizes.
  The 512-tap windowed sum is interchanged the same way (`i` outer, 32
  consecutive `U`/`D` lanes inner, identical per-output order). The
  "Shifting" step no longer moves 960 `f64`s per row: `V[]` is a
  power-of-two ring and the shift is a 64-slot rotation of the origin
  (`V[i]` ↦ `v[(pos+i) & 1023]`), with the `U` build reading 32-value
  runs that stay wrap-free by 32-alignment. Pure data-movement /
  summation-order-preserving changes — decoded PCM and the `V[]`
  history are bit-for-bit unchanged, pinned by
  `synth_row_interchanged_matrixing_matches_reference` (200 streamed
  rows against the straightforward per-output reference, comparing
  every output and every `V[]` slot by bit pattern). `stage_synth`
  drops 50% (321.9 µs → 161.2 µs per 40-granule batch).

### Changed

- huffman: **single-pass region costing for the encoder's codebook
  chooser** (r409, bench axis). `choose_best_table_for_region` costed
  the region once per selectable codebook (30 passes, each re-deriving
  magnitudes and re-resolving the table). The §C.1.5.4.4.8 per-pair cost
  decomposes as `codeword_len + esc·linbits + signs`, where the ESC
  condition (`|v| ≥ 15`) and sign count are codebook-independent, so one
  pass over the pairs now accumulates every codebook's codeword-length
  sum at once through a precomputed 256-cell × 30-table length LUT
  (`u8::MAX` marking not-codable cells), then adds the shared ESC / sign
  counts scaled per table. Same integers, same ascending-index
  tie-break, same reach filter and `None` semantics.
  `choose_best_count1_table` similarly accumulates both quad tables in
  one pass, and `partition_split` finds the last non-zero line with a
  backward scan. All pinned by
  `single_pass_chooser_matches_per_table_reference` and
  `partition_split_backward_scan_matches_forward_reference` against the
  straightforward per-table / forward-scan references. Encoded streams
  are byte-identical; `encode_stage_inner_loop` drops a further ~18%
  (15.6 ms → 12.8 ms per batch). Cumulative r409 whole-stream encode:
  tone −75%, noise −56%, sweep −71%, mixed stereo −63%
  (`encode` bench, direct path).

### Changed

- inner_loop: **the budget gain scan skips provably-uncodable gains via
  per-band probes** (r409, bench axis). `search_bit_budget` /
  `search_bit_budget_band_aligned` walk `global_gain` upward from 0 (the
  spec's `qquant + 1` step); at loud gains every candidate used to pay a
  full 576-line quantize plus the Huffman table search only for the
  count to come back `None` (a line above the 8206 linbits-13 codebook
  ceiling is uncodable by every Table 3-B.7 codebook). The scan now
  collects one probe per constant-factor quantizer group (the band's
  largest-|xr| line plus its gain-independent `2^(-mult*scalefac)`
  term, mirroring `quantize`'s long / short / mixed group structure)
  and, per gain, re-runs the quantizer's own `quantize_line` on the
  probes: if any probe magnitude exceeds the max selectable codebook
  reach, that line IS one of the quantized lines (identical expressions
  on identical inputs — an exact proof, not an estimate), the exact
  count would be `None`, and the full evaluation is skipped. A probe
  miss just falls through to the full path, so the scan returns the
  identical first-satisfying gain. Pinned by
  `search_bit_budget_matches_straightforward_scan` (production search
  vs a no-skip reference scan over random spectra, block types,
  budgets, rates) and `probe_skip_only_fires_on_truly_uncodable_gains`
  (every skipped gain re-checked to be truly uncodable). Encoded
  streams are byte-identical; `encode_stage_inner_loop` drops a further
  ~49% (30.7 ms → 15.6 ms per 38-granule batch; cumulative −63% from
  the r409 baseline).

### Changed

- quantize: **the §2.4.3.4.7.1 gain factor is hoisted to one evaluation
  per scalefactor band** (r409, bench axis). The long-range quantizer
  previously re-derived `sf_term = 2^(−mult·scalefac)` and
  `factor = gain · sf_term` for every one of the 576 lines even though
  both depend only on the band index; the loop now walks the Table 3-B.8
  band boundaries and computes the factor once per band — the identical
  expression on the identical inputs, so the quantized `is[]` is
  bit-for-bit unchanged (pinned by
  `quantize_hoisted_band_factor_matches_per_line`, which compares the
  hoisted loop against the straightforward per-line evaluation across
  all nine sample rates, random scalefactor/preflag/scalefac_scale
  configurations, the full gain range, and sub-range invocations).
  Encoded streams are byte-identical; the inner rate loop — which calls
  the quantizer once per candidate `global_gain` — drops ~27% on the
  `encode_stage_inner_loop` criterion bench (41.9 ms → 30.7 ms per
  38-granule batch on the development host).

### Changed

- huffman: **big-values decode now uses an O(1) canonical-prefix table**.
  The big-values matcher previously scanned every codebook entry for each
  candidate length, walking the bitstream one bit at a time
  (O(entries × max_len) per pair). It now peeks the codebook's longest
  codeword width once and indexes a direct-mapped table that records
  `(x, y, len)` for the codeword covering that prefix, then consumes
  exactly the codeword's length — the standard canonical-prefix decode.
  Because the Table 3-B.7 codebooks are prefix-free, every prefix beginning
  with codeword `c` of length `len` lands in the `2^(maxlen−len)`
  contiguous slots whose top `len` bits are `c`, so the `(x, y)` returned
  and the bits consumed are bit-for-bit identical to the former scan; a
  slot no codeword covers reproduces the same `InvalidCode`. The per-table
  prefix tables are built once via a `LazyLock`. A new
  `fast_table_matches_scan_for_every_codeword` test cross-checks the fast
  path against the retained reference scan for every codeword of every
  selectable codebook (same `(x, y)`, same bits consumed). Adds a
  non-advancing `MainDataReader::peek(n)` primitive that reproduces the
  identical bits (and `exhausted` transition on the follow-up `read`) as
  the former bit-at-a-time accumulation.

### Changed

- demuxer: **container-level `duration_micros` now reports the
  playable (gapless-trimmed) duration** (r408). When the Xing/Info
  trailer carries the gapless extension (encoder-delay /
  zero-padding), `duration_micros` derives from
  `trimmed_duration_samples()` — the figure a gapless-aware player
  renders — instead of the gross frame-count duration. Streams
  without the extension are unchanged (trimmed == gross there), and
  the gross sample count remains available as
  `streams()[0].duration`.

### Fixed

- decoder: **foreign MPEG-2.5 8 kHz mixed-block granules no longer
  render lines 36..72 silent** (r408) — the second r405-recorded
  defect. The mixed decode hardcoded a 36-line long-coded region and
  started the short walk at short scalefactor band 3, which at the
  8 kHz Fraunhofer tables begins at per-window line 24 = wire line 72
  — so wire lines 36..72 of every foreign 8 kHz mixed granule were
  silently zeroed. r408 single-line observer probes (crafted with the
  crate's own low-level frame assembly; staircase long scalefactors
  and subblock-gain discriminators; four independent black-box
  validators) resolved the de-facto layout:
  - **coding split = `3 · short_starts[3]`** (36 at every ISO table,
    **72 at 8 kHz**, where it coincides with the span of the six
    transmitted LSF long scalefactor bands, `long_starts[6] = 72`):
    all four validators requantize wire lines 36..72 with **long**
    bands 3..5 (attenuation follows `scalefac_l[3..6]` exactly) and
    the long gain formula — `subblock_gain` does not apply there.
    `requantize` (and the encoder's `quantize` inverse) now use the
    band-relative split; behaviour at every other rate is unchanged.
  - **window split stays 36 lines** (§2.4.2.7's two lowest polyphase
    subbands) on three of the four validators: the long-coded lines
    36..72 pass through the reorder and are consumed by the short
    IMDCT of subbands 2..3 in native `[3·k + win]` interleave —
    exactly what this crate's pipeline produces once the requantizer
    covers the range. (The fourth validator long-windows the whole
    72-line region; with a 3-1 deployed split on window geometry the
    encoder's 8 kHz mixed *emit* refusal stands.)
  - the mixed Huffman **region-0 boundary is a deployed grey zone**:
    the old fixed 36 is refuted by all four validators; beyond that
    they split three ways at 8 kHz (72 / 96 / >100) and 2-2 at
    22.05 kHz (36 vs 48 — the band-relative reading vs the literal
    eight-entry mixed band sequence). The decoder now uses the
    band-relative `3 · short_starts[3]` (the primary validator's
    reading, identical at every rate but 8 kHz), and the encoder
    hardens every emitted mixed granule with `table_select[0] ==
    table_select[1]`, so every boundary interpretation consumes
    identical bits and **no** deployed decoder can desynchronise on
    our mixed output (35-case sweep re-verified float-perfect on all
    four validators).
  - the mixed long-band walk of the intensity-stereo path is now
    band-relative too, which also fixes a latent LSF bug: the fixed
    8-band walk re-processed lines 36..54 of an LSF mixed granule
    (only 6 long bands exist below 36 there) with untransmitted
    scalefactor slots doubling as bogus intensity positions.
  New tests: `tests/mixed_8k_foreign_decode.rs` (foreign-stream
  probes pin non-silence, the long-band scalefactor mapping, and the
  subblock-gain scope), requantize unit tests for the band-relative
  split, and `mixed_granules_use_one_table_for_both_big_value_regions`.

- encoder/decoder: **mixed bursts now carry `mixed_block_flag` on the
  flanking `Start` / `End` granules** (r408) — this closes the r405
  "second validator diverges on 44.1 kHz auto+mixed transition
  sequences" defect. §2.4.2.7 scopes the flag to *every*
  window-switched granule ("If window_switching_flag==1, then the
  mixed_block_flag indicates whether lower frequency polyphase filter
  subbands are coded using normal window type"), so a conformant mixed
  burst keeps its two lowest subbands on the normal window from the
  `Start` through every `Short` to the `End` — the only window lattice
  whose §2.4.3.4 low-subband overlap-adds cancel. The auto+mixed
  scheduler previously emitted `Start(mixed=0) → Short(mixed=1) →
  End(mixed=0)`: the start/end window tail against the mixed granule's
  normal-window head left **uncancelled aliasing in subbands 0..2** of
  every transition. The defect was invisible in self-decode and on
  validators that reproduce the stream's literal window sequence
  (three of four deployed black-box decoders tracked our decode at
  ≤ 7e-5) but a fourth deployed decoder rendered the low band
  differently — nrmse 3.1e-2 (click-train) to 7.6e-2 (attack-tone) at
  44.1 kHz, localized by a per-subband DFT of the diff to exactly
  polyphase subbands 0/1 across `Start → Mixed → End` frames. Fixes:
  - `BlockTypeStateMachine::step_with_mixed` now **latches the
    burst's mixed-ness at the `Start` commit** (from the lookahead
    granule's classifier preference) and applies it to the whole
    burst — `Start`, every `Short`, and the closing `End` share one
    flag (mixed-ness cannot change inside a burst; a `normal ↔
    short-stack` low-subband pairing would not cancel either). The
    lookahead granule's mixed preference is peeked with a cloned
    classifier exactly like the attack lookahead.
  - The encoder's forward MDCT windows subbands 0/1 of a mixed-burst
    `Start` / `End` with the **normal analysis window** (upper 30
    subbands keep the transition window); the granule template
    carries the flag; the long-family outer loop accepts it (the flag
    on a long-family granule changes only the synthesis window —
    spectral layout, part2 wire format, requantize formula, and
    §2.4.2.7 region defaults all key on `block_type == 2`).
  - The decoder's `windowed_block` now honours `mixed_block_flag` on
    **any** window-switched granule (previously only on `Short`), so
    foreign `Start`/`End`+mixed granules render per the spec.
  After the fix, all four deployed black-box validators decode
  auto+mixed streams in the float-rounding regime (44.1 kHz:
  ≤ 4.6e-5 on the formerly-diverging decoder, ≤ 6e-6 on the two
  float-perfect ones; 32 kHz / 22.05 kHz equivalent). New sweep cases
  (`mpeg1-44100-automixed`, `mpeg1-32000-automixed`,
  `mpeg2-22050-automixed`), a burst-coherence regression test
  (`mixed_burst_flags_flanking_transition_granules`), and an IMDCT
  window unit test pin the behaviour.

- encoder: **auto-path mixed bursts are demoted to pure-short at the
  LSF / MPEG-2.5 rates** (r408). The flagged-flank wire combination
  above is conformant at LSF too — the ISO/IEC 13818-3 main_data
  syntax scopes the mixed scalefactor layout to `block_type == '10'`,
  and its scalefac_compress partition tables mark `mixed_block_flag`
  as don't-care ('x') for block types '00'/'01'/'11' — but r408
  black-box measurements found deployed LSF decoders **split 2-2** on
  it: two track the spec reading float-perfectly (≤ 6e-6), two
  desynchronise on the whole burst (nrmse 0.42 / 1.27, consistent
  with reading a different scalefactor partition for a flagged
  `Start`/`End`). With no de-facto consensus to conform to, the
  scheduler emits pure-short bursts at the non-MPEG-1 rates (decoded
  identically on every validator); the toggle is still accepted and
  Short geometry still engages. `force_mixed_blocks_for_testing`
  (steady mixed streams, no transition flanks — all validators agree)
  remains available at every rate except 8 kHz. Pinned by
  `lsf_auto_mixed_demotes_to_pure_short_on_the_wire` and the
  `mpeg2-22050-automixed` sweep case (float-perfect on all four
  validators).

- demuxer: **hostile free-format streams can no longer "measure" a
  frame shorter than its own header** (r405, found by the new `demux`
  fuzz target within minutes). `measure_free_format_base_len` takes
  the byte distance between the first two matching syncs as the frame
  length; a crafted stream can place a second valid sync pattern 1..3
  bytes after the first, and the subsequent
  `first_frame_buf[..4].copy_from_slice(&hdr)` sliced out of range
  and panicked. The measurement now enforces the §2.4.1.3 structural
  floor — 4-byte header + optional 2-byte CRC + fixed-size side
  information — and rejects the stream as invalid below it.
  Regression-tested (`rejects_free_format_frame_shorter_than_header`)
  and the minimized fuzz artifact replays clean.

- alias/mixed: **mixed-block granules now get the single `sb == 1`
  alias-reduction butterfly on both the decode and encode sides**
  (r405). §2.4.3.4.10.1's two scope statements are written for the
  pure cases; the staged clarification
  (`docs/audio/mp3/mp3-alias-reduction-clarification.md`) resolves
  mixed blocks to exactly one butterfly group — the boundary internal
  to the two-subband long region (lines 10..26) — and the r405
  per-line observer-trace probe confirmed deployed decoders implement
  precisely that (before the fix, the only diverging positions of a
  mixed granule were lines 14..21, the high-coefficient taps of the
  sb == 1 butterfly). `alias::alias_reduce` previously followed the
  literal `block_type == 2` reading and passed mixed granules through
  unchanged, and the encoder emitted mixed granules without the
  inverse butterfly: every mixed-block stream this encoder produced
  decoded with an uncancelled butterfly on deployed decoders
  (measured nrmse 1.9e-2 at 44.1 kHz up to 3.5e-1 at 32 kHz). Both
  sides fixed (`alias_reduce` mixed arm + new
  `inverse_alias_reduce_mixed` on the two mixed forward paths);
  mixed-block encode now decodes on both external validators in the
  float-rounding regime (≤ 7e-6) at 32 / 44.1 / 22.05 kHz and on the
  primary validator at MPEG-2.5 11.025 / 12 kHz (the second deployed
  decoder renders MPEG-2.5 mixed differently from the first — a
  de-facto grey zone outside this crate's control).

- encoder: **window-switched `Start` / `End` granules no longer route
  through `choose_region_split`** (r405). The window-switched
  side-info branch carries neither the region counts nor
  `table_select[2]`, so every decoder reconstructs the §2.4.2.7
  defaults (`region0_count = 7`, `region1_count = 63` → region 0 =
  long bands 0..=7, region 1 = rest, region 2 empty) — but the
  encoder optimized its own split and assigned codebooks to line
  ranges no decoder would use, silently desynchronizing the Huffman
  regions of any transition granule whose optimized split disagreed
  with the defaults (sporadic bursts, e.g. nrmse 2e-2 on 44.1 kHz
  auto block-type streams; self-cancelling in own-decode round-trips
  because this decoder reconstructs the same defaults). The encoder
  now uses exactly the decoder-reconstructible boundaries for every
  window-switched granule (`long_starts[8]` for Start/End,
  `3·short_starts[3]` for pure short, 36 for mixed), and
  `default_transition_gc` carries the real §2.4.2.7 sentinels
  (region1_count = 63; the emitter reads them). Auto block-type
  streams now decode on both validators at ≤ 7e-6 at every rate.

- encoder: **mixed blocks are refused at 8 kHz**
  (`StreamEncodeError::MixedBlocks8kUnsupported`, r405). The deployed
  8 kHz Fraunhofer short table (per-window starts 0, 8, 16, 24, …)
  has no band boundary at per-window line 12, and the r405
  observer-trace found deployed black-box validators render 8 kHz
  mixed granules differently from each other.
  `force_mixed_blocks_for_testing` and
  `enable_auto_block_type_with_mixed` reject at 8 kHz; pure short
  blocks and the plain auto block-type path are fully supported there
  (validator-verified at ≤ 6e-6). (The r408 probes later resolved the
  disagreement precisely — the CODING layout is unanimous, the WINDOW
  geometry splits 3-1 — and fixed the decode side, which had left
  lines 36..72 of foreign 8 kHz mixed granules silent; see the r408
  entry above. The emit refusal stands.)

- tables: **MPEG-2.5 11.025 / 12 kHz scalefactor-band tables corrected
  to the deployed de-facto layout — the 16 kHz LSF table pair** (r405).
  MPEG-2.5 is a proprietary extension with no ISO table; the staged
  derivation doc (`docs/audio/mp3/mpeg2.5-scalefactor-bands.md`,
  #147/#151) hypothesised per-rate half-rate-sibling reuse (11.025 →
  22.05 kHz tables, 12 → 24 kHz tables, long + short). r405
  observer-trace against two independent black-box decoder binaries
  refutes half of that mapping: encoder-produced streams at 12 kHz
  (long blocks) and at both 11.025 / 12 kHz (wideband short blocks)
  decoded to a *different* waveform on both validators (nrmse 0.45–1.4
  vs our own decode). The deployed layout was then measured exactly —
  per spectral line — with crafted single-line probe streams whose
  staircase scalefactors make a validator's output amplitude reveal
  its band index for every position: **both 11.025 kHz and 12 kHz use
  the ISO/IEC 13818-3 16 kHz LSF long *and* short tables** (the 16 kHz
  long table is byte-identical to the 22.05 kHz one, which is why long
  blocks at 11.025 kHz always agreed). The 8 kHz Fraunhofer tables
  from the staged doc were read back verbatim by the same probe
  (including the width-2 fillers) and are confirmed. Both dispatch
  functions (`requantize::{long,short}_band_starts`) now carry the
  measured mapping with full provenance; `requantize_tests` pins the
  deployed pair and the refuted assignments. After the fix all
  encoder-produced MPEG-2.5 streams (long, short, auto block-type,
  tone and wideband noise, mono and stereo, all three rates) decode on
  both external validators in the float-rounding regime
  (nrmse ≤ 8e-5). The staged doc needs a corrigendum (docs ask
  recorded in the round report).

- huffman/encoder: **short-block Huffman region 0 is band-relative,
  not a hardcoded 36 lines** (r405). §2.4.2.7 sets the window-switched
  short-block default `region0_count = 8` counted in window-bands
  ("each scale factor band is counted three times"), i.e. region 0
  ends at interleaved line `3 · short_starts[3]`. Both the decoder's
  `region_boundaries` and the encoder's table-select assignment
  hardcoded the evaluated form `36` — correct for every ISO table
  (`short_starts[3] = 12`) but wrong for the MPEG-2.5 8 kHz Fraunhofer
  table (`short_starts[3] = 24` → region 0 = 72 lines). Deployed
  decoders apply the band-relative rule, so every 8 kHz short-block
  stream this encoder produced desynchronized their Huffman table
  regions outright (measured NCC ≈ 0.05 — noise). Both sides now
  compute `3 · short_starts[3]` (mixed granules keep the 36-line
  long/short split per the `region0_count = 7` default); 8 kHz
  forced-short and auto-block-type streams now decode on both
  validators at nrmse ≤ 5e-6.

- decoder/encoder: **short-block band 12 (the no-scalefactor tail) is
  now requantized, reordered, and quantized instead of silently
  dropped** (r405). The short-block band walk stopped at band 11 in
  four places — `requantize_short_range` (decode: lines above
  `3·starts[12]` stayed zero), `reorder` (decode: band 12 left in
  native interleave), `quantize` (encode: band-12 lines never coded),
  and `short_block::forward_reorder` (encode mirror). The §2.4.3.4.7.1
  band structure has 13 short bands; band 12 carries frequency lines
  requantized with scalefactor 0, exactly like the long path's band
  21 (which was handled correctly). The bug was self-cancelling in
  encode→own-decode round-trips (both sides dropped the same lines)
  and invisible on the fixture corpus (no fixture carries audible
  short band-12 energy), but the r405 per-line observer-trace probe
  exposed it immediately: deployed decoders render band 12
  (per-window lines `starts[12]..192`), this decoder returned silence
  there, and encoder output discarded the top of the spectrum of
  every short granule — at 12 kHz auto-block-type that measured
  nrmse ≈ 2e-2 vs both validators, now 6e-6. All four walks now cover
  13 bands with scalefactor 0 on the tail.

- demuxer: **packet keyframe flag now reflects the bit reservoir**
  (r367). `next_packet` previously stamped *every* emitted packet with
  `keyframe = true`. A Layer III frame is only an independently-decodable
  random-access point when its `main_data_begin` back-pointer is zero —
  a frame with `main_data_begin > 0` borrows main data from the reservoir
  of earlier frames (§2.4.2.7) and cannot be decoded in isolation, so a
  seeker landing on it produces corrupt audio until the reservoir
  refills. The demuxer now reads the back-pointer directly from the
  start of the side-information region (the new `frame_main_data_begin`
  helper — 9 bits for MPEG-1, 8 bits for MPEG-2 / 2.5 LSF, after the
  4-byte header and optional 2-byte CRC slot) and sets
  `keyframe = (main_data_begin == 0)`. Layers I/II carry no reservoir, so
  every frame stays a keyframe. The helper is cross-validated against the
  full `parse_side_info` parser across both field widths
  (`frame_main_data_begin_matches_side_info_parser`), and
  `keyframe_flag_tracks_main_data_begin` proves a non-zero back-pointer
  frame is flagged non-keyframe. Combined with the exact-PTS landing
  above, a seeker can now both target the right frame *and* tell whether
  that frame is a safe entry point.

- demuxer: **`seek_to` now reports the landed frame's exact PTS** (r367).
  Both the Xing-TOC and the CBR/proportional seek paths positioned the
  read cursor on a *byte estimate* (the percentile offset or the
  `pts · bitrate / (8 · sample_rate)` proportional offset), snapped it
  forward to the next real frame syncword via `resync_to_frame`, but then
  reported — and stamped on the first emitted packet — the **requested**
  PTS rather than the PTS of the frame the cursor actually landed on.
  Because the snap moves the cursor by up to one frame, the returned
  value and the first packet's `pts` disagreed with the rest of the
  stream's timeline, breaking monotonicity for any seek target that did
  not happen to fall on a frame boundary. New
  `Mp3Demuxer::pts_at_cursor` walks the frame *headers* from
  `first_audio_frame_offset` to the post-resync cursor, summing each
  frame's exact length (table-bitrate or free-format), to recover the
  true frame index; `seek_to` sets `next_pts = frame_index ·
  samples_per_frame`. The seek now returns a whole-frame-aligned PTS that
  matches the PTS later stamped on the next packet and keeps the stream
  strictly monotone — accurate for CBR, VBR (Xing TOC), and free-format
  alike. Covered by the strengthened
  `seek_lands_on_frame_aligned_pts_and_stays_monotone` unit test
  (off-boundary CBR request) and the end-to-end
  `vbr_xing_toc_seeks_via_demuxer` test (real encoder-produced VBR
  stream: a ~50 % TOC seek lands on a real interior frame boundary and
  the following packets carry monotone frame-exact PTS). This is the
  "§2.4.3 … seeking accuracy" milestone work.

### Added

- fuzz: **`demux` fuzz target — the container surface under
  attacker-controlled bytes** (r405). Third `cargo-fuzz` lane
  alongside `decode` / `granule`: drives `Mp3Demuxer::open` (ID3v2
  skip, Xing / Info / VBRI / LAME parsing, free-format frame-length
  measurement), bounded `next_packet` drains, every metadata accessor,
  and a seek schedule (front / attacker-chosen interior /
  far-past-EOF) through the TOC and proportional seek paths plus the
  post-seek exact-PTS frame recount. Half the iterations plant a
  structurally valid attacker-parameterised first header so `open`
  reaches the deep parse paths; the rest probe the resync / reject
  paths raw. Panic-freedom contract; the three targets ran locally
  this round — decode 1.28 M execs and granule 4.16 M execs clean;
  demux found the free-format sub-header-length panic (see Fixed)
  within minutes, then ran 19 M execs clean after the fix.

- tests: **black-box validator decode sweep of encoder output**
  (`tests/validator_decode_sweep.rs`, r405). Encodes PCM at every
  supported rate — MPEG-1 (32 / 44.1 / 48 kHz), MPEG-2 LSF (16 /
  22.05 / 24 kHz), and all three MPEG-2.5 rates (8 / 11.025 /
  12 kHz) — mono and stereo, long and forced-short (tone and wideband
  noise), hands each produced stream to an external black-box decoder
  binary, and asserts (a) the validator reports the exact sample rate
  and channel count (pinning the header version / sample-rate-index
  dispatch, including the MPEG-2.5 `id`-bit layout, against deployed
  decoders) and (b) the validator's PCM tracks this crate's own
  decode of the same bytes in the float-rounding regime (25 cases,
  all nrmse ≤ 8e-5 after the r405 fixes). The validator is injected
  via the `OXIDEAV_MP3_VALIDATOR_DECODE` environment variable (a
  shell command template with `{IN}` / `{OUT}` placeholders producing
  a 16-bit RIFF/WAVE file); the test skips with a log line when the
  variable is unset, so CI stays hermetic while the sweep runs
  locally against real deployed decoders. This sweep is what caught
  the MPEG-2.5 band-table, region-0, and short-band-12 defects fixed
  in this release.

- tests: **corpus-wide differential decode sweep**
  (`tests/corpus_reference_pcm.rs`, r405). Every Layer III fixture under
  `docs/audio/mp3/fixtures/` — all 16 of them: MPEG-1 mono/stereo at
  32 / 44.1 / 48 kHz, CBR 320, VBR, joint stereo (MS and intensity),
  short and mixed blocks, ID3v2 / Xing / VBRI frontmatter, the padding
  byte cycle, MPEG-2 LSF 22.05 kHz, and MPEG-2.5 11.025 kHz — is now
  decoded through the *production* chain (`Mp3Demuxer` → registered
  `Decoder`) and compared against its staged black-box reference PCM
  (`expected.wav`), aligned by peak normalized cross-correlation.
  Measured result: all 16 fixtures track their reference in the
  float-rounding regime (steady-state normalized RMS error ≤ 1.6e-5,
  alignment NCC = 1.0000) at the canonical 1105-sample codec-delay lag;
  the test pins nrmse < 2e-4, NCC > 0.999, and the exact 1105 lag per
  fixture. This generalizes the two single-fixture reference tests
  (`lsf_reference_pcm.rs`, `mpeg25_reference_pcm.rs`) to the whole
  corpus and to the demuxer + trait-decoder path a real player uses.
  Skips with a log line when the docs corpus is absent (standalone
  CI checkout).

- bench: **whole-stream encode benchmark harness** (`benches/encode.rs`,
  r398). A depth-mode Criterion harness times the full PCM → Layer III
  encode — analysis polyphase filterbank, forward MDCT with
  long/short/mixed windowing, the psychoacoustic threshold model, the
  nested inner rate loop and outer distortion loop, Huffman table
  selection + emission, and side-info / main-data / reservoir assembly.
  Five self-contained scenarios (mono tone / noise / log-sweep at 44.1 /
  48 / 32 kHz plus a mixed stereo clip) each synthesise their PCM in
  setup and are timed both **direct** (bare `Mp3Encoder`) and **trait**
  (the registered `oxideav_core::Encoder` object). No behaviour change,
  no fixture committed. Measured medians and an encode-vs-decode cost
  analysis are documented in `BENCHMARKS.md` (§ Encoder benchmarks):
  encode runs ≈ 5× the cost of decode (≈ 668 K samples/s, ≈ 15×
  real-time mono at 44.1 kHz), input-shape-sensitive (wide-spectrum
  noise ≈ 1.5× a steady tone), with the trait wrapper adding no
  measurable overhead.

- bench: **per-stage encode benchmark harness** (`benches/encode_stages.rs`,
  r398). Isolates the encode front-half stages — the analysis polyphase
  filterbank (`analyze_granule`), the long-block forward MDCT
  (frequency inversion → `forward_overlap` → `window_long_family_analysis`
  → `mdct`, the exact production sequence), and the §C.1.5.4.4 inner
  rate loop (`search_bit_budget`) — over one captured 38-granule batch,
  mirroring `decode_stages`. Result (documented in `BENCHMARKS.md`,
  § Encoder benchmarks): the **inner rate loop dominates encode by an
  order of magnitude** — ≈ 1.09 ms/granule versus ≈ 97 µs for the
  filterbank and ≈ 47 µs for the MDCT — because each candidate
  `global_gain` re-quantizes all 576 lines and re-counts their Huffman
  bits. No behaviour change.

- demuxer: **free-format (`bitrate_index == 0`) stream iteration** (r363).
  Free format fixes the bitrate but omits it from the header table, so a
  frame's length is not derivable from its header (§2.4.1.3) — the
  demuxer previously rejected such streams ("free-format MPEG audio frame
  at start"). `open()` now measures the constant **unpadded** frame body
  once (`measure_free_format_base_len`): the byte distance between the
  first two frame syncs agreeing on `(version, layer, sample_rate)`,
  minus the first frame's own padding slot. The base is stored, and
  `next_packet` / `resync_to_frame` derive each frame's length as
  `base + padding · slot_bytes` via the new
  `Mp3FrameHeader::frame_len_free_format` (+ `slot_bytes`, §2.4.2.1:
  4 bytes Layer I, 1 byte Layers II/III). An effective constant bitrate
  (`base · 8 · sample_rate / samples_per_frame`) is derived so the
  duration estimator and the CBR/proportional seek path work on
  free-format input. `locate_first_frame` accepts a free-format first
  sync; a single free-format frame (no second sync) is rejected. New
  `tests/demuxer_free_format_roundtrip.rs` proves the full demux→decode
  pipeline byte-exact against CBR for mono + stereo (CBR encoded then
  every frame's `bitrate_index` rewritten to 0, leaving the padding bit
  and main-data slot untouched), plus the derived bitrate/duration.

- encoder: **named psychoacoustic `QualityPreset` quality knob** (r355).
  New `quality` module + [`Mp3Encoder::with_quality_preset`] collapse the
  §C.1.5 / Annex D perceptual toggles (the §C.1.5.4.3 outer loop's per-band
  threshold, the §C.1.5.3.2.1 Model 2 analysis, and the §C.1.5.2
  Model-2-driven block-type scheduler) into one named level —
  `Transparent` / `High` / `Standard` / `Fast`, ordered by §D.1 Step 3
  threshold offset (`-24` / `-12` / `0` / `+6` dB). At the three staged
  Annex D rates (32 / 44.1 / 48 kHz) the richer presets arm the full
  signal-dependent Model 2 path; at the MPEG-2 LSF / MPEG-2.5 rates (no
  staged calculation-partition tables) a preset falls back to the
  signal-independent per-band threshold-in-quiet vector translated by the
  preset offset, so a preset is usable at every supported rate.
  `quality_preset()`, `quality_preset_is_signal_dependent()`, and
  `installed_per_band_xmin()` report the applied preset, which path it
  took, and the installed fallback vector. The registry path gains the
  `make_encoder_quality_preset` factory. The preset offset reaches the
  signal-dependent path through the new
  `XminThresholds::from_layer3_granule_with_offset_db`, which translates
  the Model 2 geometric-mean anchor by `10^(offset_db/10)` (preserving
  every per-band ratio — only the level moves). A one-call
  [`Mp3Encoder::new_with_quality_preset`] constructor bundles
  `new_with_outer_loop` + the preset for the common quality-knob
  front-end. This is the "psychoacoustically tuned default-on quality
  preset" the README listed as the remaining encoder work.

### Fixed

- decoder: **part-2 / part-3 split — `decode_huffman` was reading
  Huffman codewords starting at the part-2 scalefactor bits** (r348).
  The per-granule/channel decode loop positioned the `MainDataReader`
  at the start of the granule's `part2_3_length` field and passed the
  *full* `part2_3_length` as the Huffman budget, so `decode_huffman`
  decoded the part-2 (scalefactor) bits as if they were part-3
  big_values codewords. This was silent for `scalefac_compress == 0`
  streams (part-2 length 0 — the case every synthetic encode fixture
  hit), but corrupted **every frame carrying a non-zero `slen`
  partition** across MPEG-1, MPEG-2 LSF, and MPEG-2.5. `decode_scalefactors`
  now records the part-2 bit length per granule/channel
  (`FrameScaleFactors::part2_bits`); the decode path skips those bits
  before `decode_huffman` and budgets it at `part2_3_length −
  part2_bits`. Surfaced by the new MPEG-2.5 11.025 kHz reference-PCM
  test: steady-state normalized RMS error against the staged reference
  dropped from ≈ 0.77 to ≈ 1e-4 and the decode now locks at the
  canonical 1105-sample codec delay.

- encoder/psychoacoustics: **Layer III Model 2 long-block partition
  threshold sign** (r331). `model2_layer3_long_nb` (the §C.1.5.3.2.1 /
  Figure C.6.b long-path threshold `nbb(b) = ecbb(b)·norm(b)·10^(…/10)`)
  was applying a **negative** exponent — it reused the §D.2.4 step i)
  Layer-I/II *power ratio* operator `bc_b = 10^(−SNR_b/10)`, a different
  quantity — so the long-path threshold was flipped relative to both the
  printed spec and the crate's own short-path
  (`model2_layer3_short_nb`, already positive). The in-repo ISO/IEC
  11172-3:1993 §C.1.5.3.2.1 Figure C.6.b box prints the **positive**
  exponent `10^(SNR(b)/10)`, and Figure C.6.d (short blocks) prints the
  same positive form; the §C.6.b corrigendum render
  (`docs/audio/mp3/mpeg2.5-scalefactor-bands.md` §"§C.6.b corrigendum
  check", #139) independently confirms there is no minus sign in the
  figure. The long path now uses `10^(+SNR/10)` directly, making the
  long- and short-block threshold the one identical function. Affects
  only the opt-in `enable_model2_psychoacoustics` path. Unit test
  `layer3_snr_and_nb_conventions` updated to the positive convention.

### Added

- tests: **MPEG-2.5 fixture now exercised at the demuxer level** (r348).
  `tests/docs_corpus.rs` adds `layer3-mpeg25-11025-32kbps` to the
  `every_fixture_walks_without_panic` walk and a dedicated
  `mpeg25_11025_fixture_demuxes_with_correct_params` test asserting the
  demuxer reports 11.025 kHz / mono, parses the Xing/Info frame, matches
  the first-frame offset, and walks all 18 playable frames with monotone
  PTS. The stale exclusion comment claiming the parser needed the 12-bit
  `0xFFF` syncword was removed — the framing layer has synced on the
  11-bit `0xFFE` pattern (so MPEG-2.5's `id`-bit-0 passes through) for
  many rounds.

- tests: **MPEG-2.5 11.025 kHz decode reference-PCM test** (r348). New
  `tests/mpeg25_reference_pcm.rs` decodes the staged
  `layer3-mpeg25-11025-32kbps` fixture (`input.mp3`) through the direct
  decode chain and aligns it against the toolchain's `expected.wav`,
  asserting a lock at the canonical 1105-sample codec delay and
  steady-state normalized RMS error < 0.005 (measured ≈ 1e-4). The
  MPEG-2.5 sibling of `tests/lsf_reference_pcm.rs`; it is the test that
  surfaced the part-2/part-3 decode bug fixed above. A companion
  `mpeg25_11025_trait_decoder_matches_direct_chain` test drives the same
  fixture through the registered `oxideav_core::Decoder` trait wrapper
  (`Mp3CoreDecoder`) and asserts it reconstructs the same PCM as the
  direct chain byte-exactly (modulo the leading Xing/Info granule the
  production decoder emits), proving the part-2/part-3 fix is wired into
  the production decode path, not just the test harness.

- encoder/tests: **end-to-end MPEG-2.5 encode verification at all three
  extension rates** (r340). The band tables were wired in r321 and the
  psychoacoustic threshold path is rate-generic, but encode→self-decode
  coverage only exercised 11.025 kHz. Added long-block round-trips at
  12 kHz (reuses the 24 kHz LSF table) and 8 kHz (the distinct
  Fraunhofer table, top long bands 17–21 collapsing to width 2), an
  8 kHz forced-short-block round-trip driving the distinct
  `SHORT_STARTS_MPEG25_8` layout through reorder + outer loop, and the
  milestone integration test
  `mpeg25_threshold_in_quiet_psychoacoustic_roundtrip_all_rates` —
  `XminThresholds::threshold_in_quiet` built over each rate's band
  starts feeds the §C.1.5.4.3 distortion-control loop and yields a
  decodable band-aligned stream at 8 / 11.025 / 12 kHz. A companion
  `mpeg25_threshold_in_quiet_band_vector_is_band_aligned` asserts the
  per-band `xmin` vector is finite-positive over all 21 transmitted
  long bands and genuinely non-uniform (`max/min > 4`), witnessing that
  the band partitioning shaped the threshold instead of collapsing to a
  constant. Six new tests; the stale "documented placeholder" comment
  on the 11.025 kHz round-trip (the tables are real grounded tables
  now) is corrected.
- decoder/trait: **MPEG-2.5 at 8 kHz now decodes through the `Decoder`
  trait wrapper** (r335) — completes the full MPEG-2.5 decode path
  (8 / 11.025 / 12 kHz). The wrapper previously rejected 8 kHz up front
  pending observer-trace grounding of its band table. The newly-staged
  `docs/audio/mp3/mpeg2.5-scalefactor-bands.md` (commit `c2e236e`, #147 /
  #151) documents the Fraunhofer 8 kHz long/short SFB tables verbatim
  (top long bands 17–21 and short bands 9–11 collapse to width 2 at the
  4 kHz Nyquist) as published-factual constants satisfying the
  Table-B.2 structural invariants (Σ = 576 long / 192 short, contiguous,
  22/13 bands); they were already transcribed into
  `requantize::{LONG,SHORT}_STARTS_MPEG25_8` and the whole
  side-info → scalefactor → Huffman → requantize → reorder → stereo →
  IMDCT → synthesis chain is rate-generic, so dropping the gate lets
  8 kHz decode through the identical path as the other LSF rates. The
  former `send_packet_rejects_mpeg25_8khz_pending_observer_trace` test is
  replaced by `send_packet_accepts_mpeg25_8khz_header_through_the_guard`
  and a new byte-exact `trait_decode_mpeg25_8khz_byte_exact_with_direct_chain`
  (trait wrapper output == direct decode chain, sample-for-sample, on a
  real encoder-produced 8 kHz stream).
- decoder/trait: **MPEG-2.5 at 11.025 / 12 kHz now decodes through the
  `Decoder` trait wrapper** (r326). The wrapper's version guard
  previously rejected *all* MPEG-2.5 frames; it now accepts the two
  fully-grounded extension rates. Per
  `docs/audio/mp3/mpeg2.5-scalefactor-bands.md` (#147/#151) the
  11.025 / 12 kHz scalefactor-band tables are byte-identical to the
  in-repo ISO/IEC 13818-3 22.05 / 24 kHz LSF Table B.2 entries, and the
  header `id`-field → sample-rate dispatch is grounded in the staged
  datavoyage header reference (`MPEG-2.5-GAP.md`), so these rates decode
  through the identical side-info → scalefactor → Huffman → requantize →
  reorder → stereo → IMDCT → synthesis chain that MPEG-2 LSF already
  uses. The **8 kHz** rate stays rejected: its band table is a distinct
  Fraunhofer table with no in-repo half-rate sibling and no
  observer-trace fixture, so it remains gated on the residual
  `MPEG-2.5-GAP.md` observer-trace item. New tests:
  `trait_decode_mpeg25_11025_byte_exact_with_direct_chain` (byte-exact
  trait-vs-direct), `send_packet_accepts_mpeg25_12khz_header_through_the_guard`,
  and `send_packet_rejects_mpeg25_8khz_pending_observer_trace` (replaces
  the prior all-MPEG-2.5 rejection test, whose premise no longer holds).
- encoder/band-tables: **MPEG-2.5 (8 / 11.025 / 12 kHz) Layer III
  scalefactor-band tables wired in** (r321). Closes the README "lacks
  only MPEG-2.5 band tables" tail using the newly-staged
  `docs/audio/mp3/mpeg2.5-scalefactor-bands.md` (#147/#151). The
  11.025 kHz and 12 kHz long+short tables are byte-identical to the
  in-repo ISO/IEC 13818-3 22.05 kHz / 24 kHz LSF Table B.2 entries and
  now alias those constants; 8 kHz is the distinct Fraunhofer table
  (new `LONG_STARTS_MPEG25_8` / `SHORT_STARTS_MPEG25_8`, top long bands
  17–21 and short bands 9–11 collapsing to width 2 as the 4 kHz Nyquist
  leaves no high-frequency energy). The previous half-rate MPEG-1
  placeholders (8 kHz→32 kHz, 11.025→44.1 kHz, 12→48 kHz layouts) are
  removed. Because `requantize::{long,short}_band_starts` is the single
  band-boundary source shared by quantize, the inner/outer loops,
  reorder, stereo, and the requantizer, the fix propagates through the
  whole encode (and decode) pipeline at the MPEG-2.5 rates. Eight new
  tests: exact 8 kHz long/short values + width-2 collapse, the
  11.025/12 kHz LSF aliasing, structural invariants across all three
  rates, an 8 kHz band-boundary scalefactor-selection check, a
  regression against the old 32 kHz placeholder alias, and a full
  `Mp3Encoder` 8 kHz encode producing a valid MPEG-2.5 stream.
  (Trait *decode* at MPEG-2.5 stays gated on the remaining
  `MPEG-2.5-GAP.md` observer-trace items — header `id`-field dispatch,
  bit-exact Huffman table mapping, the 8 kHz table's in-repo grounding.)

### Changed

- decoder: **§2.4.3.4.7 output-PCM rounding made spec-exact** (r318). The
  reconstructed signal is a fractional two's-complement value in
  `[-1.0, +1.0]` whose MSB carries the weight `-1`; the 16-bit signed PCM
  output therefore scales by the MSB weight `2^15 = 32768` (not `32767`)
  and rounds to the **nearest integer with half-integer values rounded
  away from zero** — the spec's §2.3 "Nearest integer operator". The
  previous conversion scaled by `32767` and truncated toward zero (`x as
  i16`), biasing every non-integer sample one step toward zero and
  slightly attenuating full-scale amplitude. New shared
  `synth::pcm_f32_to_i16` helper centralises the conversion; the trait
  decode path, the direct-chain decode helpers, and every roundtrip test's
  reference decoder now route through it. Six unit tests cover the scale
  factor, endpoint saturation, nearest-integer rounding, and
  half-away-from-zero behaviour.

### Added

- encoder: **§C.1.5.3.2.1 Model-2-driven auto block-type under
  intensity-stereo coupling** (r313). `enable_auto_block_type_model2`
  previously rejected an intensity-armed encoder with
  `IntensityShortBlocksUnsupported`; that guard was the sole blocker. The
  frame-assembly `channel_agreement_active` OR-fold already keyed off
  `intensity_start_sfb.is_some()` for the Model-2 emission path (the
  per-channel `pe > 1800` window-switching flags are OR-folded into one
  shared channel-0 scheduler whose emission is mirrored across both
  channels, so the §2.4.3.4.9 "both channels of a granule share
  `block_type` / `window_switching_flag` / `mixed_block_flag`" agreement
  intensity coupling needs holds by construction), and Pass 1 selects the
  per-granule §2.4.3.4.9.3 per-window short coupling (pure-short granules)
  vs the long-block band walk (Long / Start / End granules) from the same
  `block_type_per_gc` matrix the Model-2 path produces. The §C.1.5.2 walk
  emits no mixed block, so the §2.4.3.4.10.3 mixed carve-out coupling is
  never reached. This mirrors the energy-detector
  `enable_auto_block_type` acceptance (r307 MS+intensity, r308
  intensity-only). New public `Mp3Encoder::enable_intensity_stereo`
  (running-state counterpart of `new_joint_stereo_is`) makes the
  combination reachable on an outer-loop encoder, so the
  Model-2 per-band threshold and the Model-2 block-type scheduler can be
  armed alongside intensity coupling. Three new lib tests:
  `enable_intensity_stereo_arms_and_rejects_mono` (arming + mono / range
  rejection), `model2_block_type_accepts_intensity_coupling` (the lifted
  guard now returns `Ok`), and
  `model2_intensity_emits_valid_joint_intensity_frames` (a transient
  hard-panned stereo stimulus emits `mode = '01'` intensity frames with
  L/R window geometry consistent per granule, plus byte-deterministic
  re-encode). **Still rejected:** the mixed-promotion auto variant
  (`enable_auto_block_type_with_mixed`) under intensity.

### Fixed

- decoder: **§2.4.3.4.8 reorder restored to the trait decode path**
  (r304). `Mp3CoreDecoder` (the registered `oxideav_core::Decoder`) ran
  `requantize → process_stereo → alias_reduce → imdct_granule` with no
  short-block reorder stage. Long blocks were unaffected (reorder is the
  identity there), but short (`block_type == 2`) and mixed granules
  reached the IMDCT and the joint-stereo short-block path still in the
  native `(sfb, window, freqline)` Huffman interleave instead of the
  expected subband-window-interleaved layout, so every short-block (and
  mixed short-region) granule decoded to corrupt PCM through the trait
  API, mono and stereo. `reorder` now runs between `requantize` and
  `process_stereo`, matching the spec stage order and the proven direct
  decode helpers. New lib regression test
  `trait_decode_short_block_runs_reorder_and_is_not_silent` (force-short
  stream, byte-exact vs the reordering direct chain + non-silent
  witness); the `decoder_trait_lsf_roundtrip` reference helper, which had
  the same omission against a short-block fixture, is corrected in the
  same commit.

### Added

- encoder: **§2.4.3.4.10.3 mixed-block intensity stereo (intensity-only,
  non-MS)** (r311). `force_mixed_blocks_for_testing(true)` is now accepted
  on an intensity-only encoder (`new_joint_stereo_is`); it was the last
  block-type toggle still rejected with `IntensityShortBlocksUnsupported`.
  A mixed block (§2.4.3.4.10.3 / PDF p.26) transforms its lowest 2
  polyphase subbands (long bands 0..=7, lines 0..36) with a normal long
  window and the upper 30 subbands (short bands 3..12) as short blocks, so
  intensity coupling runs in **two regions**: the long region couples on
  the long-band walk (positions on the right channel's `scalefac_l[sfb]`)
  and the short region couples per window (positions on
  `scalefac_s[sfb][win]`) — the exact two-region geometry the decoder's
  `process_stereo` already reconstructs for a `mixed_block_flag` granule
  (long bands `0..8`, then per-window short bands
  `MIXED_FIRST_SHORT_SFB..12`). The user-facing `intensity_start_sfb`
  (1..=20) addresses the long grid directly; for the short region it is
  mapped onto a short band by frequency and clamped to
  `MIXED_FIRST_SHORT_SFB` (the three lowest short bands fall inside the
  long carve-out). All-zero bands below each region's derived bound carry
  the illegal-position marker `7` (Annex G.2 c); the right channel carries
  `scalefac_compress = 15` so the long (slen1) and short (slen1/slen2)
  positions fit. Header emits `mode = '01'` / `mode_extension = '01'`.
  **Still rejected:** mixed + intensity under MS-joint stereo (the
  §2.4.3.4.9.2 below-bound rotation over the mixed split line set) and the
  mixed-promotion *auto* variant (`enable_auto_block_type_with_mixed`
  under intensity). New `tests/mixed_block_intensity_roundtrip.rs` (4
  tests): the `'01'` header + mixed side-info, long-and-short right-channel
  positions in range, a hard-left 8 kHz intensity tone reconstructing
  strongly left-leaning through a spec-order self-decode
  (huffman → requantize → reorder → process_stereo → alias → imdct →
  synth), and a byte-deterministic re-encode. The
  `intensity_rejects_block_type_toggles` unit test now asserts force-mixed
  + intensity-only acceptance and keeps the MS-mixed / auto-mixed
  rejections.

- encoder: **§2.4.3.4.9 auto-block-type-scheduled short granules with
  intensity-only (non-MS) joint stereo** (r308). `enable_auto_block_type`
  is now accepted on a plain intensity encoder
  (`new_joint_stereo_is`, no MS) — the last rejected signal-driven
  auto + intensity combination, previously
  `IntensityShortBlocksUnsupported`. r307 lifted the rejection only when
  MS-joint stereo was also armed (the §2.4.3.4.9 channel agreement that
  intensity coupling needs held by construction via the MS scheduler
  mirroring). r308 makes that agreement structural for intensity too:
  whenever intensity coupling is armed, the auto/Model-2 scheduler walk
  now keys on `channel_agreement_active = MS-joint OR intensity-armed`,
  which OR-folds the per-channel attack flags into one shared (channel-0)
  state machine and mirrors its emission across both channels. L/R block
  types are therefore channel-consistent by construction, so each
  granule's pure-short / long intensity coupling has a well-defined fold
  geometry without MS. Header emits `mode = '01'` with
  `mode_extension = '01'` (intensity on, MS off). The mixed-promotion
  auto variant (`enable_auto_block_type_with_mixed`, §2.4.3.4.10.3
  two-region carve-out unwired) and the Model-2-driven auto path under
  intensity remain rejected; force-mixed under intensity remains
  rejected. `tests/auto_block_type_intensity_roundtrip.rs` grows 4
  intensity-only tests (block-type mix + per-granule agreement with
  `mode_extension == '01'`, per-window scalefactor positions in range, a
  hard-left intensity-region tone reconstructing left-leaning through a
  spec-order self-decode, deterministic byte-exact re-encode); the
  `intensity_rejects_block_type_toggles` unit test now asserts
  intensity-only-auto acceptance and keeps the mixed-promotion rejection.

- encoder: **§2.4.3.4.9 auto-block-type-scheduled short granules with
  MS-joint intensity stereo** (r307). `enable_auto_block_type` is now
  accepted on an MS-joint intensity encoder
  (`new_joint_stereo_ms_is` / `new_joint_stereo_auto_is`) — previously
  rejected with `IntensityShortBlocksUnsupported`. The §C.1.5.2 attack
  scheduler emits a *mix* of Long / Start / Short / End granules within
  one stream; the intensity coupling is now chosen **per granule** rather
  than per frame. A granule the scheduler emits as a **pure short** block
  takes the §2.4.3.4.9.3 per-window short coupling (the r303/r305
  machinery, now keyed on `block_type_per_gc[gr][0]` instead of the
  frame-wide `force_short_blocks`); Long / Start / End granules take the
  long-block band-walk coupling. The Pass 1.5 MS picker and rotation pick
  their region per granule the same way. MS-joint stereo mirrors one
  shared scheduler emission across both channels, so the §2.4.3.4.9
  channel agreement intensity coupling needs (it folds each granule's
  `(L, R)` band-by-band) holds by construction. **Still rejected:** the
  *intensity-only* auto path (no MS — independent per-channel scheduling
  may diverge L/R block types), mixed-block intensity
  (`enable_auto_block_type_with_mixed`, the §2.4.3.4.10.3 carve-out bound
  is not wired), and the Model-2-driven auto path under intensity. New
  `tests/auto_block_type_intensity_roundtrip.rs` (5 tests): API
  acceptance/rejection matrix, a transient stimulus that drives the
  scheduler into both long-family AND pure-short granules (with §2.4.3.4.9
  per-granule channel-agreement assertions), per-window short scalefactor
  positions in range, a hard-left intensity-region tone reconstructing
  left-leaning through a spec-order self-decode, and byte-deterministic
  encode. The `intensity_rejects_block_type_toggles` unit test now
  asserts auto + MS + intensity acceptance and intensity-only-auto
  rejection.

- encoder: **§2.4.3.4.9.2 MS-*auto* picker over the per-window short
  intensity region** (r306). The force-short toggle
  (`force_short_blocks_for_testing(true)`) is now accepted on the
  *auto*-MS + intensity encoder (`new_joint_stereo_auto_is`); it was the
  last combination still rejected with `IntensityShortBlocksUnsupported`.
  The per-frame side-energy picker (`E_S/(E_L+E_R)`) previously scored
  its fraction over the long-block bound line range `0..ms_region_hi`,
  but the r305 MS rotation for short + intensity touches the per-window
  short region `0..3·short_starts[short_start]`. The picker now recomputes
  that per-granule upper line (using `short_intensity_start_per_gr[gr]`
  from Pass 1.45) when short + intensity is armed, so the MS-vs-LR
  decision is measured on exactly the lines the rotation applies; the
  long / no-intensity picker path is byte-for-byte unchanged. Frames
  carry `mode = '01'` with per-frame `mode_extension = '11'` (MS +
  intensity) / `'01'` (intensity only). Three new
  `tests/ms_short_intensity_roundtrip.rs` tests cover acceptance +
  byte-deterministic encode, the picker firing MS on low-side-energy
  short content, and the picker declining MS for any non-zero short side
  energy at threshold 0. The `intensity_rejects_block_type_toggles` unit
  test now asserts MS-auto + short + intensity acceptance.

- encoder: **§2.4.3.4.9.2 MS + short-block + intensity stereo** (r305).
  The force-short toggle (`force_short_blocks_for_testing(true)`) is now
  accepted on the unconditional MS + intensity encoder
  (`new_joint_stereo_ms_is`); it was previously rejected with
  `IntensityShortBlocksUnsupported` because the below-bound MS rotation
  needed the §2.4.3.4.8 interleaved short layout. The MS matrix now
  applies per window below each window's short intensity bound — bands
  `0..short_start` across all three windows, which in the interleaved
  layout is the contiguous run `0..3·short_starts[short_start]` (the
  reorder is a permutation of that line set) — the exact inverse of the
  decoder's per-window `process_short`. Intensity (above the bound) and
  MS (below it) touch disjoint line sets. Frames carry `mode = '01'`,
  `mode_extension = '11'`. (The MS-*auto* + short + intensity path was
  lifted in r306.) New integration suite
  `tests/ms_short_intensity_roundtrip.rs`: `'11'` header + pure-short
  side info, right-channel positions in range, a spec-order self-decode
  with the below-bound 440 Hz MS pan reconstructing at 1.40 and the
  hard-left 8 kHz intensity tone left-leaning, and a byte-deterministic
  encode. The `intensity_rejects_block_type_toggles` unit test asserts
  MS + short force-short acceptance.

- encoder: **§2.4.3.4.9.3 short-block intensity stereo** (r303). The
  force-short toggle (`force_short_blocks_for_testing(true)`) is now
  accepted on an intensity-only encoder (`new_joint_stereo_is`); it was
  previously rejected with `IntensityShortBlocksUnsupported`. Each
  granule's 12 short bands × 3 windows are intensity-coupled with a
  **per-window** bound (ISO/IEC 13818-3 §2.4.3.2): positions are derived
  from each window's L/R band energies (Annex G.2 c) and written to the
  right channel's `scalefac_s[sfb][win]` slots (with the illegal marker
  `7` on each window's all-zero bands above its own last non-zero
  quantized line). The right channel carries `scalefac_compress = 15`
  (126-bit short part2). Mixed / auto-scheduled short granules and the
  MS + short + intensity combination remain rejected with
  `IntensityShortBlocksUnsupported`. New integration suite
  `tests/short_block_intensity_roundtrip.rs` (4 tests); the
  `intensity_rejects_block_type_toggles` unit test now asserts
  force-short acceptance plus the narrowed rejections.
- encoder: **§2.4.3.4.9.3 adaptive per-granule intensity bound** (r302).
  New `Mp3Encoder::new_joint_stereo_auto_is_adaptive(bitrate,
  sample_rate, intensity_start_floor)` treats the intensity start band as
  a *floor* and chooses the coupling bound per granule from the post-MDCT
  spectrum, rather than coupling `start_sfb..21` on every granule like the
  fixed-bound constructors. The chooser couples only the contiguous high
  tail whose every band has side-energy fraction
  `E_S/(E_L+E_R) = Σ(L−R)²/(2·Σ(L²+R²)) <= threshold` (default `0.25`,
  overridable via `with_intensity_auto_threshold`, clamped to `[0,1]`); a
  band that still carries real stereo content raises the bound, and a
  granule with no qualifying tail couples nothing (keeps a full right
  channel). The bound is implicit on the wire (§2.4.3.4.9.1), so the
  header stays `mode='01'` / `mode_extension='01'` and no syntax changes.
  Accessor `intensity_auto_threshold()` reports the armed threshold.

### Fixed

- encoder: the pass-2 intensity-right scalefactor path was gated only on
  the global intensity-armed flag, so an intensity granule that coupled
  no bands would still have written its right channel as is_pos markers
  instead of ordinary scalefactors. It is now gated per granule; a
  coupled-nothing granule writes a normal right channel. Previously only
  reachable via the new adaptive constructor (the fixed-bound modes
  always couple at least one band), so no emitted stream changes for
  pre-r302 encoder configurations.

- encoder: **§C.1.5.3 scfsi reuse auto-armed inside `push_samples`**
  (r301). r296 added `Mp3Encoder::enable_scfsi_reuse()` as an opt-in
  post-quantization pass that marks, per channel, every long-block
  scfsi_band group whose granule-1 scalefactors are byte-identical to
  granule 0's so they are transmitted once instead of twice. This round
  flips it on by default: a freshly constructed `Mp3Encoder` (every
  constructor funnels through `new`) now emits scfsi automatically. The
  detection is byte-exact and the decoder reconstructs granule 0's values
  for a marked group, so auto-arming is lossless by construction — the
  reconstructed PCM is identical to the historical `scfsi = 0` output
  while granule 1's part2 budget shrinks wherever consecutive granules
  naturally share scalefactors. The optimisation still never fires on LSF
  (one granule, no scfsi field) nor on a channel with a short granule
  (§2.4.2.7). New `Mp3Encoder::disable_scfsi_reuse()` restores the
  pre-r301 byte-for-byte `scfsi = 0` stream (compatibility /
  regression-bisection escape hatch); `enable_scfsi_reuse()` is retained
  to re-arm after an explicit disable. The renamed lib test
  `scfsi_reuse_auto_armed_by_default_disarmed_by_toggle` asserts
  default-on / `disable` clears / `enable` re-arms; a new integration
  test `scfsi_auto_armed_by_default_sets_reuse_flags` confirms a default
  encoder sets scfsi on a steady tone, never grows the stream vs.
  disarmed, and decodes sample-for-sample identical. The intensity-stereo
  roundtrip decoder's hand-rolled part2 skip is now scfsi-aware. Full
  suite green; every self-decode test reconstructs bit-exactly. Spec
  read: §2.4.2.7 / §C.1.5.3.

- encoder: **§C.1.5.4.4 band-aligned bit-budget search wired into the
  outer (distortion-control) loop** (r300). r299 swapped the fixed-gain
  CBR path to `search_bit_budget_band_aligned` but left the §C.1.5.4.3
  outer-loop branches on the default `search_bit_budget`, so the noise-
  shaping loop kept measuring `global_gain` against the pair-thirds
  `subdivide` heuristic — a part2_3 length whose region boundaries can
  land mid-band, i.e. a partition the decoder's `region_boundaries`
  cannot reconstruct and the encoder never emits. Both per-iteration
  inner-loop helpers (`run_inner` for long / Start / End blocks and
  `run_inner_short` for short / mixed blocks) now call
  `search_bit_budget_band_aligned`, so the gain the outer loop settles on
  fits the band-aligned wire partition the encoder actually writes. For
  long-family blocks the gain is now measured against the §C.1.5.4.4.6
  SUBDIVIDE snapped to scalefactor-band edges (`region0_count` /
  `region1_count` within the 4-bit / 3-bit field widths); short / mixed
  blocks share the two-subregion blocksplit path, so for those the new
  gating is bit-identical to the old. One new lib test
  (`outer_loop_long_gain_fits_band_aligned_wire_partition`): the chosen
  `is[]` re-counted via `exact_bit_count_band_aligned` at the final gain
  fits the per-granule budget and uses the band-aligned region ends. Lib
  suite 1085 → 1086; full roundtrip / PSNR suite stays green (every self-
  decode test reconstructs bit-exactly). Spec read: §C.1.5.4.3 outer loop
  + §C.1.5.4.4 / .4.6 inner loop, PDF pages 100-104.
- encoder: **§C.1.5.4.4 band-aligned bit-budget search wired into the CBR
  encode path** (r299). The fixed-gain CBR inner loop now chooses
  `global_gain` with `search_bit_budget_band_aligned` (r298) instead of
  the default `search_bit_budget`. The default search gates the gain on
  the pair-thirds `subdivide` heuristic, whose region boundaries can land
  mid-band — a part2_3 length the decoder's `region_boundaries` cannot
  reconstruct, so the gain it returned was measured against a partition
  the encoder never emits (the emitted region counts come from the
  band-aligned `choose_region_split`). The band-aligned search measures
  bits against the same scalefactor-band-edge SUBDIVIDE the encoder
  writes to the wire, so the chosen gain fits the real emitted part2_3
  length. Short / mixed blocks share the two-subregion blocksplit path
  and are unchanged. VBR (clamp-only) and outer-loop branches are
  untouched. One new integration test (`stream_encoder_roundtrip` 3 → 4)
  asserting every CBR long-block granule's `part2_3_length` stays within
  the per-frame budget; full suite green.
- encoder: **§C.1.5.4.4 band-aligned bit-budget inner-loop search**
  (r298). New `inner_loop::search_bit_budget_band_aligned` (re-exported at
  the crate root) runs the spec's upward `qquant + 1` rate-control scan
  (§C.1.5.4.4) gated on `exact_bit_count_band_aligned` — the
  wire-representable §C.1.5.4.4.6 SUBDIVIDE whose region boundaries fall
  on scalefactor-band edges — so the smallest `global_gain` it returns
  fits the part2_3 length the encoder will actually write, not an
  unrepresentable pair-thirds approximation. The default
  `search_bit_budget` is untouched (still gated on the simpler
  `exact_bit_count`), so every emitted byte is byte-for-byte the
  historical default; the band-aligned search is opt-in for a future
  bit-budget-driven encode path. Short / mixed blocks share the
  two-subregion blocksplit path and are bit-identical to
  `search_bit_budget`. Five new lib tests (inner-loop suite 34 → 39).
- encoder: **§C.1.5.4.4.6 band-aligned SUBDIVIDE** (r297). New pure
  helpers `inner_loop::subdivide_bands` and `exact_bit_count_band_aligned`
  (re-exported at the crate root, with `SubdivideBands`). The spec's
  §C.1.5.4.4.6 SUBDIVIDE "splits the *scalefactor bands*" into three
  regions and the `region0_count` / `region1_count` side-info fields are
  band counts; the decoder reconstructs the region boundaries only from
  the long-block band-start table and those counts, so a boundary chosen
  mid-band is unrepresentable on the wire. `subdivide_bands` snaps the
  "~1/3 to region 0, ~1/4 to region 2" strategy to scalefactor-band edges
  and returns valid 4-bit / 3-bit `region0_count` / `region1_count`
  values; `exact_bit_count_band_aligned` counts the §C.1.5.4.4.5 + .8
  Huffman total against those band-aligned long-family boundaries (short /
  mixed blocks fall back to the two-subregion pair split). The default
  `exact_bit_count` / `subdivide` and the inner-loop `global_gain` search
  are unchanged, so the encoder's emitted bytes are byte-for-byte the
  historical default — the band-aligned estimate is opt-in. Seven new lib
  tests.
- encoder: **§C.1.5.3 scalefactor-selection-information (scfsi) reuse**
  (r296). MPEG-1 Layer III carries two granules per frame, each with its
  own part2 scalefactor block. The §2.4.2.7 `scfsi[ch]` field lets a
  frame transmit a long-block scfsi_band group's scalefactors once (in
  granule 0) and declare them valid for granule 1, when the two granules
  already agree there. New `Mp3Encoder::enable_scfsi_reuse()` arms a pass
  that, after quantization, sets `scfsi[ch][g] = 1` for every one of the
  four scfsi_band groups (bands `{0..=5}`, `{6..=10}`, `{11..=15}`,
  `{16..=20}`, Table 3-B.8) whose granule-1 scalefactors equal granule
  0's across the whole group — **only** when both granules of the channel
  are long blocks. Per §2.4.2.7 ("if short windows are switched on …
  then scfsi is always 0 for this frame"), a short granule in the frame
  forces `scfsi[ch] = 0` for that channel. The §2.4.2.7 write path
  already skips a reused group in granule 1 and the decoder reproduces
  granule 0's values verbatim, so the reconstructed scalefactors — and
  every decoded sample — are bit-identical; only granule 1's part2 bit
  budget shrinks. Default off (every frame carries `scfsi = 0`,
  byte-for-byte the historical default); inspect with
  `scfsi_reuse_enabled()`. No-op on LSF (MPEG-2 / MPEG-2.5: one granule,
  no scfsi field). Five new lib tests (all-group reuse on identical
  granules, no reuse when every band differs, per-group independence,
  short-granule disqualification, default-off/armed-by-toggle) plus a new
  `tests/scfsi_reuse_roundtrip.rs` integration suite (5 tests): the
  disarmed encode sets no scfsi, the armed fixed-gain encode sets reuse
  on long-block frames, armed-vs-disarmed decode is bit-identical on both
  the fixed-gain and outer-loop paths, and the armed outer-loop encode
  strictly shrinks granule-1's summed part2_3 bit budget without growing
  the stream.
- encoder: **§C.1.5.3.2.1 Model-2-driven auto block-type path** (r295,
  Phase 2 step 92). r294 captured the per-granule Model 2 `pe > 1800`
  window-switching decision; this round wires it into an actual
  block-type driver. New `Mp3Encoder::enable_auto_block_type_model2()`
  arms a mode where the per-granule §C.1.5.2 attack flag fed into the
  `LONG → START → SHORT → STOP → LONG` scheduler is the spec-canonical
  Model 2 psychoacoustic-entropy decision (`pe > 1800`) rather than the
  energy-detector subframe-energy ratio of
  `enable_auto_block_type`. The transition geometry (one
  `BlockTypeStateMachine` per channel), independent/MS-stereo coupling,
  and the lookahead-granule anticipation all mirror the energy path; only
  the attack signal differs. The mode requires
  `enable_model2_psychoacoustics` (it reuses the same per-channel Model 2
  states) and runs the analysis in the block-type pre-pass, **caching the
  `Model2Layer3Granule` output so Pass 1 reuses it for the outer-loop
  `xmin(sb)`** — the §D.2.1 FFT history advances exactly once per granule,
  never twice. The lookahead granule is peeked from a cloned state so the
  borrowed next-frame PCM never perturbs the committed history. Mutually
  exclusive with the energy-detector auto path and the force toggles
  (arming any clears the others); disarming Model 2 (via
  `set_per_band_xmin`) disarms this too. Inspect with
  `auto_block_type_model2_enabled()`; turn off with
  `disable_auto_block_type_model2()`. New error
  `StreamEncodeError::Model2BlockTypeWithoutModel2` when armed without
  Model 2. Eight new lib tests: the require-Model-2 / mutual-exclusion /
  disarm-on-Model-2-disarm API paths, steady-tone-stays-long,
  valid-§C.1.5.2-sequence emission, per-band-`xmin`-still-installed, and
  two end-to-end correspondence tests proving the emitted block types
  equal the §C.1.5.2 scheduler walk over the captured `pe > 1800` attacks
  (single-frame and multi-frame, scheduler state carried across frame
  boundaries). Truth from
  `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md` (§C.1.5.3.2.1
  spreading function) and the staged ISO PDF §C.1.5.2 / §C.1.5.3.2
  window-switching prose; no new tables required. No emitted bytes change
  unless the new mode is explicitly armed.

- encoder: **Captured §C.1.5.3.2.1 Model 2 window-switching decision per
  granule** (r294, Phase 2 step 91). The automatic Model 2 mode
  (`enable_model2_psychoacoustics`) already runs each granule's PCM
  through the channel's continuous-history Model 2 state to derive the
  per-band `xmin(sb)` outer-loop threshold; that same walk also computes
  the §C.1.5.3.2.1 psychoacoustic entropy `pe` and its `pe > 1800`
  short-block switching condition (the §C.1.5.3.2 deliverable the spec
  defines for window switching), which the encode loop previously
  discarded. The Pass-1 walk now retains both into a per-(granule,
  channel) matrix committed to the encoder after each frame, exposed by
  the new
  `Mp3Encoder::last_model2_window_switch(gr, ch) -> Option<Model2WindowSwitch>`
  accessor (`Model2WindowSwitch { pe, attack }`). The decision reflects
  exactly the last frame assembled; it returns `None` before any frame
  is encoded under the armed mode, for out-of-range `gr`/`ch` (e.g.
  `gr == 1` on a single-granule LSF frame, or `ch >= nch`), and once the
  mode is disarmed (installing a static per-band vector clears the
  capture). This surfaces the spec-canonical switching signal for
  inspection and as the foundation for a future Model-2-driven
  auto-block-type path — **no bytes change** in any current
  configuration (the emitted block type is still governed by the
  attack-detector auto path or the force toggles). Four new lib tests
  cover the populated / finite / `attack == pe > 1800` invariants,
  signal-dependence across two spectra, the `None` pre-frame and
  out-of-range cases, and the disarm-clears-capture path.

- decode: **Free-format (`bitrate_index == 0`) frame decode through the
  `oxideav_core::Decoder` trait** (r292). The trait wiring previously
  rejected any free-format frame because `Mp3FrameHeader::frame_len`
  yields `None` for such a header (the ISO/IEC 11172-3 §2.4.2.3 length
  formula requires a fixed `bitrate_index ∈ 1..=14`). Since each inbound
  `Packet.data` already holds exactly one complete frame, `decode_packet`
  now uses the packet length as the authoritative free-format frame
  length, rejecting only a payload-less bare 4-byte sync. The downstream
  decode is driven by `part2_3_length` from the side-info and never by
  the advertised bitrate, so a free-format frame decodes through the
  identical chain. New `tests/decoder_trait_free_format_roundtrip.rs`
  (4 tests) proves byte-exact equality between a free-format stream and
  its CBR origin (mono + stereo), plus header-property and bare-sync
  rejection checks.

- decode: **Criterion benchmark harness + ranked hotspot map for the
  Layer III decode path** (r290, depth-mode BENCH). Two self-contained
  benches under `benches/` (no committed fixtures — input PCM is
  synthesised in-bench and round-tripped through the crate's own
  `Mp3Encoder`): `decode` times the whole-stream decode of a
  pre-encoded mono stream both through the registered `Mp3CoreDecoder`
  trait object and through the bare per-stage chain (the two are within
  measurement noise); `decode_stages` isolates each decode stage
  (side-info parse, scalefactors, Huffman big-values/count1,
  requantize, alias, IMDCT, synthesis filterbank) over one captured
  20-frame / 40-granule batch. The ranking (recorded in `BENCHMARKS.md`)
  is dominated by the back-end DSP: the synthesis filterbank (~62 %) and
  the IMDCT (~31 %) together account for ~93 % of decode time, with the
  entire entropy / requantization front half under 7 %. No behaviour
  change — decoded PCM is byte-identical.

- decode: **libFuzzer hardening of the Layer III decode path** (r289,
  depth-mode FUZZ lane). A new `fuzz/` cargo-fuzz harness adds two
  targets exercising panic-freedom against malformed bitstreams:
  `decode` drives the registered `Decoder` trait
  (send/receive/flush/reset) with crafted multi-packet streams (valid
  4-byte header + attacker-controlled CRC / side-info / main-data
  slot), reaching side-info parse, Huffman big_values/count1 decode,
  the bit-reservoir `main_data_begin` lookback, and the IMDCT /
  synthesis overlap carry-over; `granule` drives the per-granule
  primitives directly (`parse_side_info` → `decode_scalefactors` →
  `decode_huffman` → `requantize` → `alias_reduce` → `imdct_granule`
  → `synth_granule`) with attacker-controlled granule parameters. Both
  ran 180 s each (≈1.09M total iterations) with zero findings; no
  decode-path defects were surfaced.

- encoder: **Model 2 psychoacoustic threshold wired into the outer loop**
  (r288, Phase 2 step 89) — the §C.1.5.3.2.1 Layer III analysis chain
  (already producing the Figure C.6.c/d `thm(sb)` masking threshold via
  `crate::psy::Model2Layer3State::process`) is now consumable as the
  outer loop's `xmin(sb)`. `XminThresholds::from_layer3_granule` maps a
  granule's per-band masking threshold (long `thm[21]` + the new short
  `thm_short[3][12]`) into the per-band outer-loop threshold vector,
  preserving every per-band ratio exactly (a single multiplicative
  rescale anchors the granule's geometric-mean threshold to
  `DEFAULT_OUTER_LOOP_THRESHOLD`, keeping the loop's convergence dynamics
  in the same dex as the LTq / uniform paths). Silent (non-positive)
  bands are floored to the smallest rescaled positive threshold so a
  quiet band never reads `xmin = 0`; a fully silent granule yields the
  uniform default. `Mp3Encoder::set_per_band_xmin_from_model2(state,
  granule)` is the end-to-end convenience: it runs one 576-sample
  granule through a caller-owned `Model2Layer3State` (threaded across
  granules for the §D.2.1 FFT-history requirement, one per channel) and
  installs the result, replacing the *signal-independent*
  threshold-in-quiet bowl with the *signal-dependent* masking threshold.
  Restricted to the three staged Annex D Model 2 rates (32 / 44.1 /
  48 kHz); other rates and non-576 granules return the new
  `StreamEncodeError::Model2AnalysisUnsupported`. `Model2Layer3Granule`
  gains a `thm_short` field carrying the short-path per-band threshold
  (previously only the `ratio_short` quotient was exposed). Five new unit
  tests cover ratio preservation, the geometric-mean scale anchor, the
  silent / zero-band floor, the outer-loop guard, the rate / granule
  guards, and an end-to-end install producing a spectrally-shaped (not
  flat) threshold.
- encoder: **LSF (MPEG-2 / MPEG-2.5) auto block-type** (r287, Phase 2
  step 88) — `Mp3Encoder::enable_auto_block_type` /
  `enable_auto_block_type_with_mixed` now accept the LSF sample rates
  (16 / 22.05 / 24 kHz) and the MPEG-2.5 rates (8 / 11.025 / 12 kHz)
  instead of returning `LsfUnsupported`. The §C.1.5.2 transition state
  machine and the [`crate::attack_detect`] / [`crate::mixed_classifier`]
  wiring are unchanged; only the frame walk in
  `assemble_frame_with_lookahead` was generalised over `ngr ∈ {1, 2}`.
  Per ISO/IEC 13818-3 §2.4.3.2 an LSF frame carries **one** 576-sample
  granule, so the walk now builds, per channel, an attack flag for each
  of the frame's `ngr` granules plus one lookahead granule (the next
  frame's leading granule) and steps the scheduler `ngr` times: granule
  `g` is fed `(attack[g], attack[g + 1])` so its §C.1.5.2 companion is
  the following granule's flag — a single step per LSF frame, two per
  MPEG-1 frame (reproducing the prior two-granule walk byte-for-byte).
  The §2.4.3.4.10.3 window-switching geometry is version-invariant; only
  the per-frame granule count differs. Independent stereo runs a
  per-channel detector + scheduler; MS-stereo (§2.4.3.4.9) OR-folds the
  per-channel flags into a single shared scheduler so both channels of a
  granule share window geometry. Validated by a new
  `tests/lsf_auto_block_type_roundtrip.rs` (5 tests): steady-sine stays
  Long; click-train engages Short and self-decodes; the mixed-promotion
  variant; independent-stereo per-channel sequences; and MS-stereo
  per-granule block-type agreement. The legacy
  `tests/lsf_encoder_roundtrip.rs::lsf_rejects_unported_features`
  rejection test became `lsf_auto_block_type_accepted` +
  `lsf_intensity_constructors_build`.
- encoder: **LSF (MPEG-2 / MPEG-2.5) intensity-stereo encode** (r286,
  Phase 2 step 87) — `Mp3Encoder::new_joint_stereo_is` /
  `new_joint_stereo_ms_is` / `new_joint_stereo_auto_is` (and their
  registry factories) now build on the LSF / MPEG-2.5 sample rates
  instead of returning `LsfUnsupported`. Per ISO/IEC 13818-3 §2.4.3.2
  the intensity-right channel is written with the new
  `outer_loop::INTENSITY_SCALEFAC_COMPRESS_LSF = 258`: `258 >> 1 = 129
  < 180` selects the §2.4.3.2 right-channel partition `slen =
  (3, 3, 3, 0)` / `nr_of_sfb = (7, 7, 7, 0)` (3 bits on every one of
  the 21 long bands), and `258 % 2 = 0` ⇒ `intensity_scale = 0`. 3
  bits is the smallest width holding the positions `0..=6` plus the
  illegal-position marker, and makes `7` the maximum value — the
  marker the decoder tests for. Positions are derived on the §2.4.3.2
  power-law `i0 = 2^(-1/4)` reconstruction ladder
  (`derive_intensity_position_lsf`) rather than the MPEG-1 `tan` grid.
  Long-block only (LSF is single-granule; the per-window short-block
  bound stays behind the `IntensityShortBlocksUnsupported` interlock).
  Validated by a new LSF position-grid unit test and
  `tests/lsf_intensity_roundtrip.rs` (6 tests): constructor arming,
  the `scalefac_compress = 258` + (3,3,3,0)/(7,7,7,0) wire layout,
  ≈ 20.5 dB left-channel self-decode PSNR, encode/decode determinism,
  the `'11'` MS+intensity combined mode, and MPEG-2.5 at 11.025 kHz.
- encoder: **MPEG-2 LSF + MPEG-2.5 encode support** (r285, Phase 2
  step 87) — `Mp3Encoder::new` (and the registry `make_encoder` path)
  now accepts the LSF sample rates 16 / 22.05 / 24 kHz and the
  MPEG-2.5 rates 8 / 11.025 / 12 kHz. Per ISO/IEC 13818-3 §2.4.3.2
  the LSF frame carries **one** 576-sample granule
  (`slots_per_frame` constant 72), the §2.4.1.7 LSF side-info layout
  (8-bit `main_data_begin`, no `scfsi`, 9-bit `scalefac_compress`,
  no transmitted `preflag`), the LSF §2.4.2.3 bitrate ladder
  (`LSF_L3_BITRATE_LADDER_KBPS`, 8…160 kbit/s), the LSF CBR padding
  ladder, and the 255-byte LSF `main_data_begin` reservoir cap.
  Supported at LSF: CBR + VBR, mono / stereo / dual-channel, MS
  joint stereo (`new_joint_stereo_ms` / `new_joint_stereo_auto`),
  forced short / mixed block types, the §C.1.5.4.3 outer loop
  (writing `scalefac_compress = 399` — `OUTER_LOOP_SCALEFAC_COMPRESS_LSF`,
  whose §2.4.3.2 slen derivation (4, 4, 3, 3) over partition
  (6, 5, 5, 5) reproduces the MPEG-1 value-15 caps and part2 cost;
  an outer-loop `preflag` is folded into the long scalefactors since
  the sub-500 `scalefac_compress` ranges cannot carry the flag), CRC
  (`crc16_layer3_lsf`: header bits 16..31 + the full 72 / 136-bit
  LSF side info), and the Xing/Info carrier. Intensity stereo (the
  §2.4.3.2 `int_scalefac_compress` right-channel format) landed in
  r286; the §C.1.5.2 auto block-type scheduler landed in r287.
  Validated by `tests/lsf_encoder_roundtrip.rs` (10 tests): own-decoder
  round-trips at 56–88 dB PSNR across the LSF / MPEG-2.5 rates,
  framing + ladder assertions, CRC wire verification, VBR index
  bounds, and black-box external-decoder runs (`ffmpeg` / `mpg123`
  as opaque CLIs) recovering the test tones at the exact frequency
  and amplitude on every MPEG-2 rate, mono and stereo, MS included.
- decoder + encoder: **ISO/IEC 13818-3:1997 Table B.2 scalefactor-band
  tables** for the LSF rates (16 / 22.05 / 24 kHz, long + short),
  replacing the prior half-rate MPEG-1 placeholder mapping in
  `requantize::long_band_starts` / `short_band_starts`; the single
  transcription now also backs the Huffman `region_boundaries` and
  the encoder's region-split / intensity band walks.
  `tests/lsf_reference_pcm.rs` decodes the staged
  `layer3-mpeg2-22050-64kbps` fixture and tracks its reference
  `expected.wav` at 0.000026 normalized RMS error (sample-exact
  steady state at the canonical 1105-sample alignment lag) — the
  placeholder layout could not approach this. The MPEG-2.5 rates
  keep the documented placeholder (`MPEG-2.5-GAP.md`: the
  extension's band tables remain unpublished), which stays
  self-consistent encoder↔decoder.
- encoder: §2.4.3.4.9.3 **intensity-stereo encode** (Phase 2 step 86)
  — the last encoder "lacks" item. Three opt-in constructors:
  `Mp3Encoder::new_joint_stereo_is(bitrate, sample_rate, start_sfb)`
  (`mode_extension = '01'`), `new_joint_stereo_ms_is(…)` (`'11'`:
  §2.4.3.4.9.2 MS below the intensity bound, intensity above it per
  the §2.4.3.4.9.1 scoping) and `new_joint_stereo_auto_is(…)`
  (per-frame MS/LR picker over the below-bound lines, emitting `'11'`
  / `'01'`). Long scalefactor bands at or above `start_sfb`
  (`1..=20`, else the new `InvalidIntensityStartSfb` error) couple
  per Annex G.2 c): per-band position `is_pos[sfb] =
  NINT((12/π)·arctan(√(E_L/E_R)))` (0..=6; `E_R → 0` and silent
  bands map to 6), left channel rewritten to the combined `L + R`
  magnitude, right channel zeroed (the §2.4.3.4.9.1 zero-part). The
  right channel is forced to `scalefac_compress = 15` so the
  positions fit its scalefactor slots; all-zero bands between the
  last non-zero quantized right-channel line and the bound carry the
  illegal-position marker `7` (Annex G.2 c) so decoders deriving the
  bound from the zero-part do not intensity-decode them. Long-block
  only this round: the block-type toggles reject with the new
  `IntensityShortBlocksUnsupported` error while intensity is armed
  (per-window short-block `is_pos` is a follow-up). Registry path:
  `make_encoder_joint_stereo_is` / `make_encoder_joint_stereo_ms_is`.
  Accessors `intensity_stereo_enabled()` / `intensity_start_sfb()`.
  Validated by 8 new lib unit tests + 8 integration tests
  (`tests/joint_stereo_intensity_roundtrip.rs`): wire bits and
  scalefactor layout, self-decode positional fidelity (reconstructed
  6 kHz |L|/|R| = 3.733 vs the `tan(5π/12) ≈ 3.732` grid), PSNR
  parity with independent-stereo / MS-only encodes (R-channel PSNR
  improves 25.7 → 29.2 dB), auto-picker flips, byte-exact encode
  determinism + bit-exact re-decode, and black-box cross-decode via
  the `ffmpeg` and `mpg123` CLI binaries (both reproduce the
  positional ratio; binaries invoked on the emitted bytes only).

### Other

- psy: §C.1.5.3.2.1 Layer III adaptation of Model 2 + §D.2.4 step m)
  pre-echo control + §C.1.5.3.2 window switching (Phase 2 step 85),
  read from the staged ISO PDF printed pp.80–95 (Tables C.7/C.8 and
  Figures C.6.a–d/C.7 transcribed from page renders). New public
  surface: the six Table C.7 threshold-calculation-partition tables
  (`Layer3PartitionLong`/`Layer3PartitionShort` + dispatchers; note
  C.7.a = 48 kHz, the reverse of the D.3 suffix order) and six Table
  C.8 partition→scalefactor-band conversion tables
  (`Layer3SfbConversion`, 21 long / 12 short bands); the dual-path
  short-FFT primitives (`MODEL2_LAYER3_SHIFT_LONG`/`_SHORT`,
  `model2_layer3_hann_window_short` /
  `model2_layer3_step_a_reconstruct_short` /
  `model2_layer3_step_b_spectrum_short`); the printed `cw(w)`
  composition `model2_layer3_cw_compose`; the `conv1`/`conv2` and
  `NMT = 6,0` / `TMN = 29,0` dB parameter overrides
  (`model2_layer3_step_h_snr_db`); the Figure C.6.b/C.6.d partition
  thresholds (`model2_layer3_long_nb` / `model2_layer3_short_nb`);
  step m) pre-echo control (`model2_layer3_step_m_thr` +
  `Model2Layer3PreEcho`, `rpelev = 2` / `rpelev2 = 16`); the
  psychoacoustic entropy `model2_layer3_pe` with the PE > 1800
  switch, the Figure C.7 state diagram (`layer3_window_state_next`),
  the Figure C.6.a delayed retrofit (`layer3_retrofit_start`) and the
  composed `Layer3WindowSwitcher`; the Figure C.6.c reduction
  (`layer3_partitions_to_sfb` / `layer3_sfb_ratio`); and the
  integrated per-granule `Model2Layer3State::process` walk (long
  ratio + 3 short-subblock ratios + PE + attack flag). 14 new unit
  tests (1038 lib total).
- psy: Annex D Model 2 §D.2.1 inputs + §D.2.4 steps a)–e) — the
  FFT-side front half of the Model 2 threshold calculation (Phase 2
  step 84), read from the staged ISO PDF printed pp.128–130. New
  public surface: `MODEL2_FFT_LEN` / `MODEL2_FFT_LINES` /
  `model2_iblen_in_range` (strict `384<iblen<640`),
  `model2_step_a_reconstruct` (1 024-sample window reconstruction),
  `model2_hann_window` (`0,5 − 0,5·cos(2π(i−0,5)/1024)`) +
  `Model2Polar` + `model2_step_b_spectrum` (window → forward FFT →
  polar `r_ω`/`f_ω` over the 513-line DC..Nyquist domain),
  `model2_step_c_predict` / `model2_step_c_predict_polar`
  (`x̂_ω = 2,0·x_ω(t−1) − x_ω(t−2)`), `model2_step_d_cw` /
  `model2_step_d_cw_lines` + `MODEL2_CW_ABOVE_LIMIT` (the
  unpredictability measure with the verbatim 0,3
  partial-calculation default), `model2_step_e_eb` /
  `model2_step_e_cb` (partitioned energy / weighted
  unpredictability), and the `Model2State` threshold-generator
  state (§D.2.1 zeroed starting point) whose `smr()` walk chains
  steps a)–l) + n) into the previously landed back half and
  returns the 32 `SMR_n` Table D.5 outputs per call. 13 new unit
  tests incl. a value-for-value bridge replay of the full walk
  against the step primitives and a steady-sine tonality/SMR
  end-to-end check.
- psy: Annex D Model 2 §D.2.4 steps h)–l) and n) — the back half of
  the Model 2 threshold calculation (Phase 2 step 83), read from the
  staged ISO PDF printed pp.131–132. New public surface:
  `MODEL2_NMT_DB`, `model2_step_h_snr_db` / `model2_step_h_snr`
  (required SNR `maximum(minval_b, tb_b·TMN_b + (1−tb_b)·NMT_b)`
  over the Table D.3 rows), `model2_step_i_bc`
  (`bc_b = 10^(−SNR_b/10)`), `model2_step_j_nb` (`nb_b = en_b·bc_b`),
  `model2_step_k_nb_lines` (per-FFT-line spread
  `nb_ω = nb_b/(ωhigh_b−ωlow_b+1)`), `model2_absthr_energy` +
  `model2_step_l_thr` / `model2_step_l_thr_lines`
  (`thr_ω = max(nb_ω, absthr_ω)` with the documented
  D.4-uncovered-line pass-through), and `model2_step_n_epart` /
  `model2_step_n_npart` / `model2_step_n_smr_db` /
  `model2_step_n_smr` (the `SMR_n` output vector over the 32 Table
  D.5 coder partitions, with the printed `width_n` narrow/wide
  split and smallest-positive minimum). 11 new unit tests incl. an
  end-to-end h)→l) chain over the full 32 kHz tables.
- psy: Annex D Model 2 Tables D.3a–c (calculation partition table)
  + Tables D.4a–c (absolute threshold table) transcribed in full
  (Phase 2 step 82) from the staged
  `docs/audio/mp3/annex-d-renders/Table-D.3*.png` / `Table-D.4*.png`
  renders. D.3a/b/c: 49/57/58 partitions × (ωlow, ωhigh, bval,
  minval, TMN), contiguous over FFT lines 1…513; D.4a/b/c: 132/130/126
  line-range absthr rows over lines 1…480 / 1…464 / 1…428. New public
  surface: `Model2PartitionEntry`, `MODEL2_PARTITION_D3A…D3C`,
  `model2_partition_table(fs)`, `model2_bval(fs)` (feeds the step-81
  step-f) reductions), `model2_partition_index_for_line`,
  `Model2AbsThrEntry`, `MODEL2_ABSTHR_D4A…D4C`,
  `model2_absthr_table(fs)`, `model2_absthr_for_line`. Printed-spec
  quirks kept verbatim and pinned by tests: the D.4a `57 | 57` row
  (line 58 uncovered), the D.4c 4-line `329 | 332` group, and the
  D.4↔D.1 (Layer II) cross-table print differences — D.4a 51,03 vs
  51,04 at 15 kHz, and the systematic 44,1 kHz divergence (14 shared
  lines 0,01 dB lower in D.4b; 69,13 dB plateau vs D.1e's 68,00 dB).
  +9 unit tests (998 lib), including the D.4↔D.1 shared-line
  agreement check and a `bval`-vs-D.1-Bark consistency guard.
- psy: Annex D Model 2 §D.2.3 base spreading function + §D.2.4
  step f) convolution/renormalization + step g) tonality index
  (Phase 2 step 81) — the first base-Model-2 increment, unblocked by
  the docs-file erratum fix restoring the `x` term in the printed
  `sprdngf = 10^((x + tmpy)/10)` envelope. New public surface:
  `model2_sprdngf_tmpx` / `model2_sprdngf_x_db` /
  `model2_sprdngf_tmpy_db` / `model2_sprdngf` (Bark-valued, with the
  verbatim `tmpy < −100` → 0 cutoff applied to `tmpy` alone) +
  `MODEL2_SPRDNGF_TMPY_CUTOFF_DB`, `model2_step_f_spread` (one
  reduction serving both printed convolutions `ecb_b`/`ct_b` over a
  caller-injected Tables-D.3 `bval` column — the full D.3a–c tables
  remain a PNG transcription gap), `model2_step_f_rnorm` (reciprocal
  spreading row sum; printed `bb=0` lower bound vs D.2.2's
  partitions-start-at-1 noted, slice API satisfies both),
  `model2_step_f_cb` (`ct_b/ecb_b`, documented zero-energy
  convention), `model2_step_f_en`, and `model2_step_g_tonality`
  (`−0,299 − 0,43·ln(cb_b)` clamped to `[0, 1]`). +12 unit tests
  (989 lib) anchored on the Table D.3a 20-row `bval` text
  transcription, including the diagonal-unity identity
  (`15,811389 = 17,5·sqrt(1+0,474²) − 7,5·0,474`), the −8 dB
  parabola floor, the upward/downward reach asymmetry, the
  impulse-recovers-`sprdngf`-row convolution, and the
  uniform-energy `en ≈ 1` renormalization identity.
- psy: Annex D Tables D.1a–f transcription + Step 4 → Bark bridge +
  end-to-end §D.1 Step 5 sieve (Phase 2 step 80). All 704 rows × 3
  columns of the six "Frequencies, critical band rates and absolute
  threshold" pages (D.1a/b/c Layer I 108/106/102 rows, D.1d/e/f
  Layer II 132/130/126 rows) transcribed from the staged
  `docs/audio/mp3/annex-d-renders/Table-D.1*.png` renders into
  `Model1ThresholdEntry` constants `MODEL1_THRESHOLD_D1A…D1F` with a
  `model1_threshold_table(layer, fs)` dispatcher. New subsampling
  maps `model1_d1_line_for_index` / `model1_d1_index_for_line`
  (1:1/2:1/4:1/8:1 row→line regions per the printed frequency grid;
  nearest-entry inverse, tie-down) and `model1_d1_entry_for_line`;
  `model1_masker_from_component` lifts the step-79
  `Model1Step4Component` lists onto the Table D.1 Bark grid, and
  `model1_step5_components` composes bridge + Step 5(a) LTq screen +
  Step 5(b) 0,5-Bark tonal decimation — Model 1 Steps 1–7 now chain
  end-to-end (pinned by a sine → FFT → Step 4 → Step 5 → Step 6/7
  test). Cross-table redundancy (frequency = line grid for all 704
  rows; Layer I rows reprint in same-Fs Layer II tables; D.2
  boundary rows equal the D.1 rows they cite) resolved the D.2e
  band-17 illegible Bark digit (D.1e row 62 prints `16,110` — the
  stored `16.11` is exact; the docs prose estimate `16,116` is
  wrong) and surfaced a systematic 44,1 kHz print inconsistency
  (D.2b 17/20/24 = D.2e 19/22/26 print 0,001 Bark below the
  double-printed D.1 values; both verbatim prints kept, exception
  list pinned in tests). +10 unit tests (977 lib).
- psy: Annex D Model 1 §D.1 Step 4 "Finding of tonal and non-tonal
  components" (Phase 2 step 79). New public primitives:
  `model1_step4_is_local_maximum(x, k)` (verbatim operation (a)
  `X(k) > X(k-1) and X(k) >= X(k+1)`; `None` at neighbourless edges),
  `model1_step4_tonal_check_offsets(layer, k)` (the verbatim
  layer/k-range `j` table: `±2` for `2<k<63`, `±2..±3` for
  `63<=k<127`, `±2..±6` for `127<=k<=250` Layer I / `<255` Layer II,
  `±2..±12` for `255<=k<=500` Layer II; `None` outside),
  `model1_step4_is_tonal(x, layer, k)` (conjunctive
  `X(k) − X(k+j) >= 7 dB` over the whole set —
  `MODEL1_STEP4_TONAL_DELTA_DB`), `model1_step4_tonal_spl_db(x, k)`
  (verbatim three-line power sum `X_tm(k)`),
  `model1_step4_extract_tonal(&mut x, layer)` (operation (b) listing
  + the "set to −∞ dB" zeroing of each examined `k ± j_max` range,
  decisions evaluated against the pre-zeroing spectrum),
  `model1_step4_band_line_spans(layer, fs)` (Tables D.2a–f boundaries
  mapped to raw step-77 line spans via `k = round(f·N/Fs)`; carrier
  `Model1Step4BandSpan`), `model1_step4_non_tonal_components(x,
  layer, fs)` (operation (c) per-critical-band residue power at the
  geometric-mean line `round(sqrt(k_first·k_last))`), and the
  end-to-end `model1_step4_components(x, layer, fs)` returning the
  `(tonal, non_tonal)` `Model1Step4Component` lists (index `k`, SPL,
  tonal/non-tonal flag — the spec's three listed parameters). +17
  unit tests (967 lib) including a Step 1 → normalize → Step 4
  pure-tone chain.
- psy: Annex D Model 1 §D.1 Step 2 "Determination of the sound
  pressure level" (Phase 2 step 78). New public primitives
  `model1_step2_scf_term_db(scf_max)` (verbatim `20·log(scf_max·32 768)
  − 10` dB; `MODEL1_STEP2_FULL_SCALE` / `MODEL1_STEP2_PEAK_RMS_CORRECTION_DB`
  constants), `model1_step2_lsb_db(x_subband_db, scf_max)` (the verbatim
  outer `Lsb(n) = MAX[X(k), scf-term]` shared by both Step 2 methods),
  `model1_step2_xspl_db(lines)` (verbatim alternative-method
  `Xspl(n) = 10·log10(Σ 10^(X(k)/10))` dB power sum), and the Table
  D.5-driven subband selectors `model1_step2_subband_max_line_db(x, n)` /
  `model1_step2_subband_xspl_db(x, n)` mapping partition `n ∈ 1..=32`'s
  inclusive 1-based `[ωlow_n, ωhigh_n]` span onto a 513-line step-77
  half-spectrum via `k = ω − 1` (`None` for out-of-range `n` or any
  other spectrum length). Together with step 77 this produces the
  `lsb_per_partition` values that the step-70/71 SMR vectors consume.
  +8 unit tests (950 lib).
- psy: Annex D Model 1 §D.1 Step 1 "FFT Analysis" Hann-windowed
  power-density spectrum (Phase 2 step 77). New public primitives
  `model1_hann_window(i, n)` (verbatim `h(i) = sqrt(8/3)·0,5·{1 −
  cos[2·π·i/N]}`, `None` outside `0 <= i <= N−1`),
  `model1_power_density_spectrum(s)` (verbatim `X(k) = 10·log10
  |(1/N)·Σ h(l)·s(l)·e^(−j·k·l·2π/N)|² dB` over the inclusive
  half-spectrum `k = 0…N/2`; only the spec transform lengths 512 /
  1 024 accepted — `MODEL1_FFT_LEN_LAYER1` / `MODEL1_FFT_LEN_LAYER2`;
  513 output lines for the 1 024-sample block matching the Table D.5
  1-based ω ∈ 1..=513 convention via `k = ω − 1`), and
  `model1_normalize_to_96db_spl(&mut x)` (verbatim "maximum value
  corresponds to 96 dB" reference shift, `MODEL1_SPL_REFERENCE_DB`;
  returns the applied offset, `None` when no finite maximum exists).
  The DFT kernel is a private radix-2 in-place FFT cross-checked in
  tests against a direct evaluation of the spec formula; pure-tone /
  DC anchors (`10·log10(1/6)` peak, ±1 Hann sidelines exactly
  `10·log10 4` down, `X(0) = 10·log10(2/3)` for `s ≡ 1`) and window
  unit-power (`Σ h(i)² = N`) are pinned. +12 unit tests (942 lib).
- psy: Annex C §C.1.5.2.7 "Bit allocation" step-4 budget update +
  iterate/terminate test (Phase 2 step 76). Steps 73–75 selected the
  minimal-MNR subband, promoted its Table B.2 entry, and recomputed its
  MNR; this step closes the iteration with the loop's fourth verbatim
  action — "bspl is updated according to the additional number of bits
  required. If a non-zero number of bits is assigned to a subband for the
  first time, bsel has to be updated, and bscf has to be updated according
  to the number of scalefactors required for this subband" — and the
  recompute `adb = cb − (bhdr + bcrc + bbal + bsel + bscf + bspl + banc)`
  (printed p.74, verbatim), saturating at zero. New public structs
  `BitAllocBudget { bspl, bsel, bscf, first_time, adb }` and
  `BitAllocOverhead { cb, bhdr, bcrc, bbal, banc }` and free functions
  `bit_allocation_budget_update(prev, extra_sample_bits, first_time,
  sel_bits, scf_bits, overhead) -> BitAllocBudget` and
  `bit_allocation_should_iterate(adb, max_possible_increase) -> bool`,
  the latter the verbatim termination predicate "The iterative procedure
  is repeated as long as adb is not less than any possible increase of
  bspl, bsel and bscf within one loop" (`adb >= max_possible_increase`).
  Per-entry sample-bit / scalefactor-bit costs (Tables B.2 / B.4) and the
  fixed overhead terms are caller-injected (behind the numeric-table
  transcription gap), the dependency-injection pattern the surrounding
  Phase 2 steps use. Tests: 930 lib (was 918 baseline; +12 unit). Only
  ISO/IEC 11172-3:1993 Annex C §C.1.5.2.7 (printed p.74) was read; no
  external implementation was consulted.
- psy: Annex C §C.1.5.2.7 "Bit allocation" recompute-new-MNR loop action
  (Phase 2 step 75). Phase 2 step 74 (r273) advanced the minimal-MNR
  subband's Table B.2 entry to the next-higher quantization accuracy; this
  step performs the loop's next verbatim action, "The new MNR of this
  subband is calculated" (printed p.71, verbatim), recomputing the
  promoted subband's mask-to-noise ratio with the §C.1.5.2.7 definition
  `MNR = SNR − SMR`. The `SNR_n` is the Table C.5 *Layer II
  Signal-to-Noise Ratios* value for the **advanced** entry; the `SMR_n` is
  the unchanged psychoacoustic-model output the loop re-reads each
  iteration, so the new `MNR_n` rises (for a monotone Table C.5 column),
  removing the subband from its "greatest benefit" position. New public
  struct `CoderPartitionD5RecomputedMnr { subband, entry, mnr_db, smr_db }`
  and free function `bit_allocation_recompute_mnr(promotion, smr_db,
  snr_for_entry) -> CoderPartitionD5RecomputedMnr`: takes the step-74
  `BitAllocPromotion`, the selected subband's carried-through `SMR_n`, and
  a callback returning the Table C.5 `SNR_n` for the promotion's
  post-advance entry, and returns the single verbatim subtraction
  `snr_for_entry(promotion.entry) − smr_db`. The Table C.5 `SNR_n` is
  caller-injected (Table C.5 is behind the same numeric-table
  transcription gap as Tables B.2 / D.1 / D.2, the dependency-injection
  pattern the surrounding Phase 2 steps use); no spec arithmetic is
  introduced beyond the `SNR − SMR` subtraction. A saturated step-74
  promotion (top entry, no advance) recomputes at the held entry — an
  idempotent re-evaluation. Tests: 918 lib (was 909 baseline; +9 unit)
  covering the advanced-entry MNR, SMR pass-through, subband/entry echo,
  monotone-column MNR rise, saturated-promotion hold, single SNR-callback
  invocation for the post-promotion entry only, cell-wise identity with
  the step-72 `MNR = SNR − SMR` definition, negative-SMR lift, and
  determinism. Only the §C.1.5.2.7 "The new MNR of this subband is
  calculated" loop step and `MNR = SNR − SMR` definition (ISO/IEC
  11172-3:1993 Annex C, printed p.71) and the Phase 2 step 74
  `bit_allocation_promote_entry` result it consumes are read.
- psy: Annex C §C.1.5.2.7 "Bit allocation" next-higher-entry quantization
  promotion (Phase 2 step 74). Phase 2 step 73 (r272) selected the
  minimal-MNR subband — the one "that has the greatest benefit"; this step
  performs the loop's next verbatim action, "The accuracy of the
  quantization of the subband with the minimal MNR is increased by using
  the next higher entry in the relevant table B.2, Layer II Possible
  Quantization per subband" (printed p.71, verbatim). New public struct
  `BitAllocPromotion { subband, entry, advanced }` and free function
  `bit_allocation_promote_entry(subband, prev_entry, entry_count) ->
  BitAllocPromotion`: advances the selected subband's Table B.2 column
  entry index by one when a next-higher entry exists
  (`prev_entry + 1 < entry_count` → `entry = prev_entry + 1`,
  `advanced = true`), otherwise holds the prior entry and reports
  `advanced = false` (top-entry saturation, single-entry columns, and
  `entry_count == 0` all yield no advance). The Table B.2 column length is
  caller-injected (the table is behind the same numeric-table
  transcription gap as Tables C.5 / D.1 / D.2, the dependency-injection
  pattern the surrounding Phase 2 steps use); the B.2 entry *values* are
  not consulted — only the column index is advanced. No spec arithmetic is
  introduced beyond the `prev_entry + 1` increment and the bound
  comparison. Tests: 909 lib (was 900 baseline; +9 unit) covering
  mid-column advance, bottom-entry advance, top-entry saturation,
  single-entry and zero-entry columns never advancing, subband-index
  pass-through, penultimate-then-top walk, a one-per-call climb to the top
  entry, and determinism. Only the §C.1.5.2.7 "increased by using the next
  higher entry in the relevant table B.2" loop step (ISO/IEC 11172-3:1993
  Annex C, printed p.71) and the Phase 2 step 73
  `coder_partition_d5_min_mnr` selection it consumes are read.
- psy: Annex C §C.1.5.2.7 "Bit allocation" minimal-MNR subband selection
  (Phase 2 step 73). Phase 2 step 72 (r271) landed the per-partition
  `MNR_n = SNR_n − SMR_n` row-order vector; this step performs the first
  action of every bit-allocation iteration loop — "Determination of the
  minimal MNR of all subbands" (printed p.71, verbatim) — reducing the
  32-row vector to the single subband "that has the greatest benefit",
  which the loop then promotes to the next-higher quantization-accuracy
  entry. New public struct
  `CoderPartitionD5MinMnr { partition_n, mnr_db, smr_db, width_n }` and
  free function
  `coder_partition_d5_min_mnr(&[CoderPartitionD5Mnr; 32]) ->
  CoderPartitionD5MinMnr`: a row-order argmin scan over the step-72
  vector, carrying the winning partition's `mnr_db` / `smr_db` /
  `width_n` columns through verbatim and returning its 1-based partition
  index `n ∈ 1..=32`. Ties resolve to the lowest partition index (the
  spec selects "the" subband, so a deterministic tie-break is required —
  the row-order scan keeps the first occurrence), and `NaN` rows are
  skipped (never compare `<` the running minimum). No spec arithmetic is
  introduced beyond the `<` comparisons of the scan. Tests: 900 lib (was
  891 baseline; +9 unit) covering unique-minimum selection under a
  −30 dB interior-line LTg dip, low/high partition-index minima,
  lowest-index tie-break, width/SMR column pass-through of the winner,
  negative-minimum selection, `NaN`-row skipping, idempotence, and a
  brute-force argmin cross-check. Only the §C.1.5.2.7 "Determination of
  the minimal MNR of all subbands" loop step (ISO/IEC 11172-3:1993 Annex
  C, printed p.71) and the Phase 2 step 72
  `coder_partition_d5_mnr_row_order` vector it consumes are read; no
  external implementation was consulted.
- psy: Annex C §C.1.5.2.7 "Bit allocation" per-partition mask-to-noise
  ratio `MNR_n = SNR_n − SMR_n` row-order vector over Table D.5 (Phase 2
  step 72). Phase 2 step 71 (r270) exposed the §D.1 Step 9 paired
  `(SMR_n, width_n)` vector — "the output of the psychoacoustic model"
  the §C.1.5.2.7 bit-allocation loop consumes. This step lands the very
  first arithmetic of that iterative procedure: the per-subband `MNR`
  initialisation `MNR = SNR − SMR` (printed p.73, verbatim), computed
  once per partition before the loop's level-bumping begins. New public
  struct `CoderPartitionD5Mnr { mnr_db, smr_db, width_n }` and free
  function `coder_partition_d5_mnr_row_order<S, L, F>(snr_per_partition,
  lsb_per_partition, ltg_per_line) -> [CoderPartitionD5Mnr; 32]`: a pure
  per-row subtraction of a caller-supplied `SNR_n` from Phase 2 step
  71's `coder_partition_d5_smr_row_order` SMR column, carrying the
  `smr_db` and `width_n` columns through verbatim. The `SNR_n` term is
  the Table C.5 "Layer II Signal-to-Noise Ratios" column — behind the
  same numeric-table transcription gap as Tables D.1 / D.2 — so it is
  caller-injected, the same dependency-injection pattern Phase 2 steps
  58–71 use for the §D.1 Step 2 `Lsb(n)` term. No spec arithmetic is
  introduced beyond the verbatim `SNR − SMR` subtraction; the `smr_db`
  column is bit-identical to step 71's and the `width_n` column to step
  60's (`[0×12, 1×20]`). Tests: 891 lib (was 880 baseline; +11 unit)
  covering the 32-row length, zero-callback all-zero MNR, uniform pin
  (30 − 76 = −46.0 exact), cell-wise `MNR = SNR − SMR` against the
  step-71 SMR column, SMR/width column pass-through from step 71, the
  `[0×12, 1×20]` width literal, partition-index mapping
  (`SNR(n) = n`, flat SMR → `mnr_db[i] = i + 1`), the §C.1.5.2.7
  "subband with the greatest benefit" = unique minimum-MNR argmin under
  a −30 dB interior-line LTg dip, SNR fan-out once-per-partition
  ascending, sign semantics in both directions (needs-bits vs already-
  protected), and idempotence for pure callbacks.
  Only only the staged ISO/IEC 11172-3:1993 spec
  PDF (Annex C §C.1.5.2.7, printed p.73; §D.1 Step 9, printed p.115)
  and the Phase 2 step 71 row-order accessor (and through it the Table
  D.5 transcription in
  `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`) are read.
- psy: Annex D Model 1 §D.1 Step 9 paired `(SMR_n, width_n)` row-order
  vector over Table D.5 (Phase 2 step 71). Phase 2 step 70 (r269)
  exposed the bare row-order signal-to-mask-ratio vector
  `[SMR_1 … SMR_32]` (dB); the Layer I / Layer II bit-allocation loop
  reads each partition's `SMR_n` paired with its `width_n` column flag
  in lockstep at every row (SMR seeds the partition's mask-to-noise
  iteration; `width_n` drives single-line vs multi-line per-partition
  bit targeting) — the SMR analogue of the §D.1 Step 8 paired
  `(LTmin_n, width_n)` vector (Phase 2 step 61). New public struct
  `CoderPartitionD5Smr { smr_db: f64, width_n: u16 }` and free function
  `coder_partition_d5_smr_row_order<L: Fn(u16) -> f64,
  F: Fn(u16) -> f64>(lsb_per_partition, ltg_per_line) ->
  [CoderPartitionD5Smr; 32]`: a pure index-aligned zip of Phase 2 step
  70's `coder_partition_d5_smr_db_row_order` with Phase 2 step 60's
  static `coder_partition_d5_width_row_order`. No spec arithmetic is
  introduced beyond the step 70 subtraction already in the SMR column —
  only the per-row pairing of the two existing columns at the same
  array index. The `smr_db` column is bit-identical to step 70's output
  and the `width_n` column to step 60's (`[0×12, 1×20]`); `Lsb(n)`
  stays the steps-58–70 caller-callback injection (§D.1 Steps 1–2
  remain behind the PNG-only Tables D.1 / D.2 transcription gap).
  Tests: 880 lib (was 868 baseline; +12 unit) covering the 32-pair
  length, zero-callback all-zero SMR, uniform pin (96 − 20 = 76.0
  exact), cell-wise SMR equality with step 70, width equality with
  step 60 across two callbacks, the `[0×12, 1×20]` width literal,
  partition-index mapping (`Lsb(n) = n`, flat threshold →
  `smr_db[i] = i + 1`), sign semantics in both directions, the step-61
  paired-pattern cross-check (matching `width_n` and
  `lsb(n) − step61.ltmin_db` per row), Lsb fan-out once-per-partition
  ascending with LTg fan-out equal to one step-59 pass, a −30 dB
  interior-line LTg dip (ω = 300) raising exactly one row's SMR by
  +30 dB with widths and all 31 other rows unchanged, and idempotence
  for pure callbacks.1 Step 9, printed p.115)
  and the Phase 2 step 70 / step 60 row-order accessors (and through
  them the Table D.5 transcription in
  `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`) are read.
- psy: Annex D Model 1 §D.1 Step 9 row-order signal-to-mask-ratio
  vector `[SMR_1 … SMR_32]` (dB) over Table D.5 (Phase 2 step 70).
  Phase 2 step 69 landed the Step 9 subtraction
  `SMR_n = Lsb(n) − LTmin_n` dB (ISO/IEC 11172-3:1993 Annex D,
  printed p.115) in the width-gated split presentation; the Layer I /
  Layer II bit-allocation loop walks the 32 coder partitions **in
  row order**, pairing each partition's `SMR_n` with the same row's
  `width_n` flag (Phase 2 step 60's `[u16; 32]`) and `LTmin_n` value
  (Phase 2 step 59's `[f64; 32]`) at the same array index. New free
  function `coder_partition_d5_smr_db_row_order<L: Fn(u16) -> f64,
  F: Fn(u16) -> f64>(lsb_per_partition, ltg_per_line) -> [f64; 32]`
  supplies the missing row-order presentation: element `i` carries
  `SMR_{i + 1}`, the `LTmin_n` operand comes from one Phase 2 step
  59 pass (`coder_partition_d5_ltg_min_row_order`), and the `Lsb(n)`
  operand (§D.1 Step 2 sound pressure level, printed p.110) stays
  the steps-58–69 caller-callback injection (Steps 1–2 remain behind
  the PNG-only Tables D.1 / D.2 transcription gap). No new spec
  arithmetic beyond the step 69 subtraction. Because step 63's
  by-width `LTmin_n` cells are index-preserving copies of the step
  59 row-order vector (via the step 61 / 62 chain), the output is
  bit-identical to step 69's split read back in row order
  (`out[0..12] == narrow_band`, `out[12..32] == wide_band`), pinned
  by an exact-`==` test under non-trivial callbacks. Tests: +9 unit
  covering zero-callback all-zero rows, uniform pin (96 − 20 = 76.0
  exact), cell-wise equality with the independently reconstructed
  `lsb(n) − step59` difference, partition-index mapping
  (`Lsb(n) = n`, flat threshold → `out[i] = i + 1`), sign semantics
  in both directions, dual callback fan-out (Lsb exactly `[1..=32]`
  ascending; LTg equal to a directly-counted step-59 pass), the step
  69 bit-identity, a −30 dB interior-line LTg dip (ω = 300) raising
  exactly one row's SMR by +30 dB with all 31 other rows unchanged,
  and idempotence for pure callbacks.
- psy: Annex D Model 1 §D.1 Step 9 width-gated signal-to-mask-ratio
  `SMR_n = Lsb(n) − LTmin_n` (dB) over Table D.5 (Phase 2 step 69).
  The Step 9 formula (ISO/IEC 11172-3:1993 Annex D, printed p.115)
  computes the signal-to-mask ratio for every subband `n` — the
  per-band input the Layer I / Layer II bit-allocation loop seeds
  its mask-to-noise iteration from. Previously carried as
  "docs-blocked on the §D.1 SMR formula"; verified directly
  derivable from the staged spec PDF
  (`docs/audio/mp3/ISO_IEC_11172-3-MP3-1993.pdf`, §D.1 Step 9
  printed p.115 + Step 8 printed p.114), closing the gap. New free
  function `coder_partition_d5_smr_db_row_order_by_width<L:
  Fn(u16) -> f64, F: Fn(u16) -> f64>(lsb_per_partition,
  ltg_per_line) -> CoderPartitionD5SmrByWidth { narrow_band:
  [f64; 12], wide_band: [f64; 20] }`. The `LTmin_n` operand comes
  from one Phase 2 step 63 invocation
  (`coder_partition_d5_ltmin_db_row_order_by_width`); the `Lsb(n)`
  operand (§D.1 Step 2 sound pressure level, printed p.110) enters
  as a caller-supplied per-partition callback because Steps 1–2
  (FFT + SPL) remain behind the PNG-only Tables D.1 / D.2
  transcription gap — the same dependency-injection pattern steps
  58–68 use for `LTg(ω)`. The only new spec arithmetic is the
  Step 9 subtraction itself, one `Lsb(n) − LTmin_n` per row,
  presented in the step 63 width-gated split (narrow partitions
  1..=12, wide 13..=32). `lsb_per_partition` is invoked exactly
  once per partition `n ∈ 1..=32` in ascending row order;
  `ltg_per_line` fan-out is exactly one step-63 pass. Signs
  preserved without clipping (positive = audible content needing
  bits; negative = fully masked). Tests: +8 unit covering
  zero-callback all-zero cells, uniform pin (96 − 20 = 76.0
  exact), cell-wise equality with the independently reconstructed
  `lsb(n) − step63` difference, partition-index mapping
  (`Lsb(n) = n`, flat threshold → `narrow[i] = i + 1`,
  `wide[j] = j + 13`), sign semantics in both directions, dual
  callback fan-out (Lsb exactly `[1..=32]` ascending; LTg equal to
  a directly-counted step-63 pass), a −30 dB interior-line LTg dip
  (ω = 300) raising exactly one wide cell's SMR by +30 dB with all
  31 other cells unchanged, and idempotence for pure callbacks.
- psy: Annex D Model 1 §D.1 Step 8 width-gated paired
  `(narrow_total, wide_total)` signed bit-budget reduction over
  Table D.5 with a single step-65 invocation (Phase 2 step 68).
  Phase 2 step 66 (r265) exposed the wide-band weighted total
  `Σ_{n=1..=32} width_n · log2(LTmin_lin_n)` (collapsing onto
  `Σ wide_band`) and Phase 2 step 67 (r266) the complementary
  narrow-band total `Σ_{n=1..=32} (1 − width_n) · log2(LTmin_lin_n)`
  (collapsing onto `Σ narrow_band`); the two partition the full
  row-order `Σ_n log2_n` exactly. Several Step 9 / Step 10 consumers
  read *both* totals together, and calling step 66 + step 67
  back-to-back invokes the caller's `LTg(ω)` callback **twice** over
  the full `Σ_{n=1..=32} (ωhigh_n − ωlow_n + 1)` FFT-line range,
  because each total independently re-derives step 65's split
  struct. A new free function
  `coder_partition_d5_ltmin_log2_paired_bit_budget_totals<F:
  Fn(u16) -> f64>(ltg_per_line) -> (f64, f64)` fuses the two: it
  calls Phase 2 step 65's
  `coder_partition_d5_ltmin_log2_row_order_by_width` **once**, then
  sums the `narrow_band` and `wide_band` subarrays of the single
  returned struct independently, returning `(narrow_total,
  wide_total)`. The callback fan-out is exactly half the
  back-to-back step 67 + step 66 pairing — one pass over the
  FFT-line range instead of two — while the two scalars are
  bit-identical to the standalone step 67 / step 66 results. No new
  spec arithmetic is introduced beyond `+`. Tests: +6 unit covering
  zero-dB pair `(0.0, 0.0)`, bit-identity with standalone steps 67 /
  66, callback fan-out exactly one step-65 pass (= half the
  back-to-back standalone count), `narrow + wide` recovering the
  full row-order sum, block independence (a wide-only perturbation
  moves only `wide_total`, a narrow-only perturbation only
  `narrow_total`), and idempotence for a pure callback.
- psy: Annex D Model 1 §D.1 Step 8 width-gated wide-band signed
  bit-budget reduction `Σ_{n=1..=32} width_n · log2(LTmin_lin_n)`
  over Table D.5 (Phase 2 step 66). Phase 2 step 65 (r264) projected
  the width-gated per-band `LTmin_n` column onto its
  `log2`-of-linear-energy presentation, exposing two per-band
  subarrays carrying the per-row `log2(10^(LTmin_n / 10))` value
  (`narrow_band: [f64; 12]` for `n ∈ 1..=12` with `width_n = 0`;
  `wide_band: [f64; 20]` for `n ∈ 13..=32` with `width_n = 1`).
  Several Step 9 / Step 10 consumers do not read the
  `log2(LTmin_lin_n)` column cell-by-cell — they read its weighted
  total `Σ_{n=1..=32} width_n · log2_n` where `width_n` is the row's
  width column flag. Because `width_n = 0` for every narrow row and
  `width_n = 1` for every wide row (a structural invariant of
  Table D.5 verified by Phase 2 step 60 and inherited by step 65),
  the weighted total collapses algebraically onto the unweighted
  sum of step 65's `wide_band` subarray — a 20-element strict
  reduction that introduces no new spec arithmetic beyond `+`. A
  new free function
  `coder_partition_d5_ltmin_log2_wide_band_bit_budget_total<F:
  Fn(u16) -> f64>(ltg_per_line) -> f64` calls Phase 2 step 65's
  `coder_partition_d5_ltmin_log2_row_order_by_width` once and sums
  the resulting `wide_band` subarray. The `LTg(ω)` callback is
  invoked exactly as many times as Phase 2 step 65 invokes it (one
  call per FFT line in `Σ_{n=1..=32} (ωhigh_n − ωlow_n + 1)`).
  Signed: a callback that drives wide partitions' `LTmin_n` below
  0 dB produces a strictly negative contribution at the relevant
  cells; a callback that drives `LTmin_n` above 0 dB produces a
  strictly positive contribution. The reduction accumulates both
  signs without clipping. Tests: 833 lib (was 822 baseline; +11
  unit) covering zero-dB callback total = 0.0 exactly, finiteness
  under a bounded finite-dB callback, equality with the unweighted
  `wide_band` sum under a non-trivial callback (algebraic-identity
  cross-check), narrow-row perturbations leave the total unchanged
  (width-gated reduction reads only the wide block), uniform +3 dB
  total pin (= 20 × `0.3 / log10(2)`), odd-symmetry around zero
  under sign-flipped uniform callbacks, linearity-in-dB scaling
  (uniform 10 dB total = 2 × uniform 5 dB total), idempotence for
  pure callbacks, equality with the explicit weighted full-row-order
  sum `Σ_{n=1..=32} width_n · log2_n` reconstructed from Phase 2
  step 60's row-order width vector and Phase 2 step 59's row-order
  LTmin dB vector (rescaled by `log2(10) / 10`), single-wide-line
  −20 dB dip lowers the total by exactly `2 · log2(10)` relative to
  the baseline (per-cell drop sanity check), and proportionality
  to Phase 2 step 63's dB wide-block sum by the dB →
  `log2`-of-linear constant `log2(10) / 10`.
- psy: Annex D Model 1 §D.1 Step 8 width-gated `log2(LTmin_lin_n)`
  column projection over Table D.5 (Phase 2 step 65). Phase 2 step
  64 (r263) projected the width-gated paired `(LTmin_n, width_n)`
  vector onto its linear-energy presentation, exposing two
  strictly-positive per-band subarrays
  (`narrow_band: [f64; 12]` for partitions `n ∈ 1..=12` with
  `width_n = 0`; `wide_band: [f64; 20]` for partitions `n ∈ 13..=32`
  with `width_n = 1`). Several Step 9 / Step 10 / outer-loop
  consumers do not read the per-band linear-energy threshold
  directly — they read its base-2 logarithm
  `log2(10^(LTmin_n / 10))`, the natural per-band bit-budget proxy
  in the Layer I/II bit-allocation loop (every factor-of-two change
  in linear masking energy corresponds to exactly one unit on the
  `log2` axis). The `log2` conversion is the standard `f64::log2`
  primitive — it introduces no new spec arithmetic. A new free
  function `coder_partition_d5_ltmin_log2_row_order_by_width<F:
  Fn(u16) -> f64>(ltg_per_line) -> CoderPartitionD5LtminLog2ByWidth`
  calls Phase 2 step 64's
  `coder_partition_d5_ltmin_linear_row_order_by_width` once and
  applies `cell.log2()` to each of the 12 + 20 cells, producing a
  new struct `CoderPartitionD5LtminLog2ByWidth { narrow_band:
  [f64; 12], wide_band: [f64; 20] }` whose entries are finite for
  any callback returning finite dB at every FFT line (or
  `+INFINITY` only under the degenerate condition steps 63/64
  already document). The `LTg(ω)` callback is invoked exactly as
  many times as Phase 2 step 64 invokes it (one call per FFT line
  in `Σ_{n=1..=32} (ωhigh_n − ωlow_n + 1)`). Strict monotonicity
  is preserved cell-wise (`log2` is strictly monotone on the
  positive reals). Width invariant pinned structurally (implicit
  in subarray choice). Identity with the dB column: every output
  cell equals step 63's matching dB cell multiplied by
  `log2(10) / 10 ≈ 0.33219` (a strictly-proportional rescaling).
  Tests: 822 lib (was 810 baseline; +12 unit) covering subarray
  lengths (12 / 20), zero-dB callback maps to zero (`log2(1) = 0`)
  in every cell, finiteness for any finite-dB callback, cell-wise
  equality with `log2(step64_linear)` under a non-trivial callback
  (strict-projection cross-check), spot pin at uniform +3 dB
  (cell ≈ 0.9966), odd-symmetry around zero under sign-flipped
  callbacks (uniform +3 dB and −3 dB cells sum to zero per cell),
  strict proportionality to step 63's dB column with constant
  `log2(10) / 10`, strict monotonicity under a uniform −1 dB shift
  (every cell shifts by the same constant `−log2(10)/10`),
  idempotent for a pure callback, dip in narrow band only affects
  narrow band (cross-block insulation), dip in wide band only
  affects wide band (cross-block insulation), and partition-of-step
  59 cross-check (narrow ++ wide log2 cells, scaled back to dB by
  `10 · log10(2)`, equal step 59's row-order LTmin vector).
- psy: Annex D Model 1 §D.1 Step 8 width-gated `LTmin_n` column
  projection over Table D.5 converted to linear energy
  (`10^(LTmin_n / 10)`) (Phase 2 step 64). Phase 2 step 63 (r262)
  projected the width-gated paired `(LTmin_n, width_n)` vector onto
  its `ltmin_db` field, exposing two per-band dB subarrays
  (`narrow_band: [f64; 12]` for partitions `n ∈ 1..=12` with
  `width_n = 0`; `wide_band: [f64; 20]` for partitions `n ∈ 13..=32`
  with `width_n = 1`). Several Step 9 / Step 10 / outer-loop
  consumers read the per-band masking threshold in the linear energy
  domain `10^(LTmin_n / 10)` rather than in dB; the dB → linear
  conversion is the same monotone `10^(·/10)` transformation
  `db_to_xfsf_energy` already uses (line 411 of `src/psy.rs`), the
  same Step 7 `Σ 10^(LTtm/10)` global-threshold summer uses (lines
  702 / 705), and the same Model 2 Layer III spread linearisation
  uses (line 1492) — it introduces no new spec arithmetic. A new
  free function
  `coder_partition_d5_ltmin_linear_row_order_by_width<F: Fn(u16) ->
  f64>(ltg_per_line) -> CoderPartitionD5LtminLinearByWidth` calls
  Phase 2 step 63's
  `coder_partition_d5_ltmin_db_row_order_by_width` once and applies
  `(10.0_f64).powf(db / 10.0)` to each of the 12 + 20 cells, producing
  a new struct `CoderPartitionD5LtminLinearByWidth { narrow_band:
  [f64; 12], wide_band: [f64; 20] }` whose entries are strictly
  positive linear energy values (or `INFINITY` only under the
  degenerate condition step 63 already documents). The `LTg(ω)`
  callback is invoked exactly as many times as Phase 2 step 63
  invokes it (one call per FFT line in
  `Σ_{n=1..=32} (ωhigh_n − ωlow_n + 1)`). Monotonicity is preserved
  cell-wise (the conversion is strictly monotone in dB). Width
  invariant pinned structurally (implicit in subarray choice).
  Tests: 810 lib (was 799 baseline; +11 unit) covering subarray
  lengths (12 / 20), zero-dB callback linearises to unit energy
  everywhere, strict positivity for any finite callback, cell-wise
  equality with `10^(step63_db / 10)` under a non-trivial callback
  (strict-projection cross-check), spot pins at uniform +10 dB
  (factor 10) and −10 dB (factor 0.1), strict monotonicity under a
  uniform −1 dB shift (every cell shrinks by the same constant ratio
  `10^(−1/10) ≈ 0.7943`), idempotent for a pure callback, dip in
  narrow band only affects narrow band (cross-block insulation),
  dip in wide band only affects wide band (cross-block insulation),
  and a recovery test that log-maps `narrow_band ++ wide_band` back
  to dB and pins index-by-index against step 59's row-order LTmin
  vector. Implementation 18 lines (in `src/psy.rs`); tests 168 lines.

- psy: Annex D Model 1 §D.1 Step 8 width-gated `LTmin_n` column
  projection over Table D.5 (Phase 2 step 63). Phase 2 step 62 (r261)
  exposed the width-gated split of the row-order paired
  `(LTmin_n, width_n)` vector as `CoderPartitionD5ReductionByWidth`
  with `narrow_band: [_; 12]` (partitions `n ∈ 1..=12`, `width_n = 0`)
  and `wide_band: [_; 20]` (partitions `n ∈ 13..=32`, `width_n = 1`).
  Some downstream consumers (a per-band dB→linear conversion, a
  width-block bit-target-budget summation that reads only the
  threshold column) do not need the `width_n` field at the call site
  once the call site has already chosen which subarray to walk —
  `width_n` is implicit in the choice of `narrow_band` versus
  `wide_band`. A new free function
  `coder_partition_d5_ltmin_db_row_order_by_width<F: Fn(u16) -> f64>(
  ltg_per_line) -> CoderPartitionD5LtminDbByWidth` projects Phase 2
  step 62's struct onto the `ltmin_db` field of each subarray,
  producing a new struct
  `CoderPartitionD5LtminDbByWidth { narrow_band: [f64; 12],
  wide_band: [f64; 20] }`. No spec arithmetic is introduced — every
  output cell is a copy of a cell in the step 62 struct at the same
  array index. The `LTg(ω)` callback is invoked exactly as many times
  as Phase 2 step 62 invokes it (one call per FFT line in
  `Σ_{n=1..=32} (ωhigh_n − ωlow_n + 1)`). Width invariant pinned
  structurally (implicit in subarray choice). Tests: 799 lib (was 784
  baseline; +15 unit) covering subarray lengths (12 / 20), total
  length pin (12 + 20 = 32), constant callback carries the constant
  in both subarrays, narrow matches step 62 narrow field under a non-
  trivial callback (strict-projection cross-check), wide matches step
  62 wide field under a non-trivial callback (strict-projection
  cross-check), narrow matches paired-vector prefix under a non-
  trivial callback, wide matches paired-vector suffix under a non-
  trivial callback, concatenation back to step 59's row-order LTmin
  vector (partition pin), endpoints match Table D.5 edges (identity
  callback returns ωlow_n per row, pins partitions 1 / 12 / 13 / 32),
  split point pinned at partition 12 / 13 boundary, idempotent for a
  pure callback, dip in narrow band only affects narrow band (cross-
  band isolation), dip in wide band only affects wide band (dual
  cross-band isolation), and independence from width column (the
  projection reads only `ltmin_db`). Compositional — the projection
  itself adds no callback evaluations. Public struct
  `CoderPartitionD5LtminDbByWidth { narrow_band, wide_band }` added.
  Step 63 keeps the Phase 2 chain on the row-order paired vector —
  Steps 1-5 (FFT / SPL / tonality classifier) and the Layer III
  §D.2 / Model 2 calc-partition replacement remain blocked on the
  Table D.1 / D.3 PNG-only transcription gap. All truth from
  `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md` §"Table
  D.5 - Layer I and Layer II coder partition table".

- psy: Annex D Model 1 §D.1 Step 8 width-gated split of the row-order
  paired vector over Table D.5 (Phase 2 step 62). Phase 2 step 61
  (r260) exposed the row-order paired vector
  `[(LTmin_n, width_n); 32]`. Table D.5's `width_n` column is binary
  (`width_n = 0` for `n ∈ 1..=12`; `width_n = 1` for `n ∈ 13..=32`)
  and the Layer I / Layer II bit-allocation loop branches on the
  column per row — the narrow band (`width_n = 0`, partitions
  `1..=12`) drives the single-line per-partition target, the wide
  band (`width_n = 1`, partitions `13..=32`) drives the multi-line
  per-partition target. A new free function
  `coder_partition_d5_reduction_row_order_by_width<F: Fn(u16) -> f64>(
  ltg_per_line) -> CoderPartitionD5ReductionByWidth` splits Phase 2
  step 61's 32-row paired vector at the constant index 12 (the width
  column's single 0 → 1 transition) into a struct with
  `narrow_band: [CoderPartitionD5Reduction; 12]` and
  `wide_band: [CoderPartitionD5Reduction; 20]`. Each subarray
  preserves the row-order ordering of the paired vector. No spec
  arithmetic is introduced — only the re-presentation of the paired
  vector as two width-gated subarrays at the same array indices.
  Width invariant pinned structurally per side. Tests: 784 lib (was
  770 baseline; +14 unit) covering subarray lengths (12 / 20), narrow
  band carries `width_n = 0` everywhere, wide band carries
  `width_n = 1` everywhere, narrow `ltmin_db` column matches paired
  vector's first 12 entries under a non-trivial callback, wide
  `ltmin_db` column matches paired vector's last 20 entries under a
  non-trivial callback, narrow + wide concatenates back to the
  paired vector (round-trip pin), constant callback carries constant
  in both subarrays, endpoints pin partitions 1 / 12 / 13 / 32, split
  point pinned at partition 13 (`narrow_band[11]` carries partition
  12's ωlow under identity callback, `wide_band[0]` carries
  partition 13's ωlow under identity callback), width columns
  invariant across two callbacks (structural orthogonality),
  idempotent for pure callbacks, total length matches paired vector
  (12 + 20 = 32), dip in narrow band only affects narrow band
  (cross-band isolation), dip in wide band only affects wide band
  (dual cross-band isolation). Compositional — `LTg(ω)` callback
  invoked exactly once per FFT line in `Σ_{n=1..=32} (ωhigh_n −
  ωlow_n + 1)`, no extra evaluations introduced by the split. Public
  struct `CoderPartitionD5ReductionByWidth { narrow_band, wide_band }`
  added. Step 62 keeps the Phase 2 chain on the row-order paired
  vector — Steps 1-5 (FFT / SPL / tonality classifier) and the
  Layer III §D.2 / Model 2 calc-partition replacement remain blocked
  on the Table D.1 / D.3 PNG-only transcription gap. All truth from
  `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md` §"Table
  D.5 - Layer I and Layer II coder partition table".

- psy: Annex D Model 1 §D.1 Step 8 paired `(LTmin_n, width_n)`
  row-order vector over Table D.5 (Phase 2 step 61). Phase 2 step
  59 (r258) exposed the row-order LTmin vector
  `[LTmin_1, …, LTmin_32]`; Phase 2 step 60 (r259) exposed the
  row-order width vector `[width_1, …, width_32]`. The Layer I /
  Layer II bit-allocation loop walks the 32 coder partitions in
  row order and at every row consumes **both** columns paired —
  the LTmin_n value drives the per-partition target threshold, the
  width_n column flags whether the partition spans more than one
  Layer I / Layer II coder partition row. A new free function
  `coder_partition_d5_reduction_row_order<F: Fn(u16) -> f64>(
  ltg_per_line) -> [CoderPartitionD5Reduction; 32]` zips the two
  row-order columns at the same array index, producing the per-
  frame paired input the bit-allocation loop reads in lockstep.
  A new public struct `CoderPartitionD5Reduction { ltmin_db: f64,
  width_n: u16 }` carries the per-row pair. No spec arithmetic is
  introduced — only the per-row pairing of the two existing row-
  order columns. The 0-based index convention matches steps 59 and
  60 (`out[i]` holds `(LTmin_{i + 1}, width_{i + 1})`), partition 0
  excluded. Width-column endpoints `(out[0].width_n = 0,
  out[31].width_n = 1)`, transition `(out[11].width_n = 0,
  out[12].width_n = 1)`, and full literal `[0×12, 1×20]` are
  pinned in tests. Tests: 770 lib (was 757 baseline; +13 unit)
  covering length pin (32), constant-callback fills every
  ltmin_db cell with the constant, ltmin column matches step 59
  for a non-trivial line-dependent callback, width column matches
  step 60 across two callbacks (structural orthogonality), width
  invariant across callbacks, width matches Table D.5 literal,
  identity callback returns ωlow_n per row, negative-identity
  callback returns -ωhigh_n per row, transition pair at array
  index 12, endpoint pin at indices 0 and 31, idempotence for a
  pure callback, single-dip on a strict-interior line affects only
  the target partition's ltmin_db with widths untouched, and
  strict-composition pairing with step 59 + step 60 across every
  row.5
  transcription in
  `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`) were
  read.

- psy: Annex D Model 1 §D.1 Step 8 row-order `width_n` vector over
  Table D.5 (Phase 2 step 60). Phase 2 step 59 (r258) broadcast the
  Phase 2 step 58 (r257) per-partition `LTg` minimum reducer
  `coder_partition_d5_ltg_min` across the Phase 2 step 55 (r254)
  row-order iterator `coder_partition_d5_spans`, producing the
  32-element row-order LTmin vector `[LTmin_1, LTmin_2, …, LTmin_32]`
  the Layer I / Layer II bit-allocation loop consumes per frame.
  r259 closes the second half of that per-frame input: a row-order
  vector of the `width_n` column the bit-allocation loop pairs with
  the LTmin vector at every row. A new free function
  `coder_partition_d5_width_row_order() -> [u16; 32]` broadcasts the
  Phase 2 step 52 (r251) per-partition `width_n` accessor
  `coder_partition_d5_width` across the same step 55 row-order
  iterator, returning the static 32-element 0-based array
  `[0, 0, …, 0, 1, 1, …, 1]` — twelve zeros followed by twenty ones
  per the Table D.5 transcription ("rows 0..=12 have width 0; rows
  13..=32 have width 1"). The vector is fully determined by the
  static table (no run-time inputs), unlike step 59's LTmin vector
  which closes over a caller-supplied `LTg(ω)` callback. The 0-based
  index convention matches step 59's exactly:
  `out[i] = width_{i + 1}`, partition 0 excluded. Tests: 757 lib
  (was 745 baseline; +12 unit) covering length pin (32), lower-block
  rule (n ∈ 1..=12 → 0), upper-block rule (n ∈ 13..=32 → 1), the
  single-step 0 → 1 transition at array index 12 (partition 13),
  binary-only cell-value pin, strict-composition cross-check against
  the per-partition step 52 lookup, table-literal pin against the
  verbatim 32-element constant, table-wide endpoint pin
  (`out[0] = 0`, `out[31] = 1`), upper-block-count sum pin (Σ = 20),
  idempotence across back-to-back calls, non-decreasing monotonicity,
  and ascending-partition iteration order.5 transcription in
  `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`) were read.

- psy: Annex D Model 1 §D.1 Step 8 row-order LTmin vector over
  Table D.5 (Phase 2 step 59). Phase 2 step 58 (r257) reduced the
  per-FFT-line global masking threshold `LTg(ω)` over a single
  coder partition `n ∈ 1..=32` by taking the minimum
  `LTmin_n = min_{ω ∈ [ωlow_n, ωhigh_n]} LTg(ω)`. The Layer I /
  Layer II bit-allocation loop consumes the full row-order vector
  `[LTmin_1, LTmin_2, …, LTmin_32]` per frame, walking the 32
  coder partitions in ascending-`n` order (the spec table's row
  order, pinned at iteration order by Phase 2 step 55 / r254's
  `coder_partition_d5_spans`). The Layer III outer-loop SNR-budget
  analogue consumes the same per-partition vector. A new free
  function
  `coder_partition_d5_ltg_min_row_order<F: Fn(u16) -> f64>(
  ltg_per_line) -> [f64; 32]` broadcasts the step 58 per-partition
  reducer across the step 55 row-order iterator, returning a
  32-element 0-based array where element `i` holds `LTmin_{i + 1}`
  (the spec's 1-based partition index in 0-based array form).
  Partition 0 (the degenerate single-line `width_n = 0` row
  carrying `ωlow_0` only) is excluded — step 58 returns `None`
  for `n = 0` because the reduction range is undefined without a
  `ωlow_n` boundary in Table D.5; the downstream bit-allocation
  loop walks partitions `1..=32` and does not consult partition 0,
  matching the spec's coder-partition usage. Composition rather
  than introduction: the accessor is a strict broadcast of step
  58's single-partition reducer across the step 55 ascending-`n`
  iterator — no new spec arithmetic is introduced, only the
  per-partition fold expanded across all 32 recoverable rows.
  Boundary semantics inherit from step 58 unchanged: a sharp dip
  on a shared boundary `ω = ωhigh_n = ωlow_{n+1}` enters both
  adjacent partitions' `LTmin` (the conservative-bit-allocation
  reading the spec intends). A caller that wants single-assignment
  binning uses the step 56 inverse accessor
  `first_partition_containing_line` to bin per line before
  folding. Complexity is `O(513)` per frame total — the sum of
  every partition's inclusive line range — dominated by the
  caller's `ltg_per_line` cost. 11 new lib unit tests pin: a
  constant LTg ≡ C callback returns `[C; 32]` (every partition's
  minimum over a flat dB curve is the constant); the 0-based-
  array / 1-based-partition convention `out[0] = LTmin_1` (under
  identity LTg the value equals ωlow_1 = 1, the table-wide lower
  edge); strict-composition cross-check against a manual loop
  calling the step 58 per-partition reducer for a non-trivial
  callback `ω * 0.7 - 13`; the array length is exactly 32; every
  cell is finite under a finite callback (no inf leak from the
  `f64::INFINITY` seed); a single -100 dB dip at a partition's
  interior middle line pulls only that partition's row down and
  leaves every other partition at the baseline; shared-boundary
  -50 dB dip at `ωhigh_5 = ωlow_6` pulls both adjacent partitions
  to the dip value and leaves every other partition at the
  baseline; end-to-end composition pin feeding the row-order
  builder the Step 7 `global_masking_threshold_db` value at every
  line (with a tonal masker at z = 5 Bark, SPL = 60 dB and a
  synthetic `z(ω) = ω · 0.05` Bark stand-in until Step 1's
  FFT-bin → Hz mapping lands) matched against the explicit per-
  line fold; row-order vector is non-decreasing under the identity
  callback (each `ωlow_n` grows strictly with `n` per the
  Phase 2 step 50 boundary-monotonicity reading); table-wide
  endpoint pin (`out[0] = ωlow_1`, `out[31] = ωlow_32`);
  negative-identity `LTg(ω) = -ω` callback returns `-ωhigh_n` per
  row (the highest line produces the most-negative reduction).
  Tests: 745 lib (was 734 baseline; +11 unit). Provenance: only
  the Phase 2 step 58 per-partition reducer
  `coder_partition_d5_ltg_min` and the Phase 2 step 55 row-order
  iterator `coder_partition_d5_spans` (and through them the Phase
  2 step 44 Step 7 `global_masking_threshold_db` and the Table
  D.5 transcription in
  `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
  §"Table D.5 - Layer I and Layer II coder partition table") are
  consulted; the row-order broadcast reading is the spec's per
  Annex D Step 8 (informative Model 1 reduction).

- psy: Annex D Model 1 §D.1 Step 8 per-partition `LTg` minimum
  reduction (Phase 2 step 58). Phase 2 step 44 (r219) landed Step 7's
  per-FFT-line global masking threshold `LTg(i)` as
  `global_masking_threshold_db`; Phase 2 step 49 (r248) transcribed
  Table D.5 (the Layer I / Layer II coder partition table); Phase 2
  step 57 (r256) closed the per-partition FFT-line walk as
  `coder_partition_d5_omega_iter`. r257 wires the two halves
  together into the spec's Step 8 reduction
  `LTmin_n = min_{ω ∈ [ωlow_n, ωhigh_n]} LTg(ω)`. A new free
  function `coder_partition_d5_ltg_min<F: Fn(u16) -> f64>(n,
  ltg_per_line) -> Option<f64>` reduces the caller-supplied per-FFT-
  line `LTg(ω)` callback (from Step 7's `global_masking_threshold_db`,
  applied per line) over every `ω ∈ [ωlow_n, ωhigh_n]` by taking the
  minimum; returns `Some(LTmin_n)` for any `n ∈ 1..=32` and `None`
  for the two edge cases inherited from `coder_partition_d5_omega_iter`
  (`n = 0` — `ωlow_0` not in Table D.5; `n = 33` — row absent). The
  reduction is the spec's most-conservative per-partition reading —
  a single FFT line dipping below the partition's average threshold
  pulls the whole partition's bit-allocation budget down to that
  line's level. Composition rather than introduction: the accessor
  is a strict composition of the Phase 2 step 57 iterator and
  `Iterator::map ∘ Iterator::fold(f64::INFINITY, f64::min)` — no
  spec arithmetic introduced, only the per-line minimum fold over
  the recoverable line range. The Step 7 `LTg` callback is the
  caller's, keeping this accessor pure with respect to the masker
  selection pipeline (Steps 1-5), which remain blocked on the PNG-
  only Table D.1 / D.2 / D.3 transcription gap. Boundary semantics:
  inclusive on both ends, matching the per-partition sum-over-lines
  pattern Phase 2 step 57 wired into Step 7's `Σ_{ω ∈ partition}`
  form; a sharp dip on a shared boundary `ωhigh_n = ωlow_{n+1}`
  reduces both adjacent partitions' `LTmin`. A caller that wants
  single-assignment binning uses the step 56 inverse accessor
  `first_partition_containing_line` to bin per line, then folds
  outside this accessor. 10 new lib unit tests pin: out-of-band
  `None` at `n = 0`, `n = 33`, `n = 100`, `n = 1000`, `n = u16::MAX`;
  constant-`LTg` returns the constant for every partition; identity
  `LTg(ω) = ω` returns `ωlow_n` for every partition; negative
  identity `LTg(ω) = -ω` returns `-ωhigh_n` for every partition;
  a single `-100 dB` dip at each partition's middle line pulls the
  whole partition's `LTmin` to `-100 dB`; cross-check against an
  explicit `coder_partition_d5_omega_iter ∘ map ∘ fold` fold for a
  non-trivial callback `ω * 0.7 - 13`; shared-boundary double-
  influence at every `ωhigh_n = ωlow_{n+1}` (a `-50 dB` dip pulls
  both adjacent partitions' `LTmin`); end-to-end composition pin
  feeding `global_masking_threshold_db` per line with a tonal
  masker at z = 5 Bark, SPL = 60 dB and a synthetic `z(ω) = ω · 0.05`
  Bark stand-in mapping (until Step 1's FFT-bin → Hz table lands)
  matched against the explicit per-line fold. Tests: 734 lib (was
  724 baseline; +10 unit). Provenance: only the Phase 2 step 57
  per-partition iterator `coder_partition_d5_omega_iter`, the Phase 2
  step 44 Step 7 `global_masking_threshold_db`, and (transitively)
  the Table D.5 transcription in
  `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md` §"Table D.5
  - Layer I and Layer II coder partition table" are consulted; the
  minimum-reduction reading is the spec's per Annex D Step 8
  (informative Model 1 reduction).

- psy: Annex D Table D.5 per-partition FFT-line iterator (Phase 2
  step 57). Phase 2 step 51 (r250) exposed each partition's
  `(ωlow_n, ωhigh_n)` boundary pair via
  `coder_partition_d5_line_range`. Phase 2 step 53 (r252) composed
  those boundaries with `width_n` into the
  `CoderPartitionD5Span` descriptor. Phase 2 step 54 (r253) lifted
  the membership inequality on that pair to the named predicate
  `partition_n_contains_line`; Phase 2 step 55 (r254) added the
  row-order iterator `coder_partition_d5_spans`; Phase 2 step 56
  (r255) closed the inverse lookup with
  `first_partition_containing_line`. r256 closes the per-partition
  FFT-line walk: a new free function
  `coder_partition_d5_omega_iter(n) -> Option<RangeInclusive<u16>>`
  yields every `omega ∈ [ωlow_n, ωhigh_n]` in ascending order for
  any `n ∈ 1..=32`. The iterator is the foundational primitive the
  downstream Model 1 / Model 2 per-partition reduction binds its
  sum-over-lines against — Annex D Step 7's
  `Σ_{j ∈ partition} 10^(LT[j]/10)` pattern composes directly:
  `coder_partition_d5_omega_iter(span.index).unwrap().map(|ω|
  per_line_value(ω)).sum()`. The iterator is inclusive on both ends,
  matching the dual-role `ωlow_{n+1} / ωhigh_n` reading pinned by
  step 50 and the inclusive-on-both-ends membership predicate named
  by step 54: two consecutive partitions both emit the shared
  boundary line `ω = ωhigh_n = ωlow_{n+1}`, matching the spec's
  per-partition sum-over-lines reading where the shared boundary
  *does* contribute to both partitions' reductions. A caller that
  wants single-assignment binning (no double-counting) uses the
  step 56 inverse accessor instead. Returns `None` for any `n`
  outside `1..=32` — the same recoverable range as the underlying
  descriptor. Implementation is one line —
  `coder_partition_d5_line_range(n).map(|(lo, hi)| lo..=hi)` — a
  pure composition of the step 51 line-range accessor and
  `RangeInclusive::new`, with no arithmetic introduced. 12 new lib
  unit tests pin: the out-of-band `None` branches at `n = 0`,
  `n = 33`, `n = 100`, and `n = u16::MAX`; partition 1's iterator
  starts at the table-wide lower edge `ω = 1`; partition 32's
  iterator ends at the table-wide upper edge `ω = 513`;
  per-partition endpoint and length agreement with step 51's
  `coder_partition_d5_line_range` for every `n ∈ 1..=32`; the
  ascending-stride-1 walk within each partition (no gaps, no
  duplicates); per-line agreement with the step 54 membership
  predicate `partition_n_contains_line(n, ω) = Some(true)` for
  every iterator-emitted `ω`; the shared-boundary double-emission
  property at every `n ∈ 1..=31` (both `n` and `n+1`'s iterators
  contain `ωhigh_n`); the table-wide band coverage
  `⋃ iter(n) = [1, 513]`; the total-line-count identity
  `Σ_n |iter(n)| = 513 + 31 = 544` (band size + 31 double-counted
  shared boundaries); and an end-to-end composition smoke pin
  `coder_partition_d5_spans` ∘ `coder_partition_d5_omega_iter` ∘
  `sum` matching the arithmetic-series closed form
  `Σ_{ω=ωlow_n}^{ωhigh_n} ω = (ωlow_n + ωhigh_n) ·
  (ωhigh_n − ωlow_n + 1) / 2` for every recoverable partition —
  pinning the downstream Step 8 partition-threshold reduction's
  composition path directly. Tests: 724 lib (was 712 baseline; +12
  unit). Provenance: only the Phase 2 step 51 accessor
  `coder_partition_d5_line_range` and its underlying Table D.5
  transcription in
  `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
  §"Table D.5 - Layer I and Layer II coder partition table" are
  consulted; the inclusive-on-both-ends reading is the spec's,
  pinned by Phase 2 step 50 (r249) and step 54 (r253).

- psy: Annex D Table D.5 inverse line→partition lookup (Phase 2
  step 56). Phase 2 step 53 (r252) composed each partition's three
  Table D.5 columns into a `CoderPartitionD5Span` descriptor with the
  inclusive boundary pair `(ωlow_n, ωhigh_n)`. Phase 2 step 54 (r253)
  lifted the membership inequality to the named predicate
  `partition_n_contains_line`. Phase 2 step 55 (r254) added a
  row-order iterator `coder_partition_d5_spans` over the recoverable
  descriptors. r255 closes the inverse direction: a new free function
  `first_partition_containing_line(omega) -> Option<u16>` returns the
  lowest partition index `n` whose inclusive boundary range
  `[ωlow_n, ωhigh_n]` contains the FFT line `omega`. The accessor
  yields `Some(n)` with `n ∈ 1..=32` for `omega ∈ [1, 513]` and `None`
  for any out-of-band `omega`. Shared-boundary disambiguation matches
  the spec table's row-order presentation: every shared boundary line
  `ω = ωhigh_n = ωlow_{n+1}` lies in both partition `n` and partition
  `n + 1` under the inclusive-on-both-ends reading pinned by step 54,
  and the inverse accessor returns the **lower** index `n` — the
  unique deterministic choice that does not double-count the boundary
  lines and matches the row-order iterator's ascending walk pinned by
  step 55. Implementation is one line —
  `coder_partition_d5_spans().find(|s| s.omega_low <= omega && omega
  <= s.omega_high).map(|s| s.index)` — with no arithmetic beyond the
  inequality on each descriptor's pre-computed boundaries.
  Complexity is `O(32)` worst case; for a Model 1 / Model 2 reduction
  sweeping all 513 lines this is `O(513 × 32) ≈ 16 K` boundary
  comparisons, well below any performance threshold worth complicating
  the accessor over. 9 new lib unit tests pin: the out-of-band `None`
  branches at `omega = 0`, `omega = 514`, `omega = 10_000`, and
  `omega = u16::MAX`; the table-wide-lower-edge identity
  `first_partition_containing_line(1) = Some(1)`; the table-wide-
  upper-edge identity `first_partition_containing_line(513) =
  Some(32)`; the lower-index-pick at every shared boundary
  `ω = ωhigh_n` for `n ∈ 1..=31`; per-partition strict-interior
  agreement at `ω = ωlow_n + 1` against the step 53 descriptor; the
  table-wide `[1, 513]` no-gap coverage property; the agreement with
  the step 54 membership predicate
  `partition_n_contains_line(n, ω) = Some(true)` for every in-band
  `ω`; and the "lowest partition first" semantics directly — the
  inverse accessor's answer is the minimum `n` across all partitions
  that contain `ω` under the step 54 predicate (sweeping every
  in-band line). Tests: 712 lib (was 703 baseline; +9 unit).
  Provenance: only the Phase 2 step 55 iterator
  `coder_partition_d5_spans` and (through it) the Phase 2 step 53
  descriptor `coder_partition_d5_span` and its underlying Table D.5
  transcription in
  `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
  §"Table D.5 - Layer I and Layer II coder partition table" are
  consulted.

- psy: Annex D Table D.5 row-order iteration helper (Phase 2
  step 55). Pair the Phase 2 step 54 `partition_n_contains_line`
  membership predicate with a row-order iterator over the recoverable
  Table D.5 descriptors so the downstream Model 1 / Model 2
  partition-threshold reduction can walk the table in spec order
  without open-coding the `1..=32` range or the descriptor lookup at
  every reduction site. New free function `coder_partition_d5_spans()`
  returns `impl Iterator<Item = CoderPartitionD5Span>` yielding
  exactly 32 descriptors in ascending row order — `n = 1` through
  `n = 32` — and skips the two boundary-table-gap edges (`n = 0` and
  `n = 33`) the step 53 descriptor returns `None` for. Implementation
  is `(1_u16..=32).map(|n| coder_partition_d5_span(n).expect(...))`;
  the `.expect` is infallible by step 53 construction (every iterated
  `n` is a recoverable Table D.5 row). 7 new lib unit tests pin: the
  exact 32-descriptor count; the strictly ascending row-order
  `1..=32` sequence; per-descriptor agreement with
  `coder_partition_d5_span(n)`; the boundary-table-gap skip at `n = 0`
  and `n = 33`; the table-wide band coverage `[1, 513]` and the
  adjacent-row tiling identity `ωhigh_n = ωlow_{n+1}` for every
  consecutive pair; the spec-read pairing pattern with
  `partition_n_contains_line` across every `(span, ω) ∈ iter ×
  0..=520`; and the re-iteration property (cheap clone, identical
  sequence on each walk — the multi-pass walks the downstream
  reduction relies on). Tests: 703 lib (was 696 baseline; +7 unit).
  Provenance: only the Phase 2 step 53 descriptor
  `coder_partition_d5_span` and its underlying Table D.5
  transcription in
  `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
  §"Table D.5 - Layer I and Layer II coder partition table" are
  consulted; the row-order walk is the spec table's own ordering.
- psy: Annex D Table D.5 inclusive-line membership predicate
  (Phase 2 step 54). Lift the obvious inequality on the Phase 2
  step 53 `CoderPartitionD5Span` descriptor (`s.omega_low <= ω &&
  ω <= s.omega_high`) to a named predicate so the downstream
  Model 1 / Model 2 partition-threshold reduction can read like the
  spec ("for each line in partition `n` …") and the range-rejection
  behaviour at the two boundary-table gaps stays in one place. New
  free function `partition_n_contains_line(n, omega)` returns
  `Some(true)` if `ω` is inside partition `n`'s inclusive boundary
  range `[ωlow_n, ωhigh_n]`, `Some(false)` if it isn't, and `None`
  for any `n` outside `1..=32` (the same range as the descriptor).
  The predicate is a pure composition of the step 53 descriptor
  with the inclusive inequality — no arithmetic beyond the
  inequality on the descriptor's pre-computed boundaries. Under the
  inclusive-on-both-ends reading the descriptor inherits from
  Phase 2 step 50, the tiling identity `ωhigh_n = ωlow_{n+1}` means
  the shared boundary line belongs to both partitions `n` and
  `n + 1`. The `omega` argument is not range-checked against the
  table-wide FFT-line domain `[1, 513]`; out-of-band values return
  `false` at every in-range `n`. 7 new lib unit tests pin: the
  inclusive-on-both-ends reading at every recoverable partition's
  `ωlow_n` / `ωhigh_n`; the off-by-one exclusion of `ωlow_n - 1`
  and `ωhigh_n + 1`; spec-anchored membership at partitions
  `{1, 12, 13, 32}` including the shared-boundary line; `None` at
  both edges (`n ∈ {0, 33}`) and above (`n ∈ {34, 64, u16::MAX}`)
  across a sweep of `omega` values; line-level tiling (boundary
  lines belong to two consecutive partitions, interior lines
  belong to exactly one); pure-composition agreement across every
  `(n, ω) ∈ 1..=32 × 0..=520` pair; and out-of-band `omega`
  rejection (`omega ∈ {0, 514, 1024, u16::MAX}`). Tests: 696 lib
  (was 689 baseline; +7 unit). Provenance: only the Phase 2 step 53
  descriptor and its underlying Table D.5 transcription in
  `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
  §"Table D.5 - Layer I and Layer II coder partition table" are
  consulted; the inclusive-on-both-ends boundary reading is the
  spec's, pinned by Phase 2 step 50 (r249).
- psy: Annex D Table D.5 composed partition descriptor (Phase 2
  step 53). Wire the Phase 2 step 52 `width_n` column accessor into
  the Model 1 / Model 2 partition-threshold reduction by composing
  it with the Phase 2 step 51 line-range accessor into a single
  per-partition descriptor. New struct `CoderPartitionD5Span` carries
  partition index `n`, the inclusive FFT-line boundaries `ωlow_n` /
  `ωhigh_n`, and the `width_n` value; new free function
  `coder_partition_d5_span(n)` returns `Some(span)` for
  `n ∈ 1..=32` and `None` outside. The valid range is the
  intersection of the line-range accessor's range (1..=32) and the
  width accessor's range (0..=32); partitions 0 and 33 return
  `None` for the same boundary-table-gap reasons step 51 already
  documents. The composition is pure: `omega_low` is
  `coder_partition_d5_omega_low(n)`, `omega_high` is
  `coder_partition_d5_omega_high(n)`, `width` is
  `coder_partition_d5_width(n)`, and no arithmetic beyond what the
  three underlying accessors already perform is introduced. 8 new
  lib unit tests pin: spec-anchored values for `n ∈ {1, 12, 13, 32}`;
  range rejection at both edges (`n = 0`, `n = 33`) and above
  (`n ∈ {34, 64, u16::MAX}`); per-row composition agreement against
  the underlying accessors across `n ∈ 1..=32`; the uniform 17-line
  inclusive span (`omega_high - omega_low + 1` matches
  `CODER_PARTITION_D5_STRIDE + 1` everywhere); the `width_n` block
  structure preserved through the composition (`0` for
  `n ∈ 1..=12`, `1` for `n ∈ 13..=32`); the tiling property
  (every span's `omega_high` equals the next span's `omega_low`);
  the `index` field echoes the input verbatim; and `omega_low <
  omega_high` on every recoverable partition. Tests: 689 lib (was
  681 baseline; +8 unit). Provenance: only Table D.5 in
  `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
  §"Table D.5 - Layer I and Layer II coder partition table" is
  consulted; the step-50, step-51 and step-52 accessors already
  cite the same source.
- psy: Annex D Table D.5 `width_n` column accessor (Phase 2 step 52).
  Surface the third column of the verbatim Table D.5 row (after the
  index and the dual-role partition-boundary cell) as a table-level
  free function so callers don't have to round-trip through the row
  struct's `width` field. New free function
  `coder_partition_d5_width(n)` returns `Some(width_n)` for
  `n ∈ 0..=32` and `None` otherwise. The verbatim transcribed
  values are 0 for rows `n ∈ 0..=12` and 1 for rows `n ∈ 13..=32`;
  the split is a single step transition at row 13 with no
  transitional row. The accessor is a pure rename of
  `CoderPartitionD5::width` — no arithmetic and no interpretation —
  and matches `coder_partition_d5(n).map(|r| r.width)` exactly. The
  `width_n` column is structurally orthogonal to the partition
  boundary column already exposed by step 51's
  `coder_partition_d5_line_range`; no boundary value or stride is
  consulted. 7 new lib unit tests pin: four spec-anchor rows
  (`n ∈ {0, 12, 13, 32}` → `{0, 0, 1, 1}`); full-table parity with
  the row-field view; the two-block constant-within-block structure
  (0 for `n ∈ 0..=12`, 1 for `n ∈ 13..=32`); out-of-range rejection
  (`n ∈ {33, 64, u16::MAX}` → `None`); the {0, 1} range constraint;
  the single-step transition at row 13 (exactly one neighbour-pair
  changes value across the table, and it's the 12 → 13 step going
  0 → 1); and the constant-within-block orthogonality with the
  boundary column. Tests: 681 lib (was 674 baseline; +7 unit).
  Provenance: only the `width_n` column from
  `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
  §"Table D.5 - Layer I and Layer II coder partition table" is
  consulted.
- psy: Annex D Table D.5 partition FFT-line range accessor (Phase 2
  step 51). Compose the Phase 2 step 50 dual-role accessors into a
  single per-partition span accessor. New free function
  `coder_partition_d5_line_range(n)` returns the inclusive
  `(ωlow_n, ωhigh_n)` tuple of partition `n` for `n ∈ 1..=32`, and
  `None` outside. The valid range is the intersection of the two
  step-50 accessors' ranges (`ωlow_n` covers `n ∈ 1..=33`; `ωhigh_n`
  covers `n ∈ 0..=32`), so two partitions are explicitly missing one
  boundary each: `n = 0` (Table D.5 prints `ωlow_{n+1}` at row `n`,
  so partition 0's own lower boundary `ωlow_0` is not in the table);
  `n = 33` (Table D.5 tops out at row 32 with `ωhigh_32 = 513`, so
  partition 33's upper boundary is not in the table either). Both
  return `None` verbatim; no synthetic boundary is invented for
  either edge. The accessor is a pure composition of
  `coder_partition_d5_omega_low` and `coder_partition_d5_omega_high`
  with no additional arithmetic. 9 new lib unit tests pin: four
  spec-anchor spans (`n = 1` → `(1, 17)`, `n = 13` → `(193, 209)`,
  `n = 14` → `(209, 225)`, `n = 32` → `(497, 513)`); both
  edge-`None` cases (`n = 0` and `n = 33`) and out-of-range rejection
  (`n ∈ {34, 64, u16::MAX}`); span non-degeneracy (`low < high`
  across all recoverable partitions); the composition contract
  (`line_range(n) == Some((omega_low(n), omega_high(n)))`); the
  uniform 17-line inclusive span (open span equals
  `CODER_PARTITION_D5_STRIDE = 16` per partition); and the tiling
  property (every partition's `ωhigh` equals the next partition's
  `ωlow`, the band tops out at line 513 and starts at line 1). Tests:
  674 lib (was 665 baseline; +9 unit). Provenance: only the column
  heading `ωlow_{n+1} / ωhigh_n` from
  `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
  §"Table D.5 - Layer I and Layer II coder partition table" is
  consulted.
- psy: Annex D Table D.5 dual-role boundary accessors (Phase 2
  step 50). Surface the `ωlow_{n+1} / ωhigh_n` column heading's two
  spec roles as named accessors so callers no longer have to apply
  the column-rename arithmetic themselves. Two new per-row methods
  on `CoderPartitionD5` — `omega_high()` and `omega_low_of_next()`
  — each return the row's verbatim `omega_boundary` cell under one
  of the two spec role names. Two new free functions on the
  table — `coder_partition_d5_omega_high(n)` and
  `coder_partition_d5_omega_low(n)` — return the same printed
  integers but as the `ωhigh_n` / `ωlow_n` reading of partition
  `n`. The `ωhigh_n` accessor covers the full spec range `n ∈
  0..=32` (returns `None` outside); the `ωlow_n` accessor covers
  `n ∈ 1..=33` only (returns `None` for `n = 0` and `n ≥ 34`)
  because Table D.5 prints `ωlow_{n+1}` at row `n`, so partition
  0's own lower boundary `ωlow_0` is not in the table and no
  default is invented. All four accessors are pure column / row
  renames — no arithmetic is performed beyond the verbatim
  `n → n - 1` row shift that the column heading's `ωlow_{n+1}`
  half explicitly requires. 10 new unit tests pin: the two per-row
  methods rename `omega_boundary` for every in-range row; the two
  methods return the same integer on every row (the dual-role
  identity); four `ωhigh_n` spec-anchor rows
  (`ωhigh_0 = 1`, `ωhigh_12 = 193`, `ωhigh_13 = 209`,
  `ωhigh_32 = 513`); the `ωhigh_n` accessor equals
  `omega_boundary` for every in-range index; the `ωhigh_n` accessor
  rejects `n ∈ {33, 64, u16::MAX}`; four `ωlow_n` spec-anchor rows
  (`ωlow_1 = 1`, `ωlow_13 = 193`, `ωlow_14 = 209`,
  `ωlow_33 = 513`); the `ωlow_n` accessor returns `None` for
  partition 0 (the not-in-table case); the `ωlow_n` accessor
  rejects `n ∈ {34, 64, u16::MAX}`; and the table-wide dual-role
  identity `ωlow_{n+1} == ωhigh_n` across every `n ∈ 0..=32`.
  Provenance: only the column heading
  `ωlow_{n+1} / ωhigh_n` from
  `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
  §"Table D.5 - Layer I and Layer II coder partition table" is
  consulted.
- psy: Annex D Table D.5 Layer I / Layer II coder partition table
  (Phase 2 step 49). The 33-row partition table is transcribed
  verbatim from the staged
  `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md` extract
  of ISO/IEC 11172-3:1993 Annex D clause D.2 (PDF page 145 /
  printed p.139). A new `CoderPartitionD5` struct carries the
  three spec columns per row — `index` (the partition number
  `n = 0..=32`), `omega_boundary` (the FFT-line index that the
  spec table prints under the dual-role heading
  `ωlow_{n+1} / ωhigh_n`), and `width` (the spec's `width_n`
  column, 0 for rows 0..=12 and 1 for rows 13..=32). The full
  table lands as `CODER_PARTITION_TABLE_D5: [CoderPartitionD5;
  33]`; `coder_partition_d5(n)` is a thin row-index accessor that
  returns `None` for any `n` outside the spec range. The uniform
  stride between consecutive rows (16 FFT lines per partition) is
  exposed as `CODER_PARTITION_D5_STRIDE: u16 = 16`; 10 new unit
  tests cover row count, index contiguity, four spec-anchor rows
  (0 = `(1, 0)`, 12 = `(193, 0)`, 13 = `(209, 1)`, 32 =
  `(513, 1)`), strict monotonicity of the ω column, uniform
  16-line stride across all 32 row transitions, the
  width-0/width-1 split at row 13, the row-accessor round-trip on
  every in-range index, accessor rejection on out-of-range
  indices, the 1-based FFT-line indexing convention (row 0
  carries ω = 1), and the top-of-table pin (row 32 carries
  ω = 513 = 1 + 32·16, matching the 1024-sample FFT's 1..=513
  one-based half-spectrum).
- psy: Annex D Model 2 §C.1.5.3.2.1 Layer III spreading-function
  primitives (Phase 2 step 48). Two new functions land the Layer
  III modification of the Model 2 spreading function as transcribed
  in `docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md` from
  the staged ISO/IEC 11172-3:1993 PDF:
  `model2_layer3_spread_db(i, j)` returns the per-partition dB
  value `tmpy(i, j)` — `3.0 * (j - i)` for `j >= i` (upward /
  on-diagonal branch) and `1.5 * (j - i)` for `j < i` (downward
  branch); `model2_layer3_spread_linear(i, j)` returns the
  spec's linear factor `sprdngf(i, j) = 10^(tmpy/10)` with the
  spec's clamp "values greater than 1e-6 are used; all others
  set to zero" applied via the new
  `MODEL2_LAYER3_SPREAD_LINEAR_MIN = 1.0e-6` constant. The
  diagonal at `i == j` yields exactly `tmpy = 0` / linear = 1.0;
  the upward branch grows above unity (positive `tmpy`); the
  downward branch falls below unity (negative `tmpy`) and
  triggers the clamp at `j - i <= -40` (`tmpy <= -60 dB`). 9 new
  unit tests cover: diagonal returns zero across the Model 2
  partition range (1..63); upward branch matches the verbatim
  `3.0 * (j - i)` formula at +1 / +5 / +20 partition steps;
  downward branch matches `1.5 * (j - i)` at -1 / -4 / -20
  steps; diagonal linear factor is exactly 1.0; upward linear
  factor strictly exceeds 1.0 and grows monotonically with
  distance (`10^0.3 ≈ 1.9953` at +1); downward linear factor
  is strictly below 1.0 and shrinks monotonically (`10^-0.15
  ≈ 0.7079` at -1); the clamp boundary holds the spec's strict
  `> 1.0e-6` comparison (-39 survives, -40 collapses to exact
  zero, -50 stays clamped); the `MODEL2_LAYER3_SPREAD_LINEAR_MIN`
  constant reads back `1.0e-6` verbatim; the upward branch's
  diagonal value matches what the downward branch would yield if
  extended through `j == i`. The primitives are pure scalar
  functions of the two partition indices — the broader Model 2
  spreading matrix and the inter-partition energy convolution
  remain follow-up work for a later step.
- psy: Annex D Model 1 §D.1 Step 5 decimation primitives
  (Phase 2 step 47). Two new primitives wire the spec's
  Step 5 masker sieve between Step 4 placement (r229) and
  Step 6 individual-masking-threshold calculation (r219):
  `masker_above_threshold_in_quiet(masker, ltq_db)` is the
  Step 5(a) threshold-in-quiet screening predicate
  (`X_tm(k) >= LTq(k)` / `X_nm(k) >= LTq(k)`, identical for
  tonal and non-tonal maskers per the verbatim spec text),
  and `decimate_tonal_within_half_bark(maskers)` is the
  Step 5(b) tonal-cluster decimation pass — a sliding window
  of width `STEP5_TONAL_DECIMATION_WINDOW_BARK = 0.5` Bark
  in which two-or-more tonal maskers within strictly less
  than 0.5 Bark of each other collapse to the loudest member
  of the cluster (input-order stable on tied SPLs;
  first-encountered loudest wins). Non-tonal maskers pass
  through unchanged because §D.1 Step 4(c) already yields
  at most one non-tonal masker per critical band. The
  algorithm sorts the tonal subset by `z_bark`, walks
  consecutive Bark gaps to find clusters, and emits the
  surviving maskers in original-slice order; non-tonal
  maskers are interleaved back at their original positions.
  17 new unit tests cover: Step 5(a) above-LTq keep / below-LTq
  drop / at-LTq inclusive boundary on both tonal and non-tonal
  maskers; the `STEP5_TONAL_DECIMATION_WINDOW_BARK` constant
  reads back 0.5 verbatim; Step 5(b) edge cases (empty input,
  singleton passthrough, pair within window keeps loudest,
  pair at exactly 0.5 Bark both survive per the spec's strict
  "less than" wording, pair outside window both survive,
  non-tonal cluster pass-through, three-member cluster
  collapses, two separate clusters collapse independently,
  ties resolve to first-encountered, unsorted input still
  clusters correctly, mixed tonal / non-tonal preserves
  non-tonal in place); a compositional invariant (Step 5(a)
  then Step 5(b) reproduces the spec's full Step 5 sieve);
  and an end-to-end smoke that pipes Step 5(a) + 5(b) into
  Step 7 `global_masking_threshold_db` and confirms the
  result matches direct evaluation on the decimated slice.
- psy: Annex D Model 1 §D.1 Step 4 masker placement helper +
  Step 7 nearby-masker Bark-window range pre-filter
  (Phase 2 step 46). `masker_at_band(boundaries, band_no, kind,
  spl_db)` composes Step 45's `CRITICAL_BANDS_D2*` slices with
  Step 44's `Masker` carrier: the returned masker sits at the
  band's verbatim top `z_bark` per the §D.1 Step 4 rule, and the
  caller-supplied SPL (the value Steps 1-3 will produce from the
  FFT spectrum) is wrapped into the typed
  `Masker { kind, z_bark, spl_db }` already consumed by
  `individual_masking_threshold_db` and
  `global_masking_threshold_db`. `masker_in_step7_window_of_line
  (masker, z_i_bark)` exposes the §D.1 Step 7 "for a given i
  the range of j may be reduced to maskers within −8…+3 Bark of
  i" optimisation as a single inlined Bark-distance test: a
  caller computing a sparse per-line `LTg(i)` map can pre-shrink
  its masker slice once per line via `filter()` and skip the
  `individual_masking_threshold_db` call for every out-of-range
  masker. The two bounds are exposed as named `pub const`s
  (`STEP7_NEARBY_MASKER_DZ_LO_FROM_LINE = -8.0` open low,
  `STEP7_NEARBY_MASKER_DZ_HI_FROM_LINE = 3.0` closed high). The
  predicate is the open-low, closed-high intersection of the
  spec text with the §D.1 Step 6 `vf` `[-3, 8)` half-open `dz`
  window — the set of maskers it lets through is exactly the
  set for which `individual_masking_threshold_db` returns
  `Some`. 12 new unit tests cover: band-0 first-row anchor
  (D.2a `z = 0.617`, SPL passthrough); last-row anchor (D.2a
  band 23 `z = 23.923`); out-of-range band index returns
  `None`; cross-table dispatch (D.2d band 0 below D.2a band 0);
  a self-placement composition smoke test (place at band 5,
  evaluate LT at the masker's own `z` → `SPL + av_tm`); a
  loud-local-masker composition smoke test (place at D.2c band
  10 with 80 dB SPL → `LTg` >> `LTq`); the §D.1 Step 7 window
  constants reproduce the spec text; in-range edge cases
  (centred, 2 Bark above, 5 Bark below); high-edge inclusivity
  at `dz_from_line = +3` exactly + exclusion at `+3.0001`;
  low-edge exclusivity at `dz_from_line = -8` exactly + inclusion
  just above; a 0.25-Bark masker-position sweep verifying the
  predicate matches `individual_masking_threshold_db`'s `Some`
  set on every sample; and a functional invariant
  (`filter()`-then-`global_masking_threshold_db` produces the
  same `LTg(i)` as feeding the full slice — the pre-filter is
  mechanically equivalent to dropping `vf = None` contributions
  from the energy sum). Steps 1-3 / 5 (1024-sample FFT, SPL
  conversion, tonality classifier, decimation) remain blocked
  on PNG-only Annex D Tables D.1 / D.3 / D.4 (Phase 2 step 46)

- psy: Annex D Model 1 §D.1 Step 4 critical-band-boundary
  **Tables D.2a–f** transcribed verbatim from the docs file
  `mp3-annex-d-psychoacoustic-extracts.md`. New
  `CriticalBandBoundary { no, index_fcb, frequency_hz, z_bark }`
  carrier and six `pub const` arrays `CRITICAL_BANDS_D2A`
  through `CRITICAL_BANDS_D2F` for the (Layer I/II × 32/44.1/48
  kHz) Cartesian product (24 / 25 / 26 / 25 / 27 / 27 rows
  respectively, all using zero-based `no` numbering per the
  spec table). New typed sampling-rate key
  `AnnexDSamplingRate { Hz32000, Hz44100, Hz48000 }` with
  `from_hz` / `as_hz` accessors. `critical_band_boundaries(layer,
  fs)` dispatches `(Layer, AnnexDSamplingRate)` to the matching
  table, returning `None` for `Layer::LayerIII` (Annex D is
  normative only for Layer I and Layer II — Layer III's
  spreading-function override in C.1.5.3.2.1 reuses these
  tables explicitly via the matching Layer-I or Layer-II key).
  `band_of_fft_line(boundaries, fft_line_index)` is the
  §D.1 Step 4 masker-placement primitive: it maps a 1-based
  FFT-line index into Table D.1's frequency table to the
  critical-band index `no` it falls into, returning `None` for
  `0` (invalid 1-based index) and for any index above the
  top band's `index_fcb` (out of the audio band of the table).
  The staged docs file marks D.2e row 17's `z_bark` cell as
  `16,11[illegible]` (clipped final digit in the PDF render);
  the constant `D2E_BAND_17_BARK_IS_ILLEGIBLE` records that
  uncertainty, and the table cell preserves the legible-only
  prefix `16.11` — the docs file's prose estimate `16.116` is
  explicitly NOT adopted as a verbatim source value (a
  three-decimal value would mis-represent the spec render's
  legible precision). 13 new unit tests in `psy::tests`
  cover: per-table band-count + monotone `no` contiguity
  (`no == k` for every row index `k`); first-row and last-row
  cell-by-cell anchor reproduction against the docs file (D.2a
  rows 0 + 23, D.2c rows 0 + 25); strict monotone ascent in
  (`index_fcb`, `frequency_hz`, `z_bark`) across every adjacent
  pair of all six tables; `AnnexDSamplingRate::from_hz`
  round-trip on `{32_000, 44_100, 48_000}` Hz plus rejection of
  three LSF rates (`16_000, 22_050, 24_000`);
  `critical_band_boundaries` six-way valid dispatch + Layer III
  `None`; the `band_of_fft_line` locator on D.2a (zero rejection,
  bottom-band single-line edge, two mid-band ranges, top-band
  edge inclusion at line 108, out-of-range `None` at lines 109
  + 999) and on D.2e (early bands 0..4); the D.2e illegible-cell
  read-back preserves `16.11` exactly and excludes the prose
  estimate; the cross-table sanity check that D.2d's first band
  edge sits below D.2a's first band edge (Layer II's longer FFT
  window resolves a lower starting band edge); and a
  cross-check that each table's row count equals the prose
  step-4-summary count + 1 (24 cells for "23 bands", etc.).
  These tables are the §D.1 Step 4 masker-placement substrate
  the future Model 1 Step 4 (tonal-vs-non-tonal classifier)
  will iterate over; the placeholder/scaffold path of r194 /
  r197 / r204's threshold-in-quiet `XminThresholds` remains
  unchanged. (Phase 2 step 45)

- psy: Annex D Model 1 §D.1 Step 6 masking-function `vf` +
  masking-index `av_tm` / `av_nm` + Step 7 global-threshold
  summation primitives. `masking_index_tonal(z_j)` /
  `masking_index_non_tonal(z_j)` reproduce the verbatim spec
  formulas (`-1.525 - 0.275 * z(j) - 4.5` and
  `-1.525 - 0.175 * z(j) - 0.5` dB respectively).
  `masking_function_vf(dz, X)` returns the 4-branch piecewise
  `vf` for `dz ∈ [-3, 8)`, `None` outside (the spec's "masker
  ignored" range). `individual_masking_threshold_db(masker, z_i)`
  composes `LT = X + av + vf` for a single `Masker { kind, z_bark,
  spl_db }`. `global_masking_threshold_db(maskers, z_i, ltq_db)`
  carries out the Step 7 `LTg(i) = 10 * log10(10^(LTq/10) + Σ
  10^(LT/10))` energy sum across all in-range maskers. These are
  pure primitives that consume the (masker SPL, masker Bark
  position) tuples Steps 1-5 of Model 1 will eventually produce —
  Steps 1-5 themselves (1024-sample FFT, SPL conversion, tonality
  classifier, decimation, masker selection) remain blocked on
  the PNG-only Annex D Tables D.1 / D.2 / D.3 / D.4 DOCS-GAP and
  are not landed this round. Validated by 18 new lib unit tests
  in `psy::tests` covering: exact reproduction of the masking-index
  spec formulas at five Bark positions each; tonal `av_tm <` non-tonal
  `av_nm` invariant; `vf` four-branch numeric reproduction with
  hand-computed values for each branch (`vf(-3, 60) = -64`,
  `vf(-1, 60) = -30`, `vf(0.5, 60) = -8.5`, `vf(2, 60) = -25`, etc.);
  `vf` continuity across `dz = 0`; `vf` `None`-out-of-range guards
  at `dz < -3` and `dz >= 8`; individual-threshold formula
  composition at `z(i) = z(j)` (`LT = SPL + av`); tonal LT below
  non-tonal LT at matched parameters; global-threshold reductions
  (no maskers → LTq, distant masker → LTq, strong local masker
  dominates), monotone power addition across one vs two maskers,
  and exact `+10*log10(2) ≈ +3.0103 dB` for two equal-power
  co-located maskers (Phase 2 step 44 / r219)
- psy: caller-supplied §D.1 Step 3 dB offset path —
  `XminThresholds::threshold_in_quiet_with_offset_db(sample_rate_hz,
  version, offset_db)` accepts an arbitrary dB scalar in place of the
  spec's two-branch `−12 dB` / `0 dB` switch on
  `bitrate_kbps_per_channel >= 96`. The bowl-vs-bass-vs-treble per-band
  shape is preserved; the whole curve is translated up or down by
  `offset_db` dB. `Mp3Encoder::new_with_threshold_in_quiet_offset` +
  `codec_encoder::make_encoder_with_threshold_in_quiet_offset` thread
  the offset through the direct- and trait-API factories. Spec-default
  values are byte-identical to `threshold_in_quiet`:
  `offset_db = -12.0` matches the `>=96 kbit/s/ch` branch, `0.0`
  matches the `<96 kbit/s/ch` branch (Phase 2 step 43 / r213)
- psy: trait-API one-shot threshold-in-quiet factory —
  `Mp3Encoder::new_with_threshold_in_quiet` + the matching
  `codec_encoder::make_encoder_with_threshold_in_quiet` bundle
  `new_with_outer_loop(DEFAULT_OUTER_LOOP_THRESHOLD)` +
  `set_per_band_xmin(XminThresholds::threshold_in_quiet(SR, version,
  bitrate_kbps_per_channel))` into one call. Per-channel bitrate
  (`bitrate_kbps / nch`) drives the §D.1 Step 3 `−12 dB` offset
  switch — 192 kbit/s stereo (96 kbit/s per channel) is exactly the
  cutover, 128 kbit/s mono (128 ≥ 96) triggers, 128 kbit/s stereo
  (64 < 96) does not (Phase 2 step 42 / r207)
- psy: per-band threshold-vector scaffold (Annex D threshold-in-quiet
  long-block path) + `outer_loop_search_long_per_band` primitive +
  `Mp3Encoder::set_per_band_xmin` (Phase 2 step 39 / r194)
- psy: per-cell threshold-in-quiet pure-short path +
  `outer_loop_search_short_per_band` primitive +
  `XminThresholds::threshold_in_quiet` constructor (long + short cells
  populated from the same Annex D anchors) + stream-encoder dispatch on
  `BlockType::Short if !mixed_block_flag` (Phase 2 step 40 / r197)
- psy: per-band threshold-in-quiet mixed-block path +
  `outer_loop_search_mixed_per_band` primitive (long-region per-band
  `xmin_long[0..=7]` + short-region per-cell
  `xmin_short[3..=11][..]`) + stream-encoder dispatch on
  `BlockType::Short if mixed_block_flag` consuming
  `XminThresholds::mixed_long` / `XminThresholds::mixed_short`
  (Phase 2 step 41 / r204)

## [0.1.2](https://github.com/OxideAV/oxideav-mp3/releases/tag/v0.1.2) - 2026-05-30

### Other

- DEFAULT_ATTACK_THRESHOLD empirical-corpus calibration (Phase 2 step 38)
- §B1 encoder-delay + zero-padding gapless wiring (r185)
- oxideav_core::Decoder trait MPEG-2 LSF widening (Phase 2 step 37)
- oxideav_core::Decoder trait stereo widening (Phase 2 step 36)
- DEFAULT_AMBIENT_LEAK empirical-corpus calibration (Phase 2 step 35)
- §2.4.3.4.10 finer attack-detector knobs (Phase 2 step 34)
- §2.4.3.4.9 cross-channel-MS block-type agreement (Phase 2 step 33)
- §2.4.3.4.9 independent-stereo widening of the block-type override toggles (Phase 2 step 32)
- §2.4.3.4.10.3 auto block-type mixed-block promotion (Phase 2 step 31)
- §C.1.5.4.3 long-family transition-skeleton wiring (Phase 2 step 30)
- §C.1.5.4.3 mixed-block analogue `outer_loop_search_mixed` (Phase 2 step 29)
- §C.1.5.4.3 auto block-type × outer-loop integration (Phase 2 step 28)
- §C.1.5.4.3 short-block analogue `outer_loop_search_short` (Phase 2 step 27)
- auto block-type: signal-driven attack detection + §C.1.5.2 state machine (Phase 2 step 26)
- §MPEG-2.5 frame-parser support (Phase 2 step 25)
- §C.1.5.4.4.8 linbits-reach filter (Phase 2 step 24, #1106)
- §2.4.2.7 forward mixed-block MDCT encode path (Phase 2 step 23)
- §2.4.3.4.10.2 forward short-block MDCT path (Phase 2 step 22)
- joint-stereo auto MS/LR trait factory (Phase 2 step 21)
- joint-stereo auto MS/LR per-frame picker (Phase 2 step 20)
- §C.1.5.4.3.4 preemphasis decision (Phase 2 step 19)
- §C.1.5.4.3 scalefac_scale 0→1 escalation (Phase 2 step 18)
- joint-stereo MS encode (Phase 2 step 17)
- scrub pre-existing LAME decorative-attribution prose
- Phase 2 step 16 — independent-stereo (ChannelMode::Stereo / DualChannel) encode
- Phase 2 step 15 — opt-in §2.4.3.1 CRC-16 frame protection
- Phase 2 step 14 — true-VBR per-frame bitrate + Xing TOC auto-fill
- Phase 2 step 13 — Xing/Info VBR information-frame emission
- oxideav_core::Decoder trait wiring + dual-API factory
- Phase 2 step 12 — wire Mp3Encoder into oxideav_core::Encoder trait
- §C.1.5.4.3 outer (distortion-control) loop — Phase 2 step 11
- Phase 2 step 10 — stream-level PCM → MP3 driver
- Phase 2 step 9 — §2.4.2.7 cross-frame bit-reservoir scheduler (main_data_begin > 0)
- §2.4.1.7 main_data() assembler + main_data_begin=0 (Phase 2 step 8)
- §2.4.1.7 huffmancodebits() bit emission (Phase 2 step 7)
- docs — exact count is non-monotone, search uses qquant+1 scan
- exact §C.1.5.4.4.5/.8 Huffman bit count (Phase 2 step 6)
- §C.1.5.4.4 inner-loop global_gain search (Phase 2 step 5)
- §2.4.3.4.7 quantization primitive (Phase 2 step 4)
- scrub external-name disclaimer (clean-room hygiene)
- §C.1.3 / Figure C.4 polyphase analysis subband filterbank — encoder Phase 2 step 3
- §2.4.3.4.10.3/4 analysis windowing + forward overlap split — encoder Phase 2 step 2
- §2.4.3.4.10.2 forward MDCT — encoder Phase 2 step 1
- §2.4.1.3/§2.4.1.7 Layer III encoder Phase 1 (framing + silent frame)
- standalone-CI: don't assert fixture presence in docs_corpus
- oxideav_core::Demuxer impl with ID3v2 / ID3v1 / Xing-Info / seek
- §2.4.3.2 / Figure A.2 polyphase synthesis subband filterbank
- §2.4.3.4.10 IMDCT + windowing + overlap-add + frequency inversion
- complete Table 3-B.7 codebooks + §2.4.3.4.10.1 alias reduction
- clean-room §2.4.3.4.9 stereo processing (MS + intensity)
- add §2.4.3.4.8 short-block reorder stage
- §2.4.3.4.7 main-data requantization stage
- Layer III Huffman decode stage (clean-room, tables 0..=13)
- Layer III scalefactor decode stage (clean-room)
- add MPEG-2 / MPEG-2.5 LSF single-granule variant
- clean-room MPEG-1 Layer III side-information parser
- clean-room MPEG audio framing layer (header + frame-walk)
- drop fuzz workflow (no fuzz targets in scaffold)
- orphan rebuild — clean-room reset 2026-05-24

### Added

- **`DEFAULT_ATTACK_THRESHOLD` empirical-corpus calibration**
  (Phase 2 step 38, r192). Closes the dual of the r165 leak
  calibration on the encoder-side `attack_detect::AttackDetector`.
  r165 pinned `DEFAULT_AMBIENT_LEAK = 0.5` against a synthetic
  parameter sweep with the threshold knob held fixed at
  `DEFAULT_ATTACK_THRESHOLD = 10.0`; r192 reruns the same corpus
  with the leak now held at the r165-calibrated default and varies
  the threshold axis. Sweep
  `THRESHOLD_SWEEP = [1.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0,
  50.0, 100.0]` spans the qualitative bounds the module doc names
  (≤ 3 over-aggressive, around 10 the recommended sweet spot,
  ≥ 30 conservative). Per-threshold error is summed as
  `max(0, |observed − expected| − tolerance)` across the same
  8 corpus rows r165 used (`steady_sine`, `steady_noise`,
  `isolated_click`, `burst_train_period4`, `slow_swell`,
  `swell_then_click`, `sustained_drum_pair`, `level_shift`).
  Five new tests pin the result:
  `default_threshold_is_an_argmin_over_the_sweep` (no in-domain
  threshold strictly beats `10.0` on the aggregate metric);
  `default_threshold_beats_overaggressive_endpoint_and_ties_conservative`
  (`10.0` strictly beats the over-aggressive endpoint `1.0` —
  aggregate error `179` vs `0` — and ties the conservative endpoint
  `100.0` — both `0`; the rejected region is `[1.0, 3.0]`, the
  transition region is `[5.0, 7.0]`, the acceptable plateau is
  `[10.0, 100.0]`);
  `default_threshold_emits_zero_fires_on_steady_rows` (zero fires
  on `steady_sine` / `steady_noise` at the default);
  `default_threshold_catches_at_least_half_of_burst_train` (the
  burst-train row catches at least 5 of 9 expected hits at the
  default — in practice all 9); and
  `threshold_sweep_is_well_formed` (sorted, positive, finite,
  contains the default, spans the documented bounds).
  Tests: 634 pass (was 629; +5 from this step). cargo clippy
  --all-targets --no-deps -- -D warnings clean; cargo fmt --check
  clean.
  `DEFAULT_ATTACK_THRESHOLD`'s doc-comment is updated with the
  argmin + asymmetric endpoint property; the constant value
  `10.0` is unchanged — the calibration validates the existing
  default, it does not move it.
- **LAME-extension gapless playback wiring** (r185). New
  `lame_tag` module parses the LAME-tag extension that follows the
  four Xing fields inside an MP3's leading information frame,
  exposing all 17 LAME-defined fields including the
  encoder-delay / zero-padding pair that drives gapless playback.
  Provenance: `docs/audio/mp3/lame-xing-info-tag.md` — a clean-room
  transcription of Gabriel Bouvigne's independently-published
  *Mp3 Info Tag revision 1 Specifications* staged 2026-05-29 with
  sha256 chain-of-custody. The byte-offset table is consumed via
  magic-relative offsets (`LAME_MAGIC_OFFSET_ALL_FLAGS = 118`,
  `DELAY_PADDING_OFFSET_FROM_LAME_MAGIC = 23` from the staged
  doc's `$9A` / `$B1` absolutes); the gapless field is the 3-byte
  run at `$B1–$B3` packed `[xxxxxxxx][xxxxyyyy][yyyyyyyy]` to two
  12-bit unsigned values (each 0..=4095). `Mp3Demuxer` now wires
  the parser into its `open()` path: when a Xing tag with all four
  flag bits set is detected, the LAME magic offset is computed via
  `lame_magic_offset(header_bytes, side_info_bytes, &xing)` and
  fed to `parse_lame_tag`; the resulting `LameTag` is stored on
  the demuxer and surfaced through new `.lame()` /
  `.encoder_delay_samples()` / `.zero_padding_samples()` /
  `.trimmed_duration_samples()` accessors. The trimmed-duration
  accessor reports `gross_samples − encoder_delay − zero_padding`
  for LAME-tagged streams and falls back to the gross duration
  for non-LAME and zero-trim streams. Tests +18: 11 module-level
  unit tests in `lame_tag::tests` (byte-pattern propagation of
  the staged-doc §5 worked example
  `[0x6C, 0x12, 0xD2] → delay=1729, padding=722`, exhaustive
  12-bit boundary round-trip across `{0, 1, 2047, 2048, 4094,
  4095}² = 36 (delay, padding) pairs`, all-field round-trip,
  rejection of non-`"LAME"` magic and truncated payloads, MPEG-1
  Layer-III / MPEG-2 LSF `samples_per_frame ∈ {1152, 576}`
  trimmed-sample math, overflow refusal when
  `delay + padding > frames × samples_per_frame`,
  has-gapless-trim predicate), plus 7 demuxer-level integration
  tests (`lame_magic_offset` doc-table matrix for all four
  `(version, channels)` carrier-frame side-info layouts,
  all-four-flag opening, doc worked-example end-to-end through
  `Mp3Demuxer::open`, trimmed-duration math under non-zero
  delay+padding, fallback equality with gross duration when no
  LAME tag is present, fallback equality when LAME tag carries
  zero delay+padding, non-`"LAME"` encoder string yields no LAME
  tag while still keeping the Xing tag). Workspace policy noted:
  parsing only attempted for the **all-four-Xing-flags** layout
  the staged doc documents — other flag combinations return
  `None` and the staged doc would need to be extended to cover
  them. **Spec gap reported (not fished):** the staged doc's
  `$9A–$A4 | 9 bytes` cell is internally inconsistent
  (`$9A..=$A4` inclusive = 11 bytes); the parser trusts the
  absolute-offset chain (`$A5, $A6, …, $BF`) over the
  `9 bytes` annotation, leaving `$A3–$A4` as reserved padding.
  Tag-CRC verification is deferred — the staged doc names
  `CRCInitValue = 0x0000` but the polynomial is unspecified.
- **`oxideav_core::Decoder` trait MPEG-2 LSF widening** (Phase 2
  step 37, r183). Extends `Mp3CoreDecoder` from MPEG-1-only to
  MPEG-1 **and** MPEG-2 LSF Layer III decode (mono and stereo,
  independent / joint MS / joint MS+intensity all carried across).
  The header-version guard in `decode_packet` now accepts
  `MpegVersion::Mpeg1` and `MpegVersion::Mpeg2` and still rejects
  `MpegVersion::Mpeg25` with an `Error::unsupported` whose message
  cites the `docs/audio/mp3/MPEG-2.5-GAP.md` observer-trace gating
  (scalefactor-band tables, low-rate frame-size validation,
  Huffman table mapping at 8 / 11.025 / 12 kHz). The downstream
  decode chain needed no change: `parse_side_info` /
  `decode_scalefactors` / `requantize` / `process_stereo` were
  already version-aware (the LSF single-granule
  `granule_count == 1` layout, the 9-bit `scalefac_compress`
  partitioning of ISO/IEC 13818-3 §2.4.3.4, and the LSF intensity-
  position formula of `stereo::intensity_factors`), and the
  per-channel `imdct_state` / `synth_state` arrays driven by
  `si.granule_count` and `si.channels` consume one granule per
  frame on LSF without further branching. Per-channel sample count
  per emitted `AudioFrame` becomes 576 (LSF, one granule × 576
  PCM samples) vs MPEG-1's 1152 (two granules). +2 new integration
  tests in `tests/decoder_trait_lsf_roundtrip.rs` exercising the
  staged `docs/audio/mp3/fixtures/layer3-mpeg2-22050-64kbps`
  fixture (64 kbps / 22.05 kHz / stereo MPEG-2 LSF Layer III):
  `trait_decode_lsf_stereo_fixture_matches_direct_chain_byte_exact`
  pins the header version + sample-rate + channel count, walks
  every wire frame through both the trait wrapper and the existing
  direct-chain decode primitives, and asserts byte-exact per-channel
  PCM equality; `registry_built_decoder_handles_lsf_stereo_packets`
  drives the same fixture through the registry-installed factory.
  Both tests skip cleanly under standalone-crate CI (workspace
  `docs/` absent) per the `tests/docs_corpus.rs` pattern. +2 new
  lib unit tests in `codec_decoder::tests`:
  `send_packet_rejects_mpeg25_header_pending_observer_trace`
  constructs a real Fraunhofer MPEG-2.5 header via the crate's own
  `make_silent_header` + `write_header` (32 kbps / 11.025 kHz) and
  pins the `Error::Unsupported` rejection with the
  "MPEG-2.5 / observer-trace" message;
  `send_packet_accepts_mpeg2_lsf_header_through_the_guard`
  constructs an MPEG-2 LSF header (64 kbps / 22.05 kHz) and pins
  that the r177-style `"MPEG-1 only"` rejection no longer fires
  for LSF traffic. 611 tests pass total (+4 net from r177's
  baseline).
- **`oxideav_core::Decoder` trait stereo widening** (Phase 2 step 36,
  r177). Extends `Mp3CoreDecoder` from mono-only to MPEG-1 Layer III
  mono **and** stereo (independent `ChannelMode::Stereo` /
  `ChannelMode::DualChannel`, joint MS, joint MS+intensity). The
  per-channel decode state — `ImdctState` for the §2.4.3.4.10.4
  IMDCT overlap memory and `SynthState` for the §2.4.3.2 polyphase
  synthesis shift register — is carried in two-element arrays
  inside the wrapper. Each `send_packet` runs a two-pass per-granule
  decode: pass 1 walks every channel through `decode_huffman` +
  `requantize` and collects the dequantized `xr[576]` lines; on
  `JointStereo` granules the existing `process_stereo` primitive
  rewrites the L/R pair in place per `mode_extension` (MS matrix
  and / or intensity decode per §2.4.3.4.9.1–.9.3); pass 2 runs
  the per-channel `alias_reduce` → `imdct_granule` →
  `synth_granule` tail and writes each channel's PCM into its own
  plane of the emitted `AudioFrame`. The output `AudioFrame`
  switches from interleaved (`data[0]` is the only plane) to
  planar (`data[0]` = L, `data[1]` = R for stereo; single plane
  for mono) per the framework's convention. `make_decoder` accepts
  `channels = 1` or `channels = 2` and rejects every other value
  with `Error::invalid`; the registry factory installed by
  `crate::register` carries the same widening. MPEG-1 only / Layer
  III only / non-free-format guards at `send_packet` are
  unchanged. Existing mono behaviour is preserved bit-for-bit (the
  r141 byte-exact assertion still passes against the
  per-channel-state wrapper using only its `[0]` slot). +4 new
  integration tests in `tests/decoder_trait_stereo_roundtrip.rs`
  (independent-stereo byte-exact match, joint-MS byte-exact match
  with mono-on-L panning to prove the inverse rotation runs inside
  the wrapper, planar `AudioFrame` invariants, and registry-built
  decoder end-to-end) plus 1 net new unit test on `make_decoder`'s
  channel-count validation. 619 tests pass total (was 615; +4
  integration).
- **`DEFAULT_AMBIENT_LEAK` empirical-corpus calibration** (Phase 2
  step 35, r165). Replaces the hand-wave justification for the
  r164-promoted `DEFAULT_AMBIENT_LEAK = 0.5` constant with a
  synthetic-corpus parameter sweep. A 7-row corpus
  (`steady_sine`, `steady_noise`, `isolated_click`,
  `burst_train_period4`, `slow_swell`, `swell_then_click`,
  `sustained_drum_pair`, `level_shift`) covers both leak-knob
  failure-mode axes — slow-leak false-fire on rising envelopes,
  fast-leak missed-fire on sustained transients. The
  `LEAK_SWEEP = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95]` scan
  produces the per-leak aggregate error
  `Σ max(0, |obs − expected| − tolerance)`; four pinned properties
  document the result:
  - `default_leak_is_an_argmin_over_the_sweep` — no in-domain leak
    strictly beats `0.5`.
  - `default_leak_beats_slow_endpoint_and_ties_fast` — `0.5`
    strictly beats `0.05` (err 15 vs 0) and ties `0.95` (both 0).
    The asymmetry is the empirical headline: at the default `10×`
    threshold, the leak/ambient interaction saturates from the fast
    end before the slow end, so the rejected-leak region is
    `[0.05, 0.3]` and the acceptable region is `[0.5, 0.95]`.
  - `default_leak_emits_zero_fires_on_steady_rows` — zero fires on
    the two steady-state baselines at the default leak.
  - `default_leak_catches_at_least_half_of_burst_train` — the
    burst-train row catches `≥ 4` of its 9 expected hits at the
    default leak (in practice all 9 with the `10×` threshold).

  The first granule of each row is discarded as a seed-only call —
  the detector's `ambient` starts at zero and any non-silent first
  granule trips a `ratio = e_max / SILENCE_FLOOR` overflow, so the
  post-seed steady-state is what calibration measures. The
  threshold knob is held at `DEFAULT_ATTACK_THRESHOLD = 10.0`
  throughout; a future round that revisits the threshold should
  re-run the sweep at the new value and tighten the `<=` in
  property 2 into a `<` if the fast-end saturation collapses.
  Threshold-sweep calibration is itself a natural r166+ followup
  out of scope for r165. +6 unit tests (sweep + corpus
  well-formedness, argmin, asymmetric-endpoint, steady-row zero,
  burst-train ≥ half). 602 tests pass (was 596).
- **§2.4.3.4.10 finer attack-detector knobs** (Phase 2 step 34, r164).
  The encoder-side `attack_detect::AttackDetector` previously held a
  single tunable, the subframe-to-ambient ratio `threshold`; the
  ambient-estimate IIR leakage was a private `LEAK = 0.5` constant
  baked into `classify`. This round promotes the leakage factor to a
  per-instance knob via a new `AttackDetectorParams { threshold,
  leak }` value + an `AttackDetector::with_params` constructor.
  `with_threshold` keeps its previous signature (and is now defined
  as `with_params { threshold, leak: DEFAULT_AMBIENT_LEAK }`), so
  every existing caller — `Mp3Encoder::enable_auto_block_type` /
  `enable_auto_block_type_with_mixed`, the
  `make_encoder_*_with_threshold` factories, and all in-tree tests —
  keeps its r163 behaviour bit-for-bit. The new knob is validated by
  the same silently-coerce-to-default contract as the threshold knob:
  leak values outside `(0, 1)` (including the closed-interval
  endpoints, which would freeze or replace the ambient and defeat the
  IIR's purpose), NaN, or infinite all fall back to
  `DEFAULT_AMBIENT_LEAK`. Threshold and leak are validated
  independently — providing one bad knob never drags the other to its
  default. Public surface adds `AttackDetectorParams`,
  `DEFAULT_AMBIENT_LEAK`, and the methods `AttackDetector::with_params`
  / `leak` / `params`. +7 unit tests (default-params constants pin,
  in-domain round-trip, per-knob independent validation,
  `with_threshold` uses default leak, slow-leak vs fast-leak
  behavioural divergence on a repeated burst, boundary `leak == 0.0`
  / `leak == 1.0` rejection, `new == with_params(default)`
  equivalence). 490 tests pass (was 483).
- **§2.4.3.4.9 cross-channel-MS block-type agreement** (Phase 2 step
  33, r163). Closes the gap r162 left open by widening the four
  block-type override toggles (force-short, force-mixed, auto,
  auto-mixed) onto MS-stereo joint modes. The §2.4.3.4.9 same-block-
  type / same-window-switching-flag / same-mixed-block-flag agreement
  is enforced **inside** the encode pre-pass instead of via an API
  reject.
  - **API guard removal.** All four toggle entry points now return
    `Ok` for every channel layout the encoder supports — mono,
    independent stereo, AND MS-stereo joint modes. The previous
    `StreamEncodeError::StereoUnsupported` MS-stereo rejection is
    gone from `force_short_blocks_for_testing`,
    `force_mixed_blocks_for_testing`, `enable_auto_block_type`, and
    (via delegation) `enable_auto_block_type_with_mixed`.
  - **Force-short / force-mixed paths.** Trivially satisfy the
    §2.4.3.4.9 agreement: every (gr, ch) tile emits the same
    `BlockType::Short` from `[[BlockType::Short; 2]; GRANULES]`, and
    force-mixed sets `mixed_block_flag = true` uniformly via
    `default_mixed_gc()` on both channel slots. No new code path
    needed beyond comment / doc updates.
  - **Auto / auto-mixed paths.** Add an `ms_agreement_active` branch
    inside the `block_type_per_gc` pre-pass: when MS-stereo is on,
    each channel's PCM is classified by its own detector (so the
    ambient estimate stays meaningful per channel — a quiet channel
    doesn't drag the loud one's threshold around), but the
    per-channel attack flags + mixed-classifier flags are folded via
    logical OR into a single shared scheduler. The scheduler's
    per-granule `(BlockType, mixed)` emission is mirrored across
    both channel slots of `block_type_per_gc[gr]` and
    `mixed_per_gc[gr]`. The independent-stereo behaviour from r162
    is preserved (each channel runs its own scheduler) for
    `ChannelMode::Stereo` / `DualChannel`. The shared scheduler
    bypasses `scheduler[1]` entirely so its state doesn't drift.
  - **Agreement-rule rationale.** OR-fold is the "safe upper
    envelope": an attack on either L or R triggers the §C.1.5.2
    transition for both. It accepts more short bursts than a
    hypothetical per-channel sequence would (each channel sees the
    other's transients) but never under-resolves a real transient
    on either side, and produces a self-consistent §C.1.5.2 sequence
    across one shared scheduler — no half-formed
    `Start-without-Short` chains the way two independently-stepped
    schedulers might if their attack flags disagreed. Symmetric in
    L↔R by construction.

  Validated by 8 new integration tests and 5 rewrites. The rewrites
  replace the r162 "MS-stereo + toggle → rejected" assertions with
  their "accepted" counterparts. The new tests add wire-level
  §2.4.3.4.9 agreement witnesses: MS-stereo + force-short emits
  agreed Short side-info on both channels of every granule (220 /
  440 Hz interleaved stimulus); MS-stereo + force-mixed similarly
  emits agreed mixed side-info; MS-stereo + auto produces agreed
  per-channel side-info AND responds to a click train on the LEFT
  channel by emitting transition granules (witness that the OR-fold
  engages the shared scheduler when an attack hits either channel,
  not just both). All emit valid `Mp3Demuxer`-acceptable bitstreams.
  Tests: 589 pass (was 586 at r162; +3 net = +8 new − 5 rewrites).

- **§2.4.3.4.9 independent-stereo widening of the block-type override
  toggles** (Phase 2 step 32, r162). Narrows the long-standing
  "force-short / force-mixed / auto / auto-mixed are mono-only"
  restriction to its actual spec basis. The §2.4.3.4.9 same-block-type
  requirement only binds on MS-stereo joint modes (the matrix
  `M = (L+R)/√2`, `S = (L-R)/√2` rotates L/R before quantize and the
  decoder needs both halves to share window geometry); independent
  stereo (`ChannelMode::Stereo` / `ChannelMode::DualChannel` without
  joint coupling) carries per-channel side-info verbatim per
  §2.4.1.7 / §2.4.2.7 and has no such constraint.
  - **New private predicate** `Mp3Encoder::ms_joint_stereo_active(&self)
    -> bool` returns `self.ms_stereo || self.ms_auto_threshold.is_some()`
    — true when an MS-stereo joint mode is configured (built via
    `new_joint_stereo_ms` or `new_joint_stereo_auto`).
  - **API guard relaxation.** The four block-type override entry
    points reject the encoder only when `ms_joint_stereo_active()`
    is true, instead of when `nch != 1`:
    - `Mp3Encoder::force_short_blocks_for_testing`
    - `Mp3Encoder::force_mixed_blocks_for_testing`
    - `Mp3Encoder::enable_auto_block_type`
    - `Mp3Encoder::enable_auto_block_type_with_mixed`
  - **No encode-pipeline changes** beyond the guard + comment updates.
    The downstream loop was already per-(gr, ch): the
    `block_type_per_gc[gr][ch]` matrix has iterated `0..self.nch`
    since r156, `AutoBlockTypeConfig` already sized its detector /
    scheduler / mixed-classifier vectors to `nch`, and the MDCT /
    gc-template / outer-loop branches all index per-channel. Per-
    channel scheduler independence means independent-stereo auto
    behaves correctly without further wiring: a click train on the
    left and a sustained sine on the right produces non-Long
    granules on channel 0 and Long-only on channel 1 in the same
    frame.
  - **MS-stereo gap unchanged.** `new_joint_stereo_ms` /
    `new_joint_stereo_auto` encoders still return
    `StreamEncodeError::StereoUnsupported` from the four toggles
    pending the §2.4.3.4.9 cross-channel-MS agreement wiring.

  Validated by 11 new integration tests across
  `tests/short_block_encoder_roundtrip.rs`,
  `tests/mixed_block_encoder_roundtrip.rs`,
  `tests/auto_block_type_roundtrip.rs`, and
  `tests/auto_block_type_mixed_roundtrip.rs`: MS-stereo and MS-auto
  rejection for each of the four toggles; independent-stereo and
  dual-channel acceptance for each; and end-to-end stereo wire +
  `Mp3Demuxer` round-trip tests covering force-short, force-mixed,
  and a per-channel auto-scheduler witness that drives the left
  channel with a click train and the right with a sustained sine,
  then asserts the emitted side-info carries non-Long granules in
  channel 0 but stays Long-only in channel 1. Existing
  rejected-on-stereo tests were rewritten to reject MS-stereo joint
  modes (still failing per §2.4.3.4.9) rather than independent
  stereo. Tests: 586 pass (was 575 at r161; +11 integration). No
  external implementation consulted.

- **§2.4.3.4.10.3 auto-block-type mixed-block promotion** (Phase 2
  step 31, r161). Closes the long-standing "auto path can never emit
  Mixed" gap by adding a clean-room PCM-domain mixed-vs-pure-short
  classifier that runs alongside the §C.1.5.2 attack detector + state
  machine in `Mp3Encoder::enable_auto_block_type_with_mixed`.
  - **`mixed_classifier` module.** New `MixedClassifier` applies a
    one-tap moving-average low-pass kernel
    `y[n] = (x[n] + x[n-1]) / 2` (transfer function `|cos(ω/2)|`,
    unity DC gain, nulls Nyquist) to each granule's PCM and compares
    the per-subframe energies of the low-passed signal. If the
    max-to-min ratio stays at or below a caller-chosen threshold
    (default `DEFAULT_MIXED_LOW_BAND_STABILITY = 4.0`) the low band
    is judged stationary across the granule and the mixed carve-out
    is appropriate; otherwise pure-short is preferred. The
    classifier carries the previous granule's last sample across
    boundaries so the LP filter is continuous.
  - **`BlockTypeStateMachine::step_with_mixed`.** Extends the
    scheduler with a per-call `prefer_mixed: bool` and returns
    `(BlockType, bool)`. The mixed flag is set only on Short
    emissions (§2.4.2.7's syntactic invariant that
    `mixed_block_flag` is meaningful only for `block_type == 2`);
    the legacy `step` delegates with `prefer_mixed = false` so all
    prior callers keep their pure-short behaviour.
  - **`Mp3Encoder::enable_auto_block_type_with_mixed`.** New opt-in
    entry point with the same lookahead / detector / scheduler
    wiring as `enable_auto_block_type` plus a per-channel
    `MixedClassifier`. The pre-pass classifies every granule in
    parallel with the attack detector and feeds the boolean
    preference to `step_with_mixed`; the resulting `mixed_per_gc`
    matrix drives the forward MDCT branch (subbands 0..1 take the
    36-point long sine window, subbands 2..31 the three 12-point
    short windows — same dispatch as `force_mixed_blocks`) and the
    `gc_template` selection (`default_mixed_gc()` on mixed
    emissions). The existing r159 `outer_loop_search_mixed`
    primitive is reused via the `gc_template.mixed_block_flag`
    discriminator in the outer-loop branch — no further outer-loop
    wiring required. Mono-only and mutually exclusive with the
    force-toggles, inherited from `enable_auto_block_type`.

  Validated by 10 new unit tests in `mixed_classifier.rs` (silent
  granule degenerate case; DC stability; high-frequency-only attack
  judged mixed-appropriate; broadband attack judged pure-short;
  cold-start conservative-pure-short boundary case; LP unity-DC
  and Nyquist-null checks; threshold validation; reset and
  prev_last tracking) plus 4 new unit tests in `block_type_sm.rs`
  (`step_with_mixed(_,_,false)` matches `step` byte-for-byte;
  `prefer_mixed=true` sets the flag only on Short emissions;
  sustained-burst flag propagation; per-call preference toggling)
  plus 7 new integration tests in
  `tests/auto_block_type_mixed_roundtrip.rs` (stereo rejection;
  threshold round-trip; force-toggle clearing; plain auto path
  stays unmixed; low-band-DC + Nyquist-click stimulus engages ≥ 1
  mixed granule while the plain auto path on the identical PCM
  emits zero mixed granules; pure-sine stays Long under mixed-auto;
  mixed-auto + outer-loop combination engages
  `outer_loop_search_mixed` end-to-end with `scalefac_compress = 15`
  on every mixed granule and Mp3Demuxer round-trip acceptance). All
  Mp3 tests now: **575 pass** (was 554 at r160; +14 unit + 7
  integration).

- **§C.1.5.4.3 outer-loop long-family transition-skeleton wiring**
  (Phase 2 step 30, r160). `outer_loop_search_long` widens from
  pure-Long (`block_type == Long`, `window_switching_flag == false`)
  to the full long-family `block_type ∈ {Long, Start, End}`. Start
  (block_type 1) and End/Stop (block_type 3) carry the same 21 long
  scalefactor bands as Long (§2.4.2.7 + Table 3-B.5), share the
  §2.4.3.4.7.1 long-block requantize formula (no `subblock_gain`
  term), and use the same §C.1.5.4.4.6 region split rule
  (`region0_count` / `region1_count`-ignoring 1/3, 5/12, 1/4
  partition driven by `big_values` alone), so a single primitive
  covers all three correctly. Three details:
  - **Debug-assert relaxation.** The pure-Long check at
    `outer_loop_search_long`'s entry becomes a long-family check
    accepting `(Long, false)`, `(Start, true)`, or `(End, true)`
    and rejecting `mixed_block_flag` (which only applies to
    `block_type == Short`).
  - **Dispatcher widening.** `outer_loop_eligible` extends from
    `(false, Long, _) | (true, Short, _)` to also include
    `(true, Start, false) | (true, End, false)`. The
    `BlockType::Start | BlockType::End` match arm — previously a
    `debug_assert!(false)` unreachability marker — now routes onto
    the same `outer_loop_search_long` call as `BlockType::Long`,
    with `subblock_gain = [0; 3]` (no subblock_gain on the
    long-family branch).
  - **Auto block-type integration.** With `enable_auto_block_type`
    + `new_with_outer_loop` both on, every block-type the §C.1.5.2
    `LONG → START → SHORT → STOP → LONG` scheduler emits now runs
    the outer loop. Previously Start / End granules fell back to
    the fixed-gain inner-loop-only path with
    `scalefac_compress = 0`; they now seed the
    `OUTER_LOOP_SCALEFAC_COMPRESS = 15` signature on the wire and
    carry the chosen per-band scalefactors as part2 at slen1 = 4 /
    slen2 = 3.

  Validated by 5 new unit tests in `outer_loop.rs` (Start/End
  templates terminate on a huge threshold; behavioural identity
  between Long, Start, and End templates on identical `xr` —
  including scalefactors, `global_gain`, `scalefac_scale`,
  `preflag`, and the full `is[576]` output; Start template amplifies
  ≥ 1 band under a tiny threshold) and 2 new integration tests in
  `tests/auto_block_type_roundtrip.rs` (every Start / End granule
  carries the `scalefac_compress = 15` outer-loop wire signature on
  the click-train stream; the resulting bytestream remains
  demuxer-acceptable end-to-end). All Mp3 tests now: **554 pass**
  (was 547 at r159; +5 unit + 2 integration).

- **§C.1.5.4.3 outer-loop mixed-block analogue** (Phase 2 step 29).
  `outer_loop_search_mixed` composes the long-region per-band amplifier
  (sfb 0..=7) with the short-region per-(sfb, window) amplifier
  (sfb 3..=11) for `block_type == Short`, `mixed_block_flag == true`,
  `window_switching_flag == true` granules — the missing third
  primitive after `outer_loop_search_long` (r144) and
  `outer_loop_search_short` (r157). Five details:
  - **§C.1.5.4.3.6 caps under the mixed MPEG-1 part2 layout.** Every
    long band reads at slen1 (cap 15 across sfb 0..=7) — distinct from
    the pure-long path where `mpeg1_long_band_slen` would split at
    sfb 11. Short region splits as cap 15 on sfb 3..=5 (slen1) and
    cap 7 on sfb 6..=11 (slen2). New `MIXED_SCALEFAC_L_MAX`,
    `MIXED_FIRST_SHORT_SFB`, `MIXED_LAST_LONG_SFB` constants document
    the layout that mirrors
    `crate::scalefactors::write_mpeg1_granule_channel`'s mixed branch.
  - **Per-band distortion** `band_distortion_mixed_long` and
    `band_distortion_mixed_short` compute the §C.1.5.4.3.3 SSE only on
    the cells the mixed layout actually carries (long sfb 0..=7,
    short sfb 3..=11). Cells outside those ranges stay 0.0 so the
    amplifier never touches them. The long helper omits the `PRETAB`
    term `band_distortion_long` carries (§2.4.2.7 disables preflag on
    every short-family granule including mixed).
  - **Bounded `subblock_gain` search** reuses the pure-short loop's
    §C.1.5.4.4.2 magnitude-clamp follow-up — the short region's
    per-window magnitudes are what the clamp fails on, and bumping
    `subblock_gain[w]` divides window `w`'s reconstruction by 4 per
    step (saturating at the §2.4.2.7 3-bit cap of 7). The long
    region's reconstruction does NOT use `subblock_gain`
    (§2.4.3.4.7.1: the subblock_gain term only appears in the short
    reconstruction branch).
  - **`scalefac_scale = 0 → 1` escalation** halves every in-progress
    per-band scalefactor on BOTH regions (`sf.long[0..=7]` and
    `sf.short[3..=11][..]`) so the coloured spectrum is preserved
    across the scale switch; one event only.
  - **Stream encoder wiring.** `outer_loop_eligible` widened from
    `(false, Long, _) | (true, Short, false)` to
    `(false, Long, _) | (true, Short, _)`; the `BlockType::Short`
    match arm splits on `mixed_block_flag` and routes mixed onto
    `outer_loop_search_mixed`. Composing
    `Mp3Encoder::new_with_outer_loop(...)` with
    `force_mixed_blocks_for_testing(true)` now drives every assembled
    granule through the new path; the wire signature is
    `scalefac_compress = 15` on every (gr, ch) (was 0 under the r158
    fixed-gain fallback).

  Validated by 11 new unit tests inside `outer_loop.rs` (mixed
  constant-vs-spec alignment, mixed band-distortion identity +
  absorbed-band invariant, mixed termination paths for huge / tiny
  threshold, region-isolation tests confirming the long amplifier
  fires only on long-region energy and the short amplifier only on
  short-region energy, `subblock_gain` quiet-input invariance,
  `subblock_gain` escalation on extreme window-0 amplitudes, the
  `scalefac_scale` escalation branch on a cap-would-terminate fixture)
  plus 4 new integration tests in
  `tests/mixed_block_encoder_roundtrip.rs` covering the new force-mixed
  + outer-loop dispatch (scalefac_compress = 15 wire signature on
  every (gr, ch), subblock_gain field bounded ≤ 7, finite + non-silent
  PCM roundtrip via the short-aware decode chain, and Mp3Demuxer
  acceptance of the new bitstream). Tests: 547 pass (was 532 at r158;
  +11 unit + 4 integration).1.5.4.3 / §2.4.2.7 / §2.4.3.4.7.1 of
  ISO/IEC 11172-3:1993 and from this crate's own r144 / r157
  primitives. No `[package] version` bump.

- **Auto block-type × outer-loop integration** (Phase 2 step 28).
  Wires the r157 `outer_loop_search_short` primitive into
  `Mp3Encoder::assemble_frame_with_lookahead`, completing the missing
  half of the §C.1.5.4.3 distortion-control coverage for the auto
  scheduler. Three changes:
  - `enable_auto_block_type` no longer rejects encoders configured
    with `new_with_outer_loop`. The pair-rejection from r156 was a
    placeholder pending the short-block primitive; r157 added the
    primitive, and r158 dispatches onto it.
  - The per-(gr, ch) outer-loop arm in `assemble_frame_with_lookahead`
    inspects the granule's selected block type and dispatches:
    `BlockType::Long` → `outer_loop_search_long` (the r144 path);
    `BlockType::Short` with `mixed_block_flag == false` →
    `outer_loop_search_short`; `BlockType::Start` / `BlockType::End`
    (long-family transition skeletons) fall back to the fixed-gain
    inner-loop path (no outer-loop primitive covers transition
    skeletons yet — their §2.4.2.7 coefficient distribution shifts
    mid-overlap so the uniform-`xmin` heuristic over-amplifies; a
    follow-up round will target them with a psy-aware threshold).
    Mixed-block Short is a separate follow-up (cf. r157 followup #1).
  - `subblock_gain[w]` returned by `outer_loop_search_short` is now
    propagated into the granule-channel's side-info field; the
    §C.1.5.4.4.5 part2 / part3 budget split also tracks the per-block
    part2 cost (Long: 74 bits, pure-short: 126 bits, mixed: 122
    bits) so the inner-loop budget check is bit-accurate per shape.
  - `tests/auto_block_type_roundtrip.rs` rewrites the r156 rejection
    test as a positive integration test: a click-train PCM is encoded
    through `new_with_outer_loop` + `enable_auto_block_type`,
    `FrameWalker` parses every emitted frame, the §2.4.2.7 invariants
    (`preflag == false` on short, `subblock_gain[w] <= 7`) are
    verified, the demuxer accepts the stream, and at least one
    `BlockType::Short` granule is witnessed so the new dispatch path
    is provably exercised. Tests: 532 pass (same as r157; one
    rejection test rewritten in place per guardrail #3).

- **§C.1.5.4.3 outer-loop short-block analogue** (Phase 2 step 27).
  `outer_loop_search_short` is the per-(sfb, window)
  distortion-control iteration the auto-block-type dispatcher from
  step 26 needs to run with the outer loop on for `block_type ==
  Short`, `mixed_block_flag == false` granules. Mirrors
  `outer_loop_search_long` with:
  - `band_distortion_short(xr, xr_back, sf, scalefac_scale, sr,
    ver)` returning the §C.1.5.4.3.3 distortion per `(sfb, window)`
    cell. Each iteration marks every cell with `xfsf_s > xmin` and
    amplifies the marked cells' `scalefac_s` by 1. Caps follow
    §C.1.5.4.3.6 with `OUTER_LOOP_SCALEFAC_COMPRESS = 15`
    (slen1 = 4, slen2 = 3): 15 for the slen1-range sfb 0..=5, 7 for
    the slen2-range sfb 6..=11.
  - A bounded per-window `subblock_gain` search: when
    `search_magnitude_clamp` reports `satisfied == false`,
    `per_window_max_abs` identifies the over-cap windows and bumps
    each one's `subblock_gain[w]` by 1 (saturating at the §2.4.2.7
    3-bit cap of 7), then restarts. Quiet windows stay at zero.
  - `scalefac_scale = 0 → 1` escalation on cap-would-terminate, with
    round-to-nearest halving of every in-progress `scalefac_s` (same
    §C.1.5.4.3 path as the long-block loop). One event only.
  - `preflag` stays `false` unconditionally (§2.4.2.7: "preflag is
    never used if block_type == 2").

  Pure-short only this round; mixed-block variant is a follow-up.
  Integration into `Mp3Encoder::enable_auto_block_type` is a
  separate step. Validated by 9 unit tests inside `outer_loop.rs`
  covering termination paths, per-cell amplification isolation,
  `subblock_gain` escalation on extreme amplitudes, quiet-input
  invariance, preflag invariant, and the `scalefac_scale`
  escalation branch on a cap-would-terminate fixture. 
- **Signal-driven auto block-type dispatch** (Phase 2 step 26).
  Replaces the global force-toggles with a per-granule decision
  driven by content. Two new modules carry the logic:
  - `attack_detect` — a stateful `AttackDetector` that splits each
    granule's 576 PCM samples into three 192-sample subframes
    (matching the Layer III three-window short-block partition of
    §2.4.2.7), computes per-subframe sum-of-squares energy, and flags
    the granule as carrying an attack iff the loudest subframe
    exceeds `threshold ×` the running ambient (an exponentially
    smoothed `min`-floor of recent subframe energies with leakage
    `0.5`). Default ratio `10.0`; module docs explain tuning. Every
    constant is justified by the clean-room reasoning in the
    module's preamble.
  - `block_type_sm` — the §C.1.5.2
    `LONG → START → SHORT → STOP → LONG` transition state machine
    that turns the per-granule attack flags into geometrically
    valid `BlockType` decisions. Takes one granule of lookahead
    (`step(cur_attack, next_attack)`) so a `Start` window is
    committed in time to splice into the next granule's `Short`
    head; covers sustained bursts (`Short → Short`), burst-then-quiet
    (`Short → End → Long`), and back-to-back bursts separated by at
    least one Long granule. A `cur_attack` without a `next_attack`
    is conservatively dropped (no `Start` available; falls back to
    `Long`) — documented in the module preamble.
  - `Mp3Encoder::enable_auto_block_type(threshold)` opt-in API
    wires the detector + scheduler into the per-frame assemble
    path. The push/finish API contract is preserved: the
    `push_samples` loop holds back one extra granule as the
    scheduler's lookahead while still emitting one frame per
    1152-sample chunk in steady state; `finish` zero-pads the
    held-back tail with a "no attack ahead" lookahead.
  - Mutually exclusive with `force_short_blocks_for_testing`,
    `force_mixed_blocks_for_testing`, and the outer loop (the
    long-block-only `outer_loop_search_long` doesn't yet handle
    Short / Start / End granules); enabling auto clears the
    force-toggles and vice versa, and a configured outer loop
    rejects the auto enable at API time.
  - Mono-only this round; cross-channel block-type agreement
    (§2.4.3.4.9) for stereo / joint / dual-channel auto block-type
    deferred to a follow-up.
  - Side-info wiring: the encoder emits the correct
    `window_switching_flag` / `block_type` per granule, dispatches
    the MDCT path (long-family 36-point with the Start / End
    asymmetric windows from `mdct::window_long_family_analysis`,
    three 12-point short MDCTs via
    `short_block::forward_short_mdct_subband` for Short granules),
    and gates inverse alias reduction on `block_type != Short` per
    §2.4.3.4.10.1.
  - Validated by 27 new tests: 10 unit tests in `attack_detect`
    (silent / unit-DC subframe energies, pure-sine no-fire,
    step-burst flagged, pure-silence not flagged + bounded
    ambient, detector-adapts-after-bursts, click-after-silence
    flagged, invalid-threshold fallback, reset clears ambient),
    8 unit tests in `block_type_sm` (all-calm Long-only,
    single-burst Start/Short/Stop/Long emission, sustained-burst
    holds Short, two-bursts with Long gap, current-only attack
    without lookahead falls back to Long, reset, Start→Short
    invariant, Stop→Long invariant), and 9 integration tests in
    `tests/auto_block_type_roundtrip.rs` covering the API surface
    (stereo rejection, default-off, enable/disable round-trip,
    mutual exclusion with force-short / force-mixed / outer-loop),
    pure-sine staying long, click-train engaging Start / Short / End
    with §C.1.5.2 transition-validity assertions on every emitted
    pair, and demuxer-acceptance of the auto stream.

- **MPEG-2.5 frame-parser support** (Phase 2 step 25). The framing
  layer now accepts the proprietary Fraunhofer-IIS "MPEG-2.5"
  extension documented in `docs/audio/mp3/MPEG-2.5-GAP.md` (Popp /
  Brandenburg, EBU Technical Review 283; Fraunhofer-IIS U.S. patent
  RE44,897; datavoyage community header reference). The §2.4.2.3
  syncword is narrowed from 12 to 11 bits (`'1111 1111 111'` at
  header positions 31..21), and the resulting 2-bit version field at
  positions 20..19 is decoded as `'11'` = MPEG-1, `'10'` = MPEG-2
  LSF, `'01'` = reserved (new `HeaderError::ReservedVersion`),
  `'00'` = MPEG-2.5. A new third `MpegVersion::Mpeg25` enum variant
  carries through the rest of the framing stack: `samples_per_frame`
  returns 576 like MPEG-2, `frame_len` uses the 72-byte Layer-III
  coefficient, the V2,L1 / V2,L2&L3 bitrate ladders are reused
  (Fraunhofer patent "applied to ISO/IEC 13818-3"), the §13818-3 LSF
  side-info layout / scalefactor decode / stereo intensity factors /
  Xing-frame side-info-bytes / encoder `side_info_bytes` / demuxer
  `side_info_len` all dispatch on a new `MpegVersion::is_lsf()`
  helper. A new `SAMPLE_RATE_V25 = [11_025, 12_000, 8_000]` table
  is added per the patent's "preferably half the sampling rate"
  formulation and the datavoyage table. The encoder's `write_header`
  is re-grounded on the 11-bit-sync layout and a new
  `version_bits(MpegVersion) -> u32` returns the 2-bit version
  field. `make_silent_header` accepts the three MPEG-2.5 sample
  rates and infers the new version. The `oxideav_core::Decoder`
  trait wrapper this round is still MPEG-1-only and rejects
  MPEG-2.5 with the same "decoder this round is MPEG-1 only"
  message it returns for MPEG-2 LSF. Validated by 11 new unit tests
  in `src/frame.rs` (MPEG-2.5 32 kbps / 11.025 kHz parse with
  576-sample / 208-byte invariants; V2,L1&L23 ladder pinning at the
  low and high ends; 8 kbps / 8 kHz / +padding frame-length pin;
  all-three sample-rate table pin; reserved-version `'01'`
  rejection; first-two-byte wire-format invariant; FrameWalker
  iterates back-to-back MPEG-2.5 frames; FrameWalker iterates a
  mixed MPEG-1 + MPEG-2.5 stream; `is_lsf` groups MPEG-2 and
  MPEG-2.5) and 3 new unit tests in `src/encoder.rs` (writer ↔
  parser inverse on the new version; all three MPEG-2.5 sample
  rates round-trip through the writer; `make_silent_header`
  accepts the MPEG-2.5 rate set). Net: 496 tests pass (was 474).
- **§C.1.5.4.4.8 linbits-reach filter in
  `huffman::choose_best_table_for_region`** (Phase 2 step 24,
  #1106). The §B.7 codebooks have widely-varying magnitude reach —
  the small tables 0..=15 reach `xlen - 1` (no linbits escape), the
  ESC tables 16..=31 reach `15 + (2^linbits - 1)` — and the
  pre-r154 chooser only checked the codebook's `xlen` corner. The
  decoder's `decode_big_pair` clamps the Huffman symbol to 15
  before lookup, so the corner test was identically satisfied by
  every ESC table regardless of magnitude. The encoder then
  silently truncated the value at emission time: a range with
  `|is| = 100` would pick e.g. table 16 (`linbits=1`, reach 16),
  and `emit_big_pair` would write `(100 - 15) & 0x1 = 1` instead
  of the full delta, producing a decoded `15 + 1 = 16`. The new
  `huffman::big_table_reach(idx)` public helper returns the
  per-codebook reach, and `choose_best_table_for_region` now drops
  candidates whose reach is less than the range's `max|is|`. The
  duplicate reach lookup local to `stream_encoder::best_table_or`
  is removed (the function collapses to a thin wrapper around the
  in-tree chooser). Eight new unit tests pin `big_table_reach` to
  the §B.7 transcribed `xlen` / `linbits` values, exercise the
  chooser's filter at the magnitude-15 / -16 / -8191 boundaries,
  assert the all-zero / empty-range fallbacks, and assert that a
  magnitude beyond every codebook's reach (`|is| = 9000`) returns
  `None` rather than silently truncating. One `inner_loop` test
  that inadvertently relied on the silent-truncation behaviour
  (its `flat(30.0)` spectrum at very fine `global_gain` exceeded
  the §C.1.5.4.4.2 magnitude clamp) is tightened to walk only the
  clamp-respecting subset of the gain range that
  `search_bit_budget` itself walks. 474 tests pass (was 466).
- **§2.4.2.7 forward mixed-block MDCT path on the encode side**
  (Phase 2 step 23) — new `Mp3Encoder::force_mixed_blocks_for_testing`
  toggle that drives every assembled granule onto the mixed-block
  encode path. Subbands 0 and 1 (the lowest 36 frequency lines) are
  coded with the long-family forward MDCT (`forward_overlap →
  window_long_family_analysis(Long) → 36-point mdct → ÷9`,
  identical to the long-block branch); subbands 2..31 are coded
  with the short-block forward MDCT (`forward_short_mdct_subband`).
  `forward_reorder` is then invoked with a mixed `GranuleChannel` so
  the long region (lines 0..36) passes through unchanged while the
  short region's SFB 3..12 is rewritten into native bitstream
  `[sfb][win][k]` order. No inverse alias reduction (the decoder's
  `alias_reduce` tests `block_type == Short` and returns unchanged
  for both pure short and mixed granules). The per-granule-channel
  side info carries `window_switching_flag = 1`, `block_type = Short`,
  `mixed_block_flag = 1`, and the §2.4.2.7-default region sentinels
  (decoder-reconstructed at parse time). Mono-only this round (same
  §2.4.3.4.9 cross-channel block-type-agreement gap that gates the
  pure-short toggle); mutually exclusive with
  `force_short_blocks_for_testing` (enabling one clears the other).
  A new `tests/mixed_block_encoder_roundtrip.rs` integration covers
  the side-info contract, the decoder round-trip (encoded PCM →
  `huffman → requantize → reorder → alias → imdct → synth` →
  finite-energy non-silent reconstruction), the mutual-exclusivity
  rule, and the stereo-encoder rejection.
- **§2.4.3.4.10.2 forward short-block MDCT path on the encode side**
  (Phase 2 step 22): new `short_block` module exposing
  `forward_short_mdct_subband` (three independent 12-point MDCTs per
  polyphase subband over the lapped 36-sample frame, output in the
  subband-window-interleaved layout `out[3·k + win]` the decoder's
  `imdct.rs::windowed_block` short branch consumes), `forward_reorder`
  (bit-exact inverse of `crate::reorder::reorder`: subband-window-
  interleaved → native bitstream `[sfb][win][k]`), and
  `short_block_region_defaults` (the spec-default `region_address1 = 8`,
  `region_address2 = 36` short-block sentinels per §2.4.2.7, clamped to
  the 3-bit `region1_count` cap). The new `MdctState::from_saved`
  constructor exposes the per-subband forward-overlap memory so the
  short-block path can update state atomically at the end of its MDCT
  chain. The `crate::mdct::window_short_analysis` and 12-point
  `crate::mdct::mdct` primitives that ship since round 130 carry the
  windowing and transform; this round wires them together with the
  scale factor (`n/4 = 3` Princen-Bradley for `n = 12`) and the
  subband-window layout the decoder reads.
- **`Mp3Encoder::force_short_blocks_for_testing` toggle** integrating
  the forward short-block primitive into the PCM → MP3 stream
  encoder. With the toggle on (mono only this round; multi-channel
  short-block encode needs the §2.4.3.4.9 cross-channel block-type
  agreement wiring that lands in a follow-up), every assembled
  granule emits a §2.4.2.7 short block: forward analysis runs
  `forward_short_mdct_subband` per subband instead of long-block
  forward-overlap + 36-point MDCT; no alias reduction (§2.4.3.4.10.1
  scopes it to `block_type != 2`); `forward_reorder` rewrites the
  bins into the bitstream `[sfb][win][k]` interleave; the
  per-granule-channel side info carries `window_switching_flag = 1`,
  `block_type = Short`, `mixed_block_flag = 0`,
  `subblock_gain = [0; 3]`, and the spec-default region sentinels.
  The §C.1.5.4.4.6 short-block region split (region 0 hardcoded to
  the first 36 lines, region 1 to the rest of big_values, region 2
  empty) is honoured by the inner loop's region-end + table-select
  pass so the encoder's emitted big-values cost matches the decoder's
  `huffman::region_boundaries` short-block override. The toggle is a
  deterministic test handle for the encode-side primitive; the
  signal-driven §C.1.5 attack-detection auto-decision heuristic +
  the LONG → START → SHORT → STOP → LONG transition state machine
  required for *mixed* long-and-short streams remain a follow-up
  round.
- **`tests/short_block_encoder_roundtrip.rs`** integration test
  battery: confirms (a) the toggle rejects multi-channel encoders,
  (b) the long-baseline stream still carries only long granules,
  (c) the force-short stream's emitted side info carries the
  expected `(window_switching_flag, block_type, mixed_block_flag,
  subblock_gain)` quartet on every granule-channel, (d) the
  force-short stream is accepted by `Mp3Demuxer::next_packet`, and
  (e) the force-short stream decodes end-to-end through the crate's
  own primitives (`huffman → requantize → reorder → alias → imdct
  → synth`) to finite, non-silent PCM with audible zero crossings.
  Five new tests; runs in well under a second. Four new in-module
  unit tests on `short_block` exercise the forward-reorder ↔ decoder-
  reorder roundtrip (44.1 kHz pure-short), the long-block identity
  pass-through, the mixed-block long-region preservation, the
  per-subband forward-MDCT chain's energy bound, and the
  `MdctState::from_saved` constructor symmetry.
- **Trait-factory wrappers for joint-stereo auto MS/LR encoding**
  (Phase 2 step 21): `codec_encoder::make_encoder_joint_stereo_auto`
  and `make_encoder_joint_stereo_auto_with_threshold` reach the new
  `Mp3Encoder::new_joint_stereo_auto` constructor (Phase 2 step 20)
  through the framework's `oxideav_core::Encoder` factory shape, so
  trait-only consumers can opt into the per-frame MS/LR picker without
  going through the direct `Mp3Encoder` API. The threshold variant
  exposes `Mp3Encoder::with_ms_auto_threshold`'s `[0.0, 1.0]`
  clamping. Honours the workspace dual-API convention: both the
  direct `Mp3Encoder::new_joint_stereo_auto` constructor and the trait
  factory landed on the same step, neither one is the "preferred"
  entry point. The bit-rate validation is now factored out into a
  `validate_joint_stereo_params` helper shared with the existing
  `make_encoder_joint_stereo_ms` factory; the wrapper buffering /
  `flush`-time slicing in `Mp3CoreEncoder` is unchanged (the new
  factories build the same trait-object wrapper around a
  joint-stereo-armed `Mp3Encoder`). Validated by six new unit tests
  on `codec_encoder` (`make_encoder_joint_stereo_auto_emits_picked_mode_extension`
  — proves correlated stereo selects `mode_extension = '10'`;
  `make_encoder_joint_stereo_auto_with_threshold_threshold_zero_forces_lr`
  — proves `threshold = 0` suppresses MS on any non-trivial side
  energy; `make_encoder_joint_stereo_auto_rejects_mono` /
  `make_encoder_joint_stereo_auto_requires_sample_rate` /
  `make_encoder_joint_stereo_auto_defaults_bitrate_to_192k` /
  `make_encoder_joint_stereo_auto_with_threshold_clamps_out_of_range`).

- **§2.4.2.3 joint-stereo auto MS/LR per-frame picker**
  (Phase 2 step 20): a new constructor
  [`Mp3Encoder::new_joint_stereo_auto`] arms the encoder in joint mode
  (header `mode = '01'`) and decides per frame whether to apply the
  §2.4.3.4.9.2 MS rotation, based on a content-only energy heuristic.
  For each granule of the frame the post-MDCT spectra are examined
  and the side-channel energy fraction
  `E_S / (E_L + E_R) = Σ(L−R)² / (2·Σ(L² + R²))` is computed; when
  **both** granules of the frame come in at or below the configured
  threshold (default `0.5`), the §2.4.3.4.9.2 rotation
  `M = (L+R)/√2`, `S = (L-R)/√2` is applied and the frame's
  `mode_extension` is written as `'10'` (ms_stereo on,
  intensity_stereo off); otherwise the LR channels are passed through
  unchanged and `mode_extension` is written as `'00'`. The
  per-granule rejection short-circuit honours the §2.4.3.4.9
  "both granules of a frame share the same joint-stereo method"
  semantics for free — `mode_extension` is a per-frame wire field,
  not a per-granule one. ISO/IEC 11172-3 does **not** prescribe an
  encoder mode-decision algorithm (§2.4.2.3 fixes only the wire
  syntax), so the energy heuristic is a clean-room encoder choice
  using no psychoacoustic input. The `0.5` default is the symmetry
  boundary: the rotation is unitary so `E_M + E_S = E_L + E_R`, and
  below `0.5` the mid channel carries strictly more energy than
  either L or R, which the inner-loop bit-budget gain search
  exploits. `Mp3Encoder::with_ms_auto_threshold(t)` overrides the
  threshold (values are clamped into `[0.0, 1.0]`; the setter is a
  no-op when called on an encoder that was not constructed via
  `new_joint_stereo_auto`). `Mp3Encoder::ms_auto_threshold()` reads
  back the configured threshold. The picker leaves the existing
  unconditional `new_joint_stereo_ms` path (round 146) untouched —
  the two armings drive the same forward-MS branch in
  `assemble_frame` but only one is active at a time. Validated by
  six new unit tests (`auto_ms_picker_default_threshold_is_half` /
  `auto_ms_picker_threshold_override_clamps` /
  `auto_ms_picker_threshold_override_noop_on_non_auto` /
  `auto_ms_picker_correlated_input_chooses_ms` /
  `auto_ms_picker_anticorrelated_input_chooses_lr` /
  `auto_ms_picker_zero_threshold_forces_lr_on_any_side_energy`) and
  four integration tests
  (`auto_picker_correlated_one_second_self_decode_psnr` —
  84 dB / 85 dB per-channel PSNR on a 1 s correlated 440 Hz tone at
  192 kbit/s, matching the always-MS path;
  `auto_picker_anticorrelated_steady_state_picks_lr` /
  `auto_picker_mixed_stream_flips_mode_extension_mid_stream` —
  proves the decision is genuinely per-frame, not encoder-wide;
  `auto_picker_silence_does_not_panic` — handles the
  zero-energy granule edge case without dividing by zero).

- **§C.1.5.4.3.4 preemphasis decision in the outer loop**
  (Phase 2 step 19): the outer (distortion-control) loop now decides
  whether to switch on `preflag` (the §2.4.2.7 side-info bit that
  enables the Table B.6 `pretab[]` high-frequency-amplification
  shortcut). After the first inner-loop call each granule-channel
  evaluates the spec's only explicit hint for this decision —
  "preemphasis could be switched on if in all of the upper 4
  scalefactor bands the actual distortion exceeds the threshold after
  the first call of the inner loop" (§C.1.5.4.3.4). When that
  condition holds (every one of `xfsf[17..=20]` exceeds the uniform
  `xmin`), `sf.preflag` is set to `true` and the loop re-runs from
  the same iteration counter against the inflated effective per-band
  scalefactor `scalefac_l[sfb] + pretab[sfb]`. The pretab boost is
  free (one transmitted bit; no `part2_3_length` impact) and only
  amplifies the upper bands (`pretab = [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,
  2,2,3,3,3,2]`); the §C.1.5.4.3.6 cap (15 / 7) on the transmitted
  `scalefac_l[sfb]` is untouched because the cap math reads
  `sf.long[sfb]` only. The decision is one-shot per
  granule-channel — once `sf.preflag` flips on it stays on for the
  rest of the loop. Three implementation changes ship together: (a)
  `OuterLoopResult` gains a `preflag: bool` field mirroring
  `scalefactors.preflag` so the caller can read the decision without
  destructuring; (b) `band_distortion_long` adds `pretab[sfb]` into its
  colouring exponent when `sf.preflag` is set, so the §C.1.5.4.3.3
  per-band distortion is computed against the same reconstruction the
  decoder will compute (without this the loop's distortion metric
  would compare a preflag-boosted reconstruction against an
  un-boosted reference and never converge); (c) the stream encoder
  mirrors `sf.preflag` onto `gc.preflag` before the re-quantize loop,
  so the side-info bit reaches the bitstream and the decoder re-reads
  it via `sf.preflag = gc.preflag` in `read_mpeg1_granule_channel`.
  Validated by four new unit tests
  (`outer_loop_default_preflag_off_when_threshold_easily_met` /
  `outer_loop_default_preflag_off_when_only_low_bands_over_threshold` /
  `outer_loop_preflag_off_when_only_three_upper_bands_over` pin the
  negative arms — preflag must NOT fire when the §C.1.5.4.3.4
  upper-4 condition is unmet; `outer_loop_preflag_fires_when_all_upper_four_over_threshold`
  pins the positive arm under a controlled fixture covering
  `xr[196..418]`) and by a new end-to-end integration test
  `outer_loop_preflag_fires_on_hf_heavy_content` that confirms an
  HF-heavy three-tone input (10 kHz / 14 kHz / 17 kHz at 0.25 amp
  each) actually surfaces granule-channels with `preflag = 1` in the
  re-parsed side-info, end-to-end through the encoder.
- **§C.1.5.4.3 `scalefac_scale` escalation in the outer loop**
  (Phase 2 step 18): the §C.1.5.4.3 outer (distortion-control) loop
  now implements the spec's `scalefac_scale 0 → 1` switch ("If after
  some iterations the maximum length of the scalefactors would be
  exceeded … then scalefac-scale is increased to the value 1 thus
  increasing the possible dynamic range of the scalefactors. In this
  case the actual scalefactors and frequency lines have to be
  corrected accordingly"). When a §C.1.5.4.3.5 amplification step
  would push a band's `scalefac_l[sb]` past the §C.1.5.4.3.6 cap
  (15 for `sfb ∈ [0, 10]`, 7 for `sfb ∈ [11, 20]`) and the loop is
  still in `scalefac_scale = 0` mode, the loop now: (a) sets
  `scalefac_scale = 1` (multiplier 1.0 instead of 0.5 — twice the
  per-step boost, twice the dynamic range), (b) halves every
  in-progress per-band scalefactor with round-to-nearest integer
  arithmetic so the colouring factor `2^(mult·sf)` is preserved
  across the switch, (c) resets the `amplified[]` first-touch tracker
  so the post-escalation amps are tracked separately, and (d) resumes
  the loop. Each subsequent §C.1.5.4.3.5 amp step is then worth 2× as
  much energy boost. The escalation fires at most once per
  granule-channel (the spec defines only two `scalefac_scale` values),
  after which the next cap-would-exceed condition terminates the loop
  in the usual §C.1.5.4.3.6 "restore last-good state" form. The
  chosen `scalefac_scale` is now reported on
  `OuterLoopResult::scalefac_scale` and the `Mp3Encoder` stream-encode
  path propagates it into the granule-channel's side-info
  `scalefac_scale` bit so the decoder's
  `requantize::scalefac_multiplier` picks the matching 1.0 vs 0.5
  exponent. Validated by `outer_loop::tests::
  outer_loop_escalates_scalefac_scale_when_cap_would_terminate`
  (isolated sfb-19 fixture, threshold calibrated to baseline / 1e12 ⇒
  cap-would-exceed termination fires ⇒ `res.scalefac_scale == true`),
  by `outer_loop::tests::outer_loop_does_not_escalate_when_threshold_easily_met`
  (high threshold, no amps, `scalefac_scale` stays false), and by
  `tests/outer_loop_roundtrip.rs::outer_loop_tight_threshold_emits_valid_stream`
  (end-to-end: 1.0e-30 uniform threshold ⇒ encoder emits a parseable
  stream that self-decodes at finite PSNR). The pre-existing
  `outer_loop_strictly_higher_psnr_than_fixed_gain_on_multi_tone` PSNR
  regression test still passes with escalation in place — for the
  multi-tone fixture the existing cap is never tripped, so escalation
  is a no-op there. Spec gap: the §C.1.5.4.3 text does not specify
  the rounding convention for halving the existing scalefactors when
  scale switches; this implementation uses round-to-nearest integer
  ((x+1)/2). The choice is invariant for the spec's stated invariant
  ("scalefactors and frequency lines have to be corrected
  accordingly") because both round-down and round-half-up preserve
  the per-band coloured product to within one log step, and the next
  amp iteration immediately corrects any residual.

- **Joint-stereo MS encode** (Phase 2 step 17):
  `Mp3Encoder::new_joint_stereo_ms(bitrate_kbps, sample_rate_hz)` builds
  an opt-in two-channel encoder that applies the ISO/IEC 11172-3:1993
  §2.4.3.4.9.2 forward mid/side matrix `M = (L+R)/√2`, `S = (L-R)/√2`
  to each granule's post-MDCT (post-`inverse_alias_reduce`) `(L, R)`
  spectra before quantization. `M` is then coded into the channel-0
  slot, `S` into the channel-1 slot, and every emitted audio frame
  carries header `mode = '01'` (joint stereo) with
  `mode_extension = '10'` (ms_stereo on, intensity_stereo off) per
  §2.4.2.3. The decoder reverses the matrix via the existing
  `process_stereo` primitive driven by the same `mode_extension` bits.
  The matrix is its own inverse (a 2-D rotation by 45°), so in the
  absence of quantization error the round-trip is identity. The
  decoder pipeline order is `requantize → process_stereo →
  alias_reduce → imdct`, so the encoder applies the forward MS
  transform at the matching point (between `inverse_alias_reduce` and
  the quantize loop) — added as a pass-1/pass-2 split in
  `assemble_frame`. MS is applied to the **entire** spectrum
  (§2.4.3.4.9.2: "When MS-stereo is enabled but intensity stereo is
  not, the entire spectrum is decoded in MS-stereo"); intensity-stereo
  encode (§2.4.3.4.9.3) remains deferred. Both granules share the
  same Long block type (the only type this encoder emits this round),
  satisfying the §2.4.3.4.9 "both channels of a granule must share the
  same block type when MS is enabled" requirement automatically.
  `Mp3Encoder::ms_stereo_enabled()` reports the per-encoder flag.
  The trait wrapper gains
  `codec_encoder::make_encoder_joint_stereo_ms(params)` which builds
  the same `Mp3CoreEncoder` adapter around the joint-stereo
  constructor; `params.channels` must be 2 (mono cannot be joint
  stereo). Validated by `tests/joint_stereo_ms_roundtrip.rs`:
  end-to-end self-decode on a 1 s 440 Hz tone panned 70/30 toward L at
  192 kbit/s produces **per-channel PSNR L = 84.2 dB / R = 85.2 dB**,
  with a 90/10 strong-pan input recovering `|L|` strictly more than
  3× `|R|` through the round-trip — confirming the encoder's MS
  forward and the decoder's MS inverse compose to identity within
  quantization. Plus 2 unit tests on the trait factory
  (joint-stereo wire-bits + mono rejection).

- **Independent-stereo encode** (Phase 2 step 16):
  `Mp3Encoder::new` now accepts `ChannelMode::Stereo` and
  `ChannelMode::DualChannel` in addition to `ChannelMode::SingleChannel`.
  Both two-channel modes encode each channel independently — no
  joint-stereo coupling — and emit a standard MPEG-1 Layer III stereo
  frame: header `mode = '00'` (stereo) or `'10'` (dual-channel),
  `mode_extension = '00'`, 32-byte side-information block, and the
  §2.4.1.7 `main_data()` loop walks `for (gr=0..2) for (ch=0..nch)`
  emitting each granule-channel's part2 + part3 independently. Inputs
  are interleaved S16 (`[L0, R0, L1, R1, …]`); the upstream
  `Mp3Encoder::push_samples` deinterleaves the caller's stream into
  per-channel pending buffers and assembles a frame as soon as both
  channels carry a full 1152-sample granule pair. The
  `oxideav_core::Encoder` trait wrapper (`codec_encoder::make_encoder`)
  is widened in lockstep: `params.channels = 2` → `ChannelMode::Stereo`;
  the wrapper's `frame_to_i16` validator expects `samples × channels ×
  2` bytes per `AudioFrame`. `ChannelMode::JointStereo` remains rejected
  with `StreamEncodeError::StereoUnsupported` (MS / intensity coupling
  on the encode side is deferred to a later round; the decoder's
  `process_stereo` primitive already handles both methods for
  incoming joint-stereo bitstreams). Validated by
  `tests/stereo_encoder_roundtrip.rs`: a 1 s 440 Hz (L) / 880 Hz (R)
  sine encoded at 192 kbit/s round-trips through the in-tree decoder
  at per-channel PSNR **L = 85.13 dB** and **R = 80.42 dB**, with a
  single-bin DFT probe showing zero cross-channel leakage (each
  channel's tone energy is >1200× the other channel's). Plus two
  stereo unit tests on the trait wrapper (`stereo_flush_drains_to_
  complete_mp3_frames`, `stereo_send_frame_rejects_wrong_byte_count`).

- `Mp3Encoder::with_protection_bit(enabled)` — opt-in §2.4.3.1 CRC-16
  frame protection (Phase 2 step 15). When `enabled = true`, every
  emitted audio frame carries the 2-byte CRC check word in the slot
  immediately after the 4-byte header, with the wire `protection_bit`
  set to `0` per ISO/IEC 11172-3 §2.4.2.3. The CRC slot consumes two
  bytes of main-data slot capacity (the §2.4.2.3 frame_len is
  unchanged); the per-granule inner-loop bit budget shrinks by 16 bits.
  Frames that no longer fit surface `StreamEncodeError::Reservoir` at
  `finish` time — raise the bitrate or disable the CRC and retry. The
  CRC value covers exactly the Annex B Table B.5 Layer III set: header
  bits 16…31 plus the first 135 (mono) / 256 (other modes) side-info
  bits, MSB-first per the §2.4.3.1 / Figure A.9 shift-register
  procedure (`G(X) = X^16 + X^15 + X^2 + 1`, initial state `0xFFFF`).
  The carrier Xing / Info frame is left CRC-free regardless of the
  toggle. The crate's existing decode path already skips the CRC slot
  per `Mp3FrameHeader::crc_protected`, so CRC-enabled streams
  round-trip transparently.
- New `crc` module exposing the §2.4.3.1 CRC primitive:
  `crc::crc16_bits` (raw bit stream) and `crc::crc16_layer3` (the
  Annex B Table B.5 Layer III protected-set wrapper), with the
  `POLYNOMIAL_MASK = 0x8005` and `INITIAL_STATE = 0xFFFF` constants
  re-exported at the crate root as `crc16_bits`, `crc16_layer3`, and
  `CRC_INITIAL_STATE`.
- `tests/crc_roundtrip.rs` integration suite (3 tests): every emitted
  audio frame's wire CRC equals the independent `crc16_layer3`
  recomputation; CRC-on and CRC-off streams emit identical per-frame
  and total byte counts (the CRC takes from main-data capacity, not
  from frame length); a CRC-enabled stream decodes through
  `Mp3CoreDecoder` to silence-bounded PCM (no protection-bit
  regression in the decoder path).

- `Mp3Encoder::enable_vbr(min_kbps, max_kbps)` — opt-in true-VBR
  per-frame `bitrate_index` selection (Phase 2 step 14). With VBR
  active, the per-granule-channel inner-loop gain search runs the
  magnitude-clamp path alone (no bit-budget chase against a fixed
  budget — that path would saturate the constructor slot regardless
  of content), and `finish` picks for each audio frame the smallest
  §2.4.2.3 ladder index in `[min_kbps, max_kbps]` whose slot can hold
  the assembled main-data plus one optional padding byte. `enable_vbr`
  itself rejects off-ladder bitrates, reversed windows, and maxima
  above the constructor (`StreamEncodeError::InvalidVbrConfig`);
  frames overflowing the max-index slot at `finish` time surface
  `StreamEncodeError::VbrSlotTooSmall { frame_index, main_data_len,
  max_slot_bytes }`. Pairs naturally with `enable_xing_info` —
  see below.
- Xing `toc[100]` auto-fill: when `Mp3Encoder::enable_xing_info`'s
  template flags `xing_flag_bit::TOC` and `toc: None`, `finish`
  computes the 100-entry seek table from per-frame cumulative byte
  offsets: `toc[i] = floor(256 · audio_offset_for_percentile(i) /
  total_bytes)`, clamped to `255`. Each entry's offset uses the audio
  frame whose START is closest to playback fraction `i / 100`, the
  same convention `Mp3Demuxer::seek_to` reads on the way back.
  `Some(toc)` template values are written verbatim (no overwrite).
- `tests/vbr_roundtrip.rs` integration suite (13 tests): off-ladder /
  reversed / max-above-ctor / `[K, K]` configs, silence-stream landing
  on min-index, mixed-content yielding ≥2 distinct bitrates,
  `FrameWalker` consuming the varying-length stream cleanly,
  `Mp3Demuxer::next_packet` draining every audio frame without error,
  Xing TOC monotone non-decreasing (`toc[0] == 0`, tail ≥ 200), the
  auto-filled BYTES field matching the walker's audio-region byte
  sum, and the `Mp3Demuxer::xing()` view reporting the TOC flag
  exactly as written.
- `MPEG1_L3_BITRATE_LADDER_KBPS: [u32; 14]` re-export — the 14
  selectable ladder values (32 / 40 / 48 / 56 / 64 / 80 / 96 / 112 /
  128 / 160 / 192 / 224 / 256 / 320 kbps) callers can enumerate when
  picking `min_kbps` / `max_kbps`.
- `StreamEncodeError::InvalidVbrConfig` and
  `StreamEncodeError::VbrSlotTooSmall { frame_index, main_data_len,
  max_slot_bytes }` variants for VBR misconfiguration / overflow
  diagnosis.

- `xing_info` module — encoder-side inverse of
  `demuxer::parse_xing_info`. `XingTagSpec` specifies a Xing / Info
  VBR-information-frame payload (magic + flag word + up to four
  optional fields: `frames`, `bytes`, `toc[100]`, `quality`);
  `build_xing_info_payload` writes the byte run that goes immediately
  after the side-info bytes of an MPEG audio frame, in increasing
  flag-bit order (the exact order `parse_xing_info` consumes them on
  the reader side). `build_info_frame` bakes the payload into a
  complete on-wire CBR **carrier** frame — a silent Layer III frame
  (every `part2_3_length == 0`, `big_values == 0`) whose main-data
  slot starts with the Xing / Info magic + flagged fields. Layout
  verified against `docs/audio/mp3/fixtures/layer3-with-xing-vbri-tag/`
  + `layer3-with-id3v2-tag/` fixtures + `trace.txt` and the symmetric
  `parse_xing_info` reader (13 unit tests). Decoders that ignore the
  tag still see a structurally valid silent leading frame.
- `Mp3Encoder::enable_xing_info(template)` — opt-in toggle that
  prepends a Xing / Info carrier frame to `finish`'s output. The
  template carries the magic + flag word + any pre-known optional
  fields; `finish` fills in `frames` / `bytes` from the post-encode
  totals when those flag bits are set and the template field is
  `None`. The carrier itself is not counted in either total — both
  refer to the audio region that follows, matching the demuxer's
  first-frame-skip path. Pre-filled template fields are written
  verbatim (the encoder never overwrites a `Some(_)`).
- `tests/xing_info_roundtrip.rs` integration suite (7 tests): the
  carrier is the first frame, carries the right magic at the expected
  offset (4 bytes header + side-info bytes), `parse_xing_info`
  recovers the writer's intent field-for-field (both `Xing` and
  `Info` magic), the encoder's `bytes` accounting agrees with
  `FrameWalker` re-counting the audio region (including padded
  frames), pre-filled template fields are written verbatim,
  `Mp3Demuxer::open` reports the same Xing tag on the in-memory
  stream, and the carrier-without-template path produces an
  audio-only stream identical to the prior r141 behaviour (no Xing
  magic anywhere in the leading frame).
- Public re-exports in `lib.rs`: `build_info_frame`,
  `build_xing_info_payload`, `XingTagSpec`, `XingEmitError`,
  `xing_flag_bit` (the `flag_bit::{FRAMES, BYTES, TOC, QUALITY,
  ALL_FOUR}` module), `XING_MAX_PAYLOAD_BYTES`.

- `codec_decoder` module — the symmetric counterpart to the r140
  `codec_encoder` wiring: `Mp3CoreDecoder` implements
  `oxideav_core::Decoder` for mono MPEG-1 Layer III, parsing each
  inbound `Packet`'s MP3 frame and walking the existing per-granule
  `decode_huffman` → `requantize` → `alias_reduce` → `imdct_granule`
  → `synth_granule` chain, converting the float PCM run to interleaved
  S16 little-endian bytes for the returned `AudioFrame`. Per-stream
  state — the §2.4.2.7 main-data bit reservoir, the §2.4.3.4.10.4
  IMDCT overlap memory, and the §2.4.3.2 polyphase synthesis filter-
  bank shift register — is carried across packets. `reset()` wipes all
  three so the next `send_packet` decodes as if it were the first
  (the `Decoder` trait's documented contract for post-seek recovery).
- `codec_decoder::make_decoder` — direct-API factory matching the
  `oxideav-core` `DecoderFactory` signature. This is the dual-API
  convention preserved alongside the direct decode primitives
  (`decode_huffman` / `requantize` / etc.), which remain the historical
  entry point.
- `codec_decoder::register_codecs` — replacement codec-registry
  installer that registers BOTH the decoder factory AND the encoder
  factory on a single `CodecInfo` (so the registry's
  `implementations(codec_id)` lookup returns one entry that advertises
  both capabilities). `crate::register` now calls this variant; the
  prior `codec_encoder::register_codecs` (encoder-only) remains
  available as a public function for callers that want encoder-only
  registration.
- `tests/decoder_trait_roundtrip.rs` integration suite:
  - `registry_decoder_emits_audio_frames_with_monotonic_pts` — 250 ms
    of sine driven through the registered `oxideav_core::Decoder`
    trait API, confirms each emitted `AudioFrame` carries 1152 samples
    per channel and a monotonic PTS stamped from the inbound packet.
  - `registry_decoder_byte_exact_against_direct_chain` — the round-
    mandate's bit-exact check: 500 ms of sine encoded → sliced into
    per-MP3-frame packets → driven through the trait Decoder, asserts
    the resulting i16 PCM byte stream equals the direct-chain output
    (`decode_huffman` → `requantize` → `alias_reduce` →
    `imdct_granule` → `synth_granule`) on the same input bytes,
    sample-for-sample.
  - `registry_installs_both_encoder_and_decoder_factories` —
    `RuntimeContext` smoke test confirming `has_decoder` AND
    `has_encoder` on `"mp3"` after a single `register(&mut ctx)` call.
- `codec_encoder` module **Phase 2 step 12** — the runtime-context
  `oxideav_core::Encoder` trait wiring on top of the r138/r139
  `Mp3Encoder` stream encoder. `Mp3CoreEncoder` is a frame-to-packet
  adaptor: `send_frame(AudioFrame)` pushes mono `S16` PCM through the
  underlying `Mp3Encoder::push_samples`, and `flush()` runs
  `Mp3Encoder::finish` then slices the emitted CBR byte stream into one
  `Packet` per MP3 frame via the crate's own `FrameWalker`.
  `receive_packet()` returns `Error::NeedMore` before flush (the
  reservoir scheduler needs every frame's main-data up front) and
  drains the per-frame packet queue after. Per-packet PTS and duration
  are stamped in `1 / sample_rate` units.
- `codec_encoder::make_encoder` / `make_encoder_with_outer_loop` —
  direct-API factories matching the `oxideav-core` `EncoderFactory`
  signature. Both delegate into `Mp3CoreEncoder`; the outer-loop
  variant is the registry-wired equivalent of `Mp3Encoder::new_with_outer_loop`.
  This is the dual-API convention preserved alongside the historical
  direct `Mp3Encoder` entry point.
- `codec_encoder::register_codecs` — codec-registry installer. Claims
  WAVE format tag `0x0055` (MPEG Layer III) and Matroska codec id
  `A_MPEG/L3`, attaches the fixed-gain `make_encoder` factory, and is
  invoked from `crate::register` so a single `register(&mut ctx)` call
  now installs both the container demuxer and the codec encoder.
- `tests/encoder_trait_roundtrip.rs` integration suite:
  - `registry_encoder_emits_valid_mp3_frames` — drives 200 ms of sine
    through the registered `oxideav_core::Encoder` trait API, confirms
    every emitted packet starts with the 0xFFF MP3 sync, carries the
    per-frame duration `1152`, and stamps a monotonic PTS.
  - `registry_encoder_self_decode_psnr` — 1 s 440 Hz sine driven
    through the trait API only, packets re-concatenated, demuxed via
    `Mp3Demuxer`, self-decoded through `decode_huffman` → `requantize`
    → `alias_reduce` → `imdct_granule` → `synth_granule`. Achieves
    **86.17 dB PSNR**, matching the direct-API
    `stream_encoder_roundtrip` baseline.
- `outer_loop` module **Phase 2 step 11** — the §C.1.5.4.3
  distortion-control iteration loop that wraps the §C.1.5.4.4 inner-loop
  global-gain search of the r135 `inner_loop` module. Per ISO/IEC
  11172-3:1993 Annex C Figure C.9.b, `outer_loop_search_long` iterates:
  (a) runs the inner loop to pick the smallest `global_gain` whose
  Huffman bit count fits the per-granule-channel budget AND whose
  `max|is| ≤ 8191`; (b) computes the per-band §C.1.5.4.3.3 colored-domain
  distortion `xfsf(sb)` against the decoder's reconstruction;
  (c) amplifies every band with `xfsf(sb) > xmin(sb)` by
  `scalefac_l[sb] += 1`; (d) terminates on the §C.1.5.4.3.6 conditions
  (no band over threshold, every band already amplified, or the next
  amplification would exceed the per-band cap — 15 for `sfb ∈ [0,10]`,
  7 for `[11,20]` — restoring the last-good state).
- `Mp3Encoder::new_with_outer_loop` constructor — routes the Phase 2
  step 10 stream encoder through the outer loop with a uniform `xmin[sb]`
  threshold supplied by the caller (a per-band psychoacoustic threshold
  is deferred; this round uses a constant per the round mandate). Writes
  `scalefac_compress = 15` (slen1=4, slen2=3) so the chosen per-band
  scalefactors fit in part2. The fixed-gain `Mp3Encoder::new` path is
  preserved for reference / debug.
- `decode_scalefactors` is now spec-correct for non-zero
  `scalefac_compress`: it skips each granule-channel's part3 (Huffman)
  before reading the next gc's part2, per §2.4.1.7 `main_data()`'s
  `part2 part3` interleave. The earlier-round implementation only
  handled `scalefac_compress = 0` (where part2 collapses to zero bits and
  the interleave is degenerate); raising the encoder to the outer-loop
  path with `scalefac_compress = 15` exposed the latent bug.
- New `tests/outer_loop_roundtrip.rs` integration suite:
  - `outer_loop_sine_self_decode_psnr_above_floor` — single 440 Hz sine,
    confirms the outer-loop path still round-trips at PSNR > 20 dB
    (~86 dB observed, matching the fixed-gain baseline within ~0.02 dB).
  - `outer_loop_strictly_higher_psnr_than_fixed_gain_on_multi_tone` —
    six-tone fixture (110/440/880/1760/3520/7040 Hz). At 128 kbit/s mono,
    outer-loop PSNR is strictly greater than fixed-gain PSNR (typical
    delta +0.28 dB).
  - `outer_loop_silence_decodes_to_near_zero` — all-zero PCM produces
    a near-silent decode.
  - `fixed_gain_sine_decodes_through_new_decode_path` — confirms the
    test file's part2-aware decode harness works on the fixed-gain
    encode too.

- `stream_encoder` module **Phase 2 step 10** — the top-level
  **`Mp3Encoder`** that wires the Phase 2 primitives (analysis
  filterbank + forward MDCT overlap + inverse alias reduction +
  quantize + Huffman bit emission + main_data assembly + bit-reservoir
  scheduler) into a PCM-stream-in / MP3-frame-stream-out driver. The
  encoder consumes `i16` mono PCM samples via
  `Mp3Encoder::push_samples` (buffering until 1152 samples accumulate
  for one MPEG-1 frame), assembles each frame's main_data with a
  fixed-`scalefac_compress = 0` long-block configuration, picks a
  `global_gain` per granule via `inner_loop::search_bit_budget` +
  `search_magnitude_clamp` followed by a local `qquant + 1` ratchet
  that re-emits with a linbits-reach-filtered table chooser (the
  in-tree `choose_best_table_for_region` doesn't filter by linbits
  reach — it can pick e.g. table 16 (`linbits=1`, magnitude reach 16)
  for a range with `|is| = 100`, which would silently truncate
  magnitudes at emission), and flushes the buffered frames onto the
  §2.4.2.7 bit reservoir on `Mp3Encoder::finish` before writing the
  full header + side_info + main_data slot sequence to a
  `std::io::Write` sink.
- Scope this round: **mono / MPEG-1 only** (32 / 44.1 / 48 kHz), CBR,
  long blocks (`window_switching_flag = false`), zero scalefactors,
  no CRC, no Xing/Info VBR tag, no ID3 frontmatter. Stereo / LSF /
  VBR / outer noise-shaping loop / psychoacoustic model deferred to
  later rounds.
- New `tests/stream_encoder_roundtrip.rs` integration suite:
  - `sine_tone_one_second_self_decode_psnr` — synthesises a 1-second
    440 Hz mono sine at 44.1 kHz / 128 kbit/s, encodes it through
    `Mp3Encoder`, walks the resulting byte stream with the crate's
    own `Mp3Demuxer` + decoder primitives (`decode_scalefactors`,
    `decode_huffman`, `requantize`, `alias_reduce`, `imdct_granule`,
    `synth_granule`), and asserts **PSNR > 20 dB** (achieves ~86 dB
    in practice) against the input after accounting for the chain's
    1057-sample group delay (filterbank 481 + lapped MDCT 576).
  - `silence_one_frame_decodes_to_near_zero` — zero PCM in → zero
    PCM out within FP noise after warm-up.
  - `per_frame_huffman_is_buffer_roundtrips` — every emitted
    granule-channel parses through `decode_huffman` without error
    and at least one granule carries non-zero `is[]` for a sine input.
- Three new in-module unit tests:
  - `inverse_alias_roundtrip_long_block` —
    `alias_reduce(inverse_alias_reduce(xr)) ≈ xr` within FP noise.
  - `lossless_chain_finds_unit_gain_scale_factor` — the encoder
    analysis chain followed by the decoder synthesis chain
    (BYPASSING quantize) reconstructs PCM scaled by `n/4 = 9` (the
    Princen-Bradley TDAC factor); confirms the encoder's `/ 9`
    forward-MDCT normalisation choice.
  - `in_process_roundtrip_sine_psnr` — full chain WITH quantize but
    bypassing the byte-stream + demuxer round-trip; achieves
    ~87 dB PSNR at delay 1057 samples.
- `main_data` module **Phase 2 step 9** — the **§2.4.2.7 cross-frame
  BIT-RESERVOIR SCHEDULER**, `main_data_begin > 0` path. The step-8
  assembler produces each frame's main-data bytes self-contained
  (`main_data_begin = 0`); a busy frame whose main_data exceeds its
  per-frame slot byte budget (`frame_len - 4 - CRC? - side_info_bytes`)
  was unschedulable. The new `schedule_reservoir` /
  `ReservoirScheduler` route the busy frame's main_data backward into
  earlier quiet frames' unused tail bytes — the §2.4.2.7 "bit
  reservoir" — and stamp the back-pointer into the busy frame's
  `SideInfo::main_data_begin`. The scheduler enforces both §2.4.2.7
  invariants: `R_i ≥ 0` (cumulative slot ≥ cumulative main-data at
  every prefix) and `R_i ≤ 511` (MPEG-1 9-bit cap) / `≤ 255` (LSF
  8-bit cap), surfacing failures as `ReservoirError::SlotUnderflow` /
  `ReservoirError::ReservoirOverflow`. Two-pass design (compute every
  frame's `main_data_begin` up front, then carve the rolling main-data
  concatenation into per-frame slots, zero-padding the final tail).
  `RESERVOIR_MAX_MPEG1` / `RESERVOIR_MAX_LSF` constants surface the
  on-wire bit-width caps.

  Eight new round-trip unit tests in `src/main_data_tests.rs`:
  - `schedule_single_quiet_frame_self_contained` — one-frame schedule
    is the `main_data_begin = 0` path with zero-padded slot tail
    (regression of step-8 behaviour through the new API).
  - **`schedule_three_frame_busy_middle_via_prior_quiet`** — the
    user-prompt scenario: middle frame's main_data (50 B) exceeds its
    own slot (30 B) and is scheduled into the prior quiet frame's
    unused tail. Expected `main_data_begin` sequence `[0, 20, 0]` and
    decoder-side bit-exact recovery of all three frames' main_data via
    `scalefactors::Reservoir::assemble`.
  - **`three_frame_pipeline_round_trip_with_real_main_data`** — full
    pipeline cross-check at MPEG-1 128 kbps / 44.1 kHz mono: three
    real `assemble_main_data` outputs (frame 0 / 2 with
    `big_values = 1`, frame 1 with `big_values = 200` per granule,
    both granules per frame) scheduled into 200-byte slots, decoded
    back through `Reservoir::assemble` + the existing §2.4.1.7
    main_data() loop, recovering every granule's scalefactors + `is[]`
    bit-exactly.
  - `reservoir_scheduler_builder_matches_one_shot` — `push` /
    `finish` stateful builder is equivalent to one-shot
    `schedule_reservoir`.
  - `schedule_busy_frame_with_no_prior_reservoir_underflows` /
    `schedule_reservoir_cap_mpeg1_511` /
    `schedule_reservoir_cap_lsf_255` — error surfaces for the
    schedulability and reservoir-cap invariants.
  - `reservoir_cap_constants_match_spec_bit_widths` — the 511 / 255
    constants are the literal §2.4.2.7 9-bit / 8-bit field widths.

  Single-channel MPEG-1 only this round; LSF cap enforcement aside,
  every test runs MPEG-1. The outer-loop / multi-channel /
  encoder-side `silent_side_info`-driven multi-frame stream wiring is
  deferred to a later round, and the scheduler itself is layout-only —
  it does not run the analysis / quantization stack.

- `main_data` module **Phase 2 step 8** — the **§2.4.1.7 `main_data()`
  ASSEMBLER** plus the no-reservoir `main_data_begin = 0` path.
  `assemble_main_data` composes a complete per-frame main-data block by
  walking the spec's `for (gr) for (ch)` loop, emitting each
  granule/channel's **part2** scalefactors immediately followed by its
  **part3** Huffman codewords into one shared `MainDataWriter`, with no
  byte alignment between fields (the contiguous on-wire layout):
  - **part2** — a new scalefactor writer (`MainDataWriter` +
    `write_mpeg1_granule_channel` / `write_lsf_channel`) inverts the
    `scalefactors` decode path: MPEG-1 long (four scfsi band groups,
    granule-1 reuse skips), short (per `(sfb, window)`), and mixed blocks,
    and the LSF four-partition `slen`/`nr_of_sfb` scheme from
    `scalefac_compress`.
  - **part3** — `huffman::emit_huffman` emits the `huffmancodebits()`
    payload into the same shared writer at its current bit position (the
    region split + `table_select` derived from the side info exactly as
    `decode_huffman` reads them, so the per-line region/table assignment
    cannot desync).
  - Records each granule/channel's `part2_3_length` (= part2 + part3
    bits) back into the supplied `SideInfo`, sets `main_data_begin = 0`,
    and returns the byte-padded block plus the `total_bits` sum.
  - `MainDataWriter` (the MSB-first inverse of `MainDataReader`) and
    `emit_huffman` are now public; `encode_huffman` is reimplemented on
    top of `emit_huffman` (unchanged API/behaviour).

  5 new round-trip unit tests in `src/main_data_tests.rs` assemble known
  scalefactors + quantized `is[]` then read the block back through the
  exact §2.4.1.7 loop (part2 then part3 per granule/channel), recovering
  the scalefactors and `is[]` bit-exactly with the reader consuming
  exactly `total_bits`: MPEG-1 long two-channel, MPEG-1 two-granule mono
  (scfsi all-false), MPEG-1 short block, LSF long, and a cross-check that
  the first granule's part2 matches `decode_scalefactors`. No
  psychoacoustic model, no outer/distortion loop, no real reservoir
  scheduling this round — just the assembler + `main_data_begin = 0`.

- `huffman` module **Phase 2 step 7** — the **§2.4.1.7
  `huffmancodebits()` bit EMISSION**, the forward encoder counterpart to
  `decode_huffman`. Given a quantized `is[576]`, the big-values region
  split (`region_ends`), and the per-region `table_select` (from the
  step-6 `choose_best_table_for_region` / `choose_best_count1_table`),
  `encode_huffman` writes the actual Layer III main-data Huffman payload:
  - **big_values** three regions — the Table 3-B.7 codeword for the
    magnitude-clamped `(min(15,|x|), min(15,|y|))` cell, a `linbits` ESC
    field carrying `|v| - 15` for each component of magnitude `≥ 15`, and
    a sign bit per non-zero component, in the §2.4.1.7 order
    codeword → linbits_x → sign_x → linbits_y → sign_y — the exact
    inverse of `decode_big_pair`.
  - **count1** quadruples — quad table A (`QUAD_A`) or the trivial 4-bit
    table B code, plus a sign bit per non-zero value, inverting
    `decode_count1_quad`.
  - Output is byte-aligned (trailing partial byte zero-padded) and reads
    back through `MainDataReader` bit-for-bit. `Mp3HuffmanData` carries
    the packed `bytes` plus the exact `bit_len` (excluding the pad),
    which equals the step-6 `count_huffman_bits` for the same inputs.
  - `HuffmanEncodeError` reports a non-codable pair
    (`PairNotCodable(table)`), an unused/out-of-range codebook
    (`UnusedTable`), or `big_pairs*2 > 576` (`BigValuesTooLarge`).

  9 new unit tests in `src/huffman_tests.rs` verify the round-trip
  `encode_huffman` → `decode_huffman` recovers the original `is[]`
  bit-exactly with `bit_len == count_huffman_bits` and the decoder
  consuming exactly `bit_len` bits: a mixed big-values + count1 granule,
  a `linbits` ESC pair (table 16) and a larger negative escape
  (table 24), the count1 table-B path, a three-region split aligned to
  the 44.1 kHz long-block bands, and an end-to-end pipeline that derives
  the partition split + per-region tables via the step-6 choosers before
  emitting. Error paths cover the uncodable pair, unused table, and
  oversized big_values. No bit reservoir, side-info packing, or
  full-frame assembly this round — just the codeword emitter.

- `inner_loop` module **Phase 2 step 5** — the **§C.1.5.4.4 inner-loop
  `global_gain` search** wrapping the §2.4.3.4.7 `quantize` primitive.
  Holding a chosen scalefactor configuration fixed, it binary-searches
  the 8-bit `global_gain` field for the **smallest** gain (finest
  quantization) whose quantized `is[576]` satisfies a constraint. The
  search is valid because `|is_i|` is monotone non-increasing in
  `global_gain` (a larger gain divides by a larger `2^((gg-210)/4)`),
  making the "constraint satisfied" predicate a step function over
  `[GAIN_MIN, GAIN_MAX] = [0, 255]`.
  - `search_magnitude_clamp` — the §C.1.5.4.4.2 maximum-value test:
    smallest gain with `max|is| ≤ 8191` (`BIG_VALUES_LIMIT`, from the
    §2.4.1.7 big-values definition). For a fixed `sf` the coarsest gain
    `GAIN_MAX` divides by only `2^((255-210)/4)`, so a target louder
    than `2^11.25 · 8191^(4/3) ≈ 4.0e8` (at `sf = 0`) cannot be clamped
    by gain alone — the result then reports `satisfied == false` with
    the `GAIN_MAX` fallback (the outer loop / scalefactors, not in
    scope, would extend the range).
  - `search_bit_budget` — smallest gain whose `coarse_bit_estimate`
    (a `bits(|is_i|) + 1` placeholder, **not** the exact §C.1.5.4.4.5 /
    §C.1.5.4.4.8 Huffman count) fits a supplied budget.
  - `InnerLoopResult` carries `global_gain`, `is[576]`, `max_abs`, and
    a `satisfied` flag; `max_abs` and `coarse_bit_estimate` are public
    helpers.

  14 new unit tests in `src/inner_loop_tests.rs` verify: `max|is|` and
  the coarse bit count are monotone across all 256 gains; the chosen
  gain is minimal (gain−1 violates) and keeps `max|is| ≤ 8191`; louder
  targets pick coarser-or-equal gains across a 6-decade amplitude sweep;
  tighter budgets pick coarser gains; `requantize(is)` at the chosen
  gain reproduces the target within the quantizer-grid bound; the
  clamp-reach boundary is honoured on both sides; and zero / silence
  edge cases pick `GAIN_MIN` / all-zero `is`.

  Scope of this step is the `global_gain` scalar only: no psychoacoustic
  model, no §C.1.5.4.3 outer (distortion-control) loop, no scalefactor
  estimation, no exact Huffman bit count.

- `quantize` module **Phase 2 step 4** — the **§2.4.3.4.7 quantization
  primitive**, the algebraic inverse of `requantize::requantize`. Given
  a target float spectrum `xr[576]` and an already-chosen
  `GranuleChannel` + `ScaleFactors` configuration,
  [`quantize::quantize`] computes the integer Huffman-input buffer
  `is[576]` such that feeding `is` back through `requantize` with the
  same configuration reproduces `xr` within `f32` round-to-nearest
  precision. Implements both block forms:
  - Long: `|is_i| = round((|xr_i| / G_long(sfb))^(3/4))` with
    `G_long(sfb) = 2^((gg-210)/4) * 2^(-mult*(sf_l[sfb] +
    preflag*pretab[sfb]))`.
  - Short / per-window: `G_short(sfb, w) = 2^((gg-210-8*subblock_gain[w])/4)
    * 2^(-mult*sf_s[sfb][w])`.
  - Mixed-block split (lines 0..36 long, 36..short-active-end short
    starting at short sfb 3) mirrors the decoder.
  - Sign of `xr` is reapplied after the magnitude round; zero target
    yields zero quantizer output regardless of gains.

  Round-trip `is -> xr_ref -> is' -> xr_back` is **bit-exact** on every
  tested configuration: long-block at every `global_gain ∈ {180, 200,
  210, 224, 240}`, long with scalefactors 0..4 and both
  `scalefac_scale` settings, long with `preflag`, short with non-zero
  per-window `subblock_gain`, short with per-window `scalefac_s`,
  mixed-block with non-trivial long/short scalefactors, and LSF at
  24 kHz. Bin-level RMS between `xr_back` and `xr_ref` is `0.0e0` —
  the integer-power-law grid is closed under the round-trip the moment
  `xr` already lies on it (and a `xr` produced by `requantize` always
  does). 14 new unit tests in `src/quantize_tests.rs`.

  Scope of this step is the primitive only: no `global_gain` search,
  no bit allocation, no scalefactor estimation, no §C.1.5.4
  (informational) noise-shaping iteration loop. Those become
  subsequent primitives built on top of this one.

- `analysis` module **Phase 2 step 3** — the **polyphase analysis
  subband filterbank** (ISO/IEC 11172-3:1993 Annex C §C.1.3 / Figure
  C.4), the algebraic dual of the §2.4.3.2 / Figure A.2 synthesis
  filterbank in `synth`. This is the first encoder stage on the
  forward signal path: it splits a broadband PCM input into 32
  critically-sampled subbands at sample rate `f_s / 32`.
  - `m_coefficient(i, k) = cos((2i+1)(k-16)π/64)` — the §C.1.3
    matrixing kernel; spot-check `M[i, 16] = 1` for every `i` (the
    `(k-16)` zero column) and a full bin-for-bin recompute test on
    the 32×64 cell grid.
  - `AnalysisState` — the 512-element polyphase shift register
    `X[0..512]`, zero-initialised at stream start (the analysis
    mirror of `synth::SynthState`'s `V[]`).
  - `analyze_row(&pcm, &mut state)` — one pass of Figure C.4: input
    shift (`X[i] = X[i-32]` for `i = 511..32` then `X[31-j] = pcm[j]`
    for `j = 0..32`), 512-tap windowing by `C[i]`, 8-fold partial
    calculation `Y[i] = Σ_{j=0..7} C[i + 64j] · X[i + 64j]`, and
    64×32 matrix multiply by `M[i, k]`. Consumes 32 PCM samples,
    produces 32 subband samples.
  - `analyze_granule(&pcm576, &mut state)` — 18 sequential
    `analyze_row` invocations producing the 32×18 subband-time block
    that feeds the §2.4.3.4.10.2 forward MDCT, the exact analysis-
    side mirror of `synth::synth_granule`.
  - `C_TABLE: [f64; 512]` — Annex C Table C.1 prototype window;
    first 256 entries transcribed verbatim from the staged ISO/IEC
    11172-3:1993 PDF (Annex C, pages 67–69) with OCR fixes
    cross-checked against the local monotone trend of each
    neighbour; second 256 entries derived by the cosine-modulated-
    prototype symmetry `C[512-i] = +C[i]` if `i ≡ 0 (mod 64)` and
    `-C[i]` otherwise (verified on every spot-check pair in the
    first half).
  - 16 new tests: `C[]` length / boundary / global-max checks, the
    full polyphase symmetry sweep over `i = 1..256`, three
    matrixing-coefficient tests (k=16 identity column, i=0 closed
    form, full 32×64 grid recompute), four shift-register tests
    (zero state, single-block injection at indices 31..0, two-block
    history check at 32..63, no-spill beyond), an `analyze_row`
    linearity check, a per-subband DC-tone subband-domain round-trip
    test (`synth_row → analyze_row` for `S[sb0] = 1` per-subband,
    settled-row ripple < 1×10⁻¹² for every one of the 32 subbands —
    the spec-derivable cyclostationary invariant), a PCM-domain
    round-trip test (`analyze_row → synth_row` for a broadband
    multi-tone signal, RMS deviation < 1×10⁻⁴ at the 481-sample
    prototype group delay — measured ≈ 3×10⁻⁵), and two
    `analyze_granule` shape tests (zero-in-zero-out, first-row
    equality with `analyze_row`).
- `mdct` module **Phase 2 step 2** — analysis windowing
  (encoder mirror of §2.4.3.4.10.3) and the forward overlap split
  (encoder mirror of §2.4.3.4.10.4):
  - `analysis_long_window(i)` / `analysis_short_window(i)` —
    `sin((π/36)(i+½))` and `sin((π/12)(i+½))` primitives, identical
    to the synthesis-side windows (Princen-Bradley TDAC requires the
    same window on both halves).
  - `window_long_family_analysis(&xn, block_type)` — apply the four
    long-family window shapes to 36 input samples: plain sine
    (`block_type 0`); start (long-half 0..17, pass-through 18..23,
    short-half 24..29, zero 30..35) for `block_type 1`; stop (zero
    0..5, short-half 6..11, pass-through 12..17, long-half 18..35)
    for `block_type 3`. Partitioning matches the §2.4.3.4.10.3
    synthesis table exactly.
  - `window_short_analysis(&xn)` — extract the three 12-sample short
    sub-blocks from the 36-sample input via the analysis-side
    inverse of the §2.4.3.4.10.3 d concatenation
    (`xj_in[j][k] = xn[6 + 6·j + k]`), each pre-multiplied by the
    short analysis window.
  - `MdctState` / `forward_overlap(&current, &mut state)` — the
    analysis mirror of `imdct::ImdctState` / §2.4.3.4.10.4
    overlap-add: stream-start state is all zeros; each call assembles
    a 36-sample forward-MDCT input frame `[prev_18, current_18]` and
    rolls `prev_18 := current_18`.
  - 12 new tests: long-window byte-for-byte vs `sin((π/36)(i+½))`
    plus symmetry and `Σw² = 18` cross-check; short-window analog
    with `Σw² = 6`; long / start / stop windowing per spec
    partitioning, including the start↔stop complementary zero-region
    check; short-block 3-sub-extraction with the half-overlap source
    spans (i=6..17, 12..23, 18..29) matched against the synthesis
    `y₀+y₁`/`y₁+y₂` overlap regions; forward-overlap state default-
    zero, first-granule head-zero/tail-current, second-granule
    head-prev/tail-current, and zero-input zero-output sanity; the
    standalone time-space round-trip identity
    `imdct(mdct(δ_0))[i] = (n/4)·(δ_0[i] ∓ reflection)` (the
    aliased-with-self structure that TDAC cancels via adjacent-frame
    overlap-add). The headline test is **end-to-end long-block
    Princen-Bradley TDAC recovery** — feed three granules through
    `forward_overlap` → `window_long_family_analysis(Long)` → `mdct`
    → `imdct` → long-window → overlap-add, and the middle granule is
    recovered scaled by `n/4 = 9` exactly (the time-space factor
    `n/4` × the sine-window TDAC sum `w(i)² + w(i+n/2)² = 1`), on
    arbitrary mixed-frequency input. This is the strongest single
    test on the new analysis chain: it requires every analysis
    primitive (window, overlap split, MDCT) and every synthesis
    primitive (IMDCT, window, overlap-add) to line up exactly.
- New `mdct` module — Layer III encoder **Phase 2 step 1**: the
  §2.4.3.4.10.2 **forward MDCT** primitive (`mdct(xn, n)`), the
  analysis companion of the synthesis-side `imdct::imdct`. Implements
  the 36-point (long block) and 12-point (short sub-block) transforms
  using the same `cos((π / (2·n)) · (2·i + 1 + n/2) · (2·k + 1))`
  kernel as the IMDCT but summed over the `n` time samples for each of
  the `n / 2` output bins. With the spec normalisation the analysis
  transform is the left inverse of the synthesis transform on the
  bin space: `mdct(imdct(X), n)[k] = (n/2) · X[k]`. Unit tests cover
  per-bin impulse closed forms, arbitrary spec-sum re-evaluation,
  linearity, output-length contract, and exact frequency-domain
  round-trip against the shipped IMDCT for both n = 36 and n = 12. No
  analysis windowing, no forward overlap split, no psychoacoustic
  model, no Huffman encode — those are subsequent Phase 2 rounds.
- New `encoder` module — Layer III encoder **Phase 1**
  (bitstream-formatting half, no psychoacoustic model):
  - `write_header` writes the four-byte frame header
    (ISO/IEC 11172-3 §2.4.1.3 / §2.4.2.3), the exact byte-for-byte
    inverse of `frame::parse_header`.
  - `write_side_info` writes the Layer III side-information block
    (ISO/IEC 11172-3 §2.4.1.7 MPEG-1, ISO/IEC 13818-3 §2.4.1.7
    MPEG-2 / MPEG-2.5 LSF), the exact inverse of
    `side_info::parse_side_info` for both layouts — `main_data_begin`,
    `private_bits`, `scfsi`, and the full per-granule-per-channel
    record across both window branches.
  - `encode_silent_frame` emits a complete, self-delimiting
    all-zero-quantization Layer III frame (`part2_3_length == 0`,
    `big_values == 0`, no CRC, zero-filled main data) sized to
    `Mp3FrameHeader::frame_len`; reconstructs to silence.
    `make_silent_header` resolves a bitrate / sample-rate /
    channel-mode triple to the raw header indices; `silent_side_info`
    builds the matching empty side info.
  - Validated: the emitted frame round-trips through this crate's own
    `parse_header` / `parse_side_info` / `FrameWalker` / `Mp3Demuxer`,
    and a black-box `ffmpeg` decode of a 50-frame stream yields PCM
    that is bit-exact silence (max |sample| = 0). The encoder still
    lacks the forward analysis path (MDCT / psychoacoustics / bit
    allocation / Huffman encode) — a later round.
- New `demuxer` module wrapping the framing layer in an
  `oxideav_core::Demuxer`:
  - `Mp3Demuxer::open` reads the input head and tail to skip the
    optional ID3v2 tag (10-byte header + synchsafe-sized body +
    optional v2.4 footer per `docs/container/id3/id3v2.3.0.html` +
    `id3v2.4.0-structure.html`) and the optional ID3v1 trailer
    (last 128 bytes whose first three are `TAG`, layout per
    `docs/audio/mp3/datavoyage-mpgscript-mpeghdr.html` §"MPEG
    Audio Tag ID3v1").
  - `parse_xing_info` detects a Xing / Info VBR-info frame at
    `4 + crc_bytes + side_info_bytes` past the first audio
    frame's syncword and decodes the four prompt-enumerated
    fields (`frames`, `bytes`, 100-byte `toc`, `quality`) gated
    by the low four bits of a big-endian 32-bit flag word.
    Layout verified byte-for-byte against
    `docs/audio/mp3/fixtures/layer3-with-xing-vbri-tag/input.mp3`
    and `.../layer3-with-id3v2-tag/input.mp3` + their
    `trace.txt`. Pending a canonical Xing/Info doc under
    `docs/audio/mp3/`.
  - `next_packet` emits one MPEG audio frame per call with
    monotonic per-packet PTS in the stream's `1/sample_rate`
    time base, `keyframe = true`, and `duration =
    samples_per_frame`. Resyncs on bad / overrun /
    sample-rate-changing headers per the same one-byte-step
    pattern as the existing `FrameWalker`. Returns
    `Error::Eof` past the audio region (`total_len - 128` when
    an ID3v1 trailer was detected).
  - `seek_to(pts)` uses the Xing TOC's 100 percentile entries
    (VBR) or proportional byte-offset arithmetic
    (`pts × bitrate / 8 / sample_rate`, CBR) and snaps to the
    next valid frame sync via an 8 KB look-ahead.
  - `duration_micros` returns the duration computed from
    `frames × samples_per_frame / sample_rate` (VBR with Xing
    `frames`) or `audio_bytes × 8 / bitrate × sample_rate`
    (CBR). Δ vs `ffprobe -show_entries format=duration` for
    the four fixtures (CBR-320 / VBR-q5 / ID3v2-tagged /
    Xing-tagged) is ~+4.5% (35.9 ms over an 800 ms file) —
    the LAME encoder-delay/padding consumes 1152 PCM samples
    at the head + tail and lives in the bytes after the four
    Xing fields parsed at this layer.
  - `probe` content-scores `ID3v2 + Layer III sync` and bare
    Layer III sync candidates (100 / 75 + extension-tied 100).
  - `register()` is now non-trivial: it installs the demuxer
    under format name `"mp3"` with the `.mp3` / `.mp2` / `.mp1`
    extensions and the content probe.
  - 13 unit tests under `demuxer::tests` cover the synchsafe
    decoder, the ID3v2 total-length computation, a synthetic
    CBR walk (PTS monotonicity + frame count + EOF idempotency),
    an ID3v2 + ID3v1-wrapped CBR walk asserting the demuxer's
    `first_audio_frame_offset` matches the trace's
    `total=85`, four Xing/Info parse permutations (all-flags +
    Info-magic + mono side-info offset + no-magic), a
    `side_info_len` matrix across MPEG-1/2 × mono/stereo, the
    probe-scoring matrix, the CBR `duration_micros` formula,
    and a CBR seek round-trip (`seek_to(5_760) → packet.pts
    matches actual`).
  - 5 docs/audio/mp3/fixtures/ integration tests
    (`tests/docs_corpus.rs`) walk the on-disk corpus — CBR-320,
    VBR-q5, ID3v2-tagged, Xing-tagged, and a broad pass over
    the 15 Layer III fixtures — and assert the demuxed frame
    count, ID3v2 size, Xing-tag fields, and first-frame byte
    offset all match the trace file byte-for-byte. MPEG-2.5
    fixtures are excluded pending the frame-parser extension
    tracked in `docs/audio/mp3/MPEG-2.5-GAP.md`; Layer II
    fixtures are excluded per the round-121 brief.
  - 1 `tests/duration_comparison.rs` integration test prints
    the demuxer's `duration_micros` next to ffprobe's reported
    duration for the four fixtures (CBR-320 / VBR-q5 /
    ID3v2-tagged / Xing-tagged) so the duration delta is
    visible in CI output.

- Clean-room Layer III **polyphase synthesis subband filterbank** — the
  last decode stage — in the new `synth` module, built solely from
  ISO/IEC 11172-3:1993 §2.4.3.2 / Figure A.2 ("Synthesis subband filter
  flow chart" on p.39), the §2.4.3.2.2 coefficient formula
  `N[i,k] = cos((16+i)·(2k+1)·π/64)`, and Annex B Table B.3
  (the 512 `D[]` window coefficients, pages 50–52 of the body):
  - `synth_row(s, &mut state)` runs one Figure A.2 pass — Shifting
    (`V[i] = V[i-64]` for i = 1023..64), Matrixing (`V[i] = Σ_k N[i,k]·S[k]`
    for i = 0..64), Build U (`U[64i+j] = V[128i+j]`, `U[64i+32+j] =
    V[128i+96+j]`), Window (`W[i] = U[i]·D[i]`), and the 16-tap
    summation `S_out[j] = Σ_{i=0..16} W[j+32i]` — to turn 32 input
    subband samples into 32 PCM samples.
  - `synth_granule(subband_time, &mut state)` chains 18 `synth_row`
    calls over the 32×18 IMDCT output of one granule-channel, yielding
    576 PCM samples in playback order.
  - `SynthState` carries the per-channel 1024-value shift register
    `V[]`; Figure A.2 footnote 1 ("V to be initialised with zeroes
    during startup") makes `SynthState::default()` the correct
    stream-start state.
  - `D_TABLE` is the 512-value Table B.3 window: every coefficient was
    hand-transcribed from the staged ISO/IEC 11172-3:1993 PDF (Annex B
    pages 50–52 of the body; rendered PNGs at
    `docs/audio/mp3/annex-b-renders/Table-B.3-coefficients-Di-p5{6,7,8}.png`),
    with OCR cross-checks for every text-extraction-suspect character.
  - `n_coefficient(i, k)` exposes the §2.4.3.2.2 matrix coefficient
    formula directly for use in tests and external derivations.
  - 19 synth unit tests: `D[]` length is 512, the four boundary values
    `D[0] / D[1] / D[255] / D[256] / D[257] / D[511]` match Table B.3
    byte-for-byte, `D[256] = +1.144989014` is the unique global maximum
    and `D[255] = -1.144287109` the unique global minimum, the
    `|D[256±k]|` mirror pairs match the printed values, `N[i,k]` matches
    `cos(π/4)`, `cos(π/2)` and `cos(π)` at the four corner / midpoint
    cases, a hand-computed known vector derivation (`S[k0]=1`, all-zero
    V → `S_out[j] = N[j,k0]·D[j]`) for both k0=0 and k0=5 covering
    steps 2–5 of Figure A.2 byte-exactly, linearity of the whole
    filter, the shift register propagating an impulse from V[0..64]
    into V[64..128] on the next iteration, `synth_granule` agreeing
    with a manual `synth_row` on time-row 0, an end-to-end zero through
    `imdct_granule → synth_granule` yielding 576 PCM zeros, and an
    end-to-end synthetic frame with `xr[0] = 1.0` (DC in subband 0)
    through `imdct_granule → synth_granule` producing 576 finite,
    partially non-zero PCM samples.
  - This completes the granule-level decode chain end-to-end: a
    decoded `[i32; 576]` from `decode_huffman` → `requantize` →
    `reorder` → `process_stereo` → `alias_reduce` → `imdct_granule` →
    `synth_granule` now yields `[f32; 576]` PCM. The remaining
    decoder work is the frame-driver / `Decoder` trait wiring that
    iterates `FrameWalker` frames and feeds each through the chain.
- Clean-room Layer III **IMDCT, windowing, overlap-add and frequency
  inversion** stages in the new `imdct` module, built solely from
  ISO/IEC 11172-3:1993 §2.4.3.4.10.2 (IMDCT formula), §2.4.3.4.10.3
  (the four block-type window shapes and the short-block concatenation
  table), §2.4.3.4.10.4 (overlap-add with the previous granule's saved
  second half), and §2.4.3.4.10.5 (polyphase-filterbank frequency
  inversion):
  - `imdct(xk, n)` evaluates
    `x[i] = Σ_{k=0..n/2-1} X[k]·cos((π/(2n))·(2i+1+n/2)·(2k+1))` for
    `i = 0..n-1`, with `n = 36` for long blocks and `n = 12` for short.
  - `ImdctState` carries the per-subband saved second-half overlap
    across granules (stream-start state is all zeros).
  - `imdct_granule(xr, gc, &mut state)` runs the full §2.4.3.4.10
    pipeline for one granule-channel: per subband — single 36-point
    IMDCT + per-block-type window (long block-types 0/1/3, with a mixed
    block's two lowest subbands using the long path) or three 12-point
    IMDCTs + short window + concatenation (short block, and the upper
    30 subbands of a mixed block); overlap-add the first 18 samples
    with `s_prev`; save the second 18 into the state; negate every odd
    time sample of every odd subband.
  - 22 new `imdct` tests: closed-form impulse response for n=12
    (`x[i] = cos((π/24)(2i+7))`), hand-computed all-ones reference
    values (n=12 and n=36), linearity, byte-exact long and short window
    tables (with symmetry and Σw²=18 cross-checks), the normal / start /
    stop window shapes per spec, the short-block concatenation table
    (including the `y₀+y₁` and `y₁+y₂` overlap), zero-input zero-output,
    overlap state initial zeroness, first-granule equals `z[0..17]` when
    `s_prev = 0`, second-granule output adds the saved overlap from
    granule 1, per-subband overlap isolation, frequency inversion on
    odd subbands' odd time samples with even subbands unaffected,
    short-block three-sub-IMDCT dispatch, mixed-block long-window
    dispatch in subbands 0 and 1 (with the §2.4.3.4.10.5 sign-flip on
    subband 1), start-block tail-zero through `imdct_granule`, and
    stop-block head-zero through `imdct_granule`.
- Completed the Layer III **Huffman big-values codebooks**: the large
  16×16 tables 15, 16 and 24, plus the linbits aliases 17..=23 (which
  reuse table 16's `(x, y)` codes) and 25..=31 (table 24's codes). With
  these the Huffman stage now covers **all** of ISO/IEC 11172-3:1993
  Annex B Table 3-B.7 (codebooks 0..=31, excluding the "not used" 4 and
  14). Every code/length value was hand-transcribed from the Table 3-B.7
  pages of the ISO/IEC 11172-3:1993 PDF; the alias linbits widths come
  from the "same as table N, but linbits=L" notes on the same pages.
  - `decode_huffman` resolves `table_select` 15..=31 to the new tables;
    `HuffmanError::TableNotYetTranscribed` is retained for API stability
    but is no longer produced.
  - 8 new Huffman tests: zero / signed pairs from table 15, table 16's
    1-bit linbits ESC on magnitude-15 (and the small no-ESC path), the
    table 17 alias at linbits=2, table 24's 4-bit ESC, and the table 25
    alias at linbits=5. A `large_tables_prefix_free_and_complete` test
    proves all three 16×16 codebooks are prefix-free with a Kraft sum of
    exactly 1 (256 symbols each) — a strong transcription cross-check.
- Clean-room Layer III **alias reduction** stage in the new `alias`
  module, built solely from ISO/IEC 11172-3:1993 §2.4.3.4.10.1 (the
  butterfly pseudo code) and Annex B Table 3-B.9 (the coefficients):
  - `alias_reduce` applies the eight per-boundary butterflies across all
    31 subband boundaries of a granule-channel's reordered `xr[576]`:
    `xar[18·sb-1-i] = xr[18·sb-1-i]·cs[i] − xr[18·sb+i]·ca[i]` and
    `xar[18·sb+i] = xr[18·sb+i]·cs[i] + xr[18·sb-1-i]·ca[i]`, computing
    both outputs from the original inputs. Granules with `block_type == 2`
    (short or mixed) pass through unchanged per the spec's literal
    `block_type`-only scope.
  - Table 3-B.9 raw coefficients `ALIAS_C = [−0.6, −0.535, −0.33, −0.185,
    −0.095, −0.041, −0.0142, −0.0037]` with the derived butterfly
    multipliers `alias_cs()` = `1/√(1+c²)` and `alias_ca()` = `c/√(1+c²)`.
  - 9 alias unit tests: Table B.9 verbatim coefficients, the
    `cs²+ca² == 1` / `ca/cs == c` derivation identities, known `cs0`/`ca0`
    values, short- and mixed-block pass-through, the first-boundary
    butterfly, original-input cross terms, all-31-boundaries coverage, and
    the absence of a boundary below subband 0.
- Clean-room Layer III **stereo processing** stage in the new `stereo`
  module, built solely from ISO/IEC 11172-3:1993 §2.4.3.4.9 (the MS
  matrix and the intensity-stereo steps), §2.4.2.3 (the `mode_extension`
  table), and ISO/IEC 13818-3:1997 §2.4.3.2 (the LSF intensity step 4/5
  replacement and the `intensity_scale` factor):
  - `process_stereo` reconstructs a joint-stereo granule's left/right
    channels in place from the transmitted mid/side and intensity-position
    forms, dispatching on the two `mode_extension` bits: `'00'`
    pass-through, `'10'` whole-spectrum MS, `'01'`/`'11'` intensity (MS
    applying below the intensity bound when `'11'`).
  - MS matrix `L = (M+S)/√2`, `R = (M−S)/√2` (§2.4.3.4.9.2). Intensity
    stereo above the bound: MPEG-1 `is_ratio = tan(is_pos·π/12)` →
    `L = L·is_ratio/(1+is_ratio)`, `R = L/(1+is_ratio)`; LSF power-law
    factor `i0` (`1/√2` or `1/√√2` by `intensity_scale`) with `R = L·kr`,
    `L = L·kl` (§2.4.3.4.9.3 / ISO/IEC 13818-3 §2.4.3.2). `is_pos == 7`
    marks an illegal (non-intensity) band → MS fallback if MS is enabled,
    else independent channels.
  - The intensity bound is the band of the last non-zero right-channel
    line (§2.4.3.4.9.1), computed per window for short blocks (ISO/IEC
    13818-3 §2.4.3.2); mixed blocks process their long region with the
    long-band layout. `long_band_starts` is now `pub(crate)` (shared with
    `requantize` / `reorder`'s Table B.8 tables).
  - 16 stereo unit tests from spec-derived patterns: `'00'` no-op, MS
    matrix (whole-spectrum + orientation), MPEG-1 intensity (`is_pos`
    0 / mid / illegal, with/without MS fallback), MS-below / intensity-
    above split, LSF factors (both `intensity_scale`, both parities), and
    short-block per-window intensity bounds.
- Clean-room Layer III **short-block reordering** stage in the new
  `reorder` module, built solely from ISO/IEC 11172-3:1993 §2.4.3.4.8
  (the reorder requirement), §2.4.2.7 (the native Huffman
  `(scf_band, window, freqline)` interleave), and the Table B.8
  short-block band-start indices:
  - `reorder` → fresh `[f32; 576]` `xr` in subband order. For a short
    band with per-window start `s` and width `w`, the native span
    `[3·s, 3·(s+w))` laid out as `[win0][win1][win2]` is rewritten to
    `out[3·s + 3·k + win] = in[3·s + win·w + k]`, so each consecutive
    run of 18 lines forms one polyphase subband (6 frequency lines × 3
    windows) for the §2.4.3.4.10 IMDCT.
  - Long / start / end blocks (`block_type != 2`) pass through
    unchanged (already in increasing-frequency order, §2.4.2.7); a
    mixed block reorders only its short region (short bands 3..12,
    interleaved lines 36..) while its long region (lines 0..36) passes
    through.
  - Shares the Table B.8 short-block band-start table with the
    `requantize` stage (`short_band_starts` is now `pub(crate)`).
  - 11 reorder unit tests from spec-derived patterns: long pass-through,
    band-0/band-6 three-window interleave, the first-18-line subband
    structure, bijection (permutation) checks at 44.1/48/32 kHz,
    mixed-block long-region preservation + short-region reorder from
    band 3, above-highest-band pass-through, and start/end pass-through.
- Clean-room Layer III **requantization** stage in the new
  `requantize` module, built solely from ISO/IEC 11172-3:1993
  §2.4.3.4.7.1 (the requantization formula), Table B.6 (the `pretab`
  preemphasis table), and Table B.8 (scalefactor-band start indices);
  ISO/IEC 13818-3:1997 defers the LSF formula to §2.4.3.4, so the LSF
  path is the identical equation. Numeric constants transcribed by hand
  from the PDF pages rendered with `pdftoppm` (formula p.34–35, Table
  B.6 p.53, Table B.8 p.62–64):
  - `requantize` → `[f32; 576]` `xr` buffer for one granule-channel:
    `sign(is)·|is|^(4/3)·2^((global_gain − 210 [− 8·subblock_gain])/4)·
    2^(−(scalefac_multiplier·(scalefac + preflag·pretab)))`. Covers the
    long-block formula, the short-block per-window form (with
    `subblock_gain` in the gain exponent and `scalefac_s[sfb][window]`),
    the mixed-block split (lowest 36 lines / long bands 0..8 long, short
    bands 3..12 short), and the LSF (MPEG-2 / MPEG-2.5) variant.
  - `PRETAB` (Annex B Table B.6, 21 entries) and `scalefac_multiplier`
    (`0.5` / `1.0` per `scalefac_scale`, §2.4.2.7) exported. The system
    constant `210` scales the output into `[−1.0, +1.0]`.
  - Short-block lines remain in native `(sfb, window, freqline)`
    interleave; the §2.4.3.4.8 reorder and §2.4.3.4.9 stereo processing
    are deferred to later rounds.
  - 19 requantize unit tests from spec-derived patterns: long-block
    unit-gain identity, global-gain 4-step doubling, scalefactor and
    `scalefac_scale` terms, preflag/pretab on and off, per-window
    `subblock_gain` / `scalefac_s`, short-band interleave, mixed-block
    long-then-short split, the LSF path, sign preservation, and a
    large-magnitude finiteness check.
- Clean-room Layer III main-data **Huffman decode** stage in the new
  `huffman` module, built solely from ISO/IEC 11172-3:1993 §2.4.1.7
  (`Huffmancodebits()` syntax), §2.4.2.7 (semantics), Table 3-B.7
  (Huffman codebooks), and Table 3-B.8 (scalefactor-band start
  indices); codebook entries transcribed by hand from the Annex B
  render of the 157-page PDF (rendered with `pdftoppm`):
  - `decode_huffman` → `[i32; 576]` quantized-line buffer for one
    granule-channel, covering the three-region big_values partition
    (region split aligned to the long-block scalefactor-band start
    indices for 32 / 44.1 / 48 kHz from Table 3-B.8), the count1
    quadruple partition (Table A variable-length Huffman or Table B
    trivial 4-bit code per `count1table_select`), bit-budget
    termination, and zero-fill of the remainder. Sign bits and the
    `linbits` ESC mechanism are wired per the §2.4.1.7
    `Huffmancodebits()` `No. of bits` table on p.18.
  - Codebook coverage: Table 3-B.7 tables 0..=13 (quad A/B + the
    small/medium big-values tables, including the 16×16 table 13)
    fully transcribed. Tables 4 and 14 are spec-marked "not used" and
    are rejected with `HuffmanError::UnusedTable`. The large 16×16
    tables 15, 16, 24 and the `linbits` aliases 17..=23, 25..=31 are
    deferred to a follow-up round and surface as
    `HuffmanError::TableNotYetTranscribed` when requested.
  - 27 huffman-stage unit tests including a Table 1 big-values pair
    with sign, a Table 13 magnitude-15 literal (linbits=0), count1
    quad A and B decode paths, a region-boundary split, a bit-budget
    exhaustion case, and per-table prefix-freeness + Kraft-inequality
    self-checks on every transcribed codebook.
- Clean-room MPEG audio **framing** layer in the new `frame` module,
  built solely from ISO/IEC 11172-3:1993 (§2.4.1.3 / §2.4.2.3) and
  ISO/IEC 13818-3:1997 (§2.4.2.3 lower-sampling-frequency
  redefinitions):
  - `parse_header` → typed `Mp3FrameHeader` (syncword validation;
    MPEG-1/MPEG-2 version via the `ID` bit; layer; CRC-protection
    flag; per-version/per-layer bitrate ladders; sampling frequency;
    padding; channel mode; mode_extension; copyright/original;
    emphasis). Reserved/forbidden field values are rejected via a
    typed `HeaderError`.
  - Per-frame byte-length computation including the padding slot
    (`Mp3FrameHeader::frame_len`), with `samples_per_frame` and
    `channel_count` helpers. Free-format frames report no length.
  - `FrameWalker`: a self-delimiting frame iterator with mid-stream
    resynchronisation on bad sync, garbage gaps, and truncated
    trailing frames.
  - 22 unit tests built from spec-derived byte patterns.
- Clean-room MPEG-1 Layer III **side-information** parser in the new
  `side_info` module, built solely from ISO/IEC 11172-3:1993 §2.4.1.7
  (syntax) and §2.4.2.7 (semantics):
  - `parse_side_info` → typed `SideInfo` over the fixed-size block
    (17 bytes mono / 32 bytes stereo): `main_data_begin` (9 bits),
    `private_bits` (5-bit mono / 3-bit stereo), and `scfsi[ch][band]`
    (4 bands/channel).
  - Per-granule (×2) per-channel `GranuleChannel`: `part2_3_length`,
    `big_values`, `global_gain`, `scalefac_compress`,
    `window_switching_flag`, the window-switching branch
    (`block_type` via a typed `BlockType`, `mixed_block_flag`,
    `table_select`, `subblock_gain`, plus the §2.4.2.7 default
    `region0_count` / `region1_count`) versus the long-block branch
    (`table_select[3]`, `region0_count`, `region1_count`), and
    `preflag` / `scalefac_scale` / `count1table_select`.
  - Non-Layer-III headers and short slices are rejected via a typed
    `SideInfoError` (`NotLayer3` / `TooShort`).
- MPEG-2 / MPEG-2.5 **LSF (lower-sampling-frequency) side-info**
  variant in `side_info`, built solely from ISO/IEC 13818-3:1997
  §2.4.1.7 (syntax) and §2.4.2.7 (semantics). `parse_side_info` now
  dispatches on `Mp3FrameHeader::version`:
  - LSF form decodes a fixed-size block (9 bytes mono / 17 bytes
    stereo, matching the §2.4.2.4 CRC bit ranges) with an 8-bit
    `main_data_begin`, `private_bits` (1-bit mono / 2-bit stereo),
    **no** `scfsi`, and a **single** granule.
  - Per-channel `scalefac_compress` widens to 9 bits; `preflag` is
    not transmitted in LSF (§2.4.2.7 derives it from
    `scalefac_compress`), so the parser leaves it `false`.
  - `GranuleChannel::scalefac_compress` widened `u8` → `u16`;
    `SideInfo` gains `lsf` and `granule_count` markers and a
    layout-aware `byte_len()`; new `GRANULES_LSF`,
    `SIDE_INFO_BYTES_LSF_MONO`, `SIDE_INFO_BYTES_LSF_STEREO`
    constants. `SideInfoError::NotMpeg1` renamed `NotLayer3` (both
    MPEG-1 and MPEG-2 Layer III are now parsed).
  - 16 side-info unit tests total (was 10) from spec-derived byte
    patterns: MPEG-1 mono/stereo long + window-switching +
    saturation, plus LSF mono/stereo long, LSF window-switched short,
    LSF max-field saturation, LSF too-short, and LSF trailing-byte
    cases.
- Clean-room Layer III **scalefactor decode** stage in the new
  `scalefactors` module, built solely from ISO/IEC 11172-3:1993
  (§2.4.1.7 main_data syntax, §2.4.2.7 semantics, §2.4.3.4.5) and
  ISO/IEC 13818-3:1997 (§2.4.1.7 / §2.4.3.4):
  - A main-data **bit reservoir**: `Reservoir::assemble` reconstructs a
    frame's contiguous main-data run from its `main_data_begin`
    back-pointer plus the frame's own bytes, with bounded retention
    (512-byte trailing window) and a `ReservoirUnderflow` error.
    `MainDataReader` is the MSB-first bit reader over that run.
  - **MPEG-1**: the `MPEG1_SLEN` `scalefac_compress → (slen1, slen2)`
    table; `decode_scalefactors` reads long blocks across the four
    `scfsi` band groups (`[0,6) [6,11) [11,16) [16,21)`), reusing
    granule 0's values into granule 1 where `scfsi[ch][group]` is set,
    and reads pure-short / mixed-short blocks per window (mixed reads
    the 8-band long-window portion first).
  - **MPEG-2 / MPEG-2.5 LSF**: `lsf_scale_params` derives
    `slen1..slen4`, `nr_of_sfb1..4`, `preflag`, and `intensity_scale`
    from the 9-bit `scalefac_compress` across all six §2.4.3.4 ranges
    (incl. the right-channel intensity-stereo `int_scalefac_compress`
    branch); the four partitions are read consecutively into long /
    short / mixed-short layouts. One granule, no `scfsi`.
  - Typed `FrameScaleFactors` / `ScaleFactors` output indexed
    `[granule][channel]`; `ScaleFactorError` for reservoir underflow
    and main-data exhaustion. 24 unit tests from spec-derived bit
    patterns, including `scfsi` reuse, the `part2_length` formulas as a
    bit-count cross-check, LSF four-partition / intensity / short
    layouts, and reservoir back-reference / underflow / trim cases.
- `register()` remains a no-op (no decoder/demuxer wired yet); the
  decode/encode surface still returns `Error::NotImplemented`.

### Erased

- Prior master history was force-erased on **2026-05-24** under
  Hat-3 cold enforcement of the workspace clean-room policy
  (`docs/IMPLEMENTOR_ROUND.md`). The retired implementation
  documented several of its data tables and decode-loop structures
  as having been consulted from external reference implementations
  rather than derived solely from the ISO/IEC specification. The
  clean-room policy forbids consulting any external implementation's
  source for any reason, regardless of that reference's licensing.

### Reset

- Crate reduced to a minimal `oxideav_core::register!` stub. Every
  public API returns `Error::NotImplemented`. The cross-crate
  dependencies on `oxideav-id3` and `oxideav-mp1` (the latter for a
  re-exported synthesis-window table) were dropped. The crates.io
  version (`0.1.2`) is preserved on the new master to avoid breaking
  downstream version pins; the published versions on crates.io will
  be yanked by the maintainer.

### Next

- Clean-room re-implementation against the staged ISO/IEC 11172-3 /
  13818-3 Layer III specification (numeric tables read only from the
  standard) in a future round.
