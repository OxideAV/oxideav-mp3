# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the crate adheres
to [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
