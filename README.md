# oxideav-mp3

A pure-Rust **MPEG-1 / MPEG-2 Audio Layer III** (MP3) codec for the
[oxideav](https://github.com/OxideAV/oxideav-workspace) framework.

## Status

**Clean-room rebuild in progress (started 2026-05-24).** The prior
implementation was retired under the workspace
[clean-room policy](https://github.com/OxideAV/oxideav-workspace/blob/master/docs/IMPLEMENTOR_ROUND.md):
several of its data tables and decode-loop structures were documented
as having been consulted from external reference implementations
(their source, not the ISO/IEC specification). The clean-room policy
forbids consulting any external implementation's source for any
reason, independent of that reference's licensing, so the provenance
could not be defended. Master history was fully erased per the Hat-3
cold-enforcement procedure.

The rebuild proceeds against ISO/IEC 11172-3:1993 and
ISO/IEC 13818-3:1997, with every numeric constant transcribed from
those standards and from no other source.

### Implemented

The `frame` module covers the MPEG audio **framing** layer
(ISO/IEC 11172-3 §2.4.1.3 / §2.4.2.3, with the lower-sampling-frequency
field redefinitions of ISO/IEC 13818-3 §2.4.2.3):

- `parse_header` decodes the four-byte frame header into a typed
  `Mp3FrameHeader`: 12-bit syncword check, MPEG version (`ID` bit,
  MPEG-1 vs MPEG-2 LSF), layer, CRC-protection flag, bitrate
  (per-version/per-layer ladders), sampling frequency (32/44.1/48 kHz
  for MPEG-1, 16/22.05/24 kHz for MPEG-2), padding, channel mode,
  mode_extension, copyright, original, and emphasis. Reserved /
  forbidden field values are rejected.
- Per-frame byte length including the padding slot, derived from the
  §2.4.2.3 padding formula (Layer III: `144·bitrate/Fs` for MPEG-1,
  `72·bitrate/Fs` for MPEG-2; Layer I/II covered too). Free-format
  frames report `None` (not derivable from the header alone).
- `FrameWalker` iterates self-delimiting frames over a byte buffer and
  resynchronises mid-stream on bad sync, garbage gaps, or a truncated
  trailing frame.

The `side_info` module covers the **Layer III side-information** block
that follows the header (and optional CRC), for both layouts:
MPEG-1 (ISO/IEC 11172-3 §2.4.1.7 syntax, §2.4.2.7 semantics) and
MPEG-2 / MPEG-2.5 lower-sampling-frequency (ISO/IEC 13818-3 §2.4.1.7
/ §2.4.2.7). `parse_side_info` dispatches on the header's MPEG version:

- **MPEG-1**: fixed-size block (17 bytes mono / 32 bytes stereo) →
  typed `SideInfo` with `main_data_begin` (9-bit), `private_bits`
  (5-bit mono / 3-bit stereo), `scfsi[ch][band]`, and two granules.
- **MPEG-2 / MPEG-2.5 LSF**: fixed-size block (9 bytes mono / 17 bytes
  stereo) with `main_data_begin` (8-bit), `private_bits` (1-bit mono /
  2-bit stereo), **no** `scfsi`, and a single granule. Per-channel
  `scalefac_compress` widens to 9 bits, and `preflag` is **not**
  transmitted (the §2.4.2.7 LSF procedure derives it from
  `scalefac_compress` at scalefactor-decode time, so the parser leaves
  it `false`). `SideInfo` exposes `lsf` and `granule_count` markers,
  and `byte_len()` is layout-aware.
- Per granule per channel, a `GranuleChannel` carries
  `part2_3_length`, `big_values`, `global_gain`, `scalefac_compress`,
  `window_switching_flag`, and its branch — `block_type` /
  `mixed_block_flag` / two `table_select` / three `subblock_gain` when
  window-switched (with the §2.4.2.7 default `region0_count` 7 or 8 and
  `region1_count` 63), or three `table_select` / `region0_count` /
  `region1_count` for a long block — plus `preflag`, `scalefac_scale`,
  and `count1table_select`. Mono and stereo, long-block and
  window-switching cases are unit-tested from spec-derived byte
  patterns for both MPEG-1 and LSF.

The `scalefactors` module covers the Layer III **scalefactor decode**
stage — the main-data step between side-info parsing and Huffman
decode:

- A main-data **bit reservoir** (`Reservoir`) that assembles each
  frame's contiguous main-data run from its `main_data_begin`
  back-pointer plus the frame's own main-data bytes (ISO/IEC 11172-3
  §2.4.2.7), with bounded retention and an underflow error for
  back-references that precede the buffered history. `MainDataReader`
  reads it MSB-first.
- **MPEG-1** (ISO/IEC 11172-3 §2.4.2.7 / §2.4.3.4.5): `slen1`/`slen2`
  from the 4-bit `scalefac_compress` via the `MPEG1_SLEN` table; long
  blocks read 21 bands across four `scfsi` band groups, reusing
  granule 0's scalefactors into granule 1 where `scfsi[ch][group]` is
  set; pure-short and mixed-short blocks read per-window scalefactors
  (mixed populates the long-window portion first). `preflag` is the
  transmitted side-info bit (never set for short blocks).
- **MPEG-2 / MPEG-2.5 LSF** (ISO/IEC 13818-3 §2.4.3.4): `lsf_scale_params`
  derives `slen1..slen4`, the `nr_of_sfb1..4` partition sizes,
  `preflag`, and `intensity_scale` from the 9-bit `scalefac_compress`
  (and, for the right channel of an intensity-stereo frame, from
  `int_scalefac_compress = scalefac_compress >> 1`). The four
  partitions are read consecutively into long / short / mixed-short
  layouts. There is one granule, so no `scfsi` reuse.
- `decode_scalefactors` produces a typed `FrameScaleFactors`
  (`[granule][channel]` of `ScaleFactors`). All cases above are
  unit-tested from spec-derived bit patterns, including `scfsi` reuse,
  the `part2_length` formulas as a bit-count cross-check, and reservoir
  back-references / underflow.

The `huffman` module covers the Layer III main-data **Huffman decode**
stage — the `Huffmancodebits()` syntax of ISO/IEC 11172-3 §2.4.1.7
with the semantics of §2.4.2.7, producing the 576 quantized frequency
lines `is[0..576]` of one granule-channel:

- **big_values** partition (§2.4.2.7, "table_select" / "region0_count"):
  the `big_values` value-pairs are split into three regions by
  `region0_count` / `region1_count`, each region selecting one of the
  Annex B Table 3-B.7 codebooks via `table_select`. Region boundaries
  come from Table 3-B.8 long-block band-start indices (all three
  sampling rates transcribed). Each pair is decoded as
  `hcod[|x|][|y|]` (variable-length Huffman), optionally followed by
  `linbits` ESC extension fields on magnitude-15 symbols, then sign
  bits for non-zero values. Window-switched short blocks fall back to
  the spec's fixed 36-line region-0 / rest-region-1 split.
- **count1** partition (§2.4.2.7, "count1table_select"): quadruples
  `(v, w, x, y)` of magnitude ≤ 1 are decoded via Table A (variable-
  length Huffman) or Table B (the trivial 4-bit code) per
  `count1table_select`, with sign bits on non-zero values. The loop
  terminates the moment the granule's part-3 bit budget is exhausted
  (the spec's "until all bits are spent" rule).
- Remaining lines up to index 576 are zero-filled.

Codebook coverage: **all** of Table 3-B.7 — codebooks 0..=31 minus the
spec-marked "not used" tables 4 and 14, which are rejected. The quad
tables A/B and the small/medium big-values tables (1×1 … 8×8, plus the
16×16 table 13) and the three large 16×16 codebooks 15, 16 and 24 were
transcribed by hand from the Annex B render of ISO/IEC 11172-3:1993
(rendered with `pdftoppm`). The `linbits` aliases 17..=23 reuse table
16's codes (with `linbits` 2/3/4/6/8/10/13) and 25..=31 reuse table
24's codes (`linbits` 5/6/7/8/9/11/13), per the "same as table N, but
linbits=L" notes in the Annex. `HuffmanError::TableNotYetTranscribed`
is retained for API stability but is no longer produced. Thirty-four
huffman-stage unit tests cover spec-derived bitstreams (including the
new tables' zero/signed/ESC paths and two alias families) plus
per-table prefix-freeness + Kraft-inequality self-checks, and a
256-symbol prefix-free / Kraft-sum-equals-1 proof on each 16×16 table.

The `requantize` module covers the Layer III **requantization** stage
(ISO/IEC 11172-3:1993 §2.4.3.4.7.1) — the step that turns the 576
quantized integer lines `is[576]` from the Huffman decode into 576
float frequency lines `xr[576]` for one granule-channel:

- `requantize` evaluates `xr_i = sign(is_i)·|is_i|^(4/3)·2^gain·2^sf`,
  where the gain term is `2^((global_gain − 210)/4)` for long blocks and
  `2^((global_gain − 210 − 8·subblock_gain[window])/4)` for short
  blocks, and the scalefactor term is `2^(−(scalefac_multiplier·
  (scalefac + preflag·pretab)))`. The `scalefac_multiplier` is `0.5`
  (`scalefac_scale == 0`) or `1.0` (`scalefac_scale == 1`) per the
  §2.4.2.7 table; `210` is the §2.4.3.4.7.1 system constant scaling the
  output into `[−1.0, +1.0]`.
- `PRETAB` is the 21-entry Annex B Table B.6 preemphasis table; it is
  added to the long-block scalefactors only when the effective
  `preflag` is set (never for short blocks, §2.4.2.7).
- Long, pure-short (with per-window `subblock_gain` / `scalefac_s`), and
  **mixed** blocks (lowest 36 lines / long bands 0..8 use the long
  formula, short bands 3..12 use the short formula) are all handled, as
  is the LSF (MPEG-2 / MPEG-2.5) variant, which shares the identical
  §2.4.3.4 formula (ISO/IEC 13818-3 defers to ISO/IEC 11172-3 here).
  Short-block lines stay in their native `(sfb, window, freqline)`
  interleave; the §2.4.3.4.8 reorder is the next stage (`reorder` module
  below).
- Band→scalefactor mapping uses the Table B.8 long- and short-block
  start indices for 32 / 44.1 / 48 kHz (LSF reuses the long layouts
  this round; LSF-specific band tables are deferred). 19 unit tests
  cover long-block unit-gain identity, the global-gain 4-step doubling,
  the scalefactor and `scalefac_scale` terms, the preflag/pretab effect
  (on and off), per-window `subblock_gain` and `scalefac_s`, short-band
  interleave, the mixed-block split, the LSF path, sign preservation,
  and a large-magnitude finiteness check.

The `reorder` module covers the Layer III **short-block reordering**
stage (ISO/IEC 11172-3:1993 §2.4.3.4.8) — the step between
requantization and the IMDCT that rearranges short-block coefficients
into frequency (subband) order:

- `reorder` rewrites the requantized short-block lines from their native
  `(scf_band, window, freqline)` Huffman interleave (§2.4.2.7) into
  subband order `xr[subband][window][freqline]`. For a short band with
  per-window start `s` and width `w`, the native span `[3·s, 3·(s+w))`
  laid out as three runs `[win0][win1][win2]` becomes
  `out[3·s + 3·k + win] = in[3·s + win·w + k]`, so each consecutive run
  of 18 output lines forms one polyphase subband (6 frequency lines × 3
  windows) — the unit the §2.4.3.4.10 synthesis filterbank consumes.
- Long / start / end blocks (`block_type != 2`) pass through unchanged
  (already increasing-frequency-ordered). A **mixed** block reorders
  only its short region (short bands 3..12, interleaved lines 36..)
  while its long region (lines 0..36) passes through. Band boundaries
  reuse the same Table B.8 short-block start indices as `requantize`.
- 11 unit tests cover long pass-through, the band-0 / band-6 three-window
  interleave, the first-18-line subband structure, bijection checks at
  44.1 / 48 / 32 kHz, mixed-block long-region preservation and
  short-region reorder from band 3, above-highest-band pass-through, and
  start/end pass-through.

The `stereo` module covers the Layer III **stereo processing** stage
(ISO/IEC 11172-3:1993 §2.4.3.4.9, with the ISO/IEC 13818-3:1997
§2.4.3.2 intensity modifications for MPEG-2 / MPEG-2.5 LSF) — the step
between short-block reorder and alias reduction that reconstructs a
joint-stereo granule's left/right channels from the transmitted
mid/side and intensity-position representations:

- `process_stereo` reconstructs both channels in place per the two
  `mode_extension` header bits (§2.4.2.3): `'00'` (neither) passes
  through; `'10'` (MS only) decodes the **entire** spectrum with the MS
  matrix; `'01'` / `'11'` enable intensity stereo (with MS applying
  below the intensity bound when `'11'`).
- **MS matrix** (§2.4.3.4.9.2): `L = (M+S)/√2`, `R = (M−S)/√2`, with the
  mid signal in the left channel and the side signal in the right.
- **Intensity stereo** (§2.4.3.4.9.3): above the intensity bound the
  right channel's per-band scalefactor is reused as the stereo position
  `is_pos`. MPEG-1 uses `is_ratio = tan(is_pos·π/12)` →
  `L = L·is_ratio/(1+is_ratio)`, `R = L/(1+is_ratio)` (with `is_pos == 7`
  the illegal-position marker). LSF (ISO/IEC 13818-3 §2.4.3.2 step 4/5
  replacement) uses a power-law factor `i0` (`1/√2` when
  `intensity_scale == 1`, else `1/√√2`): `R = L·kr`, `L = L·kl` where
  `kl`/`kr` derive from `is_pos` parity. An illegal position falls back
  to the MS matrix when MS is enabled, or leaves the channels
  independent otherwise.
- The intensity bound is derived from the last non-zero right-channel
  line (§2.4.3.4.9.1), computed **per window** for short blocks (ISO/IEC
  13818-3 §2.4.3.2). Mixed blocks handle their long region (lines 0..36)
  with the long-band layout. Band boundaries reuse the Table B.8 long-
  and short-block start tables from `requantize` (`long_band_starts` is
  now `pub(crate)`).
- 16 unit tests cover the no-op `'00'` case, the MS matrix (whole-spectrum
  and orientation), MPEG-1 intensity (`is_pos` 0 / mid / illegal, with
  and without the MS fallback), the combined MS-below / intensity-above
  split, the LSF power-law factors for both `intensity_scale` values and
  both `is_pos` parities, and short-block per-window intensity bounds.

The `alias` module covers the Layer III **alias reduction** stage
(ISO/IEC 11172-3:1993 §2.4.3.4.10.1) — the eight-butterfly
decorrelation across each subband boundary that runs after the reorder
and immediately before the IMDCT:

- `alias_reduce` applies the spec's pseudo code over all 31 subband
  boundaries (`sb = 1..32`) of a granule-channel's reordered `xr[576]`:
  `xar[18·sb-1-i] = xr[18·sb-1-i]·cs[i] − xr[18·sb+i]·ca[i]` and
  `xar[18·sb+i] = xr[18·sb+i]·cs[i] + xr[18·sb-1-i]·ca[i]` for
  `i = 0..8`, with both outputs of each butterfly computed from the
  original inputs.
- The butterfly multipliers derive from Table 3-B.9's raw coefficients
  `ALIAS_C = [−0.6, −0.535, −0.33, −0.185, −0.095, −0.041, −0.0142,
  −0.0037]`: `alias_cs()` = `1/√(1+c²)`, `alias_ca()` = `c/√(1+c²)`.
- Granules with `block_type == 2` (short or mixed) pass through
  unchanged — the spec scopes the stage on `block_type` alone ("not
  applied for granules with block-type == 2"), so a mixed block is
  excluded too (see the spec gap below).
- 9 unit tests cover the verbatim Table B.9 coefficients, the
  `cs²+ca² == 1` / `ca/cs == c` derivation identities, known `cs0`/`ca0`
  values, short- and mixed-block pass-through, the first-boundary
  butterfly, original-input cross terms, all-31-boundary coverage, and
  the absence of a boundary below subband 0.

The `imdct` module covers the Layer III **IMDCT, windowing, overlap-add
and frequency inversion** stages (ISO/IEC 11172-3:1993 §2.4.3.4.10.2 /
§2.4.3.4.10.3 / §2.4.3.4.10.4 / §2.4.3.4.10.5) — the per-subband
transform stack that runs after alias reduction and produces the 32×18
subband-domain time samples consumed by the polyphase synthesis
filterbank (next stage):

- `imdct(xk, n)` evaluates the spec formula
  `x[i] = Σ_{k=0..n/2-1} X[k]·cos((π/(2n))·(2i+1+n/2)·(2k+1))` for
  `i = 0..n-1`, with `n = 36` for long blocks and `n = 12` for short
  blocks (a short block runs three independent 12-point IMDCTs over its
  three 6-line windows, which the reorder stage has already interleaved
  as `lines[3·k + j]` = freq-line `k` of window `j`).
- Windowing (§2.4.3.4.10.3) applies the four block-type window shapes:
  the normal sine window `sin((π/36)(i+½))` for block-type 0; the start
  block (long half-window 0..17, pass-through 18..23, short half-window
  24..29, zero 30..35); the stop block (zero 0..5, short half-window
  6..11, pass-through 12..17, long half-window 18..35); and the short
  block (each 12-point IMDCT windowed with `sin((π/12)(i+½))` then the
  three windowed sub-blocks overlapped/concatenated per the §2.4.3.4.10.3
  table — `y₀` in 6..17, `y₀+y₁` in 12..17, `y₁+y₂` in 18..23, `y₂` in
  24..29, zeros in 0..5 and 30..35). A **mixed** block's two lowest
  subbands (0, 1) use the long IMDCT + normal window per §2.4.2.7; the
  remaining 30 subbands use the short path.
- `ImdctState` carries the per-subband saved second-half overlap across
  granules. `imdct_granule(xr, gc, &mut state)` runs the full §2.4.3.4.10
  pipeline for one granule-channel: per subband, IMDCT → window → overlap
  the first 18 samples with the saved second half from the previous
  granule (`result[i] = z[i] + s_prev[i]`) and save the new second half
  (`s_next[i] = z[i+18]`). Finally §2.4.3.4.10.5 frequency inversion
  negates every odd time sample of every odd subband to compensate for
  the polyphase filterbank's frequency inversion (next stage).
- 22 unit tests cover the IMDCT closed-form impulse response (n=12),
  hand-computed all-ones reference values (n=12 and n=36), linearity,
  byte-exact long and short window tables (with symmetry and Σw²=18
  cross-checks), the normal / start / stop window shape per spec, the
  short-block concatenation table (including the mid-block sum of two
  adjacent windowed sub-blocks), zero-input zero-output, overlap state
  initial zeroness, first-granule output equals z[0..17] when s_prev=0,
  second-granule output adds the saved overlap from granule 1,
  per-subband overlap isolation, frequency inversion of odd subbands'
  odd time samples (with even subbands unaffected), short-block
  three-sub-IMDCT dispatch in subband 0, mixed-block long-window
  dispatch in subbands 0 and 1 (with the §2.4.3.4.10.5 sign-flip on
  subband 1), start-block tail-zero through `imdct_granule`, and
  stop-block head-zero through `imdct_granule`.

The `synth` module covers the Layer III **polyphase synthesis subband
filterbank** (ISO/IEC 11172-3:1993 §2.4.3.2 / Figure A.2; the same
filter as Layer I §2.4.3.2.2 and Layer II §2.4.3.3.5) — the **last**
decode stage. With this in place the granule-level decode chain is now
complete end-to-end (Huffman → requantize → reorder → stereo → alias →
IMDCT → polyphase synthesis → PCM):

- `synth_row(s, &mut state)` runs one pass of the Figure A.2 flow chart
  over 32 input subband samples `S[0..32]` and produces 32 PCM samples:
  the 1024-value shift register `V[]` is moved up 64 places, the 64×32
  matrixing operation `V[i] = Σ_{k=0..32} N[i,k]·S[k]` writes 64 fresh
  values with the §2.4.3.2.2 coefficient
  `N[i,k] = cos((16+i)·(2k+1)·π/64)`, the 512-element vector `U[]` is
  built from `V` via the spec's `U[64i+j] = V[128i+j]` /
  `U[64i+32+j] = V[128i+96+j]` index map, multiplied entrywise by the
  Annex B Table B.3 window `D[]`, and summed in 16-tap groups
  (`S_out[j] = Σ_{i=0..16} U[j+32i]·D[j+32i]`).
- `D_TABLE` is the 512-value Table B.3 window: every coefficient was
  hand-transcribed from the ISO/IEC 11172-3:1993 PDF (annex pages 50–52
  of the body; rendered PNGs are staged at
  `docs/audio/mp3/annex-b-renders/Table-B.3-coefficients-Di-p5{6,7,8}.png`),
  with the unique global maximum `D[256] = +1.144989014` and the unique
  global minimum `D[255] = −1.144287109` from the prototype filter's
  centre.
- `SynthState` is the per-channel shift register; Figure A.2's
  footnote 1 ("V to be initialised with zeroes during startup") makes
  `SynthState::default()` the correct stream-start state.
- `synth_granule(subband_time, &mut state)` runs the per-row filter 18
  times over the 32×18 IMDCT output of one granule-channel, producing
  the 576 PCM samples in playback order (time-row 0 of all 32 subbands
  → time-row 1 → … → time-row 17). The shift register persists across
  rows and across granules.
- 19 synth unit tests, all derived directly from the spec formulas: D[]
  length / boundary values (D[0], D[1], D[255], D[256], D[257], D[511]
  match Table B.3 byte-for-byte), D[256] is the unique global maximum
  and D[255] the unique global minimum, the inner |D[256±k]| symmetry
  pairs match the printed values, `N[i,k]` matches `cos(π/4)`, `cos(π)`
  and `cos(π/2)` at the four corner / midpoint cases, a hand-computed
  known vector derivation (`S[k0]=1`, all-zero V → `S_out[j] = N[j,k0]·D[j]`)
  for both k0=0 and k0=5 covering steps 2-5 byte-exactly, linearity of
  the whole filter, the shift register propagating an impulse across
  iterations (V[64..128] picks up the previous V[0..64]), `synth_granule`
  agreeing with a manual `synth_row` on the first time-row, an
  end-to-end zero through `imdct_granule → synth_granule` yielding 576
  PCM zeros, and an end-to-end synthetic frame with `xr[0] = 1.0` (DC
  in subband 0) producing 576 finite, partially-non-zero PCM samples.

The `demuxer` module wires the framing layer into an
`oxideav_core::Demuxer` so a pipeline can consume MP3 files end-to-end
at the container level. `Mp3Demuxer::open` is the entry point:

- **ID3v2 frontmatter skip.** When the file begins with the `ID3`
  three-byte magic, the 10-byte tag header is parsed for its
  synchsafe-encoded body size (per `docs/container/id3/id3v2.3.0.html`
  + `id3v2.4.0-structure.html`: `size = (b[0]<<21) | (b[1]<<14) |
  (b[2]<<7) | b[3]`) and the read cursor is advanced past the tag —
  including the optional v2.4 10-byte footer when flag bit `0x10` is
  set. The tag's content is not parsed; only its on-disk length is
  needed to find the first MPEG audio frame.
- **ID3v1 trailer skip.** The last 128 bytes of the file are
  inspected; when the first three are the `TAG` magic
  (datavoyage-mpgscript §"MPEG Audio Tag ID3v1": positions 0..=127,
  fixed 128-byte layout) the audio region ends 128 bytes before the
  file's end so the demuxer never tries to interpret the trailer as
  another frame.
- **Xing / Info VBR-info-frame detection.** After locating the first
  MPEG audio frame the demuxer reads its full payload and checks for
  the `Xing` (true VBR) or `Info` (LAME-CBR convention) four-byte
  magic immediately after the Layer III side-info bytes — 17 bytes
  for MPEG-1 mono, 32 bytes for MPEG-1 stereo, 9 bytes for MPEG-2 LSF
  mono, 17 bytes for MPEG-2 LSF stereo (per `crate::side_info`). The
  four optional fields (`frames`, `bytes`, 100-byte `toc`, `quality`)
  are decoded from a four-bit flag word; when the info frame is
  present it is consumed as a metadata carrier and packet emission
  starts at the next audio frame. **The Xing / Info wire layout is
  not yet staged in `docs/audio/mp3/`** — every numeric field
  offset / width is verified byte-for-byte against the two on-disk
  fixtures `docs/audio/mp3/fixtures/layer3-with-xing-vbri-tag/` and
  `docs/audio/mp3/fixtures/layer3-with-id3v2-tag/` and their
  companion `trace.txt`. A canonical layout doc (e.g. a Xing
  programming guide) would close the residual provenance gap.
- **Duration estimation.** VBR streams with a Xing `frames` field
  report `frames × samples_per_frame / sample_rate`; CBR streams
  use `audio_bytes × 8 / bitrate × sample_rate`. The four-fixture
  reference (CBR-320, VBR-q5, ID3v2-tagged, Xing-tagged) all report
  ~835.9 ms vs ffprobe's 800.0 ms (`Δ = +4.5%`); the residual is the
  LAME encoder-delay/padding that lives in the bytes after the
  prompt-enumerated four Xing fields and is not consumed here.
- **Packet emission.** Each call to `next_packet` reads one MPEG
  audio frame at the current cursor (resyncing on bad / overrun /
  sample-rate-mismatched headers), stamps it with a monotonic PTS in
  the stream's `1/sample_rate` time base, sets `keyframe = true`
  (every Layer III frame is a self-contained random-access point at
  the codec level — the bit-reservoir back-reference is the only
  cross-frame dependency, and decoders handle that internally), and
  returns. EOF returns `Error::Eof`.
- **Seek.** VBR streams with a Xing TOC look up the requested
  percentile in the `toc[100]` table and snap to the next valid
  frame sync; CBR streams use proportional byte-offset arithmetic
  (`pts × bitrate / 8 / sample_rate`). Both then resync over the
  next ~8 KB.
- **Probe + registration.** `register()` installs the demuxer +
  the `.mp3` / `.mp2` / `.mp1` extensions + a content probe that
  scores `ID3v2 + frame sync` and `bare frame sync` candidates
  (100 / 75 / extension-tied 100) into the runtime context.

A 5-test docs/audio/mp3/fixtures/ integration harness drives the
demuxer through the on-disk corpus (CBR-320 / VBR-q5 / ID3v2-tagged /
Xing-tagged + a broad walk over the 15 Layer-III fixtures) and
asserts the demuxed frame count matches the trace, the Xing fields
match the trace, the ID3v2 size matches the trace, and the byte
offset of the first audio frame matches the trace. MPEG-2.5
(`0xFFE3..` sync) is excluded pending the parser extension tracked
in `docs/audio/mp3/MPEG-2.5-GAP.md`; Layer II fixtures are excluded
because the round-121 brief explicitly scopes to Layer III.

The `encoder` module begins the Layer III **encoder** with its Phase 1
**bitstream-formatting** half (the part that needs no psychoacoustic
model):

- `write_header` writes the four-byte frame header
  (ISO/IEC 11172-3 §2.4.1.3 / §2.4.2.3) — the exact byte-for-byte
  inverse of `parse_header`.
- `write_side_info` writes the Layer III side-information block
  (ISO/IEC 11172-3 §2.4.1.7 for MPEG-1, ISO/IEC 13818-3 §2.4.1.7 for
  MPEG-2 / MPEG-2.5 LSF): `main_data_begin`, `private_bits`, `scfsi`,
  and the per-granule-per-channel record (`part2_3_length`,
  `big_values`, `global_gain`, `scalefac_compress`,
  `window_switching_flag`, both window branches, `region*_count`,
  `preflag`, `scalefac_scale`, `count1table_select`) — the exact
  inverse of `parse_side_info` for both layouts.
- `encode_silent_frame` produces a complete, self-delimiting
  all-zero-quantization Layer III frame (`part2_3_length == 0`,
  `big_values == 0` for every granule-channel, no CRC, zero-filled main
  data) sized to `Mp3FrameHeader::frame_len`. The frame round-trips
  through this crate's own `parse_header` / `parse_side_info` /
  `FrameWalker` / `Mp3Demuxer`, and a black-box `ffmpeg` decode of a
  multi-frame stream reconstructs pure silence (every sample 0).
  `make_silent_header` is a CBR convenience constructor that resolves a
  bitrate / sample-rate / channel-mode triple to the raw header
  indices.

The `mdct` module covers the Layer III **encoder Phase 2** analysis
filterbank up to (and including) the forward overlap split — the
encoder-side companion of `imdct::imdct_granule`:

- `mdct(xn, n)` transforms `n` time samples into `n / 2` frequency bins
  using the same cosine kernel
  `cos((π / (2·n)) · (2·i + 1 + n/2) · (2·k + 1))` summed transposed
  (over `i` rather than `k`). `n = 36` for long blocks; `n = 12` for
  each of the three short sub-blocks.
- Exact round-trip against the existing IMDCT, derivable from the
  spec orthogonality of the cosine kernel:
  `mdct(imdct(X), n)[k] = (n / 2) · X[k]`. Verified on impulse,
  multi-bin spectrum, and arbitrary mixed-frequency inputs for both
  `n = 36` and `n = 12`.
- **Analysis windowing** (encoder mirror of §2.4.3.4.10.3): the
  `analysis_long_window(i) = sin((π/36)(i+½))` and
  `analysis_short_window(i) = sin((π/12)(i+½))` primitives are
  identical to the synthesis-side windows (Princen-Bradley TDAC
  requires the same window on both halves of a lapped MDCT codec).
  `window_long_family_analysis(&xn, block_type)` applies the four
  long-family window shapes — plain sine for `Long` (block_type 0);
  long-half 0..17 + pass-through 18..23 + short-half 24..29 + zero
  30..35 for `Start` (block_type 1); zero 0..5 + short-half 6..11 +
  pass-through 12..17 + long-half 18..35 for `End` (block_type 3) —
  to 36 input samples, with the partitioning matching the
  §2.4.3.4.10.3 synthesis table exactly. `window_short_analysis(&xn)`
  extracts the three 12-sample short sub-blocks per the analysis
  inverse `xj_in[j][k] = xn[6 + 6·j + k]` (the same 12-sample spans
  the synthesis side reads y_0/y_1/y_2 from, including the
  half-overlap source regions), each pre-multiplied by the short
  window.
- **Forward overlap split** (encoder mirror of §2.4.3.4.10.4):
  `MdctState` carries the previous granule's 18 subband-time samples
  across calls; `forward_overlap(&current_18, &mut state)` assembles
  the 36-sample forward-MDCT input frame as `[prev_18, current_18]`
  and rolls `prev_18 := current_18`, the structural analog of the
  synthesis side's `result[i] = z[i] + s_prev[i]` /
  `s_next[i] = z[i+18]` overlap-add. Stream-start state is all zeros.
- **End-to-end TDAC verified** on the long-block path: three
  successive granules pushed through `forward_overlap` →
  `window_long_family_analysis(Long)` → `mdct` → `imdct` → long
  window → overlap-add recover the interior granule scaled by
  `n/4 = 9` exactly, on arbitrary mixed-frequency input — the
  Princen-Bradley identity that makes the lapped MDCT a critically-
  sampled exact-reconstruction transform. (Scaling factor: the spec
  MDCT/IMDCT pair has bin-space `mdct(imdct(X)) = (n/2)·X` and
  time-space `imdct(mdct(x))[i] = (n/4)·(x[i] ∓ reflection)`. The
  sine-window pair sum `sin²θ + cos²θ = 1` then yields the `n/4`
  overall.)

The `analysis` module covers the Layer I/II/III **polyphase analysis
subband filterbank** — the first encoder stage on the forward signal
path, splitting a broadband PCM input into 32 critically-sampled
subbands at sample rate `f_s / 32`, per ISO/IEC 11172-3:1993 Annex C
§C.1.3 ("Analysis subband filter") with the per-step pseudo code of
Figure C.4 and the 512 prototype-window coefficients of Annex C Table
C.1. It is the algebraic dual of the §2.4.3.2 / Figure A.2 synthesis
filterbank in `synth`:

- `analyze_row(&pcm, &mut state)` runs one pass of Figure C.4: input
  shift (`X[i] = X[i-32]` for `i = 511..32`, then the 32 new PCM
  samples land at `X[31..0]` in `pcm[0]→X[31] … pcm[31]→X[0]` order),
  512-tap window by the Table C.1 `C[i]` coefficients, 8-fold partial
  calculation `Y[i] = Σ_{j=0..7} C[i + 64·j] · X[i + 64·j]`, and a
  64×32 cosine-modulated matrix multiply by the §C.1.3 kernel
  `M[i, k] = cos((2i+1)(k-16)π/64)`. Consumes 32 PCM samples, returns
  32 subband samples.
- `AnalysisState` carries the 512-element shift register `X[]` across
  rows and persists across granules in the stream. Stream-start state
  is all zeros (the analysis mirror of `synth::SynthState`'s `V[]`).
- `analyze_granule(&pcm576, &mut state)` runs 18 sequential
  `analyze_row` passes over the 576 PCM samples of one Layer III
  granule-channel and lays the result out as `subband_time[sb][t]` —
  exactly the layout `imdct_granule` consumes on the decode side and
  that the forward MDCT chain (`mdct::forward_overlap` →
  `mdct::window_long_family_analysis` → `mdct::mdct`) consumes on the
  encode side. Exact analysis-side mirror of `synth::synth_granule`.
- `C_TABLE` is the 512-entry Annex C Table C.1 prototype window
  transcribed from the staged ISO/IEC 11172-3:1993 PDF. The
  first-half entries are transcribed verbatim from the literal text
  (with OCR letter-for-digit fixes cross-checked against the local
  monotone trend); the second-half entries are derived by the
  cosine-modulated-prototype symmetry `C[512-i] = +C[i]` if
  `i ≡ 0 (mod 64)` and `-C[i]` otherwise (the §C.1.3 polyphase
  construction's structural symmetry, asserted in
  `c_table_satisfies_polyphase_symmetry`).
- **TDAC round-trip verified**, with two algebraically distinct
  identities both passing on the same `C[]` and `M[i,k]`:
  - **PCM round-trip with 481-sample group delay.** Pumping a
    broadband multi-tone PCM signal through `analyze_row → synth_row`
    recovers the input delayed by 481 samples within RMS deviation
    `< 1×10⁻⁴` (measured ≈ 3×10⁻⁵ on the test signal, ≈ -90 dB —
    well inside the spec prototype's near-PR design ripple).
  - **Per-subband DC-tone subband round-trip is exactly
    cyclostationary in steady state.** Driving `synth_row` with a
    constant unit input in one subband (`S[sb0] = 1`, others 0) for
    48 rows and analysing the resulting PCM yields, after a 32-row
    settling delay, a steady-state recovered coefficient that does
    *not* vary from row to row — row-to-row RMS ripple is below
    `1×10⁻¹²` for *every one of the 32 subbands*, the float-
    precision spec-derivable invariant of the bank's cosine-
    modulated structure.

**Phase 2 step 4 (quantization primitive)** is now wired up in the
[`quantize`] module. Given a target spectrum `xr[576]` and an
already-chosen `GranuleChannel` + `ScaleFactors` configuration,
[`quantize::quantize`] computes the integer Huffman-input buffer
`is[576]` that — when fed back through [`requantize::requantize`] with
the same configuration — reproduces `xr[576]` within `f32`
round-to-nearest precision. It is the exact algebraic inverse of the
§2.4.3.4.7.1 decoder formula:

```text
|is_i| = round( (|xr_i| / G(sfb, w))^(3/4) )
G_long (sfb)    = 2^((gg-210)/4) * 2^(-mult * (sf_l[sfb] + preflag*pretab[sfb]))
G_short(sfb, w) = 2^((gg-210-8*subblock_gain[w])/4) * 2^(-mult * sf_s[sfb][w])
```

with the same long / short / mixed split (lines 0..36 long, 36..short_end
short for mixed blocks) and the same per-window subblock-gain bookkeeping
as the decoder. The round-trip `is -> xr -> is' -> xr_back` is **bit-exact
on every tested configuration** (long-block at every `global_gain` in
180..240, scalefactors 0..4, `preflag` on/off, both `scalefac_scale`
settings; short-block with non-zero `subblock_gain[w]`; mixed-block with
non-trivial `scalefac_l` and `scalefac_s`; LSF at 24 kHz). Bin-level RMS
between `xr_back` and `xr_ref` measures `0.0e0` — within `f32` precision
the integer-power-law grid is closed under the encoder/decoder round-trip
the moment `xr` is already on the grid.

The primitive does **not** itself search for `global_gain`, allocate
bits, choose scalefactors, or run §C.1.5.4 (informational) noise-shaping
iterations.

**Phase 2 step 5 (inner-loop `global_gain` search)** wraps that
primitive in the [`inner_loop`] module — the §C.1.5.4.4 rate-control
loop. Holding a chosen scalefactor configuration fixed, it binary-
searches the 8-bit `global_gain` field for the **smallest** gain (finest
quantization, largest output magnitudes) whose quantized `is[576]` meets
a constraint. The search is valid because `|is_i|` — and hence both
`max|is|` and any non-negative bit weighting — is monotone
non-increasing in `global_gain` (a larger gain divides by a larger
`2^((gg−210)/4)`), so the "constraint satisfied" predicate is a step
function over `[0, 255]`.

- `search_magnitude_clamp` enforces the §2.4.1.7 big-values bound
  `max|is| ≤ 8191` (`BIG_VALUES_LIMIT`, the §C.1.5.4.4.2 maximum-value
  test). For a fixed `sf` the coarsest gain (`GAIN_MAX = 255`) divides by
  only a finite `2^((255−210)/4)`, so amplitudes louder than
  `2^11.25 · 8191^(4/3) ≈ 4.0×10⁸` (at `sf = 0`) cannot be clamped by
  gain alone; the result then reports `satisfied == false` and carries
  the `GAIN_MAX` fallback (the outer loop / scalefactors, not in scope,
  would extend the range).
- `search_bit_budget` finds the smallest gain whose **exact**
  §C.1.5.4.4.5 + §C.1.5.4.4.8 Huffman bit count fits a supplied budget
  (Phase 2 step 6, below). The legacy `coarse_bit_estimate`
  (`bits(|is_i|) + 1` summed over non-zero lines) is kept only for
  reference.

**Phase 2 step 6 (exact Huffman bit count)** replaces that placeholder
with the real §C.1.5.4.4.5 / §C.1.5.4.4.8 `count_bits`.
`count_huffman_bits` partitions `is[576]` (§C.1.5.4.4.3 r_zero /
§C.1.5.4.4.4 count1 run-lengths via `partition_split`), splits the
big-values range into three sub-regions (§C.1.5.4.4.6 SUBDIVIDE), and
sums Table 3-B.7 codeword lengths: per big-values pair the
`bitz[min(15,|x|)][min(15,|y|)]` codeword plus a `linbits` ESC field for
each component of magnitude ≥ 15 and a sign bit per non-zero component;
per count1 quad the table-A/B code length plus its sign bits.
`choose_best_table_for_region` / `choose_best_count1_table` pick the
minimum-bit codebook (§C.1.5.4.4.7). The count is **bit-for-bit
identical** to what `decode_huffman` consumes — verified by a
forward-count ⇄ decoder round-trip on hand-assembled big-values + count1
+ linbits bitstreams. Because Huffman codeword lengths are *not* monotone
in magnitude (and the best codebook shifts as values shrink), the exact
count is not monotone in `global_gain`, so the budget search uses the
spec's upward `qquant + 1` scan rather than a bisection.

Verified: per-pair / per-quad / linbits / multi-region counts equal the
hand-summed codeword lengths; table selection returns the true minimum;
the count matches decoder consumption exactly; `max|is|` is monotone and
the chosen clamp gain is minimal; and `requantize(is)` reproduces the
target to within the quantizer-grid bound.

**Phase 2 step 7 (Huffman bit emission)** turns the step-6 count into
actual codewords. `encode_huffman` is the forward counterpart to
`decode_huffman`: given `is[576]`, the big-values `region_ends`, the
per-region `table_select` (from the step-6 choosers), the `count1_quads`
count and `count1table_select`, it emits the §2.4.1.7 `huffmancodebits()`
payload. Each big-values pair writes the Table 3-B.7 codeword for the
clamped `(min(15,|x|), min(15,|y|))` cell, then a `linbits` ESC field
(`|v| - 15`) for any magnitude-≥-15 component, then a sign bit per
non-zero component — in the order codeword → linbits_x → sign_x →
linbits_y → sign_y, the exact inverse of `decode_big_pair`. Each count1
quad writes the table-A/B code plus its sign bits. The result
(`Mp3HuffmanData`) is byte-aligned and reads back through `MainDataReader`
bit-for-bit; its `bit_len` equals the step-6 `count_huffman_bits` for the
same inputs. Verified by a `encode_huffman` ⇄ `decode_huffman` round-trip
that recovers the original `is[]` exactly with `bit_len ==
count_huffman_bits`, across mixed big-values + count1, `linbits` escapes
(tables 16 / 24), table-B count1, a band-aligned three-region split, and
an end-to-end pipeline that derives the split + tables via the step-6
choosers.

**Phase 2 step 8 (main-data assembly)** binds the step-7 codewords into a
complete per-granule/channel main-data block. The `main_data` module's
`assemble_main_data` walks the §2.4.1.7 `main_data()` loop
(`for (gr) for (ch)`), emitting each granule/channel's **part2**
scalefactors immediately followed by its **part3** `huffmancodebits()`
into one shared `MainDataWriter` (the MSB-first inverse of
`MainDataReader`) with no byte alignment between fields. Part2 uses a new
scalefactor writer (`write_mpeg1_granule_channel` / `write_lsf_channel`)
that inverts the `scalefactors` decode path — MPEG-1 long (four scfsi
band groups, granule-1 reuse skips) / short / mixed, and the LSF
four-partition `slen` / `nr_of_sfb` scheme; part3 uses the new public
`emit_huffman` (`encode_huffman` is now a thin wrapper over it). The
assembler derives each granule's region split + `table_select` from the
side info exactly as `decode_huffman` reads them, records every
`part2_3_length` back into the `SideInfo`, sets `main_data_begin = 0`
(the no-reservoir, self-contained-frame schedule), and returns the
byte-padded block plus the `total_bits` sum. Verified by five round-trip
tests that read the block back through the exact §2.4.1.7 loop and
recover the scalefactors + `is[]` bit-exactly with the reader consuming
exactly `total_bits`: MPEG-1 long two-channel, MPEG-1 two-granule mono
(scfsi all-false), MPEG-1 short, LSF long, and a `decode_scalefactors`
first-granule cross-check.

**Phase 2 step 9 (cross-frame bit-reservoir scheduling)** turns the
step-8 self-contained `main_data_begin = 0` path into a real §2.4.2.7
reservoir scheduler that can route a busy frame's main_data backward
into earlier quiet frames' unused tails. The `main_data` module's
`schedule_reservoir` (plus the stateful `ReservoirScheduler` builder)
takes a parallel slice of `(main_data bytes, slot_bytes, lsf?)` triples
and per-frame `SideInfo`s, validates the two §2.4.2.7 schedulability
invariants — `R_i ≥ 0` (cumulative slot ≥ cumulative main_data at every
prefix) and `R_i ≤ 511` / `≤ 255` (the on-wire 9-bit MPEG-1 / 8-bit LSF
cap, surfaced as `RESERVOIR_MAX_MPEG1` / `RESERVOIR_MAX_LSF`) — then
carves the rolling main-data concatenation into per-frame slots and
stamps each frame's `SideInfo::main_data_begin` to point back to where
its main_data actually starts. Schedulability failures surface as
`ReservoirError::SlotUnderflow` (busy frame outran prior reservoir + own
slot) or `ReservoirError::ReservoirOverflow` (reservoir would exceed the
on-wire cap). Verified by eight tests, anchored on a synthetic 3-frame
sequence (quiet / busy / quiet) where the middle frame's main_data
(50 B) exceeds its slot (30 B) and is scheduled into the prior quiet
frame's 20-byte tail (`main_data_begin` sequence `[0, 20, 0]`), and an
end-to-end pipeline cross-check (`big_values = 200` middle frame at
MPEG-1 128 kbps / 44.1 kHz mono) that round-trips every granule's
scalefactors + `is[]` bit-exactly through `Reservoir::assemble` plus the
existing §2.4.1.7 decoder loop. Single-channel MPEG-1 only this round —
multi-channel / LSF / outer-loop integration is deferred.

**Phase 2 step 10 (stream-level PCM → MP3 encoder)** ties the steps
1–9 primitives into a top-level `Mp3Encoder` (`stream_encoder` module)
that consumes `i16` PCM samples and writes a sequence of complete
self-delimiting MP3 frames (header + side_info + main_data slot) to a
`std::io::Write` sink. The driver: (a) runs the §C.1.3 polyphase
analysis filterbank per granule, (b) applies the §2.4.3.4.10.5
frequency inversion + the inverse §2.4.3.4.10.1 alias-reduction butterfly,
(c) runs the §2.4.3.4.10.2 forward MDCT per subband through the
`mdct::MdctState` overlap, (d) quantizes (`quantize::quantize`) at a
gain chosen by the in-tree `inner_loop::search_bit_budget` +
`search_magnitude_clamp` followed by a local `qquant + 1` ratchet that
re-emits with a linbits-reach-filtered Huffman table chooser, (e)
emits `huffman::emit_huffman` into a `main_data::assemble_main_data`
blob, and (f) schedules every assembled frame onto the §2.4.2.7 bit
reservoir via `schedule_reservoir`. **Scope** — mono / MPEG-1 only
this round (`SingleChannel` ChannelMode, 32 / 44.1 / 48 kHz), CBR,
long blocks (no window switching), zero scalefactors (`scalefac_compress
= 0`), no CRC, no Xing/Info VBR tag, no ID3 frontmatter. Verified
end-to-end by `tests/stream_encoder_roundtrip.rs`: a one-second
440 Hz mono sine tone at 128 kbit/s / 44.1 kHz, encoded into the
on-wire byte sequence and re-decoded through the crate's own pipeline
(`Mp3Demuxer` + `decode_scalefactors` + `decode_huffman` + `requantize`
+ `alias_reduce` + `imdct_granule` + `synth_granule`), achieves
**PSNR > 80 dB** against the input (group delay 1057 samples = the
filterbank 481-sample prototype delay + 576-sample lapped-MDCT 1-granule
overlap). Stereo / LSF / VBR / Xing tag / outer-loop noise shaping /
psychoacoustic model are deferred.

**Phase 2 step 11 (outer (distortion-control) iteration loop)** wraps
the inner-loop global-gain search of step 5 in the §C.1.5.4.3 noise
shaping iteration (`outer_loop` module, `outer_loop_search_long`). Per
ISO/IEC 11172-3:1993 Annex C Figure C.9.b the loop: (a) runs the inner
loop to pick the smallest `global_gain` whose Huffman count fits the
budget AND keeps `max|is| ≤ 8191`, (b) computes the per-band
§C.1.5.4.3.3 colored-domain distortion `xfsf(sb)` against the decoder's
reconstruction, (c) amplifies every band with `xfsf(sb) > xmin(sb)` by
`scalefac_l[sb] += 1`, and (d) terminates on the §C.1.5.4.3.6 conditions
(no band over threshold / every band already amplified / next
amplification would exceed the per-band cap — 15 for `sfb ∈ [0,10]`,
7 for `[11,20]`), restoring the last-good state. The stream encoder
gains `Mp3Encoder::new_with_outer_loop(bitrate, sample_rate, mode,
uniform_threshold)` to route every per-granule-channel quantization
through this loop with `scalefac_compress = 15` (slen1=4, slen2=3) so
the chosen per-band scalefactors fit in part2. Threshold derivation is
the psychoacoustic model's job (Annex D, deferred); this round uses a
caller-supplied uniform constant. Validated by
`tests/outer_loop_roundtrip.rs`: a 6-tone multi-tone fixture at
128 kbit/s mono / 44.1 kHz produces **strictly-higher self-decode PSNR
than the fixed-gain path at the same bitrate** (typical +0.28 dB,
fixed-gain 73.7 dB → outer-loop 74.0 dB); the single-tone sine
roundtrips at ~86 dB matching the fixed-gain baseline. The fixed-gain
`Mp3Encoder::new` path is preserved as the debug / reference route.

As of round 147 the loop also implements §C.1.5.4.3's
**`scalefac_scale 0 → 1` escalation**. When a §C.1.5.4.3.5
amplification step would push a band's `scalefac_l[sb]` past the
§C.1.5.4.3.6 cap and the loop is still in `scalefac_scale = 0` mode,
the loop now switches to `scalefac_scale = 1` (multiplier 1.0 instead
of 0.5 — twice the per-step boost), halves every in-progress per-band
scalefactor with round-to-nearest integer arithmetic so the colouring
factor `2^(mult·sf)` is preserved across the switch, resets the
`amplified[]` first-touch tracker, and resumes the loop. Each
subsequent amp step is then worth 2× as much energy boost. The
escalation fires at most once per granule-channel (the spec defines
only two `scalefac_scale` values); the chosen flag is reported on
`OuterLoopResult::scalefac_scale` and propagated by the stream encoder
into the side-info `scalefac_scale` bit so the decoder's matching
multiplier recovers a coherent reconstruction. Validated by the
unit-test
`outer_loop::tests::outer_loop_escalates_scalefac_scale_when_cap_would_terminate`
(isolated-sfb fixture, threshold calibrated so the loop's only
termination path is cap-would-exceed ⇒ `res.scalefac_scale == true`)
and by the integration test
`outer_loop_tight_threshold_emits_valid_stream` (1.0e-30 uniform
threshold ⇒ encoder still emits a parseable stream that self-decodes
at finite PSNR). The pre-existing multi-tone PSNR regression test
passes unchanged — for that fixture the cap is never tripped, so the
escalation is a no-op there.

The `codec_encoder` module ships **Phase 2 step 12** — the
runtime-context `oxideav_core::Encoder` trait wiring on top of
`Mp3Encoder`. `Mp3CoreEncoder` is a frame-to-packet adaptor:
`send_frame` accepts mono or independent-stereo S16 PCM (`channels =
1` or `2`; the stereo path was widened in Phase 2 step 16),
`flush()` runs the bit-reservoir schedule, and `receive_packet`
drains one MP3 frame at a time (PTS and duration stamped in
`1 / sample_rate` units). Two direct factories —
`codec_encoder::make_encoder` (fixed-gain) and
`codec_encoder::make_encoder_with_outer_loop` (distortion-control
loop) — match the `oxideav-core` `EncoderFactory` signature.
Validated by `tests/encoder_trait_roundtrip.rs`: the registered
encoder, driven through the trait API only, round-trips a 1 s 440 Hz
sine at **86.17 dB PSNR** — matching the direct-API baseline,
confirming the adapter introduces no PSNR loss.

The `codec_decoder` module ships the **symmetric** decoder-side
trait wiring. `Mp3CoreDecoder` implements `oxideav_core::Decoder`:
`send_packet` parses one inbound MP3 frame (header + optional CRC +
side-info + main-data slot), runs the per-granule
`decode_huffman` → `requantize` → `alias_reduce` → `imdct_granule` →
`synth_granule` chain, and queues an `AudioFrame` of interleaved S16
PCM (1152 samples/channel for MPEG-1 Layer III). Per-stream state —
the §2.4.2.7 bit reservoir, the §2.4.3.4.10.4 IMDCT overlap memory,
and the §2.4.3.2 polyphase synthesis filterbank shift register — is
carried across packets; `reset()` wipes all three for post-seek
recovery. `codec_decoder::make_decoder` is the direct-API factory
matching `oxideav-core`'s `DecoderFactory` signature.
`crate::register` now installs the container demuxer **and** both
codec factories on a single `CodecInfo` (codec id `"mp3"`, WAVE tag
`0x0055`, Matroska `A_MPEG/L3`). Validated by
`tests/decoder_trait_roundtrip.rs`: a 500 ms sine encoded → sliced
into per-frame packets → driven through the trait Decoder produces
i16 PCM **byte-exact identical** to the direct-chain output on the
same input bytes (sample-for-sample match), and 250 ms of sine yields
the expected count of `AudioFrame`s with 1152 samples/channel and
monotonic PTS.

The `xing_info` module ships **Phase 2 step 13** — the encode-side
inverse of the demuxer's existing `parse_xing_info`. `XingTagSpec`
specifies a Xing / Info VBR information-frame payload (magic + flag
word + up to four optional fields: `frames`, `bytes`, `toc[100]`,
`quality`); `build_xing_info_payload` writes the byte run that goes
immediately after the side-info bytes of an MPEG audio frame; and
`build_info_frame` bakes the payload into a complete on-wire CBR
carrier frame (a silent Layer III frame whose main-data slot starts
with the Xing/Info magic and trails zero-fill out to
`Mp3FrameHeader::frame_len`). The carrier is a structurally-valid
Layer III frame — `part2_3_length == 0` and `big_values == 0` on
every granule-channel — so decoders that ignore the tag still see a
valid silent leading frame and reconstruct silence from it. Layout
verified against the on-disk `docs/audio/mp3/fixtures/layer3-with-xing-vbri-tag/`
+ `layer3-with-id3v2-tag/` fixtures + `trace.txt` and the symmetric
`parse_xing_info` reader.

The `Mp3Encoder` ships an opt-in `enable_xing_info(template)` method
that prepends the carrier frame to its `finish` output. The template
carries the magic + flag word + any pre-known fields; `finish` fills
in `frames` / `bytes` from the post-encode totals when those flag
bits are set and the template field is `None`. The carrier itself is
not counted in either total — both refer to the audio region that
follows, matching the demuxer's first-frame-skip behaviour. Validated
end-to-end by `tests/xing_info_roundtrip.rs` (7 tests): the carrier
is the first frame, it carries the right magic at the expected
offset, `parse_xing_info` recovers the writer's intent field-for-field
(both `Xing` and `Info` magic), the encoder's `bytes` accounting
agrees with `FrameWalker` re-counting the audio region, pre-filled
template fields are written verbatim, and `Mp3Demuxer::open` reports
the same Xing tag on the in-memory stream.

**Phase 2 step 14 (true-VBR per-frame bitrate selection + Xing TOC
auto-fill)** extends the r142 Xing emission with a content-driven
`bitrate_index` per audio frame. `Mp3Encoder::enable_vbr(min_kbps,
max_kbps)` activates the path: every frame's quantization runs the
magnitude-clamp inner-loop gain alone (no bit-budget chase against a
fixed target — that would saturate the constructor slot regardless of
content), and `finish` then picks the smallest §2.4.2.3 ladder index
in `[min_kbps, max_kbps]` whose slot can hold the assembled main-data
(plus one optional padding byte). Frames overflowing the max-index
slot surface `StreamEncodeError::VbrSlotTooSmall { frame_index, … }`;
`enable_vbr` itself rejects off-ladder values, reversed windows, and
maxima above the constructor bitrate (`InvalidVbrConfig`). When the
caller also `enable_xing_info`'s a template with
`flag_bit::TOC` set and `toc: None`, the writer fills the 100-entry
seek table from the post-encode per-frame cumulative byte offsets
(`toc[i] = floor(256 · audio_offset_for_percentile(i) / total_bytes)`,
clamped to `255`) — so `Mp3Demuxer`'s `seek_to` path resolves
percentile lookups on a real VBR stream the same way it does on
fixtures. Validated end-to-end by `tests/vbr_roundtrip.rs` (13
tests): all four `enable_vbr` reject paths, silence-stream landing on
the min-index, mixed-content emitting ≥2 distinct bitrates,
`FrameWalker` consuming the varying-length stream cleanly, the
demuxer's `next_packet` loop draining every audio frame without
error, auto-filled TOC monotone non-decreasing with `toc[0] == 0`
and tail ≥ 200, BYTES field matching the walker's audio-region byte
sum, and the degenerate `[K, K]` window emitting structurally
identical frames to the CBR-at-K path. `MPEG1_L3_BITRATE_LADDER_KBPS`
is re-exported for callers that need to enumerate the 14 selectable
ladder values.

**Phase 2 step 15 (opt-in §2.4.3.1 CRC-16 frame protection)** wires
the ISO/IEC 11172-3 §2.4.3.1 / Annex B Table B.5 CRC-16 mechanism into
the stream encoder. `Mp3Encoder::with_protection_bit(true)` flips a
crate-wide toggle: every emitted audio frame thereafter sets the wire
`protection_bit = 0` and carries the 16-bit CRC check word in the
two-byte slot between the 4-byte header and the side-information
block. The CRC slot is *inside* the §2.4.2.3 frame_len (per-frame and
total stream byte counts are unchanged); the 2 bytes the CRC claims
come out of main-data slot capacity, so the per-granule inner-loop
bit budget shrinks by 16 bits. The CRC itself covers exactly the
Annex B Table B.5 Layer III protected set: header bits 16…31 (bytes
2..4) plus the first 135 bits of side-info in single-channel mode (or
the first 256 bits in every other channel mode), MSB-first per the
§2.4.3.1 / Figure A.9 shift-register procedure (`G(X) = X^16 + X^15 +
X^2 + 1`, initial state `0xFFFF`). The new `crc` module exposes the
primitive (`crc16_bits` over a raw bit sequence; `crc16_layer3` over
the Annex B Table B.5 set) plus the spec constants. The Xing / Info
carrier frame stays CRC-free regardless of the toggle so leading
demuxer / seeker probes still see the standard Xing layout. The
crate's existing `Mp3CoreDecoder` path already skips the 2-byte CRC
slot per `Mp3FrameHeader::crc_protected`, so CRC-enabled streams
round-trip the decoder transparently. Validated by `crc_roundtrip.rs`
(3 integration tests) + 6 unit tests on the CRC primitive + 4 unit
tests on the encoder toggle.

**Phase 2 step 16 (stereo through the encoder trait)** widens
`Mp3Encoder::new` to accept `ChannelMode::Stereo` and
`ChannelMode::DualChannel` and routes the existing per-channel
filterbank + MDCT + quantizer + Huffman pipeline through `nch == 2`
unchanged. The two channels are encoded **independently**: header
`mode = '00'` (or `'10'` for dual-channel), `mode_extension = '00'`
(neither MS nor intensity), 32-byte MPEG-1 side-info block, and the
§2.4.1.7 `main_data()` loop walks `for (gr=0..2) for (ch=0..nch)`
emitting each granule-channel's part2 + part3 independently. The
upstream `pending_pcm` field becomes a `Vec<Vec<f32>>` (one PCM
buffer per channel); `push_samples` deinterleaves the caller's
`[L0, R0, L1, R1, …]` interleaved S16 input into the per-channel
buffers and assembles a frame as soon as both channels carry a full
1152-sample granule pair. The same widening lands in the
`oxideav_core::Encoder` trait wrapper: `codec_encoder::make_encoder`
now maps `params.channels = 2` → `ChannelMode::Stereo`, and the
`Mp3CoreEncoder::frame_to_i16` adapter validates `samples × channels
× 2` bytes per `AudioFrame`. `ChannelMode::JointStereo` remains
rejected (MS / intensity coupling on the encode side requires an
encoder-side stereo analysis stage — out of scope this round; the
decoder's `process_stereo` primitive already handles both methods
for incoming joint-stereo bitstreams). Validated by
`tests/stereo_encoder_roundtrip.rs`: a 1 s 440 Hz / 880 Hz LR sine
encoded at 192 kbit/s and round-tripped through the in-tree decoder
produces **per-channel PSNR L = 85.1 dB / R = 80.4 dB** (the L
channel matches the mono ~86 dB baseline; the R channel sits within
a few dB of it under the same fixed-gain inner-loop budget). A
single-bin DFT probe confirms zero cross-channel leakage (L energy
at 440 Hz is ~1250× the L energy at 880 Hz, and symmetrically for R)
— the two channels round-trip as fully independent signals through
the encoder.

**Phase 2 step 17 (joint-stereo MS encode)** adds an opt-in MS-stereo
constructor `Mp3Encoder::new_joint_stereo_ms(bitrate, sample_rate)`.
After per-channel analysis filterbank + forward MDCT + inverse alias
reduction (the existing independent-stereo pipeline of step 16), and
**before** the inner-loop quantization, each granule's `(L, R)` xr
pair is rewritten in place into the normalized mid/side pair per
ISO/IEC 11172-3:1993 §2.4.3.4.9.2:

```text
M[i] = (L[i] + R[i]) / √2
S[i] = (L[i] - R[i]) / √2
```

`M` is then quantized into the channel-0 slot, `S` into the channel-1
slot, and the emitted header carries `mode = '01'` (joint stereo)
with `mode_extension = '10'` (ms_stereo on, intensity_stereo off) per
§2.4.2.3. A conformant decoder — including this crate's own
`process_stereo` driven by the same `mode_extension` bits — applies
the inverse `L = (M+S)/√2`, `R = (M-S)/√2` to recover the L/R pair.
The matrix is its own inverse (a 2-D rotation by 45°), so in the
absence of quantization error the round-trip is identity. The decode
pipeline order is `requantize → process_stereo → alias_reduce →
imdct`, so the encoder applies the forward MS transform at the
matching point (between `inverse_alias_reduce` and the quantize
loop). MS is applied to the **entire** spectrum (§2.4.3.4.9.2: "When
MS-stereo is enabled but intensity stereo is not, the entire
spectrum is decoded in MS-stereo"); intensity-stereo encode
(§2.4.3.4.9.3) remains deferred. Both granules share the same Long
block type (the only type this encoder emits this round), satisfying
the §2.4.3.4.9 "both channels of a granule must share the same block
type when MS is enabled" requirement automatically. The trait
wrapper gains `codec_encoder::make_encoder_joint_stereo_ms(params)`
that builds the same `Mp3CoreEncoder` adapter against the
joint-stereo constructor. Validated by
`tests/joint_stereo_ms_roundtrip.rs`: 4 integration tests covering
(a) end-to-end self-decode on a 1 s 440 Hz tone panned 70/30 toward
L at 192 kbit/s — **per-channel PSNR L = 84.2 dB / R = 85.2 dB**;
(b) silence round-trip with the joint header bits preserved;
(c) first-frame wire-byte layout (mode bits `'01'`, mode_ext bits
`'10'`); (d) pan preservation through the MS round-trip (a 90/10
input recovers `|L|` strictly more than 3× `|R|`) — plus 2 unit
tests on the trait factory (mode bits + mono rejection).

**Phase 2 step 19 (preemphasis encode)** wires the §C.1.5.4.3.4
preemphasis lever the outer loop has carried as a placeholder field
since step 11 into the actual noise-shaping decision. After the first
inner-loop call each granule-channel checks the spec's explicit worked
example for switching on preemphasis — "if in all of the upper 4
scalefactor bands the actual distortion exceeds the threshold after the
first call of the inner loop". When that condition holds, `sf.preflag`
is set to `true` and the rest of the outer loop runs against the
inflated effective per-band scalefactor `scalefac_l[sfb] + pretab[sfb]`
(Table B.6). The pretab boost is **free** (one transmitted bit; no
`part2_3_length` impact) and only affects the upper bands (`pretab[sfb]
∈ {0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,2,2,3,3,3,2}`), so it leaves the
§C.1.5.4.3.6 cap (15 / 7 on the transmitted `scalefac_l[sfb]`)
untouched while giving the encoder a costless head-start on the
spectrum's high-frequency residue. Three changes land together: (a)
`OuterLoopResult` gains a `preflag: bool` field mirroring
`scalefactors.preflag`; (b) `band_distortion_long` adds `pretab[sfb]`
to its colouring exponent when `sf.preflag` is set so the §C.1.5.4.3.3
distortion is computed against the same reconstruction the decoder
will compute; (c) the stream encoder mirrors `sf.preflag` onto
`gc.preflag` before re-quantize, so the side-info bit reaches the
bitstream and the decoder re-applies the same pretab on the way out.
Validated by four unit tests (`outer_loop_default_preflag_off_when_*`
pin the negative arm — preflag does NOT fire with a giant threshold,
nor with only low bands carrying energy, nor when only three of the
upper four exceed threshold; `outer_loop_preflag_fires_when_all_upper_four_over_threshold`
pins the positive arm under a controlled spectral fixture) and by an
end-to-end integration test
`outer_loop_preflag_fires_on_hf_heavy_content` that confirms the
encoder produces a stream with `gc.preflag = 1` granule-channels
recoverable via `parse_side_info` on an HF-heavy multi-tone input.

**Phase 2 step 20 (joint-stereo auto MS/LR per-frame picker)** adds an
encoder-side per-frame mode-decision driver next to the round-146
unconditional MS path. The new constructor
`Mp3Encoder::new_joint_stereo_auto(bitrate_kbps, sample_rate_hz)` arms
joint mode (`header.mode = '01'`) and computes, for each assembled
frame, the side-channel energy fraction
`E_S / (E_L + E_R) = Σ(L−R)² / (2·Σ(L² + R²))` on the post-MDCT L/R
spectra of both granules; the §2.4.3.4.9.2 forward MS rotation fires
only when **both** granules sit at or below the configured threshold
(default `0.5`), and the frame's `mode_extension` field is written as
`'10'` (ms_stereo on) in that case or `'00'` (neither method) when at
least one granule exceeds the threshold. The per-granule rejection
short-circuit honours the §2.4.3.4.9 "both granules of a frame share
the same joint-stereo method" semantics for free, because
`mode_extension` is a per-frame wire field. ISO/IEC 11172-3 does
**not** prescribe an encoder mode-decision algorithm — §2.4.2.3 fixes
only the wire syntax of the `mode_extension` field — so the energy
heuristic is a clean-room encoder choice that uses no psychoacoustic
input. The `0.5` default is the symmetry boundary: the rotation is
unitary so `E_M + E_S = E_L + E_R`, and below `0.5` the mid channel
carries strictly more energy than either L or R, which the
inner-loop bit-budget gain search exploits.
`Mp3Encoder::with_ms_auto_threshold(t)` overrides the threshold
(values clamped into `[0.0, 1.0]`; the setter is a no-op when called
on an encoder that was not constructed via `new_joint_stereo_auto`).
`Mp3Encoder::ms_auto_threshold()` reads back the configured
threshold. The picker leaves the existing unconditional
`new_joint_stereo_ms` path untouched. Validated by six unit tests
(`auto_ms_picker_default_threshold_is_half` /
`auto_ms_picker_threshold_override_clamps` /
`auto_ms_picker_threshold_override_noop_on_non_auto` /
`auto_ms_picker_correlated_input_chooses_ms` /
`auto_ms_picker_anticorrelated_input_chooses_lr` /
`auto_ms_picker_zero_threshold_forces_lr_on_any_side_energy`) and
four integration tests in `tests/joint_stereo_auto_roundtrip.rs`:
(a) end-to-end self-decode on a 1 s correlated 440 Hz tone panned
70/30 toward L at 192 kbit/s — **per-channel PSNR L = 84.2 dB /
R = 85.2 dB**, matching the always-MS path, with every steady-state
frame carrying `mode_extension = '10'`; (b) anti-correlated input
(`R = -L`) where every steady-state frame must carry
`mode_extension = '00'`; (c) a mixed correlated-then-anti-phase
stream where the picker flips the wire `mode_extension` mid-stream,
proving the decision is genuinely per-frame and not encoder-wide;
(d) all-silence input handled without dividing by zero
(`E_L + E_R = 0` short-circuits to "MS by convention").

**Phase 2 step 21 (joint-stereo auto MS/LR through the trait factory)**
exposes the round-149 picker through the framework's
`oxideav_core::Encoder` factory shape, so trait-only consumers can
reach it without dropping down to the direct `Mp3Encoder` API. Two
new entry points on `codec_encoder` mirror the existing
`make_encoder_joint_stereo_ms`: `make_encoder_joint_stereo_auto(params)`
builds the auto picker with the default `0.5` threshold;
`make_encoder_joint_stereo_auto_with_threshold(params, t)` lets the
caller pick the side-channel energy threshold (clamped to
`[0.0, 1.0]` by `Mp3Encoder::with_ms_auto_threshold`). Both
validate `params.channels == 2` (joint stereo is two-channel by
definition), require a `sample_rate`, default `bit_rate` to 192 kbit/s
when absent, and produce the same `Mp3CoreEncoder` trait-object the
mono / independent-stereo factories produce — the per-frame picker
runs inside the wrapped `Mp3Encoder`, so the trait-side
`send_frame` / `flush` / `receive_packet` shape is unchanged. The
factory ergonomics keep the workspace's dual-API convention intact:
the direct `Mp3Encoder::new_joint_stereo_auto` constructor (round
149) and this trait factory (round 150) are equally first-class
entry points. Validated by six new unit tests in
`src/codec_encoder.rs`
(`make_encoder_joint_stereo_auto_emits_picked_mode_extension` proves
correlated stereo selects `mode_extension = '10'` on every
steady-state frame through the trait wrapper;
`make_encoder_joint_stereo_auto_with_threshold_threshold_zero_forces_lr`
proves `threshold = 0` suppresses MS on any non-trivial side energy;
`make_encoder_joint_stereo_auto_rejects_mono` /
`make_encoder_joint_stereo_auto_requires_sample_rate` /
`make_encoder_joint_stereo_auto_defaults_bitrate_to_192k` /
`make_encoder_joint_stereo_auto_with_threshold_clamps_out_of_range`
exercise the param-validation surface).

**Phase 2 step 22 (§2.4.3.4.10.2 forward short-block MDCT path)** wires
the encoder's per-subband forward MDCT to switch from the long-block
36-point transform to three independent 12-point transforms per
polyphase subband when the granule's chosen `block_type` is `Short`.
The new `short_block` module exposes `forward_short_mdct_subband` (the
three short MDCTs + the subband-window-interleaved layout the
decoder's `imdct.rs::windowed_block` consumes; output divided by the
Princen-Bradley `n/4 = 3` scale for unit gain), `forward_reorder`
(bit-exact inverse of `crate::reorder::reorder`: subband-window-
interleaved → native bitstream `[sfb][win][k]`), and
`short_block_region_defaults` (the spec-default short-block region
sentinels). `Mp3Encoder::force_short_blocks_for_testing` is the
deterministic test handle for the encode-side primitive: with the
toggle on (mono only this round; multi-channel needs the §2.4.3.4.9
cross-channel block-type agreement wiring deferred to a follow-up),
every assembled granule emits a §2.4.2.7 short block — forward
analysis runs the three-window short MDCT per subband, no alias
reduction is applied (§2.4.3.4.10.1 scopes alias reduction to
`block_type != 2`), the forward reorder rewrites the bins into the
bitstream interleave, and the per-granule-channel side info carries
`window_switching_flag = 1`, `block_type = Short`,
`mixed_block_flag = 0`, `subblock_gain = [0; 3]`, with the §C.1.5.4.4.6
short-block region split (region 0 hardcoded to the first 36 lines,
region 1 to the rest of big_values, region 2 empty) honoured by the
inner loop's region-end + table-select pass. The signal-driven
attack-detection auto-decision heuristic + the
LONG → START → SHORT → STOP → LONG transition state machine required
for mixed long-and-short streams remain a follow-up round; this step
lands the bitstream-side primitive + the side-info wiring so that
follow-up only needs to add the decision layer on top. Validated by
five new integration tests in `tests/short_block_encoder_roundtrip.rs`
(toggle rejected on stereo encoder; long baseline carries only long
granules; force-short stream's side info matches the
window-switched-short skeleton; force-short stream is accepted by
`Mp3Demuxer::next_packet`; force-short stream decodes end-to-end to
finite, non-silent PCM with audible zero crossings) and four new
unit tests in `src/short_block.rs` (forward-reorder ↔ decoder-reorder
roundtrip at 44.1 kHz pure-short, long-block identity pass-through,
mixed-block long-region preservation, per-subband forward-MDCT chain
energy bound).

**Phase 2 step 23 (§2.4.2.7 forward mixed-block MDCT path)** layers a
mixed-block dispatch on top of the step-22 forward short MDCT, so the
encoder can emit a granule whose lowest two polyphase subbands (lines
0..36) are coded with the long-family window + one 36-point MDCT
each, and whose upper 30 subbands (lines 36..576) are coded with the
three-window short MDCT. `Mp3Encoder::force_mixed_blocks_for_testing`
is the deterministic test handle for this primitive: with the toggle
on (mono only this round, mutually exclusive with the step-22
force-short toggle), every assembled granule emits a §2.4.2.7 mixed
block — the long-block forward path (`forward_overlap →
window_long_family_analysis(Long) → 36-pt mdct → ÷9`, identical to
the long-only baseline) runs on subbands 0 and 1, the short-block
forward path (`forward_short_mdct_subband`) runs on subbands 2..31,
and `forward_reorder` is then invoked with a mixed `GranuleChannel`
so the long region (lines 0..36) passes through unchanged while the
short region's SFB 3..12 is rewritten into native bitstream
`[sfb][win][k]` order. No inverse alias reduction is applied — the
decoder's `alias_reduce` tests `block_type == Short` and returns
unchanged for both pure-short and mixed granules. The
per-granule-channel side info carries `window_switching_flag = 1`,
`block_type = Short`, `mixed_block_flag = 1`, with the §C.1.5.4.4.6
short-family region split (region 0 hardcoded to the first 36 lines,
region 1 to the rest of big_values, region 2 empty) — the same split
the pure-short path uses, which here happens to align exactly with
the long / short subband boundary. Validated by six new integration
tests in `tests/mixed_block_encoder_roundtrip.rs` (toggle rejected on
stereo; force-short ↔ force-mixed mutual exclusion in both
directions; default off; force-mixed stream's side info matches the
window-switched-mixed skeleton; force-mixed stream accepted by
`Mp3Demuxer::next_packet`; force-mixed stream decodes end-to-end to
finite, non-silent PCM with audible zero crossings). The
signal-driven attack-detection heuristic that picks among Long /
Start / Short / Stop / Mixed per-granule from signal energy remains
a follow-up round; the present step lands the bitstream-side
primitive + the dispatch wiring so that follow-up only needs to add
the decision layer on top.

The remaining Phase 2 work — a real per-band psychoacoustic threshold
(so the loop can spectrally redistribute bits without a hand-tuned
constant), intensity-stereo encode (§2.4.3.4.9.3), signal-driven
**auto** block-type decision (the §C.1.5 attack-detection heuristic +
the LONG → START → SHORT → STOP → LONG transition state machine; the
forward short-block and mixed-block MDCT paths themselves are
available behind the force-short / force-mixed toggles as of rounds
151 / 152), LSF encode, and stereo / LSF decode through the trait
wrapper — is still a later round.

### Not yet implemented

Stereo / MPEG-2 LSF decode through the `Decoder` trait wrapper (the
underlying primitives — `process_stereo` and the LSF side-info /
scalefactor paths — are present; the wrapper is mono MPEG-1 only this
round). The encoder is **Phase 1 framing + Phase 2 steps 1–22 (forward
MDCT primitive + analysis windowing + forward overlap split +
polyphase analysis subband filterbank + §2.4.3.4.7 quantization
primitive + §C.1.5.4.4 inner-loop `global_gain` search + exact
§C.1.5.4.4.5/.8 Huffman bit count + §2.4.1.7 Huffman bit emission +
§2.4.1.7 main-data assembly + §2.4.2.7 cross-frame bit-reservoir
scheduling with `main_data_begin > 0` + stream-level PCM → MP3 driver
+ §C.1.5.4.3 outer (distortion-control) loop + `oxideav_core::Encoder`
trait wiring + opt-in Xing / Info VBR information-frame emission +
true-VBR per-frame bitrate + Xing TOC auto-fill + opt-in §2.4.3.1
CRC-16 frame protection + independent-stereo (`ChannelMode::Stereo` /
`ChannelMode::DualChannel`) encode through the trait wrapper +
§2.4.3.4.9.2 joint-stereo MS encode + §C.1.5.4.3 `scalefac_scale 0→1`
escalation in the outer loop + §C.1.5.4.3.4 preemphasis decision in
the outer loop + §2.4.2.3 joint-stereo auto MS/LR per-frame picker +
trait-factory wrappers for the auto MS/LR picker + §2.4.3.4.10.2
forward short-block MDCT path with `Mp3Encoder::force_short_blocks_for_testing`
toggle)** — it still lacks the psychoacoustic model (so the outer
loop's `xmin(sb)` is a uniform constant rather than per-band
masking-aware), intensity-stereo encode (§2.4.3.4.9.3), signal-driven
**auto** block-type decision (the §C.1.5 attack-detection heuristic +
the LONG → START → SHORT → STOP → LONG transition state machine for
*mixed* long-and-short streams; the forward short-block MDCT path
itself is available behind the force-short toggle), mixed-block encode
(the short forward path is pure-short only this round), and LSF
encode.

**Spec gap (alias reduction, mixed blocks):** §2.4.3.4.10.1 scopes the
stage purely on `block_type` ("block-type != 2" applies; "block-type ==
2 (short block)" does not). A *mixed* block is `block_type == 2` but
codes its two lowest subbands long; the standard gives no separate rule
for that long region, so this crate follows the literal text and does
not alias-reduce mixed blocks. A clarifying note in §2.4.3.4.10.1 on the
mixed-block long region would remove the ambiguity.

## License

MIT — see [LICENSE](./LICENSE).
