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

The remaining Phase 2 work — a real per-band psychoacoustic threshold
(so the loop can spectrally redistribute bits without a hand-tuned
constant), preemphasis (§C.1.5.4.3.4) and `scalefac_scale = 1`
escalation, short / mixed block-type switching, stereo / LSF, and the
runtime-context `Encoder::encode` trait wiring on top of `Mp3Encoder` —
is still a later round.

### Not yet implemented

No frame-driver / `Decoder` plumbing yet (the granule-level chain is
complete; what's missing is the per-frame iteration that consumes
[`FrameWalker`] frames, parses header + side-info + scalefactors,
Huffman-decodes both granules per channel, runs the full pipeline, and
emits a contiguous PCM buffer to the runtime context). The encoder is
**Phase 1 framing + Phase 2 steps 1–11 (forward MDCT primitive +
analysis windowing + forward overlap split + polyphase analysis
subband filterbank + §2.4.3.4.7 quantization primitive + §C.1.5.4.4
inner-loop `global_gain` search + exact §C.1.5.4.4.5/.8 Huffman bit
count + §2.4.1.7 Huffman bit emission + §2.4.1.7 main-data assembly +
§2.4.2.7 cross-frame bit-reservoir scheduling with
`main_data_begin > 0` + stream-level PCM → MP3 driver + §C.1.5.4.3
outer (distortion-control) loop)** — it still lacks the psychoacoustic
model (so the outer loop's `xmin(sb)` is a uniform constant rather than
per-band masking-aware), preemphasis / `scalefac_scale = 1` escalation,
short / mixed block-type switching, stereo / LSF, and the stream-level
`Encoder::encode` driver that would tie the scheduler into a CBR file
emitter. `register()` installs the container demuxer; the codec
`Decoder` / `Encoder` trait surfaces remain stubs.

**Spec gap (alias reduction, mixed blocks):** §2.4.3.4.10.1 scopes the
stage purely on `block_type` ("block-type != 2" applies; "block-type ==
2 (short block)" does not). A *mixed* block is `block_type == 2` but
codes its two lowest subbands long; the standard gives no separate rule
for that long region, so this crate follows the literal text and does
not alias-reduce mixed blocks. A clarifying note in §2.4.3.4.10.1 on the
mixed-block long region would remove the ambiguity.

## License

MIT — see [LICENSE](./LICENSE).
