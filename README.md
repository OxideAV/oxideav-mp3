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
- Band→scalefactor mapping uses the ISO/IEC 11172-3 Table B.8 long- and
  short-block start indices for 32 / 44.1 / 48 kHz and (as of r285) the
  ISO/IEC 13818-3:1997 Table B.2 indices for the LSF rates 16 / 22.05 /
  24 kHz (long + short; 16 and 22.05 kHz share one long layout per the
  spec). The MPEG-2.5 rates 8 / 11.025 / 12 kHz keep the half-rate
  MPEG-1 layouts as a self-consistent placeholder — their band tables
  are a documented residual gap (`docs/audio/mp3/MPEG-2.5-GAP.md`).
  These tables are the crate's single transcription: the Huffman
  region-boundary derivation and the encoder's region-split /
  intensity band walks delegate here. 19 unit tests
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
  starts at the next audio frame. The Xing / Info wire layout is
  now staged in `docs/audio/mp3/lame-xing-info-tag.md` (a clean-room
  transcription of Gabriel Bouvigne's independently-published
  *Mp3 Info Tag revision 1 Specifications* — independent format
  documentation, not LAME source — staged 2026-05-29 with sha256
  provenance). The previous "verified against the on-disk fixtures"
  qualifier is preserved as cross-validation.
- **LAME-extension gapless playback** (r185). The `lame_tag` module
  parses the LAME extension that follows the four Xing fields at the
  documented magic-relative offset (`$9A` from the Xing magic on the
  staged doc's all-flags worked example, byte-aligned to the four
  side-info layouts). The `LameTag` struct surfaces all 17
  LAME-defined fields (encoder version, info-tag revision, VBR
  method, lowpass, Replay-Gain peak / radio / audiophile, encoding
  flags + ATH type, bitrate, **encoder delay**, **zero padding**,
  misc, mp3-gain, preset + surround, music length, music CRC, tag
  CRC). The gapless pair is the 3-byte run at magic-relative `$B1`
  packed `[xxxxxxxx][xxxxyyyy][yyyyyyyy]` → 12-bit encoder delay +
  12-bit zero padding (each 0..=4095). `Mp3Demuxer::open` calls the
  parser only when all four Xing flag bits are set (the staged doc
  covers no other layout) and the encoder string is `"LAME"`; for
  other emitters (`"Lavc"`, `"Lavf"`, …) the parser refuses and
  `lame()` returns `None`. The demuxer surfaces the trim through
  `encoder_delay_samples()` / `zero_padding_samples()` and
  `trimmed_duration_samples()` (= gross − delay − padding for
  LAME-tagged streams, = gross duration otherwise). The on-wire
  3-byte pack/unpack is exhaustively round-tripped across the
  12-bit boundary corners (`0, 1, 2047, 2048, 4094, 4095` ×
  `delay × padding`), the §5 worked-example byte pattern
  `[0x6C, 0x12, 0xD2] → delay = 1729, padding = 722` propagates
  byte-for-byte from raw bytes through `parse_lame_tag` and again
  through the carrier-frame-driven `Mp3Demuxer::open` path, and a
  zero-delay zero-padding LAME tag yields `trimmed = gross`
  (no spurious trim) per the explicit
  `has_gapless_trim` predicate.
  **Tag-CRC verification** is deferred — the staged doc names
  `CRCInitValue = 0x0000` but does not specify the polynomial; the
  parser records the on-wire CRC for caller inspection without
  validating it. **Spec gap:** the staged doc's `$9A–$A4 | 9 bytes`
  cell is internally inconsistent (`$9A..=$A4` is 11 bytes
  inclusive); the parser trusts the absolute-offset chain
  (`$A5, $A6, …, $BF`) over the "9 bytes" annotation, leaving the
  two bytes at `$A3–$A4` as reserved padding. Filling that
  inconsistency in is a follow-up `docs/` patch.
- **Duration estimation.** VBR streams with a Xing `frames` field
  report `frames × samples_per_frame / sample_rate`; CBR streams
  use `audio_bytes × 8 / bitrate × sample_rate`. The four-fixture
  reference (CBR-320, VBR-q5, ID3v2-tagged, Xing-tagged) all report
  ~835.9 ms vs ffprobe's 800.0 ms (`Δ = +4.5%`); the residual is the
  LAME encoder-delay/padding overhead that **`trimmed_duration_samples()`
  now removes for LAME-tagged streams** — for those streams the
  trimmed value should converge on the reference duration once the
  on-disk fixture's LAME tag carries non-zero delay/padding values
  (none of the in-tree fixtures currently do; the existing
  `Xing`-tagged fixture's `Lavc61.19` carrier is not LAME-emitted so
  this code path is exercised via the synthetic unit tests).
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
`decode_huffman` → `requantize` → `reorder` (§2.4.3.4.8) →
`process_stereo` (joint granules only) → `alias_reduce` →
`imdct_granule` → `synth_granule` chain,
and queues an `AudioFrame` of planar S16 PCM (1152 samples/channel
for MPEG-1 Layer III; one plane per channel — `data[0]` = L,
`data[1]` = R for stereo, single plane for mono). Per-stream state —
the §2.4.2.7 bit reservoir and the **per-channel** §2.4.3.4.10.4
IMDCT overlap memory + §2.4.3.2 polyphase synthesis filterbank
shift register pair — is carried across packets; `reset()` wipes
all of it for post-seek recovery. `codec_decoder::make_decoder` is
the direct-API factory matching `oxideav-core`'s `DecoderFactory`
signature. `crate::register` now installs the container demuxer
**and** both codec factories on a single `CodecInfo` (codec id
`"mp3"`, WAVE tag `0x0055`, Matroska `A_MPEG/L3`). Mono was
validated in r141 by `tests/decoder_trait_roundtrip.rs`: a 500 ms
sine encoded → sliced into per-frame packets → driven through the
trait Decoder produces i16 PCM **byte-exact identical** to the
direct-chain output on the same input bytes (sample-for-sample
match), and 250 ms of sine yields the expected count of
`AudioFrame`s with 1152 samples/channel and monotonic PTS. Stereo
is validated in r177 by `tests/decoder_trait_stereo_roundtrip.rs`
(see "Phase 2 step 36" below).

**Round 297 (§C.1.5.4.4.6 band-aligned SUBDIVIDE).** §C.1.5.4.4.6 states
the inner loop's SUBDIVIDE "splits the **scalefactor bands** corresponding
to these values into three groups", and the side-info fields
`region0_count` / `region1_count` are band counts (`region0_count + 1`
bands in region 0, `region1_count + 1` in region 1). The decoder
reconstructs the region boundaries solely from the long-block band-start
table and those counts, so a boundary chosen mid-band is unrepresentable
on the wire. The default inner-loop estimate ([`inner_loop::subdivide`])
splits on raw big-values pairs (`big_pairs/3`, `big_pairs/4`), which can
land between band edges — so its bit count is for a partition the encoder
cannot emit. This round adds `subdivide_bands(sample_rate, version,
big_pairs)`, which snaps the same "~1/3 to region 0, ~1/4 to region 2"
strategy to scalefactor-band edges and returns valid 4-bit / 3-bit
`region0_count` / `region1_count` field values (the boundaries the
decoder's `region_boundaries` reproduces exactly), plus
`exact_bit_count_band_aligned`, which counts the §C.1.5.4.4.5 + .8 Huffman
total against those band-aligned long-family boundaries (short / mixed
blocks fall back to the two-subregion pair split — they carry the
§C.1.5.4.4.6 blocksplit `region1_count` defaults the decoder ignores).
Both are pure helpers; the default `exact_bit_count` / `subdivide` and the
inner-loop `global_gain` search are untouched, so every emitted byte is
byte-for-byte the historical default — the band-aligned estimate is opt-in
for callers that want the bit count to match the wire partition the
existing `choose_region_split` produces. Seven new lib tests
(empty-range, 44.1 kHz edge alignment, field-bounds / ordering across
32 / 44.1 / 48 kHz, decoder-round-trip of the chosen counts,
band-aligned-vs-default self-consistency on a long block, and
short-block fallback equivalence). Spec read: SUBDIVIDE text on PDF
page 104 (§C.1.5.4.4.6).

**Round 304 (§2.4.3.4.8 reorder restored to the trait decode path —
short-block decode fix).** The `Mp3CoreDecoder` trait wrapper
(`codec_decoder::decode_packet`) was running
`requantize → process_stereo → alias_reduce → imdct_granule` with **no
§2.4.3.4.8 `reorder` call**. For a long block that is harmless (long
blocks are already increasing-frequency-ordered and `reorder` is the
identity), but a short (`block_type == 2`) or mixed granule leaves
`requantize` in the native `(sfb, window, freqline)` Huffman interleave,
while `imdct_granule`'s short path gathers `lines[3·k + win]` (the
subband-window-interleaved layout) and `process_stereo`'s short
intensity/MS path indexes `base + 3·k + win` — both reading the wrong
samples, so every short-block (and mixed-block short-region) granule
decoded to corrupt PCM through the registered `Decoder` trait, mono and
stereo alike. r304 inserts `reorder` between `requantize` and
`process_stereo` (the spec order — §2.4.3.4.8 precedes the §2.4.3.4.9
stereo stage, whose own short-block path already assumes reordered
input), matching the proven `decode_mp3_mono_short_aware` helper in
`short_block_encoder_roundtrip`. New lib test
`trait_decode_short_block_runs_reorder_and_is_not_silent`: a force-short
mono stream (every granule `block_type == 2`, asserted on the wire)
decoded through the trait API is byte-exact against the in-module
`decode_direct` reference (now also reordering) **and** yields finite,
non-silent, zero-crossing PCM. The standing `decoder_trait_lsf_roundtrip`
reference helper was missing the same stage — its fixture carries short
blocks — so it is corrected in the same commit (the trait decoder is now
the spec-correct side). Spec read: ISO/IEC 11172-3 §2.4.3.4.8 (short
reorder), §2.4.3.4.9 (stereo runs after reorder), §2.4.3.4.10
(IMDCT consumes subband-ordered lines).

**Round 306 (§2.4.3.4.9.2 MS-*auto* picker over the per-window short
intensity region).** r305 lifted the short + intensity rejection on the
*unconditional* MS path but left the **auto-MS** path
(`new_joint_stereo_auto_is` + `force_short_blocks_for_testing(true)`)
rejected: the side-energy picker that decides MS-vs-LR per frame scored
its fraction `E_S/(E_L+E_R) = Σ(L−R)² / (2·Σ(L²+R²))` over the
*long-block* bound line range `0..ms_region_hi`, not the per-window short
region the rotation actually touches. r306 teaches the picker the short
layout: when short + intensity is armed it recomputes the per-granule
upper line as `0..3·short_starts[short_start]` (the same contiguous run
the r305 rotation applies, with the granule's per-window bound
`short_intensity_start_per_gr[gr]` from Pass 1.45), so the decision is
measured on exactly the lines MS would rotate. With the picker corrected,
the `IntensityShortBlocksUnsupported` rejection on
`force_short_blocks_for_testing` (the MS-auto clause) is dropped; the
long / no-intensity picker path is byte-for-byte unchanged (it still
reads the frame-constant `ms_region_hi`). Frames carry `mode = '01'` with
per-frame `mode_extension = '11'` (MS + intensity) when both granules'
short MS regions fall at/under the threshold and `'01'` (intensity only)
otherwise. Validated by three new `ms_short_intensity_roundtrip` tests:
acceptance + byte-deterministic encode with a valid `'01'`/`'11'` header
stream, the picker firing MS on low-side-energy short content (default
0.5 threshold), and the picker declining MS for any non-zero short side
energy at threshold 0 (which also proves the region read is non-empty —
a stale long/empty bound would score zero and wrongly pick MS). The
`intensity_rejects_block_type_toggles` unit test now asserts MS-auto +
short + intensity acceptance alongside the still-rejected mixed /
auto-block-type short paths. Spec read: ISO/IEC 11172-3 §2.4.3.4.8
(short reorder) / §2.4.3.4.9.2 (MS matrix + the encoder's free choice of
MS); ISO/IEC 13818-3 §2.4.3.2 (per-window bound).

**Round 305 (§2.4.3.4.9.2 MS + short-block + intensity stereo, per-window
bound).** r303 wired the per-window short intensity bound for the
intensity-*only* force-short path; the combination with the §2.4.3.4.9.2
MS matrix stayed rejected (`IntensityShortBlocksUnsupported`) because the
below-bound MS rotation has to follow the §2.4.3.4.8 interleaved short
layout rather than the single contiguous line range the long-block path
uses. r305 lifts that rejection on the **unconditional MS** path
(`new_joint_stereo_ms_is` + `force_short_blocks_for_testing(true)`). The
intensity coupling (Pass 1.45) is unchanged — it folds `L += R; R = 0`
over `short_start..12` for every window. The MS rotation (Pass 1.5) now
branches on the short-intensity case: instead of rotating
`0..ms_region_hi` it rotates the below-bound region, bands
`0..short_start` across all three windows. Because every window of those
bands is rotated and MS is a per-line operation, that line set is exactly
the contiguous run `0..3·short_starts[short_start]` (the reorder is a
permutation of it), so the rotation reduces to a contiguous loop while
remaining the exact inverse of the decoder's per-window `process_short`:
it MS-decodes the bands below each window's derived bound (the right
channel's side signal keeps that bound at `short_start`) and
intensity-decodes the rest (ISO/IEC 13818-3 §2.4.3.2). Intensity and MS
touch disjoint line sets, so applying intensity first then MS never
double-rotates a line. Frames carry `mode = '01'`, `mode_extension =
'11'`. (The MS-*auto* + short + intensity path was lifted the following
round — see Round 306.) Validated by a new integration suite
(`ms_short_intensity_roundtrip`): `'11'` header + pure-short
side info, right-channel positions in range, a spec-order self-decode in
which the below-bound 440 Hz MS pan reconstructs at its true 1.40 ratio
and the hard-left 8 kHz intensity tone reconstructs strongly
left-leaning, and a byte-deterministic encode. The
`intensity_rejects_block_type_toggles` unit test asserts MS + short
force-short acceptance. Spec read: ISO/IEC 11172-3 §2.4.3.4.8 (short reorder) /
§2.4.3.4.9.2 (MS matrix) / §2.4.3.4.9.3 (coupling); ISO/IEC 13818-3
§2.4.3.2 (per-window bound).

**Round 303 (§2.4.3.4.9.3 short-block intensity stereo, per-window
bound).** Every intensity constructor before this round was long-block
only: arming intensity (`new_joint_stereo_is`) rejected the force-short
toggle with `IntensityShortBlocksUnsupported`, because the short-block
bound is derived **per window** (ISO/IEC 13818-3 §2.4.3.2: "the
calculation of the intensity bound is applied to the values of each short
window") and the positions ride the right channel's `scalefac_s[sfb][win]`
slots rather than `scalefac_l`. r303 wires that path:
`force_short_blocks_for_testing(true)` is now accepted on an
intensity-only encoder. Pass 1.45 maps the public `intensity_start_sfb`
(a long-band index) onto a short start band by frequency, then walks the
12 short bands × 3 windows of every granule — deriving each window's
position from its own L/R band energies (Annex G.2 c) /
`derive_intensity_position[_lsf]`) and folding `L += R; R = 0` over that
window's lines in the native `[sfb][win][k]` interleave (coupling is
per-line and the §2.4.3.4.8 reorder is a permutation within the band, so
the operation is layout-invariant against the decoder's per-window
reconstruction). Pass 2 writes `scalefac_s[sfb][win]` with the derived
positions at/above the start band and the Annex G.2 c) illegal marker `7`
on each window's all-zero bands above its own last non-zero quantized
line, so the decoder's per-window zero-part bound lands exactly where
intended. The right channel takes `scalefac_compress = 15` (slen1 = 4,
slen2 = 3 ⇒ 126-bit short part2 — every position fits). Mixed and
auto-scheduled short granules, and the MS + short + intensity combination
(whose below-bound MS rotation still needs the interleaved short layout),
keep their rejection. Validated by a new integration suite
(`short_block_intensity_roundtrip`, 4 tests): intensity-only short header
+ pure-short side info, per-window positions in range, a hard-left 8 kHz
tone reconstructing left-leaning through a spec-order self-decode
(huffman → requantize → **reorder** → process_stereo → alias → imdct →
synth) with the below-bound 440 Hz pan preserved, and byte-deterministic
encode. The rejection unit test (`intensity_rejects_block_type_toggles`)
now asserts force-short acceptance + the narrowed rejections. Spec read:
ISO/IEC 11172-3 §2.4.3.4.8 (short reorder) / §2.4.3.4.9.3 (coupling) +
Annex G.2 c); ISO/IEC 13818-3 §2.4.3.2 (per-window bound).

**Round 302 (§2.4.3.4.9.3 adaptive per-granule intensity bound).** Every
joint-stereo intensity constructor so far fixes the coupling start band
at construction (`new_joint_stereo_is(_ms / _auto)` couple
`start_sfb..21` on *every* granule). r302 adds
`Mp3Encoder::new_joint_stereo_auto_is_adaptive(bitrate, sample_rate,
intensity_start_floor)`, which treats the start band as a **floor** and
picks the actual bound per granule from the post-MDCT spectrum: the
chooser ([`stream_encoder::choose_intensity_bound`]) walks bands from
the top down to the floor and couples only the **contiguous high tail**
whose every band carries little right-channel stereo information,
measured by the same side-energy fraction the §2.4.3.4.9.2 MS picker
uses — `E_S/(E_L+E_R) = Σ(L−R)² / (2·Σ(L²+R²)) ≤ threshold` (default
`0.25`, `with_intensity_auto_threshold` overrides, clamped to `[0,1]`).
A band that still carries real stereo content raises the bound so it —
and everything below it — stays independently coded; with no qualifying
tail the granule couples **nothing** (effective bound 21) and keeps a
full right channel. The bound is implicit on the wire (§2.4.3.4.9.1: the
decoder derives it from the right channel's last non-zero line), so the
per-granule bound varies with no syntax change; the header stays
`mode='01'` / `mode_extension='01'`. This also fixed a latent bug: the
pass-2 `intensity_right` flag was gated only on the global
`intensity_active`, so a granule that coupled nothing would still have
written its right channel as is_pos markers; it is now also gated on a
new per-granule `intensity_coupled_per_gr[gr]` (a coupled-nothing
granule writes an ordinary right channel and the top partial-region
coupling is skipped). Validated by two lib tests
(`intensity_auto_adaptive_constructor_state`,
`choose_intensity_bound_picks_low_stereo_tail` — floor honoured,
mid-band stereo raises the bound, top-band stereo couples nothing,
silent/below-floor cases) and a self-decode integration test
(`adaptive_intensity_bound_couples_low_stereo_high_tail`): a near-mono
6 kHz tail couples (right channel reconstructed balanced), an anti-phase
6 kHz tail stays independent (right channel keeps its real tone), the
below-bound 440 Hz pan survives both, and the encode is byte-deterministic.
Lib suite 1086 → 1088 (+2); intensity integration suite 8 → 9. The
heuristic is a clean-room encoder choice — ISO/IEC 11172-3 fixes only
the `mode_extension` syntax (§2.4.2.3), not how to pick the bound. Spec
read: ISO/IEC 11172-3 §2.4.3.4.9.1/.9.3 (intensity bound + coupling),
Annex G.2 c) (position derivation).

**Round 301 (§C.1.5.3 scfsi reuse auto-armed inside `push_samples`).**
r296 added `Mp3Encoder::enable_scfsi_reuse()` as an opt-in
post-quantization pass; this round flips it on by default. Because the
detection is byte-exact (a scfsi_band group is marked only when the two
granules' scalefactors already agree across every band in it) and the
decoder reconstructs granule 0's values for a marked group, auto-arming
is lossless by construction: the reconstructed PCM is identical to the
historical `scfsi = 0` output while granule 1's part2 budget shrinks
wherever consecutive granules naturally share scalefactors. A fresh
`Mp3Encoder` (every constructor funnels through `new`, so the auto-arm
covers single-channel / stereo / dual / all four joint-stereo
constructors / the outer-loop and threshold-in-quiet builders) now emits
scfsi automatically; the new `Mp3Encoder::disable_scfsi_reuse()` restores
the pre-r301 byte-for-byte `scfsi = 0` stream as a
compatibility / regression-bisection escape hatch, and
`enable_scfsi_reuse()` is retained to re-arm after an explicit disable.
The optimisation still never fires on LSF (MPEG-2 / MPEG-2.5 have one
granule and no scfsi field) nor on any channel whose either granule is a
short block (§2.4.2.7). One lib test renamed + extended
(`scfsi_reuse_auto_armed_by_default_disarmed_by_toggle`: default-on,
`disable` clears, `enable` re-arms); `tests/scfsi_reuse_roundtrip.rs`
gains `scfsi_auto_armed_by_default_sets_reuse_flags` (a default encoder
sets scfsi on a steady tone, never grows the stream vs. disarmed, and
decodes sample-for-sample identical) and routes its disarmed baselines
through `disable_scfsi_reuse()`. The intensity-stereo roundtrip decoder's
hand-rolled part2 skip is now scfsi-aware (its old "the only shape the
intensity encoder emits is no-scfsi" premise no longer holds once reuse
is auto-armed). Full suite green (1086 lib + integration); every
self-decode test still reconstructs bit-exactly. Spec read: §2.4.2.7 /
§C.1.5.3, ISO/IEC 11172-3 body PDF.

**Round 300 (§C.1.5.4.4 band-aligned bit-budget search wired into the
outer (distortion-control) loop).** r299 swapped the fixed-gain CBR path
to `search_bit_budget_band_aligned` but explicitly left the §C.1.5.4.3
outer-loop branches on the default `search_bit_budget` (the documented
followup). So the noise-shaping loop kept choosing `global_gain` against
the pair-thirds [`inner_loop::subdivide`] heuristic, whose region
boundaries can land mid-band — a part2_3 length the decoder's
`region_boundaries` cannot reconstruct and the encoder never emits — even
though the gain it returns is the one the outer loop carries into the
emitted granule. This round swaps **both** per-iteration inner-loop
helpers to the band-aligned search: `run_inner` (long / Start / End
blocks) and `run_inner_short` (short / mixed blocks). For long-family
blocks the gain is now gated on the §C.1.5.4.4.6 SUBDIVIDE snapped to
scalefactor-band edges ([`inner_loop::subdivide_bands`], region ends
clamped to the 4-bit / 3-bit `region0_count` / `region1_count` field
widths the decoder reproduces) — the same wire partition r299 made the
CBR fixed-gain path use, now matched on the distortion-control path too.
Short / mixed blocks share the two-subregion blocksplit path, so for
those the new gating is bit-identical to the old (the band-aligned count
falls back to the block-type-steered pair split there). The magnitude
clamp (`search_magnitude_clamp`), the §C.1.5.4.3 amplification / preflag /
scalefac_scale escalation, and the VBR clamp-only choice are all
untouched. One new lib test
(`outer_loop_long_gain_fits_band_aligned_wire_partition`): the chosen
`is[]` re-counted via `exact_bit_count_band_aligned` at the final gain
fits the per-granule budget and uses the band-aligned region ends. Lib
suite 1085 → 1086; the full encoder roundtrip + PSNR suite stays green
(every existing self-decode test reconstructs bit-exactly through the
changed path). Spec read: §C.1.5.4.3 outer loop + §C.1.5.4.4 / .4.6 inner
loop on PDF pages 100-104.

**Round 299 (§C.1.5.4.4 band-aligned bit-budget search wired into the CBR
encode path).** r298 added `search_bit_budget_band_aligned` as an opt-in
primitive but the fixed-gain CBR encode path still picked `global_gain`
with the default `search_bit_budget`. That default gates the gain on the
pair-thirds `subdivide` heuristic, whose region boundaries may land
mid-band — a part2_3 length the decoder's `region_boundaries` cannot
reconstruct — while the encoder downstream actually emits the
band-aligned `choose_region_split` partition. So the gain was being
measured against a partition the encoder never writes. This round swaps
the fixed-gain CBR branch to `search_bit_budget_band_aligned`, which
measures the §C.1.5.4.4.5 + .8 Huffman total against the same
scalefactor-band-edge SUBDIVIDE the encoder emits (§C.1.5.4.4.6, region
ends snapped to band edges and clamped to the 4-bit / 3-bit
`region0_count` / `region1_count` field widths). The chosen gain now fits
the real emitted part2_3 length rather than an unrepresentable
approximation of it. Short / mixed blocks share the two-subregion
blocksplit path, so they are unchanged; the VBR (clamp-only) gain choice
and the outer-loop branches are untouched. One new integration test
(`cbr_long_block_part2_3_within_per_granule_budget`,
`stream_encoder_roundtrip` 3 → 4) asserting every CBR long-block
granule's `part2_3_length` stays within the per-frame main-data budget;
the full encoder roundtrip + PSNR suite stays green (every existing
self-decode test reconstructs bit-exactly through the changed path). Spec
read: inner-iteration-loop §C.1.5.4.4 + §C.1.5.4.4.6 on PDF pages
103-104.

**Round 298 (§C.1.5.4.4 band-aligned bit-budget inner-loop search).**
r297 added the band-aligned SUBDIVIDE helpers
([`inner_loop::subdivide_bands`] / [`inner_loop::exact_bit_count_band_aligned`])
but left them as pure estimators — the actual `global_gain` rate-control
search ([`inner_loop::search_bit_budget`]) still gated on the default
pair-thirds heuristic ([`inner_loop::exact_bit_count`]), whose region
boundaries may land mid-band, i.e. on a partition the decoder's
`region_boundaries` cannot reconstruct and the encoder can therefore
never emit. This round wires the band-aligned estimate into a search:
`search_bit_budget_band_aligned` runs the spec's upward `qquant + 1`
scan (§C.1.5.4.4 "increases the quantizer step size until the output
vector can be coded with the available number of bits") gated on
`exact_bit_count_band_aligned`, so the smallest `global_gain` it returns
fits the part2_3 length the encoder will *actually* write — the
§C.1.5.4.4.6 SUBDIVIDE region ends snapped to scalefactor-band edges and
clamped to the 4-bit / 3-bit `region0_count` / `region1_count` field
widths (long-family blocks); short / mixed blocks share the
two-subregion blocksplit path, so for those the new search is
bit-identical to `search_bit_budget`. Like the existing budget search the
gating count is **not** monotone in the gain (Huffman codeword lengths
are not monotone in magnitude — §C.1.5.4.4.7), so the upward scan returns
the *smallest* fitting gain, never a bisection. The default
`search_bit_budget` is untouched, so every byte the existing encoder
emits is unchanged — the band-aligned search is an opt-in primitive for a
future bit-budget-driven encode path. Five new lib tests
(budget-0-to-silence, minimality of the chosen gain, tighter-budget ⇒
coarser-gain, short-block equivalence with the default search across a
budget sweep, and wire-partition gating on a long block); inner-loop
suite 34 → 39 tests. Spec read: inner-iteration-loop §C.1.5.4.4 +
§C.1.5.4.4.2 / .6 / .7 / .8 on PDF pages 103-105.

**Round 296 (§C.1.5.3 scalefactor-selection-information / scfsi
reuse).** MPEG-1 Layer III carries two granules per frame, each with
its own part2 scalefactor block; the §2.4.2.7 `scfsi[ch]` field lets a
frame send a long-block scfsi_band group's scalefactors **once** (in
granule 0) and declare them valid for granule 1 when the two granules
already agree there. The decode path has always honoured this on read;
the encoder previously emitted `scfsi = 0` on every frame. New
`Mp3Encoder::enable_scfsi_reuse()` arms a post-quantization pass that
sets `scfsi[ch][g] = 1` for each of the four scfsi_band groups (bands
`{0..=5}`, `{6..=10}`, `{11..=15}`, `{16..=20}`, Table 3-B.8) whose
granule-1 scalefactors are byte-identical to granule 0's across the
whole group — **only** when both granules of the channel are long
blocks. Per §2.4.2.7 ("if short windows are switched on … then scfsi is
always 0 for this frame"), a short granule disqualifies all four groups
for that channel. The §2.4.2.7 write path already skips a reused group
in granule 1 and the decoder reproduces granule 0's values verbatim, so
the optimisation is lossless by construction: every reconstructed sample
is bit-identical, only granule 1's part2 scalefactor bits shrink.
Default off (byte-for-byte the historical `scfsi = 0` output);
`scfsi_reuse_enabled()` inspects the flag; the toggle is a no-op on LSF
(MPEG-2 / MPEG-2.5 have one granule and no scfsi field). Five lib tests
(all-group reuse / no reuse / per-group independence /
short-granule-disqualification / default-off-armed-by-toggle) plus a new
`tests/scfsi_reuse_roundtrip.rs` (5 tests): disarmed sets no scfsi, the
armed fixed-gain encode sets reuse on long-block frames, armed-vs-
disarmed decode is bit-identical on both the fixed-gain and outer-loop
paths, and the armed outer-loop encode strictly shrinks granule-1's
summed part2_3 budget without growing the CBR stream. (Auto-arming
inside `push_samples` — the reuse default-on rather than opt-in — landed
in r301; see above.)

**Round 295 (Phase 2 step 92 — §C.1.5.3.2.1 Model-2-driven auto
block-type path).** r294 captured the per-granule Model 2 `pe > 1800`
window-switching decision; this round wires it into an actual
block-type driver. The new `Mp3Encoder::enable_auto_block_type_model2()`
arms a mode where the per-granule §C.1.5.2 attack flag fed into the
`LONG → START → SHORT → STOP → LONG` scheduler is the spec-canonical
Model 2 psychoacoustic-entropy decision (`pe > 1800`) instead of the
energy-detector subframe-energy ratio used by
`enable_auto_block_type`. The transition geometry (one
`BlockTypeStateMachine` per channel), the independent / MS-stereo
OR-fold coupling, and the lookahead-granule anticipation all mirror the
energy path — only the attack signal differs. The mode requires
`enable_model2_psychoacoustics` (it reuses the same per-channel Model 2
states): the analysis runs once in the block-type pre-pass and its full
`Model2Layer3Granule` output is **cached so Pass 1 reuses it for the
outer-loop `xmin(sb)`** — the §D.2.1 FFT history advances exactly once
per granule, never twice. The lookahead granule is peeked from a cloned
state so the borrowed next-frame PCM never perturbs the committed
history. Mutually exclusive with the energy-detector auto path and the
force toggles (arming any clears the others); disarming Model 2 (via
`set_per_band_xmin`) disarms this too. Inspect with
`auto_block_type_model2_enabled()`, turn off with
`disable_auto_block_type_model2()`; armed without Model 2 yields the new
`StreamEncodeError::Model2BlockTypeWithoutModel2`. Eight new lib tests
cover the API guards, steady-tone-stays-long, valid-§C.1.5.2-sequence
emission, per-band-`xmin`-still-installed, and two end-to-end
correspondence tests proving the emitted block types equal the §C.1.5.2
scheduler walk over the captured `pe > 1800` attacks (single-frame and
multi-frame, scheduler state carried across frame boundaries). **No
emitted bytes change** unless the new mode is explicitly armed.

**Round 292 (free-format decode through the `Decoder` trait).** The
trait wiring previously rejected any **free-format** frame
(`bitrate_index == 0`, ISO/IEC 11172-3 §2.4.2.3) outright, because
`Mp3FrameHeader::frame_len` returns `None` for such a header — the
§2.4.2.3 length formula needs a fixed `bitrate_index ∈ 1..=14`. A
free-format encoder instead picks a constant frame length the
bitstream itself never encodes, and the framing layer recovers it as
the distance from one syncword to the next. Since the trait contract
is that each `Packet.data` already holds exactly one complete MP3
frame, the authoritative free-format length is simply the packet
length: `decode_packet` now derives `frame_len` from `bytes.len()`
when the header yields none (rejecting only a bare 4-byte sync with no
main-data slot). The entire downstream decode (side-info → Huffman →
requantize → IMDCT → synthesis) is driven by `part2_3_length` from the
side-info and never by the advertised bitrate, so a free-format frame
decodes through the identical chain. Validated by
`tests/decoder_trait_free_format_roundtrip.rs` (4 tests): a CBR stream
is sliced into per-frame packets, each frame's `bitrate_index` is
rewritten to `0` (changing no other byte — the field lives in the high
nibble of header byte 2), and the resulting free-format stream decodes
**byte-exact identical** to the original CBR PCM, for both mono and
stereo; plus header-property and bare-sync-rejection checks. This
closes the long-standing decode-side free-format "lacks" item; the
encoder still emits only fixed-`bitrate_index` frames, and demuxer
free-format framing (deriving the constant length from the first
syncword interval) remains a follow-up.

**Round 294 (Phase 2 step 91 — captured Model 2 window-switching
decision per granule).** The automatic Model 2 mode (r293, step 90)
runs each granule's PCM through the channel's Model 2 state to derive
the per-band `xmin(sb)` threshold; the same §C.1.5.3.2.1 walk also
yields the psychoacoustic entropy `pe` and its `pe > 1800` short-block
switching condition — the §C.1.5.3.2 deliverable the spec defines for
window switching — which the encode loop previously threw away. The
Pass-1 walk now retains both into a per-(granule, channel) matrix
committed after each frame, exposed by
`Mp3Encoder::last_model2_window_switch(gr, ch) -> Option<Model2WindowSwitch>`
(`Model2WindowSwitch { pe, attack }`). It reflects exactly the last
frame assembled and returns `None` before any frame is encoded under
the armed mode, for out-of-range `gr`/`ch` (e.g. `gr == 1` on a
single-granule LSF frame), and once the mode is disarmed. This makes
the spec-canonical switching signal observable and lays the foundation
for a future Model-2-driven auto-block-type path; **no emitted bytes
change** in any current configuration (block type is still governed by
the attack-detector auto path or the force toggles). 4 new unit tests:
populated / finite / `attack == pe > 1800` invariants,
signal-dependence across two spectra, the `None` pre-frame and
out-of-range cases, and disarm-clears-capture. Truth from
`docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md` (§C.1.5.3.2.1
PE + window-switching threshold); no new tables required.

**Round 293 (Phase 2 step 90 — automatic per-granule Model 2
psychoacoustics in the encode loop).** Step 89 (r288) wired the
§C.1.5.3.2.1 Layer III Model 2 masking threshold `thm(sb)` into the
outer loop, but only through a one-shot convenience
(`set_per_band_xmin_from_model2`): the caller had to own a
`Model2Layer3State`, call `process` on each granule, and install the
result before every `push_samples`. This round makes the analysis a
running encoder mode. `enable_model2_psychoacoustics()` arms one
`Model2Layer3State` **per channel** (each threaded across every
granule of the stream so the §D.2.1 continuous-FFT-history requirement
holds — a channel's previous-granule spectrum feeds the next granule's
unpredictability prediction), and the encode loop drives it
automatically: Pass 1 runs each granule's 576-sample PCM through the
channel's Model 2 state and stashes the resulting signal-dependent
`xmin(sb)` (Figure C.6.c/d `thm(sb)` mapped via
`XminThresholds::from_layer3_granule`, per-band ratios preserved with
the geometric-mean offset anchored to the outer-loop threshold); Pass 2
installs that granule's threshold before the existing outer-loop
dispatch reads `per_band_xmin`. The mode is mutually exclusive with a
caller-installed static per-band vector (arming clears it; a later
`set_per_band_xmin` disarms the mode), requires the outer loop, and is
restricted to the three staged Annex D rates (32 / 44.1 / 48 kHz) — an
LSF / MPEG-2.5 rate is rejected with `Model2AnalysisUnsupported`
(those rates lack staged calculation-partition tables; still a
documented gap). 5 new unit tests: the outer-loop / unsupported-rate
guards, static-vs-automatic mutual exclusion, an end-to-end tone encode
whose frames all parse and which leaves a per-granule threshold
installed, and a two-tone run asserting the installed threshold is
spectrally shaped (not the flat uniform bowl). Truth from
`docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md` (Annex D
Model 2 §D.2 / §C.1.5.3.2.1); no new tables required.

**Phase 2 step 38 (`DEFAULT_ATTACK_THRESHOLD` empirical-corpus
calibration)** closes the dual of the r165 leak calibration on the
encoder-side `attack_detect::AttackDetector`. r165 pinned
`DEFAULT_AMBIENT_LEAK = 0.5` against a synthetic-corpus parameter
sweep while holding the threshold knob fixed at
`DEFAULT_ATTACK_THRESHOLD = 10.0`; the closing paragraph of that
step's README block called out the natural followup — *"a future
round that revisits `DEFAULT_ATTACK_THRESHOLD` should re-run the
sweep at the new threshold."* r192 runs that followup on the
threshold axis directly.

The sweep `THRESHOLD_SWEEP = [1.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0,
30.0, 50.0, 100.0]` spans the qualitative bounds the module doc
names: `≤ 3` over-aggressive, around `10` the recommended detection
sweet spot, `≥ 30` reserves short blocks for only the most extreme
bursts. Spacing is denser at the small-threshold end (where the
over-aggressive failure mode transitions sharply) and coarser at
the large end (where conservative thresholds saturate smoothly on
the corpus's burst-energy distribution). The leak knob is held at
the r165-calibrated `DEFAULT_AMBIENT_LEAK = 0.5` throughout so the
only varying axis is the threshold. The corpus is the same 8 rows
r165 built (`steady_sine`, `steady_noise`, `isolated_click`,
`burst_train_period4`, `slow_swell`, `swell_then_click`,
`sustained_drum_pair`, `level_shift`); the per-row error metric
`max(0, |observed − expected| − tolerance)` is identical to r165.

Five new tests pin the result (mirroring the r165 dual at the
threshold axis):

* `default_threshold_is_an_argmin_over_the_sweep` — no in-domain
  threshold strictly beats `10.0` on the aggregate metric.
* `default_threshold_beats_overaggressive_endpoint_and_ties_conservative`
  — `10.0` strictly beats the over-aggressive endpoint `1.0` and
  ties the conservative endpoint `100.0`. The asymmetry is the
  empirical headline at the threshold axis: at the calibrated leak
  the rejected region is `[1.0, 3.0]` (aggregate errors `179` and
  `4`), the transition region is `[5.0, 7.0]` (errors `2` and `2`
  — one residual fire each on the `slow_swell` and
  `swell_then_click` rows), and the acceptable plateau is
  `[10.0, 100.0]` (all tied at zero aggregate error). The default
  `10.0` is the *lowest-bound* argmin — every higher threshold
  ties because the corpus's burst magnitudes (`0.5–0.9` on a
  `1e-4` floor → subframe-vs-ambient ratios in the 10⁵–10⁶ range)
  sit orders of magnitude past every sweep point.
* `default_threshold_emits_zero_fires_on_steady_rows` — zero fires
  on both steady-state rows (`steady_sine`, `steady_noise`),
  isolating the false-fire failure mode (small threshold on
  modulated material) from the missed-fire failure mode (large
  threshold on a burst train).
* `default_threshold_catches_at_least_half_of_burst_train` — the
  burst-train row catches `≥ 5` of its 9 expected hits at the
  default threshold (in practice all 9 catch at the calibrated
  leak).
* `threshold_sweep_is_well_formed` — sorted, positive, finite,
  contains the running default, and spans the module-doc
  qualitative bounds (first ≤ 3, last ≥ 30).

The honest empirical finding the calibration leaves on the record:
the threshold knob's *tuning-relevant* range at the calibrated
leak is `[1.0, 7.0]` — the conservative plateau `[10.0, 100.0]`
ties at zero, so a workload that wants to lower the false-fire
rate further can only do so by tightening the *leak* in concert
(the r165 calibration block lays out that direction). The `10.0`
choice sits at the lowest-bound argmin, which is the best
operational policy for an encoder: lower thresholds forfeit the
zero-fire guarantee on steady-state rows, higher thresholds buy
nothing on the corpus while plausibly missing softer transients
on richer material. The module-doc qualitative bounds (≤ 3
over-aggressive, ≥ 30 conservative) are confirmed.

The first granule of each row is discarded as a seed-only call,
matching r165's procedure and the operational shape of the
detector (the encoder's `block_type_per_gc` pre-pass and the
§C.1.5.2 state machine both begin in `Long` regardless of the
first granule's classification). Tests: 634 pass (was 629; +5
from this step). The
corpus is the r165 corpus, the sweep is a one-axis extension of
the r165 sweep, and the metric is the r165 metric.

**Phase 2 step 37 (`oxideav_core::Decoder` trait MPEG-2 LSF
widening)** extends `Mp3CoreDecoder` from MPEG-1-only to **MPEG-1
and MPEG-2 LSF Layer III**, carrying the r177 stereo widening
across both versions (mono and stereo, independent / joint MS /
joint MS+intensity). The change is small in code and large in
scope because the downstream primitives were already version-
aware: `parse_side_info` already dispatches the §2.4.1.7 LSF
9-byte-mono / 17-byte-stereo single-granule layout
(`granule_count == 1`), `decode_scalefactors` already implements
the §2.4.3.4 LSF 9-bit `scalefac_compress` partitioning with
`slen1..4` and `nr_of_sfb1..4`, `requantize` already routes the
22-band long / 13-band short table per `(sample_rate, version)`,
and `stereo::process_stereo` already evaluates the LSF
intensity-position formula and the LSF
`int_scalefac_compress = scalefac_compress >> 1` right-channel
derivation. The only change in the trait wrapper is the
header-version guard: `MpegVersion::Mpeg1` and
`MpegVersion::Mpeg2` are both accepted, `MpegVersion::Mpeg25`
keeps its `Error::Unsupported` rejection with a message that
cites the `docs/audio/mp3/MPEG-2.5-GAP.md` observer-trace items
(scalefactor-band tables / Huffman table mapping / low-rate
frame-size validation at 8 / 11.025 / 12 kHz) as the gating
condition. The per-channel `imdct_state` / `synth_state` arrays
and the per-granule loop over `si.granule_count` / `si.channels`
need no further branching: an LSF frame iterates one granule
instead of two and emits an `AudioFrame` carrying
`PCM_PER_GRANULE = 576` samples per channel (vs MPEG-1's 1152).

Validated by 2 new integration tests in
`tests/decoder_trait_lsf_roundtrip.rs` against the staged
`docs/audio/mp3/fixtures/layer3-mpeg2-22050-64kbps` fixture
(64 kbps / 22.05 kHz / stereo MPEG-2 LSF Layer III, ID3v2-tagged):
`trait_decode_lsf_stereo_fixture_matches_direct_chain_byte_exact`
strips the ID3v2.4 frontmatter, pins the first audio frame's
`(version, sample_rate, channel_count) == (Mpeg2, 22050, 2)`,
walks every wire frame through both the trait wrapper and the
existing direct-chain decode primitives (parse_header →
parse_side_info → decode_scalefactors → decode_huffman →
requantize → process_stereo → alias_reduce → imdct_granule →
synth_granule), and asserts byte-exact per-channel `i16` PCM
equality plus the `samples == PCM_PER_GRANULE` / planar
`AudioFrame` invariants; `registry_built_decoder_handles_lsf_stereo_packets`
drives the same fixture through the `oxideav_mp3::register`-
installed factory and confirms byte-exact match. Both tests
skip cleanly under standalone-crate CI (workspace `docs/`
absent) per the established `tests/docs_corpus.rs` pattern.

Two new lib unit tests in `codec_decoder::tests` pin the
version-guard contract:
`send_packet_rejects_mpeg25_header_pending_observer_trace`
synthesises a real Fraunhofer MPEG-2.5 4-byte header via the
crate's own `make_silent_header(32, 11_025, SingleChannel)` +
`write_header` (round-trip-verified in the
`encoder::tests::header_writer_is_parse_inverse_mpeg25` test)
and asserts the wrapper rejects with an `Error::Unsupported`
whose message names "MPEG-2.5" or "observer-trace";
`send_packet_accepts_mpeg2_lsf_header_through_the_guard`
synthesises a real MPEG-2 LSF header (64 kbps / 22.05 kHz /
mono) and asserts the r177-style "MPEG-1 only" rejection no
longer fires — proving the version guard widened correctly
rather than relaxing into accept-all. Tests: 611 pass total
(+4 net from r177). 
**Phase 2 step 36 (`oxideav_core::Decoder` trait stereo widening)**
extends `Mp3CoreDecoder` from mono-only to MPEG-1 Layer III mono
**and** stereo (independent `ChannelMode::Stereo` /
`ChannelMode::DualChannel`, joint MS, joint MS+intensity). The
per-channel decode state — `ImdctState` for the §2.4.3.4.10.4
overlap memory and `SynthState` for the §2.4.3.2 polyphase
synthesis filterbank shift register — is carried in two-element
arrays inside the wrapper, with index `[0]` always live and
index `[1]` exercised on stereo packets. Each `send_packet` runs
a two-pass per-granule decode: first pass walks every channel
through `decode_huffman` + `requantize` and collects the
dequantized `xr[576]` lines; on `JointStereo` granules the
crate's existing `process_stereo` primitive then rewrites the
L/R pair in place per `mode_extension` (MS matrix and / or
intensity decode per §2.4.3.4.9.1–.9.3) using the right
channel's scalefactors and granule-channel side info for the
intensity bound; the second pass runs the per-channel
`alias_reduce` → `imdct_granule` → `synth_granule` tail and
writes each channel's PCM into its own plane of the emitted
`AudioFrame`. The output `AudioFrame` switches from a single
interleaved `data[0]` byte run to planar layout —
`data[0]` = L, `data[1]` = R for stereo, single plane for mono —
matching the framework's convention. `make_decoder` accepts
`channels = 1` or `channels = 2` and rejects every other value
with `Error::invalid`; the registry factory installed by
`crate::register` carries the same widening. The MPEG-1 only /
Layer III only checks at `send_packet` are unchanged. Validated
by 4 new integration tests in
`tests/decoder_trait_stereo_roundtrip.rs`:
`trait_decode_independent_stereo_matches_direct_chain_byte_exact`
encodes a 250 ms 440 Hz / 880 Hz LR-distinct sine pair through
`Mp3Encoder::new(192, 44_100, ChannelMode::Stereo)` and confirms
the trait wrapper produces **sample-for-sample-identical** L and
R PCM compared to a per-channel-state direct decode chain
(`process_stereo` is a pass-through on `mode_extension == '00'`,
exercising the per-channel state arrays);
`trait_decode_joint_ms_stereo_matches_direct_chain_byte_exact`
encodes a mono-on-L panned 440 Hz tone (`R = 0`) through
`Mp3Encoder::new_joint_stereo_ms` so the MS rotation moves real
energy onto the side channel, asserts the first frame's header
carries `JointStereo` + `mode_extension.ms_stereo == true`, and
confirms the trait wrapper still recovers the same L / R PCM
byte-exactly — proving the §2.4.3.4.9.2 inverse runs correctly
inside the wrapper rather than the pass-through path;
`trait_decode_stereo_emits_planar_audioframes_with_correct_sample_count`
pins the planar `AudioFrame` invariants (two `data[]` planes of
equal length, `samples == 1152` per MPEG-1 Layer III frame,
2 bytes per S16 sample on each plane); and
`registry_built_decoder_handles_stereo_packets` confirms the
factory installed by `crate::register` carries the widening
end-to-end. Existing mono behaviour is preserved bit-for-bit
(same byte-exact `trait_decode_matches_direct_chain_byte_exact`
assertion as r141, now passing against the per-channel-state
wrapper using only its `[0]` slot). Tests across the crate stay
green at 619 total (497 lib + 122 integration). Remaining work
flagged here: MPEG-2 LSF / MPEG-2.5 decode through the trait
wrapper (the framing layer, side-info parser, and scalefactor
decoder all accept LSF / MPEG-2.5 streams; the trait wrapper's
header guard rejects them pending end-to-end LSF synth-chain
fixtures and the `MPEG-2.5-GAP.md` observer-trace items).

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

**Phase 2 step 24 (§C.1.5.4.4.8 linbits-reach filter)** closes a
silent-truncation gap in the encoder's Huffman codebook chooser. The
§B.7 codebooks have widely-varying magnitude reach: the small tables
0..=15 reach `xlen - 1` (no linbits escape), the ESC tables 16..=31
reach `15 + (2^linbits - 1)` (e.g. table 16 `linbits = 1` reach 16;
table 23 `linbits = 13` reach 8206). The pre-r154
`huffman::choose_best_table_for_region` only verified the codebook's
`xlen` corner — and because the decoder's `decode_big_pair` clamps
the Huffman symbol to 15 before the table lookup, that corner test
was identically satisfied by **every** ESC table regardless of
magnitude. The encoder's `emit_big_pair` then silently truncated:
for a range with `|is| = 100`, the chooser could pick table 16
(`linbits = 1`), and emission would write `(100 − 15) & 0x1 = 1`
instead of the full delta, decoding back to `15 + 1 = 16`. The
in-tree workaround in `stream_encoder::best_table_or` had its own
hand-tabulated reach lookup; with the filter folded into the public
chooser, that local function collapses to a thin wrapper. The new
`huffman::big_table_reach(idx)` public helper exposes the
per-codebook reach the chooser uses. Verified by eight new unit tests
(`big_table_reach` pinned to §B.7 transcribed `xlen` / `linbits` for
all 32 codebooks; chooser-filter behaviour at the magnitude-15 /
-16 / -100 / -8191 boundaries with `encode_huffman` →
`decode_huffman` round-trip assertions; chooser returns `None` rather
than truncating for a magnitude past every codebook's reach;
all-zero / empty-range fallbacks). One pre-existing `inner_loop`
non-monotonicity test that inadvertently relied on the
silent-truncation behaviour — its `flat(30.0)` spectrum at very fine
`global_gain` exceeded the §C.1.5.4.4.2 magnitude clamp — is
tightened to walk only the clamp-respecting subset of the gain range
that `search_bit_budget` itself walks. Net: 474 tests pass (was 466).

**Phase 2 step 25 (MPEG-2.5 frame-parser support)** extends the
framing layer to the proprietary Fraunhofer-IIS "MPEG-2.5" extension
documented in `docs/audio/mp3/MPEG-2.5-GAP.md` (K. Brandenburg /
H. Popp, *An introduction to MPEG Layer-3*, EBU Technical Review 283,
June 2000; Fraunhofer-IIS U.S. patent RE44,897; datavoyage community
header reference). The §2.4.2.3 syncword is narrowed from 12 to
11 bits (`'1111 1111 111'`, header positions 31..21) and bit 20 is
repurposed as a second version-selector, so the 2-bit version field
at positions 20..19 is `'11'` = MPEG-1, `'10'` = MPEG-2 LSF, `'01'` =
reserved (new `HeaderError::ReservedVersion`), `'00'` = MPEG-2.5. A
new third [`MpegVersion::Mpeg25`] enum variant, a new
`SAMPLE_RATE_V25 = [11_025, 12_000, 8_000]` table (per the patent's
"preferably half the sampling rate" and the datavoyage table), and a
new `MpegVersion::is_lsf()` helper carry MPEG-2.5 through every
downstream call site: `samples_per_frame` returns 576 like MPEG-2,
`frame_len` uses the 72-byte coefficient, the V2,L1 / V2,L2&L3
bitrate ladders are reused (Fraunhofer patent "applied to ISO/IEC
13818-3"), the §13818-3 LSF side-info layout / scalefactor decode /
intensity-stereo factors / Xing-frame side-info-bytes calculation /
encoder `side_info_bytes` / demuxer `side_info_len` all dispatch on
the `is_lsf()` helper. The encoder's `write_header` writer is
re-grounded on the same 11-bit-sync layout and a new
`version_bits(MpegVersion) -> u32` returns the 2-bit field. The
`FrameWalker` pre-filter (already `& 0xE0 == 0xE0`) now matches the
narrowed sync exactly without further change. `make_silent_header`
accepts the three MPEG-2.5 sample rates and infers the new version;
the `oxideav_core::Decoder` trait wrapper this round is still
MPEG-1-only and rejects MPEG-2.5 with the same "decoder this round
is MPEG-1 only" message it returns for MPEG-2 LSF (consistent with
the in-tree `Mpeg25 != Mpeg1` test). Validated by 11 new unit tests
in `src/frame.rs` (single MPEG-2.5 32 kbps / 11.025 kHz parse with
576-sample / 208-byte invariants; full V2,L1&L23 ladder pinning at
the low and high ends; 8 kbps / 8 kHz / +padding frame-length pin;
all-three sample-rate table pin; reserved-version `'01'`
rejection; first-two-byte wire-format invariant; FrameWalker yields
back-to-back MPEG-2.5 frames; FrameWalker yields a mixed MPEG-1 +
MPEG-2.5 stream; `is_lsf` groups MPEG-2 and MPEG-2.5) and 3 new
unit tests in `src/encoder.rs` (writer ↔ parser inverse for the
new version; all three MPEG-2.5 sample rates round-trip through
the writer; `make_silent_header` accepts the MPEG-2.5 rate set
including 7 350 Hz rejection as the new unrecognised-rate stand-in
in place of the old 11 025 Hz stand-in). Net: 496 tests pass
(was 474). Spec-coverage note: the docs collaborator's residual
observer-trace items in `MPEG-2.5-GAP.md` (scalefactor-band index
tables for the 8 / 11.025 / 12 kHz rates, bit-exact Huffman table
mapping, and frame-size formula verification at the low rates) are
needed before a *decoded* MPEG-2.5 stream can be PCM-validated; the
present step lands only the framing layer so a downstream demuxer
can iterate / introspect / re-emit MPEG-2.5 frames without
upgrading the audio-decode chain.

**Phase 2 step 26 (signal-driven auto block-type dispatch)** replaces
the global force-toggles with a per-granule decision driven by
content, finally closing the long-standing follow-up the
step-22 / step-23 short-block / mixed-block primitives flagged. Two
new modules carry the logic:

* The `attack_detect` module exposes a stateful `AttackDetector` that
  splits each granule's 576 PCM samples into three 192-sample
  subframes (matching the §2.4.2.7 three-window short-block
  partition), computes per-subframe sum-of-squares energy, and flags
  the granule as carrying an attack iff the loudest subframe exceeds
  `threshold ×` the running ambient. The ambient is an exponentially
  smoothed `min`-floor of recent subframe energies with a configurable
  leakage (default `DEFAULT_AMBIENT_LEAK` = `0.5`): slow enough to
  ride a sustained transient train without rising into it, fast
  enough to track genuine background-level changes within ≈ 4
  granules. Default threshold `10.0`; the module docs detail the
  tuning envelope (≥ 30 reserves shorts for only the most extreme
  bursts, ≤ 3 over-fires on almost any modulated signal). As of r164
  the leakage factor is a per-instance knob alongside the threshold
  via `AttackDetectorParams { threshold, leak }` +
  `AttackDetector::with_params`; `with_threshold` keeps its
  signature and forwards through `with_params` with the default
  leak, so pre-r164 callers are unaffected. Both knobs are
  silently coerced to their `DEFAULT_*` counterparts on out-of-domain
  input (threshold ≤ 0 or non-finite; leak outside `(0, 1)` or
  non-finite), independently — one bad knob never drags the other to
  its default. Every
  constant and every formula is justified by the clean-room
  reasoning at the top of the module (energy localisation,
  ambient-floor stability, IIR leakage).

* The `block_type_sm` module exposes the §C.1.5.2
  `LONG → START → SHORT → STOP → LONG` transition state machine that
  turns the per-granule attack flags into geometrically valid
  `BlockType` decisions. The machine takes one granule of lookahead
  (`step(cur_attack, next_attack)`) so a `Start` window can be
  committed in time to splice into the next granule's `Short` head.
  The state transitions cover: sustained burst trains
  (`Short → Short` as long as either current or next granule still
  carries an attack), burst-then-quiet
  (`Short → End → Long`), back-to-back bursts separated by at least
  one Long granule (the post-`Stop` Long is mandatory because
  `End`'s tail is the long-window tail; a `Short` head can't
  splice into it), and the
  `cur_attack`-without-`next_attack` "burst arrived without
  anticipation" case that conservatively falls back to `Long`
  (we can't retroactively emit a `Start` for the previous
  granule). The transition geometry is all derived from
  §2.4.3.4.10.3's window shapes plus the §2.4.3.4.10.4
  overlap-add identity.

`Mp3Encoder::enable_auto_block_type(threshold)` is the opt-in API
that wires these together. The push / finish API contract is
preserved: `push_samples` holds back one extra granule of PCM as
the scheduler's lookahead while still emitting one frame per
1152-sample chunk in steady state; `finish` zero-pads the
held-back tail with a "no attack ahead" lookahead so the burst
geometry closes cleanly with a `Stop` if the stream's last
granule was inside a burst. The encoder dispatches per granule
between four MDCT paths — the long-family forward 36-point MDCT
with `BlockType::Long` / `Start` / `End` selecting the
appropriate `window_long_family_analysis` shape for the granule,
or three independent 12-point forward MDCTs via
`short_block::forward_short_mdct_subband` for `Short` — and
gates inverse alias reduction on `block_type != Short` per the
literal §2.4.3.4.10.1 wording. The side-info wiring
(`window_switching_flag` / `block_type` / window-switched-branch
defaults) follows the chosen block type per granule too.

Restrictions on the auto path: mono only (cross-channel
block-type agreement under §2.4.3.4.9 needed for stereo / joint /
dual-channel) and mutually exclusive with the existing
force-toggles (the testing toggles set the block type globally;
auto chooses per granule). The outer-loop combination is
**accepted** as of step 28 — see the "Phase 2 step 28" section
below for the dispatch shape; the r156 rejection of that
combination has been replaced by a positive integration test.
Validated by 27 new tests landed in step 26: 10 unit tests in
`attack_detect.rs` (silent / unit-DC subframe energies, pure sine
no-fire, step-burst flagged, pure silence not flagged with
bounded ambient, detector adapts after repeated bursts, click
after silence flagged, invalid-threshold fallback, reset clears
ambient), 8 unit tests in `block_type_sm.rs` (all-calm Long-only,
single burst emits Start/Short/Stop/Long, sustained burst holds
Short, two bursts with a Long gap both fire, current-only attack
without lookahead falls back to Long, Start → Short invariant,
Stop → Long invariant even on immediate burst, reset to Long),
and 9 integration tests in
`tests/auto_block_type_roundtrip.rs` (stereo rejection, default
off, enable/disable round trip, mutual exclusion with
force-short / force-mixed, pure sine stays Long, click train
engages Start / Short / End with §C.1.5.2 transition-validity
assertions on every emitted pair, demuxer acceptance of the auto
stream, auto + outer-loop combined integration roundtrip). Net:
523 tests pass at step 26 (was 496); step 27 raised it to 532;
step 28 preserves 532 (one test rewritten in place).

**Phase 2 step 27 (§C.1.5.4.3 short-block outer-loop analogue)**
lands `outer_loop_search_short` — the per-(sfb, window)
distortion-control iteration the auto-block-type dispatcher from
step 26 needs to run with the outer loop on. The new primitive
mirrors `outer_loop_search_long` for `block_type == Short`,
`mixed_block_flag == false` granules:

* **Per-cell amplification.** `band_distortion_short(xr, xr_back,
  sf, scalefac_scale, sr, ver)` returns the §C.1.5.4.3.3
  distortion as a `[[f64; 3]; 12]` keyed by `(sfb, window)`. Each
  iteration marks every cell with `xfsf_s[sfb][win] >
  uniform_threshold` and amplifies `scalefac_s[sfb][win] += 1` for
  the marked cells. Caps follow §C.1.5.4.3.6 with our
  `OUTER_LOOP_SCALEFAC_COMPRESS = 15` (slen1 = 4, slen2 = 3): 15 for
  the slen1-range short sfb 0..=5, 7 for the slen2-range sfb
  6..=11.
* **Bounded `subblock_gain` search.** Spec-silent (§C.1.5 leaves
  the heuristic to the implementation); we adopt the §C.1.5.4.4.2
  magnitude-clamp-driven scheme: when `search_magnitude_clamp`
  reports `satisfied == false` (a single window grossly outranges
  the others — the §2.4.3.4.7.1 `8·subblock_gain[w]` term exists
  exactly for this case), `per_window_max_abs` identifies the
  over-cap windows and bumps each one's `subblock_gain[w]` by 1
  (saturating at the §2.4.2.7 3-bit cap of 7), then restarts the
  iteration body. Quiet windows stay at zero; the field is never
  spent on bands that don't need it.
* **`scalefac_scale = 1` escalation.** Same §C.1.5.4.3 path as the
  long-block loop: when an amplification would push any cell past
  its §C.1.5.4.3.6 cap AND the loop is still in
  `scalefac_scale = 0` mode, halve every in-progress per-cell
  `scalefac_s` (round-to-nearest) and switch to
  `scalefac_scale = 1`. The `2^(mult·sf_s)` colouring is preserved
  (mult doubles 0.5 → 1.0; halving sf keeps the product
  unchanged). One escalation event only.
* **`preflag` invariant.** §2.4.2.7 says preflag is never set for
  short blocks; the result's `scalefactors.preflag` stays `false`
  unconditionally, mirroring the spec and the existing decoder
  invariant.

Restrictions on the step-27 primitive: pure short only
(`mixed_block_flag == 0`). The mixed-block analogue layers the
long-block amplifier (over the long-window bands 0..=7) on top of
this short-block amplifier (over the short-window bands 3..=11)
with the spec's remapped cap split; it is mechanically
straightforward but its own piece of work. **Integration into
`Mp3Encoder` lands in step 28 below** — `enable_auto_block_type`
can now run with the outer loop on for Short granules; Start / End
transition skeletons fall back to the fixed-gain inner-loop path
pending a transition-aware outer-loop primitive.

Validated by 9 new tests inside `outer_loop.rs`:
`short_upper_limits_match_spec`,
`short_band_distortion_zero_when_perfect`,
`outer_loop_short_terminates_with_huge_threshold`,
`outer_loop_short_terminates_with_tiny_threshold`,
`outer_loop_short_amplifies_only_offending_cells` (only the planted
(sfb, win) cell ever amplifies; silent cells stay at zero),
`outer_loop_short_raises_subblock_gain_on_extreme_window` (5e9
amplitudes in window 0 only — well past the GAIN_MAX reach of
~4.4e8 at default `subblock_gain` — must escalate
`subblock_gain[0]` while leaving windows 1 and 2 at zero),
`outer_loop_short_subblock_gain_stays_zero_on_quiet_input` (modest
amplitudes never spend a `subblock_gain` bit),
`outer_loop_short_default_preflag_off` (§2.4.2.7 invariant),
`outer_loop_short_escalates_scalefac_scale_when_cap_would_terminate`
(planted sfb-11 win-1 cell drives that cell past its slen2 cap of 7,
exercising the §C.1.5.4.3 escalation branch). Net: 532 tests pass
(was 523).

**Phase 2 step 28 (auto block-type × outer-loop integration)** wires
`outer_loop_search_short` into `Mp3Encoder::assemble_frame_with_lookahead`,
completing the missing half of §C.1.5.4.3 distortion-control coverage
for the auto scheduler.

`enable_auto_block_type` no longer rejects encoders configured with
`new_with_outer_loop` — the r156 pair-rejection was a placeholder
pending the short-block primitive that r157 landed. The per-(gr, ch)
outer-loop arm now inspects the granule's selected block type and
dispatches:

* **`BlockType::Long`** (window_switching off) →
  `outer_loop_search_long` (the r144 path); part2 cost 74 bits
  (11·slen1 + 10·slen2 under `scalefac_compress = 15`).
* **`BlockType::Short`** + `mixed_block_flag == false` →
  `outer_loop_search_short` (the r157 primitive); part2 cost
  126 bits (6·3·slen1 + 6·3·slen2). The returned per-window
  `subblock_gain[w]` is propagated into the granule's
  `subblock_gain` side-info field; the returned `scalefac_scale`
  is mirrored on the side-info as in the long path.
* **`BlockType::Start`** / **`BlockType::End`** (long-family
  transition skeletons, window_switching on) → fall back to the
  fixed-gain inner-loop path (magnitude clamp + bit-budget search).
  No outer-loop primitive covers transition skeletons yet — their
  §2.4.2.7 coefficient distribution shifts mid-overlap, so the
  uniform-`xmin` heuristic over-amplifies; a follow-up round will
  target them with a psy-aware threshold.

The §C.1.5.4.4.5 part2 / part3 budget split also tracks the per-block
part2 cost (Long: 74 bits, pure-short: 126 bits, mixed: 122 bits) so
the inner-loop budget check is bit-accurate per shape (the previous
single constant assumed Long).

Validated by `tests/auto_block_type_roundtrip.rs::auto_block_type_combines_with_outer_loop_and_roundtrips`,
which rewrites the r156 rejection assertion as a positive integration
test: a click-train PCM is encoded through
`Mp3Encoder::new_with_outer_loop(192, 44_100, SingleChannel,
/*xmin=*/ 1e-6)` + `enable_auto_block_type(DEFAULT_ATTACK_THRESHOLD)`,
`FrameWalker` parses every emitted frame, the §2.4.2.7 invariants
(`preflag == false` on every short granule; `subblock_gain[w] <= 7`
on every granule) are asserted, the demuxer accepts the stream, and
at least one `BlockType::Short` granule is witnessed so the new
dispatch path is provably exercised. Tests: 532 pass (same total as
r157; the r156 rejection test was rewritten in place per guardrail
#3).

**Phase 2 step 29 (§C.1.5.4.3 outer-loop mixed-block analogue)**
introduces `outer_loop_search_mixed` — the missing third
distortion-control primitive after `outer_loop_search_long` (r144) and
`outer_loop_search_short` (r157). For `block_type == Short`,
`mixed_block_flag == true`, `window_switching_flag == true` granules
the loop composes:

* The long-region per-band amplifier over `sf.long[0..=7]` (8 long
  scalefactor bands; long region covers exactly lines 0..36 at every
  MPEG-1 sampling rate per `long_band_starts`).
* The short-region per-(sfb, window) amplifier over
  `sf.short[3..=11][..]` (27 cells; short SFB 0..=2 are absorbed by
  the long-window portion since `short_band_starts[3] = 12` ⇒
  interleaved line 36 = the long / short partition).

§C.1.5.4.3.6 caps follow the mixed MPEG-1 part2 wire layout: every
long band reads at `slen1` ⇒ cap 15 across `sfb 0..=7` (distinct from
the pure-long path where `mpeg1_long_band_slen` would split at sfb 11);
short region splits as cap 15 on `sfb 3..=5` (slen1) and cap 7 on
`sfb 6..=11` (slen2). The §C.1.5.4.4.2 bounded `subblock_gain` search
fires when the magnitude clamp fails on a window (each step divides
that window's reconstruction by 4 per §2.4.3.4.7.1 short formula;
saturates at the 3-bit field cap of 7); the long region's
reconstruction does NOT use `subblock_gain`. The §C.1.5.4.3
`scalefac_scale = 0 → 1` escalation halves every in-progress per-band
scalefactor on BOTH regions so the coloured spectrum is preserved
across the scale switch. `preflag` stays `false` (§2.4.2.7 disables
preflag on every short-family granule including mixed).

Stream-encoder wiring: `outer_loop_eligible` widens from
`(false, Long, _) | (true, Short, false)` to
`(false, Long, _) | (true, Short, _)`; the `BlockType::Short` dispatch
arm now splits on `mixed_block_flag` and routes mixed onto
`outer_loop_search_mixed`. Composing
`Mp3Encoder::new_with_outer_loop(...)` with
`force_mixed_blocks_for_testing(true)` drives every assembled granule
through the new path. The wire signature is `scalefac_compress = 15`
on every (gr, ch) (the r158 fallback wrote 0 — confirming
distinguishability of the new path on the assembled bitstream).

Validated by 11 new unit tests in `outer_loop.rs` (mixed constant-vs-
spec alignment, distortion-helper identity + absorbed-band invariant,
termination on huge / tiny threshold, region-isolation tests
confirming the long amplifier fires only on long-region energy and
the short amplifier only on short-region energy, `subblock_gain`
quiet-input invariance, `subblock_gain` escalation on extreme
window-0 amplitudes, the `scalefac_scale` escalation branch on a
cap-would-terminate fixture) plus 4 new integration tests in
`tests/mixed_block_encoder_roundtrip.rs` (scalefac_compress = 15 wire
signature on every assembled (gr, ch); `subblock_gain` bounded ≤ 7;
finite + non-silent PCM roundtrip via the short-aware decode chain;
Mp3Demuxer accepts the new bitstream). Tests: 547 pass (was 532 at
r158; +11 unit + 4 integration). 
**Phase 2 step 30 (§C.1.5.4.3 outer-loop long-family transition-skeleton
wiring)** widens `outer_loop_search_long` from pure-Long
(`block_type == Long`, `window_switching_flag == false`) to the full
long-family `block_type ∈ {Long, Start, End}`. Start (block_type 1)
and End/Stop (block_type 3) carry the same 21 long scalefactor bands
as Long (§2.4.2.7 + Table 3-B.5), share the §2.4.3.4.7.1 long-block
requantize formula (no `subblock_gain` term), and use the same
§C.1.5.4.4.6 region-split rule (the inner-loop SUBDIVIDE function
dispatches on `block_type == Short` alone, so Long / Start / End all
take the same 1/3, 5/12, 1/4 partition driven by `big_values`). The
primitive therefore handles all three correctly with a relaxed
debug-assert and no logic change inside the loop body.

Stream-encoder wiring: `outer_loop_eligible` extends from
`(false, Long, _) | (true, Short, _)` to also include
`(true, Start, false) | (true, End, false)`. The
`BlockType::Start | BlockType::End` match arm — previously a
`debug_assert!(false)` unreachability marker because outer-loop
gating forbade those tags — now routes onto the same
`outer_loop_search_long` call as `BlockType::Long`, with
`subblock_gain = [0; 3]` (no subblock_gain on the long-family
branch). With `enable_auto_block_type` + `new_with_outer_loop` both
on, every block-type the §C.1.5.2 `LONG → START → SHORT → STOP →
LONG` scheduler emits now runs the outer loop. Previously Start /
End granules fell back to the fixed-gain inner-loop-only path with
`scalefac_compress = 0`; they now seed the
`OUTER_LOOP_SCALEFAC_COMPRESS = 15` signature on the wire and carry
the chosen per-band scalefactors as part2 at slen1 = 4 / slen2 = 3.

Validated by 5 new unit tests in `outer_loop.rs` (Start/End
templates terminate on a huge threshold; behavioural identity
between Long, Start, and End templates on identical `xr` —
including scalefactors, `global_gain`, `scalefac_scale`, `preflag`,
and the full `is[576]` output; Start template amplifies ≥ 1 band
under a tiny threshold) plus 2 new integration tests in
`tests/auto_block_type_roundtrip.rs` (every Start / End granule on
the click-train fixture carries the `scalefac_compress = 15`
outer-loop wire signature, confirming the fixed-gain fallback path
is no longer taken; the resulting bytestream remains
Mp3Demuxer-acceptable end-to-end). Tests: 554 pass (was 547 at
r159; +5 unit + 2 integration). 
**Phase 2 step 31 (§2.4.3.4.10.3 auto-block-type mixed-block
promotion)** closes the long-standing "auto path can never emit
Mixed" gap. The §C.1.5.2 LONG → START → SHORT → STOP → LONG
scheduler is geometric — it cannot tell *whether* a Short emission
should carve out the lowest 2 subbands as long (§2.4.3.4.10.3
mixed) or treat every subband as short. The new
[`mixed_classifier`] module decides per granule, from PCM alone, by
applying a one-tap moving-average low-pass kernel
`y[n] = (x[n] + x[n-1]) / 2` (transfer function `|cos(ω/2)|`, unity
DC gain, nulls Nyquist) and comparing the per-subframe energies of
the low-passed signal — if the max-to-min ratio stays at or below a
caller-chosen threshold (default `4.0`) the low band is judged
stationary across the granule and the mixed carve-out is
appropriate; otherwise the granule is judged broadband-bursting and
pure-short is preferred. The classifier is stateful only to carry
the previous granule's last sample across boundaries (so the LP
filter is continuous).

`BlockTypeStateMachine::step_with_mixed(cur, next, prefer_mixed)`
extends the scheduler with a per-call mixed-vs-pure-short preference;
the returned `(block_type, mixed_flag)` pair sets the flag only on
Short emissions (§2.4.2.7's syntactic invariant that
`mixed_block_flag` is meaningful only for `block_type == 2`). The
legacy `step` delegates with `prefer_mixed = false` so all prior
callers keep their pure-short behaviour.

`Mp3Encoder::enable_auto_block_type_with_mixed(attack_threshold,
low_band_stability)` is the new opt-in entry point: same lookahead /
detector / scheduler wiring as `enable_auto_block_type` plus a
per-channel `MixedClassifier`. The pre-pass classifies every granule
in parallel with the attack detector and feeds the boolean preference
to `step_with_mixed`; the resulting `mixed_per_gc` matrix drives the
forward MDCT branch and the `gc_template` selection. A Short emission
with `mixed_per_gc=true` takes the same forward path as
`force_mixed_blocks` (subbands 0..1 → 36-point long sine window;
subbands 2..31 → three 12-point short windows; `default_mixed_gc()`
template; no inverse alias reduction) and dispatches to the existing
r159 `outer_loop_search_mixed` primitive via the
`gc_template.mixed_block_flag` discriminator in the outer-loop
branch — no further outer-loop wiring required.

Validated by 10 new unit tests in `mixed_classifier.rs` (silent
granule degenerate case; DC stability; high-frequency-only attack
mixed-appropriate; broadband attack pure-short; cold-start
conservative-pure-short boundary case; LP unity-DC and Nyquist-null
checks; threshold validation; reset and prev_last tracking) plus 4
new unit tests in `block_type_sm.rs` (`step_with_mixed(_,_,false)`
matches `step` byte-for-byte; `prefer_mixed=true` sets the flag only
on Short emissions; sustained-burst flag propagation; per-call
preference toggling) plus 7 new integration tests in
`tests/auto_block_type_mixed_roundtrip.rs` (stereo rejection
inherited from plain auto; threshold round-trip; force-toggle
clearing; plain auto path stays unmixed; low-band-DC + Nyquist-click
stimulus engages ≥ 1 mixed granule while the plain auto path on the
identical PCM emits zero mixed granules; pure-sine stays Long under
mixed-auto; mixed-auto + outer-loop combination engages
`outer_loop_search_mixed` end-to-end with `scalefac_compress = 15`
on every mixed granule and Mp3Demuxer round-trip acceptance). Tests:
575 pass (was 554 at r160; +14 unit + 7 integration). 
**Phase 2 step 35 (`DEFAULT_AMBIENT_LEAK` empirical-corpus
calibration)** replaces the hand-wave justification for the
`DEFAULT_AMBIENT_LEAK = 0.5` constant promoted to public API in
r164 with a synthetic-corpus parameter sweep. The constant carries
the IIR adaptation rate of the encoder-side
`attack_detect::AttackDetector`'s ambient-energy estimate; r164's
README and module-doc justified the value on the same heuristic
ground the original private `LEAK = 0.5` constant carried
("halfway toward the new floor sample per granule"). r165 closes
that gap with a 7-row corpus and a `LEAK_SWEEP = [0.05, 0.1, 0.2,
0.3, 0.5, 0.7, 0.9, 0.95]` parameter scan.

The corpus enumerates the two failure-mode axes the leak knob
trades off against each other: false-fire from a lagging ambient
on a rising envelope (slow-leak failure) and missed-fire from
ambient absorption of a sustained transient (fast-leak failure).
Each row is synthesised in-test from a closed-form expression with
the expected fire-count derived from its construction:

* `steady_sine` (440 Hz constant amplitude, 40 granules): expected
  0 fires, tolerance 0. The steady-state baseline.
* `steady_noise` (deterministic xorshift32-driven white floor at
  ≈ −40 dB, 40 granules): expected 0, tolerance 0. The
  steady-state stochastic baseline.
* `isolated_click` (quiet floor with one mid-burst in the final
  granule's middle subframe): expected 1, tolerance 0.
* `burst_train_period4` (quiet floor with a sharp burst every
  4th granule, 9 burst granules in a 40-granule run): expected 9,
  tolerance 2. The slow-end-favouring signal — fast leak's
  ambient absorbs each burst and risks missing later ones.
* `slow_swell` (440 Hz sine whose amplitude grows linearly from
  0.001 to 0.5 over the run, no transient): expected 0, tolerance
  1. The fast-end-favouring signal — slow leak's lagging ambient
  reads the rising envelope as a transient and trips false fires.
* `swell_then_click` (the slow-swell signal with one terminal
  burst): expected 1, tolerance 1.
* `sustained_drum_pair` (two 3-granule loud hits in an otherwise
  quiet 40-granule run, separated by ≈ 14 quiet granules): expected
  2, tolerance 1.
* `level_shift` (10 quiet granules then 30 sustained loud
  granules): expected 1, tolerance 1.

The per-leak error is summed as
`max(0, |observed − expected| − tolerance)` across the corpus.
Four properties pin the result:

* `default_leak_is_an_argmin_over_the_sweep` — no in-domain leak
  strictly beats `0.5` on the aggregate metric.
* `default_leak_beats_slow_endpoint_and_ties_fast` — `0.5`
  achieves *strictly* lower error than the slow endpoint `0.05`
  and at most equal error to the fast endpoint `0.95`. The
  asymmetry is the empirical headline: at the default `10×`
  threshold, the slow-end failure mode bites long before the
  fast-end failure mode, so the rejected-leak region is
  `[0.05, 0.3]` (errors `15, 6, 1, 1`) while the
  acceptable-leak region is `[0.5, 0.95]` (errors all `0`).
* `default_leak_emits_zero_fires_on_steady_rows` — zero fires on
  both steady-state rows (`steady_sine`, `steady_noise`).
* `default_leak_catches_at_least_half_of_burst_train` — the
  burst-train row catches `≥ 4` of its 9 expected hits at the
  default leak (in practice all 9 catch at the `10×` threshold).

The first granule of each row is discarded as a seed-only call —
the detector's `ambient` starts at zero and any non-silent first
granule always trips a `ratio = e_max / SILENCE_FLOOR` overflow,
so the post-seed steady-state is what calibration measures (the
same `gr == 0` tolerance the pre-existing `pure_sine_not_flagged`
test already encoded). The encoder's `block_type_per_gc` pre-pass
sees the same seed semantics — the §C.1.5.2 state machine begins
in `Long` regardless of the first granule's classification — so
the corpus matches the operational shape of the detector.

The honest empirical finding the calibration leaves on the
record: the leak knob's relevant tuning range with the default
`10×` threshold is `[0.5, 0.95]` — the fast-end ties because the
threshold dominates IIR-relaxation dynamics there, the slow-end
loses on rising-envelope rows. A future round that revisits
`DEFAULT_ATTACK_THRESHOLD` should re-run the sweep at the new
threshold and tighten the `<=` in property 2 into a `<` if the
fast-end saturation collapses. The threshold sweep is itself a
natural r166+ followup but is out of scope for r165 — the present
step calibrates one knob in isolation. Tests: 602 pass (was 596;
+6 from this step). 
**Phase 2 step 34 (§2.4.3.4.10 finer attack-detector knobs)** turns
the encoder-side `attack_detect::AttackDetector`'s IIR adaptation
rate into a per-instance tunable. Before r164 the detector exposed
exactly one knob — the `subframe-to-ambient ratio` `threshold` that
the loudest subframe must exceed for a granule to be flagged — and
its ambient-update IIR was a private `LEAK = 0.5` constant baked
into `classify`. The leakage controls *adaptation* (how fast the
running ambient catches up to a changed background level), which is
the orthogonal axis to *sensitivity*: a slower leak makes the
detector ride a sustained transient train without absorbing it into
the ambient and missing subsequent attacks; a faster leak follows a
gradually-swelling background better but is more likely to drop a
real burst on the second occurrence. The two are independent design
choices and the spec leaves both unconstrained (§2.4.3.4.10's
window-switching policy is non-normative), so the right move was to
surface both knobs symmetrically rather than freeze the second one.

The new public surface is the `AttackDetectorParams { threshold,
leak }` value, the `DEFAULT_AMBIENT_LEAK` constant (`0.5`, the same
value the private constant carried), and three new methods on
`AttackDetector`: `with_params(params)` for the two-knob
constructor, `leak()` to read back the effective leak, and `params()`
for round-tripping the full tuning. `with_threshold` keeps its
signature and is now defined as
`with_params { threshold, leak: DEFAULT_AMBIENT_LEAK }`, so every
existing caller — `Mp3Encoder::enable_auto_block_type` /
`enable_auto_block_type_with_mixed`, the
`make_encoder_*_with_threshold` factories, all in-tree tests — keeps
its pre-r164 behaviour bit-for-bit. Validation matches the threshold
knob's silently-coerce-to-default contract: leak values outside the
open interval `(0, 1)` (NaN, infinities, ≤ 0, ≥ 1, or exactly the
endpoints — `0` would freeze the ambient at its seed value and `1`
would replace it on every granule, both of which defeat the IIR's
smoothing purpose) all fall back to `DEFAULT_AMBIENT_LEAK`. The two
knobs are validated independently so providing one bad value never
drags the other to its default. +7 unit tests in `attack_detect.rs`:
`default_params_match_documented_constants` pins the `0.5` /
`10.0` defaults; `with_params_round_trips_in_domain_values` and
`new_equivalent_to_with_params_default` cover the
constructor-equivalence contract; `with_params_validates_each_knob_independently`
and `leak_boundary_values_are_rejected` exercise the validation;
`with_threshold_uses_default_leak` proves no behaviour regression
for the legacy entry point; and
`slower_leak_keeps_firing_longer_than_faster_leak` is the
end-to-end behavioural witness — a slow-leak (`0.05`) and a
fast-leak (`0.95`) detector are seeded with the same quiet granule,
then fed an identical 10-granule sequence of loud bursts; the slow
detector fires at least as many times as the fast detector (and on
the test construction it fires strictly more, with the fast
detector adapting within ≈ 2 granules and falling silent). Tests:
596 pass (was 589; +7 net from this step). 
**Phase 2 step 33 (§2.4.3.4.9 cross-channel-MS block-type
agreement)** closes the gap step 32 left open by widening the four
block-type override toggles (`force_short_blocks_for_testing`,
`force_mixed_blocks_for_testing`, `enable_auto_block_type`,
`enable_auto_block_type_with_mixed`) onto MS-stereo joint modes
(encoders built via `Mp3Encoder::new_joint_stereo_ms` /
`Mp3Encoder::new_joint_stereo_auto`). The §2.4.3.4.9 agreement —
both channels of an MS-stereo granule must share `block_type` /
`window_switching_flag` / `mixed_block_flag` so that the
§2.4.3.4.9.2 forward MS matrix `M = (L+R)/√2`, `S = (L-R)/√2`
sees both halves in the same time-frequency tile — is now
enforced **inside** the encode pre-pass rather than via an API
reject. The four entry points return `Ok` for every channel
layout the encoder supports.

The force paths trivially agree: every (gr, ch) tile in
`block_type_per_gc` is `[[BlockType::Short; 2]; GRANULES]` (or
the mixed-tagged equivalent under `force_mixed_blocks`), so both
channel slots carry the same block-type and the same
`mixed_block_flag` by construction. The auto paths add a new
`ms_agreement_active` branch in the `block_type_per_gc` pre-pass:
each channel's PCM is still classified by its own attack detector
and mixed classifier (per-channel ambient estimates so a quiet
channel doesn't drag the loud one's threshold around), but the
per-channel per-granule attack flags and mixed-classifier flags
are folded via logical OR before stepping a single shared
scheduler (we use `scheduler[0]`'s slot; the channel-1 scheduler
is bypassed in this regime). The shared scheduler's per-granule
`(BlockType, mixed_flag)` emission is then mirrored across both
channel slots of `block_type_per_gc[gr]` and `mixed_per_gc[gr]`.

The OR-fold is the "safe upper envelope" agreement rule: an
attack on either L or R triggers the §C.1.5.2 transition for
both. It accepts more short bursts than a hypothetical per-channel
sequence would (each channel sees the other's transients) but
never under-resolves a real transient on either side, and
produces a self-consistent §C.1.5.2 sequence across one shared
scheduler — no half-formed `Start-without-Short` chain the way
two independently-stepped schedulers might if their flags
happened to disagree at a transition boundary. Symmetric in L↔R
by construction. Independent stereo (`ChannelMode::Stereo` /
`ChannelMode::DualChannel` without joint coupling) keeps the
r162 per-channel-scheduler behaviour: each channel runs its own
detector + scheduler and picks its own §C.1.5.2 transition state.

Validated by 8 new integration tests and 5 rewrites. The
rewrites flip the r162 "MS-stereo + toggle → rejected" assertions
to their "accepted" counterparts. The new tests add wire-level
§2.4.3.4.9 agreement witnesses:
`force_short_blocks_on_ms_stereo_writes_short_agreed_side_info`,
`force_mixed_blocks_on_ms_stereo_writes_agreed_mixed_side_info`,
and
`auto_block_type_on_ms_stereo_agrees_per_granule_and_responds_to_either_channel`
encode 250 ms / 1 s of stereo PCM through a MS-stereo encoder
with the respective toggle on, then assert that every emitted
granule's L and R side-info agrees on `window_switching_flag` /
`block_type` / `mixed_block_flag`. The auto witness additionally
puts a click train on the LEFT channel and a sustained sine on
the RIGHT; the OR-fold means at least one §C.1.5.2 transition
must fire (the right channel follows the left into Start /
Short), which would NOT hold under the independent-stereo
per-channel-scheduler rule. All emit valid `Mp3Demuxer`-acceptable
bitstreams. Tests: 589 pass (was 586 at r162; +3 net). No
external implementation consulted.

**Phase 2 step 32 (§2.4.3.4.9 independent-stereo widening of the
block-type override toggles)** narrows the long-standing
"force-short / force-mixed / auto / auto-mixed are mono-only"
restriction to its actual spec basis. The §2.4.3.4.9 same-block-type
requirement only binds when **MS-stereo** is active (the joint-mode
matrix `M = (L+R)/√2`, `S = (L-R)/√2` rotates the L/R pair before
quantize, and the decoder needs both halves to share window
geometry); independent stereo ([`ChannelMode::Stereo`] /
[`ChannelMode::DualChannel`]) carries per-channel side-info verbatim
per §2.4.1.7 / §2.4.2.7 and has no such constraint. The four
override entry points are therefore widened:

- `force_short_blocks_for_testing(true)`
- `force_mixed_blocks_for_testing(true)`
- `enable_auto_block_type(threshold)`
- `enable_auto_block_type_with_mixed(attack, low_band_stability)`

now accept mono **and** independent stereo (`nch == 2` without joint
coupling) and still reject MS-stereo joint modes built via
`Mp3Encoder::new_joint_stereo_ms` / `Mp3Encoder::new_joint_stereo_auto`.
The gate is the new private `Mp3Encoder::ms_joint_stereo_active`
predicate (`self.ms_stereo || self.ms_auto_threshold.is_some()`),
which captures both the unconditional MS path and the per-frame
MS/LR auto-picker. The MS-stereo gap remains the §2.4.3.4.9
follow-up; force-mode + MS-stereo and auto + MS-stereo continue to
return `StreamEncodeError::StereoUnsupported`.

The downstream encode loop was already per-(gr, ch): the
`block_type_per_gc[gr][ch]` matrix has iterated `0..self.nch` since
r156, the auto path's `AutoBlockTypeConfig` already sized its
detector / scheduler / mixed-classifier vectors to `nch`, and the
MDCT / gc-template / outer-loop branches all index per-channel. The
only change to the encode pipeline is the API-time guard relaxation
+ updated comments on the §C.1.5.2 / §2.4.3.4.9.2 reasoning. The
per-channel scheduler independence means independent-stereo auto
behaves correctly without further wiring: a click train on the left
and a sustained sine on the right produces non-Long granules on
channel 0 and Long-only on channel 1 in the same frame.

Validated by 11 new integration tests: 3 in
`short_block_encoder_roundtrip.rs` (MS-stereo + force-short
rejected, MS-auto + force-short rejected, independent-stereo +
force-short accepted including a per-channel side-info wire check
on a stereo click+sine stimulus and Mp3Demuxer round-trip); 4 in
`mixed_block_encoder_roundtrip.rs` (MS-stereo + force-mixed
rejected, MS-auto + force-mixed rejected, independent-stereo +
force-mixed accepted, and a stereo force-mixed end-to-end wire +
demuxer test on a 220 Hz / 440 Hz interleaved stimulus); 3 in
`auto_block_type_roundtrip.rs` (MS-stereo + auto rejected, MS-auto
+ auto rejected, independent-stereo + auto accepted, plus a
per-channel scheduler witness that pushes a click train on the
left and a sustained sine on the right and asserts non-Long
granules in channel 0 but Long-only in channel 1); and 3 in
`auto_block_type_mixed_roundtrip.rs` (mirrors of the above for
`enable_auto_block_type_with_mixed`). Tests: 586 pass (was 575 at
r161; +11 integration). 
**Phase 2 step 39 (r194)** — per-band psychoacoustic-threshold
scaffold (Annex D threshold-in-quiet long-block path). The new
`psy` module exposes `XminThresholds` — a typed per-band threshold
vector (`[f64; LONG_SFB]` long, plus short / mixed cells for the
follow-up rounds) and two constructors: `XminThresholds::uniform`
(byte-equivalent shim for the pre-r194 scalar threshold path) and
`XminThresholds::threshold_in_quiet_long`, which derives a per-band
`xmin[sfb]` from the Annex D Table D.1 *threshold in quiet* curve.
The curve is sampled via monotone piecewise-linear interpolation
through **only the textually-transcribed anchors** in
`docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md` — i = 1
(62.5 Hz / 33.44 dB), i = 2…5 (the first five rows), i = 51
(prose-anchored minimum near 3.375 kHz / −4.97 dB), and i = 108
(15 kHz / 51.04 dB). The PNG-only inner rows of D.1a–f are
**deliberately not OCR'd** this round (DOCS-GAP); the
under-sampled curve is the conservative direction (under-estimated
`xmin` → more aggressive amplification → more bits on that band →
strictly higher quality). The §D.1 Step 3 bitrate-dependent offset
is applied verbatim: −12 dB for ≥ 96 kbit/s, 0 dB below.

The outer-loop `outer_loop_search_long` was refactored: the
existing `(uniform_threshold: f64, …)` API becomes a thin shim over
a new `outer_loop_search_long_per_band(…, xmin_per_band: &[f64; 21],
…)` primitive that reads the per-band entry in every §C.1.5.4.3.5
amplification + §C.1.5.4.3.6 termination test (and in the
§C.1.5.4.3.4 preemphasis decision). The per-band primitive is a
strict generalisation: broadcasting the scalar into a uniform
vector recovers byte-for-byte the scalar path. The stream encoder
exposes `Mp3Encoder::set_per_band_xmin` to install the per-band
vector; the long-block outer-loop dispatch routes to the per-band
primitive when set. The short / mixed branches still consume the
scalar threshold this round (their per-band variants land in a
follow-up; the `XminThresholds` struct exposes the short /
mixed-short cells today so the API doesn't churn).

Validated by 6 new integration tests in
`per_band_xmin_roundtrip.rs`: byte-equivalence between the per-band
uniform shim and the scalar path (regression anchor for the
refactor), threshold-in-quiet long-block self-decode at finite
PSNR > 20 dB on a 6-tone multi-tone fixture (measured 69.94 dB),
divergence between the per-band LTq path and the uniform path at
the same scalar threshold (witnesses the LTq vector actually
propagating into the §C.1.5.4.3 decision), API rejection of
`set_per_band_xmin` without the outer loop enabled, silence
round-trip, and single-tone 440 Hz self-decode at 65.77 dB. The
`psy` module adds 9 unit tests. Tests: 649 pass (was 633 baseline;
+10 unit, +6 integration). 
**Phase 2 step 41 (r204)** — per-band psychoacoustic-threshold
plumbing for the **mixed-block** path. Closes the dispatcher gap
left by r194 (long path) and r197 (pure-short path): the
`outer_loop_search_mixed_per_band` primitive now accepts a
`xmin_long: &[f64; LONG_SFB]` long-region per-band vector (entries
`[0, MIXED_LAST_LONG_SFB]` = `0..=7` are read; the rest are
ignored) AND a `xmin_short: &[[f64; SHORT_WINDOWS]; SHORT_SFB]`
short-region per-cell matrix (entries
`[MIXED_FIRST_SHORT_SFB, SHORT_SFB)` = `[3, 12)` are read; the rest
are ignored). Every §C.1.5.4.3.5 amplification + §C.1.5.4.3.6
termination cell test reads the appropriate vector / matrix entry.

The pre-r204 `outer_loop_search_mixed(_, _, _, _, _, thr, _)` is
refactored to a thin scalar shim that broadcasts the uniform
threshold into a uniform `[thr; LONG_SFB]` long-region vector AND a
uniform `[[thr; SHORT_WINDOWS]; SHORT_SFB]` short-region matrix,
then dispatches into the new per-band primitive — byte-for-byte
equivalent to the pre-r204 inline body (regression-anchored by
`mixed_per_band_uniform_matches_scalar_bit_for_bit`).

The `stream_encoder` dispatch is wired so the
`BlockType::Short if mixed_block_flag` arm routes onto the new
per-band primitive whenever `set_per_band_xmin` has installed a
matrix, consuming `XminThresholds::mixed_long` for the long region
and `XminThresholds::mixed_short` for the short region. Without
`set_per_band_xmin` the dispatch falls back to the scalar shim, so
existing callers see no change. `XminThresholds::threshold_in_quiet`
(landed in r197) already populates both `mixed_long` and
`mixed_short` from the same Annex D anchors — `mixed_long[0..=7]`
shares the long-band derivation with `long[0..=7]`, and
`mixed_short[3..=11][..]` shares the per-window short-band derivation
with `short[3..=11][..]` — so installing the LTq vector is enough to
exercise the new path end-to-end.

Validated by 3 new integration tests in
`per_band_xmin_roundtrip.rs`: byte-equivalence between the per-band
uniform shim and the scalar path on a force-mixed encoder (regression
anchor for the refactor); threshold-in-quiet mixed-block self-decode
to finite, non-silent PCM through the crate's own decode chain;
divergence between the per-band LTq path and the uniform path at the
same scalar threshold on a force-mixed encoder (witnesses both the
`mixed_long` vector and the `mixed_short` matrix actually propagating
into the §C.1.5.4.3 decision). The `outer_loop` module adds 6 unit
tests: shim equivalence, huge-threshold iter-1 termination, a tighter
long band amplifies only that long band, a tighter short cell
amplifies only that short cell, out-of-range entries are ignored, and
a long-region skew diverges from the long-region uniform path. Tests:
671 pass (was 662 baseline; +6 unit, +3 integration). 
**Phase 2 step 42 (r207)** — trait-API one-shot threshold-in-quiet
factory. The r194 / r197 / r204 per-band scaffolds required callers
to build the encoder via `Mp3Encoder::new_with_outer_loop(…,
DEFAULT_OUTER_LOOP_THRESHOLD)` and **then** install the per-band
vector via `Mp3Encoder::set_per_band_xmin(XminThresholds::threshold_in_quiet(SR,
version, bitrate_kbps_per_channel))`. r207 collapses that recipe to a
single constructor `Mp3Encoder::new_with_threshold_in_quiet(bitrate_kbps,
sample_rate_hz, mode)` (direct API) plus a matching trait-API factory
`codec_encoder::make_encoder_with_threshold_in_quiet(&params)`. Both
derive the per-channel bitrate (`bitrate_kbps / nch`, with `nch = 1`
for `SingleChannel` and `nch = 2` for `Stereo` / `DualChannel`) and
pass it to `XminThresholds::threshold_in_quiet` so the §D.1 Step 3
`−12 dB` offset switches on exactly when
`bitrate_kbps_per_channel >= 96` — 128 kbit/s mono (per-ch 128, ≥ 96)
triggers, 64 kbit/s mono (per-ch 64, < 96) does not, 192 kbit/s
stereo (per-ch 96) is exactly the cutover. The carried uniform-scalar
slot is `DEFAULT_OUTER_LOOP_THRESHOLD` so a follow-up
`set_per_band_xmin` re-override sees the same convergence dynamics
as `new_with_outer_loop` at the default threshold.

Validated by 9 new lib unit tests: 4 in `stream_encoder::tests`
(`new_with_threshold_in_quiet_enables_outer_loop_and_per_band` pins
both knobs armed; `…_carries_long_band_bowl_shape` extracts the
installed `XminThresholds` and confirms the bass/treble extremes sit
above the mid-spectrum minimum, witnessing the
threshold-in-quiet derivation actually fired; `…_applies_step3_offset_per_channel_bitrate`
pins the 10^1.2 ratio between high-br and low-br mono at the same
sample rate; `…_stereo_uses_per_channel_bitrate_for_step3` pins the
same ratio between 192-kbit/s stereo and 128-kbit/s stereo, proving
the offset reads the per-channel bitrate, not the aggregate), plus
5 in `codec_encoder::tests`
(`make_encoder_with_threshold_in_quiet_constructs_and_reports_params`,
`…_accepts_stereo`, `…_rejects_more_than_two_channels`,
`…_requires_sample_rate`, and an end-to-end
`…_emits_self_decoding_stream` that drives 4 frames of 440 Hz mono
sine through `send_frame` + `flush` + `receive_packet` and confirms
the assembled byte stream walks cleanly via `FrameWalker` +
`parse_header` at the configured 44.1 kHz). Tests: 556 lib (was 547
baseline; +9 unit). 
**Phase 2 step 43 (r213)** — caller-supplied §D.1 Step 3 dB offset
path. The Phase 2 step 42 factory `new_with_threshold_in_quiet` (and
its trait companion) derives the spec's two-branch offset (`−12 dB`
when `bitrate_kbps_per_channel >= 96`, `0 dB` otherwise) from the
per-channel bitrate. r213 surfaces the dB knob directly via
`XminThresholds::threshold_in_quiet_with_offset_db(sample_rate_hz,
version, offset_db)` + `Mp3Encoder::new_with_threshold_in_quiet_offset(bitrate_kbps,
sample_rate_hz, mode, offset_db)` + the trait factory
`codec_encoder::make_encoder_with_threshold_in_quiet_offset(&params,
offset_db)`. The caller can now sweep the offset continuously (e.g.
exposing a transparency / quality slider, or letting a VBR front-end
pick a running offset from a recent-bitrate accumulator) instead of
being limited to the two spec values. The bowl-vs-bass-vs-treble
per-band ordering is preserved — `offset_db` is a uniform dB
translation of the whole curve. Spec defaults remain bit-compatible:
`offset_db = -12.0` reproduces the high-bitrate path of
`new_with_threshold_in_quiet(128, 44_100, mono)` to within FP, and
`offset_db = 0.0` reproduces the low-bitrate path of
`new_with_threshold_in_quiet(64, 44_100, mono)`.

Validated by 13 new lib unit tests: 5 in `psy::tests`
(`threshold_in_quiet_with_offset_db_recovers_spec_high_bitrate_path` +
`…_recovers_spec_low_bitrate_path` pin FP-tolerance equivalence with
the two spec branches across long, short, mixed, mixed-short cells;
`…_tightens_below_spec_minus12` + `…_loosens_above_zero` pin the
`10^(Δdb/10)` linear-ratio invariant for translation; `…_preserves_bowl_shape`
confirms the bass/treble extremes still sit above the mid-spectrum
minimum at `offset_db = -30 dB`), 4 in `stream_encoder::tests`
(`new_with_threshold_in_quiet_offset_arms_outer_loop_and_per_band` pins
both knobs armed; `…_minus12_matches_spec_high_bitrate_path` +
`…_zero_matches_spec_low_bitrate_path` pin per-band equivalence with
the spec-default constructor at the two anchor offsets;
`…_monotone_in_offset_db` pins the strict ordering between
`-24 dB` and `0 dB`), plus 4 in `codec_encoder::tests`
(`make_encoder_with_threshold_in_quiet_offset_constructs_and_reports_params`,
`…_emits_self_decoding_stream` end-to-ending 4 frames of 440 Hz mono
sine at `offset_db = -18 dB` through `send_frame` + `flush` +
`FrameWalker`, plus the `…_rejects_more_than_two_channels` /
`…_requires_sample_rate` validation guards). Tests: 569 lib (was 556
baseline; +13 unit). 
**Phase 2 step 44 (r219)** — Annex D Model 1 §D.1 Step 6
masking-function `vf` + masking-index `av_tm` / `av_nm` + Step 7
global-threshold summation primitives. The prior threshold-in-quiet
work (r194 / r197 / r204 / r207 / r213) gave the outer loop a
per-band lower bound on its `xmin` vector — the signal-independent
floor of any psychoacoustic threshold. r219 lands the masker-driven
upper structure of Model 1 itself as a set of pure, future-callable
primitives:

* `masking_index_tonal(z_j)` reproduces the verbatim §D.1 Step 6
  formula `av_tm = -1.525 - 0.275 * z(j) - 4.5` dB.
* `masking_index_non_tonal(z_j)` reproduces `av_nm = -1.525 - 0.175
  * z(j) - 0.5` dB.
* `masking_function_vf(dz, X)` returns the verbatim 4-branch
  piecewise `vf` for `dz ∈ [-3, 8)`:
  * Branch 1, `-3 ≤ dz < -1`: `vf = 17·(dz+1) - (0.4·X + 6)` dB.
  * Branch 2, `-1 ≤ dz < 0`: `vf = (0.4·X + 6)·dz` dB.
  * Branch 3, `0 ≤ dz < 1`: `vf = -17·dz` dB.
  * Branch 4, `1 ≤ dz < 8`: `vf = -(dz-1)·(17 - 0.15·X) - 17` dB.
  * Outside the window: `None` (masker ignored — `LT = -∞ dB`).
* `individual_masking_threshold_db(masker, z_i)` composes a
  single masker's `LT = X + av + vf` at any target Bark line
  `z(i)`, returning `None` when the line falls outside the
  masker's `[z_j - 3, z_j + 8)` reach.
* `global_masking_threshold_db(maskers, z_i, ltq_db)` carries out
  the §D.1 Step 7 power sum `LTg(i) = 10·log10( 10^(LTq/10) + Σ_j
  10^(LT_j/10) )` across every in-range masker plus the
  threshold-in-quiet anchor, returning the global masking
  threshold in dB at `z(i)`.

The Bark coordinates the primitive operates on (`z_bark` on each
`Masker { kind, z_bark, spl_db }`) are *abstract* — the caller is
free to derive them from any subband / FFT-bin Bark mapping. The
spec's recommended Bark mapping table comes from the PNG-only Annex
D Table D.2 set; this round deliberately does not consume those
tables. Steps 1–5 of Model 1 (1024-sample FFT, SPL conversion,
tonality classifier, decimation / reorganisation, masker selection)
similarly remain blocked on the PNG-only Tables D.1 / D.2 / D.3 /
D.4 DOCS-GAP and are not landed this round; r219 supplies the
masker → masking-threshold half of the model that the future Steps
1–5 will eventually drive.

Validated by 18 new lib unit tests in `psy::tests`:
`masking_index_tonal_recovers_spec_formula` +
`masking_index_non_tonal_recovers_spec_formula` reproduce the two
verbatim equations at five Bark positions each;
`masking_index_tonal_below_non_tonal_at_same_z` pins the
`av_tm < av_nm` ordering across the band; six
`masking_function_vf_*` tests cover the four piecewise branches
with hand-computed numeric anchors (`vf(-3, 60) = -64`,
`vf(-2, 80) = -55`, `vf(-1, 60) = -30`, `vf(-0.5, 60) = -15`,
`vf(0.5, 60) = -8.5`, `vf(1, 60) = -17`, `vf(2, 60) = -25`,
`vf(5, 80) = -37`) plus continuity at `dz = 0` and `None`
out-of-range guards on both sides of `[-3, 8)`;
`individual_masking_threshold_db_tonal_at_self_is_spl_plus_av` +
`…_non_tonal_at_self_is_spl_plus_av` pin the `LT = SPL + av` at
`z(i) = z(j)` identity for both classifications;
`…_returns_none_outside_window` pins the masker reach;
`…_tonal_below_non_tonal_at_same_z` confirms the tonal LT sits
below the non-tonal LT at matched parameters across five `z(i)`
samples inside the window. Five `global_masking_threshold_db_*`
tests cover Step 7 reductions: no maskers → LTg = LTq; distant
masker (outside `[-3, 8)` Bark) → LTg = LTq; strong nearby
masker → LTg dominates LTq within < 1 dB of the masker's
own LT; LTg(both) > LTg(either) for two maskers (monotone
power addition); exact `+10·log10(2) ≈ +3.0103 dB` for two
equal-power co-located maskers. Tests: 587 lib (was 569
baseline; +18 unit). Only
the textually-transcribed `av` / `vf` / `LTg` equations from
`docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md` were
read.

**Phase 2 step 86 (r284)** — **§2.4.3.4.9.3 intensity-stereo encode**,
closing the last encoder "lacks" item. Three opt-in constructors arm
the coupling on the stream encoder: `Mp3Encoder::new_joint_stereo_is
(bitrate, sample_rate, start_sfb)` emits `mode = '01'` with
`mode_extension = '01'` (intensity only),
`new_joint_stereo_ms_is(…)` emits `'11'` (§2.4.3.4.9.2 MS below the
intensity bound + intensity above it, per the §2.4.3.4.9.1 scoping),
and `new_joint_stereo_auto_is(…)` runs the r147 per-frame MS/LR
energy picker — evaluated over the below-bound lines only — emitting
`'11'` / `'01'` per frame. Coupling (per Annex G.2 c): for every long
scalefactor band at or above `start_sfb` (validated `1..=20`), the
per-band stereo position `is_pos[sfb] =
NINT((12/π)·arctan(√(E_L/E_R)))` (positions 0..=6; the `E_R → 0`
limit and fully-silent bands map to 6) is derived from the pre-MS
L/R band energies, the left channel is rewritten to the combined
magnitude `L + R`, and the right channel to the all-zero
§2.4.3.4.9.1 zero-part; the partial top region above the last
Table B.8 band boundary (no scalefactor slot exists for it) is
coupled the same way and decodes left-only. Wire-up: the right
channel of an intensity frame is forced to `scalefac_compress = 15`
(`slen` 4/3 — every position fits) with the 74-bit part2 cost
deducted from the fixed-gain path's part3 budget; after the
quantizer converges, the intensity region's scalefactors are
overwritten with the positions and every all-zero band between the
last non-zero quantized right-channel line and the bound gets the
illegal-position marker `7` (Annex G.2 c) — without it a decoder
deriving the bound from the zero-part would intensity-decode those
bands with leftover scalefactors. Long-block only **this** round: the
short-window bound is per window (per-window `is_pos` from
`scalefac_s`), so the force-short / force-mixed / auto-block-type
toggles rejected with the new `IntensityShortBlocksUnsupported` error
while intensity was armed (`InvalidIntensityStartSfb` covers the range
check). **Round 303 lifted the force-short rejection** for the
intensity-only path (per-window positions on `scalefac_s[sfb][win]`) and
**Round 305 lifted it for the unconditional MS + intensity path** (the
§2.4.3.4.9.2 matrix now rotates per window below the short bound); see the
Round 305 / Round 303 entries below. Registry path:
`make_encoder_joint_stereo_is` / `make_encoder_joint_stereo_ms_is`
mirror the direct constructors. Validated by 8 new lib unit tests
(position grid + monotonicity, constructor state/template bits,
range rejection, toggle interlocks, two registry factories + reject
matrix) and 8 integration tests in
`tests/joint_stereo_intensity_roundtrip.rs`: header + wire bits
(`'01'`), wire-level scalefactor layout (positions ≤ 6 in the
region, derived `is_pos = 5` on the 6 kHz probe band, marker `7` on
the zero tail, zero-part above the bound), a hard-left probe
(all-21-band marker/position split), self-decode positional
fidelity (reconstructed 6 kHz |L|/|R| = 3.733 vs the `tan(5π/12) ≈
3.732` grid angle; below-bound 440 Hz pan preserved) with PSNR at
parity with the independent-stereo / MS-only encodes of the same
PCM (R-channel PSNR *improves* 25.7 → 29.2 dB under intensity),
`'11'` MS+intensity round-trip, the auto picker flipping `'11'` ↔
`'01'` on correlated vs anti-phase below-bound content, byte-exact
encode determinism + bit-exact re-decode, and black-box
cross-decode through `ffmpeg` and `mpg123` CLI binaries (both
reproduce the 3.728 positional ratio and the below-bound pan; bytes
only). Tests: 1046 lib (was 1038;
+8 unit) + 8 integration.

**Phase 2 step 87 (r286)** — **LSF (MPEG-2 / MPEG-2.5) intensity-stereo
encode** (ISO/IEC 13818-3 §2.4.3.2), lifting the r285
`LsfUnsupported` rejection on the three intensity constructors. The
coupling geometry is shared with the MPEG-1 path (left := L+R, right
:= the §2.4.3.4.9.1 zero-part, per-band positions in the right
channel's scalefactor slots, the illegal-position marker on the
below-bound zero tail), but the wire format differs: the LSF
intensity-right channel writes the new
`INTENSITY_SCALEFAC_COMPRESS_LSF = 258` — `258 >> 1 = 129 < 180` ⇒
the §2.4.3.2 right-channel partition `slen = (3, 3, 3, 0)` /
`nr_of_sfb = (7, 7, 7, 0)` (3 bits on every one of the 21 long
bands), `258 % 2 = 0` ⇒ `intensity_scale = 0`. 3 bits is the
smallest width holding positions `0..=6` plus the marker, and makes
`7` the *maximum* value (hence the illegal-position marker the
decoder tests for, per §13818-3 "the maximum value for intensity
position will indicate an illegal intensity position"). Positions
are derived on the §2.4.3.2 power-law `i0 = 2^(-1/4)` ladder
(`derive_intensity_position_lsf` picks the closest decoded amplitude
ratio `kl/kr ∈ {1, i0, 1/i0, i0², 1/i0², i0³, 1/i0³}` in log space)
rather than the MPEG-1 `tan` grid — matching the decoder's §2.4.3.2
step-4/5 reconstruction. Long-block only (LSF is single-granule; the
short-window per-window bound stays deferred behind the same
`IntensityShortBlocksUnsupported` interlock). Validated by a new LSF
grid + degenerate-energy + range unit test and 6 integration tests
in `tests/lsf_intensity_roundtrip.rs` (constructor arming, wire
layout asserting `scalefac_compress = 258` + the (3,3,3,0)/(7,7,7,0)
partition, self-decode round-trip at ≈ 20.5 dB left-channel PSNR,
encode/decode determinism, the `'11'` MS+intensity combined mode,
and MPEG-2.5 at 11.025 kHz). `tests/lsf_encoder_roundtrip.rs`'s
`lsf_rejects_unported_features` rewritten: the intensity
constructors now build on LSF; only auto block-type still rejects.
Tests: 1047 lib (was 1046; +1 unit) + 6 integration.

**Phase 2 step 88 (r287)** — **LSF (MPEG-2 / MPEG-2.5) auto block-type**
(§C.1.5.2 over ISO/IEC 13818-3 §2.4.3.2), lifting the last
`LsfUnsupported` rejection — `Mp3Encoder::enable_auto_block_type` /
`enable_auto_block_type_with_mixed` now accept the LSF rates (16 /
22.05 / 24 kHz) and MPEG-2.5 rates (8 / 11.025 / 12 kHz). The
`attack_detect` / `block_type_sm` / `mixed_classifier` modules are
unchanged; the only edit is in `assemble_frame_with_lookahead`, whose
auto-path frame walk was hard-shaped to the MPEG-1 two-granule
geometry (classify gr0, gr1, lookahead; step the scheduler twice). It
is now generalised over `ngr ∈ {1, 2}`: the walk builds, per channel,
an attack flag for each of the frame's `ngr` granules plus one
lookahead granule (the next frame's leading granule held in
`per_ch_lookahead_pcm`, already version-agnostic in `push_samples` /
`finish`), then steps the scheduler `ngr` times — granule `g` is fed
`(attack[g], attack[g + 1])` so its §C.1.5.2 companion is the
following granule's flag (the lookahead for the frame's last granule).
On LSF (`ngr == 1`, one 576-sample granule per frame) this is a single
step with the next frame's granule as the lookahead; on MPEG-1
(`ngr == 2`) it reproduces the prior two-step walk byte-for-byte. The
lookahead granule is peeked non-destructively (the detector is cloned
so the zero-padded or borrowed next-frame PCM never perturbs the
ambient estimate; an empty lookahead at end-of-stream feeds
`next_attack = false` so the burst closes with a `Stop`). The
§2.4.3.4.10.3 window-switching geometry is version-invariant — only
the per-frame granule count differs — so no spec table changes are
needed. Independent stereo runs a per-channel detector + scheduler;
MS-stereo (§2.4.3.4.9) OR-folds the per-channel flags into a single
shared (channel-0) scheduler and mirrors its emission across both
channels of the granule, exactly as on MPEG-1. Validated by a new
`tests/lsf_auto_block_type_roundtrip.rs` (5 integration tests):
steady-sine stays Long; click-train engages Short and self-decodes;
the mixed-promotion variant; independent-stereo per-channel sequences;
and MS-stereo per-granule block-type / `window_switching_flag`
agreement. `tests/lsf_encoder_roundtrip.rs`'s
`lsf_rejects_unported_features` split into `lsf_auto_block_type_accepted`
+ `lsf_intensity_constructors_build`. Tests: 1047 lib + 6 new
integration (auto-block-type LSF) — no LSF feature now returns
`LsfUnsupported` on a supported channel layout.

**Phase 2 step 85 (r283)** — **§C.1.5.3.2.1 Layer III adaptation of
Model 2 + §D.2.4 step m) pre-echo control + §C.1.5.3.2 window
switching**, closing the "remaining Layer III adaptations" tail left
open by step 84. Read from the staged ISO PDF (printed pp.80–95 /
PDF pp.86–101; Tables C.7/C.8 and Figures C.6.a–d/C.7 transcribed
from 150-DPI page renders with 300–600-DPI re-reads of every cell
the PDF text layer disagreed on — the renders resolved C.7.a row 12
`bval = 5,437`, rows 57/58 `qthr = 22,607`, C.7.b row 47
`norm = 0,527`, and the replaced-spreading slopes `3,0(j−i)` /
`1,5(j−i)` verbatim). Tables (all in `psy`): the six **Table C.7**
threshold-calculation-partition tables (`Layer3PartitionLong` ×
62/63/59 rows with `FFT-lines`/`minval`/`qthr`/`norm`/`bval`;
`Layer3PartitionShort` × 38/39/42 rows with the constant `SNR (db)`
column) and the six **Table C.8** partition→scalefactor-band
conversion tables (`Layer3SfbConversion` × 21 long / 12 short rows
with `cbw`/`bu`/`bo`/`w1`/`w2`), with suffix→rate dispatchers
(C.7.a/C.8.a = 48 kHz — the *reverse* of the D.3 suffix order).
Model surface: dual-path constants (576/192 shifts, 256-point short
FFT via `model2_layer3_hann_window_short` /
`model2_layer3_step_a_reconstruct_short` /
`model2_layer3_step_b_spectrum_short`); the printed `cw(w)`
composition `model2_layer3_cw_compose` (long FFT lines 0–5, second
short block `(w+2) DIV 4` for 6–205, 0,4 above); `conv1/conv2`
(= step g) constants); `NMT = 6,0` / `TMN = 29,0` dB overrides in
`model2_layer3_step_h_snr_db`; the Figure C.6.b threshold
`model2_layer3_long_nb` (step i) sign convention documented against
the unsigned printed exponent) and the C.6.d short-path
`model2_layer3_short_nb` over the negative table SNR; **step m)**
`model2_layer3_step_m_thr` + `Model2Layer3PreEcho` (the printed
`thr = MAX(qthr, nbb, rpelev·nbb_l, rpelev2·nbb_ll)` with
`rpelev = 2` / `rpelev2 = 16` and two-block history rotation;
short path history-free per C.6.d); the psychoacoustic entropy
`model2_layer3_pe` (`−Σ cbwidth·ln(thr/(eb+1))`) with the verbatim
PE > 1800 switch (`layer3_pe_attack`), the Figure C.7 state diagram
(`layer3_window_state_next`), the Figure C.6.a one-block-delay
retrofit (`layer3_retrofit_start`) and the composed
`Layer3WindowSwitcher` (PE in → delayed block type out); the Figure
C.6.c reduction `layer3_partitions_to_sfb` + `layer3_sfb_ratio`;
and `Model2Layer3State::process` running the whole dual-FFT walk
per 576-sample granule (long ratio over 21 bands + 3 × 12 short
subblock ratios + PE + attack flag). 14 new unit tests: row counts
+ printed width sums (481/465/488 long, 107/109/110 short within
the 513/129-line domains); bval monotonicity + column domain
membership; render-anchor spot pins (incl. every text-layer
disagreement); C.8 `bu`/`bo` tiling chains against the C.7 row
counts; the `cw` composition mapping with boundary lines 6/9/10/205
/206; SNR/`nb` conventions (short ≡ long with the sign pre-baked);
the step m) maximum + 2×/16×/expiry history walk; the PE zero/sign/
width-scaling anchors + strict-1800 boundary; all eight printed
state-diagram arrows + the start-short-short-stop burst; the
delayed switcher; partition accumulation/reduction/ratio edge
cases; short-window symmetry + DC spectrum; impulse spreading at
±1 partition; and an end-to-end silence/tone/3-rate
`Model2Layer3State` walk (silence keeps `en = 0`, `thm > 0`,
no attack). Tests: 1038 lib (was 1024; +14). Only the staged ISO PDF and
`docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md` were read.

**Phase 2 step 84 (r282)** — Annex D Model 2 **§D.2.1 inputs +
§D.2.4 steps a)–e)** — the FFT-side front half of the Model 2
threshold calculation, completing the §D.2.4 chain end-to-end
against the r279–r281 back half. Read from the staged ISO PDF
(printed pp.128–130 / PDF pp.134–136; fresh 150-dpi renders of the
three prose/equation pages, which are images-only in the text
layer). Public surface (all in `psy`): `MODEL2_FFT_LEN` (1024) +
`MODEL2_FFT_LINES` (513, the §D.2.2 DC..Nyquist line domain) +
`model2_iblen_in_range` (the verbatim strict `384<iblen<640`
§D.2.1 shift-length constraint); step a)
`model2_step_a_reconstruct` (keep `1024−iblen` previous samples,
concatenate the `iblen` newest); step b) `model2_hann_window` (the
1-based half-sample-offset raised cosine
`0,5 − 0,5·cos(2π(i−0,5)/1024)` — no Model 1 `sqrt(8/3)` power
prefactor) + `Model2Polar` + `model2_step_b_spectrum` (window →
unnormalized forward FFT → polar `r_ω`/`f_ω`, normalization
absorbed by the step l) `model2_absthr_energy` calibration
parameter per the printed "after considering the FFT normalization
actually used"); step c) `model2_step_c_predict` /
`model2_step_c_predict_polar` (`x̂_ω = 2,0·x_ω(t−1) − x_ω(t−2)`,
branch-cut-safe through the step d) cos/sin consumption); step d)
`model2_step_d_cw` / `model2_step_d_cw_lines` (the unpredictability
measure with documented `0/0 → 0` silent-line convention) +
`MODEL2_CW_ABOVE_LIMIT` (the verbatim 0,3 partial-calculation
default above the caller's line limit); step e) `model2_step_e_eb`
/ `model2_step_e_cb` (`e_b = Σ r_ω²`, `c_b = Σ r_ω²·c_ω` over the
Table D.3 partitions). Integration: `Model2State` (the §D.2.1
zeroed "known starting point" — preceding source window + `t−1`/
`t−2` polar spectra) with a `smr()` full walk chaining a)–l) + n)
into the existing back-half entry points and returning the 32
`SMR_n` Table D.5 outputs per call, advancing state only on
success. 13 new unit tests: strict iblen bounds;
reconstruction concatenation + domain rejections; window symmetry
/ no-zero-endpoint / `Σw = 512` DC and `Σw² = 384` power anchors;
DC-block and bin-exact-sine spectra (`r = 512` / `256` with
Hann ±1-line leakage at exactly half peak); prediction
extrapolation; the four `c_ω` endpoints (0 / 1 / opposite-phase 1
/ silent 0); the 0,3 above-limit fill; partition energy
conservation `Σe_b = Σr_ω²` at all three rates (+ `c_ω ≡ 1 ⇒
c_b = e_b`); short/mismatched-slice rejections; a full-walk
bridge test replaying `Model2State::smr` call 3 value-for-value
against a by-hand a)→n) primitive chain; a steady bin-exact
sinusoid driving tonality high with pinned positive SMR at the
tone's coder partition; and failed-call state-untouched +
silence-is-a-fixed-point semantics. Tests: 1024 lib (was 1009;
+15). The Model 2 §D.2.4 chain for Layers I/II is now complete;
the remaining Layer III adaptations (§C.1.5.3.2.1 window
switching, step m) pre-echo control) stay open. Only the staged ISO PDF and the in-tree
r279–r281 surfaces were read.

**Phase 2 step 83 (r281)** — Annex D Model 2 **§D.2.4 steps h)–l)
and n)** — the entire back half of the Model 2 threshold
calculation, from required SNR down to the `SMR_n` output vector
the coder consumes. Read from the staged ISO PDF (printed
pp.131–132; the step-i/k/l/n equation blocks were re-read from
fresh 200-dpi renders of those two pages since they are
images-only in the text layer). Public surface (all in `psy`,
matching the step-f/g slice conventions — `Option` on length
mismatch, scalar core + slice driver): step h) `MODEL2_NMT_DB`
(= 5,5 dB), `model2_step_h_snr_db` (`SNR_b = maximum(minval_b,
tb_b·TMN_b + (1−tb_b)·NMT_b)`) + `model2_step_h_snr` reading the
`minval`/`TMN` columns straight off the r280
`Model2PartitionEntry` rows; step i) `model2_step_i_bc`
(`bc_b = 10^(−SNR_b/10)`); step j) `model2_step_j_nb`
(`nb_b = en_b·bc_b`); step k) `model2_step_k_nb_lines`
(`nb_ω = nb_b/(ωhigh_b−ωlow_b+1)`, emitting the 1-based line
domain as a 513-entry vector with pinned energy conservation);
step l) `model2_absthr_energy` (the spec's
"converted into the energy domain after considering the FFT
normalization actually used" as an explicit ±½-lsb-sine
calibration parameter), `model2_step_l_thr` +
`model2_step_l_thr_lines` (`thr_ω = max(nb_ω, absthr_ω)`;
D.4-uncovered lines — the D.4a line-58 gap and the >480/464/428
tails — pass `nb_ω` through via the documented `absthr_ω = 0`
convention); step n) `model2_step_n_epart` (`Σ r_ω²` over a Table
D.5 span), `model2_step_n_npart` (the printed `width_n` split:
narrow → `Σ thr_ω`, wide → smallest-**positive** argument ×
line count, with a documented `0` convention when no argument is
positive), `model2_step_n_smr_db` (`10·log10(epart/npart)`) and
the `model2_step_n_smr` driver over `coder_partition_d5_spans()`
(32 SMRs, partitions 1..=32). Step m) (pre-echo) is prose-noted
as Layer-III-only ("omitted for Layers I and II"). 11 new unit
tests: step h) TMN↔NMT interpolation endpoints + the `minval`
floor + the slice form against the first three D.3a rows; step i)
0/10/20 dB anchors + monotonicity; step j) elementwise product;
step k) per-line spread on the full D.3a table (single-line
partition 1, 3-line partition 2, length 513, Σnb_ω = Σnb_b);
step l) max in both orders + dB→energy anchors + the uncovered-line
pass-through; step n) epart squares on span 1 (17 inclusive
lines), npart width split incl. the smallest-positive skip and
the all-nonpositive `0` convention, the 0/10 dB SMR anchors and
the 32-entry uniform-input driver; plus an end-to-end h)→l)
chain over the full 32 kHz tables (uniform `en_b = 1`, `tb_b = 0`)
pinning `nb_1 = 10^(−0,55)` and the D.4a 58,23 dB floor on line
1. Tests: 1009 lib (was 998; +11 unit). Remaining Model 2 gap is
now exactly the FFT-side front half — steps a)–e) (1024-sample
reconstruction, Hann + FFT + polar, `r̂`/`f̂` prediction, the
unpredictability measure `c_ω`, and the partitioned `e_b`/`c_b`
sums). Only the staged ISO
PDF (printed pp.127–132) and the in-tree r279/r280 surfaces were
read.

**Phase 2 step 82 (r280)** — Annex D Model 2 **Tables D.3a–c**
(calculation partition table) + **Tables D.4a–c** (absolute threshold
table), transcribed in full from the staged renders
`docs/audio/mp3/annex-d-renders/Table-D.3*.png` / `Table-D.4*.png`
(printed pp.133–138) — the Model 2 table material the step-81
reductions were parameterized for. D.3a/b/c carry 49 / 57 / 58
partitions × (ωlow, ωhigh, bval, minval, TMN) with exact contiguous
coverage of FFT lines 1…513 (the docs extracts file's "63 partitions
at 32 kHz" prose is an erratum — the printed D.3a ends at Index 49
with ωhigh = 513); D.4a/b/c carry 132 / 130 / 126 line-range rows
covering lines 1…480 / 1…464 / 1…428. Public surface:
`Model2PartitionEntry`, `MODEL2_PARTITION_D3A…D3C`,
`model2_partition_table(fs)` (no Layer dimension — the Model 2 tables
are rate-only), `model2_bval(fs)` (feeds `model2_step_f_spread` /
`model2_step_f_rnorm` directly), `model2_partition_index_for_line`,
`Model2AbsThrEntry`, `MODEL2_ABSTHR_D4A…D4C`, `model2_absthr_table(fs)`
and `model2_absthr_for_line`. Printed-spec quirks kept verbatim and
pinned by tests: D.4a's `57 | 57` row followed by `59 | 60` (line 58
uncovered — its 0,55 dB equals Table D.1d's LTq at line 58, so the
cell is almost certainly a misprint for 58); D.4c's lone 4-line group
`329 | 332` inside the 8-line tail. Transcription is cross-validated
against the r278 Layer II Tables D.1 on the shared 1024-point FFT
line grid: every shared line agrees at 32 / 48 kHz except D.4a's last
row (51,03 vs 51,04), while at 44,1 kHz the printed D.4b
systematically diverges from the printed D.1e — 14 shared lines print
exactly 0,01 dB lower and the saturation plateau prints 69,13 dB vs
D.1e's 68,00 dB (sample cells of both sides re-verified at 300–400 %
zoom; the full 26-entry exception list is pinned). 9 new unit tests:
D.3 lengths + contiguous 1…513 coverage, column well-formedness
(strict `bval` ascent, printed `minval` value set, non-decreasing
TMN), spot rows, `bval`-vs-D.1-Bark consistency, `model2_bval`
extraction, partition lookup bounds, D.4 lengths/coverage with the
printed quirks, absthr lookup incl. the line-58 gap, and the D.4↔D.1
cross-table agreement. Tests: 998 lib (was 989; +9 unit). Next Model
2 steps: h) required SNR per partition (`minval`/`TMN` now in-tree),
i)–l) power ratio / energy threshold / line spread /
absolute-threshold floor (D.4 now in-tree), then b)–e) FFT-side
inputs. Only the six staged
D.3/D.4 renders, the D.1e render (cross-check), and the extracts file
were read.

**Phase 2 step 81 (r279)** — Annex D Model 2 **§D.2.3 "The spreading
function"** + **§D.2.4 step f)** convolution / renormalization +
**step g)** tonality index — the first base-Model-2 increment (the
§C.1.5.3.2.1 Layer III *modified* spread landed back in step 48; this
is the unmodified Layers I/II function it replaces). §D.2.3 is read
from the staged ISO PDF (printed p.129) via the corrected extract in
`docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md` — the
envelope is `10^((x + tmpy)/10)` (an earlier docs revision dropped
the `x` term; the erratum fix is what unblocked this arc). Public
surface: `model2_sprdngf_tmpx` (`1,05·(j−i)` over *Bark values*, not
partition indices), `model2_sprdngf_x_db` (the parabolic correction,
non-zero only on the near-upward skirt `0,5 < tmpx < 2,5`, floor −8
dB at `tmpx = 1,5`), `model2_sprdngf_tmpy_db` (the asymmetric
envelope — exactly 0 dB on the diagonal because the printed
`15,811389` equals `17,5·sqrt(1+0,474²) − 7,5·0,474`; ≈ −10 dB/tmpx
upward vs ≈ −25 dB downward), `model2_sprdngf` (with the verbatim
`tmpy < −100` → 0 cutoff, applied to `tmpy` alone, zeroing spreads
beyond ≈ 4,8 Bark down / ≈ 10,5 Bark up), `MODEL2_SPRDNGF_TMPY_CUTOFF_DB`,
`model2_step_f_spread(per_partition, bval)` (one reduction serving
both printed convolutions `ecb_b` and `ct_b`; `bval` caller-injected
— the full Tables D.3a–c stay a PNG transcription gap),
`model2_step_f_rnorm` (reciprocal row sum; the printed `bb=0` lower
bound vs D.2.2's "partition numbering starts at 1" is noted — the
slice API sums every provided partition, satisfying both readings),
`model2_step_f_cb` (`ct_b/ecb_b`, documented `0` convention at
`ecb_b = 0`), `model2_step_f_en` (`ecb_b·rnorm_b`), and
`model2_step_g_tonality` (`−0,299 − 0,43·ln(cb_b)` clamped to
`[0, 1]`). 12 new unit tests: diagonal-unity within the printed
constant's rounding, parabola active-region + −8 dB floor,
hand-substituted 1-Bark-up value, cutoff at ±far spreads,
upward-reach > downward-reach asymmetry, impulse convolution
recovers the `sprdngf` row (on the Table D.3a 20-row `bval` text
anchor), uniform-energy `en ≈ 1` renormalization identity,
constant-`c_ω` recovery through `cb`, zero-energy convention,
tonality clamps at both ends + formula/monotonicity. Tests: 989 lib
(was 977; +12 unit). Only the
staged ISO PDF pages (printed 127–132 prose + tables) and the
extracts/anchor transcriptions in
`docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md` were read.
Next Model 2 steps: h) required SNR per partition (needs the
`minval`/`TMN` columns — full D.3 tables are PNG-only), i)–l) power
ratio / energy threshold / line spread / absolute-threshold floor
(D.4 tables PNG-only), then b)–e) FFT-side inputs.

**Phase 2 step 80 (r278)** — Annex D **Tables D.1a–f** (frequencies,
critical band rates and absolute thresholds) + the **Step 4 → Bark
bridge** + the end-to-end **§D.1 Step 5 sieve**. The six per-(Layer,
Fs) Table D.1 pages — D.1a/b/c Layer I (108/106/102 rows, printed
p.116–118), D.1d/e/f Layer II (132/130/126 rows, printed p.119–121) —
are transcribed in full (all 704 rows × 3 columns) from the staged
renders `docs/audio/mp3/annex-d-renders/Table-D.1*.png`, read at high
magnification in cropped strips with every ambiguous cell re-read in a
dedicated zoom. Public surface: `Model1ThresholdEntry { frequency_hz,
z_bark, ltq_db }`, the `MODEL1_THRESHOLD_D1A…D1F` constants, the
`model1_threshold_table(layer, fs)` dispatcher, the subsampling maps
`model1_d1_line_for_index` / `model1_d1_index_for_line` (rows 1…48 are
FFT lines 1…48, rows 49…72 every 2nd line 50…96, rows 73…96 every 4th
line 100…192 — Layer I continues this region to its end — and Layer II
rows 97… every 8th line 200…; nearest-entry inverse with documented
tie-down), `model1_d1_entry_for_line`, the bridge
`model1_masker_from_component` (lifts an r277 `Model1Step4Component`
onto the Table D.1 Bark grid), and `model1_step5_components(tonal,
non_tonal, layer, fs)` — the spec's full Step 5: bridge + Step 5(a)
threshold-in-quiet screen against the same row's `Absolute Thresh.`
column + Step 5(b) 0,5-Bark tonal decimation, emitting `Masker`s ready
for the Step 6/7 evaluators. **Model 1 Steps 1–7 now compose
end-to-end** (pinned by a chain test: sine → Step 1 FFT → 96 dB
normalize → Step 4 classification → Step 5 sieve → Step 6/7 global
threshold). Transcription integrity is pinned by structural
redundancy: the printed frequency column equals the line grid
`k·Fs/N` for all 704 rows; z is strictly increasing per table; every
Layer I row reprints in the same-Fs Layer II table (line L ↔ 2L) with
identical z/LTq; and every Tables D.2 boundary row's frequency/Bark
pair equals the Table D.1 row its `index F&CB` cites. The
cross-checks resolved the **D.2e band-17 illegible Bark digit** —
D.1e row 62 legibly prints `16,110` (= D.1b row 38), so the stored
`16.11` is exact and the docs file's prose estimate `16,116` is wrong
— and surfaced a **systematic spec print inconsistency**: at 44,1 kHz
the D.2 tables print exactly 0,001 Bark below the D.1 tables at three
frequencies (D.2b 17/20/24 = D.2e 19/22/26: `17,904`/`20,971`/`24,573`
vs the double-printed D.1 `17,905`/`20,972`/`24,574`); both verbatim
prints are kept and the exception list is pinned in
`table_d1_agrees_with_d2_boundary_rows`. 10 new unit tests (the five
integrity sweeps above, the seven textual LTq anchors = rows 1–5 / 51
/ 108 of D.1a, round-trip + nearest/tie/bounds cases for the index
maps, bridge placement + out-of-band/Layer III refusals, a synthetic
screen-and-decimate end-to-end, and the Steps 1–7 chain). Tests: 977
lib (was 967; +10 unit). Only
the six Table D.1 PNG renders plus the in-repo Tables D.2
transcription were read.

**Phase 2 step 79 (r277)** — Annex D Model 1 §D.1 **Step 4 "Finding
of tonal and non-tonal components"**: the tonality classifier that
turns the step-77/78 SPL spectrum into the discrete masker lists that
Steps 5–7 (landed r229/r219) consume. Eight public items in `psy`:
`model1_step4_is_local_maximum(x, k)` is the verbatim operation (a)
label (`X(k) > X(k-1) and X(k) >= X(k+1)` — strict low side,
non-strict high side; `None` at neighbourless edges);
`model1_step4_tonal_check_offsets(layer, k)` transcribes the verbatim
layer/k-range `j` table (`j = ±2` for `2 < k < 63`, `±2, ±3` for
`63 <= k < 127`, `±2…±6` for `127 <= k <= 250` Layer I / `< 255`
Layer II, `±2…±12` for `255 <= k <= 500` Layer II; `None` outside the
listed ranges and for Layer III — the D.1 preamble adapts the Layer II
1 024-point model, so Layer III callers pass `LayerII` exactly as with
`critical_band_boundaries`); `model1_step4_is_tonal(x, layer, k)`
applies the operation (b) `X(k) − X(k+j) >= 7 dB` test
(`MODEL1_STEP4_TONAL_DELTA_DB`) conjunctively over the whole offset
set; `model1_step4_tonal_spl_db(x, k)` is the verbatim three-line
power sum `X_tm(k) = 10·log10(10^(X(k−1)/10) + 10^(X(k)/10) +
10^(X(k+1)/10))`; `model1_step4_extract_tonal(&mut x, layer)` scans
the examined ranges in ascending `k`, lists each passing line as a
`Model1Step4Component` (index `k`, SPL, tonal flag — the spec's three
listed parameters), and applies the verbatim "all spectral lines
within the examined frequency range are set to -∞ dB" zeroing over
`k ± j_max` (all decisions evaluate against the pre-zeroing spectrum:
operation (a) labels the maxima *before* any zeroing, so in-pass
zeroing must not manufacture candidates — close tonal pairs both list
and Step 5(b)'s 0,5-Bark decimation dedups them);
`model1_step4_band_line_spans(layer, fs)` maps the Tables D.2a–f
boundary rows onto raw step-77 line spans (`Model1Step4BandSpan`) via
`k = round(f·N/Fs)` from each row's exact `frequency [Hz]` column
(the `index F&CB` column addresses the subsampled Table D.1 domain
and cannot index the full-resolution spectrum operation (c) sums;
rows are inclusive band *tops* per the established
`band_of_fft_line` reading, band 0 starts at line 1, DC is in no
band); `model1_step4_non_tonal_components(x, layer, fs)` forms the
operation (c) per-critical-band residue power (the step-78 `Xspl`
power sum) listed at the geometric-mean line
`round(sqrt(k_first·k_last))` with the non-tonal flag (an all-zeroed
band yields a `-∞` dB component verbatim — Step 5(a) screens it
against LTq); and `model1_step4_components(x, layer, fs)` composes
the three operations end-to-end into `(tonal, non_tonal)` lists. The
remaining bridge to Steps 5–7's Bark-domain `Masker` carrier is the
line-index → `z(k)` mapping through Tables D.1a–f, which are still
PNG-only renders. 17 new unit tests: both layers' `j`-range
boundaries (including the 254/255 Layer II split and the ±1/0
exclusions), strict/non-strict local-maximum sides + edge rejections,
inclusive 7,0 dB margin vs 6,9 dB failure, single-offset veto in the
±12 top range, out-of-range `None`s, the three-line SPL identity with
step-78 `Xspl`, extraction listing + exact `k ± 3` zeroing with
untouched floor, a snapshot-vs-sequential discriminating case (a
blocker line inside an earlier peak's zeroed range must still veto a
later candidate), length/Layer III rejections, contiguous-tiling +
top-line anchors for all six (layer, Fs) span tables (240/232/216
Layer I, 480/464/432 Layer II), D.2d anchor rows (1/3/6, 297..368,
369..480), flat-spectrum per-band `10·log10(width)` non-tonal power +
geometric-mean placement (band 24 → line 421), geometric-mean-inside-
span across all tables, an end-to-end peak-plus-floor run (D.2e band
19 loses the zeroed peak energy, far bands bit-identical), a Step 1 →
normalize → Step 4 pure-tone chain (single above-floor tonal at
`96 + 10·log10(1,5) ≈ 97,76 dB`, all-residue bands below 0 dB), and
the silent-spectrum case (no tonal, all `-∞` non-tonal). Tests: 967
lib (was 950; +17 unit). Only
the §D.1 Step 4 prose (printed p.111–112) read directly from
`docs/audio/mp3/ISO_IEC_11172-3-MP3-1993.pdf` plus the in-repo Tables
D.2 transcription in
`docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`.

**Phase 2 step 78 (r276)** — Annex D Model 1 §D.1 **Step 2
"Determination of the sound pressure level"**: `Lsb(n)` from the
step-77 spectrum — the second of the "Steps 1–3" lacks items (and,
with step 77, the producer for the `lsb_per_partition` callback that
the step-70/71 SMR row-order vectors have consumed caller-injected
until now). Five public primitives in `psy`:
`model1_step2_scf_term_db(scf_max)` evaluates the verbatim
scalefactor term `20·log(scf_max(n)·32 768) − 10` dB
(`MODEL1_STEP2_FULL_SCALE = 32768.0`,
`MODEL1_STEP2_PEAK_RMS_CORRECTION_DB = 10.0` — "the '-10 dB' term
corrects for the difference between peak and RMS level"; `scf_max`
is caller-supplied per the verbatim Layer I scalefactor / Layer II
max-of-three rule); `model1_step2_lsb_db(x_subband_db, scf_max)` is
the verbatim outer `Lsb(n) = MAX[X(k), scf-term]` shared by both
Step 2 methods; `model1_step2_xspl_db(lines)` is the verbatim
alternative-method power sum `Xspl(n) = 10·log10(Σ 10^(X(k)/10))` dB
over caller-selected lines (`−∞` lines contribute zero; empty
selection → `−∞`); and the two Table D.5-driven subband selectors
`model1_step2_subband_max_line_db(x, n)` ("the spectral line … with
the maximum amplitude in the frequency range corresponding to
subband n") and `model1_step2_subband_xspl_db(x, n)`, which read
partition `n ∈ 1..=32`'s inclusive 1-based span `[ωlow_n, ωhigh_n]`
via the step-50 accessors and map it onto a 513-line step-77
half-spectrum through `k = ω − 1` (`None` for out-of-range `n` or
any other spectrum length; adjacent spans share their dual-role
boundary cell exactly as the D.5 column prints it). 8 new unit
tests: scf-term anchors (scf 1 → `20·log10 32768 − 10 ≈ 80,309`,
scf `1/32768` → exactly −10, doubling adds `20·log10 2`), outer-MAX
dominance both ways + silent-subband fallback, Xspl single-line
identity / equal-pair `+3,0103 dB` / silence-transparency /
empty-`−∞` anchors, subband-accessor rejections (n = 0 / 33, 257-line
spectrum), planted-interior-line recovery across all 32 D.5 spans,
shared-boundary-cell visibility from both adjacent subbands,
`Xspl ≥ max-line` across every partition of a real broadband step-77
spectrum, and an end-to-end Step 1 → normalize → Step 2 chain
(tone at line ω = 101 → partition 7, `Lsb = 96 dB` against a small
scalefactor, scf-term dominance against a huge one). Tests: 950 lib
(was 942; +8 unit). Only the
§D.1 Step 2 prose/formulas (printed p.110–111) read directly from
`docs/audio/mp3/ISO_IEC_11172-3-MP3-1993.pdf` plus the in-repo Table
D.5 transcription.

**Phase 2 step 77 (r276)** — Annex D Model 1 §D.1 **Step 1 "FFT
Analysis"**: the Hann-windowed power-density spectrum that Steps 2–9
consume — the first of the long-standing "Steps 1–3" lacks items.
Three public primitives in `psy`: `model1_hann_window(i, n)`
implements the verbatim window formula `h(i) = sqrt(8/3) · 0,5 ·
{1 − cos[2·π·i/N]}` for `0 <= i <= N−1` (`None` outside the domain;
the `sqrt(8/3)` prefactor makes the window unit-power, `Σ h(i)² = N`);
`model1_power_density_spectrum(s)` evaluates the verbatim
`X(k) = 10·log10 |(1/N)·Σ h(l)·s(l)·e^(−j·k·l·2π/N)|² dB` over the
spec's inclusive half-spectrum `k = 0…N/2` (so 513 lines for the
1 024-sample block — exactly the 1-based ω ∈ 1..=513 Table D.5
convention via `k = ω − 1` — and 257 for Layer I's 512), accepting
only the two spec transform lengths (`MODEL1_FFT_LEN_LAYER1 = 512`,
`MODEL1_FFT_LEN_LAYER2 = 1024`; Layer III adapts the 1 024 variant)
and returning `None` for any other length; and
`model1_normalize_to_96db_spl(&mut x)` applies the verbatim "maximum
value corresponds to 96 dB" reference shift
(`MODEL1_SPL_REFERENCE_DB = 96.0`) in place, returning the applied
offset, or `None` (slice untouched) when no finite maximum exists
(empty / all-silent `−∞` spectrum). The DFT kernel is a private
split-re/im radix-2 in-place FFT (pure standard mathematics, nothing
codec-specific). Silence maps to `−∞` dB lines verbatim (no invented
floor); the Step 1 PCM window-placement rules (256-sample subband
filter delay compensation, ±64-sample Hann/frame alignment) remain
caller responsibilities. 12 new unit tests: window domain rejections
+ `h(0) = 0` / `h(N/2) = sqrt(8/3)` endpoints, symmetry +
unit-power (`Σh² = N` within 1e−6), non-spec-length rejection,
inclusive half-spectrum lengths (257/513), the DC anchor
`X(0) = 10·log10(2/3)` for `s ≡ 1`, pure-tone anchors (peak
`10·log10(1/6)`, ±1 Hann sidelines exactly `10·log10 4 ≈ 6,02 dB`
down, no leakage beyond ±1), a radix-2 vs direct-DFT cross-check on
deterministic broadband blocks at both lengths (≤ 1e−8 dB), silent
block → all `−∞`, normalization max-pin + delta preservation +
identity-at-reference + refusal cases, and an end-to-end
window→FFT→normalize pipeline pinning the tonal peak at exactly
96 dB. Tests: 942 lib (was 930; +12 unit). Only the §D.1 Step 1 prose/formulas
(printed p.110) read directly from
`docs/audio/mp3/ISO_IEC_11172-3-MP3-1993.pdf`.

**Phase 2 step 76 (r275)** — Annex C §C.1.5.2.7 "Bit allocation"
**step-4 budget update + iterate/terminate test** — the fourth and
final action of every Layer II bit-allocation iteration, plus the loop's
continuation predicate. Steps 73–75 selected the minimal-MNR subband,
promoted its Table B.2 entry, and recomputed its MNR; this step closes
the iteration by folding the promotion's bit cost into the running
budget and recomputing the available-data-bits `adb`, then deciding
whether to loop again. New public structs `BitAllocBudget { bspl, bsel,
bscf, first_time, adb }` (the loop-mutated `bspl` / `bsel` / `bscf`
accumulators + derived `adb`) and `BitAllocOverhead { cb, bhdr, bcrc,
bbal, banc }` (the fixed per-frame `adb`-formula terms), and free
functions `bit_allocation_budget_update(prev, extra_sample_bits,
first_time, sel_bits, scf_bits, overhead) -> BitAllocBudget` and
`bit_allocation_should_iterate(adb, max_possible_increase) -> bool`. The
update applies the verbatim §C.1.5.2.7 rule "bspl is updated according
to the additional number of bits required. If a non-zero number of bits
is assigned to a subband for the first time, bsel has to be updated, and
bscf has to be updated according to the number of scalefactors required
for this subband" (`bspl += extra_sample_bits`; on a `first_time`
promotion also `bsel += sel_bits`, `bscf += scf_bits`), then recomputes
`adb = cb − (bhdr + bcrc + bbal + bsel + bscf + bspl + banc)` (saturating
at zero). The predicate is the verbatim termination sentence "The
iterative procedure is repeated as long as adb is not less than any
possible increase of bspl, bsel and bscf within one loop" — `adb >=
max_possible_increase`. The per-entry sample-bit / scalefactor-bit costs
(Tables B.2 / B.4) and the fixed overhead terms are caller-injected (the
tables are behind the numeric-table transcription gap), the
dependency-injection pattern the surrounding Phase 2 steps use; no spec
arithmetic is introduced beyond the accumulator additions, the `adb`
subtraction, and the `>=` comparison. Tests: 930 lib (was 918 baseline;
+12 unit) covering the non-first-time bspl-only growth with bsel/bscf
carried through, first-time growth of all three accumulators, the
verbatim `adb` formula (both first-time and not), adb saturation on
overcommit, zero-extra-bits accumulator idempotence, determinism, a
three-iteration threading chain with monotone-shrinking adb, the
not-less-than boundary (`adb == increase` continues), termination below
the increase, the zero-increase trivial-continue case, and an
end-to-end budget-then-iterate exhaustion. Only the §C.1.5.2.7 "bspl is
updated … Then adb is calculated again" loop step, its `adb` formula,
and the "repeated as long as adb is not less than any possible increase"
termination sentence (ISO/IEC 11172-3:1993 Annex C §C.1.5.2.7, printed
p.74, in `docs/audio/mp3/ISO_IEC_11172-3-MP3-1993.pdf`) were read. No
external implementation was consulted.

**Phase 2 step 73 (r272)** — Annex C §C.1.5.2.7 "Bit allocation"
**minimal-MNR subband selection** — the first action of every Layer I /
Layer II bit-allocation iteration. Phase 2 step 72 (r271) landed the
per-partition `MNR_n = SNR_n − SMR_n` row-order vector; this step
performs the loop's opening move, "Determination of the minimal MNR of
all subbands" (printed p.71, verbatim), reducing the 32-row vector to
the single subband "that has the greatest benefit", which the loop then
promotes to the next-higher quantization-accuracy entry. New public
struct `CoderPartitionD5MinMnr { partition_n: u16, mnr_db: f64,
smr_db: f64, width_n: u16 }` and free function
`coder_partition_d5_min_mnr(mnr: &[CoderPartitionD5Mnr; 32]) ->
CoderPartitionD5MinMnr`: a row-order argmin scan over step 72's
`coder_partition_d5_mnr_row_order` output, returning the winning
partition's 1-based index `n ∈ 1..=32` alongside its `mnr_db` /
`smr_db` / `width_n` columns carried through verbatim. Ties resolve to
the lowest partition index — the spec selects "the" subband (singular),
so a deterministic tie-break is required and the row-order scan keeps
the first occurrence, matching the order in which the §C.1.5.2.7 loop
walks Table D.5. `NaN` rows are skipped (a `NaN` `mnr_db` never compares
`<` the running minimum, so an all-`NaN` vector retains the `n = 1`
seed). No spec arithmetic is introduced beyond the `<` comparisons of
the scan. Tests: 900 lib (was 891 baseline; +9 unit) covering the
unique-minimum selection under a −30 dB interior-line LTg dip (with
verbatim winning-row echo), low/high partition-index minima
(`SNR(n) = n` → `n = 1`; `SNR(n) = 100 − n` → `n = 32`), the
lowest-index tie-break on an all-equal MNR vector, width/SMR-column
pass-through of the winner in a `width_n = 1` partition, negative-MNR
selection (signal needing more bits than current quantization gives),
`NaN`-row skipping, idempotence, and a brute-force argmin cross-check.
Steps 1–2 (FFT/SPL) and Tables D.1 / D.2 / C.5 stay numeric-table-
blocked (#1262/#1538); only the §C.1.5.2.7 "Determination of the minimal
MNR of all subbands" loop step (ISO/IEC 11172-3:1993 Annex C, printed
p.71, in `docs/audio/mp3/ISO_IEC_11172-3-MP3-1993.pdf`) and the Phase 2
step 72 `coder_partition_d5_mnr_row_order` vector it consumes (and
through it the Table D.5 transcription in
`docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`) were read. No
external implementation was consulted.

**Phase 2 step 72 (r271)** — Annex C §C.1.5.2.7 "Bit allocation"
per-partition **mask-to-noise ratio `MNR_n = SNR_n − SMR_n`** row-order
vector over Table D.5. Phase 2 step 71 (r270) exposed the §D.1 Step 9
paired `(SMR_n, width_n)` vector — "the output of the psychoacoustic
model" the §C.1.5.2.7 bit-allocation loop consumes. This step lands the
very first arithmetic of that iterative procedure: the per-subband `MNR`
initialisation `MNR = SNR − SMR` (printed p.73, verbatim), computed once
per partition before the loop's level-bumping ("increase the accuracy of
the subband that has the greatest benefit") begins. New public struct
`CoderPartitionD5Mnr { mnr_db: f64, smr_db: f64, width_n: u16 }` and free
function `coder_partition_d5_mnr_row_order<S: Fn(u16) -> f64, L: Fn(u16)
-> f64, F: Fn(u16) -> f64>(snr_per_partition, lsb_per_partition,
ltg_per_line) -> [CoderPartitionD5Mnr; 32]`: a pure per-row subtraction
of a caller-supplied `SNR_n` from Phase 2 step 71's
`coder_partition_d5_smr_row_order` SMR column, carrying the `smr_db` and
`width_n` columns through verbatim. The `SNR_n` term is the Table C.5
"Layer II Signal-to-Noise Ratios" column — behind the same numeric-table
transcription gap as Tables D.1 / D.2 — so it is caller-injected, the
same dependency-injection pattern Phase 2 steps 58–71 use for the §D.1
Step 2 `Lsb(n)` term. No spec arithmetic is introduced beyond the
verbatim `SNR − SMR` subtraction; the `smr_db` column is bit-identical
to step 71's, the `width_n` column to step 60's (`[0×12, 1×20]`). Tests:
891 lib (was 880 baseline; +11 unit) covering the 32-row length,
zero-callback all-zero MNR, uniform pin (30 − 76 = −46.0 exact),
cell-wise `MNR = SNR − SMR` against the step-71 SMR column, SMR/width
column pass-through from step 71, the `[0×12, 1×20]` width literal,
partition-index mapping (`SNR(n) = n`, flat SMR → `mnr_db[i] = i + 1`),
the §C.1.5.2.7 "greatest benefit" = unique minimum-MNR argmin under a
−30 dB interior-line LTg dip, SNR fan-out once-per-partition ascending,
sign semantics in both directions (needs-bits vs already-protected), and
idempotence for pure callbacks. Steps 1–2 (FFT/SPL) and Tables D.1 / D.2
/ C.5 stay numeric-table-blocked (#1262/#1538); only the staged
ISO/IEC 11172-3:1993 spec PDF (Annex C §C.1.5.2.7 printed p.73; §D.1
Step 9 printed p.115) and the Phase 2 step 71 row-order accessor (and
through it the Table D.5 transcription in
`docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`) were read.

**Phase 2 step 71 (r270)** — Annex D Model 1 §D.1 Step 9 **paired
`(SMR_n, width_n)` row-order** vector over Table D.5. Phase 2 step 70
(r269) exposed the bare row-order signal-to-mask-ratio vector
`[SMR_1 … SMR_32]` (dB); the Layer I / Layer II bit-allocation loop,
however, reads each partition's `SMR_n` **paired with** its `width_n`
column flag in lockstep at every row (the SMR value seeds the
partition's mask-to-noise iteration; the `width_n` flag drives
single-line vs multi-line per-partition bit targeting) — exactly the
way the §D.1 Step 8 paired `(LTmin_n, width_n)` vector (Phase 2 step
61) is read. New public struct `CoderPartitionD5Smr { smr_db: f64,
width_n: u16 }` and free function `coder_partition_d5_smr_row_order<L:
Fn(u16) -> f64, F: Fn(u16) -> f64>(lsb_per_partition, ltg_per_line) ->
[CoderPartitionD5Smr; 32]` supply that paired presentation: a pure
index-aligned zip of step 70's row-order SMR vector
(`coder_partition_d5_smr_db_row_order`) with step 60's static
row-order width vector (`coder_partition_d5_width_row_order`). No spec
arithmetic is introduced beyond the step 70 subtraction already in the
SMR column — only the per-row pairing of the two existing columns at
the same array index. The `smr_db` column is bit-identical to step
70's output and the `width_n` column to step 60's
(`[0×12, 1×20]`); `Lsb(n)` stays the steps-58–70 caller-callback
injection (§D.1 Steps 1–2 remain behind the PNG-only Tables D.1 / D.2
transcription gap). Tests: 880 lib (was 868 baseline; +12 unit)
covering the 32-pair length, zero-callback all-zero SMR, uniform pin
(96 − 20 = 76.0 exact), cell-wise SMR equality with step 70, width
equality with step 60 across two callbacks, the `[0×12, 1×20]` width
literal, partition-index mapping (`Lsb(n) = n`, flat threshold →
`smr_db[i] = i + 1`), sign semantics in both directions, the step-61
paired-pattern cross-check (matching `width_n` and
`lsb(n) − step61.ltmin_db` per row), Lsb fan-out once-per-partition
ascending with LTg fan-out equal to one step-59 pass, a −30 dB
interior-line LTg dip (ω = 300, partition via the step 56 inverse
lookup) raising exactly one row's SMR by +30 dB with widths and all 31
other rows unchanged, and idempotence for pure callbacks. Only the staged ISO/IEC 11172-3:1993 spec
PDF (§D.1 Step 9, printed p.115) and the Phase 2 step 70 / step 60
row-order accessors (and through them the cascade down to the Table
D.5 transcription in
`docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`) are read.

**Phase 2 step 70 (r269)** — Annex D Model 1 §D.1 Step 9 **row-order**
signal-to-mask-ratio vector over Table D.5. Phase 2 step 69 (r268)
landed the Step 9 subtraction `SMR_n = Lsb(n) − LTmin_n` dB
(ISO/IEC 11172-3:1993 Annex D, printed p.115) in the width-gated
split presentation (12 narrow + 20 wide subarrays); the Layer I /
Layer II bit-allocation loop, however, walks the 32 coder partitions
of Table D.5 **in row order**, pairing each partition's `SMR_n` with
the same row's `width_n` flag (Phase 2 step 60's `[u16; 32]`) and
`LTmin_n` value (Phase 2 step 59's `[f64; 32]`) at the same array
index. New free function `coder_partition_d5_smr_db_row_order<L:
Fn(u16) -> f64, F: Fn(u16) -> f64>(lsb_per_partition, ltg_per_line)
-> [f64; 32]` supplies that missing row-order presentation: element
`i` carries `SMR_{i + 1}`, the `LTmin_n` operand comes from one
Phase 2 step 59 pass (`coder_partition_d5_ltg_min_row_order`), and
the `Lsb(n)` operand stays the steps-58–69 caller-callback injection
(§D.1 Steps 1–2 remain behind the PNG-only Tables D.1 / D.2
transcription gap). No new spec arithmetic beyond the step 69
subtraction. Because step 63's by-width `LTmin_n` cells are
index-preserving copies of the step 59 row-order vector (via the
step 61 / 62 chain), the output is **bit-identical** to step 69's
split read back in row order (`out[0..12] == narrow_band`,
`out[12..32] == wide_band`) — pinned by an exact-`==` test under
non-trivial callbacks. Tests: 868 lib (was 859 baseline; +9 unit)
covering zero-callback all-zero rows, uniform pin (96 − 20 = 76.0
exact), cell-wise equality with the independently reconstructed
`lsb(n) − step59` difference, partition-index mapping (`Lsb(n) = n`,
flat threshold → `out[i] = i + 1`), sign semantics in both
directions, dual callback fan-out (Lsb exactly `[1..=32]` ascending;
LTg equal to a directly-counted step-59 pass), the step 69
bit-identity, a −30 dB interior-line LTg dip (ω = 300, partition via
the step 56 inverse lookup) raising exactly one row's SMR by +30 dB
with all 31 other rows unchanged, and idempotence for pure
callbacks. Only the staged
ISO/IEC 11172-3:1993 spec PDF (§D.1 Steps 2 / 8 / 9, printed
pp.110/114/115) and the Phase 2 step 59 row-order reducer (and
through it the cascade down to the Table D.5 transcription in
`docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`) are read.

**Phase 2 step 69 (r268)** — Annex D Model 1 §D.1 **Step 9**
width-gated signal-to-mask-ratio over Table D.5. The Step 9 formula
(ISO/IEC 11172-3:1993 Annex D, printed p.115) is
`SMR_sb(n) = Lsb(n) − LTmin(n)` dB, computed for every subband `n` —
the per-band input the Layer I / Layer II bit-allocation loop seeds
its mask-to-noise iteration from. This step had been carried as
"docs-blocked on the §D.1 SMR formula"; r268 verified the formula is
directly derivable from the staged spec PDF
(`docs/audio/mp3/ISO_IEC_11172-3-MP3-1993.pdf`, §D.1 Step 9 printed
p.115 + Step 8 `LTmin(n) = MIN[LTg(i)]` printed p.114), closing the
gap without any new commission. The `LTmin_n` operand comes from the
Phase 2 step 58–63 chain (one step-63 invocation of
`coder_partition_d5_ltmin_db_row_order_by_width`); the `Lsb(n)`
operand is the §D.1 Step 2 sound pressure level
(`MAX[X(k), 20·log10(scf_max(n)·32768) − 10]` dB, printed p.110),
supplied as a caller callback because Steps 1–2 (FFT + SPL) remain
behind the PNG-only Tables D.1 / D.2 transcription gap — the same
dependency-injection pattern steps 58–68 use for `LTg(ω)`. New free
function `coder_partition_d5_smr_db_row_order_by_width<L: Fn(u16) ->
f64, F: Fn(u16) -> f64>(lsb_per_partition, ltg_per_line) ->
CoderPartitionD5SmrByWidth { narrow_band: [f64; 12], wide_band:
[f64; 20] }` — the only new spec arithmetic is the Step 9
subtraction itself, one `Lsb(n) − LTmin_n` per row, presented in the
step 63 width-gated split (narrow partitions 1..=12, wide 13..=32).
`lsb_per_partition` is invoked exactly once per partition `n ∈
1..=32` in ascending row order; `ltg_per_line` fan-out is exactly
one step-63 pass. Signs preserved without clipping (positive =
audible content needing bits; negative = fully masked). Tests: 859
lib (was 851 baseline; +8 unit) covering zero-callback all-zero
cells, uniform pin (96 − 20 = 76.0 exact), cell-wise equality with
the independently reconstructed `lsb(n) − step63` difference under
non-trivial callbacks, partition-index mapping (`Lsb(n) = n`, flat
threshold → `narrow[i] = i + 1`, `wide[j] = j + 13`), sign
semantics in both directions, dual callback fan-out (Lsb exactly
`[1..=32]` ascending; LTg equal to a directly-counted step-63
pass), a −30 dB interior-line LTg dip (ω = 300, partition located
via the step 56 inverse lookup) raising exactly one wide cell's SMR
by +30 dB with all 31 other cells unchanged, and idempotence for
pure callbacks. Only the
staged ISO/IEC 11172-3:1993 spec PDF (§D.1 Steps 2 / 8 / 9, printed
pp.110/114/115) and the Phase 2 step 63 width-gated dB accessor
(and through it the cascade down to the Table D.5 transcription in
`docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`) are read.

**Phase 2 step 68 (r267)** — Annex D Model 1 §D.1 Step 8 width-
gated paired `(narrow_total, wide_total)` signed bit-budget
reduction over Table D.5 with a **single** step-65 invocation.
Phase 2 step 66 (r265) exposed the wide-band weighted total
`Σ_{n=1..=32} width_n · log2(LTmin_lin_n)` (collapsing onto the
unweighted sum of step 65's `wide_band` subarray); Phase 2 step
67 (r266) exposed the complementary narrow-band total
`Σ_{n=1..=32} (1 − width_n) · log2(LTmin_lin_n)` (collapsing onto
`Σ narrow_band`). The two reductions partition the full row-order
`Σ_n log2_n` exactly. Several Step 9 / Step 10 consumers read
**both** totals together, and calling step 66 + step 67
back-to-back invokes the caller's `LTg(ω)` callback **twice** over
the full `Σ_{n=1..=32} (ωhigh_n − ωlow_n + 1)` FFT-line range,
because each total independently re-derives step 65's split
struct — doubling the per-line work for a non-trivial callback. The
new free function `coder_partition_d5_ltmin_log2_paired_bit_budget_totals<F:
Fn(u16) -> f64>(ltg_per_line) -> (f64, f64)` fuses the two: it
calls Phase 2 step 65's
`coder_partition_d5_ltmin_log2_row_order_by_width` **once**, then
sums the `narrow_band` (12 cells) and `wide_band` (20 cells)
subarrays of the single returned struct independently, returning
`(narrow_total, wide_total)` — narrow first, matching step 65's
field order. The callback fan-out is exactly half the back-to-back
step 67 + step 66 pairing (one FFT-line pass instead of two), while
the two scalars are bit-identical to the standalone step 67 / step
66 results. No new spec arithmetic beyond `+`. Pairing identity:
`narrow_total + wide_total` recovers the unweighted full row-order
`Σ_n log2_n` exactly. Tests: +6 unit covering zero-dB pair
`(0.0, 0.0)`, bit-identity with standalone steps 67 / 66, callback
fan-out exactly one step-65 pass (= half the back-to-back standalone
count, verified by a `Cell`-counting callback), `narrow + wide`
recovering the full row-order sum, block independence (a wide-only
omega-400 perturbation moves only `wide_total`; a narrow-only
omega-100 perturbation only `narrow_total`), and idempotence for a
pure callback. Only the Phase
2 step 65 width-gated `log2(LTmin_lin_n)` column accessor (and
through it the cascade down to the Table D.5 transcription in
`docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
§"Table D.5 — Layer I and Layer II coder partition table") is read;
the reduction is plain `f64` addition.

**Phase 2 step 65 (r264)** — Annex D Model 1 §D.1 Step 8 width-
gated `log2(LTmin_lin_n)` column projection over Table D.5. Phase
2 step 64 (r263) projected the width-gated paired
`(LTmin_n, width_n)` vector onto its linear-energy presentation,
exposing two strictly-positive per-band subarrays
(`narrow_band: [f64; 12]` for partitions `n ∈ 1..=12` with
`width_n = 0`; `wide_band: [f64; 20]` for partitions `n ∈ 13..=32`
with `width_n = 1`). Several Step 9 / Step 10 / outer-loop
consumers do not read the per-band linear-energy threshold
directly — they read its base-2 logarithm
`log2(10^(LTmin_n / 10))`, the natural per-band bit-budget proxy
in the Layer I/II bit-allocation loop (every factor-of-two change
in linear masking energy corresponds to exactly one unit on the
`log2` axis). The `log2` conversion is the standard `f64::log2`
primitive — it introduces no new spec arithmetic. The new free
function `coder_partition_d5_ltmin_log2_row_order_by_width<F:
Fn(u16) -> f64>(ltg_per_line) -> CoderPartitionD5LtminLog2ByWidth`
calls Phase 2 step 64's
`coder_partition_d5_ltmin_linear_row_order_by_width` once and
applies `cell.log2()` to each of the 12 + 20 cells, producing a
new struct `CoderPartitionD5LtminLog2ByWidth { narrow_band:
[f64; 12], wide_band: [f64; 20] }` whose entries are finite for
any callback returning finite dB at every FFT line. The `LTg(ω)`
callback is invoked exactly as many times as Phase 2 step 64
invokes it (one call per FFT line in
`Σ_{n=1..=32} (ωhigh_n − ωlow_n + 1)`). Strict monotonicity is
preserved cell-wise; width invariant pinned structurally
(implicit in subarray choice). Identity with the dB column:
every output cell equals step 63's matching dB cell multiplied by
the constant `log2(10) / 10 ≈ 0.33219`. Tests: 822 lib (was 810
baseline; +12 unit) covering subarray lengths (12 / 20), zero-dB
callback maps to zero (`log2(1) = 0`) in every cell, finiteness
for any finite-dB callback, cell-wise equality with
`log2(step64_linear)` under a non-trivial callback (strict-
projection cross-check), spot pin at uniform +3 dB (cell ≈
0.9966), odd-symmetry around zero under sign-flipped callbacks
(uniform +3 dB and −3 dB cells sum to zero per cell), strict
proportionality to step 63's dB column with constant
`log2(10) / 10`, strict monotonicity under a uniform −1 dB shift
(every cell shifts by the same constant `−log2(10)/10`),
idempotent for a pure callback, dip in narrow band only affects
narrow band (cross-block insulation), dip in wide band only
affects wide band (cross-block insulation), and a recovery test
that scales `narrow_band ++ wide_band` back to dB by
`10 · log10(2)` and pins index-by-index against step 59's
row-order LTmin vector. Only the Phase 2 step 64 width-gated linear-energy `LTmin_n`
accessor (and through it the cascade down to the Table D.5
transcription in
`docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
§"Table D.5 — Layer I and Layer II coder partition table") is
read; the `f64::log2` primitive is the in-tree standard library.

**Phase 2 step 64 (r263)** — Annex D Model 1 §D.1 Step 8 width-
gated `LTmin_n` column projection over Table D.5 converted to
linear energy (`10^(LTmin_n / 10)`). Phase 2 step 63 (r262)
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
uses (line 1492) — it introduces no new spec arithmetic. The new
free function
`coder_partition_d5_ltmin_linear_row_order_by_width<F: Fn(u16) ->
f64>(ltg_per_line) -> CoderPartitionD5LtminLinearByWidth` calls
Phase 2 step 63's
`coder_partition_d5_ltmin_db_row_order_by_width` once and applies
`(10.0_f64).powf(db / 10.0)` to each of the 12 + 20 cells,
producing a new struct `CoderPartitionD5LtminLinearByWidth {
narrow_band: [f64; 12], wide_band: [f64; 20] }` whose entries are
strictly positive linear energy values. The `LTg(ω)` callback is
invoked exactly as many times as Phase 2 step 63 invokes it (one
call per FFT line in `Σ_{n=1..=32} (ωhigh_n − ωlow_n + 1)`).
Monotonicity is preserved cell-wise; width invariant pinned
structurally (implicit in subarray choice). Tests: 810 lib (was
799 baseline; +11 unit) covering subarray lengths (12 / 20),
zero-dB callback linearises to unit energy everywhere, strict
positivity for any finite callback, cell-wise equality with
`10^(step63_db / 10)` under a non-trivial callback (strict-
projection cross-check), spot pins at uniform +10 dB (factor 10)
and −10 dB (factor 0.1), strict monotonicity under a uniform −1
dB shift (every cell shrinks by the same constant ratio
`10^(−1/10) ≈ 0.7943`), idempotent for a pure callback, dip in
narrow band only affects narrow band (cross-block insulation),
dip in wide band only affects wide band (cross-block insulation),
and a recovery test that log-maps `narrow_band ++ wide_band` back
to dB and pins index-by-index against step 59's row-order LTmin
vector. Only the Phase 2
step 63 width-gated `LTmin_n` (dB) column accessor (and through
it the cascade down to the Table D.5 transcription in
`docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
§"Table D.5 — Layer I and Layer II coder partition table") is
read; the `10^(x / 10)` dB → linear conversion is the in-tree
convention.

**Phase 2 step 61 (r260)** — Annex D Model 1 §D.1 Step 8 paired
`(LTmin_n, width_n)` row-order vector over Table D.5. Phase 2
step 59 (r258) exposed the row-order LTmin vector
`[LTmin_1, …, LTmin_32]` the Layer I / Layer II bit-allocation
loop reduces from `LTg(ω)`; Phase 2 step 60 (r259) exposed the
row-order width vector `[width_1, …, width_32]` it pairs with
LTmin at every row. r260 closes the per-row pairing: the bit-
allocation loop walks the 32 coder partitions in row order and at
every row consumes **both** columns paired as the partition's
"per-row brief" (target threshold + width flag). The LTmin column
closes over the caller's `LTg(ω)` callback (run-time-dependent);
the width column is a static Table D.5 column.

A new free function plus a new public struct:

* `CoderPartitionD5Reduction { ltmin_db: f64, width_n: u16 }` — a
  single row of the bit-allocation input. `ltmin_db` carries the
  inclusive minimum of `LTg(ω)` (dB) over the partition's FFT-
  line range; `width_n` is the static Table D.5 column value.
* `coder_partition_d5_reduction_row_order<F: Fn(u16) -> f64>(
  ltg_per_line) -> [CoderPartitionD5Reduction; 32]` — index-
  aligned zip of Phase 2 step 59's row-order LTmin reducer
  `coder_partition_d5_ltg_min_row_order` (closed over the
  caller's `LTg(ω)` callback) with Phase 2 step 60's row-order
  width vector `coder_partition_d5_width_row_order`. Element `i`
  holds the `(LTmin_{i + 1}, width_{i + 1})` pair (the spec's
  1-based `n` in 0-based array form).

**Index convention.** 0-based on the returned slice; element
`i` holds `(LTmin_{i + 1}, width_{i + 1})`. The spec's 1-based
partition index `n ∈ 1..=32` maps to array index `i = n - 1 ∈
0..=31`. Partition 0 (the degenerate single-line `width_n = 0`
row carrying `ωlow_0` only) is excluded from the vector for
index consistency with steps 59 and 60 — the downstream bit-
allocation loop walks partitions `1..=32` and does not consult
partition 0, matching the spec's coder-partition usage.

**Composition rather than introduction.** A pure index-aligned
zip of the two existing row-order columns. No spec arithmetic is
introduced — only the per-row pairing at the same array index,
which is exactly the per-row input the Layer I / Layer II bit-
allocation loop reads in lockstep. The width column is fully
determined by the static Table D.5 column (no run-time inputs)
and so is invariant across callbacks; the LTmin column closes
over the caller's `LTg(ω)`. Structural orthogonality keeps the
two columns independent — neither influences the other's
computation.

**Caller cost.** The `LTg(ω)` callback is invoked exactly as
many times as Phase 2 step 59 invokes it (one call per FFT line
in `Σ_{n=1..=32} (ωhigh_n − ωlow_n + 1)` summed over the table);
the width vector adds no callback invocations.

**Boundary semantics.** Inherit Phase 2 step 59's inclusive-on-
both-ends reduction semantics unchanged for the LTmin column — a
sharp dip on a shared boundary `ω = ωhigh_n = ωlow_{n+1}` enters
**both** adjacent partitions' `LTmin`. The width column has no
boundary semantics (it is a static per-row table value).

Validated by 13 new lib unit tests in `psy::tests`. Length pin
(32). Constant-callback fills every `ltmin_db` cell with the
constant. LTmin column matches step 59 for a non-trivial line-
dependent callback `LTg(ω) = sin(ω)`. Width column matches step
60 across two different callbacks (structural orthogonality).
Width invariant across callbacks. Width column matches the full
verbatim Table D.5 literal `[0×12, 1×20]`. Identity callback
`LTg(ω) = ω` returns `ωlow_n` per row. Negative-identity
callback `LTg(ω) = -ω` returns `-ωhigh_n` per row. Transition
pair pinned at array indices 11 and 12 (`width_n = 0` and `1`
respectively). Endpoint pin at array indices 0 (partition 1,
width 0) and 31 (partition 32, width 1). Idempotence for a pure
callback `LTg(ω) = cos(ω)`. Single dip on a strict-interior line
of partition 5 affects only that partition's `ltmin_db` with the
width vector untouched. Strict-composition pairing at every row:
both columns of the paired output agree with the underlying
single-column accessors (`coder_partition_d5_ltg_min_row_order`
and `coder_partition_d5_width_row_order`) under a non-trivial
affine callback `LTg(ω) = (ω − 256) × 0.5`. Tests: 770 lib (was
757 baseline; +13 unit). Only the Phase 2 step 59 row-order LTmin reducer
`coder_partition_d5_ltg_min_row_order` and the Phase 2 step 60
row-order width vector `coder_partition_d5_width_row_order` (and
through them the Phase 2 step 58 per-partition reducer
`coder_partition_d5_ltg_min`, the Phase 2 step 52 per-partition
width accessor `coder_partition_d5_width`, and the underlying
Table D.5 transcription in
`docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
§"Table D.5 - Layer I and Layer II coder partition table") were
read.

**Phase 2 step 60 (r259)** — Annex D Model 1 §D.1 Step 8 row-order
`width_n` vector over Table D.5. Phase 2 step 59 (r258) broadcast
the Phase 2 step 58 (r257) per-partition `LTg` minimum reducer
across the Phase 2 step 55 (r254) row-order iterator, producing the
32-element row-order LTmin vector `[LTmin_1, LTmin_2, …, LTmin_32]`
the Layer I / Layer II bit-allocation loop consumes per frame. r259
closes the second half of that per-frame input: a row-order vector
of the `width_n` column the bit-allocation loop pairs with the
LTmin vector at every row. Together steps 59 + 60 expose the
complete per-frame coder-partition input the bit-allocation loop
consumes:

```text
step 59 LTmin: [LTmin_1, LTmin_2, …, LTmin_32]  (f64,  dB)
step 60 width: [width_1, width_2, …, width_32]  (u16,  0/1)
```

A new free function:

* `coder_partition_d5_width_row_order() -> [u16; 32]` — broadcasts
  the Phase 2 step 52 per-partition `width_n` accessor across the
  Phase 2 step 55 row-order iterator, returning the static 32-
  element 0-based array where element `i` holds `width_{i + 1}`
  (the spec's 1-based partition index in 0-based array form).

**Index convention.** 0-based on the returned slice;
`out[i] = width_{i + 1}`. The spec's 1-based partition index
`n ∈ 1..=32` maps to array index `i = n - 1 ∈ 0..=31`. Partition 0
(the degenerate single-line `width_n = 0` row carrying `ωlow_0`
only) is excluded from the vector for index consistency with Phase
2 step 59 (r258)'s LTmin vector — the downstream bit-allocation
loop walks partitions `1..=32` in lockstep across the LTmin and
width vectors and does not consult partition 0.

**Composition rather than introduction.** A pure broadcast of
Phase 2 step 52's per-partition `width_n` accessor
`coder_partition_d5_width` across the Phase 2 step 55 row-order
iterator `coder_partition_d5_spans`. No spec arithmetic is
introduced — only the broadcast of step 52's single-partition
lookup across all 32 recoverable partitions. Unlike step 59, this
accessor has **no run-time inputs**: the `width_n` column is a
static property of Table D.5, so the returned vector is the same
`[u16; 32]` on every call. The output is structurally orthogonal
to the LTmin column (the run-time-dependent half of the per-frame
bit-allocation input pair), keeping the two columns as independent
accessors the downstream loop pairs at the call site rather than
fused into a single `(LTmin, width)` carrier.

**Constant values.** Per the Table D.5 transcription the vector is
exactly twelve zeros followed by twenty ones:

```text
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```

The single 0 → 1 transition lies between array indices 11 and 12
(partitions 12 and 13), pinned by Phase 2 step 52's transcription
rule "rows 0..=12 have width 0; rows 13..=32 have width 1" and the
step 55 row-order iterator's ascending-`n` ordering.

Validated by 12 new lib unit tests in `psy::tests`. The 32-element
length is pinned. The lower-block rule (n ∈ 1..=12 → 0) is
verified across array indices 0..=11; the upper-block rule
(n ∈ 13..=32 → 1) is verified across array indices 12..=31. The
single-step 0 → 1 transition at array index 12 (partition 13) is
pinned by asserting `v[11] = 0` and `v[12] = 1`. Every cell is
verified to carry only 0 or 1 (the binary `width_n` column has no
intermediate values). A strict-composition cross-check asserts the
row-order vector matches the per-partition step 52 lookup
`coder_partition_d5_width(n)` at every n ∈ 1..=32. A
table-literal pin asserts the row-order vector equals the verbatim
32-element constant `[0×12, 1×20]` — any future change to Table
D.5's `width_n` column would surface here independently of the
step 52 / step 55 underlying accessors. Table-wide endpoints are
pinned (`out[0] = width_1 = 0`, `out[31] = width_32 = 1`). The
upper-block sum is pinned as `Σ v = 20` (twenty partitions in the
upper block, each contributing 1). Idempotence across back-to-back
calls is pinned (the accessor has no run-time state). Non-
decreasing monotonicity across array indices is pinned (one
ascending step from 0 to 1, no later drops). Finally, the
ascending-partition iteration order is verified by reconstructing
the vector via a manual `n ∈ 1..=32` walk and comparing it to the
function's output. Tests: 757 lib (was 745 baseline; +12 unit). No
external implementation consulted; only the Phase 2 step 52 per-
partition `width_n` accessor `coder_partition_d5_width` and the
Phase 2 step 55 row-order iterator `coder_partition_d5_spans` (and
through them the Table D.5 transcription in
`docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
§"Table D.5 - Layer I and Layer II coder partition table") were
read.

**Phase 2 step 59 (r258)** — Annex D Model 1 §D.1 Step 8 row-order
LTmin vector over Table D.5. Phase 2 step 58 (r257) reduced the
per-FFT-line global masking threshold `LTg(ω)` over a single
coder partition `n ∈ 1..=32` by taking the minimum
`LTmin_n = min_{ω ∈ [ωlow_n, ωhigh_n]} LTg(ω)`. The Layer I /
Layer II bit-allocation loop consumes the full row-order vector
`[LTmin_1, LTmin_2, …, LTmin_32]` per frame, walking the 32
coder partitions in ascending-`n` order (the spec table's row
order, pinned at iteration order by Phase 2 step 55 / r254's
`coder_partition_d5_spans`). The Layer III outer-loop SNR-budget
analogue consumes the same per-partition vector.

A new free function:

* `coder_partition_d5_ltg_min_row_order<F: Fn(u16) -> f64>(
  ltg_per_line) -> [f64; 32]` — broadcasts the Phase 2 step 58
  per-partition reducer across the Phase 2 step 55 row-order
  iterator, returning a 32-element 0-based array where element
  `i` holds `LTmin_{i + 1}` (the spec's 1-based partition index
  in 0-based array form).

**Index convention.** 0-based on the returned slice;
`out[i] = LTmin_{i + 1}`. The spec's 1-based partition index
`n ∈ 1..=32` maps to array index `i = n - 1 ∈ 0..=31`. Partition
0 (the degenerate single-line `width_n = 0` row carrying `ωlow_0`
only) is excluded from the vector — Phase 2 step 58 returns
`None` for `n = 0` because the reduction range is undefined
without a `ωlow_n` boundary in Table D.5. The downstream bit-
allocation loop walks partitions `1..=32` and does not consult
partition 0, matching the spec's coder-partition usage.

**Composition rather than introduction.** A pure broadcast of
Phase 2 step 58's per-partition reducer across the Phase 2 step
55 row-order iterator. No spec arithmetic is introduced — only
the broadcast of step 58's single-partition reduction across all
32 recoverable partitions, which is the row-order vector form the
Layer I / Layer II bit-allocation loop consumes per frame. The
`LTg(ω)` callback is the caller's — typically a closure closing
over the static masker list + threshold-in-quiet curve — keeping
this accessor pure with respect to the masker selection pipeline
(Steps 1-5), which remain blocked on the PNG-only Table D.1 /
D.2 / D.3 transcription gap. Once Steps 1-5 land the concrete
`LTg(ω)` closure will be the one produced by Step 7's
`global_masking_threshold_db` applied per line.

**Boundary semantics.** Inherits Phase 2 step 58's inclusive-on-
both-ends reduction semantics unchanged: a sharp dip on a shared
boundary `ω = ωhigh_n = ωlow_{n+1}` enters **both** adjacent
partitions' `LTmin` (the conservative-bit-allocation reading the
spec intends). A caller that wants every FFT line to enter
exactly one partition's reduction (single-assignment binning)
uses `first_partition_containing_line` to bin per line before
folding outside this accessor.

Validated by 11 new lib unit tests in `psy::tests`. A constant
`LTg ≡ C` callback returns `[C; 32]` for every cell. The
0-based-array / 1-based-partition convention is pinned by
asserting `out[0] = LTmin_1`, which under the identity callback
`LTg(ω) = ω` equals `ωlow_1 = 1` (the table-wide lower edge).
A strict-composition cross-check asserts the row-order vector
matches a manual loop calling the step 58 per-partition reducer
for a non-trivial callback `ω * 0.7 − 13`. The array length is
exactly 32 and every cell is finite under a finite callback (no
infinity leak from the `f64::INFINITY` seed). A single `-100 dB`
dip placed at a partition's interior middle line (not on a shared
boundary) is verified to pull only that partition's row down and
leave every other partition at the baseline — the single-
assignment regression pin. A `-50 dB` dip at the shared boundary
`ωhigh_5 = ωlow_6` is verified to pull both adjacent partitions
to the dip value (and leave every other partition at the
baseline) — the shared-boundary semantics pin. An end-to-end
composition pin feeds the row-order builder the Phase 2 step 44
`global_masking_threshold_db` value at every line (with a tonal
masker at z = 5 Bark, SPL = 60 dB, and a synthetic
`z(ω) = ω · 0.05` Bark stand-in until Step 1's FFT-bin → Hz
mapping lands) and asserts agreement with the explicit per-line
fold via the step 57 iterator. The row-order vector is verified
non-decreasing under the identity callback (each `ωlow_n` grows
strictly with `n` per the Phase 2 step 50 boundary-monotonicity
reading). Table-wide endpoints are pinned (`out[0] = ωlow_1`,
`out[31] = ωlow_32`). Finally, a negative-identity
`LTg(ω) = -ω` callback is verified to return `-ωhigh_n` per row
(the highest line produces the most-negative reduction).
Tests: 745 lib (was 734 baseline; +11 unit). Only the Phase 2 step 58 per-partition
reducer `coder_partition_d5_ltg_min` and the Phase 2 step 55
row-order iterator `coder_partition_d5_spans` (and through them
the Phase 2 step 44 Step 7 `global_masking_threshold_db` and the
Table D.5 transcription in
`docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
§"Table D.5 - Layer I and Layer II coder partition table") were
read.

**Phase 2 step 58 (r257)** — Annex D Model 1 §D.1 Step 8 per-partition
`LTg` minimum reduction. Phase 2 step 44 (r219) landed Step 7's
per-FFT-line global masking threshold `LTg(i)` as
`global_masking_threshold_db`; Phase 2 step 49 (r248) transcribed
Table D.5 (the Layer I / Layer II coder partition table); Phase 2
step 57 (r256) closed the per-partition FFT-line walk as
`coder_partition_d5_omega_iter`. r257 wires the two halves together
into the spec's Step 8 reduction:

```text
LTmin_n = min_{ω ∈ [ωlow_n, ωhigh_n]} LTg(ω)   dB
```

A new free function:

* `coder_partition_d5_ltg_min<F: Fn(u16) -> f64>(n, ltg_per_line) ->
  Option<f64>` — reduces the caller-supplied per-FFT-line `LTg(ω)`
  callback (from Step 7's `global_masking_threshold_db`, applied per
  line) over every `ω ∈ [ωlow_n, ωhigh_n]` by taking the minimum.
  Returns `Some(LTmin_n)` for any `n ∈ 1..=32` and `None` for the
  two edge cases inherited from `coder_partition_d5_omega_iter`
  (`n = 0` — `ωlow_0` not in Table D.5; `n = 33` — row absent).

The reduction is the spec's most-conservative per-partition reading
— a single FFT line dipping below the partition's average threshold
pulls the whole partition's bit-allocation budget down to that
line's level. This is the value the Layer I / Layer II bit-allocation
loop consumes per partition (Layer III's outer-loop SNR budget is
the analogue).

**Composition rather than introduction.** The accessor is a strict
composition of the Phase 2 step 57 per-partition FFT-line iterator
and `Iterator::map ∘ Iterator::fold(f64::INFINITY, f64::min)`. No
spec arithmetic is introduced — only the per-line minimum fold over
the recoverable line range. The `f64::INFINITY` seed pairs with
`f64::min` to produce the per-partition minimum for any partition
with at least one line. The Step 7 `LTg` callback is the caller's
— typically a closure closing over the static masker list +
threshold-in-quiet curve — keeping this accessor pure with respect
to the masker selection pipeline (Steps 1-5), which remain blocked
on the PNG-only Table D.1 / D.2 / D.3 transcription gap.

**Boundary semantics.** The reduction is inclusive on both ends,
matching the per-partition sum-over-lines pattern Phase 2 step 57
(r256) wired into Step 7's own `Σ_{ω ∈ partition}` form. Two
consecutive partitions `n` and `n + 1` therefore both consider the
shared boundary line `ω = ωhigh_n = ωlow_{n+1}` in their minimum
— a sharp dip located exactly on a shared boundary reduces both
adjacent partitions' `LTmin`, which is the conservative-bit-
allocation reading the spec intends. A caller that wants every FFT
line to enter exactly one partition's minimum (single-assignment
binning, no shared-boundary double-influence) uses the step 56
inverse accessor `first_partition_containing_line` to bin per line,
then folds per partition outside this accessor.

Validated by 10 new lib unit tests in `psy::tests`. The out-of-band
`None` branches are pinned at `n = 0`, `n = 33`, `n = 100`,
`n = 1000`, and `n = u16::MAX`. A constant `LTg ≡ C` callback
returns exactly `C` for every partition. An identity `LTg(ω) = ω`
callback returns exactly `ωlow_n` (the minimum line in the
partition's inclusive range). A negative-identity `LTg(ω) = -ω`
callback returns exactly `-ωhigh_n` (the highest line produces the
most-negative reduction). A single `-100 dB` dip placed at each
partition's middle line is verified to pull the whole partition's
`LTmin` down to `-100 dB`, pinning the conservative-bit-allocation
reading. The accessor's output is cross-checked against an explicit
`coder_partition_d5_omega_iter ∘ map ∘ fold` fold with the same
callback for a non-trivial value `ω * 0.7 − 13`, pinning the
strict-composition implementation. The shared-boundary double-
influence property is pinned by placing a `-50 dB` dip at every
shared boundary `ωhigh_n = ωlow_{n+1}` and asserting both
neighbouring partitions record the dip. Finally an end-to-end
composition pin feeds the step 58 reducer the Phase 2 step 44
`global_masking_threshold_db` value at every FFT line (with a
tonal masker at z = 5 Bark, SPL = 60 dB, and a synthetic
`z(ω) = ω · 0.05` Bark mapping until Step 1's FFT-bin → Hz table
lands) and asserts agreement with the explicit per-line fold —
exercising the masking-function piecewise branches through the
reduction without introducing any new spec arithmetic. Tests: 734
lib (was 724 baseline; +10 unit). Only the Phase 2 step 57 per-partition iterator
`coder_partition_d5_omega_iter`, the Phase 2 step 44 Step 7
`global_masking_threshold_db`, and (transitively) the Table D.5
transcription in
`docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md` §"Table D.5
- Layer I and Layer II coder partition table" were read.

**Phase 2 step 57 (r256)** — Annex D Table D.5 per-partition FFT-line
iterator. Phase 2 step 51 (r250) exposed each partition's
`(ωlow_n, ωhigh_n)` boundary pair via
`coder_partition_d5_line_range`. Phase 2 step 53 (r252) composed
those boundaries with `width_n` into the `CoderPartitionD5Span`
descriptor. Phase 2 step 54 (r253) lifted the membership inequality
on that pair to the named predicate `partition_n_contains_line`;
Phase 2 step 55 (r254) added the row-order iterator
`coder_partition_d5_spans`; Phase 2 step 56 (r255) closed the inverse
lookup with `first_partition_containing_line`. r256 closes the
per-partition FFT-line walk — instead of asking "is line `ω` in
partition `n`?" or "given line `ω`, which partition?", the
downstream Model 1 / Model 2 reduction now walks each partition's
FFT lines directly:

* `coder_partition_d5_omega_iter(n) -> Option<RangeInclusive<u16>>`
  yields every `omega ∈ [ωlow_n, ωhigh_n]` in ascending order for
  any `n ∈ 1..=32`. Returns `None` for any `n` outside that range,
  inheriting from the step 51 line-range accessor.

The iterator is the foundational primitive the downstream Step 8
partition-threshold reduction binds its sum-over-lines against —
Annex D Step 7's `Σ_{j ∈ partition} 10^(LT[j]/10)` composes
directly:

```text
    for span in coder_partition_d5_spans() {
        let acc: f64 = coder_partition_d5_omega_iter(span.index)
            .expect("span.index ∈ 1..=32")
            .map(|omega| per_line_value(omega))
            .sum();
        // … per-partition threshold reduction continues here
    }
```

**Boundary semantics.** The iterator is **inclusive on both ends**,
matching the dual-role `ωlow_{n+1} / ωhigh_n` reading Phase 2
step 50 (r249) pinned and the membership predicate Phase 2 step 54
(r253) named. Two consecutive partitions both emit the shared
boundary line `ω = ωhigh_n = ωlow_{n+1}` — matching the spec's
per-partition sum-over-lines reading where the shared boundary
*does* contribute to both partitions' reductions. A caller that
wants single-assignment binning (no double-counting) uses the
step 56 inverse accessor `first_partition_containing_line` instead.
Implementation is one line —
`coder_partition_d5_line_range(n).map(|(lo, hi)| lo..=hi)` — a pure
composition of the step 51 line-range accessor and
`RangeInclusive::new`, with no arithmetic introduced.

Validated by 12 new lib unit tests in `psy::tests`. Out-of-band
`None` branches are pinned at `n = 0`, `n = 33`, `n = 100`, and
`n = u16::MAX`. Partition 1's iterator starts at the table-wide
lower edge `ω = 1`; partition 32's ends at the table-wide upper
edge `ω = 513`. Per-partition endpoint and length agreement with
step 51's `coder_partition_d5_line_range` is pinned for every
`n ∈ 1..=32`. The ascending-stride-1 walk within each partition
(no gaps, no duplicates) is pinned across every partition. Per-line
agreement with the step 54 membership predicate
`partition_n_contains_line(n, ω) = Some(true)` is pinned for every
iterator-emitted `ω`. The shared-boundary double-emission property
(both `n` and `n+1`'s iterators contain `ωhigh_n`) is pinned at
every `n ∈ 1..=31`. The table-wide band coverage
`⋃ iter(n) = [1, 513]` is pinned by collection into a `BTreeSet`.
The total-line-count identity `Σ_n |iter(n)| = 513 + 31 = 544`
(band size + 31 double-counted shared boundaries) is pinned
directly. Finally an end-to-end composition smoke pin
`coder_partition_d5_spans` ∘ `coder_partition_d5_omega_iter` ∘
`sum` is matched against the arithmetic-series closed form
`Σ_{ω=ωlow_n}^{ωhigh_n} ω = (ωlow_n + ωhigh_n) ·
(ωhigh_n − ωlow_n + 1) / 2` for every recoverable partition —
pinning the downstream Step 8 partition-threshold reduction's
composition path directly. Tests: 724 lib (was 712 baseline; +12
unit). Only the Phase 2
step 51 accessor `coder_partition_d5_line_range` and its underlying
Table D.5 transcription in
`docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
§"Table D.5 - Layer I and Layer II coder partition table" were
read.

**Phase 2 step 56 (r255)** — Annex D Table D.5 inverse line→partition
lookup. Phase 2 step 53 (r252) composed each partition's three
Table D.5 columns into a `CoderPartitionD5Span` descriptor with the
inclusive boundary pair `(ωlow_n, ωhigh_n)`. Phase 2 step 54 (r253)
lifted the membership inequality to the named predicate
`partition_n_contains_line`. Phase 2 step 55 (r254) added the
row-order iterator `coder_partition_d5_spans` over the recoverable
descriptors. r255 closes the inverse direction: instead of asking
"is line `ω` in partition `n`?", the downstream Model 1 / Model 2
reduction asks "given line `ω`, which partition `n` does it land
in?" — the natural question when walking the FFT-line domain and
binning each line into its partition:

* `first_partition_containing_line(omega) -> Option<u16>` returns
  the index `n` of the **lowest** partition whose inclusive boundary
  range `[ωlow_n, ωhigh_n]` contains the FFT line `omega`. The
  accessor yields `Some(n)` with `n ∈ 1..=32` for `omega ∈ [1, 513]`
  (the table-wide FFT-line domain Table D.5 covers) and `None` for
  any out-of-band `omega`.

**Shared-boundary disambiguation.** Phase 2 step 50 (r249) pinned
the column-heading `ωlow_{n+1} / ωhigh_n` dual reading, and Phase 2
step 54 (r253) pinned the inclusive-on-both-ends boundary
semantics: every shared boundary line `ω = ωhigh_n = ωlow_{n+1}` is
a member of **both** partition `n` (as its `ωhigh_n`) and partition
`n + 1` (as its `ωlow_{n+1}`). When the caller asks the inverse
question, this accessor returns the **lower** index `n` — the
unique deterministic choice that does not double-count the
boundary lines, matching both the spec table's row-order
presentation (the boundary cell is printed on row `n`'s line, not
on row `n + 1`'s) and the row-order iterator's ascending walk
pinned by step 55. Implementation is one line —
`coder_partition_d5_spans().find(|s| s.omega_low <= omega && omega
<= s.omega_high).map(|s| s.index)` — with no arithmetic beyond the
inequality on each descriptor's pre-computed boundaries. Complexity
is `O(32)` worst case; for a Model 1 / Model 2 reduction sweeping
all 513 lines this is `O(513 × 32) ≈ 16 K` boundary comparisons,
well below any threshold worth complicating the accessor over.

Validated by 9 new lib unit tests in `psy::tests`:
`first_partition_returns_none_below_band` and
`…_above_band` pin the out-of-band `None` branches at `omega = 0`,
`omega = 514`, `omega = 10_000`, and `omega = u16::MAX`;
`…_at_table_wide_lower_edge_is_partition_one` and
`…_at_table_wide_upper_edge_is_partition_thirty_two` pin the
table-wide boundary identities `first_partition_containing_line(1)
= Some(1)` and `first_partition_containing_line(513) = Some(32)`;
`…_at_shared_boundary_picks_lower_index` pins the lower-index pick
at every shared boundary `ω = ωhigh_n` for `n ∈ 1..=31` (sweeping
the table directly, no representative subset);
`…_at_strict_interior_lines_matches_step_53_descriptor` pins the
per-partition strict-interior agreement at `ω = ωlow_n + 1`
against the step 53 descriptor;
`…_walks_the_full_band_with_no_gaps` pins the table-wide `[1,
513]` no-gap coverage property;
`…_n_agrees_with_step_54_membership_predicate` pins the agreement
with the step 54 predicate
`partition_n_contains_line(n, ω) = Some(true)` for every in-band
`ω`; and `…_n_is_the_minimum_of_all_containing_partitions` pins
the "lowest partition first" semantics directly — the inverse
accessor's answer is computed from first principles as the minimum
`n` across all partitions that contain `ω` under the step 54
predicate (sweeping every in-band line).
Tests: 712 lib (was 703 baseline; +9 unit). Only the Phase 2 step 55 iterator
`coder_partition_d5_spans` and (through it) the Phase 2 step 53
descriptor `coder_partition_d5_span` and its underlying Table D.5
transcription in
`docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
§"Table D.5 - Layer I and Layer II coder partition table" were
read.

**Phase 2 step 55 (r254)** — Annex D Table D.5 row-order iteration
helper. Phase 2 step 53 (r252) composed the verbatim Table D.5
columns of partition `n` into a single `CoderPartitionD5Span`
descriptor; Phase 2 step 54 (r253) added the
`partition_n_contains_line(n, ω)` inclusive-line membership
predicate on that descriptor. The downstream Model 1 / Model 2
partition-threshold reduction walks Table D.5 row by row,
accumulating per-partition values across the in-range FFT lines;
r254 closes that loop with a row-order iterator over the
recoverable Table D.5 descriptors, so the reduction reads as

```text
    for span in coder_partition_d5_spans() {
        // bin every FFT line ω with partition_n_contains_line(span.index, ω)
        …
    }
```

— matching the spec table's row-order presentation without
open-coding the `1..=32` range or the descriptor lookup at every
reduction site:

* `coder_partition_d5_spans()` returns `impl Iterator<Item =
  CoderPartitionD5Span>` yielding **exactly 32 descriptors** — one
  per recoverable Table D.5 row — in strictly ascending row order
  `n = 1, 2, …, 32`. The two boundary-table-gap edges (`n = 0`,
  `n = 33`) that the step 53 descriptor returns `None` for are
  **not** emitted: a row-order walk of Table D.5 sees the same
  boundary-table gaps the descriptor sees, so emitting either edge
  would force the caller to filter back to `1..=32` immediately.

The implementation is a single line:
`(1_u16..=32).map(|n| coder_partition_d5_span(n).expect("n ∈ 1..=32
is recoverable"))`. The `.expect` is **infallible** for the
iterated range — every `n ∈ 1..=32` is a recoverable Table D.5 row
by Phase 2 step 53 construction, pinned by step 53's tests. The
returned iterator is `Clone` and `ExactSizeIterator +
DoubleEndedIterator` via the `Range<u16>::Map` passthrough, but its
public signature is kept generic (`impl Iterator<Item =
CoderPartitionD5Span>`) so future implementation changes don't
break consumers.

Validated by 7 new lib unit tests in `psy::tests`:
`coder_partition_d5_spans_yields_thirty_two_descriptors` pins the
exact emission count;
`…_yields_row_order` pins the strictly ascending `1..=32` `index`
sequence — no gaps, no repetition, no reordering;
`…_each_descriptor_matches_table_lookup` pins per-descriptor
agreement with `coder_partition_d5_span(n)` (the iterator is a
pure row-walk of the step 53 accessor and invents no new field
values);
`…_skips_boundary_table_gaps` pins the boundary-table-gap skip
(`n = 0` and `n = 33` never appear in the emitted index sequence);
`…_tiles_the_full_band` pins the table-wide band coverage `[1,
513]` (the first span's `ωlow = 1` and the last span's `ωhigh =
513`) and the adjacent-row tiling identity `ωhigh_n = ωlow_{n+1}`
for every consecutive emitted pair;
`…_pairs_with_membership_predicate` pins the spec-read pairing
pattern across every `(span, ω) ∈ iter × 0..=520` — the
iterator/predicate composition agrees with the descriptor's
inequality at every line in the band (sweeping past the upper edge
to exercise the out-of-band false branch);
`…_is_clone_and_repeatable` pins the multi-pass walk property the
downstream reduction relies on — the iterator is cheap to clone
and yields an identical sequence on each walk.
Tests: 703 lib (was 696 baseline; +7 unit). Only the Phase 2 step 53 descriptor
`coder_partition_d5_span` and its underlying Table D.5
transcription in
`docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
§"Table D.5 - Layer I and Layer II coder partition table" were
read.

**Phase 2 step 54 (r253)** — Annex D Table D.5 inclusive-line
membership predicate. Phase 2 step 53 (r252) composed the verbatim
Table D.5 columns of partition `n` into a single
`CoderPartitionD5Span` descriptor carrying `index`, `omega_low`
(`ωlow_n`), `omega_high` (`ωhigh_n`) and `width` (`width_n`). The
downstream Model 1 / Model 2 partition-threshold reduction iterates
the table row by row and, per iteration, asks "does FFT-line index
`ω` belong to partition `n`?" — the obvious inequality on the
descriptor `s.omega_low <= ω && ω <= s.omega_high`. r253 lifts that
inequality to a named predicate so the reduction can read like the
spec ("for each line in partition `n` …") and the
range-rejection behaviour at the two boundary-table gaps stays in
one place:

* `partition_n_contains_line(n, omega)` returns `Some(true)` if `ω`
  is inside partition `n`'s inclusive boundary range
  `[ωlow_n, ωhigh_n]`, `Some(false)` if it isn't, and `None` for
  any `n` outside `1..=32` (`n = 0` lacks a Table D.5 `ωlow_0`;
  `n = 33` has no row at all).

The predicate is a **pure composition** of the step 53 descriptor
with the inclusive inequality — no arithmetic beyond the inequality
on the descriptor's pre-computed boundaries is introduced. It is
`coder_partition_d5_span(n).map(|s| s.omega_low <= omega && omega
<= s.omega_high)` exactly.

The inclusive-on-both-ends reading the predicate uses is the same
one Phase 2 step 50 (r249) pinned at the dual-role boundary
accessors: the column heading `ωlow_{n+1} / ωhigh_n` names two
distinct spec roles for the row's verbatim integer, so the tiling
identity `ωhigh_n = ωlow_{n+1}` means the shared boundary line
belongs to **both** partitions `n` and `n + 1` under inclusive-on-
both-ends membership. The downstream partition reduction handles
the shared boundary as the spec prescribes (typically: read
partition `n` up through `ωhigh_n`, then read partition `n + 1`
from `ωlow_{n+1} = ωhigh_n`); both readings are sample-exact
against the spec table. The predicate itself is silent on which
reduction strategy a caller picks — it just answers the
membership question.

The `omega` argument is **not** range-checked against the
table-wide FFT-line domain `[1, 513]`. A caller passing an
out-of-band value (e.g. `omega = 0`, `omega = 514`, `omega =
u16::MAX`) gets a well-defined `false` answer for every in-range
`n`, exactly as the inequality on the descriptor's `[ωlow_n,
ωhigh_n]` dictates — the predicate is a pure boolean over the
descriptor and does not re-invent the table-wide line domain.

Validated by 7 new lib unit tests in `psy::tests`:
`partition_n_contains_line_inclusive_at_both_boundaries` pins the
inclusive-on-both-ends reading at every recoverable partition's
`ωlow_n` and `ωhigh_n`;
`…_rejects_just_outside_each_boundary` pins the off-by-one
exclusion of `ωlow_n - 1` and `ωhigh_n + 1`;
`…_anchor_lines` pins spec-anchored membership at partitions
`{1, 12, 13, 32}` covering the lower-block edge, the
last/first row of each `width_n` block, and the table top
(including the shared-boundary line at partition 13's `ωlow_13 =
193`);
`…_rejects_partition_index_edges_and_out_of_range` pins `None` at
both edges (`n ∈ {0, 33}`) and above (`n ∈ {34, 64, u16::MAX}`)
across a sweep of `omega` values, confirming the answer doesn't
depend on the line argument at an unrecoverable partition index;
`…_every_in_band_line_belongs_to_exactly_one_partition` pins
the tiling property at the line level — boundary lines belong to
two consecutive partitions, interior lines belong to exactly one;
`…_matches_descriptor_inequality_for_every_in_range_pair` pins
the pure-composition property across every `(n, ω) ∈ 1..=32 ×
0..=520` pair (sweeping past the table-wide upper bound to
exercise the out-of-band false branch);
`…_out_of_band_omega_is_false_at_every_in_range_partition` pins
the predicate's silence on out-of-band `omega` values — `false`
for `omega ∈ {0, 514, 1024, u16::MAX}` at every in-range `n`.
Tests: 696 lib (was 689 baseline; +7 unit). Only the Phase 2 step 53 descriptor
and its underlying Table D.5 transcription in
`docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
§"Table D.5 - Layer I and Layer II coder partition table" were
read.

**Phase 2 step 53 (r252)** — Annex D Table D.5 composed partition
descriptor. Phase 2 steps 51 / 52 (r250 / r251) landed the inclusive
FFT-line range accessor `(ωlow_n, ωhigh_n)` and the `width_n` value
accessor. The downstream Model 1 / Model 2 partition-threshold
reduction iterates Table D.5 row by row and, per iteration, reads the
two boundary FFT-line indices plus the `width_n` value. r252 composes
the two existing accessors into a single per-partition descriptor that
the reduction loop can consume verbatim — no new arithmetic, no new
data, only a composition of the three Table D.5 columns into one
record:

* `CoderPartitionD5Span { index, omega_low, omega_high, width }` —
  the three verbatim columns of partition `n` reassembled into one
  struct, with the dual-role boundary column already resolved into
  the two distinct spec roles `ωlow_n` and `ωhigh_n` (Phase 2 step 50
  did the role disambiguation; the descriptor just inherits both).
* `coder_partition_d5_span(n)` returns `Some(span)` for
  `n ∈ 1..=32` and `None` outside.

The descriptor's valid range is the **intersection** of the two
underlying accessors' ranges (`coder_partition_d5_line_range`'s
`1..=32` and `coder_partition_d5_width`'s `0..=32`), i.e. `1..=32`.
The two edges are explicitly `None`:

* `n = 0` — partition 0's `width_n = 0` cell **is** in Table D.5, but
  its lower boundary `ωlow_0` is **not** (the column heading's
  `ωlow_{n+1}` shift removes it). The descriptor declines verbatim
  rather than inventing a synthetic lower boundary.
* `n = 33` — neither row 33's boundary nor its `width_n` cell exists
  in Table D.5 (the table tops out at row `n = 32` with
  `ωhigh_32 = 513`). The descriptor returns `None`.

The composition is **pure**: `omega_low` is
`coder_partition_d5_omega_low(n)`, `omega_high` is
`coder_partition_d5_omega_high(n)`, `width` is
`coder_partition_d5_width(n)`. No arithmetic beyond what the three
underlying accessors already perform is introduced. The composition
preserves every structural property already pinned at the column-level
accessors — the uniform 17-line inclusive span, the `width_n` block
split at `n = 13`, the tiling property — and step 53's tests pin each
on the composed descriptor explicitly.

Validated by 8 new lib unit tests in `psy::tests`:
`coder_partition_d5_span_anchor_rows` pins spec-anchored values at
`n ∈ {1, 12, 13, 32}` covering the lower-block edge, the
last/first row of each width-block, and the table top;
`…_rejects_edges_and_out_of_range` pins both edge-`None` cases
(`n = 0`, `n = 33`) plus out-of-range rejection
(`n ∈ {34, 64, u16::MAX}`);
`…_composes_underlying_accessors_for_every_in_range_index` pins per-row
composition agreement against `coder_partition_d5_line_range` and
`coder_partition_d5_width`; `…_inclusive_span_is_17_lines_everywhere`
pins the uniform 17-line inclusive span against
`CODER_PARTITION_D5_STRIDE + 1`;
`…_width_block_structure_is_preserved` pins the `width_n = 0`
(rows `n ∈ 1..=12`) / `width_n = 1` (rows `n ∈ 13..=32`) block
split through the composition; `…_tiles_the_band` pins the tiling
property (every span's `omega_high` equals the next span's
`omega_low`); `…_index_field_matches_input` pins that the
descriptor's `index` field echoes the input verbatim with no
off-by-one bleeding through from the `omega_low` row shift; and
`…_low_is_strictly_less_than_high` pins span non-degeneracy across
the table. Tests: 689 lib (was 681 baseline; +8 unit). No
external implementation consulted; only Table D.5 in
`docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
§"Table D.5 - Layer I and Layer II coder partition table" was read.

**Phase 2 step 52 (r251)** — Annex D Table D.5 `width_n` column
accessor. Phase 2 steps 49 / 50 / 51 (r248 / r249 / r250) landed the
33-row verbatim Table D.5 (Layer I / Layer II coder partition table)
and its two boundary-column readings: the dual-role `ωhigh_n` /
`ωlow_n` accessors at step 50 and the composed `(ωlow_n, ωhigh_n)`
inclusive line range at step 51. The third column of the row —
`width_n` — was only accessible by going through the
`CoderPartitionD5` row struct directly; r251 lifts it to a
table-level free function so callers track the same surface as the
boundary column:

* `coder_partition_d5_width(n)` returns `Some(width_n)` for
  `n ∈ 0..=32` and `None` otherwise.

The transcribed `width_n` column carries exactly two values: **0**
for the lower block (rows `n ∈ 0..=12`) and **1** for the upper
block (rows `n ∈ 13..=32`). The transition between the two blocks
is a single step at row 13 — no transitional row. The accessor is a
**pure rename** of the row struct's `width` field — no arithmetic
and no interpretation — matching
`coder_partition_d5(n).map(|r| r.width)` exactly.

The `width_n` column is structurally orthogonal to the partition
boundary column already exposed by step 51's
`coder_partition_d5_line_range`: the boundary column advances by a
uniform 16-line stride across every row pair, whereas `width_n` is
constant within each block. No boundary value or stride is consulted
by this accessor.

Validated by 7 new lib unit tests in `psy::tests`:
`coder_partition_d5_width_anchor_rows` pins the four spec-anchor
rows (`n ∈ {0, 12, 13, 32}` → `{0, 0, 1, 1}`);
`…_matches_row_field_for_every_in_range_index` pins full-table
parity with the row-field view;
`…_is_zero_for_lower_block_one_for_upper_block` pins the
constant-within-block structure across all 33 in-range partitions;
`…_rejects_out_of_range` pins the `None` return at
`n ∈ {33, 64, u16::MAX}`; `…_range_is_exactly_zero_or_one` pins
the `{0, 1}` value constraint; `…_transition_is_a_single_step_at_row_thirteen`
pins the single-step structural property (exactly one
neighbour-pair across the 32 transitions changes value, and it is
the row 12 → row 13 step going 0 → 1); and
`…_is_orthogonal_to_omega_boundary` pins the constant-within-block
orthogonality between the `width_n` column and the partition
boundary column. Tests: 681 lib (was 674 baseline; +7 unit). No
external implementation consulted; only the `width_n` column from
`docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
§"Table D.5 - Layer I and Layer II coder partition table" was
read.

**Phase 2 step 51 (r250)** — Annex D Table D.5 partition FFT-line
range accessor. The Phase 2 step 50 (r249) commit landed the
`ωhigh_n` and `ωlow_n` dual-role accessors for the partition
boundary column; r250 composes them into the single per-partition
FFT-line span accessor that downstream Model 1 / Model 2 threshold
aggregation needs:

* `coder_partition_d5_line_range(n)` returns the inclusive
  `(ωlow_n, ωhigh_n)` tuple of partition `n` for `n ∈ 1..=32`
  (and `None` outside).

The valid range is exactly the intersection of the two step-50
accessors' ranges: `coder_partition_d5_omega_low` covers
`n ∈ 1..=33`, `coder_partition_d5_omega_high` covers `n ∈ 0..=32`,
so the full span `(ωlow_n, ωhigh_n)` is recoverable from Table D.5
only for `n ∈ 1..=32` — i.e. **32 partitions of 16 lines each**,
inclusive at both ends. Two partitions are explicitly missing one
boundary each:

* `n = 0` — `ωlow_0` is **not** in Table D.5 (the column heading
  `ωlow_{n+1}` shifts the lower-boundary cell up by one). The
  accessor returns `None` verbatim; no synthetic lower boundary is
  invented. The "what is partition 0's lower boundary?" question
  remains a downstream DOCS-GAP for the eventual Model 1 / Model 2
  threshold-aggregation work that needs partition 0 in practice.
* `n = 33` — `ωhigh_33` is **not** in Table D.5 (the table tops
  out at row 32 with `ωhigh_32 = 513`). The accessor returns
  `None` verbatim. The structural anchor `ωlow_33 = 513` already
  lands as row 32's `ωlow_{n+1}` reading, so the table's top edge
  is the partition-32 / partition-33 boundary line.

The accessor is a **pure composition** of
`coder_partition_d5_omega_low` and `coder_partition_d5_omega_high`
— no arithmetic beyond the verbatim `n → n - 1` row shift that the
column heading's `ωlow_{n+1}` half already encodes inside
`coder_partition_d5_omega_low`. The `width_n` column is not
consulted.

Validated by 9 new lib unit tests in `psy::tests`:
`coder_partition_d5_line_range_anchor_rows` pins the four
spec-anchor partition spans (`n = 1` → `(1, 17)`, `n = 13` →
`(193, 209)`, `n = 14` → `(209, 225)`, `n = 32` → `(497, 513)`);
`coder_partition_d5_line_range_partition_zero_missing_low_boundary`
+ `…_partition_thirty_three_missing_high_boundary` pin the two
edge-`None` cases; `…_rejects_out_of_range` pins the `None`
return at `n ∈ {34, 64, u16::MAX}`; `…_low_le_high_for_all_in_range`
+ `…_strict_inequality_for_all_in_range` pin non-degeneracy across
the 32 recoverable partitions; `…_composes_omega_low_and_omega_high`
pins the verbatim composition contract; `…_uses_stride_plus_one_lines`
pins the uniform 17-line inclusive span (open span equals
`CODER_PARTITION_D5_STRIDE = 16` per partition); and
`…_partitions_tile_fft_line_band_two_to_513` pins the tiling
property (every partition's `ωhigh` equals the next partition's
`ωlow`, the band starts at line 1 and tops out at line 513).
Tests: 674 lib (was 665 baseline; +9 unit). Only the column heading
`ωlow_{n+1} / ωhigh_n` from
`docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
§"Table D.5 - Layer I and Layer II coder partition table" was
read.

**Phase 2 step 50 (r249)** — Annex D Table D.5 dual-role
boundary accessors. The Phase 2 step 49 (r248) commit landed the
full 33-row Table D.5 — Layer I / Layer II coder partition
table — as a verbatim `CODER_PARTITION_TABLE_D5: [CoderPartitionD5;
33]` keyed on the spec column `index n` plus the printed
`omega_boundary` (the cell under the dual-role column heading
`ωlow_{n+1} / ωhigh_n`) plus the `width_n` value. That row
accessor deliberately did **not** interpret the dual role —
callers had to apply the column-rename arithmetic themselves.
r249 lands the dual-role accessor surface so the two spec roles
are visible at the API level:

* `CoderPartitionD5::omega_high()` is a pure rename of the row's
  `omega_boundary` field under its `ωhigh_n` spec role — the
  upper boundary FFT-line index of partition `n` (where
  `n = self.index`).
* `CoderPartitionD5::omega_low_of_next()` is the same pure rename
  under the column heading's other role — the lower boundary
  FFT-line index of the **next** partition `n + 1`.
* `coder_partition_d5_omega_high(n)` is a table-level wrapper
  that returns the `ωhigh_n` reading of partition `n` for
  `n ∈ 0..=32` (and `None` outside).
* `coder_partition_d5_omega_low(n)` is a table-level wrapper
  that returns the `ωlow_n` reading of partition `n` for
  `n ∈ 1..=33` (and `None` outside, including `n = 0`).

The `n = 0` case is deliberate: the column heading
`ωlow_{n+1} / ωhigh_n` prints `ωlow_{n+1}` at row `n`, so the
table covers `ωlow_n` only for `n ∈ 1..=33` — partition 0's own
lower boundary `ωlow_0` is **not** in Table D.5. r249 does
**not** invent a default; the accessor returns `None` for `n = 0`
verbatim, and the "what is partition 0's lower boundary?"
question is logged as a downstream DOCS-GAP for the Specifier
round that will eventually need to use partition 0 in an actual
threshold aggregation. (The structural anchor `ωhigh_0 = 1`
already lands as the row-0 transcription, so callers that need
the upper boundary of partition 0 are unblocked.)

Every accessor is a pure column / row rename — no arithmetic
beyond the verbatim `n → n - 1` row shift the column heading's
`ωlow_{n+1}` half explicitly requires. The four accessors do not
read the `width_n` column.

Validated by 10 new lib unit tests in `psy::tests`:
`coder_partition_d5_omega_high_method_renames_omega_boundary` +
`coder_partition_d5_omega_low_of_next_method_renames_omega_boundary`
pin the two per-row methods against the verbatim
`omega_boundary` field on every row;
`coder_partition_d5_dual_role_methods_return_same_value` pins
the dual-role identity at the per-row level;
`coder_partition_d5_omega_high_table_accessor_anchor_rows` pins
the four spec-anchor `ωhigh_n` values (`ωhigh_0 = 1`,
`ωhigh_12 = 193`, `ωhigh_13 = 209`, `ωhigh_32 = 513`);
`coder_partition_d5_omega_high_table_accessor_matches_omega_boundary_for_all_rows`
pins the table accessor against `omega_boundary` for every
in-range index;
`coder_partition_d5_omega_high_table_accessor_rejects_out_of_range`
pins the `None` return at `n ∈ {33, 64, u16::MAX}`;
`coder_partition_d5_omega_low_table_accessor_anchor_rows` pins
the four `ωlow_n` spec anchors (`ωlow_1 = 1`, `ωlow_13 = 193`,
`ωlow_14 = 209`, `ωlow_33 = 513`);
`coder_partition_d5_omega_low_partition_zero_is_not_in_table`
pins the verbatim `None` return for `n = 0` (no default
invented);
`coder_partition_d5_omega_low_rejects_out_of_range` pins the
`None` return at `n ∈ {34, 64, u16::MAX}`; and
`coder_partition_d5_omega_low_n_plus_1_equals_omega_high_n`
exercises the table-wide dual-role identity
`ωlow_{n+1} == ωhigh_n` across every `n ∈ 0..=32`. Tests: 665
lib (was 655 baseline; +10 unit). Only the column heading
`ωlow_{n+1} / ωhigh_n` from
`docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
§"Table D.5 - Layer I and Layer II coder partition table" was
read.

Remaining Phase 2 work: Model 1 Steps 1–5 (1024-sample FFT +
SPL conversion + tonality classifier + decimation + masker
selection — these consume the PNG-only Annex D Tables D.1 / D.2
that this crate's docs collaborator has staged as PNG renders
only; OCR to text is the DOCS-GAP) plus the Bark / Hz / line
mapping needed to feed the §D.1 Step 6 / Step 7 primitives that
landed this round, the full Annex D Model 2 (calculation
partition table D.3 — also PNG-only — plus the Model 2
spreading-function `tmpy` line that is typeset as image in the
PDF and is not text-extractable), intensity-stereo encode
(§2.4.3.4.9.3), LSF / MPEG-2.5 encode (blocked on the
`MPEG-2.5-GAP.md` observer-trace items for scalefactor-band
tables / Huffman mapping / frame-size validation at low rates),
and stereo / LSF decode through the trait wrapper.

### Not yet implemented

MPEG-2.5 decode through the `Decoder` trait wrapper (MPEG-1 lands
in r141, MPEG-1 stereo in r177 step 36, and MPEG-2 LSF mono +
stereo in r183 step 37 — see above; the trait wrapper's header
guard now accepts `MpegVersion::Mpeg1` and `MpegVersion::Mpeg2`
and rejects `MpegVersion::Mpeg25` only, pending the
`MPEG-2.5-GAP.md` observer-trace items — scalefactor-band tables /
Huffman table mapping / low-rate frame-size validation at
8 / 11.025 / 12 kHz). The encoder is **Phase 1
framing + Phase 2 steps 1–33 (forward MDCT primitive + analysis
windowing + forward overlap split + polyphase analysis subband
filterbank + §2.4.3.4.7 quantization primitive + §C.1.5.4.4
inner-loop `global_gain` search + exact §C.1.5.4.4.5/.8 Huffman bit
count + §2.4.1.7 Huffman bit emission + §2.4.1.7 main-data assembly
+ §2.4.2.7 cross-frame bit-reservoir scheduling with
`main_data_begin > 0` + stream-level PCM → MP3 driver + §C.1.5.4.3
outer (distortion-control) loop + `oxideav_core::Encoder` trait
wiring + opt-in Xing / Info VBR information-frame emission + true-VBR
per-frame bitrate + Xing TOC auto-fill + opt-in §2.4.3.1 CRC-16 frame
protection + independent-stereo (`ChannelMode::Stereo` /
`ChannelMode::DualChannel`) encode through the trait wrapper +
§2.4.3.4.9.2 joint-stereo MS encode + §C.1.5.4.3 `scalefac_scale 0→1`
escalation in the outer loop + §C.1.5.4.3.4 preemphasis decision in
the outer loop + §2.4.2.3 joint-stereo auto MS/LR per-frame picker +
trait-factory wrappers for the auto MS/LR picker + §2.4.3.4.10.2
forward short-block MDCT path with `Mp3Encoder::force_short_blocks_for_testing`
toggle + §2.4.2.7 forward mixed-block MDCT path with
`Mp3Encoder::force_mixed_blocks_for_testing` toggle + §C.1.5.4.4.8
linbits-reach filter in the Huffman table chooser + MPEG-2.5 frame
header parse + writer + sample-rate-table dispatch + §C.1.5
signal-driven attack-detection heuristic and §C.1.5.2
`LONG → START → SHORT → STOP → LONG` transition state machine via
`Mp3Encoder::enable_auto_block_type` + §C.1.5.4.3 pure-short
outer-loop primitive `outer_loop_search_short` with per-(sfb,
window) `scalefac_s` amplification and bounded `subblock_gain`
search + auto-block-type × outer-loop integration so the auto
scheduler runs `outer_loop_search_short` on Short granules and
`outer_loop_search_long` on Long / Start / End granules + §C.1.5.4.3
mixed-block outer-loop primitive `outer_loop_search_mixed` composing
the long-region amplifier over `sf.long[0..=7]` with the short-region
per-(sfb, window) amplifier over `sf.short[3..=11][..]` and wiring
into the stream encoder so `force_mixed_blocks_for_testing` +
`new_with_outer_loop` runs the new distortion-control path on every
granule + §C.1.5.4.3 long-family transition-skeleton wiring so
`outer_loop_search_long` accepts `block_type ∈ {Long, Start, End}`
and the auto + outer-loop dispatcher routes Start / End onto the
same primitive (no more fixed-gain fallback for any block-type the
scheduler emits) + §2.4.3.4.10.3 auto-block-type mixed-block
promotion via the new clean-room PCM-domain `MixedClassifier`
(one-tap low-pass + subframe-energy stability ratio) and the
`BlockTypeStateMachine::step_with_mixed` extension, opt-in through
`Mp3Encoder::enable_auto_block_type_with_mixed` and reusing the
r159 `outer_loop_search_mixed` primitive via the
`gc_template.mixed_block_flag` discriminator so mixed-promoted
granules also run the §C.1.5.4.3 distortion-control loop +
§2.4.3.4.9 independent-stereo widening of the block-type override
toggles so `force_short_blocks_for_testing`,
`force_mixed_blocks_for_testing`, `enable_auto_block_type`, and
`enable_auto_block_type_with_mixed` accept
`ChannelMode::Stereo` / `ChannelMode::DualChannel` in addition to
mono + §2.4.3.4.9 cross-channel-MS block-type agreement that
widens the same four toggles onto MS-stereo joint modes, with the
auto-path agreement enforced by OR-folding per-channel attack /
mixed-classifier flags into a single shared `scheduler[0]` and
mirroring its emission across both channel slots of
`block_type_per_gc[gr]` / `mixed_per_gc[gr]` + Annex D
threshold-in-quiet **per-band threshold scaffold** —
`crate::psy::XminThresholds` + `outer_loop_search_long_per_band`
primitive + `Mp3Encoder::set_per_band_xmin` opt-in so the
long-block outer loop reads `xmin[sfb]` from the Annex D
Table D.1 *threshold in quiet* curve (sampled via monotone
piecewise-linear interpolation through the textually-transcribed
anchors at 62.5 Hz / 33.44 dB, the i=2..5 rows, the i=51 prose
minimum near 3.375 kHz / −4.97 dB, and 15 kHz / 51.04 dB — the
PNG-only inner rows of D.1a–f are deliberately not OCR'd this
round) with the §D.1 Step 3 `−12 dB` offset at ≥ 96 kbit/s +
r197 §C.1.5.4.3 pure-short outer-loop per-cell threshold —
`outer_loop_search_short_per_band` primitive accepts a
`[[f64; SHORT_WINDOWS]; SHORT_SFB]` per-cell `xmin` matrix that
every §C.1.5.4.3.5 amplification + §C.1.5.4.3.6 termination cell
test reads, with `outer_loop_search_short` refactored to a thin
scalar shim that broadcasts the uniform threshold into a uniform
matrix (byte-for-byte equivalent to the pre-r197 inline body),
plus a new `XminThresholds::threshold_in_quiet(SR, version,
br_per_ch)` constructor that fills BOTH the `long[sfb]` cells
(identical derivation to the r194 `threshold_in_quiet_long`) AND
the `short[sfb][win]` cells (each short SFB's centre frequency
read from the same Annex D anchors, broadcast across the three
windows of the band — Annex D Table D.1 is a function of
frequency only) and the §D.1 Step 3 offset applied to both
shapes, wired through the `stream_encoder` dispatch so
`BlockType::Short if !mixed_block_flag` granules route onto the
new per-band primitive whenever `set_per_band_xmin` has installed
a matrix + r204 §C.1.5.4.3 mixed-block outer-loop per-band
threshold — `outer_loop_search_mixed_per_band` primitive accepts a
`xmin_long: &[f64; LONG_SFB]` long-region per-band vector (sfb
0..=7 are read) AND a
`xmin_short: &[[f64; SHORT_WINDOWS]; SHORT_SFB]` short-region
per-cell matrix (sfb 3..=11 cells are read) that every §C.1.5.4.3.5
amplification + §C.1.5.4.3.6 termination cell test reads, with
`outer_loop_search_mixed` refactored to a thin scalar shim that
broadcasts the uniform threshold into uniform long + short
vectors (byte-for-byte equivalent to the pre-r204 inline body);
the stream-encoder dispatch routes `BlockType::Short if
mixed_block_flag` granules onto the new per-band primitive,
consuming `XminThresholds::mixed_long` for the long region and
`XminThresholds::mixed_short` for the short region whenever
`set_per_band_xmin` has installed an `XminThresholds` (already
populated by the r197 `XminThresholds::threshold_in_quiet`
constructor from the same Annex D anchors) + r219 Annex D
Model 1 §D.1 Step 6 masking-function `vf` + masking-index
`av_tm` / `av_nm` + Step 7 global-threshold summation
primitives (`masking_function_vf`, `masking_index_tonal`,
`masking_index_non_tonal`, `individual_masking_threshold_db`,
`global_masking_threshold_db`) accepting a slice of
`Masker { kind, z_bark, spl_db }` and a `LTq(i)` scalar
threshold-in-quiet anchor, reproducing the verbatim spec
equations including the half-open `[-3, 8)` masker window +
r224 Annex D Model 1 §D.1 Step 4 critical-band-boundary
**Tables D.2a–f** transcribed verbatim from the staged
text-extracted docs file (no PNG OCR involved — the six tables
were textually transcribed in `mp3-annex-d-psychoacoustic-extracts.md`
as of docs commit `dc78918`): `CriticalBandBoundary { no,
index_fcb, frequency_hz, z_bark }` rows for the six
(Layer I/II × 32/44.1/48 kHz) combinations, dispatched by
`critical_band_boundaries(layer, fs)` on the new typed key
`AnnexDSamplingRate { Hz32000, Hz44100, Hz48000 }` (Layer III
returns `None` per Annex D's normative Layer-I/II scope; the
spec's clause C.1.5.3.2.1 Layer-III spreading-function override
reuses the Layer-I/II tables, so a Layer III caller passes the
Layer parameter explicitly), with an FFT-line-to-band locator
`band_of_fft_line(boundaries, fft_line_index)` for §D.1 Step 4
masker placement, the `D2E_BAND_17_BARK_IS_ILLEGIBLE` marker
preserving the staged-doc-noted clipped `z_bark` digit in
D.2e row 17 as the legible-prefix `16.11` (the doc's prose
estimate `16.116` is explicitly NOT adopted as a verbatim source
value), and 13 unit tests covering: per-table band-count + `no`
contiguity (24/25/26/25/27/27 rows = prose count + 1),
first/last-row anchor cross-checks against the docs file,
strict (`index_fcb`, `frequency_hz`, `z_bark`) monotonicity,
`AnnexDSamplingRate::from_hz` round-trip + non-Annex-D-rate
rejection, the six-way (Layer, Fs) dispatch table, the
band-of-FFT-line locator (zero-rejection, bottom-band single
line, mid-band ranges, top-band edge inclusion, out-of-range
None), the D.2e illegibility cell read-back, and the
cross-Layer sanity check that D.2d's first band edge sits
below D.2a's first band edge (Layer II's longer window resolves
a lower starting band edge) + r229 Annex D Model 1 §D.1 Step 4
masker placement helper + Step 7 nearby-masker Bark-window
range pre-filter. `masker_at_band(boundaries, band_no, kind,
spl_db)` composes Step 45's `CRITICAL_BANDS_D2*` slices and
Step 44's `Masker` carrier: the masker is placed at the
band's verbatim top `z_bark` coordinate per the §D.1 Step 4
rule (the band's top Bark coordinate is the spec's masker
representative position), and the caller-supplied SPL (the
value Steps 1-3 will produce from the FFT spectrum) is wrapped
into the typed `Masker { kind, z_bark, spl_db }` carrier
already consumed by `individual_masking_threshold_db` and
`global_masking_threshold_db`. The §D.1 Step 7 nearby-masker
window predicate `masker_in_step7_window_of_line(masker,
z_i_bark)` exposes the spec's "for a given i the range of j
may be reduced to maskers within −8…+3 Bark of i" optimisation
as a single inlined Bark-distance test: a caller computing a
sparse per-line `LTg(i)` map can pre-shrink its masker slice
once per line via `filter()` and skip the
`individual_masking_threshold_db` call (and its branch on the
`vf` piecewise function) for every out-of-range masker. The
two bounds are exposed as named `pub const`s
(`STEP7_NEARBY_MASKER_DZ_LO_FROM_LINE = -8.0` open low,
`STEP7_NEARBY_MASKER_DZ_HI_FROM_LINE = 3.0` closed high) and
the predicate is the open-low, closed-high intersection of
the spec text with the §D.1 Step 6 `vf` `[-3, 8)` half-open
`dz` window — the set of maskers it lets through is exactly
the set for which `individual_masking_threshold_db` returns
`Some`. 12 new unit tests cover: band-0 first-row anchor
(D.2a `z = 0.617`, SPL passthrough) and last-row anchor
(D.2a band 23 `z = 23.923`) for placement; out-of-range band
index returns `None`; cross-table dispatch (D.2d band 0 sits
below D.2a band 0 — Layer II longer window); a self-placement
composition smoke test (place at band 5, evaluate LT at the
masker's own `z` → `SPL + av_tm`); a loud-local-masker
composition smoke test (place at D.2c band 10 with 80 dB SPL
→ `LTg` >> `LTq`); the §D.1 Step 7 window constants reproduce
the spec text (`-8` open, `+3` closed); in-range edge cases
(centred, 2 Bark above, 5 Bark below); high-edge inclusivity
at `dz_from_line = +3` exactly + exclusion at `+3.0001`;
low-edge exclusivity at `dz_from_line = -8` exactly +
inclusion just above; a 0.25-Bark masker-position sweep
verifying the predicate matches `individual_masking_threshold_db`'s
`Some` set on every sample; a functional invariant
(`filter()`-then-`global_masking_threshold_db` produces the
same `LTg(i)` as feeding the full slice — the pre-filter is
mechanically equivalent to dropping `vf = None` contributions
from the energy sum) + r232 Annex D Model 1 §D.1 Step 5
decimation primitives bridging Step 4 placement (r229) and
Step 6 individual-masking-threshold calculation (r219).
`masker_above_threshold_in_quiet(masker, ltq_db)` is the
verbatim Step 5(a) threshold-in-quiet screening predicate
(`X_tm(k) >= LTq(k)` / `X_nm(k) >= LTq(k)`, identical
comparison for tonal and non-tonal maskers) and
`decimate_tonal_within_half_bark(maskers)` is the Step 5(b)
tonal-cluster decimation sieve: a sliding window of width
`STEP5_TONAL_DECIMATION_WINDOW_BARK = 0.5` Bark (named
`pub const`, verbatim spec text "a sliding window in the
critical band domain is used with a width of 0,5 Bark")
collapses every cluster of two-or-more tonal maskers within
**strictly less than** 0.5 Bark of each other to the
loudest member of the cluster (input-order stable on tied
SPLs — first-encountered wins). Non-tonal maskers pass
through unchanged because §D.1 Step 4(c) already yields at
most one non-tonal masker per critical band, and the
output preserves the caller's original slice order (the
non-tonal subset is interleaved back at its original
positions). 17 new unit tests cover: Step 5(a) above-LTq
keep / below-LTq drop / at-LTq inclusive boundary on both
tonal and non-tonal maskers; the
`STEP5_TONAL_DECIMATION_WINDOW_BARK` constant reads back
0.5; Step 5(b) edge cases (empty input, singleton
passthrough, pair within window keeps loudest, pair at
exactly 0.5 Bark both survive per the spec's strict
"less than" wording, pair outside window both survive,
non-tonal cluster pass-through unchanged, three-member
tonal cluster collapses to single loudest, two separated
clusters collapse independently, ties resolve to
first-encountered for output stability, unsorted input
still clusters correctly, mixed tonal / non-tonal
preserves non-tonal in place); a compositional invariant
(Step 5(a) then Step 5(b) reproduces the spec's full
Step 5 sieve); and an end-to-end smoke that pipes
Step 5(a) + 5(b) into Step 7
`global_masking_threshold_db` and confirms the result
matches direct evaluation on the decimated slice + r237
Annex D Model 2 §C.1.5.3.2.1 Layer III spreading-function
primitives. The spec describes Layer III's modification of
the Model 2 spreading function as two branches —
`tmpy = 3.0 * (j - i)` for `j >= i` (upward / on-diagonal)
and `tmpy = 1.5 * (j - i)` for `j < i` (downward) — followed
by the linear conversion `sprdngf(i, j) = 10^(tmpy/10)`
and the clamp "Only spreading-function values greater than
1e-6 are used; all others set to zero".
`model2_layer3_spread_db(i, j)` returns the per-partition dB
value and `model2_layer3_spread_linear(i, j)` returns the
clamped linear factor (with the spec's `1.0e-6` threshold
exposed as `MODEL2_LAYER3_SPREAD_LINEAR_MIN`); the
on-diagonal value is exactly `1.0`, the upward branch's
linear factor grows above unity, and the downward branch
falls below unity until the clamp boundary at `j - i = -40`
(`tmpy = -60 dB` → exactly `1.0e-6` linear, collapsed to 0
by the spec's strict `> 1e-6` survival comparison). 9 new
unit tests cover: diagonal returns zero across the Model 2
partition range; upward branch matches the verbatim
`3.0 * (j - i)` formula at +1 / +5 / +20 partition steps;
downward branch matches `1.5 * (j - i)` at -1 / -4 / -20
steps; diagonal linear factor is exactly 1.0; upward linear
factor strictly exceeds 1.0 and grows monotonically with
distance (`10^0.3 ≈ 1.9953` at +1); downward linear factor
is strictly below 1.0 and shrinks monotonically (`10^-0.15
≈ 0.7079` at -1); clamp boundary holds the spec's strict
comparison (-39 survives, -40 collapses to exact zero, -50
stays clamped); the `1.0e-6` constant matches the spec's
verbatim figure; the upward branch's diagonal value agrees
with the downward-branch extension + r247 Annex D Table D.5
Layer I / Layer II coder partition table. The 33-row
partition table is transcribed verbatim from the staged
`docs/audio/mp3/mp3-annex-d-psychoacoustic-extracts.md`
extract of ISO/IEC 11172-3:1993 Annex D clause D.2 (PDF
page 145 / printed p.139). `CoderPartitionD5` carries the
three spec columns per row — `index` (partition number
`n = 0..=32`), `omega_boundary` (the FFT-line index the
spec table prints under the dual-role heading
`ωlow_{n+1} / ωhigh_n`), and `width` (the spec's `width_n`
column, 0 for rows 0..=12 and 1 for rows 13..=32). The full
table lands as
`CODER_PARTITION_TABLE_D5: [CoderPartitionD5; 33]` and
`coder_partition_d5(n)` is a thin row-index accessor that
returns `None` outside the spec range. The uniform 16-line
stride between consecutive rows is exposed as
`CODER_PARTITION_D5_STRIDE: u16 = 16`. 10 new unit tests
cover row count, index contiguity, four spec-anchor rows
(0 = `(1, 0)`, 12 = `(193, 0)`, 13 = `(209, 1)`,
32 = `(513, 1)`), strict monotonicity of the ω column,
uniform 16-line stride across every transition, the
width-0 / width-1 split at row 13, the row-accessor
round-trip on every in-range index, accessor rejection on
out-of-range indices, the 1-based FFT-line indexing
convention (row 0 carries ω = 1), and the top-of-table pin
(row 32 carries ω = 513 = 1 + 32·16, matching the
1024-sample FFT's 1..=513 one-based half-spectrum) + r285
**MPEG-2 LSF + MPEG-2.5 stream-level encode**: `Mp3Encoder::new`
(and the registry `make_encoder` path) accepts 16 / 22.05 / 24 kHz
(MPEG-2 LSF) and 8 / 11.025 / 12 kHz (MPEG-2.5) — one 576-sample
granule per frame, `slots_per_frame` 72 (ISO/IEC 13818-3
§2.4.3.2), the §2.4.1.7 LSF side-info layout, the LSF §2.4.2.3
bitrate ladder (`LSF_L3_BITRATE_LADDER_KBPS`), the LSF CBR
padding ladder, the 255-byte reservoir cap, LSF CRC
(`crc16_layer3_lsf` over header bits 16..31 + the full 72 /
136-bit LSF side info), VBR on the LSF ladder, MS joint stereo,
forced short / mixed blocks, and the outer loop writing
`scalefac_compress = 399` (`OUTER_LOOP_SCALEFAC_COMPRESS_LSF` —
the §2.4.3.2 slen derivation (4, 4, 3, 3) over partition
(6, 5, 5, 5) reproducing the MPEG-1 value-15 caps and part2
cost; an outer-loop preflag is folded into the long
scalefactors since sub-500 `scalefac_compress` cannot carry the
flag). The 13818-3 Table B.2 scalefactor-band tables for the
three LSF rates (long + short) land in `requantize` and back
every band-mapping consumer (requantizer, Huffman region split,
encoder region / intensity walks);
`tests/lsf_reference_pcm.rs` pins the 22.05 kHz staged
fixture's decode to its reference `expected.wav` at 0.000026
steady-state normalized RMS error, and
`tests/lsf_encoder_roundtrip.rs` (10 tests) round-trips the
LSF / MPEG-2.5 rates at 56–88 dB self-decode PSNR with
black-box `ffmpeg` / `mpg123` cross-decodes recovering the
exact test tones at every MPEG-2 rate
+ r288 **Model 2 psychoacoustic threshold wired into the outer
loop** (Phase 2 step 89). The §C.1.5.3.2.1 Layer III analysis
chain — already producing the Figure C.6.c/d per-scalefactor-band
masking threshold `thm(sb)` through `psy::Model2Layer3State::
process` — now feeds the encoder's distortion-control loop.
`XminThresholds::from_layer3_granule` maps a granule's per-band
threshold (long `thm[21]` + the new short `thm_short[3][12]`
field) into the outer-loop `xmin(sb)` vector, preserving every
per-band ratio exactly: a single multiplicative rescale anchors
the granule's geometric-mean threshold to
`DEFAULT_OUTER_LOOP_THRESHOLD`, so the loop's convergence
dynamics stay in the same dex as the LTq / uniform paths while
the perceptual ordering Model 2 produced is untouched. Silent
bands floor to the smallest rescaled positive threshold (never
`xmin = 0`); a fully silent granule yields the uniform default.
`Mp3Encoder::set_per_band_xmin_from_model2(state, granule)` is
the end-to-end convenience — one 576-sample granule through a
caller-owned `Model2Layer3State` (threaded across granules for
the §D.2.1 FFT-history requirement, one per channel) installs the
*signal-dependent* masking threshold in place of the
*signal-independent* threshold-in-quiet bowl. Restricted to the
three staged Annex D Model 2 rates (32 / 44.1 / 48 kHz); other
rates and non-576 granules return the new
`StreamEncodeError::Model2AnalysisUnsupported`. 5 new unit tests
cover ratio preservation, the geometric-mean scale anchor, the
silent / zero-band floor, the outer-loop / rate / granule guards,
and an end-to-end install producing a spectrally-shaped (not
flat) threshold — it
still lacks
the §C.1.5.3.2 driver that runs Model 2 *automatically* inside
`push_samples` per granule (the threshold is installed via the
explicit `set_per_band_xmin_from_model2` entry point this round;
auto-wiring it into the per-granule encode walk — including the
MS-vs-independent channel split and the auto-block-type lookahead
— is the follow-up),
LSF / MPEG-2.5 Model 2 (the Annex D Tables D.3 / D.4 / C.7 / C.8
are staged only for 32 / 44.1 / 48 kHz), and
externally-valid MPEG-2.5 encode (the encoder emits
self-consistent MPEG-2.5 streams this crate's decoder
round-trips, but the MPEG-2.5-specific scalefactor-band tables
remain the documented `MPEG-2.5-GAP.md` placeholder — external
decoders place the bands differently at 8 / 11.025 / 12 kHz
until the gap's observer-trace tables land).

**Spec gap (alias reduction, mixed blocks):** §2.4.3.4.10.1 scopes the
stage purely on `block_type` ("block-type != 2" applies; "block-type ==
2 (short block)" does not). A *mixed* block is `block_type == 2` but
codes its two lowest subbands long; the standard gives no separate rule
for that long region, so this crate follows the literal text and does
not alias-reduce mixed blocks. A clarifying note in §2.4.3.4.10.1 on the
mixed-block long region would remove the ambiguity.

## Fuzzing

A [cargo-fuzz](https://github.com/rust-fuzz/cargo-fuzz) / libFuzzer
harness under `fuzz/` hardens the Layer III **decode** path against
malformed bitstreams. It carries two targets:

- **`decode`** — drives attacker bytes through the registered
  `Decoder` trait (`send_packet` / `receive_frame` / `flush` /
  `reset`) across a multi-packet stream, so the cross-packet bit
  reservoir (`main_data_begin` lookback, §2.4.3.4) plus the IMDCT and
  synthesis overlap carry-over are exercised, not just a cold-start
  frame. Each crafted packet carries a structurally-valid 4-byte
  header (so the deep chain past frame-sync is actually reached) whose
  every field is attacker-chosen, with the CRC / side-info / main-data
  slot filled from attacker bytes; raw-byte packets cover the
  short-frame and bad-sync rejections.
- **`granule`** — drives the per-granule decode primitives directly
  below the trait surface (`parse_side_info` → `decode_scalefactors` →
  `decode_huffman` big_values/count1 → `requantize` → `alias_reduce` →
  `imdct_granule` → `synth_granule`), since those are reached through
  the trait only once the reservoir lookback is satisfiable. The
  side-info parse yields real attacker-controlled `GranuleChannel`
  parameters (`part2_3_length`, `big_values`, `block_type`,
  `region*_count`, `table_select`, `subblock_gain`, …) feeding the
  dequant / window-overlap / filterbank math on every iteration.

The contract under test is panic-freedom: every entry returns a value
or an `Err`, never panicking, overflowing in a debug build, or
indexing out of bounds. Round 289 ran both targets for 180 s each
(≈759k iterations on `granule`, ≈335k on `decode`, ≈1.09M total) with
zero findings.

Run a target with `cargo +nightly fuzz run decode` (or `granule`)
from the crate root.

## Benchmarks

Criterion benchmarks for the Layer III **decode** hot path live under
`benches/`; the ranked hotspot map is in
[`BENCHMARKS.md`](./BENCHMARKS.md). They synthesise their input PCM
in-bench and round-trip it through the crate's own `Mp3Encoder` (no
committed fixtures), then time only the decode side.

- **`decode`** — whole-stream decode of a pre-encoded mono stream,
  timed both through the registered `Mp3CoreDecoder` trait object and
  through the bare per-stage chain (the two are within measurement
  noise — trait dispatch + `AudioFrame` packing add nothing measurable
  over the DSP).
- **`decode_stages`** — isolates each stage (side-info parse,
  scalefactors, Huffman big-values/count1, requantize, alias, IMDCT,
  synthesis filterbank) over one captured 20-frame / 40-granule batch.

The ranking is dominated by the back-end DSP: the **synthesis
filterbank (~62 %)** and the **IMDCT (~31 %)** together account for
~93 % of decode time; the entire bitstream-parse / entropy /
requantization front half (Huffman, requantize, scalefactors, alias,
side-info) sums to under 7 %. This round added the harness and the
ranking only — decoded PCM is unchanged.

Run with `cargo bench -p oxideav-mp3 --bench decode` (or
`decode_stages`).

## License

MIT — see [LICENSE](./LICENSE).
