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

Codebook coverage so far: tables 0..=13 (the quad tables A/B plus the
small/medium big-values tables — 1×1, 2×2, 3×3, 4×4, 6×6, 8×8, and
the 16×16 table 13) transcribed by hand from the Annex B render of
ISO/IEC 11172-3:1993 (PDF pages 54..=57 rendered with `pdftoppm`).
Tables 4 and 14 are spec-marked "not used" and are rejected. The
16×16 codebooks 15, 16, 24 and the `linbits` aliases 17..=23,
25..=31 are pending a follow-up transcription round; calling them
returns `HuffmanError::TableNotYetTranscribed`. Twenty-seven
huffman-stage unit tests cover spec-derived bitstreams plus per-table
prefix-freeness + Kraft-inequality self-checks on every transcribed
codebook.

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
  interleave; the §2.4.3.4.8 reorder is a later stage.
- Band→scalefactor mapping uses the Table B.8 long- and short-block
  start indices for 32 / 44.1 / 48 kHz (LSF reuses the long layouts
  this round; LSF-specific band tables are deferred). 19 unit tests
  cover long-block unit-gain identity, the global-gain 4-step doubling,
  the scalefactor and `scalefac_scale` terms, the preflag/pretab effect
  (on and off), per-window `subblock_gain` and `scalefac_s`, short-band
  interleave, the mixed-block split, the LSF path, sign preservation,
  and a large-magnitude finiteness check.

### Not yet implemented

Short-block reordering (§2.4.3.4.8), stereo processing (MS / intensity,
§2.4.3.4.9), the IMDCT, and the synthesis filterbank, plus any encoder.
`register()` is a no-op until a decoder/demuxer is wired up. The
remaining big-values codebooks (15, 16, 24 and their 17..=23 / 25..=31
`linbits` aliases) are still a pending main-data sub-stage.

## License

MIT — see [LICENSE](./LICENSE).
