# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the crate adheres
to [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Other

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
  consulted; no external reference implementation read.
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
  consulted; no external reference implementation read.
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
  consulted; no external reference implementation read.
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
  clean. No external implementation consulted (the corpus, the
  sweep, and the metric are all derived from the
  `attack_detect` module's own clean-room reasoning, extending the
  r165 calibration along the threshold axis).
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
  baseline). No external implementation consulted.
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
  integration). No external implementation consulted.
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
  No external implementation consulted.

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
  integration). No external implementation consulted.

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
  +11 unit + 4 integration). No external implementation consulted;
  every rule is derived from §C.1.5.4.3 / §2.4.2.7 / §2.4.3.4.7.1 of
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
  escalation branch on a cap-would-terminate fixture. No external
  implementation consulted.

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
    `0.5`). Default ratio `10.0`; module docs explain tuning. No
    external reference was consulted — every constant is justified
    by the clean-room reasoning in the module's preamble.
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
