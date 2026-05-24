//! Layer III **alias reduction** — the §2.4.3.4.10.1 stage that runs
//! after §2.4.3.4.8 reordering and immediately before the IMDCT of the
//! §2.4.3.4.10 synthesis filterbank.
//!
//! # What alias reduction does (ISO/IEC 11172-3:1993 §2.4.3.4.10.1)
//!
//! The hybrid filterbank's polyphase stage and the per-subband IMDCT
//! introduce aliasing between adjacent subbands. For **long** block-type
//! granules (`block_type != 2`) the spec removes it with eight butterfly
//! operations across each pair of neighbouring 18-line subband blocks,
//! per the spec's pseudo code:
//!
//! ```text
//! for (sb = 1; sb < 32; sb++)
//!     for (i = 0; i < 8; i++) {
//!         xar[18*sb-1-i] = xr[18*sb-1-i]·cs[i] - xr[18*sb+i]·ca[i]
//!         xar[18*sb+i]   = xr[18*sb+i]·cs[i]   + xr[18*sb-1-i]·ca[i]
//!     }
//! ```
//!
//! The indices label the 576 frequency lines of a granule in increasing
//! frequency order (line 0 lowest, 575 highest). Each butterfly mixes a
//! line just below a subband boundary (`18·sb-1-i`) with a line just
//! above it (`18·sb+i`); both outputs are computed from the **original**
//! inputs, so the stage reads `xr` and writes a fresh `xar`.
//!
//! # The butterfly coefficients (Table B.9)
//!
//! Table 3-B.9 gives the eight raw aliasing coefficients `c[i]`:
//!
//! | i | 0    | 1     | 2    | 3     | 4     | 5     | 6      | 7      |
//! |---|------|-------|------|-------|-------|-------|--------|--------|
//! | c | −0.6 | −0.535| −0.33| −0.185| −0.095| −0.041| −0.0142| −0.0037|
//!
//! The butterfly coefficients are derived from them:
//!
//! ```text
//! cs[i] = 1 / sqrt(1 + c[i]²)      ca[i] = c[i] / sqrt(1 + c[i]²)
//! ```
//!
//! # Short blocks
//!
//! §2.4.3.4.10.1 scopes the stage with two complementary statements:
//! "For long block-type granules (block-type != 2) the input to the
//! synthesis filterbank is processed for alias reduction" and "Alias
//! reduction is not applied for granules with block-type == 2 (short
//! block)." Both are phrased purely on `block_type`, so this module
//! applies alias reduction exactly when `block_type != 2` and passes the
//! granule through unchanged when `block_type == 2`.
//!
//! A **mixed** block (`block_type == 2 && mixed_block_flag`) codes its
//! lowest two subbands with a long window but is still `block_type == 2`;
//! the spec's literal `block_type`-only test therefore excludes it.
//! ISO/IEC 11172-3:1993 does not state a separate rule for the long
//! region of a mixed block, so this module follows the literal text and
//! does not alias-reduce mixed blocks. See the module's reported spec gap.

use crate::side_info::{BlockType, GranuleChannel};

/// Number of frequency lines in a granule-channel (32 subbands × 18).
pub const NUM_LINES: usize = 576;

/// Number of polyphase subbands per granule.
const NUM_SUBBANDS: usize = 32;

/// Lines per subband block fed to one IMDCT (§2.4.3.4.10).
const LINES_PER_SUBBAND: usize = 18;

/// Number of butterflies straddling each subband boundary (Table B.9).
const NUM_BUTTERFLIES: usize = 8;

/// Table 3-B.9 raw aliasing coefficients `c[i]`, `i = 0..8`. Transcribed
/// from ISO/IEC 11172-3:1993 Annex B, Table 3-B.9 (PDF page 71, printed
/// page 65).
pub const ALIAS_C: [f64; NUM_BUTTERFLIES] = [
    -0.6, -0.535, -0.33, -0.185, -0.095, -0.041, -0.0142, -0.0037,
];

/// The eight butterfly `cs[i] = 1 / sqrt(1 + c[i]²)` multipliers derived
/// from [`ALIAS_C`] per the Table B.9 derivation, as `f32` to match the
/// requantized line buffer.
#[must_use]
pub fn alias_cs() -> [f32; NUM_BUTTERFLIES] {
    let mut cs = [0.0f32; NUM_BUTTERFLIES];
    for (i, &c) in ALIAS_C.iter().enumerate() {
        cs[i] = (1.0 / (1.0 + c * c).sqrt()) as f32;
    }
    cs
}

/// The eight butterfly `ca[i] = c[i] / sqrt(1 + c[i]²)` multipliers
/// derived from [`ALIAS_C`].
#[must_use]
pub fn alias_ca() -> [f32; NUM_BUTTERFLIES] {
    let mut ca = [0.0f32; NUM_BUTTERFLIES];
    for (i, &c) in ALIAS_C.iter().enumerate() {
        ca[i] = (c / (1.0 + c * c).sqrt()) as f32;
    }
    ca
}

/// Apply Layer III alias reduction (§2.4.3.4.10.1) to one
/// granule-channel's reordered frequency lines.
///
/// * `xr` is the `[f32; 576]` buffer in subband order — the output of
///   [`crate::reorder::reorder`] (or the requantized buffer directly for
///   long blocks, which reorder passes through unchanged).
/// * `gc` supplies `block_type` (and `window_switching_flag`): a granule
///   with `block_type == 2` (short or mixed) is returned unchanged; every
///   other block type alias-reduces the whole spectrum (31 subband
///   boundaries × 8 butterflies).
///
/// Returns a fresh `[f32; 576]` (`xar` in the spec). Every butterfly is
/// computed from the original `xr` values.
#[must_use]
pub fn alias_reduce(xr: &[f32; NUM_LINES], gc: &GranuleChannel) -> [f32; NUM_LINES] {
    let mut xar = *xr;

    let is_short = gc.window_switching_flag && gc.block_type == BlockType::Short;
    if is_short {
        // §2.4.3.4.10.1: alias reduction is not applied for granules with
        // block-type == 2 (short block). The spec's test is on
        // `block_type` only, so a mixed block (also block_type == 2) is
        // excluded too.
        return xar;
    }

    let cs = alias_cs();
    let ca = alias_ca();
    for sb in 1..NUM_SUBBANDS {
        let boundary = LINES_PER_SUBBAND * sb;
        for i in 0..NUM_BUTTERFLIES {
            // `lo` is the line just below the boundary, `hi` just above.
            let lo = boundary - 1 - i;
            let hi = boundary + i;
            // Read originals from `xr` so both outputs use pre-butterfly
            // inputs (the spec computes xar[] from xr[]).
            let xr_lo = xr[lo];
            let xr_hi = xr[hi];
            xar[lo] = xr_lo * cs[i] - xr_hi * ca[i];
            xar[hi] = xr_hi * cs[i] + xr_lo * ca[i];
        }
    }

    xar
}

#[cfg(test)]
include!("alias_tests.rs");
