//! Layer III **inner-loop global-gain search** — the rate-control loop
//! of ISO/IEC 11172-3:1993 Annex C §C.1.5.4.4 (informational). This
//! wraps the §2.4.3.4.7 [`crate::quantize::quantize`] primitive in the
//! `global_gain` step-size search the spec's inner iteration loop
//! performs.
//!
//! # Scope (Phase 2 step 5)
//!
//! Given a target magnitude spectrum `xr[576]` and an already-chosen
//! scalefactor configuration (the `GranuleChannel` minus its
//! `global_gain`, plus the `ScaleFactors`), find the **smallest**
//! `global_gain` — i.e. the finest quantization, the largest output
//! magnitudes — whose quantized `is[576]` still satisfies a constraint.
//! Two forms are provided:
//!
//! * **Magnitude clamp** ([`search_magnitude_clamp`]) — §C.1.5.4.4.2
//!   ("Test of the maximum of the quantized values"): the largest
//!   allowed quantized value is bounded. The §2.4.1.7 `big_values`
//!   definition states *"The maximum absolute value in this range is
//!   constrained to 8191"*, so the constraint here is `max|is| ≤ 8191`
//!   ([`BIG_VALUES_LIMIT`]). The spec increases the quantizer step size
//!   (`qquant = qquant + 1`, i.e. raises `global_gain`) until the
//!   maximum is within range; we binary-search the threshold instead.
//!
//! * **Bit budget** ([`search_bit_budget`]) — §C.1.5.4.4 in spirit
//!   ("increases the quantizer step size until the output vector can be
//!   coded with the available number of bits"): the smallest
//!   `global_gain` whose **coarse** bit estimate fits a supplied budget.
//!   The estimate here is an order-of-magnitude placeholder
//!   ([`coarse_bit_estimate`]); the exact §C.1.5.4.4.5 / §C.1.5.4.4.8
//!   Huffman bit count (count1 + big-values codebook lengths) is a
//!   later step and is **not** computed this round.
//!
//! This file does **not** implement the psychoacoustic model, the
//! §C.1.5.4.3 outer (distortion-control) loop, scalefactor estimation,
//! or the exact Huffman count. It only searches the one scalar
//! (`global_gain`) that the inner loop varies.
//!
//! # Why binary search is valid
//!
//! The §2.4.3.4.7.1 requantization gain is
//! `2^((global_gain - 210)/4) · …`, so the per-line quantizer
//! `|is_i| = round((|xr_i| / factor)^(3/4))` is **monotone
//! non-increasing** in `global_gain`: a larger `global_gain` multiplies
//! the divisor, shrinking every `|is_i|`. Therefore both
//! `max_i |is_i|` and any non-negative weighting of the `|is_i|`
//! (the coarse bit estimate) are monotone non-increasing in
//! `global_gain`. The predicate *"constraint satisfied"* is thus a
//! step function of `global_gain` over the 8-bit field range `[0, 255]`
//! (false below a threshold, true at and above it), so the smallest
//! satisfying gain is found by binary search.
//!
//! No external implementation was consulted; every rule is taken from
//! the ISO/IEC 11172-3:1993 §C.1.5.4 / §2.4.1.7 / §2.4.3.4.7 text.

use crate::frame::MpegVersion;
use crate::quantize::quantize;
use crate::requantize::NUM_LINES;
use crate::scalefactors::ScaleFactors;
use crate::side_info::GranuleChannel;

/// The maximum absolute quantized value the big-values partition may
/// carry, from the §2.4.1.7 `big_values` definition: *"The maximum
/// absolute value in this range is constrained to 8191."*
pub const BIG_VALUES_LIMIT: i32 = 8191;

/// Inclusive lower bound of the 8-bit `global_gain` side-info field.
pub const GAIN_MIN: u8 = u8::MIN;
/// Inclusive upper bound of the 8-bit `global_gain` side-info field.
pub const GAIN_MAX: u8 = u8::MAX;

/// Outcome of an inner-loop `global_gain` search.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InnerLoopResult {
    /// The chosen `global_gain` — the smallest gain whose `is[]`
    /// satisfies the search constraint, or [`GAIN_MAX`] when no gain in
    /// `[GAIN_MIN, GAIN_MAX]` does (in which case `satisfied` is false
    /// and `is` / `max_abs` are the values at [`GAIN_MAX`], the coarsest
    /// available quantization).
    pub global_gain: u8,
    /// The quantized Huffman-input buffer at the chosen `global_gain`.
    pub is: [i32; NUM_LINES],
    /// `max_i |is[i]|` for the returned `is` (`0` for an all-zero
    /// buffer).
    pub max_abs: i32,
    /// Whether the chosen `global_gain` actually satisfies the
    /// constraint. False only when even the coarsest quantization
    /// ([`GAIN_MAX`]) cannot meet it.
    pub satisfied: bool,
}

/// `max_i |is[i]|` over the 576 lines (`0` for an all-zero buffer).
#[must_use]
pub fn max_abs(is: &[i32; NUM_LINES]) -> i32 {
    is.iter().map(|&v| v.abs()).max().unwrap_or(0)
}

/// A deliberately coarse stand-in for the §C.1.5.4.4.5 / §C.1.5.4.4.8
/// Huffman bit count: it approximates the cost of one granule-channel
/// as the sum over all non-zero lines of `bits(|is_i|) + 1` (a linear
/// PCM word length plus one sign bit), where `bits(n)` is the number of
/// bits to hold `n` (`0` for `n == 0`, `1` for `n == 1`, …). This is an
/// order-of-magnitude proxy used only to exercise the budget-form
/// search; it is **not** the exact codebook-length sum and must be
/// replaced by the real §C.1.5.4.4 count in a later step.
#[must_use]
pub fn coarse_bit_estimate(is: &[i32; NUM_LINES]) -> u64 {
    let mut bits = 0u64;
    for &v in is.iter() {
        let mag = v.unsigned_abs();
        if mag != 0 {
            // bit length of `mag` plus one sign bit.
            bits += u64::from(u32::BITS - mag.leading_zeros()) + 1;
        }
    }
    bits
}

/// Quantize `xr` at the given `global_gain` (everything else in
/// `gc_template` / `sf` held fixed) and return the resulting `is[]`.
fn quantize_at(
    xr: &[f32; NUM_LINES],
    gc_template: &GranuleChannel,
    sf: &ScaleFactors,
    sample_rate_hz: u32,
    version: MpegVersion,
    global_gain: u8,
) -> [i32; NUM_LINES] {
    let mut gc = *gc_template;
    gc.global_gain = global_gain;
    quantize(xr, &gc, sf, sample_rate_hz, version)
}

/// Generic binary search for the smallest `global_gain` in
/// `[GAIN_MIN, GAIN_MAX]` whose quantized `is[]` satisfies `predicate`.
///
/// `predicate(is)` must be monotone in `global_gain`: false for gains
/// below some threshold and true at and above it. (Both the
/// magnitude-clamp and coarse-bit-budget predicates are, because
/// `|is_i|` is monotone non-increasing in `global_gain`.) Returns the
/// threshold gain and its `is[]`; if no gain satisfies the predicate
/// the result carries [`GAIN_MAX`] with `satisfied == false`.
fn search<F>(
    xr: &[f32; NUM_LINES],
    gc_template: &GranuleChannel,
    sf: &ScaleFactors,
    sample_rate_hz: u32,
    version: MpegVersion,
    predicate: F,
) -> InnerLoopResult
where
    F: Fn(&[i32; NUM_LINES]) -> bool,
{
    let quant = |gain: u8| quantize_at(xr, gc_template, sf, sample_rate_hz, version, gain);

    // Standard lower-bound binary search over the inclusive integer
    // range [GAIN_MIN, GAIN_MAX]: find the smallest `gain` with
    // predicate(quant(gain)) == true. `lo` is a candidate, `hi` is one
    // past the searched range, kept as u16 to avoid u8 overflow.
    let mut lo: u16 = u16::from(GAIN_MIN);
    let mut hi: u16 = u16::from(GAIN_MAX) + 1;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let is_mid = quant(mid as u8);
        if predicate(&is_mid) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }

    if lo > u16::from(GAIN_MAX) {
        // No gain satisfies the predicate; report the coarsest one.
        let is = quant(GAIN_MAX);
        let m = max_abs(&is);
        InnerLoopResult {
            global_gain: GAIN_MAX,
            is,
            max_abs: m,
            satisfied: false,
        }
    } else {
        let gain = lo as u8;
        let is = quant(gain);
        let m = max_abs(&is);
        InnerLoopResult {
            global_gain: gain,
            is,
            max_abs: m,
            satisfied: true,
        }
    }
}

/// §C.1.5.4.4.2 magnitude-clamp inner loop: find the smallest
/// `global_gain` whose quantized `is[]` keeps `max|is| ≤`
/// [`BIG_VALUES_LIMIT`].
///
/// `gc_template.global_gain` is ignored — it is the field being
/// searched; every other field (`scalefac_scale`, `subblock_gain`,
/// `block_type`, `preflag`, …) and `sf` are held fixed. The returned
/// `is[]` is the finest available quantization (smallest gain) that
/// still fits the 8191 limit. When even [`GAIN_MIN`] fits (a small-
/// magnitude `xr`), [`GAIN_MIN`] is returned.
///
/// Raising `global_gain` only divides by `2^((global_gain - 210)/4)`,
/// so for a fixed scalefactor config there is a finite largest
/// amplitude the coarsest gain ([`GAIN_MAX`]) can bring under the
/// limit. A target louder than that reach cannot be clamped by
/// `global_gain` alone — the §C.1.5.4.3 outer loop would raise
/// scalefactors / `subblock_gain` instead, which is out of scope here —
/// so the result reports `satisfied == false` and carries the
/// [`GAIN_MAX`] fallback (`is[]` and `max_abs` at the coarsest gain).
#[must_use]
pub fn search_magnitude_clamp(
    xr: &[f32; NUM_LINES],
    gc_template: &GranuleChannel,
    sf: &ScaleFactors,
    sample_rate_hz: u32,
    version: MpegVersion,
) -> InnerLoopResult {
    search(xr, gc_template, sf, sample_rate_hz, version, |is| {
        max_abs(is) <= BIG_VALUES_LIMIT
    })
}

/// §C.1.5.4.4 bit-budget inner loop: find the smallest `global_gain`
/// whose [`coarse_bit_estimate`] of the quantized `is[]` is `≤ budget`.
///
/// As with [`search_magnitude_clamp`], `gc_template.global_gain` is the
/// searched field and all other fields plus `sf` are held fixed. If no
/// gain in `[GAIN_MIN, GAIN_MAX]` fits the budget (a budget smaller
/// than the all-zero cost of `0`, which is impossible, or — more
/// realistically — never, since [`GAIN_MAX`] drives `is[]` to all-zero
/// at cost `0`), the result carries `satisfied == false`. In practice a
/// budget `≥ 0` is always met by the all-zero quantization at the
/// coarsest gain, so `satisfied` is true whenever `budget` is reachable
/// before that point.
///
/// The bit estimate is a coarse placeholder, not the exact
/// §C.1.5.4.4.5 / §C.1.5.4.4.8 Huffman count — see
/// [`coarse_bit_estimate`].
#[must_use]
pub fn search_bit_budget(
    xr: &[f32; NUM_LINES],
    gc_template: &GranuleChannel,
    sf: &ScaleFactors,
    sample_rate_hz: u32,
    version: MpegVersion,
    budget: u64,
) -> InnerLoopResult {
    search(xr, gc_template, sf, sample_rate_hz, version, |is| {
        coarse_bit_estimate(is) <= budget
    })
}

#[cfg(test)]
include!("inner_loop_tests.rs");
