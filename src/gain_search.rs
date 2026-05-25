//! Layer III **inner-loop global-gain search** — the rate-control step
//! that wraps the §2.4.3.4.7 [`crate::quantize::quantize`] primitive and
//! picks a `global_gain` so the resulting integer Huffman input `is[576]`
//! satisfies a magnitude (or bit-budget) constraint.
//!
//! # Scope (Phase 2 step 5)
//!
//! This file is the **inner iteration loop's step-size search**, and
//! *only* that. From the informational Annex C §C.1.5.4.4 loop:
//!
//! > *"The inner loop quantizes the input vector and increases the
//! > quantizer step size until the output vector can be coded with the
//! > available number of bits."* (§C.1.5.4)
//!
//! and the magnitude clamp that precedes the bit count:
//!
//! > *"The maximum allowed quantized value is limited. … before any bit
//! > counting is done the quantizer stepsize is increased by `qquant =
//! > qquant+1` until the maximum of the quantized values is within the
//! > range of the largest Huffman code table."* (§C.1.5.4.4.2)
//!
//! It does **not** run the psychoacoustic model, the outer
//! (distortion-control) loop, scalefactor amplification, region
//! subdivision (`SUBDIVIDE`, §C.1.5.4.4.6), table selection
//! (§C.1.5.4.4.7), or any exact Huffman bit count (§C.1.5.4.4.5/8).
//! Those are later primitives in the encoder build-out. This step takes
//! an already-chosen scalefactor configuration and finds the smallest
//! `global_gain` (finest quantization) whose quantized output fits the
//! supplied constraint.
//!
//! # The step-size / `global_gain` relationship
//!
//! §C.1.5.4 quantizes with
//!
//! ```text
//! ix(i) = nint( (|xr(i)| / 2^((qquant + quantanf)/4))^0.75 - 0.0946 )
//! ```
//!
//! and outputs `global_gain = qquant + system-constant`. In the decoder
//! formula the matching divisor is `2^((global_gain - 210)/4)` (see
//! [`crate::requantize`]). So an integer increment of `qquant` is an
//! integer increment of `global_gain`: increasing `global_gain` enlarges
//! the divisor, which shrinks every `|ix|`. The maximum quantized
//! magnitude is therefore **monotonically non-increasing** in
//! `global_gain`, which is what makes the §C.1.5.4.4.2 linear search (and
//! the binary search below that accelerates it) well-defined.
//!
//! # Constraint forms
//!
//! Two equivalent predicates select the smallest acceptable gain:
//!
//! * **magnitude** — `max|is| <= max_quant`. The default `max_quant`
//!   is [`MAX_HUFFMAN_VALUE`] (8191), the largest magnitude the brief
//!   admits for the big-values partition. The §C.1.5.4.4.2 hardware
//!   ceiling of the largest code table is one higher
//!   ([`MAX_TABLE_VALUE`] = 8206 = `15 + (2^13 - 1)`, symbol 15 plus a
//!   13-bit `linbits` escape); 8191 is the conservative budget the
//!   build-out uses until exact per-region table selection lands.
//! * **bit budget** — a coarse `sum_i ceil(log2(|is_i| + 1))` estimate
//!   under a supplied `max_bits`. This is *not* the exact Huffman bit
//!   count (§C.1.5.4.4.5/8) — that needs region subdivision and table
//!   selection, which are out of scope. It is the placeholder estimate
//!   the brief asks for, monotone in `global_gain` for the same reason
//!   the magnitude predicate is.
//!
//! No external implementation was consulted; the loop structure and the
//! `qquant+1` step rule are from the ISO/IEC 11172-3:1993 Annex C text
//! cited above.

use crate::frame::MpegVersion;
use crate::quantize::quantize;
use crate::requantize::NUM_LINES;
use crate::scalefactors::ScaleFactors;
use crate::side_info::GranuleChannel;

/// The largest quantized magnitude the brief admits for the big-values
/// partition. The §C.1.5.4.4.2 search drives `max|is|` to or below this.
pub const MAX_HUFFMAN_VALUE: i32 = 8191;

/// The absolute ceiling of the largest Huffman code table: symbol `15`
/// plus a 13-bit `linbits` escape, `15 + (2^13 - 1) = 8206` (Table
/// 3-B.7, tables 23 and 31 carry `linbits = 13`). [`MAX_HUFFMAN_VALUE`]
/// is the conservative budget used until exact table selection lands.
pub const MAX_TABLE_VALUE: i32 = 8206;

/// `global_gain` is an 8-bit side-info field, so the search space is the
/// closed range `[0, 255]`.
const GAIN_MAX: u8 = u8::MAX;

/// The largest magnitude in a quantized granule-channel.
#[must_use]
pub fn max_abs_is(is: &[i32; NUM_LINES]) -> i32 {
    is.iter()
        .map(|&v| v.unsigned_abs() as i32)
        .max()
        .unwrap_or(0)
}

/// A very coarse bit estimate: `sum_i ceil(log2(|is_i| + 1))` over the
/// non-zero lines, plus one sign bit per non-zero line.
///
/// This is **not** the exact §C.1.5.4.4.5/8 Huffman bit count — it omits
/// region subdivision, table selection, the count1 partition split, and
/// `linbits` escapes. It is a monotone-in-`global_gain` placeholder for
/// the bit-budget search form; the exact count is a later primitive.
#[must_use]
pub fn estimate_bits(is: &[i32; NUM_LINES]) -> u64 {
    let mut bits = 0u64;
    for &v in is.iter() {
        let m = v.unsigned_abs();
        if m != 0 {
            // ceil(log2(m + 1)) magnitude bits + 1 sign bit.
            let mag_bits = u64::from(32 - (m).leading_zeros());
            bits += mag_bits + 1;
        }
    }
    bits
}

/// Outcome of a global-gain search.
#[derive(Debug, Clone)]
pub struct GainSearch {
    /// The chosen `global_gain` (smallest gain satisfying the
    /// constraint), already written into the returned [`GranuleChannel`].
    pub global_gain: u8,
    /// The quantized integer Huffman input at the chosen gain.
    pub is: [i32; NUM_LINES],
    /// The largest magnitude in [`GainSearch::is`].
    pub max_abs: i32,
    /// The coarse bit estimate (see [`estimate_bits`]) at the chosen
    /// gain.
    pub estimated_bits: u64,
    /// `false` when no gain in `[0, 255]` satisfies the constraint (the
    /// returned fields are from `global_gain == 255`, the finest the
    /// field can express, even though it still violates the limit). A
    /// caller hitting this needs scalefactor amplification, which is a
    /// later loop stage.
    pub satisfied: bool,
}

/// Quantize `xr` at `global_gain == gain` against the rest of `gc` / `sf`
/// and return the result. The caller's `gc.global_gain` is ignored; only
/// `gain` is used so the search can probe gains without mutating the
/// input.
fn quantize_at(
    xr: &[f32; NUM_LINES],
    gc: &GranuleChannel,
    sf: &ScaleFactors,
    sample_rate_hz: u32,
    version: MpegVersion,
    gain: u8,
) -> [i32; NUM_LINES] {
    let mut probe = *gc;
    probe.global_gain = gain;
    quantize(xr, &probe, sf, sample_rate_hz, version)
}

/// Find the smallest `global_gain` whose quantized output keeps
/// `max|is| <= max_quant`.
///
/// This is the §C.1.5.4.4.2 magnitude clamp. The predicate "fits" is
/// monotone in `global_gain` (larger gain → finer divisor → smaller
/// magnitudes), so the linear `qquant+1` walk of the spec is replaced by
/// a binary search over `[0, 255]` that returns the same fixpoint.
///
/// `max_quant` should not exceed [`MAX_TABLE_VALUE`]; the default budget
/// is [`MAX_HUFFMAN_VALUE`]. The returned [`GranuleChannel`]-bound gain is
/// the smallest acceptable one (finest quantization), matching the spec's
/// "increase until it fits, no further" intent.
#[must_use]
pub fn search_gain_for_max_value(
    xr: &[f32; NUM_LINES],
    gc: &GranuleChannel,
    sf: &ScaleFactors,
    sample_rate_hz: u32,
    version: MpegVersion,
    max_quant: i32,
) -> GainSearch {
    search(xr, gc, sf, sample_rate_hz, version, |is| {
        max_abs_is(is) <= max_quant
    })
}

/// Find the smallest `global_gain` whose quantized output fits under the
/// coarse `max_bits` budget (see [`estimate_bits`]).
///
/// This is the §C.1.5.4 "increases the quantizer step size until the
/// output vector can be coded with the available number of bits" loop,
/// using the placeholder bit estimate rather than the exact
/// §C.1.5.4.4.5/8 count (region subdivision + table selection are later
/// primitives). The predicate is monotone in `global_gain` for the same
/// reason the magnitude form is.
#[must_use]
pub fn search_gain_for_bit_budget(
    xr: &[f32; NUM_LINES],
    gc: &GranuleChannel,
    sf: &ScaleFactors,
    sample_rate_hz: u32,
    version: MpegVersion,
    max_bits: u64,
) -> GainSearch {
    search(xr, gc, sf, sample_rate_hz, version, |is| {
        estimate_bits(is) <= max_bits
    })
}

/// Shared binary search: find the smallest `global_gain` in `[0, 255]`
/// for which `fits` holds, exploiting the predicate's monotonicity in the
/// gain. Returns the gain-255 result with `satisfied == false` when no
/// gain fits.
fn search<F>(
    xr: &[f32; NUM_LINES],
    gc: &GranuleChannel,
    sf: &ScaleFactors,
    sample_rate_hz: u32,
    version: MpegVersion,
    fits: F,
) -> GainSearch
where
    F: Fn(&[i32; NUM_LINES]) -> bool,
{
    // `lo` is the smallest gain known to fit (initialised to "none");
    // `hi` walks down toward it. The predicate is monotone: if a gain
    // fits, every larger gain fits too. Find the boundary.
    let probe = |gain: u8| quantize_at(xr, gc, sf, sample_rate_hz, version, gain);

    // First check the coarsest acceptable gain: if even the maximum field
    // value can't satisfy the constraint, report unsatisfied at 255.
    let top = probe(GAIN_MAX);
    if !fits(&top) {
        return GainSearch {
            global_gain: GAIN_MAX,
            max_abs: max_abs_is(&top),
            estimated_bits: estimate_bits(&top),
            is: top,
            satisfied: false,
        };
    }

    // Binary search for the smallest gain that fits in [0, 255]. Invariant:
    // `lo` never fits (or is below the range) and `hi` always fits.
    let mut lo: i32 = -1; // sentinel: "below 0", never fits
    let mut hi: i32 = i32::from(GAIN_MAX); // known to fit (checked above)
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        let is = probe(mid as u8);
        if fits(&is) {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    let gain = hi as u8;
    let is = probe(gain);
    GainSearch {
        global_gain: gain,
        max_abs: max_abs_is(&is),
        estimated_bits: estimate_bits(&is),
        is,
        satisfied: true,
    }
}

#[cfg(test)]
include!("gain_search_tests.rs");
