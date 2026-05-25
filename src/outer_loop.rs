//! Layer III **outer (distortion-control) iteration loop** — the noise
//! shaping loop of ISO/IEC 11172-3:1993 Annex C §C.1.5.4.3 (informational).
//! This wraps the §C.1.5.4.4 inner loop (the
//! [`crate::inner_loop::search_bit_budget`] global-gain search) in the
//! per-scalefactor-band amplification iteration the spec's outer loop
//! performs.
//!
//! # Scope (Phase 2 step 11)
//!
//! Given a target magnitude spectrum `xr[576]` and a per-band noise
//! threshold vector `xmin[21]` (long-block only this round), iterate:
//!
//! 1. Run the inner loop ([`crate::inner_loop::search_bit_budget`]) over
//!    `xr` + current scalefactor configuration to pick the smallest
//!    `global_gain` whose §C.1.5.4.4.5 + §C.1.5.4.4.8 Huffman count fits
//!    the per-granule-channel bit budget.
//! 2. Compute the per-band actual distortion `xfsf[sb]` (§C.1.5.4.3.3):
//!    `xfsf(sb) = Σᵢ (|xr(i)| − ix(i)^(4/3) · 2^((qquant+quantanf)/4))² /
//!    bandwidth(sb)` summed over the lines of band `sb`. In our
//!    implementation the inner loop's quantization already applies
//!    `gc.global_gain` + per-band scalefactor + `scalefac_scale`, so the
//!    decoder-side requantizer ([`crate::requantize::requantize`]) gives
//!    the same line values the spec writes as
//!    `ix(i)^(4/3) · 2^((qquant+quantanf)/4) · 2^(-mult·scalefac(sb))`.
//! 3. Mark every band with `xfsf(sb) > xmin(sb)`.
//! 4. Termination (§C.1.5.4.3.6): stop if (a) no band exceeds its
//!    threshold, (b) every band has already been amplified at least once
//!    this iteration cycle, or (c) any band that *would* be amplified
//!    would exceed its scalefactor upper limit (15 for `sfb ∈ [0,10]`,
//!    7 for `sfb ∈ [11,20]` — the §C.1.5.4.3.6 transmission-format cap
//!    derived from MPEG-1 long-block Table 3-B.5 / Table B.3 `slen2`
//!    widths). Save the *previous* (last-good) scalefactor state and
//!    return it.
//! 5. Otherwise, for each band over threshold, `scalefac_l[sb] += 1` and
//!    re-enter step 1. The increment of `scalefac_l[sb]` shrinks the
//!    per-band requantization divisor `2^(-mult·scalefac(sb))`, which in
//!    the encoder's [`crate::quantize::quantize`] call grows the
//!    pre-scalefactor quotient `|xr(i)| / factor` by a factor of
//!    `2^(mult)` (i.e. √2 with `scalefac_scale = 0`), making the line
//!    more finely quantized after the inner loop re-picks `global_gain`.
//!
//! # Uniform threshold this round
//!
//! A true psychoacoustic model (Annex D) computes a per-band masking
//! threshold that varies across the spectrum. **This round uses a
//! uniform constant `xmin[sb] = uniform_threshold` for every band** —
//! the spec explicitly says (§C.1.5.4.1) that the loops module takes
//! `xmin(sb)` as input and does not specify how it is derived, so a
//! constant is a valid (if uninformative) choice. Once the psy model
//! lands as its own Phase-2 step, this module's signature is unchanged
//! — only the caller's threshold vector becomes per-band.
//!
//! # Scope limits
//!
//! * **Long blocks only.** Short / mixed blocks have a per-window
//!   scalefactor table (`scalefac_s[sfb][win]`) and a different upper
//!   limit; deferred until the encoder's block-type-switching step lands.
//! * **No preemphasis (§C.1.5.4.3.4).** The pretab amplification is an
//!   optional precondition step the spec ties to "all of the upper 4
//!   scalefactor bands" exceeding threshold after the first inner pass;
//!   we leave `preflag = false` so the band-by-band amplification is the
//!   only noise-shaping lever.
//! * **No `scalefac_scale = 1` escalation.** The spec allows raising
//!   `scalefac_scale` to 1 when the scalefactor range overflows; this
//!   round terminates instead. With `scalefac_compress = 15` and
//!   `scalefac_scale = 0` (our choice) the dynamic range is the spec's
//!   smallest, which is conservative — the outer loop saturates earlier
//!   than a real encoder would.
//!
//! No external implementation was consulted; every rule is taken from
//! the §C.1.5.4.3 / §C.1.5.4.3.x text and Figure C.9.b.

use crate::frame::MpegVersion;
use crate::inner_loop::{search_bit_budget, search_magnitude_clamp, GAIN_MAX};
use crate::quantize::quantize;
use crate::requantize::{long_band_starts, requantize, NUM_LINES};
use crate::scalefactors::{ScaleFactors, LONG_SFB};
use crate::side_info::{BlockType, GranuleChannel};

/// Upper bound on the per-band scalefactor stored in `scalefac_l[sfb]`
/// for the low band group (`sfb ∈ [0, 10]`): the §C.1.5.4.3.6
/// transmission-format cap. With our `scalefac_compress = 15` the `slen1`
/// width is 4 bits → cap 15.
pub const SCALEFAC_MAX_LOW: u8 = 15;

/// Upper bound on the per-band scalefactor stored in `scalefac_l[sfb]`
/// for the high band group (`sfb ∈ [11, 20]`): the §C.1.5.4.3.6
/// transmission-format cap. With our `scalefac_compress = 15` the `slen2`
/// width is 3 bits → cap 7.
pub const SCALEFAC_MAX_HIGH: u8 = 7;

/// The `scalefac_compress` we choose for the outer loop's long-block
/// path: index 15 ⇒ `(slen1=4, slen2=3)` per Table 3-B.5 (MPEG-1
/// §2.4.2.7), which spans the full §C.1.5.4.3.6 per-band scalefactor
/// range without ever overflowing the field width. With this choice the
/// part2 (scalefactor) sub-block costs a fixed `11·4 + 10·3 = 74` bits per
/// granule-channel, which fits comfortably inside the per-granule-channel
/// CBR slot at every MPEG-1 bitrate the encoder supports.
pub const OUTER_LOOP_SCALEFAC_COMPRESS: u16 = 15;

/// The per-band scalefactor upper limit for long-block `sfb` per
/// §C.1.5.4.3.6.
#[must_use]
pub fn scalefac_long_upper_limit(sfb: usize) -> u8 {
    if sfb <= 10 {
        SCALEFAC_MAX_LOW
    } else {
        SCALEFAC_MAX_HIGH
    }
}

/// One outer-loop iteration's accounting. Returned so the stream
/// encoder can report iteration counts in debug builds and so the unit
/// tests can probe convergence behaviour without re-running the loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OuterLoopStats {
    /// Total number of outer-loop iterations completed (≥ 1; the first
    /// inner-loop call alone counts as one iteration).
    pub iterations: u32,
    /// Number of bands that were amplified at least once across all
    /// iterations.
    pub bands_amplified: u32,
    /// `true` if the loop terminated because no band exceeded its
    /// threshold (§C.1.5.4.3.6 path "no band over threshold"). `false`
    /// otherwise (cap reached, every band already amplified).
    pub converged: bool,
}

/// Outcome of an outer-loop search for one granule-channel.
#[derive(Debug, Clone, PartialEq)]
pub struct OuterLoopResult {
    /// The chosen per-band scalefactors (long-block only). Bands above
    /// 21 are left zero; short-block fields are zero too.
    pub scalefactors: ScaleFactors,
    /// The chosen `global_gain` the last inner-loop pass settled on for
    /// the returned `scalefactors`.
    pub global_gain: u8,
    /// The chosen `is[576]` quantized buffer matching the returned
    /// `scalefactors` + `global_gain`.
    pub is: [i32; NUM_LINES],
    /// Iteration accounting.
    pub stats: OuterLoopStats,
}

/// Compute the per-band actual distortion `xfsf[sb]` (§C.1.5.4.3.3) for
/// the long-block scalefactor bands, **in the colored domain** the spec
/// defines.
///
/// The spec preamble §C.1.5.4.3 multiplies `xr(i)` by the per-band
/// colouring factor `sqrt(2)^((1+scalefac_scale)·scalefac(sb))`; after
/// that the §C.1.5.4.3.3 distortion is
///
/// ```text
/// xfsf(sb) = Σᵢ ( |xr_amplified(i)|
///                 − ix(i)^(4/3) · 2^((qquant+quantanf)/4) )² / bandwidth(sb)
/// ```
///
/// where the reconstruction `ix(i)^(4/3) · 2^((qquant+quantanf)/4)` is
/// the requantization without the per-band scalefactor unfolding (because
/// the scalefactor is already baked into `xr_amplified`).
///
/// In our implementation we keep `xr` un-amplified across the outer
/// loop's iterations and apply the scalefactor *inside* the quantizer
/// (`gc.scalefac_scale` + `sf.long[sfb]` are the divisor the
/// [`crate::quantize::quantize`] uses). To reproduce the spec's metric we
/// therefore scale our original-domain residual UP by the per-band
/// colouring factor `2^(mult·scalefac(sb))` (the squared equivalent of
/// `sqrt(2)^((1+scalefac_scale)·scalefac(sb))`) — which makes the
/// comparison invariant to the choice of where the scalefactor lives in
/// the math.
///
/// Equivalently: with `our_residual(i) = |xr_orig(i)| − |xr_back(i)|`
/// (both in the original domain), the spec's per-band sum is
/// `Σ our_residual(i)² · 2^(2·mult·scalefac(sb)) / bandwidth(sb)`.
///
/// `xr` is the ORIGINAL spectrum the encoder fed to quantize (un-amplified
/// across outer-loop iterations). `xr_back` is
/// [`crate::requantize::requantize`]'s output — the original-domain
/// reconstruction (per-band scalefactor already unfolded back out).
/// `sf` is the current per-band scalefactor state the outer loop is
/// testing. `scalefac_scale` is the `gc.scalefac_scale` flag (`false`
/// in this round so `mult = 0.5`).
#[must_use]
pub fn band_distortion_long(
    xr: &[f32; NUM_LINES],
    xr_back: &[f32; NUM_LINES],
    sf: &ScaleFactors,
    scalefac_scale: bool,
    sample_rate_hz: u32,
    version: MpegVersion,
) -> [f64; LONG_SFB] {
    use crate::requantize::scalefac_multiplier;
    let starts = long_band_starts(sample_rate_hz, version);
    let mult = f64::from(scalefac_multiplier(scalefac_scale));
    let mut out = [0.0f64; LONG_SFB];
    for (sfb, slot) in out.iter_mut().enumerate() {
        let lo = starts[sfb];
        let hi = starts[sfb + 1].min(NUM_LINES);
        if hi <= lo {
            continue;
        }
        let mut sse = 0.0f64;
        for i in lo..hi {
            let d = f64::from(xr[i].abs()) - f64::from(xr_back[i].abs());
            sse += d * d;
        }
        let bw = (hi - lo) as f64;
        // Re-scale our original-domain SSE to the colored-domain SSE
        // the spec's metric uses. Per-band amplification by
        // sqrt(2)^((1+scalefac_scale)·scalefac(sb)) means the colored
        // residual is the original-domain residual times the same factor;
        // squaring gives the SSE multiplier 2^(2·mult·scalefac(sb))·...
        // where `mult` already encodes the (1+scalefac_scale)/2 split.
        let sf_val = f64::from(sf.long[sfb.min(LONG_SFB - 1)]);
        let scale = (2.0 * mult * sf_val).exp2();
        *slot = (sse / bw) * scale;
    }
    out
}

/// Run the §C.1.5.4.3 outer (distortion-control) iteration loop for one
/// long-block granule-channel.
///
/// `xr` is the target spectrum (post-alias-reduction, encoder
/// pre-quantization). `gc_template` carries the long-block configuration
/// (`block_type = Long`, `window_switching_flag = false`, the chosen
/// region split, the table_select skeleton); its `scalefac_compress`
/// **MUST** be [`OUTER_LOOP_SCALEFAC_COMPRESS`] so the encoder can later
/// write the chosen scalefactors back as part2 with non-zero `slen`. Its
/// `global_gain` is irrelevant — the inner loop re-picks it on every
/// iteration. `per_gc_bit_budget` is the §C.1.5.3.4 mean-bits budget the
/// inner loop must fit. `uniform_threshold` is the `xmin[sb]` constant
/// applied to every band (psychoacoustic model deferred).
///
/// `max_iter` caps the outer loop at a finite count — the spec's
/// §C.1.5.4.3.6 termination conditions are guaranteed to fire eventually
/// (every iteration either amplifies a band or terminates, and the
/// per-band cap is 15 / 7), so a soft cap is a defensive guard against
/// FP precision artifacts.
///
/// Returns the converged scalefactor state, the inner loop's last
/// `global_gain`, and the matching `is[576]`.
#[must_use]
pub fn outer_loop_search_long(
    xr: &[f32; NUM_LINES],
    gc_template: &GranuleChannel,
    sample_rate_hz: u32,
    version: MpegVersion,
    per_gc_bit_budget: u64,
    uniform_threshold: f64,
    max_iter: u32,
) -> OuterLoopResult {
    debug_assert!(!gc_template.window_switching_flag);
    debug_assert_eq!(gc_template.block_type, BlockType::Long);

    // §C.1.5.4.2.1: init scalefactors to zero, preflag off, scalefac_scale 0.
    let mut sf = ScaleFactors::default();
    let mut amplified = [false; LONG_SFB];

    // Saved last-good state. The spec saves scalefactors BEFORE each
    // amplification, so that if the next iteration trips a termination
    // condition the encoder transmits the previous (in-range) state.
    let mut last_good_sf = sf;
    let mut last_good_inner = run_inner(
        xr,
        gc_template,
        &sf,
        sample_rate_hz,
        version,
        per_gc_bit_budget,
    );

    let mut iterations: u32 = 1; // the first inner-loop call counts
    let converged;
    let mut bands_amplified_total: u32 = 0;

    loop {
        // Run the inner loop with the *current* sf state.
        let inner = run_inner(
            xr,
            gc_template,
            &sf,
            sample_rate_hz,
            version,
            per_gc_bit_budget,
        );

        // Decode-side reconstruction to compute per-band distortion.
        let mut gc_full = *gc_template;
        gc_full.global_gain = inner.global_gain;
        gc_full.scalefac_compress = OUTER_LOOP_SCALEFAC_COMPRESS;
        gc_full.preflag = false;
        gc_full.scalefac_scale = false;
        let xr_back = requantize(&inner.is, &gc_full, &sf, sample_rate_hz, version);
        let xfsf = band_distortion_long(
            xr,
            &xr_back,
            &sf,
            gc_full.scalefac_scale,
            sample_rate_hz,
            version,
        );

        // Identify bands over threshold.
        let mut any_over = false;
        let mut would_exceed_cap = false;
        for (sfb, &d) in xfsf.iter().enumerate() {
            if d > uniform_threshold {
                any_over = true;
                let next = u16::from(sf.long[sfb]) + 1;
                if next > u16::from(scalefac_long_upper_limit(sfb)) {
                    would_exceed_cap = true;
                }
            }
        }

        // §C.1.5.4.3.6 termination paths.
        if !any_over {
            // No band over threshold — converged on a clean result.
            // The CURRENT sf + inner is the keep-state.
            converged = true;
            last_good_sf = sf;
            last_good_inner = inner;
            break;
        }
        if would_exceed_cap || amplified.iter().all(|&a| a) || iterations >= max_iter {
            // Any of the three §C.1.5.4.3.6 termination conditions:
            //   (a) all bands already amplified
            //   (b) amplification of any band would exceed upper limit
            //   (c) defensive iteration cap
            // The spec instructs to restore the saved (last-good) state,
            // which is what last_good_sf / last_good_inner already hold.
            converged = false;
            break;
        }

        // Save current state as the new last-good before amplifying.
        last_good_sf = sf;
        last_good_inner = inner;

        // §C.1.5.4.3.5: amplify the offending bands by one step each.
        for (sfb, &d) in xfsf.iter().enumerate() {
            if d > uniform_threshold {
                let cap = scalefac_long_upper_limit(sfb);
                if sf.long[sfb] < cap {
                    sf.long[sfb] = sf.long[sfb].saturating_add(1);
                    if !amplified[sfb] {
                        amplified[sfb] = true;
                        bands_amplified_total += 1;
                    }
                }
            }
        }
        iterations += 1;
    }

    OuterLoopResult {
        scalefactors: last_good_sf,
        global_gain: last_good_inner.global_gain,
        is: last_good_inner.is,
        stats: OuterLoopStats {
            iterations,
            bands_amplified: bands_amplified_total,
            converged,
        },
    }
}

/// Per-iteration helper: build a fully-populated `GranuleChannel`
/// (template + outer-loop's chosen `scalefac_compress`) and pick the
/// smallest `global_gain` that BOTH (a) fits the inner-loop bit budget
/// AND (b) keeps `max|is| ≤ 8191` (the §2.4.1.7 big-values cap that
/// [`search_magnitude_clamp`] enforces). Without (b) the encoder's
/// downstream `clamp_above` would silently truncate magnitudes, leaving
/// a decoder reconstruction at the clamp value rather than at the
/// algebraically-correct line.
fn run_inner(
    xr: &[f32; NUM_LINES],
    gc_template: &GranuleChannel,
    sf: &ScaleFactors,
    sample_rate_hz: u32,
    version: MpegVersion,
    per_gc_bit_budget: u64,
) -> InnerInvocation {
    let mut gc = *gc_template;
    gc.scalefac_compress = OUTER_LOOP_SCALEFAC_COMPRESS;
    gc.preflag = false;
    gc.scalefac_scale = false;
    let res_budget = search_bit_budget(xr, &gc, sf, sample_rate_hz, version, per_gc_bit_budget);
    let res_clamp = search_magnitude_clamp(xr, &gc, sf, sample_rate_hz, version);
    let gg = res_budget.global_gain.max(res_clamp.global_gain);
    // Re-quantize at the combined-constraint gain so the returned `is[]`
    // matches the actual gg the outer loop will surface.
    let mut gc_final = gc;
    gc_final.global_gain = gg;
    let is = quantize(xr, &gc_final, sf, sample_rate_hz, version);
    let _ = (GAIN_MAX, res_budget.satisfied); // re-export / future-use keep-alive
    InnerInvocation {
        global_gain: gg,
        is,
    }
}

/// Internal: a single inner-loop result the outer loop carries.
#[derive(Debug, Clone)]
struct InnerInvocation {
    global_gain: u8,
    is: [i32; NUM_LINES],
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::side_info::GranuleChannel;

    fn long_template() -> GranuleChannel {
        GranuleChannel {
            part2_3_length: 0,
            big_values: 0,
            global_gain: 0,
            scalefac_compress: OUTER_LOOP_SCALEFAC_COMPRESS,
            window_switching_flag: false,
            block_type: BlockType::Long,
            mixed_block_flag: false,
            table_select: [0; 3],
            subblock_gain: [0; 3],
            region0_count: 20,
            region1_count: 0,
            preflag: false,
            scalefac_scale: false,
            count1table_select: false,
        }
    }

    #[test]
    fn upper_limits_match_spec() {
        // Low group: sfb 0..=10 → 15.
        for sfb in 0..=10 {
            assert_eq!(scalefac_long_upper_limit(sfb), SCALEFAC_MAX_LOW);
        }
        // High group: sfb 11..=20 → 7.
        for sfb in 11..=20 {
            assert_eq!(scalefac_long_upper_limit(sfb), SCALEFAC_MAX_HIGH);
        }
    }

    #[test]
    fn band_distortion_zero_when_perfect() {
        // Identity reconstruction → zero distortion in every band.
        let mut xr = [0.0f32; NUM_LINES];
        for (i, v) in xr.iter_mut().enumerate() {
            *v = ((i as f32) * 0.013).sin() * 100.0;
        }
        let xr_back = xr;
        let sf = ScaleFactors::default();
        let d = band_distortion_long(&xr, &xr_back, &sf, false, 44_100, MpegVersion::Mpeg1);
        for v in d.iter() {
            assert!(*v < 1e-12, "expected zero distortion, got {v}");
        }
    }

    #[test]
    fn outer_loop_terminates_with_huge_threshold() {
        // With a threshold so large nothing ever exceeds it the loop
        // converges on the first iteration (no amplification).
        let mut xr = [0.0f32; NUM_LINES];
        for (i, v) in xr.iter_mut().enumerate() {
            *v = ((i as f32) * 0.013).sin() * 100.0;
        }
        let gc = long_template();
        let res = outer_loop_search_long(&xr, &gc, 44_100, MpegVersion::Mpeg1, 1500, 1.0e30, 64);
        assert!(res.stats.converged);
        assert_eq!(res.stats.iterations, 1);
        assert_eq!(res.stats.bands_amplified, 0);
    }

    #[test]
    fn outer_loop_terminates_with_tiny_threshold() {
        // With a threshold of zero every band exceeds it; the loop runs
        // until either the per-band cap is hit or all bands are amplified.
        let mut xr = [0.0f32; NUM_LINES];
        for (i, v) in xr.iter_mut().enumerate() {
            *v = ((i as f32) * 0.013).sin() * 100.0;
        }
        let gc = long_template();
        let res = outer_loop_search_long(&xr, &gc, 44_100, MpegVersion::Mpeg1, 1500, 0.0, 64);
        // Must terminate (cap-or-amplified branch); convergence flag
        // false in this degenerate threshold scenario. The returned
        // scalefactors are the LAST-GOOD state (§C.1.5.4.3.1 "If the
        // computation … is cancelled without having reached a proper
        // result, this value together with the quantized spectrum give
        // an approximation and can be transmitted"), so they may be the
        // pre-amplification zeros if termination fired on the first
        // cycle's "all bands amplified" check.
        assert!(!res.stats.converged);
        assert!(res.stats.bands_amplified > 0);
        assert!(res.stats.iterations >= 2);
    }

    #[test]
    fn outer_loop_amplifies_only_offending_bands() {
        // The §C.1.5.4.3.5 amplification step only touches bands whose
        // distortion exceeds `xmin(sb)`. With a threshold tuned just
        // BELOW one specific band's baseline distortion (and above all
        // others), only that one band should amplify, and the loop
        // should converge in a few iterations.
        let mut xr = [0.0f32; NUM_LINES];
        // Single high-energy tone in the second long band (sfb = 1,
        // lines [4, 8) at 44.1 kHz).
        xr[5] = 80.0;
        let gc = long_template();
        let budget = 2000u64;

        // Baseline distortion to calibrate threshold.
        let baseline_sf = ScaleFactors::default();
        let baseline = run_inner(&xr, &gc, &baseline_sf, 44_100, MpegVersion::Mpeg1, budget);
        let mut gc_b = gc;
        gc_b.global_gain = baseline.global_gain;
        let xr_back_b = requantize(
            &baseline.is,
            &gc_b,
            &baseline_sf,
            44_100,
            MpegVersion::Mpeg1,
        );
        let d_b = band_distortion_long(
            &xr,
            &xr_back_b,
            &baseline_sf,
            false,
            44_100,
            MpegVersion::Mpeg1,
        );

        // Threshold just below the loudest band's baseline distortion.
        let max_band = d_b.iter().cloned().fold(0.0f64, f64::max);
        let thr = max_band * 0.5;
        let res = outer_loop_search_long(&xr, &gc, 44_100, MpegVersion::Mpeg1, budget, thr, 32);
        // Some band amplified; loop terminated in a small number of
        // iterations.
        assert!(res.stats.bands_amplified >= 1, "no amplification happened");
        assert!(res.stats.iterations <= 32);
        // Bands that had ZERO baseline distortion (silent bands) MUST
        // still carry scalefactor 0 in the returned state — the loop
        // never amplifies bands under their threshold.
        for (sfb, &d) in d_b.iter().enumerate() {
            if d == 0.0 && res.scalefactors.long[sfb] != 0 {
                panic!(
                    "silent band sfb={sfb} got amplified to {} (should stay 0)",
                    res.scalefactors.long[sfb]
                );
            }
        }
    }
}
