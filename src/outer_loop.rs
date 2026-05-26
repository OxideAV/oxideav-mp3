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
//! * **Preemphasis (§C.1.5.4.3.4)** (round 148): after the first inner
//!   loop call the loop checks the spec's suggested condition — "if in
//!   all of the upper 4 scalefactor bands the actual distortion exceeds
//!   the threshold". When that holds, `preflag` is set to `1`, which
//!   adds the Table B.6 `pretab[]` values
//!   `[0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,2,2,3,3,3,2]` to the effective
//!   per-band scalefactor for the rest of the loop. The pretab boost
//!   is `free` (one transmitted bit; no `part2_3_length` impact), and
//!   it does NOT raise the §C.1.5.4.3.6 cap (15/7) on the transmitted
//!   `scalefac_l[sfb]` — the cap math reads `sf.long[sfb]` only and the
//!   amplifier still tops out there. Once decided, `preflag` stays on
//!   for the remainder of the loop and is reflected in the returned
//!   [`OuterLoopResult::scalefactors`]'s `preflag` field for the
//!   side-info writer.
//! * **`scalefac_scale = 1` escalation** (round 147): when an
//!   amplification step would push a band's scalefactor past its
//!   §C.1.5.4.3.6 cap (15 for `sfb ∈ [0, 10]`, 7 for `sfb ∈ [11, 20]`)
//!   AND the loop is still in `scalefac_scale = 0` mode, the loop
//!   escalates to `scalefac_scale = 1` per §C.1.5.4.3 ("If after some
//!   iterations the maximum length of the scalefactors would be
//!   exceeded … then scalefac-scale is increased to the value 1 thus
//!   increasing the possible dynamic range of the scalefactors. In this
//!   case the actual scalefactors and frequency lines have to be
//!   corrected accordingly"). The escalation halves every in-progress
//!   per-band scalefactor (round-to-nearest) so that the per-band
//!   colouring factor `2^(mult·scalefac(sb))` is preserved across the
//!   scale switch (mult doubles from 0.5 to 1.0; halving sf keeps the
//!   product unchanged), then resets the per-band `amplified[]` flags
//!   and resumes the loop. Each subsequent §C.1.5.4.3.5 amplification
//!   step is then worth 2× as much energy boost. Only one escalation
//!   ever fires (the spec sets only two distinct `scalefac_scale`
//!   values).
//!
//! No external implementation was consulted; every rule is taken from
//! the §C.1.5.4.3 / §C.1.5.4.3.x text and Figure C.9.b.

use crate::frame::MpegVersion;
use crate::inner_loop::{search_bit_budget, search_magnitude_clamp, GAIN_MAX};
use crate::quantize::quantize;
use crate::requantize::{long_band_starts, requantize, short_band_starts, NUM_LINES};
use crate::scalefactors::{ScaleFactors, LONG_SFB, SHORT_SFB, SHORT_WINDOWS};
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
    /// `scalefac_scale` flag (§2.4.2.7): `false` ⇒ multiplier 0.5
    /// (√2 per scalefactor step); `true` ⇒ multiplier 1.0 (2× per step,
    /// twice the dynamic range). The §C.1.5.4.3 outer loop sets this to
    /// `true` when the scalefactor-cap would otherwise terminate the
    /// loop and additional dynamic range is available. The caller MUST
    /// propagate this into the granule-channel's `scalefac_scale` bit
    /// before re-quantizing or writing the side-info.
    pub scalefac_scale: bool,
    /// `preflag` flag (§2.4.2.7 + §C.1.5.4.3.4): `true` when the loop
    /// chose to switch on the Table B.6 pretab high-frequency
    /// amplification. The same value lives at
    /// [`OuterLoopResult::scalefactors`]'s `preflag` field; both must
    /// be propagated by the caller — `gc.preflag` for the side-info
    /// write, and `sf.preflag` (already in `scalefactors`) for any
    /// re-quantize / re-requantize step downstream.
    pub preflag: bool,
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
    use crate::requantize::{scalefac_multiplier, PRETAB};
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
        // When `sf.preflag` is set (§C.1.5.4.3.4 preemphasis), the
        // effective scalefactor the decoder reconstructs against is
        // `scalefac_l[sfb] + pretab[sfb]` (Table B.6) — both
        // [`crate::requantize::requantize`] and
        // [`crate::quantize::quantize`] add the pretab term — so the
        // colouring factor here MUST add it as well, otherwise the
        // outer loop's distortion metric would compare a reconstruction
        // boosted by pretab against an original NOT boosted by pretab.
        let sf_idx = sfb.min(LONG_SFB - 1);
        let pre = if sf.preflag {
            f64::from(PRETAB[sf_idx])
        } else {
            0.0
        };
        let sf_val = f64::from(sf.long[sf_idx]) + pre;
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
    // §C.1.5.4.3 escalation state. The spec mandates starting at
    // scalefac_scale = 0 and escalating to 1 at most once when the cap
    // would terminate the loop.
    let mut scalefac_scale = false;
    let mut escalated_once = false;
    // §C.1.5.4.3.4 preemphasis state. The spec's suggested heuristic
    // (the one explicit hint the spec provides) is to switch on preflag
    // when, after the first inner-loop call, the upper-4 long
    // scalefactor bands (sfb 17..=20) all exceed their threshold. The
    // decision is taken once; once set, preflag stays on for the rest
    // of the loop (one transmitted bit; cheap).
    let mut preflag_decided = false;

    // Saved last-good state. The spec saves scalefactors BEFORE each
    // amplification, so that if the next iteration trips a termination
    // condition the encoder transmits the previous (in-range) state.
    let mut last_good_sf = sf;
    let mut last_good_scale = scalefac_scale;
    let mut last_good_inner = run_inner(
        xr,
        gc_template,
        &sf,
        scalefac_scale,
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
            scalefac_scale,
            sample_rate_hz,
            version,
            per_gc_bit_budget,
        );

        // Decode-side reconstruction to compute per-band distortion.
        // `gc_full.preflag` mirrors `sf.preflag` so the requantize step
        // here matches what the side-info write will instruct the
        // decoder to do; without this, the §C.1.5.4.3.3 distortion
        // would be computed against the wrong reconstruction once
        // preemphasis is on.
        let mut gc_full = *gc_template;
        gc_full.global_gain = inner.global_gain;
        gc_full.scalefac_compress = OUTER_LOOP_SCALEFAC_COMPRESS;
        gc_full.preflag = sf.preflag;
        gc_full.scalefac_scale = scalefac_scale;
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

        // §C.1.5.4.3.4 preemphasis decision (taken after the first
        // inner-loop call, exactly as the spec phrases it: "the
        // condition to switch on the preemphasis is up to the
        // implementation. For example preemphasis could be switched on
        // if in all of the upper 4 scalefactor bands the actual
        // distortion exceeds the threshold after the first call of the
        // inner loop"). The spec's worked example IS the implementation
        // we adopt here — it is the only explicit hint the spec offers,
        // and it costs at most one re-run of the iteration body. Once
        // preflag flips on:
        //   * `sf.preflag = true` so the next [`run_inner`] / `quantize`
        //     reads the inflated effective scalefactor (and the
        //     `xfsf` computed next iteration matches it);
        //   * `last_good_*` is NOT updated here — preflag is a
        //     decoder-side reconstruction setting that the saved state
        //     should also reflect, which is automatic because we just
        //     `continue` so the next loop iteration's "save last-good
        //     BEFORE amplifying" path covers the new state;
        //   * the `iterations` counter ticks up (the re-evaluation
        //     after switching on preflag is genuinely a fresh
        //     inner-loop call against a different sf).
        if !preflag_decided {
            preflag_decided = true;
            let upper_four_all_over = xfsf[17..=20].iter().all(|&d| d > uniform_threshold);
            if upper_four_all_over {
                sf.preflag = true;
                iterations += 1;
                continue;
            }
        }

        // §C.1.5.4.3.6 termination paths.
        if !any_over {
            // No band over threshold — converged on a clean result.
            // The CURRENT sf + inner is the keep-state.
            converged = true;
            last_good_sf = sf;
            last_good_scale = scalefac_scale;
            last_good_inner = inner;
            break;
        }
        if would_exceed_cap && !escalated_once {
            // §C.1.5.4.3 escalation: switch to scalefac_scale = 1 so the
            // dynamic range doubles. The coloring factor is
            //   2^(mult·sf)
            // with mult = 0.5 (scale = 0) → 1.0 (scale = 1). To preserve
            // the *current* coloured spectrum across the switch ("the
            // actual scalefactors and frequency lines have to be
            // corrected accordingly"), halve every in-progress per-band
            // scalefactor (round-to-nearest: (x + 1) / 2 with integer
            // arithmetic). After this:
            //   * the next §C.1.5.4.3.5 amplification step is worth 2×
            //     as much energy boost,
            //   * the cap headroom is restored (a halved 15 ⇒ 8 leaves
            //     room for 7 more amp steps, each twice as strong).
            // The "amplified" tracker is reset because the new step is
            // a different quantity and the loop's "every band amplified"
            // termination should re-arm against the larger steps.
            // last_good_* are NOT updated here: they already represent
            // the best in-range state under the previous scale and
            // remain a valid fallback if the rest of the loop trips.
            scalefac_scale = true;
            escalated_once = true;
            for v in sf.long.iter_mut() {
                // Round-to-nearest halving: (x + 1) / 2 expressed via
                // `div_ceil(x, 2)` for clippy's manual_div_ceil lint.
                *v = (*v).div_ceil(2);
            }
            amplified = [false; LONG_SFB];
            iterations += 1;
            continue;
        }
        if would_exceed_cap || amplified.iter().all(|&a| a) || iterations >= max_iter {
            // Any of the three §C.1.5.4.3.6 termination conditions:
            //   (a) all bands already amplified
            //   (b) amplification of any band would exceed upper limit
            //       (after escalation has already fired, or in any
            //       short-block / mixed path that lacks escalation)
            //   (c) defensive iteration cap
            // The spec instructs to restore the saved (last-good) state,
            // which is what last_good_sf / last_good_inner already hold.
            converged = false;
            break;
        }

        // Save current state as the new last-good before amplifying.
        last_good_sf = sf;
        last_good_scale = scalefac_scale;
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
        // `last_good_sf.preflag` already carries the resolved §C.1.5.4.3.4
        // decision because `sf.preflag` is set in-place when the
        // upper-4-bands condition fires (before any subsequent
        // last-good save), and every later `last_good_sf = sf`
        // assignment copies the same flag along with the band values.
        preflag: last_good_sf.preflag,
        scalefactors: last_good_sf,
        global_gain: last_good_inner.global_gain,
        is: last_good_inner.is,
        scalefac_scale: last_good_scale,
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
    scalefac_scale: bool,
    sample_rate_hz: u32,
    version: MpegVersion,
    per_gc_bit_budget: u64,
) -> InnerInvocation {
    let mut gc = *gc_template;
    gc.scalefac_compress = OUTER_LOOP_SCALEFAC_COMPRESS;
    gc.preflag = false;
    gc.scalefac_scale = scalefac_scale;
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

// =========================================================================
// Short-block outer loop (Phase 2 step 27)
// =========================================================================
//
// §C.1.5.4.3 with the §2.4.2.7 short-block reading: a `block_type == 2`,
// `mixed_block_flag == 0` granule carries 12 scalefactor bands × 3 windows
// (36 cells). Each cell has its own `scalefac_s[sfb][window]` and each of
// the 3 windows shares one `subblock_gain[window]` (3-bit field, range
// 0..=7) that subtracts an extra `8·subblock_gain[w]` from the long-block
// `(global_gain - 210)/4` exponent (§2.4.3.4.7.1). The Annex C outer loop
// is silent on *how* to search `subblock_gain`; we adopt the bounded
// scheme below.
//
// # Per-(sfb, window) amplification
//
// Mirroring the long-block loop: each iteration computes per-cell
// distortion `xfsf_s[sfb][window]` (the spec's §C.1.5.4.3.3 sum, applied
// over the cell's freqline range), marks every cell over
// `uniform_threshold`, and amplifies `scalefac_s[sfb][window] += 1` for
// the marked cells. Termination is the same three §C.1.5.4.3.6 conditions
// — no cell over threshold, every cell already amplified, or any cell
// would exceed the §C.1.5.4.3.6 cap (15 for the slen1-range sfb 0..=5,
// 7 for the slen2-range sfb 6..=11 — Table 3-B.5 with
// `scalefac_compress = 15` giving `(slen1, slen2) = (4, 3)`).
//
// # `subblock_gain` search
//
// The §C.1.5.4.4.2 magnitude-clamp inner loop (`search_magnitude_clamp`)
// already finds the smallest `global_gain` whose quantization keeps every
// line within the 8191 big-values cap. For short blocks that global cap is
// often tighter on one window than the others — a transient confined to
// one short subframe forces the global gain coarser than the quieter
// windows need. The §2.4.3.4.7.1 `subblock_gain[w]` field exists exactly
// to relieve this: raising `subblock_gain[w]` by one divides window `w`'s
// reconstruction by `2^(8/4) = 4`, so the corresponding pre-quantization
// magnitudes shrink by the same factor relative to the global gain.
//
// We escalate `subblock_gain[w]` only when the per-window max magnitude
// can't fit under 8191 with the chosen `global_gain` — i.e. when the
// `search_magnitude_clamp` result reports `satisfied == false`. In that
// case we identify which window(s) sit hardest against the cap and bump
// their `subblock_gain[w]` by 1 (saturating at the §2.4.2.7 cap of 7).
// This keeps the search bounded (`subblock_gain[w]` only ever rises, at
// most 7 times per window) and never wastes bits on quiet windows.
//
// # `scalefac_scale` escalation
//
// Same §C.1.5.4.3 path as the long-block loop: when an amplification step
// would push any cell past its §C.1.5.4.3.6 cap AND we're still in
// `scalefac_scale = 0`, halve every in-progress per-cell scalefactor
// (round-to-nearest) and switch to `scalefac_scale = 1`. The cap doubles
// in dynamic-range value; one escalation event only.
//
// # `preflag`
//
// §2.4.2.7 says "preflag is never used if block_type == 2 (short blocks)",
// so the long-block §C.1.5.4.3.4 preemphasis branch has no analogue here.
// `preflag` stays `false` in the returned [`OuterLoopResult`].
//
// # `mixed_block_flag == 1` scope
//
// Excluded from this round. Mixed blocks have 8 long-window scalefactor
// bands plus 9 short-window bands (sfb 3..=11) × 3 windows, with the
// §C.1.5.4.3.6 cap remapped (15 for sfb 0..=17, 7 for the rest per the
// spec's mixed-block text). The mixed analogue is a follow-up that
// composes the long-block amplifier (over the long-window bands 0..=7)
// with this short-block amplifier (over the short-window bands 3..=11);
// it is mechanically straightforward but its own piece of work and is
// dispatched as a separate round.
//
// # Acknowledgement
//
// No external implementation was consulted. The per-(sfb, window)
// amplification mirror, the bounded subblock_gain escalation triggered
// on §C.1.5.4.4.2 magnitude-clamp failure, and the `scalefac_scale`
// halving step are derived from the §C.1.5.4.3 outer-loop pseudocode and
// the §2.4.3.4.7.1 short-block formula directly.

/// Per-cell short-block scalefactor cap (§C.1.5.4.3.6 with our
/// `scalefac_compress = 15` ⇒ `(slen1, slen2) = (4, 3)`): 15 for the
/// slen1-range short scalefactor bands `sfb ∈ [0, 5]`.
pub const SCALEFAC_S_MAX_LOW: u8 = 15;

/// Per-cell short-block scalefactor cap (§C.1.5.4.3.6 with our
/// `scalefac_compress = 15` ⇒ `(slen1, slen2) = (4, 3)`): 7 for the
/// slen2-range short scalefactor bands `sfb ∈ [6, 11]`.
pub const SCALEFAC_S_MAX_HIGH: u8 = 7;

/// The per-cell short-block scalefactor upper limit for short `sfb` per
/// §C.1.5.4.3.6 / Table B.5 (with our `OUTER_LOOP_SCALEFAC_COMPRESS`).
#[must_use]
pub fn scalefac_short_upper_limit(sfb: usize) -> u8 {
    if sfb <= 5 {
        SCALEFAC_S_MAX_LOW
    } else {
        SCALEFAC_S_MAX_HIGH
    }
}

/// Compute the per-(sfb, window) actual short-block distortion
/// `xfsf_s[sfb][window]` (§C.1.5.4.3.3 applied to each short cell).
///
/// Layout invariant: `xr` and `xr_back` are in the native short-block
/// `(sfb, window, freqline)` interleave (the §2.4.3.4.8 reorder lives
/// downstream of this stage). For a short band with per-window start `s`
/// and width `w`, the native span `[3·s, 3·(s+w))` is laid out as three
/// runs `[win0 (w lines)][win1 (w lines)][win2 (w lines)]`.
///
/// The coloured-domain scaling matches `band_distortion_long`: with the
/// scalefactor applied inside the encoder's quantizer, our original-domain
/// residual must be multiplied by `2^(mult·scalefac_s(sfb, win))` (squared
/// to `2^(2·mult·scalefac_s(sfb, win))` on the SSE) to recover the spec's
/// coloured residual. Preflag does not apply to short blocks.
#[must_use]
pub fn band_distortion_short(
    xr: &[f32; NUM_LINES],
    xr_back: &[f32; NUM_LINES],
    sf: &ScaleFactors,
    scalefac_scale: bool,
    sample_rate_hz: u32,
    version: MpegVersion,
) -> [[f64; SHORT_WINDOWS]; SHORT_SFB] {
    use crate::requantize::scalefac_multiplier;
    let starts = short_band_starts(sample_rate_hz, version);
    let mult = f64::from(scalefac_multiplier(scalefac_scale));
    let mut out = [[0.0f64; SHORT_WINDOWS]; SHORT_SFB];
    for sfb in 0..SHORT_SFB {
        let win_start = starts[sfb];
        let win_width = starts[sfb + 1] - starts[sfb];
        if win_width == 0 {
            continue;
        }
        for (win, slot) in out[sfb].iter_mut().enumerate() {
            let base = 3 * win_start + win * win_width;
            let mut sse = 0.0f64;
            let mut count = 0u32;
            for k in 0..win_width {
                let i = base + k;
                if i >= NUM_LINES {
                    break;
                }
                let d = f64::from(xr[i].abs()) - f64::from(xr_back[i].abs());
                sse += d * d;
                count += 1;
            }
            if count == 0 {
                continue;
            }
            let bw = f64::from(count);
            // Re-scale our original-domain SSE to the coloured-domain SSE
            // the spec's metric uses, mirroring the long-block helper.
            // `mult` = 0.5 (scalefac_scale = 0) → 1.0 (= 1).
            let sf_val = f64::from(sf.short[sfb][win]);
            let scale = (2.0 * mult * sf_val).exp2();
            *slot = (sse / bw) * scale;
        }
    }
    out
}

/// Per-window max `|is_i|` after quantization, looking only at the
/// freqline range of window `w` across all short scalefactor bands.
/// Used to decide which window(s) to escalate `subblock_gain` on when
/// the global magnitude clamp fails.
fn per_window_max_abs(
    is: &[i32; NUM_LINES],
    sample_rate_hz: u32,
    version: MpegVersion,
) -> [i32; SHORT_WINDOWS] {
    let starts = short_band_starts(sample_rate_hz, version);
    let mut out = [0i32; SHORT_WINDOWS];
    for sfb in 0..SHORT_SFB {
        let win_start = starts[sfb];
        let win_width = starts[sfb + 1] - starts[sfb];
        if win_width == 0 {
            continue;
        }
        for (win, slot) in out.iter_mut().enumerate() {
            let base = 3 * win_start + win * win_width;
            for k in 0..win_width {
                let i = base + k;
                if i >= NUM_LINES {
                    break;
                }
                let v = is[i].unsigned_abs() as i32;
                if v > *slot {
                    *slot = v;
                }
            }
        }
    }
    out
}

/// Outcome of a short-block outer-loop search for one granule-channel.
#[derive(Debug, Clone, PartialEq)]
pub struct OuterLoopShortResult {
    /// The chosen per-(sfb, window) short-block scalefactors. `long` is
    /// left zero (pure-short path; the mixed-block variant is a separate
    /// follow-up).
    pub scalefactors: ScaleFactors,
    /// The chosen `global_gain` the last inner-loop pass settled on.
    pub global_gain: u8,
    /// The chosen per-window `subblock_gain[w]` (§2.4.2.7 3-bit field).
    /// Each is in `[0, 7]`; raised from zero only when the §C.1.5.4.4.2
    /// magnitude clamp couldn't fit window `w` under 8191.
    pub subblock_gain: [u8; SHORT_WINDOWS],
    /// The chosen `is[576]` quantized buffer matching the returned state.
    pub is: [i32; NUM_LINES],
    /// `scalefac_scale` flag (§2.4.2.7). `false` ⇒ multiplier 0.5; the
    /// outer loop may escalate to `true` (multiplier 1.0) when the cap
    /// would terminate amplification (§C.1.5.4.3).
    pub scalefac_scale: bool,
    /// Iteration accounting (same shape as [`OuterLoopStats`]).
    pub stats: OuterLoopStats,
}

/// Run the §C.1.5.4.3 outer (distortion-control) iteration loop for one
/// **pure-short** (`block_type == Short`, `mixed_block_flag == false`,
/// `window_switching_flag == true`) granule-channel.
///
/// Per-(sfb, window) amplification, plus a bounded `subblock_gain` search
/// triggered when the §C.1.5.4.4.2 magnitude clamp can't fit a window
/// under 8191. `gc_template.scalefac_compress` MUST be
/// [`OUTER_LOOP_SCALEFAC_COMPRESS`] so the part2 layout matches what the
/// side-info writer will emit (slen1=4, slen2=3 ⇒ caps 15 / 7).
///
/// `uniform_threshold` is `xmin(sb)` applied uniformly across every cell
/// (psychoacoustic model deferred — same convention as the long-block
/// loop). `max_iter` caps the outer loop at a finite count.
#[must_use]
pub fn outer_loop_search_short(
    xr: &[f32; NUM_LINES],
    gc_template: &GranuleChannel,
    sample_rate_hz: u32,
    version: MpegVersion,
    per_gc_bit_budget: u64,
    uniform_threshold: f64,
    max_iter: u32,
) -> OuterLoopShortResult {
    debug_assert!(gc_template.window_switching_flag);
    debug_assert_eq!(gc_template.block_type, BlockType::Short);
    debug_assert!(!gc_template.mixed_block_flag);

    // §C.1.5.4.2.1 init: scalefactors zero, scalefac_scale 0,
    // subblock_gain zero. preflag stays false (never set for short).
    let mut sf = ScaleFactors::default();
    let mut amplified = [[false; SHORT_WINDOWS]; SHORT_SFB];
    let mut scalefac_scale = false;
    let mut escalated_once = false;
    let mut subblock_gain: [u8; SHORT_WINDOWS] = [0, 0, 0];

    // Saved last-good state.
    let mut last_good_sf = sf;
    let mut last_good_scale = scalefac_scale;
    let mut last_good_sg = subblock_gain;
    let mut last_good_inner = run_inner_short(
        xr,
        gc_template,
        &sf,
        scalefac_scale,
        subblock_gain,
        sample_rate_hz,
        version,
        per_gc_bit_budget,
    );

    let mut iterations: u32 = 1;
    let converged;
    let mut bands_amplified_total: u32 = 0;

    loop {
        let inner = run_inner_short(
            xr,
            gc_template,
            &sf,
            scalefac_scale,
            subblock_gain,
            sample_rate_hz,
            version,
            per_gc_bit_budget,
        );

        // Decode-side reconstruction at the *current* state.
        let mut gc_full = *gc_template;
        gc_full.global_gain = inner.global_gain;
        gc_full.scalefac_compress = OUTER_LOOP_SCALEFAC_COMPRESS;
        gc_full.scalefac_scale = scalefac_scale;
        gc_full.subblock_gain = subblock_gain;
        gc_full.preflag = false;
        let xr_back = requantize(&inner.is, &gc_full, &sf, sample_rate_hz, version);
        let xfsf =
            band_distortion_short(xr, &xr_back, &sf, scalefac_scale, sample_rate_hz, version);

        // §C.1.5.4.4.2 magnitude-clamp follow-up: when the inner search
        // could not bring `max|is| ≤ 8191` (a single window grossly
        // outranges the others), raise the offending window's
        // `subblock_gain[w]` by 1 (saturating at the §2.4.2.7 cap of 7)
        // and restart the iteration body. This is the only path that
        // ever moves `subblock_gain` off zero; it strictly shrinks the
        // remaining work because raising `subblock_gain[w]` shrinks
        // window `w`'s pre-quantization magnitudes.
        if !inner.magnitude_clamped {
            let per_win = per_window_max_abs(&inner.is, sample_rate_hz, version);
            let mut bumped = false;
            // Identify the window(s) sitting hardest against the cap.
            // Bump every window that exceeds the cap; saturate at 7.
            for w in 0..SHORT_WINDOWS {
                if per_win[w] > crate::inner_loop::BIG_VALUES_LIMIT && subblock_gain[w] < 7 {
                    subblock_gain[w] += 1;
                    bumped = true;
                }
            }
            // If at least one window still has headroom on subblock_gain,
            // restart the iteration. If every over-cap window has already
            // saturated at 7, fall through to the standard outer-loop
            // termination below (the spec's "lack of computing time" /
            // "scalefactors would exceed limit" branches conservatively
            // accept the saturated state).
            if bumped {
                iterations += 1;
                if iterations >= max_iter {
                    converged = false;
                    last_good_sf = sf;
                    last_good_scale = scalefac_scale;
                    last_good_sg = subblock_gain;
                    last_good_inner = inner;
                    break;
                }
                continue;
            }
        }

        // Identify cells over threshold (§C.1.5.4.3.3 per-cell xfsf).
        let mut any_over = false;
        let mut would_exceed_cap = false;
        for (sfb, xfsf_row) in xfsf.iter().enumerate() {
            let cap = scalefac_short_upper_limit(sfb);
            for (win, &d) in xfsf_row.iter().enumerate() {
                if d > uniform_threshold {
                    any_over = true;
                    let next = u16::from(sf.short[sfb][win]) + 1;
                    if next > u16::from(cap) {
                        would_exceed_cap = true;
                    }
                }
            }
        }

        // §C.1.5.4.3.6 termination paths.
        if !any_over {
            converged = true;
            last_good_sf = sf;
            last_good_scale = scalefac_scale;
            last_good_sg = subblock_gain;
            last_good_inner = inner;
            break;
        }
        if would_exceed_cap && !escalated_once {
            // §C.1.5.4.3 escalation: switch to `scalefac_scale = 1`. To
            // preserve the current coloured spectrum the per-cell
            // scalefactors are halved (mult doubles 0.5 → 1.0 ⇒ halving
            // sf keeps `2^(mult·sf)` unchanged). The "amplified" tracker
            // resets so the doubled-step amplifications can re-fire.
            scalefac_scale = true;
            escalated_once = true;
            for row in sf.short.iter_mut() {
                for v in row.iter_mut() {
                    *v = (*v).div_ceil(2);
                }
            }
            amplified = [[false; SHORT_WINDOWS]; SHORT_SFB];
            iterations += 1;
            continue;
        }
        // (a) all cells already amplified, (b) cap exceeded (after one
        // escalation has fired or in mixed/short paths that lack further
        // headroom), or (c) defensive iteration cap.
        let all_amplified = amplified.iter().all(|row| row.iter().all(|&a| a));
        if would_exceed_cap || all_amplified || iterations >= max_iter {
            converged = false;
            break;
        }

        // Save current state as the new last-good before amplifying.
        last_good_sf = sf;
        last_good_scale = scalefac_scale;
        last_good_sg = subblock_gain;
        last_good_inner = inner;

        // §C.1.5.4.3.5 amplification, per-cell.
        for (sfb, xfsf_row) in xfsf.iter().enumerate() {
            let cap = scalefac_short_upper_limit(sfb);
            for (win, &d) in xfsf_row.iter().enumerate() {
                if d > uniform_threshold && sf.short[sfb][win] < cap {
                    sf.short[sfb][win] = sf.short[sfb][win].saturating_add(1);
                    if !amplified[sfb][win] {
                        amplified[sfb][win] = true;
                        bands_amplified_total += 1;
                    }
                }
            }
        }
        iterations += 1;
    }

    OuterLoopShortResult {
        scalefactors: last_good_sf,
        global_gain: last_good_inner.global_gain,
        subblock_gain: last_good_sg,
        is: last_good_inner.is,
        scalefac_scale: last_good_scale,
        stats: OuterLoopStats {
            iterations,
            bands_amplified: bands_amplified_total,
            converged,
        },
    }
}

/// Per-iteration short-block helper: install the outer loop's chosen
/// `scalefac_compress`, `scalefac_scale` and `subblock_gain` on top of
/// the template, then pick the smallest `global_gain` that fits the
/// inner-loop bit budget AND the §C.1.5.4.4.2 magnitude clamp. Returns
/// the chosen `global_gain`, the resulting `is[]`, and whether the
/// magnitude clamp was actually satisfied (so the caller can escalate
/// `subblock_gain` on a per-window basis when not).
#[allow(clippy::too_many_arguments)]
fn run_inner_short(
    xr: &[f32; NUM_LINES],
    gc_template: &GranuleChannel,
    sf: &ScaleFactors,
    scalefac_scale: bool,
    subblock_gain: [u8; SHORT_WINDOWS],
    sample_rate_hz: u32,
    version: MpegVersion,
    per_gc_bit_budget: u64,
) -> InnerShortInvocation {
    let mut gc = *gc_template;
    gc.scalefac_compress = OUTER_LOOP_SCALEFAC_COMPRESS;
    gc.preflag = false;
    gc.scalefac_scale = scalefac_scale;
    gc.subblock_gain = subblock_gain;
    let res_budget = search_bit_budget(xr, &gc, sf, sample_rate_hz, version, per_gc_bit_budget);
    let res_clamp = search_magnitude_clamp(xr, &gc, sf, sample_rate_hz, version);
    let gg = res_budget.global_gain.max(res_clamp.global_gain);
    let mut gc_final = gc;
    gc_final.global_gain = gg;
    let is = quantize(xr, &gc_final, sf, sample_rate_hz, version);
    let _ = (GAIN_MAX, res_budget.satisfied);
    InnerShortInvocation {
        global_gain: gg,
        is,
        magnitude_clamped: res_clamp.satisfied,
    }
}

#[derive(Debug, Clone)]
struct InnerShortInvocation {
    global_gain: u8,
    is: [i32; NUM_LINES],
    /// `true` if the §C.1.5.4.4.2 magnitude clamp could be satisfied at
    /// the chosen `global_gain` and `subblock_gain` configuration. When
    /// `false`, the caller bumps `subblock_gain[w]` on the window(s) that
    /// still exceed the cap.
    magnitude_clamped: bool,
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
    fn outer_loop_does_not_escalate_when_threshold_easily_met() {
        // A high threshold lets the loop converge on the very first
        // pass — no amplification, no escalation, scalefac_scale stays
        // false. This pins the §C.1.5.4.3 default behaviour: escalation
        // is conditional on the cap being hit, not the default state.
        let mut xr = [0.0f32; NUM_LINES];
        for (i, v) in xr.iter_mut().enumerate() {
            *v = ((i as f32) * 0.013).sin() * 100.0;
        }
        let gc = long_template();
        let res = outer_loop_search_long(&xr, &gc, 44_100, MpegVersion::Mpeg1, 2000, 1.0e30, 64);
        assert!(res.stats.converged);
        assert_eq!(res.stats.bands_amplified, 0);
        assert!(
            !res.scalefac_scale,
            "loop must not escalate scalefac_scale when no band exceeds threshold",
        );
    }

    #[test]
    fn outer_loop_escalates_scalefac_scale_when_cap_would_terminate() {
        // The §C.1.5.4.3 escalation path is exercised by a fixture that
        // (a) puts non-trivial energy ONLY in a single high-band (so
        // only that band exceeds threshold and the "all bands
        // amplified" termination cannot fire), and (b) drives that
        // band's per-iteration amplification past its §C.1.5.4.3.6 cap
        // (7 for sfb 11..=20). With every other band quiet
        // (distortion = 0), the amplified[] tracker only ever flips for
        // the one over-threshold band; the only termination path
        // reachable is "amplification of this band would exceed cap" →
        // the new code-path enters the §C.1.5.4.3 escalation.
        //
        // 44.1 kHz long-band sfb 19 covers lines [464, 540) per Table
        // 3-B.8. Plant several non-quantum-friendly values in that
        // range so quantization introduces residual error, then
        // calibrate the threshold to a tiny fraction of the baseline
        // distortion. This guarantees sfb 19 is always over-threshold
        // through every amp step until its cap (7) fires.
        let mut xr = [0.0f32; NUM_LINES];
        // Fill a single high-band (sfb 19 at 44.1 kHz covers lines
        // [288, 342) per Table 3-B.8b) with broadband high-energy data
        // so the inner loop must use a coarse global_gain to fit the
        // budget — guaranteeing non-trivial post-quantization residual
        // in that band.
        for (offset, slot) in xr[288..342].iter_mut().enumerate() {
            *slot = 1000.0 + (offset as f32) * 13.7;
        }
        let gc = long_template();

        // Calibrate threshold to baseline sfb-19 distortion / a big
        // factor so each amp step (which shrinks distortion by ~2× per
        // √2 boost) still leaves sfb 19 above threshold for at least 7
        // cycles. Other bands have zero baseline distortion, so they
        // never amplify (only sfb 19 is ever over threshold).
        let baseline_sf = ScaleFactors::default();
        let baseline = run_inner(
            &xr,
            &gc,
            &baseline_sf,
            false,
            44_100,
            MpegVersion::Mpeg1,
            2000,
        );
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
        // sfb 19 must carry the lion's share of the baseline distortion
        // — otherwise the fixture isn't isolating the high band.
        assert!(
            d_b[19] > 1.0e-4,
            "fixture failed to load distortion into sfb 19 (d_b[19]={})",
            d_b[19],
        );
        // Threshold tiny relative to baseline so sfb 19 stays
        // over-threshold across the seven amplification steps it takes
        // to hit its cap (7 for high-bands).
        let thr = d_b[19] / 1.0e12;

        let res = outer_loop_search_long(&xr, &gc, 44_100, MpegVersion::Mpeg1, 2000, thr, 128);
        assert!(
            res.scalefac_scale,
            "loop should have escalated to scalefac_scale = 1 \
             (sfb 19 cap-would-exceed termination), got scale={} \
             converged={} iters={} bands_amp={} sf={:?} d_b={:?}",
            res.scalefac_scale,
            res.stats.converged,
            res.stats.iterations,
            res.stats.bands_amplified,
            &res.scalefactors.long,
            &d_b,
        );
        // The loop terminated with last-good state. Bands that
        // amplified must still be within the §C.1.5.4.3.6 cap (the
        // escalation halved them and any post-escalation amps would
        // saturate at the cap rather than exceed it).
        for (sfb, &sf_val) in res.scalefactors.long.iter().enumerate() {
            let cap = scalefac_long_upper_limit(sfb);
            assert!(
                sf_val <= cap,
                "sfb={sfb} sf={sf_val} exceeds §C.1.5.4.3.6 cap {cap}",
            );
        }
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
        let baseline = run_inner(
            &xr,
            &gc,
            &baseline_sf,
            false,
            44_100,
            MpegVersion::Mpeg1,
            budget,
        );
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

    #[test]
    fn outer_loop_default_preflag_off_when_threshold_easily_met() {
        // §C.1.5.4.3.4: the spec's suggested heuristic switches preflag
        // on only when ALL of the upper-4 long bands (sfb 17..=20)
        // exceed threshold after the first inner pass. With a giant
        // threshold no band exceeds it, so preflag must stay off.
        let mut xr = [0.0f32; NUM_LINES];
        for (i, v) in xr.iter_mut().enumerate() {
            *v = ((i as f32) * 0.013).sin() * 100.0;
        }
        let gc = long_template();
        let res = outer_loop_search_long(&xr, &gc, 44_100, MpegVersion::Mpeg1, 2000, 1.0e30, 64);
        assert!(!res.preflag, "preflag must default to false");
        assert!(
            !res.scalefactors.preflag,
            "scalefactors.preflag must mirror result.preflag"
        );
    }

    #[test]
    fn outer_loop_default_preflag_off_when_only_low_bands_over_threshold() {
        // §C.1.5.4.3.4 heuristic requires ALL of sfb 17..=20 over
        // threshold. A fixture where only a low band has any energy
        // exposes those upper bands to zero baseline distortion — the
        // condition is unmet and preflag must stay off, exercising the
        // negative arm of the new decision branch.
        let mut xr = [0.0f32; NUM_LINES];
        xr[5] = 100.0;
        let gc = long_template();
        let res = outer_loop_search_long(&xr, &gc, 44_100, MpegVersion::Mpeg1, 2000, 1.0e-30, 64);
        assert!(
            !res.preflag,
            "preflag must NOT fire when upper-4 bands carry no energy",
        );
    }

    #[test]
    fn outer_loop_preflag_fires_when_all_upper_four_over_threshold() {
        // §C.1.5.4.3.4: build a fixture where all four of sfb 17..=20
        // carry energy producing a baseline distortion well above the
        // chosen threshold. The 44.1 kHz long-block band starts
        // (Table B.8b LONG_STARTS_44) are
        //   sfb 17 → [196, 238)
        //   sfb 18 → [238, 288)
        //   sfb 19 → [288, 342)
        //   sfb 20 → [342, 418)
        // Plant non-power-of-two amplitudes that quantize with residual
        // error so xfsf > 0 on every one of those bands; with a tiny
        // threshold the first-iteration §C.1.5.4.3.4 condition is met
        // and preflag flips on.
        let mut xr = [0.0f32; NUM_LINES];
        for (offset, slot) in xr[196..418].iter_mut().enumerate() {
            *slot = 500.0 + (offset as f32) * 7.3;
        }
        let gc = long_template();

        // Threshold tiny relative to the per-band baseline distortion
        // so xfsf[17..=20] all sit well above it after the very first
        // inner-loop pass.
        let res = outer_loop_search_long(&xr, &gc, 44_100, MpegVersion::Mpeg1, 2000, 1.0e-30, 128);
        assert!(
            res.preflag,
            "preflag should have fired (sfb 17..=20 all over threshold), \
             got preflag={} sf={:?} converged={} iters={}",
            res.preflag, &res.scalefactors.long, res.stats.converged, res.stats.iterations,
        );
        assert!(
            res.scalefactors.preflag,
            "result.scalefactors.preflag must mirror result.preflag \
             so the caller's quantize() / side-info write pick up the same flag",
        );
    }

    #[test]
    fn outer_loop_preflag_off_when_only_three_upper_bands_over() {
        // Strict reading of §C.1.5.4.3.4: ALL FOUR of sfb 17..=20 must
        // exceed threshold. If sfb 17 has zero energy (and therefore
        // zero baseline distortion) while 18/19/20 have plenty, the
        // condition is unmet and preflag stays off.
        let mut xr = [0.0f32; NUM_LINES];
        // Skip sfb 17 (lines [196, 238)); load sfb 18..=20.
        for (offset, slot) in xr[238..418].iter_mut().enumerate() {
            *slot = 500.0 + (offset as f32) * 7.3;
        }
        let gc = long_template();
        let res = outer_loop_search_long(&xr, &gc, 44_100, MpegVersion::Mpeg1, 2000, 1.0e-30, 128);
        assert!(
            !res.preflag,
            "preflag must NOT fire if any of sfb 17..=20 is at zero distortion \
             (got preflag={} sf={:?})",
            res.preflag, &res.scalefactors.long,
        );
    }

    // =====================================================================
    // Short-block outer loop tests (Phase 2 step 27)
    // =====================================================================

    fn short_template() -> GranuleChannel {
        GranuleChannel {
            part2_3_length: 0,
            big_values: 0,
            global_gain: 0,
            scalefac_compress: OUTER_LOOP_SCALEFAC_COMPRESS,
            window_switching_flag: true,
            block_type: BlockType::Short,
            mixed_block_flag: false,
            // Window-switched defaults from §2.4.2.7: region0_count = 8
            // (since short blocks have no long-region transmitted
            // region split — the fixed §C.1.5.4.4 SUBDIVIDE applies);
            // region1_count is unused for short.
            table_select: [0; 3],
            subblock_gain: [0; 3],
            region0_count: 8,
            region1_count: 0,
            preflag: false,
            scalefac_scale: false,
            count1table_select: false,
        }
    }

    #[test]
    fn short_upper_limits_match_spec() {
        // §C.1.5.4.3.6 caps with scalefac_compress=15 ⇒ slen1=4 → cap 15,
        // slen2=3 → cap 7. Pure-short split: sfb 0..=5 (slen1), sfb 6..=11
        // (slen2).
        for sfb in 0..=5 {
            assert_eq!(scalefac_short_upper_limit(sfb), SCALEFAC_S_MAX_LOW);
        }
        for sfb in 6..=11 {
            assert_eq!(scalefac_short_upper_limit(sfb), SCALEFAC_S_MAX_HIGH);
        }
    }

    #[test]
    fn short_band_distortion_zero_when_perfect() {
        // Identity reconstruction ⇒ zero distortion in every (sfb, win)
        // cell. Mirrors band_distortion_zero_when_perfect for shorts.
        let mut xr = [0.0f32; NUM_LINES];
        for (i, v) in xr.iter_mut().enumerate() {
            *v = ((i as f32) * 0.011).sin() * 80.0;
        }
        let xr_back = xr;
        let sf = ScaleFactors::default();
        let d = band_distortion_short(&xr, &xr_back, &sf, false, 44_100, MpegVersion::Mpeg1);
        for row in &d {
            for &v in row {
                assert!(v < 1e-12, "expected zero distortion, got {v}");
            }
        }
    }

    #[test]
    fn outer_loop_short_terminates_with_huge_threshold() {
        // Huge threshold ⇒ no cell exceeds it; the loop converges on the
        // first iteration with no amplification, no subblock_gain bumps.
        let mut xr = [0.0f32; NUM_LINES];
        for (i, v) in xr.iter_mut().enumerate() {
            *v = ((i as f32) * 0.011).sin() * 80.0;
        }
        let gc = short_template();
        let res = outer_loop_search_short(&xr, &gc, 44_100, MpegVersion::Mpeg1, 2000, 1.0e30, 64);
        assert!(res.stats.converged);
        assert_eq!(res.stats.iterations, 1);
        assert_eq!(res.stats.bands_amplified, 0);
        assert_eq!(res.subblock_gain, [0, 0, 0]);
        assert!(!res.scalefac_scale);
        // long array must stay zero (pure-short path).
        assert_eq!(res.scalefactors.long, [0u8; LONG_SFB]);
    }

    #[test]
    fn outer_loop_short_terminates_with_tiny_threshold() {
        // Threshold zero ⇒ every cell with any baseline distortion
        // exceeds it; the loop runs until cap-or-amplified termination.
        let mut xr = [0.0f32; NUM_LINES];
        for (i, v) in xr.iter_mut().enumerate() {
            *v = ((i as f32) * 0.011).sin() * 80.0;
        }
        let gc = short_template();
        let res = outer_loop_search_short(&xr, &gc, 44_100, MpegVersion::Mpeg1, 2000, 0.0, 64);
        assert!(!res.stats.converged);
        assert!(res.stats.bands_amplified > 0);
        assert!(res.stats.iterations >= 2);
        // Returned scalefactors must respect the §C.1.5.4.3.6 caps.
        for (sfb, row) in res.scalefactors.short.iter().enumerate() {
            let cap = scalefac_short_upper_limit(sfb);
            for (win, &sf_val) in row.iter().enumerate() {
                assert!(
                    sf_val <= cap,
                    "sfb={sfb} win={win} sf={sf_val} exceeds §C.1.5.4.3.6 cap {cap}",
                );
            }
        }
    }

    #[test]
    fn outer_loop_short_amplifies_only_offending_cells() {
        // Plant energy in exactly one (sfb, window) cell — sfb 1, window 1
        // at 44.1 kHz. SHORT_STARTS_44 places sfb 1 at per-window lines
        // [4, 8) (width 4); window 1's interleaved range starts at
        // 3*4 + 1*4 = 16. Other cells stay zero.
        let mut xr = [0.0f32; NUM_LINES];
        for k in 0..4 {
            xr[16 + k] = 60.0;
        }
        let gc = short_template();
        let budget = 2000u64;

        // Baseline distortion to calibrate the threshold.
        let baseline_sf = ScaleFactors::default();
        let baseline = run_inner_short(
            &xr,
            &gc,
            &baseline_sf,
            false,
            [0, 0, 0],
            44_100,
            MpegVersion::Mpeg1,
            budget,
        );
        let mut gc_b = gc;
        gc_b.global_gain = baseline.global_gain;
        let xr_back_b = requantize(
            &baseline.is,
            &gc_b,
            &baseline_sf,
            44_100,
            MpegVersion::Mpeg1,
        );
        let d_b = band_distortion_short(
            &xr,
            &xr_back_b,
            &baseline_sf,
            false,
            44_100,
            MpegVersion::Mpeg1,
        );
        // Threshold just below the loudest cell's baseline distortion so
        // only that cell is over threshold and is the only one amplified.
        let max_cell = d_b
            .iter()
            .flat_map(|r| r.iter().copied())
            .fold(0.0f64, f64::max);
        assert!(max_cell > 0.0, "fixture failed to introduce any distortion");
        let thr = max_cell * 0.5;
        let res = outer_loop_search_short(&xr, &gc, 44_100, MpegVersion::Mpeg1, budget, thr, 32);
        assert!(res.stats.bands_amplified >= 1);
        // No cell whose baseline distortion was exactly zero should ever
        // have been amplified — the loop's per-cell guard must prevent
        // amplifying silent cells (mirrors the long-block invariant).
        for (sfb, d_row) in d_b.iter().enumerate() {
            for (win, &d) in d_row.iter().enumerate() {
                if d == 0.0 && res.scalefactors.short[sfb][win] != 0 {
                    panic!(
                        "silent cell sfb={sfb} win={win} got amplified to {} (should stay 0)",
                        res.scalefactors.short[sfb][win],
                    );
                }
            }
        }
    }

    #[test]
    fn outer_loop_short_raises_subblock_gain_on_extreme_window() {
        // Build a fixture where window 0 carries amplitudes too large for
        // the §C.1.5.4.4.2 magnitude clamp at GAIN_MAX with default
        // (subblock_gain == 0). The clamp's reach is bounded: with
        // global_gain at 255, the gain factor `2^((255-210)/4)` ≈ 2435
        // divides the input before the `^0.75` power, so the largest
        // unclamped input magnitude is `8191^(4/3) · 2435 ≈ 4.4e8`.
        // Plant 5e9 magnitudes in window 0 only — well above the
        // GAIN_MAX reach — and confirm the loop bumps subblock_gain[0]
        // above zero. Each subblock_gain[0]+=1 adds an extra factor 4
        // to the divisor (`2^(8/4) = 4`), multiplying the reach by 4
        // per step.
        let mut xr = [0.0f32; NUM_LINES];
        // Window 0 only: sfb 0..=5 at 44.1 kHz cover per-window lines
        // [0, 4), [4, 8), [8, 12), [12, 16), [16, 22), [22, 30) (widths
        // 4, 4, 4, 4, 6, 8 — sum 30). Window 0's interleaved lines for
        // band sfb start at `3 * win_start + 0 * win_width = 3 * win_start`.
        // Stuff sfb 0 win 0 lines (interleaved [0, 4)) and sfb 1 win 0
        // lines (interleaved [12, 16)).
        for slot in xr.iter_mut().take(4) {
            *slot = 5.0e9; // sfb 0, win 0
        }
        for slot in xr.iter_mut().skip(12).take(4) {
            *slot = 5.0e9; // sfb 1, win 0
        }
        let gc = short_template();
        let res = outer_loop_search_short(&xr, &gc, 44_100, MpegVersion::Mpeg1, 2000, 1.0e30, 32);
        // subblock_gain[0] must have been bumped off zero (the only
        // window with over-cap energy). Windows 1 and 2 stay zero (no
        // energy planted there ⇒ per_window_max_abs stays 0 ⇒ no bump).
        assert!(
            res.subblock_gain[0] > 0,
            "subblock_gain[0] should have escalated; got {:?}",
            res.subblock_gain,
        );
        assert_eq!(res.subblock_gain[1], 0);
        assert_eq!(res.subblock_gain[2], 0);
        // subblock_gain stays within the §2.4.2.7 3-bit field range.
        for &sg in &res.subblock_gain {
            assert!(sg <= 7);
        }
    }

    #[test]
    fn outer_loop_short_subblock_gain_stays_zero_on_quiet_input() {
        // Modest amplitudes the magnitude clamp can fit at default
        // subblock_gain — confirm the loop never bumps the 3-bit field.
        let mut xr = [0.0f32; NUM_LINES];
        for (i, v) in xr.iter_mut().enumerate() {
            *v = ((i as f32) * 0.011).sin() * 50.0;
        }
        let gc = short_template();
        let res = outer_loop_search_short(&xr, &gc, 44_100, MpegVersion::Mpeg1, 2000, 1.0e30, 64);
        assert_eq!(res.subblock_gain, [0, 0, 0]);
    }

    #[test]
    fn outer_loop_short_default_preflag_off() {
        // Spec invariant: preflag is never set for short blocks
        // (§2.4.2.7). The result's `scalefactors.preflag` must stay
        // `false` regardless of input.
        let mut xr = [0.0f32; NUM_LINES];
        for (i, v) in xr.iter_mut().enumerate() {
            *v = ((i as f32) * 0.013).sin() * 100.0;
        }
        let gc = short_template();
        let res = outer_loop_search_short(&xr, &gc, 44_100, MpegVersion::Mpeg1, 2000, 1.0e30, 64);
        assert!(!res.scalefactors.preflag);
    }

    #[test]
    fn outer_loop_short_escalates_scalefac_scale_when_cap_would_terminate() {
        // §C.1.5.4.3 escalation path on the short-block loop: plant a
        // single high-band cell with non-quantum-friendly residual
        // energy so its post-quantization SSE stays above a tiny
        // threshold across every amp step until that cell's §C.1.5.4.3.6
        // cap fires. With every other cell quiet, only the one
        // over-threshold cell ever amplifies; the only reachable
        // termination is "cap exceeded" ⇒ the §C.1.5.4.3 escalation
        // branch flips scalefac_scale to true.
        //
        // SHORT_STARTS_44 places sfb 11 at per-window lines [106, 136)
        // (width 30); window 1's interleaved range starts at
        // 3*106 + 1*30 = 348 and runs to 378. Plant a band of
        // non-power-of-two values there.
        let mut xr = [0.0f32; NUM_LINES];
        for k in 0..30 {
            xr[348 + k] = 800.0 + (k as f32) * 11.7;
        }
        let gc = short_template();
        let baseline_sf = ScaleFactors::default();
        let baseline = run_inner_short(
            &xr,
            &gc,
            &baseline_sf,
            false,
            [0, 0, 0],
            44_100,
            MpegVersion::Mpeg1,
            2000,
        );
        let mut gc_b = gc;
        gc_b.global_gain = baseline.global_gain;
        let xr_back_b = requantize(
            &baseline.is,
            &gc_b,
            &baseline_sf,
            44_100,
            MpegVersion::Mpeg1,
        );
        let d_b = band_distortion_short(
            &xr,
            &xr_back_b,
            &baseline_sf,
            false,
            44_100,
            MpegVersion::Mpeg1,
        );
        // The chosen cell (sfb 11, win 1) must carry meaningful baseline
        // distortion for the fixture to exercise the escalation path.
        assert!(
            d_b[11][1] > 1.0e-6,
            "fixture failed to load distortion into sfb 11 win 1 \
             (d_b[11][1]={})",
            d_b[11][1],
        );
        let thr = d_b[11][1] / 1.0e12;
        let res = outer_loop_search_short(&xr, &gc, 44_100, MpegVersion::Mpeg1, 2000, thr, 128);
        assert!(
            res.scalefac_scale,
            "loop should have escalated to scalefac_scale = 1 \
             (sfb 11 win 1 cap-would-exceed termination), got scale={} \
             converged={} iters={} bands_amp={} sf_short={:?}",
            res.scalefac_scale,
            res.stats.converged,
            res.stats.iterations,
            res.stats.bands_amplified,
            &res.scalefactors.short,
        );
        // All cells must stay within the §C.1.5.4.3.6 cap.
        for (sfb, row) in res.scalefactors.short.iter().enumerate() {
            let cap = scalefac_short_upper_limit(sfb);
            for (win, &sf_val) in row.iter().enumerate() {
                assert!(
                    sf_val <= cap,
                    "sfb={sfb} win={win} sf={sf_val} exceeds §C.1.5.4.3.6 cap {cap}",
                );
            }
        }
    }
}
