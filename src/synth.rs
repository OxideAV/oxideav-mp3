//! Layer III **polyphase synthesis subband filterbank** — the final
//! decode stage that turns the 32×18 subband-domain time samples produced
//! by the IMDCT/overlap pipeline into 32×18 = 576 PCM samples per granule
//! per channel, per ISO/IEC 11172-3:1993 §2.4.3.2 / Figure A.2 ("Synthesis
//! subband filter flow chart") with the matrix coefficient formula
//! `N[i,k] = cos((16+i)·(2k+1)·π/64)` of §2.4.3.2.2 and the 512
//! window coefficients `D[]` of Annex B Table B.3.
//!
//! # Layer-vs-Layer
//!
//! The synthesis subband filter is shared by all three MPEG-1 audio
//! layers (§2.4.3.2.2 for Layer I, §2.4.3.3.5 for Layer II,
//! §2.4.3.4.10's "Synthesis via polyphase filterbank" stage for
//! Layer III). MPEG-2 LSF and MPEG-2.5 reuse the same filterbank
//! (the standard adds no LSF-specific changes to this stage).
//!
//! # Pseudo code (verbatim from Figure A.2)
//!
//! The flow chart on p.39 of ISO/IEC 11172-3:1993, applied per
//! subband-row (32 input samples `S[0..32]`), is:
//!
//! ```text
//! Shifting
//!   for i = 1023 down to 64 do  V[i] = V[i-64]   (V zero-initialised
//!                                                 at stream start)
//! Matrixing
//!   for i = 0 to 63 do  V[i] = sum_{k=0..31} N[i,k] * S[k]
//!     where  N[i,k] = cos((16+i)·(2k+1)·π/64)     (§2.4.3.2.2)
//!
//! Build a 512 values vector U
//!   for i = 0 to 7 do
//!     for j = 0 to 31 do
//!       U[64*i + j]      = V[128*i + j]
//!       U[64*i + 32 + j] = V[128*i + 96 + j]
//!
//! Window by 512 coefficients
//!   for i = 0 to 511 do  W[i] = U[i] * D[i]    (D[] from Table B.3)
//!
//! Calculate 32 samples
//!   for j = 0 to 31 do  S_out[j] = sum_{i=0..15} W[j + 32*i]
//! ```
//!
//! One call consumes 32 subband samples and produces 32 PCM samples; a
//! Layer III granule has 18 subband-time rows, so [`synth_granule`]
//! invokes the per-row filter 18 times and yields 18*32 = 576 PCM samples
//! per granule per channel.
//!
//! # Provenance
//!
//! Every numeric value in this module — the 512 `D[i]` Table B.3
//! coefficients and the `N[i,k]` matrixing formula — was transcribed
//! from the staged ISO/IEC 11172-3:1993 PDF (Annex B Table B.3 pages
//! 50–52 of the body; the formula and Figure A.2 pseudo code on pages
//! 31, 32 and 39).

use crate::imdct::{NUM_SUBBANDS, SAMPLES_PER_SUBBAND};

/// Output PCM samples produced per granule per channel (576 = 32 × 18).
pub const PCM_PER_GRANULE: usize = NUM_SUBBANDS * SAMPLES_PER_SUBBAND;

/// Size of the polyphase shift register `V[]` (Figure A.2 "Shifting"
/// step: indices `0..1024`).
pub const V_LEN: usize = 1024;

/// Size of the intermediate `U[]` / `W[]` vectors (Figure A.2 "Build
/// a 512 values vector U" and "Window by 512 coefficients" steps).
pub const U_LEN: usize = 512;

/// Per-channel polyphase shift register `V[0..1024]`.
///
/// Figure A.2's footnote 1: *"V to be initialized with zeroes during
/// startup."* A fresh [`SynthState::default`] is the correct decoder
/// start state.
#[derive(Debug, Clone)]
pub struct SynthState {
    v: [f64; V_LEN],
}

impl Default for SynthState {
    fn default() -> Self {
        SynthState { v: [0.0; V_LEN] }
    }
}

impl SynthState {
    /// A fresh all-zero shift register (stream-start state).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Read a single `V[i]` value (debug / test helper).
    #[must_use]
    pub fn v(&self, i: usize) -> f64 {
        if i < V_LEN {
            self.v[i]
        } else {
            0.0
        }
    }
}

/// Compute one matrixing coefficient `N[i,k] = cos((16+i)·(2k+1)·π/64)`
/// per §2.4.3.2.2 (the formula is identical in §2.4.3.3.5 for Layer II).
///
/// `i` ranges 0..64, `k` ranges 0..32.
#[must_use]
pub fn n_coefficient(i: usize, k: usize) -> f64 {
    let arg = ((16 + i) as f64) * ((2 * k + 1) as f64) * core::f64::consts::PI / 64.0;
    arg.cos()
}

/// Run one pass of the Figure A.2 polyphase synthesis filter: consume 32
/// subband samples `s[0..32]` and produce 32 PCM samples (returned).
///
/// The shift register `state.v` is updated in place. The returned values
/// are the 32 PCM samples in subband-frame order (lowest subband → highest;
/// the Layer I / II convention is to emit them as one consecutive run of
/// 32 audio samples per call).
#[must_use]
pub fn synth_row(s: &[f64; NUM_SUBBANDS], state: &mut SynthState) -> [f64; NUM_SUBBANDS] {
    // Step 1: Shifting — for i = 1023 down to 64 do V[i] = V[i-64].
    // `copy_within` performs the shift correctly (LLVM lowers to memmove,
    // which preserves overlapping semantics for descending indices).
    state.v.copy_within(0..(V_LEN - 64), 64);

    // Step 2: Matrixing — V[i] = sum_{k=0..31} N[i,k] * S[k] for i = 0..64.
    for (i, slot) in state.v[..64].iter_mut().enumerate() {
        let mut acc = 0.0f64;
        for (k, &sk) in s.iter().enumerate() {
            acc += n_coefficient(i, k) * sk;
        }
        *slot = acc;
    }

    // Step 3: Build a 512-values vector U[].
    //   for i = 0..8, for j = 0..32:
    //     U[64*i + j]      = V[128*i + j]
    //     U[64*i + 32 + j] = V[128*i + 96 + j]
    let mut u = [0.0f64; U_LEN];
    for i in 0..8 {
        let v_base = 128 * i;
        let u_base = 64 * i;
        for j in 0..32 {
            u[u_base + j] = state.v[v_base + j];
            u[u_base + 32 + j] = state.v[v_base + 96 + j];
        }
    }

    // Step 4: Window — W[i] = U[i] * D[i]. (We fold W into the summation
    // step rather than materialising it.)
    // Step 5: Sum — S_out[j] = sum_{i=0..16} W[j + 32*i] for j = 0..32.
    let mut out = [0.0f64; NUM_SUBBANDS];
    for (j, slot) in out.iter_mut().enumerate() {
        let mut acc = 0.0f64;
        for i in 0..16 {
            let idx = j + 32 * i;
            acc += u[idx] * D_TABLE[idx];
        }
        *slot = acc;
    }
    out
}

/// Convert one decoder output PCM sample (the §2.4.3.4 "range of the
/// output values of the decoder (PCM samples) is between -1,0 and +1,0"
/// fractional value) into a 16-bit signed integer sample.
///
/// The synthesis filterbank emits the reconstructed signal as a
/// fractional two's-complement number whose MSB carries the value -1
/// (§2.4.3.4.7.1: *"the requantized value … two's complement fractional
/// number, where the MSB represents the value -1"*), and §2.4.3.4.7
/// states the decoder's PCM output lies in `[-1.0, +1.0]`. A full-scale
/// 16-bit two's-complement sample spans `[-2^15, 2^15 - 1] = [-32768,
/// 32767]`, so the fractional value maps to the integer grid by scaling
/// by `2^15 = 32768` (the magnitude of the MSB weight), not `32767`.
///
/// The real-valued product is then rounded to the **nearest integer**,
/// with half-integer values rounded **away from zero** — exactly the
/// spec's "Nearest integer operator" (§2.3, the `Round()` / `[ ]`
/// operator: *"Returns the nearest integer value to the real-valued
/// argument. Half-integer values are rounded away from zero."*).
/// Rust's [`f32::round`] implements that rounding rule. Truncation
/// toward zero (`x as i16` on the un-rounded product) would bias every
/// non-integer sample one step toward zero and is not what the spec's
/// nearest-integer rule prescribes.
///
/// Finally the result is clipped to the representable 16-bit range. Only
/// an exact `+1.0` input (or a value that rounds up to `+32768`) needs
/// the high clip; `-1.0` maps to `-32768` without clipping.
#[must_use]
pub fn pcm_f32_to_i16(sample: f32) -> i16 {
    // Scale by the MSB weight 2^15, round half-away-from-zero, clip to
    // the signed 16-bit range [-32768, 32767].
    let scaled = (f64::from(sample) * 32768.0).round();
    scaled.clamp(f64::from(i16::MIN), f64::from(i16::MAX)) as i16
}

/// Run the §2.4.3.4.10 / §2.4.3.2 polyphase synthesis filter for one
/// granule-channel: 18 sequential calls to [`synth_row`] over the 32×18
/// subband-time block produced by [`crate::imdct::imdct_granule`].
///
/// `subband_time[sb][t]` is the `t`-th time sample of subband `sb` after
/// IMDCT / overlap / frequency inversion. The output is 576 PCM samples
/// in playback order: time-row 0 (32 samples, one per subband) → time-row
/// 1 → … → time-row 17. The shift register in `state` is updated in
/// place across rows and persists across granules within a stream.
#[must_use]
pub fn synth_granule(
    subband_time: &[[f32; SAMPLES_PER_SUBBAND]; NUM_SUBBANDS],
    state: &mut SynthState,
) -> [f32; PCM_PER_GRANULE] {
    let mut pcm = [0.0f32; PCM_PER_GRANULE];
    for t in 0..SAMPLES_PER_SUBBAND {
        // Gather one row of 32 subband samples (one per subband).
        let mut row = [0.0f64; NUM_SUBBANDS];
        for (sb, subband) in subband_time.iter().enumerate() {
            row[sb] = f64::from(subband[t]);
        }
        let out = synth_row(&row, state);
        let pcm_base = t * NUM_SUBBANDS;
        for (sb, &v) in out.iter().enumerate() {
            pcm[pcm_base + sb] = v as f32;
        }
    }
    pcm
}

/// Annex B Table B.3 — "Coefficients D_i of the synthesis window".
///
/// 512 values transcribed verbatim from ISO/IEC 11172-3:1993, Annex B,
/// Table B.3 (pages 50–52 of the body; rendered PNGs are staged at
/// `docs/audio/mp3/annex-b-renders/Table-B.3-coefficients-Di-p5{6,7,8}.png`).
///
/// The numerical values were chosen by the spec authors via numerical
/// optimisation (per the §2.4.3.2.2 prose); no closed-form derivation is
/// available, so each entry was hand-transcribed (with OCR cross-check
/// against the PNG renders for any text-extraction-suspect characters).
pub static D_TABLE: [f64; U_LEN] = [
    // D[0..32]
    0.000000000,
    -0.000015259,
    -0.000015259,
    -0.000015259,
    -0.000015259,
    -0.000015259,
    -0.000015259,
    -0.000030518,
    -0.000030518,
    -0.000030518,
    -0.000030518,
    -0.000045776,
    -0.000045776,
    -0.000061035,
    -0.000061035,
    -0.000076294,
    -0.000076294,
    -0.000091553,
    -0.000106812,
    -0.000106812,
    -0.000122070,
    -0.000137329,
    -0.000152588,
    -0.000167847,
    -0.000198364,
    -0.000213623,
    -0.000244141,
    -0.000259399,
    -0.000289917,
    -0.000320435,
    -0.000366211,
    -0.000396729,
    // D[32..64]
    -0.000442505,
    -0.000473022,
    -0.000534058,
    -0.000579834,
    -0.000625610,
    -0.000686646,
    -0.000747681,
    -0.000808716,
    -0.000885010,
    -0.000961304,
    -0.001037598,
    -0.001113892,
    -0.001205444,
    -0.001296997,
    -0.001388550,
    -0.001480103,
    -0.001586914,
    -0.001693726,
    -0.001785278,
    -0.001907349,
    -0.002014160,
    -0.002120972,
    -0.002243042,
    -0.002349854,
    -0.002456665,
    -0.002578735,
    -0.002685547,
    -0.002792358,
    -0.002899170,
    -0.002990723,
    -0.003082275,
    -0.003173828,
    // D[64..96]
    0.003250122,
    0.003326416,
    0.003387451,
    0.003433228,
    0.003463745,
    0.003479004,
    0.003479004,
    0.003463745,
    0.003417969,
    0.003372192,
    0.003280640,
    0.003173828,
    0.003051758,
    0.002883911,
    0.002700806,
    0.002487183,
    0.002227783,
    0.001937866,
    0.001617432,
    0.001266479,
    0.000869751,
    0.000442505,
    -0.000030518,
    -0.000549316,
    -0.001098633,
    -0.001693726,
    -0.002334595,
    -0.003005981,
    -0.003723145,
    -0.004486084,
    -0.005294800,
    -0.006118774,
    // D[96..128]
    -0.007003784,
    -0.007919312,
    -0.008865356,
    -0.009841919,
    -0.010848999,
    -0.011886597,
    -0.012939453,
    -0.014022827,
    -0.015121460,
    -0.016235352,
    -0.017349243,
    -0.018463135,
    -0.019577026,
    -0.020690918,
    -0.021789551,
    -0.022857666,
    -0.023910522,
    -0.024932861,
    -0.025909424,
    -0.026840210,
    -0.027725220,
    -0.028533936,
    -0.029281616,
    -0.029937744,
    -0.030532837,
    -0.031005859,
    -0.031387329,
    -0.031661987,
    -0.031814575,
    -0.031845093,
    -0.031738281,
    -0.031478882,
    // D[128..160]
    0.031082153,
    0.030517578,
    0.029785156,
    0.028884888,
    0.027801514,
    0.026535034,
    0.025085449,
    0.023422241,
    0.021575928,
    0.019531250,
    0.017257690,
    0.014801025,
    0.012115479,
    0.009231567,
    0.006134033,
    0.002822876,
    -0.000686646,
    -0.004394531,
    -0.008316040,
    -0.012420654,
    -0.016708374,
    -0.021179199,
    -0.025817871,
    -0.030609131,
    -0.035552979,
    -0.040634155,
    -0.045837402,
    -0.051132202,
    -0.056533813,
    -0.061996460,
    -0.067520142,
    -0.073059082,
    // D[160..192]
    -0.078628540,
    -0.084182739,
    -0.089706421,
    -0.095169067,
    -0.100540161,
    -0.105819702,
    -0.110946655,
    -0.115921021,
    -0.120697021,
    -0.125259399,
    -0.129562378,
    -0.133590698,
    -0.137298584,
    -0.140670776,
    -0.143676758,
    -0.146255493,
    -0.148422241,
    -0.150115967,
    -0.151306152,
    -0.151962280,
    -0.152069092,
    -0.151596069,
    -0.150497437,
    -0.148773193,
    -0.146362305,
    -0.143264771,
    -0.139450073,
    -0.134887695,
    -0.129577637,
    -0.123474121,
    -0.116577148,
    -0.108856201,
    // D[192..224]
    0.100311279,
    0.090927124,
    0.080688477,
    0.069595337,
    0.057617187,
    0.044784546,
    0.031082153,
    0.016510010,
    0.001068115,
    -0.015228271,
    -0.032379150,
    -0.050354004,
    -0.069168091,
    -0.088775635,
    -0.109161377,
    -0.130310059,
    -0.152206421,
    -0.174789429,
    -0.198059082,
    -0.221984863,
    -0.246505737,
    -0.271591187,
    -0.297210693,
    -0.323318481,
    -0.349868774,
    -0.376800537,
    -0.404083252,
    -0.431655884,
    -0.459472656,
    -0.487472534,
    -0.515609741,
    -0.543823242,
    // D[224..256]
    -0.572036743,
    -0.600219727,
    -0.628295898,
    -0.656219482,
    -0.683914185,
    -0.711318970,
    -0.738372803,
    -0.765029907,
    -0.791213989,
    -0.816864014,
    -0.841949463,
    -0.866363525,
    -0.890090942,
    -0.913055420,
    -0.935195923,
    -0.956481934,
    -0.976852417,
    -0.996246338,
    -1.014617920,
    -1.031936646,
    -1.048156738,
    -1.063217163,
    -1.077117920,
    -1.089782715,
    -1.101211548,
    -1.111373901,
    -1.120223999,
    -1.127746582,
    -1.133926392,
    -1.138763428,
    -1.142211914,
    -1.144287109,
    // D[256..288]
    1.144989014,
    1.144287109,
    1.142211914,
    1.138763428,
    1.133926392,
    1.127746582,
    1.120223999,
    1.111373901,
    1.101211548,
    1.089782715,
    1.077117920,
    1.063217163,
    1.048156738,
    1.031936646,
    1.014617920,
    0.996246338,
    0.976852417,
    0.956481934,
    0.935195923,
    0.913055420,
    0.890090942,
    0.866363525,
    0.841949463,
    0.816864014,
    0.791213989,
    0.765029907,
    0.738372803,
    0.711318970,
    0.683914185,
    0.656219482,
    0.628295898,
    0.600219727,
    // D[288..320]
    0.572036743,
    0.543823242,
    0.515609741,
    0.487472534,
    0.459472656,
    0.431655884,
    0.404083252,
    0.376800537,
    0.349868774,
    0.323318481,
    0.297210693,
    0.271591187,
    0.246505737,
    0.221984863,
    0.198059082,
    0.174789429,
    0.152206421,
    0.130310059,
    0.109161377,
    0.088775635,
    0.069168091,
    0.050354004,
    0.032379150,
    0.015228271,
    -0.001068115,
    -0.016510010,
    -0.031082153,
    -0.044784546,
    -0.057617187,
    -0.069595337,
    -0.080688477,
    -0.090927124,
    // D[320..352]
    0.100311279,
    0.108856201,
    0.116577148,
    0.123474121,
    0.129577637,
    0.134887695,
    0.139450073,
    0.143264771,
    0.146362305,
    0.148773193,
    0.150497437,
    0.151596069,
    0.152069092,
    0.151962280,
    0.151306152,
    0.150115967,
    0.148422241,
    0.146255493,
    0.143676758,
    0.140670776,
    0.137298584,
    0.133590698,
    0.129562378,
    0.125259399,
    0.120697021,
    0.115921021,
    0.110946655,
    0.105819702,
    0.100540161,
    0.095169067,
    0.089706421,
    0.084182739,
    // D[352..384]
    0.078628540,
    0.073059082,
    0.067520142,
    0.061996460,
    0.056533813,
    0.051132202,
    0.045837402,
    0.040634155,
    0.035552979,
    0.030609131,
    0.025817871,
    0.021179199,
    0.016708374,
    0.012420654,
    0.008316040,
    0.004394531,
    0.000686646,
    -0.002822876,
    -0.006134033,
    -0.009231567,
    -0.012115479,
    -0.014801025,
    -0.017257690,
    -0.019531250,
    -0.021575928,
    -0.023422241,
    -0.025085449,
    -0.026535034,
    -0.027801514,
    -0.028884888,
    -0.029785156,
    -0.030517578,
    // D[384..416]
    0.031082153,
    0.031478882,
    0.031738281,
    0.031845093,
    0.031814575,
    0.031661987,
    0.031387329,
    0.031005859,
    0.030532837,
    0.029937744,
    0.029281616,
    0.028533936,
    0.027725220,
    0.026840210,
    0.025909424,
    0.024932861,
    0.023910522,
    0.022857666,
    0.021789551,
    0.020690918,
    0.019577026,
    0.018463135,
    0.017349243,
    0.016235352,
    0.015121460,
    0.014022827,
    0.012939453,
    0.011886597,
    0.010848999,
    0.009841919,
    0.008865356,
    0.007919312,
    // D[416..448]
    0.007003784,
    0.006118774,
    0.005294800,
    0.004486084,
    0.003723145,
    0.003005981,
    0.002334595,
    0.001693726,
    0.001098633,
    0.000549316,
    0.000030518,
    -0.000442505,
    -0.000869751,
    -0.001266479,
    -0.001617432,
    -0.001937866,
    -0.002227783,
    -0.002487183,
    -0.002700806,
    -0.002883911,
    -0.003051758,
    -0.003173828,
    -0.003280640,
    -0.003372192,
    -0.003417969,
    -0.003463745,
    -0.003479004,
    -0.003479004,
    -0.003463745,
    -0.003433228,
    -0.003387451,
    -0.003326416,
    // D[448..480]
    0.003250122,
    0.003173828,
    0.003082275,
    0.002990723,
    0.002899170,
    0.002792358,
    0.002685547,
    0.002578735,
    0.002456665,
    0.002349854,
    0.002243042,
    0.002120972,
    0.002014160,
    0.001907349,
    0.001785278,
    0.001693726,
    0.001586914,
    0.001480103,
    0.001388550,
    0.001296997,
    0.001205444,
    0.001113892,
    0.001037598,
    0.000961304,
    0.000885010,
    0.000808716,
    0.000747681,
    0.000686646,
    0.000625610,
    0.000579834,
    0.000534058,
    0.000473022,
    // D[480..512]
    0.000442505,
    0.000396729,
    0.000366211,
    0.000320435,
    0.000289917,
    0.000259399,
    0.000244141,
    0.000213623,
    0.000198364,
    0.000167847,
    0.000152588,
    0.000137329,
    0.000122070,
    0.000106812,
    0.000106812,
    0.000091553,
    0.000076294,
    0.000076294,
    0.000061035,
    0.000061035,
    0.000045776,
    0.000045776,
    0.000030518,
    0.000030518,
    0.000030518,
    0.000030518,
    0.000015259,
    0.000015259,
    0.000015259,
    0.000015259,
    0.000015259,
    0.000015259,
];

#[cfg(test)]
// The tests recompute every reference value directly from §2.4.3.2 /
// Figure A.2 formulas, written in their natural `for i in 0..N` index-
// driven shape (the index variable is part of the spec formula, not just
// a buffer iterator), so the explicit loops mirror the spec text more
// faithfully than iterator chains.
#[allow(clippy::needless_range_loop)]
mod tests_inner {
    use super::*;
    include!("synth_tests.rs");
}
