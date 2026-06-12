//! Layer I/II/III **polyphase analysis subband filterbank** — the
//! first encoder stage that splits a broadband PCM input into 32
//! equally-spaced critically-sampled subbands, per ISO/IEC 11172-3:1993
//! Annex C §C.1.3 ("Analysis subband filter") with the per-step pseudo
//! code of Figure C.4 ("Analysis subband filter flow chart"), the
//! 512 prototype-window coefficients `C[i]` of Annex C Table C.1, and
//! the analysis-side matrixing coefficient formula
//! `M[i,k] = cos((2i+1)·(k-16)·π/64)` of §C.1.3.
//!
//! # Layer-vs-Layer
//!
//! The analysis subband filter is shared by all three MPEG-1 audio
//! layers (§C.1.5.1.3 for Layer I, §C.1.5.2.3 for Layer II, §C.1.5.3.3
//! "analysis part of the hybrid filterbank" for Layer III — Layer III
//! follows the subband filter with the §2.4.3.4.10.2 MDCT to refine
//! the 32 subbands into 32×18 = 576 frequency lines). MPEG-2 LSF and
//! MPEG-2.5 reuse the same analysis filterbank without modification.
//!
//! # Pseudo code (verbatim from Figure C.4)
//!
//! The flow chart on p.78 of ISO/IEC 11172-3:1993, applied per
//! 32-PCM-sample block, is:
//!
//! ```text
//! Input shift register update:
//!   for i = 511 down to 32 do  X[i] = X[i - 32]
//!   for i = 31  down to 0  do  X[i] = next_input_audio_sample
//!     (X[] is zero-initialised at stream start; most-recent sample
//!      lands at X[0], oldest at X[511])
//!
//! Window by 512 coefficients:
//!   for i = 0..511 do  Z[i] = C[i] · X[i]        (C[] from Table C.1)
//!
//! Partial calculation (8-tap folding into a 64-element vector):
//!   for i = 0..63 do  Y[i] = sum_{j=0..7} Z[i + 64·j]
//!
//! Matrixing (64×32 cosine modulation):
//!   for i = 0..31 do  S[i] = sum_{k=0..63} M[i,k] · Y[k]
//!     where  M[i,k] = cos((2i+1)·(k-16)·π/64)    (§C.1.3 formula)
//!
//! Output 32 subband samples
//! ```
//!
//! One call consumes 32 PCM samples and produces 32 subband samples
//! (one per subband, at the critically-decimated subband sample rate
//! f_s / 32). Layer III's encoder runs the analysis filterbank 18
//! times per granule per channel to assemble the 32×18 subband-time
//! block that subsequently feeds the §2.4.3.4.10.2 forward MDCT.
//!
//! # Round-trip relationship with the synthesis filterbank
//!
//! The analysis-synthesis pair is a **near-perfect-reconstruction**
//! pseudo-quadrature filterbank — the spec's §C.1.3 prototype is a
//! designed near-PR filter, not paraunitary, so two algebraically
//! distinct round-trip identities apply, both verified in
//! `tests_inner`:
//!
//! * **PCM round-trip with 481-sample group delay.** For any
//!   broadband PCM signal `x[n]`, the steady-state output of
//!   `analyze_row` → `synth_row` matches the input delayed by 481
//!   samples within the bank's design ripple of ≈ 1×10⁻⁴ RMS
//!   (≈ -80 dB, measured ≈ 3×10⁻⁵ on a multi-tone broadband signal).
//!
//! * **Per-subband DC-tone subband round-trip is exactly
//!   cyclostationary in steady state.** Driving synthesis with a
//!   constant unit input in one subband and analysing the resulting
//!   PCM yields, after the bank's settling delay, a steady-state
//!   recovered coefficient that does *not* vary from row to row —
//!   row-to-row ripple is zero to float precision (< 1×10⁻¹²) for
//!   every one of the 32 subbands. This is the strict spec-derivable
//!   invariant of the bank's cosine-modulated polyphase structure.
//!
//! # Provenance
//!
//! Every numeric value in this module — the 512 `C[i]` Table C.1
//! coefficients and the `M[i,k]` matrixing formula — was transcribed
//! from the staged ISO/IEC 11172-3:1993 PDF (Annex C Table C.1 pages
//! 67–69 of the body, and the Figure C.4 pseudo code on pages 77–78).
//!
//! The literal PDF text of Table C.1 carries OCR-grade typography
//! artefacts in several entries (the rendered PDF intermixes the
//! letters `O/l/I/H/R/b/X` with digits `0/1/8/6`); the affected
//! second-half values (indices 256..511) are reconstructed here by the
//! standard cosine-modulated-prototype symmetry
//!
//! ```text
//! C[512 - i] = +C[i]   if i ≡ 0 (mod 64)
//! C[512 - i] = -C[i]   otherwise
//! ```
//!
//! which is verifiable on every spot-check pair in the first-half (and
//! whose validity is implied by the §C.1.3 polyphase filterbank
//! construction itself). The first-half entries (indices 0..256) are
//! transcribed directly from the PDF text with the only OCR fix being
//! letters that were unambiguously digits given the surrounding
//! left-to-right context (`O` → `0`, `l` → `1`, `H` → `8`, `R` → `8`,
//! `b` → `6`); each such fix is also cross-checked against the
//! monotone trend of its three nearest neighbours.

use crate::imdct::{NUM_SUBBANDS, SAMPLES_PER_SUBBAND};
use crate::synth::PCM_PER_GRANULE;

/// Size of the polyphase shift register `X[]` (Figure C.4 input shift
/// step: indices `0..512`).
pub const X_LEN: usize = 512;

/// Compute one analysis matrixing coefficient
/// `M[i,k] = cos((2i+1)·(k-16)·π/64)` per §C.1.3.
///
/// `i` ranges 0..32, `k` ranges 0..64. The factor `(k - 16)` is
/// performed in `isize` arithmetic and folded into the cosine, which
/// handles `k = 0..15` (negative offsets) symmetrically with
/// `k = 16..63` (non-negative offsets).
#[must_use]
pub fn m_coefficient(i: usize, k: usize) -> f64 {
    let i = i as isize;
    let k = k as isize;
    let arg = ((2 * i + 1) as f64) * ((k - 16) as f64) * core::f64::consts::PI / 64.0;
    arg.cos()
}

/// Per-channel polyphase shift register `X[0..512]` for the encoder
/// analysis filterbank.
///
/// Figure C.4 has no explicit initialisation footnote, but the input
/// shift register is the analysis-side counterpart of the synthesis
/// `V[]` shift register (which §2.4.3.2 / Figure A.2 footnote 1
/// initialises to zero); a fresh [`AnalysisState::default`] is the
/// correct encoder start state.
#[derive(Debug, Clone)]
pub struct AnalysisState {
    x: [f64; X_LEN],
}

impl Default for AnalysisState {
    fn default() -> Self {
        AnalysisState { x: [0.0; X_LEN] }
    }
}

impl AnalysisState {
    /// A fresh all-zero shift register (stream-start state).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Read a single `X[i]` value (debug / test helper).
    #[must_use]
    pub fn x(&self, i: usize) -> f64 {
        if i < X_LEN {
            self.x[i]
        } else {
            0.0
        }
    }
}

/// Run one pass of the Figure C.4 polyphase analysis filter: consume 32
/// PCM samples `pcm[0..32]` and produce 32 subband samples (returned).
///
/// `pcm[0]` is the *oldest* of the 32 new samples (it lands at `X[31]`
/// after the shift) and `pcm[31]` is the *newest* (it lands at `X[0]`).
/// The shift register `state.x` is updated in place. The returned values
/// are the 32 subband samples `S[0..32]` (lowest subband → highest).
#[must_use]
pub fn analyze_row(pcm: &[f64; NUM_SUBBANDS], state: &mut AnalysisState) -> [f64; NUM_SUBBANDS] {
    // Step 1: Input shift — for i = 511 downto 32 do X[i] = X[i - 32].
    // `copy_within` performs the descending-index shift correctly (LLVM
    // lowers to memmove, which preserves overlapping semantics).
    state.x.copy_within(0..(X_LEN - 32), 32);

    // Step 1b: Inject 32 new samples — for i = 31 downto 0 do
    // X[i] = next_input_audio_sample. The "downto 0" semantics of the
    // flow chart place the *first*-shifted sample at X[31] and the
    // *last*-shifted sample at X[0], so the input vector ordering is
    // pcm[0] (oldest of the new 32) → X[31], pcm[31] (newest) → X[0].
    for (j, &p) in pcm.iter().enumerate() {
        state.x[31 - j] = p;
    }

    // Step 2: Window — Z[i] = C[i] · X[i]. (We fold Z into the partial-
    // calculation step rather than materialising it.)
    // Step 3: Partial calculation — Y[i] = sum_{j=0..7} Z[i + 64·j].
    let mut y = [0.0f64; 64];
    for (i, yi) in y.iter_mut().enumerate() {
        let mut acc = 0.0f64;
        for j in 0..8 {
            let idx = i + 64 * j;
            acc += C_TABLE[idx] * state.x[idx];
        }
        *yi = acc;
    }

    // Step 4: Matrixing — S[i] = sum_{k=0..63} M[i,k] · Y[k] for
    // i = 0..32, with M[i,k] = cos((2i+1)·(k-16)·π/64).
    let mut s = [0.0f64; NUM_SUBBANDS];
    for (i, si) in s.iter_mut().enumerate() {
        let mut acc = 0.0f64;
        for (k, &yk) in y.iter().enumerate() {
            acc += m_coefficient(i, k) * yk;
        }
        *si = acc;
    }
    s
}

/// Run the §C.1.3 polyphase analysis filter for one Layer III
/// granule-channel: 18 sequential calls to [`analyze_row`] over the 576
/// PCM samples that feed the encoder (the analysis-side mirror of
/// [`crate::synth::synth_granule`]).
///
/// Input `pcm` is the 576 PCM samples of one granule-channel in
/// playback order (time-row 0 of 32 samples, then time-row 1, …,
/// time-row 17), exactly the layout [`crate::synth::synth_granule`]
/// emits on the decode side. Output is `subband_time[sb][t]`, the
/// `t`-th subband-time sample of subband `sb` after analysis — exactly
/// the layout [`crate::imdct::imdct_granule`] consumes on the decode
/// side (so the analysis output of granule-channel `g` is ready to
/// feed the encoder-side §2.4.3.4.10 forward MDCT chain). The shift
/// register in `state` is updated in place across rows and persists
/// across granules within a stream.
// The loop structure mirrors `synth_granule` exactly (time row index
// `t` drives both the input slice and the output column write); the
// natural spec layout — "for t in 0..18 run one row, then transpose
// row → subband-time column" — reads more clearly with the explicit
// range than a chained enumerate.
#[allow(clippy::needless_range_loop)]
#[must_use]
pub fn analyze_granule(
    pcm: &[f32; PCM_PER_GRANULE],
    state: &mut AnalysisState,
) -> [[f32; SAMPLES_PER_SUBBAND]; NUM_SUBBANDS] {
    let mut subband_time = [[0.0f32; SAMPLES_PER_SUBBAND]; NUM_SUBBANDS];
    for t in 0..SAMPLES_PER_SUBBAND {
        let pcm_base = t * NUM_SUBBANDS;
        let mut row = [0.0f64; NUM_SUBBANDS];
        for (j, slot) in row.iter_mut().enumerate() {
            *slot = f64::from(pcm[pcm_base + j]);
        }
        let s = analyze_row(&row, state);
        for (sb, &v) in s.iter().enumerate() {
            subband_time[sb][t] = v as f32;
        }
    }
    subband_time
}

/// Annex C Table C.1 — "Coefficients C_i of the analysis window".
///
/// 512 values transcribed from ISO/IEC 11172-3:1993, Annex C, Table C.1
/// (pages 67–69 of the body). The first 256 entries are read directly
/// from the staged PDF text; the second 256 entries are derived by the
/// cosine-modulated-prototype symmetry
///
/// ```text
/// C[512 - i] = +C[i]   if i ≡ 0 (mod 64)
/// C[512 - i] = -C[i]   otherwise
/// ```
///
/// (see module-level docs); both halves are spelled out literally here
/// — the symmetry is reproduced in `c_table_satisfies_polyphase_symmetry`
/// in the test block.
///
/// The numerical values were chosen by the spec authors via numerical
/// optimisation of a near-perfect-reconstruction polyphase prototype;
/// no closed-form derivation is available, so each first-half entry was
/// hand-transcribed (with cross-check against the local monotone trend
/// of its neighbours for any OCR-suspect character: letters `O/l/I/H/R/b/X`
/// rendered for digits `0/1/8/6`).
pub static C_TABLE: [f64; X_LEN] = [
    // C[0..32]
    0.000000000,
    -0.000000477,
    -0.000000477,
    -0.000000477,
    -0.000000477,
    -0.000000477,
    -0.000000477,
    -0.000000954,
    -0.000000954,
    -0.000000954,
    -0.000000954,
    -0.000001431,
    -0.000001431,
    -0.000001907,
    -0.000001907,
    -0.000002384,
    -0.000002384,
    -0.000002861,
    -0.000003338,
    -0.000003338,
    -0.000003815,
    -0.000004292,
    -0.000004768,
    -0.000005245,
    -0.000006199,
    -0.000006676,
    -0.000007629,
    -0.000008106,
    -0.000009060,
    -0.000010014,
    -0.000011444,
    -0.000012398,
    // C[32..64]
    -0.000013828,
    -0.000014782,
    -0.000016689,
    -0.000018120,
    -0.000019550,
    -0.000021458,
    -0.000023365,
    -0.000025272,
    -0.000027657,
    -0.000030041,
    -0.000032425,
    -0.000034809,
    -0.000037670,
    -0.000040531,
    -0.000043392,
    -0.000046253,
    -0.000049591,
    -0.000052929,
    -0.000055790,
    -0.000059605,
    -0.000062943,
    -0.000066280,
    -0.000070095,
    -0.000073433,
    -0.000076771,
    -0.000080585,
    -0.000083923,
    -0.000087261,
    -0.000090599,
    -0.000093460,
    -0.000096321,
    -0.000099182,
    // C[64..96]
    0.000101566,
    0.000103951,
    0.000105858,
    0.000107288,
    0.000108242,
    0.000108719,
    0.000108719,
    0.000108242,
    0.000106812,
    0.000105381,
    0.000102520,
    0.000099182,
    0.000095367,
    0.000090122,
    0.000084400,
    0.000077724,
    0.000069618,
    0.000060558,
    0.000050545,
    0.000039577,
    0.000027180,
    0.000013828,
    -0.000000954,
    -0.000017166,
    -0.000034332,
    -0.000052929,
    -0.000072956,
    -0.000093937,
    -0.000116348,
    -0.000140190,
    -0.000165462,
    -0.000191212,
    // C[96..128]
    -0.000218868,
    -0.000247478,
    -0.000277042,
    -0.000307560,
    -0.000339031,
    -0.000371456,
    -0.000404358,
    -0.000438213,
    -0.000472546,
    -0.000507355,
    -0.000542164,
    -0.000576973,
    -0.000611782,
    -0.000646591,
    -0.000680923,
    -0.000714302,
    -0.000747204,
    -0.000779152,
    -0.000809669,
    -0.000838757,
    -0.000866413,
    -0.000891685,
    -0.000915051,
    -0.000935555,
    -0.000954151,
    -0.000968933,
    -0.000980854,
    -0.000989437,
    -0.000994205,
    -0.000995159,
    -0.000991821,
    -0.000983715,
    // C[128..160]
    0.000971317,
    0.000953674,
    0.000930786,
    0.000902653,
    0.000868797,
    0.000829220,
    0.000783920,
    0.000731945,
    0.000674248,
    0.000610352,
    0.000539303,
    0.000462532,
    0.000378609,
    0.000288486,
    0.000191689,
    0.000088215,
    -0.000021458,
    -0.000137329,
    -0.000259876,
    -0.000388145,
    -0.000522137,
    -0.000661850,
    -0.000806808,
    -0.000956535,
    -0.001111031,
    -0.001269817,
    -0.001432419,
    -0.001597881,
    -0.001766082,
    -0.001937389,
    -0.002110004,
    -0.002283096,
    // C[160..192]
    -0.002457142,
    -0.002630711,
    -0.002803326,
    -0.002974033,
    -0.003141880,
    -0.003300866,
    -0.003467083,
    -0.003622532,
    -0.003771782,
    -0.003914356,
    -0.004048824,
    -0.004174709,
    -0.004290581,
    -0.004395962,
    -0.004489899,
    -0.004570484,
    -0.004638195,
    -0.004691124,
    -0.004728317,
    -0.004748821,
    -0.004752159,
    -0.004737377,
    -0.004703045,
    -0.004649162,
    -0.004573822,
    -0.004477024,
    -0.004357815,
    -0.004215240,
    -0.004049301,
    -0.003858566,
    -0.003643036,
    -0.003401756,
    // C[192..224]
    0.003134727,
    0.002841473,
    0.002521515,
    0.002174854,
    0.001800537,
    0.001399517,
    0.000971317,
    0.000515938,
    0.000033379,
    -0.000475883,
    -0.001011848,
    -0.001573563,
    -0.002161503,
    -0.002774239,
    -0.003411293,
    -0.004072189,
    -0.004756451,
    -0.005462170,
    -0.006189346,
    -0.006937027,
    -0.007703304,
    -0.008487225,
    -0.009287834,
    -0.010103703,
    -0.010933399,
    -0.011775017,
    -0.012627602,
    -0.013489246,
    -0.014358521,
    -0.015233517,
    -0.016112804,
    -0.016994476,
    // C[224..256]
    -0.017876148,
    -0.018756866,
    -0.019634247,
    -0.020506859,
    -0.021372318,
    -0.022228718,
    -0.023074150,
    -0.023907185,
    -0.024725437,
    -0.025527000,
    -0.026310921,
    -0.027073860,
    -0.027815342,
    -0.028532982,
    -0.029224873,
    -0.029890060,
    -0.030526638,
    -0.031132698,
    -0.031706810,
    -0.032248020,
    -0.032754898,
    -0.033225536,
    -0.033659935,
    -0.034055710,
    -0.034412861,
    -0.034730434,
    -0.035007000,
    -0.035242081,
    -0.035435200,
    -0.035586357,
    -0.035694122,
    -0.035758972,
    // C[256..288] — first half of the mirrored second half.
    // C[256] is the spec's unique peak. C[256+i] for i = 1..255 is
    // derived by the cosine-modulated-prototype symmetry
    //   C[256+i] = -C[256-i]  if (256-i) mod 64 != 0
    //   C[256+i] = +C[256-i]  if (256-i) mod 64 == 0  (i = 64, 128, 192).
    0.035780907,
    0.035758972,
    0.035694122,
    0.035586357,
    0.035435200,
    0.035242081,
    0.035007000,
    0.034730434,
    0.034412861,
    0.034055710,
    0.033659935,
    0.033225536,
    0.032754898,
    0.032248020,
    0.031706810,
    0.031132698,
    0.030526638,
    0.029890060,
    0.029224873,
    0.028532982,
    0.027815342,
    0.027073860,
    0.026310921,
    0.025527000,
    0.024725437,
    0.023907185,
    0.023074150,
    0.022228718,
    0.021372318,
    0.020506859,
    0.019634247,
    0.018756866,
    // C[288..320]
    0.017876148,
    0.016994476,
    0.016112804,
    0.015233517,
    0.014358521,
    0.013489246,
    0.012627602,
    0.011775017,
    0.010933399,
    0.010103703,
    0.009287834,
    0.008487225,
    0.007703304,
    0.006937027,
    0.006189346,
    0.005462170,
    0.004756451,
    0.004072189,
    0.003411293,
    0.002774239,
    0.002161503,
    0.001573563,
    0.001011848,
    0.000475883,
    -0.000033379,
    -0.000515938,
    -0.000971317,
    -0.001399517,
    -0.001800537,
    -0.002174854,
    -0.002521515,
    -0.002841473,
    // C[320..352]
    // C[320] = +C[192] (192 = 256 - 64; 64 is a multiple of 64 → same sign).
    0.003134727,
    0.003401756,
    0.003643036,
    0.003858566,
    0.004049301,
    0.004215240,
    0.004357815,
    0.004477024,
    0.004573822,
    0.004649162,
    0.004703045,
    0.004737377,
    0.004752159,
    0.004748821,
    0.004728317,
    0.004691124,
    0.004638195,
    0.004570484,
    0.004489899,
    0.004395962,
    0.004290581,
    0.004174709,
    0.004048824,
    0.003914356,
    0.003771782,
    0.003622532,
    0.003467083,
    0.003300866,
    0.003141880,
    0.002974033,
    0.002803326,
    0.002630711,
    // C[352..384]
    0.002457142,
    0.002283096,
    0.002110004,
    0.001937389,
    0.001766082,
    0.001597881,
    0.001432419,
    0.001269817,
    0.001111031,
    0.000956535,
    0.000806808,
    0.000661850,
    0.000522137,
    0.000388145,
    0.000259876,
    0.000137329,
    0.000021458,
    -0.000088215,
    -0.000191689,
    -0.000288486,
    -0.000378609,
    -0.000462532,
    -0.000539303,
    -0.000610352,
    -0.000674248,
    -0.000731945,
    -0.000783920,
    -0.000829220,
    -0.000868797,
    -0.000902653,
    -0.000930786,
    -0.000953674,
    // C[384..416]
    // C[384] = +C[128] (128 is a multiple of 64 → same sign).
    0.000971317,
    0.000983715,
    0.000991821,
    0.000995159,
    0.000994205,
    0.000989437,
    0.000980854,
    0.000968933,
    0.000954151,
    0.000935555,
    0.000915051,
    0.000891685,
    0.000866413,
    0.000838757,
    0.000809669,
    0.000779152,
    0.000747204,
    0.000714302,
    0.000680923,
    0.000646591,
    0.000611782,
    0.000576973,
    0.000542164,
    0.000507355,
    0.000472546,
    0.000438213,
    0.000404358,
    0.000371456,
    0.000339031,
    0.000307560,
    0.000277042,
    0.000247478,
    // C[416..448]
    0.000218868,
    0.000191212,
    0.000165462,
    0.000140190,
    0.000116348,
    0.000093937,
    0.000072956,
    0.000052929,
    0.000034332,
    0.000017166,
    0.000000954,
    -0.000013828,
    -0.000027180,
    -0.000039577,
    -0.000050545,
    -0.000060558,
    -0.000069618,
    -0.000077724,
    -0.000084400,
    -0.000090122,
    -0.000095367,
    -0.000099182,
    -0.000102520,
    -0.000105381,
    -0.000106812,
    -0.000108242,
    -0.000108719,
    -0.000108719,
    -0.000108242,
    -0.000107288,
    -0.000105858,
    -0.000103951,
    // C[448..480]
    // C[448] = +C[64] (64 is a multiple of 64 → same sign).
    0.000101566,
    0.000099182,
    0.000096321,
    0.000093460,
    0.000090599,
    0.000087261,
    0.000083923,
    0.000080585,
    0.000076771,
    0.000073433,
    0.000070095,
    0.000066280,
    0.000062943,
    0.000059605,
    0.000055790,
    0.000052929,
    0.000049591,
    0.000046253,
    0.000043392,
    0.000040531,
    0.000037670,
    0.000034809,
    0.000032425,
    0.000030041,
    0.000027657,
    0.000025272,
    0.000023365,
    0.000021458,
    0.000019550,
    0.000018120,
    0.000016689,
    0.000014782,
    // C[480..512]
    0.000013828,
    0.000012398,
    0.000011444,
    0.000010014,
    0.000009060,
    0.000008106,
    0.000007629,
    0.000006676,
    0.000006199,
    0.000005245,
    0.000004768,
    0.000004292,
    0.000003815,
    0.000003338,
    0.000003338,
    0.000002861,
    0.000002384,
    0.000002384,
    0.000001907,
    0.000001907,
    0.000001431,
    0.000001431,
    0.000000954,
    0.000000954,
    0.000000954,
    0.000000954,
    0.000000477,
    0.000000477,
    0.000000477,
    0.000000477,
    0.000000477,
    0.000000477,
];

#[cfg(test)]
// The tests recompute every reference value directly from §C.1.3 /
// Figure C.4 formulas, written in their natural `for i in 0..N` index-
// driven shape (the index variable is part of the spec formula, not just
// a buffer iterator), so the explicit loops mirror the spec text more
// faithfully than iterator chains.
#[allow(clippy::needless_range_loop)]
mod tests_inner {
    use super::*;
    use crate::synth::{synth_row, SynthState};

    const EPS_F64: f64 = 1e-12;

    // ----- C[] table cross-checks -----

    #[test]
    fn c_table_length_is_512() {
        assert_eq!(C_TABLE.len(), X_LEN);
        assert_eq!(X_LEN, 512);
    }

    #[test]
    fn c_table_boundary_values_match_spec() {
        // Spot-check the table boundaries against the literal §C.1.3
        // Table C.1 entries:
        //   C[0]    =  0.000000000     (origin)
        //   C[1]    = -0.000000477     (smallest non-zero magnitude)
        //   C[255]  = -0.035758972     (negative peak, last before flip)
        //   C[256]  =  0.035780907     (positive peak, the global maximum)
        //   C[257]  =  0.035758972     (mirror of C[255], opposite sign)
        //   C[511]  =  0.000000477     (tail)
        assert_eq!(C_TABLE[0], 0.000000000);
        assert_eq!(C_TABLE[1], -0.000000477);
        assert_eq!(C_TABLE[255], -0.035758972);
        assert_eq!(C_TABLE[256], 0.035780907);
        assert_eq!(C_TABLE[257], 0.035758972);
        assert_eq!(C_TABLE[511], 0.000000477);
    }

    #[test]
    fn c_table_c256_is_global_maximum() {
        // Per the spec design C[256] = 0.035780907 is the unique global
        // maximum of the analysis prototype window.
        let mut max = f64::NEG_INFINITY;
        let mut argmax = 0usize;
        for (i, &v) in C_TABLE.iter().enumerate() {
            if v > max {
                max = v;
                argmax = i;
            }
        }
        assert_eq!(argmax, 256, "C[] max at index {argmax} = {max}");
        assert_eq!(max, 0.035780907);
    }

    #[test]
    fn c_table_satisfies_polyphase_symmetry() {
        // The cosine-modulated-prototype symmetry that defines the
        // second-half values (derived from the §C.1.3 polyphase
        // construction; verified on every spot-check pair in the
        // first-half entries):
        //   C[512 - i] = +C[i]   if i mod 64 == 0
        //   C[512 - i] = -C[i]   otherwise
        for i in 1..256 {
            let mirror = C_TABLE[X_LEN - i];
            let expect = if i % 64 == 0 { C_TABLE[i] } else { -C_TABLE[i] };
            assert!(
                (mirror - expect).abs() < EPS_F64,
                "C[{}] = {}, expected {} (= {} · C[{i}])",
                X_LEN - i,
                mirror,
                expect,
                if i % 64 == 0 { "+" } else { "-" }
            );
        }
    }

    // ----- §C.1.3 M[i,k] = cos((2i+1)·(k-16)·π/64) matrixing coefficient -----

    #[test]
    fn m_coefficient_k16_is_unity() {
        // M[i,16] = cos(0) = 1 for every i — the (k - 16) factor zeros
        // the argument, the matrix's identity column.
        for i in 0..32 {
            let v = m_coefficient(i, 16);
            assert!((v - 1.0).abs() < EPS_F64, "M[{i},16] = {v} expected 1.0");
        }
    }

    #[test]
    fn m_coefficient_i0_specific_values() {
        // i = 0 → (2·0 + 1) = 1, so M[0,k] = cos((k - 16)·π/64).
        for k in 0..64 {
            let expect = (((k as isize - 16) as f64) * core::f64::consts::PI / 64.0).cos();
            let got = m_coefficient(0, k);
            assert!(
                (got - expect).abs() < EPS_F64,
                "M[0,{k}] = {got} expected {expect}"
            );
        }
    }

    #[test]
    fn m_coefficient_matches_spec_formula_full_range() {
        // Re-evaluate the §C.1.3 formula directly and compare every
        // (i, k) cell.
        for i in 0..32 {
            for k in 0..64 {
                let arg = ((2 * i + 1) as f64) * ((k as isize - 16) as f64) * core::f64::consts::PI
                    / 64.0;
                let expect = arg.cos();
                let got = m_coefficient(i, k);
                assert!(
                    (got - expect).abs() < EPS_F64,
                    "M[{i},{k}] = {got} expected {expect}"
                );
            }
        }
    }

    // ----- Figure C.4 input shift register -----

    #[test]
    fn analysis_state_default_is_all_zero() {
        let s = AnalysisState::new();
        for &v in s.x.iter() {
            assert_eq!(v, 0.0);
        }
        let d = AnalysisState::default();
        for &v in d.x.iter() {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn analyze_row_zero_input_yields_zero_output_at_start() {
        // Zero shift register + zero PCM ⇒ zero subband samples
        // (Y[] = 0 ⇒ S[] = 0). Sanity check before non-trivial inputs.
        let mut state = AnalysisState::new();
        let zero = [0.0f64; 32];
        let s = analyze_row(&zero, &mut state);
        for (i, &v) in s.iter().enumerate() {
            assert_eq!(v, 0.0, "zero analyze_row out[{i}] = {v}");
        }
    }

    #[test]
    fn analyze_row_input_shift_places_pcm_in_low_indices() {
        // After one call with pcm[j] = (j+1) (distinct per slot), the
        // shift register's low 32 entries must hold the input with the
        // §C.1.3 "downto 0" ordering: pcm[0] → X[31], pcm[31] → X[0].
        // The remaining X[32..512] is still zero (no prior shifts).
        let mut state = AnalysisState::new();
        let mut pcm = [0.0f64; 32];
        for (j, slot) in pcm.iter_mut().enumerate() {
            *slot = (j + 1) as f64;
        }
        let _ = analyze_row(&pcm, &mut state);
        for j in 0..32 {
            assert_eq!(
                state.x(31 - j),
                pcm[j],
                "input shift: X[{}] expected pcm[{j}] = {}",
                31 - j,
                pcm[j]
            );
        }
        for i in 32..X_LEN {
            assert_eq!(state.x(i), 0.0, "tail must remain zero at i={i}");
        }
    }

    #[test]
    fn analyze_row_second_call_shifts_previous_block_to_positions_32_to_63() {
        // After two calls (first with block A, second with block B),
        // X[0..32] is B (newest), X[32..64] is A (one block older),
        // X[64..512] is zero (no third block yet).
        let mut state = AnalysisState::new();
        let mut a = [0.0f64; 32];
        let mut b = [0.0f64; 32];
        for j in 0..32 {
            a[j] = (j + 1) as f64;
            b[j] = -((j + 1) as f64) * 0.5;
        }
        let _ = analyze_row(&a, &mut state);
        let _ = analyze_row(&b, &mut state);
        // Newest block (B) at indices 0..32 (b[0] → X[31], b[31] → X[0]).
        for j in 0..32 {
            assert_eq!(state.x(31 - j), b[j], "second call X[{}] = b[{j}]", 31 - j);
        }
        // Previous block (A) at indices 32..64 (a[0] → X[63], a[31] → X[32]).
        for j in 0..32 {
            assert_eq!(state.x(63 - j), a[j], "second call X[{}] = a[{j}]", 63 - j);
        }
        for i in 64..X_LEN {
            assert_eq!(state.x(i), 0.0, "tail still zero at i={i}");
        }
    }

    #[test]
    fn analyze_row_is_linear_in_input() {
        // analyze_row over the same state is a linear map of the input
        // PCM block. (More precisely, analyze_row updates the *state*
        // by shifting+injecting, then runs a linear map; if the state
        // is the same on entry, the input → output portion is linear.)
        let mut s1 = AnalysisState::new();
        let mut s2 = AnalysisState::new();
        let mut sc = AnalysisState::new();
        let mut a = [0.0f64; 32];
        let mut b = [0.0f64; 32];
        for j in 0..32 {
            a[j] = (j as f64 * 0.13).sin();
            b[j] = (j as f64 * 0.27 + 0.4).cos();
        }
        let alpha = 0.6;
        let beta = -1.3;
        let mut combined = [0.0f64; 32];
        for j in 0..32 {
            combined[j] = alpha * a[j] + beta * b[j];
        }
        let out_a = analyze_row(&a, &mut s1);
        let out_b = analyze_row(&b, &mut s2);
        let out_c = analyze_row(&combined, &mut sc);
        for j in 0..32 {
            let expect = alpha * out_a[j] + beta * out_b[j];
            assert!(
                (out_c[j] - expect).abs() < EPS_F64 * 64.0,
                "linearity j={j}: {} vs {}",
                out_c[j],
                expect
            );
        }
    }

    // ----- TDAC-style round-trip identities -----
    //
    // The MP3 polyphase analysis / synthesis pair is a *near-perfect*-
    // reconstruction pseudo-quadrature mirror filterbank. The two
    // characteristic round-trip identities that the spec design provides
    // (and that this module's analysis primitives must satisfy if their
    // window table and matrix kernel are correct) are:
    //
    // (1) **PCM round-trip with prototype group delay 481**. For any
    //     broadband PCM signal `x[n]`, the steady-state analysis-then-
    //     synthesis output `y[n]` matches the input delayed by 481
    //     samples: `y[n] ≈ x[n - 481]`. The spec's prototype is a
    //     designed near-PR filter, not paraunitary, so equality holds
    //     to within the prototype's design ripple of ≈ 1×10⁻⁴ RMS
    //     (≈ -80 dB, see §C.1.3 prose on prototype quality).
    //
    // (2) **Per-subband DC-tone subband round-trip is exactly
    //     cyclostationary in steady state**. Driving synthesis with a
    //     constant unit input in one subband (S[sb0] = 1, others 0)
    //     and analysing the resulting PCM yields, after the bank's
    //     settling delay, a steady-state recovered coefficient
    //     `K_sb0` that does *not* vary from row to row. The
    //     row-to-row ripple is **zero to float precision** (numerical
    //     noise only, < 1×10⁻¹²) — a strict spec-derivable invariant
    //     of the bank's cosine-modulated matrix structure.
    //
    // (1) bounds the absolute reconstruction error; (2) bounds the
    // *consistency* of the recovered subband-domain coefficients. Both
    // tests exercise the same C[] table and M[i,k] formula in
    // different aggregation modes, and a single arithmetic error in
    // either fails both with high amplitude.

    #[test]
    fn pcm_round_trip_steady_state_delay_481_rms_under_1e_minus_4() {
        // Build a broadband PCM signal (mixture of well-separated tones),
        // pass it through analyze → synth row-by-row, and check that
        // the steady-state output matches the input delayed by 481
        // samples (the analysis + synthesis prototype group delay,
        // 512 prototype taps − 32 + 1 phase offset).
        const N_ROWS: usize = 200;
        const SETTLE_SAMPLES: usize = 1024; // > 481 + several rows of fill
        const DELAY: usize = 481;

        let total = N_ROWS * NUM_SUBBANDS;
        let mut pcm_in = vec![0.0f64; total];
        for (n, slot) in pcm_in.iter_mut().enumerate() {
            let t = n as f64;
            *slot = 0.4 * (t * 0.02).sin() + 0.3 * (t * 0.13 + 1.0).cos() + 0.2 * (t * 0.37).sin();
        }

        let mut ana = AnalysisState::new();
        let mut syn = SynthState::new();
        let mut pcm_out = vec![0.0f64; total];
        for r in 0..N_ROWS {
            let mut row = [0.0f64; NUM_SUBBANDS];
            for (j, slot) in row.iter_mut().enumerate() {
                *slot = pcm_in[r * NUM_SUBBANDS + j];
            }
            let s = analyze_row(&row, &mut ana);
            let recov = synth_row(&s, &mut syn);
            for (j, v) in recov.iter().enumerate() {
                pcm_out[r * NUM_SUBBANDS + j] = *v;
            }
        }

        // RMS of (pcm_out[n] - pcm_in[n - DELAY]) for n in the settled
        // window. The spec's near-PR ripple is well below 1e-4 (the
        // measured value on this signal is ~3e-5).
        let mut sq = 0.0f64;
        let mut max_abs = 0.0f64;
        let mut count = 0usize;
        for n in SETTLE_SAMPLES..total {
            let d = pcm_out[n] - pcm_in[n - DELAY];
            sq += d * d;
            if d.abs() > max_abs {
                max_abs = d.abs();
            }
            count += 1;
        }
        let rms = (sq / count as f64).sqrt();
        assert!(
            rms < 1e-4,
            "PCM round-trip RMS at delay {DELAY} = {rms} (max {max_abs}); expected < 1e-4"
        );
    }

    #[test]
    fn synth_then_analyze_per_subband_dc_tone_round_trip_ripple_under_1e_minus_12() {
        // For each subband sb0 in 0..32, drive synth_row with a constant
        // unit input only in that subband (S[sb0] = 1, others 0), for
        // ROWS = 32 rows. The synthesis emits 32 PCM samples per row.
        // Pipe those PCM samples one row at a time through analyze_row,
        // and read the per-row analysis output's sb0 component back.
        //
        // After the bank's combined settling delay — 16 rows for synth
        // to fully fill its 1024-element V[] shift register (which is
        // fed 64 values per row) and another 16 rows for the analysis
        // 512-element X[] shift register to fully fill on the synth
        // output — the recovered coefficient at subband sb0 is a
        // constant `K_sb0` — *exact* row-to-row constancy at float
        // precision is a spec-derivable invariant of the cosine-
        // modulated polyphase pair (the bank is exactly cyclostationary
        // on per-subband DC drive). We assert the row-to-row RMS
        // deviation around its mean is below 1e-12, the float-precision
        // bar promised by the round.
        const ROWS: usize = 48;
        const SETTLE: usize = 32;
        let mut max_rms = 0.0f64;
        for sb0 in 0..NUM_SUBBANDS {
            let mut syn = SynthState::new();
            let mut ana = AnalysisState::new();
            let mut s_in = [0.0f64; NUM_SUBBANDS];
            s_in[sb0] = 1.0;

            let mut recovered_row = [0.0f64; ROWS];
            for r in 0..ROWS {
                let pcm = synth_row(&s_in, &mut syn);
                let s_out = analyze_row(&pcm, &mut ana);
                recovered_row[r] = s_out[sb0];
            }

            // RMS of the settled sb0 column around its mean — i.e.
            // the ripple of the analysed steady-state coefficient.
            let mut mean = 0.0f64;
            for r in SETTLE..ROWS {
                mean += recovered_row[r];
            }
            mean /= (ROWS - SETTLE) as f64;
            let mut sq_sum = 0.0f64;
            for r in SETTLE..ROWS {
                let d = recovered_row[r] - mean;
                sq_sum += d * d;
            }
            let rms = (sq_sum / (ROWS - SETTLE) as f64).sqrt();
            assert!(
                rms < 1e-12,
                "sb0={sb0} settled-row RMS ripple = {rms} (mean = {mean}); expected < 1e-12"
            );
            if rms > max_rms {
                max_rms = rms;
            }
        }
        assert!(max_rms < 1e-12, "max settled-row RMS ripple = {max_rms}");
    }

    // ----- analyze_granule shape contract -----

    #[test]
    fn analyze_granule_zero_input_yields_zero_subband_block() {
        let zero = [0.0f32; PCM_PER_GRANULE];
        let mut state = AnalysisState::new();
        let block = analyze_granule(&zero, &mut state);
        for sb in 0..NUM_SUBBANDS {
            for t in 0..SAMPLES_PER_SUBBAND {
                assert_eq!(block[sb][t], 0.0, "zero granule out[{sb}][{t}]");
            }
        }
    }

    #[test]
    fn analyze_granule_first_row_matches_analyze_row() {
        // Build a 576-sample PCM granule where the first row of 32 is
        // a distinct test pattern and the rest is zero; analyze_granule's
        // first time-column must match analyze_row over that single row
        // from a fresh state.
        let mut pcm = [0.0f32; PCM_PER_GRANULE];
        for j in 0..NUM_SUBBANDS {
            pcm[j] = (j + 1) as f32 * 0.01;
        }
        let mut state_g = AnalysisState::new();
        let block = analyze_granule(&pcm, &mut state_g);

        let mut row = [0.0f64; NUM_SUBBANDS];
        for j in 0..NUM_SUBBANDS {
            row[j] = f64::from(pcm[j]);
        }
        let mut state_r = AnalysisState::new();
        let out = analyze_row(&row, &mut state_r);

        for sb in 0..NUM_SUBBANDS {
            let g = block[sb][0];
            let r = out[sb] as f32;
            assert!(
                (g - r).abs() <= f32::EPSILON * 64.0_f32.max(r.abs() * 16.0),
                "sb={sb}: granule[{sb}][0] = {g} vs analyze_row[{sb}] = {r}"
            );
        }
    }
}
