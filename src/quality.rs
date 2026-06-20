//! Named psychoacoustic **quality presets** for the Layer III encoder.
//!
//! The streaming encoder ([`crate::stream_encoder::Mp3Encoder`]) exposes
//! the §C.1.5 / Annex D perceptual machinery as a set of independent
//! opt-in toggles: the §C.1.5.4.3 outer loop, the per-band Annex D
//! threshold-in-quiet vector, the §C.1.5.3.2.1 Model 2 analysis (driving
//! both the per-band masking threshold and the `pe > 1800` window-switch
//! decision), and the §C.1.5.2 block-type state machine. Each toggle is
//! individually documented and individually testable, but a front-end
//! that only wants "good MP3, please" has to know which combination to
//! arm and in what order.
//!
//! A [`QualityPreset`] is that combination expressed as one named knob.
//! Every preset maps to a small bundle of the existing toggles plus a
//! single perceptual **threshold offset** (§D.1 Step 3 `offset_db`, see
//! [`crate::psy::XminThresholds::threshold_in_quiet_with_offset_db`]):
//! a *tighter* (more-negative) offset spends more bits per band for a
//! higher SNR-vs-LTq, a *looser* (less-negative / positive) offset
//! spends fewer. The presets walk that knob from transparency-leaning to
//! bit-thrifty while keeping every value spec-grounded — no preset
//! invents a threshold the spec doesn't already sanction.
//!
//! ## What a preset arms
//!
//! At the three Annex D Model 2 sampling rates (32 / 44.1 / 48 kHz) the
//! richer presets arm the full signal-dependent path: the per-granule
//! Model 2 analysis installs a content-driven `xmin(sb)` and the
//! Model-2-driven §C.1.5.2 block-type scheduler switches to short blocks
//! on `pe > 1800` granules. At every other rate (the MPEG-2 LSF and
//! MPEG-2.5 extension rates, which have no staged Annex D
//! calculation-partition tables) a preset **falls back** to the
//! signal-independent per-band threshold-in-quiet vector translated by
//! the preset's `offset_db`, so a preset is usable at every rate the
//! encoder supports — it simply uses the best perceptual model the
//! staged tables allow for that rate.
//!
//! The preset never changes the channel mode, the target bitrate, or the
//! stereo-coupling decision — those are the caller's to pick when
//! constructing the encoder. A preset only governs *how aggressively the
//! quantization noise is shaped under the perceptual model* for whatever
//! transport the caller already chose.

/// A named psychoacoustic quality level for the Layer III encoder.
///
/// Ordered from most bits / highest fidelity ([`Self::Transparent`]) to
/// fewest bits / fastest convergence ([`Self::Fast`]). The ordering is
/// the §D.1 Step 3 threshold-offset ordering: a higher preset uses a
/// more-negative `offset_db`, which tightens the per-band masking
/// threshold so the outer loop drives quantization noise further below
/// the masking curve.
///
/// Apply a preset with
/// [`crate::stream_encoder::Mp3Encoder::with_quality_preset`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QualityPreset {
    /// Transparency-leaning: the tightest staged threshold (§D.1 Step 3
    /// `offset_db = -24 dB`, two §D.1 steps below the spec's own
    /// high-bitrate `-12 dB`). Arms the Model 2 per-band threshold and
    /// the Model-2-driven block-type scheduler at the Annex D rates.
    /// This is the preset to pick when the bit budget is generous and
    /// near-transparency is the goal.
    Transparent,
    /// High quality: the spec's own high-bitrate threshold (§D.1 Step 3
    /// `offset_db = -12 dB`, the `>= 96 kbit/s per channel` branch). Arms
    /// the Model 2 per-band threshold and the Model-2-driven block-type
    /// scheduler at the Annex D rates. The recommended **default**
    /// perceptual preset.
    High,
    /// Standard quality: the spec's low-bitrate threshold (§D.1 Step 3
    /// `offset_db = 0 dB`, the `< 96 kbit/s per channel` branch). Arms
    /// the Model 2 per-band threshold at the Annex D rates but leaves the
    /// block type to the caller (no automatic short-block switching), so
    /// a steady-state signal converges quickly.
    Standard,
    /// Fast / bit-thrifty: a loosened threshold (§D.1 Step 3
    /// `offset_db = +6 dB`, above the spec's low-bitrate branch) on the
    /// signal-independent threshold-in-quiet vector, with no Model 2
    /// analysis armed at all. The cheapest perceptual shaping — the outer
    /// loop still shapes noise toward the LTq bowl, but no per-granule
    /// FFT analysis runs.
    Fast,
}

/// The spec-grounded parameters a [`QualityPreset`] expands to.
///
/// This is the internal lowering of a preset into the existing encoder
/// toggles; it is surfaced publicly so a caller can inspect exactly what
/// a preset will arm before applying it (and so the integration tests can
/// assert the mapping without re-deriving it).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QualityPresetParams {
    /// §D.1 Step 3 threshold offset in dB applied to the per-band
    /// threshold-in-quiet bowl. More negative ⇒ tighter ⇒ more bits.
    pub threshold_offset_db: f64,
    /// Arm the §C.1.5.3.2.1 Model 2 per-band masking analysis when the
    /// sampling rate is one of the staged Annex D rates (32 / 44.1 /
    /// 48 kHz). At other rates the per-band threshold-in-quiet vector
    /// (translated by `threshold_offset_db`) is used instead.
    pub model2_threshold: bool,
    /// Arm the §C.1.5.2 Model-2-driven block-type scheduler (the
    /// `pe > 1800` short-block switch). Only meaningful when
    /// `model2_threshold` is set and the rate is a staged Annex D rate;
    /// ignored otherwise.
    pub model2_block_type: bool,
}

impl QualityPreset {
    /// Lower the preset to its spec-grounded encoder parameters.
    ///
    /// The mapping is fixed and rate-independent; the encoder decides at
    /// apply-time whether the Model 2 flags can actually be honoured for
    /// the configured sampling rate (see
    /// [`crate::stream_encoder::Mp3Encoder::with_quality_preset`]).
    #[must_use]
    pub const fn params(self) -> QualityPresetParams {
        match self {
            Self::Transparent => QualityPresetParams {
                threshold_offset_db: -24.0,
                model2_threshold: true,
                model2_block_type: true,
            },
            Self::High => QualityPresetParams {
                threshold_offset_db: -12.0,
                model2_threshold: true,
                model2_block_type: true,
            },
            Self::Standard => QualityPresetParams {
                threshold_offset_db: 0.0,
                model2_threshold: true,
                model2_block_type: false,
            },
            Self::Fast => QualityPresetParams {
                threshold_offset_db: 6.0,
                model2_threshold: false,
                model2_block_type: false,
            },
        }
    }

    /// The recommended default perceptual preset ([`Self::High`]).
    #[must_use]
    pub const fn default_preset() -> Self {
        Self::High
    }
}

impl Default for QualityPreset {
    /// [`QualityPreset::High`] — the recommended default perceptual
    /// preset.
    fn default() -> Self {
        Self::default_preset()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threshold_offset_is_monotone_from_transparent_to_fast() {
        // Higher fidelity ⇒ more-negative (tighter) §D.1 Step 3 offset.
        let order = [
            QualityPreset::Transparent,
            QualityPreset::High,
            QualityPreset::Standard,
            QualityPreset::Fast,
        ];
        for pair in order.windows(2) {
            let hi = pair[0].params().threshold_offset_db;
            let lo = pair[1].params().threshold_offset_db;
            assert!(
                hi < lo,
                "{:?} ({hi} dB) must be tighter than {:?} ({lo} dB)",
                pair[0],
                pair[1]
            );
        }
    }

    #[test]
    fn high_matches_spec_high_bitrate_offset() {
        // §D.1 Step 3: -12 dB is the `>= 96 kbit/s per channel` branch.
        assert_eq!(QualityPreset::High.params().threshold_offset_db, -12.0);
    }

    #[test]
    fn standard_matches_spec_low_bitrate_offset() {
        // §D.1 Step 3: 0 dB is the `< 96 kbit/s per channel` branch.
        assert_eq!(QualityPreset::Standard.params().threshold_offset_db, 0.0);
    }

    #[test]
    fn model2_block_type_implies_model2_threshold() {
        // Arming the Model-2 block-type scheduler is only meaningful when
        // the Model-2 per-band threshold is also armed.
        for p in [
            QualityPreset::Transparent,
            QualityPreset::High,
            QualityPreset::Standard,
            QualityPreset::Fast,
        ] {
            let params = p.params();
            if params.model2_block_type {
                assert!(
                    params.model2_threshold,
                    "{p:?} arms model2_block_type without model2_threshold"
                );
            }
        }
    }

    #[test]
    fn default_is_high() {
        assert_eq!(QualityPreset::default(), QualityPreset::High);
        assert_eq!(QualityPreset::default_preset(), QualityPreset::High);
    }
}
