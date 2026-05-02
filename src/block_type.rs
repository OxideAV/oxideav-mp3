//! Block-type / window-switching decision for the MP3 encoder.
//!
//! Per ISO/IEC 11172-3 §2.4.2.2 (block type / window switching) and
//! Annex C (psychoacoustic model) the encoder is allowed to switch from
//! the default 18-coefficient long-block MDCT to a 3×6-coefficient
//! short-block MDCT on transients to reduce pre-echo. The switch is not
//! free: the long→short and short→long boundaries must be bridged by
//! `start` (type 1) and `stop` (type 3) transition windows so the
//! overlap-add of the IMDCT matches up. The legal per-channel block
//! sequence is therefore:
//!
//! ```text
//!   long → long
//!   long → start  → short
//!   short → short
//!   short → stop  → long
//! ```
//!
//! which means a transient at granule N with N preceded by a long block
//! costs the previous granule (N-1) a `start` window — the encoder
//! needs a single granule of lookahead to decide retroactively. We
//! handle that here with a one-granule decision delay.
//!
//! ### Transient detector
//!
//! Inspired by ISO 11172-3 Annex C §C.1.1 (PE / "perceptual entropy"
//! growth) and the same energy-ratio scheme used by AC-3's spec-grade
//! transient detector (round 24 in this workspace): each 576-sample
//! granule is split into three 192-sample sub-frames, and a sub-frame
//! is flagged transient when its energy exceeds the long-term smoothed
//! energy by more than [`TRANSIENT_RATIO`]. A granule is "transient"
//! if any of its sub-frames trips. To avoid false positives on quiet
//! noise we also require an absolute energy floor.

/// Block type as encoded in the side-info `block_type` field (only
/// meaningful when `window_switching_flag = 1`; 0 implies long).
///
/// The numeric values match the bitstream:
///   * 0 = long / normal
///   * 1 = start  (long → short transition)
///   * 2 = short  (3 × 192-sample MDCTs)
///   * 3 = stop   (short → long transition)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BlockType {
    #[default]
    Long = 0,
    Start = 1,
    Short = 2,
    Stop = 3,
}

impl BlockType {
    /// `true` when the bitstream's `window_switching_flag` must be 1
    /// (i.e. anything other than long needs the side-info long form).
    pub fn needs_window_switching(self) -> bool {
        !matches!(self, BlockType::Long)
    }

    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

/// Energy ratio above which a sub-frame is flagged as a transient.
/// 4× ≈ +6 dB step, comfortably above sustained-tone modulation but
/// well below typical castanet / drum-stick onsets which jump 12-20 dB.
/// Same order of magnitude AC-3's spec-grade transient detector uses
/// (round 24's `TRANSIENT_THRESHOLD_DB ≈ 9 dB`).
const TRANSIENT_RATIO: f32 = 4.0;

/// Absolute energy floor — a sub-frame whose energy is below this is
/// considered silent and never flips the detector regardless of ratio.
/// Picks up onsets after near-silence (the worst pre-echo case) while
/// staying out of the way of dithered DC noise.
const SILENCE_FLOOR: f32 = 1.0e-6;

/// Smoothing factor for the long-term energy estimate. New sub-frame
/// energies are blended in with this weight: `e_avg = α · e_new + (1-α)
/// · e_avg`. A small α gives a slow tracker that responds to true onsets
/// rather than tracking the onset itself.
const SMOOTH_ALPHA: f32 = 0.25;

/// Per-channel transient detector. Tracks a slow exponential average
/// of sub-frame energy and reports `is_transient` when a new sub-frame
/// jumps above the average by more than [`TRANSIENT_RATIO`].
#[derive(Clone, Debug, Default)]
pub struct TransientDetector {
    /// Smoothed long-term energy estimate (E_avg in the docs above).
    /// Initialised to 0; the first non-silent sub-frame seeds it.
    avg: f32,
}

impl TransientDetector {
    pub fn new() -> Self {
        Self::default()
    }

    /// Inspect a 576-sample granule of PCM and return `true` when at
    /// least one of its three 192-sample sub-frames trips the
    /// energy-ratio test. The detector's smoothed-average state is
    /// advanced by every sub-frame regardless of the verdict.
    pub fn is_transient(&mut self, pcm: &[f32; 576]) -> bool {
        let mut hit = false;
        for w in 0..3 {
            let mut e = 0.0f32;
            for i in 0..192 {
                let v = pcm[w * 192 + i];
                e += v * v;
            }
            // Per-sample average so detector behaviour is invariant to
            // sub-frame length (matters if we ever switch granule shape).
            e /= 192.0;
            // Compare *before* updating the average so the very sub-frame
            // carrying the onset doesn't immediately mask itself.
            if e > SILENCE_FLOOR && self.avg > 0.0 && e > self.avg * TRANSIENT_RATIO {
                hit = true;
            }
            // Smooth update.
            self.avg = SMOOTH_ALPHA * e + (1.0 - SMOOTH_ALPHA) * self.avg;
        }
        hit
    }
}

/// Per-channel block-type decision machine. Owns the previous granule's
/// chosen block type so the next call can enforce the legal transitions
/// listed in the module docstring.
///
/// Decision call order — `decide(prev_long, this_transient, next_transient)`:
///
/// * `prev_was_short` — what we emitted for the previous granule.
/// * `this_transient` — detector verdict for the current granule.
/// * `next_transient` — detector verdict for the granule that follows.
///
/// We need both verdicts to pick the right window for the *current*
/// granule because a `start` window has to *precede* a short. The
/// encoder buffers one granule of lookahead so this lookahead read is
/// always available.
#[derive(Clone, Debug, Default)]
pub struct BlockTypeMachine {
    /// Previous granule's block type. Initialised to Long.
    prev: BlockType,
}

impl BlockTypeMachine {
    pub fn new() -> Self {
        Self {
            prev: BlockType::Long,
        }
    }

    /// Reset to initial state (used between independent encode runs).
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.prev = BlockType::Long;
    }

    /// Pick the block type for the current granule, given the detector
    /// verdicts for *this* and the *next* granule, and update internal
    /// state so the next call sees the latest choice.
    ///
    /// Truth table (cur ≡ this granule, nxt ≡ next granule):
    ///
    /// | prev    | cur  | nxt  | -> emit |
    /// |---------|------|------|---------|
    /// | Long    | T    | _    | Start   |
    /// | Long    | F    | T    | Start   |
    /// | Long    | F    | F    | Long    |
    /// | Start   | _    | _    | Short   |
    /// | Short   | T    | _    | Short   |
    /// | Short   | F    | _    | Stop    |
    /// | Stop    | T    | _    | Start   |
    /// | Stop    | F    | T    | Start   |
    /// | Stop    | F    | F    | Long    |
    pub fn decide(&mut self, this_transient: bool, next_transient: bool) -> BlockType {
        let cur = match self.prev {
            BlockType::Long | BlockType::Stop => {
                // From long-class: a transient now or imminent gets a
                // start window so the next granule can be short.
                if this_transient || next_transient {
                    BlockType::Start
                } else {
                    BlockType::Long
                }
            }
            BlockType::Start => {
                // After start the next granule MUST be short to honour
                // the IMDCT overlap-add envelope.
                BlockType::Short
            }
            BlockType::Short => {
                // Stay in short while the transient region persists,
                // then bridge back via stop.
                if this_transient {
                    BlockType::Short
                } else {
                    BlockType::Stop
                }
            }
        };
        self.prev = cur;
        cur
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silence_never_trips_detector() {
        let mut det = TransientDetector::new();
        let pcm = [0.0f32; 576];
        for _ in 0..16 {
            assert!(!det.is_transient(&pcm));
        }
    }

    #[test]
    fn step_onset_after_silence_trips_detector() {
        let mut det = TransientDetector::new();
        // Seed with low-level noise so the average is non-zero.
        let mut warm = [0.0f32; 576];
        for (i, v) in warm.iter_mut().enumerate() {
            *v = ((i as f32) * 0.001).sin() * 0.001; // ~1e-3 amplitude
        }
        for _ in 0..6 {
            det.is_transient(&warm);
        }
        // Sharp onset: full-scale step in the middle sub-frame.
        let mut step = [0.0f32; 576];
        for i in 192..384 {
            step[i] = 0.5;
        }
        assert!(
            det.is_transient(&step),
            "expected step onset to trip transient detector"
        );
    }

    #[test]
    fn sustained_tone_does_not_trip_detector_repeatedly() {
        let mut det = TransientDetector::new();
        // Steady sine — energy is constant per sub-frame after warmup.
        let mut tone = [0.0f32; 576];
        let two_pi = 2.0 * std::f32::consts::PI;
        for (i, v) in tone.iter_mut().enumerate() {
            *v = (two_pi * 440.0 * i as f32 / 44_100.0).sin() * 0.3;
        }
        // Run a few warmup granules then count transient hits in steady state.
        for _ in 0..6 {
            det.is_transient(&tone);
        }
        let mut hits = 0;
        for _ in 0..32 {
            if det.is_transient(&tone) {
                hits += 1;
            }
        }
        assert!(hits == 0, "steady tone tripped detector {hits} times");
    }

    #[test]
    fn block_machine_long_chain_stays_long() {
        let mut m = BlockTypeMachine::new();
        for _ in 0..10 {
            assert_eq!(m.decide(false, false), BlockType::Long);
        }
    }

    #[test]
    fn block_machine_transient_emits_start_then_short() {
        let mut m = BlockTypeMachine::new();
        // First granule sees a transient about to happen on the *next*
        // granule -> emit start now.
        assert_eq!(m.decide(false, true), BlockType::Start);
        // Then the transient granule itself -> short (forced by start).
        assert_eq!(m.decide(true, false), BlockType::Short);
        // Then transient subsides -> stop bridges back.
        assert_eq!(m.decide(false, false), BlockType::Stop);
        // Then long again.
        assert_eq!(m.decide(false, false), BlockType::Long);
    }

    #[test]
    fn block_machine_consecutive_transients_stay_in_short() {
        let mut m = BlockTypeMachine::new();
        assert_eq!(m.decide(false, true), BlockType::Start);
        assert_eq!(m.decide(true, true), BlockType::Short);
        assert_eq!(m.decide(true, true), BlockType::Short);
        assert_eq!(m.decide(true, false), BlockType::Short);
        assert_eq!(m.decide(false, false), BlockType::Stop);
        assert_eq!(m.decide(false, false), BlockType::Long);
    }

    #[test]
    fn block_machine_transient_immediately_after_stop_re_enters_start() {
        let mut m = BlockTypeMachine::new();
        // Long → Start → Short → Stop, then a new transient.
        m.decide(false, true);
        m.decide(true, false);
        m.decide(false, false); // -> Stop
                                // Now another transient: stop is treated as long-class for
                                // sequencing purposes, so we re-emit start.
        let bt = m.decide(false, true);
        assert_eq!(bt, BlockType::Start);
    }
}
