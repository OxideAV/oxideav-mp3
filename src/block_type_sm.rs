//! Layer III **block-type transition state machine** — the
//! encoder-side scheduler that turns a stream of per-granule
//! attack-detection flags (see [`crate::attack_detect`]) into a
//! syntactically valid sequence of [`BlockType`] decisions per
//! granule, honouring the §C.1.5.2 LONG → START → SHORT → STOP → LONG
//! transition geometry.
//!
//! # Geometry
//!
//! ISO/IEC 11172-3:1993 §2.4.3.4.10.3 defines four windowing modes
//! for the Layer III IMDCT:
//!
//! * **Long** (`block_type = 0`): one 36-point MDCT with the plain
//!   sine window, lapped 18 samples either side. Decoder splices to a
//!   previous Long via standard 18-sample overlap-add.
//! * **Start** (`block_type = 1`): one 36-point MDCT with an
//!   asymmetric window — long sine window over `i = 0..17`,
//!   pass-through over `i = 18..23`, short sine window over
//!   `i = 24..29`, zero over `i = 30..35`. This window's *tail* is
//!   shaped so it overlap-adds cleanly with a Short block's *head*.
//! * **Short** (`block_type = 2`): three independent 12-point MDCTs
//!   with three sine windows. Splices cleanly to another Short
//!   (head-to-tail match) and to a Start (head matches Start's tail)
//!   and to a Stop (tail matches Stop's head).
//! * **Stop** (`block_type = 3`, sometimes called "End"): one
//!   36-point MDCT with an asymmetric window — zero over `i = 0..5`,
//!   short window over `i = 6..11`, pass-through over `i = 12..17`,
//!   long window over `i = 18..35`. This window's *head* matches
//!   Short's *tail*.
//!
//! The decoder's overlap-add buffer therefore requires that the
//! transmitted sequence of block types form one of the patterns:
//!
//! * Long-only: `Long → Long → Long …`
//! * Burst: `Long → Start → (Short)+ → Stop → Long …`
//!
//! Any other pairing — e.g. `Long` immediately followed by `Short`,
//! or `Stop` followed by `Stop` — produces an aliasing discontinuity
//! that the decoder cannot algebraically cancel. The state machine
//! below guarantees that only valid pairings are emitted, regardless
//! of what attack-flag pattern the detector hands it.
//!
//! # Lookahead requirement
//!
//! Because `Start` must *precede* the first `Short` (the geometry is
//! anticipatory — the Start window prepares the overlap buffer for
//! Short's head), the scheduler needs **one granule of lookahead**:
//! at the time it commits granule `n`'s block type, it must already
//! know whether granule `n + 1` carries an attack. The caller is
//! responsible for buffering one granule of PCM such that
//! [`BlockTypeStateMachine::step`] receives both the current and
//! next granule's attack flags.
//!
//! At end-of-stream, when there is no next granule, the caller
//! passes `next_attack = false`. The state machine then emits a
//! geometrically valid termination (a trailing `Short` is followed
//! by `Stop`; a `Start` is followed by a `Short` and a `Stop` even
//! if `next_attack = false`, because once a `Start` has been
//! committed the geometry mandates the closing pair).
//!
//! # Design choice when an attack arrives without lookahead warning
//!
//! If `cur_attack = true` arrives while `prev = Long` AND
//! `next_attack = false` (i.e., the burst lands on the current
//! granule but the lookahead didn't see it because the burst is
//! entirely confined to one granule), the scheduler cannot retroactively
//! emit a `Start` for the *previous* granule (it's already been
//! committed). In that situation the scheduler emits `Long` for the
//! current granule too — accepting a one-granule miss in transient
//! sharpness rather than producing a geometrically invalid
//! `Long → Short` jump. This is conservative by design; a longer
//! lookahead buffer could reduce these misses but materially
//! complicates the encoder's latency contract.
//!
//! No external reference implementation was consulted while writing
//! this state machine. The geometry constraints are all derived from
//! §2.4.3.4.10.3's window shapes plus the overlap-add identity of
//! §2.4.3.4.10.4.

use crate::side_info::BlockType;

/// Stateful block-type scheduler. Holds the previously-emitted
/// [`BlockType`] (the carry that determines what's geometrically
/// allowable next).
#[derive(Debug, Clone, Copy)]
pub struct BlockTypeStateMachine {
    /// The block type emitted on the previous granule. Determines
    /// what the current granule is allowed to emit.
    prev: BlockType,
}

impl BlockTypeStateMachine {
    /// Construct a fresh scheduler in the steady-state `Long` mode.
    /// A new MP3 stream's first granule is always `Long` because
    /// nothing precedes it on the overlap-add buffer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            prev: BlockType::Long,
        }
    }

    /// Reset the scheduler to the initial `Long` state.
    pub fn reset(&mut self) {
        self.prev = BlockType::Long;
    }

    /// The block type the scheduler last emitted.
    #[must_use]
    pub fn prev_block_type(&self) -> BlockType {
        self.prev
    }

    /// Advance the scheduler by one granule.
    ///
    /// * `cur_attack` — `true` iff the detector flagged the
    ///   **current** granule as carrying an attack (informational; the
    ///   geometry constraint dominates).
    /// * `next_attack` — `true` iff the detector flagged the
    ///   **next** granule as carrying an attack. The scheduler uses
    ///   this to anticipate the need for a `Start` window. At
    ///   end-of-stream pass `false`.
    ///
    /// Returns the [`BlockType`] for the current granule. Updates the
    /// internal state in-place.
    ///
    /// This is the legacy entry point that never emits the mixed
    /// flag (every short emission is pure-short). The
    /// [`Self::step_with_mixed`] method extends the scheduler with a
    /// mixed-vs-pure-short signal on the same transition geometry.
    pub fn step(&mut self, cur_attack: bool, next_attack: bool) -> BlockType {
        self.step_with_mixed(cur_attack, next_attack, false).0
    }

    /// Advance the scheduler by one granule with an optional
    /// mixed-vs-pure-short preference.
    ///
    /// Identical to [`Self::step`] in transition geometry — the
    /// §C.1.5.2 `LONG → START → SHORT → STOP → LONG` envelope is
    /// unchanged — except that when the scheduler emits
    /// [`BlockType::Short`] **and** `prefer_mixed` is `true`, the
    /// returned `mixed_flag` companion bool is set. Callers wire
    /// this into the encoder's `mixed_block_flag` side-info field
    /// (§2.4.2.7) which selects the §2.4.3.4.10.3 mixed-block
    /// scalefactor-band layout (lowest 2 subbands long, the rest
    /// short).
    ///
    /// The mixed flag is **always** `false` for any non-Short
    /// emission: the §2.4.2.7 spec scopes `mixed_block_flag` to
    /// `block_type == 2`, so a Long / Start / End granule cannot
    /// carry it regardless of `prefer_mixed`. This keeps the
    /// scheduler honest with the wire syntax.
    ///
    /// Returns `(block_type, mixed_flag)` for the current granule.
    pub fn step_with_mixed(
        &mut self,
        cur_attack: bool,
        next_attack: bool,
        prefer_mixed: bool,
    ) -> (BlockType, bool) {
        let emitted = match self.prev {
            BlockType::Long => {
                // Long can transition to Start (preparing for a
                // Short on next granule) or stay Long. It cannot
                // jump directly to Short — that would require a
                // Start first.
                if next_attack {
                    BlockType::Start
                } else {
                    // cur_attack with no next_attack: the burst is
                    // entirely within the current granule; we have
                    // no Start lined up to splice into Short, so we
                    // emit Long. Documented in the module docs.
                    let _ = cur_attack;
                    BlockType::Long
                }
            }
            BlockType::Start => {
                // Start mandates Short next: the Start window's tail
                // shape only splices cleanly with a Short head.
                BlockType::Short
            }
            BlockType::Short => {
                // Short can stay Short (sustained transient train) or
                // transition to Stop (returning to Long). The choice
                // is driven by `cur_attack`: a still-bursty current
                // granule keeps Short; an attack-free current granule
                // tells us the burst has ended and we should
                // transition out via Stop.
                if cur_attack || next_attack {
                    BlockType::Short
                } else {
                    BlockType::End
                }
            }
            BlockType::End => {
                // Stop mandates Long next: the Stop window's tail is
                // the standard long-sine tail, so a Long next granule
                // splices into it cleanly. A re-entry into Short
                // immediately after Stop would require another Start
                // — i.e., we must spend at least one Long granule
                // before re-entering the burst geometry.
                BlockType::Long
            }
        };
        self.prev = emitted;
        // Per §2.4.2.7, `mixed_block_flag` is meaningful only when
        // `block_type == 2` (Short). Suppress the flag on every
        // other emission regardless of the caller's preference so
        // the scheduler never produces a syntactically invalid
        // (block_type, mixed_flag) pair.
        let mixed_flag = prefer_mixed && matches!(emitted, BlockType::Short);
        (emitted, mixed_flag)
    }
}

impl Default for BlockTypeStateMachine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A steady-state non-attack input stream produces an unbroken
    /// run of `Long` blocks. This is the baseline preservation
    /// guarantee: when the detector says nothing, the scheduler
    /// emits Long.
    #[test]
    fn all_calm_emits_only_long() {
        let mut sm = BlockTypeStateMachine::new();
        for _ in 0..10 {
            let bt = sm.step(false, false);
            assert_eq!(bt, BlockType::Long);
        }
    }

    /// A single attack landing on the next granule transitions
    /// `Long → Start → Short → Stop → Long`.
    #[test]
    fn single_burst_emits_start_short_stop_long() {
        // Stream model: granule 0 calm, granule 1 attack, granule 2
        // calm, granule 3 calm.
        //
        // Lookahead pair feeds:
        //   gr0: (cur=false, next=true)  — gr1 has attack
        //   gr1: (cur=true,  next=false) — gr1 itself has attack
        //   gr2: (cur=false, next=false)
        //   gr3: (cur=false, next=false)
        let mut sm = BlockTypeStateMachine::new();
        assert_eq!(sm.step(false, true), BlockType::Start);
        assert_eq!(sm.step(true, false), BlockType::Short);
        assert_eq!(sm.step(false, false), BlockType::End);
        assert_eq!(sm.step(false, false), BlockType::Long);
    }

    /// A sustained burst (multiple attack-flagged granules in a row)
    /// stays in `Short` until the burst ends.
    #[test]
    fn sustained_burst_holds_short() {
        let mut sm = BlockTypeStateMachine::new();
        // gr0: prelude calm with lookahead attack → Start
        assert_eq!(sm.step(false, true), BlockType::Start);
        // gr1, gr2, gr3: all carry their own attack + look ahead to
        // more attack → Short, Short, Short
        assert_eq!(sm.step(true, true), BlockType::Short);
        assert_eq!(sm.step(true, true), BlockType::Short);
        assert_eq!(sm.step(true, false), BlockType::Short);
        // gr4: calm, ends the burst → Stop
        assert_eq!(sm.step(false, false), BlockType::End);
        // gr5: returns to Long
        assert_eq!(sm.step(false, false), BlockType::Long);
    }

    /// Two consecutive bursts separated by a single Long granule
    /// each fire their own Long→Start→Short→Stop→Long sequence.
    /// Important: the gap between bursts must be at least one Long
    /// granule because Stop must precede Long.
    #[test]
    fn two_bursts_with_long_gap_both_fire() {
        let mut sm = BlockTypeStateMachine::new();
        // Burst 1: lookahead-attack on gr0 → Start, etc.
        assert_eq!(sm.step(false, true), BlockType::Start);
        assert_eq!(sm.step(true, false), BlockType::Short);
        assert_eq!(sm.step(false, false), BlockType::End);
        // Gap of one Long granule (gr3); gr4 has next-attack.
        assert_eq!(sm.step(false, false), BlockType::Long);
        // Burst 2: same pattern.
        assert_eq!(sm.step(false, true), BlockType::Start);
        assert_eq!(sm.step(true, false), BlockType::Short);
        assert_eq!(sm.step(false, false), BlockType::End);
        assert_eq!(sm.step(false, false), BlockType::Long);
    }

    /// An isolated attack on the current granule with no lookahead
    /// warning is **not** converted into a Short — the geometry
    /// requires a Start first, and we have no time to insert one.
    /// The conservative fallback is Long.
    #[test]
    fn current_only_attack_without_lookahead_falls_back_to_long() {
        let mut sm = BlockTypeStateMachine::new();
        // gr0: cur=true, next=false — burst confined to gr0 with no
        // anticipatory signal. We can't retroactively emit a Start
        // for gr-prev, so gr0 stays Long.
        assert_eq!(sm.step(true, false), BlockType::Long);
        // gr1: returns to standard calm.
        assert_eq!(sm.step(false, false), BlockType::Long);
    }

    /// Reset restores the initial `Long` state.
    #[test]
    fn reset_returns_to_long() {
        let mut sm = BlockTypeStateMachine::new();
        sm.step(false, true); // Start
        sm.step(true, false); // Short
        assert_eq!(sm.prev_block_type(), BlockType::Short);
        sm.reset();
        assert_eq!(sm.prev_block_type(), BlockType::Long);
        // Reset is also functional after multiple bursts.
        assert_eq!(sm.step(false, false), BlockType::Long);
    }

    /// After a Start has been emitted, the next granule MUST be
    /// Short — even if the detector unexpectedly clears
    /// `cur_attack`. The geometry is committed.
    #[test]
    fn start_always_followed_by_short() {
        let mut sm = BlockTypeStateMachine::new();
        assert_eq!(sm.step(false, true), BlockType::Start);
        // Even with no attack signal at all, the Short MUST follow.
        assert_eq!(sm.step(false, false), BlockType::Short);
        // And the next granule with no attack closes with Stop.
        assert_eq!(sm.step(false, false), BlockType::End);
        // And the next is back to Long.
        assert_eq!(sm.step(false, false), BlockType::Long);
    }

    /// `step_with_mixed(_, _, false)` is byte-for-byte equivalent
    /// to `step(_, _)` — the mixed flag is suppressed and the
    /// transition geometry is unchanged.
    #[test]
    fn step_with_mixed_false_matches_step() {
        let mut sm_a = BlockTypeStateMachine::new();
        let mut sm_b = BlockTypeStateMachine::new();
        // Build up a representative attack pattern: calm, lookahead
        // attack, sustained burst, then calm.
        let pattern = [
            (false, true),
            (true, true),
            (true, false),
            (false, false),
            (false, false),
        ];
        for &(cur, nxt) in pattern.iter() {
            let a = sm_a.step(cur, nxt);
            let (b, mixed) = sm_b.step_with_mixed(cur, nxt, false);
            assert_eq!(a, b, "block_type differs on (cur={cur}, next={nxt})");
            assert!(!mixed, "mixed flag must be false when prefer_mixed=false");
        }
    }

    /// `step_with_mixed(_, _, true)` propagates the mixed flag onto
    /// every Short emission and suppresses it on every other
    /// emission (Long / Start / End). This is the syntactic
    /// invariant §2.4.2.7 requires: `mixed_block_flag` is only
    /// meaningful for `block_type == 2`.
    #[test]
    fn step_with_mixed_true_sets_flag_only_on_short() {
        let mut sm = BlockTypeStateMachine::new();
        // Long → Start (flag must be false; Start is not Short)
        let (bt, mixed) = sm.step_with_mixed(false, true, true);
        assert_eq!(bt, BlockType::Start);
        assert!(!mixed);
        // Start → Short (flag must be true — caller asked for mixed)
        let (bt, mixed) = sm.step_with_mixed(true, false, true);
        assert_eq!(bt, BlockType::Short);
        assert!(mixed);
        // Short → End (flag must be false again)
        let (bt, mixed) = sm.step_with_mixed(false, false, true);
        assert_eq!(bt, BlockType::End);
        assert!(!mixed);
        // End → Long (flag must be false)
        let (bt, mixed) = sm.step_with_mixed(false, false, true);
        assert_eq!(bt, BlockType::Long);
        assert!(!mixed);
    }

    /// A sustained burst with prefer_mixed=true emits Start, then
    /// several Short-with-mixed, then End — each Short carries the
    /// flag, Start and End do not.
    #[test]
    fn step_with_mixed_sustained_burst_flags_each_short() {
        let mut sm = BlockTypeStateMachine::new();
        let (bt, mixed) = sm.step_with_mixed(false, true, true);
        assert_eq!(bt, BlockType::Start);
        assert!(!mixed);
        for _ in 0..3 {
            let (bt, mixed) = sm.step_with_mixed(true, true, true);
            assert_eq!(bt, BlockType::Short);
            assert!(mixed);
        }
        let (bt, mixed) = sm.step_with_mixed(true, false, true);
        assert_eq!(bt, BlockType::Short);
        assert!(mixed);
        let (bt, mixed) = sm.step_with_mixed(false, false, true);
        assert_eq!(bt, BlockType::End);
        assert!(!mixed);
    }

    /// `prefer_mixed` can vary per call, so the flag tracks the
    /// preference of the call that *emits* the Short, not any
    /// committed earlier setting.
    #[test]
    fn step_with_mixed_per_call_preference() {
        let mut sm = BlockTypeStateMachine::new();
        // prefer_mixed=false on the Start call → no flag on Start
        // (trivially — Start is never flagged).
        let (bt, mixed) = sm.step_with_mixed(false, true, false);
        assert_eq!(bt, BlockType::Start);
        assert!(!mixed);
        // prefer_mixed=true on the Short call → flag set on Short.
        let (bt, mixed) = sm.step_with_mixed(true, true, true);
        assert_eq!(bt, BlockType::Short);
        assert!(mixed);
        // prefer_mixed=false on the next Short call → flag NOT set.
        let (bt, mixed) = sm.step_with_mixed(true, false, false);
        assert_eq!(bt, BlockType::Short);
        assert!(!mixed);
    }

    /// After Stop, the next granule MUST be Long — even if the
    /// detector immediately flags the next granule (the Stop→Long
    /// constraint dominates over the detector's anticipation; we
    /// need at least one Long granule before re-entering the
    /// `Long → Start` burst geometry).
    #[test]
    fn stop_always_followed_by_long_even_on_immediate_burst() {
        let mut sm = BlockTypeStateMachine::new();
        // Build up to a Stop.
        assert_eq!(sm.step(false, true), BlockType::Start);
        assert_eq!(sm.step(true, false), BlockType::Short);
        // Short with cur=false,next=false closes via Stop.
        assert_eq!(sm.step(false, false), BlockType::End);
        // Even though the lookahead says the next granule has an
        // attack, gr-post-Stop must be Long (Stop's tail is the
        // long-window tail; a Short head can't splice into it).
        // The detector's flag on the NEXT granule is dropped this
        // turn; the next call (with cur=true,next=...) will go
        // Long→Start in the standard way.
        assert_eq!(sm.step(false, true), BlockType::Long);
        // And the next granule (with next_attack=true ahead of it)
        // transitions Long→Start as the newly-armed scheduler picks
        // up the lookahead.
        assert_eq!(sm.step(false, true), BlockType::Start);
    }
}
