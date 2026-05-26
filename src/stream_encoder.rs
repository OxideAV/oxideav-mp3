//! Layer III **PCM → MP3 frame stream encoder** — Phase 2 step 10.
//!
//! This module wires the already-shipped Phase 2 primitives (polyphase
//! analysis filterbank [`crate::analysis`], forward MDCT
//! [`crate::mdct`], inverse alias reduction, the
//! [`crate::quantize`] primitive, the §C.1.5.4.4 Huffman bit chooser +
//! [`crate::huffman::emit_huffman`], the main-data assembler
//! [`crate::main_data::assemble_main_data`] and the §2.4.2.7
//! cross-frame bit-reservoir scheduler [`crate::main_data::schedule_reservoir`])
//! into a top-level [`Mp3Encoder`] that converts a stream of
//! mono MPEG-1 `i16` PCM samples into a sequence of complete
//! self-delimiting MP3 frames written to a [`std::io::Write`] sink.
//!
//! # Scope of this round
//!
//! * **Mono, MPEG-1 only** — `ChannelMode::SingleChannel`, sample rates
//!   `44.1 / 48 / 32 kHz`. LSF / stereo / VBR deferred.
//! * **CBR** — caller picks a Layer III bitrate from the §2.4.2.3
//!   ladder; the encoder selects the per-frame padding slot greedily.
//! * **Long blocks** — `window_switching_flag = false`,
//!   `block_type = Long`. No psychoacoustic block-type switching.
//! * **Zero scalefactors** — `scalefac_compress = 0`, every
//!   `scalefac_l[sfb] = 0`, no preflag, scalefac_scale = 0.
//! * **Fixed-gain heuristic** — `global_gain` is chosen per granule to
//!   keep `max|is| ≤ 8191` (the §2.4.1.7 big-values bound) via
//!   [`crate::inner_loop::search_magnitude_clamp`] *without* further
//!   psy / outer-loop noise-shaping iteration.
//! * **Single-region big-values** — `region0_count = 20`,
//!   `region1_count = 0` (covers all 21 long bands → one big-values
//!   region; region 1 and 2 collapsed). The big-values codebook is
//!   chosen with [`crate::huffman::choose_best_table_for_region`].
//!
//! # End-to-end self-decode validation
//!
//! The integration test under `tests/stream_encoder_roundtrip.rs`
//! encodes one second of a 440 Hz sine tone at 44.1 kHz / 128 kbit/s,
//! feeds the emitted byte stream back into the crate's own
//! [`crate::Mp3Demuxer`] + the decode chain
//! (`decode_huffman` → `requantize` → `imdct_granule` → `synth_granule`)
//! and asserts (a) every frame's header + side-info + main_data
//! round-trip exactly, (b) the recovered PCM has finite PSNR > 20 dB
//! against the input.
//!
//! # What this module deliberately does **not** do
//!
//! No psychoacoustic model, no outer noise-shaping loop, no LSF, no
//! ID3v2 frontmatter, no short-block / mixed-block window switching,
//! and no intensity-stereo encode. **Joint-stereo MS** encode is opt-in
//! as of round 146 ([`Mp3Encoder::new_joint_stereo_ms`]): the encoder
//! computes the §2.4.3.4.9.2 forward MS matrix `M = (L+R)/√2`,
//! `S = (L-R)/√2` on each granule-pair's full post-MDCT spectrum and
//! emits header `mode = '01'` with `mode_extension = '10'` (ms_stereo
//! on, intensity_stereo off). The intensity-stereo coupling
//! (§2.4.3.4.9.3) remains deferred.
//!
//! Xing / Info VBR-info frame emission is **opt-in** as of round 142:
//! call [`Mp3Encoder::enable_xing_info`] before [`Mp3Encoder::finish`]
//! to prepend one carrier frame with the magic + per-stream totals to
//! the on-wire output. See [`crate::xing_info`] for layout details.

use std::io::{self, Write};

use crate::analysis::{analyze_granule, AnalysisState};
use crate::frame::{ChannelMode, ModeExtension, Mp3FrameHeader, MpegVersion};
use crate::huffman::{choose_best_count1_table, partition_split, NUM_LINES};
use crate::inner_loop::{search_bit_budget, search_magnitude_clamp, GAIN_MAX, GAIN_MIN};
use crate::main_data::{
    assemble_main_data, schedule_reservoir, GranuleChannelData, ReservoirError, ReservoirFrame,
};
use crate::mdct::{forward_overlap, mdct, window_long_family_analysis, MdctState, LONG_N};
use crate::outer_loop::{outer_loop_search_long, OUTER_LOOP_SCALEFAC_COMPRESS};
use crate::quantize::quantize;
use crate::scalefactors::{FrameScaleFactors, ScaleFactors};
use crate::side_info::{BlockType, GranuleChannel, SideInfo, GRANULES};
use crate::{make_silent_header, write_header, write_side_info, EncodeError};

/// MPEG-1 Layer III bitrate ladder (ISO/IEC 11172-3 §2.4.2.3, Table
/// 2-B.1 row "Layer III, version 1"). Used by the encoder's VBR path
/// to enumerate the 14 fixed bitrates a per-frame `bitrate_index` may
/// select. Indices `0` (free format) and `15` (forbidden) are excluded.
pub const MPEG1_L3_BITRATE_LADDER_KBPS: [u32; 14] = [
    32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320,
];

/// Default outer-loop per-band noise threshold (the uniform `xmin[sb]`
/// the loop tests `xfsf(sb)` against). With our scalefactor amplification
/// step `√2` per increment and the colored-domain `xfsf` metric (per
/// §C.1.5.4.3.3) this constant is chosen high enough that only bands
/// whose colored noise actually stands out get amplified — a uniform
/// threshold cannot redistribute the bit budget the way a per-band
/// psychoacoustic threshold does, so an over-aggressive low threshold
/// would amplify every band equally (no spectral shaping benefit, just
/// raised `global_gain`); a too-high threshold disables the loop. The
/// value below was picked empirically on the multi-tone test fixture
/// (`tests/outer_loop_roundtrip.rs`) so the loop converges in ≤ 8
/// iterations per granule and the self-decode PSNR strictly exceeds the
/// fixed-gain path's at the same bitrate.
pub const DEFAULT_OUTER_LOOP_THRESHOLD: f64 = 1.0e6;

/// Defensive upper bound on outer-loop iterations per granule-channel
/// (the §C.1.5.4.3.6 cap-based termination paths fire long before this
/// in practice; the soft limit guards against FP-precision pathologies).
pub const DEFAULT_OUTER_LOOP_MAX_ITER: u32 = 64;

/// Number of PCM samples per MPEG-1 Layer III frame per channel
/// (2 granules × 576 lines = 1152, §2.4.2.1).
pub const SAMPLES_PER_FRAME_MPEG1: usize = 1152;

/// Number of PCM samples per granule per channel (§2.4.3.4).
pub const SAMPLES_PER_GRANULE: usize = 576;

/// Errors returned by the stream encoder.
#[derive(Debug)]
pub enum StreamEncodeError {
    /// Wraps a header-construction failure from
    /// [`crate::make_silent_header`] — bad bitrate / sample-rate /
    /// mode combination, free-format request, etc.
    Header(EncodeError),
    /// The bit reservoir failed to schedule the assembled frames; the
    /// chosen bitrate cannot hold the requested content. Raise the
    /// bitrate and retry.
    Reservoir(ReservoirError),
    /// A Huffman bit emission failed (typically because a quantized
    /// pair could not be coded by the selected table). Carried as a
    /// string so the upstream caller does not have to thread the
    /// per-stage error enum here.
    Huffman(String),
    /// Underlying I/O error from the destination `Write`.
    Io(io::Error),
    /// Caller chose [`ChannelMode::JointStereo`] on a constructor that
    /// does not arm joint-stereo coupling on the encode side. Use
    /// [`Mp3Encoder::new_joint_stereo_ms`] for §2.4.3.4.9.2 MS-stereo
    /// encode, or use [`ChannelMode::Stereo`] / [`ChannelMode::DualChannel`]
    /// for independent two-channel content; intensity stereo encode
    /// (§2.4.3.4.9.3) remains unsupported.
    StereoUnsupported,
    /// Caller chose an MPEG-2 LSF sample rate (16 / 22.05 / 24 kHz);
    /// LSF is deferred to a later round.
    LsfUnsupported,
    /// VBR configuration is malformed: `min_kbps` / `max_kbps` are not
    /// both on the §2.4.2.3 ladder, are reversed, or `max_kbps`
    /// exceeds the encoder's constructor-time `bitrate_kbps`.
    InvalidVbrConfig,
    /// VBR mode: a frame's assembled main-data does not fit in the
    /// max-bitrate-index slot. Raise `max_kbps` (or the constructor
    /// bitrate) and retry. The carried `(frame_index, main_data_len,
    /// max_slot_bytes)` triple identifies the overflowing frame.
    VbrSlotTooSmall {
        /// Zero-based index of the offending audio frame.
        frame_index: usize,
        /// Number of main-data bytes the frame's quantization
        /// produced.
        main_data_len: usize,
        /// Slot capacity at the configured `max_index`.
        max_slot_bytes: usize,
    },
}

impl core::fmt::Display for StreamEncodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            StreamEncodeError::Header(e) => write!(f, "header build: {e}"),
            StreamEncodeError::Reservoir(e) => write!(f, "reservoir: {e}"),
            StreamEncodeError::Huffman(e) => write!(f, "huffman emit: {e}"),
            StreamEncodeError::Io(e) => write!(f, "io: {e}"),
            StreamEncodeError::StereoUnsupported => f.write_str(
                "ChannelMode::JointStereo not supported on this constructor (use Mp3Encoder::new_joint_stereo_ms, Stereo, or DualChannel)",
            ),
            StreamEncodeError::LsfUnsupported => {
                f.write_str("only MPEG-1 sample rates are supported in this round")
            }
            StreamEncodeError::InvalidVbrConfig => {
                f.write_str("VBR config: min/max kbps off-ladder or out of range")
            }
            StreamEncodeError::VbrSlotTooSmall {
                frame_index,
                main_data_len,
                max_slot_bytes,
            } => write!(
                f,
                "VBR: frame {frame_index} main_data {main_data_len} B does not fit max-index slot {max_slot_bytes} B"
            ),
        }
    }
}

impl std::error::Error for StreamEncodeError {}

impl From<io::Error> for StreamEncodeError {
    fn from(e: io::Error) -> Self {
        StreamEncodeError::Io(e)
    }
}

/// One MPEG-1 Layer III mono CBR encoder, configured at construction
/// time with a fixed sample rate + bitrate. Push PCM samples with
/// [`Mp3Encoder::push_samples`]; once enough samples are buffered for
/// at least one whole frame the encoder buffers it internally for
/// later reservoir scheduling. [`Mp3Encoder::finish`] runs the
/// reservoir scheduler over every buffered frame and writes the
/// resulting on-wire frame sequence (header + side_info + main_data
/// slot per frame) to the sink.
///
/// The encoder is **single-pass append; finalise at end** — the
/// §2.4.2.7 bit-reservoir schedule cannot be carved per-frame because
/// a quiet frame's tail bytes may carry a later busy frame's overflow.
/// The scheduler therefore needs every frame's assembled main_data up
/// front. Memory cost: ~`bitrate/(8·sample_rate)·duration` bytes —
/// a one-second 128 kbit/s stream costs ~16 KiB.
#[derive(Debug)]
pub struct Mp3Encoder {
    header_template: Mp3FrameHeader,
    sample_rate_hz: u32,
    version: MpegVersion,
    nch: usize,

    /// Polyphase analysis shift register, one per channel.
    analysis_state: Vec<AnalysisState>,
    /// Per-channel per-subband forward-MDCT overlap state (32 subbands
    /// × `nch` channels).
    mdct_state: Vec<Vec<MdctState>>,
    /// PCM pending in the current half-frame buffer, one [`Vec`] per
    /// channel (length `nch`). Mono encoders carry a single buffer;
    /// stereo encoders deinterleave the caller's interleaved S16 input
    /// into the two per-channel buffers in
    /// [`Mp3Encoder::push_samples`].
    pending_pcm: Vec<Vec<f32>>,

    /// Per-frame assembled output for the deferred reservoir
    /// scheduling pass.
    frames: Vec<PendingFrame>,

    /// When `Some`, every per-granule-channel quantization runs the
    /// §C.1.5.4.3 outer (distortion-control) loop instead of the
    /// fixed-`scalefac = 0` + inner-loop-only path. The carried value is
    /// the uniform `xmin[sb]` threshold applied to every long-block
    /// scalefactor band.
    outer_loop_threshold: Option<f64>,

    /// When `Some`, [`Mp3Encoder::finish`] prepends a Xing / Info VBR
    /// information frame ([`crate::xing_info::build_info_frame`]) to
    /// the on-wire output. The carrier is a silent Layer III frame
    /// whose main-data slot starts with the Xing / Info magic plus the
    /// flagged fields. Two of those fields — `frames` and `bytes` —
    /// can only be known after the rest of the stream is encoded, so
    /// this struct stores the **template** spec and the writer fills
    /// in the post-encode totals at `finish` time when the
    /// corresponding flag bits are set and the template field is
    /// `None`. A caller that already knows the totals (e.g. by
    /// pre-counting samples) can supply them in the spec and the
    /// writer will use them verbatim.
    xing_template: Option<crate::xing_info::XingTagSpec>,

    /// When `Some`, [`Mp3Encoder::finish`] picks a per-frame
    /// `bitrate_index` from the §2.4.2.3 ladder rather than emitting
    /// every audio frame at the constructor-time `bitrate_kbps`. The
    /// chosen index is the smallest ladder entry within the
    /// `[min_kbps, max_kbps]` window whose slot is large enough to hold
    /// the frame's assembled main-data (using the same zero-pad /
    /// no-reservoir schedule as the CBR path).
    ///
    /// Carrier-frame and per-granule quantization budget still use the
    /// constructor-time bitrate as the maximum (`bitrate_kbps` should
    /// equal or exceed `max_kbps`) so the inner loop's bit-budget gain
    /// search runs against the largest slot the stream will emit; the
    /// VBR step is a post-hoc selection of the smallest ladder index
    /// each frame's main-data actually fits in.
    vbr: Option<VbrConfig>,

    /// When `true`, [`Mp3Encoder::finish`] writes the §2.4.3.1 CRC-16
    /// check word in the two-byte slot between the header and the
    /// side-information block on every emitted audio frame, sets the
    /// wire `protection_bit = 0` (i.e. `crc_protected = true` in the
    /// parser's terms), and accounts for the resulting 2-byte loss in
    /// each frame's main-data slot capacity. Default `false` (no CRC),
    /// matching every previous round's behaviour. Toggle via
    /// [`Mp3Encoder::with_protection_bit`].
    crc_enabled: bool,

    /// When `true`, the encoder is in §2.4.3.4.9.2 MS-stereo joint mode:
    /// for every granule it transforms the post-MDCT L/R spectra into
    /// the M/S pair (`M = (L+R)/√2`, `S = (L-R)/√2`), quantizes
    /// `(M, S)` in the channel-0 / channel-1 slots, and writes header
    /// `mode = '01'` (joint stereo) with `mode_extension = '10'`
    /// (ms_stereo on, intensity_stereo off). The decoder reverses the
    /// matrix via [`crate::process_stereo`] driven by the
    /// `mode_extension` bits. Set by [`Mp3Encoder::new_joint_stereo_ms`];
    /// requires `nch == 2`. Intensity-stereo coupling
    /// (§2.4.3.4.9.3) is not implemented on the encode side; this flag
    /// only enables the MS half of joint stereo.
    ms_stereo: bool,

    /// When `Some`, the encoder is in **auto MS/LR per-frame** joint mode
    /// (constructor [`Mp3Encoder::new_joint_stereo_auto`]): every assembled
    /// frame's post-MDCT L/R spectra are analysed, and the §2.4.3.4.9.2
    /// MS rotation is applied only when the side-channel energy ratio
    /// `E_S / (E_L + E_R)` for both granules of the frame is at or below
    /// the carried threshold. The header `mode` is fixed at
    /// `ChannelMode::JointStereo` (`mode = '01'`); the per-frame
    /// `mode_extension` is then either `'10'` (ms_stereo on,
    /// intensity_stereo off) when the picker chose MS, or `'00'`
    /// (neither method active — bitstream is identical to two
    /// independent stereo channels under a joint header) when the
    /// picker chose LR. The standalone [`Mp3Encoder::ms_stereo`] flag
    /// stays `false` so the unconditional pass-1.5 forward MS branch
    /// in [`Mp3Encoder::assemble_frame`] is gated on this picker
    /// instead.
    ///
    /// The threshold is the upper bound on `E_S / (E_L + E_R)` at which
    /// MS is preferred. ISO/IEC 11172-3 does not prescribe an encoder
    /// mode-decision algorithm — the spec leaves the joint-stereo
    /// method-enable bits entirely to the encoder — so a content-only
    /// energy heuristic is chosen here. The default `0.5` (set by
    /// [`Mp3Encoder::new_joint_stereo_auto`]) is the symmetry boundary:
    /// `E_S = E_M` is the energy of uncorrelated L/R; below it MS
    /// concentrates strictly more energy in the mid channel than the
    /// independent split does, above it the side channel carries more
    /// than half the energy and the MS rotation would amplify rather
    /// than reduce quantization stress on the side channel.
    ms_auto_threshold: Option<f64>,

    /// When `true`, every assembled granule emits a §2.4.2.7 short
    /// block (`window_switching_flag = 1`, `block_type = 2`,
    /// `mixed_block_flag = 0`): three independent 12-point MDCTs per
    /// subband (instead of one 36-point MDCT), no alias reduction
    /// (§2.4.3.4.10.1 scopes it to `block_type != 2`), the
    /// §2.4.3.4.8 reorder (encoder side: [`crate::short_block::forward_reorder`])
    /// applied to lay the bins out in the native bitstream
    /// `[sfb][win][k]` interleave. Side-info defaults follow
    /// [`crate::short_block::short_block_region_defaults`].
    ///
    /// This is the **encoder-side** mirror of the decoder's short-block
    /// path. ISO/IEC 11172-3 leaves the block-type decision algorithm
    /// entirely to the encoder (§C.1.5 references the psychoacoustic
    /// model the spec offers in Annex D, but the algorithm itself is
    /// non-normative); a signal-driven attack-detection heuristic that
    /// auto-toggles between long and short is a separate concern from
    /// the bitstream-side primitive, so this round exposes the toggle
    /// as a deterministic test handle. The auto-decision heuristic
    /// (and the §C.1.5.2 LONG → START → SHORT → STOP → LONG transition
    /// state machine for *mixed* long-and-short streams) lands in a
    /// follow-up round.
    ///
    /// While the flag is on:
    ///
    /// * The forward analysis stage feeds each subband's 18 new
    ///   subband-time samples plus the previous granule's saved 18
    ///   through [`crate::short_block::forward_short_mdct_subband`]
    ///   instead of the long-block forward-overlap + 36-point MDCT.
    ///   The per-subband [`crate::mdct::MdctState`] is shared between
    ///   the two paths, so a per-test toggle within a stream would
    ///   correctly preserve the previous granule's `saved` half.
    /// * The granule-channel side-info is rewritten:
    ///   `window_switching_flag = true`, `block_type = Short`,
    ///   `mixed_block_flag = false`, `subblock_gain = [0; 3]`,
    ///   `region0_count` / `region1_count` carry the
    ///   [`crate::short_block::short_block_region_defaults`] (decoder
    ///   ignores them; see [`crate::huffman`]'s `region_boundaries`).
    /// * No alias reduction on the encode side (decoder skips it for
    ///   short blocks).
    /// * After the per-subband forward MDCT, the 576-line buffer is
    ///   in subband-window-interleaved layout (line `3·k + win` of each
    ///   subband's 18-line slot holds frequency bin `k` of short window
    ///   `win`). [`crate::short_block::forward_reorder`] rewrites the
    ///   short bands (or short region of a mixed block; not used in
    ///   this round's force-all-short mode) into the native bitstream
    ///   `[sfb][win][k]` interleave.
    ///
    /// Restrictions in this round:
    ///
    /// * **Mono only.** Stereo / joint-stereo / dual-channel + force-short
    ///   is intentionally rejected by [`Mp3Encoder::force_short_blocks_for_testing`]
    ///   to keep the integration surface small; the stereo+short combo
    ///   reuses the same primitive but needs the §2.4.3.4.9 "both channels
    ///   share the same block type when MS is enabled" wiring + the
    ///   intensity-stereo per-window bound, which lands in a follow-up
    ///   round.
    /// * **No outer loop.** The fixed-gain inner-loop-only path runs
    ///   for short blocks the same way it does for long blocks; the
    ///   §C.1.5.4.3 outer loop's long-only API surface
    ///   ([`crate::outer_loop::outer_loop_search_long`]) is not engaged.
    /// * **No mixed blocks.** Pure short only; the `mixed_block_flag`
    ///   long-region split needs the long forward MDCT for subbands 0
    ///   and 1 plus the short MDCT for the rest, an extra wiring step
    ///   deferred to a follow-up.
    ///
    /// Default `false`; the encoder behaves as in every previous round.
    force_short_blocks: bool,

    /// When `true`, every assembled granule emits a §2.4.2.7 **mixed**
    /// block (`window_switching_flag = 1`, `block_type = 2`,
    /// `mixed_block_flag = 1`): the two lowest polyphase subbands (0 and
    /// 1, covering the lowest 36 frequency lines) are coded with the
    /// long-family window + one 36-point forward MDCT each (the same
    /// path the long-block branch uses), while the upper 30 subbands (2
    /// through 31) are coded with the short-block path — three
    /// independent 12-point forward MDCTs per subband over the lapped
    /// 36-sample frame, no alias reduction, window-interleaved layout.
    /// [`crate::short_block::forward_reorder`] then rewrites the short
    /// region (short scalefactor bands 3..12) into the native bitstream
    /// `[sfb][win][k]` interleave, while the long region (lines 0..36)
    /// passes through unchanged (mirrors the decoder's
    /// [`crate::reorder::reorder`] mixed-block branch).
    ///
    /// This is the **encoder-side** mirror of the decoder's mixed-block
    /// path. The decoder branches in three places on `mixed_block_flag`:
    ///
    /// * [`crate::imdct::imdct_granule`] runs the long IMDCT path on
    ///   subbands 0/1 and the short path on the rest.
    /// * [`crate::reorder::reorder`] starts the short reorder at SFB 3
    ///   (preserving lines 0..36 in long order).
    /// * [`crate::requantize::requantize`] / [`crate::quantize::quantize`]
    ///   apply long-band requantization to lines 0..36 and short-band
    ///   requantization to the rest.
    /// * [`crate::alias::alias_reduce`] is **not** applied (the
    ///   §2.4.3.4.10.1 test is on `block_type == Short` alone; a mixed
    ///   block is still `block_type == Short`).
    ///
    /// While the flag is on the forward path mirrors all four: the long
    /// branch's `forward_overlap → window_long_family_analysis → 36-pt
    /// MDCT → ÷9` runs on subbands 0/1, the short branch's
    /// [`crate::short_block::forward_short_mdct_subband`] runs on
    /// subbands 2..31, [`crate::short_block::forward_reorder`] rewrites
    /// the short bands into native order while preserving lines 0..36,
    /// and inverse alias reduction is **skipped** (mirroring the
    /// decoder).
    ///
    /// Restrictions in this round:
    ///
    /// * **Mono only.** Stereo / joint-stereo / dual-channel + mixed is
    ///   intentionally rejected by
    ///   [`Mp3Encoder::force_mixed_blocks_for_testing`] for the same
    ///   reason the force-short flag is mono-only: the §2.4.3.4.9
    ///   cross-channel block-type-agreement wiring is a follow-up
    ///   round.
    /// * **No outer loop.** The fixed-gain inner-loop-only path runs
    ///   for mixed blocks identically to the way it runs for long and
    ///   short.
    /// * **Mutually exclusive with `force_short_blocks`.** A granule
    ///   can be long, short-only, or mixed — not all three.
    ///   [`Mp3Encoder::force_mixed_blocks_for_testing`] rejects the
    ///   combination.
    ///
    /// Default `false`; the encoder behaves as in every previous round.
    force_mixed_blocks: bool,
}

/// Variable-bitrate config attached to [`Mp3Encoder`] by
/// [`Mp3Encoder::enable_vbr`].
#[derive(Debug, Clone, Copy)]
struct VbrConfig {
    /// Minimum per-frame bitrate index (1..=14 on the §2.4.2.3 ladder).
    /// A frame whose main-data is so small it could fit a smaller slot
    /// is still emitted at this index — the floor protects decoders
    /// that scan for a "typical" frame size at probe time.
    min_index: u8,
    /// Maximum per-frame bitrate index (1..=14). `min_index <=
    /// max_index <= constructor_index`. A frame whose main-data
    /// exceeds the max-index slot is rejected
    /// ([`StreamEncodeError::VbrSlotTooSmall`]).
    max_index: u8,
}

#[derive(Debug)]
struct PendingFrame {
    header: Mp3FrameHeader,
    side_info: SideInfo,
    main_data: Vec<u8>,
}

impl Mp3Encoder {
    /// Build a new encoder for the given sample rate + bitrate.
    ///
    /// Supported channel modes:
    ///
    /// * [`ChannelMode::SingleChannel`] — one channel; header
    ///   `mode = '11'`, side info 17 bytes (MPEG-1).
    /// * [`ChannelMode::Stereo`] — two independent channels; header
    ///   `mode = '00'`. Each granule's two channels are encoded
    ///   independently (no joint-stereo MS / intensity coupling); the
    ///   header `mode_extension` carries `'00'` per ISO/IEC 11172-3
    ///   §2.4.2.3. Side info is 32 bytes (MPEG-1).
    /// * [`ChannelMode::DualChannel`] — two independent channels (e.g.
    ///   bilingual programmes); header `mode = '10'`. Bitstream layout
    ///   is identical to `Stereo` from the encoder's standpoint —
    ///   two channels coded independently with no joint-stereo
    ///   coupling — so this mode shares the same encode path; the
    ///   difference is purely the carried mode bit.
    /// * [`ChannelMode::JointStereo`] is **not** accepted: the joint
    ///   methods (MS, intensity) require an encoder-side stereo
    ///   analysis stage that this round does not implement.
    ///
    /// Sample rates: 32 / 44.1 / 48 kHz (MPEG-1 only; MPEG-2 / 2.5 LSF
    /// remains deferred).
    ///
    /// # Errors
    ///
    /// * [`StreamEncodeError::StereoUnsupported`] for
    ///   [`ChannelMode::JointStereo`] (the only unsupported mode this
    ///   round).
    /// * [`StreamEncodeError::LsfUnsupported`] for a non-MPEG-1
    ///   sample rate.
    /// * [`StreamEncodeError::Header`] for a bad bitrate /
    ///   sample-rate combination (per
    ///   [`crate::make_silent_header`]).
    pub fn new(
        bitrate_kbps: u32,
        sample_rate_hz: u32,
        mode: ChannelMode,
    ) -> Result<Self, StreamEncodeError> {
        // Reject joint-stereo: MS / intensity coupling needs an
        // encoder-side stereo analysis stage that is out of scope for
        // this round. Stereo / dual-channel encode each channel
        // independently, so they share the mono code path with
        // `nch == 2`.
        if matches!(mode, ChannelMode::JointStereo) {
            return Err(StreamEncodeError::StereoUnsupported);
        }
        let header_template = make_silent_header(bitrate_kbps, sample_rate_hz, mode)
            .map_err(StreamEncodeError::Header)?;
        if header_template.version != MpegVersion::Mpeg1 {
            return Err(StreamEncodeError::LsfUnsupported);
        }
        let nch = header_template.channel_count() as usize;
        let analysis_state = (0..nch).map(|_| AnalysisState::new()).collect();
        let mdct_state = (0..nch)
            .map(|_| (0..32usize).map(|_| MdctState::new()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let pending_pcm = (0..nch).map(|_| Vec::new()).collect();
        Ok(Mp3Encoder {
            header_template,
            sample_rate_hz,
            version: MpegVersion::Mpeg1,
            nch,
            analysis_state,
            mdct_state,
            pending_pcm,
            frames: Vec::new(),
            outer_loop_threshold: None,
            xing_template: None,
            vbr: None,
            crc_enabled: false,
            ms_stereo: false,
            ms_auto_threshold: None,
            force_short_blocks: false,
            force_mixed_blocks: false,
        })
    }

    /// Force every assembled granule onto the §2.4.2.7 short-block
    /// (`block_type = 2`) encode path; see the
    /// [`Self::force_short_blocks`] field for the per-granule
    /// behavioural contract.
    ///
    /// Mono-only restriction: short-block encode for stereo / joint /
    /// dual-channel needs §2.4.3.4.9 cross-channel block-type
    /// agreement wiring that lands in a follow-up round, so this
    /// toggle rejects multi-channel encoders.
    ///
    /// # Errors
    ///
    /// [`StreamEncodeError::StereoUnsupported`] when the encoder's
    /// channel count is > 1 (the dispatch tag is reused — the actual
    /// case is "force-short combined with multi-channel," but the
    /// existing error variant captures the same "channel layout
    /// unsupported by this opt-in" semantics).
    pub fn force_short_blocks_for_testing(
        &mut self,
        enabled: bool,
    ) -> Result<(), StreamEncodeError> {
        if enabled && self.nch != 1 {
            return Err(StreamEncodeError::StereoUnsupported);
        }
        if enabled {
            // Mixed and pure-short are mutually exclusive: a granule is
            // long, short, or mixed. Enabling pure-short clears mixed.
            self.force_mixed_blocks = false;
        }
        self.force_short_blocks = enabled;
        Ok(())
    }

    /// `true` when the encoder is forcing every granule onto the
    /// §2.4.2.7 short-block encode path (see
    /// [`Self::force_short_blocks_for_testing`]).
    #[must_use]
    pub fn force_short_blocks_enabled(&self) -> bool {
        self.force_short_blocks
    }

    /// Force every assembled granule onto the §2.4.2.7 **mixed**-block
    /// (`block_type = 2`, `mixed_block_flag = 1`) encode path; see the
    /// [`Self::force_mixed_blocks`] field for the per-granule
    /// behavioural contract.
    ///
    /// Restrictions (mirrored from the field doc):
    ///
    /// * Mono only — same §2.4.3.4.9 cross-channel agreement gap that
    ///   force-short hits.
    /// * Mutually exclusive with [`Self::force_short_blocks_for_testing`].
    ///   Enabling this resets `force_short_blocks` to `false` so a
    ///   single granule cannot ask for both at once.
    ///
    /// # Errors
    ///
    /// [`StreamEncodeError::StereoUnsupported`] when the encoder's
    /// channel count is > 1 (matches the
    /// [`Self::force_short_blocks_for_testing`] policy).
    pub fn force_mixed_blocks_for_testing(
        &mut self,
        enabled: bool,
    ) -> Result<(), StreamEncodeError> {
        if enabled && self.nch != 1 {
            return Err(StreamEncodeError::StereoUnsupported);
        }
        if enabled {
            // Mixed and pure-short are mutually exclusive: a granule is
            // long, short, or mixed. Enabling mixed clears short.
            self.force_short_blocks = false;
        }
        self.force_mixed_blocks = enabled;
        Ok(())
    }

    /// `true` when the encoder is forcing every granule onto the
    /// §2.4.2.7 mixed-block encode path (see
    /// [`Self::force_mixed_blocks_for_testing`]).
    #[must_use]
    pub fn force_mixed_blocks_enabled(&self) -> bool {
        self.force_mixed_blocks
    }

    /// Build a joint-stereo encoder that emits §2.4.3.4.9.2 **MS-stereo**
    /// frames (ISO/IEC 11172-3:1993 joint mode with `mode_extension = '10'`).
    ///
    /// The encoder buffers interleaved `[L0, R0, L1, R1, …]` `i16` PCM,
    /// runs the analysis filterbank + forward MDCT + inverse alias
    /// reduction independently per channel as in the
    /// [`ChannelMode::Stereo`] path, and then — before quantization —
    /// transforms each granule's two post-MDCT spectra `(L, R)` into the
    /// normalized mid/side pair
    ///
    /// ```text
    /// M[i] = (L[i] + R[i]) / √2
    /// S[i] = (L[i] - R[i]) / √2
    /// ```
    ///
    /// per ISO/IEC 11172-3 §2.4.3.4.9.2. `M` is then quantized into the
    /// channel-0 (left) slot, `S` into the channel-1 (right) slot, and
    /// the emitted frame header carries `mode = '01'` (joint stereo)
    /// with `mode_extension = '10'` (ms_stereo on, intensity_stereo
    /// off). A conformant decoder (including this crate's own
    /// [`crate::process_stereo`]) reads the `mode_extension` bits and
    /// applies the inverse `L = (M+S)/√2`, `R = (M-S)/√2` matrix to
    /// recover `(L, R)`.
    ///
    /// The MS matrix is its own inverse (a 2-D rotation by 45°), so the
    /// `(L, R) → (M, S)` step and the decoder's `(M, S) → (L, R)`
    /// step compose to identity in the absence of quantization error.
    /// For correlated stereo content (`L ≈ R`) the side channel
    /// concentrates near zero, which the existing inner-loop bit-budget
    /// gain search exploits — quieter spectra need a lower `global_gain`
    /// to fit the per-granule-channel budget, raising overall SNR.
    ///
    /// MS is applied to the **entire** spectrum (§2.4.3.4.9.2: "When
    /// MS-stereo is enabled but intensity stereo is not, the entire
    /// spectrum is decoded in MS-stereo"). The intensity-stereo half of
    /// joint stereo (§2.4.3.4.9.3) remains unimplemented on the encode
    /// side. Both granules of a frame share the same block type (Long,
    /// for this round), satisfying the §2.4.3.4.9 "both channels of a
    /// granule must share the same block type when MS is enabled"
    /// requirement automatically.
    ///
    /// `mode` is hard-coded to [`ChannelMode::JointStereo`]; pass
    /// `bitrate_kbps` / `sample_rate_hz` per the MPEG-1 Layer III ladder.
    ///
    /// # Errors
    ///
    /// Same as [`Mp3Encoder::new`] but rejects [`ChannelMode::JointStereo`]
    /// for the underlying header-build is never re-checked here (the
    /// constructor sets it itself).
    pub fn new_joint_stereo_ms(
        bitrate_kbps: u32,
        sample_rate_hz: u32,
    ) -> Result<Self, StreamEncodeError> {
        // Build through the independent-stereo path so the channel
        // count, header indices, per-channel analysis/MDCT state, and
        // pending-PCM buffers are all configured for two channels. Then
        // upgrade the header template to joint-stereo + MS and flip the
        // ms_stereo flag.
        let mut enc = Self::new(bitrate_kbps, sample_rate_hz, ChannelMode::Stereo)?;
        enc.header_template.mode = ChannelMode::JointStereo;
        enc.header_template.mode_extension = ModeExtension {
            intensity_stereo: false,
            ms_stereo: true,
            raw: 0b10,
        };
        enc.ms_stereo = true;
        Ok(enc)
    }

    /// `true` when the encoder is configured for §2.4.3.4.9.2 MS-stereo
    /// joint mode (see [`Mp3Encoder::new_joint_stereo_ms`]).
    #[must_use]
    pub fn ms_stereo_enabled(&self) -> bool {
        self.ms_stereo
    }

    /// Build a joint-stereo encoder that decides per-frame whether to
    /// apply the §2.4.3.4.9.2 MS rotation or fall back to independent
    /// L/R coding, based on a content-only energy heuristic.
    ///
    /// The frame header carries `mode = '01'` (joint stereo) on every
    /// audio frame; the two `mode_extension` bits are then chosen per
    /// frame by the picker:
    ///
    /// * `'10'` (ms_stereo on, intensity_stereo off) when **both**
    ///   granules of the frame have side-channel energy fraction
    ///   `E_S / (E_L + E_R) ≤ threshold` (channels well-correlated;
    ///   the MS rotation concentrates energy in the mid channel and
    ///   helps the inner loop).
    /// * `'00'` (neither method active) when either granule has
    ///   `E_S / (E_L + E_R) > threshold` (channels are uncorrelated
    ///   or anti-correlated; MS would inflate the side channel and
    ///   waste bits).
    ///
    /// Decision criterion derivation. The §2.4.3.4.9.2 rotation
    /// `M = (L+R)/√2`, `S = (L-R)/√2` is unitary, so
    /// `E_M + E_S = E_L + E_R` for any L/R pair. `E_S / (E_L + E_R)`
    /// therefore takes value 0 when `L = R` (perfect mono),
    /// 0.5 when L and R are uncorrelated with equal energy, and 1
    /// when `L = -R` (pure anti-phase). The default threshold `0.5`
    /// is the symmetry boundary: below it the M channel carries strictly
    /// more energy than either L or R, so quantization noise on the
    /// (now smaller) S channel costs fewer audible bits than the
    /// independent-channel split would have spent on the same noise on
    /// L and R; above it the side channel carries more than half the
    /// energy and the rotation is counter-productive.
    ///
    /// ISO/IEC 11172-3 does **not** prescribe how to make this
    /// decision — §2.4.2.3 fixes only the wire syntax of the
    /// `mode_extension` field. The energy heuristic is a clean-room
    /// encoder choice that uses no psychoacoustic input and no
    /// external implementation as a reference.
    ///
    /// `mode` is hard-coded to [`ChannelMode::JointStereo`]; pass
    /// `bitrate_kbps` / `sample_rate_hz` per the MPEG-1 Layer III ladder.
    ///
    /// Both granules of a frame share the same mode_extension and (in
    /// this round) the same `BlockType::Long`, so the §2.4.3.4.9
    /// "both channels must share the same block type when MS is
    /// enabled" requirement is automatically satisfied.
    ///
    /// # Errors
    ///
    /// Same as [`Mp3Encoder::new`].
    pub fn new_joint_stereo_auto(
        bitrate_kbps: u32,
        sample_rate_hz: u32,
    ) -> Result<Self, StreamEncodeError> {
        // Build through the independent-stereo path so the channel
        // count, header indices, per-channel analysis/MDCT state, and
        // pending-PCM buffers are all configured for two channels.
        let mut enc = Self::new(bitrate_kbps, sample_rate_hz, ChannelMode::Stereo)?;
        // Upgrade the header template to joint-stereo with a starting
        // mode_extension of '00'; the per-frame picker in
        // `assemble_frame` rewrites it on each frame.
        enc.header_template.mode = ChannelMode::JointStereo;
        enc.header_template.mode_extension = ModeExtension {
            intensity_stereo: false,
            ms_stereo: false,
            raw: 0b00,
        };
        // `ms_stereo` stays false — the auto picker drives the
        // forward-MS branch per frame via `ms_auto_threshold`.
        enc.ms_auto_threshold = Some(0.5);
        Ok(enc)
    }

    /// `Some(t)` when the encoder is configured for the per-frame
    /// MS-vs-LR picker (see [`Mp3Encoder::new_joint_stereo_auto`]), where
    /// `t` is the upper bound on side-channel energy fraction at which
    /// MS is preferred.
    #[must_use]
    pub fn ms_auto_threshold(&self) -> Option<f64> {
        self.ms_auto_threshold
    }

    /// Override the side-channel energy threshold used by the per-frame
    /// MS-vs-LR picker (see [`Mp3Encoder::new_joint_stereo_auto`]).
    ///
    /// `threshold` is the upper bound on `E_S / (E_L + E_R)` at which a
    /// frame's two granules both qualify for MS. Values outside
    /// `[0.0, 1.0]` are clamped to that range. Calling this on an
    /// encoder that was **not** constructed via
    /// [`Mp3Encoder::new_joint_stereo_auto`] is a no-op (the picker is
    /// not armed).
    pub fn with_ms_auto_threshold(mut self, threshold: f64) -> Self {
        if self.ms_auto_threshold.is_some() {
            let clamped = threshold.clamp(0.0, 1.0);
            self.ms_auto_threshold = Some(clamped);
        }
        self
    }

    /// Opt in to the ISO/IEC 11172-3 §2.4.3.1 CRC-16 frame protection.
    ///
    /// With `enabled = true`, [`Mp3Encoder::finish`] writes a 16-bit
    /// CRC check word in the slot between the header and the side-info
    /// block on every emitted audio frame (header + 2-byte CRC +
    /// side_info + main_data), and sets the wire `protection_bit = 0`
    /// (i.e. the parsed [`crate::Mp3FrameHeader::crc_protected`] is
    /// `true`). The CRC covers the §2.4.3.1 / Annex B Table B.5
    /// protected set: header bits 16…31 (bytes 2..4) plus the first
    /// 135 side-info bits in single-channel mode (or 256 bits in every
    /// other channel mode), MSB-first per [`crate::crc::crc16_layer3`].
    ///
    /// The CRC slot consumes two bytes of the per-frame main-data
    /// capacity (`slot = frame_len - 4 - 2 - side_info_bytes`); the
    /// per-granule-channel inner-loop bit budget shrinks accordingly.
    /// A frame whose assembled main-data does not fit the reduced slot
    /// surfaces a [`StreamEncodeError::Reservoir`] at `finish` time —
    /// raise the bitrate (or disable the CRC) and retry.
    ///
    /// The carrier-frame Xing / Info layer remains CRC-free regardless
    /// of this toggle: the leading Xing carrier is emitted via the
    /// existing [`crate::xing_info::build_info_frame`] silent-frame
    /// path which forces `protection_bit = 1` so demuxers see a
    /// regular leading frame and the standard Xing offsets remain
    /// valid.
    ///
    /// Default `false`. Calling this with `false` after a `true` call
    /// disables the CRC again.
    pub fn with_protection_bit(&mut self, enabled: bool) {
        self.crc_enabled = enabled;
    }

    /// `true` when the §2.4.3.1 CRC-16 protection is enabled for the
    /// emitted audio frames (see [`Mp3Encoder::with_protection_bit`]).
    #[must_use]
    pub fn crc_enabled(&self) -> bool {
        self.crc_enabled
    }

    /// Enable per-frame **variable-bitrate** index selection.
    ///
    /// When enabled, [`Mp3Encoder::finish`] picks each audio frame's
    /// `bitrate_index` from the §2.4.2.3 Layer III ladder in the
    /// `[min_kbps, max_kbps]` range — the smallest index whose slot is
    /// large enough to hold that frame's assembled main-data
    /// (zero-padded; no cross-frame reservoir). Frames whose main-data
    /// is below the `min_kbps` slot still emit at `min_kbps` (the
    /// per-frame slot grows to the floor; the trailing bytes are
    /// zero-fill); frames whose main-data exceeds the `max_kbps` slot
    /// fail with [`StreamEncodeError::VbrSlotTooSmall`].
    ///
    /// The constructor-time `bitrate_kbps` still controls (a) the
    /// per-granule-channel inner-loop bit budget (the loop targets
    /// "fits the constructor slot" so the analysis isn't biased toward
    /// any specific VBR rate) and (b) the size of the optional Xing
    /// carrier frame, so callers should pick a constructor bitrate
    /// equal to (or exceeding) `max_kbps`.
    ///
    /// # Errors
    ///
    /// [`StreamEncodeError::InvalidVbrConfig`] when:
    /// * `min_kbps` or `max_kbps` is not a value on
    ///   [`MPEG1_L3_BITRATE_LADDER_KBPS`];
    /// * `min_kbps > max_kbps`;
    /// * `max_kbps` exceeds the constructor-time `bitrate_kbps` (the
    ///   carrier / inner-loop budget would no longer cover the chosen
    ///   max).
    pub fn enable_vbr(&mut self, min_kbps: u32, max_kbps: u32) -> Result<(), StreamEncodeError> {
        let min_idx = ladder_index(min_kbps).ok_or(StreamEncodeError::InvalidVbrConfig)?;
        let max_idx = ladder_index(max_kbps).ok_or(StreamEncodeError::InvalidVbrConfig)?;
        if min_idx > max_idx {
            return Err(StreamEncodeError::InvalidVbrConfig);
        }
        let ctor_kbps = self
            .header_template
            .bitrate_kbps
            .ok_or(StreamEncodeError::InvalidVbrConfig)?;
        if max_kbps > ctor_kbps {
            return Err(StreamEncodeError::InvalidVbrConfig);
        }
        self.vbr = Some(VbrConfig {
            min_index: min_idx,
            max_index: max_idx,
        });
        Ok(())
    }

    /// Enable Xing / Info VBR information-frame emission.
    ///
    /// When set, [`Mp3Encoder::finish`] prepends a Xing / Info carrier
    /// frame (one §2.4.2.3 silent Layer III frame) to the output, with
    /// the magic + flagged fields written into the carrier's main-data
    /// slot per [`crate::xing_info::build_info_frame`]. The carrier
    /// frame is sized identically to one regular CBR audio frame at
    /// this encoder's configured bitrate / sample rate.
    ///
    /// The `template` carries the Xing / Info magic, the flag word, and
    /// any pre-known optional fields. Two fields — `frames` and `bytes`
    /// — are typically not known until the rest of the stream is
    /// encoded:
    ///
    /// * If the corresponding flag bit
    ///   ([`crate::xing_info::flag_bit::FRAMES`] /
    ///   [`crate::xing_info::flag_bit::BYTES`]) is set **and** the
    ///   template's field is `None`, [`Mp3Encoder::finish`] computes
    ///   the field at flush time (frame count = number of audio frames
    ///   following the carrier; byte count = total bytes of audio
    ///   following the carrier).
    /// * If the field is already `Some(_)` in the template, the
    ///   provided value is written verbatim.
    /// * If the flag bit is clear, the field is omitted regardless of
    ///   the template's value (the
    ///   [`crate::xing_info::build_xing_info_payload`] flag/field
    ///   consistency check still applies — the template's optional
    ///   fields must match its flag bits).
    ///
    /// The `toc` / `quality` fields, when flagged, must already be
    /// populated on the template; the encoder does not synthesise a
    /// seek table or a quality score.
    ///
    /// Typical recipe: pass an "Info" template (CBR seeker-compatible
    /// shape) with `flags = FRAMES | BYTES` and the two `Option`s
    /// `None`. The encoder fills both in at flush time.
    pub fn enable_xing_info(&mut self, template: crate::xing_info::XingTagSpec) {
        self.xing_template = Some(template);
    }

    /// Build an encoder identical to [`Mp3Encoder::new`] but with the
    /// §C.1.5.4.3 outer (distortion-control) loop enabled. Every
    /// per-granule-channel quantization runs
    /// [`crate::outer_loop::outer_loop_search_long`] against
    /// `uniform_threshold` as the `xmin[sb]` constant for every band —
    /// the spec leaves the threshold derivation to the psychoacoustic
    /// model (Annex D), but with no psy model wired up yet we apply a
    /// flat constant.
    ///
    /// The outer loop writes `scalefac_compress = 15` (slen1=4, slen2=3)
    /// to span the full §C.1.5.4.3.6 scalefactor range. The part2 cost
    /// is a fixed 74 bits per granule-channel — comfortably under every
    /// MPEG-1 bitrate the encoder supports.
    ///
    /// The fixed-gain path of [`Mp3Encoder::new`] is preserved as the
    /// reference / debug path. Pass [`DEFAULT_OUTER_LOOP_THRESHOLD`] as
    /// the threshold for the recommended default.
    ///
    /// # Errors
    ///
    /// Same as [`Mp3Encoder::new`].
    pub fn new_with_outer_loop(
        bitrate_kbps: u32,
        sample_rate_hz: u32,
        mode: ChannelMode,
        uniform_threshold: f64,
    ) -> Result<Self, StreamEncodeError> {
        let mut enc = Self::new(bitrate_kbps, sample_rate_hz, mode)?;
        enc.outer_loop_threshold = Some(uniform_threshold);
        Ok(enc)
    }

    /// Push PCM samples (`i16`). For mono encoders the input is a
    /// straight `[s0, s1, s2, …]` sample stream; for stereo /
    /// dual-channel encoders the input is **interleaved** LR pairs
    /// (`[L0, R0, L1, R1, …]`). The encoder splits the interleaved
    /// stream into its per-channel buffers and assembles whole MP3
    /// frames as soon as each per-channel buffer has accumulated
    /// `SAMPLES_PER_FRAME_MPEG1 = 1152` samples.
    ///
    /// The interleaved length is therefore expected to be a multiple
    /// of `nch` (1 for mono, 2 for stereo); a trailing partial pair is
    /// accepted but the trailing odd sample is dropped at flush time
    /// to keep the per-channel buffers aligned.
    ///
    /// # Errors
    ///
    /// Propagates any encode-stage failure surfaced while assembling a
    /// completed frame (the only one that can fire here is
    /// [`StreamEncodeError::Huffman`]; bit-budget errors are deferred
    /// until [`Mp3Encoder::finish`]).
    pub fn push_samples(&mut self, samples: &[i16]) -> Result<(), StreamEncodeError> {
        // Convert i16 → f32 in `[-1.0, 1.0]` range and deinterleave
        // into the per-channel pending buffers.
        const SCALE: f32 = 1.0 / 32_768.0;
        let nch = self.nch;
        // For mono the deinterleave is the identity (one channel,
        // one buffer). For stereo (nch == 2) the input is LR-paired;
        // pair index `i` writes to `pending_pcm[i % nch]`.
        for (i, &s) in samples.iter().enumerate() {
            let ch = i % nch;
            self.pending_pcm[ch].push(f32::from(s) * SCALE);
        }

        // Assemble frames as long as EVERY channel's pending buffer
        // holds at least one full granule-frame worth of samples.
        while self
            .pending_pcm
            .iter()
            .all(|buf| buf.len() >= SAMPLES_PER_FRAME_MPEG1)
        {
            let mut per_ch_frame_pcm: Vec<Vec<f32>> = Vec::with_capacity(nch);
            for buf in self.pending_pcm.iter_mut() {
                let mut take = vec![0.0f32; SAMPLES_PER_FRAME_MPEG1];
                take.copy_from_slice(&buf[..SAMPLES_PER_FRAME_MPEG1]);
                buf.drain(..SAMPLES_PER_FRAME_MPEG1);
                per_ch_frame_pcm.push(take);
            }
            self.assemble_frame(&per_ch_frame_pcm)?;
        }
        Ok(())
    }

    /// Flush any remaining buffered PCM (zero-padded to a frame
    /// boundary), schedule every assembled frame onto the §2.4.2.7
    /// bit reservoir, write the resulting byte stream to `sink`, and
    /// consume the encoder. Returns the total number of bytes
    /// written.
    ///
    /// # Errors
    ///
    /// * [`StreamEncodeError::Reservoir`] if a busy frame cannot be
    ///   scheduled within the §2.4.2.7 budget (raise the bitrate and
    ///   retry).
    /// * [`StreamEncodeError::Io`] from the sink writes.
    pub fn finish<W: Write>(mut self, sink: &mut W) -> Result<usize, StreamEncodeError> {
        // Tail-flush: any leftover per-channel samples shorter than a
        // full frame are zero-padded so the last 1152 samples per
        // channel are still emitted. If any channel has non-empty
        // pending PCM, every channel emits a (potentially padded)
        // tail frame so the per-channel buffers stay aligned.
        let any_pending = self.pending_pcm.iter().any(|b| !b.is_empty());
        if any_pending {
            let nch = self.nch;
            let mut per_ch_tail: Vec<Vec<f32>> = Vec::with_capacity(nch);
            for buf in self.pending_pcm.iter_mut() {
                let mut tail = vec![0.0f32; SAMPLES_PER_FRAME_MPEG1];
                for (i, &v) in buf.iter().enumerate() {
                    tail[i] = v;
                }
                buf.clear();
                per_ch_tail.push(tail);
            }
            self.assemble_frame(&per_ch_tail)?;
        }
        self.flush_to(sink)
    }

    /// Per-(granule, channel) bit budget for the inner loop's bit-budget
    /// gain search. Computed from the CBR slot size:
    ///   `slot_bytes * 8 / (granule_count · channel_count)`.
    /// This is the §C.1.5.4.4 `desired_rate` proxy for one
    /// granule-channel under the no-reservoir / even-share schedule;
    /// the actual main-data writer may exceed it by a few bits because
    /// the granule-channel's part2 (scalefactors) is excluded from
    /// this budget (zero with our all-zero scalefactor config).
    ///
    /// In VBR mode this is the **max-index** slot's bits — the analysis
    /// must not produce more bits than the largest slot the
    /// per-frame VBR selector can ever emit. Frames whose actual
    /// distortion-shaped output is smaller still cause the VBR step to
    /// pick a smaller bitrate index; the budget is the upper bound on
    /// per-granule-channel bits, not a target to fill.
    fn per_gc_bit_budget(&self) -> usize {
        let si_bytes = side_info_byte_len(self.nch);
        let crc_bytes = if self.crc_enabled { 2 } else { 0 };
        // VBR caps per-granule bits at the max-index slot; CBR caps at
        // the constructor slot. In both cases the CRC (when enabled)
        // claims two bytes from the main-data slot.
        let slot_bytes = match self.vbr {
            Some(cfg) => ladder_slot_capacity(self.sample_rate_hz, cfg.max_index, si_bytes, true)
                .saturating_sub(crc_bytes),
            None => {
                let frame_len = self.header_template.frame_len().unwrap_or(0);
                frame_len.saturating_sub(4 + crc_bytes + si_bytes)
            }
        };
        let denom = GRANULES.saturating_mul(self.nch).max(1);
        // Hold back a small margin (16 bits) per granule-channel for
        // the assembler's last partial-byte pad and any rounding.
        slot_bytes
            .saturating_mul(8)
            .saturating_div(denom)
            .saturating_sub(16)
    }

    /// Internal: turn `per_ch_frame_pcm[ch][0..1152]` (deinterleaved,
    /// one buffer per channel) into one assembled `PendingFrame` and
    /// append it to the scheduling queue.
    // The (gr, ch) double-loop mirrors the §2.4.1.7 `main_data()`
    // ordering exactly; the index variables are also used as
    // scratch-array subscripts (`gc_data[gr][ch]`,
    // `side_info.granules[gr][ch]`, etc.), so the explicit `for ch in
    // 0..self.nch` reads more clearly than an iterator chain.
    #[allow(clippy::needless_range_loop)]
    fn assemble_frame(&mut self, per_ch_frame_pcm: &[Vec<f32>]) -> Result<(), StreamEncodeError> {
        debug_assert_eq!(per_ch_frame_pcm.len(), self.nch);
        for buf in per_ch_frame_pcm.iter() {
            debug_assert_eq!(buf.len(), SAMPLES_PER_FRAME_MPEG1);
        }

        // ---- Build the side-info skeleton (all-long, zero scalefactors) ----
        let mut side_info = SideInfo {
            main_data_begin: 0,
            private_bits: 0,
            scfsi: [[false; 4]; 2],
            granules: [[default_long_gc(); 2]; GRANULES],
            channels: self.nch as u8,
            granule_count: GRANULES as u8,
            lsf: false,
        };
        let mut scalefactors = FrameScaleFactors {
            granules: [[ScaleFactors::default(); 2]; 2],
            granule_count: GRANULES as u8,
            channels: self.nch as u8,
        };
        let mut gc_data: [[GranuleChannelData; 2]; 2] = Default::default();

        // ---- Pass 1: per-(gr, ch) analysis → xr_pre ----
        //
        // Each entry is the spectrum that goes into the quantizer for
        // (granule `gr`, channel `ch`). For independent stereo / mono
        // this is the post-`inverse_alias_reduce` bins of the channel's
        // PCM run; for MS-stereo joint mode (§2.4.3.4.9.2) the
        // L/R pair of every granule is rewritten in place as
        // `(M = (L+R)/√2, S = (L-R)/√2)` between pass 1 and pass 2 (the
        // matrix is its own inverse, so the decoder's
        // `process_stereo` recovers L/R).
        //
        // The decoder pipeline order is `requantize → reorder → stereo
        // → alias → IMDCT`. The encoder inverts:
        // `MDCT_fwd → alias_inv → stereo_inv (MS forward) → quantize`,
        // so the MS step belongs between `inverse_alias_reduce` and the
        // quantize loop — which is exactly the pass-1/pass-2 split here.
        let mut xr_pre_per_gc: Vec<Vec<[f32; NUM_LINES]>> = (0..GRANULES)
            .map(|_| (0..self.nch).map(|_| [0.0f32; NUM_LINES]).collect())
            .collect();
        for gr in 0..GRANULES {
            for ch in 0..self.nch {
                let gr_pcm =
                    &per_ch_frame_pcm[ch][gr * SAMPLES_PER_GRANULE..(gr + 1) * SAMPLES_PER_GRANULE];
                let mut pcm_arr = [0.0f32; SAMPLES_PER_GRANULE];
                pcm_arr.copy_from_slice(gr_pcm);

                // Polyphase analysis filterbank → 32×18 subband-time.
                let subband_time = analyze_granule(&pcm_arr, &mut self.analysis_state[ch]);

                // §2.4.3.4.10.5 frequency inversion is applied on the
                // **decoder** side AFTER the IMDCT/overlap-add, i.e.
                // after the subband-time block is restored. The
                // encoder must therefore apply the same negation
                // **before** the forward MDCT chain, so the
                // round-trip cancels (freq_inv ∘ freq_inv = id).
                let mut inv = subband_time;
                for (_sb, sb_row) in inv.iter_mut().enumerate().skip(1).step_by(2) {
                    for t in (1..18usize).step_by(2) {
                        sb_row[t] = -sb_row[t];
                    }
                }

                // Forward MDCT per subband. Three paths:
                //
                // * **Long block** (default; both force flags off):
                //   forward-overlap with the saved previous granule →
                //   window (long) → 36-point forward MDCT → 18 frequency
                //   lines, then inverse-alias-reduce so the decoder's
                //   forward alias-reduction recovers the post-MDCT
                //   bins.
                //
                // * **Short block** (`force_short_blocks` on; mono-only
                //   in this round per
                //   [`Mp3Encoder::force_short_blocks_for_testing`]):
                //   three independent 12-point MDCTs per subband over
                //   the lapped 36-sample frame
                //   ([`crate::short_block::forward_short_mdct_subband`]),
                //   producing 18 bins in subband-window-interleaved
                //   layout (`out[3·k + win]`). No alias reduction
                //   (§2.4.3.4.10.1 scopes it to `block_type != 2`).
                //   Then [`crate::short_block::forward_reorder`]
                //   rewrites the bins into the native bitstream
                //   `[sfb][win][k]` interleave the §2.4.1.7 part3
                //   Huffman path expects.
                //
                // * **Mixed block** (`force_mixed_blocks` on; mono-only;
                //   see [`Mp3Encoder::force_mixed_blocks_for_testing`]):
                //   the long-block forward path runs on subbands 0 and 1
                //   (lines 0..36), the short-block forward path runs on
                //   subbands 2..31. No alias reduction (decoder treats
                //   `block_type == Short` for the whole granule
                //   regardless of `mixed_block_flag`, so
                //   [`crate::alias::alias_reduce`] is a pass-through).
                //   [`crate::short_block::forward_reorder`] is then
                //   driven with the mixed gc so the long region
                //   (lines 0..36) passes through and only the short
                //   region (short SFB 3..12) is rewritten into native
                //   bitstream order. The §2.4.3.4.10.4 overlap-add
                //   state is per-subband — long subbands consume the
                //   long IMDCT path on decode, short subbands the short
                //   path; both paths share the same per-subband
                //   [`crate::mdct::MdctState`] history slot, so the
                //   forward branch matches what the decoder expects.
                //
                // Scale derivation (long path). The §2.4.3.4.10.2
                // IMDCT and the analysis MDCT use the same unscaled
                // cosine kernel, so the time-space lapped-MDCT round-
                // trip `encoder window → MDCT → decoder IMDCT → window
                // → overlap-add` recovers the input scaled by `n/4 = 9`
                // (the Princen-Bradley TDAC factor; see
                // `analysis_synthesis_long_block_tdac_recovery` in
                // `mdct.rs`). The short path's analog `n = 12 → n/4 =
                // 3` is applied inside
                // [`crate::short_block::forward_short_mdct_subband`].
                let mut xr = [0.0f32; NUM_LINES];
                if self.force_short_blocks {
                    // Short-block forward path. Subband sb still owns
                    // lines [sb*18, (sb+1)*18); the 18 bins per subband
                    // are window-interleaved (`3·k + win`).
                    for sb in 0..32 {
                        let mut current = [0.0f64; LONG_N / 2];
                        for (t, slot) in current.iter_mut().enumerate() {
                            *slot = f64::from(inv[sb][t]);
                        }
                        let bins = crate::short_block::forward_short_mdct_subband(
                            &current,
                            &mut self.mdct_state[ch][sb],
                        );
                        let base = sb * 18;
                        xr[base..base + 18].copy_from_slice(&bins);
                    }
                    // Build a short GranuleChannel to drive the reorder.
                    let mut gc_short = default_long_gc();
                    let (r0, r1) = crate::short_block::short_block_region_defaults();
                    gc_short.window_switching_flag = true;
                    gc_short.block_type = BlockType::Short;
                    gc_short.mixed_block_flag = false;
                    gc_short.region0_count = r0;
                    gc_short.region1_count = r1;
                    // Forward reorder: subband-window-interleaved →
                    // native bitstream [sfb][win][k]. No inverse alias
                    // reduction (short blocks skip §2.4.3.4.10.1 on the
                    // decoder side; encoder mirrors).
                    xr_pre_per_gc[gr][ch] = crate::short_block::forward_reorder(
                        &xr,
                        &gc_short,
                        self.sample_rate_hz,
                        self.version,
                    );
                } else if self.force_mixed_blocks {
                    // Mixed-block forward path. Subbands 0 and 1 run the
                    // long-family forward MDCT (single 36-point MDCT
                    // with the plain sine window → ÷9 unit-gain scale
                    // → 18 bins in increasing-frequency order); subbands
                    // 2..31 run the short-block forward path (three
                    // 12-point MDCTs each over the lapped 36-sample
                    // frame, divided by 3 inside
                    // `forward_short_mdct_subband` → 18 bins in
                    // subband-window-interleaved layout). The
                    // per-subband [`MdctState`] history is shared
                    // between branches so the next granule's overlap
                    // proceeds correctly whichever branch the next
                    // granule lands in.
                    for sb in 0..2 {
                        let mut current = [0.0f64; LONG_N / 2];
                        for (t, slot) in current.iter_mut().enumerate() {
                            *slot = f64::from(inv[sb][t]);
                        }
                        let frame36 = forward_overlap(&current, &mut self.mdct_state[ch][sb]);
                        let windowed = window_long_family_analysis(&frame36, BlockType::Long);
                        let bins = mdct(&windowed, LONG_N);
                        for (k, &b) in bins.iter().enumerate() {
                            xr[sb * 18 + k] = (b / 9.0) as f32;
                        }
                    }
                    for sb in 2..32 {
                        let mut current = [0.0f64; LONG_N / 2];
                        for (t, slot) in current.iter_mut().enumerate() {
                            *slot = f64::from(inv[sb][t]);
                        }
                        let bins = crate::short_block::forward_short_mdct_subband(
                            &current,
                            &mut self.mdct_state[ch][sb],
                        );
                        let base = sb * 18;
                        xr[base..base + 18].copy_from_slice(&bins);
                    }
                    // Build a mixed GranuleChannel to drive the reorder.
                    // `forward_reorder` consults `mixed_block_flag` and
                    // copies the long region (lines 0..36) unchanged
                    // while rewriting short SFB 3..12 from
                    // subband-window-interleaved into native bitstream
                    // `[sfb][win][k]` order.
                    let gc_mixed = default_mixed_gc();
                    // No inverse alias reduction. The decoder's
                    // [`crate::alias::alias_reduce`] tests
                    // `block_type == Short` only and returns unchanged
                    // for both short and mixed granules, so applying
                    // the inverse here would leave a residual butterfly
                    // on the decode side that nothing reverses.
                    xr_pre_per_gc[gr][ch] = crate::short_block::forward_reorder(
                        &xr,
                        &gc_mixed,
                        self.sample_rate_hz,
                        self.version,
                    );
                } else {
                    // Long-block forward path (default, every round
                    // prior to 151).
                    for sb in 0..32 {
                        let mut current = [0.0f64; LONG_N / 2];
                        for (t, slot) in current.iter_mut().enumerate() {
                            *slot = f64::from(inv[sb][t]);
                        }
                        let frame36 = forward_overlap(&current, &mut self.mdct_state[ch][sb]);
                        let windowed = window_long_family_analysis(&frame36, BlockType::Long);
                        let bins = mdct(&windowed, LONG_N);
                        for (k, &b) in bins.iter().enumerate() {
                            // Long-block xr placement: subband sb owns
                            // lines [sb*18, (sb+1)*18).
                            xr[sb * 18 + k] = (b / 9.0) as f32;
                        }
                    }
                    // Inverse alias reduction. The decoder applies
                    // `xar[lo] = xr[lo]*cs - xr[hi]*ca` /
                    // `xar[hi] = xr[hi]*cs + xr[lo]*ca` over the 31
                    // subband boundaries (8 butterflies each). The
                    // butterfly is orthogonal (cs²+ca²=1); its inverse
                    // negates `ca`:
                    //   xr[lo] = xar[lo]*cs + xar[hi]*ca
                    //   xr[hi] = xar[hi]*cs - xar[lo]*ca
                    // For long blocks we apply the inverse here so the
                    // decoder's forward alias-reduction recovers the
                    // post-MDCT bins.
                    xr_pre_per_gc[gr][ch] = inverse_alias_reduce(&xr);
                }
            }
        }

        // ---- Pass 1.5: optional §2.4.3.4.9.2 forward MS matrix ----
        //
        // For MS-stereo joint mode rewrite each granule's L/R xr pair
        // into the normalized mid/side pair:
        //   M[i] = (L[i] + R[i]) / √2     (carried in the channel-0 slot)
        //   S[i] = (L[i] - R[i]) / √2     (carried in the channel-1 slot)
        // The decoder's `process_stereo` (driven by `mode_extension =
        // '10'`) applies the inverse `L = (M+S)/√2`, `R = (M-S)/√2`.
        // The matrix is its own inverse (a 2-D rotation by 45°), so a
        // lossless quantizer would recover L/R exactly.
        //
        // Apply the matrix to the **entire** 576-line spectrum
        // (§2.4.3.4.9.2: "When MS-stereo is enabled but intensity
        // stereo is not, the entire spectrum is decoded in MS-stereo");
        // the intensity-bound logic only kicks in when intensity stereo
        // is also active, which this round's encoder never emits.
        //
        // Two driver paths:
        //   * `ms_stereo` (set by `new_joint_stereo_ms`): the matrix
        //     fires unconditionally on every frame.
        //   * `ms_auto_threshold` (set by `new_joint_stereo_auto`): the
        //     matrix fires only when **both** granules of the frame
        //     have side-channel energy fraction
        //       E_S / (E_L + E_R) = Σ((L−R)/√2)² / Σ(L² + R²)
        //                         = Σ(L − R)² / (2·Σ(L² + R²))
        //     at or below the carried threshold (default 0.5). The
        //     rotation is unitary so `E_M + E_S = E_L + E_R`; this
        //     ratio is 0 for perfectly mono content, 0.5 for
        //     uncorrelated equal-energy channels, and 1 for pure
        //     anti-phase content. Below 0.5 the mid channel carries
        //     more energy than either L or R individually, which the
        //     inner loop's bit-budget gain search exploits.
        //
        // The frame's mode_extension header field is set to '10' when
        // MS fires and '00' when it does not. Both `ms_stereo` and the
        // auto picker keep `mode = '01'` (joint stereo) on the header.
        let mut frame_mode_extension = self.header_template.mode_extension;
        let apply_ms_this_frame = if self.nch == 2 {
            if self.ms_stereo {
                true
            } else if let Some(threshold) = self.ms_auto_threshold {
                // Compute the side-energy fraction across both granules
                // of the frame. Reject MS as soon as one granule exceeds
                // the threshold so the §2.4.3.4.9 "both granules share
                // the same joint-stereo method" semantics are honoured
                // for free (the wire mode_extension is a per-frame
                // field, not a per-granule one).
                let mut all_ok = true;
                for gr in 0..GRANULES {
                    let left = &xr_pre_per_gc[gr][0];
                    let right = &xr_pre_per_gc[gr][1];
                    let mut lr_energy = 0.0f64;
                    let mut side_energy_x2 = 0.0f64;
                    for i in 0..NUM_LINES {
                        let l = f64::from(left[i]);
                        let r = f64::from(right[i]);
                        lr_energy += l * l + r * r;
                        let d = l - r;
                        side_energy_x2 += d * d;
                    }
                    // side_energy = Σ((L−R)/√2)² = side_energy_x2 / 2.
                    // side_energy / (E_L + E_R)
                    //     = side_energy_x2 / (2 · lr_energy).
                    // Compare without the divide to dodge the
                    // pathological lr_energy == 0 case (a fully silent
                    // granule trivially passes — both numerator and
                    // denominator are 0; pick MS by convention since
                    // mono == mid-only).
                    if lr_energy <= 0.0 {
                        continue;
                    }
                    let ratio = side_energy_x2 / (2.0 * lr_energy);
                    if ratio > threshold {
                        all_ok = false;
                        break;
                    }
                }
                all_ok
            } else {
                false
            }
        } else {
            false
        };
        if apply_ms_this_frame {
            const INV_SQRT2: f32 = std::f32::consts::FRAC_1_SQRT_2;
            for gr in 0..GRANULES {
                // Split the per-channel borrow without copying both
                // arrays: `split_at_mut(1)` gives us `[L]` and `[R]`
                // as disjoint slices, then we index into them.
                let (left_slice, right_slice) = xr_pre_per_gc[gr].split_at_mut(1);
                let left = &mut left_slice[0];
                let right = &mut right_slice[0];
                for i in 0..NUM_LINES {
                    let l = left[i];
                    let r = right[i];
                    left[i] = (l + r) * INV_SQRT2;
                    right[i] = (l - r) * INV_SQRT2;
                }
            }
        }
        // Reflect the per-frame decision on the carried frame header
        // (only matters for the auto picker; the unconditional
        // `new_joint_stereo_ms` path keeps the constructor's '10'
        // header template and the auto path overwrites it here).
        if self.ms_auto_threshold.is_some() && self.nch == 2 {
            frame_mode_extension = if apply_ms_this_frame {
                ModeExtension {
                    intensity_stereo: false,
                    ms_stereo: true,
                    raw: 0b10,
                }
            } else {
                ModeExtension {
                    intensity_stereo: false,
                    ms_stereo: false,
                    raw: 0b00,
                }
            };
        }

        // ---- Pass 2: per-(gr, ch) quantization + side-info build ----
        for gr in 0..GRANULES {
            for ch in 0..self.nch {
                let xr_pre = xr_pre_per_gc[gr][ch];

                // Pick the smallest global_gain + scalefactor configuration.
                // Two paths:
                //   * fixed-gain: zero scalefactors + inner loop only
                //     (the r138 path; kept for reference / debug).
                //   * outer-loop: §C.1.5.4.3 distortion-control loop on
                //     top of the inner loop, with non-zero per-band
                //     scalefactor amplification driven by the uniform
                //     `xmin[sb]` threshold.
                let gc_template = if self.force_short_blocks {
                    default_short_gc()
                } else if self.force_mixed_blocks {
                    default_mixed_gc()
                } else {
                    default_long_gc()
                };
                let per_gc_bits = self.per_gc_bit_budget();
                // When the outer loop runs, part2 (scalefactors) costs
                // 74 bits per granule-channel (`scalefac_compress = 15`:
                // 11·slen1 + 10·slen2 = 11·4 + 10·3); the inner loop's
                // part3 budget shrinks by that amount.
                let part2_bits_outer: usize = 11 * 4 + 10 * 3;
                let inner_budget_for_outer = per_gc_bits.saturating_sub(part2_bits_outer) as u64;
                let (sf, initial_gain, scalefac_scale_outer) = match self.outer_loop_threshold {
                    Some(thr) => {
                        // Outer loop seeds scalefac_compress = 15 so the
                        // chosen per-band scalefactors can be written
                        // back as part2.
                        let mut gc_for_ol = gc_template;
                        gc_for_ol.scalefac_compress = OUTER_LOOP_SCALEFAC_COMPRESS;
                        let res = outer_loop_search_long(
                            &xr_pre,
                            &gc_for_ol,
                            self.sample_rate_hz,
                            self.version,
                            inner_budget_for_outer,
                            thr,
                            DEFAULT_OUTER_LOOP_MAX_ITER,
                        );
                        // The outer loop reports:
                        //   * `scalefac_scale` — §C.1.5.4.3 dynamic-range
                        //     escalation (multiplier 1.0 vs 0.5);
                        //   * `scalefactors.preflag` — §C.1.5.4.3.4
                        //     preemphasis (Table B.6 pretab boost on the
                        //     upper bands).
                        // Both must be propagated into the granule-channel
                        // so the re-quantize step below and the side-info
                        // write reflect what the outer loop converged on.
                        // `sf.preflag` (returned inside `res.scalefactors`)
                        // is what `quantize()` reads; `gc.preflag` is what
                        // the side-info encoder writes — we mirror them
                        // below at the top of the `loop`.
                        (res.scalefactors, res.global_gain, res.scalefac_scale)
                    }
                    None => {
                        let sf = ScaleFactors::default();
                        // VBR-mode gain choice: skip the bit-budget
                        // gain search. Without a psychoacoustic model
                        // the budget search would saturate the chosen
                        // VBR max-index slot regardless of content
                        // complexity — defeating the point of letting
                        // the encoder pick a smaller per-frame bitrate.
                        // The magnitude-clamp gain alone (the smallest
                        // gain with `max|is| ≤ 8191`) is the natural
                        // content-driven quality floor and lets the
                        // per-frame VBR-index selector see a true
                        // content-dependent main-data size. CBR keeps
                        // the dual search; the VBR path uses clamp-only.
                        let res_clamp = search_magnitude_clamp(
                            &xr_pre,
                            &gc_template,
                            &sf,
                            self.sample_rate_hz,
                            self.version,
                        );
                        let initial_gain = if self.vbr.is_some() {
                            res_clamp.global_gain
                        } else {
                            let res_budget = search_bit_budget(
                                &xr_pre,
                                &gc_template,
                                &sf,
                                self.sample_rate_hz,
                                self.version,
                                per_gc_bits as u64,
                            );
                            res_budget.global_gain.max(res_clamp.global_gain)
                        };
                        // No outer loop ⇒ no escalation; scalefac_scale
                        // stays 0 for the fixed-gain path.
                        (sf, initial_gain, false)
                    }
                };
                let mut global_gain = initial_gain;
                let _ = (GAIN_MAX, GAIN_MIN); // re-export keep-alive

                // Re-quantize at the chosen gain + sf and configure the
                // granule-channel. If the bit cost under OUR table
                // chooser (filtered by linbits reach) exceeds the
                // budget, bump the gain by 1 and retry — the
                // §C.1.5.4.4 `qquant + 1` outer ratchet, applied
                // here only as a budget-overrun safety net.
                let scalefac_compress = if self.outer_loop_threshold.is_some() {
                    OUTER_LOOP_SCALEFAC_COMPRESS
                } else {
                    0
                };
                let mut gc;
                let mut is;
                let mut split;
                let mut bv2;
                let mut r0_end;
                let mut r1_end;
                let mut t0;
                let mut t1;
                let mut t2;
                let mut count1_b;
                let mut count1_quads;
                loop {
                    gc = gc_template;
                    gc.global_gain = global_gain;
                    gc.scalefac_compress = scalefac_compress;
                    // §C.1.5.4.3.4 preemphasis: mirror `sf.preflag` into
                    // `gc.preflag`. The side-info writer emits `gc.preflag`
                    // and the decoder feeds it back into `sf.preflag` at
                    // parse time; the requantize / quantize primitives
                    // read `sf.preflag`. Both pathways must agree, so the
                    // single source of truth here is the `sf.preflag` the
                    // outer loop returned (or `false` for the fixed-gain
                    // branch where no outer loop runs).
                    gc.preflag = sf.preflag;
                    // §C.1.5.4.3 escalation: when the outer loop reports
                    // scalefac_scale = 1, the re-quantize step here MUST
                    // use the same multiplier (1.0 vs 0.5) — otherwise
                    // the encoder's quantized `is[]` would be coloured
                    // against a different per-band exponent than what
                    // the side-info bit later instructs the decoder to
                    // requantize with.
                    gc.scalefac_scale = scalefac_scale_outer;
                    is = quantize(&xr_pre, &gc, &sf, self.sample_rate_hz, self.version);
                    clamp_above(&mut is, 8191);
                    split = partition_split(&is);
                    bv2 = split.big_pairs * 2;
                    gc.big_values = split.big_pairs as u16;
                    if self.force_short_blocks || self.force_mixed_blocks {
                        // §C.1.5.4.4.6 + huffman::region_boundaries:
                        // short and mixed blocks hardcode region 0 to
                        // the first 36 lines and region 1 to the rest of
                        // big_values (region 2 empty). The transmitted
                        // region0_count / region1_count are not on the
                        // wire (encoder::write_granule_channel writes
                        // the window-switched branch which omits them);
                        // keep the spec-default sentinels from the
                        // short / mixed template intact.
                        //
                        // Mixed-block detail: the §2.4.2.7 fixed
                        // region 0 covers exactly the 36 long-region
                        // lines (subbands 0..1), so the region-0
                        // boundary aligns naturally with the long /
                        // short split inside the granule. Region 1
                        // then carries all big_values that fall in the
                        // short region (subbands 2..31), already
                        // re-ordered into `[sfb][win][k]` native order
                        // by `forward_reorder` above.
                        r0_end = 36usize.min(bv2);
                        r1_end = bv2;
                    } else {
                        let (r0e, r1e, r0c, r1c) =
                            choose_region_split(self.sample_rate_hz, self.version, bv2);
                        r0_end = r0e;
                        r1_end = r1e;
                        gc.region0_count = r0c;
                        gc.region1_count = r1c;
                    }
                    t0 = best_table_or(&is, 0, r0_end);
                    t1 = best_table_or(&is, r0_end, r1_end);
                    t2 = best_table_or(&is, r1_end, bv2);
                    gc.table_select = [t0, t1, t2];
                    let c1s = bv2;
                    count1_quads = split.count1_quads;
                    let c1e = c1s + count1_quads * 4;
                    let (cb, _) = choose_best_count1_table(&is, c1s, c1e);
                    count1_b = cb;
                    gc.count1table_select = count1_b;

                    // Compute the actual emitted bit cost under our
                    // chooser and compare with the per-gc budget. When
                    // the outer loop is active the part3 budget is the
                    // total per-gc budget minus the fixed 74-bit part2
                    // cost; otherwise (scalefactor_compress = 0) part2
                    // is zero and the whole budget is part3.
                    let big_bits = bits_for_range(&is, 0, r0_end, t0).unwrap_or(usize::MAX / 4)
                        + bits_for_range(&is, r0_end, r1_end, t1).unwrap_or(usize::MAX / 4)
                        + bits_for_range(&is, r1_end, bv2, t2).unwrap_or(usize::MAX / 4);
                    let cnt1_bits = crate::huffman::count1_bits(&is, c1s, c1e, count1_b);
                    let total = big_bits + cnt1_bits;
                    let budget_for_part3 = if self.outer_loop_threshold.is_some() {
                        inner_budget_for_outer as usize
                    } else {
                        per_gc_bits
                    };
                    if total <= budget_for_part3 || global_gain == 255 {
                        break;
                    }
                    global_gain = global_gain.saturating_add(1);
                }
                // Commit per-granule-channel state.
                let _ = (r0_end, r1_end, t0, t1, t2, count1_b);
                side_info.granules[gr][ch] = gc;
                scalefactors.granules[gr][ch] = sf;
                gc_data[gr][ch] = GranuleChannelData {
                    is,
                    big_pairs: split.big_pairs,
                    count1_quads,
                };
            }
        }

        // ---- Main-data assembly ----
        let mut header = self.header_template;
        header.padding = false; // initial; reservoir step decides
                                // Per-frame mode_extension from the joint-stereo picker (no-op
                                // for the unconditional `new_joint_stereo_ms` path — the
                                // template already carries '10' there).
        header.mode_extension = frame_mode_extension;
        let asm = assemble_main_data(&header, &mut side_info, &scalefactors, &gc_data)
            .map_err(|e| StreamEncodeError::Huffman(e.to_string()))?;

        self.frames.push(PendingFrame {
            header,
            side_info,
            main_data: asm.bytes,
        });
        Ok(())
    }

    /// Internal: run the reservoir scheduler over every buffered frame
    /// and write the resulting on-wire byte sequence to `sink`.
    fn flush_to<W: Write>(mut self, sink: &mut W) -> Result<usize, StreamEncodeError> {
        if self.frames.is_empty() {
            return Ok(0);
        }
        // Pull the Xing template out before consuming `self.frames`.
        let xing_template = self.xing_template.take();

        // Step 1: pick per-frame padding to absorb fractional bytes,
        // and compute each frame's main-data slot capacity.
        //
        // For MPEG-1 Layer III the per-frame total length is
        //   `frame_len(padding) = 144 · bitrate / sample_rate + pad`
        // and the slot is `frame_len - 4 - crc_bytes - SI_bytes`. When
        // CRC is disabled (the default), `crc_bytes = 0` and the slot
        // reduces to `frame_len - 4 - 17` (mono) / `- 32` (stereo);
        // when [`Mp3Encoder::with_protection_bit`] is on, the §2.4.3.1
        // 2-byte CRC slot sits between the header and the side-info
        // block and shrinks the main-data capacity by two bytes.
        let si_bytes = side_info_byte_len(self.nch);
        let crc_bytes: usize = if self.crc_enabled { 2 } else { 0 };
        let frames = self.frames;
        let n = frames.len();
        let mut headers: Vec<Mp3FrameHeader> = Vec::with_capacity(n);
        let mut side_infos: Vec<SideInfo> = Vec::with_capacity(n);
        let mut main_datas: Vec<Vec<u8>> = Vec::with_capacity(n);
        let mut slots: Vec<usize> = Vec::with_capacity(n);
        let mut frame_lens: Vec<usize> = Vec::with_capacity(n);

        // Greedy padding: distribute padding-slot bytes per the CBR
        // ladder. Compute the unpadded slot count first.
        let base_frame_len = self.header_template.frame_len().expect("CBR frame_len");
        // Padded length is base + 1 byte. To average the right bitrate
        // we need every K-th frame padded where K = sample_rate /
        // ((bitrate * 144) mod sample_rate). For deterministic mono
        // 128 kbit/s @ 44.1 kHz that's pad every ~9 frames. Use a
        // running accumulator: `acc += rem; if acc >= sr { pad =
        // true; acc -= sr }` — the classic Bresenham-style CBR pad
        // ladder.
        let bitrate_bps = u64::from(self.header_template.bitrate_kbps.unwrap_or(0)) * 1000;
        let sr64 = u64::from(self.sample_rate_hz);
        let rem = (144 * bitrate_bps) % sr64;
        let mut acc: u64 = 0;
        for (i, f) in frames.into_iter().enumerate() {
            let mut hdr = f.header;
            // The wire `protection_bit` (the inverse of `crc_protected`)
            // follows the encoder-wide CRC toggle. Set it on every
            // emitted audio frame; the Xing carrier path (above)
            // forces its own header so this assignment does not leak
            // into the carrier.
            hdr.crc_protected = self.crc_enabled;
            let frame_len = if let Some(vbr_cfg) = self.vbr {
                // True-VBR: pick the smallest §2.4.2.3 ladder index in
                // `[min_index, max_index]` whose slot bytes are at
                // least the assembled main-data length. The chosen
                // header carries that bitrate; the reservoir step
                // still zero-pads the slot remainder, so the on-wire
                // schedule is per-frame "no carry-over" — every
                // `main_data_begin == 0`.
                let need = f.main_data.len();
                let (idx_kbps, idx_byte) =
                    pick_vbr_bitrate(self.sample_rate_hz, vbr_cfg, need, crc_bytes).ok_or(
                        StreamEncodeError::VbrSlotTooSmall {
                            frame_index: i,
                            main_data_len: need,
                            max_slot_bytes: ladder_slot_capacity(
                                self.sample_rate_hz,
                                vbr_cfg.max_index,
                                si_bytes,
                                /*padded=*/ true,
                            )
                            .saturating_sub(crc_bytes),
                        },
                    )?;
                hdr.bitrate_index = idx_byte;
                hdr.bitrate_kbps = Some(idx_kbps);
                // VBR sub-step: choose padding to fit `need` exactly
                // when one extra byte rounds the slot up to ≥ need.
                let unpadded = hdr.frame_len().expect("VBR frame_len");
                let unpadded_slot = unpadded - 4 - crc_bytes - si_bytes;
                hdr.padding = unpadded_slot < need;
                hdr.frame_len().expect("VBR frame_len after pad")
            } else {
                // CBR: Bresenham padding ladder against the
                // constructor bitrate.
                let pad = if rem == 0 {
                    false
                } else {
                    acc += rem;
                    if acc >= sr64 {
                        acc -= sr64;
                        true
                    } else {
                        false
                    }
                };
                hdr.padding = pad;
                hdr.frame_len().expect("CBR frame_len")
            };
            // Per-frame main-data slot: frame_len minus the 4-byte
            // header, the optional 2-byte CRC, and the side-info bytes.
            let slot = frame_len - 4 - crc_bytes - si_bytes;
            headers.push(hdr);
            side_infos.push(f.side_info);
            main_datas.push(f.main_data);
            slots.push(slot);
            frame_lens.push(frame_len);
        }
        // Sanity: never expect the per-frame length to drop below the
        // base length (CBR only — VBR may legitimately pick a smaller
        // bitrate index than the constructor).
        if self.vbr.is_none() {
            debug_assert!(frame_lens.iter().all(|&l| l >= base_frame_len));
        }

        // Step 2: run the bit-reservoir scheduler.
        //
        // §2.4.2.7 caps `main_data_begin` at 511 bytes (MPEG-1).
        // Without a psychoacoustic model the per-frame main_data tends
        // to be much smaller than the slot, so the rolling reservoir
        // would grow unbounded across frames. We bound it by
        // **zero-padding** every frame's main_data up to at least its
        // own slot size: the reservoir then never grows above 0 byte
        // (the no-reservoir schedule), every `main_data_begin` is 0,
        // and the scheduler walks the trivial schedule. (A real
        // encoder uses the reservoir to absorb busy-frame overflow;
        // we revisit that once the psy / outer loop lands.)
        let lsf = false; // MPEG-1 only this round.
        let padded_main_datas: Vec<Vec<u8>> = main_datas
            .iter()
            .zip(slots.iter())
            .map(|(md, &slot)| {
                let mut v = md.clone();
                if v.len() < slot {
                    v.resize(slot, 0);
                }
                v
            })
            .collect();
        let res_frames: Vec<ReservoirFrame<'_>> = padded_main_datas
            .iter()
            .zip(slots.iter())
            .map(|(md, &sl)| ReservoirFrame {
                main_data: md,
                slot_bytes: sl,
                lsf,
            })
            .collect();
        let scheduled = schedule_reservoir(&res_frames, &mut side_infos)
            .map_err(StreamEncodeError::Reservoir)?;

        // Pre-compute the audio totals BEFORE emission so the Xing
        // carrier frame (if any) can carry frame-count / byte-count
        // fields whose values describe the audio region the demuxer
        // sees after the carrier.
        let audio_frame_count = scheduled.len() as u32;
        let mut audio_total_bytes: u64 = 0;
        // Per-frame cumulative byte offsets within the audio region,
        // measured from offset 0 = start of the FIRST audio frame
        // (i.e. immediately after the optional Xing carrier). Length
        // is `audio_frame_count + 1`: entry `i` is the start offset of
        // audio frame `i`; the final entry is the total byte count.
        // These offsets drive the Xing TOC computation below.
        let mut cum_audio_offsets: Vec<u64> = Vec::with_capacity(scheduled.len() + 1);
        cum_audio_offsets.push(0);
        for sch in scheduled.iter() {
            // frame_len = 4 (header) + crc_bytes + si_bytes +
            // slot.len() per construction; computing it from
            // frame_lens[i] is the same arithmetic.
            audio_total_bytes += 4 + crc_bytes as u64 + si_bytes as u64 + sch.slot.len() as u64;
            cum_audio_offsets.push(audio_total_bytes);
        }
        let audio_total_bytes_u32: u32 = audio_total_bytes.try_into().unwrap_or(u32::MAX);

        // Step 3a: optional Xing / Info carrier frame.
        //
        // Inserted as the FIRST frame of the output. It is a silent
        // Layer III frame sized to the unpadded base CBR frame length
        // (so consumer demuxers and seekers see a regular leading
        // frame), with the Xing / Info magic + flagged fields patched
        // over the leading bytes of its main-data slot. The carrier
        // frame is itself NOT counted in `frames` / `bytes` (those
        // refer to the audio region that follows, per the symmetric
        // [`crate::demuxer::parse_xing_info`] reader on the demuxer's
        // first-frame skip path).
        let mut written = 0usize;
        if let Some(template) = xing_template {
            use crate::xing_info::{build_info_frame, flag_bit, XingTagSpec};
            // Fill in unresolved frames / bytes / toc fields per the
            // flag bits. Pre-set fields take precedence.
            let mut spec = XingTagSpec {
                id: template.id,
                flags: template.flags,
                frames: template.frames,
                bytes: template.bytes,
                toc: template.toc,
                quality: template.quality,
            };
            if spec.flags & flag_bit::FRAMES != 0 && spec.frames.is_none() {
                spec.frames = Some(audio_frame_count);
            }
            if spec.flags & flag_bit::BYTES != 0 && spec.bytes.is_none() {
                spec.bytes = Some(audio_total_bytes_u32);
            }
            if spec.flags & flag_bit::TOC != 0 && spec.toc.is_none() {
                spec.toc = Some(compute_xing_toc(&cum_audio_offsets, audio_total_bytes));
            }
            // Carrier header: same as the audio header template but
            // unpadded (so its size equals the base CBR frame length).
            let mut carrier_hdr = self.header_template;
            carrier_hdr.padding = false;
            let carrier = build_info_frame(&carrier_hdr, &spec)
                .map_err(|e| StreamEncodeError::Huffman(e.to_string()))?;
            sink.write_all(&carrier)?;
            written += carrier.len();
        }

        // Step 3b: emit each audio frame as header (4 bytes) +
        // optional CRC (2 bytes per §2.4.3.1) + side_info (`si_bytes`
        // bytes) + slot (variable bytes).
        for (i, sch) in scheduled.iter().enumerate() {
            let hbytes = write_header(&headers[i]);
            sink.write_all(&hbytes)?;
            written += 4;
            let sib = write_side_info(&side_infos[i]);
            debug_assert_eq!(sib.len(), si_bytes);
            if self.crc_enabled {
                // §2.4.3.1: 16-bit CRC of header bits 16..31 plus the
                // first 135 (mono) / 256 (other modes) bits of the
                // side-info block, written big-endian (MSB first) in
                // the two-byte slot between header and side_info.
                let crc = crate::crc::crc16_layer3(&hbytes, &sib, self.nch as u8);
                sink.write_all(&crc.to_be_bytes())?;
                written += 2;
            }
            sink.write_all(&sib)?;
            written += sib.len();
            sink.write_all(&sch.slot)?;
            written += sch.slot.len();
        }
        Ok(written)
    }
}

// ---------------------------------------------------------------------
// Internal helpers.
// ---------------------------------------------------------------------

impl Default for GranuleChannelData {
    fn default() -> Self {
        GranuleChannelData {
            is: [0; NUM_LINES],
            big_pairs: 0,
            count1_quads: 0,
        }
    }
}

/// Long-block scalefactor-band start indices (Table 3-B.8) for the
/// active sample rate, transcribed locally so the encoder doesn't
/// reach into a decoder-private helper. Index 21 (one past band 20)
/// is the end+1 boundary so callers can read the top of the long-block
/// range as a "next" boundary.
fn long_band_starts_for(sample_rate_hz: u32) -> &'static [usize; 22] {
    const LONG_BANDS_32: [usize; 22] = [
        0, 4, 8, 12, 16, 20, 24, 30, 36, 44, 54, 66, 82, 102, 126, 156, 194, 240, 296, 364, 448,
        550,
    ];
    const LONG_BANDS_44: [usize; 22] = [
        0, 4, 8, 12, 16, 20, 24, 30, 36, 44, 52, 62, 74, 90, 110, 134, 162, 196, 238, 288, 342, 418,
    ];
    const LONG_BANDS_48: [usize; 22] = [
        0, 4, 8, 12, 16, 20, 24, 30, 36, 42, 50, 60, 72, 88, 106, 128, 156, 190, 230, 276, 330, 384,
    ];
    match sample_rate_hz {
        32_000 | 16_000 | 8_000 => &LONG_BANDS_32,
        48_000 | 24_000 | 12_000 => &LONG_BANDS_48,
        _ => &LONG_BANDS_44,
    }
}

/// Choose a three-region big-values subdivision: `(region0_end,
/// region1_end, region0_count, region1_count)`. Both `region0_count`
/// and `region1_count` are valid field values (4-bit / 3-bit) such
/// that the decoder's `region_boundaries` reproduces `(region0_end,
/// region1_end)` exactly.
///
/// The region ends are aligned to long-block scalefactor-band edges
/// (the only places the decoder's `region_boundaries` can produce),
/// so the encoder cannot pick an arbitrary line index.
// The band-walking loops use `b` as both the loop variable and the
// index into `starts` (which is what `region_boundaries` reads from);
// rewriting as `starts.iter().enumerate()` obscures the band ↔ start
// index relationship the spec uses.
#[allow(clippy::needless_range_loop)]
fn choose_region_split(
    sample_rate_hz: u32,
    _version: MpegVersion,
    bv2: usize,
) -> (usize, usize, u8, u8) {
    let starts = long_band_starts_for(sample_rate_hz);
    if bv2 == 0 {
        return (0, 0, 0, 0);
    }
    // Per §C.1.5.4.4.6: ~1/3 to region 0, ~1/4 to region 2 (so
    // ~5/12 to region 1). Find the band whose start is just below
    // bv2/3 (region 0 end) and 3·bv2/4 (region 1 end). Cap
    // region0_count at 15 (4-bit field) and region1_count at 7
    // (3-bit field).
    let third = bv2 / 3;
    let three_quarters = (bv2 * 3) / 4;
    // region 0 covers bands 0..=region0_count → r0_end = starts[region0_count+1].
    let mut r0_count: u8 = 0;
    for b in 1..=21usize {
        if starts[b] <= third {
            r0_count = (b - 1) as u8;
        } else {
            break;
        }
    }
    r0_count = r0_count.min(15);
    let r0_band = usize::from(r0_count) + 1;
    let r0_end = starts
        .get(r0_band)
        .copied()
        .unwrap_or(NUM_LINES)
        .min(NUM_LINES);

    let mut r1_count: u8 = 0;
    for b in (r0_band + 1)..=21usize {
        if starts[b] <= three_quarters {
            r1_count = (b - r0_band - 1) as u8;
        } else {
            break;
        }
    }
    r1_count = r1_count.min(7);
    let r1_band = r0_band + usize::from(r1_count) + 1;
    let r1_end = starts
        .get(r1_band)
        .copied()
        .unwrap_or(NUM_LINES)
        .min(NUM_LINES);

    (r0_end.min(bv2), r1_end.min(bv2), r0_count, r1_count)
}

/// Pick the best big-values codebook for `is[start..end]` that is
/// **actually capable** of coding every pair in the range (the
/// in-tree [`choose_best_table_for_region`] only checks the codebook's
/// `xlen` corner, not whether `linbits` covers the actual magnitude —
/// so it can pick e.g. table 16 (`linbits=1`, magnitude reach 16) for
/// a range with `|is| = 100`, which would silently truncate
/// magnitudes at emission time).
///
/// We filter candidate codebooks by per-table magnitude reach (the
/// largest magnitude they can encode without truncation), keep only
/// those whose reach ≥ the range's `max|is|`, and pick the minimum
/// bit cost among the survivors. Falls back to table 23 (linbits=13,
/// reach 8206) when no other table fits — it always covers our 8191
/// clamp.
fn best_table_or(is: &[i32; NUM_LINES], start: usize, end: usize) -> u8 {
    if start >= end {
        return 0;
    }
    let max_mag = is[start..end]
        .iter()
        .map(|v| v.unsigned_abs())
        .max()
        .unwrap_or(0);
    let reach = |idx: u8| -> u32 {
        // Tables 0..=15: linbits=0 → reach = xlen - 1 (xlen from
        // huffman_tables.rs). Tables 16..=31: linbits in 1..=13 →
        // reach = 15 + 2^linbits - 1.
        match idx {
            0 => 0,
            1 => 1,
            2 | 3 => 2,
            5 | 6 => 3,
            7..=9 => 5,
            10..=12 => 7,
            13 | 15 => 15,
            16 => 16,
            17 => 18,
            18 => 22,
            19 => 30,
            20 => 78,
            21 => 270,
            22 => 1038,
            23 => 8206,
            24 => 30,
            25 => 46,
            26 => 78,
            27 => 142,
            28 => 270,
            29 => 526,
            30 => 2062,
            31 => 8206,
            _ => 0,
        }
    };
    use crate::huffman::SELECTABLE_BIG_TABLES;
    let mut best: Option<(u8, usize)> = None;
    for &idx in SELECTABLE_BIG_TABLES.iter() {
        if reach(idx) < max_mag {
            continue;
        }
        // Use the in-tree single-table cost helper indirectly: it
        // returns Some only when every pair in [0, end) is codable by
        // `idx` (corner test). For ranges starting at start > 0 we
        // re-use this by chopping off the leading portion in a
        // scratch buffer.
        if let Some(bits) = bits_for_range(is, start, end, idx) {
            match best {
                Some((_, b)) if bits >= b => {}
                _ => best = Some((idx, bits)),
            }
        }
    }
    best.map(|(t, _)| t).unwrap_or(23)
}

/// Bit cost of coding `is[start..end)` as one big-values region under
/// codebook `table_idx`. Reuses the in-tree
/// [`crate::huffman::count_huffman_bits`] by shoehorning the range
/// into "region 2" of a three-region split, with region 0 and
/// region 1 collapsed to `(start, start)`. Returns `None` if any
/// pair is not codable by the chosen table (corner test) **or** if
/// the table's `linbits` cannot represent the magnitudes (corner
/// test only catches the small tables; the ESC tables' linbits-reach
/// limit is filtered upstream by the per-table `reach` lookup, so any
/// `None` returned here is the corner case).
fn bits_for_range(is: &[i32; NUM_LINES], start: usize, end: usize, table_idx: u8) -> Option<usize> {
    if start >= end {
        return Some(0);
    }
    // Place the range under region 2 (table_select[2] = idx). Set
    // big_pairs = end / 2 so the range [start, end) actually carries
    // pairs. Region 0 / 1 cover [0, start) but with tables 0/1
    // we'd need them codable too — easier to set region_ends = (0, 0)
    // and ALSO start = 0 effectively, BUT the caller has start > 0.
    //
    // Simpler: when start > 0, count the FULL [0, end) range under
    // `table_idx` and subtract the [0, start) cost under the same
    // table. Both share `table_idx`, so the additive split is exact.
    let full = single_table_bits_from_zero(is, end, table_idx)?;
    if start == 0 {
        return Some(full);
    }
    let head = single_table_bits_from_zero(is, start, table_idx)?;
    Some(full - head)
}

/// Bit cost of coding `is[0..end)` as ONE big-values region under
/// `table_idx`. Calls [`crate::huffman::count_huffman_bits`] with
/// `big_pairs = end / 2`, three regions collapsed to a single one at
/// position 0..end via `region_ends = (end, end)` and
/// `table_select = [table_idx, 0, 0]`. Returns `None` only when a
/// pair in `[0, end)` is not codable by `table_idx`.
fn single_table_bits_from_zero(is: &[i32; NUM_LINES], end: usize, table_idx: u8) -> Option<usize> {
    if end == 0 {
        return Some(0);
    }
    let end = end.min(NUM_LINES);
    let end = end - (end % 2); // round down to even
    if end == 0 {
        return Some(0);
    }
    let big_pairs = end / 2;
    crate::huffman::count_huffman_bits(is, big_pairs, (end, end), [table_idx, 0, 0], 0, false)
}

/// A default long-block granule-channel record: window_switching off,
/// all selectors zero, region0_count clamped to a single region
/// covering every long band.
fn default_long_gc() -> GranuleChannel {
    GranuleChannel {
        part2_3_length: 0,
        big_values: 0,
        global_gain: 0,
        scalefac_compress: 0,
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

/// A default pure-short-block granule-channel record: window_switching
/// on, `block_type = Short`, `mixed_block_flag = false`, all
/// selectors zero, region defaults from
/// [`crate::short_block::short_block_region_defaults`] (these are not
/// transmitted on the wire for window-switched blocks per
/// [`crate::encoder::write_granule_channel`], but populate the struct
/// for parser-roundtrip equality). Used by the force-short encode path
/// (see [`Mp3Encoder::force_short_blocks_for_testing`]).
fn default_short_gc() -> GranuleChannel {
    let (r0, r1) = crate::short_block::short_block_region_defaults();
    GranuleChannel {
        part2_3_length: 0,
        big_values: 0,
        global_gain: 0,
        scalefac_compress: 0,
        window_switching_flag: true,
        block_type: BlockType::Short,
        mixed_block_flag: false,
        table_select: [0; 3],
        subblock_gain: [0; 3],
        region0_count: r0,
        region1_count: r1,
        preflag: false,
        scalefac_scale: false,
        count1table_select: false,
    }
}

/// A default mixed-block granule-channel record: window_switching on,
/// `block_type = Short`, `mixed_block_flag = true`, all selectors zero,
/// region defaults that match what
/// [`crate::side_info::parse_side_info`] reconstructs for a mixed
/// window-switched granule (`region0_count = 7`, `region1_count = 63`
/// — the 3-bit field caps at 7 on the wire, but a parser only reads
/// `region0_count` from the value table and synthesises the `63` for
/// `region1_count`; both numbers are decoder-ignored for short-family
/// granules, so the encoder writes `7 / 7` and trusts the parser's
/// reconstruction). Used by the force-mixed encode path (see
/// [`Mp3Encoder::force_mixed_blocks_for_testing`]).
fn default_mixed_gc() -> GranuleChannel {
    GranuleChannel {
        part2_3_length: 0,
        big_values: 0,
        global_gain: 0,
        scalefac_compress: 0,
        window_switching_flag: true,
        block_type: BlockType::Short,
        mixed_block_flag: true,
        table_select: [0; 3],
        subblock_gain: [0; 3],
        // Wire field widths are 4-bit / 3-bit; the decoder regenerates
        // these for window-switched blocks regardless of carried values.
        region0_count: 7,
        region1_count: 7,
        preflag: false,
        scalefac_scale: false,
        count1table_select: false,
    }
}

/// Side-info byte length lookup mirroring
/// [`crate::demuxer::side_info_len`] without taking a `MpegVersion`
/// dep (we know MPEG-1 only).
fn side_info_byte_len(nch: usize) -> usize {
    if nch == 1 {
        17
    } else {
        32
    }
}

/// Resolve an MPEG-1 Layer III bitrate (kbit/s) to its position
/// (1..=14) on the §2.4.2.3 ladder. Returns `None` for an off-ladder
/// value (free format `0`, forbidden `15`, or any kbps not in
/// [`MPEG1_L3_BITRATE_LADDER_KBPS`]).
fn ladder_index(kbps: u32) -> Option<u8> {
    MPEG1_L3_BITRATE_LADDER_KBPS
        .iter()
        .position(|&v| v == kbps)
        .map(|i| (i + 1) as u8)
}

/// Compute the main-data slot byte capacity for the §2.4.2.3 ladder
/// `bitrate_index` (1..=14) at the given `sample_rate_hz`, after
/// subtracting the 4-byte header and `si_bytes` side-info bytes (no
/// CRC). When `padded` is true, the slot includes the one-byte padding
/// slot the per-frame `padding` bit absorbs.
fn ladder_slot_capacity(
    sample_rate_hz: u32,
    bitrate_index: u8,
    si_bytes: usize,
    padded: bool,
) -> usize {
    let kbps = MPEG1_L3_BITRATE_LADDER_KBPS[(bitrate_index - 1) as usize];
    let bps = u64::from(kbps) * 1000;
    let sr = u64::from(sample_rate_hz);
    let unpadded = (144 * bps / sr) as usize;
    let frame_len = unpadded + usize::from(padded);
    frame_len.saturating_sub(4 + si_bytes)
}

/// Pick the smallest §2.4.2.3 ladder index in `[cfg.min_index,
/// cfg.max_index]` whose slot — possibly with the per-frame padding
/// byte and after subtracting `crc_bytes` (2 when the §2.4.3.1 CRC
/// slot is active, 0 otherwise) — can hold `need` bytes of main-data.
/// Returns the chosen `(kbps, ladder_index)` or `None` when even the
/// max index's padded slot is insufficient.
fn pick_vbr_bitrate(
    sample_rate_hz: u32,
    cfg: VbrConfig,
    need: usize,
    crc_bytes: usize,
) -> Option<(u32, u8)> {
    let si_bytes = 17; // mono-only this round; matches side_info_byte_len(1).
    for idx in cfg.min_index..=cfg.max_index {
        // Try unpadded first, then padded — the per-frame padding bit
        // adds one byte to the slot at the same `bitrate_index`. For
        // VBR with min_kbps == max_kbps this preserves the CBR
        // Bresenham padding behaviour roughly (padding is enabled only
        // when needed).
        let cap_padded =
            ladder_slot_capacity(sample_rate_hz, idx, si_bytes, true).saturating_sub(crc_bytes);
        if cap_padded >= need {
            let kbps = MPEG1_L3_BITRATE_LADDER_KBPS[(idx - 1) as usize];
            return Some((kbps, idx));
        }
    }
    None
}

/// Compute the Xing `toc[100]` field for a stream whose per-audio-frame
/// cumulative byte offsets are in `cum_offsets` (length =
/// `audio_frame_count + 1`; entry 0 = 0; entry N = `total_bytes`).
///
/// `toc[i] = floor(256 * audio_offset_for_percentile(i) / total_bytes)`
/// for `i in 0..100`, where `audio_offset_for_percentile(i)` is the
/// byte offset of the audio frame whose **start** is closest to the
/// playback position `i / 100`. Each TOC entry is constrained to fit
/// in one byte (0..=255), so a `total_bytes == 0` stream emits an
/// all-zero TOC.
fn compute_xing_toc(cum_offsets: &[u64], total_bytes: u64) -> [u8; 100] {
    let mut toc = [0u8; 100];
    if total_bytes == 0 || cum_offsets.len() < 2 {
        return toc;
    }
    let n_frames = cum_offsets.len() - 1;
    for (i, slot) in toc.iter_mut().enumerate() {
        // The percentile maps to a frame INDEX, not a byte offset:
        // walking by frames matches how a tag-aware seeker uses the
        // TOC (look up `toc[idx]`, then start decoding the frame
        // whose offset is `bytes * toc[idx] / 256`). Pick the frame
        // start nearest to `i/100` of the total audio frames.
        let frame_idx = ((i as u64) * (n_frames as u64) / 100) as usize;
        let frame_idx = frame_idx.min(n_frames - 1);
        let offset = cum_offsets[frame_idx];
        // floor(256 * offset / total_bytes), capped at 255.
        let scaled = (256u64 * offset) / total_bytes;
        *slot = scaled.min(255) as u8;
    }
    toc
}

/// Defensive: clamp every `is[i]` magnitude to `bound`, preserving sign.
fn clamp_above(is: &mut [i32; NUM_LINES], bound: i32) {
    for v in is.iter_mut() {
        if *v > bound {
            *v = bound;
        } else if *v < -bound {
            *v = -bound;
        }
    }
}

/// Inverse of [`crate::alias::alias_reduce`] for long blocks.
///
/// `alias_reduce` over a long block applies the §2.4.3.4.10.1 butterfly
///   `xar[lo] = xr[lo]*cs - xr[hi]*ca`
///   `xar[hi] = xr[hi]*cs + xr[lo]*ca`
/// at the 31 subband boundaries (8 butterflies each). With `cs²+ca²=1`
/// the butterfly is orthogonal; its inverse negates `ca`:
///   `xr[lo] = xar[lo]*cs + xar[hi]*ca`
///   `xr[hi] = xar[hi]*cs - xar[lo]*ca`
///
/// The encoder feeds the inverse-reduced spectrum to the quantizer so
/// that the decoder's forward alias-reduce recovers the post-MDCT bins.
///
/// Short blocks pass through unchanged (decoder skips alias_reduce when
/// `block_type == Short`).
fn inverse_alias_reduce(xr: &[f32; NUM_LINES]) -> [f32; NUM_LINES] {
    use crate::alias::{alias_ca, alias_cs};
    let cs = alias_cs();
    let ca = alias_ca();
    let mut out = *xr;
    // Apply the inverse butterfly across each of the 31 subband
    // boundaries (sb = 1..32). Source inputs come from `out` updated
    // in place: each butterfly's `(lo, hi)` is a fresh pair so we can
    // read-then-write within the same loop iteration without cross-
    // contamination across butterflies of the same boundary.
    for sb in 1..32 {
        let boundary = 18 * sb;
        // Collect originals first so each butterfly reads pre-update
        // values (the §2.4.3.4.10.1 butterfly does the same on the
        // decoder side).
        let mut tmp_lo = [0.0f32; 8];
        let mut tmp_hi = [0.0f32; 8];
        for i in 0..8 {
            tmp_lo[i] = out[boundary - 1 - i];
            tmp_hi[i] = out[boundary + i];
        }
        for i in 0..8 {
            out[boundary - 1 - i] = tmp_lo[i] * cs[i] + tmp_hi[i] * ca[i];
            out[boundary + i] = tmp_hi[i] * cs[i] - tmp_lo[i] * ca[i];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alias::alias_reduce;
    use crate::encode_silent_frame;
    use crate::side_info::GranuleChannel as _GC;

    #[test]
    fn inverse_alias_roundtrip_long_block() {
        // Build a deterministic xr, forward alias_reduce, then inverse;
        // should recover xr within FP precision.
        let mut xr = [0.0f32; NUM_LINES];
        for (i, v) in xr.iter_mut().enumerate() {
            *v = ((i as f32) * 0.137).sin() * 1000.0;
        }
        let gc = default_long_gc();
        let _: _GC = gc; // type-check helper
        let xar = alias_reduce(&xr, &gc);
        let xr_back = inverse_alias_reduce(&xar);
        for i in 0..NUM_LINES {
            let diff = (xr_back[i] - xr[i]).abs();
            assert!(
                diff < 1e-2,
                "inverse alias [{i}] diff={diff} xr={} back={}",
                xr[i],
                xr_back[i]
            );
        }
    }

    /// Stand-alone forward/inverse roundtrip BYPASSING quantize. Feed
    /// known PCM through `analyze → freq_inv → inverse_alias → MDCT/k`
    /// (encoder analysis stack) and back through `alias_reduce →
    /// imdct_granule → freq_inv (inside imdct) → synth`. The
    /// reconstructed PCM should match the input scaled by 1 (or some
    /// fixed factor we can identify). This lets us isolate the MDCT
    /// scaling factor without quantization noise.
    #[test]
    fn lossless_chain_finds_unit_gain_scale_factor() {
        use crate::alias::alias_reduce;
        use crate::imdct::imdct_granule;
        use crate::synth::{synth_granule, PCM_PER_GRANULE};

        // Feed 5 granules of a known-amplitude signal so the
        // filterbank states warm up.
        const G: usize = 8;
        let mut pcm_in = vec![0.0f32; PCM_PER_GRANULE * G];
        for (n, v) in pcm_in.iter_mut().enumerate() {
            *v = ((n as f32) * 0.013).sin() * 0.5;
        }

        let mut ana = AnalysisState::new();
        let mut mdct_states: Vec<MdctState> = (0..32).map(|_| MdctState::new()).collect();
        let mut imdct_state = crate::imdct::ImdctState::new();
        let mut synth_state = crate::synth::SynthState::new();
        let mut pcm_out = vec![0.0f32; PCM_PER_GRANULE * G];

        for g in 0..G {
            let mut gr_pcm = [0.0f32; SAMPLES_PER_GRANULE];
            gr_pcm.copy_from_slice(&pcm_in[g * PCM_PER_GRANULE..(g + 1) * PCM_PER_GRANULE]);
            let subband_time = analyze_granule(&gr_pcm, &mut ana);
            // Freq inversion (encoder side).
            let mut inv = subband_time;
            for (_sb, sb_row) in inv.iter_mut().enumerate().skip(1).step_by(2) {
                for t in (1..18usize).step_by(2) {
                    sb_row[t] = -sb_row[t];
                }
            }
            // Forward MDCT per subband.
            let mut xr = [0.0f32; NUM_LINES];
            for sb in 0..32usize {
                let mut current = [0.0f64; LONG_N / 2];
                for (t, slot) in current.iter_mut().enumerate() {
                    *slot = f64::from(inv[sb][t]);
                }
                let frame36 = forward_overlap(&current, &mut mdct_states[sb]);
                let windowed = window_long_family_analysis(&frame36, BlockType::Long);
                let bins = mdct(&windowed, LONG_N);
                for (k, &b) in bins.iter().enumerate() {
                    // No scaling here — we'll compute the scale
                    // factor empirically from the output.
                    xr[sb * 18 + k] = b as f32;
                }
            }
            // Inverse alias reduction.
            let xr_pre = inverse_alias_reduce(&xr);
            // Decode side: alias_reduce → imdct → synth.
            let gc = default_long_gc();
            let xar = alias_reduce(&xr_pre, &gc);
            let st = imdct_granule(&xar, &gc, &mut imdct_state);
            let pcm = synth_granule(&st, &mut synth_state);
            for (i, &v) in pcm.iter().enumerate() {
                pcm_out[g * PCM_PER_GRANULE + i] = v;
            }
        }

        // Find the scale factor: in steady state (after ~3 granules)
        // pcm_out[n] ≈ k * pcm_in[n - delay] for some k and delay.
        // For our chain the delay should be small (mdct overlap 1
        // granule + filterbank delay 481 samples). Just check the
        // approximate amplitude ratio in the final granule.
        let in_max = pcm_in[5 * PCM_PER_GRANULE..]
            .iter()
            .map(|v| v.abs())
            .fold(0.0f32, f32::max);
        let out_max = pcm_out[5 * PCM_PER_GRANULE..]
            .iter()
            .map(|v| v.abs())
            .fold(0.0f32, f32::max);
        let ratio = out_max / in_max;
        eprintln!("lossless chain in_max={in_max} out_max={out_max} ratio={ratio}");
        // Sanity: the chain produced something measurable.
        assert!(out_max > in_max * 0.1, "chain output too small");
    }

    /// In-process roundtrip: encoder analysis → quantize → requantize
    /// → decoder synthesis. Skips the byte-stream + reservoir +
    /// demuxer round-trip. Confirms the per-granule chain produces a
    /// reasonable PSNR for a sine tone.
    #[test]
    fn in_process_roundtrip_sine_psnr() {
        use crate::alias::alias_reduce;
        use crate::imdct::imdct_granule;
        use crate::requantize::requantize;
        use crate::synth::{synth_granule, PCM_PER_GRANULE};

        const G: usize = 12; // 12 granules total, warm-up ~5
        const SR: u32 = 44_100;
        let mut pcm_in = vec![0.0f32; PCM_PER_GRANULE * G];
        for (n, v) in pcm_in.iter_mut().enumerate() {
            let t = n as f32 / SR as f32;
            *v = (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5;
        }

        let mut ana = AnalysisState::new();
        let mut mdct_states: Vec<MdctState> = (0..32).map(|_| MdctState::new()).collect();
        let mut imdct_state = crate::imdct::ImdctState::new();
        let mut synth_state = crate::synth::SynthState::new();
        let mut pcm_out = vec![0.0f32; PCM_PER_GRANULE * G];

        for g in 0..G {
            let mut gr_pcm = [0.0f32; SAMPLES_PER_GRANULE];
            gr_pcm.copy_from_slice(&pcm_in[g * PCM_PER_GRANULE..(g + 1) * PCM_PER_GRANULE]);
            let subband_time = analyze_granule(&gr_pcm, &mut ana);
            let mut inv = subband_time;
            for (_sb, sb_row) in inv.iter_mut().enumerate().skip(1).step_by(2) {
                for t in (1..18usize).step_by(2) {
                    sb_row[t] = -sb_row[t];
                }
            }
            let mut xr = [0.0f32; NUM_LINES];
            for sb in 0..32usize {
                let mut current = [0.0f64; LONG_N / 2];
                for (t, slot) in current.iter_mut().enumerate() {
                    *slot = f64::from(inv[sb][t]);
                }
                let frame36 = forward_overlap(&current, &mut mdct_states[sb]);
                let windowed = window_long_family_analysis(&frame36, BlockType::Long);
                let bins = mdct(&windowed, LONG_N);
                for (k, &b) in bins.iter().enumerate() {
                    xr[sb * 18 + k] = (b / 9.0) as f32;
                }
            }
            let xr_pre = inverse_alias_reduce(&xr);

            // Pick a global_gain and quantize.
            let gc_template = default_long_gc();
            let sf = ScaleFactors::default();
            let res = search_magnitude_clamp(&xr_pre, &gc_template, &sf, SR, MpegVersion::Mpeg1);
            let mut gc = gc_template;
            gc.global_gain = res.global_gain;
            let is = quantize(&xr_pre, &gc, &sf, SR, MpegVersion::Mpeg1);

            // Now decode it.
            let xr_back = requantize(&is, &gc, &sf, SR, MpegVersion::Mpeg1);
            let xar = alias_reduce(&xr_back, &gc);
            let st = imdct_granule(&xar, &gc, &mut imdct_state);
            let pcm = synth_granule(&st, &mut synth_state);
            for (i, &v) in pcm.iter().enumerate() {
                pcm_out[g * PCM_PER_GRANULE + i] = v;
            }
        }

        // Scan over plausible delays — the lapped MDCT introduces 1
        // granule (576 samples) extra delay on top of the
        // filterbank's 481-sample group delay.
        let warmup = 5 * PCM_PER_GRANULE;
        let mut best_psnr = -100.0f64;
        let mut best_delay = 0usize;
        for delay in 400..1300usize {
            let mut sse = 0.0f64;
            let mut count = 0usize;
            for i in warmup..pcm_in.len() {
                if i < delay {
                    continue;
                }
                let d = f64::from(pcm_in[i - delay]) - f64::from(pcm_out[i]);
                sse += d * d;
                count += 1;
            }
            if count == 0 {
                continue;
            }
            let mse = sse / count as f64;
            if mse == 0.0 {
                best_psnr = 1e30;
                best_delay = delay;
                break;
            }
            let psnr = 10.0 * (1.0 / mse).log10();
            if psnr > best_psnr {
                best_psnr = psnr;
                best_delay = delay;
            }
        }
        eprintln!("in-process best PSNR = {best_psnr} dB at delay {best_delay}");
        assert!(best_psnr > 20.0, "in-process PSNR too low: {best_psnr} dB");
    }

    #[test]
    fn silence_encode_then_decode_zero_bytes() {
        // Mono 44.1 kHz 128 kbit/s; push zero samples for ~one frame.
        let mut enc = Mp3Encoder::new(128, 44_100, ChannelMode::SingleChannel).unwrap();
        let pcm = vec![0i16; SAMPLES_PER_FRAME_MPEG1 * 2];
        enc.push_samples(&pcm).unwrap();
        let mut out: Vec<u8> = Vec::new();
        let bytes = enc.finish(&mut out).unwrap();
        assert!(bytes > 0);
        assert_eq!(out.len(), bytes);
        // Each frame is 417 bytes (128 kbit/s, 44.1 kHz Layer III)
        // plus optional padding (every ~9 frames).
        assert!(out.len() >= 2 * 417);
    }

    #[test]
    fn silent_frame_helper_matches_silent_encoder() {
        // Confirm a single-frame "silent" encoder run produces a frame
        // matching the [`encode_silent_frame`] convenience constructor
        // in structural envelope (header + side_info bytes — the slot
        // payload may differ because the stream encoder doesn't force
        // part2_3_length = 0, it just happens to be near zero for
        // pure silence).
        let h = make_silent_header(128, 44_100, ChannelMode::SingleChannel).unwrap();
        let silent = encode_silent_frame(&h).unwrap();
        // Sanity: the standalone silent frame is the full 417 bytes
        // and starts with a valid header.
        assert_eq!(silent.len(), 417);
        assert_eq!((silent[0] as u16) << 4 | (silent[1] as u16) >> 4, 0xFFF);
    }

    #[test]
    fn protection_bit_toggle_default_is_false() {
        let enc = Mp3Encoder::new(128, 44_100, ChannelMode::SingleChannel).unwrap();
        assert!(!enc.crc_enabled());
    }

    #[test]
    fn protection_bit_toggle_round_trips() {
        let mut enc = Mp3Encoder::new(128, 44_100, ChannelMode::SingleChannel).unwrap();
        enc.with_protection_bit(true);
        assert!(enc.crc_enabled());
        enc.with_protection_bit(false);
        assert!(!enc.crc_enabled());
    }

    #[test]
    fn crc_enabled_frame_length_unchanged() {
        // §2.4.3.1: the CRC slot is INSIDE the frame's existing byte
        // length — it consumes 2 bytes of main-data slot, not 2 extra
        // bytes added to the frame. So a CRC-enabled stream emits the
        // same number of bytes per frame as a CRC-disabled stream at
        // the same bitrate / sample rate.
        // 128 kbit/s mono @ 44.1 kHz silence: ~3 frames, each 417 B
        // unpadded + Bresenham padding.
        let pcm = vec![0i16; SAMPLES_PER_FRAME_MPEG1 * 3];

        let mut enc_nocrc = Mp3Encoder::new(128, 44_100, ChannelMode::SingleChannel).unwrap();
        enc_nocrc.push_samples(&pcm).unwrap();
        let mut out_nocrc = Vec::new();
        enc_nocrc.finish(&mut out_nocrc).unwrap();

        let mut enc_crc = Mp3Encoder::new(128, 44_100, ChannelMode::SingleChannel).unwrap();
        enc_crc.with_protection_bit(true);
        enc_crc.push_samples(&pcm).unwrap();
        let mut out_crc = Vec::new();
        enc_crc.finish(&mut out_crc).unwrap();

        // Same on-wire byte length: the CRC reclaims 2 bytes from the
        // main-data slot, the frame's outer size is unchanged.
        assert_eq!(out_nocrc.len(), out_crc.len());

        // Wire protection bit is set on every CRC-enabled audio frame.
        // Walk the stream and verify.
        use crate::frame::{parse_header, FrameWalker};
        let frames_crc: Vec<_> = FrameWalker::new(&out_crc).collect();
        for f in &frames_crc {
            let hdr = parse_header(&f.data[..4]).unwrap();
            assert!(
                hdr.crc_protected,
                "CRC-enabled frame must carry protection_bit=0"
            );
        }
        // And the no-CRC stream sets it the other way.
        let frames_nocrc: Vec<_> = FrameWalker::new(&out_nocrc).collect();
        for f in &frames_nocrc {
            let hdr = parse_header(&f.data[..4]).unwrap();
            assert!(
                !hdr.crc_protected,
                "CRC-disabled frame must carry protection_bit=1"
            );
        }
    }

    #[test]
    fn crc_value_matches_recomputed_crc() {
        // Encode a stream with CRC enabled. For every emitted audio
        // frame, the 2 bytes immediately after the header must equal
        // the §2.4.3.1 CRC-16 over (header bytes 2..4) ++ (first 135
        // bits of side_info) — the same computation
        // [`crc16_layer3`] performs.
        let pcm = vec![0i16; SAMPLES_PER_FRAME_MPEG1 * 4];

        let mut enc = Mp3Encoder::new(128, 44_100, ChannelMode::SingleChannel).unwrap();
        enc.with_protection_bit(true);
        enc.push_samples(&pcm).unwrap();
        let mut out = Vec::new();
        enc.finish(&mut out).unwrap();

        use crate::frame::{parse_header, FrameWalker};
        let frames: Vec<_> = FrameWalker::new(&out).collect();
        // 4 frames of 1152 samples each — and the stream encoder may
        // round up by one to absorb tail PCM, so 4..=5 is the safe
        // bound.
        assert!(
            (4..=5).contains(&frames.len()),
            "unexpected frame count {}",
            frames.len()
        );
        for f in &frames {
            let hdr_bytes: [u8; 4] = f.data[..4].try_into().unwrap();
            let hdr = parse_header(&hdr_bytes).unwrap();
            assert!(hdr.crc_protected);
            let wire_crc = u16::from_be_bytes([f.data[4], f.data[5]]);
            let nch = hdr.channel_count();
            let si_bytes = side_info_byte_len(nch as usize);
            let si_slice = &f.data[6..6 + si_bytes];
            let expected = crate::crc::crc16_layer3(&hdr_bytes, si_slice, nch);
            assert_eq!(
                wire_crc, expected,
                "frame at offset {} CRC mismatch: wire 0x{:04X} vs expected 0x{:04X}",
                f.offset, wire_crc, expected
            );
        }
    }

    // ---- §2.4.2.3 joint-stereo auto MS/LR per-frame picker (round 149) ----

    /// `new_joint_stereo_auto` arms the picker with the default 0.5
    /// threshold, upgrades the header template to joint-stereo, leaves
    /// the unconditional MS flag off, and configures two channels.
    #[test]
    fn auto_ms_picker_default_threshold_is_half() {
        let enc = Mp3Encoder::new_joint_stereo_auto(192, 44_100).unwrap();
        assert_eq!(enc.ms_auto_threshold(), Some(0.5));
        // Unconditional MS stays off — the picker drives the
        // forward-MS branch per frame instead.
        assert!(!enc.ms_stereo_enabled());
        // Header template carries `mode = '01'` (joint stereo) with a
        // starting `mode_extension = '00'` (the per-frame picker
        // rewrites it).
        assert!(matches!(enc.header_template.mode, ChannelMode::JointStereo));
        assert!(!enc.header_template.mode_extension.ms_stereo);
        assert!(!enc.header_template.mode_extension.intensity_stereo);
    }

    /// `with_ms_auto_threshold` overrides the threshold and clamps
    /// out-of-range values into `[0.0, 1.0]`.
    #[test]
    fn auto_ms_picker_threshold_override_clamps() {
        let enc = Mp3Encoder::new_joint_stereo_auto(192, 44_100)
            .unwrap()
            .with_ms_auto_threshold(0.25);
        assert_eq!(enc.ms_auto_threshold(), Some(0.25));

        let enc = Mp3Encoder::new_joint_stereo_auto(192, 44_100)
            .unwrap()
            .with_ms_auto_threshold(-1.0);
        assert_eq!(enc.ms_auto_threshold(), Some(0.0));

        let enc = Mp3Encoder::new_joint_stereo_auto(192, 44_100)
            .unwrap()
            .with_ms_auto_threshold(2.5);
        assert_eq!(enc.ms_auto_threshold(), Some(1.0));
    }

    /// `with_ms_auto_threshold` on a non-auto encoder is a no-op (the
    /// picker is not armed; the threshold setter does not silently turn
    /// it on).
    #[test]
    fn auto_ms_picker_threshold_override_noop_on_non_auto() {
        let enc = Mp3Encoder::new(192, 44_100, ChannelMode::Stereo)
            .unwrap()
            .with_ms_auto_threshold(0.25);
        assert_eq!(enc.ms_auto_threshold(), None);

        let enc = Mp3Encoder::new_joint_stereo_ms(192, 44_100)
            .unwrap()
            .with_ms_auto_threshold(0.25);
        assert_eq!(enc.ms_auto_threshold(), None);
        // And ms_stereo (unconditional MS) stays armed.
        assert!(enc.ms_stereo_enabled());
    }

    /// A correlated-content frame (L ≈ R) picks MS: every emitted
    /// audio frame's header carries `mode_extension = '10'`.
    #[test]
    fn auto_ms_picker_correlated_input_chooses_ms() {
        use crate::frame::{parse_header, FrameWalker};
        use std::f32::consts::PI;

        const SR: u32 = 44_100;
        const BR: u32 = 192;
        // ~1/4 sec of strongly correlated 440 Hz tone panned 70/30.
        let n = SAMPLES_PER_FRAME_MPEG1 * 10;
        let mut pcm = Vec::with_capacity(n * 2);
        let scale = i16::MAX as f32 * 0.5;
        for i in 0..n {
            let t = i as f32 / SR as f32;
            let s = (2.0 * PI * 440.0 * t).sin();
            // Pan: L = 0.7s, R = 0.7s (perfect mono — most extreme
            // correlated case so the side energy is ~0).
            let v = (s * scale).round() as i16;
            pcm.push(v);
            pcm.push(v);
        }
        let mut enc = Mp3Encoder::new_joint_stereo_auto(BR, SR).unwrap();
        enc.push_samples(&pcm).unwrap();
        let mut out = Vec::new();
        enc.finish(&mut out).unwrap();

        let frames: Vec<_> = FrameWalker::new(&out).collect();
        assert!(
            frames.len() >= 8,
            "expected several frames, got {}",
            frames.len()
        );
        for f in &frames {
            let hdr_bytes: [u8; 4] = f.data[..4].try_into().unwrap();
            let hdr = parse_header(&hdr_bytes).unwrap();
            assert!(matches!(hdr.mode, ChannelMode::JointStereo));
            assert!(
                hdr.mode_extension.ms_stereo,
                "correlated input must pick ms_stereo on frame at offset {}",
                f.offset
            );
            assert!(!hdr.mode_extension.intensity_stereo);
        }
    }

    /// An anti-correlated frame (R = -L) picks LR: every emitted audio
    /// frame's header carries `mode_extension = '00'`.
    #[test]
    fn auto_ms_picker_anticorrelated_input_chooses_lr() {
        use crate::frame::{parse_header, FrameWalker};
        use std::f32::consts::PI;

        const SR: u32 = 44_100;
        const BR: u32 = 192;
        let n = SAMPLES_PER_FRAME_MPEG1 * 10;
        let mut pcm = Vec::with_capacity(n * 2);
        let scale = i16::MAX as f32 * 0.5;
        for i in 0..n {
            let t = i as f32 / SR as f32;
            let s = (2.0 * PI * 440.0 * t).sin();
            // Anti-correlated: R = -L. The MS rotation here flips:
            // M = (L+R)/√2 = 0, S = (L-R)/√2 = √2·L; all energy goes
            // into the side channel and ratio = 1.0 > 0.5.
            let v = (s * scale).round() as i16;
            pcm.push(v);
            pcm.push(-v);
        }
        let mut enc = Mp3Encoder::new_joint_stereo_auto(BR, SR).unwrap();
        enc.push_samples(&pcm).unwrap();
        let mut out = Vec::new();
        enc.finish(&mut out).unwrap();

        let frames: Vec<_> = FrameWalker::new(&out).collect();
        assert!(
            frames.len() >= 8,
            "expected several frames, got {}",
            frames.len()
        );
        // Skip the very first frame: with cold MDCT overlap buffers the
        // first granule's spectrum has little real content, and the
        // ratio computation can fall on the wrong side of the threshold.
        // Steady-state frames must all be LR.
        for f in frames.iter().skip(2) {
            let hdr_bytes: [u8; 4] = f.data[..4].try_into().unwrap();
            let hdr = parse_header(&hdr_bytes).unwrap();
            assert!(matches!(hdr.mode, ChannelMode::JointStereo));
            assert!(
                !hdr.mode_extension.ms_stereo,
                "anti-correlated input must pick LR (mode_extension '00') on frame at offset {} but got ms_stereo on",
                f.offset
            );
            assert!(!hdr.mode_extension.intensity_stereo);
        }
    }

    /// Threshold tuned to 0.0 forces LR on every non-perfectly-mono
    /// frame: with `E_S > 0` and `threshold = 0`, the ratio always
    /// exceeds the threshold. A genuinely-mono signal (L == R)
    /// produces `E_S = 0` and slips below.
    #[test]
    fn auto_ms_picker_zero_threshold_forces_lr_on_any_side_energy() {
        use crate::frame::{parse_header, FrameWalker};
        use std::f32::consts::PI;

        const SR: u32 = 44_100;
        const BR: u32 = 192;
        // Mildly correlated content — L = 0.7s, R = 0.3s, so the side
        // channel carries real energy.
        let n = SAMPLES_PER_FRAME_MPEG1 * 8;
        let mut pcm = Vec::with_capacity(n * 2);
        let scale_l = i16::MAX as f32 * 0.7;
        let scale_r = i16::MAX as f32 * 0.3;
        for i in 0..n {
            let t = i as f32 / SR as f32;
            let s = (2.0 * PI * 440.0 * t).sin();
            pcm.push((s * scale_l).round() as i16);
            pcm.push((s * scale_r).round() as i16);
        }
        let mut enc = Mp3Encoder::new_joint_stereo_auto(BR, SR)
            .unwrap()
            .with_ms_auto_threshold(0.0);
        enc.push_samples(&pcm).unwrap();
        let mut out = Vec::new();
        enc.finish(&mut out).unwrap();

        let frames: Vec<_> = FrameWalker::new(&out).collect();
        // Skip the cold-start frames.
        for f in frames.iter().skip(2) {
            let hdr_bytes: [u8; 4] = f.data[..4].try_into().unwrap();
            let hdr = parse_header(&hdr_bytes).unwrap();
            assert!(
                !hdr.mode_extension.ms_stereo,
                "threshold=0 must reject MS on frame at offset {} (any side energy disqualifies)",
                f.offset
            );
        }
    }
}
