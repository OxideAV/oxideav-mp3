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
//! ID3v2 frontmatter, no short-block / mixed-block window switching.
//! **Joint-stereo MS** encode is opt-in
//! as of round 146 ([`Mp3Encoder::new_joint_stereo_ms`]): the encoder
//! computes the §2.4.3.4.9.2 forward MS matrix `M = (L+R)/√2`,
//! `S = (L-R)/√2` on each granule-pair's full post-MDCT spectrum and
//! emits header `mode = '01'` with `mode_extension = '10'` (ms_stereo
//! on, intensity_stereo off). **Intensity-stereo** encode
//! (§2.4.3.4.9.3) is opt-in as of round 284
//! ([`Mp3Encoder::new_joint_stereo_is`] /
//! [`Mp3Encoder::new_joint_stereo_ms_is`] /
//! [`Mp3Encoder::new_joint_stereo_auto_is`]): long scalefactor bands at
//! or above a caller-chosen start band are coupled to a single
//! magnitude in the left channel plus a per-band stereo position
//! carried as the right channel's scalefactor, and the header
//! `mode_extension` low bit is set (`'01'` intensity only, `'11'` MS
//! below the bound + intensity above it).
//!
//! Xing / Info VBR-info frame emission is **opt-in** as of round 142:
//! call [`Mp3Encoder::enable_xing_info`] before [`Mp3Encoder::finish`]
//! to prepend one carrier frame with the magic + per-stream totals to
//! the on-wire output. See [`crate::xing_info`] for layout details.

use std::io::{self, Write};

use crate::analysis::{analyze_granule, AnalysisState};
use crate::frame::{ChannelMode, ModeExtension, Mp3FrameHeader, MpegVersion};
use crate::huffman::{
    choose_best_count1_table, choose_best_table_for_region, partition_split, NUM_LINES,
};
use crate::inner_loop::{
    search_bit_budget_band_aligned, search_magnitude_clamp, GAIN_MAX, GAIN_MIN,
};
use crate::main_data::{
    assemble_main_data, schedule_reservoir, GranuleChannelData, ReservoirError, ReservoirFrame,
};
use crate::mdct::{forward_overlap, mdct, window_long_family_analysis, MdctState, LONG_N};
use crate::outer_loop::{
    outer_loop_search_long, outer_loop_search_mixed, outer_loop_search_short,
    INTENSITY_SCALEFAC_COMPRESS_LSF, MIXED_FIRST_SHORT_SFB, MIXED_LAST_LONG_SFB,
    OUTER_LOOP_SCALEFAC_COMPRESS, OUTER_LOOP_SCALEFAC_COMPRESS_LSF,
};
use crate::quantize::quantize;
use crate::scalefactors::{FrameScaleFactors, ScaleFactors, LONG_SFB};
use crate::side_info::{BlockType, GranuleChannel, SideInfo, GRANULES, GRANULES_LSF};
use crate::{make_silent_header, write_header, write_side_info, EncodeError};

/// MPEG-1 Layer III bitrate ladder (ISO/IEC 11172-3 §2.4.2.3, Table
/// 2-B.1 row "Layer III, version 1"). Used by the encoder's VBR path
/// to enumerate the 14 fixed bitrates a per-frame `bitrate_index` may
/// select. Indices `0` (free format) and `15` (forbidden) are excluded.
pub const MPEG1_L3_BITRATE_LADDER_KBPS: [u32; 14] = [
    32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320,
];

/// MPEG-2 / MPEG-2.5 LSF Layer III bitrate ladder (ISO/IEC 13818-3
/// §2.4.2.3, the `ID = 0` bitrate_index table shared by Layers II and
/// III; the MPEG-2.5 extension inherits it). Same indexing convention
/// as [`MPEG1_L3_BITRATE_LADDER_KBPS`]: entry `i` is `bitrate_index
/// i + 1`, indices `0` (free format) and `15` (forbidden) excluded.
pub const LSF_L3_BITRATE_LADDER_KBPS: [u32; 14] =
    [8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160];

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
    /// encode, [`Mp3Encoder::new_joint_stereo_is`] /
    /// [`Mp3Encoder::new_joint_stereo_ms_is`] for §2.4.3.4.9.3
    /// intensity-stereo encode, or [`ChannelMode::Stereo`] /
    /// [`ChannelMode::DualChannel`] for independent two-channel
    /// content.
    StereoUnsupported,
    /// An LSF (MPEG-2 16 / 22.05 / 24 kHz or MPEG-2.5 8 / 11.025 /
    /// 12 kHz) encoder was asked for a feature whose LSF wire format
    /// is not implemented yet: the §C.1.5.2 auto block-type scheduler
    /// (whose frame walk is still two-granule shaped). Core LSF encode
    /// — CBR / VBR, mono / stereo / dual-channel, MS joint stereo,
    /// §2.4.3.2 intensity stereo (r286), forced block types, outer
    /// loop, CRC, Xing — is supported.
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
    /// Caller invoked [`Mp3Encoder::set_per_band_xmin`] on an encoder
    /// that does not have the §C.1.5.4.3 outer (distortion-control)
    /// loop enabled. The per-band threshold vector is only consumed by
    /// the outer-loop path — install the outer loop first via
    /// [`Mp3Encoder::new_with_outer_loop`].
    PerBandXminWithoutOuterLoop,
    /// The intensity-stereo start band passed to
    /// [`Mp3Encoder::new_joint_stereo_is`] /
    /// [`Mp3Encoder::new_joint_stereo_ms_is`] /
    /// [`Mp3Encoder::new_joint_stereo_auto_is`] is out of range. The
    /// MPEG-1 long-block layout has 21 scalefactor bands (§2.4.2.7
    /// `scalefac_l[0..21]`); the start band must leave at least one
    /// normally-coded band below it and one intensity-coded band at or
    /// above it, i.e. `1 ..= 20`.
    InvalidIntensityStartSfb {
        /// The rejected start band.
        start_sfb: usize,
    },
    /// A block-type toggle that the intensity-stereo encode path does
    /// not yet support was requested on an encoder with intensity
    /// coupling armed. Supported: force-short and force-mixed short
    /// coupling (r303 / r305 / r306), and signal-driven
    /// `enable_auto_block_type` scheduling with MS-joint stereo (r307) OR
    /// on the intensity-only path (r308) — in both cases the
    /// §2.4.3.4.9 channel agreement is forced by mirroring the channel-0
    /// scheduler emission across both channels, so the per-band fold
    /// geometry is well-defined. Still rejected: the mixed-promotion auto
    /// variant (`enable_auto_block_type_with_mixed` — the §2.4.3.4.10.3
    /// carve-out bound is unwired) and the Model-2-driven auto path under
    /// intensity.
    IntensityShortBlocksUnsupported,
    /// [`Mp3Encoder::set_per_band_xmin_from_model2`] was called with a
    /// granule whose length is not `SAMPLES_PER_GRANULE` (576), or on an
    /// encoder whose sample rate is not one of the three Annex D Model 2
    /// rates (32 / 44.1 / 48 kHz). The §C.1.5.3.2.1 Model 2 analysis
    /// is built on Annex D Tables D.3 / D.4 / C.7 / C.8, which the docs
    /// stage only for those three rates; the carried value is the
    /// offending sample rate (0 signals a granule-length mismatch).
    Model2AnalysisUnsupported {
        /// The encoder sample rate that has no staged Model 2 tables, or
        /// `0` when the failure was a granule-length mismatch.
        sample_rate_hz: u32,
    },
    /// Mixed blocks were requested on an 8 kHz (MPEG-2.5) encoder —
    /// [`Mp3Encoder::force_mixed_blocks_for_testing`] or
    /// [`Mp3Encoder::enable_auto_block_type_with_mixed`]. The mixed
    /// carve-out fixes the long region at the two lowest polyphase
    /// subbands (36 lines) and starts the short region at the short
    /// band whose tripled start index is 36; the deployed 8 kHz
    /// Fraunhofer short table (per-window starts 0, 8, 16, 24, …) has
    /// **no** band boundary at per-window line 12. The r408
    /// observer probes resolved the CODING layout — all four deployed
    /// black-box validators requantize an 8 kHz mixed granule with a
    /// 72-line long-coded region (`3·short_starts[3]`, matching the
    /// six transmitted LSF long scalefactor bands; see
    /// `requantize::mixed_long_lines`) — but the deployed world still
    /// splits 3-1 on the WINDOW geometry (three validators keep the
    /// §2.4.2.7 two-subband / 36-line window split; one long-windows
    /// the whole 72-line region), so an emitted 8 kHz mixed stream
    /// would render differently on a quarter of deployed decoders.
    /// The encoder therefore refuses to emit mixed granules at 8 kHz;
    /// pure short blocks (and the plain auto block-type path) are
    /// fully supported there. The DECODER renders foreign 8 kHz mixed
    /// granules per the majority reading (r408; previously lines
    /// 36..72 were left silent).
    MixedBlocks8kUnsupported,
    /// [`Mp3Encoder::enable_auto_block_type_model2`] was called on an
    /// encoder that does not have the §C.1.5.3.2.1 automatic Model 2
    /// psychoacoustics armed. The Model-2-driven block-type path reuses
    /// the Model 2 per-granule `pe > 1800` decision as its attack
    /// signal, so [`Mp3Encoder::enable_model2_psychoacoustics`] must be
    /// enabled first (which in turn fixes the sampling rate to one of
    /// the three staged Annex D Model 2 rates).
    Model2BlockTypeWithoutModel2,
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
            StreamEncodeError::LsfUnsupported => f.write_str(
                "feature not yet ported to the LSF (MPEG-2 / MPEG-2.5) wire format \
                 (auto block-type remains MPEG-1 only)",
            ),
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
            StreamEncodeError::PerBandXminWithoutOuterLoop => f.write_str(
                "set_per_band_xmin requires the outer loop to be enabled first (use Mp3Encoder::new_with_outer_loop)",
            ),
            StreamEncodeError::InvalidIntensityStartSfb { start_sfb } => write!(
                f,
                "intensity-stereo start band {start_sfb} out of range (must be 1..=20: at least one normal band below the bound and one intensity band at or above it)"
            ),
            StreamEncodeError::IntensityShortBlocksUnsupported => f.write_str(
                "this block-type toggle is not supported with intensity-stereo coupling armed (mixed-promotion auto and Model-2-driven auto remain unavailable; force-short / force-mixed and signal-driven auto block-type — MS-joint or intensity-only — are supported)",
            ),
            StreamEncodeError::MixedBlocks8kUnsupported => f.write_str(
                "mixed blocks are not supported at 8 kHz: deployed decoders disagree on \
                 the window geometry of the MPEG-2.5 8 kHz mixed carve-out (use pure \
                 short blocks)",
            ),
            StreamEncodeError::Model2AnalysisUnsupported { sample_rate_hz } => {
                if *sample_rate_hz == 0 {
                    f.write_str(
                        "set_per_band_xmin_from_model2 requires a 576-sample granule",
                    )
                } else {
                    write!(
                        f,
                        "Model 2 analysis is staged only for 32 / 44.1 / 48 kHz (got {sample_rate_hz} Hz)"
                    )
                }
            }
            StreamEncodeError::Model2BlockTypeWithoutModel2 => f.write_str(
                "enable_auto_block_type_model2 requires enable_model2_psychoacoustics to be armed first (the block-type path reuses the Model 2 pe > 1800 decision)",
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

    /// When `Some`, the long-block outer-loop branch consumes the
    /// per-band threshold vector `xmin[sb]` from this
    /// [`crate::psy::XminThresholds`] instead of the uniform scalar
    /// stashed in [`Self::outer_loop_threshold`]. The short / mixed
    /// branches in this round still consume the uniform scalar (their
    /// `*_per_band` outer-loop variants land in a follow-up — see
    /// [`crate::psy`]'s scope note).
    ///
    /// **Activation rules** — the per-band path requires the outer
    /// loop to be enabled in the first place (i.e.
    /// [`Self::outer_loop_threshold`] is also `Some`). Setting
    /// `per_band_xmin` without first calling
    /// [`Self::new_with_outer_loop`] is rejected at API time.
    ///
    /// Set by [`Self::set_per_band_xmin`] (the only public path that
    /// installs an [`crate::psy::XminThresholds`]). Default `None`
    /// preserves the pre-r194 uniform-threshold behaviour byte-for-byte.
    per_band_xmin: Option<crate::psy::XminThresholds>,

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
    /// `mode_extension` bits. Set by [`Mp3Encoder::new_joint_stereo_ms`]
    /// (MS only, the full spectrum) and
    /// [`Mp3Encoder::new_joint_stereo_ms_is`] (MS below the intensity
    /// bound — see [`Self::intensity_start_sfb`]); requires `nch == 2`.
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

    /// When `Some(b)`, §2.4.3.4.9.3 **intensity-stereo** coupling is
    /// armed: every granule's long scalefactor bands `b..21` (plus the
    /// partial region above the last band boundary) are rewritten so
    /// the left channel carries the combined magnitude `L + R`, the
    /// right channel carries zeros, and the right channel's per-band
    /// scalefactor doubles as the stereo position
    ///
    /// ```text
    /// is_pos[sfb] = NINT( (12/π) · arctan( √(E_L[sfb] / E_R[sfb]) ) )
    /// ```
    ///
    /// (Annex G.2 c) of ISO/IEC 11172-3:1993; positions `0..=6`, `7` is
    /// the illegal-position marker). The decoder reverses the coupling
    /// per §2.4.3.4.9.3: `is_ratio = tan(is_pos·π/12)`,
    /// `L' = T·is_ratio/(1+is_ratio)`, `R' = T/(1+is_ratio)` with `T`
    /// the transmitted left-channel value.
    ///
    /// All-zero right-channel bands **below** the bound that follow the
    /// last non-zero right-channel line are transmitted with
    /// scalefactor `7` (Annex G.2 c): "scalefactor bands of the
    /// right/difference channel containing only zeros after coding
    /// which do not belong to the intensity coded part should be
    /// transmitted with the scalefactor 7 to prevent intensity stereo
    /// decoding") — the §2.4.3.4.9.1 decode-side bound is derived from
    /// the zero-part, so without the marker a decoder would treat those
    /// bands as intensity-coded with a bogus position.
    ///
    /// Set by [`Mp3Encoder::new_joint_stereo_is`] (`mode_extension =
    /// '01'`), [`Mp3Encoder::new_joint_stereo_ms_is`] (`'11'`: MS below
    /// the bound, intensity above), and
    /// [`Mp3Encoder::new_joint_stereo_auto_is`] (per-frame `'11'` /
    /// `'01'` from the MS picker). Long-block only this round (the
    /// short-window per-window bound is deferred); the block-type
    /// toggles reject while this is `Some`.
    intensity_start_sfb: Option<usize>,

    /// `Some(t)` when the per-granule **adaptive intensity-bound**
    /// chooser is armed (constructor
    /// [`Mp3Encoder::new_joint_stereo_auto_is_adaptive`]). When set,
    /// [`Self::intensity_start_sfb`] is treated as a *floor* — the
    /// lowest band the encoder is allowed to couple — and the actual
    /// coupling start band is chosen per granule as the lowest band
    /// `b >= floor` such that **every** band in `b..21` carries little
    /// right-channel stereo information relative to the combined
    /// magnitude, i.e. its side-energy fraction
    /// `E_S / (E_L + E_R) = Σ(L−R)² / (2·Σ(L²+R²)) <= t`. A band that
    /// still carries real stereo content (fraction above `t`) raises
    /// the bound so it (and everything below it) stays independently
    /// coded. With no qualifying tail the granule is coded with no
    /// intensity coupling at all (effective bound = 21).
    ///
    /// The decoder derives the intensity bound implicitly from the
    /// position of the right channel's last non-zero line
    /// (§2.4.3.4.9.1), so the per-granule bound varies freely on the
    /// wire with no syntax change: a higher chosen bound simply means
    /// the right channel's zero-part starts higher. ISO/IEC 11172-3
    /// fixes only the wire syntax; the energy heuristic mirrors the
    /// §2.4.3.4.9.2 MS picker ([`Self::ms_auto_threshold`]) and is a
    /// clean-room encoder choice using no psychoacoustic input.
    intensity_auto_threshold: Option<f64>,

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
    /// * **MS-stereo not supported (r162 widening).** Mono and
    ///   independent stereo ([`ChannelMode::Stereo`] /
    ///   [`ChannelMode::DualChannel`]) accept the toggle; MS-stereo
    ///   joint modes ([`Mp3Encoder::new_joint_stereo_ms`] /
    ///   [`Mp3Encoder::new_joint_stereo_auto`]) still reject because
    ///   §2.4.3.4.9 requires both channels of an MS-stereo granule to
    ///   share `block_type` / `window_switching_flag` /
    ///   `mixed_block_flag`, and that cross-channel-MS agreement
    ///   wiring is the next follow-up.
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

    /// When `Some`, every assembled frame's per-granule block type is
    /// chosen automatically from the PCM input by the
    /// [`crate::attack_detect::AttackDetector`] + the
    /// [`crate::block_type_sm::BlockTypeStateMachine`] (§C.1.5.2
    /// `LONG → START → SHORT → STOP → LONG` transition geometry),
    /// instead of forcing every granule onto a single block type via
    /// the [`Self::force_short_blocks_for_testing`] /
    /// [`Self::force_mixed_blocks_for_testing`] testing toggles.
    ///
    /// The carried struct holds the per-channel detector + scheduler
    /// state across calls to [`Self::push_samples`] / [`Self::finish`].
    ///
    /// **Lookahead model.** The §C.1.5.2 transition geometry needs
    /// **one granule of lookahead** to insert the `Start` window in
    /// time for the next granule's `Short`. The encoder achieves this
    /// by holding back the last granule of each
    /// [`Self::push_samples`] burst until the next granule arrives,
    /// so a frame is emitted only when its second granule's
    /// successor is known (`finish` flushes the held-back granules
    /// as a final frame with a zero-padded lookahead, equivalent to
    /// "no attack ahead").
    ///
    /// **Channel-mode restriction.** Relaxed in r162 to accept
    /// independent stereo ([`ChannelMode::Stereo`] /
    /// [`ChannelMode::DualChannel`]) in addition to mono — the
    /// per-channel scheduler vector picks each channel's §C.1.5.2
    /// transition state independently, mirroring §2.4.1.7 / §2.4.2.7
    /// which carry per-channel side-info verbatim. MS-stereo joint
    /// modes still rejected; they require the §2.4.3.4.9
    /// cross-channel agreement wiring deferred to a follow-up.
    /// Mutually exclusive with both
    /// [`Self::force_short_blocks_for_testing`] and
    /// [`Self::force_mixed_blocks_for_testing`].
    ///
    /// Default `None`; the encoder behaves as in every previous
    /// round (every granule emits a long block).
    auto_block_type: Option<AutoBlockTypeConfig>,

    /// When `Some`, the encoder runs the §C.1.5.3.2.1 Layer III
    /// Model 2 psychoacoustic analysis **per granule, automatically**
    /// inside the encode loop and installs the resulting signal-
    /// dependent `xmin(sb)` vector before each granule's outer-loop
    /// search — the running-state generalisation of the one-shot
    /// [`Self::set_per_band_xmin_from_model2`] convenience.
    ///
    /// One [`crate::psy::Model2Layer3State`] per channel (vector
    /// length `nch`), each threaded across the granules of the whole
    /// stream so the §D.2.1 FFT-history requirement ("the model needs
    /// a known starting point and runs continuously") is honoured: a
    /// channel's state carries the previous granule's spectrum into
    /// the next granule's unpredictability prediction, exactly as the
    /// spec's continuous analysis demands.
    ///
    /// Set by [`Self::enable_model2_psychoacoustics`]. Mutually
    /// exclusive at API time with a caller-installed static
    /// [`Self::per_band_xmin`] (enabling clears it, and
    /// [`Self::set_per_band_xmin`] clears this). Restricted to the
    /// three staged Annex D Model 2 sampling rates (32 / 44.1 /
    /// 48 kHz) and to 576-sample granules — guaranteed by the
    /// constructor guard, so the per-granule analysis below never
    /// observes an unsupported rate.
    ///
    /// Default `None`: the encoder uses whatever static threshold the
    /// outer-loop / per-band setters installed (or the uniform
    /// scalar), byte-for-byte as in every previous round.
    model2_psy: Option<Vec<crate::psy::Model2Layer3State>>,

    /// Per-(granule, channel) §C.1.5.3.2.1 window-switching decision
    /// captured from the most recently encoded frame's automatic
    /// Model 2 Pass-1 walk, when
    /// [`Self::enable_model2_psychoacoustics`] is armed.
    ///
    /// Each cell holds the [`crate::psy::Model2Layer3Granule`]'s
    /// `pe` (the §C.1.5.3.2.1 psychoacoustic entropy of the long path)
    /// and `attack` (`pe > 1800`, the §C.1.5.3.2 short-block
    /// switching condition) for that granule/channel. The Model 2
    /// analysis already computes both as it derives the per-band
    /// `xmin(sb)` threshold; this field simply retains the deliverable
    /// instead of discarding it, making the spec-canonical
    /// window-switching signal observable through
    /// [`Self::last_model2_window_switch`].
    ///
    /// `None` outer (no frame yet encoded with the mode armed) or
    /// `None` per cell (a granule/channel the last frame did not
    /// populate — e.g. the second granule of an LSF frame, which has
    /// only one granule). Reset to a fresh per-frame matrix at the
    /// start of every `push_samples`/`flush` frame assembly so the
    /// accessor always reflects exactly the last emitted frame.
    last_model2_switch: Option<[[Option<Model2WindowSwitch>; 2]; GRANULES]>,

    /// Per-channel §C.1.5.2 block-type schedulers driven by the
    /// §C.1.5.3.2.1 **Model 2** `pe > 1800` window-switching decision,
    /// when [`Self::enable_auto_block_type_model2`] is armed.
    ///
    /// This is the Model-2-driven counterpart of the energy-detector
    /// [`Self::auto_block_type`] path: instead of an
    /// [`crate::attack_detect::AttackDetector`] classifying each
    /// granule from subframe energy, the per-granule attack flag is the
    /// spec-canonical §C.1.5.3.2.1 psychoacoustic-entropy decision
    /// (`pe > 1800`) that the armed Model 2 analysis already computes
    /// while deriving the per-band `xmin(sb)` threshold. The same
    /// per-channel [`crate::block_type_sm::BlockTypeStateMachine`]
    /// translates those flags into the §C.1.5.2
    /// `LONG → START → SHORT → STOP → LONG` transition geometry, so the
    /// emitted window sequence is identical in shape to the energy
    /// path — only the attack signal differs.
    ///
    /// Requires [`Self::enable_model2_psychoacoustics`] (this mode
    /// shares its per-channel Model 2 states; the analysis runs once
    /// per granule and its output feeds **both** the block-type
    /// decision and the outer-loop threshold). Mutually exclusive with
    /// the energy-detector auto path and the force toggles.
    ///
    /// Default `None`: the emitted block type is governed by the
    /// energy-detector auto path, the force toggles, or the all-long
    /// default, byte-for-byte as in every previous round.
    model2_block_type: Option<Vec<crate::block_type_sm::BlockTypeStateMachine>>,
    /// The named [`crate::quality::QualityPreset`] most recently applied
    /// via [`Self::with_quality_preset`], recorded for the
    /// [`Self::quality_preset`] accessor. `None` until a preset is
    /// applied; a record of intent only (raw `enable_*` calls after a
    /// preset do not clear it).
    quality_preset: Option<crate::quality::QualityPreset>,
    /// `true` when the last-applied preset armed the signal-dependent
    /// Model 2 path (vs the signal-independent threshold-in-quiet
    /// fallback). Only meaningful while `quality_preset.is_some()`.
    quality_preset_signal_dependent: bool,
    /// §D.1 Step 3 threshold offset (dB) the Model 2 per-granule install
    /// applies to its geometric-mean anchor
    /// ([`crate::psy::XminThresholds::from_layer3_granule_with_offset_db`]).
    /// `0.0` reproduces the unoffset Model 2 path; a quality preset sets
    /// it so the preset's level reaches the signal-dependent path. Reset
    /// to `0.0` whenever Model 2 is (re-)armed outside a preset.
    model2_offset_db: f64,
    /// §C.1.5.3 scalefactor-selection-information (scfsi) reuse.
    ///
    /// When `true`, the MPEG-1 two-granule assembler examines each
    /// channel's pair of granules after quantization and, for every
    /// long-block scfsi_band group whose granule-1 scalefactors are
    /// byte-identical to granule 0's, sets `scfsi[ch][group] = 1` so
    /// those scalefactors are transmitted once (in granule 0) instead
    /// of twice. The decoder reuses granule 0's values verbatim for
    /// the marked groups (ISO/IEC 11172-3 §2.4.2.7 / decode notes
    /// p.??), so the reconstructed scalefactors — and therefore every
    /// decoded sample — are bit-identical; only the part2 scalefactor
    /// bits of granule 1 shrink.
    ///
    /// **Default `true` as of r301: scfsi reuse is auto-armed.** Because
    /// the detection is byte-exact (a group is marked only when the two
    /// granules' scalefactors already agree across every band in it) and
    /// the decoder reconstructs granule 0's values for a marked group,
    /// auto-arming is lossless by construction: the reconstructed PCM is
    /// identical to the historical `scfsi = 0` output while granule 1's
    /// part2 budget shrinks wherever consecutive granules naturally share
    /// scalefactors. The optimisation never fires on LSF (MPEG-2 /
    /// MPEG-2.5 have no scfsi field and one granule) nor on any channel
    /// whose either granule is a short block (§2.4.2.7: "if short windows
    /// are switched on … then scfsi is always 0 for this frame").
    ///
    /// [`Self::disable_scfsi_reuse`] forces the historical
    /// byte-for-byte `scfsi = 0` output back on; [`Self::enable_scfsi_reuse`]
    /// is retained (now a no-op on a fresh encoder) for callers that
    /// re-arm after an explicit disable.
    scfsi_reuse: bool,
}

/// The §C.1.5.3.2.1 Layer III window-switching deliverable for one
/// granule/channel, as computed by the automatic Model 2 analysis.
///
/// `pe` is the long-path psychoacoustic entropy; `attack` is the
/// `pe > 1800` short-block switching condition the spec derives from
/// it (Phase 2 step 81 of [`crate::psy`] transcribes the threshold
/// from the staged ISO PDF). This is the signal an encoder uses to
/// decide whether a granule should switch to short blocks; the
/// automatic Model 2 path surfaces it here so callers can inspect the
/// decision the analysis reached for the granules of the last frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Model2WindowSwitch {
    /// §C.1.5.3.2.1 psychoacoustic entropy of the long path for this
    /// granule/channel.
    pub pe: f64,
    /// `pe > 1800` — the §C.1.5.3.2 short-block switching condition
    /// (strictly greater than the 1800 threshold).
    pub attack: bool,
}

/// Per-channel auto-block-type state for [`Mp3Encoder`]. Holds the
/// stateful attack detector + the block-type scheduler that
/// translate the per-granule attack flags into the §C.1.5.2 transition
/// geometry. One detector + one scheduler per channel; r162 widened
/// this to accept independent stereo (`nch == 2`, no MS-stereo
/// coupling) by running an independent scheduler per channel — each
/// channel of an independent-stereo granule picks its own §C.1.5.2
/// transition state. MS-stereo joint modes remain rejected at API
/// time so this struct never holds an MS-stereo configuration.
#[derive(Debug, Clone)]
struct AutoBlockTypeConfig {
    /// Attack-detection ratio threshold (per-subframe energy over the
    /// running ambient). See
    /// [`crate::attack_detect::AttackDetector`] for semantics.
    threshold: f64,
    /// Per-channel attack detectors. Their ambient-energy estimate
    /// carries across [`Mp3Encoder::push_samples`] calls.
    detector: Vec<crate::attack_detect::AttackDetector>,
    /// Per-channel block-type schedulers. Their `prev` block-type
    /// carries across [`Mp3Encoder::push_samples`] calls so a burst
    /// that spans frame boundaries still emits a single coherent
    /// `Long → Start → Short → Stop → Long` sequence.
    scheduler: Vec<crate::block_type_sm::BlockTypeStateMachine>,
    /// Optional mixed-vs-pure-short classifier per channel. When
    /// `Some`, each Short emission from the scheduler is candidate
    /// for promotion to mixed (block_type 2 + mixed_block_flag = 1)
    /// driven by the per-channel
    /// [`crate::mixed_classifier::MixedClassifier`]'s low-band
    /// stability ratio. When `None`, every Short is pure-short — the
    /// pre-r161 auto behaviour. Configured via
    /// [`Mp3Encoder::enable_auto_block_type_with_mixed`].
    mixed_classifier: Option<Vec<crate::mixed_classifier::MixedClassifier>>,
    /// Configured low-band stability threshold (cached for
    /// inspection via [`Mp3Encoder::auto_block_type_mixed_threshold`];
    /// `None` when no mixed classifier is wired).
    mixed_threshold: Option<f64>,
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
    /// Sample rates: 32 / 44.1 / 48 kHz (MPEG-1, ISO/IEC 11172-3,
    /// 1152 samples / two granules per frame), 16 / 22.05 / 24 kHz
    /// (MPEG-2 LSF, ISO/IEC 13818-3 §2.4.3.2: one 576-sample granule
    /// per frame, `slots_per_frame` constant 72), and 8 / 11.025 /
    /// 12 kHz (MPEG-2.5, the Fraunhofer-IIS lower-rate extension that
    /// inherits the 13818-3 LSF framing — see
    /// `docs/audio/mp3/MPEG-2.5-GAP.md`). The version is inferred from
    /// the sample rate; the LSF versions emit the §2.4.1.7 LSF
    /// side-info layout (8-bit `main_data_begin`, no `scfsi`, 9-bit
    /// `scalefac_compress`) and the §2.4.2.3 LSF bitrate ladder.
    ///
    /// # Errors
    ///
    /// * [`StreamEncodeError::StereoUnsupported`] for
    ///   [`ChannelMode::JointStereo`] (the only unsupported mode this
    ///   round).
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
        let nch = header_template.channel_count() as usize;
        let analysis_state = (0..nch).map(|_| AnalysisState::new()).collect();
        let mdct_state = (0..nch)
            .map(|_| (0..32usize).map(|_| MdctState::new()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let pending_pcm = (0..nch).map(|_| Vec::new()).collect();
        Ok(Mp3Encoder {
            header_template,
            sample_rate_hz,
            version: header_template.version,
            nch,
            analysis_state,
            mdct_state,
            pending_pcm,
            frames: Vec::new(),
            outer_loop_threshold: None,
            per_band_xmin: None,
            xing_template: None,
            vbr: None,
            crc_enabled: false,
            ms_stereo: false,
            ms_auto_threshold: None,
            intensity_start_sfb: None,
            intensity_auto_threshold: None,
            force_short_blocks: false,
            force_mixed_blocks: false,
            auto_block_type: None,
            model2_psy: None,
            last_model2_switch: None,
            model2_block_type: None,
            quality_preset: None,
            quality_preset_signal_dependent: false,
            model2_offset_db: 0.0,
            scfsi_reuse: true,
        })
    }

    /// `true` when the encoder is in a §2.4.3.4.9.2 MS-stereo joint mode
    /// — either the unconditional MS path
    /// ([`Mp3Encoder::new_joint_stereo_ms`], `self.ms_stereo == true`)
    /// or the per-frame MS/LR picker
    /// ([`Mp3Encoder::new_joint_stereo_auto`],
    /// `self.ms_auto_threshold.is_some()`).
    ///
    /// Used as the "do we need cross-channel block-type agreement?"
    /// predicate in `encode_frame`'s pre-pass. r163 wired the
    /// agreement: when this returns `true`, the encoder reduces
    /// `block_type_per_gc[gr][L]` and `[gr][R]` to a single shared
    /// `(block_type, mixed_flag)` per granule before MDCT dispatch by
    /// running a single shared §C.1.5.2 scheduler driven by the OR of
    /// per-channel attack flags (auto path) or by emitting the
    /// trivially-agreed `[Short; 2]` (force-short / force-mixed
    /// paths). The four block-type override toggles (force-short,
    /// force-mixed, auto, auto+mixed) now accept MS-stereo joint
    /// modes; the predicate is retained for the per-pass agreement
    /// branch and for documentation of which mode is active.
    #[must_use]
    fn ms_joint_stereo_active(&self) -> bool {
        self.ms_stereo || self.ms_auto_threshold.is_some()
    }

    /// Granules per frame: 2 for MPEG-1 (ISO/IEC 11172-3 §2.4.1.7),
    /// 1 for MPEG-2 / MPEG-2.5 LSF (ISO/IEC 13818-3 §2.4.3.2: "a
    /// Layer III frame contains only one granule").
    #[must_use]
    fn granules_per_frame(&self) -> usize {
        if self.version.is_lsf() {
            GRANULES_LSF
        } else {
            GRANULES
        }
    }

    /// PCM samples consumed per assembled frame and per channel:
    /// `granules_per_frame() × 576` — 1152 for MPEG-1, 576 for the LSF
    /// versions (ISO/IEC 13818-3 §2.4.3.2 "the number of samples per
    /// frame is 576").
    #[must_use]
    fn samples_per_frame(&self) -> usize {
        self.granules_per_frame() * SAMPLES_PER_GRANULE
    }

    /// Force every assembled granule onto the §2.4.2.7 short-block
    /// (`block_type = 2`) encode path; see the
    /// [`Self::force_short_blocks`] field for the per-granule
    /// behavioural contract.
    ///
    /// Channel-mode acceptance (widened in r163): every channel layout
    /// the encoder supports — mono ([`ChannelMode::SingleChannel`],
    /// `nch == 1`), independent stereo ([`ChannelMode::Stereo`] /
    /// [`ChannelMode::DualChannel`]), and MS-stereo joint modes
    /// ([`Mp3Encoder::new_joint_stereo_ms`] /
    /// [`Mp3Encoder::new_joint_stereo_auto`]) — accepts the toggle.
    /// Force-short trivially satisfies the §2.4.3.4.9 MS agreement
    /// constraint because every (gr, ch) tile emits the same
    /// `BlockType::Short`.
    ///
    /// Combining force-short with intensity stereo and the *auto*-MS
    /// picker ([`Mp3Encoder::new_joint_stereo_auto_is`]) is supported:
    /// the §2.4.3.4.9.2 side-energy picker measures the per-window short
    /// MS region (the contiguous `0..3*short_starts[short_start]` run for
    /// each granule's per-window bound), matching exactly the line set
    /// the MS rotation applies below each window's short intensity bound.
    ///
    /// # Errors
    ///
    /// Returns `Ok` for every channel layout this round; the
    /// previous round's [`StreamEncodeError::StereoUnsupported`]
    /// guard was dropped when the cross-channel-MS agreement wiring
    /// landed (r163), and the r306 picker-region fix dropped the
    /// auto-MS + short + intensity [`StreamEncodeError::IntensityShortBlocksUnsupported`]
    /// rejection.
    pub fn force_short_blocks_for_testing(
        &mut self,
        enabled: bool,
    ) -> Result<(), StreamEncodeError> {
        if enabled {
            // Mixed and pure-short are mutually exclusive: a granule is
            // long, short, or mixed. Enabling pure-short clears mixed.
            self.force_mixed_blocks = false;
            // Force-toggles and auto block-type are mutually
            // exclusive: enabling a force-toggle clears auto (both the
            // energy-detector and the Model-2-driven paths).
            self.auto_block_type = None;
            self.model2_block_type = None;
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
    /// Channel-mode acceptance (widened in r163; see
    /// [`Self::force_short_blocks_for_testing`] for the §2.4.3.4.9
    /// rationale): every channel layout the encoder supports — mono,
    /// independent stereo, and MS-stereo joint modes — accepts the
    /// toggle. Force-mixed trivially satisfies the §2.4.3.4.9
    /// MS-agreement constraint because every (gr, ch) tile emits the
    /// same `(BlockType::Short, mixed_block_flag = true)` pair.
    ///
    /// Mutually exclusive with [`Self::force_short_blocks_for_testing`].
    ///   Enabling this resets `force_short_blocks` to `false` so a
    ///   single granule cannot ask for both at once.
    ///
    /// # Errors
    ///
    /// Returns `Ok` for every channel layout this round; the
    /// previous round's [`StreamEncodeError::StereoUnsupported`]
    /// guard was dropped in r163.
    pub fn force_mixed_blocks_for_testing(
        &mut self,
        enabled: bool,
    ) -> Result<(), StreamEncodeError> {
        if enabled && self.sample_rate_hz == 8000 {
            // Deployed decoders split 3-1 on the WINDOW geometry of
            // the 8 kHz mixed carve-out (r408 observer probes; the
            // coding layout itself is resolved — see
            // [`StreamEncodeError::MixedBlocks8kUnsupported`]) —
            // refuse to emit.
            return Err(StreamEncodeError::MixedBlocks8kUnsupported);
        }
        if enabled && self.intensity_start_sfb.is_some() && self.ms_joint_stereo_active() {
            // §2.4.3.4.10.3 mixed-block intensity coupling is wired for
            // the *intensity-only* (non-MS) path (r311): the carve-out's
            // long lowest 2 subbands take the long-band intensity walk and
            // the upper short region takes the per-window short walk (the
            // exact two-region geometry the decoder's `process_short`
            // applies for a `mixed_block_flag` granule). The MS-joint
            // combination — the §2.4.3.4.9.2 below-bound rotation over the
            // mixed block's split line set — remains a follow-up, so it
            // stays rejected.
            return Err(StreamEncodeError::IntensityShortBlocksUnsupported);
        }
        if enabled {
            // Mixed and pure-short are mutually exclusive: a granule is
            // long, short, or mixed. Enabling mixed clears short.
            self.force_short_blocks = false;
            // Force-toggles and auto block-type are mutually
            // exclusive: enabling a force-toggle clears auto (both the
            // energy-detector and the Model-2-driven paths).
            self.auto_block_type = None;
            self.model2_block_type = None;
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

    /// Enable **signal-driven auto block-type** dispatch. With this
    /// on, every assembled frame's per-granule
    /// [`crate::side_info::BlockType`] is chosen by the
    /// [`crate::attack_detect::AttackDetector`] +
    /// [`crate::block_type_sm::BlockTypeStateMachine`] pair, instead
    /// of the all-long default or the
    /// [`Self::force_short_blocks_for_testing`] /
    /// [`Self::force_mixed_blocks_for_testing`] all-forced testing
    /// paths.
    ///
    /// The detector's ratio threshold (subframe energy over the
    /// running ambient that the loudest subframe must exceed for the
    /// granule to be flagged) is supplied here; pass
    /// [`crate::attack_detect::DEFAULT_ATTACK_THRESHOLD`] for the
    /// suggested-default `10.0`. See
    /// [`crate::attack_detect`] for tuning guidance.
    ///
    /// The lookahead model uses one extra granule of latency, taken
    /// out of the encoder's existing buffering envelope so the
    /// `push_samples` / `finish` API contract is unchanged for
    /// callers (the trailing granule is zero-padded at `finish`).
    ///
    /// Restrictions:
    ///
    /// * **Channel-mode.** Widened in r163: accepted on mono,
    ///   independent stereo ([`ChannelMode::Stereo`] /
    ///   [`ChannelMode::DualChannel`]), AND MS-stereo joint modes
    ///   ([`Mp3Encoder::new_joint_stereo_ms`] /
    ///   [`Mp3Encoder::new_joint_stereo_auto`]). Independent stereo
    ///   runs one detector + scheduler per channel (each channel
    ///   picks its own §C.1.5.2 transition state). MS-stereo runs
    ///   per-channel detectors (so each channel keeps a coherent
    ///   ambient estimate) but folds the per-channel attack flags via
    ///   logical OR into a single shared scheduler before stepping —
    ///   that scheduler's emission is mirrored across both channels,
    ///   so the §2.4.3.4.9 "both channels of an MS-stereo granule
    ///   share the same `block_type` /
    ///   `window_switching_flag`" agreement holds by construction.
    /// * **Mutually exclusive with the force-toggles.** Enabling auto
    ///   clears [`Self::force_short_blocks`] and
    ///   [`Self::force_mixed_blocks`] (the testing toggles), and vice
    ///   versa.
    ///
    /// # Errors
    ///
    /// Returns `Ok` for every channel layout, including intensity-stereo
    /// coupling with or without MS-joint stereo. r307 lifted the
    /// [`StreamEncodeError::IntensityShortBlocksUnsupported`] rejection
    /// for the MS+intensity combination; r308 lifts it for the remaining
    /// intensity-only path ([`Mp3Encoder::new_joint_stereo_is`]). The
    /// concern that the auto scheduler runs an independent state machine
    /// per channel — so the per-granule block types could diverge between
    /// L and R while intensity coupling folds each granule's `(L, R)`
    /// band-by-band and needs both channels to share window geometry
    /// (§2.4.3.4.9) — is resolved structurally: whenever intensity
    /// coupling is armed, `block_types_for_frame` forces the same
    /// channel-agreement OR-fold the MS path uses (channel-0's emission
    /// mirrored across both channels, the per-channel attack flags
    /// OR-folded into one shared scheduler). The block types are
    /// therefore channel-consistent by construction, so each granule's
    /// Short / Long intensity coupling is well-defined.
    ///
    /// The previous round's [`StreamEncodeError::StereoUnsupported`]
    /// guard was dropped in r163.
    ///
    /// As of r287 the auto block-type scheduler is **version-agnostic**:
    /// the §C.1.5.2 walk in `assemble_frame_with_lookahead` steps the
    /// per-channel state machine `granules_per_frame()` times — twice
    /// per frame on MPEG-1 (the two-granule geometry) and once per frame
    /// on LSF (MPEG-2 16 / 22.05 / 24 kHz and MPEG-2.5 8 / 11.025 /
    /// 12 kHz, where ISO/IEC 13818-3 frames carry a single 576-sample
    /// granule). The §2.4.3.4.10.3 window-switching geometry and the
    /// per-granule attack/lookahead model are identical across versions;
    /// only the granule count per frame differs, so the LSF walk reuses
    /// the same detector + scheduler + mixed-classifier wiring with one
    /// scheduler step per frame and the next frame's single granule as
    /// the §C.1.5.2 lookahead.
    pub fn enable_auto_block_type(&mut self, threshold: f64) -> Result<(), StreamEncodeError> {
        // Auto block-type under intensity coupling (r307: MS-joint armed;
        // r308: intensity-only too). Intensity coupling folds each
        // granule's `(L, R)` band-by-band, which requires both channels
        // to share window geometry (§2.4.3.4.9 channel agreement). r307
        // covered the case where MS-joint stereo is also armed: MS-joint
        // (`new_joint_stereo_ms` / `new_joint_stereo_auto`) mirrors one
        // shared scheduler emission across both channels, so the
        // agreement holds by construction. r308 lifts the remaining
        // intensity-only rejection: when intensity coupling is armed, the
        // encode-side scheduler walk forces the SAME channel-agreement
        // OR-fold (channel-0's emission mirrored across both channels)
        // regardless of MS — `channel_agreement_active` in
        // `block_types_for_frame` is now `MS-joint OR intensity-armed`.
        // That makes the per-granule block types channel-consistent by
        // construction, so every Short granule takes the §2.4.3.4.9.3
        // per-window short coupling and the long-family granules take the
        // long-block band walk, exactly as the MS path already did.
        //
        // Outer loop is now compatible with auto block-type for every
        // block-type the auto scheduler ever emits:
        //   * Long granules — `outer_loop_search_long`
        //   * Short granules (`mixed_block_flag == false`) —
        //     `outer_loop_search_short`
        //   * Start / End transition granules — `outer_loop_search_long`
        //     (the long-family primitive; see its doc on long-family
        //     acceptance — same part2 wire layout, same requantize
        //     formula, same region-split rule as `Long`). Added in
        //     r160: Start / End were previously fixed-gain fallbacks.
        // Mixed granules are unreachable from the auto scheduler this
        // round (no Mixed transition in §C.1.5.2's
        // `LONG → START → SHORT → STOP` path).
        // Mutually exclusive with the force-toggles and the
        // Model-2-driven block-type path; clear them.
        self.force_short_blocks = false;
        self.force_mixed_blocks = false;
        self.model2_block_type = None;
        let detector: Vec<_> = (0..self.nch)
            .map(|_| crate::attack_detect::AttackDetector::with_threshold(threshold))
            .collect();
        let scheduler: Vec<_> = (0..self.nch)
            .map(|_| crate::block_type_sm::BlockTypeStateMachine::new())
            .collect();
        // Read back the effective threshold (the detector clamps
        // pathological inputs to the default, so capture what it
        // settled on).
        let effective_threshold = detector
            .first()
            .map(|d| d.threshold())
            .unwrap_or(crate::attack_detect::DEFAULT_ATTACK_THRESHOLD);
        self.auto_block_type = Some(AutoBlockTypeConfig {
            threshold: effective_threshold,
            detector,
            scheduler,
            mixed_classifier: None,
            mixed_threshold: None,
        });
        Ok(())
    }

    /// Enable §C.1.5.2 auto block-type **with mixed-block
    /// promotion** — the r161 extension to
    /// [`Self::enable_auto_block_type`]. Identical lookahead /
    /// detector / scheduler wiring, plus an additional
    /// [`crate::mixed_classifier::MixedClassifier`] per channel that
    /// decides — on every granule the scheduler emits as Short —
    /// whether to promote it to a §2.4.3.4.10.3 mixed block
    /// (`block_type == 2`, `mixed_block_flag == 1`: lowest 2
    /// subbands long, the rest short) or keep it pure-short.
    ///
    /// The mixed classifier is a clean-room PCM-domain one-tap
    /// low-pass + subframe-energy ratio. A small ratio means the
    /// low band is stationary across the granule → the mixed
    /// carve-out's long lowest subbands are appropriate (they keep
    /// frequency resolution on the steady low-frequency content
    /// while letting the upper subbands resolve the transient in
    /// time). A large ratio means the low band is bursting too → a
    /// pure-short block is preferred (every subband resolves the
    /// burst in time). See [`crate::mixed_classifier`] for the
    /// signal-driven heuristic in detail.
    ///
    /// `attack_threshold` is the same parameter
    /// [`Self::enable_auto_block_type`] takes (subframe-energy /
    /// running-ambient ratio that the loudest subframe must exceed
    /// to be flagged as carrying an attack); pass
    /// [`crate::attack_detect::DEFAULT_ATTACK_THRESHOLD`] for the
    /// suggested default `10.0`.
    /// `mixed_low_band_stability` is the mixed classifier's
    /// max-to-min low-passed subframe-energy ratio that the granule
    /// must stay at or below to be promoted to mixed; pass
    /// [`crate::mixed_classifier::DEFAULT_MIXED_LOW_BAND_STABILITY`]
    /// for the suggested default `4.0`. Both thresholds are checked
    /// for non-finite / non-positive values and silently coerced to
    /// their respective defaults.
    ///
    /// All [`Self::enable_auto_block_type`] restrictions apply
    /// (mutually exclusive with the force-toggles); as of r163 every
    /// channel layout is accepted, including MS-stereo joint modes
    /// (the per-channel mixed classifier vector mirrors the per-
    /// channel detector vector; on MS-stereo the per-granule
    /// classifier flags are OR-folded together before stepping the
    /// single shared scheduler, so the §2.4.3.4.9 agreement holds
    /// over `mixed_block_flag` too). Mixed promotion is an *opt-in
    /// extension* — callers using `enable_auto_block_type` keep the
    /// pre-r161 pure-short behaviour.
    ///
    /// **Rate scope of the promotion (r408):** mixed bursts are put on
    /// the wire only at the MPEG-1 rates (32 / 44.1 / 48 kHz). A mixed
    /// burst's flanking `Start` / `End` granules must carry the
    /// §2.4.2.7 `mixed_block_flag` for the low-subband §2.4.3.4
    /// overlap-add to cancel, and while that combination is conformant
    /// at every rate (the ISO/IEC 13818-3 main_data syntax scopes the
    /// mixed scalefactor layout to `block_type == '10'`, and its
    /// scalefac_compress partition tables mark the flag don't-care for
    /// block types '00'/'01'/'11'), r408 black-box measurements found
    /// deployed LSF decoders split 2-2 on it — two track the spec
    /// reading float-perfectly, two desynchronise on the burst. At the
    /// LSF / MPEG-2.5 rates the scheduler therefore demotes mixed
    /// bursts to pure-short (identically decoded everywhere); the
    /// classifier still runs and the toggle is still accepted, but no
    /// mixed granule reaches the wire from the auto path at those
    /// rates. [`Self::force_mixed_blocks_for_testing`] (steady mixed
    /// streams, no transition flanks) remains available at every rate
    /// except 8 kHz.
    ///
    /// # Errors
    ///
    /// Returns `Ok` for every channel layout this round; the
    /// previous round's [`StreamEncodeError::StereoUnsupported`]
    /// guard was dropped in r163.
    pub fn enable_auto_block_type_with_mixed(
        &mut self,
        attack_threshold: f64,
        mixed_low_band_stability: f64,
    ) -> Result<(), StreamEncodeError> {
        // Deployed decoders disagree on the 8 kHz mixed window
        // geometry (see [`StreamEncodeError::MixedBlocks8kUnsupported`]);
        // the plain [`Self::enable_auto_block_type`] path
        // (long/start/short/stop) is the supported auto configuration
        // there.
        if self.sample_rate_hz == 8000 {
            return Err(StreamEncodeError::MixedBlocks8kUnsupported);
        }
        // Mixed-block intensity coupling is not wired (the §2.4.3.4.10.3
        // mixed carve-out's long-lowest-subbands + short-rest split needs
        // a two-region intensity bound that this round does not derive).
        // r307 lifted the rejection on the *pure-short* auto path under
        // MS-joint stereo, but the mixed-promotion variant stays rejected
        // when intensity is armed — guard before the plain path would
        // accept it.
        if self.intensity_start_sfb.is_some() {
            return Err(StreamEncodeError::IntensityShortBlocksUnsupported);
        }
        // Reuse the plain path for detector / scheduler construction
        // + the MS-stereo reject + force-toggle clearing it does,
        // then augment the resulting config with the mixed classifier
        // (one classifier per channel; the per-channel vector is
        // independent so each channel of an independent-stereo
        // granule classifies its own low-band stability).
        self.enable_auto_block_type(attack_threshold)?;
        let mixed_classifier: Vec<_> = (0..self.nch)
            .map(|_| {
                crate::mixed_classifier::MixedClassifier::with_threshold(mixed_low_band_stability)
            })
            .collect();
        let effective_mixed_threshold = mixed_classifier
            .first()
            .map(|c| c.threshold())
            .unwrap_or(crate::mixed_classifier::DEFAULT_MIXED_LOW_BAND_STABILITY);
        if let Some(ref mut cfg) = self.auto_block_type {
            cfg.mixed_classifier = Some(mixed_classifier);
            cfg.mixed_threshold = Some(effective_mixed_threshold);
        }
        Ok(())
    }

    /// `true` when [`Self::enable_auto_block_type`] is in effect.
    #[must_use]
    pub fn auto_block_type_enabled(&self) -> bool {
        self.auto_block_type.is_some()
    }

    /// The active auto-block-type threshold (subframe-energy /
    /// running-ambient ratio that the loudest subframe of a granule
    /// must exceed to be flagged as carrying an attack). Returns
    /// `None` when auto block-type is not enabled.
    #[must_use]
    pub fn auto_block_type_threshold(&self) -> Option<f64> {
        self.auto_block_type.as_ref().map(|c| c.threshold)
    }

    /// `true` when [`Self::enable_auto_block_type_with_mixed`] is in
    /// effect (a strict subset of [`Self::auto_block_type_enabled`]
    /// — the mixed classifier is an opt-in extension on top of the
    /// auto dispatch).
    #[must_use]
    pub fn auto_block_type_mixed_enabled(&self) -> bool {
        self.auto_block_type
            .as_ref()
            .map(|c| c.mixed_classifier.is_some())
            .unwrap_or(false)
    }

    /// The configured mixed-promotion low-band stability threshold
    /// (max-to-min subframe-energy ratio of the low-passed PCM that
    /// a granule must stay at or below to be promoted from
    /// pure-short to mixed). `None` when
    /// [`Self::enable_auto_block_type_with_mixed`] was not called.
    #[must_use]
    pub fn auto_block_type_mixed_threshold(&self) -> Option<f64> {
        self.auto_block_type
            .as_ref()
            .and_then(|c| c.mixed_threshold)
    }

    /// Disable the [`Self::enable_auto_block_type`] auto dispatch
    /// (returns the encoder to the all-long default path). No-op
    /// when auto was not enabled.
    pub fn disable_auto_block_type(&mut self) {
        self.auto_block_type = None;
    }

    /// Enable **§C.1.5.3.2.1 Model-2-driven auto block-type** dispatch.
    ///
    /// This is the spec-canonical counterpart of the energy-detector
    /// [`Self::enable_auto_block_type`]: the per-granule attack signal
    /// fed into the §C.1.5.2 `LONG → START → SHORT → STOP → LONG`
    /// scheduler is the **Model 2** window-switching decision —
    /// `pe > 1800`, where `pe` is the §C.1.5.3.2.1 psychoacoustic
    /// entropy of the long path — rather than a PCM-domain
    /// subframe-energy ratio. The transition geometry (a
    /// [`crate::block_type_sm::BlockTypeStateMachine`] per channel) is
    /// the same as the energy path; only the attack signal differs.
    ///
    /// The Model 2 analysis that yields `pe`/`attack` for the
    /// block-type decision is the *same* per-channel
    /// [`crate::psy::Model2Layer3State`] walk that
    /// [`Self::enable_model2_psychoacoustics`] arms to derive the
    /// per-band `xmin(sb)` outer-loop threshold. To keep the §D.2.1
    /// continuous-FFT-history advancing exactly once per granule, this
    /// mode runs the walk in the encode-loop pre-pass and **reuses the
    /// captured output** for the granule's outer-loop threshold — the
    /// analysis is never run twice for the same granule. The lookahead
    /// granule (the §C.1.5.2 `next_attack` anticipation) is evaluated
    /// non-destructively by cloning the channel state, so the borrowed
    /// next-frame PCM never perturbs the committed FFT history.
    ///
    /// Channel coupling mirrors the energy path:
    ///
    /// * **Independent** (mono / `Stereo` / `DualChannel`): one
    ///   scheduler per channel, each picking its own §C.1.5.2
    ///   transition state (§2.4.1.7 / §2.4.2.7 carry per-channel
    ///   side-info verbatim).
    /// * **MS-stereo** (`new_joint_stereo_ms` / `new_joint_stereo_auto`):
    ///   the per-channel `pe > 1800` flags are OR-folded into a single
    ///   shared scheduler whose emission is mirrored across both
    ///   channels of the granule, so the §2.4.3.4.9 "both channels of
    ///   an MS-stereo granule share the same `block_type` /
    ///   `window_switching_flag`" agreement holds by construction.
    ///
    /// Mutually exclusive with the energy-detector
    /// [`Self::enable_auto_block_type`] path and the force toggles —
    /// enabling this clears them, and enabling any of them clears this.
    ///
    /// As of r313 this path accepts **intensity-stereo coupling** (with
    /// or without MS-joint stereo), mirroring the energy-detector
    /// [`Self::enable_auto_block_type`] acceptance (r307 MS+intensity,
    /// r308 intensity-only). Arming intensity coupling forces the same
    /// §2.4.3.4.9 channel-agreement OR-fold the energy path uses
    /// (`channel_agreement_active = MS-joint OR intensity-armed`): the
    /// per-channel `pe > 1800` flags are OR-folded into one shared
    /// (channel-0) scheduler whose emission is mirrored across both
    /// channels of each granule, so L/R block types stay consistent and
    /// the per-window short / long-block intensity coupling is
    /// well-defined. A granule the Model-2 scheduler emits as **pure
    /// short** takes the §2.4.3.4.9.3 per-window short coupling; Long /
    /// Start / End granules take the long-block band walk. The §C.1.5.2
    /// walk never emits a mixed block, so the §2.4.3.4.10.3 mixed
    /// carve-out coupling is not exercised here.
    ///
    /// # Errors
    ///
    /// * [`StreamEncodeError::Model2BlockTypeWithoutModel2`] — the
    ///   automatic Model 2 psychoacoustics are not armed. Call
    ///   [`Self::enable_model2_psychoacoustics`] first; that
    ///   constructor also fixes the sampling rate to one of the three
    ///   staged Annex D Model 2 rates (32 / 44.1 / 48 kHz, all MPEG-1
    ///   two-granule frames).
    pub fn enable_auto_block_type_model2(&mut self) -> Result<(), StreamEncodeError> {
        if self.model2_psy.is_none() {
            return Err(StreamEncodeError::Model2BlockTypeWithoutModel2);
        }
        // Intensity-stereo coupling is now accepted (r313): the
        // frame-assembly `channel_agreement_active` OR-fold already keyed
        // off `intensity_start_sfb.is_some()` for the Model-2 emission
        // path, and Pass 1 already selects the per-granule short / long
        // intensity coupling from the same `block_type_per_gc` matrix this
        // path produces. The §C.1.5.2 walk emits no mixed block, so the
        // §2.4.3.4.10.3 mixed carve-out coupling is never reached.
        //
        // Mutually exclusive with the force-toggles and the
        // energy-detector auto path; clear them.
        self.force_short_blocks = false;
        self.force_mixed_blocks = false;
        self.auto_block_type = None;
        let scheduler: Vec<_> = (0..self.nch)
            .map(|_| crate::block_type_sm::BlockTypeStateMachine::new())
            .collect();
        self.model2_block_type = Some(scheduler);
        Ok(())
    }

    /// `true` when [`Self::enable_auto_block_type_model2`] is in
    /// effect.
    #[must_use]
    pub fn auto_block_type_model2_enabled(&self) -> bool {
        self.model2_block_type.is_some()
    }

    /// Disable the [`Self::enable_auto_block_type_model2`] dispatch
    /// (returns the encoder to the all-long default path; the Model 2
    /// per-band threshold path stays armed). No-op when the
    /// Model-2-driven block-type mode was not enabled.
    pub fn disable_auto_block_type_model2(&mut self) {
        self.model2_block_type = None;
    }

    /// Apply a named psychoacoustic [`QualityPreset`], arming the
    /// perceptual machinery as one call instead of a sequence of
    /// individual `enable_*` toggles.
    ///
    /// This is the quality-knob front-end the README's "remaining work"
    /// item asked for: a caller that has already chosen a transport
    /// (bitrate, sampling rate, channel mode) via one of the
    /// `new_with_outer_loop`-family constructors picks *how aggressively
    /// the quantization noise is shaped under the perceptual model* with
    /// a single named level. Each preset lowers to the spec-grounded
    /// bundle documented on [`crate::quality`]:
    ///
    /// * a §D.1 Step 3 threshold offset
    ///   ([`QualityPresetParams::threshold_offset_db`]) applied to the
    ///   per-band threshold-in-quiet bowl;
    /// * optionally the §C.1.5.3.2.1 **Model 2** per-band masking
    ///   analysis ([`Self::enable_model2_psychoacoustics`]);
    /// * optionally the §C.1.5.2 Model-2-driven **block-type** scheduler
    ///   ([`Self::enable_auto_block_type_model2`]).
    ///
    /// **Rate-graceful.** The Model 2 analysis is staged only for the
    /// three Annex D rates (32 / 44.1 / 48 kHz). When the encoder's rate
    /// is one of those and the preset requests Model 2, this arms the
    /// full signal-dependent path. When the rate is **not** an Annex D
    /// rate (the MPEG-2 LSF and MPEG-2.5 extension rates), the Model 2
    /// flags cannot be honoured — there are no staged calculation-
    /// partition tables — so the encoder installs the signal-independent
    /// per-band threshold-in-quiet vector translated by the preset's
    /// `offset_db` instead, via [`Self::set_per_band_xmin`]. Either way
    /// the call succeeds; the perceptual model used is the richest the
    /// staged tables allow for that rate, and
    /// [`Self::quality_preset_is_signal_dependent`] reports which path
    /// was taken.
    ///
    /// Applying a preset is **idempotent and re-applicable**: a later
    /// `with_quality_preset` (or any of the individual `enable_*` /
    /// `set_per_band_xmin` calls) overrides an earlier one. The preset
    /// last applied is recorded and surfaced by [`Self::quality_preset`].
    ///
    /// The preset does not touch the channel mode, the target bitrate, or
    /// the stereo-coupling decision; those stay exactly as the caller
    /// constructed them.
    ///
    /// # Errors
    ///
    /// * [`StreamEncodeError::PerBandXminWithoutOuterLoop`] — the encoder
    ///   has no outer loop (it was built via [`Mp3Encoder::new`] rather
    ///   than [`Mp3Encoder::new_with_outer_loop`] or a
    ///   `new_with_threshold_in_quiet*` variant). A perceptual preset has
    ///   nowhere to feed without the §C.1.5.4.3 outer loop.
    pub fn with_quality_preset(
        &mut self,
        preset: crate::quality::QualityPreset,
    ) -> Result<(), StreamEncodeError> {
        if self.outer_loop_threshold.is_none() {
            return Err(StreamEncodeError::PerBandXminWithoutOuterLoop);
        }
        let params = preset.params();
        // Whether the configured rate can run the Model 2 analysis at all
        // (the staged Annex D calculation-partition tables exist only for
        // 32 / 44.1 / 48 kHz).
        let rate_is_annex_d =
            crate::psy::AnnexDSamplingRate::from_hz(self.sample_rate_hz).is_some();
        let signal_dependent = params.model2_threshold && rate_is_annex_d;

        // Start from a clean perceptual slate so re-applying a preset is
        // deterministic regardless of what was armed before.
        self.disable_auto_block_type_model2();

        if signal_dependent {
            // Annex D rate + preset wants Model 2: arm the per-granule
            // signal-dependent analysis. The §D.1 Step 3 offset is folded
            // into the Model 2 path through the same geometric-mean anchor
            // the analysis uses, so the masking threshold remains the
            // content-driven one; the offset shapes the static fallback
            // the analysis collapses to on a silent granule.
            self.enable_model2_psychoacoustics()?;
            // Thread the preset's §D.1 Step 3 offset into the Model 2
            // per-granule anchor so the preset's level (not just its
            // per-band shape) reaches the signal-dependent path.
            self.model2_offset_db = params.threshold_offset_db;
            if params.model2_block_type {
                // Cannot fail: Model 2 was just armed above.
                self.enable_auto_block_type_model2()?;
            }
        } else {
            // Non-Annex-D rate, or a preset that intentionally skips
            // Model 2: install the signal-independent per-band threshold-
            // in-quiet vector translated by the preset's §D.1 Step 3
            // offset. This works at every rate the encoder supports.
            let xmin = crate::psy::XminThresholds::threshold_in_quiet_with_offset_db(
                self.sample_rate_hz,
                self.version,
                params.threshold_offset_db,
            );
            // Cannot fail: outer-loop threshold is present (checked above).
            self.set_per_band_xmin(xmin)?;
        }

        self.quality_preset = Some(preset);
        self.quality_preset_signal_dependent = signal_dependent;
        Ok(())
    }

    /// The [`QualityPreset`] most recently applied via
    /// [`Self::with_quality_preset`], or `None` if no preset has been
    /// applied (the encoder is on whatever threshold its constructor
    /// installed, or a preset was overridden by a later raw
    /// `enable_*` / `set_per_band_xmin` call).
    ///
    /// Note this is a *record of intent*, not a live reflection of the
    /// armed toggles: calling `enable_model2_psychoacoustics` or
    /// `set_per_band_xmin` directly after a preset does **not** clear this
    /// field. Use the individual `*_enabled` accessors to query the live
    /// toggle state.
    #[must_use]
    pub fn quality_preset(&self) -> Option<crate::quality::QualityPreset> {
        self.quality_preset
    }

    /// `true` when the last-applied [`Self::with_quality_preset`] armed
    /// the **signal-dependent** Model 2 path (the rate was a staged Annex
    /// D rate and the preset requested Model 2); `false` when the preset
    /// fell back to the signal-independent threshold-in-quiet vector
    /// (non-Annex-D rate, or a preset like [`crate::quality::QualityPreset::Fast`]
    /// that skips Model 2). Returns `false` when no preset has been
    /// applied.
    #[must_use]
    pub fn quality_preset_is_signal_dependent(&self) -> bool {
        self.quality_preset.is_some() && self.quality_preset_signal_dependent
    }

    /// The static per-band threshold vector currently installed (via
    /// [`Self::set_per_band_xmin`], a `new_with_threshold_in_quiet*`
    /// constructor, or a [`Self::with_quality_preset`] that fell back to
    /// the signal-independent path), or `None` when no static vector is
    /// installed.
    ///
    /// Returns `None` while the per-granule Model 2 analysis is armed: the
    /// signal-dependent path overwrites the vector on every granule, so
    /// there is no stable static vector to surface (query
    /// [`Self::model2_psychoacoustics_enabled`] /
    /// [`Self::quality_preset_is_signal_dependent`] instead). Exposed for
    /// integration tests / observability so a caller can confirm a
    /// preset's §D.1 Step 3 offset reached the installed threshold on the
    /// fallback path.
    #[must_use]
    pub fn installed_per_band_xmin(&self) -> Option<&crate::psy::XminThresholds> {
        self.per_band_xmin.as_ref()
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
    /// spectrum is decoded in MS-stereo"). For the intensity-stereo
    /// half of joint stereo (§2.4.3.4.9.3) see
    /// [`Mp3Encoder::new_joint_stereo_is`] /
    /// [`Mp3Encoder::new_joint_stereo_ms_is`].
    /// Both granules of a frame share the same block type (Long,
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

    /// Build a joint-stereo encoder that emits §2.4.3.4.9.3
    /// **intensity-stereo** frames (`mode = '01'`, `mode_extension =
    /// '01'`: intensity_stereo on, ms_stereo off).
    ///
    /// Long scalefactor bands below `intensity_start_sfb` carry the two
    /// channels independently (plain L/R). Bands at or above it are
    /// **intensity-coupled** per Annex G.2 c) of ISO/IEC 11172-3:1993:
    ///
    /// * the left channel transmits the combined magnitude
    ///   `L_i + R_i` over the band,
    /// * the right channel transmits zeros over the band, and
    /// * the right channel's scalefactor for the band is replaced by
    ///   the stereo position
    ///   `is_pos[sfb] = NINT((12/π)·arctan(√(E_L[sfb]/E_R[sfb])))`
    ///   (positions `0..=6`; a band with zero right-channel energy maps
    ///   to the `R → 0` limit `6`; `7` is reserved as the
    ///   illegal-position marker per §2.4.3.4.9.3).
    ///
    /// A conformant decoder derives the intensity bound from the
    /// zero-part of the right channel (§2.4.3.4.9.1 / §2.4.3.4.9.3) and
    /// reconstructs `L'_i = T_i·is_ratio/(1+is_ratio)`,
    /// `R'_i = T_i/(1+is_ratio)` with `is_ratio = tan(is_pos·π/12)` and
    /// `T` the transmitted left-channel band. All-zero right-channel
    /// bands below the bound are transmitted with scalefactor `7` so
    /// they are **not** intensity-decoded (Annex G.2 c) guidance; they
    /// would otherwise extend the decoder-derived zero-part downward).
    ///
    /// The spectrum above the last long band boundary (Table B.8
    /// `scalefac_l` covers 21 bands; the top lines up to 576 belong to
    /// no band) carries no scalefactor and therefore no position; those
    /// lines are coupled into the left channel the same way and decode
    /// as left-only — the §2.4.3.4.9.3 layout simply has no is_pos slot
    /// for them.
    ///
    /// On an **LSF** (MPEG-2 / MPEG-2.5) rate the same coupling geometry
    /// applies, but the right channel is written in the ISO/IEC 13818-3
    /// §2.4.3.2 format: `scalefac_compress = 258` selects the
    /// right-channel partition `int_scalefac_compress = 129 < 180` ⇒
    /// `slen = (3, 3, 3, 0)` / `nr_of_sfb = (7, 7, 7, 0)` (3 bits on
    /// every one of the 21 long bands; `7` is the max value, hence the
    /// illegal-position marker) with `intensity_scale = 0`. The decoder
    /// reconstructs through the §2.4.3.2 power-law `i0 = 2^(-1/4)`
    /// ladder (`kl`/`kr` step-4/5 replacement) rather than the MPEG-1
    /// `tan` grid, so the encoder derives the positions on that ladder.
    ///
    /// Long-block only this round: the short-window intensity bound is
    /// per window (each window has its own zero-part), so the
    /// block-type toggles ([`Mp3Encoder::force_short_blocks_for_testing`]
    /// / [`Mp3Encoder::force_mixed_blocks_for_testing`] /
    /// [`Mp3Encoder::enable_auto_block_type`]) reject while intensity
    /// coupling is armed.
    ///
    /// # Errors
    ///
    /// Same as [`Mp3Encoder::new`], plus
    /// [`StreamEncodeError::InvalidIntensityStartSfb`] when
    /// `intensity_start_sfb` is outside `1..=20`.
    pub fn new_joint_stereo_is(
        bitrate_kbps: u32,
        sample_rate_hz: u32,
        intensity_start_sfb: usize,
    ) -> Result<Self, StreamEncodeError> {
        let mut enc = Self::new(bitrate_kbps, sample_rate_hz, ChannelMode::Stereo)?;
        enc.arm_intensity(intensity_start_sfb)?;
        enc.header_template.mode = ChannelMode::JointStereo;
        enc.header_template.mode_extension = ModeExtension {
            intensity_stereo: true,
            ms_stereo: false,
            raw: 0b01,
        };
        Ok(enc)
    }

    /// Build a joint-stereo encoder that combines §2.4.3.4.9.2
    /// **MS-stereo** below the intensity bound with §2.4.3.4.9.3
    /// **intensity stereo** at and above it (`mode = '01'`,
    /// `mode_extension = '11'`).
    ///
    /// Bands below `intensity_start_sfb` are MS-coded: the forward
    /// rotation `M = (L+R)/√2`, `S = (L−R)/√2` is applied to lines
    /// `[0, starts[intensity_start_sfb])` only — §2.4.3.4.9.1 scopes
    /// the MS equations to the scalefactor bands below the intensity
    /// bound when both methods are enabled. Bands at or above the bound
    /// take the same intensity coupling as
    /// [`Mp3Encoder::new_joint_stereo_is`].
    ///
    /// # Errors
    ///
    /// Same as [`Mp3Encoder::new_joint_stereo_is`].
    pub fn new_joint_stereo_ms_is(
        bitrate_kbps: u32,
        sample_rate_hz: u32,
        intensity_start_sfb: usize,
    ) -> Result<Self, StreamEncodeError> {
        let mut enc = Self::new(bitrate_kbps, sample_rate_hz, ChannelMode::Stereo)?;
        enc.arm_intensity(intensity_start_sfb)?;
        enc.header_template.mode = ChannelMode::JointStereo;
        enc.header_template.mode_extension = ModeExtension {
            intensity_stereo: true,
            ms_stereo: true,
            raw: 0b11,
        };
        enc.ms_stereo = true;
        Ok(enc)
    }

    /// Build a joint-stereo encoder with intensity coupling always on
    /// and the §2.4.3.4.9.2 MS rotation decided per frame by the
    /// [`Mp3Encoder::new_joint_stereo_auto`] energy picker, evaluated
    /// over the below-bound lines only (the region MS would apply to).
    /// Every frame carries `mode = '01'`; the per-frame
    /// `mode_extension` is `'11'` (MS + intensity) when both granules
    /// qualify for MS and `'01'` (intensity only) otherwise.
    ///
    /// # Errors
    ///
    /// Same as [`Mp3Encoder::new_joint_stereo_is`].
    pub fn new_joint_stereo_auto_is(
        bitrate_kbps: u32,
        sample_rate_hz: u32,
        intensity_start_sfb: usize,
    ) -> Result<Self, StreamEncodeError> {
        let mut enc = Self::new_joint_stereo_auto(bitrate_kbps, sample_rate_hz)?;
        enc.arm_intensity(intensity_start_sfb)?;
        // The auto picker rewrites the per-frame mode_extension; start
        // from the intensity-only pattern so even a hypothetical
        // pre-picker frame carries the armed coupling.
        enc.header_template.mode_extension = ModeExtension {
            intensity_stereo: true,
            ms_stereo: false,
            raw: 0b01,
        };
        Ok(enc)
    }

    /// Build a joint-stereo encoder whose §2.4.3.4.9.3 **intensity
    /// bound is chosen per granule** from the signal, rather than fixed
    /// at construction.
    ///
    /// `intensity_start_floor` is the *lowest* long scalefactor band the
    /// encoder is permitted to intensity-couple (it must satisfy the
    /// same `1..=20` range as [`Mp3Encoder::new_joint_stereo_is`]).
    /// Bands below the floor are always coded independently. For each
    /// granule the chooser scans the bands `floor..21` and couples only
    /// the **contiguous high tail** whose every band carries little
    /// right-channel stereo information, measured by the same
    /// side-energy fraction the §2.4.3.4.9.2 MS picker uses:
    ///
    /// ```text
    /// E_S / (E_L + E_R) = Σ(L − R)² / (2·Σ(L² + R²))   over the band
    /// ```
    ///
    /// The effective start band is the lowest `b >= floor` such that
    /// every band in `b..21` has fraction `<= threshold` (default
    /// `0.25`). A band that still carries real stereo content (fraction
    /// above the threshold) pushes the bound up so that band — and
    /// everything below it — stays independently coded. If no high tail
    /// qualifies, the granule emits **no** intensity coupling at all
    /// (its right channel keeps full spectral data).
    ///
    /// The decoder derives the intensity bound implicitly from the
    /// right channel's last non-zero line (§2.4.3.4.9.1), so a
    /// per-granule bound needs no wire-syntax change: a higher bound
    /// just means the right channel's zero-part begins higher up. The
    /// frame header carries `mode = '01'` with `mode_extension = '01'`
    /// (intensity on, MS off) on every frame; a granule that couples
    /// nothing still rides under that header but transmits a full
    /// right channel, which the decoder reconstructs as ordinary
    /// stereo (its derived bound lands at band 21).
    ///
    /// Like the MS picker, ISO/IEC 11172-3 does **not** prescribe how
    /// to pick the bound — §2.4.2.3 fixes only the `mode_extension`
    /// syntax. This is a clean-room encoder heuristic using no
    /// psychoacoustic input and no external implementation as a
    /// reference.
    ///
    /// # Errors
    ///
    /// Same as [`Mp3Encoder::new_joint_stereo_is`].
    pub fn new_joint_stereo_auto_is_adaptive(
        bitrate_kbps: u32,
        sample_rate_hz: u32,
        intensity_start_floor: usize,
    ) -> Result<Self, StreamEncodeError> {
        let mut enc = Self::new(bitrate_kbps, sample_rate_hz, ChannelMode::Stereo)?;
        enc.arm_intensity(intensity_start_floor)?;
        enc.intensity_auto_threshold = Some(0.25);
        enc.header_template.mode = ChannelMode::JointStereo;
        enc.header_template.mode_extension = ModeExtension {
            intensity_stereo: true,
            ms_stereo: false,
            raw: 0b01,
        };
        Ok(enc)
    }

    /// `Some(t)` when the per-granule adaptive intensity-bound chooser
    /// is armed (see [`Mp3Encoder::new_joint_stereo_auto_is_adaptive`]),
    /// where `t` is the upper bound on a band's side-energy fraction at
    /// which it qualifies to join the intensity-coupled high tail.
    #[must_use]
    pub fn intensity_auto_threshold(&self) -> Option<f64> {
        self.intensity_auto_threshold
    }

    /// Override the side-energy fraction threshold used by the adaptive
    /// intensity-bound chooser (see
    /// [`Mp3Encoder::new_joint_stereo_auto_is_adaptive`]).
    ///
    /// `threshold` is the upper bound on a band's `E_S / (E_L + E_R)`
    /// at which it may join the intensity-coupled high tail. Values
    /// outside `[0.0, 1.0]` are clamped to that range. Calling this on
    /// an encoder that was **not** constructed via
    /// [`Mp3Encoder::new_joint_stereo_auto_is_adaptive`] is a no-op.
    #[must_use]
    pub fn with_intensity_auto_threshold(mut self, threshold: f64) -> Self {
        if self.intensity_auto_threshold.is_some() {
            self.intensity_auto_threshold = Some(threshold.clamp(0.0, 1.0));
        }
        self
    }

    /// Shared validation + arming for the intensity-stereo
    /// constructors. `start_sfb` must leave at least one normally-coded
    /// long band below the bound and one intensity band at or above it.
    fn arm_intensity(&mut self, start_sfb: usize) -> Result<(), StreamEncodeError> {
        // Both MPEG-1 (ISO/IEC 11172-3 §2.4.3.4.9.3) and LSF (ISO/IEC
        // 13818-3 §2.4.3.2) intensity stereo are supported. The two
        // share the coupling geometry (left := L+R, right := zero-part,
        // positions in the right channel's scalefactor slots) but differ
        // on the wire: the LSF right channel routes its 9-bit
        // `scalefac_compress` through `int_scalefac_compress =
        // scalefac_compress >> 1` with its own partition tables, an
        // `intensity_scale = scalefac_compress % 2` selector, and a
        // power-law `i0` reconstruction ladder (§2.4.3.2 step 4/5
        // replacement) rather than the MPEG-1 `tan` grid. Pass 2 picks
        // the version-appropriate `scalefac_compress` and position
        // derivation below.
        if !(1..=20).contains(&start_sfb) {
            return Err(StreamEncodeError::InvalidIntensityStartSfb { start_sfb });
        }
        self.intensity_start_sfb = Some(start_sfb);
        Ok(())
    }

    /// Arm §2.4.3.4.9.3 intensity-stereo coupling on an already-built
    /// stereo encoder, emitting `mode = '01'` /
    /// `mode_extension = '01'` (intensity on, MS off) — the running-state
    /// counterpart of the [`Mp3Encoder::new_joint_stereo_is`]
    /// constructor for encoders that were built another way (e.g. via
    /// [`Mp3Encoder::new_with_outer_loop`], so the §C.1.5.3.2.1 Model 2
    /// per-band threshold and the Model-2-driven block-type scheduler can
    /// be armed alongside intensity coupling).
    ///
    /// `intensity_start_sfb` is the first intensity-coded long
    /// scalefactor band (same `1..=20` meaning as the constructors).
    ///
    /// # Errors
    ///
    /// * [`StreamEncodeError::StereoUnsupported`] — the encoder is not a
    ///   two-channel encoder; intensity coupling folds an `(L, R)` pair.
    /// * [`StreamEncodeError::InvalidIntensityStartSfb`] —
    ///   `intensity_start_sfb` is outside `1..=20`.
    pub fn enable_intensity_stereo(
        &mut self,
        intensity_start_sfb: usize,
    ) -> Result<(), StreamEncodeError> {
        if self.nch != 2 {
            return Err(StreamEncodeError::StereoUnsupported);
        }
        self.arm_intensity(intensity_start_sfb)?;
        self.header_template.mode = ChannelMode::JointStereo;
        self.header_template.mode_extension = ModeExtension {
            intensity_stereo: true,
            ms_stereo: false,
            raw: 0b01,
        };
        Ok(())
    }

    /// `true` when §2.4.3.4.9.3 intensity-stereo coupling is armed
    /// (see [`Mp3Encoder::new_joint_stereo_is`] /
    /// [`Mp3Encoder::new_joint_stereo_ms_is`] /
    /// [`Mp3Encoder::new_joint_stereo_auto_is`] /
    /// [`Mp3Encoder::enable_intensity_stereo`]).
    #[must_use]
    pub fn intensity_stereo_enabled(&self) -> bool {
        self.intensity_start_sfb.is_some()
    }

    /// `Some(b)` when intensity-stereo coupling is armed, where `b` is
    /// the first intensity-coded long scalefactor band.
    #[must_use]
    pub fn intensity_start_sfb(&self) -> Option<usize> {
        self.intensity_start_sfb
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
        let min_idx =
            ladder_index(self.version, min_kbps).ok_or(StreamEncodeError::InvalidVbrConfig)?;
        let max_idx =
            ladder_index(self.version, max_kbps).ok_or(StreamEncodeError::InvalidVbrConfig)?;
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

    /// Build an encoder identical to [`Mp3Encoder::new_with_outer_loop`]
    /// with the per-band Annex D **threshold-in-quiet** vector
    /// ([`crate::psy::XminThresholds::threshold_in_quiet`]) pre-installed
    /// as the §C.1.5.4.3 outer-loop threshold for every block-type
    /// branch (long / pure-short / mixed). One-shot bundle of
    /// [`Mp3Encoder::new_with_outer_loop`] +
    /// [`Mp3Encoder::set_per_band_xmin`] for the
    /// most common psychoacoustic-thresholding recipe: the long-block
    /// outer loop reads `xmin.long[sfb]`, the pure-short loop reads
    /// `xmin.short[sfb][win]`, and the mixed-block loop reads
    /// `xmin.mixed_long[sfb]` / `xmin.mixed_short[sfb][win]`, each band
    /// / cell sampled from the §"Table D.1a" anchors (see
    /// [`crate::psy`] module docstring) with the §D.1 Step 3
    /// `−12 dB` offset applied when
    /// `bitrate_kbps_per_channel >= 96` (i.e.
    /// `bitrate_kbps / nch >= 96`).
    ///
    /// The uniform-scalar outer-loop slot itself is set to
    /// [`DEFAULT_OUTER_LOOP_THRESHOLD`] so any future caller that
    /// re-overrides the per-band vector via
    /// [`Mp3Encoder::set_per_band_xmin`] with a different
    /// [`crate::psy::XminThresholds`] (or installs
    /// [`crate::psy::XminThresholds::uniform`] to revert to uniform)
    /// observes the same convergence dynamics as
    /// [`Mp3Encoder::new_with_outer_loop`] at that threshold.
    ///
    /// `bitrate_kbps_per_channel` is `bitrate_kbps / nch` where `nch`
    /// is the channel count implied by `mode` (1 for
    /// [`ChannelMode::SingleChannel`], 2 for the others). The §D.1
    /// Step 3 offset is keyed on the per-channel bitrate because the
    /// transparency reference is per-channel — at 192 kbit/s stereo
    /// each channel carries 96 kbit/s, exactly the cutover point.
    ///
    /// # Errors
    ///
    /// Same as [`Mp3Encoder::new_with_outer_loop`].
    pub fn new_with_threshold_in_quiet(
        bitrate_kbps: u32,
        sample_rate_hz: u32,
        mode: ChannelMode,
    ) -> Result<Self, StreamEncodeError> {
        let mut enc = Self::new_with_outer_loop(
            bitrate_kbps,
            sample_rate_hz,
            mode,
            DEFAULT_OUTER_LOOP_THRESHOLD,
        )?;
        // `nch` is established by `new()`; divide by it to get the
        // §D.1 Step 3 per-channel reference (192 kbit/s stereo →
        // 96 kbit/s per channel, the cutover).
        let nch = enc.nch as u32;
        let bitrate_per_channel = bitrate_kbps / nch.max(1);
        let xmin = crate::psy::XminThresholds::threshold_in_quiet(
            sample_rate_hz,
            enc.version,
            bitrate_per_channel,
        );
        // Cannot fail: outer-loop threshold was just set above.
        enc.set_per_band_xmin(xmin)?;
        Ok(enc)
    }

    /// Build an encoder identical to [`Mp3Encoder::new_with_threshold_in_quiet`]
    /// except the caller supplies the §D.1 Step 3 dB **offset** directly
    /// instead of letting it be derived from the per-channel bitrate.
    ///
    /// The spec's §D.1 Step 3 mandates exactly two offsets: `−12 dB`
    /// when `bitrate_kbps_per_channel >= 96` and `0 dB` otherwise.
    /// Every spec-conformant transparency target falls into one of those
    /// two values, and callers wanting the spec default should use
    /// [`Mp3Encoder::new_with_threshold_in_quiet`]. This `_offset`
    /// variant exists for quality-knob front-ends (a continuous
    /// transparency slider), VBR encoders that pick a running offset
    /// from a recent-bitrate accumulator, and regression-test sweeps
    /// over the offset.
    ///
    /// `offset_db` is applied uniformly across every band — long,
    /// pure-short, and mixed — on top of the per-frequency `LTq` shape
    /// (the bowl is preserved, the curve is translated up or down by
    /// `offset_db` dB).
    ///
    /// # Errors
    ///
    /// Same as [`Mp3Encoder::new_with_outer_loop`].
    pub fn new_with_threshold_in_quiet_offset(
        bitrate_kbps: u32,
        sample_rate_hz: u32,
        mode: ChannelMode,
        offset_db: f64,
    ) -> Result<Self, StreamEncodeError> {
        let mut enc = Self::new_with_outer_loop(
            bitrate_kbps,
            sample_rate_hz,
            mode,
            DEFAULT_OUTER_LOOP_THRESHOLD,
        )?;
        let xmin = crate::psy::XminThresholds::threshold_in_quiet_with_offset_db(
            sample_rate_hz,
            enc.version,
            offset_db,
        );
        // Cannot fail: outer-loop threshold was just set above.
        enc.set_per_band_xmin(xmin)?;
        Ok(enc)
    }

    /// Build an encoder with the §C.1.5.4.3 outer loop and a named
    /// psychoacoustic [`crate::quality::QualityPreset`] applied in one
    /// call — the one-shot bundle of [`Self::new_with_outer_loop`] (at
    /// [`DEFAULT_OUTER_LOOP_THRESHOLD`]) + [`Self::with_quality_preset`].
    ///
    /// This is the most direct entry point for a quality-knob front-end:
    /// pick the transport (`bitrate_kbps`, `sample_rate_hz`, `mode`) and a
    /// named perceptual level, and the perceptual machinery is armed per
    /// [`Self::with_quality_preset`] (the full signal-dependent Model 2
    /// path at the staged Annex D rates, the signal-independent
    /// threshold-in-quiet fallback elsewhere).
    ///
    /// # Errors
    ///
    /// Same as [`Self::new_with_outer_loop`]; the inner
    /// [`Self::with_quality_preset`] cannot fail here because the outer
    /// loop is always installed first.
    pub fn new_with_quality_preset(
        bitrate_kbps: u32,
        sample_rate_hz: u32,
        mode: ChannelMode,
        preset: crate::quality::QualityPreset,
    ) -> Result<Self, StreamEncodeError> {
        let mut enc = Self::new_with_outer_loop(
            bitrate_kbps,
            sample_rate_hz,
            mode,
            DEFAULT_OUTER_LOOP_THRESHOLD,
        )?;
        // Cannot fail: the outer loop was just installed above.
        enc.with_quality_preset(preset)?;
        Ok(enc)
    }

    /// Install a per-band threshold vector
    /// ([`crate::psy::XminThresholds`]) the long-block outer-loop
    /// branch will consume INSTEAD of the uniform scalar threshold the
    /// encoder was constructed with. The short / mixed branches in this
    /// round still consume the uniform scalar; their per-band variants
    /// land in a follow-up.
    ///
    /// This is the entry point for spectrally-shaped psychoacoustic
    /// thresholds — e.g. the Annex D threshold-in-quiet derived by
    /// [`crate::psy::XminThresholds::threshold_in_quiet_long`]. The
    /// encoder's `xfsf(sb)` over-threshold test for each long-band
    /// scalefactor band switches from `xfsf(sb) > uniform_threshold` to
    /// `xfsf(sb) > xmin.long[sfb]`, so a band with a low per-band
    /// threshold (e.g. the 3.4 kHz region near the LTq minimum) is
    /// amplified more aggressively than a band with a high per-band
    /// threshold (e.g. the bass or treble extremes).
    ///
    /// **Bit-exact compat:** installing
    /// [`crate::psy::XminThresholds::uniform`] with the same value the
    /// encoder was constructed with is functionally a no-op — the
    /// long-block outer-loop branch produces bit-identical output to
    /// the uniform-scalar path (the `*_per_band` primitive is a strict
    /// generalisation; uniform fill broadcasts the scalar over all 21
    /// bands and every comparison resolves identically). This is the
    /// regression-test anchor for the per-band integration.
    ///
    /// # Errors
    ///
    /// Returns [`StreamEncodeError::PerBandXminWithoutOuterLoop`] if the
    /// encoder was constructed via [`Mp3Encoder::new`] (no outer loop) —
    /// install the outer loop first via
    /// [`Mp3Encoder::new_with_outer_loop`].
    pub fn set_per_band_xmin(
        &mut self,
        xmin: crate::psy::XminThresholds,
    ) -> Result<(), StreamEncodeError> {
        if self.outer_loop_threshold.is_none() {
            return Err(StreamEncodeError::PerBandXminWithoutOuterLoop);
        }
        // A static per-band threshold and the per-granule Model 2
        // analysis are mutually exclusive (the latter overwrites
        // `per_band_xmin` on every granule); installing a static
        // vector turns the automatic analysis off, so the captured
        // window-switching decision no longer reflects an active mode.
        self.model2_psy = None;
        self.last_model2_switch = None;
        // The Model-2-driven block-type path sources its attack signal
        // from the now-disarmed Model 2 analysis; disarm it too so the
        // encoder never tries to drive block types from an inactive
        // model (it would otherwise emit all-long with no attack
        // source, but clearing makes the state coherent).
        self.model2_block_type = None;
        self.per_band_xmin = Some(xmin);
        Ok(())
    }

    /// `true` when [`Self::set_per_band_xmin`] has installed a per-band
    /// threshold vector — used by integration tests / observability.
    #[must_use]
    pub fn per_band_xmin_enabled(&self) -> bool {
        self.per_band_xmin.is_some()
    }

    /// Enable **automatic per-granule §C.1.5.3.2.1 Model 2
    /// psychoacoustics** for the whole stream: every granule's PCM is
    /// run through the Model 2 analysis chain and the resulting
    /// signal-dependent `xmin(sb)` vector is installed before that
    /// granule's outer-loop search — no caller bookkeeping per
    /// granule.
    ///
    /// This is the running-state generalisation of the one-shot
    /// [`Self::set_per_band_xmin_from_model2`]: instead of the caller
    /// owning a [`crate::psy::Model2Layer3State`] and calling
    /// `process` + install before each `push_samples`, the encoder
    /// owns one state **per channel** (threaded across every granule
    /// of the stream for the §D.2.1 continuous-FFT-history
    /// requirement) and drives the analysis inside the encode loop.
    ///
    /// Each granule's threshold is the Figure C.6.c/d `thm(sb)`
    /// mapped to the outer loop's `xmin(sb)` exactly as
    /// [`crate::psy::XminThresholds::from_layer3_granule`] specifies
    /// (per-band ratios preserved, geometric-mean offset anchored to
    /// the encoder's outer-loop threshold). On a fully silent granule
    /// the analysis yields the uniform default, so a quiet passage
    /// converges identically to the threshold-in-quiet path.
    ///
    /// Mutually exclusive with a caller-installed static per-band
    /// threshold: enabling this clears any vector previously set via
    /// [`Self::set_per_band_xmin`], and a later
    /// [`Self::set_per_band_xmin`] turns this off.
    ///
    /// # Errors
    ///
    /// * [`StreamEncodeError::PerBandXminWithoutOuterLoop`] — the
    ///   encoder was built without the outer loop (use
    ///   [`Mp3Encoder::new_with_outer_loop`]); the Model 2 threshold
    ///   has nowhere to feed.
    /// * [`StreamEncodeError::Model2AnalysisUnsupported`] — the
    ///   encoder's sampling rate is not one of the three staged
    ///   Annex D Model 2 rates (32 / 44.1 / 48 kHz). The MPEG-2.5 and
    ///   MPEG-2 LSF rates lack staged calculation-partition tables and
    ///   are rejected here rather than guessed.
    pub fn enable_model2_psychoacoustics(&mut self) -> Result<(), StreamEncodeError> {
        if self.outer_loop_threshold.is_none() {
            return Err(StreamEncodeError::PerBandXminWithoutOuterLoop);
        }
        let rate = crate::psy::AnnexDSamplingRate::from_hz(self.sample_rate_hz).ok_or(
            StreamEncodeError::Model2AnalysisUnsupported {
                sample_rate_hz: self.sample_rate_hz,
            },
        )?;
        // One continuous-history state per channel; independent so an
        // independent-stereo (or MS-stereo) pair never shares an FFT
        // history across channels.
        let states = (0..self.nch)
            .map(|_| crate::psy::Model2Layer3State::new(rate))
            .collect::<Vec<_>>();
        // Mutually exclusive with a static per-band vector.
        self.per_band_xmin = None;
        self.model2_psy = Some(states);
        // A raw arm uses the unoffset anchor; a quality preset overrides
        // this immediately after via `with_quality_preset`.
        self.model2_offset_db = 0.0;
        // No frame has been analysed under the freshly-armed mode yet;
        // drop any decision captured by a previous arming.
        self.last_model2_switch = None;
        Ok(())
    }

    /// `true` when [`Self::enable_model2_psychoacoustics`] has armed
    /// the automatic per-granule Model 2 analysis — used by
    /// integration tests / observability.
    #[must_use]
    pub fn model2_psychoacoustics_enabled(&self) -> bool {
        self.model2_psy.is_some()
    }

    /// Arm §C.1.5.3 scalefactor-selection-information (scfsi) reuse.
    ///
    /// MPEG-1 Layer III carries **two** granules per frame, each with
    /// its own block of part2 scalefactors. The §2.4.2.7 `scfsi[ch]`
    /// field lets a frame transmit a long-block scfsi_band group's
    /// scalefactors **once** — in granule 0 — and declare them valid
    /// for granule 1, when the two granules happen to share identical
    /// scalefactors in that group. This is a lossless main-data
    /// saving: the decoder reuses granule 0's values verbatim for the
    /// marked groups, so every reconstructed sample is bit-identical;
    /// only granule 1's part2 length shrinks.
    ///
    /// With this armed, the assembler computes, for each channel,
    /// `scfsi[ch][g] = 1` for every one of the four §2.4.2.7
    /// scfsi_band groups
    /// (`{0..5}`, `{6..10}`, `{11..15}`, `{16..20}`) whose granule-1
    /// scalefactors equal granule 0's across every band in the group —
    /// **only** when both granules of that channel are long blocks
    /// (`block_type != 2`). Per §2.4.2.7 ("if short windows are
    /// switched on, i.e. `block_type == 2` for one of the granules,
    /// then scfsi is always 0 for this frame"), a channel with a short
    /// granule keeps `scfsi[ch] = 0` and transmits both granules in
    /// full.
    ///
    /// **Auto-armed as of r301:** a freshly constructed encoder already
    /// has scfsi reuse on (the detection is byte-exact and the saving is
    /// lossless, so there is no reason to leave it off). This method
    /// re-arms reuse after a prior [`Self::disable_scfsi_reuse`] call; on
    /// a default encoder it is a no-op. The flag is a no-op on LSF
    /// (MPEG-2 / MPEG-2.5) frames, which have a single granule and no
    /// scfsi field.
    pub fn enable_scfsi_reuse(&mut self) {
        self.scfsi_reuse = true;
    }

    /// Disarm §C.1.5.3 scalefactor-selection-information (scfsi) reuse.
    ///
    /// Forces every emitted MPEG-1 frame back to `scfsi = 0` throughout,
    /// reproducing the byte-for-byte pre-r301 output (granule 1 always
    /// retransmits its full part2 scalefactor block). This is purely a
    /// compatibility / regression-bisection escape hatch — the
    /// auto-armed default is lossless, so disabling it only enlarges the
    /// stream. No-op on LSF, which never carried scfsi.
    pub fn disable_scfsi_reuse(&mut self) {
        self.scfsi_reuse = false;
    }

    /// `true` when §C.1.5.3 scalefactor-selection-information reuse is
    /// armed (the auto-armed default; cleared by
    /// [`Self::disable_scfsi_reuse`]) — used by integration tests /
    /// observability.
    #[must_use]
    pub fn scfsi_reuse_enabled(&self) -> bool {
        self.scfsi_reuse
    }

    /// The §C.1.5.3.2.1 window-switching decision the automatic Model 2
    /// analysis reached for a granule/channel of the **most recently
    /// emitted frame**.
    ///
    /// When [`Self::enable_model2_psychoacoustics`] is armed, every
    /// granule's PCM passes through the channel's continuous-history
    /// Model 2 state, which yields both the per-band masking threshold
    /// (used to drive the outer loop) and the §C.1.5.3.2.1
    /// psychoacoustic entropy `pe` plus its `pe > 1800` short-block
    /// switching condition. This accessor surfaces the latter — the
    /// spec-canonical signal an encoder uses to decide window
    /// switching — for the granules of the frame the encoder last
    /// assembled.
    ///
    /// Returns `None` when:
    /// * no frame has been assembled yet with the mode armed;
    /// * `gr >= granules_per_frame()` (e.g. `gr == 1` for an LSF frame,
    ///   which carries a single granule), or `ch >= nch`;
    /// * the Model 2 mode is not armed (the analysis never ran).
    ///
    /// The block-type the encoder actually emits is still governed by
    /// the [`Self::enable_auto_block_type`] path or the force toggles;
    /// this accessor exposes the Model 2 decision for inspection and as
    /// the foundation for a future Model-2-driven auto-block-type mode,
    /// without changing the bytes any current configuration emits.
    #[must_use]
    pub fn last_model2_window_switch(&self, gr: usize, ch: usize) -> Option<Model2WindowSwitch> {
        if gr >= self.granules_per_frame() || ch >= self.nch {
            return None;
        }
        self.last_model2_switch.as_ref().and_then(|m| m[gr][ch])
    }

    /// Run one granule of PCM through the §C.1.5.3.2.1 **Model 2**
    /// psychoacoustic analysis and install the resulting per-band
    /// masking threshold as the outer loop's `xmin(sb)`.
    ///
    /// This is the end-to-end perceptual path: the granule's signal-
    /// dependent masking threshold (Figure C.6.c/d `thm(sb)`, via
    /// [`crate::psy::Model2Layer3State::process`] →
    /// [`crate::psy::XminThresholds::from_layer3_granule`]) replaces the
    /// signal-independent threshold-in-quiet bowl the outer loop would
    /// otherwise use. The outer loop then amplifies a band whose
    /// content is *masked* (high `thm`) less aggressively than a band
    /// whose content is *audible* (low `thm`), spending bits where the
    /// ear can hear the noise.
    ///
    /// The caller owns the [`crate::psy::Model2Layer3State`] and threads
    /// it across granules, because the Model 2 unpredictability measure
    /// is computed from the two preceding analysis blocks (§D.2.1: the
    /// FFT history "must remain constant over any particular
    /// application" and starts from a known zeroed state). Pass the same
    /// `state` for every granule of a channel, in time order; use one
    /// state per channel. The state's construction-time sample rate
    /// must match the encoder's.
    ///
    /// `granule` is exactly [`SAMPLES_PER_GRANULE`] (576) samples in
    /// `[-1.0, 1.0]`, the same domain the encoder's analysis filterbank
    /// consumes.
    ///
    /// # Errors
    ///
    /// * [`StreamEncodeError::PerBandXminWithoutOuterLoop`] — the
    ///   encoder has no outer loop (use
    ///   [`Mp3Encoder::new_with_outer_loop`]).
    /// * [`StreamEncodeError::Model2AnalysisUnsupported`] — `granule` is
    ///   not 576 samples (carried `sample_rate_hz == 0`), or the
    ///   encoder's rate is not a staged Model 2 rate (32 / 44.1 /
    ///   48 kHz; the LSF / MPEG-2.5 rates have no staged Annex D Model 2
    ///   tables).
    pub fn set_per_band_xmin_from_model2(
        &mut self,
        state: &mut crate::psy::Model2Layer3State,
        granule: &[f64],
    ) -> Result<(), StreamEncodeError> {
        if self.outer_loop_threshold.is_none() {
            return Err(StreamEncodeError::PerBandXminWithoutOuterLoop);
        }
        if granule.len() != SAMPLES_PER_GRANULE {
            return Err(StreamEncodeError::Model2AnalysisUnsupported { sample_rate_hz: 0 });
        }
        if crate::psy::AnnexDSamplingRate::from_hz(self.sample_rate_hz).is_none() {
            return Err(StreamEncodeError::Model2AnalysisUnsupported {
                sample_rate_hz: self.sample_rate_hz,
            });
        }
        // The analysis returns `None` only on a granule-length mismatch,
        // already screened above; treat any residual `None` as the same
        // length error rather than panicking.
        let out = state
            .process(granule)
            .ok_or(StreamEncodeError::Model2AnalysisUnsupported { sample_rate_hz: 0 })?;
        let xmin = crate::psy::XminThresholds::from_layer3_granule(&out);
        self.per_band_xmin = Some(xmin);
        Ok(())
    }

    /// Push PCM samples (`i16`). For mono encoders the input is a
    /// straight `[s0, s1, s2, …]` sample stream; for stereo /
    /// dual-channel encoders the input is **interleaved** LR pairs
    /// (`[L0, R0, L1, R1, …]`). The encoder splits the interleaved
    /// stream into its per-channel buffers and assembles whole MP3
    /// frames as soon as each per-channel buffer has accumulated one
    /// frame's worth of samples — `SAMPLES_PER_FRAME_MPEG1 = 1152`
    /// (two granules) for MPEG-1, 576 (one granule, ISO/IEC 13818-3
    /// §2.4.3.2) for the MPEG-2 / MPEG-2.5 LSF rates.
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

        // Auto block-type needs one granule of PCM lookahead beyond
        // the frame it's currently encoding so the §C.1.5.2
        // `Long → Start` decision can anticipate the next granule's
        // attack flag. Hold one extra granule in `pending_pcm` to
        // satisfy that. The Model-2-driven block-type path
        // (`enable_auto_block_type_model2`) has the same one-granule
        // §C.1.5.2 lookahead requirement, so it pads identically.
        let lookahead_pad = if self.auto_block_type.is_some() || self.model2_block_type.is_some() {
            SAMPLES_PER_GRANULE
        } else {
            0
        };

        // Assemble frames as long as EVERY channel's pending buffer
        // holds at least one full granule-frame worth of samples
        // PLUS the auto-block-type lookahead (zero when auto is off).
        let frame_samples = self.samples_per_frame();
        while self
            .pending_pcm
            .iter()
            .all(|buf| buf.len() >= frame_samples + lookahead_pad)
        {
            let mut per_ch_frame_pcm: Vec<Vec<f32>> = Vec::with_capacity(nch);
            let mut per_ch_lookahead: Vec<Vec<f32>> = Vec::with_capacity(nch);
            for buf in self.pending_pcm.iter_mut() {
                let mut take = vec![0.0f32; frame_samples];
                take.copy_from_slice(&buf[..frame_samples]);
                let mut peek = vec![0.0f32; lookahead_pad];
                if lookahead_pad > 0 {
                    peek.copy_from_slice(&buf[frame_samples..frame_samples + lookahead_pad]);
                }
                buf.drain(..frame_samples);
                per_ch_frame_pcm.push(take);
                per_ch_lookahead.push(peek);
            }
            self.assemble_frame_with_lookahead(&per_ch_frame_pcm, &per_ch_lookahead)?;
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
        // Tail-flush. With auto block-type on, `push_samples` holds
        // back one extra granule of PCM as the §C.1.5.2 lookahead,
        // so at finish time we may have between 1 and
        // `samples_per_frame() + 576` samples per channel still
        // buffered. We emit them as one or two frames as needed, with
        // a zero-padded "no attack ahead" lookahead for the very last
        // frame.
        let nch = self.nch;
        // Both the energy-detector and the Model-2-driven block-type
        // paths hold back a §C.1.5.2 lookahead granule, so the
        // tail-flush must reconstruct it for either.
        let auto_on = self.auto_block_type.is_some() || self.model2_block_type.is_some();
        let frame_samples = self.samples_per_frame();
        loop {
            let any_pending = self.pending_pcm.iter().any(|b| !b.is_empty());
            if !any_pending {
                break;
            }
            let mut per_ch_tail: Vec<Vec<f32>> = Vec::with_capacity(nch);
            let mut per_ch_lookahead: Vec<Vec<f32>> = Vec::with_capacity(nch);
            for buf in self.pending_pcm.iter_mut() {
                let mut tail = vec![0.0f32; frame_samples];
                let take = buf.len().min(frame_samples);
                tail[..take].copy_from_slice(&buf[..take]);
                buf.drain(..take);
                per_ch_tail.push(tail);
                // Build the lookahead PCM: if there's a next granule
                // already in the buffer (still partially full from
                // the held-back PCM), use it; otherwise zero-pad.
                let mut peek = vec![0.0f32; if auto_on { SAMPLES_PER_GRANULE } else { 0 }];
                if auto_on {
                    let pk = buf.len().min(SAMPLES_PER_GRANULE);
                    peek[..pk].copy_from_slice(&buf[..pk]);
                }
                per_ch_lookahead.push(peek);
            }
            self.assemble_frame_with_lookahead(&per_ch_tail, &per_ch_lookahead)?;
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
        let si_bytes = side_info_byte_len(self.version, self.nch);
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
        let denom = self.granules_per_frame().saturating_mul(self.nch).max(1);
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
    /// Back-compatible shim: drives [`Self::assemble_frame_with_lookahead`]
    /// with an empty lookahead PCM (auto block-type off path).
    #[allow(dead_code)]
    fn assemble_frame(&mut self, per_ch_frame_pcm: &[Vec<f32>]) -> Result<(), StreamEncodeError> {
        let empty_lookahead: Vec<Vec<f32>> = (0..self.nch).map(|_| Vec::new()).collect();
        self.assemble_frame_with_lookahead(per_ch_frame_pcm, &empty_lookahead)
    }

    #[allow(clippy::needless_range_loop)]
    fn assemble_frame_with_lookahead(
        &mut self,
        per_ch_frame_pcm: &[Vec<f32>],
        per_ch_lookahead_pcm: &[Vec<f32>],
    ) -> Result<(), StreamEncodeError> {
        debug_assert_eq!(per_ch_frame_pcm.len(), self.nch);
        for buf in per_ch_frame_pcm.iter() {
            debug_assert_eq!(buf.len(), self.samples_per_frame());
        }
        // Granules in this frame: 2 (MPEG-1) or 1 (LSF). The
        // fixed-size `[..; 2]` scratch arrays keep their MPEG-1 shape;
        // the LSF path simply never touches index 1.
        let ngr = self.granules_per_frame();

        // ---- Build the side-info skeleton (all-long, zero scalefactors) ----
        let mut side_info = SideInfo {
            main_data_begin: 0,
            private_bits: 0,
            scfsi: [[false; 4]; 2],
            granules: [[default_long_gc(); 2]; GRANULES],
            channels: self.nch as u8,
            granule_count: ngr as u8,
            lsf: self.version.is_lsf(),
        };
        let mut scalefactors = FrameScaleFactors {
            granules: [[ScaleFactors::default(); 2]; 2],
            granule_count: ngr as u8,
            channels: self.nch as u8,
            part2_bits: [[0; 2]; 2],
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
        let mut xr_pre_per_gc: Vec<Vec<[f32; NUM_LINES]>> = (0..ngr)
            .map(|_| (0..self.nch).map(|_| [0.0f32; NUM_LINES]).collect())
            .collect();

        // Per-(gr, ch) automatic Model 2 psychoacoustic threshold,
        // populated in Pass 1 below when
        // [`Self::enable_model2_psychoacoustics`] is armed. `None`
        // entries (every entry, when the mode is off) leave Pass 2 on
        // whatever static `self.per_band_xmin` / uniform-scalar path
        // it would otherwise take. The granule order matters: the
        // §D.2.1 FFT history must advance granule-by-granule in stream
        // order, so the analysis runs in the Pass 1 loop (gr outer,
        // ch inner) rather than lazily in Pass 2.
        let mut model2_xmin_per_gc: [[Option<crate::psy::XminThresholds>; 2]; GRANULES] =
            Default::default();

        // Parallel to `model2_xmin_per_gc`: the §C.1.5.3.2.1
        // window-switching deliverable (`pe` + `attack`) captured from
        // the same Pass-1 Model 2 walk. Discarded today by the encode
        // path (the xmin is what drives the outer loop), but retained
        // here so the last frame's spec-canonical switching decision is
        // observable via [`Self::last_model2_window_switch`]. Every cell
        // starts `None`; only granules/channels the Model 2 walk
        // actually processed populate a `Some`.
        let mut model2_switch_per_gc: [[Option<Model2WindowSwitch>; 2]; GRANULES] =
            Default::default();

        // When the §C.1.5.3.2.1 **Model-2-driven block-type** mode
        // (`enable_auto_block_type_model2`) is armed, the block-type
        // pre-pass below runs each frame granule's PCM through the
        // channel's continuous-history Model 2 state to obtain the
        // `pe > 1800` window-switching attack flag. That same walk
        // produces the full [`crate::psy::Model2Layer3Granule`] the
        // Pass-1 xmin derivation needs. To keep the §D.2.1 FFT history
        // advancing exactly once per granule, the pre-pass caches each
        // granule's output here and Pass 1 reuses it (the
        // `model2_psy.process` call in Pass 1 is skipped for any
        // (gr, ch) already populated). Empty (every cell `None`) unless
        // the Model-2-driven block-type mode is armed; the lookahead
        // granule is peeked non-destructively (cloned state) and is
        // never cached, so it never feeds an xmin.
        let mut model2_granule_per_gc: [[Option<crate::psy::Model2Layer3Granule>; 2]; GRANULES] =
            Default::default();

        // ---- Pre-pass: per-(gr, ch) block type ----
        //
        // The per-granule block type is the §C.1.5.2 transition state
        // for the granule, chosen by one of three policies:
        //
        // * **Force toggles** (`force_short_blocks` / `force_mixed_blocks`):
        //   every granule of every channel takes the forced type. As
        //   of r162 this branch accepts independent stereo too
        //   (each channel's side-info carries the same forced
        //   block_type, which §2.4.1.7 / §2.4.2.7 permit verbatim).
        //   MS-stereo is still rejected at API time.
        // * **Auto** (`auto_block_type.is_some()`): the
        //   [`crate::attack_detect::AttackDetector`] classifies each
        //   granule (and the lookahead granule) and the
        //   [`crate::block_type_sm::BlockTypeStateMachine`] schedules
        //   the §C.1.5.2 LONG → START → SHORT → STOP → LONG sequence
        //   per channel.
        // * **Default (all-long)**: every granule emits `Long`; this
        //   is the path every round prior to 156 took.
        //
        // The block-type matrix decides which MDCT path each
        // (gr, ch) tile takes below (long-family forward MDCT for
        // Long / Start / End; three short forward MDCTs for Short;
        // mixed for force-mixed). It also picks which
        // `default_*_gc()` skeleton becomes the side-info template
        // in pass 2.
        // Parallel matrix to `block_type_per_gc`: `mixed_per_gc[gr][ch]
        // == true` iff the corresponding emission is `BlockType::Short`
        // AND the auto path's optional mixed classifier decided this
        // granule warrants the §2.4.3.4.10.3 mixed-block carve-out.
        // The flag is always `false` on long-family emissions and on
        // any path that doesn't have the mixed classifier wired
        // (force-toggles, default-long, plain
        // `enable_auto_block_type`).
        let mut mixed_per_gc: [[bool; 2]; GRANULES] = [[false; 2]; GRANULES];
        // §2.4.3.4.9 cross-channel agreement for the auto/Model-2
        // scheduler walk. The two-channel joint regimes that fold L/R
        // together — MS-joint stereo (the §2.4.3.4.9.2 matrix rotates the
        // pair before quantize) and intensity coupling (the §2.4.3.4.9.3
        // fold reads each granule's `(L, R)` band-by-band) — both require
        // both channels of a granule to share `block_type` /
        // `window_switching_flag` / `mixed_block_flag`. When this holds
        // the scheduler runs ONE shared (channel-0) state machine fed the
        // OR-fold of the per-channel attack flags, and mirrors its
        // emission across both channels (the "safe upper envelope"). r307
        // armed this for MS-joint; r308 extends it to the intensity-only
        // path (`new_joint_stereo_is` + `enable_auto_block_type`), where
        // the same mirroring keeps L/R block types consistent so the
        // per-window short / long intensity coupling is well-defined.
        let channel_agreement_active =
            (self.ms_joint_stereo_active() || self.intensity_start_sfb.is_some()) && self.nch == 2;
        let block_type_per_gc: [[BlockType; 2]; GRANULES] = if let Some(schedulers) =
            self.model2_block_type.as_mut()
        {
            // ---- §C.1.5.3.2.1 Model-2-driven block-type path ----
            //
            // The per-granule attack flag is the spec-canonical Model 2
            // window-switching decision `pe > 1800` (the
            // §C.1.5.3.2.1 psychoacoustic-entropy threshold), not the
            // energy-detector ratio. The walk that yields it is the
            // *same* per-channel continuous-history Model 2 state that
            // `enable_model2_psychoacoustics` armed; its full
            // `Model2Layer3Granule` output is cached into
            // `model2_granule_per_gc` so Pass 1 reuses it for xmin
            // (the §D.2.1 FFT history thus advances exactly once per
            // granule — never twice).
            //
            // The frame granules advance the committed state; the
            // lookahead granule (the §C.1.5.2 `next_attack`
            // anticipation) is peeked by cloning the channel state so
            // the borrowed next-frame PCM never perturbs the FFT
            // history. The §C.1.5.2 transition geometry and the two
            // channel-coupling regimes (independent / MS-stereo OR-fold)
            // mirror the energy-detector path exactly.
            //
            // `schedulers` (the per-channel block-type state machines)
            // is bound by the `if let` above. The Model 2 states live in
            // a separate field, borrowed disjointly here. `model2_psy`
            // is always `Some` in this arm: the mode arms
            // `model2_block_type` only after
            // `enable_model2_psychoacoustics` set `model2_psy`, and
            // disarming `model2_psy` (via `set_per_band_xmin`) clears
            // `model2_block_type` too.
            // Capture the preset's Model 2 anchor offset before the
            // disjoint `&mut` borrow of `model2_psy` below (f64 is Copy).
            let model2_offset_db = self.model2_offset_db;
            let states = self
                .model2_psy
                .as_mut()
                .expect("model2_block_type implies model2_psy is armed");
            let mut out: [[BlockType; 2]; GRANULES] = [[BlockType::Long; 2]; GRANULES];

            // Helper: extract one granule's PCM as an f64 vector.
            // `g < ngr` indexes the frame PCM; `g == ngr` is the
            // lookahead granule. Returns the 576-sample vector plus
            // whether real (non-padded) lookahead PCM exists.
            let grab_granule_f64 =
                |frame: &[Vec<f32>], look: &[Vec<f32>], ch: usize, g: usize| -> (Vec<f64>, bool) {
                    if g < ngr {
                        let slice =
                            &frame[ch][g * SAMPLES_PER_GRANULE..(g + 1) * SAMPLES_PER_GRANULE];
                        (slice.iter().map(|&s| f64::from(s)).collect(), true)
                    } else {
                        let lk = &look[ch];
                        if lk.is_empty() {
                            (vec![0.0f64; SAMPLES_PER_GRANULE], false)
                        } else {
                            let mut slab = vec![0.0f64; SAMPLES_PER_GRANULE];
                            let n = lk.len().min(SAMPLES_PER_GRANULE);
                            for (dst, &src) in slab.iter_mut().zip(&lk[..n]) {
                                *dst = f64::from(src);
                            }
                            (slab, true)
                        }
                    }
                };

            // Compute, per channel, the `pe > 1800` attack flag for
            // each of the frame's `ngr` granules (advancing the
            // committed state and caching the granule output) plus the
            // lookahead granule (non-destructive clone peek).
            //
            // `attack[ch][g]` for `g` in `0..=ngr`: the last entry is
            // the lookahead. A `process` that returns `None`
            // (granule-length mismatch — unreachable here) leaves the
            // flag `false` and the cache `None`, so Pass 1 falls back
            // to the static threshold for that cell.
            let mut attack: [[bool; GRANULES + 1]; 2] = [[false; GRANULES + 1]; 2];
            for ch in 0..self.nch {
                for g in 0..ngr {
                    let (granule_f64, _present) =
                        grab_granule_f64(per_ch_frame_pcm, per_ch_lookahead_pcm, ch, g);
                    if let Some(gran) = states[ch].process(&granule_f64) {
                        attack[ch][g] = gran.attack;
                        model2_switch_per_gc[g][ch] = Some(Model2WindowSwitch {
                            pe: gran.pe,
                            attack: gran.attack,
                        });
                        model2_xmin_per_gc[g][ch] = Some(
                            crate::psy::XminThresholds::from_layer3_granule_with_offset_db(
                                &gran,
                                model2_offset_db,
                            ),
                        );
                        model2_granule_per_gc[g][ch] = Some(gran);
                    }
                }
                // Lookahead granule: peek without committing FFT
                // history. Empty lookahead (end-of-stream) → no attack
                // ahead, so the burst geometry closes with a Stop.
                let (granule_f64, present) =
                    grab_granule_f64(per_ch_frame_pcm, per_ch_lookahead_pcm, ch, ngr);
                if present {
                    let mut peek = states[ch].clone();
                    if let Some(gran) = peek.process(&granule_f64) {
                        attack[ch][ngr] = gran.attack;
                    }
                }
            }

            if channel_agreement_active {
                // OR-fold the per-channel `pe > 1800` flags into the
                // channel-0 scheduler (§2.4.3.4.9 cross-channel
                // agreement) and mirror its emission across both
                // channels of each granule. The channel-1 scheduler
                // stays at default and carries no state.
                for g in 0..ngr {
                    let folded_cur = attack[0][g] || attack[1][g];
                    let folded_next = attack[0][g + 1] || attack[1][g + 1];
                    let bt = schedulers[0].step(folded_cur, folded_next);
                    out[g][0] = bt;
                    out[g][1] = bt;
                }
            } else {
                // Independent: one scheduler per channel; granule `g`'s
                // companion is granule `g + 1`'s flag (the lookahead
                // for the last frame granule).
                for ch in 0..self.nch {
                    for g in 0..ngr {
                        out[g][ch] = schedulers[ch].step(attack[ch][g], attack[ch][g + 1]);
                    }
                }
            }
            out
        } else if let Some(ref mut cfg) = self.auto_block_type {
            // Auto path: classify each of the 2 frame granules + 1
            // lookahead granule per channel, then feed the scheduler.
            //
            // Two channel-coupling regimes (r163):
            //
            // * **Independent** (mono, `ChannelMode::Stereo`,
            //   `ChannelMode::DualChannel`): each channel runs an
            //   independent detector + scheduler. The per-channel
            //   §C.1.5.2 transition states never need to match —
            //   §2.4.1.7 / §2.4.2.7 carry side-info per channel
            //   verbatim. The matrix `block_type_per_gc[gr][ch]`
            //   can hold a different value for each channel.
            //
            // * **MS-stereo** (`new_joint_stereo_ms` or
            //   `new_joint_stereo_auto`, the
            //   `ms_agreement_active` branch): §2.4.3.4.9 requires
            //   both channels of a granule to share the same
            //   `block_type` / `window_switching_flag` /
            //   `mixed_block_flag`, because the §2.4.3.4.9.2 MS
            //   matrix `M = (L+R)/√2`, `S = (L-R)/√2` rotates L/R
            //   before quantize and the decoder needs both halves
            //   to share window geometry. The per-channel attack
            //   detectors are still run (each channel keeps a
            //   coherent ambient estimate so a quiet channel
            //   doesn't drag the loud channel's threshold around),
            //   but their per-granule attack flags are folded via
            //   logical OR into the channel-0 scheduler and the
            //   channel-0 scheduler's emission is mirrored across
            //   both channels of the granule. The channel-1
            //   scheduler is bypassed — it stays at default and
            //   carries no state. If either channel demands a
            //   Short, the granule emits Short on both; same for
            //   the mixed-block-flag promotion. This is the
            //   "safe upper envelope" agreement: it accepts more
            //   short bursts than a per-channel sequence would
            //   (each channel sees the other's transients) but
            //   never under-resolves a real transient on either
            //   side, and produces a self-consistent §C.1.5.2
            //   sequence across the shared scheduler. Symmetric in
            //   L↔R: an attack on either channel triggers the
            //   transition for both.
            let mut out: [[BlockType; 2]; GRANULES] = [[BlockType::Long; 2]; GRANULES];
            // Mixed-burst promotion is emitted on the wire only at the
            // MPEG-1 rates (r408). A mixed burst needs its flanking
            // Start / End granules to carry the §2.4.2.7
            // `mixed_block_flag` (normal window on the two lowest
            // subbands, so the §2.4.3.4 low-subband overlap-add
            // cancels across the whole burst). At the MPEG-1 rates
            // every deployed black-box validator decodes that wire
            // combination float-perfectly. At the LSF / MPEG-2.5
            // rates the combination is CONFORMANT — the ISO/IEC
            // 13818-3 main_data syntax scopes the mixed scalefactor
            // layout to `block_type == '10'`, and its
            // scalefac_compress partition tables mark
            // mixed_block_flag as don't-care ('x') for block types
            // '00'/'01'/'11' — but the r408 observer measurements
            // found the deployed world split 2-2: two independent
            // black-box decoders track the spec reading
            // float-perfectly while two others desynchronise on the
            // whole burst (nrmse ≈ 0.4–1.3, consistent with reading a
            // different scalefactor partition for the flagged
            // Start / End). With no de-facto consensus to conform to,
            // the auto scheduler demotes mixed bursts to pure-short
            // at the non-MPEG-1 rates (pure-short transitions decode
            // identically on every validator); `force_mixed_blocks`
            // — steady mixed streams without transition flanks —
            // remains available at every rate except 8 kHz.
            let mixed_promotion_wire_safe = self.version == MpegVersion::Mpeg1;
            // Generalised over `ngr ∈ {1, 2}`: the §C.1.5.2 walk
            // builds, per channel, an attack flag for each of this
            // frame's `ngr` granules plus one lookahead granule
            // (the next frame's leading granule), then steps the
            // scheduler `ngr` times — granule `g` is fed
            // `(attack[g], attack[g + 1])` so its companion is the
            // following granule's flag (the lookahead for the last
            // granule of the frame). On MPEG-1 (`ngr == 2`) this
            // reproduces the prior two-step walk verbatim; on LSF
            // (`ngr == 1`, ISO/IEC 13818-3 single 576-sample
            // granule) it steps once with the next frame's granule
            // as the §C.1.5.2 lookahead. The window-switching
            // geometry of §2.4.3.4.10.3 is identical across
            // versions — only the per-frame granule count differs.
            //
            // Helper: extract one granule's PCM (`g` in
            // `0..ngr` indexes into `per_ch_frame_pcm`; `g == ngr`
            // is the lookahead granule held in
            // `per_ch_lookahead_pcm`). Returns the 576-sample slab
            // plus whether real (non-padded) lookahead PCM exists.
            let grab_granule = |frame: &[Vec<f32>],
                                look: &[Vec<f32>],
                                ch: usize,
                                g: usize|
             -> ([f32; SAMPLES_PER_GRANULE], bool) {
                let mut slab = [0.0f32; SAMPLES_PER_GRANULE];
                if g < ngr {
                    slab.copy_from_slice(
                        &frame[ch][g * SAMPLES_PER_GRANULE..(g + 1) * SAMPLES_PER_GRANULE],
                    );
                    (slab, true)
                } else {
                    let lk = &look[ch];
                    if lk.is_empty() {
                        (slab, false)
                    } else {
                        let n = lk.len().min(SAMPLES_PER_GRANULE);
                        slab[..n].copy_from_slice(&lk[..n]);
                        (slab, true)
                    }
                }
            };
            if channel_agreement_active {
                // Per-channel attack-flag computation. Each channel's
                // detector still classifies its own PCM so the
                // ambient estimate stays meaningful for that
                // channel (a sudden L burst doesn't get hidden by
                // R's running ambient). The per-channel flags are
                // OR-folded into a single shared (channel-0)
                // scheduler so both channels of a granule share
                // window geometry, as §2.4.3.4.9 requires.
                //
                // `attack[g]` / `mixed[g]` for `g` in `0..=ngr`:
                // the last entry is the lookahead granule.
                let mut attack = [false; GRANULES + 1];
                let mut mixed = [false; GRANULES + 1];
                for ch in 0..self.nch {
                    for g in 0..ngr {
                        let (slab, _present) =
                            grab_granule(per_ch_frame_pcm, per_ch_lookahead_pcm, ch, g);
                        attack[g] |= cfg.detector[ch].classify(&slab);
                        if let Some(ref mut classifiers) = cfg.mixed_classifier {
                            mixed[g] |= classifiers[ch].classify_mixed(&slab);
                        }
                    }
                    // Lookahead granule (`g == ngr`): peek
                    // non-destructively (clone the detector /
                    // classifier) so the zero-padded or borrowed
                    // next-frame PCM never perturbs the ambient / LP
                    // state. Empty lookahead (end-of-stream) →
                    // `next_attack = false`, `next mixed = false`.
                    let (slab, present) =
                        grab_granule(per_ch_frame_pcm, per_ch_lookahead_pcm, ch, ngr);
                    if present {
                        let mut det_peek = cfg.detector[ch].clone();
                        attack[ngr] |= det_peek.classify(&slab);
                        if let Some(ref classifiers) = cfg.mixed_classifier {
                            let mut cls_peek = classifiers[ch].clone();
                            mixed[ngr] |= cls_peek.classify_mixed(&slab);
                        }
                    }
                }
                // Single shared scheduler (channel-0's slot;
                // channel-1's scheduler is left at default and
                // carries no state in the MS-stereo regime). Mirror
                // each emission across both channels of the granule.
                // `mixed[g + 1]` is the lookahead granule's mixed
                // preference — the scheduler latches it as the
                // burst's mixed-ness when it commits a Start (the
                // lookahead granule is the burst's first Short).
                for g in 0..ngr {
                    let (bt, mx) = cfg.scheduler[0].step_with_mixed(
                        attack[g],
                        attack[g + 1],
                        mixed[g + 1] && mixed_promotion_wire_safe,
                    );
                    out[g][0] = bt;
                    out[g][1] = bt;
                    mixed_per_gc[g][0] = mx;
                    mixed_per_gc[g][1] = mx;
                }
            } else {
                for ch in 0..self.nch {
                    // `ngr + 1` attack flags: one per frame granule
                    // plus the lookahead granule. Lookahead is empty
                    // at end-of-stream — treated as "no attack
                    // ahead" so the burst geometry closes with a
                    // Stop. We do NOT classify a zero-padded
                    // lookahead buffer with the live detector
                    // because that would inject a spurious
                    // silent-floor sample into the ambient estimate;
                    // instead we peek non-destructively via a
                    // detector clone and discard the result.
                    let mut attack = [false; GRANULES + 1];
                    // Optional mixed-vs-pure-short classification per
                    // granule. The classifier is stateful (its
                    // one-tap LP carries the previous granule's last
                    // sample across boundaries), so we always advance
                    // it on every frame granule to keep the seed
                    // continuous — independent of whether the
                    // emission ends up Short. The flag is consumed
                    // only when the scheduler returns Short; the
                    // §2.4.2.7 mixed_block_flag is valid only on
                    // Short emissions, which `step_with_mixed`
                    // enforces.
                    let mut mixed = [false; GRANULES + 1];
                    for g in 0..ngr {
                        let (slab, _present) =
                            grab_granule(per_ch_frame_pcm, per_ch_lookahead_pcm, ch, g);
                        attack[g] = cfg.detector[ch].classify(&slab);
                        if let Some(ref mut classifiers) = cfg.mixed_classifier {
                            mixed[g] = classifiers[ch].classify_mixed(&slab);
                        }
                    }
                    let (slab, present) =
                        grab_granule(per_ch_frame_pcm, per_ch_lookahead_pcm, ch, ngr);
                    attack[ngr] = if present {
                        let mut det_peek = cfg.detector[ch].clone();
                        if let Some(ref classifiers) = cfg.mixed_classifier {
                            let mut cls_peek = classifiers[ch].clone();
                            mixed[ngr] = cls_peek.classify_mixed(&slab);
                        }
                        det_peek.classify(&slab)
                    } else {
                        false
                    };
                    // Feed the scheduler in granule-major order;
                    // granule `g`'s companion is granule `g + 1`'s
                    // attack flag (the lookahead for the last one).
                    // `mixed[g + 1]` is the lookahead granule's mixed
                    // preference — latched as the burst's mixed-ness
                    // when the scheduler commits a Start (the
                    // lookahead granule is the burst's first Short).
                    for g in 0..ngr {
                        let (bt, mx) = cfg.scheduler[ch].step_with_mixed(
                            attack[g],
                            attack[g + 1],
                            mixed[g + 1] && mixed_promotion_wire_safe,
                        );
                        out[g][ch] = bt;
                        mixed_per_gc[g][ch] = mx;
                    }
                }
            }
            out
        } else if self.force_short_blocks {
            [[BlockType::Short; 2]; GRANULES]
        } else if self.force_mixed_blocks {
            // Mixed is still `block_type == Short` on the wire (the
            // mixed_block_flag selects long-region carve-out
            // separately).
            [[BlockType::Short; 2]; GRANULES]
        } else {
            [[BlockType::Long; 2]; GRANULES]
        };

        for gr in 0..ngr {
            for ch in 0..self.nch {
                let gr_pcm =
                    &per_ch_frame_pcm[ch][gr * SAMPLES_PER_GRANULE..(gr + 1) * SAMPLES_PER_GRANULE];
                let mut pcm_arr = [0.0f32; SAMPLES_PER_GRANULE];
                pcm_arr.copy_from_slice(gr_pcm);

                // Automatic per-granule §C.1.5.3.2.1 Model 2
                // psychoacoustics: run this granule's PCM through the
                // channel's continuous-history Model 2 state and stash
                // the resulting signal-dependent `xmin(sb)` for Pass 2
                // to install before the granule's outer-loop search.
                //
                // The state MUST advance once per granule in stream
                // order (the §D.2.1 FFT history feeds the next
                // granule's prediction), so the analysis runs here in
                // the (gr outer, ch inner) Pass 1 walk — never lazily.
                // `process` returns `None` only on a granule-length
                // mismatch, which cannot occur (`pcm_arr` is exactly
                // `SAMPLES_PER_GRANULE`); a residual `None` simply
                // leaves the entry unset and Pass 2 falls back to the
                // static threshold rather than panicking.
                //
                // When the §C.1.5.3.2.1 Model-2-driven block-type mode
                // is armed the block-type pre-pass already advanced this
                // channel's Model 2 state for this granule and cached
                // the output in `model2_granule_per_gc[gr][ch]` (the xmin
                // / switch matrices were filled there too). Re-running
                // `process` here would advance the §D.2.1 FFT history a
                // second time, so we reuse the cached granule instead and
                // skip the call entirely for any (gr, ch) the pre-pass
                // populated.
                if model2_granule_per_gc[gr][ch].is_none() {
                    let model2_offset_db = self.model2_offset_db;
                    if let Some(states) = self.model2_psy.as_mut() {
                        let granule_f64: Vec<f64> = pcm_arr.iter().map(|&s| f64::from(s)).collect();
                        if let Some(out) = states[ch].process(&granule_f64) {
                            // Retain the §C.1.5.3.2.1 window-switching
                            // deliverable alongside the xmin threshold — the
                            // analysis computed both in one walk.
                            model2_switch_per_gc[gr][ch] = Some(Model2WindowSwitch {
                                pe: out.pe,
                                attack: out.attack,
                            });
                            model2_xmin_per_gc[gr][ch] = Some(
                                crate::psy::XminThresholds::from_layer3_granule_with_offset_db(
                                    &out,
                                    model2_offset_db,
                                ),
                            );
                        }
                    }
                }

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
                // * **Short block** (`force_short_blocks` on; r162
                //   accepts mono + independent stereo per
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
                // * **Mixed block** (`force_mixed_blocks` on; r162
                //   accepts mono + independent stereo per
                //   [`Mp3Encoder::force_mixed_blocks_for_testing`]):
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
                // Both the energy-detector and the §C.1.5.3.2.1
                // Model-2-driven auto paths populate `block_type_per_gc`
                // with the full §C.1.5.2 sequence (Long / Start / Short /
                // End), so the per-granule MDCT dispatch below must take
                // the auto branch for either.
                let auto_bt_active =
                    self.auto_block_type.is_some() || self.model2_block_type.is_some();
                let auto_bt = block_type_per_gc[gr][ch];
                if auto_bt_active {
                    // Auto block-type dispatch. The §C.1.5.2 sequence
                    // can put any of Long / Start / Short / End on this
                    // granule. Long-family (Long / Start / End) all use
                    // the 36-point forward MDCT with a block-type-specific
                    // window from `window_long_family_analysis`; Short
                    // uses the three 12-point forward MDCTs per subband
                    // from `forward_short_mdct_subband`. Alias reduction
                    // is applied on Long / Start / End (decoder's
                    // §2.4.3.4.10.1 gate is `block_type == Short`).
                    match auto_bt {
                        BlockType::Short if mixed_per_gc[gr][ch] => {
                            // Auto-mixed forward path (r161): the
                            // scheduler emitted Short AND the mixed
                            // classifier judged the granule's low
                            // band stable enough to warrant the
                            // §2.4.3.4.10.3 carve-out. Subbands 0,1
                            // take the 36-point long sine window
                            // (lines 0..36); subbands 2..31 take the
                            // three 12-point short windows. The
                            // dispatch mirrors the `force_mixed_blocks`
                            // branch below — same per-subband MDCT
                            // state slots, same scale derivation,
                            // same `forward_reorder` driver with a
                            // mixed gc. The §C.1.5.4.3 mixed-block
                            // outer-loop primitive
                            // (`outer_loop_search_mixed`) already
                            // exists from r159 and is selected by
                            // the gc_template's mixed_block_flag
                            // below.
                            for sb in 0..2 {
                                let mut current = [0.0f64; LONG_N / 2];
                                for (t, slot) in current.iter_mut().enumerate() {
                                    *slot = f64::from(inv[sb][t]);
                                }
                                let frame36 =
                                    forward_overlap(&current, &mut self.mdct_state[ch][sb]);
                                let windowed =
                                    window_long_family_analysis(&frame36, BlockType::Long);
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
                            let gc_mixed = default_mixed_gc();
                            // Inverse mixed alias reduction (r405):
                            // the decoder applies the single sb == 1
                            // butterfly to a mixed granule's long
                            // region, so the encoder inverts it here
                            // (lines 10..26 — inside the reorder's
                            // long-region passthrough).
                            let xr = inverse_alias_reduce_mixed(&xr);
                            xr_pre_per_gc[gr][ch] = crate::short_block::forward_reorder(
                                &xr,
                                &gc_mixed,
                                self.sample_rate_hz,
                                self.version,
                            );
                        }
                        BlockType::Short => {
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
                            let mut gc_short = default_long_gc();
                            let (r0, r1) = crate::short_block::short_block_region_defaults();
                            gc_short.window_switching_flag = true;
                            gc_short.block_type = BlockType::Short;
                            gc_short.mixed_block_flag = false;
                            gc_short.region0_count = r0;
                            gc_short.region1_count = r1;
                            xr_pre_per_gc[gr][ch] = crate::short_block::forward_reorder(
                                &xr,
                                &gc_short,
                                self.sample_rate_hz,
                                self.version,
                            );
                        }
                        BlockType::Long | BlockType::Start | BlockType::End => {
                            // A Start / End granule that flanks a
                            // **mixed** burst carries
                            // `mixed_block_flag = 1` (§2.4.2.7: for any
                            // window-switched granule the flag means
                            // the two lowest polyphase subbands are
                            // transformed with the normal window while
                            // the remaining 30 follow the block type).
                            // Keeping subbands 0..2 on the normal
                            // analysis window across the whole burst is
                            // what makes the low-subband §2.4.3.4
                            // overlap-add cancel — a start/end window
                            // against the mixed granule's normal window
                            // leaves uncancelled aliasing there
                            // (measured as a low-band divergence on an
                            // independent black-box validator, r408).
                            let mixed_transition = mixed_per_gc[gr][ch]
                                && matches!(auto_bt, BlockType::Start | BlockType::End);
                            for sb in 0..32 {
                                let mut current = [0.0f64; LONG_N / 2];
                                for (t, slot) in current.iter_mut().enumerate() {
                                    *slot = f64::from(inv[sb][t]);
                                }
                                let frame36 =
                                    forward_overlap(&current, &mut self.mdct_state[ch][sb]);
                                let window_bt = if mixed_transition && sb < 2 {
                                    BlockType::Long
                                } else {
                                    auto_bt
                                };
                                let windowed = window_long_family_analysis(&frame36, window_bt);
                                let bins = mdct(&windowed, LONG_N);
                                for (k, &b) in bins.iter().enumerate() {
                                    xr[sb * 18 + k] = (b / 9.0) as f32;
                                }
                            }
                            // Alias reduction is applied by the decoder
                            // for block_type != Short (all 32 subbands
                            // are 36-point long-family transforms, mixed
                            // flag or not), so we invert it here just as
                            // the all-long default path does.
                            xr_pre_per_gc[gr][ch] = inverse_alias_reduce(&xr);
                        }
                    }
                } else if self.force_short_blocks {
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
                    // Inverse mixed alias reduction (r405): the
                    // decoder applies the single sb == 1 butterfly to
                    // a mixed granule's long region
                    // (`mp3-alias-reduction-clarification.md`), so the
                    // encoder inverts it here. The butterfly touches
                    // lines 10..26 only — inside the long region the
                    // reorder passes through unchanged, so applying it
                    // before `forward_reorder` is exact.
                    let xr = inverse_alias_reduce_mixed(&xr);
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

        // Commit this frame's §C.1.5.3.2.1 window-switching decisions so
        // `last_model2_window_switch` reflects exactly the frame just
        // assembled. Only meaningful when the automatic Model 2 mode is
        // armed; otherwise the matrix is all-`None` and we leave the
        // accessor at `None` (no decision was made).
        if self.model2_psy.is_some() {
            self.last_model2_switch = Some(model2_switch_per_gc);
        }

        // ---- Pass 1.45: optional §2.4.3.4.9.3 intensity coupling ----
        //
        // When intensity-stereo is armed (`intensity_start_sfb =
        // Some(b)`), every granule's long scalefactor bands `b..21`
        // (plus the partial top region above the last Table B.8 band
        // boundary, which has no scalefactor of its own) are coupled:
        //
        //   * per band, derive the stereo position from the band
        //     energies (Annex G.2 c):
        //       is_pos[sfb] = NINT((12/π)·arctan(√(E_L/E_R)))
        //   * left channel := L + R (the combined magnitude),
        //   * right channel := 0 (the §2.4.3.4.9.1 zero-part the
        //     decoder derives the intensity bound from).
        //
        // The positions land on the wire as the right channel's
        // scalefactors in pass 2. The decoder reconstructs
        // L' = T·is_ratio/(1+is_ratio), R' = T/(1+is_ratio) with
        // is_ratio = tan(is_pos·π/12) (§2.4.3.4.9.3 steps 3-5), so the
        // reconstructed amplitude ratio L'/R' = is_ratio approximates
        // the original √(E_L/E_R) to the nearest of the 7 grid angles.
        //
        // `is_pos_per_gr[gr][sfb]` defaults to the illegal marker 7 so
        // non-intensity paths and below-bound bands never leak a stale
        // position.
        let mut is_pos_per_gr = [[7u8; 21]; GRANULES];
        // Short-block per-window intensity positions, parallel to
        // `is_pos_per_gr` but indexed `[gr][sfb][win]` over the 12 short
        // scalefactor bands × 3 windows (ISO/IEC 11172-3 §2.4.3.4.9.3 +
        // 13818-3 §2.4.3.2: the intensity bound is derived per window).
        // Defaults to the illegal marker `7`; only force-short
        // intensity granules populate it. Carried into Pass 2 where it
        // lands on the right channel's `scalefac_s[sfb][win]` slots.
        let mut is_pos_short_per_gr = [[[7u8; 3]; 12]; GRANULES];
        // Per-granule short-block intensity start band (the uniform
        // per-window bound on the force-short coupling path): bands
        // `0..short_start` are below the bound (MS or pass-through),
        // `short_start..12` are intensity-coded. Captured in Pass 1.45 so
        // the §2.4.3.4.9.2 MS rotation (Pass 1.5) can apply the matrix per
        // window below each window's bound in the interleaved short layout
        // (ISO/IEC 13818-3 §2.4.3.2). Defaults to 12 (couples nothing).
        let mut short_intensity_start_per_gr = [12usize; GRANULES];
        // Per-granule flag: did intensity coupling actually fire for
        // this granule? Always `true` for the fixed-bound modes (every
        // granule couples `start_sfb..21`); per-granule for the adaptive
        // chooser, where a granule whose effective bound reaches 21
        // couples nothing and must therefore write its right channel as
        // an ordinary (non-intensity) channel. Indexed by granule;
        // entries beyond `ngr` stay `false` and are never read.
        let mut intensity_coupled_per_gr = [false; GRANULES];
        // Upper line bound of the MS / independent-LR region: the whole
        // spectrum without intensity, the intensity start band's first
        // line with it (§2.4.3.4.9.1: when both methods are enabled the
        // MS equations apply only to the bands below the intensity
        // bound).
        let intensity_active = self.intensity_start_sfb.is_some() && self.nch == 2;
        let ms_region_hi = match self.intensity_start_sfb {
            Some(start_sfb) if self.nch == 2 => {
                long_band_starts_for(self.sample_rate_hz)[start_sfb]
            }
            _ => NUM_LINES,
        };
        // Per-granule intensity block-type dispatch. The intensity
        // coupling fold reads each granule's `(L, R)` band-by-band and
        // requires both channels to share the same window geometry
        // (§2.4.3.4.9 channel agreement); the per-window short layout
        // (§2.4.3.4.8 reorder, then 13818-3 §2.4.3.2) and the long-block
        // band walk are entirely different line partitions, so the choice
        // is made per granule:
        //
        //   * `force_short_blocks` — every granule is pure-short; the
        //     whole frame takes the short coupling (r303 / r305).
        //   * auto block-type under channel agreement — the scheduler may
        //     emit a *mix* of Long / Start / End / Short granules within
        //     one frame, but `channel_agreement_active` guarantees both
        //     channels of each granule share the same `block_type` /
        //     `window_switching_flag` / `mixed_block_flag` (the channel-0
        //     emission is mirrored across the granule). That agreement is
        //     forced whenever intensity coupling is armed — with MS-joint
        //     stereo (`new_joint_stereo_ms_is` / `new_joint_stereo_auto_is`,
        //     r307) OR on the intensity-only path (`new_joint_stereo_is`,
        //     r308). A granule the scheduler emitted as **pure short**
        //     (`block_type == Short`, `mixed_block_flag == false`) takes
        //     the §2.4.3.4.9.3 per-window short coupling; Long / Start /
        //     End granules take the long-block band walk.
        //
        // `short_intensity_gr[gr]` is the per-granule selector; the
        // long-block branch below runs for every granule it is `false`
        // for. Mixed blocks (`mixed_block_flag == true`) keep the
        // `IntensityShortBlocksUnsupported` rejection (the §2.4.3.4.10.3
        // two-region carve-out bound is not wired); the
        // `channel_agreement_active` OR-fold removes the L/R divergence
        // that previously blocked the intensity-only auto path.
        //
        // r311 wires the §2.4.3.4.10.3 mixed carve-out for the
        // intensity-only (non-MS) force-mixed path: a mixed granule
        // couples its long lowest 2 subbands (long bands `start..=7`,
        // lines 0..36) on the long-band walk AND its upper short region
        // (short bands 3..12) per window — exactly the two regions the
        // decoder's `process_short` reconstructs for a
        // `mixed_block_flag == true` granule. `mixed_intensity_gr[gr]`
        // selects this path; it is mutually exclusive with the pure-short
        // `short_intensity_gr[gr]` below.
        let mut mixed_intensity_gr = [false; GRANULES];
        let mut short_intensity_gr = [false; GRANULES];
        for gr in 0..ngr {
            mixed_intensity_gr[gr] = self.force_mixed_blocks
                || (block_type_per_gc[gr][0] == BlockType::Short && mixed_per_gc[gr][0]);
            short_intensity_gr[gr] = !mixed_intensity_gr[gr]
                && (self.force_short_blocks
                    || (block_type_per_gc[gr][0] == BlockType::Short && !mixed_per_gc[gr][0]));
        }
        if let Some(start_sfb) = self.intensity_start_sfb {
            if self.nch == 2 {
                let short_starts = short_band_starts_for(self.sample_rate_hz);
                let long_starts = long_band_starts_for(self.sample_rate_hz);
                let starts = long_starts;
                for gr in 0..ngr {
                    if mixed_intensity_gr[gr] {
                        // §2.4.3.4.10.3 mixed-block intensity coupling.
                        // The granule's lowest 2 polyphase subbands
                        // (long bands 0..=7, lines 0..36) are long-windowed
                        // and the upper 30 subbands (short bands 3..12) are
                        // short-windowed. Couple the long region on the
                        // long-band walk and the short region per window —
                        // the exact two regions the decoder's
                        // `process_short` rebuilds for a `mixed_block_flag`
                        // granule (`for sfb in 0..8` long, then per-window
                        // short bands `MIXED_FIRST_SHORT_SFB..12`).
                        let (left_slice, right_slice) = xr_pre_per_gc[gr].split_at_mut(1);
                        let left = &mut left_slice[0];
                        let right = &mut right_slice[0];
                        let is_pos_bands = &mut is_pos_per_gr[gr];

                        // --- Long region: long bands `start_sfb..8` ---
                        // The user `intensity_start_sfb` (1..=20) addresses
                        // the long-band grid directly. A granule couples at
                        // all whenever it has a long-region band at/above
                        // the bound (`start_sfb < 8`) or any short region
                        // above its mapped bound (handled below).
                        let long_eff_end = MIXED_LAST_LONG_SFB + 1; // = 8
                        for sfb in start_sfb..long_eff_end {
                            let lo = long_starts[sfb];
                            let hi = long_starts[sfb + 1];
                            let mut l_energy = 0.0f64;
                            let mut r_energy = 0.0f64;
                            for i in lo..hi {
                                l_energy += f64::from(left[i]) * f64::from(left[i]);
                                r_energy += f64::from(right[i]) * f64::from(right[i]);
                            }
                            is_pos_bands[sfb] = if self.version.is_lsf() {
                                derive_intensity_position_lsf(l_energy, r_energy)
                            } else {
                                derive_intensity_position(l_energy, r_energy)
                            };
                            for i in lo..hi {
                                left[i] += right[i];
                                right[i] = 0.0;
                            }
                        }

                        // --- Short region: per-window short bands ---
                        // Map the long start line onto a short band, then
                        // clamp to `MIXED_FIRST_SHORT_SFB` (= 3): the lowest
                        // three short bands fall inside the long-windowed
                        // carve-out and carry no short-region intensity.
                        let start_line = long_starts[start_sfb];
                        let mapped = (0..12)
                            .find(|&sfb| short_starts[sfb] >= start_line)
                            .unwrap_or(12);
                        let short_start = mapped.max(MIXED_FIRST_SHORT_SFB);
                        short_intensity_start_per_gr[gr] = short_start;
                        // A mixed granule couples whenever either region has
                        // a band at/above the bound.
                        intensity_coupled_per_gr[gr] = start_sfb < long_eff_end || short_start < 12;
                        for sfb in short_start..12 {
                            let s = short_starts[sfb];
                            let w = short_starts[sfb + 1] - short_starts[sfb];
                            let base = 3 * s;
                            for win in 0..3 {
                                let mut l_energy = 0.0f64;
                                let mut r_energy = 0.0f64;
                                let win_base = base + win * w;
                                for k in 0..w {
                                    let i = win_base + k;
                                    if i < NUM_LINES {
                                        l_energy += f64::from(left[i]) * f64::from(left[i]);
                                        r_energy += f64::from(right[i]) * f64::from(right[i]);
                                    }
                                }
                                is_pos_short_per_gr[gr][sfb][win] = if self.version.is_lsf() {
                                    derive_intensity_position_lsf(l_energy, r_energy)
                                } else {
                                    derive_intensity_position(l_energy, r_energy)
                                };
                                for k in 0..w {
                                    let i = win_base + k;
                                    if i < NUM_LINES {
                                        left[i] += right[i];
                                        right[i] = 0.0;
                                    }
                                }
                            }
                        }
                        continue;
                    }
                    if short_intensity_gr[gr] {
                        let (left_slice, right_slice) = xr_pre_per_gc[gr].split_at_mut(1);
                        let left = &mut left_slice[0];
                        let right = &mut right_slice[0];
                        // Map the long-block `intensity_start_sfb` (1..=20,
                        // the public API's only knob) onto a short-block
                        // start band by frequency: the short band whose first
                        // line is at or beyond the long start line. This
                        // keeps the one user-facing bound consistent across
                        // block types without a second API surface.
                        let start_line = long_starts[start_sfb];
                        let short_start = (0..12)
                            .find(|&sfb| short_starts[sfb] >= start_line)
                            .unwrap_or(12);
                        // A short granule always couples (it shares the short
                        // geometry); the per-window zero-part below the bound
                        // is what makes the decoder's per-window bound
                        // derivation land where intended.
                        intensity_coupled_per_gr[gr] = short_start < 12;
                        short_intensity_start_per_gr[gr] = short_start;
                        for sfb in short_start..12 {
                            let s = short_starts[sfb];
                            let w = short_starts[sfb + 1] - short_starts[sfb];
                            let base = 3 * s;
                            for win in 0..3 {
                                let mut l_energy = 0.0f64;
                                let mut r_energy = 0.0f64;
                                // Native bitstream layout (post-reorder):
                                // band `sfb` window `win` occupies the
                                // contiguous run `base + win·w .. base +
                                // (win+1)·w`. Coupling is a per-line
                                // operation, so deriving the position and
                                // folding the magnitude over this run yields
                                // exactly the spectrum the decoder rebuilds
                                // from `scalefac_s[sfb][win]` after its
                                // §2.4.3.4.8 reorder.
                                let win_base = base + win * w;
                                for k in 0..w {
                                    let i = win_base + k;
                                    if i < NUM_LINES {
                                        l_energy += f64::from(left[i]) * f64::from(left[i]);
                                        r_energy += f64::from(right[i]) * f64::from(right[i]);
                                    }
                                }
                                is_pos_short_per_gr[gr][sfb][win] = if self.version.is_lsf() {
                                    derive_intensity_position_lsf(l_energy, r_energy)
                                } else {
                                    derive_intensity_position(l_energy, r_energy)
                                };
                                for k in 0..w {
                                    let i = win_base + k;
                                    if i < NUM_LINES {
                                        left[i] += right[i];
                                        right[i] = 0.0;
                                    }
                                }
                            }
                        }
                        continue;
                    }
                    let is_pos_bands = &mut is_pos_per_gr[gr];
                    let (left_slice, right_slice) = xr_pre_per_gc[gr].split_at_mut(1);
                    let left = &mut left_slice[0];
                    let right = &mut right_slice[0];
                    // Per-granule effective start band. Fixed-bound modes
                    // couple `start_sfb..21` verbatim; the adaptive mode
                    // (`intensity_auto_threshold`) raises the bound to the
                    // lowest band whose contiguous high tail all carries
                    // little right-channel stereo information. `eff_start
                    // == 21` ⇒ this granule couples nothing (it keeps a
                    // full right channel and decodes as ordinary stereo).
                    let eff_start = match self.intensity_auto_threshold {
                        Some(t) => choose_intensity_bound(left, right, starts, start_sfb, t),
                        None => start_sfb,
                    };
                    // A granule whose adaptive bound reaches 21 couples
                    // nothing: its right channel keeps full spectral data
                    // and pass 2 must write it as an ordinary channel.
                    intensity_coupled_per_gr[gr] = eff_start < 21;
                    for (sfb, slot) in is_pos_bands.iter_mut().enumerate().skip(eff_start) {
                        let lo = starts[sfb];
                        let hi = starts[sfb + 1];
                        let mut l_energy = 0.0f64;
                        let mut r_energy = 0.0f64;
                        for i in lo..hi {
                            l_energy += f64::from(left[i]) * f64::from(left[i]);
                            r_energy += f64::from(right[i]) * f64::from(right[i]);
                        }
                        *slot = if self.version.is_lsf() {
                            derive_intensity_position_lsf(l_energy, r_energy)
                        } else {
                            derive_intensity_position(l_energy, r_energy)
                        };
                        for i in lo..hi {
                            left[i] += right[i];
                            right[i] = 0.0;
                        }
                    }
                    // Partial top region above the last band boundary:
                    // no scalefactor slot exists for these lines
                    // (Table B.8 `scalefac_l` stops at `starts[21]`),
                    // so there is no position to transmit — couple the
                    // magnitude into the left channel anyway so the
                    // right channel's zero-part reaches the Nyquist
                    // rate (§2.4.3.4.9.1); the lines decode left-only.
                    //
                    // Skip the top region when the adaptive chooser
                    // coupled nothing (`eff_start == 21`): zeroing only
                    // the top lines while the rest of the right channel
                    // stays full would create a spurious left-only tail.
                    // A coupled-nothing granule must keep a complete
                    // right channel so the decoder's derived bound lands
                    // at band 21 and reconstructs plain stereo.
                    if eff_start < 21 {
                        for i in starts[21]..NUM_LINES {
                            left[i] += right[i];
                            right[i] = 0.0;
                        }
                    }
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
                // With intensity armed the sums run over the
                // below-bound lines only — the region the MS rotation
                // would actually apply to (above the bound the right
                // channel is already the coupled zero-part).
                //
                // The picker must measure the *same* line set the
                // rotation (Pass 1.5 below) will touch, or it scores the
                // decision on the wrong spectrum. Two intensity regimes:
                //
                //   * **Long / no-intensity:** the MS region is the
                //     single contiguous run `0..ms_region_hi`
                //     (`ms_region_hi == NUM_LINES` without intensity, the
                //     long-block bound line with it). Frame-constant.
                //   * **Short + intensity** (`short_intensity_gr[gr] &&
                //     intensity_active`): the §2.4.3.4.9.2 MS rotation
                //     runs per window below each window's short bound, and
                //     in the §2.4.3.4.8 interleaved layout that is exactly
                //     the contiguous run `0..3*short_starts[short_start]`
                //     for the granule's per-window bound `short_start`
                //     (`short_intensity_start_per_gr[gr]`, set in Pass
                //     1.45). The bound is *per granule*, so the upper line
                //     is recomputed each granule rather than reused from
                //     the long-derived `ms_region_hi`.
                // The short-block bound is per granule (auto-scheduled
                // frames mix Long / Short granules), so the table lookup
                // is done per granule below rather than once for the
                // frame.
                let short_starts_for_picker = if intensity_active {
                    Some(short_band_starts_for(self.sample_rate_hz))
                } else {
                    None
                };
                let mut all_ok = true;
                for gr in 0..ngr {
                    let left = &xr_pre_per_gc[gr][0];
                    let right = &xr_pre_per_gc[gr][1];
                    // A short-coupled granule's MS region is the contiguous
                    // run `0..3*short_starts[short_start]` for that
                    // granule's per-window bound; a long-coupled granule's
                    // region is the frame-constant `ms_region_hi`.
                    let gr_region_hi = match short_starts_for_picker {
                        Some(short_starts) if short_intensity_gr[gr] => {
                            (3 * short_starts[short_intensity_start_per_gr[gr]]).min(NUM_LINES)
                        }
                        _ => ms_region_hi,
                    };
                    let mut lr_energy = 0.0f64;
                    let mut side_energy_x2 = 0.0f64;
                    for i in 0..gr_region_hi {
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
            // The MS region is decided **per granule**: an auto-scheduled
            // frame can mix Long / Short granules, and each granule's
            // below-intensity-bound region has a different geometry.
            //
            //   * **Short + intensity** (`short_intensity_gr[gr]`, r305 /
            //     r307): the intensity bound is per window, but in the
            //     §2.4.3.4.8 interleaved short layout (post-reorder) band
            //     `sfb` window `win` occupies the run `3*s + win*w ..
            //     3*s + (win+1)*w`. Pass 1.45 coupled intensity over
            //     `short_start..12` for every window (zeroing the right
            //     channel there), so the below-bound MS region is bands
            //     `0..short_start` across all three windows — and taking
            //     all three windows of every band `0..short_start` is
            //     exactly the contiguous run `0 ..
            //     3*short_starts[short_start]` (the reorder is a
            //     permutation of that set). Rotating those lines (and only
            //     those) is the exact inverse of the decoder's per-window
            //     `process_short`: it MS-decodes bands below each window's
            //     derived bound and intensity-decodes the rest (ISO/IEC
            //     13818-3 §2.4.3.2). The two regions are disjoint line
            //     sets, so applying intensity first (Pass 1.45) then MS
            //     here never double-touches a line.
            //   * **Long / no-intensity:** the region is the single
            //     contiguous run `0..ms_region_hi` (`ms_region_hi ==
            //     NUM_LINES` without intensity; the long-block bound line
            //     with it — §2.4.3.4.9.1: with both methods enabled the MS
            //     equations apply to the bands below the bound).
            //
            // MS is a per-line rotation, so rotating the contiguous run is
            // identical to walking each (sfb, win, k) cell.
            let short_starts = short_band_starts_for(self.sample_rate_hz);
            for gr in 0..ngr {
                let hi = if intensity_active && short_intensity_gr[gr] {
                    (3 * short_starts[short_intensity_start_per_gr[gr]]).min(NUM_LINES)
                } else {
                    ms_region_hi
                };
                // Split the per-channel borrow without copying both
                // arrays: `split_at_mut(1)` gives us `[L]` and `[R]` as
                // disjoint slices, then we index into them.
                let (left_slice, right_slice) = xr_pre_per_gc[gr].split_at_mut(1);
                let left = &mut left_slice[0];
                let right = &mut right_slice[0];
                for i in 0..hi {
                    let l = left[i];
                    let r = right[i];
                    left[i] = (l + r) * INV_SQRT2;
                    right[i] = (l - r) * INV_SQRT2;
                }
            }
        }
        // Reflect the per-frame decision on the carried frame header
        // (only matters for the auto picker; the unconditional
        // `new_joint_stereo_ms` / `new_joint_stereo_ms_is` paths keep
        // the constructor's '10' / '11' header template and the auto
        // path overwrites it here). With intensity armed the low
        // mode_extension bit stays set on both picker outcomes
        // (§2.4.2.3: '11' = both methods, '01' = intensity only).
        if self.ms_auto_threshold.is_some() && self.nch == 2 {
            frame_mode_extension = if apply_ms_this_frame {
                ModeExtension {
                    intensity_stereo: intensity_active,
                    ms_stereo: true,
                    raw: if intensity_active { 0b11 } else { 0b10 },
                }
            } else {
                ModeExtension {
                    intensity_stereo: intensity_active,
                    ms_stereo: false,
                    raw: if intensity_active { 0b01 } else { 0b00 },
                }
            };
        }

        // ---- Pass 2: per-(gr, ch) quantization + side-info build ----
        for gr in 0..ngr {
            for ch in 0..self.nch {
                let xr_pre = xr_pre_per_gc[gr][ch];

                // Automatic Model 2 path: install this granule's
                // signal-dependent `xmin(sb)` (computed in Pass 1)
                // before the outer-loop dispatch reads
                // `self.per_band_xmin`. The static-threshold and
                // automatic-Model-2 paths are mutually exclusive, so
                // when the mode is armed `self.per_band_xmin` starts
                // `None` and is rewritten per granule here; the
                // unchanged outer-loop dispatch below picks it up via
                // the existing `self.per_band_xmin` read. When the
                // mode is off, `model2_xmin_per_gc[gr][ch]` is `None`
                // and `self.per_band_xmin` is left exactly as the
                // static setters configured it.
                if let Some(xmin) = model2_xmin_per_gc[gr][ch].take() {
                    self.per_band_xmin = Some(xmin);
                }

                // Pick the smallest global_gain + scalefactor configuration.
                // Two paths:
                //   * fixed-gain: zero scalefactors + inner loop only
                //     (the r138 path; kept for reference / debug).
                //   * outer-loop: §C.1.5.4.3 distortion-control loop on
                //     top of the inner loop, with non-zero per-band
                //     scalefactor amplification driven by the uniform
                //     `xmin[sb]` threshold.
                let gc_template =
                    if self.auto_block_type.is_some() || self.model2_block_type.is_some() {
                        // Auto block-type (energy-detector or
                        // §C.1.5.3.2.1 Model-2-driven): pick the side-info
                        // skeleton that matches the per-(gr, ch) chosen
                        // block type from the pre-pass above. Long stays on
                        // the default-long skeleton; Start/End take the
                        // long-family transition skeleton; Short takes
                        // either the pure-short skeleton or the mixed
                        // skeleton based on `mixed_per_gc[gr][ch]` from
                        // the mixed classifier (r161:
                        // `enable_auto_block_type_with_mixed`). The
                        // Model-2-driven path never promotes to mixed (it
                        // wires no mixed classifier), so `mixed_per_gc` is
                        // always false there → pure-short. Pre-r161 energy
                        // auto paths likewise have `mixed_per_gc` false.
                        match block_type_per_gc[gr][ch] {
                            BlockType::Long => default_long_gc(),
                            BlockType::Start | BlockType::End => {
                                let mut gc = default_transition_gc(block_type_per_gc[gr][ch]);
                                // A Start / End flanking a mixed burst
                                // carries the §2.4.2.7 mixed_block_flag
                                // (normal window on the two lowest
                                // subbands — see the forward-MDCT
                                // dispatch above). The flag changes
                                // ONLY the synthesis window choice:
                                // spectral layout, scalefactor
                                // partitions, requantization, and the
                                // §2.4.2.7 region defaults all key on
                                // `block_type == 2`, so the long-family
                                // coding path below is untouched.
                                gc.mixed_block_flag = mixed_per_gc[gr][ch];
                                gc
                            }
                            BlockType::Short if mixed_per_gc[gr][ch] => default_mixed_gc(),
                            BlockType::Short => default_short_gc(),
                        }
                    } else if self.force_short_blocks {
                        default_short_gc()
                    } else if self.force_mixed_blocks {
                        default_mixed_gc()
                    } else {
                        default_long_gc()
                    };
                let per_gc_bits = self.per_gc_bit_budget();
                // Part2 (scalefactor) cost under `scalefac_compress = 15`
                // (slen1 = 4, slen2 = 3) depends on block type:
                //
                //   * Long: 21 long bands grouped 11·slen1 + 10·slen2
                //     (the §C.1.5.4.3.6 partitioning the outer loop uses).
                //     Cost: 11·4 + 10·3 = 74 bits.
                //   * Pure-short (`block_type == Short`,
                //     `mixed_block_flag == false`): 12 short SFB × 3
                //     windows, slen1 for sfb 0..6 and slen2 for sfb 6..12.
                //     Cost: 6·3·4 + 6·3·3 = 126 bits.
                //   * Mixed (`block_type == Short`,
                //     `mixed_block_flag == true`): 8 long sfb (slen1
                //     each) + short sfb 3..12 × 3 windows (slen1 for
                //     sfb 3..6, slen2 for sfb 6..12). Cost:
                //     8·4 + 3·3·4 + 6·3·3 = 32 + 36 + 54 = 122 bits.
                //
                // See `crate::scalefactors::write_mpeg1_granule_channel`
                // for the wire layout each branch mirrors.
                let part2_bits_outer: usize = if gc_template.window_switching_flag
                    && gc_template.block_type == BlockType::Short
                {
                    if gc_template.mixed_block_flag {
                        8 * 4 + 3 * 3 * 4 + 6 * 3 * 3
                    } else {
                        6 * 3 * 4 + 6 * 3 * 3
                    }
                } else {
                    11 * 4 + 10 * 3
                };
                let inner_budget_for_outer = per_gc_bits.saturating_sub(part2_bits_outer) as u64;
                // The outer loop runs on all four block-type shapes the
                // encoder ever emits: long (r144 path,
                // `outer_loop_search_long`), pure-short (r157
                // `outer_loop_search_short`), mixed (r159
                // `outer_loop_search_mixed`), and — new in r160 — the
                // long-family transition skeletons Start (block_type 1)
                // and End (block_type 3). Start / End share part2 wire
                // layout, requantize formula, and region-split rule
                // with Long (see `outer_loop_search_long`'s doc on
                // long-family acceptance), so they reuse the same
                // primitive with a relaxed debug_assert. No block-type
                // ever falls back to the fixed-gain inner-loop-only
                // path while the outer loop is enabled.
                // Start / End are eligible with or without the
                // mixed_block_flag: the flag on a long-family granule
                // selects only the synthesis window of subbands 0..2
                // (§2.4.2.7) — part2 wire layout, requantize formula,
                // and region split are the plain long-family ones
                // either way.
                let outer_loop_eligible = matches!(
                    (
                        gc_template.window_switching_flag,
                        gc_template.block_type,
                        gc_template.mixed_block_flag,
                    ),
                    (false, BlockType::Long, _)
                        | (true, BlockType::Short, _)
                        | (true, BlockType::Start, _)
                        | (true, BlockType::End, _)
                );
                let (mut sf, initial_gain, scalefac_scale_outer, subblock_gain_outer) =
                    match self.outer_loop_threshold {
                        Some(thr) if outer_loop_eligible => {
                            // Outer loop seeds scalefac_compress = 15 so the
                            // chosen per-band scalefactors can be written
                            // back as part2.
                            let mut gc_for_ol = gc_template;
                            gc_for_ol.scalefac_compress = OUTER_LOOP_SCALEFAC_COMPRESS;
                            match gc_template.block_type {
                                // Long-family — Long, Start, End share
                                // a primitive. The Long arm runs on
                                // every (`window_switching_flag = false`,
                                // `block_type = Long`) granule; the
                                // Start / End arms below it route the
                                // long-family transition skeletons
                                // (`window_switching_flag = true`,
                                // `block_type ∈ {Start, End}`) onto the
                                // same `outer_loop_search_long`
                                // primitive (it accepts the long family
                                // — see its doc on long-family
                                // acceptance).
                                BlockType::Long => {
                                    // r194 step 39: when a per-band
                                    // threshold vector is installed,
                                    // dispatch onto the
                                    // `*_per_band` long primitive that
                                    // reads `xmin.long[sfb]` instead of
                                    // the uniform scalar. The per-band
                                    // primitive is a strict
                                    // generalisation — installing
                                    // `XminThresholds::uniform(thr)`
                                    // recovers byte-for-byte the
                                    // scalar-path output.
                                    let res = if let Some(xmin) = &self.per_band_xmin {
                                        crate::outer_loop::outer_loop_search_long_per_band(
                                            &xr_pre,
                                            &gc_for_ol,
                                            self.sample_rate_hz,
                                            self.version,
                                            inner_budget_for_outer,
                                            &xmin.long,
                                            DEFAULT_OUTER_LOOP_MAX_ITER,
                                        )
                                    } else {
                                        outer_loop_search_long(
                                            &xr_pre,
                                            &gc_for_ol,
                                            self.sample_rate_hz,
                                            self.version,
                                            inner_budget_for_outer,
                                            thr,
                                            DEFAULT_OUTER_LOOP_MAX_ITER,
                                        )
                                    };
                                    // The outer loop reports:
                                    //   * `scalefac_scale` — §C.1.5.4.3
                                    //     dynamic-range escalation
                                    //     (multiplier 1.0 vs 0.5);
                                    //   * `scalefactors.preflag` —
                                    //     §C.1.5.4.3.4 preemphasis
                                    //     (Table B.6 pretab boost on the
                                    //     upper bands).
                                    // Both must be propagated into the
                                    // granule-channel so the re-quantize
                                    // step below and the side-info write
                                    // reflect what the outer loop
                                    // converged on. `sf.preflag` is what
                                    // `quantize()` reads; `gc.preflag` is
                                    // what the side-info encoder writes —
                                    // we mirror them below at the top of
                                    // the `loop`.
                                    (
                                        res.scalefactors,
                                        res.global_gain,
                                        res.scalefac_scale,
                                        [0u8; 3],
                                    )
                                }
                                BlockType::Short if !gc_template.mixed_block_flag => {
                                    // Pure-short outer loop (§C.1.5.4.3
                                    // analogue) — r157
                                    // `outer_loop_search_short`. Reports
                                    // per-window `subblock_gain` (raised
                                    // from zero on §C.1.5.4.4.2
                                    // magnitude-clamp failures); preflag
                                    // is invariant `false` for short blocks
                                    // (§2.4.2.7). The MPEG-1 part2 wire
                                    // layout for pure-short reads
                                    // `sf.short[sfb][win]` for sfb 0..12
                                    // (see `write_mpeg1_granule_channel`).
                                    //
                                    // r197 step 40: when a per-band
                                    // threshold matrix is installed,
                                    // dispatch onto the
                                    // `*_per_band` short primitive that
                                    // reads `xmin.short[sfb][win]` instead
                                    // of the uniform scalar. The per-band
                                    // primitive is a strict generalisation
                                    // — installing
                                    // `XminThresholds::uniform(thr)`
                                    // recovers byte-for-byte the
                                    // scalar-path output.
                                    let res = if let Some(xmin) = &self.per_band_xmin {
                                        crate::outer_loop::outer_loop_search_short_per_band(
                                            &xr_pre,
                                            &gc_for_ol,
                                            self.sample_rate_hz,
                                            self.version,
                                            inner_budget_for_outer,
                                            &xmin.short,
                                            DEFAULT_OUTER_LOOP_MAX_ITER,
                                        )
                                    } else {
                                        outer_loop_search_short(
                                            &xr_pre,
                                            &gc_for_ol,
                                            self.sample_rate_hz,
                                            self.version,
                                            inner_budget_for_outer,
                                            thr,
                                            DEFAULT_OUTER_LOOP_MAX_ITER,
                                        )
                                    };
                                    (
                                        res.scalefactors,
                                        res.global_gain,
                                        res.scalefac_scale,
                                        res.subblock_gain,
                                    )
                                }
                                BlockType::Short => {
                                    // Mixed-block outer loop (§C.1.5.4.3
                                    // analogue) — r159
                                    // `outer_loop_search_mixed`. Composes
                                    // the long-region amplifier over
                                    // sf.long[0..=7] with the short-region
                                    // per-(sfb, window) amplifier over
                                    // sf.short[3..=11][..]; reports per-
                                    // window `subblock_gain` like the
                                    // pure-short loop. Preflag stays
                                    // `false` (§2.4.2.7 disables preflag
                                    // on every short-family granule
                                    // including mixed). The MPEG-1 part2
                                    // wire layout for mixed reads
                                    // `sf.long[0..8]` at slen1 then
                                    // `sf.short[3..12][..]` (see
                                    // `write_mpeg1_granule_channel`).
                                    //
                                    // r204 step 41: when a per-band
                                    // threshold matrix is installed,
                                    // dispatch onto the
                                    // `*_per_band` mixed primitive that
                                    // reads `xmin.mixed_long[sfb]` for
                                    // the long region (sfb 0..=7) and
                                    // `xmin.mixed_short[sfb][win]` for
                                    // the short region (sfb 3..=11)
                                    // instead of the uniform scalar.
                                    // The per-band primitive is a strict
                                    // generalisation — installing
                                    // `XminThresholds::uniform(thr)`
                                    // recovers byte-for-byte the
                                    // scalar-path output.
                                    let res = if let Some(xmin) = &self.per_band_xmin {
                                        crate::outer_loop::outer_loop_search_mixed_per_band(
                                            &xr_pre,
                                            &gc_for_ol,
                                            self.sample_rate_hz,
                                            self.version,
                                            inner_budget_for_outer,
                                            &xmin.mixed_long,
                                            &xmin.mixed_short,
                                            DEFAULT_OUTER_LOOP_MAX_ITER,
                                        )
                                    } else {
                                        outer_loop_search_mixed(
                                            &xr_pre,
                                            &gc_for_ol,
                                            self.sample_rate_hz,
                                            self.version,
                                            inner_budget_for_outer,
                                            thr,
                                            DEFAULT_OUTER_LOOP_MAX_ITER,
                                        )
                                    };
                                    (
                                        res.scalefactors,
                                        res.global_gain,
                                        res.scalefac_scale,
                                        res.subblock_gain,
                                    )
                                }
                                // Long-family transition skeletons —
                                // route onto `outer_loop_search_long`
                                // exactly like `BlockType::Long` above
                                // (the primitive accepts the long
                                // family). New in r160: Start / End
                                // were previously fixed-gain
                                // fallbacks; their part2 layout +
                                // requantize formula is identical to
                                // Long so the same primitive serves
                                // all three.
                                BlockType::Start | BlockType::End => {
                                    // Same long-family primitive as the
                                    // `BlockType::Long` arm — see the
                                    // r194 per-band dispatch comment
                                    // above.
                                    let res = if let Some(xmin) = &self.per_band_xmin {
                                        crate::outer_loop::outer_loop_search_long_per_band(
                                            &xr_pre,
                                            &gc_for_ol,
                                            self.sample_rate_hz,
                                            self.version,
                                            inner_budget_for_outer,
                                            &xmin.long,
                                            DEFAULT_OUTER_LOOP_MAX_ITER,
                                        )
                                    } else {
                                        outer_loop_search_long(
                                            &xr_pre,
                                            &gc_for_ol,
                                            self.sample_rate_hz,
                                            self.version,
                                            inner_budget_for_outer,
                                            thr,
                                            DEFAULT_OUTER_LOOP_MAX_ITER,
                                        )
                                    };
                                    (
                                        res.scalefactors,
                                        res.global_gain,
                                        res.scalefac_scale,
                                        [0u8; 3],
                                    )
                                }
                            }
                        }
                        _ => {
                            // Fixed-gain inner-loop path. Reached when
                            // (a) outer loop disabled, or (b) outer
                            // loop enabled but block type is Start /
                            // End and no outer-loop primitive covers
                            // that shape yet.
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
                                // §C.1.5.4.4 rate control: choose the gain
                                // against the *band-aligned* SUBDIVIDE bit
                                // count, the wire-representable §C.1.5.4.4.6
                                // partition this encoder actually emits below
                                // (`choose_region_split` / `subdivide_bands`
                                // both snap the boundaries to scalefactor-band
                                // edges). The default `search_bit_budget`
                                // counts bits against the pair-thirds
                                // heuristic ([`subdivide`]), whose boundaries
                                // can land mid-band — a part2_3 length the
                                // decoder's `region_boundaries` cannot
                                // reconstruct, so the gain it returned was
                                // gated on a split the encoder never writes.
                                // The band-aligned count matches the emitted
                                // part2_3 length, so the gain fits the real
                                // wire budget. (Short / mixed blocks share the
                                // two-subregion blocksplit path, so this is
                                // bit-identical to the default for them.)
                                let res_budget = search_bit_budget_band_aligned(
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
                            (sf, initial_gain, false, [0u8; 3])
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
                //
                // The outer-loop branches above bake non-zero
                // scalefactors into `sf`; the fallback path leaves
                // `sf` at default (zeros). Only the outer-loop branches
                // need `scalefac_compress = 15` written into the
                // side-info — the fallback path's zero scalefactors
                // round-trip equally with `scalefac_compress = 0`.
                let ran_outer_loop = self.outer_loop_threshold.is_some() && outer_loop_eligible;
                // The right channel of an intensity frame must carry
                // its per-band stereo positions (`is_pos` ∈ 0..=6 plus
                // the marker 7) as scalefactors, so its
                // scalefac_compress needs ≥ 3 bits per band even when
                // the outer loop (which would pick 15 anyway) is off.
                // `scalefac_compress = 15` gives `slen1 = 4` /
                // `slen2 = 3` (§2.4.2.7 table) — every position fits.
                // `intensity_active` gates the mode globally;
                // `intensity_coupled_per_gr[gr]` gates it per granule for
                // the adaptive chooser (a granule that coupled nothing
                // writes a normal right channel, not is_pos markers).
                let intensity_right = intensity_active && ch == 1 && intensity_coupled_per_gr[gr];
                // LSF carries a 9-bit scalefac_compress with the
                // §2.4.3.2 (13818-3) slen derivation. The LSF
                // outer-loop value 399 decodes to slen (4, 4, 3, 3)
                // over the long-block partition (6, 5, 5, 5) — i.e.
                // slen 4 on bands 0..=10 and slen 3 on bands 11..=20,
                // the same per-band caps (15 / 7) and the same 74-bit
                // part2 cost as the MPEG-1 outer-loop value 15
                // (slen1 = 4, slen2 = 3); the short-block partition
                // (9, 9, 9, 9) likewise reproduces the MPEG-1 short
                // caps (sfb 0..=5 at 4 bits, 6..=11 at 3 bits,
                // 126 bits). The outer loop's internal bit accounting
                // therefore carries over unchanged; only the written
                // field value differs.
                let outer_sfc = if self.version.is_lsf() {
                    OUTER_LOOP_SCALEFAC_COMPRESS_LSF
                } else {
                    OUTER_LOOP_SCALEFAC_COMPRESS
                };
                // The LSF intensity-stereo right channel takes the
                // §2.4.3.2 `int_scalefac_compress` partition value, not
                // the non-intensity LSF outer-loop value: its
                // scalefactor slots carry intensity positions read back
                // through the right-channel-only partition tables (and
                // the slen-3 width makes `7` the illegal-position
                // marker the decoder tests for). MPEG-1 intensity-right
                // keeps the 4-bit `scalefac_compress = 15` (slen1=4,
                // slen2=3) layout — positions fit at ≥ 3 bits per band.
                let scalefac_compress = if intensity_right && self.version.is_lsf() {
                    INTENSITY_SCALEFAC_COMPRESS_LSF
                } else if ran_outer_loop || intensity_right {
                    outer_sfc
                } else {
                    0
                };
                // LSF has no preflag bit on the wire and the
                // scalefac_compress ranges below 500 all decode
                // preflag = 0 (13818-3 §2.4.3.2), so a preflag the
                // outer loop converged on cannot be transmitted as a
                // flag. Fold the Table B.6 pretab into the long
                // scalefactors instead — the §2.4.3.4.7.1 exponent
                // reads `scalefac + preflag·pretab`, so transmitting
                // `scalefac + pretab` with preflag = 0 reproduces the
                // identical requantize exponent — and drop the fold
                // (clearing preflag before the re-quantize below, so
                // encoder and side-info stay in agreement) on the rare
                // band where the folded value would overflow its
                // 3-bit upper-band cap.
                if self.version.is_lsf() && sf.preflag {
                    let fits = (0..21).all(|sfb| {
                        let cap = if sfb < 11 { 15 } else { 7 };
                        u16::from(sf.long[sfb]) + u16::from(crate::requantize::PRETAB[sfb]) <= cap
                    });
                    if fits {
                        for sfb in 0..21 {
                            sf.long[sfb] += crate::requantize::PRETAB[sfb];
                        }
                    }
                    sf.preflag = false;
                }
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
                    // §2.4.2.7 subblock_gain: the short-block outer loop
                    // raises individual windows' per-window gain off zero
                    // when the §C.1.5.4.4.2 magnitude clamp can't fit a
                    // window under 8191. Mirror those into the granule;
                    // for every other block type / fallback path the
                    // outer-loop dispatcher leaves it as `[0; 3]`, which
                    // matches the template default.
                    gc.subblock_gain = subblock_gain_outer;
                    is = quantize(&xr_pre, &gc, &sf, self.sample_rate_hz, self.version);
                    clamp_above(&mut is, 8191);
                    split = partition_split(&is);
                    bv2 = split.big_pairs * 2;
                    gc.big_values = split.big_pairs as u16;
                    if self.force_short_blocks
                        || self.force_mixed_blocks
                        || gc_template.window_switching_flag
                    {
                        // §C.1.5.4.4.6 + huffman::region_boundaries:
                        // for window-switched short-family granules
                        // the region counts are not on the wire
                        // (encoder::write_granule_channel writes the
                        // window-switched branch which omits them), so
                        // the encoder must use exactly the boundaries
                        // every decoder reconstructs from the §2.4.2.7
                        // defaults: pure short blocks put nine
                        // window-bands (short sfb 0..=2 × 3 windows)
                        // in region 0, i.e. region 0 ends at
                        // interleaved line `3 · short_starts[3]` — 36
                        // lines for every ISO table, 72 lines for the
                        // MPEG-2.5 8 kHz Fraunhofer table
                        // (`short_starts[3] = 24`; r405
                        // observer-trace) — and region 1 runs to the
                        // rest of big_values (region 2 empty). Keep
                        // the spec-default sentinels from the short /
                        // mixed template intact.
                        //
                        // Mixed-block detail: §2.4.2.7 sets
                        // `region0_count = 7` for mixed granules; the
                        // mixed band sequence opens with the
                        // transmitted long bands of the long-coded
                        // region, whose span equals
                        // `3 · short_starts[3]` at every rate, so the
                        // mixed region 0 ends at the same
                        // band-relative line as the pure-short one
                        // (r408). Region 1 then carries all
                        // big_values that fall in the short region,
                        // already re-ordered into `[sfb][win][k]`
                        // native order by `forward_reorder` above.
                        //
                        // Window-switched **long-family** granules
                        // (`Start` / `End`, r405 fix): these take the
                        // same 22-bit window-switched side-info branch
                        // — region counts and `table_select[2]` are
                        // NOT on the wire — so every decoder
                        // reconstructs the §2.4.2.7 defaults
                        // `region0_count = 7`, `region1_count = 63`:
                        // region 0 = long bands 0..=7
                        // (`long_starts[8]` lines), region 1 = the
                        // rest of big_values, region 2 empty. This
                        // arm previously routed Start / End through
                        // `choose_region_split`, whose optimized
                        // boundaries (and third table) can never
                        // reach the decoder — any granule where the
                        // chosen split disagreed with the defaults
                        // desynchronized the Huffman regions of
                        // every conforming decoder (observed as
                        // sporadic transition-granule corruption in
                        // the r405 black-box validator sweep, e.g.
                        // nrmse 2e-2 bursts at 44.1 kHz auto
                        // block-type).
                        //
                        // The auto-block-type path
                        // ([`Mp3Encoder::enable_auto_block_type`]) may
                        // emit any window-switched block type without
                        // a force-toggle, so the gate covers every
                        // window-switched template — the wire layout
                        // and the §2.4.4.5 bit-cost check are then
                        // consistent.
                        let r0_lines = match gc_template.block_type {
                            // Pure short AND mixed both end region 0
                            // at `3 · short_starts[3]` (36 lines at
                            // every ISO table; 72 at the 8 kHz
                            // Fraunhofer tables — mixed is refused at
                            // 8 kHz but the formulas stay unified;
                            // r408 observer probes confirmed the
                            // band-relative mixed boundary on all
                            // four deployed validators, see
                            // huffman::region_boundaries).
                            BlockType::Short => 3 * short_band_starts_for(self.sample_rate_hz)[3],
                            // Start / End: §2.4.2.7 default
                            // region0_count = 7 in long-band units.
                            _ => long_band_starts_for(self.sample_rate_hz)[8],
                        };
                        r0_end = r0_lines.min(bv2);
                        r1_end = bv2;
                    } else {
                        let (r0e, r1e, r0c, r1c) =
                            choose_region_split(self.sample_rate_hz, self.version, bv2);
                        r0_end = r0e;
                        r1_end = r1e;
                        gc.region0_count = r0c;
                        gc.region1_count = r1c;
                    }
                    if gc_template.block_type == BlockType::Short && gc_template.mixed_block_flag {
                        // Mixed granules: one codebook for BOTH
                        // big-values regions (r408). The §2.4.2.7
                        // mixed region-0 boundary is a deployed grey
                        // zone — the r408 observer probes measured
                        // four black-box validators splitting 2-2 at
                        // the LSF rates (two put it at
                        // `3·short_starts[3]`, two at the end of the
                        // literal eight-entry mixed band sequence)
                        // and three ways at 8 kHz. With
                        // `table_select[0] == table_select[1]` every
                        // boundary interpretation consumes identical
                        // bits, so the ambiguity cannot desynchronise
                        // any deployed decoder. Costs a few bits on
                        // mixed granules only.
                        t0 = best_table_or(&is, 0, r1_end);
                        t1 = t0;
                    } else {
                        t0 = best_table_or(&is, 0, r0_end);
                        t1 = best_table_or(&is, r0_end, r1_end);
                    }
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
                    let budget_for_part3 = if ran_outer_loop {
                        inner_budget_for_outer as usize
                    } else if intensity_right {
                        // Fixed-gain path with the forced
                        // `scalefac_compress = 15`: part2 claims the
                        // long-layout 74 bits out of the per-gc budget.
                        per_gc_bits.saturating_sub(part2_bits_outer)
                    } else {
                        per_gc_bits
                    };
                    if total <= budget_for_part3 || global_gain == 255 {
                        break;
                    }
                    global_gain = global_gain.saturating_add(1);
                }
                // §2.4.3.4.9.3 wire-up for the right channel of an
                // intensity frame, applied after the quantizer
                // converged (the rewritten bands hold only zero lines,
                // so the scalefactor values cannot perturb `is[]`):
                //
                //   * bands at/above the intensity bound carry the
                //     pass-1.45 `is_pos` positions;
                //   * all-zero bands *below* the bound that follow the
                //     last non-zero quantized right-channel line carry
                //     the marker 7 (Annex G.2 c) — the decode-side
                //     bound is the band after that last non-zero line
                //     (§2.4.3.4.9.1), so without the marker those
                //     bands would be intensity-decoded with whatever
                //     scalefactor the quantizer left behind.
                if intensity_right && mixed_intensity_gr[gr] {
                    // §2.4.3.4.10.3 mixed-block right channel: positions
                    // and markers go on BOTH `scalefac_l[sfb]` (long
                    // region, bands 0..8) and `scalefac_s[sfb][win]`
                    // (short region, bands 3..12 per window) — the two
                    // regions the decoder's `process_short` rebuilds for a
                    // `mixed_block_flag` granule.
                    let short_starts = short_band_starts_for(self.sample_rate_hz);
                    let long_starts = long_band_starts_for(self.sample_rate_hz);
                    let start_sfb = self.intensity_start_sfb.unwrap_or(LONG_SFB);
                    let long_end = MIXED_LAST_LONG_SFB + 1; // = 8

                    // --- Long region markers + positions (bands 0..8) ---
                    // The decoder derives the long-region bound from the
                    // last non-zero right line in lines 0..36 (one band
                    // past it), so the all-zero bands between that bound and
                    // the coupled start carry the illegal marker 7; coupled
                    // bands carry the pass-1.45 `is_pos`.
                    // The mixed long region is the lowest 2 polyphase
                    // subbands = lines 0..36; long band 8 starts at line 36
                    // for every sample rate, so `long_starts[long_end]`
                    // bounds the region exactly.
                    let long_region_hi = long_starts[long_end];
                    let mut last_nz_long: Option<usize> = None;
                    for (i, &v) in is.iter().enumerate().take(long_region_hi) {
                        if v != 0 {
                            last_nz_long = Some(i);
                        }
                    }
                    let zero_tail_from_long = match last_nz_long {
                        None => 0,
                        Some(line) => {
                            let mut band = long_end;
                            for sfb in 0..long_end {
                                if line < long_starts[sfb + 1] {
                                    band = sfb + 1;
                                    break;
                                }
                            }
                            band
                        }
                    };
                    for sfb in zero_tail_from_long..start_sfb.min(long_end) {
                        sf.long[sfb] = 7;
                    }
                    for sfb in start_sfb.min(long_end)..long_end {
                        sf.long[sfb] = is_pos_per_gr[gr][sfb];
                    }

                    // --- Short region markers + positions (bands 3..12) ---
                    // Map the long start line onto a short band, clamped to
                    // `MIXED_FIRST_SHORT_SFB` (the lowest three short bands
                    // are absorbed by the long carve-out and never carry a
                    // short-region position).
                    let start_line = long_starts[start_sfb];
                    let mapped = (0..12)
                        .find(|&sfb| short_starts[sfb] >= start_line)
                        .unwrap_or(12);
                    let short_start = mapped.max(MIXED_FIRST_SHORT_SFB);
                    for win in 0..3 {
                        let mut last_nz_sfb: Option<usize> = None;
                        for sfb in MIXED_FIRST_SHORT_SFB..12 {
                            let s = short_starts[sfb];
                            let w = short_starts[sfb + 1] - short_starts[sfb];
                            let win_base = 3 * s + win * w;
                            for k in 0..w {
                                let i = win_base + k;
                                if i < NUM_LINES && is[i] != 0 {
                                    last_nz_sfb = Some(sfb);
                                }
                            }
                        }
                        let zero_tail_from = match last_nz_sfb {
                            None => MIXED_FIRST_SHORT_SFB,
                            Some(sfb) => sfb + 1,
                        };
                        for sfb in zero_tail_from..short_start.min(12) {
                            sf.short[sfb][win] = 7;
                        }
                        for sfb in short_start..12 {
                            sf.short[sfb][win] = is_pos_short_per_gr[gr][sfb][win];
                        }
                    }
                } else if intensity_right && short_intensity_gr[gr] {
                    // Short-block right channel: positions and markers go
                    // on `scalefac_s[sfb][win]`, derived independently per
                    // window. The decoder's per-window bound
                    // (§2.4.3.4.8 reorder, then 13818-3 §2.4.3.2) is the
                    // short band one past that window's last non-zero
                    // quantized right line, so each window's all-zero
                    // bands above its own last non-zero line — and below
                    // the coupled start — take the illegal marker 7 to
                    // keep them out of the intensity reconstruction.
                    let short_starts = short_band_starts_for(self.sample_rate_hz);
                    let long_starts = long_band_starts_for(self.sample_rate_hz);
                    let start_sfb = self.intensity_start_sfb.unwrap_or(LONG_SFB);
                    let start_line = long_starts[start_sfb];
                    let short_start = (0..12)
                        .find(|&sfb| short_starts[sfb] >= start_line)
                        .unwrap_or(12);
                    for win in 0..3 {
                        // Per-window last non-zero quantized line, in
                        // native bitstream layout (band `sfb` window `win`
                        // line `k` at `3·s + win·w + k`).
                        let mut last_nz_sfb: Option<usize> = None;
                        for sfb in 0..12 {
                            let s = short_starts[sfb];
                            let w = short_starts[sfb + 1] - short_starts[sfb];
                            let win_base = 3 * s + win * w;
                            for k in 0..w {
                                let i = win_base + k;
                                if i < NUM_LINES && is[i] != 0 {
                                    last_nz_sfb = Some(sfb);
                                }
                            }
                        }
                        let zero_tail_from = match last_nz_sfb {
                            None => 0,
                            Some(sfb) => sfb + 1,
                        };
                        for sfb in zero_tail_from..short_start.min(12) {
                            sf.short[sfb][win] = 7;
                        }
                        for sfb in short_start..12 {
                            sf.short[sfb][win] = is_pos_short_per_gr[gr][sfb][win];
                        }
                    }
                } else if intensity_right {
                    let starts = long_band_starts_for(self.sample_rate_hz);
                    let start_sfb = self.intensity_start_sfb.unwrap_or(LONG_SFB);
                    let zero_tail_from = match is.iter().rposition(|&v| v != 0) {
                        // Fully-zero right channel: the zero-part spans
                        // the whole spectrum and every band below the
                        // bound takes the marker.
                        None => 0,
                        Some(line) => {
                            // Band holding `line`, plus one (mirrors
                            // the decoder's bound derivation).
                            let mut band = LONG_SFB;
                            for sfb in 0..LONG_SFB {
                                if line < starts[sfb + 1] {
                                    band = sfb + 1;
                                    break;
                                }
                            }
                            band
                        }
                    };
                    for sfb in zero_tail_from..start_sfb {
                        sf.long[sfb] = 7;
                    }
                    for sfb in start_sfb..LONG_SFB {
                        sf.long[sfb] = is_pos_per_gr[gr][sfb];
                    }
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

        // ---- §C.1.5.3 scfsi reuse (MPEG-1, two granules only) ----
        //
        // Detect, per channel, every long-block scfsi_band group whose
        // granule-1 scalefactors already equal granule 0's, and mark it
        // for reuse. The §2.4.2.7 write path skips a reused group in
        // granule 1; the decoder reproduces granule 0's values there,
        // so the reconstructed scalefactors are bit-identical and only
        // the part2 bit count shrinks. Auto-armed by default (r301);
        // `disable_scfsi_reuse` clears `self.scfsi_reuse` to restore the
        // historical `scfsi == 0` output.
        if self.scfsi_reuse && !self.version.is_lsf() && ngr == 2 {
            for ch in 0..self.nch {
                side_info.scfsi[ch] = compute_scfsi_reuse(
                    &side_info.granules[0][ch],
                    &side_info.granules[1][ch],
                    &scalefactors.granules[0][ch],
                    &scalefactors.granules[1][ch],
                );
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
        let si_bytes = side_info_byte_len(self.version, self.nch);
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
        // ((bitrate * slots_per_frame) mod sample_rate). For
        // deterministic mono 128 kbit/s @ 44.1 kHz that's pad every ~9
        // frames. Use a running accumulator: `acc += rem; if acc >= sr
        // { pad = true; acc -= sr }` — the classic Bresenham-style CBR
        // pad ladder. The slots_per_frame constant is 144 for MPEG-1
        // and 72 for the LSF versions (ISO/IEC 13818-3 §2.4.3.2,
        // "Changed constants for ISO/IEC 13818-3 Layer III").
        let bitrate_bps = u64::from(self.header_template.bitrate_kbps.unwrap_or(0)) * 1000;
        let sr64 = u64::from(self.sample_rate_hz);
        let slots_coeff: u64 = if self.version.is_lsf() { 72 } else { 144 };
        let rem = (slots_coeff * bitrate_bps) % sr64;
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
                    pick_vbr_bitrate(self.sample_rate_hz, vbr_cfg, need, si_bytes, crc_bytes)
                        .ok_or(StreamEncodeError::VbrSlotTooSmall {
                            frame_index: i,
                            main_data_len: need,
                            max_slot_bytes: ladder_slot_capacity(
                                self.sample_rate_hz,
                                vbr_cfg.max_index,
                                si_bytes,
                                /*padded=*/ true,
                            )
                            .saturating_sub(crc_bytes),
                        })?;
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
        // §2.4.2.7 caps `main_data_begin` at 511 bytes (MPEG-1) / 255
        // bytes (LSF, 8-bit field — `schedule_reservoir` reads the
        // per-frame `lsf` flag for the cap). Without a psychoacoustic
        // model the per-frame main_data tends to be much smaller than
        // the slot, so the rolling reservoir would grow unbounded
        // across frames. We bound it by **zero-padding** every frame's
        // main_data up to at least its own slot size: the reservoir
        // then never grows above 0 byte (the no-reservoir schedule),
        // every `main_data_begin` is 0, and the scheduler walks the
        // trivial schedule. (A real encoder uses the reservoir to
        // absorb busy-frame overflow; we revisit that once the psy /
        // outer loop lands.)
        let lsf = self.version.is_lsf();
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
                // side-info bits, written big-endian (MSB first) in
                // the two-byte slot between header and side_info. The
                // protected side-info window follows the version: 135
                // (mono) / 256 (other modes) bits for MPEG-1, 72 / 136
                // bits for the shorter LSF side info (ISO/IEC 13818-3
                // §2.4.1.4 defers to the 11172-3 definition over the
                // LSF layout).
                let crc = if lsf {
                    crate::crc::crc16_layer3_lsf(&hbytes, &sib, self.nch as u8)
                } else {
                    crate::crc::crc16_layer3(&hbytes, &sib, self.nch as u8)
                };
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

/// Long-block scalefactor-band start indices for the active sample
/// rate. Index 21 (one past band 20) is the end+1 boundary so callers
/// can read the top of the long-block range as a "next" boundary.
///
/// Delegates to [`crate::requantize::long_band_starts`] — the single
/// in-crate transcription of ISO/IEC 11172-3 Table 3-B.8 (MPEG-1
/// rates) and ISO/IEC 13818-3:1997 Table B.2 (MPEG-2 LSF rates) — so
/// the encoder's region split / intensity band walk and the decoder's
/// requantizer always agree on band boundaries. The sample rate alone
/// determines the version (the three rate families are disjoint), so
/// the version argument is derived here.
fn long_band_starts_for(sample_rate_hz: u32) -> &'static [usize; 22] {
    let version = match sample_rate_hz {
        16_000 | 22_050 | 24_000 => MpegVersion::Mpeg2,
        8_000 | 11_025 | 12_000 => MpegVersion::Mpeg25,
        _ => MpegVersion::Mpeg1,
    };
    crate::requantize::long_band_starts(sample_rate_hz, version)
}

/// Table B.8 §2.4.3.4.8 short-block scalefactor-band boundaries
/// (per-window subband widths) for `sample_rate_hz`, mapping the wire
/// sample-rate to its MPEG version exactly as [`long_band_starts_for`].
/// Each `starts[sfb]` is the per-window first frequency line of short
/// band `sfb`; the native bitstream layout interleaves the three windows
/// of one band so band `sfb` window `win` line `k` lands at
/// `3·starts[sfb] + win·width + k` (`width = starts[sfb+1] -
/// starts[sfb]`), which is the same line index the requantizer fills and
/// — after the §2.4.3.4.8 reorder — the decoder's per-window stereo stage
/// addresses as `3·starts[sfb] + 3·k + win`.
fn short_band_starts_for(sample_rate_hz: u32) -> &'static [usize; 13] {
    let version = match sample_rate_hz {
        16_000 | 22_050 | 24_000 => MpegVersion::Mpeg2,
        8_000 | 11_025 | 12_000 => MpegVersion::Mpeg25,
        _ => MpegVersion::Mpeg1,
    };
    crate::requantize::short_band_starts(sample_rate_hz, version)
}

/// Pick the per-granule §2.4.3.4.9.3 intensity-stereo bound for the
/// adaptive joint-stereo mode
/// ([`Mp3Encoder::new_joint_stereo_auto_is_adaptive`]).
///
/// Returns the lowest long scalefactor band `b` with
/// `floor <= b <= 21` such that **every** band in `b..21` carries
/// little right-channel stereo information, measured by the side-energy
/// fraction
///
/// ```text
/// E_S / (E_L + E_R) = Σ(L − R)² / (2·Σ(L² + R²))   over the band
/// ```
///
/// being at or below `threshold`. The chooser walks the bands from the
/// top (band 20) down to `floor`: it extends the coupled tail downward
/// while each band qualifies and stops at the first band that does not
/// (that band, and everything below it, stays independently coded). A
/// fully-silent band (`E_L + E_R == 0`) has no stereo information and
/// always qualifies. The return value equals `21` when not even the top
/// band qualifies — i.e. the granule is coded with no intensity
/// coupling at all.
///
/// The criterion mirrors the §2.4.3.4.9.2 MS picker
/// ([`Mp3Encoder::ms_auto_threshold`]): a band whose side-energy is a
/// small fraction of the total is one where the two channels are nearly
/// equal in magnitude, so replacing the right channel with a single
/// stereo-position scalar loses little. ISO/IEC 11172-3 fixes only the
/// wire syntax of the bound (the right channel's zero-part); the
/// threshold is a clean-room encoder heuristic.
fn choose_intensity_bound(
    left: &[f32; NUM_LINES],
    right: &[f32; NUM_LINES],
    starts: &[usize; 22],
    floor: usize,
    threshold: f64,
) -> usize {
    let mut bound = 21usize;
    for sfb in (floor..21).rev() {
        let lo = starts[sfb];
        let hi = starts[sfb + 1];
        let mut total = 0.0f64;
        let mut side = 0.0f64;
        for i in lo..hi {
            let l = f64::from(left[i]);
            let r = f64::from(right[i]);
            total += l * l + r * r;
            let d = l - r;
            side += d * d;
        }
        // E_S / (E_L + E_R) = Σ(L−R)² / (2·Σ(L²+R²)). A silent band
        // (total == 0) carries no stereo information ⇒ qualifies.
        let qualifies = if total <= 0.0 {
            true
        } else {
            side / (2.0 * total) <= threshold
        };
        if qualifies {
            bound = sfb;
        } else {
            break;
        }
    }
    bound
}

/// Derive the §2.4.3.4.9.3 intensity-stereo position for one
/// scalefactor band from the per-band channel energies, per Annex
/// G.2 c) of ISO/IEC 11172-3:1993:
///
/// ```text
/// is_pos = NINT( (12/π) · arctan( √(E_L / E_R) ) )
/// ```
///
/// `arctan` of a non-negative amplitude ratio lies in `[0, π/2)`, so
/// the rounded position lies in `0..=6`; the decode side then
/// reproduces the amplitude ratio as `is_ratio = tan(is_pos·π/12)`
/// (§2.4.3.4.9.3 step 3), quantized to the nearest of the seven grid
/// angles. A band with zero right-channel energy maps to the
/// `E_R → 0` limit `6` (all-left); a fully-silent band takes the same
/// value (any position decodes a zero magnitude to zero). The value 7
/// is never produced — it is the §2.4.3.4.9.3 illegal-position marker.
fn derive_intensity_position(l_energy: f64, r_energy: f64) -> u8 {
    if r_energy <= 0.0 {
        return 6;
    }
    let amplitude_ratio = (l_energy / r_energy).sqrt();
    let pos = (amplitude_ratio.atan() * 12.0 / std::f64::consts::PI).round();
    pos.clamp(0.0, 6.0) as u8
}

/// Derive the **LSF** (ISO/IEC 13818-3 §2.4.3.2) intensity-stereo
/// position for one band from the per-band channel energies.
///
/// LSF replaces the MPEG-1 `tan` grid (steps 3-5 of §2.4.3.4.9.3) with
/// the §2.4.3.2 step-4/5 power-law ladder: the decoder reconstructs the
/// transmitted magnitude `T = L_i + R_i` (the encoder couples
/// `left := L+R`, `right := 0`) as `L' = T·kl`, `R' = T·kr` with
///
/// ```text
/// i0 = 2^(-1/4)              (intensity_scale == 0, this encoder's choice)
/// is_pos == 0   -> kl = 1,            kr = 1
/// is_pos odd    -> kl = i0^((p+1)/2), kr = 1
/// is_pos even>0 -> kl = 1,            kr = i0^(p/2)
/// ```
///
/// so the decoded amplitude ratio `L'/R' = kl/kr` lands on the
/// non-uniform grid `{1, i0, 1/i0, i0², 1/i0², i0³, 1/i0³}` for
/// `p = 0..=6`. This routine picks the `p` whose decoded ratio is
/// closest **in log space** to the original amplitude ratio
/// `√(E_L/E_R)`, which is the natural distance for a geometric grid
/// (equal multiplicative error above and below). `7` is never produced
/// — it is the §13818-3 illegal-position marker (the maximum value for
/// the `slen = 3` partition this encoder writes).
fn derive_intensity_position_lsf(l_energy: f64, r_energy: f64) -> u8 {
    // i0 = 2^(-1/4); ln(grid ratio) for p = 0..=6 is a multiple of
    // ln(i0): p odd contributes -((p+1)/2)·ln(i0⁻¹) on the left,
    // p even>0 contributes +(p/2)·ln(i0⁻¹) on the right. With
    // a = ln(i0⁻¹) = (1/4)·ln 2 > 0 the grid log-ratios are:
    //   p: 0    1    2    3    4    5    6
    //   ln(L'/R')/a: 0   -1   +1   -2   +2   -3   +3
    const A: f64 = std::f64::consts::LN_2 * 0.25; // ln(i0^-1)
                                                  // Original target in the same log units. Guard the degenerate
                                                  // energies: a silent right channel pans fully left (largest grid
                                                  // ratio, p = 6); a silent left channel pans fully right (p = 5,
                                                  // the smallest grid ratio i0³). A fully-silent band decodes to
                                                  // zero under any position, so its value is immaterial — fall into
                                                  // the r_energy <= 0 branch (p = 6) by convention.
    if r_energy <= 0.0 {
        return 6;
    }
    if l_energy <= 0.0 {
        return 5;
    }
    // target_units = ln(√(E_L/E_R)) / a = ln(E_L/E_R) / (2a).
    let target_units = (l_energy / r_energy).ln() / (2.0 * A);
    // Grid log-ratios in units of `a`, indexed by position 0..=6.
    const GRID_UNITS: [f64; 7] = [0.0, -1.0, 1.0, -2.0, 2.0, -3.0, 3.0];
    let mut best = 0u8;
    let mut best_err = f64::INFINITY;
    for (p, &g) in GRID_UNITS.iter().enumerate() {
        let err = (g - target_units).abs();
        if err < best_err {
            best_err = err;
            best = p as u8;
        }
    }
    best
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

/// Pick the best big-values codebook for `is[start..end]`, delegating
/// to the in-tree [`choose_best_table_for_region`] which (as of r154)
/// applies the §C.1.5.4.4.8 linbits-reach filter so the returned
/// codebook is guaranteed to encode every magnitude in the range
/// without truncation. Falls back to table 23 (`linbits=13`, reach
/// 8206 — covers the §C.1.5.4.4.2 clamp of 8191) on the
/// no-table-in-reach edge case the chooser reserves for corrupt
/// input.
fn best_table_or(is: &[i32; NUM_LINES], start: usize, end: usize) -> u8 {
    choose_best_table_for_region(is, start, end)
        .map(|(t, _)| t)
        .unwrap_or(23)
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
/// §C.1.5.3 scalefactor-selection-information for one channel of an
/// MPEG-1 two-granule frame.
///
/// Returns the four-element `scfsi[ch]` array: `scfsi[g] == true` marks
/// scfsi_band group `g` for reuse, i.e. granule 0's scalefactors in
/// that group are declared valid for granule 1 and are not retransmitted
/// in granule 1's part2.
///
/// The four §2.4.2.7 scfsi_band groups span scalefactor bands
/// `{0..=5}`, `{6..=10}`, `{11..=15}`, `{16..=20}` (Table 3-B.8). A
/// group is eligible iff:
///
/// * **Both granules are long blocks** (`block_type != Short`). Per
///   §2.4.2.7 ("if short windows are switched on … then scfsi is always
///   0 for this frame"), a channel with a short granule in the frame
///   keeps every `scfsi[g] == 0`; this guard makes a single short
///   granule disqualify all four groups for the channel.
/// * **The granule-1 scalefactors equal granule 0's across every band
///   in the group.** The decoder reuses granule 0's `scalefac_l` values
///   verbatim for a marked group, so reuse is lossless only when the
///   two granules already agree there.
///
/// When either granule is short this returns `[false; 4]`.
fn compute_scfsi_reuse(
    gc0: &GranuleChannel,
    gc1: &GranuleChannel,
    sf0: &ScaleFactors,
    sf1: &ScaleFactors,
) -> [bool; 4] {
    // A window-switched short block (block_type == 2) in either granule
    // forbids scfsi for the whole channel (§2.4.2.7). Mixed blocks are
    // also block_type == 2, so this single test covers both.
    if gc0.block_type == BlockType::Short || gc1.block_type == BlockType::Short {
        return [false; 4];
    }
    // scfsi_band groups (§2.4.2.7, Table 3-B.8), as half-open ranges
    // over the 21 long scalefactor bands.
    const GROUPS: [(usize, usize); 4] = [(0, 6), (6, 11), (11, 16), (16, 21)];
    let mut scfsi = [false; 4];
    for (g, &(lo, hi)) in GROUPS.iter().enumerate() {
        scfsi[g] = (lo..hi).all(|sfb| sf0.long[sfb] == sf1.long[sfb]);
    }
    scfsi
}

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

/// A default window-switched long-family granule-channel record for
/// the §C.1.5.2 transition block types — `Start` (block_type 1) and
/// `End` (block_type 3, "Stop"). Identical to
/// [`default_long_gc`] except `window_switching_flag = true`,
/// `block_type` is the carried value, and the region-count fields
/// carry the §2.4.2.7 window-switched defaults every decoder
/// reconstructs (`region0_count = 7`, `region1_count = 63` — not on
/// the wire in the window-switched side-info branch). Used by the
/// auto block-type path ([`Mp3Encoder::enable_auto_block_type`]).
fn default_transition_gc(block_type: BlockType) -> GranuleChannel {
    debug_assert!(matches!(block_type, BlockType::Start | BlockType::End));
    GranuleChannel {
        part2_3_length: 0,
        big_values: 0,
        global_gain: 0,
        scalefac_compress: 0,
        window_switching_flag: true,
        block_type,
        mixed_block_flag: false,
        table_select: [0; 3],
        subblock_gain: [0; 3],
        // §2.4.2.7 window-switched defaults. These fields are not
        // transmitted for window-switched granules, but the in-memory
        // values must match what `parse_side_info` reconstructs
        // (region0_count = 7, region1_count = 63): the Huffman
        // emitter's region mapping reads them, so a mismatched
        // sentinel here silently assigns codebooks to line ranges no
        // decoder will use (the r405 Start/End region fix).
        region0_count: 7,
        region1_count: 63,
        preflag: false,
        scalefac_scale: false,
        count1table_select: false,
    }
}

/// Side-info byte length for the active version + channel count:
/// ISO/IEC 11172-3 §2.4.1.7 (17 / 32 bytes) for MPEG-1, ISO/IEC
/// 13818-3 §2.4.1.7 (9 / 17 bytes) for the LSF versions. Delegates to
/// the demuxer's shared lookup.
fn side_info_byte_len(version: MpegVersion, nch: usize) -> usize {
    crate::demuxer::side_info_len(version, nch as u8)
}

/// Resolve an MPEG-1 Layer III bitrate (kbit/s) to its position
/// (1..=14) on the §2.4.2.3 ladder. Returns `None` for an off-ladder
/// value (free format `0`, forbidden `15`, or any kbps not in
/// [`MPEG1_L3_BITRATE_LADDER_KBPS`]).
fn ladder_index(version: MpegVersion, kbps: u32) -> Option<u8> {
    bitrate_ladder_for(version)
        .iter()
        .position(|&v| v == kbps)
        .map(|i| (i + 1) as u8)
}

/// The Layer III bitrate ladder for `version`: the MPEG-1 §2.4.2.3
/// table or the 13818-3 LSF table (which MPEG-2.5 inherits).
fn bitrate_ladder_for(version: MpegVersion) -> &'static [u32; 14] {
    if version.is_lsf() {
        &LSF_L3_BITRATE_LADDER_KBPS
    } else {
        &MPEG1_L3_BITRATE_LADDER_KBPS
    }
}

/// Compute the main-data slot byte capacity for the §2.4.2.3 ladder
/// `bitrate_index` (1..=14) at the given `sample_rate_hz`, after
/// subtracting the 4-byte header and `si_bytes` side-info bytes (no
/// CRC). When `padded` is true, the slot includes the one-byte padding
/// slot the per-frame `padding` bit absorbs. The version (and with it
/// the bitrate ladder + the 144-vs-72 `slots_per_frame` constant of
/// ISO/IEC 13818-3 §2.4.3.2) follows from the sample rate — the three
/// rate families are disjoint.
fn ladder_slot_capacity(
    sample_rate_hz: u32,
    bitrate_index: u8,
    si_bytes: usize,
    padded: bool,
) -> usize {
    let lsf = sample_rate_hz < 32_000;
    let ladder = if lsf {
        &LSF_L3_BITRATE_LADDER_KBPS
    } else {
        &MPEG1_L3_BITRATE_LADDER_KBPS
    };
    let kbps = ladder[(bitrate_index - 1) as usize];
    let bps = u64::from(kbps) * 1000;
    let sr = u64::from(sample_rate_hz);
    let coeff: u64 = if lsf { 72 } else { 144 };
    let unpadded = (coeff * bps / sr) as usize;
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
    si_bytes: usize,
    crc_bytes: usize,
) -> Option<(u32, u8)> {
    let lsf = sample_rate_hz < 32_000;
    for idx in cfg.min_index..=cfg.max_index {
        // Try unpadded first, then padded — the per-frame padding bit
        // adds one byte to the slot at the same `bitrate_index`. For
        // VBR with min_kbps == max_kbps this preserves the CBR
        // Bresenham padding behaviour roughly (padding is enabled only
        // when needed).
        let cap_padded =
            ladder_slot_capacity(sample_rate_hz, idx, si_bytes, true).saturating_sub(crc_bytes);
        if cap_padded >= need {
            let ladder = if lsf {
                &LSF_L3_BITRATE_LADDER_KBPS
            } else {
                &MPEG1_L3_BITRATE_LADDER_KBPS
            };
            let kbps = ladder[(idx - 1) as usize];
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
    inverse_alias_reduce_boundaries(xr, 32)
}

/// Inverse of the decoder's single-butterfly **mixed-block** alias
/// reduction: only the `sb == 1` boundary internal to the two-subband
/// long region is long/long, so only that butterfly group is inverted
/// (`docs/audio/mp3/mp3-alias-reduction-clarification.md`; r405
/// observer-trace). Mirrors `alias::alias_reduce` for
/// `mixed_block_flag` granules.
fn inverse_alias_reduce_mixed(xr: &[f32; NUM_LINES]) -> [f32; NUM_LINES] {
    inverse_alias_reduce_boundaries(xr, 2)
}

fn inverse_alias_reduce_boundaries(xr: &[f32; NUM_LINES], sb_end: usize) -> [f32; NUM_LINES] {
    use crate::alias::{alias_ca, alias_cs};
    let cs = alias_cs();
    let ca = alias_ca();
    let mut out = *xr;
    // Apply the inverse butterfly across each covered subband
    // boundary. Source inputs come from `out` updated
    // in place: each butterfly's `(lo, hi)` is a fresh pair so we can
    // read-then-write within the same loop iteration without cross-
    // contamination across butterflies of the same boundary.
    for sb in 1..sb_end {
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
    #[test]
    fn mixed_blocks_rejected_at_8khz() {
        // The 8 kHz Fraunhofer short table has no boundary at the
        // 36-line long/short split, so mixed emission is refused
        // (r405); pure short and plain auto stay available.
        let mut enc = super::Mp3Encoder::new(32, 8_000, crate::ChannelMode::SingleChannel).unwrap();
        assert!(matches!(
            enc.force_mixed_blocks_for_testing(true),
            Err(super::StreamEncodeError::MixedBlocks8kUnsupported)
        ));
        assert!(matches!(
            enc.enable_auto_block_type_with_mixed(2.0, 6.0),
            Err(super::StreamEncodeError::MixedBlocks8kUnsupported)
        ));
        assert!(enc.force_short_blocks_for_testing(true).is_ok());
        let mut enc2 =
            super::Mp3Encoder::new(32, 8_000, crate::ChannelMode::SingleChannel).unwrap();
        assert!(enc2.enable_auto_block_type(2.0).is_ok());
        // Mixed stays available at the other MPEG-2.5 rates.
        let mut enc3 =
            super::Mp3Encoder::new(40, 12_000, crate::ChannelMode::SingleChannel).unwrap();
        assert!(enc3.force_mixed_blocks_for_testing(true).is_ok());
    }

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
    fn mpeg25_8khz_encode_produces_valid_stream() {
        // End-to-end reachability of the MPEG-2.5 8 kHz scalefactor-band
        // tables (docs/audio/mp3/mpeg2.5-scalefactor-bands.md, #147/#151):
        // a full `Mp3Encoder` run at 8 kHz exercises quantize / inner-loop
        // / Huffman / main-data assembly against the new 8 kHz band
        // layout (LONG/SHORT_STARTS_MPEG25_8 via long/short_band_starts).
        // MPEG-2.5 is single-granule (576 samples/frame/channel).
        let mut enc = Mp3Encoder::new(32, 8_000, ChannelMode::SingleChannel).unwrap();
        assert_eq!(enc.version, MpegVersion::Mpeg25);
        let mut pcm = vec![0i16; SAMPLES_PER_GRANULE * 6];
        for (n, v) in pcm.iter_mut().enumerate() {
            // 800 Hz tone at 8 kHz (well below the 4 kHz Nyquist).
            let t = n as f32 / 8_000.0;
            *v = (8000.0 * (2.0 * std::f32::consts::PI * 800.0 * t).sin()) as i16;
        }
        enc.push_samples(&pcm).unwrap();
        let mut out: Vec<u8> = Vec::new();
        let bytes = enc.finish(&mut out).unwrap();
        assert!(bytes > 0, "8 kHz encode produced no bytes");
        assert_eq!(out.len(), bytes);
        // Every emitted frame must carry a valid sync word + MPEG-2.5 id.
        assert!(out.len() >= 4);
        assert_eq!(out[0], 0xFF, "frame sync byte 0");
        assert_eq!(out[1] & 0xE0, 0xE0, "frame sync byte 1 top 3 bits");
        // version field (bits 4..3 of byte 1) == 0b00 => MPEG-2.5.
        assert_eq!((out[1] >> 3) & 0b11, 0b00, "MPEG-2.5 version id");
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
            let si_bytes = side_info_byte_len(hdr.version, nch as usize);
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

    // =====================================================================
    // §2.4.3.4.9.3 intensity-stereo encode — position derivation +
    // constructor state + toggle interlocks. End-to-end wire / decode
    // coverage lives in `tests/joint_stereo_intensity_roundtrip.rs`.
    // =====================================================================

    /// Annex G.2 c) position grid: spot values of
    /// `NINT((12/π)·arctan(√(E_L/E_R)))` across the energy-ratio range,
    /// plus the degenerate-energy conventions.
    #[test]
    fn intensity_position_derivation_grid() {
        // Equal band energies → amplitude ratio 1 → arctan = π/4 →
        // position 3 (the center of the 0..=6 grid; tan(3π/12) = 1
        // reproduces the ratio exactly on the decode side).
        assert_eq!(derive_intensity_position(1.0, 1.0), 3);
        // Energy ratio 16 → amplitude ratio 4 →
        // (12/π)·arctan(4) ≈ 5.06 → 5.
        assert_eq!(derive_intensity_position(16.0, 1.0), 5);
        // The mirrored band leans right: amplitude ratio 1/4 →
        // (12/π)·arctan(0.25) ≈ 0.94 → 1.
        assert_eq!(derive_intensity_position(1.0, 16.0), 1);
        // Hard-left (zero right-channel energy) → the E_R → 0 limit 6.
        assert_eq!(derive_intensity_position(1.0, 0.0), 6);
        // Hard-right (zero left-channel energy) → arctan(0) = 0 → 0.
        assert_eq!(derive_intensity_position(0.0, 1.0), 0);
        // Fully-silent band → same convention as the E_R → 0 limit
        // (any position decodes the zero magnitude to zero).
        assert_eq!(derive_intensity_position(0.0, 0.0), 6);
        // Extreme but finite ratios stay on the legal grid (7 is the
        // §2.4.3.4.9.3 illegal-position marker and is never derived).
        assert_eq!(derive_intensity_position(1.0e30, 1.0e-30), 6);
        for &(le, re) in &[
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (3.5, 0.7),
            (1.0e-20, 1.0e-19),
            (1.0e30, 2.0),
        ] {
            assert!(
                derive_intensity_position(le, re) <= 6,
                "({le}, {re}) produced an off-grid position"
            );
        }
    }

    /// The derived position is monotone non-decreasing in the
    /// left/right energy ratio (more left-leaning bands take larger
    /// positions, matching the §2.4.3.4.9.3 tan() grid orientation).
    #[test]
    fn intensity_position_monotone_in_energy_ratio() {
        let mut prev = 0u8;
        for k in -40..=40 {
            let ratio = 10f64.powf(f64::from(k) / 4.0);
            let pos = derive_intensity_position(ratio, 1.0);
            assert!(
                pos >= prev,
                "position dropped from {prev} to {pos} at energy ratio {ratio}"
            );
            prev = pos;
        }
        assert_eq!(prev, 6, "ratio sweep should end hard-left");
    }

    /// The LSF (§2.4.3.2) position derivation lands on the §2.4.3.2
    /// power-law `i0 = 2^(-1/4)` ladder, not the MPEG-1 `tan` grid. The
    /// decoded amplitude ratio `L'/R' = kl/kr` for `p = 0..=6` is
    /// `{1, i0, 1/i0, i0², 1/i0², i0³, 1/i0³}`. Each test ratio is set
    /// exactly on a grid point so the closest-in-log-space pick is
    /// unambiguous.
    #[test]
    fn intensity_position_derivation_grid_lsf() {
        // i0 = 2^(-1/4) ≈ 0.840896. The energy ratio E_L/E_R is the
        // square of the amplitude ratio.
        let i0 = 2f64.powf(-0.25);
        // p, amplitude-ratio target (L'/R').
        let cases: [(u8, f64); 7] = [
            (0, 1.0),
            (1, i0),       // odd: kl = i0
            (2, 1.0 / i0), // even: kr = i0
            (3, i0 * i0),  // odd: kl = i0²
            (4, 1.0 / (i0 * i0)),
            (5, i0 * i0 * i0), // odd: kl = i0³
            (6, 1.0 / (i0 * i0 * i0)),
        ];
        for (p, amp_ratio) in cases {
            let e_l = amp_ratio * amp_ratio;
            let e_r = 1.0;
            assert_eq!(
                derive_intensity_position_lsf(e_l, e_r),
                p,
                "amplitude ratio {amp_ratio} should map to LSF position {p}"
            );
        }
        // Degenerate energies: silent right ⇒ hard-left p=6; silent
        // left ⇒ hard-right p=5 (the smallest grid ratio i0³).
        assert_eq!(derive_intensity_position_lsf(1.0, 0.0), 6);
        assert_eq!(derive_intensity_position_lsf(0.0, 1.0), 5);
        // Every output is a valid (non-marker) position.
        for k in -40..=40 {
            let e_l = 10f64.powf(f64::from(k) / 4.0);
            let pos = derive_intensity_position_lsf(e_l, 1.0);
            assert!(pos <= 6, "LSF position {pos} out of range");
        }
    }

    /// Constructor state: the three intensity constructors arm the
    /// coupling, carry the §2.4.2.3 mode/mode_extension template bits,
    /// and validate the start band.
    #[test]
    fn intensity_constructors_arm_state_and_template_bits() {
        let enc = Mp3Encoder::new_joint_stereo_is(192, 44_100, 8).expect("is ctor");
        assert!(enc.intensity_stereo_enabled());
        assert_eq!(enc.intensity_start_sfb(), Some(8));
        assert!(!enc.ms_stereo_enabled());
        assert!(matches!(enc.header_template.mode, ChannelMode::JointStereo));
        assert_eq!(enc.header_template.mode_extension.raw, 0b01);
        assert!(enc.header_template.mode_extension.intensity_stereo);
        assert!(!enc.header_template.mode_extension.ms_stereo);

        let enc = Mp3Encoder::new_joint_stereo_ms_is(192, 44_100, 11).expect("ms+is ctor");
        assert!(enc.intensity_stereo_enabled());
        assert_eq!(enc.intensity_start_sfb(), Some(11));
        assert!(enc.ms_stereo_enabled());
        assert_eq!(enc.header_template.mode_extension.raw, 0b11);
        assert!(enc.header_template.mode_extension.intensity_stereo);
        assert!(enc.header_template.mode_extension.ms_stereo);

        let enc = Mp3Encoder::new_joint_stereo_auto_is(192, 44_100, 14).expect("auto+is ctor");
        assert!(enc.intensity_stereo_enabled());
        assert_eq!(enc.intensity_start_sfb(), Some(14));
        assert!(!enc.ms_stereo_enabled());
        assert_eq!(enc.ms_auto_threshold(), Some(0.5));
        assert_eq!(enc.header_template.mode_extension.raw, 0b01);

        // Plain constructors stay disarmed.
        let enc = Mp3Encoder::new_joint_stereo_ms(192, 44_100).expect("ms ctor");
        assert!(!enc.intensity_stereo_enabled());
        assert_eq!(enc.intensity_start_sfb(), None);
    }

    /// The adaptive-bound constructor arms the intensity coupling, sets
    /// the default side-energy threshold, carries the intensity-only
    /// `mode_extension` template bits, and treats its `start` argument
    /// as the coupling *floor*. `with_intensity_auto_threshold` clamps
    /// to `[0, 1]` and is a no-op on a non-adaptive encoder.
    #[test]
    fn intensity_auto_adaptive_constructor_state() {
        let enc =
            Mp3Encoder::new_joint_stereo_auto_is_adaptive(192, 44_100, 7).expect("adaptive ctor");
        assert!(enc.intensity_stereo_enabled());
        assert_eq!(enc.intensity_start_sfb(), Some(7));
        assert_eq!(enc.intensity_auto_threshold(), Some(0.25));
        assert!(!enc.ms_stereo_enabled());
        assert!(matches!(enc.header_template.mode, ChannelMode::JointStereo));
        assert_eq!(enc.header_template.mode_extension.raw, 0b01);
        assert!(enc.header_template.mode_extension.intensity_stereo);
        assert!(!enc.header_template.mode_extension.ms_stereo);

        // Same `1..=20` floor validation as the fixed-bound ctors.
        assert!(matches!(
            Mp3Encoder::new_joint_stereo_auto_is_adaptive(192, 44_100, 0),
            Err(StreamEncodeError::InvalidIntensityStartSfb { start_sfb: 0 })
        ));
        assert!(matches!(
            Mp3Encoder::new_joint_stereo_auto_is_adaptive(192, 44_100, 21),
            Err(StreamEncodeError::InvalidIntensityStartSfb { start_sfb: 21 })
        ));

        // Threshold override clamps; no-op on a fixed-bound encoder.
        let enc = enc.with_intensity_auto_threshold(2.0);
        assert_eq!(enc.intensity_auto_threshold(), Some(1.0));
        let enc = enc.with_intensity_auto_threshold(-1.0);
        assert_eq!(enc.intensity_auto_threshold(), Some(0.0));
        let fixed = Mp3Encoder::new_joint_stereo_is(192, 44_100, 8)
            .expect("fixed ctor")
            .with_intensity_auto_threshold(0.1);
        assert_eq!(fixed.intensity_auto_threshold(), None);
    }

    /// `choose_intensity_bound` returns the lowest band whose contiguous
    /// high tail all has side-energy fraction `E_S/(E_L+E_R) <= t`,
    /// honours the floor, and returns 21 (couple nothing) when not even
    /// the top band qualifies.
    #[test]
    fn choose_intensity_bound_picks_low_stereo_tail() {
        let starts = long_band_starts_for(44_100);
        let floor = 7usize;
        let t = 0.25f64;

        // (a) Everything from the floor up is near-mono (L == R ⇒ side
        // energy 0): the whole tail qualifies, bound == floor.
        let mut l = [0.0f32; NUM_LINES];
        let mut r = [0.0f32; NUM_LINES];
        for i in starts[floor]..NUM_LINES {
            l[i] = 1.0;
            r[i] = 1.0;
        }
        assert_eq!(choose_intensity_bound(&l, &r, starts, floor, t), floor);

        // (b) A single mid band (say band 14) carries strong stereo
        // information (L = -R ⇒ side fraction 1.0 > t): it and
        // everything below it stay independent, so the bound is the band
        // just above it.
        let mut l = [0.0f32; NUM_LINES];
        let mut r = [0.0f32; NUM_LINES];
        for i in starts[floor]..NUM_LINES {
            l[i] = 1.0;
            r[i] = 1.0;
        }
        for i in starts[14]..starts[15] {
            l[i] = 1.0;
            r[i] = -1.0;
        }
        assert_eq!(choose_intensity_bound(&l, &r, starts, floor, t), 15);

        // (c) The top band itself carries stereo information ⇒ no tail
        // qualifies ⇒ couple nothing (bound 21).
        let mut l = [0.0f32; NUM_LINES];
        let mut r = [0.0f32; NUM_LINES];
        for i in starts[floor]..NUM_LINES {
            l[i] = 1.0;
            r[i] = 1.0;
        }
        for i in starts[20]..starts[21] {
            l[i] = 1.0;
            r[i] = -1.0;
        }
        assert_eq!(choose_intensity_bound(&l, &r, starts, floor, t), 21);

        // (d) A fully-silent tail (no energy at all) carries no stereo
        // information and qualifies down to the floor.
        let l = [0.0f32; NUM_LINES];
        let r = [0.0f32; NUM_LINES];
        assert_eq!(choose_intensity_bound(&l, &r, starts, floor, t), floor);

        // (e) Bands below the floor never participate even when the
        // signal there is pure mono.
        let mut l = [0.0f32; NUM_LINES];
        let mut r = [0.0f32; NUM_LINES];
        for i in 0..NUM_LINES {
            l[i] = 1.0;
            r[i] = 1.0;
        }
        assert_eq!(choose_intensity_bound(&l, &r, starts, floor, t), floor);
    }

    /// Start-band validation: at least one normal band below the bound
    /// and one intensity band at or above it (`1..=20`).
    #[test]
    fn intensity_start_sfb_out_of_range_rejected() {
        for bad in [0usize, 21, 22, 100] {
            for ctor in [
                Mp3Encoder::new_joint_stereo_is,
                Mp3Encoder::new_joint_stereo_ms_is,
                Mp3Encoder::new_joint_stereo_auto_is,
            ] {
                let err = ctor(192, 44_100, bad).expect_err("out-of-range start band");
                assert!(
                    matches!(
                        err,
                        StreamEncodeError::InvalidIntensityStartSfb { start_sfb } if start_sfb == bad
                    ),
                    "expected InvalidIntensityStartSfb for {bad}, got {err:?}"
                );
            }
        }
        // Boundary values are accepted.
        assert!(Mp3Encoder::new_joint_stereo_is(192, 44_100, 1).is_ok());
        assert!(Mp3Encoder::new_joint_stereo_is(192, 44_100, 20).is_ok());
    }

    /// Block-type toggle acceptance while intensity coupling is armed.
    /// r303 wired the §2.4.3.4.9.3 *per-window* short-block intensity
    /// bound, so force-short on the intensity-only path
    /// ([`Mp3Encoder::new_joint_stereo_is`]) now succeeds. r305 wired the
    /// §2.4.3.4.9.2 per-window MS rotation below the short bound, so
    /// force-short on the unconditional MS + intensity path
    /// ([`Mp3Encoder::new_joint_stereo_ms_is`]) now succeeds too. The
    /// r306 wired the §2.4.3.4.9.2 *auto*-MS picker over the per-window
    /// short region, so force-short on the auto-MS + intensity path
    /// ([`Mp3Encoder::new_joint_stereo_auto_is`]) now succeeds too. Only
    /// the mixed and auto-scheduled short paths (whose per-window bound
    /// geometry isn't wired yet) stay rejected.
    #[test]
    fn intensity_rejects_block_type_toggles() {
        // Force-short on the intensity-only path is accepted (r303).
        let mut enc = Mp3Encoder::new_joint_stereo_is(192, 44_100, 8).unwrap();
        assert!(enc.force_short_blocks_for_testing(true).is_ok());
        assert!(enc.force_short_blocks_enabled());

        // Force-short on the unconditional MS + intensity path is
        // accepted (r305).
        let mut ms_enc = Mp3Encoder::new_joint_stereo_ms_is(192, 44_100, 8).unwrap();
        assert!(ms_enc.force_short_blocks_for_testing(true).is_ok());
        assert!(ms_enc.force_short_blocks_enabled());

        // Force-short on the auto-MS + intensity path is accepted (r306):
        // the side-energy picker now measures the per-window short MS
        // region.
        let mut auto_enc = Mp3Encoder::new_joint_stereo_auto_is(192, 44_100, 8).unwrap();
        assert!(auto_enc.force_short_blocks_for_testing(true).is_ok());
        assert!(auto_enc.force_short_blocks_enabled());

        // Signal-driven auto block-type under MS-joint intensity is
        // accepted (r307): MS-agreement mirrors one scheduler emission
        // across both channels, so the per-granule short/long intensity
        // coupling has a channel-consistent geometry.
        let mut auto_ms = Mp3Encoder::new_joint_stereo_ms_is(192, 44_100, 8).unwrap();
        assert!(auto_ms.enable_auto_block_type(10.0).is_ok());
        assert!(auto_ms.auto_block_type_enabled());
        let mut auto_pick = Mp3Encoder::new_joint_stereo_auto_is(192, 44_100, 8).unwrap();
        assert!(auto_pick.enable_auto_block_type(10.0).is_ok());

        // The *intensity-only* signal-driven auto path is now accepted
        // (r308): arming intensity coupling forces the §2.4.3.4.9
        // channel-agreement OR-fold (channel-0 emission mirrored across
        // both channels), so L/R block types stay consistent even
        // without MS.
        let mut auto_is_only = Mp3Encoder::new_joint_stereo_is(192, 44_100, 8).unwrap();
        assert!(auto_is_only.enable_auto_block_type(10.0).is_ok());
        assert!(auto_is_only.auto_block_type_enabled());

        // Force-mixed + intensity on the *intensity-only* (non-MS) path
        // is now accepted (r311): the §2.4.3.4.10.3 carve-out couples the
        // long lowest 2 subbands on the long walk and the short rest per
        // window.
        let mut enc = Mp3Encoder::new_joint_stereo_is(192, 44_100, 8).unwrap();
        assert!(enc.force_mixed_blocks_for_testing(true).is_ok());
        assert!(enc.force_mixed_blocks_enabled());

        // Force-mixed + intensity under MS-joint stereo stays rejected
        // (the below-bound MS rotation over the mixed split line set is a
        // follow-up).
        let mut ms_enc = Mp3Encoder::new_joint_stereo_ms_is(192, 44_100, 8).unwrap();
        assert!(matches!(
            ms_enc.force_mixed_blocks_for_testing(true),
            Err(StreamEncodeError::IntensityShortBlocksUnsupported)
        ));

        // The mixed-promotion *auto* variant remains unsupported (the
        // signal-driven mixed classifier carve-out is unwired) — on the
        // intensity-only path AND the MS+intensity path.
        let mut auto_mixed = Mp3Encoder::new_joint_stereo_is(192, 44_100, 8).unwrap();
        assert!(matches!(
            auto_mixed.enable_auto_block_type_with_mixed(10.0, 4.0),
            Err(StreamEncodeError::IntensityShortBlocksUnsupported)
        ));
        let mut ms_mixed = Mp3Encoder::new_joint_stereo_ms_is(192, 44_100, 8).unwrap();
        assert!(matches!(
            ms_mixed.enable_auto_block_type_with_mixed(10.0, 4.0),
            Err(StreamEncodeError::IntensityShortBlocksUnsupported)
        ));

        // Disabling a toggle remains a no-op success.
        assert!(enc.force_short_blocks_for_testing(false).is_ok());
        assert!(enc.force_mixed_blocks_for_testing(false).is_ok());
        assert!(!enc.force_short_blocks_enabled());
        assert!(!enc.auto_block_type_enabled());
    }

    // =====================================================================
    // `new_with_threshold_in_quiet` — outer-loop + Annex D LTq vector
    // bundled construction. Verifies the one-shot constructor sets the
    // same state the two-step `new_with_outer_loop` + `set_per_band_xmin`
    // recipe produces.
    // =====================================================================

    #[test]
    fn new_with_threshold_in_quiet_enables_outer_loop_and_per_band() {
        // Mono — straight-line case. The constructor must arm both the
        // outer loop AND the per-band vector; either left disarmed is a
        // regression.
        let enc = Mp3Encoder::new_with_threshold_in_quiet(128, 44_100, ChannelMode::SingleChannel)
            .expect("mono ctor");
        assert!(
            enc.outer_loop_threshold.is_some(),
            "new_with_threshold_in_quiet must arm the outer loop",
        );
        assert!(
            enc.per_band_xmin_enabled(),
            "new_with_threshold_in_quiet must arm the per-band vector",
        );
        // The carried uniform scalar is the documented
        // DEFAULT_OUTER_LOOP_THRESHOLD so subsequent re-overrides of
        // the per-band vector see the same uniform fallback.
        assert_eq!(enc.outer_loop_threshold, Some(DEFAULT_OUTER_LOOP_THRESHOLD));
    }

    #[test]
    fn new_with_threshold_in_quiet_carries_long_band_bowl_shape() {
        // Pull the installed per-band vector back out and check the
        // long-block bowl shape Annex D Table D.1 prescribes — the
        // mid-spectrum minimum must sit strictly below the bass /
        // treble extremes. This proves the constructor wired the
        // `threshold_in_quiet` derivation (and not the uniform fill).
        let enc = Mp3Encoder::new_with_threshold_in_quiet(128, 44_100, ChannelMode::SingleChannel)
            .expect("mono ctor");
        let xmin = enc
            .per_band_xmin
            .as_ref()
            .expect("per-band vector installed");
        // Bass (sfb 0) > minimum, treble (sfb 20) > minimum, with the
        // minimum sitting in the mid-spectrum (not at either edge).
        let (min_sfb, &min_v) = xmin
            .long
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        assert!(
            xmin.long[0] > min_v,
            "bass sfb 0 ({}) must be > minimum sfb {min_sfb} ({min_v})",
            xmin.long[0],
        );
        let last = xmin.long.len() - 1;
        assert!(
            xmin.long[last] > min_v,
            "treble sfb {last} ({}) must be > minimum sfb {min_sfb} ({min_v})",
            xmin.long[last],
        );
        assert!(
            (1..last).contains(&min_sfb),
            "min sfb {min_sfb} must sit in the mid-spectrum (not at either edge)",
        );
    }

    #[test]
    fn new_with_threshold_in_quiet_applies_step3_offset_per_channel_bitrate() {
        // The §D.1 Step 3 `−12 dB` offset switches on
        // `bitrate_kbps_per_channel >= 96`. Mono at 128 kbit/s (per-ch
        // 128 ≥ 96) triggers; mono at 64 kbit/s (per-ch 64 < 96) does
        // not. The per-band entries at the trigger bitrate must be
        // strictly lower (= more aggressive amplification target).
        let high = Mp3Encoder::new_with_threshold_in_quiet(128, 44_100, ChannelMode::SingleChannel)
            .expect("high-br mono");
        let low = Mp3Encoder::new_with_threshold_in_quiet(64, 44_100, ChannelMode::SingleChannel)
            .expect("low-br mono");
        let xh = high.per_band_xmin.as_ref().expect("hi xmin");
        let xl = low.per_band_xmin.as_ref().expect("lo xmin");
        for sfb in 0..xh.long.len() {
            assert!(
                xh.long[sfb] < xl.long[sfb],
                "sfb {sfb}: high-br {} must be < low-br {} (Step 3 offset)",
                xh.long[sfb],
                xl.long[sfb],
            );
            let ratio = xl.long[sfb] / xh.long[sfb];
            // 10^(12/10) ≈ 15.85.
            assert!(
                (ratio - 10.0_f64.powf(12.0 / 10.0)).abs() < 1.0e-6,
                "sfb {sfb}: ratio {ratio} must equal 10^1.2",
            );
        }
    }

    #[test]
    fn new_with_threshold_in_quiet_stereo_uses_per_channel_bitrate_for_step3() {
        // Stereo at 192 kbit/s has 96 kbit/s per channel — exactly the
        // §D.1 Step 3 trigger. Stereo at 128 kbit/s has 64 kbit/s per
        // channel — below the trigger. The constructor must compute
        // the offset on the per-channel bitrate, NOT the aggregate.
        let trigger = Mp3Encoder::new_with_threshold_in_quiet(192, 44_100, ChannelMode::Stereo)
            .expect("stereo 192");
        let below = Mp3Encoder::new_with_threshold_in_quiet(128, 44_100, ChannelMode::Stereo)
            .expect("stereo 128");
        let xt = trigger.per_band_xmin.as_ref().unwrap();
        let xb = below.per_band_xmin.as_ref().unwrap();
        // Identical 10^1.2 ratio between trigger (offset applied) and
        // below (offset zero) at every long band.
        for sfb in 0..xt.long.len() {
            let ratio = xb.long[sfb] / xt.long[sfb];
            assert!(
                (ratio - 10.0_f64.powf(12.0 / 10.0)).abs() < 1.0e-6,
                "stereo sfb {sfb}: ratio {ratio} must equal 10^1.2 (per-channel offset)",
            );
        }
    }

    // =====================================================================
    // `new_with_threshold_in_quiet_offset` — caller-supplied §D.1 Step 3
    // offset (r213).
    // =====================================================================

    #[test]
    fn new_with_threshold_in_quiet_offset_arms_outer_loop_and_per_band() {
        let enc = Mp3Encoder::new_with_threshold_in_quiet_offset(
            128,
            44_100,
            ChannelMode::SingleChannel,
            -6.0,
        )
        .expect("ltq offset mono");
        assert!(
            enc.outer_loop_threshold.is_some(),
            "outer-loop must be armed",
        );
        assert!(
            enc.per_band_xmin_enabled(),
            "per-band vector must be installed",
        );
    }

    #[test]
    fn new_with_threshold_in_quiet_offset_minus12_matches_spec_high_bitrate_path() {
        // `offset_db = -12.0` reproduces the §D.1 Step 3 high-bitrate
        // path — equivalent (to FP tolerance) to the per-channel
        // `bitrate_kbps_per_channel >= 96` branch of
        // `new_with_threshold_in_quiet`.
        let custom = Mp3Encoder::new_with_threshold_in_quiet_offset(
            128,
            44_100,
            ChannelMode::SingleChannel,
            -12.0,
        )
        .expect("custom");
        let spec = Mp3Encoder::new_with_threshold_in_quiet(128, 44_100, ChannelMode::SingleChannel)
            .expect("spec");
        let xc = custom.per_band_xmin.as_ref().unwrap();
        let xs = spec.per_band_xmin.as_ref().unwrap();
        for sfb in 0..xc.long.len() {
            assert!(
                (xc.long[sfb] - xs.long[sfb]).abs() < 1.0e-9,
                "long sfb {sfb}: custom {} vs spec {}",
                xc.long[sfb],
                xs.long[sfb],
            );
        }
    }

    #[test]
    fn new_with_threshold_in_quiet_offset_zero_matches_spec_low_bitrate_path() {
        let custom = Mp3Encoder::new_with_threshold_in_quiet_offset(
            128,
            44_100,
            ChannelMode::SingleChannel,
            0.0,
        )
        .expect("custom");
        // Spec low-bitrate path: per-channel bitrate < 96.
        let spec = Mp3Encoder::new_with_threshold_in_quiet(64, 44_100, ChannelMode::SingleChannel)
            .expect("spec");
        let xc = custom.per_band_xmin.as_ref().unwrap();
        let xs = spec.per_band_xmin.as_ref().unwrap();
        for sfb in 0..xc.long.len() {
            assert!(
                (xc.long[sfb] - xs.long[sfb]).abs() < 1.0e-9,
                "long sfb {sfb}: custom {} vs spec {}",
                xc.long[sfb],
                xs.long[sfb],
            );
        }
    }

    #[test]
    fn new_with_threshold_in_quiet_offset_monotone_in_offset_db() {
        // A lower (more negative) offset_db must produce strictly
        // smaller per-band xmin values (more aggressive amplification
        // target) at every long band — the offset is a uniform dB
        // translation of the bowl.
        let strict = Mp3Encoder::new_with_threshold_in_quiet_offset(
            128,
            44_100,
            ChannelMode::SingleChannel,
            -24.0,
        )
        .expect("strict");
        let loose = Mp3Encoder::new_with_threshold_in_quiet_offset(
            128,
            44_100,
            ChannelMode::SingleChannel,
            0.0,
        )
        .expect("loose");
        let xs = strict.per_band_xmin.as_ref().unwrap();
        let xl = loose.per_band_xmin.as_ref().unwrap();
        for sfb in 0..xs.long.len() {
            assert!(
                xs.long[sfb] < xl.long[sfb],
                "sfb {sfb}: strict {} should be < loose {}",
                xs.long[sfb],
                xl.long[sfb],
            );
        }
    }

    #[test]
    fn model2_xmin_requires_outer_loop() {
        // No outer loop → the per-band install is rejected.
        let mut enc = Mp3Encoder::new(128, 44_100, ChannelMode::SingleChannel).unwrap();
        let mut state = crate::psy::Model2Layer3State::new(crate::psy::AnnexDSamplingRate::Hz44100);
        let g = vec![0.0_f64; SAMPLES_PER_GRANULE];
        assert!(matches!(
            enc.set_per_band_xmin_from_model2(&mut state, &g),
            Err(StreamEncodeError::PerBandXminWithoutOuterLoop)
        ));
    }

    #[test]
    fn model2_xmin_rejects_bad_granule_and_unsupported_rate() {
        // Wrong granule length → length error (sample_rate_hz == 0).
        let mut enc =
            Mp3Encoder::new_with_outer_loop(128, 44_100, ChannelMode::SingleChannel, 1.0e6)
                .unwrap();
        let mut state = crate::psy::Model2Layer3State::new(crate::psy::AnnexDSamplingRate::Hz44100);
        assert!(matches!(
            enc.set_per_band_xmin_from_model2(&mut state, &vec![0.0_f64; 575]),
            Err(StreamEncodeError::Model2AnalysisUnsupported { sample_rate_hz: 0 })
        ));
        // An LSF rate (22.05 kHz) has no staged Model 2 tables → rejected
        // with the offending rate carried.
        let mut lsf =
            Mp3Encoder::new_with_outer_loop(64, 22_050, ChannelMode::SingleChannel, 1.0e6).unwrap();
        let mut st32 = crate::psy::Model2Layer3State::new(crate::psy::AnnexDSamplingRate::Hz32000);
        assert!(matches!(
            lsf.set_per_band_xmin_from_model2(&mut st32, &vec![0.0_f64; SAMPLES_PER_GRANULE]),
            Err(StreamEncodeError::Model2AnalysisUnsupported {
                sample_rate_hz: 22_050
            })
        ));
    }

    #[test]
    fn model2_xmin_installs_signal_dependent_threshold() {
        // A live tone granule produces a per-band threshold the outer
        // loop will consume; the install flips per_band_xmin_enabled and
        // the long vector is signal-dependent (not the flat uniform
        // fill the encoder shipped with).
        let mut enc =
            Mp3Encoder::new_with_outer_loop(128, 44_100, ChannelMode::SingleChannel, 1.0e6)
                .unwrap();
        assert!(!enc.per_band_xmin_enabled());
        let mut state = crate::psy::Model2Layer3State::new(crate::psy::AnnexDSamplingRate::Hz44100);
        // Prime the FFT history then install on a later granule so the
        // unpredictability measure has real predecessors.
        let tone = |n: usize| -> Vec<f64> {
            (0..SAMPLES_PER_GRANULE)
                .map(|i| {
                    let t = (n * SAMPLES_PER_GRANULE + i) as f64;
                    0.5 * (2.0 * core::f64::consts::PI * 1000.0 * t / 44_100.0).sin()
                })
                .collect()
        };
        for n in 0..3 {
            enc.set_per_band_xmin_from_model2(&mut state, &tone(n))
                .unwrap();
        }
        assert!(enc.per_band_xmin_enabled());
        let x = enc.per_band_xmin.as_ref().unwrap();
        assert!(x.long.iter().all(|&v| v.is_finite() && v > 0.0));
        // Signal-dependent: not every band carries the identical value a
        // uniform fill would.
        let first = x.long[0];
        assert!(
            x.long
                .iter()
                .any(|&v| (v - first).abs() > 1e-6 * first.max(1.0)),
            "expected a spectrally-shaped threshold, got a flat vector"
        );
    }

    #[test]
    fn auto_model2_requires_outer_loop() {
        // Without the outer loop there is nowhere to feed the Model 2
        // threshold → the enable call is rejected.
        let mut enc = Mp3Encoder::new(128, 44_100, ChannelMode::SingleChannel).unwrap();
        assert!(!enc.model2_psychoacoustics_enabled());
        assert!(matches!(
            enc.enable_model2_psychoacoustics(),
            Err(StreamEncodeError::PerBandXminWithoutOuterLoop)
        ));
        assert!(!enc.model2_psychoacoustics_enabled());
    }

    #[test]
    fn auto_model2_rejects_unsupported_rate() {
        // An LSF rate (22.05 kHz) has no staged Annex D Model 2
        // calculation-partition tables → rejected with the offending
        // rate carried, leaving the mode disarmed.
        let mut lsf =
            Mp3Encoder::new_with_outer_loop(64, 22_050, ChannelMode::SingleChannel, 1.0e6).unwrap();
        assert!(matches!(
            lsf.enable_model2_psychoacoustics(),
            Err(StreamEncodeError::Model2AnalysisUnsupported {
                sample_rate_hz: 22_050
            })
        ));
        assert!(!lsf.model2_psychoacoustics_enabled());
    }

    #[test]
    fn auto_model2_and_static_xmin_are_mutually_exclusive() {
        // Arming the automatic mode clears any static per-band vector,
        // and installing a static vector turns the automatic mode off.
        let mut enc =
            Mp3Encoder::new_with_outer_loop(128, 44_100, ChannelMode::SingleChannel, 1.0e6)
                .unwrap();
        enc.set_per_band_xmin(crate::psy::XminThresholds::uniform(1.0e6))
            .unwrap();
        assert!(enc.per_band_xmin_enabled());
        enc.enable_model2_psychoacoustics().unwrap();
        assert!(enc.model2_psychoacoustics_enabled());
        // The static vector was cleared (the per-granule analysis will
        // repopulate it inside the encode loop).
        assert!(!enc.per_band_xmin_enabled());
        // Re-installing a static vector disarms the automatic mode.
        enc.set_per_band_xmin(crate::psy::XminThresholds::uniform(1.0e6))
            .unwrap();
        assert!(enc.per_band_xmin_enabled());
        assert!(!enc.model2_psychoacoustics_enabled());
    }

    #[test]
    fn auto_model2_end_to_end_encodes_decodable_frames() {
        // A live tone pushed through an encoder with the automatic
        // per-granule Model 2 mode armed produces a well-formed MP3
        // stream: every frame parses, and the per-granule analysis
        // leaves a signal-dependent per-band threshold installed
        // (the last granule's `xmin`) after the run.
        use crate::frame::{parse_header, FrameWalker};
        let mut enc =
            Mp3Encoder::new_with_outer_loop(128, 44_100, ChannelMode::SingleChannel, 1.0e6)
                .unwrap();
        enc.enable_model2_psychoacoustics().unwrap();
        // Four frames of a 1 kHz tone (eight granules of FFT history).
        let total = SAMPLES_PER_FRAME_MPEG1 * 4;
        let pcm: Vec<i16> = (0..total)
            .map(|i| {
                let t = i as f64 / 44_100.0;
                (8000.0 * (2.0 * core::f64::consts::PI * 1000.0 * t).sin()) as i16
            })
            .collect();
        enc.push_samples(&pcm).unwrap();
        // Capture the per-band-enabled flag BEFORE `finish` consumes
        // the encoder: the per-granule analysis installs a real
        // threshold during the encode loop above.
        assert!(
            enc.per_band_xmin_enabled(),
            "automatic Model 2 mode should leave a per-granule xmin installed after encoding"
        );
        let mut out: Vec<u8> = Vec::new();
        let bytes = enc.finish(&mut out).unwrap();
        assert!(bytes > 0);
        assert_eq!(out.len(), bytes);
        // Every emitted frame parses with a valid header.
        let frames: Vec<_> = FrameWalker::new(&out).collect();
        assert!(
            frames.len() >= 4,
            "expected >= 4 frames, got {}",
            frames.len()
        );
        for f in &frames {
            let hdr = parse_header(&f.data[..4]).expect("valid header");
            assert_eq!(hdr.sample_rate_hz, 44_100);
        }
    }

    #[test]
    fn auto_model2_threshold_is_spectrally_shaped_not_flat() {
        // The automatic mode installs a spectrally-shaped (non-flat)
        // threshold derived from the actual signal — the whole point
        // of running Model 2 over the uniform-bowl path. Drive enough
        // granules to prime the FFT history, then inspect the
        // installed vector.
        let mut enc =
            Mp3Encoder::new_with_outer_loop(128, 44_100, ChannelMode::SingleChannel, 1.0e6)
                .unwrap();
        enc.enable_model2_psychoacoustics().unwrap();
        let total = SAMPLES_PER_FRAME_MPEG1 * 3;
        let pcm: Vec<i16> = (0..total)
            .map(|i| {
                let t = i as f64 / 44_100.0;
                // Two tones so the threshold has structure across bands.
                let s = 0.4 * (2.0 * core::f64::consts::PI * 1000.0 * t).sin()
                    + 0.3 * (2.0 * core::f64::consts::PI * 6000.0 * t).sin();
                (8000.0 * s) as i16
            })
            .collect();
        enc.push_samples(&pcm).unwrap();
        let x = enc
            .per_band_xmin
            .as_ref()
            .expect("a per-granule threshold should be installed");
        assert!(x.long.iter().all(|&v| v.is_finite() && v > 0.0));
        let first = x.long[0];
        assert!(
            x.long
                .iter()
                .any(|&v| (v - first).abs() > 1e-6 * first.max(1.0)),
            "expected a spectrally-shaped threshold, got a flat vector"
        );
    }

    #[test]
    fn last_model2_window_switch_none_before_any_frame() {
        // The accessor reports `None` for every cell until a frame has
        // been encoded under the armed mode — and also when the mode
        // was never armed at all.
        let off = Mp3Encoder::new(128, 44_100, ChannelMode::SingleChannel).unwrap();
        assert_eq!(off.last_model2_window_switch(0, 0), None);

        let mut enc =
            Mp3Encoder::new_with_outer_loop(128, 44_100, ChannelMode::SingleChannel, 1.0e6)
                .unwrap();
        enc.enable_model2_psychoacoustics().unwrap();
        // Armed, but nothing pushed yet.
        assert_eq!(enc.last_model2_window_switch(0, 0), None);
    }

    #[test]
    fn last_model2_window_switch_populated_after_encode() {
        // After encoding, the accessor returns the §C.1.5.3.2.1
        // window-switching decision (pe + `pe > 1800` attack) for every
        // granule of the last frame, with the flag exactly `pe > 1800`.
        // Two structurally different signals yield different captured
        // pe — confirming the value is signal-derived, not a constant.
        let rate = 44_100;
        let make = |f: f64| -> Vec<i16> {
            let total = SAMPLES_PER_FRAME_MPEG1 * 2;
            (0..total)
                .map(|i| {
                    let t = i as f64 / f64::from(rate);
                    let burst = if i % 800 < 40 { 1.0 } else { 0.2 };
                    let s = burst * (2.0 * core::f64::consts::PI * f * t).sin();
                    (9000.0 * s) as i16
                })
                .collect()
        };

        let mut enc_a =
            Mp3Encoder::new_with_outer_loop(128, rate, ChannelMode::SingleChannel, 1.0e6).unwrap();
        enc_a.enable_model2_psychoacoustics().unwrap();
        enc_a.push_samples(&make(1200.0)).unwrap();

        let mut pe_a = [0.0_f64; GRANULES];
        for (gr, slot) in pe_a.iter_mut().enumerate() {
            let sw = enc_a
                .last_model2_window_switch(gr, 0)
                .expect("both granules of the last frame should carry a decision");
            // pe is a log-domain entropy sum: finite, and may be
            // negative for a low-energy granule. The only hard
            // invariant on the flag is that it equals `pe > 1800`.
            assert!(sw.pe.is_finite(), "pe = {}", sw.pe);
            assert_eq!(
                sw.attack,
                sw.pe > 1800.0,
                "attack flag must be exactly pe > 1800"
            );
            *slot = sw.pe;
        }

        // A different tone frequency through a fresh encoder: at least
        // one granule's captured pe must differ, so the captured
        // decision genuinely depends on the spectrum.
        let mut enc_b =
            Mp3Encoder::new_with_outer_loop(128, rate, ChannelMode::SingleChannel, 1.0e6).unwrap();
        enc_b.enable_model2_psychoacoustics().unwrap();
        enc_b.push_samples(&make(6000.0)).unwrap();
        let mut differs = false;
        for (gr, &prev) in pe_a.iter().enumerate() {
            let sw = enc_b.last_model2_window_switch(gr, 0).unwrap();
            if (sw.pe - prev).abs() > 1e-6 * prev.abs().max(1.0) {
                differs = true;
            }
        }
        assert!(
            differs,
            "captured pe should differ between two spectrally distinct signals"
        );
    }

    #[test]
    fn last_model2_window_switch_rejects_out_of_range() {
        // gr beyond granules_per_frame() and ch beyond nch both yield
        // `None`, even after a populated frame.
        let mut enc =
            Mp3Encoder::new_with_outer_loop(128, 44_100, ChannelMode::SingleChannel, 1.0e6)
                .unwrap();
        enc.enable_model2_psychoacoustics().unwrap();
        enc.push_samples(&vec![0_i16; SAMPLES_PER_FRAME_MPEG1])
            .unwrap();
        // gr 0 is in range; gr == GRANULES and ch == 1 (mono) are not.
        assert!(enc.last_model2_window_switch(0, 0).is_some());
        assert_eq!(enc.last_model2_window_switch(GRANULES, 0), None);
        assert_eq!(enc.last_model2_window_switch(0, 1), None);
    }

    #[test]
    fn last_model2_window_switch_cleared_when_mode_disarmed() {
        // Installing a static per-band vector disarms the automatic
        // mode and drops the captured decision (it no longer reflects an
        // active analysis).
        let mut enc =
            Mp3Encoder::new_with_outer_loop(128, 44_100, ChannelMode::SingleChannel, 1.0e6)
                .unwrap();
        enc.enable_model2_psychoacoustics().unwrap();
        enc.push_samples(&vec![1234_i16; SAMPLES_PER_FRAME_MPEG1])
            .unwrap();
        assert!(enc.last_model2_window_switch(0, 0).is_some());
        enc.set_per_band_xmin(crate::psy::XminThresholds::uniform(1.0e6))
            .unwrap();
        assert!(!enc.model2_psychoacoustics_enabled());
        assert_eq!(enc.last_model2_window_switch(0, 0), None);
    }

    #[test]
    fn model2_block_type_requires_model2_armed() {
        // The Model-2-driven block-type path sources its attack signal
        // from the armed Model 2 analysis; without it the enable call is
        // rejected and the mode stays off.
        let mut enc =
            Mp3Encoder::new_with_outer_loop(128, 44_100, ChannelMode::SingleChannel, 1.0e6)
                .unwrap();
        assert!(!enc.auto_block_type_model2_enabled());
        assert!(matches!(
            enc.enable_auto_block_type_model2(),
            Err(StreamEncodeError::Model2BlockTypeWithoutModel2)
        ));
        assert!(!enc.auto_block_type_model2_enabled());
        // After arming Model 2 it succeeds.
        enc.enable_model2_psychoacoustics().unwrap();
        enc.enable_auto_block_type_model2().unwrap();
        assert!(enc.auto_block_type_model2_enabled());
    }

    #[test]
    fn model2_block_type_mutually_exclusive_with_energy_auto_and_force() {
        // Arming the Model-2-driven path clears the energy-detector auto
        // path and the force toggles, and arming any of those clears it.
        let mut enc =
            Mp3Encoder::new_with_outer_loop(128, 44_100, ChannelMode::SingleChannel, 1.0e6)
                .unwrap();
        enc.enable_model2_psychoacoustics().unwrap();

        // energy auto -> model2 clears energy auto.
        enc.enable_auto_block_type(10.0).unwrap();
        assert!(enc.auto_block_type_enabled());
        enc.enable_auto_block_type_model2().unwrap();
        assert!(enc.auto_block_type_model2_enabled());
        assert!(!enc.auto_block_type_enabled());

        // model2 -> energy auto clears model2.
        enc.enable_auto_block_type(10.0).unwrap();
        assert!(enc.auto_block_type_enabled());
        assert!(!enc.auto_block_type_model2_enabled());

        // re-arm model2, then a force toggle clears it.
        enc.enable_auto_block_type_model2().unwrap();
        assert!(enc.auto_block_type_model2_enabled());
        enc.force_short_blocks_for_testing(true).unwrap();
        assert!(!enc.auto_block_type_model2_enabled());
        assert!(enc.force_short_blocks_enabled());
    }

    #[test]
    fn model2_block_type_disarmed_when_model2_disarmed() {
        // Installing a static per-band vector disarms Model 2; the
        // Model-2-driven block-type path, which depends on it, is
        // disarmed too so the encoder never tries to drive block types
        // from an inactive model.
        let mut enc =
            Mp3Encoder::new_with_outer_loop(128, 44_100, ChannelMode::SingleChannel, 1.0e6)
                .unwrap();
        enc.enable_model2_psychoacoustics().unwrap();
        enc.enable_auto_block_type_model2().unwrap();
        assert!(enc.auto_block_type_model2_enabled());
        enc.set_per_band_xmin(crate::psy::XminThresholds::uniform(1.0e6))
            .unwrap();
        assert!(!enc.model2_psychoacoustics_enabled());
        assert!(!enc.auto_block_type_model2_enabled());

        // The explicit disable accessor is also a no-op-safe toggle.
        enc.enable_model2_psychoacoustics().unwrap();
        enc.enable_auto_block_type_model2().unwrap();
        enc.disable_auto_block_type_model2();
        assert!(!enc.auto_block_type_model2_enabled());
        // Disabling the block-type path leaves the Model 2 threshold
        // path armed (only the block-type scheduler was removed).
        assert!(enc.model2_psychoacoustics_enabled());
    }

    #[test]
    fn model2_block_type_steady_tone_stays_long() {
        // A steady tone never crosses the §C.1.5.3.2.1 `pe > 1800`
        // window-switching threshold (no transient), so every captured
        // decision is `attack == false` and every emitted granule is a
        // Long block — the §C.1.5.2 scheduler never leaves Long without
        // an attack ahead.
        use crate::frame::{parse_header, FrameWalker};
        use crate::side_info::{parse_side_info, BlockType};
        let mut enc =
            Mp3Encoder::new_with_outer_loop(128, 44_100, ChannelMode::SingleChannel, 1.0e6)
                .unwrap();
        enc.enable_model2_psychoacoustics().unwrap();
        enc.enable_auto_block_type_model2().unwrap();
        let total = SAMPLES_PER_FRAME_MPEG1 * 4;
        let pcm: Vec<i16> = (0..total)
            .map(|i| {
                let t = i as f64 / 44_100.0;
                (8000.0 * (2.0 * core::f64::consts::PI * 1000.0 * t).sin()) as i16
            })
            .collect();
        enc.push_samples(&pcm).unwrap();
        // No granule of the last frame flagged an attack on a steady
        // tone.
        for gr in 0..GRANULES {
            let sw = enc.last_model2_window_switch(gr, 0).unwrap();
            assert!(
                !sw.attack,
                "steady tone should not exceed pe > 1800 (gr {gr}, pe {})",
                sw.pe
            );
        }
        let mut out: Vec<u8> = Vec::new();
        enc.finish(&mut out).unwrap();
        // Every emitted granule across every frame is a Long block.
        for f in FrameWalker::new(&out) {
            let hdr = parse_header(&f.data[..4]).expect("valid header");
            let si = parse_side_info(&hdr, &f.data[4..]).expect("side info parses");
            for gr in 0..GRANULES {
                let gc = &si.granules[gr][0];
                assert!(!gc.window_switching_flag, "steady tone must stay long");
                assert_eq!(gc.block_type, BlockType::Long);
            }
        }
    }

    #[test]
    fn model2_block_type_emits_valid_c152_sequence() {
        // Across a multi-frame encode, every emitted block-type
        // transition is a valid §C.1.5.2 walk (Short only ever follows
        // Start or Short; Start only follows Long or End; End only
        // follows Short; Long only follows Long or End). This holds for
        // any signal regardless of whether `pe > 1800` ever fires, and
        // confirms the Model-2-driven scheduler produces self-consistent
        // window geometry rather than arbitrary per-granule types.
        use crate::frame::{parse_header, FrameWalker};
        use crate::side_info::{parse_side_info, BlockType};
        let rate = 44_100;
        let total = SAMPLES_PER_FRAME_MPEG1 * 6;
        let pcm: Vec<i16> = (0..total)
            .map(|i| {
                let t = i as f64 / f64::from(rate);
                let gate = if (i / 300) % 2 == 0 { 1.0 } else { 0.05 };
                let s = (2.0 * core::f64::consts::PI * 4000.0 * t).sin()
                    + (2.0 * core::f64::consts::PI * 9000.0 * t).sin()
                    + (2.0 * core::f64::consts::PI * 14000.0 * t).sin();
                (9000.0 * gate * s / 3.0) as i16
            })
            .collect();

        let mut enc =
            Mp3Encoder::new_with_outer_loop(128, rate, ChannelMode::SingleChannel, 1.0e6).unwrap();
        enc.enable_model2_psychoacoustics().unwrap();
        enc.enable_auto_block_type_model2().unwrap();
        enc.push_samples(&pcm).unwrap();
        let mut out: Vec<u8> = Vec::new();
        enc.finish(&mut out).unwrap();

        let mut seq: Vec<BlockType> = Vec::new();
        for f in FrameWalker::new(&out) {
            let hdr = parse_header(&f.data[..4]).expect("valid header");
            let si = parse_side_info(&hdr, &f.data[4..]).expect("side info parses");
            for gr in 0..GRANULES {
                seq.push(si.granules[gr][0].block_type);
            }
        }
        assert!(!seq.is_empty());
        let mut prev = BlockType::Long;
        for &bt in &seq {
            let ok = match prev {
                BlockType::Long => matches!(bt, BlockType::Long | BlockType::Start),
                BlockType::Start => matches!(bt, BlockType::Short),
                BlockType::Short => matches!(bt, BlockType::Short | BlockType::End),
                BlockType::End => matches!(bt, BlockType::Long | BlockType::Start),
            };
            assert!(ok, "invalid §C.1.5.2 transition {prev:?} -> {bt:?}");
            prev = bt;
        }
    }

    #[test]
    fn model2_block_type_per_frame_capture_matches_emission() {
        // Drive the encoder one frame at a time, capturing each frame's
        // §C.1.5.3.2.1 window-switching decision before pushing the
        // next, and confirm the emitted block types reproduce exactly
        // what replaying those captured `pe > 1800` attack flags through
        // a single continuous BlockTypeStateMachine yields. This is the
        // end-to-end proof that the Model 2 decision — not the energy
        // detector — governs the emitted window geometry, across frame
        // boundaries (the scheduler state carries between frames).
        use crate::block_type_sm::BlockTypeStateMachine;
        use crate::frame::{parse_header, FrameWalker};
        use crate::side_info::parse_side_info;
        let rate = 44_100;
        let nframes = 5;
        // Build per-frame PCM chunks of exactly one frame each.
        let make_frame = |fi: usize| -> Vec<i16> {
            (0..SAMPLES_PER_FRAME_MPEG1)
                .map(|j| {
                    let i = fi * SAMPLES_PER_FRAME_MPEG1 + j;
                    let t = i as f64 / f64::from(rate);
                    let gate = if (i / 350) % 2 == 0 { 1.0 } else { 0.03 };
                    let s = (2.0 * core::f64::consts::PI * 9000.0 * t).sin();
                    (9000.0 * gate * s) as i16
                })
                .collect()
        };

        let mut enc =
            Mp3Encoder::new_with_outer_loop(128, rate, ChannelMode::SingleChannel, 1.0e6).unwrap();
        enc.enable_model2_psychoacoustics().unwrap();
        enc.enable_auto_block_type_model2().unwrap();

        // Push one frame's PCM plus the next frame's leading granule as
        // lookahead each step so exactly one frame assembles per push,
        // and capture that frame's per-granule attack flags. We push
        // frames sequentially; the held-back lookahead granule rolls
        // forward in `pending_pcm`.
        let mut captured_attacks: Vec<[bool; GRANULES]> = Vec::new();
        for fi in 0..nframes {
            enc.push_samples(&make_frame(fi)).unwrap();
            // After the first push the encoder holds one frame + an
            // incomplete lookahead, so a frame only assembles from the
            // second push onward. Capture whenever a decision exists.
            if enc.last_model2_window_switch(0, 0).is_some() {
                let mut a = [false; GRANULES];
                for (gr, slot) in a.iter_mut().enumerate() {
                    *slot = enc.last_model2_window_switch(gr, 0).unwrap().attack;
                }
                captured_attacks.push(a);
            }
        }
        let mut out: Vec<u8> = Vec::new();
        enc.finish(&mut out).unwrap();

        // The captured frames are the leading `captured_attacks.len()`
        // emitted frames (the tail flush may add zero-padded frames we
        // don't compare). Replay the captured attacks through one
        // continuous scheduler and compare to the emitted block types.
        // The §C.1.5.2 companion of granule g is granule g+1's attack;
        // the last granule of the last captured frame uses `false`
        // (we have no further captured lookahead).
        let frames: Vec<_> = FrameWalker::new(&out).collect();
        let n = captured_attacks.len();
        assert!(n >= 2, "expected at least two captured frames, got {n}");
        // Flatten attacks into a granule stream with a trailing false.
        let mut flat: Vec<bool> = Vec::new();
        for fa in &captured_attacks {
            flat.extend_from_slice(fa);
        }
        flat.push(false); // lookahead past the last captured granule
        let mut sm = BlockTypeStateMachine::new();
        for (fidx, frame) in frames.iter().enumerate().take(n) {
            let hdr = parse_header(&frame.data[..4]).unwrap();
            let si = parse_side_info(&hdr, &frame.data[4..]).unwrap();
            for gr in 0..GRANULES {
                let g = fidx * GRANULES + gr;
                let expect = sm.step(flat[g], flat[g + 1]);
                assert_eq!(
                    si.granules[gr][0].block_type, expect,
                    "frame {fidx} gr {gr}: emitted block type must equal the \
                     scheduler walk over captured pe > 1800 attacks"
                );
            }
        }
    }

    #[test]
    fn model2_block_type_keeps_per_band_xmin_installed() {
        // The Model-2-driven block-type path reuses the same Model 2
        // walk for the per-band outer-loop threshold, so a real
        // signal-dependent `xmin` is still installed after the run (the
        // FFT history advances exactly once per granule — the block-type
        // pre-pass and Pass 1 do not double-process).
        let mut enc =
            Mp3Encoder::new_with_outer_loop(128, 44_100, ChannelMode::SingleChannel, 1.0e6)
                .unwrap();
        enc.enable_model2_psychoacoustics().unwrap();
        enc.enable_auto_block_type_model2().unwrap();
        let total = SAMPLES_PER_FRAME_MPEG1 * 3;
        let pcm: Vec<i16> = (0..total)
            .map(|i| {
                let t = i as f64 / 44_100.0;
                let s = 0.4 * (2.0 * core::f64::consts::PI * 1000.0 * t).sin()
                    + 0.3 * (2.0 * core::f64::consts::PI * 6000.0 * t).sin();
                (8000.0 * s) as i16
            })
            .collect();
        enc.push_samples(&pcm).unwrap();
        let x = enc
            .per_band_xmin
            .as_ref()
            .expect("a per-granule threshold should be installed");
        assert!(x.long.iter().all(|&v| v.is_finite() && v > 0.0));
        let first = x.long[0];
        assert!(
            x.long
                .iter()
                .any(|&v| (v - first).abs() > 1e-6 * first.max(1.0)),
            "expected a spectrally-shaped threshold under the Model-2 block-type path"
        );
    }

    #[test]
    fn enable_intensity_stereo_arms_and_rejects_mono() {
        // The running-state intensity arming method (r313) sets the
        // `mode = '01'` / `mode_extension = '01'` header template and the
        // intensity bound on an already-built stereo encoder, and rejects
        // a mono encoder (intensity folds an (L, R) pair).
        let mut stereo =
            Mp3Encoder::new_with_outer_loop(256, 44_100, ChannelMode::Stereo, 1.0e6).unwrap();
        assert!(!stereo.intensity_stereo_enabled());
        stereo.enable_intensity_stereo(8).unwrap();
        assert!(stereo.intensity_stereo_enabled());
        assert_eq!(stereo.intensity_start_sfb(), Some(8));
        assert_eq!(stereo.header_template.mode, ChannelMode::JointStereo);
        assert!(stereo.header_template.mode_extension.intensity_stereo);
        assert!(!stereo.header_template.mode_extension.ms_stereo);

        // Out-of-range bound is rejected.
        let mut s2 =
            Mp3Encoder::new_with_outer_loop(256, 44_100, ChannelMode::Stereo, 1.0e6).unwrap();
        assert!(matches!(
            s2.enable_intensity_stereo(0),
            Err(StreamEncodeError::InvalidIntensityStartSfb { start_sfb: 0 })
        ));

        // Mono is rejected.
        let mut mono =
            Mp3Encoder::new_with_outer_loop(128, 44_100, ChannelMode::SingleChannel, 1.0e6)
                .unwrap();
        assert!(matches!(
            mono.enable_intensity_stereo(8),
            Err(StreamEncodeError::StereoUnsupported)
        ));
    }

    #[test]
    fn model2_block_type_accepts_intensity_coupling() {
        // r313: the Model-2-driven block-type scheduler now accepts
        // intensity-stereo coupling (previously rejected with
        // `IntensityShortBlocksUnsupported`). The frame-assembly
        // `channel_agreement_active` OR-fold already keyed off
        // `intensity_start_sfb.is_some()` for the Model-2 emission path,
        // and Pass 1 selects the per-granule short / long intensity
        // coupling from the same block-type matrix this path produces, so
        // the only blocker was the constructor guard.
        let mut enc =
            Mp3Encoder::new_with_outer_loop(256, 44_100, ChannelMode::Stereo, 1.0e6).unwrap();
        enc.enable_intensity_stereo(8).unwrap();
        enc.enable_model2_psychoacoustics().unwrap();
        // Was previously `Err(IntensityShortBlocksUnsupported)`.
        enc.enable_auto_block_type_model2().unwrap();
        assert!(enc.auto_block_type_model2_enabled());
        assert!(enc.intensity_stereo_enabled());
    }

    #[test]
    fn model2_intensity_emits_valid_joint_intensity_frames() {
        // A transient hard-panned stereo stimulus drives the
        // Model-2-driven scheduler under intensity coupling. Every
        // emitted frame must carry `mode = '01'` (joint stereo) with the
        // intensity-stereo bit set, parse cleanly, and keep both channels
        // of each granule on the same window geometry (the §2.4.3.4.9
        // channel agreement the OR-fold enforces). Encoding the same PCM
        // twice yields byte-identical output (determinism).
        use crate::frame::{parse_header, FrameWalker};
        use crate::side_info::parse_side_info;
        let rate = 44_100;
        let total = SAMPLES_PER_FRAME_MPEG1 * 6;
        // Interleaved stereo: a high (intensity-region) tone gated into
        // bursts, panned mostly to the left channel.
        let pcm: Vec<i16> = (0..total)
            .flat_map(|i| {
                let t = i as f64 / f64::from(rate);
                let gate = if (i / 320) % 2 == 0 { 1.0 } else { 0.04 };
                let s = (2.0 * core::f64::consts::PI * 11000.0 * t).sin();
                let l = (9000.0 * gate * s) as i16;
                let r = (900.0 * gate * s) as i16;
                [l, r]
            })
            .collect();

        let encode_once = || -> Vec<u8> {
            let mut enc =
                Mp3Encoder::new_with_outer_loop(256, rate, ChannelMode::Stereo, 1.0e6).unwrap();
            enc.enable_intensity_stereo(8).unwrap();
            enc.enable_model2_psychoacoustics().unwrap();
            enc.enable_auto_block_type_model2().unwrap();
            enc.push_samples(&pcm).unwrap();
            let mut out: Vec<u8> = Vec::new();
            enc.finish(&mut out).unwrap();
            out
        };

        let out = encode_once();
        let mut nframes = 0usize;
        for f in FrameWalker::new(&out) {
            let hdr = parse_header(&f.data[..4]).expect("valid header");
            assert_eq!(
                hdr.mode,
                ChannelMode::JointStereo,
                "intensity frames carry mode = '01'"
            );
            assert!(
                hdr.mode_extension.intensity_stereo,
                "the intensity-stereo bit must be set"
            );
            let si = parse_side_info(&hdr, &f.data[4..]).expect("side info parses");
            for gr in 0..GRANULES {
                let l = &si.granules[gr][0];
                let r = &si.granules[gr][1];
                assert_eq!(
                    l.window_switching_flag, r.window_switching_flag,
                    "frame {nframes} gr {gr}: both channels must share window geometry"
                );
                assert_eq!(
                    l.block_type, r.block_type,
                    "frame {nframes} gr {gr}: both channels must share block_type"
                );
                assert_eq!(
                    l.mixed_block_flag, r.mixed_block_flag,
                    "frame {nframes} gr {gr}: both channels must share mixed_block_flag"
                );
            }
            nframes += 1;
        }
        assert!(nframes >= 4, "expected several frames, got {nframes}");

        // Determinism: same PCM in ⇒ same bytes out.
        let out2 = encode_once();
        assert_eq!(
            out, out2,
            "Model-2 + intensity encode must be deterministic"
        );
    }

    #[test]
    fn model2_block_type_matches_captured_attack_after_single_frame() {
        // Encode exactly one frame (+ one lookahead granule). With the
        // scheduler starting fresh at Long, the emitted block types of
        // the single frame must equal what feeding the captured per-
        // granule attack flags into a fresh BlockTypeStateMachine
        // produces — directly validating that the captured `pe > 1800`
        // decision is the signal that drives emission. The lookahead
        // attack for the last granule (gr 1's companion) is peeked from
        // a clone and never captured, so we assert only gr 0, whose
        // §C.1.5.2 companion is the captured gr 1 attack flag.
        use crate::block_type_sm::BlockTypeStateMachine;
        use crate::frame::{parse_header, FrameWalker};
        use crate::side_info::parse_side_info;
        let rate = 44_100;
        // One frame of audio plus one lookahead granule so exactly one
        // frame is assembled by push_samples.
        let total = SAMPLES_PER_FRAME_MPEG1 + SAMPLES_PER_GRANULE;
        let pcm: Vec<i16> = (0..total)
            .map(|i| {
                let t = i as f64 / f64::from(rate);
                let gate = if (i / 300) % 2 == 0 { 1.0 } else { 0.02 };
                let s = (2.0 * core::f64::consts::PI * 9000.0 * t).sin();
                (9000.0 * gate * s) as i16
            })
            .collect();
        let mut enc =
            Mp3Encoder::new_with_outer_loop(128, rate, ChannelMode::SingleChannel, 1.0e6).unwrap();
        enc.enable_model2_psychoacoustics().unwrap();
        enc.enable_auto_block_type_model2().unwrap();
        enc.push_samples(&pcm).unwrap();
        // Exactly one frame assembled; capture its decisions.
        let a0 = enc.last_model2_window_switch(0, 0).unwrap().attack;
        let a1 = enc.last_model2_window_switch(1, 0).unwrap().attack;

        let mut out: Vec<u8> = Vec::new();
        enc.finish(&mut out).unwrap();
        let frames: Vec<_> = FrameWalker::new(&out).collect();
        // The first emitted frame is the one whose decisions we captured
        // (finish only zero-pads a possible trailing tail; the first
        // frame is fully signal-driven).
        let f0 = &frames[0];
        let hdr = parse_header(&f0.data[..4]).unwrap();
        let si = parse_side_info(&hdr, &f0.data[4..]).unwrap();

        // Replay the captured gr0 attack into a fresh scheduler with
        // gr1's captured attack as its §C.1.5.2 lookahead companion.
        let mut sm = BlockTypeStateMachine::new();
        let expect_gr0 = sm.step(a0, a1);
        assert_eq!(
            si.granules[0][0].block_type, expect_gr0,
            "gr0 emitted block type must equal scheduler step(attack[0], attack[1])"
        );
    }

    // ---- §C.1.5.3 scfsi reuse detection ----

    fn sf_with_long(vals: [u8; 21]) -> ScaleFactors {
        let mut sf = ScaleFactors::default();
        sf.long[..21].copy_from_slice(&vals);
        sf
    }

    #[test]
    fn scfsi_all_groups_reuse_when_granules_identical() {
        // Two long granules with byte-identical scalefactors: every one
        // of the four scfsi_band groups is eligible for reuse.
        let gc = default_long_gc();
        let vals: [u8; 21] = core::array::from_fn(|i| (i % 8) as u8);
        let sf0 = sf_with_long(vals);
        let sf1 = sf_with_long(vals);
        assert_eq!(
            compute_scfsi_reuse(&gc, &gc, &sf0, &sf1),
            [true, true, true, true]
        );
    }

    #[test]
    fn scfsi_no_reuse_when_all_bands_differ() {
        let gc = default_long_gc();
        let sf0 = sf_with_long([0; 21]);
        let sf1 = sf_with_long([1; 21]);
        assert_eq!(
            compute_scfsi_reuse(&gc, &gc, &sf0, &sf1),
            [false, false, false, false]
        );
    }

    #[test]
    fn scfsi_per_group_independence() {
        // Make groups 0 and 2 agree, groups 1 and 3 differ. Group
        // ranges: {0..6}, {6..11}, {11..16}, {16..21}.
        let gc = default_long_gc();
        let a = [3u8; 21];
        let mut b = [3u8; 21];
        // Group 1 (band 6..11): perturb one band in granule 1.
        b[7] = 5;
        // Group 3 (band 16..21): perturb one band in granule 1.
        b[20] = 1;
        let sf0 = sf_with_long(a);
        let sf1 = sf_with_long(b);
        assert_eq!(
            compute_scfsi_reuse(&gc, &gc, &sf0, &sf1),
            [true, false, true, false]
        );
    }

    #[test]
    fn scfsi_disabled_when_either_granule_short() {
        // §2.4.2.7: a short block (block_type == 2) in either granule
        // forces scfsi to 0 for the whole channel, even when the
        // (long-array) scalefactors happen to be identical.
        let vals = [2u8; 21];
        let sf0 = sf_with_long(vals);
        let sf1 = sf_with_long(vals);
        let long = default_long_gc();
        let short = default_short_gc();
        assert_eq!(short.block_type, BlockType::Short);
        // gr0 short.
        assert_eq!(compute_scfsi_reuse(&short, &long, &sf0, &sf1), [false; 4]);
        // gr1 short.
        assert_eq!(compute_scfsi_reuse(&long, &short, &sf0, &sf1), [false; 4]);
        // both short.
        assert_eq!(compute_scfsi_reuse(&short, &short, &sf0, &sf1), [false; 4]);
    }

    #[test]
    fn scfsi_reuse_auto_armed_by_default_disarmed_by_toggle() {
        // r301: scfsi reuse is on out of the box (auto-armed in
        // `push_samples`). `disable_scfsi_reuse` restores the historical
        // `scfsi = 0` output, and `enable_scfsi_reuse` re-arms.
        let mut enc = Mp3Encoder::new(128, 44_100, ChannelMode::SingleChannel).unwrap();
        assert!(
            enc.scfsi_reuse_enabled(),
            "scfsi reuse must be auto-armed by default as of r301"
        );
        enc.disable_scfsi_reuse();
        assert!(!enc.scfsi_reuse_enabled());
        enc.enable_scfsi_reuse();
        assert!(enc.scfsi_reuse_enabled());
    }
}
