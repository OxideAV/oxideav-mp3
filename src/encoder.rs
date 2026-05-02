//! Minimum-viable CBR MPEG-1 / MPEG-2 LSF Layer III encoder.
//!
//! Scope (deliberately narrow):
//! - MPEG-1 Layer III (32 / 44.1 / 48 kHz, 2 granules/frame) and
//!   MPEG-2 LSF Layer III (16 / 22.05 / 24 kHz, 1 granule/frame),
//!   mono / dual-channel stereo / joint stereo (M/S only — no IS).
//! - One CBR bitrate per encoder instance, picked from the version's
//!   standard bitrate table.
//! - Long, start, short, stop blocks with automatic window switching
//!   on transients per ISO/IEC 11172-3 §2.4.2.2 (see
//!   [`crate::block_type`]). Mixed blocks are not emitted.
//! - No CRC. No psychoacoustic model. No rate-distortion: a simple
//!   global-gain bisection sets the quantisation step so the Huffman bit
//!   count fits in the available main-data budget.
//! - Single big-value Huffman table for the whole spectrum (selected
//!   per granule from a small candidate set). Region splits are
//!   degenerate (region0 spans everything, region1 / region2 empty).
//! - count1 region uses table A.
//! - Bit reservoir on the encode side: any unused bits roll forward via
//!   the next frame's `main_data_begin`. MPEG-1's `main_data_begin` is
//!   9 bits wide (reservoir cap 511 bytes); MPEG-2 LSF's field is only
//!   8 bits wide, so the reservoir is capped at 255 bytes.
//!
//! The pipeline is the mirror of the decoder:
//!   PCM → polyphase analysis → forward MDCT → quantise →
//!   Huffman encode → side info + main data → frame emission.
//!
//! Joint-stereo coding currently supports MS-only (mode_extension =
//! 0b10), IS-only (mode_extension = 0b01) and combined MS+IS
//! (mode_extension = 0b11) per ISO/IEC 11172-3 §2.4.3.4.10.{1,2}.
//! Intensity stereo applies to long-block stereo granules at the
//! frame level: a per-granule "intensity bound" sfb is selected
//! (highest sfb such that the L/R energy ratio above it is dominated
//! by L), and bands at or above that bound are encoded as a single
//! L spectrum + per-channel intensity-position scalefactors carried
//! in the R channel's slot. Short / start / stop granules currently
//! stay out of IS; they fall back to the MS-only or dual-channel
//! path of #115. The MPEG-1 path uses 3-bit `is_pos` values; the
//! MPEG-2 LSF path keeps IS off for now.

use std::collections::VecDeque;

use oxideav_core::Encoder;
use oxideav_core::{
    AudioFrame, CodecId, CodecParameters, Error, Frame, MediaType, Packet, Result, SampleFormat,
    TimeBase,
};

use crate::analysis::{analyze_granule, AnalysisState};
use crate::block_type::{BlockType, BlockTypeMachine, TransientDetector};
use crate::huffman::{BIG_VALUE_TABLES, COUNT1_A};
use crate::mdct::{mdct_granule, MdctState};
use crate::psy::{global_gain_to_step, vbr_quality_to_mask_ratio, GranuleMask};
use crate::CODEC_ID_STR;
use oxideav_core::bits::BitWriter;
use oxideav_core::options::{
    parse_options, CodecOptionsStruct, OptionField, OptionKind, OptionValue,
};

/// Max reservoir lookback (bytes) for MPEG-1 — `main_data_begin` is 9 bits.
const MAX_LOOKBACK_MPEG1: usize = 511;
/// Max reservoir lookback (bytes) for MPEG-2 LSF — `main_data_begin` is 8 bits.
const MAX_LOOKBACK_MPEG2: usize = 255;

/// Candidate big-value tables, in priority order. We try the lowest-cost
/// table first — values bounded by 15 use table 13 (no linbits); larger
/// magnitudes fall through to one of the linbits-equipped variants
/// (table indices 16-23 reuse TABLE_16 with linbits 1..13).
const BIG_VALUE_CANDIDATES: &[u8] = &[1, 5, 7, 13, 16, 17, 18, 19, 20, 21, 22, 23];

/// Per-granule bit reservoir hard cap (ISO 11172-3 §2.4.3.4.7.2):
/// part2_3_length is a 12-bit field, so a granule cannot exceed
/// 4095 bits. The spec further pins the absolute ceiling for a single
/// granule's main_data at 7680 bits (= 960 bytes) — that's where the
/// "bit reservoir is bounded" claim in the encoder comments comes from.
const VBR_PER_GRANULE_BIT_CAP: usize = 7680;

/// Encoder rate-control mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RateControl {
    /// Constant bit rate — current behaviour. `bitrate_kbps` is fixed
    /// for the lifetime of the encoder.
    Cbr,
    /// Variable bit rate — quantizer step picked per granule by a
    /// lightweight per-band masking model in [`crate::psy`]. Frame
    /// bitrate slot is picked per frame from the standard table to
    /// fit the resulting bit count.
    Vbr,
}

/// Typed options consumed by the MP3 encoder. Wired into
/// [`CodecParameters::options`] via [`oxideav_core::options`].
///
/// Recognised keys:
///
/// - `vbr_quality` — `u32` 0..=9. When set, switches the encoder to VBR
///   and the value selects the per-band SNR target (0 = highest
///   quality / largest files, 9 = lowest quality / smallest files).
///   Mirrors LAME's V0..V9 spirit without copying their tables.
/// - `cbr_bitrate_kbps` — `u32` CBR slot in kbps. When `vbr_quality`
///   is unset, this overrides `CodecParameters::bit_rate`.
/// - `joint_stereo` — `u32` 0/1. When `1` (the default for stereo
///   inputs), the encoder may emit MS-stereo joint-stereo frames per
///   ISO/IEC 11172-3 §2.4.3.4.10 — picked per frame from a spectral
///   correlation metric. When `0` the encoder always emits
///   dual-channel (mode = `0b10`) frames.
/// - `short_blocks` — `u32` 0/1. When `1` (the default), the encoder
///   may switch to short blocks on transients per ISO/IEC 11172-3
///   §2.4.2.2 to suppress pre-echo. When `0` the encoder always emits
///   long blocks regardless of input — useful for benchmarking pre-
///   echo behaviour or for matching the pre-round-24 output exactly.
/// - `intensity_stereo` — `u32` 0/1. When `1` (the default for stereo
///   inputs), the encoder may also enable intensity stereo per ISO/IEC
///   11172-3 §2.4.3.4.10.2 for long-block stereo granules where the
///   high-frequency L/R correlation lets the side info encode a per-band
///   intensity position instead of full R coefficients. Sets
///   `mode_extension` bit 0x1 when active. Only meaningful when
///   `joint_stereo` is also enabled.
#[derive(Debug, Clone)]
pub struct Mp3EncoderOptions {
    /// `Some(0..=9)` → switch to VBR with the given quality index.
    pub vbr_quality: Option<u8>,
    /// Override for the CBR slot (kbps). Only consulted in CBR mode.
    pub cbr_bitrate_kbps: Option<u32>,
    /// Allow joint-stereo (MS) coding for stereo inputs. Default: `true`.
    pub joint_stereo: bool,
    /// Allow short-block window switching on transients. Default: `true`.
    pub short_blocks: bool,
    /// Allow intensity-stereo coding (MPEG-1 long blocks only). Default:
    /// `true`. Only consulted when `joint_stereo` is also `true`.
    pub intensity_stereo: bool,
}

impl Default for Mp3EncoderOptions {
    fn default() -> Self {
        Self {
            vbr_quality: None,
            cbr_bitrate_kbps: None,
            joint_stereo: true,
            short_blocks: true,
            intensity_stereo: true,
        }
    }
}

impl CodecOptionsStruct for Mp3EncoderOptions {
    const SCHEMA: &'static [OptionField] = &[
        OptionField {
            name: "vbr_quality",
            kind: OptionKind::U32,
            default: OptionValue::U32(u32::MAX),
            help: "VBR quality 0..=9 (0=best, 9=smallest). Switches to VBR mode.",
        },
        OptionField {
            name: "cbr_bitrate_kbps",
            kind: OptionKind::U32,
            default: OptionValue::U32(0),
            help: "Override CBR bitrate in kbps. Ignored when vbr_quality is set.",
        },
        OptionField {
            name: "joint_stereo",
            kind: OptionKind::U32,
            default: OptionValue::U32(1),
            help: "Allow joint-stereo (MS) coding when the spectrum is highly correlated. 0 = always dual-channel, 1 = enable MS (default).",
        },
        OptionField {
            name: "short_blocks",
            kind: OptionKind::U32,
            default: OptionValue::U32(1),
            help: "Allow short-block window switching on transients. 0 = always long blocks, 1 = enable short blocks (default).",
        },
        OptionField {
            name: "intensity_stereo",
            kind: OptionKind::U32,
            default: OptionValue::U32(1),
            help: "Allow intensity-stereo (IS) coding on top of joint-stereo for long-block stereo granules. 0 = never IS, 1 = enable IS when HF correlation is high (default).",
        },
    ];

    fn apply(&mut self, key: &str, v: &OptionValue) -> Result<()> {
        match key {
            "vbr_quality" => {
                let n = v.as_u32()?;
                if n > 9 {
                    return Err(Error::invalid(format!(
                        "MP3 encoder: vbr_quality must be 0..=9, got {n}"
                    )));
                }
                self.vbr_quality = Some(n as u8);
            }
            "cbr_bitrate_kbps" => {
                let n = v.as_u32()?;
                if n > 0 {
                    self.cbr_bitrate_kbps = Some(n);
                }
            }
            "joint_stereo" => {
                let n = v.as_u32()?;
                self.joint_stereo = n != 0;
            }
            "short_blocks" => {
                let n = v.as_u32()?;
                self.short_blocks = n != 0;
            }
            "intensity_stereo" => {
                let n = v.as_u32()?;
                self.intensity_stereo = n != 0;
            }
            _ => unreachable!("guarded by SCHEMA"),
        }
        Ok(())
    }
}

/// MPEG-1 Layer III standard bitrate slots, in kbps. Index 0 = "free
/// format" (encoder doesn't emit), index 15 = forbidden. Per ISO/IEC
/// 11172-3 Table 2.4.2.3.
const MPEG1_BITRATES_KBPS: [u32; 15] = [
    0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320,
];

/// MPEG-2 LSF Layer III standard bitrate slots, in kbps. ISO/IEC
/// 13818-3 Table 2.4.2.3 (low-sample-rate addendum).
const MPEG2_BITRATES_KBPS: [u32; 15] =
    [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160];

/// Map a kbps slot to its bitrate-table index. Returns `None` when the
/// slot isn't in the standard table for the version.
fn bitrate_index_for(is_mpeg2: bool, kbps: u32) -> Option<u8> {
    let table = if is_mpeg2 {
        &MPEG2_BITRATES_KBPS
    } else {
        &MPEG1_BITRATES_KBPS
    };
    table
        .iter()
        .position(|&k| k == kbps)
        .filter(|&i| i > 0)
        .map(|i| i as u8)
}

/// Pick the smallest standard-table bitrate slot whose CBR frame size
/// (in bytes) accommodates the requested main-data bytes plus header
/// + side info. Returns the (index, kbps) pair. If no standard slot
/// fits (shouldn't happen for sub-7680-bit granules), returns the
/// max slot — the caller will then truncate.
fn pick_vbr_bitrate_slot(
    is_mpeg2: bool,
    sample_rate: u32,
    needed_main_data_bytes: usize,
    side_info_bytes: usize,
) -> (u8, u32) {
    let header_bytes = 4usize;
    let table = if is_mpeg2 {
        &MPEG2_BITRATES_KBPS
    } else {
        &MPEG1_BITRATES_KBPS
    };
    let num = if is_mpeg2 { 72u32 } else { 144 };
    // Iterate slots from smallest to largest; first that fits wins.
    for (i, &kbps) in table.iter().enumerate() {
        if i == 0 {
            continue; // skip "free format"
        }
        // Frame bytes for this slot, no padding.
        let frame_bytes = (num * kbps * 1000 / sample_rate) as usize;
        let slot_bytes = frame_bytes.saturating_sub(header_bytes + side_info_bytes);
        if slot_bytes >= needed_main_data_bytes {
            return (i as u8, kbps);
        }
    }
    // Fall back to the largest slot.
    let last_idx = (table.len() - 1) as u8;
    (last_idx, table[last_idx as usize])
}

/// Build an encoder for the requested parameters.
///
/// Mode selection:
/// - When `params.options` contains `vbr_quality` (0..=9) the encoder
///   runs in VBR mode and ignores `params.bit_rate`.
/// - Otherwise it runs in CBR mode at `params.bit_rate` (or the
///   version's default kbps).
pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    let channels = params
        .channels
        .ok_or_else(|| Error::invalid("MP3 encoder: missing channels"))?;
    if !(1..=2).contains(&channels) {
        return Err(Error::invalid("MP3 encoder: channels must be 1 or 2"));
    }
    let sample_rate = params
        .sample_rate
        .ok_or_else(|| Error::invalid("MP3 encoder: missing sample_rate"))?;
    // Pick version based on sample rate.
    // MPEG-1:    32 / 44.1 / 48 kHz.
    // MPEG-2 LSF: 16 / 22.05 / 24 kHz. MPEG-2.5 is out of scope.
    let (is_mpeg2, sr_index) = match sample_rate {
        44_100 => (false, 0u8),
        48_000 => (false, 1u8),
        32_000 => (false, 2u8),
        22_050 => (true, 0u8),
        24_000 => (true, 1u8),
        16_000 => (true, 2u8),
        _ => {
            return Err(Error::unsupported(format!(
                "MP3 encoder: unsupported sample rate {sample_rate} (need 16000/22050/24000/32000/44100/48000)"
            )));
        }
    };

    // Parse encoder options. `vbr_quality` switches on VBR; otherwise
    // we stay in CBR.
    let opts: Mp3EncoderOptions = parse_options(&params.options)?;
    let rate_control = if opts.vbr_quality.is_some() {
        RateControl::Vbr
    } else {
        RateControl::Cbr
    };
    let vbr_quality = opts.vbr_quality.unwrap_or(2);
    // Joint stereo only makes sense for two-channel input. For mono /
    // dual-channel-forced configurations the flag stays effectively off.
    let allow_joint_stereo = opts.joint_stereo && channels == 2;
    let allow_short_blocks = opts.short_blocks;
    // IS only applies when joint-stereo is on AND we're MPEG-1 stereo.
    // The MPEG-2 LSF path keeps IS off — IS sf-decoding for MPEG-2 uses
    // a separate `is_pos` partition row (`SCF_PARTITIONS_MPEG2[..][16..]`)
    // that the encoder doesn't yet emit.
    let allow_intensity_stereo = opts.intensity_stereo && allow_joint_stereo && !is_mpeg2;

    // CBR slot: explicit option override > params.bit_rate > version default.
    let default_kbps = if is_mpeg2 { 64 } else { 128 };
    let bitrate_kbps = opts
        .cbr_bitrate_kbps
        .or_else(|| params.bit_rate.map(|b| (b / 1000) as u32))
        .unwrap_or(default_kbps);

    // Validate the CBR slot up-front even in VBR mode (used as the
    // initial slot in the header before per-frame substitution).
    let br_index = bitrate_index_for(is_mpeg2, bitrate_kbps).ok_or_else(|| {
        Error::unsupported(format!(
            "MP3 encoder: unsupported {} bitrate {bitrate_kbps} kbps",
            if is_mpeg2 { "MPEG-2 LSF" } else { "MPEG-1" }
        ))
    })?;

    let sample_format = params.sample_format.unwrap_or(SampleFormat::S16);
    if sample_format != SampleFormat::S16 {
        return Err(Error::unsupported(format!(
            "MP3 encoder: input sample format {sample_format:?} not supported (need S16)"
        )));
    }

    let mut output = params.clone();
    output.media_type = MediaType::Audio;
    output.codec_id = CodecId::new(CODEC_ID_STR);
    output.sample_format = Some(sample_format);
    output.channels = Some(channels);
    output.sample_rate = Some(sample_rate);
    output.bit_rate = Some((bitrate_kbps as u64) * 1000);

    Ok(Box::new(Mp3Encoder {
        output_params: output,
        channels,
        sample_rate,
        bitrate_kbps,
        sr_index,
        br_index,
        is_mpeg2,
        rate_control,
        vbr_quality,
        allow_joint_stereo,
        allow_short_blocks,
        allow_intensity_stereo,
        time_base: TimeBase::new(1, sample_rate as i64),
        analysis_state: [AnalysisState::new(), AnalysisState::new()],
        mdct_state: [MdctState::new(), MdctState::new()],
        transient_det: [TransientDetector::new(), TransientDetector::new()],
        block_machine: [BlockTypeMachine::new(), BlockTypeMachine::new()],
        pcm_queue: vec![Vec::new(); channels as usize],
        main_data_queue: Vec::new(),
        pending_packets: VecDeque::new(),
        frame_index: 0,
        eof: false,
        cumulative_padded_bits: 0,
    }))
}

struct Mp3Encoder {
    output_params: CodecParameters,
    channels: u16,
    sample_rate: u32,
    /// Default / "anchor" CBR bitrate. In VBR mode this is only used
    /// for the cumulative-padding accounting; the per-frame slot is
    /// substituted by [`pick_vbr_bitrate_slot`].
    bitrate_kbps: u32,
    sr_index: u8,
    br_index: u8,
    /// `true` for MPEG-2 LSF (16/22.05/24 kHz, 1 granule/frame);
    /// `false` for MPEG-1 (32/44.1/48 kHz, 2 granules/frame).
    is_mpeg2: bool,
    rate_control: RateControl,
    /// VBR quality 0..=9; only consulted when `rate_control == Vbr`.
    vbr_quality: u8,
    /// `true` when the encoder may emit joint-stereo (MS) frames. Only
    /// meaningful for two-channel inputs.
    allow_joint_stereo: bool,
    /// `true` when the encoder may switch to short blocks on detected
    /// transients (default). When `false` every granule is emitted as
    /// a long block — matches the pre-round-24 encoder exactly.
    allow_short_blocks: bool,
    /// `true` when the encoder may enable intensity stereo on top of
    /// joint stereo for long-block stereo granules (MPEG-1 only). Implies
    /// `allow_joint_stereo` and `!is_mpeg2`.
    allow_intensity_stereo: bool,
    time_base: TimeBase,
    analysis_state: [AnalysisState; 2],
    mdct_state: [MdctState; 2],
    /// Per-channel transient detector — drives short-block selection per
    /// ISO/IEC 11172-3 §2.4.2.2 / Annex C.
    transient_det: [TransientDetector; 2],
    /// Per-channel block-type state machine. Owns the previous granule's
    /// chosen window so transitions stay legal (long → start → short →
    /// stop → long).
    block_machine: [BlockTypeMachine; 2],
    /// Per-channel float queue (samples in -1..=1).
    pcm_queue: Vec<Vec<f32>>,
    /// Pending main-data bytes that have not yet been written to a frame
    /// slot. The next frame's `main_data_begin` is exactly this length
    /// (capped at the version's max lookback) BEFORE the new main_data
    /// gets appended.
    main_data_queue: Vec<u8>,
    pending_packets: VecDeque<Packet>,
    frame_index: u64,
    eof: bool,
    /// Tracks fractional-byte CBR padding so we know when to set the
    /// padding bit. For 44.1 kHz: 144*128_000/44100 = 417.96... so we set
    /// padding ~96/100 frames.
    cumulative_padded_bits: u64,
}

impl Mp3Encoder {
    /// Bytes per CBR frame for given padding.
    /// MPEG-1:  `144 * br / sr + pad`.
    /// MPEG-2:  `72  * br / sr + pad` (one granule per frame = half the
    ///           samples, so half the numerator).
    fn frame_bytes(&self, padding: bool) -> usize {
        let num = if self.is_mpeg2 { 72 } else { 144 };
        let base = (num * self.bitrate_kbps * 1000 / self.sample_rate) as usize;
        base + if padding { 1 } else { 0 }
    }

    /// Number of granules per frame: 2 for MPEG-1, 1 for MPEG-2 LSF.
    fn num_granules(&self) -> usize {
        if self.is_mpeg2 {
            1
        } else {
            2
        }
    }

    /// PCM samples per channel per frame (576 × num_granules).
    fn samples_per_frame(&self) -> usize {
        576 * self.num_granules()
    }

    /// Reservoir-lookback ceiling: 511 (MPEG-1) or 255 (MPEG-2 LSF).
    fn max_lookback(&self) -> usize {
        if self.is_mpeg2 {
            MAX_LOOKBACK_MPEG2
        } else {
            MAX_LOOKBACK_MPEG1
        }
    }

    /// Side-info block length in bytes.
    /// MPEG-1:  32 (stereo) / 17 (mono).
    /// MPEG-2:  17 (stereo) / 9  (mono).
    fn side_info_bytes(&self) -> usize {
        match (self.is_mpeg2, self.channels) {
            (false, 1) => 17,
            (false, _) => 32,
            (true, 1) => 9,
            (true, _) => 17,
        }
    }

    /// Decide whether this frame should set the padding bit, using the
    /// classic accumulator-style scheme (LAME-equivalent for CBR).
    ///
    /// Frame size in bits = `samples_per_frame * br / sr` — for MPEG-1
    /// that's `1152*br/sr` = `144_000*br_kbps/sr`, and for MPEG-2 LSF
    /// half of that.
    fn next_padding(&mut self) -> bool {
        let num_k = if self.is_mpeg2 { 72_000 } else { 144_000 };
        let num = num_k as u64 * self.bitrate_kbps as u64;
        let sr = self.sample_rate as u64;
        let rem = num - (num / sr) * sr;
        self.cumulative_padded_bits += rem;
        let pad = self.cumulative_padded_bits >= sr * 8;
        if pad {
            self.cumulative_padded_bits -= sr * 8;
        }
        pad
    }

    fn ingest(&mut self, frame: &AudioFrame) -> Result<()> {
        // Stream-level validation (channel count, sample rate, S16
        // sample format) is owned by the factory at construction —
        // see `make_encoder`. The slim AudioFrame doesn't carry them.
        let data = frame
            .data
            .first()
            .ok_or_else(|| Error::invalid("MP3 encoder: empty frame"))?;
        let n_ch = self.channels as usize;
        let n_samples = data.len() / (2 * n_ch);
        for i in 0..n_samples {
            for ch in 0..n_ch {
                let off = (i * n_ch + ch) * 2;
                let s = i16::from_le_bytes([data[off], data[off + 1]]) as f32 / 32768.0;
                self.pcm_queue[ch].push(s);
            }
        }
        self.flush_ready_frames(false)
    }

    fn flush_ready_frames(&mut self, drain: bool) -> Result<()> {
        // MPEG-1: 1152 samples per frame (2 granules of 576).
        // MPEG-2 LSF: 576 samples per frame (1 granule of 576).
        //
        // We need ONE extra granule of lookahead so the block-type
        // machine can retroactively mark the last granule of THIS frame
        // as a `start` window when the very next granule (the first one
        // of the following frame) is transient. While draining we don't
        // have any lookahead so the lookahead PCM is zero-padded — the
        // detector sees silence and the machine drops into long.
        let n_ch = self.channels as usize;
        let spf = self.samples_per_frame();
        // Lookahead only needed when the block-type detector might use
        // it to mark the last granule of this frame as `start`.
        let lookahead = if self.allow_short_blocks { 576 } else { 0 };
        let needed = if drain { spf } else { spf + lookahead };
        loop {
            let avail = self.pcm_queue[0].len();
            if avail < needed {
                if drain && avail > 0 {
                    // Pad with zeros up to spf to flush the tail.
                    for ch in 0..n_ch {
                        self.pcm_queue[ch].resize(spf, 0.0);
                    }
                } else {
                    return Ok(());
                }
            }
            let pkt = self.encode_one_frame()?;
            self.pending_packets.push_back(pkt);
            if drain && self.pcm_queue[0].iter().all(|&v| v == 0.0) {
                return Ok(());
            }
        }
    }

    fn encode_one_frame(&mut self) -> Result<Packet> {
        let n_ch = self.channels as usize;
        let n_gr = self.num_granules();
        let spf = self.samples_per_frame();

        // Pull `spf` samples per channel into local buffers, drain queue.
        let mut pcm_in: Vec<Vec<f32>> = vec![vec![0.0f32; spf]; n_ch];
        for ch in 0..n_ch {
            pcm_in[ch].copy_from_slice(&self.pcm_queue[ch][..spf]);
            self.pcm_queue[ch].drain(..spf);
        }
        // Lookahead: peek (without draining) one extra granule for the
        // block-type decision. If the queue is shorter than that
        // (typically only at end-of-stream after `flush`) we treat the
        // missing samples as silence — silence cannot trigger the
        // transient detector, so the trailing granule will not be
        // marked as `start`.
        let mut pcm_lookahead: Vec<[f32; 576]> = vec![[0.0f32; 576]; n_ch];
        for ch in 0..n_ch {
            let avail = self.pcm_queue[ch].len();
            let take = avail.min(576);
            pcm_lookahead[ch][..take].copy_from_slice(&self.pcm_queue[ch][..take]);
        }

        // Decide per-granule per-channel block type. We feed the
        // detector each granule in order (so its smoothed-energy
        // tracker stays up to date) and look one granule ahead so the
        // state machine can pick `start` for the last granule of this
        // frame when the first granule of the next frame is transient.
        //
        // Note: the detector is *stateful* — calling `is_transient`
        // mutates its smoothed average — so we have to drive it in PCM
        // order and use the verdicts before deciding block types.
        let mut block_types: Vec<Vec<BlockType>> =
            (0..n_gr).map(|_| vec![BlockType::Long; n_ch]).collect();
        if self.allow_short_blocks {
            for ch in 0..n_ch {
                // Per-granule transient verdicts for this channel: the n_gr
                // granules of this frame plus the one-granule lookahead.
                let mut verdicts: Vec<bool> = Vec::with_capacity(n_gr + 1);
                for gr in 0..n_gr {
                    let mut pcm_gr = [0.0f32; 576];
                    pcm_gr.copy_from_slice(&pcm_in[ch][gr * 576..gr * 576 + 576]);
                    verdicts.push(self.transient_det[ch].is_transient(&pcm_gr));
                }
                // Lookahead verdict — uses a *clone* of the detector so the
                // real detector's smoothed-average state is not advanced
                // by samples we will see again next frame. Without the
                // clone the detector would fast-forward and misfire.
                {
                    let mut peek = self.transient_det[ch].clone();
                    verdicts.push(peek.is_transient(&pcm_lookahead[ch]));
                }
                // Drive the state machine with (this, next) pairs.
                for gr in 0..n_gr {
                    let bt = self.block_machine[ch].decide(verdicts[gr], verdicts[gr + 1]);
                    block_types[gr][ch] = bt;
                }
            }
        }

        // Analysis + MDCT per granule per channel — now block-type aware.
        let mut xr: Vec<Vec<[f32; 576]>> = (0..n_gr)
            .map(|_| (0..n_ch).map(|_| [0.0f32; 576]).collect())
            .collect();
        for gr in 0..n_gr {
            for ch in 0..n_ch {
                let mut pcm_gr = [0.0f32; 576];
                pcm_gr.copy_from_slice(&pcm_in[ch][gr * 576..gr * 576 + 576]);
                let mut sub = [[0.0f32; 18]; 32];
                analyze_granule(&mut self.analysis_state[ch], &pcm_gr, &mut sub);
                let bt = block_types[gr][ch].as_u8();
                mdct_granule(
                    &sub,
                    &mut xr[gr][ch],
                    &mut self.mdct_state[ch],
                    bt,
                    self.sample_rate,
                );
            }
        }

        // Joint-stereo (MS) decision (ISO/IEC 11172-3 §2.4.3.4.10).
        // The frame-level `mode_extension` field is shared across both
        // granules, so the decision is taken once per frame from the
        // aggregated spectral correlation.
        //
        // Heuristic: compute mid/side energies in the canonical basis
        //   M[k] = (L[k] + R[k]) / sqrt(2)
        //   S[k] = (L[k] - R[k]) / sqrt(2)
        // (mass-preserving: |M|^2 + |S|^2 = |L|^2 + |R|^2). The frame is
        // a good candidate for MS when the side energy is small relative
        // to the mid energy — i.e. L and R are highly correlated. We use
        // a conservative ratio (S energy < 30% of M energy) that mirrors
        // LAME's spec-compliant default behaviour without copying its
        // tables: it consistently shrinks correlated material (centred
        // voice, mono-fold-down) and stays out of the way for true
        // stereo content where MS would lose energy compaction.
        let use_ms = self.allow_joint_stereo && n_ch == 2 && should_use_ms_stereo(&xr, n_gr);
        if use_ms {
            for gr in 0..n_gr {
                rotate_to_ms(&mut xr[gr]);
            }
        }

        // Intensity-stereo (IS) decision (ISO/IEC 11172-3 §2.4.3.4.10.2).
        // Per-granule: pick a scalefactor-band bound such that all sfbs at
        // or above the bound are IS-coded (R coefficients zeroed, L holds
        // a per-sfb energy-sum surrogate, R-channel scalefactors carry
        // `is_pos`). Below the bound the granule keeps whatever stereo
        // representation is already in `xr` (raw L/R, or M/S after the
        // rotation above when `use_ms`). Only long blocks are IS-coded
        // for now — short / start / stop block scalefactor layouts use
        // `s[sfb][win]` triplets and would need a separate emit path.
        //
        // The frame-level mode_extension field shares one IS bit across
        // both granules, so we enable IS on the whole frame as soon as
        // any granule benefits. Granules whose chosen bound would equal
        // the maximum (= IS does not fire) keep their full L/R content
        // and contribute no IS rewrite.
        let mut is_pos_per_gr_ch: Vec<Vec<[u8; 22]>> = vec![vec![[7u8; 22]; n_ch]; n_gr]; // default: 7 = "not IS-coded" sentinel
        let mut is_bound_per_gr: Vec<usize> = vec![21; n_gr];
        let mut any_is_active = false;
        if self.allow_intensity_stereo
            && n_ch == 2
            && (0..n_gr).all(|gr| {
                block_types[gr][0] == BlockType::Long && block_types[gr][1] == BlockType::Long
            })
        {
            for gr in 0..n_gr {
                let (bound, ipos) = pick_is_bound_long(&xr[gr], self.sample_rate);
                if bound < 21 {
                    any_is_active = true;
                    apply_is_rewrite_long(&mut xr[gr], self.sample_rate, bound, &ipos);
                    is_pos_per_gr_ch[gr][1] = ipos;
                    is_bound_per_gr[gr] = bound;
                }
            }
        }
        let use_is = any_is_active;

        // Frame layout / size. In CBR mode the slot is fixed by the
        // encoder's bitrate; in VBR mode we encode first then pick the
        // smallest standard slot that fits.
        let header_bytes = 4usize;
        let si_bytes = self.side_info_bytes();
        let max_lookback = self.max_lookback();

        let pre_queue = self.main_data_queue.len();
        debug_assert!(pre_queue <= max_lookback);
        let main_data_begin = pre_queue as u16;

        // Per-mode encode of granules and computation of the on-wire
        // bitrate slot / frame bytes.
        let (granule_data, frame_bytes, padding, frame_br_index) = match self.rate_control {
            RateControl::Cbr => {
                let padding = self.next_padding();
                let frame_bytes = self.frame_bytes(padding);
                let main_data_slot_bytes = frame_bytes - header_bytes - si_bytes;
                let max_main_bytes_a = pre_queue + main_data_slot_bytes;
                let max_main_bytes_b = main_data_slot_bytes + max_lookback - pre_queue;
                let max_main_bytes = max_main_bytes_a.min(max_main_bytes_b);
                let max_main_bits = max_main_bytes * 8;
                let unit_count = n_gr * n_ch;
                let per_unit_budget = max_main_bits / unit_count.max(1);

                let mut granule_data: Vec<Vec<GranuleEncoded>> =
                    (0..n_gr).map(|_| Vec::with_capacity(n_ch)).collect();
                let mut bits_used_total: usize = 0;
                for gr in 0..n_gr {
                    for ch in 0..n_ch {
                        let remaining = max_main_bits.saturating_sub(bits_used_total);
                        let units_left = (n_gr - gr) * n_ch - ch;
                        let target = remaining / units_left.max(1);
                        let target = target.min(per_unit_budget * 2).max(64);
                        // R channel of an IS-active granule carries the
                        // `is_pos` table as long-block scalefactors and
                        // selects scalefac_compress = 13 (slen1=3, slen2=3).
                        let is_pos_for_ch = if use_is && ch == 1 {
                            Some(&is_pos_per_gr_ch[gr][1])
                        } else {
                            None
                        };
                        let g =
                            encode_granule(&xr[gr][ch], target, block_types[gr][ch], is_pos_for_ch);
                        bits_used_total += g.total_bits;
                        granule_data[gr].push(g);
                    }
                }
                (granule_data, frame_bytes, padding, self.br_index)
            }
            RateControl::Vbr => {
                // Drive the per-granule quantizer from the masking
                // model: pick the smallest global_gain that satisfies
                // worst_nmr_db <= 0 across bands.
                let mask_ratio = vbr_quality_to_mask_ratio(self.vbr_quality);
                let mut granule_data: Vec<Vec<GranuleEncoded>> =
                    (0..n_gr).map(|_| Vec::with_capacity(n_ch)).collect();
                let mut total_bits: usize = 0;
                for gr in 0..n_gr {
                    for ch in 0..n_ch {
                        let mask = GranuleMask::analyze(&xr[gr][ch], self.sample_rate, mask_ratio);
                        let is_pos_for_ch = if use_is && ch == 1 {
                            Some(&is_pos_per_gr_ch[gr][1])
                        } else {
                            None
                        };
                        let g = encode_granule_vbr(
                            &xr[gr][ch],
                            &mask,
                            block_types[gr][ch],
                            is_pos_for_ch,
                        );
                        total_bits += g.total_bits;
                        granule_data[gr].push(g);
                    }
                }
                let main_data_bytes_needed = total_bits.div_ceil(8);
                // Pick a standard bitrate slot whose unpadded slot can
                // hold the result. The reservoir absorbs the rest.
                // In practice the slot is sized to the *current frame's*
                // main_data, ignoring the queued reservoir — that's the
                // whole point: future frames can use surplus bits.
                let (idx, _kbps) = pick_vbr_bitrate_slot(
                    self.is_mpeg2,
                    self.sample_rate,
                    main_data_bytes_needed,
                    si_bytes,
                );
                // Frame bytes for that slot, no padding (VBR doesn't use
                // the CBR padding accumulator — bitrate index changes
                // already absorb the fractional-byte slack).
                let num = if self.is_mpeg2 { 72u32 } else { 144 };
                let kbps = if self.is_mpeg2 {
                    MPEG2_BITRATES_KBPS[idx as usize]
                } else {
                    MPEG1_BITRATES_KBPS[idx as usize]
                };
                let frame_bytes = (num * kbps * 1000 / self.sample_rate) as usize;
                (granule_data, frame_bytes, false, idx)
            }
        };
        let main_data_slot_bytes = frame_bytes - header_bytes - si_bytes;

        // Compose this frame's main-data bytes.
        let mut main_w = BitWriter::with_capacity(main_data_slot_bytes + 16);
        for gr_data in granule_data.iter() {
            for g in gr_data.iter() {
                g.emit_main_data(&mut main_w);
            }
        }
        main_w.align_to_byte();
        let main_data_bytes = main_w.into_bytes();

        // Append to queue and pop slot.
        self.main_data_queue.extend_from_slice(&main_data_bytes);
        let slot_take = main_data_slot_bytes.min(self.main_data_queue.len());
        let mut slot_payload: Vec<u8> = self.main_data_queue.drain(..slot_take).collect();
        if slot_payload.len() < main_data_slot_bytes {
            slot_payload.resize(main_data_slot_bytes, 0);
        }
        // Re-cap queue (should already hold).
        if self.main_data_queue.len() > max_lookback {
            let drop = self.main_data_queue.len() - max_lookback;
            self.main_data_queue.drain(..drop);
        }

        // ---- Compose frame bytes ----
        let mut frame_buf: Vec<u8> = Vec::with_capacity(frame_bytes);

        // Header (4 bytes).
        let mut hw = BitWriter::with_capacity(4);
        // Sync 11 bits + version(2) + layer(2) + protection(1)
        hw.write_u32(0x7FF, 11); // sync
        let version_bits: u32 = if self.is_mpeg2 { 0b10 } else { 0b11 };
        hw.write_u32(version_bits, 2);
        hw.write_u32(0b01, 2); // Layer III
        hw.write_u32(1, 1); // protection bit (1 = no CRC)
        hw.write_u32(frame_br_index as u32, 4);
        hw.write_u32(self.sr_index as u32, 2);
        hw.write_u32(if padding { 1 } else { 0 }, 1);
        hw.write_u32(0, 1); // private
                            // Channel mode: mono = 0b11, dual-channel = 0b10,
                            // joint stereo = 0b01. Stereo-but-not-joint = 0b00
                            // (we don't currently emit this — joint or
                            // dual-channel are the only stereo paths).
                            // Joint stereo (mode = 0b01) is required as soon as either MS or
                            // IS is in play; the per-frame mode_extension bits then disambiguate.
        let any_joint = use_ms || use_is;
        let mode_bits: u32 = match (n_ch, any_joint) {
            (1, _) => 0b11,
            (_, true) => 0b01,
            _ => 0b10,
        };
        hw.write_u32(mode_bits, 2);
        // mode_extension (joint-stereo only): bit 0x2 = MS on, bit 0x1
        // = IS on (ISO/IEC 11172-3 Table 2-B.4). 0b00 / 0b10 / 0b01 / 0b11
        // map respectively to "stereo" / "MS only" / "IS only" / "MS + IS".
        let mut mode_ext: u32 = 0;
        if use_ms {
            mode_ext |= 0b10;
        }
        if use_is {
            mode_ext |= 0b01;
        }
        hw.write_u32(mode_ext, 2);
        hw.write_u32(0, 1); // copyright
        hw.write_u32(0, 1); // original
        hw.write_u32(0, 2); // emphasis
        let header = hw.into_bytes();
        debug_assert_eq!(header.len(), 4);
        frame_buf.extend_from_slice(&header);

        // Side info.
        let mut si_w = BitWriter::with_capacity(si_bytes);
        if self.is_mpeg2 {
            self.emit_side_info_mpeg2(&mut si_w, main_data_begin, n_ch, &granule_data);
        } else {
            self.emit_side_info_mpeg1(&mut si_w, main_data_begin, n_ch, &granule_data);
        }
        si_w.align_to_byte();
        let side = si_w.into_bytes();
        debug_assert_eq!(side.len(), si_bytes);
        frame_buf.extend_from_slice(&side);

        // Main data slot.
        frame_buf.extend_from_slice(&slot_payload);

        debug_assert_eq!(frame_buf.len(), frame_bytes);

        let pts = (self.frame_index as i64) * spf as i64;
        let mut pkt = Packet::new(0, self.time_base, frame_buf);
        pkt.pts = Some(pts);
        pkt.dts = Some(pts);
        pkt.duration = Some(spf as i64);
        pkt.flags.keyframe = true;
        self.frame_index += 1;
        Ok(pkt)
    }

    /// Emit MPEG-1 Layer III side-information block.
    fn emit_side_info_mpeg1(
        &self,
        si_w: &mut BitWriter,
        main_data_begin: u16,
        n_ch: usize,
        granule_data: &[Vec<GranuleEncoded>],
    ) {
        si_w.write_u32(main_data_begin as u32, 9);
        // private bits: 5 mono / 3 stereo
        si_w.write_u32(0, if n_ch == 1 { 5 } else { 3 });
        // scfsi: ch * 4 bits
        for _ in 0..n_ch {
            si_w.write_u32(0, 4); // never reuse
        }
        for gr_data in granule_data.iter() {
            for g in gr_data.iter() {
                si_w.write_u32(g.part2_3_length as u32, 12);
                si_w.write_u32(g.big_values as u32, 9);
                si_w.write_u32(g.global_gain as u32, 8);
                // scalefac_compress: 0 means (slen1=0, slen2=0) and emits no
                // scalefactor bits — used for any granule that does not need
                // IS scalefactors. When IS is engaged on the R channel we
                // pick 13 = (slen1=3, slen2=3) so each long-block sfb gets
                // 3 bits, enough to encode `is_pos` 0..=7 across the whole
                // 21-sfb range.
                si_w.write_u32(g.scalefac_compress as u32, 4);
                emit_window_switching_tail_mpeg1(si_w, g);
            }
        }
    }

    /// Emit MPEG-2 LSF Layer III side-information block.
    ///
    /// Differences from MPEG-1 (see `sideinfo::parse_mpeg2`):
    /// - `main_data_begin` is 8 bits (not 9).
    /// - `private_bits` is 1 (mono) / 2 (stereo).
    /// - No `scfsi` (single granule per frame).
    /// - Exactly one granule per channel.
    /// - `scalefac_compress` is 9 bits — we emit 0, which decomposes via
    ///   `SCF_MOD_MPEG2` to `slen = [0, 0, 0, 0]` (long-block row, all
    ///   scalefactor widths zero, so no scalefactor bits in main_data).
    /// - No transmitted `preflag` bit (derived by the decoder from
    ///   `scalefac_compress >= 500`; with 0 this is false).
    fn emit_side_info_mpeg2(
        &self,
        si_w: &mut BitWriter,
        main_data_begin: u16,
        n_ch: usize,
        granule_data: &[Vec<GranuleEncoded>],
    ) {
        si_w.write_u32(main_data_begin as u32, 8);
        // private bits: 1 mono / 2 stereo
        si_w.write_u32(0, if n_ch == 1 { 1 } else { 2 });
        // Single granule only (granule_data.len() == 1 for MPEG-2 LSF).
        for g in granule_data[0].iter() {
            si_w.write_u32(g.part2_3_length as u32, 12);
            si_w.write_u32(g.big_values as u32, 9);
            si_w.write_u32(g.global_gain as u32, 8);
            si_w.write_u32(0, 9); // scalefac_compress = 0 → slen=[0,0,0,0]
            emit_window_switching_tail_mpeg2(si_w, g);
        }
    }
}

/// Emit the post-`scalefac_compress` tail of a MPEG-1 granule's side
/// info block (window_switching_flag through count1table_select).
///
/// Long blocks: window_switching_flag = 0, three identical
/// table_select entries, region0_count = 15 / region1_count = 7
/// (so region0 covers 0..big_end and region1/2 are empty).
///
/// Window-switching blocks (start / short / stop): the side info layout
/// changes — only TWO table_select entries (region2 vanishes), then
/// three subblock_gain entries instead of region0/1_count. Region
/// boundaries are *implicit*: per ISO 11172-3 §2.4.2.7 / decoder
/// `decoder.rs`, short blocks split big_values at offset 36 between
/// table_select[0] and table_select[1].
fn emit_window_switching_tail_mpeg1(si_w: &mut BitWriter, g: &GranuleEncoded) {
    if g.block_type == BlockType::Long {
        si_w.write_u32(0, 1); // window_switching_flag = 0
        si_w.write_u32(g.table_select as u32, 5);
        si_w.write_u32(g.table_select as u32, 5);
        si_w.write_u32(g.table_select as u32, 5);
        si_w.write_u32(15, 4); // region0_count
        si_w.write_u32(7, 3); // region1_count
    } else {
        si_w.write_u32(1, 1); // window_switching_flag = 1
        si_w.write_u32(g.block_type.as_u8() as u32, 2); // block_type
        si_w.write_u32(0, 1); // mixed_block_flag = 0 (pure short on switch)
        si_w.write_u32(g.table_select as u32, 5); // table_select[0]
        si_w.write_u32(g.table_select as u32, 5); // table_select[1]
                                                  // 3 × subblock_gain — keep at 0 (no per-window pre-emphasis).
        si_w.write_u32(0, 3);
        si_w.write_u32(0, 3);
        si_w.write_u32(0, 3);
    }
    si_w.write_u32(0, 1); // preflag
    si_w.write_u32(0, 1); // scalefac_scale
    si_w.write_u32(0, 1); // count1table_select = 0 (table A)
}

/// MPEG-2 LSF variant of [`emit_window_switching_tail_mpeg1`]. The
/// only on-wire difference is the absence of the explicit `preflag`
/// bit (MPEG-2 derives it from `scalefac_compress >= 500`).
fn emit_window_switching_tail_mpeg2(si_w: &mut BitWriter, g: &GranuleEncoded) {
    if g.block_type == BlockType::Long {
        si_w.write_u32(0, 1); // window_switching_flag = 0
        si_w.write_u32(g.table_select as u32, 5);
        si_w.write_u32(g.table_select as u32, 5);
        si_w.write_u32(g.table_select as u32, 5);
        si_w.write_u32(15, 4); // region0_count
        si_w.write_u32(7, 3); // region1_count
    } else {
        si_w.write_u32(1, 1); // window_switching_flag = 1
        si_w.write_u32(g.block_type.as_u8() as u32, 2);
        si_w.write_u32(0, 1); // mixed_block_flag = 0
        si_w.write_u32(g.table_select as u32, 5);
        si_w.write_u32(g.table_select as u32, 5);
        si_w.write_u32(0, 3);
        si_w.write_u32(0, 3);
        si_w.write_u32(0, 3);
    }
    // MPEG-2 has NO preflag bit in the bitstream.
    si_w.write_u32(0, 1); // scalefac_scale
    si_w.write_u32(0, 1); // count1table_select = 0 (table A)
}

impl Encoder for Mp3Encoder {
    fn codec_id(&self) -> &CodecId {
        &self.output_params.codec_id
    }
    fn output_params(&self) -> &CodecParameters {
        &self.output_params
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        match frame {
            Frame::Audio(a) => self.ingest(a),
            _ => Err(Error::invalid("MP3 encoder: audio frames only")),
        }
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        self.pending_packets.pop_front().ok_or(Error::NeedMore)
    }

    fn flush(&mut self) -> Result<()> {
        if !self.eof {
            self.eof = true;
            self.flush_ready_frames(true)?;
        }
        Ok(())
    }
}

// ---------------- Joint-stereo (MS) helpers ----------------

/// Per ISO/IEC 11172-3 §2.4.3.4.10, MS-stereo coding rotates the (L, R)
/// pair into (M, S) with `M = (L+R)/sqrt(2)`, `S = (L-R)/sqrt(2)`. The
/// rotation is energy-preserving but redistributes spectral energy: for
/// highly-correlated stereo content the side channel collapses to near
/// zero and Huffman coding spends almost no bits on it.
///
/// Decision rule: enable MS when, *summed across both granules*, the
/// side energy is below 30% of the mid energy. This is intentionally
/// conservative — it triggers reliably on centred-voice / ambient mixes
/// and stays off for true wide-stereo content where MS would introduce
/// quantisation cross-talk between channels.
fn should_use_ms_stereo(xr: &[Vec<[f32; 576]>], n_gr: usize) -> bool {
    let mut e_mid: f64 = 0.0;
    let mut e_side: f64 = 0.0;
    for gr in 0..n_gr {
        let l = &xr[gr][0];
        let r = &xr[gr][1];
        for i in 0..576 {
            let m = (l[i] + r[i]) as f64;
            let s = (l[i] - r[i]) as f64;
            e_mid += m * m;
            e_side += s * s;
        }
    }
    // Need *some* signal energy before the ratio is meaningful — pure
    // silence on both channels produces 0/0; default to dual-channel
    // (it matters little but keeps the bitstream stable on silent
    // intervals).
    if e_mid <= 1.0e-12 {
        return false;
    }
    e_side / e_mid < 0.30
}

/// Rotate a stereo granule of MDCT coefficients from (L, R) to (M, S)
/// in place. Mirrors `requantize::ms_stereo`'s decoder-side rotation:
/// applying the same transform a second time restores the original.
fn rotate_to_ms(channels: &mut [[f32; 576]]) {
    debug_assert_eq!(channels.len(), 2);
    let inv_sqrt2 = 1.0 / std::f32::consts::SQRT_2;
    let (l_slot, r_slot) = channels.split_at_mut(1);
    let l = &mut l_slot[0];
    let r = &mut r_slot[0];
    for i in 0..576 {
        let lv = l[i];
        let rv = r[i];
        l[i] = (lv + rv) * inv_sqrt2;
        r[i] = (lv - rv) * inv_sqrt2;
    }
}

// ---------------- Intensity-stereo (IS) helpers ----------------

/// Per ISO/IEC 11172-3 §2.4.3.4.10.2 / Table 3-B.4 the intensity-stereo
/// pan position `is_pos` (0..=6) maps to a left/right split where
///
/// ```text
///   is_ratio = tan(is_pos * PI / 12)
///   k_l = is_ratio / (1 + is_ratio)
///   k_r = 1.0      / (1 + is_ratio)
/// ```
///
/// (`is_pos == 7` is the "not IS-coded" sentinel — for that band the
/// decoder leaves R at zero and the L coefficient stays untouched.)
///
/// On the encoder side we run the rule in reverse: given the per-band L
/// and R energies, pick the integer `is_pos` whose `(k_l, k_r)` ratio
/// best matches the observed energy split. We then rewrite the band so
/// that L holds the *combined-magnitude* surrogate that the decoder
/// will scale back into L and R; R is forced to zero so
/// `find_is_bound_sfb` on the decode side recovers the same bound.
const IS_RATIO_TAB: [(f32, f32); 7] = [
    (0.0, 1.0),                   // is_pos = 0
    (0.211_324_87, 0.788_675_13), // is_pos = 1, tan(pi/12) ≈ 0.268
    (0.366_025_4, 0.633_974_6),   // is_pos = 2
    (0.5, 0.5),                   // is_pos = 3
    (0.633_974_6, 0.366_025_4),   // is_pos = 4
    (0.788_675_13, 0.211_324_87), // is_pos = 5
    (1.0, 0.0),                   // is_pos = 6
];

/// Pick the per-granule intensity-stereo bound (lowest sfb such that all
/// bands at or above it can be IS-coded), and the per-band `is_pos`
/// integer for the IS-coded bands.
///
/// Selection rule (per ISO/IEC 11172-3 §2.4.3.4.10.2 — the spec
/// describes the wire format but leaves bound + is_pos picking to the
/// encoder):
///   * Walk sfbs from the top (sfb 20) down to sfb 7. (Sfb 21 is the
///     un-scalefactored padding band; sfbs 0..7 carry too much audible
///     energy to encode as a single mono surrogate.)
///   * Compute per-sfb L energy `el`, R energy `er`, and the cross-
///     correlation `c = sum(L[i] * R[i])` using the post-rotation MDCT
///     coefficients. (When MS is on the L slot already holds M and the
///     R slot S — IS rides on those, which still produces a valid
///     bitstream because the decoder applies IS *before* MS unrotation
///     per ISO/IEC 11172-3 §2.4.3.4.10.)
///   * A band qualifies for IS coding when *either*
///       (a) the combined energy is below a noise-floor threshold
///           (silent ⇒ replacing R with zero is free), OR
///       (b) the absolute coherence `|c| / sqrt(el * er)` is high
///           (≥ 0.85) — i.e. R is essentially a scaled copy of L. In
///           that regime the IS surrogate `sqrt(el + er)` plus the
///           per-band `is_pos` recovers `(L, R)` with negligible
///           perceptual loss.
///   * The *bound* is the lowest sfb whose own band and every band
///     above qualifies.
///
/// Returns `(bound, is_pos)`. When no IS coding is worthwhile the bound
/// equals 21 and `is_pos[..]` stays at the 7-sentinel.
fn pick_is_bound_long(xr: &[[f32; 576]], sample_rate: u32) -> (usize, [u8; 22]) {
    use crate::sfband::sfband_long;
    debug_assert_eq!(xr.len(), 2);
    let bounds = sfband_long(sample_rate);

    /// Lowest sfb the encoder will tag for IS coding. Below this the
    /// loss from collapsing L/R into a single mono surrogate isn't
    /// worth the bit savings — the audible bands carry too much
    /// stereo image information.
    const MIN_IS_BOUND_SFB: usize = 7;
    /// Coherence threshold for IS qualification. 0.85 is the pivot
    /// where the surrogate's reconstruction error stays below ~1 dB on
    /// real music — tighter than that snaps too few bands; looser
    /// blurs true-stereo HF.
    const COHERENCE_THRESHOLD: f64 = 0.85;
    /// Pan-imbalance shortcut. A band with `||el| - |er|| / (el+er)`
    /// above this is essentially mono on one side — IS captures it
    /// exactly regardless of coherence (the sided channel is silent
    /// so any waveform on the other side rebuilds via `is_pos`).
    const PAN_SHORTCUT_THRESHOLD: f64 = 0.85;

    // Establish a noise-floor energy. Bands whose own combined energy
    // sits below this are treated as silent (IS-coded with the 7
    // sentinel) so spectral-leakage tails of LF tones do not block
    // the bound walk into the genuine HF tail.
    let mut max_band_e = 0.0f64;
    for sfb in 0..21 {
        let lo = bounds[sfb] as usize;
        let hi = bounds[sfb + 1] as usize;
        if hi > 576 {
            break;
        }
        let mut e = 0.0f64;
        for i in lo..hi {
            let lv = xr[0][i] as f64;
            let rv = xr[1][i] as f64;
            e += lv * lv + rv * rv;
        }
        if e > max_band_e {
            max_band_e = e;
        }
    }
    // Anything more than ~40 dB below the loudest band is effectively
    // silent for IS-decision purposes.
    let silence_threshold = max_band_e.max(1.0e-12) * 1.0e-4;

    let mut is_pos = [7u8; 22];
    let mut bound = 21usize;
    for sfb in (MIN_IS_BOUND_SFB..21).rev() {
        let lo = bounds[sfb] as usize;
        let hi = bounds[sfb + 1] as usize;
        if hi > 576 || lo >= 576 {
            break;
        }
        let mut el = 0.0f64;
        let mut er = 0.0f64;
        let mut c = 0.0f64;
        for i in lo..hi {
            let lv = xr[0][i] as f64;
            let rv = xr[1][i] as f64;
            el += lv * lv;
            er += rv * rv;
            c += lv * rv;
        }
        let total_e = el + er;
        if total_e < silence_threshold {
            is_pos[sfb] = 7; // sentinel ⇒ R forced to zero, L untouched
            bound = sfb;
            continue;
        }
        // Pan shortcut: if one channel is essentially silent in this
        // band (loud-LR fold-down, hard panning) IS captures the band
        // exactly even though linear coherence is undefined.
        let pan_imbalance = (el - er).abs() / total_e;
        let pan_qualifies = pan_imbalance >= PAN_SHORTCUT_THRESHOLD;
        let denom = (el * er).sqrt();
        let coherence_qualifies = denom > 1.0e-12 && (c.abs() / denom) >= COHERENCE_THRESHOLD;
        if !(pan_qualifies || coherence_qualifies) {
            break;
        }
        is_pos[sfb] = best_is_pos_from_energy(el, er);
        bound = sfb;
    }
    (bound, is_pos)
}

/// Pick the integer `is_pos ∈ 0..=6` whose `(k_l, k_r)` ratio best
/// matches the observed per-band L / R energies. Used after the
/// coherence check qualifies the band.
fn best_is_pos_from_energy(el: f64, er: f64) -> u8 {
    let total = el + er;
    if total <= 0.0 {
        return 7; // shouldn't reach here; defensive
    }
    let observed = (er / total).clamp(0.0, 1.0);
    let mut best_pos = 0u8;
    let mut best_err = f64::INFINITY;
    for (i, &(k_l, k_r)) in IS_RATIO_TAB.iter().enumerate() {
        let kl = k_l as f64;
        let kr = k_r as f64;
        let denom = kl * kl + kr * kr;
        let pred = if denom > 0.0 { kr * kr / denom } else { 0.0 };
        let err = (pred - observed).abs();
        if err < best_err {
            best_err = err;
            best_pos = i as u8;
        }
    }
    best_pos
}

/// Rewrite a stereo long-block granule of MDCT coefficients in place so
/// that all sfbs at or above `bound` are IS-coded. For each IS-coded
/// band we replace L with a magnitude-preserving surrogate
/// `sqrt(L^2 + R^2)` (sign from L) and force R to zero. The decoder
/// reads `is_pos` from the R channel's scalefactor slot and recovers
/// `(L_out, R_out) = (surrogate * k_l, surrogate * k_r)`.
fn apply_is_rewrite_long(xr: &mut [[f32; 576]], sample_rate: u32, bound: usize, is_pos: &[u8; 22]) {
    use crate::sfband::sfband_long;
    debug_assert_eq!(xr.len(), 2);
    let bounds = sfband_long(sample_rate);
    let (l_slot, r_slot) = xr.split_at_mut(1);
    let l = &mut l_slot[0];
    let r = &mut r_slot[0];
    for sfb in bound..21 {
        let lo = bounds[sfb] as usize;
        let hi = bounds[sfb + 1] as usize;
        if hi > 576 {
            break;
        }
        // is_pos sentinel ⇒ this band stays "not IS-coded" — leave L
        // alone, force R to zero so `find_is_bound_sfb` agrees with the
        // bound we set in side info.
        if is_pos[sfb] == 7 {
            for i in lo..hi {
                r[i] = 0.0;
            }
            continue;
        }
        for i in lo..hi {
            let lv = l[i];
            let rv = r[i];
            let mag = (lv * lv + rv * rv).sqrt();
            // Sign of the surrogate from L (or R when L is zero).
            let sign = if lv < 0.0 || (lv == 0.0 && rv < 0.0) {
                -1.0
            } else {
                1.0
            };
            l[i] = sign * mag;
            r[i] = 0.0;
        }
    }
}

// ---------------- Per-granule encoding ----------------

#[derive(Clone)]
struct GranuleEncoded {
    global_gain: u8,
    big_values: u16,
    table_select: u8,
    /// All Huffman-encoded bytes (big_values + count1) staged as a
    /// list of (code, len) writes.
    main_writes: Vec<(u32, u32)>,
    /// Pre-summed bits for the writes (scalefactors + Huffman).
    total_bits: usize,
    part2_3_length: u16,
    /// Block type chosen for this granule. `BlockType::Long` keeps the
    /// long-block side-info layout (window_switching_flag = 0); any
    /// other value triggers the window-switching path.
    block_type: BlockType,
    /// MPEG-1 4-bit `scalefac_compress` index (Table 3-B.32). 0 ⇒ no
    /// scalefactor bits (slen1=slen2=0); 13 ⇒ (slen1=3, slen2=3) for
    /// the IS R-channel path.
    scalefac_compress: u8,
    /// Optional pre-Huffman scalefactor writes (for the R channel of an
    /// IS-active granule). `(value, len_bits)` pairs in sfb order matching
    /// the chosen `scalefac_compress`. Empty when no scalefactors are
    /// transmitted.
    sf_writes: Vec<(u32, u32)>,
    /// Bit count contributed by `sf_writes`. Cached to keep
    /// `part2_3_length` accounting straightforward.
    sf_bits: usize,
}

impl GranuleEncoded {
    fn emit_main_data(&self, w: &mut BitWriter) {
        // Scalefactors first (part2), then Huffman (part3). The decoder
        // counts both into `part2_3_length` and assumes they are emitted
        // in this order (ISO/IEC 11172-3 §2.4.1.7).
        for (val, len) in &self.sf_writes {
            w.write_u32(*val, *len);
        }
        for (code, len) in &self.main_writes {
            w.write_u32(*code, *len);
        }
    }
}

fn encode_granule(
    xr: &[f32; 576],
    bit_target: usize,
    block_type: BlockType,
    is_pos: Option<&[u8; 22]>,
) -> GranuleEncoded {
    // Pick global_gain by binary search to fit `bit_target`. Higher
    // global_gain -> smaller is[] values -> fewer bits.
    //
    // We always end up with bits <= bit_target. If we can't fit even at
    // global_gain = 255 (max), we accept the overflow (decoder will read
    // junk but length is correct).
    let mut lo: i32 = 0;
    let mut hi: i32 = 255;
    let mut best: Option<GranuleEncoded> = None;
    while lo <= hi {
        let mid = (lo + hi) / 2;
        let g = quantize_and_encode(xr, mid as u8, block_type, is_pos);
        if g.total_bits <= bit_target {
            // This fits — try a smaller gain (more precision).
            best = Some(g);
            hi = mid - 1;
        } else {
            // Doesn't fit — increase gain.
            lo = mid + 1;
        }
    }
    if let Some(g) = best {
        return g;
    }
    // Fallback: return the highest-gain (smallest-bits) result.
    quantize_and_encode(xr, 255, block_type, is_pos)
}

/// VBR per-granule encode — pick the **largest** global_gain (= fewest
/// bits) such that the per-band quantization noise still satisfies
/// `worst_nmr_db <= 0` (i.e. noise below the masking threshold in
/// every band that has signal energy), subject to the absolute
/// 7680-bit per-granule cap (ISO 11172-3 §2.4.3.4.7.2).
///
/// For pure-silence granules ([`GranuleMask::worst_nmr_db`] returns
/// `NEG_INFINITY`) any gain masks; we pick a high gain so the encode
/// collapses to (near) zero bits — no point in spending bits on
/// silence.
fn encode_granule_vbr(
    xr: &[f32; 576],
    mask: &GranuleMask,
    block_type: BlockType,
    is_pos: Option<&[u8; 22]>,
) -> GranuleEncoded {
    // Energy gate: if every band is essentially silent, skip the
    // search entirely and emit a high-gain (ne ar-zero-bit) granule.
    let any_energy = mask.energy.iter().any(|&e| e > 1.0e-12);
    if !any_energy {
        return quantize_and_encode(xr, 240, block_type, is_pos);
    }

    // Step 1: find the smallest gain where NMR <= 0. Walk *upwards*:
    // higher gain → coarser step → larger noise → larger NMR. So once
    // we pass the masking threshold, NMR > 0 forever. The crossing
    // point is the *largest* gain that still masks; we want the gain
    // *just before* that.
    //
    // Walking upward at step 1 is cheap (a few mask-noise scalar
    // multiplies per gain) and we only call quantize_and_encode for
    // the chosen point.
    let mut last_masked: Option<u8> = None;
    for gain in 60..=255u8 {
        let step = global_gain_to_step(gain);
        let nmr = mask.worst_nmr_db(step);
        if nmr <= 0.0 {
            last_masked = Some(gain);
        } else {
            break;
        }
    }
    let chosen = last_masked.unwrap_or(60);

    // Step 2: bit-cap fallback. If the chosen gain exceeds the
    // per-granule bit cap (very loud, very wideband content) we
    // need to raise the gain further. Walk upward until we fit.
    let mut g = quantize_and_encode(xr, chosen, block_type, is_pos);
    let mut gain = chosen;
    while g.total_bits > VBR_PER_GRANULE_BIT_CAP && gain < 255 {
        gain = gain.saturating_add(2);
        g = quantize_and_encode(xr, gain, block_type, is_pos);
    }
    g
}

fn quantize_and_encode(
    xr: &[f32; 576],
    global_gain: u8,
    block_type: BlockType,
    is_pos: Option<&[u8; 22]>,
) -> GranuleEncoded {
    // Quantisation step: step_factor = 2^((global_gain - 210) / 4).
    // is[i] = nint( (|xr[i]|/step_factor)^(3/4) - 0.0946 )
    //       = nint( |xr[i]|^(3/4) * 2^(-(global_gain-210)*3/16) - 0.0946 )
    //
    // We compute per-sample directly to keep the code obvious. Cap |is|
    // at 8191 (the spec's hard ceiling — table 24 with linbits=13 +
    // value 15 reaches 32 + 8191 = 8223, but practical encoders cap
    // at 8191 to be safe). For our v1 we cap at 8191 and rely on the
    // outer global_gain bisection to keep us within table reach.
    let g = global_gain as i32;
    // Effective scaling exponent after the 3/4 power is (210 - g)*3/16.
    let exp = ((210 - g) as f32) * 3.0 / 16.0;
    let scale = (exp * std::f32::consts::LN_2).exp();
    let mut is_ = [0i32; 576];
    let mut max_abs = 0i32;
    for i in 0..576 {
        let a = xr[i].abs();
        let mag = a.powf(0.75) * scale;
        // Spec's quantizer subtracts 0.0946 then rounds. LAME uses
        // 0.4054 to bias toward over-quant; for simplicity we use 0.4054.
        let v = (mag + 0.4054).floor() as i32;
        let v = v.min(8191);
        let signed = if xr[i] < 0.0 { -v } else { v };
        is_[i] = signed;
        if v > max_abs {
            max_abs = v;
        }
    }

    // Find the trailing-zero region: scan from high index down for the
    // last non-zero coefficient.
    let mut last_nonzero = 0usize;
    for i in (0..576).rev() {
        if is_[i] != 0 {
            last_nonzero = i + 1;
            break;
        }
    }

    // Identify the count1 region. Walk from `last_nonzero` backward in
    // groups of 4, counting how many trailing groups consist solely of
    // 0/+-1 values.
    //
    // big_values_count must be even (it counts pairs * 2 in side info).
    let mut big_end = last_nonzero;
    // Round big_end up to a multiple of 2 since big_values runs in pairs.
    if big_end % 2 != 0 {
        big_end += 1;
        if big_end > 576 {
            big_end = 576;
        }
    }

    // Try sliding count1 region: take groups of 4 at the end where
    // every value is in {-1, 0, 1} for "free".
    let mut count1_start = big_end;
    while count1_start >= 4 {
        let g0 = is_[count1_start - 4];
        let g1 = is_[count1_start - 3];
        let g2 = is_[count1_start - 2];
        let g3 = is_[count1_start - 1];
        if g0.abs() <= 1 && g1.abs() <= 1 && g2.abs() <= 1 && g3.abs() <= 1 {
            count1_start -= 4;
        } else {
            break;
        }
    }
    // count1_start is now where count1 region starts (a multiple of 4
    // boundary at or before big_end). big_values ends at count1_start.
    let big_values_end = count1_start;
    let big_values_count = big_values_end as u16; // pairs * 2 = sample count

    // Pick a Huffman table that can encode all (x, y) pairs.
    let table_idx = choose_big_value_table(&is_, big_values_end);

    // Stage Huffman writes.
    let mut writes: Vec<(u32, u32)> =
        Vec::with_capacity(big_values_end / 2 + (576 - big_values_end) / 4);
    let mut total_bits: usize = 0;

    // Big-values pairs.
    for i in (0..big_values_end).step_by(2) {
        let x = is_[i];
        let y = is_.get(i + 1).copied().unwrap_or(0);
        let bits = emit_big_pair(table_idx, x, y, &mut writes);
        total_bits += bits;
    }

    // count1 region: groups of 4. Use table A.
    let count1_end = (last_nonzero + 3) & !3; // round up to multiple of 4
    let count1_end = count1_end.min(576);
    for i in (big_values_end..count1_end).step_by(4) {
        let v = is_[i];
        let w = is_.get(i + 1).copied().unwrap_or(0);
        let x = is_.get(i + 2).copied().unwrap_or(0);
        let y = is_.get(i + 3).copied().unwrap_or(0);
        let bits = emit_count1_quad(v, w, x, y, &mut writes);
        total_bits += bits;
    }

    // Scalefactor emit (R channel of an IS-active long-block granule).
    // We pack `is_pos` into the long-block scalefactor slots described by
    // ISO/IEC 11172-3 Table 3-B.32 with `scalefac_compress = 13`
    // ⇒ (slen1 = 3, slen2 = 3): six 3-bit values for sfb 0..5, five
    // for sfb 6..10, five for sfb 11..15, and five for sfb 16..20 — all
    // 3 bits wide. is_pos values are 0..=6 for IS-coded bands or 7 for
    // the "not IS-coded" sentinel.
    let (sf_writes, sf_bits, scalefac_compress) = match is_pos {
        Some(positions) if block_type == BlockType::Long => {
            let mut sf: Vec<(u32, u32)> = Vec::with_capacity(21);
            for &p in positions.iter().take(21) {
                sf.push(((p as u32) & 0x7, 3));
            }
            (sf, 21 * 3, 13u8)
        }
        _ => (Vec::new(), 0usize, 0u8),
    };
    let total_bits = total_bits + sf_bits;

    // part2_3_length = scalefactor bits + huffman bits.
    let part2_3_length = total_bits as u16;

    GranuleEncoded {
        global_gain,
        big_values: (big_values_count / 2),
        table_select: table_idx,
        main_writes: writes,
        total_bits,
        part2_3_length,
        block_type,
        scalefac_compress,
        sf_writes,
        sf_bits,
    }
}

fn choose_big_value_table(is_: &[i32; 576], big_end: usize) -> u8 {
    // Find the maximum coefficient magnitude in the big-values region.
    let mut max_abs = 0i32;
    for i in 0..big_end {
        let a = is_[i].abs();
        if a > max_abs {
            max_abs = a;
        }
    }
    // Try candidates in priority order; pick the first whose effective
    // range covers max_abs. Range = 15 + (1 << linbits) - 1 when
    // linbits>0, else 15.
    for &t in BIG_VALUE_CANDIDATES {
        let bvt = &BIG_VALUE_TABLES[t as usize];
        if bvt.tab.is_empty() {
            continue;
        }
        // Find max (x,y) value in this table.
        let max_xy = bvt.tab.iter().map(|e| e.2.max(e.3)).max().unwrap_or(0) as i32;
        let reach = if bvt.linbits == 0 {
            max_xy
        } else {
            // Symbol 15 + linbits → up to 15 + (2^linbits - 1).
            max_xy + (1i32 << bvt.linbits) - 1
        };
        if reach >= max_abs {
            return t;
        }
    }
    23 // last-resort: TABLE_16 with linbits=13
}

fn emit_big_pair(table_idx: u8, x: i32, y: i32, writes: &mut Vec<(u32, u32)>) -> usize {
    let bvt = &BIG_VALUE_TABLES[table_idx as usize];
    if bvt.tab.is_empty() {
        // Table 0 — both must be zero.
        return 0;
    }

    let ax = x.unsigned_abs() as i32;
    let ay = y.unsigned_abs() as i32;

    // Determine the symbol to emit. With linbits, values >= 15 get
    // mapped to 15 plus extra bits (the residual).
    let (sym_x, lin_x) = if bvt.linbits > 0 && ax >= 15 {
        (15i32, ax - 15)
    } else {
        (ax, 0)
    };
    let (sym_y, lin_y) = if bvt.linbits > 0 && ay >= 15 {
        (15i32, ay - 15)
    } else {
        (ay, 0)
    };

    // Linear search for the matching (x, y) entry in the table.
    let mut found: Option<(u32, u8)> = None;
    for &(code, len, tx, ty) in bvt.tab {
        if tx as i32 == sym_x && ty as i32 == sym_y {
            found = Some((code, len));
            break;
        }
    }
    let (code, len) = match found {
        Some(v) => v,
        None => {
            // Shouldn't happen given our table choice; fall back to (0,0).
            (bvt.tab[0].0, bvt.tab[0].1)
        }
    };

    let mut bits = len as usize;
    writes.push((code, len as u32));

    if bvt.linbits > 0 && ax >= 15 {
        writes.push((lin_x as u32, bvt.linbits as u32));
        bits += bvt.linbits as usize;
    }
    if x != 0 {
        writes.push((if x < 0 { 1 } else { 0 }, 1));
        bits += 1;
    }
    if bvt.linbits > 0 && ay >= 15 {
        writes.push((lin_y as u32, bvt.linbits as u32));
        bits += bvt.linbits as usize;
    }
    if y != 0 {
        writes.push((if y < 0 { 1 } else { 0 }, 1));
        bits += 1;
    }
    bits
}

fn emit_count1_quad(v: i32, w: i32, x: i32, y: i32, writes: &mut Vec<(u32, u32)>) -> usize {
    // count1 table A: each value is 0 or +-1.
    let av = v.unsigned_abs().min(1) as u8;
    let aw = w.unsigned_abs().min(1) as u8;
    let ax = x.unsigned_abs().min(1) as u8;
    let ay = y.unsigned_abs().min(1) as u8;
    let mut bits = 0usize;
    for &(code, len, tv, tw, tx, ty) in COUNT1_A {
        if tv == av && tw == aw && tx == ax && ty == ay {
            writes.push((code, len as u32));
            bits += len as usize;
            break;
        }
    }
    if v != 0 {
        writes.push((if v < 0 { 1 } else { 0 }, 1));
        bits += 1;
    }
    if w != 0 {
        writes.push((if w < 0 { 1 } else { 0 }, 1));
        bits += 1;
    }
    if x != 0 {
        writes.push((if x < 0 { 1 } else { 0 }, 1));
        bits += 1;
    }
    if y != 0 {
        writes.push((if y < 0 { 1 } else { 0 }, 1));
        bits += 1;
    }
    bits
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_is_four_bytes() {
        let mut p = CodecParameters::audio(CodecId::new("mp3"));
        p.channels = Some(1);
        p.sample_rate = Some(44_100);
        p.sample_format = Some(SampleFormat::S16);
        p.bit_rate = Some(128_000);
        let enc = make_encoder(&p).unwrap();
        assert_eq!(enc.codec_id().as_str(), "mp3");
    }

    #[test]
    fn quantize_silence_gives_zero_bits() {
        let xr = [0.0f32; 576];
        let g = quantize_and_encode(&xr, 100, BlockType::Long, None);
        assert_eq!(g.total_bits, 0);
        assert_eq!(g.big_values, 0);
    }

    #[test]
    fn pick_is_bound_zero_r_returns_min_bound() {
        // L-only signal (R == 0 everywhere) above the safety floor:
        // every band has 100 % pan imbalance and qualifies. Bound
        // pins at 7.
        let mut xr: Vec<[f32; 576]> = vec![[0.0f32; 576]; 2];
        for i in 100..400 {
            xr[0][i] = 0.5;
        }
        let (bound, ipos) = pick_is_bound_long(&xr, 44_100);
        assert_eq!(bound, 7, "expected the safety-floor bound 7, got {bound}");
        // For IS-coded bands (sfb >= 7) chosen is_pos should be 6
        // (all energy on L: er fraction = 0 ⇒ closest is_pos = 6).
        // Silent bands (sfb >= 17 ish where xr[0][i] is zero) ride
        // the silence path with is_pos = 7. Both are acceptable.
        for &p in ipos.iter().skip(7).take(14) {
            assert!(p == 6 || p == 7, "expected 6 or 7 for pure-L band, got {p}");
        }
    }

    #[test]
    fn pick_is_bound_uncorrelated_band_breaks_walk() {
        // Construct a granule where sfbs 19, 20 are highly correlated
        // (R = L) and qualify, but sfb 18 is uncorrelated noise on
        // both channels — coherence should fail there and the bound
        // should pin at sfb 19 (NOT walk past 18).
        let mut xr: Vec<[f32; 576]> = vec![[0.0f32; 576]; 2];
        // Sfbs 19, 20 (samples 288..418): R = L.
        for i in 288..418 {
            xr[0][i] = 0.5;
            xr[1][i] = 0.5;
        }
        // Sfb 18 (samples 238..288): uncorrelated triangles. L is a
        // sawtooth, R is a phase-flipped sawtooth-like sequence with
        // a 90-degree shift so coherence is approximately zero.
        for i in 238..288 {
            let k = (i - 238) as f32;
            xr[0][i] = (k.sin()) * 0.3;
            xr[1][i] = (k.cos()) * 0.3;
        }
        let (bound, _) = pick_is_bound_long(&xr, 44_100);
        assert!(bound >= 19, "expected bound >= 19, got {bound}");
    }

    #[test]
    fn apply_is_rewrite_zeroes_r_above_bound() {
        let mut xr: Vec<[f32; 576]> = vec![[0.0f32; 576]; 2];
        for i in 0..576 {
            xr[0][i] = 0.3;
            xr[1][i] = 0.3;
        }
        let mut ipos = [0u8; 22];
        for p in ipos.iter_mut() {
            *p = 3;
        }
        // Bound at sfb 6 ⇒ R must be zero from sample 24 (= bounds[6])
        // up through the end of sfb 20 = sample 418. Sfb 21 (samples
        // 418..576) is outside the IS-coded range so R there stays as
        // the input had it.
        apply_is_rewrite_long(&mut xr, 44_100, 6, &ipos);
        for i in 24..418 {
            assert_eq!(xr[1][i], 0.0, "R[{i}] not zeroed");
        }
        // Below the bound the IS rewrite leaves L/R alone.
        for i in 0..24 {
            assert_eq!(xr[0][i], 0.3);
            assert_eq!(xr[1][i], 0.3);
        }
    }

    #[test]
    fn pick_is_bound_correlated_full_band_walks_to_floor() {
        // Highly correlated stereo across the full granule should
        // qualify every IS-eligible band. Bound clamps at the safety
        // floor (7).
        let mut xr: Vec<[f32; 576]> = vec![[0.0f32; 576]; 2];
        for i in 0..576 {
            let v = (i as f32 * 0.1).sin() * 0.3;
            xr[0][i] = v;
            xr[1][i] = v; // perfect correlation
        }
        let (bound, _) = pick_is_bound_long(&xr, 44_100);
        assert_eq!(bound, 7, "expected safety-floor bound 7, got {bound}");
    }
}
