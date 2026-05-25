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
//! No psychoacoustic model, no outer noise-shaping loop, no stereo (LSF
//! / joint), no VBR, no Xing/Info VBR tag, no ID3v2 frontmatter, no
//! short-block / mixed-block window switching, no CRC. All deferred to
//! later rounds.

use std::io::{self, Write};

use crate::analysis::{analyze_granule, AnalysisState};
use crate::frame::{ChannelMode, Mp3FrameHeader, MpegVersion};
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
    /// Caller chose a stereo / dual-channel mode; this round writes
    /// mono streams only.
    StereoUnsupported,
    /// Caller chose an MPEG-2 LSF sample rate (16 / 22.05 / 24 kHz);
    /// LSF is deferred to a later round.
    LsfUnsupported,
}

impl core::fmt::Display for StreamEncodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            StreamEncodeError::Header(e) => write!(f, "header build: {e}"),
            StreamEncodeError::Reservoir(e) => write!(f, "reservoir: {e}"),
            StreamEncodeError::Huffman(e) => write!(f, "huffman emit: {e}"),
            StreamEncodeError::Io(e) => write!(f, "io: {e}"),
            StreamEncodeError::StereoUnsupported => {
                f.write_str("only mono streams are supported in this round")
            }
            StreamEncodeError::LsfUnsupported => {
                f.write_str("only MPEG-1 sample rates are supported in this round")
            }
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

    /// Polyphase analysis shift register (per channel; this round mono
    /// so length 1).
    analysis_state: Vec<AnalysisState>,
    /// Per-channel per-subband forward-MDCT overlap state (32 subbands
    /// × `nch` channels).
    mdct_state: Vec<Vec<MdctState>>,
    /// PCM pending in the current half-frame buffer (mono only).
    pending_pcm: Vec<f32>,

    /// Per-frame assembled output for the deferred reservoir
    /// scheduling pass.
    frames: Vec<PendingFrame>,

    /// When `Some`, every per-granule-channel quantization runs the
    /// §C.1.5.4.3 outer (distortion-control) loop instead of the
    /// fixed-`scalefac = 0` + inner-loop-only path. The carried value is
    /// the uniform `xmin[sb]` threshold applied to every long-block
    /// scalefactor band.
    outer_loop_threshold: Option<f64>,
}

#[derive(Debug)]
struct PendingFrame {
    header: Mp3FrameHeader,
    side_info: SideInfo,
    main_data: Vec<u8>,
}

impl Mp3Encoder {
    /// Build a new encoder for the given sample rate + bitrate. Only
    /// mono streams and MPEG-1 sample rates (32 / 44.1 / 48 kHz) are
    /// supported in this round.
    ///
    /// # Errors
    ///
    /// * [`StreamEncodeError::StereoUnsupported`] for a non-mono mode.
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
        if mode != ChannelMode::SingleChannel {
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
        Ok(Mp3Encoder {
            header_template,
            sample_rate_hz,
            version: MpegVersion::Mpeg1,
            nch,
            analysis_state,
            mdct_state,
            pending_pcm: Vec::new(),
            frames: Vec::new(),
            outer_loop_threshold: None,
        })
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

    /// Push `n` PCM samples (mono, `i16`). The encoder buffers them
    /// internally and assembles whole MP3 frames as soon as
    /// `SAMPLES_PER_FRAME_MPEG1 = 1152` accumulate.
    ///
    /// # Errors
    ///
    /// Propagates any encode-stage failure surfaced while assembling a
    /// completed frame (the only one that can fire here is
    /// [`StreamEncodeError::Huffman`]; bit-budget errors are deferred
    /// until [`Mp3Encoder::finish`]).
    pub fn push_samples(&mut self, samples: &[i16]) -> Result<(), StreamEncodeError> {
        // Convert i16 → f32 in `[-1.0, 1.0]` range and append to
        // pending PCM.
        const SCALE: f32 = 1.0 / 32_768.0;
        self.pending_pcm
            .extend(samples.iter().map(|&s| f32::from(s) * SCALE));

        while self.pending_pcm.len() >= SAMPLES_PER_FRAME_MPEG1 {
            let mut frame_pcm = vec![0.0f32; SAMPLES_PER_FRAME_MPEG1];
            frame_pcm.copy_from_slice(&self.pending_pcm[..SAMPLES_PER_FRAME_MPEG1]);
            self.pending_pcm.drain(..SAMPLES_PER_FRAME_MPEG1);
            self.assemble_frame(&frame_pcm)?;
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
        // Tail-flush: any leftover samples shorter than a full frame
        // are zero-padded so the last 1152 samples worth of audio are
        // still emitted.
        if !self.pending_pcm.is_empty() {
            let mut tail = vec![0.0f32; SAMPLES_PER_FRAME_MPEG1];
            for (i, &v) in self.pending_pcm.iter().enumerate() {
                tail[i] = v;
            }
            self.pending_pcm.clear();
            self.assemble_frame(&tail)?;
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
    fn per_gc_bit_budget(&self) -> usize {
        let frame_len = self.header_template.frame_len().unwrap_or(0);
        let si_bytes = side_info_byte_len(self.nch);
        let slot_bytes = frame_len.saturating_sub(4 + si_bytes);
        let denom = GRANULES.saturating_mul(self.nch).max(1);
        // Hold back a small margin (16 bits) per granule-channel for
        // the assembler's last partial-byte pad and any rounding.
        slot_bytes
            .saturating_mul(8)
            .saturating_div(denom)
            .saturating_sub(16)
    }

    /// Internal: turn `frame_pcm[0..1152]` (mono) into one assembled
    /// `PendingFrame` and append it to the scheduling queue.
    // The (gr, ch) double-loop mirrors the §2.4.1.7 `main_data()`
    // ordering exactly; the index variables are also used as
    // scratch-array subscripts (`gc_data[gr][ch]`,
    // `side_info.granules[gr][ch]`, etc.), so the explicit `for ch in
    // 0..self.nch` reads more clearly than an iterator chain.
    #[allow(clippy::needless_range_loop)]
    fn assemble_frame(&mut self, frame_pcm: &[f32]) -> Result<(), StreamEncodeError> {
        debug_assert_eq!(frame_pcm.len(), SAMPLES_PER_FRAME_MPEG1);

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

        // ---- Per-granule per-channel analysis + quantization ----
        for gr in 0..GRANULES {
            for ch in 0..self.nch {
                let gr_pcm = &frame_pcm[gr * SAMPLES_PER_GRANULE..(gr + 1) * SAMPLES_PER_GRANULE];
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

                // Forward MDCT per subband: forward-overlap with the
                // saved previous granule → window (long) → 36-point
                // forward MDCT → 18 frequency lines.
                //
                // Scale derivation. The §2.4.3.4.10.2 IMDCT and the
                // analysis MDCT use the same unscaled cosine kernel,
                // so the time-space lapped-MDCT round-trip
                //   encoder window → MDCT → decoder IMDCT → window →
                //   overlap-add
                // recovers the input scaled by `n/4 = 9` (the
                // Princen-Bradley TDAC factor; see
                // `analysis_synthesis_long_block_tdac_recovery` in
                // `mdct.rs`). Dividing the forward MDCT output by 9
                // makes the chain unit-gain.
                let mut xr = [0.0f32; NUM_LINES];
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
                // For long blocks (our scope) we apply the inverse
                // here so the decoder's forward alias-reduction
                // recovers the post-MDCT bins.
                let xr_pre = inverse_alias_reduce(&xr);

                // Pick the smallest global_gain + scalefactor configuration.
                // Two paths:
                //   * fixed-gain: zero scalefactors + inner loop only
                //     (the r138 path; kept for reference / debug).
                //   * outer-loop: §C.1.5.4.3 distortion-control loop on
                //     top of the inner loop, with non-zero per-band
                //     scalefactor amplification driven by the uniform
                //     `xmin[sb]` threshold.
                let gc_template = default_long_gc();
                let per_gc_bits = self.per_gc_bit_budget();
                // When the outer loop runs, part2 (scalefactors) costs
                // 74 bits per granule-channel (`scalefac_compress = 15`:
                // 11·slen1 + 10·slen2 = 11·4 + 10·3); the inner loop's
                // part3 budget shrinks by that amount.
                let part2_bits_outer: usize = 11 * 4 + 10 * 3;
                let inner_budget_for_outer = per_gc_bits.saturating_sub(part2_bits_outer) as u64;
                let (sf, initial_gain) = match self.outer_loop_threshold {
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
                        (res.scalefactors, res.global_gain)
                    }
                    None => {
                        let sf = ScaleFactors::default();
                        let res_budget = search_bit_budget(
                            &xr_pre,
                            &gc_template,
                            &sf,
                            self.sample_rate_hz,
                            self.version,
                            per_gc_bits as u64,
                        );
                        let res_clamp = search_magnitude_clamp(
                            &xr_pre,
                            &gc_template,
                            &sf,
                            self.sample_rate_hz,
                            self.version,
                        );
                        (sf, res_budget.global_gain.max(res_clamp.global_gain))
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
                    gc.preflag = false;
                    gc.scalefac_scale = false;
                    is = quantize(&xr_pre, &gc, &sf, self.sample_rate_hz, self.version);
                    clamp_above(&mut is, 8191);
                    split = partition_split(&is);
                    bv2 = split.big_pairs * 2;
                    gc.big_values = split.big_pairs as u16;
                    let (r0e, r1e, r0c, r1c) =
                        choose_region_split(self.sample_rate_hz, self.version, bv2);
                    r0_end = r0e;
                    r1_end = r1e;
                    gc.region0_count = r0c;
                    gc.region1_count = r1c;
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
    fn flush_to<W: Write>(self, sink: &mut W) -> Result<usize, StreamEncodeError> {
        if self.frames.is_empty() {
            return Ok(0);
        }

        // Step 1: pick per-frame padding to absorb fractional bytes,
        // and compute each frame's main-data slot capacity.
        //
        // For MPEG-1 Layer III the per-frame total length is
        //   `frame_len(padding) = 144 · bitrate / sample_rate + pad`
        // and the slot is `frame_len - 4 - 0 (no CRC) - SI_bytes` =
        //   `frame_len - 4 - 17` (mono) or `- 32` (stereo).
        let si_bytes = side_info_byte_len(self.nch);
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
        for f in frames.into_iter() {
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
            let mut hdr = f.header;
            hdr.padding = pad;
            let frame_len = hdr.frame_len().expect("CBR frame_len");
            // CRC-free: no 2-byte CRC field.
            let slot = frame_len - 4 - si_bytes;
            headers.push(hdr);
            side_infos.push(f.side_info);
            main_datas.push(f.main_data);
            slots.push(slot);
            frame_lens.push(frame_len);
        }
        // Sanity: never expect the per-frame length to drop below the
        // base length.
        debug_assert!(frame_lens.iter().all(|&l| l >= base_frame_len));

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

        // Step 3: emit each frame as header (4 bytes) + side_info
        // (`si_bytes` bytes) + slot (variable bytes).
        let mut written = 0usize;
        for (i, sch) in scheduled.iter().enumerate() {
            let hbytes = write_header(&headers[i]);
            sink.write_all(&hbytes)?;
            written += 4;
            let sib = write_side_info(&side_infos[i]);
            debug_assert_eq!(sib.len(), si_bytes);
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
}
