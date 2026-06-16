//! `oxideav_core::Decoder` wiring for MPEG-1 Audio Layer III.
//!
//! Symmetric counterpart to [`crate::codec_encoder`]. Where the encoder
//! wrapper drives the existing Phase 2 [`Mp3Encoder`] through the
//! framework's frame-in / packet-out trait, this module drives the
//! existing Layer III decode primitives through the packet-in /
//! frame-out [`oxideav_core::Decoder`] trait.
//!
//! ## Trait-API adaptation
//!
//! The framework trait is *packet-in, frame-out*:
//!
//! * [`send_packet`](Decoder::send_packet) accepts one [`Packet`] whose
//!   `data` is **one complete MP3 frame** (header + optional CRC +
//!   side-info + main-data slot), matching what
//!   [`crate::Mp3Demuxer::next_packet`] emits.
//! * [`receive_frame`](Decoder::receive_frame) returns one
//!   [`AudioFrame`] holding planar S16 PCM for that frame's granules
//!   (MPEG-1 = two granules × 576 samples = 1152 samples per channel,
//!   MPEG-2 LSF = one granule × 576 samples = 576 samples per channel).
//! * [`flush`](Decoder::flush) drains any buffered frame and signals
//!   end-of-stream so subsequent `receive_frame` calls eventually
//!   return [`Error::Eof`].
//! * [`reset`](Decoder::reset) wipes all carry-over state — the bit
//!   reservoir, the IMDCT overlap memory, and the polyphase synthesis
//!   filterbank shift register — so the next `send_packet` starts as
//!   if no prior packets had been processed (the documented requirement
//!   in the trait contract: "zero any per-stream filter / predictor /
//!   overlap memory so the next `send_packet` decodes as if it were the
//!   first").
//!
//! The wrapper parses the packet's MP3 frame, walks the existing
//! [`decode_huffman`] → [`requantize`] → [`alias_reduce`] →
//! [`imdct_granule`] → [`synth_granule`] chain (per granule per
//! channel), and converts the float PCM run to interleaved `i16` little-
//! endian bytes for the returned [`AudioFrame`]. Per-frame PTS is taken
//! verbatim from the inbound packet when present.
//!
//! ## Scope
//!
//! This module wires the Layer III decode path — MPEG-1, MPEG-2 LSF, and
//! MPEG-2.5 (11.025 / 12 kHz), mono **and** stereo (independent, joint
//! MS, joint MS+intensity) — through the framework trait.
//!
//! * **Mono and stereo.** `channels == 1` or `2`. For stereo frames the
//!   per-channel state — `ImdctState` and `SynthState` — is carried
//!   in a two-element array, and the §2.4.3.4.9 stereo processing stage
//!   (`process_stereo`) runs between requantize and alias reduction per
//!   the established decode pipeline order. Mono behaviour is unchanged
//!   from earlier rounds (only the channel-0 slot of the per-channel
//!   state arrays is used).
//! * **MPEG-1 and MPEG-2 LSF.** Sample rates 32 / 44.1 / 48 kHz
//!   (MPEG-1, ISO/IEC 11172-3 §2.4.2.3) and 16 / 22.05 / 24 kHz
//!   (MPEG-2 lower-sampling-frequency, ISO/IEC 13818-3 §2.4.2.3).
//!   On LSF the §2.4.1.7 side-info layout is the single-granule form
//!   (`granule_count == 1`) and the §2.4.3.4 scalefactor decode uses
//!   the 9-bit `scalefac_compress` partitioning; both are already
//!   honoured by [`parse_side_info`] / [`decode_scalefactors`] and
//!   downstream by [`requantize`] / [`crate::process_stereo`], so
//!   widening the trait wrapper to LSF is just dropping the
//!   MPEG-1-only header guard and letting `si.granule_count` /
//!   `si.channels` drive the per-frame iteration count.
//! * **MPEG-2.5 (11.025 / 12 kHz).** The Fraunhofer-IIS extension
//!   reuses the §13818-3 LSF framing on the half-rate sample rates. At
//!   11.025 and 12 kHz the scalefactor-band tables are byte-identical to
//!   the in-repo 13818-3 22.05 / 24 kHz LSF tables (fully grounded —
//!   `docs/audio/mp3/mpeg2.5-scalefactor-bands.md`, #147/#151) and the
//!   sample-rate dispatch is grounded in the staged datavoyage header
//!   reference (`docs/audio/mp3/MPEG-2.5-GAP.md`), so these two rates
//!   decode through the identical LSF chain and are accepted. The
//!   **8 kHz** rate uses a distinct Fraunhofer SFB table with no in-repo
//!   sibling and no observer-trace fixture, so it remains rejected here
//!   pending the residual `MPEG-2.5-GAP.md` observer-trace item.
//! * **Layer III only.** Layer I / Layer II frames are rejected at the
//!   `send_packet` boundary.
//!
//! Output PCM follows the framework's `AudioFrame` convention: one
//! `data[plane]` entry per channel (planar layout), with each plane
//! holding little-endian `i16` samples. Mono output keeps the single
//! plane; stereo output writes two planes (`data[0]` = L, `data[1]` = R).
//! Per-channel sample count per frame is 1152 on MPEG-1 (two granules)
//! and 576 on MPEG-2 LSF (one granule).

use std::collections::VecDeque;

use oxideav_core::{
    AudioFrame, CodecCapabilities, CodecId, CodecInfo, CodecParameters, CodecRegistry, CodecTag,
    Decoder, Error, Frame, Packet, Result, SampleFormat,
};

use crate::alias::alias_reduce;
use crate::codec_encoder::make_encoder;
use crate::demuxer::{CODEC_ID_STR, WAVE_FORMAT_MP3};
use crate::frame::{parse_header, Layer, Mp3FrameHeader, MpegVersion};
use crate::huffman::decode_huffman;
use crate::imdct::{imdct_granule, ImdctState};
use crate::reorder::reorder;
use crate::requantize::requantize;
use crate::scalefactors::{decode_scalefactors, MainDataReader, Reservoir};
use crate::side_info::parse_side_info;
use crate::stream_encoder::SAMPLES_PER_FRAME_MPEG1;
use crate::synth::{pcm_f32_to_i16, synth_granule, SynthState, PCM_PER_GRANULE};

/// Build a boxed MPEG-1 / MPEG-2 LSF Audio Layer III [`Decoder`] from
/// `params`.
///
/// `params.sample_rate` (32_000 / 44_100 / 48_000 for MPEG-1, or
/// 16_000 / 22_050 / 24_000 for MPEG-2 LSF) and `params.channels`
/// (1 or 2) configure the returned decoder's stream parameters; the
/// actual per-frame sample rate, channel count, and MPEG version are
/// re-derived from each MP3 frame header on `send_packet`, so the
/// values supplied here are a hint used only to construct the
/// `output_params()` shape.
///
/// # Errors
///
/// Returns [`Error::invalid`] when `channels` is supplied and not 1 or
/// 2. MPEG-1 Layer III carries at most two channels per §2.4.2.1, so
/// `channels >= 3` is unrepresentable on the wire and is rejected at
/// build time. `sample_rate` is optional (defaults to 44_100 when
/// absent): the real value is re-read from every frame header anyway.
pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    let channels = params.channels.unwrap_or(1);
    if channels != 1 && channels != 2 {
        return Err(Error::invalid(format!(
            "oxideav-mp3: decoder supports 1 or 2 channels (channels={channels})"
        )));
    }
    let sample_rate = params.sample_rate.unwrap_or(44_100);

    let mut out_params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
    out_params.sample_rate = Some(sample_rate);
    out_params.channels = Some(channels);
    out_params.sample_format = Some(SampleFormat::S16);
    out_params.tag = Some(CodecTag::wave_format(WAVE_FORMAT_MP3));

    Ok(Box::new(Mp3CoreDecoder::new(
        CodecId::new(CODEC_ID_STR),
        out_params,
    )))
}

/// Packet-to-frame adaptor that wires the existing Layer III decode
/// chain into the framework [`Decoder`] trait.
///
/// State carried across packets:
///
/// * `reservoir` — the §2.4.2.7 main-data bit reservoir: each frame's
///   main-data slot is appended, and per-frame decode reads back through
///   the `main_data_begin` lookback from prior frames.
/// * `imdct_state` — the §2.4.3.4.10.4 overlap-add memory: each
///   granule's IMDCT saves a second-half tail consumed by the next
///   granule.
/// * `synth_state` — the §2.4.3.2 polyphase synthesis filterbank's
///   1024-value shift register: each granule's 18 row passes update it.
///
/// `pending_frames` queues at-most-one [`AudioFrame`] produced by the
/// previous `send_packet`; `receive_frame` pops it. The
/// reservoir-frame-prefix issue (a frame whose `main_data_begin >
/// available reservoir bytes` cannot be decoded yet) is reported as
/// [`Error::NeedMore`] from `receive_frame` after `send_packet` declines
/// to produce a frame for it.
pub struct Mp3CoreDecoder {
    codec_id: CodecId,
    output: CodecParameters,
    reservoir: Reservoir,
    /// Per-channel IMDCT overlap memory. Index `[0]` is always used;
    /// `[1]` is touched only when a stereo header parses on a packet.
    imdct_state: [ImdctState; 2],
    /// Per-channel polyphase synthesis shift register. Same indexing
    /// convention as `imdct_state`.
    synth_state: [SynthState; 2],
    pending_frames: VecDeque<AudioFrame>,
    /// Set once [`Decoder::flush`] has been called; `receive_frame`
    /// returns [`Error::Eof`] after `pending_frames` is empty.
    eof: bool,
}

impl std::fmt::Debug for Mp3CoreDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mp3CoreDecoder")
            .field("codec_id", &self.codec_id)
            .field("pending_frames", &self.pending_frames.len())
            .field("eof", &self.eof)
            .finish()
    }
}

impl Mp3CoreDecoder {
    fn new(codec_id: CodecId, output: CodecParameters) -> Self {
        Self {
            codec_id,
            output,
            reservoir: Reservoir::new(),
            imdct_state: [ImdctState::new(), ImdctState::new()],
            synth_state: [SynthState::new(), SynthState::new()],
            pending_frames: VecDeque::new(),
            eof: false,
        }
    }

    /// Decode one MP3 frame from `packet` (header + optional CRC +
    /// side-info + main-data slot). Returns the decoded
    /// [`AudioFrame`] on success, or `Ok(None)` if the frame's
    /// `main_data_begin` lookback exceeds the bytes currently in the
    /// reservoir (caller should request more packets — but in practice
    /// this only happens for the first ~2 frames of a freshly-opened
    /// stream).
    fn decode_packet(&mut self, packet: &Packet) -> Result<Option<AudioFrame>> {
        let bytes = &packet.data;
        if bytes.len() < 4 {
            return Err(Error::invalid("oxideav-mp3: packet shorter than 4 bytes"));
        }
        let hdr = parse_header(&bytes[..4])
            .map_err(|e| Error::other(format!("oxideav-mp3: header parse: {e:?}")))?;
        if hdr.layer != Layer::LayerIII {
            return Err(Error::unsupported(format!(
                "oxideav-mp3: decoder requires Layer III (got {:?})",
                hdr.layer
            )));
        }
        match hdr.version {
            MpegVersion::Mpeg1 | MpegVersion::Mpeg2 => {}
            MpegVersion::Mpeg25 => {
                // MPEG-2.5 framing parses (§MPEG-2.5 step 25) and the
                // whole LSF decode chain (side-info → scalefactors →
                // Huffman → requantize → reorder → stereo → IMDCT →
                // synthesis) already routes the extension sample rates
                // through the §13818-3 LSF path. The remaining question
                // is the scalefactor-band table grounding, which differs
                // by rate (`docs/audio/mp3/mpeg2.5-scalefactor-bands.md`,
                // #147/#151):
                //
                // * **11.025 kHz** and **12 kHz**: the long+short SFB
                //   tables are *byte-identical* to the in-repo ISO/IEC
                //   13818-3 22.05 / 24 kHz LSF Table B.2 entries — fully
                //   grounded in the staged 13818-3 PDF. The header
                //   `id`-field → sample-rate dispatch (`00→11025`,
                //   `01→12000`, `10→8000`) is grounded in the staged
                //   datavoyage header reference (`MPEG-2.5-GAP.md`). These
                //   two rates decode through the identical chain MPEG-2
                //   LSF uses, so they are accepted here.
                // * **8 kHz**: the SFB table is a distinct Fraunhofer
                //   table with no in-repo half-rate sibling and no
                //   observer-trace fixture, so it remains
                //   published-factual-but-ungrounded. Decoding it would
                //   emit PCM whose band layout we cannot yet attest to
                //   from the in-repo specs alone, so 8 kHz stays gated on
                //   the residual `MPEG-2.5-GAP.md` observer-trace item.
                if hdr.sample_rate_hz == 8_000 {
                    return Err(Error::unsupported(
                        "oxideav-mp3: MPEG-2.5 8 kHz trait decode pending observer-trace \
                         grounding of the 8 kHz scalefactor-band table",
                    ));
                }
            }
        }
        let nch = hdr.channel_count() as usize;
        if nch != 1 && nch != 2 {
            return Err(Error::unsupported(format!(
                "oxideav-mp3: unsupported channel count {nch} (expected 1 or 2)"
            )));
        }
        // §2.4.1.3 frame length. For a fixed-bitrate header (`bitrate_index
        // ∈ 1..=14`) the length is derivable from the §2.4.2.3 formula and
        // we bounds-check the packet against it. For a **free-format**
        // header (`bitrate_index == 0`, §2.4.2.3) the standard does not
        // give a closed-form length: the encoder picked a constant
        // frame length the bitstream itself doesn't encode, and the
        // framer (demuxer / caller) recovers it as the distance from one
        // syncword to the next. The trait contract is that `packet.data`
        // is exactly one complete MP3 frame, so for free format the
        // authoritative length is simply the packet length. The whole
        // downstream decode (side-info → Huffman → requantize → IMDCT →
        // synthesis) is driven by `part2_3_length` from the side-info,
        // never by the bitrate, so a free-format frame decodes through the
        // identical chain once we know where its main-data slot ends.
        let frame_len = match hdr.frame_len() {
            Some(l) => {
                if bytes.len() < l {
                    return Err(Error::invalid(format!(
                        "oxideav-mp3: packet len {} < header-implied frame len {l}",
                        bytes.len()
                    )));
                }
                l
            }
            None => {
                // Free format (`bitrate_index == 0`): the packet itself is
                // the frame. A 4-byte sync alone carries no audio.
                if bytes.len() <= 4 {
                    return Err(Error::invalid(
                        "oxideav-mp3: free-format frame has no main-data slot",
                    ));
                }
                bytes.len()
            }
        };

        // Skip the optional 2-byte CRC slot that follows the 4-byte
        // header when `crc_protected` is set.
        let crc_bytes = if hdr.crc_protected { 2 } else { 0 };
        let si_start = 4 + crc_bytes;
        if bytes.len() < si_start {
            return Err(Error::invalid(
                "oxideav-mp3: frame truncated before side_info",
            ));
        }
        let si = parse_side_info(&hdr, &bytes[si_start..])
            .map_err(|e| Error::other(format!("oxideav-mp3: side_info parse: {e:?}")))?;
        let si_len = si.byte_len();
        let main_slot_start = si_start + si_len;
        if main_slot_start > frame_len {
            return Err(Error::invalid("oxideav-mp3: side_info overruns frame_len"));
        }
        let main_slot = &bytes[main_slot_start..frame_len];

        // Assemble main_data through the bit reservoir. If the look-
        // back is larger than the reservoir's current contents we
        // can't decode this frame yet — buffer its main-data and tell
        // the caller to send more packets.
        let run = match self
            .reservoir
            .assemble(usize::from(si.main_data_begin), main_slot)
        {
            Ok(run) => run,
            Err(_) => return Ok(None),
        };
        let fsf = decode_scalefactors(&hdr, &si, &run)
            .map_err(|e| Error::other(format!("oxideav-mp3: scalefactors: {e:?}")))?;

        let nch_si = si.channels as usize;
        // Per-channel planar PCM buffer, one Vec<i16> per channel.
        let mut pcm_planes: Vec<Vec<i16>> = (0..nch_si)
            .map(|_| Vec::with_capacity(SAMPLES_PER_FRAME_MPEG1))
            .collect();
        let mut bit_cursor = 0usize;
        for gr in 0..si.granule_count as usize {
            // Per-granule first pass: decode huffman + requantize for
            // every channel; collect the dequantized `xr` lines so a
            // stereo granule can run §2.4.3.4.9 joint-stereo processing
            // (MS / intensity) on the (L, R) pair before alias
            // reduction. Mono granules fall through with no stereo
            // step.
            let mut xr_per_ch: Vec<[f32; 576]> = (0..nch_si).map(|_| [0.0; 576]).collect();
            for (ch, xr_slot) in xr_per_ch.iter_mut().enumerate() {
                let gc = &si.granules[gr][ch];
                let mut r = MainDataReader::new(&run);
                let mut left = bit_cursor;
                while left >= 32 {
                    let _ = r.read(32);
                    left -= 32;
                }
                if left > 0 {
                    let _ = r.read(left as u32);
                }
                let part3_bits = u32::from(gc.part2_3_length);
                let is = decode_huffman(&mut r, gc, part3_bits, hdr.sample_rate_hz, hdr.version)
                    .map_err(|e| Error::other(format!("oxideav-mp3: huffman: {e:?}")))?;
                let sf = &fsf.granules[gr][ch];
                let xr = requantize(&is, gc, sf, hdr.sample_rate_hz, hdr.version);
                // §2.4.3.4.8 reorder: short-block (and mixed short-region)
                // lines leave requantize in the native `(sfb, window,
                // freqline)` Huffman interleave and must be rewritten into
                // subband-window-interleaved order before the §2.4.3.4.9
                // stereo stage (whose intensity/MS short-block path indexes
                // the reordered layout) and the §2.4.3.4.10 IMDCT (whose
                // short-block path gathers `lines[3·k + win]`). Long blocks
                // pass through unchanged.
                *xr_slot = reorder(&xr, gc, hdr.sample_rate_hz, hdr.version);
                bit_cursor += gc.part2_3_length as usize;
            }

            // §2.4.3.4.9 stereo processing — runs on stereo joint
            // granules only; independent-channel and mono granules pass
            // through with no change. `process_stereo` rewrites L / R in
            // place per the header's `mode_extension` bits using the
            // right channel's scalefactors / granule-channel side info
            // for the intensity bound.
            if nch_si == 2 && hdr.mode == crate::frame::ChannelMode::JointStereo {
                let (left_xr, right_xr) = xr_per_ch.split_at_mut(1);
                let left_arr: &mut [f32; 576] = &mut left_xr[0];
                let right_arr: &mut [f32; 576] = &mut right_xr[0];
                let right_sf = &fsf.granules[gr][1];
                let right_gc = &si.granules[gr][1];
                crate::stereo::process_stereo(
                    left_arr,
                    right_arr,
                    right_sf,
                    right_gc,
                    hdr.mode_extension,
                    hdr.sample_rate_hz,
                    hdr.version,
                );
            }

            // Per-channel alias reduction + IMDCT + synthesis.
            for (ch, xr_ch) in xr_per_ch.iter().enumerate() {
                let gc = &si.granules[gr][ch];
                let xar = alias_reduce(xr_ch, gc);
                let subband_time = imdct_granule(&xar, gc, &mut self.imdct_state[ch]);
                let pcm_f32 = synth_granule(&subband_time, &mut self.synth_state[ch]);
                for &p in pcm_f32.iter().take(PCM_PER_GRANULE) {
                    pcm_planes[ch].push(pcm_f32_to_i16(p));
                }
            }
        }

        // Per-channel sample count this frame: MPEG-1 Layer III = 1152
        // (two granules × 576 PCM samples each).
        let samples_per_ch = (si.granule_count as usize) * PCM_PER_GRANULE;
        // Pack each plane to little-endian i16 bytes.
        let data: Vec<Vec<u8>> = pcm_planes
            .iter()
            .map(|plane| {
                let mut bytes_le: Vec<u8> = Vec::with_capacity(plane.len() * 2);
                for s in plane {
                    bytes_le.extend_from_slice(&s.to_le_bytes());
                }
                bytes_le
            })
            .collect();
        let frame = AudioFrame {
            samples: samples_per_ch as u32,
            pts: packet.pts,
            data,
        };
        Ok(Some(frame))
    }

    /// Re-derive and update `self.output` from a freshly-parsed frame
    /// header. Stream-level parameter updates take effect after the
    /// first successfully decoded packet, so a caller that asks for the
    /// real sample rate via [`Decoder::codec_id`] / parameters after
    /// decoding has started gets the on-the-wire value rather than the
    /// at-construction hint.
    fn refresh_output_params(&mut self, hdr: &Mp3FrameHeader) {
        self.output.sample_rate = Some(hdr.sample_rate_hz);
        self.output.channels = Some(u16::from(hdr.channel_count()));
    }
}

impl Decoder for Mp3CoreDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        if self.eof {
            return Err(Error::other("oxideav-mp3: cannot send_packet after flush"));
        }
        if let Some(frame) = self.decode_packet(packet)? {
            // Update the cached output params from this frame's header.
            if let Ok(hdr) = parse_header(&packet.data[..4]) {
                self.refresh_output_params(&hdr);
            }
            self.pending_frames.push_back(frame);
        }
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        if let Some(audio) = self.pending_frames.pop_front() {
            return Ok(Frame::Audio(audio));
        }
        if self.eof {
            return Err(Error::Eof);
        }
        Err(Error::NeedMore)
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.reservoir = Reservoir::new();
        self.imdct_state = [ImdctState::new(), ImdctState::new()];
        self.synth_state = [SynthState::new(), SynthState::new()];
        self.pending_frames.clear();
        self.eof = false;
        Ok(())
    }
}

/// Install the MPEG-1 / MPEG-2 LSF Audio Layer III decoder factory
/// (alongside the existing MPEG-1 encoder factory from r140) into
/// `reg`.
///
/// Claims the WAVE format tag `0x0055` (MPEG Layer III) and the
/// Matroska codec id `A_MPEG/L3`. Both factories install on the same
/// `CodecInfo` so a single `register_codecs` call covers both
/// directions.
pub fn register_codecs(reg: &mut CodecRegistry) {
    let info = CodecInfo::new(CodecId::new(CODEC_ID_STR))
        .capabilities(
            CodecCapabilities::audio("mp3")
                .with_decode()
                .with_encode()
                .with_lossy(true),
        )
        .decoder(make_decoder)
        .encoder(make_encoder)
        .tags([
            CodecTag::wave_format(WAVE_FORMAT_MP3),
            CodecTag::matroska("A_MPEG/L3"),
        ]);
    reg.register(info);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream_encoder::{Mp3Encoder, SAMPLES_PER_FRAME_MPEG1};
    use oxideav_core::{Frame, TimeBase};
    use std::f32::consts::PI;

    /// Build a complete MP3 byte stream by running the direct
    /// `Mp3Encoder` over `pcm`.
    fn encode_to_mp3(pcm: &[i16], sample_rate: u32, bitrate_kbps: u32) -> Vec<u8> {
        let mut enc = Mp3Encoder::new(
            bitrate_kbps,
            sample_rate,
            crate::frame::ChannelMode::SingleChannel,
        )
        .expect("Mp3Encoder build");
        enc.push_samples(pcm).expect("push_samples");
        let mut bytes: Vec<u8> = Vec::new();
        enc.finish(&mut bytes).expect("finish");
        bytes
    }

    /// Build an MP3 byte stream whose every granule is a `block_type == 2`
    /// short block (via the encoder's force-short testing toggle), so the
    /// decode path under test exercises the §2.4.3.4.8 reorder stage.
    fn encode_to_mp3_short(pcm: &[i16], sample_rate: u32, bitrate_kbps: u32) -> Vec<u8> {
        let mut enc = Mp3Encoder::new(
            bitrate_kbps,
            sample_rate,
            crate::frame::ChannelMode::SingleChannel,
        )
        .expect("Mp3Encoder build");
        enc.force_short_blocks_for_testing(true)
            .expect("force-short toggle");
        enc.push_samples(pcm).expect("push_samples");
        let mut bytes: Vec<u8> = Vec::new();
        enc.finish(&mut bytes).expect("finish");
        bytes
    }

    /// Synthesize a mono sine in i16.
    fn sine_pcm(n: usize, freq_hz: f32, sr: f32, amp: f32) -> Vec<i16> {
        let two_pi = 2.0 * PI;
        let scale = amp * (i16::MAX as f32);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f32 / sr;
            let s = (two_pi * freq_hz * t).sin() * scale;
            out.push(s.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16);
        }
        out
    }

    /// Slice a contiguous MP3 byte stream into per-MP3-frame packets
    /// via the crate's own `FrameWalker`, mirroring what a real
    /// demuxer would emit.
    fn mp3_to_packets(bytes: &[u8], sample_rate: u32) -> Vec<Packet> {
        let tb = TimeBase::new(1, i64::from(sample_rate));
        let mut out = Vec::new();
        let mut pts: i64 = 0;
        for f in crate::frame::FrameWalker::new(bytes) {
            let mut pkt = Packet::new(0, tb, f.data.to_vec());
            pkt.pts = Some(pts);
            pkt.duration = Some(SAMPLES_PER_FRAME_MPEG1 as i64);
            out.push(pkt);
            pts += SAMPLES_PER_FRAME_MPEG1 as i64;
        }
        out
    }

    /// Drive the decode chain directly (matching the
    /// `decode_mp3_mono` helper used in
    /// `tests/encoder_trait_roundtrip.rs`) so the trait-wrapper output
    /// can be byte-exact compared.
    fn decode_direct(bytes: &[u8]) -> Vec<i16> {
        let mut reservoir = Reservoir::new();
        let mut synth_state = SynthState::new();
        let mut imdct_state = ImdctState::new();
        let mut out_pcm: Vec<i16> = Vec::new();
        for frame in crate::frame::FrameWalker::new(bytes) {
            let hdr = parse_header(&frame.data[..4]).unwrap();
            let si_start = 4 + if hdr.crc_protected { 2 } else { 0 };
            let si = parse_side_info(&hdr, &frame.data[si_start..]).unwrap();
            let si_len = si.byte_len();
            let main_slot = &frame.data[si_start + si_len..frame.data.len()];
            let run = reservoir
                .assemble(usize::from(si.main_data_begin), main_slot)
                .expect("reservoir");
            let fsf = decode_scalefactors(&hdr, &si, &run).unwrap();
            let mut bit_cursor = 0usize;
            for gr in 0..si.granule_count as usize {
                for ch in 0..si.channels as usize {
                    let gc = &si.granules[gr][ch];
                    let mut r = MainDataReader::new(&run);
                    let mut left = bit_cursor;
                    while left >= 32 {
                        let _ = r.read(32);
                        left -= 32;
                    }
                    if left > 0 {
                        let _ = r.read(left as u32);
                    }
                    let part3_bits = u32::from(gc.part2_3_length);
                    let is =
                        decode_huffman(&mut r, gc, part3_bits, hdr.sample_rate_hz, hdr.version)
                            .unwrap();
                    let sf = &fsf.granules[gr][ch];
                    let xr = requantize(&is, gc, sf, hdr.sample_rate_hz, hdr.version);
                    let xr = reorder(&xr, gc, hdr.sample_rate_hz, hdr.version);
                    let xar = alias_reduce(&xr, gc);
                    let st = imdct_granule(&xar, gc, &mut imdct_state);
                    let pcm_f32 = synth_granule(&st, &mut synth_state);
                    for &p in pcm_f32.iter().take(PCM_PER_GRANULE) {
                        out_pcm.push(pcm_f32_to_i16(p));
                    }
                    bit_cursor += gc.part2_3_length as usize;
                }
            }
        }
        out_pcm
    }

    fn build_decoder_params(sample_rate: u32) -> CodecParameters {
        let mut p = CodecParameters::audio(CodecId::new("mp3"));
        p.sample_rate = Some(sample_rate);
        p.channels = Some(1);
        p.sample_format = Some(SampleFormat::S16);
        p
    }

    #[test]
    fn make_decoder_accepts_mono_and_stereo() {
        let mut mono = CodecParameters::audio(CodecId::new("mp3"));
        mono.channels = Some(1);
        assert!(make_decoder(&mono).is_ok());
        let mut stereo = CodecParameters::audio(CodecId::new("mp3"));
        stereo.channels = Some(2);
        let dec = make_decoder(&stereo).expect("stereo decoder builds");
        assert_eq!(dec.codec_id().as_str(), "mp3");
    }

    #[test]
    fn make_decoder_rejects_unsupported_channel_count() {
        let mut p = CodecParameters::audio(CodecId::new("mp3"));
        p.channels = Some(3);
        assert!(make_decoder(&p).is_err());
        p.channels = Some(0);
        assert!(make_decoder(&p).is_err());
    }

    #[test]
    fn make_decoder_builds_and_reports_output_params() {
        let p = build_decoder_params(44_100);
        let dec = make_decoder(&p).expect("make_decoder");
        assert_eq!(dec.codec_id().as_str(), "mp3");
    }

    #[test]
    fn register_codecs_installs_both_factories() {
        let mut reg = CodecRegistry::new();
        register_codecs(&mut reg);
        let id = CodecId::new("mp3");
        assert!(reg.has_decoder(&id));
        assert!(reg.has_encoder(&id));
    }

    #[test]
    fn trait_decode_matches_direct_chain_byte_exact() {
        // Encode a known PCM through the direct stream encoder, slice
        // the output into per-frame packets, drive them through the
        // trait Decoder, and confirm the trait wrapper produces the
        // same i16 PCM bytes the direct decode chain produces on the
        // same input bytes. This is the test the round mandate calls
        // for: "drive a synthetic-or-staged MP3 byte stream through
        // the registered Decoder trait API, recover i16 PCM frames,
        // byte-exact match to what the direct decode chain produces."
        const SR: u32 = 44_100;
        let n = (SR as usize) / 4; // 250 ms
        let pcm = sine_pcm(n, 440.0, SR as f32, 0.5);
        let wire = encode_to_mp3(&pcm, SR, 128);
        assert!(wire.len() > 100, "encoded stream too small");

        // Direct-chain reference output.
        let direct = decode_direct(&wire);

        // Trait-driven decode.
        let mut dec = make_decoder(&build_decoder_params(SR)).expect("make_decoder");
        let mut trait_out: Vec<i16> = Vec::new();
        for pkt in mp3_to_packets(&wire, SR) {
            dec.send_packet(&pkt).expect("send_packet");
            loop {
                match dec.receive_frame() {
                    Ok(Frame::Audio(a)) => {
                        // Mono frame: data[0] is the interleaved S16
                        // byte run (interleaved == planar for mono).
                        for chunk in a.data[0].chunks_exact(2) {
                            trait_out.push(i16::from_le_bytes([chunk[0], chunk[1]]));
                        }
                    }
                    Ok(other) => panic!("non-audio frame: {other:?}"),
                    Err(Error::NeedMore) => break,
                    Err(e) => panic!("receive_frame: {e}"),
                }
            }
        }
        dec.flush().expect("flush");
        // Drain residue (should be none — flush is a no-op for the
        // mono-MP3 path because every send_packet produces at most
        // one frame inline).
        loop {
            match dec.receive_frame() {
                Ok(Frame::Audio(a)) => {
                    for chunk in a.data[0].chunks_exact(2) {
                        trait_out.push(i16::from_le_bytes([chunk[0], chunk[1]]));
                    }
                }
                Ok(other) => panic!("non-audio frame on flush: {other:?}"),
                Err(Error::Eof) => break,
                Err(Error::NeedMore) => break,
                Err(e) => panic!("post-flush receive_frame: {e}"),
            }
        }

        assert_eq!(
            trait_out.len(),
            direct.len(),
            "trait-driven sample count {} != direct-chain {}",
            trait_out.len(),
            direct.len()
        );
        // Byte-exact match: the trait wrapper must produce identical
        // i16 PCM samples to the direct chain on the same input.
        let mismatches = trait_out
            .iter()
            .zip(direct.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            mismatches, 0,
            "trait-driven decode differs from direct chain in {mismatches} samples"
        );
    }

    #[test]
    fn trait_decode_short_block_runs_reorder_and_is_not_silent() {
        // Regression test for the missing §2.4.3.4.8 reorder stage in the
        // trait decode path. Before the fix, `decode_packet` ran
        // requantize → (stereo) → alias → imdct with NO reorder, so a
        // short-block (`block_type == 2`) granule's frequency lines were
        // still in the native `(sfb, window, freqline)` Huffman interleave
        // when the IMDCT gathered `lines[3·k + win]` — reading the wrong
        // samples and producing corrupt PCM. Forcing every granule to a
        // short block (the encoder's force-short testing toggle) drives
        // that path.
        //
        // Two assertions: (1) the trait decoder is byte-exact with the
        // in-module `decode_direct` reference (which now also calls
        // reorder), and (2) the reconstructed PCM is finite, non-silent,
        // and crosses zero — a runaway / mis-gathered IMDCT would saturate
        // the i16 clamp or collapse to silence.
        const SR: u32 = 44_100;
        let n = (SR as usize) / 4; // 250 ms
        let pcm = sine_pcm(n, 440.0, SR as f32, 0.5);
        let wire = encode_to_mp3_short(&pcm, SR, 192);
        assert!(wire.len() > 100, "encoded short-block stream too small");

        // Confirm the stream really is short-block coded (otherwise this
        // test would silently pass on the long-block pass-through path).
        let mut short_granules = 0usize;
        for frame in crate::frame::FrameWalker::new(&wire) {
            let hdr = parse_header(&frame.data[..4]).unwrap();
            let si = parse_side_info(&hdr, &frame.data[4..]).unwrap();
            for gr in 0..si.granule_count as usize {
                for ch in 0..si.channels as usize {
                    let gc = &si.granules[gr][ch];
                    if gc.window_switching_flag
                        && gc.block_type == crate::side_info::BlockType::Short
                    {
                        short_granules += 1;
                    }
                }
            }
        }
        assert!(
            short_granules > 0,
            "force-short stream carried no short-block granules"
        );

        let direct = decode_direct(&wire);

        let mut dec = make_decoder(&build_decoder_params(SR)).expect("make_decoder");
        let mut trait_out: Vec<i16> = Vec::new();
        for pkt in mp3_to_packets(&wire, SR) {
            dec.send_packet(&pkt).expect("send_packet");
            loop {
                match dec.receive_frame() {
                    Ok(Frame::Audio(a)) => {
                        for chunk in a.data[0].chunks_exact(2) {
                            trait_out.push(i16::from_le_bytes([chunk[0], chunk[1]]));
                        }
                    }
                    Ok(other) => panic!("non-audio frame: {other:?}"),
                    Err(Error::NeedMore) => break,
                    Err(e) => panic!("receive_frame: {e}"),
                }
            }
        }

        assert_eq!(
            trait_out.len(),
            direct.len(),
            "short-block trait sample count {} != direct-chain {}",
            trait_out.len(),
            direct.len()
        );
        let mismatches = trait_out
            .iter()
            .zip(direct.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            mismatches, 0,
            "short-block trait decode differs from direct chain in {mismatches} samples"
        );

        // Non-silent / finite / zero-crossing witness.
        let energy: f64 = trait_out
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum::<f64>()
            / trait_out.len().max(1) as f64;
        assert!(
            energy.is_finite() && energy > 0.0,
            "short-block decode produced zero / non-finite energy ({energy})"
        );
        let zero_crossings = trait_out
            .windows(2)
            .filter(|w| (w[0] >= 0) != (w[1] >= 0))
            .count();
        assert!(
            zero_crossings > 10,
            "short-block decode had too few zero crossings ({zero_crossings})"
        );
    }

    #[test]
    fn receive_frame_before_send_returns_need_more() {
        let mut dec = make_decoder(&build_decoder_params(44_100)).expect("make_decoder");
        assert!(matches!(dec.receive_frame(), Err(Error::NeedMore)));
    }

    #[test]
    fn receive_frame_after_flush_returns_eof_once_drained() {
        let mut dec = make_decoder(&build_decoder_params(44_100)).expect("make_decoder");
        dec.flush().expect("flush");
        assert!(matches!(dec.receive_frame(), Err(Error::Eof)));
    }

    #[test]
    fn send_packet_after_flush_errors() {
        let mut dec = make_decoder(&build_decoder_params(44_100)).expect("make_decoder");
        dec.flush().expect("flush");
        let tb = TimeBase::new(1, 44_100);
        let pkt = Packet::new(0, tb, vec![0xFF, 0xFB, 0x90, 0x00]); // bogus, but enough header
        assert!(dec.send_packet(&pkt).is_err());
    }

    #[test]
    fn reset_clears_state_so_first_subsequent_packet_is_decodable() {
        // After feeding a partial stream and reset()-ing, a fresh
        // stream should decode from scratch.
        const SR: u32 = 44_100;
        let n = SR as usize / 10;
        let pcm = sine_pcm(n, 440.0, SR as f32, 0.5);
        let wire = encode_to_mp3(&pcm, SR, 128);
        let mut dec = make_decoder(&build_decoder_params(SR)).expect("make_decoder");
        // Feed a handful of packets, then reset.
        let pkts = mp3_to_packets(&wire, SR);
        for pkt in pkts.iter().take(2) {
            dec.send_packet(pkt).expect("send_packet");
            // Drain to keep pending_frames empty before reset.
            while dec.receive_frame().is_ok() {}
        }
        dec.reset().expect("reset");
        // After reset, a fresh stream from packet 0 should produce
        // exactly the same trait-driven byte stream as a fresh decoder.
        let mut a: Vec<i16> = Vec::new();
        let mut b: Vec<i16> = Vec::new();
        let mut dec2 = make_decoder(&build_decoder_params(SR)).expect("make_decoder");
        for pkt in &pkts {
            dec.send_packet(pkt).expect("send_packet (reset)");
            dec2.send_packet(pkt).expect("send_packet (fresh)");
            while let Ok(Frame::Audio(af)) = dec.receive_frame() {
                for chunk in af.data[0].chunks_exact(2) {
                    a.push(i16::from_le_bytes([chunk[0], chunk[1]]));
                }
            }
            while let Ok(Frame::Audio(af)) = dec2.receive_frame() {
                for chunk in af.data[0].chunks_exact(2) {
                    b.push(i16::from_le_bytes([chunk[0], chunk[1]]));
                }
            }
        }
        assert_eq!(a, b, "reset state did not produce a clean restart");
    }

    #[test]
    fn send_packet_rejects_mpeg25_8khz_pending_observer_trace() {
        // The 8 kHz MPEG-2.5 rate is the one residual gap: its
        // scalefactor-band table is a distinct Fraunhofer table with no
        // in-repo half-rate sibling and no observer-trace fixture
        // (`docs/audio/mp3/mpeg2.5-scalefactor-bands.md` §"8 kHz
        // provenance"; `MPEG-2.5-GAP.md`). Until that table is grounded
        // the trait decoder must reject 8 kHz frames rather than emit
        // PCM whose band layout we cannot attest to from the in-repo
        // specs alone. Build a real Fraunhofer MPEG-2.5 8 kHz header via
        // the crate's own header writer (24 kbps Layer III, mono) and
        // confirm an `Error::Unsupported` fires.
        use crate::encoder::{make_silent_header, write_header};
        use crate::frame::ChannelMode;
        let hdr = make_silent_header(24, 8_000, ChannelMode::SingleChannel)
            .expect("mpeg-2.5 8 kHz silent header build");
        assert_eq!(hdr.version, MpegVersion::Mpeg25);
        assert_eq!(hdr.sample_rate_hz, 8_000);
        let hdr_bytes = write_header(&hdr);
        let frame_len = hdr.frame_len().expect("mpeg-2.5 frame_len derivable");
        // Pad to the header-implied frame_len so the rejection arm
        // is reached before the truncation check (we want to assert
        // the 8 kHz guard fires, not the length guard).
        let mut payload = hdr_bytes.to_vec();
        payload.resize(frame_len, 0u8);
        let tb = TimeBase::new(1, 8_000);
        let pkt = Packet::new(0, tb, payload);
        let mut dec = make_decoder(&build_decoder_params(8_000)).expect("make_decoder");
        match dec.send_packet(&pkt) {
            Err(e) => {
                let msg = format!("{e}");
                assert!(
                    msg.contains("8 kHz") || msg.contains("observer-trace"),
                    "expected MPEG-2.5 8 kHz-pending error, got: {msg}"
                );
            }
            Ok(()) => panic!("send_packet must reject MPEG-2.5 8 kHz pending observer-trace"),
        }
    }

    #[test]
    fn trait_decode_mpeg25_11025_byte_exact_with_direct_chain() {
        // MPEG-2.5 at 11.025 kHz now decodes through the trait wrapper:
        // its scalefactor-band tables are byte-identical to the in-repo
        // ISO/IEC 13818-3 22.05 kHz LSF tables (fully grounded —
        // `docs/audio/mp3/mpeg2.5-scalefactor-bands.md`, #147/#151) and
        // the header `id`-field → sample-rate dispatch is grounded in
        // the staged datavoyage reference (`MPEG-2.5-GAP.md`). Encode a
        // real MPEG-2.5 11.025 kHz stream, then assert the trait wrapper
        // produces the identical i16 PCM as the in-module `decode_direct`
        // reference chain (same chain MPEG-1 / MPEG-2 LSF already use).
        const SR: u32 = 11_025;
        let n = SR as usize / 2; // 0.5 s
        let pcm = sine_pcm(n, 220.0, SR as f32, 0.5);
        let wire = encode_to_mp3(&pcm, SR, 32);
        assert!(wire.len() > 100, "encoded MPEG-2.5 stream too small");

        // Confirm the stream really is MPEG-2.5 at 11.025 kHz.
        let first = crate::frame::FrameWalker::new(&wire)
            .next()
            .expect("at least one frame");
        let hdr = parse_header(&first.data[..4]).unwrap();
        assert_eq!(hdr.version, MpegVersion::Mpeg25);
        assert_eq!(hdr.sample_rate_hz, SR);
        assert_eq!(hdr.samples_per_frame(), 576);

        let direct = decode_direct(&wire);

        let mut dec = make_decoder(&build_decoder_params(SR)).expect("make_decoder");
        let mut trait_out: Vec<i16> = Vec::new();
        for pkt in mp3_to_packets(&wire, SR) {
            dec.send_packet(&pkt).expect("send_packet");
            loop {
                match dec.receive_frame() {
                    Ok(Frame::Audio(a)) => {
                        for chunk in a.data[0].chunks_exact(2) {
                            trait_out.push(i16::from_le_bytes([chunk[0], chunk[1]]));
                        }
                    }
                    Ok(other) => panic!("non-audio frame: {other:?}"),
                    Err(Error::NeedMore) => break,
                    Err(e) => panic!("receive_frame: {e}"),
                }
            }
        }
        dec.flush().expect("flush");
        loop {
            match dec.receive_frame() {
                Ok(Frame::Audio(a)) => {
                    for chunk in a.data[0].chunks_exact(2) {
                        trait_out.push(i16::from_le_bytes([chunk[0], chunk[1]]));
                    }
                }
                Ok(other) => panic!("non-audio frame on flush: {other:?}"),
                Err(Error::Eof) | Err(Error::NeedMore) => break,
                Err(e) => panic!("post-flush receive_frame: {e}"),
            }
        }

        assert_eq!(
            trait_out.len(),
            direct.len(),
            "trait-driven sample count {} != direct-chain {}",
            trait_out.len(),
            direct.len()
        );
        let mismatches = trait_out
            .iter()
            .zip(direct.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            mismatches, 0,
            "MPEG-2.5 trait decode differs from direct chain in {mismatches} samples"
        );
        // Sanity: the decode is non-silent (a runaway / mis-routed band
        // table would collapse to silence or saturate).
        assert!(
            trait_out.iter().any(|&s| s.abs() > 64),
            "MPEG-2.5 11.025 kHz trait decode produced silence"
        );
    }

    #[test]
    fn send_packet_accepts_mpeg25_12khz_header_through_the_guard() {
        // The 12 kHz MPEG-2.5 rate is the other fully-grounded extension
        // rate (its SFB tables are byte-identical to the in-repo 24 kHz
        // LSF tables). A header at 12 kHz must pass the version/rate
        // guard through to the side-info / reservoir stage rather than
        // being rejected up front; the main-data slot here is zeros, so
        // the frame may buffer (Ok) or error at a *later* decode stage,
        // but the 8 kHz-style up-front rejection must NOT fire.
        use crate::encoder::{make_silent_header, write_header};
        use crate::frame::ChannelMode;
        let hdr = make_silent_header(32, 12_000, ChannelMode::SingleChannel)
            .expect("mpeg-2.5 12 kHz silent header build");
        assert_eq!(hdr.version, MpegVersion::Mpeg25);
        assert_eq!(hdr.sample_rate_hz, 12_000);
        let hdr_bytes = write_header(&hdr);
        let frame_len = hdr.frame_len().expect("mpeg-2.5 frame_len derivable");
        let mut payload = hdr_bytes.to_vec();
        payload.resize(frame_len, 0u8);
        let tb = TimeBase::new(1, 12_000);
        let pkt = Packet::new(0, tb, payload);
        let mut dec = make_decoder(&build_decoder_params(12_000)).expect("make_decoder");
        match dec.send_packet(&pkt) {
            Ok(()) => {}
            Err(e) => {
                let msg = format!("{e}");
                assert!(
                    !msg.contains("8 kHz") && !msg.contains("observer-trace"),
                    "MPEG-2.5 12 kHz must not hit the 8 kHz rejection arm (got: {msg})"
                );
            }
        }
    }

    #[test]
    fn send_packet_accepts_mpeg2_lsf_header_through_the_guard() {
        // Build a real MPEG-2 LSF 4-byte header (64 kbps Layer III at
        // 22.05 kHz, mono). The packet's main-data slot is zeros so
        // the decoder won't produce meaningful PCM, but the header
        // version-field guard must let the packet through to the
        // side-info / reservoir stage rather than rejecting at the
        // `Mpeg1`-only branch the way r177's wrapper did. Either
        // `Ok(())` (frame produced or buffered) or an error from a
        // *later* stage (side-info parse / scalefactors / huffman)
        // proves the guard widened; what we must NOT see is the
        // r177-style "decoder this round is MPEG-1 only" rejection.
        use crate::encoder::{make_silent_header, write_header};
        use crate::frame::ChannelMode;
        let hdr = make_silent_header(64, 22_050, ChannelMode::SingleChannel)
            .expect("mpeg-2 lsf silent header build");
        assert_eq!(hdr.version, MpegVersion::Mpeg2);
        let hdr_bytes = write_header(&hdr);
        let frame_len = hdr.frame_len().expect("lsf frame_len derivable");
        let mut payload = hdr_bytes.to_vec();
        payload.resize(frame_len, 0u8);
        let tb = TimeBase::new(1, 22_050);
        let pkt = Packet::new(0, tb, payload);
        let mut dec = make_decoder(&build_decoder_params(22_050)).expect("make_decoder");
        match dec.send_packet(&pkt) {
            Ok(()) => {}
            Err(e) => {
                let msg = format!("{e}");
                assert!(
                    !msg.contains("MPEG-1 only"),
                    "MPEG-2 LSF header must not be rejected by the version guard \
                     (got: {msg})"
                );
            }
        }
    }
}
