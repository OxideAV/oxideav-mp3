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
//!   [`AudioFrame`] holding interleaved S16 PCM for that frame's two
//!   granules (MPEG-1 Layer III mono = 1152 samples per frame per
//!   channel).
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
//! This module wires the MPEG-1 Layer III decode path — mono **and**
//! stereo (independent, joint MS, joint MS+intensity) — through the
//! framework trait.
//!
//! * **Mono and stereo.** `channels == 1` or `2`. For stereo frames the
//!   per-channel state — `ImdctState` and `SynthState` — is carried
//!   in a two-element array, and the §2.4.3.4.9 stereo processing stage
//!   (`process_stereo`) runs between requantize and alias reduction per
//!   the established decode pipeline order. Mono behaviour is unchanged
//!   from earlier rounds (only the channel-0 slot of the per-channel
//!   state arrays is used).
//! * **MPEG-1 only.** Sample rates 32 / 44.1 / 48 kHz. The MPEG-2 /
//!   MPEG-2.5 LSF parsing path of [`parse_side_info`] /
//!   [`decode_scalefactors`] is reachable but the synth chain has been
//!   exercised end-to-end only against MPEG-1 fixtures so far; the LSF
//!   trait wiring is a separate later round.
//! * **Layer III only.** Layer I / Layer II frames are rejected at the
//!   `send_packet` boundary.
//!
//! Output PCM follows the framework's `AudioFrame` convention: one
//! `data[plane]` entry per channel (planar layout), with each plane
//! holding little-endian `i16` samples. Mono output keeps the single
//! plane; stereo output writes two planes (`data[0]` = L, `data[1]` = R)
//! covering the same 1152 samples per granule pair.

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
use crate::requantize::requantize;
use crate::scalefactors::{decode_scalefactors, MainDataReader, Reservoir};
use crate::side_info::parse_side_info;
use crate::stream_encoder::SAMPLES_PER_FRAME_MPEG1;
use crate::synth::{synth_granule, SynthState, PCM_PER_GRANULE};

/// Build a boxed MPEG-1 Audio Layer III [`Decoder`] from `params`.
///
/// `params.sample_rate` (32_000 / 44_100 / 48_000) and
/// `params.channels` (1 or 2) configure the returned decoder's stream
/// parameters; the actual per-frame sample rate and channel count are
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
        if hdr.version != MpegVersion::Mpeg1 {
            return Err(Error::unsupported(
                "oxideav-mp3: decoder this round is MPEG-1 only",
            ));
        }
        let nch = hdr.channel_count() as usize;
        if nch != 1 && nch != 2 {
            return Err(Error::unsupported(format!(
                "oxideav-mp3: unsupported channel count {nch} (expected 1 or 2)"
            )));
        }
        let frame_len = hdr.frame_len().ok_or_else(|| {
            Error::unsupported("oxideav-mp3: free-format frames not supported in trait wiring")
        })?;
        if bytes.len() < frame_len {
            return Err(Error::invalid(format!(
                "oxideav-mp3: packet len {} < header-implied frame len {frame_len}",
                bytes.len()
            )));
        }

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
                *xr_slot = requantize(&is, gc, sf, hdr.sample_rate_hz, hdr.version);
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
                    let v = p * f32::from(i16::MAX);
                    pcm_planes[ch].push(v.clamp(i16::MIN as f32, i16::MAX as f32) as i16);
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

/// Install the MPEG-1 Audio Layer III decoder factory (alongside the
/// existing encoder factory from r140) into `reg`.
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
                    let xar = alias_reduce(&xr, gc);
                    let st = imdct_granule(&xar, gc, &mut imdct_state);
                    let pcm_f32 = synth_granule(&st, &mut synth_state);
                    for &p in pcm_f32.iter().take(PCM_PER_GRANULE) {
                        let v = p * f32::from(i16::MAX);
                        out_pcm.push(v.clamp(i16::MIN as f32, i16::MAX as f32) as i16);
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
}
