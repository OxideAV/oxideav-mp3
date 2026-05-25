//! `oxideav_core::Encoder` wiring for MPEG-1 Audio Layer III.
//!
//! This module exposes the existing [`Mp3Encoder`] (the Phase 2 PCM →
//! MP3 frame stream encoder built on top of the analysis filterbank +
//! forward MDCT + quantizer + Huffman emitter + main-data assembler +
//! cross-frame bit-reservoir scheduler) through the framework's
//! [`oxideav_core::Encoder`] trait.
//!
//! ## Trait-API adaptation
//!
//! The framework trait is *frame-in, packet-out*:
//!
//! * [`send_frame`](Encoder::send_frame) accepts one [`AudioFrame`] of
//!   PCM samples (interleaved or planar S16).
//! * [`receive_packet`](Encoder::receive_packet) returns one
//!   [`Packet`] holding **one complete MP3 frame** (header +
//!   side-info + main-data slot).
//! * [`flush`](Encoder::flush) signals end-of-stream so the cross-frame
//!   bit-reservoir schedule can drain.
//!
//! The wrapper buffers incoming PCM samples (re-using
//! [`Mp3Encoder::push_samples`]) and, on `flush`, calls
//! [`Mp3Encoder::finish`] to emit the complete byte stream. That byte
//! stream is then sliced into per-MP3-frame packets via the crate's own
//! [`FrameWalker`] so each [`receive_packet`](Encoder::receive_packet)
//! call yields exactly one MP3 frame. Per-packet PTS is derived from a
//! sample counter in the encoder's `time_base = 1 / sample_rate`.
//!
//! The reservoir-schedule-at-end shape (every frame's main-data needs
//! to be known before any frame can be emitted) is unchanged from the
//! existing [`Mp3Encoder`] — the same constraint surfaces here as
//! "every packet drops out of [`receive_packet`](Encoder::receive_packet)
//! at flush time". Callers that drive the trait incrementally
//! (`send_frame` → `receive_packet` → `send_frame` → …) will see
//! [`Error::NeedMore`] from `receive_packet` until they call
//! [`Encoder::flush`].
//!
//! ## Scope
//!
//! Mirrors the scope of the underlying [`Mp3Encoder`] (Round 138/139):
//!
//! * **Mono only.** `channels == 1`.
//! * **MPEG-1 only.** Sample rates 32 / 44.1 / 48 kHz.
//! * **CBR.** Caller picks a bitrate from the §2.4.2.3 Layer III ladder.
//! * **Long blocks, fixed-gain or outer-loop.** Same two encoder
//!   variants as the direct API — [`make_encoder`] builds the
//!   fixed-gain path, [`make_encoder_with_outer_loop`] the
//!   distortion-control loop.
//!
//! Stereo / MPEG-2 LSF / VBR / short-block switching remain followups.

use std::collections::VecDeque;

use oxideav_core::{
    AudioFrame, CodecCapabilities, CodecId, CodecInfo, CodecParameters, CodecRegistry, CodecTag,
    Encoder, Error, Frame, Packet, Result, SampleFormat, TimeBase,
};

use crate::demuxer::{CODEC_ID_STR, WAVE_FORMAT_MP3};
use crate::frame::{ChannelMode, FrameWalker};
use crate::stream_encoder::{Mp3Encoder, SAMPLES_PER_FRAME_MPEG1};

/// Build a boxed MPEG-1 Audio Layer III mono CBR [`Encoder`] from
/// `params`.
///
/// `params.sample_rate` (32_000 / 44_100 / 48_000), `params.channels`
/// (must be 1), and `params.bit_rate` (one of the §2.4.2.3 Layer III
/// MPEG-1 ladder values, in bits/s — defaults to 128_000 when absent)
/// drive the underlying [`Mp3Encoder`] configuration.
///
/// # Errors
///
/// Returns [`Error::invalid`] if `sample_rate` / `channels` are missing
/// or out of the supported scope, and [`Error::other`] if the underlying
/// [`Mp3Encoder::new`] rejects the combination (e.g. a bitrate not on
/// the §2.4.2.3 ladder).
pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    make_encoder_inner(params, /*outer_loop_threshold=*/ None)
}

/// Build a boxed MPEG-1 Audio Layer III mono CBR [`Encoder`] from
/// `params` with the §C.1.5.4.3 outer (distortion-control) loop
/// enabled, using the supplied uniform `xmin(sb)` threshold.
///
/// Equivalent to [`make_encoder`] except the underlying
/// [`Mp3Encoder`] is built via
/// [`Mp3Encoder::new_with_outer_loop`]. With no psychoacoustic model
/// wired up yet the threshold is a flat constant — pass
/// [`DEFAULT_OUTER_LOOP_THRESHOLD`] for the recommended default.
pub fn make_encoder_with_outer_loop(
    params: &CodecParameters,
    uniform_threshold: f64,
) -> Result<Box<dyn Encoder>> {
    make_encoder_inner(params, Some(uniform_threshold))
}

fn make_encoder_inner(
    params: &CodecParameters,
    outer_loop_threshold: Option<f64>,
) -> Result<Box<dyn Encoder>> {
    let sample_rate = params
        .sample_rate
        .ok_or_else(|| Error::invalid("oxideav-mp3: sample_rate required"))?;
    let channels = params
        .channels
        .ok_or_else(|| Error::invalid("oxideav-mp3: channels required"))?;
    if channels != 1 {
        return Err(Error::invalid(format!(
            "oxideav-mp3: encoder supports mono only (channels={channels})"
        )));
    }
    let mode = ChannelMode::SingleChannel;
    // Default the bitrate when absent. 128 kbit/s is the standard
    // mono / 44.1 kHz reference; it's a valid entry on the §2.4.2.3
    // ladder for every MPEG-1 sample rate the encoder supports.
    let bitrate_bps = params.bit_rate.unwrap_or(128_000);
    if bitrate_bps == 0 || bitrate_bps > 1_000_000 {
        return Err(Error::invalid(format!(
            "oxideav-mp3: bit_rate {bitrate_bps} out of range"
        )));
    }
    let bitrate_kbps = (bitrate_bps / 1000) as u32;
    let inner = match outer_loop_threshold {
        Some(thr) => Mp3Encoder::new_with_outer_loop(bitrate_kbps, sample_rate, mode, thr),
        None => Mp3Encoder::new(bitrate_kbps, sample_rate, mode),
    }
    .map_err(|e| Error::other(format!("oxideav-mp3: encoder build: {e}")))?;

    let mut out_params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
    out_params.sample_rate = Some(sample_rate);
    out_params.channels = Some(channels);
    out_params.sample_format = Some(SampleFormat::S16);
    out_params.bit_rate = Some(u64::from(bitrate_kbps) * 1000);
    out_params.tag = Some(CodecTag::wave_format(WAVE_FORMAT_MP3));

    Ok(Box::new(Mp3CoreEncoder::new(
        CodecId::new(CODEC_ID_STR),
        inner,
        out_params,
        sample_rate,
    )))
}

/// Frame-to-packet adaptor that wires [`Mp3Encoder`] into the
/// framework [`Encoder`] trait.
///
/// The wrapper buffers PCM through [`Mp3Encoder::push_samples`] and,
/// on [`Encoder::flush`], runs [`Mp3Encoder::finish`] to produce the
/// complete CBR stream. The stream is then sliced into one
/// [`Packet`] per MP3 frame via [`FrameWalker`], queued in
/// `pending_packets`, and drained by successive
/// [`Encoder::receive_packet`] calls.
pub struct Mp3CoreEncoder {
    codec_id: CodecId,
    /// Owned [`Mp3Encoder`]; consumed (via `Option::take`) on flush.
    inner: Option<Mp3Encoder>,
    output: CodecParameters,
    sample_rate: u32,
    /// Pending packets carved out of the encoder's flushed byte
    /// stream. Each [`Encoder::receive_packet`] call pops one.
    pending_packets: VecDeque<Packet>,
    /// Total PCM samples (per channel) accepted so far via
    /// [`Encoder::send_frame`]. Used for per-packet PTS stamping
    /// across the time base `1 / sample_rate`.
    samples_in: u64,
    /// Set once [`Encoder::flush`] has run [`Mp3Encoder::finish`]; any
    /// further [`Encoder::send_frame`] errors and any future
    /// [`Encoder::receive_packet`] after `pending_packets` is empty
    /// returns [`Error::Eof`].
    eof: bool,
}

impl std::fmt::Debug for Mp3CoreEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mp3CoreEncoder")
            .field("codec_id", &self.codec_id)
            .field("inner_present", &self.inner.is_some())
            .field("sample_rate", &self.sample_rate)
            .field("pending_packets", &self.pending_packets.len())
            .field("samples_in", &self.samples_in)
            .field("eof", &self.eof)
            .finish()
    }
}

impl Mp3CoreEncoder {
    fn new(
        codec_id: CodecId,
        inner: Mp3Encoder,
        output: CodecParameters,
        sample_rate: u32,
    ) -> Self {
        Self {
            codec_id,
            inner: Some(inner),
            output,
            sample_rate,
            pending_packets: VecDeque::new(),
            samples_in: 0,
            eof: false,
        }
    }

    /// Decode an [`AudioFrame`]'s raw bytes into a mono `i16` PCM
    /// vector. Accepts interleaved S16 (`data.len() == 1`) and
    /// single-plane planar S16P (`data.len() == 1`, mono is its own
    /// planar layout). Errors on multi-plane / multi-channel input
    /// (stereo encode is out of scope for this round).
    fn frame_to_mono_i16(&self, frame: &AudioFrame) -> Result<Vec<i16>> {
        let samples = frame.samples as usize;
        if frame.data.len() != 1 {
            return Err(Error::invalid(format!(
                "oxideav-mp3: encoder expects mono (1 plane), got {} planes",
                frame.data.len()
            )));
        }
        let bytes = &frame.data[0];
        if bytes.len() != samples * 2 {
            return Err(Error::invalid(format!(
                "oxideav-mp3: frame data len {} != {} expected for {samples} mono S16 samples",
                bytes.len(),
                samples * 2
            )));
        }
        let mut out = Vec::with_capacity(samples);
        for i in 0..samples {
            let lo = bytes[i * 2];
            let hi = bytes[i * 2 + 1];
            out.push(i16::from_le_bytes([lo, hi]));
        }
        Ok(out)
    }
}

impl Encoder for Mp3CoreEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.output
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        if self.eof {
            return Err(Error::other("oxideav-mp3: cannot send_frame after flush"));
        }
        let Frame::Audio(a) = frame else {
            return Err(Error::invalid("oxideav-mp3: encoder requires audio frame"));
        };
        let pcm = self.frame_to_mono_i16(a)?;
        let inner = self
            .inner
            .as_mut()
            .expect("inner encoder present until flush");
        inner
            .push_samples(&pcm)
            .map_err(|e| Error::other(format!("oxideav-mp3: push_samples: {e}")))?;
        self.samples_in = self.samples_in.saturating_add(a.samples as u64);
        Ok(())
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        if let Some(pkt) = self.pending_packets.pop_front() {
            return Ok(pkt);
        }
        if self.eof {
            return Err(Error::Eof);
        }
        Err(Error::NeedMore)
    }

    fn flush(&mut self) -> Result<()> {
        if self.eof {
            // Idempotent: a second flush is a no-op, not an error.
            return Ok(());
        }
        let inner = self
            .inner
            .take()
            .expect("inner encoder present until flush");
        let mut bytes: Vec<u8> = Vec::new();
        inner
            .finish(&mut bytes)
            .map_err(|e| Error::other(format!("oxideav-mp3: finish: {e}")))?;

        // Slice the emitted CBR stream into one Packet per MP3 frame.
        // Per-packet PTS is computed from a running sample-position
        // counter under the encoder's time_base = 1 / sample_rate.
        // Each MPEG-1 Layer III frame represents
        // SAMPLES_PER_FRAME_MPEG1 = 1152 PCM samples per channel.
        let tb = TimeBase::new(1, i64::from(self.sample_rate));
        let mut pts_samples: i64 = 0;
        for f in FrameWalker::new(&bytes) {
            let mut pkt = Packet::new(0, tb, f.data.to_vec());
            pkt.pts = Some(pts_samples);
            pkt.duration = Some(SAMPLES_PER_FRAME_MPEG1 as i64);
            pkt.flags.keyframe = true;
            self.pending_packets.push_back(pkt);
            pts_samples = pts_samples.saturating_add(SAMPLES_PER_FRAME_MPEG1 as i64);
        }
        self.eof = true;
        Ok(())
    }
}

/// Install the MPEG-1 Audio Layer III encoder factory (and the codec
/// container tag claims) into `reg`.
///
/// Claims the WAVE format tag `0x0055` (MPEG Layer III) and the
/// Matroska codec id `A_MPEG/L3`. The encoder factory builds a
/// [`Mp3CoreEncoder`] wrapping the fixed-gain [`Mp3Encoder`]; the
/// outer-loop variant is reachable via the direct [`make_encoder_with_outer_loop`]
/// API.
pub fn register_codecs(reg: &mut CodecRegistry) {
    let info = CodecInfo::new(CodecId::new(CODEC_ID_STR))
        .capabilities(
            CodecCapabilities::audio("mp3")
                .with_encode()
                .with_lossy(true),
        )
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
    use crate::stream_encoder::DEFAULT_OUTER_LOOP_THRESHOLD;
    use oxideav_core::Frame;

    /// A 440 Hz mono sine in interleaved S16, `n_samples` per channel.
    fn sine_s16(n: usize, freq_hz: f32, sample_rate_hz: f32, amp: f32) -> Vec<u8> {
        let two_pi = 2.0 * std::f32::consts::PI;
        let scale = amp * (i16::MAX as f32);
        let mut out = Vec::with_capacity(n * 2);
        for i in 0..n {
            let t = i as f32 / sample_rate_hz;
            let v = (two_pi * freq_hz * t).sin() * scale;
            let s = v.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16;
            out.extend_from_slice(&s.to_le_bytes());
        }
        out
    }

    fn build_params(sample_rate: u32, bit_rate_bps: u64) -> CodecParameters {
        let mut p = CodecParameters::audio(CodecId::new("mp3"));
        p.sample_rate = Some(sample_rate);
        p.channels = Some(1);
        p.sample_format = Some(SampleFormat::S16);
        p.bit_rate = Some(bit_rate_bps);
        p
    }

    #[test]
    fn make_encoder_requires_sample_rate_and_channels() {
        let bare = CodecParameters::audio(CodecId::new("mp3"));
        assert!(make_encoder(&bare).is_err());
    }

    #[test]
    fn make_encoder_rejects_stereo() {
        let mut p = CodecParameters::audio(CodecId::new("mp3"));
        p.sample_rate = Some(44_100);
        p.channels = Some(2);
        assert!(make_encoder(&p).is_err());
    }

    #[test]
    fn make_encoder_builds_and_reports_output_params() {
        let p = build_params(44_100, 128_000);
        let enc = make_encoder(&p).expect("make_encoder");
        assert_eq!(enc.codec_id().as_str(), "mp3");
        assert_eq!(enc.output_params().sample_rate, Some(44_100));
        assert_eq!(enc.output_params().channels, Some(1));
        assert_eq!(enc.output_params().bit_rate, Some(128_000));
        assert_eq!(
            enc.output_params().tag,
            Some(CodecTag::wave_format(WAVE_FORMAT_MP3))
        );
    }

    #[test]
    fn receive_packet_before_flush_returns_need_more() {
        let p = build_params(44_100, 128_000);
        let mut enc = make_encoder(&p).expect("make_encoder");
        let pcm = sine_s16(SAMPLES_PER_FRAME_MPEG1 * 2, 440.0, 44_100.0, 0.5);
        let frame = AudioFrame {
            samples: (SAMPLES_PER_FRAME_MPEG1 * 2) as u32,
            pts: None,
            data: vec![pcm],
        };
        enc.send_frame(&Frame::Audio(frame)).unwrap();
        // No flush yet — packets are queued only at flush time.
        assert!(matches!(enc.receive_packet(), Err(Error::NeedMore)));
    }

    #[test]
    fn flush_drains_to_complete_mp3_frames() {
        // Drive ~50 ms of audio, flush, drain packets, verify each
        // packet starts with the MP3 sync pattern.
        let p = build_params(44_100, 128_000);
        let mut enc = make_encoder(&p).expect("make_encoder");
        // 3 frames worth of PCM (3 · 1152 = 3456 samples).
        let pcm = sine_s16(SAMPLES_PER_FRAME_MPEG1 * 3, 440.0, 44_100.0, 0.5);
        let frame = AudioFrame {
            samples: (SAMPLES_PER_FRAME_MPEG1 * 3) as u32,
            pts: None,
            data: vec![pcm],
        };
        enc.send_frame(&Frame::Audio(frame)).unwrap();
        enc.flush().unwrap();

        let mut packets: Vec<Packet> = Vec::new();
        loop {
            match enc.receive_packet() {
                Ok(pkt) => packets.push(pkt),
                Err(Error::Eof) => break,
                Err(e) => panic!("unexpected error: {e}"),
            }
        }
        // 3456 samples / 1152 per frame = 3 MP3 frames; the encoder may
        // round up one frame to absorb buffered tail PCM.
        assert!(
            (3..=4).contains(&packets.len()),
            "unexpected packet count {}",
            packets.len()
        );
        // Every packet begins with the 12-bit MP3 syncword 0xFFF.
        for pkt in &packets {
            assert!(pkt.data.len() >= 4, "packet too short");
            assert_eq!(pkt.data[0], 0xFF, "missing sync byte 0");
            assert_eq!(pkt.data[1] & 0xE0, 0xE0, "missing sync bits in byte 1");
            assert_eq!(pkt.duration, Some(SAMPLES_PER_FRAME_MPEG1 as i64));
        }
        // PTS is monotonic in 1/sample_rate units.
        for w in packets.windows(2) {
            let a = w[0].pts.unwrap();
            let b = w[1].pts.unwrap();
            assert!(b > a, "PTS not monotonic ({a} >= {b})");
        }
    }

    #[test]
    fn flush_idempotent() {
        let p = build_params(44_100, 128_000);
        let mut enc = make_encoder(&p).expect("make_encoder");
        let pcm = sine_s16(SAMPLES_PER_FRAME_MPEG1, 440.0, 44_100.0, 0.5);
        let frame = AudioFrame {
            samples: SAMPLES_PER_FRAME_MPEG1 as u32,
            pts: None,
            data: vec![pcm],
        };
        enc.send_frame(&Frame::Audio(frame)).unwrap();
        enc.flush().unwrap();
        enc.flush().expect("second flush no-op");
    }

    #[test]
    fn send_frame_after_flush_errors() {
        let p = build_params(44_100, 128_000);
        let mut enc = make_encoder(&p).expect("make_encoder");
        let pcm = sine_s16(SAMPLES_PER_FRAME_MPEG1, 440.0, 44_100.0, 0.5);
        let frame = AudioFrame {
            samples: SAMPLES_PER_FRAME_MPEG1 as u32,
            pts: None,
            data: vec![pcm.clone()],
        };
        enc.send_frame(&Frame::Audio(frame.clone())).unwrap();
        enc.flush().unwrap();
        assert!(enc.send_frame(&Frame::Audio(frame)).is_err());
    }

    #[test]
    fn register_codecs_installs_encoder_factory() {
        let mut reg = CodecRegistry::new();
        register_codecs(&mut reg);
        let id = CodecId::new("mp3");
        assert!(reg.has_encoder(&id));
        let p = build_params(44_100, 128_000);
        let enc = reg.first_encoder(&p).expect("first_encoder");
        assert_eq!(enc.codec_id().as_str(), "mp3");
    }

    #[test]
    fn register_codecs_tag_claims_resolve() {
        let mut reg = CodecRegistry::new();
        register_codecs(&mut reg);
        let wf = CodecTag::wave_format(WAVE_FORMAT_MP3);
        assert_eq!(
            reg.resolve_tag_ref(&oxideav_core::ProbeContext::new(&wf))
                .map(|c| c.as_str()),
            Some("mp3"),
        );
        let mk = CodecTag::matroska("A_MPEG/L3");
        assert_eq!(
            reg.resolve_tag_ref(&oxideav_core::ProbeContext::new(&mk))
                .map(|c| c.as_str()),
            Some("mp3"),
        );
    }

    #[test]
    fn make_encoder_with_outer_loop_constructs() {
        let p = build_params(44_100, 128_000);
        let enc =
            make_encoder_with_outer_loop(&p, DEFAULT_OUTER_LOOP_THRESHOLD).expect("outer-loop");
        assert_eq!(enc.codec_id().as_str(), "mp3");
    }
}
