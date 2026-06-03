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
//! Mirrors the scope of the underlying [`Mp3Encoder`]:
//!
//! * **Mono or stereo (independent channels).** `channels == 1` →
//!   header `mode = '11'`; `channels == 2` → header `mode = '00'`
//!   (independent stereo). Each channel of a stereo input is encoded
//!   independently — there is no joint-stereo MS / intensity coupling
//!   at the encoder side this round.
//! * **MPEG-1 only.** Sample rates 32 / 44.1 / 48 kHz.
//! * **CBR.** Caller picks a bitrate from the §2.4.2.3 Layer III ladder.
//! * **Long blocks, fixed-gain or outer-loop.** Same two encoder
//!   variants as the direct API — [`make_encoder`] builds the
//!   fixed-gain path, [`make_encoder_with_outer_loop`] the
//!   distortion-control loop.
//!
//! Joint-stereo (MS / intensity) / MPEG-2 LSF / VBR / short-block
//! switching remain followups.

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

/// Build a boxed MPEG-1 Audio Layer III mono / independent-stereo CBR
/// [`Encoder`] with the §C.1.5.4.3 outer (distortion-control) loop
/// enabled and the per-band Annex D **threshold-in-quiet** vector
/// ([`crate::psy::XminThresholds::threshold_in_quiet`]) pre-installed
/// as the per-band `xmin(sb)` for every block-type branch (long /
/// pure-short / mixed). Trait-API one-shot bundle of
/// [`Mp3Encoder::new_with_outer_loop`] +
/// [`Mp3Encoder::set_per_band_xmin`] (equivalent to
/// [`Mp3Encoder::new_with_threshold_in_quiet`]).
///
/// The per-band path is a strict generalisation of the uniform-scalar
/// path: a band whose distortion stays below its `xmin` is left at the
/// current quantization, while a band whose distortion exceeds the
/// per-band threshold is amplified by the outer loop. With the
/// threshold-in-quiet shape this spends bits where the ear actually
/// needs them — mid-spectrum bands near the 3.4 kHz LTq minimum carry
/// a stricter `xmin` than the bass / treble extremes — and the §D.1
/// Step 3 `−12 dB` offset at `bitrate_kbps_per_channel >= 96` lowers
/// the whole curve another factor of `10^(−12/10)` to recover the
/// reference's transparency target. The per-channel bitrate
/// (`bit_rate / channels`) is what the offset switches on; the
/// `bit_rate` field is in bits/s, so 192_000 bit/s stereo → 96 kbit/s
/// per channel, exactly the cutover point.
///
/// Equivalent direct-API recipe:
///
/// ```ignore
/// use oxideav_mp3::Mp3Encoder;
/// let enc = Mp3Encoder::new_with_threshold_in_quiet(
///     128, 44_100, ChannelMode::SingleChannel,
/// )?;
/// ```
///
/// # Errors
///
/// Same as [`make_encoder_with_outer_loop`] — bad sample-rate /
/// bit-rate / channel-count combination, or a non-MPEG-1 sample rate
/// (LSF remains deferred).
pub fn make_encoder_with_threshold_in_quiet(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    make_encoder_inner_threshold_in_quiet(params, None)
}

/// Build a boxed MPEG-1 Audio Layer III mono / independent-stereo CBR
/// [`Encoder`] with the §C.1.5.4.3 outer-loop **plus** the per-band
/// Annex D threshold-in-quiet vector, using a **caller-supplied** §D.1
/// Step 3 dB offset instead of the spec-default per-channel-bitrate
/// branching. Trait-API one-shot bundle of
/// [`Mp3Encoder::new_with_threshold_in_quiet_offset`].
///
/// The spec's §D.1 Step 3 mandates exactly two offsets — `−12 dB` when
/// `bitrate_kbps_per_channel >= 96` and `0 dB` otherwise — and every
/// spec-conformant transparency target maps to one of those two
/// values. Callers wanting the spec default should continue to use
/// [`make_encoder_with_threshold_in_quiet`]; this `_offset` variant is
/// for front-ends that expose a continuous transparency / quality
/// slider, for VBR encoders that compute a running offset from a
/// recent-bitrate accumulator, and for test sweeps over the offset.
///
/// `offset_db` is applied uniformly across every long, pure-short, and
/// mixed cell on top of the per-frequency `LTq` shape — the bowl is
/// preserved and the whole curve is translated up or down by
/// `offset_db` dB.
///
/// # Errors
///
/// Same as [`make_encoder_with_threshold_in_quiet`].
pub fn make_encoder_with_threshold_in_quiet_offset(
    params: &CodecParameters,
    offset_db: f64,
) -> Result<Box<dyn Encoder>> {
    make_encoder_inner_threshold_in_quiet(params, Some(offset_db))
}

/// Build a boxed MPEG-1 Audio Layer III stereo CBR [`Encoder`] in
/// **joint-stereo MS** mode (ISO/IEC 11172-3:1993 §2.4.3.4.9.2).
///
/// Wraps [`Mp3Encoder::new_joint_stereo_ms`]. `params.channels` must be
/// 2 (joint stereo is by definition a two-channel encoding). The
/// emitted stream carries header `mode = '01'` (joint stereo) with
/// `mode_extension = '10'` (ms_stereo on, intensity_stereo off) per
/// §2.4.2.3 on every audio frame.
///
/// # Errors
///
/// Returns [`Error::invalid`] when `params.channels != 2` (mono cannot
/// be joint-stereo), [`Error::invalid`] when `sample_rate` is missing,
/// and [`Error::other`] when the underlying
/// [`Mp3Encoder::new_joint_stereo_ms`] rejects the bitrate / sample
/// rate combination.
pub fn make_encoder_joint_stereo_ms(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    let (sample_rate, bitrate_kbps) = validate_joint_stereo_params(params)?;
    let inner = Mp3Encoder::new_joint_stereo_ms(bitrate_kbps, sample_rate)
        .map_err(|e| Error::other(format!("oxideav-mp3: joint-stereo MS build: {e}")))?;
    Ok(boxed_joint_stereo_core(inner, sample_rate, bitrate_kbps))
}

/// Build a boxed MPEG-1 Audio Layer III stereo CBR [`Encoder`] in
/// **joint-stereo auto MS/LR** mode (Phase 2 step 20).
///
/// Wraps [`Mp3Encoder::new_joint_stereo_auto`]. `params.channels` must
/// be 2. The emitted stream carries header `mode = '01'` (joint stereo)
/// on every audio frame; the per-frame picker selects
/// `mode_extension = '10'` (MS active) when both granules satisfy
/// `E_S / (E_L + E_R) ≤ 0.5` and `mode_extension = '00'` (no joint
/// rotation) otherwise, per the heuristic documented on
/// [`Mp3Encoder::new_joint_stereo_auto`]. Use
/// [`make_encoder_joint_stereo_auto_with_threshold`] when a non-default
/// energy threshold is needed.
///
/// # Errors
///
/// Returns [`Error::invalid`] when `params.channels != 2` or
/// `sample_rate` is missing, and [`Error::other`] when the underlying
/// [`Mp3Encoder::new_joint_stereo_auto`] rejects the bitrate / sample
/// rate combination.
pub fn make_encoder_joint_stereo_auto(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    let (sample_rate, bitrate_kbps) = validate_joint_stereo_params(params)?;
    let inner = Mp3Encoder::new_joint_stereo_auto(bitrate_kbps, sample_rate)
        .map_err(|e| Error::other(format!("oxideav-mp3: joint-stereo auto build: {e}")))?;
    Ok(boxed_joint_stereo_core(inner, sample_rate, bitrate_kbps))
}

/// As [`make_encoder_joint_stereo_auto`] but with a caller-supplied
/// side-channel energy threshold (the upper bound on
/// `E_S / (E_L + E_R)` at which a frame's two granules both qualify
/// for MS). Values outside `[0.0, 1.0]` are clamped to that range by
/// [`Mp3Encoder::with_ms_auto_threshold`].
///
/// # Errors
///
/// Same as [`make_encoder_joint_stereo_auto`].
pub fn make_encoder_joint_stereo_auto_with_threshold(
    params: &CodecParameters,
    threshold: f64,
) -> Result<Box<dyn Encoder>> {
    let (sample_rate, bitrate_kbps) = validate_joint_stereo_params(params)?;
    let inner = Mp3Encoder::new_joint_stereo_auto(bitrate_kbps, sample_rate)
        .map_err(|e| Error::other(format!("oxideav-mp3: joint-stereo auto build: {e}")))?
        .with_ms_auto_threshold(threshold);
    Ok(boxed_joint_stereo_core(inner, sample_rate, bitrate_kbps))
}

/// Shared `(sample_rate, bitrate_kbps)` validation for the
/// joint-stereo factory wrappers — `channels == 2` and a sane
/// in-range `bit_rate` are universal preconditions for the joint
/// modes; the underlying [`Mp3Encoder`] constructor checks the
/// §2.4.2.3 ladder match. `bit_rate` defaults to 192_000 bps when
/// the caller doesn't supply one (the standard 96-kbps-per-channel
/// reference for 44.1 kHz stereo).
fn validate_joint_stereo_params(params: &CodecParameters) -> Result<(u32, u32)> {
    let sample_rate = params
        .sample_rate
        .ok_or_else(|| Error::invalid("oxideav-mp3: sample_rate required"))?;
    let channels = params
        .channels
        .ok_or_else(|| Error::invalid("oxideav-mp3: channels required"))?;
    if channels != 2 {
        return Err(Error::invalid(format!(
            "oxideav-mp3: joint-stereo requires channels == 2 (got {channels})"
        )));
    }
    let bitrate_bps = params.bit_rate.unwrap_or(192_000);
    if bitrate_bps == 0 || bitrate_bps > 1_000_000 {
        return Err(Error::invalid(format!(
            "oxideav-mp3: bit_rate {bitrate_bps} out of range"
        )));
    }
    Ok((sample_rate, (bitrate_bps / 1000) as u32))
}

/// Build the trait-object wrapper around a joint-stereo
/// [`Mp3Encoder`], with output [`CodecParameters`] populated for the
/// fixed two-channel case.
fn boxed_joint_stereo_core(
    inner: Mp3Encoder,
    sample_rate: u32,
    bitrate_kbps: u32,
) -> Box<dyn Encoder> {
    let mut out_params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
    out_params.sample_rate = Some(sample_rate);
    out_params.channels = Some(2);
    out_params.sample_format = Some(SampleFormat::S16);
    out_params.bit_rate = Some(u64::from(bitrate_kbps) * 1000);
    out_params.tag = Some(CodecTag::wave_format(WAVE_FORMAT_MP3));

    Box::new(Mp3CoreEncoder::new(
        CodecId::new(CODEC_ID_STR),
        inner,
        out_params,
        sample_rate,
        2,
    ))
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
    // Map `channels` → ISO/IEC 11172-3 §2.4.2.3 `mode`:
    //  * `1` → `SingleChannel` (`mode = '11'`), 17-byte side info.
    //  * `2` → `Stereo` (`mode = '00'`), 32-byte side info, two
    //          independent channels (no joint-stereo coupling this
    //          round; `mode_extension` stays `'00'`).
    // Joint-stereo (`mode = '01'`) needs an encoder-side MS / intensity
    // analysis stage that is out of scope.
    let mode = match channels {
        1 => ChannelMode::SingleChannel,
        2 => ChannelMode::Stereo,
        _ => {
            return Err(Error::invalid(format!(
                "oxideav-mp3: encoder supports 1 or 2 channels (channels={channels})"
            )));
        }
    };
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
        channels as usize,
    )))
}

/// Sibling of [`make_encoder_inner`] that constructs the underlying
/// [`Mp3Encoder`] via [`Mp3Encoder::new_with_threshold_in_quiet`]
/// (outer loop on, per-band Annex D threshold-in-quiet vector
/// pre-installed). Validation rules — sample-rate present, channels in
/// `{1, 2}`, bit-rate in range — are identical to
/// [`make_encoder_inner`]; only the underlying constructor differs.
fn make_encoder_inner_threshold_in_quiet(
    params: &CodecParameters,
    offset_db: Option<f64>,
) -> Result<Box<dyn Encoder>> {
    let sample_rate = params
        .sample_rate
        .ok_or_else(|| Error::invalid("oxideav-mp3: sample_rate required"))?;
    let channels = params
        .channels
        .ok_or_else(|| Error::invalid("oxideav-mp3: channels required"))?;
    let mode = match channels {
        1 => ChannelMode::SingleChannel,
        2 => ChannelMode::Stereo,
        _ => {
            return Err(Error::invalid(format!(
                "oxideav-mp3: encoder supports 1 or 2 channels (channels={channels})"
            )));
        }
    };
    let bitrate_bps = params.bit_rate.unwrap_or(128_000);
    if bitrate_bps == 0 || bitrate_bps > 1_000_000 {
        return Err(Error::invalid(format!(
            "oxideav-mp3: bit_rate {bitrate_bps} out of range"
        )));
    }
    let bitrate_kbps = (bitrate_bps / 1000) as u32;
    let inner = match offset_db {
        Some(off) => {
            Mp3Encoder::new_with_threshold_in_quiet_offset(bitrate_kbps, sample_rate, mode, off)
        }
        None => Mp3Encoder::new_with_threshold_in_quiet(bitrate_kbps, sample_rate, mode),
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
        channels as usize,
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
    /// Number of channels in the input PCM the wrapper accepts (`1`
    /// or `2`). Drives the interleaved-bytes validation in
    /// [`Mp3CoreEncoder::frame_to_i16`] and is the unit `samples_in`
    /// counts on (per-channel).
    channels: usize,
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
            .field("channels", &self.channels)
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
        channels: usize,
    ) -> Self {
        Self {
            codec_id,
            inner: Some(inner),
            output,
            sample_rate,
            channels,
            pending_packets: VecDeque::new(),
            samples_in: 0,
            eof: false,
        }
    }

    /// Decode an [`AudioFrame`]'s raw bytes into the interleaved
    /// `i16` PCM the underlying [`Mp3Encoder::push_samples`] expects
    /// (`[L0, R0, L1, R1, …]` for stereo, `[s0, s1, s2, …]` for mono).
    ///
    /// Interleaved S16 (`SampleFormat::S16`) carries the LR pairs in a
    /// single plane (`data.len() == 1`); the raw bytes are already in
    /// the layout the inner encoder consumes, so this helper just
    /// validates the plane / byte counts and decodes little-endian
    /// `i16`. Mono is its own degenerate interleaving (one sample per
    /// frame position).
    fn frame_to_i16(&self, frame: &AudioFrame, channels: usize) -> Result<Vec<i16>> {
        let samples = frame.samples as usize;
        if frame.data.len() != 1 {
            return Err(Error::invalid(format!(
                "oxideav-mp3: encoder expects interleaved S16 (1 plane), got {} planes",
                frame.data.len()
            )));
        }
        let bytes = &frame.data[0];
        let expected_bytes = samples * channels * 2;
        if bytes.len() != expected_bytes {
            let got = bytes.len();
            return Err(Error::invalid(format!(
                "oxideav-mp3: frame data len {got} != {expected_bytes} expected for {samples} samples × {channels} ch S16"
            )));
        }
        let total = samples * channels;
        let mut out = Vec::with_capacity(total);
        for i in 0..total {
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
        let pcm = self.frame_to_i16(a, self.channels)?;
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
    fn make_encoder_accepts_stereo() {
        // channels == 2 → ChannelMode::Stereo (no joint-stereo
        // coupling). The wrapper should build successfully and report
        // `channels = 2` on its output parameters.
        let mut p = CodecParameters::audio(CodecId::new("mp3"));
        p.sample_rate = Some(44_100);
        p.channels = Some(2);
        p.sample_format = Some(SampleFormat::S16);
        p.bit_rate = Some(192_000);
        let enc = make_encoder(&p).expect("make_encoder stereo");
        assert_eq!(enc.output_params().channels, Some(2));
    }

    #[test]
    fn make_encoder_rejects_more_than_two_channels() {
        // Joint-stereo / multi-channel modes are out of scope: only
        // 1 (mono) and 2 (independent stereo) are accepted.
        let mut p = CodecParameters::audio(CodecId::new("mp3"));
        p.sample_rate = Some(44_100);
        p.channels = Some(3);
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

    /// Build interleaved S16 LR bytes for two independent sine tones.
    fn sine_s16_stereo(
        n: usize,
        freq_l: f32,
        freq_r: f32,
        sample_rate_hz: f32,
        amp: f32,
    ) -> Vec<u8> {
        let two_pi = 2.0 * std::f32::consts::PI;
        let scale = amp * (i16::MAX as f32);
        let mut out = Vec::with_capacity(n * 2 * 2);
        for i in 0..n {
            let t = i as f32 / sample_rate_hz;
            let vl = (two_pi * freq_l * t).sin() * scale;
            let vr = (two_pi * freq_r * t).sin() * scale;
            let sl = vl.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16;
            let sr = vr.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16;
            out.extend_from_slice(&sl.to_le_bytes());
            out.extend_from_slice(&sr.to_le_bytes());
        }
        out
    }

    #[test]
    fn stereo_flush_drains_to_complete_mp3_frames() {
        // Drive ~3 frames of 440/880 Hz LR sine through the stereo
        // trait wrapper. Every emitted packet must (a) start with the
        // 12-bit sync, (b) carry `mode = '00'` (stereo) in its
        // header's byte 3 (mode field is bits 7..6 of byte 3 →
        // shift-right-6 == 0).
        let mut p = CodecParameters::audio(CodecId::new("mp3"));
        p.sample_rate = Some(44_100);
        p.channels = Some(2);
        p.sample_format = Some(SampleFormat::S16);
        p.bit_rate = Some(192_000);
        let mut enc = make_encoder(&p).expect("stereo make_encoder");

        let pcm = sine_s16_stereo(SAMPLES_PER_FRAME_MPEG1 * 3, 440.0, 880.0, 44_100.0, 0.5);
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
        assert!(
            (3..=4).contains(&packets.len()),
            "unexpected stereo packet count {}",
            packets.len()
        );
        for pkt in &packets {
            assert!(pkt.data.len() >= 4);
            assert_eq!(pkt.data[0], 0xFF);
            assert_eq!(pkt.data[1] & 0xE0, 0xE0);
            // Header byte 3 layout (§2.4.2.3): bits 7..6 = mode.
            // Stereo mode '00' → top two bits are zero.
            assert_eq!(pkt.data[3] & 0xC0, 0x00, "expected stereo mode '00'");
        }
    }

    #[test]
    fn stereo_send_frame_rejects_wrong_byte_count() {
        // Stereo wrapper expects `samples × 2 × 2` bytes per frame
        // (LR interleaved S16). Submitting `samples × 2` (the mono
        // byte count) should error.
        let mut p = CodecParameters::audio(CodecId::new("mp3"));
        p.sample_rate = Some(44_100);
        p.channels = Some(2);
        p.sample_format = Some(SampleFormat::S16);
        p.bit_rate = Some(192_000);
        let mut enc = make_encoder(&p).expect("stereo make_encoder");
        // Pass mono-sized bytes claiming the stereo sample count.
        let bad = sine_s16(SAMPLES_PER_FRAME_MPEG1, 440.0, 44_100.0, 0.5);
        let frame = AudioFrame {
            samples: SAMPLES_PER_FRAME_MPEG1 as u32,
            pts: None,
            data: vec![bad],
        };
        assert!(enc.send_frame(&Frame::Audio(frame)).is_err());
    }

    #[test]
    fn make_encoder_with_outer_loop_constructs() {
        let p = build_params(44_100, 128_000);
        let enc =
            make_encoder_with_outer_loop(&p, DEFAULT_OUTER_LOOP_THRESHOLD).expect("outer-loop");
        assert_eq!(enc.codec_id().as_str(), "mp3");
    }

    #[test]
    fn make_encoder_with_threshold_in_quiet_constructs_and_reports_params() {
        // Mono — straight-line construction through the factory must
        // report the same `output_params` as the uniform-scalar
        // outer-loop factory at the same `bit_rate`.
        let p = build_params(44_100, 128_000);
        let enc = make_encoder_with_threshold_in_quiet(&p).expect("ltq factory mono");
        assert_eq!(enc.codec_id().as_str(), "mp3");
        assert_eq!(enc.output_params().sample_rate, Some(44_100));
        assert_eq!(enc.output_params().channels, Some(1));
        assert_eq!(enc.output_params().bit_rate, Some(128_000));
    }

    #[test]
    fn make_encoder_with_threshold_in_quiet_accepts_stereo() {
        // Stereo at 192 kbit/s — per-channel bitrate 96 kbit/s, exactly
        // the §D.1 Step 3 trigger. The factory must accept it and
        // report `channels = 2`.
        let mut p = CodecParameters::audio(CodecId::new("mp3"));
        p.sample_rate = Some(44_100);
        p.channels = Some(2);
        p.sample_format = Some(SampleFormat::S16);
        p.bit_rate = Some(192_000);
        let enc = make_encoder_with_threshold_in_quiet(&p).expect("ltq factory stereo");
        assert_eq!(enc.output_params().channels, Some(2));
        assert_eq!(enc.output_params().bit_rate, Some(192_000));
    }

    #[test]
    fn make_encoder_with_threshold_in_quiet_rejects_more_than_two_channels() {
        // Same channel-count guard as `make_encoder` — multi-channel /
        // joint-stereo are out of scope for the threshold-in-quiet
        // factory.
        let mut p = CodecParameters::audio(CodecId::new("mp3"));
        p.sample_rate = Some(44_100);
        p.channels = Some(3);
        assert!(make_encoder_with_threshold_in_quiet(&p).is_err());
    }

    #[test]
    fn make_encoder_with_threshold_in_quiet_requires_sample_rate() {
        // Missing `sample_rate` is rejected just like `make_encoder`.
        let bare = CodecParameters::audio(CodecId::new("mp3"));
        assert!(make_encoder_with_threshold_in_quiet(&bare).is_err());
    }

    #[test]
    fn make_encoder_with_threshold_in_quiet_emits_self_decoding_stream() {
        // End-to-end sanity: the factory builds an encoder whose flushed
        // byte stream parses cleanly as a sequence of MPEG-1 Layer III
        // frames with the §2.4.2.3 sync word. Confirms the per-band
        // outer-loop dispatch wires through the trait wrapper without
        // changing the wire shape.
        use crate::frame::{parse_header, FrameWalker};

        let p = build_params(44_100, 128_000);
        let mut enc = make_encoder_with_threshold_in_quiet(&p).expect("ltq factory");
        // 4 frames of 440 Hz mono sine (1152 samples / frame).
        let pcm_bytes = sine_s16(SAMPLES_PER_FRAME_MPEG1 * 4, 440.0, 44_100.0, 0.3);
        let frame = AudioFrame {
            samples: (SAMPLES_PER_FRAME_MPEG1 * 4) as u32,
            pts: None,
            data: vec![pcm_bytes],
        };
        enc.send_frame(&Frame::Audio(frame)).expect("send");
        enc.flush().expect("flush");

        let mut packets: Vec<Packet> = Vec::new();
        loop {
            match enc.receive_packet() {
                Ok(pkt) => packets.push(pkt),
                Err(Error::Eof) => break,
                Err(e) => panic!("unexpected: {e}"),
            }
        }
        assert!(
            packets.len() >= 4,
            "expected ≥ 4 frames, got {}",
            packets.len()
        );
        // Reconstruct the wire stream and confirm every frame walks
        // cleanly: 12-bit sync, parse_header accepts the 4 header bytes.
        let mut wire = Vec::new();
        for pkt in &packets {
            wire.extend_from_slice(&pkt.data);
        }
        let frames: Vec<_> = FrameWalker::new(&wire).collect();
        assert!(
            frames.len() >= 4,
            "walker found {} frames, expected ≥ 4",
            frames.len(),
        );
        for f in &frames {
            let hdr_bytes: [u8; 4] = f.data[..4].try_into().expect("header bytes");
            let hdr = parse_header(&hdr_bytes).expect("header parses");
            assert_eq!(hdr.sample_rate_hz, 44_100);
        }
    }

    #[test]
    fn make_encoder_with_threshold_in_quiet_offset_constructs_and_reports_params() {
        // Caller-supplied offset path mirrors the spec-default path's
        // construction surface: same `output_params`, same `codec_id`,
        // just the threshold-vector dB translation differs.
        let p = build_params(44_100, 128_000);
        let enc =
            make_encoder_with_threshold_in_quiet_offset(&p, -6.0).expect("ltq offset factory");
        assert_eq!(enc.codec_id().as_str(), "mp3");
        assert_eq!(enc.output_params().sample_rate, Some(44_100));
        assert_eq!(enc.output_params().channels, Some(1));
        assert_eq!(enc.output_params().bit_rate, Some(128_000));
    }

    #[test]
    fn make_encoder_with_threshold_in_quiet_offset_emits_self_decoding_stream() {
        // End-to-end sanity for the custom-offset variant: encoded
        // stream parses cleanly as MPEG-1 Layer III frames at the
        // requested sample rate.
        use crate::frame::{parse_header, FrameWalker};

        let p = build_params(44_100, 128_000);
        let mut enc =
            make_encoder_with_threshold_in_quiet_offset(&p, -18.0).expect("ltq offset factory");
        let pcm_bytes = sine_s16(SAMPLES_PER_FRAME_MPEG1 * 4, 440.0, 44_100.0, 0.3);
        let frame = AudioFrame {
            samples: (SAMPLES_PER_FRAME_MPEG1 * 4) as u32,
            pts: None,
            data: vec![pcm_bytes],
        };
        enc.send_frame(&Frame::Audio(frame)).expect("send");
        enc.flush().expect("flush");

        let mut packets: Vec<Packet> = Vec::new();
        loop {
            match enc.receive_packet() {
                Ok(pkt) => packets.push(pkt),
                Err(Error::Eof) => break,
                Err(e) => panic!("unexpected: {e}"),
            }
        }
        assert!(
            packets.len() >= 4,
            "expected ≥ 4 frames, got {}",
            packets.len(),
        );
        let mut wire = Vec::new();
        for pkt in &packets {
            wire.extend_from_slice(&pkt.data);
        }
        let frames: Vec<_> = FrameWalker::new(&wire).collect();
        assert!(frames.len() >= 4, "walker found {}", frames.len());
        for f in &frames {
            let hdr_bytes: [u8; 4] = f.data[..4].try_into().expect("header bytes");
            let hdr = parse_header(&hdr_bytes).expect("header parses");
            assert_eq!(hdr.sample_rate_hz, 44_100);
        }
    }

    #[test]
    fn make_encoder_with_threshold_in_quiet_offset_rejects_more_than_two_channels() {
        let mut p = CodecParameters::audio(CodecId::new("mp3"));
        p.sample_rate = Some(44_100);
        p.channels = Some(3);
        assert!(make_encoder_with_threshold_in_quiet_offset(&p, -12.0).is_err());
    }

    #[test]
    fn make_encoder_with_threshold_in_quiet_offset_requires_sample_rate() {
        let bare = CodecParameters::audio(CodecId::new("mp3"));
        assert!(make_encoder_with_threshold_in_quiet_offset(&bare, -12.0).is_err());
    }

    #[test]
    fn make_encoder_joint_stereo_ms_builds_and_emits_joint_frames() {
        // Build the joint-stereo MS factory, push correlated stereo
        // PCM through send_frame, and check every emitted packet
        // carries `mode = '01'` (joint stereo) with
        // `mode_extension = '10'` (MS only).
        let mut p = CodecParameters::audio(CodecId::new("mp3"));
        p.sample_rate = Some(44_100);
        p.channels = Some(2);
        p.sample_format = Some(SampleFormat::S16);
        p.bit_rate = Some(192_000);
        let mut enc = make_encoder_joint_stereo_ms(&p).expect("ms factory");

        let pcm = sine_s16_stereo(SAMPLES_PER_FRAME_MPEG1 * 3, 440.0, 440.0, 44_100.0, 0.5);
        let frame = AudioFrame {
            samples: (SAMPLES_PER_FRAME_MPEG1 * 3) as u32,
            pts: None,
            data: vec![pcm],
        };
        enc.send_frame(&Frame::Audio(frame)).unwrap();
        enc.flush().unwrap();

        let mut packets = Vec::new();
        loop {
            match enc.receive_packet() {
                Ok(p) => packets.push(p),
                Err(Error::Eof) => break,
                Err(e) => panic!("unexpected: {e}"),
            }
        }
        assert!(packets.len() >= 3, "packet count {}", packets.len());
        for pkt in &packets {
            assert_eq!(pkt.data[0], 0xFF);
            assert_eq!(pkt.data[1] & 0xE0, 0xE0);
            // mode '01' (joint stereo).
            assert_eq!(pkt.data[3] & 0xC0, 0x40, "expected mode '01'");
            // mode_extension '10' (MS only).
            assert_eq!(pkt.data[3] & 0x30, 0x20, "expected mode_ext '10'");
        }
    }

    #[test]
    fn make_encoder_joint_stereo_ms_rejects_mono() {
        let mut p = CodecParameters::audio(CodecId::new("mp3"));
        p.sample_rate = Some(44_100);
        p.channels = Some(1);
        p.sample_format = Some(SampleFormat::S16);
        p.bit_rate = Some(128_000);
        assert!(make_encoder_joint_stereo_ms(&p).is_err());
    }

    #[test]
    fn make_encoder_joint_stereo_auto_emits_picked_mode_extension() {
        // Build the auto-picker factory and feed it correlated stereo
        // (perfect mono: L = R, side energy = 0). The per-frame picker
        // must pick MS (`mode_extension = '10'`) on every steady-state
        // frame. mode field stays '01' (joint stereo).
        let mut p = CodecParameters::audio(CodecId::new("mp3"));
        p.sample_rate = Some(44_100);
        p.channels = Some(2);
        p.sample_format = Some(SampleFormat::S16);
        p.bit_rate = Some(192_000);
        let mut enc = make_encoder_joint_stereo_auto(&p).expect("auto factory");

        // Correlated stereo: left and right carry the SAME 440 Hz tone.
        let pcm = sine_s16_stereo(SAMPLES_PER_FRAME_MPEG1 * 4, 440.0, 440.0, 44_100.0, 0.5);
        let frame = AudioFrame {
            samples: (SAMPLES_PER_FRAME_MPEG1 * 4) as u32,
            pts: None,
            data: vec![pcm],
        };
        enc.send_frame(&Frame::Audio(frame)).unwrap();
        enc.flush().unwrap();

        let mut packets = Vec::new();
        loop {
            match enc.receive_packet() {
                Ok(pkt) => packets.push(pkt),
                Err(Error::Eof) => break,
                Err(e) => panic!("unexpected: {e}"),
            }
        }
        // The first frame is the steady-state-cold pre-roll (zero MDCT
        // overlap, side energy can be non-trivial); from frame 1 onward
        // the picker must converge to MS for L == R input.
        assert!(packets.len() >= 4, "packet count {}", packets.len());
        for pkt in &packets[1..] {
            assert_eq!(pkt.data[0], 0xFF);
            assert_eq!(pkt.data[1] & 0xE0, 0xE0);
            // mode '01' (joint stereo) on every audio frame.
            assert_eq!(pkt.data[3] & 0xC0, 0x40, "expected mode '01' (joint)");
            // mode_extension '10' (MS only) when the picker chose MS.
            assert_eq!(pkt.data[3] & 0x30, 0x20, "expected MS on correlated input");
        }
    }

    #[test]
    fn make_encoder_joint_stereo_auto_rejects_mono() {
        // Mono input cannot be joint-stereo.
        let mut p = CodecParameters::audio(CodecId::new("mp3"));
        p.sample_rate = Some(44_100);
        p.channels = Some(1);
        p.sample_format = Some(SampleFormat::S16);
        p.bit_rate = Some(128_000);
        assert!(make_encoder_joint_stereo_auto(&p).is_err());
        assert!(make_encoder_joint_stereo_auto_with_threshold(&p, 0.4).is_err());
    }

    #[test]
    fn make_encoder_joint_stereo_auto_requires_sample_rate() {
        let mut p = CodecParameters::audio(CodecId::new("mp3"));
        p.channels = Some(2);
        p.sample_format = Some(SampleFormat::S16);
        p.bit_rate = Some(192_000);
        assert!(make_encoder_joint_stereo_auto(&p).is_err());
    }

    #[test]
    fn make_encoder_joint_stereo_auto_defaults_bitrate_to_192k() {
        // No `bit_rate` field — the factory should default to 192_000
        // and report it on the output parameters.
        let mut p = CodecParameters::audio(CodecId::new("mp3"));
        p.sample_rate = Some(44_100);
        p.channels = Some(2);
        p.sample_format = Some(SampleFormat::S16);
        let enc = make_encoder_joint_stereo_auto(&p).expect("auto factory");
        assert_eq!(enc.output_params().bit_rate, Some(192_000));
        assert_eq!(enc.output_params().channels, Some(2));
        assert_eq!(
            enc.output_params().tag,
            Some(CodecTag::wave_format(WAVE_FORMAT_MP3))
        );
    }

    #[test]
    fn make_encoder_joint_stereo_auto_with_threshold_threshold_zero_forces_lr() {
        // `threshold = 0` means "MS only when side energy is exactly
        // zero". For a steady-state non-trivial stereo signal the
        // picker will reject MS on every frame and the wire
        // `mode_extension` stays '00'.
        let mut p = CodecParameters::audio(CodecId::new("mp3"));
        p.sample_rate = Some(44_100);
        p.channels = Some(2);
        p.sample_format = Some(SampleFormat::S16);
        p.bit_rate = Some(192_000);
        let mut enc =
            make_encoder_joint_stereo_auto_with_threshold(&p, 0.0).expect("auto with threshold");

        // Distinct L (440 Hz) and R (880 Hz) tones → non-zero side energy.
        let pcm = sine_s16_stereo(SAMPLES_PER_FRAME_MPEG1 * 3, 440.0, 880.0, 44_100.0, 0.5);
        let frame = AudioFrame {
            samples: (SAMPLES_PER_FRAME_MPEG1 * 3) as u32,
            pts: None,
            data: vec![pcm],
        };
        enc.send_frame(&Frame::Audio(frame)).unwrap();
        enc.flush().unwrap();

        let mut packets = Vec::new();
        loop {
            match enc.receive_packet() {
                Ok(pkt) => packets.push(pkt),
                Err(Error::Eof) => break,
                Err(e) => panic!("unexpected: {e}"),
            }
        }
        assert!(packets.len() >= 3);
        for pkt in &packets {
            // mode '01' (joint stereo): the wire flag stays joint
            // regardless of the per-frame mode_extension decision.
            assert_eq!(pkt.data[3] & 0xC0, 0x40, "expected mode '01' (joint)");
            // mode_extension '00' (no MS, no intensity): threshold = 0
            // suppresses MS for any non-trivial side energy.
            assert_eq!(
                pkt.data[3] & 0x30,
                0x00,
                "expected mode_ext '00' under threshold=0"
            );
        }
    }

    #[test]
    fn make_encoder_joint_stereo_auto_with_threshold_clamps_out_of_range() {
        // The underlying `with_ms_auto_threshold` clamps inputs to
        // [0.0, 1.0]; the factory should accept both extremes without
        // erroring (and build a working encoder).
        let mut p = CodecParameters::audio(CodecId::new("mp3"));
        p.sample_rate = Some(44_100);
        p.channels = Some(2);
        p.sample_format = Some(SampleFormat::S16);
        p.bit_rate = Some(192_000);
        assert!(make_encoder_joint_stereo_auto_with_threshold(&p, -1.0).is_ok());
        assert!(make_encoder_joint_stereo_auto_with_threshold(&p, 2.5).is_ok());
    }
}
