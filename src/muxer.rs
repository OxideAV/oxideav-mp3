//! Raw MPEG audio elementary-stream muxer (`.mp3` output).
//!
//! An `.mp3` file is the bare sequence of MPEG audio frames (ISO/IEC
//! 11172-3 §2.4.1 / ISO/IEC 13818-3): each [`Packet`] the encoder emits
//! is one complete self-delimiting frame and is written verbatim.
//!
//! For Layer III streams the muxer additionally leads the file with a
//! Xing / Info information frame ([`crate::xing_info`]): a silent frame
//! at the stream's own header geometry whose main-data slot carries the
//! total frame count, the audio byte count and a 100-entry seek table.
//! The frame is reserved when the first packet arrives (its header
//! fixes version / rate / mode / bit rate) and back-patched in
//! [`Muxer::write_trailer`] once the totals are known. The tag id is
//! `Info` for a constant-bit-rate stream and `Xing` when the per-frame
//! bit rate varied. Decoders that do not know the tag decode it as one
//! frame of silence; [`crate::Mp3Demuxer`] and other tag-aware readers
//! use it for duration and seeking.
//!
//! If the output cannot seek back, the placeholder info frame keeps
//! zero totals; the audio frames themselves are unaffected.

use std::io::{Seek, SeekFrom, Write};

use oxideav_core::{Error, MediaType, Muxer, Packet, Result, StreamInfo, WriteSeek};

use crate::demuxer::{XingTagId, FORMAT_NAME};
use crate::frame::{parse_header, Layer, Mp3FrameHeader};
use crate::xing_info::{build_info_frame, flag_bit, XingTagSpec};

/// Codec ids the raw MPEG-audio muxer carries.
const ACCEPTED_CODECS: &[&str] = &["mp3", "mp2", "mp1"];

/// Open the raw MPEG-audio muxer over `output` for `streams` (exactly
/// one MPEG audio stream).
pub fn open_muxer(output: Box<dyn WriteSeek>, streams: &[StreamInfo]) -> Result<Box<dyn Muxer>> {
    if streams.len() != 1 {
        return Err(Error::unsupported(format!(
            "mp3 muxer: exactly one audio stream supported (got {})",
            streams.len()
        )));
    }
    let s = &streams[0];
    if s.params.media_type != MediaType::Audio {
        return Err(Error::invalid("mp3 muxer: stream must be audio"));
    }
    let id = s.params.codec_id.as_str();
    if !ACCEPTED_CODECS.contains(&id) {
        return Err(Error::unsupported(format!(
            "mp3 muxer: codec '{id}' cannot be stored as a raw MPEG audio stream \
             (mp3 / mp2 / mp1 only)"
        )));
    }
    Ok(Box::new(Mp3Muxer::new(output)))
}

/// Which optional Xing fields fit the carrier frame.
#[derive(Debug, Clone, Copy)]
struct InfoSlot {
    /// Byte offset of the reserved info frame in the output.
    pos: u64,
    /// Header the carrier frame was built from.
    header: Mp3FrameHeader,
    /// Flag word the reserved frame was sized for.
    flags: u32,
}

/// Raw MPEG audio stream writer. See the module docs.
pub struct Mp3Muxer {
    output: Box<dyn WriteSeek>,
    header_written: bool,
    trailer_written: bool,
    info: Option<InfoSlot>,
    /// Byte offset (relative to the first audio frame) of every audio
    /// frame written, for the seek table.
    frame_offsets: Vec<u64>,
    /// Audio bytes written after the info frame.
    audio_bytes: u64,
    /// Bit-rate index of the first frame; `None` once frames disagree.
    cbr_index: Option<u8>,
    vbr: bool,
}

impl std::fmt::Debug for Mp3Muxer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mp3Muxer")
            .field("frames", &self.frame_offsets.len())
            .field("audio_bytes", &self.audio_bytes)
            .field("vbr", &self.vbr)
            .finish_non_exhaustive()
    }
}

impl Mp3Muxer {
    fn new(output: Box<dyn WriteSeek>) -> Self {
        Self {
            output,
            header_written: false,
            trailer_written: false,
            info: None,
            frame_offsets: Vec::new(),
            audio_bytes: 0,
            cbr_index: None,
            vbr: false,
        }
    }

    /// Reserve the info frame in front of the first audio frame.
    fn reserve_info(&mut self, first: &Mp3FrameHeader) -> Result<()> {
        if first.layer != Layer::LayerIII || first.is_free_format() {
            return Ok(());
        }
        let mut header = *first;
        header.padding = false;
        header.crc_protected = false;
        // Largest field set that fits the carrier's main-data slot.
        for flags in [
            flag_bit::FRAMES | flag_bit::BYTES | flag_bit::TOC,
            flag_bit::FRAMES | flag_bit::BYTES,
        ] {
            let spec = spec_for(XingTagId::Info, flags, 0, 0, [0; 100]);
            if let Ok(frame) = build_info_frame(&header, &spec) {
                let pos = self.output.stream_position()?;
                self.output.write_all(&frame)?;
                self.info = Some(InfoSlot { pos, header, flags });
                return Ok(());
            }
        }
        Ok(())
    }

    /// Back-patch the reserved info frame with the final totals.
    fn patch_info(&mut self) -> Result<()> {
        let Some(slot) = self.info else {
            return Ok(());
        };
        let frames = u32::try_from(self.frame_offsets.len()).unwrap_or(u32::MAX);
        let bytes = u32::try_from(self.audio_bytes).unwrap_or(u32::MAX);
        let toc = seek_table(&self.frame_offsets, self.audio_bytes);
        let id = if self.vbr {
            XingTagId::Xing
        } else {
            XingTagId::Info
        };
        let spec = spec_for(id, slot.flags, frames, bytes, toc);
        let frame = build_info_frame(&slot.header, &spec)
            .map_err(|e| Error::other(format!("mp3 muxer: info frame: {e}")))?;
        let end = self.output.stream_position()?;
        if self.output.seek(SeekFrom::Start(slot.pos)).is_err() {
            // Not seekable: leave the placeholder totals.
            return Ok(());
        }
        self.output.write_all(&frame)?;
        self.output.seek(SeekFrom::Start(end))?;
        Ok(())
    }
}

/// Assemble a [`XingTagSpec`] carrying the fields named by `flags`.
fn spec_for(id: XingTagId, flags: u32, frames: u32, bytes: u32, toc: [u8; 100]) -> XingTagSpec {
    XingTagSpec {
        id,
        flags,
        frames: (flags & flag_bit::FRAMES != 0).then_some(frames),
        bytes: (flags & flag_bit::BYTES != 0).then_some(bytes),
        toc: (flags & flag_bit::TOC != 0).then_some(toc),
        quality: None,
    }
}

/// Xing seek table: entry `i` is `floor(256 · offset / total_bytes)` of
/// the frame that starts playback position `i / 100`.
fn seek_table(frame_offsets: &[u64], total_bytes: u64) -> [u8; 100] {
    let mut toc = [0u8; 100];
    let n = frame_offsets.len();
    if n == 0 || total_bytes == 0 {
        return toc;
    }
    for (i, slot) in toc.iter_mut().enumerate() {
        let frame = (i * n) / 100;
        let off = frame_offsets[frame.min(n - 1)];
        *slot = ((off * 256) / total_bytes).min(255) as u8;
    }
    toc
}

impl Muxer for Mp3Muxer {
    fn format_name(&self) -> &str {
        FORMAT_NAME
    }

    fn write_header(&mut self) -> Result<()> {
        if self.header_written {
            return Err(Error::other("mp3 muxer: write_header called twice"));
        }
        self.header_written = true;
        Ok(())
    }

    fn write_packet(&mut self, packet: &Packet) -> Result<()> {
        if !self.header_written {
            return Err(Error::other("mp3 muxer: write_header not called"));
        }
        if self.trailer_written {
            return Err(Error::other("mp3 muxer: write_packet after trailer"));
        }
        if packet.data.is_empty() {
            return Ok(());
        }
        let header = parse_header(&packet.data).map_err(|e| {
            Error::invalid(format!("mp3 muxer: packet is not an MPEG frame: {e:?}"))
        })?;
        if self.frame_offsets.is_empty() {
            self.reserve_info(&header)?;
            self.cbr_index = Some(header.bitrate_index);
        } else if self.cbr_index != Some(header.bitrate_index) {
            self.vbr = true;
        }
        self.frame_offsets.push(self.audio_bytes);
        self.output.write_all(&packet.data)?;
        self.audio_bytes += packet.data.len() as u64;
        Ok(())
    }

    fn write_trailer(&mut self) -> Result<()> {
        if self.trailer_written {
            return Ok(());
        }
        self.trailer_written = true;
        self.patch_info()?;
        self.output.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::demuxer::{parse_xing_info, side_info_len};
    use oxideav_core::{CodecId, CodecParameters, TimeBase};
    use std::io::Cursor;
    use std::sync::{Arc, Mutex};

    /// `WriteSeek` over a shared buffer so the test can read it back.
    #[derive(Clone, Default)]
    struct Shared(Arc<Mutex<Cursor<Vec<u8>>>>);
    impl Write for Shared {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().write(b)
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    impl Seek for Shared {
        fn seek(&mut self, p: SeekFrom) -> std::io::Result<u64> {
            self.0.lock().unwrap().seek(p)
        }
    }

    fn stream(codec: &str) -> StreamInfo {
        let mut params = CodecParameters::audio(CodecId::new(codec));
        params.sample_rate = Some(44_100);
        params.channels = Some(2);
        StreamInfo {
            index: 0,
            time_base: TimeBase::new(1, 44_100),
            duration: None,
            start_time: Some(0),
            params,
        }
    }

    fn encoded_frames(n_samples: usize) -> Vec<Vec<u8>> {
        let mut enc = crate::Mp3Encoder::new(128, 44_100, crate::frame::ChannelMode::Stereo)
            .expect("encoder");
        let pcm: Vec<i16> = (0..n_samples * 2)
            .map(|i| ((i as f32 * 0.05).sin() * 8000.0) as i16)
            .collect();
        enc.push_samples(&pcm).expect("push");
        let mut bytes = Vec::new();
        enc.finish(&mut bytes).expect("finish");
        crate::frame::FrameWalker::new(&bytes)
            .map(|f| f.data.to_vec())
            .collect()
    }

    #[test]
    fn rejects_non_mpeg_codecs_and_multiple_streams() {
        let out = Box::new(Shared::default());
        assert!(open_muxer(out.clone(), &[stream("opus")]).is_err());
        assert!(open_muxer(out, &[stream("mp3"), stream("mp3")]).is_err());
    }

    #[test]
    fn writes_info_frame_then_verbatim_frames() {
        let frames = encoded_frames(44_100 / 2);
        assert!(frames.len() > 5);
        let shared = Shared::default();
        let mut mux = open_muxer(Box::new(shared.clone()), &[stream("mp3")]).unwrap();
        mux.write_header().unwrap();
        for f in &frames {
            let p = Packet::new(0, TimeBase::new(1, 44_100), f.clone());
            mux.write_packet(&p).unwrap();
        }
        mux.write_trailer().unwrap();
        let out = shared.0.lock().unwrap().get_ref().clone();

        let first = parse_header(&out).expect("info frame header");
        let info_len = first.frame_len().unwrap();
        let tag = parse_xing_info(
            &out[..info_len],
            side_info_len(first.version, first.channel_count()),
        )
        .expect("tag");
        assert_eq!(tag.id, XingTagId::Info);
        assert_eq!(tag.frames, Some(frames.len() as u32));
        let audio: Vec<u8> = frames.concat();
        assert_eq!(tag.bytes, Some(audio.len() as u32));
        assert_eq!(&out[info_len..], &audio[..]);
        let toc = tag.toc.expect("toc");
        assert_eq!(toc[0], 0);
        assert!(toc.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn round_trips_through_the_demuxer() {
        let frames = encoded_frames(44_100);
        let shared = Shared::default();
        let mut mux = open_muxer(Box::new(shared.clone()), &[stream("mp3")]).unwrap();
        mux.write_header().unwrap();
        for f in &frames {
            mux.write_packet(&Packet::new(0, TimeBase::new(1, 44_100), f.clone()))
                .unwrap();
        }
        mux.write_trailer().unwrap();
        let out = shared.0.lock().unwrap().get_ref().clone();
        let mut demux = crate::Mp3Demuxer::open(Box::new(Cursor::new(out))).expect("demux");
        let mut n = 0;
        while let Ok(p) = oxideav_core::Demuxer::next_packet(&mut demux) {
            assert_eq!(p.data, frames[n]);
            n += 1;
        }
        assert_eq!(n, frames.len(), "the info frame is metadata, not audio");
    }

    #[test]
    fn seek_table_is_monotone_and_scaled() {
        let offs: Vec<u64> = (0..1000).map(|i| i * 100).collect();
        let toc = seek_table(&offs, 100_000);
        assert_eq!(toc[0], 0);
        assert_eq!(toc[50], 128);
        assert!(toc.windows(2).all(|w| w[0] <= w[1]));
    }
}
