//! MP3 container demuxer.
//!
//! An `oxideav_core::Demuxer` over an MPEG audio elementary stream: it
//! skips the ID3v2/ID3v1 frontmatter and trailer, parses the Xing/Info
//! VBR-info frame and its gapless extension, walks every frame header
//! once at open time (exact duration, seek index), and streams packets
//! one MPEG audio frame at a time — across mid-stream changes of
//! sampling frequency, MPEG version, channel mode, bitrate and free
//! format (frame-boundary discovery lives in `crate::framing`).
//!
//! # What is in `docs/audio/mp3/` and what is not
//!
//! The framing layer (the four-byte header parse, the per-frame byte
//! length, the syncword-driven mid-stream resync) is fully described
//! by ISO/IEC 11172-3 §2.4.1.3 / §2.4.2.3 + ISO/IEC 13818-3 §2.4.2.3
//! (both PDFs are on disk under `docs/audio/mp3/`) and was already
//! implemented in [`crate::frame`].
//!
//! The **ID3v1** trailer is documented inline in
//! `docs/audio/mp3/datavoyage-mpgscript-mpeghdr.html` §"MPEG Audio
//! Tag ID3v1" (positions 0..=127 fixed, "TAG" magic at bytes 0..=2).
//!
//! The **ID3v2** tag header is documented in `docs/container/id3/`:
//! both `id3v2.3.0.html` (the v2.3 spec) and
//! `id3v2.4.0-structure.html` (the v2.4 spec) describe the 10-byte
//! header — "ID3" magic at bytes 0..=2, version-major / version-minor
//! at 3..=4, flags at 5, and a 4-byte synchsafe size at 6..=9 giving
//! the body length (not including the 10-byte header itself). A
//! v2.4-only optional footer adds another 10 bytes after the body
//! when flag bit 0x10 is set.
//!
//! **The Xing / Info / VBRI VBR-info frame layout is not yet
//! staged in `docs/audio/mp3/`.** The prompt for round 121 names a
//! pending "LAME-extension-staging task" that has not landed in
//! `docs/` yet. We implement enough of the Xing/Info parse to (a)
//! detect the magic at the well-known position immediately after the
//! Layer III side-info bytes of the first frame and (b) extract the
//! four fields the prompt enumerates (`frames`, `bytes`, `toc`,
//! `quality`) guarded by the four low bits of a big-endian 32-bit
//! flag word that follows the magic. **Every numeric field offset
//! and width in [`parse_xing_info`] is verified byte-for-byte
//! against the two on-disk fixtures
//! `docs/audio/mp3/fixtures/layer3-with-xing-vbri-tag/input.mp3`
//! (Xing) and `docs/audio/mp3/fixtures/layer3-with-id3v2-tag/input.mp3`
//! (Info) and their companion `trace.txt`** files, which record the
//! parser's expected output:
//!
//! ```text
//! XING_HEADER  tag=Xing  flags=0x0000000f  frames=32  bytes=6114  toc_present=1  quality=0
//! XING_HEADER  tag=Info  flags=0x0000000f  frames=32  bytes=13791 toc_present=1  quality=0
//! ```
//!
//! When the canonical Xing/Info layout doc lands in `docs/audio/mp3/`,
//! re-validate this module against it and drop the fixture-based
//! provenance note.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

use crate::framing::{self, CachedInput, FramerState, FramingKey, Located};

use oxideav_core::{
    CodecId, CodecParameters, CodecResolver, CodecTag, ContainerRegistry, Demuxer, Error, Packet,
    ProbeData, ReadSeek, Result, StreamInfo, TimeBase,
};

use crate::frame::{Layer, Mp3FrameHeader, MpegVersion, parse_header};
use crate::lame_tag::{LameTag, parse_lame_tag};
use crate::side_info::{
    SIDE_INFO_BYTES_LSF_MONO, SIDE_INFO_BYTES_LSF_STEREO, SIDE_INFO_BYTES_MONO,
    SIDE_INFO_BYTES_STEREO,
};

/// Format-registry name used by the container registry.
pub const FORMAT_NAME: &str = "mp3";

/// CodecId stream parameters carry for the audio inside an MP3
/// container. The codec crate itself doesn't yet expose a decoder,
/// but pipelines that go through `CodecResolver` use this identifier.
pub const CODEC_ID_STR: &str = "mp3";

/// `WAVEFORMATEX::wFormatTag` value historically assigned to MP3 by
/// Microsoft for the Windows AVI / WAV container ecosystem (0x0055,
/// "MPEGLAYER3"). Surfaced on `CodecParameters::tag` so re-muxers can
/// preserve the original on-wire tag.
pub const WAVE_FORMAT_MP3: u16 = 0x0055;

/// MP3 metadata extracted from the on-disk container frontmatter.
///
/// We only ever read enough of the ID3v2 header to advance past the
/// tag — we do not currently parse individual frames. The
/// `id3v2_present` / `id3v1_present` flags expose whether either tag
/// was found so callers (and tests) can confirm the demuxer noticed
/// them.
#[derive(Debug, Clone, Copy, Default)]
#[non_exhaustive]
pub struct Mp3Tags {
    /// `true` when an ID3v2 tag was detected at offset 0.
    pub id3v2_present: bool,
    /// Byte length of the ID3v2 tag (the 10-byte header + body +
    /// optional footer, if any). Zero when `!id3v2_present`.
    pub id3v2_size: u64,
    /// `true` when the last 128 bytes of the file start with "TAG"
    /// (ID3v1 / ID3v1.1).
    pub id3v1_present: bool,
}

/// Xing / Info VBR information frame.
///
/// "Xing" magic indicates a true VBR file; "Info" magic indicates a
/// CBR file that some encoders nevertheless decorate with the
/// same structure for compatibility with VBR-aware seeking. The two
/// payload layouts are identical — the magic alone selects the tag —
/// so we collapse them to one type with an [`XingTagId`] discriminant.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct XingTag {
    /// Whether the magic was "Xing" or "Info".
    pub id: XingTagId,
    /// Bit 0: `frames` present. Bit 1: `bytes` present. Bit 2: `toc`
    /// present. Bit 3: `quality` present. Higher bits, if set,
    /// indicate fields the prompt does not enumerate; they are
    /// preserved on the struct for forwards-compatibility but are
    /// otherwise ignored at this layer.
    pub flags: u32,
    /// Total MPEG-audio frame count after this info frame, when
    /// flag bit 0 is set.
    pub frames: Option<u32>,
    /// Total compressed-audio byte count after this info frame, when
    /// flag bit 1 is set.
    pub bytes: Option<u32>,
    /// 100-entry seek table when flag bit 2 is set. Each entry `i`
    /// holds `floor(256 * file_offset / bytes)` for the playback
    /// position `i / 100` of the file, so `bytes * toc[i] / 256` is
    /// the byte offset for the percentile `i`.
    pub toc: Option<[u8; 100]>,
    /// Encoder quality indicator (0..=100), when flag bit 3 is set.
    pub quality: Option<u32>,
}

/// Xing-versus-Info discriminator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum XingTagId {
    /// Encoder believes the stream is variable-bitrate.
    Xing,
    /// Encoder believes the stream is constant-bitrate but still
    /// emitted the info-frame layout (by historical convention).
    Info,
}

/// Parse a candidate Xing / Info info frame given the entire first
/// MPEG audio frame's payload (header + optional CRC + side info +
/// main data) and the side-info length implied by the frame header.
///
/// Returns `None` when the bytes immediately after the side-info
/// region don't carry the "Xing" or "Info" four-byte magic.
///
/// # Layout (Xing / Info)
///
/// Verified byte-for-byte against
/// `docs/audio/mp3/fixtures/layer3-with-xing-vbri-tag/input.mp3` and
/// `docs/audio/mp3/fixtures/layer3-with-id3v2-tag/input.mp3` and
/// their companion `trace.txt` files, which record the parser's
/// expected output. Pending a canonical spec doc under
/// `docs/audio/mp3/` (see module doc).
///
/// ```text
/// +0..+3   "Xing" or "Info"
/// +4..+7   flags (BE u32, low four bits enumerated)
/// +8..+11  frames     (BE u32, present iff flags & 0x1)
/// +12..+15 bytes      (BE u32, present iff flags & 0x2)
/// +16..+115 toc[100]  (raw bytes, present iff flags & 0x4)
/// +116..+119 quality  (BE u32, present iff flags & 0x8)
/// ```
///
/// The fields are packed in flag-bit order: when only `frames` is
/// set the `bytes`/`toc`/`quality` fields are absent and the next
/// optional field (if any) starts at offset +8. This module assembles
/// them in increasing-offset order based on the four guard bits.
pub fn parse_xing_info(frame_payload: &[u8], side_info_bytes: usize) -> Option<XingTag> {
    // Header is 4 bytes; CRC (when present) is another 2 — but the
    // caller's `side_info_bytes` is the value we got from the header,
    // and the prompt locates the magic "after the side-info bytes of
    // the first frame", which (per ISO/IEC 11172-3 §2.4.1.7) starts
    // immediately after the optional CRC. We need the header byte to
    // know whether CRC is present, so re-parse it here.
    if frame_payload.len() < 4 {
        return None;
    }
    let header = parse_header(&frame_payload[..4]).ok()?;
    // Xing/Info is a Layer III convention only.
    if header.layer != Layer::LayerIII {
        return None;
    }
    let crc_bytes = if header.crc_protected { 2 } else { 0 };
    let magic_offset = 4 + crc_bytes + side_info_bytes;
    let magic_end = magic_offset + 4;
    if magic_end > frame_payload.len() {
        return None;
    }
    let magic = &frame_payload[magic_offset..magic_end];
    let id = if magic == b"Xing" {
        XingTagId::Xing
    } else if magic == b"Info" {
        XingTagId::Info
    } else {
        return None;
    };

    let flags_offset = magic_end;
    let flags_end = flags_offset + 4;
    if flags_end > frame_payload.len() {
        return None;
    }
    let flags = u32::from_be_bytes([
        frame_payload[flags_offset],
        frame_payload[flags_offset + 1],
        frame_payload[flags_offset + 2],
        frame_payload[flags_offset + 3],
    ]);

    let mut cursor = flags_end;
    let mut frames = None;
    let mut bytes = None;
    let mut toc = None;
    let mut quality = None;
    let read_be_u32 = |buf: &[u8], at: usize| -> Option<u32> {
        if at + 4 > buf.len() {
            return None;
        }
        Some(u32::from_be_bytes([
            buf[at],
            buf[at + 1],
            buf[at + 2],
            buf[at + 3],
        ]))
    };
    if flags & 0x1 != 0 {
        frames = Some(read_be_u32(frame_payload, cursor)?);
        cursor += 4;
    }
    if flags & 0x2 != 0 {
        bytes = Some(read_be_u32(frame_payload, cursor)?);
        cursor += 4;
    }
    if flags & 0x4 != 0 {
        if cursor + 100 > frame_payload.len() {
            return None;
        }
        let mut t = [0u8; 100];
        t.copy_from_slice(&frame_payload[cursor..cursor + 100]);
        toc = Some(t);
        cursor += 100;
    }
    if flags & 0x8 != 0 {
        quality = Some(read_be_u32(frame_payload, cursor)?);
        // cursor advance not needed past the last enumerated field
    }
    Some(XingTag {
        id,
        flags,
        frames,
        bytes,
        toc,
        quality,
    })
}

/// Compute the byte offset of the LAME-extension magic inside an MP3
/// carrier frame, given the [`XingTag`] that precedes it.
///
/// Per `docs/audio/mp3/lame-xing-info-tag.md`, on the worked
/// MPEG-1-stereo example with **all four** Xing flags set the LAME
/// magic sits at absolute `$9A` and the Xing magic at `$24` — i.e.
/// 118 bytes after the Xing magic. The 118 figure breaks down as
/// `4 (Xing magic) + 4 (flags) + 4 (frames) + 4 (bytes) + 100 (toc) +
/// 4 (quality) − 2 (overlap between the trailing quality bytes and
/// the leading encoder-version bytes)`. The returned offset is
/// **frame-relative** — zero is the `0xFFE0...` sync byte at the
/// start of the frame.
///
/// Returns `None` when fewer than all four Xing flag bits are set.
/// The staged doc only documents the all-flags-set carrier-frame
/// layout, and we refuse to guess where the LAME magic lands for the
/// flag combinations the staged doc does not cover (DOCS-GAP — see
/// the module-level comment).
#[must_use]
#[doc(hidden)] // internal: demuxer layout helper, public for tests only (not stable API)
pub fn lame_magic_offset(
    header_bytes: usize,
    side_info_bytes: usize,
    xing: &XingTag,
) -> Option<usize> {
    const ALL_FOUR: u32 = 0x0F;
    if xing.flags & ALL_FOUR != ALL_FOUR {
        return None;
    }
    // header + (CRC bytes — caller already accounted for them in
    // header_bytes if needed) + side info + Xing-magic-relative 118.
    Some(header_bytes + side_info_bytes + crate::lame_tag::LAME_MAGIC_OFFSET_ALL_FLAGS)
}

/// Return the Layer-III side-info length implied by an MPEG version
/// and channel count, per the four `SIDE_INFO_BYTES_*` constants in
/// [`crate::side_info`].
#[must_use]
#[doc(hidden)] // internal: demuxer layout helper, public for tests only (not stable API)
pub fn side_info_len(version: MpegVersion, channels: u8) -> usize {
    // MPEG-2.5 inherits the LSF side-info layout per the
    // `MpegVersion::Mpeg25` doc-comment + `MPEG-2.5-GAP.md`.
    match (version.is_lsf(), channels) {
        (false, 1) => SIDE_INFO_BYTES_MONO,
        (false, _) => SIDE_INFO_BYTES_STEREO,
        (true, 1) => SIDE_INFO_BYTES_LSF_MONO,
        (true, _) => SIDE_INFO_BYTES_LSF_STEREO,
    }
}

/// Read up to `buf.len()` bytes into `buf`, returning the number
/// actually read. Mirrors `Read::read_exact` but tolerates short
/// reads at EOF without erroring.
fn read_up_to<R: Read + ?Sized>(input: &mut R, buf: &mut [u8]) -> std::io::Result<usize> {
    let mut got = 0;
    while got < buf.len() {
        match input.read(&mut buf[got..]) {
            Ok(0) => break,
            Ok(n) => got += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(got)
}

/// Decode the synchsafe 28-bit size encoded in four 7-bit big-endian
/// bytes (the standard ID3v2 frontmatter size encoding).
fn synchsafe_size(b: [u8; 4]) -> u32 {
    ((b[0] as u32) << 21) | ((b[1] as u32) << 14) | ((b[2] as u32) << 7) | (b[3] as u32)
}

/// Inspect the first 10 bytes for an ID3v2 tag and, when present,
/// return its on-disk size (header + body + optional v2.4 footer).
///
/// Layout per `docs/container/id3/id3v2.3.0.html` and
/// `docs/container/id3/id3v2.4.0-structure.html`:
///
/// ```text
/// "ID3" magic (3) | version_major (1) | version_minor (1) |
/// flags (1)      | synchsafe_size (4)
/// ```
///
/// The `flags & 0x10` bit (v2.4 footer present) adds 10 bytes to the
/// total. The synchsafe size does NOT include the 10-byte header or
/// the optional footer.
fn id3v2_total_len(head10: &[u8]) -> Option<u64> {
    if head10.len() < 10 {
        return None;
    }
    if &head10[..3] != b"ID3" {
        return None;
    }
    let flags = head10[5];
    let size = synchsafe_size([head10[6], head10[7], head10[8], head10[9]]) as u64;
    let footer = if flags & 0x10 != 0 { 10 } else { 0 };
    Some(10 + size + footer)
}

/// Read the trailing 128 bytes and return `true` when they start
/// with the ID3v1 "TAG" magic.
fn detect_id3v1(input: &mut Box<dyn ReadSeek>, total_len: u64) -> Result<bool> {
    if total_len < 128 {
        return Ok(false);
    }
    let saved = input.stream_position()?;
    input.seek(SeekFrom::Start(total_len - 128))?;
    let mut tag = [0u8; 3];
    let n = read_up_to(input.as_mut(), &mut tag)?;
    input.seek(SeekFrom::Start(saved))?;
    Ok(n == 3 && &tag == b"TAG")
}

/// Container-level probe — `register_probe` invokes this with the
/// first ~256 KB of an unknown input to score it as MP3.
///
/// Returns a high score on the canonical ID3v2 / "Xing" / "Info" /
/// frame-sync prefixes, a lower score on a bare frame sync, and zero
/// otherwise. The extension hint `.mp3` (or `.mp2`/`.mp1`) bumps the
/// confidence for streams that only carry a frame-sync prefix.
pub fn probe(p: &ProbeData) -> u8 {
    // ID3v2 prefix at the start — strongest signal, since the tag
    // could otherwise be a different audio container that also
    // tolerates ID3v2 (FLAC, AAC). Validate that the bytes following
    // the tag start with an MPEG audio frame sync to break ties.
    if p.buf.len() >= 10
        && &p.buf[..3] == b"ID3"
        && let Some(total) = id3v2_total_len(&p.buf[..10])
    {
        let off = total as usize;
        if off + 4 <= p.buf.len()
            && p.buf[off] == 0xFF
            && (p.buf[off + 1] & 0xE0) == 0xE0
            && parse_header(&p.buf[off..off + 4])
                .map(|h| h.layer == Layer::LayerIII)
                .unwrap_or(false)
        {
            return 100;
        }
    }
    // Bare frame sync at offset 0 with a valid Layer III header.
    if p.buf.len() >= 4
        && p.buf[0] == 0xFF
        && (p.buf[1] & 0xE0) == 0xE0
        && let Ok(h) = parse_header(&p.buf[..4])
        && h.layer == Layer::LayerIII
    {
        return if matches!(p.ext, Some("mp3") | Some("mp2") | Some("mp1")) {
            100
        } else {
            75
        };
    }
    // Extension-only fallback handled by the registry; we return 0
    // when no signature matched so the container registry can fall
    // back to the extension table.
    0
}

/// Install the MP3 demuxer + extension + probe into a
/// `ContainerRegistry`. Called by [`crate::register`].
pub fn register_container(reg: &mut ContainerRegistry) {
    reg.register_demuxer(FORMAT_NAME, open_demuxer);
    reg.register_muxer(FORMAT_NAME, crate::muxer::open_muxer);
    reg.register_extension("mp3", FORMAT_NAME);
    reg.register_extension("mp2", FORMAT_NAME);
    reg.register_extension("mp1", FORMAT_NAME);
    reg.register_probe(FORMAT_NAME, probe);
}

/// `OpenDemuxerFn` entry point — opens an MP3 stream as a `Demuxer`.
fn open_demuxer(input: Box<dyn ReadSeek>, _codecs: &dyn CodecResolver) -> Result<Box<dyn Demuxer>> {
    let demuxer = Mp3Demuxer::open(input)?;
    Ok(Box::new(demuxer))
}

/// The MP3 demuxer.
///
/// # Mid-stream header changes
///
/// The stream may switch sampling frequency, MPEG version (MPEG-1 /
/// MPEG-2 LSF / MPEG-2.5), layer, channel mode, bitrate or free-format
/// slot count at any frame boundary — a byte-concatenation of files
/// does exactly that, and may also carry the later files' ID3v2 tags,
/// ID3v1 trailers and Xing / Info frames in between. Every audio frame
/// is emitted once; tags and info frames are stepped over. A header
/// exactly where the previous frame ended that keeps the version /
/// layer / sampling frequency is trusted; one that switches them must
/// be followed by a header with its own key; a candidate found by
/// scanning must chain three more correctly-spaced headers, so a
/// resync cannot lock onto a false sync inside main data.
///
/// **Time base.** The stream declares `1 / f` where `f` is the
/// sampling frequency of the first audio frame, the same as for a
/// stream that never switches, so packet timestamps of such a stream
/// are plain sample counts. Across a switch the running time is kept
/// exactly in units of 1 / 14 112 000 s (the least common multiple of
/// every MPEG audio sampling frequency, in which every frame duration
/// is whole) and each packet's `pts` is that time floored to the
/// declared base; its `duration` is the distance to the next packet's
/// `pts`. Timestamps are therefore continuous (`pts + duration` is the
/// next `pts`) and strictly increasing, and never drift: after a
/// 44.1 kHz → 32 kHz switch a 1152-sample frame lasts 1587 or 1588
/// ticks of the 1/44 100 base, alternating so the sum stays exact. The
/// decoder announces the new output shape with `Frame::StreamParams`;
/// the stream's declared time base never changes.
///
/// **Duration and seeking.** `open()` walks every frame header once,
/// so `streams()[0].duration` is the exact sum of the emitted packets'
/// durations whatever the stream switches to (a Xing / Info frame
/// describes only the file it was written for), and keeps a sparse
/// index of frame positions so `seek_to` lands on the frame that
/// contains the requested time — at or before it, as the `Demuxer`
/// contract asks — with the PTS that frame is then emitted with.
pub struct Mp3Demuxer {
    input: Box<dyn ReadSeek>,
    streams: Vec<StreamInfo>,
    tags: Mp3Tags,
    /// Optional Xing/Info frame parsed from the first audio frame
    /// (or `None` for plain CBR streams).
    xing: Option<XingTag>,
    /// Optional LAME-extension tag parsed from the carrier frame's
    /// LAME magic (when present and all four Xing flag bits are set).
    /// Drives **gapless playback** — the `encoder_delay` and
    /// `zero_padding` fields tell the decoder how many priming samples
    /// to trim off the front and how many zero-pad samples to trim
    /// off the back.
    lame: Option<LameTag>,
    /// Byte offset of the first MPEG audio frame (after ID3v2,
    /// before the first frame's syncword).
    first_frame_offset: u64,
    /// Byte offset of the first **playable** frame — equal to
    /// `first_frame_offset` for streams without a Xing/Info info
    /// frame, or to `first_frame_offset + info_frame_len` otherwise.
    first_audio_frame_offset: u64,
    /// Total file length.
    total_len: u64,
    /// End-of-audio offset = `total_len - 128` when an ID3v1 trailer
    /// was detected, else `total_len`.
    audio_end_offset: u64,
    /// Current read cursor inside the audio region.
    cursor: u64,
    /// Whether `cursor` sits exactly where the previous frame ended.
    on_boundary: bool,
    /// Framer state in force at `cursor`.
    state: FramerState,
    /// Sampling frequency of the first audio frame: the declared time
    /// base is `1 / sample_rate`.
    sample_rate: u32,
    /// Running time of the next packet, in `framing::TICKS_PER_SECOND`
    /// units.
    next_ticks: u64,
    /// Exact total running time of the audio frames, in ticks.
    total_ticks: u64,
    /// Unpadded free-format frame length measured on the first frame
    /// (§2.4.1.3), `None` for a table-bitrate first frame.
    free_format_base_len: Option<usize>,
    /// Number of audio frames (info frames excluded).
    audio_frames: u64,
    /// Number of framing-key changes (sampling frequency / MPEG
    /// version / layer / free-format switches) between audio frames.
    framing_switches: u64,
    /// Sparse seek index: every [`SEEK_INDEX_STRIDE`]-th audio frame.
    seek_index: Vec<SeekPoint>,
    /// Whether the EOF path of [`Self::next_packet`] was entered, so
    /// repeated calls keep returning `Eof`.
    finished: bool,
    /// Trimmed PCM sample count after applying the LAME encoder-delay
    /// and zero-padding fields. Equal to `streams[0].duration` for
    /// streams without a LAME tag; smaller when gapless trim applies.
    trimmed_duration_samples: Option<i64>,
}

/// One entry of the demuxer's seek index: where an audio frame starts,
/// the running time at its start, and the framer state to resume with.
#[derive(Debug, Clone, Copy)]
struct SeekPoint {
    offset: u64,
    ticks: u64,
    state: FramerState,
}

/// Audio frames between two seek-index entries.
const SEEK_INDEX_STRIDE: u64 = 32;

/// How many junk bytes may precede the first frame after the ID3v2
/// tag (a tag whose size is off by its padding, say).
const MAX_LEADING_JUNK: u64 = 4096;

impl std::fmt::Debug for Mp3Demuxer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mp3Demuxer")
            .field("streams", &self.streams)
            .field("tags", &self.tags)
            .field("xing", &self.xing)
            .field("lame", &self.lame)
            .field("first_frame_offset", &self.first_frame_offset)
            .field("first_audio_frame_offset", &self.first_audio_frame_offset)
            .field("total_len", &self.total_len)
            .field("audio_end_offset", &self.audio_end_offset)
            .field("cursor", &self.cursor)
            .field("sample_rate", &self.sample_rate)
            .field("next_ticks", &self.next_ticks)
            .field("total_ticks", &self.total_ticks)
            .field("audio_frames", &self.audio_frames)
            .field("framing_switches", &self.framing_switches)
            .field("free_format_base_len", &self.free_format_base_len)
            .field("finished", &self.finished)
            .field("trimmed_duration_samples", &self.trimmed_duration_samples)
            .finish()
    }
}

impl Mp3Demuxer {
    /// Open an MP3 stream from a generic [`ReadSeek`] source.
    pub fn open(input: Box<dyn ReadSeek>) -> Result<Self> {
        let mut input: Box<dyn ReadSeek> = Box::new(CachedInput::new(input));
        // 1. Bound the input length so we can spot ID3v1 trailers.
        let total_len = input.seek(SeekFrom::End(0))?;
        input.seek(SeekFrom::Start(0))?;

        // 2. ID3v2 frontmatter — skip past the tag if present.
        let mut head10 = [0u8; 10];
        let n = read_up_to(input.as_mut(), &mut head10)?;
        let (id3v2_present, id3v2_size, cursor) = if n == 10 {
            if let Some(total) = id3v2_total_len(&head10) {
                if total > total_len {
                    return Err(Error::invalid("ID3v2 size overruns the file"));
                }
                (true, total, total)
            } else {
                (false, 0u64, 0u64)
            }
        } else {
            (false, 0u64, 0u64)
        };

        // 3. ID3v1 trailer.
        let id3v1_present = detect_id3v1(&mut input, total_len)?;
        let audio_end_offset = if id3v1_present {
            total_len - 128
        } else {
            total_len
        };

        let tags = Mp3Tags {
            id3v2_present,
            id3v2_size,
            id3v1_present,
        };

        if cursor >= audio_end_offset {
            return Err(Error::invalid(
                "no MPEG audio data between ID3 frontmatter and trailer",
            ));
        }

        // 4. First frame. A few stray bytes between an ID3v2 tag
        //    (whose size is sometimes off by its padding in the wild)
        //    and the first frame are tolerated; a free-format first
        //    frame has its length measured from the next sync
        //    (§2.4.1.3).
        let mut state = FramerState::default();
        let first = framing::next_frame(
            input.as_mut(),
            cursor,
            audio_end_offset,
            &mut state,
            true,
            Some(MAX_LEADING_JUNK),
        )?
        .ok_or_else(|| {
            Error::invalid("no MPEG audio frame sync within the scan window after ID3v2")
        })?;
        let first_header = first.header;
        let first_frame_offset = first.offset;
        let first_len = first.len;
        let free_format_base_len = state.free_base;

        let mut first_frame_buf = vec![0u8; first_len];
        if framing::read_at(input.as_mut(), first_frame_offset, &mut first_frame_buf)? < first_len {
            return Err(Error::invalid("truncated first MPEG audio frame"));
        }

        let channels = first_header.channel_count();
        let side_bytes = side_info_len(first_header.version, channels);
        let xing = parse_xing_info(&first_frame_buf, side_bytes);

        // 5b. LAME-extension tag (gapless playback). Only attempted
        //     when all four Xing flag bits are set, since that is the
        //     only layout `docs/audio/mp3/lame-xing-info-tag.md`
        //     documents. The encoder version, encoder-delay /
        //     zero-padding pair, and the rest of the LAME-defined
        //     fields are unpacked from the canonical
        //     `header_bytes + side_info_bytes + 118` byte offset.
        let lame = xing.as_ref().and_then(|xt| {
            let crc_bytes = if first_header.crc_protected { 2 } else { 0 };
            let header_bytes = 4 + crc_bytes;
            let off = lame_magic_offset(header_bytes, side_bytes, xt)?;
            parse_lame_tag(&first_frame_buf, off).ok()
        });

        // 6. Walk every frame header once: exact frame count and
        //    running time across any switch, and the seek index.
        //    A Xing/Info frame carries no PCM — its slot is a metadata
        //    carrier — so it is skipped wherever it sits.
        // `before` is the framer state to resume *at* the frame being
        // processed; `after` the state once it is accepted.
        let mut before = FramerState::default();
        let mut after = state;
        let mut first_audio: Option<(u64, FramerState, Mp3FrameHeader)> = None;
        let mut seek_index = Vec::new();
        let mut ticks: u64 = 0;
        let mut audio_frames: u64 = 0;
        let mut framing_switches: u64 = 0;
        let mut last_key: Option<FramingKey> = None;
        let mut cur = Some(first);
        while let Some(f) = cur {
            if !f.info_frame {
                if first_audio.is_none() {
                    first_audio = Some((f.offset, before, f.header));
                }
                if audio_frames.is_multiple_of(SEEK_INDEX_STRIDE) {
                    seek_index.push(SeekPoint {
                        offset: f.offset,
                        ticks,
                        state: before,
                    });
                }
                let key = FramingKey::of(&f.header);
                if last_key.is_some_and(|k| k != key) {
                    framing_switches += 1;
                }
                last_key = Some(key);
                ticks += framing::frame_ticks(&f.header);
                audio_frames += 1;
            }
            let mut next_state = after;
            cur = framing::next_frame(
                input.as_mut(),
                f.end(),
                audio_end_offset,
                &mut next_state,
                true,
                None,
            )?;
            before = after;
            after = next_state;
        }

        let (first_audio_frame_offset, cursor_state, audio_header) = match first_audio {
            Some(v) => v,
            None => (first.end(), state, first_header),
        };

        // 7. Build the StreamInfo from the first audio frame. The
        //    codec parameters carry the same MP3 WAVEFORMATEX tag
        //    muxers consume to round-trip into AVI/WAV containers.
        let sample_rate = audio_header.sample_rate_hz;
        let samples_per_frame = first_header.samples_per_frame();
        // Free-format streams have no table bitrate, but a fixed one:
        // the unpadded body carries `samples_per_frame` samples, so
        // the effective constant bitrate is
        // `base_len * 8 * sample_rate / samples_per_frame` bit/s.
        // Table-bitrate streams take their declared rate directly.
        let bitrate_bps = match first_header.bitrate_kbps {
            Some(k) => Some(k * 1000),
            None => free_format_base_len.map(|base| {
                ((base as u128) * 8 * first_header.sample_rate_hz as u128
                    / samples_per_frame as u128) as u32
            }),
        };
        let time_base = TimeBase::new(1, sample_rate as i64);
        let duration_samples = framing::ticks_to_units(ticks, sample_rate);

        // Gapless trim: when a LAME tag with non-zero encoder_delay or
        // zero_padding is present, the playable PCM sample count is
        // the gross duration minus the two trim values.
        let trimmed_duration_samples = match lame.as_ref() {
            Some(tag) if tag.has_gapless_trim() => {
                let trim = (tag.encoder_delay as i64) + (tag.zero_padding as i64);
                Some((duration_samples - trim).max(0))
            }
            _ => Some(duration_samples),
        };

        let mut params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
        params.sample_rate = Some(sample_rate);
        params.channels = Some(audio_header.channel_count() as u16);
        params.bit_rate = bitrate_bps.map(|b| b as u64);
        params.tag = Some(CodecTag::wave_format(WAVE_FORMAT_MP3));

        let stream = StreamInfo {
            index: 0,
            time_base,
            duration: Some(duration_samples),
            start_time: Some(0),
            params,
        };

        Ok(Self {
            input,
            streams: vec![stream],
            tags,
            xing,
            lame,
            first_frame_offset,
            first_audio_frame_offset,
            total_len,
            audio_end_offset,
            cursor: first_audio_frame_offset,
            on_boundary: true,
            state: cursor_state,
            sample_rate,
            next_ticks: 0,
            total_ticks: ticks,
            free_format_base_len,
            audio_frames,
            framing_switches,
            seek_index,
            finished: false,
            trimmed_duration_samples,
        })
    }

    /// Tags observed in the on-disk frontmatter / trailer.
    #[must_use]
    pub fn tags(&self) -> &Mp3Tags {
        &self.tags
    }

    /// The Xing/Info VBR-info frame, if one was found.
    #[must_use]
    pub fn xing(&self) -> Option<&XingTag> {
        self.xing.as_ref()
    }

    /// The LAME-extension tag parsed from the carrier frame, if one
    /// was found.
    ///
    /// Carries the gapless-playback `encoder_delay` and `zero_padding`
    /// pair (see [`crate::lame_tag::LameTag`]). Returns `None` when
    /// the stream has no Xing/Info frame, when fewer than all four
    /// Xing flag bits were set (the LAME-magic offset is undocumented
    /// in the staged spec for those cases), or when the carrier
    /// frame's encoder string was something other than `"LAME"` (e.g.
    /// `"Lavc"`, `"Lavf"`).
    #[must_use]
    pub fn lame(&self) -> Option<&LameTag> {
        self.lame.as_ref()
    }

    /// Encoder-delay PCM-sample count from the LAME extension, or
    /// `None` when no LAME tag was found. This is the number of zero
    /// samples the encoder added at the start of the stream to flush
    /// the analysis filter bank; a gapless-aware decoder trims this
    /// many samples (plus any decoder-intrinsic priming) off the
    /// front of the decoded PCM.
    #[must_use]
    pub fn encoder_delay_samples(&self) -> Option<u32> {
        self.lame.as_ref().map(|t| t.encoder_delay as u32)
    }

    /// Zero-padding PCM-sample count from the LAME extension, or
    /// `None` when no LAME tag was found. This is the number of zero
    /// samples the encoder appended to the last frame to fill the
    /// granular boundary; a gapless-aware decoder drops this many
    /// samples off the back of the decoded PCM.
    #[must_use]
    pub fn zero_padding_samples(&self) -> Option<u32> {
        self.lame.as_ref().map(|t| t.zero_padding as u32)
    }

    /// Playable-PCM sample count after applying the LAME encoder-delay
    /// and zero-padding trim, or the gross duration when no LAME tag
    /// was found.
    ///
    /// Distinct from `streams()[0].duration` (which carries the
    /// **gross** sample count including the encoder priming and the
    /// trailing padding) — pipelines that need the on-disk-original
    /// PCM length should consult this method.
    #[must_use]
    pub fn trimmed_duration_samples(&self) -> Option<i64> {
        self.trimmed_duration_samples
    }

    /// Byte offset of the *first* MPEG audio frame in the file —
    /// either the Xing/Info info frame (when present) or the first
    /// playable audio frame (when not).
    #[must_use]
    pub fn first_frame_offset(&self) -> u64 {
        self.first_frame_offset
    }

    /// Byte offset of the first **playable** audio frame.
    #[must_use]
    pub fn first_audio_frame_offset(&self) -> u64 {
        self.first_audio_frame_offset
    }

    /// One-past-the-end byte offset of the audio region (file end
    /// minus the ID3v1 trailer, if any).
    #[must_use]
    pub fn audio_end_offset(&self) -> u64 {
        self.audio_end_offset
    }

    /// True when the open() probe identified the stream as VBR via
    /// a "Xing" magic (as opposed to "Info" or no magic at all).
    #[must_use]
    pub fn is_vbr(&self) -> bool {
        matches!(self.xing.as_ref().map(|x| x.id), Some(XingTagId::Xing))
    }

    /// Number of audio frames (packets) the stream holds, counted by
    /// the open-time header walk. Xing / Info carrier frames are not
    /// counted.
    #[must_use]
    pub fn audio_frame_count(&self) -> u64 {
        self.audio_frames
    }

    /// Number of points where consecutive audio frames change their
    /// framing — sampling frequency, MPEG version, layer, or free
    /// format versus table bitrate. Zero for a stream that never
    /// switches. Channel-mode and bitrate changes do not count.
    #[must_use]
    pub fn framing_switch_count(&self) -> u64 {
        self.framing_switches
    }
}

/// Extract a Layer III frame's `main_data_begin` reservoir back-pointer
/// (§2.4.1.7 / §2.4.2.7) from the start of its side-information region.
///
/// The back-pointer is the first field of the side info: a 9-bit value
/// for MPEG-1, an 8-bit value for MPEG-2 / 2.5 (LSF). It sits
/// immediately after the 4-byte header and the optional 2-byte CRC
/// slot. We only need this one field to decide whether the frame is an
/// independently-decodable random-access point, so we read it directly
/// rather than parsing the whole side info.
///
/// Returns `None` for non-Layer-III frames (Layers I/II have no
/// reservoir — every frame is self-contained) or when the buffer is too
/// short to hold the field.
fn frame_main_data_begin(header: &Mp3FrameHeader, frame: &[u8]) -> Option<u16> {
    if header.layer != Layer::LayerIII {
        return None;
    }
    let crc_bytes = if header.crc_protected { 2 } else { 0 };
    let si_start = 4 + crc_bytes;
    if header.version.is_lsf() {
        // 8-bit field: one whole byte at `si_start`.
        frame.get(si_start).map(|&b| u16::from(b))
    } else {
        // 9-bit field: byte `si_start` (high 8) + top bit of the next.
        let hi = *frame.get(si_start)?;
        let lo = *frame.get(si_start + 1)?;
        Some((u16::from(hi) << 1) | u16::from(lo >> 7))
    }
}

impl Demuxer for Mp3Demuxer {
    fn format_name(&self) -> &str {
        FORMAT_NAME
    }

    fn streams(&self) -> &[StreamInfo] {
        &self.streams
    }

    fn next_packet(&mut self) -> Result<Packet> {
        if self.finished {
            return Err(Error::Eof);
        }
        loop {
            let Some(f) = self.advance()? else {
                self.finished = true;
                return Err(Error::Eof);
            };
            if f.info_frame {
                continue;
            }
            let mut data = vec![0u8; f.len];
            if framing::read_at(self.input.as_mut(), f.offset, &mut data)? < f.len {
                self.finished = true;
                return Err(Error::Eof);
            }
            let pts = framing::ticks_to_units(self.next_ticks, self.sample_rate);
            self.next_ticks += framing::frame_ticks(&f.header);
            let duration = framing::ticks_to_units(self.next_ticks, self.sample_rate) - pts;
            // A Layer III frame is a true random-access point (keyframe)
            // only when its `main_data_begin` back-pointer is zero — i.e.
            // its main data is wholly contained in its own slot and does
            // not borrow from the bit reservoir of earlier frames
            // (§2.4.2.7). A frame with `main_data_begin > 0` cannot be
            // decoded in isolation, so a seeker must not treat it as a
            // safe landing point. Layers I/II carry no reservoir, so
            // every frame is self-contained.
            let keyframe = frame_main_data_begin(&f.header, &data)
                .map(|m| m == 0)
                .unwrap_or(true);
            let pkt = Packet::new(0, self.streams[0].time_base, data)
                .with_pts(pts)
                .with_dts(pts)
                .with_duration(duration)
                .with_keyframe(keyframe);
            return Ok(pkt);
        }
    }

    fn seek_to(&mut self, _stream_index: u32, pts: i64) -> Result<i64> {
        if pts < 0 {
            return Err(Error::invalid("negative seek PTS"));
        }
        let rate = self.sample_rate;
        // Compare in the declared time base: a frame is "at or before"
        // the request when the PTS it is emitted with is.
        let total = framing::ticks_to_units(self.total_ticks, rate);
        let target = pts.min(total);
        // Last index entry at or before the target (entry 0 is the
        // first audio frame at time 0).
        let idx = self
            .seek_index
            .partition_point(|p| framing::ticks_to_units(p.ticks, rate) <= target);
        let Some(point) = idx.checked_sub(1).map(|i| self.seek_index[i]) else {
            // No audio frame at all.
            self.finished = true;
            return Ok(0);
        };
        self.cursor = point.offset;
        self.state = point.state;
        self.on_boundary = true;
        self.next_ticks = point.ticks;
        self.finished = false;
        // Walk forward to the frame containing the target: the last
        // frame that starts at or before it.
        loop {
            let (cursor, state, on_boundary) = (self.cursor, self.state, self.on_boundary);
            let Some(f) = self.advance()? else {
                break;
            };
            if f.info_frame {
                continue;
            }
            let ft = framing::frame_ticks(&f.header);
            if framing::ticks_to_units(self.next_ticks + ft, rate) > target {
                // Land on this frame: rewind to just before it.
                self.cursor = cursor;
                self.state = state;
                self.on_boundary = on_boundary;
                break;
            }
            self.next_ticks += ft;
        }
        Ok(framing::ticks_to_units(self.next_ticks, self.sample_rate))
    }

    fn duration_micros(&self) -> Option<i64> {
        // Container-level duration is the PLAYABLE duration: when the
        // Xing/Info trailer carries the gapless extension
        // (encoder-delay / zero-padding trim), report the trimmed
        // sample count -- that is the figure a gapless-aware player
        // renders (r408). For streams without the extension
        // `trimmed_duration_samples` equals the gross per-stream
        // duration, so nothing changes there. The gross count stays
        // available as `streams()[0].duration`.
        let dur = self.trimmed_duration_samples.or(self.streams[0].duration)?;
        // duration ticks are in 1/sample_rate seconds.
        Some(((dur as i128) * 1_000_000 / self.sample_rate as i128) as i64)
    }
}

impl Mp3Demuxer {
    /// Locate the next frame from the cursor and step past it.
    fn advance(&mut self) -> Result<Option<Located>> {
        let f = framing::next_frame(
            self.input.as_mut(),
            self.cursor,
            self.audio_end_offset,
            &mut self.state,
            self.on_boundary,
            None,
        )?;
        if let Some(f) = &f {
            self.cursor = f.end();
            self.on_boundary = true;
        }
        Ok(f)
    }
}

/// Open an MP3 demuxer over a file path. Convenience helper that
/// mirrors the direct-API convention used elsewhere in the workspace.
pub fn open_file_demuxer<P: AsRef<std::path::Path>>(path: P) -> Result<Mp3Demuxer> {
    let f = File::open(path)?;
    Mp3Demuxer::open(Box::new(f))
}

#[cfg(test)]
#[allow(clippy::identity_op)] // bit-literal header constructions stay parallel to the spec layout
mod tests {
    use super::*;
    use std::io::Cursor;

    /// A 1-byte MP3 buffer fails the open() length check.
    #[test]
    fn rejects_truncated_input() {
        let buf: Vec<u8> = vec![0xFF];
        let err = Mp3Demuxer::open(Box::new(Cursor::new(buf))).err();
        assert!(err.is_some(), "1-byte input should not open");
    }

    /// r405 fuzz regression: a hostile free-format stream can place a
    /// second matching sync 1..3 bytes after the first, "measuring" a
    /// frame shorter than its own header + side info. `open()` must
    /// reject it (previously the header copy into the first-frame
    /// buffer sliced out of range and panicked).
    #[test]
    fn rejects_free_format_frame_shorter_than_header() {
        // The minimized fuzz artifact's stream body: a valid MPEG-2.5
        // Layer III free-format sync at offset 0 whose next matching
        // sync lands 3 bytes later.
        let buf: Vec<u8> = vec![0xFF, 0xFF, 0x01, 0xFF, 0xFF, 0x00, 0xFF];
        let err = Mp3Demuxer::open(Box::new(Cursor::new(buf))).err();
        assert!(
            err.is_some(),
            "sub-header-size free-format measurement must be rejected"
        );
    }

    /// `synchsafe_size` reads the four 7-bit groups in MSB-first
    /// order (per ID3v2.3 §3.1 — `size = (b[0]<<21) | (b[1]<<14) |
    /// (b[2]<<7) | b[3]`).
    #[test]
    fn synchsafe_decode_matches_spec_examples() {
        assert_eq!(synchsafe_size([0, 0, 0, 0]), 0);
        assert_eq!(synchsafe_size([0, 0, 0, 0x7F]), 0x7F);
        assert_eq!(synchsafe_size([0, 0, 1, 0]), 0x80);
        // (0x4B) — id3v2 fixture body length.
        assert_eq!(synchsafe_size([0, 0, 0, 0x4B]), 75);
        // ID3v2.3 max representable size = 0x0FFFFFFF.
        assert_eq!(synchsafe_size([0x7F, 0x7F, 0x7F, 0x7F]), 0x0FFFFFFF);
    }

    #[test]
    fn id3v2_total_len_includes_header_and_footer() {
        // No footer: total = 10 + size.
        let h = [b'I', b'D', b'3', 3, 0, 0x00, 0, 0, 0, 0x4B];
        assert_eq!(id3v2_total_len(&h), Some(10 + 75));
        // Footer flag (0x10): total += 10.
        let h = [b'I', b'D', b'3', 4, 0, 0x10, 0, 0, 0, 0x4B];
        assert_eq!(id3v2_total_len(&h), Some(20 + 75));
        // Not an ID3 tag.
        assert_eq!(
            id3v2_total_len(&[b'X', b'Y', b'Z', 0, 0, 0, 0, 0, 0, 0]),
            None
        );
    }

    /// Build a synthetic CBR stream by repeating one MPEG-1 L3
    /// 128k/44.1k frame N times. The resulting cursor walk should
    /// emit N packets with PTS [0, spf, 2*spf, …] and an EOF.
    #[test]
    fn cbr_emits_n_packets_with_monotonic_pts() {
        // Build a header for 128 kbps / 44.1 kHz / stereo / Layer III
        // / MPEG-1 / no padding / no CRC.
        let raw: u32 = (0xFFF << 20)
            | (1 << 19)        // ID = 1 (MPEG-1)
            | (0b01 << 17)     // Layer III
            | (1 << 16)        // protection = 1 (no CRC)
            | (0b1001 << 12)   // bitrate_index = 9 -> 128 kbps
            | (0b00 << 10)     // sampling = 44.1 kHz
            | (0 << 9)         // padding = 0
            | (0 << 8)         // private = 0
            | (0b00 << 6)      // mode = stereo
            | (0b00 << 4)      // mode_ext = 0
            | (0 << 3)         // copyright = 0
            | (1 << 2)         // original = 1
            | 0b00; // emphasis = none
        let hdr = raw.to_be_bytes();
        let frame_len = 144 * 128_000 / 44_100; // = 417
        let mut frame = vec![0u8; frame_len];
        frame[..4].copy_from_slice(&hdr);
        let n_frames = 8usize;
        let mut buf = Vec::with_capacity(n_frames * frame_len);
        for _ in 0..n_frames {
            buf.extend_from_slice(&frame);
        }
        let mut d = Mp3Demuxer::open(Box::new(Cursor::new(buf))).expect("open synthetic CBR");
        assert_eq!(d.streams().len(), 1);
        let info = &d.streams()[0];
        assert_eq!(info.params.sample_rate, Some(44_100));
        assert_eq!(info.params.channels, Some(2));
        assert_eq!(info.params.bit_rate, Some(128_000));
        let mut got = 0;
        let mut expected_pts = 0i64;
        loop {
            match d.next_packet() {
                Ok(p) => {
                    assert_eq!(p.pts, Some(expected_pts));
                    assert_eq!(p.duration, Some(1152));
                    assert!(p.flags.keyframe);
                    expected_pts += 1152;
                    got += 1;
                }
                Err(Error::Eof) => break,
                Err(e) => panic!("unexpected error: {e:?}"),
            }
        }
        assert_eq!(got, n_frames);
        // Repeat next_packet — should stay at EOF.
        assert!(matches!(d.next_packet(), Err(Error::Eof)));
    }

    /// Build a free-format (`bitrate_index == 0`) MPEG-1 L3 header
    /// with the given padding bit. Free format carries no table
    /// bitrate; the demuxer measures the frame length from the
    /// distance to the next sync (§2.4.1.3).
    fn free_format_l3_header(padding: bool) -> [u8; 4] {
        let raw: u32 = (0xFFF << 20)
            | (1 << 19)        // ID = 1 (MPEG-1)
            | (0b01 << 17)     // Layer III
            | (1 << 16)        // protection = 1 (no CRC)
            | (0b0000 << 12)   // bitrate_index = 0 -> free format
            | (0b00 << 10)     // sampling = 44.1 kHz
            | (u32::from(padding) << 9)
            | (0 << 8)         // private = 0
            | (0b00 << 6)      // mode = stereo
            | (0b00 << 4)      // mode_ext = 0
            | (0 << 3)         // copyright = 0
            | (1 << 2)         // original = 1
            | 0b00; // emphasis = none
        raw.to_be_bytes()
    }

    /// A free-format stream of N identical unpadded frames: the
    /// demuxer measures the constant frame length from the first two
    /// syncs and then walks all N frames.
    #[test]
    fn free_format_unpadded_emits_n_packets() {
        let hdr = free_format_l3_header(false);
        // Pick an arbitrary fixed body length (not on the bitrate
        // ladder for 44.1 kHz). 521 bytes ≈ a ~160 kbit/s frame.
        let frame_len = 521usize;
        let mut frame = vec![0u8; frame_len];
        frame[..4].copy_from_slice(&hdr);
        let n_frames = 6usize;
        let mut buf = Vec::with_capacity(n_frames * frame_len);
        for _ in 0..n_frames {
            buf.extend_from_slice(&frame);
        }
        let mut d = Mp3Demuxer::open(Box::new(Cursor::new(buf))).expect("open free-format stream");
        assert_eq!(d.streams().len(), 1);
        let info = &d.streams()[0];
        assert_eq!(info.params.sample_rate, Some(44_100));
        // Effective bitrate derived from the measured body:
        // 521 * 8 * 44100 / 1152 ≈ 159_600 bit/s.
        let expected_br = (521u64 * 8 * 44_100) / 1152;
        assert_eq!(info.params.bit_rate, Some(expected_br));

        let mut got = 0;
        let mut expected_pts = 0i64;
        loop {
            match d.next_packet() {
                Ok(p) => {
                    assert_eq!(p.pts, Some(expected_pts));
                    assert_eq!(p.data.len(), frame_len);
                    assert_eq!(p.duration, Some(1152));
                    expected_pts += 1152;
                    got += 1;
                }
                Err(Error::Eof) => break,
                Err(e) => panic!("unexpected error: {e:?}"),
            }
        }
        assert_eq!(got, n_frames);
    }

    /// A free-format stream whose first frame is **padded** but whose
    /// later frames are not: the measured unpadded base must subtract
    /// the first frame's own padding slot, and each later frame's
    /// length follows its own padding bit.
    #[test]
    fn free_format_mixed_padding_lengths() {
        let base = 521usize; // unpadded body
        // Frame 0: padded (+1 byte slot). Frame 1: unpadded.
        // Frame 2: padded. Frame 3: unpadded.
        let pads = [true, false, true, false];
        let mut buf = Vec::new();
        let mut offsets = Vec::new();
        for &p in &pads {
            offsets.push(buf.len());
            let len = base + usize::from(p); // Layer III slot = 1 byte
            let mut frame = vec![0u8; len];
            frame[..4].copy_from_slice(&free_format_l3_header(p));
            buf.extend_from_slice(&frame);
        }
        let mut d = Mp3Demuxer::open(Box::new(Cursor::new(buf))).expect("open mixed-padding free");
        // The measured base must be the *unpadded* 521 (first frame
        // is padded, so the raw inter-sync distance is 522).
        assert_eq!(d.free_format_base_len, Some(base));

        let mut lens = Vec::new();
        loop {
            match d.next_packet() {
                Ok(p) => lens.push(p.data.len()),
                Err(Error::Eof) => break,
                Err(e) => panic!("unexpected error: {e:?}"),
            }
        }
        assert_eq!(lens, vec![base + 1, base, base + 1, base]);
    }

    /// A single free-format frame cannot be length-measured (there is
    /// no second sync to bound it), so open() rejects it.
    #[test]
    fn free_format_single_frame_is_rejected() {
        let hdr = free_format_l3_header(false);
        let mut frame = vec![0u8; 521];
        frame[..4].copy_from_slice(&hdr);
        let err = Mp3Demuxer::open(Box::new(Cursor::new(frame))).err();
        assert!(err.is_some(), "single free-format frame must not open");
    }

    /// Wrap the synthetic CBR stream in an ID3v2 header + ID3v1
    /// trailer and confirm the demuxer skips both. Tag offsets and
    /// frame counts checked against the on-disk fixture trace
    /// `docs/audio/mp3/fixtures/layer3-with-id3v2-tag/trace.txt`
    /// (`ID3V2_HEADER ... size=75 total=85`).
    #[test]
    fn skips_id3v2_header_and_id3v1_trailer() {
        let raw: u32 =
            (0xFFF << 20) | (1 << 19) | (0b01 << 17) | (1 << 16) | (0b1001 << 12) | (0b00 << 10);
        let hdr = raw.to_be_bytes();
        let frame_len = 417usize;
        let mut audio = Vec::new();
        for _ in 0..3 {
            audio.extend_from_slice(&hdr);
            audio.extend(std::iter::repeat_n(0u8, frame_len - 4));
        }
        // ID3v2.3 header: size = 75 synchsafe-encoded as [0, 0, 0, 0x4B].
        let id3_size = 75usize;
        let mut id3 = Vec::with_capacity(10 + id3_size);
        id3.extend_from_slice(&[b'I', b'D', b'3', 3, 0, 0x00, 0, 0, 0, 0x4B]);
        id3.extend(std::iter::repeat_n(0u8, id3_size));
        // ID3v1 trailer: "TAG" + 125 zero padding bytes (matches the
        // 128-byte total documented in datavoyage §"MPEG Audio Tag
        // ID3v1").
        let mut id3v1 = vec![b'T', b'A', b'G'];
        id3v1.extend(std::iter::repeat_n(0u8, 125));

        let mut buf = Vec::new();
        buf.extend_from_slice(&id3);
        buf.extend_from_slice(&audio);
        buf.extend_from_slice(&id3v1);

        let mut d = Mp3Demuxer::open(Box::new(Cursor::new(buf))).unwrap();
        assert!(d.tags().id3v2_present);
        assert_eq!(d.tags().id3v2_size, 85);
        assert!(d.tags().id3v1_present);
        // First frame should be located at offset 85 (immediately
        // after the ID3v2 tag), and there should be exactly 3
        // packets emitted with no trailer bleed-through.
        assert_eq!(d.first_audio_frame_offset(), 85);
        let mut got = 0;
        while let Ok(_pkt) = d.next_packet() {
            got += 1;
        }
        assert_eq!(got, 3);
    }

    /// Parse a synthetic Xing info frame and confirm flag-gated
    /// fields land in the expected struct positions.
    #[test]
    fn parses_xing_frame_with_all_four_flags() {
        // Build the MPEG-1 stereo Layer III header so the side-info
        // length is 32 bytes per ISO/IEC 11172-3 §2.4.1.7.
        let raw: u32 =
            (0xFFF << 20) | (1 << 19) | (0b01 << 17) | (1 << 16) | (0b1001 << 12) | (0b00 << 10);
        let hdr = raw.to_be_bytes();
        let frame_len = 417usize;
        let mut frame = vec![0u8; frame_len];
        frame[..4].copy_from_slice(&hdr);
        // side info: 32 zero bytes from offset 4..36.
        // Xing magic at offset 36.
        frame[36..40].copy_from_slice(b"Xing");
        // flags = 0x0F.
        frame[40..44].copy_from_slice(&0x0000000Fu32.to_be_bytes());
        // frames = 100.
        frame[44..48].copy_from_slice(&100u32.to_be_bytes());
        // bytes = 50000.
        frame[48..52].copy_from_slice(&50_000u32.to_be_bytes());
        // toc = identity ramp.
        for i in 0..100 {
            frame[52 + i] = (i * 255 / 99) as u8;
        }
        // quality = 75.
        frame[152..156].copy_from_slice(&75u32.to_be_bytes());

        let xt = parse_xing_info(&frame, 32).expect("Xing parse");
        assert_eq!(xt.id, XingTagId::Xing);
        assert_eq!(xt.flags, 0x0F);
        assert_eq!(xt.frames, Some(100));
        assert_eq!(xt.bytes, Some(50_000));
        assert!(xt.toc.is_some());
        assert_eq!(xt.quality, Some(75));
        // Identity ramp end values: toc[0] = 0, toc[99] = 255.
        let toc = xt.toc.unwrap();
        assert_eq!(toc[0], 0);
        assert_eq!(toc[99], 255);
    }

    /// "Info" magic == CBR-shaped Xing tag.
    #[test]
    fn parses_info_frame_distinct_from_xing() {
        let raw: u32 =
            (0xFFF << 20) | (1 << 19) | (0b01 << 17) | (1 << 16) | (0b1001 << 12) | (0b00 << 10);
        let hdr = raw.to_be_bytes();
        let mut frame = vec![0u8; 417];
        frame[..4].copy_from_slice(&hdr);
        frame[36..40].copy_from_slice(b"Info");
        frame[40..44].copy_from_slice(&0x00000001u32.to_be_bytes()); // frames only
        frame[44..48].copy_from_slice(&7u32.to_be_bytes());
        let xt = parse_xing_info(&frame, 32).unwrap();
        assert_eq!(xt.id, XingTagId::Info);
        assert_eq!(xt.frames, Some(7));
        assert!(xt.bytes.is_none());
        assert!(xt.toc.is_none());
        assert!(xt.quality.is_none());
    }

    /// Mono MPEG-1 streams have a 17-byte side-info; the Xing magic
    /// must still be located correctly.
    #[test]
    fn xing_offset_respects_mono_side_info_length() {
        // mono header: same as before but mode = 0b11 (single ch).
        let raw: u32 = (0xFFF << 20)
            | (1 << 19)
            | (0b01 << 17)
            | (1 << 16)
            | (0b1001 << 12)
            | (0b00 << 10)
            | (0b11 << 6);
        let hdr = raw.to_be_bytes();
        let mut frame = vec![0u8; 417];
        frame[..4].copy_from_slice(&hdr);
        // mono side-info length is 17 bytes per crate::side_info::SIDE_INFO_BYTES_MONO.
        let offset = 4 + SIDE_INFO_BYTES_MONO;
        frame[offset..offset + 4].copy_from_slice(b"Xing");
        frame[offset + 4..offset + 8].copy_from_slice(&0u32.to_be_bytes());
        let xt = parse_xing_info(&frame, SIDE_INFO_BYTES_MONO).unwrap();
        assert_eq!(xt.id, XingTagId::Xing);
        assert_eq!(xt.flags, 0);
        assert!(xt.frames.is_none());
    }

    /// No magic at the expected offset → no Xing tag (None).
    #[test]
    fn parses_returns_none_on_missing_magic() {
        let raw: u32 =
            (0xFFF << 20) | (1 << 19) | (0b01 << 17) | (1 << 16) | (0b1001 << 12) | (0b00 << 10);
        let hdr = raw.to_be_bytes();
        let mut frame = vec![0u8; 417];
        frame[..4].copy_from_slice(&hdr);
        // No "Xing"/"Info" at offset 36 — pure CBR frame.
        assert!(parse_xing_info(&frame, 32).is_none());
    }

    #[test]
    fn side_info_len_matrix() {
        assert_eq!(side_info_len(MpegVersion::Mpeg1, 1), SIDE_INFO_BYTES_MONO);
        assert_eq!(side_info_len(MpegVersion::Mpeg1, 2), SIDE_INFO_BYTES_STEREO);
        assert_eq!(
            side_info_len(MpegVersion::Mpeg2, 1),
            SIDE_INFO_BYTES_LSF_MONO
        );
        assert_eq!(
            side_info_len(MpegVersion::Mpeg2, 2),
            SIDE_INFO_BYTES_LSF_STEREO
        );
    }

    /// Probe scores: ID3v2 + frame sync = 100; bare frame sync with
    /// no extension hint = 75; bare frame sync with .mp3 = 100;
    /// arbitrary noise = 0.
    #[test]
    fn probe_scoring() {
        // Build a valid Layer III header.
        let raw: u32 =
            (0xFFF << 20) | (1 << 19) | (0b01 << 17) | (1 << 16) | (0b1001 << 12) | (0b00 << 10);
        let hdr = raw.to_be_bytes();
        // Bare frame sync — no ext hint.
        let p1 = ProbeData {
            buf: &hdr,
            ext: None,
        };
        assert_eq!(probe(&p1), 75);
        // With .mp3 hint.
        let p2 = ProbeData {
            buf: &hdr,
            ext: Some("mp3"),
        };
        assert_eq!(probe(&p2), 100);
        // ID3v2 + frame sync.
        let mut id3 = vec![b'I', b'D', b'3', 3, 0, 0x00, 0, 0, 0, 0x00];
        id3.extend_from_slice(&hdr);
        let p3 = ProbeData {
            buf: &id3,
            ext: None,
        };
        assert_eq!(probe(&p3), 100);
        // Garbage.
        let p4 = ProbeData {
            buf: &[0; 10],
            ext: None,
        };
        assert_eq!(probe(&p4), 0);
    }

    #[test]
    fn duration_micros_for_cbr() {
        // 8 frames of 128 kbps / 44.1 kHz Layer III stereo.
        let raw: u32 =
            (0xFFF << 20) | (1 << 19) | (0b01 << 17) | (1 << 16) | (0b1001 << 12) | (0b00 << 10);
        let hdr = raw.to_be_bytes();
        let frame_len = 417usize;
        let mut buf = Vec::new();
        for _ in 0..8 {
            buf.extend_from_slice(&hdr);
            buf.extend(std::iter::repeat_n(0u8, frame_len - 4));
        }
        let d = Mp3Demuxer::open(Box::new(Cursor::new(buf))).unwrap();
        // The open-time header walk counts the frames exactly:
        // 8 * 1152 samples at 44.1 kHz = 9216 / 44100 s = 208_979 µs
        // (a bytes / bitrate estimate would say 208_500, off by the
        // 417-vs-417.96 byte rounding of the unpadded frame).
        assert_eq!(d.streams()[0].duration, Some(8 * 1152));
        assert_eq!(d.audio_frame_count(), 8);
        let micros = d.duration_micros().unwrap();
        assert_eq!(micros, 208_979);
    }

    #[test]
    fn seek_to_returns_pts_for_cbr() {
        let raw: u32 =
            (0xFFF << 20) | (1 << 19) | (0b01 << 17) | (1 << 16) | (0b1001 << 12) | (0b00 << 10);
        let hdr = raw.to_be_bytes();
        let frame_len = 417usize;
        let mut buf = Vec::new();
        for _ in 0..16 {
            buf.extend_from_slice(&hdr);
            buf.extend(std::iter::repeat_n(0u8, frame_len - 4));
        }
        let mut d = Mp3Demuxer::open(Box::new(Cursor::new(buf))).unwrap();
        // Seek to PTS 5760 (== 5 frames * 1152 samples).
        let actual = d.seek_to(0, 5_760).unwrap();
        // CBR seek lands within one frame of the request.
        assert!((actual - 5_760).abs() <= 1152);
        // The returned PTS is now the *exact* PTS of the frame the
        // cursor landed on (a whole multiple of samples_per_frame),
        // and the next packet carries that same PTS.
        assert_eq!(actual % 1152, 0);
        let pkt = d.next_packet().unwrap();
        assert_eq!(pkt.pts, Some(actual));
    }

    /// After a CBR seek the reported PTS equals the landed frame's PTS,
    /// the next packet carries it, and the stream stays strictly
    /// monotone from there — i.e. the proportional estimate no longer
    /// leaks into the emitted timestamps.
    /// `frame_main_data_begin` extracts the same 9-bit (MPEG-1) /
    /// 8-bit (LSF) reservoir back-pointer as the full side-info parser,
    /// across both bit widths and the CRC-slot offset.
    #[test]
    fn frame_main_data_begin_matches_side_info_parser() {
        use crate::frame::ChannelMode;
        use crate::side_info::parse_side_info;
        // MPEG-1 mono Layer III, 128 kbps, 44.1 kHz, no CRC. The header
        // is 4 bytes; side info begins at byte 4.
        let raw: u32 =
            (0xFFF << 20) | (1 << 19) | (0b01 << 17) | (1 << 16) | (0b1001 << 12) | (0b11 << 6);
        let hdr = raw.to_be_bytes();
        let h = parse_header(&hdr).unwrap();
        assert_eq!(h.mode, ChannelMode::SingleChannel);
        // Try several 9-bit main_data_begin values; encode them MSB-first
        // into the first 2 side-info bytes, zero the rest of a mono
        // side-info region, and confirm both extractors agree.
        for mdb in [0u16, 1, 7, 255, 256, 300, 511] {
            let mut frame = vec![0u8; 4 + SIDE_INFO_BYTES_MONO + 8];
            frame[..4].copy_from_slice(&hdr);
            // 9 bits MSB-first at byte 4: high 8 -> frame[4], low 1 ->
            // top bit of frame[5].
            frame[4] = (mdb >> 1) as u8;
            frame[5] = ((mdb & 1) << 7) as u8;
            let via_helper = frame_main_data_begin(&h, &frame).unwrap();
            let si = parse_side_info(&h, &frame[4..]).unwrap();
            assert_eq!(via_helper, mdb, "helper mismatch for {mdb}");
            assert_eq!(si.main_data_begin, mdb, "parser mismatch for {mdb}");
        }
    }

    /// A frame carrying a non-zero `main_data_begin` is *not* a keyframe
    /// (it borrows from the bit reservoir and cannot be decoded in
    /// isolation); a zero back-pointer frame is.
    #[test]
    fn keyframe_flag_tracks_main_data_begin() {
        let raw: u32 =
            (0xFFF << 20) | (1 << 19) | (0b01 << 17) | (1 << 16) | (0b1001 << 12) | (0b11 << 6);
        let hdr = raw.to_be_bytes();
        let h = parse_header(&hdr).unwrap();
        let frame_len = h.frame_len().unwrap();
        let make = |mdb: u16| {
            let mut buf = vec![0u8; frame_len * 3];
            for f in 0..3 {
                let off = f * frame_len;
                buf[off..off + 4].copy_from_slice(&hdr);
                buf[off + 4] = (mdb >> 1) as u8;
                buf[off + 5] = ((mdb & 1) << 7) as u8;
            }
            buf
        };

        // mdb == 0 -> keyframe.
        let mut d0 = Mp3Demuxer::open(Box::new(Cursor::new(make(0)))).unwrap();
        assert!(d0.next_packet().unwrap().is_keyframe());

        // mdb == 9 -> not a keyframe.
        let mut d1 = Mp3Demuxer::open(Box::new(Cursor::new(make(9)))).unwrap();
        assert!(!d1.next_packet().unwrap().is_keyframe());
    }

    #[test]
    fn seek_lands_on_frame_aligned_pts_and_stays_monotone() {
        let raw: u32 =
            (0xFFF << 20) | (1 << 19) | (0b01 << 17) | (1 << 16) | (0b1001 << 12) | (0b00 << 10);
        let hdr = raw.to_be_bytes();
        let frame_len = 417usize;
        let mut buf = Vec::new();
        for _ in 0..32 {
            buf.extend_from_slice(&hdr);
            buf.extend(std::iter::repeat_n(0u8, frame_len - 4));
        }
        let mut d = Mp3Demuxer::open(Box::new(Cursor::new(buf))).unwrap();
        // Request a PTS that does NOT fall on a frame boundary: 8564
        // lies inside frame 7 (8064 ..= 9215). The seek lands on the
        // frame containing it — at or before the request, as the
        // `Demuxer::seek_to` contract asks.
        let actual = d.seek_to(0, 8_564).unwrap();
        assert_eq!(actual, 7 * 1152);
        // The next four packets carry strictly increasing, frame-exact
        // PTS values starting at the landed one.
        let mut expect = actual;
        for _ in 0..4 {
            let pkt = d.next_packet().unwrap();
            assert_eq!(pkt.pts, Some(expect));
            expect += 1152;
        }
    }

    /// `lame_magic_offset` reflects the staged-doc all-flags layout
    /// for the four `(version, channel_count)` carrier-frame cases.
    #[test]
    fn lame_magic_offset_matches_staged_doc_table() {
        let mut xt = XingTag {
            id: XingTagId::Xing,
            flags: 0x0F,
            frames: Some(0),
            bytes: Some(0),
            toc: Some([0u8; 100]),
            quality: Some(0),
        };
        // MPEG-1 stereo: 4 header + 32 side-info + 118 = $9A absolute
        // = 154 — matches the staged doc's worked example.
        assert_eq!(
            lame_magic_offset(4, SIDE_INFO_BYTES_STEREO, &xt),
            Some(4 + 32 + 118)
        );
        // MPEG-1 mono: 4 header + 17 + 118.
        assert_eq!(
            lame_magic_offset(4, SIDE_INFO_BYTES_MONO, &xt),
            Some(4 + 17 + 118)
        );
        // MPEG-2/2.5 stereo: 4 + 17 + 118.
        assert_eq!(
            lame_magic_offset(4, SIDE_INFO_BYTES_LSF_STEREO, &xt),
            Some(4 + 17 + 118)
        );
        // MPEG-2/2.5 mono: 4 + 9 + 118.
        assert_eq!(
            lame_magic_offset(4, SIDE_INFO_BYTES_LSF_MONO, &xt),
            Some(4 + 9 + 118)
        );
        // Fewer than all four flags → docs-gap, return None.
        xt.flags = 0x07; // missing VBR_SCALE
        assert_eq!(lame_magic_offset(4, SIDE_INFO_BYTES_STEREO, &xt), None);
        xt.flags = 0x0E; // missing FRAMES
        assert_eq!(lame_magic_offset(4, SIDE_INFO_BYTES_STEREO, &xt), None);
    }

    /// Build a synthetic carrier-frame buffer ("Xing" + all four
    /// flags + LAME extension) inside an MPEG-1 stereo 128 kbps
    /// 44.1 kHz frame, return the buffer + the parsed `Mp3Demuxer`.
    /// The carrier is followed by `n_audio` zero-filled audio frames
    /// so `Mp3Demuxer::open()` exits cleanly after the metadata
    /// frame. Used by the gapless-trim tests below.
    fn build_lame_carrier_stream(delay: u16, padding: u16, n_audio: usize) -> Vec<u8> {
        // MPEG-1 stereo Layer III, 128 kbps, 44.1 kHz, no CRC.
        let raw: u32 =
            (0xFFF << 20) | (1 << 19) | (0b01 << 17) | (1 << 16) | (0b1001 << 12) | (0b00 << 10);
        let hdr = raw.to_be_bytes();
        let frame_len = 417usize;

        // Carrier frame: Xing magic at 4+32=36; flags 0x0F at +40;
        // frames=n_audio at +44; bytes=n_audio*frame_len at +48; toc at
        // +52..152; quality=0 at +152..156. The LAME magic falls at
        // +154 (4 + 32 + 118) and overlaps the last 2 bytes of the
        // quality field — i.e. the high 2 bytes of "LAME" overwrite
        // the last 2 zero bytes of quality. Total LAME extension run
        // is 38 bytes ending at +154+38 = +192, comfortably inside
        // the 417-byte frame slot.
        let mut carrier = vec![0u8; frame_len];
        carrier[..4].copy_from_slice(&hdr);
        carrier[36..40].copy_from_slice(b"Xing");
        carrier[40..44].copy_from_slice(&0x0000000Fu32.to_be_bytes());
        carrier[44..48].copy_from_slice(&(n_audio as u32).to_be_bytes());
        carrier[48..52].copy_from_slice(&((n_audio * frame_len) as u32).to_be_bytes());
        for i in 0..100 {
            carrier[52 + i] = ((i * 255) / 99) as u8;
        }
        carrier[152..156].copy_from_slice(&0u32.to_be_bytes()); // quality = 0
        // LAME extension @ +154 (overlaps last 2 zero bytes of quality).
        let lame_off = 4 + SIDE_INFO_BYTES_STEREO + 118;
        assert_eq!(lame_off, 154);
        carrier[lame_off..lame_off + 9].copy_from_slice(b"LAME3.100");
        // Skip the 2 unmoved bytes ($A3-$A4) — already zero.
        // Revision/method at +11.
        carrier[lame_off + 11] = 0x10;
        // Lowpass at +12.
        carrier[lame_off + 12] = 196;
        // Peak amplitude (f32 1.0 = 0x3F80_0000) at +13..+17.
        carrier[lame_off + 13..lame_off + 17].copy_from_slice(&0x3F80_0000u32.to_be_bytes());
        // Radio + Audiophile RG.
        carrier[lame_off + 17..lame_off + 19].copy_from_slice(&0u16.to_be_bytes());
        carrier[lame_off + 19..lame_off + 21].copy_from_slice(&0u16.to_be_bytes());
        // Flags/ATH + bitrate.
        carrier[lame_off + 21] = 0;
        carrier[lame_off + 22] = 128;
        // Delay+padding 12+12-bit pack at +23..+26.
        let b0 = (delay >> 4) as u8;
        let b1 = (((delay & 0x0F) << 4) | ((padding >> 8) & 0x0F)) as u8;
        let b2 = (padding & 0xFF) as u8;
        carrier[lame_off + 23] = b0;
        carrier[lame_off + 24] = b1;
        carrier[lame_off + 25] = b2;
        // Misc/mp3-gain/preset/music-length/music-CRC/tag-CRC all zero.

        // n_audio plain frames (zero-filled bodies).
        let mut audio_frame = vec![0u8; frame_len];
        audio_frame[..4].copy_from_slice(&hdr);
        let mut buf = carrier;
        for _ in 0..n_audio {
            buf.extend_from_slice(&audio_frame);
        }
        buf
    }

    #[test]
    fn lame_tag_parsed_via_open_with_full_xing_flags() {
        // Build a stream with the LAME tag carrying delay=1729,
        // padding=722 — the §5 worked-example values from the staged
        // doc. Confirm the demuxer surfaces them through `.lame()` /
        // `.encoder_delay_samples()` / `.zero_padding_samples()`.
        let buf = build_lame_carrier_stream(1729, 722, 4);
        let d = Mp3Demuxer::open(Box::new(Cursor::new(buf))).expect("open");
        let lame = d.lame().expect("LAME tag present");
        assert_eq!(lame.encoder_delay, 1729);
        assert_eq!(lame.zero_padding, 722);
        assert_eq!(&lame.encoder_version, b"LAME3.100");
        assert_eq!(d.encoder_delay_samples(), Some(1729));
        assert_eq!(d.zero_padding_samples(), Some(722));
    }

    #[test]
    fn trimmed_duration_subtracts_gapless_field() {
        // 4 audio frames × 1152 samples = 4608 gross.
        // delay=1729 + padding=722 = 2451 trim.
        // trimmed = 4608 - 2451 = 2157.
        let buf = build_lame_carrier_stream(1729, 722, 4);
        let d = Mp3Demuxer::open(Box::new(Cursor::new(buf))).unwrap();
        // Gross duration is still in streams[0].duration (the LAME tag
        // does not change the on-wire MPEG frame count).
        assert_eq!(d.streams()[0].duration, Some(4 * 1152));
        // Trimmed duration honours the LAME-extension trim.
        assert_eq!(d.trimmed_duration_samples(), Some(4 * 1152 - 1729 - 722));
        // The container-level duration reports the PLAYABLE (trimmed)
        // figure when the gapless extension is present (r408) -- the
        // duration a gapless-aware player renders.
        let expected_micros = (i128::from(4i64 * 1152 - 1729 - 722) * 1_000_000 / 44_100) as i64;
        assert_eq!(d.duration_micros(), Some(expected_micros));
    }

    #[test]
    fn trimmed_duration_equals_gross_without_lame_tag() {
        // Build the same synthetic CBR stream as
        // `cbr_emits_n_packets_with_monotonic_pts` (no Xing or LAME).
        let raw: u32 =
            (0xFFF << 20) | (1 << 19) | (0b01 << 17) | (1 << 16) | (0b1001 << 12) | (0b00 << 10);
        let hdr = raw.to_be_bytes();
        let frame_len = 417usize;
        let mut buf = Vec::new();
        for _ in 0..6 {
            buf.extend_from_slice(&hdr);
            buf.extend(std::iter::repeat_n(0u8, frame_len - 4));
        }
        let d = Mp3Demuxer::open(Box::new(Cursor::new(buf))).unwrap();
        // No Xing tag → no LAME tag.
        assert!(d.xing().is_none());
        assert!(d.lame().is_none());
        // Trimmed duration falls back to gross CBR duration.
        let gross = d.streams()[0].duration;
        assert_eq!(d.trimmed_duration_samples(), gross);
    }

    #[test]
    fn trimmed_duration_equals_gross_for_zero_delay_padding() {
        // LAME tag present but delay=0, padding=0 → no trim applies.
        let buf = build_lame_carrier_stream(0, 0, 4);
        let d = Mp3Demuxer::open(Box::new(Cursor::new(buf))).unwrap();
        let lame = d.lame().expect("LAME tag present");
        assert!(!lame.has_gapless_trim());
        let gross = d.streams()[0].duration;
        assert_eq!(d.trimmed_duration_samples(), gross);
    }

    /// Doc worked-example byte pattern propagates byte-for-byte
    /// through the demuxer. The §5 example fixes the exact 3 bytes
    /// [0x6C, 0x12, 0xD2] in the delay+padding slot and asserts
    /// delay=1729, padding=722.
    #[test]
    fn doc_worked_example_propagates_through_demuxer() {
        let raw: u32 =
            (0xFFF << 20) | (1 << 19) | (0b01 << 17) | (1 << 16) | (0b1001 << 12) | (0b00 << 10);
        let hdr = raw.to_be_bytes();
        let frame_len = 417usize;
        let mut carrier = vec![0u8; frame_len];
        carrier[..4].copy_from_slice(&hdr);
        carrier[36..40].copy_from_slice(b"Xing");
        carrier[40..44].copy_from_slice(&0x0000000Fu32.to_be_bytes());
        carrier[44..48].copy_from_slice(&1u32.to_be_bytes());
        carrier[48..52].copy_from_slice(&(frame_len as u32).to_be_bytes());
        // TOC + zero quality kept zero; LAME magic at +154.
        let lame_off = 154usize;
        carrier[lame_off..lame_off + 4].copy_from_slice(b"LAME");
        // §5 byte pattern at delay+padding offset +23.
        carrier[lame_off + 23] = 0b0110_1100;
        carrier[lame_off + 24] = 0b0001_0010;
        carrier[lame_off + 25] = 0b1101_0010;
        // Append a single audio frame so the demuxer terminates.
        let mut audio_frame = vec![0u8; frame_len];
        audio_frame[..4].copy_from_slice(&hdr);
        let mut buf = carrier;
        buf.extend_from_slice(&audio_frame);
        let d = Mp3Demuxer::open(Box::new(Cursor::new(buf))).unwrap();
        let lame = d.lame().expect("LAME tag present");
        assert_eq!(lame.encoder_delay, 1729);
        assert_eq!(lame.zero_padding, 722);
    }

    /// A carrier frame whose encoder string is something other than
    /// "LAME" (e.g. "Lavc" — a common third-party encoder tag) yields a
    /// Xing tag but no LAME tag.
    #[test]
    fn non_lame_encoder_yields_no_lame_tag() {
        let raw: u32 =
            (0xFFF << 20) | (1 << 19) | (0b01 << 17) | (1 << 16) | (0b1001 << 12) | (0b00 << 10);
        let hdr = raw.to_be_bytes();
        let frame_len = 417usize;
        let mut carrier = vec![0u8; frame_len];
        carrier[..4].copy_from_slice(&hdr);
        carrier[36..40].copy_from_slice(b"Xing");
        carrier[40..44].copy_from_slice(&0x0000000Fu32.to_be_bytes());
        carrier[44..48].copy_from_slice(&1u32.to_be_bytes());
        carrier[48..52].copy_from_slice(&(frame_len as u32).to_be_bytes());
        let lame_off = 154usize;
        // Write "Lavc" instead of "LAME" — same offset.
        carrier[lame_off..lame_off + 4].copy_from_slice(b"Lavc");
        let mut audio_frame = vec![0u8; frame_len];
        audio_frame[..4].copy_from_slice(&hdr);
        let mut buf = carrier;
        buf.extend_from_slice(&audio_frame);
        let d = Mp3Demuxer::open(Box::new(Cursor::new(buf))).unwrap();
        assert!(d.xing().is_some(), "Xing tag still present");
        assert!(d.lame().is_none(), "non-LAME encoder yields no LAME tag");
        // Trimmed duration must still match gross (no LAME tag = no trim).
        assert_eq!(d.trimmed_duration_samples(), d.streams()[0].duration);
    }
}
