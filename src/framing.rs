//! Frame-boundary discovery shared by the demuxer's open-time walk,
//! its packet loop and its seek path.
//!
//! An MPEG audio elementary stream is a plain sequence of
//! self-delimiting frames (ISO/IEC 11172-3 §2.4.1.2 / §2.4.2.3): each
//! one starts on the 12-bit syncword (11 bits for the MPEG-2.5
//! extension, `docs/audio/mp3/MPEG-2.5-GAP.md`) and its length follows
//! from its own header — the bitrate table, the sampling frequency and
//! the padding slot — or, in free format (`bitrate_index == '0000'`),
//! from the fixed but unsignalled bitrate measured as the distance
//! between two syncs. Nothing in the bitstream forbids the header
//! fields from changing from one frame to the next: a byte-concatenation
//! of two files (`cat a.mp3 b.mp3`, or a broadcast that switches
//! programmes) changes the sampling frequency, the MPEG version
//! (MPEG-1 / MPEG-2 LSF / MPEG-2.5), the channel mode, the bitrate or
//! the free-format slot count at a frame boundary, and may carry the
//! second file's ID3v2 tag, ID3v1 trailer and Xing / Info frame in
//! between.
//!
//! # Acceptance policy
//!
//! The syncword is only 11–12 bits, so the bit pattern also turns up
//! inside main data. The framer therefore distinguishes two positions:
//!
//! * **On a boundary** — exactly where the previous accepted frame (or
//!   a tag skipped after it) ended. A header there that keeps the
//!   stream's *framing key* (MPEG version, layer, sampling frequency,
//!   free-format-ness) is accepted on its own: VBR bitrate changes,
//!   padding, and channel-mode changes need no further proof. A header
//!   there that changes the key (a switch) is accepted when the
//!   position right after it is itself a frame header, an ID3 tag, or
//!   the end of the audio region.
//! * **Scanning** — after bytes had to be skipped (junk, a damaged
//!   frame, or a seek into the middle of a frame). A candidate is
//!   accepted only when the frame after it is a header with the same
//!   version / layer / sampling frequency (a two-frame chain), an ID3
//!   tag, or the end of the audio region. A false sync inside main
//!   data would have to forge a second, correctly-spaced header to
//!   pass.
//!
//! Tags met on a boundary are stepped over: an ID3v2 tag
//! (`docs/container/id3/id3v2.4.0-structure.html` §3.1, header + body
//! + optional footer), an ID3v1 tag ("TAG", 128 bytes,
//!   `docs/container/id3/id3v1.md`) and the enhanced "TAG+" block
//!   (227 bytes, `docs/container/id3/enhanced-tag.md`). A Layer III
//!   frame whose main-data area starts with the "Xing" / "Info" magic
//!   (`docs/audio/mp3/lame-xing-info-tag.md` §1) carries no audio and is
//!   reported as an info frame so the caller can drop it wherever it
//!   sits in the stream.
//!
//! # Time
//!
//! [`TICKS_PER_SECOND`] is the least common multiple of the nine
//! sampling frequencies MPEG audio can signal (32 / 44.1 / 48 kHz,
//! 16 / 22.05 / 24 kHz, 8 / 11.025 / 12 kHz), so every frame duration
//! (384, 576 or 1152 samples at any of them) is a whole number of
//! ticks and the running time of a stream that switches rates is
//! exact, with no rounding drift.

use std::io::{Read, Seek, SeekFrom};

use oxideav_core::{ReadSeek, Result};

use crate::frame::{Emphasis, Layer, Mp3FrameHeader, MpegVersion, parse_header};

/// `2^8 · 3^2 · 5^3 · 7^2` — the least common multiple of every MPEG
/// audio sampling frequency (ISO/IEC 11172-3 §2.4.2.3, ISO/IEC 13818-3
/// §2.4.2.3, the MPEG-2.5 table in `datavoyage-mpgscript-mpeghdr.html`).
pub(crate) const TICKS_PER_SECOND: u64 = 14_112_000;

/// Longest frame (padded) a free-format stream may use: ISO/IEC
/// 11172-3 §2.4.2.3 does not require a decoder to support free-format
/// bitrates above 448 / 384 / 320 kbit/s for Layers I / II / III, and
/// ISO/IEC 13818-3 §2.4.2.3 above 256 / 160 / 160 kbit/s for the lower
/// sampling frequencies (MPEG-2.5 inherits the latter). The measurement
/// never looks further than one such frame, so it cannot span other
/// frames to a far-away sync of the same key.
fn max_free_format_len(h: &Mp3FrameHeader) -> usize {
    let kbps: u64 = match (h.version.is_lsf(), h.layer) {
        (false, Layer::LayerI) => 448,
        (false, Layer::LayerII) => 384,
        (false, Layer::LayerIII) => 320,
        (true, Layer::LayerI) => 256,
        (true, Layer::LayerII | Layer::LayerIII) => 160,
    };
    let body = kbps * 1000 * u64::from(h.samples_per_frame()) / (8 * u64::from(h.sample_rate_hz));
    // Whole slots, plus the padding slot.
    let slot = h.slot_bytes() as u64;
    ((body / slot) * slot + slot) as usize
}

/// The header fields that fix how frames are delimited and timed. A
/// change of any of them is a stream switch; the bitrate (VBR), the
/// padding bit and the channel mode are not part of the key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FramingKey {
    pub version: MpegVersion,
    pub layer: Layer,
    pub sample_rate: u32,
    pub free_format: bool,
}

impl FramingKey {
    pub fn of(h: &Mp3FrameHeader) -> Self {
        Self {
            version: h.version,
            layer: h.layer,
            sample_rate: h.sample_rate_hz,
            free_format: h.is_free_format(),
        }
    }
}

/// Framer state carried from one accepted frame to the next.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct FramerState {
    /// Key of the last accepted frame (`None` before the first one).
    pub key: Option<FramingKey>,
    /// Unpadded free-format frame length in force, when `key` is a
    /// free-format key.
    pub free_base: Option<usize>,
}

/// One frame found by [`next_frame`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct Located {
    /// Byte offset of the syncword.
    pub offset: u64,
    /// Frame length in bytes (header through padding slot).
    pub len: usize,
    pub header: Mp3FrameHeader,
    /// The frame is a Xing / Info carrier and holds no audio.
    pub info_frame: bool,
}

impl Located {
    pub fn end(&self) -> u64 {
        self.offset + self.len as u64
    }
}

/// Duration of one frame in [`TICKS_PER_SECOND`] units.
pub(crate) fn frame_ticks(h: &Mp3FrameHeader) -> u64 {
    u64::from(h.samples_per_frame()) * (TICKS_PER_SECOND / u64::from(h.sample_rate_hz))
}

/// Convert a tick count to a timestamp in `1 / rate` units (floor).
pub(crate) fn ticks_to_units(ticks: u64, rate: u32) -> i64 {
    ((u128::from(ticks) * u128::from(rate)) / u128::from(TICKS_PER_SECOND)) as i64
}

/// Read up to `buf.len()` bytes at `offset`; returns the count read.
pub(crate) fn read_at(input: &mut dyn ReadSeek, offset: u64, buf: &mut [u8]) -> Result<usize> {
    input.seek(SeekFrom::Start(offset))?;
    let mut got = 0;
    while got < buf.len() {
        match input.read(&mut buf[got..]) {
            Ok(0) => break,
            Ok(n) => got += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e.into()),
        }
    }
    Ok(got)
}

/// Header + CRC + side-information byte count of a Layer III frame
/// (§2.4.1.7 / ISO/IEC 13818-3 §2.4.1.7); 4 (+2) for Layers I / II.
fn structural_min_len(h: &Mp3FrameHeader) -> usize {
    let crc = if h.crc_protected { 2 } else { 0 };
    let side = if h.layer == Layer::LayerIII {
        crate::demuxer::side_info_len(h.version, h.channel_count())
    } else {
        0
    };
    4 + crc + side
}

/// Total ID3v2 tag length (header + body + optional footer) when
/// `head` starts with a well-formed ID3v2 header: "ID3", a version
/// byte other than 0xFF, and a size whose four bytes all have bit 7
/// clear (synchsafe, §3.1 of the v2.4 structure document).
pub(crate) fn id3v2_len(head: &[u8]) -> Option<u64> {
    if head.len() < 10 || &head[..3] != b"ID3" {
        return None;
    }
    if head[3] == 0xFF || head[4] == 0xFF || head[6..10].iter().any(|b| b & 0x80 != 0) {
        return None;
    }
    let size = (u64::from(head[6]) << 21)
        | (u64::from(head[7]) << 14)
        | (u64::from(head[8]) << 7)
        | u64::from(head[9]);
    let footer = if head[5] & 0x10 != 0 { 10 } else { 0 };
    Some(10 + size + footer)
}

/// What sits at `pos`, judged from its first bytes, for the purpose of
/// confirming that the frame before it ended exactly there.
fn plausible_follower(input: &mut dyn ReadSeek, pos: u64, end: u64) -> Result<Follower> {
    if pos >= end {
        return Ok(Follower::End);
    }
    let mut head = [0u8; 10];
    let n = read_at(input, pos, &mut head)?;
    let head = &head[..n];
    if id3v2_len(head).is_some() || head.starts_with(b"TAG") {
        return Ok(Follower::Tag);
    }
    match parse_header(head) {
        Ok(h) => Ok(Follower::Header(h)),
        // Fewer than four bytes of trailing junk before the end of the
        // audio region.
        Err(_) if end - pos < 4 => Ok(Follower::End),
        Err(_) => Ok(Follower::Nothing),
    }
}

enum Follower {
    End,
    Tag,
    Header(Mp3FrameHeader),
    Nothing,
}

/// Measure the unpadded free-format length of the frame at `pos`:
/// the distance to the next free-format sync with the same version,
/// layer and sampling frequency, minus this frame's padding slot,
/// confirmed by a plausible follower after that second frame too.
fn measure_free_format(
    input: &mut dyn ReadSeek,
    pos: u64,
    h: &Mp3FrameHeader,
    end: u64,
) -> Result<Option<usize>> {
    let key = FramingKey::of(h);
    let pad = usize::from(h.padding) * h.slot_bytes();
    let min_len = structural_min_len(h).max(pad + 1);
    let start = pos + min_len as u64;
    if start + 4 > end {
        return Ok(None);
    }
    // The next sync starts at most one maximum-length frame on.
    let limit = end.min(pos + max_free_format_len(h) as u64 + 4);
    if limit <= start {
        return Ok(None);
    }
    let mut buf = vec![0u8; (limit - start) as usize];
    let n = read_at(input, start, &mut buf)?;
    buf.truncate(n);
    for i in 0..buf.len().saturating_sub(3) {
        if buf[i] != 0xFF || buf[i + 1] & 0xE0 != 0xE0 {
            continue;
        }
        let Ok(h2) = parse_header(&buf[i..i + 4]) else {
            continue;
        };
        if FramingKey::of(&h2) != key {
            continue;
        }
        let padded = min_len + i;
        let base = padded - pad;
        if base < structural_min_len(h) {
            continue;
        }
        let q = start + i as u64;
        let len2 = base + usize::from(h2.padding) * h2.slot_bytes();
        let r = q + len2 as u64;
        let ok = r >= end
            || match plausible_follower(input, r, end)? {
                Follower::End | Follower::Tag => true,
                Follower::Header(h3) => FramingKey::of(&h3) == key,
                Follower::Nothing => false,
            };
        if ok {
            return Ok(Some(base));
        }
    }
    Ok(None)
}

/// Frames that must follow a candidate found by scanning (each with
/// the candidate's version, layer and sampling frequency) before it is
/// accepted, unless the end of the audio region or a tag comes first.
const SCAN_CHAIN: usize = 3;

/// Walk up to `depth` frames from `pos` (where the candidate `h` ends)
/// and check each is a header with `h`'s framing key; reaching the end
/// of the audio region or a tag first also confirms.
fn chain_ok(
    input: &mut dyn ReadSeek,
    mut pos: u64,
    end: u64,
    h: &Mp3FrameHeader,
    free_base: Option<usize>,
    depth: usize,
) -> Result<bool> {
    let key = FramingKey::of(h);
    for _ in 0..depth {
        match plausible_follower(input, pos, end)? {
            Follower::End | Follower::Tag => return Ok(true),
            Follower::Nothing => return Ok(false),
            Follower::Header(h2) => {
                if FramingKey::of(&h2) != key {
                    return Ok(false);
                }
                let len = match h2.frame_len() {
                    Some(l) => l,
                    None => match free_base.and_then(|b| h2.frame_len_free_format(b)) {
                        Some(l) => l,
                        None => return Ok(false),
                    },
                };
                if len < structural_min_len(&h2) {
                    return Ok(false);
                }
                pos += len as u64;
                if pos > end {
                    // The follower is the truncated final frame.
                    return Ok(true);
                }
            }
        }
    }
    Ok(true)
}

/// `true` when the Layer III frame at `pos` carries the "Xing" /
/// "Info" magic right after its side information.
fn is_info_frame(
    input: &mut dyn ReadSeek,
    pos: u64,
    h: &Mp3FrameHeader,
    len: usize,
) -> Result<bool> {
    if h.layer != Layer::LayerIII {
        return Ok(false);
    }
    let off = structural_min_len(h);
    if off + 4 > len {
        return Ok(false);
    }
    let mut magic = [0u8; 4];
    if read_at(input, pos + off as u64, &mut magic)? < 4 {
        return Ok(false);
    }
    Ok(&magic == b"Xing" || &magic == b"Info")
}

/// Find the next frame at or after `pos` (before `end`), updating
/// `state` on acceptance.
///
/// `on_boundary` says whether `pos` is exactly where the previous
/// frame ended (see the module doc). `max_skip` bounds how many bytes
/// may be skipped while scanning (`None` = up to `end`). Returns
/// `Ok(None)` at the end of the audio region, including when a frame
/// on a boundary is cut short by it.
pub(crate) fn next_frame(
    input: &mut dyn ReadSeek,
    mut pos: u64,
    end: u64,
    state: &mut FramerState,
    mut on_boundary: bool,
    max_skip: Option<u64>,
) -> Result<Option<Located>> {
    let scan_limit = max_skip.map(|m| pos.saturating_add(m));
    loop {
        if pos + 4 > end {
            return Ok(None);
        }
        if let Some(lim) = scan_limit
            && pos > lim
        {
            return Ok(None);
        }
        let mut head = [0u8; 10];
        let n = read_at(input, pos, &mut head)?;
        let head = &head[..n];

        // Tags between frames: only trusted on a boundary, where the
        // previous frame's length pins the position.
        if on_boundary {
            if let Some(total) = id3v2_len(head)
                && pos + total <= end
            {
                pos += total;
                continue;
            }
            let tag_len = if head.starts_with(b"TAG+") {
                Some(227)
            } else if head.starts_with(b"TAG") {
                Some(128)
            } else {
                None
            };
            if let Some(t) = tag_len
                && pos + t <= end
                && !matches!(plausible_follower(input, pos + t, end)?, Follower::Nothing)
            {
                pos += t;
                continue;
            }
        }

        let header = match parse_header(head) {
            // The reserved emphasis code is never written by a
            // conforming encoder (§2.4.2.3); while scanning it is one
            // more way to reject a false sync.
            Ok(h) if on_boundary || h.emphasis != Emphasis::Reserved => h,
            _ => {
                pos += 1;
                on_boundary = false;
                continue;
            }
        };
        let key = FramingKey::of(&header);
        let same_key = state.key == Some(key);

        // Frame length.
        let mut free_base = None;
        let len = match header.frame_len() {
            Some(l) => Some(l),
            None => {
                // Free format: reuse the base in force on a boundary
                // that keeps the key, unless the following position
                // disagrees (a free-format segment with a different
                // fixed bitrate); otherwise measure afresh.
                let mut base = None;
                if on_boundary
                    && same_key
                    && let Some(b) = state.free_base
                    && let Some(l) = header.frame_len_free_format(b)
                    && chain_ok(input, pos + l as u64, end, &header, Some(b), 1)?
                {
                    base = Some(b);
                }
                if base.is_none() {
                    base = measure_free_format(input, pos, &header, end)?;
                }
                if base.is_none() && on_boundary && same_key {
                    // Last frame(s) of a free-format stream followed by
                    // junk: keep the base in force.
                    base = state.free_base;
                }
                free_base = base;
                base.and_then(|b| header.frame_len_free_format(b))
            }
        };
        let Some(len) = len.filter(|&l| l >= structural_min_len(&header)) else {
            pos += 1;
            on_boundary = false;
            continue;
        };
        let next = pos + len as u64;
        if next > end {
            if on_boundary && same_key {
                // Truncated final frame.
                return Ok(None);
            }
            pos += 1;
            on_boundary = false;
            continue;
        }

        // A boundary that keeps the key is trusted; a switch on a
        // boundary needs the next frame to agree with the new key; a
        // candidate found by scanning needs a longer chain.
        let confirmed = if on_boundary && same_key {
            true
        } else {
            let depth = if on_boundary { 1 } else { SCAN_CHAIN };
            chain_ok(input, next, end, &header, free_base, depth)?
        };
        if !confirmed {
            pos += 1;
            on_boundary = false;
            continue;
        }

        state.key = Some(key);
        state.free_base = if key.free_format { free_base } else { None };
        let info_frame = is_info_frame(input, pos, &header, len)?;
        return Ok(Some(Located {
            offset: pos,
            len,
            header,
            info_frame,
        }));
    }
}

/// A read-ahead cache in front of the demuxer's input, so that the
/// open-time header walk and the per-frame boundary checks (a handful
/// of small reads per frame) do not each turn into a seek + read on
/// the underlying source.
pub(crate) struct CachedInput {
    inner: Box<dyn ReadSeek>,
    buf: Vec<u8>,
    /// File offset of `buf[0]`.
    buf_start: u64,
    /// Logical read position.
    pos: u64,
}

const CACHE_WINDOW: usize = 64 * 1024;

impl CachedInput {
    pub fn new(inner: Box<dyn ReadSeek>) -> Self {
        Self {
            inner,
            buf: Vec::new(),
            buf_start: 0,
            pos: 0,
        }
    }

    fn refill(&mut self) -> std::io::Result<()> {
        self.inner.seek(SeekFrom::Start(self.pos))?;
        self.buf.resize(CACHE_WINDOW, 0);
        let mut got = 0;
        while got < CACHE_WINDOW {
            match self.inner.read(&mut self.buf[got..]) {
                Ok(0) => break,
                Ok(n) => got += n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            }
        }
        self.buf.truncate(got);
        self.buf_start = self.pos;
        Ok(())
    }
}

impl Read for CachedInput {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        if out.is_empty() {
            return Ok(0);
        }
        let in_window = |s: &Self| s.pos >= s.buf_start && s.pos < s.buf_start + s.buf.len() as u64;
        if !in_window(self) {
            if out.len() >= CACHE_WINDOW {
                // Large read: bypass the cache.
                self.inner.seek(SeekFrom::Start(self.pos))?;
                let n = self.inner.read(out)?;
                self.pos += n as u64;
                return Ok(n);
            }
            self.refill()?;
            if !in_window(self) {
                return Ok(0);
            }
        }
        let at = (self.pos - self.buf_start) as usize;
        let n = out.len().min(self.buf.len() - at);
        out[..n].copy_from_slice(&self.buf[at..at + n]);
        self.pos += n as u64;
        Ok(n)
    }
}

impl Seek for CachedInput {
    fn seek(&mut self, to: SeekFrom) -> std::io::Result<u64> {
        self.pos = match to {
            SeekFrom::Start(p) => p,
            SeekFrom::Current(d) => self
                .pos
                .checked_add_signed(d)
                .ok_or_else(|| std::io::Error::other("seek before start"))?,
            SeekFrom::End(d) => {
                let end = self.inner.seek(SeekFrom::End(0))?;
                end.checked_add_signed(d)
                    .ok_or_else(|| std::io::Error::other("seek before start"))?
            }
        };
        Ok(self.pos)
    }
}
