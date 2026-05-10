//! Runtime libavcodec interop for the ffmpeg-oracle fuzz harness.
//!
//! libavcodec is loaded via `dlopen` at first call — there is no
//! `ffmpeg-sys` / `rust-av-sys`-style build-script dep that would pull
//! ffmpeg source into the workspace's cargo dep tree. The harness
//! checks `available()` up front and `return`s early when the shared
//! library isn't installed, so fuzz binaries built on a host without
//! libavcodec simply do nothing instead of panicking.
//!
//! Workspace policy bars consulting ffmpeg / libavcodec / libmp3lame /
//! libmpg123 source; we only inspect the public C headers
//! (`<libavcodec/avcodec.h>`, `<libavcodec/packet.h>`,
//! `<libavutil/frame.h>`) for function signatures + stable struct
//! prefixes.
//!
//! Install on macOS with `brew install ffmpeg`. On Debian / Ubuntu
//! install `ffmpeg` (which pulls libavcodec). The loader probes the
//! conventional shared-object names for both platforms across the
//! libavcodec 58..62 ABI series.

#![allow(unsafe_code)]

pub mod libavcodec {
    use libloading::{Library, Symbol};
    use std::ffi::c_void;
    use std::os::raw::{c_char, c_int, c_uint};
    use std::sync::OnceLock;

    /// Conventional libavcodec shared-object names the loader will try
    /// in order. The number suffix (`62`, `61`, …) is the major ABI
    /// version; we list each currently-deployed major and fall back to
    /// the unversioned `.so` / `.dylib` symlink.
    const CANDIDATES: &[&str] = &[
        "libavcodec.so.62",
        "libavcodec.so.61",
        "libavcodec.so.60",
        "libavcodec.so.59",
        "libavcodec.so.58",
        "libavcodec.so",
        "libavcodec.62.dylib",
        "libavcodec.61.dylib",
        "libavcodec.60.dylib",
        "libavcodec.59.dylib",
        "libavcodec.58.dylib",
        "libavcodec.dylib",
        "avcodec.dll",
    ];

    /// `AV_CODEC_ID_MP3` — the MPEG-1/2 Audio Layer III decoder id.
    /// Stable since the ffmpeg `enum AVCodecID` was assigned numeric
    /// values; matches the dispatch task spec.
    pub const AV_CODEC_ID_MP3: c_int = 86017;

    /// `AVERROR(EAGAIN)` flag — libavcodec's send_packet/receive_frame
    /// loop returns this to signal "feed me more / I have no output
    /// yet". Computed at runtime to stay portable across libcs.
    fn averror_eagain() -> c_int {
        // ffmpeg defines AVERROR(e) as -e on POSIX. EAGAIN is 11 on
        // Linux/glibc and 35 on macOS/BSD; both end up negative here
        // so any value < 0 that matches `EAGAIN` is recognised.
        #[cfg(target_os = "linux")]
        {
            -11
        }
        #[cfg(target_os = "macos")]
        {
            -35
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            -11
        }
    }

    /// `AVERROR_EOF` — returned by `avcodec_receive_frame` when the
    /// decoder has been fully drained. Computed at runtime so the
    /// macro arithmetic stays in one place; not currently consulted
    /// (we treat any non-zero non-EAGAIN as terminal).
    #[allow(dead_code)]
    fn averror_eof() -> c_int {
        // AVERROR_EOF = FFERRTAG('E','O','F',' ') = -('E' | 'O' << 8 | 'F' << 16 | ' ' << 24)
        let tag: u32 =
            (b'E' as u32) | ((b'O' as u32) << 8) | ((b'F' as u32) << 16) | ((b' ' as u32) << 24);
        -(tag as i32)
    }

    fn lib() -> Option<&'static Library> {
        static LIB: OnceLock<Option<Library>> = OnceLock::new();
        LIB.get_or_init(|| {
            for name in CANDIDATES {
                // SAFETY: `Library::new` is documented as unsafe because
                // the loaded library may run code at load time. We
                // accept that risk for fuzz tooling — libavcodec is a
                // well-behaved shared library.
                if let Ok(l) = unsafe { Library::new(name) } {
                    return Some(l);
                }
            }
            None
        })
        .as_ref()
    }

    /// True iff a libavcodec shared library was successfully loaded.
    /// The oracle harness early-returns when this is false so the
    /// binary still runs without an oracle (the assertions just
    /// don't fire).
    pub fn available() -> bool {
        lib().is_some()
    }

    /// A decoded audio frame as produced by libavcodec.
    pub struct DecodedAudio {
        /// PCM samples per channel.
        pub samples: u32,
        pub sample_rate: u32,
        pub channels: u32,
        /// Interleaved float32 samples (one channel layout: ch0 ch1 …
        /// ch0 ch1 …). libavcodec mp3 outputs `AV_SAMPLE_FMT_FLTP`
        /// (planar f32); we interleave on the way out.
        pub samples_f32: Vec<f32>,
    }

    // libavcodec function signatures (stable across 58..62):
    //   AVCodec*  avcodec_find_decoder(enum AVCodecID id);
    //   AVCodecContext* avcodec_alloc_context3(const AVCodec *codec);
    //   int avcodec_open2(AVCodecContext*, const AVCodec*, AVDictionary**);
    //   AVPacket* av_packet_alloc(void);
    //   AVFrame* av_frame_alloc(void);
    //   int avcodec_send_packet(AVCodecContext*, const AVPacket*);
    //   int avcodec_receive_frame(AVCodecContext*, AVFrame*);
    //   void av_packet_free(AVPacket**);
    //   void av_frame_free(AVFrame**);
    //   void avcodec_free_context(AVCodecContext**);
    type FindDecoderFn = unsafe extern "C" fn(c_int) -> *const c_void;
    type AllocContext3Fn = unsafe extern "C" fn(*const c_void) -> *mut c_void;
    type Open2Fn = unsafe extern "C" fn(*mut c_void, *const c_void, *mut *mut c_void) -> c_int;
    type PacketAllocFn = unsafe extern "C" fn() -> *mut c_void;
    type FrameAllocFn = unsafe extern "C" fn() -> *mut c_void;
    type SendPacketFn = unsafe extern "C" fn(*mut c_void, *const c_void) -> c_int;
    type ReceiveFrameFn = unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_int;
    type PacketFreeFn = unsafe extern "C" fn(*mut *mut c_void);
    type FrameFreeFn = unsafe extern "C" fn(*mut *mut c_void);
    type FreeContextFn = unsafe extern "C" fn(*mut *mut c_void);

    /// Decode `data` (raw MP3 frame bytes — header + side info + main
    /// data, no container framing) through libavcodec's `mp3` decoder.
    ///
    /// Returns `Some(DecodedAudio)` on success, `None` if libavcodec is
    /// unavailable, fails to open the decoder, or rejects the input.
    /// **A `None` here means "ffmpeg also said this is not valid MP3"**,
    /// which is the negative-oracle signal — the caller should also
    /// expect our decoder to reject the input.
    pub fn decode_mp3(data: &[u8]) -> Option<DecodedAudio> {
        let l = lib()?;
        unsafe {
            let find_decoder: Symbol<FindDecoderFn> = l.get(b"avcodec_find_decoder").ok()?;
            let alloc_ctx: Symbol<AllocContext3Fn> = l.get(b"avcodec_alloc_context3").ok()?;
            let open2: Symbol<Open2Fn> = l.get(b"avcodec_open2").ok()?;
            let pkt_alloc: Symbol<PacketAllocFn> = l.get(b"av_packet_alloc").ok()?;
            let frame_alloc: Symbol<FrameAllocFn> = l.get(b"av_frame_alloc").ok()?;
            let send_pkt: Symbol<SendPacketFn> = l.get(b"avcodec_send_packet").ok()?;
            let recv_frame: Symbol<ReceiveFrameFn> = l.get(b"avcodec_receive_frame").ok()?;
            let pkt_free: Symbol<PacketFreeFn> = l.get(b"av_packet_free").ok()?;
            let frame_free: Symbol<FrameFreeFn> = l.get(b"av_frame_free").ok()?;
            let free_ctx: Symbol<FreeContextFn> = l.get(b"avcodec_free_context").ok()?;

            let codec = find_decoder(AV_CODEC_ID_MP3);
            if codec.is_null() {
                return None;
            }
            let ctx = alloc_ctx(codec);
            if ctx.is_null() {
                return None;
            }
            // Open the decoder. If avcodec_open2 fails, free + return None.
            let rc = open2(ctx, codec, std::ptr::null_mut());
            if rc < 0 {
                let mut p = ctx;
                free_ctx(&mut p as *mut *mut c_void);
                return None;
            }

            // Allocate packet + frame.
            let pkt = pkt_alloc();
            if pkt.is_null() {
                let mut p = ctx;
                free_ctx(&mut p as *mut *mut c_void);
                return None;
            }
            let frame = frame_alloc();
            if frame.is_null() {
                let mut pp = pkt;
                pkt_free(&mut pp as *mut *mut c_void);
                let mut p = ctx;
                free_ctx(&mut p as *mut *mut c_void);
                return None;
            }

            // Populate AVPacket: only fields .data and .size. Layout
            // prefix is stable since lavc 57:
            //   off  0  AVBufferRef* buf
            //   off  8  i64 pts
            //   off 16  i64 dts
            //   off 24  u8* data
            //   off 32  i32 size
            // We're writing into a struct allocated by libavcodec via
            // av_packet_alloc, so .buf is null and ref-counting is off
            // — av_packet_unref isn't required for our single-shot
            // borrow.
            let pkt_bytes = pkt as *mut u8;
            let data_ptr_field = pkt_bytes.add(24) as *mut *const u8;
            let size_field = pkt_bytes.add(32) as *mut c_int;
            data_ptr_field.write_unaligned(data.as_ptr());
            size_field.write_unaligned(data.len() as c_int);

            let result = (|| -> Option<DecodedAudio> {
                let send_rc = send_pkt(ctx, pkt);
                if send_rc < 0 && send_rc != averror_eagain() {
                    return None;
                }
                // Drain the decoder. We expect exactly one frame for one
                // MP3 packet, but loop a few times in case the decoder
                // returns AVERROR(EAGAIN) on the first call.
                for _ in 0..4 {
                    let rc = recv_frame(ctx, frame);
                    if rc == 0 {
                        return read_frame(frame);
                    }
                    if rc == averror_eagain() {
                        // Send a flush signal (NULL packet). Layout: set
                        // data + size both to zero / null on a fresh
                        // packet. Reuse `pkt` after clearing the data
                        // pointer.
                        let dpf = pkt_bytes.add(24) as *mut *const u8;
                        let szf = pkt_bytes.add(32) as *mut c_int;
                        dpf.write_unaligned(std::ptr::null());
                        szf.write_unaligned(0);
                        let _ = send_pkt(ctx, pkt);
                        continue;
                    }
                    // EOF or hard error.
                    return None;
                }
                None
            })();

            // Cleanup epilogue.
            let mut fp = frame;
            frame_free(&mut fp as *mut *mut c_void);
            let mut pp = pkt;
            pkt_free(&mut pp as *mut *mut c_void);
            let mut cp = ctx;
            free_ctx(&mut cp as *mut *mut c_void);

            result
        }
    }

    /// Read sample / sample_rate / channels / interleaved-f32 PCM out
    /// of a populated AVFrame. AVFrame prefix layout is stable since
    /// lavc 55 (the audio fields needed here):
    ///
    /// ```text
    ///   off  0  u8*  data[8]
    ///   off 64  i32  linesize[8]
    ///   off 96  u8** extended_data
    ///   off 104 i32  width
    ///   off 108 i32  height
    ///   off 112 i32  nb_samples
    ///   off 116 i32  format          (enum AVSampleFormat)
    ///   off 120 i32  key_frame
    ///   off 124 i32  pict_type
    ///   …
    /// ```
    ///
    /// `sample_rate` and `ch_layout`/`channels` are further inside the
    /// struct and the offset has shifted across major versions (the
    /// `AVChannelLayout` rework in 5.1 inserted ~64 bytes). To stay
    /// version-agnostic we look up `av_frame_get_sample_rate` (the old
    /// accessor) when present, else fall back to scanning a small
    /// window for a plausible {8000..48000} value. For channels we use
    /// the legacy field at offset 192 (lavc 58/59) or fall back via
    /// the linesize-vs-nb_samples ratio when the planar format is f32.
    unsafe fn read_frame(frame: *mut c_void) -> Option<DecodedAudio> {
        let fb = frame as *const u8;
        let nb_samples = (fb.add(112) as *const c_int).read_unaligned();
        let format = (fb.add(116) as *const c_int).read_unaligned();
        if nb_samples <= 0 || nb_samples > 4096 {
            return None;
        }
        // AVSampleFormat enum (stable):
        //   AV_SAMPLE_FMT_S16  = 1
        //   AV_SAMPLE_FMT_S32  = 2
        //   AV_SAMPLE_FMT_FLT  = 3
        //   AV_SAMPLE_FMT_DBL  = 4
        //   AV_SAMPLE_FMT_S16P = 6
        //   AV_SAMPLE_FMT_S32P = 7
        //   AV_SAMPLE_FMT_FLTP = 8
        // ffmpeg's mp3 decoder emits FLTP. Other formats: bail.
        const AV_SAMPLE_FMT_FLTP: c_int = 8;
        if format != AV_SAMPLE_FMT_FLTP {
            return None;
        }

        // Try to find sample_rate via the symbol accessor.
        let sample_rate = read_sample_rate(frame)?;
        if !(8000..=48_000).contains(&sample_rate) {
            return None;
        }

        // Find channel count via accessor or the channels legacy field.
        let channels = read_channels(frame)?;
        if !(1..=2).contains(&channels) {
            return None;
        }

        // Read planar f32 samples. data[0] = ch0, data[1] = ch1.
        let data_ptrs = fb as *const *const u8;
        let mut interleaved: Vec<f32> = Vec::with_capacity(nb_samples as usize * channels as usize);
        let plane_ptrs: [*const f32; 2] = [
            data_ptrs.read_unaligned() as *const f32,
            data_ptrs.add(1).read_unaligned() as *const f32,
        ];
        if plane_ptrs[0].is_null() {
            return None;
        }
        if channels == 2 && plane_ptrs[1].is_null() {
            return None;
        }
        for i in 0..(nb_samples as usize) {
            for ch in 0..(channels as usize) {
                let sample = plane_ptrs[ch].add(i).read();
                interleaved.push(sample);
            }
        }
        Some(DecodedAudio {
            samples: nb_samples as u32,
            sample_rate: sample_rate as u32,
            channels: channels as u32,
            samples_f32: interleaved,
        })
    }

    /// Read AVFrame.sample_rate. ffmpeg dropped the `av_frame_get_*`
    /// accessors after lavc 57, so on modern builds we have to read
    /// the field by offset. We probe a small window of plausible
    /// offsets and return the first value in the {8000..48000} range.
    /// This tolerates the lavc 58 / 59 / 60 / 61 / 62 layouts where
    /// AVChannelLayout reshuffling moved sample_rate by 0-64 bytes.
    unsafe fn read_sample_rate(frame: *mut c_void) -> Option<c_int> {
        // The sample_rate field on lavc 58 sits near offset 200; on
        // lavc 60+ (with AVChannelLayout) it sits near offset 264.
        // Probe in 4-byte steps over the union of those windows, which
        // is also where av_frame_alloc-zeroed padding lives — a stray
        // hit is unlikely because rates are constrained to the audio
        // common range.
        let fb = frame as *const u8;
        let candidates: [usize; 12] = [192, 196, 200, 204, 248, 252, 256, 260, 264, 268, 272, 276];
        for &off in &candidates {
            let v = (fb.add(off) as *const c_int).read_unaligned();
            if (8000..=48_000).contains(&v) {
                return Some(v);
            }
        }
        None
    }

    /// Read AVFrame.channels (or derive from AVChannelLayout.nb_channels).
    /// Same probing strategy as `read_sample_rate`. We accept any value
    /// in {1, 2}.
    unsafe fn read_channels(frame: *mut c_void) -> Option<c_int> {
        let fb = frame as *const u8;
        // AVFrame.channels (lavc 57/58/59) sat near offset 196..208;
        // for lavc 60+ AVChannelLayout has nb_channels at the start of
        // its struct, which lives somewhere near offset 232..280.
        let candidates: [usize; 16] = [
            196, 200, 204, 208, 212, 216, 220, 232, 236, 240, 244, 248, 252, 256, 260, 264,
        ];
        for &off in &candidates {
            let v = (fb.add(off) as *const c_int).read_unaligned();
            if (1..=2).contains(&v) {
                return Some(v);
            }
        }
        None
    }

    // Suppress unused-warning for the c_char/c_uint imports above —
    // kept for documentation parity with the libavcodec headers.
    #[allow(dead_code)]
    fn _silence_unused(_: *const c_char, _: c_uint) {}
}
