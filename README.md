# oxideav-mp3

Pure-Rust **MPEG-1 / MPEG-2 Audio Layer III (MP3)** decoder + CBR
encoder + container demuxer. MPEG-1 covers 32 / 44.1 / 48 kHz; MPEG-2
LSF covers 16 / 22.05 / 24 kHz. Handles mono, stereo, joint-stereo
(M/S), dual-channel, every block type (long, short, start, stop,
mixed), the bit reservoir, scfsi scalefactor reuse, and ID3v2 / ID3v1
tags including attached pictures. Zero C dependencies.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework but usable standalone.

## Installation

```toml
[dependencies]
oxideav-core = "0.1"
oxideav-codec = "0.1"
oxideav-container = "0.1"
oxideav-mp3 = "0.0"
```

## Quick use

```rust
use oxideav_codec::CodecRegistry;
use oxideav_container::ContainerRegistry;
use oxideav_core::Frame;

let mut codecs = CodecRegistry::new();
let mut containers = ContainerRegistry::new();
oxideav_mp3::register_codecs(&mut codecs);
oxideav_mp3::register_containers(&mut containers);

let input: Box<dyn oxideav_container::ReadSeek> = Box::new(
    std::io::Cursor::new(std::fs::read("song.mp3")?),
);
let mut dmx = containers.open("mp3", input)?;
let stream = &dmx.streams()[0];
let mut dec = codecs.make_decoder(&stream.params)?;

loop {
    match dmx.next_packet() {
        Ok(pkt) => {
            dec.send_packet(&pkt)?;
            while let Ok(Frame::Audio(af)) = dec.receive_frame() {
                // af.format == SampleFormat::S16 interleaved
                // af.channels, af.sample_rate, af.samples
            }
        }
        Err(oxideav_core::Error::Eof) => break,
        Err(e) => return Err(e.into()),
    }
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

### Encoder

```rust
use oxideav_core::{CodecId, CodecParameters, SampleFormat};

let mut params = CodecParameters::audio(CodecId::new("mp3"));
params.channels = Some(2);
params.sample_rate = Some(44_100);
params.sample_format = Some(SampleFormat::S16);
params.bit_rate = Some(192_000);
let mut enc = codecs.make_encoder(&params)?;

enc.send_frame(&Frame::Audio(pcm_frame_s16_interleaved))?;
enc.flush()?;
while let Ok(pkt) = enc.receive_packet() {
    // pkt.data is a raw MPEG audio frame ready to concatenate.
}
```

## Decoder support

- **Versions**: MPEG-1 Layer III, MPEG-2 LSF Layer III.
- **Sample rates**: 16, 22.05, 24, 32, 44.1, 48 kHz.
- **Channel modes**: mono, stereo, joint-stereo (M/S), dual-channel.
- **Block types**: normal long, start, short, stop, mixed.
- **Bit reservoir**: 511-byte lookback (MPEG-1) / 255-byte (MPEG-2 LSF).
- **scfsi**: scalefactor reuse across the two granules of an MPEG-1 frame.
- **Bitrates**: every standard CBR slot in both version tables.

Not implemented: MPEG-2.5 (8 / 11.025 / 12 kHz), intensity stereo,
CRC-16 verification (CRC bytes are consumed but not checked), free-
format bitstreams.

## Encoder support

Minimum-viable CBR encoder for archival / interop:

- **Versions**: MPEG-1 (32 / 44.1 / 48 kHz) and MPEG-2 LSF
  (16 / 22.05 / 24 kHz). MPEG-1 emits 2 granules per frame, MPEG-2 LSF
  emits 1.
- **Channels**: mono or dual-channel stereo. No joint-stereo, no
  intensity-stereo.
- **Blocks**: long blocks only (block_type = 0). No window switching.
- **Bitrate**: one CBR rate per encoder instance, picked from the
  standard version-specific bitrate table (defaults: 128 kbps MPEG-1,
  64 kbps MPEG-2 LSF).
- **Bit reservoir**: rolled forward via `main_data_begin` within the
  per-version cap (511 bytes MPEG-1 / 255 bytes MPEG-2 LSF).
- **Quantisation**: global-gain bisection to fit the per-granule bit
  budget. No psychoacoustic model. No CRC. count1 uses table A.

Input must be `SampleFormat::S16` interleaved PCM.

## Container

The `mp3` demuxer accepts raw `.mp1` / `.mp2` / `.mp3` streams:

- Skips an ID3v2 prefix at offset 0, parsing text frames and attached
  pictures through [`oxideav-id3`](https://crates.io/crates/oxideav-id3).
- Probes the first valid MPEG audio frame and emits `CodecParameters`
  with the correct codec id — `"mp1"`, `"mp2"`, or `"mp3"` — so the
  codec registry dispatches to the matching decoder.
- Emits one frame per packet, with PTS in samples at the native rate.
- Resyncs across up to 64 KiB of mid-stream garbage (embedded tags, APE
  footers, padding) the way mpg123 / ffmpeg / VLC do.
- Merges an ID3v1 trailer (last 128 bytes) if present, filling in
  whatever the v2 tag didn't already carry.

Free-format streams (`bitrate_index == 0`) are rejected since their
frame length has to be recovered by sync search.

## Codec / container IDs

- Codec: `"mp3"`; I/O sample format `S16` interleaved.
- Container: `"mp3"`, matches `.mp1` / `.mp2` / `.mp3` by extension and
  by sync-word probe (or an ID3v2 prefix).

## License

MIT — see [LICENSE](LICENSE).
