# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.6](https://github.com/OxideAV/oxideav-mp3/compare/v0.0.5...v0.0.6) - 2026-04-26

### Other

- adopt slim VideoFrame/AudioFrame shape
- pin release-plz to patch-only bumps

## [0.0.5](https://github.com/OxideAV/oxideav-mp3/compare/v0.0.4...v0.0.5) - 2026-04-25

### Other

- drop oxideav-codec/oxideav-container shims, import from oxideav-core
- implement MPEG-2 / MPEG-2.5 intensity stereo
- add NOTICES for minimp3 references
- add ffmpeg-based intensity-stereo end-to-end test
- implement MPEG-1 intensity stereo decode
- enable MPEG-2.5 decode (8 / 11.025 / 12 kHz)
- bump oxideav-container dep to "0.1"
- drop Cargo.lock — this crate is a library
- bump oxideav-core / oxideav-codec dep examples to "0.1"
- bump to oxideav-core 0.1.1 + codec 0.1.1
- migrate register() to CodecInfo builder
- bump oxideav-core + oxideav-codec deps to "0.1"
- thread &dyn CodecResolver through open()

## [0.0.4](https://github.com/OxideAV/oxideav-mp3/compare/v0.0.3...v0.0.4) - 2026-04-19

### Other

- claim WAVEFORMATEX tag via oxideav-codec CodecTag registry
- claim WAVEFORMATEX tag via oxideav-codec CodecTag registry
- bump oxideav-container + oxideav-id3 to 0.0.4
- drop duplicate oxideav-codec entry
- bump oxideav-core to 0.0.5
- migrate to oxideav_core::bits shared BitReader / BitWriter
- refresh README + module docstrings to match shipped decoder/encoder
- add MPEG-2 LSF encoder — 16/22.05/24 kHz, 576-sample single-granule frames
