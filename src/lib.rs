//! MPEG-1 / MPEG-2 Audio Layer III (MP3) pure-Rust decoder + CBR encoder.
//!
//! Decoder covers MPEG-1 (32 / 44.1 / 48 kHz), MPEG-2 LSF
//! (16 / 22.05 / 24 kHz), and MPEG-2.5 (unofficial Fraunhofer extension
//! at 8 / 11.025 / 12 kHz) Layer III — mono / stereo / joint-stereo
//! (M/S and intensity) / dual-channel, all block types, bit reservoir,
//! and scfsi reuse. CRC bytes are consumed but not verified.
//!
//! Encoder is a minimum-viable MPEG-1 / MPEG-2 LSF Layer III encoder
//! covering all four block types (long / start / short / stop) with
//! transient-driven window switching (see [`block_type`]); optional MS
//! joint-stereo (picked per frame from spectral correlation; ISO/IEC
//! 11172-3 §2.4.3.4.10), with two rate-control strategies: CBR
//! (global-gain bisection to fit a fixed per-frame bit budget) and VBR
//! with a switchable psychoacoustic model — `psy_model = 0` is the
//! lightweight per-sfb energy mask in [`psy`]; `psy_model = 1`
//! (default) is the full ISO/IEC 11172-3 Annex D Psy Model 1 in
//! [`psy1`] with 24 Bark-partition spreading, SFM-based tonality
//! estimate, tonal-vs-noise SNR offsets, and an iterate-until-stable
//! noise-allocation outer loop. The bit reservoir rolls forward via
//! `main_data_begin` across frames within the per-version lookback cap
//! (511 bytes MPEG-1 / 255 bytes MPEG-2 LSF).
//!
//! The container demuxer at [`container`] accepts `.mp1` / `.mp2` / `.mp3`
//! raw streams (with ID3v2 prefix + ID3v1 trailer) and routes packets to
//! the matching `mp1`/`mp2`/`mp3` codec.

#![allow(
    dead_code,
    clippy::needless_range_loop,
    clippy::unnecessary_cast,
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items,
    clippy::excessive_precision,
    clippy::unusual_byte_groupings,
    clippy::too_many_arguments
)]

pub mod analysis;
pub mod block_type;
pub mod container;
pub mod decoder;
pub mod encoder;
pub mod fft;
pub mod frame;
pub mod huffman;
pub mod imdct;
pub mod mdct;
pub mod psy;
pub mod psy1;
pub mod psy2;
pub mod requantize;
pub mod reservoir;
pub mod scalefactor;
pub mod sfband;
pub mod sideinfo;
pub mod synthesis;
pub mod window;

use oxideav_core::ContainerRegistry;
use oxideav_core::RuntimeContext;
use oxideav_core::{CodecCapabilities, CodecId, CodecParameters, CodecTag, Result};
use oxideav_core::{CodecInfo, CodecRegistry, Decoder, Encoder};

pub const CODEC_ID_STR: &str = "mp3";

/// Unified entry point: install every codec and container provided by
/// `oxideav-mp3` into a [`RuntimeContext`].
///
/// Also wired into [`oxideav_meta::register_all`] via the
/// [`oxideav_core::register!`] macro below.
pub fn register(ctx: &mut RuntimeContext) {
    register_codecs(&mut ctx.codecs);
    register_containers(&mut ctx.containers);
}

oxideav_core::register!("mp3", register);

pub fn register_codecs(reg: &mut CodecRegistry) {
    let cid = CodecId::new(CODEC_ID_STR);
    let dec_caps = CodecCapabilities::audio("mp3_sw_dec")
        .with_lossy(true)
        .with_intra_only(false) // MP3 uses a bit reservoir — not intra-only
        .with_max_channels(2)
        .with_max_sample_rate(48_000);
    // AVI / WAVEFORMATEX tag: WAVE_FORMAT_MPEGLAYER3 = 0x0055. The
    // generic MPEG audio tag 0x0050 is owned by oxideav-mp2 (Layers
    // I + II).
    reg.register(
        CodecInfo::new(cid.clone())
            .capabilities(dec_caps)
            .decoder(make_decoder)
            .tag(CodecTag::wave_format(0x0055)),
    );

    let enc_caps = CodecCapabilities::audio("mp3_sw_enc")
        .with_lossy(true)
        .with_intra_only(false)
        .with_max_channels(2)
        .with_max_sample_rate(48_000);
    reg.register(
        CodecInfo::new(cid)
            .capabilities(enc_caps)
            .encoder(make_encoder),
    );
}

pub fn register_containers(reg: &mut ContainerRegistry) {
    container::register(reg);
}

fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    decoder::make_decoder(params)
}

fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    encoder::make_encoder(params)
}

#[cfg(test)]
mod register_tests {
    use super::*;

    #[test]
    fn register_via_runtime_context_installs_factories() {
        let mut ctx = RuntimeContext::new();
        register(&mut ctx);
        assert!(
            ctx.codecs.decoder_ids().next().is_some(),
            "register(ctx) should install codec decoder factories"
        );
        assert!(
            ctx.containers.demuxer_names().next().is_some(),
            "register(ctx) should install container demuxer factories"
        );
    }
}
