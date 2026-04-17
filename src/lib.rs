//! MPEG-1 / MPEG-2 Audio Layer III (MP3) pure-Rust decoder + CBR encoder.
//!
//! Decoder covers MPEG-1 (32 / 44.1 / 48 kHz) and MPEG-2 LSF
//! (16 / 22.05 / 24 kHz) Layer III, mono / stereo / joint-stereo M/S /
//! dual-channel, all block types, bit reservoir, and scfsi reuse.
//! MPEG-2.5 and intensity-stereo are not implemented; CRC bytes are
//! consumed but not verified.
//!
//! Encoder is a minimum-viable CBR MPEG-1 / MPEG-2 LSF Layer III encoder:
//! long blocks only, no joint-stereo, no psychoacoustic model, global-gain
//! bisection to fit a per-frame bit budget, with the bit reservoir rolled
//! forward so `main_data_begin` carries across frames within the per-
//! version lookback cap (511 bytes MPEG-1 / 255 bytes MPEG-2).
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
pub mod bitreader;
pub mod bitwriter;
pub mod container;
pub mod decoder;
pub mod encoder;
pub mod frame;
pub mod huffman;
pub mod imdct;
pub mod mdct;
pub mod requantize;
pub mod reservoir;
pub mod scalefactor;
pub mod sfband;
pub mod sideinfo;
pub mod synthesis;
pub mod window;

use oxideav_codec::{CodecRegistry, Decoder, Encoder};
use oxideav_container::ContainerRegistry;
use oxideav_core::{CodecCapabilities, CodecId, CodecParameters, Result};

pub const CODEC_ID_STR: &str = "mp3";

/// Back-compat alias for callers that wired up the codec-only `register`.
/// Prefer `register_codecs` + `register_containers`.
pub fn register(reg: &mut CodecRegistry) {
    register_codecs(reg);
}

pub fn register_codecs(reg: &mut CodecRegistry) {
    let dec_caps = CodecCapabilities::audio("mp3_sw_dec")
        .with_lossy(true)
        .with_intra_only(false) // MP3 uses a bit reservoir — not intra-only
        .with_max_channels(2)
        .with_max_sample_rate(48_000);
    reg.register_decoder_impl(CodecId::new(CODEC_ID_STR), dec_caps, make_decoder);

    let enc_caps = CodecCapabilities::audio("mp3_sw_enc")
        .with_lossy(true)
        .with_intra_only(false)
        .with_max_channels(2)
        .with_max_sample_rate(48_000);
    reg.register_encoder_impl(CodecId::new(CODEC_ID_STR), enc_caps, make_encoder);
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
