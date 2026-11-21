//! Encode a 16-bit PCM RIFF/WAVE file to MPEG audio Layer III.
//!
//! ```text
//! cargo run --release --example wav_to_mp3 -- in.wav out.mp3 128 [joint|stereo|mono] [preset]
//! ```
//!
//! `preset` is one of `fast`, `standard`, `high`, `transparent` (the
//! psychoacoustic quality presets); without it the encoder's default
//! rate/distortion loop is used. The output leads with the gapless
//! information frame. Prints the encode speed (x realtime, one thread)
//! on stderr.

use std::time::Instant;

use oxideav_mp3::{ChannelMode, Mp3Encoder, QualityPreset};

/// `(sampling frequency, channels, interleaved samples)` of a 16-bit
/// PCM WAVE file.
fn read_wav(bytes: &[u8]) -> Option<(u32, usize, Vec<i16>)> {
    if bytes.len() < 12 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return None;
    }
    let (mut rate, mut channels, mut bits) = (0u32, 0usize, 0u16);
    let mut at = 12usize;
    while at + 8 <= bytes.len() {
        let id = &bytes[at..at + 4];
        let len = u32::from_le_bytes(bytes[at + 4..at + 8].try_into().ok()?) as usize;
        let body = &bytes[at + 8..(at + 8 + len).min(bytes.len())];
        if id == b"fmt " && body.len() >= 16 {
            channels = usize::from(u16::from_le_bytes([body[2], body[3]]));
            rate = u32::from_le_bytes([body[4], body[5], body[6], body[7]]);
            bits = u16::from_le_bytes([body[14], body[15]]);
        } else if id == b"data" {
            if bits != 16 || channels == 0 {
                return None;
            }
            let pcm = body
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| i16::from_le_bytes(*c))
                .collect();
            return Some((rate, channels, pcm));
        }
        at += 8 + len + (len & 1);
    }
    None
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: wav_to_mp3 <in.wav> <out.mp3> <kbit/s> [joint|stereo|mono] [preset]");
        std::process::exit(2);
    }
    let (rate, channels, mut pcm) =
        read_wav(&std::fs::read(&args[1])?).ok_or("not a 16-bit PCM WAVE file")?;
    let kbps: u32 = args[3].parse()?;
    let mode = args.get(4).map_or("joint", String::as_str);
    let preset = match args.get(5).map(String::as_str) {
        None => None,
        Some("fast") => Some(QualityPreset::Fast),
        Some("standard") => Some(QualityPreset::Standard),
        Some("high") => Some(QualityPreset::High),
        Some("transparent") => Some(QualityPreset::Transparent),
        Some(other) => return Err(format!("unknown preset {other}").into()),
    };
    if channels > 2 {
        return Err("more than two channels".into());
    }
    if channels == 2 && mode == "mono" {
        pcm = pcm
            .as_chunks::<2>()
            .0
            .iter()
            .map(|p| ((i32::from(p[0]) + i32::from(p[1])) / 2) as i16)
            .collect();
    }
    let mut enc = match (channels, mode) {
        (1, _) | (_, "mono") => Mp3Encoder::new(kbps, rate, ChannelMode::SingleChannel)?,
        (_, "stereo") => Mp3Encoder::new(kbps, rate, ChannelMode::Stereo)?,
        _ => Mp3Encoder::new_joint_stereo_auto(kbps, rate)?,
    };
    if let Some(p) = preset {
        enc.with_quality_preset(p)?;
    }
    enc.enable_info_frame();
    let out_channels = if mode == "mono" { 1 } else { channels };
    let seconds = (pcm.len() / out_channels) as f64 / f64::from(rate);
    let t0 = Instant::now();
    enc.push_samples(&pcm)?;
    let mut out = Vec::new();
    enc.finish(&mut out)?;
    let dt = t0.elapsed().as_secs_f64();
    std::fs::write(&args[2], &out)?;
    eprintln!(
        "{rate} Hz, {out_channels} ch, {seconds:.2} s -> {} bytes ({:.1} kbit/s) in {dt:.3} s = {:.1}x realtime",
        out.len(),
        out.len() as f64 * 8.0 / seconds / 1000.0,
        seconds / dt.max(1e-9)
    );
    Ok(())
}
