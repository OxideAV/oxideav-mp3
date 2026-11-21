//! Objective quality figure for a lossy round trip: the **segmental
//! critical-band signal-to-noise ratio** of a decode against its
//! original.
//!
//! ```text
//! cargo run --release --example band_snr -- original.wav decoded.wav
//! ```
//!
//! Both files are 16-bit PCM WAVE, sample-aligned (decode with gapless
//! trimming on). The signals are cut into 1024-sample Hann frames
//! (hop 512). In every frame the power spectra of the original and of
//! the error (original minus decode) are summed into 25 critical bands
//! (edges below); a band counts when the original has audible energy
//! in it (above -70 dB re full scale). Its figure is
//! `10 log10(signal / noise)` clamped to `[-10, 50]` dB, and the
//! result is the mean over all counted (frame, band) cells, per file
//! and over channels.
//!
//! Unlike a plain SNR this does not let one loud band hide noise in a
//! quiet one, and unlike a waveform PSNR it weights every band of every
//! frame equally, so noise that a coder moves into quiet bands or
//! quiet frames (pre-echo before an attack) lowers it. It is not a
//! listening test. The plain SNR is printed next to it.

use std::f64::consts::PI;

/// Critical-band edges in Hz (Zwicker's 24 Bark bands, the last one
/// extended to cover up to 22.05 kHz).
const BAND_EDGES_HZ: [f64; 26] = [
    0.0, 100.0, 200.0, 300.0, 400.0, 510.0, 630.0, 770.0, 920.0, 1080.0, 1270.0, 1480.0, 1720.0,
    2000.0, 2320.0, 2700.0, 3150.0, 3700.0, 4400.0, 5300.0, 6400.0, 7700.0, 9500.0, 12000.0,
    15500.0, 22050.0,
];

const FRAME: usize = 1024;
const HOP: usize = 512;

fn read_wav(path: &str) -> Result<(u32, usize, Vec<i16>), Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() < 12 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(format!("{path}: not a RIFF/WAVE file").into());
    }
    let (mut rate, mut channels) = (0u32, 0usize);
    let mut at = 12usize;
    while at + 8 <= bytes.len() {
        let id = &bytes[at..at + 4];
        let len = u32::from_le_bytes(bytes[at + 4..at + 8].try_into()?) as usize;
        let body = &bytes[at + 8..(at + 8 + len).min(bytes.len())];
        if id == b"fmt " && body.len() >= 16 {
            channels = usize::from(u16::from_le_bytes([body[2], body[3]]));
            rate = u32::from_le_bytes([body[4], body[5], body[6], body[7]]);
            if u16::from_le_bytes([body[14], body[15]]) != 16 {
                return Err(format!("{path}: not 16-bit PCM").into());
            }
        } else if id == b"data" {
            let pcm = body
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| i16::from_le_bytes(*c))
                .collect();
            return Ok((rate, channels.max(1), pcm));
        }
        at += 8 + len + (len & 1);
    }
    Err(format!("{path}: no data chunk").into())
}

/// In-place radix-2 FFT of `(re, im)`; the length is a power of two.
fn fft(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let angle = -2.0 * PI / len as f64;
        for start in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let (s, c) = (angle * k as f64).sin_cos();
                let (a, b) = (start + k, start + k + len / 2);
                let (tr, ti) = (re[b] * c - im[b] * s, re[b] * s + im[b] * c);
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
            }
        }
        len <<= 1;
    }
}

/// Band powers of one Hann-windowed frame.
fn band_powers(frame: &[f64], rate: u32, window: &[f64]) -> [f64; 25] {
    let mut re: Vec<f64> = frame.iter().zip(window).map(|(x, w)| x * w).collect();
    let mut im = vec![0.0; FRAME];
    fft(&mut re, &mut im);
    let mut bands = [0.0f64; 25];
    let hz_per_bin = f64::from(rate) / FRAME as f64;
    for bin in 1..FRAME / 2 {
        let hz = bin as f64 * hz_per_bin;
        if let Some(b) = BAND_EDGES_HZ
            .windows(2)
            .position(|e| hz >= e[0] && hz < e[1])
        {
            bands[b] += re[bin] * re[bin] + im[bin] * im[bin];
        }
    }
    bands
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: band_snr <original.wav> <decoded.wav>");
        std::process::exit(2);
    }
    let (rate, channels, original) = read_wav(&args[1])?;
    let (rate2, channels2, decoded) = read_wav(&args[2])?;
    if rate != rate2 || channels != channels2 {
        return Err("sampling frequency / channel count differ".into());
    }
    let frames = original.len().min(decoded.len()) / channels;
    let window: Vec<f64> = (0..FRAME)
        .map(|n| 0.5 - 0.5 * (2.0 * PI * n as f64 / FRAME as f64).cos())
        .collect();
    // -70 dB re full scale, as a band power of a windowed frame.
    let audible = 10f64.powf(-7.0) * (FRAME as f64 * 0.375) * (FRAME as f64 / 2.0);
    let (mut sum_db, mut cells) = (0.0f64, 0u64);
    let (mut signal, mut noise) = (0.0f64, 0.0f64);
    for ch in 0..channels {
        let x: Vec<f64> = (0..frames)
            .map(|i| f64::from(original[i * channels + ch]) / 32768.0)
            .collect();
        let e: Vec<f64> = (0..frames)
            .map(|i| {
                (f64::from(original[i * channels + ch]) - f64::from(decoded[i * channels + ch]))
                    / 32768.0
            })
            .collect();
        signal += x.iter().map(|v| v * v).sum::<f64>();
        noise += e.iter().map(|v| v * v).sum::<f64>();
        let mut at = 0usize;
        while at + FRAME <= frames {
            let s = band_powers(&x[at..at + FRAME], rate, &window);
            let n = band_powers(&e[at..at + FRAME], rate, &window);
            for b in 0..25 {
                if s[b] > audible {
                    let db = 10.0 * (s[b] / n[b].max(1e-30)).log10();
                    sum_db += db.clamp(-10.0, 50.0);
                    cells += 1;
                }
            }
            at += HOP;
        }
    }
    let band_snr = if cells > 0 {
        sum_db / cells as f64
    } else {
        0.0
    };
    let snr = 10.0 * (signal / noise.max(1e-30)).log10();
    println!("band-SNR {band_snr:6.2} dB   SNR {snr:6.2} dB   ({cells} cells)");
    Ok(())
}
