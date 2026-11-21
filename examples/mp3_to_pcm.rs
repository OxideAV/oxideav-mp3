//! Decode an MPEG audio Layer III file to raw interleaved 16-bit
//! little-endian PCM.
//!
//! ```text
//! cargo run --release --example mp3_to_pcm -- in.mp3 out.pcm
//! ```
//!
//! Prints the sampling frequency, channel count, sample count and the
//! decode speed (× realtime, one thread) on stderr.

use std::io::{Cursor, Write};
use std::time::Instant;

use oxideav_core::{CodecId, CodecParameters, Decoder, Demuxer, Error, Frame};
use oxideav_mp3::Mp3Demuxer;
use oxideav_mp3::codec_decoder::make_decoder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: mp3_to_pcm <in.mp3> <out.pcm>");
        std::process::exit(2);
    }
    let bytes = std::fs::read(&args[1])?;
    let t0 = Instant::now();
    let mut demux = Mp3Demuxer::open(Box::new(Cursor::new(bytes)))?;
    let mut dec = make_decoder(&CodecParameters::audio(CodecId::new("mp3")))?;
    let mut pcm: Vec<u8> = Vec::new();
    let mut frames = 0u64;
    let mut seconds = 0f64;
    let mut shape = (0u32, 0usize);
    let mut drain = |dec: &mut dyn Decoder, pcm: &mut Vec<u8>| -> Result<(), Error> {
        loop {
            match dec.receive_frame() {
                Ok(Frame::Audio(af)) => {
                    let n = af.data[0].len() / 2;
                    for i in 0..n {
                        for plane in &af.data {
                            pcm.extend_from_slice(&plane[2 * i..2 * i + 2]);
                        }
                    }
                    let rate = dec.output_params().and_then(|p| p.sample_rate);
                    shape = (rate.unwrap_or(0), af.data.len());
                    frames += n as u64;
                    seconds += n as f64 / f64::from(shape.0.max(1));
                }
                Ok(_) => {}
                Err(Error::NeedMore) | Err(Error::Eof) => return Ok(()),
                Err(e) => return Err(e),
            }
        }
    };
    loop {
        match demux.next_packet() {
            Ok(p) => {
                dec.send_packet(&p)?;
                drain(dec.as_mut(), &mut pcm)?;
            }
            Err(Error::Eof) => break,
            Err(e) => return Err(e.into()),
        }
    }
    dec.flush()?;
    drain(dec.as_mut(), &mut pcm)?;
    let dt = t0.elapsed().as_secs_f64();
    std::fs::File::create(&args[2])?.write_all(&pcm)?;
    eprintln!(
        "{} Hz, {} ch, {frames} samples, {seconds:.2} s decoded in {dt:.3} s = {:.0}x realtime",
        shape.0,
        shape.1,
        seconds / dt.max(1e-9)
    );
    Ok(())
}
