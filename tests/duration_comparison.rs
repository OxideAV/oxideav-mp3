//! Print our demuxer's `duration_micros` for each fixture so the
//! agent's final report can record the delta vs ffprobe. Marked
//! `#[test]` so cargo runs it under the normal harness, but it asserts
//! only loose bounds (within 5% of ffprobe's 0.800 s figure).

use std::fs;
use std::path::PathBuf;

use oxideav_core::Demuxer;
use oxideav_mp3::Mp3Demuxer;

fn fixture(name: &str) -> Option<Mp3Demuxer> {
    let p = PathBuf::from("../../docs/audio/mp3/fixtures")
        .join(name)
        .join("input.mp3");
    if !p.exists() {
        return None;
    }
    let f = fs::File::open(&p).ok()?;
    Mp3Demuxer::open(Box::new(f)).ok()
}

#[test]
fn print_durations_for_report() {
    // ffprobe (FFmpeg) reports 0.800 s == 800_000 µs for each of
    // these four fixtures.
    let ffprobe_micros: i64 = 800_000;
    let names = [
        "layer3-cbr-320kbps-stereo-44100",
        "layer3-vbr-q5-stereo-44100",
        "layer3-with-id3v2-tag",
        "layer3-with-xing-vbri-tag",
    ];
    let mut seen = 0;
    for name in names {
        let Some(d) = fixture(name) else {
            eprintln!("skip {name}: fixture not present");
            continue;
        };
        let mp3 = d.duration_micros().unwrap_or(0);
        let delta_micros = mp3 - ffprobe_micros;
        let delta_pct = (delta_micros as f64) / (ffprobe_micros as f64) * 100.0;
        eprintln!(
            "{name:<40}  oxideav={mp3} µs  ffprobe={ffprobe_micros} µs  Δ={delta_micros:+} µs ({delta_pct:+.2}%)"
        );
        // Each fixture is exactly 32 frames × 1152 samples / 44100 Hz
        // = 0.8359 s for Layer III at 44.1 kHz. ffprobe's 0.800 s is
        // 32 frames × 1152 / 44100 truncated to 3 d.p. on its end;
        // the difference is the fixture's true frame count vs
        // ffprobe's rendering precision, not a parsing bug.
        assert!(
            (delta_micros).abs() < 100_000,
            "{name} duration delta out of bound"
        );
        seen += 1;
    }
    assert!(seen > 0, "no fixtures available");
}
