//! The demuxer announces the codec of the layer a file carries, on
//! files written by black-box encoders.
//!
//! `ffmpeg` writes Layer II (`.mp2`) at the MPEG-1 and the lower
//! (MPEG-2 LSF) sampling frequencies, mono and stereo, and `lame`
//! writes Layer III; each file is opened through the container
//! registry the way a player does (probe on the leading bytes + the
//! extension), and the codec id, codec tag, packet count and duration
//! are checked against what `ffprobe` reports for the same file. The
//! test skips when the tools are absent; nothing is committed.
//!
//! No black-box tool on the test machines writes Layer I: Layer I is
//! covered by the hand-built streams of the demuxer's unit tests.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use oxideav_core::{CodecTag, Demuxer, Error, ProbeData, RuntimeContext};
use oxideav_mp3::{Layer, open_file_demuxer, parse_header, probe};

fn have(tool: &str) -> bool {
    Command::new(tool)
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok()
}

fn have_lame() -> bool {
    Command::new("lame")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok()
}

struct TempDir(PathBuf);

impl TempDir {
    fn new(what: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "oxideav-mp3-layer-labels-{what}-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        Self(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn run(cmd: &mut Command) -> bool {
    cmd.stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

/// What ffprobe reports for `path`: (codec name, packet count,
/// Σ packet durations in samples, sampling frequency, channels).
fn ffprobe(path: &Path) -> Option<(String, u64, u64, u32, u16)> {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-count_packets",
            "-show_entries",
            "stream=codec_name,sample_rate,channels,nb_read_packets,time_base",
            "-of",
            "default=noprint_wrappers=1",
        ])
        .arg(path)
        .output()
        .ok()?;
    let text = String::from_utf8(out.stdout).ok()?;
    let field = |k: &str| {
        text.lines()
            .find_map(|l| l.strip_prefix(k)?.strip_prefix('='))
            .map(str::to_string)
    };
    let rate: u32 = field("sample_rate")?.parse().ok()?;
    let packets: u64 = field("nb_read_packets")?.parse().ok()?;
    // Packet durations, in the reader's stream time base `1 / den`.
    let den: u64 = field("time_base")?.strip_prefix("1/")?.parse().ok()?;
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "packet=duration",
            "-of",
            "csv=p=0",
        ])
        .arg(path)
        .output()
        .ok()?;
    let ticks: u64 = String::from_utf8(out.stdout)
        .ok()?
        .lines()
        .filter_map(|l| l.trim().trim_end_matches(',').parse::<u64>().ok())
        .sum();
    let samples = u128::from(ticks) * u128::from(rate);
    if samples % u128::from(den) != 0 {
        return None;
    }
    Some((
        field("codec_name")?,
        packets,
        (samples / u128::from(den)) as u64,
        rate,
        field("channels")?.parse().ok()?,
    ))
}

/// Open `path` the way a player does and check it against ffprobe.
fn check(path: &Path, id: &str, tag: u16, layer: Layer, samples_per_frame: u64) {
    let what = path.file_name().unwrap().to_string_lossy().into_owned();
    let (codec, packets, samples, rate, channels) = ffprobe(path).expect("ffprobe output");
    assert_eq!(codec, id, "{what}: the black-box reader's codec");

    // Probe on the leading bytes, with and without the extension.
    let bytes = std::fs::read(path).unwrap();
    let ext = path.extension().and_then(|e| e.to_str());
    assert_eq!(probe(&ProbeData { buf: &bytes, ext }), 100, "{what}");
    assert!(
        probe(&ProbeData {
            buf: &bytes,
            ext: None
        }) >= 75,
        "{what}"
    );

    // Through the registry.
    let mut ctx = RuntimeContext::new();
    oxideav_mp3::register(&mut ctx);
    let file = std::fs::File::open(path).unwrap();
    let mut d = ctx
        .containers
        .open_demuxer("mp3", Box::new(file), &ctx.codecs)
        .expect("registry open");
    let st = d.streams()[0].clone();
    assert_eq!(st.params.codec_id.as_str(), id, "{what}: codec id");
    assert_eq!(
        st.params.tag,
        Some(CodecTag::wave_format(tag)),
        "{what}: codec tag"
    );
    assert_eq!(st.params.sample_rate, Some(rate), "{what}");
    assert_eq!(st.params.channels, Some(channels), "{what}");
    assert_eq!(st.time_base.0.den, i64::from(rate), "{what}");

    let mut count = 0u64;
    let mut total = 0u64;
    loop {
        match d.next_packet() {
            Ok(p) => {
                let h = parse_header(&p.data).expect("packet starts with a header");
                assert_eq!(h.layer, layer, "{what}: packet {count} layer");
                assert_eq!(h.frame_len(), Some(p.data.len()), "{what}: one frame");
                assert_eq!(p.pts, Some(total as i64), "{what}: packet {count} pts");
                assert_eq!(p.duration, Some(samples_per_frame as i64), "{what}");
                if layer != Layer::LayerIII {
                    assert!(p.flags.keyframe, "{what}: packet {count} keyframe");
                }
                count += 1;
                total += p.duration.unwrap() as u64;
            }
            Err(Error::Eof) => break,
            Err(e) => panic!("{what}: next_packet: {e}"),
        }
    }
    assert_eq!(count, packets, "{what}: frame count vs ffprobe");
    assert_eq!(total, samples, "{what}: Σ durations vs ffprobe");
    assert_eq!(st.duration, Some(total as i64), "{what}: declared duration");
    assert_eq!(count * samples_per_frame, total, "{what}");

    // Direct API agrees, and a seek lands on a frame boundary.
    let mut direct = open_file_demuxer(path).expect("direct open");
    assert_eq!(direct.layer(), layer, "{what}");
    assert_eq!(direct.layer_change_offset(), None, "{what}");
    assert_eq!(direct.audio_frame_count(), packets, "{what}");
    let target = (total / 2) as i64;
    let landed = direct.seek_to(0, target).expect("seek");
    assert!(landed <= target && target - landed < samples_per_frame as i64);
    assert_eq!(landed % samples_per_frame as i64, 0, "{what}");
    assert_eq!(direct.next_packet().unwrap().pts, Some(landed), "{what}");
}

#[test]
fn layer_two_files_are_announced_as_mp2() {
    if !have("ffmpeg") || !have("ffprobe") {
        eprintln!("skip: ffmpeg / ffprobe not available");
        return;
    }
    let tmp = TempDir::new("l2");
    let mut cells = 0;
    // MPEG-1 and MPEG-2 LSF sampling frequencies; bit rates each
    // version / channel mode allows.
    for (rate, kbps_mono, kbps_stereo) in [
        (48_000u32, [64u32, 96, 192], [128u32, 192, 384]),
        (44_100, [56, 96, 160], [112, 192, 320]),
        (32_000, [32, 80, 192], [96, 160, 256]),
        (24_000, [8, 64, 160], [32, 64, 160]),
        (22_050, [16, 48, 128], [24, 64, 144]),
        (16_000, [24, 56, 112], [40, 80, 160]),
    ] {
        for (channels, rates) in [(1u16, kbps_mono), (2, kbps_stereo)] {
            for kbps in rates {
                let path = tmp.0.join(format!("l2_{rate}_{channels}_{kbps}.mp2"));
                let ok = run(Command::new("ffmpeg")
                    .args(["-y", "-v", "error", "-f", "lavfi", "-i"])
                    .arg(format!("sine=f=440:d=1.3:r={rate}"))
                    .args(["-ac", &channels.to_string(), "-ar", &rate.to_string()])
                    .args(["-c:a", "mp2", "-b:a", &format!("{kbps}k")])
                    .arg(&path));
                if !ok {
                    eprintln!("skip: ffmpeg could not write {}", path.display());
                    continue;
                }
                check(&path, "mp2", 0x0050, Layer::LayerII, 1152);
                cells += 1;
            }
        }
    }
    eprintln!("layer II: {cells} of 36 files checked");
    assert!(cells >= 30, "the encoder wrote too few of the matrix cells");
}

/// The same check on Layer III files (no information frame, so the
/// black-box reader's packet count is the audio frame count).
#[test]
fn layer_three_files_are_announced_as_mp3() {
    if !have("ffmpeg") || !have("ffprobe") || !have_lame() {
        eprintln!("skip: ffmpeg / ffprobe / lame not available");
        return;
    }
    let tmp = TempDir::new("l3");
    for (rate, spf, kbps) in [
        (44_100u32, 1152u64, "128"),
        (22_050, 576, "64"),
        (8_000, 576, "32"),
    ] {
        for mode in ["m", "j"] {
            let wav = tmp.0.join(format!("in_{rate}_{mode}.wav"));
            let channels = if mode == "m" { "1" } else { "2" };
            assert!(run(Command::new("ffmpeg")
                .args(["-y", "-v", "error", "-f", "lavfi", "-i"])
                .arg(format!("sine=f=440:d=1.3:r={rate}"))
                .args(["-ac", channels, "-c:a", "pcm_s16le"])
                .arg(&wav)));
            let path = tmp.0.join(format!("l3_{rate}_{mode}.mp3"));
            assert!(run(Command::new("lame")
                .args(["--quiet", "-t", "-m", mode, "-b", kbps])
                .args(["--resample", &format!("{}", f64::from(rate) / 1000.0)])
                .arg(&wav)
                .arg(&path)));
            check(&path, "mp3", 0x0055, Layer::LayerIII, spf);
        }
    }
}
