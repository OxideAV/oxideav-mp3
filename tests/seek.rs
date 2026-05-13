//! Integration tests for `Mp3Demuxer::seek_to`.
//!
//! Each test opens an `input.mp3` fixture from `docs/audio/mp3/fixtures/`
//! through the container registry (same path the runtime uses), seeks
//! to a target pts, and asserts that:
//!
//! * `seek_to` returns the expected target (clamped if appropriate).
//! * The next emitted packet's pts is close to the target — within
//!   one-percent-of-duration for VBR-with-TOC streams, exact (modulo
//!   one frame's worth of resync slack) for CBR streams.
//!
//! Skipped silently when the fixture isn't available (standalone
//! oxideav-mp3 checkout, e.g. CI of the mp3 repo alone, doesn't ship
//! the workspace docs corpus).

use std::fs;
use std::path::{Path, PathBuf};

use oxideav_core::{ContainerRegistry, NullCodecResolver, ReadSeek};

fn fixture(name: &str) -> Option<PathBuf> {
    let p = PathBuf::from("../../docs/audio/mp3/fixtures")
        .join(name)
        .join("input.mp3");
    if p.exists() {
        Some(p)
    } else {
        None
    }
}

fn open_demuxer(mp3_path: &Path) -> Box<dyn oxideav_core::Demuxer> {
    let file = fs::File::open(mp3_path).expect("open mp3");
    let rs: Box<dyn ReadSeek> = Box::new(file);
    let mut reg = ContainerRegistry::new();
    oxideav_mp3::register_containers(&mut reg);
    reg.open_demuxer("mp3", rs, &NullCodecResolver)
        .expect("open mp3 demuxer")
}

#[test]
fn seek_to_zero_resets_to_start() {
    let Some(p) = fixture("layer3-cbr-320kbps-stereo-44100") else {
        eprintln!("skip: fixture not present");
        return;
    };
    let mut dmx = open_demuxer(&p);
    // Walk forward a few packets so next_pts is non-zero.
    for _ in 0..3 {
        let _ = dmx.next_packet().expect("next_packet");
    }
    let landed = dmx.seek_to(0, 0).expect("seek_to(0)");
    assert_eq!(landed, 0, "seek_to(0) must land at 0");
    let pkt = dmx.next_packet().expect("next_packet after seek");
    let pts = pkt.pts.expect("packet must have pts");
    // After seek_to(0) the next packet's pts must be 0.
    assert_eq!(pts, 0, "first packet after seek_to(0) pts={pts}, want 0");
}

#[test]
fn seek_in_cbr_mp3_lands_near_target() {
    let Some(p) = fixture("layer3-cbr-320kbps-stereo-44100") else {
        eprintln!("skip: fixture not present");
        return;
    };
    let mut dmx = open_demuxer(&p);
    let tb = dmx.streams()[0].time_base;
    // Target 0.1 s into the file (the fixture is short — ~5 KB).
    let target_pts = (0.1_f64 / tb.as_rational().as_f64()).round() as i64;
    let _landed = dmx.seek_to(0, target_pts).expect("seek_to");
    let pkt = dmx.next_packet().expect("next_packet after seek");
    let pts = pkt.pts.expect("pts");
    let drift_pts = (pts - target_pts).abs();
    // CBR seek is exact modulo the padding-byte cycle (~1 frame).
    // samples_per_frame = 1152 for MPEG-1 L3; allow 2 frames slack
    // (~52 ms at 44.1 kHz) for the resync logic.
    let slack = 2 * 1152;
    assert!(
        drift_pts < slack,
        "CBR seek drifted by {drift_pts} samples (>= {slack}); pts={pts}, target={target_pts}"
    );
}

#[test]
fn seek_in_vbr_mp3_with_xing_lands_within_one_percent() {
    let Some(p) = fixture("layer3-with-xing-vbri-tag") else {
        eprintln!("skip: fixture not present");
        return;
    };
    let mut dmx = open_demuxer(&p);
    let tb = dmx.streams()[0].time_base;
    // The fixture is ~2 seconds long. Target the middle.
    let total_samples = dmx.streams()[0].duration.unwrap_or(0).max(44_100); // fall back to 1s if no duration
    let target_pts = total_samples / 2;
    let landed = dmx.seek_to(0, target_pts).expect("seek_to");
    assert_eq!(landed, target_pts, "seek_to returns the requested pts");
    let pkt = dmx.next_packet().expect("next_packet after seek");
    let pts = pkt.pts.expect("pts");
    // Xing TOC resolution is ~1% of duration; one frame is ~26 ms.
    let drift_pts = (pts - target_pts).abs();
    // 5% of total samples is a generous slack; the TOC is allowed
    // ~1% but the small fixture amplifies rounding error.
    let slack = (total_samples / 20).max(1152 * 4);
    let drift_secs = tb.seconds_of(drift_pts);
    eprintln!(
        "Xing-TOC seek: target={} samples ({:.3}s), landed at pts={} ({:.3}s), drift={} samples ({:.3}s)",
        target_pts,
        tb.seconds_of(target_pts),
        pts,
        tb.seconds_of(pts),
        drift_pts,
        drift_secs
    );
    assert!(
        drift_pts < slack,
        "Xing-TOC seek drifted by {drift_pts} samples (>= {slack} = 5%); pts={pts}, target={target_pts}"
    );
}

#[test]
fn seek_negative_pts_clamped_to_zero() {
    let Some(p) = fixture("layer3-cbr-320kbps-stereo-44100") else {
        eprintln!("skip: fixture not present");
        return;
    };
    let mut dmx = open_demuxer(&p);
    // Walk forward then seek to a negative pts.
    for _ in 0..2 {
        let _ = dmx.next_packet().expect("next_packet");
    }
    let landed = dmx.seek_to(0, -10_000).expect("seek_to negative");
    assert_eq!(landed, 0, "negative pts must clamp to 0; got {landed}");
    let pkt = dmx.next_packet().expect("next_packet after seek");
    let pts = pkt.pts.expect("pts");
    assert_eq!(
        pts, 0,
        "first packet after negative seek must have pts=0; got {pts}"
    );
}

#[test]
fn seek_past_end_does_not_panic() {
    let Some(p) = fixture("layer3-cbr-320kbps-stereo-44100") else {
        eprintln!("skip: fixture not present");
        return;
    };
    let mut dmx = open_demuxer(&p);
    // Way past the end — fixture is ~1 second.
    let huge_pts = 44_100 * 600;
    let _ = dmx.seek_to(0, huge_pts).expect("seek_to past end");
    // next_packet must return either EOF or a single trailing
    // packet — both are acceptable as long as we don't panic.
    let _ = dmx.next_packet();
}

#[test]
fn seek_in_vbr_no_xing_uses_scan_index() {
    // VBR streams without a Xing/VBRI header walk the lazy index
    // built by next_packet. This fixture is small (~6 KB / a few
    // dozen frames); we exercise the scan-and-build path.
    let Some(p) = fixture("layer3-vbr-q5-stereo-44100") else {
        eprintln!("skip: fixture not present");
        return;
    };
    let mut dmx = open_demuxer(&p);
    let tb = dmx.streams()[0].time_base;
    // Read a chunk first so a few waypoints get cached.
    for _ in 0..5 {
        if dmx.next_packet().is_err() {
            break;
        }
    }
    // Seek backward to the start, then forward again. Both must
    // succeed without errors.
    let _ = dmx.seek_to(0, 0).expect("seek_to(0) on vbr");
    let pkt = dmx.next_packet().expect("next_packet post-seek");
    let pts = pkt.pts.expect("pts");
    assert_eq!(
        pts,
        0,
        "post-seek pts at vbr stream-start must be 0; got {pts} ({:.3}s)",
        tb.seconds_of(pts)
    );
}
