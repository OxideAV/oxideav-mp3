//! End-to-end: an MP3 file with an ID3v2.3 tag (title + artist + APIC)
//! in front of a minimal MPEG-1 Layer III frame. Opens through the
//! container registry and asserts metadata + attached_pictures flow
//! through to the demuxer's API.
//!
//! The MPEG frame here is a synthesised 128 kbps / 44.1 kHz / mono
//! header + zeroed body — the decoder isn't exercised, just the
//! container's first-frame probe.

use std::io::Cursor;

use oxideav_core::ContainerRegistry;
use oxideav_core::PictureType;

/// Build a minimal ID3v2.3 tag carrying TIT2, TPE1, and an APIC.
fn build_id3v23_tag() -> Vec<u8> {
    // TIT2 "Song"
    let mut tit2 = Vec::new();
    let title_payload = [&[0u8][..], b"Song"].concat();
    tit2.extend_from_slice(b"TIT2");
    tit2.extend_from_slice(&(title_payload.len() as u32).to_be_bytes());
    tit2.extend_from_slice(&[0, 0]);
    tit2.extend_from_slice(&title_payload);

    // TPE1 "Artist"
    let mut tpe1 = Vec::new();
    let artist_payload = [&[0u8][..], b"Artist"].concat();
    tpe1.extend_from_slice(b"TPE1");
    tpe1.extend_from_slice(&(artist_payload.len() as u32).to_be_bytes());
    tpe1.extend_from_slice(&[0, 0]);
    tpe1.extend_from_slice(&artist_payload);

    // APIC: encoding=0, mime "image/png\0", type=0x03, desc "cover\0", data.
    let mut apic_payload = Vec::new();
    apic_payload.push(0u8);
    apic_payload.extend_from_slice(b"image/png\0");
    apic_payload.push(0x03);
    apic_payload.extend_from_slice(b"cover\0");
    apic_payload.extend_from_slice(b"PNGBYTES");
    let mut apic = Vec::new();
    apic.extend_from_slice(b"APIC");
    apic.extend_from_slice(&(apic_payload.len() as u32).to_be_bytes());
    apic.extend_from_slice(&[0, 0]);
    apic.extend_from_slice(&apic_payload);

    let body = [tit2, tpe1, apic].concat();
    let size = body.len() as u32;
    let mut tag = Vec::new();
    tag.extend_from_slice(b"ID3");
    tag.push(3);
    tag.push(0);
    tag.push(0);
    tag.push(((size >> 21) & 0x7F) as u8);
    tag.push(((size >> 14) & 0x7F) as u8);
    tag.push(((size >> 7) & 0x7F) as u8);
    tag.push((size & 0x7F) as u8);
    tag.extend_from_slice(&body);
    tag
}

/// Build a 128 kbps / 44.1 kHz / stereo MPEG-1 Layer III frame with a
/// zero-filled body. Frame length is exactly 417 bytes.
fn build_mp3_frame() -> Vec<u8> {
    let mut f = vec![0u8; 417];
    f[0] = 0xFF;
    f[1] = 0xFB;
    f[2] = 0x90;
    f[3] = 0x00;
    f
}

/// Build a 128 kbps / 44.1 kHz / stereo MPEG-1 Layer III frame carrying
/// an "Info" tag (Xing layout, all four flags set) followed by the
/// optional extension whose first 9 bytes are the ASCII encoder version
/// string. For MPEG-1 stereo the tag magic sits at frame offset 36
/// (see `docs/audio/mp3/mp3-fixtures-and-traces.md` §7.2).
fn build_mp3_frame_with_info_tag(encoder: &[u8; 9]) -> Vec<u8> {
    let mut f = build_mp3_frame();
    let off = 36;
    f[off..off + 4].copy_from_slice(b"Info");
    // flags = 0x0000000f (frames | bytes | toc | quality).
    f[off + 4..off + 8].copy_from_slice(&0x0000_000f_u32.to_be_bytes());
    // frames = 2, bytes = 834 (two 417-byte frames), TOC all-zero,
    // quality = 0. Layout: 4 + 4 + 4(frames) + 4(bytes) + 100(toc)
    // + 4(quality) = 120 bytes, then the extension.
    f[off + 8..off + 12].copy_from_slice(&2u32.to_be_bytes());
    f[off + 12..off + 16].copy_from_slice(&834u32.to_be_bytes());
    // toc (100 bytes) left zeroed; quality (4 bytes) left zeroed.
    let ext = off + 120;
    f[ext..ext + 9].copy_from_slice(encoder);
    f
}

#[test]
fn mp3_metadata_and_pictures_flow_through() {
    let mut file = Vec::new();
    file.extend_from_slice(&build_id3v23_tag());
    file.extend_from_slice(&build_mp3_frame());
    file.extend_from_slice(&build_mp3_frame());

    let mut reg = ContainerRegistry::new();
    oxideav_mp3::register_containers(&mut reg);

    let cursor: Box<dyn oxideav_core::ReadSeek> = Box::new(Cursor::new(file));
    let demuxer = reg
        .open_demuxer("mp3", cursor, &oxideav_core::NullCodecResolver)
        .expect("open mp3 demuxer");

    let md = demuxer.metadata();
    assert!(
        md.iter().any(|(k, v)| k == "title" && v == "Song"),
        "title not in metadata: {:?}",
        md
    );
    assert!(
        md.iter().any(|(k, v)| k == "artist" && v == "Artist"),
        "artist not in metadata: {:?}",
        md
    );

    let pics = demuxer.attached_pictures();
    assert_eq!(pics.len(), 1);
    assert_eq!(pics[0].mime_type, "image/png");
    assert_eq!(pics[0].picture_type, PictureType::FrontCover);
    assert_eq!(pics[0].description, "cover");
    assert_eq!(pics[0].data, b"PNGBYTES");
}

#[test]
fn xing_info_encoder_version_flows_to_metadata() {
    // First frame carries an Info tag + extension; the encoder version
    // string ("Lavc61.19") must surface as the "encoder" metadata key.
    let mut file = Vec::new();
    file.extend_from_slice(&build_mp3_frame_with_info_tag(b"Lavc61.19"));
    file.extend_from_slice(&build_mp3_frame());
    file.extend_from_slice(&build_mp3_frame());

    let mut reg = ContainerRegistry::new();
    oxideav_mp3::register_containers(&mut reg);

    let cursor: Box<dyn oxideav_core::ReadSeek> = Box::new(Cursor::new(file));
    let demuxer = reg
        .open_demuxer("mp3", cursor, &oxideav_core::NullCodecResolver)
        .expect("open mp3 demuxer");

    let md = demuxer.metadata();
    assert!(
        md.iter().any(|(k, v)| k == "encoder" && v == "Lavc61.19"),
        "encoder not surfaced from Info extension: {:?}",
        md
    );
}

#[test]
fn xing_info_without_extension_yields_no_encoder() {
    // An Info tag whose extension area is all-zero (no version string)
    // must NOT invent an "encoder" key.
    let mut file = Vec::new();
    file.extend_from_slice(&build_mp3_frame_with_info_tag(&[0u8; 9]));
    file.extend_from_slice(&build_mp3_frame());

    let mut reg = ContainerRegistry::new();
    oxideav_mp3::register_containers(&mut reg);

    let cursor: Box<dyn oxideav_core::ReadSeek> = Box::new(Cursor::new(file));
    let demuxer = reg
        .open_demuxer("mp3", cursor, &oxideav_core::NullCodecResolver)
        .expect("open mp3 demuxer");

    assert!(
        !demuxer.metadata().iter().any(|(k, _)| k == "encoder"),
        "encoder key must be absent when the extension carries no version string"
    );
}

#[test]
fn fixture_encoder_version_surfaces() {
    // Real corpus fixture: the demuxer must surface an "encoder" key.
    // These FFmpeg-built fixtures carry BOTH an ID3v2 TSSE frame
    // ("Lavf61.7.100", the muxer) and a Xing extension version
    // ("Lavc61.19", the audio encoder); the ID3 frame wins by the
    // precedence rule, so we only assert the key is present and a
    // plausible non-empty encoder string. Skipped silently in a
    // standalone checkout that lacks the workspace docs corpus.
    use std::path::PathBuf;
    let p = PathBuf::from("../../docs/audio/mp3/fixtures/layer3-stereo-44100-128kbps/input.mp3");
    if !p.exists() {
        eprintln!("skip: fixture not present");
        return;
    }
    let file = std::fs::read(&p).expect("read fixture");
    let mut reg = ContainerRegistry::new();
    oxideav_mp3::register_containers(&mut reg);
    let cursor: Box<dyn oxideav_core::ReadSeek> = Box::new(Cursor::new(file));
    let demuxer = reg
        .open_demuxer("mp3", cursor, &oxideav_core::NullCodecResolver)
        .expect("open mp3 demuxer");
    let md = demuxer.metadata();
    let enc = md.iter().find(|(k, _)| k == "encoder");
    assert!(enc.is_some(), "encoder key missing for fixture: {:?}", md);
    let v = &enc.unwrap().1;
    assert!(
        v.starts_with("Lav") || v.starts_with("LAME"),
        "unexpected encoder string: {v:?}"
    );
}
