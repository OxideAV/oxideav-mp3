//! Integration test for the round 161 **auto block-type WITH mixed
//! promotion** encode path
//! ([`oxideav_mp3::Mp3Encoder::enable_auto_block_type_with_mixed`]).
//!
//! Extends the round-156 auto block-type tests with the §2.4.3.4.10.3
//! mixed-block carve-out: on every granule the §C.1.5.2 scheduler
//! emits as Short, the
//! [`oxideav_mp3::MixedClassifier`]'s PCM-domain low-band stability
//! check decides whether to promote to mixed (block_type 2 +
//! mixed_block_flag = 1: lowest 2 subbands long, the rest short) or
//! keep it pure-short (mixed_block_flag = 0).
//!
//! The test:
//!
//! 1. Sanity-checks the new API surface: mono-only restriction
//!    inherited from `enable_auto_block_type`, mutual exclusion with
//!    the force-toggles, the configured threshold is readable back.
//! 2. Confirms that on a click-train stimulus with a low-frequency
//!    DC carrier (low-band stable, transient is broadband on top of
//!    the DC), some granules carry the mixed flag and the resulting
//!    bytestream is structurally valid.
//! 3. Confirms the §2.4.2.7 invariants on every mixed granule:
//!    `block_type == Short`, `mixed_block_flag == true`, `preflag ==
//!    false`.
//! 4. Confirms `Mp3Demuxer` round-trips the encoder's bytestream.
//! 5. Cross-confirms that the pure-auto path (no mixed classifier)
//!    on the same PCM emits **zero** mixed granules — the r161 mixed
//!    promotion is the only path that can produce them from the
//!    scheduler.
//!
//! The test uses ONLY this crate; no external library is invoked.

use std::io::Cursor;

use oxideav_core::Demuxer;
use oxideav_mp3::{
    parse_header, parse_side_info, BlockType, ChannelMode, FrameWalker, Mp3Demuxer, Mp3Encoder,
    DEFAULT_ATTACK_THRESHOLD, DEFAULT_MIXED_LOW_BAND_STABILITY,
};

const SR: u32 = 44_100;
const BR: u32 = 192;

/// Synthesise a "low-band-DC + high-frequency click-train" stimulus.
/// The DC carrier component is a slowly-varying sub-100 Hz tone (so
/// the low band of every granule is stationary across its three
/// subframes — the mixed classifier's stability condition) and the
/// click overlay is a 64-sample full-scale rectangular burst whose
/// alternating ±polarity puts the energy at the Nyquist frequency
/// (which the one-tap LP filter rejects, keeping the low-band
/// energy ratio low). Together the input is what the mixed-block
/// carve-out is designed for.
fn lf_dc_with_hf_click_train(total_samples: usize, click_period: usize) -> Vec<i16> {
    use std::f32::consts::PI;
    let two_pi = 2.0 * PI;
    let mut out = vec![0i16; total_samples];
    let sr_f = SR as f32;
    // 50 Hz DC carrier at modest amplitude — well below the
    // ~330 Hz boundary of the mixed-block's lowest two subbands.
    let carrier_freq = 50.0f32;
    let carrier_amp = (i16::MAX as f32) * 0.05;
    for (i, slot) in out.iter_mut().enumerate() {
        let t = i as f32 / sr_f;
        *slot = (carrier_amp * (two_pi * carrier_freq * t).sin())
            .round()
            .clamp(i16::MIN as f32, i16::MAX as f32) as i16;
    }
    // High-frequency click overlay: 64-sample full-scale ±-alternation
    // bursts. The bursts sit at Nyquist, so the LP filter cancels them
    // (low-band ratio stays near 1.0 even during the burst), but the
    // total subframe energy spikes massively → attack detector fires.
    let mut pos = click_period;
    while pos + 64 < total_samples {
        for j in 0..64 {
            // Add ±30_000 to the existing carrier; saturate at i16.
            let hf = if j % 2 == 0 { 30_000i32 } else { -30_000i32 };
            let v = i32::from(out[pos + j]).saturating_add(hf);
            out[pos + j] = v.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
        }
        pos += click_period;
    }
    out
}

#[test]
fn enable_auto_block_type_with_mixed_accepted_on_ms_stereo() {
    // r163: MS-stereo now accepts the mixed-auto toggle. The
    // per-channel mixed-classifier vector mirrors the per-channel
    // detector vector; on MS-stereo the per-granule classifier flags
    // are OR-folded together before stepping the shared scheduler,
    // so the §2.4.3.4.9 agreement holds over `mixed_block_flag` too.
    let mut enc = Mp3Encoder::new_joint_stereo_ms(BR, SR).expect("MS-stereo encoder build");
    assert!(!enc.auto_block_type_enabled());
    assert!(!enc.auto_block_type_mixed_enabled());
    enc.enable_auto_block_type_with_mixed(
        DEFAULT_ATTACK_THRESHOLD,
        DEFAULT_MIXED_LOW_BAND_STABILITY,
    )
    .expect("mixed-auto accepted on MS-stereo (r163)");
    assert!(enc.auto_block_type_enabled());
    assert!(enc.auto_block_type_mixed_enabled());
}

#[test]
fn enable_auto_block_type_with_mixed_accepted_on_ms_auto_stereo() {
    let mut enc = Mp3Encoder::new_joint_stereo_auto(BR, SR).expect("MS-auto encoder build");
    enc.enable_auto_block_type_with_mixed(
        DEFAULT_ATTACK_THRESHOLD,
        DEFAULT_MIXED_LOW_BAND_STABILITY,
    )
    .expect("mixed-auto accepted on MS-auto (r163)");
    assert!(enc.auto_block_type_enabled());
    assert!(enc.auto_block_type_mixed_enabled());
}

#[test]
fn enable_auto_block_type_with_mixed_accepted_on_independent_stereo() {
    // r162: independent stereo accepts the mixed-auto toggle (one
    // detector + scheduler + classifier per channel, all independent).
    let mut enc = Mp3Encoder::new(BR, SR, ChannelMode::Stereo).expect("stereo encoder build");
    enc.enable_auto_block_type_with_mixed(
        DEFAULT_ATTACK_THRESHOLD,
        DEFAULT_MIXED_LOW_BAND_STABILITY,
    )
    .expect("mixed-auto accepted on independent stereo");
    assert!(enc.auto_block_type_enabled());
    assert!(enc.auto_block_type_mixed_enabled());
    assert_eq!(
        enc.auto_block_type_threshold(),
        Some(DEFAULT_ATTACK_THRESHOLD)
    );
    assert_eq!(
        enc.auto_block_type_mixed_threshold(),
        Some(DEFAULT_MIXED_LOW_BAND_STABILITY)
    );

    let mut enc2 =
        Mp3Encoder::new(BR, SR, ChannelMode::DualChannel).expect("dual-channel encoder build");
    enc2.enable_auto_block_type_with_mixed(
        DEFAULT_ATTACK_THRESHOLD,
        DEFAULT_MIXED_LOW_BAND_STABILITY,
    )
    .expect("mixed-auto accepted on dual-channel");
    assert!(enc2.auto_block_type_mixed_enabled());
}

#[test]
fn enable_auto_block_type_with_mixed_thresholds_round_trip() {
    let mut enc = Mp3Encoder::new(BR, SR, ChannelMode::SingleChannel).expect("mono encoder build");
    assert!(enc.auto_block_type_threshold().is_none());
    assert!(enc.auto_block_type_mixed_threshold().is_none());
    enc.enable_auto_block_type_with_mixed(
        DEFAULT_ATTACK_THRESHOLD,
        DEFAULT_MIXED_LOW_BAND_STABILITY,
    )
    .expect("enable mixed-auto");
    assert!(enc.auto_block_type_enabled());
    assert!(enc.auto_block_type_mixed_enabled());
    assert_eq!(
        enc.auto_block_type_threshold(),
        Some(DEFAULT_ATTACK_THRESHOLD)
    );
    assert_eq!(
        enc.auto_block_type_mixed_threshold(),
        Some(DEFAULT_MIXED_LOW_BAND_STABILITY)
    );
}

#[test]
fn enable_auto_block_type_with_mixed_clears_force_toggles() {
    let mut enc = Mp3Encoder::new(BR, SR, ChannelMode::SingleChannel).expect("mono encoder build");
    enc.force_short_blocks_for_testing(true)
        .expect("force short toggle accepted on mono encoder");
    assert!(enc.force_short_blocks_enabled());
    enc.enable_auto_block_type_with_mixed(
        DEFAULT_ATTACK_THRESHOLD,
        DEFAULT_MIXED_LOW_BAND_STABILITY,
    )
    .expect("enable mixed-auto");
    assert!(!enc.force_short_blocks_enabled());
    assert!(!enc.force_mixed_blocks_enabled());
    assert!(enc.auto_block_type_enabled());
    assert!(enc.auto_block_type_mixed_enabled());
}

#[test]
fn plain_auto_path_keeps_mixed_threshold_none() {
    // The plain `enable_auto_block_type` must NOT enable mixed
    // promotion — the r161 extension is strictly opt-in.
    let mut enc = Mp3Encoder::new(BR, SR, ChannelMode::SingleChannel).expect("mono encoder build");
    enc.enable_auto_block_type(DEFAULT_ATTACK_THRESHOLD)
        .expect("enable plain auto");
    assert!(enc.auto_block_type_enabled());
    assert!(!enc.auto_block_type_mixed_enabled());
    assert!(enc.auto_block_type_mixed_threshold().is_none());
}

/// On a low-band-DC + Nyquist-click stimulus, the mixed-auto path
/// emits at least one mixed granule (block_type Short with
/// mixed_block_flag = 1). The same stimulus through the plain auto
/// path emits zero mixed granules.
#[test]
fn mixed_auto_engages_mixed_block_on_low_band_dc_with_hf_clicks() {
    let n = SR as usize; // 1 s
    let pcm = lf_dc_with_hf_click_train(n, /*click_period=*/ 6600);

    // Plain auto: expect zero mixed emissions.
    let mut enc_plain =
        Mp3Encoder::new(BR, SR, ChannelMode::SingleChannel).expect("mono encoder build");
    enc_plain
        .enable_auto_block_type(DEFAULT_ATTACK_THRESHOLD)
        .expect("enable plain auto");
    enc_plain.push_samples(&pcm).expect("push pcm plain");
    let mut bytes_plain = Vec::new();
    let _ = enc_plain
        .finish(&mut bytes_plain)
        .expect("encoder finish plain");

    let mut plain_mixed = 0usize;
    for frame in FrameWalker::new(&bytes_plain) {
        let hdr = parse_header(&frame.data[..4]).expect("header plain");
        let si = parse_side_info(&hdr, &frame.data[4..]).expect("side_info plain");
        for gr in 0..si.granule_count as usize {
            for ch in 0..si.channels as usize {
                let gc = &si.granules[gr][ch];
                if gc.window_switching_flag
                    && gc.block_type == BlockType::Short
                    && gc.mixed_block_flag
                {
                    plain_mixed += 1;
                }
            }
        }
    }
    assert_eq!(
        plain_mixed, 0,
        "plain auto path emitted {plain_mixed} mixed granules — r161 mixed promotion is opt-in"
    );

    // Mixed-auto: expect at least one mixed emission. We use a
    // *relaxed* low-band stability threshold (8.0 instead of the
    // default 4.0) so the Nyquist-click bursts — which the LP filter
    // mostly but not perfectly rejects, and which sit on top of a
    // 50 Hz DC that has its own subframe-energy modulation — still
    // pass the stability check.
    let mut enc_mixed =
        Mp3Encoder::new(BR, SR, ChannelMode::SingleChannel).expect("mono encoder build");
    enc_mixed
        .enable_auto_block_type_with_mixed(DEFAULT_ATTACK_THRESHOLD, 8.0)
        .expect("enable mixed-auto");
    enc_mixed.push_samples(&pcm).expect("push pcm mixed");
    let mut bytes_mixed = Vec::new();
    let _ = enc_mixed
        .finish(&mut bytes_mixed)
        .expect("encoder finish mixed");

    let mut mixed_count = 0usize;
    let mut short_count = 0usize;
    let mut total_frames = 0usize;
    for frame in FrameWalker::new(&bytes_mixed) {
        total_frames += 1;
        let hdr = parse_header(&frame.data[..4]).expect("header mixed");
        let si = parse_side_info(&hdr, &frame.data[4..]).expect("side_info mixed");
        for gr in 0..si.granule_count as usize {
            for ch in 0..si.channels as usize {
                let gc = &si.granules[gr][ch];
                if gc.window_switching_flag && gc.block_type == BlockType::Short {
                    short_count += 1;
                    if gc.mixed_block_flag {
                        mixed_count += 1;
                        // §2.4.2.7 invariants on a mixed granule.
                        assert!(
                            !gc.preflag,
                            "mixed granule emitted with preflag set (§2.4.2.7)"
                        );
                    }
                }
                // 3-bit field check on subblock_gain (every block
                // type — the wire field is the same).
                for &sg in &gc.subblock_gain {
                    assert!(sg <= 7, "subblock_gain {sg} exceeds 3-bit field");
                }
            }
        }
    }
    assert!(total_frames > 0, "no frames in mixed-auto stream");
    assert!(
        short_count > 0,
        "click-train mixed-auto did not engage Short geometry at all"
    );
    assert!(
        mixed_count > 0,
        "mixed-auto did not promote any Short granule to mixed \
         (short={short_count}, expected at least 1 mixed)"
    );

    // Mp3Demuxer end-to-end check on the mixed-auto bytestream.
    let cursor = Cursor::new(bytes_mixed);
    let mut demux = Mp3Demuxer::open(Box::new(cursor)).expect("demuxer open mixed-auto");
    let mut packets = 0usize;
    loop {
        match demux.next_packet() {
            Ok(_pkt) => {
                packets += 1;
                if packets > 4_000 {
                    break;
                }
            }
            Err(oxideav_core::Error::Eof) => break,
            Err(e) => panic!("demux next_packet on mixed-auto: {e}"),
        }
    }
    assert!(
        packets > 0,
        "demuxer accepted no packets from mixed-auto stream"
    );
}

/// On a steady pure sine (no transient at all), neither plain auto
/// nor mixed-auto should emit any Short / mixed granule — the
/// scheduler stays in Long throughout.
#[test]
fn mixed_auto_on_pure_sine_stays_long() {
    use std::f32::consts::PI;
    let n = SR as usize / 2; // 500 ms
    let two_pi = 2.0 * PI;
    let sr_f = SR as f32;
    let amp = (i16::MAX as f32) * 0.5;
    let mut pcm = vec![0i16; n];
    for (i, slot) in pcm.iter_mut().enumerate() {
        let t = i as f32 / sr_f;
        *slot = (amp * (two_pi * 440.0 * t).sin())
            .round()
            .clamp(i16::MIN as f32, i16::MAX as f32) as i16;
    }

    let mut enc = Mp3Encoder::new(BR, SR, ChannelMode::SingleChannel).expect("mono encoder build");
    enc.enable_auto_block_type_with_mixed(
        DEFAULT_ATTACK_THRESHOLD,
        DEFAULT_MIXED_LOW_BAND_STABILITY,
    )
    .expect("enable mixed-auto");
    enc.push_samples(&pcm).expect("push pcm");
    let mut bytes = Vec::new();
    let _ = enc.finish(&mut bytes).expect("encoder finish");

    let mut frames = 0usize;
    let mut any_short_or_mixed = false;
    for frame in FrameWalker::new(&bytes) {
        frames += 1;
        let hdr = parse_header(&frame.data[..4]).expect("header");
        let si = parse_side_info(&hdr, &frame.data[4..]).expect("side_info");
        for gr in 0..si.granule_count as usize {
            for ch in 0..si.channels as usize {
                let gc = &si.granules[gr][ch];
                if gc.window_switching_flag || gc.block_type != BlockType::Long {
                    any_short_or_mixed = true;
                }
            }
        }
    }
    assert!(frames > 0, "no frames in pure-sine mixed-auto stream");
    assert!(
        !any_short_or_mixed,
        "mixed-auto on pure sine emitted non-Long granules \
         (the §C.1.5.2 detector should classify a steady tone as non-attack)"
    );
}

/// `enable_auto_block_type_with_mixed` combined with
/// `new_with_outer_loop` engages the §C.1.5.4.3 outer loop on every
/// emitted block type — including mixed (the
/// `outer_loop_search_mixed` primitive already wired in r159 was
/// previously unreachable from the auto path).
#[test]
fn mixed_auto_combines_with_outer_loop_and_roundtrips() {
    let mut enc = Mp3Encoder::new_with_outer_loop(
        BR,
        SR,
        ChannelMode::SingleChannel,
        /*uniform_threshold=*/ 1.0e-6,
    )
    .expect("outer-loop encoder build");
    enc.enable_auto_block_type_with_mixed(DEFAULT_ATTACK_THRESHOLD, 8.0)
        .expect("enable mixed-auto + outer-loop combination");
    assert!(enc.auto_block_type_enabled());
    assert!(enc.auto_block_type_mixed_enabled());

    let n = SR as usize;
    let pcm = lf_dc_with_hf_click_train(n, /*click_period=*/ 6600);
    enc.push_samples(&pcm).expect("push pcm");
    let mut bytes = Vec::new();
    let _ = enc.finish(&mut bytes).expect("encoder finish");

    let mut frames = 0usize;
    let mut saw_mixed = false;
    for frame in FrameWalker::new(&bytes) {
        frames += 1;
        let hdr = parse_header(&frame.data[..4]).expect("header");
        let si = parse_side_info(&hdr, &frame.data[4..]).expect("side_info");
        for gr in 0..si.granule_count as usize {
            for ch in 0..si.channels as usize {
                let gc = &si.granules[gr][ch];
                if gc.window_switching_flag && gc.block_type == BlockType::Short {
                    if gc.mixed_block_flag {
                        saw_mixed = true;
                        // Mixed granule under the outer loop must
                        // carry the §C.1.5.4.3 wire signature: the
                        // outer-loop seed `scalefac_compress = 15`.
                        assert_eq!(
                            gc.scalefac_compress, 15,
                            "mixed + outer-loop did not write scalefac_compress=15"
                        );
                    }
                    assert!(!gc.preflag, "short/mixed block with preflag set (§2.4.2.7)");
                }
            }
        }
    }
    assert!(frames > 0);
    assert!(
        saw_mixed,
        "mixed-auto + outer-loop did not engage any mixed granule \
         (`outer_loop_search_mixed` dispatch unexercised)"
    );

    let cursor = Cursor::new(bytes);
    let mut demux = Mp3Demuxer::open(Box::new(cursor)).expect("demuxer open");
    let mut packets = 0usize;
    loop {
        match demux.next_packet() {
            Ok(_pkt) => {
                packets += 1;
                if packets > 4_000 {
                    break;
                }
            }
            Err(oxideav_core::Error::Eof) => break,
            Err(e) => panic!("demux next_packet: {e}"),
        }
    }
    assert!(packets > 0);
}

/// **r408 — mixed bursts carry `mixed_block_flag` on the flanking
/// `Start` / `End` granules, and mixed-ness is constant within a
/// burst.**
///
/// §2.4.2.7 scopes `mixed_block_flag` to every window-switched
/// granule (block types 1 / 2 / 3): with the flag set, the two lowest
/// polyphase subbands are transformed with the normal window while
/// the remaining 30 follow the block type. The §2.4.3.4 overlap-add
/// only cancels between complementary window halves, so the low
/// subbands of a mixed burst must be normal-windowed from the `Start`
/// through every `Short` to the `End` — a `Start(mixed=0) →
/// Short(mixed=1)` pairing leaves uncancelled low-band aliasing in
/// subbands 0..2 (observed before the fix as an nrmse ≈ 3e-2..8e-2
/// low-band divergence on an independent black-box validator, while
/// validators that reproduce the stream's literal window sequence
/// agreed with our own decode — internal consistency masking a
/// non-conformant stream).
#[test]
fn mixed_burst_flags_flanking_transition_granules() {
    let n = SR as usize; // 1 s
    let pcm = lf_dc_with_hf_click_train(n, /*click_period=*/ 6600);

    let mut enc = Mp3Encoder::new(BR, SR, ChannelMode::SingleChannel).expect("mono encoder build");
    enc.enable_auto_block_type_with_mixed(DEFAULT_ATTACK_THRESHOLD, 8.0)
        .expect("enable mixed-auto");
    enc.push_samples(&pcm).expect("push pcm");
    let mut bytes = Vec::new();
    enc.finish(&mut bytes).expect("encoder finish");

    // Flatten the stream into one granule-major sequence of
    // (block_type, window_switching_flag, mixed_block_flag).
    let mut seq: Vec<(BlockType, bool, bool)> = Vec::new();
    for frame in FrameWalker::new(&bytes) {
        let hdr = parse_header(&frame.data[..4]).expect("header");
        let si = parse_side_info(&hdr, &frame.data[4..]).expect("side info");
        for gr in 0..si.granule_count as usize {
            let gc = &si.granules[gr][0];
            seq.push((gc.block_type, gc.window_switching_flag, gc.mixed_block_flag));
        }
    }
    assert!(!seq.is_empty(), "no granules");

    let mut mixed_bursts = 0usize;
    for (i, &(bt, wsf, mixed)) in seq.iter().enumerate() {
        // The flag never appears on a non-window-switched granule.
        if !wsf {
            assert!(
                !mixed,
                "granule {i}: mixed_block_flag on a non-switched Long"
            );
            continue;
        }
        match bt {
            BlockType::Short => {
                // Every neighbour of a Short inside the burst must
                // carry the same mixed-ness: predecessor is Start or
                // Short, successor is Short or End (the §C.1.5.2
                // geometry), and both share this granule's flag.
                let prev = seq[i - 1];
                assert!(
                    matches!(prev.0, BlockType::Start | BlockType::Short) && prev.1,
                    "granule {i}: Short not preceded by Start/Short"
                );
                assert_eq!(
                    prev.2, mixed,
                    "granule {i}: mixed-ness changed inside a burst \
                     (prev {:?} mixed={}, cur Short mixed={mixed})",
                    prev.0, prev.2
                );
                if let Some(&next) = seq.get(i + 1) {
                    assert!(
                        matches!(next.0, BlockType::Short | BlockType::End) && next.1,
                        "granule {i}: Short not followed by Short/End"
                    );
                    assert_eq!(
                        next.2, mixed,
                        "granule {i}: mixed-ness changed inside a burst \
                         (cur Short mixed={mixed}, next {:?} mixed={})",
                        next.0, next.2
                    );
                }
                if mixed {
                    mixed_bursts += 1;
                }
            }
            BlockType::Start | BlockType::End => {
                // A flanking transition granule's flag always equals
                // its burst's Short flag — checked from the Short side
                // above; here just confirm a flagged Start/End is
                // actually adjacent to a mixed Short.
                if mixed {
                    let adjacent = if bt == BlockType::Start {
                        seq.get(i + 1)
                    } else {
                        seq.get(i - 1)
                    };
                    if let Some(&(abt, awsf, amixed)) = adjacent {
                        assert!(
                            abt == BlockType::Short && awsf && amixed,
                            "granule {i}: {bt:?} carries mixed_block_flag but its burst \
                             Short is ({abt:?}, mixed={amixed})"
                        );
                    }
                }
            }
            BlockType::Long => {
                assert!(!mixed, "granule {i}: window-switched Long with mixed flag");
            }
        }
    }
    assert!(
        mixed_bursts > 0,
        "stimulus engaged no mixed burst — the invariant above was vacuous"
    );
}
