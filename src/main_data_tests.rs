// Round-trip tests for the Layer III main-data assembler. Each test
// builds known scalefactors + quantized is[] for one or more
// granule/channels, assembles the contiguous part2+part3 main-data block
// (main_data_begin == 0), then reads it back through the exact §2.4.1.7
// `main_data()` loop — per granule/channel: scalefactors (part2) then
// Huffmancodebits() (part3) — using the existing decode path, and
// verifies the recovered scalefactors and is[] are bit-exact and that the
// reader consumes exactly the emitted bit count.

mod tests {
    use super::*;
    use crate::encoder::make_silent_header;
    use crate::frame::ChannelMode;
    use crate::huffman::decode_huffman;
    use crate::scalefactors::{
        decode_scalefactors, read_lsf_channel, read_mpeg1_granule_channel, MainDataReader, LONG_SFB,
        SHORT_SFB, SHORT_WINDOWS,
    };
    use crate::side_info::{BlockType, GranuleChannel, SideInfo, GRANULES};

    fn default_gc() -> GranuleChannel {
        GranuleChannel {
            part2_3_length: 0,
            big_values: 0,
            global_gain: 0,
            scalefac_compress: 0,
            window_switching_flag: false,
            block_type: BlockType::Long,
            mixed_block_flag: false,
            table_select: [0; 3],
            subblock_gain: [0; 3],
            region0_count: 0,
            region1_count: 0,
            preflag: false,
            scalefac_scale: false,
            count1table_select: false,
        }
    }

    fn empty_side_info(channels: u8, granule_count: u8, lsf: bool) -> SideInfo {
        SideInfo {
            main_data_begin: 0,
            private_bits: 0,
            scfsi: [[false; 4]; 2],
            granules: [[default_gc(); 2]; GRANULES],
            channels,
            granule_count,
            lsf,
        }
    }

    fn empty_scalefactors(channels: u8, granule_count: u8) -> FrameScaleFactors {
        FrameScaleFactors {
            granules: [[ScaleFactors::default(); 2]; 2],
            granule_count,
            channels,
            part2_bits: [[0; 2]; 2],
        }
    }

    fn empty_gc_data() -> GranuleChannelData {
        GranuleChannelData {
            is: [0; NUM_LINES],
            big_pairs: 0,
            count1_quads: 0,
        }
    }

    /// Read back one granule/channel's part2 (scalefactors) + part3
    /// (Huffman) from `r`, mirroring the §2.4.1.7 `main_data()` loop, and
    /// return the recovered scalefactors and the 576-line is[].
    fn read_gc(
        r: &mut MainDataReader<'_>,
        header: &Mp3FrameHeader,
        si: &SideInfo,
        gr: usize,
        ch: usize,
        prev_sf: Option<&ScaleFactors>,
        intensity: bool,
    ) -> (ScaleFactors, [i32; NUM_LINES]) {
        let gc = &si.granules[gr][ch];
        let p2_start = r.bit_pos();
        let sf = match header.version {
            MpegVersion::Mpeg1 => {
                read_mpeg1_granule_channel(r, gc, &si.scfsi[ch], gr, prev_sf)
            }
            MpegVersion::Mpeg2 | MpegVersion::Mpeg25 => {
                let is_intensity_right = intensity && ch == 1;
                read_lsf_channel(r, gc, is_intensity_right)
            }
        };
        let part2_bits = (r.bit_pos() - p2_start) as u32;
        let part3_bits = u32::from(gc.part2_3_length) - part2_bits;
        let is = decode_huffman(r, gc, part3_bits, header.sample_rate_hz, header.version).unwrap();
        (sf, is)
    }

    // ---- MPEG-1 long block, two channels, single granule ----

    #[test]
    fn mpeg1_long_single_granule_two_channels_roundtrip() {
        let header = make_silent_header(128, 44100, ChannelMode::Stereo).unwrap();
        let mut si = empty_side_info(2, 1, false);
        let mut sf = empty_scalefactors(2, 1);
        let mut gcd = [[empty_gc_data(), empty_gc_data()], [empty_gc_data(), empty_gc_data()]];

        // Indexes three parallel arrays (si / sf / gcd) by `ch`.
        #[allow(clippy::needless_range_loop)]
        for ch in 0..2 {
            // Long block: scalefac_compress 13 => slen1=3, slen2=3 (each
            // band fits 0..=7). Fill all 21 long scalefactors with a
            // varying pattern.
            let g = &mut si.granules[0][ch];
            g.scalefac_compress = 13;
            g.big_values = 2; // two pairs => lines 0..4
            g.table_select = [1, 1, 1];
            g.region0_count = 5; // big region0 covers all big_values here
            g.region1_count = 2;
            g.count1table_select = false;

            for b in 0..LONG_SFB {
                sf.granules[0][ch].long[b] = ((b + ch) % 8) as u8;
            }

            // is[]: two big-values pairs of magnitude <= 1 (table 1 codes
            // (0,0)/(0,1)/(1,0)/(1,1)), plus one count1 quad of |v|<=1.
            let d = &mut gcd[0][ch];
            d.is[0] = 1;
            d.is[1] = if ch == 0 { -1 } else { 0 };
            d.is[2] = 0;
            d.is[3] = 1;
            d.big_pairs = 2;
            // count1 quad on lines 4..8.
            d.is[4] = 1;
            d.is[5] = 0;
            d.is[6] = -1;
            d.is[7] = 1;
            d.count1_quads = 1;
        }

        let asm = assemble_main_data(&header, &mut si, &sf, &gcd).unwrap();
        assert_eq!(asm.main_data_begin, 0);
        assert_eq!(si.main_data_begin, 0);

        // The block's payload bit count equals the sum of part2_3_length.
        let sum: usize = (0..2).map(|ch| usize::from(si.granules[0][ch].part2_3_length)).sum();
        assert_eq!(asm.total_bits, sum);

        // Read back per the main_data() loop and compare.
        let mut r = MainDataReader::new(&asm.bytes);
        // Indexes parallel arrays (sf / gcd) by `ch`.
        #[allow(clippy::needless_range_loop)]
        for ch in 0..2 {
            let (got_sf, got_is) = read_gc(&mut r, &header, &si, 0, ch, None, false);
            assert_eq!(got_sf.long, sf.granules[0][ch].long, "scalefactors ch{ch}");
            assert_eq!(&got_is[..8], &gcd[0][ch].is[..8], "is[] ch{ch}");
            for (i, &v) in got_is.iter().enumerate().skip(8) {
                assert_eq!(v, 0, "is[{i}] should be zero ch{ch}");
            }
        }
        // Consumed bit count == emitted payload length.
        assert_eq!(r.bit_pos(), asm.total_bits);
        assert!(!r.exhausted());
    }

    // ---- MPEG-1 two granules, mono, scfsi all false (no reuse) ----

    #[test]
    fn mpeg1_two_granules_mono_roundtrip() {
        let header = make_silent_header(128, 44100, ChannelMode::SingleChannel).unwrap();
        let mut si = empty_side_info(1, 2, false);
        let mut sf = empty_scalefactors(1, 2);
        let mut gcd = [[empty_gc_data(), empty_gc_data()], [empty_gc_data(), empty_gc_data()]];

        // Indexes three parallel arrays (si / sf / gcd) by `gr`.
        #[allow(clippy::needless_range_loop)]
        for gr in 0..2 {
            let g = &mut si.granules[gr][0];
            g.scalefac_compress = 13; // slen1=3, slen2=3
            g.big_values = 1;
            g.table_select = [1, 1, 1];
            g.region0_count = 5;
            g.region1_count = 2;

            for b in 0..LONG_SFB {
                sf.granules[gr][0].long[b] = ((b + gr * 2) % 8) as u8;
            }

            let d = &mut gcd[gr][0];
            d.is[0] = if gr == 0 { 1 } else { -1 };
            d.is[1] = 1;
            d.big_pairs = 1;
            d.count1_quads = 0;
        }

        let asm = assemble_main_data(&header, &mut si, &sf, &gcd).unwrap();
        assert_eq!(asm.main_data_begin, 0);

        let mut r = MainDataReader::new(&asm.bytes);
        let mut prev: Option<ScaleFactors> = None;
        // Indexes parallel arrays (sf / gcd) by `gr`.
        #[allow(clippy::needless_range_loop)]
        for gr in 0..2 {
            let (got_sf, got_is) = read_gc(&mut r, &header, &si, gr, 0, prev.as_ref(), false);
            assert_eq!(got_sf.long, sf.granules[gr][0].long, "scalefactors gr{gr}");
            assert_eq!(&got_is[..2], &gcd[gr][0].is[..2], "is[] gr{gr}");
            prev = Some(got_sf);
        }
        assert_eq!(r.bit_pos(), asm.total_bits);
    }

    // ---- Whole-frame cross-check via decode_scalefactors ----
    //
    // decode_scalefactors reads every granule/channel's part2 fields
    // back-to-back. Because part3 sits *between* granules in the real
    // layout, this only lines up for a single granule/channel (gr=ch=0),
    // where there is no following part3 before the next part2. This test
    // confirms the assembler's first part2 block is byte-for-bit what
    // decode_scalefactors expects for that one granule/channel.

    #[test]
    fn first_granule_part2_matches_decode_scalefactors() {
        let header = make_silent_header(128, 44100, ChannelMode::SingleChannel).unwrap();
        let mut si = empty_side_info(1, 1, false);
        let mut sf = empty_scalefactors(1, 1);
        let mut gcd = [[empty_gc_data(), empty_gc_data()], [empty_gc_data(), empty_gc_data()]];

        let g = &mut si.granules[0][0];
        g.scalefac_compress = 13;
        g.big_values = 0; // no Huffman big values
        g.table_select = [0, 0, 0];
        for b in 0..LONG_SFB {
            sf.granules[0][0].long[b] = (b % 8) as u8;
        }
        gcd[0][0].big_pairs = 0;
        gcd[0][0].count1_quads = 0;

        let asm = assemble_main_data(&header, &mut si, &sf, &gcd).unwrap();
        let fsf = decode_scalefactors(&header, &si, &asm.bytes).unwrap();
        assert_eq!(fsf.granules[0][0].long, sf.granules[0][0].long);
    }

    // ---- LSF (MPEG-2) long block round-trip ----

    #[test]
    fn lsf_long_single_granule_roundtrip() {
        let header = make_silent_header(64, 24000, ChannelMode::SingleChannel).unwrap();
        let mut si = empty_side_info(1, 1, true);
        let mut sf = empty_scalefactors(1, 1);
        let mut gcd = [[empty_gc_data(), empty_gc_data()], [empty_gc_data(), empty_gc_data()]];

        // scalefac_compress < 400 long block => slen via lsf_scale_params.
        let g = &mut si.granules[0][0];
        g.scalefac_compress = 200; // < 400 path
        g.big_values = 1;
        g.table_select = [1, 1, 1];
        g.region0_count = 5;
        g.region1_count = 2;

        let params = crate::scalefactors::lsf_scale_params(200, BlockType::Long, false, false);
        // Fill the long scalefactors that the four LSF partitions cover,
        // keeping each value within its partition's slen width.
        let mut idx = 0usize;
        for p in 0..4 {
            let maxv = if params.slen[p] == 0 {
                0
            } else {
                (1u8 << params.slen[p]) - 1
            };
            for _ in 0..params.nr_of_sfb[p] {
                if idx < LONG_SFB {
                    sf.granules[0][0].long[idx] = (idx as u8).min(maxv);
                }
                idx += 1;
            }
        }

        let d = &mut gcd[0][0];
        d.is[0] = 1;
        d.is[1] = -1;
        d.big_pairs = 1;

        let asm = assemble_main_data(&header, &mut si, &sf, &gcd).unwrap();
        assert_eq!(asm.main_data_begin, 0);

        let mut r = MainDataReader::new(&asm.bytes);
        let (got_sf, got_is) = read_gc(&mut r, &header, &si, 0, 0, None, false);
        assert_eq!(got_sf.long, sf.granules[0][0].long, "LSF scalefactors");
        assert_eq!(&got_is[..2], &gcd[0][0].is[..2], "LSF is[]");
        assert_eq!(r.bit_pos(), asm.total_bits);
    }

    // ---- Bit-reservoir scheduling (encoder ↔ Reservoir decoder) ----

    use crate::main_data::{
        schedule_reservoir, ReservoirError, ReservoirFrame, ReservoirScheduler,
        RESERVOIR_MAX_LSF, RESERVOIR_MAX_MPEG1,
    };
    use crate::scalefactors::Reservoir;

    /// Single-frame schedule: `main_data_begin = 0`, slot byte budget
    /// exceeds main_data, tail zero-padded.
    #[test]
    fn schedule_single_quiet_frame_self_contained() {
        let md: Vec<u8> = (1..=10).collect();
        let frames = [ReservoirFrame { main_data: &md, slot_bytes: 20, lsf: false }];
        let mut sis = vec![empty_side_info(1, 1, false)];
        let out = schedule_reservoir(&frames, &mut sis).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].main_data_begin, 0);
        assert_eq!(out[0].slot.len(), 20);
        assert_eq!(&out[0].slot[..10], &md[..]);
        // Tail zero-padded (no future frame's bytes to fill it).
        assert!(out[0].slot[10..].iter().all(|&b| b == 0));
        assert_eq!(sis[0].main_data_begin, 0);
    }

    /// The §2.4.2.7 user-prompt scenario: three CBR-like frames in which
    /// the middle frame's main_data exceeds its own slot. The scheduler
    /// places the busy frame's main_data starting inside frame 0's
    /// unused tail; the middle frame's `main_data_begin` points back
    /// across that boundary; the decoder reproduces the original
    /// main_data bytes for all three frames bit-exactly.
    #[test]
    fn schedule_three_frame_busy_middle_via_prior_quiet() {
        // Plan:
        //   frame 0: slot 30, main_data 10 (quiet, leaves 20 B in tail)
        //   frame 1: slot 30, main_data 50 (busy, needs 20 B from prior
        //                                   reservoir + 30 B of own slot)
        //   frame 2: slot 30, main_data 10 (quiet, leaves 20 B in tail
        //                                   that the file pads with zeros)
        //
        // Cumulative: L = 10/60/70, S = 30/60/90.
        //   R_0 = 0,  R_1 = 30-10 = 20, R_2 = 60-60 = 0, R_3 = 90-70 = 20.
        //   main_data_begin = [0, 20, 0].
        //
        // The middle frame's main_data extends from MD offset 10 to MD
        // offset 60 — i.e. the last 20 B of slot 0 and the first 30 B of
        // slot 1.
        let md0: Vec<u8> = (1..=10).collect();
        let md1: Vec<u8> = (100..=149).collect(); // 50 bytes
        let md2: Vec<u8> = (200..=209).collect();
        let frames = [
            ReservoirFrame { main_data: &md0, slot_bytes: 30, lsf: false },
            ReservoirFrame { main_data: &md1, slot_bytes: 30, lsf: false },
            ReservoirFrame { main_data: &md2, slot_bytes: 30, lsf: false },
        ];
        let mut sis = vec![
            empty_side_info(1, 1, false),
            empty_side_info(1, 1, false),
            empty_side_info(1, 1, false),
        ];
        let scheduled = schedule_reservoir(&frames, &mut sis).unwrap();
        assert_eq!(
            [scheduled[0].main_data_begin, scheduled[1].main_data_begin, scheduled[2].main_data_begin],
            [0, 20, 0]
        );
        // side info was stamped.
        assert_eq!([sis[0].main_data_begin, sis[1].main_data_begin, sis[2].main_data_begin], [0, 20, 0]);

        // Decoder-side: feed each slot through Reservoir::assemble with
        // the corresponding main_data_begin and recover the original
        // main_data bytes bit-exactly.
        let mut decoder_res = Reservoir::new();
        let mds = [&md0, &md1, &md2];
        for (i, s) in scheduled.iter().enumerate() {
            let run = decoder_res
                .assemble(s.main_data_begin as usize, &s.slot)
                .unwrap();
            // Each frame's main_data is the first L_i bytes of the run.
            assert_eq!(
                &run[..mds[i].len()],
                &mds[i][..],
                "frame {i} main_data bit-exact recovery"
            );
        }
    }

    /// `ReservoirScheduler` stateful builder is equivalent to the
    /// one-shot `schedule_reservoir` on the same inputs.
    #[test]
    fn reservoir_scheduler_builder_matches_one_shot() {
        let md0: Vec<u8> = (1..=10).collect();
        let md1: Vec<u8> = (100..=149).collect();
        let md2: Vec<u8> = (200..=209).collect();
        let frames = [
            ReservoirFrame { main_data: &md0, slot_bytes: 30, lsf: false },
            ReservoirFrame { main_data: &md1, slot_bytes: 30, lsf: false },
            ReservoirFrame { main_data: &md2, slot_bytes: 30, lsf: false },
        ];

        // One-shot.
        let mut sis_a = vec![
            empty_side_info(1, 1, false),
            empty_side_info(1, 1, false),
            empty_side_info(1, 1, false),
        ];
        let out_a = schedule_reservoir(&frames, &mut sis_a).unwrap();

        // Builder.
        let mut sched = ReservoirScheduler::new();
        for f in &frames {
            sched.push(f);
        }
        assert_eq!(sched.len(), 3);
        assert!(!sched.is_empty());
        let mut sis_b = vec![
            empty_side_info(1, 1, false),
            empty_side_info(1, 1, false),
            empty_side_info(1, 1, false),
        ];
        let out_b = sched.finish(&mut sis_b).unwrap();

        assert_eq!(out_a, out_b);
        assert_eq!(
            sis_a.iter().map(|s| s.main_data_begin).collect::<Vec<_>>(),
            sis_b.iter().map(|s| s.main_data_begin).collect::<Vec<_>>(),
        );
    }

    /// Schedulability check: a busy frame that needs more bytes than
    /// prior reservoir + own slot is rejected as SlotUnderflow.
    #[test]
    fn schedule_busy_frame_with_no_prior_reservoir_underflows() {
        // 700-byte main_data into a 100-byte slot, no prior frame:
        // L_0 = 700 > R_0 + S_0 = 0 + 100 → SlotUnderflow.
        let big = vec![0xAAu8; 700];
        let frames = [ReservoirFrame { main_data: &big, slot_bytes: 100, lsf: false }];
        let mut sis = vec![empty_side_info(1, 1, false)];
        let err = schedule_reservoir(&frames, &mut sis).unwrap_err();
        match err {
            ReservoirError::SlotUnderflow {
                frame_index,
                main_data_len,
                slot_bytes,
                prior_reservoir,
            } => {
                assert_eq!(frame_index, 0);
                assert_eq!(main_data_len, 700);
                assert_eq!(slot_bytes, 100);
                assert_eq!(prior_reservoir, 0);
            }
            other => panic!("expected SlotUnderflow, got {other:?}"),
        }
    }

    /// Reservoir cap (511 bytes MPEG-1): a sequence whose running
    /// reservoir would exceed the on-wire 9-bit `main_data_begin` field
    /// is rejected as ReservoirOverflow. Setup: frame 0 is quiet by a
    /// lot, building >511 B of reservoir before any frame attempts to
    /// use it.
    #[test]
    fn schedule_reservoir_cap_mpeg1_511() {
        // Frame 0 emits 0 bytes of main_data but its slot is 600 B → R_1
        // = 600 > 511 cap → ReservoirOverflow at frame 1.
        let empty = vec![];
        let small: Vec<u8> = (1..=5).collect();
        let frames = [
            ReservoirFrame { main_data: &empty, slot_bytes: 600, lsf: false },
            ReservoirFrame { main_data: &small, slot_bytes: 30, lsf: false },
        ];
        let mut sis = vec![empty_side_info(1, 1, false), empty_side_info(1, 1, false)];
        let err = schedule_reservoir(&frames, &mut sis).unwrap_err();
        match err {
            ReservoirError::ReservoirOverflow { frame_index, attempted, cap } => {
                assert_eq!(frame_index, 1);
                assert_eq!(attempted, 600);
                assert_eq!(cap, 511);
            }
            other => panic!("expected ReservoirOverflow, got {other:?}"),
        }
    }

    /// LSF (MPEG-2) reservoir cap is the tighter 8-bit 255 bytes.
    #[test]
    fn schedule_reservoir_cap_lsf_255() {
        // 300-byte main_data into a 30-byte slot at LSF: L_0 = 300 > R_0
        // + S_0 = 30 → SlotUnderflow (own-frame budget exceeded).
        let md = vec![0x55u8; 300];
        let frames = [ReservoirFrame { main_data: &md, slot_bytes: 30, lsf: true }];
        let mut sis = vec![empty_side_info(1, 1, true)];
        let err = schedule_reservoir(&frames, &mut sis).unwrap_err();
        assert!(matches!(err, ReservoirError::SlotUnderflow { .. }));

        // LSF reservoir-cap path: frame 0 emits 0 B into a 300-B slot →
        // R_1 = 300 > 255 LSF cap → ReservoirOverflow at frame 1.
        let empty = vec![];
        let small = vec![0xAAu8; 5];
        let frames2 = [
            ReservoirFrame { main_data: &empty, slot_bytes: 300, lsf: true },
            ReservoirFrame { main_data: &small, slot_bytes: 30, lsf: true },
        ];
        let mut sis2 = vec![empty_side_info(1, 1, true), empty_side_info(1, 1, true)];
        let err2 = schedule_reservoir(&frames2, &mut sis2).unwrap_err();
        match err2 {
            ReservoirError::ReservoirOverflow { frame_index, attempted, cap } => {
                assert_eq!(frame_index, 1);
                assert_eq!(attempted, 300);
                assert_eq!(cap, 255);
            }
            other => panic!("expected ReservoirOverflow, got {other:?}"),
        }
    }

    /// Reservoir caps surfaced as constants.
    #[test]
    fn reservoir_cap_constants_match_spec_bit_widths() {
        assert_eq!(RESERVOIR_MAX_MPEG1, 511); // 9 bits
        assert_eq!(RESERVOIR_MAX_LSF, 255); //   8 bits
    }

    /// End-to-end three-frame round-trip across the FULL pipeline:
    /// assemble each frame's main-data through `assemble_main_data` with
    /// real per-granule scalefactor + Huffman payloads, schedule the
    /// three through `schedule_reservoir` (busy middle frame → non-zero
    /// `main_data_begin`), then decode each frame's main_data back
    /// through `Reservoir::assemble` + the existing decoder, and verify
    /// every granule/channel's scalefactors + `is[]` recover bit-exactly.
    #[test]
    fn three_frame_pipeline_round_trip_with_real_main_data() {
        // MPEG-1 mono. Each frame gets its own SideInfo + scalefactors +
        // Huffman input; the middle frame is engineered to have more
        // main-data bytes than would fit a tight slot, so its scheduled
        // main_data_begin will be > 0.
        let header = make_silent_header(128, 44100, ChannelMode::SingleChannel).unwrap();

        // Helper to build one (SideInfo, ScaleFactors, GranuleChannelData)
        // tuple with a configurable big_values count (so the middle
        // frame is fatter than the bookends).
        let build_frame = |big_pairs: usize, sf_shift: u8| {
            let mut si = empty_side_info(1, 2, false);
            let mut sf = empty_scalefactors(1, 2);
            let mut gcd = [[empty_gc_data(), empty_gc_data()], [empty_gc_data(), empty_gc_data()]];
            #[allow(clippy::needless_range_loop)]
            for gr in 0..2 {
                let g = &mut si.granules[gr][0];
                g.scalefac_compress = 13; // slen1=3, slen2=3
                g.big_values = big_pairs as u16;
                g.table_select = [1, 1, 1]; // table 1 codes magnitudes 0..=1
                g.region0_count = 5;
                g.region1_count = 2;
                g.count1table_select = false;
                for b in 0..LONG_SFB {
                    sf.granules[gr][0].long[b] = ((b + sf_shift as usize) % 8) as u8;
                }
                let d = &mut gcd[gr][0];
                #[allow(clippy::needless_range_loop)]
                for k in 0..(big_pairs * 2) {
                    d.is[k] = if k % 2 == 0 { 1 } else { -1 };
                }
                d.big_pairs = big_pairs;
                d.count1_quads = 0;
            }
            (si, sf, gcd)
        };

        // Frame 0 / 2 quiet (big_pairs=1), Frame 1 busy (big_pairs=200
        // → 400 codewords of table 1 ≈ 1200 bits ≈ 150 B per granule, ×2
        // granules ≈ 300 B of main_data alone, plus scalefactors).
        let (mut si0, sf0, gcd0) = build_frame(1, 0);
        let (mut si1, sf1, gcd1) = build_frame(200, 1);
        let (mut si2, sf2, gcd2) = build_frame(1, 2);

        let asm0 = assemble_main_data(&header, &mut si0, &sf0, &gcd0).unwrap();
        let asm1 = assemble_main_data(&header, &mut si1, &sf1, &gcd1).unwrap();
        let asm2 = assemble_main_data(&header, &mut si2, &sf2, &gcd2).unwrap();

        // Pick a slot size that lets frame 0 build enough reservoir for
        // frame 1's main_data to fit in R_1 + S_1. Frame 0's asm bytes
        // is small (~15 B), so a 200 B slot leaves R_1 ≈ 185 B; frame 1
        // needs ~300 B which exceeds R_1 + S_1 = 385 B comfortably.
        let slot_bytes = 200usize;
        assert!(asm1.bytes.len() > slot_bytes, "middle frame must overflow its own slot");
        assert!(
            asm0.bytes.len() < slot_bytes,
            "first frame must be quiet enough to build reservoir"
        );

        let frames = [
            ReservoirFrame { main_data: &asm0.bytes, slot_bytes, lsf: false },
            ReservoirFrame { main_data: &asm1.bytes, slot_bytes, lsf: false },
            ReservoirFrame { main_data: &asm2.bytes, slot_bytes, lsf: false },
        ];
        let mut sis = vec![si0.clone(), si1.clone(), si2.clone()];
        let scheduled = schedule_reservoir(&frames, &mut sis).unwrap();

        // Frame 0 starts the stream → mdb = 0.
        assert_eq!(sis[0].main_data_begin, 0);
        // Frame 1 is busy, leans on frame 0's tail → mdb > 0.
        assert!(
            sis[1].main_data_begin > 0,
            "middle frame should borrow from frame 0's unused tail"
        );

        // Decoder-side round-trip: replay slots through Reservoir, then
        // re-decode each frame's main_data through the same read_gc loop
        // used by the prior tests, and confirm bit-exact recovery.
        let mut decoder_res = Reservoir::new();
        let assembleds = [&asm0, &asm1, &asm2];
        let sis_in = [&si0, &si1, &si2];
        let sfs = [&sf0, &sf1, &sf2];
        let gcds = [&gcd0, &gcd1, &gcd2];
        for (i, s) in scheduled.iter().enumerate() {
            let run = decoder_res
                .assemble(s.main_data_begin as usize, &s.slot)
                .unwrap();
            // run >= asm[i].bytes (it carries the requested back-bytes +
            // the slot). The original main_data is the first len bytes.
            let orig_len = assembleds[i].bytes.len();
            assert!(
                run.len() >= orig_len,
                "frame {i}: run len {} < expected main_data len {}",
                run.len(),
                orig_len
            );
            assert_eq!(
                &run[..orig_len],
                &assembleds[i].bytes[..],
                "frame {i} main_data bit-exact"
            );

            // Now feed `run` through the same §2.4.1.7 main_data() loop
            // the prior tests used and confirm each granule's scalefactors
            // and is[] recover bit-exactly.
            let mut r = MainDataReader::new(&run);
            let mut prev: Option<ScaleFactors> = None;
            // Indexes parallel arrays (sfs / gcds) by `gr`.
            #[allow(clippy::needless_range_loop)]
            for gr in 0..2 {
                let (got_sf, got_is) =
                    read_gc(&mut r, &header, sis_in[i], gr, 0, prev.as_ref(), false);
                assert_eq!(
                    got_sf.long,
                    sfs[i].granules[gr][0].long,
                    "frame {i} gr{gr} scalefactors"
                );
                let big = gcds[i][gr][0].big_pairs * 2;
                assert_eq!(
                    &got_is[..big],
                    &gcds[i][gr][0].is[..big],
                    "frame {i} gr{gr} is[] big_values"
                );
                prev = Some(got_sf);
            }
        }
    }

    // ---- MPEG-1 short block round-trip ----

    #[test]
    fn mpeg1_short_block_roundtrip() {
        let header = make_silent_header(128, 44100, ChannelMode::SingleChannel).unwrap();
        let mut si = empty_side_info(1, 1, false);
        let mut sf = empty_scalefactors(1, 1);
        let mut gcd = [[empty_gc_data(), empty_gc_data()], [empty_gc_data(), empty_gc_data()]];

        let g = &mut si.granules[0][0];
        g.scalefac_compress = 5; // slen1=1, slen2=1 => values 0..=1
        g.window_switching_flag = true;
        g.block_type = BlockType::Short;
        g.mixed_block_flag = false;
        g.big_values = 1;
        g.table_select = [1, 1, 0];
        // window-switched: region0_count=8, region1_count=63 defaults; the
        // region split clamps to big_values*2 anyway.
        g.region0_count = 8;
        g.region1_count = 63;

        for sfb in 0..SHORT_SFB {
            for win in 0..SHORT_WINDOWS {
                sf.granules[0][0].short[sfb][win] = ((sfb + win) % 2) as u8;
            }
        }

        let d = &mut gcd[0][0];
        d.is[0] = 1;
        d.is[1] = 1;
        d.big_pairs = 1;

        let asm = assemble_main_data(&header, &mut si, &sf, &gcd).unwrap();
        let mut r = MainDataReader::new(&asm.bytes);
        let (got_sf, got_is) = read_gc(&mut r, &header, &si, 0, 0, None, false);
        assert_eq!(got_sf.short, sf.granules[0][0].short, "short scalefactors");
        assert_eq!(&got_is[..2], &gcd[0][0].is[..2], "short is[]");
        assert_eq!(r.bit_pos(), asm.total_bits);
    }
}
