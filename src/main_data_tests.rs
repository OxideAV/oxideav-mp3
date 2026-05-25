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
            MpegVersion::Mpeg2 => {
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
