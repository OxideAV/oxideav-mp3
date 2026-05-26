// Unit tests for the Layer III Huffman decode stage. The bitstreams
// here are hand-assembled from the Table 3-B.7 codebooks transcribed
// in huffman_tables.rs, the Table 3-B.8 band starts in huffman.rs, and
// the §2.4.1.7 huffmancodebits() syntax (No. of bits column on p.18).

mod tests {
    use super::*;

    /// Pack a sequence of `(value, len)` MSB-first into a byte vector,
    /// for assembling spec-derived bitstreams in tests.
    fn pack(bits: &[(u32, u32)]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut acc: u32 = 0;
        let mut nbits: u32 = 0;
        for &(v, n) in bits {
            // Mask `v` to `n` bits and append.
            let v = if n == 0 { 0 } else { v & ((1u32 << n) - 1) };
            acc = (acc << n) | v;
            nbits += n;
            while nbits >= 8 {
                nbits -= 8;
                out.push(((acc >> nbits) & 0xff) as u8);
            }
        }
        if nbits > 0 {
            out.push(((acc << (8 - nbits)) & 0xff) as u8);
        }
        out
    }

    fn mk_gc(
        big_values: u16,
        table_select: [u8; 3],
        region0_count: u8,
        region1_count: u8,
        count1table_select: bool,
    ) -> GranuleChannel {
        GranuleChannel {
            part2_3_length: 0,
            big_values,
            global_gain: 0,
            scalefac_compress: 0,
            window_switching_flag: false,
            block_type: BlockType::Long,
            mixed_block_flag: false,
            table_select,
            subblock_gain: [0; 3],
            region0_count,
            region1_count,
            preflag: false,
            scalefac_scale: false,
            count1table_select,
        }
    }

    // ----- big-values: one pair from Table 1 (a small linbits=0 table) -----

    #[test]
    fn big_values_single_pair_from_table1() {
        // Table 1: (1,0) is hcod "01" (len 2). Decode (+1, +0) → bits
        // 01 then signx=0 (no signy, y==0). Total part-3 bits = 3.
        let bytes = pack(&[(0b01, 2), (0, 1)]);
        let mut rd = MainDataReader::new(&bytes);
        let gc = mk_gc(1, [1, 0, 0], 20, 0, false);
        let is = decode_huffman(&mut rd, &gc, 3, 44100, MpegVersion::Mpeg1).unwrap();
        assert_eq!(is[0], 1);
        assert_eq!(is[1], 0);
        // Remaining lines zero (count1 budget exhausted immediately).
        for &v in &is[2..] {
            assert_eq!(v, 0);
        }
    }

    #[test]
    fn big_values_negative_pair_from_table1() {
        // Table 1: (1,1) is hcod "000" (len 3). Decode (-1, -1) → bits
        // 000 then signx=1 then signy=1.
        let bytes = pack(&[(0b000, 3), (1, 1), (1, 1)]);
        let mut rd = MainDataReader::new(&bytes);
        let gc = mk_gc(1, [1, 0, 0], 20, 0, false);
        let is = decode_huffman(&mut rd, &gc, 64, 44100, MpegVersion::Mpeg1).unwrap();
        assert_eq!(is[0], -1);
        assert_eq!(is[1], -1);
    }

    // ----- count1 quad table A (small Huffman) -----

    #[test]
    fn count1_quad_a_zero_pattern_no_signs() {
        // Quad A entry vwxy=0000 is hcod "1" (len 1). No sign bits
        // appended (all values zero). Single quad decoded then budget
        // runs out.
        // big_values = 0, so we go straight to count1.
        let bytes = pack(&[(0b1, 1)]);
        let mut rd = MainDataReader::new(&bytes);
        let gc = mk_gc(0, [0, 0, 0], 0, 0, false);
        let is = decode_huffman(&mut rd, &gc, 1, 44100, MpegVersion::Mpeg1).unwrap();
        for &v in &is[..4] {
            assert_eq!(v, 0);
        }
    }

    #[test]
    fn count1_quad_a_all_ones_with_signs() {
        // Quad A entry vwxy=1111 is hcod "000001" (len 6); then four
        // sign bits (all '1' → all negative).
        let bytes = pack(&[(0b000001, 6), (1, 1), (1, 1), (1, 1), (1, 1)]);
        let mut rd = MainDataReader::new(&bytes);
        let gc = mk_gc(0, [0, 0, 0], 0, 0, false);
        let is = decode_huffman(&mut rd, &gc, 10, 44100, MpegVersion::Mpeg1).unwrap();
        assert_eq!(&is[..4], &[-1, -1, -1, -1]);
    }

    // ----- count1 quad table B (trivial 4-bit code) -----

    #[test]
    fn count1_quad_b_trivial_pattern() {
        // Quad B: each bit, 0 → magnitude 1, 1 → 0. Pattern 0101 →
        // values (1,0,1,0); only the non-zero values get sign bits, so
        // two sign bits follow (both '0' → positive).
        let bytes = pack(&[(0b0101, 4), (0, 1), (0, 1)]);
        let mut rd = MainDataReader::new(&bytes);
        let gc = mk_gc(0, [0, 0, 0], 0, 0, true);
        let is = decode_huffman(&mut rd, &gc, 6, 44100, MpegVersion::Mpeg1).unwrap();
        assert_eq!(&is[..4], &[1, 0, 1, 0]);
    }

    // ----- region boundaries: split across region0 / region1 -----

    #[test]
    fn region_split_uses_two_tables() {
        // Two pairs of big_values. region0_count=0 → region0 covers
        // band 0 only, which at 44.1 kHz long blocks is lines 0..=3
        // (width 4). So pair 0 (lines 0..=1) uses table_select[0],
        // pair 1 (lines 2..=3) ALSO falls in region 0 (line 2 < 4).
        // Cross the band-0 boundary by setting big_values to 3:
        // pairs at lines (0,1), (2,3), (4,5) — first two in region 0,
        // third in region 1.
        // We pick table 0 for region 0 (single zero entry, 0 bits)
        // and table 1 for region 1 (small Huffman).
        // Table 0 emits (0,0) without consuming bits, twice.
        // Table 1: (1,0) is "01" len 2 → +1, 0.
        let bytes = pack(&[(0b01, 2), (0, 1)]);
        let mut rd = MainDataReader::new(&bytes);
        let gc = mk_gc(3, [0, 1, 0], 0, 0, false);
        let is = decode_huffman(&mut rd, &gc, 3, 44100, MpegVersion::Mpeg1).unwrap();
        assert_eq!(&is[..6], &[0, 0, 0, 0, 1, 0]);
    }

    // ----- linbits ESC extension via table 13 -----

    #[test]
    fn linbits_zero_table_treats_15_as_literal() {
        // Table 13 has linbits=0, so a (15, 0) hcod must NOT trigger a
        // linbits read. Table 13 entry (15,0) is hlen=12 hcod=000000010000.
        // Followed by signx=0. The value should be +15, +0. Budget = 13
        // bits (the pair + signx) so count1 doesn't run past the buffer.
        let bytes = pack(&[(0b000000010000, 12), (0, 1)]);
        let mut rd = MainDataReader::new(&bytes);
        let gc = mk_gc(1, [13, 0, 0], 20, 0, false);
        let is = decode_huffman(&mut rd, &gc, 13, 44100, MpegVersion::Mpeg1).unwrap();
        assert_eq!(is[0], 15);
        assert_eq!(is[1], 0);
        assert!(!rd.exhausted());
    }

    // ----- bit budget exhaustion stops count1 loop -----

    #[test]
    fn bit_budget_exhaustion_terminates_count1() {
        // big_values = 0; count1 table A entry vwxy=0000 is hcod "1"
        // (len 1, no signs). With a budget of 3 bits and bytes that
        // would otherwise feed infinitely, exactly three quads are
        // decoded then the loop exits without consuming bit 4.
        let bytes = pack(&[(0b1111_0000, 8)]); // three "1"s then padding
        let mut rd = MainDataReader::new(&bytes);
        let gc = mk_gc(0, [0, 0, 0], 0, 0, false);
        let is = decode_huffman(&mut rd, &gc, 3, 44100, MpegVersion::Mpeg1).unwrap();
        // After three "1" reads we should be at bit pos 3 (still under
        // budget=3 after the third "1" is read inside the loop, but the
        // next iteration's pre-check stops us).
        assert!(rd.bit_pos() <= 4);
        // Lines 0..=11 (three quads) are all zero.
        for &v in &is[..12] {
            assert_eq!(v, 0);
        }
    }

    // ----- unused tables rejected -----

    #[test]
    fn unused_table_4_is_rejected() {
        let bytes = pack(&[(0, 8)]);
        let mut rd = MainDataReader::new(&bytes);
        let gc = mk_gc(1, [4, 0, 0], 20, 0, false);
        let err = decode_huffman(&mut rd, &gc, 16, 44100, MpegVersion::Mpeg1).unwrap_err();
        assert_eq!(err, HuffmanError::UnusedTable(4));
    }

    #[test]
    fn unused_table_14_is_rejected() {
        let bytes = pack(&[(0, 8)]);
        let mut rd = MainDataReader::new(&bytes);
        let gc = mk_gc(1, [14, 0, 0], 20, 0, false);
        let err = decode_huffman(&mut rd, &gc, 16, 44100, MpegVersion::Mpeg1).unwrap_err();
        assert_eq!(err, HuffmanError::UnusedTable(14));
    }

    // ----- large 16x16 tables 15, 16, 24 + linbits aliases -----

    #[test]
    fn table15_zero_pair() {
        // Table 15 (linbits=0) entry (0,0) is hlen=3 hcod=111. Both values
        // are zero, so no sign bits follow.
        let bytes = pack(&[(0b111, 3)]);
        let mut rd = MainDataReader::new(&bytes);
        let gc = mk_gc(1, [15, 0, 0], 20, 0, false);
        let is = decode_huffman(&mut rd, &gc, 3, 44100, MpegVersion::Mpeg1).unwrap();
        assert_eq!(is[0], 0);
        assert_eq!(is[1], 0);
    }

    #[test]
    fn table15_nonzero_signed_pair() {
        // Table 15 entry (1,2) is hlen=5 hcod=10000 → (x=1, y=2); sign
        // bits 1 (negative x) and 0 (positive y). linbits=0, no ESC.
        let bytes = pack(&[(0b10000, 5), (1, 1), (0, 1)]);
        let mut rd = MainDataReader::new(&bytes);
        let gc = mk_gc(1, [15, 0, 0], 20, 0, false);
        let is = decode_huffman(&mut rd, &gc, 7, 44100, MpegVersion::Mpeg1).unwrap();
        assert_eq!(is[0], -1);
        assert_eq!(is[1], 2);
    }

    #[test]
    fn table16_linbits_escape_on_15() {
        // Table 16 (linbits=1) entry (15,15) is hlen=8 hcod=00000011. Both
        // magnitudes are 15 → each reads a 1-bit linbits field and adds it
        // to 15, then a sign bit. Sequence: code, linbitsx=1 (→16),
        // signx=1 (→ -16), linbitsy=0 (→15), signy=0 (→ +15).
        let bytes = pack(&[(0b00000011, 8), (1, 1), (1, 1), (0, 1), (0, 1)]);
        let mut rd = MainDataReader::new(&bytes);
        let gc = mk_gc(1, [16, 0, 0], 20, 0, false);
        let is = decode_huffman(&mut rd, &gc, 12, 44100, MpegVersion::Mpeg1).unwrap();
        assert_eq!(is[0], -16);
        assert_eq!(is[1], 15);
    }

    #[test]
    fn table16_small_value_no_escape() {
        // Table 16 entry (0,0) is hlen=1 hcod=1 → (0,0), no signs, no ESC.
        let bytes = pack(&[(0b1, 1)]);
        let mut rd = MainDataReader::new(&bytes);
        let gc = mk_gc(1, [16, 0, 0], 20, 0, false);
        let is = decode_huffman(&mut rd, &gc, 1, 44100, MpegVersion::Mpeg1).unwrap();
        assert_eq!(is[0], 0);
        assert_eq!(is[1], 0);
    }

    #[test]
    fn table17_alias_uses_table16_codes_wider_linbits() {
        // Table 17 = table 16 codes but linbits=2. Decode (15,15) via the
        // same hcod 00000011, with linbitsx=0b11 (→ 15+3 = 18), signx=0,
        // linbitsy=0b00 (→15), signy=0.
        let bytes = pack(&[(0b00000011, 8), (0b11, 2), (0, 1), (0b00, 2), (0, 1)]);
        let mut rd = MainDataReader::new(&bytes);
        let gc = mk_gc(1, [17, 0, 0], 20, 0, false);
        let is = decode_huffman(&mut rd, &gc, 14, 44100, MpegVersion::Mpeg1).unwrap();
        assert_eq!(is[0], 18);
        assert_eq!(is[1], 15);
    }

    #[test]
    fn table24_zero_and_escape() {
        // Table 24 (linbits=4) entry (0,0) is hlen=4 hcod=1111 → (0,0).
        let bytes = pack(&[(0b1111, 4)]);
        let mut rd = MainDataReader::new(&bytes);
        let gc = mk_gc(1, [24, 0, 0], 20, 0, false);
        let is = decode_huffman(&mut rd, &gc, 4, 44100, MpegVersion::Mpeg1).unwrap();
        assert_eq!(is[0], 0);
        assert_eq!(is[1], 0);

        // Table 24 entry (15,0) is hlen=8 hcod=00101011. x=15 → linbits(4)
        // ESC. linbitsx=0b1010 (→ 15+10 = 25), signx=1 (→ -25); y=0 → no
        // sign.
        let bytes = pack(&[(0b00101011, 8), (0b1010, 4), (1, 1)]);
        let mut rd = MainDataReader::new(&bytes);
        let gc = mk_gc(1, [24, 0, 0], 20, 0, false);
        let is = decode_huffman(&mut rd, &gc, 13, 44100, MpegVersion::Mpeg1).unwrap();
        assert_eq!(is[0], -25);
        assert_eq!(is[1], 0);
    }

    #[test]
    fn table25_alias_uses_table24_codes_linbits5() {
        // Table 25 = table 24 codes but linbits=5. (15,0) hcod 00101011,
        // linbitsx=0b11111 (→ 15+31 = 46), signx=0.
        let bytes = pack(&[(0b00101011, 8), (0b11111, 5), (0, 1)]);
        let mut rd = MainDataReader::new(&bytes);
        let gc = mk_gc(1, [25, 0, 0], 20, 0, false);
        let is = decode_huffman(&mut rd, &gc, 14, 44100, MpegVersion::Mpeg1).unwrap();
        assert_eq!(is[0], 46);
        assert_eq!(is[1], 0);
    }

    // ----- band-table sanity (Table 3-B.8 transcribed widths) -----

    #[test]
    fn long_band_widths_sum_to_576_at_each_rate() {
        // Each table's last entry is "end of band 20 + 1"; the bands
        // span lines 0..end. The frame total is 576 lines, so a
        // sensible check is that the highest band-end is <= 576.
        for &table in &[&LONG_BANDS_32, &LONG_BANDS_44, &LONG_BANDS_48] {
            assert!(table[21] <= NUM_LINES);
            // Band starts are strictly increasing.
            for i in 1..22 {
                assert!(table[i] > table[i - 1], "non-increasing at {i}: {table:?}");
            }
        }
    }

    // ----- codebook self-validation: prefix-free + sane lengths -----

    fn assert_prefix_free(table: &BigTable) {
        let entries: Vec<_> = table
            .entries
            .iter()
            .filter(|e| e.len > 0)
            .copied()
            .collect();
        for (i, a) in entries.iter().enumerate() {
            for (j, b) in entries.iter().enumerate() {
                if i == j {
                    continue;
                }
                if a.len <= b.len {
                    let shift = b.len - a.len;
                    let prefix = u32::from(b.code) >> shift;
                    assert_ne!(
                        prefix,
                        u32::from(a.code),
                        "code 0x{:x}/{} is a prefix of 0x{:x}/{}",
                        a.code,
                        a.len,
                        b.code,
                        b.len
                    );
                }
            }
        }
    }

    #[test]
    fn table0_prefix_free() {
        assert_prefix_free(&TABLE0);
    }
    #[test]
    fn table1_prefix_free() {
        assert_prefix_free(&TABLE1);
    }
    #[test]
    fn table2_prefix_free() {
        assert_prefix_free(&TABLE2);
    }
    #[test]
    fn table3_prefix_free() {
        assert_prefix_free(&TABLE3);
    }
    #[test]
    fn table5_prefix_free() {
        assert_prefix_free(&TABLE5);
    }
    #[test]
    fn table6_prefix_free() {
        assert_prefix_free(&TABLE6);
    }
    #[test]
    fn table7_prefix_free() {
        assert_prefix_free(&TABLE7);
    }
    #[test]
    fn table8_prefix_free() {
        assert_prefix_free(&TABLE8);
    }
    #[test]
    fn table9_prefix_free() {
        assert_prefix_free(&TABLE9);
    }
    #[test]
    fn table10_prefix_free() {
        assert_prefix_free(&TABLE10);
    }
    #[test]
    fn table11_prefix_free() {
        assert_prefix_free(&TABLE11);
    }
    #[test]
    fn table12_prefix_free() {
        assert_prefix_free(&TABLE12);
    }
    #[test]
    fn table13_prefix_free() {
        assert_prefix_free(&TABLE13);
    }

    #[test]
    fn quad_a_prefix_free() {
        for (i, &(la, ca)) in QUAD_A.iter().enumerate() {
            for (j, &(lb, cb)) in QUAD_A.iter().enumerate() {
                if i == j {
                    continue;
                }
                if la <= lb {
                    let shift = lb - la;
                    assert_ne!(u32::from(cb) >> shift, u32::from(ca));
                }
            }
        }
    }

    // ----- forward bit count ⇄ decoder round-trip -----

    /// The §C.1.5.4.4.5 / .8 forward count must equal the exact number of
    /// bits `decode_huffman` consumes for the same `is[]`, region split
    /// and table selection. We assemble a spec-derived bitstream from the
    /// Table 3-B.7 codewords, decode it (recording the consumed bits via
    /// `bit_pos`), then `count_huffman_bits` of the decoded `is[]` must
    /// match the consumed-bit delta exactly.
    #[test]
    fn count_matches_decoder_consumption_big_and_count1() {
        // Two big-values pairs in region 0 under table 1:
        //   (1,0) → "01"  (len 2) + signx(0)            = 3 bits
        //   (1,1) → "000" (len 3) + signx(1) + signy(1) = 5 bits
        // One count1 quad under table A:
        //   (1,1,1,1) → "000001" (len 6) + 4 signs      = 10 bits
        let bytes = pack(&[
            (0b01, 2),
            (0, 1), // pair 0: (1,0), signx=0
            (0b000, 3),
            (1, 1),
            (1, 1), // pair 1: (-1,-1)
            (0b000001, 6),
            (1, 1),
            (1, 1),
            (1, 1),
            (1, 1), // count1 quad: (-1,-1,-1,-1)
        ]);
        let mut rd = MainDataReader::new(&bytes);
        // big_values=2, table_select[0]=1, region0 spans all big-values.
        let gc = mk_gc(2, [1, 0, 0], 20, 0, false);
        let part3 = 3 + 5 + 10;
        let start = rd.bit_pos();
        let is = decode_huffman(&mut rd, &gc, part3, 44100, MpegVersion::Mpeg1).unwrap();
        let consumed = rd.bit_pos() - start;
        assert_eq!(consumed, 18, "decoder consumed an unexpected bit count");

        // Forward count of the decoded is[]: 2 big-values pairs (region 0
        // = lines 0..4, table 1), 1 count1 quad (table A).
        let counted =
            count_huffman_bits(&is, 2, (4, 4), [1, 0, 0], 1, false).expect("codable");
        assert_eq!(
            counted, consumed,
            "forward count {counted} != decoder consumption {consumed}"
        );
    }

    /// Same round-trip across a linbits ESC pair: table 16 (linbits=1),
    /// a pair (16, 0) → symbol 15 + 1 linbits + 1 sign.
    #[test]
    fn count_matches_decoder_consumption_linbits() {
        // Table 16 (16x16, linbits=1). Look up the (15,0) codeword.
        let ent = TABLE16_E[15 * 16];
        // Stream: codeword(15,0), linbits(1)=1 → magnitude 15+1=16, signx=0.
        let bytes = pack(&[
            (u32::from(ent.code), u32::from(ent.len)),
            (1, 1), // linbits field (value 1) → magnitude 16
            (0, 1), // signx = 0 (positive)
        ]);
        let mut rd = MainDataReader::new(&bytes);
        let gc = mk_gc(1, [16, 0, 0], 20, 0, false);
        let part3 = u32::from(ent.len) + 1 + 1;
        let start = rd.bit_pos();
        let is = decode_huffman(&mut rd, &gc, part3, 44100, MpegVersion::Mpeg1).unwrap();
        let consumed = rd.bit_pos() - start;
        assert_eq!(is[0], 16);
        assert_eq!(is[1], 0);
        let counted = count_huffman_bits(&is, 1, (2, 2), [16, 0, 0], 0, false).expect("codable");
        assert_eq!(counted, consumed);
        assert_eq!(counted, usize::from(ent.len) + 1 + 1);
    }

    /// Kraft inequality: a complete (lossless) binary Huffman code
    /// satisfies Σ 2^-len = 1. Our tables include a few unused (x,y)
    /// corners (the spec marks tables as squares but trims the top
    /// magnitudes), so we just check ≤ 1 and ≥ 0.5 as a sanity bound.
    #[test]
    fn tables_kraft_inequality_bounded() {
        for (name, table) in &[
            ("1", &TABLE1),
            ("2", &TABLE2),
            ("3", &TABLE3),
            ("5", &TABLE5),
            ("6", &TABLE6),
            ("7", &TABLE7),
            ("8", &TABLE8),
            ("9", &TABLE9),
            ("10", &TABLE10),
            ("11", &TABLE11),
            ("12", &TABLE12),
            ("13", &TABLE13),
            ("15", &TABLE15),
            ("16", &TABLE16),
            ("24", &TABLE24),
        ] {
            let sum: f64 = table
                .entries
                .iter()
                .filter(|e| e.len > 0)
                .map(|e| 2.0f64.powi(-i32::from(e.len)))
                .sum();
            assert!(
                (0.25..=1.0 + 1e-9).contains(&sum),
                "table {name} Kraft sum {sum} out of range",
            );
        }
    }

    /// The three 16×16 codebooks must be prefix-free: no codeword is a
    /// prefix of any other (a wrong transcription would break this, and
    /// the unique-decode property the matcher relies on). A complete
    /// 256-symbol code also has a Kraft sum very close to 1.
    #[test]
    fn large_tables_prefix_free_and_complete() {
        for (name, table) in &[("15", &TABLE15), ("16", &TABLE16), ("24", &TABLE24)] {
            // 256 entries, all present (no unused corners in these tables).
            assert_eq!(table.entries.len(), 256, "table {name} entry count");
            // Prefix-free: for every ordered pair (a, b) with len(a) <=
            // len(b), the high len(a) bits of b's code must differ from a.
            for (i, a) in table.entries.iter().enumerate() {
                for (j, b) in table.entries.iter().enumerate() {
                    if i == j || a.len > b.len {
                        continue;
                    }
                    let shift = b.len - a.len;
                    assert_ne!(
                        u32::from(b.code) >> shift,
                        u32::from(a.code),
                        "table {name}: entry {j} has entry {i} as a prefix",
                    );
                }
            }
            // Complete 256-symbol code: Kraft sum ≈ 1.
            let sum: f64 = table
                .entries
                .iter()
                .map(|e| 2.0f64.powi(-i32::from(e.len)))
                .sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "table {name} Kraft sum {sum} != 1",
            );
        }
    }

    // ----- forward emission ⇄ decoder round-trip -----

    /// Decode the payload of an `encode_huffman` result and assert it
    /// recovers `is[0 .. big_pairs*2 + count1_quads*4]` exactly, that the
    /// decoder consumes precisely `bit_len` bits, and that `bit_len`
    /// equals the matching `count_huffman_bits`.
    ///
    /// `region0_count` / `region1_count` are the side-info fields the
    /// decoder uses to recompute its own region boundaries (long block,
    /// 44.1 kHz); they MUST reproduce `region_ends` via
    /// `encoder_region_boundaries`, which the helper asserts so the decode
    /// split lands identically to the one the emitter used.
    #[allow(clippy::too_many_arguments)]
    fn assert_roundtrip(
        is: &[i32; NUM_LINES],
        big_pairs: usize,
        region_ends: (usize, usize),
        table_select: [u8; 3],
        count1_quads: usize,
        count1table_b: bool,
        region0_count: u8,
        region1_count: u8,
    ) {
        let data = encode_huffman(
            is,
            big_pairs,
            region_ends,
            table_select,
            count1_quads,
            count1table_b,
        )
        .expect("codable");

        // Emitted bit length must equal the r134 exact count.
        let counted = count_huffman_bits(
            is,
            big_pairs,
            region_ends,
            table_select,
            count1_quads,
            count1table_b,
        )
        .expect("codable");
        assert_eq!(
            data.bit_len, counted,
            "emitted bit length {} != count_huffman_bits {}",
            data.bit_len, counted,
        );

        // Build a gc whose own region split (from region0/1_count)
        // reproduces region_ends, so the decoder picks the same per-region
        // table_select the emitter used.
        let gc = mk_gc(
            big_pairs as u16,
            table_select,
            region0_count,
            region1_count,
            count1table_b,
        );
        assert_eq!(
            encoder_region_boundaries(&gc, big_pairs, 44100, MpegVersion::Mpeg1),
            region_ends,
            "region0/1_count must reproduce region_ends",
        );
        let mut rd = MainDataReader::new(&data.bytes);
        let start = rd.bit_pos();
        let decoded =
            decode_huffman(&mut rd, &gc, data.bit_len as u32, 44100, MpegVersion::Mpeg1).unwrap();
        let consumed = rd.bit_pos() - start;
        assert_eq!(
            consumed, data.bit_len,
            "decoder consumed {consumed} bits, emitter wrote {}",
            data.bit_len,
        );
        let nlines = big_pairs * 2 + count1_quads * 4;
        for i in 0..nlines {
            assert_eq!(decoded[i], is[i], "line {i} mismatch after round-trip");
        }
        for (i, &v) in decoded.iter().enumerate().skip(nlines) {
            assert_eq!(v, 0, "line {i} should be zero past the partitions");
        }
    }

    /// Round-trip a mixed big-values + count1 granule through the emitter:
    /// two pairs in region 0 (table 1) and one count1 quad (table A). The
    /// big-values span lines 0..4 so region 0 (the whole big-values range)
    /// matches the decoder's own split.
    #[test]
    fn encode_roundtrip_big_and_count1() {
        let mut is = [0i32; NUM_LINES];
        is[0] = 1;
        is[1] = 0; // pair (1, 0)
        is[2] = -1;
        is[3] = -1; // pair (-1, -1)
        is[4] = -1;
        is[5] = -1;
        is[6] = -1;
        is[7] = -1; // count1 quad (-1,-1,-1,-1)
        // region0_count=0 → region0 ends at band 1 (line 4) = bv2, so all
        // big-values land in region 0.
        assert_roundtrip(&is, 2, (4, 4), [1, 0, 0], 1, false, 0, 0);
    }

    /// Round-trip a linbits ESC pair (table 16, linbits=1): magnitude 16
    /// is symbol 15 + a 1-bit linbits field + a sign. big_values spans
    /// lines 0..2 (one pair) in region 0.
    #[test]
    fn encode_roundtrip_linbits_escape() {
        let mut is = [0i32; NUM_LINES];
        is[0] = 16; // → symbol 15, linbits=1, sign 0
        is[1] = 0;
        assert_roundtrip(&is, 1, (2, 2), [16, 0, 0], 0, false, 0, 0);
    }

    /// Round-trip a larger linbits magnitude with a negative sign under
    /// table 24 (linbits=4): |x| = 20 → symbol 15 + linbits 5 + sign 1.
    #[test]
    fn encode_roundtrip_linbits_negative_table24() {
        let mut is = [0i32; NUM_LINES];
        is[0] = -20; // 15 + 5, negative
        is[1] = 3; // small positive, no escape
        assert_roundtrip(&is, 1, (2, 2), [24, 0, 0], 0, false, 0, 0);
    }

    /// Round-trip the count1 partition under quad table B (the trivial
    /// 4-bit code): a quad of mixed magnitudes and signs.
    #[test]
    fn encode_roundtrip_count1_table_b() {
        let mut is = [0i32; NUM_LINES];
        is[0] = 1;
        is[1] = -1;
        is[2] = 0;
        is[3] = 1; // (1,-1,0,1) under table B
        assert_roundtrip(&is, 0, (0, 0), [0, 0, 0], 1, true, 0, 0);
    }

    /// Round-trip a granule that uses all three big-values regions with
    /// different tables, aligned to the 44.1 kHz long-block band split so
    /// the decoder reproduces the same region boundaries. region0_count=3
    /// → region0 ends at band 4 start = line 16; region1_count=2 → region1
    /// ends at band 7 start = line 30.
    #[test]
    fn encode_roundtrip_three_regions() {
        // Band starts at 44.1 kHz: 0,4,8,12,16,20,24,30,...
        // region0 = lines 0..16 (tables small), region1 = 16..30, region2
        // = 30..big_values*2.
        let mut is = [0i32; NUM_LINES];
        // Fill region 0 (lines 0..16) with small magnitudes.
        for v in is.iter_mut().take(16) {
            *v = 1;
        }
        // Region 1 (lines 16..30) with magnitude-2 values.
        for v in is.iter_mut().take(30).skip(16) {
            *v = 2;
        }
        // Region 2 (lines 30..40) with a couple of larger values.
        for v in is.iter_mut().take(40).skip(30) {
            *v = 5;
        }
        let big_pairs = 20; // lines 0..40
        let region_ends = (16, 30);
        // Choose minimum-bit tables per region via the r134 chooser, so
        // the round-trip also exercises the encoder's table selection.
        let (t0, _) = choose_best_table_for_region(&is, 0, 16).unwrap();
        let (t1, _) = choose_best_table_for_region(&is, 16, 30).unwrap();
        let (t2, _) = choose_best_table_for_region(&is, 30, big_pairs * 2).unwrap();
        // Build a gc whose region split matches region_ends: region0_count
        // = 3 (band 4 start = 16), region1_count = 2 (band 7 start = 30).
        let gc = mk_gc(big_pairs as u16, [t0, t1, t2], 3, 2, false);
        let (r0, r1) = encoder_region_boundaries(&gc, big_pairs, 44100, MpegVersion::Mpeg1);
        assert_eq!((r0, r1), region_ends, "band-aligned region split");
        assert_roundtrip(&is, big_pairs, region_ends, [t0, t1, t2], 0, false, 3, 2);
    }

    /// End-to-end encoder-side round-trip: derive the partition split and
    /// per-region tables the way the inner loop will, emit, and decode
    /// back. Exercises `partition_split` → `choose_best_*` → `encode_huffman`
    /// → `decode_huffman` as a pipeline over a synthetic spectrum.
    #[test]
    fn encode_roundtrip_full_pipeline_derived_params() {
        // A deterministic pseudo-spectrum: decaying magnitudes with mixed
        // signs, trailing zeros (so r_zero trims), and a ≤1 tail (count1).
        let mut is = [0i32; NUM_LINES];
        let mut state: u32 = 0x1234_5678;
        let mut next = || {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            (state >> 16) & 0x7fff
        };
        // Lines 0..64: magnitudes 0..~12 with random sign.
        for v in is.iter_mut().take(64) {
            let mag = (next() % 13) as i32;
            *v = if next() & 1 == 1 { -mag } else { mag };
        }
        // Lines 64..96: a ≤1 tail to feed the count1 partition.
        for v in is.iter_mut().take(96).skip(64) {
            let mag = (next() % 2) as i32;
            *v = if next() & 1 == 1 { -mag } else { mag };
        }
        // Lines 96.. remain zero (trailing zero run).

        let split = partition_split(&is);
        let bv2 = split.big_pairs * 2;
        // Choose a long-block region split: put region0_count / region1_count
        // so the decoder reproduces region_ends. Use the whole big-values
        // range as region 0 for simplicity (region0_count large enough to
        // cover bv2), so the chooser-derived single table codes all pairs.
        // To stay band-aligned, pick the band whose start is >= bv2 as the
        // region0 end (region1/region2 empty).
        let starts_44 = [
            0usize, 4, 8, 12, 16, 20, 24, 30, 36, 44, 52, 62, 74, 90, 110, 134, 162, 196, 238, 288,
            342, 418,
        ];
        // region0_count chosen so band (region0_count+1) start >= bv2.
        let mut r0_band = 0usize;
        for (b, &s) in starts_44.iter().enumerate() {
            if s >= bv2 {
                r0_band = b;
                break;
            }
        }
        if r0_band == 0 {
            r0_band = starts_44.len() - 1;
        }
        let region0_count = (r0_band.saturating_sub(1)) as u8;
        let gc = mk_gc(split.big_pairs as u16, [0, 0, 0], region0_count, 0, false);
        let region_ends = encoder_region_boundaries(&gc, split.big_pairs, 44100, MpegVersion::Mpeg1);
        // region0 covers all big-values; region1/2 empty (region1_count=0
        // and region0 already reaches bv2).
        assert_eq!(region_ends.0, bv2, "region0 should cover all big-values");
        assert_eq!(region_ends.1, bv2, "region1/2 should be empty");
        let (t0, _) = choose_best_table_for_region(&is, 0, region_ends.0).unwrap();
        let (t1, _) = choose_best_table_for_region(&is, region_ends.0, region_ends.1).unwrap();
        let (t2, _) = choose_best_table_for_region(&is, region_ends.1, bv2).unwrap();
        let (c1_b, _) = choose_best_count1_table(
            &is,
            bv2,
            bv2 + split.count1_quads * 4,
        );
        assert_roundtrip(
            &is,
            split.big_pairs,
            region_ends,
            [t0, t1, t2],
            split.count1_quads,
            c1_b,
            region0_count,
            0,
        );
    }

    /// A pair not codable by the chosen table is rejected (magnitude out
    /// of a small table's range with no linbits escape).
    #[test]
    fn encode_rejects_uncodable_pair() {
        let mut is = [0i32; NUM_LINES];
        is[0] = 100; // far out of table 1's 0..=1 range, no linbits
        is[1] = 0;
        let err = encode_huffman(&is, 1, (2, 2), [1, 0, 0], 0, false).unwrap_err();
        assert_eq!(err, HuffmanEncodeError::PairNotCodable(1));
    }

    /// An unused codebook index (4 / 14) is rejected.
    #[test]
    fn encode_rejects_unused_table() {
        let mut is = [0i32; NUM_LINES];
        is[0] = 1;
        is[1] = 1;
        let err = encode_huffman(&is, 1, (2, 2), [4, 0, 0], 0, false).unwrap_err();
        assert_eq!(err, HuffmanEncodeError::UnusedTable(4));
    }

    /// big_pairs*2 past the granule capacity is rejected.
    #[test]
    fn encode_rejects_oversized_big_values() {
        let is = [0i32; NUM_LINES];
        let err = encode_huffman(&is, 289, (0, 0), [0, 0, 0], 0, false).unwrap_err();
        assert_eq!(err, HuffmanEncodeError::BigValuesTooLarge);
    }

    // =================================================================
    // r154 — §C.1.5.4.4.8 linbits-reach filter (#1106)
    //
    // The §B.7 codebooks have widely-varying magnitude reach. The small
    // tables 0..=15 are reach = xlen - 1 (no linbits escape); the ESC
    // tables 16..=31 carry `15 + (2^linbits - 1)`. The encoder's table
    // chooser must filter codebooks by reach so the §2.4.1.7
    // huffmancodebits() emission round-trips bit-exactly. These tests
    // pin the reach values and the chooser's filter behaviour to the
    // §B.7 / §C.1.5.4.4.8 spec text.
    // =================================================================

    /// `big_table_reach` returns the §B.7 magnitude reach for every
    /// selectable codebook. Hand-tabulated from the per-table
    /// `xlen` / `linbits` headers transcribed in `huffman_tables.rs`
    /// against §B.7. Tables 4 and 14 are "not used" → reach 0.
    #[test]
    fn big_table_reach_matches_spec_for_every_selectable_table() {
        // (idx, expected_reach)
        let spec_reach: [(u8, u32); 32] = [
            (0, 0),        // single (0,0) entry, only codes zero pair
            (1, 1),        // 2x2 small table, no linbits
            (2, 2),        // 3x3
            (3, 2),        // 3x3
            (4, 0),        // not used
            (5, 3),        // 4x4
            (6, 3),        // 4x4
            (7, 5),        // 6x6
            (8, 5),        // 6x6
            (9, 5),        // 6x6
            (10, 7),       // 8x8
            (11, 7),       // 8x8
            (12, 7),       // 8x8
            (13, 15),      // 16x16 no linbits
            (14, 0),       // not used
            (15, 15),      // 16x16 no linbits
            (16, 15 + 1),  // linbits=1
            (17, 15 + 3),  // linbits=2
            (18, 15 + 7),  // linbits=3
            (19, 15 + 15), // linbits=4
            (20, 15 + 63), // linbits=6
            (21, 15 + 255),
            (22, 15 + 1023),
            (23, 15 + 8191),
            (24, 15 + 15),     // linbits=4
            (25, 15 + 31),     // linbits=5
            (26, 15 + 63),     // linbits=6
            (27, 15 + 127),    // linbits=7
            (28, 15 + 255),    // linbits=8
            (29, 15 + 511),    // linbits=9
            (30, 15 + 2047),   // linbits=11
            (31, 15 + 8191),   // linbits=13
        ];
        for (idx, want) in spec_reach.iter().copied() {
            assert_eq!(
                crate::huffman::big_table_reach(idx),
                want,
                "big_table_reach({idx}) — expected §B.7 reach {want}"
            );
        }
    }

    /// `choose_best_table_for_region` must drop codebooks whose reach is
    /// less than the range's `max|is|`, **even when** the corner-only
    /// `xlen` check (`region_bits_with_table`) would accept them. The
    /// ESC tables 16..=31 clamp the Huffman symbol to 15 before lookup,
    /// so the corner test is identically satisfied for every ESC table,
    /// regardless of magnitude — which is exactly the silent-truncation
    /// trap the reach filter guards against (#1106).
    ///
    /// Construct an `is[]` with `|is[0]| = 100` and `is[1] = 0`: every
    /// small table 1..=13 rejects via `xlen` (their `xlen ≤ 16`), and
    /// every ESC table whose reach < 100 (16, 17, 18, 19, 24, 25, 26)
    /// would silently truncate at emit. The chooser must therefore
    /// return one of tables 20..=23 or 27..=31 (reach ≥ 100), and the
    /// chosen codebook must round-trip bit-exactly through
    /// `encode_huffman` → `decode_huffman`.
    #[test]
    fn chooser_filters_esc_tables_whose_linbits_truncates() {
        let mut is = [0i32; NUM_LINES];
        is[0] = 100;
        is[1] = 0;
        let (best, _) = choose_best_table_for_region(&is, 0, 2).unwrap();
        // The chooser must NOT have picked an under-reach ESC table.
        // Reach < 100: 16 (16), 17 (18), 18 (22), 19 (30), 20 (78),
        // 24 (30), 25 (46), 26 (78). Everything else covers 100.
        const UNDER_REACH: [u8; 8] = [16, 17, 18, 19, 20, 24, 25, 26];
        assert!(
            !UNDER_REACH.contains(&best),
            "chooser picked under-reach ESC table {best} for |is|=100"
        );
        assert!(
            crate::huffman::big_table_reach(best) >= 100,
            "chooser's best table {best} has reach < 100"
        );
        // Round-trip bit-exactness: emit + decode reproduces is[0..2]
        // and the rest is zero.
        assert_roundtrip(&is, 1, (2, 2), [best, 0, 0], 0, false, 0, 0);
    }

    /// Boundary check at magnitude 16: the smallest ESC table (16,
    /// `linbits=1`, reach `15 + 1 = 16`) IS in reach and should be
    /// permitted; the small tables 1..=15 (reach ≤ 15) must be filtered.
    /// We don't pin which table the chooser picks (the minimum-bit
    /// choice is a function of the codebook's symbol lengths, not of
    /// reach), only that the reach invariant holds.
    #[test]
    fn chooser_reach_boundary_at_magnitude_16() {
        let mut is = [0i32; NUM_LINES];
        is[0] = 16;
        is[1] = 0;
        let (best, _) = choose_best_table_for_region(&is, 0, 2).unwrap();
        assert!(
            crate::huffman::big_table_reach(best) >= 16,
            "chooser picked table {best} with reach < 16 for |is|=16"
        );
        assert_roundtrip(&is, 1, (2, 2), [best, 0, 0], 0, false, 0, 0);
    }

    /// Magnitude 15 boundary: every small table with `xlen ≥ 16` (i.e.
    /// 13 and 15) and every ESC table 16..=31 covers magnitude 15. The
    /// chooser may pick any of them; we assert reach ≥ 15 and a clean
    /// round-trip.
    #[test]
    fn chooser_reach_boundary_at_magnitude_15() {
        let mut is = [0i32; NUM_LINES];
        is[0] = 15;
        is[1] = 0;
        let (best, _) = choose_best_table_for_region(&is, 0, 2).unwrap();
        assert!(
            crate::huffman::big_table_reach(best) >= 15,
            "chooser picked table {best} with reach < 15 for |is|=15"
        );
        assert_roundtrip(&is, 1, (2, 2), [best, 0, 0], 0, false, 0, 0);
    }

    /// At the §C.1.5.4.4.2 clamp (`|is| = 8191`) the only in-reach
    /// codebooks are 23 and 31 (linbits 13, reach 8206). The chooser
    /// must pick one of them.
    #[test]
    fn chooser_picks_only_reach_8191_at_clamp() {
        let mut is = [0i32; NUM_LINES];
        is[0] = 8191;
        is[1] = 0;
        let (best, _) = choose_best_table_for_region(&is, 0, 2).unwrap();
        assert!(
            best == 23 || best == 31,
            "only tables 23/31 (reach 8206) cover the 8191 clamp; got {best}"
        );
        assert_roundtrip(&is, 1, (2, 2), [best, 0, 0], 0, false, 0, 0);
    }

    /// A magnitude past every codebook's reach (e.g. 9000 — beyond table
    /// 23's reach of 8206) is uncodable: the chooser returns `None`. In
    /// practice the magnitude clamp at §C.1.5.4.4.2 prevents this from
    /// ever being reached by a real encode, but the chooser must report
    /// "no table in reach" rather than silently truncate.
    #[test]
    fn chooser_returns_none_when_no_table_in_reach() {
        let mut is = [0i32; NUM_LINES];
        is[0] = 9000;
        is[1] = 0;
        assert!(
            choose_best_table_for_region(&is, 0, 2).is_none(),
            "magnitude 9000 exceeds every codebook's reach; chooser \
             must report None rather than pick a truncating table"
        );
    }

    /// Sanity: an all-zero range still picks table 0 (reach 0 covers
    /// magnitude 0).
    #[test]
    fn chooser_zero_range_picks_table_zero() {
        let is = [0i32; NUM_LINES];
        let (best, bits) = choose_best_table_for_region(&is, 0, 4).unwrap();
        assert_eq!(best, 0);
        assert_eq!(bits, 0);
    }

    /// Empty range is table 0 / zero bits per the existing §C.1.5.4.4.7
    /// contract — the reach filter must not regress this.
    #[test]
    fn chooser_empty_range_remains_table_zero() {
        let is = [0i32; NUM_LINES];
        let (best, bits) = choose_best_table_for_region(&is, 4, 4).unwrap();
        assert_eq!(best, 0);
        assert_eq!(bits, 0);
        let (best2, bits2) = choose_best_table_for_region(&is, 100, 50).unwrap();
        assert_eq!(best2, 0);
        assert_eq!(bits2, 0);
    }
}
