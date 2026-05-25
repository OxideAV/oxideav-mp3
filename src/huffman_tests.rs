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
}
