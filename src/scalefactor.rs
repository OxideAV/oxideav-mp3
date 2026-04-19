//! MPEG-1 Layer III scalefactor decode.
//!
//! For each granule + channel the scalefactors control per-sfb gain.
//! The scalefac_compress index (4 bits in side info) selects slen1 and
//! slen2 from ISO/IEC 11172-3 Table 3-B.32.
//!
//! Scalefactor band partition:
//! - Long blocks: 21 bands (sfb 0..20) arranged in 4 groups —
//!   sfb 0-5 use slen1, sfb 6-10 use slen1, sfb 11-15 use slen2,
//!   sfb 16-20 use slen2.
//!   If the previous granule of the same channel had the same scfsi
//!   bit set, the scalefactors for that group are reused (not re-sent).
//! - Short (non-switched) blocks: 12 bands × 3 windows, all sent each
//!   granule (scfsi is ignored). Sfb 0-5 use slen1, sfb 6-11 use slen2.
//! - Mixed blocks (block_type == 2 && mixed_block_flag): sfb 0-7 long
//!   use slen1, sfb 3-11 (short ×3) use slen1/slen2 split at sfb 5.

use oxideav_core::{Error, Result};

use crate::sideinfo::{GranuleChannel, SideInfo};
use oxideav_core::bits::BitReader;

/// (slen1, slen2) pair by scalefac_compress (MPEG-1, Table 3-B.32).
pub const SLEN_TABLE: [(u8, u8); 16] = [
    (0, 0),
    (0, 1),
    (0, 2),
    (0, 3),
    (3, 0),
    (1, 1),
    (1, 2),
    (1, 3),
    (2, 1),
    (2, 2),
    (2, 3),
    (3, 1),
    (3, 2),
    (3, 3),
    (4, 2),
    (4, 3),
];

/// Decoded scalefactors for a single (granule, channel).
#[derive(Clone, Debug, Default)]
pub struct ScaleFactors {
    /// Long-block scalefactors, sfb 0..21. Index 22 is never used but
    /// we keep 22 entries for safe indexing when reordering.
    pub l: [u8; 22],
    /// Short-block scalefactors — `s[sfb][window]`.
    pub s: [[u8; 3]; 13],
}

/// Decode scalefactors for MPEG-1 one granule + one channel. `prev` holds
/// the previous granule's scalefactors on the same channel, used for
/// scfsi reuse. Consumes bits from `br`.
pub fn decode_mpeg1(
    br: &mut BitReader<'_>,
    gc: &GranuleChannel,
    scfsi: &[bool; 4],
    gr: usize,
    prev: &ScaleFactors,
) -> Result<ScaleFactors> {
    let (slen1, slen2) = SLEN_TABLE[gc.scalefac_compress as usize];
    let mut sf = ScaleFactors::default();

    if gc.window_switching_flag && gc.block_type == 2 {
        // Short-block or mixed-block case — scfsi is ignored; always send
        // fresh scalefactors.
        if gc.mixed_block_flag {
            // Long portion: sfb 0..7 with slen1.
            for sfb in 0..8 {
                sf.l[sfb] = br.read_u32(slen1 as u32)? as u8;
            }
            // Short portion: sfb 3..5 use slen1, sfb 6..11 use slen2.
            for sfb in 3..6 {
                for win in 0..3 {
                    sf.s[sfb][win] = br.read_u32(slen1 as u32)? as u8;
                }
            }
            for sfb in 6..12 {
                for win in 0..3 {
                    sf.s[sfb][win] = br.read_u32(slen2 as u32)? as u8;
                }
            }
        } else {
            // Pure short block.
            for sfb in 0..6 {
                for win in 0..3 {
                    sf.s[sfb][win] = br.read_u32(slen1 as u32)? as u8;
                }
            }
            for sfb in 6..12 {
                for win in 0..3 {
                    sf.s[sfb][win] = br.read_u32(slen2 as u32)? as u8;
                }
            }
        }
    } else {
        // Long-block case. 4 scfsi groups.
        // Group 0: sfb 0..5, slen1.
        if gr == 0 || !scfsi[0] {
            for sfb in 0..6 {
                sf.l[sfb] = br.read_u32(slen1 as u32)? as u8;
            }
        } else {
            for sfb in 0..6 {
                sf.l[sfb] = prev.l[sfb];
            }
        }
        // Group 1: sfb 6..10, slen1.
        if gr == 0 || !scfsi[1] {
            for sfb in 6..11 {
                sf.l[sfb] = br.read_u32(slen1 as u32)? as u8;
            }
        } else {
            for sfb in 6..11 {
                sf.l[sfb] = prev.l[sfb];
            }
        }
        // Group 2: sfb 11..15, slen2.
        if gr == 0 || !scfsi[2] {
            for sfb in 11..16 {
                sf.l[sfb] = br.read_u32(slen2 as u32)? as u8;
            }
        } else {
            for sfb in 11..16 {
                sf.l[sfb] = prev.l[sfb];
            }
        }
        // Group 3: sfb 16..20, slen2.
        if gr == 0 || !scfsi[3] {
            for sfb in 16..21 {
                sf.l[sfb] = br.read_u32(slen2 as u32)? as u8;
            }
        } else {
            for sfb in 16..21 {
                sf.l[sfb] = prev.l[sfb];
            }
        }
    }

    Ok(sf)
}

/// Decode scalefactors for a whole frame (both granules, all channels
/// present). Returns `[gr][ch]` layout. Reads bits from `br` which must
/// already be positioned at the start of a granule/channel's part2 data.
///
/// `part2_3_length` is tracked externally so callers can resume reading
/// Huffman codes from exactly the right place.
pub fn decode_frame(br: &mut BitReader<'_>, si: &SideInfo) -> Result<[[ScaleFactors; 2]; 2]> {
    let _ = (br, si);
    Err(Error::unsupported(
        "decode_frame placeholder — use decode_mpeg1 per-granule-per-channel",
    ))
}

// -------------------- MPEG-2 LSF scalefactor decode --------------------

/// Per ISO/IEC 13818-3 §2.4.3.2 / Table A.4. `g_mod` and `g_scf_partitions`
/// below are the canonical 3-row MPEG-2 tables used to decompose a 9-bit
/// `scalefac_compress` into `slen[0..3]` bit widths and `nr_of_sfb[0..3]`
/// band counts.
///
/// Row selection based on block shape:
///   - row 0: long block (n_long_sfb == 22, n_short_sfb == 0)
///   - row 1: mixed block (n_long_sfb == 6 or 8, n_short_sfb == 30)
///   - row 2: short block (n_long_sfb == 0, n_short_sfb == 39)
///
/// The second half of each partition row (offsets 16..28) is for
/// intensity-stereo scalefactors on the R channel (when `ist=1`). Same
/// layout as minimp3's `g_scf_partitions`.
const SCF_PARTITIONS_MPEG2: [[u8; 28]; 3] = [
    [
        6, 5, 5, 5, 6, 5, 5, 5, 6, 5, 7, 3, 11, 10, 0, 0, 7, 7, 7, 0, 6, 6, 6, 3, 8, 8, 5, 0,
    ],
    [
        8, 9, 6, 12, 6, 9, 9, 9, 6, 9, 12, 6, 15, 18, 0, 0, 6, 15, 12, 0, 6, 12, 9, 6, 6, 18, 9, 0,
    ],
    [
        9, 9, 6, 12, 9, 9, 9, 9, 9, 9, 12, 6, 18, 18, 0, 0, 12, 12, 12, 0, 12, 9, 9, 6, 15, 12, 9,
        0,
    ],
];

/// Modular arithmetic table used to iteratively decompose
/// `scalefac_compress` into 4 `slen` values. Rows 0..2 are for normal
/// (non-intensity-stereo) decode of `scf_comp`, rows 3..5 are for
/// intensity-stereo R-channel where `ist_scf_comp = scf_comp >> 1`.
const SCF_MOD_MPEG2: [u8; 24] = [
    5, 5, 4, 4, 5, 5, 4, 1, 4, 3, 1, 1, 5, 6, 6, 1, 4, 4, 4, 1, 4, 3, 1, 1,
];

/// Block-shape description for MPEG-2 scalefactor decode.
#[derive(Clone, Copy, Debug)]
pub struct Mpeg2BlockShape {
    /// Total long-block sfbs in the partition (0, 6, 8, or 22).
    pub n_long: u8,
    /// Total short-block sfbs × 3 windows (0 or 30 or 39).
    pub n_short: u8,
}

impl Mpeg2BlockShape {
    /// Pick the shape from a granule-channel's window/block flags.
    /// Row 0 = long, row 1 = mixed, row 2 = short.
    pub fn from_granule(gc: &crate::sideinfo::GranuleChannel) -> Self {
        if gc.window_switching_flag && gc.block_type == 2 {
            if gc.mixed_block_flag {
                // MPEG-2 mixed: 6 long sfbs then 30 short (10 × 3).
                Self {
                    n_long: 6,
                    n_short: 30,
                }
            } else {
                // Pure short: 39 = 13 × 3.
                Self {
                    n_long: 0,
                    n_short: 39,
                }
            }
        } else {
            Self {
                n_long: 22,
                n_short: 0,
            }
        }
    }

    /// Index into `SCF_PARTITIONS_MPEG2`.
    fn partition_row(&self) -> usize {
        // Same formula as minimp3: !!n_short + !n_long.
        let a = usize::from(self.n_short != 0);
        let b = usize::from(self.n_long == 0);
        a + b
    }
}

/// Decode scalefactors for one MPEG-2 LSF granule-channel.
///
/// `scf_comp` is the 9-bit `scalefac_compress` field from side info.
/// `intensity_stereo` should be `true` on the R channel of an IS-stereo
/// frame — for now the decoder ignores IS (M/S is the common case), so
/// pass `false`.
///
/// Returns a `ScaleFactors` with long-block values in `.l` (up to 22
/// sfbs) and short-block values in `.s` (sfb 0..13 × 3 windows). For a
/// pure-long block, `.s` is zero. For a pure-short block, `.l` is zero.
pub fn decode_mpeg2(
    br: &mut BitReader<'_>,
    gc: &crate::sideinfo::GranuleChannel,
    intensity_stereo: bool,
) -> Result<ScaleFactors> {
    let shape = Mpeg2BlockShape::from_granule(gc);
    let ist: u8 = u8::from(intensity_stereo);
    let mut sfc: i32 = (gc.scalefac_compress_9 >> ist) as i32;

    // Decompose `sfc` into 4 slens using g_mod. `k` is the 4-aligned
    // offset into `SCF_MOD_MPEG2`; we increment it by 4 each time the
    // residual `sfc` is still non-negative after subtracting `modprod`.
    // Mirrors minimp3's C for-loop exactly so that `k` after the loop
    // is the partition-table byte offset.
    let mut slen: [u8; 4] = [0; 4];
    let mut k: usize = (ist as usize) * 3 * 4;
    while sfc >= 0 {
        if k + 4 > SCF_MOD_MPEG2.len() {
            return Err(Error::invalid(
                "MP3 MPEG-2 scalefac: scalefac_compress out of range",
            ));
        }
        let mut modprod: i32 = 1;
        for i in (0..4usize).rev() {
            let m = SCF_MOD_MPEG2[k + i] as i32;
            slen[i] = ((sfc / modprod) % m) as u8;
            modprod *= m;
        }
        sfc -= modprod;
        k += 4;
    }

    // Partition row (long / mixed / short), then advance by `k` as
    // accumulated above. `k` is also the index into the partition table;
    // IS-stereo rows start at offset 16 within the same partition row.
    let part_row_base = shape.partition_row();
    let part_base = &SCF_PARTITIONS_MPEG2[part_row_base];
    let part_slice_start = k;
    if part_slice_start + 4 > part_base.len() {
        return Err(Error::invalid(
            "MP3 MPEG-2 scalefac: partition offset out of range",
        ));
    }
    let nr_of_sfb: [u8; 4] = [
        part_base[part_slice_start],
        part_base[part_slice_start + 1],
        part_base[part_slice_start + 2],
        part_base[part_slice_start + 3],
    ];

    // Read scalefactors in 4 groups: group i reads nr_of_sfb[i]
    // values of width slen[i] bits each. Emit them into a flat iscf[40]
    // array following the sfb order:
    //   long sfbs first (0..n_long), then short sfbs × 3 windows.
    let mut iscf: [u8; 40] = [0; 40];
    let mut out = 0usize;
    for i in 0..4 {
        let cnt = nr_of_sfb[i] as usize;
        if cnt == 0 {
            continue;
        }
        let bits = slen[i] as u32;
        if bits == 0 {
            out += cnt;
            continue;
        }
        for _k in 0..cnt {
            if out >= iscf.len() {
                return Err(Error::invalid(
                    "MP3 MPEG-2 scalefac: too many sfbs for buffer",
                ));
            }
            iscf[out] = br.read_u32(bits)? as u8;
            out += 1;
        }
    }

    // Fan out iscf into ScaleFactors. For a long block, first n_long go
    // into .l; for short / mixed, the first n_long go into .l (mixed
    // case) and the remaining go into .s as [sfb][win].
    let mut sf = ScaleFactors::default();
    let n_long = shape.n_long as usize;
    let n_short = shape.n_short as usize;
    for (i, &v) in iscf.iter().enumerate().take(n_long) {
        if i < sf.l.len() {
            sf.l[i] = v;
        }
    }
    // Short sfbs are packed as 3 windows per sfb; total = n_short values.
    let short_base = if gc.mixed_block_flag {
        // Mixed MPEG-2: long part covers sfb 0..5 (n_long=6). Short
        // region starts at sfb 3 per ISO 13818-3 (same as MPEG-1 mixed
        // offset).
        3
    } else {
        0
    };
    for j in 0..n_short {
        let sfb = short_base + j / 3;
        let win = j % 3;
        if sfb < sf.s.len() && win < 3 {
            sf.s[sfb][win] = iscf[n_long + j];
        }
    }

    Ok(sf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sideinfo::GranuleChannel;

    #[test]
    fn long_block_scalefactors_compress_0() {
        // scalefac_compress = 0 -> (slen1, slen2) = (0, 0). No bits read.
        let gc = GranuleChannel {
            scalefac_compress: 0,
            ..Default::default()
        };
        let scfsi = [false; 4];
        let data = [0u8; 1];
        let mut br = BitReader::new(&data);
        let prev = ScaleFactors::default();
        let sf = decode_mpeg1(&mut br, &gc, &scfsi, 0, &prev).unwrap();
        assert!(sf.l.iter().all(|&v| v == 0));
    }

    #[test]
    fn scfsi_reuses_from_prev() {
        // Gr 1, scfsi bits set -> take all groups from prev (no bits read).
        let gc = GranuleChannel {
            scalefac_compress: 5, // slen1=1, slen2=1
            ..Default::default()
        };
        let scfsi = [true; 4];
        let mut prev = ScaleFactors::default();
        for (i, v) in prev.l.iter_mut().enumerate() {
            *v = i as u8;
        }
        let data = [0xFFu8; 1];
        let mut br = BitReader::new(&data);
        let sf = decode_mpeg1(&mut br, &gc, &scfsi, 1, &prev).unwrap();
        for i in 0..21 {
            assert_eq!(sf.l[i], i as u8);
        }
    }
}
