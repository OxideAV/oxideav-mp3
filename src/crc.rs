//! MPEG audio CRC-16 — ISO/IEC 11172-3 §2.4.3.1 / Annex A Figure A.9.
//!
//! All three Layers share the same generator polynomial, initial state,
//! and shift-register procedure (the per-Layer differences are confined
//! to the **set of protected bits** — Annex B Table B.5). For Layer III
//! the protected bits are:
//!
//!   * bits 16…31 of the 32-bit frame header (i.e. the second 16 bits
//!     of the header — bytes 2 and 3 of the four-byte frame header),
//!   * the **first 135 bits of audio-data** in single-channel mode,
//!     i.e. `parts_of_side_info` for mono — the spec's "audio data"
//!     here means the side-information block that immediately follows
//!     the (optional) CRC in the wire layout,
//!   * the **first 256 bits of audio-data** in every other channel
//!     mode (stereo / joint stereo / dual channel).
//!
//! Per §2.4.3.1: "If the protection bit in the header equals '0', a
//! CRC-check word has been inserted in the bitstream just after the
//! header." The CRC field is two bytes wide; it occupies bytes 4..6
//! of the frame when present and shifts the side-information block
//! and main-data slot down by two bytes (the frame length computed by
//! [`crate::frame::Mp3FrameHeader::frame_len`] is constant — the CRC
//! consumes two bytes of what would otherwise be main-data slot
//! capacity).
//!
//! Provenance: every numeric value in this module is transcribed from
//! the ISO/IEC 11172-3:1993 PDF staged in `docs/audio/mp3/` and its
//! companion extract `docs/audio/mp3/mp1-crc-iso-extracts.md`. No
//! external library source was consulted, quoted, or paraphrased.

/// Generator polynomial coefficient mask (ISO/IEC 11172-3 §2.4.3.1):
///
/// G(X) = X^16 + X^15 + X^2 + 1.
///
/// As a 17-bit mask the polynomial is `0x18005`; the top bit corresponds
/// to X^16 (the bit that falls off the shift register's MSB and is
/// discarded after the conditional XOR). The 16-bit feedback mask that
/// the shift-register implementation actually XORs into the register
/// when the dropped MSB is `1` therefore strips X^16 and is
/// `0x8005` = X^15 | X^2 | 1.
pub const POLYNOMIAL_MASK: u32 = 0x8005;

/// Initial state of the 16-bit shift register (ISO/IEC 11172-3
/// §2.4.3.1): "the initial state of the shift register is
/// `'1111 1111 1111 1111'`", i.e. `0xFFFF`.
pub const INITIAL_STATE: u16 = 0xFFFF;

/// CRC-16 over a sequence of single bits (each `0` or `1` in the low
/// bit of the supplied `u8`), MSB-first per the §2.4.3.1 procedure:
///
/// "After each bit is input the shift register is shifted by one bit.
/// After the last shift operation, the outputs b15…b0 constitute a
/// word to be compared with the CRC-check word in the bitstream."
///
/// The shift-register update at each input bit `b_in` is:
///
/// 1. Compute `feedback = (register_msb) XOR (b_in)`.
/// 2. Left-shift the register by 1 (the MSB drops off).
/// 3. If `feedback` is `1`, XOR the register with the §2.4.3.1
///    polynomial mask `0x8005` (the X^15 + X^2 + 1 contributions; the
///    X^16 term is the bit that just dropped off).
///
/// This is the textbook standard-MSB-first CRC-16 implementation of the
/// §2.4.3.1 generator with no bit-reversal and no final XOR — the spec
/// specifies neither.
#[must_use]
pub fn crc16_bits<I: IntoIterator<Item = u8>>(bits: I) -> u16 {
    let mut reg: u32 = u32::from(INITIAL_STATE);
    for bit in bits {
        let b = u32::from(bit & 1);
        let feedback = ((reg >> 15) ^ b) & 1;
        reg = (reg << 1) & 0xFFFF;
        if feedback != 0 {
            reg ^= POLYNOMIAL_MASK;
        }
    }
    reg as u16
}

/// Compute the Layer III CRC-16 over the §2.4.3.1 / Annex B Table B.5
/// protected-bit set for one frame.
///
/// `header_bytes` is the four-byte frame header (the same bytes
/// [`crate::write_header`] returns). The CRC includes header bits
/// 16…31 — i.e. the second 16 bits — which are `header_bytes[2]`
/// followed by `header_bytes[3]`, fed MSB-first.
///
/// `side_info_bytes` is the byte-packed side-information block
/// produced by [`crate::write_side_info`]. The CRC includes the
/// **first 135 side-info bits** for `channels == 1` (mono /
/// single-channel mode) or the **first 256 side-info bits** for every
/// other channel mode, fed MSB-first from `side_info_bytes`.
///
/// # Panics
///
/// Panics if `side_info_bytes` is shorter than the required protected
/// length (17 bytes for mono — 135 bits rounded up — or 32 bytes for
/// other modes — 256 bits). Both are exactly the side-info block
/// lengths the writer produces, so a caller passing the writer's
/// output cannot trip the panic.
#[must_use]
pub fn crc16_layer3(header_bytes: &[u8; 4], side_info_bytes: &[u8], channels: u8) -> u16 {
    let si_bits: usize = if channels == 1 { 135 } else { 256 };
    // Need ceil(si_bits / 8) bytes of side info to draw from.
    let needed_bytes = si_bits.div_ceil(8);
    assert!(
        side_info_bytes.len() >= needed_bytes,
        "crc16_layer3: side_info_bytes len {} < {needed_bytes} required for channels={channels}",
        side_info_bytes.len(),
    );

    // Stream MSB-first: header bytes 2..4 (16 bits) then `si_bits`
    // bits drawn MSB-first from `side_info_bytes`.
    let header_iter = [header_bytes[2], header_bytes[3]]
        .into_iter()
        .flat_map(byte_bits_msb_first);

    let side_iter = side_info_bytes
        .iter()
        .copied()
        .flat_map(byte_bits_msb_first)
        .take(si_bits);

    crc16_bits(header_iter.chain(side_iter))
}

/// Decompose one byte into its eight bits, MSB first (bit 7, then bit
/// 6, …, then bit 0).
fn byte_bits_msb_first(byte: u8) -> [u8; 8] {
    [
        (byte >> 7) & 1,
        (byte >> 6) & 1,
        (byte >> 5) & 1,
        (byte >> 4) & 1,
        (byte >> 3) & 1,
        (byte >> 2) & 1,
        (byte >> 1) & 1,
        byte & 1,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_returns_initial_state() {
        // No input bits ⇒ the register is never shifted and the
        // returned value is the initial state itself.
        let crc = crc16_bits(core::iter::empty());
        assert_eq!(crc, INITIAL_STATE);
    }

    #[test]
    fn known_test_vector_polynomial_step() {
        // Hand-trace one input bit `b = 0`:
        //   feedback = (reg_msb XOR 0) = reg_msb = 1   (initial state)
        //   reg <<= 1                                 → 0xFFFE
        //   feedback != 0 ⇒ reg ^= 0x8005             → 0xFFFE ^ 0x8005
        //                                              = 0x7FFB
        let crc = crc16_bits(core::iter::once(0u8));
        assert_eq!(crc, 0x7FFB);

        // One input bit `b = 1`:
        //   feedback = (1 XOR 1) = 0
        //   reg <<= 1                                 → 0xFFFE
        //   feedback == 0 ⇒ no XOR
        let crc = crc16_bits(core::iter::once(1u8));
        assert_eq!(crc, 0xFFFE);
    }

    #[test]
    fn deterministic_over_assorted_inputs() {
        // Stability: changing one bit changes the CRC; same input twice
        // gives same CRC. (Not a known-answer test from the standard —
        // ISO 11172-3 does not include a worked example — but a sanity
        // bound on the implementation.)
        let bits_a: Vec<u8> = (0..40).map(|i| (i % 2) as u8).collect();
        let bits_b: Vec<u8> = (0..40).map(|i| ((i + 1) % 2) as u8).collect();
        let a1 = crc16_bits(bits_a.iter().copied());
        let a2 = crc16_bits(bits_a.iter().copied());
        let b1 = crc16_bits(bits_b.iter().copied());
        assert_eq!(a1, a2);
        assert_ne!(a1, b1);
    }

    #[test]
    fn layer3_crc_mono_matches_byte_oriented_recomputation() {
        // Build a frame's header + side-info and confirm
        // [`crc16_layer3`] takes the first 135 bits of side_info for
        // mono. Compare against an explicit-bit recomputation.
        let header = [0xFF, 0xFB, 0x90, 0xC4]; // 128 kbps, 44.1 kHz, mono, no CRC bit set yet
                                               // 17-byte mono side info, deterministic non-trivial content.
        let mut side: Vec<u8> = (0..17u8).collect();
        side[16] = 0xA5;

        let expected = {
            let mut bits: Vec<u8> = Vec::with_capacity(16 + 135);
            // Header bits 16..32 (i.e. bytes 2..4).
            for b in byte_bits_msb_first(header[2]) {
                bits.push(b);
            }
            for b in byte_bits_msb_first(header[3]) {
                bits.push(b);
            }
            // First 135 bits of side info.
            let mut count = 0;
            'outer: for &byte in side.iter() {
                for b in byte_bits_msb_first(byte) {
                    bits.push(b);
                    count += 1;
                    if count == 135 {
                        break 'outer;
                    }
                }
            }
            crc16_bits(bits)
        };

        let computed = crc16_layer3(&header, &side, 1);
        assert_eq!(computed, expected);
    }

    #[test]
    fn layer3_crc_stereo_uses_256_bits() {
        // For non-mono modes the protected side-info window is 256 bits
        // (32 bytes). Build a 32-byte stereo side info; spot-check that
        // the CRC changes when bit 255 changes and is unaffected by bit
        // 256.
        let header = [0xFF, 0xFB, 0x90, 0xC4];
        let mut side = vec![0x5Au8; 32];
        let crc_base = crc16_layer3(&header, &side, 2);

        // Flip the LAST protected bit (bit 255 = LSB of byte 31).
        side[31] ^= 0x01;
        let crc_flipped_last = crc16_layer3(&header, &side, 2);
        assert_ne!(crc_base, crc_flipped_last);

        // Reset and flip a bit beyond the protected window (need a 33rd
        // byte to test that — we just confirm that the 32-byte window
        // suffices: adding a 33rd byte should not affect the result).
        side[31] ^= 0x01;
        let mut side33 = side.clone();
        side33.push(0xFF);
        let crc_extra = crc16_layer3(&header, &side33, 2);
        assert_eq!(crc_base, crc_extra);
    }

    #[test]
    fn layer3_crc_independent_of_header_bits_0_15() {
        // §2.4.3.1 says the CRC covers bits 16..31 of the header only.
        // Bits 0..15 (the syncword + version + layer + protection bit)
        // must not affect the CRC value.
        let mut header = [0xFF, 0xFB, 0x90, 0xC4];
        let side = vec![0x33u8; 17];
        let base = crc16_layer3(&header, &side, 1);

        // Flip the protection bit (bit 16 of the header = bit 0 of
        // byte 2). Wait — bit 16 IS in bytes 2..4 per the index
        // convention "bit 0 = MSB" used by the spec. The "protection
        // bit" is at position 15 from the LSB of the 32-bit word; per
        // big-endian byte order it lives in `header[1] & 0x01`.
        // Flipping `header[1]` should NOT change the CRC.
        header[1] ^= 0x01;
        let flipped = crc16_layer3(&header, &side, 1);
        assert_eq!(base, flipped);

        // Flipping a bit in header[2] (which IS covered) MUST change
        // the CRC.
        header[1] ^= 0x01; // restore
        header[2] ^= 0x01;
        let flipped2 = crc16_layer3(&header, &side, 1);
        assert_ne!(base, flipped2);
    }
}
