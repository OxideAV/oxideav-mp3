//! MP3 packet → AudioFrame decoder, wired into [`oxideav_codec::Decoder`].
//!
//! The decoder threads side info, scalefactors, Huffman, requantise,
//! antialias, IMDCT, and polyphase synthesis. It maintains a per-channel
//! IMDCT overlap state and a per-channel synthesis FIFO across frames,
//! plus a 4 KiB bit reservoir.
//!
//! **Coverage**:
//! - MPEG-1 Layer III: 32 / 44.1 / 48 kHz. Two granules per frame.
//! - MPEG-2 LSF Layer III: 16 / 22.05 / 24 kHz. One granule per frame.
//! - MPEG-2.5 Layer III: 8 / 11.025 / 12 kHz (unofficial Fraunhofer
//!   low-sample-rate extension). One granule per frame.
//! - mono / stereo / joint-stereo (M/S and intensity) / dual-channel.
//! - Long / short / start / stop / mixed block types. scfsi reuse.
//!   Bit reservoir look-back up to the per-version cap.
//! - MPEG-1 (3-bit `is_pos`, tan-ratio) and MPEG-2 / MPEG-2.5 (5-bit
//!   `is_pos` with selectable `intensity_scale`) intensity stereo;
//!   ISO/IEC 11172-3 §2.4.3.4.9.3 and ISO/IEC 13818-3 §2.4.3.2.
//!
//! **Not implemented**:
//! - CRC-16 verification — the CRC bytes are consumed but not checked.

use oxideav_codec::Decoder;
use oxideav_core::{
    AudioFrame, CodecId, CodecParameters, Error, Frame, Packet, Result, SampleFormat, TimeBase,
};

use crate::frame::{parse_frame_header, ChannelMode, MpegVersion};
use crate::huffman::{decode_count1, decode_pair};
use crate::imdct::{imdct_granule, ImdctState};
use crate::requantize::{
    antialias, find_is_bound_sfb, intensity_stereo_mpeg1, intensity_stereo_mpeg2,
    ms_boundary_sample, ms_stereo, ms_stereo_range, reorder_short, requantize_granule,
};
use crate::reservoir::Reservoir;
use crate::scalefactor::{
    decode_mpeg1 as decode_sf_mpeg1, decode_mpeg2 as decode_sf_mpeg2, ScaleFactors,
};
use crate::sfband::sfband_long;
use crate::sideinfo::SideInfo;
use crate::synthesis::{synthesize_granule, SynthesisState};
use oxideav_core::bits::BitReader;

pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    Ok(Box::new(Mp3Decoder {
        codec_id: params.codec_id.clone(),
        time_base: TimeBase::new(1, 48_000),
        pending: None,
        reservoir: Reservoir::new(),
        prev_sf: [[ScaleFactors::default(); 2]; 2],
        imdct_state: [ImdctState::new(), ImdctState::new()],
        synth_state: [SynthesisState::new(), SynthesisState::new()],
        eof: false,
    }))
}

// ScaleFactors needs Copy for the array init — it's small enough.
impl Copy for ScaleFactors {}

struct Mp3Decoder {
    codec_id: CodecId,
    time_base: TimeBase,
    pending: Option<Packet>,
    reservoir: Reservoir,
    /// prev_sf[gr][ch] — only [1][ch] matters for scfsi reuse.
    prev_sf: [[ScaleFactors; 2]; 2],
    imdct_state: [ImdctState; 2],
    synth_state: [SynthesisState; 2],
    eof: bool,
}

impl Decoder for Mp3Decoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        if self.pending.is_some() {
            return Err(Error::other(
                "MP3 decoder: receive_frame must be called before sending another packet",
            ));
        }
        self.pending = Some(packet.clone());
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        let Some(pkt) = self.pending.take() else {
            return if self.eof {
                Err(Error::Eof)
            } else {
                Err(Error::NeedMore)
            };
        };
        self.decode_packet(&pkt)
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        // Wipe the main pieces of Layer III state carried across frames:
        //   * bit reservoir (up to 4 KiB of previous main_data),
        //   * per-(granule,channel) previous scalefactors used by scfsi reuse,
        //   * per-channel IMDCT overlap buffer (18 samples × 32 subbands),
        //   * per-channel 1024-sample polyphase-synthesis FIFO.
        // Without this wipe the first decoded frame after a seek may
        // silently prepend up to one frame of pre-seek audio (the reservoir
        // view) and the synthesis FIFO will blend pre- and post-seek
        // content for the first ~32 samples.
        self.reservoir = Reservoir::new();
        self.prev_sf = [[ScaleFactors::default(); 2]; 2];
        self.imdct_state = [ImdctState::new(), ImdctState::new()];
        self.synth_state = [SynthesisState::new(), SynthesisState::new()];
        self.pending = None;
        self.eof = false;
        Ok(())
    }
}

impl Mp3Decoder {
    fn decode_packet(&mut self, pkt: &Packet) -> Result<Frame> {
        let data = &pkt.data;
        let hdr = parse_frame_header(data)?;
        // MPEG-1, MPEG-2 LSF, and MPEG-2.5 (unofficial Fraunhofer
        // low-sample-rate extension for 8 / 11.025 / 12 kHz) are all
        // supported. MPEG-2.5 shares the side-info shape, scalefactor
        // compressor, and single-granule layout of MPEG-2 LSF — only the
        // sample-rate tables (already in `frame.rs`) and sfb partitions
        // (see `sfband::sfband_{long,short}`) differ.
        let is_mpeg2_or_25 = matches!(hdr.version, MpegVersion::Mpeg2 | MpegVersion::Mpeg25);
        let is_mpeg2 = is_mpeg2_or_25;
        let channels = hdr.channels() as usize;
        let crc_bytes = if hdr.no_crc { 0 } else { 2 };
        let header_len = 4 + crc_bytes;
        let si_bytes = hdr.side_info_bytes();
        if data.len() < header_len + si_bytes {
            return Err(Error::NeedMore);
        }
        let si = if is_mpeg2 {
            SideInfo::parse_mpeg2(&hdr, &data[header_len..])?
        } else {
            SideInfo::parse_mpeg1(&hdr, &data[header_len..])?
        };

        // Update time base on first frame.
        self.time_base = TimeBase::new(1, hdr.sample_rate as i64);

        let main_data_start = header_len + si_bytes;
        let main_data = &data[main_data_start..];

        // Combine reservoir + this frame's main data. The first few
        // frames of a real MP3 stream commonly reference reservoir
        // data that doesn't exist yet (encoders fill the reservoir
        // gradually); refresh the reservoir with this frame's data and
        // emit a silent frame in that case rather than erroring out.
        let prev_view: Vec<u8> = match self.reservoir.view_from_lookback(si.main_data_begin) {
            Some(v) => v.to_vec(),
            None => {
                self.reservoir.append(main_data);
                let n = hdr.samples_per_frame() as usize;
                let bytes = vec![0u8; n * channels * 2];
                return Ok(Frame::Audio(AudioFrame {
                    format: SampleFormat::S16,
                    channels: channels as u16,
                    sample_rate: hdr.sample_rate,
                    samples: n as u32,
                    pts: pkt.pts,
                    time_base: self.time_base,
                    data: vec![bytes],
                }));
            }
        };
        let mut combined = prev_view;
        combined.extend_from_slice(main_data);

        // Decode all granules.
        let num_granules = si.num_granules as usize;
        let mut pcm = vec![[[0.0f32; 576]; 2]; 2]; // pcm[gr][ch][i]

        // scfsi reuse for granule 1 must read from gr=0 of the CURRENT
        // frame (same channel), not the previous frame's gr=1. Track per-
        // channel within this frame.
        let mut frame_gr0_sf: [ScaleFactors; 2] = Default::default();

        // Track the expected next-granule bit offset within `combined`;
        // this lets each granule start cleanly at part2_3_length boundary
        // even when the previous one over- or under-read by a few bits.
        let mut next_granule_bit: u64 = 0;
        // Precompute joint-stereo flags; needed inside the channel loop so
        // the R channel's scalefactor decode can switch to MPEG-2 IS mode
        // (where R's scalefactor partition row changes and the LSB of
        // `scalefac_compress` carries `intensity_scale`).
        let js = channels == 2 && hdr.channel_mode == ChannelMode::JointStereo;
        let is_on_hdr = js && (hdr.mode_extension & 0x1) != 0;
        for gr in 0..num_granules {
            // Per-channel raw-integer coefficients and scalefactors from this
            // granule — kept in full resolution until after stereo decoupling
            // so intensity-stereo has the information it needs (R-channel
            // scalefactors are reused as `is_pos`, and the IS bound is the
            // lowest sfb at which R's raw coefficients are all zero upward).
            let mut is_ch = [[0i32; 576]; 2];
            let mut sf_ch: [ScaleFactors; 2] = Default::default();
            for ch in 0..channels {
                let gc = si.granules[gr][ch];

                // Build a fresh BitReader for this granule that starts at
                // the precise expected bit offset within `combined`.
                let bit_off = next_granule_bit;
                let byte_off = (bit_off / 8) as usize;
                if byte_off >= combined.len() {
                    return Err(Error::invalid(
                        "MP3 decoder: ran out of main_data while decoding granule",
                    ));
                }
                let mut br = BitReader::new(&combined[byte_off..]);
                let skip = (bit_off % 8) as u32;
                if skip > 0 {
                    br.read_u32(skip)?;
                }
                let part_start = br.bit_position();
                let part_end_bit = part_start + gc.part2_3_length as u64;

                // Scalefactors. MPEG-1: scfsi-based reuse across the two
                // granules of one frame. MPEG-2 / MPEG-2.5: one granule per
                // frame, no scfsi; slens come from a 9-bit scalefac_compress.
                //
                // For the R channel of an IS-enabled MPEG-2 / MPEG-2.5 frame,
                // `decode_mpeg2` must be called with `intensity_stereo=true`
                // so the partition row / `slen` widths come from the IS half
                // of the MPEG-2 scalefactor-compress tables (ISO/IEC 13818-3
                // §2.4.3.2, SCF_PARTITIONS_MPEG2 offsets 16..28). The
                // resulting `.l` / `.s` values then hold `is_pos` for each
                // sfb, which the IS coupling step consumes.
                let ist_r = is_mpeg2 && is_on_hdr && ch == 1;
                let sf = if is_mpeg2 {
                    decode_sf_mpeg2(&mut br, &gc, ist_r)?
                } else {
                    decode_sf_mpeg1(&mut br, &gc, &si.scfsi[ch], gr, &frame_gr0_sf[ch])?
                };
                if gr == 0 {
                    frame_gr0_sf[ch] = sf;
                }
                self.prev_sf[gr][ch] = sf;

                // Huffman big-value pairs.
                let mut is_ = [0i32; 576];
                let mut idx = 0usize;
                let big = (gc.big_values * 2) as usize;

                // Compute region boundaries (long-block layout).
                let bounds = sfband_long(hdr.sample_rate);
                let r0_end = if gc.window_switching_flag && gc.block_type == 2 {
                    36 // shortcut for short blocks
                } else {
                    bounds[(gc.region0_count as usize + 1).min(22)] as usize
                };
                let r1_end = if gc.window_switching_flag && gc.block_type == 2 {
                    576
                } else {
                    bounds[(gc.region0_count as usize + gc.region1_count as usize + 2).min(22)]
                        as usize
                };

                // Big-values: read exactly `big_values` pairs as advertised
                // in the side info. The encoder allocates enough part2_3
                // bits for this many pairs.
                while idx < big.min(576) {
                    let table = if idx < r0_end {
                        gc.table_select[0]
                    } else if idx < r1_end {
                        gc.table_select[1]
                    } else {
                        gc.table_select[2]
                    };
                    if table == 0 {
                        // pair of zeros
                        is_[idx] = 0;
                        if idx + 1 < 576 {
                            is_[idx + 1] = 0;
                        }
                        idx += 2;
                        continue;
                    }
                    let (x, y) = decode_pair(&mut br, table)?;
                    is_[idx] = x;
                    if idx + 1 < 576 {
                        is_[idx + 1] = y;
                    }
                    idx += 2;
                }

                // Count1 region.
                while idx + 4 <= 576 && br.bit_position() < part_end_bit {
                    let (v, w, x, y) = decode_count1(&mut br, gc.count1table_select)?;
                    is_[idx] = v;
                    is_[idx + 1] = w;
                    is_[idx + 2] = x;
                    is_[idx + 3] = y;
                    idx += 4;
                }

                // Advance the granule cursor — small over- or under-reads
                // within the part2_3_length envelope are absorbed here.
                next_granule_bit = bit_off + gc.part2_3_length as u64;

                // Requantise. Stash for stereo processing first; antialias
                // happens later (after stereo) per ISO 11172-3 §2.4.3.4.
                let mut xr = [0.0f32; 576];
                requantize_granule(&is_, &mut xr, &gc, &sf, hdr.sample_rate);
                // Reorder short-block coefficients from window-major to
                // interleaved-by-window so the IMDCT sees them in the
                // expected layout. No-op for long blocks.
                reorder_short(&mut xr, &gc, hdr.sample_rate);
                pcm[gr][ch] = xr;
                is_ch[ch] = is_;
                sf_ch[ch] = sf;
            }

            // Stereo processing on the granule (after both channels) —
            // applied to requantised (NOT yet antialiased) coefficients.
            if channels == 2 && hdr.channel_mode == ChannelMode::JointStereo {
                let ms_on = (hdr.mode_extension & 0x2) != 0;
                let is_on = (hdr.mode_extension & 0x1) != 0;

                // Determine the intensity-stereo bound from the R channel's
                // raw integer coefficients (encoder guarantees zeros above
                // the bound). For non-IS frames the bound equals the full
                // range and the IS path is a no-op.
                //
                // MPEG-1 IS uses a 3-bit `is_pos` (0..=6, 7 = sentinel) with
                // the `tan(is_pos * pi/12)` ratio. MPEG-2 / MPEG-2.5 IS uses
                // a 5-bit `is_pos` (0..=30, 31 = sentinel) with a geometric
                // step selected by `intensity_scale` (the LSB of R's 9-bit
                // `scalefac_compress`); see `intensity_stereo_mpeg2`.
                let is_gc_r = si.granules[gr][1];
                let is_bound = if is_on {
                    find_is_bound_sfb(&is_ch[1], &is_gc_r, hdr.sample_rate)
                } else {
                    // IS off → boundary at end → IS path processes nothing.
                    if is_gc_r.window_switching_flag && is_gc_r.block_type == 2 {
                        13
                    } else {
                        21
                    }
                };

                if is_on && !is_mpeg2 {
                    // Apply MPEG-1 IS on bands ≥ is_bound. This runs BEFORE
                    // M/S so that M/S only touches bands below the bound.
                    let (l, r) = pcm[gr].split_at_mut(1);
                    intensity_stereo_mpeg1(
                        &mut l[0],
                        &mut r[0],
                        &sf_ch[1],
                        &is_gc_r,
                        hdr.sample_rate,
                        is_bound,
                    );
                } else if is_on && is_mpeg2 {
                    // MPEG-2 / MPEG-2.5 IS. `sf_ch[1]` contains per-sfb
                    // `is_pos` values decoded via the IS scalefactor
                    // partition rows; `intensity_scale` is the LSB of the
                    // R channel's 9-bit `scalefac_compress`.
                    let intensity_scale = (si.granules[gr][1].scalefac_compress_9 & 1) as u8;
                    let (l, r) = pcm[gr].split_at_mut(1);
                    intensity_stereo_mpeg2(
                        &mut l[0],
                        &mut r[0],
                        &sf_ch[1],
                        &is_gc_r,
                        hdr.sample_rate,
                        is_bound,
                        intensity_scale,
                    );
                }

                if ms_on {
                    // M/S on the below-bound region only when IS is also on
                    // (the IS path owns the above-bound coefficients). When
                    // IS is off, M/S covers the full granule.
                    let (l, r) = pcm[gr].split_at_mut(1);
                    if is_on {
                        // Rotate only samples below the IS bound. For long
                        // blocks the bound is a single sample offset; for
                        // short/mixed blocks use the same conservative
                        // sample-index derived from the sfb bound.
                        let boundary_sample = ms_boundary_sample(
                            &is_gc_r,
                            hdr.sample_rate,
                            is_bound,
                        );
                        ms_stereo_range(&mut l[0], &mut r[0], 0, boundary_sample);
                    } else {
                        ms_stereo(&mut l[0], &mut r[0]);
                    }
                }
            }

            // Now antialias each channel (long blocks only / mixed-block
            // long part).
            for ch in 0..channels {
                let gc = si.granules[gr][ch];
                antialias(&mut pcm[gr][ch], &gc);
            }
        }

        // IMDCT + polyphase synthesis per granule per channel.
        // MPEG-1: 2 granules × 576 = 1152 samples/channel/frame.
        // MPEG-2 LSF: 1 granule × 576 = 576 samples/channel/frame.
        let total_samples = 576u32 * num_granules as u32;
        let bytes_per_sample = SampleFormat::S16.bytes_per_sample();
        let mut out_bytes =
            Vec::with_capacity(total_samples as usize * channels * bytes_per_sample);

        let mut pcm_per_gr = [[0.0f32; 576]; 2]; // [ch][i] for current granule
        for gr in 0..num_granules {
            for ch in 0..channels {
                let mut sb = [[0.0f32; 18]; 32];
                let gc = si.granules[gr][ch];
                imdct_granule(
                    &pcm[gr][ch],
                    &mut sb,
                    &mut self.imdct_state[ch],
                    if gc.window_switching_flag {
                        gc.block_type
                    } else {
                        0
                    },
                    gc.mixed_block_flag,
                );
                synthesize_granule(&mut self.synth_state[ch], &sb, &mut pcm_per_gr[ch]);
            }
            // Interleave samples.
            for i in 0..576 {
                for ch in 0..channels {
                    let f = pcm_per_gr[ch][i].clamp(-1.0, 1.0);
                    let s = (f * 32767.0) as i16;
                    out_bytes.extend_from_slice(&s.to_le_bytes());
                }
            }
        }

        // Append this frame's main_data to the reservoir for next frame.
        self.reservoir.append(main_data);

        Ok(Frame::Audio(AudioFrame {
            format: SampleFormat::S16,
            channels: channels as u16,
            sample_rate: hdr.sample_rate,
            samples: total_samples,
            pts: pkt.pts,
            time_base: self.time_base,
            data: vec![out_bytes],
        }))
    }
}
