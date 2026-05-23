# oxideav-mp3

A pure-Rust **MPEG-1 / MPEG-2 Audio Layer III** (MP3) codec for the
[oxideav](https://github.com/OxideAV/oxideav-workspace) framework.

## Status

**Clean-room rebuild in progress (started 2026-05-24).** The prior
implementation was retired under the workspace
[clean-room policy](https://github.com/OxideAV/oxideav-workspace/blob/master/docs/IMPLEMENTOR_ROUND.md):
several of its data tables and decode-loop structures were documented
as having been consulted from external reference implementations
(their source, not the ISO/IEC specification). The clean-room policy
forbids consulting any external implementation's source for any
reason, independent of that reference's licensing, so the provenance
could not be defended. Master history was fully erased per the Hat-3
cold-enforcement procedure.

The rebuild proceeds against ISO/IEC 11172-3:1993 and
ISO/IEC 13818-3:1997, with every numeric constant transcribed from
those standards and from no other source.

### Implemented

The `frame` module covers the MPEG audio **framing** layer
(ISO/IEC 11172-3 §2.4.1.3 / §2.4.2.3, with the lower-sampling-frequency
field redefinitions of ISO/IEC 13818-3 §2.4.2.3):

- `parse_header` decodes the four-byte frame header into a typed
  `Mp3FrameHeader`: 12-bit syncword check, MPEG version (`ID` bit,
  MPEG-1 vs MPEG-2 LSF), layer, CRC-protection flag, bitrate
  (per-version/per-layer ladders), sampling frequency (32/44.1/48 kHz
  for MPEG-1, 16/22.05/24 kHz for MPEG-2), padding, channel mode,
  mode_extension, copyright, original, and emphasis. Reserved /
  forbidden field values are rejected.
- Per-frame byte length including the padding slot, derived from the
  §2.4.2.3 padding formula (Layer III: `144·bitrate/Fs` for MPEG-1,
  `72·bitrate/Fs` for MPEG-2; Layer I/II covered too). Free-format
  frames report `None` (not derivable from the header alone).
- `FrameWalker` iterates self-delimiting frames over a byte buffer and
  resynchronises mid-stream on bad sync, garbage gaps, or a truncated
  trailing frame.

### Not yet implemented

Layer III audio decode (side-info parser, Huffman, requantisation,
IMDCT, synthesis filterbank) and any encoder. `register()` is a no-op
until a decoder/demuxer is wired up. MPEG-2.5 (8/11.025/12 kHz) is not
defined by either ISO standard and is therefore out of scope until a
clean-room reference for it is staged.

## License

MIT — see [LICENSE](./LICENSE).
