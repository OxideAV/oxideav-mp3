# Third-party acknowledgements

This crate is 100% pure Rust and does not link or statically embed any C
code. The tables and formulae it uses are drawn from the MPEG audio
specifications (ISO/IEC 11172-3 and ISO/IEC 13818-3), which are spec
constants and not subject to copyright. This file enumerates other
reference implementations the source consulted for algorithmic structure,
and the license under which they are available.

## minimp3 (CC0 / public domain)

Upstream: https://github.com/lieff/minimp3
License:  CC0 1.0 Universal (public domain dedication)

`src/scalefactor.rs` decodes the MPEG-2 LSF 9-bit `scalefac_compress`
field by iteratively decomposing the value via the `g_mod` modular-
arithmetic table and then indexing into `g_scf_partitions`. The
layout of both tables and the structural form of the decomposition
loop (partition row selection via `!!n_short + !n_long`, the
stride-4 offset `k`) mirror minimp3's approach. These tables are
themselves direct transcriptions of ISO/IEC 13818-3 §2.4.3.2 /
Annex A.4 and carry no independent copyright.

`src/frame.rs` — the `frame_bytes()` formula set references
`minimp3 hdr_frame_bytes` alongside the ISO §2.4.3.1 formulas. The
math is identical across both sources and identical to the spec.

`src/requantize.rs` — the comments at the top of
`intensity_stereo_mpeg1` and `find_is_bound_sfb` note that the
short-block IS bound approach is the same pattern as minimp3 /
libmad. The code itself was written fresh from the ISO spec.

Because minimp3 is public domain (CC0), no attribution is legally
required for any of the above, but this file is provided for
transparency about where algorithmic design was cross-checked.
