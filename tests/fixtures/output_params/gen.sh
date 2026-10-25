#!/bin/sh
# Regenerates the tests/output_params.rs fixtures with black-box ffmpeg
# (libmp3lame) and prints what ffprobe reports for each, which is what
# the test pins (ffprobe's sample_fmt is ffmpeg's own decoder output,
# not this crate's). No ID3 tag, no Xing/Info frame: every frame is
# audio. Run from this directory.
set -e
enc() {
    out=$1; rate=$2; ch=$3; shift 3
    ffmpeg -hide_banner -loglevel error -y -f lavfi \
        -i "sine=frequency=440:sample_rate=$rate:duration=0.25" \
        -ac "$ch" -c:a libmp3lame -b:a 32k "$@" \
        -id3v2_version 0 -write_xing 0 -f mp3 "$out"
    printf '%s: ' "$out"
    ffprobe -v error -show_streams -of compact "$out" | tr '|' '\n' |
        grep -E '^(sample_rate|channels|channel_layout)=' | tr '\n' ' '
    echo
}
enc mpeg1_44100_stereo.mp3 44100 2
enc mpeg1_32000_mono.mp3 32000 1
enc mpeg2_22050_stereo.mp3 22050 2
enc mpeg2_16000_mono.mp3 16000 1
enc mpeg25_11025_mono.mp3 11025 1
enc mpeg25_8000_stereo.mp3 8000 2
