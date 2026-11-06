#!/bin/sh
# Regenerates the tests/stream_params.rs fixtures with black-box ffmpeg
# (libmp3lame): single-shape files that the test byte-concatenates into
# chained streams. No ID3 tag, no Xing/Info frame: every frame is audio.
# Prints what ffprobe reports for each. Run from this directory.
set -e
enc() {
    out=$1; rate=$2; ch=$3; shift 3
    ffmpeg -hide_banner -loglevel error -y -f lavfi \
        -i "sine=frequency=440:sample_rate=$rate:duration=0.25" \
        -ac "$ch" -c:a libmp3lame "$@" \
        -id3v2_version 0 -write_xing 0 -f mp3 "$out"
    printf '%s: ' "$out"
    ffprobe -v error -show_streams -of compact "$out" | tr '|' '\n' |
        grep -E '^(sample_rate|channels|channel_layout)=' | tr '\n' ' '
    echo
}
# MPEG-1 44.1 kHz stereo, joint-stereo header mode (`01`).
enc joint_44100.mp3 44100 2 -b:a 64k
# MPEG-1 44.1 kHz stereo, independent-stereo header mode (`00`).
enc stereo_44100.mp3 44100 2 -b:a 64k -joint_stereo 0
# MPEG-1 44.1 kHz mono.
enc mono_44100.mp3 44100 1 -b:a 32k
# MPEG-2 LSF 22.05 kHz mono.
enc mono_22050.mp3 22050 1 -b:a 32k
# MPEG-1 48 kHz stereo VBR (bitrate_index varies frame to frame).
enc vbr_48000.mp3 48000 2 -q:a 2
