#!/bin/sh
# Regenerates the tests/demuxer_splice.rs fixtures with the black-box
# ffmpeg CLI: one short single-shape MP3 per sampling
# frequency / channel mode / bitrate, which the test byte-concatenates
# into streams that switch mid-stream. `raw_*` files carry no ID3 tag
# and no Xing/Info frame (every frame is audio); `tagged_*` files keep
# the encoder's ID3v2 tag + Info/Xing frame and add an ID3v1 trailer.
# manifest.txt records what ffprobe counts in each file (audio packets,
# sampling frequency, channels) as the test's independent oracle.
# Run from this directory.
set -e
: > manifest.txt
# encw <codec> <muxer> <out> <rate> <channels> <tone Hz> [ffmpeg options]
encw() {
    codec=$1; fmt=$2; out=$3; rate=$4; ch=$5; freq=$6; shift 6
    ffmpeg -hide_banner -loglevel error -y -f lavfi \
        -i "sine=frequency=$freq:sample_rate=$rate:duration=0.15" \
        -ac "$ch" -c:a "$codec" "$@" -f "$fmt" "$out"
    printf '%s %s\n' "$out" "$(ffprobe -v error -count_packets \
        -show_entries stream=nb_read_packets,sample_rate,channels \
        -of csv=p=0 "$out" | tr ',' ' ')" >> manifest.txt
}
enc() { encw libmp3lame mp3 "$@"; }
raw() { enc "$@" -id3v2_version 0 -write_xing 0; }
tagged() { enc "$@" -id3v2_version 3 -write_id3v1 1 -metadata title=splice; }
# Every sampling frequency: MPEG-1, MPEG-2 LSF, MPEG-2.5; stereo
# (joint-stereo header mode).
raw raw_48000.mp3 48000 2 440 -b:a 64k
raw raw_44100.mp3 44100 2 523 -b:a 64k
raw raw_32000.mp3 32000 2 587 -b:a 64k
raw raw_24000.mp3 24000 2 659 -b:a 32k
raw raw_22050.mp3 22050 2 698 -b:a 32k
raw raw_16000.mp3 16000 2 784 -b:a 32k
raw raw_12000.mp3 12000 2 880 -b:a 24k
raw raw_11025.mp3 11025 2 988 -b:a 24k
raw raw_8000.mp3 8000 2 1047 -b:a 24k
# Channel mode and bitrate switches at one sampling frequency.
raw raw_44100_mono.mp3 44100 1 330 -b:a 64k
raw raw_44100_lr.mp3 44100 2 370 -b:a 64k -joint_stereo 0
raw raw_44100_128k.mp3 44100 2 415 -b:a 128k
raw raw_44100_vbr.mp3 44100 2 466 -q:a 2
raw raw_22050_mono.mp3 22050 1 554 -b:a 32k
raw raw_8000_mono.mp3 8000 1 622 -b:a 16k
# Layer II (MPEG-1 and MPEG-2 LSF) segments, for layer switches.
encw mp2 mp2 raw_l2_48000.mp2 48000 2 740 -b:a 192k
encw mp2 mp2 raw_l2_22050.mp2 22050 1 830 -b:a 64k
# Full files: ID3v2 + Info / Xing frame + ID3v1 trailer.
tagged tagged_44100.mp3 44100 2 523 -b:a 64k
tagged tagged_32000_vbr.mp3 32000 2 587 -q:a 4
tagged tagged_22050_mono.mp3 22050 1 698 -b:a 32k
cat manifest.txt
