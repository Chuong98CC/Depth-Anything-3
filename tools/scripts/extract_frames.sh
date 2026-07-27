#!/usr/bin/env bash
#
# extract_frames.sh — Extract frames from Astribot stereo video recordings.
#
# Usage:
#   ./tools/extract_frames.sh [OPTIONS]
#
# Options:
#   --base-dir DIR       Root data directory (default: data/astribot_stereo_lrb)
#   --start N            First frame number to extract   (default: 1)
#   --stop N             Last frame number to extract    (default: 10)
#   --step N             Extract every N-th frame        (default: 1)
#   --format FMT         Output image format: jpg, png   (default: jpg)
#   --quality Q          ffmpeg -q:v value, 1-31 (lower is better; default: 2)
#   --dry-run            Print what would be done without extracting
#   -h, --help           Show this help message
#
# Examples:
#   # Extract first 10 frames (frames 1–10)
#   ./tools/extract_frames.sh
#
#   # Extract frames 100–200 with step 5 (every 5th frame)
#   ./tools/extract_frames.sh --start 100 --stop 200 --step 5
#
#   # Dry-run to see what would happen
#   ./tools/extract_frames.sh --start 50 --stop 100 --step 10 --dry-run

set -euo pipefail

# ---- defaults ---------------------------------------------------------------
BASE_DIR="data/astribot_stereo_lrb"
START=1
STOP=10
STEP=1
FORMAT="jpg"
QUALITY=2
DRY_RUN=false

# ---- helpers ----------------------------------------------------------------
die() { echo "[ERROR]" "$@" >&2; exit 1; }
info() { echo "[INFO]" "$@"; }

usage() {
    head -30 "$0" | sed -n 's/^# //p'
    exit 0
}

# ---- parse args -------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --base-dir) BASE_DIR="$2"; shift 2 ;;
        --start)    START="$2";    shift 2 ;;
        --stop)     STOP="$2";     shift 2 ;;
        --step)     STEP="$2";     shift 2 ;;
        --format)   FORMAT="$2";   shift 2 ;;
        --quality)  QUALITY="$2";  shift 2 ;;
        --dry-run)  DRY_RUN=true;  shift   ;;
        -h|--help)  usage ;;
        *) die "Unknown option: $1 (try --help)" ;;
    esac
done

# ---- validation -------------------------------------------------------------
[[ "$START" -ge 1 ]]  || die "--start must be >= 1 (got $START)"
[[ "$STOP"  -ge "$START" ]] || die "--stop ($STOP) must be >= --start ($START)"
[[ "$STEP"  -ge 1 ]]  || die "--step must be >= 1 (got $STEP)"
[[ -d "$BASE_DIR/videos" ]] || die "Videos directory not found: $BASE_DIR/videos"

VIDEO_DIR="$BASE_DIR/videos"
IMAGE_DIR="$BASE_DIR/images"
TOTAL_FRAMES=$(( (STOP - START) / STEP + 1 ))

info "Base dir  : $BASE_DIR"
info "Frame range: $START → $STOP  (step $STEP, $TOTAL_FRAMES frames per video)"
info "Format    : $FORMAT  (quality $QUALITY)"
[[ "$DRY_RUN" == true ]] && info "DRY RUN — no frames will be written"
echo

# ---- main loop --------------------------------------------------------------
EXTRACTED=0
FAILED=0

for video_dir in "$VIDEO_DIR"/*/; do
    cam=$(basename "$video_dir")
    info "Processing: $cam"

    for chunk_dir in "$video_dir"*/; do
        chunk=$(basename "$chunk_dir")

        for video in "$chunk_dir"*.mp4; do
            [ -f "$video" ] || continue

            video_name=$(basename "$video" .mp4)
            out_dir="$IMAGE_DIR/$cam/$chunk/$video_name"

            # Build ffmpeg select expression: frames matching [start, stop] with step.
            # ffmpeg frame index n is 0-based, so frame 1 = index 0.
            # Include: n in [START-1, STOP-1] that align to the step offset.
            sel_start=$((START - 1))
            sel_expr="if(lt(n\\,${sel_start})\\,0\\,if(gte(n\\,${STOP})\\,0\\,not(mod(n-${sel_start}\\,${STEP}))))"

            if [[ "$DRY_RUN" == true ]]; then
                echo "  → $out_dir/  (select='$sel_expr')"
                EXTRACTED=$((EXTRACTED + TOTAL_FRAMES))
                continue
            fi

            mkdir -p "$out_dir"

            if ffmpeg -y -v error \
                -i "$video" \
                -vf "select='$sel_expr',setpts=N/FRAME_RATE/TB" \
                -vsync 0 \
                -q:v "$QUALITY" \
                -frame_pts 1 \
                "$out_dir/frame_%06d.$FORMAT"; then
                actual=$(find "$out_dir" -name "*.$FORMAT" | wc -l)
                info "  ✓ $out_dir/  ($actual frames)"
                EXTRACTED=$((EXTRACTED + actual))
            else
                info "  ✗ $out_dir/  — ffmpeg failed"
                FAILED=$((FAILED + 1))
            fi
        done
    done
    echo
done

# ---- summary ----------------------------------------------------------------
if [[ "$DRY_RUN" == true ]]; then
    info "Dry-run complete.  Would extract ~$EXTRACTED frames."
else
    info "Done.  Extracted $EXTRACTED frames, $FAILED failures."
fi
