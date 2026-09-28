#!/bin/bash
# Render the seven scenes of the supplementary video into videos/ (git-ignored), from results/runs:
# six side-by-side replays (compare_video.py) and one per-map episode (scripts/visualize_run.py).
# About ten minutes, the renders in parallel; then `python scripts/videos/merge_videos.py` joins them.
#
#     scripts/videos/render_scenes.sh
#
# Needs ffmpeg from imageio-ffmpeg (`uv sync --extra viz`). No solver and no API key.
set -u
cd "$(dirname "$0")/../.." || exit 1
PY="${PYTHON:-python}"
OUT=videos
mkdir -p "$OUT"
export MPLBACKEND=Agg

for preset in luna312 breakdown422 rules121 rules246 rules280 rules162; do
  "$PY" scripts/videos/compare_video.py "$preset" > "$OUT/$preset.log" 2>&1 \
    && echo "done $preset" || echo "FAILED $preset (see $OUT/$preset.log)" &
done
# Scene 7: layout 307, fifth visit of the second per-map run, in the replay viewer's style, every second
# control step (4x real time at 10 fps).
tmp="$OUT/.tmp_permap307"; mkdir -p "$tmp"
( "$PY" scripts/visualize_run.py results/runs/permap_307_r2 --mode video --episode 5 --stride 2 --out "$tmp" > "$tmp.log" 2>&1 \
    && mv "$tmp/episode_005.mp4" "$OUT/permap307_visit5.mp4" && rm -rf "$tmp" "$tmp.log" \
    && echo "done permap307_visit5" || echo "FAILED permap307_visit5 (see $tmp.log)" ) &
wait
