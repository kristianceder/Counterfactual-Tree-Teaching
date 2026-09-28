"""The supplementary video: the seven scenes, each after a title card, in the order the paper reaches
them. Render the scenes first (`scripts/videos/render_scenes.sh`).

    python scripts/videos/merge_videos.py                   # -> videos/supplementary.mp4
    python scripts/videos/merge_videos.py --card-seconds 3

Clips are scaled and padded to 1920x1080 at 10 fps (the per-map clip is 1530x990), re-encoded with
libx264, and concatenated with ffmpeg's concat filter (the ffmpeg binary comes from imageio-ffmpeg, the
`viz` extra). Cards are drawn with matplotlib in the clips' typeface.
"""
from __future__ import annotations

import os
import subprocess
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import tyro  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEOS = os.path.normpath(os.path.join(HERE, "..", "..", "videos"))
OUT = os.path.join(VIDEOS, "supplementary.mp4")

CLIPS = [  # file, card title, card text (one or two lines)
    ("luna312.mp4", "1  One held-out layout, four deciders",
     "Seed 312, which none of them learned on: no supervisor, the scripted rules, the LLM alone, the frozen tree.\n"
     "The rules and the tree arrive; the LLM alone orders a route resume out of a gate hold, reverses, and runs out of time.  (Sec. 4.2, Table 1)"),
    ("breakdown422.mp4", "2  A breakdown on an unseen layout",
     "Seed 422, no LLM on board: the method's frozen tree against distillation alone's. Both wait while the blocker stands;\n"
     "once it has stood 30 s the method's rule calls a human, distillation keeps waiting until the watchdog stops the controller.  (Sec. 4.3)"),
    ("rules121.mp4", "3  Where the scripted rules fail, 1 of 3: layout 121",
     "The same two recoveries on both sides; the tree replans before the robot has stopped, the rules once it has stood 4 s.\n"
     "The 6 s head start is the margin.  (Sec. 4.3, App. B)"),
    ("rules246.mp4", "4  Where the scripted rules fail, 2 of 3: layout 246",
     "The rules replan 8 s after the tree and the robot arrives at the gate as the pedestrian crosses it.  (Sec. 4.3, App. B)"),
    ("rules280.mp4", "5  Where the scripted rules fail, 3 of 3: layout 280",
     "The tree replans at once and turns to the new route; the rules replan 18 s later and stand behind a pedestrian in the gate.  (Sec. 4.3, App. B)"),
    ("rules162.mp4", "6  Where the method fails: layout 162",
     "The one learning layout the method ends wrong in every run and the rules end right: the tree replans 12 s sooner, holds twice at the gate,\n"
     "then holds 20 s more for a pedestrian on the route, and the step budget runs out.  (Sec. 4.3, App. B)"),
    ("permap307_visit5.mp4", "7  A repeated layout: 307, fifth visit",
     "Learning from an empty tree on one layout, visit five: the tree waits and resumes behind pedestrians until the step budget runs out.  (Sec. 4.4)"),
]
W, H, FPS = 1920, 1080, 10


def card(path: str, title: str, text: str) -> None:
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_facecolor("white")
    fig.text(0.05, 0.60, title, fontsize=40, fontweight="bold", color="#222220", va="bottom")
    fig.text(0.05, 0.54, text, fontsize=17, color="#4a4a47", va="top", linespacing=1.6)
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)


def main(out: str = OUT, card_seconds: float = 2.5) -> None:
    """Write the merged video; `card_seconds` is how long each title card stays."""
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.TemporaryDirectory() as tmp:
        args, filters, n = [ffmpeg, "-y"], [], 0
        for i, (rel, title, text) in enumerate(CLIPS):
            clip = os.path.join(VIDEOS, rel)
            if not os.path.exists(clip):
                raise SystemExit(f"missing clip: {clip}")
            png = os.path.join(tmp, f"card{i}.png")
            card(png, title, text)
            args += ["-loop", "1", "-t", str(card_seconds), "-i", png, "-i", clip]
            for k in (2 * i, 2 * i + 1):
                filters.append(f"[{k}:v]scale={W}:{H}:force_original_aspect_ratio=decrease,"
                               f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:white,fps={FPS},format=yuv420p,setsar=1[v{k}]")
                n += 1
        filters.append("".join(f"[v{k}]" for k in range(n)) + f"concat=n={n}:v=1:a=0[out]")
        args += ["-filter_complex", ";".join(filters), "-map", "[out]", "-c:v", "libx264", "-crf", "23",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
        subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    probe = subprocess.run([ffmpeg, "-i", out], capture_output=True, text=True).stderr
    dur = next((l.strip() for l in probe.splitlines() if "Duration" in l), "")
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB; {dur})")


if __name__ == "__main__":
    tyro.cli(main)
