"""Replay a run recorded with `scripts/run.py --record DIR`.

    python scripts/visualize_run.py DIR                      interactive window, from episode 1
    python scripts/visualize_run.py DIR --mode video         one MP4 (or GIF) per episode, in DIR/videos
    python scripts/visualize_run.py DIR --mode summary       DIR/summary.png across all episodes

In the window: space plays/pauses, left/right step one frame, n/p switch episode. Playback runs on
into the next episode, and the learning-curve panel keeps the finished ones. Every run under
`results/runs/` can be opened this way.

See src/visualizer/recovery_replay.py for what each panel shows.
"""
import os
from dataclasses import dataclass
from typing import Literal

import tyro  # type: ignore


@dataclass
class Args:
    """
    Args:
        run: Directory written by `--record`.
        mode: "window" (interactive), "video" (one file per episode) or "summary" (static figure).
        episode: 1-based episode to open (window) or render (video). 0 renders every episode.
        fps: Frames per second of playback. The control period is 0.2 s, so 5 is real time.
        stride: Render every n-th control step in videos (2 halves the file and render time).
        format: Video container. "mp4" needs ffmpeg (system, or `uv pip install imageio-ffmpeg`)
            and falls back to GIF without it.
        out: Output directory (video) or file (summary). Defaults to inside `run`.
    """
    run: tyro.conf.Positional[str]
    mode: Literal["window", "video", "summary"] = "window"
    episode: int = 1
    fps: float = 10.0
    stride: int = 1
    format: Literal["mp4", "gif"] = "mp4"
    out: str = ""


def main(args: Args) -> None:
    if args.mode != "window":
        import matplotlib
        matplotlib.use("Agg")
    from visualizer.recovery_replay import ReplayFigure, open_run, save_video, summary_figure

    run, episodes = open_run(args.run)
    if not episodes:
        raise SystemExit(f"No episodes recorded in {args.run}")
    print(f"{len(episodes)} episode(s) in {args.run} (teacher: {run['meta'].get('teacher')})")

    if args.mode == "window":
        ReplayFigure(run, episodes).interactive(max(0, args.episode - 1), fps=args.fps)
    elif args.mode == "video":
        out = args.out or os.path.join(args.run, "videos")
        os.makedirs(out, exist_ok=True)
        which = range(len(episodes)) if args.episode == 0 else [args.episode - 1]
        for i in which:
            path = save_video(run, episodes, i, os.path.join(out, f"episode_{i + 1:03d}.{args.format}"),
                              fps=args.fps, stride=args.stride)
            print(f"  wrote {path}")
    else:
        path = args.out or os.path.join(args.run, "summary.png")
        summary_figure(run, episodes, path)
        print(f"  wrote {path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
