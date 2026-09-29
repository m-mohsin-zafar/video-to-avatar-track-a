#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.bot_render_pil import render_bot_pair

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description="Source | humanoid bot pair video")
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--motion", required=True, type=Path)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    d = np.load(args.motion, allow_pickle=True)
    motion = {
        "fps": float(d["fps"]),
        "n_frames": int(d["n_frames"]),
        "visible": d["visible"],
        "landmarks_2d": d["landmarks_2d"],
    }
    out = args.out or (ROOT / "output" / f"{args.video.stem}_bot.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    render_bot_pair(args.video, motion, out, ROOT / "output" / "_bot_work")
    print("wrote", out)


if __name__ == "__main__":
    main()
