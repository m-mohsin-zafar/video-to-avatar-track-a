#!/usr/bin/env python3
"""Video -> pose motion + skeleton review MP4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.clean import clean_motion
from src.overlay import render_review
from src.pose2d import extract_pose

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "pose_landmarker_lite.task"


def save_motion(motion: dict, path: Path) -> None:
    np.savez_compressed(
        path,
        source=motion["source"],
        landmark_names=motion["landmark_names"],
        fps=motion["fps"],
        width=motion["width"],
        height=motion["height"],
        n_frames=motion["n_frames"],
        visible=motion["visible"],
        landmarks_2d=motion["landmarks_2d"],
        landmarks_world=motion["landmarks_world"],
        left_foot_contact=motion["contacts"]["left_foot"],
        right_foot_contact=motion["contacts"]["right_foot"],
        smpl_ready=np.array(False),
        smpl_body_pose=np.zeros((motion["n_frames"], 23, 3), dtype=np.float32),
        smpl_global_orient=np.zeros((motion["n_frames"], 3), dtype=np.float32),
        smpl_transl=np.zeros((motion["n_frames"], 3), dtype=np.float32),
        smpl_betas=np.zeros((10,), dtype=np.float32),
        note=np.array(motion["note"]),
    )
    meta = {
        "source": motion["source"],
        "fps": motion["fps"],
        "n_frames": int(motion["n_frames"]),
        "detected_frames": int(motion["visible"].sum()),
        "smpl_ready": False,
        "upgrade": "Run GVHMR on Colab and overwrite smpl_* with scripts/ingest_smpl.py",
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description="Video to pose motion + skeleton review MP4")
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--out-dir", type=Path, default=ROOT / "output")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = p.parse_args()

    video = args.video.expanduser().resolve()
    if not video.exists():
        raise SystemExit(f"Video not found: {video}")
    if not args.model.exists():
        raise SystemExit(f"Missing pose model: {args.model}\nRun bash scripts/download_model.sh")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = video.stem

    print(f"[1/3] pose extract  {video.name}")
    motion = extract_pose(video, args.model)
    print(f"      frames={motion['n_frames']}  detected={int(motion['visible'].sum())}  fps={motion['fps']:.2f}")

    print("[2/3] smooth + foot contacts")
    motion = clean_motion(motion)

    motion_path = args.out_dir / f"{stem}_motion.npz"
    save_motion(motion, motion_path)
    print(f"      wrote {motion_path}")

    print("[3/3] review video")
    review_path = args.out_dir / f"{stem}_review.mp4"
    render_review(video, motion, review_path)
    print(f"      wrote {review_path}")
    print("done. smpl_ready=false until you ingest GVHMR.")


if __name__ == "__main__":
    main()
