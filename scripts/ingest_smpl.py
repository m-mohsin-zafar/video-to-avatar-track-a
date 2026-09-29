#!/usr/bin/env python3
"""Copy GVHMR / SMPL arrays into an existing motion.npz."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_gvhmr(path: Path) -> dict:
    obj = np.load(path, allow_pickle=True)
    if hasattr(obj, "files"):
        data = {k: obj[k] for k in obj.files}
    else:
        data = obj.item() if hasattr(obj, "item") else obj
    return data


def _pick(data, *names):
    for n in names:
        if isinstance(data, dict) and n in data:
            return np.asarray(data[n])
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--motion", required=True, type=Path, help="existing *_motion.npz from pipeline.py")
    p.add_argument("--gvhmr", required=True, type=Path, help=".pt / .npz / .pkl from Colab")
    args = p.parse_args()

    motion = dict(np.load(args.motion, allow_pickle=True))
    raw = _load_gvhmr(args.gvhmr) if args.gvhmr.suffix in {".npz", ".npy"} else None

    if raw is None:
        try:
            import torch
            blob = torch.load(args.gvhmr, map_location="cpu")
        except Exception as exc:
            raise SystemExit(
                f"Could not read {args.gvhmr}: {exc}\n"
                "On Colab export numpy arrays: body_pose, global_orient, transl, betas."
            ) from exc
        if isinstance(blob, dict):
            raw = blob.get("smpl_params", blob)
        else:
            raw = {"value": blob}

    body = _pick(raw, "body_pose", "smpl_body_pose", "pose_body")
    orient = _pick(raw, "global_orient", "smpl_global_orient", "root_orient")
    transl = _pick(raw, "transl", "smpl_transl", "trans")
    betas = _pick(raw, "betas", "smpl_betas")

    if body is None:
        raise SystemExit(
            "No body_pose in the GVHMR file. Keys: "
            + ", ".join(sorted(map(str, raw.keys())) if isinstance(raw, dict) else [type(raw).__name__])
        )

    body = np.asarray(body, dtype=np.float32)
    if body.ndim == 2 and body.shape[-1] == 69:
        body = body.reshape(body.shape[0], 23, 3)
    if body.ndim == 2 and body.shape[-1] == 72:
        # include global orient packed in first 3
        if orient is None:
            orient = body[:, :3]
        body = body[:, 3:].reshape(body.shape[0], 23, 3)

    n = int(motion["n_frames"])
    t = min(n, body.shape[0])
    motion["smpl_body_pose"] = np.zeros((n, 23, 3), dtype=np.float32)
    motion["smpl_body_pose"][:t] = body[:t]
    if orient is not None:
        orient = np.asarray(orient, dtype=np.float32).reshape(-1, 3)
        motion["smpl_global_orient"] = np.zeros((n, 3), dtype=np.float32)
        motion["smpl_global_orient"][: min(t, orient.shape[0])] = orient[:t]
    if transl is not None:
        transl = np.asarray(transl, dtype=np.float32).reshape(-1, 3)
        motion["smpl_transl"] = np.zeros((n, 3), dtype=np.float32)
        motion["smpl_transl"][: min(t, transl.shape[0])] = transl[:t]
    if betas is not None:
        b = np.asarray(betas, dtype=np.float32).reshape(-1)
        motion["smpl_betas"] = np.zeros((10,), dtype=np.float32)
        motion["smpl_betas"][: min(10, b.shape[0])] = b[:10]
    motion["smpl_ready"] = np.array(True)

    np.savez_compressed(args.motion, **motion)
    meta_path = args.motion.with_suffix(".json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta["smpl_ready"] = True
    meta["smpl_frames"] = int(t)
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"ingested {t} SMPL frames into {args.motion}")


if __name__ == "__main__":
    main()
