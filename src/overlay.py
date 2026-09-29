"""Draw skeleton + joint-angle HUD onto the source video."""

from __future__ import annotations

from pathlib import Path
import subprocess

import cv2
import numpy as np

from .constants import HIP_TRIPLETS, KNEE_TRIPLETS, LANDMARK_NAMES, POSE_CONNECTIONS

NAME_TO_IDX = {n: i for i, n in enumerate(LANDMARK_NAMES)}


def _angle(a, b, c):
    ba = a - b
    bc = c - b
    n1 = np.linalg.norm(ba)
    n2 = np.linalg.norm(bc)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cos = float(np.clip(np.dot(ba, bc) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def _px(lm_xy, w, h):
    return int(lm_xy[0] * w), int(lm_xy[1] * h)


def render_review(video_path: Path, motion: dict, out_path: Path) -> Path:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = motion["fps"]
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tmp = out_path.with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    lm = motion["landmarks_2d"]
    vis = motion["visible"]
    contacts = motion.get("contacts", {})
    detected = int(vis.sum()) if vis.size else 0
    n = motion["n_frames"]

    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        overlay = frame.copy()
        if vis[i]:
            pts = lm[i]
            for a, b in POSE_CONNECTIONS:
                if pts[a, 3] < 0.3 or pts[b, 3] < 0.3:
                    continue
                cv2.line(overlay, _px(pts[a, :2], w, h), _px(pts[b, :2], w, h), (0, 220, 255), 3, cv2.LINE_AA)
            for j in range(33):
                if pts[j, 3] < 0.3:
                    continue
                color = (0, 255, 160)
                if j in (27, 29, 31) and contacts.get("left_foot", np.zeros(n, dtype=bool))[i]:
                    color = (0, 90, 255)
                if j in (28, 30, 32) and contacts.get("right_foot", np.zeros(n, dtype=bool))[i]:
                    color = (0, 90, 255)
                cv2.circle(overlay, _px(pts[j, :2], w, h), 5, color, -1, cv2.LINE_AA)
        frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)
        hud = [
            "TRACK A  review  |  MediaPipe pose",
            f"frame {i+1}/{n}   detected {detected}/{n}   fps {fps:.1f}",
        ]
        if vis[i]:
            for label, (a, b, c) in {**KNEE_TRIPLETS, **HIP_TRIPLETS}.items():
                ang = _angle(lm[i, NAME_TO_IDX[a], :3], lm[i, NAME_TO_IDX[b], :3], lm[i, NAME_TO_IDX[c], :3])
                if ang is not None:
                    hud.append(f"{label.replace('_', ' ')}  {ang:5.1f} deg")
        y = 28
        for line in hud:
            cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
            y += 22
        writer.write(frame)

    cap.release()
    writer.release()
    cmd = ["ffmpeg", "-y", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(out_path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        tmp.replace(out_path)
    else:
        tmp.unlink(missing_ok=True)
    return out_path
