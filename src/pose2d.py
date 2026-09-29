"""Frame-by-frame MediaPipe Pose Landmarker (video mode)."""

from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .constants import LANDMARK_NAMES


def _lm_to_array(landmarks) -> np.ndarray:
    arr = np.zeros((33, 4), dtype=np.float32)
    if not landmarks:
        return arr
    for i, lm in enumerate(landmarks[:33]):
        vis = getattr(lm, "visibility", 0.0) or 0.0
        arr[i] = (lm.x, lm.y, lm.z, vis)
    return arr


def _letterbox_square(rgb: np.ndarray) -> tuple[np.ndarray, int, int, int]:
    h, w = rgb.shape[:2]
    side = max(h, w)
    canvas = np.zeros((side, side, 3), dtype=rgb.dtype)
    y0 = (side - h) // 2
    x0 = (side - w) // 2
    canvas[y0 : y0 + h, x0 : x0 + w] = rgb
    return canvas, x0, y0, side


def _unmap_xy(lm: np.ndarray, x0: int, y0: int, side: int, w: int, h: int) -> np.ndarray:
    out = lm.copy()
    out[:, 0] = (lm[:, 0] * side - x0) / w
    out[:, 1] = (lm[:, 1] * side - y0) / h
    return out


def extract_pose(video_path: Path, model_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    lm2d, lm_world, visible = [], [], []

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        idx = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            square, x0, y0, side = _letterbox_square(rgb)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=square)
            ts_ms = int(round(idx * 1000.0 / fps))
            result = landmarker.detect_for_video(mp_image, ts_ms)

            if result.pose_landmarks:
                raw = _lm_to_array(result.pose_landmarks[0])
                lm2d.append(_unmap_xy(raw, x0, y0, side, width, height))
                if result.pose_world_landmarks:
                    world = _lm_to_array(result.pose_world_landmarks[0])
                else:
                    world = np.zeros((33, 4), dtype=np.float32)
                lm_world.append(world)
                visible.append(True)
            else:
                lm2d.append(np.zeros((33, 4), dtype=np.float32))
                lm_world.append(np.zeros((33, 4), dtype=np.float32))
                visible.append(False)
            idx += 1

    cap.release()
    landmarks_2d = np.stack(lm2d, axis=0) if lm2d else np.zeros((0, 33, 4), np.float32)
    landmarks_world = np.stack(lm_world, axis=0) if lm_world else np.zeros((0, 33, 4), np.float32)

    return {
        "source": "mediapipe_pose_landmarker_lite",
        "landmark_names": np.array(LANDMARK_NAMES),
        "fps": float(fps),
        "width": width,
        "height": height,
        "n_frames": landmarks_2d.shape[0],
        "n_frames_hint": n_hint,
        "visible": np.array(visible, dtype=np.bool_),
        "landmarks_2d": landmarks_2d,
        "landmarks_world": landmarks_world,
        "smpl_ready": False,
        "note": "MediaPipe 2D + world landmarks. Fill smpl_* via Colab GVHMR + ingest_smpl.py.",
    }
