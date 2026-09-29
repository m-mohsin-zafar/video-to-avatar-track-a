"""Temporal smoothing + simple foot-plant heuristic."""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def _smooth_axis(data: np.ndarray, window: int) -> np.ndarray:
    if data.shape[0] < window:
        return data
    out = data.copy()
    for j in range(data.shape[1]):
        for c in range(min(3, data.shape[2])):
            out[:, j, c] = savgol_filter(data[:, j, c], window_length=window, polyorder=2, mode="interp")
    return out


def clean_motion(motion: dict) -> dict:
    vis = motion["visible"]
    if vis.size == 0 or not vis.any():
        motion["contacts"] = {
            "left_foot": np.zeros(motion["n_frames"], dtype=np.bool_),
            "right_foot": np.zeros(motion["n_frames"], dtype=np.bool_),
        }
        return motion

    n = motion["n_frames"]
    window = min(9, n if n % 2 == 1 else n - 1)
    if window >= 5:
        motion["landmarks_2d"] = _smooth_axis(motion["landmarks_2d"], window)
        motion["landmarks_world"] = _smooth_axis(motion["landmarks_world"], window)

    world = motion["landmarks_world"]
    contacts = {}
    fps = max(motion["fps"], 1.0)
    for name, idx in (("left_foot", 27), ("right_foot", 28)):
        pos = world[:, idx, :3]
        vel = np.linalg.norm(np.diff(pos, axis=0, prepend=pos[:1]), axis=1) * fps
        height = pos[:, 1]
        floor = np.percentile(height[vis], 15) if vis.any() else 0.0
        planted = (vel < 0.35) & (height < floor + 0.06) & vis
        contacts[name] = planted

    motion["contacts"] = contacts
    return motion
