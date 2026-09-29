"""Humanoid bot renderer using numpy + Pillow + ffmpeg."""

from __future__ import annotations

import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

from .constants import LANDMARK_NAMES

NAME = {n: i for i, n in enumerate(LANDMARK_NAMES)}
BONES = [
    ("torso", "left_shoulder", "right_shoulder"),
    ("torso", "left_hip", "right_hip"),
    ("torso", "left_shoulder", "left_hip"),
    ("torso", "right_shoulder", "right_hip"),
    ("arm", "left_shoulder", "left_elbow"),
    ("arm", "left_elbow", "left_wrist"),
    ("arm", "right_shoulder", "right_elbow"),
    ("arm", "right_elbow", "right_wrist"),
    ("leg", "left_hip", "left_knee"),
    ("leg", "left_knee", "left_ankle"),
    ("leg", "right_hip", "right_knee"),
    ("leg", "right_knee", "right_ankle"),
    ("foot", "left_ankle", "left_heel"),
    ("foot", "left_ankle", "left_foot_index"),
    ("foot", "right_ankle", "right_heel"),
    ("foot", "right_ankle", "right_foot_index"),
]
COLORS = {
    "torso": (255, 200, 70),
    "arm": (255, 180, 0),
    "leg": (160, 220, 0),
    "foot": (255, 90, 40),
}


def _pt(lm, name, w, h):
    i = NAME[name]
    return int(lm[i, 0] * w), int(lm[i, 1] * h), float(lm[i, 3])


def draw_bot(w, h, lm) -> Image.Image:
    img = Image.new("RGB", (w, h), (18, 18, 22))
    d = ImageDraw.Draw(img)
    for y in range(h // 2, h, 48):
        d.line([(0, y), (w, y)], fill=(32, 32, 38), width=1)
    for x in range(0, w, 48):
        d.line([(x, h // 2), (x, h)], fill=(32, 32, 38), width=1)

    ls, rs = _pt(lm, "left_shoulder", w, h), _pt(lm, "right_shoulder", w, h)
    lh, rh = _pt(lm, "left_hip", w, h), _pt(lm, "right_hip", w, h)
    nose = _pt(lm, "nose", w, h)
    if min(ls[2], rs[2], lh[2], rh[2]) > 0.25:
        d.polygon([(ls[0], ls[1]), (rs[0], rs[1]), (rh[0], rh[1]), (lh[0], lh[1])], fill=(40, 70, 90))

    for kind, a, b in BONES:
        p1, p2 = _pt(lm, a, w, h), _pt(lm, b, w, h)
        if p1[2] < 0.25 or p2[2] < 0.25:
            continue
        thick = 14 if kind == "torso" else 11 if kind in ("arm", "leg") else 7
        d.line([(p1[0], p1[1]), (p2[0], p2[1])], fill=COLORS[kind], width=thick)
        r = max(4, thick // 2)
        d.ellipse((p1[0] - r, p1[1] - r, p1[0] + r, p1[1] + r), fill=(240, 240, 240))
        d.ellipse((p2[0] - r, p2[1] - r, p2[0] + r, p2[1] + r), fill=(240, 240, 240))

    if nose[2] > 0.25 and ls[2] > 0.25 and rs[2] > 0.25:
        neck = ((ls[0] + rs[0]) // 2, (ls[1] + rs[1]) // 2)
        radius = max(18, int(0.28 * ((ls[0] - rs[0]) ** 2 + (ls[1] - rs[1]) ** 2) ** 0.5))
        cx, cy = nose[0], min(nose[1], neck[1] - radius // 2)
        d.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=(255, 200, 70), width=3, fill=(28, 40, 48))
        d.ellipse((cx - radius // 3 - 3, cy - 5, cx - radius // 3 + 3, cy + 1), fill=(80, 255, 180))
        d.ellipse((cx + radius // 3 - 3, cy - 5, cx + radius // 3 + 3, cy + 1), fill=(80, 255, 180))
        d.line([neck, (cx, cy + radius)], fill=(255, 200, 70), width=6)

    d.text((16, 16), "HUMANOID BOT  |  pose-driven", fill=(200, 200, 200))
    d.text((16, 36), "CPU stand-in  |  SMPL/Mixamo next", fill=(120, 120, 130))
    return img


def render_bot_pair(video: Path, motion: dict, out: Path, work: Path) -> Path:
    src_dir = work / "src"
    pair_dir = work / "pair"
    for p in (src_dir, pair_dir):
        p.mkdir(parents=True, exist_ok=True)

    subprocess.check_call(["ffmpeg", "-y", "-i", str(video), "-vsync", "0", str(src_dir / "f_%04d.png")], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    frames = sorted(src_dir.glob("f_*.png"))
    n = min(len(frames), motion["n_frames"])
    w, h = Image.open(frames[0]).size
    lm = motion["landmarks_2d"]
    vis = motion["visible"]

    for i in range(n):
        src = Image.open(frames[i]).convert("RGB")
        bot = draw_bot(w, h, lm[i]) if vis[i] else Image.new("RGB", (w, h), (18, 18, 22))
        pair = Image.new("RGB", (w * 2, h))
        pair.paste(src, (0, 0))
        pair.paste(bot, (w, 0))
        d = ImageDraw.Draw(pair)
        d.text((16, h - 28), "SOURCE", fill=(255, 255, 255))
        d.text((w + 16, h - 28), "BOT", fill=(255, 200, 70))
        pair.save(pair_dir / f"f_{i+1:04d}.png")

    fps = motion["fps"]
    subprocess.check_call([
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", str(pair_dir / "f_%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(out),
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out
