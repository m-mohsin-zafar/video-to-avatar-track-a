# video-to-avatar-track-a

Turn an exercise video into motion data, a skeleton review, and a humanoid-bot preview. Real SMPL joint rotations are produced on **Google Colab (T4)** and consumed on a **Mac (M4 Air is fine)** in Blender.

```
Mac (MediaPipe)          Colab T4 (GVHMR)           Mac (Blender)
video --> motion.npz --> smpl_* filled --> Y-Bot / VRM render
          review.mp4
          bot.mp4
```

This is **Track A**: structured motion, not a pixel-diffusion lookalike.

## What works where

| Step | Machine | GPU |
|---|---|---|
| Pose extract + review overlay + bot preview | MacBook Air M4 | No (CPU / ANE) |
| SMPL / world-grounded lift (GVHMR) | Colab T4 | Yes, CUDA |
| Mixamo Y-Bot retarget + render | Mac + Blender | No |

Do not try to install GVHMR natively on the Air. It expects CUDA.

## Quick start (Mac)

```bash
git clone https://github.com/m-mohsin-zafar/video-to-avatar-track-a.git
cd video-to-avatar-track-a

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

bash scripts/download_model.sh
# brew install ffmpeg

python pipeline.py --video path/to/exercise.mp4
python scripts/render_bot.py --video path/to/exercise.mp4 --motion output/<stem>_motion.npz
```

## Fill SMPL on Colab

Use **our** notebook, not the 2024 official one:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/m-mohsin-zafar/video-to-avatar-track-a/blob/main/notebooks/GVHMR_colab.ipynb)

Details: [docs/COLAB.md](docs/COLAB.md)

```bash
python scripts/ingest_smpl.py \
  --motion output/<stem>_motion.npz \
  --gvhmr ~/Downloads/gvhmr_smpl.npz
```

Then [docs/MAC_BLENDER.md](docs/MAC_BLENDER.md).

## License

Code in this repo is MIT. SMPL / SMPL-X models are not included. Mixamo characters follow Adobe terms.
