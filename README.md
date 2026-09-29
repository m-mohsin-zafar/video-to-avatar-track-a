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

bash scripts/download_model.sh   # MediaPipe pose model ~5.6 MB
# brew install ffmpeg

python pipeline.py --video path/to/exercise.mp4
python scripts/render_bot.py --video path/to/exercise.mp4 --motion output/<stem>_motion.npz
```

Outputs in `output/`:

- `<stem>_motion.npz` -- landmarks + empty `smpl_*` slots
- `<stem>_motion.json` -- fps / detected frames
- `<stem>_review.mp4` -- skeleton overlay
- `<stem>_bot.mp4` -- source | humanoid bot

## Capture rules

- Tripod, full body head-to-floor, 1080p30+
- Side or 45 deg for squats / hinges; front is OK for a march demo
- One person, no burned-in text if you can avoid it
- Static camera so Colab can use GVHMR static-camera mode

## Next: fill SMPL on Colab

Follow [docs/COLAB.md](docs/COLAB.md). Download the result back to the Mac, then:

```bash
python scripts/ingest_smpl.py \
  --motion output/<stem>_motion.npz \
  --gvhmr path/to/hmr4d_results.pt
```

Then follow [docs/MAC_BLENDER.md](docs/MAC_BLENDER.md) to retarget onto Mixamo Y-Bot.

## License

Code in this repo is MIT. SMPL / SMPL-X models are not included; register at the MPI sites and respect their licenses. Mixamo characters follow Adobe terms.
