# Colab + GVHMR

**Use this notebook (copy of upstream, updated for 2026 Colab):**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/m-mohsin-zafar/video-to-avatar-track-a/blob/main/notebooks/GVHMR_colab.ipynb)

File in this repo: [`notebooks/GVHMR_colab.ipynb`](../notebooks/GVHMR_colab.ipynb)

Upstream: https://github.com/zju3dv/GVHMR  
Official notebook (stale pins): https://colab.research.google.com/drive/1N9WSchizHv2bfQqkE9Wuiegw_OT7mtGj

## What we changed vs official `colab_demo.ipynb`

| Official (2024) | This copy |
|---|---|
| `pip install -r requirements.txt` pins `torch==2.3.0+cu121` | Keeps Colab's current CUDA torch |
| py3.10 `pytorch3d` wheel | Optional; inference works without preview video |
| DPVO compile cell (commented, still in the way) | Removed. Always `-s` for static exercise clips |
| Google Drive `gdown` for custom video | `files.upload()` |
| Stops at overlay mp4 | Exports `gvhmr_smpl.npz` for `scripts/ingest_smpl.py` |

## Run it

1. Open the badge above. Runtime → **T4 GPU**.
2. If Colab says restart after pip, click **Cancel**.
3. Run cells top to bottom.
4. Skip the tennis smoke test if you are short on quota.
5. Upload the same mp4 you ran through `pipeline.py`.
6. Download `gvhmr_smpl.npz`.

```bash
python scripts/ingest_smpl.py \
  --motion output/<stem>_motion.npz \
  --gvhmr ~/Downloads/gvhmr_smpl.npz
```

Then [MAC_BLENDER.md](MAC_BLENDER.md).

## Fallback

- HuggingFace Space (no install, weaker export): https://huggingface.co/spaces/LittleFrog/GVHMR
- Paid lift: DeepMotion / QuickMagic / Rokoko Vision → FBX → Blender
