# Colab + GVHMR

Upstream: https://github.com/zju3dv/GVHMR  
Official Colab (2024 snapshot): https://colab.research.google.com/drive/1N9WSchizHv2bfQqkE9Wuiegw_OT7mtGj  
In-repo notebook: `tools/demo/colab_demo.ipynb`

You are right: **that Colab is old relative to current Colab runtimes.**
The paper code is still the right model. The notebook pins CUDA 12.1, `torch-scatter` for cu121, and compiles **DPVO**. Colab in 2026 is a newer Python + CUDA stack, so those cells often die on `pytorch3d` / `torch-scatter` / `CUDA_HOME`.

Good news for our clip: the camera is effectively static. GVHMR added **SimpleVO** (2025-03-08) and marked DPVO optional. For a phone-on-a-table / standing-in-place exercise you should **never compile DPVO**. Pass `-s`.

## What to use instead of the official notebook as-is

Pick one, in this order:

### A. HuggingFace Space (zero install)

https://huggingface.co/spaces/LittleFrog/GVHMR

Upload the same mp4. If it finishes, download whatever motion file it offers. Fastest way to see if the lift is worth it. Limits apply.

### B. Fresh Colab, skip the 2024 install cells

1. New notebook, Runtime → **T4 GPU**.
2. Do **not** run the official notebook's DPVO / torch-scatter / pytorch3d cells.
3. Run a trimmed install:

```python
!nvidia-smi
!git clone --depth 1 https://github.com/zju3dv/GVHMR
%cd GVHMR
!pip install -q -r requirements.txt
!pip install -q -e .
```

If `requirements.txt` pins an ancient torch, let Colab keep its preinstalled CUDA torch and only install the rest (`einops`, `hydra-core`, `pytorch-lightning`, `smplx`, `ultralytics`, `timm`, …). Fighting Colab's torch is how the old notebook dies.

4. Weights (Google Drive folder linked from [INSTALL.md](https://github.com/zju3dv/GVHMR/blob/main/docs/INSTALL.md)):

```
inputs/checkpoints/
  gvhmr/gvhmr_siga24_release.ckpt
  hmr2/epoch=10-step=25000.ckpt
  vitpose/vitpose-h-multi-coco.pth
  yolo/yolov8x.pt
```

5. SMPL / SMPL-X body models from MPI (sign up, academic license). Put:

```
inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz
inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl
```

Do not commit those files to GitHub.

6. Upload your clip, then:

```bash
python tools/demo/demo.py --video=/content/your_clip.mp4 -s
```

`-s` = static camera = no DPVO. That is the correct flag for the high-knee demo.

7. Find the result `.pt` under `outputs/` and export:

```python
from pathlib import Path
import torch, numpy as np
from google.colab import files

pts = list(Path("outputs").rglob("*.pt"))
print("\n".join(map(str, pts)))
blob = torch.load(pts[0], map_location="cpu")
print(type(blob), getattr(blob, "keys", lambda: None)())

def to_np(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)

params = blob.get("smpl_params_incam", blob.get("smpl_params_global", blob.get("smpl_params", blob)))
# print keys and pick the ones that exist
if isinstance(params, dict):
    print(params.keys())
    pack = {}
    for src, dst in [
        ("body_pose", "body_pose"),
        ("global_orient", "global_orient"),
        ("transl", "transl"),
        ("betas", "betas"),
    ]:
        if src in params:
            pack[dst] = to_np(params[src])
    np.savez("gvhmr_smpl.npz", **pack)
    files.download("gvhmr_smpl.npz")
```

Key names vary (`smpl_params_global` vs `incam`). Print `blob.keys()` and adapt. `scripts/ingest_smpl.py` on the Mac accepts any of `body_pose` / `pose_body`, `transl` / `trans`.

### C. Modernized fork (if official install keeps breaking)

https://github.com/ryanrudes/gvhmr — same released weights, `uv` installer, DPVO optional. Still needs a CUDA box for the full stack; not for the M4 Air as the lift machine.

## Back on the Mac

```bash
python scripts/ingest_smpl.py \
  --motion output/<stem>_motion.npz \
  --gvhmr ~/Downloads/gvhmr_smpl.npz
```

Then [MAC_BLENDER.md](MAC_BLENDER.md).

## If Colab quota or weights fight you

DeepMotion / QuickMagic / Rokoko Vision → FBX → Blender. Same Track A, paid lift.
