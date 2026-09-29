# Colab: GVHMR SMPL lift

The MacBook Air M4 cannot run official GVHMR (CUDA). Use a free T4 runtime.

## 1. Open a GPU notebook

1. https://colab.research.google.com
2. Runtime → Change runtime type → **T4 GPU**
3. Confirm:

```python
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

## 2. Use the official GVHMR demo

Follow the current GVHMR repo / Colab (search "GVHMR colab" — the project lives under `zju3dv` / `hyunsoocha` depending on the year).

Required settings for this pipeline:

- Upload the **same** exercise clip you ran through `pipeline.py` on the Mac
- Enable **static camera** if the footage is on a tripod (`-s` in their CLI)
- One person in frame

Export whatever file the demo writes (`hmr4d_results.pt` or a pack of numpy arrays).

If the demo only dumps a `.pt`, add a cell:

```python
import torch, numpy as np
from google.colab import files

blob = torch.load("hmr4d_results.pt", map_location="cpu")
# inspect:
print(type(blob), blob.keys() if isinstance(blob, dict) else "")

# Adjust keys to whatever that checkpoint actually uses:
params = blob.get("smpl_params", blob)
np.savez(
    "gvhmr_smpl.npz",
    body_pose=params["body_pose"].detach().cpu().numpy() if hasattr(params["body_pose"], "detach") else params["body_pose"],
    global_orient=params["global_orient"].detach().cpu().numpy() if hasattr(params["global_orient"], "detach") else params["global_orient"],
    transl=params["transl"].detach().cpu().numpy() if hasattr(params["transl"], "detach") else params["transl"],
    betas=params["betas"].detach().cpu().numpy() if hasattr(params["betas"], "detach") else params["betas"],
)
files.download("gvhmr_smpl.npz")
```

SMPL / SMPL-X `.pkl` body models must be downloaded from the MPI sites (free academic license). GVHMR's README tells you where to put them. Do not commit those files here.

## 3. Back on the Mac

```bash
python scripts/ingest_smpl.py \
  --motion output/<stem>_motion.npz \
  --gvhmr ~/Downloads/gvhmr_smpl.npz
```

`smpl_ready` in the sidecar JSON should flip to `true`.

## 4. If Colab quota is gone

Use a paid markerless service (DeepMotion, QuickMagic, Rokoko Vision), download FBX, and skip ingest — go straight to [MAC_BLENDER.md](MAC_BLENDER.md).
