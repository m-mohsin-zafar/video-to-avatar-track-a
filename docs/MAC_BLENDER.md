# Mac: Mixamo Y-Bot in Blender

Do this **after** Colab fills SMPL, or after you download an FBX from a mocap SaaS.

## Install

- Blender 4.x Apple Silicon: https://www.blender.org/download/
- ffmpeg: `brew install ffmpeg`
- Mixamo Y-Bot: https://www.mixamo.com — character **Y Bot**, bind pose, download FBX

## Path A — you already have an FBX from DeepMotion / QuickMagic

1. File → Import → FBX (the motion)
2. File → Import → FBX (Y Bot) if the service did not include a mesh
3. If two armatures: select Y Bot, add Object Constraint → Copy Transforms / use Blender's Animation Retargeting addon
4. Place a floor plane, lock feet if they skate (graph editor, or Rokoko / Auto-Rig Pro if you have it)
5. Render → Eevee, output MP4 or PNG sequence + ffmpeg

## Path B — you have `smpl_*` in motion.npz

SMPL rotations are not Mixamo bone names. You need a retarget step:

1. Export SMPL as a mesh sequence or as an armature. Community tools:
   - `smplx` Python + a small Blender script that keys `body_pose` onto an SMPL rig
   - Rokoko Blender addon (free tier often enough) to retarget SMPL → Mixamo
2. Scale root translation by hip-height ratio so a short bot does not squat through the floor
3. Re-IK the feet after retarget

A minimal SMPL-to-Blender keyframing script is intentionally not bundled: it needs the licensed SMPL `.pkl`. Once you have that file locally (never commit it), you can drive the official SMPL add-on.

## Render settings that look decent on an Air

- Engine: Eevee
- Resolution: 1080×1920 if the source was portrait, else 1920×1080
- Frame rate: match `motion.json` fps (30 for the demo clip)
- Samples: Eevee default is enough for a clay / plastic bot

## Sanity checks

- Raised knee on the bot matches the plate
- Planted foot does not slide
- No 180 deg shoulder pop (quaternion continuity)
