"""Dump a fixed inference fixture (inputs + reference final latents) so the JS/TS port
can be verified in Node against the validated numpy/ONNX pipeline."""
import json
import os
import numpy as np
import onnxruntime as ort
from PIL import Image

MOEBIUS_REPO = "/tmp/Moebius/Moebius"
MODELS = "/tmp/Moebius/moebius-web/models"
OUT = "/tmp/Moebius/moebius-web/web/test/fixture"
SCALING_FACTOR = 0.13025
NOISE_OFFSET = 0.0357

import sys
sys.path.insert(0, "python")
from onnx_pipeline import make_ddim, ddim_step  # reuse validated math

os.makedirs(OUT, exist_ok=True)
enc = ort.InferenceSession(f"{MODELS}/vae_encoder.onnx", providers=["CPUExecutionProvider"])
unet = ort.InferenceSession(f"{MODELS}/unet.onnx", providers=["CPUExecutionProvider"])

steps, guidance = 20, 2.0
image = Image.open(f"{MOEBIUS_REPO}/data/images/0.png").convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
mask = Image.open(f"{MOEBIUS_REPO}/data/masks/000000.png").convert("L").resize((512, 512), Image.Resampling.NEAREST)
mask = mask.point(lambda x: 0 if x < 128 else 255, "L")

img_np = np.asarray(image).astype(np.float32) / 255.0 * 2 - 1
img_chw = img_np.transpose(2, 0, 1)[None]
m_bin = (np.asarray(mask).astype(np.float32) / 255.0 >= 0.5).astype(np.float32)
masked_img = img_chw * (1 - m_bin[None, None])

masked_lat = enc.run(None, {"image": masked_img})[0][:, :4] * SCALING_FACTOR
Hl = Wl = 64
mask_small = np.asarray(
    Image.fromarray((m_bin * 255).astype(np.uint8)).resize((Wl, Hl), Image.Resampling.NEAREST)
).astype(np.float32) / 255.0
mask_small = mask_small[None, None]

import torch
torch.manual_seed(0)
noise = torch.randn(1, 4, Hl, Wl)
noise = noise + NOISE_OFFSET * torch.randn(1, 4, 1, 1)
noise = noise.numpy().astype(np.float32)

ac, ts = make_ddim(steps)
ts = ts[1:]
latents = noise.copy()
input_ids = np.stack([np.arange(10, 20), np.arange(0, 10)]).astype(np.int64)
for i, t in enumerate(ts):
    prev_t = ts[i + 1] if i + 1 < len(ts) else -1
    nine = np.concatenate([latents, mask_small, masked_lat], axis=1)
    nine2 = np.concatenate([nine, nine], axis=0).astype(np.float32)
    npred = unet.run(None, {"latent": nine2, "timesteps": np.array([t, t], dtype=np.int64),
                            "input_ids": input_ids})[0]
    nu, nc = npred[0:1], npred[1:2]
    cfg = nu + guidance * (nc - nu)
    latents = ddim_step(cfg, int(t), latents, ac, int(prev_t)).astype(np.float32)

# dump (float32 little-endian raw)
masked_lat.astype("<f4").tofile(f"{OUT}/masked_lat.bin")  # (1,4,64,64)
mask_small.astype("<f4").tofile(f"{OUT}/mask64.bin")      # (1,1,64,64)
noise.astype("<f4").tofile(f"{OUT}/init_latents.bin")     # (1,4,64,64)
latents.astype("<f4").tofile(f"{OUT}/final_latents.bin")  # (1,4,64,64)
with open(f"{OUT}/meta.json", "w") as f:
    json.dump({"steps": steps, "guidance": guidance, "timesteps": [int(t) for t in ts],
               "scaling_factor": SCALING_FACTOR}, f, indent=2)
print("dumped fixture to", OUT)
print("final latents stats: mean=%.5f std=%.5f" % (latents.mean(), latents.std()))
