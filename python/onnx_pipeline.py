"""Full inference pipeline reimplemented in numpy on top of the ONNX sessions.

This mirrors exactly what the JS/TS web app will do (DDIM loop, CFG, 9-ch assembly,
scaling, pre/post-processing), and validates it against the torch reference pipeline
using IDENTICAL initial noise so we can diff the final image.
"""
import os
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageFilter

MOEBIUS_REPO = "/tmp/Moebius/Moebius"
VAE_DIR = "/tmp/Moebius/moebius-web/weights/PixelHacker/vae"
WEIGHTS_DIR = "/tmp/Moebius/Moebius-weights"
MODELS = "/tmp/Moebius/moebius-web/models"
OUT = "/tmp/Moebius/moebius-web/reference_out"

SCALING_FACTOR = 0.13025
NUM_TRAIN_TIMESTEPS = 1000
BETA_START, BETA_END = 0.00085, 0.012


# ---------------- DDIM scheduler in numpy (to be reproduced in JS) -----------
def make_ddim(num_steps, num_train=1000, beta_start=BETA_START, beta_end=BETA_END):
    betas = np.linspace(beta_start**0.5, beta_end**0.5, num_train, dtype=np.float64) ** 2
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas)
    # diffusers: timesteps spaced by floor(num_train/num_steps), reversed
    step_ratio = num_train // num_steps
    timesteps = (np.arange(0, num_steps) * step_ratio).round()[::-1].astype(np.int64)
    return alphas_cumprod, timesteps


def ddim_step(model_output, t, sample, alphas_cumprod, prev_t, final_alpha_cumprod=1.0):
    """DDIM, eta=0, clip_sample=False (matches diffusers config)."""
    ac_t = alphas_cumprod[t]
    ac_prev = alphas_cumprod[prev_t] if prev_t >= 0 else final_alpha_cumprod
    beta_t = 1 - ac_t
    pred_x0 = (sample - np.sqrt(beta_t) * model_output) / np.sqrt(ac_t)
    pred_dir = np.sqrt(1 - ac_prev) * model_output
    return np.sqrt(ac_prev) * pred_x0 + pred_dir


# ---------------- preprocessing (mirrors pipeline.py) ------------------------
def resize_mult64(img, image_size=512):
    w, h = img.size
    if w < h:
        scale = image_size / w; wt, ht = image_size, int(h * scale)
    else:
        scale = image_size / h; wt, ht = int(w * scale), image_size
    wt, ht = wt // 64 * 64, ht // 64 * 64
    return img.resize((wt, ht), Image.Resampling.LANCZOS)


def main():
    weight = f"{WEIGHTS_DIR}/ft_places2/diffusion_pytorch_model.bin"
    image_path = f"{MOEBIUS_REPO}/data/images/0.png"
    mask_path = f"{MOEBIUS_REPO}/data/masks/000000.png"
    num_steps = 20
    guidance = 2.0
    noise_offset = 0.0357

    so = ort.SessionOptions()
    enc = ort.InferenceSession(f"{MODELS}/vae_encoder.onnx", so, providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(f"{MODELS}/vae_decoder.onnx", so, providers=["CPUExecutionProvider"])
    unet = ort.InferenceSession(f"{MODELS}/unet.onnx", so, providers=["CPUExecutionProvider"])

    # --- preprocess (force square 512 for the static graph) ---
    image = Image.open(image_path).convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
    mask = Image.open(mask_path).convert("L").resize((512, 512), Image.Resampling.NEAREST)
    mask = mask.point(lambda x: 0 if x < 128 else 255, "L")

    img_np = np.asarray(image).astype(np.float32) / 255.0 * 2 - 1      # [-1,1] HWC
    img_chw = img_np.transpose(2, 0, 1)[None]                          # 1,3,H,W
    m_np = (np.asarray(mask).astype(np.float32) / 255.0)               # H,W in {0,1}
    m_bin = (m_np >= 0.5).astype(np.float32)
    masked_img = img_chw * (1 - m_bin[None, None])                     # zero the hole

    # --- VAE encode (mode = mean channels) ---
    lat = enc.run(None, {"image": img_chw})[0][:, :4] * SCALING_FACTOR
    masked_lat = enc.run(None, {"image": masked_img})[0][:, :4] * SCALING_FACTOR

    # mask -> latent size (nearest, like F.interpolate default = nearest)
    Hl, Wl = lat.shape[-2:]
    mask_small = np.asarray(
        Image.fromarray((m_bin * 255).astype(np.uint8)).resize((Wl, Hl), Image.Resampling.NEAREST)
    ).astype(np.float32) / 255.0
    mask_small = mask_small[None, None]                                # 1,1,Hl,Wl

    # --- init noise (SAME as torch reference for parity) ---
    import torch
    torch.manual_seed(0)
    noise = torch.randn(lat.shape[0], 4, Hl, Wl)
    noise = noise + noise_offset * torch.randn(lat.shape[0], 4, 1, 1)
    noise = noise.numpy().astype(np.float32)

    alphas_cumprod, timesteps = make_ddim(num_steps)
    timesteps = timesteps[1:]  # strength 0.99 drops the first
    latents = noise.copy()

    half = 10
    uncond_ids = np.arange(half, 2 * half, dtype=np.int64)
    cond_ids = np.arange(0, half, dtype=np.int64)
    input_ids = np.stack([uncond_ids, cond_ids])  # (2,10)

    for i, t in enumerate(timesteps):
        prev_t = timesteps[i + 1] if i + 1 < len(timesteps) else -1
        nine = np.concatenate([latents, mask_small, masked_lat], axis=1)  # 1,9,H,W
        nine2 = np.concatenate([nine, nine], axis=0)                       # CFG batch=2
        ts = np.array([t, t], dtype=np.int64)
        noise_pred = unet.run(None, {"latent": nine2.astype(np.float32),
                                     "timesteps": ts, "input_ids": input_ids})[0]
        nu, nc = noise_pred[0:1], noise_pred[1:2]
        np_cfg = nu + guidance * (nc - nu)
        latents = ddim_step(np_cfg, int(t), latents, alphas_cumprod, int(prev_t))

    img_out = dec.run(None, {"latent": (latents / SCALING_FACTOR).astype(np.float32)})[0]
    img_out = (img_out + 1) / 2
    img_out = np.clip(img_out[0].transpose(1, 2, 0), 0, 1)
    out = Image.fromarray((img_out * 255).astype(np.uint8))

    # paste back (blurred mask blend) — same as pipeline _post_process(paste=True)
    m_img = mask.convert("RGB").filter(ImageFilter.GaussianBlur(radius=3))
    m_img = np.asarray(m_img).astype(np.float32) / 255.0
    base = np.asarray(image).astype(np.float32) / 255.0
    ours = np.asarray(out).astype(np.float32) / 255.0
    blended = ours * m_img + (1 - m_img) * base
    out_pasted = Image.fromarray((blended * 255).astype(np.uint8))

    os.makedirs(OUT, exist_ok=True)
    out.save(f"{OUT}/onnx_result_raw.png")
    out_pasted.save(f"{OUT}/onnx_result.png")
    print("[onnx] saved", f"{OUT}/onnx_result.png")

    # --- compare against torch reference run with IDENTICAL noise ---
    compare_to_torch(weight, img_chw, masked_img, mask_small, noise, timesteps,
                     alphas_cumprod, guidance, latents, out)


def compare_to_torch(weight, img_chw, masked_img, mask_small, noise, timesteps,
                     alphas_cumprod, guidance, onnx_latents, onnx_img):
    import torch
    sys.path.insert(0, MOEBIUS_REPO); os.chdir(MOEBIUS_REPO)
    from diffusers import AutoencoderKL
    from removal.v1_2.removal_model import build_removal_model, load_cfg, load_removal_model

    cfg = load_cfg(f"{MOEBIUS_REPO}/config/model_cfg/moebius.yaml"); cfg["vae"]["model_dir"] = VAE_DIR
    rm = build_removal_model(cfg, 20).eval(); load_removal_model(rm, weight, "cpu")
    vae = AutoencoderKL.from_pretrained(VAE_DIR).eval()

    with torch.no_grad():
        lat = vae.encode(torch.tensor(img_chw)).latent_dist.mean * SCALING_FACTOR
        mlat = vae.encode(torch.tensor(masked_img)).latent_dist.mean * SCALING_FACTOR
        latents = torch.tensor(noise)
        ms = torch.tensor(mask_small)
        input_ids = torch.tensor(np.stack([np.arange(10, 20), np.arange(0, 10)]))
        for i, t in enumerate(timesteps):
            prev_t = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            nine = torch.cat([latents, ms, mlat], dim=1)
            nine2 = torch.cat([nine, nine], dim=0)
            ts = torch.tensor([int(t), int(t)], dtype=torch.int64)
            npred = rm(nine2, ts, input_ids).sample
            nu, nc = npred[0:1], npred[1:2]
            cfg_pred = nu + guidance * (nc - nu)
            latents = torch.tensor(
                ddim_step(cfg_pred.numpy(), int(t), latents.numpy(), alphas_cumprod, int(prev_t)).astype(np.float32))
        timg = vae.decode(latents / SCALING_FACTOR).sample
        timg = ((timg + 1) / 2).clamp(0, 1)[0].permute(1, 2, 0).numpy()

    print(f"[cmp] final latents max|Δ| (onnx vs torch) = {np.abs(onnx_latents - latents.numpy()).max():.4e}")
    onnx_arr = np.asarray(onnx_img).astype(np.float32) / 255.0
    print(f"[cmp] decoded image max|Δ| (0..1)          = {np.abs(onnx_arr - timg).max():.4e}")
    print(f"[cmp] decoded image mean|Δ|                = {np.abs(onnx_arr - timg).mean():.4e}")


if __name__ == "__main__":
    main()
