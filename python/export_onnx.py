"""Export Moebius student UNet + SD VAE encoder/decoder to ONNX (static 512x512).

Outputs to ./models/*.onnx and verifies numeric parity vs PyTorch.
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch import nn

MOEBIUS_REPO = "/tmp/Moebius/Moebius"
VAE_DIR = "/tmp/Moebius/moebius-web/weights/PixelHacker/vae"
WEIGHTS_DIR = "/tmp/Moebius/Moebius-weights"
OUT_DIR = "/tmp/Moebius/moebius-web/models"

sys.path.insert(0, MOEBIUS_REPO)
os.chdir(MOEBIUS_REPO)


# ---- export wrappers (return plain tensors) ---------------------------------
class UNetExport(nn.Module):
    def __init__(self, removal_model):
        super().__init__()
        self.m = removal_model

    def forward(self, latent_in9, timesteps, input_ids):
        return self.m(latent_in9, timesteps, input_ids).sample


class VaeEncExport(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, image):
        # moments = quant_conv(encoder(x)) : (B,8,64,64) = [mean(4), logvar(4)]
        h = self.vae.encoder(image)
        moments = self.vae.quant_conv(h)
        return moments


class VaeDecExport(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latent):
        z = self.vae.post_quant_conv(latent)
        return self.vae.decoder(z)


def build_models(weight):
    from diffusers import AutoencoderKL
    from removal.v1_2.removal_model import build_removal_model, load_cfg, load_removal_model

    cfg = load_cfg(f"{MOEBIUS_REPO}/config/model_cfg/moebius.yaml")
    cfg["vae"]["model_dir"] = VAE_DIR

    removal_model = build_removal_model(cfg, 20).eval()
    load_removal_model(removal_model, weight, "cpu")
    vae = AutoencoderKL.from_pretrained(VAE_DIR).eval()
    return removal_model, vae, cfg


def amax(a, b):
    return float(np.abs(a - b).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", default=f"{WEIGHTS_DIR}/ft_places2/diffusion_pytorch_model.bin")
    ap.add_argument("--opset", type=int, default=18)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(0)

    removal_model, vae, cfg = build_models(args.weight)
    print("[exp] models built")

    import onnxruntime as ort

    # ---------------- VAE decoder ----------------
    dec = VaeDecExport(vae)
    lat = torch.randn(1, 4, 64, 64)
    with torch.no_grad():
        ref = dec(lat).numpy()
    dec_path = os.path.join(OUT_DIR, "vae_decoder.onnx")
    torch.onnx.export(
        dec, (lat,), dec_path, opset_version=args.opset,
        input_names=["latent"], output_names=["image"],
        dynamic_axes={"latent": {0: "B"}, "image": {0: "B"}},
    )
    sess = ort.InferenceSession(dec_path, providers=["CPUExecutionProvider"])
    got = sess.run(None, {"latent": lat.numpy()})[0]
    print(f"[exp] vae_decoder  exported. max|Δ| = {amax(ref, got):.3e}")

    # ---------------- VAE encoder ----------------
    enc = VaeEncExport(vae)
    img = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        ref = enc(img).numpy()
    enc_path = os.path.join(OUT_DIR, "vae_encoder.onnx")
    torch.onnx.export(
        enc, (img,), enc_path, opset_version=args.opset,
        input_names=["image"], output_names=["moments"],
        dynamic_axes={"image": {0: "B"}, "moments": {0: "B"}},
    )
    sess = ort.InferenceSession(enc_path, providers=["CPUExecutionProvider"])
    got = sess.run(None, {"image": img.numpy()})[0]
    print(f"[exp] vae_encoder  exported. max|Δ| = {amax(ref, got):.3e}")

    # ---------------- UNet ----------------
    unet = UNetExport(removal_model)
    latent9 = torch.randn(2, 9, 64, 64)
    timesteps = torch.tensor([999, 999], dtype=torch.int64)
    half = removal_model.num_embeddings // 2
    input_ids = torch.tensor(
        [list(range(half, removal_model.num_embeddings)), list(range(half))],
        dtype=torch.int64)
    with torch.no_grad():
        ref = unet(latent9, timesteps, input_ids).numpy()
    unet_path = os.path.join(OUT_DIR, "unet.onnx")
    torch.onnx.export(
        unet, (latent9, timesteps, input_ids), unet_path, opset_version=args.opset,
        input_names=["latent", "timesteps", "input_ids"], output_names=["noise"],
        dynamic_axes={"latent": {0: "B"}, "timesteps": {0: "B"},
                      "input_ids": {0: "B"}, "noise": {0: "B"}},
    )
    sess = ort.InferenceSession(unet_path, providers=["CPUExecutionProvider"])
    got = sess.run(None, {"latent": latent9.numpy(),
                          "timesteps": timesteps.numpy(),
                          "input_ids": input_ids.numpy()})[0]
    print(f"[exp] unet         exported. max|Δ| = {amax(ref, got):.3e}")

    # sizes
    for f in ("vae_encoder.onnx", "vae_decoder.onnx", "unet.onnx"):
        p = os.path.join(OUT_DIR, f)
        print(f"      {f:18s} {os.path.getsize(p)/1e6:8.2f} MB")


if __name__ == "__main__":
    main()
