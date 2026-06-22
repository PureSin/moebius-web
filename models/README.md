---
license: apache-2.0
library_name: onnx
pipeline_tag: image-to-image
base_model: hustvl/Moebius
tags:
  - image-inpainting
  - inpainting
  - diffusion
  - onnx
  - onnxruntime-web
  - webgpu
  - in-browser
language:
  - en
---

# Moebius — ONNX (browser / WebGPU)

ONNX exports of the [Moebius](https://huggingface.co/hustvl/Moebius) image-inpainting model
([hustvl/Moebius](https://github.com/hustvl/Moebius), ECCV'26; 0.22B parameters), for running in
a web browser with [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) on the WebGPU
backend.

Moebius conditions on a learned embedding table rather than a text encoder, so there is no
tokenizer or text model to export. The export is three graphs — VAE encoder, UNet, VAE decoder —
and the sampling loop (DDIM with classifier-free guidance) runs in JavaScript.

## Files

| File | Graph | Input → Output | Size (fp32) |
|------|-------|----------------|-------------|
| `unet.onnx` | student denoiser (`RemovalModel`: embedding + lambda-DWConv UNet) | `latent (B,9,64,64)`, `timesteps (B,)`, `input_ids (B,10)` → `noise (B,4,64,64)` | ~907 MB |
| `vae_encoder.onnx` | SD VAE encoder | `image (B,3,512,512)` → `moments (B,8,64,64)` | ~137 MB |
| `vae_decoder.onnx` | SD VAE decoder | `latent (B,4,64,64)` → `image (B,3,512,512)` | ~198 MB |

- Exported at a **static 512×512** resolution (64×64 latent). The model's cross-attention uses a
  relative-position embedding tied to the trained resolution, so spatial size is fixed.
- The learned-embedding "prompt" conditioning stays inside `unet.onnx` as an `nn.Embedding(20, 3072)`
  gather. For classifier-free guidance: `input_ids` rows `[0..9]` = conditional, `[10..19]` = unconditional.

## Pipeline notes (must match for correct output)

- **VAE `scaling_factor = 0.13025`** (this is a custom VAE — *not* the usual SD `0.18215`).
  Encode: `latent = mean(moments[:, :4]) * 0.13025`. Decode: feed `latent / 0.13025`.
- 9-channel UNet input = `concat([noisy_latent(4), mask(1), masked_image_latent(4)], dim=1)`.
- Scheduler: DDIM, `beta_start=0.00085`, `beta_end=0.012`, `scaled_linear`, 1000 train steps,
  `clip_sample=false`. 20 steps with `strength≈0.99` ⇒ 19 actual steps.
- VAE encoder source: [`hustvl/PixelHacker`](https://huggingface.co/hustvl/PixelHacker) `vae/`.

A reference TypeScript implementation (DDIM loop, CFG, 9-channel assembly, pre/post-processing)
that loads these files lives in the accompanying web demo.

## Precision

These are fp32 exports, for numeric parity with the reference pipeline. Parity vs PyTorch on the
CPU execution provider: decoder `max|Δ|≈5.7e-5`, unet `≈3.6e-6`. A full-pipeline check against the
PyTorch reference (identical initial noise) gives a decoded-image `mean|Δ|≈0.0022`. fp16 halves the
download size, but can reduce quality in the lambda layers and is numerically unstable for this
VAE; validate before use.

## License & attribution

Licensed under **Apache 2.0**, inherited from the upstream
[hustvl/Moebius](https://huggingface.co/hustvl/Moebius). These artifacts are a format conversion
(PyTorch → ONNX) of the original weights; all model credit belongs to the original authors.

```bibtex
@misc{DuanAndXu2026Moebius,
  title  = {Moebius: 0.2B Lightweight Image Inpainting Framework with 10B-Level Performance},
  author = {Kangsheng Duan and Ziyang Xu and Wenyu Liu and Xiaohu Ruan and Xiaoxin Chen and Xinggang Wang},
  year   = {2026},
  eprint = {2606.19195},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url    = {https://arxiv.org/abs/2606.19195}
}
```
