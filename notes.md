# Notes / lab log

Running log of what I figure out. Newest at the bottom of each section.

## Environment
- macOS (darwin arm64), Apple Silicon. No CUDA → torch CPU/MPS build.
- System python 3.9.6; using `uv` to manage an isolated env.
- git-lfs 3.7.1 present. Weights cloned to `/tmp/Moebius/Moebius-weights`
  (pretrained, ft_celebahq, ft_ffhq, ft_places2 — each ~450MB fp32 `.bin`).
- Code repo at `/tmp/Moebius/Moebius`.

## Code map (what matters for the port)
- Entry: `infer/infer_moebius.py` → `infer/utils.py:build_pipeline`.
- Pipeline: `removal/v1_2/pipeline.py` (`RemovalSDXLPipeline_BatchMode`).
- Model wrapper: `removal/v1_2/removal_model.py` (`RemovalModel` = embedding + diff UNet).
- Core helpers: `utils_infer.py` (`encode_clean_latents`, `predict_noise`).
- UNet impl: `model_lib/nets/unet_lambda_prune_lite.py` (+ lambda layers under
  `model_lib/nets/layers/λ/vanillaλ.py`).
- Config: `config/model_cfg/moebius.yaml`.

## Key findings
- The CUDA/Triton `fla` dependency is ONLY imported in `model_lib/nets/layers/gla/gla.py`
  (the GLA teacher variant). Moebius's student UNet (lambda-DWConv) does not need it —
  must avoid importing `unet_gla` to keep the graph clean for export.
- "Prompt" conditioning is a plain `nn.Embedding(20, 3072)`. CFG uses fixed ids:
  cond=[0..9], uncond=[10..19]. So encoder_hidden_states is a constant per branch →
  can be precomputed and baked into the ONNX UNet as a constant, OR passed as input.
- 9-channel UNet input = cat([noisy_latents(4), resized_mask(1), masked_latents(4)], dim=1).
- CFG batches uncond+cond into one forward (batch dim ×2), then splits.
- einsum is used in the λ layers (linear attention). Supported in ONNX; need to check
  ORT-Web WebGPU coverage.

## TODO / unknowns
- Need the SD VAE weights (hustvl/PixelHacker/vae) — not in the Moebius weights repo.
- Confirm DDIM `scale_model_input` is identity (it is for DDIM) → can skip in TS.
- Confirm VAE scaling_factor value from the VAE config.
