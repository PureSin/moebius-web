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

## Phase 1 results (reference inference — DONE)
- Got the real pipeline running end-to-end on CPU (macOS, torch 2.7.1).
- Patches needed to load student on CPU/mac:
  - `model_lib/__init__.py`: wrapped teacher `unet_gla` import in try/except (needs `fla`).
  - Don't import `utils_train` (drags in orjson/library); `build_vae` is just
    `AutoencoderKL.from_pretrained(vae_dir)`.
- **VAE scaling_factor = 0.13025** (NOT the usual SD 0.18215!). Custom VAE.
  block_out_channels = [128,256,512,512] → vae_scale_factor 8. This MUST be hardcoded
  correctly in the JS port or colors/contrast will be wrong.
- removal_model params = 226.04M confirmed. load_state_dict: all keys matched.
- Perf: ~8.9 s/step on CPU (×19 steps + CFG ×2 = 38 UNet passes ≈ 2:48 total). WebGPU
  expected far faster. Confirms CPU/WASM is unusable; WebGPU is the whole game.
- Output saved to reference_out/reference_result.png — plausible inpaint. Mask convention:
  white(255) → 1 → region to inpaint (zeroed in masked_image); black → keep.
- num_inference_steps=20 with strength=0.99 → DDIM uses 19 steps (drops first).

## Parity strategy
- Won't try to reproduce torch RNG in JS. For PyTorch↔ONNX parity: dump identical input
  tensors and compare outputs. For the web app: generate noise with a seedable JS RNG;
  diffusion is robust to the particular noise draw, so visual results will be valid even
  if not bit-identical to the torch reference.
- DDIM `scale_model_input` is identity → skip in TS. Need to reproduce DDIM alphas/betas
  (scaled_linear, beta 0.00085→0.012, 1000 steps) and the DDIM step update in JS.

## Architecture: spatial size is FIXED (important!)
- Self-attn (attn1): MQSλ with `r=15` → local-context path (Conv3d pos_conv). Spatially
  dynamic, fine at any size.
- Cross-attn (attn2): MQCλ with NO `r` → global path → `rel_pos_emb` is an
  `nn.Parameter(n*n, m, dim_k, dim_u)` where n = per-block sample_size, m = 10. This is
  TIED to the trained spatial resolution. Different spatial size → wrong/oob indexing.
- ⇒ Export at STATIC 512×512 image (64×64 latent). Web app resizes user input to 512×512,
  inpaints, resizes result back + pastes. Square only. This is the benchmark resolution.

## ONNX export plan
- Three graphs, spatial static, batch dynamic where cheap:
  - vae_encoder: (B,3,512,512) → moments (B,8,64,64); JS uses mean=moments[:,:4]*sf.
  - unet (RemovalModel): (B,9,64,64), timesteps(B,), input_ids(B,10) → noise(B,4,64,64).
    Embedding (nn.Embedding 20×3072) stays IN the graph (cheap gather). CFG batches B=2.
  - vae_decoder: (B,4,64,64) → (B,3,512,512).
- scaling_factor = 0.13025 applied in JS (encode: latent*sf; decode: latent/sf).

## TODO / unknowns
- ORT-Web WebGPU coverage for einsum / GroupNorm / Conv3d (pos_conv!) — Conv3d may be a
  problem on WebGPU; watch for CPU fallback.
