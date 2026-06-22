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

## Phase 2 results (ONNX export — DONE)
- torch.onnx.export (legacy tracer, opset 18) traced all 3 graphs cleanly. No op-coverage
  failures. The einsum/lambda/Conv3d ops all exported.
- Parity vs PyTorch (CPU EP): decoder 5.7e-5, unet 3.6e-6, encoder mean ch ~2e-2.
- FULL pipeline parity test (python/onnx_pipeline.py): reimplemented DDIM+CFG+9ch+scaling
  in numpy on the ONNX sessions, vs torch models with identical noise:
    final latents max|Δ| 0.149, decoded image mean|Δ| 0.0022, max 0.090 → visually identical.
  This validates the ENTIRE orchestration I'll port to TS.
- numpy DDIM vs diffusers DDIMScheduler: step max|Δ| 5e-7, timesteps identical. ✓

## DDIM constants for JS (validated)
- betas = linspace(sqrt(0.00085), sqrt(0.012), 1000)^2 ; alphas_cumprod = cumprod(1-betas)
- timesteps(20 steps) = [950,900,...,50,0]; strength 0.99 ⇒ drop first ⇒ [900,...,0] (19).
- ddim_step (eta=0, clip_sample=False):
    pred_x0 = (sample - sqrt(1-ac_t)*eps) / sqrt(ac_t)
    prev    = sqrt(ac_prev)*pred_x0 + sqrt(1-ac_prev)*eps
    ac_prev = alphas_cumprod[prev_t], or final_alpha_cumprod=1.0 when prev_t<0 (last step).
- noise_offset 0.0357: noise += 0.0357 * randn(B,4,1,1). (optional; small)

## Web pipeline recipe (numpy → TS)
1. resize image+mask to 512×512 (mask NEAREST, binarize ≥128).
2. img→[-1,1] CHW; masked = img*(1-mask).
3. encode img & masked → moments; take mean[:4] * 0.13025.
4. mask→64×64 NEAREST, 1ch.
5. latents = randn(1,4,64,64) [+ noise_offset].
6. loop t in timesteps: nine=cat([latents,mask64,maskedLat]); batch×2; unet;
   cfg = u + g*(c-u); latents = ddim_step.
7. decode(latents/0.13025); (x+1)/2; clip; → image.
8. paste: out*blur(mask) + (1-blur(mask))*orig.

## TODO / unknowns
- Verify ORT-Web WebGPU actually runs these ops on GPU (not silent CPU fallback),
  esp. einsum / GroupNorm / Conv3d (pos_conv). Conv3d is the riskiest on WebGPU.
- UNet fp32 = 907MB download. Consider fp16 export to ~450MB (precision risk in λ layers).
