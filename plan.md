# Plan: Moebius inpainting in the browser (ONNX + WebGPU)

Goal: run the Moebius 0.22B image-inpainting model fully client-side in a browser via
ONNX Runtime Web on the WebGPU backend, with a simple UI (upload image, paint a mask,
get an inpainted result).

## Strategy (from research.md)
- No text encoder: the "prompt" is an `nn.Embedding` lookup (Latent Categories Guidance).
  The conditional/unconditional embeddings are fixed constants → bake them in.
- Export 3 ONNX graphs: VAE encoder, student UNet, VAE decoder.
- Reimplement the orchestration in TS: DDIM loop, CFG double-pass, 9-channel input
  assembly (noisy latent[4] + mask[1] + masked-image latent[4]), pre/post-processing.
- WebGPU is mandatory for usable speed.

## Model facts (confirmed from code)
- UNet: `UNet2DLambdaDWConvMixFFNConditionModel_prune_down_mid_up_block_8x8`
  - in_channels=9, out_channels=4, sample_size=64 (512/8)
  - block_out_channels [320,640,1280], 3 down / no mid / 3 up blocks
  - encoder_hid_dim=3072, cross_attention_dim=768, num_embeddings=20
  - LλMI blocks: linear-attention via einsum + depthwise-separable convs
- Embedding: num_embeddings=20, dim=3072. cond ids=[0..9], uncond ids=[10..19].
  encoder_hidden_states shape = (B, 10, 3072).
- VAE: SD f8d4 AutoencoderKL (from hustvl/PixelHacker/vae), scaling_factor ~0.18215.
- Scheduler: DDIM beta_start=0.00085 beta_end=0.012 scaled_linear, 1000 train ts,
  clip_sample=False. Default 20 steps, strength 0.99 → 19 actual steps.
- CFG default guidance 2.0 (README infer example) / 2.5 (argparse default).
- Weights available: pretrained, ft_celebahq, ft_ffhq, ft_places2 (~450MB fp32 each).

## Inference flow to port (from pipeline.py + utils_infer.py)
1. Preprocess: resize so short side=512, round to multiple of 64; binarize mask.
2. image → [-1,1]; masked_image = image*(1-mask).
3. VAE encode image → latents; encode masked_image → masked_latents; *scaling_factor.
4. mask → interpolate to latent size (H/8, W/8), 1 channel.
5. init noisy_latents = randn (strength≈1) [+ noise_offset].
6. DDIM loop over timesteps:
   - scale_model_input (DDIM = identity)
   - build 9ch input, duplicate for CFG (uncond+cond), run UNet
   - CFG: uncond + scale*(cond-uncond)
   - scheduler.step → next latents
7. VAE decode(latents / scaling_factor) → image, (img+1)/2.
8. Post: resize back, paste original outside mask (gaussian-blurred mask blend).

## Phases / tasks
1. [ ] Python env + reference inference (capture ground-truth + intermediates).  (task #1)
2. [ ] ONNX export of UNet + VAE enc/dec, numeric parity check.                  (task #2)
3. [ ] Web app: ORT-Web WebGPU + TS orchestration + UI.                          (task #3)

## Open risks
- ONNX export of the diffusers UNet subclass tracing cleanly (main schedule risk).
- ORT-Web WebGPU kernel coverage for einsum / GroupNorm (else CPU fallback = slow).
- fp32 vs fp16 numerics in lambda layers (precision-sensitive).
- Model size on mobile (ArrayBuffer / WebGPU buffer ceilings).

## Decisions log
- (pending) Which checkpoint to ship first: planning ft_places2 (natural scenes) as
  the general default; celebahq/ffhq are portrait-specialized.
