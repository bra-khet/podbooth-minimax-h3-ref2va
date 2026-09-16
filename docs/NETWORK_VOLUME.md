# Network volume layout — MiniMax H3 Ref2VA

Sibling of the I2V worker. **Same** RunPod network volume, **same** AP-JP-1 pin, **same** mount `/runpod-volume`. Weights stay on the volume, never in the Docker image.

The I2V stack (~41 GB) is already here. This worker adds **one** pruned INT8 Ref2VA DiT (~21 GB). Shared: QwenVL text encoder + both VAEs. Do not replace the FL2VA file. Do not download BF16.

Turbo LoRA is **not** downloaded yet. Speed for now is Comfy Kitchen Attention (see `docs/KITCHEN_ATTENTION.md`). If turbo is added later, use the **Ref2V** file `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` (4 steps) — never the I2V `fl2v` 8-step turbo.

## Folder tree

```
/runpod-volume/
  models/
    diffusion_models/     # FL2VA (I2V sibling) + Ref2VA (this worker). Not interchangeable.
    text_encoders/        # Qwen3-VL MiniMax pack (shared)
    vae/                  # video VAE + audio VAE (shared)
    loras/
    embeddings/
  loras/
  inputs/                 # stills / ref packs the client can address by path
  outputs/
```

## This worker — download this one extra file

| Role | Exact filename | Size | Hugging Face |
|---|---|---|---|
| Ref2VA diffusion | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | 21 GB | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors) |

Already on the volume from the I2V sibling (do not re-download):

| Role | Exact filename |
|---|---|
| Text encoder | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` |
| Video VAE | `minimax_h3_video_vae_fp16.safetensors` |
| Audio VAE | `minimax_h3_audio_vae_fp32.safetensors` |
| FL2VA DiT | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` (I2V sibling only) |

Projected used after this add: **~62 GB of 100 GB**. BF16 Ref2VA (66 GB) does **not** fit next to the I2V stack. Expand the volume before considering BF16.

On-volume destination:

```
/runpod-volume/models/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors
```

Run `scripts/provision-volume.sh` **on a pod in AP-JP-1**, not on the laptop.

## Optional turbo (later, not now)

| File | Steps |
|---|---|
| `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` | 4 |

Drop into `/runpod-volume/loras/` and name it in `loras` when you want it. Quality drops vs 20-step Kitchen.
