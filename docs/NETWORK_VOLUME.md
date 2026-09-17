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
  loras/                  # test LoRAs (Phase C): copy from laptop + Civitai/source sidecars
  inputs/                 # stills / ref packs the client can address by path
  outputs/
```

Phase C copies **test LoRAs from the laptop** into `loras/` (not a Hub `hf download`). Keep the Civitai/source provenance next to each file (`.civitai.info` / Manager JSON, trained words, published SHA256). Do not bake LoRAs into the Docker image.

| File | Civitai | Trigger | SHA256 (published) |
|---|---|---|---|
| `zero-two-dance-mh3-e20-az420.safetensors` | [1819613 @ 3264304](https://civitai.com/models/1819613?modelVersionId=3264304) | `doing the zero-two dance` | `814D8C1E8897FF87FEE47E5DBB36D4518FBFD38A052952ACF37144D3565A15BB` |
| `NSFW_ANIME_V7_H3-step00019500.safetensors` | [2861135 @ 3286171](https://civitai.com/models/2861135?modelVersionId=3286171) (primary 19.5k) | `2d anime style` | `C69A8E719B6784A8E475004CD47D34D1DDEFBB5DAA2D7670632CD3B459490B8D` |

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
