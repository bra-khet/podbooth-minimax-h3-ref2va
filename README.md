# PodBooth MiniMax H3 Ref2VA

RunPod Serverless worker + Python client + local Gradio panel for **MiniMax Hailuo H3 reference-to-video**.

This is a **direct sibling** of [bra-khet/podbooth-minimax-h3](https://github.com/bra-khet/podbooth-minimax-h3) (I2V / FL2VA). Same booth bones, same Japan network volume, same CUDA 12.8 Comfy base. Split into two images so each worker stays lightweight: one DiT, one API graph, one endpoint. Do not mix `fl2va_*` and `ref2va_*` in one graph.

Architecture is still the Wan-booth clone:

```
local machine
  parent h3_ref2va_gui.py (run-ref2va.ps1 :7865)   # daily driver, later sprint
    ──►  RunPod /v2/{id}/run
sidecar booth_ui.py is a convenience panel on :7867, not the daily driver.

RunPod Serverless worker (GPU)
  entrypoint.sh  →  ComfyUI :8188  +  handler.py
  handler loads workflows/h3_r2v_api.json
  patches node inputs from job["input"] by class_type
  injects up to 9 LoadImage + 3 LoadVideo/GetVideoComponents + 3 LoadAudio
  waits on Comfy websocket
  returns {"video": "<base64 mp4>"} or {"error": "..."}
```

## Features

- Full Ref2VA pack: **9 images / 3 videos / 3 standalone audio / 12 mixed files**
- Path / URL / Base64 per slot. Prefer `/runpod-volume/…` — RunPod `/run` is **10 MB**
- Videos are demuxed with native `GetVideoComponents` (IMAGE frames + optional soundtrack)
- Flat LoRA list (not Wan high/low). Turbo LoRA is **not** provisioned yet
- **Comfy Kitchen Attention** (`--use-ck-attention` + `ModelAttentionBackend`) for ~30% step speedup with no quality LoRA. **Off on the live CUDA 12.8 AP-JP-1 endpoint** — Kitchen INT8 kernels need cu130; see `docs/KITCHEN_ATTENTION.md`.
- Native stereo audio; `disable_audio` to drop it
- Duration in **seconds** `{5, 10, 15}` at 24 fps, 17k+5 frame grid
- Canvas snapped to a multiple of **32**. Official default **1344×768**
- `ref_image_size`: `match` (faster) or `max` (2048 short edge, stronger identity)
- I2V / first-last jobs are rejected here. Send those to the FL2VA sibling.

## Do not download H3 weights into the image

Current Hub tag (never `:latest`): `brakhet/podbooth-minimax-h3-ref2va:v0.1.0-r2v-cu128`.

The Docker image only contains:

- CUDA **12.8** + PyTorch (`runpod/pytorch:1.2.0-cu1281-torch280-ubuntu2404`) + ComfyUI **v0.35.2**
- ffmpeg, ComfyUI-Manager, `runpod`, `websocket-client`
- `handler.py`, `entrypoint.sh`, `workflows/h3_r2v_api.json`, `extra_model_paths.yaml`

Exact filenames: [`docs/NETWORK_VOLUME.md`](docs/NETWORK_VOLUME.md).

This worker needs **one extra file** on the volume the I2V sibling already uses:

| File | Folder |
|---|---|
| `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | `models/diffusion_models/` |

Shared with I2V (already on the volume, do not re-download): Qwen3-VL NVFP4 text encoder, video VAE fp16, audio VAE fp32.

FL2VA and Ref2VA are **not interchangeable**. Do not point this graph at a `fl2va_*` file.

## Network volume (Japan only)

Pinned to **AP-JP-1** (MiniMax H3 Community License). Same volume as the I2V sibling. Serverless mounts at `/runpod-volume`. Download the Ref2VA INT8 file with `scripts/provision-volume.sh` **on a pod in AP-JP-1**.

## Suggested endpoint settings (testing)

| Setting | Value |
|---|---|
| Datacenter | **AP-JP-1 only** |
| GPU | H100 SXM (`ADA_80_PRO`). H200 SXM allowed as fallback. One card. |
| CUDA | **12.8** |
| Workers | min **0** / max **1** while testing. Drain I2V max to 0 first. |
| Execution timeout | **1200–1800 s** |
| Flashboot | off while testing |
| Container disk | **60 GB**. Weights are on the volume |

Client wait default is 2700 s.

## Payload size (12 refs)

RunPod `/run` rejects bodies over **10 MB**. Twelve stills plus videos as base64 will not fit. Put packs on `/runpod-volume/inputs/` or pass `https` URLs. The client refuses the POST if the JSON is over 10 MB and warns above 8 MB.

## Python client

```python
from generate_video_client import GenerateVideoClient

client = GenerateVideoClient(endpoint_id, api_key)
result = client.create_video_r2v(
    prompt=open("docs/PROMPTING.md").read(),  # six-section Ref2VA shape
    reference_images=["/runpod-volume/inputs/face.png", "/runpod-volume/inputs/outfit.png"],
    reference_videos=[],
    duration=5,
    width=1344,
    height=768,
    ref_image_size="match",
)
client.save_video_result(result, "outputs/r2v.mp4")
```

## Build notes

Windows can author this repo. The CUDA image is `linux/amd64` (Docker Desktop). Do not pull H3 checkpoints onto the workstation.

```bash
docker build --platform linux/amd64 -t brakhet/podbooth-minimax-h3-ref2va:v0.1.0-r2v-cu128 .
```

Do not add `wget` lines for H3 checkpoints or LoRAs.

## License / credits

MiniMax H3 Community License (territory restrictions — AP-JP-1 only). Comfy-Org templates. Sibling pattern: `bra-khet/podbooth-minimax-h3` ← `bra-khet/podbooth-wan` ← `wlsdml1114/generate_video`.
