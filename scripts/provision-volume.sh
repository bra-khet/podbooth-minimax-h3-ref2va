#!/bin/bash
# Download the Ref2VA INT8 DiT onto the Japan network volume.
# Run this ON a pod in AP-JP-1 with the volume mounted at /runpod-volume.
# Never run this on the laptop. Never wget these files into the Docker image.
# Does NOT re-download the shared TE/VAEs or the I2V FL2VA file.
set -euo pipefail

ROOT="${VOLUME_ROOT:-/runpod-volume}"
mkdir -p \
  "${ROOT}/models/diffusion_models" \
  "${ROOT}/models/text_encoders" \
  "${ROOT}/models/vae" \
  "${ROOT}/models/loras" \
  "${ROOT}/models/embeddings" \
  "${ROOT}/loras" \
  "${ROOT}/inputs" \
  "${ROOT}/outputs"

python3 -m pip install --no-cache-dir -U --break-system-packages "huggingface_hub[hf_transfer]" hf_transfer >/dev/null
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "Downloading Comfy-Org MiniMax-H3 Ref2VA INT8 into ${ROOT}/models ..."
hf download Comfy-Org/MiniMax-H3 \
  --include "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors" \
  --local-dir "${ROOT}/models"

echo "Checking shared I2V files are still present..."
for f in \
  "${ROOT}/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors" \
  "${ROOT}/models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" \
  "${ROOT}/models/vae/minimax_h3_video_vae_fp16.safetensors" \
  "${ROOT}/models/vae/minimax_h3_audio_vae_fp32.safetensors"
do
  if [ -f "$f" ]; then
    echo "OK $f"
  else
    echo "WARN missing shared file: $f"
  fi
done

echo "Volume layout:"
find "${ROOT}/models" "${ROOT}/inputs" -type f | sort
du -sh "${ROOT}/models"/* 2>/dev/null || true
echo "DONE (turbo LoRA not downloaded — Kitchen Attention is the v1 speed path)"
