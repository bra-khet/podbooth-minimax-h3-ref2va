# Comfy Kitchen Attention (default speed path)

Turbo LoRA is **not** on the volume for v1. Quality first; Kitchen Attention is the lossless-ish speedup.

## What it is

ComfyUI core (PR [#15479](https://github.com/Comfy-Org/ComfyUI/pull/15479), Aug 2026) added INT8 Q/K attention from the official `comfy-kitchen` package. ComfyUI **v0.35.2** already pins `comfy-kitchen==0.2.33`.

Two enable paths (this worker uses both):

1. CLI: `python main.py --use-ck-attention` (entrypoint default)
2. Graph node: `ModelAttentionBackend` with `attention: "comfy kitchen attention"` (baked in `workflows/h3_r2v_api.json`)

Do **not** also pass `--use-sage-attention`. Kitchen and Sage fight; one backend only.

## Why not turbo LoRA yet

PixelEasel's H3 comparison (Aug 2026): Kitchen cut step time ~30% with no visible quality loss vs 20-step pytorch. The turbo LoRA (4-step Ref2V / 8-step I2V) is a different trade: fewer steps, visible identity/detail drop.

When turbo is wanted later:

| Worker | File | Steps |
|---|---|---|
| This Ref2VA sibling | `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` | 4 |
| I2V / FL2VA sibling | `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` | 8 |

Pass it in `loras`. Do not mix the two files across workers.

## Override

Set `COMFY_EXTRA_ARGS=--disable-auto-launch` on the endpoint if a card cannot load the kernels. Job key `use_kitchen_attention: false` strips the graph node only; the CLI flag still applies unless you override `COMFY_EXTRA_ARGS`.

# CHANGED: live AP-JP-1 CUDA 12.8 hosts cannot run Kitchen INT8
# WHY: Phase D proof on NVIDIA H100 80GB HBM3 failed in
# `_attention_comfy_kitchen_int8_containers` → `prequantize_int8_attention`
# → `detect_k_anchor` with "CUDA driver version is insufficient for CUDA
# runtime version". Comfy logs "need pytorch with cu130". The I2V sibling
# (pytorch attention, no Kitchen) already proved INT8 DiT on this same pool.
# The live Ref2VA endpoint therefore sets `COMFY_EXTRA_ARGS=--disable-auto-launch`
# and proof jobs send `use_kitchen_attention: false`. Re-enable Kitchen only
# on a CUDA 13 host (or after a cu130 image bump).
