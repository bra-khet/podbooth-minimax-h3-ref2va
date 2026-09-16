# INIT PROMPT — PodBooth MiniMax H3 Ref2VA
# Direct sibling of bra-khet/podbooth-minimax-h3 (I2V / FL2VA).
# Destination: this directory, GitHub bra-khet/podbooth-minimax-h3-ref2va,
# Docker Hub brakhet/podbooth-minimax-h3-ref2va. Never tag :latest.

You are building the **Ref2VA** serverless worker. Do not mutate the I2V sibling
image, graph, or endpoint. Same Japan volume, same CUDA 12.8 Comfy base, different
DiT (`minimax_h3_ref2va_pruned_int8_convrot.safetensors`) and graph
(`MiniMaxH3ReferenceToVideo`). The split exists so each image stays lightweight.

Read the I2V sibling in full first: handler staging fix, extra_model_paths,
entrypoint, Dockerfile. Copy the I/O spine. Rewrite patching for a full
9 / 3 / 3 / 12-ref pack. Official UI template:

https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_r2v.json

Kitchen Attention (`--use-ck-attention` + ModelAttentionBackend) is the v1 speed
path. Do not wget turbo LoRA unless asked. If turbo is added later, use the
Ref2V 4-step file, never the I2V fl2v 8-step file.

AP-JP-1 only. Never delete the network volume. Never bake weights. Never mix
FL2VA and Ref2VA in one graph. RunPod /run payload is 10 MB — prefer volume paths.
