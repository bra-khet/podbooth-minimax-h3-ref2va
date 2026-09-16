# PodBooth MiniMax H3 Ref2VA — RunPod Serverless worker
#
# Direct sibling of brakhet/podbooth-minimax-h3 (I2V / FL2VA). Same CUDA 12.8
# Comfy base, same Japan volume, different DiT and graph. Split exists so each
# image stays lightweight: one checkpoint family, one API graph, one endpoint.
#
# Do NOT wget MiniMax H3 checkpoints. Ref2VA + the shared encoder/VAEs live on
# the network volume. A baked H3 image is hundreds of GB.

FROM runpod/pytorch:1.2.0-cu1281-torch280-ubuntu2404

ARG COMFYUI_VERSION=v0.35.2

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV COMFY_DIR=/ComfyUI
ENV COMFY_PYTHON=python3
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        ffmpeg \
        wget \
        curl \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Native MiniMaxH3ReferenceToVideo + ModelAttentionBackend (Kitchen) ship in Comfy ≥ 0.30 / Aug 2026.
RUN git clone --depth 1 --branch "${COMFYUI_VERSION}" \
        https://github.com/comfyanonymous/ComfyUI.git /ComfyUI \
    && python3 -m pip install --no-cache-dir -r /ComfyUI/requirements.txt

# Manager is the usual operator toolbox; H3 Ref2VA itself is comfy-core.
RUN git clone --depth 1 \
        https://github.com/Comfy-Org/ComfyUI-Manager.git /ComfyUI/custom_nodes/ComfyUI-Manager \
    && if [ -f /ComfyUI/custom_nodes/ComfyUI-Manager/requirements.txt ]; then \
         python3 -m pip install --no-cache-dir -r /ComfyUI/custom_nodes/ComfyUI-Manager/requirements.txt; \
       fi

# Serverless I/O spine. Gradio lives on the laptop, not in this image.
# comfy-kitchen is already pinned by ComfyUI v0.35.2 requirements (0.2.33).
# Do NOT also install sageattention — Kitchen and Sage fight; one backend only.
RUN python3 -m pip install --no-cache-dir \
        "runpod>=1.7.0" \
        "websocket-client>=1.7.0" \
        "requests>=2.31.0"

COPY handler.py /handler.py
COPY h3_graph.py /h3_graph.py
COPY extra_model_paths.yaml /ComfyUI/extra_model_paths.yaml
COPY workflows /workflows
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh \
    && mkdir -p /ComfyUI/input /ComfyUI/output /ComfyUI/temp

CMD ["/entrypoint.sh"]
