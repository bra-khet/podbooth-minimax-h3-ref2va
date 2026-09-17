# TODO

## Phase D (done)

- [x] QUEUE endpoint `podbooth-minimax-h3-ref2va` on AP-JP-1 + volume `pd2no1miqa`. Proof job COMPLETED (`r2v-test-startpng-5s.mp4`). Workers drained max=0.
- [x] Kitchen INT8 does **not** run on AP-JP-1 CUDA 12.8 H100. Live endpoint env is `COMFY_EXTRA_ARGS=--disable-auto-launch`. Send `use_kitchen_attention: false`.

## Later (not v1)

- [ ] Optional Ref2V 4-step turbo LoRA on the volume (`minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors`) once prototyping cost matters.
- [ ] Parent Gradio `h3_ref2va_gui.py` + `run-ref2va.ps1` on **7865** (Phase E). Default Kitchen **off** until a CUDA 13 host exists.
- [ ] Surface Comfy `execution_error` in the handler instead of `"No video could be found."` (needs an image tag bump).
- [ ] Friend-auth tokens if this booth is ever `share=True` — same note as the I2V sibling `TODO.md`.
