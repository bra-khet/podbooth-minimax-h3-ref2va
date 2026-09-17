# TODO

## Phase C (done)

- [x] Download Ref2VA INT8 onto `pd2no1miqa` from an AP-JP-1 pod (`scripts/provision-volume.sh`). Confirm I2V files still present. Terminate the pod.
- [x] Copy **test LoRAs from the laptop** into `/runpod-volume/loras/` with Civitai/source provenance (`.civitai.info` / Manager JSON + SHA256 from the published version — not a naked `.safetensors`). Operator picks files at the start of that sprint. Do not bake LoRAs into the Docker image.

## Later (not v1)

- [ ] Optional Ref2V 4-step turbo LoRA on the volume (`minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors`) once prototyping cost matters. Kitchen Attention stays the lossless path.
- [ ] Parent Gradio `h3_ref2va_gui.py` + `run-ref2va.ps1` on **7865** (Phase E, after a proof mp4).
- [ ] Friend-auth tokens if this booth is ever `share=True` — same note as the I2V sibling `TODO.md`.
