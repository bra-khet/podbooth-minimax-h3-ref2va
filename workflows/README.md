# Workflows

| File | Mode | Status |
|---|---|---|
| `h3_r2v_api.json` | Ref2VA — up to 9 images / 3 videos / 3 audio / 12 mixed | **v1** |

API format (`{ "NODE_ID": { "class_type", "inputs" } }`), flattened from:

https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_r2v.json

Flatten drops UI-only nodes (MarkdownNote, ResolutionSelector, Primitive*, ComfySwitchNode, `control_after_generate`). Handler injects LoadImage / LoadVideo+GetVideoComponents / LoadAudio per job. Autogrow keys are dotted and 0-based (`ref_images.ref_image_0`) because nested dicts are ignored on POST `/prompt` (Comfy-Org/ComfyUI#15667).

I2V / FL2VA is a **sibling image**, not a second graph in this repo.
