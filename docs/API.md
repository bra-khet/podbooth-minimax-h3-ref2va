# Job contract — PodBooth MiniMax H3 Ref2VA

Worker input is the RunPod `{"input": {…}}` body. Output is `{"video": "<base64 mp4>"}` or `{"error": "…"}`.

This image is **Ref2VA only**. A start/end frame (`image_path` / `end_image_*`) returns an error pointing at the I2V sibling.

## R2V

```json
{
  "input": {
    "mode": "r2v",
    "prompt": "subject_definitions:\n<Subject 1> is the person matching <Picture 1>.\n\nsummary:\n[reference generation] …",
    "reference_images": [
      {"path": "/runpod-volume/inputs/face.png"},
      {"url": "https://example.com/outfit.png"}
    ],
    "reference_videos": [
      {"path": "/runpod-volume/inputs/motion.mp4", "include_audio": false}
    ],
    "reference_audios": [],
    "width": 1344,
    "height": 768,
    "duration": 5,
    "fps": 24,
    "steps": 20,
    "seed": 42,
    "ref_image_size": "match",
    "disable_audio": false,
    "use_kitchen_attention": true,
    "loras": []
  }
}
```

Each ref item is one of `{path|url|base64}` (a bare string is treated as a path or URL). Order **is** the prompt contract: first image is `<Picture 1>`, first video is `<Video 1>`, first standalone audio is `<Audio 1>`. Video soundtracks (`include_audio: true`) sit on `ref_video_audio_N` and take an `<Audio j>` slot **before** that video in MiniMax's presentation order.

### Caps (official MiniMax)

| Kind | Cap |
|---|---|
| Images | 9 |
| Videos (2–15 s, ≥5 frames) | 3 |
| Standalone audio (2–15 s) | 3 |
| Mixed files (images + videos + video soundtracks + standalone audio) | **12** |
| Audio-only | invalid |

### Generation knobs

| Key | Default | Notes |
|---|---|---|
| `prompt` | required | Six-section Ref2VA shape. See `docs/PROMPTING.md`. |
| `negative_prompt` | `""` | No native negative. Non-empty values are appended as `Avoid: …`. |
| `width` / `height` | `1344` / `768` | Multiple of 32. Official landscape default. |
| `duration` | `5` | `{5,10,15}` → length 124 / 243 / 362. |
| `steps` | `20` | Official non-turbo. 4 if you later attach the Ref2V turbo LoRA. |
| `ref_image_size` | `match` | `match` scales refs down to canvas. `max` keeps 2048 short edge (slower, better identity). |
| `use_kitchen_attention` | `true` | Strip `ModelAttentionBackend` if false. CLI still has `--use-ck-attention`. |
| `loras` | `[]` | Flat `{name, strength}`. Not Wan pairs. |
| `disable_audio` | `false` | Drops CreateVideo audio input. |
| `cfg` | ignored | `BasicGuider`. |

Optional filename overrides: `unet_name`, `clip_name`, `video_vae_name`, `audio_vae_name`.

### Payload size

RunPod **`/run` = 10 MB**. `/runsync` = 20 MB. This worker is QUEUE `/run`. Twelve refs as base64 will not fit. Prefer volume paths or URLs. The Python client refuses the POST above 10 MB.

### Output

Success: `{ "video": "<base64 mp4>" }`. Failure: `{ "error": "…" }`.
