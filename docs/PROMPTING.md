# Prompting MiniMax H3 (Ref2VA)

Official full-reference shape, from [MiniMax-H3 `VIDEO_PROMPT_WRITING_GUIDE_ref_en.md`](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md). Six sections, in this order. Vague “use these images” prompts waste Ref2VA.

Tags are 1-based per type and must match upload order: `<Picture 1>` is the first `reference_images` item, `<Video 1>` the first video, `<Audio 1>` the first standalone audio (or a video soundtrack if `include_audio` is on — MiniMax presents soundtrack `<Audio j>` immediately before its `<Video k>`).

## Template (fill in the placeholders)

```
subject_definitions:
<Subject 1> is the person matching <Picture 1>, preserving face, hair, and clothing.
<Picture 1> is a character still used as the identity reference for <Subject 1>.

summary:
[reference generation] A single-shot clip of <Subject 1> in [SETTING_PLACEHOLDER].

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - identity, clothing, and proportions from <Picture 1>.

detailed_description:
Live-action, cinematic.
[Shot 1] <Subject 1> from <Picture 1> stands in [SETTING_PLACEHOLDER]. The camera [CAMERA_PLACEHOLDER] as they [ACTION_PLACEHOLDER].

overall_soundscape:
[ROOM_TONE_PLACEHOLDER]

non_diegetic_music:
N/A
```

## Retention markers

Visible (`<Subject N>`, `<Picture N>`, `<Video N>`): `fully_preserved` / `partially_preserved` / `attribute_transfer` / `weak_reference`.

Audio (`<Audio N>`): `fully_copy` (1:1 final track — required for lip-sync to a source take) / `partially_copy` / `reference` (timbre only) / `weak_reference`.

`summary` starts with a square-bracketed task type: `[reference generation]`, `[video editing]`, `[video continuation]`, `[keyframe completion]`, `[audio reuse]`, `[audio reference]`, combined with ` + `.

## What not to do

- Do not re-use an I2VA first-frame alignment line. That is the FL2VA sibling.
- Do not dump a character sheet without assigning each file a job.
- Do not send audio as the only reference.
- Do not put OC trigger words in this worker repo. Those belong in the parent GUI (`h3_ref2va_gui.py` / `run-ref2va.ps1` on 7865).
