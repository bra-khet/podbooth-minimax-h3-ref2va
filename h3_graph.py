"""MiniMax H3 Ref2VA workflow helpers.

Find nodes by class_type (and optional title), never by hard-coded numeric IDs.
Inject up to 9 images / 3 videos / 3 audio (12 mixed) onto MiniMaxH3ReferenceToVideo
using the dotted Autogrow keys Comfy's API export actually honors
(`ref_images.ref_image_0`, not a nested dict — see Comfy-Org/ComfyUI#15667).
"""

from __future__ import annotations

import copy
from typing import Any

H3_FPS = 24
H3_LENGTH_MOD = 17
H3_LENGTH_OFFSET = 5  # trained grid is 17k + 5
H3_SIZE_MULTIPLE = 32
H3_MIN_DIM = 32
ALLOWED_DURATIONS = (5, 10, 15)
DEFAULT_UNET = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
DEFAULT_CLIP = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
DEFAULT_VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
DEFAULT_AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
DEFAULT_WIDTH = 1344
DEFAULT_HEIGHT = 768
DEFAULT_DURATION = 5
DEFAULT_STEPS = 20
DEFAULT_SAMPLER = "res_multistep"
DEFAULT_REF_IMAGE_SIZE = "match"
MAX_REF_IMAGES = 9
MAX_REF_VIDEOS = 3
MAX_REF_AUDIOS = 3
MAX_MIXED_REFS = 12
REF_IMAGE_SIZES = ("match", "max")


def find_nodes(prompt: dict[str, Any], class_type: str, title: str | None = None) -> list[str]:
    """Return node ids whose class_type matches, optionally filtered by _meta.title."""
    hits: list[str] = []
    for node_id, node in prompt.items():
        if not isinstance(node, dict):
            continue
        if node.get("class_type") != class_type:
            continue
        if title is not None:
            meta_title = (node.get("_meta") or {}).get("title")
            if meta_title != title:
                continue
        hits.append(str(node_id))
    return hits


def find_one(prompt: dict[str, Any], class_type: str, title: str | None = None) -> str:
    hits = find_nodes(prompt, class_type, title=title)
    if not hits:
        label = class_type if title is None else f"{class_type} ({title})"
        raise KeyError(f"No node with class_type {label!r} in workflow")
    return hits[0]


def duration_to_length(duration_seconds: float, fps: int = H3_FPS) -> int:
    """Convert seconds to H3 frame count: snap up onto the 17k+5 grid at 24 fps.

    Official template expression:
        max(5, round(a * 24)) + (5 - (max(5, round(a * 24)) % 17)) % 17
    5s → 124, 10s → 243, 15s → 362.
    """
    duration = max(1.0, min(float(duration_seconds), 15.0))
    raw = max(5, int(round(duration * fps)))
    add = (H3_LENGTH_OFFSET - (raw % H3_LENGTH_MOD)) % H3_LENGTH_MOD
    return raw + add


def snap_dim(value: Any, multiple: int = H3_SIZE_MULTIPLE, minimum: int = H3_MIN_DIM) -> int:
    """H3 canvas is a multiple of 32 (not Wan's multiple of 16)."""
    numeric = float(value)
    snapped = int(round(numeric / multiple) * multiple)
    return max(minimum, snapped)


def _set_input(prompt: dict[str, Any], node_id: str, key: str, value: Any) -> None:
    prompt[node_id].setdefault("inputs", {})[key] = value


def _clear_ref_slots(inputs: dict[str, Any]) -> None:
    for key in list(inputs.keys()):
        if key.startswith("ref_images.") or key.startswith("ref_videos.") or key.startswith("ref_video_audios.") or key.startswith("ref_audios."):
            inputs.pop(key, None)
        if key in {"ref_images", "ref_videos", "ref_video_audios", "ref_audios"}:
            inputs.pop(key, None)


def mixed_ref_count(
    n_images: int,
    n_videos: int,
    n_audios: int,
    n_video_soundtracks: int = 0,
) -> int:
    """Every staged file counts toward MiniMax's 12-file mixed cap.

    A reference video with its paired soundtrack is two files.
    """
    return int(n_images) + int(n_videos) + int(n_audios) + int(n_video_soundtracks)


def validate_ref_pack(
    images: list[Any],
    videos: list[Any],
    audios: list[Any],
    video_soundtracks: int = 0,
) -> None:
    n_images = len(images)
    n_videos = len(videos)
    n_audios = len(audios)
    if n_images == 0 and n_videos == 0:
        raise ValueError(
            "Ref2VA needs at least one reference image or video. Audio-only jobs are invalid."
        )
    if n_images > MAX_REF_IMAGES:
        raise ValueError(f"Too many reference images ({n_images}). Cap is {MAX_REF_IMAGES}.")
    if n_videos > MAX_REF_VIDEOS:
        raise ValueError(f"Too many reference videos ({n_videos}). Cap is {MAX_REF_VIDEOS}.")
    if n_audios > MAX_REF_AUDIOS:
        raise ValueError(f"Too many standalone reference audios ({n_audios}). Cap is {MAX_REF_AUDIOS}.")
    mixed = mixed_ref_count(n_images, n_videos, n_audios, video_soundtracks)
    if mixed > MAX_MIXED_REFS:
        raise ValueError(
            f"Too many mixed references ({mixed} files). Cap is {MAX_MIXED_REFS} "
            f"across images + videos + video soundtracks + standalone audio "
            f"(also {MAX_REF_IMAGES}/{MAX_REF_VIDEOS}/{MAX_REF_AUDIOS} per type)."
        )


def apply_r2v_job(
    prompt: dict[str, Any],
    *,
    image_files: list[str],
    video_files: list[dict[str, Any]] | None = None,
    audio_files: list[str] | None = None,
    text_prompt: str,
    width: int,
    height: int,
    length: int,
    seed: int,
    steps: int,
    sampler: str | None = None,
    ref_image_size: str | None = None,
    unet_name: str | None = None,
    clip_name: str | None = None,
    video_vae_name: str | None = None,
    audio_vae_name: str | None = None,
    loras: list[dict[str, Any]] | None = None,
    disable_audio: bool = False,
    use_kitchen_attention: bool = True,
) -> dict[str, Any]:
    """Return a patched copy of the R2V API graph for one job.

    image_files / audio_files are LoadImage / LoadAudio basenames already staged
    into Comfy's input folder. video_files is a list of
    ``{"file": basename, "include_audio": bool}``.
    Picture/Video/Audio tags in the prompt are 1-based in upload order.
    """
    videos = list(video_files or [])
    audios = list(audio_files or [])
    soundtrack_count = sum(1 for item in videos if item.get("include_audio"))
    validate_ref_pack(image_files, videos, audios, soundtrack_count)

    graph = copy.deepcopy(prompt)
    r2v_id = find_one(graph, "MiniMaxH3ReferenceToVideo")
    r2v_inputs = graph[r2v_id].setdefault("inputs", {})
    _clear_ref_slots(r2v_inputs)

    _set_input(graph, r2v_id, "prompt", text_prompt)
    _set_input(graph, r2v_id, "width", width)
    _set_input(graph, r2v_id, "height", height)
    _set_input(graph, r2v_id, "length", length)
    size = (ref_image_size or DEFAULT_REF_IMAGE_SIZE).strip().lower()
    if size not in REF_IMAGE_SIZES:
        raise ValueError(f"ref_image_size must be one of {REF_IMAGE_SIZES}, got {ref_image_size!r}")
    _set_input(graph, r2v_id, "ref_image_size", size)

    for index, filename in enumerate(image_files):
        node_id = f"ref_img_{index}"
        graph[node_id] = {
            "inputs": {"image": filename},
            "class_type": "LoadImage",
            "_meta": {"title": f"Reference Image {index + 1}"},
        }
        # CHANGED: dotted Autogrow keys, 0-based
        # WHY: nested {"ref_images": {"ref_image_1": ...}} is ignored on POST /prompt
        # (Comfy-Org/ComfyUI#15667). Official API exports use ref_images.ref_image_0.
        r2v_inputs[f"ref_images.ref_image_{index}"] = [node_id, 0]

    for index, item in enumerate(videos):
        filename = item.get("file") or item.get("path") or item.get("name")
        if not filename:
            raise ValueError(f"reference video {index + 1} is missing a staged filename")
        load_id = f"ref_vid_{index}"
        split_id = f"ref_vid_split_{index}"
        graph[load_id] = {
            "inputs": {"file": filename},
            "class_type": "LoadVideo",
            "_meta": {"title": f"Reference Video {index + 1}"},
        }
        # LoadVideo outputs VIDEO. MiniMaxH3ReferenceToVideo.ref_videos wants IMAGE frames.
        graph[split_id] = {
            "inputs": {"video": [load_id, 0]},
            "class_type": "GetVideoComponents",
            "_meta": {"title": f"Split Reference Video {index + 1}"},
        }
        r2v_inputs[f"ref_videos.ref_video_{index}"] = [split_id, 0]
        if item.get("include_audio"):
            r2v_inputs[f"ref_video_audios.ref_video_audio_{index}"] = [split_id, 1]

    for index, filename in enumerate(audios):
        node_id = f"ref_aud_{index}"
        graph[node_id] = {
            "inputs": {"audio": filename},
            "class_type": "LoadAudio",
            "_meta": {"title": f"Reference Audio {index + 1}"},
        }
        r2v_inputs[f"ref_audios.ref_audio_{index}"] = [node_id, 0]

    noise_id = find_one(graph, "RandomNoise")
    _set_input(graph, noise_id, "noise_seed", int(seed))
    graph[noise_id]["inputs"].pop("control_after_generate", None)

    sched_id = find_one(graph, "BasicScheduler")
    _set_input(graph, sched_id, "steps", int(steps))

    if sampler:
        sampler_id = find_one(graph, "KSamplerSelect")
        _set_input(graph, sampler_id, "sampler_name", sampler)

    if unet_name:
        _set_input(graph, find_one(graph, "UNETLoader"), "unet_name", unet_name)
    if clip_name:
        _set_input(graph, find_one(graph, "CLIPLoader"), "clip_name", clip_name)
    if video_vae_name:
        _set_input(graph, find_one(graph, "VAELoader", title="Video VAE"), "vae_name", video_vae_name)
    if audio_vae_name:
        _set_input(graph, find_one(graph, "VAELoader", title="Audio VAE"), "vae_name", audio_vae_name)

    apply_kitchen_attention(graph, enabled=use_kitchen_attention)
    apply_loras(graph, loras or [])

    if disable_audio:
        create_id = find_one(graph, "CreateVideo")
        graph[create_id]["inputs"].pop("audio", None)

    return graph


def apply_kitchen_attention(prompt: dict[str, Any], enabled: bool = True) -> None:
    """Keep or strip ModelAttentionBackend. Kitchen is the default speed path (no quality LoRA)."""
    hits = find_nodes(prompt, "ModelAttentionBackend")
    if enabled:
        if not hits:
            return
        node_id = hits[0]
        _set_input(prompt, node_id, "attention", "comfy kitchen attention")
        return
    if not hits:
        return
    attn_id = hits[0]
    unet_id = find_one(prompt, "UNETLoader")
    for node_id, node in prompt.items():
        inputs = node.get("inputs") or {}
        for key, value in list(inputs.items()):
            if isinstance(value, list) and len(value) >= 1 and str(value[0]) == str(attn_id):
                inputs[key] = [unet_id, 0]
    prompt.pop(attn_id, None)


def apply_loras(prompt: dict[str, Any], loras: list[dict[str, Any]]) -> None:
    """Chain LoraLoaderModelOnly nodes after UNETLoader. Flat list, not Wan high/low pairs.

    Kitchen Attention (if present) stays downstream of the last LoRA so the patched
    model is what BasicGuider / BasicScheduler see.
    """
    cleaned: list[tuple[str, float]] = []
    for item in loras:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("lora") or item.get("lora_name")
        if not name or str(name).lower() in {"none", "undefined"}:
            continue
        strength = float(item.get("strength", item.get("strength_model", 1.0)))
        cleaned.append((str(name), strength))
    if not cleaned:
        return

    unet_id = find_one(prompt, "UNETLoader")
    prev = unet_id
    injected: list[str] = []
    for index, (name, strength) in enumerate(cleaned, start=1):
        node_id = f"lora_{index}"
        prompt[node_id] = {
            "inputs": {
                "model": [prev, 0],
                "lora_name": name,
                "strength_model": strength,
            },
            "class_type": "LoraLoaderModelOnly",
            "_meta": {"title": f"LoRA {index}"},
        }
        injected.append(node_id)
        prev = node_id

    last_lora = injected[-1]
    for node_id, node in prompt.items():
        if node_id in injected:
            continue
        inputs = node.get("inputs") or {}
        for key, value in list(inputs.items()):
            if isinstance(value, list) and len(value) >= 1 and str(value[0]) == str(unet_id):
                inputs[key] = [last_lora, 0]


def has_reference_pack(job_input: dict[str, Any]) -> bool:
    if job_input.get("mode") in {"r2v", "ref2v", "ref2va"}:
        return True
    for key in ("references", "reference_images", "reference_videos", "reference_audios"):
        value = job_input.get(key)
        if value:
            return True
    return False


def has_start_image(job_input: dict[str, Any]) -> bool:
    return any(key in job_input for key in ("image_path", "image_url", "image_base64"))


def has_end_image(job_input: dict[str, Any]) -> bool:
    return any(key in job_input for key in ("end_image_path", "end_image_url", "end_image_base64"))
