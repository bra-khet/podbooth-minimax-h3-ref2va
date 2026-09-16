"""RunPod Serverless handler for MiniMax H3 Ref2VA (reference-to-video).

Sibling of the I2V/FL2VA worker (podbooth-minimax-h3). Same I/O spine
(path/url/base64 → stage into /ComfyUI/input → Comfy websocket → base64 mp4),
different DiT, different graph, full 9/3/3/12-ref pack.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import os
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any

import runpod
import websocket

from h3_graph import (
    DEFAULT_DURATION,
    DEFAULT_HEIGHT,
    DEFAULT_REF_IMAGE_SIZE,
    DEFAULT_SAMPLER,
    DEFAULT_STEPS,
    DEFAULT_WIDTH,
    apply_r2v_job,
    duration_to_length,
    has_end_image,
    has_reference_pack,
    has_start_image,
    snap_dim,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

server_address = os.getenv("SERVER_ADDRESS", "127.0.0.1")
client_id = str(uuid.uuid4())
WORKFLOW_R2V = os.getenv("H3_R2V_WORKFLOW", "/workflows/h3_r2v_api.json")
# BUG FIX: LoadImage combo 400
# Fix: LoadImage/LoadVideo/LoadAudio widgets are combos of files in Comfy's input
# folder. An absolute /runpod-volume path is rejected by POST /prompt with HTTP 400.
# Sync: podbooth-minimax-h3/handler.py process_input (I2V sibling)
COMFY_INPUT_DIR = os.getenv("COMFY_INPUT_DIR", "/ComfyUI/input")
VIDEO_EXTS = (".mp4", ".webm", ".mkv", ".mov")


def process_input(input_data: str, temp_dir: str, output_filename: str, input_type: str) -> str:
    """Resolve path / url / base64 to a basename inside Comfy's input folder."""
    os.makedirs(COMFY_INPUT_DIR, exist_ok=True)
    dest_name = f"{uuid.uuid4().hex}_{os.path.basename(output_filename)}"
    dest_path = os.path.join(COMFY_INPUT_DIR, dest_name)
    if input_type == "path":
        logger.info("Path input: %s", input_data)
        if not os.path.isfile(input_data):
            raise Exception(f"Input file not found: {input_data}")
        shutil.copy2(input_data, dest_path)
        logger.info("Staged %s -> %s", input_data, dest_path)
        return dest_name
    if input_type == "url":
        logger.info("URL input: %s", input_data)
        download_file_from_url(input_data, dest_path)
        return dest_name
    if input_type == "base64":
        logger.info("Base64 input")
        save_base64_to_file(input_data, COMFY_INPUT_DIR, dest_name)
        return dest_name
    raise Exception(f"Unsupported input type: {input_type}")


def download_file_from_url(url: str, output_path: str) -> str:
    try:
        urllib.request.urlretrieve(url, output_path)
        logger.info("Downloaded %s -> %s", url, output_path)
        return output_path
    except Exception as exc:
        raise Exception(f"URL download failed: {exc}") from exc


def save_base64_to_file(base64_data: str, temp_dir: str, output_filename: str) -> str:
    payload = base64_data
    if "," in payload and payload.strip().startswith("data:"):
        payload = payload.split(",", 1)[1]
    try:
        decoded = base64.b64decode(payload)
    except (binascii.Error, ValueError) as exc:
        raise Exception(f"Base64 decoding failed: {exc}") from exc
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
    with open(file_path, "wb") as handle:
        handle.write(decoded)
    logger.info("Saved Base64 input to %s", file_path)
    return file_path


def _guess_name(item: dict[str, Any], fallback: str) -> str:
    for key in ("filename", "name", "file"):
        value = item.get(key)
        if value:
            return os.path.basename(str(value))
    return fallback


def _resolve_ref_item(item: Any, task_id: str, fallback: str) -> str:
    if isinstance(item, str):
        if item.startswith("http://") or item.startswith("https://"):
            return process_input(item, task_id, fallback, "url")
        return process_input(item, task_id, os.path.basename(item) or fallback, "path")
    if not isinstance(item, dict):
        raise Exception(f"Reference item must be a path/url string or {{path|url|base64}} dict, got {type(item).__name__}")
    name = _guess_name(item, fallback)
    if "path" in item:
        return process_input(item["path"], task_id, name, "path")
    if "url" in item:
        return process_input(item["url"], task_id, name, "url")
    if "base64" in item:
        return process_input(item["base64"], task_id, name, "base64")
    raise Exception(f"Reference item {fallback} needs path, url, or base64")


def resolve_ref_list(items: Any, task_id: str, prefix: str, default_ext: str) -> list[str]:
    if not items:
        return []
    if not isinstance(items, list):
        raise Exception(f"{prefix} must be a list")
    staged: list[str] = []
    for index, item in enumerate(items):
        fallback = f"{prefix}_{index + 1}{default_ext}"
        staged.append(_resolve_ref_item(item, task_id, fallback))
    return staged


def resolve_video_list(items: Any, task_id: str) -> list[dict[str, Any]]:
    if not items:
        return []
    if not isinstance(items, list):
        raise Exception("reference_videos must be a list")
    staged: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        fallback = f"ref_video_{index + 1}.mp4"
        include_audio = False
        if isinstance(item, dict):
            include_audio = bool(item.get("include_audio", item.get("audio", False)))
        filename = _resolve_ref_item(item, task_id, fallback)
        staged.append({"file": filename, "include_audio": include_audio})
    return staged


def queue_prompt(prompt: dict[str, Any]) -> dict[str, Any]:
    url = f"http://{server_address}:8188/prompt"
    logger.info("Queueing prompt to %s", url)
    payload = json.dumps({"prompt": prompt, "client_id": client_id}).encode("utf-8")
    req = urllib.request.Request(url, data=payload)
    try:
        return json.loads(urllib.request.urlopen(req).read())
    except urllib.error.HTTPError as exc:
        # BUG FIX: Comfy /prompt 400 was opaque
        # Fix: include the response body so combo/schema errors are diagnosable
        # Sync: podbooth-minimax-h3/handler.py queue_prompt
        body = exc.read().decode("utf-8", errors="replace")
        logger.error("Comfy /prompt HTTP %s: %s", exc.code, body)
        raise Exception(f"Comfy /prompt HTTP {exc.code}: {body}") from exc


def get_history(prompt_id: str) -> dict[str, Any]:
    url = f"http://{server_address}:8188/history/{prompt_id}"
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read())


def _collect_media_paths(node_output: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for bucket in ("videos", "gifs", "images", "audio"):
        items = node_output.get(bucket) or []
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            fullpath = item.get("fullpath")
            if fullpath and os.path.isfile(fullpath):
                paths.append(fullpath)
                continue
            filename = item.get("filename")
            subfolder = item.get("subfolder") or ""
            folder_type = item.get("type") or "output"
            if not filename:
                continue
            candidate = os.path.join("/ComfyUI", folder_type, subfolder, filename)
            if os.path.isfile(candidate):
                paths.append(candidate)
    return paths


def wait_for_videos(ws: websocket.WebSocket, prompt: dict[str, Any]) -> list[str]:
    queued = queue_prompt(prompt)
    prompt_id = queued["prompt_id"]
    while True:
        raw = ws.recv()
        if not isinstance(raw, str):
            continue
        message = json.loads(raw)
        if message.get("type") != "executing":
            continue
        data = message.get("data") or {}
        if data.get("node") is None and data.get("prompt_id") == prompt_id:
            break

    history = get_history(prompt_id)[prompt_id]
    found: list[str] = []
    for node_output in (history.get("outputs") or {}).values():
        found.extend(_collect_media_paths(node_output))

    videos = [path for path in found if path.lower().endswith(VIDEO_EXTS)]
    return videos or found


def load_workflow(workflow_path: str) -> dict[str, Any]:
    with open(workflow_path, encoding="utf-8") as handle:
        return json.load(handle)


def _compose_prompt(job_input: dict[str, Any]) -> str:
    text = (job_input.get("prompt") or "").strip()
    if not text:
        raise Exception("prompt is required")
    negative = (job_input.get("negative_prompt") or "").strip()
    if negative:
        text = f"{text}\n\nAvoid: {negative}"
    return text


def _wait_http(timeout_s: int = 180) -> None:
    http_url = f"http://{server_address}:8188/"
    for attempt in range(timeout_s):
        try:
            urllib.request.urlopen(http_url, timeout=5)
            logger.info("HTTP connection successful (attempt %s)", attempt + 1)
            return
        except Exception as exc:
            logger.warning("HTTP connection failed (%s/%s): %s", attempt + 1, timeout_s, exc)
            time.sleep(1)
    raise Exception("Cannot connect to ComfyUI server. Please check if the server is running.")


def handler(job: dict[str, Any]) -> dict[str, Any]:
    job_input = job.get("input") or {}
    logger.info("Received job keys: %s", sorted(job_input.keys()))
    task_id = f"task_{uuid.uuid4()}"

    try:
        if has_start_image(job_input) or has_end_image(job_input):
            return {
                "error": (
                    "This worker is Ref2VA only. I2V / first-last jobs belong on the sibling "
                    "podbooth-minimax-h3 endpoint (FL2VA), not here."
                )
            }
        if not has_reference_pack(job_input):
            return {
                "error": (
                    "H3 Ref2VA requires a reference pack "
                    "(reference_images / reference_videos / mode=r2v)."
                )
            }

        image_files = resolve_ref_list(job_input.get("reference_images"), task_id, "ref_image", ".png")
        video_files = resolve_video_list(job_input.get("reference_videos"), task_id)
        audio_files = resolve_ref_list(job_input.get("reference_audios"), task_id, "ref_audio", ".wav")

        duration = job_input.get("duration", DEFAULT_DURATION)
        fps = int(job_input.get("fps", 24))
        if "length" in job_input and "duration" not in job_input:
            length = int(job_input["length"])
        else:
            length = duration_to_length(duration, fps=fps)

        width = snap_dim(job_input.get("width", DEFAULT_WIDTH))
        height = snap_dim(job_input.get("height", DEFAULT_HEIGHT))
        steps = int(job_input.get("steps", DEFAULT_STEPS))
        seed = int(job_input.get("seed", 42))
        sampler = job_input.get("sampler") or DEFAULT_SAMPLER
        loras = job_input.get("loras") or []
        disable_audio = bool(job_input.get("disable_audio", False))
        use_kitchen = job_input.get("use_kitchen_attention")
        if use_kitchen is None:
            use_kitchen = True
        text_prompt = _compose_prompt(job_input)

        template = load_workflow(WORKFLOW_R2V)
        prompt = apply_r2v_job(
            template,
            image_files=image_files,
            video_files=video_files,
            audio_files=audio_files,
            text_prompt=text_prompt,
            width=width,
            height=height,
            length=length,
            seed=seed,
            steps=steps,
            sampler=sampler,
            ref_image_size=job_input.get("ref_image_size", DEFAULT_REF_IMAGE_SIZE),
            unet_name=job_input.get("unet_name"),
            clip_name=job_input.get("clip_name"),
            video_vae_name=job_input.get("video_vae_name"),
            audio_vae_name=job_input.get("audio_vae_name"),
            loras=loras,
            disable_audio=disable_audio,
            use_kitchen_attention=bool(use_kitchen),
        )
        logger.info(
            "Patched Ref2VA graph: %sx%s length=%s steps=%s seed=%s images=%s videos=%s audios=%s kitchen=%s",
            width,
            height,
            length,
            steps,
            seed,
            len(image_files),
            len(video_files),
            len(audio_files),
            bool(use_kitchen),
        )

        _wait_http()
        ws_url = f"ws://{server_address}:8188/ws?clientId={client_id}"
        ws = websocket.WebSocket()
        max_attempts = 36
        for attempt in range(max_attempts):
            try:
                ws.connect(ws_url)
                logger.info("WebSocket connection successful (attempt %s)", attempt + 1)
                break
            except Exception as exc:
                logger.warning("WebSocket connection failed (%s/%s): %s", attempt + 1, max_attempts, exc)
                if attempt == max_attempts - 1:
                    raise Exception("WebSocket connection timeout (3 minutes)") from exc
                time.sleep(5)

        videos = wait_for_videos(ws, prompt)
        ws.close()
        if not videos:
            return {"error": "No video could be found."}

        with open(videos[0], "rb") as handle:
            encoded = base64.b64encode(handle.read()).decode("utf-8")
        return {"video": encoded}
    except Exception as exc:
        logger.exception("Handler failed")
        return {"error": str(exc)}


runpod.serverless.start({"handler": handler})
