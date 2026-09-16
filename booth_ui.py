#!/usr/bin/env python3
"""Thin sidecar panel for the PodBooth MiniMax H3 Ref2VA worker.

The daily-driver booth will live in the parent repo: `h3_ref2va_gui.py` via
`.\run-ref2va.ps1` on port **7865**. This file stays as a worker-repo convenience
and defaults to 7867 so it cannot collide with the parent GUI or the I2V booth
on 7864.
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

from generate_video_client import GenerateVideoClient

APP_DIR = Path(__file__).resolve().parent
load_dotenv(APP_DIR / ".env")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

OUTPUT_DIR = APP_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_PROMPT = """subject_definitions:
<Subject 1> is the person matching <Picture 1>, preserving face, hair, and clothing.

summary:
[reference generation] A single-shot clip of <Subject 1> in [SETTING_PLACEHOLDER].

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - identity from <Picture 1>.

detailed_description:
Live-action, cinematic.
[Shot 1] <Subject 1> from <Picture 1> holds a still pose. The camera [CAMERA_PLACEHOLDER] as they [ACTION_PLACEHOLDER].

overall_soundscape:
[ROOM_TONE_PLACEHOLDER]

non_diegetic_music:
N/A
"""


def _client() -> GenerateVideoClient:
    endpoint = (
        os.getenv("RUNPOD_H3_REF2VA_ENDPOINT_ID") or os.getenv("RUNPOD_ENDPOINT_ID", "")
    ).strip()
    api_key = os.getenv("RUNPOD_API_KEY", "").strip()
    if not endpoint or not api_key:
        raise gr.Error("Set RUNPOD_H3_REF2VA_ENDPOINT_ID and RUNPOD_API_KEY in the environment or .env")
    return GenerateVideoClient(endpoint, api_key)


def _paths(files) -> list[str]:
    if not files:
        return []
    if not isinstance(files, list):
        files = [files]
    out: list[str] = []
    for item in files:
        path = item.name if hasattr(item, "name") else item
        if path:
            out.append(str(path))
    return out


def submit(images, videos, audios, prompt, duration, width, height, steps, seed, ref_image_size, status):
    client = _client()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = str(OUTPUT_DIR / f"h3_ref2va_{stamp}.mp4")
    status = (status or "") + f"\nSubmitting {len(_paths(images))} images / {len(_paths(videos))} videos / {len(_paths(audios))} audios…"
    result = client.create_video_r2v(
        prompt=prompt,
        reference_images=_paths(images),
        reference_videos=_paths(videos),
        reference_audios=_paths(audios),
        duration=int(duration),
        width=int(width),
        height=int(height),
        steps=int(steps),
        seed=int(seed),
        ref_image_size=ref_image_size,
        use_volume_path=False,
    )
    if result.get("error"):
        return None, (status + f"\nERROR: {result['error']}")
    if result.get("status") != "COMPLETED":
        return None, (status + f"\n{result.get('status')}: {result}")
    if not client.save_video_result(result, out_path):
        return None, (status + "\nFailed to save mp4")
    return out_path, (status + f"\nSaved {out_path}")


def build() -> gr.Blocks:
    with gr.Blocks(title="PodBooth MiniMax H3 Ref2VA") as demo:
        gr.Markdown(
            "# PodBooth MiniMax H3 Ref2VA\n"
            "Convenience panel (port 7867). Daily driver will be parent `run-ref2va.ps1` on **7865**. "
            "Caps: 9 images / 3 videos / 3 audio / 12 mixed. Prefer small files; `/run` is 10 MB."
        )
        images = gr.File(label="Reference images (≤9)", file_count="multiple", file_types=["image"])
        videos = gr.File(label="Reference videos (≤3)", file_count="multiple", file_types=["video"])
        audios = gr.File(label="Standalone audio (≤3)", file_count="multiple", file_types=["audio"])
        prompt = gr.Textbox(label="Prompt (six-section Ref2VA)", value=DEFAULT_PROMPT, lines=18)
        with gr.Row():
            duration = gr.Dropdown([5, 10, 15], value=5, label="Duration (s)")
            width = gr.Number(value=1344, label="Width", precision=0)
            height = gr.Number(value=768, label="Height", precision=0)
            steps = gr.Number(value=20, label="Steps", precision=0)
            seed = gr.Number(value=42, label="Seed", precision=0)
            ref_image_size = gr.Dropdown(["match", "max"], value="match", label="ref_image_size")
        status = gr.Textbox(label="Status", lines=8)
        video = gr.Video(label="Result")
        gr.Button("Submit").click(
            submit,
            [images, videos, audios, prompt, duration, width, height, steps, seed, ref_image_size, status],
            [video, status],
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.getenv("PODBOOTH_H3_REF2VA_SERVER_PORT", "7867")))
    args = parser.parse_args()
    demo = build()
    demo.queue()
    demo.launch(server_name="127.0.0.1", server_port=args.port, prevent_thread_lock=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
