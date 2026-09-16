#!/usr/bin/env python3
"""Python client for the PodBooth MiniMax H3 Ref2VA RunPod serverless worker."""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from typing import Any

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REDACT_KEYS = {
    "image_base64",
    "end_image_base64",
    "reference_images",
    "reference_videos",
    "reference_audios",
}

# RunPod QUEUE /run is 10 MB. /runsync is 20 MB. This client uses /run.
# Twelve refs as base64 will not fit. Prefer /runpod-volume paths or URLs.
RUN_PAYLOAD_LIMIT_BYTES = 10 * 1024 * 1024
RUN_PAYLOAD_SOFT_LIMIT_BYTES = 8 * 1024 * 1024


class GenerateVideoClient:
    def __init__(self, runpod_endpoint_id: str, runpod_api_key: str):
        self.runpod_endpoint_id = runpod_endpoint_id
        self.runpod_api_endpoint = f"https://api.runpod.ai/v2/{runpod_endpoint_id}/run"
        self.status_url = f"https://api.runpod.ai/v2/{runpod_endpoint_id}/status"
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {runpod_api_key}",
                "Content-Type": "application/json",
            }
        )
        logger.info("GenerateVideoClient initialized — endpoint %s", runpod_endpoint_id)

    def encode_file_to_base64(self, file_path: str) -> str | None:
        try:
            if not os.path.exists(file_path):
                logger.error("File does not exist: %s", file_path)
                return None
            with open(file_path, "rb") as handle:
                return base64.b64encode(handle.read()).decode("utf-8")
        except Exception as exc:
            logger.error("File base64 encoding failed: %s", exc)
            return None

    def _redact(self, input_data: dict[str, Any]) -> dict[str, Any]:
        redacted = dict(input_data)
        for key in REDACT_KEYS:
            if key in redacted and redacted[key]:
                redacted[key] = f"<{key} omitted, {len(str(redacted[key]))} chars>"
        return redacted

    def estimate_payload_bytes(self, input_data: dict[str, Any]) -> int:
        return len(json.dumps({"input": input_data}).encode("utf-8"))

    def _guard_payload(self, input_data: dict[str, Any]) -> dict[str, Any] | None:
        size = self.estimate_payload_bytes(input_data)
        if size <= RUN_PAYLOAD_LIMIT_BYTES:
            if size > RUN_PAYLOAD_SOFT_LIMIT_BYTES:
                logger.warning(
                    "Job JSON is %.1f MB (soft limit 8 MB, hard /run limit 10 MB). "
                    "Prefer /runpod-volume paths or URLs for extra refs.",
                    size / (1024 * 1024),
                )
            return None
        return {
            "error": (
                f"Job JSON is {size / (1024 * 1024):.1f} MB; RunPod /run rejects bodies over 10 MB. "
                "Do not base64 a full 12-ref pack. Put files on the network volume "
                "(/runpod-volume/inputs/...) or pass https URLs so the worker downloads them."
            )
        }

    def _slot_from_source(self, source: str | None, prefix: str, use_volume_path: bool | None) -> dict[str, str]:
        if not source:
            return {}
        if source.startswith("http://") or source.startswith("https://"):
            return {f"{prefix}_url": source}
        volume = source.startswith("/runpod-volume")
        if use_volume_path is True or (use_volume_path is None and volume):
            return {f"{prefix}_path": source}
        encoded = self.encode_file_to_base64(source)
        if not encoded:
            raise FileNotFoundError(f"Could not encode {source}")
        return {f"{prefix}_base64": encoded}

    def _ref_from_source(self, source: Any, use_volume_path: bool | None) -> dict[str, Any]:
        if isinstance(source, dict):
            item = dict(source)
            file_source = item.pop("file", None) or item.pop("source", None)
            if file_source and not any(k in item for k in ("path", "url", "base64")):
                item.update(self._ref_from_source(file_source, use_volume_path))
            return item
        if not isinstance(source, str):
            raise TypeError(f"Reference must be a path/url string or dict, got {type(source).__name__}")
        if source.startswith("http://") or source.startswith("https://"):
            return {"url": source}
        volume = source.startswith("/runpod-volume")
        if use_volume_path is True or (use_volume_path is None and volume):
            return {"path": source}
        encoded = self.encode_file_to_base64(source)
        if not encoded:
            raise FileNotFoundError(f"Could not encode {source}")
        return {"base64": encoded, "filename": os.path.basename(source)}

    def submit_job(self, input_data: dict[str, Any]) -> str | None:
        blocked = self._guard_payload(input_data)
        if blocked:
            logger.error(blocked["error"])
            return None
        payload = {"input": input_data}
        try:
            logger.info("Submitting job to %s", self.runpod_api_endpoint)
            logger.info("Input data: %s", json.dumps(self._redact(input_data), indent=2, ensure_ascii=False))
            response = self.session.post(self.runpod_api_endpoint, json=payload, timeout=30)
            response.raise_for_status()
            job_id = response.json().get("id")
            if not job_id:
                logger.error("Failed to receive Job ID: %s", response.json())
                return None
            logger.info("Job submission successful. Job ID: %s", job_id)
            return job_id
        except requests.exceptions.RequestException as exc:
            logger.error("Job submission failed: %s", exc)
            return None

    def wait_for_completion(
        self,
        job_id: str,
        check_interval: int = 10,
        max_wait_time: int = 2700,
    ) -> dict[str, Any]:
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            try:
                logger.info("Checking job status (Job ID: %s)", job_id)
                response = self.session.get(f"{self.status_url}/{job_id}", timeout=30)
                response.raise_for_status()
                status_data = response.json()
                status = status_data.get("status")
                if status == "COMPLETED":
                    return {"status": "COMPLETED", "output": status_data.get("output"), "job_id": job_id}
                if status == "FAILED":
                    return {
                        "status": "FAILED",
                        "error": status_data.get("error", "Unknown error"),
                        "job_id": job_id,
                    }
                if status in {"IN_QUEUE", "IN_PROGRESS"}:
                    logger.info("Job in progress (status: %s)", status)
                    time.sleep(check_interval)
                    continue
                return {"status": "UNKNOWN", "data": status_data, "job_id": job_id}
            except requests.exceptions.RequestException as exc:
                logger.error("Status check error: %s", exc)
                time.sleep(check_interval)
        return {"status": "TIMEOUT", "job_id": job_id}

    def save_video_result(self, result: dict[str, Any], output_path: str) -> bool:
        try:
            if result.get("status") != "COMPLETED":
                logger.error("Job not completed: %s", result.get("status"))
                return False
            output = result.get("output") or {}
            video_b64 = output.get("video")
            if not video_b64:
                logger.error("Video data not found")
                return False
            parent = os.path.dirname(output_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            payload = video_b64.split(",", 1)[1] if video_b64.startswith("data:") else video_b64
            with open(output_path, "wb") as handle:
                handle.write(base64.b64decode(payload))
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            logger.info("Video saved: %s (%.1f MB)", output_path, size_mb)
            return True
        except Exception as exc:
            logger.error("Video save failed: %s", exc)
            return False

    def create_video_i2v(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "error": (
                "This client talks to the Ref2VA sibling worker. I2V / FL2VA jobs belong on "
                "podbooth-minimax-h3 (parent booth run-fl2va.ps1 on 7864)."
            )
        }

    def create_video_r2v(
        self,
        prompt: str,
        reference_images: list[Any] | None = None,
        reference_videos: list[Any] | None = None,
        reference_audios: list[Any] | None = None,
        negative_prompt: str | None = None,
        width: int = 1344,
        height: int = 768,
        duration: int = 5,
        fps: int = 24,
        steps: int = 20,
        seed: int = 42,
        ref_image_size: str = "match",
        loras: list[dict[str, Any]] | None = None,
        disable_audio: bool = False,
        use_kitchen_attention: bool = True,
        use_volume_path: bool | None = None,
        max_wait_time: int = 2700,
    ) -> dict[str, Any]:
        """Reference pack required. Caps: 9 images / 3 videos / 3 audio / 12 mixed files."""
        try:
            images = [self._ref_from_source(item, use_volume_path) for item in (reference_images or [])]
            videos = [self._ref_from_source(item, use_volume_path) for item in (reference_videos or [])]
            audios = [self._ref_from_source(item, use_volume_path) for item in (reference_audios or [])]
            input_data: dict[str, Any] = {
                "mode": "r2v",
                "prompt": prompt,
                "reference_images": images,
                "reference_videos": videos,
                "reference_audios": audios,
                "width": width,
                "height": height,
                "duration": duration,
                "fps": fps,
                "steps": steps,
                "seed": seed,
                "ref_image_size": ref_image_size,
                "disable_audio": disable_audio,
                "use_kitchen_attention": use_kitchen_attention,
                "loras": loras or [],
            }
            if negative_prompt:
                input_data["negative_prompt"] = negative_prompt
        except FileNotFoundError as exc:
            return {"error": str(exc)}

        blocked = self._guard_payload(input_data)
        if blocked:
            return blocked

        job_id = self.submit_job(input_data)
        if not job_id:
            return {"error": "Job submission failed"}
        return self.wait_for_completion(job_id, max_wait_time=max_wait_time)


def main() -> None:
    endpoint_id = os.getenv("RUNPOD_H3_REF2VA_ENDPOINT_ID") or os.getenv("RUNPOD_ENDPOINT_ID", "")
    endpoint_id = endpoint_id.strip()
    api_key = os.getenv("RUNPOD_API_KEY", "").strip()
    if not endpoint_id or not api_key:
        raise SystemExit("Set RUNPOD_H3_REF2VA_ENDPOINT_ID and RUNPOD_API_KEY")
    client = GenerateVideoClient(endpoint_id, api_key)
    print("Example (does not run unless you replace paths and env):")
    print(
        "client.create_video_r2v("
        "reference_images=['/runpod-volume/inputs/face.png'], "
        "prompt='subject_definitions:\\n<Subject 1> is the person matching <Picture 1>.', "
        "duration=5, width=1344, height=768)"
    )
    _ = client


if __name__ == "__main__":
    main()
