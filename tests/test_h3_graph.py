from __future__ import annotations

import json
from pathlib import Path

from h3_graph import (
    MAX_MIXED_REFS,
    apply_kitchen_attention,
    apply_loras,
    apply_r2v_job,
    duration_to_length,
    find_one,
    has_reference_pack,
    has_start_image,
    mixed_ref_count,
    snap_dim,
    validate_ref_pack,
)
from generate_video_client import (
    RUN_PAYLOAD_LIMIT_BYTES,
    GenerateVideoClient,
)

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = json.loads((ROOT / "workflows" / "h3_r2v_api.json").read_text(encoding="utf-8"))


def test_duration_grid():
    assert duration_to_length(5) == 124
    assert duration_to_length(10) == 243
    assert duration_to_length(15) == 362
    assert duration_to_length(5) % 17 == 5


def test_snap_dim_is_multiple_of_32():
    assert snap_dim(1344) == 1344
    assert snap_dim(768) == 768
    assert snap_dim(770) == 768
    assert snap_dim(16) == 32


def test_workflow_is_native_ref2va_not_fl2va_or_wan():
    class_types = {node["class_type"] for node in WORKFLOW.values()}
    assert "MiniMaxH3ReferenceToVideo" in class_types
    assert "ModelAttentionBackend" in class_types
    assert "SaveVideo" in class_types
    assert "MiniMaxH3ImageToVideo" not in class_types
    assert not any(name.startswith("Wan") for name in class_types)
    unet = WORKFLOW[find_one(WORKFLOW, "UNETLoader")]["inputs"]
    assert "ref2va" in unet["unet_name"]
    assert "fl2va" not in unet["unet_name"]
    noise = WORKFLOW[find_one(WORKFLOW, "RandomNoise")]["inputs"]
    assert "control_after_generate" not in noise
    save = WORKFLOW[find_one(WORKFLOW, "SaveVideo")]["inputs"]
    assert save["format"] == "auto"
    assert save["codec"] == "auto"
    attn = WORKFLOW[find_one(WORKFLOW, "ModelAttentionBackend")]["inputs"]
    assert attn["attention"] == "comfy kitchen attention"


def test_patch_full_ref_pack_uses_dotted_zero_based_keys():
    videos = [
        {"file": "motion.mp4", "include_audio": True},
        {"file": "cam.mp4", "include_audio": False},
    ]
    patched = apply_r2v_job(
        WORKFLOW,
        image_files=[f"face_{i}.png" for i in range(9)],
        video_files=videos,
        audio_files=[],
        text_prompt="subject_definitions:\n<Subject 1> is the person matching <Picture 1>.",
        width=1344,
        height=768,
        length=124,
        seed=7,
        steps=20,
        ref_image_size="max",
        loras=[{"name": "style.safetensors", "strength": 0.8}],
    )
    r2v = patched["136"]["inputs"]
    assert r2v["ref_image_size"] == "max"
    assert r2v["ref_images.ref_image_0"] == ["ref_img_0", 0]
    assert r2v["ref_images.ref_image_8"] == ["ref_img_8", 0]
    assert "ref_images" not in r2v
    assert r2v["ref_videos.ref_video_0"] == ["ref_vid_split_0", 0]
    assert r2v["ref_video_audios.ref_video_audio_0"] == ["ref_vid_split_0", 1]
    assert "ref_video_audios.ref_video_audio_1" not in r2v
    assert "ref_audios.ref_audio_0" not in r2v
    assert patched["ref_vid_0"]["class_type"] == "LoadVideo"
    assert patched["ref_vid_split_0"]["class_type"] == "GetVideoComponents"
    assert patched["lora_1"]["inputs"]["model"] == ["6", 0]
    assert patched["50"]["inputs"]["model"] == ["lora_1", 0]
    assert patched["16"]["inputs"]["model"] == ["50", 0]
    assert patched["9"]["inputs"]["model"] == ["50", 0]


def test_mixed_cap_rejects_thirteen_files():
    try:
        validate_ref_pack(["a"] * 9, ["v"] * 3, ["au"] * 1, video_soundtracks=1)
        raised = False
    except ValueError as exc:
        raised = True
        assert "12" in str(exc)
    assert raised
    validate_ref_pack(["a"] * 9, ["v"] * 2, ["au"] * 1, video_soundtracks=0)
    assert mixed_ref_count(9, 3, 0, 0) == 12
    assert mixed_ref_count(9, 3, 0, 0) == MAX_MIXED_REFS


def test_audio_only_rejected():
    try:
        validate_ref_pack([], [], ["voice.wav"])
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_too_many_images_rejected():
    try:
        apply_r2v_job(
            WORKFLOW,
            image_files=[f"{i}.png" for i in range(10)],
            text_prompt="x",
            width=1344,
            height=768,
            length=124,
            seed=1,
            steps=20,
        )
        raised = False
    except ValueError as exc:
        raised = True
        assert "9" in str(exc)
    assert raised


def test_kitchen_can_be_stripped():
    patched = apply_r2v_job(
        WORKFLOW,
        image_files=["a.png"],
        text_prompt="x",
        width=1344,
        height=768,
        length=124,
        seed=1,
        steps=20,
        use_kitchen_attention=False,
    )
    class_types = {node["class_type"] for node in patched.values()}
    assert "ModelAttentionBackend" not in class_types
    assert patched["16"]["inputs"]["model"] == ["6", 0]


def test_loras_do_not_rewire_themselves():
    graph = json.loads(json.dumps(WORKFLOW))
    apply_loras(
        graph,
        [
            {"name": "a.safetensors", "strength": 1.0},
            {"name": "b.safetensors", "strength": 0.5},
        ],
    )
    assert graph["lora_1"]["inputs"]["model"] == ["6", 0]
    assert graph["lora_2"]["inputs"]["model"] == ["lora_1", 0]
    assert graph["50"]["inputs"]["model"] == ["lora_2", 0]
    assert graph["16"]["inputs"]["model"] == ["50", 0]


def test_mode_detection():
    assert has_start_image({"image_path": "/x.png"})
    assert not has_start_image({"prompt": "x"})
    assert has_reference_pack({"reference_images": [{}]})
    assert has_reference_pack({"mode": "r2v"})
    assert not has_reference_pack({"image_path": "/x.png"})


def test_payload_guard_blocks_over_10mb():
    client = GenerateVideoClient.__new__(GenerateVideoClient)
    huge = {"mode": "r2v", "prompt": "x", "reference_images": [{"base64": "A" * (11 * 1024 * 1024)}]}
    blocked = GenerateVideoClient._guard_payload(client, huge)
    assert blocked is not None
    assert "10 MB" in blocked["error"]
    small = {"mode": "r2v", "prompt": "x", "reference_images": [{"path": "/runpod-volume/inputs/a.png"}]}
    assert GenerateVideoClient._guard_payload(client, small) is None
    assert RUN_PAYLOAD_LIMIT_BYTES == 10 * 1024 * 1024


def test_apply_kitchen_attention_sets_backend():
    graph = json.loads(json.dumps(WORKFLOW))
    apply_kitchen_attention(graph, enabled=True)
    assert graph["50"]["inputs"]["attention"] == "comfy kitchen attention"
