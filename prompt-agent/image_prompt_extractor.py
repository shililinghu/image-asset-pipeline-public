from __future__ import annotations

from typing import Any

from storage import JsonStore
from utils import new_id, now_iso


def extract_image_prompts(
    request: dict[str, Any],
    reference_images: list[dict[str, Any]],
    *,
    store: JsonStore | None = None,
    provider: str = "mock",
) -> dict[str, Any]:
    json_store = store or JsonStore()

    topic = str(request.get("topic") or request.get("title") or "主题")
    style = str(request.get("style") or "visual style")
    description = str(request.get("description") or request.get("requirements") or "")
    subject = topic or "人物主体"
    skeleton = (
        f"{subject}, {style}, high quality image, refined composition, "
        f"consistent subject identity and mood, {description}".strip(", ")
    )
    extracted = {
        "subject": subject,
        "style_keywords": [style, "editorial portrait", "cinematic lighting"],
        "composition_keywords": ["center composition", "close-up portrait", "clean visual hierarchy"],
        "lighting_keywords": ["studio light", "soft contrast", "controlled highlights"],
        "background_keywords": ["minimal background", "context-aware backdrop"],
        "negative_keywords": ["bad hands", "deformed face", "low quality", "watermark", "extra text"],
        "prompt_skeletons": [
            skeleton,
            f"{subject}, {style}, thematic portrait variation, strong composition, premium details",
        ],
    }
    record = {
        "extract_run_id": new_id("extract"),
        "request_id": request.get("request_id", ""),
        "reference_image_ids": [item.get("reference_image_id", "") for item in reference_images],
        "provider": provider,
        "status": "success",
        "extracted": extracted,
        "created_at": now_iso(),
    }
    json_store.add("image_prompt_extractions", record)
    return record
