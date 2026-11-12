from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import IMAGE_API_BASE_URL, IMAGE_API_KEY_ENV, IMAGE_MODEL, IMAGE_PROVIDER
from image_generation_client import ImageGenerationClient


def main() -> None:
    provider = os.getenv("IMAGE_PROVIDER", IMAGE_PROVIDER)
    model = os.getenv("IMAGE_MODEL", IMAGE_MODEL)
    base_url = os.getenv("IMAGE_API_BASE_URL", IMAGE_API_BASE_URL)
    key_env = os.getenv("IMAGE_API_KEY_ENV", IMAGE_API_KEY_ENV)
    has_key = bool(os.getenv(key_env, "").strip())
    print("Manual custom_http image provider test")
    print(f"provider: {provider}")
    print(f"forced_provider: custom_http")
    print(f"model: {model}")
    print(f"base_url_configured: {bool(base_url)}")
    print(f"api_key_env: {key_env}")
    print(f"api_key_configured: {has_key}")

    client = ImageGenerationClient(provider="custom_http", model=model)
    result = client.generate(
        "A clean educational cover image about aldehydes, ketones, and quinones, readable title, 3:4 vertical layout.",
        negative_prompt="messy layout, tiny unreadable text",
        ratio="3:4",
        metadata={
            "request_id": "manual_request",
            "pack_id": "manual_pack",
            "prompt_id": "manual_prompt",
            "task_id": "manual_task",
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("status") == "success":
        print(f"image_path: {result.get('image_path', '')}")
        print(f"image_url: {result.get('image_url', '')}")
    else:
        print(f"error_message: {result.get('error_message', '')}")


if __name__ == "__main__":
    main()
