from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from image_generation_client import ImageGenerationClient
from provider_profile_loader import load_provider_profile


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual provider profile image generation test")
    parser.add_argument("--profile", required=True, help="Path to provider profile JSON")
    args = parser.parse_args()

    profile = load_provider_profile(args.profile)
    key_env = profile["api_key_env"]
    has_key = bool(os.getenv(key_env, "").strip())
    print("Manual provider profile test")
    print(f"profile_id: {profile.get('profile_id', '')}")
    print(f"provider: {profile.get('provider', '')}")
    print(f"model: {profile.get('model', '')}")
    print(f"base_url_configured: {bool(profile.get('base_url', ''))}")
    print(f"api_key_env: {key_env}")
    print(f"api_key_configured: {has_key}")

    client = ImageGenerationClient.from_profile(args.profile)
    result = client.generate(
        "A clean educational image about aldehydes, ketones, and quinones, 3:4 vertical layout.",
        negative_prompt="messy layout, tiny unreadable text",
        ratio="3:4",
        metadata={
            "request_id": "manual_profile_request",
            "pack_id": "manual_profile_pack",
            "prompt_id": "manual_profile_prompt",
            "task_id": "manual_profile_task",
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
