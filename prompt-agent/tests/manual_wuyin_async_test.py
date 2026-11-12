from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from image_generation_client import ImageGenerationClient


PROMPT = "一只可爱的布偶猫，穿着宇航服，漂浮在绚烂的银河中，照片级真实感，4K高清细节"


def main() -> int:
    api_key = os.getenv("WUYIN_API_KEY", "").strip()
    if api_key:
        print(f"WUYIN_API_KEY configured: prefix={api_key[:6]}..., length={len(api_key)}")
    else:
        print("WUYIN_API_KEY is not configured.")

    client = ImageGenerationClient(provider="wuyin_async")
    result = client.generate(
        PROMPT,
        ratio="3:4",
        metadata={
            "request_id": "manual_wuyin_request",
            "pack_id": "",
            "prompt_id": "manual_wuyin_prompt",
            "task_id": "manual_wuyin_task",
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("status") == "success":
        print(f"image_url: {result.get('image_url', '')}")
        print(f"image_path: {result.get('image_path', '')}")
    else:
        print(f"error_message: {result.get('error_message', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
