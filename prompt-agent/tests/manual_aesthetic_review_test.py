from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aesthetic_review_client import AestheticReviewClient
from config import AESTHETIC_REVIEW_API_URL, AESTHETIC_REVIEW_PROVIDER


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one TasteLens aesthetic review request.")
    parser.add_argument("--image", required=True, help="Local image path to review.")
    args = parser.parse_args()

    print(f"AESTHETIC_REVIEW_PROVIDER: {AESTHETIC_REVIEW_PROVIDER}")
    print(f"AESTHETIC_REVIEW_API_URL: {AESTHETIC_REVIEW_API_URL}")
    client = AestheticReviewClient()
    result = client.review(image_path=args.image)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("status") == "success":
        print(f"score: {result.get('aesthetic_score')}")
        print(f"rating: {result.get('rating')}")
        print(f"reason: {result.get('reason')}")
    else:
        print(f"error_message: {result.get('error_message')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
