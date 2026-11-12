from __future__ import annotations

import json
import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aesthetic_review_client import AestheticReviewClient
from image_auto_reviewer import review_image_results


class FakeResponse:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self.payload = payload
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class AestheticReviewClientTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.image_path = Path(self.tmp.name) / "image.png"
        self.image_path.write_bytes(b"\x89PNG\r\n\x1a\nmock")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_mock_provider_returns_stable_structure(self) -> None:
        result = AestheticReviewClient(provider="mock").review(image_path=str(self.image_path))
        self.assertEqual(result["status"], "success")
        for field in ["provider", "aesthetic_score", "rating", "reason", "raw", "error_message"]:
            self.assertIn(field, result)

    def test_tastelens_without_api_url_fails(self) -> None:
        result = AestheticReviewClient(provider="tastelens_http", api_url="").review(image_path=str(self.image_path))
        self.assertEqual(result["status"], "failed")
        self.assertIn("AESTHETIC_REVIEW_API_URL", result["error_message"])

    def test_tastelens_missing_image_path_fails(self) -> None:
        missing = Path(self.tmp.name) / "missing.png"
        result = AestheticReviewClient(provider="tastelens_http", api_url="http://127.0.0.1:3000/api/analyze").review(image_path=str(missing))
        self.assertEqual(result["status"], "failed")
        self.assertIn("does not exist", result["error_message"])

    def test_tastelens_http_response_parses_configured_fields(self) -> None:
        payload = {"displayScore": 83, "rating": "成熟审美水平", "analysis": "composition is strong"}
        client = AestheticReviewClient(provider="tastelens_http", api_url="http://127.0.0.1:3000/api/analyze")
        with patch("urllib.request.urlopen", return_value=FakeResponse(payload)):
            result = client.review(image_path=str(self.image_path))
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["aesthetic_score"], 83)
        self.assertEqual(result["rating"], "成熟审美水平")
        self.assertEqual(result["reason"], "composition is strong")

    def test_tastelens_http_response_parses_actual_tastelens_fields(self) -> None:
        payload = {"overallScore": 86, "overallRating": "顶级", "intent": "clean portrait", "dimensions": []}
        client = AestheticReviewClient(provider="tastelens_http", api_url="http://127.0.0.1:3000/api/analyze")
        with patch("urllib.request.urlopen", return_value=FakeResponse(payload)):
            result = client.review(image_path=str(self.image_path))
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["aesthetic_score"], 86)
        self.assertEqual(result["rating"], "顶级")
        self.assertEqual(result["reason"], "clean portrait")

    def test_image_auto_reviewer_writes_aesthetic_field(self) -> None:
        class StaticClient:
            def review(self, image_path="", image_url=""):
                return {
                    "status": "success",
                    "provider": "tastelens_http",
                    "aesthetic_score": 86,
                    "rating": "顶级",
                    "reason": "strong image",
                    "raw": {"response_keys": ["overallScore"]},
                    "error_message": "",
                }

        result = {
            "image_result_id": "image_1",
            "request_id": "request_1",
            "pack_id": "pack_1",
            "prompt_id": "prompt_1",
            "image_path": str(self.image_path),
        }
        reviews = review_image_results([result], save=False, aesthetic_client=StaticClient())
        self.assertEqual(len(reviews), 1)
        self.assertEqual(reviews[0]["aesthetic"]["status"], "success")
        self.assertEqual(reviews[0]["aesthetic"]["score"], 86)
        self.assertGreaterEqual(reviews[0]["dimensions"]["visual_quality"], 90)

    def test_image_auto_reviewer_continues_when_tastelens_fails(self) -> None:
        class FailingClient:
            def review(self, image_path="", image_url=""):
                return {
                    "status": "failed",
                    "provider": "tastelens_http",
                    "aesthetic_score": 0,
                    "rating": "",
                    "reason": "",
                    "raw": {},
                    "error_message": "service unavailable",
                }

        result = {
            "image_result_id": "image_2",
            "request_id": "request_1",
            "pack_id": "pack_1",
            "prompt_id": "prompt_1",
            "image_path": str(self.image_path),
        }
        reviews = review_image_results([result], save=False, aesthetic_client=FailingClient())
        self.assertEqual(len(reviews), 1)
        self.assertEqual(reviews[0]["aesthetic"]["status"], "failed")
        self.assertIn("service unavailable", reviews[0]["aesthetic"]["error_message"])


if __name__ == "__main__":
    unittest.main()
