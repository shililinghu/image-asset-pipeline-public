from __future__ import annotations

import json
import mimetypes
import os
import uuid
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from config import (
    AESTHETIC_REVIEW_API_KEY_ENV,
    AESTHETIC_REVIEW_API_URL,
    AESTHETIC_REVIEW_IMAGE_FIELD,
    AESTHETIC_REVIEW_PROVIDER,
    AESTHETIC_REVIEW_RATING_FIELD,
    AESTHETIC_REVIEW_REASON_FIELD,
    AESTHETIC_REVIEW_REQUEST_STYLE,
    AESTHETIC_REVIEW_SCORE_FIELD,
    AESTHETIC_REVIEW_TIMEOUT,
)


class AestheticReviewClient:
    def __init__(
        self,
        provider: str = AESTHETIC_REVIEW_PROVIDER,
        *,
        api_url: str = AESTHETIC_REVIEW_API_URL,
        api_key_env: str = AESTHETIC_REVIEW_API_KEY_ENV,
        timeout: float = AESTHETIC_REVIEW_TIMEOUT,
        image_field: str = AESTHETIC_REVIEW_IMAGE_FIELD,
        request_style: str = AESTHETIC_REVIEW_REQUEST_STYLE,
        score_field: str = AESTHETIC_REVIEW_SCORE_FIELD,
        rating_field: str = AESTHETIC_REVIEW_RATING_FIELD,
        reason_field: str = AESTHETIC_REVIEW_REASON_FIELD,
    ) -> None:
        self.provider = provider
        self.api_url = api_url
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.image_field = image_field
        self.request_style = request_style
        self.score_field = score_field
        self.rating_field = rating_field
        self.reason_field = reason_field

    def review(self, image_path: str = "", image_url: str = "") -> dict[str, Any]:
        if self.provider == "mock":
            return self._mock_review(image_path=image_path, image_url=image_url)
        if self.provider == "tastelens_http":
            return self._review_tastelens_http(image_path=image_path, image_url=image_url)
        return self._failed(f"Unsupported aesthetic review provider: {self.provider}")

    def _mock_review(self, image_path: str = "", image_url: str = "") -> dict[str, Any]:
        seed = sum(ord(char) for char in f"{image_path}{image_url}")
        score = 72 + seed % 16
        return {
            "status": "success",
            "provider": "mock",
            "aesthetic_score": float(score),
            "rating": "mock aesthetic review",
            "reason": "mock aesthetic review",
            "raw": {},
            "error_message": "",
        }

    def _review_tastelens_http(self, image_path: str = "", image_url: str = "") -> dict[str, Any]:
        if not self.api_url:
            return self._failed("AESTHETIC_REVIEW_API_URL is required for tastelens_http")
        if not image_path:
            return self._failed("tastelens_http requires a local image_path")
        path = Path(image_path)
        if not path.exists() or not path.is_file():
            return self._failed(f"image_path does not exist: {image_path}")
        if self.request_style != "multipart":
            return self._failed(f"Unsupported AESTHETIC_REVIEW_REQUEST_STYLE: {self.request_style}")

        try:
            payload = self._post_multipart(path)
            return self._success_from_payload(payload)
        except Exception as exc:
            return self._failed(str(exc))

    def _post_multipart(self, image_path: Path) -> dict[str, Any]:
        boundary = "----TasteLensBoundary" + uuid.uuid4().hex
        mime_type = mimetypes.guess_type(str(image_path))[0] or "application/octet-stream"
        file_bytes = image_path.read_bytes()
        body = b"".join(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                (
                    f'Content-Disposition: form-data; name="{self.image_field}"; '
                    f'filename="{image_path.name}"\r\n'
                ).encode("utf-8"),
                f"Content-Type: {mime_type}\r\n\r\n".encode("utf-8"),
                file_bytes,
                b"\r\n",
                f"--{boundary}--\r\n".encode("utf-8"),
            ]
        )
        headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
        api_key = os.getenv(self.api_key_env, "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(self.api_url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                status = int(getattr(response, "status", 200))
                text = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            raise RuntimeError(f"TasteLens HTTP {exc.code}: {body_text[:300]}") from exc
        if not (200 <= status < 300):
            raise RuntimeError(f"TasteLens HTTP {status}: {text[:300]}")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"TasteLens JSON parse failed: {exc}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("TasteLens response must be a JSON object")
        return payload

    def _success_from_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        score = _safe_float(_get_first(payload, [self.score_field, "overallScore", "displayScore", "score"]))
        rating = str(_get_first(payload, [self.rating_field, "overallRating", "rating"]) or "")
        reason = str(_get_first(payload, [self.reason_field, "analysis", "intent", "reason"]) or "")
        return {
            "status": "success",
            "provider": self.provider,
            "aesthetic_score": score,
            "rating": rating,
            "reason": reason,
            "raw": _summarize_payload(payload),
            "error_message": "",
        }

    def _failed(self, message: str) -> dict[str, Any]:
        return {
            "status": "failed",
            "provider": self.provider,
            "aesthetic_score": 0,
            "rating": "",
            "reason": "",
            "raw": {},
            "error_message": message,
        }


def _get_first(payload: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key and key in payload:
            return payload[key]
    return None


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _summarize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "response_keys": sorted(str(key) for key in payload.keys()),
        "dimension_count": len(payload.get("dimensions", [])) if isinstance(payload.get("dimensions"), list) else 0,
        "has_references": bool(payload.get("references")),
    }
