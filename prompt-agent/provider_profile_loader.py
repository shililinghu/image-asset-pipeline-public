from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


CUSTOM_HTTP_REQUIRED_FIELDS = ["provider", "method", "base_url", "auth_type", "api_key_env", "model", "payload_style"]
WUYIN_REQUIRED_FIELDS = ["provider", "base_url", "api_key_env", "model", "submit_path", "detail_path"]


DEFAULT_PROFILE = {
    "default_size": "1024x1024",
    "download_remote": True,
    "response_image_url_path": "data.0.url",
    "response_base64_path": "data.0.b64_json",
    "timeout": 60,
    "max_retries": 2,
    "headers": {"Content-Type": "application/json"},
    "output_format": "png",
    "quality": "medium",
    "include_negative_prompt": True,
    "include_response_format": False,
    "moderation": None,
    "submit_path": "/image_gpt",
    "detail_path": "/detail",
    "size": "auto",
    "poll_interval": 2,
    "max_poll_seconds": 180,
}


class ProviderProfileError(ValueError):
    pass


def load_provider_profile(profile_path: str | Path) -> dict[str, Any]:
    path = Path(profile_path)
    if not path.exists():
        raise ProviderProfileError(f"Provider profile file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProviderProfileError(f"Provider profile JSON parse failed: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProviderProfileError("Provider profile must be a JSON object")

    provider = str(payload.get("provider", "")).strip().lower()
    if provider == "wuyin_async":
        required_fields = WUYIN_REQUIRED_FIELDS
    elif provider == "custom_http":
        required_fields = CUSTOM_HTTP_REQUIRED_FIELDS
    else:
        required_fields = ["provider", "model"]

    missing = [field for field in required_fields if not str(payload.get(field, "")).strip()]
    if missing:
        raise ProviderProfileError("Provider profile missing required fields: " + ", ".join(missing))

    profile = dict(DEFAULT_PROFILE)
    profile.update(payload)
    profile["method"] = str(profile.get("method", "POST")).upper()
    profile["provider"] = str(profile.get("provider", "")).strip().lower()
    profile["auth_type"] = str(profile.get("auth_type", "")).strip().lower()
    profile["payload_style"] = str(profile.get("payload_style", "")).strip().lower()
    default_key_env = "WUYIN_API_KEY" if profile["provider"] == "wuyin_async" else "IMAGE_API_KEY"
    profile["api_key_env"] = os.getenv("IMAGE_API_KEY_ENV", str(profile.get("api_key_env", default_key_env)).strip() or default_key_env)
    profile["timeout"] = float(profile.get("timeout", 60) or 60)
    profile["max_retries"] = int(profile.get("max_retries", 2) or 2)
    profile["download_remote"] = _as_bool(profile.get("download_remote", True))
    profile["include_negative_prompt"] = _as_bool(profile.get("include_negative_prompt", True))
    profile["include_response_format"] = _as_bool(profile.get("include_response_format", False))
    profile["poll_interval"] = float(profile.get("poll_interval", 2) or 2)
    profile["max_poll_seconds"] = float(profile.get("max_poll_seconds", 180) or 180)
    headers = profile.get("headers", {})
    profile["headers"] = headers if isinstance(headers, dict) else {}
    return profile


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off"}
