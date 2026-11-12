from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import config


SECRET_KEYS = ["WUYIN_API_KEY", "OPENAI_API_KEY", "IMAGE_API_KEY", "TASTELENS_API_KEY"]

DEFAULT_SETTINGS: dict[str, Any] = {
    "image_provider": "mock",
    "image_provider_profile_path": "",
    "image_model": "mock-image-model",
    "aesthetic_review_provider": "mock",
    "aesthetic_review_api_url": "",
    "aesthetic_review_image_field": "image",
    "aesthetic_review_score_field": "displayScore",
    "aesthetic_review_rating_field": "rating",
    "aesthetic_review_reason_field": "analysis",
    "download_remote_images": True,
}


def load_settings() -> dict[str, Any]:
    payload = _read_json_object(config.APP_SETTINGS_FILE)
    settings = dict(DEFAULT_SETTINGS)
    settings.update(payload)
    return settings


def save_settings(updates: dict[str, Any]) -> dict[str, Any]:
    allowed = set(DEFAULT_SETTINGS)
    current = load_settings()
    for key, value in updates.items():
        if key in allowed:
            current[key] = value
    _write_json_object(config.APP_SETTINGS_FILE, current)
    apply_settings_to_environ(current)
    return current


def load_local_secrets() -> dict[str, str]:
    payload = _read_json_object(config.SECRETS_LOCAL_FILE)
    return {key: str(value) for key, value in payload.items() if isinstance(value, str)}


def save_secrets(updates: dict[str, Any], *, clear: bool = False) -> dict[str, dict[str, Any]]:
    if config.SETTINGS_MODE == "env_only":
        raise PermissionError("SETTINGS_MODE=env_only does not allow saving local secrets")
    secrets = load_local_secrets()
    for key, value in updates.items():
        if key not in SECRET_KEYS:
            continue
        text = str(value or "").strip()
        if text:
            secrets[key] = text
            os.environ[key] = text
        elif clear:
            secrets.pop(key, None)
            os.environ.pop(key, None)
    _write_json_object(config.SECRETS_LOCAL_FILE, secrets)
    return secret_status()


def delete_secret(key_name: str) -> dict[str, Any]:
    if config.SETTINGS_MODE == "env_only":
        raise PermissionError("SETTINGS_MODE=env_only does not allow deleting local secrets")
    secrets = load_local_secrets()
    removed = key_name in secrets
    secrets.pop(key_name, None)
    _write_json_object(config.SECRETS_LOCAL_FILE, secrets)
    return {"removed": removed, "secret": secret_status([key_name]).get(key_name, {})}


def get_setting(name: str, default: Any = None) -> Any:
    return load_settings().get(name, default)


def get_secret(name: str) -> str:
    env_value = os.getenv(name, "").strip()
    if env_value:
        return env_value
    return load_local_secrets().get(name, "").strip()


def secret_status(keys: list[str] | None = None) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    local = load_local_secrets()
    for key in keys or SECRET_KEYS:
        env_value = os.getenv(key, "").strip()
        local_value = local.get(key, "").strip()
        value = env_value or local_value
        rows[key] = {
            "exists": bool(value),
            "source": "environment" if env_value else "secrets.local.json" if local_value else "",
            "masked": mask_secret(value),
            "length": len(value),
        }
    return rows


def mask_secret(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return value[:2] + "****"
    return value[:3] + "****" + value[-4:]


def apply_secrets_to_environ() -> None:
    for key, value in load_local_secrets().items():
        if key in SECRET_KEYS and not os.getenv(key):
            os.environ[key] = value


def apply_settings_to_environ(settings: dict[str, Any] | None = None) -> None:
    settings = settings or load_settings()
    mapping = {
        "image_provider": "IMAGE_PROVIDER",
        "image_provider_profile_path": "IMAGE_PROVIDER_PROFILE_PATH",
        "image_model": "IMAGE_MODEL",
        "aesthetic_review_provider": "AESTHETIC_REVIEW_PROVIDER",
        "aesthetic_review_api_url": "AESTHETIC_REVIEW_API_URL",
        "aesthetic_review_image_field": "AESTHETIC_REVIEW_IMAGE_FIELD",
        "aesthetic_review_score_field": "AESTHETIC_REVIEW_SCORE_FIELD",
        "aesthetic_review_rating_field": "AESTHETIC_REVIEW_RATING_FIELD",
        "aesthetic_review_reason_field": "AESTHETIC_REVIEW_REASON_FIELD",
    }
    for setting_key, env_key in mapping.items():
        value = settings.get(setting_key)
        if value is not None and not os.getenv(env_key):
            os.environ[env_key] = str(value)


def get_effective_image_provider_config() -> dict[str, Any]:
    settings = load_settings()
    provider = os.getenv("IMAGE_PROVIDER", "").strip() or str(settings.get("image_provider", "mock"))
    profile_path = os.getenv("IMAGE_PROVIDER_PROFILE_PATH", "").strip() or str(settings.get("image_provider_profile_path", ""))
    model = os.getenv("IMAGE_MODEL", "").strip() or str(settings.get("image_model", ""))
    key_name = "WUYIN_API_KEY" if provider == "wuyin_async" else "OPENAI_API_KEY" if "openai" in provider else "IMAGE_API_KEY"
    return {
        "provider": provider,
        "provider_profile_path": profile_path,
        "model": model,
        "api_key_env": key_name,
        "api_key": get_secret(key_name),
    }


def get_effective_aesthetic_review_config() -> dict[str, Any]:
    settings = load_settings()
    provider = os.getenv("AESTHETIC_REVIEW_PROVIDER", "").strip() or str(settings.get("aesthetic_review_provider", "mock"))
    api_url = os.getenv("AESTHETIC_REVIEW_API_URL", "").strip() or str(settings.get("aesthetic_review_api_url", ""))
    return {
        "provider": provider,
        "api_url": api_url,
        "image_field": str(settings.get("aesthetic_review_image_field", "image")),
        "score_field": str(settings.get("aesthetic_review_score_field", "displayScore")),
        "rating_field": str(settings.get("aesthetic_review_rating_field", "rating")),
        "reason_field": str(settings.get("aesthetic_review_reason_field", "analysis")),
        "api_key_env": "TASTELENS_API_KEY",
        "api_key": get_secret("TASTELENS_API_KEY"),
    }


def public_settings_payload() -> dict[str, Any]:
    return {
        "ok": True,
        "settings_mode": config.SETTINGS_MODE,
        "settings": load_settings(),
        "secrets": secret_status(),
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        if not path.exists():
            return {}
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return {}
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json_object(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)
