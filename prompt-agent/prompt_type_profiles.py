from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import config


PROFILE_PATH = config.BASE_DIR / "config" / "prompt_type_profiles.json"
DEFAULT_TYPE_ID = "portrait"


@lru_cache(maxsize=1)
def load_prompt_type_profiles() -> list[dict[str, Any]]:
    with PROFILE_PATH.open("r", encoding="utf-8") as file:
        profiles = json.load(file)
    if not isinstance(profiles, list):
        raise ValueError("prompt_type_profiles.json must contain a list")
    return [dict(item) for item in profiles if isinstance(item, dict) and item.get("type_id")]


def get_prompt_type_profile(type_id: str | None) -> dict[str, Any]:
    wanted = str(type_id or DEFAULT_TYPE_ID).strip() or DEFAULT_TYPE_ID
    profiles = load_prompt_type_profiles()
    for profile in profiles:
        if profile.get("type_id") == wanted:
            return dict(profile)
    for profile in profiles:
        if profile.get("type_id") == DEFAULT_TYPE_ID:
            return dict(profile)
    raise ValueError("default prompt type profile not found")


def apply_prompt_type_defaults(data: dict[str, Any]) -> dict[str, Any]:
    profile = get_prompt_type_profile(data.get("prompt_type"))
    result = dict(data)
    result["prompt_type"] = profile.get("type_id", DEFAULT_TYPE_ID)
    result["domain"] = result.get("domain") or profile.get("domain", "")
    result["platform"] = result.get("platform") or profile.get("platform", "general")
    result["use_case"] = result.get("use_case") or profile.get("use_case", "")
    result["style"] = result.get("style") or profile.get("style", "")
    result["review_profile"] = result.get("review_profile") or profile.get("review_profile", "")
    result["ratio"] = result.get("ratio") or profile.get("default_ratio", "3:4")
    result["candidate_count"] = result.get("candidate_count") or profile.get("candidate_count_default", 8)
    result["images_per_prompt"] = result.get("images_per_prompt") or profile.get("images_per_prompt_default", 1)
    result["requires_reference_images"] = bool(profile.get("requires_reference_images", False))
    return result
