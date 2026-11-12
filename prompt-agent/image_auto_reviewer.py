from __future__ import annotations

from typing import Any

from aesthetic_review_client import AestheticReviewClient
from models import create_image_review
from storage import JsonStore


REVIEW_DIMENSIONS = [
    "prompt_match",
    "visual_quality",
    "text_quality",
    "platform_fit",
    "brand_fit",
    "safety",
    "publish_potential",
]


def review_image_results(
    image_results: list[dict[str, Any]] | None = None,
    store: JsonStore | None = None,
    save: bool = True,
    aesthetic_client: AestheticReviewClient | None = None,
) -> list[dict[str, Any]]:
    json_store = store or JsonStore()
    client = aesthetic_client or AestheticReviewClient()
    results = image_results if image_results is not None else json_store.load("image_results")
    existing_ids = {item.get("image_result_id") for item in json_store.load("image_reviews")}
    reviews: list[dict[str, Any]] = []
    requests = {item.get("request_id"): item for item in json_store.load("idea_requests")}
    packs = {item.get("pack_id"): item for item in json_store.load("prompt_packs")}
    for result in results:
        if result.get("image_result_id") in existing_ids:
            continue
        review_profile = _review_profile_for_result(result, requests, packs)
        dimensions = _mock_dimensions(result, review_profile)
        aesthetic_result = client.review(
            image_path=str(result.get("image_path", "") or ""),
            image_url=str(result.get("image_url", "") or ""),
        )
        dimensions, aesthetic_decision = _apply_aesthetic_to_dimensions(dimensions, aesthetic_result)
        auto_score = round(sum(dimensions.values()) / len(dimensions), 2)
        review = create_image_review(
            {
                "image_result_id": result.get("image_result_id", ""),
                "request_id": result.get("request_id", ""),
                "pack_id": result.get("pack_id", ""),
                "prompt_id": result.get("prompt_id", ""),
                "dimensions": dimensions,
                "auto_score": auto_score,
                "auto_decision": aesthetic_decision or _decision(auto_score, dimensions["safety"]),
                "safety_passed": dimensions["safety"] >= 70,
                "notes": f"mock auto review / {review_profile}",
            }
        )
        review["review_profile"] = review_profile
        review["aesthetic"] = _aesthetic_review_payload(aesthetic_result)
        reviews.append(review)
        if save:
            json_store.add("image_reviews", review)
    return reviews


def _aesthetic_review_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider": result.get("provider", "mock"),
        "status": result.get("status", "failed"),
        "score": float(result.get("aesthetic_score", 0) or 0),
        "rating": result.get("rating", ""),
        "reason": result.get("reason", ""),
        "error_message": result.get("error_message", ""),
        "raw": dict(result.get("raw", {}) or {}),
    }


def _apply_aesthetic_to_dimensions(
    dimensions: dict[str, float],
    aesthetic_result: dict[str, Any],
) -> tuple[dict[str, float], str]:
    if aesthetic_result.get("status") != "success":
        return dimensions, ""
    score = float(aesthetic_result.get("aesthetic_score", 0) or 0)
    updated = dict(dimensions)
    decision = ""
    if score >= 85:
        _floor(updated, ["visual_quality"], 90)
        _floor(updated, ["brand_fit", "platform_fit", "publish_potential"], 80)
    elif score >= 70:
        _floor(updated, ["visual_quality", "brand_fit", "platform_fit", "publish_potential"], 70)
    elif score >= 50:
        _cap(updated, ["visual_quality", "brand_fit", "platform_fit"], 60)
        decision = "manual_review"
    else:
        _cap(updated, ["visual_quality", "brand_fit", "platform_fit", "publish_potential"], 40)
        decision = "reject"
    return updated, decision


def _floor(dimensions: dict[str, float], keys: list[str], value: float) -> None:
    for key in keys:
        dimensions[key] = max(float(dimensions.get(key, 0) or 0), value)


def _cap(dimensions: dict[str, float], keys: list[str], value: float) -> None:
    for key in keys:
        dimensions[key] = min(float(dimensions.get(key, 0) or 0), value)


def _mock_dimensions(result: dict[str, Any], review_profile: str = "portrait_review") -> dict[str, float]:
    seed = sum(ord(char) for char in str(result.get("image_result_id", "")))
    base = 70 + seed % 18
    dimensions = {}
    for index, key in enumerate(REVIEW_DIMENSIONS):
        dimensions[key] = max(0, min(100, base + ((seed + index * 7) % 11) - 5))
    if review_profile in {"poster_review", "infographic_review"}:
        dimensions["platform_fit"] = min(100, dimensions["platform_fit"] + 4)
        dimensions["text_quality"] = min(100, dimensions["text_quality"] + 4)
    elif review_profile in {"game_asset_review", "icon_review"}:
        dimensions["visual_quality"] = min(100, dimensions["visual_quality"] + 3)
        dimensions["brand_fit"] = min(100, dimensions["brand_fit"] + 3)
    elif review_profile == "product_review":
        dimensions["prompt_match"] = min(100, dimensions["prompt_match"] + 3)
        dimensions["visual_quality"] = min(100, dimensions["visual_quality"] + 3)
    return dimensions


def _review_profile_for_result(result: dict[str, Any], requests: dict[str, dict[str, Any]], packs: dict[str, dict[str, Any]]) -> str:
    metadata = result.get("metadata", {}) or {}
    if metadata.get("review_profile"):
        return str(metadata.get("review_profile"))
    pack = packs.get(result.get("pack_id", ""))
    if pack and pack.get("review_profile"):
        return str(pack.get("review_profile"))
    request = requests.get(result.get("request_id", ""))
    if request and request.get("review_profile"):
        return str(request.get("review_profile"))
    return "portrait_review"


def _decision(score: float, safety: float) -> str:
    if safety < 70:
        return "reject"
    if score >= 82:
        return "keep"
    if score >= 74:
        return "revise"
    return "reject"
