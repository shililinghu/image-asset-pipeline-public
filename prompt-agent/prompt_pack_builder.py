from __future__ import annotations

from collections import defaultdict
from typing import Any

from config import (
    PACK_MAX_SERIOUS_FAIL_RATE,
    PACK_MIN_AVG_SCORE,
    PACK_MIN_KEEP_RATE,
    PROMPT_MIN_AVG_AUTO_SCORE,
    PROMPT_MIN_HUMAN_SCORE,
    PROMPT_MIN_TESTED_IMAGES,
    PROMPT_PACK_MIN_PROMPTS,
)
from models import create_prompt_pack_build_run
from prompt_pack_registry import PromptPackRegistry
from storage import JsonStore


def build_pack_from_prompt_results(
    request: dict[str, Any],
    prompt_ids: list[str] | None = None,
    store: JsonStore | None = None,
) -> dict[str, Any]:
    json_store = store or JsonStore()
    prompts = json_store.load("prompts")
    if prompt_ids:
        prompt_id_set = set(prompt_ids)
        prompts = [prompt for prompt in prompts if prompt.get("prompt_id") in prompt_id_set]
    stats = collect_prompt_image_stats(json_store, request_id=request.get("request_id", ""))
    selected: list[dict[str, Any]] = []
    rejected_ids: list[str] = []
    for prompt in prompts:
        prompt_id = str(prompt.get("prompt_id", ""))
        prompt_stats = stats.get(prompt_id, {})
        if _prompt_qualifies(prompt_stats):
            selected.append(prompt)
        elif prompt_id in stats:
            rejected_ids.append(prompt_id)

    pack_score = _average([stats.get(item.get("prompt_id", ""), {}).get("average_auto_score", 0) for item in selected])
    keep_rate = _pack_keep_rate(stats, [item.get("prompt_id", "") for item in selected])
    serious_fail_rate = _pack_serious_fail_rate(stats, [item.get("prompt_id", "") for item in selected])
    status = "approved" if (
        len(selected) >= PROMPT_PACK_MIN_PROMPTS
        and pack_score >= PACK_MIN_AVG_SCORE
        and keep_rate >= PACK_MIN_KEEP_RATE
        and serious_fail_rate <= PACK_MAX_SERIOUS_FAIL_RATE
    ) else "candidate"

    registry = PromptPackRegistry(json_store)
    pack_data = {
        "pack_name": f"Generated Pack - {request.get('topic', request.get('raw_idea', 'untitled'))}"[:120],
        "description": f"Generated from image review loop for request {request.get('request_id', '')}",
        "domain": request.get("domain", "general"),
        "platform": request.get("platform", "general"),
        "use_case": request.get("use_case", "image"),
        "style": request.get("style", "general"),
        "ratio": request.get("ratio", "1:1"),
        "status": status,
        "version": "v1",
        "quality_score": pack_score,
        "match_tags": _build_tags(request),
        "created_from": "image_loop",
        "source_prompt_ids": [item.get("prompt_id", "") for item in selected],
        "source_feedback_ids": _source_feedback_ids(json_store, selected),
        "source_image_result_ids": _source_image_result_ids(stats, selected),
    }
    items = [_pack_item_from_prompt(prompt, request, stats.get(prompt.get("prompt_id", ""), {})) for prompt in selected]
    pack = registry.create_pack(pack_data, items)
    build_run = create_prompt_pack_build_run(
        {
            "request_id": request.get("request_id", ""),
            "pack_id": pack.get("pack_id", ""),
            "source_prompt_ids": pack_data["source_prompt_ids"],
            "source_feedback_ids": pack_data["source_feedback_ids"],
            "source_image_result_ids": pack_data["source_image_result_ids"],
            "selected_prompt_ids": pack_data["source_prompt_ids"],
            "rejected_prompt_ids": rejected_ids,
            "status": "completed",
            "notes": f"status={status}; score={pack_score}; keep_rate={keep_rate}; serious_fail_rate={serious_fail_rate}",
        }
    )
    json_store.add("prompt_pack_build_runs", build_run)
    return {"pack": pack, "items": registry.get_items(pack["pack_id"]), "build_run": build_run}


def collect_prompt_image_stats(store: JsonStore, request_id: str = "") -> dict[str, dict[str, Any]]:
    results = store.load("image_results")
    auto_reviews = store.load("image_reviews")
    human_reviews = store.load("human_image_reviews")
    if request_id:
        results = [item for item in results if item.get("request_id") == request_id]
        auto_reviews = [item for item in auto_reviews if item.get("request_id") == request_id]
        human_reviews = [item for item in human_reviews if item.get("request_id") == request_id]
    result_ids_by_prompt: dict[str, list[str]] = defaultdict(list)
    for result in results:
        result_ids_by_prompt[str(result.get("prompt_id", ""))].append(str(result.get("image_result_id", "")))
    auto_by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    human_by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for review in auto_reviews:
        auto_by_prompt[str(review.get("prompt_id", ""))].append(review)
    for review in human_reviews:
        human_by_prompt[str(review.get("prompt_id", ""))].append(review)

    stats: dict[str, dict[str, Any]] = {}
    for prompt_id, image_ids in result_ids_by_prompt.items():
        autos = auto_by_prompt.get(prompt_id, [])
        humans = human_by_prompt.get(prompt_id, [])
        human_scores = [float(item.get("human_score", 0) or 0) for item in humans if float(item.get("human_score", 0) or 0) > 0]
        keep_count = sum(1 for item in humans if item.get("human_decision") in {"keep", "high_potential"})
        serious_fail_count = sum(1 for item in humans if item.get("human_decision") == "reject")
        stats[prompt_id] = {
            "tested_image_count": len(image_ids),
            "image_result_ids": image_ids,
            "average_auto_score": _average([float(item.get("auto_score", 0) or 0) for item in autos]),
            "human_average_score": _average(human_scores),
            "safety_passed": all(bool(item.get("safety_passed", True)) for item in autos),
            "keep_count": keep_count,
            "serious_fail_count": serious_fail_count,
        }
    return stats


def _prompt_qualifies(stats: dict[str, Any]) -> bool:
    return bool(stats) and (
        stats.get("tested_image_count", 0) >= PROMPT_MIN_TESTED_IMAGES
        and stats.get("average_auto_score", 0) >= PROMPT_MIN_AVG_AUTO_SCORE
        and stats.get("human_average_score", 0) >= PROMPT_MIN_HUMAN_SCORE
        and bool(stats.get("safety_passed", True))
        and stats.get("keep_count", 0) >= 1
    )


def _pack_item_from_prompt(prompt: dict[str, Any], request: dict[str, Any], stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt_id": prompt.get("prompt_id", ""),
        "prompt_content": prompt.get("full_prompt", prompt.get("prompt_content", "")),
        "negative_prompt": "",
        "variables": ["topic"],
        "use_case": request.get("use_case", "image"),
        "style_name": request.get("style", "general"),
        "ratio": request.get("ratio", "1:1"),
        "count": 1,
        "item_score": stats.get("average_auto_score", 0),
        "approved_image_count": stats.get("keep_count", 0),
        "tested_image_count": stats.get("tested_image_count", 0),
    }


def _build_tags(request: dict[str, Any]) -> list[str]:
    values = [
        request.get("domain", ""),
        request.get("platform", ""),
        request.get("use_case", ""),
        request.get("style", ""),
        "text_required" if request.get("text_required") else "",
    ]
    return [str(value) for value in values if str(value).strip()]


def _source_feedback_ids(store: JsonStore, prompts: list[dict[str, Any]]) -> list[str]:
    prompt_ids = {item.get("prompt_id", "") for item in prompts}
    return [item.get("feedback_id", "") for item in store.load("prompt_image_feedback") if item.get("prompt_id") in prompt_ids]


def _source_image_result_ids(stats: dict[str, dict[str, Any]], prompts: list[dict[str, Any]]) -> list[str]:
    image_ids: list[str] = []
    for prompt in prompts:
        image_ids.extend(stats.get(str(prompt.get("prompt_id", "")), {}).get("image_result_ids", []))
    return image_ids


def _pack_keep_rate(stats: dict[str, dict[str, Any]], prompt_ids: list[str]) -> float:
    tested = sum(int(stats.get(prompt_id, {}).get("tested_image_count", 0)) for prompt_id in prompt_ids)
    kept = sum(int(stats.get(prompt_id, {}).get("keep_count", 0)) for prompt_id in prompt_ids)
    return round(kept / tested, 3) if tested else 0


def _pack_serious_fail_rate(stats: dict[str, dict[str, Any]], prompt_ids: list[str]) -> float:
    tested = sum(int(stats.get(prompt_id, {}).get("tested_image_count", 0)) for prompt_id in prompt_ids)
    failed = sum(int(stats.get(prompt_id, {}).get("serious_fail_count", 0)) for prompt_id in prompt_ids)
    return round(failed / tested, 3) if tested else 0


def _average(values: list[float]) -> float:
    values = [float(value) for value in values if value is not None]
    return round(sum(values) / len(values), 2) if values else 0
