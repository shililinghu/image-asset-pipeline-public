from __future__ import annotations

from collections import defaultdict
from typing import Any

from models import create_prompt_image_feedback, make_feedback
from storage import JsonStore


def build_prompt_image_feedback(
    request_id: str = "",
    store: JsonStore | None = None,
    sync_legacy_feedback: bool = True,
) -> list[dict[str, Any]]:
    json_store = store or JsonStore()
    image_results = json_store.load("image_results")
    auto_reviews = json_store.load("image_reviews")
    human_reviews = json_store.load("human_image_reviews")
    if request_id:
        image_results = [item for item in image_results if item.get("request_id") == request_id]
        auto_reviews = [item for item in auto_reviews if item.get("request_id") == request_id]
        human_reviews = [item for item in human_reviews if item.get("request_id") == request_id]

    result_by_id = {item.get("image_result_id"): item for item in image_results}
    auto_by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    human_by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for review in auto_reviews:
        if review.get("image_result_id") in result_by_id:
            auto_by_prompt[str(review.get("prompt_id", ""))].append(review)
    for review in human_reviews:
        if review.get("image_result_id") in result_by_id:
            human_by_prompt[str(review.get("prompt_id", ""))].append(review)

    feedback_items: list[dict[str, Any]] = []
    existing_keys = {
        (item.get("request_id"), item.get("prompt_id"))
        for item in json_store.load("prompt_image_feedback")
    }
    prompt_ids = sorted(set(auto_by_prompt) | set(human_by_prompt))
    for prompt_id in prompt_ids:
        if not prompt_id or (request_id, prompt_id) in existing_keys:
            continue
        autos = auto_by_prompt.get(prompt_id, [])
        humans = human_by_prompt.get(prompt_id, [])
        feedback = _summarize_prompt_feedback(prompt_id, request_id, autos, humans)
        json_store.add("prompt_image_feedback", feedback)
        feedback_items.append(feedback)
        if sync_legacy_feedback:
            _sync_to_legacy_feedback(json_store, feedback)
    return feedback_items


def _summarize_prompt_feedback(
    prompt_id: str,
    request_id: str,
    auto_reviews: list[dict[str, Any]],
    human_reviews: list[dict[str, Any]],
) -> dict[str, Any]:
    auto_scores = [float(item.get("auto_score", 0) or 0) for item in auto_reviews]
    human_scores = [float(item.get("human_score", 0) or 0) for item in human_reviews if float(item.get("human_score", 0) or 0) > 0]
    keep_count = sum(1 for item in human_reviews if item.get("human_decision") in {"keep", "high_potential"})
    reject_count = sum(1 for item in human_reviews if item.get("human_decision") == "reject")
    average_auto = round(sum(auto_scores) / len(auto_scores), 2) if auto_scores else 0
    average_human = round(sum(human_scores) / len(human_scores), 2) if human_scores else 0
    positive_patterns = []
    negative_patterns = []
    suggestions = []
    if average_auto >= 75:
        positive_patterns.append("mock auto review indicates acceptable prompt-image alignment")
    else:
        negative_patterns.append("auto score is below pack threshold")
        suggestions.append("make the prompt more concrete about subject, layout, and platform fit")
    if keep_count:
        positive_patterns.append("human review kept or marked high potential images")
    if reject_count:
        negative_patterns.append("human review rejected some images")
        suggestions.append("reduce ambiguity and add clearer visual constraints")
    summary = (
        f"Image loop feedback for {prompt_id}: "
        f"avg_auto={average_auto}, avg_human={average_human}, keep_or_high={keep_count}, reject={reject_count}."
    )
    return create_prompt_image_feedback(
        {
            "prompt_id": prompt_id,
            "request_id": request_id,
            "summary": summary,
            "positive_patterns": positive_patterns,
            "negative_patterns": negative_patterns,
            "optimization_suggestions": suggestions,
            "based_on_image_results": [item.get("image_result_id", "") for item in auto_reviews],
            "based_on_human_reviews": [item.get("human_review_id", "") for item in human_reviews],
        }
    )


def _sync_to_legacy_feedback(store: JsonStore, feedback: dict[str, Any]) -> None:
    prompt = store.get("prompts", "prompt_id", feedback.get("prompt_id", ""))
    if not prompt:
        return
    legacy = make_feedback(
        {
            "project_id": prompt.get("project_id", ""),
            "prompt_id": feedback.get("prompt_id", ""),
            "real_scenario": "image_review_loop",
            "what_worked": "; ".join(feedback.get("positive_patterns", [])),
            "what_failed": "; ".join(feedback.get("negative_patterns", [])),
            "voice_match_score": 0,
            "continue_use": "yes",
            "need_more": "; ".join(feedback.get("optimization_suggestions", [])),
            "need_less": "",
            "overall_score": 0,
            "source_feedback_id": feedback.get("feedback_id", ""),
        }
    )
    store.add("feedback", legacy)
