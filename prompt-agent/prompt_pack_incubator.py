from __future__ import annotations

from typing import Any

from image_generation_runner import run_prompts_for_request
from models import create_prompt_pack_run, make_prompt
from prompt_pack_registry import PromptPackRegistry
from review_grouping import classify_reviewed_image
from storage import JsonStore
from subject_profile_builder import inject_subject_lock, subject_profile_metadata
from utils import new_id, now_iso


def generate_candidate_prompts(
    request: dict[str, Any],
    extraction: dict[str, Any],
    *,
    candidate_count: int = 8,
    store: JsonStore | None = None,
    subject_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    json_store = store or JsonStore()
    candidate_count = max(1, int(candidate_count or 8))
    extracted = extraction.get("extracted", {}) or {}
    skeletons = list(extracted.get("prompt_skeletons", []) or [])
    if not skeletons:
        skeletons = [str(request.get("description") or request.get("topic") or "image prompt")]
    negative = ", ".join(extracted.get("negative_keywords", []) or [])
    prompt_ids: list[str] = []
    previews: list[dict[str, Any]] = []
    project_id = f"candidate_{request.get('request_id', '')}"
    for index in range(candidate_count):
        skeleton = skeletons[index % len(skeletons)]
        style_hint = ", ".join((extracted.get("style_keywords", []) or [])[:3])
        composition_hint = ", ".join((extracted.get("composition_keywords", []) or [])[:2])
        full_prompt = (
            f"{skeleton}, theme: {request.get('topic', '')}, style: {request.get('style', '')}, "
            f"{style_hint}, {composition_hint}, ratio {request.get('ratio', '3:4')}"
        )
        full_prompt = inject_subject_lock(full_prompt, subject_profile)
        prompt = make_prompt(
            project_id,
            index + 1,
            f"Candidate {index + 1}",
            full_prompt,
            "candidate_image_prompt",
            prompt_type="candidate_image_prompt",
            usage_advice=f"negative prompt: {negative}",
        )
        prompt["request_id"] = request.get("request_id", "")
        prompt["extract_run_id"] = extraction.get("extract_run_id", "")
        prompt["style_name"] = request.get("style", "")
        prompt["negative_prompt"] = negative
        prompt["subject_profile_id"] = (subject_profile or {}).get("subject_profile_id", "")
        json_store.add("prompts", prompt)
        prompt_ids.append(prompt["prompt_id"])
        previews.append(
            {
                "prompt_id": prompt["prompt_id"],
                "title": prompt["title"],
                "style_name": prompt.get("style_name", ""),
                "preview": prompt["full_prompt"][:80],
            }
        )
    run = {
        "candidate_run_id": new_id("candidate_run"),
        "request_id": request.get("request_id", ""),
        "extract_run_id": extraction.get("extract_run_id", ""),
        "candidate_prompt_ids": prompt_ids,
        "candidate_prompt_count": len(prompt_ids),
        "created_at": now_iso(),
    }
    json_store.add("candidate_prompt_runs", run)
    return {"run": run, "candidate_prompt_ids": prompt_ids, "previews": previews}


def validate_candidate_prompts(
    request: dict[str, Any],
    candidate_prompt_ids: list[str],
    *,
    provider_profile_path: str = "",
    images_per_prompt: int = 1,
    store: JsonStore | None = None,
    subject_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    json_store = store or JsonStore()
    if not request.get("request_id") or not candidate_prompt_ids:
        raise ValueError("request_id and candidate_prompt_ids are required")
    prompt_map = {item.get("prompt_id"): item for item in json_store.load("prompts")}
    prompts = []
    for prompt_id in candidate_prompt_ids:
        prompt = prompt_map.get(prompt_id)
        if not prompt:
            continue
        prompts.append(
            {
                "prompt_id": prompt_id,
                "prompt_content": prompt.get("full_prompt", ""),
                "negative_prompt": prompt.get("negative_prompt", ""),
                "ratio": request.get("ratio", "3:4"),
            }
        )
    if not prompts:
        raise ValueError("candidate prompts not found")
    tasks, results = run_prompts_for_request(
        request,
        prompts,
        pack_id="candidate_validation",
        count_per_prompt=max(1, int(images_per_prompt or 1)),
        store=json_store,
        provider_profile_path=provider_profile_path or None,
        subject_profile=subject_profile,
    )
    run = create_prompt_pack_run(
        {
            "request_id": request.get("request_id", ""),
            "pack_id": "candidate_validation",
            "task_ids": [task["task_id"] for task in tasks],
            "image_result_ids": [result["image_result_id"] for result in results],
            "selected_item_count": len(prompts),
            "images_per_prompt": max(1, int(images_per_prompt or 1)),
            "generated_task_count": len(tasks),
            "selection_mode": "candidate_prompts",
            "variation_enabled": True,
            "metadata": {"candidate_prompt_ids": candidate_prompt_ids, **subject_profile_metadata(subject_profile)},
            "subject_profile_id": (subject_profile or {}).get("subject_profile_id", ""),
            "status": "completed",
        }
    )
    json_store.add("prompt_pack_runs", run)
    json_store.add(
        "candidate_prompt_runs",
        {
            "candidate_run_id": new_id("candidate_validate"),
            "request_id": request.get("request_id", ""),
            "candidate_prompt_ids": candidate_prompt_ids,
            "validation_run_id": run["run_id"],
            "image_result_ids": [result["image_result_id"] for result in results],
            "subject_profile_id": (subject_profile or {}).get("subject_profile_id", ""),
            "created_at": now_iso(),
        },
    )
    return {"run": run, "tasks": tasks, "image_results": results}


def build_prompt_pack_from_reviewed_candidates(
    request: dict[str, Any],
    candidate_prompt_ids: list[str],
    *,
    min_aesthetic_score: float = 70,
    accepted_decisions: list[str] | None = None,
    pack_name: str = "",
    status: str = "candidate",
    store: JsonStore | None = None,
) -> dict[str, Any]:
    json_store = store or JsonStore()
    accepted = set(accepted_decisions or ["keep", "high_potential"])
    human_reviews = json_store.load("human_image_reviews")
    image_reviews = json_store.load("image_reviews")
    image_results = json_store.load("image_results")
    prompts = {item.get("prompt_id"): item for item in json_store.load("prompts")}
    result_by_id = {item.get("image_result_id"): item for item in image_results}

    kept_prompt_ids: set[str] = set()
    rejected_prompt_ids: set[str] = set()
    source_image_result_ids: set[str] = set()
    source_feedback_ids: set[str] = set()
    for review in human_reviews:
        prompt_id = str(review.get("prompt_id", ""))
        if prompt_id not in candidate_prompt_ids:
            continue
        if review.get("human_decision") in accepted:
            kept_prompt_ids.add(prompt_id)
            source_feedback_ids.add(review.get("human_review_id", ""))
            source_image_result_ids.add(review.get("image_result_id", ""))
        elif review.get("human_decision") == "reject":
            rejected_prompt_ids.add(prompt_id)

    for review in image_reviews:
        prompt_id = str(review.get("prompt_id", ""))
        if prompt_id not in candidate_prompt_ids or prompt_id in rejected_prompt_ids:
            continue
        aesthetic = review.get("aesthetic", {}) or {}
        score = float(aesthetic.get("score", 0) or 0)
        if score >= float(min_aesthetic_score or 70):
            kept_prompt_ids.add(prompt_id)
            source_image_result_ids.add(review.get("image_result_id", ""))

    kept_prompt_ids = {prompt_id for prompt_id in kept_prompt_ids if prompt_id in prompts}
    if not kept_prompt_ids:
        raise ValueError("no reviewed candidate prompts qualified for pack building")

    registry = PromptPackRegistry(json_store)
    pack = registry.create_pack(
        {
            "pack_name": pack_name or f"{request.get('title') or request.get('topic') or 'Incubated'} Prompt Pack",
            "description": request.get("description", ""),
            "prompt_type": request.get("prompt_type", "portrait"),
            "review_profile": request.get("review_profile", "portrait_review"),
            "domain": request.get("domain", "general"),
            "platform": request.get("platform", "general"),
            "use_case": request.get("use_case", "image"),
            "style": request.get("style", "general"),
            "ratio": request.get("ratio", "3:4"),
            "status": status or "candidate",
            "version": "v1",
            "match_tags": [request.get("domain", ""), request.get("style", ""), request.get("use_case", "")],
            "created_from": "incubation",
            "source_prompt_ids": sorted(kept_prompt_ids),
            "source_feedback_ids": sorted(item for item in source_feedback_ids if item),
            "source_image_result_ids": sorted(item for item in source_image_result_ids if item),
        },
        [
            {
                "prompt_id": prompt_id,
                "prompt_content": prompts[prompt_id].get("full_prompt", ""),
                "negative_prompt": prompts[prompt_id].get("negative_prompt", ""),
                "variables": ["topic"],
                "use_case": request.get("use_case", "image"),
                "style_name": request.get("style", ""),
                "ratio": request.get("ratio", "3:4"),
                "count": 1,
            }
            for prompt_id in sorted(kept_prompt_ids)
        ],
    )
    json_store.add(
        "prompt_pack_incubation_runs",
        {
            "incubation_run_id": new_id("incubation"),
            "request_id": request.get("request_id", ""),
            "candidate_prompt_ids": candidate_prompt_ids,
            "built_pack_id": pack["pack_id"],
            "status": "completed",
            "created_at": now_iso(),
            "updated_at": now_iso(),
        },
    )
    return {
        "pack": pack,
        "kept_prompt_count": len(kept_prompt_ids),
        "rejected_prompt_count": len(set(candidate_prompt_ids) - kept_prompt_ids),
        "source_image_result_ids": sorted(item for item in source_image_result_ids if item),
    }


def build_prompt_pack_from_incubation(
    incubation_run: dict[str, Any],
    *,
    pack_name: str = "",
    accepted_decisions: list[str] | None = None,
    min_aesthetic_score: float = 70,
    include_unmarked_high_score: bool = False,
    include_revise: bool = False,
    include_subject_drift: bool = False,
    status: str = "candidate",
    store: JsonStore | None = None,
) -> dict[str, Any]:
    json_store = store or JsonStore()
    request = json_store.get("idea_requests", "request_id", incubation_run.get("request_id", "")) or {}
    subject_profile = json_store.get("subject_profiles", "subject_profile_id", incubation_run.get("subject_profile_id", "")) or {}
    prompt_ids = list(incubation_run.get("candidate_prompt_ids", []) or [])
    accepted = set(accepted_decisions or ["keep", "high_potential", "rescue"])
    if include_revise:
        accepted.add("revise")
    result_ids = set(_validation_image_result_ids(incubation_run, json_store))
    image_results = [item for item in json_store.load("image_results") if item.get("image_result_id") in result_ids]
    image_reviews = [item for item in json_store.load("image_reviews") if item.get("image_result_id") in result_ids]
    human_reviews = [item for item in json_store.load("human_image_reviews") if item.get("image_result_id") in result_ids]
    prompts = {item.get("prompt_id"): item for item in json_store.load("prompts")}
    reviews_by_image = {item.get("image_result_id"): item for item in image_reviews}
    human_by_image = {item.get("image_result_id"): item for item in human_reviews}
    tested_counts: dict[str, int] = {}
    for result in image_results:
        prompt_id = str(result.get("prompt_id", ""))
        if prompt_id in prompt_ids:
            tested_counts[prompt_id] = tested_counts.get(prompt_id, 0) + 1

    grouped: dict[str, dict[str, Any]] = {}
    for result in image_results:
        prompt_id = str(result.get("prompt_id", ""))
        if prompt_id not in prompt_ids:
            continue
        review = reviews_by_image.get(result.get("image_result_id"), {})
        human = human_by_image.get(result.get("image_result_id"), {})
        aesthetic = review.get("aesthetic", {}) if isinstance(review, dict) else {}
        score = float((aesthetic or {}).get("score", 0) or 0)
        human_decision = human.get("human_decision", "")
        if human.get("subject_consistency") == "drifted" and not include_subject_drift:
            continue
        qualifies = human_decision in accepted or (include_unmarked_high_score and not human_decision and score >= float(min_aesthetic_score or 70))
        if not qualifies:
            continue
        auto_reasons = _incubation_auto_reject_reasons(result, review, min_aesthetic_score)
        bucket = grouped.setdefault(
            prompt_id,
            {
                "source_image_result_ids": [],
                "source_review_ids": [],
                "source_human_review_ids": [],
                "source_selection_decisions": [],
                "auto_reject_reasons": [],
                "rescued_from_filtered_out": False,
                "tested_image_count": tested_counts.get(prompt_id, 0),
                "approved_image_count": 0,
            },
        )
        bucket["approved_image_count"] += 1
        bucket["source_image_result_ids"].append(result.get("image_result_id", ""))
        bucket["source_selection_decisions"].append(human_decision or "auto_high_score")
        bucket["auto_reject_reasons"].extend(auto_reasons)
        if human_decision == "rescue":
            bucket["rescued_from_filtered_out"] = True
        if review:
            bucket["source_review_ids"].append(review.get("review_id", ""))
        if human:
            bucket["source_human_review_ids"].append(human.get("human_review_id", ""))

    kept_prompt_ids = [prompt_id for prompt_id in prompt_ids if prompt_id in grouped and prompt_id in prompts]
    if not kept_prompt_ids:
        raise ValueError("no reviewed incubation images qualified for pack building")

    registry = PromptPackRegistry(json_store)
    source_image_ids = sorted({image_id for data in grouped.values() for image_id in data["source_image_result_ids"] if image_id})
    source_feedback_ids = sorted({review_id for data in grouped.values() for review_id in data["source_human_review_ids"] if review_id})
    pack = registry.create_pack(
        {
            "pack_name": pack_name or f"{request.get('title') or request.get('topic') or 'Incubated'} Prompt Pack",
            "description": request.get("description", ""),
            "prompt_type": request.get("prompt_type", "portrait"),
            "review_profile": request.get("review_profile", "portrait_review"),
            "domain": request.get("domain", "general"),
            "platform": request.get("platform", "general"),
            "use_case": request.get("use_case", "image"),
            "style": request.get("style", "general"),
            "ratio": request.get("ratio", "3:4"),
            "status": status or "candidate",
            "version": "v1",
            "match_tags": [request.get("domain", ""), request.get("style", ""), request.get("use_case", "")],
            "created_from": "incubation",
            "source_prompt_ids": kept_prompt_ids,
            "source_feedback_ids": source_feedback_ids,
            "source_image_result_ids": source_image_ids,
            "subject_profile_id": subject_profile.get("subject_profile_id", ""),
            "subject_lock_required": bool(subject_profile),
            "subject_description": subject_profile.get("subject_description", ""),
            "reference_image_ids": list(subject_profile.get("reference_image_ids", []) or []),
        },
        [
            {
                "prompt_id": prompt_id,
                "prompt_content": prompts[prompt_id].get("full_prompt", ""),
                "negative_prompt": prompts[prompt_id].get("negative_prompt", ""),
                "variables": ["topic"],
                "use_case": request.get("use_case", "image"),
                "style_name": request.get("style", ""),
                "ratio": request.get("ratio", "3:4"),
                "count": 1,
                "source_image_result_ids": grouped[prompt_id]["source_image_result_ids"],
                "source_review_ids": grouped[prompt_id]["source_review_ids"],
                "source_human_review_ids": grouped[prompt_id]["source_human_review_ids"],
                "source_incubation_run_id": incubation_run.get("incubation_run_id", ""),
                "subject_profile_id": subject_profile.get("subject_profile_id", ""),
                "subject_lock_instruction": subject_profile.get("consistency_instruction", ""),
                "source_selection_decisions": sorted(set(grouped[prompt_id]["source_selection_decisions"])),
                "rescued_from_filtered_out": bool(grouped[prompt_id]["rescued_from_filtered_out"]),
                "auto_reject_reasons": sorted(set(reason for reason in grouped[prompt_id]["auto_reject_reasons"] if reason)),
                "tested_image_count": grouped[prompt_id]["tested_image_count"],
                "approved_image_count": grouped[prompt_id]["approved_image_count"],
            }
            for prompt_id in kept_prompt_ids
        ],
    )
    return {
        "pack": pack,
        "kept_prompt_count": len(kept_prompt_ids),
        "rejected_prompt_count": len(set(prompt_ids) - set(kept_prompt_ids)),
        "kept_prompt_ids": kept_prompt_ids,
        "rescued_prompt_count": sum(1 for prompt_id in kept_prompt_ids if grouped[prompt_id]["rescued_from_filtered_out"]),
    }


def _incubation_auto_reject_reasons(result: dict[str, Any], review: dict[str, Any], min_score: float) -> list[str]:
    return list(classify_reviewed_image(result, review).get("auto_reject_reasons", []) or [])


def _validation_image_result_ids(incubation_run: dict[str, Any], store: JsonStore) -> list[str]:
    validation_run_id = incubation_run.get("validation_run_id", "")
    run = store.get("prompt_pack_runs", "run_id", validation_run_id) if validation_run_id else None
    if run:
        return list(run.get("image_result_ids", []) or [])
    return list(incubation_run.get("image_result_ids", []) or [])
