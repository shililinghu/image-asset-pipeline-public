from __future__ import annotations

import re
from typing import Any

from image_auto_reviewer import review_image_results
from image_generation_runner import run_prompts_for_request
from models import create_prompt_evolution_run, create_prompt_pack_run, create_prompt_version
from storage import JsonStore
from utils import new_id, now_iso


def ensure_pack_items_versioned(pack_id: str, store: JsonStore | None = None) -> None:
    json_store = store or JsonStore()
    items = json_store.load("prompt_pack_items")
    changed = False
    versions = json_store.load("prompt_versions")
    existing_version_ids = {item.get("prompt_version_id") for item in versions}
    lp_counter = _next_lp_counter(items, versions)
    for item in items:
        if item.get("pack_id") != pack_id:
            continue
        if item.get("logical_prompt_id") and item.get("active_prompt_version_id"):
            continue
        logical_id = item.get("logical_prompt_id") or f"LP{lp_counter:03d}"
        lp_counter += 1
        version_id = item.get("active_prompt_version_id") or f"{logical_id}_v1"
        item["logical_prompt_id"] = logical_id
        item["active_prompt_version_id"] = version_id
        item["status"] = item.get("status") or "active"
        item["updated_at"] = now_iso()
        if version_id not in existing_version_ids:
            versions.append(
                create_prompt_version(
                    {
                        "prompt_version_id": version_id,
                        "logical_prompt_id": logical_id,
                        "version_number": 1,
                        "pack_id": pack_id,
                        "prompt_content": item.get("prompt_content", ""),
                        "negative_prompt": item.get("negative_prompt", ""),
                        "status": "active",
                        "evolution_source": "initial",
                        "source_image_result_ids": item.get("source_image_result_ids", []),
                        "source_review_ids": item.get("source_review_ids", []),
                        "source_human_review_ids": item.get("source_human_review_ids", []),
                        "subject_profile_id": item.get("subject_profile_id", ""),
                    }
                )
            )
            existing_version_ids.add(version_id)
        changed = True
    if changed:
        json_store.save("prompt_pack_items", items)
        json_store.save("prompt_versions", versions)


def active_prompt_items(pack_id: str, items: list[dict[str, Any]], store: JsonStore | None = None) -> list[dict[str, Any]]:
    json_store = store or JsonStore()
    ensure_pack_items_versioned(pack_id, json_store)
    versions = {item.get("prompt_version_id"): item for item in json_store.load("prompt_versions")}
    result: list[dict[str, Any]] = []
    for item in items:
        if item.get("pack_id") != pack_id or item.get("status", "active") != "active":
            continue
        active_id = item.get("active_prompt_version_id", "")
        version = versions.get(active_id, {})
        merged = dict(item)
        if version:
            merged["prompt_id"] = item.get("prompt_id") or item.get("logical_prompt_id", "")
            merged["prompt_content"] = version.get("prompt_content", item.get("prompt_content", ""))
            merged["negative_prompt"] = version.get("negative_prompt", item.get("negative_prompt", ""))
            merged["active_prompt_version_id"] = active_id
            merged["logical_prompt_id"] = version.get("logical_prompt_id", item.get("logical_prompt_id", ""))
        result.append(merged)
    return result


def evolve_prompt(
    pack_id: str,
    logical_prompt_id: str,
    base_prompt_version_id: str,
    *,
    selected_image_result_ids: list[str] | None = None,
    selected_human_review_ids: list[str] | None = None,
    selected_review_ids: list[str] | None = None,
    evolution_notes: str = "",
    provider_profile_path: str = "",
    images_per_prompt: int = 1,
    validation_count: int = 3,
    store: JsonStore | None = None,
) -> dict[str, Any]:
    json_store = store or JsonStore()
    base = json_store.get("prompt_versions", "prompt_version_id", base_prompt_version_id)
    if not base:
        raise ValueError("base_prompt_version_id not found")
    next_number = _next_version_number(logical_prompt_id, json_store)
    candidate_id = f"{logical_prompt_id}_v{next_number}"
    new_prompt = _build_evolved_prompt(base.get("prompt_content", ""), evolution_notes)
    version = create_prompt_version(
        {
            "prompt_version_id": candidate_id,
            "logical_prompt_id": logical_prompt_id,
            "version_number": next_number,
            "pack_id": pack_id,
            "prompt_content": new_prompt,
            "negative_prompt": base.get("negative_prompt", ""),
            "parent_prompt_version_id": base_prompt_version_id,
            "supersedes_prompt_version_id": base_prompt_version_id,
            "status": "candidate",
            "evolution_source": "image_feedback",
            "evolution_notes": evolution_notes,
            "source_image_result_ids": selected_image_result_ids or [],
            "source_review_ids": selected_review_ids or [],
            "source_human_review_ids": selected_human_review_ids or [],
            "subject_profile_id": base.get("subject_profile_id", ""),
        }
    )
    json_store.add("prompt_versions", version)
    request = _request_for_pack(pack_id, json_store)
    prompts = [
        {
            "prompt_id": logical_prompt_id,
            "prompt_content": new_prompt,
            "base_prompt": new_prompt,
            "negative_prompt": base.get("negative_prompt", ""),
            "ratio": request.get("ratio", "3:4"),
        }
    ]
    tasks, results = run_prompts_for_request(
        request,
        prompts,
        pack_id=pack_id,
        count_per_prompt=max(1, int(validation_count or images_per_prompt or 1)),
        store=json_store,
        provider_profile_path=provider_profile_path or None,
    )
    validation_run = create_prompt_pack_run(
        {
            "request_id": request.get("request_id", ""),
            "pack_id": pack_id,
            "task_ids": [task["task_id"] for task in tasks],
            "image_result_ids": [result["image_result_id"] for result in results],
            "selected_item_count": 1,
            "images_per_prompt": max(1, int(validation_count or images_per_prompt or 1)),
            "generated_task_count": len(tasks),
            "selection_mode": "prompt_evolution",
            "variation_enabled": True,
            "status": "completed",
            "metadata": {"candidate_prompt_version_id": candidate_id},
        }
    )
    json_store.add("prompt_pack_runs", validation_run)
    reviews = review_image_results(results, store=json_store)
    validation_summary = {
        "tested_image_count": len(results),
        "approved_image_count": sum(1 for review in reviews if review.get("auto_decision") in {"keep", "manual_review"}),
        "rescued_image_count": 0,
        "average_aesthetic_score": _average_aesthetic_score(reviews),
    }
    validation_run_id = validation_run["run_id"]
    json_store.update(
        "prompt_versions",
        "prompt_version_id",
        candidate_id,
        {
            "validation_run_id": validation_run_id,
            "validation_status": "waiting_human_review",
            "validation_summary": validation_summary,
            "updated_at": now_iso(),
        },
    )
    run = create_prompt_evolution_run(
        {
            "pack_id": pack_id,
            "logical_prompt_id": logical_prompt_id,
            "base_prompt_version_id": base_prompt_version_id,
            "candidate_prompt_version_id": candidate_id,
            "status": "waiting_human_review",
            "validation_run_id": validation_run_id,
            "selected_image_result_ids": selected_image_result_ids or [],
            "selected_human_review_ids": selected_human_review_ids or [],
            "selected_review_ids": selected_review_ids or [],
            "evolution_notes": evolution_notes,
        }
    )
    json_store.add("prompt_evolution_runs", run)
    return {"run": run, "candidate_version": json_store.get("prompt_versions", "prompt_version_id", candidate_id), "tasks": tasks, "image_results": results}


def commit_prompt_evolution(
    evolution_run_id: str,
    *,
    include_high_potential: bool = True,
    include_rescue: bool = True,
    include_revise: bool = False,
    store: JsonStore | None = None,
) -> dict[str, Any]:
    json_store = store or JsonStore()
    run = json_store.get("prompt_evolution_runs", "evolution_run_id", evolution_run_id)
    if not run:
        raise ValueError("evolution_run_id not found")
    candidate_id = run.get("candidate_prompt_version_id", "")
    candidate = json_store.get("prompt_versions", "prompt_version_id", candidate_id)
    if not candidate:
        raise ValueError("candidate prompt version not found")
    accepted = {"keep"}
    if include_high_potential:
        accepted.add("high_potential")
    if include_rescue:
        accepted.add("rescue")
    if include_revise:
        accepted.add("revise")
    validation_result_ids = _result_ids_for_run(run.get("validation_run_id", ""), json_store)
    human_reviews = [item for item in json_store.load("human_image_reviews") if item.get("image_result_id") in validation_result_ids]
    if not any(item.get("human_decision") in accepted for item in human_reviews):
        raise ValueError("candidate version has no accepted human review")
    pack_id = run.get("pack_id", "")
    logical_id = run.get("logical_prompt_id", "")
    items = json_store.load("prompt_pack_items")
    item = next((row for row in items if row.get("pack_id") == pack_id and row.get("logical_prompt_id") == logical_id), None)
    if not item:
        raise ValueError("logical prompt is not mounted in pack")
    old_id = item.get("active_prompt_version_id", "")
    archive_time = now_iso()
    versions = json_store.load("prompt_versions")
    for version in versions:
        if version.get("prompt_version_id") == old_id:
            version.update(
                {
                    "status": "archived",
                    "is_archived": True,
                    "archived_at": archive_time,
                    "archive_reason": f"Replaced by {candidate_id}",
                    "superseded_by_prompt_version_id": candidate_id,
                    "updated_at": archive_time,
                }
            )
        if version.get("prompt_version_id") == candidate_id:
            version.update({"status": "active", "is_archived": False, "validation_status": "passed", "updated_at": archive_time})
    item["active_prompt_version_id"] = candidate_id
    item["prompt_content"] = candidate.get("prompt_content", item.get("prompt_content", ""))
    item["negative_prompt"] = candidate.get("negative_prompt", item.get("negative_prompt", ""))
    item["updated_at"] = archive_time
    json_store.save("prompt_versions", versions)
    json_store.save("prompt_pack_items", items)
    json_store.add(
        "prompt_version_archives",
        {
            "archive_id": new_id("prompt_archive"),
            "pack_id": pack_id,
            "logical_prompt_id": logical_id,
            "old_prompt_version_id": old_id,
            "new_prompt_version_id": candidate_id,
            "archive_reason": f"Replaced by {candidate_id}",
            "created_at": archive_time,
        },
    )
    json_store.update("prompt_evolution_runs", "evolution_run_id", evolution_run_id, {"status": "committed", "updated_at": archive_time})
    return {
        "logical_prompt_id": logical_id,
        "old_prompt_version_id": old_id,
        "new_prompt_version_id": candidate_id,
        "archived_old_version": True,
    }


def _build_evolved_prompt(base_prompt: str, notes: str) -> str:
    notes = str(notes or "").strip()
    if not notes:
        notes = "基于历史图片反馈，提升主体一致性、画面质量和可用性。"
    return f"{base_prompt.rstrip()}\n\n【进化要求】\n{notes}\n请保留原 Prompt 的核心意图，只改进上述问题。"


def _next_version_number(logical_prompt_id: str, store: JsonStore) -> int:
    numbers = [int(item.get("version_number", 0) or 0) for item in store.load("prompt_versions") if item.get("logical_prompt_id") == logical_prompt_id]
    return (max(numbers) if numbers else 0) + 1


def _next_lp_counter(items: list[dict[str, Any]], versions: list[dict[str, Any]]) -> int:
    max_seen = 0
    for value in [*(item.get("logical_prompt_id", "") for item in items), *(item.get("logical_prompt_id", "") for item in versions)]:
        match = re.match(r"LP(\d+)$", str(value))
        if match:
            max_seen = max(max_seen, int(match.group(1)))
    return max_seen + 1


def _request_for_pack(pack_id: str, store: JsonStore) -> dict[str, Any]:
    pack = store.get("prompt_packs", "pack_id", pack_id) or {}
    return {
        "request_id": f"evolve_{pack_id}",
        "ratio": pack.get("ratio", "3:4"),
        "prompt_type": pack.get("prompt_type", "portrait"),
        "review_profile": pack.get("review_profile", "portrait_review"),
        "description": pack.get("description", ""),
        "topic": pack.get("pack_name", pack_id),
    }


def _average_aesthetic_score(reviews: list[dict[str, Any]]) -> float:
    scores = [float((review.get("aesthetic", {}) or {}).get("score", 0) or 0) for review in reviews]
    return round(sum(scores) / len(scores), 2) if scores else 0


def _latest_run_id_for_results(result_ids: list[str], store: JsonStore) -> str:
    wanted = set(result_ids)
    for run in reversed(store.load("prompt_pack_runs")):
        if wanted.intersection(set(run.get("image_result_ids", []) or [])):
            return run.get("run_id", "")
    return ""


def _result_ids_for_run(run_id: str, store: JsonStore) -> set[str]:
    run = store.get("prompt_pack_runs", "run_id", run_id)
    return set((run or {}).get("image_result_ids", []) or [])
