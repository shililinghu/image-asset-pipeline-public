from __future__ import annotations

from typing import Any

from utils import new_id, now_iso


def make_project(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "project_id": new_id("project"),
        "name": data["name"],
        "scenario": data["scenario"],
        "goal": data["goal"],
        "audience": data["audience"],
        "desired_output": data["desired_output"],
        "avoid": data["avoid"],
        "style_preference": data["style_preference"],
        "evaluation_criteria": data["evaluation_criteria"],
        "created_at": now_iso(),
    }


def make_test_case(project_id: str, title: str, case_input: str, expected_focus: str) -> dict[str, Any]:
    return {
        "case_id": new_id("case"),
        "project_id": project_id,
        "title": title,
        "input": case_input,
        "expected_focus": expected_focus,
        "created_at": now_iso(),
    }


def make_prompt(project_id: str, version: int, title: str, full_prompt: str, kind: str, **extra: Any) -> dict[str, Any]:
    return {
        "prompt_id": new_id("prompt"),
        "project_id": project_id,
        "version": version,
        "kind": kind,
        "prompt_type": extra.get("prompt_type", kind),
        "title": title,
        "full_prompt": full_prompt,
        "suitable_for": extra.get("suitable_for", ""),
        "not_suitable_for": extra.get("not_suitable_for", ""),
        "risk": extra.get("risk", ""),
        "usage_advice": extra.get("usage_advice", ""),
        "source_prompt_ids": list(extra.get("source_prompt_ids", []) or []),
        "fusion_notes": dict(extra.get("fusion_notes", {}) or {}),
        "created_at": now_iso(),
    }


def make_test_result(project_id: str, prompt_id: str, case_id: str, output: str, mode: str) -> dict[str, Any]:
    return {
        "result_id": new_id("result"),
        "project_id": project_id,
        "prompt_id": prompt_id,
        "case_id": case_id,
        "output": output,
        "mode": mode,
        "created_at": now_iso(),
    }


def make_score(
    project_id: str,
    prompt_id: str,
    case_id: str,
    test_result_id: str,
    scores: dict[str, float | None],
    strengths: list[str] | str,
    weaknesses: list[str] | str,
    suggestions: list[str] | str,
    *,
    scoring_mode: str = "general",
    total_score: float | None = None,
    risk_level: str = "low",
    hard_reject_reasons: list[str] | None = None,
    next_variable_to_change: str = "",
    attention_budget_diagnosis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if total_score is None:
        valid_scores = [float(value) for value in scores.values() if value is not None]
        total_score = round(sum(valid_scores) / len(valid_scores), 2) if valid_scores else 0
    return {
        "score_id": new_id("score"),
        "project_id": project_id,
        "prompt_id": prompt_id,
        "case_id": case_id,
        "test_result_id": test_result_id,
        "scoring_mode": scoring_mode,
        "scores": scores,
        "total_score": total_score,
        "risk_level": risk_level,
        "hard_reject_reasons": hard_reject_reasons or [],
        "strengths": strengths if isinstance(strengths, list) else [strengths] if strengths else [],
        "weaknesses": weaknesses if isinstance(weaknesses, list) else [weaknesses] if weaknesses else [],
        "improvement_suggestions": suggestions if isinstance(suggestions, list) else [suggestions] if suggestions else [],
        "next_variable_to_change": next_variable_to_change,
        "attention_budget_diagnosis": attention_budget_diagnosis or {},
        "created_at": now_iso(),
    }


def make_feedback(data: dict[str, Any]) -> dict[str, Any]:
    data = dict(data)
    data["feedback_id"] = new_id("feedback")
    data["created_at"] = now_iso()
    return data


def make_iteration(project_id: str, old_prompt_id: str, new_prompt_id: str, version: int, changed_points: str, reason: str) -> dict[str, Any]:
    return {
        "iteration_id": new_id("iteration"),
        "project_id": project_id,
        "old_prompt_id": old_prompt_id,
        "new_prompt_id": new_prompt_id,
        "version": version,
        "changed_points": changed_points,
        "reason": reason,
        "created_at": now_iso(),
    }


def create_idea_request(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "request_id": data.get("request_id") or new_id("request"),
        "raw_idea": data.get("raw_idea", ""),
        "domain": data.get("domain", "general"),
        "platform": data.get("platform", "general"),
        "use_case": data.get("use_case", "image"),
        "style": data.get("style", "general"),
        "ratio": data.get("ratio", "1:1"),
        "topic": data.get("topic", data.get("raw_idea", "")),
        "constraints": list(data.get("constraints", []) or []),
        "target_audience": data.get("target_audience", ""),
        "text_required": bool(data.get("text_required", True)),
        "batch_size": int(data.get("batch_size", 10) or 10),
        "created_at": data.get("created_at") or now_iso(),
    }


def create_subject_profile(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "subject_profile_id": data.get("subject_profile_id") or new_id("subject"),
        "request_id": data.get("request_id", ""),
        "source": data.get("source", "text_description"),
        "subject_title": data.get("subject_title", ""),
        "subject_description": data.get("subject_description", ""),
        "identity_keywords": list(data.get("identity_keywords", []) or []),
        "visual_anchors": list(data.get("visual_anchors", []) or []),
        "must_keep": list(data.get("must_keep", []) or []),
        "must_avoid": list(data.get("must_avoid", []) or []),
        "reference_image_ids": list(data.get("reference_image_ids", []) or []),
        "reference_image_paths": list(data.get("reference_image_paths", []) or []),
        "consistency_instruction": data.get("consistency_instruction", ""),
        "created_at": data.get("created_at") or now_iso(),
    }


def create_prompt_pack(data: dict[str, Any]) -> dict[str, Any]:
    timestamp = now_iso()
    return {
        "pack_id": data.get("pack_id") or new_id("pack"),
        "pack_name": data.get("pack_name", "Untitled Prompt Pack"),
        "description": data.get("description", ""),
        "prompt_type": data.get("prompt_type", "portrait"),
        "review_profile": data.get("review_profile", "portrait_review"),
        "domain": data.get("domain", "general"),
        "platform": data.get("platform", "general"),
        "use_case": data.get("use_case", "image"),
        "style": data.get("style", "general"),
        "ratio": data.get("ratio", "1:1"),
        "status": data.get("status", "draft"),
        "version": data.get("version", "v1"),
        "quality_score": float(data.get("quality_score", 0) or 0),
        "match_tags": list(data.get("match_tags", []) or []),
        "created_from": data.get("created_from", "manual"),
        "source_prompt_ids": list(data.get("source_prompt_ids", []) or []),
        "source_feedback_ids": list(data.get("source_feedback_ids", []) or []),
        "source_image_result_ids": list(data.get("source_image_result_ids", []) or []),
        "subject_profile_id": data.get("subject_profile_id", ""),
        "subject_lock_required": bool(data.get("subject_lock_required", False)),
        "subject_description": data.get("subject_description", ""),
        "reference_image_ids": list(data.get("reference_image_ids", []) or []),
        "created_at": data.get("created_at") or timestamp,
        "updated_at": data.get("updated_at") or timestamp,
    }


def create_prompt_pack_item(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "item_id": data.get("item_id") or new_id("pack_item"),
        "pack_id": data.get("pack_id", ""),
        "logical_prompt_id": data.get("logical_prompt_id", ""),
        "active_prompt_version_id": data.get("active_prompt_version_id", ""),
        "status": data.get("status", "active"),
        "prompt_id": data.get("prompt_id", ""),
        "prompt_content": data.get("prompt_content", ""),
        "negative_prompt": data.get("negative_prompt", ""),
        "variables": list(data.get("variables", []) or []),
        "use_case": data.get("use_case", "image"),
        "style_name": data.get("style_name", data.get("style", "")),
        "ratio": data.get("ratio", "1:1"),
        "count": _safe_positive_int(data.get("count", 1), 1),
        "item_score": float(data.get("item_score", 0) or 0),
        "approved_image_count": int(data.get("approved_image_count", 0) or 0),
        "tested_image_count": int(data.get("tested_image_count", 0) or 0),
        "source_image_result_ids": list(data.get("source_image_result_ids", []) or []),
        "source_review_ids": list(data.get("source_review_ids", []) or []),
        "source_human_review_ids": list(data.get("source_human_review_ids", []) or []),
        "source_incubation_run_id": data.get("source_incubation_run_id", ""),
        "source_selection_decisions": list(data.get("source_selection_decisions", []) or []),
        "rescued_from_filtered_out": bool(data.get("rescued_from_filtered_out", False)),
        "auto_reject_reasons": list(data.get("auto_reject_reasons", []) or []),
        "subject_profile_id": data.get("subject_profile_id", ""),
        "subject_lock_instruction": data.get("subject_lock_instruction", ""),
        "metadata": dict(data.get("metadata", {}) or {}),
        "created_at": data.get("created_at") or now_iso(),
        "updated_at": data.get("updated_at") or data.get("created_at") or now_iso(),
    }


def create_prompt_version(data: dict[str, Any]) -> dict[str, Any]:
    timestamp = now_iso()
    return {
        "prompt_version_id": data.get("prompt_version_id", ""),
        "logical_prompt_id": data.get("logical_prompt_id", ""),
        "version_number": int(data.get("version_number", 1) or 1),
        "pack_id": data.get("pack_id", ""),
        "prompt_content": data.get("prompt_content", ""),
        "negative_prompt": data.get("negative_prompt", ""),
        "parent_prompt_version_id": data.get("parent_prompt_version_id", ""),
        "supersedes_prompt_version_id": data.get("supersedes_prompt_version_id", ""),
        "superseded_by_prompt_version_id": data.get("superseded_by_prompt_version_id", ""),
        "status": data.get("status", "candidate"),
        "is_archived": bool(data.get("is_archived", False)),
        "archived_at": data.get("archived_at", ""),
        "archive_reason": data.get("archive_reason", ""),
        "evolution_source": data.get("evolution_source", "initial"),
        "evolution_notes": data.get("evolution_notes", ""),
        "source_image_result_ids": list(data.get("source_image_result_ids", []) or []),
        "source_review_ids": list(data.get("source_review_ids", []) or []),
        "source_human_review_ids": list(data.get("source_human_review_ids", []) or []),
        "source_prompt_feedback_ids": list(data.get("source_prompt_feedback_ids", []) or []),
        "validation_run_id": data.get("validation_run_id", ""),
        "validation_status": data.get("validation_status", ""),
        "validation_summary": dict(data.get("validation_summary", {}) or {}),
        "subject_profile_id": data.get("subject_profile_id", ""),
        "created_at": data.get("created_at") or timestamp,
        "updated_at": data.get("updated_at") or timestamp,
    }


def create_prompt_evolution_run(data: dict[str, Any]) -> dict[str, Any]:
    timestamp = now_iso()
    return {
        "evolution_run_id": data.get("evolution_run_id") or new_id("evolution_run"),
        "pack_id": data.get("pack_id", ""),
        "logical_prompt_id": data.get("logical_prompt_id", ""),
        "base_prompt_version_id": data.get("base_prompt_version_id", ""),
        "candidate_prompt_version_id": data.get("candidate_prompt_version_id", ""),
        "status": data.get("status", "running"),
        "validation_run_id": data.get("validation_run_id", ""),
        "selected_image_result_ids": list(data.get("selected_image_result_ids", []) or []),
        "selected_human_review_ids": list(data.get("selected_human_review_ids", []) or []),
        "selected_review_ids": list(data.get("selected_review_ids", []) or []),
        "evolution_notes": data.get("evolution_notes", ""),
        "created_at": data.get("created_at") or timestamp,
        "updated_at": data.get("updated_at") or timestamp,
    }


def create_prompt_pack_match_log(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "match_id": data.get("match_id") or new_id("match"),
        "request_id": data.get("request_id", ""),
        "matched": bool(data.get("matched", False)),
        "pack_id": data.get("pack_id", ""),
        "match_score": int(data.get("match_score", 0) or 0),
        "match_reasons": list(data.get("match_reasons", []) or []),
        "missing_requirements": list(data.get("missing_requirements", []) or []),
        "decision": data.get("decision", "incubate_new_pack"),
        "manual_selected": bool(data.get("manual_selected", False)),
        "created_at": data.get("created_at") or now_iso(),
    }


def _safe_positive_int(value: Any, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return number if number > 0 else default


def create_prompt_pack_run(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": data.get("run_id") or new_id("pack_run"),
        "request_id": data.get("request_id", ""),
        "pack_id": data.get("pack_id", ""),
        "task_ids": list(data.get("task_ids", []) or []),
        "image_result_ids": list(data.get("image_result_ids", []) or []),
        "selected_item_count": int(data.get("selected_item_count", 0) or 0),
        "images_per_prompt": data.get("images_per_prompt", 1),
        "generated_task_count": int(data.get("generated_task_count", 0) or 0),
        "selection_mode": data.get("selection_mode", ""),
        "variation_enabled": bool(data.get("variation_enabled", False)),
        "subject_profile_id": data.get("subject_profile_id", ""),
        "metadata": dict(data.get("metadata", {}) or {}),
        "status": data.get("status", "completed"),
        "created_at": data.get("created_at") or now_iso(),
    }


def create_prompt_pack_build_run(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "build_run_id": data.get("build_run_id") or new_id("pack_build"),
        "request_id": data.get("request_id", ""),
        "pack_id": data.get("pack_id", ""),
        "source_prompt_ids": list(data.get("source_prompt_ids", []) or []),
        "source_feedback_ids": list(data.get("source_feedback_ids", []) or []),
        "source_image_result_ids": list(data.get("source_image_result_ids", []) or []),
        "selected_prompt_ids": list(data.get("selected_prompt_ids", []) or []),
        "rejected_prompt_ids": list(data.get("rejected_prompt_ids", []) or []),
        "status": data.get("status", "completed"),
        "notes": data.get("notes", ""),
        "created_at": data.get("created_at") or now_iso(),
    }


def create_image_generation_task(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": data.get("task_id") or new_id("img_task"),
        "run_id": data.get("run_id", ""),
        "request_id": data.get("request_id", ""),
        "pack_id": data.get("pack_id", ""),
        "prompt_id": data.get("prompt_id", ""),
        "variant_index": int(data.get("variant_index", 1) or 1),
        "images_per_prompt": int(data.get("images_per_prompt", 1) or 1),
        "prompt_nonce": data.get("prompt_nonce", ""),
        "base_prompt": data.get("base_prompt", data.get("prompt_content", "")),
        "submitted_prompt": data.get("submitted_prompt", data.get("prompt_content", "")),
        "prompt_content": data.get("prompt_content", ""),
        "subject_profile_id": data.get("subject_profile_id", ""),
        "subject_lock_enabled": bool(data.get("subject_lock_enabled", False)),
        "subject_description": data.get("subject_description", ""),
        "negative_prompt": data.get("negative_prompt", ""),
        "ratio": data.get("ratio", "1:1"),
        "provider": data.get("provider", "mock"),
        "model": data.get("model", "mock-image-v0"),
        "metadata": dict(data.get("metadata", {}) or {}),
        "status": data.get("status", "pending"),
        "created_at": data.get("created_at") or now_iso(),
    }


def create_image_result(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "image_result_id": data.get("image_result_id") or new_id("image"),
        "task_id": data.get("task_id", ""),
        "request_id": data.get("request_id", ""),
        "pack_id": data.get("pack_id", ""),
        "prompt_id": data.get("prompt_id", ""),
        "image_path": data.get("image_path", ""),
        "image_url": data.get("image_url", ""),
        "provider": data.get("provider", "mock"),
        "model": data.get("model", ""),
        "status": data.get("status", "success"),
        "error_message": data.get("error_message", ""),
        "metadata": dict(data.get("metadata", {}) or {}),
        "created_at": data.get("created_at") or now_iso(),
    }


def create_image_review(data: dict[str, Any]) -> dict[str, Any]:
    dimensions = dict(data.get("dimensions", {}) or {})
    return {
        "review_id": data.get("review_id") or new_id("img_review"),
        "image_result_id": data.get("image_result_id", ""),
        "request_id": data.get("request_id", ""),
        "pack_id": data.get("pack_id", ""),
        "prompt_id": data.get("prompt_id", ""),
        "dimensions": dimensions,
        "auto_score": float(data.get("auto_score", 0) or 0),
        "auto_decision": data.get("auto_decision", "revise"),
        "safety_passed": bool(data.get("safety_passed", True)),
        "notes": data.get("notes", ""),
        "created_at": data.get("created_at") or now_iso(),
    }


def create_human_image_review(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "human_review_id": data.get("human_review_id") or new_id("human_img_review"),
        "image_result_id": data.get("image_result_id", ""),
        "request_id": data.get("request_id", ""),
        "pack_id": data.get("pack_id", ""),
        "prompt_id": data.get("prompt_id", ""),
        "human_decision": data.get("human_decision", "revise"),
        "human_score": float(data.get("human_score", 0) or 0),
        "reject_reasons": data.get("reject_reasons", ""),
        "notes": data.get("notes", ""),
        "source": data.get("source", ""),
        "subject_consistency": data.get("subject_consistency", ""),
        "created_at": data.get("created_at") or now_iso(),
    }


def create_prompt_image_feedback(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "feedback_id": data.get("feedback_id") or new_id("img_feedback"),
        "prompt_id": data.get("prompt_id", ""),
        "request_id": data.get("request_id", ""),
        "source": data.get("source", "image_review_loop"),
        "summary": data.get("summary", ""),
        "positive_patterns": list(data.get("positive_patterns", []) or []),
        "negative_patterns": list(data.get("negative_patterns", []) or []),
        "optimization_suggestions": list(data.get("optimization_suggestions", []) or []),
        "based_on_image_results": list(data.get("based_on_image_results", []) or []),
        "based_on_human_reviews": list(data.get("based_on_human_reviews", []) or []),
        "created_at": data.get("created_at") or now_iso(),
    }
