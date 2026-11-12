from __future__ import annotations

from pathlib import Path
from typing import Any

import config


def classify_reviewed_image(
    image_result: dict[str, Any],
    image_review: dict[str, Any] | None = None,
    human_review: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reasons: list[str] = []
    image_review = image_review or {}
    aesthetic = image_review.get("aesthetic", {}) if isinstance(image_review, dict) else {}
    pass_score = float(getattr(config, "INCUBATION_AUTO_PASS_SCORE", 70) or 70)

    if str(image_result.get("status", "") or "") == "failed":
        message = str(image_result.get("error_message", "") or "").strip()
        reasons.append(f"图片生成失败：{message}" if message else "图片生成失败")

    file_error = str(image_result.get("file_error", "") or "")
    image_path = str(image_result.get("image_path", "") or "")
    if file_error:
        reasons.append("图片文件不存在" if "not found" in file_error else file_error)
    elif image_path and not image_result.get("image_url") and not Path(image_path).exists():
        reasons.append("图片文件不存在")

    aesthetic_status = str((aesthetic or {}).get("status", "") or "")
    if aesthetic_status == "failed":
        message = str((aesthetic or {}).get("error_message", "") or "").strip()
        reasons.append(f"自动审美审核失败：{message}" if message else "自动审美审核失败")

    score = _safe_float((aesthetic or {}).get("score", 0), 0)
    if aesthetic_status == "success" and score < pass_score:
        reasons.append(f"TasteLens 分数低于 {int(pass_score)}")

    scores = image_review.get("scores", {}) if isinstance(image_review, dict) else {}
    visual_quality = _safe_float(scores.get("visual_quality", image_review.get("visual_quality", 0) if isinstance(image_review, dict) else 0), 0)
    publish_potential = _safe_float(scores.get("publish_potential", image_review.get("publish_potential", 0) if isinstance(image_review, dict) else 0), 0)
    if visual_quality and visual_quality < 5:
        reasons.append("visual_quality 过低")
    if publish_potential and publish_potential < 5:
        reasons.append("publish_potential 过低")

    decision = str(image_review.get("decision", image_review.get("recommendation", "")) if isinstance(image_review, dict) else "").strip()
    if decision == "reject":
        reasons.append("自动审核判定为 reject")

    reasons = _dedupe(reasons)
    return {
        "auto_group": "filtered_out" if reasons else "included",
        "auto_reject_reason": reasons[0] if reasons else "",
        "auto_reject_reasons": reasons,
    }


def _safe_float(value: Any, default: float = 0) -> float:
    try:
        return float(value or default)
    except (TypeError, ValueError):
        return default


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result
