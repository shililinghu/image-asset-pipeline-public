from __future__ import annotations

import re
from typing import Any

from models import create_idea_request
from storage import JsonStore


def analyze_idea(raw_idea: str, store: JsonStore | None = None, save: bool = True, **overrides: Any) -> dict[str, Any]:
    text = " ".join(str(raw_idea or "").split())
    lowered = text.lower()
    request = create_idea_request(
        {
            "raw_idea": text,
            "domain": overrides.get("domain") or _infer_domain(lowered),
            "platform": overrides.get("platform") or _infer_platform(lowered),
            "use_case": overrides.get("use_case") or _infer_use_case(lowered),
            "style": overrides.get("style") or _infer_style(lowered),
            "ratio": overrides.get("ratio") or _infer_ratio(lowered),
            "topic": overrides.get("topic") or _infer_topic(text),
            "constraints": overrides.get("constraints") or _infer_constraints(lowered),
            "target_audience": overrides.get("target_audience") or _infer_target_audience(text),
            "text_required": overrides.get("text_required", _infer_text_required(lowered)),
            "batch_size": overrides.get("batch_size") or _infer_batch_size(lowered),
        }
    )
    if save:
        (store or JsonStore()).add("idea_requests", request)
    return request


def _infer_domain(text: str) -> str:
    if any(token in text for token in ["chemistry", "chemical", "molecule", "lab", "化学", "分子", "实验"]):
        return "chemistry"
    if any(token in text for token in ["comic", "漫画", "分镜"]):
        return "comic"
    if any(token in text for token in ["education", "course", "科普", "教育", "知识"]):
        return "education"
    return "general"


def _infer_platform(text: str) -> str:
    if any(token in text for token in ["xiaohongshu", "小红书", "rednote"]):
        return "xiaohongshu"
    if any(token in text for token in ["wechat", "公众号", "微信"]):
        return "wechat"
    if any(token in text for token in ["douyin", "抖音", "tiktok"]):
        return "douyin"
    return "general"


def _infer_use_case(text: str) -> str:
    if any(token in text for token in ["cover", "封面", "首图"]):
        return "cover"
    if any(token in text for token in ["poster", "海报"]):
        return "poster"
    if any(token in text for token in ["comic", "漫画", "分镜"]):
        return "comic"
    return "image"


def _infer_style(text: str) -> str:
    if any(token in text for token in ["clean", "minimal", "简洁", "清爽", "干净"]):
        if any(token in text for token in ["education", "科普", "知识", "化学"]):
            return "clean_education"
        return "clean"
    if any(token in text for token in ["cute", "可爱"]):
        return "cute"
    return "general"


def _infer_ratio(text: str) -> str:
    match = re.search(r"(\d+)\s*[:：]\s*(\d+)", text)
    if match:
        return f"{match.group(1)}:{match.group(2)}"
    if any(token in text for token in ["小红书", "xiaohongshu", "竖版", "vertical"]):
        return "3:4"
    return "1:1"


def _infer_topic(text: str) -> str:
    for marker in ["关于", "about", "主题", "topic"]:
        if marker in text:
            return text.split(marker, 1)[1].strip(" :：，,。.")[:80] or text[:80]
    return text[:80]


def _infer_constraints(text: str) -> list[str]:
    constraints: list[str] = []
    if any(token in text for token in ["不要", "avoid", "no "]):
        constraints.append("has_negative_constraints")
    if any(token in text for token in ["文字", "text", "标题", "headline"]):
        constraints.append("requires_readable_text")
    return constraints


def _infer_target_audience(text: str) -> str:
    if any(token in text for token in ["学生", "student"]):
        return "students"
    if any(token in text for token in ["读者", "reader"]):
        return "readers"
    if any(token in text for token in ["用户", "user"]):
        return "users"
    return ""


def _infer_text_required(text: str) -> bool:
    return any(token in text for token in ["文字", "标题", "headline", "text", "封面", "cover"])


def _infer_batch_size(text: str) -> int:
    match = re.search(r"(\d+)\s*(张|个|images|pics)", text)
    if match:
        return max(1, min(50, int(match.group(1))))
    return 10
