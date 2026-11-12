from __future__ import annotations

from typing import Any

from config import PACK_MATCH_THRESHOLD, PACK_WEAK_MATCH_THRESHOLD
from models import create_prompt_pack_match_log
from prompt_pack_registry import PromptPackRegistry
from storage import JsonStore


FIELD_WEIGHTS = {
    "domain": 20,
    "platform": 20,
    "use_case": 18,
    "style": 14,
    "ratio": 10,
    "target_audience": 8,
    "text_required": 5,
    "tags": 5,
}


def match_prompt_pack(request: dict[str, Any], store: JsonStore | None = None, save: bool = True) -> dict[str, Any]:
    store = store or JsonStore()
    registry = PromptPackRegistry(store)
    candidates = registry.list_callable_packs()
    best_pack: dict[str, Any] | None = None
    best_score = -1
    best_reasons: list[str] = []
    best_missing: list[str] = []

    for pack in candidates:
        score, reasons, missing = score_pack(request, pack)
        if score > best_score:
            best_pack = pack
            best_score = score
            best_reasons = reasons
            best_missing = missing

    matched = bool(best_pack) and best_score >= PACK_MATCH_THRESHOLD
    weak_match = bool(best_pack) and PACK_WEAK_MATCH_THRESHOLD <= best_score < PACK_MATCH_THRESHOLD
    decision = "use_existing_pack" if matched else ("weak_match" if weak_match else "incubate_new_pack")
    log = create_prompt_pack_match_log(
        {
            "request_id": request.get("request_id", ""),
            "matched": matched,
            "pack_id": best_pack.get("pack_id", "") if best_pack else "",
            "match_score": max(0, best_score),
            "match_reasons": best_reasons,
            "missing_requirements": best_missing,
            "decision": decision,
        }
    )
    if save:
        store.add("prompt_pack_match_logs", log)
    return log


def score_pack(request: dict[str, Any], pack: dict[str, Any]) -> tuple[int, list[str], list[str]]:
    score = 0
    reasons: list[str] = []
    missing: list[str] = []
    for field in ["domain", "platform", "use_case", "style", "ratio"]:
        req_value = _norm(request.get(field, ""))
        pack_value = _norm(pack.get(field, ""))
        if req_value and pack_value == req_value:
            score += FIELD_WEIGHTS[field]
            reasons.append(f"{field} matched")
        elif req_value and pack_value in {"", "general"}:
            score += FIELD_WEIGHTS[field] // 3
            reasons.append(f"{field} is generic")
        elif req_value:
            missing.append(field)

    req_audience = _norm(request.get("target_audience", ""))
    pack_tags = {_norm(tag) for tag in pack.get("match_tags", [])}
    if req_audience and req_audience in pack_tags:
        score += FIELD_WEIGHTS["target_audience"]
        reasons.append("target_audience matched by tags")
    elif req_audience:
        missing.append("target_audience")

    if bool(request.get("text_required", True)):
        if "text_required" in pack_tags or "cover" in pack_tags:
            score += FIELD_WEIGHTS["text_required"]
            reasons.append("text requirement matched")
        else:
            missing.append("text_required")
    else:
        score += FIELD_WEIGHTS["text_required"]

    request_tokens = _token_set(
        [
            request.get("domain", ""),
            request.get("platform", ""),
            request.get("use_case", ""),
            request.get("style", ""),
            request.get("topic", ""),
        ]
    )
    tag_hits = request_tokens.intersection(pack_tags)
    if tag_hits:
        score += min(FIELD_WEIGHTS["tags"], len(tag_hits) * 2)
        reasons.append("tags matched: " + ", ".join(sorted(tag_hits)))

    return min(100, score), reasons, missing


def _norm(value: object) -> str:
    return str(value or "").strip().lower()


def _token_set(values: list[object]) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        for token in str(value or "").replace("_", " ").replace("-", " ").lower().split():
            if len(token) >= 3:
                tokens.add(token)
    return tokens
