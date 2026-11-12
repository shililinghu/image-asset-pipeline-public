from __future__ import annotations

import hashlib
import re
import uuid
from typing import Any

import config
from image_generation_runner import run_prompts_for_request
from models import create_prompt_pack_run
from prompt_pack_registry import PromptPackRegistry
from prompt_version_manager import active_prompt_items, ensure_pack_items_versioned
from storage import JsonStore
from subject_profile_builder import build_subject_profile, inject_subject_lock, subject_profile_metadata
from utils import new_id


def run_prompt_pack(
    pack_id: str,
    request: dict[str, Any] | str,
    store: JsonStore | None = None,
    provider_profile_path: str | None = None,
    prompt_selection: dict[str, Any] | None = None,
    images_per_prompt: int | str | None = None,
    subject_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    json_store = store or JsonStore()
    if isinstance(request, str):
        loaded_request = json_store.get("idea_requests", "request_id", request)
        if not loaded_request:
            raise ValueError(f"Idea request not found: {request}")
        request = loaded_request

    registry = PromptPackRegistry(json_store)
    pack = registry.get_pack(pack_id)
    if not pack:
        raise ValueError(f"Prompt pack not found: {pack_id}")

    ensure_pack_items_versioned(pack_id, json_store)
    items = sorted(active_prompt_items(pack_id, registry.get_items(pack_id), json_store), key=_item_sort_key)
    selected_items = select_prompt_pack_items(items, prompt_selection)
    run_id = new_id("pack_run")
    subject_profile = subject_profile or build_subject_profile(request, store=json_store)
    override_count = _normalize_count(images_per_prompt, default=None)
    prompts = _build_variant_prompts(selected_items, request, run_id, override_count, subject_profile)
    tasks, results = run_prompts_for_request(
        request,
        prompts,
        pack_id=pack_id,
        count_per_prompt=1,
        store=json_store,
        provider_profile_path=provider_profile_path,
        subject_profile=subject_profile,
    )

    selection_mode = (prompt_selection or {"mode": "all"}).get("mode", "all")
    summary_images_per_prompt = override_count if override_count is not None else _summary_images_per_prompt(selected_items)
    summary = {
        "pack_id": pack_id,
        "total_item_count": len(items),
        "selected_item_count": len(selected_items),
        "images_per_prompt": summary_images_per_prompt,
        "generated_task_count": len(tasks),
        "variant_task_count": len(tasks),
        "image_result_count": len(results),
        "selection_mode": selection_mode,
        "variation_enabled": bool(config.ENABLE_PROMPT_VARIATION_NONCE),
    }
    run = create_prompt_pack_run(
        {
            "run_id": run_id,
            "request_id": request.get("request_id", ""),
            "pack_id": pack_id,
            "task_ids": [task["task_id"] for task in tasks],
            "image_result_ids": [result["image_result_id"] for result in results],
            "subject_profile_id": subject_profile.get("subject_profile_id", ""),
            **summary,
            "metadata": subject_profile_metadata(subject_profile),
            "status": "completed",
        }
    )
    json_store.add("prompt_pack_runs", run)
    return {
        "pack": pack,
        "items": selected_items,
        "all_items": items,
        "tasks": tasks,
        "image_results": results,
        "run": run,
        "summary": summary,
    }


def build_image_tasks_from_pack(
    pack_id: str,
    request: dict[str, Any],
    store: JsonStore | None = None,
    provider_profile_path: str | None = None,
    images_per_prompt: int | str | None = None,
) -> list[dict[str, Any]]:
    return run_prompt_pack(
        pack_id,
        request,
        store=store,
        provider_profile_path=provider_profile_path,
        images_per_prompt=images_per_prompt,
    )["tasks"]


def select_prompt_pack_items(items: list[dict[str, Any]], prompt_selection: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    sorted_items = sorted(items, key=_item_sort_key)
    selection = prompt_selection or {"mode": "all"}
    mode = str(selection.get("mode", "all"))
    if mode == "all":
        return sorted_items
    if mode == "first_n":
        try:
            n = int(selection.get("n", 3) or 3)
        except (TypeError, ValueError):
            n = 3
        return sorted_items[: max(0, n)]
    if mode == "range":
        start = str(selection.get("start", "")).strip()
        end = str(selection.get("end", "")).strip()
        start_key = natural_prompt_id_key(start)
        end_key = natural_prompt_id_key(end)
        if start_key > end_key:
            start_key, end_key = end_key, start_key
        return [item for item in sorted_items if start_key <= natural_prompt_id_key(str(item.get("prompt_id", ""))) <= end_key]
    if mode == "list":
        wanted = {str(prompt_id).strip() for prompt_id in selection.get("prompt_ids", []) if str(prompt_id).strip()}
        return [item for item in sorted_items if str(item.get("prompt_id", "")).strip() in wanted]
    return sorted_items


def natural_prompt_id_key(prompt_id: str) -> tuple[str, int, str]:
    text = str(prompt_id or "")
    match = re.match(r"^([A-Za-z_\-]*?)(\d+)$", text.strip())
    if match:
        return (match.group(1).lower(), int(match.group(2)), text.lower())
    return (text.lower(), 10**9, text.lower())


def build_variant_prompt(
    base_prompt: str,
    variant_index: int,
    run_id: str,
    prompt_id: str,
    prompt_nonce: str | None = None,
    subject_profile: dict[str, Any] | None = None,
) -> str:
    if not config.ENABLE_PROMPT_VARIATION_NONCE:
        return inject_subject_lock(base_prompt, subject_profile)
    nonce = prompt_nonce or f"{run_id}_{prompt_id}_v{variant_index}_{uuid.uuid4().hex[:8]}"
    locked_prompt = inject_subject_lock(base_prompt, subject_profile)
    instruction = (
        "\n\n【本张变体要求】\n"
        f"在保持同一主体的前提下，生成第 {variant_index} 张变体。\n"
        "- 只改变主题、服装、场景、光线、构图或道具，不改变主体身份\n"
        "- 构图、姿势、微表情、光线细节、背景细节与其他变体不同\n"
        "- 不要复制上一张图的构图\n"
        "- 不要在画面中出现编号、水印或解释文字\n"
        f"变体编号 / 随机锚点：{nonce}"
    )
    return locked_prompt.rstrip() + instruction


def _item_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    prompt_id = str(item.get("prompt_id", ""))
    if prompt_id:
        return (0, *natural_prompt_id_key(prompt_id))
    created_at = str(item.get("created_at", ""))
    if created_at:
        return (1, created_at)
    return (2, "")


def _materialize_item(item: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    prompt_content = str(item.get("prompt_content", ""))
    values = {key: str(value) for key, value in request.items() if isinstance(value, (str, int, float, bool))}
    values.setdefault("topic", str(request.get("topic", request.get("raw_idea", ""))))
    for variable in re.findall(r"{([a-zA-Z0-9_]+)}", prompt_content):
        prompt_content = prompt_content.replace("{" + variable + "}", values.get(variable, ""))
    return {
        "pack_id": item.get("pack_id", ""),
        "prompt_id": item.get("prompt_id") or item.get("item_id", ""),
        "prompt_content": prompt_content,
        "negative_prompt": item.get("negative_prompt", ""),
        "ratio": item.get("ratio", request.get("ratio", "1:1")),
        "count": _normalize_count(item.get("count"), 1),
    }


def _build_variant_prompts(
    items: list[dict[str, Any]],
    request: dict[str, Any],
    run_id: str,
    override_count: int | None,
    subject_profile: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    for item in items:
        base = _materialize_item(item, request)
        repeat_count = override_count if override_count is not None else _normalize_count(item.get("count"), 1)
        repeat_count = max(1, int(repeat_count or 1))
        for variant_index in range(1, repeat_count + 1):
            prompt_id = str(base.get("prompt_id", ""))
            prompt_nonce = f"{run_id}_{prompt_id}_v{variant_index}_{uuid.uuid4().hex[:8]}"
            base_prompt = str(base.get("prompt_content", ""))
            submitted_prompt = build_variant_prompt(base_prompt, variant_index, run_id, prompt_id, prompt_nonce, subject_profile)
            base_hash = _hash_text(base_prompt)
            submitted_hash = _hash_text(submitted_prompt)
            subject_meta = subject_profile_metadata(subject_profile)
            prompts.append(
                {
                    **base,
                    "base_prompt": base_prompt,
                    "prompt_content": base_prompt,
                    "submitted_prompt": submitted_prompt,
                    "variant_index": variant_index,
                    "images_per_prompt": repeat_count,
                    "prompt_nonce": prompt_nonce,
                    "subject_profile_id": subject_meta.get("subject_profile_id", ""),
                    "subject_lock_enabled": subject_meta.get("subject_lock_enabled", False),
                    "subject_description": subject_meta.get("subject_description", ""),
                    "metadata": {
                        "run_id": run_id,
                        "prompt_id": prompt_id,
                        "variant_index": variant_index,
                        "images_per_prompt": repeat_count,
                        "prompt_nonce": prompt_nonce,
                        "base_prompt_hash": base_hash,
                        "submitted_prompt_hash": submitted_hash,
                        **subject_meta,
                    },
                }
            )
    return prompts


def _normalize_count(value: Any, default: int | None = 1) -> int | None:
    if value is None or value == "":
        return default
    try:
        count = int(value)
    except (TypeError, ValueError):
        return default
    return count if count > 0 else default


def _summary_images_per_prompt(items: list[dict[str, Any]]) -> int | str:
    counts = {_normalize_count(item.get("count"), 1) for item in items}
    if len(counts) == 1:
        return counts.pop() or 1
    return "item_count"


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
