from __future__ import annotations

from typing import Any

from config import IMAGE_MODEL, IMAGE_PROVIDER, IMAGE_PROVIDER_PROFILE_PATH
from image_generation_client import ImageGenerationClient
from models import create_image_generation_task, create_image_result
from storage import JsonStore
from subject_profile_builder import build_subject_profile, inject_subject_lock, subject_profile_metadata


def create_tasks_from_prompts(
    request: dict[str, Any],
    prompts: list[dict[str, Any]],
    *,
    pack_id: str = "",
    count_per_prompt: int = 1,
    store: JsonStore | None = None,
    save: bool = True,
    subject_profile: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    subject_profile = subject_profile or build_subject_profile(request, save=save, store=store)
    subject_meta = subject_profile_metadata(subject_profile)
    for prompt in prompts:
        prompt_id = str(prompt.get("prompt_id", ""))
        base_prompt = str(prompt.get("base_prompt", prompt.get("prompt_content", prompt.get("full_prompt", ""))))
        prompt_content = str(prompt.get("prompt_content", base_prompt))
        submitted_prompt = inject_subject_lock(str(prompt.get("submitted_prompt", prompt_content)), subject_profile)
        prompt_repeat = int(prompt.get("task_repeat_count", count_per_prompt) or 1)
        for repeat_index in range(1, max(1, prompt_repeat) + 1):
            variant_index = int(prompt.get("variant_index", repeat_index) or repeat_index)
            images_per_prompt = int(prompt.get("images_per_prompt", prompt_repeat) or prompt_repeat)
            prompt_nonce = str(prompt.get("prompt_nonce", ""))
            metadata = dict(prompt.get("metadata", {}) or {})
            metadata.update(
                {
                    "run_id": prompt.get("run_id") or metadata.get("run_id", ""),
                    "prompt_id": prompt_id,
                    "variant_index": variant_index,
                    "images_per_prompt": images_per_prompt,
                    "prompt_nonce": prompt_nonce,
                    "base_prompt_hash": metadata.get("base_prompt_hash", ""),
                    "submitted_prompt_hash": metadata.get("submitted_prompt_hash", ""),
                    "prompt_type": request.get("prompt_type", ""),
                    "review_profile": request.get("review_profile", ""),
                    **subject_meta,
                }
            )
            tasks.append(
                create_image_generation_task(
                    {
                        "run_id": metadata.get("run_id", ""),
                        "request_id": request.get("request_id", ""),
                        "pack_id": pack_id or prompt.get("pack_id", ""),
                        "prompt_id": prompt_id,
                        "variant_index": variant_index,
                        "images_per_prompt": images_per_prompt,
                        "prompt_nonce": prompt_nonce,
                        "base_prompt": base_prompt,
                        "submitted_prompt": submitted_prompt,
                        "prompt_content": prompt_content,
                        "subject_profile_id": subject_meta.get("subject_profile_id", ""),
                        "subject_lock_enabled": subject_meta.get("subject_lock_enabled", False),
                        "subject_description": subject_meta.get("subject_description", ""),
                        "negative_prompt": prompt.get("negative_prompt", ""),
                        "ratio": prompt.get("ratio", request.get("ratio", "1:1")),
                        "provider": IMAGE_PROVIDER,
                        "model": IMAGE_MODEL,
                        "metadata": metadata,
                    }
                )
            )
    if save:
        json_store = store or JsonStore()
        for task in tasks:
            json_store.add("image_generation_tasks", task)
    return tasks


def run_image_generation_tasks(
    tasks: list[dict[str, Any]],
    store: JsonStore | None = None,
    client: ImageGenerationClient | None = None,
    provider_profile_path: str | None = None,
    save: bool = True,
) -> list[dict[str, Any]]:
    json_store = store or JsonStore()
    profile_path = provider_profile_path if provider_profile_path is not None else IMAGE_PROVIDER_PROFILE_PATH
    client_error = ""
    if client is None:
        try:
            client = ImageGenerationClient.from_profile(profile_path) if profile_path else ImageGenerationClient()
        except Exception as exc:
            client_error = str(exc)
    results: list[dict[str, Any]] = []
    for task in tasks:
        submitted_prompt = task.get("submitted_prompt") or task.get("prompt_content", "")
        task_metadata = dict(task.get("metadata", {}) or {})
        metadata = {
            "request_id": task.get("request_id", ""),
            "pack_id": task.get("pack_id", ""),
            "prompt_id": task.get("prompt_id", ""),
            "task_id": task.get("task_id", ""),
            "run_id": task.get("run_id", "") or task_metadata.get("run_id", ""),
            "variant_index": task.get("variant_index", task_metadata.get("variant_index", 1)),
            "images_per_prompt": task.get("images_per_prompt", task_metadata.get("images_per_prompt", 1)),
            "prompt_nonce": task.get("prompt_nonce", task_metadata.get("prompt_nonce", "")),
            "base_prompt": task.get("base_prompt", task.get("prompt_content", "")),
            "submitted_prompt": submitted_prompt,
            "base_prompt_hash": task_metadata.get("base_prompt_hash", ""),
            "submitted_prompt_hash": task_metadata.get("submitted_prompt_hash", ""),
            "provider_profile_path": profile_path or "",
            "prompt_type": task_metadata.get("prompt_type", ""),
            "review_profile": task_metadata.get("review_profile", ""),
            "subject_profile_id": task.get("subject_profile_id", task_metadata.get("subject_profile_id", "")),
            "subject_lock_enabled": task.get("subject_lock_enabled", task_metadata.get("subject_lock_enabled", False)),
            "subject_source": task_metadata.get("subject_source", ""),
            "subject_description": task.get("subject_description", task_metadata.get("subject_description", "")),
            "reference_image_ids": task_metadata.get("reference_image_ids", []),
            "reference_image_paths": task_metadata.get("reference_image_paths", []),
            "provider_supports_reference_image": task_metadata.get("provider_supports_reference_image", False),
        }
        if client_error:
            generated = {
                "image_path": "",
                "image_url": "",
                "provider": "profile_load_failed" if profile_path else IMAGE_PROVIDER,
                "model": IMAGE_MODEL,
                "status": "failed",
                "error_message": client_error,
                "metadata": metadata,
            }
        else:
            try:
                generated = client.generate(
                    submitted_prompt,
                    negative_prompt=task.get("negative_prompt", ""),
                    ratio=task.get("ratio", "1:1"),
                    metadata=metadata,
                )
            except Exception as exc:
                generated = {
                    "image_path": "",
                    "image_url": "",
                    "provider": getattr(client, "provider", IMAGE_PROVIDER),
                    "model": getattr(client, "model", IMAGE_MODEL),
                    "status": "failed",
                    "error_message": str(exc),
                    "metadata": metadata,
                }
        generated_metadata = dict(generated.get("metadata", {}) or {})
        generated_metadata.update({key: value for key, value in metadata.items() if key not in generated_metadata or _is_empty_metadata_value(generated_metadata.get(key))})
        generated_metadata.setdefault("provider_profile_path", profile_path or "")
        generated["metadata"] = generated_metadata
        result = create_image_result(
            {
                **generated,
                "task_id": task.get("task_id", ""),
                "request_id": task.get("request_id", ""),
                "pack_id": task.get("pack_id", ""),
                "prompt_id": task.get("prompt_id", ""),
            }
        )
        results.append(result)
        if save:
            json_store.add("image_results", result)
            json_store.update(
                "image_generation_tasks",
                "task_id",
                task["task_id"],
                {"status": result.get("status", "success"), "error_message": result.get("error_message", "")},
            )
    return results


def run_prompts_for_request(
    request: dict[str, Any],
    prompts: list[dict[str, Any]],
    *,
    pack_id: str = "",
    count_per_prompt: int = 1,
    store: JsonStore | None = None,
    provider_profile_path: str | None = None,
    subject_profile: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    json_store = store or JsonStore()
    tasks = create_tasks_from_prompts(request, prompts, pack_id=pack_id, count_per_prompt=count_per_prompt, store=json_store, subject_profile=subject_profile)
    results = run_image_generation_tasks(tasks, store=json_store, provider_profile_path=provider_profile_path)
    return tasks, results


def _is_empty_metadata_value(value: Any) -> bool:
    return value is None or value == "" or value == []
