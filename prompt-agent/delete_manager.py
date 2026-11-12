from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from config import BACKUP_DIR, DATA_FILES
from storage import JsonStore


DATASET_NAMES = ["projects", "prompts", "test_cases", "test_results", "scores", "feedback", "iterations"]


store = JsonStore()


def _empty_counts() -> dict[str, int]:
    return {name: 0 for name in DATASET_NAMES}


def _result(success: bool, counts: dict[str, int] | None = None, backup_path: str = "", message: str = "") -> dict[str, Any]:
    return {
        "success": success,
        "deleted_counts": counts or _empty_counts(),
        "backup_path": backup_path,
        "message": message,
    }


def _load_all() -> dict[str, list[dict[str, Any]]]:
    return {name: store.load(name) for name in DATASET_NAMES}


def _save_all(data: dict[str, list[dict[str, Any]]]) -> None:
    for name in DATASET_NAMES:
        store.save(name, data.get(name, []))


def _safe_record_id(record: dict[str, Any], id_key: str, index: int) -> str:
    value = str(record.get(id_key, "")).strip()
    return value or f"unknown_{index}"


def backup_data() -> str:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup_dir = BACKUP_DIR / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    for name, path in DATA_FILES.items():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("[]", encoding="utf-8")
        shutil.copy2(path, backup_dir / path.name)
    return str(backup_dir)


def backup_prompt_pack_files() -> list[str]:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup_dir = BACKUP_DIR / f"prompt_pack_delete_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    backup_files: list[str] = []
    for name in ["prompt_packs", "prompt_pack_items"]:
        path = DATA_FILES[name]
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("[]", encoding="utf-8")
        target = backup_dir / path.name
        shutil.copy2(path, target)
        backup_files.append(str(target))
    return backup_files


def delete_prompt_pack(pack_id: str, json_store: JsonStore | None = None) -> dict[str, Any]:
    active_store = json_store or JsonStore()
    packs = active_store.load("prompt_packs")
    items = active_store.load("prompt_pack_items")
    target_packs = [pack for pack in packs if str(pack.get("pack_id", "")) == str(pack_id)]
    if not target_packs:
        return {
            "success": False,
            "message": "未找到该 Prompt 包",
            "pack_id": pack_id,
            "deleted_pack_count": 0,
            "deleted_item_count": 0,
            "backup_files": [],
        }

    backup_files = backup_prompt_pack_files()
    kept_packs, deleted_pack_count = _delete_records(packs, lambda item: str(item.get("pack_id", "")) == str(pack_id))
    kept_items, deleted_item_count = _delete_records(items, lambda item: str(item.get("pack_id", "")) == str(pack_id))
    active_store.save("prompt_packs", kept_packs)
    active_store.save("prompt_pack_items", kept_items)
    message = "Prompt 包及其 prompt items 已删除。"
    if deleted_item_count == 0:
        message += " 该 Prompt 包没有对应 prompt items，可能是导入不完整。"
    return {
        "success": True,
        "message": message,
        "pack_id": pack_id,
        "deleted_pack_count": deleted_pack_count,
        "deleted_item_count": deleted_item_count,
        "backup_files": backup_files,
    }


def get_data_stats() -> dict[str, Any]:
    data = _load_all()
    projects = data["projects"]
    project_stats = []
    for project in projects:
        project_id = str(project.get("project_id", ""))
        counts = get_project_related_counts(project_id, data=data)
        project_stats.append(
            {
                "project_id": project_id or "未知ID",
                "name": str(project.get("name", "未命名项目")) or "未命名项目",
                **counts,
            }
        )

    backup_count = len([path for path in BACKUP_DIR.iterdir() if path.is_dir()]) if BACKUP_DIR.exists() else 0
    return {
        "global": {
            "projects": len(data["projects"]),
            "prompts": len(data["prompts"]),
            "test_cases": len(data["test_cases"]),
            "test_results": len(data["test_results"]),
            "scores": len(data["scores"]),
            "feedback": len(data["feedback"]),
            "iterations": len(data["iterations"]),
            "backups": backup_count,
        },
        "projects": project_stats,
    }


def get_project_related_counts(project_id: str, data: dict[str, list[dict[str, Any]]] | None = None) -> dict[str, int]:
    payload = data or _load_all()
    return {
        "prompts": sum(1 for item in payload["prompts"] if str(item.get("project_id", "")) == str(project_id)),
        "test_cases": sum(1 for item in payload["test_cases"] if str(item.get("project_id", "")) == str(project_id)),
        "test_results": sum(1 for item in payload["test_results"] if str(item.get("project_id", "")) == str(project_id)),
        "scores": sum(1 for item in payload["scores"] if str(item.get("project_id", "")) == str(project_id)),
        "feedback": sum(1 for item in payload["feedback"] if str(item.get("project_id", "")) == str(project_id)),
        "iterations": sum(1 for item in payload["iterations"] if str(item.get("project_id", "")) == str(project_id)),
    }


def _match_id(record: dict[str, Any], id_key: str, target_id: str, index: int) -> bool:
    return _safe_record_id(record, id_key, index) == target_id


def _delete_records(records: list[dict[str, Any]], predicate) -> tuple[list[dict[str, Any]], int]:
    kept = []
    deleted = 0
    for item in records:
        if predicate(item):
            deleted += 1
        else:
            kept.append(item)
    return kept, deleted


def _delete_by_display_id(records: list[dict[str, Any]], id_key: str, target_id: str) -> tuple[list[dict[str, Any]], int]:
    kept = []
    deleted = 0
    for index, item in enumerate(records):
        if _match_id(item, id_key, target_id, index):
            deleted += 1
        else:
            kept.append(item)
    return kept, deleted


def _remove_scores_for_results(scores: list[dict[str, Any]], results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    result_ids = {str(item.get("result_id", "")) for item in results if item.get("result_id")}
    prompt_case_pairs = {(str(item.get("prompt_id", "")), str(item.get("case_id", ""))) for item in results}

    def predicate(item: dict[str, Any]) -> bool:
        result_id = str(item.get("test_result_id", ""))
        if result_id and result_id in result_ids:
            return True
        return (str(item.get("prompt_id", "")), str(item.get("case_id", ""))) in prompt_case_pairs

    return _delete_records(scores, predicate)


def preview_delete_prompt(prompt_id: str) -> dict[str, Any]:
    data = _load_all()
    prompts = [item for idx, item in enumerate(data["prompts"]) if _match_id(item, "prompt_id", prompt_id, idx)]
    if not prompts:
        return _result(False, message="未找到对应提示词。")
    target_ids = {str(item.get("prompt_id", "")) for item in prompts}
    counts = _empty_counts()
    counts["prompts"] = len(prompts)
    counts["test_results"] = sum(1 for item in data["test_results"] if str(item.get("prompt_id", "")) in target_ids)
    counts["scores"] = sum(1 for item in data["scores"] if str(item.get("prompt_id", "")) in target_ids)
    counts["feedback"] = sum(1 for item in data["feedback"] if str(item.get("prompt_id", "")) in target_ids)
    counts["iterations"] = sum(
        1
        for item in data["iterations"]
        if str(item.get("old_prompt_id", "")) in target_ids or str(item.get("new_prompt_id", "")) in target_ids
    )
    return _result(True, counts, message="提示词删除预览")


def preview_delete_test_case(case_id: str) -> dict[str, Any]:
    data = _load_all()
    cases = [item for idx, item in enumerate(data["test_cases"]) if _match_id(item, "case_id", case_id, idx)]
    if not cases:
        return _result(False, message="未找到对应测试案例。")
    target_ids = {str(item.get("case_id", "")) for item in cases}
    counts = _empty_counts()
    counts["test_cases"] = len(cases)
    related_results = [item for item in data["test_results"] if str(item.get("case_id", "")) in target_ids]
    counts["test_results"] = len(related_results)
    counts["scores"] = sum(1 for item in data["scores"] if str(item.get("case_id", "")) in target_ids)
    return _result(True, counts, message="测试案例删除预览")


def preview_delete_test_result(result_id: str) -> dict[str, Any]:
    data = _load_all()
    results = [item for idx, item in enumerate(data["test_results"]) if _match_id(item, "result_id", result_id, idx)]
    if not results:
        return _result(False, message="未找到对应测试结果。")
    counts = _empty_counts()
    counts["test_results"] = len(results)
    counts["scores"] = len(data["scores"]) - len(_remove_scores_for_results(data["scores"], results)[0])
    return _result(True, counts, message="测试结果删除预览")


def preview_delete_simple(dataset: str, id_key: str, target_id: str) -> dict[str, Any]:
    data = _load_all()
    matches = [item for idx, item in enumerate(data[dataset]) if _match_id(item, id_key, target_id, idx)]
    if not matches:
        return _result(False, message="未找到对应记录。")
    counts = _empty_counts()
    counts[dataset] = len(matches)
    return _result(True, counts, message="删除预览")


def preview_project_action(project_id: str, action: str) -> dict[str, Any]:
    data = _load_all()
    project_prompts = [item for item in data["prompts"] if str(item.get("project_id", "")) == str(project_id)]
    project_cases = [item for item in data["test_cases"] if str(item.get("project_id", "")) == str(project_id)]
    project_results = [item for item in data["test_results"] if str(item.get("project_id", "")) == str(project_id)]
    project_scores = [item for item in data["scores"] if str(item.get("project_id", "")) == str(project_id)]
    project_feedback = [item for item in data["feedback"] if str(item.get("project_id", "")) == str(project_id)]
    project_iterations = [item for item in data["iterations"] if str(item.get("project_id", "")) == str(project_id)]
    counts = _empty_counts()

    if action == "clear_prompts":
        counts.update(
            {
                "prompts": len(project_prompts),
                "test_results": len(project_results),
                "scores": len(project_scores),
                "feedback": len(project_feedback),
                "iterations": len(project_iterations),
            }
        )
    elif action == "clear_test_cases":
        related_case_ids = {str(item.get("case_id", "")) for item in project_cases}
        related_results = [item for item in project_results if str(item.get("case_id", "")) in related_case_ids]
        counts.update(
            {
                "test_cases": len(project_cases),
                "test_results": len(related_results),
                "scores": sum(1 for item in project_scores if str(item.get("case_id", "")) in related_case_ids),
            }
        )
    elif action == "clear_results":
        counts.update({"test_results": len(project_results), "scores": len(project_scores)})
    elif action == "clear_scores":
        counts.update({"scores": len(project_scores)})
    elif action == "clear_feedback_iterations":
        counts.update({"feedback": len(project_feedback), "iterations": len(project_iterations)})
    elif action == "clear_generated_data":
        counts.update(
            {
                "prompts": len(project_prompts),
                "test_cases": len(project_cases),
                "test_results": len(project_results),
                "scores": len(project_scores),
                "feedback": len(project_feedback),
                "iterations": len(project_iterations),
            }
        )
    elif action == "delete_project":
        counts.update(
            {
                "projects": sum(1 for item in data["projects"] if str(item.get("project_id", "")) == str(project_id)),
                "prompts": len(project_prompts),
                "test_cases": len(project_cases),
                "test_results": len(project_results),
                "scores": len(project_scores),
                "feedback": len(project_feedback),
                "iterations": len(project_iterations),
            }
        )
    return _result(True, counts, message="项目清理预览")


def _with_backup(delete_fn):
    try:
        backup_path = backup_data()
    except Exception as exc:
        return _result(False, message=f"备份失败，已阻止删除。原因：{exc}")
    result = delete_fn()
    result["backup_path"] = backup_path
    return result


def delete_prompt(prompt_id: str) -> dict[str, Any]:
    def action():
        data = _load_all()
        prompts = [item for idx, item in enumerate(data["prompts"]) if _match_id(item, "prompt_id", prompt_id, idx)]
        if not prompts:
            return _result(False, message="未找到对应提示词。")
        prompt_ids = {str(item.get("prompt_id", "")) for item in prompts}
        counts = _empty_counts()
        data["prompts"], counts["prompts"] = _delete_records(data["prompts"], lambda item: str(item.get("prompt_id", "")) in prompt_ids)
        data["test_results"], counts["test_results"] = _delete_records(data["test_results"], lambda item: str(item.get("prompt_id", "")) in prompt_ids)
        data["scores"], counts["scores"] = _delete_records(data["scores"], lambda item: str(item.get("prompt_id", "")) in prompt_ids)
        data["feedback"], counts["feedback"] = _delete_records(data["feedback"], lambda item: str(item.get("prompt_id", "")) in prompt_ids)
        data["iterations"], counts["iterations"] = _delete_records(
            data["iterations"],
            lambda item: str(item.get("old_prompt_id", "")) in prompt_ids or str(item.get("new_prompt_id", "")) in prompt_ids,
        )
        _save_all(data)
        return _result(True, counts, message="提示词及其关联数据已删除。")

    return _with_backup(action)


def delete_test_case(case_id: str) -> dict[str, Any]:
    def action():
        data = _load_all()
        cases = [item for idx, item in enumerate(data["test_cases"]) if _match_id(item, "case_id", case_id, idx)]
        if not cases:
            return _result(False, message="未找到对应测试案例。")
        case_ids = {str(item.get("case_id", "")) for item in cases}
        counts = _empty_counts()
        data["test_cases"], counts["test_cases"] = _delete_records(data["test_cases"], lambda item: str(item.get("case_id", "")) in case_ids)
        related_results = [item for item in data["test_results"] if str(item.get("case_id", "")) in case_ids]
        data["test_results"], counts["test_results"] = _delete_records(data["test_results"], lambda item: str(item.get("case_id", "")) in case_ids)
        data["scores"], counts["scores"] = _remove_scores_for_results(data["scores"], related_results)
        _save_all(data)
        return _result(True, counts, message="测试案例及其关联结果已删除。")

    return _with_backup(action)


def delete_test_result(result_id: str) -> dict[str, Any]:
    def action():
        data = _load_all()
        results = [item for idx, item in enumerate(data["test_results"]) if _match_id(item, "result_id", result_id, idx)]
        if not results:
            return _result(False, message="未找到对应测试结果。")
        counts = _empty_counts()
        result_ids = {str(item.get("result_id", "")) for item in results}
        data["test_results"], counts["test_results"] = _delete_records(data["test_results"], lambda item: str(item.get("result_id", "")) in result_ids)
        data["scores"], counts["scores"] = _remove_scores_for_results(data["scores"], results)
        _save_all(data)
        return _result(True, counts, message="测试结果及其关联评分已删除。")

    return _with_backup(action)


def delete_score(score_id: str) -> dict[str, Any]:
    def action():
        data = _load_all()
        counts = _empty_counts()
        data["scores"], counts["scores"] = _delete_by_display_id(data["scores"], "score_id", score_id)
        if counts["scores"] <= 0:
            return _result(False, message="未找到对应评分结果。")
        _save_all(data)
        return _result(True, counts, message="评分结果已删除。")

    return _with_backup(action)


def delete_feedback(feedback_id: str) -> dict[str, Any]:
    def action():
        data = _load_all()
        counts = _empty_counts()
        data["feedback"], counts["feedback"] = _delete_by_display_id(data["feedback"], "feedback_id", feedback_id)
        if counts["feedback"] <= 0:
            return _result(False, message="未找到对应反馈记录。")
        _save_all(data)
        return _result(True, counts, message="反馈记录已删除。")

    return _with_backup(action)


def delete_iteration(iteration_id: str) -> dict[str, Any]:
    def action():
        data = _load_all()
        counts = _empty_counts()
        data["iterations"], counts["iterations"] = _delete_by_display_id(data["iterations"], "iteration_id", iteration_id)
        if counts["iterations"] <= 0:
            return _result(False, message="未找到对应迭代记录。")
        _save_all(data)
        return _result(True, counts, message="迭代记录已删除。")

    return _with_backup(action)


def clear_project_prompts(project_id: str) -> dict[str, Any]:
    def action():
        data = _load_all()
        prompt_ids = {str(item.get("prompt_id", "")) for item in data["prompts"] if str(item.get("project_id", "")) == str(project_id)}
        counts = _empty_counts()
        data["prompts"], counts["prompts"] = _delete_records(data["prompts"], lambda item: str(item.get("project_id", "")) == str(project_id))
        data["test_results"], counts["test_results"] = _delete_records(data["test_results"], lambda item: str(item.get("prompt_id", "")) in prompt_ids)
        data["scores"], counts["scores"] = _delete_records(data["scores"], lambda item: str(item.get("prompt_id", "")) in prompt_ids)
        data["feedback"], counts["feedback"] = _delete_records(data["feedback"], lambda item: str(item.get("prompt_id", "")) in prompt_ids)
        data["iterations"], counts["iterations"] = _delete_records(
            data["iterations"],
            lambda item: str(item.get("old_prompt_id", "")) in prompt_ids or str(item.get("new_prompt_id", "")) in prompt_ids or str(item.get("project_id", "")) == str(project_id),
        )
        _save_all(data)
        return _result(True, counts, message="项目下全部提示词及关联数据已清理。")

    return _with_backup(action)


def clear_project_test_cases(project_id: str) -> dict[str, Any]:
    def action():
        data = _load_all()
        case_ids = {str(item.get("case_id", "")) for item in data["test_cases"] if str(item.get("project_id", "")) == str(project_id)}
        counts = _empty_counts()
        data["test_cases"], counts["test_cases"] = _delete_records(data["test_cases"], lambda item: str(item.get("project_id", "")) == str(project_id))
        related_results = [item for item in data["test_results"] if str(item.get("case_id", "")) in case_ids]
        data["test_results"], counts["test_results"] = _delete_records(data["test_results"], lambda item: str(item.get("case_id", "")) in case_ids)
        data["scores"], counts["scores"] = _remove_scores_for_results(data["scores"], related_results)
        _save_all(data)
        return _result(True, counts, message="项目下全部测试案例及关联结果已清理。")

    return _with_backup(action)


def clear_project_results(project_id: str) -> dict[str, Any]:
    def action():
        data = _load_all()
        results = [item for item in data["test_results"] if str(item.get("project_id", "")) == str(project_id)]
        counts = _empty_counts()
        data["test_results"], counts["test_results"] = _delete_records(data["test_results"], lambda item: str(item.get("project_id", "")) == str(project_id))
        data["scores"], counts["scores"] = _remove_scores_for_results(data["scores"], results)
        _save_all(data)
        return _result(True, counts, message="项目下全部测试结果及关联评分已清理。")

    return _with_backup(action)


def clear_project_scores(project_id: str) -> dict[str, Any]:
    def action():
        data = _load_all()
        counts = _empty_counts()
        data["scores"], counts["scores"] = _delete_records(data["scores"], lambda item: str(item.get("project_id", "")) == str(project_id))
        _save_all(data)
        return _result(True, counts, message="项目下全部评分结果已清理。")

    return _with_backup(action)


def clear_project_feedback_iterations(project_id: str) -> dict[str, Any]:
    def action():
        data = _load_all()
        counts = _empty_counts()
        data["feedback"], counts["feedback"] = _delete_records(data["feedback"], lambda item: str(item.get("project_id", "")) == str(project_id))
        data["iterations"], counts["iterations"] = _delete_records(data["iterations"], lambda item: str(item.get("project_id", "")) == str(project_id))
        _save_all(data)
        return _result(True, counts, message="项目下全部反馈和迭代记录已清理。")

    return _with_backup(action)


def clear_project_generated_data(project_id: str) -> dict[str, Any]:
    def action():
        data = _load_all()
        counts = _empty_counts()
        data["prompts"], counts["prompts"] = _delete_records(data["prompts"], lambda item: str(item.get("project_id", "")) == str(project_id))
        data["test_cases"], counts["test_cases"] = _delete_records(data["test_cases"], lambda item: str(item.get("project_id", "")) == str(project_id))
        data["test_results"], counts["test_results"] = _delete_records(data["test_results"], lambda item: str(item.get("project_id", "")) == str(project_id))
        data["scores"], counts["scores"] = _delete_records(data["scores"], lambda item: str(item.get("project_id", "")) == str(project_id))
        data["feedback"], counts["feedback"] = _delete_records(data["feedback"], lambda item: str(item.get("project_id", "")) == str(project_id))
        data["iterations"], counts["iterations"] = _delete_records(data["iterations"], lambda item: str(item.get("project_id", "")) == str(project_id))
        _save_all(data)
        return _result(True, counts, message="项目下所有生成数据已清理，项目本身已保留。")

    return _with_backup(action)


def delete_project_cascade(project_id: str) -> dict[str, Any]:
    def action():
        data = _load_all()
        counts = _empty_counts()
        data["projects"], counts["projects"] = _delete_records(data["projects"], lambda item: str(item.get("project_id", "")) == str(project_id))
        data["prompts"], counts["prompts"] = _delete_records(data["prompts"], lambda item: str(item.get("project_id", "")) == str(project_id))
        data["test_cases"], counts["test_cases"] = _delete_records(data["test_cases"], lambda item: str(item.get("project_id", "")) == str(project_id))
        data["test_results"], counts["test_results"] = _delete_records(data["test_results"], lambda item: str(item.get("project_id", "")) == str(project_id))
        data["scores"], counts["scores"] = _delete_records(data["scores"], lambda item: str(item.get("project_id", "")) == str(project_id))
        data["feedback"], counts["feedback"] = _delete_records(data["feedback"], lambda item: str(item.get("project_id", "")) == str(project_id))
        data["iterations"], counts["iterations"] = _delete_records(data["iterations"], lambda item: str(item.get("project_id", "")) == str(project_id))
        _save_all(data)
        return _result(True, counts, message="项目及其全部关联数据已删除。")

    return _with_backup(action)
