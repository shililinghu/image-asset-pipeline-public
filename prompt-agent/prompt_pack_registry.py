from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from models import create_prompt_pack, create_prompt_pack_item
from prompt_version_manager import ensure_pack_items_versioned
from storage import JsonStore
from utils import now_iso


CALLABLE_PACK_STATUSES = {"approved", "candidate"}
VALID_PACK_STATUSES = {"draft", "testing", "candidate", "approved", "archived"}


class PromptPackRegistry:
    def __init__(self, store: JsonStore | None = None) -> None:
        self.store = store or JsonStore()

    def list_packs(self, include_archived: bool = False) -> list[dict[str, Any]]:
        packs = self.store.load("prompt_packs")
        if include_archived:
            return packs
        return [pack for pack in packs if pack.get("status") != "archived"]

    def list_callable_packs(self) -> list[dict[str, Any]]:
        return [pack for pack in self.list_packs() if pack.get("status") in CALLABLE_PACK_STATUSES]

    def get_pack(self, pack_id: str) -> dict[str, Any] | None:
        return self.store.get("prompt_packs", "pack_id", pack_id)

    def get_items(self, pack_id: str) -> list[dict[str, Any]]:
        return self.store.filter("prompt_pack_items", pack_id=pack_id)

    def create_pack(self, pack_data: dict[str, Any], items: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        pack = create_prompt_pack(pack_data)
        self.store.add("prompt_packs", pack)
        for item_data in items or []:
            self.add_item(pack["pack_id"], item_data)
        ensure_pack_items_versioned(pack["pack_id"], self.store)
        return pack

    def add_item(self, pack_id: str, item_data: dict[str, Any]) -> dict[str, Any]:
        item_payload = dict(item_data)
        item_payload["pack_id"] = pack_id
        item = create_prompt_pack_item(item_payload)
        self.store.add("prompt_pack_items", item)
        ensure_pack_items_versioned(pack_id, self.store)
        return item

    def update_pack_status(self, pack_id: str, status: str) -> bool:
        if status not in VALID_PACK_STATUSES:
            raise ValueError(f"invalid pack status: {status}")
        return self.store.update("prompt_packs", "pack_id", pack_id, {"status": status, "updated_at": now_iso()})

    def archive_pack(self, pack_id: str) -> bool:
        return self.update_pack_status(pack_id, "archived")

    def register_pack(self, pack_data: dict[str, Any], items: list[dict[str, Any]]) -> dict[str, Any]:
        existing_id = str(pack_data.get("pack_id", "")).strip()
        if existing_id and self.get_pack(existing_id):
            self.store.update("prompt_packs", "pack_id", existing_id, {**pack_data, "updated_at": now_iso()})
            return self.get_pack(existing_id) or dict(pack_data)
        return self.create_pack(pack_data, items)

    def import_pack_file(self, path: str | Path) -> dict[str, Any]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("prompt pack file must contain a JSON object")
        pack_data = payload.get("pack", payload)
        items = payload.get("items", [])
        prompts = payload.get("prompts", [])
        if not isinstance(pack_data, dict):
            raise ValueError("prompt pack file must contain pack object")
        if not isinstance(items, list):
            raise ValueError("prompt pack items must be a list")
        if not isinstance(prompts, list):
            raise ValueError("prompt pack prompts must be a list")

        pack = self._upsert_pack_shell(pack_data)
        pack_id = pack["pack_id"]
        normalized_items, skipped_invalid_count, warnings = self._normalize_import_items(pack, items, prompts)
        summary = self._upsert_pack_items(pack_id, normalized_items)
        summary.update(
            {
                "pack_id": pack_id,
                "pack_imported_updated": True,
                "prompts_found": len(prompts),
                "items_found": len(items),
                "compatible_items_found": len(items) if not prompts else 0,
                "skipped_invalid_count": skipped_invalid_count,
                "warnings": warnings,
            }
        )
        pack["import_summary"] = summary
        self._print_import_summary(summary)
        return pack

    def _upsert_pack_shell(self, pack_data: dict[str, Any]) -> dict[str, Any]:
        pack = create_prompt_pack(pack_data)
        existing = self.get_pack(pack["pack_id"])
        if existing:
            self.store.update("prompt_packs", "pack_id", pack["pack_id"], {**pack, "updated_at": now_iso()})
            return self.get_pack(pack["pack_id"]) or pack
        self.store.add("prompt_packs", pack)
        return pack

    def _normalize_import_items(
        self,
        pack: dict[str, Any],
        items: list[dict[str, Any]],
        prompts: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], int, list[str]]:
        source_rows = prompts if prompts else items
        skipped_invalid_count = 0
        warnings: list[str] = []
        normalized: list[dict[str, Any]] = []
        for index, row in enumerate(source_rows, start=1):
            if not isinstance(row, dict):
                skipped_invalid_count += 1
                warnings.append(f"invalid prompt at index {index}: not an object")
                continue
            prompt_content = _first_text(row, ["prompt_content", "prompt", "content"])
            if not prompt_content:
                skipped_invalid_count += 1
                prompt_id = row.get("prompt_id", f"prompt_{index}")
                title = row.get("title", "")
                warning = f"跳过无正文 prompt：prompt_id={prompt_id}, title={title}"
                warnings.append(warning)
                print(f"warning: {warning}")
                continue
            normalized.append(
                {
                    "pack_id": pack["pack_id"],
                    "prompt_id": row.get("prompt_id") or f"{pack['pack_id']}_prompt_{index}",
                    "prompt_content": prompt_content,
                    "negative_prompt": row.get("negative_prompt", ""),
                    "variables": row.get("variables") if isinstance(row.get("variables"), list) else ["topic"],
                    "use_case": row.get("use_case", pack.get("use_case", "image")),
                    "style_name": row.get("style_name", row.get("style", pack.get("style", ""))),
                    "ratio": row.get("ratio") or pack.get("ratio") or "3:4",
                    "count": _safe_count(row.get("count", 1)),
                    "tags": list(row.get("tags", []) or []) if isinstance(row.get("tags", []), list) else [],
                    "metadata": dict(row.get("metadata", {}) or {}) if isinstance(row.get("metadata", {}), dict) else {},
                    "item_score": row.get("item_score", 0),
                    "approved_image_count": row.get("approved_image_count", 0),
                    "tested_image_count": row.get("tested_image_count", 0),
                }
            )
        return normalized, skipped_invalid_count, warnings

    def _upsert_pack_items(self, pack_id: str, normalized_items: list[dict[str, Any]]) -> dict[str, int]:
        data = self.store.load("prompt_pack_items")
        existing_index = {
            (item.get("pack_id", ""), item.get("prompt_id", "")): index
            for index, item in enumerate(data)
        }
        imported_count = 0
        updated_count = 0
        skipped_existing_count = 0
        for item_data in normalized_items:
            item = create_prompt_pack_item(item_data)
            item["tags"] = list(item_data.get("tags", []) or [])
            item["metadata"] = dict(item_data.get("metadata", {}) or {})
            key = (pack_id, item["prompt_id"])
            if key in existing_index:
                current = data[existing_index[key]]
                updates = {
                    "prompt_content": item["prompt_content"],
                    "negative_prompt": item["negative_prompt"],
                    "variables": item["variables"],
                    "use_case": item["use_case"],
                    "style_name": item["style_name"],
                    "ratio": item["ratio"],
                    "count": item["count"],
                    "tags": item["tags"],
                    "metadata": item["metadata"],
                }
                changed = any(current.get(field) != value for field, value in updates.items())
                if changed:
                    current.update(updates)
                    updated_count += 1
                else:
                    skipped_existing_count += 1
                continue
            data.append(item)
            existing_index[key] = len(data) - 1
            imported_count += 1
        self.store.save("prompt_pack_items", data)
        ensure_pack_items_versioned(pack_id, self.store)
        return {
            "imported_count": imported_count,
            "updated_count": updated_count,
            "skipped_existing_count": skipped_existing_count,
        }

    def _print_import_summary(self, summary: dict[str, Any]) -> None:
        print(f"pack_id: {summary.get('pack_id', '')}")
        print(f"pack imported/updated: {str(summary.get('pack_imported_updated', False)).lower()}")
        print(f"prompts found: {summary.get('prompts_found', 0)}")
        print(f"compatible items found: {summary.get('compatible_items_found', 0)}")
        print(f"items imported: {summary.get('imported_count', 0)}")
        print(f"items updated: {summary.get('updated_count', 0)}")
        print(f"items skipped existing: {summary.get('skipped_existing_count', 0)}")
        print(f"items skipped invalid: {summary.get('skipped_invalid_count', 0)}")
        prompts_found = int(summary.get("prompts_found", 0) or 0)
        compatible_found = int(summary.get("compatible_items_found", 0) or 0)
        imported = int(summary.get("imported_count", 0) or 0)
        updated = int(summary.get("updated_count", 0) or 0)
        skipped_existing = int(summary.get("skipped_existing_count", 0) or 0)
        if prompts_found == 0 and compatible_found > 0 and (imported > 0 or updated > 0 or skipped_existing > 0):
            print("该 Prompt 包未使用 prompts 数组，已通过兼容格式导入 prompt items。")
        elif prompts_found == 0 and imported == 0 and updated == 0 and skipped_existing == 0:
            print("该 Prompt 包没有可导入的 prompt items。")


def list_prompt_packs(store: JsonStore | None = None, include_archived: bool = False) -> list[dict[str, Any]]:
    return PromptPackRegistry(store).list_packs(include_archived=include_archived)


def get_prompt_pack_with_items(pack_id: str, store: JsonStore | None = None) -> dict[str, Any] | None:
    registry = PromptPackRegistry(store)
    pack = registry.get_pack(pack_id)
    if not pack:
        return None
    return {"pack": pack, "items": registry.get_items(pack_id)}


def _first_text(row: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _safe_count(value: object) -> int:
    try:
        count = int(value or 1)
    except (TypeError, ValueError):
        return 1
    return max(1, count)
