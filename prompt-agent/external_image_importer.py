from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from config import INPUT_IMAGES_DIR
from models import create_image_result
from storage import JsonStore
from utils import new_id, now_iso


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
CSV_FIELDS = ["filename", "image_path", "request_id", "prompt_id", "pack_id", "notes"]


def import_external_images(
    *,
    request_id: str = "",
    prompt_id: str = "",
    pack_id: str = "",
    image_dir: str | Path | None = None,
    mapping_csv: str | Path | None = None,
    store: JsonStore | None = None,
) -> list[dict[str, Any]]:
    json_store = store or JsonStore()
    source_dir = Path(image_dir) if image_dir else INPUT_IMAGES_DIR
    source_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_mapping_rows(source_dir, mapping_csv) if mapping_csv else _rows_from_directory(source_dir)
    imported: list[dict[str, Any]] = []
    existing_keys = _existing_import_keys(json_store)

    for row in rows:
        resolved_request_id = str(row.get("request_id") or request_id).strip()
        resolved_prompt_id = str(row.get("prompt_id") or prompt_id).strip()
        resolved_pack_id = str(row.get("pack_id") or pack_id).strip()
        image_path = _resolve_image_path(source_dir, row)
        if not image_path or not image_path.exists() or not image_path.is_file():
            continue
        if image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            continue
        if not resolved_request_id or not resolved_prompt_id:
            raise ValueError("External image import requires request_id and prompt_id for every image.")

        import_key = (str(image_path.resolve()), resolved_request_id, resolved_prompt_id, resolved_pack_id)
        if import_key in existing_keys:
            continue

        result = create_image_result(
            {
                "task_id": new_id("external_import"),
                "request_id": resolved_request_id,
                "pack_id": resolved_pack_id,
                "prompt_id": resolved_prompt_id,
                "image_path": str(image_path.resolve()),
                "image_url": "",
                "provider": "external_import",
                "model": "manual_external_image",
                "status": "success",
                "error_message": "",
                "metadata": {
                    "source": "external_import",
                    "original_filename": image_path.name,
                    "imported_at": now_iso(),
                    "notes": str(row.get("notes", "") or ""),
                },
            }
        )
        result["source"] = "external_import"
        json_store.add("image_results", result)
        imported.append(result)
        existing_keys.add(import_key)
    return imported


def write_mapping_csv_template(output_path: str | Path | None = None, image_dir: str | Path | None = None) -> Path:
    source_dir = Path(image_dir) if image_dir else INPUT_IMAGES_DIR
    source_dir.mkdir(parents=True, exist_ok=True)
    path = Path(output_path) if output_path else source_dir / "external_image_import_template.csv"
    images = _rows_from_directory(source_dir)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in images:
            writer.writerow({**row, "request_id": "", "prompt_id": "", "pack_id": "", "notes": ""})
    return path


def _rows_from_directory(image_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(image_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            rows.append({"filename": path.name, "image_path": str(path)})
    return rows


def _load_mapping_rows(image_dir: Path, mapping_csv: str | Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with Path(mapping_csv).open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append({key: str(value or "").strip() for key, value in row.items()})
    return rows


def _resolve_image_path(image_dir: Path, row: dict[str, Any]) -> Path | None:
    raw_image_path = str(row.get("image_path", "") or "").strip()
    raw_filename = str(row.get("filename", "") or "").strip()
    if raw_image_path:
        path = Path(raw_image_path)
        return path if path.is_absolute() else image_dir / path
    if raw_filename:
        return image_dir / raw_filename
    return None


def _existing_import_keys(store: JsonStore) -> set[tuple[str, str, str, str]]:
    keys: set[tuple[str, str, str, str]] = set()
    for item in store.load("image_results"):
        if item.get("provider") != "external_import" and item.get("source") != "external_import":
            continue
        image_path = str(item.get("image_path", "") or "")
        if not image_path:
            continue
        keys.add(
            (
                str(Path(image_path).resolve()),
                str(item.get("request_id", "") or ""),
                str(item.get("prompt_id", "") or ""),
                str(item.get("pack_id", "") or ""),
            )
        )
    return keys
