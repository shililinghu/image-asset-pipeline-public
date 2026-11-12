from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from models import create_human_image_review
from storage import JsonStore
from utils import now_iso, safe_filename


HUMAN_DECISIONS = {"keep", "reject", "revise", "high_potential"}
CSV_FIELDS = [
    "image_result_id",
    "request_id",
    "pack_id",
    "prompt_id",
    "image_path",
    "auto_decision",
    "auto_score",
    "human_decision",
    "human_score",
    "reject_reasons",
    "notes",
]


def export_human_review_csv(
    request_id: str = "",
    output_path: str | Path | None = None,
    store: JsonStore | None = None,
) -> Path:
    json_store = store or JsonStore()
    results = json_store.load("image_results")
    reviews = {item.get("image_result_id"): item for item in json_store.load("image_reviews")}
    if request_id:
        results = [result for result in results if result.get("request_id") == request_id]
    export_dir = Path(__file__).resolve().parent / "exports"
    export_dir.mkdir(exist_ok=True)
    path = Path(output_path) if output_path else export_dir / safe_filename(f"human_image_review_{request_id or now_iso()}.csv")
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for result in results:
            review = reviews.get(result.get("image_result_id"), {})
            writer.writerow(
                {
                    "image_result_id": result.get("image_result_id", ""),
                    "request_id": result.get("request_id", ""),
                    "pack_id": result.get("pack_id", ""),
                    "prompt_id": result.get("prompt_id", ""),
                    "image_path": result.get("image_path", ""),
                    "auto_decision": review.get("auto_decision", ""),
                    "auto_score": review.get("auto_score", ""),
                    "human_decision": "",
                    "human_score": "",
                    "reject_reasons": "",
                    "notes": "",
                }
            )
    return path


def import_human_review_csv(path: str | Path, store: JsonStore | None = None) -> list[dict[str, Any]]:
    json_store = store or JsonStore()
    imported: list[dict[str, Any]] = []
    existing_ids = {item.get("image_result_id") for item in json_store.load("human_image_reviews")}
    with Path(path).open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        for row in reader:
            image_result_id = str(row.get("image_result_id", "")).strip()
            if not image_result_id or image_result_id in existing_ids:
                continue
            decision = str(row.get("human_decision", "")).strip() or "revise"
            if decision not in HUMAN_DECISIONS:
                decision = "revise"
            review = create_human_image_review(
                {
                    "image_result_id": image_result_id,
                    "request_id": row.get("request_id", ""),
                    "pack_id": row.get("pack_id", ""),
                    "prompt_id": row.get("prompt_id", ""),
                    "human_decision": decision,
                    "human_score": _safe_float(row.get("human_score", 0)),
                    "reject_reasons": row.get("reject_reasons", ""),
                    "notes": row.get("notes", ""),
                }
            )
            imported.append(review)
            json_store.add("human_image_reviews", review)
            existing_ids.add(image_result_id)
    return imported


def _safe_float(value: object) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0
