from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import INPUT_IMAGES_DIR
from external_image_importer import import_external_images, write_mapping_csv_template
from human_review_exporter import export_human_review_csv
from image_auto_reviewer import review_image_results


def main() -> int:
    parser = argparse.ArgumentParser(description="Manually import local images into image_results.json.")
    parser.add_argument("--image-dir", default=str(INPUT_IMAGES_DIR), help="Folder containing manually generated images.")
    parser.add_argument("--mapping-csv", default="", help="CSV mapping each image to request_id/prompt_id/pack_id.")
    parser.add_argument("--request-id", default="", help="Request id used for every imported image when no CSV is provided.")
    parser.add_argument("--prompt-id", default="", help="Prompt id used for every imported image when no CSV is provided.")
    parser.add_argument("--pack-id", default="", help="Optional pack id used for every imported image when no CSV is provided.")
    parser.add_argument("--write-template", action="store_true", help="Write a CSV template for images in the folder.")
    parser.add_argument("--auto-review", action="store_true", help="Run mock auto review after import.")
    parser.add_argument("--export-human-csv", action="store_true", help="Export human review CSV after import/review.")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    if args.write_template:
        path = write_mapping_csv_template(image_dir=image_dir)
        print(f"Mapping CSV template written: {path}")
        return 0

    try:
        imported = import_external_images(
            request_id=args.request_id,
            prompt_id=args.prompt_id,
            pack_id=args.pack_id,
            image_dir=image_dir,
            mapping_csv=args.mapping_csv or None,
        )
    except ValueError as exc:
        print(f"Import failed: {exc}")
        return 2

    summary = {
        "image_dir": str(image_dir.resolve()),
        "mapping_csv": args.mapping_csv,
        "imported_count": len(imported),
        "image_result_ids": [item.get("image_result_id", "") for item in imported],
    }

    if args.auto_review:
        reviews = review_image_results(imported)
        summary["auto_review_count"] = len(reviews)

    if args.export_human_csv:
        request_id = args.request_id or (imported[0].get("request_id", "") if imported else "")
        path = export_human_review_csv(request_id=request_id)
        summary["human_review_csv"] = str(path)

    print("EXTERNAL_IMAGE_IMPORT_SUMMARY")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
