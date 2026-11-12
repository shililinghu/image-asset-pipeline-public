from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
import storage
from external_image_importer import import_external_images
from human_review_exporter import export_human_review_csv
from image_auto_reviewer import review_image_results
from storage import JsonStore


class ExternalImageImporterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        data_dir = Path(self.tmp.name) / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        files = {name: data_dir / f"{name}.json" for name in config.DATA_FILES}
        config.DATA_FILES.clear()
        config.DATA_FILES.update(files)
        storage.DATA_FILES = config.DATA_FILES
        storage.DATA_DIR = data_dir
        self.store = JsonStore()
        self.image_dir = Path(self.tmp.name) / "input_images"
        self.image_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_import_external_images_and_export_review_csv(self) -> None:
        image_path = self.image_dir / "manual.png"
        image_path.write_bytes(b"\x89PNG\r\n\x1a\nmanual")

        imported = import_external_images(
            request_id="request_external",
            prompt_id="prompt_external",
            pack_id="pack_external",
            image_dir=self.image_dir,
            store=self.store,
        )

        self.assertEqual(len(imported), 1)
        result = imported[0]
        self.assertEqual(result["request_id"], "request_external")
        self.assertEqual(result["prompt_id"], "prompt_external")
        self.assertEqual(result["pack_id"], "pack_external")
        self.assertEqual(result["provider"], "external_import")
        self.assertEqual(result["source"], "external_import")
        self.assertTrue(Path(result["image_path"]).exists())

        reviews = review_image_results(imported, store=self.store)
        self.assertEqual(len(reviews), 1)

        csv_path = export_human_review_csv(
            request_id="request_external",
            output_path=Path(self.tmp.name) / "human_review.csv",
            store=self.store,
        )
        with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["image_result_id"], result["image_result_id"])

    def test_import_external_images_from_mapping_csv(self) -> None:
        image_path = self.image_dir / "mapped.png"
        image_path.write_bytes(b"\x89PNG\r\n\x1a\nmapped")
        mapping_path = self.image_dir / "mapping.csv"
        with mapping_path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=["filename", "request_id", "prompt_id", "pack_id"])
            writer.writeheader()
            writer.writerow(
                {
                    "filename": "mapped.png",
                    "request_id": "request_csv",
                    "prompt_id": "prompt_csv",
                    "pack_id": "",
                }
            )

        imported = import_external_images(image_dir=self.image_dir, mapping_csv=mapping_path, store=self.store)
        self.assertEqual(len(imported), 1)
        self.assertEqual(imported[0]["request_id"], "request_csv")
        self.assertEqual(imported[0]["prompt_id"], "prompt_csv")


if __name__ == "__main__":
    unittest.main()
