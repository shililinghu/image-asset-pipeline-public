from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
import delete_manager
import storage
import image_generation_client
from human_review_exporter import export_human_review_csv
from idea_analyzer import analyze_idea
from image_auto_reviewer import review_image_results
from image_generation_runner import create_tasks_from_prompts, run_image_generation_tasks, run_prompts_for_request
from models import create_human_image_review, make_prompt
from prompt_feedback_bridge import build_prompt_image_feedback
from prompt_pack_builder import build_pack_from_prompt_results
from prompt_pack_matcher import match_prompt_pack
from prompt_pack_registry import PromptPackRegistry
from prompt_pack_runner import natural_prompt_id_key, run_prompt_pack, select_prompt_pack_items
from prompt_version_manager import commit_prompt_evolution, evolve_prompt
from storage import JsonStore


class PromptPackWorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        data_dir = Path(self.tmp.name) / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        files = {name: data_dir / f"{name}.json" for name in config.DATA_FILES}
        config.DATA_FILES.clear()
        config.DATA_FILES.update(files)
        storage.DATA_FILES = config.DATA_FILES
        storage.DATA_DIR = data_dir
        delete_manager.BACKUP_DIR = Path(self.tmp.name) / "backups"
        image_generation_client.IMAGE_OUTPUT_DIR = Path(self.tmp.name) / "mock_images"
        self.store = JsonStore()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_v01_workflow_components(self) -> None:
        request = analyze_idea("小红书 化学 封面 3:4 clean education about 分子结构, 生成 3 张", store=self.store)
        self.assertTrue(request["request_id"])
        self.assertEqual(request["domain"], "chemistry")

        registry = PromptPackRegistry(self.store)
        imported_pack = registry.import_pack_file(ROOT / "examples" / "sample_prompt_pack.json")
        self.assertEqual(imported_pack["domain"], "chemistry")
        pack = registry.create_pack(
            {
                "pack_name": "Chemistry Pack",
                "domain": "chemistry",
                "platform": "xiaohongshu",
                "use_case": "cover",
                "style": "clean_education",
                "ratio": "3:4",
                "status": "approved",
                "match_tags": ["chemistry", "xiaohongshu", "cover", "text_required"],
            },
            [
                {
                    "prompt_id": "p1",
                    "prompt_content": "cover for {topic}",
                    "variables": ["topic"],
                    "ratio": "3:4",
                    "count": 1,
                }
            ],
        )
        self.assertEqual(registry.get_pack(pack["pack_id"])["pack_name"], "Chemistry Pack")

        match = match_prompt_pack(request, store=self.store)
        self.assertTrue(match["matched"])
        self.assertIn(match["pack_id"], {pack["pack_id"], imported_pack["pack_id"]})

        pack_run = run_prompt_pack(pack["pack_id"], request, store=self.store)
        self.assertTrue(pack_run["tasks"])
        self.assertTrue(pack_run["image_results"])

        reviews = review_image_results(pack_run["image_results"], store=self.store)
        self.assertTrue(reviews)

        csv_path = export_human_review_csv(request["request_id"], output_path=Path(self.tmp.name) / "human.csv", store=self.store)
        self.assertTrue(csv_path.exists())
        with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        self.assertTrue(rows)

        for result in self.store.load("image_results"):
            self.store.add(
                "human_image_reviews",
                create_human_image_review(
                    {
                        "image_result_id": result["image_result_id"],
                        "request_id": result["request_id"],
                        "pack_id": result["pack_id"],
                        "prompt_id": result["prompt_id"],
                        "human_decision": "keep",
                        "human_score": 5,
                    }
                ),
            )
        feedback = build_prompt_image_feedback(request["request_id"], store=self.store, sync_legacy_feedback=False)
        self.assertTrue(feedback)

    def test_pack_builder_selects_qualified_prompts(self) -> None:
        request = analyze_idea("小红书 化学 封面 3:4 clean education about 酸碱反应", store=self.store)
        prompts = [
            make_prompt("project_x", 1, f"Prompt {index}", f"Create chemistry cover {index}", "imported")
            for index in range(3)
        ]
        for prompt in prompts:
            self.store.add("prompts", prompt)
        for prompt in prompts:
            _, results = run_prompts_for_request(request, [prompt], count_per_prompt=3, store=self.store)
            reviews = review_image_results(results, store=self.store)
            for review in reviews:
                self.store.update(
                    "image_reviews",
                    "review_id",
                    review["review_id"],
                    {"auto_score": 90, "safety_passed": True, "auto_decision": "keep"},
                )
            for result in results:
                self.store.add(
                    "human_image_reviews",
                    create_human_image_review(
                        {
                            "image_result_id": result["image_result_id"],
                            "request_id": request["request_id"],
                            "pack_id": "",
                            "prompt_id": prompt["prompt_id"],
                            "human_decision": "keep",
                            "human_score": 5,
                        }
                    ),
                )
        built = build_pack_from_prompt_results(request, store=self.store)
        self.assertGreaterEqual(len(built["items"]), 3)
        self.assertIn(built["pack"]["status"], {"candidate", "approved"})

    def test_import_pack_file_expands_prompts_array_to_items(self) -> None:
        pack_path = Path(self.tmp.name) / "portrait_pack.json"
        pack_path.write_text(
            json.dumps(
                {
                    "pack_id": "ai_portrait_photography_v1",
                    "pack_name": "AI Portrait Pack",
                    "domain": "portrait",
                    "platform": "general",
                    "use_case": "portrait",
                    "style": "editorial",
                    "ratio": "3:4",
                    "status": "candidate",
                    "prompts": [
                        {
                            "prompt_id": "P001",
                            "title": "valid prompt field",
                            "prompt": "portrait prompt one",
                            "negative_prompt": "bad hands",
                            "count": "bad-count",
                        },
                        {
                            "prompt_id": "P002",
                            "prompt_content": "portrait prompt two",
                            "ratio": "",
                        },
                        {
                            "prompt_id": "P003",
                            "content": "portrait prompt three",
                            "ratio": "1:1",
                            "count": 2,
                        },
                        {
                            "prompt_id": "P004",
                            "title": "missing body",
                        },
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        registry = PromptPackRegistry(self.store)
        pack = registry.import_pack_file(pack_path)
        summary = pack["import_summary"]
        items = registry.get_items("ai_portrait_photography_v1")

        self.assertEqual(summary["prompts_found"], 4)
        self.assertEqual(summary["imported_count"], 3)
        self.assertEqual(summary["updated_count"], 0)
        self.assertEqual(summary["skipped_invalid_count"], 1)
        self.assertEqual(len(items), 3)
        item_map = {item["prompt_id"]: item for item in items}
        self.assertEqual(item_map["P001"]["prompt_content"], "portrait prompt one")
        self.assertEqual(item_map["P001"]["count"], 1)
        self.assertEqual(item_map["P001"]["ratio"], "3:4")
        self.assertEqual(item_map["P001"]["variables"], ["topic"])
        self.assertEqual(item_map["P003"]["prompt_content"], "portrait prompt three")
        self.assertEqual(item_map["P003"]["count"], 2)
        self.assertEqual(item_map["P003"]["ratio"], "1:1")

        second_pack = registry.import_pack_file(pack_path)
        second_summary = second_pack["import_summary"]
        self.assertEqual(second_summary["imported_count"], 0)
        self.assertEqual(second_summary["skipped_existing_count"], 3)
        self.assertEqual(len(registry.get_items("ai_portrait_photography_v1")), 3)

    def test_delete_prompt_pack_removes_pack_and_items_with_backup(self) -> None:
        registry = PromptPackRegistry(self.store)
        pack = registry.create_pack(
            {"pack_id": "delete_pack", "pack_name": "Delete Pack", "status": "candidate"},
            [
                {"prompt_id": "P001", "prompt_content": "one"},
                {"prompt_id": "P002", "prompt_content": "two"},
            ],
        )
        result = delete_manager.delete_prompt_pack(pack["pack_id"], self.store)
        self.assertTrue(result["success"])
        self.assertEqual(result["deleted_pack_count"], 1)
        self.assertEqual(result["deleted_item_count"], 2)
        self.assertIsNone(registry.get_pack(pack["pack_id"]))
        self.assertEqual(registry.get_items(pack["pack_id"]), [])
        self.assertEqual(len(result["backup_files"]), 2)
        for path in result["backup_files"]:
            self.assertTrue(Path(path).exists())

    def test_delete_prompt_pack_with_zero_items(self) -> None:
        registry = PromptPackRegistry(self.store)
        pack = registry.create_pack({"pack_id": "empty_pack", "pack_name": "Empty Pack", "status": "candidate"}, [])
        result = delete_manager.delete_prompt_pack(pack["pack_id"], self.store)
        self.assertTrue(result["success"])
        self.assertEqual(result["deleted_pack_count"], 1)
        self.assertEqual(result["deleted_item_count"], 0)
        self.assertIn("没有对应 prompt items", result["message"])

    def test_prompt_pack_runner_selection_modes_and_natural_sort(self) -> None:
        registry = PromptPackRegistry(self.store)
        pack = registry.create_pack(
            {"pack_id": "range_pack", "pack_name": "Range Pack", "status": "candidate"},
            [
                {"prompt_id": "P001", "prompt_content": "one"},
                {"prompt_id": "P010", "prompt_content": "ten"},
                {"prompt_id": "P002", "prompt_content": "two"},
                {"prompt_id": "P003", "prompt_content": "three"},
            ],
        )
        items = registry.get_items(pack["pack_id"])
        self.assertEqual([item["prompt_id"] for item in select_prompt_pack_items(items, {"mode": "all"})], ["P001", "P002", "P003", "P010"])
        self.assertEqual([item["prompt_id"] for item in select_prompt_pack_items(items, {"mode": "first_n", "n": 2})], ["P001", "P002"])
        self.assertEqual([item["prompt_id"] for item in select_prompt_pack_items(items, {"mode": "range", "start": "P001", "end": "P003"})], ["P001", "P002", "P003"])
        self.assertEqual([item["prompt_id"] for item in select_prompt_pack_items(items, {"mode": "list", "prompt_ids": ["P001", "P003"]})], ["P001", "P003"])
        self.assertEqual(select_prompt_pack_items(items, {"mode": "list", "prompt_ids": ["PX"]}), [])
        self.assertLess(natural_prompt_id_key("P2"), natural_prompt_id_key("P10"))

        request = {"request_id": "request_range", "topic": "topic", "ratio": "3:4"}
        result = run_prompt_pack(pack["pack_id"], request, store=self.store, prompt_selection={"mode": "first_n", "n": 3})
        self.assertEqual(result["summary"]["total_item_count"], 4)
        self.assertEqual(result["summary"]["selected_item_count"], 3)
        self.assertEqual(result["summary"]["generated_task_count"], 3)
        self.assertEqual(result["summary"]["selection_mode"], "first_n")

    def test_prompt_pack_runner_generates_prompt_variants(self) -> None:
        registry = PromptPackRegistry(self.store)
        pack = registry.create_pack(
            {"pack_id": "variant_pack", "pack_name": "Variant Pack", "status": "candidate"},
            [{"prompt_id": "P001", "prompt_content": "portrait of {topic}", "count": 7}],
        )
        request = {"request_id": "request_variant", "topic": "chemist", "ratio": "3:4"}

        first_run = run_prompt_pack(
            pack["pack_id"],
            request,
            store=self.store,
            prompt_selection={"mode": "first_n", "n": 1},
            images_per_prompt=3,
        )
        tasks = first_run["tasks"]
        self.assertEqual(len(tasks), 3)
        self.assertEqual(first_run["summary"]["selected_item_count"], 1)
        self.assertEqual(first_run["summary"]["images_per_prompt"], 3)
        self.assertEqual(first_run["summary"]["generated_task_count"], 3)
        self.assertEqual(first_run["summary"]["variant_task_count"], 3)
        self.assertEqual(first_run["run"]["images_per_prompt"], 3)
        self.assertEqual([task["variant_index"] for task in tasks], [1, 2, 3])
        self.assertEqual(len({task["task_id"] for task in tasks}), 3)
        self.assertEqual(len({task["prompt_nonce"] for task in tasks}), 3)
        self.assertEqual(len({task["submitted_prompt"] for task in tasks}), 3)
        for task in tasks:
            self.assertEqual(task["base_prompt"], "portrait of chemist")
            self.assertIn("变体编号", task["submitted_prompt"])
            self.assertEqual(task["metadata"]["images_per_prompt"], 3)
            self.assertTrue(task["metadata"]["submitted_prompt_hash"])

        second_run = run_prompt_pack(
            pack["pack_id"],
            request,
            store=self.store,
            prompt_selection={"mode": "first_n", "n": 1},
            images_per_prompt=3,
        )
        self.assertFalse({task["task_id"] for task in tasks} & {task["task_id"] for task in second_run["tasks"]})
        self.assertFalse({task["prompt_nonce"] for task in tasks} & {task["prompt_nonce"] for task in second_run["tasks"]})
        self.assertEqual(len({result["image_result_id"] for result in first_run["image_results"] + second_run["image_results"]}), 6)

    def test_prompt_pack_runner_images_per_prompt_defaults_to_one(self) -> None:
        registry = PromptPackRegistry(self.store)
        pack = registry.create_pack(
            {"pack_id": "default_variant_pack", "pack_name": "Default Variant Pack", "status": "candidate"},
            [{"prompt_id": "P001", "prompt_content": "one", "count": "bad"}],
        )
        request = {"request_id": "request_default_variant", "topic": "topic", "ratio": "3:4"}
        result = run_prompt_pack(pack["pack_id"], request, store=self.store, images_per_prompt="bad")
        self.assertEqual(len(result["tasks"]), 1)
        self.assertEqual(result["tasks"][0]["variant_index"], 1)
        self.assertEqual(result["summary"]["images_per_prompt"], 1)

    def test_prompt_pack_items_are_versioned_and_runner_uses_active_version(self) -> None:
        registry = PromptPackRegistry(self.store)
        pack = registry.create_pack(
            {"pack_id": "versioned_pack", "pack_name": "Versioned Pack", "status": "candidate"},
            [{"prompt_id": "P001", "prompt_content": "version one prompt"}],
        )
        item = registry.get_items(pack["pack_id"])[0]
        self.assertTrue(item["logical_prompt_id"])
        self.assertTrue(item["active_prompt_version_id"])
        version = self.store.get("prompt_versions", "prompt_version_id", item["active_prompt_version_id"])
        self.assertEqual(version["version_number"], 1)
        self.assertEqual(version["status"], "active")

        self.store.update(
            "prompt_versions",
            "prompt_version_id",
            item["active_prompt_version_id"],
            {"prompt_content": "active version prompt"},
        )
        result = run_prompt_pack(pack["pack_id"], {"request_id": "request_version", "description": "same subject identity"}, store=self.store)
        self.assertEqual(result["tasks"][0]["base_prompt"], "active version prompt")

    def test_prompt_evolution_creates_candidate_and_commit_archives_old_version(self) -> None:
        registry = PromptPackRegistry(self.store)
        pack = registry.create_pack(
            {"pack_id": "evolve_pack", "pack_name": "Evolve Pack", "status": "candidate"},
            [{"prompt_id": "P001", "prompt_content": "cinematic portrait prompt"}],
        )
        item = registry.get_items(pack["pack_id"])[0]
        base_version_id = item["active_prompt_version_id"]
        evolved = evolve_prompt(
            pack["pack_id"],
            item["logical_prompt_id"],
            base_version_id,
            evolution_notes="增强电影感，减少背景干扰",
            validation_count=2,
            store=self.store,
        )
        candidate = evolved["candidate_version"]
        self.assertEqual(candidate["version_number"], 2)
        self.assertEqual(candidate["status"], "candidate")
        self.assertIn("增强电影感", candidate["prompt_content"])
        self.assertEqual(len(self.store.filter("prompt_versions", logical_prompt_id=item["logical_prompt_id"])), 2)

        first_result = evolved["image_results"][0]
        self.store.add(
            "human_image_reviews",
            {
                "human_review_id": "human_evolve_keep",
                "image_result_id": first_result["image_result_id"],
                "prompt_id": first_result["prompt_id"],
                "human_decision": "keep",
                "human_score": 5,
                "created_at": "2026-01-01T00:00:00",
            },
        )
        committed = commit_prompt_evolution(evolved["run"]["evolution_run_id"], store=self.store)
        self.assertTrue(committed["archived_old_version"])
        old_version = self.store.get("prompt_versions", "prompt_version_id", base_version_id)
        new_version = self.store.get("prompt_versions", "prompt_version_id", candidate["prompt_version_id"])
        updated_item = self.store.get("prompt_pack_items", "logical_prompt_id", item["logical_prompt_id"])
        self.assertEqual(old_version["status"], "archived")
        self.assertTrue(old_version["is_archived"])
        self.assertEqual(new_version["status"], "active")
        self.assertEqual(updated_item["active_prompt_version_id"], candidate["prompt_version_id"])

        evolved_again = evolve_prompt(
            pack["pack_id"],
            item["logical_prompt_id"],
            candidate["prompt_version_id"],
            evolution_notes="进一步统一主体气质",
            validation_count=1,
            store=self.store,
        )
        self.assertEqual(evolved_again["candidate_version"]["version_number"], 3)
        self.assertEqual(len(self.store.filter("prompt_versions", logical_prompt_id=item["logical_prompt_id"])), 3)

    def test_image_generation_runner_uses_submitted_prompt(self) -> None:
        request = {"request_id": "request_submit", "ratio": "3:4"}
        prompts = [
            {
                "prompt_id": "P001",
                "prompt_content": "base prompt",
                "base_prompt": "base prompt",
                "submitted_prompt": "submitted prompt with nonce",
                "variant_index": 2,
                "images_per_prompt": 3,
                "prompt_nonce": "nonce_2",
                "metadata": {
                    "run_id": "run_x",
                    "variant_index": 2,
                    "images_per_prompt": 3,
                    "prompt_nonce": "nonce_2",
                    "base_prompt_hash": "basehash",
                    "submitted_prompt_hash": "subhash",
                },
            }
        ]
        tasks = create_tasks_from_prompts(request, prompts, pack_id="pack_x", store=self.store, save=False)

        class CaptureClient:
            provider = "capture"
            model = "capture-model"

            def __init__(self) -> None:
                self.prompts: list[str] = []

            def generate(self, prompt, negative_prompt=None, ratio="3:4", metadata=None):
                self.prompts.append(prompt)
                return {
                    "image_path": "",
                    "image_url": "http://x/image.png",
                    "provider": self.provider,
                    "model": self.model,
                    "status": "success",
                    "error_message": "",
                    "metadata": metadata or {},
                }

        client = CaptureClient()
        results = run_image_generation_tasks(tasks, store=self.store, client=client, save=False)
        self.assertIn("submitted prompt with nonce", client.prompts[0])
        self.assertIn("【主体锁定】", client.prompts[0])
        self.assertEqual(results[0]["metadata"]["base_prompt"], "base prompt")
        self.assertIn("submitted prompt with nonce", results[0]["metadata"]["submitted_prompt"])
        self.assertEqual(results[0]["metadata"]["variant_index"], 2)
        self.assertTrue(results[0]["metadata"]["subject_profile_id"])

    def test_subject_lock_creates_profile_and_injects_variant_prompt(self) -> None:
        registry = PromptPackRegistry(self.store)
        pack = registry.create_pack(
            {"pack_id": "subject_lock_pack", "pack_name": "Subject Lock Pack", "status": "candidate"},
            [{"prompt_id": "P001", "prompt_content": "commercial portrait in city"}],
        )
        request = {
            "request_id": "request_subject_lock",
            "title": "同一人物多主题人像",
            "description": "年轻男性，黑发，冷静表情，高级商业摄影气质",
            "topic": "人像",
            "ratio": "3:4",
        }
        result = run_prompt_pack(pack["pack_id"], request, store=self.store, images_per_prompt=2)
        profile_ids = {task["subject_profile_id"] for task in result["tasks"]}
        self.assertEqual(len(profile_ids), 1)
        self.assertTrue(next(iter(profile_ids)))
        for task in result["tasks"]:
            self.assertIn("【主体锁定】", task["submitted_prompt"])
            self.assertIn("年轻男性", task["submitted_prompt"])
            self.assertTrue(task["metadata"]["subject_profile_id"])
            self.assertFalse(task["metadata"]["provider_supports_reference_image"])
        for image_result in result["image_results"]:
            self.assertEqual(image_result["metadata"]["subject_profile_id"], next(iter(profile_ids)))


if __name__ == "__main__":
    unittest.main()
