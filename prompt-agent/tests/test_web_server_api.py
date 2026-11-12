from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
import delete_manager
import storage
import web_server
from prompt_type_profiles import get_prompt_type_profile, load_prompt_type_profiles
from prompt_pack_registry import PromptPackRegistry
from storage import JsonStore


class WebServerApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.old_env = {key: os.environ.get(key) for key in ["WUYIN_API_KEY", "OPENAI_API_KEY", "IMAGE_API_KEY", "TASTELENS_API_KEY"]}
        self.data_dir = self.root / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.old_config = {
            "BASE_DIR": config.BASE_DIR,
            "DATA_DIR": config.DATA_DIR,
            "MOCK_IMAGE_OUTPUT_DIR": config.MOCK_IMAGE_OUTPUT_DIR,
            "REAL_IMAGE_OUTPUT_DIR": config.REAL_IMAGE_OUTPUT_DIR,
            "INPUT_IMAGES_DIR": config.INPUT_IMAGES_DIR,
            "DATA_FILES": dict(config.DATA_FILES),
            "APP_SETTINGS_FILE": config.APP_SETTINGS_FILE,
            "SECRETS_LOCAL_FILE": config.SECRETS_LOCAL_FILE,
            "SETTINGS_MODE": config.SETTINGS_MODE,
        }
        config.BASE_DIR = self.root
        config.DATA_DIR = self.data_dir
        config.MOCK_IMAGE_OUTPUT_DIR = self.root / "exports" / "mock_images"
        config.REAL_IMAGE_OUTPUT_DIR = self.root / "exports" / "real_images"
        config.INPUT_IMAGES_DIR = self.root / "input_images"
        config.APP_SETTINGS_FILE = self.data_dir / "app_settings.json"
        config.SECRETS_LOCAL_FILE = self.data_dir / "secrets.local.json"
        config.SETTINGS_MODE = "local"
        config.MOCK_IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        config.REAL_IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        config.INPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        files = {name: self.data_dir / f"{name}.json" for name in self.old_config["DATA_FILES"]}
        config.DATA_FILES.clear()
        config.DATA_FILES.update(files)
        storage.DATA_FILES = config.DATA_FILES
        storage.DATA_DIR = self.data_dir
        delete_manager.BACKUP_DIR = self.root / "backups"
        self.store = JsonStore()
        self.registry = PromptPackRegistry(self.store)
        self.pack = self.registry.create_pack(
            {"pack_id": "web_pack", "pack_name": "Web Pack", "status": "candidate"},
            [{"prompt_id": "P001", "prompt_content": "test prompt"}],
        )
        self.store.add(
            "idea_requests",
            {
                "request_id": "request_web",
                "topic": "topic",
                "description": "同一人物主体，保持脸型、发型和整体识别感一致",
                "ratio": "3:4",
                "created_at": "2026-01-01T00:00:00",
            },
        )
        profile_dir = self.root / "provider_profiles"
        profile_dir.mkdir(parents=True, exist_ok=True)
        (profile_dir / "mock.example.json").write_text(
            json.dumps({"profile_id": "mock_profile", "provider": "mock", "model": "mock"}),
            encoding="utf-8",
        )
        web_dir = self.root / "web"
        web_dir.mkdir(parents=True, exist_ok=True)
        (web_dir / "index.html").write_text("<!doctype html><script src='/app.js'></script>", encoding="utf-8")
        (web_dir / "app.js").write_text("console.log('ok');", encoding="utf-8")
        (web_dir / "style.css").write_text("body{}", encoding="utf-8")
        self.server = web_server.make_server("127.0.0.1", 0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        config.BASE_DIR = self.old_config["BASE_DIR"]
        config.DATA_DIR = self.old_config["DATA_DIR"]
        config.MOCK_IMAGE_OUTPUT_DIR = self.old_config["MOCK_IMAGE_OUTPUT_DIR"]
        config.REAL_IMAGE_OUTPUT_DIR = self.old_config["REAL_IMAGE_OUTPUT_DIR"]
        config.INPUT_IMAGES_DIR = self.old_config["INPUT_IMAGES_DIR"]
        config.DATA_FILES.clear()
        config.DATA_FILES.update(self.old_config["DATA_FILES"])
        storage.DATA_FILES = config.DATA_FILES
        storage.DATA_DIR = self.old_config["DATA_DIR"]
        self.tmp.cleanup()
        config.APP_SETTINGS_FILE = self.old_config["APP_SETTINGS_FILE"]
        config.SECRETS_LOCAL_FILE = self.old_config["SECRETS_LOCAL_FILE"]
        config.SETTINGS_MODE = self.old_config["SETTINGS_MODE"]
        for key, value in self.old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_status_returns_200(self) -> None:
        payload = self._get_json("/api/status")
        self.assertIn("prompt_pack_count", payload)
        self.assertEqual(payload["prompt_pack_count"], 1)

    def test_prompt_packs_returns_list(self) -> None:
        payload = self._get_json("/api/prompt-packs")
        self.assertEqual(len(payload["prompt_packs"]), 1)
        self.assertEqual(payload["prompt_packs"][0]["item_count"], 1)
        self.assertEqual(payload["prompt_packs"][0]["prompt_type"], "portrait")
        self.assertEqual(payload["prompt_packs"][0]["review_profile"], "portrait_review")

    def test_provider_profiles_returns_list(self) -> None:
        payload = self._get_json("/api/provider-profiles")
        self.assertEqual(payload["provider_profiles"][0]["profile_id"], "mock_profile")
        self.assertNotIn("api_key", payload["provider_profiles"][0])

    def test_settings_api_masks_and_saves_secrets(self) -> None:
        settings = self._get_json("/api/settings")
        self.assertTrue(settings["ok"])
        self.assertIn("WUYIN_API_KEY", settings["secrets"])
        saved = self._post_json(
            "/api/settings",
            {
                "image_provider": "wuyin_async",
                "image_provider_profile_path": "provider_profiles/mock.example.json",
                "aesthetic_review_provider": "tastelens_http",
                "aesthetic_review_api_url": "http://127.0.0.1:3000/api/analyze",
            },
        )
        self.assertEqual(saved["settings"]["image_provider"], "wuyin_async")
        secret = self._post_json("/api/settings/secrets", {"WUYIN_API_KEY": "example-test-value"})
        row = secret["secrets"]["WUYIN_API_KEY"]
        self.assertTrue(row["exists"])
        self.assertNotIn("example-test-value", json.dumps(secret))
        self.assertEqual(os.environ.get("WUYIN_API_KEY"), "example-test-value")
        settings = self._get_json("/api/settings")
        self.assertNotIn("example-test-value", json.dumps(settings))

    def test_delete_secret_api_removes_local_secret(self) -> None:
        self._post_json("/api/settings/secrets", {"WUYIN_API_KEY": "local-secret"})
        payload = self._delete_json("/api/settings/secrets/WUYIN_API_KEY")
        self.assertTrue(payload["ok"])
        settings = self._get_json("/api/settings")
        if settings["secrets"]["WUYIN_API_KEY"]["source"] != "environment":
            self.assertFalse(settings["secrets"]["WUYIN_API_KEY"]["exists"])

    def test_ports_and_diagnostics_apis(self) -> None:
        ports = self._get_json("/api/ports/check")
        self.assertIn("web_console", ports)
        self.assertIn("tastelens", ports)
        diagnostics = self._post_json("/api/diagnostics/run", {"include_real_image_test": False, "include_aesthetic_test": False})
        self.assertIn("overall_status", diagnostics)
        self.assertTrue(diagnostics["checks"])
        names = {item["name"] for item in diagnostics["checks"]}
        self.assertIn("Python", names)

    def test_prompt_type_profiles_load(self) -> None:
        profiles = load_prompt_type_profiles()
        type_ids = {item["type_id"] for item in profiles}
        self.assertIn("portrait", type_ids)
        self.assertIn("game_asset", type_ids)
        self.assertTrue(get_prompt_type_profile("portrait")["requires_reference_images"])
        self.assertFalse(get_prompt_type_profile("poster")["requires_reference_images"])
        payload = self._get_json("/api/prompt-type-profiles")
        self.assertTrue(payload["prompt_type_profiles"])

    def test_reference_images_returns_list(self) -> None:
        reference_dir = self.root / web_server.REFERENCE_IMAGE_DIR_NAME
        reference_dir.mkdir(parents=True, exist_ok=True)
        (reference_dir / "ref.png").write_bytes(b"png")
        payload = self._get_json("/api/reference-images")
        self.assertEqual(len(payload["reference_images"]), 1)
        self.assertEqual(payload["reference_images"][0]["file_url"], "/files/input_reference_images/ref.png")

    def test_create_idea_request(self) -> None:
        payload = self._post_json(
            "/api/idea-requests",
            {
                "title": "Portrait",
                "description": "Make portrait prompts",
                "domain": "portrait",
                "platform": "general",
                "use_case": "portrait",
                "style": "studio",
                "ratio": "3:4",
                "topic": "portrait",
                "requirements": "consistent person",
            },
        )
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["request_id"])
        self.assertEqual(payload["request"]["title"], "Portrait")

    def test_create_idea_request_uses_incubation_defaults(self) -> None:
        payload = self._post_json(
            "/api/idea-requests",
            {
                "title": "Same Person Portraits",
                "description": "Generate portrait variations from references",
                "ratio": "3:4",
            },
        )
        request = payload["request"]
        self.assertTrue(payload["ok"])
        self.assertEqual(request["domain"], "portrait_photography")
        self.assertEqual(request["platform"], "general")
        self.assertEqual(request["use_case"], "portrait_image_generation")
        self.assertEqual(request["style"], "ai_portrait_photography")
        self.assertEqual(request["topic"], "Same Person Portraits")
        self.assertEqual(request["requirements"], "Generate portrait variations from references")

    def test_create_idea_request_uses_prompt_type_defaults(self) -> None:
        payload = self._post_json(
            "/api/idea-requests",
            {
                "prompt_type": "game_asset",
                "title": "色环游戏图标",
                "description": "生成一组适合网页卡牌游戏使用的图标，风格统一，易识别。",
            },
        )
        request = payload["request"]
        self.assertEqual(request["prompt_type"], "game_asset")
        self.assertEqual(request["domain"], "game_asset_design")
        self.assertEqual(request["platform"], "game")
        self.assertEqual(request["use_case"], "game_asset_generation")
        self.assertEqual(request["style"], "stylized_game_asset")
        self.assertEqual(request["review_profile"], "game_asset_review")

    def test_create_idea_request_default_topic_when_title_missing(self) -> None:
        payload = self._post_json("/api/idea-requests", {"description": "Only description"})
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["request"]["topic"], "人像")

    def test_reference_upload_multiple_images_and_filter(self) -> None:
        payload = self._multipart(
            "/api/reference-images/upload",
            {"request_id": "request_web"},
            [("files", "one.png", b"png1"), ("files", "two.jpg", b"jpg2")],
        )
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["uploaded_count"], 2)
        filtered = self._get_json("/api/reference-images?request_id=request_web")
        self.assertEqual(len(filtered["reference_images"]), 2)

    def test_reference_upload_zip_extracts_images_and_blocks_traversal(self) -> None:
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("good.png", b"png")
            archive.writestr("../evil.png", b"evil")
            archive.writestr("note.txt", b"text")
        payload = self._multipart(
            "/api/reference-images/upload",
            {"request_id": "request_zip"},
            [("files", "refs.zip", buffer.getvalue())],
        )
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["uploaded_count"], 2)
        for item in payload["reference_images"]:
            self.assertTrue(str(item["path"]).startswith("input_reference_images"))
            self.assertNotIn("..", item["path"])

    def test_extract_image_prompts_requires_reference_images(self) -> None:
        payload = self._post_json("/api/extract-image-prompts", {"request_id": "request_web", "reference_image_ids": []})
        self.assertFalse(payload["ok"])
        self.assertIn("reference", payload["error"])

    def test_incubation_flow_builds_pack(self) -> None:
        upload = self._multipart(
            "/api/reference-images/upload",
            {"request_id": "request_web"},
            [("files", "ref.png", b"png")],
        )
        extract = self._post_json(
            "/api/extract-image-prompts",
            {
                "request_id": "request_web",
                "reference_image_ids": [item["reference_image_id"] for item in upload["reference_images"]],
            },
        )
        self.assertTrue(extract["ok"])
        candidates = self._post_json(
            "/api/generate-candidate-prompts",
            {"request_id": "request_web", "extract_run_id": extract["extract_run_id"], "candidate_count": 2},
        )
        self.assertTrue(candidates["ok"])
        self.assertEqual(candidates["candidate_prompt_count"], 2)
        missing = self._post_json("/api/validate-candidate-prompts", {"request_id": "request_web"})
        self.assertFalse(missing["ok"])
        validation = self._post_json(
            "/api/validate-candidate-prompts",
            {
                "request_id": "request_web",
                "candidate_prompt_ids": candidates["candidate_prompt_ids"],
                "images_per_prompt": 1,
            },
        )
        self.assertTrue(validation["ok"])
        self._post_json("/api/auto-review", {"request_id": "request_web"})
        first_result = self.store.load("image_results")[0]
        self._post_json(
            "/api/human-review",
            {
                "image_result_id": first_result["image_result_id"],
                "prompt_id": first_result["prompt_id"],
                "human_decision": "keep",
                "human_score": 5,
            },
        )
        built = self._post_json(
            "/api/build-prompt-pack-from-reviewed-candidates",
            {
                "request_id": "request_web",
                "candidate_prompt_ids": candidates["candidate_prompt_ids"],
                "pack_name": "Built From Candidates",
                "min_aesthetic_score": 70,
                "accepted_decisions": ["keep", "high_potential"],
            },
        )
        self.assertTrue(built["ok"])
        self.assertTrue(built["built_pack_id"])

    def test_one_click_incubation_and_build_pack_from_incubation(self) -> None:
        self.store.add("idea_requests", {"request_id": "request_one_click_short", "topic": "人像", "description": "短", "ratio": "3:4"})
        no_refs = self._post_json("/api/incubate-prompt-pack", {"request_id": "request_one_click_short", "reference_image_ids": []})
        self.assertFalse(no_refs["ok"])

        upload = self._multipart(
            "/api/reference-images/upload",
            {"request_id": "request_web"},
            [("files", "ref.png", b"png")],
        )
        incubation = self._post_json(
            "/api/incubate-prompt-pack",
            {
                "request_id": "request_web",
                "reference_image_ids": [item["reference_image_id"] for item in upload["reference_images"]],
                "candidate_count": 2,
                "images_per_prompt": 1,
            },
        )
        self.assertTrue(incubation["ok"])
        status = self._wait_for_incubation(incubation["incubation_run_id"])
        self.assertIn(status["status"], {"waiting_human_review", "completed"})
        self.assertIn("stage", status)
        self.assertIn("percent", status)
        images = self._get_json(f"/api/incubation-reviewed-images/{incubation['incubation_run_id']}")
        self.assertTrue(images["images"])

        first = images["images"][0]
        self._post_json(
            "/api/human-review",
            {
                "image_result_id": first["image_result_id"],
                "prompt_id": first["prompt_id"],
                "human_decision": "keep",
                "human_score": 5,
            },
        )
        built = self._post_json(
            "/api/build-pack-from-incubation",
            {
                "incubation_run_id": incubation["incubation_run_id"],
                "pack_name": "One Click Pack",
                "accepted_decisions": ["keep", "high_potential"],
            },
        )
        self.assertTrue(built["ok"])
        items = self.store.filter("prompt_pack_items", pack_id=built["built_pack_id"])
        self.assertTrue(items)
        self.assertIn(first["prompt_id"], {item["prompt_id"] for item in items})
        self.assertTrue(items[0]["source_incubation_run_id"])
        self.assertTrue(items[0]["subject_profile_id"])
        built_pack = self.registry.get_pack(built["built_pack_id"])
        self.assertTrue(built_pack["subject_profile_id"])
        self.assertTrue(built_pack["subject_lock_required"])
        self.assertFalse(any("structure" in item["prompt_id"] or "content" in item["prompt_id"] for item in items))

    def test_one_click_incubation_uses_default_candidate_and_image_counts(self) -> None:
        upload = self._multipart(
            "/api/reference-images/upload",
            {"request_id": "request_web"},
            [("files", "ref.png", b"png")],
        )
        incubation = self._post_json(
            "/api/incubate-prompt-pack",
            {
                "request_id": "request_web",
                "reference_image_ids": [item["reference_image_id"] for item in upload["reference_images"]],
                "candidate_count": "",
                "images_per_prompt": "",
            },
        )
        self.assertTrue(incubation["ok"])
        record = self.store.get("prompt_pack_incubation_runs", "incubation_run_id", incubation["incubation_run_id"])
        self.assertEqual(record["candidate_count"], 8)
        self.assertEqual(record["images_per_prompt"], 1)

    def test_incubation_without_reference_requires_clear_subject_description(self) -> None:
        self.store.add("idea_requests", {"request_id": "request_short", "topic": "人像", "description": "短", "ratio": "3:4"})
        payload = self._post_json("/api/incubate-prompt-pack", {"request_id": "request_short", "reference_image_ids": []})
        self.assertFalse(payload["ok"])
        self.assertIn("样图", payload["error"])

    def test_optional_reference_type_requires_clear_subject_description_without_image(self) -> None:
        self.store.add("idea_requests", {"request_id": "request_poster_short", "prompt_type": "poster", "topic": "海报", "description": "短", "ratio": "3:4"})
        payload = self._post_json("/api/incubate-prompt-pack", {"request_id": "request_poster_short", "reference_image_ids": []})
        self.assertFalse(payload["ok"])
        self.assertIn("主体描述", payload["error"])

    def test_incubation_without_reference_uses_text_subject_profile(self) -> None:
        created = self._post_json(
            "/api/idea-requests",
            {
                "prompt_type": "poster",
                "title": "同一人物",
                "description": "年轻男性，黑发，冷静表情，高级商业摄影气质，保持人物识别感一致",
            },
        )
        payload = self._post_json(
            "/api/incubate-prompt-pack",
            {
                "request_id": created["request_id"],
                "reference_image_ids": [],
                "candidate_count": 1,
                "images_per_prompt": 1,
            },
        )
        self.assertTrue(payload["ok"])
        record = self.store.get("prompt_pack_incubation_runs", "incubation_run_id", payload["incubation_run_id"])
        self.assertEqual(record["prompt_type"], "poster")
        self.assertEqual(record["review_profile"], "poster_review")
        profile = self.store.get("subject_profiles", "subject_profile_id", record["subject_profile_id"])
        self.assertEqual(profile["source"], "text_description")
        self.assertIn("年轻男性", profile["consistency_instruction"])

    def test_run_prompt_pack_uses_reference_subject_profile(self) -> None:
        upload = self._multipart(
            "/api/reference-images/upload",
            {"request_id": "request_web"},
            [("files", "ref.png", b"png")],
        )
        response = self._post_json(
            "/api/run-prompt-pack",
            {
                "request_id": "request_web",
                "pack_id": "web_pack",
                "prompt_selection": {"mode": "first_n", "n": 1},
                "reference_image_path": upload["reference_images"][0]["path"],
                "subject_lock": {"enabled": True, "subject_description": "参考样图中的同一人物，保持脸型和气质一致"},
            },
        )
        self.assertTrue(response["ok"])
        run = self.store.get("prompt_pack_runs", "run_id", response["run_id"])
        profile = self.store.get("subject_profiles", "subject_profile_id", run["subject_profile_id"])
        self.assertIn(upload["reference_images"][0]["path"], profile["reference_image_paths"])
        task = self.store.load("image_generation_tasks")[-1]
        self.assertEqual(task["subject_profile_id"], profile["subject_profile_id"])
        self.assertIn("【主体锁定】", task["submitted_prompt"])
        result = self.store.load("image_results")[-1]
        self.assertEqual(result["metadata"]["subject_profile_id"], profile["subject_profile_id"])
        self.assertFalse(result["metadata"]["provider_supports_reference_image"])

    def test_incubation_reviewed_images_include_filtered_out_reasons(self) -> None:
        self._seed_review_split_records()
        payload = self._get_json("/api/incubation-reviewed-images/inc_review_split")
        groups = {item["image_result_id"]: item for item in payload["images"]}

        self.assertEqual(groups["img_high"]["auto_group"], "included")
        self.assertEqual(groups["img_low"]["auto_group"], "filtered_out")
        self.assertIn("TasteLens 分数低于 70", groups["img_low"]["auto_reject_reasons"])
        self.assertEqual(groups["img_failed"]["auto_group"], "filtered_out")
        self.assertIn("图片生成失败", groups["img_failed"]["auto_reject_reason"])

    def test_reviewed_images_api_supports_latest_and_run_filters(self) -> None:
        self._seed_review_split_records()
        latest = self._get_json("/api/reviewed-images?latest=24")
        latest_by_id = {item["image_result_id"]: item for item in latest["images"]}
        self.assertIn("img_low", latest_by_id)
        self.assertEqual(latest_by_id["img_low"]["auto_group"], "filtered_out")
        self.assertIn("TasteLens 分数低于 70", latest_by_id["img_low"]["auto_reject_reason"])

        by_run = self._get_json("/api/reviewed-images?run_id=validation_split")
        self.assertEqual({item["image_result_id"] for item in by_run["images"]}, {"img_high", "img_low", "img_low_2", "img_failed", "img_revise", "img_reject"})

    def test_review_grouping_matches_incubation_and_generation_api(self) -> None:
        self._seed_review_split_records()
        incubation = self._get_json("/api/incubation-reviewed-images/inc_review_split")
        generation = self._get_json("/api/reviewed-images?run_id=validation_split")
        inc_groups = {item["image_result_id"]: item["auto_group"] for item in incubation["images"]}
        gen_groups = {item["image_result_id"]: item["auto_group"] for item in generation["images"]}
        self.assertEqual(inc_groups, gen_groups)

    def test_human_review_supports_rescue_source(self) -> None:
        self.store.add("image_results", {"image_result_id": "img_rescue", "request_id": "request_web", "prompt_id": "PRESCUE", "status": "success"})
        payload = self._post_json(
            "/api/human-review",
            {
                "image_result_id": "img_rescue",
                "prompt_id": "PRESCUE",
                "human_decision": "rescue",
                "human_score": 4,
            },
        )
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["human_review"]["human_decision"], "rescue")
        self.assertEqual(payload["human_review"]["source"], "filtered_out_rescue")

    def test_human_review_supports_generation_rescue_source(self) -> None:
        self.store.add("image_results", {"image_result_id": "img_gen_rescue", "request_id": "request_web", "prompt_id": "PGEN", "status": "success"})
        payload = self._post_json(
            "/api/human-review",
            {
                "image_result_id": "img_gen_rescue",
                "prompt_id": "PGEN",
                "human_decision": "rescue",
                "human_score": 4,
                "source": "generation_filtered_out_rescue",
            },
        )
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["human_review"]["source"], "generation_filtered_out_rescue")

    def test_build_pack_from_incubation_uses_human_keep_high_and_rescue_only(self) -> None:
        self._seed_review_split_records()
        decisions = {
            "img_high": ("keep", 5),
            "img_low": ("rescue", 4),
            "img_low_2": ("high_potential", 4),
            "img_revise": ("revise", 3),
            "img_reject": ("reject", 1),
        }
        for image_id, (decision, score) in decisions.items():
            self._post_json(
                "/api/human-review",
                {
                    "image_result_id": image_id,
                    "human_decision": decision,
                    "human_score": score,
                },
            )
        built = self._post_json(
            "/api/build-pack-from-incubation",
            {
                "incubation_run_id": "inc_review_split",
                "pack_name": "Rescue Pack",
            },
        )
        self.assertTrue(built["ok"])
        items = self.store.filter("prompt_pack_items", pack_id=built["built_pack_id"])
        by_prompt = {item["prompt_id"]: item for item in items}
        self.assertEqual(set(by_prompt), {"PHIGH", "PLOW"})
        self.assertTrue(by_prompt["PLOW"]["rescued_from_filtered_out"])
        self.assertIn("rescue", by_prompt["PLOW"]["source_selection_decisions"])
        self.assertIn("high_potential", by_prompt["PLOW"]["source_selection_decisions"])
        self.assertEqual(len(by_prompt["PLOW"]["source_image_result_ids"]), 2)
        self.assertTrue(by_prompt["PLOW"]["source_human_review_ids"])
        self.assertTrue(by_prompt["PLOW"]["auto_reject_reasons"])
        self.assertEqual(by_prompt["PLOW"]["approved_image_count"], 2)
        self.assertEqual(by_prompt["PLOW"]["tested_image_count"], 2)

    def test_build_pack_from_incubation_include_revise_only_when_enabled(self) -> None:
        self._seed_review_split_records()
        self._post_json("/api/human-review", {"image_result_id": "img_revise", "human_decision": "revise", "human_score": 3})
        rejected = self._post_json(
            "/api/build-pack-from-incubation",
            {"incubation_run_id": "inc_review_split", "pack_name": "No Revise Pack"},
        )
        self.assertFalse(rejected["ok"])
        built = self._post_json(
            "/api/build-pack-from-incubation",
            {
                "incubation_run_id": "inc_review_split",
                "pack_name": "Revise Pack",
                "accepted_decisions": ["keep", "high_potential", "rescue", "revise"],
                "include_revise": True,
            },
        )
        self.assertTrue(built["ok"])
        items = self.store.filter("prompt_pack_items", pack_id=built["built_pack_id"])
        self.assertEqual({item["prompt_id"] for item in items}, {"PREVISE"})

    def test_run_prompt_pack_missing_params_returns_error(self) -> None:
        payload = self._post_json("/api/run-prompt-pack", {})
        self.assertFalse(payload["ok"])
        self.assertIn("request_id", payload["error"])

    def test_run_status_returns_structure(self) -> None:
        run_payload = self._post_json(
            "/api/run-prompt-pack",
            {
                "request_id": "request_web",
                "pack_id": "web_pack",
                "prompt_selection": {"mode": "first_n", "n": 1},
            },
        )
        self.assertTrue(run_payload["ok"])
        status = self._get_json(f"/api/run-status/{run_payload['run_id']}")
        self.assertEqual(status["run_id"], run_payload["run_id"])
        self.assertIn("percent", status)
        self.assertIn("items", status)

    def test_evolve_and_commit_prompt_api(self) -> None:
        item = self.registry.get_items("web_pack")[0]
        evolve = self._post_json(
            "/api/evolve-prompt",
            {
                "pack_id": "web_pack",
                "logical_prompt_id": item["logical_prompt_id"],
                "base_prompt_version_id": item["active_prompt_version_id"],
                "evolution_notes": "增强清晰度和统一风格",
                "validation_count": 1,
            },
        )
        self.assertTrue(evolve["ok"])
        self.assertTrue(evolve["new_prompt_version_id"].endswith("_v2"))
        run = self.store.get("prompt_evolution_runs", "evolution_run_id", evolve["evolution_run_id"])
        validation = self.store.get("prompt_pack_runs", "run_id", run["validation_run_id"])
        image_result_id = validation["image_result_ids"][0]
        self._post_json(
            "/api/human-review",
            {
                "image_result_id": image_result_id,
                "human_decision": "keep",
                "human_score": 5,
            },
        )
        committed = self._post_json(
            "/api/commit-prompt-evolution",
            {
                "evolution_run_id": evolve["evolution_run_id"],
                "include_high_potential": True,
                "include_rescue": True,
            },
        )
        self.assertTrue(committed["ok"])
        self.assertEqual(committed["old_prompt_version_id"], item["active_prompt_version_id"])
        updated_item = self.store.get("prompt_pack_items", "logical_prompt_id", item["logical_prompt_id"])
        self.assertEqual(updated_item["active_prompt_version_id"], evolve["new_prompt_version_id"])

    def test_file_path_conversion_blocks_traversal(self) -> None:
        self.assertIsNone(web_server.resolve_safe_file_path("../config.py"))
        image_path = config.REAL_IMAGE_OUTPUT_DIR / "sample.png"
        image_path.write_bytes(b"png")
        url = web_server.file_url_for_path(str(image_path))
        self.assertEqual(url, "/files/exports/real_images/sample.png")

    def test_static_files_are_served(self) -> None:
        with urllib.request.urlopen(f"{self.base_url}/index.html", timeout=5) as response:
            self.assertEqual(response.status, 200)
        with urllib.request.urlopen(f"{self.base_url}/app.js", timeout=5) as response:
            self.assertEqual(response.status, 200)
        with urllib.request.urlopen(f"{self.base_url}/style.css", timeout=5) as response:
            self.assertEqual(response.status, 200)

    def test_delete_pack_without_confirm_does_not_delete(self) -> None:
        request = urllib.request.Request(f"{self.base_url}/api/prompt-packs/web_pack", method="DELETE")
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request, timeout=5)
        self.assertEqual(caught.exception.code, 400)
        self.assertIsNotNone(self.registry.get_pack("web_pack"))

    def _get_json(self, path: str) -> dict:
        with urllib.request.urlopen(f"{self.base_url}{path}", timeout=5) as response:
            self.assertEqual(response.status, 200)
            return json.loads(response.read().decode("utf-8"))

    def _post_json(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            self.assertEqual(response.status, 200)
            return json.loads(response.read().decode("utf-8"))

    def _delete_json(self, path: str) -> dict:
        request = urllib.request.Request(f"{self.base_url}{path}", method="DELETE")
        with urllib.request.urlopen(request, timeout=5) as response:
            self.assertEqual(response.status, 200)
            return json.loads(response.read().decode("utf-8"))

    def _multipart(self, path: str, fields: dict[str, str], files: list[tuple[str, str, bytes]]) -> dict:
        boundary = "----testboundary"
        parts: list[bytes] = []
        for name, value in fields.items():
            parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode("utf-8"))
        for field, filename, content in files:
            parts.append(
                (
                    f"--{boundary}\r\n"
                    f"Content-Disposition: form-data; name=\"{field}\"; filename=\"{filename}\"\r\n"
                    "Content-Type: application/octet-stream\r\n\r\n"
                ).encode("utf-8")
                + content
                + b"\r\n"
            )
        parts.append(f"--{boundary}--\r\n".encode("utf-8"))
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=b"".join(parts),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            self.assertEqual(response.status, 200)
            return json.loads(response.read().decode("utf-8"))

    def _wait_for_incubation(self, incubation_run_id: str) -> dict:
        for _ in range(40):
            payload = self._get_json(f"/api/incubation-status/{incubation_run_id}")
            if payload.get("status") in {"waiting_human_review", "completed", "failed"}:
                return payload
            time.sleep(0.1)
        self.fail("incubation did not finish")

    def _seed_review_split_records(self) -> None:
        prompt_ids = ["PHIGH", "PLOW", "PFAILED", "PREVISE", "PREJECT"]
        for prompt_id in prompt_ids:
            self.store.add(
                "prompts",
                {
                    "prompt_id": prompt_id,
                    "full_prompt": f"Prompt content {prompt_id}",
                    "negative_prompt": "bad quality",
                },
            )
        results = [
            ("img_high", "PHIGH", "success", ""),
            ("img_low", "PLOW", "success", ""),
            ("img_low_2", "PLOW", "success", ""),
            ("img_failed", "PFAILED", "failed", "provider failed"),
            ("img_revise", "PREVISE", "success", ""),
            ("img_reject", "PREJECT", "success", ""),
        ]
        for image_id, prompt_id, status, error in results:
            self.store.add(
                "image_results",
                {
                    "image_result_id": image_id,
                    "request_id": "request_web",
                    "pack_id": "candidate_validation",
                    "prompt_id": prompt_id,
                    "status": status,
                    "error_message": error,
                    "metadata": {"variant_index": 1},
                    "created_at": "2026-01-01T00:00:00",
                },
            )
        review_scores = {
            "img_high": 88,
            "img_low": 50,
            "img_low_2": 55,
            "img_failed": 0,
            "img_revise": 80,
            "img_reject": 82,
        }
        for image_id, score in review_scores.items():
            self.store.add(
                "image_reviews",
                {
                    "review_id": f"review_{image_id}",
                    "image_result_id": image_id,
                    "prompt_id": next(item[1] for item in results if item[0] == image_id),
                    "aesthetic": {"status": "success", "score": score, "rating": "test", "reason": "test"},
                    "decision": "reject" if image_id == "img_reject" else "manual_review",
                    "created_at": "2026-01-01T00:00:00",
                },
            )
        self.store.add(
            "prompt_pack_runs",
            {
                "run_id": "validation_split",
                "request_id": "request_web",
                "pack_id": "candidate_validation",
                "image_result_ids": [item[0] for item in results],
                "created_at": "2026-01-01T00:00:00",
            },
        )
        self.store.add(
            "prompt_pack_incubation_runs",
            {
                "incubation_run_id": "inc_review_split",
                "request_id": "request_web",
                "candidate_prompt_ids": prompt_ids,
                "validation_run_id": "validation_split",
                "status": "waiting_human_review",
                "stage": "waiting_human_review",
                "created_at": "2026-01-01T00:00:00",
                "updated_at": "2026-01-01T00:00:00",
            },
        )


if __name__ == "__main__":
    unittest.main()
