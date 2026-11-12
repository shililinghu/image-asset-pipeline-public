from __future__ import annotations

import os
import sys
import tempfile
import unittest
import json
from base64 import b64encode
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
import image_generation_client
import storage
from image_generation_client import ImageGenerationClient
from image_generation_runner import create_tasks_from_prompts, run_image_generation_tasks
from prompt_pack_registry import PromptPackRegistry
from prompt_pack_runner import run_prompt_pack
from storage import JsonStore


class ImageGenerationProviderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        data_dir = Path(self.tmp.name) / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        files = {name: data_dir / f"{name}.json" for name in config.DATA_FILES}
        config.DATA_FILES.clear()
        config.DATA_FILES.update(files)
        storage.DATA_FILES = config.DATA_FILES
        storage.DATA_DIR = data_dir
        image_generation_client.IMAGE_OUTPUT_DIR = Path(self.tmp.name) / "mock_images"
        image_generation_client.MOCK_IMAGE_OUTPUT_DIR = Path(self.tmp.name) / "mock_images"
        image_generation_client.REAL_IMAGE_OUTPUT_DIR = Path(self.tmp.name) / "real_images"
        self.store = JsonStore()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_mock_provider_returns_stable_fields(self) -> None:
        client = ImageGenerationClient(provider="mock", model="mock-image-model")
        result = client.generate(
            "clean chemistry cover",
            negative_prompt="messy",
            ratio="3:4",
            metadata={"request_id": "r1", "pack_id": "pack1", "prompt_id": "p1", "task_id": "t1"},
        )
        self.assertEqual(result["status"], "success")
        for field in ["image_path", "image_url", "provider", "model", "status", "error_message", "metadata"]:
            self.assertIn(field, result)
        self.assertTrue(Path(result["image_path"]).exists())
        self.assertIn("mock_images", result["image_path"])

    def test_real_provider_without_api_key_fails_cleanly(self) -> None:
        os.environ.pop("IMAGE_API_KEY", None)
        client = ImageGenerationClient(provider="real_api", model="real-model", api_key_env="IMAGE_API_KEY")
        result = client.generate("prompt", metadata={"task_id": "t1"})
        self.assertEqual(result["status"], "failed")
        self.assertIn("Missing API key", result["error_message"])

    def test_custom_http_without_api_key_fails_cleanly(self) -> None:
        os.environ.pop("IMAGE_API_KEY", None)
        client = ImageGenerationClient(provider="custom_http", api_key_env="IMAGE_API_KEY", api_base_url="https://example.test/images")
        result = client.generate("prompt", metadata={"task_id": "t1"})
        self.assertEqual(result["status"], "failed")
        self.assertIn("Missing API key", result["error_message"])

    def test_wuyin_async_without_api_key_fails_cleanly(self) -> None:
        os.environ.pop("WUYIN_API_KEY", None)
        client = ImageGenerationClient(provider="wuyin_async")
        result = client.generate("prompt", metadata={"task_id": "t1"})
        self.assertEqual(result["status"], "failed")
        self.assertIn("Missing API key", result["error_message"])
        self.assertEqual(result["provider"], "wuyin_async")

    def test_client_can_initialize_from_profile(self) -> None:
        client = ImageGenerationClient.from_profile(ROOT / "provider_profiles" / "custom_http_openai_compatible.example.json")
        self.assertEqual(client.provider, "custom_http")
        self.assertEqual(client.model, "your-image-model")
        self.assertEqual(client.api_base_url, "https://provider.example.com/v1/images")
        self.assertEqual(client.response_image_url_path, "data.0.url")

    def test_wuyin_async_can_initialize_from_profile(self) -> None:
        client = ImageGenerationClient.from_profile(ROOT / "provider_profiles" / "wuyin_async.example.json")
        self.assertEqual(client.provider, "wuyin_async")
        self.assertEqual(client.model, "wuyin_image_gpt")
        self.assertEqual(client.api_key_env, "WUYIN_API_KEY")
        self.assertEqual(client.wuyin_size, "auto")

    def test_openai_gpt_image_2_profile_builds_expected_payload(self) -> None:
        client = ImageGenerationClient.from_profile(ROOT / "provider_profiles" / "openai_gpt_image_2.example.json")
        payload = client._build_custom_http_payload(
            {
                "prompt_content": "clean chemistry cover",
                "negative_prompt": "messy text",
                "ratio": "1:1",
            }
        )
        self.assertEqual(payload["model"], "gpt-image-2")
        self.assertEqual(payload["prompt"], "clean chemistry cover")
        self.assertEqual(payload["n"], 1)
        self.assertEqual(payload["size"], "1024x1024")
        self.assertEqual(payload["output_format"], "png")
        self.assertEqual(payload["quality"], "medium")
        self.assertNotIn("negative_prompt", payload)
        self.assertNotIn("response_format", payload)

    def test_custom_http_without_base_url_fails_cleanly(self) -> None:
        os.environ["IMAGE_API_KEY"] = "test-key"
        client = ImageGenerationClient(provider="custom_http", api_key_env="IMAGE_API_KEY", api_base_url="")
        result = client.generate("prompt", metadata={"task_id": "t1"})
        self.assertEqual(result["status"], "failed")
        self.assertIn("Missing IMAGE_API_BASE_URL", result["error_message"])

    def test_response_path_parser(self) -> None:
        client = ImageGenerationClient(provider="custom_http")
        payload = {"data": [{"url": "https://example.test/image.png", "b64_json": "abc"}]}
        self.assertEqual(client._get_path(payload, "data.0.url"), "https://example.test/image.png")
        self.assertEqual(client._get_path(payload, "data.0.b64_json"), "abc")
        self.assertIsNone(client._get_path(payload, "data.1.url"))
        self.assertIsNone(client._get_path(payload, ""))

    def test_custom_http_base64_image_saved(self) -> None:
        os.environ["IMAGE_API_KEY"] = "test-key"
        png_bytes = b"\x89PNG\r\n\x1a\nmock"
        b64_json = b64encode(png_bytes).decode("ascii")
        client = ImageGenerationClient(
            provider="custom_http",
            model="image-model",
            api_key_env="IMAGE_API_KEY",
            api_base_url="https://example.test/images",
            response_base64_path="data.0.b64_json",
            response_image_url_path="data.0.url",
            max_retries=0,
        )
        with patch.object(client, "_request_json", return_value={"data": [{"b64_json": b64_json}]}):
            result = client.generate(
                "prompt",
                ratio="3:4",
                metadata={"request_id": "r1", "pack_id": "pack1", "prompt_id": "p1", "task_id": "t1"},
            )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["provider"], "custom_http")
        self.assertTrue(Path(result["image_path"]).exists())
        self.assertIn("real_images", result["image_path"])
        self.assertEqual(Path(result["image_path"]).read_bytes(), png_bytes)
        self.assertTrue(result["metadata"]["raw_response_summary"]["base64_found"])

    def test_openai_gpt_image_2_profile_saves_base64_png(self) -> None:
        os.environ["OPENAI_API_KEY"] = "test-key"
        png_bytes = b"\x89PNG\r\n\x1a\nopenai"
        b64_json = b64encode(png_bytes).decode("ascii")
        client = ImageGenerationClient.from_profile(ROOT / "provider_profiles" / "openai_gpt_image_2.example.json")
        with patch.object(client, "_request_json", return_value={"data": [{"b64_json": b64_json}]}):
            result = client.generate(
                "prompt",
                negative_prompt="should not be sent",
                ratio="1:1",
                metadata={"request_id": "r1", "pack_id": "pack1", "prompt_id": "p1", "task_id": "t1"},
            )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["provider"], "custom_http")
        self.assertEqual(result["model"], "gpt-image-2")
        self.assertEqual(result["image_url"], "")
        self.assertTrue(result["image_path"].endswith(".png"))
        self.assertIn("real_images", result["image_path"])
        self.assertEqual(Path(result["image_path"]).read_bytes(), png_bytes)
        summary = result["metadata"]["raw_response_summary"]
        self.assertTrue(summary["base64_found"])
        self.assertNotIn("b64_json", json.dumps(summary))

    def test_wuyin_submit_task_returns_provider_task_id(self) -> None:
        client = ImageGenerationClient(provider="wuyin_async", download_remote=False)
        with patch.object(client, "_wuyin_post_json", return_value=(200, {"id": "task_123"})):
            result = client._wuyin_submit_task({"prompt_content": "prompt"}, "test-key")
        self.assertEqual(result["state"], "success")
        self.assertEqual(result["provider_task_id"], "task_123")
        self.assertEqual(result["submit_status_code"], 200)

    def test_wuyin_submit_task_id_extracts_fallback_paths(self) -> None:
        client = ImageGenerationClient(provider="wuyin_async", download_remote=False)
        cases = [
            ({"id": "task_1"}, "task_1"),
            ({"data": {"id": "task_2"}}, "task_2"),
            ({"data": {"task_id": "task_3"}}, "task_3"),
            ({"data": "task_4"}, "task_4"),
            ({"code": 1, "data": {"id": "task_5"}, "msg": "success"}, "task_5"),
        ]
        for response, expected in cases:
            with self.subTest(response=response):
                self.assertEqual(client._extract_wuyin_task_id(response), expected)

    def test_wuyin_submit_missing_task_id_error_has_context(self) -> None:
        client = ImageGenerationClient(provider="wuyin_async", download_remote=False)
        response = {"code": 1, "data": {"url": "not-task-id"}, "msg": "success"}
        with patch.object(client, "_wuyin_post_json", return_value=(200, response)):
            result = client._wuyin_submit_task({"prompt_content": "prompt"}, "test-key")
        self.assertEqual(result["state"], "failed")
        self.assertIn("code=1", result["error_message"])
        self.assertIn("msg=success", result["error_message"])
        self.assertIn("data_type=dict", result["error_message"])
        self.assertIn("data_keys=['url']", result["error_message"])
        summary = result["raw_summary"]
        self.assertFalse(summary["provider_task_id_found"])
        self.assertEqual(summary["code"], 1)
        self.assertEqual(summary["msg"], "success")
        self.assertEqual(summary["data_type"], "dict")
        self.assertEqual(summary["data_keys"], ["url"])

    def test_wuyin_submit_business_failure_uses_msg(self) -> None:
        client = ImageGenerationClient(provider="wuyin_async", download_remote=False)
        with patch.object(client, "_wuyin_post_json", return_value=(200, {"code": 2, "msg": "quota exceeded", "data": {}})):
            result = client._wuyin_submit_task({"prompt_content": "prompt"}, "test-key")
        self.assertEqual(result["state"], "failed")
        self.assertIn("quota exceeded", result["error_message"])

    def test_wuyin_generate_submits_then_polls(self) -> None:
        os.environ["WUYIN_API_KEY"] = "test-key"
        client = ImageGenerationClient(provider="wuyin_async", download_remote=False)
        with patch.object(client, "_wuyin_submit_task", return_value={"state": "success", "provider_task_id": "task_123", "submit_status_code": 200}):
            with patch.object(client, "_wuyin_poll_until_complete", return_value={"state": "success", "image_url": "https://example.test/image.png", "raw_summary": {"code": 1}}):
                result = client.generate("prompt", metadata={"task_id": "local_task"})
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["image_url"], "https://example.test/image.png")
        self.assertEqual(result["metadata"]["provider_task_id"], "task_123")

    def test_wuyin_poll_pending_then_success(self) -> None:
        client = ImageGenerationClient(provider="wuyin_async", wuyin_poll_interval=1, wuyin_max_poll_seconds=3)
        pending = {"state": "pending", "image_url": "", "error_message": "", "raw_summary": {"code": 0}}
        success = {"state": "success", "image_url": "https://example.test/image.png", "error_message": "", "raw_summary": {"code": 1}}
        with patch.object(client, "_wuyin_poll_result", side_effect=[pending, success]):
            with patch("image_generation_client.time.sleep", return_value=None):
                result = client._wuyin_poll_until_complete("task_123", "test-key")
        self.assertEqual(result["state"], "success")
        self.assertEqual(result["image_url"], "https://example.test/image.png")
        self.assertEqual(result["raw_summary"]["poll_attempt"], 2)

    def test_wuyin_poll_code_one_returns_image_url(self) -> None:
        client = ImageGenerationClient(provider="wuyin_async")
        with patch.object(client, "_wuyin_get_json", return_value=(200, {"code": 1, "data": "https://example.test/image.png"})):
            result = client._wuyin_poll_result("task_123", "test-key")
        self.assertEqual(result["state"], "success")
        self.assertEqual(result["image_url"], "https://example.test/image.png")

    def test_wuyin_poll_detail_success_shapes(self) -> None:
        client = ImageGenerationClient(provider="wuyin_async")
        cases = [
            ({"code": 1, "data": "http://x/image.png"}, "http://x/image.png"),
            ({"code": 200, "data": "http://x/image.png"}, "http://x/image.png"),
            ({"code": 200, "data": {"url": "http://x/image.png"}}, "http://x/image.png"),
            ({"code": 200, "data": {"image_url": "http://x/image.png"}}, "http://x/image.png"),
            ({"code": 200, "data": ["http://x/image.png"]}, "http://x/image.png"),
            ({"code": 200, "data": [{"url": "http://x/image.png"}]}, "http://x/image.png"),
        ]
        for response, expected in cases:
            with self.subTest(response=response):
                with patch.object(client, "_wuyin_get_json", return_value=(200, response)):
                    result = client._wuyin_poll_result("task_123", "test-key")
                self.assertEqual(result["state"], "success")
                self.assertEqual(result["image_url"], expected)
                self.assertTrue(result["raw_summary"]["image_url_found"])

    def test_wuyin_poll_nested_processing_state_is_pending(self) -> None:
        client = ImageGenerationClient(provider="wuyin_async")
        response = {"code": 200, "msg": "成功", "data": {"status": "processing", "result": None}}
        with patch.object(client, "_wuyin_get_json", return_value=(200, response)):
            result = client._wuyin_poll_result("task_123", "test-key")
        self.assertEqual(result["state"], "pending")
        self.assertEqual(result["raw_summary"]["task_state"], "processing")
        self.assertEqual(result["raw_summary"]["result_type"], "NoneType")
        self.assertFalse(result["raw_summary"]["image_url_found"])

    def test_wuyin_poll_nested_completed_string_result_succeeds(self) -> None:
        client = ImageGenerationClient(provider="wuyin_async")
        response = {"code": 200, "msg": "成功", "data": {"status": "completed", "result": "http://x/image.png"}}
        with patch.object(client, "_wuyin_get_json", return_value=(200, response)):
            result = client._wuyin_poll_result("task_123", "test-key")
        self.assertEqual(result["state"], "success")
        self.assertEqual(result["image_url"], "http://x/image.png")
        self.assertEqual(result["raw_summary"]["task_state"], "completed")
        self.assertEqual(result["raw_summary"]["result_type"], "str")
        self.assertTrue(result["raw_summary"]["image_url_found"])

    def test_wuyin_poll_nested_completed_dict_result_succeeds(self) -> None:
        client = ImageGenerationClient(provider="wuyin_async")
        response = {"code": 200, "data": {"status": "completed", "result": {"url": "http://x/image.png"}}}
        with patch.object(client, "_wuyin_get_json", return_value=(200, response)):
            result = client._wuyin_poll_result("task_123", "test-key")
        self.assertEqual(result["state"], "success")
        self.assertEqual(result["image_url"], "http://x/image.png")
        self.assertEqual(result["raw_summary"]["result_keys"], ["url"])

    def test_wuyin_poll_nested_completed_list_result_succeeds(self) -> None:
        client = ImageGenerationClient(provider="wuyin_async")
        response = {"code": 200, "data": {"status": "completed", "result": [{"url": "http://x/image.png"}]}}
        with patch.object(client, "_wuyin_get_json", return_value=(200, response)):
            result = client._wuyin_poll_result("task_123", "test-key")
        self.assertEqual(result["state"], "success")
        self.assertEqual(result["image_url"], "http://x/image.png")
        self.assertEqual(result["raw_summary"]["result_list_len"], 1)

    def test_wuyin_poll_status_two_list_result_succeeds(self) -> None:
        client = ImageGenerationClient(provider="wuyin_async")
        response = {"code": 200, "msg": "成功", "data": {"status": 2, "result": ["http://x/image.png"]}}
        with patch.object(client, "_wuyin_get_json", return_value=(200, response)):
            result = client._wuyin_poll_result("task_123", "test-key")
        self.assertEqual(result["state"], "success")
        self.assertEqual(result["image_url"], "http://x/image.png")
        self.assertEqual(result["error_message"], "")
        self.assertEqual(result["raw_summary"]["task_state"], 2)
        self.assertTrue(result["raw_summary"]["image_url_found"])
        self.assertEqual(result["raw_summary"]["classified_state"], "success")
        self.assertEqual(result["raw_summary"]["classification_reason"], "image_url_found")

    def test_wuyin_poll_status_two_dict_result_succeeds(self) -> None:
        client = ImageGenerationClient(provider="wuyin_async")
        response = {"code": 200, "msg": "成功", "data": {"status": 2, "result": [{"url": "http://x/image.png"}]}}
        with patch.object(client, "_wuyin_get_json", return_value=(200, response)):
            result = client._wuyin_poll_result("task_123", "test-key")
        self.assertEqual(result["state"], "success")
        self.assertEqual(result["image_url"], "http://x/image.png")

    def test_wuyin_poll_status_two_without_result_is_pending(self) -> None:
        client = ImageGenerationClient(provider="wuyin_async")
        response = {"code": 200, "msg": "成功", "data": {"status": 2, "result": None}}
        with patch.object(client, "_wuyin_get_json", return_value=(200, response)):
            result = client._wuyin_poll_result("task_123", "test-key")
        self.assertEqual(result["state"], "pending")
        self.assertNotEqual(result["error_message"], "成功")

    def test_wuyin_poll_nested_failed_state_fails(self) -> None:
        client = ImageGenerationClient(provider="wuyin_async")
        response = {"code": 200, "data": {"status": "failed", "message": "generation failed"}}
        with patch.object(client, "_wuyin_get_json", return_value=(200, response)):
            result = client._wuyin_poll_result("task_123", "test-key")
        self.assertEqual(result["state"], "failed")
        self.assertIn("generation failed", result["error_message"])

    def test_wuyin_poll_unknown_state_with_result_succeeds(self) -> None:
        client = ImageGenerationClient(provider="wuyin_async")
        response = {"code": 200, "data": {"status": "unknown", "result": "http://x/image.png"}}
        with patch.object(client, "_wuyin_get_json", return_value=(200, response)):
            result = client._wuyin_poll_result("task_123", "test-key")
        self.assertEqual(result["state"], "success")
        self.assertEqual(result["image_url"], "http://x/image.png")
        self.assertTrue(result["raw_summary"]["unknown_state_with_image_url"])

    def test_wuyin_poll_unknown_state_without_result_is_pending(self) -> None:
        client = ImageGenerationClient(provider="wuyin_async")
        response = {"code": 200, "data": {"status": "unknown", "result": None}}
        with patch.object(client, "_wuyin_get_json", return_value=(200, response)):
            result = client._wuyin_poll_result("task_123", "test-key")
        self.assertEqual(result["state"], "pending")

    def test_wuyin_poll_nested_pending_until_timeout(self) -> None:
        client = ImageGenerationClient(provider="wuyin_async", wuyin_poll_interval=1, wuyin_max_poll_seconds=1)
        pending = {"state": "pending", "image_url": "", "error_message": "", "raw_summary": {"task_state": "processing", "image_url_found": False}}
        with patch.object(client, "_wuyin_poll_result", return_value=pending):
            with patch("image_generation_client.time.sleep", return_value=None):
                result = client._wuyin_poll_until_complete("task_123", "test-key")
        self.assertEqual(result["state"], "failed")
        self.assertEqual(result["error_message"], "wuyin_async timeout")

    def test_wuyin_poll_success_code_without_image_url_fails(self) -> None:
        client = ImageGenerationClient(provider="wuyin_async")
        with patch.object(client, "_wuyin_get_json", return_value=(200, {"code": 200, "data": {}})):
            result = client._wuyin_poll_result("task_123", "test-key")
        self.assertEqual(result["state"], "pending")
        self.assertFalse(result["raw_summary"]["image_url_found"])

    def test_wuyin_poll_unknown_code_with_image_url_succeeds(self) -> None:
        client = ImageGenerationClient(provider="wuyin_async")
        with patch.object(client, "_wuyin_get_json", return_value=(200, {"code": 999, "data": "http://x/image.png"})):
            result = client._wuyin_poll_result("task_123", "test-key")
        self.assertEqual(result["state"], "success")
        self.assertEqual(result["image_url"], "http://x/image.png")
        self.assertTrue(result["raw_summary"]["unknown_code_with_image_url"])

    def test_wuyin_poll_unknown_code_without_image_url_fails_with_context(self) -> None:
        client = ImageGenerationClient(provider="wuyin_async")
        with patch.object(client, "_wuyin_get_json", return_value=(200, {"code": 999, "msg": "unknown", "data": None})):
            result = client._wuyin_poll_result("task_123", "test-key")
        self.assertEqual(result["state"], "failed")
        self.assertIn("code: 999", result["error_message"])
        self.assertIn("msg=unknown", result["error_message"])
        self.assertIn("data_type=NoneType", result["error_message"])
        self.assertFalse(result["raw_summary"]["image_url_found"])

    def test_wuyin_poll_code_two_returns_failed(self) -> None:
        client = ImageGenerationClient(provider="wuyin_async")
        with patch.object(client, "_wuyin_get_json", return_value=(200, {"code": 2, "msg": "bad prompt"})):
            result = client._wuyin_poll_result("task_123", "test-key")
        self.assertEqual(result["state"], "failed")
        self.assertIn("bad prompt", result["error_message"])

    def test_wuyin_poll_outer_code_two_still_failed(self) -> None:
        client = ImageGenerationClient(provider="wuyin_async")
        response = {"code": 2, "msg": "任务失败", "data": {}}
        with patch.object(client, "_wuyin_get_json", return_value=(200, response)):
            result = client._wuyin_poll_result("task_123", "test-key")
        self.assertEqual(result["state"], "failed")
        self.assertIn("任务失败", result["error_message"])

    def test_wuyin_poll_timeout_returns_failed(self) -> None:
        client = ImageGenerationClient(provider="wuyin_async", wuyin_poll_interval=1, wuyin_max_poll_seconds=1)
        pending = {"state": "pending", "image_url": "", "error_message": "", "raw_summary": {"code": 0}}
        with patch.object(client, "_wuyin_poll_result", return_value=pending):
            with patch("image_generation_client.time.sleep", return_value=None):
                result = client._wuyin_poll_until_complete("task_123", "test-key")
        self.assertEqual(result["state"], "failed")
        self.assertEqual(result["error_message"], "wuyin_async timeout")

    def test_runner_continues_when_one_task_fails(self) -> None:
        class FailingClient(ImageGenerationClient):
            def __init__(self) -> None:
                super().__init__(provider="custom_http", model="test-model")
                self.calls = 0

            def generate(self, prompt, negative_prompt=None, ratio="3:4", metadata=None):
                self.calls += 1
                if self.calls == 1:
                    return {
                        "image_path": "",
                        "image_url": "",
                        "provider": self.provider,
                        "model": self.model,
                        "status": "failed",
                        "error_message": "simulated failure",
                        "metadata": metadata or {},
                    }
                return {
                    "image_path": "mock-path",
                    "image_url": "",
                    "provider": self.provider,
                    "model": self.model,
                    "status": "success",
                    "error_message": "",
                    "metadata": metadata or {},
                }

        request = {"request_id": "request_1", "ratio": "3:4"}
        prompts = [
            {"prompt_id": "prompt_1", "full_prompt": "first"},
            {"prompt_id": "prompt_2", "full_prompt": "second"},
        ]
        tasks = create_tasks_from_prompts(request, prompts, store=self.store)
        results = run_image_generation_tasks(tasks, store=self.store, client=FailingClient())
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["status"], "failed")
        self.assertEqual(results[1]["status"], "success")
        self.assertEqual(len(self.store.load("image_results")), 2)
        for result in results:
            for field in ["image_path", "image_url", "provider", "model", "status", "error_message", "metadata"]:
                self.assertIn(field, result)

    def test_runner_without_profile_still_uses_mock(self) -> None:
        request = {"request_id": "request_mock", "ratio": "3:4"}
        prompts = [{"prompt_id": "prompt_mock", "full_prompt": "mock prompt"}]
        tasks = create_tasks_from_prompts(request, prompts, store=self.store)
        results = run_image_generation_tasks(tasks, store=self.store)
        self.assertEqual(results[0]["provider"], "mock")
        self.assertEqual(results[0]["status"], "success")

    def test_runner_with_mock_profile_initializes_client(self) -> None:
        profile_path = self._write_mock_profile()
        request = {"request_id": "request_profile", "ratio": "3:4"}
        prompts = [{"prompt_id": "prompt_profile", "full_prompt": "profile prompt"}]
        tasks = create_tasks_from_prompts(request, prompts, store=self.store)
        results = run_image_generation_tasks(tasks, store=self.store, provider_profile_path=str(profile_path))
        self.assertEqual(results[0]["provider"], "mock")
        self.assertEqual(results[0]["status"], "success")
        self.assertEqual(results[0]["metadata"]["provider_profile_path"], str(profile_path))

    def test_runner_with_missing_profile_writes_failed_result(self) -> None:
        missing_profile = Path(self.tmp.name) / "missing_profile.json"
        request = {"request_id": "request_missing_profile", "ratio": "3:4"}
        prompts = [{"prompt_id": "prompt_missing_profile", "full_prompt": "prompt"}]
        tasks = create_tasks_from_prompts(request, prompts, store=self.store)
        results = run_image_generation_tasks(tasks, store=self.store, provider_profile_path=str(missing_profile))
        self.assertEqual(results[0]["status"], "failed")
        self.assertIn("not found", results[0]["error_message"])
        self.assertEqual(results[0]["metadata"]["provider_profile_path"], str(missing_profile))

    def test_prompt_pack_runner_passes_provider_profile_path(self) -> None:
        profile_path = self._write_mock_profile()
        registry = PromptPackRegistry(self.store)
        pack = registry.create_pack(
            {
                "pack_name": "Mock Profile Pack",
                "domain": "test",
                "platform": "test",
                "use_case": "cover",
                "style": "clean",
                "ratio": "3:4",
                "status": "approved",
            },
            [{"prompt_id": "pack_prompt_1", "prompt_content": "cover for {topic}", "ratio": "3:4"}],
        )
        request = {"request_id": "request_pack_profile", "topic": "topic", "ratio": "3:4"}
        result = run_prompt_pack(pack["pack_id"], request, store=self.store, provider_profile_path=str(profile_path))
        image_result = result["image_results"][0]
        self.assertEqual(image_result["provider"], "mock")
        self.assertEqual(image_result["metadata"]["provider_profile_path"], str(profile_path))

    def _write_mock_profile(self) -> Path:
        path = Path(self.tmp.name) / "mock_profile.json"
        path.write_text(
            json.dumps(
                {
                    "profile_id": "mock_profile",
                    "provider": "mock",
                    "method": "POST",
                    "base_url": "https://example.test/mock",
                    "auth_type": "bearer",
                    "api_key_env": "IMAGE_API_KEY",
                    "model": "mock-image-model",
                    "payload_style": "openai_compatible",
                }
            ),
            encoding="utf-8",
        )
        return path


if __name__ == "__main__":
    unittest.main()
