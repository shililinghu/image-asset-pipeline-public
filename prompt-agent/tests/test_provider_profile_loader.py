from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provider_profile_loader import ProviderProfileError, load_provider_profile


class ProviderProfileLoaderTest(unittest.TestCase):
    def test_load_valid_profile(self) -> None:
        profile = load_provider_profile(ROOT / "provider_profiles" / "custom_http_openai_compatible.example.json")
        self.assertEqual(profile["provider"], "custom_http")
        self.assertEqual(profile["method"], "POST")
        self.assertEqual(profile["payload_style"], "openai_compatible")
        self.assertEqual(profile["api_key_env"], "IMAGE_API_KEY")
        self.assertEqual(profile["output_format"], "png")
        self.assertEqual(profile["quality"], "medium")
        self.assertTrue(profile["include_negative_prompt"])
        self.assertFalse(profile["include_response_format"])
        self.assertIsNone(profile["moderation"])

    def test_load_openai_gpt_image_2_profile(self) -> None:
        profile = load_provider_profile(ROOT / "provider_profiles" / "openai_gpt_image_2.example.json")
        self.assertEqual(profile["provider"], "custom_http")
        self.assertEqual(profile["api_key_env"], "OPENAI_API_KEY")
        self.assertEqual(profile["model"], "gpt-image-2")
        self.assertEqual(profile["response_base64_path"], "data.0.b64_json")
        self.assertEqual(profile["output_format"], "png")
        self.assertEqual(profile["quality"], "medium")
        self.assertFalse(profile["include_negative_prompt"])
        self.assertFalse(profile["include_response_format"])

    def test_load_wuyin_async_profile(self) -> None:
        profile = load_provider_profile(ROOT / "provider_profiles" / "wuyin_async.example.json")
        self.assertEqual(profile["provider"], "wuyin_async")
        self.assertEqual(profile["api_key_env"], "WUYIN_API_KEY")
        self.assertEqual(profile["model"], "wuyin_image_gpt")
        self.assertEqual(profile["submit_path"], "/image_gpt")
        self.assertEqual(profile["detail_path"], "/detail")
        self.assertEqual(profile["size"], "auto")
        self.assertEqual(profile["poll_interval"], 2)
        self.assertEqual(profile["max_poll_seconds"], 180)

    def test_missing_file_error_is_clear(self) -> None:
        with self.assertRaisesRegex(ProviderProfileError, "not found"):
            load_provider_profile(ROOT / "provider_profiles" / "missing.json")

    def test_missing_required_field_error_is_clear(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps({"provider": "custom_http"}), encoding="utf-8")
            with self.assertRaisesRegex(ProviderProfileError, "missing required fields"):
                load_provider_profile(path)

    def test_invalid_json_error_is_clear(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("{bad json", encoding="utf-8")
            with self.assertRaisesRegex(ProviderProfileError, "JSON parse failed"):
                load_provider_profile(path)


if __name__ == "__main__":
    unittest.main()
