from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


SOURCE_DIR = Path(__file__).resolve().parents[1]
RELEASE_DIR = Path(r"C:/ExampleWorkspace/workspace")

ARRAY_DATA_FILES = [
    "idea_requests.json",
    "prompt_packs.json",
    "prompt_pack_items.json",
    "prompt_pack_runs.json",
    "prompt_pack_build_runs.json",
    "prompt_pack_match_logs.json",
    "prompts.json",
    "test_cases.json",
    "test_results.json",
    "scores.json",
    "feedback.json",
    "iterations.json",
    "image_generation_tasks.json",
    "image_results.json",
    "image_reviews.json",
    "human_image_reviews.json",
    "prompt_image_feedback.json",
    "external_image_imports.json",
    "image_prompt_extractions.json",
    "candidate_prompts.json",
    "candidate_prompt_runs.json",
    "prompt_pack_incubation_runs.json",
    "subject_profiles.json",
    "prompt_versions.json",
    "prompt_evolution_runs.json",
    "prompt_version_archives.json",
]

KEEP_DIRS = [
    "data",
    "exports/mock_images",
    "exports/real_images",
    "input_images",
    "input_reference_images",
    "logs",
    "backups",
]

EXCLUDE_DIR_NAMES = {"__pycache__", ".pytest_cache", "venv", ".venv", ".git"}
EXCLUDE_TOP_DIRS = {"exports", "input_images", "input_reference_images", "logs", "backups", "scripts"}
EXCLUDE_FILE_PATTERNS = [".env", "data/secrets.local.json"]


PUBLIC_IMAGE_CLIENT = r'''from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from config import (
    IMAGE_API_BASE_URL,
    IMAGE_API_KEY_ENV,
    IMAGE_DEFAULT_SIZE,
    IMAGE_DOWNLOAD_REMOTE,
    IMAGE_HTTP_AUTH_TYPE,
    IMAGE_HTTP_METHOD,
    IMAGE_HTTP_PAYLOAD_STYLE,
    IMAGE_HTTP_RESPONSE_BASE64_PATH,
    IMAGE_HTTP_RESPONSE_IMAGE_URL_PATH,
    IMAGE_MAX_RETRIES,
    IMAGE_MODEL,
    IMAGE_PROVIDER,
    IMAGE_REQUEST_TIMEOUT,
    MOCK_IMAGE_OUTPUT_DIR,
    REAL_IMAGE_OUTPUT_DIR,
)
from provider_profile_loader import load_provider_profile
from utils import safe_filename


class ImageGenerationClient:
    def __init__(
        self,
        provider: str = IMAGE_PROVIDER,
        model: str = IMAGE_MODEL,
        *,
        api_key_env: str = IMAGE_API_KEY_ENV,
        api_base_url: str = IMAGE_API_BASE_URL,
        timeout: float = IMAGE_REQUEST_TIMEOUT,
        max_retries: int = IMAGE_MAX_RETRIES,
        download_remote: bool = IMAGE_DOWNLOAD_REMOTE,
        http_method: str = IMAGE_HTTP_METHOD,
        auth_type: str = IMAGE_HTTP_AUTH_TYPE,
        response_image_url_path: str = IMAGE_HTTP_RESPONSE_IMAGE_URL_PATH,
        response_base64_path: str = IMAGE_HTTP_RESPONSE_BASE64_PATH,
        payload_style: str = IMAGE_HTTP_PAYLOAD_STYLE,
        default_size: str = IMAGE_DEFAULT_SIZE,
        headers: dict[str, str] | None = None,
        profile_path: str | Path | None = None,
        output_format: str = "",
        quality: str = "",
        include_negative_prompt: bool = True,
        include_response_format: bool = False,
        moderation: str | None = None,
    ) -> None:
        if profile_path:
            profile = load_provider_profile(profile_path)
            provider = str(profile.get("provider", provider))
            model = str(profile.get("model", model))
            api_key_env = str(profile.get("api_key_env", api_key_env))
            api_base_url = str(profile.get("base_url", api_base_url))
            timeout = float(profile.get("timeout", timeout))
            max_retries = int(profile.get("max_retries", max_retries))
            download_remote = bool(profile.get("download_remote", download_remote))
            http_method = str(profile.get("method", http_method))
            auth_type = str(profile.get("auth_type", auth_type))
            response_image_url_path = str(profile.get("response_image_url_path", response_image_url_path))
            response_base64_path = str(profile.get("response_base64_path", response_base64_path))
            payload_style = str(profile.get("payload_style", payload_style))
            default_size = str(profile.get("default_size", default_size))
            headers = dict(profile.get("headers", {}) or {})
            output_format = str(profile.get("output_format", output_format) or "")
            quality = str(profile.get("quality", quality) or "")
            include_negative_prompt = bool(profile.get("include_negative_prompt", include_negative_prompt))
            include_response_format = bool(profile.get("include_response_format", include_response_format))
            moderation_value = profile.get("moderation", moderation)
            moderation = str(moderation_value) if moderation_value else None
        self.provider = provider
        self.model = model
        self.api_key_env = api_key_env
        self.api_base_url = api_base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.download_remote = download_remote
        self.http_method = http_method.upper()
        self.auth_type = auth_type.lower()
        self.response_image_url_path = response_image_url_path
        self.response_base64_path = response_base64_path
        self.payload_style = payload_style
        self.default_size = default_size
        self.headers = headers or {"Content-Type": "application/json"}
        self.output_format = output_format
        self.quality = quality
        self.include_negative_prompt = include_negative_prompt
        self.include_response_format = include_response_format
        self.moderation = moderation

    @classmethod
    def from_profile(cls, profile_path: str | Path) -> "ImageGenerationClient":
        return cls(profile_path=profile_path)

    def generate(self, prompt: str | dict[str, Any], negative_prompt: str | None = None, ratio: str = "3:4", metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        task = self._normalize_input(prompt, negative_prompt, ratio, metadata)
        if self.provider == "mock":
            return self._generate_mock(task)
        if self.provider in {"custom_http", "real_api"}:
            return self._generate_custom_http(task)
        return self._failed(f"Unsupported image provider in public release: {self.provider}", task)

    def _normalize_input(self, prompt: str | dict[str, Any], negative_prompt: str | None, ratio: str, metadata: dict[str, Any] | None) -> dict[str, Any]:
        if isinstance(prompt, dict):
            task = dict(prompt)
            task.setdefault("prompt_content", task.get("submitted_prompt") or task.get("prompt") or "")
            task.setdefault("negative_prompt", negative_prompt or task.get("negative_prompt", ""))
            task.setdefault("ratio", ratio)
            task.setdefault("metadata", metadata or task.get("metadata", {}) or {})
            return task
        return {"prompt_content": prompt, "negative_prompt": negative_prompt or "", "ratio": ratio, "metadata": metadata or {}}

    def _generate_mock(self, task: dict[str, Any]) -> dict[str, Any]:
        metadata = task.get("metadata", {}) or {}
        output_dir = Path(MOCK_IMAGE_OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        name = self._image_filename(metadata, suffix=".txt")
        path = output_dir / name
        path.write_text(task.get("prompt_content", ""), encoding="utf-8")
        return self._success(str(path), "", "mock", "mock-image-model", task, {})

    def _generate_custom_http(self, task: dict[str, Any]) -> dict[str, Any]:
        if not self.api_base_url:
            return self._failed("IMAGE_API_BASE_URL is required for custom_http provider.", task)
        api_key = os.getenv(self.api_key_env, "").strip()
        if not api_key:
            return self._failed(f"Missing API key. Set environment variable {self.api_key_env}.", task)
        payload = self._build_openai_compatible_payload(task)
        headers = dict(self.headers)
        if self.auth_type == "bearer":
            headers["Authorization"] = f"Bearer {api_key}"
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(self.api_base_url, data=data, headers=headers, method=self.http_method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                status_code = int(getattr(response, "status", 200))
                text = response.read().decode("utf-8", errors="replace")
        except Exception as exc:
            return self._failed(f"custom_http request failed: {exc}", task)
        if not (200 <= status_code < 300):
            return self._failed(f"custom_http HTTP {status_code}: {text[:300]}", task)
        try:
            body = json.loads(text)
        except json.JSONDecodeError as exc:
            return self._failed(f"custom_http JSON parse failed: {exc}", task)
        image_url = self._extract_by_dot_path(body, self.response_image_url_path)
        b64 = self._extract_by_dot_path(body, self.response_base64_path)
        summary = {"response_keys": sorted(str(k) for k in body.keys()), "image_url_found": bool(image_url), "base64_found": bool(b64)}
        if isinstance(b64, str) and b64:
            try:
                path = self._save_base64_image(b64, task)
                return self._success(path, "", self.provider, self.model, task, {"raw_response_summary": summary})
            except Exception as exc:
                return self._failed(f"base64 decode failed: {exc}", task, {"raw_response_summary": summary})
        if isinstance(image_url, str) and image_url:
            image_path = ""
            error_message = ""
            if self.download_remote:
                try:
                    image_path = self._download_image(image_url, task)
                except Exception as exc:
                    error_message = f"image generated but download failed: {exc}"
            result = self._success(image_path, image_url, self.provider, self.model, task, {"raw_response_summary": summary})
            result["error_message"] = error_message
            return result
        return self._failed("custom_http response did not include image_url or base64 image.", task, {"raw_response_summary": summary})

    def _build_openai_compatible_payload(self, task: dict[str, Any]) -> dict[str, Any]:
        payload = {"model": self.model, "prompt": task.get("prompt_content", ""), "n": 1, "size": self._size_for_ratio(task.get("ratio", "1:1"))}
        if self.include_negative_prompt and task.get("negative_prompt"):
            payload["negative_prompt"] = task["negative_prompt"]
        if self.output_format:
            payload["output_format"] = self.output_format
        if self.quality:
            payload["quality"] = self.quality
        if self.moderation:
            payload["moderation"] = self.moderation
        if self.include_response_format:
            payload["response_format"] = "b64_json"
        return payload

    def _save_base64_image(self, value: str, task: dict[str, Any]) -> str:
        output_dir = Path(REAL_IMAGE_OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / self._image_filename(task.get("metadata", {}) or {}, suffix=".png")
        path.write_bytes(base64.b64decode(value))
        return str(path)

    def _download_image(self, url: str, task: dict[str, Any]) -> str:
        output_dir = Path(REAL_IMAGE_OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / self._image_filename(task.get("metadata", {}) or {}, suffix=".png")
        with urllib.request.urlopen(url, timeout=self.timeout) as response:
            path.write_bytes(response.read())
        return str(path)

    def _image_filename(self, metadata: dict[str, Any], suffix: str = ".png") -> str:
        parts = [metadata.get("request_id") or "none", metadata.get("pack_id") or "none", metadata.get("prompt_id") or "none", metadata.get("task_id") or "none"]
        return safe_filename("_".join(str(part) for part in parts)) + suffix

    def _success(self, image_path: str, image_url: str, provider: str, model: str, task: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
        return {"image_path": image_path, "image_url": image_url, "provider": provider, "model": model, "status": "success", "error_message": "", "metadata": metadata}

    def _failed(self, message: str, task: dict[str, Any] | None = None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"image_path": "", "image_url": "", "provider": self.provider, "model": self.model, "status": "failed", "error_message": message, "metadata": metadata or {}}

    def _extract_by_dot_path(self, obj: Any, path: str) -> Any:
        if not path:
            return None
        current = obj
        for part in path.split("."):
            if isinstance(current, list):
                try:
                    current = current[int(part)]
                except (ValueError, IndexError):
                    return None
            elif isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        return current

    def _size_for_ratio(self, ratio: str) -> str:
        return {"1:1": "1024x1024", "3:4": "1024x1365", "4:3": "1365x1024", "16:9": "1536x864", "9:16": "864x1536"}.get(str(ratio), self.default_size)
'''


PUBLIC_README = """# Prompt Pack Lab

Prompt Pack Lab is a local HTML console for building, testing, reviewing, and evolving reusable Prompt Packs.

The public release defaults to the `mock` image provider, so you can run the project without any API key.

## What Is Included

- Prompt Pack asset library
- Local HTML console
- Mock image generation
- Custom HTTP provider adapter
- OpenAI GPT Image 2 example provider profile
- TasteLens review integration in mock/default mode
- Prompt Pack incubation workflow
- Prompt version/evolution workflow
- Settings Center and diagnostics

## What Is Not Included

The public release does not include private relay providers or private async image APIs.

## Start Locally

```powershell
cd C:/ExampleWorkspace/workspace
pip install -r requirements.txt
python web_server.py
```

Open:

```text
http://127.0.0.1:7860
```

## Real Image Providers

To connect a real image provider, use:

- `custom_http`
- `provider_profiles/openai_gpt_image_2.example.json`

Copy an example profile to a local profile file, configure endpoint/model/response paths, and set the API key through environment variables or the local Settings Center.

Do not commit:

- `data/secrets.local.json`
- `.env`
- `provider_profiles/*.local.json`

## GitHub Safety

Runtime data, local images, logs, backups, and local secrets are ignored by `.gitignore`.
"""


PUBLIC_GITIGNORE = """# secrets
.env
*.local.json
data/secrets.local.json
provider_profiles/*local*.json
provider_profiles/*.local.json

# runtime data
logs/*
backups/*
exports/real_images/*
exports/mock_images/*
input_images/*
input_reference_images/*

# keep dirs
!logs/.gitkeep
!backups/.gitkeep
!exports/real_images/.gitkeep
!exports/mock_images/.gitkeep
!input_images/.gitkeep
!input_reference_images/.gitkeep

# python
__pycache__/
*.pyc
.pytest_cache/
.venv/
venv/

# node
node_modules/
npm-debug.log*

# system
.DS_Store
Thumbs.db
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Prompt Pack Lab public release copy.")
    parser.add_argument("--force", action="store_true", help="Delete existing release directory before copying.")
    args = parser.parse_args()

    if RELEASE_DIR.exists():
        if not args.force:
            raise SystemExit(f"{RELEASE_DIR} already exists. Re-run with --force to replace it.")
        shutil.rmtree(RELEASE_DIR)

    copy_project()
    clean_runtime_data()
    scrub_private_provider()
    write_public_files()
    print(f"Release copy prepared at: {RELEASE_DIR}")


def copy_project() -> None:
    def ignore(dir_path: str, names: list[str]) -> set[str]:
        current = Path(dir_path)
        ignored: set[str] = set()
        for name in names:
            path = current / name
            rel = path.relative_to(SOURCE_DIR) if path.exists() else Path(name)
            if name in EXCLUDE_DIR_NAMES:
                ignored.add(name)
            elif current == SOURCE_DIR and name in EXCLUDE_TOP_DIRS:
                ignored.add(name)
            elif name == ".env":
                ignored.add(name)
            elif rel.as_posix() == "data/secrets.local.json":
                ignored.add(name)
            elif rel.match("provider_profiles/*.local.json") or rel.match("provider_profiles/*local*.json"):
                ignored.add(name)
        return ignored

    shutil.copytree(SOURCE_DIR, RELEASE_DIR, ignore=ignore)
    for directory in KEEP_DIRS:
        path = RELEASE_DIR / directory
        path.mkdir(parents=True, exist_ok=True)
        (path / ".gitkeep").write_text("", encoding="utf-8")


def clean_runtime_data() -> None:
    data_dir = RELEASE_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in ARRAY_DATA_FILES:
        (data_dir / name).write_text("[]\n", encoding="utf-8")
    (data_dir / "app_settings.json").write_text("{}\n", encoding="utf-8")
    secret = data_dir / "secrets.local.json"
    if secret.exists():
        secret.unlink()


def scrub_private_provider() -> None:
    for path in (RELEASE_DIR / "provider_profiles").glob("*"):
        lower = path.name.lower()
        if any(token in lower for token in ["wuyin", "wuyinkeji"]):
            path.unlink()
    for path in [
        RELEASE_DIR / "docs" / "wuyin_async_provider.md",
        RELEASE_DIR / "tests" / "manual_wuyin_async_test.py",
        RELEASE_DIR / "tests" / "test_image_generation_client_provider.py",
        RELEASE_DIR / "tests" / "test_provider_profile_loader.py",
    ]:
        if path.exists():
            path.unlink()

    (RELEASE_DIR / "image_generation_client.py").write_text(PUBLIC_IMAGE_CLIENT, encoding="utf-8")
    scrub_config()
    scrub_provider_profile_loader()
    scrub_settings_manager()
    scrub_system_diagnostics()
    scrub_frontend()
    scrub_tests_and_docs()


def scrub_config() -> None:
    path = RELEASE_DIR / "config.py"
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"\nWUYIN_API_KEY_ENV = .*?\nWUYIN_DOWNLOAD_REMOTE = .*?\n", "\n", text, flags=re.S)
    path.write_text(text, encoding="utf-8")


def scrub_settings_manager() -> None:
    path = RELEASE_DIR / "settings_manager.py"
    text = path.read_text(encoding="utf-8")
    text = text.replace('SECRET_KEYS = ["WUYIN_API_KEY", "OPENAI_API_KEY", "IMAGE_API_KEY", "TASTELENS_API_KEY"]', 'SECRET_KEYS = ["OPENAI_API_KEY", "IMAGE_API_KEY", "TASTELENS_API_KEY"]')
    text = re.sub(r'key_name = "WUYIN_API_KEY" if provider == "wuyin_async" else "OPENAI_API_KEY" if "openai" in provider else "IMAGE_API_KEY"', 'key_name = "OPENAI_API_KEY" if "openai" in provider else "IMAGE_API_KEY"', text)
    path.write_text(text, encoding="utf-8")


def scrub_provider_profile_loader() -> None:
    path = RELEASE_DIR / "provider_profile_loader.py"
    path.write_text(
        '''from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = ["provider", "method", "base_url", "auth_type", "api_key_env", "model", "payload_style"]


def load_provider_profile(profile_path: str | Path) -> dict[str, Any]:
    path = Path(profile_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    if not path.exists():
        raise FileNotFoundError(f"Provider profile does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Provider profile JSON parse failed: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Provider profile must be a JSON object: {path}")
    return normalize_provider_profile(payload)


def normalize_provider_profile(payload: dict[str, Any]) -> dict[str, Any]:
    profile = dict(payload)
    provider = str(profile.get("provider", "")).strip()
    if provider == "mock":
        profile.setdefault("method", "POST")
        profile.setdefault("base_url", "")
        profile.setdefault("auth_type", "none")
        profile.setdefault("api_key_env", "")
        profile.setdefault("model", "mock-image-model")
        profile.setdefault("payload_style", "mock")
    else:
        missing = [field for field in REQUIRED_FIELDS if not profile.get(field)]
        if missing:
            raise ValueError(f"Provider profile missing required fields: {', '.join(missing)}")
    profile.setdefault("default_size", "1024x1024")
    profile.setdefault("download_remote", True)
    profile.setdefault("response_image_url_path", "data.0.url")
    profile.setdefault("response_base64_path", "data.0.b64_json")
    profile.setdefault("timeout", 60)
    profile.setdefault("max_retries", 2)
    profile.setdefault("headers", {"Content-Type": "application/json"})
    profile.setdefault("output_format", "png")
    profile.setdefault("quality", "medium")
    profile.setdefault("include_negative_prompt", True)
    profile.setdefault("include_response_format", False)
    profile.setdefault("moderation", None)
    if profile.get("api_key_env"):
        profile["api_key_env"] = os.getenv("IMAGE_API_KEY_ENV", str(profile.get("api_key_env")).strip())
    return profile
''',
        encoding="utf-8",
    )


def scrub_system_diagnostics() -> None:
    path = RELEASE_DIR / "system_diagnostics.py"
    text = path.read_text(encoding="utf-8")
    text = re.sub(
        r'    if provider == "wuyin_async":\n.*?    if include_real_image_test:',
        "    if include_real_image_test:",
        text,
        flags=re.S,
    )
    path.write_text(text, encoding="utf-8")


def scrub_frontend() -> None:
    index = RELEASE_DIR / "web" / "index.html"
    text = index.read_text(encoding="utf-8")
    text = re.sub(r'\s*<option value="wuyin_async">.*?</option>', "", text)
    text = text.replace("wuyin_image_gpt / gpt-image-2", "gpt-image-2 / custom model")
    index.write_text(text, encoding="utf-8")

    app = RELEASE_DIR / "web" / "app.js"
    text = app.read_text(encoding="utf-8")
    text = text.replace('imageProvider === "wuyin_async" ? "WUYIN_API_KEY" : imageProvider === "openai_gpt_image_2" ? "OPENAI_API_KEY" : "IMAGE_API_KEY"', 'imageProvider === "openai_gpt_image_2" ? "OPENAI_API_KEY" : "IMAGE_API_KEY"')
    text = text.replace('provider === "wuyin_async" ? "WUYIN_API_KEY" : provider === "openai_gpt_image_2" ? "OPENAI_API_KEY" : "IMAGE_API_KEY"', 'provider === "openai_gpt_image_2" ? "OPENAI_API_KEY" : "IMAGE_API_KEY"')
    app.write_text(text, encoding="utf-8")


def scrub_tests_and_docs() -> None:
    replacements = {
        "wuyin_async": "custom_http",
        "wuyin": "custom_http",
        "wuyinkeji": "example-provider",
        "WUYIN_API_KEY": "IMAGE_API_KEY",
        "速创": "私有",
        "sk-test": "key-test",
        "sk-****abcd": "key-****abcd",
    }
    for root in [RELEASE_DIR / "tests", RELEASE_DIR / "docs"]:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".py", ".md"}:
                continue
            text = path.read_text(encoding="utf-8")
            for old, new in replacements.items():
                text = text.replace(old, new)
            path.write_text(text, encoding="utf-8")


def write_public_files() -> None:
    (RELEASE_DIR / ".gitignore").write_text(PUBLIC_GITIGNORE, encoding="utf-8")
    (RELEASE_DIR / "README.md").write_text(PUBLIC_README, encoding="utf-8")


if __name__ == "__main__":
    main()
