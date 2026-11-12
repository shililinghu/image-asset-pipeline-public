from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import base64
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
    IMAGE_OUTPUT_DIR,
    IMAGE_PROVIDER,
    IMAGE_REQUEST_TIMEOUT,
    MOCK_IMAGE_OUTPUT_DIR,
    REAL_IMAGE_OUTPUT_DIR,
    WUYIN_API_KEY_ENV,
    WUYIN_BASE_URL,
    WUYIN_DETAIL_PATH,
    WUYIN_DOWNLOAD_REMOTE,
    WUYIN_DETAIL_FAILED_CODES,
    WUYIN_DETAIL_IMAGE_URL_PATH,
    WUYIN_DETAIL_PENDING_CODES,
    WUYIN_DETAIL_RESULT_PATH,
    WUYIN_DETAIL_STATE_PATH,
    WUYIN_DETAIL_SUCCESS_CODES,
    WUYIN_MAX_POLL_SECONDS,
    WUYIN_POLL_INTERVAL,
    WUYIN_SUBMIT_PATH,
    WUYIN_SUBMIT_TASK_ID_PATH,
    WUYIN_TASK_FAILED_STATES,
    WUYIN_TASK_PENDING_STATES,
    WUYIN_TASK_SUCCESS_STATES,
)
from provider_profile_loader import load_provider_profile
from utils import new_id, safe_filename


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
        wuyin_base_url: str = WUYIN_BASE_URL,
        wuyin_submit_path: str = WUYIN_SUBMIT_PATH,
        wuyin_detail_path: str = WUYIN_DETAIL_PATH,
        wuyin_submit_task_id_path: str = WUYIN_SUBMIT_TASK_ID_PATH,
        wuyin_detail_image_url_path: str = WUYIN_DETAIL_IMAGE_URL_PATH,
        wuyin_detail_state_path: str = WUYIN_DETAIL_STATE_PATH,
        wuyin_detail_result_path: str = WUYIN_DETAIL_RESULT_PATH,
        wuyin_size: str = "auto",
        wuyin_poll_interval: float = WUYIN_POLL_INTERVAL,
        wuyin_max_poll_seconds: float = WUYIN_MAX_POLL_SECONDS,
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
            wuyin_base_url = str(profile.get("base_url", wuyin_base_url))
            wuyin_submit_path = str(profile.get("submit_path", wuyin_submit_path))
            wuyin_detail_path = str(profile.get("detail_path", wuyin_detail_path))
            wuyin_submit_task_id_path = str(profile.get("submit_task_id_path", wuyin_submit_task_id_path))
            wuyin_detail_image_url_path = str(profile.get("detail_image_url_path", wuyin_detail_image_url_path))
            wuyin_detail_state_path = str(profile.get("detail_state_path", wuyin_detail_state_path))
            wuyin_detail_result_path = str(profile.get("detail_result_path", wuyin_detail_result_path))
            wuyin_size = str(profile.get("size", wuyin_size) or "auto")
            wuyin_poll_interval = float(profile.get("poll_interval", wuyin_poll_interval) or wuyin_poll_interval)
            wuyin_max_poll_seconds = float(profile.get("max_poll_seconds", wuyin_max_poll_seconds) or wuyin_max_poll_seconds)
        if provider == "wuyin_async" and api_key_env == IMAGE_API_KEY_ENV:
            api_key_env = WUYIN_API_KEY_ENV
        if provider == "wuyin_async" and model == IMAGE_MODEL:
            model = "wuyin_image_gpt"
            download_remote = WUYIN_DOWNLOAD_REMOTE
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
        self.wuyin_base_url = wuyin_base_url
        self.wuyin_submit_path = wuyin_submit_path
        self.wuyin_detail_path = wuyin_detail_path
        self.wuyin_submit_task_id_path = wuyin_submit_task_id_path
        self.wuyin_detail_image_url_path = wuyin_detail_image_url_path
        self.wuyin_detail_state_path = wuyin_detail_state_path
        self.wuyin_detail_result_path = wuyin_detail_result_path
        self.wuyin_size = wuyin_size
        self.wuyin_poll_interval = wuyin_poll_interval
        self.wuyin_max_poll_seconds = wuyin_max_poll_seconds

    @classmethod
    def from_profile(cls, profile_path: str | Path) -> "ImageGenerationClient":
        return cls(profile_path=profile_path)

    def generate(
        self,
        prompt: str | dict[str, Any],
        negative_prompt: str | None = None,
        ratio: str = "3:4",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        task = self._normalize_input(prompt, negative_prompt, ratio, metadata)
        if self.provider == "mock":
            return self._generate_mock(task)
        if self.provider == "real_api":
            return self._generate_real_api(task)
        if self.provider == "custom_http":
            return self._generate_custom_http(task)
        if self.provider == "wuyin_async":
            return self._generate_wuyin_async(task)
        return self._failed(f"Unsupported image provider: {self.provider}", task)

    def _normalize_input(
        self,
        prompt: str | dict[str, Any],
        negative_prompt: str | None,
        ratio: str,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if isinstance(prompt, dict):
            task = dict(prompt)
            task.setdefault("prompt_content", task.get("prompt", ""))
            task.setdefault("negative_prompt", negative_prompt or "")
            task.setdefault("ratio", ratio)
            task.setdefault("metadata", metadata or {})
            return task
        return {
            "prompt_content": prompt,
            "negative_prompt": negative_prompt or "",
            "ratio": ratio,
            "request_id": (metadata or {}).get("request_id", ""),
            "pack_id": (metadata or {}).get("pack_id", ""),
            "prompt_id": (metadata or {}).get("prompt_id", ""),
            "task_id": (metadata or {}).get("task_id", ""),
            "metadata": metadata or {},
        }

    def _generate_mock(self, task: dict[str, Any]) -> dict[str, Any]:
        _mock_output_dir().mkdir(parents=True, exist_ok=True)
        image_id = new_id("image")
        filename = self._local_filename(image_id, task, ".txt")
        image_path = _mock_output_dir() / filename
        payload = {
            "mock_image_id": image_id,
            "task_id": task.get("task_id", ""),
            "request_id": task.get("request_id", ""),
            "pack_id": task.get("pack_id", ""),
            "prompt_id": task.get("prompt_id", ""),
            "ratio": task.get("ratio", ""),
            "prompt_preview": str(task.get("prompt_content", ""))[:500],
            "negative_prompt": task.get("negative_prompt", ""),
            "metadata": task.get("metadata", {}),
        }
        image_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return self._success(task, image_id=image_id, image_path=str(Path(image_path).resolve()), metadata={"mock": True})

    def _generate_real_api(self, task: dict[str, Any]) -> dict[str, Any]:
        api_key = os.getenv(self.api_key_env, "").strip()
        if not api_key:
            return self._failed(f"Missing API key. Set environment variable {self.api_key_env}.", task)
        if not self.api_base_url:
            return self._failed("Missing IMAGE_API_BASE_URL for real_api provider.", task)

        payload = self._build_real_api_payload(task)
        last_error = ""
        for attempt in range(max(1, self.max_retries + 1)):
            try:
                response_data = self._post_json(self.api_base_url, payload, api_key)
                image_url = self._extract_image_url(response_data)
                image_path = ""
                if image_url and self.download_remote:
                    image_path = self._download_image(image_url, task)
                return self._success(
                    task,
                    image_url=image_url,
                    image_path=image_path,
                    metadata={"raw_response": response_data, "attempt": attempt + 1},
                )
            except Exception as exc:
                last_error = str(exc)
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 5))
        return self._failed(last_error or "real_api request failed", task)

    def _generate_custom_http(self, task: dict[str, Any]) -> dict[str, Any]:
        api_key = os.getenv(self.api_key_env, "").strip()
        if not api_key:
            return self._failed(f"Missing API key. Set environment variable {self.api_key_env}.", task)
        if not self.api_base_url:
            return self._failed("Missing IMAGE_API_BASE_URL for custom_http provider.", task)
        if self.http_method != "POST":
            return self._failed(f"Unsupported IMAGE_HTTP_METHOD for custom_http: {self.http_method}", task)

        payload = self._build_custom_http_payload(task)
        last_error = ""
        for attempt in range(max(1, self.max_retries + 1)):
            try:
                response_data = self._request_json(self.api_base_url, payload, api_key, method=self.http_method)
                image_url = self._get_path(response_data, self.response_image_url_path)
                b64_json = self._get_path(response_data, self.response_base64_path)
                image_url = str(image_url) if image_url else ""
                b64_json = str(b64_json) if b64_json else ""
                summary = self._response_summary(response_data, bool(image_url), bool(b64_json), attempt + 1)

                image_path = ""
                if image_url and self.download_remote:
                    image_path = self._download_image(image_url, task)
                elif b64_json:
                    image_path = self._save_base64_image(b64_json, task)

                if image_url or b64_json:
                    return self._success(
                        task,
                        image_url=image_url,
                        image_path=image_path,
                        metadata={"raw_response_summary": summary},
                    )
                return self._failed("custom_http response did not include image_url or base64 image data", task)
            except Exception as exc:
                last_error = str(exc)
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 5))
        return self._failed(last_error or "custom_http request failed", task)

    def _generate_wuyin_async(self, task: dict[str, Any]) -> dict[str, Any]:
        api_key = os.getenv(self.api_key_env, "").strip()
        if not api_key:
            return self._failed(f"Missing API key. Set environment variable {self.api_key_env}.", task)

        submit_result = self._wuyin_submit_task(task, api_key)
        if submit_result.get("state") == "failed":
            return self._failed(
                submit_result.get("error_message", "wuyin_async submit failed"),
                task,
                metadata={"raw_response_summary": submit_result.get("raw_summary", {})},
            )

        provider_task_id = str(submit_result.get("provider_task_id", "") or "")
        poll_result = self._wuyin_poll_until_complete(provider_task_id, api_key)
        metadata = {
            "provider_task_id": provider_task_id,
            "submit_status_code": submit_result.get("submit_status_code", 0),
            "raw_response_summary": poll_result.get("raw_summary", {}),
        }
        if poll_result.get("state") != "success":
            return self._failed(
                poll_result.get("error_message", "wuyin_async failed"),
                task,
                metadata=metadata,
            )

        image_url = str(poll_result.get("image_url", "") or "")
        image_path = ""
        error_message = ""
        if image_url and self.download_remote:
            try:
                image_path = self._download_image(image_url, task)
            except Exception as exc:
                error_message = f"image generated but download failed: {exc}"
        result = self._success(task, image_url=image_url, image_path=image_path, metadata=metadata)
        if error_message:
            result["error_message"] = error_message
        return result

    def _wuyin_submit_task(self, task: dict[str, Any], api_key: str) -> dict[str, Any]:
        url = self._join_url(self.wuyin_base_url, self.wuyin_submit_path)
        payload = {
            "prompt": task.get("prompt_content", ""),
            "size": self.wuyin_size or "auto",
        }
        try:
            status_code, response_data = self._wuyin_post_json(url, payload, api_key)
        except Exception as exc:
            return {
                "state": "failed",
                "error_message": f"wuyin_async submit request failed: {exc}",
                "raw_summary": {"response_keys": [], "submit_status_code": 0},
            }
        raw_summary = {
            "response_keys": sorted(str(key) for key in response_data.keys()),
            "submit_status_code": status_code,
            "provider_task_id_found": False,
            "code": response_data.get("code"),
            "msg": response_data.get("msg"),
            "data_type": type(response_data.get("data")).__name__,
            "data_keys": sorted(str(key) for key in response_data.get("data", {}).keys()) if isinstance(response_data.get("data"), dict) else [],
        }
        if not (200 <= int(status_code) < 300):
            return {
                "state": "failed",
                "error_message": f"wuyin_async submit HTTP {status_code}",
                "raw_summary": raw_summary,
            }
        if self._wuyin_submit_code_is_failure(response_data.get("code")):
            return {
                "state": "failed",
                "error_message": str(response_data.get("msg", "") or f"wuyin_async submit failed with code={response_data.get('code')}"),
                "raw_summary": raw_summary,
            }
        provider_task_id = self._extract_wuyin_task_id(response_data)
        raw_summary["provider_task_id_found"] = bool(provider_task_id)
        if not provider_task_id:
            return {
                "state": "failed",
                "error_message": self._wuyin_missing_task_id_message(response_data),
                "raw_summary": raw_summary,
            }
        return {
            "state": "success",
            "provider_task_id": str(provider_task_id),
            "submit_status_code": status_code,
            "raw_summary": raw_summary,
        }

    def _extract_wuyin_task_id(self, response_data: dict[str, Any]) -> str | None:
        paths = [
            self.wuyin_submit_task_id_path,
            "id",
            "task_id",
            "taskId",
            "data.id",
            "data.task_id",
            "data.taskId",
            "data",
        ]
        seen: set[str] = set()
        for path in paths:
            path = str(path or "").strip()
            if not path or path in seen:
                continue
            seen.add(path)
            value = self._extract_by_dot_path(response_data, path)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if path == "data" and value is not None and not isinstance(value, (dict, list)):
                value_text = str(value).strip()
                if value_text:
                    return value_text
        data = response_data.get("data")
        if isinstance(data, dict):
            for key in ["id", "task_id", "taskId"]:
                value = data.get(key)
                if value:
                    return str(value).strip()
        return None

    def _extract_by_dot_path(self, data: Any, path: str) -> Any:
        if not str(path or "").strip():
            return None
        current = data
        for part in [item for item in str(path).split(".") if item != ""]:
            if isinstance(current, list):
                if not part.isdigit():
                    return None
                index = int(part)
                if index >= len(current):
                    return None
                current = current[index]
            elif isinstance(current, dict):
                if part not in current:
                    return None
                current = current[part]
            else:
                return None
        return current

    def _wuyin_submit_code_is_failure(self, code: Any) -> bool:
        if code is None:
            return False
        code_text = str(code).strip().lower()
        return code_text in {"2", "failed", "fail", "error", "false"}

    def _wuyin_missing_task_id_message(self, response_data: dict[str, Any]) -> str:
        data = response_data.get("data")
        data_keys = sorted(str(key) for key in data.keys()) if isinstance(data, dict) else []
        return (
            "wuyin_async submit response did not include task id. "
            f"code={response_data.get('code')}, "
            f"msg={response_data.get('msg')}, "
            f"data_type={type(data).__name__}, "
            f"data_keys={data_keys}"
        )

    def _wuyin_poll_until_complete(self, provider_task_id: str, api_key: str) -> dict[str, Any]:
        interval = max(0.1, float(self.wuyin_poll_interval or 2))
        max_seconds = max(0.1, float(self.wuyin_max_poll_seconds or 180))
        max_attempts = max(1, int(max_seconds / interval) + 1)
        last_result: dict[str, Any] = {}
        for attempt in range(max_attempts):
            result = self._wuyin_poll_result(provider_task_id, api_key)
            raw_summary = dict(result.get("raw_summary", {}) or {})
            raw_summary["poll_attempt"] = attempt + 1
            result["raw_summary"] = raw_summary
            if result.get("state") != "pending":
                return result
            last_result = result
            if attempt < max_attempts - 1:
                time.sleep(interval)
        raw_summary = dict(last_result.get("raw_summary", {}) or {})
        raw_summary["poll_attempts"] = max_attempts
        return {
            "state": "failed",
            "image_url": "",
            "error_message": "wuyin_async timeout",
            "raw_summary": raw_summary,
        }

    def _wuyin_poll_result(self, provider_task_id: str, api_key: str) -> dict[str, Any]:
        url = self._join_url(self.wuyin_base_url, self.wuyin_detail_path)
        try:
            status_code, response_data = self._wuyin_get_json(
                url,
                {"key": api_key, "id": provider_task_id},
            )
        except Exception as exc:
            return {
                "state": "failed",
                "image_url": "",
                "error_message": f"wuyin_async detail request failed: {exc}",
                "raw_summary": {"response_keys": [], "detail_status_code": 0},
            }
        raw_summary = {
            "response_keys": sorted(str(key) for key in response_data.keys()),
            "detail_status_code": status_code,
            "code": response_data.get("code"),
            "msg": response_data.get("msg"),
            "has_data": bool(response_data.get("data")),
            "data_type": type(response_data.get("data")).__name__,
            "data_keys": sorted(str(key) for key in response_data.get("data", {}).keys()) if isinstance(response_data.get("data"), dict) else [],
            "task_state": self._extract_by_dot_path(response_data, self.wuyin_detail_state_path),
            "result_type": type(self._extract_by_dot_path(response_data, self.wuyin_detail_result_path)).__name__,
            "result_keys": self._wuyin_result_keys(response_data),
            "result_list_len": self._wuyin_result_list_len(response_data),
            "image_url_found": False,
        }
        if not (200 <= int(status_code) < 300):
            return {
                "state": "failed",
                "image_url": "",
                "error_message": f"wuyin_async detail HTTP {status_code}",
                "raw_summary": raw_summary,
            }
        code = response_data.get("code")
        image_url = self._extract_wuyin_image_url(response_data)
        raw_summary["image_url_found"] = bool(image_url)
        if self._code_in(code, WUYIN_DETAIL_PENDING_CODES):
            raw_summary["classified_state"] = "pending"
            raw_summary["classification_reason"] = "outer_code_pending"
            return {"state": "pending", "image_url": "", "error_message": "", "raw_summary": raw_summary}
        if self._code_in(code, WUYIN_DETAIL_FAILED_CODES):
            raw_summary["classified_state"] = "failed"
            raw_summary["classification_reason"] = "outer_code_failed"
            return {
                "state": "failed",
                "image_url": "",
                "error_message": str(response_data.get("msg", "") or "wuyin_async generation failed"),
                "raw_summary": raw_summary,
            }
        task_state = raw_summary["task_state"]
        if image_url and not self._state_in(task_state, WUYIN_TASK_FAILED_STATES):
            raw_summary["classified_state"] = "success"
            raw_summary["classification_reason"] = "image_url_found"
            if not self._code_in(code, WUYIN_DETAIL_SUCCESS_CODES):
                raw_summary["unknown_code_with_image_url"] = True
            if not self._state_in(task_state, WUYIN_TASK_SUCCESS_STATES):
                raw_summary["unknown_state_with_image_url"] = True
            return {"state": "success", "image_url": image_url, "error_message": "", "raw_summary": raw_summary}
        if self._state_in(task_state, WUYIN_TASK_PENDING_STATES):
            raw_summary["classified_state"] = "pending"
            raw_summary["classification_reason"] = "task_state_pending"
            return {"state": "pending", "image_url": "", "error_message": "", "raw_summary": raw_summary}
        if self._state_in(task_state, WUYIN_TASK_FAILED_STATES):
            raw_summary["classified_state"] = "failed"
            raw_summary["classification_reason"] = "task_state_failed"
            return {
                "state": "failed",
                "image_url": "",
                "error_message": self._wuyin_task_failure_message(response_data),
                "raw_summary": raw_summary,
            }
        if self._state_in(task_state, WUYIN_TASK_SUCCESS_STATES):
            raw_summary["classified_state"] = "pending"
            raw_summary["classification_reason"] = "task_state_success_without_image_url"
            return {"state": "pending", "image_url": "", "error_message": "", "raw_summary": raw_summary}
        if self._code_in(code, WUYIN_DETAIL_SUCCESS_CODES):
            raw_summary["classified_state"] = "pending"
            raw_summary["classification_reason"] = "outer_code_success_waiting_for_result"
            return {"state": "pending", "image_url": "", "error_message": "", "raw_summary": raw_summary}
        raw_summary["classified_state"] = "failed"
        raw_summary["classification_reason"] = "unknown_code_without_image_url"
        return {
            "state": "failed",
            "image_url": "",
            "error_message": self._wuyin_unknown_detail_code_message(response_data, bool(image_url)),
            "raw_summary": raw_summary,
        }

    def _extract_wuyin_image_url(self, response_data: dict[str, Any]) -> str | None:
        paths = [
            self.wuyin_detail_image_url_path,
            "data.result",
            "data.result.url",
            "data.result.image_url",
            "data.result.output",
            "data.result.result",
            "data.result.0",
            "data.result.0.url",
            "data.result.0.image_url",
            "data.result.images.0.url",
            "data.result.urls.0",
            "data.result.data.0.url",
            "data.url",
            "data.image_url",
            "data",
            "data.output",
            "data.0",
            "data.0.url",
            "data.0.image_url",
        ]
        seen: set[str] = set()
        for path in paths:
            path = str(path or "").strip()
            if not path or path in seen:
                continue
            seen.add(path)
            value = self._extract_by_dot_path(response_data, path)
            image_url = self._coerce_wuyin_image_url(value)
            if image_url:
                return image_url
        return self._coerce_wuyin_image_url(response_data.get("data"))

    def _coerce_wuyin_image_url(self, value: Any) -> str | None:
        if isinstance(value, str):
            value = value.strip()
            return value or None
        if isinstance(value, list) and value:
            first = value[0]
            if isinstance(first, str):
                return first.strip() or None
            if isinstance(first, dict):
                for key in ["url", "image_url"]:
                    if first.get(key):
                        return str(first[key]).strip()
        if isinstance(value, dict):
            for key in ["url", "image_url", "result", "output", "images", "urls", "data"]:
                candidate = value.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()
                if isinstance(candidate, (list, dict)) and candidate:
                    nested = self._coerce_wuyin_image_url(candidate)
                    if nested:
                        return nested
        return None

    def _code_in(self, code: Any, expected_codes: list[int]) -> bool:
        return str(code).strip() in {str(item) for item in expected_codes}

    def _state_in(self, state: Any, expected_states: list[Any]) -> bool:
        return str(state).strip() in {str(item).strip() for item in expected_states}

    def _wuyin_result_keys(self, response_data: dict[str, Any]) -> list[str]:
        result = self._extract_by_dot_path(response_data, self.wuyin_detail_result_path)
        return sorted(str(key) for key in result.keys()) if isinstance(result, dict) else []

    def _wuyin_result_list_len(self, response_data: dict[str, Any]) -> int:
        result = self._extract_by_dot_path(response_data, self.wuyin_detail_result_path)
        return len(result) if isinstance(result, list) else 0

    def _wuyin_task_failure_message(self, response_data: dict[str, Any]) -> str:
        data = response_data.get("data")
        if isinstance(data, dict):
            for key in ["message", "msg", "error"]:
                if data.get(key):
                    return str(data[key])
        return str(response_data.get("msg", "") or "wuyin_async generation failed")

    def _wuyin_unknown_detail_code_message(self, response_data: dict[str, Any], image_url_found: bool) -> str:
        data = response_data.get("data")
        data_keys = sorted(str(key) for key in data.keys()) if isinstance(data, dict) else []
        return (
            f"wuyin_async unknown detail status code: {response_data.get('code')}, "
            f"msg={response_data.get('msg')}, "
            f"data_type={type(data).__name__}, "
            f"data_keys={data_keys}, "
            f"image_url_found={str(image_url_found).lower()}"
        )

    def _build_real_api_payload(self, task: dict[str, Any]) -> dict[str, Any]:
        # TODO: Adjust endpoint payload for the selected real image provider.
        return {
            "model": self.model,
            "prompt": task.get("prompt_content", ""),
            "negative_prompt": task.get("negative_prompt", ""),
            "ratio": task.get("ratio", "3:4"),
            "metadata": task.get("metadata", {}),
        }

    def _build_custom_http_payload(self, task: dict[str, Any]) -> dict[str, Any]:
        if self.payload_style != "openai_compatible":
            raise ValueError(f"Unsupported IMAGE_HTTP_PAYLOAD_STYLE: {self.payload_style}")
        payload = {
            "model": self.model,
            "prompt": task.get("prompt_content", ""),
            "n": 1,
            "size": self._size_from_ratio(str(task.get("ratio", "1:1"))),
        }
        if self.include_negative_prompt and task.get("negative_prompt", ""):
            payload["negative_prompt"] = task.get("negative_prompt", "")
        if self.output_format:
            payload["output_format"] = self.output_format
        if self.quality:
            payload["quality"] = self.quality
        if self.moderation:
            payload["moderation"] = self.moderation
        if self.include_response_format:
            payload["response_format"] = "b64_json"
        return payload

    def _post_json(self, url: str, payload: dict[str, Any], api_key: str) -> dict[str, Any]:
        return self._request_json(url, payload, api_key, method="POST")

    def _wuyin_post_json(self, url: str, payload: dict[str, Any], api_key: str) -> tuple[int, dict[str, Any]]:
        headers = {
            "Authorization": api_key,
            "Content-Type": "application/json",
        }
        return self._request_json_with_headers(url, payload, headers, method="POST")

    def _wuyin_get_json(self, url: str, params: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        query = urllib.parse.urlencode({key: str(value) for key, value in params.items()})
        separator = "&" if "?" in url else "?"
        request_url = f"{url}{separator}{query}"
        request = urllib.request.Request(request_url, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                status = int(getattr(response, "status", 200))
                text = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            raise RuntimeError(f"HTTP {exc.code}: {body_text[:300]}") from exc
        except TimeoutError as exc:
            raise RuntimeError("HTTP request timed out") from exc
        return status, self._parse_json_response(text)

    def _request_json_with_headers(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        method: str = "POST",
    ) -> tuple[int, dict[str, Any]]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                status = int(getattr(response, "status", 200))
                text = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            raise RuntimeError(f"HTTP {exc.code}: {body_text[:300]}") from exc
        except TimeoutError as exc:
            raise RuntimeError("HTTP request timed out") from exc
        return status, self._parse_json_response(text)

    def _request_json(self, url: str, payload: dict[str, Any], api_key: str, method: str = "POST") -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = dict(self.headers)
        headers.setdefault("Content-Type", "application/json")
        if self.auth_type == "bearer":
            headers["Authorization"] = f"Bearer {api_key}"
        elif self.auth_type == "api_key":
            headers["X-API-Key"] = api_key
        elif self.auth_type in {"none", "no_auth"}:
            pass
        else:
            raise ValueError(f"Unsupported IMAGE_HTTP_AUTH_TYPE: {self.auth_type}")
        request = urllib.request.Request(
            url,
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                status = getattr(response, "status", 200)
                text = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            raise RuntimeError(f"HTTP {exc.code}: {body_text[:300]}") from exc
        except TimeoutError as exc:
            raise RuntimeError("HTTP request timed out") from exc
        if not (200 <= int(status) < 300):
            raise RuntimeError(f"HTTP {status}: {text[:300]}")
        try:
            return self._parse_json_response(text)
        except RuntimeError:
            raise

    def _parse_json_response(self, text: str) -> dict[str, Any]:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Response JSON parse failed: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("image provider response must be a JSON object")
        return parsed

    def _extract_image_url(self, response_data: dict[str, Any]) -> str:
        if response_data.get("image_url"):
            return str(response_data["image_url"])
        images = response_data.get("images")
        if isinstance(images, list) and images:
            first = images[0]
            if isinstance(first, dict) and first.get("url"):
                return str(first["url"])
            if isinstance(first, str):
                return first
        data = response_data.get("data")
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict) and first.get("url"):
                return str(first["url"])
        raise ValueError("real_api response did not include an image URL")

    def _download_image(self, image_url: str, task: dict[str, Any]) -> str:
        output_dir = self._real_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        extension = Path(image_url.split("?", 1)[0]).suffix or ".png"
        filename = self._real_image_filename(task, extension)
        image_path = output_dir / filename
        request = urllib.request.Request(image_url, headers={"User-Agent": "prompt-agent/0.2"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                image_path.write_bytes(response.read())
        except urllib.error.URLError as exc:
            raise RuntimeError(f"failed to download image URL: {exc}") from exc
        return str(Path(image_path).resolve())

    def _save_base64_image(self, b64_json: str, task: dict[str, Any]) -> str:
        output_dir = self._real_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = output_dir / self._real_image_filename(task, ".png")
        try:
            image_path.write_bytes(base64.b64decode(b64_json, validate=True))
        except Exception as exc:
            raise RuntimeError(f"base64 image decode failed: {exc}") from exc
        return str(Path(image_path).resolve())

    def _local_filename(self, image_id: str, task: dict[str, Any], extension: str) -> str:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        stem = f"{image_id}_{task.get('prompt_id', 'prompt')}_{task.get('task_id', 'task')}_{timestamp}"
        return safe_filename(stem) + extension

    def _real_output_dir(self) -> Path:
        return _real_output_dir()

    def _join_url(self, base_url: str, path: str) -> str:
        return str(base_url or "").rstrip("/") + "/" + str(path or "").lstrip("/")

    def _real_image_filename(self, task: dict[str, Any], extension: str) -> str:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        values = [
            task.get("request_id") or "none",
            task.get("pack_id") or "none",
            task.get("prompt_id") or "none",
            task.get("task_id") or "none",
            timestamp,
        ]
        return safe_filename("_".join(str(value) for value in values)) + extension

    def _size_from_ratio(self, ratio: str) -> str:
        return {
            "1:1": "1024x1024",
            "3:4": "1024x1365",
            "4:3": "1365x1024",
            "16:9": "1536x864",
            "9:16": "864x1536",
        }.get(ratio, self.default_size)

    def _get_path(self, data: Any, path: str) -> Any:
        if not str(path or "").strip():
            return None
        current = data
        for part in [item for item in str(path or "").split(".") if item != ""]:
            if isinstance(current, list):
                if not part.isdigit():
                    return None
                index = int(part)
                if index >= len(current):
                    return None
                current = current[index]
            elif isinstance(current, dict):
                if part not in current:
                    return None
                current = current[part]
            else:
                return None
        return current

    def _response_summary(self, response_data: dict[str, Any], image_url_found: bool, base64_found: bool, attempt: int) -> dict[str, Any]:
        return {
            "response_keys": sorted(str(key) for key in response_data.keys()),
            "image_url_found": image_url_found,
            "base64_found": base64_found,
            "attempt": attempt,
        }

    def _success(
        self,
        task: dict[str, Any],
        *,
        image_id: str | None = None,
        image_path: str = "",
        image_url: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "image_result_id": image_id or new_id("image"),
            "image_path": image_path,
            "image_url": image_url,
            "provider": self.provider,
            "model": self.model,
            "status": "success",
            "error_message": "",
            "metadata": metadata or {},
        }

    def _failed(
        self,
        message: str,
        task: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result_metadata = {
            "task_id": (task or {}).get("task_id", ""),
            "raw_response_summary": {
                "response_keys": [],
                "image_url_found": False,
                "base64_found": False,
            },
        }
        result_metadata.update(metadata or {})
        return {
            "image_result_id": new_id("image"),
            "image_path": "",
            "image_url": "",
            "provider": self.provider,
            "model": self.model,
            "status": "failed",
            "error_message": message,
            "metadata": result_metadata,
        }


def _mock_output_dir() -> Path:
    if IMAGE_OUTPUT_DIR != MOCK_IMAGE_OUTPUT_DIR:
        return IMAGE_OUTPUT_DIR
    return MOCK_IMAGE_OUTPUT_DIR


def _real_output_dir() -> Path:
    return REAL_IMAGE_OUTPUT_DIR
