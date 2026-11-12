from __future__ import annotations

import json
import socket
from typing import Any

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    DEEPSEEK_SUPPORTED_MODELS,
    LLM_ERROR_LOG,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    MANUAL_MODE,
)
from utils import extract_json_candidate, now_iso


class LLMClient:
    def __init__(self) -> None:
        self.provider = LLM_PROVIDER
        self.model = DEEPSEEK_MODEL
        self.base_url = DEEPSEEK_BASE_URL
        self.timeout = LLM_TIMEOUT
        self.default_temperature = LLM_TEMPERATURE
        self.manual_mode = MANUAL_MODE
        self.api_key = DEEPSEEK_API_KEY
        self.client: Any | None = None

        if self.manual_mode or self.provider != "deepseek":
            return

        if self.model not in DEEPSEEK_SUPPORTED_MODELS:
            message = f"当前模型 {self.model} 不在支持列表中，将回退为 {DEEPSEEK_SUPPORTED_MODELS[0]}。"
            self._log_error("init", ValueError(message))
            print(message)
            self.model = DEEPSEEK_SUPPORTED_MODELS[0]

        if not self._is_ascii_text(self.api_key):
            message = "检测到 DEEPSEEK_API_KEY 含有非 ASCII 字符，可能仍是示例值或粘贴错误。"
            self._log_error("init", ValueError(message))
            print(f"{message} 程序将进入手动模式。")
            self.manual_mode = True
            return

        try:
            from openai import OpenAI

            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
        except Exception as exc:
            self._log_error("init", exc)
            print(f"DeepSeek 客户端初始化失败，将进入手动模式。原因：{self.explain_error(exc)}")
            self.client = None
            self.manual_mode = True

    @property
    def available(self) -> bool:
        return self.is_available()

    def is_available(self) -> bool:
        return bool(self.client) and not self.manual_mode

    def set_model(self, model_name: str) -> bool:
        if model_name not in DEEPSEEK_SUPPORTED_MODELS:
            return False
        self.model = model_name
        return True

    def generate_text(self, system_prompt: str, user_prompt: str, temperature: float | None = None) -> str:
        if not self.is_available():
            raise RuntimeError("当前未配置可用的大模型 API，将使用手动模式。")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.default_temperature if temperature is None else temperature,
            )
            message = response.choices[0].message
            content = getattr(message, "content", "") or ""
            if isinstance(content, list):
                parts = []
                for item in content:
                    text = getattr(item, "text", None)
                    if text:
                        parts.append(text)
                    elif isinstance(item, dict) and item.get("text"):
                        parts.append(str(item["text"]))
                content = "\n".join(parts).strip()
            if isinstance(content, str) and content.strip():
                return content.strip()
            raise RuntimeError("模型返回为空，未提取到文本内容。")
        except Exception as exc:
            self._log_error("generate_text", exc)
            raise RuntimeError(self.explain_error(exc)) from exc

    def generate_json(self, system_prompt: str, user_prompt: str, temperature: float | None = None) -> dict[str, Any]:
        try:
            raw_text = self.generate_text(system_prompt, user_prompt, temperature=temperature)
        except Exception as exc:
            return {"raw_text": "", "parse_error": str(exc)}

        try:
            parsed = json.loads(raw_text)
            return parsed if isinstance(parsed, dict) else {"data": parsed, "raw_text": raw_text}
        except Exception as exc:
            candidate = extract_json_candidate(raw_text)
            if candidate:
                try:
                    parsed = json.loads(candidate)
                    return parsed if isinstance(parsed, dict) else {"data": parsed, "raw_text": raw_text}
                except Exception as second_exc:
                    return {"raw_text": raw_text, "parse_error": f"{exc}; {second_exc}"}
            return {"raw_text": raw_text, "parse_error": str(exc)}

    def health_check(self) -> bool:
        if not self.is_available():
            return False
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Reply with OK only."}],
                temperature=0,
            )
            content = response.choices[0].message.content or ""
            return isinstance(content, str) and bool(content.strip())
        except Exception as exc:
            self._log_error("health_check", exc)
            return False

    def explain_error(self, error: Exception) -> str:
        message = str(error)
        lowered = message.lower()
        if not self.api_key:
            return "没有检测到 DEEPSEEK_API_KEY。"
        if isinstance(error, UnicodeEncodeError):
            return "DEEPSEEK_API_KEY 或请求头中包含非 ASCII 字符，请检查是否误填了中文示例值。"
        if "401" in lowered or "invalid api key" in lowered or "incorrect api key" in lowered:
            return "API Key 无效或已失效，请检查 DEEPSEEK_API_KEY。"
        if "403" in lowered or "permission" in lowered:
            return "当前账号没有调用该模型的权限。"
        if "404" in lowered or ("model" in lowered and "not found" in lowered):
            return "模型名无效或当前模型不可用，请检查 DEEPSEEK_MODEL。"
        if "timeout" in lowered:
            return "请求超时，请检查网络或适当增大 LLM_TIMEOUT。"
        if isinstance(error, (socket.gaierror, ConnectionError)) or "connection" in lowered or "network" in lowered:
            return "网络连接失败，请检查网络环境。"
        if "rate limit" in lowered or "quota" in lowered or "insufficient" in lowered:
            return "请求被限流，或账户额度不足。"
        return f"API 调用失败：{message}"

    def _log_error(self, function_name: str, error: Exception) -> None:
        try:
            LLM_ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "time": now_iso(),
                "provider": self.provider,
                "model": self.model,
                "function": function_name,
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
            with LLM_ERROR_LOG.open("a", encoding="utf-8") as file:
                file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _is_ascii_text(self, value: str) -> bool:
        try:
            value.encode("ascii")
            return True
        except UnicodeEncodeError:
            return False
