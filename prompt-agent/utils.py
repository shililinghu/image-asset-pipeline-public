from __future__ import annotations

import json
import re
import sys
import textwrap
import uuid
from datetime import datetime
from typing import Any


def ensure_console_utf8() -> None:
    for stream_name in ("stdin", "stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def print_box(title: str, body: str = "") -> None:
    line = "=" * 70
    print(f"\n{line}\n{title}\n{line}")
    if body:
        print(body)


def prompt_required(label: str) -> str:
    while True:
        value = input(f"{label}: ").strip()
        if value:
            return value
        print("此项不能为空，请重新输入。")


def prompt_optional(label: str, default: str = "") -> str:
    suffix = f"（默认：{default}）" if default else ""
    value = input(f"{label}{suffix}: ").strip()
    return value or default


def prompt_multiline(label: str) -> str:
    print(f"{label}（可多行输入，单独输入 END 结束）：")
    lines: list[str] = []
    while True:
        line = input()
        if line.strip().upper() == "END":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def choose_from_list(items: list[dict[str, Any]], label_key: str, title: str, id_key: str | None = None) -> dict[str, Any] | None:
    if not items:
        print("暂无可选数据。")
        return None
    print_box(title)
    for idx, item in enumerate(items, start=1):
        chosen_id_key = id_key or detect_id_key(item)
        item_id = item.get(chosen_id_key, "") if chosen_id_key else ""
        print(f"{idx}. {item.get(label_key, '未命名')} [{item_id}]")
    while True:
        raw = input("请输入编号（回车取消）: ").strip()
        if not raw:
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(items):
            return items[int(raw) - 1]
        print("编号无效，请重试。")


def detect_id_key(item: dict[str, Any]) -> str:
    for key in ("project_id", "prompt_id", "case_id", "result_id", "score_id", "feedback_id", "iteration_id", "id"):
        if key in item:
            return key
    return "id"


def extract_json_candidate(text: str) -> str | None:
    if not text:
        return None

    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match:
        return fenced_match.group(1).strip()

    for open_char, close_char in (("{", "}"), ("[", "]")):
        start = text.find(open_char)
        end = text.rfind(close_char)
        if start >= 0 and end > start:
            return text[start : end + 1].strip()
    return None


def safe_json_loads(text: str, fallback: Any) -> Any:
    try:
        return json.loads(text)
    except Exception:
        candidate = extract_json_candidate(text)
        if candidate:
            try:
                return json.loads(candidate)
            except Exception:
                return fallback
        return fallback


def wrap(text: str, width: int = 88) -> str:
    return "\n".join(textwrap.wrap(text, width=width, replace_whitespace=False)) if text else ""


def clamp_score(value: str) -> float:
    try:
        score = float(value)
    except ValueError:
        return 0.0
    return max(0.0, min(10.0, score))


def safe_filename(name: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\n\r\t]+', "_", name).strip(" .")
    return cleaned or "prompt_export"
