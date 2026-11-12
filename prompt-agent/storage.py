from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import DATA_DIR, DATA_FILES


class JsonStore:
    def __init__(self) -> None:
        self.ensure_files()

    def ensure_files(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        for path in DATA_FILES.values():
            if not path.exists():
                self._write_raw(path, [])

    def _write_raw(self, path: Path, data: Any) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = path.with_suffix(path.suffix + ".tmp")
            temp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            temp_path.replace(path)
        except OSError as exc:
            print(f"写入文件失败：{path}，原因：{exc}")

    def load(self, name: str) -> list[dict[str, Any]]:
        path = DATA_FILES[name]
        try:
            if not path.exists():
                self._write_raw(path, [])
                return []
            text = path.read_text(encoding="utf-8").strip()
            if not text:
                return []
            data = json.loads(text)
            return data if isinstance(data, list) else []
        except (OSError, json.JSONDecodeError) as exc:
            print(f"读取 JSON 失败：{path}，原因：{exc}")
            backup = path.with_suffix(path.suffix + ".broken")
            try:
                path.replace(backup)
                self._write_raw(path, [])
                print(f"已备份损坏文件到：{backup}")
            except OSError:
                pass
            return []

    def save(self, name: str, data: list[dict[str, Any]]) -> None:
        self._write_raw(DATA_FILES[name], data)

    def add(self, name: str, item: dict[str, Any]) -> dict[str, Any]:
        data = self.load(name)
        data.append(item)
        self.save(name, data)
        return item

    def update(self, name: str, key: str, value: str, updates: dict[str, Any]) -> bool:
        data = self.load(name)
        changed = False
        for item in data:
            if item.get(key) == value:
                item.update(updates)
                changed = True
                break
        if changed:
            self.save(name, data)
        return changed

    def filter(self, name: str, **conditions: str) -> list[dict[str, Any]]:
        data = self.load(name)
        return [
            item
            for item in data
            if all(str(item.get(key, "")) == str(value) for key, value in conditions.items())
        ]

    def get(self, name: str, key: str, value: str) -> dict[str, Any] | None:
        for item in self.load(name):
            if item.get(key) == value:
                return item
        return None
