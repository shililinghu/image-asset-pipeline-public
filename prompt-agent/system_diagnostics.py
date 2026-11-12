from __future__ import annotations

import importlib
import json
import os
import platform
import socket
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import config
from aesthetic_review_client import AestheticReviewClient
from image_generation_client import ImageGenerationClient
from settings_manager import (
    get_effective_aesthetic_review_config,
    get_effective_image_provider_config,
    load_settings,
    secret_status,
)
from utils import now_iso


def check_ports(web_host: str = "127.0.0.1", web_port: int = 7860) -> dict[str, Any]:
    settings = load_settings()
    aesthetic_url = str(settings.get("aesthetic_review_api_url", "") or os.getenv("AESTHETIC_REVIEW_API_URL", ""))
    return {
        "web_console": {
            "host": web_host,
            "port": web_port,
            "status": "running" if _tcp_connects(web_host, web_port) else "not_running",
        },
        "tastelens": _check_url_port(aesthetic_url),
    }


def run_diagnostics(
    *,
    include_real_image_test: bool = False,
    include_aesthetic_test: bool = False,
    web_host: str = "127.0.0.1",
    web_port: int = 7860,
) -> dict[str, Any]:
    checks: list[dict[str, str]] = []
    checks.append(_check("Python", "pass", platform.python_version(), ""))
    checks.append(_check("工作目录", "pass", str(config.BASE_DIR), ""))
    checks.extend(_check_directories())
    checks.extend(_check_json_files())
    checks.extend(_check_imports())
    checks.extend(_check_provider_profiles())
    checks.extend(_check_image_provider(include_real_image_test))
    checks.extend(_check_aesthetic_provider(include_aesthetic_test))
    checks.extend(_check_web_files(web_host, web_port))
    checks.extend(_check_logs_dir())
    overall = _overall_status(checks)
    payload = {
        "ok": overall != "failed",
        "overall_status": overall,
        "checks": checks,
        "created_at": now_iso(),
    }
    _write_diagnostics_log(payload)
    return payload


def _check_directories() -> list[dict[str, str]]:
    rows = []
    for name, path in {
        "data/": config.DATA_DIR,
        "exports/": config.BASE_DIR / "exports",
        "provider_profiles/": config.BASE_DIR / "provider_profiles",
        "input_reference_images/": config.BASE_DIR / "input_reference_images",
        "logs/": config.LOG_DIR,
        "backups/": config.BACKUP_DIR,
    }.items():
        rows.append(_check(name, "pass" if path.exists() else "failed", str(path), "请创建该目录后重试" if not path.exists() else ""))
    return rows


def _check_json_files() -> list[dict[str, str]]:
    rows = []
    for name in ["prompt_packs", "prompt_pack_items", "image_results", "image_reviews", "idea_requests"]:
        path = config.DATA_FILES.get(name)
        if not path:
            rows.append(_check(name, "failed", "配置缺失", "请检查 config.DATA_FILES"))
            continue
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_text("[]", encoding="utf-8")
            json.loads(path.read_text(encoding="utf-8") or "[]")
            rows.append(_check(str(path.name), "pass", "可读写", ""))
        except Exception as exc:
            rows.append(_check(str(path.name), "failed", str(exc), "请修复 JSON 文件或恢复备份"))
    return rows


def _check_imports() -> list[dict[str, str]]:
    rows = []
    for name in ["json", "pathlib", "requests", "PIL"]:
        try:
            importlib.import_module(name)
            rows.append(_check(f"import {name}", "pass", "可导入", ""))
        except ImportError:
            severity = "warning" if name in {"requests", "PIL"} else "failed"
            rows.append(_check(f"import {name}", severity, "未安装", f"如需该能力，请安装 {name}"))
    return rows


def _check_provider_profiles() -> list[dict[str, str]]:
    rows = []
    profile_dir = config.BASE_DIR / "provider_profiles"
    profiles = list(profile_dir.glob("*.json")) if profile_dir.exists() else []
    if not profiles:
        return [_check("Provider Profiles", "warning", "未找到 provider profile", "可先使用 mock provider")]
    for path in profiles:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            provider = payload.get("provider")
            rows.append(_check(path.name, "pass" if provider else "warning", f"provider={provider or 'missing'}", "请补充 provider 字段" if not provider else ""))
        except Exception as exc:
            rows.append(_check(path.name, "failed", str(exc), "请修复该 profile JSON"))
    current = str(load_settings().get("image_provider_profile_path", "") or "")
    if current and not (config.BASE_DIR / current).exists():
        rows.append(_check("当前 Provider Profile", "failed", current, "设置中心中选择的 profile 文件不存在"))
    return rows


def _check_image_provider(include_real_image_test: bool) -> list[dict[str, str]]:
    cfg = get_effective_image_provider_config()
    provider = cfg["provider"]
    rows = [_check("Image Provider", "pass", provider, "")]
    if provider == "mock":
        return rows
    if provider == "wuyin_async":
        if cfg["api_key"]:
            rows.append(_check("WUYIN_API_KEY", "pass", "已配置", ""))
        else:
            rows.append(_check("WUYIN_API_KEY", "failed", "未配置", '请在设置中心填写 WUYIN_API_KEY，或启动前设置 $env:WUYIN_API_KEY = "你的 key"'))
        rows.append(_check("Wuyin Base URL", "pass" if config.WUYIN_BASE_URL else "failed", config.WUYIN_BASE_URL, "请检查 WUYIN_BASE_URL"))
    else:
        key_status = secret_status([str(cfg.get("api_key_env") or "IMAGE_API_KEY")])
        key_name, row = next(iter(key_status.items()))
        rows.append(_check(key_name, "pass" if row["exists"] else "warning", "已配置" if row["exists"] else "未配置", "真实 provider 需要配置 API key"))
    if include_real_image_test:
        try:
            profile_path = str(cfg.get("provider_profile_path") or "")
            client = ImageGenerationClient.from_profile(config.BASE_DIR / profile_path) if profile_path else ImageGenerationClient(provider=provider)
            result = client.generate("diagnostic image, simple abstract gradient", ratio="1:1", metadata={"request_id": "diagnostic"})
            rows.append(_check("真实生图测试", "pass" if result.get("status") == "success" else "failed", result.get("status", ""), result.get("error_message", "")))
        except Exception as exc:
            rows.append(_check("真实生图测试", "failed", str(exc), "请检查 provider、key 和网络"))
    return rows


def _check_aesthetic_provider(include_aesthetic_test: bool) -> list[dict[str, str]]:
    cfg = get_effective_aesthetic_review_config()
    provider = cfg["provider"]
    rows = [_check("Aesthetic Provider", "pass", provider, "")]
    if provider == "mock":
        return rows
    if not cfg["api_url"]:
        rows.append(_check("TasteLens URL", "failed", "未配置", "请在设置中心填写 TasteLens API URL"))
        return rows
    port_status = _check_url_port(cfg["api_url"])
    rows.append(_check("TasteLens 端口", "pass" if port_status["status"] == "running" else "warning", port_status["status"], "请启动 TasteLens 服务" if port_status["status"] != "running" else ""))
    if include_aesthetic_test:
        image_path = _latest_local_image_path()
        if not image_path:
            rows.append(_check("TasteLens 单图测试", "warning", "没有可用本地图片", "请先生成或上传一张图片"))
        else:
            client = AestheticReviewClient(provider="tastelens_http", api_url=cfg["api_url"])
            result = client.review(image_path=image_path)
            rows.append(_check("TasteLens 单图测试", "pass" if result.get("status") == "success" else "failed", str(result.get("aesthetic_score", "")), result.get("error_message", "")))
    return rows


def _check_web_files(web_host: str, web_port: int) -> list[dict[str, str]]:
    rows = [_check("Web Console 端口", "pass" if _tcp_connects(web_host, web_port) else "warning", f"{web_host}:{web_port}", "如未启动 web_server.py 可忽略")]
    for name in ["index.html", "app.js", "style.css"]:
        path = config.BASE_DIR / "web" / name
        rows.append(_check(f"web/{name}", "pass" if path.exists() else "failed", str(path), "请恢复 web 静态文件" if not path.exists() else ""))
    return rows


def _check_logs_dir() -> list[dict[str, str]]:
    try:
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        probe = config.LOG_DIR / ".diagnostics_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return [_check("logs/ 可写", "pass", str(config.LOG_DIR), "")]
    except Exception as exc:
        return [_check("logs/ 可写", "failed", str(exc), "请检查 logs 目录权限")]


def _check_url_port(url: str) -> dict[str, Any]:
    if not url:
        return {"url": "", "port": 0, "status": "unknown", "error_message": "URL 未配置"}
    parsed = urllib.parse.urlparse(url)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    host = parsed.hostname or "127.0.0.1"
    status = "running" if _tcp_connects(host, port) else "not_running"
    return {"url": url, "port": port, "status": status, "error_message": "" if status == "running" else "connection refused"}


def _tcp_connects(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def _latest_local_image_path() -> str:
    for root in [config.REAL_IMAGE_OUTPUT_DIR, config.MOCK_IMAGE_OUTPUT_DIR, config.INPUT_IMAGES_DIR, config.BASE_DIR / "input_reference_images"]:
        if not root.exists():
            continue
        for suffix in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            matches = sorted(root.rglob(suffix), key=lambda path: path.stat().st_mtime, reverse=True)
            if matches:
                return str(matches[0])
    return ""


def _overall_status(checks: list[dict[str, str]]) -> str:
    if any(item["status"] == "failed" for item in checks):
        return "failed"
    if any(item["status"] == "warning" for item in checks):
        return "warning"
    return "pass"


def _check(name: str, status: str, message: str, suggestion: str = "") -> dict[str, str]:
    return {"name": name, "status": status, "message": message, "suggestion": suggestion}


def _write_diagnostics_log(payload: dict[str, Any]) -> None:
    try:
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = now_iso().replace(":", "").replace("-", "").replace("T", "_")
        (config.LOG_DIR / f"diagnostics_{stamp}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass
