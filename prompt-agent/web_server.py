from __future__ import annotations

import json
import mimetypes
import os
import re
import threading
import zipfile
from io import BytesIO
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import config
from delete_manager import delete_prompt_pack
from image_auto_reviewer import review_image_results
from image_prompt_extractor import extract_image_prompts
from models import create_human_image_review, create_idea_request
from prompt_pack_incubator import (
    build_prompt_pack_from_incubation,
    build_prompt_pack_from_reviewed_candidates,
    generate_candidate_prompts,
    validate_candidate_prompts,
)
from prompt_pack_registry import PromptPackRegistry
from prompt_pack_runner import run_prompt_pack, select_prompt_pack_items
from prompt_type_profiles import apply_prompt_type_defaults, get_prompt_type_profile, load_prompt_type_profiles
from prompt_version_manager import commit_prompt_evolution, evolve_prompt
from review_grouping import classify_reviewed_image
from settings_manager import (
    apply_secrets_to_environ,
    public_settings_payload,
    save_secrets,
    save_settings,
    delete_secret,
    get_effective_aesthetic_review_config,
    get_effective_image_provider_config,
)
from storage import JsonStore
from subject_profile_builder import build_subject_profile, validate_subject_lock_input
from system_diagnostics import check_ports, run_diagnostics
from utils import new_id, now_iso


HOST = "127.0.0.1"
PORT = int(os.getenv("PROMPT_AGENT_WEB_PORT", "7860") or "7860")
WEB_DIR = config.BASE_DIR / "web"
REFERENCE_IMAGE_DIR_NAME = "input_reference_images"
HUMAN_DECISIONS = {"keep", "reject", "revise", "high_potential", "rescue"}
RUN_STATUS_CACHE: dict[str, dict[str, Any]] = {}
SUPPORTED_REFERENCE_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def make_server(host: str = HOST, port: int = PORT) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), PromptAgentWebHandler)


def run(host: str = HOST, port: int = PORT) -> None:
    apply_secrets_to_environ()
    try:
        server = make_server(host, port)
    except OSError as exc:
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = config.LOG_DIR / f"web_server_startup_error_{now_iso().replace(':', '').replace('-', '').replace('T', '_')}.log"
        log_path.write_text(str(exc), encoding="utf-8")
        print(f"{host}:{port} 已被占用或无法监听。请关闭旧 web_server.py 或换端口。")
        print(f"详细错误已写入：{log_path}")
        return
    image_cfg = get_effective_image_provider_config()
    review_cfg = get_effective_aesthetic_review_config()
    print(f"Prompt Agent local console listening on http://{host}:{port}")
    print(f"Settings mode: {config.SETTINGS_MODE}")
    print(f"Image provider: {image_cfg.get('provider')} | key exists: {bool(image_cfg.get('api_key'))}")
    print(f"Aesthetic provider: {review_cfg.get('provider')} | TasteLens URL: {review_cfg.get('api_url') or '-'}")
    print(f"Data directory: {config.DATA_DIR} | exists: {config.DATA_DIR.exists()}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.server_close()


class PromptAgentWebHandler(BaseHTTPRequestHandler):
    server_version = "PromptAgentWeb/0.4"

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        route = parsed.path
        query = urllib.parse.parse_qs(parsed.query)
        if route == "/api/status":
            self._send_json(api_status())
        elif route == "/api/provider-profiles":
            self._send_json({"provider_profiles": list_provider_profiles()})
        elif route == "/api/prompt-type-profiles":
            self._send_json({"prompt_type_profiles": load_prompt_type_profiles()})
        elif route == "/api/prompt-packs":
            self._send_json({"prompt_packs": list_prompt_packs()})
        elif route == "/api/idea-requests":
            self._send_json({"idea_requests": sorted_latest(JsonStore().load("idea_requests"))})
        elif route == "/api/image-results":
            self._send_json({"image_results": list_image_results(query)})
        elif route == "/api/image-reviews":
            self._send_json({"image_reviews": list_image_reviews(query)})
        elif route == "/api/reviewed-images":
            self._send_json({"images": list_reviewed_images(query)})
        elif route == "/api/reference-images":
            self._send_json({"reference_images": list_reference_images(_query_value(query, "request_id"))})
        elif route == "/api/settings":
            self._send_json(public_settings_payload())
        elif route == "/api/ports/check":
            self._send_json(check_ports(HOST, PORT))
        elif route.startswith("/api/run-status/"):
            run_id = urllib.parse.unquote(route.rsplit("/", 1)[-1])
            self._send_json(api_run_status(run_id))
        elif route.startswith("/api/incubation-status/"):
            incubation_run_id = urllib.parse.unquote(route.rsplit("/", 1)[-1])
            self._send_json(api_incubation_status(incubation_run_id))
        elif route.startswith("/api/incubation-reviewed-images/"):
            incubation_run_id = urllib.parse.unquote(route.rsplit("/", 1)[-1])
            self._send_json({"images": list_incubation_reviewed_images(incubation_run_id)})
        elif route.startswith("/files/"):
            self._serve_project_file(route[len("/files/") :])
        else:
            self._serve_static(route)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/reference-images/upload":
            result, status = self._handle_reference_upload()
            self._send_json(result, status=status)
            return
        body = self._read_json_body()
        if body is None:
            self._send_json({"ok": False, "error": "Invalid JSON body"}, status=400)
            return
        if parsed.path == "/api/run-prompt-pack":
            self._send_json(api_run_prompt_pack(body))
        elif parsed.path == "/api/idea-requests":
            result, status = api_create_idea_request(body)
            self._send_json(result, status=status)
        elif parsed.path == "/api/extract-image-prompts":
            self._send_json(api_extract_image_prompts(body))
        elif parsed.path == "/api/generate-candidate-prompts":
            self._send_json(api_generate_candidate_prompts(body))
        elif parsed.path == "/api/validate-candidate-prompts":
            self._send_json(api_validate_candidate_prompts(body))
        elif parsed.path == "/api/build-prompt-pack-from-reviewed-candidates":
            self._send_json(api_build_pack_from_reviewed_candidates(body))
        elif parsed.path == "/api/incubate-prompt-pack":
            self._send_json(api_incubate_prompt_pack(body))
        elif parsed.path == "/api/build-pack-from-incubation":
            self._send_json(api_build_pack_from_incubation(body))
        elif parsed.path == "/api/auto-review":
            self._send_json(api_auto_review(body))
        elif parsed.path == "/api/human-review":
            result, status = api_human_review(body)
            self._send_json(result, status=status)
        elif parsed.path == "/api/evolve-prompt":
            self._send_json(api_evolve_prompt(body))
        elif parsed.path == "/api/commit-prompt-evolution":
            self._send_json(api_commit_prompt_evolution(body))
        elif parsed.path == "/api/settings":
            self._send_json(api_save_settings(body))
        elif parsed.path == "/api/settings/secrets":
            result, status = api_save_secrets(body)
            self._send_json(result, status=status)
        elif parsed.path == "/api/diagnostics/run":
            self._send_json(run_diagnostics(
                include_real_image_test=bool(body.get("include_real_image_test")),
                include_aesthetic_test=bool(body.get("include_aesthetic_test")),
                web_host=HOST,
                web_port=PORT,
            ))
        else:
            self._send_json({"ok": False, "error": "Not found"}, status=404)

    def do_DELETE(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        secret_match = re.fullmatch(r"/api/settings/secrets/([^/]+)", parsed.path)
        if secret_match:
            result, status = api_delete_secret(urllib.parse.unquote(secret_match.group(1)))
            self._send_json(result, status=status)
            return
        match = re.fullmatch(r"/api/prompt-packs/([^/]+)", parsed.path)
        if not match:
            self._send_json({"ok": False, "error": "Not found"}, status=404)
            return
        query = urllib.parse.parse_qs(parsed.query)
        if query.get("confirm", ["false"])[0].lower() != "true":
            self._send_json({"ok": False, "error": "confirm=true is required"}, status=400)
            return
        pack_id = urllib.parse.unquote(match.group(1))
        result = delete_prompt_pack(pack_id, JsonStore())
        self._send_json({"ok": bool(result.get("success")), "result": result})

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _read_json_body(self) -> dict[str, Any] | None:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "http://127.0.0.1:%s" % PORT)
        self.end_headers()
        self.wfile.write(data)

    def _serve_static(self, route: str) -> None:
        if route in {"", "/"}:
            route = "/index.html"
        relative = route.lstrip("/")
        web_dir = (config.BASE_DIR / "web").resolve()
        target = (web_dir / relative).resolve()
        try:
            target.relative_to(web_dir)
        except ValueError:
            self.send_error(403)
            return
        if not target.exists() or not target.is_file():
            self.send_error(404)
            return
        self._send_file(target)

    def _serve_project_file(self, relative_path: str) -> None:
        target = resolve_safe_file_path(relative_path)
        if not target or not target.exists() or not target.is_file():
            self.send_error(404)
            return
        self._send_file(target)

    def _send_file(self, path: Path) -> None:
        data = path.read_bytes()
        mime, _ = mimetypes.guess_type(str(path))
        self.send_response(200)
        self.send_header("Content-Type", mime or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle_reference_upload(self) -> tuple[dict[str, Any], int]:
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            return {"ok": False, "error": "multipart/form-data is required"}, 400
        boundary_match = re.search(r"boundary=(?P<boundary>[^;]+)", content_type)
        if not boundary_match:
            return {"ok": False, "error": "multipart boundary is missing"}, 400
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {"ok": False, "error": "empty upload body"}, 400
        raw = self.rfile.read(length)
        fields, files = _parse_multipart(raw, boundary_match.group("boundary").strip('"'))
        request_id = fields.get("request_id", "")
        if not files:
            return {"ok": False, "error": "image files are required"}, 400
        saved_images: list[dict[str, Any]] = []
        for filename, content in files:
            saved_images.extend(save_reference_upload(filename, content, request_id=request_id))
        return {
            "ok": True,
            "request_id": request_id,
            "uploaded_count": len(saved_images),
            "reference_images": saved_images,
            "reference_image": saved_images[0] if saved_images else {},
        }, 200


def api_status() -> dict[str, Any]:
    store = JsonStore()
    packs = store.load("prompt_packs")
    items = store.load("prompt_pack_items")
    results = store.load("image_results")
    reviews = store.load("image_reviews")
    runs = store.load("prompt_pack_runs")
    return {
        "provider_profiles": list_provider_profiles(),
        "prompt_pack_count": len(packs),
        "prompt_pack_item_count": len(items),
        "image_result_count": len(results),
        "image_review_count": len(reviews),
        "latest_run": sorted_latest(runs, 1)[0] if runs else {},
        "aesthetic_review_provider": config.AESTHETIC_REVIEW_PROVIDER,
        "aesthetic_review_api_url": config.AESTHETIC_REVIEW_API_URL,
        "aesthetic_review_api_configured": bool(config.AESTHETIC_REVIEW_API_URL),
        "settings_mode": config.SETTINGS_MODE,
        "effective_image_provider": get_effective_image_provider_config().get("provider"),
        "effective_aesthetic_provider": get_effective_aesthetic_review_config().get("provider"),
    }


def api_save_settings(body: dict[str, Any]) -> dict[str, Any]:
    settings = save_settings(body)
    return {"ok": True, "settings": settings}


def api_save_secrets(body: dict[str, Any]) -> tuple[dict[str, Any], int]:
    try:
        clear = bool(body.get("clear"))
        status = save_secrets(body, clear=clear)
        return {"ok": True, "secrets": status}, 200
    except PermissionError as exc:
        return {"ok": False, "error": str(exc)}, 403


def api_delete_secret(key_name: str) -> tuple[dict[str, Any], int]:
    try:
        result = delete_secret(key_name)
        return {"ok": True, **result}, 200
    except PermissionError as exc:
        return {"ok": False, "error": str(exc)}, 403


def list_provider_profiles() -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    profile_dir = config.BASE_DIR / "provider_profiles"
    if not profile_dir.exists():
        return profiles
    for path in sorted(profile_dir.glob("*.json")):
        profile: dict[str, Any] = {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                profile = payload
        except (OSError, json.JSONDecodeError):
            profile = {}
        profiles.append(
            {
                "profile_path": _relative_to_base(path),
                "file_name": path.name,
                "profile_id": profile.get("profile_id", path.stem),
                "provider": profile.get("provider", ""),
                "display_name": profile.get("display_name", path.stem),
                "model": profile.get("model", ""),
                "api_key_env": profile.get("api_key_env", ""),
            }
        )
    return profiles


def list_prompt_packs() -> list[dict[str, Any]]:
    store = JsonStore()
    packs = store.load("prompt_packs")
    items = store.load("prompt_pack_items")
    counts: dict[str, int] = {}
    for item in items:
        pack_id = str(item.get("pack_id", ""))
        counts[pack_id] = counts.get(pack_id, 0) + 1
    rows = []
    for pack in packs:
        pack_id = str(pack.get("pack_id", ""))
        rows.append(
            {
                "pack_id": pack_id,
                "pack_name": pack.get("pack_name", ""),
                "status": pack.get("status", ""),
                "domain": pack.get("domain", ""),
                "platform": pack.get("platform", ""),
                "use_case": pack.get("use_case", ""),
                "style": pack.get("style", ""),
                "prompt_type": pack.get("prompt_type", "portrait"),
                "review_profile": pack.get("review_profile", "portrait_review"),
                "item_count": counts.get(pack_id, 0),
                "quality_score": pack.get("quality_score", 0),
                "match_tags": list(pack.get("match_tags", []) or []),
            }
        )
    return sorted(rows, key=lambda item: str(item.get("pack_name") or item.get("pack_id")))


def api_run_prompt_pack(body: dict[str, Any]) -> dict[str, Any]:
    request_id = str(body.get("request_id", "")).strip()
    pack_id = str(body.get("pack_id", "")).strip()
    if not request_id or not pack_id:
        return {"ok": False, "error": "request_id and pack_id are required"}

    store = JsonStore()
    request = store.get("idea_requests", "request_id", request_id)
    if not request:
        return {"ok": False, "error": f"idea request not found: {request_id}"}
    registry = PromptPackRegistry(store)
    pack = registry.get_pack(pack_id)
    if not pack:
        return {"ok": False, "error": f"prompt pack not found: {pack_id}"}

    selection = normalize_prompt_selection(body.get("prompt_selection"))
    selected_items = select_prompt_pack_items(registry.get_items(pack_id), selection)
    selected_count = len(selected_items)
    images_per_prompt = _normalize_positive_int(body.get("images_per_prompt"), 1)
    total_task_count = selected_count * images_per_prompt
    if selected_count <= 0:
        return {
            "ok": False,
            "error": "No prompt items matched the selected range",
            "summary": {"pack_id": pack_id, "selected_item_count": 0, "selection_mode": selection.get("mode", "all")},
        }

    provider_profile_path = str(body.get("provider_profile_path", "") or "").strip()
    confirmed = bool(body.get("confirm_large_run") or body.get("confirm"))
    if _is_real_provider(provider_profile_path) and total_task_count > 10 and not confirmed:
        return {
            "ok": False,
            "require_confirm": True,
            "message": f"Current selection will generate {total_task_count} images and may incur cost.",
            "selected_item_count": selected_count,
            "images_per_prompt": images_per_prompt,
            "total_task_count": total_task_count,
        }

    try:
        reference_image_path = str(body.get("reference_image_path", "") or "").strip()
        subject_lock = body.get("subject_lock", {}) if isinstance(body.get("subject_lock", {}), dict) else {}
        subject_enabled = bool(subject_lock.get("enabled", True))
        subject_description = str(subject_lock.get("subject_description", "") or body.get("subject_description", "") or "").strip()
        reference_images = _reference_images_from_path(reference_image_path) if reference_image_path else []
        if subject_enabled:
            ok, message = validate_subject_lock_input(request, reference_images, subject_description)
            if not ok:
                return {"ok": False, "error": message}
        subject_profile = build_subject_profile(request, reference_images, subject_description=subject_description, store=store) if subject_enabled else None
        if subject_profile:
            store.update("idea_requests", "request_id", request_id, {"subject_profile_id": subject_profile.get("subject_profile_id", "")})
        result = run_prompt_pack(
            pack_id,
            request,
            store=store,
            provider_profile_path=provider_profile_path or None,
            prompt_selection=selection,
            images_per_prompt=images_per_prompt,
            subject_profile=subject_profile,
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    run_id = result["run"].get("run_id", "")
    metadata = {
        "provider_profile_path": provider_profile_path,
        "prompt_selection": selection,
        "reference_image_path": reference_image_path,
        "subject_profile_id": (subject_profile or {}).get("subject_profile_id", ""),
        "subject_lock_enabled": bool(subject_profile),
    }
    if run_id:
        store.update("prompt_pack_runs", "run_id", run_id, {"metadata": metadata})
        status = build_run_status(run_id, result["run"], result["summary"], result["image_results"])
        RUN_STATUS_CACHE[run_id] = status
    return {
        "ok": True,
        "run_id": run_id,
        "summary": result["summary"],
        "run_status": RUN_STATUS_CACHE.get(run_id, {}),
    }


def api_run_status(run_id: str) -> dict[str, Any]:
    if run_id in RUN_STATUS_CACHE:
        return RUN_STATUS_CACHE[run_id]
    store = JsonStore()
    run = store.get("prompt_pack_runs", "run_id", run_id)
    if not run:
        return {
            "run_id": run_id,
            "status": "failed",
            "total": 0,
            "completed": 0,
            "success": 0,
            "failed": 0,
            "percent": 0,
            "current_prompt_id": "",
            "message": "run not found",
            "items": [],
            "updated_at": now_iso(),
        }
    result_ids = set(run.get("image_result_ids", []) or [])
    results = [item for item in store.load("image_results") if item.get("image_result_id") in result_ids]
    status = build_run_status(run_id, run, {}, results)
    RUN_STATUS_CACHE[run_id] = status
    return status


def api_create_idea_request(body: dict[str, Any]) -> tuple[dict[str, Any], int]:
    body = apply_prompt_type_defaults(body)
    title = str(body.get("title", "") or "").strip()
    description = str(body.get("description", "") or "").strip()
    if not title and not description:
        return {"ok": False, "error": "title or description is required"}, 400
    requirements = str(body.get("requirements", "") or description).strip()
    request = create_idea_request(
        {
            "raw_idea": description or title,
            "domain": body.get("domain"),
            "platform": body.get("platform"),
            "use_case": body.get("use_case"),
            "style": body.get("style"),
            "ratio": body.get("ratio") or "3:4",
            "topic": body.get("topic") or title or "人像",
            "constraints": [requirements] if requirements else [],
            "target_audience": body.get("target_audience", ""),
            "text_required": bool(body.get("text_required", False)),
            "batch_size": body.get("batch_size", 8),
        }
    )
    request.update(
        {
            "title": title,
            "description": description,
            "requirements": requirements,
            "prompt_type": body.get("prompt_type", "portrait"),
            "review_profile": body.get("review_profile", "portrait_review"),
            "updated_at": now_iso(),
        }
    )
    JsonStore().add("idea_requests", request)
    return {"ok": True, "request_id": request["request_id"], "request": request}, 200


def api_extract_image_prompts(body: dict[str, Any]) -> dict[str, Any]:
    store = JsonStore()
    request_id = str(body.get("request_id", "") or "").strip()
    request = store.get("idea_requests", "request_id", request_id)
    if not request:
        return {"ok": False, "error": "request_id is required"}
    reference_ids = [str(item) for item in body.get("reference_image_ids", []) or []]
    references = list_reference_images(request_id)
    if reference_ids:
        references = [item for item in references if item.get("reference_image_id") in reference_ids]
    if not references:
        return {"ok": False, "error": "reference images are required"}
    try:
        record = extract_image_prompts(request, references, store=store)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    extracted = record.get("extracted", {}) or {}
    return {
        "ok": True,
        "extract_run_id": record["extract_run_id"],
        "summary": {
            "request_id": request_id,
            "reference_image_count": len(references),
            "prompt_skeleton_count": len(extracted.get("prompt_skeletons", []) or []),
            "status": record.get("status", ""),
        },
    }


def api_generate_candidate_prompts(body: dict[str, Any]) -> dict[str, Any]:
    store = JsonStore()
    request = store.get("idea_requests", "request_id", str(body.get("request_id", "") or ""))
    extraction = store.get("image_prompt_extractions", "extract_run_id", str(body.get("extract_run_id", "") or ""))
    if not request or not extraction:
        return {"ok": False, "error": "request_id and extract_run_id are required"}
    try:
        result = generate_candidate_prompts(
            request,
            extraction,
            candidate_count=_normalize_positive_int(body.get("candidate_count"), 8),
            store=store,
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {
        "ok": True,
        "candidate_prompt_count": len(result["candidate_prompt_ids"]),
        "candidate_prompt_ids": result["candidate_prompt_ids"],
        "previews": result["previews"],
        "candidate_run_id": result["run"]["candidate_run_id"],
    }


def api_validate_candidate_prompts(body: dict[str, Any]) -> dict[str, Any]:
    store = JsonStore()
    request = store.get("idea_requests", "request_id", str(body.get("request_id", "") or ""))
    prompt_ids = [str(item) for item in body.get("candidate_prompt_ids", []) or [] if str(item)]
    if not request or not prompt_ids:
        return {"ok": False, "error": "request_id and candidate_prompt_ids are required"}
    try:
        result = validate_candidate_prompts(
            request,
            prompt_ids,
            provider_profile_path=str(body.get("provider_profile_path", "") or ""),
            images_per_prompt=_normalize_positive_int(body.get("images_per_prompt"), 1),
            store=store,
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    run = result["run"]
    summary = {
        "candidate_prompt_count": len(prompt_ids),
        "images_per_prompt": _normalize_positive_int(body.get("images_per_prompt"), 1),
        "generated_task_count": len(result["tasks"]),
    }
    RUN_STATUS_CACHE[run["run_id"]] = build_run_status(run["run_id"], run, summary, result["image_results"])
    return {"ok": True, "run_id": run["run_id"], "summary": summary, "run_status": RUN_STATUS_CACHE[run["run_id"]]}


def api_build_pack_from_reviewed_candidates(body: dict[str, Any]) -> dict[str, Any]:
    store = JsonStore()
    request = store.get("idea_requests", "request_id", str(body.get("request_id", "") or ""))
    prompt_ids = [str(item) for item in body.get("candidate_prompt_ids", []) or [] if str(item)]
    if not request or not prompt_ids:
        return {"ok": False, "error": "request_id and candidate_prompt_ids are required"}
    try:
        result = build_prompt_pack_from_reviewed_candidates(
            request,
            prompt_ids,
            min_aesthetic_score=float(body.get("min_aesthetic_score", 70) or 70),
            accepted_decisions=list(body.get("accepted_decisions", ["keep", "high_potential"]) or []),
            pack_name=str(body.get("pack_name", "") or ""),
            status=str(body.get("status", "candidate") or "candidate"),
            store=store,
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {
        "ok": True,
        "built_pack_id": result["pack"]["pack_id"],
        "kept_prompt_count": result["kept_prompt_count"],
        "rejected_prompt_count": result["rejected_prompt_count"],
    }


def api_incubate_prompt_pack(body: dict[str, Any]) -> dict[str, Any]:
    store = JsonStore()
    request_id = str(body.get("request_id", "") or "").strip()
    request = store.get("idea_requests", "request_id", request_id) if request_id else None
    if not request:
        request_payload = body.get("request", {})
        if isinstance(request_payload, dict) and (request_payload.get("title") or request_payload.get("description")):
            created, status = api_create_idea_request(request_payload)
            if status != 200:
                return created
            request = created["request"]
            request_id = request["request_id"]
        else:
            return {"ok": False, "error": "request_id or request is required"}
    reference_ids = [str(item) for item in body.get("reference_image_ids", []) or []]
    references = list_reference_images(request_id)
    if reference_ids:
        references = [item for item in references if item.get("reference_image_id") in reference_ids]
    type_profile = get_prompt_type_profile(request.get("prompt_type"))
    if bool(type_profile.get("requires_reference_images", False)) and not references:
        return {"ok": False, "error": "当前 Prompt 类型需要上传样图。"}
    subject_lock = body.get("subject_lock", {}) if isinstance(body.get("subject_lock", {}), dict) else {}
    subject_enabled = bool(subject_lock.get("enabled", True))
    subject_description = str(subject_lock.get("subject_description", "") or body.get("subject_description", "") or "").strip()
    if subject_enabled:
        ok, message = validate_subject_lock_input(request, references, subject_description)
        if not ok:
            return {"ok": False, "error": message}
    elif not references:
        return {"ok": False, "error": "reference images are required"}
    subject_profile = build_subject_profile(request, references, subject_description=subject_description, store=store) if subject_enabled else None
    if subject_profile:
        store.update("idea_requests", "request_id", request_id, {"subject_profile_id": subject_profile.get("subject_profile_id", "")})
    incubation = {
        "incubation_run_id": new_id("incubation"),
        "request_id": request_id,
        "reference_image_ids": [item.get("reference_image_id", "") for item in references],
        "subject_profile_id": (subject_profile or {}).get("subject_profile_id", ""),
        "subject_lock_enabled": bool(subject_profile),
        "extract_run_id": "",
        "candidate_prompt_ids": [],
        "validation_run_id": "",
        "auto_review_run_id": "",
        "built_pack_id": "",
        "status": "running",
        "stage": "preparing",
        "percent": 5,
        "message": "准备样图",
        "provider_profile_path": str(body.get("provider_profile_path", "") or ""),
        "prompt_type": request.get("prompt_type", type_profile.get("type_id", "portrait")),
        "review_profile": request.get("review_profile", type_profile.get("review_profile", "portrait_review")),
        "candidate_count": _normalize_positive_int(body.get("candidate_count"), int(type_profile.get("candidate_count_default", 8) or 8)),
        "images_per_prompt": _normalize_positive_int(body.get("images_per_prompt"), int(type_profile.get("images_per_prompt_default", 1) or 1)),
        "created_at": now_iso(),
        "updated_at": now_iso(),
    }
    store.add("prompt_pack_incubation_runs", incubation)
    threading.Thread(target=_run_incubation_pipeline, args=(incubation["incubation_run_id"],), daemon=True).start()
    return {
        "ok": True,
        "incubation_run_id": incubation["incubation_run_id"],
        "run_id": "",
        "status": "running",
        "message": "孵化流程已开始",
    }


def _run_incubation_pipeline(incubation_run_id: str) -> None:
    store = JsonStore()
    incubation = store.get("prompt_pack_incubation_runs", "incubation_run_id", incubation_run_id)
    if not incubation:
        return
    try:
        request = store.get("idea_requests", "request_id", incubation.get("request_id", "")) or {}
        subject_profile = store.get("subject_profiles", "subject_profile_id", incubation.get("subject_profile_id", "")) or None
        wanted = set(incubation.get("reference_image_ids", []) or [])
        references = [item for item in list_reference_images(incubation.get("request_id", "")) if item.get("reference_image_id") in wanted]
        _update_incubation(incubation_run_id, {"stage": "extracting", "percent": 20, "message": "正在提取样图 Prompt"})
        extraction = extract_image_prompts(request, references, store=store)
        _update_incubation(incubation_run_id, {"extract_run_id": extraction["extract_run_id"]})

        _update_incubation(incubation_run_id, {"stage": "generating_candidates", "percent": 35, "message": "正在生成候选 Prompt"})
        candidates = generate_candidate_prompts(request, extraction, candidate_count=incubation.get("candidate_count", 8), store=store, subject_profile=subject_profile)
        candidate_prompt_ids = candidates["candidate_prompt_ids"]
        _update_incubation(incubation_run_id, {"candidate_prompt_ids": candidate_prompt_ids, "candidate_prompt_count": len(candidate_prompt_ids)})

        _update_incubation(incubation_run_id, {"stage": "validating", "percent": 55, "message": "正在验证候选 Prompt"})
        validation = validate_candidate_prompts(
            request,
            candidate_prompt_ids,
            provider_profile_path=str(incubation.get("provider_profile_path", "") or ""),
            images_per_prompt=int(incubation.get("images_per_prompt", 1) or 1),
            store=store,
            subject_profile=subject_profile,
        )
        validation_run = validation["run"]
        _update_incubation(
            incubation_run_id,
            {
                "validation_run_id": validation_run["run_id"],
                "generated_image_count": len(validation["image_results"]),
            },
        )
        RUN_STATUS_CACHE[validation_run["run_id"]] = build_run_status(
            validation_run["run_id"],
            validation_run,
            {"generated_task_count": len(validation["tasks"])},
            validation["image_results"],
        )

        _update_incubation(incubation_run_id, {"stage": "auto_reviewing", "percent": 75, "message": "正在自动审核图片"})
        reviews = review_image_results(validation["image_results"], store=store)
        _update_incubation(
            incubation_run_id,
            {
                "auto_review_run_id": new_id("auto_review"),
                "reviewed_image_count": len(reviews),
                "status": "waiting_human_review",
                "stage": "waiting_human_review",
                "percent": 90,
                "message": "等待人工筛选",
            },
        )
    except Exception as exc:
        _update_incubation(incubation_run_id, {"status": "failed", "stage": "failed", "percent": 100, "message": str(exc)})


def _update_incubation(incubation_run_id: str, updates: dict[str, Any]) -> None:
    JsonStore().update("prompt_pack_incubation_runs", "incubation_run_id", incubation_run_id, {**updates, "updated_at": now_iso()})


def api_incubation_status(incubation_run_id: str) -> dict[str, Any]:
    run = JsonStore().get("prompt_pack_incubation_runs", "incubation_run_id", incubation_run_id)
    if not run:
        return {"ok": False, "error": "incubation run not found"}
    return {
        "ok": True,
        "incubation_run_id": incubation_run_id,
        "status": run.get("status", "running"),
        "stage": run.get("stage", ""),
        "percent": run.get("percent", 0),
        "message": run.get("message", ""),
        "request_id": run.get("request_id", ""),
        "candidate_prompt_count": run.get("candidate_prompt_count", len(run.get("candidate_prompt_ids", []) or [])),
        "generated_image_count": run.get("generated_image_count", 0),
        "reviewed_image_count": run.get("reviewed_image_count", 0),
        "validation_run_id": run.get("validation_run_id", ""),
        "built_pack_id": run.get("built_pack_id", ""),
    }


def list_incubation_reviewed_images(incubation_run_id: str) -> list[dict[str, Any]]:
    store = JsonStore()
    run = store.get("prompt_pack_incubation_runs", "incubation_run_id", incubation_run_id)
    if not run or run.get("stage") not in {"waiting_human_review", "building_pack", "completed"}:
        return []
    result_ids = set(_incubation_image_result_ids(run, store))
    results = [item for item in store.load("image_results") if item.get("image_result_id") in result_ids]
    return build_reviewed_image_rows(results, store)


def list_reviewed_images(query: dict[str, list[str]]) -> list[dict[str, Any]]:
    store = JsonStore()
    results = store.load("image_results")
    run_id = _query_value(query, "run_id")
    if run_id:
        run = store.get("prompt_pack_runs", "run_id", run_id)
        result_ids = set((run or {}).get("image_result_ids", []) or [])
        results = [item for item in results if item.get("image_result_id") in result_ids]
    for key in ["pack_id", "request_id", "prompt_id"]:
        value = _query_value(query, key)
        if value:
            results = [item for item in results if str(item.get(key, "")) == value]
    if not run_id:
        results = sorted_latest(results, _query_int(query, "latest", 24))
    return build_reviewed_image_rows(results, store)


def build_reviewed_image_rows(results: list[dict[str, Any]], store: JsonStore) -> list[dict[str, Any]]:
    decorated = [decorate_image_result(item) for item in results]
    reviews = {item.get("image_result_id"): item for item in store.load("image_reviews")}
    humans = {item.get("image_result_id"): item for item in store.load("human_image_reviews")}
    prompts = {item.get("prompt_id"): item for item in store.load("prompts")}
    rows: list[dict[str, Any]] = []
    for result in decorated:
        review = reviews.get(result.get("image_result_id"), {})
        human = humans.get(result.get("image_result_id"), {})
        aesthetic = review.get("aesthetic", {}) if isinstance(review, dict) else {}
        metadata = result.get("metadata", {}) or {}
        prompt = prompts.get(result.get("prompt_id"), {})
        grouping = classify_reviewed_image(result, review, human)
        human_decision = human.get("human_decision", "")
        rows.append(
            {
                **result,
                "variant_index": metadata.get("variant_index", 1),
                "aesthetic_score": aesthetic.get("score", 0),
                "rating": aesthetic.get("rating", ""),
                "reason": aesthetic.get("reason", ""),
                "prompt_preview": str(prompt.get("full_prompt") or metadata.get("base_prompt") or metadata.get("submitted_prompt") or result.get("prompt_preview", ""))[:160],
                "auto_group": grouping["auto_group"],
                "auto_reject_reason": grouping["auto_reject_reason"],
                "auto_reject_reasons": grouping["auto_reject_reasons"],
                "human_decision": human_decision,
                "final_selected": human_decision in {"keep", "high_potential", "rescue"},
            }
        )
    return rows


def api_build_pack_from_incubation(body: dict[str, Any]) -> dict[str, Any]:
    store = JsonStore()
    incubation_run_id = str(body.get("incubation_run_id", "") or "").strip()
    run = store.get("prompt_pack_incubation_runs", "incubation_run_id", incubation_run_id)
    if not run:
        return {"ok": False, "error": "incubation_run_id is required"}
    try:
        _update_incubation(incubation_run_id, {"status": "running", "stage": "building_pack", "percent": 95, "message": "正在生成新 Prompt 包"})
        result = build_prompt_pack_from_incubation(
            run,
            pack_name=str(body.get("pack_name", "") or ""),
            accepted_decisions=list(body.get("accepted_decisions", ["keep", "high_potential", "rescue"]) or []),
            min_aesthetic_score=float(body.get("min_aesthetic_score", 70) or 70),
            include_unmarked_high_score=bool(body.get("include_unmarked_high_score", False)),
            include_revise=bool(body.get("include_revise", False)),
            include_subject_drift=bool(body.get("include_subject_drift", False)),
            status=str(body.get("status", "candidate") or "candidate"),
            store=store,
        )
    except Exception as exc:
        _update_incubation(incubation_run_id, {"status": "failed", "stage": "failed", "percent": 100, "message": str(exc)})
        return {"ok": False, "error": str(exc)}
    store.update(
        "prompt_pack_incubation_runs",
        "incubation_run_id",
        incubation_run_id,
        {
            "built_pack_id": result["pack"]["pack_id"],
            "status": "completed",
            "stage": "completed",
            "percent": 100,
            "message": "新 Prompt 包已生成",
            "updated_at": now_iso(),
        },
    )
    return {
        "ok": True,
        "built_pack_id": result["pack"]["pack_id"],
        "kept_prompt_count": result["kept_prompt_count"],
        "rejected_prompt_count": result["rejected_prompt_count"],
        "rescued_prompt_count": result.get("rescued_prompt_count", 0),
    }


def _incubation_image_result_ids(run: dict[str, Any], store: JsonStore) -> list[str]:
    validation_run = store.get("prompt_pack_runs", "run_id", run.get("validation_run_id", ""))
    return list((validation_run or {}).get("image_result_ids", []) or [])


def api_auto_review(body: dict[str, Any]) -> dict[str, Any]:
    store = JsonStore()
    results = store.load("image_results")
    for key in ["pack_id", "request_id", "prompt_id"]:
        value = str(body.get(key, "") or "").strip()
        if value:
            results = [item for item in results if str(item.get(key, "")) == value]
    reviews = review_image_results(results, store=store)
    return {"ok": True, "review_count": len(reviews)}


def api_human_review(body: dict[str, Any]) -> tuple[dict[str, Any], int]:
    decision = str(body.get("human_decision", "")).strip()
    if decision not in HUMAN_DECISIONS:
        return {"ok": False, "error": "invalid human_decision"}, 400
    image_result_id = str(body.get("image_result_id", "")).strip()
    if not image_result_id:
        return {"ok": False, "error": "image_result_id is required"}, 400

    store = JsonStore()
    result = store.get("image_results", "image_result_id", image_result_id)
    review = create_human_image_review(
        {
            "image_result_id": image_result_id,
            "request_id": body.get("request_id") or (result or {}).get("request_id", ""),
            "pack_id": body.get("pack_id") or (result or {}).get("pack_id", ""),
            "prompt_id": body.get("prompt_id") or (result or {}).get("prompt_id", ""),
            "human_decision": decision,
            "human_score": body.get("human_score", 0),
            "reject_reasons": body.get("reject_reasons", []),
            "notes": body.get("notes", ""),
            "source": body.get("source") or ("filtered_out_rescue" if decision == "rescue" else "included_review"),
            "subject_consistency": body.get("subject_consistency", ""),
        }
    )
    store.add("human_image_reviews", review)
    return {"ok": True, "human_review": review}, 200


def api_evolve_prompt(body: dict[str, Any]) -> dict[str, Any]:
    try:
        result = evolve_prompt(
            str(body.get("pack_id", "") or ""),
            str(body.get("logical_prompt_id", "") or ""),
            str(body.get("base_prompt_version_id", "") or ""),
            selected_image_result_ids=list(body.get("selected_image_result_ids", []) or []),
            selected_human_review_ids=list(body.get("selected_human_review_ids", []) or []),
            selected_review_ids=list(body.get("selected_review_ids", []) or []),
            evolution_notes=str(body.get("evolution_notes", "") or ""),
            provider_profile_path=str(body.get("provider_profile_path", "") or ""),
            images_per_prompt=_normalize_positive_int(body.get("images_per_prompt"), 1),
            validation_count=_normalize_positive_int(body.get("validation_count"), 3),
            store=JsonStore(),
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    run = result["run"]
    return {
        "ok": True,
        "evolution_run_id": run["evolution_run_id"],
        "new_prompt_version_id": run["candidate_prompt_version_id"],
        "validation_run_id": run.get("validation_run_id", ""),
        "status": run["status"],
    }


def api_commit_prompt_evolution(body: dict[str, Any]) -> dict[str, Any]:
    try:
        result = commit_prompt_evolution(
            str(body.get("evolution_run_id", "") or ""),
            include_high_potential=bool(body.get("include_high_potential", True)),
            include_rescue=bool(body.get("include_rescue", True)),
            include_revise=bool(body.get("include_revise", False)),
            store=JsonStore(),
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, **result}


def list_image_results(query: dict[str, list[str]]) -> list[dict[str, Any]]:
    results = JsonStore().load("image_results")
    for key in ["pack_id", "request_id", "prompt_id"]:
        value = _query_value(query, key)
        if value:
            results = [item for item in results if str(item.get(key, "")) == value]
    rows = sorted_latest(results, _query_int(query, "latest", 20))
    return [decorate_image_result(row) for row in rows]


def list_image_reviews(query: dict[str, list[str]]) -> list[dict[str, Any]]:
    reviews = JsonStore().load("image_reviews")
    for key in ["image_result_id", "prompt_id"]:
        value = _query_value(query, key)
        if value:
            reviews = [item for item in reviews if str(item.get(key, "")) == value]
    return sorted_latest(reviews, _query_int(query, "latest", 20))


def list_reference_images(request_id: str = "") -> list[dict[str, Any]]:
    root = _reference_image_dir(request_id)
    root.mkdir(parents=True, exist_ok=True)
    images: list[dict[str, Any]] = []
    paths = root.rglob("*") if not request_id else root.iterdir()
    for path in sorted([item for item in paths if item.is_file()], key=lambda item: item.stat().st_mtime, reverse=True):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_REFERENCE_IMAGE_SUFFIXES:
            continue
        images.append(
            {
                "reference_image_id": _reference_image_id(path),
                "filename": path.name,
                "file_name": path.name,
                "image_path": _relative_to_base(path),
                "path": _relative_to_base(path),
                "url": file_url_for_path(str(path)),
                "file_url": file_url_for_path(str(path)),
                "size": path.stat().st_size,
                "request_id": _request_id_from_reference_path(path),
                "updated_at": now_iso(),
            }
        )
    return images


def _reference_images_from_path(reference_image_path: str) -> list[dict[str, Any]]:
    if not reference_image_path:
        return []
    normalized = reference_image_path.replace("\\", "/")
    for image in list_reference_images():
        if str(image.get("path", "")).replace("\\", "/") == normalized or str(image.get("image_path", "")).replace("\\", "/") == normalized:
            return [image]
    return [
        {
            "reference_image_id": "",
            "path": reference_image_path,
            "image_path": reference_image_path,
        }
    ]


def save_reference_upload(filename: str, content: bytes, request_id: str = "") -> list[dict[str, Any]]:
    suffix = Path(filename).suffix.lower()
    if suffix == ".zip":
        return _save_reference_zip(content, request_id)
    if suffix not in SUPPORTED_REFERENCE_IMAGE_SUFFIXES:
        return []
    return [save_reference_image(filename, content, request_id=request_id)]


def save_reference_image(filename: str, content: bytes, request_id: str = "") -> dict[str, Any]:
    root = _reference_image_dir(request_id)
    root.mkdir(parents=True, exist_ok=True)
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_REFERENCE_IMAGE_SUFFIXES:
        suffix = ".png"
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(filename).stem).strip("._") or "reference"
    target = root / f"{safe_stem}_{now_iso().replace(':', '').replace('-', '').replace('T', '_')}{suffix}"
    target.write_bytes(content)
    return {
        "reference_image_id": _reference_image_id(target),
        "file_name": target.name,
        "filename": target.name,
        "image_path": _relative_to_base(target),
        "path": _relative_to_base(target),
        "url": file_url_for_path(str(target)),
        "file_url": file_url_for_path(str(target)),
        "size": len(content),
        "request_id": request_id,
        "created_at": now_iso(),
    }


def decorate_image_result(result: dict[str, Any]) -> dict[str, Any]:
    row = dict(result)
    image_path = str(row.get("image_path", "") or "")
    row["file_url"] = file_url_for_path(image_path) if image_path else ""
    if image_path and not row["file_url"] and Path(image_path).exists():
        row["file_error"] = "image path is outside allowed file roots"
    elif image_path and not Path(image_path).exists():
        row["file_error"] = "image file not found"
    else:
        row["file_error"] = ""
    if str(row.get("provider", "")) == "mock" and image_path and Path(image_path).exists():
        try:
            row["mock_preview"] = Path(image_path).read_text(encoding="utf-8")[:500]
        except OSError:
            row["mock_preview"] = ""
    return row


def build_run_status(
    run_id: str,
    run: dict[str, Any],
    summary: dict[str, Any] | None,
    image_results: list[dict[str, Any]],
) -> dict[str, Any]:
    total = int((summary or {}).get("generated_task_count") or len(run.get("task_ids", []) or []) or len(image_results))
    completed = len(image_results)
    success = sum(1 for item in image_results if item.get("status") == "success")
    failed = sum(1 for item in image_results if item.get("status") == "failed")
    status = str(run.get("status", "completed") or "completed")
    if failed and completed >= total and success == 0:
        status = "failed"
    elif completed >= total:
        status = "completed"
    percent = int(round((completed / total) * 100)) if total else 0
    items = [
        {
            "image_result_id": item.get("image_result_id", ""),
            "prompt_id": item.get("prompt_id", ""),
            "variant_index": (item.get("metadata", {}) or {}).get("variant_index", 1),
            "images_per_prompt": (item.get("metadata", {}) or {}).get("images_per_prompt", 1),
            "status": item.get("status", ""),
            "error_message": item.get("error_message", ""),
        }
        for item in image_results
    ]
    current_prompt_id = ""
    current_variant_index = 0
    images_per_prompt = 1
    if 0 <= completed - 1 < len(items):
        current_prompt_id = str(items[completed - 1].get("prompt_id", ""))
        current_variant_index = int(items[completed - 1].get("variant_index", 1) or 1)
        images_per_prompt = int(items[completed - 1].get("images_per_prompt", 1) or 1)
    message = (
        "生成完成"
        if status == "completed"
        else "生成失败"
        if status == "failed"
        else f"正在生成 {current_prompt_id} 的第 {current_variant_index or 1} 张变体"
    )
    return {
        "run_id": run_id,
        "status": status,
        "total": total,
        "completed": completed,
        "success": success,
        "failed": failed,
        "percent": percent,
        "current_prompt_id": current_prompt_id,
        "current_variant_index": current_variant_index,
        "images_per_prompt": images_per_prompt,
        "message": message,
        "items": items,
        "updated_at": now_iso(),
    }


def normalize_prompt_selection(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"mode": "first_n", "n": 3}
    mode = str(value.get("mode", "first_n") or "first_n")
    if mode == "all":
        return {"mode": "all"}
    if mode == "range":
        return {"mode": "range", "start": str(value.get("start", "")).strip(), "end": str(value.get("end", "")).strip()}
    if mode == "list":
        prompt_ids = value.get("prompt_ids", [])
        if isinstance(prompt_ids, str):
            prompt_ids = [item.strip() for item in prompt_ids.split(",") if item.strip()]
        return {"mode": "list", "prompt_ids": list(prompt_ids or [])}
    try:
        n = int(value.get("n", 3) or 3)
    except (TypeError, ValueError):
        n = 3
    return {"mode": "first_n", "n": max(1, n)}


def file_url_for_path(path_value: str) -> str:
    path = _normalize_project_path(path_value)
    if not path or not _is_allowed_file_path(path):
        return ""
    return "/files/" + urllib.parse.quote(_relative_to_base(path).replace("\\", "/"))


def resolve_safe_file_path(relative_url_path: str) -> Path | None:
    relative = urllib.parse.unquote(relative_url_path).replace("\\", "/").lstrip("/")
    if not relative or ".." in Path(relative).parts:
        return None
    candidate = (config.BASE_DIR / relative).resolve()
    return candidate if _is_allowed_file_path(candidate) else None


def _parse_multipart(raw: bytes, boundary: str) -> tuple[dict[str, str], list[tuple[str, bytes]]]:
    fields: dict[str, str] = {}
    files: list[tuple[str, bytes]] = []
    delimiter = ("--" + boundary).encode("utf-8")
    for part in raw.split(delimiter):
        if not part or part in {b"--\r\n", b"--"}:
            continue
        header_blob, sep, content = part.partition(b"\r\n\r\n")
        if not sep:
            continue
        headers = header_blob.decode("utf-8", errors="ignore")
        disposition = next((line for line in headers.split("\r\n") if line.lower().startswith("content-disposition:")), "")
        name_match = re.search(r'name="([^"]*)"', disposition)
        name = name_match.group(1) if name_match else ""
        content = content.rstrip(b"\r\n-")
        if "filename=" not in disposition:
            fields[name] = content.decode("utf-8", errors="ignore")
            continue
        filename_match = re.search(r'filename="([^"]*)"', disposition)
        filename = filename_match.group(1) if filename_match else f"{name or 'file'}.png"
        files.append((filename, content))
    return fields, files


def _save_reference_zip(content: bytes, request_id: str = "") -> list[dict[str, Any]]:
    saved: list[dict[str, Any]] = []
    with zipfile.ZipFile(BytesIO(content)) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename.replace("\\", "/")).name
            if not name or Path(name).suffix.lower() not in SUPPORTED_REFERENCE_IMAGE_SUFFIXES:
                continue
            saved.append(save_reference_image(name, archive.read(info), request_id=request_id))
    return saved


def sorted_latest(records: list[dict[str, Any]], limit: int | None = None) -> list[dict[str, Any]]:
    rows = sorted(records, key=lambda item: str(item.get("created_at", "")), reverse=True)
    return rows[:limit] if limit else rows


def _query_value(query: dict[str, list[str]], key: str) -> str:
    values = query.get(key, [])
    return str(values[0]).strip() if values else ""


def _query_int(query: dict[str, list[str]], key: str, default: int) -> int:
    try:
        return int(_query_value(query, key) or default)
    except ValueError:
        return default


def _normalize_positive_int(value: Any, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return number if number > 0 else default


def _relative_to_base(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(config.BASE_DIR.resolve()))
    except ValueError:
        return str(path)


def _normalize_project_path(path_value: str) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = config.BASE_DIR / path
    return path.resolve()


def _allowed_file_roots() -> list[Path]:
    return [
        config.MOCK_IMAGE_OUTPUT_DIR.resolve(),
        config.REAL_IMAGE_OUTPUT_DIR.resolve(),
        (config.MOCK_IMAGE_OUTPUT_DIR / "real_images").resolve(),
        config.INPUT_IMAGES_DIR.resolve(),
        _reference_image_dir("").resolve(),
    ]


def _is_allowed_file_path(path: Path) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        return False
    for root in _allowed_file_roots():
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _is_real_provider(provider_profile_path: str) -> bool:
    if provider_profile_path:
        try:
            payload = json.loads((config.BASE_DIR / provider_profile_path).read_text(encoding="utf-8"))
            return str(payload.get("provider", "")).strip().lower() != "mock"
        except (OSError, json.JSONDecodeError):
            return True
    return str(config.IMAGE_PROVIDER).strip().lower() != "mock"


def _reference_image_dir(request_id: str = "") -> Path:
    safe_request_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(request_id or "").strip()).strip("._")
    root = config.BASE_DIR / REFERENCE_IMAGE_DIR_NAME
    return root / safe_request_id if safe_request_id else root


def _reference_image_id(path: Path) -> str:
    return "ref_" + re.sub(r"[^A-Za-z0-9]+", "_", _relative_to_base(path)).strip("_")


def _request_id_from_reference_path(path: Path) -> str:
    try:
        relative = path.resolve().relative_to((config.BASE_DIR / REFERENCE_IMAGE_DIR_NAME).resolve())
    except ValueError:
        return ""
    return relative.parts[0] if len(relative.parts) > 1 else ""


if __name__ == "__main__":
    run()
