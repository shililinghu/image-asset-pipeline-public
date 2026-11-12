from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
import image_generation_client
import storage
from human_review_exporter import export_human_review_csv, import_human_review_csv
from idea_analyzer import analyze_idea
from image_auto_reviewer import review_image_results
from image_generation_runner import run_prompts_for_request
from models import make_project
from optimizer import optimize_prompt_from_image_feedback
from prompt_feedback_bridge import build_prompt_image_feedback
from prompt_generator import generate_candidates, next_version
from prompt_pack_builder import build_pack_from_prompt_results
from prompt_pack_matcher import match_prompt_pack
from prompt_pack_registry import PromptPackRegistry
from prompt_pack_runner import run_prompt_pack
from storage import JsonStore


class OfflineLLM:
    def is_available(self) -> bool:
        return False


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="prompt_pack_smoke_") as tmp:
        tmp_dir = Path(tmp)
        store = _isolated_store(tmp_dir)
        registry = PromptPackRegistry(store)
        sample_pack = registry.import_pack_file(ROOT / "examples" / "sample_prompt_pack.json")

        # Route A: existing pack -> match -> run -> mock image results -> review -> CSV.
        route_a_request = analyze_idea(
            "我要做一批小红书化学知识封面图，主题是醛酮醌，风格清爽高级，比例 3:4。",
            store=store,
            platform="xiaohongshu",
            domain="chemistry",
            use_case="cover",
            style="clean_education",
            ratio="3:4",
            topic="醛酮醌",
        )
        route_a_match = match_prompt_pack(route_a_request, store=store)
        _assert(route_a_match["matched"], "Route A should match sample pack")
        _assert(route_a_match["pack_id"] == sample_pack["pack_id"], "Route A should match imported sample pack")
        route_a_run = run_prompt_pack(route_a_match["pack_id"], route_a_request, store=store)
        route_a_reviews = review_image_results(route_a_run["image_results"], store=store)
        route_a_csv = export_human_review_csv(route_a_request["request_id"], tmp_dir / "route_a_human.csv", store=store)
        _write_mock_human_csv(route_a_csv)
        route_a_human = import_human_review_csv(route_a_csv, store=store)

        # Route B: no suitable pack -> candidate prompts -> image test -> feedback -> optimize -> build pack.
        route_b_request = analyze_idea(
            "我要做一批抖音健身课程海报，主题是办公室肩颈放松，风格运动清爽，比例 16:9。",
            store=store,
            platform="douyin",
            domain="fitness",
            use_case="poster",
            style="sport_clean",
            ratio="16:9",
            topic="办公室肩颈放松",
        )
        route_b_match = match_prompt_pack(route_b_request, store=store)
        _assert(route_b_match["decision"] != "use_existing_pack", "Route B should not use existing pack")
        project = store.add("projects", make_project(_project_from_request(route_b_request)))
        version = next_version(store.load("prompts"), project["project_id"])
        candidates = generate_candidates(project, version, OfflineLLM())
        for prompt in candidates:
            prompt["source_request_id"] = route_b_request["request_id"]
            store.add("prompts", prompt)
        tasks_b, results_b = run_prompts_for_request(route_b_request, candidates, count_per_prompt=3, store=store)
        reviews_b = review_image_results(results_b, store=store)
        _force_reviews_to_pass(store, reviews_b)
        route_b_csv = export_human_review_csv(route_b_request["request_id"], tmp_dir / "route_b_human.csv", store=store)
        _write_mock_human_csv(route_b_csv)
        route_b_human = import_human_review_csv(route_b_csv, store=store)
        feedback_items = build_prompt_image_feedback(route_b_request["request_id"], store=store)
        _assert(feedback_items, "Route B should create prompt_image_feedback")
        optimized_prompt, _iteration = optimize_prompt_from_image_feedback(
            feedback_items[0]["prompt_id"],
            feedback_items[0]["feedback_id"],
            store=store,
            llm=OfflineLLM(),
        )
        built = build_pack_from_prompt_results(route_b_request, store=store)
        built_pack = built["pack"]
        _assert(built_pack["pack_id"], "Route B should build a new pack")

        # Route C: new pack is registered -> matched -> run again.
        route_c_match = match_prompt_pack(route_b_request, store=store)
        _assert(route_c_match["matched"], "Route C should match a registered pack")
        _assert(route_c_match["pack_id"] == built_pack["pack_id"], "Route C should match the newly built pack")
        route_c_run = run_prompt_pack(built_pack["pack_id"], route_b_request, store=store)

        _assert_traceability(store, built_pack["pack_id"])

        summary = {
            "created_request_id": route_a_request["request_id"],
            "matched_pack_id": route_a_match["pack_id"],
            "generated_task_count": len(route_a_run["tasks"]) + len(tasks_b),
            "image_result_count": len(route_a_run["image_results"]) + len(results_b),
            "image_review_count": len(route_a_reviews) + len(reviews_b),
            "human_review_count": len(route_a_human) + len(route_b_human),
            "prompt_feedback_count": len(feedback_items),
            "optimized_prompt_id": optimized_prompt["prompt_id"],
            "built_pack_id": built_pack["pack_id"],
            "rerun_image_result_count": len(route_c_run["image_results"]),
        }
        print("SMOKE_TEST_SUMMARY")
        print(json.dumps(summary, ensure_ascii=False, indent=2))


def _isolated_store(tmp_dir: Path) -> JsonStore:
    data_dir = tmp_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    files = {name: data_dir / f"{name}.json" for name in config.DATA_FILES}
    config.DATA_FILES.clear()
    config.DATA_FILES.update(files)
    storage.DATA_FILES = config.DATA_FILES
    storage.DATA_DIR = data_dir
    image_generation_client.IMAGE_OUTPUT_DIR = tmp_dir / "mock_images"
    return JsonStore()


def _project_from_request(request: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": f"Smoke incubation - {request['topic']}",
        "scenario": request["raw_idea"],
        "goal": f"Create image prompts for {request['use_case']}",
        "audience": request.get("target_audience", ""),
        "desired_output": f"Mock image prompt candidates, ratio {request['ratio']}",
        "avoid": "; ".join(request.get("constraints", [])),
        "style_preference": request.get("style", ""),
        "evaluation_criteria": "mock image quality, human review, safety",
    }


def _write_mock_human_csv(path: Path) -> None:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
        fieldnames = list(rows[0].keys()) if rows else []
    decisions = ["keep", "high_potential", "reject"]
    for index, row in enumerate(rows):
        row["human_decision"] = decisions[index % len(decisions)]
        row["human_score"] = "5" if row["human_decision"] != "reject" else "4"
        row["reject_reasons"] = "composition issue" if row["human_decision"] == "reject" else ""
        row["notes"] = "mock human review"
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _force_reviews_to_pass(store: JsonStore, reviews: list[dict[str, Any]]) -> None:
    for review in reviews:
        store.update(
            "image_reviews",
            "review_id",
            review["review_id"],
            {"auto_score": 90, "safety_passed": True, "auto_decision": "keep"},
        )


def _assert_traceability(store: JsonStore, built_pack_id: str) -> None:
    for result in store.load("image_results"):
        for field in ["request_id", "pack_id", "prompt_id", "image_result_id", "image_path", "status"]:
            _assert(field in result, f"image_result missing {field}")
    for feedback in store.load("prompt_image_feedback"):
        for field in ["prompt_id", "request_id", "based_on_image_results", "based_on_human_reviews"]:
            _assert(field in feedback, f"prompt_image_feedback missing {field}")
    pack = store.get("prompt_packs", "pack_id", built_pack_id)
    _assert(pack is not None, "built pack missing from registry")
    for field in ["source_prompt_ids", "source_feedback_ids", "source_image_result_ids"]:
        _assert(field in pack, f"built pack missing {field}")
        _assert(bool(pack[field]), f"built pack should have non-empty {field}")


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
