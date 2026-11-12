from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from config import DEFAULT_EXPECTED_FOCUS, DEEPSEEK_API_KEY, DEEPSEEK_MODEL, IMAGE_PROVIDER, LLM_PROVIDER, MANUAL_MODE
from delete_manager import (
    clear_project_feedback_iterations,
    clear_project_generated_data,
    clear_project_prompts,
    clear_project_results,
    clear_project_scores,
    clear_project_test_cases,
    delete_feedback,
    delete_iteration,
    delete_project_cascade,
    delete_prompt_pack,
    delete_prompt,
    delete_score,
    delete_test_case,
    delete_test_result,
    get_data_stats,
    preview_delete_prompt,
    preview_delete_simple,
    preview_delete_test_case,
    preview_delete_test_result,
    preview_project_action,
)
from feedback import collect_feedback
from human_review_exporter import export_human_review_csv, import_human_review_csv
from idea_analyzer import analyze_idea
from image_auto_reviewer import review_image_results
from image_generation_runner import run_prompts_for_request
from identity_builder import build_task_profile
from llm_client import LLMClient
from models import create_prompt_pack_match_log, make_project, make_test_case
from optimizer import (
    aggregate_document_dimensions,
    explain_best,
    format_dimension_item,
    get_preferred_scoring_mode,
    filter_scores_by_mode,
    get_no_qualified_prompt_message,
    optimize_from_feedback,
    optimize_prompt_from_image_feedback,
    select_best,
)
from prompt_feedback_bridge import build_prompt_image_feedback
from prompt_pack_builder import build_pack_from_prompt_results
from prompt_pack_matcher import match_prompt_pack
from prompt_pack_registry import PromptPackRegistry
from prompt_pack_runner import run_prompt_pack
from prompt_generator import build_auto_test_case, generate_candidates, generate_fusion_candidate, generate_test_cases, next_version
from scorer import score_result
from scoring_rules import DOCUMENT_RULES_MODE, GENERAL_MODE
from storage import JsonStore
from tester import run_test
from utils import choose_from_list, ensure_console_utf8, print_box, prompt_multiline, prompt_optional, prompt_required, safe_filename


ensure_console_utf8()
store = JsonStore()
llm = LLMClient()
current_provider_profile_path = ""


class BackToMenu(Exception):
    pass


def mode_label() -> str:
    return "手动模式" if MANUAL_MODE or not llm.is_available() else "API 自动模式"


def current_model_label() -> str:
    return getattr(llm, "model", DEEPSEEK_MODEL)


def choose_project() -> dict | None:
    return choose_from_list(store.load("projects"), "name", "选择项目", "project_id")


def choose_prompt(project_id: str, title: str = "选择提示词") -> dict | None:
    prompts = sorted(
        store.filter("prompts", project_id=project_id),
        key=lambda item: (int(item.get("version", 0)), item.get("title", "")),
    )
    return choose_from_list(prompts, "title", title, "prompt_id")


def choose_case(project_id: str, title: str = "选择测试案例") -> dict | None:
    cases = sorted(store.filter("test_cases", project_id=project_id), key=lambda item: item.get("title", ""))
    return choose_from_list(cases, "title", title, "case_id")


def choose_result(project_id: str, title: str = "选择测试结果") -> dict | None:
    results = store.filter("test_results", project_id=project_id)
    items = []
    prompt_map = {item.get("prompt_id"): item for item in store.filter("prompts", project_id=project_id)}
    case_map = {item.get("case_id"): item for item in store.filter("test_cases", project_id=project_id)}
    for index, result in enumerate(results, start=1):
        prompt_title = prompt_map.get(result.get("prompt_id"), {}).get("title", "未知提示词")
        case_title = case_map.get(result.get("case_id"), {}).get("title", "未知案例")
        copy = dict(result)
        copy["_label"] = f"{prompt_title} | {case_title} | {result.get('result_id', f'unknown_{index}')}"
        items.append(copy)
    return choose_from_list(items, "_label", title, "result_id")


def choose_score(project_id: str, title: str = "选择评分结果") -> dict | None:
    scores = store.filter("scores", project_id=project_id)
    prompts = {item.get("prompt_id"): item for item in store.filter("prompts", project_id=project_id)}
    items = []
    for index, score in enumerate(scores, start=1):
        prompt_title = prompts.get(score.get("prompt_id"), {}).get("title", "未知提示词")
        copy = dict(score)
        copy["score_id"] = copy.get("score_id") or f"unknown_{index}"
        copy["_label"] = f"{prompt_title} | {score.get('total_score', 0)} 分 | {copy['score_id']}"
        items.append(copy)
    return choose_from_list(items, "_label", title, "score_id")


def choose_feedback(project_id: str, title: str = "选择反馈记录") -> dict | None:
    feedback_items = [dict(item) for item in store.filter("feedback", project_id=project_id)]
    prompts = {item.get("prompt_id"): item for item in store.filter("prompts", project_id=project_id)}
    items = []
    for index, item in enumerate(feedback_items, start=1):
        item["feedback_id"] = item.get("feedback_id") or f"unknown_{index}"
        prompt_title = prompts.get(item.get("prompt_id"), {}).get("title", "未知提示词")
        item["_label"] = f"{prompt_title} | {item.get('real_scenario', '无场景')} | {item['feedback_id']}"
        items.append(item)
    return choose_from_list(items, "_label", title, "feedback_id")


def choose_iteration(project_id: str, title: str = "选择迭代记录") -> dict | None:
    iterations = [dict(item) for item in store.filter("iterations", project_id=project_id)]
    items = []
    for index, item in enumerate(iterations, start=1):
        item["iteration_id"] = item.get("iteration_id") or f"unknown_{index}"
        item["_label"] = f"version {item.get('version', '?')} | {item.get('old_prompt_id', '未知')} -> {item.get('new_prompt_id', '未知')} | {item['iteration_id']}"
        items.append(item)
    return choose_from_list(items, "_label", title, "iteration_id")


def create_project() -> None:
    print_box("创建新优化项目")
    data = {
        "name": prompt_required("任务名称"),
        "scenario": prompt_required("使用场景"),
        "goal": prompt_required("目标"),
        "audience": prompt_required("目标读者 / 使用对象"),
        "desired_output": prompt_required("希望输出内容"),
        "avoid": prompt_optional("不希望出现的内容", "空泛、跑题、过度夸张"),
        "style_preference": prompt_optional("风格偏好", "清晰、具体、自然"),
        "evaluation_criteria": prompt_optional("评价标准", "贴合需求、清晰、可执行、有吸引力"),
    }
    project = store.add("projects", make_project(data))
    print(f"已创建项目：{project['name']} [{project['project_id']}]")
    add_now = prompt_optional("是否现在添加测试案例（yes/no）", "yes").lower()
    while add_now == "yes":
        save_manual_test_case(project["project_id"])
        add_now = prompt_optional("是否继续添加测试案例（yes/no）", "no").lower()


def save_manual_test_case(project_id: str) -> dict:
    print_box("添加测试案例")
    title = prompt_required("案例标题")
    case_input = prompt_multiline("测试输入")
    expected_focus = prompt_multiline("期望关注点")
    case = store.add("test_cases", make_test_case(project_id, title, case_input, expected_focus or DEFAULT_EXPECTED_FOCUS))
    print(f"已添加测试案例：{case['title']} [{case['case_id']}]")
    return case


def list_projects() -> None:
    print_box("已有项目")
    projects = store.load("projects")
    if not projects:
        print("暂无项目。")
        return
    for project in projects:
        print(f"- {project.get('name', '未命名项目')} [{project.get('project_id', '未知ID')}] | 目标：{project.get('goal', '')} | 创建：{project.get('created_at', '')}")


def generate_project_prompts() -> None:
    project = choose_project()
    if not project:
        return
    version = next_version(store.load("prompts"), project["project_id"])
    candidates = generate_candidates(project, version, llm)
    for item in candidates:
        store.add("prompts", item)
    print_box(f"已生成 {len(candidates)} 个候选提示词（version {version}）")
    for item in candidates:
        print(f"\n{item['title']} [{item['prompt_id']}]\n适合：{item.get('suitable_for', '')}\n风险：{item.get('risk', '')}")


def generate_fusion_prompt_for_project() -> None:
    project = choose_project()
    if not project:
        return
    prompts = sorted(
        store.filter("prompts", project_id=project["project_id"]),
        key=lambda item: (int(item.get("version", 0)), item.get("title", "")),
    )
    if not prompts:
        print("当前项目还没有 A/B/C 候选提示词，请先生成候选提示词。")
        return

    grouped: dict[int, list[dict]] = defaultdict(list)
    for prompt in prompts:
        grouped[int(prompt.get("version", 0))].append(prompt)

    version_items: list[dict] = []
    for version in sorted(grouped):
        base_prompts = [item for item in grouped[version] if str(item.get("prompt_type", item.get("kind", ""))) != "fusion_optimized"]
        if len(base_prompts) >= 3:
            version_items.append(
                {
                    "version": version,
                    "_label": f"version {version} | {len(base_prompts)} 个基础候选 | {sum(1 for item in grouped[version] if str(item.get('prompt_type', item.get('kind', ''))) == 'fusion_optimized')} 个融合版",
                }
            )
    selected_version = choose_from_list(version_items, "_label", "选择要生成 D版 的版本", "version") if version_items else None
    if not selected_version:
        print("没有找到可用于融合的同版本 A/B/C 候选提示词。")
        return

    version = int(selected_version["version"])
    base_prompts = [item for item in grouped[version] if str(item.get("prompt_type", item.get("kind", ""))) != "fusion_optimized"][:3]
    fusion_prompt = generate_fusion_candidate(project, version, base_prompts, llm)
    if not fusion_prompt:
        print("D版 融合优化型生成失败。")
        return
    store.add("prompts", fusion_prompt)
    print_box("已生成 D版：融合优化型")
    print(f"{fusion_prompt['title']} [{fusion_prompt['prompt_id']}]")
    print("建议下一步执行：")
    print("5. 运行提示词测试")
    print("6. 给测试结果评分 / 筛选提示词")


def add_test_case() -> None:
    project = choose_project()
    if not project:
        return
    save_manual_test_case(project["project_id"])


def run_prompt_tests() -> None:
    project = choose_project()
    if not project:
        return

    prompts = store.filter("prompts", project_id=project["project_id"])
    if not prompts:
        print("当前项目还没有候选提示词，请先生成候选提示词。")
        return

    print_box("请选择测试对象")
    print("1. 测试当前项目下全部提示词")
    print("2. 只测试某一个提示词")
    print("0. 返回")
    target_choice = input("请输入编号: ").strip()
    if target_choice == "0":
        return
    if target_choice == "2":
        prompt = choose_prompt(project["project_id"], "选择要测试的提示词")
        if not prompt:
            return
        prompts = [prompt]
    elif target_choice != "1":
        print("无效选项。")
        return

    cases = store.filter("test_cases", project_id=project["project_id"])
    if not cases:
        case = handle_no_test_cases(project, prompts)
        if not case:
            return
        cases = [case]

    print_box("运行提示词测试")
    all_results = store.load("test_results")
    for prompt in prompts:
        for case in cases:
            exists = [
                result
                for result in all_results
                if result.get("prompt_id") == prompt.get("prompt_id") and result.get("case_id") == case.get("case_id")
            ]
            if exists:
                print(f"已存在测试结果，跳过：{prompt['title']} -> {case['title']}")
                continue
            print(f"\n测试：{prompt['title']} -> {case['title']}")
            result = run_test(project["project_id"], prompt, case, llm)
            store.add("test_results", result)
            all_results.append(result)
            print(f"已保存测试结果：{result['result_id']}")


def handle_no_test_cases(project: dict, prompts: list[dict]) -> dict | None:
    print("当前项目还没有测试案例。")
    print("请选择：")
    print("1. 根据项目需求自动生成一个测试案例并运行")
    print("2. 手动新增测试案例")
    print("0. 返回")
    choice = input("请输入编号: ").strip()
    if choice == "0":
        return None
    if choice == "2":
        return save_manual_test_case(project["project_id"])
    if choice != "1":
        print("无效选项。")
        return None
    current_version = max(int(prompt.get("version", 1)) for prompt in prompts) if prompts else 1
    case = build_auto_test_case(project, current_version)
    store.add("test_cases", case)
    print_box("已自动生成测试案例")
    print(f"标题：{case['title']}\n\n测试输入：\n{case['input']}\n\n期望关注点：\n{case['expected_focus']}")
    return case


def choose_scoring_mode() -> str | None:
    print_box("给测试结果评分 / 筛选提示词")
    print("推荐：2. 使用文档规则评分模式")
    print("1. 使用通用评分规则")
    print("2. 使用文档规则评分模式")
    print("0. 返回")
    choice = input("请输入编号: ").strip()
    if choice == "0":
        return None
    if choice == "1":
        return GENERAL_MODE
    if choice == "2":
        return DOCUMENT_RULES_MODE
    print("无效选项。")
    return None


def score_test_results() -> None:
    scoring_mode = choose_scoring_mode()
    if not scoring_mode:
        return
    project = choose_project()
    if not project:
        return
    results = store.filter("test_results", project_id=project["project_id"])
    if not results:
        print("暂无测试结果，请先运行提示词测试。")
        return
    score_items = store.load("scores")
    for result in results:
        already = [
            item
            for item in score_items
            if item.get("prompt_id") == result.get("prompt_id")
            and item.get("case_id") == result.get("case_id")
            and str(item.get("scoring_mode", GENERAL_MODE) or GENERAL_MODE) == scoring_mode
        ]
        if already:
            continue
        prompt = store.get("prompts", "prompt_id", result["prompt_id"])
        case = store.get("test_cases", "case_id", result["case_id"])
        if not prompt or not case:
            continue
        print_box(f"评分：{prompt['title']} -> {case['title']}", result["output"])
        score = score_result(project, prompt, case, result, llm, scoring_mode=scoring_mode)
        store.add("scores", score)
        score_items.append(score)
        print(f"已保存评分：{score['total_score']} [{score.get('score_id', '未知ID')}] | mode={scoring_mode}")


def score_results_for_prompt(project: dict, prompt: dict, scoring_mode: str = DOCUMENT_RULES_MODE) -> list[dict]:
    results = [item for item in store.filter("test_results", project_id=project["project_id"]) if item.get("prompt_id") == prompt.get("prompt_id")]
    if not results:
        return []
    created_scores: list[dict] = []
    score_items = store.load("scores")
    for result in results:
        already = [
            item
            for item in score_items
            if item.get("prompt_id") == result.get("prompt_id")
            and item.get("case_id") == result.get("case_id")
            and str(item.get("scoring_mode", GENERAL_MODE) or GENERAL_MODE) == scoring_mode
        ]
        if already:
            continue
        case = store.get("test_cases", "case_id", result["case_id"])
        if not case:
            continue
        print_box(f"评分：{prompt['title']} -> {case['title']}", result["output"])
        score = score_result(project, prompt, case, result, llm, scoring_mode=scoring_mode)
        store.add("scores", score)
        score_items.append(score)
        created_scores.append(score)
        print(f"已保存评分：{score['total_score']} [{score.get('score_id', '未知ID')}] | mode={scoring_mode}")
    return created_scores


def show_best_prompt_for_project(project: dict) -> dict | None:
    prompts = store.filter("prompts", project_id=project["project_id"])
    scores = store.filter("scores", project_id=project["project_id"])
    results = store.filter("test_results", project_id=project["project_id"])
    if prompts:
        print_box("候选提示词评估状态", build_prompt_status_report(project, prompts, results, scores))
    best, averages, scoring_mode = select_best(prompts, scores)
    if not best:
        if scoring_mode == DOCUMENT_RULES_MODE and any(str(item.get("scoring_mode", GENERAL_MODE) or GENERAL_MODE) == DOCUMENT_RULES_MODE for item in scores):
            print_box("当前最优提示词", get_no_qualified_prompt_message(scores))
        else:
            print("暂无可选最优提示词，请先完成测试和评分。")
        print_unscored_version_reminder(prompts, results, scores)
        return None
    mode_scores = filter_scores_by_mode(scores, scoring_mode)
    print_box("当前最优提示词", explain_best(best, averages, mode_scores, scoring_mode))
    print(best["full_prompt"])
    fusion_prompts = [item for item in prompts if str(item.get("prompt_type", item.get("kind", ""))) == "fusion_optimized"]
    if fusion_prompts:
        fusion_prompt = max(fusion_prompts, key=lambda item: int(item.get("version", 0)))
        fusion_id = fusion_prompt.get("prompt_id", "")
        fusion_scores = [item for item in mode_scores if item.get("prompt_id") == fusion_id]
        if fusion_prompt.get("prompt_id") != best.get("prompt_id"):
            if not fusion_scores:
                print("说明：D版融合优化型尚未评分，暂不参与最优选择。")
            else:
                fusion_avg = averages.get(fusion_id, 0)
                best_avg = averages.get(best.get("prompt_id", ""), 0)
                weakness_note = ""
                if scoring_mode == DOCUMENT_RULES_MODE:
                    fusion_dimensions = aggregate_document_dimensions(fusion_scores)
                    weakness_items = fusion_dimensions.get("weakest", [])[:2]
                    if weakness_items:
                        weakness_note = " 它当前更弱的维度是：" + "、".join(format_dimension_item(item) for item in weakness_items) + "。"
                print(f"说明：D版融合优化型当前未被选中，因为它的分数为 {fusion_avg}，低于当前最优的 {best_avg}。{weakness_note}")
    print_unscored_version_reminder(prompts, results, scores)
    return best


def show_best_prompt() -> dict | None:
    project = choose_project()
    if not project:
        return None
    return show_best_prompt_for_project(project)


def build_prompt_status_report(project: dict, prompts: list[dict], results: list[dict], scores: list[dict]) -> str:
    preferred_mode = get_preferred_scoring_mode(scores) if scores else DOCUMENT_RULES_MODE
    lines: list[str] = []
    for prompt in sorted(prompts, key=lambda item: (int(item.get("version", 0)), item.get("title", ""))):
        prompt_id = prompt.get("prompt_id", "未知ID")
        prompt_results = [item for item in results if item.get("prompt_id") == prompt_id]
        prompt_scores = [item for item in scores if item.get("prompt_id") == prompt_id]
        doc_scores = [item for item in prompt_scores if str(item.get("scoring_mode", GENERAL_MODE) or GENERAL_MODE) == DOCUMENT_RULES_MODE]
        general_scores = [item for item in prompt_scores if str(item.get("scoring_mode", GENERAL_MODE) or GENERAL_MODE) == GENERAL_MODE]
        preferred_scores = [item for item in prompt_scores if str(item.get("scoring_mode", GENERAL_MODE) or GENERAL_MODE) == preferred_mode]

        avg_score = round(sum(float(item.get("total_score", 0)) for item in preferred_scores) / len(preferred_scores), 2) if preferred_scores else None
        max_score = round(max((float(item.get("total_score", 0)) for item in preferred_scores), default=0), 2) if preferred_scores else None
        risk_level = "—"
        if preferred_scores:
            levels = [str(item.get("risk_level", "low")) for item in preferred_scores]
            risk_level = "high" if "high" in levels else ("medium" if "medium" in levels else "low")

        if not prompt_scores:
            participation = "D版融合优化型尚未评分，暂不参与最优选择。" if str(prompt.get("prompt_type", prompt.get("kind", ""))) == "fusion_optimized" else "未评分，暂不参与最优选择。"
        elif preferred_mode == DOCUMENT_RULES_MODE and not doc_scores:
            participation = "D版融合优化型没有文档规则评分，暂不参与文档规则最优选择。" if str(prompt.get("prompt_type", prompt.get("kind", ""))) == "fusion_optimized" else "没有文档规则评分，暂不参与文档规则最优选择。"
        elif risk_level == "high":
            participation = "risk_level = high，已排除。"
        else:
            participation = "参与最优选择。"

        lines.append(
            "\n".join(
                [
                    f"- {prompt_id} | {prompt.get('title', '')} | version {prompt.get('version', '?')} | type={prompt.get('prompt_type', prompt.get('kind', ''))}",
                    f"  test_results：{'是' if prompt_results else '否'} | scores：{'是' if prompt_scores else '否'}",
                    f"  document_rules 平均分：{format_score_average(doc_scores)} | 最高分：{format_score_max(doc_scores)} | 风险等级：{risk_level}",
                    f"  参与状态：{participation}",
                    f"  通用评分数量：{len(general_scores)} | 文档规则评分数量：{len(doc_scores)}",
                ]
            )
        )
    return "\n\n".join(lines) if lines else "当前项目还没有候选提示词。"


def format_score_average(score_items: list[dict]) -> str:
    if not score_items:
        return "—"
    return str(round(sum(float(item.get("total_score", 0)) for item in score_items) / len(score_items), 2))


def format_score_max(score_items: list[dict]) -> str:
    if not score_items:
        return "—"
    return str(round(max(float(item.get("total_score", 0)) for item in score_items), 2))


def print_unscored_version_reminder(prompts: list[dict], results: list[dict], scores: list[dict]) -> None:
    if not prompts:
        return
    highest_scored_version = max(
        (
            int(prompt.get("version", 0))
            for prompt in prompts
            if any(item.get("prompt_id") == prompt.get("prompt_id") for item in scores)
        ),
        default=0,
    )
    reminders: list[str] = []
    for prompt in prompts:
        if int(prompt.get("version", 0)) <= highest_scored_version:
            continue
        prompt_id = prompt.get("prompt_id", "未知ID")
        has_result = any(item.get("prompt_id") == prompt_id for item in results)
        has_doc_score = any(
            item.get("prompt_id") == prompt_id and str(item.get("scoring_mode", GENERAL_MODE) or GENERAL_MODE) == DOCUMENT_RULES_MODE
            for item in scores
        )
        if not has_result or not has_doc_score:
            reminders.append(prompt_id)
    if reminders:
        print_box(
            "重新评分提醒",
            "存在未评估的新版本：" + "、".join(reminders) + "。建议执行：\n5. 运行提示词测试\n6. 给测试结果评分 / 筛选提示词\n再查看最优提示词。",
        )


def ensure_project_cases_for_prompt(project: dict, prompts: list[dict]) -> list[dict]:
    cases = store.filter("test_cases", project_id=project["project_id"])
    if cases:
        return cases
    case = handle_no_test_cases(project, prompts)
    return [case] if case else []


def run_tests_for_prompts(project: dict, prompts: list[dict], cases: list[dict]) -> list[dict]:
    created_results: list[dict] = []
    all_results = store.load("test_results")
    for prompt in prompts:
        for case in cases:
            exists = [
                result
                for result in all_results
                if result.get("prompt_id") == prompt.get("prompt_id") and result.get("case_id") == case.get("case_id")
            ]
            if exists:
                continue
            print(f"\n测试：{prompt['title']} -> {case['title']}")
            result = run_test(project["project_id"], prompt, case, llm)
            store.add("test_results", result)
            all_results.append(result)
            created_results.append(result)
            print(f"已保存测试结果：{result['result_id']}")
    return created_results


def followup_after_new_prompt(project: dict, new_prompt: dict) -> None:
    run_now = prompt_optional("是否立即对新版本运行测试（yes/no）", "yes").lower()
    if run_now != "yes":
        return
    cases = ensure_project_cases_for_prompt(project, [new_prompt])
    if not cases:
        print("没有可用测试案例，已跳过自动测试。")
        return
    created_results = run_tests_for_prompts(project, [new_prompt], cases)
    if created_results:
        print_box("新版本测试完成", f"已为 {new_prompt['title']} 新增 {len(created_results)} 条测试结果。")
    else:
        print("新版本对应的测试结果已存在，本次没有新增测试。")

    score_now = prompt_optional("是否立即使用文档规则评分新版本（yes/no）", "yes").lower()
    if score_now != "yes":
        return
    created_scores = score_results_for_prompt(project, new_prompt, scoring_mode=DOCUMENT_RULES_MODE)
    if created_scores:
        print_box("新版本评分完成", f"已为 {new_prompt['title']} 新增 {len(created_scores)} 条文档规则评分。")
    else:
        print("新版本对应的文档规则评分已存在，本次没有新增评分。")

    refresh_best = prompt_optional("是否立即刷新并查看当前最优提示词（yes/no）", "yes").lower()
    if refresh_best == "yes":
        show_best_prompt_for_project(project)


def add_feedback() -> None:
    project = choose_project()
    if not project:
        return
    prompt = choose_prompt(project["project_id"], "选择要反馈的提示词")
    if not prompt:
        return
    item = store.add("feedback", collect_feedback(project["project_id"], prompt["prompt_id"]))
    print(f"已保存反馈：{item.get('feedback_id', '未知ID')}")


def optimize_next_version() -> None:
    project = choose_project()
    if not project:
        return
    prompts = store.filter("prompts", project_id=project["project_id"])
    all_scores = store.filter("scores", project_id=project["project_id"])
    best, _, scoring_mode = select_best(prompts, all_scores)
    old_prompt = best or choose_prompt(project["project_id"], "选择要优化的提示词")
    if not old_prompt:
        print("暂无提示词可优化。")
        return
    prompt_scores = [
        item
        for item in all_scores
        if item.get("prompt_id") == old_prompt.get("prompt_id") and str(item.get("scoring_mode", GENERAL_MODE) or GENERAL_MODE) == scoring_mode
    ]
    feedback_items = store.filter("feedback", project_id=project["project_id"], prompt_id=old_prompt["prompt_id"])
    iterations = store.filter("iterations", project_id=project["project_id"])
    new_prompt, iteration = optimize_from_feedback(project, old_prompt, prompt_scores, feedback_items, iterations, llm)
    store.add("prompts", new_prompt)
    store.add("iterations", iteration)
    print_box("已生成下一版提示词")
    print(f"{new_prompt['title']} [{new_prompt['prompt_id']}]\n")
    print(new_prompt["full_prompt"])
    followup_after_new_prompt(project, new_prompt)


def export_best_prompt() -> None:
    project = choose_project()
    if not project:
        return
    prompts = store.filter("prompts", project_id=project["project_id"])
    scores = store.filter("scores", project_id=project["project_id"])
    best, averages, scoring_mode = select_best(prompts, scores)
    if not best:
        print("暂无最优提示词可导出。")
        return
    path = export_prompt_to_txt(project, best, explain_best(best, averages, filter_scores_by_mode(scores, scoring_mode), scoring_mode))
    print(f"已导出：{path}")


def export_prompt_to_txt(project: dict, prompt: dict, summary: str = "") -> Path:
    export_dir = Path(__file__).resolve().parent / "exports"
    export_dir.mkdir(exist_ok=True)
    filename = f"prompt_{safe_filename(project['name'])}_{safe_filename(prompt['title'])}_{prompt['prompt_id']}.txt"
    path = export_dir / filename
    content = build_prompt_detail_text(project, prompt)
    if summary:
        content = summary + "\n\n" + content
    path.write_text(content, encoding="utf-8")
    return path


def build_prompt_detail_text(project: dict, prompt: dict) -> str:
    task_profile = build_task_profile(project)
    fusion_notes = dict(prompt.get("fusion_notes", {}) or {})
    extra_lines: list[str] = []
    source_prompt_ids = list(prompt.get("source_prompt_ids", []) or [])
    if source_prompt_ids:
        extra_lines.append(f"source_prompt_ids：{', '.join(source_prompt_ids)}")
    absorbed = fusion_notes.get("absorbed_strengths", []) or []
    avoided = fusion_notes.get("avoided_weaknesses", []) or []
    if absorbed:
        extra_lines.append(f"absorbed_strengths：{'；'.join(str(item) for item in absorbed if str(item).strip())}")
    if avoided:
        extra_lines.append(f"avoided_weaknesses：{'；'.join(str(item) for item in avoided if str(item).strip())}")
    if fusion_notes.get("main_tradeoff"):
        extra_lines.append(f"main_tradeoff：{fusion_notes.get('main_tradeoff', '')}")
    if fusion_notes.get("next_variable_to_watch"):
        extra_lines.append(f"next_variable_to_watch：{fusion_notes.get('next_variable_to_watch', '')}")
    extra_block = ("\n" + "\n".join(extra_lines)) if extra_lines else ""
    task_profile_block = f"""

【任务画像】
task_type：{task_profile.get('task_type', '')}
output_medium：{task_profile.get('output_medium', '')}
platform：{task_profile.get('platform', '')}
target_audience：{task_profile.get('target_audience', '')}
input_object：{task_profile.get('input_object', '')}
role_archetype：{task_profile.get('role_archetype', '')}
work_scene：{task_profile.get('work_scene', '')}
target_object：{task_profile.get('target_object', '')}
action_chain：{'；'.join(str(item) for item in task_profile.get('action_chain', []) if str(item).strip())}
judgment_standard：{task_profile.get('judgment_standard', '')}
risks：{'；'.join(str(item) for item in task_profile.get('risks', []) if str(item).strip())}
"""
    return f"""项目名称：{project.get('name', '')}
版本：{prompt.get('version', '')}
提示词ID：{prompt.get('prompt_id', '未知ID')}
标题：{prompt.get('title', '')}
类型：{prompt.get('prompt_type', prompt.get('kind', ''))}
适合：{prompt.get('suitable_for', '')}
不适合：{prompt.get('not_suitable_for', '')}
风险：{prompt.get('risk', '')}
使用建议：{prompt.get('usage_advice', '')}{extra_block}
{task_profile_block}

【完整提示词】
{prompt.get('full_prompt', '')}
"""


def view_candidate_prompt_content() -> None:
    project = choose_project()
    if not project:
        return
    prompts = sorted(store.filter("prompts", project_id=project["project_id"]), key=lambda item: (int(item.get("version", 0)), item.get("title", "")))
    if not prompts:
        print("当前项目还没有候选提示词。")
        return

    while True:
        print_box("候选提示词列表")
        grouped: dict[int, list[dict]] = defaultdict(list)
        for prompt in prompts:
            grouped[int(prompt.get("version", 0))].append(prompt)
        flat_list: list[dict] = []
        index = 1
        for version in sorted(grouped):
            print(f"version {version}")
            for prompt in grouped[version]:
                print(f"{index}. {prompt['title']} [{prompt['prompt_id']}]")
                flat_list.append(prompt)
                index += 1
        raw = input("请输入编号（0 返回主菜单）: ").strip()
        if raw == "0":
            return
        if not raw.isdigit() or not (1 <= int(raw) <= len(flat_list)):
            print("编号无效，请重试。")
            continue
        show_prompt_detail(project, flat_list[int(raw) - 1])


def show_prompt_detail(project: dict, prompt: dict) -> None:
    while True:
        print_box("提示词详情", build_prompt_detail_text(project, prompt))
        print("1. 返回列表")
        print("2. 导出该提示词为 txt")
        print("0. 返回主菜单")
        choice = input("请输入编号: ").strip()
        if choice == "1":
            return
        if choice == "2":
            path = export_prompt_to_txt(project, prompt)
            print(f"已导出：{path}")
            continue
        if choice == "0":
            raise BackToMenu
        print("无效选项，请重试。")


def generate_project_test_cases() -> None:
    project = choose_project()
    if not project:
        return
    cases = generate_test_cases(project, llm, max_cases=3)
    if not cases:
        print("没有生成到测试案例。")
        return

    print_box("生成的测试案例")
    for idx, case in enumerate(cases, start=1):
        print(f"{idx}. {case['title']} [{case['case_id']}]")
        print(f"测试输入：\n{case['input']}")
        print(f"期望关注点：\n{case['expected_focus']}\n")

    print("1. 保存全部")
    print("2. 逐个确认保存")
    print("0. 放弃")
    choice = input("请输入编号: ").strip()
    if choice == "0":
        print("已放弃保存。")
        return
    if choice == "1":
        for case in cases:
            store.add("test_cases", case)
        print(f"已保存 {len(cases)} 个测试案例。")
        return
    if choice == "2":
        saved = 0
        for case in cases:
            print_box(case["title"], f"测试输入：\n{case['input']}\n\n期望关注点：\n{case['expected_focus']}")
            confirm = prompt_optional("是否保存该案例（yes/no）", "yes").lower()
            if confirm == "yes":
                store.add("test_cases", case)
                saved += 1
        print(f"已保存 {saved} 个测试案例。")
        return
    print("无效选项。")


def test_llm_connection() -> None:
    print_box("大模型连接测试")
    print(f"provider: {LLM_PROVIDER}")
    print(f"model: {current_model_label()}")
    print(f"检测到 API Key: {'是' if bool(DEEPSEEK_API_KEY) else '否'}")
    print(f"当前手动模式: {'是' if MANUAL_MODE or not llm.is_available() else '否'}")
    ok = llm.health_check()
    if ok:
        print("\n大模型连接成功。")
        return
    print("\n大模型连接失败。")
    if not DEEPSEEK_API_KEY:
        reason = "没有检测到 DEEPSEEK_API_KEY，程序已自动进入手动模式。"
    elif not llm.is_available():
        reason = "DeepSeek 客户端未成功初始化，当前无法调用大模型。"
    else:
        reason = "已发起健康检查，但接口没有返回可用结果。"
    print(f"原因：{reason}")
    print("解决建议：")
    print("- 检查 DEEPSEEK_API_KEY 是否设置")
    print("- 检查网络")
    print("- 检查模型名是否正确")
    print("- 检查账户额度或权限")


def show_data_stats() -> None:
    stats = get_data_stats()
    global_stats = stats["global"]
    print_box("数据统计")
    print("全局统计：")
    print(f"- 项目数量：{global_stats['projects']}")
    print(f"- 提示词数量：{global_stats['prompts']}")
    print(f"- 测试案例数量：{global_stats['test_cases']}")
    print(f"- 测试结果数量：{global_stats['test_results']}")
    print(f"- 评分数量：{global_stats['scores']}")
    print(f"- 反馈数量：{global_stats['feedback']}")
    print(f"- 迭代记录数量：{global_stats['iterations']}")
    print(f"- 备份数量：{global_stats['backups']}")
    print("\n按项目统计：")
    for item in stats["projects"]:
        print(
            f"- {item['name']} [{item['project_id']}] | prompts={item['prompts']} | "
            f"test_cases={item['test_cases']} | test_results={item['test_results']} | "
            f"scores={item['scores']} | feedback={item['feedback']} | iterations={item['iterations']}"
        )


def print_deleted_counts(counts: dict) -> None:
    for key, value in counts.items():
        if value:
            print(f"- {key}: {value} 条")


def confirm_delete_prompt() -> None:
    project = choose_project()
    if not project:
        return
    prompt = choose_prompt(project["project_id"], "选择要删除的提示词")
    if not prompt:
        return
    preview = preview_delete_prompt(prompt.get("prompt_id", ""))
    if not preview["success"]:
        print(preview["message"])
        return
    print_box("删除提示词确认")
    print(f"你将删除提示词：\n{prompt.get('prompt_id', '未知ID')}\n标题：{prompt.get('title', '')}\n所属项目：{project.get('name', '')}")
    print("\n此操作会同时删除关联的：")
    print_deleted_counts(preview["deleted_counts"])
    if input("\n请输入 DELETE 确认删除：").strip() != "DELETE":
        print("已取消删除。")
        return
    result = delete_prompt(prompt.get("prompt_id", ""))
    print_delete_result(result)


def confirm_delete_case() -> None:
    project = choose_project()
    if not project:
        return
    case = choose_case(project["project_id"], "选择要删除的测试案例")
    if not case:
        return
    preview = preview_delete_test_case(case.get("case_id", ""))
    if not preview["success"]:
        print(preview["message"])
        return
    print_box("删除测试案例确认")
    print(f"你将删除测试案例：\n{case.get('case_id', '未知ID')}\n标题：{case.get('title', '')}\n所属项目：{project.get('name', '')}")
    print("\n此操作会同时删除关联的：")
    print_deleted_counts(preview["deleted_counts"])
    if input("\n请输入 DELETE 确认删除：").strip() != "DELETE":
        print("已取消删除。")
        return
    result = delete_test_case(case.get("case_id", ""))
    print_delete_result(result)


def confirm_delete_result() -> None:
    project = choose_project()
    if not project:
        return
    result_item = choose_result(project["project_id"], "选择要删除的测试结果")
    if not result_item:
        return
    preview = preview_delete_test_result(result_item.get("result_id", ""))
    if not preview["success"]:
        print(preview["message"])
        return
    print_box("删除测试结果确认")
    print(f"你将删除测试结果：\n{result_item.get('result_id', '未知ID')}\n所属项目：{project.get('name', '')}")
    print("\n此操作会同时删除关联的：")
    print_deleted_counts(preview["deleted_counts"])
    if input("\n请输入 DELETE 确认删除：").strip() != "DELETE":
        print("已取消删除。")
        return
    result = delete_test_result(result_item.get("result_id", ""))
    print_delete_result(result)


def confirm_delete_score() -> None:
    project = choose_project()
    if not project:
        return
    score = choose_score(project["project_id"], "选择要删除的评分结果")
    if not score:
        return
    preview = preview_delete_simple("scores", "score_id", score.get("score_id", ""))
    if not preview["success"]:
        print(preview["message"])
        return
    print_box("删除评分结果确认")
    print(f"你将删除评分结果：\n{score.get('score_id', '未知ID')}\n所属项目：{project.get('name', '')}")
    if input("\n请输入 DELETE 确认删除：").strip() != "DELETE":
        print("已取消删除。")
        return
    result = delete_score(score.get("score_id", ""))
    print_delete_result(result)


def confirm_delete_feedback() -> None:
    project = choose_project()
    if not project:
        return
    item = choose_feedback(project["project_id"], "选择要删除的反馈记录")
    if not item:
        return
    preview = preview_delete_simple("feedback", "feedback_id", item.get("feedback_id", ""))
    if not preview["success"]:
        print(preview["message"])
        return
    print_box("删除反馈记录确认")
    print(f"你将删除反馈记录：\n{item.get('feedback_id', '未知ID')}\n所属项目：{project.get('name', '')}")
    if input("\n请输入 DELETE 确认删除：").strip() != "DELETE":
        print("已取消删除。")
        return
    result = delete_feedback(item.get("feedback_id", ""))
    print_delete_result(result)


def confirm_delete_iteration() -> None:
    project = choose_project()
    if not project:
        return
    item = choose_iteration(project["project_id"], "选择要删除的迭代记录")
    if not item:
        return
    preview = preview_delete_simple("iterations", "iteration_id", item.get("iteration_id", ""))
    if not preview["success"]:
        print(preview["message"])
        return
    print_box("删除迭代记录确认")
    print(f"你将删除迭代记录：\n{item.get('iteration_id', '未知ID')}\n所属项目：{project.get('name', '')}")
    if input("\n请输入 DELETE 确认删除：").strip() != "DELETE":
        print("已取消删除。")
        return
    result = delete_iteration(item.get("iteration_id", ""))
    print_delete_result(result)


def confirm_project_cleanup(action_key: str, action_fn) -> None:
    project = choose_project()
    if not project:
        return
    preview = preview_project_action(project.get("project_id", ""), action_key)
    print_box("批量清理确认")
    print(f"项目：{project.get('name', '')} [{project.get('project_id', '未知ID')}]")
    print("将删除：")
    print_deleted_counts(preview["deleted_counts"])
    if input("\n请输入项目名称以确认清理：").strip() != str(project.get("name", "")):
        print("项目名称不匹配，已取消。")
        return
    result = action_fn(project.get("project_id", ""))
    print_delete_result(result)


def confirm_delete_project() -> None:
    project = choose_project()
    if not project:
        return
    preview = preview_project_action(project.get("project_id", ""), "delete_project")
    print_box("删除整个项目确认")
    print(f"项目：{project.get('name', '')} [{project.get('project_id', '未知ID')}]")
    print("将删除以下数据：")
    print_deleted_counts(preview["deleted_counts"])
    if input("\n请输入 DELETE_ALL 确认批量删除：").strip() != "DELETE_ALL":
        print("已取消删除。")
        return
    result = delete_project_cascade(project.get("project_id", ""))
    print_delete_result(result)


def print_delete_result(result: dict) -> None:
    if not result.get("success"):
        print(f"操作失败：{result.get('message', '未知错误')}")
        return
    print_box("删除 / 清理完成")
    print(result.get("message", "操作完成。"))
    print("删除数量：")
    print_deleted_counts(result.get("deleted_counts", {}))
    if result.get("backup_path"):
        print(f"\n备份路径：{result['backup_path']}")


def manage_data_menu() -> None:
    while True:
        print_box("数据管理 / 删除清理")
        print(
            """1. 删除某个提示词
2. 删除某个测试案例
3. 删除某个测试结果
4. 删除某个评分结果
5. 删除某条反馈记录
6. 删除某条迭代记录
7. 清理某个项目下的全部提示词
8. 清理某个项目下的全部测试案例
9. 清理某个项目下的全部测试结果
10. 清理某个项目下的全部评分结果
11. 清理某个项目下的全部反馈和迭代记录
12. 清理某个项目下的所有生成数据，但保留项目本身
13. 删除整个项目及其关联数据
14. 查看数据统计
0. 返回主菜单"""
        )
        choice = input("请输入编号: ").strip()
        actions = {
            "1": confirm_delete_prompt,
            "2": confirm_delete_case,
            "3": confirm_delete_result,
            "4": confirm_delete_score,
            "5": confirm_delete_feedback,
            "6": confirm_delete_iteration,
            "7": lambda: confirm_project_cleanup("clear_prompts", clear_project_prompts),
            "8": lambda: confirm_project_cleanup("clear_test_cases", clear_project_test_cases),
            "9": lambda: confirm_project_cleanup("clear_results", clear_project_results),
            "10": lambda: confirm_project_cleanup("clear_scores", clear_project_scores),
            "11": lambda: confirm_project_cleanup("clear_feedback_iterations", clear_project_feedback_iterations),
            "12": lambda: confirm_project_cleanup("clear_generated_data", clear_project_generated_data),
            "13": confirm_delete_project,
            "14": show_data_stats,
        }
        if choice == "0":
            return
        action = actions.get(choice)
        if action:
            action()
        else:
            print("无效选项，请重新输入。")

def choose_idea_request(title: str = "选择想法请求") -> dict | None:
    requests = sorted(store.load("idea_requests"), key=lambda item: item.get("created_at", ""), reverse=True)
    label_items = []
    for item in requests:
        copy = dict(item)
        copy["_label"] = f"{item.get('topic', '')} | {item.get('request_id', '')}"
        label_items.append(copy)
    return choose_from_list(label_items, "_label", title, "request_id")


def choose_prompt_pack(title: str = "选择 Prompt 包") -> dict | None:
    packs = PromptPackRegistry(store).list_packs()
    label_items = []
    for item in packs:
        copy = dict(item)
        copy["_label"] = f"{item.get('pack_name', '')} | {item.get('status', '')} | score={item.get('quality_score', 0)}"
        label_items.append(copy)
    return choose_from_list(label_items, "_label", title, "pack_id")


def choose_prompt_image_feedback(title: str = "选择 Prompt 图片反馈") -> dict | None:
    items = sorted(store.load("prompt_image_feedback"), key=lambda item: item.get("created_at", ""), reverse=True)
    label_items = []
    for item in items:
        copy = dict(item)
        copy["_label"] = f"{item.get('prompt_id', '')} | {item.get('request_id', '')} | {item.get('feedback_id', '')}"
        label_items.append(copy)
    return choose_from_list(label_items, "_label", title, "feedback_id")


def choose_prompt_pack_cold_start_fallback(request: dict) -> dict | None:
    registry = PromptPackRegistry(store)
    allowed_statuses = {"candidate", "approved", "testing", "draft"}
    packs = [pack for pack in registry.list_packs(include_archived=False) if pack.get("status") in allowed_statuses]
    if not packs:
        print("当前 Prompt 包资产库中没有可手动选择的 Prompt 包。")
        return None

    print_box("没有找到高匹配 Prompt 包")
    print("没有找到高匹配 Prompt 包。当前资产库较少，你可以手动选择一个近似 Prompt 包继续测试。")
    print("可选 Prompt 包：")
    label_items = []
    for pack in packs:
        match_tags = ", ".join(str(tag) for tag in pack.get("match_tags", []))
        print(
            f"- pack_id: {pack.get('pack_id', '')}\n"
            f"  pack_name: {pack.get('pack_name', '')}\n"
            f"  domain: {pack.get('domain', '')}\n"
            f"  platform: {pack.get('platform', '')}\n"
            f"  use_case: {pack.get('use_case', '')}\n"
            f"  style: {pack.get('style', '')}\n"
            f"  status: {pack.get('status', '')}\n"
            f"  match_tags: {match_tags}"
        )
        copy = dict(pack)
        copy["_label"] = (
            f"{pack.get('pack_name', '')} | {pack.get('domain', '')}/{pack.get('platform', '')}/"
            f"{pack.get('use_case', '')} | {pack.get('style', '')} | {pack.get('status', '')}"
        )
        label_items.append(copy)

    selected = choose_from_list(label_items, "_label", "手动选择近似 Prompt 包（回车返回）", "pack_id")
    if not selected:
        return None

    log = create_prompt_pack_match_log(
        {
            "request_id": request.get("request_id", ""),
            "matched": False,
            "pack_id": selected.get("pack_id", ""),
            "match_score": 0,
            "match_reasons": ["manual selected from cold-start fallback"],
            "missing_requirements": [],
            "decision": "manual_selected_pack",
            "manual_selected": True,
        }
    )
    store.add("prompt_pack_match_logs", log)
    print(f"已手动选择 Prompt 包：{selected.get('pack_name', '')} [{selected.get('pack_id', '')}]")
    return selected


def current_provider_profile_label() -> str:
    return current_provider_profile_path or "默认 / Mock Provider 或 IMAGE_PROVIDER"


def pack_workflow_choose_provider_profile() -> None:
    global current_provider_profile_path
    profile_dir = Path(__file__).resolve().parent / "provider_profiles"
    profiles = sorted(profile_dir.glob("*.json")) if profile_dir.exists() else []
    print_box("请选择生图 Provider Profile")
    print("0. 使用默认 / Mock Provider")
    for index, path in enumerate(profiles, start=1):
        print(f"{index}. {path.relative_to(Path(__file__).resolve().parent)}")
    raw = input("请输入选项：").strip()
    if raw == "0":
        current_provider_profile_path = ""
        print("当前生图配置：默认 / Mock Provider 或 IMAGE_PROVIDER")
        return
    if raw.isdigit() and 1 <= int(raw) <= len(profiles):
        current_provider_profile_path = str(profiles[int(raw) - 1])
        print(f"当前生图配置：{current_provider_profile_path}")
        return
    print("无效选项。")


def pack_workflow_analyze_idea() -> None:
    print_box("Prompt 包工作流 - 分析新想法")
    raw_idea = prompt_multiline("请输入想法")
    if not raw_idea:
        return
    request = analyze_idea(raw_idea, store=store)
    print_box("已创建想法请求")
    for key in ["request_id", "domain", "platform", "use_case", "style", "ratio", "topic", "batch_size"]:
        print(f"{key}: {request.get(key, '')}")


def pack_workflow_match_pack() -> None:
    request = choose_idea_request()
    if not request:
        return
    match = match_prompt_pack(request, store=store)
    print_box("Prompt 包匹配结果")
    print(f"决策：{match['decision']}")
    print(f"是否匹配：{match['matched']}")
    print(f"pack_id：{match.get('pack_id', '')}")
    print(f"匹配分：{match['match_score']}")
    print(f"匹配原因：{', '.join(match.get('match_reasons', []))}")
    print(f"缺失条件：{', '.join(match.get('missing_requirements', []))}")


    if match.get("decision") != "use_existing_pack":
        choose_prompt_pack_cold_start_fallback(request)


def pack_workflow_run_matched_pack() -> None:
    request = choose_idea_request()
    if not request:
        return
    matches = [
        item for item in store.load("prompt_pack_match_logs")
        if item.get("request_id") == request.get("request_id") and item.get("decision") in {"use_existing_pack", "manual_selected_pack"}
    ]
    pack_id = matches[-1]["pack_id"] if matches else ""
    if not pack_id:
        pack = choose_prompt_pack("没有找到匹配的 Prompt 包，请手动选择 Prompt 包")
        if not pack:
            return
        pack_id = pack["pack_id"]
    result = run_prompt_pack(pack_id, request, store=store, provider_profile_path=current_provider_profile_path or None)
    print_box("已生成图片结果")
    print(f"pack_id：{pack_id}")
    print(f"任务数：{len(result['tasks'])}")
    print(f"图片结果数：{len(result['image_results'])}")


def pack_workflow_generate_candidate_prompts() -> None:
    request = choose_idea_request()
    if not request:
        return
    project_data = {
        "name": f"Pack incubation - {request.get('topic', request.get('request_id', 'idea'))}"[:80],
        "scenario": request.get("raw_idea", ""),
        "goal": f"Generate image prompts for {request.get('use_case', 'image')}",
        "audience": request.get("target_audience", ""),
        "desired_output": f"{request.get('batch_size', 10)} mock image candidates, ratio {request.get('ratio', '1:1')}",
        "avoid": "; ".join(request.get("constraints", [])),
        "style_preference": request.get("style", ""),
        "evaluation_criteria": "image review score, human keep rate, safety",
    }
    project = store.add("projects", make_project(project_data))
    version = next_version(store.load("prompts"), project["project_id"])
    candidates = generate_candidates(project, version, llm)
    for item in candidates:
        item["source_request_id"] = request.get("request_id", "")
        store.add("prompts", item)
    print_box("已生成候选 Prompt")
    print(f"project_id：{project['project_id']}")
    print(f"request_id：{request['request_id']}")
    print(f"Prompt 数：{len(candidates)}")


def pack_workflow_test_candidate_prompts() -> None:
    request = choose_idea_request()
    if not request:
        return
    prompts = [item for item in store.load("prompts") if item.get("source_request_id") == request.get("request_id")]
    if not prompts:
        project = choose_project()
        if not project:
            return
        prompts = store.filter("prompts", project_id=project["project_id"])
    tasks, results = run_prompts_for_request(
        request,
        prompts,
        count_per_prompt=1,
        store=store,
        provider_profile_path=current_provider_profile_path or None,
    )
    print_box("候选 Prompt 图片测试完成")
    print(f"任务数：{len(tasks)}")
    print(f"图片结果数：{len(results)}")


def pack_workflow_auto_review() -> None:
    request = choose_idea_request()
    if not request:
        return
    results = [item for item in store.load("image_results") if item.get("request_id") == request.get("request_id")]
    reviews = review_image_results(results, store=store)
    print_box("已自动审核图片")
    print(f"图片审核记录数：{len(reviews)}")


def pack_workflow_export_human_csv() -> None:
    request = choose_idea_request()
    if not request:
        return
    path = export_human_review_csv(request.get("request_id", ""), store=store)
    print(f"已导出人工审核 CSV：{path}")


def pack_workflow_import_human_csv() -> None:
    path = prompt_required("CSV 路径")
    imported = import_human_review_csv(path, store=store)
    print(f"已导入人工审核 CSV：{len(imported)} 条")


def pack_workflow_build_feedback() -> None:
    request = choose_idea_request()
    if not request:
        return
    feedback_items = build_prompt_image_feedback(request.get("request_id", ""), store=store)
    print_box("已生成 Prompt 反馈")
    print(f"Prompt 反馈数：{len(feedback_items)}")


def pack_workflow_optimize_prompt() -> None:
    feedback = choose_prompt_image_feedback()
    if not feedback:
        return
    new_prompt, iteration = optimize_prompt_from_image_feedback(feedback["prompt_id"], feedback["feedback_id"], store=store, llm=llm)
    print_box("已根据图片反馈优化 Prompt")
    print(f"from_prompt_id：{feedback['prompt_id']}")
    print(f"to_prompt_id：{new_prompt['prompt_id']}")
    print(f"iteration_id：{iteration['iteration_id']}")


def pack_workflow_build_pack() -> None:
    request = choose_idea_request()
    if not request:
        return
    result = build_pack_from_prompt_results(request, store=store)
    print_box("已打包 Prompt 包")
    print(f"pack_id：{result['pack']['pack_id']}")
    print(f"状态：{result['pack']['status']}")
    print(f"条目数：{len(result['items'])}")


def pack_workflow_list_packs() -> None:
    registry = PromptPackRegistry(store)
    packs = registry.list_packs(include_archived=True)
    print_box("Prompt 包资产库")
    if not packs:
        print("暂无 Prompt 包。")
        return
    for pack in packs:
        print(
            f"- {pack.get('pack_name', '')} [{pack.get('pack_id', '')}] "
            f"status={pack.get('status', '')} score={pack.get('quality_score', 0)} "
            f"{pack.get('domain', '')}/{pack.get('platform', '')}/{pack.get('use_case', '')}"
        )


def pack_workflow_run_specific_pack() -> None:
    request = choose_idea_request()
    if not request:
        return
    pack = choose_prompt_pack()
    if not pack:
        return
    registry = PromptPackRegistry(store)
    total_item_count = len(registry.get_items(pack["pack_id"]))
    prompt_selection = choose_pack_prompt_selection(pack["pack_id"], total_item_count)
    if prompt_selection is None:
        return
    selected_count = len(run_prompt_pack_preview_items(pack["pack_id"], prompt_selection))
    if selected_count <= 0:
        print("没有找到符合范围的 prompt items，请重新选择。")
        return
    if selected_count > 10 and not is_mock_image_provider():
        confirm = input(f"当前选择将生成 {selected_count} 张图片，可能产生费用。请输入 YES 确认继续：").strip()
        if confirm != "YES":
            print("已取消生成。")
            return
    result = run_prompt_pack(
        pack["pack_id"],
        request,
        store=store,
        provider_profile_path=current_provider_profile_path or None,
        prompt_selection=prompt_selection,
    )
    print_box("指定 Prompt 包生成图片完成")
    print(f"run_id：{result['run']['run_id']}")
    print(f"总 Prompt 数：{result['summary']['total_item_count']}")
    print(f"本次选择：{result['summary']['selected_item_count']}")
    print(f"生成任务数：{result['summary']['generated_task_count']}")
    print(f"图片结果数：{len(result['image_results'])}")


def run_prompt_pack_preview_items(pack_id: str, prompt_selection: dict) -> list[dict]:
    from prompt_pack_runner import select_prompt_pack_items

    registry = PromptPackRegistry(store)
    return select_prompt_pack_items(registry.get_items(pack_id), prompt_selection)


def choose_pack_prompt_selection(pack_id: str, total_item_count: int) -> dict | None:
    print_box("请选择生成范围")
    print(f"当前 Prompt 包共有 {total_item_count} 条 prompt items。")
    print("1. 全部 Prompts")
    print("2. 前 N 条 Prompts（推荐）")
    print("3. 指定 prompt_id 范围")
    print("4. 指定 prompt_id 列表")
    raw = input("请选择生成范围（回车默认前 3 条）：").strip()
    if not raw:
        return {"mode": "first_n", "n": 3}
    if raw == "1":
        if not is_mock_image_provider():
            confirm = input(f"当前选择将生成全部 {total_item_count} 张图片，可能产生费用。请输入 YES 确认继续：").strip()
            if confirm != "YES":
                print("已取消生成。")
                return None
        return {"mode": "all"}
    if raw == "2":
        n_raw = input("请输入 N（默认 3）：").strip()
        try:
            n = int(n_raw or "3")
        except ValueError:
            n = 3
        return {"mode": "first_n", "n": max(1, n)}
    if raw == "3":
        range_raw = input("请输入 prompt_id 范围，例如 P001-P010：").strip()
        if "-" not in range_raw:
            print("范围格式无效。")
            return None
        start, end = [part.strip() for part in range_raw.split("-", 1)]
        return {"mode": "range", "start": start, "end": end}
    if raw == "4":
        list_raw = input("请输入 prompt_id 列表，例如 P001,P003,P007：").strip()
        prompt_ids = [item.strip() for item in list_raw.split(",") if item.strip()]
        return {"mode": "list", "prompt_ids": prompt_ids}
    print("无效选项。")
    return None


def is_mock_image_provider() -> bool:
    return not current_provider_profile_path and IMAGE_PROVIDER == "mock"


def pack_workflow_delete_prompt_pack() -> None:
    registry = PromptPackRegistry(store)
    packs = registry.list_packs(include_archived=True)
    if not packs:
        print("暂无 Prompt 包。")
        return
    pack = choose_prompt_pack("选择要删除的 Prompt 包")
    if not pack:
        return
    item_count = len(registry.get_items(pack["pack_id"]))
    print_box("删除 Prompt 包确认")
    print(f"pack_id: {pack.get('pack_id', '')}")
    print(f"pack_name: {pack.get('pack_name', '')}")
    print(f"status: {pack.get('status', '')}")
    print(f"item_count: {item_count}")
    if item_count == 0:
        print("该 Prompt 包没有对应 prompt items，可能是导入不完整。")
    confirm = input("确认删除该 Prompt 包及其 prompt items？请输入 YES：").strip()
    if confirm != "YES":
        print("已取消删除。")
        return
    result = delete_prompt_pack(pack["pack_id"], store)
    print_box("Prompt 包删除完成")
    print(result)


def prompt_pack_workflow_menu() -> None:
    actions = {
        "1": pack_workflow_analyze_idea,
        "2": pack_workflow_match_pack,
        "3": pack_workflow_run_matched_pack,
        "4": pack_workflow_generate_candidate_prompts,
        "5": pack_workflow_test_candidate_prompts,
        "6": pack_workflow_auto_review,
        "7": pack_workflow_export_human_csv,
        "8": pack_workflow_import_human_csv,
        "9": pack_workflow_build_feedback,
        "10": pack_workflow_optimize_prompt,
        "11": pack_workflow_build_pack,
        "12": pack_workflow_list_packs,
        "13": pack_workflow_run_specific_pack,
        "14": pack_workflow_choose_provider_profile,
        "15": pack_workflow_delete_prompt_pack,
    }
    while True:
        print_box("Prompt 包工作流")
        print(f"当前生图配置：{current_provider_profile_label()}")
        print(
            """1. 分析新想法
2. 匹配已有 Prompt 包
3. 使用匹配到的 Prompt 包生成图片
4. 没有合适 Prompt 包时生成候选 Prompt
5. 使用候选 Prompt 进行图片测试
6. 自动审核图片
7. 导出人工审核 CSV
8. 导入人工审核 CSV
9. 根据图片审核结果生成 Prompt 反馈
10. 根据图片反馈优化 Prompt
11. 将达标 Prompt 打包成 Prompt 包
12. 查看 Prompt 包资产库
13. 指定 Prompt 包生成图片
14. 选择当前生图 Provider Profile
15. 删除 Prompt 包
0. 返回上一级"""
        )
        choice = input("请输入选项：").strip()
        if choice == "0":
            return
        action = actions.get(choice)
        if action:
            action()
        else:
            print("无效选项。")


def menu() -> None:
    while True:
        try:
            print_box(f"欢迎使用提示词优化 Agent | 当前：{mode_label()} | 模型：{current_model_label()}")
            print(
                """请选择操作：
1. 创建新优化项目
2. 查看已有项目
3. 为项目生成候选提示词
4. 添加测试案例
5. 运行提示词测试
6. 给测试结果评分 / 筛选提示词
7. 查看当前最优提示词
8. 录入真实使用反馈
9. 基于反馈生成下一版提示词
10. 导出最优提示词为 txt
11. 查看候选提示词内容
12. 根据项目需求生成测试案例
13. 测试大模型连接
14. 数据管理 / 删除清理
15. 基于 A/B/C 生成融合优化版
16. Prompt 包工作流
0. 退出"""
            )
            choice = input("请输入编号: ").strip()
            actions = {
                "1": create_project,
                "2": list_projects,
                "3": generate_project_prompts,
                "4": add_test_case,
                "5": run_prompt_tests,
                "6": score_test_results,
                "7": show_best_prompt,
                "8": add_feedback,
                "9": optimize_next_version,
                "10": export_best_prompt,
                "11": view_candidate_prompt_content,
                "12": generate_project_test_cases,
                "13": test_llm_connection,
                "14": manage_data_menu,
                "15": generate_fusion_prompt_for_project,
                "16": prompt_pack_workflow_menu,
            }
            if choice == "0":
                print("已退出。")
                break
            action = actions.get(choice)
            if action:
                action()
            else:
                print("无效选项，请重新输入。")
        except BackToMenu:
            continue


if __name__ == "__main__":
    menu()
