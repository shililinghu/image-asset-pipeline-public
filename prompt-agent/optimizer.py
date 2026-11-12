from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from identity_builder import build_identity_intro, build_task_profile, replace_identity_intro
from llm_client import LLMClient
from models import make_iteration, make_prompt
from prompt_generator import post_process_prompt
from scoring_rules import (
    DOCUMENT_LABEL_BY_KEY,
    DOCUMENT_RULES_MODE,
    GENERAL_MODE,
    choose_next_variable,
    compress_prompt_by_attention_budget,
    detect_project_type,
    normalize_identity_intro,
    repair_execution_clarity_after_compression,
    should_prioritize_attention_budget,
)
from storage import JsonStore


def normalize_mode(score: dict[str, Any]) -> str:
    return str(score.get("scoring_mode", GENERAL_MODE) or GENERAL_MODE)


def get_preferred_scoring_mode(scores: list[dict[str, Any]]) -> str:
    return DOCUMENT_RULES_MODE if any(normalize_mode(score) == DOCUMENT_RULES_MODE for score in scores) else GENERAL_MODE


def filter_scores_by_mode(scores: list[dict[str, Any]], scoring_mode: str) -> list[dict[str, Any]]:
    return [score for score in scores if normalize_mode(score) == scoring_mode]


def average_scores(scores: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for item in scores:
        grouped[item["prompt_id"]].append(float(item.get("total_score", 0)))
    return {prompt_id: round(sum(values) / len(values), 2) for prompt_id, values in grouped.items()}


def is_fusion_prompt(prompt: dict[str, Any]) -> bool:
    return str(prompt.get("prompt_type", prompt.get("kind", ""))) == "fusion_optimized"


def get_prompt_mode_scores(prompt_id: str, scores: list[dict[str, Any]], scoring_mode: str) -> list[dict[str, Any]]:
    return [score for score in scores if score.get("prompt_id") == prompt_id and normalize_mode(score) == scoring_mode]


def select_best(prompts: list[dict[str, Any]], scores: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, float], str]:
    preferred_mode = get_preferred_scoring_mode(scores)
    usable_scores = filter_scores_by_mode(scores, preferred_mode)

    if preferred_mode == DOCUMENT_RULES_MODE:
        qualified_scores = [score for score in usable_scores if score.get("risk_level") != "high"]
        if qualified_scores:
            usable_scores = qualified_scores
        else:
            return None, {}, preferred_mode

    averages = average_scores(usable_scores)
    if not prompts or not averages:
        return None, averages, preferred_mode
    ranked = sorted(prompts, key=lambda prompt: averages.get(prompt["prompt_id"], -1), reverse=True)
    best = ranked[0] if averages.get(ranked[0]["prompt_id"], -1) >= 0 else None
    if best and preferred_mode == DOCUMENT_RULES_MODE:
        top_score = averages.get(best["prompt_id"], 0)
        fusion_candidates = [prompt for prompt in prompts if is_fusion_prompt(prompt)]
        for prompt in fusion_candidates:
            prompt_id = prompt.get("prompt_id", "")
            fusion_score = averages.get(prompt_id)
            if fusion_score is None or (top_score - fusion_score) >= 0.3:
                continue
            prompt_scores = get_prompt_mode_scores(prompt_id, usable_scores, preferred_mode)
            if not prompt_scores:
                continue
            if any(score.get("risk_level") == "high" for score in prompt_scores):
                continue
            if any(score.get("hard_reject_reasons") for score in prompt_scores):
                continue
            best = prompt
            break
    return best, averages, preferred_mode


def explain_best(best: dict[str, Any], averages: dict[str, float], scores: list[dict[str, Any]], scoring_mode: str) -> str:
    prompt_scores = [score for score in scores if score["prompt_id"] == best["prompt_id"]]
    fusion_notes = dict(best.get("fusion_notes", {}) or {})
    if scoring_mode == DOCUMENT_RULES_MODE:
        aggregated = aggregate_document_dimensions(prompt_scores)
        strongest = ", ".join(format_dimension_item(item) for item in aggregated["strongest"]) or "暂无"
        weakest = ", ".join(format_dimension_item(item) for item in aggregated["weakest"]) or "暂无"
        hard_rejects = "；".join(aggregated["hard_reject_reasons"]) or "无"
        next_variable = aggregated["next_variable_to_change"] or "未指定"
        risk_level = aggregated["risk_level"]
        diagnosis = aggregated.get("attention_budget_diagnosis", {})
        overload = "是" if diagnosis.get("is_overloaded") else "否"
        main_waste = "、".join(diagnosis.get("main_waste", [])) or "无明显浪费"
        should_compress = "是" if diagnosis.get("should_compress") else "否"
        compression_priority = diagnosis.get("compression_priority", "保持当前结构")
        fusion_summary = ""
        if is_fusion_prompt(best):
            strengths_text = "；".join(str(item) for item in fusion_notes.get("absorbed_strengths", []) if str(item).strip()) or "未填写"
            weaknesses_text = "；".join(str(item) for item in fusion_notes.get("avoided_weaknesses", []) if str(item).strip()) or "未填写"
            tradeoff_text = str(fusion_notes.get("main_tradeoff", "")).strip() or "以稳定可用为优先。"
            fusion_summary = f"""

融合版推荐理由：
- 吸收的优点：{strengths_text}
- 避开的缺点：{weaknesses_text}
- 主要取舍：{tradeoff_text}
"""
        return f"""当前最优提示词：{best['title']} [{best['prompt_id']}]
总分：{averages.get(best['prompt_id'], 0)}
风险等级：{risk_level}

最强的 3 个维度：
{strongest}

最弱的 3 个维度：
{weakest}

硬性淘汰原因：
{hard_rejects}

注意力预算压缩诊断：
- 是否注意力过载：{overload}
- 主要浪费来源：{main_waste}
- 下一轮是否建议压缩：{should_compress}
- 压缩优先级：{compression_priority}
{fusion_summary}

下一轮只建议修改的一个变量：
{next_variable}
"""

    strengths = "；".join(flatten_text(score.get("strengths", [])) for score in prompt_scores if flatten_text(score.get("strengths", []))) or "平均分最高，整体表现更稳定。"
    weaknesses = "；".join(flatten_text(score.get("weaknesses", [])) for score in prompt_scores if flatten_text(score.get("weaknesses", []))) or best.get("risk", "暂无明显问题。")
    suggestions = "；".join(flatten_text(score.get("improvement_suggestions", [])) for score in prompt_scores if flatten_text(score.get("improvement_suggestions", []))) or "下一轮可结合真实反馈继续微调。"
    if is_fusion_prompt(best):
        strengths = "；".join(
            part for part in [strengths, flatten_text(fusion_notes.get("absorbed_strengths", []))] if part
        )
        weaknesses = "；".join(
            part for part in [weaknesses, flatten_text(fusion_notes.get("avoided_weaknesses", []))] if part
        )
    return f"""当前最优提示词：{best['title']} [{best['prompt_id']}]
总分：{averages.get(best['prompt_id'], 0)}

为什么选它：{strengths}

适合场景：{best.get('suitable_for', '')}

它的问题：{weaknesses}

下一轮优化方向：
{suggestions}
"""


def get_no_qualified_prompt_message(scores: list[dict[str, Any]]) -> str:
    doc_scores = [score for score in scores if normalize_mode(score) == DOCUMENT_RULES_MODE]
    aggregated = aggregate_document_dimensions(doc_scores) if doc_scores else {"hard_reject_reasons": [], "next_variable_to_change": ""}
    fix_target = aggregated["next_variable_to_change"] or "身份命名"
    return f"""当前没有合格提示词。建议重新生成候选提示词，并强制使用文档规则模板。
请先修复：{fix_target}
"""


def optimize_from_feedback(
    project: dict[str, Any],
    old_prompt: dict[str, Any],
    scores: list[dict[str, Any]],
    feedback_items: list[dict[str, Any]],
    iterations: list[dict[str, Any]],
    llm: LLMClient,
) -> tuple[dict[str, Any], dict[str, Any]]:
    version = int(old_prompt.get("version", 1)) + 1
    focus_info = determine_focus_variable(old_prompt, scores, feedback_items)
    if llm.is_available():
        generated = _optimize_with_api(project, old_prompt, scores, feedback_items, iterations, version, llm, focus_info)
        if generated:
            return generated
    return _optimize_with_template(project, old_prompt, scores, feedback_items, version, focus_info)


def optimize_prompt_from_image_feedback(
    prompt_id: str,
    feedback_id: str,
    store: JsonStore | None = None,
    llm: LLMClient | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    json_store = store or JsonStore()
    prompt = json_store.get("prompts", "prompt_id", prompt_id)
    if not prompt:
        raise ValueError(f"Prompt not found: {prompt_id}")
    feedback = json_store.get("prompt_image_feedback", "feedback_id", feedback_id)
    if not feedback:
        raise ValueError(f"Prompt image feedback not found: {feedback_id}")
    project = json_store.get("projects", "project_id", prompt.get("project_id", ""))
    if not project:
        project = _project_from_prompt(prompt)

    legacy_feedback = _image_feedback_as_optimizer_feedback(project, prompt, feedback)
    scores = json_store.filter("scores", project_id=project.get("project_id", ""), prompt_id=prompt_id)
    iterations = json_store.filter("iterations", project_id=project.get("project_id", ""))
    new_prompt, iteration = optimize_from_feedback(project, prompt, scores, [legacy_feedback], iterations, llm or LLMClient())
    iteration.update(
        {
            "from_prompt_id": prompt_id,
            "to_prompt_id": new_prompt.get("prompt_id", ""),
            "source_feedback_ids": [feedback_id],
            "iteration_reason": iteration.get("reason", ""),
        }
    )
    json_store.add("prompts", new_prompt)
    json_store.add("iterations", iteration)
    return new_prompt, iteration


def _image_feedback_as_optimizer_feedback(project: dict[str, Any], prompt: dict[str, Any], feedback: dict[str, Any]) -> dict[str, Any]:
    return {
        "project_id": project.get("project_id", prompt.get("project_id", "")),
        "prompt_id": prompt.get("prompt_id", ""),
        "real_scenario": "image_review_loop",
        "what_worked": "; ".join(str(item) for item in feedback.get("positive_patterns", [])),
        "what_failed": "; ".join(str(item) for item in feedback.get("negative_patterns", [])),
        "need_more": "; ".join(str(item) for item in feedback.get("optimization_suggestions", [])),
        "need_less": "",
        "overall_score": 0,
        "source_feedback_id": feedback.get("feedback_id", ""),
        "summary": feedback.get("summary", ""),
    }


def _project_from_prompt(prompt: dict[str, Any]) -> dict[str, Any]:
    return {
        "project_id": prompt.get("project_id", "image_loop_project"),
        "name": prompt.get("title", "Image Loop Project"),
        "scenario": "image generation prompt optimization",
        "goal": "improve prompt quality from image review feedback",
        "audience": "",
        "desired_output": "usable image generation prompts",
        "avoid": "",
        "style_preference": "",
        "evaluation_criteria": "",
    }


def merge_feedback_context(feedback_items: list[dict[str, Any]]) -> dict[str, Any]:
    if not feedback_items:
        return {}
    return {"feedback_text": "\n".join(" ".join(str(value) for value in item.values()) for item in feedback_items)}


def determine_focus_variable(old_prompt: dict[str, Any], scores: list[dict[str, Any]], feedback_items: list[dict[str, Any]]) -> dict[str, str]:
    doc_scores = [score for score in scores if normalize_mode(score) == DOCUMENT_RULES_MODE]
    if should_prioritize_attention_budget(feedback_items, doc_scores):
        return {
            "variable_key": "attention_budget_compression",
            "variable_label": "注意力预算压缩度",
            "reason": "当前提示词过长、重复或信息过密，下一轮优先只做压缩，不再继续加规则。",
            "hard_reject_reason": "",
        }

    if not doc_scores:
        return {
            "variable_key": "",
            "variable_label": "结构与表达",
            "reason": "当前没有文档规则评分，因此先做温和优化。",
            "hard_reject_reason": "",
        }

    aggregated = aggregate_document_dimensions(doc_scores)
    hard_reject_reason = aggregated["hard_reject_reasons"][0] if aggregated["hard_reject_reasons"] else ""
    variable_key = aggregated["next_variable_to_change"] or choose_next_variable(
        aggregated["scores"],
        aggregated["hard_reject_reasons"],
        aggregated.get("attention_budget_diagnosis", {}),
        old_prompt.get("full_prompt", ""),
        old_prompt.get("prompt_type", ""),
    )
    variable_label = DOCUMENT_LABEL_BY_KEY.get(variable_key, variable_key or "结构与表达")
    if hard_reject_reason:
        reason = f"先处理硬性淘汰问题：{hard_reject_reason}"
    elif variable_key == "execution_clarity_after_compression":
        reason = "当前不是继续压缩的问题，而是压缩后身份段、输入格式或 P0 约束不够清楚。本轮只修执行清晰度。"
    else:
        reason = f"优先修复当前最低分维度：{variable_label}"
    return {
        "variable_key": variable_key,
        "variable_label": variable_label,
        "reason": reason,
        "hard_reject_reason": hard_reject_reason,
    }


def _optimize_with_api(
    project: dict[str, Any],
    old_prompt: dict[str, Any],
    scores: list[dict[str, Any]],
    feedback_items: list[dict[str, Any]],
    iterations: list[dict[str, Any]],
    version: int,
    llm: LLMClient,
    focus_info: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    system = "你是提示词迭代优化专家。请只返回 JSON 对象，不要输出 Markdown。"
    user = f"""
请基于以下信息生成下一版提示词，并严格遵守：
1. 如果存在 hard_reject_reasons，先修第一个 hard_reject_reason。
2. 如果本轮变量是 attention_budget_compression，则不要新增新模块、不要新增新变量，只做合并、删除、压缩、改写。
3. 如果本轮变量是 execution_clarity_after_compression，则不要继续压缩。请只修复压缩后留下的执行清晰度问题，例如：把不稳的身份段改成稳定的人物速写、把“原文或文章段落”补回用户输入格式、把故事型提示词的 P0 补上“不要凭空虚构人物、事件或冲突，所有叙事元素必须能从原文推导出来”。
4. 如果本轮变量涉及 identity_activation，请先判断问题到底是身份太抽象、缺少具体人物、缺少场景、缺少动作、身份段过长，还是身份和任务不匹配。允许保留“你不是AI”“你不是改写工具”这类开头，只要它后面立刻接具体人物速写；不要机械删除“你不是……”。
5. 每轮只改一个变量。
6. 保留框架层，局部修改变量层。

本轮只允许优先修改的变量：
- variable_key: {focus_info['variable_key']}
- variable_label: {focus_info['variable_label']}
- reason: {focus_info['reason']}

请返回格式：
{{
  "new_prompt": {{
    "title": "...",
    "type": "...",
    "full_prompt": "...",
    "suitable_for": "...",
    "not_suitable_for": "...",
    "risk": "...",
    "usage_advice": "..."
  }},
  "changed_points": [
    "删除了什么",
    "合并了什么",
    "保留了什么",
    "为什么这次只改这个变量"
  ],
  "reason": "..."
}}

项目：{json.dumps(project, ensure_ascii=False)}
上一版提示词：{json.dumps(old_prompt, ensure_ascii=False)}
评分：{json.dumps(scores, ensure_ascii=False)}
真实反馈：{json.dumps(feedback_items, ensure_ascii=False)}
历史迭代：{json.dumps(iterations, ensure_ascii=False)}
"""
    payload = llm.generate_json(system, user)
    prompt_data = payload.get("new_prompt", {})
    full_prompt = str(prompt_data.get("full_prompt", "")).strip()
    if not isinstance(prompt_data, dict) or not full_prompt:
        print(f"API 优化失败，切换到模板优化。原因：{payload.get('parse_error', '未生成有效的新提示词')}")
        return None

    prompt_type = str(prompt_data.get("type", "")).strip() or "feedback_optimized"
    task_profile = build_task_profile(project)
    new_prompt = make_prompt(
        project["project_id"],
        version,
        str(prompt_data.get("title", f"{old_prompt['title']} 优化版")).strip(),
        normalize_identity_intro(full_prompt),
        prompt_type,
        prompt_type=prompt_type,
        suitable_for=str(prompt_data.get("suitable_for", old_prompt.get("suitable_for", ""))).strip(),
        not_suitable_for=str(prompt_data.get("not_suitable_for", old_prompt.get("not_suitable_for", ""))).strip(),
        risk=str(prompt_data.get("risk", "")).strip(),
        usage_advice=str(prompt_data.get("usage_advice", "")).strip(),
    )
    if focus_info["variable_key"] == "identity_activation":
        new_prompt["full_prompt"] = replace_identity_intro(old_prompt["full_prompt"], build_identity_intro(task_profile))
    elif focus_info["variable_key"] == "attention_budget_compression":
        new_prompt["full_prompt"] = compress_prompt_by_attention_budget(new_prompt["full_prompt"], detect_project_type(project))
    elif focus_info["variable_key"] == "execution_clarity_after_compression":
        new_prompt["full_prompt"] = repair_execution_clarity_after_compression(new_prompt["full_prompt"], prompt_type)
    new_prompt = post_process_prompt(new_prompt, project, merge_feedback_context(feedback_items))
    changed_points = (
        f"本轮只替换身份段；保留输出骨架、P0、IF-THEN 和变量控制；新的身份由任务画像自动生成。"
        if focus_info["variable_key"] == "identity_activation"
        else flatten_text(payload.get("changed_points", [])) or f"本轮只改：{focus_info['variable_label']}"
    )
    reason = str(payload.get("reason", focus_info["reason"])).strip() or focus_info["reason"]
    iteration = make_iteration(
        project["project_id"],
        old_prompt["prompt_id"],
        new_prompt["prompt_id"],
        version,
        changed_points,
        reason,
    )
    return new_prompt, iteration


def _optimize_with_template(
    project: dict[str, Any],
    old_prompt: dict[str, Any],
    scores: list[dict[str, Any]],
    feedback_items: list[dict[str, Any]],
    version: int,
    focus_info: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    project_type = detect_project_type(project)
    variable_change = focus_info["variable_label"] or "结构与表达"
    task_profile = build_task_profile(project)

    if focus_info["variable_key"] == "identity_activation":
        new_full_prompt = replace_identity_intro(old_prompt["full_prompt"], build_identity_intro(task_profile))
        changed_points = "本轮只替换身份段；保留输出骨架、P0、IF-THEN 和变量控制；新的身份由任务画像自动生成。"
        reason = focus_info["reason"]
        usage_advice = "本轮只验证身份激活是否更贴合任务类型，不改动其它模块。"
    elif focus_info["variable_key"] == "attention_budget_compression":
        new_full_prompt = compress_prompt_by_attention_budget(old_prompt["full_prompt"], project_type)
        changed_points = (
            "删除了重复解释；"
            "合并了重复判断；"
            "保留了人物速写、静默判断、图文确定性、输出骨架、IF-THEN、变量控制和 P0 禁忌；"
            "因为当前问题是规则过密，所以这次只改注意力预算。"
        )
        reason = focus_info["reason"]
        usage_advice = "本轮只验证压缩后的提示词是否更短、更清楚、更易执行，不同时改别的变量。"
    elif focus_info["variable_key"] == "execution_clarity_after_compression":
        new_full_prompt = repair_execution_clarity_after_compression(old_prompt["full_prompt"], old_prompt.get("prompt_type", ""))
        changed_points = (
            "删除了不稳的机构履历式身份表述；"
            "补回了“原文或文章段落”这一项用户输入；"
            "为故事表达型补上了禁止凭空虚构的 P0 约束；"
            "因为当前问题是压缩后不够清楚，所以这次只修执行清晰度。"
        )
        reason = focus_info["reason"]
        usage_advice = "本轮只验证压缩后的提示词是否更稳、更清楚、更可执行，不继续压缩，也不改动输出骨架。"
    else:
        new_full_prompt = normalize_identity_intro(old_prompt["full_prompt"])
        changed_points = f"本轮只改变量：{variable_change}；保留框架层，局部调整变量层。"
        reason = focus_info["reason"]
        usage_advice = f"本轮只观察 {variable_change} 是否改善，其它维度先不一起改。"

    new_prompt = make_prompt(
        project["project_id"],
        version,
        f"{old_prompt['title']} - 单变量优化版",
        new_full_prompt,
        "feedback_optimized",
        prompt_type="feedback_optimized",
        suitable_for=old_prompt.get("suitable_for", ""),
        not_suitable_for=old_prompt.get("not_suitable_for", ""),
        risk="模板优化依赖人工反馈质量，建议继续测试验证。",
        usage_advice=usage_advice,
    )
    new_prompt = post_process_prompt(new_prompt, project, merge_feedback_context(feedback_items))
    iteration = make_iteration(
        project["project_id"],
        old_prompt["prompt_id"],
        new_prompt["prompt_id"],
        version,
        changed_points,
        reason,
    )
    return new_prompt, iteration


def aggregate_document_dimensions(scores: list[dict[str, Any]]) -> dict[str, Any]:
    collected: dict[str, list[float]] = defaultdict(list)
    next_variable_votes: dict[str, int] = defaultdict(int)
    hard_reject_reasons: list[str] = []
    risk_level = "low"
    diagnosis = {"is_overloaded": False, "main_waste": [], "should_compress": False, "compression_priority": ""}

    for score in scores:
        for key, value in score.get("scores", {}).items():
            if value is not None:
                collected[key].append(float(value))
        for reason in score.get("hard_reject_reasons", []):
            if reason not in hard_reject_reasons:
                hard_reject_reasons.append(reason)
        current_diagnosis = score.get("attention_budget_diagnosis", {}) or {}
        if current_diagnosis.get("is_overloaded"):
            diagnosis["is_overloaded"] = True
        for waste in current_diagnosis.get("main_waste", []):
            if waste not in diagnosis["main_waste"]:
                diagnosis["main_waste"].append(waste)
        if current_diagnosis.get("should_compress"):
            diagnosis["should_compress"] = True
        if not diagnosis["compression_priority"] and current_diagnosis.get("compression_priority"):
            diagnosis["compression_priority"] = current_diagnosis["compression_priority"]
        suggested = str(score.get("next_variable_to_change", "")).strip()
        if suggested:
            next_variable_votes[suggested] += 1
        if score.get("risk_level") == "high":
            risk_level = "high"
        elif score.get("risk_level") == "medium" and risk_level != "high":
            risk_level = "medium"

    averages = {key: round(sum(values) / len(values), 2) for key, values in collected.items()}
    sorted_items = sorted(averages.items(), key=lambda item: item[1], reverse=True)
    if diagnosis["should_compress"] and not hard_reject_reasons:
        next_variable = "attention_budget_compression"
    elif next_variable_votes:
        next_variable = max(next_variable_votes.items(), key=lambda item: item[1])[0]
    else:
        next_variable = choose_next_variable(averages, hard_reject_reasons, diagnosis)
    return {
        "scores": averages,
        "strongest": sorted_items[:3],
        "weakest": list(reversed(sorted_items[-3:])) if sorted_items else [],
        "hard_reject_reasons": hard_reject_reasons,
        "next_variable_to_change": next_variable,
        "risk_level": risk_level,
        "attention_budget_diagnosis": diagnosis,
    }


def format_dimension_item(item: tuple[str, float]) -> str:
    key, value = item
    return f"{DOCUMENT_LABEL_BY_KEY.get(key, key)} {value}"


def flatten_text(value: object) -> str:
    if isinstance(value, list):
        return "；".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()
