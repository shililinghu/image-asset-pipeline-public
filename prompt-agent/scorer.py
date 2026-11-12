from __future__ import annotations

import json

from config import GENERAL_SCORE_DIMENSIONS
from identity_builder import build_task_profile, extract_identity_intro, validate_identity_intro
from llm_client import LLMClient
from models import make_score
from scoring_rules import (
    DOCUMENT_RULES_MODE,
    DOCUMENT_SCORING_SYSTEM_PROMPT,
    DOCUMENT_SCORE_DIMENSIONS,
    GENERAL_MODE,
    GENERAL_SCORE_KEY_MAP,
    analyze_attention_budget,
    build_attention_budget_diagnosis,
    choose_next_variable,
    compute_weighted_total,
    evaluate_identity_activation,
    evaluate_hard_rejects,
    get_document_weights,
    needs_execution_clarity_after_compression,
    normalize_list,
)
from utils import clamp_score, prompt_multiline


def score_result(project: dict, prompt: dict, case: dict, result: dict, llm: LLMClient, scoring_mode: str = GENERAL_MODE) -> dict:
    if scoring_mode == DOCUMENT_RULES_MODE:
        return score_result_with_document_rules(project, prompt, case, result, llm)
    return score_result_with_general_rules(project, prompt, case, result, llm)


def score_result_with_general_rules(project: dict, prompt: dict, case: dict, result: dict, llm: LLMClient) -> dict:
    if llm.is_available():
        scored = _score_general_with_api(project, prompt, case, result, llm)
        if scored:
            return scored
    return _score_general_manually(project["project_id"], prompt["prompt_id"], case["case_id"], result["result_id"])


def score_result_with_document_rules(project: dict, prompt: dict, case: dict, result: dict, llm: LLMClient) -> dict:
    if llm.is_available():
        scored = _score_document_with_api(project, prompt, case, result, llm)
        if scored:
            return scored
    return _score_document_manually(project, prompt, case, result)


def _score_general_with_api(project: dict, prompt: dict, case: dict, result: dict, llm: LLMClient) -> dict | None:
    system = "你是严格但建设性的提示词评分员。请只返回 JSON 对象，不要输出 Markdown。"
    user = f"""
请基于项目需求、提示词、测试案例和测试输出进行评分。
评分维度固定为：
- 贴合需求
- 结构清晰
- 表达吸引力
- 可执行性
- 用户风格匹配度
- 可复用性

返回格式：
{{
  "scores": {{
    "fit": 8,
    "structure": 8,
    "appeal": 7,
    "actionability": 8,
    "style_match": 7,
    "reusability": 8
  }},
  "total_score": 7.67,
  "strengths": ["..."],
  "weaknesses": ["..."],
  "improvement_suggestions": ["..."]
}}

项目：{json.dumps(project, ensure_ascii=False)}
提示词：{prompt['full_prompt']}
测试案例：{json.dumps(case, ensure_ascii=False)}
测试输出：{result['output']}
"""
    payload = llm.generate_json(system, user, temperature=0.2)
    raw_scores = payload.get("scores", {})
    if not isinstance(raw_scores, dict):
        print(f"API 评分失败，切换到手动评分。原因：{payload.get('parse_error', '评分 JSON 不完整')}")
        return None

    scores = {}
    for label in GENERAL_SCORE_DIMENSIONS:
        api_key = GENERAL_SCORE_KEY_MAP[label]
        scores[label] = clamp_score(str(raw_scores.get(api_key, 0)))

    if not any(scores.values()):
        print(f"API 评分失败，切换到手动评分。原因：{payload.get('parse_error', '未解析到有效分数')}")
        return None

    return make_score(
        project["project_id"],
        prompt["prompt_id"],
        case["case_id"],
        result["result_id"],
        scores,
        normalize_list(payload.get("strengths", "")),
        normalize_list(payload.get("weaknesses", "")),
        normalize_list(payload.get("improvement_suggestions", "")),
        scoring_mode=GENERAL_MODE,
        risk_level="low",
    )


def _score_general_manually(project_id: str, prompt_id: str, case_id: str, test_result_id: str) -> dict:
    print("请为该测试结果手动评分，每项 0-10 分。")
    scores = {}
    for dim in GENERAL_SCORE_DIMENSIONS:
        scores[dim] = clamp_score(input(f"{dim}: ").strip())
    strengths = normalize_list(prompt_multiline("优点"))
    weaknesses = normalize_list(prompt_multiline("问题"))
    suggestions = normalize_list(prompt_multiline("改进建议"))
    return make_score(
        project_id,
        prompt_id,
        case_id,
        test_result_id,
        scores,
        strengths,
        weaknesses,
        suggestions,
        scoring_mode=GENERAL_MODE,
        risk_level="low",
    )


def _score_document_with_api(project: dict, prompt: dict, case: dict, result: dict, llm: LLMClient) -> dict | None:
    project_type, weights = get_document_weights(project)
    local_analysis = analyze_attention_budget(prompt["full_prompt"])
    local_diagnosis = build_attention_budget_diagnosis(prompt["full_prompt"])
    task_profile = build_task_profile(project)
    identity_report = validate_identity_intro(extract_identity_intro(prompt["full_prompt"]), task_profile)
    user = f"""
你主要评价的是候选提示词本身，而不是生成结果本身。生成结果只作为验证参考。
请优先看 prompt.full_prompt 是否符合文档规则和注意力预算压缩规则。

项目类型判断参考：{project_type}

项目：
{json.dumps(project, ensure_ascii=False, indent=2)}

候选提示词 full_prompt：
{prompt['full_prompt']}

测试案例 input：
{case['input']}

测试结果 output：
{result['output']}

本地注意力预算预诊断：
{json.dumps(local_analysis, ensure_ascii=False, indent=2)}

本地 identity_activation 预诊断：
{json.dumps(identity_report, ensure_ascii=False, indent=2)}
"""
    payload = llm.generate_json(DOCUMENT_SCORING_SYSTEM_PROMPT, user, temperature=0.1)
    raw_scores = payload.get("scores", {})
    if not isinstance(raw_scores, dict):
        print(f"API 文档规则评分失败，切换到手动评分。原因：{payload.get('parse_error', '评分 JSON 不完整')}")
        return None

    scores: dict[str, float | None] = {}
    for item in DOCUMENT_SCORE_DIMENSIONS:
        key = item["key"]
        value = raw_scores.get(key)
        scores[key] = clamp_score(str(value)) if value is not None else None

    current_identity_score = scores.get("identity_activation")
    if current_identity_score is None:
        scores["identity_activation"] = float(identity_report["score"])
    else:
        scores["identity_activation"] = min(float(current_identity_score), float(identity_report["score"]))

    if scores.get("attention_budget_compression") is None:
        scores["attention_budget_compression"] = 5 if local_analysis["is_overloaded"] else 8

    hard_reject_reasons = normalize_list(payload.get("hard_reject_reasons", []))
    local_rejects = evaluate_hard_rejects(prompt, project)
    for reason in local_rejects:
        if reason not in hard_reject_reasons:
            hard_reject_reasons.append(reason)
    if not identity_report["valid"]:
        issue_text = " / ".join(identity_report["issues"])
        if issue_text.startswith("缺少"):
            issue_text = issue_text.replace("缺少", "", 1)
        identity_reason = "identity_activation 不合格：身份段缺少" + issue_text + f"。建议改成：{identity_report['suggested_fix']}"
        if identity_reason not in hard_reject_reasons:
            hard_reject_reasons.append(identity_reason)

    diagnosis = payload.get("attention_budget_diagnosis", {})
    if not isinstance(diagnosis, dict):
        diagnosis = {}
    merged_diagnosis = {
        "is_overloaded": bool(diagnosis.get("is_overloaded", local_diagnosis["is_overloaded"])),
        "main_waste": normalize_list(diagnosis.get("main_waste", [])) or local_diagnosis["main_waste"],
        "should_compress": bool(diagnosis.get("should_compress", local_diagnosis["should_compress"])),
        "compression_priority": str(diagnosis.get("compression_priority", local_diagnosis["compression_priority"])).strip()
        or local_diagnosis["compression_priority"],
    }

    if merged_diagnosis["should_compress"] and (scores.get("attention_budget_compression") or 0) > 6:
        scores["attention_budget_compression"] = 6

    total_score = compute_weighted_total(scores, weights)
    risk_level = str(payload.get("risk_level", "low")).strip().lower() or "low"
    if hard_reject_reasons:
        risk_level = "high"
    elif total_score < 5 or merged_diagnosis["is_overloaded"]:
        risk_level = "medium"

    next_variable = str(payload.get("next_variable_to_change", "")).strip() or choose_next_variable(
        scores,
        hard_reject_reasons,
        merged_diagnosis,
        prompt["full_prompt"],
        prompt.get("prompt_type", ""),
    )
    clarity_check = needs_execution_clarity_after_compression(prompt["full_prompt"], prompt.get("prompt_type", ""))
    if (
        next_variable == "attention_budget_compression"
        and not merged_diagnosis["is_overloaded"]
        and not merged_diagnosis["should_compress"]
        and clarity_check["has_issue"]
    ):
        next_variable = "execution_clarity_after_compression"

    return make_score(
        project["project_id"],
        prompt["prompt_id"],
        case["case_id"],
        result["result_id"],
        scores,
        normalize_list(payload.get("strengths", [])),
        normalize_list(payload.get("weaknesses", [])),
        normalize_list(payload.get("improvement_suggestions", [])),
        scoring_mode=DOCUMENT_RULES_MODE,
        total_score=total_score,
        risk_level=risk_level,
        hard_reject_reasons=hard_reject_reasons,
        next_variable_to_change=next_variable,
        attention_budget_diagnosis=merged_diagnosis,
    )


def _score_document_manually(project: dict, prompt: dict, case: dict, result: dict) -> dict:
    project_type, weights = get_document_weights(project)
    identity_report = evaluate_identity_activation(prompt, project)
    print(f"当前使用：文档规则评分模式。项目类型判断为：{project_type}")
    print("主要评价候选提示词本身，测试结果只作辅助参考。")
    print("可回车跳过不适用项。")
    if not identity_report["valid"]:
        print("身份激活预诊断：")
        print(" - " + " / ".join(identity_report["issues"]))
        print(f" - 建议改成：{identity_report['suggested_fix']}")
    scores: dict[str, float | None] = {}
    for item in DOCUMENT_SCORE_DIMENSIONS:
        raw = input(f"{item['label']}（{item['hint']}，0-10 分）: ").strip()
        scores[item["key"]] = clamp_score(raw) if raw else None
    if scores.get("identity_activation") is not None:
        scores["identity_activation"] = min(float(scores["identity_activation"]), float(identity_report["score"]))
    elif identity_report["score"] > 0:
        scores["identity_activation"] = float(identity_report["score"])

    hard_reject_reasons = []
    for reason in evaluate_hard_rejects(prompt, project):
        print(f"检测到硬性风险：{reason}")
        hard_reject_reasons.append(reason)

    diagnosis = build_attention_budget_diagnosis(prompt["full_prompt"])
    if diagnosis["is_overloaded"]:
        print(f"检测到注意力预算过载：{', '.join(diagnosis['main_waste']) or '结构过密'}")
        if scores.get("attention_budget_compression") is None:
            scores["attention_budget_compression"] = 6

    strengths = normalize_list(prompt_multiline("优点"))
    weaknesses = normalize_list(prompt_multiline("问题"))
    suggestions = normalize_list(prompt_multiline("改进建议"))
    next_variable = prompt_multiline("下一轮只建议修改的一个变量（回车可留空）").strip() or choose_next_variable(
        scores,
        hard_reject_reasons,
        diagnosis,
        prompt["full_prompt"],
        prompt.get("prompt_type", ""),
    )
    total_score = compute_weighted_total(scores, weights)
    risk_level = "high" if hard_reject_reasons else ("medium" if total_score < 5 or diagnosis["is_overloaded"] else "low")
    return make_score(
        project["project_id"],
        prompt["prompt_id"],
        case["case_id"],
        result["result_id"],
        scores,
        strengths,
        weaknesses,
        suggestions,
        scoring_mode=DOCUMENT_RULES_MODE,
        total_score=total_score,
        risk_level=risk_level,
        hard_reject_reasons=hard_reject_reasons,
        next_variable_to_change=next_variable,
        attention_budget_diagnosis=diagnosis,
    )
