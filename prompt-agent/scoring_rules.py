from __future__ import annotations

import re
from typing import Any

from identity_builder import build_task_profile, extract_identity_intro, validate_identity_intro

GENERAL_MODE = "general"
DOCUMENT_RULES_MODE = "document_rules"

GENERAL_SCORE_KEY_MAP = {
    "贴合需求": "fit",
    "结构清晰": "structure",
    "表达吸引力": "appeal",
    "可执行性": "actionability",
    "用户风格匹配度": "style_match",
    "可复用性": "reusability",
}

DOCUMENT_SCORE_DIMENSIONS = [
    {
        "key": "identity_activation",
        "label": "身份激活度",
        "hint": "是否同时具备角色原型、工作场景、处理对象、动作链和判断标准，允许“你不是……”但后面必须立刻接具体人物速写。",
    },
    {
        "key": "silent_labeling",
        "label": "静默判断层",
        "hint": "是否要求模型先在内部判断关键变量，再输出结果。",
    },
    {
        "key": "middle_layer_completeness",
        "label": "中间层完整度",
        "hint": "是否包含锚点、对偶面、读者心理、信息层级等关键中间变量。",
    },
    {
        "key": "output_skeleton",
        "label": "输出骨架确定性",
        "hint": "是否提前固定输出顺序、模块、格式和每部分作用。",
    },
    {
        "key": "p0_constraints",
        "label": "P0 约束强度",
        "hint": "是否明确堵住模型最容易走的默认通道。",
    },
    {
        "key": "balance_structure_and_state",
        "label": "结构化与非结构化平衡",
        "hint": "是否既锁边界，又保留人物状态和人味。",
    },
    {
        "key": "attention_budget_compression",
        "label": "注意力预算压缩度",
        "hint": "是否短而强，规则少但关键，没有明显重复和堆叠。",
    },
    {
        "key": "executable_rules",
        "label": "执行层清晰度",
        "hint": "是否把抽象意图改成可执行、可自检的动作。",
    },
    {
        "key": "self_iteration",
        "label": "自我迭代能力",
        "hint": "是否有 IF-THEN 或防重复机制。",
    },
    {
        "key": "variable_control",
        "label": "变量控制能力",
        "hint": "是否区分框架层和变量层，并建议每次只改一个变量。",
    },
    {
        "key": "topic_judgment",
        "label": "选题判断力",
        "hint": "是否能反推核心断言、对偶面、锚点、旧地图和认知落差。",
    },
    {
        "key": "graphic_certainty",
        "label": "图文确定性",
        "hint": "是否判断内容是否适合图形化，而不是只谈画风。",
    },
]

DOCUMENT_LABEL_BY_KEY = {item["key"]: item["label"] for item in DOCUMENT_SCORE_DIMENSIONS}
DOCUMENT_LABEL_BY_KEY["execution_clarity_after_compression"] = "压缩后执行清晰度"

DOCUMENT_WEIGHTS = {
    "graphic_content": {
        "identity_activation": 0.08,
        "silent_labeling": 0.08,
        "middle_layer_completeness": 0.08,
        "output_skeleton": 0.12,
        "p0_constraints": 0.10,
        "balance_structure_and_state": 0.08,
        "attention_budget_compression": 0.08,
        "executable_rules": 0.10,
        "self_iteration": 0.06,
        "variable_control": 0.06,
        "topic_judgment": 0.08,
        "graphic_certainty": 0.08,
    },
    "text_content": {
        "identity_activation": 0.12,
        "silent_labeling": 0.08,
        "middle_layer_completeness": 0.10,
        "output_skeleton": 0.08,
        "p0_constraints": 0.10,
        "balance_structure_and_state": 0.10,
        "attention_budget_compression": 0.08,
        "executable_rules": 0.10,
        "self_iteration": 0.08,
        "variable_control": 0.06,
        "topic_judgment": 0.10,
        "graphic_certainty": 0.00,
    },
    "default": {
        "identity_activation": 0.10,
        "silent_labeling": 0.08,
        "middle_layer_completeness": 0.09,
        "output_skeleton": 0.10,
        "p0_constraints": 0.10,
        "balance_structure_and_state": 0.09,
        "attention_budget_compression": 0.08,
        "executable_rules": 0.10,
        "self_iteration": 0.08,
        "variable_control": 0.07,
        "topic_judgment": 0.08,
        "graphic_certainty": 0.03,
    },
}

GRAPHIC_PROJECT_KEYWORDS = ["漫画", "图文", "配图", "视觉", "画风", "小红书图文", "系统性内容改编"]
TEXT_PROJECT_KEYWORDS = ["文章", "标题", "写作", "内容", "文案", "IP"]

attention_budget_rules = {
    "max_total_sections": 10,
    "max_middle_layer_variables": 8,
    "max_p0_constraints": 8,
    "max_if_then_rules": 6,
    "max_output_blocks": 7,
    "max_identity_words": 120,
    "prefer_merge_repeated_rules": True,
    "prefer_executable_rules": True,
    "avoid_intent_only_phrases": True,
    "avoid_duplicate_graphic_judgment": True,
    "avoid_over_specified_style": True,
}

INTENT_ONLY_TOKENS = ["共鸣感", "高级", "自然", "吸引人", "有温度", "温暖", "专业", "简洁美观", "治愈"]
ACTION_TOKENS = ["输出", "判断", "删除", "保留", "压缩", "改写", "拆成", "增加", "减少", "检查", "自检", "列出", "说明"]
FEEDBACK_COMPRESSION_TOKENS = ["太长", "太复杂", "重复", "信息太密", "用起来累", "不够轻", "主体不突出", "规则太多"]
NEGATIVE_IDENTITY_OPENERS = [
    "你不是普通内容助手。",
    "你不是AI。",
    "你不是改写工具。",
    "你不是一个普通的助手。",
    "你不是一名普通的内容创作者。",
    "你不是普通内容助手",
    "你不是AI",
    "你不是改写工具",
    "你不是一个普通的助手",
    "你不是一名普通的内容创作者",
]

DOCUMENT_SCORING_SYSTEM_PROMPT = """你是一个提示词筛选与诊断器。
你的任务不是判断内容好不好，而是判断“这个提示词是否具备稳定生成好内容的结构”。

你要基于以下规则评分：
1. 身份命名：抽象标签低分；允许“你不是AI”“你不是改写工具”这类否定式开头，但后面必须立刻接具体人物、场景、动作、对象或任务方向；只有否定没有人物速写仍然低分。
2. 静默标签：没有生成前内部判断低分，有内部变量锁定高分。
3. 中间层：只有格式低分，有内容锚点、对偶角度、读者心理、说话人设计高分。
4. 输出端：输出结构不明确低分，骨架稳定高分。
5. P0约束：没有禁忌低分，能堵住AI默认通道高分。
6. 对偶性：只有结构化规则会有AI味，只有感觉描述会失控，结构化+非结构化配合高分。
7. 注意力预算压缩：规则堆叠、重复、冲突、意图词过多会扣分；短而强、关键规则少但有力高分。
8. 执行层：抽象意图低分，AI可自检动作高分。
9. 自我迭代：没有防重复低分，有 IF-THEN 路由高分。
10. 变量控制：一次改太多低分，能区分框架层和变量层高分。
11. 选题判断：只看标题表面低分，能反推核心断言、对偶面、锚点、旧地图、认知落差高分。
12. 图文确定性：只谈画风低分，能判断信息是否适合图形化高分。

你主要评价的是候选提示词本身，而不是生成结果本身。生成结果只作为验证参考。
你必须返回 JSON，不要返回解释性正文。

返回格式：
{
  "scores": {
    "identity_activation": 0,
    "silent_labeling": 0,
    "middle_layer_completeness": 0,
    "output_skeleton": 0,
    "p0_constraints": 0,
    "balance_structure_and_state": 0,
    "attention_budget_compression": 0,
    "executable_rules": 0,
    "self_iteration": 0,
    "variable_control": 0,
    "topic_judgment": 0,
    "graphic_certainty": 0
  },
  "risk_level": "low | medium | high",
  "hard_reject_reasons": [],
  "strengths": [],
  "weaknesses": [],
  "improvement_suggestions": [],
  "next_variable_to_change": "",
  "attention_budget_diagnosis": {
    "is_overloaded": false,
    "main_waste": [],
    "should_compress": false,
    "compression_priority": ""
  }
}"""


def detect_project_type(project: dict[str, Any]) -> str:
    haystack = " ".join(
        [
            str(project.get("name", "")),
            str(project.get("scenario", "")),
            str(project.get("goal", "")),
            str(project.get("desired_output", "")),
        ]
    )
    if any(keyword in haystack for keyword in GRAPHIC_PROJECT_KEYWORDS):
        return "graphic_content"
    if any(keyword in haystack for keyword in TEXT_PROJECT_KEYWORDS):
        return "text_content"
    return "default"


def get_document_weights(project: dict[str, Any]) -> tuple[str, dict[str, float]]:
    project_type = detect_project_type(project)
    return project_type, DOCUMENT_WEIGHTS[project_type]


def compute_weighted_total(scores: dict[str, float | None], weights: dict[str, float]) -> float:
    total_weight = 0.0
    weighted_sum = 0.0
    for key, weight in weights.items():
        value = scores.get(key)
        if value is None or weight <= 0:
            continue
        weighted_sum += float(value) * weight
        total_weight += weight
    if total_weight <= 0:
        return 0.0
    return round(weighted_sum / total_weight, 2)


def is_graphic_project(project: dict[str, Any]) -> bool:
    return detect_project_type(project) == "graphic_content"


def normalize_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def split_sections(prompt_text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_title = ""
    current_lines: list[str] = []
    for raw_line in str(prompt_text).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("【") and "】" in line:
            if current_title or current_lines:
                sections.append((current_title or "正文", "\n".join(current_lines).strip()))
            title, _, rest = line.partition("】")
            current_title = f"{title}】"
            current_lines = [rest.strip()] if rest.strip() else []
        else:
            current_lines.append(line)
    if current_title or current_lines:
        sections.append((current_title or "正文", "\n".join(current_lines).strip()))
    return sections


def _count_ordered_items(text: str) -> int:
    parts = re.split(r"(?:^|\n)(?:[-•]|\d+[)）.、]\s*)", text)
    items = [part.strip() for part in parts if part.strip()]
    return len(items)


def _extract_section_text(prompt_text: str, keyword: str) -> str:
    for title, body in split_sections(prompt_text):
        if keyword in title:
            return body
    return ""


def _identity_has_positive_sketch(identity_text: str) -> bool:
    text = str(identity_text or "")
    has_person = "你像" in text or "像一位" in text or "像一个" in text
    has_scene = any(token in text for token in ["坐在", "站在", "面对", "白板前", "桌前", "场景"])
    has_action = any(token in text for token in ["拆", "判断", "改成", "改写", "筛选", "处理", "分析"])
    return has_person and has_scene and has_action


def normalize_identity_intro(prompt_text: str) -> str:
    text = str(prompt_text or "")
    if not text:
        return text

    lines = text.splitlines()
    normalized_lines: list[str] = []
    for line in lines:
        current = re.sub(r"\s+", " ", line).strip()
        current = current.replace("。。", "。").replace("，，", "，")
        normalized_lines.append(current)
    return "\n".join(normalized_lines)


def evaluate_identity_activation(prompt: dict[str, Any], project: dict[str, Any]) -> dict[str, Any]:
    task_profile = build_task_profile(project)
    identity_text = extract_identity_intro(str(prompt.get("full_prompt", "")))
    return validate_identity_intro(identity_text, task_profile)


def _is_valid_section_body(text: str) -> bool:
    return bool(str(text or "").strip())


def _find_duplicate_signals(prompt_text: str) -> list[str]:
    duplicates: list[str] = []
    sections = split_sections(prompt_text)
    joined = "\n".join(body for _, body in sections)
    duplicate_groups = {
        "graphic_judgment": ["是否适合图形化", "主体画面", "保留文字", "不能塞进图里"],
        "output_requirements": ["按以下顺序输出", "输出骨架", "输出顺序"],
        "style_requirements": ["温暖", "治愈", "简洁", "高级", "画风"],
    }
    for name, tokens in duplicate_groups.items():
        hits = sum(1 for token in tokens if joined.count(token) >= 2)
        if hits >= 2:
            duplicates.append(name)
    if len(sections) != len({title for title, _ in sections}):
        duplicates.append("repeated_section_titles")
    return duplicates


def _find_conflicting_rules(prompt_text: str) -> list[str]:
    conflicts: list[str] = []
    text = str(prompt_text)
    if "极简" in text and ("详尽" in text or "完整详尽" in text):
        conflicts.append("同时要求极简和完整详尽，但没有区分场景")
    if "像朋友聊天" in text and ("极度权威" in text or "绝对权威" in text):
        conflicts.append("同时要求像朋友聊天和极度权威，但没有区分场景")
    return conflicts


def analyze_attention_budget(prompt_text: str) -> dict[str, Any]:
    text = str(prompt_text or "")
    sections = split_sections(text)
    section_count = len(sections)
    middle_text = _extract_section_text(text, "中间层")
    p0_text = _extract_section_text(text, "P0")
    if_then_text = _extract_section_text(text, "IF-THEN")
    output_text = _extract_section_text(text, "输出骨架")
    identity_text = _extract_section_text(text, "人物速写式身份") or (sections[0][1] if sections else "")

    middle_count = _count_ordered_items(middle_text)
    p0_count = _count_ordered_items(p0_text)
    if_then_count = max(len(re.findall(r"\bIF\b", if_then_text, flags=re.IGNORECASE)), _count_ordered_items(if_then_text))
    output_count = _count_ordered_items(output_text)
    identity_words = len(identity_text.replace("\n", " "))

    overload_reasons: list[str] = []
    too_many_variables: list[str] = []
    if section_count > attention_budget_rules["max_total_sections"]:
        overload_reasons.append(f"段落模块过多：{section_count} > {attention_budget_rules['max_total_sections']}")
    if middle_count > attention_budget_rules["max_middle_layer_variables"]:
        overload_reasons.append(f"中间层变量过多：{middle_count} > {attention_budget_rules['max_middle_layer_variables']}")
        too_many_variables.append("middle_layer_variables")
    if p0_count > attention_budget_rules["max_p0_constraints"]:
        overload_reasons.append(f"P0 禁忌过多：{p0_count} > {attention_budget_rules['max_p0_constraints']}")
        too_many_variables.append("p0_constraints")
    if if_then_count > attention_budget_rules["max_if_then_rules"]:
        overload_reasons.append(f"IF-THEN 规则过多：{if_then_count} > {attention_budget_rules['max_if_then_rules']}")
        too_many_variables.append("if_then_rules")
    if output_count > attention_budget_rules["max_output_blocks"]:
        overload_reasons.append(f"输出骨架过多：{output_count} > {attention_budget_rules['max_output_blocks']}")
    if identity_words > attention_budget_rules["max_identity_words"]:
        overload_reasons.append(f"身份段过长：{identity_words} > {attention_budget_rules['max_identity_words']}")

    duplicated_sections = _find_duplicate_signals(text)
    conflicting_rules = _find_conflicting_rules(text)
    intent_only_phrases: list[str] = []
    for token in INTENT_ONLY_TOKENS:
        for line in text.splitlines():
            stripped = line.strip()
            if token not in stripped:
                continue
            if "P0" in stripped or "不要只说" in stripped or "不要只给" in stripped:
                continue
            intent_only_phrases.append(token)
            break

    style_over_specified = "画风" in text and sum(text.count(token) for token in ["温暖", "治愈", "高级", "复杂", "电影感", "插画感"]) >= 4
    if duplicated_sections:
        overload_reasons.append("存在重复规则或重复模块")
    if conflicting_rules:
        overload_reasons.append("存在风格冲突或目标冲突")
    if intent_only_phrases and not any(token in text for token in ACTION_TOKENS):
        overload_reasons.append("存在大量意图层形容词，但缺少执行动作")
    if style_over_specified:
        overload_reasons.append("画风描述过度指定，可能压住主体")

    compression_suggestions: list[str] = []
    if section_count > attention_budget_rules["max_total_sections"]:
        compression_suggestions.append("合并重复模块，只保留关键结构。")
    if middle_count > attention_budget_rules["max_middle_layer_variables"]:
        compression_suggestions.append("把中间层变量并入静默判断层或输出骨架，保留 6-8 个最关键变量。")
    if p0_count > attention_budget_rules["max_p0_constraints"]:
        compression_suggestions.append("把 P0 禁忌压成 6-8 条短、硬、具体的负向约束。")
    if if_then_count > attention_budget_rules["max_if_then_rules"]:
        compression_suggestions.append("只保留最常见的 5-6 条 IF-THEN 路由。")
    if output_count > attention_budget_rules["max_output_blocks"]:
        compression_suggestions.append("把输出骨架压到 6-7 项，避免和前文重复。")
    if duplicated_sections:
        compression_suggestions.append("删除重复说明，把同类判断合并到一个模块。")
    if intent_only_phrases:
        compression_suggestions.append("把抽象形容词改写成可执行动作，例如限制句数、先后顺序、删减规则。")
    if style_over_specified:
        compression_suggestions.append("压缩画风词，优先突出主体、动作和信息层级。")

    is_overloaded = bool(overload_reasons)
    if conflicting_rules:
        recommended_action = "rewrite"
    elif is_overloaded or duplicated_sections or intent_only_phrases:
        recommended_action = "compress"
    else:
        recommended_action = "keep"

    return {
        "is_overloaded": is_overloaded,
        "overload_reasons": overload_reasons,
        "duplicated_sections": duplicated_sections,
        "conflicting_rules": conflicting_rules,
        "intent_only_phrases": intent_only_phrases,
        "too_many_variables": too_many_variables,
        "compression_suggestions": compression_suggestions,
        "recommended_action": recommended_action,
    }


def _limit_items(items: list[str], limit: int) -> list[str]:
    cleaned = [item.strip(" -•") for item in items if item.strip(" -•")]
    return cleaned[:limit]


def _extract_list_items(text: str) -> list[str]:
    if not text.strip():
        return []
    matches = re.split(r"(?:^|\n)(?:[-•]|\d+[)）.、]\s*)", text)
    items = [item.strip() for item in matches if item.strip()]
    if items:
        return items
    return [line.strip() for line in text.splitlines() if line.strip()]


def _extract_numbered_or_inline_items(text: str) -> list[str]:
    source = str(text or "").strip()
    if not source:
        return []
    numbered = re.findall(r"\d+[)）.、]\s*([^；;\n]+)", source)
    if numbered:
        return [item.strip() for item in numbered if item.strip()]
    return _extract_list_items(source)


def _clean_inline_item(item: str) -> str:
    return re.sub(r"^\d+[)）.、]\s*", "", item).strip(" ；;。")


def _shorten_identity(text: str) -> str:
    identity = normalize_identity_intro(text).replace("\n", " ").strip()
    if len(identity) <= 120:
        return identity
    chunks = [part.strip() for part in re.split(r"[。；，]", identity) if part.strip()]
    summary = "，".join(chunks[:5]).strip("，")
    return summary[:120].rstrip("，。") + "。"


def _infer_user_input_format(prompt_text: str) -> list[str]:
    candidates = ["原文主题", "核心目标", "读者对象", "发布场景", "风格偏好"]
    text = str(prompt_text)
    inferred: list[str] = []
    mapping = {
        "任务名称": "原文主题",
        "目标": "核心目标",
        "读者": "读者对象",
        "场景": "发布场景",
        "风格": "风格偏好",
    }
    for token, label in mapping.items():
        if token in text and label not in inferred:
            inferred.append(label)
    return inferred[:5] or candidates[:5]


def build_user_input_format(prompt_text: str) -> list[str]:
    source = _extract_section_text(prompt_text, "用户输入格式")
    if source:
        stripped = source.replace("请用户提供：", "").replace("用户需要提供：", "")
        raw_items = re.split(r"[；;\n]+", stripped)
        items = [_clean_inline_item(item) for item in raw_items if _clean_inline_item(item)]
    else:
        items = _infer_user_input_format(prompt_text)

    normalized: list[str] = []
    has_original_text = False
    for item in items:
        current = item
        if "原文" in item or "文章段落" in item:
            current = "原文或文章段落"
            has_original_text = True
        elif item == "原文主题":
            current = "原文或文章段落"
            has_original_text = True
        if current not in normalized:
            normalized.append(current)

    if not has_original_text:
        normalized.insert(0, "原文或文章段落")

    fallback_order = ["核心目标", "读者对象", "发布场景", "风格偏好"]
    for item in fallback_order:
        if item not in normalized:
            normalized.append(item)
    return normalized[:5]


def has_story_hallucination_guard(prompt_text: str) -> bool:
    return "不要凭空虚构人物、事件或冲突" in str(prompt_text or "")


def needs_execution_clarity_after_compression(prompt_text: str, prompt_type: str = "") -> dict[str, Any]:
    text = str(prompt_text or "")
    prompt_type = str(prompt_type or "")
    identity_text = _extract_section_text(text, "人物速写式身份")
    user_input_text = _extract_section_text(text, "用户输入格式")
    p0_text = _extract_section_text(text, "P0")

    issues: list[str] = []
    if "曾为" in identity_text or ("《" in identity_text and "》" in identity_text):
        issues.append("identity_resume_claim")
    if "原文" not in user_input_text and "文章段落" not in user_input_text:
        issues.append("missing_original_text_input")
    if ("story" in prompt_type or "narrative" in prompt_type or "故事" in prompt_type) and not has_story_hallucination_guard(p0_text):
        issues.append("missing_story_grounding_p0")
    if identity_text and len(identity_text) < 40:
        issues.append("identity_too_thin")

    return {
        "has_issue": bool(issues),
        "issues": issues,
    }


def repair_execution_clarity_after_compression(prompt_text: str, prompt_type: str = "") -> str:
    text = str(prompt_text or "")
    prompt_type = str(prompt_type or "")
    sections = split_sections(text)
    if not sections:
        return text

    repaired_lines: list[str] = []
    is_story_prompt = "story" in prompt_type or "narrative" in prompt_type or "故事" in prompt_type

    for title, body in sections:
        new_body = body
        if "人物速写式身份" in title and is_story_prompt:
            new_body = "你不是自动摘要工具。你像一位信息漫画故事编辑，坐在白板前从文章里提取人物、冲突和转折，把系统性内容改成有起承转合的视觉叙事。"
        elif "用户输入格式" in title:
            user_inputs = build_user_input_format(text)
            new_body = "请用户提供：" + "；".join(user_inputs) + "。"
        elif "P0" in title and is_story_prompt and not has_story_hallucination_guard(body):
            p0_items = [_clean_inline_item(item) for item in _extract_numbered_or_inline_items(body) if _clean_inline_item(item)]
            p0_items.append("不要凭空虚构人物、事件或冲突，所有叙事元素必须能从原文推导出来。")
            new_body = "；".join(f"{idx + 1}) {item.rstrip('。')}" for idx, item in enumerate(p0_items))
        repaired_lines.append(f"{title}{new_body}")

    return normalize_identity_intro("\n".join(repaired_lines))


def _build_graphic_output_blocks(output_text: str) -> list[str]:
    items = _limit_items(_extract_list_items(output_text), 7)
    if items:
        return items
    return [
        "内容可视化判断",
        "漫画化核心",
        "推荐画风",
        "3-5 套漫画提示词",
        "每套提示词适合的内容类型",
        "P0 禁忌检查",
        "下一轮只建议改的一个变量",
    ]


def _build_graphic_silent_questions(prompt_text: str) -> list[str]:
    source = _extract_section_text(prompt_text, "静默判断层") + "\n" + _extract_section_text(prompt_text, "中间层")
    source = source.replace("在正式输出前，请先在内部完成以下判断，不要展示推理过程：", "")
    source = source.replace("在正式输出前，请在内部完成以下判断：", "")
    items = _limit_items(_extract_list_items(source), 6)
    if items:
        return items
    return [
        "核心断言是什么",
        "对偶面是什么",
        "最硬锚点是什么",
        "读者旧地图是什么",
        "认知落差出现在哪里",
        "最适合漫画化的是场景、对比、流程、数据还是情绪变化",
    ]


def _build_graphic_if_then(prompt_text: str) -> list[str]:
    items = _limit_items(_extract_list_items(_extract_section_text(prompt_text, "IF-THEN")), 6)
    if items:
        return items
    return [
        "IF 主体不突出 THEN 减少背景和装饰，强化主体动作。",
        "IF 画风描述过多 THEN 压缩风格词，强化主体动作和构图。",
        "IF 分镜信息过密 THEN 拆成多张图，不要硬塞进一张图。",
        "IF 输出像插画而不是漫画 THEN 增加分镜、角色动作和文字位置。",
        "IF 文字太多 THEN 保留核心断言，删除解释性文字。",
    ]


def _build_graphic_p0(prompt_text: str) -> list[str]:
    items = _limit_items(_extract_list_items(_extract_section_text(prompt_text, "P0")), 8)
    if items:
        return items
    return [
        "不要只给抽象画风，不判断内容是否适合图形化。",
        "不要把所有信息塞进一张图。",
        "不要为了好看牺牲主体清晰。",
        "不要用过度浮夸或过度复杂的视觉元素。",
        "不要生成无法直接复制使用的提示词。",
        "不要只说温暖、治愈、简洁，必须给具体画面。",
        "不要让文字喧宾夺主。",
        "不要忽略读者第一眼能否看懂。",
    ]


def compress_prompt_by_attention_budget(prompt_text: str, project_type: str) -> str:
    text = normalize_identity_intro(str(prompt_text or "").strip())
    if not text:
        return text

    if project_type == "graphic_content":
        sections = split_sections(text)
        identity = _shorten_identity(_extract_section_text(text, "人物速写式身份") or (sections[0][1] if sections else text))
        goal = (_extract_section_text(text, "任务目标") or "把系统性内容压成适合图文化表达的漫画提示词。").splitlines()[0].strip()
        user_inputs = build_user_input_format(text)
        silent_questions = _build_graphic_silent_questions(text)
        graphic_checks = _limit_items(_extract_list_items(_extract_section_text(text, "图文确定性判断")), 4) or [
            "判断图文类型：事实、对比、流程、数据或情绪场景。",
            "锁定主体画面是什么。",
            "判断哪些信息保留为文字。",
            "判断哪些信息必须删除，避免塞进图里。",
        ]
        output_blocks = _build_graphic_output_blocks(_extract_section_text(text, "输出骨架"))
        if_then_rules = _build_graphic_if_then(text)
        p0_rules = _build_graphic_p0(text)

        lines = [
            f"【人物速写式身份】{identity}",
            f"【任务目标】{goal}",
            "【用户输入格式】请用户提供：" + "；".join(user_inputs[:5]) + "。",
            "【静默判断层】在正式输出前，请先在内部完成以下判断，不要展示推理过程：" + "；".join(
                f"{idx + 1}) {item.rstrip('。？?')}" for idx, item in enumerate(silent_questions)
            ) + "。",
            "【图文确定性判断】" + "；".join(
                f"{idx + 1}) {item.rstrip('。')}" for idx, item in enumerate(graphic_checks)
            ) + "。",
            "【输出骨架】请按以下顺序输出：" + "；".join(
                f"{idx + 1}) {item.rstrip('。')}" for idx, item in enumerate(output_blocks)
            ) + "。",
            "【自我迭代 IF-THEN】" + "；".join(rule.rstrip("。") for rule in if_then_rules[:6]) + "。",
            "【变量控制】如果需要迭代，请只修改一个变量：画风、主体、分镜、文字密度、色调、读者情绪入口、信息层级中的一个。",
            "【P0 禁忌】" + "；".join(f"{idx + 1}) {item.rstrip('。')}" for idx, item in enumerate(p0_rules[:8])) + "。",
        ]
        return normalize_identity_intro("\n".join(lines))

    diagnosis = analyze_attention_budget(text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if diagnosis["recommended_action"] == "keep":
        return normalize_identity_intro("\n".join(lines))

    compact_lines: list[str] = []
    seen = set()
    for line in lines:
        normalized = re.sub(r"\s+", "", line)
        if normalized in seen:
            continue
        seen.add(normalized)
        new_line = line
        for token in INTENT_ONLY_TOKENS:
            if token in new_line and not any(action in new_line for action in ACTION_TOKENS):
                new_line = new_line.replace(token, "")
        compact_lines.append(new_line.strip())
    return normalize_identity_intro("\n".join(compact_lines))


def build_attention_budget_diagnosis(prompt_text: str) -> dict[str, Any]:
    analysis = analyze_attention_budget(prompt_text)
    main_waste: list[str] = []
    if analysis["duplicated_sections"]:
        main_waste.append("duplicated_rules")
    if "middle_layer_variables" in analysis["too_many_variables"]:
        main_waste.append("too_many_variables")
    if "p0_constraints" in analysis["too_many_variables"]:
        main_waste.append("too_many_p0_constraints")
    if "if_then_rules" in analysis["too_many_variables"]:
        main_waste.append("too_many_if_then_rules")
    if analysis["intent_only_phrases"]:
        main_waste.append("intent_layer_words")
    if any("画风描述过度指定" in reason for reason in analysis["overload_reasons"]):
        main_waste.append("over_specified_style")
    if any("输出骨架过多" in reason for reason in analysis["overload_reasons"]):
        main_waste.append("repeated_output_requirements")
    if analysis["conflicting_rules"]:
        main_waste.append("conflicting_rules")

    if "too_many_variables" in main_waste:
        compression_priority = "先压缩中间层变量"
    elif "duplicated_rules" in main_waste:
        compression_priority = "先合并重复规则"
    elif "too_many_p0_constraints" in main_waste:
        compression_priority = "先压缩 P0 禁忌"
    elif "intent_layer_words" in main_waste:
        compression_priority = "先把意图词改成执行动作"
    elif "over_specified_style" in main_waste:
        compression_priority = "先压缩画风描述，突出主体"
    else:
        compression_priority = "保持当前结构"

    return {
        "is_overloaded": analysis["is_overloaded"],
        "main_waste": main_waste,
        "should_compress": analysis["recommended_action"] == "compress",
        "compression_priority": compression_priority,
    }


def evaluate_hard_rejects(prompt: dict[str, Any], project: dict[str, Any]) -> list[str]:
    text = str(prompt.get("full_prompt", ""))
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    first_chunk = "\n".join(lines[:6])
    identity_text = _extract_section_text(text, "人物速写式身份") or first_chunk
    reasons: list[str] = []
    identity_report = evaluate_identity_activation(prompt, project)

    abstract_identity_tokens = ["你是一名专业", "你是一名温暖且专业", "你是一个专业", "你是资深", "写作助手", "内容创作者", "任务拆解与内容生成助手"]
    if any(token in identity_text for token in abstract_identity_tokens):
        reasons.append("身份命名只是抽象标签。")
    if any(token in identity_text for token in NEGATIVE_IDENTITY_OPENERS) and not _identity_has_positive_sketch(identity_text):
        reasons.append("身份段只有否定式开头，没有具体人物、场景和动作。")
    if not identity_report["valid"]:
        issue_text = " / ".join(identity_report["issues"])
        if issue_text.startswith("缺少"):
            issue_text = issue_text.replace("缺少", "", 1)
        reasons.append(
            "identity_activation 不合格：身份段缺少"
            + issue_text
            + f"。建议改成：{identity_report['suggested_fix']}"
        )

    silent_tokens = ["在正式输出前", "先在内部完成", "不要展示推理过程", "不要展示判断过程", "静默判断"]
    if not any(token in text for token in silent_tokens):
        reasons.append("没有静默判断层。")

    skeleton_tokens = ["【输出骨架】", "请按以下顺序输出", "【内容可视化判断】", "【漫画化核心】"]
    if not any(token in text for token in skeleton_tokens):
        reasons.append("没有输出骨架。")

    p0_tokens = ["【P0 禁忌】", "【P0禁忌】", "P0", "不要只给抽象画风"]
    if not any(token in text for token in p0_tokens):
        reasons.append("没有 P0 禁忌。")

    if is_graphic_project(project):
        graphic_tokens = ["是否适合图形化", "图文确定性判断", "主体画面", "保留文字", "不能塞进图里"]
        if not any(token in text for token in graphic_tokens):
            reasons.append("漫画 / 图文项目没有图文确定性判断。")

    if not any(token in text for token in ACTION_TOKENS):
        reasons.append("没有执行层动作，只有意图层描述。")

    iteration_tokens = ["IF", "THEN", "防重复", "换表达路径", "换对偶面"]
    if not any(token in text for token in iteration_tokens):
        reasons.append("没有自我迭代机制。")

    variable_tokens = ["只修改一个变量", "只改一个变量", "变量控制", "框架层", "变量层"]
    if not any(token in text for token in variable_tokens):
        reasons.append("没有变量控制说明。")

    return reasons


def choose_next_variable(
    scores: dict[str, float | None],
    hard_reject_reasons: list[str],
    attention_diagnosis: dict[str, Any] | None = None,
    prompt_text: str = "",
    prompt_type: str = "",
) -> str:
    if hard_reject_reasons:
        mapping = {
            "身份": "identity_activation",
            "静默": "silent_labeling",
            "输出骨架": "output_skeleton",
            "P0": "p0_constraints",
            "图文确定性": "graphic_certainty",
            "执行层": "executable_rules",
            "自我迭代": "self_iteration",
            "变量控制": "variable_control",
            "注意力预算": "attention_budget_compression",
        }
        reason = hard_reject_reasons[0]
        for token, key in mapping.items():
            if token in reason:
                return key
    diagnosis = attention_diagnosis or {}
    if diagnosis.get("should_compress") or diagnosis.get("is_overloaded"):
        return "attention_budget_compression"
    clarity = needs_execution_clarity_after_compression(prompt_text, prompt_type)
    attention_score = scores.get("attention_budget_compression")
    if (
        attention_score is not None
        and float(attention_score) < 7
        and not diagnosis.get("should_compress")
        and not diagnosis.get("is_overloaded")
        and clarity["has_issue"]
    ):
        return "execution_clarity_after_compression"
    valid = [(key, value) for key, value in scores.items() if value is not None]
    if not valid:
        return ""
    return min(valid, key=lambda item: item[1])[0]


def should_prioritize_attention_budget(feedback_items: list[dict[str, Any]], scores: list[dict[str, Any]]) -> bool:
    for score in scores:
        mode = str(score.get("scoring_mode", GENERAL_MODE) or GENERAL_MODE)
        if mode != DOCUMENT_RULES_MODE:
            continue
        diagnosis = score.get("attention_budget_diagnosis", {}) or {}
        if diagnosis.get("should_compress") or diagnosis.get("is_overloaded"):
            return True

    for feedback in feedback_items:
        joined = " ".join(str(feedback.get(key, "")) for key in feedback.keys())
        if any(token in joined for token in FEEDBACK_COMPRESSION_TOKENS):
            return True
    return False
