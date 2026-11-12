from __future__ import annotations

import json
import re
from typing import Any

from config import DEFAULT_EXPECTED_FOCUS
from identity_builder import (
    build_identity_intro,
    build_task_profile,
    extract_identity_intro,
    replace_identity_intro,
    validate_identity_intro,
)
from llm_client import LLMClient
from models import make_prompt, make_test_case
from scoring_rules import (
    DOCUMENT_RULES_MODE,
    GRAPHIC_PROJECT_KEYWORDS,
    build_user_input_format,
    compress_prompt_by_attention_budget,
    detect_project_type,
    normalize_identity_intro,
    repair_execution_clarity_after_compression,
    split_sections,
)


PROMPT_KINDS = [
    ("A", "A版：结构清晰型"),
    ("B", "B版：故事表达型"),
    ("C", "C版：传播转化型"),
]

GRAPHIC_TRIGGER_KEYWORDS = GRAPHIC_PROJECT_KEYWORDS + ["系统性内容改编"]
ARTICLE_TO_IMAGE_KEYWORDS = ["小红书", "文章转图", "图片排版", "图文排版", "排版", "配图", "图片方案"]
COMIC_PROJECT_KEYWORDS = ["漫画", "分镜", "故事图", "视觉叙事", "系统性内容改编"]
STORY_PROMPT_TYPES = {"document_graphic_story", "document_graphic_narrative"}
STORY_IDENTITY_TEXT = "你不是自动摘要工具。你像一位信息漫画故事编辑，坐在白板前从文章里提取人物、冲突和转折，把系统性内容改成有起承转合的视觉叙事。"
STORY_RISK_TEXT = "可能过度添加虚构情节或人物，偏离原文事实。"
STORY_USAGE_ADVICE_TEXT = "所有叙事元素须能从原文推导出，避免无依据的戏剧化。"
STORY_P0_TEXT = "不要凭空虚构人物、事件或冲突，所有叙事元素必须能从原文推导出来。"


def next_version(existing_prompts: list[dict[str, Any]], project_id: str) -> int:
    versions = [int(prompt.get("version", 0)) for prompt in existing_prompts if prompt.get("project_id") == project_id]
    return max(versions or [0]) + 1


def should_use_document_prompt_rules(project: dict[str, Any]) -> bool:
    haystack = " ".join(
        [
            str(project.get("name", "")),
            str(project.get("scenario", "")),
            str(project.get("goal", "")),
            str(project.get("desired_output", "")),
            str(project.get("evaluation_criteria", "")),
            str(project.get("scoring_mode", "")),
        ]
    )
    return DOCUMENT_RULES_MODE in haystack or any(keyword in haystack for keyword in GRAPHIC_TRIGGER_KEYWORDS)


def project_text(project: dict[str, Any]) -> str:
    return " ".join(
        str(project.get(key, ""))
        for key in ["name", "scenario", "goal", "desired_output", "style_preference", "evaluation_criteria"]
    )


def is_article_to_image_project(project: dict[str, Any]) -> bool:
    haystack = project_text(project)
    return any(keyword in haystack for keyword in ARTICLE_TO_IMAGE_KEYWORDS) and "漫画" not in haystack and "分镜" not in haystack


def is_comic_project(project: dict[str, Any]) -> bool:
    haystack = project_text(project)
    return any(keyword in haystack for keyword in COMIC_PROJECT_KEYWORDS)


def generate_candidates(project: dict[str, Any], version: int, llm: LLMClient) -> list[dict[str, Any]]:
    use_document_rules = should_use_document_prompt_rules(project)
    if llm.is_available():
        generated = _generate_with_api(project, version, llm, use_document_rules)
        if generated:
            return _append_fusion_prompt(project, version, generated, llm)
    return _append_fusion_prompt(project, version, _generate_with_templates(project, version, use_document_rules), llm)


def generate_fusion_candidate(
    project: dict[str, Any],
    version: int,
    base_prompts: list[dict[str, Any]],
    llm: LLMClient,
) -> dict[str, Any] | None:
    usable_prompts = [dict(item) for item in base_prompts if str(item.get("prompt_type", item.get("kind", ""))) != "fusion_optimized"]
    if len(usable_prompts) < 3:
        return None
    base_triplet = usable_prompts[:3]
    if llm.is_available():
        fused = _generate_fusion_with_api(project, version, base_triplet, llm)
        if fused:
            return fused
    return _generate_fusion_with_template(project, version, base_triplet)


def _append_fusion_prompt(
    project: dict[str, Any],
    version: int,
    prompts: list[dict[str, Any]],
    llm: LLMClient,
) -> list[dict[str, Any]]:
    if not prompts:
        return prompts
    base_prompts = [item for item in prompts if str(item.get("prompt_type", item.get("kind", ""))) != "fusion_optimized"]
    if len(base_prompts) < 3:
        return prompts
    fusion_prompt = generate_fusion_candidate(project, version, base_prompts[:3], llm)
    if fusion_prompt:
        prompts.append(fusion_prompt)
    return prompts


def is_story_graphic_prompt(prompt_type: str) -> bool:
    return str(prompt_type or "") in STORY_PROMPT_TYPES


def _replace_section(text: str, section_title: str, new_body: str) -> str:
    lines = str(text or "").splitlines()
    if not lines:
        return text
    output: list[str] = []
    i = 0
    replaced = False
    while i < len(lines):
        line = lines[i]
        if line.startswith(f"【{section_title}】"):
            output.append(f"【{section_title}】{new_body}")
            replaced = True
            i += 1
            while i < len(lines) and not lines[i].startswith("【"):
                i += 1
            continue
        output.append(line)
        i += 1
    if not replaced:
        output.append(f"【{section_title}】{new_body}")
    return "\n".join(output)


def _normalize_field_text(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def _split_sentence_units(text: str) -> list[str]:
    normalized = _normalize_field_text(text)
    if not normalized:
        return []
    units = re.split(r"(?<=[。！？!?])\s*", normalized)
    return [unit.strip() for unit in units if unit.strip()]


def deduplicate_short_text(text: str) -> str:
    normalized = _normalize_field_text(text)
    repeated_phrases = [
        "所有叙事元素须能从原文推导出，避免无依据的戏剧化",
        "原文或文章段落",
        "不要凭空虚构人物、事件或冲突，所有叙事元素必须能从原文推导出来",
    ]
    for phrase in repeated_phrases:
        normalized = normalized.replace(f"{phrase} {phrase}。", f"{phrase}。")
        normalized = normalized.replace(f"{phrase} {phrase}", phrase)
    units = _split_sentence_units(normalized)
    if not units:
        return normalized
    deduped: list[str] = []
    for unit in units:
        if unit not in deduped:
            deduped.append(unit)
    joined = "".join(deduped).strip()
    return joined if joined else normalized


def deduplicate_prompt_text(prompt_text: str) -> str:
    text = normalize_identity_intro(prompt_text)
    if not text:
        return text

    text = text.replace("请按以下顺序输出：1) 请按以下顺序输出：1)", "请按以下顺序输出：1)")
    text = text.replace("请按以下顺序输出：请按以下顺序输出：", "请按以下顺序输出：")

    lines = text.splitlines()
    deduped_lines: list[str] = []
    previous_normalized = ""
    for line in lines:
        normalized = _normalize_field_text(line)
        if normalized and normalized == previous_normalized:
            continue
        deduped_lines.append(line.strip())
        previous_normalized = normalized
    text = "\n".join(line for line in deduped_lines if line)

    sections = split_sections(text)
    if not sections:
        return text

    rebuilt: list[str] = []
    for title, body in sections:
        new_body = body
        if "输出骨架" in title:
            cleaned = body.replace("请按以下顺序输出：1) 请按以下顺序输出：1)", "请按以下顺序输出：1)")
            cleaned = cleaned.replace("请按以下顺序输出：请按以下顺序输出：", "请按以下顺序输出：")
            prefix = "请按以下顺序输出："
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
            cleaned = re.sub(r"^\s*1\)\s*", "1) ", cleaned)
            items = [item.strip() for item in re.findall(r"\d+\)\s*([^；。\n]+)", cleaned)]
            if items:
                new_body = prefix + "\n" + "\n".join(f"{index}) {item.rstrip('；。')}；" for index, item in enumerate(items, start=1))
                new_body = new_body.rstrip("；") + "。"
        elif "用户输入格式" in title:
            if "原文或文章段落" in body:
                seen_inputs: list[str] = []
                for item in build_user_input_format(f"【用户输入格式】{body}"):
                    if item not in seen_inputs:
                        seen_inputs.append(item)
                new_body = "请用户提供：" + "；".join(seen_inputs) + "。"
        elif "P0" in title and STORY_P0_TEXT in body:
            p0_items: list[str] = []
            for item in [part.strip() for part in re.findall(r"\d+\)\s*([^；。\n]+)", body)]:
                item = item.rstrip("；。")
                if item == STORY_P0_TEXT.rstrip("。") and any(existing.startswith("不要凭空虚构人物、事件或冲突") for existing in p0_items):
                    continue
                if item not in p0_items:
                    p0_items.append(item)
            if p0_items:
                new_body = "；".join(f"{index}) {item}" for index, item in enumerate(p0_items, start=1)) + "。"
        rebuilt.append(f"{title}{new_body}")

    return normalize_identity_intro("\n".join(rebuilt))


def post_process_prompt(prompt: dict, project: dict, feedback: dict | None = None) -> dict:
    prompt = dict(prompt)
    full_prompt = str(prompt.get("full_prompt", "")).strip()
    prompt_type = str(prompt.get("prompt_type", prompt.get("kind", "")))
    project_type = detect_project_type(project)
    task_profile = build_task_profile(project)
    generated_identity = build_identity_intro(task_profile)
    feedback_text = " ".join(str(value) for value in (feedback or {}).values())

    if is_story_graphic_prompt(prompt_type):
        should_force_story_fix = any(
            token in full_prompt or token in feedback_text
            for token in ["国家地理", "曾为", "中文版制作", "具体机构履历", "原文或文章段落", "不凭空虚构"]
        )
        if should_force_story_fix:
            full_prompt = repair_execution_clarity_after_compression(full_prompt, prompt_type)

        if any(token in full_prompt for token in ["国家地理", "曾为", "中文版制作"]):
            full_prompt = _replace_section(full_prompt, "人物速写式身份", STORY_IDENTITY_TEXT)
        if "原文或文章段落" not in full_prompt:
            user_inputs = build_user_input_format(full_prompt)
            full_prompt = _replace_section(full_prompt, "用户输入格式", "请用户提供：" + "；".join(user_inputs) + "。")
        if STORY_P0_TEXT not in full_prompt:
            full_prompt = repair_execution_clarity_after_compression(full_prompt, prompt_type)

        prompt["risk"] = STORY_RISK_TEXT
        usage_advice = str(prompt.get("usage_advice", "")).strip()
        if STORY_USAGE_ADVICE_TEXT not in usage_advice:
            prompt["usage_advice"] = f"{usage_advice} {STORY_USAGE_ADVICE_TEXT}".strip() if usage_advice else STORY_USAGE_ADVICE_TEXT

    full_prompt = normalize_identity_intro(full_prompt)
    identity_validation = validate_identity_intro(extract_identity_intro(full_prompt), task_profile)
    if not identity_validation["valid"]:
        full_prompt = replace_identity_intro(full_prompt, generated_identity)
    if project_type == "graphic_content":
        full_prompt = compress_prompt_by_attention_budget(full_prompt, project_type)
        if is_story_graphic_prompt(prompt_type):
            full_prompt = repair_execution_clarity_after_compression(full_prompt, prompt_type)
        identity_validation = validate_identity_intro(extract_identity_intro(full_prompt), task_profile)
        if not identity_validation["valid"]:
            full_prompt = replace_identity_intro(full_prompt, generated_identity)

    full_prompt = deduplicate_prompt_text(full_prompt)
    prompt["full_prompt"] = full_prompt
    for field in ["suitable_for", "not_suitable_for", "risk", "usage_advice"]:
        prompt[field] = deduplicate_short_text(str(prompt.get(field, "")).replace(
            "所有叙事元素须能从原文推导出，避免无依据的戏剧化 所有叙事元素须能从原文推导出，避免无依据的戏剧化。",
            "所有叙事元素须能从原文推导出，避免无依据的戏剧化。",
        ))
    return prompt


def _generate_fusion_with_api(
    project: dict[str, Any],
    version: int,
    base_prompts: list[dict[str, Any]],
    llm: LLMClient,
) -> dict[str, Any] | None:
    system = """你是一个提示词融合优化器。
你的任务不是再生成一个随机版本，而是把 A/B/C 三个候选提示词的优点合并成 D版，同时避开它们的缺点。

你需要：
1. 识别 A/B/C 各自最强的 1-3 个优点。
2. 识别 A/B/C 各自最明显的 1-3 个缺点。
3. 判断用户项目的真实任务类型。
4. 如果任务类型是“小红书文章转图/图片排版”，不要生成漫画分镜提示词。
5. 如果任务类型是“漫画/分镜/故事图”，才生成漫画提示词。
6. 生成一个更稳定、更可用、更贴合需求的 D版提示词。
7. 不要让 D版比 A/B/C 更臃肿。
8. D版必须可直接复制使用。

返回 JSON：
{
  "prompt": {
    "title": "D版：融合优化型",
    "type": "fusion_optimized",
    "full_prompt": "...",
    "suitable_for": "...",
    "not_suitable_for": "...",
    "risk": "...",
    "usage_advice": "..."
  },
  "fusion_notes": {
    "absorbed_strengths": ["..."],
    "avoided_weaknesses": ["..."],
    "main_tradeoff": "...",
    "next_variable_to_watch": "..."
  }
}
只返回 JSON，不要输出解释性正文。"""
    payload = {
        "project": project,
        "candidates": [
            {
                "title": item.get("title", ""),
                "prompt_id": item.get("prompt_id", ""),
                "prompt_type": item.get("prompt_type", item.get("kind", "")),
                "full_prompt": item.get("full_prompt", ""),
                "suitable_for": item.get("suitable_for", ""),
                "risk": item.get("risk", ""),
                "usage_advice": item.get("usage_advice", ""),
            }
            for item in base_prompts[:3]
        ],
    }
    result = llm.generate_json(system, json.dumps(payload, ensure_ascii=False, indent=2))
    prompt_data = result.get("prompt", {})
    if not isinstance(prompt_data, dict):
        return None
    full_prompt = str(prompt_data.get("full_prompt", "")).strip()
    if not full_prompt:
        return None
    fusion_notes = result.get("fusion_notes", {}) if isinstance(result.get("fusion_notes", {}), dict) else {}
    built_prompt = make_prompt(
        project["project_id"],
        version,
        str(prompt_data.get("title", "D版：融合优化型")).strip() or "D版：融合优化型",
        full_prompt,
        "fusion_optimized",
        prompt_type=str(prompt_data.get("type", "fusion_optimized")).strip() or "fusion_optimized",
        suitable_for=str(prompt_data.get("suitable_for", "")).strip(),
        not_suitable_for=str(prompt_data.get("not_suitable_for", "")).strip(),
        risk=str(prompt_data.get("risk", "")).strip(),
        usage_advice=str(prompt_data.get("usage_advice", "")).strip(),
        source_prompt_ids=[item.get("prompt_id", "") for item in base_prompts[:3]],
        fusion_notes={
            "absorbed_strengths": list(fusion_notes.get("absorbed_strengths", []) or []),
            "avoided_weaknesses": list(fusion_notes.get("avoided_weaknesses", []) or []),
            "main_tradeoff": str(fusion_notes.get("main_tradeoff", "")).strip(),
            "next_variable_to_watch": str(fusion_notes.get("next_variable_to_watch", "")).strip(),
        },
    )
    return post_process_prompt(built_prompt, project)


def _generate_fusion_with_template(
    project: dict[str, Any],
    version: int,
    base_prompts: list[dict[str, Any]],
) -> dict[str, Any]:
    if is_article_to_image_project(project):
        full_prompt = _build_article_to_image_fusion_prompt(project)
        suitable_for = "小红书文章转图、图文排版、信息压缩、封面与多图内容设计。"
        not_suitable_for = "漫画分镜、角色叙事、需要大量动作设计的视觉脚本任务。"
        risk = "如果原文信息层级不清，容易出现排版好看但重点不够突出的情况。"
        usage_advice = "先提供原文或文章段落，再用这条提示词收敛标题、分块、版式和高亮金句。"
        tradeoff = "优先稳住图片信息排布和可读性，弱化夸张叙事感。"
        next_watch = "标题力度与正文压缩比例"
    else:
        full_prompt = _build_comic_fusion_prompt(project)
        suitable_for = "系统性内容改编成漫画、故事图、需要主体动作和视觉叙事的图文任务。"
        not_suitable_for = "纯排版图、只需要信息卡片而不需要漫画表达的项目。"
        risk = "如果原文本身缺少场景和对比关系，漫画化后可能仍显得偏硬。"
        usage_advice = "先确认原文是否适合漫画化，再用这条提示词稳定分镜、主体动作和反虚构约束。"
        tradeoff = "保留漫画叙事张力，但优先压住过度虚构。"
        next_watch = "分镜密度与文字保留比例"

    built_prompt = make_prompt(
        project["project_id"],
        version,
        "D版：融合优化型",
        full_prompt,
        "fusion_optimized",
        prompt_type="fusion_optimized",
        suitable_for=suitable_for,
        not_suitable_for=not_suitable_for,
        risk=risk,
        usage_advice=usage_advice,
        source_prompt_ids=[item.get("prompt_id", "") for item in base_prompts[:3]],
        fusion_notes={
            "absorbed_strengths": [
                "保留 A版 的结构清晰和输出稳定",
                "保留 B版 的表达吸引力和读者代入感",
                "保留 C版 的传播意识和平台适配",
            ],
            "avoided_weaknesses": [
                "避免 A版 过硬",
                "避免 B版 过度虚构或跑偏",
                "避免 C版 过度营销和标题党",
            ],
            "main_tradeoff": tradeoff,
            "next_variable_to_watch": next_watch,
        },
    )
    return post_process_prompt(built_prompt, project)


def _build_article_to_image_fusion_prompt(project: dict[str, Any]) -> str:
    identity = build_identity_intro(build_task_profile(project))
    return f"""【人物速写式身份】{identity}
【任务目标】把用户提供的文章内容转成适合小红书图片排版的文案与版式方案，优先保证读者第一眼看懂、愿意继续看、愿意收藏。
【用户输入格式】请用户提供：原文或文章段落；核心目标；读者对象；发布场景；风格偏好。
【静默判断层】正式输出前，请先在内部判断，不要展示过程：1) 核心论点是什么；2) 哪些内容适合做标题或小标题；3) 哪些句子必须压缩；4) 哪些信息值得高亮成金句；5) 最适合做成几张图；6) 读者第一眼最该看到什么。
【输出骨架】请按以下顺序输出：
1) 内容压缩判断；
2) 标题方案；
3) 小标题与分块结构；
4) 正文短句与高亮金句；
5) 图片排版建议（布局、颜色、字号）；
6) 最终图片文案；
7) P0 禁忌检查；
8) 下一轮只建议改的一个变量。
【P0 禁忌】1) 不要写成分镜脚本；2) 不要引入角色行为或戏剧化情节；3) 不要把所有信息塞进一张图；4) 不要为了好看牺牲信息清晰；5) 不要只给抽象风格词；6) 不要让正文过长难以排版；7) 不要输出无法直接复制使用的图片文案；8) 不要忽略标题、字号、布局和高亮层级。
【变量控制】如果需要迭代，请只修改一个变量：标题力度、正文压缩比例、布局密度、颜色方向、字号层级中的一个。"""


def _build_comic_fusion_prompt(project: dict[str, Any]) -> str:
    identity = build_identity_intro(build_task_profile(project))
    return f"""【人物速写式身份】{identity}
【任务目标】把用户提供的系统性内容转成更稳定的漫画提示词，兼顾结构清晰、主体动作、视觉叙事、画面可读性、读者代入感和平台传播性。
【用户输入格式】请用户提供：原文或文章段落；核心目标；读者对象；发布场景；风格偏好。
【静默判断层】正式输出前，请先在内部判断，不要展示过程：1) 核心断言是什么；2) 对偶面是什么；3) 最适合漫画化的是场景、对比、流程还是情绪变化；4) 哪些信息必须保留成文字；5) 哪些画面最能帮助读者看懂；6) 哪个变量最值得下轮微调。
【输出骨架】请按以下顺序输出：
1) 内容可视化判断；
2) 漫画化核心；
3) 推荐画风；
4) 3-5 套漫画提示词；
5) 每套提示词适合的内容类型；
6) P0 禁忌检查；
7) 下一轮只建议改的一个变量。
【P0 禁忌】1) 不要只给抽象画风，不判断内容是否适合图形化；2) 不要把所有信息塞进一张图；3) 不要为了好看牺牲主体清晰；4) 不要用过度浮夸、过度复杂的视觉元素；5) 不要让文字喧宾夺主；6) 不要凭空虚构人物、事件或冲突，所有叙事元素必须能从原文推导出来；7) 不要忽略读者第一眼能否看懂；8) 不要输出无法直接复制使用的提示词。
【变量控制】如果需要迭代，请只修改一个变量：画风、主体、分镜、文字密度、色调、情绪入口、信息层级中的一个。"""
def _generate_with_api(project: dict[str, Any], version: int, llm: LLMClient, use_document_rules: bool) -> list[dict[str, Any]]:
    system = build_generation_system_prompt(use_document_rules)
    user = build_generation_user_prompt(project, use_document_rules)
    result = llm.generate_json(system, user)
    prompt_items = result.get("prompts", [])
    if not isinstance(prompt_items, list):
        print(f"API 生成候选提示词失败，将切换到模板生成。原因：{result.get('parse_error', '返回格式不正确')}")
        return []

    prompts: list[dict[str, Any]] = []
    project_type = detect_project_type(project)
    for item in prompt_items[:3]:
        full_prompt = str(item.get("full_prompt", "")).strip()
        title = str(item.get("title", "")).strip()
        if not title or not full_prompt:
            continue
        prompt_type = str(item.get("type", "")).strip() or ("document_rules" if use_document_rules else "api")
        built_prompt = make_prompt(
            project["project_id"],
            version,
            title,
            full_prompt,
            prompt_type,
            prompt_type=prompt_type,
            suitable_for=str(item.get("suitable_for", "")).strip(),
            not_suitable_for=str(item.get("not_suitable_for", "")).strip(),
            risk=str(item.get("risk", "")).strip(),
            usage_advice=str(item.get("usage_advice", "")).strip(),
        )
        prompts.append(post_process_prompt(built_prompt, project))

    if not prompts:
        print(f"API 生成候选提示词失败，将切换到模板生成。原因：{result.get('parse_error', '未生成有效候选提示词')}")
    return prompts


def build_generation_system_prompt(use_document_rules: bool) -> str:
    if not use_document_rules:
        return "你是资深提示词工程师。请只返回 JSON 对象，不要输出 Markdown。"
    return """你是提示词骨架设计师，不是普通写作助手。你现在要生成的是“能稳定筛选漫画 / 图文提示词的提示词”，而不是普通任务助手提示词。
必须遵守：
1. 禁止只用抽象身份开头，例如“你是一名专业内容创作者”“你是一名温暖且专业的助手”。
2. 身份段可以直接用正向人物速写开头，也可以先用一句“你不是AI”“你不是改写工具”之类的否定式开头，但后面必须立刻接具体人物速写、场景、动作和判断方式。
3. 禁止只留下孤立的否定句，例如“你不是普通内容助手。”如果使用“你不是……”，必须紧接真实人物状态。
4. 必须包含静默判断层，并明确“不要展示推理过程”。
5. 漫画 / 图文项目必须包含图文确定性判断。
6. 必须有清晰输出骨架。
7. 必须包含 IF-THEN 自我迭代和变量控制。
8. 必须把 P0 禁忌放在最后。
9. 生成完整草稿后，请自行压缩注意力预算：身份段控制在 80-120 字内，“你不是……”最多一句，后面立刻接人物速写；静默判断不超过 6 条，输出骨架不超过 7 项，IF-THEN 不超过 6 条，P0 不超过 8 条。
10. 返回的是可直接复制使用的完整提示词，不要返回讲解。

请只返回 JSON 对象，不要输出 Markdown。"""


def build_generation_user_prompt(project: dict[str, Any], use_document_rules: bool) -> str:
    if not use_document_rules:
        return f"""
请基于以下项目需求生成 3 个候选提示词，分别是：
1. A版：结构清晰型
2. B版：故事表达型
3. C版：传播转化型

返回格式：
{{
  "prompts": [
    {{
      "title": "A版：结构清晰型",
      "type": "structure",
      "full_prompt": "...",
      "suitable_for": "...",
      "not_suitable_for": "...",
      "risk": "...",
      "usage_advice": "..."
    }}
  ]
}}

项目需求：
{json.dumps(project, ensure_ascii=False, indent=2)}
"""

    return f"""
项目需求：
{json.dumps(project, ensure_ascii=False, indent=2)}

请生成 3 个候选提示词：
1. A版：结构清晰型
2. B版：故事表达型
3. C版：传播转化型

每个候选提示词的 full_prompt 都必须包含并压缩到以下结构：
- 【人物速写式身份】控制在 80-120 字内；允许两种写法：
  1) 直接正向人物速写开头
  2) 先用一句“你不是AI，也不是改写工具。”之类的否定式身份激活，再立刻接具体人物速写
- 【任务目标】一句话说明任务
- 【用户输入格式】列出 3-5 项，并包含“原文或文章段落”
- 【静默判断层】最多 6 个内部判断问题，不展示过程
- 【图文确定性判断】最多 4 项
- 【输出骨架】最多 7 项
- 【自我迭代 IF-THEN】最多 6 条
- 【变量控制】明确下一轮只改一个变量
- 【P0 禁忌】最多 8 条，放最后

重点要求：
1. 如果使用“你不是……”，必须后接具体人物、场景、动作和判断方式，不能只停留在否定句。
2. 身份段要像真实做过“系统性内容改编成漫画”的编辑，不是抽象职业标签。
3. 身份段只保留人物身份、工作场景、核心动作、判断方式，不要堆太多道具。
4. 图文确定性判断必须覆盖：图文类型、主体画面、文字保留、删除信息。
5. 如果出现重复规则，请合并；如果出现抽象意图词，请改成可执行动作。
6. 故事表达型不要声称具体机构履历，例如“曾为某某机构制作”；要改成稳定的人物速写。

返回格式：
{{
  "prompts": [
    {{
      "title": "A版：结构清晰型",
      "type": "document_graphic_structure",
      "full_prompt": "...",
      "suitable_for": "...",
      "not_suitable_for": "...",
      "risk": "...",
      "usage_advice": "..."
    }}
  ]
}}
"""


def _generate_with_templates(project: dict[str, Any], version: int, use_document_rules: bool) -> list[dict[str, Any]]:
    if use_document_rules:
        return _generate_with_document_templates(project, version)

    base = {
        "scenario": project.get("scenario", ""),
        "goal": project.get("goal", ""),
        "audience": project.get("audience", ""),
        "desired_output": project.get("desired_output", ""),
        "avoid": project.get("avoid", ""),
        "style": project.get("style_preference", ""),
        "criteria": project.get("evaluation_criteria") or DEFAULT_EXPECTED_FOCUS,
    }
    templates = [
        (
            "A版：结构清晰型",
            "structure",
            f"""你是一名专业任务拆解与内容生成助手。请基于以下需求完成任务：
- 使用场景：{base['scenario']}
- 目标：{base['goal']}
- 使用对象：{base['audience']}
- 希望输出：{base['desired_output']}
- 风格偏好：{base['style']}
- 不希望出现：{base['avoid']}

请按以下结构输出：
1. 先简要确认你理解的任务目标
2. 拆解关键要点
3. 生成最终内容
4. 给出可执行建议或下一步

评价标准：{base['criteria']}
请确保输出具体、清楚、可直接使用。""",
            "需要稳定、清楚、可复用输出的任务。",
            "需要强烈情绪感染或广告化表达的场景。",
            "可能略显正式，缺少故事张力。",
            "适合作为默认版本；如果输出太平，可要求模型增加案例和语气变化。",
        ),
        (
            "B版：故事表达型",
            "story",
            f"""你是一名擅长叙事表达的内容策略助手。我要完成的目标是：{base['goal']}
使用场景：{base['scenario']}
目标读者 / 对象：{base['audience']}

请先找到这个需求背后的真实问题、人物处境和情绪动机，然后用有画面感但不夸张的方式生成内容。

输出要求：
- 内容形式：{base['desired_output']}
- 风格：{base['style']}
- 必须围绕真实痛点、转折和行动建议展开
- 避免：{base['avoid']}
- 最后补充“为什么这样写有效”

评价标准：{base['criteria']}""",
            "需要增强感染力、共鸣感、案例感的内容。",
            "强流程、强合规、强技术精确性任务。",
            "如果叙事过多，可能稀释执行步骤。",
            "适合内容创作、演讲、品牌表达；必要时可要求减少铺垫。",
        ),
        (
            "C版：传播转化型",
            "conversion",
            f"""你是一名增长与转化导向的提示词助手。请围绕以下需求生成高转化输出：
- 目标：{base['goal']}
- 场景：{base['scenario']}
- 目标对象：{base['audience']}
- 输出内容：{base['desired_output']}
- 风格偏好：{base['style']}
- 禁止或减少：{base['avoid']}

请输出：
1. 核心卖点或核心主张
2. 目标对象最关心的问题
3. 3 个可选表达角度
4. 最终内容
5. 行动引导或使用建议

请让内容更容易被理解、记住和采取行动。
评价标准：{base['criteria']}""",
            "需要标题、营销文案、转化脚本、社媒传播的任务。",
            "纯研究、严肃报告、低主观表达任务。",
            "可能为了吸引力而牺牲克制感。",
            "适合传播场景；若过度营销，可要求模型降低销售感。",
        ),
    ]
    prompts: list[dict[str, Any]] = []
    for title, prompt_type, full_prompt, suitable_for, not_suitable_for, risk, usage_advice in templates:
        built = make_prompt(
            project["project_id"],
            version,
            title,
            full_prompt,
            prompt_type,
            prompt_type=prompt_type,
            suitable_for=suitable_for,
            not_suitable_for=not_suitable_for,
            risk=risk,
            usage_advice=usage_advice,
        )
        prompts.append(post_process_prompt(built, project))
    return prompts


def _generate_with_document_templates(project: dict[str, Any], version: int) -> list[dict[str, Any]]:
    project_type = detect_project_type(project)
    variants = [
        ("A版：结构清晰型", "document_graphic_structure", "把输出骨架和分镜逻辑锁稳，适合先筛结构。"),
        ("B版：故事表达型", "document_graphic_story", "把读者情绪入口和画面叙事拉出来，适合漫画表达。"),
        ("C版：传播转化型", "document_graphic_conversion", "把可视化判断和传播入口结合，适合社媒图文。"),
    ]
    prompts: list[dict[str, Any]] = []
    for title, prompt_type, advice in variants:
        raw_prompt = build_graphic_prompt_document_template(project, title, prompt_type)
        built_prompt = make_prompt(
            project["project_id"],
            version,
            title,
            raw_prompt,
            prompt_type,
            prompt_type=prompt_type,
            suitable_for="系统性内容改编成漫画、图文、配图、视觉内容筛选与提示词设计。",
            not_suitable_for="纯文本研究报告、无需图形化判断的普通写作任务。",
            risk="如果内容本身不适合图形化，强行漫画化会损失信息密度。",
            usage_advice=advice,
        )
        prompts.append(post_process_prompt(built_prompt, project))
    return prompts


def build_graphic_prompt_document_template(project: dict[str, Any], title: str, prompt_type: str) -> str:
    task_profile = build_task_profile(project)
    emphasis_map = {
        "document_graphic_structure": "优先锁定信息层级、输出骨架和分镜逻辑。",
        "document_graphic_story": "优先强化读者情绪入口、人物状态和画面叙事。",
        "document_graphic_narrative": "优先强化读者情绪入口、人物状态和画面叙事。",
        "document_graphic_conversion": "优先压实主体辨识度、传播入口和可复制性。",
    }
    p0_map = {
        "document_graphic_story": "1) 不要只给抽象画风，不判断内容是否适合图形化；2) 不要把所有信息塞进一张图；3) 不要为了好看牺牲主体清晰；4) 不要用过度浮夸或过度复杂的视觉元素；5) 不要生成无法直接复制使用的提示词；6) 不要只说“温暖、治愈、简洁”，必须给具体画面；7) 不要忽略读者第一眼能否看懂；8) 不要凭空虚构人物、事件或冲突，所有叙事元素必须能从原文推导出来。",
        "document_graphic_narrative": "1) 不要只给抽象画风，不判断内容是否适合图形化；2) 不要把所有信息塞进一张图；3) 不要为了好看牺牲主体清晰；4) 不要用过度浮夸或过度复杂的视觉元素；5) 不要生成无法直接复制使用的提示词；6) 不要只说“温暖、治愈、简洁”，必须给具体画面；7) 不要忽略读者第一眼能否看懂；8) 不要凭空虚构人物、事件或冲突，所有叙事元素必须能从原文推导出来。",
    }
    emphasis = emphasis_map.get(prompt_type, "优先把图文结构和漫画化判断做扎实。")
    identity = build_identity_intro(task_profile)
    p0 = p0_map.get(
        prompt_type,
        "1) 不要只给抽象画风，不判断内容是否适合图形化；2) 不要把所有信息塞进一张图；3) 不要为了好看牺牲主体清晰；4) 不要用过度浮夸或过度复杂的视觉元素；5) 不要生成无法直接复制使用的提示词；6) 不要只说“温暖、治愈、简洁”，必须给具体画面；7) 不要让文字喧宾夺主；8) 不要忽略读者第一眼能否看懂。",
    )
    return f"""【人物速写式身份】{identity}
【任务目标】把用户提供的系统性内容转成适合漫画化表达的提示词，优先保证读者第一眼看懂、主体突出、信息轻量但结构清楚。本版重点：{emphasis}
【用户输入格式】请用户提供：原文或文章段落；核心目标；读者对象；发布场景；风格偏好。
【静默判断层】在正式输出前，请先在内部完成以下判断，不要展示推理过程：1) 核心断言是什么；2) 对偶面是什么；3) 最硬锚点是什么；4) 读者旧地图是什么；5) 认知落差出现在哪里；6) 最适合漫画化的是场景、对比、流程、数据还是情绪变化。
【图文确定性判断】1) 判断这段内容属于事实、对比、流程、数据或情绪场景中的哪一类；2) 锁定主体画面是什么；3) 判断哪些信息保留成文字；4) 判断哪些信息必须删除，避免塞进图里。
【输出骨架】请按以下顺序输出：1) 内容可视化判断；2) 漫画化核心；3) 推荐画风；4) 3-5 套漫画提示词；5) 每套提示词适合的内容类型；6) P0 禁忌检查；7) 下一轮只建议改的一个变量。
【自我迭代 IF-THEN】IF 主体不突出 THEN 减少背景和装饰，强化主体动作。IF 画风描述过多 THEN 压缩风格词，强化主体动作和构图。IF 分镜信息过密 THEN 拆成多张图。IF 输出像普通插画而不是漫画 THEN 增加分镜、角色动作和文字位置。IF 文字太多 THEN 保留核心断言，删除解释性文字。
【变量控制】如果需要迭代，请只修改一个变量：画风、主体、分镜、文字密度、色调、读者情绪入口、信息层级中的一个。
【P0 禁忌】{p0}"""


def generate_test_cases(project: dict[str, Any], llm: LLMClient, max_cases: int = 3) -> list[dict[str, Any]]:
    if llm.is_available():
        generated = _generate_test_cases_with_api(project, llm, max_cases)
        if generated:
            return generated
    return _generate_test_cases_with_templates(project, max_cases)


def build_auto_test_case(project: dict[str, Any], version: int) -> dict[str, Any]:
    case_input = "\n".join(
        [
            f"任务名称：{project.get('name', '')}",
            f"使用场景：{project.get('scenario', '')}",
            f"目标：{project.get('goal', '')}",
            f"目标读者 / 使用对象：{project.get('audience', '')}",
            f"希望输出内容：{project.get('desired_output', '')}",
            f"不希望出现的内容：{project.get('avoid', '')}",
            f"风格偏好：{project.get('style_preference', '')}",
        ]
    )
    expected_focus = project.get("evaluation_criteria") or DEFAULT_EXPECTED_FOCUS
    return make_test_case(project["project_id"], f"项目自测案例 - version {version}", case_input, expected_focus)


def _generate_test_cases_with_api(project: dict[str, Any], llm: LLMClient, max_cases: int) -> list[dict[str, Any]]:
    system = "你是提示词测试设计助手。请只返回 JSON 对象，不要输出 Markdown。"
    user = f"""
请根据以下项目需求生成 1 到 {max_cases} 个测试案例。

返回格式：
{{
  "test_cases": [
    {{
      "title": "...",
      "input": "...",
      "expected_focus": "..."
    }}
  ]
}}

项目需求：
{json.dumps(project, ensure_ascii=False, indent=2)}
"""
    result = llm.generate_json(system, user)
    case_items = result.get("test_cases", [])
    if not isinstance(case_items, list):
        print(f"API 生成测试案例失败，将切换到模板生成。原因：{result.get('parse_error', '返回格式不正确')}")
        return []

    cases: list[dict[str, Any]] = []
    for item in case_items[:max_cases]:
        title = str(item.get("title", "")).strip()
        case_input = str(item.get("input", "")).strip()
        expected_focus = str(item.get("expected_focus", "")).strip() or DEFAULT_EXPECTED_FOCUS
        if title and case_input:
            cases.append(make_test_case(project["project_id"], title, case_input, expected_focus))

    if not cases:
        print(f"API 生成测试案例失败，将切换到模板生成。原因：{result.get('parse_error', '未生成有效测试案例')}")
    return cases


def _generate_test_cases_with_templates(project: dict[str, Any], max_cases: int) -> list[dict[str, Any]]:
    expected_focus = project.get("evaluation_criteria") or DEFAULT_EXPECTED_FOCUS
    templates = [
        (
            "核心需求验证案例",
            "\n".join(
                [
                    f"任务名称：{project.get('name', '')}",
                    f"请围绕使用场景“{project.get('scenario', '')}”完成任务。",
                    f"目标：{project.get('goal', '')}",
                    f"面向对象：{project.get('audience', '')}",
                    f"输出形式：{project.get('desired_output', '')}",
                ]
            ),
        ),
        (
            "风格与边界验证案例",
            "\n".join(
                [
                    f"请生成一版内容，风格偏好为：{project.get('style_preference', '')}",
                    f"必须避免：{project.get('avoid', '')}",
                    f"依然要达成目标：{project.get('goal', '')}",
                    f"最终交付：{project.get('desired_output', '')}",
                ]
            ),
        ),
        (
            "真实使用场景压测案例",
            "\n".join(
                [
                    f"你正在处理一个真实任务：{project.get('name', '')}",
                    f"场景：{project.get('scenario', '')}",
                    f"对象：{project.get('audience', '')}",
                    "请在信息有限的情况下，优先输出最有价值的内容。",
                ]
            ),
        ),
    ]
    return [make_test_case(project["project_id"], title, case_input, expected_focus) for title, case_input in templates[:max_cases]]
