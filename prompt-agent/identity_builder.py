from __future__ import annotations

from typing import Any


TASK_PROFILE_EXAMPLES = {
    "image_layout": "你像一位长期做小红书信息图排版的视觉编辑，坐在电脑前把长文章压成清晰可读的图片：先提炼标题，再压缩正文，最后安排字号、颜色、留白和高亮区域。你的判断标准是读者第一眼能看懂，并愿意收藏。",
    "comic_script": "你不是自动摘要工具。你像一位信息漫画故事编辑，坐在白板前从文章里提取人物、冲突和转折，把系统性内容改成有起承转合的视觉叙事。你的判断标准是画面主体清楚，叙事元素能从原文推导。",
    "zodiac_article": "你像一个懂生肖、懂家事、懂钱事的老朋友，坐在小区长椅上跟老人慢慢说话：先说他们最近的处境，再把担心和委屈说圆，最后给一句稳妥的生活提醒。你的判断标准是通俗、安定、能让老人愿意读下去。",
    "learning_structure": "你像一位站在白板前的知识结构教练，先帮学习者搭出知识骨架，再拆关键概念、变量关系和反例，最后给出能立刻练习的最小输出。你的判断标准是学习者能理解、能复述、能使用。",
    "title_topic": "你像一位坐在选题墙前的内容编辑，先从素材里找核心断言，再反推读者心里的问题，最后把选题压成有张力但不标题党的标题。你的判断标准是标题有进入感，但不牺牲真实性。",
}

ABSTRACT_IDENTITY_TOKENS = [
    "专业助手",
    "内容创作者",
    "提示词专家",
    "写作助手",
    "任务拆解助手",
    "内容生成助手",
    "专业内容创作者",
]

ROLE_MISMATCH_RULES = {
    "image_layout": ["漫画", "分镜", "故事编辑", "漫画脚本"],
    "comic_script": ["排版视觉编辑", "信息图排版", "标题策划"],
    "title_topic": ["漫画编辑", "排版视觉编辑", "知识结构教练"],
    "learning_structure": ["营销文案", "排版视觉编辑", "漫画故事编辑"],
    "article": ["信息图排版", "漫画脚本", "代码审查工程师"],
}


def _project_value(project: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = project.get(key, "")
        if str(value).strip():
            return str(value).strip()
    return ""


def _project_text(project: dict[str, Any]) -> str:
    return " ".join(
        _project_value(
            project,
            "name",
            "task_name",
            "scenario",
            "use_case",
            "goal",
            "audience",
            "target_audience",
            "desired_output",
            "avoid",
            "style_preference",
            "evaluation_criteria",
        )
        for _ in [0]
    ).strip()


def _pick_output_medium(text: str) -> str:
    if any(token in text for token in ["小红书", "图片排版", "图文排版", "文章转图", "信息图", "配图"]):
        return "image_layout"
    if any(token in text for token in ["漫画", "分镜", "故事图", "视觉叙事"]):
        return "comic_script"
    if any(token in text for token in ["标题", "选题", "话题"]):
        return "title_topic"
    if any(token in text for token in ["学习", "课程", "知识模块", "复习", "理解", "练习"]):
        return "learning_structure"
    if any(token in text for token in ["代码", "开发", "review", "审查", "需求文档", "bug"]):
        return "code_task"
    if any(token in text for token in ["复盘", "反馈", "回顾"]):
        return "review_feedback"
    if any(token in text for token in ["agent", "提示词优化", "系统设计", "工作流"]):
        return "agent_design"
    if any(token in text for token in ["文章", "写作", "文案", "生肖"]):
        return "article"
    return "general"


def _pick_platform(text: str) -> str:
    if "小红书" in text:
        return "xiaohongshu"
    if "微信" in text or "公众号" in text:
        return "wechat"
    if "B站" in text or "bilibili" in text.lower():
        return "bilibili"
    return "general"


def _pick_task_action(text: str) -> str:
    if any(token in text for token in ["转", "改编", "压缩", "转化"]):
        return "transform"
    if any(token in text for token in ["排版", "布局"]):
        return "layout"
    if any(token in text for token in ["评分", "筛选", "评估", "review", "审查"]):
        return "evaluate"
    if any(token in text for token in ["解释", "讲解"]):
        return "explain"
    if any(token in text for token in ["优化", "迭代", "改进"]):
        return "optimize"
    if any(token in text for token in ["设计", "搭建"]):
        return "design"
    if any(token in text for token in ["写", "生成"]):
        return "write"
    return "generate"


def _build_target_object(project: dict[str, Any], output_medium: str) -> str:
    name = _project_value(project, "name", "task_name")
    desired_output = _project_value(project, "desired_output")
    if output_medium == "image_layout":
        return "长文章和小红书图文素材"
    if output_medium == "comic_script":
        return "系统性内容和文章段落"
    if output_medium == "title_topic":
        return "标题素材和选题草稿"
    if output_medium == "learning_structure":
        return "知识模块和学习目标"
    if output_medium == "code_task":
        return "需求文档和代码任务"
    if output_medium == "review_feedback":
        return "复盘记录和真实反馈"
    if output_medium == "agent_design":
        return "提示词优化流程和 Agent 需求"
    if "生肖" in name:
        return "老年人生肖文章"
    return desired_output or name or "当前任务素材"


def _role_scene_actions(output_medium: str, text: str, audience: str) -> tuple[str, str, list[str], str, list[str]]:
    if output_medium == "image_layout":
        return (
            "小红书信息图排版视觉编辑" if "小红书" in text else "信息图排版视觉编辑",
            "坐在电脑前",
            ["提炼标题", "压缩正文", "安排字号、颜色、留白和高亮"],
            "读者第一眼能看懂，并愿意继续看或收藏",
            ["文字过密", "排版空泛", "只讲风格不讲布局"],
        )
    if output_medium == "comic_script":
        return (
            "信息漫画故事编辑",
            "坐在白板前",
            ["提取核心断言和对偶面", "判断哪些内容适合变成画面", "把系统性内容改成可读的视觉叙事"],
            "画面主体清楚，叙事元素能从原文推导",
            ["凭空虚构", "分镜过密", "主体不突出"],
        )
    if output_medium == "title_topic":
        return (
            "标题策划" if "标题" in text and "选题" not in text else "内容选题编辑",
            "坐在选题墙前",
            ["找核心断言", "反推读者心里的问题", "把素材压成有张力但不标题党的标题和选题"],
            "标题有进入感，但不牺牲真实性",
            ["标题党", "只改表面", "不反推选题"],
        )
    if output_medium == "learning_structure":
        return (
            "知识结构教练",
            "站在白板前",
            ["搭知识骨架", "拆关键概念和变量关系", "给能立刻练习的最小输出"],
            "学习者能理解、能复述、能使用",
            ["堆知识点", "无骨架", "无练习"],
        )
    if output_medium == "code_task":
        role = "代码审查工程师" if any(token in text for token in ["review", "审查", "代码评审"]) else "产品需求拆解师"
        return (
            role,
            "面对一份需求文档",
            ["澄清边界", "拆出可执行任务", "控制改动范围和风险"],
            "结构清楚、可执行、不做过度设计",
            ["需求不清", "过度设计", "改动不可控"],
        )
    if output_medium == "review_feedback":
        return (
            "复盘教练",
            "拿着一份复盘记录",
            ["先还原处境", "找出关键误差", "留下下一步可执行动作"],
            "不回避问题，但能让人愿意继续改",
            ["只做情绪判断", "没有行动项", "复盘失焦"],
        )
    if output_medium == "agent_design":
        return (
            "提示词系统设计师",
            "面对一张流程草图",
            ["拆需求变量", "锁住关键输入输出", "把流程收敛成可复用规则"],
            "结构清楚、可追踪、能被重复使用",
            ["过度设计", "变量混乱", "闭环不完整"],
        )
    if "生肖" in text or "老人" in text or "老年" in audience:
        return (
            "懂生肖、懂家事、懂钱事的老朋友",
            "坐在小区长椅上",
            ["先说他们最近的处境", "把担心和委屈说圆", "给一句稳妥的生活提醒"],
            "通俗、安定、能让老人愿意读下去",
            ["鸡汤", "AI味", "结构平铺"],
        )
    if output_medium == "article":
        return (
            "内容写作编辑",
            "坐在电脑前",
            ["先抓核心问题", "把信息整理成可读段落", "收成能直接发布的表达"],
            "不跑题、不空泛、能直接使用",
            ["鸡汤", "AI味", "结构平铺"],
        )
    return (
        "任务拆解编辑",
        "面对一份需求清单",
        ["先澄清任务对象", "拆出关键步骤", "收成能直接使用的输出"],
        "结构清楚、可执行、不过度浮夸",
        ["任务过散", "输出空泛", "不可复用"],
    )


def build_task_profile(project: dict[str, Any]) -> dict[str, Any]:
    name = _project_value(project, "name", "task_name")
    scenario = _project_value(project, "scenario", "use_case")
    goal = _project_value(project, "goal")
    audience = _project_value(project, "audience", "target_audience")
    desired_output = _project_value(project, "desired_output")
    avoid = _project_value(project, "avoid")
    style = _project_value(project, "style_preference")
    criteria = _project_value(project, "evaluation_criteria")
    text = " ".join([name, scenario, goal, audience, desired_output, avoid, style, criteria]).strip()

    output_medium = _pick_output_medium(text)
    platform = _pick_platform(text)
    task_action = _pick_task_action(text)
    role_archetype, work_scene, action_chain, judgment_standard, risks = _role_scene_actions(output_medium, text, audience)
    target_object = _build_target_object(project, output_medium)

    return {
        "task_type": task_action,
        "output_medium": output_medium,
        "target_audience": audience,
        "platform": platform,
        "input_object": target_object,
        "desired_output": desired_output,
        "role_archetype": role_archetype,
        "work_scene": work_scene,
        "target_object": target_object,
        "action_chain": action_chain[:3],
        "judgment_standard": judgment_standard if criteria == "" else criteria,
        "risks": risks,
        "project_name": name,
        "scenario": scenario,
        "goal": goal,
        "style_preference": style,
    }


def build_identity_intro(task_profile: dict[str, Any]) -> str:
    role = str(task_profile.get("role_archetype", "")).strip() or "任务拆解编辑"
    scene = str(task_profile.get("work_scene", "")).strip() or "面对一份任务草稿"
    target = str(task_profile.get("target_object", "")).strip() or "当前任务素材"
    actions = [str(item).strip() for item in (task_profile.get("action_chain", []) or [])]
    actions = [action.removeprefix("先").removeprefix("再").removeprefix("最后").strip() for action in actions]
    while len(actions) < 3:
        actions.append("收成可直接使用的结果")
    standard = str(task_profile.get("judgment_standard", "")).strip() or "结构清楚、可执行、可直接使用"
    prefix = "你不是自动摘要工具。" if task_profile.get("output_medium") in {"comic_script", "agent_design"} else ""
    identity = f"{prefix}你像一位{role}，{scene}处理{target}：先{actions[0]}，再{actions[1]}，最后{actions[2]}。你的判断标准是{standard}。"
    if len(identity) > 120:
        identity = f"{prefix}你像一位{role}，{scene}处理{target}：先{actions[0]}，再{actions[1]}，最后{actions[2]}。"
    return identity.strip()


def extract_identity_intro(prompt_text: str) -> str:
    text = str(prompt_text or "")
    if not text:
        return ""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("【人物速写式身份】"):
            body = line.replace("【人物速写式身份】", "", 1).strip()
            following: list[str] = []
            cursor = index + 1
            while cursor < len(lines) and not lines[cursor].startswith("【"):
                if lines[cursor].strip():
                    following.append(lines[cursor].strip())
                cursor += 1
            return " ".join(part for part in [body, *following] if part).strip()
    return ""


def replace_identity_intro(prompt_text: str, identity_text: str) -> str:
    text = str(prompt_text or "")
    identity_line = f"【人物速写式身份】{identity_text.strip()}"
    if not text.strip():
        return identity_line
    lines = text.splitlines()
    output: list[str] = []
    replaced = False
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("【人物速写式身份】"):
            if not replaced:
                output.append(identity_line)
                replaced = True
            index += 1
            while index < len(lines) and not lines[index].startswith("【"):
                index += 1
            continue
        output.append(line)
        index += 1
    if replaced:
        return "\n".join(output)
    return identity_line + "\n" + text.strip()


def _has_role(identity_text: str, task_profile: dict[str, Any]) -> bool:
    role = str(task_profile.get("role_archetype", "")).strip()
    if role and role in identity_text:
        return True
    return "你像一位" in identity_text or "你像一个" in identity_text or "你像一名" in identity_text


def _has_scene(identity_text: str, task_profile: dict[str, Any]) -> bool:
    scene = str(task_profile.get("work_scene", "")).strip()
    if scene and scene in identity_text:
        return True
    return any(token in identity_text for token in ["坐在", "站在", "面对", "拿着", "对着", "白板前", "电脑前", "茶馆里", "长椅上"])


def _has_target_object(identity_text: str, task_profile: dict[str, Any]) -> bool:
    target = str(task_profile.get("target_object", "")).strip()
    if target and any(chunk and chunk in identity_text for chunk in [target, target.replace("和", "")]):
        return True
    return any(token in identity_text for token in ["文章", "内容", "图文", "代码", "标题", "选题", "知识模块", "复盘"])


def _has_action_chain(identity_text: str, task_profile: dict[str, Any]) -> bool:
    actions = [str(item).strip() for item in task_profile.get("action_chain", []) if str(item).strip()]
    hits = sum(1 for action in actions if action and action[:2] in identity_text or action in identity_text)
    if hits >= 2 and all(token in identity_text for token in ["先", "再", "最后"]):
        return True
    return all(token in identity_text for token in ["先", "再", "最后"])


def _has_judgment_standard(identity_text: str) -> bool:
    return "判断标准是" in identity_text or "标准是" in identity_text


def _is_role_mismatch(identity_text: str, task_profile: dict[str, Any]) -> bool:
    output_medium = str(task_profile.get("output_medium", "general"))
    mismatch_tokens = ROLE_MISMATCH_RULES.get(output_medium, [])
    return any(token in identity_text for token in mismatch_tokens)


def validate_identity_intro(identity_text: str, task_profile: dict[str, Any]) -> dict[str, Any]:
    text = str(identity_text or "").strip()
    issues: list[str] = []
    score = 0
    if not text:
        issues.append("缺少身份段")
        return {"valid": False, "score": 0, "issues": issues, "suggested_fix": build_identity_intro(task_profile)}

    role_ok = _has_role(text, task_profile)
    scene_ok = _has_scene(text, task_profile)
    object_ok = _has_target_object(text, task_profile)
    action_ok = _has_action_chain(text, task_profile)
    judgment_ok = _has_judgment_standard(text)

    if any(token in text for token in ABSTRACT_IDENTITY_TOKENS):
        issues.append("缺少具体角色原型，仍然偏抽象助手")
    if not role_ok:
        issues.append("缺少角色原型")
    if not scene_ok:
        issues.append("缺少工作场景")
    if not object_ok:
        issues.append("缺少处理对象")
    if not action_ok:
        issues.append("缺少核心动作链")
    if not judgment_ok:
        issues.append("缺少判断标准")
    if _is_role_mismatch(text, task_profile):
        issues.append("身份段与项目任务类型不匹配")
    if len(text) > 150:
        issues.append("身份段过长，像堆设定")
    if text.startswith("你不是") and "你像" not in text:
        issues.append("只有否定式开头，没有后续人物速写")

    element_count = sum([role_ok, scene_ok, object_ok, action_ok, judgment_ok])
    if element_count <= 1:
        score = 2
    elif element_count == 2:
        score = 4
    elif element_count == 3:
        score = 6
    elif element_count == 4:
        score = 8
    else:
        score = 10
    if _is_role_mismatch(text, task_profile):
        score = min(score, 6)
    if len(text) > 150:
        score = max(0, score - 1)

    return {
        "valid": score >= 7 and not _is_role_mismatch(text, task_profile),
        "score": score,
        "issues": issues,
        "suggested_fix": build_identity_intro(task_profile),
    }
