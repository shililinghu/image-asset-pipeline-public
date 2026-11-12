from __future__ import annotations

from models import make_feedback
from utils import clamp_score, prompt_multiline, prompt_optional, prompt_required


def collect_feedback(project_id: str, prompt_id: str) -> dict:
    print("请录入真实使用反馈。")
    data = {
        "project_id": project_id,
        "prompt_id": prompt_id,
        "real_scenario": prompt_required("实际使用场景"),
        "what_worked": prompt_multiline("好用的地方"),
        "what_failed": prompt_multiline("不好用的地方"),
        "voice_match_score": clamp_score(prompt_required("输出结果像不像我（0-10）")),
        "continue_use": prompt_optional("是否愿意继续使用（yes/no）", "yes").lower(),
        "need_more": prompt_multiline("需要增强什么"),
        "need_less": prompt_multiline("需要减少什么"),
        "overall_score": clamp_score(prompt_required("我的总评分（0-10）")),
    }
    return make_feedback(data)
