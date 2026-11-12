from __future__ import annotations

from llm_client import LLMClient
from models import make_test_result
from utils import print_box, prompt_multiline


def build_manual_test_prompt(prompt: dict, case: dict) -> str:
    return f"""请把以下内容复制到任意模型中测试。

【系统提示词 / system_prompt】
{prompt['full_prompt']}

【用户输入 / user_prompt】
{case['input']}

【期望关注点】
{case['expected_focus']}
"""


def run_test(project_id: str, prompt: dict, case: dict, llm: LLMClient) -> dict:
    if llm.is_available():
        try:
            output = llm.generate_text(prompt["full_prompt"], case["input"])
            return make_test_result(project_id, prompt["prompt_id"], case["case_id"], output, "api")
        except Exception as exc:
            print(f"API 测试失败，切换到手动模式。原因：{exc}")

    print_box("手动测试：请复制以下内容到 ChatGPT / Codex / 其他模型")
    print(build_manual_test_prompt(prompt, case))
    output = prompt_multiline("请把模型返回的测试结果粘贴回来")
    return make_test_result(project_id, prompt["prompt_id"], case["case_id"], output, "manual")
