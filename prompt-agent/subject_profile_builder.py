from __future__ import annotations

from typing import Any

from models import create_subject_profile
from storage import JsonStore


DEFAULT_MUST_KEEP = ["同一主体", "主体年龄感", "发型", "面部气质", "整体识别感"]
DEFAULT_MUST_AVOID = ["换脸", "换性别", "变成不同年龄", "变成动漫角色", "主体漂移", "多人出现"]
DEFAULT_ANCHORS = ["脸型保持一致", "发型保持一致", "人物气质保持一致", "不要换成不同人物"]


def build_subject_profile(
    request: dict[str, Any],
    reference_images: list[dict[str, Any]] | None = None,
    *,
    subject_description: str = "",
    store: JsonStore | None = None,
    save: bool = True,
) -> dict[str, Any]:
    refs = reference_images or []
    title = str(request.get("title") or request.get("topic") or request.get("raw_idea") or "主体").strip()
    description = (
        subject_description
        or request.get("subject_description")
        or request.get("description")
        or request.get("requirements")
        or request.get("raw_idea")
        or request.get("topic")
        or title
    )
    description = str(description or "").strip()
    ref_ids = [str(item.get("reference_image_id", "")) for item in refs if item.get("reference_image_id")]
    ref_paths = [str(item.get("path") or item.get("image_path") or "") for item in refs if item.get("path") or item.get("image_path")]
    source = "mixed" if refs and description else "reference_image" if refs else "text_description"
    profile = create_subject_profile(
        {
            "request_id": request.get("request_id", ""),
            "source": source,
            "subject_title": title,
            "subject_description": description,
            "identity_keywords": _keywords(description),
            "visual_anchors": DEFAULT_ANCHORS,
            "must_keep": DEFAULT_MUST_KEEP,
            "must_avoid": DEFAULT_MUST_AVOID,
            "reference_image_ids": ref_ids,
            "reference_image_paths": ref_paths,
            "consistency_instruction": build_subject_lock_instruction(
                {
                    "source": source,
                    "subject_description": description,
                    "must_keep": DEFAULT_MUST_KEEP,
                    "must_avoid": DEFAULT_MUST_AVOID,
                    "reference_image_paths": ref_paths,
                }
            ),
        }
    )
    if save:
        (store or JsonStore()).add("subject_profiles", profile)
    return profile


def build_subject_lock_instruction(subject_profile: dict[str, Any] | None) -> str:
    if not subject_profile:
        return ""
    description = str(subject_profile.get("subject_description", "") or subject_profile.get("subject_title", "") or "同一主体").strip()
    must_keep = "、".join(subject_profile.get("must_keep", []) or DEFAULT_MUST_KEEP)
    must_avoid = "、".join(subject_profile.get("must_avoid", []) or DEFAULT_MUST_AVOID)
    reference_note = ""
    if subject_profile.get("reference_image_paths"):
        reference_note = "\n请以参考样图中的主体为唯一人物锚点。当前 provider 如果支持参考图，则使用样图；如果不支持，也必须在文本描述上保持主体一致。"
    return (
        "【主体锁定】\n"
        "本轮生成必须保持同一主体：\n"
        f"{description}\n"
        f"必须保持：{must_keep}。\n"
        f"禁止：{must_avoid}。\n"
        "允许变化：服装、主题、场景、光线、构图、道具、色调、背景。\n"
        "禁止变化：主体身份、脸型、年龄感、性别、核心气质、人物识别感。"
        f"{reference_note}"
    )


def inject_subject_lock(prompt: str, subject_profile: dict[str, Any] | None) -> str:
    text = str(prompt or "")
    if not subject_profile or "【主体锁定】" in text or "【主体一致性要求】" in text:
        return text
    return f"{build_subject_lock_instruction(subject_profile)}\n\n【画面 Prompt】\n{text}"


def subject_profile_metadata(subject_profile: dict[str, Any] | None, enabled: bool = True) -> dict[str, Any]:
    if not subject_profile:
        return {"subject_lock_enabled": False}
    return {
        "subject_profile_id": subject_profile.get("subject_profile_id", ""),
        "subject_lock_enabled": bool(enabled),
        "subject_source": subject_profile.get("source", ""),
        "subject_description": subject_profile.get("subject_description", ""),
        "reference_image_ids": list(subject_profile.get("reference_image_ids", []) or []),
        "reference_image_paths": list(subject_profile.get("reference_image_paths", []) or []),
        "provider_supports_reference_image": False,
    }


def validate_subject_lock_input(request: dict[str, Any], reference_images: list[dict[str, Any]] | None = None, subject_description: str = "") -> tuple[bool, str]:
    if reference_images:
        return True, ""
    text = " ".join(
        str(value or "")
        for value in [
            subject_description,
            request.get("description"),
            request.get("requirements"),
            request.get("raw_idea"),
            request.get("topic"),
            request.get("title"),
        ]
    ).strip()
    if len(text) < 10:
        return False, "没有样图时，请填写更明确的主体描述，否则无法保证主体一致性。"
    return True, ""


def _keywords(description: str) -> list[str]:
    text = str(description or "").replace("，", ",").replace("。", ",")
    values = [item.strip() for item in text.split(",") if item.strip()]
    return values[:8]
