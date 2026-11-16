"""多轮提问改写：none（不改写）/ rule（规则）/ llm（模型改写），默认 rule。

改写的产物只用于检索（effective_query），发给 LLM 的仍是用户原问题，
避免改写误差影响回答内容。
"""

from __future__ import annotations

from typing import Any, Sequence

from app.config import settings
from app.services.rag.llm import complete_chat
from app.services.rag.retriever import extract_identifiers

PRONOUNS = (
    "它", "他", "这个", "那个", "该", "上述", "以上", "刚才", "前面",
    "这些", "那些", "此", "其中", "上面", "之前",
)

REWRITE_PROMPT = """你是检索查询改写器。根据上一轮问题补全本轮问题里的指代，
输出一条独立、可检索的中文查询。只输出改写后的查询本身，不要解释。"""


def _last_user_message(history: Sequence[Any]) -> str:
    for item in reversed(list(history or [])):
        if isinstance(item, dict):
            role, content = item.get("role"), item.get("content")
        else:
            role, content = getattr(item, "role", None), getattr(item, "content", None)
        if role == "user" and content:
            return str(content)
    return ""


def _needs_context(message: str) -> bool:
    """含指代词或问题很短时，才需要借助上一轮补全。"""
    text = (message or "").strip()
    return len(text) <= 12 or any(word in text for word in PRONOUNS)


def rule_rewrite(message: str, history: Sequence[Any]) -> str:
    """规则改写：把上一轮的标识符补到当前问题后面，供关键词通道使用。"""
    if not _needs_context(message):
        return message
    previous = _last_user_message(history)
    if not previous:
        return message
    identifiers = extract_identifiers(previous, limit=3)
    if not identifiers:
        return message
    return f"{message}（上下文标识符：{' '.join(identifiers)}）"


async def rewrite_query(
    message: str,
    history: Sequence[Any] | None = None,
    mode: str | None = None,
) -> tuple[str, str]:
    """返回 (effective_query, 实际使用的模式)；llm 失败时自动降级为 rule。"""
    history = list(history or [])
    mode = (mode or settings.chat_rewrite_mode or "rule").lower()
    if mode not in ("none", "rule", "llm"):
        mode = "rule"

    if mode == "none" or not history:
        return message, mode
    if mode == "rule":
        return rule_rewrite(message, history), "rule"

    # llm 模式：只在确实需要上下文时多花一次调用
    if not _needs_context(message):
        return message, "llm"
    previous = _last_user_message(history)
    if not previous:
        return message, "llm"
    try:
        content, _ = await complete_chat(
            [
                {"role": "system", "content": REWRITE_PROMPT},
                {
                    "role": "user",
                    "content": f"上一轮问题：{previous[:500]}\n本轮问题：{message}",
                },
            ],
            temperature=0.0,
            max_tokens=200,
        )
    except Exception:  # noqa: BLE001 - 改写失败不应阻断问答
        return rule_rewrite(message, history), "rule_fallback"

    lines = [line.strip() for line in (content or "").splitlines() if line.strip()]
    rewritten = lines[0].strip('"“”') if lines else ""
    if not rewritten:
        return rule_rewrite(message, history), "rule_fallback"
    return rewritten[:1000], "llm"
