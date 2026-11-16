"""上下文组装：来源块格式化、历史裁剪与 prompt 拼装。"""

from __future__ import annotations

from typing import Any, Sequence

from app.schemas.chat import SourceOut

SYSTEM_PROMPT = """你是代码库问答助手。请严格依据提供的代码来源回答：
1. 先给结论，再说明关键实现路径；不要整段复述源码。
2. 涉及具体实现时给出行号，并用 [n] 标注引用的来源编号；n 必须落在来源范围内。
3. 来源不足时直接说明“现有代码中没有找到相关实现”，不要编造。
4. 用中文回答；类名、方法名、代码保持原文。"""

MAX_HISTORY_MESSAGES = 6
MAX_HISTORY_CHARS = 800


def estimate_tokens(text: str) -> int:
    """粗略 token 估算：中文/代码混合按约 3.2 字符/token。"""
    return max(1, int(len(text or "") / 3.2))


def format_source(source: SourceOut) -> str:
    """单个来源块，编号与 sources[].index 一致，也是回答中 [n] 的映射目标。"""
    name = f" {source.name}" if source.name else ""
    return (
        f"[{source.index}] repo: {source.repo_name} | file: {source.file_path}\n"
        f"    | type: {source.chunk_type}{name} | "
        f"lines: {source.start_line}-{source.end_line} | score: {source.score:.2f}\n"
        f"```{source.language or ''}\n{source.content}\n```"
    )


def _history_item(item: Any) -> tuple[str, str]:
    """兼容 Pydantic 模型与 dict 两种历史消息形态。"""
    if isinstance(item, dict):
        return str(item.get("role", "user")), str(item.get("content", ""))
    return str(getattr(item, "role", "user")), str(getattr(item, "content", ""))


def build_messages(
    question: str,
    sources: Sequence[SourceOut],
    history: Sequence[Any] | None = None,
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    """系统约束 + 最近 6 条历史 + 来源块 + 当前问题。"""
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt or SYSTEM_PROMPT}
    ]

    for item in list(history or [])[-MAX_HISTORY_MESSAGES:]:
        role, content = _history_item(item)
        if role not in ("user", "assistant") or not content:
            continue
        messages.append({"role": role, "content": content[:MAX_HISTORY_CHARS]})

    blocks = "\n\n".join(format_source(source) for source in sources)
    messages.append(
        {
            "role": "user",
            "content": f"以下是从代码库检索到的来源：\n\n{blocks}\n\n用户问题：{question}",
        }
    )
    return messages


def prompt_token_estimate(messages: Sequence[dict[str, str]]) -> int:
    """估算 prompt token，用于 done 事件与日志。"""
    return sum(estimate_tokens(message.get("content", "")) for message in messages)
