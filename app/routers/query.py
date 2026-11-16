"""问答接口：保留原路径，作为 `/api/chat`（stream=false）的薄封装。

请求体沿用一期的 `QueryRequest`（question / session_id / top_k），
内部转成 ChatRequest 后走同一条检索 + 生成链路；不传改写方案时按 `none` 处理，
避免旧调用方为一次内部调用额外承担改写开销。
"""

from __future__ import annotations

from fastapi import APIRouter

from app.routers.chat import _chat_complete
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.query import QueryRequest

router = APIRouter(prefix="/api", tags=["query"])


@router.post("/query", response_model=ChatResponse)
def query(payload: QueryRequest):
    """问答（非流式，兼容旧契约）。"""
    request = ChatRequest(
        message=payload.question,
        session_id=payload.session_id,
        top_k=payload.top_k,
        stream=False,
        rewrite_mode="none",
    )
    return _chat_complete(request, payload.question, "none")
