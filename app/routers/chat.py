"""对话接口：检索调试、SSE 问答、非流式兼容与会话管理。

- `POST /api/retrieve`：只检索，不调模型，便于调试与回归；
- `POST /api/chat`：`stream=true` 走 SSE，`stream=false` 返回 JSON；
- `/api/chat/sessions`：会话列表 / 新建 / 详情 / 重命名 / 删除；
- `/api/chat/config`：前端读取默认改写方案与 LLM 配置状态。
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.config import settings
from app.database import SessionLocal, get_db
from app.models import ChatMessage, ChatSession, Repo
from app.schemas.chat import (
    ChatMessageOut,
    ChatRequest,
    ChatResponse,
    ChatSessionCreate,
    ChatSessionDetail,
    ChatSessionList,
    ChatSessionOut,
    ChatSessionPatch,
    RetrieveDebug,
    RetrieveRequest,
    RetrieveResponse,
    SourceOut,
)
from app.services.rag.context import (
    build_messages,
    prompt_token_estimate,
)
from app.services.rag.llm import (
    LLMError,
    LLMNotConfigured,
    complete_chat,
    stream_chat,
)
from app.services.rag.retriever import retrieve
from app.services.rag.rewrite import rewrite_query

router = APIRouter(prefix="/api", tags=["chat"])

# 2G2核：同时只允许 2 路 LLM 流，避免入库任务与流式回答争内存
_chat_semaphore = asyncio.Semaphore(max(1, settings.chat_max_concurrency))

NO_SOURCE_MESSAGE = (
    "在所选仓库中没有检索到与该问题相关的代码片段，无法基于代码回答。"
    "可以换一个类名、方法名或更具体的问题再试。"
)
CITATION_RE = re.compile(r"\[(\d+)\]")


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------


def _sse(event: str, payload: dict[str, Any]) -> str:
    """拼一条 SSE 消息（event + JSON data）。"""
    data = json.dumps(payload, ensure_ascii=False)
    return f"event: {event}\ndata: {data}\n\n"


def _title_from(message: str) -> str:
    title = (message or "新会话").strip().replace("\n", " ")
    return title[:30] or "新会话"


def _session_out(db: Session, session: ChatSession, message_count: int = 0) -> ChatSessionOut:
    repo = db.get(Repo, session.repo_id) if session.repo_id is not None else None
    return ChatSessionOut(
        id=session.id,
        title=session.title,
        repo_id=session.repo_id,
        repo_name=repo.name if repo is not None else None,
        message_count=message_count,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


def _message_out(message: ChatMessage) -> ChatMessageOut:
    citations = [SourceOut(**item) for item in (message.citations or [])]
    return ChatMessageOut(
        id=message.id,
        role=message.role,
        content=message.content,
        citations=citations,
        created_at=message.created_at,
    )


def _get_or_create_session(
    db: Session, payload: ChatRequest
) -> tuple[ChatSession, bool]:
    """按 session_id 取会话；没传则新建，并把单仓库范围锁定在会话上。"""
    if payload.session_id is not None:
        session = db.get(ChatSession, payload.session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        return session, False

    repo_id: int | None = None
    repo_ids = list(dict.fromkeys(payload.repo_ids or []))
    if len(repo_ids) == 1:
        repo_id = repo_ids[0]
        if db.get(Repo, repo_id) is None:
            raise HTTPException(status_code=404, detail="仓库不存在")
    session = ChatSession(title=_title_from(payload.message), repo_id=repo_id)
    db.add(session)
    db.flush()
    return session, True


def _scope_for(
    session: ChatSession, payload: ChatRequest, created: bool
) -> list[int] | None:
    """会话一旦建立，仓库范围就锁定：单仓库 → 该仓库，否则全部仓库。"""
    if session.repo_id is not None:
        return [session.repo_id]
    if created:
        # 首轮允许传多个仓库（此时会话语义仍为“全部”，仅本轮收窄）
        return list(dict.fromkeys(payload.repo_ids or [])) or None
    return None


def _prepare_turn(
    db: Session, payload: ChatRequest, effective_query: str
) -> tuple[ChatSession, list[SourceOut], RetrieveDebug, str]:
    """建/取会话 + 检索 + 记录用户消息（不提交，由调用方统一提交）。"""
    session, created = _get_or_create_session(db, payload)
    scope = _scope_for(session, payload, created)
    sources, debug = retrieve(
        db, effective_query, repo_ids=scope, top_k=payload.top_k
    )
    db.add(ChatMessage(session_id=session.id, role="user", content=payload.message))
    return session, sources, debug, "created" if created else "existing"


def _persist_assistant(
    session_id: int,
    content: str,
    sources: list[SourceOut],
    debug: RetrieveDebug,
    usage: dict[str, Any],
    rewrite_mode: str,
    finish_reason: str,
) -> None:
    """保存助手消息并刷新会话时间；独立开 Session，避免流式期间依赖已关闭的请求会话。"""
    db = SessionLocal()
    try:
        retrieval = debug.model_dump()
        retrieval["rewrite_mode"] = rewrite_mode
        retrieval["finish_reason"] = finish_reason
        db.add(
            ChatMessage(
                session_id=session_id,
                role="assistant",
                content=content or "（无回答）",
                citations=[source.model_dump() for source in sources] or None,
                retrieval=retrieval,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
            )
        )
        session = db.get(ChatSession, session_id)
        if session is not None:
            session.updated_at = func.now()
        db.commit()
    finally:
        db.close()


def _citation_warning(content: str, source_count: int) -> bool:
    """回答里出现超出 sources 范围的编号时给出提示。"""
    return any(int(match.group(1)) > source_count for match in CITATION_RE.finditer(content))


def _usage(
    usage: dict[str, Any], messages: list[dict[str, str]] | None, content: str
) -> dict[str, int]:
    """优先用厂商返回的 usage，缺失时按字符数估算。"""
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if prompt_tokens is None and messages is not None:
        prompt_tokens = prompt_token_estimate(messages)
    if completion_tokens is None:
        completion_tokens = max(1, int(len(content or "") / 3.2))
    return {
        "prompt_tokens": int(prompt_tokens or 0),
        "completion_tokens": int(completion_tokens or 0),
    }


# ---------------------------------------------------------------------------
# 检索调试
# ---------------------------------------------------------------------------


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve_api(payload: RetrieveRequest, db: Session = Depends(get_db)):
    """混合检索，不调用模型。"""
    sources, debug = retrieve(
        db,
        payload.query,
        repo_ids=payload.repo_ids,
        top_k=payload.top_k,
        candidate_k=payload.candidate_k,
    )
    return RetrieveResponse(
        query=payload.query,
        effective_query=payload.query,
        took_ms=debug.took_ms,
        strategy=debug.strategy,
        sources=sources,
        debug=debug,
    )


# ---------------------------------------------------------------------------
# 问答
# ---------------------------------------------------------------------------


@router.get("/chat/config")
def chat_config():
    """前端设置页启动时读取：默认改写方案 + LLM 是否可用。"""
    return {
        "llm_configured": settings.llm_configured,
        "llm_model": settings.llm_model or None,
        "default_rewrite_mode": settings.chat_rewrite_mode,
        "top_k": settings.retrieval_top_k,
        "max_concurrency": settings.chat_max_concurrency,
    }


def _chat_complete(
    payload: ChatRequest, effective_query: str, mode_used: str
) -> ChatResponse:
    """非流式问答：检索 → prompt → 一次 complete → 落库。"""
    db = SessionLocal()
    try:
        session, sources, debug, _ = _prepare_turn(db, payload, effective_query)
        db.commit()

        if not sources:
            content, usage, finish_reason = NO_SOURCE_MESSAGE, {}, "no_sources"
        else:
            messages = build_messages(payload.message, sources, payload.history)
            try:
                content, usage = asyncio.run(complete_chat(messages))
                finish_reason = "stop"
            except LLMNotConfigured as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            except LLMError as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc

        _persist_assistant(
            session.id, content, sources, debug, usage, mode_used, finish_reason
        )
        return ChatResponse(
            content=content,
            sources=sources,
            session_id=session.id,
            effective_query=effective_query,
        )
    finally:
        db.close()


@router.post("/chat", response_model=None)
async def chat(payload: ChatRequest):
    """流式（SSE）或非流式问答；仓库范围在会话建立后锁定。"""
    if not settings.llm_configured:
        raise HTTPException(
            status_code=503,
            detail=(
                "LLM 未配置：请在 .env 中设置 LLM_BASE_URL、LLM_API_KEY、LLM_MODEL "
                "后重启后端"
            ),
        )

    effective_query, mode_used = await rewrite_query(
        payload.message, payload.history, payload.rewrite_mode
    )

    if not payload.stream:
        return await run_in_threadpool(_chat_complete, payload, effective_query, mode_used)

    prepared = await run_in_threadpool(_prepare_stream_turn, payload, effective_query)
    return StreamingResponse(
        _event_stream(payload, prepared, mode_used),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _prepare_stream_turn(payload: ChatRequest, effective_query: str) -> dict[str, Any]:
    """流式前的同步准备：建会话、检索、写用户消息。"""
    db = SessionLocal()
    try:
        session, sources, debug, state = _prepare_turn(db, payload, effective_query)
        db.commit()
        return {
            "session_id": session.id,
            "title": session.title,
            "repo_id": session.repo_id,
            "created": state == "created",
            "sources": sources,
            "debug": debug,
        }
    finally:
        db.close()


async def _event_stream(
    payload: ChatRequest, prepared: dict[str, Any], mode_used: str
):
    """SSE 生成器：session → sources → delta* → done / error。"""
    started = time.perf_counter()
    sources: list[SourceOut] = prepared["sources"]
    debug: RetrieveDebug = prepared["debug"]
    session_id: int = prepared["session_id"]

    yield _sse(
        "session",
        {
            "session_id": session_id,
            "title": prepared["title"],
            "repo_id": prepared["repo_id"],
            "repo_ids": [prepared["repo_id"]] if prepared["repo_id"] is not None else None,
            "created": prepared["created"],
            "rewrite_mode": mode_used,
        },
    )
    yield _sse(
        "sources",
        {
            "sources": [source.model_dump() for source in sources],
            "retrieval": debug.model_dump(),
        },
    )

    chunks: list[str] = []
    usage: dict[str, Any] = {}
    finish_reason = "stop"
    persisted = False
    messages: list[dict[str, str]] | None = None

    try:
        if not sources:
            chunks.append(NO_SOURCE_MESSAGE)
            finish_reason = "no_sources"
            yield _sse("delta", {"text": NO_SOURCE_MESSAGE})
        else:
            messages = build_messages(payload.message, sources, payload.history)
            try:
                async with _chat_semaphore:
                    async for delta in stream_chat(messages, usage_out=usage):
                        chunks.append(delta)
                        yield _sse("delta", {"text": delta})
            except LLMError as exc:
                finish_reason = "error"
                yield _sse("error", {"code": "llm_error", "message": str(exc)})
            except asyncio.CancelledError:
                finish_reason = "canceled"
                raise

        content = "".join(chunks)
        latency_ms = int((time.perf_counter() - started) * 1000)
        usage_out = _usage(usage, messages, content)

        if content:
            await run_in_threadpool(
                _persist_assistant,
                session_id,
                content,
                sources,
                debug,
                usage_out,
                mode_used,
                finish_reason,
            )
            persisted = True

        if finish_reason == "error":
            return

        yield _sse(
            "done",
            {
                "finish_reason": finish_reason,
                "latency_ms": latency_ms,
                "usage": usage_out,
                "rewrite_mode": mode_used,
                "citation_warning": _citation_warning(content, len(sources)),
            },
        )
    finally:
        # 客户端提前断开时也保存已生成的部分，避免会话历史缺口
        if not persisted and chunks:
            _persist_assistant(
                session_id,
                "".join(chunks),
                sources,
                debug,
                _usage(usage, messages, "".join(chunks)),
                mode_used,
                finish_reason,
            )


# ---------------------------------------------------------------------------
# 会话管理
# ---------------------------------------------------------------------------


@router.get("/chat/sessions", response_model=ChatSessionList)
def list_sessions(db: Session = Depends(get_db)):
    """会话列表，按最近更新时间倒序（最多 100 条）。"""
    sessions = list(
        db.scalars(
            select(ChatSession).order_by(ChatSession.updated_at.desc()).limit(100)
        ).all()
    )
    if not sessions:
        return ChatSessionList(items=[])

    counts = dict(
        db.execute(
            select(ChatMessage.session_id, func.count())
            .where(ChatMessage.session_id.in_([s.id for s in sessions]))
            .group_by(ChatMessage.session_id)
        ).all()
    )
    repo_ids = {s.repo_id for s in sessions if s.repo_id is not None}
    repos = (
        {repo.id: repo.name for repo in db.scalars(select(Repo).where(Repo.id.in_(repo_ids)))}
        if repo_ids
        else {}
    )
    items = [
        ChatSessionOut(
            id=session.id,
            title=session.title,
            repo_id=session.repo_id,
            repo_name=repos.get(session.repo_id) if session.repo_id else None,
            message_count=int(counts.get(session.id, 0)),
            created_at=session.created_at,
            updated_at=session.updated_at,
        )
        for session in sessions
    ]
    return ChatSessionList(items=items)


@router.post("/chat/sessions", response_model=ChatSessionOut, status_code=201)
def create_session(payload: ChatSessionCreate, db: Session = Depends(get_db)):
    """新建会话；repo_id 为空表示“所有仓库”。"""
    if payload.repo_id is not None and db.get(Repo, payload.repo_id) is None:
        raise HTTPException(status_code=404, detail="仓库不存在")
    session = ChatSession(title=payload.title or "新会话", repo_id=payload.repo_id)
    db.add(session)
    db.commit()
    db.refresh(session)
    return _session_out(db, session, 0)


@router.get("/chat/sessions/{session_id}", response_model=ChatSessionDetail)
def get_session(session_id: int, db: Session = Depends(get_db)):
    """会话详情：基本信息 + 全部历史消息（含引用快照）。"""
    session = db.get(ChatSession, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    messages = list(
        db.scalars(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.id)
        ).all()
    )
    return ChatSessionDetail(
        session=_session_out(db, session, len(messages)),
        messages=[_message_out(message) for message in messages],
    )


@router.patch("/chat/sessions/{session_id}", response_model=ChatSessionOut)
def rename_session(
    session_id: int, payload: ChatSessionPatch, db: Session = Depends(get_db)
):
    """重命名会话。"""
    session = db.get(ChatSession, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    session.title = payload.title
    db.commit()
    db.refresh(session)
    count = db.scalar(
        select(func.count()).select_from(ChatMessage).where(
            ChatMessage.session_id == session_id
        )
    )
    return _session_out(db, session, int(count or 0))


@router.delete("/chat/sessions/{session_id}")
def delete_session(session_id: int, db: Session = Depends(get_db)):
    """删除会话（消息随外键级联删除）。"""
    session = db.get(ChatSession, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    db.delete(session)
    db.commit()
    return {"deleted": session_id}
