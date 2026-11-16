"""对话检索 / SSE 问答 / 会话管理的出入参模型。"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SourceOut(BaseModel):
    """一条检索来源，也是回答中 [n] 的映射目标。"""

    index: int
    chunk_id: int
    module_id: int | None = None
    chunk_type: str
    name: str | None = None
    repo_id: int
    repo_name: str
    file_id: int
    file_path: str
    language: str | None = None
    start_line: int
    end_line: int
    score: float
    content: str


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    repo_ids: list[int] | None = Field(default=None, description="为空表示全部已入库仓库")
    top_k: int | None = Field(default=None, ge=1, le=10)
    candidate_k: int | None = Field(default=None, ge=5, le=100)


class RetrieveDebug(BaseModel):
    vector_hits: int = 0
    keyword_hits: int = 0
    filtered: int = 0
    truncated: int = 0
    took_ms: int = 0
    strategy: str = "hybrid"


class RetrieveResponse(BaseModel):
    query: str
    effective_query: str
    took_ms: int
    strategy: str
    sources: list[SourceOut]
    debug: RetrieveDebug


class ChatHistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    repo_ids: list[int] | None = None
    session_id: int | None = None
    history: list[ChatHistoryItem] = Field(default_factory=list)
    top_k: int | None = Field(default=None, ge=1, le=10)
    stream: bool = True
    rewrite_mode: Literal["none", "rule", "llm"] | None = Field(
        default=None, description="不传则用后端默认（CHAT_REWRITE_MODE）"
    )


class ChatResponse(BaseModel):
    """非流式返回，字段与前端旧契约兼容。"""

    content: str
    sources: list[SourceOut] = Field(default_factory=list)
    session_id: int
    effective_query: str | None = None


class ChatSessionCreate(BaseModel):
    repo_id: int | None = Field(default=None, description="为空表示所有仓库")
    title: str | None = Field(default=None, max_length=255)


class ChatSessionPatch(BaseModel):
    title: str = Field(min_length=1, max_length=255)


class ChatMessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    role: str
    content: str
    citations: list[SourceOut] = Field(default_factory=list)
    created_at: datetime


class ChatSessionOut(BaseModel):
    id: int
    title: str
    repo_id: int | None = None
    repo_name: str | None = None
    message_count: int = 0
    created_at: datetime
    updated_at: datetime


class ChatSessionList(BaseModel):
    items: list[ChatSessionOut]


class ChatSessionDetail(BaseModel):
    session: ChatSessionOut
    messages: list[ChatMessageOut]
