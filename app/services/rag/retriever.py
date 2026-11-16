"""混合检索：向量召回 + 标识符关键词召回，RRF 融合，不做重排。

流程：
1. 向量通道：bge-small 编码查询 → Chroma `code_chunks`，n_results=30，按 repo_id 过滤；
2. 关键词通道：从问题里抽英文标识符（类名/方法名），在 PG 中匹配 name/signature/content；
3. RRF（k=60）融合：向量权重 0.7、关键词权重 0.3；
4. 阈值过滤、父子重叠去重（成员优先）、上溯 module_id、超长切片展开/截断、总字符预算。

检索本身是同步阻塞的（embedding + SQL），在路由层通过线程池调用。
"""

from __future__ import annotations

import re
import threading
import time
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Chunk, CodeFile, Repo
from app.schemas.chat import RetrieveDebug, SourceOut
from app.services.indexing.embedder import (
    get_chroma_client,
    get_embedder,
    get_or_create_collection,
)

MODULE_CHUNK_TYPES = ("class", "interface", "enum")
RRF_K = 60

# 超长切片处理阈值（字符）
LONG_CHUNK_CHARS = 6000
TYPE_HEADER_LINES = 20
HEAD_LINES = 120
TAIL_LINES = 40
MAX_MEMBERS = 3

IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
# 只过滤英文虚词；代码标识符（Gson/TypeToken/json 等）一律保留
STOPWORDS = {
    "the", "and", "for", "with", "this", "that", "what", "how", "why", "does",
    "did", "are", "was", "were", "from", "into", "can", "could", "should",
    "would", "you", "your", "our", "its", "has", "have", "had", "use", "using",
    "used", "about", "which", "when", "where", "there", "their", "them", "then",
    "than", "not", "but", "all", "any", "some", "will", "just", "like", "get",
}

_collection: Any = None
_collection_lock = threading.Lock()


def get_collection() -> Any:
    """进程内复用 Chroma 集合句柄，避免每次请求都新建 PersistentClient。"""
    global _collection
    if _collection is None:
        with _collection_lock:
            if _collection is None:
                client = get_chroma_client(settings.chroma_dir)
                _collection = get_or_create_collection(client)
    return _collection


def extract_identifiers(text: str, limit: int = 8) -> list[str]:
    """抽取英文标识符，用作关键词通道的查询词。"""
    seen: set[str] = set()
    out: list[str] = []
    for match in IDENTIFIER_RE.finditer(text or ""):
        token = match.group(0)
        if token.lower() in STOPWORDS or token in seen:
            continue
        seen.add(token)
        out.append(token)
        if len(out) >= limit:
            break
    return out


def resolve_repo_ids(db: Session, repo_ids: list[int] | None) -> list[int]:
    """解析检索范围：None → 全部已入库仓库；显式列表去重后原样返回。"""
    if repo_ids is not None:
        return list(dict.fromkeys(int(i) for i in repo_ids if i is not None))
    return list(db.scalars(select(Repo.id).where(Repo.indexed_at.is_not(None))).all())


def _vector_channel(
    query: str, repo_ids: list[int], candidate_k: int
) -> list[tuple[int, float]]:
    """Chroma 向量召回，返回 (chunk_id, similarity) 列表。"""
    collection = get_collection()
    total = collection.count()
    if total == 0:
        return []

    vector = get_embedder().encode([query])[0]
    kwargs: dict[str, Any] = {
        "query_embeddings": [vector],
        "n_results": min(candidate_k, total),
        "include": ["metadatas", "distances"],
    }
    if repo_ids:
        kwargs["where"] = {"repo_id": {"$in": repo_ids}}

    result = collection.query(**kwargs)
    ids = (result.get("ids") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    hits: list[tuple[int, float]] = []
    for raw_id, meta, distance in zip(ids, metas, distances):
        meta = meta or {}
        chunk_id = meta.get("chunk_id", raw_id)
        try:
            chunk_id = int(chunk_id)
        except (TypeError, ValueError):
            continue
        similarity = max(0.0, min(1.0, 1.0 - float(distance)))
        hits.append((chunk_id, similarity))
    return hits


def _keyword_channel(
    db: Session, identifiers: list[str], repo_ids: list[int], candidate_k: int
) -> list[tuple[int, float]]:
    """PG 关键词召回：name 精确 > name/signature 子串 > content 子串。"""
    if not identifiers:
        return []

    # 1) name / signature 命中的切片（不拉 content，避免大文本传输）
    name_conds = []
    for token in identifiers:
        like = f"%{token}%"
        name_conds.append(Chunk.name.ilike(like))
        name_conds.append(Chunk.signature.ilike(like))
    stmt = (
        select(Chunk.id, Chunk.name, Chunk.signature)
        .join(CodeFile, Chunk.file_id == CodeFile.id)
        .where(or_(*name_conds))
    )
    if repo_ids:
        stmt = stmt.where(CodeFile.repo_id.in_(repo_ids))

    scores: dict[int, float] = {}
    for chunk_id, name, signature in db.execute(stmt).all():
        low_name = (name or "").lower()
        low_sig = (signature or "").lower()
        score = 0.0
        for token in identifiers:
            token_low = token.lower()
            if low_name == token_low:
                score += 3.0
            elif token_low in low_name:
                score += 2.0
            if token_low in low_sig:
                score += 1.5
        if score:
            scores[chunk_id] = scores.get(chunk_id, 0.0) + score

    # 2) content 命中的切片（只取 id，限制条数）
    content_conds = [Chunk.content.ilike(f"%{token}%") for token in identifiers]
    content_stmt = (
        select(Chunk.id)
        .join(CodeFile, Chunk.file_id == CodeFile.id)
        .where(or_(*content_conds))
        .limit(candidate_k * 10)
    )
    if repo_ids:
        content_stmt = content_stmt.where(CodeFile.repo_id.in_(repo_ids))
    for chunk_id in db.scalars(content_stmt).all():
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0

    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return ordered[:candidate_k]


def _fuse(
    vector_hits: list[tuple[int, float]],
    keyword_hits: list[tuple[int, float]],
    vector_weight: float,
    keyword_weight: float,
) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
    """RRF 融合；同时保留向量相似度与关键词分数，用于展示分与阈值判定。"""
    fused: dict[int, float] = {}
    similarities: dict[int, float] = {}
    keyword_scores: dict[int, float] = {}

    for rank, (chunk_id, similarity) in enumerate(vector_hits, start=1):
        fused[chunk_id] = fused.get(chunk_id, 0.0) + vector_weight / (RRF_K + rank)
        similarities[chunk_id] = similarity
    for rank, (chunk_id, score) in enumerate(keyword_hits, start=1):
        fused[chunk_id] = fused.get(chunk_id, 0.0) + keyword_weight / (RRF_K + rank)
        keyword_scores[chunk_id] = score

    return fused, similarities, keyword_scores


def _display_score(
    chunk_id: int,
    similarities: dict[int, float],
    keyword_scores: dict[int, float],
    vector_weight: float,
    keyword_weight: float,
) -> float:
    """展示分：向量相似度与关键词命中归一化后的加权和，便于前端直观理解。"""
    similarity = similarities.get(chunk_id, 0.0)
    keyword_norm = min(1.0, keyword_scores.get(chunk_id, 0.0) / 4.0)
    return round(vector_weight * similarity + keyword_weight * keyword_norm, 4)


def _dedupe_parent_child(ordered_ids: list[int], db: Session) -> tuple[list[int], int]:
    """父子重叠去重：成员切片命中时丢弃对应的类型切片。"""
    if not ordered_ids:
        return [], 0
    rows = db.execute(
        select(Chunk.id, Chunk.chunk_type, Chunk.parent_chunk_id).where(
            Chunk.id.in_(ordered_ids)
        )
    ).all()
    info = {chunk_id: (chunk_type, parent) for chunk_id, chunk_type, parent in rows}
    candidate_set = set(ordered_ids)
    type_candidates = {
        chunk_id
        for chunk_id in candidate_set
        if info.get(chunk_id, (None, None))[0] in MODULE_CHUNK_TYPES
    }
    parents_with_member = {
        parent
        for chunk_id in candidate_set
        if (parent := info.get(chunk_id, (None, None))[1]) in type_candidates
    }
    kept = [chunk_id for chunk_id in ordered_ids if chunk_id not in parents_with_member]
    return kept, len(ordered_ids) - len(kept)


def _load_rows(db: Session, chunk_ids: list[int]) -> dict[int, tuple[Chunk, CodeFile, Repo]]:
    """一次查询补全切片、文件、仓库信息。"""
    rows = db.execute(
        select(Chunk, CodeFile, Repo)
        .join(CodeFile, Chunk.file_id == CodeFile.id)
        .join(Repo, CodeFile.repo_id == Repo.id)
        .where(Chunk.id.in_(chunk_ids))
    ).all()
    return {chunk.id: (chunk, code_file, repo) for chunk, code_file, repo in rows}


def _module_id(db: Session, chunk: Chunk) -> int | None:
    """向上找类型切片作为 module_id；最多 3 跳，找不到返回 None。"""
    if chunk.chunk_type in MODULE_CHUNK_TYPES:
        return chunk.id
    current = chunk.parent_chunk_id
    for _ in range(3):
        if current is None:
            return None
        parent = db.get(Chunk, current)
        if parent is None:
            return None
        if parent.chunk_type in MODULE_CHUNK_TYPES:
            return parent.id
        current = parent.parent_chunk_id
    return None


def _member_score(member: Chunk, identifiers: list[str]) -> int:
    name = (member.name or "").lower()
    signature = (member.signature or "").lower()
    score = 0
    for token in identifiers:
        token_low = token.lower()
        if name == token_low:
            score += 3
        elif token_low in name:
            score += 2
        if token_low in signature:
            score += 1
    return score


def _head_tail(text: str, limit: int) -> str:
    """头 120 行 + 尾 40 行，中间省略；仍超硬上限则直接截断。"""
    lines = text.splitlines()
    if len(lines) > HEAD_LINES + TAIL_LINES:
        head = "\n".join(lines[:HEAD_LINES])
        tail = "\n".join(lines[-TAIL_LINES:])
        omitted = len(lines) - HEAD_LINES - TAIL_LINES
        out = f"{head}\n// ... 省略 {omitted} 行 ...\n{tail}"
    else:
        out = text
    if len(out) > limit:
        out = out[: max(0, limit - 20)] + "\n// ... 已截断 ..."
    return out


def _prepare_content(
    db: Session, chunk: Chunk, identifiers: list[str]
) -> tuple[str, bool]:
    """按设计文档 3.3 处理超长切片：类型切片展开成员，其余头尾截断。"""
    content = chunk.content or ""
    limit = settings.retrieval_max_source_chars
    if len(content) <= LONG_CHUNK_CHARS:
        return content, False

    if chunk.chunk_type in MODULE_CHUNK_TYPES:
        members = list(
            db.scalars(
                select(Chunk)
                .where(Chunk.parent_chunk_id == chunk.id)
                .order_by(Chunk.start_line)
            ).all()
        )
        if members:
            ranked = sorted(
                members, key=lambda m: (-_member_score(m, identifiers), m.start_line)
            )[:MAX_MEMBERS]
            lines = content.splitlines()
            head = "\n".join(lines[:TYPE_HEADER_LINES])
            parts = [head]
            if len(lines) > TYPE_HEADER_LINES:
                parts.append("    // ... 省略类型其余部分，仅展示命中的成员 ...")
            for member in ranked:
                parts.append(
                    f"    // [{member.name or member.chunk_type}] "
                    f"lines {member.start_line}-{member.end_line}"
                )
                parts.append(member.content or "")
            assembled = "\n".join(parts)
            if len(assembled) > limit:
                return _head_tail(assembled, limit), True
            return assembled, True

    return _head_tail(content, limit), True


def retrieve(
    db: Session,
    query: str,
    repo_ids: list[int] | None = None,
    top_k: int | None = None,
    candidate_k: int | None = None,
) -> tuple[list[SourceOut], RetrieveDebug]:
    """执行混合检索，返回带编号、module_id 与截断后正文的来源列表。"""
    started = time.perf_counter()
    top_k = top_k or settings.retrieval_top_k
    candidate_k = candidate_k or settings.retrieval_candidate_k
    vector_weight = settings.retrieval_vector_weight
    keyword_weight = settings.retrieval_keyword_weight
    debug = RetrieveDebug()

    scope = resolve_repo_ids(db, repo_ids)
    # 显式指定了空范围（例如没有可用仓库）时直接返回空结果
    if repo_ids is not None and not scope:
        return [], debug

    vector_hits = _vector_channel(query, scope, candidate_k)
    identifiers = extract_identifiers(query)
    keyword_hits = _keyword_channel(db, identifiers, scope, candidate_k)
    debug.vector_hits = len(vector_hits)
    debug.keyword_hits = len(keyword_hits)

    fused, similarities, keyword_scores = _fuse(
        vector_hits, keyword_hits, vector_weight, keyword_weight
    )
    ordered = [
        chunk_id
        for chunk_id, _ in sorted(fused.items(), key=lambda item: (-item[1], item[0]))
    ]

    # 软阈值：向量相似度过低且没有 name/signature 命中时丢弃
    threshold_kept = [
        chunk_id
        for chunk_id in ordered
        if similarities.get(chunk_id, 0.0) >= settings.retrieval_min_score
        or keyword_scores.get(chunk_id, 0.0) >= 2.0
    ]
    debug.filtered = len(ordered) - len(threshold_kept)

    deduped, removed = _dedupe_parent_child(threshold_kept, db)
    debug.filtered += removed

    selected = deduped[:top_k]
    if not selected:
        return [], debug

    row_map = _load_rows(db, selected)
    sources: list[SourceOut] = []
    used_chars = 0
    total_budget = settings.retrieval_max_total_chars

    for chunk_id in selected:
        row = row_map.get(chunk_id)
        if row is None:
            continue
        chunk, code_file, repo = row
        content, truncated = _prepare_content(db, chunk, identifiers)
        if truncated:
            debug.truncated += 1
        # 总预算：超限的按分数顺序丢弃（后面的通常分数更低）
        if sources and used_chars + len(content) > total_budget:
            debug.filtered += 1
            continue
        used_chars += len(content)
        sources.append(
            SourceOut(
                index=len(sources) + 1,
                chunk_id=chunk.id,
                module_id=_module_id(db, chunk),
                chunk_type=chunk.chunk_type,
                name=chunk.name,
                repo_id=repo.id,
                repo_name=repo.name,
                file_id=code_file.id,
                file_path=code_file.path,
                language=code_file.language,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                score=_display_score(
                    chunk_id, similarities, keyword_scores, vector_weight, keyword_weight
                ),
                content=content,
            )
        )

    debug.took_ms = int((time.perf_counter() - started) * 1000)
    return sources, debug
