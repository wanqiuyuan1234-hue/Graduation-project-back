"""仓库入库接口：提交任务、查询进度、取消、删除。"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import delete, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Chunk, CodeFile, Repo, RepoJob
from app.schemas.job import JobOut, RepoCreateIn, RepoCreateOut
from app.services.ingest import (
    delete_repo_vectors,
    derive_repo_name,
    manager,
    remove_clone_dir,
    resolve_clone_dir,
    update_job,
)

router = APIRouter(prefix="/api", tags=["repos"])

ACTIVE_STATUSES = ("queued", "running")


def _validate_url(raw: str) -> str:
    """只接受 http/https 的 git 地址；去掉首尾空白。"""
    url = raw.strip()
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise HTTPException(
            status_code=422,
            detail="仓库地址必须是 http(s):// 开头的完整 git URL",
        )
    return url


@router.post("/repos", response_model=RepoCreateOut, status_code=202)
def create_repo(payload: RepoCreateIn, db: Session = Depends(get_db)):
    """提交仓库入库任务（进程内单并发，等待队列最多 2 个）。

    错误码：
    - 422 URL 不合法
    - 409 同一 URL 已有进行中的任务
    - 429 队列已满
    """
    url = _validate_url(payload.url)

    active = db.execute(
        select(RepoJob).where(
            RepoJob.repo_url == url,
            RepoJob.status.in_(ACTIVE_STATUSES),
        )
    ).scalar_one_or_none()
    if active is not None:
        raise HTTPException(
            status_code=409,
            detail=f"该仓库已有进行中的任务（job {active.id}，状态 {active.status}）",
        )

    # 关键：先把仓库行查/建出来并把 repo_id 写进任务，任务一提交就能被
    # GET /api/repos 的 latest job 查询命中，前端卡片才会立刻进入 syncing 并展示进度。
    name = derive_repo_name(url)
    clone_dir = resolve_clone_dir(db, name, url)
    repo = db.execute(select(Repo).where(Repo.url == url)).scalar_one_or_none()
    created_repo = repo is None
    if repo is None:
        repo = Repo(name=name, url=url, clone_dir=str(clone_dir))
        db.add(repo)
        message = "任务已排队"
    else:
        repo.name = name
        repo.clone_dir = str(clone_dir)
        message = "任务已排队，将全量重建该仓库"

    try:
        # flush 拿到自增 id，再写进任务行
        db.flush()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(
            status_code=409, detail="该仓库地址已存在，请刷新后重试"
        ) from exc

    job = RepoJob(
        repo_url=url,
        repo_id=repo.id,
        status="queued",
        phase="queued",
        progress=0,
        message="排队中",
    )
    db.add(job)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(
            status_code=409, detail="该仓库地址已存在，请刷新后重试"
        ) from exc
    db.refresh(job)

    if not manager.submit(job.id):
        db.delete(job)
        if created_repo:
            # 队列满时回滚刚建的仓库行，避免留下空卡片
            db.delete(repo)
        db.commit()
        raise HTTPException(
            status_code=429,
            detail="入库队列已满（正在处理 1 个，排队 2 个），请稍后再试",
        )

    return RepoCreateOut(
        job_id=job.id,
        repo_id=repo.id,
        status=job.status,
        phase=job.phase,
        message=message,
    )


@router.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: int, db: Session = Depends(get_db)):
    """查询任务阶段与进度（前端轮询用）。"""
    job = db.get(RepoJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@router.post("/jobs/{job_id}/cancel", response_model=JobOut)
def cancel_job(job_id: int, db: Session = Depends(get_db)):
    """请求取消任务：排队中的立即生效，运行中的在下一个检查点生效。"""
    job = db.get(RepoJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.status not in ACTIVE_STATUSES:
        raise HTTPException(
            status_code=409, detail=f"任务已结束（{job.status}），无法取消"
        )

    job.cancel_requested = True
    db.commit()

    # 还在排队：直接从等待队列摘掉并立即置为已取消
    if job.status == "queued" and manager.cancel_pending(job_id):
        update_job(
            job_id,
            status="canceled",
            phase="canceled",
            message="任务已取消",
            finished_at=datetime.now(timezone.utc),
        )

    db.refresh(job)
    return job


@router.delete("/repos/{repo_id}")
def delete_repo(repo_id: int, db: Session = Depends(get_db)):
    """彻底删除仓库：PG 行、向量、任务记录与磁盘克隆目录。

    覆盖“从未成功入库”的仓库（可能只有 repo 行 + 失败任务，没有文件/切片）。
    有进行中的任务时拒绝删除，避免删除与入库并发写同一仓库。
    """
    repo = db.get(Repo, repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="仓库不存在")

    active = db.execute(
        select(RepoJob).where(
            RepoJob.status.in_(ACTIVE_STATUSES),
            or_(RepoJob.repo_id == repo_id, RepoJob.repo_url == repo.url),
        )
    ).scalars().first()
    if active is not None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"该仓库有进行中的任务（job {active.id}，状态 {active.status}），"
                "请先取消任务再删除"
            ),
        )

    # 1) 向量：按 metadata.repo_id 精确删除
    vectors_removed = delete_repo_vectors(repo_id)

    # 2) PostgreSQL：chunk → code_file → repo_job → repo
    # 先解除 parent_chunk_id 自引用，避免自外键阻塞批量删除
    file_ids = list(
        db.execute(select(CodeFile.id).where(CodeFile.repo_id == repo_id)).scalars()
    )
    if file_ids:
        db.execute(
            update(Chunk)
            .where(Chunk.file_id.in_(file_ids))
            .values(parent_chunk_id=None)
        )
        db.execute(delete(Chunk).where(Chunk.file_id.in_(file_ids)))
    db.execute(delete(CodeFile).where(CodeFile.repo_id == repo_id))
    # 同时按 url 清理 repo_id 为空的历史任务记录
    db.execute(
        delete(RepoJob).where(
            or_(RepoJob.repo_id == repo_id, RepoJob.repo_url == repo.url)
        )
    )
    clone_dir = repo.clone_dir
    db.delete(repo)
    db.commit()

    # 3) 磁盘克隆目录（仅允许 data/repos 下的路径）
    dir_removed = False
    if clone_dir:
        dir_removed = remove_clone_dir(Path(clone_dir))

    return {
        "deleted": repo_id,
        "vectors_removed": vectors_removed,
        "dir_removed": dir_removed,
    }
