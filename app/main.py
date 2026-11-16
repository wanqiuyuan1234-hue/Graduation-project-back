"""FastAPI 应用入口：注册仓库入库、切片浏览、模块浏览与对话问答路由。"""

from contextlib import asynccontextmanager

import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import chat, chunks, modules, query, repos
from app.services.ingest import manager, recover_interrupted_jobs


def _warmup_retrieval() -> None:
    """后台预热 embedding 与 Chroma：首个问答请求不必再等模型加载。

    失败只打印提示，不阻断服务启动（例如本地缓存缺失时会退化为首个请求再试）。
    """
    try:
        from app.services.indexing.embedder import get_embedder
        from app.services.rag.retriever import get_collection

        get_embedder()
        get_collection()
        print("[启动] 检索组件预热完成")
    except Exception as exc:  # noqa: BLE001 - 预热失败不应影响服务可用性
        print(f"[启动] 检索组件预热失败（首个检索请求会重试）：{exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动入库 worker；把上次进程遗留的未完成任务标记失败。"""
    interrupted = recover_interrupted_jobs()
    if interrupted:
        print(f"[启动] 已将 {interrupted} 个未完成任务标记为失败（服务重启中断）")
    manager.start()
    threading.Thread(
        target=_warmup_retrieval, name="retrieval-warmup", daemon=True
    ).start()
    yield


app = FastAPI(title="Code RAG API", version="0.2.0", lifespan=lifespan)

# 允许前端（默认 localhost:3000）跨域访问只读接口与入库接口。
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(chunks.router)
app.include_router(modules.router)
app.include_router(repos.router)
app.include_router(chat.router)
app.include_router(query.router)


@app.get("/")
def root():
    return {"message": "Code RAG backend is running"}
