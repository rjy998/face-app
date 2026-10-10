"""
知识库路由（预留接入点）：导入文档 / 测试检索 / 查看统计。

自建知识库的三种用法：
  1. 把 .md/.txt 放进 backend/knowledge_base/docs/，然后 `python scripts/ingest_local.py` 一键导入；
  2. 启动时若知识库为空会自动导入该目录（KB_AUTO_INGEST=true，默认开启）；
  3. 调用 POST /api/kb/ingest 上传文件或指定服务器目录。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile
from sqlalchemy.orm import Session

from config import settings
from models.db import User, get_db
from routers.deps import get_current_user
from schemas.models import KbIngestResponse, KbQueryRequest, KbQueryResponse, KbRef, KbStatsResponse
from services import rag

logger = logging.getLogger("face_app.kb")
router = APIRouter(prefix="/api/kb", tags=["知识库"])


def _check_permission(x_kb_token: str | None, user: User) -> None:
    """配置了 KB_INGEST_TOKEN 时必须带对口令；否则要求已登录。"""
    if settings.KB_INGEST_TOKEN:
        if x_kb_token != settings.KB_INGEST_TOKEN:
            raise HTTPException(status_code=403, detail="知识库导入口令不正确")
    elif user is None:  # pragma: no cover - 依赖已保证
        raise HTTPException(status_code=401, detail="请先登录")


@router.post("/ingest", response_model=KbIngestResponse, summary="导入知识文档（.md/.txt）")
async def ingest(
    files: list[UploadFile] = File(default=[], description="可一次上传多个 .md/.txt"),
    directory: str | None = Form(default=None, description="服务器上的目录路径（与 files 二选一）"),
    source: str | None = Form(default=None, description="来源说明，如书名或链接"),
    x_kb_token: str | None = Header(default=None, alias="X-KB-Token"),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> KbIngestResponse:
    _check_permission(x_kb_token, user)

    documents = chunks = skipped = 0
    titles: list[str] = []

    if directory:
        result = rag.ingest_directory(db, directory)
        documents += result["documents"]
        chunks += result["chunks"]
        skipped += result["skipped"]
        titles.extend(result["titles"])

    for upload in files or []:
        raw = await upload.read()
        if not raw:
            skipped += 1
            continue
        text = raw.decode("utf-8", errors="ignore")
        title = (upload.filename or "未命名文档").rsplit(".", 1)[0]
        outcome = rag.ingest_text(db, title=title, text=text, source=source or upload.filename)
        if outcome["skipped"]:
            skipped += 1
        else:
            documents += 1
            chunks += outcome["chunks"]
            titles.append(outcome["title"])

    if documents == 0 and skipped == 0:
        raise HTTPException(status_code=400, detail="没有可导入的内容，请上传 .md/.txt 或指定 directory")

    return KbIngestResponse(documents=documents, chunks=chunks, skipped=skipped, titles=titles)


@router.post("/query", response_model=KbQueryResponse, summary="测试检索（调试用）")
def query(
    payload: KbQueryRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> KbQueryResponse:
    hits = rag.search(db, payload.query, payload.top_k)
    return KbQueryResponse(query=payload.query, hits=[KbRef(**hit.to_dict()) for hit in hits])


@router.get("/stats", response_model=KbStatsResponse, summary="知识库统计")
def kb_stats(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> KbStatsResponse:
    return KbStatsResponse(**rag.stats(db))
