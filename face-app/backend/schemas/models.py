"""请求 / 响应数据结构（Pydantic）。

只用 v1、v2 都支持的写法（BaseModel + Field），避免版本差异导致启动失败。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 认证
# ---------------------------------------------------------------------------
class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=32, description="用户名 3-32 位")
    password: str = Field(..., min_length=6, max_length=64, description="密码至少 6 位")


class LoginRequest(BaseModel):
    username: str
    password: str


class UserOut(BaseModel):
    id: int
    username: str
    created_at: str | None = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserOut


# ---------------------------------------------------------------------------
# 面相分析
# ---------------------------------------------------------------------------
class KbRef(BaseModel):
    chunk_id: int
    doc_title: str
    score: float
    excerpt: str


class AnalyzeResponse(BaseModel):
    record_id: int
    created_at: str
    engine: str
    llm_provider: str
    image_url: str | None = None          # 带签名的私有照片地址
    features: dict[str, Any]
    reading: str
    kb_refs: list[KbRef] = []
    disclaimer: str


class HistoryItem(BaseModel):
    record_id: int
    created_at: str
    image_url: str | None = None
    face_shape: str | None = None
    summary: str | None = None
    reading_preview: str | None = None


class HistoryDetail(AnalyzeResponse):
    pass


# ---------------------------------------------------------------------------
# 知识库
# ---------------------------------------------------------------------------
class KbIngestResponse(BaseModel):
    documents: int
    chunks: int
    skipped: int
    titles: list[str] = []


class KbQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=200)
    top_k: int = Field(5, ge=1, le=20)


class KbQueryResponse(BaseModel):
    query: str
    hits: list[KbRef] = []


class KbStatsResponse(BaseModel):
    documents: int
    chunks: int
    embedding_provider: str
    embedding_dim: int
    vector_store: str
    docs_dir: str
