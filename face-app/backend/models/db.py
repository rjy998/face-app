"""
数据库层（SQLAlchemy 2.0）。

表结构与《需求规格》第四部分一致：
    users            用户
    face_records     面相分析记录
    kb_documents     知识库文档
    kb_chunks        知识库切片（含向量）

切换数据库只需改 .env 里的 DATABASE_URL（SQLite -> MySQL/PostgreSQL），本文件无需改动。
"""

from __future__ import annotations

import datetime as dt
from typing import Generator

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from config import settings


class Base(DeclarativeBase):
    """所有 ORM 模型的基类。"""


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class User(Base):
    """用户表：密码只存 bcrypt 哈希，绝不存明文。"""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)

    records: Mapped[list["FaceRecord"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class FaceRecord(Base):
    """面相分析记录：一次上传 = 一条记录。"""

    __tablename__ = "face_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    image_url: Mapped[str | None] = mapped_column(String(512), nullable=True)      # 本地路径 / 对象存储 key
    image_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)    # 便于校验与去重
    features: Mapped[dict | None] = mapped_column(JSON, nullable=True)             # 结构化特征 JSON
    reading: Mapped[str | None] = mapped_column(Text, nullable=True)               # 大模型/模板生成的解读
    kb_refs: Mapped[list | None] = mapped_column(JSON, nullable=True)              # 引用的知识片段（id/标题/摘要）
    engine: Mapped[str | None] = mapped_column(String(64), nullable=True)          # 检测引擎，便于排查
    llm_provider: Mapped[str | None] = mapped_column(String(32), nullable=True)    # 解读来源：doubao/openai/template
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow, nullable=False, index=True)

    user: Mapped[User] = relationship(back_populates="records")


class KbDocument(Base):
    """知识库文档（一篇 .md/.txt = 一条记录）。"""

    __tablename__ = "kb_documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    source: Mapped[str | None] = mapped_column(String(512), nullable=True)
    sha256: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)  # 内容指纹，防重复导入
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)

    chunks: Mapped[list["KbChunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class KbChunk(Base):
    """知识库切片：chunk_text + embedding（JSON 序列化后的向量）。"""

    __tablename__ = "kb_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_id: Mapped[int] = mapped_column(ForeignKey("kb_documents.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)  # UTF-8 JSON 数组
    meta: Mapped[dict | None] = mapped_column(JSON, nullable=True)               # 标题路径/标签
    norm: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)      # 向量模长缓存，加速余弦相似度

    document: Mapped[KbDocument] = relationship(back_populates="chunks")


# ---------------------------------------------------------------------------
# 引擎与会话
# ---------------------------------------------------------------------------
_connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(settings.DATABASE_URL, echo=False, future=True, pool_pre_ping=True, connect_args=_connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, future=True)


def init_db() -> None:
    """建表（幂等）。生产环境建议改为 Alembic 迁移。"""
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI 依赖：每个请求一个 Session。"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
