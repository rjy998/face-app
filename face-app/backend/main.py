"""
FastAPI 入口：注册路由、中间件、静态前端与启动初始化。

启动：
    cd backend
    uvicorn main:app --reload --port 8000
接口文档：
    http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from config import BASE_DIR, PROJECT_ROOT, settings
from models.db import get_db, init_db
from routers import auth, face, history, kb
from schemas.models import KbStatsResponse
from services import rag
from services.face_detector import available_engines
from services.security import crypto_backend

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("face_app")

APP_VERSION = "1.0.0"
FRONTEND_DIR = PROJECT_ROOT / "frontend"


def _configure_tempdir() -> None:
    """
    把临时目录固定到项目内的 storage/tmp。

    好处：容器/受限沙箱里不会因为系统临时目录不可写而失败，也便于清理。
    MediaPipe 会间接导入 matplotlib，其字体缓存也一并指到这里，避免写系统用户目录。
    """
    temp_dir = settings.ensure_temp_dir()
    os.environ["TMPDIR"] = str(temp_dir)
    os.environ["TEMP"] = str(temp_dir)
    os.environ["TMP"] = str(temp_dir)
    os.environ["MPLCONFIGDIR"] = str(temp_dir / "matplotlib")
    tempfile.tempdir = str(temp_dir)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动初始化：目录、建表、知识库自检与自动导入、引擎能力打印。"""
    settings.ensure_dirs()
    _configure_tempdir()
    init_db()
    logger.info("数据库就绪：%s", settings.DATABASE_URL)

    engines = available_engines()
    logger.info("人脸检测引擎可用性：%s", engines)
    if not engines["mediapipe"] and not engines["opencv"]:
        logger.error("没有可用的人脸检测引擎，请 pip install mediapipe 或 opencv-python")
    logger.info("加密后端：%s", crypto_backend())
    logger.info("解读后端：LLM_PROVIDER=%s，Ark 密钥=%s", settings.LLM_PROVIDER, bool(settings.ARK_API_KEY))

    # 知识库兜底：空库时自动导入 knowledge_base/docs/，保证首次运行就有参考资料
    from models.db import SessionLocal

    with SessionLocal() as db:
        if settings.KB_AUTO_INGEST and rag.is_empty(db):
            result = rag.ingest_directory(db)
            logger.info("知识库为空，已自动导入默认知识：%s", result)
        logger.info("知识库统计：%s", rag.stats(db))

    yield
    logger.info("服务已停止")


app = FastAPI(
    title=settings.APP_NAME,
    version=APP_VERSION,
    description=(
        "面相识别 / 解读网页应用后端：人脸关键点 → 三庭五眼等结构化特征 → RAG 知识库检索 → "
        "大模型生成传统文化趣味解读。定位为传统文化参考，不做算命与命运承诺。"
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# 简易限流（起步实现；上量后换成 Redis + 网关限流）
# ---------------------------------------------------------------------------
_hits: dict[str, deque[float]] = defaultdict(deque)


@app.middleware("http")
async def rate_limit_and_log(request: Request, call_next):
    started = time.perf_counter()
    if settings.RATE_LIMIT_PER_MINUTE > 0 and request.method in {"POST", "PUT", "PATCH"}:
        client = request.client.host if request.client else "unknown"
        bucket = _hits[client]
        now = time.time()
        while bucket and now - bucket[0] > 60:
            bucket.popleft()
        if len(bucket) >= settings.RATE_LIMIT_PER_MINUTE:
            return JSONResponse(status_code=429, content={"detail": "请求过于频繁，请稍后再试"})
        bucket.append(now)

    response = await call_next(request)
    cost = (time.perf_counter() - started) * 1000
    if request.url.path.startswith("/api"):
        logger.info("%s %s -> %s (%.0fms)", request.method, request.url.path, response.status_code, cost)
    return response


# ---------------------------------------------------------------------------
# 路由注册
# ---------------------------------------------------------------------------
app.include_router(auth.router)
app.include_router(face.router)
app.include_router(history.router)
app.include_router(kb.router)


@app.get("/api/health", tags=["系统"], summary="健康检查（部署验证用）")
def health(db: Session = Depends(get_db)) -> dict[str, object]:
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "version": APP_VERSION,
        "env": settings.APP_ENV,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "engines": available_engines(),
        "crypto": crypto_backend(),
        "llm": {
            "provider": settings.LLM_PROVIDER,
            "doubao_configured": bool(settings.ARK_API_KEY),
            "openai_configured": bool(settings.OPENAI_API_KEY),
            "anthropic_configured": bool(settings.ANTHROPIC_API_KEY),
            "anthropic_base_url": settings.ANTHROPIC_BASE_URL,
            "anthropic_model": settings.ANTHROPIC_MODEL,
        },
        "kb": KbStatsResponse(**rag.stats(db)).model_dump(),
        "compliance": {"disclaimer": settings.COMPLIANCE_DISCLAIMER},
    }


# 前端静态页面（放在最后挂载，避免抢占 /api 路由）；生产环境一般用 Nginx 托管
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
    logger.info("已挂载前端静态目录：%s", FRONTEND_DIR)
else:  # pragma: no cover
    logger.warning("未找到前端目录 %s，仅提供 API", FRONTEND_DIR)


if __name__ == "__main__":  # 便于 `python main.py` 直接启动
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=settings.DEBUG)
