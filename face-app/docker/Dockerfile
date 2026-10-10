# ============================================================
# 后端镜像：Python + FastAPI + OpenCV + MediaPipe
# 构建（在 face-app/ 目录执行）：
#   docker build -f docker/Dockerfile -t face-app-backend:latest .
# ============================================================
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Shanghai

WORKDIR /app

# 基础系统依赖：curl 用于健康检查；libgl/glib 供 OpenCV 使用
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libglib2.0-0 \
        libgl1 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# 先装依赖，利用 Docker 层缓存
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# 再拷贝源码与默认知识库
COPY backend/ /app/
COPY frontend/ /frontend/

# 运行期目录（照片、临时文件、SQLite）
RUN mkdir -p /app/storage/images /app/storage/tmp \
    && useradd -m -u 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers"]
