"""
面相分析路由（核心）：POST /api/face/analyze

内部流程（与需求规格一致）：
    接收图片
      → 校验（大小 / 格式 / 分辨率 / 是否包含清晰人脸）
      → 人脸检测 + 关键点
      → 提取特征：三庭五眼 / 五官 / 脸型 / 气色（结构化 JSON）
      → 用特征构造检索词 → RAG 检索知识库片段
      → 组装 Prompt → 调用大模型生成解读（无密钥时用本地模板兜底）
      → 图片、特征、解读、引用片段写入数据库
      → 返回结果（含签名后的私有照片地址）
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from config import IMAGE_DIR, settings
from models.db import FaceRecord, User, get_db
from routers.deps import get_current_user
from schemas.models import AnalyzeResponse, KbRef
from services import interpreter, rag
from services.compliance import ensure_disclaimer
from services.face_detector import decode_image, detect_face
from services.feature_extractor import extract_features
from services.security import sha256_file, sign_resource, verify_resource_signature

logger = logging.getLogger("face_app.face")
router = APIRouter(prefix="/api/face", tags=["面相分析"])


def signed_image_url(record_id: int) -> str:
    """私有照片不裸奔：返回带签名与有效期的地址。"""
    return f"/api/face/image/{record_id}?token={sign_resource(record_id)}"


def _validate_upload(file: UploadFile, data: bytes) -> None:
    extension = os.path.splitext(file.filename or "")[1].lower()
    if extension not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"仅支持 {', '.join(sorted(settings.ALLOWED_EXTENSIONS))} 格式")
    if file.content_type and file.content_type not in settings.ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="文件类型不是支持的图片格式")
    if not data:
        raise HTTPException(status_code=400, detail="上传内容为空")
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail=f"图片不能超过 {settings.MAX_UPLOAD_MB}MB")
    # 简单魔数校验，防止改扩展名的伪装文件
    magic_ok = (
        data[:3] == b"\xff\xd8\xff"                      # JPEG
        or data[:8] == b"\x89PNG\r\n\x1a\n"              # PNG
        or (data[:4] == b"RIFF" and data[8:12] == b"WEBP")  # WebP
    )
    if not magic_ok:
        raise HTTPException(status_code=415, detail="文件内容不是有效的 JPG/PNG/WebP 图片")


@router.post("/analyze", response_model=AnalyzeResponse, summary="上传照片做面相特征分析 + 解读")
async def analyze(
    file: UploadFile = File(..., description="正面清晰人脸照片（jpg/png/webp，≤10MB）"),
    current: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AnalyzeResponse:
    # 1. 读取 + 校验
    data = await file.read(settings.max_upload_bytes + 1)
    _validate_upload(file, data)

    # 2. 解码
    try:
        image, warnings = decode_image(data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # 3. 人脸检测
    try:
        detection = detect_face(image)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    detection.warnings = list(detection.warnings) + warnings
    if not detection.ok:
        raise HTTPException(
            status_code=422,
            detail="没有检测到清晰的人脸，请上传正面、光线充足、无遮挡的单人照片",
        )

    # 4. 特征提取
    try:
        features: dict[str, Any] = extract_features(image, detection)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # 5. RAG 检索（按部位多路检索，命中率远高于单条长检索词）
    queries = rag.build_queries(features)
    refs = rag.search_multi(
        db, queries,
        per_query=settings.KB_PER_QUERY,
        max_total=max(8, settings.KB_TOP_K * 2),
    )
    # 耳朵形态/痣/法令纹等无法从关键点测量的部位，对应的资料片段不参与解读
    refs = rag.drop_unmeasurable(refs, enabled=settings.KB_DROP_UNMEASURABLE)
    features["kb_queries"] = queries
    features["kb_query"] = " | ".join(queries[:4])       # 给结果页展示
    features["kb_hits"] = [ref.to_dict() for ref in refs]

    # 6. 生成解读
    reading, provider = interpreter.generate_reading(features, refs)
    reading = ensure_disclaimer(reading)

    # 7. 落库（照片存私有目录，数据库只存相对路径；生产可换成对象存储私有桶）
    record = FaceRecord(
        user_id=current.id,
        features=features,
        reading=reading,
        kb_refs=[ref.to_dict() for ref in refs],
        engine=detection.engine,
        llm_provider=provider,
    )
    db.add(record)
    db.flush()  # 拿到 record.id

    extension = os.path.splitext(file.filename or ".jpg")[1].lower() or ".jpg"
    relative_path = os.path.join("images", f"{record.id}_{uuid.uuid4().hex[:8]}{extension}")
    absolute_path = os.path.join(str(IMAGE_DIR), os.path.basename(relative_path))
    with open(absolute_path, "wb") as handle:
        handle.write(data)

    record.image_url = relative_path.replace("\\", "/")
    record.image_sha256 = sha256_file(absolute_path)
    db.commit()
    db.refresh(record)

    logger.info(
        "分析完成 user=%s record=%s engine=%s llm=%s kb_hits=%d",
        current.username, record.id, detection.engine, provider, len(refs),
    )

    return AnalyzeResponse(
        record_id=record.id,
        created_at=record.created_at.isoformat(),
        engine=record.engine or detection.engine,
        llm_provider=provider,
        image_url=signed_image_url(record.id),
        features=features,
        reading=reading,
        kb_refs=[KbRef(**ref.to_dict()) for ref in refs],
        disclaimer=settings.COMPLIANCE_DISCLAIMER,
    )


@router.get("/image/{record_id}", summary="读取私有照片（签名 URL 或本人 token）")
def get_image(
    record_id: int,
    token: str | None = Query(default=None, description="analyze 返回的签名票据"),
    db: Session = Depends(get_db),
) -> FileResponse:
    record = db.get(FaceRecord, record_id)
    if record is None or not record.image_url:
        raise HTTPException(status_code=404, detail="照片不存在")
    if not token or not verify_resource_signature(record_id, token):
        raise HTTPException(status_code=403, detail="照片访问票据无效或已过期，请重新分析或登录后查看")
    path = os.path.join(str(IMAGE_DIR), os.path.basename(record.image_url))
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="照片文件已不存在")
    return FileResponse(path, headers={"Cache-Control": "private, max-age=300"})
