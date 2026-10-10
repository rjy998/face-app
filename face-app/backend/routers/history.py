"""历史记录路由：列表 / 详情 / 删除（用户对自己的数据拥有完全控制权）。"""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from config import IMAGE_DIR, settings
from models.db import FaceRecord, User, get_db
from routers.deps import get_current_user
from routers.face import signed_image_url
from schemas.models import HistoryDetail, HistoryItem, KbRef

logger = logging.getLogger("face_app.history")
router = APIRouter(prefix="/api/history", tags=["历史记录"])


@router.get("", response_model=list[HistoryItem], summary="我的历史记录")
def list_history(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[HistoryItem]:
    rows = (
        db.execute(
            select(FaceRecord)
            .where(FaceRecord.user_id == current.id)
            .order_by(FaceRecord.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        .scalars()
        .all()
    )
    items: list[HistoryItem] = []
    for record in rows:
        features = record.features or {}
        summary_lines = features.get("summary") or []
        items.append(
            HistoryItem(
                record_id=record.id,
                created_at=record.created_at.isoformat(),
                image_url=signed_image_url(record.id),
                face_shape=(features.get("face_shape") or {}).get("label"),
                summary=" / ".join(summary_lines[:2]) if summary_lines else None,
                reading_preview=(record.reading or "")[:80],
            )
        )
    return items


@router.get("/{record_id}", response_model=HistoryDetail, summary="单条记录详情")
def get_history(
    record_id: int,
    current: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> HistoryDetail:
    record = db.get(FaceRecord, record_id)
    if record is None or record.user_id != current.id:
        raise HTTPException(status_code=404, detail="记录不存在")
    return HistoryDetail(
        record_id=record.id,
        created_at=record.created_at.isoformat(),
        engine=record.engine or "unknown",
        llm_provider=record.llm_provider or "unknown",
        image_url=signed_image_url(record.id),
        features=record.features or {},
        reading=record.reading or "",
        kb_refs=[KbRef(**ref) for ref in (record.kb_refs or [])],
        disclaimer=settings.COMPLIANCE_DISCLAIMER,
    )


@router.delete("/{record_id}", summary="删除记录及其照片（合规要求：用户随时可删）")
def delete_history(
    record_id: int,
    current: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    record = db.get(FaceRecord, record_id)
    if record is None or record.user_id != current.id:
        raise HTTPException(status_code=404, detail="记录不存在")

    removed_file = False
    if record.image_url:
        path = os.path.join(str(IMAGE_DIR), os.path.basename(record.image_url))
        if os.path.isfile(path):
            try:
                os.remove(path)
                removed_file = True
            except OSError as exc:  # 文件删不掉不影响记录删除
                logger.warning("删除照片失败：%s（%s）", path, exc)

    db.delete(record)
    db.commit()
    logger.info("用户 %s 删除了记录 %s", current.username, record_id)
    return {"record_id": record_id, "deleted": True, "photo_deleted": removed_file}
