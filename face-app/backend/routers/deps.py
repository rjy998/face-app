"""路由公共依赖：从 `Authorization: Bearer <token>` 解析当前用户。"""

from __future__ import annotations

import logging

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from models.db import User, get_db
from services.security import decode_access_token

logger = logging.getLogger("face_app.auth")


def _extract_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    parts = authorization.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return authorization.strip()


def get_current_user(
    authorization: str | None = Header(default=None, description="Bearer <token>"),
    db: Session = Depends(get_db),
) -> User:
    """必须登录的接口依赖。"""
    token = _extract_token(authorization)
    payload = decode_access_token(token) if token else None
    if not payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="未登录或登录已过期，请重新登录")
    user = db.get(User, int(payload.get("sub", 0)))
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户不存在，请重新登录")
    return user


def get_optional_user(
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> User | None:
    """可选登录：有合法 token 就返回用户，否则返回 None。"""
    token = _extract_token(authorization)
    if not token:
        return None
    payload = decode_access_token(token)
    if not payload:
        return None
    return db.execute(select(User).where(User.id == int(payload.get("sub", 0)))).scalar_one_or_none()
