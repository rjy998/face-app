"""认证路由：注册 / 登录 / 游客一键体验 / 当前用户。"""

from __future__ import annotations

import logging
import re
import secrets

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from config import settings
from models.db import User, get_db
from routers.deps import get_current_user
from schemas.models import LoginRequest, RegisterRequest, TokenResponse, UserOut
from services.security import create_access_token, hash_password, verify_password

logger = logging.getLogger("face_app.auth")
router = APIRouter(prefix="/api/auth", tags=["认证"])

_USERNAME_RE = re.compile(r"^[A-Za-z0-9_\u4e00-\u9fa5]{3,32}$")


def _token_response(user: User) -> TokenResponse:
    return TokenResponse(
        access_token=create_access_token(user.id, user.username),
        expires_in=settings.JWT_EXPIRE_MINUTES * 60,
        user=UserOut(id=user.id, username=user.username, created_at=user.created_at.isoformat()),
    )


@router.post("/register", response_model=TokenResponse, summary="注册")
def register(payload: RegisterRequest, db: Session = Depends(get_db)) -> TokenResponse:
    username = payload.username.strip()
    if not _USERNAME_RE.match(username):
        raise HTTPException(status_code=400, detail="用户名需为 3-32 位中文、字母、数字或下划线")
    exists = db.execute(select(User).where(User.username == username)).scalar_one_or_none()
    if exists is not None:
        raise HTTPException(status_code=409, detail="该用户名已被注册")
    user = User(username=username, password_hash=hash_password(payload.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    logger.info("新用户注册：%s", username)
    return _token_response(user)


@router.post("/login", response_model=TokenResponse, summary="登录")
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    user = db.execute(select(User).where(User.username == payload.username.strip())).scalar_one_or_none()
    if user is None or not verify_password(payload.password, user.password_hash):
        # 不区分「用户不存在」与「密码错误」，避免账号枚举
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户名或密码不正确")
    return _token_response(user)


@router.post("/guest", response_model=TokenResponse, summary="游客一键体验（自动创建临时账号）")
def guest(db: Session = Depends(get_db)) -> TokenResponse:
    """前端「立即体验」使用：自动开一个独立账号，避免游客之间互相看到记录。"""
    for _ in range(5):
        username = f"guest_{secrets.token_hex(4)}"
        if db.execute(select(User).where(User.username == username)).scalar_one_or_none() is None:
            user = User(username=username, password_hash=hash_password(secrets.token_urlsafe(12)))
            db.add(user)
            db.commit()
            db.refresh(user)
            return _token_response(user)
    raise HTTPException(status_code=500, detail="临时账号创建失败，请稍后重试")


@router.get("/me", response_model=UserOut, summary="当前登录用户")
def me(user: User = Depends(get_current_user)) -> UserOut:
    return UserOut(id=user.id, username=user.username, created_at=user.created_at.isoformat())
