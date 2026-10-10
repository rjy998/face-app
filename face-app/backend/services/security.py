"""
安全工具：密码哈希、JWT 签发/校验、私有照片签名 URL。

依赖策略：优先使用 bcrypt / PyJWT（生产推荐）；
若环境未安装，自动退化为纯标准库实现（PBKDF2 + HMAC-SHA256 JWT），
保证「零额外依赖也能把接口跑通」，但在 .env.example 中仍默认要求安装。
"""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import hmac
import json
import os
import secrets
from typing import Any

from config import settings

# ---------------------------------------------------------------------------
# 可选依赖探测
# ---------------------------------------------------------------------------
try:  # pragma: no cover - 取决于运行环境
    import bcrypt as _bcrypt
except Exception:  # noqa: BLE001
    _bcrypt = None

try:  # pragma: no cover - 取决于运行环境
    import jwt as _pyjwt
except Exception:  # noqa: BLE001
    _pyjwt = None


def crypto_backend() -> dict[str, str]:
    """返回当前实际生效的加密后端，便于 /api/health 自检。"""
    return {
        "password_hash": "bcrypt" if _bcrypt else "pbkdf2-sha256(fallback)",
        "jwt": "pyjwt" if _pyjwt else "hmac-sha256(fallback)",
    }


# ---------------------------------------------------------------------------
# 密码哈希
# ---------------------------------------------------------------------------
_PBKDF2_ROUNDS = 240_000


def hash_password(password: str) -> str:
    """生成密码哈希。bcrypt 可用时用 bcrypt，否则用 PBKDF2-SHA256。"""
    if not password or len(password) < 6:
        raise ValueError("密码长度至少 6 位")
    raw = password.encode("utf-8")
    if _bcrypt is not None:
        return _bcrypt.hashpw(raw[:72], _bcrypt.gensalt()).decode("utf-8")
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", raw, salt, _PBKDF2_ROUNDS)
    return "pbkdf2${}${}${}".format(
        _PBKDF2_ROUNDS, base64.b64encode(salt).decode(), base64.b64encode(digest).decode()
    )


def verify_password(password: str, password_hash: str) -> bool:
    """校验密码，兼容两种存储格式。"""
    if not password or not password_hash:
        return False
    raw = password.encode("utf-8")
    if password_hash.startswith("pbkdf2$"):
        try:
            _, rounds, salt_b64, digest_b64 = password_hash.split("$")
            digest = hashlib.pbkdf2_hmac(
                "sha256", raw, base64.b64decode(salt_b64), int(rounds)
            )
            return hmac.compare_digest(digest, base64.b64decode(digest_b64))
        except Exception:  # noqa: BLE001
            return False
    if _bcrypt is None:
        return False
    try:
        return _bcrypt.checkpw(raw[:72], password_hash.encode("utf-8"))
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# JWT（HS256）
# ---------------------------------------------------------------------------
def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(text: str) -> bytes:
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + padding)


def create_access_token(user_id: int, username: str, expires_minutes: int | None = None) -> str:
    """签发访问 token（JWT，带过期时间）。"""
    expires_minutes = expires_minutes or settings.JWT_EXPIRE_MINUTES
    now = dt.datetime.now(dt.timezone.utc)
    payload: dict[str, Any] = {
        "sub": str(user_id),
        "username": username,
        "iat": int(now.timestamp()),
        "exp": int((now + dt.timedelta(minutes=expires_minutes)).timestamp()),
        "iss": "face-app",
    }
    if _pyjwt is not None:
        return _pyjwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)
    header = {"alg": "HS256", "typ": "JWT"}
    segments = [
        _b64url(json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")),
        _b64url(json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")),
    ]
    signing_input = ".".join(segments).encode("ascii")
    signature = hmac.new(settings.JWT_SECRET.encode("utf-8"), signing_input, hashlib.sha256).digest()
    segments.append(_b64url(signature))
    return ".".join(segments)


def decode_access_token(token: str) -> dict[str, Any] | None:
    """校验并解析 token；失败/过期返回 None。"""
    if not token:
        return None
    try:
        if _pyjwt is not None:
            return _pyjwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
        header_b64, payload_b64, signature_b64 = token.split(".")
        signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
        expected = hmac.new(settings.JWT_SECRET.encode("utf-8"), signing_input, hashlib.sha256).digest()
        if not hmac.compare_digest(expected, _b64url_decode(signature_b64)):
            return None
        payload = json.loads(_b64url_decode(payload_b64))
        if int(payload.get("exp", 0)) < int(dt.datetime.now(dt.timezone.utc).timestamp()):
            return None
        return payload
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# 私有照片签名 URL（等价于对象存储的签名 URL 语义）
# ---------------------------------------------------------------------------
def sign_resource(record_id: int, expires_at: int | None = None) -> str:
    """生成 `expires.signature` 形式的访问票据。"""
    expires_at = expires_at or int(
        (dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=settings.SIGNED_URL_EXPIRE_SECONDS)).timestamp()
    )
    message = f"{record_id}:{expires_at}".encode("utf-8")
    signature = hmac.new(settings.SIGNED_URL_SECRET.encode("utf-8"), message, hashlib.sha256).hexdigest()[:32]
    return f"{expires_at}.{signature}"


def verify_resource_signature(record_id: int, token: str) -> bool:
    """校验访问票据是否有效且未过期。"""
    try:
        expires_raw, _, signature = token.partition(".")
        expires_at = int(expires_raw)
    except (TypeError, ValueError):
        return False
    if expires_at < int(dt.datetime.now(dt.timezone.utc).timestamp()):
        return False
    return hmac.compare_digest(sign_resource(record_id, expires_at).split(".", 1)[1], signature)


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1 << 20) -> str:
    """计算文件 SHA256（照片指纹，用于去重与完整性校验）。"""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
