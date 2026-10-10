"""
全局配置模块。

设计原则：
1. 所有可变参数（密钥、数据库地址、模型名）都通过环境变量注入，代码里不写死密钥；
2. 本地开发用 backend/.env（已加入 .gitignore），生产用容器环境变量；
3. 每一项都有安全默认值，保证「不填任何密钥也能把整个流程跑通」（AI 解读退化为模板生成）。
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent            # backend/
PROJECT_ROOT = BASE_DIR.parent                        # face-app/
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", str(BASE_DIR / "storage")))
IMAGE_DIR = STORAGE_DIR / "images"                    # 本地私有照片目录（生产建议换对象存储私有桶）
KB_DOCS_DIR = Path(os.getenv("KB_DOCS_DIR", str(BASE_DIR / "knowledge_base" / "docs")))

# ---------------------------------------------------------------------------
# .env 载入（python-dotenv 缺失时退化为极简解析，保证零依赖也能跑）
# ---------------------------------------------------------------------------
def _load_dotenv() -> None:
    env_file = BASE_DIR / ".env"
    if not env_file.exists():
        return
    try:  # 优先用 python-dotenv
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(env_file, override=False)
        return
    except Exception:  # pragma: no cover - 兜底分支
        pass
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if not raw:
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


class Settings:
    """集中式配置对象（简单可靠，不引入 pydantic-settings 依赖）。"""

    # --- 应用 ---
    APP_NAME: str = os.getenv("APP_NAME", "面相识别解读应用")
    APP_ENV: str = os.getenv("APP_ENV", "dev")
    DEBUG: bool = _env_bool("DEBUG", True)
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # --- 数据库 ---
    # 开发：SQLite；生产：mysql+pymysql://user:pass@host:3306/face_app?charset=utf8mb4
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{(BASE_DIR / 'face_app.db').as_posix()}")

    # --- 认证（JWT / bcrypt） ---
    JWT_SECRET: str = os.getenv("JWT_SECRET", "dev-only-change-me-in-production")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    JWT_EXPIRE_MINUTES: int = _env_int("JWT_EXPIRE_MINUTES", 60 * 24 * 7)  # 默认 7 天

    # --- 上传校验 ---
    MAX_UPLOAD_MB: int = _env_int("MAX_UPLOAD_MB", 10)
    ALLOWED_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".webp"}
    ALLOWED_CONTENT_TYPES: set[str] = {"image/jpeg", "image/png", "image/webp"}
    MIN_IMAGE_SIDE: int = _env_int("MIN_IMAGE_SIDE", 120)  # 太小的图直接拒绝（无法算比例）

    # --- 照片访问：签名 URL（私有资源不裸奔） ---
    SIGNED_URL_SECRET: str = os.getenv("SIGNED_URL_SECRET", "") or os.getenv("JWT_SECRET", "dev-only-change-me")
    SIGNED_URL_EXPIRE_SECONDS: int = _env_int("SIGNED_URL_EXPIRE_SECONDS", 900)  # 15 分钟

    # --- 人脸检测 ---
    # mediapipe：本地免费、隐私好（默认）；cloud：百度/腾讯/阿里云 API
    FACE_DETECTOR: str = os.getenv("FACE_DETECTOR", "mediapipe")
    CLOUD_FACE_PROVIDER: str = os.getenv("CLOUD_FACE_PROVIDER", "baidu")
    CLOUD_FACE_API_KEY: str = os.getenv("CLOUD_FACE_API_KEY", "")
    CLOUD_FACE_SECRET_KEY: str = os.getenv("CLOUD_FACE_SECRET_KEY", "")
    # MediaPipe >= 1.0 使用 Tasks API，需要 .task 模型文件；缺省时自动下载一次
    FACE_LANDMARKER_MODEL: str = os.getenv("FACE_LANDMARKER_MODEL", str(STORAGE_DIR / "models" / "face_landmarker.task"))
    FACE_LANDMARKER_URL: str = os.getenv(
        "FACE_LANDMARKER_URL",
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    )
    FACE_LANDMARKER_AUTO_DOWNLOAD: bool = _env_bool("FACE_LANDMARKER_AUTO_DOWNLOAD", True)

    # --- 大模型解读 ---
    # doubao（火山方舟）| openai（任意 OpenAI 兼容接口）| anthropic（Claude 协议，MiniMax 等）| template
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "doubao")
    ARK_API_KEY: str = os.getenv("ARK_API_KEY", "")
    ARK_BASE_URL: str = os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
    ARK_MODEL: str = os.getenv("ARK_MODEL", "doubao-seed-1-6-250615")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    # Anthropic 协议（Claude Messages API）：MiniMax 官方兼容地址为 https://api.minimax.cn/anthropic
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_BASE_URL: str = os.getenv("ANTHROPIC_BASE_URL", "https://api.minimax.cn/anthropic")
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "MiniMax-M3.1-Flash-Preview")
    LLM_TIMEOUT_SECONDS: int = _env_int("LLM_TIMEOUT_SECONDS", 60)
    LLM_MAX_TOKENS: int = _env_int("LLM_MAX_TOKENS", 800)

    # --- 知识库 RAG ---
    # local：内置字符 n-gram 哈希向量（零依赖，离线可用）；可换 bge / 豆包 / 阿里 Embedding API
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "local")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    EMBEDDING_DIM: int = _env_int("EMBEDDING_DIM", 512)
    KB_TOP_K: int = _env_int("KB_TOP_K", 5)
    KB_PER_QUERY: int = _env_int("KB_PER_QUERY", 3)   # 多路检索时每条检索词取前 N 条
    # 内置兜底知识的标题：检索时它排在自有资料之后，只用于补空位
    KB_FALLBACK_TITLE: str = os.getenv("KB_FALLBACK_TITLE", "默认相学常识")
    # 是否允许兜底知识参与检索（想让解读 100% 基于自建资料就设为 false）
    KB_INCLUDE_FALLBACK: bool = _env_bool("KB_INCLUDE_FALLBACK", True)
    # 过滤掉照片测不到的部位（耳朵/痣/法令纹等）的相关资料，避免对没测过的部位下判断
    KB_DROP_UNMEASURABLE: bool = _env_bool("KB_DROP_UNMEASURABLE", True)
    KB_CHUNK_SIZE: int = _env_int("KB_CHUNK_SIZE", 350)
    KB_CHUNK_OVERLAP: int = _env_int("KB_CHUNK_OVERLAP", 50)
    KB_AUTO_INGEST: bool = _env_bool("KB_AUTO_INGEST", True)  # 启动时若库为空则自动导入 docs/
    # 导入知识库的口令：生产环境请设置，调用 /api/kb/ingest 时带 X-KB-Token 头
    KB_INGEST_TOKEN: str = os.getenv("KB_INGEST_TOKEN", "")
    # 向量库：sqlite（默认，向量存 kb_chunks.embedding）| chroma（本地 ChromaDB）
    VECTOR_STORE: str = os.getenv("VECTOR_STORE", "sqlite")

    # --- 缓存 / 限流（预留，起步用进程内实现；上量后换 Redis） ---
    REDIS_URL: str = os.getenv("REDIS_URL", "")
    RATE_LIMIT_PER_MINUTE: int = _env_int("RATE_LIMIT_PER_MINUTE", 30)  # 0 = 关闭

    # --- CORS ---
    CORS_ORIGINS: list[str] = _env_list(
        "CORS_ORIGINS",
        ["*"] if os.getenv("APP_ENV", "dev") == "dev" else [],
    )

    # --- 合规文案（解读结果过滤，禁止绝对化 / 恐吓性表述） ---
    BANNED_PHRASES: tuple[str, ...] = (
        "改运", "转运", "必发财", "一定发财", "必大发", "必定富贵", "保证发财",
        "克夫", "克妻", "克父母", "克子", "大凶", "凶兆", "死", "短命", "绝后",
        "命中注定", "注定", "必然成功", "包治", "消灾", "辟邪", "法事",
        # 古籍整理时常见、但同属绝对化或恐吓性的表述
        "主贫", "主病", "主贱", "贫贱", "贱格", "寿元不长", "寿元", "寿数", "短寿", "夭折", "折寿",
        "早逝", "早亡", "先亡", "先损", "损父", "损母", "寿终正寝", "父母双亡", "离世", "易逝",
        "孤寡", "刑克", "无子", "破财", "牢狱", "血光", "横死", "苦命", "残疾", "淫贱",
        # 生育与胎儿性别预测（传统相学常见，但会给家庭带来压力，不做保留）
        "生男孩", "生女孩", "生男", "生女", "头胎", "怀男孩", "怀女孩",
    )
    COMPLIANCE_DISCLAIMER: str = (
        "以上内容基于传统面相文化常识的趣味参考，不代表科学结论，也不构成任何命运、健康或投资建议。"
    )
    # 是否在解读末尾自动追加免责声明（合规建议保持 true；关闭后解读只剩【结论】+【注意事项】）
    COMPLIANCE_DISCLAIMER_ENABLED: bool = _env_bool("COMPLIANCE_DISCLAIMER_ENABLED", True)

    def ensure_dirs(self) -> None:
        """启动时确保运行期目录存在。"""
        IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        KB_DOCS_DIR.mkdir(parents=True, exist_ok=True)
        Path(self.FACE_LANDMARKER_MODEL).parent.mkdir(parents=True, exist_ok=True)

    def ensure_temp_dir(self) -> Path:
        """项目内临时目录（容器/沙箱里比系统临时目录更可靠）。"""
        temp_dir = STORAGE_DIR / "tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_MB * 1024 * 1024


settings = Settings()
