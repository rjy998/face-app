"""
知识库（RAG 检索增强）——本项目最关键的「可插拔」部分。

流程：
    知识文档(.md/.txt)
      → 切片（按标题分层，再按长度切分并留重叠）
      → Embedding 转向量
      → 存库（默认存 kb_chunks.embedding；可换 ChromaDB / pgvector）
    检索：
      人脸特征 → 检索词 → 向量检索 top-k → 作为「参考资料」拼进大模型 Prompt

三种 Embedding 后端（.env 切换，都不影响其它代码）：
    local  ：内置字符 n-gram 哈希向量，零依赖、离线可用（默认，起步够用）
    openai ：任意 OpenAI 兼容的 Embedding 接口（豆包/阿里/自建均可）
    bge    ：本地 sentence-transformers + BAAI/bge-small-zh-v1.5（效果最好，需额外安装）
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Iterable

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from config import KB_DOCS_DIR, settings
from models.db import KbChunk, KbDocument

logger = logging.getLogger("face_app.rag")

# ---------------------------------------------------------------------------
# 切片
# ---------------------------------------------------------------------------
_HEADING_RE = re.compile(r"^(#{1,6})\s*(.+?)\s*$")
_SENTENCE_END = "。！？；\n"


@dataclass
class Chunk:
    text: str
    heading_path: str


def _split_long(text: str, size: int, overlap: int) -> list[str]:
    """把一段长文本按句子边界切成 ~size 字、带 overlap 字重叠的小片。"""
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []

    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # 尽量在句末切开，避免把一句话劈成两半
            window = text[start:end]
            cut = max((window.rfind(ch) for ch in _SENTENCE_END), default=-1)
            if cut > size * 0.5:
                end = start + cut + 1
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [piece for piece in pieces if piece]


def chunk_document(text: str, size: int | None = None, overlap: int | None = None) -> list[Chunk]:
    """按 Markdown 标题分层切片：同一标题下的内容作为整体，过长再按句子切分。"""
    size = size or settings.KB_CHUNK_SIZE
    overlap = overlap if overlap is not None else settings.KB_CHUNK_OVERLAP

    chunks: list[Chunk] = []
    heading_stack: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if not buffer:
            return
        body = "\n".join(buffer).strip()
        buffer.clear()
        if not body:
            return
        path = " > ".join(heading_stack)
        for piece in _split_long(body, size, overlap):
            prefix = f"【{path}】" if path else ""
            chunks.append(Chunk(text=f"{prefix}{piece}", heading_path=path))

    for raw_line in text.splitlines():
        match = _HEADING_RE.match(raw_line)
        if match:
            flush()
            level = len(match.group(1))
            title = match.group(2).strip()
            heading_stack = heading_stack[: level - 1]
            heading_stack.append(title)
        else:
            buffer.append(raw_line)
    flush()

    if not chunks and text.strip():  # 无标题的纯文本
        for piece in _split_long(text, size, overlap):
            chunks.append(Chunk(text=piece, heading_path=""))
    return chunks


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


class LocalHashEmbedding:
    """
    零依赖的中文友好向量：字符 1~3 gram 哈希 + 词频，再 L2 归一化。

    优点：离线、快、确定性；缺点：语义泛化弱于神经向量。
    对「按关键词命中相学条目」这类任务，效果足够，后续可无缝换 bge / 云 Embedding。
    """

    def __init__(self, dim: int | None = None) -> None:
        self.dim = dim or settings.EMBEDDING_DIM

    def _tokens(self, text: str) -> Iterable[str]:
        cleaned = re.sub(r"\s+", "", text)
        for n in (1, 2, 3):
            for i in range(len(cleaned) - n + 1):
                yield cleaned[i : i + n]

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dim
            for token in self._tokens(text):
                index = int(hashlib.md5(token.encode("utf-8")).hexdigest()[:8], 16) % self.dim
                vector[index] += 1.0
            vectors.append(_l2_normalize(vector))
        return vectors


class OpenAICompatEmbedding:
    """调用任意 OpenAI 兼容的 Embedding 接口（豆包方舟 / 阿里百炼 / 自建 vLLM 等）。"""

    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        import httpx  # 延迟导入

        response = httpx.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "input": texts},
            timeout=settings.LLM_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()["data"]
        return [_l2_normalize(item["embedding"]) for item in data]


class BgeEmbedding:
    """本地 sentence-transformers（BAAI/bge-small-zh-v1.5）。未安装依赖时抛错并触发降级。"""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore

        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return [list(map(float, vector)) for vector in vectors]


_embedder_cache: Any = None


def get_embedder() -> Any:
    """按配置返回 Embedding 实现；任何异常都降级到 LocalHashEmbedding，保证接口不挂。"""
    global _embedder_cache
    if _embedder_cache is not None:
        return _embedder_cache

    provider = (settings.EMBEDDING_PROVIDER or "local").lower()
    try:
        if provider == "openai" and settings.OPENAI_API_KEY:
            _embedder_cache = OpenAICompatEmbedding(
                settings.OPENAI_API_KEY, settings.OPENAI_BASE_URL, settings.EMBEDDING_MODEL
            )
        elif provider == "bge":
            _embedder_cache = BgeEmbedding(settings.EMBEDDING_MODEL)
        else:
            _embedder_cache = LocalHashEmbedding()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Embedding 后端 %s 不可用（%s），已降级为 local 哈希向量", provider, exc)
        _embedder_cache = LocalHashEmbedding()
    return _embedder_cache


def embed_texts(texts: list[str]) -> list[list[float]]:
    return get_embedder().embed(texts)


# ---------------------------------------------------------------------------
# 入库
# ---------------------------------------------------------------------------
def _serialize(vector: list[float]) -> bytes:
    return json.dumps(vector, separators=(",", ":")).encode("utf-8")


def _deserialize(blob: bytes | None) -> list[float]:
    if not blob:
        return []
    try:
        return json.loads(blob.decode("utf-8"))
    except Exception:  # noqa: BLE001
        return []


def ingest_text(db: Session, title: str, text: str, source: str | None = None) -> dict[str, Any]:
    """
    把一篇文档切片、向量化并入库。

    去重与更新策略：
      - 同标题已存在 → 先删旧文档（连同切片）再写新的，避免改完 Prompt 重跑后出现两条同名文档；
      - 其它文档内容 sha256 相同 → 跳过，避免重复导入。
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()

    same_title = db.execute(select(KbDocument).where(KbDocument.title == title)).scalar_one_or_none()
    replaced = False
    if same_title is not None:
        if same_title.sha256 == digest:
            return {"title": title, "chunks": 0, "skipped": True, "reason": "内容未变化"}
        db.delete(same_title)          # 级联删除旧切片
        db.flush()
        replaced = True
        logger.info("知识库更新：%s（旧版本已替换）", title)

    duplicate = db.execute(select(KbDocument).where(KbDocument.sha256 == digest)).scalar_one_or_none()
    if duplicate is not None:
        return {"title": title, "chunks": 0, "skipped": True, "reason": "内容与其它文档重复"}

    chunks = chunk_document(text)
    if not chunks:
        return {"title": title, "chunks": 0, "skipped": True, "reason": "文档为空"}

    document = KbDocument(title=title, source=source, sha256=digest)
    db.add(document)
    db.flush()  # 拿到 document.id

    vectors = embed_texts([chunk.text for chunk in chunks])
    for chunk, vector in zip(chunks, vectors):
        db.add(
            KbChunk(
                doc_id=document.id,
                chunk_text=chunk.text,
                embedding=_serialize(vector),
                meta={"heading_path": chunk.heading_path},
                norm=math.sqrt(sum(value * value for value in vector)) or 1.0,
            )
        )
    db.commit()
    logger.info("知识入库：%s，共 %d 片%s", title, len(chunks), "（替换旧版本）" if replaced else "")
    return {"title": title, "chunks": len(chunks), "skipped": False, "replaced": replaced}


def ingest_file(db: Session, path: str | os.PathLike[str], source: str | None = None) -> dict[str, Any]:
    """导入单个 .md/.txt 文件。"""
    file_path = str(path)
    with open(file_path, "r", encoding="utf-8", errors="ignore") as handle:
        text = handle.read()
    title = os.path.splitext(os.path.basename(file_path))[0]
    return ingest_text(db, title=title, text=text, source=source or file_path)


def ingest_directory(db: Session, directory: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """批量导入目录下的 .md/.txt（递归）。启动时若库为空也会自动调用一次。"""
    directory = str(directory or KB_DOCS_DIR)
    result = {"documents": 0, "chunks": 0, "skipped": 0, "titles": []}
    if not os.path.isdir(directory):
        return result
    for root, _dirs, files in os.walk(directory):
        for name in sorted(files):
            if not name.lower().endswith((".md", ".txt")):
                continue
            outcome = ingest_file(db, os.path.join(root, name))
            if outcome["skipped"]:
                result["skipped"] += 1
            else:
                result["documents"] += 1
                result["chunks"] += outcome["chunks"]
                result["titles"].append(outcome["title"])
    return result


def is_empty(db: Session) -> bool:
    return (db.execute(select(func.count(KbChunk.id))).scalar() or 0) == 0


# ---------------------------------------------------------------------------
# 检索
# ---------------------------------------------------------------------------
@dataclass
class RetrievedChunk:
    chunk_id: int
    doc_title: str
    text: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_title": self.doc_title,
            "score": round(self.score, 4),
            "excerpt": self.text[:220],
        }


def search(db: Session, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
    """
    向量检索 top-k。

    起步实现：向量存在 SQLite，全量算余弦相似度（几千片毫无压力）。
    上量后把这里换成 ChromaDB / pgvector / 阿里云向量检索即可，
    函数签名与返回值保持不变，调用方（interpreter）无需修改。
    """
    top_k = top_k or settings.KB_TOP_K
    query = (query or "").strip()
    if not query:
        return []

    vector = embed_texts([query])[0]
    query_norm = math.sqrt(sum(value * value for value in vector)) or 1.0

    rows = db.execute(select(KbChunk, KbDocument.title).join(KbDocument, KbChunk.doc_id == KbDocument.id)).all()
    scored: list[RetrievedChunk] = []
    for chunk, doc_title in rows:
        candidate = _deserialize(chunk.embedding)
        if not candidate or len(candidate) != len(vector):
            continue
        dot = sum(a * b for a, b in zip(vector, candidate))
        score = dot / (query_norm * (chunk.norm or 1.0))
        if score > 0:
            scored.append(RetrievedChunk(chunk_id=chunk.id, doc_title=doc_title, text=chunk.chunk_text, score=score))

    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[:top_k]


def build_query(features: dict[str, Any], extra: str | None = None) -> str:
    """用特征里的标签拼检索词（保留给接口调试用）。"""
    return " ".join(build_queries(features, extra))


def build_queries(features: dict[str, Any], extra: str | None = None) -> list[str]:
    """
    把结构化特征展开成**多条**定向检索词。

    为什么不用一条总检索词：单条长句向量会被稀释，容易只命中泛泛的条目
    （实测只捞到一条与本人无关的「耳相」片段）。按部位分头检索，
    每个部位都能拿到该部位最相关的知识片段，再合并去重，召回和相关性都明显更好。
    """
    shape = features.get("face_shape", {}) or {}
    courts = features.get("three_courts", {}) or {}
    eyes = features.get("five_eyes", {}) or {}
    facial = features.get("facial_features", {}) or {}
    complexion = features.get("complexion", {}) or {}

    def label(block: str) -> str:
        item = facial.get(block, {}) or {}
        return str(item.get("label", "") or "")

    queries: list[str] = [
        # 先说「本部位是什么样」，再问「这种相理传统上怎么评价（吉与不足都要）」
        "三庭 上庭 中庭 下庭 比例 面相",
        f"上庭 额部 {label('forehead')} 早年 面相",
        f"眼 眼距 {eyes.get('inner_gap_ratio', {}).get('label', '')} 眼型 {label('eye')} 五眼 面相",
        f"眉 {label('brow')} 眉眼距离 保寿官 面相",
        f"鼻 {label('nose')} 鼻头 鼻翼 审辨官 面相",
        f"口唇 {label('mouth')} 唇厚 出纳官 面相",
        f"气色 面色 {complexion.get('tone', '')} 印堂 面相",
        f"脸型 {shape.get('label', '')} 轮廓 气质 面相",
        # 不足面：捞出「劣相 / 偏弱 / 需留意」类条目，保证解读能讲出问题
        "三庭 上庭偏短 中庭偏长 下庭偏短 面相",
        "眉的劣相 眉短 眉薄 眉乱 眉逆生 面相",
        "眼的不足 眼突 眼凹 眼睑浮肿 面相",
        "鼻的不足 鼻梁低陷 鼻头无肉 面相",
        "口唇 薄唇 嘴角下垂 面相",
        "气色 偏暗 偏黄 面色不均 面相",
    ]
    if extra:
        queries.insert(0, extra)
    # 去掉空标签造成的重复/无效检索词
    seen: set[str] = set()
    cleaned: list[str] = []
    for query in queries:
        query = " ".join(query.split())
        if query and query not in seen:
            seen.add(query)
            cleaned.append(query)
    return cleaned


# 照片测不到的部位：耳朵形态、痣、法令纹、牙齿等无法从人脸关键点得出，
# 相关资料片段不参与解读，避免模型对着"没测过的部位"编解读。
UNMEASURABLE_KEYWORDS: tuple[str, ...] = (
    "耳", "痣", "法令纹", "牙齿", "牙", "手相", "掌纹", "骨相", "声音", "步态", "胎记",
)


def drop_unmeasurable(hits: list[RetrievedChunk], enabled: bool = True) -> list[RetrievedChunk]:
    """
    过滤掉「本次照片无法测量」的部位相关片段（标题里带耳朵/痣/法令纹等）。

    注意：只按**标题路径**判断，不会误伤正文里顺带提到这些字的内容。
    """
    if not enabled:
        return hits
    kept: list[RetrievedChunk] = []
    for hit in hits:
        heading = hit.text.strip()
        if heading.startswith("【") and "】" in heading:
            heading = heading[1 : heading.find("】")]
        else:
            heading = ""
        if heading and any(word in heading for word in UNMEASURABLE_KEYWORDS):
            continue
        kept.append(hit)
    return kept


def search_multi(
    db: Session, queries: list[str], per_query: int = 3, max_total: int = 8
) -> list[RetrievedChunk]:
    """
    多路检索：每条检索词各取 top-N，按 chunk 去重（保留最高分），再整体排序截断。

    排序策略：**你自己的知识优先**，内置兜底知识（默认相学常识）只用来补空位——
    否则兜底文档措辞更贴近检索词，会把用户自己的资料挤到后面，解读看起来「像没喂资料」。
    设 KB_INCLUDE_FALLBACK=false 可以完全不用兜底文档（此时若自有资料缺某主题，该主题就没有依据）。
    """
    merged: dict[int, RetrievedChunk] = {}
    for query in queries:
        for hit in search(db, query, per_query):
            current = merged.get(hit.chunk_id)
            if current is None or hit.score > current.score:
                merged[hit.chunk_id] = hit

    ranked = sorted(merged.values(), key=lambda item: item.score, reverse=True)
    fallback_title = settings.KB_FALLBACK_TITLE

    if not settings.KB_INCLUDE_FALLBACK:
        ranked = [item for item in ranked if item.doc_title != fallback_title]
        return ranked[:max_total]

    own = [item for item in ranked if item.doc_title != fallback_title]
    fallback = [item for item in ranked if item.doc_title == fallback_title]
    return (own + fallback)[:max_total]


def stats(db: Session) -> dict[str, Any]:
    documents = db.execute(select(func.count(KbDocument.id))).scalar() or 0
    chunks = db.execute(select(func.count(KbChunk.id))).scalar() or 0
    return {
        "documents": int(documents),
        "chunks": int(chunks),
        "embedding_provider": type(get_embedder()).__name__,
        "embedding_dim": settings.EMBEDDING_DIM,
        "vector_store": settings.VECTOR_STORE,
        "docs_dir": str(KB_DOCS_DIR),
    }
