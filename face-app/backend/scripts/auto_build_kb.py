"""
auto_build_kb.py —— 从原始 PDF 资料一键构建面相知识库

整条流水线（三步可分开跑，也可以一条命令跑完）：

    ① --extract  raw_sources/*.pdf
                   ├─ 数字版页面（能选中文字）→ PyMuPDF 直接抽文字层
                   └─ 扫描版页面（整页是图）  → 栅格化后交给中文 OCR（PaddleOCR / RapidOCR）
                 → raw_sources/extracted/*.txt（带页码与抽取方式标记的原文）

    ② --distill  extracted/*.txt
                 → 交给大模型整理：去绝对化/恐吓表述、中性现代中文、
                   按 # 主题（三庭/五眼/五官/气色/脸型/十二宫）+ ## 小节 组织、
                   保留「三庭五眼、印堂、气色、脸型」等关键术语
                 → knowledge_base/docs/*.md

    ③ --ingest   docs/*.md → 切片 + 向量化入库（与 scripts/ingest_local.py 同一代码路径）

常用命令：
    python scripts/auto_build_kb.py --all                 # 抽取 + 整理 + 入库（推荐）
    python scripts/auto_build_kb.py --extract             # 只抽取（先看 OCR 质量再决定要不要整理）
    python scripts/auto_build_kb.py --distill             # 只整理
    python scripts/auto_build_kb.py --ingest              # 只入库
    python scripts/auto_build_kb.py --extract --pages 1-30 --dpi 250
    python scripts/auto_build_kb.py --extract --ocr-engine rapidocr
    python scripts/auto_build_kb.py --distill --dry-run    # 只打印将发给大模型的 Prompt，不实际调用
    python scripts/auto_build_kb.py --list                 # 看目录里都有什么

依赖：
    pip install pymupdf                    # 必需：抽文字层 + 扫描页栅格化
    pip install paddleocr paddlepaddle     # 可选：中文 OCR（效果最好，但体积大）
    pip install rapidocr-onnxruntime       # 可选：中文 OCR（onnx 版，轻量，无需 paddle）
    （两个 OCR 装哪个都行，脚本会自动挑；都没装时只能处理数字版 PDF）
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# 让脚本能直接 import backend 下的模块
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from config import KB_DOCS_DIR, settings  # noqa: E402
from models.db import SessionLocal, init_db  # noqa: E402
from services import compliance, interpreter, rag  # noqa: E402

RAW_DIR = BACKEND_DIR / "raw_sources"
EXTRACT_DIR = RAW_DIR / "extracted"

# 整理后的 Markdown 允许出现的一级主题（原文没涉及的可以不出现）
TOPICS = ("三庭", "五眼", "五官", "气色", "脸型", "十二宫")

# 一页里文字层字数低于这个值，就认为它是扫描页，走 OCR
MIN_TEXT_CHARS_PER_PAGE = 30

HEADER_RE = re.compile(r"^###\s")
PAGE_MARK_RE = re.compile(r"^\[第\s*\d+\s*页[^\]]*\]\s*$")
TOPIC_RE = re.compile(r"^#\s+(.+?)\s*$")
SECTION_RE = re.compile(r"^##\s+(.+?)\s*$")


# ===========================================================================
# 一、PDF 抽取
# ===========================================================================
@dataclass
class PageResult:
    number: int
    method: str          # "text" | "ocr" | "skipped"
    text: str
    note: str = ""


@dataclass
class ExtractResult:
    source: Path
    output: Path | None
    pages: list[PageResult] = field(default_factory=list)

    @property
    def text_pages(self) -> int:
        return sum(1 for page in self.pages if page.method == "text")

    @property
    def ocr_pages(self) -> int:
        return sum(1 for page in self.pages if page.method == "ocr")

    @property
    def total_chars(self) -> int:
        return sum(len(page.text) for page in self.pages)


def _clean_stem(path: Path) -> str:
    """
    取文件名主干，并清掉重复的扩展名。

    很多人重命名时会手滑成 `面相资料.pdf.pdf`，直接取 stem 会得到「面相资料.pdf」，
    生成的知识库标题就很难看，这里统一处理掉。
    """
    stem = path.stem
    for suffix in (".pdf", ".txt", ".md", ".PDF", ".TXT", ".MD"):
        while stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem or path.stem


def _require_fitz():
    """PyMuPDF：1.24+ 推荐 `import pymupdf`，老版本用 `import fitz`（同一个库）。"""
    try:
        import pymupdf  # PyMuPDF >= 1.24
        return pymupdf
    except Exception:  # noqa: BLE001
        pass
    try:
        import fitz  # 旧导入名
        return fitz
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "缺少 PyMuPDF，无法读取 PDF。请先安装：\n"
            "    pip install pymupdf\n"
            f"（原始错误：{exc}）"
        ) from exc


def _page_to_bgr(page, dpi: int):
    """把 PDF 页面栅格化成 OpenCV 的 BGR 图像（OCR 用）。"""
    import numpy as np

    pixmap = page.get_pixmap(dpi=dpi)
    array = np.frombuffer(pixmap.samples, dtype=np.uint8)
    channels = pixmap.n
    array = array.reshape(pixmap.height, pixmap.width, channels)
    if channels == 4:      # RGBA → RGB
        array = array[:, :, :3]
    if channels == 1:      # 灰度 → 三通道
        array = np.repeat(array, 3, axis=2)
    return array[:, :, ::-1].copy()   # RGB → BGR


# ---------------------------------------------------------------------------
# OCR 引擎：统一封装，自动挑可用的那个
# ---------------------------------------------------------------------------
class OcrBase:
    name = "base"

    def read(self, image_bgr) -> list[tuple[str, float, list[list[float]]]]:
        """返回 [(文本, 置信度, 四点框), ...]。"""
        raise NotImplementedError

    @staticmethod
    def _rect(quad: list) -> tuple[float, float, float, float]:
        xs = [float(point[0]) for point in quad]
        ys = [float(point[1]) for point in quad]
        return min(xs), min(ys), max(xs), max(ys)


class PaddleOcrEngine(OcrBase):
    """PaddleOCR（2.x / 3.x 两种 API 都兼容）。"""

    name = "paddleocr"

    def __init__(self) -> None:
        from paddleocr import PaddleOCR  # type: ignore

        try:   # 3.x
            self.engine = PaddleOCR(lang="ch", use_textline_orientation=True)
            self.api = "v3"
        except TypeError:   # 2.x
            self.engine = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
            self.api = "v2"

    def read(self, image_bgr):
        items: list[tuple[str, float, list[list[float]]]] = []
        if self.api == "v3":
            results = self.engine.predict(image_bgr)
            for result in results or []:
                texts = result.get("rec_texts") or []
                scores = result.get("rec_scores") or []
                polys = result.get("dt_polys") or result.get("rec_polys") or []
                for text, score, poly in zip(texts, scores, polys):
                    items.append((str(text), float(score), [[float(x), float(y)] for x, y in poly]))
            return items

        results = self.engine.ocr(image_bgr, cls=True)
        for page in results or []:
            for line in page or []:
                box, (text, score) = line[0], line[1]
                items.append((str(text), float(score), [[float(x), float(y)] for x, y in box]))
        return items


class RapidOcrEngine(OcrBase):
    """RapidOCR（PP-OCR 模型的 onnxruntime 版，轻量、无需 paddle）。"""

    name = "rapidocr"

    def __init__(self) -> None:
        from rapidocr_onnxruntime import RapidOCR  # type: ignore

        self.engine = RapidOCR()

    def read(self, image_bgr):
        result, _elapse = self.engine(image_bgr)
        items: list[tuple[str, float, list[list[float]]]] = []
        for row in result or []:
            box, text, score = row[0], row[1], row[2]
            items.append((str(text), float(score), [[float(x), float(y)] for x, y in box]))
        return items


def build_ocr_engine(preference: str):
    """按偏好挑 OCR 引擎；装不上的自动跳过。preference: auto | paddleocr | rapidocr | none"""
    if preference == "none":
        return None

    candidates = []
    if preference in ("auto", "paddleocr"):
        candidates.append(("paddleocr", PaddleOcrEngine))
    if preference in ("auto", "rapidocr"):
        candidates.append(("rapidocr", RapidOcrEngine))
    if preference not in ("auto", "paddleocr", "rapidocr"):
        raise SystemExit(f"未知的 OCR 引擎：{preference}（可选 auto/paddleocr/rapidocr/none）")

    errors = []
    for label, engine_class in candidates:
        try:
            engine = engine_class()
            print(f"   OCR 引擎：{label} 已就绪")
            return engine
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{label}: {type(exc).__name__}: {exc}")

    print("   ⚠️ 没有可用的 OCR 引擎，扫描版页面会被跳过")
    for line in errors:
        print(f"     - {line}")
    print("     安装任一即可：pip install rapidocr-onnxruntime  或  pip install paddleocr paddlepaddle")
    return None


# ---------------------------------------------------------------------------
# 阅读顺序重排：竖排古籍按「从右到左、每列从上到下」
# ---------------------------------------------------------------------------
def _cluster(values, tolerance: float):
    """把一维坐标聚类（用于把 OCR 框归成行或列），返回 [(中心, [索引]), ...]，按中心升序。"""
    groups: list[dict] = []
    for index, value in sorted(values, key=lambda item: item[1]):
        center = value
        for group in groups:
            if abs(group["center"] - center) <= tolerance:
                group["items"].append(index)
                group["center"] = (group["center"] * group["n"] + center) / (group["n"] + 1)
                group["n"] += 1
                break
        else:
            groups.append({"center": center, "n": 1, "items": [index]})
    return [(group["center"], group["items"]) for group in groups]


def boxes_to_text(items: list[tuple[str, float, list[list[float]]]], min_score: float = 0.5) -> str:
    """
    把 OCR 结果按阅读顺序拼成文本。

    古籍常见竖排：整页以纵向列为主时，按「列从右到左 + 列内从上到下」重排；
    横排则按「行从上到下 + 行内从左到右」。判断依据是框的宽高比中位数。
    """
    kept = [
        (text, score, quad, OcrBase._rect(quad))
        for text, score, quad in items
        if score >= min_score and text.strip()
    ]
    if not kept:
        return ""

    widths = [rect[2] - rect[0] for _t, _s, _q, rect in kept]
    heights = [rect[3] - rect[1] for _t, _s, _q, rect in kept]
    widths.sort()
    heights.sort()
    median_w = widths[len(widths) // 2] or 1.0
    median_h = heights[len(heights) // 2] or 1.0
    vertical = (median_h / max(median_w, 1.0)) >= 1.6

    rects = [rect for _t, _s, _q, rect in kept]
    if vertical:
        # 竖排：先按 x 聚成列（从右到左），列内按 y 从上到下
        columns = _cluster([(index, (rects[index][0] + rects[index][2]) / 2) for index in range(len(kept))],
                           tolerance=median_w * 0.6)
        lines = []
        for _center, indexes in sorted(columns, key=lambda item: -item[0]):
            ordered = sorted(indexes, key=lambda index: (rects[index][1] + rects[index][3]) / 2)
            lines.append("".join(kept[index][0].strip() for index in ordered))
        return "\n".join(line for line in lines if line)

    # 横排：先按 y 聚成行，行内按 x 从左到右
    rows = _cluster([(index, (rects[index][1] + rects[index][3]) / 2) for index in range(len(kept))],
                    tolerance=median_h * 0.6)
    lines = []
    for _center, indexes in sorted(rows, key=lambda item: item[0]):
        ordered = sorted(indexes, key=lambda index: (rects[index][0] + rects[index][2]) / 2)
        lines.append("".join(kept[index][0].strip() for index in ordered))
    return "\n".join(line for line in lines if line)


# ---------------------------------------------------------------------------
# 抽取单份 PDF
# ---------------------------------------------------------------------------
def _enhance_for_ocr(image):
    """
    针对老旧/低对比扫描件的预处理：灰度 → 轻度去噪 → Otsu 二值化。

    只在 OCR 首轮结果不理想时作为候选方案之一（对照片型扫描可能反而变差，
    所以用「哪个结果字数更多」来择优，而不是强行预处理）。
    """
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def ocr_page(page, ocr_engine, args) -> tuple[str, str]:
    """
    对一页做 OCR，返回 (文本, 说明)。

    策略：先按 --dpi 跑一遍；若结果太差（框少/字数少），自动提高 DPI 并追加一次二值化预处理，
    取字数最多的那个结果。竖排古籍的关键是**足够的分辨率**，DPI 不够时 PP-OCR 会漏检整列。
    """
    primary = max(args.dpi, 100)
    attempts: list[tuple[int, bool]] = [(primary, False)]
    if primary < 300:
        attempts.append((300, False))
    attempts.append((max(primary, 300), True))

    best_text, best_note = "", ""
    for dpi_value, enhance in attempts:
        try:
            image = _page_to_bgr(page, dpi_value)
            if enhance:
                image = _enhance_for_ocr(image)
            items = ocr_engine.read(image)
            text = boxes_to_text(items, min_score=args.min_score)
        except Exception as exc:  # noqa: BLE001
            best_note = best_note or f"OCR 失败：{type(exc).__name__}: {exc}"
            continue

        note = f"{ocr_engine.name} @{dpi_value}dpi" + ("+二值化" if enhance else "")
        note += f" / {len(items)} 框"
        if len(text) > len(best_text):
            best_text, best_note = text, note
        # 质量已经够好就不再折腾（省时间）
        if len(items) >= 6 and len(text) >= 40:
            break
    return best_text, best_note


def parse_pages(spec: str | None, total: int) -> list[int]:
    """解析 --pages，如 "1-30,45"；页码从 1 开始，返回 0 基索引列表。"""
    if not spec:
        return list(range(total))
    wanted: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, _, end = part.partition("-")
            start = int(start) if start.strip() else 1
            end = int(end) if end.strip() else total
            wanted.update(range(start, end + 1))
        else:
            wanted.add(int(part))
    return sorted(index - 1 for index in wanted if 1 <= index <= total)


def extract_pdf(path: Path, args, ocr_engine) -> ExtractResult:
    """抽取一份 PDF：逐页判断文字层 / 扫描页，扫描页走 OCR。"""
    fitz = _require_fitz()
    result = ExtractResult(source=path, output=None)

    with fitz.open(path) as document:
        page_indexes = parse_pages(args.pages, document.page_count)
        for index in page_indexes:
            page = document.load_page(index)
            raw_text = (page.get_text("text") or "").strip()
            number = index + 1

            use_ocr = args.force_ocr or len(raw_text) < MIN_TEXT_CHARS_PER_PAGE
            if use_ocr and ocr_engine is not None:
                text, note = ocr_page(page, ocr_engine, args)
                if text.strip():
                    result.pages.append(PageResult(number, "ocr", text, note))
                else:
                    result.pages.append(PageResult(number, "ocr", "", f"OCR 未识别出文字（{note}）"))
                continue

            if use_ocr and ocr_engine is None:
                result.pages.append(
                    PageResult(number, "skipped", "", "扫描页但没有可用 OCR 引擎")
                )
                continue

            result.pages.append(PageResult(number, "text", raw_text, "文字层"))

    if not any(page.text.strip() for page in result.pages):
        result.output = None
        return result

    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    output = EXTRACT_DIR / f"{_clean_stem(path)}.txt"
    generated_at = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    source_label = "OCR" if result.ocr_pages and not result.text_pages else (
        "文字层" if result.text_pages and not result.ocr_pages else "文字层 + OCR"
    )
    lines = [
        f"### 来源: {path.name}",
        f"### 抽取方式: {source_label}（文字层 {result.text_pages} 页 / OCR {result.ocr_pages} 页）",
        f"### 生成时间: {generated_at}",
        "### 说明: 本文件是原始资料的第一步产物，尚未做中性化整理，请勿直接入库。",
        "",
    ]
    for page in result.pages:
        lines.append(f"[第 {page.number} 页 | {page.method} | {page.note}]")
        if page.text.strip():
            lines.append(page.text.strip())
        lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")
    result.output = output
    return result


def run_extract(args) -> list[ExtractResult]:
    print("=" * 72)
    print("① 抽取原文：raw_sources/*.pdf → raw_sources/extracted/*.txt")
    print("=" * 72)

    # Windows 下 glob 不区分大小写，"*.pdf" 与 "*.PDF" 会匹配到同一批文件，这里去重
    pdfs: list[Path] = []
    seen: set[str] = set()
    for pattern in ("*.pdf", "*.PDF"):
        for path in sorted(RAW_DIR.glob(pattern)):
            key = str(path.resolve()).lower()
            if key not in seen:
                seen.add(key)
                pdfs.append(path)
    if not pdfs:
        print(f"   {RAW_DIR} 下没有找到 PDF。把你的原始资料（含扫描版）放进去再跑一次。")
        return []

    ocr_engine = build_ocr_engine(args.ocr_engine)
    results: list[ExtractResult] = []
    for path in pdfs:
        output = EXTRACT_DIR / f"{_clean_stem(path)}.txt"
        if output.exists() and not args.force:
            print(f"   · {path.name}：已存在 {output.name}，跳过（--force 可重新抽取）")
            continue
        started = time.time()
        print(f"   · {path.name} 处理中…")
        result = extract_pdf(path, args, ocr_engine)
        cost = time.time() - started
        if result.output is None:
            print(f"     ✗ 没有抽到任何文字（OCR {result.ocr_pages} 页 / 跳过 {len(result.pages) - result.ocr_pages} 页）")
        else:
            print(
                f"     ✓ {result.output.name}：{result.total_chars} 字，"
                f"文字层 {result.text_pages} 页 / OCR {result.ocr_pages} 页，用时 {cost:.1f}s"
            )
            for page in result.pages:
                if page.method == "skipped":
                    print(f"       ! 第 {page.number} 页被跳过：{page.note}")
        results.append(result)
    return results


# ===========================================================================
# 二、大模型整理
# ===========================================================================
SYSTEM_PROMPT = """你是面相传世文献的整理编辑，任务是把古籍（含 OCR 结果）改写成可公开展示的传统文化科普文字。

必须遵守：
1. 只做改写、归纳与去芜存菁，不得增补原文没有的说法，不得编造出处或数据；
2. 语言中性、客观、现代中文，定位是「传统文化参考」；不承诺命运结果，不制造焦虑；
3. 严禁任何绝对化或恐吓性表述（例如「必定发财」「大富大贵」「克夫」「克妻」「大凶」「主死」「绝后」），
   一律改写成「传统相学认为……，仅供文化参考」这类中性描述，或直接删除；
4. 按主题归类，一级标题只能从下列六个里选（原文确实涉及的才写）：
   # 三庭   # 五眼   # 五官   # 气色   # 脸型   # 十二宫
   一级标题下用 `## 小节名` 组织；内容多时可分成多个小节；
5. 必须自然保留并可检索到关键术语：三庭五眼、上庭、中庭、下庭、印堂、山根、气色、脸型、
   鹅蛋脸、方形脸、圆形脸、长形脸、瓜子脸、眉、眼、鼻、口、唇 等；
6. 只输出 Markdown 正文，不要代码块、不要额外解释、不要写免责声明结尾；
7. 原文若为繁体、竖排或存在 OCR 错字，请按上下文还原为通顺的简体表述；
8. 与面部比例、气色无关的命理推断内容（如择日、风水、禄命）直接舍弃；
9. **保留限定条件**：原文里的部位、左右、形态、年龄等限定词是结论成立的前提，必须原样带进整理结果，
   不得删掉限定词把结论泛化。例如「耳根部有痣」不能写成「有痣」，「左耳小」不能写成「耳小」。
10. **把断言换成「注意事项」式的说法，而不是整条删除**——用户看到的是提醒，不是预言：
   - 「耳根部有痣：有阑尾炎」→「传统说法认为耳根部有痣与消化系统状况相关，可留意日常体检」
     （保留「耳根部」，去掉具体病名与「溃烂的正发作」这类描述）；
   - 「左耳小主父先亡，右耳短小主母先亡」→「传统说法提示：可多留意父亲（左耳）与母亲（右耳）的身体健康」；
   - 「招风耳：命不好，克父特严重」→「传统说法提示可多关心家人健康，与个人命运无关」；
   - 「耳廓凸的心脏不好」「耳轮不好心脑血管病」→「传统说法提示可留意心脏与心脑血管方面的健康检查」；
   - 「下眼睑饱满者生男孩」→「传统说法中有与生育情况相关的观察角度，无科学依据」；
   - 「鼻头有肉者必发财」→「传统说法认为鼻头有肉与积蓄能力相关」。
   原则：**只保留"观察角度 + 可留意的方向"，不保留诊断、不保留吉凶断言、不做生死预测**。
11. 只输出整理后的正文：不要写「原文中某某属绝对化表述，已删除/已改为中性」这类编辑说明，
   不要复述原文里的绝对化、恐吓性用语，也不要写「本节无内容可整理」这类占位小节——
   某个主题原文没有涉及，就整节不要输出；
12. 现代人写的笔记、讲义、口语记录同样按上述规则处理：保留可公开传播的面部比例与观察角度，
   去掉健康诊断、命运预言以及他人隐私（人名、班级、联系方式）；
13. **小节标题必须是中性词**，不要出现生死、吉凶、克害、病症等字眼。
   例如「父母寿夭观察」要写成「父母的健康观察角度」，「克夫相」要写成「婚姻相处的观察角度」。"""

USER_TEMPLATE = """【资料出处】{source}{page_hint}

【原文（可能含 OCR 错字、繁体、竖排导致的断行，请自行判断）】
{content}

【任务】把上面的原文整理成中性的现代中文，按主题组织为 Markdown（一级标题限：{topics}）。
只输出 Markdown 正文。"""


def _clean_extracted(text: str) -> str:
    """去掉抽取阶段的 ### 头与页码标记，只留正文。"""
    lines = []
    for line in text.splitlines():
        if HEADER_RE.match(line) or PAGE_MARK_RE.match(line):
            continue
        lines.append(line)
    return "\n".join(lines)


def split_chunks(text: str, size: int) -> list[str]:
    """按段落边界把长文切成 ~size 字的块。"""
    paragraphs = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    chunks: list[str] = []
    buffer: list[str] = []
    length = 0
    for paragraph in paragraphs:
        if length + len(paragraph) > size and buffer:
            chunks.append("\n\n".join(buffer))
            buffer, length = [], 0
        buffer.append(paragraph)
        length += len(paragraph)
    if buffer:
        chunks.append("\n\n".join(buffer))
    return chunks


def distill_chunk(source: str, chunk: str, page_hint: str, args) -> tuple[str, str]:
    """调用大模型整理一段原文，返回 (markdown, provider)。"""
    user_prompt = USER_TEMPLATE.format(
        source=source, page_hint=page_hint, content=chunk, topics="、".join(TOPICS)
    )
    if args.dry_run:
        print("-" * 72)
        print("[dry-run] 将发送给大模型的 Prompt：")
        print(user_prompt[:2000])
        print("-" * 72)
        return "", "dry-run"

    text, provider = interpreter.complete(
        SYSTEM_PROMPT, user_prompt, max_tokens=args.max_tokens, temperature=0.3
    )
    return text.strip(), provider


def _sanitize_heading(line: str) -> str:
    """
    标题专用清洗。

    直接对标题跑 sanitize 会把「父母寿元观察角度」变成
    「父母传统相学认为需注意保养观察角度」这种读不通的标题，
    所以标题改用一个中性的词表来替换（敏感词 → 中性主题词）。
    """
    text = line
    for words, replacement in HEADING_WORD_MAP:
        for word in words:
            if word in text:
                text = text.replace(word, replacement)
    # 兜底：仍有残留就退化成通用提示词，保证标题可读
    for word in settings.BANNED_PHRASES:
        if word in text:
            text = text.replace(word, "相关")
    return text


HEADING_WORD_MAP: tuple[tuple[tuple[str, ...], str], ...] = (
    (("寿元不长", "寿元", "寿数", "短寿", "夭折", "折寿", "早逝", "早亡", "先亡", "先损",
      "损父", "损母", "寿终正寝", "父母双亡", "主病", "死", "亡"), "健康"),
    (("克夫", "克妻", "克父母", "克子", "刑克", "孤寡", "无子"), "家庭相处"),
    (("生男孩", "生女孩", "生男", "生女", "头胎"), "生育"),
    (("贫贱", "贱格", "主贫", "破财"), "生活状况"),
    (("大凶", "凶兆", "血光", "牢狱", "横死", "必发财"), "相关说法"),
)


def _filter_compliance(markdown: str) -> tuple[str, list[str]]:
    """
    合规兜底（按句处理，尽量少误伤）：

      1. 正文行用 compliance.sanitize 把已知的绝对化/恐吓表述软化成中性说法；
         标题行用 _sanitize_heading 换成中性主题词（避免出现读不通的标题）；
      2. 仍含违禁词的行，按句子拆开，只丢问题句、保留同行的其它句子；
      3. 丢弃模型可能写出的「编辑说明」句（如「原文中'克夫'属绝对化表述，已删除」）——
         这些话对知识库没价值，还会把敏感词带进入库内容。
    """
    kept_lines: list[str] = []
    dropped: list[str] = []

    for raw_line in markdown.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            kept_lines.append(raw_line)
            continue

        line = _sanitize_heading(raw_line) if stripped.startswith("#") else compliance.sanitize(raw_line)
        stripped = line.strip()

        if not compliance.find_violations(stripped):
            if EDITORIAL_NOTE_RE.search(stripped):
                dropped.append(stripped)
            else:
                kept_lines.append(line)
            continue

        good: list[str] = []
        for sentence in (piece for piece in SENTENCE_SPLIT_RE.split(stripped) if piece.strip()):
            if compliance.find_violations(sentence) or EDITORIAL_NOTE_RE.search(sentence):
                dropped.append(sentence.strip())
            else:
                good.append(sentence)
        if good:
            kept_lines.append("".join(good).strip())

    return "\n".join(kept_lines).strip(), dropped


# 「原文中…属绝对化/恐吓性表述，已删除」这类编辑说明
EDITORIAL_NOTE_RE = re.compile(
    r"(属绝对化|属恐吓|已按规范删除|已改为中性描述|已删除|本节无内容|无内容可整理)"
)
# 中英文句末标点后切分
SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？；!?;])")

COMPLIANCE_SYSTEM_PROMPT = """你是内容合规审核员。用户会给你一份「面相文化资料」的 Markdown。
你只做一件事：把其中仍然存在的绝对化、恐吓性、命定论表述，改写成中性的「观察角度 + 可留意的方向」。

改写原则（**换说法，不要整条删除**，同时不得丢掉部位、左右等限定条件）：
- 健康类：「耳根部有痣有阑尾炎」→「传统说法认为耳根部有痣与消化系统状况相关，可留意日常体检」；
- 六亲类：「左耳小主父先亡」→「传统说法提示可多留意父亲的身体健康」；
- 婚姻类：「不好找老公」→「传统说法认为感情上宜多沟通」；
- 贫富类：「鼻头有肉者必发财」→「传统说法认为鼻头有肉与积蓄能力相关」；
- 生育类：不预测性别，改为「传统说法中有与生育情况相关的观察角度，无科学依据」；
- 保留部位、左右、形态等限定词，**不要**把「耳根部有痣」写成「有痣」。

另外：不要写编辑说明或合规说明、不要复述恐吓性原文、不要调整标题结构、不要添加免责声明结尾。
小节标题若含生死/吉凶字眼（如「父母寿夭观察」），请改成中性标题（如「父母的健康观察角度」）。
直接输出修改后的 Markdown 全文。"""


def compliance_review(markdown: str, source: str, args, provider_box: dict) -> str:
    """可选的大模型合规复审：对任意古籍内容都更稳（代价是多一次模型调用）。"""
    if args.dry_run:
        print("     [dry-run] 跳过合规复审")
        return markdown
    try:
        text, provider = interpreter.complete(
            COMPLIANCE_SYSTEM_PROMPT,
            f"【资料】{source}\n\n【待审核 Markdown】\n{markdown}",
            max_tokens=args.max_tokens,
            temperature=0.1,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"     合规复审失败（{type(exc).__name__}: {exc}），沿用规则过滤结果")
        return markdown
    provider_box["provider"] = provider
    if not text.strip():
        return markdown
    return text.strip()


def _is_placeholder(body: str) -> bool:
    """判断某个小节是不是「原文未涉及」这类占位内容（入库只会制造噪声，直接丢掉）。"""
    text = body.strip().strip(">").strip()
    if len(text) > 90:
        return False
    return bool(re.search(r"(原文未|未涉及|不涉及|无相关论述|无内容|未作|没有涉及)", text))


def merge_markdown(blocks: list[str]) -> str:
    """
    把多次大模型输出的 Markdown 合并成一篇：同一主题下的多个小节归到一起。

    小节标题重复时自动加序号，避免同一个小节名出现多次而互相盖掉。
    """
    topics: dict[str, list[tuple[str, str]]] = {}
    order: list[str] = []
    fallback_topic = "面相常识"

    for block in blocks:
        current_topic = None
        current_section = None
        buffer: list[str] = []

        def flush() -> None:
            nonlocal buffer, current_section
            body = "\n".join(buffer).strip()
            buffer = []
            if not body or _is_placeholder(body):
                return
            topic = current_topic or fallback_topic
            if topic not in topics:
                topics[topic] = []
                order.append(topic)
            title = current_section or "综述"
            existing = [item[0] for item in topics[topic]]
            if title in existing:
                index = existing.count(title) + 1
                title = f"{title}（{index}）"
            topics[topic].append((title, body))

        for line in block.splitlines():
            topic_match = TOPIC_RE.match(line)
            section_match = SECTION_RE.match(line)
            if topic_match:
                flush()
                current_topic = topic_match.group(1).strip()
                current_section = None
                continue
            if section_match:
                flush()
                current_section = section_match.group(1).strip()
                continue
            buffer.append(line)
        flush()

    if not topics:
        return ""

    # 让六个基本主题按固定顺序排前面，其余（如模型自造的标题）按出现顺序排后面
    ordered_topics = [topic for topic in TOPICS if topic in topics]
    ordered_topics += [topic for topic in order if topic not in ordered_topics]

    lines: list[str] = []
    for topic in ordered_topics:
        lines.append(f"# {topic}")
        lines.append("")
        for title, body in topics[topic]:
            lines.append(f"## {title}")
            lines.append("")
            lines.append(body)
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def distill_file(txt_path: Path, args, provider_box: dict) -> tuple[Path | None, dict]:
    raw = txt_path.read_text(encoding="utf-8")
    source_name = _clean_stem(txt_path)
    body = _clean_extracted(raw)
    if len(body.strip()) < 40:
        return None, {"skipped": "内容过短"}

    chunks = split_chunks(body, args.chunk_chars)
    print(f"   · {txt_path.name} → {len(chunks)} 段待整理")
    blocks: list[str] = []
    dropped_lines: list[str] = []
    failed_chunks: list[int] = []

    if args.dry_run:
        distill_chunk(source_name, chunks[0], "", args)
        print("   （dry-run：只打印了第 1 段的 Prompt，未调用大模型、未写出文件）")
        return None, {"skipped": "dry-run"}

    for index, chunk in enumerate(chunks, start=1):
        print(f"     第 {index}/{len(chunks)} 段（{len(chunk)} 字）…", end="", flush=True)
        markdown = ""
        provider = ""
        for attempt in (1, 2):          # 大模型偶尔超时，重试一次再放弃
            try:
                markdown, provider = distill_chunk(source_name, chunk, "", args)
                break
            except interpreter.LLMNotConfigured as exc:
                raise SystemExit(f"整理需要大模型：{exc}") from exc
            except Exception as exc:  # noqa: BLE001
                if attempt == 1:
                    print(f" {type(exc).__name__}，重试…", end="", flush=True)
                    time.sleep(5)
                    continue
                failed_chunks.append(index)
                print(f" 失败：{type(exc).__name__}: {exc}")

        if provider:
            provider_box["provider"] = provider
        if index in failed_chunks:
            continue
        if not markdown:
            print(" 空输出")
            failed_chunks.append(index)
            continue
        markdown, dropped = _filter_compliance(markdown)
        dropped_lines.extend(dropped)
        blocks.append(markdown)
        print(f" ✓ {len(markdown)} 字")

    # 有任何一段失败就不写出，避免生成「缺了一块」的知识库文档（旧版本会保留）
    if failed_chunks:
        return None, {
            "skipped": f"第 {failed_chunks} 段整理失败，已跳过写出以免知识库缺内容；"
                       f"可调大 LLM_TIMEOUT_SECONDS 或调小 --chunk-chars 后重跑",
            "dropped": dropped_lines,
        }

    merged = merge_markdown(blocks)
    if not merged:
        return None, {"skipped": "整理后没有内容", "dropped": dropped_lines}

    if args.compliance_review:
        print("     合规复审（大模型）…", end="", flush=True)
        merged = compliance_review(merged, source_name, args, provider_box)
        merged, extra_dropped = _filter_compliance(merged)
        dropped_lines.extend(extra_dropped)
        print(f" ✓ 剩余 {len(merged)} 字")
        if not merged:
            return None, {"skipped": "合规复审后没有可保留的内容", "dropped": dropped_lines}

    header = "\n".join([
        "<!-- 由 scripts/auto_build_kb.py 自动整理生成，请人工通读后再对外使用。",
        f"     资料来源: {source_name}（原始抽取文本见 raw_sources/extracted/{txt_path.name}）",
        f"     生成时间: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"     整理模型: {provider_box.get('provider', 'unknown')}",
        "     说明: 内容基于原文改写，定位为传统文化趣味参考；绝对化/恐吓性表述已中性化。-->",
    ])
    KB_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    output = KB_DOCS_DIR / f"{source_name}.md"
    if output.exists() and not args.force:
        return None, {"skipped": f"{output.name} 已存在（--force 可覆盖）"}
    output.write_text(f"{header}\n\n{merged}", encoding="utf-8")
    return output, {"dropped": dropped_lines, "chunks": len(chunks), "chars": len(merged)}


def run_distill(args) -> list[Path]:
    print("=" * 72)
    print("② 整理入库文本：raw_sources/extracted/*.txt → knowledge_base/docs/*.md")
    print("=" * 72)

    txts = sorted(EXTRACT_DIR.glob("*.txt"))
    plain = sorted(path for path in RAW_DIR.glob("*.txt"))
    if not txts and not plain:
        print(f"   {EXTRACT_DIR} 下没有抽取结果，请先跑 --extract（或把 .txt 直接放进 {RAW_DIR}）")
        return []

    # raw_sources 根目录下的 .md 视为「已整理」，直接复制进 docs，不调用大模型
    # （README.md 是本目录的说明文件，_ 开头的是临时文件，都不算资料）
    for ready in sorted(RAW_DIR.glob("*.md")):
        if ready.name.lower() == "readme.md" or ready.name.startswith("_"):
            continue
        target = KB_DOCS_DIR / ready.name
        if target.exists() and not args.force:
            print(f"   · {ready.name}：docs 里已存在，跳过")
            continue
        KB_DOCS_DIR.mkdir(parents=True, exist_ok=True)
        target.write_text(ready.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"   · {ready.name}：视为已整理，直接复制到 docs/")

    provider_box: dict[str, str] = {}
    outputs: list[Path] = []
    for txt_path in txts + plain:
        output, stats = distill_file(txt_path, args, provider_box)
        if output is None:
            print(f"   · {txt_path.name}：跳过（{stats.get('skipped')}）")
            continue
        outputs.append(output)
        dropped = stats.get("dropped") or []
        print(f"   ✓ 写出 {output.name}：{stats.get('chars')} 字，{stats.get('chunks')} 段整理")
        if dropped:
            print(f"     ⚠️ 合规过滤：丢弃了 {len(dropped)} 行仍含绝对化表述的内容，例如：")
            for line in dropped[:3]:
                print(f"        - {line[:60]}")
    return outputs


# ===========================================================================
# 三、入库
# ===========================================================================
def run_ingest(args) -> dict:
    print("=" * 72)
    print("③ 切片向量化入库（与 scripts/ingest_local.py 同一代码路径）")
    print("=" * 72)
    init_db()
    with SessionLocal() as db:
        result = rag.ingest_directory(db)
        stats = rag.stats(db)
    print(f"   新增文档 {result['documents']} 篇 / 切片 {result['chunks']} 条 / 跳过 {result['skipped']} 篇")
    if result["titles"]:
        print(f"   本次入库：{'、'.join(result['titles'])}")
    print(f"   知识库现状：{stats['documents']} 篇文档，{stats['chunks']} 条切片")

    if args.probe:
        print("   检索抽查：")
        probes = args.probe if isinstance(args.probe, list) else ["三庭 五眼 比例", "印堂 气色", "脸型 鹅蛋脸"]
        with SessionLocal() as db:
            for question in probes:
                hits = rag.search(db, question, 3)
                top = hits[0] if hits else None
                print(f"     · 「{question}」 → {len(hits)} 条命中" + (f"，最佳：{top.doc_title} / {top.score:.3f}" if top else ""))
    return result


# ===========================================================================
# 四、清单与入口
# ===========================================================================
def run_sanitize_only(args) -> None:
    """
    只对 docs/ 里已有的 Markdown 重新跑一遍规则过滤（不调用大模型、不花钱）。

    用途：往违禁词表里补了新词之后，用它把旧文档再过一遍，不必重新整理。
    """
    print("=" * 72)
    print("规则过滤：knowledge_base/docs/*.md（不调用大模型）")
    print("=" * 72)
    changed = 0
    for path in sorted(KB_DOCS_DIR.glob("*.md")):
        original = path.read_text(encoding="utf-8")
        cleaned, dropped = _filter_compliance(original)
        if cleaned.strip() == original.strip():
            print(f"   · {path.name}：无需改动")
            continue
        if args.dry_run:
            print(f"   · {path.name}：将修改 {len(dropped)} 处（dry-run 未写入）")
            for line in dropped[:5]:
                print(f"       - {line[:70]}")
            continue
        path.write_text(cleaned + "\n", encoding="utf-8")
        changed += 1
        print(f"   ✓ {path.name}：已更新，处理 {len(dropped)} 处")
        for line in dropped[:5]:
            print(f"       - {line[:70]}")
    print(f"   共更新 {changed} 篇文档；改完记得跑 --ingest 刷新向量库")


def run_list() -> None:
    print(f"原始资料目录：{RAW_DIR}")
    for path in sorted(RAW_DIR.glob("*")):
        if path.is_file() and path.suffix.lower() in {".pdf", ".txt", ".md"}:
            print(f"   · {path.name}  ({path.stat().st_size / 1024:.0f} KB)")
    print(f"抽取结果目录：{EXTRACT_DIR}")
    if EXTRACT_DIR.is_dir():
        for path in sorted(EXTRACT_DIR.glob("*.txt")):
            print(f"   · {path.name}  ({path.stat().st_size / 1024:.0f} KB)")
    print(f"整理结果目录：{KB_DOCS_DIR}")
    for path in sorted(KB_DOCS_DIR.glob("*.md")):
        print(f"   · {path.name}  ({path.stat().st_size / 1024:.0f} KB)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="从 raw_sources/ 里的 PDF（含扫描版）一键构建面相知识库",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--all", action="store_true", help="抽取 + 整理 + 入库（默认）")
    mode.add_argument("--extract", action="store_true", help="只抽取：PDF → extracted/*.txt")
    mode.add_argument("--distill", action="store_true", help="只整理：txt → docs/*.md")
    mode.add_argument("--ingest", action="store_true", help="只入库：docs/*.md → 向量库")
    mode.add_argument("--sanitize-only", action="store_true", help="只对已有 docs/*.md 重跑规则过滤（不调用大模型）")

    parser.add_argument("--force", action="store_true", help="覆盖已有产物（默认跳过已存在的）")
    parser.add_argument("--list", action="store_true", help="列出各目录现状后退出")
    parser.add_argument("--pages", default=None, help="只处理指定页，如 1-30,45（默认全部）")
    parser.add_argument("--dpi", type=int, default=300,
                        help="扫描页栅格化精度，默认 300（竖排古籍/小字建议 300-400，DPI 不够会整列漏检）")
    parser.add_argument("--min-score", type=float, default=0.5, help="OCR 置信度下限，默认 0.5")
    parser.add_argument("--force-ocr", action="store_true", help="忽略文字层，全部页面走 OCR")
    parser.add_argument("--ocr-engine", default="auto", choices=["auto", "paddleocr", "rapidocr", "none"],
                        help="OCR 引擎，默认 auto（paddleocr 优先）")
    parser.add_argument("--chunk-chars", type=int, default=3000, help="整理时每段原文的字数，默认 3000")
    parser.add_argument("--max-tokens", type=int, default=4000,
                        help="每段整理的最大输出长度（中文长文建议 ≥4000，太小会被截断）")
    parser.add_argument("--dry-run", action="store_true", help="只打印将发给大模型的 Prompt，不实际调用")
    parser.add_argument("--compliance-review", action="store_true",
                        help="整理后再让大模型做一次合规复审（古籍内容更保险，代价是多一次模型调用）")
    parser.add_argument("--probe", nargs="*", default=None, help="入库后做几条检索抽查")

    args = parser.parse_args()

    if args.list:
        run_list()
        return 0

    if args.sanitize_only:
        KB_DOCS_DIR.mkdir(parents=True, exist_ok=True)
        run_sanitize_only(args)
        return 0

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    KB_DOCS_DIR.mkdir(parents=True, exist_ok=True)

    do_extract = args.extract or args.all or not (args.distill or args.ingest or args.sanitize_only)
    do_distill = args.distill or args.all or not (args.extract or args.ingest or args.sanitize_only)
    do_ingest = args.ingest or args.all or not (args.extract or args.distill or args.sanitize_only)

    started = time.time()
    if do_extract:
        run_extract(args)
    if do_distill:
        run_distill(args)
    if do_ingest:
        try:
            run_ingest(args)
        except Exception as exc:  # noqa: BLE001
            print(f"   入库失败：{type(exc).__name__}: {exc}")
            return 1

    print("=" * 72)
    print(f"完成，用时 {time.time() - started:.1f}s")
    print("下一步建议：")
    print("   1) 打开 knowledge_base/docs/ 下新生成的 .md 通读一遍（自动整理不是 100% 可靠）")
    print("   2) 用 POST /api/kb/query 或前端结果页看检索是否命中新内容")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
