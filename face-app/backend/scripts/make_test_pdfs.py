"""
生成两份「测试用」PDF，用来验证 scripts/auto_build_kb.py 的两条抽取路径，不需要真古籍。

    1) 测试-数字版.pdf   ：有文字层（PyMuPDF 可直接抽出）      → 走「文字层」路径
    2) 测试-扫描竖排.pdf ：整页是一张图、繁体竖排、无文字层      → 走「OCR + 阅读顺序重排」路径

两份内容都故意混入了绝对化/恐吓性表述（必发财、克夫、大凶、寿元不长…），
用来验证整理环节确实会做中性化，以及合规过滤是否生效。

用法（在 backend 目录下）：
    python scripts/make_test_pdfs.py                 # 输出到 raw_sources/
    python scripts/make_test_pdfs.py --out D:/tmp    # 输出到指定目录

跑完可以接着验证整条流水线：
    python scripts/auto_build_kb.py --all
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUT = BACKEND_DIR / "raw_sources"

try:
    import pymupdf  # PyMuPDF >= 1.24
except ImportError:  # pragma: no cover - 兼容旧版本导入名
    import fitz as pymupdf  # type: ignore

from PIL import Image, ImageDraw, ImageFont

# --- 数字版内容：故意混入绝对化 / 恐吓性表述 ---
DIGITAL_LINES = [
    "相法撮要·卷一",
    "",
    "三庭者，自发际至眉为上庭，自眉至鼻底为中庭，自鼻底至颐为下庭。",
    "三庭均等者，一生平顺，行事有度。上庭开阔者，主聪慧，早年得志。",
    "中庭长者，性刚而能任事，然多操劳。下庭丰隆者，晚景必发大财，子孙满堂。",
    "五眼看脸宽，以眼之广狭为准，分面为五。眼距适中者，心正而不偏。",
    "印堂开阔平满者，主贵，仕途通达。印堂有纹者，大凶，主克夫克妻。",
    "鼻者审辨官也，鼻头有肉者必发财，鼻梁低陷者一生贫贱。",
    "眉为保寿官，眉长过目者兄弟众多，眉短者六亲无靠。",
    "面色红润者气足，面青面黑者主病，寿元不长。",
    "女子颧高而露骨者，克夫；男子口大而唇薄者，主口舌是非。",
]

# --- 扫描版内容：繁体、竖排、从右到左（模拟古籍影印页）---
SCAN_COLUMNS = [
    "相法撮要卷二",
    "五眼者以眼廣狹分面為五",
    "眼距適中者心正不偏",
    "印堂開闊者主貴",
    "鼻頭有肉者必發財",
    "眉長過目者兄弟眾多",
    "女子顴高者剋夫",
    "面色紅潤者氣足",
]


def _font_path() -> str:
    for candidate in (
        r"C:\Windows\Fonts\msyh.ttc",       # 微软雅黑（简繁都有）
        r"C:\Windows\Fonts\msjh.ttc",       # 微软正黑（繁体）
        r"C:\Windows\Fonts\simsun.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        "/System/Library/Fonts/PingFang.ttc",           # macOS
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",  # Linux（文泉驿）
    ):
        if Path(candidate).is_file():
            return candidate
    raise SystemExit("找不到中文字体，无法生成竖排测试图（可手动指定字体后重试）")


def make_digital_pdf(path: Path) -> None:
    """有文字层的 PDF：模拟「电子版」古籍。"""
    document = pymupdf.open()
    page = document.new_page(width=595, height=842)   # A4
    y = 70
    for line in DIGITAL_LINES:
        if line:
            page.insert_text((60, y), line, fontname="china-s", fontsize=13)   # 内置简体 CJK 字体
        y += 26
    document.save(path)
    document.close()
    print(f"已生成数字版 PDF：{path}")


def make_scan_pdf(path: Path) -> None:
    """无文字层的 PDF：整页是一张繁体竖排图片，模拟「扫描件 / 影印本」。"""
    width, height = 1000, 1400
    image = Image.new("RGB", (width, height), (250, 246, 238))   # 仿旧纸色
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(_font_path(), 44)

    # 繁体竖排：列从右到左，每列自上而下
    column_x = width - 140
    for column_text in SCAN_COLUMNS:
        y = 120
        for char in column_text:
            draw.text((column_x, y), char, font=font, fill=(30, 26, 22))
            y += 52
        column_x -= 96
        if column_x < 80:
            break

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    document = pymupdf.open()
    page = document.new_page(width=width, height=height)
    page.insert_image(pymupdf.Rect(0, 0, width, height), stream=buffer.getvalue())
    document.save(path)
    document.close()
    print(f"已生成扫描竖排 PDF：{path}（{path.stat().st_size // 1024} KB，无文字层，需要 OCR）")


def main() -> int:
    parser = argparse.ArgumentParser(description="生成用于验证 PDF→知识库 流水线的测试 PDF")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help=f"输出目录，默认 {DEFAULT_OUT}")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    make_digital_pdf(out_dir / "测试-数字版.pdf")
    make_scan_pdf(out_dir / "测试-扫描竖排.pdf")
    print("\n下一步：python scripts/auto_build_kb.py --all")
    return 0


if __name__ == "__main__":
    sys.exit(main())
