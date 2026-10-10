"""
生成一张「合成测试人脸」图片，用于本地自测 / 联调，不涉及任何真人照片。

用法：
    python scripts/make_test_face.py                      # 输出 storage/testdata/test_face.jpg
    python scripts/make_test_face.py --out D:/tmp/a.jpg

说明：合成图只是给检测器一个「有五官、明暗关系接近照片」的输入，
用于验证整条链路（解码 → 检测 → 特征 → RAG → 解读 → 落库）是否通畅。
合成图不追求被真实人脸模型精确识别；若检测不到，请改用一张真实的正面照片做自测。
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

BACKEND_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUT = BACKEND_DIR / "storage" / "testdata" / "test_face.jpg"

WIDTH, HEIGHT = 640, 800


def _vignette(image: Image.Image) -> Image.Image:
    """加一层暗角，让图像更接近真实照片的照度分布。"""
    mask = Image.new("L", (WIDTH, HEIGHT), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((-160, -200, WIDTH + 160, HEIGHT + 200), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(120))
    dark = Image.new("RGB", (WIDTH, HEIGHT), (30, 30, 40))
    return Image.composite(image, dark, mask)


def build_face() -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), (176, 186, 198))
    draw = ImageDraw.Draw(image, "RGBA")

    cx = WIDTH // 2
    face_top, face_bottom = 150, 700
    face_left, face_right = 195, 445

    # 脖子与肩
    draw.rectangle((cx - 80, face_bottom - 60, cx + 80, HEIGHT), fill=(226, 186, 158))
    draw.ellipse((-120, HEIGHT - 180, WIDTH + 120, HEIGHT + 260), fill=(88, 96, 120))

    # 脸型：上宽下略窄的椭圆
    draw.ellipse((face_left, face_top, face_right, face_bottom), fill=(238, 202, 174))
    # 颧骨与下颌阴影，增加立体感
    draw.ellipse((face_left + 8, face_top + 190, face_left + 90, face_top + 330), fill=(222, 180, 152, 120))
    draw.ellipse((face_right - 90, face_top + 190, face_right - 8, face_top + 330), fill=(222, 180, 152, 120))
    draw.ellipse((cx - 120, face_bottom - 200, cx + 120, face_bottom + 10), fill=(228, 188, 160, 110))

    # 头发
    draw.ellipse((face_left - 26, face_top - 70, face_right + 26, face_top + 150), fill=(48, 40, 42))
    draw.rectangle((face_left - 26, face_top - 20, face_left + 6, face_top + 180), fill=(48, 40, 42))
    draw.rectangle((face_right - 6, face_top - 20, face_right + 26, face_top + 180), fill=(48, 40, 42))

    eye_y = face_top + 195
    left_eye_x, right_eye_x = cx - 72, cx + 72

    # 眉毛
    for x in (left_eye_x, right_eye_x):
        draw.polygon(
            [
                (x - 58, eye_y - 52),
                (x + 58, eye_y - 62),
                (x + 58, eye_y - 48),
                (x - 58, eye_y - 38),
            ],
            fill=(66, 52, 48),
        )

    # 眼睛：眼白 + 虹膜 + 瞳孔 + 上眼睑阴影
    for x in (left_eye_x, right_eye_x):
        draw.ellipse((x - 52, eye_y - 26, x + 52, eye_y + 24), fill=(250, 250, 252))
        draw.ellipse((x - 20, eye_y - 22, x + 20, eye_y + 18), fill=(96, 68, 48))
        draw.ellipse((x - 10, eye_y - 12, x + 10, eye_y + 8), fill=(18, 16, 18))
        draw.arc((x - 54, eye_y - 34, x + 54, eye_y + 22), start=185, end=355, fill=(40, 32, 34), width=7)
        draw.arc((x - 54, eye_y - 22, x + 54, eye_y + 30), start=5, end=175, fill=(150, 120, 110), width=3)

    # 鼻：鼻梁高光 + 鼻翼阴影 + 鼻头
    draw.polygon(
        [(cx - 12, eye_y + 40), (cx + 12, eye_y + 40), (cx + 26, eye_y + 150), (cx - 26, eye_y + 150)],
        fill=(230, 190, 162, 150),
    )
    draw.polygon(
        [(cx - 6, eye_y + 150), (cx + 6, eye_y + 150), (cx + 22, eye_y + 176), (cx - 22, eye_y + 176)],
        fill=(206, 162, 136, 170),
    )
    draw.ellipse((cx - 40, eye_y + 166, cx - 14, eye_y + 186), fill=(60, 44, 40, 150))
    draw.ellipse((cx + 14, eye_y + 166, cx + 40, eye_y + 186), fill=(60, 44, 40, 150))

    # 嘴唇
    mouth_y = eye_y + 250
    draw.polygon(
        [
            (cx - 78, mouth_y),
            (cx - 30, mouth_y - 22),
            (cx, mouth_y - 10),
            (cx + 30, mouth_y - 22),
            (cx + 78, mouth_y),
            (cx + 30, mouth_y + 16),
            (cx - 30, mouth_y + 16),
        ],
        fill=(186, 92, 92),
    )
    draw.line([(cx - 76, mouth_y), (cx + 76, mouth_y)], fill=(140, 62, 62), width=3)

    # 脸颊血色
    draw.ellipse((left_eye_x - 60, mouth_y - 130, left_eye_x + 30, mouth_y - 40), fill=(226, 132, 116, 60))
    draw.ellipse((right_eye_x - 30, mouth_y - 130, right_eye_x + 60, mouth_y - 40), fill=(226, 132, 116, 60))

    # 轻微模糊 + 噪声 + 暗角：模拟真实拍摄质感
    image = image.filter(ImageFilter.GaussianBlur(1.1))
    image = _vignette(image)

    pixels = image.load()
    rng = random.Random(20240521)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            r, g, b = pixels[x, y]
            noise = rng.randint(-9, 9)
            pixels[x, y] = (
                max(0, min(255, r + noise)),
                max(0, min(255, g + noise)),
                max(0, min(255, b + noise)),
            )
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description="生成合成测试人脸图")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="输出路径")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    build_face().save(out_path, quality=92)
    print(f"已生成合成测试人脸：{out_path}（{out_path.stat().st_size // 1024} KB，{WIDTH}x{HEIGHT}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
