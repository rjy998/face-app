"""
引擎体检脚本：打印当前可用的人脸检测引擎，并可选地对一张图片试跑一遍完整特征提取。

用法（在 backend 目录下执行）：
    python scripts/check_detector.py                        # 只看引擎可用性
    python scripts/check_detector.py storage/testdata/test_face.jpg
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from services.face_detector import available_engines, decode_image, detect_face  # noqa: E402
from services.feature_extractor import extract_features  # noqa: E402


def main() -> int:
    print("引擎可用性：", json.dumps(available_engines(), ensure_ascii=False))
    if len(sys.argv) < 2:
        return 0

    image_path = Path(sys.argv[1])
    if not image_path.is_file():
        print(f"图片不存在：{image_path}")
        return 2

    image, warnings = decode_image(image_path.read_bytes())
    detection = detect_face(image)
    print(f"检测结果：engine={detection.engine} faces={detection.faces_found} "
          f"estimated={detection.estimated} bbox={detection.bbox} warnings={detection.warnings + warnings}")
    if not detection.ok:
        print("→ 没有检测到人脸：请换一张真实正面照，或安装 mediapipe")
        return 1

    features = extract_features(image, detection)
    print("特征摘要：")
    for line in features["summary"]:
        print("   -", line)
    print("检索词：", features["search_terms"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
