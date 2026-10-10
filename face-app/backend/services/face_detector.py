"""
人脸检测（第一段 AI）：从图片里找出人脸并给出关键点。

实现优先级（自动降级，保证任何环境都能跑通）：
  1. mediapipe FaceMesh —— 本地运行、免费、隐私好，返回 468(478) 个三维关键点（默认方案）；
  2. OpenCV Haar 级联 —— mediapipe 不可用时的兜底：只给外接框，再按人类学比例推出
     「名义关键点」，此时结果里 estimated=True，解读文案会提示精度下降；
  3. 云 API（百度/腾讯/阿里）—— 预留接口 `_detect_cloud`，填好密钥后启用。

对外统一返回 `FaceDetection`，命名关键点字典，避免下游代码依赖具体引擎。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import settings

logger = logging.getLogger("face_app.detector")

try:  # pragma: no cover
    import numpy as np
except Exception:  # noqa: BLE001
    np = None  # type: ignore

try:  # pragma: no cover
    import cv2
except Exception:  # noqa: BLE001
    cv2 = None  # type: ignore

# MediaPipe FaceMesh 关键点索引 -> 可读名称（只取特征计算需要的点）
FACE_MESH_INDEX: dict[str, int] = {
    "forehead_top": 10,
    "glabella": 9,        # 眉心 / 印堂
    "nose_bridge": 6,
    "nose_tip": 4,
    "subnasale": 2,       # 鼻底
    "chin": 152,
    "cheek_l": 234,       # 颧骨最宽处（图像左）
    "cheek_r": 454,       # 颧骨最宽处（图像右）
    "jaw_l": 172,
    "jaw_r": 397,
    "temple_l": 127,      # 颞部（眼水平的额头两侧）
    "temple_r": 356,
    "eye_l_outer": 33,
    "eye_l_inner": 133,
    "eye_r_inner": 362,
    "eye_r_outer": 263,
    "eye_l_top": 159,
    "eye_l_bottom": 145,
    "eye_r_top": 386,
    "eye_r_bottom": 374,
    "iris_l": 468,        # 需要 refine_landmarks=True
    "iris_r": 473,
    "brow_l_outer": 55,
    "brow_l_inner": 107,
    "brow_r_inner": 336,
    "brow_r_outer": 285,
    "nose_l": 129,
    "nose_r": 358,
    "mouth_l": 61,
    "mouth_r": 291,
    "lip_top": 13,
    "lip_bottom": 14,
}

# 兜底模板：人脸外接框内的相对坐标（竖:三庭近似等分，横:五眼近似等分）
_SYNTHETIC_TEMPLATE: dict[str, tuple[float, float]] = {
    "forehead_top": (0.500, 0.040),
    "glabella": (0.500, 0.360),
    "nose_bridge": (0.500, 0.470),
    "nose_tip": (0.500, 0.580),
    "subnasale": (0.500, 0.660),
    "chin": (0.500, 0.980),
    "cheek_l": (0.070, 0.520),
    "cheek_r": (0.930, 0.520),
    "jaw_l": (0.215, 0.800),
    "jaw_r": (0.785, 0.800),
    "temple_l": (0.055, 0.400),
    "temple_r": (0.945, 0.400),
    "eye_l_outer": (0.165, 0.420),
    "eye_l_inner": (0.420, 0.430),
    "eye_r_inner": (0.580, 0.430),
    "eye_r_outer": (0.835, 0.420),
    "eye_l_top": (0.290, 0.405),
    "eye_l_bottom": (0.290, 0.450),
    "eye_r_top": (0.710, 0.405),
    "eye_r_bottom": (0.710, 0.450),
    "iris_l": (0.290, 0.428),
    "iris_r": (0.710, 0.428),
    "brow_l_outer": (0.150, 0.335),
    "brow_l_inner": (0.430, 0.345),
    "brow_r_inner": (0.570, 0.345),
    "brow_r_outer": (0.850, 0.335),
    "nose_l": (0.415, 0.600),
    "nose_r": (0.585, 0.600),
    "mouth_l": (0.350, 0.765),
    "mouth_r": (0.650, 0.765),
    "lip_top": (0.500, 0.740),
    "lip_bottom": (0.500, 0.790),
}


@dataclass
class FaceDetection:
    """检测结果（引擎无关）。"""

    engine: str
    faces_found: int
    image_size: tuple[int, int]                       # (w, h)
    bbox: tuple[int, int, int, int] | None = None     # (x, y, w, h)
    points: dict[str, tuple[float, float]] = field(default_factory=dict)  # 命名关键点（像素坐标）
    estimated: bool = False                           # True = 未用真实关键点，属比例推算
    confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.faces_found > 0 and self.bbox is not None


# ---------------------------------------------------------------------------
# 引擎可用性探测（供 /api/health 展示）
# ---------------------------------------------------------------------------
_mesh_cache: Any = None
_landmarker_cache: Any = None
_model_failed = False


def _ensure_landmarker_model() -> str | None:
    """
    确保 MediaPipe Tasks API 的 .task 模型存在（只下载一次，缓存在 storage/models/）。

    MediaPipe >= 1.0 移除了旧的 `mp.solutions` API，改用 Tasks API，需要一个模型文件。
    离线环境可以把模型文件放到 FACE_LANDMARKER_MODEL 指定的路径，或设置
    FACE_LANDMARKER_AUTO_DOWNLOAD=false 让它直接走 OpenCV 兜底。
    """
    global _model_failed
    model_path = Path(settings.FACE_LANDMARKER_MODEL)
    if model_path.is_file() and model_path.stat().st_size > 1024:
        return str(model_path)
    if _model_failed or not settings.FACE_LANDMARKER_AUTO_DOWNLOAD:
        return None

    import urllib.request

    try:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("首次使用 MediaPipe Tasks API，正在下载人脸关键点模型：%s", settings.FACE_LANDMARKER_URL)
        urllib.request.urlretrieve(settings.FACE_LANDMARKER_URL, str(model_path))
        if model_path.is_file() and model_path.stat().st_size > 1024:
            logger.info("模型已就绪：%s（%.1f MB）", model_path, model_path.stat().st_size / 1024 / 1024)
            return str(model_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("模型下载失败（%s），将使用 OpenCV 兜底引擎", exc)
    _model_failed = True
    return None


def _get_face_landmarker() -> Any:
    """MediaPipe >= 1.0 的 Tasks API（FaceLandmarker，478 个关键点）。"""
    global _landmarker_cache
    if _landmarker_cache is not None:
        return _landmarker_cache

    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    model_path = _ensure_landmarker_model()
    if not model_path:
        return None

    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
    )
    _landmarker_cache = vision.FaceLandmarker.create_from_options(options)
    return _landmarker_cache


def _get_face_mesh() -> Any:
    """MediaPipe < 1.0 的旧 API（mp.solutions.face_mesh）。"""
    global _mesh_cache
    if _mesh_cache is not None:
        return _mesh_cache
    import mediapipe as mp  # 延迟导入：未安装时才走到这里

    _mesh_cache = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )
    return _mesh_cache


def available_engines() -> dict[str, bool]:
    ok_mp = False
    try:  # pragma: no cover
        import mediapipe  # noqa: F401

        ok_mp = True
    except Exception:  # noqa: BLE001
        ok_mp = False
    return {
        "mediapipe": ok_mp,
        # OpenCV 5.x 移除了 Haar 级联（CascadeClassifier），这里如实反映能力
        "opencv_haar": bool(cv2 is not None and hasattr(cv2, "CascadeClassifier")),
        "opencv": cv2 is not None,
        "numpy": np is not None,
        "cloud_configured": bool(settings.CLOUD_FACE_API_KEY and settings.CLOUD_FACE_SECRET_KEY),
    }


# ---------------------------------------------------------------------------
# 检测实现
# ---------------------------------------------------------------------------
def _points_to_detection(
    points: dict[str, tuple[float, float]], width: int, height: int, engine: str, confidence: float, warnings: list[str]
) -> FaceDetection:
    """命名关键点 → FaceDetection（统一处理外接框）。"""
    xs = [point[0] for point in points.values()]
    ys = [point[1] for point in points.values()]
    x0, y0 = max(0, int(min(xs))), max(0, int(min(ys)))
    x1, y1 = min(width, int(max(xs))), min(height, int(max(ys)))
    return FaceDetection(
        engine=engine,
        faces_found=1,
        image_size=(width, height),
        bbox=(x0, y0, max(1, x1 - x0), max(1, y1 - y0)),
        points=points,
        estimated=False,
        confidence=confidence,
        warnings=warnings,
    )


def _detect_mediapipe(image_bgr: "np.ndarray") -> FaceDetection | None:
    """
    mediapipe 人脸关键点：优先 Tasks API（>=1.0），退回旧 solutions API（<1.0）。
    引擎不可用时返回 None（交给下一级兜底）。
    """
    height, width = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # --- 方案 1：Tasks API（MediaPipe >= 1.0）---
    try:
        landmarker = _get_face_landmarker()
    except Exception as exc:  # noqa: BLE001
        logger.info("MediaPipe Tasks API 初始化失败：%s", exc)
        landmarker = None

    if landmarker is not None:
        try:
            import mediapipe as mp

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)
            faces = getattr(result, "face_landmarks", None)
            if faces:
                landmarks = faces[0]
                points = {
                    name: (float(landmarks[index].x * width), float(landmarks[index].y * height))
                    for name, index in FACE_MESH_INDEX.items()
                    if index < len(landmarks)
                }
                return _points_to_detection(points, width, height, "mediapipe-tasks", 0.95, [])
            return FaceDetection(
                engine="mediapipe-tasks", faces_found=0, image_size=(width, height),
                warnings=["MediaPipe 未检测到人脸"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("MediaPipe Tasks API 推理失败：%s", exc)

    # --- 方案 2：旧 solutions API（MediaPipe < 1.0）---
    try:
        import mediapipe as mp

        if not hasattr(mp, "solutions"):
            return None
        mesh = _get_face_mesh()
    except Exception as exc:  # noqa: BLE001
        logger.info("mediapipe 不可用，将降级：%s", exc)
        return None

    try:
        result = mesh.process(rgb)
    except Exception as exc:  # noqa: BLE001
        logger.warning("mediapipe 推理失败：%s", exc)
        return None

    if not getattr(result, "multi_face_landmarks", None):
        return FaceDetection(
            engine="mediapipe-facemesh", faces_found=0, image_size=(width, height),
            warnings=["MediaPipe 未检测到人脸"],
        )

    landmarks = result.multi_face_landmarks[0].landmark
    points = {
        name: (float(landmarks[index].x * width), float(landmarks[index].y * height))
        for name, index in FACE_MESH_INDEX.items()
        if index < len(landmarks)
    }
    return _points_to_detection(points, width, height, "mediapipe-facemesh", 0.95, [])


def _detect_haar(image_bgr: "np.ndarray") -> FaceDetection | None:
    """OpenCV Haar 级联兜底：只给人脸外接框。OpenCV 5.x 起该 API 已被移除。"""
    if cv2 is None or not hasattr(cv2, "CascadeClassifier"):
        logger.info("当前 OpenCV 版本不含 Haar 级联，跳过兜底引擎（可安装 opencv-python-headless<5 或 mediapipe）")
        return None
    height, width = image_bgr.shape[:2]
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    except Exception as exc:  # noqa: BLE001
        logger.warning("OpenCV 人脸检测失败：%s", exc)
        return None

    if len(faces) == 0:
        return FaceDetection(
            engine="opencv-haar",
            faces_found=0,
            image_size=(width, height),
            warnings=["未检测到人脸（兜底引擎）"],
        )

    # 取面积最大的脸
    x, y, w, h = max(faces, key=lambda box: int(box[2]) * int(box[3]))
    points = {
        name: (x + rx * w, y + ry * h) for name, (rx, ry) in _SYNTHETIC_TEMPLATE.items()
    }
    return FaceDetection(
        engine="opencv-haar",
        faces_found=1,
        image_size=(width, height),
        bbox=(int(x), int(y), int(w), int(h)),
        points=points,
        estimated=True,
        confidence=0.6,
        warnings=["未安装 mediapipe，已用 OpenCV 兜底并按人类学比例推算关键点，比例结论仅供参考"],
    )


def _detect_cloud(image_bgr: "np.ndarray") -> FaceDetection | None:
    """
    云 API 预留实现（百度/腾讯/阿里）。

    启用步骤：
      1. 在 .env 里设置 FACE_DETECTOR=cloud 及 CLOUD_FACE_API_KEY / CLOUD_FACE_SECRET_KEY；
      2. 在此处按厂商 SDK 文档请求 `人脸检测` 接口，把返回的关键点映射成上面的命名键；
      3. 其余流程（特征提取 / RAG / 解读）无需任何改动。
    """
    if not (settings.CLOUD_FACE_API_KEY and settings.CLOUD_FACE_SECRET_KEY):
        return None
    logger.warning("FACE_DETECTOR=cloud 但云 API 尚未实现，自动降级到本地引擎")
    return None


def detect_face(image_bgr: "np.ndarray") -> FaceDetection:
    """
    统一入口：按配置与可用性选择引擎，逐级降级。

    策略：依次尝试「配置的引擎 → mediapipe → opencv 兜底」；
    第一个成功检出人脸的引擎胜出；若全都检出不了，则返回最后一次的结果（faces_found=0），
    由上层告知用户「请上传正面清晰照片」。
    """
    if np is None or cv2 is None:
        raise RuntimeError("缺少 numpy/opencv 依赖，无法进行人脸检测，请先 pip install -r requirements.txt")

    order = [settings.FACE_DETECTOR, "mediapipe", "opencv"]
    tried: set[str] = set()
    last: FaceDetection | None = None
    attempted = 0

    for name in order:
        if name in tried:
            continue
        tried.add(name)
        if name == "cloud":
            result = _detect_cloud(image_bgr)
        elif name == "mediapipe":
            result = _detect_mediapipe(image_bgr)
        else:
            result = _detect_haar(image_bgr)
        if result is None:
            continue

        attempted += 1
        if result.ok:
            if attempted > 1:
                result.warnings.append("主检测引擎未检出人脸，本次由兜底引擎完成，精度略有下降")
            return result
        last = result

    if last is not None:
        return last
    raise RuntimeError(
        "没有可用的人脸检测引擎：请安装 mediapipe（推荐 Python 3.11/3.12），"
        "或使用 opencv-python-headless<5（自带 Haar 级联兜底）"
    )


def decode_image(data: bytes) -> tuple["np.ndarray", list[str]]:
    """把上传的字节流解码成 BGR 图像，返回 (图像, 警告列表)。"""
    if np is None or cv2 is None:
        raise RuntimeError("缺少 numpy/opencv 依赖，无法解析图片")
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    warnings: list[str] = []
    if image is None:
        raise ValueError("图片无法解码，请上传有效的 JPG/PNG/WebP 图片")
    height, width = image.shape[:2]
    if min(width, height) < settings.MIN_IMAGE_SIDE:
        raise ValueError(f"图片分辨率过低（{width}x{height}），请上传边长 ≥ {settings.MIN_IMAGE_SIDE}px 的清晰照片")
    if max(width, height) > 4096:  # 控制算力开销
        scale = 4096 / max(width, height)
        image = cv2.resize(image, (int(width * scale), int(height * scale)))
        warnings.append("图片过大已自动缩放，建议上传 2000px 以内的照片")
    return image, warnings
