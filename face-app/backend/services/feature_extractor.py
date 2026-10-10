"""
特征提取（第一段 AI 的后半段）：把关键点换算成「三庭五眼 / 五官 / 脸型 / 气色」结构化特征。

设计要点：
- 全部指标都是**比例**（除以脸宽或脸高），因此与照片分辨率、拍摄距离无关；
- 每个指标同时输出 `value`（数值，便于排查）和 `label`（中文结论，便于喂给 RAG 与大模型）；
- 气色模块基于像素统计，会明确标注「受光线/妆容影响」，避免误导；
- 输出末尾的 `search_terms` 专门用于构造知识库检索词。
"""

from __future__ import annotations

import logging
import math
from typing import Any

try:  # pragma: no cover
    import numpy as np
except Exception:  # noqa: BLE001
    np = None  # type: ignore

try:  # pragma: no cover
    import cv2
except Exception:  # noqa: BLE001
    cv2 = None  # type: ignore

from services.face_detector import FaceDetection

logger = logging.getLogger("face_app.features")


# ---------------------------------------------------------------------------
# 基础几何工具
# ---------------------------------------------------------------------------
def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _dx(a: tuple[float, float], b: tuple[float, float]) -> float:
    return abs(a[0] - b[0])


def _dy(a: tuple[float, float], b: tuple[float, float]) -> float:
    return abs(a[1] - b[1])


def _pct(value: float, digits: int = 3) -> float:
    return round(float(value), digits)


def _pick(level: float, bins: list[tuple[float, str]], default: str) -> str:
    """按阈值区间给出中文标签（bins 需按阈值升序）。"""
    for threshold, label in bins:
        if level <= threshold:
            return label
    return default


# ---------------------------------------------------------------------------
# 姿态归一化与测量质量（比例测量最容易出问题的地方）
# ---------------------------------------------------------------------------
def _normalize_roll(points: dict[str, tuple[float, float]]) -> tuple[dict[str, tuple[float, float]], float]:
    """
    把关键点整体旋转，使两眼连线水平（消除头部侧倾/roll）。

    不归一化时，所有「水平距离」（眼宽、眼距、脸宽）都会随头歪而失真——
    这是三庭五眼测量最常见的偏差来源之一。
    """
    left = points.get("iris_l") or points.get("eye_l_outer")
    right = points.get("iris_r") or points.get("eye_r_outer")
    if not left or not right:
        return points, 0.0

    dx = right[0] - left[0]
    dy = right[1] - left[1]
    if abs(dx) < 1e-6:
        return points, 0.0

    angle = math.atan2(dy, dx)
    if abs(angle) < math.radians(1.0):        # 基本水平就不折腾
        return points, math.degrees(angle)

    cos_a, sin_a = math.cos(-angle), math.sin(-angle)
    center_x = (left[0] + right[0]) / 2.0
    center_y = (left[1] + right[1]) / 2.0
    rotated: dict[str, tuple[float, float]] = {}
    for name, (x, y) in points.items():
        px, py = x - center_x, y - center_y
        rotated[name] = (px * cos_a - py * sin_a, px * sin_a + py * cos_a)
    return rotated, math.degrees(angle)


def _face_width_at_y(points: dict[str, tuple[float, float]], y: float) -> float | None:
    """
    用面部轮廓点在指定高度上插值出脸宽。

    五眼要的是「眼水平线处的脸宽」，而关键点里没有正好落在该高度的轮廓点，
    所以沿左/右两条轮廓链（颞点→颧点→下颌点）做线性插值。
    """
    left_chain = ("temple_l", "cheek_l", "jaw_l")
    right_chain = ("temple_r", "cheek_r", "jaw_r")

    def interp(chain: tuple[str, ...]) -> float | None:
        nodes = sorted((points[name][1], points[name][0]) for name in chain if name in points)
        if not nodes:
            return None
        if y <= nodes[0][0]:
            return nodes[0][1]
        if y >= nodes[-1][0]:
            return nodes[-1][1]
        for (y0, x0), (y1, x1) in zip(nodes, nodes[1:]):
            if y0 <= y <= y1:
                ratio = (y - y0) / max(y1 - y0, 1e-6)
                return x0 + ratio * (x1 - x0)
        return nodes[-1][1]

    left_x, right_x = interp(left_chain), interp(right_chain)
    if left_x is None or right_x is None:
        return None
    return abs(right_x - left_x)


def _pose_quality(points: dict[str, tuple[float, float]], roll_degrees: float) -> tuple[dict[str, Any], list[str]]:
    """
    估计拍摄质量：侧脸程度（yaw）、头部侧倾（roll）。

    正脸照的比例才有意义；侧脸照要如实告诉用户「这次测量不可靠」，
    而不是继续输出一个看似精确的数字。
    """
    warnings: list[str] = []
    midline_x = sum(points[key][0] for key in ("forehead_top", "glabella", "nose_tip", "chin") if key in points) / 4.0
    left_eye = points.get("eye_l_outer")
    right_eye = points.get("eye_r_outer")
    yaw = 0.0
    if left_eye and right_eye:
        gap_left = abs(left_eye[0] - midline_x)
        gap_right = abs(right_eye[0] - midline_x)
        if gap_left + gap_right > 1e-6:
            yaw = abs(gap_left - gap_right) / (gap_left + gap_right)

    if yaw > 0.12:
        warnings.append(
            f"照片偏侧（侧脸角度约 {yaw * 100:.0f}%），三庭五眼比例会有明显偏差，建议正对镜头重拍"
        )
    if abs(roll_degrees) > 8:
        warnings.append(f"头部侧倾约 {abs(roll_degrees):.0f}°，已自动校正水平，比例仍可能有轻微误差")

    return {"yaw_ratio": _pct(yaw), "roll_degrees": _pct(roll_degrees, 1)}, warnings


def _estimate_hairline(
    image: "np.ndarray",
    points: dict[str, tuple[float, float]],
    face_height: float,
    face_width: float,
) -> tuple[float | None, float]:
    """
    在图片像素里找发际线（额头皮肤与头发的交界）。

    关键点里没有发际线：索引 10 只是"模型认为的额头最高处"，被刘海遮挡时会落在额头中间，
    导致上庭系统性偏小（实测只有 19%，而三庭应接近 1:1:1）。
    这里改为在面部中线上从眉心往上扫：先取眉心上方一小段作为肤色基准，
    再逐行判断是否变成头发（明显更暗、或暗且低饱和、或很亮的白发），
    取连续 3 行为头发的第一行作为发际线。

    返回 (发际线 y 像素, 置信度)；找不到返回 (None, 0)。
    """
    if image is None or np is None or cv2 is None:
        return None, 0.0

    height, width = image.shape[:2]
    glabella = points.get("glabella")
    if glabella is None:
        return None, 0.0

    x_center = int(round(glabella[0]))
    half = max(3, int(round(face_width * 0.07)))
    x0, x1 = max(0, x_center - half), min(width, x_center + half + 1)
    if x1 - x0 < 4:
        return None, 0.0

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    def band_value(y: int) -> float:
        return float(np.median(hsv[y, x0:x1], axis=0)[2]) if 0 <= y < height else 0.0

    # 参考肤色：在眉心上方取几段候选带，选**最亮**的一段当肤色基准。
    # 为什么不只用眉心正上方：若额头被刘海遮住，那一段本身就是头发（很暗），
    # 拿头发当基准会导致后面把所有头发都判成"皮肤"而找不到发际线。
    candidates = [
        np.median(hsv[y, x0:x1], axis=0)
        for y in range(int(glabella[1] - face_height * 0.12), int(glabella[1] - face_height * 0.01))
        if 0 <= y < height
    ]
    if not candidates:
        return None, 0.0
    reference = max(candidates, key=lambda item: float(item[2]))
    ref_v, ref_s = float(reference[2]), float(reference[1])
    if ref_v < 90:
        # 连最亮的一段都很暗：说明额头被头发/帽子完全遮住，或整体曝光过低
        return None, 0.0

    y_start = int(glabella[1] - face_height * 0.08)
    y_end = max(0, int(glabella[1] - face_height * 1.1))

    def scan(darken: float, need_gradient: bool) -> float | None:
        """从眉心往上扫，返回发际线 y；darken 越小越严格，need_gradient 要求该处有明显亮度跳变。"""
        consecutive = 0
        for y in range(y_start, y_end - 1, -1):
            row = np.median(hsv[y, x0:x1], axis=0)
            value, saturation = float(row[2]), float(row[1])
            is_hair = (
                value < ref_v * darken                                  # 更暗（黑/棕发）
                or (saturation < ref_s * 0.55 and value < ref_v * 0.85)  # 暗且低饱和（灰发）
                or (value > ref_v * 1.25 and saturation < ref_s * 0.6)   # 明显更亮且低饱和（白发）
            )
            if is_hair:
                consecutive += 1
                if consecutive >= 3:
                    candidate = float(y + 2)
                    if need_gradient:
                        below = band_value(int(candidate) + 4)
                        above = band_value(int(candidate) - 4)
                        if abs(above - below) < ref_v * 0.10:
                            consecutive = 0      # 只是渐暗（阴影/鬓角），不是发际线
                            continue
                    return candidate
            else:
                consecutive = 0
        return None

    # 先严格（深色发际线），没找到再放宽并要求有明显跳变（浅色/棕色头发）
    hairline_y = scan(0.62, False) or scan(0.80, True)

    if hairline_y is None:
        return None, 0.0

    hair_row = np.median(hsv[int(hairline_y), x0:x1], axis=0)
    contrast = abs(ref_v - float(hair_row[2])) / max(ref_v, 1.0)
    confidence = max(0.35, min(0.95, contrast * 1.6))
    return hairline_y, confidence


# ---------------------------------------------------------------------------
# 三庭
# ---------------------------------------------------------------------------
def _three_courts(
    points: dict[str, tuple[float, float]],
    face_height: float,
    hairline_y: float | None = None,
    hairline_confidence: float = 0.0,
) -> dict[str, Any]:
    """
    三庭 = 上庭 : 中庭 : 下庭，理想状态接近 1:1:1。

    上庭起点优先用**像素检测到的发际线**（hairline_y）；检测不到才退回关键点 10 的近似值。
    """
    top_y = hairline_y if hairline_y is not None else points["forehead_top"][1]
    # 上庭/中庭的分界取「眉线」：相学上是"自发际至眉为上庭，自眉至鼻底为中庭"，
    # 所以用内眉点的 y 而不是关键点 9（眉心在眉上方，会把上庭算小、中庭算大）。
    brow_ys = [
        points[name][1]
        for name in ("brow_l_inner", "brow_r_inner")
        if name in points
    ]
    boundary_y = (sum(brow_ys) / len(brow_ys)) if brow_ys else points["glabella"][1]

    upper = abs(boundary_y - top_y)
    middle = abs(points["subnasale"][1] - boundary_y)
    lower = _dy(points["subnasale"], points["chin"])
    total = max(upper + middle + lower, 1e-6)
    parts = {"upper": upper / total, "middle": middle / total, "lower": lower / total}

    spread = max(parts.values()) - min(parts.values())
    # 「测到了」才算数：头发遮住额头或检测失败时，上庭只能给近似值，不能拿它下判断
    upper_measured = hairline_y is not None and 0.18 <= parts["upper"] <= 0.52

    names = {"upper": "上庭（额部）", "middle": "中庭（眉眼鼻）", "lower": "下庭（口唇下巴）"}

    if not upper_measured:
        measured_parts = {"middle": parts["middle"], "lower": parts["lower"]}
        longest = max(measured_parts, key=lambda key: measured_parts[key])
        judgement = (
            f"上庭未测得（发际线不可见或额头被头发遮挡）；就中庭与下庭相比，{names[longest]}相对偏长"
        )
        balance = "上庭未测得"
    elif spread <= 0.03:
        judgement = "三庭比例较为均衡"
        balance = "均衡"
    else:
        longest = max(parts, key=lambda key: parts[key])
        judgement = f"{names[longest]}相对偏长，其余两庭略短"
        balance = {"upper": "上庭偏长", "middle": "中庭偏长", "lower": "下庭偏长"}[longest]

    def _label(ratio: float, measured: bool = True) -> str:
        if not measured:
            return "未测得"
        if ratio >= 0.37:
            return "偏长"
        if ratio <= 0.29:
            return "偏短"
        return "适中"

    return {
        "upper": {
            "ratio": _pct(parts["upper"]),
            "label": _label(parts["upper"], upper_measured),
            "measured": upper_measured,
        },
        "middle": {"ratio": _pct(parts["middle"]), "label": _label(parts["middle"]), "measured": True},
        "lower": {"ratio": _pct(parts["lower"]), "label": _label(parts["lower"]), "measured": True},
        "balance": balance,
        "judgement": judgement,
        "hairline": {
            "source": "像素检测" if hairline_y is not None else "未检出（改用关键点近似，仅供参考）",
            "confidence": _pct(hairline_confidence, 2),
        },
        "ideal": "1:1:1",
        "note": "上庭起点取像素检测到的发际线；未检出时上庭标记为未测得，解读不对上庭下判断",
        "reliable": upper_measured,
    }


# ---------------------------------------------------------------------------
# 五眼
# ---------------------------------------------------------------------------
def _five_eyes(points: dict[str, tuple[float, float]]) -> dict[str, Any]:
    """
    五眼：以**眼水平线处的脸宽**为 5 等分，衡量眼宽与眼距。

    脸宽不是取颞点距离（那是额部轮廓，高于眼水平线），而是沿轮廓链插值到眼睛高度，
    更接近传统五眼的量法。
    """
    eye_y = (points["eye_l_outer"][1] + points["eye_r_outer"][1]) / 2.0
    interpolated = _face_width_at_y(points, eye_y)
    temple_width = _dx(points["temple_l"], points["temple_r"])
    face_width_eye = interpolated or temple_width
    face_width_eye = max(face_width_eye, 1e-6)
    width_source = "轮廓插值" if interpolated else "颞点距离"
    unit = face_width_eye / 5.0
    eye_a = _dx(points["eye_l_outer"], points["eye_l_inner"])
    eye_b = _dx(points["eye_r_outer"], points["eye_r_inner"])
    eye_width = (eye_a + eye_b) / 2.0
    inner_gap = _dx(points["eye_l_inner"], points["eye_r_inner"])
    outer_l = _dx(points["temple_l"], points["eye_l_outer"])
    outer_r = _dx(points["eye_r_outer"], points["temple_r"])
    outer_avg = (outer_l + outer_r) / 2.0

    def _label(ratio: float, wide: float = 1.15, narrow: float = 0.85, mid: str = "适中") -> str:
        if ratio > wide:
            return "偏宽"
        if ratio < narrow:
            return "偏窄"
        return mid

    gap_ratio = inner_gap / unit
    eye_ratio = eye_width / unit
    gap_label = _label(gap_ratio)
    return {
        "face_width_at_eye": _pct(face_width_eye, 1),
        "width_source": width_source,
        "eye_width_ratio": {"value": _pct(eye_ratio), "label": _label(eye_ratio, 1.12, 0.88)},
        "inner_gap_ratio": {"value": _pct(gap_ratio), "label": gap_label},
        "outer_gap_ratio": {"value": _pct(outer_avg / unit), "label": _label(outer_avg / unit, 1.2, 0.8)},
        "judgement": f"眼距{gap_label}，眼宽{'舒展' if eye_ratio >= 1.0 else '偏秀气'}，整体接近五眼比例"
        if 0.8 <= gap_ratio <= 1.2
        else f"眼距{gap_label}（约为标准眼宽的 {gap_ratio:.2f} 倍）",
        "note": "以眼水平线的脸宽为基准；正视镜头、不戴粗框眼镜时最准，侧脸会失真",
    }


# ---------------------------------------------------------------------------
# 脸型
# ---------------------------------------------------------------------------
def _face_shape(points: dict[str, tuple[float, float]], face_height: float) -> dict[str, Any]:
    """脸型 = 长宽比 + 下颌/颧骨/额宽三个宽度比。"""
    cheek_w = max(_dx(points["cheek_l"], points["cheek_r"]), 1e-6)
    jaw_w = max(_dx(points["jaw_l"], points["jaw_r"]), 1e-6)
    forehead_w = max(_dx(points["temple_l"], points["temple_r"]), 1e-6)
    length = max(_dy(points["forehead_top"], points["chin"]), 1e-6)

    length_ratio = length / cheek_w
    jaw_ratio = jaw_w / cheek_w
    forehead_ratio = forehead_w / cheek_w

    if length_ratio >= 1.6:
        shape, reason = "长形脸", "脸长明显大于脸宽"
    elif forehead_ratio >= 1.02 and jaw_ratio <= 0.78:
        shape, reason = "心形脸（倒三角）", "额部偏宽、下颌收窄"
    elif jaw_ratio <= 0.74 and length_ratio >= 1.3:
        shape, reason = "瓜子脸", "下颌线条收窄，下巴偏尖"
    elif jaw_ratio >= 0.9 and length_ratio <= 1.45:
        shape, reason = "方形脸", "下颌角较宽、脸长适中"
    elif jaw_ratio >= 0.82 and length_ratio <= 1.4:
        shape, reason = "圆形脸", "脸长与脸宽接近，下颌圆润"
    elif 1.3 <= length_ratio < 1.6:
        shape, reason = "鹅蛋脸", "长宽比例适中，轮廓过渡柔和"
    else:
        shape, reason = "椭圆偏方脸", "宽度与长度均在中间区间"

    confidence = 0.55 if not points else 0.8
    return {
        "label": shape,
        "reason": reason,
        "length_ratio": _pct(length_ratio),
        "jaw_cheek_ratio": _pct(jaw_ratio),
        "forehead_cheek_ratio": _pct(forehead_ratio),
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# 五官
# ---------------------------------------------------------------------------
def _facial_features(
    points: dict[str, tuple[float, float]],
    courts: dict[str, Any],
    cheek_w: float,
) -> dict[str, Any]:
    # 眉：长度、与眼的距离、（若有像素信息则在气色模块补充浓淡）
    brow_len = (
        _dx(points["brow_l_outer"], points["brow_l_inner"]) + _dx(points["brow_r_inner"], points["brow_r_outer"])
    ) / 2.0
    brow_eye_gap = (
        _dy(points["brow_l_inner"], points["eye_l_top"]) + _dy(points["brow_r_inner"], points["eye_r_top"])
    ) / 2.0
    brow_span = _dx(points["brow_l_outer"], points["brow_r_outer"])

    # 眼：眼裂长宽比（偏圆 / 细长）
    eye_w = (_dx(points["eye_l_outer"], points["eye_l_inner"]) + _dx(points["eye_r_outer"], points["eye_r_inner"])) / 2
    eye_h = (_dy(points["eye_l_top"], points["eye_l_bottom"]) + _dy(points["eye_r_top"], points["eye_r_bottom"])) / 2
    eye_aspect = eye_h / max(eye_w, 1e-6)

    # 鼻：鼻宽（鼻翼）与鼻长
    nose_w = _dx(points["nose_l"], points["nose_r"])
    nose_len = _dy(points["nose_bridge"], points["subnasale"])

    # 口唇：宽度与厚度
    mouth_w = _dx(points["mouth_l"], points["mouth_r"])
    lip_thick = _dy(points["lip_top"], points["lip_bottom"])

    return {
        "brow": {
            "length_ratio": _pct(brow_len / max(cheek_w, 1e-6)),
            "span_ratio": _pct(brow_span / max(cheek_w, 1e-6)),
            "eye_gap_ratio": _pct(brow_eye_gap / max(cheek_w, 1e-6)),
            "label": "眉眼距离适中" if 0.03 <= brow_eye_gap / max(cheek_w, 1e-6) <= 0.07 else "眉眼距离偏"
            + ("开阔" if brow_eye_gap / max(cheek_w, 1e-6) > 0.07 else "紧凑"),
            "span_label": _pick(brow_span / max(cheek_w, 1e-6), [(0.66, "眉形偏收敛")], "眉形舒展"),
        },
        "eye": {
            "aspect_ratio": _pct(eye_aspect),
            "label": "眼型偏圆润" if eye_aspect >= 0.34 else ("眼型清秀偏细长" if eye_aspect <= 0.26 else "眼型适中"),
        },
        "nose": {
            "width_ratio": _pct(nose_w / max(cheek_w, 1e-6)),
            "length_ratio": _pct(nose_len / max(cheek_w, 1e-6)),
            "label": _pick(
                nose_w / max(cheek_w, 1e-6),
                [(0.24, "鼻翼偏秀气"), (0.30, "鼻宽适中")],
                "鼻翼偏宽厚",
            ),
        },
        "mouth": {
            "width_ratio": _pct(mouth_w / max(cheek_w, 1e-6)),
            "thickness_ratio": _pct(lip_thick / max(mouth_w, 1e-6)),
            "label": _pick(
                lip_thick / max(mouth_w, 1e-6),
                [(0.18, "唇形偏薄"), (0.30, "唇厚适中")],
                "唇形偏丰润",
            ),
        },
        "forehead": _forehead_block(points, courts, cheek_w),
    }


def _forehead_block(
    points: dict[str, tuple[float, float]], courts: dict[str, Any], cheek_w: float
) -> dict[str, Any]:
    """
    额部：**宽窄看横向宽度，高低看纵向比例**，两者不能混。

    之前这里错用上庭的"高度比例"去判定"额部偏窄"，导致上庭测得偏短的人被说成"额头窄"；
    而且宽度用的是眉间距（brow_span），不是额部轮廓宽度。现在改为：
        额宽 / 颧宽（取额部轮廓左右两点，缺失时用眉弓外端兜底）
    """
    forehead_w = _dx(points["temple_l"], points["temple_r"]) if ("temple_l" in points and "temple_r" in points) else 0.0
    width_source = "额部轮廓"
    if forehead_w <= 0:
        forehead_w = _dx(points["brow_l_outer"], points["brow_r_outer"]) if (
            "brow_l_outer" in points and "brow_r_outer" in points
        ) else 0.0
        width_source = "眉弓兜底"
    width_ratio = forehead_w / max(cheek_w, 1e-6)

    if width_ratio >= 1.02:
        width_label = "额部开阔"
    elif width_ratio >= 0.93:
        width_label = "额宽适中"
    else:
        width_label = "额部偏窄"

    upper = courts.get("upper", {}) or {}
    measured = bool(upper.get("measured", True))
    height_label = upper.get("label", "未测得") if measured else "未测得"

    return {
        "width_ratio": _pct(width_ratio),
        "width_source": width_source,
        "width_label": width_label,
        "height_ratio": upper.get("ratio"),
        "height_label": height_label,
        "label": width_label,          # 兼容旧字段：额部宽窄看宽度
        "note": "宽窄按「额宽/颧宽」判断，高低按上庭比例判断，二者互不代表",
    }


# ---------------------------------------------------------------------------
# 气色（像素统计）
# ---------------------------------------------------------------------------
def _roi_stats(image: "np.ndarray", points: dict[str, tuple[float, float]], center: str) -> dict[str, float] | None:
    """在某个关键点周围取小方块，返回 BGR 均值等统计量。"""
    if np is None or cv2 is None:
        return None
    height, width = image.shape[:2]
    cx, cy = points[center]
    scale = _dist(points["cheek_l"], points["cheek_r"]) * 0.13
    half = max(4, int(scale))
    x0, x1 = max(0, int(cx - half)), min(width, int(cx + half))
    y0, y1 = max(0, int(cy - half)), min(height, int(cy + half))
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    patch = image[y0:y1, x0:x1].astype("float32")
    mean_bgr = patch.reshape(-1, 3).mean(axis=0)
    gray = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype("float32")
    return {
        "b": float(mean_bgr[0]),
        "g": float(mean_bgr[1]),
        "r": float(mean_bgr[2]),
        "gray_mean": float(gray.mean()),
        "gray_std": float(gray.std()),
    }


def _complexion(image: "np.ndarray", points: dict[str, tuple[float, float]], bbox: tuple[int, int, int, int]) -> dict[str, Any]:
    """气色：基于双颊 + 额部的颜色统计（明确标注受拍摄条件影响）。"""
    caveat = "气色受光线、白平衡、妆容与摄像头影响很大，仅作画面观感参考，不反映健康状况"
    if np is None or cv2 is None or image is None:
        return {"available": False, "label": "未采集", "caveat": caveat}

    samples = [
        stats
        for stats in (
            _roi_stats(image, points, "cheek_l"),
            _roi_stats(image, points, "cheek_r"),
            _roi_stats(image, points, "glabella"),
        )
        if stats
    ]
    if not samples:
        return {"available": False, "label": "未采集", "caveat": caveat}

    mean_r = sum(item["r"] for item in samples) / len(samples)
    mean_g = sum(item["g"] for item in samples) / len(samples)
    mean_b = sum(item["b"] for item in samples) / len(samples)
    brightness = (mean_r + mean_g + mean_b) / 3.0
    redness = mean_r - (mean_g + mean_b) / 2.0
    yellowness = (mean_r + mean_g) / 2.0 - mean_b
    gray_means = [item["gray_mean"] for item in samples]
    evenness = 1.0 - (max(gray_means) - min(gray_means)) / max(sum(gray_means) / len(gray_means), 1.0)

    if brightness >= 205:
        label, tone = "画面整体偏亮，肤色显白皙（可能曝光偏高）", "偏亮"
    elif brightness <= 90:
        label, tone = "画面偏暗，气色观感受光线影响", "偏暗"
    elif redness >= 16:
        label, tone = "面颊血色较明显，观感红润有神", "红润"
    elif yellowness >= 32 and brightness < 165:
        label, tone = "肤色观感偏暖偏黄，可能与光线色温有关", "偏暖黄"
    elif redness <= 6 and brightness >= 165:
        label, tone = "肤色观感偏清透偏白", "偏清透"
    else:
        label, tone = "肤色观感自然平和，光泽度适中", "自然"

    return {
        "available": True,
        "brightness": _pct(brightness, 1),
        "redness": _pct(redness, 1),
        "yellowness": _pct(yellowness, 1),
        "evenness": _pct(max(0.0, min(1.0, evenness))),
        "tone": tone,
        "label": label,
        "caveat": caveat,
    }


# ---------------------------------------------------------------------------
# 对称度
# ---------------------------------------------------------------------------
def _symmetry(points: dict[str, tuple[float, float]], cheek_w: float) -> dict[str, Any]:
    midline_x = sum(points[key][0] for key in ("forehead_top", "glabella", "nose_tip", "chin")) / 4.0
    pairs = [
        ("eye_l_outer", "eye_r_outer"),
        ("cheek_l", "cheek_r"),
        ("jaw_l", "jaw_r"),
        ("mouth_l", "mouth_r"),
        ("brow_l_outer", "brow_r_outer"),
    ]
    diffs = [abs(abs(points[a][0] - midline_x) - abs(points[b][0] - midline_x)) for a, b in pairs]
    score = 1.0 - (sum(diffs) / len(diffs)) / max(cheek_w, 1e-6)
    score = max(0.0, min(1.0, score))
    if score >= 0.94:
        label = "左右对称度很好"
    elif score >= 0.88:
        label = "左右对称度良好"
    else:
        label = "左右略有差异（多数人本就如此）"
    return {"score": _pct(score), "label": label}


# ---------------------------------------------------------------------------
# 对外主函数
# ---------------------------------------------------------------------------
def extract_features(image: "np.ndarray", detection: FaceDetection) -> dict[str, Any]:
    """
    把检测结果转成结构化特征 JSON。

    流程：先做姿态归一化（消除头部侧倾），再算各项比例；同时给出拍摄质量提示，
    测量不可靠时如实标注，而不是输出一个看似精确的假数字。
    """
    points = detection.points
    if not points:
        raise ValueError("没有可用的关键点，无法提取特征")

    # 1) 姿态归一化：把两眼连线转成水平，之后的「水平距离」才有意义
    points, roll_degrees = _normalize_roll(points)
    pose, quality_warnings = _pose_quality(points, roll_degrees)

    cheek_w = max(_dx(points["cheek_l"], points["cheek_r"]), 1e-6)
    face_height = max(_dy(points["forehead_top"], points["chin"]), 1e-6)

    # 2) 发际线：像素检测（关键点里没有），失败才退回关键点近似
    hairline_y, hairline_confidence = _estimate_hairline(image, points, face_height, cheek_w)

    courts = _three_courts(points, face_height, hairline_y, hairline_confidence)
    eyes = _five_eyes(points)
    shape = _face_shape(points, face_height)
    facial = _facial_features(points, courts, cheek_w)
    complexion = _complexion(image, points, detection.bbox or (0, 0, 1, 1))
    symmetry = _symmetry(points, cheek_w)

    # 3) 单项可信度提示
    if hairline_y is None:
        quality_warnings.append(
            "未能在照片中识别发际线（刘海遮挡、逆光或额头未完整入镜），"
            "**上庭记为未测得**，解读不对上庭高低/额部高低下判断；露出额头重拍可测得"
        )
    elif not courts.get("upper", {}).get("measured", False):
        quality_warnings.append(
            f"测得的上庭仅占 {courts['upper']['ratio'] * 100:.0f}%（三庭理想为 1:1:1），"
            "通常是刘海盖住额头，**上庭记为未测得**，建议露出额头再拍一张"
        )

    features: dict[str, Any] = {
        "engine": detection.engine,
        "estimated": detection.estimated,
        "image": {
            "width": detection.image_size[0],
            "height": detection.image_size[1],
            "bbox": detection.bbox,
        },
        "pose": pose,
        "three_courts": courts,
        "five_eyes": eyes,
        "face_shape": shape,
        "facial_features": facial,
        "complexion": complexion,
        "symmetry": symmetry,
        "warnings": list(detection.warnings) + quality_warnings,
    }
    features["summary"] = _summary_lines(features)
    features["search_terms"] = _search_terms(features)
    return features


def _summary_lines(features: dict[str, Any]) -> list[str]:
    """给历史列表 / 结果页用的要点短句。"""
    forehead = features["facial_features"]["forehead"]
    return [
        f"脸型：{features['face_shape']['label']}（长宽比 {features['face_shape']['length_ratio']}）",
        f"三庭：{features['three_courts']['judgement']}",
        f"五眼：{features['five_eyes']['judgement']}",
        f"额部：{forehead['width_label']}（额宽/颧宽 {forehead['width_ratio']}）",
        f"气色：{features['complexion']['label']}",
        f"对称度：{features['symmetry']['label']}（{features['symmetry']['score']}）",
    ]


def _search_terms(features: dict[str, Any]) -> list[str]:
    """
    构造 RAG 检索词。

    这是「人脸特征 → 知识库检索」的连接点：
    标签化术语（如 鹅蛋脸、三庭均衡）更容易命中面相知识条目。
    """
    terms: list[str] = [
        features["face_shape"]["label"],
        f"三庭 {features['three_courts']['balance']}",
        f"眼距 {features['five_eyes']['inner_gap_ratio']['label']}",
        f"眼宽 {features['five_eyes']['eye_width_ratio']['label']}",
        features["facial_features"]["brow"]["label"],
        features["facial_features"]["eye"]["label"],
        features["facial_features"]["nose"]["label"],
        features["facial_features"]["mouth"]["label"],
        features["facial_features"]["forehead"]["width_label"],
        f"额部 {features['facial_features']['forehead']['width_label']} 面相",
        f"气色 {features['complexion'].get('tone', '自然')}",
        "印堂 眉间 气色",
        "面相 五官 比例 传统参考",
    ]
    return [term for term in terms if term]
