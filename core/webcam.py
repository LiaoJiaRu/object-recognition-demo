"""core/webcam.py

摄像头实时识别的可测试工具层（阶段 10）。
- 单线程同步使用：每个工具函数只处理一帧，不持有 VideoCapture，
  避免多线程竞争摄像头导致黑屏。
- 识别仍走 Recognizer.predict_feature；本模块只做 BGR 帧到特征的适配。
- 显示层只做文字叠加，不含主循环（主循环在 webcam_app.py）。
- 中文字幕条用 PIL 中转绘制（OpenCV putText 不支持中文）。

自动测试只调用本模块纯函数，不打开真实摄像头。
"""
from __future__ import annotations

import os
import sys
from typing import Any, Optional

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

DEFAULT_INFER_EVERY = 5
QUIT_KEY = ord("q")


# ---------------------------------------------------------------------
# 推理节拍：每 N 帧识别一次，其余帧沿用上一次结果
# ---------------------------------------------------------------------
class InferenceGate:
    """帧计数器 + 最近一次识别结果缓存。

    第 1 帧立即识别，之后每 infer_every 帧识别一次。
    未识别的帧通过 label/score 读取上一次结果，画面不会闪烁清空。
    """

    def __init__(self, infer_every: int = DEFAULT_INFER_EVERY) -> None:
        if infer_every < 1:
            raise ValueError("infer_every 必须 >= 1")
        self.infer_every = int(infer_every)
        self.frame_count = 0
        self.last_label: Optional[str] = None
        self.last_score: Optional[float] = None
        self.last_topk: list[dict[str, Any]] = []

    def tick(self) -> bool:
        """推进一帧，返回本帧是否应当执行识别。"""
        self.frame_count += 1
        return (self.frame_count - 1) % self.infer_every == 0

    def update(self, label: str, score: float,
               topk: Optional[list[dict[str, Any]]] = None) -> None:
        """记录本帧刚得到的识别结果。"""
        self.last_label = str(label)
        self.last_score = float(score)
        self.last_topk = list(topk or [])

    @property
    def has_result(self) -> bool:
        return self.last_label is not None


# ---------------------------------------------------------------------
# 帧 -> 特征 -> Recognizer
# ---------------------------------------------------------------------
def frame_to_feature(extractor, frame_bgr: np.ndarray) -> np.ndarray:
    """把 OpenCV 的 BGR 帧转换成 extractor 输出的 512 维特征（未归一化也可，
    Recognizer.predict_feature 内部会归一化）。"""
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("空帧无法提取特征")
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = extractor.preprocess(Image.fromarray(rgb))
    batch = tensor.unsqueeze(0).to(extractor.device)
    with torch.no_grad():
        feat = extractor.model(batch).cpu().numpy()[0]
    return np.asarray(feat, dtype=np.float32)


def recognize_frame(recognizer, extractor, frame_bgr: np.ndarray):
    """识别单帧：适配为特征后调用核心 Recognizer，返回 (label, score, topk)。"""
    feat = frame_to_feature(extractor, frame_bgr)
    return recognizer.predict_feature(feat)


# ---------------------------------------------------------------------
# 显示与按键
# ---------------------------------------------------------------------
def is_quit_key(key: int, quit_key: int = QUIT_KEY) -> bool:
    """是否按下退出键（q / Q 均可）。"""
    if key < 0:
        return False
    code = key & 0xFF
    return code in (quit_key & 0xFF, ord("Q"))


# ---------------------------------------------------------------------
# 取景框：坐标计算 + ROI 裁剪
# ---------------------------------------------------------------------
def get_guide_box_rect(h: int, w: int, ratio: float = 0.6) -> tuple[int, int, int, int]:
    """计算参考取景框的坐标 (x1, y1, x2, y2)。

    - 框尺寸为画面短边 * ratio（默认 60%），居中放置。
    - 返回坐标是整数裁剪边界，保证 x2 <= w, y2 <= h。
    """
    short = min(h, w)
    side = int(short * ratio)
    x1 = (w - side) // 2
    y1 = (h - side) // 2
    x2 = x1 + side
    y2 = y1 + side
    return x1, y1, x2, y2


def crop_guide_box_roi(frame_bgr: np.ndarray, ratio: float = 0.6) -> np.ndarray:
    """把画面裁剪到参考取景框的 ROI 区域，返回 BGR 子图（可能比原图小）。

    识别器应该用 ROI 而非整帧提特征，避免背景干扰。
    """
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("空帧无法裁剪")
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = get_guide_box_rect(h, w, ratio)
    # clip 边界防止越界（极端尺寸下）
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    return frame_bgr[y1:y2, x1:x2].copy()


# ---------------------------------------------------------------------
# 显示与按键
# ---------------------------------------------------------------------
def draw_overlay(
    frame_bgr: np.ndarray,
    label: Optional[str],
    score: Optional[float],
    unknown_label: str = "unknown",
) -> np.ndarray:
    """在帧左上角叠加 label / score，返回同一帧（in-place）。

    - 尚无识别结果（预热）：灰色提示；
    - unknown：红色；
    - 已注册类别：绿色。
    """
    h, w = frame_bgr.shape[:2]
    panel_h = 64 if h >= 64 else h
    cv2.rectangle(frame_bgr, (0, 0), (w, panel_h), (0, 0, 0), thickness=-1)

    if label is None:
        text = "warming up..."
        color = (160, 160, 160)  # BGR 灰
    else:
        score_text = f"{score:.3f}" if isinstance(score, (int, float)) else "--"
        text = f"{label}  {score_text}"
        if label == unknown_label:
            color = (60, 60, 230)   # BGR 红
        else:
            color = (70, 200, 70)   # BGR 绿

    scale = 0.8 if w >= 320 else 0.5
    thickness = 2
    cv2.putText(frame_bgr, text, (12, 40 if h >= 64 else h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
    return frame_bgr


# ---------------------------------------------------------------------
# 中文字幕条（PIL 中转绘制，支持中文）
# ---------------------------------------------------------------------

# 颜色常量（BGR，和取景框同色）
COLOR_GREEN = (70, 200, 70)
COLOR_RED = (60, 60, 230)

# 字幕条高度占比（画面高度的 15%）
SUBTITLE_HEIGHT_RATIO = 0.15
# 字幕条背景透明度
SUBTITLE_BG_ALPHA = 0.7

# 中文字体路径缓存
_CN_FONT_PATH: Optional[str] = None


def _find_cn_font() -> Optional[str]:
    """查找系统中文字体文件路径，找到第一个可用的就返回。

    优先顺序：微软雅黑 → 黑体 → 宋体。
    跨平台兼容：Windows / macOS / Linux 各尝试常见路径。
    """
    global _CN_FONT_PATH
    if _CN_FONT_PATH is not None:
        return _CN_FONT_PATH

    candidates: list[str] = []

    if sys.platform.startswith("win"):
        font_dir = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts")
        candidates += [
            os.path.join(font_dir, "msyh.ttc"),      # 微软雅黑
            os.path.join(font_dir, "msyhbd.ttc"),    # 微软雅黑粗体
            os.path.join(font_dir, "simhei.ttf"),    # 黑体
            os.path.join(font_dir, "simsun.ttc"),    # 宋体
        ]
    elif sys.platform == "darwin":
        candidates += [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
        ]
    else:  # Linux
        candidates += [
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        ]

    for path in candidates:
        if os.path.isfile(path):
            _CN_FONT_PATH = path
            return path

    _CN_FONT_PATH = ""  # 标记为已查找但未找到
    return None


def _rgb_to_bgr(color_rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """RGB → BGR。"""
    return (color_rgb[2], color_rgb[1], color_rgb[0])


def draw_subtitle_bar(
    frame_bgr: np.ndarray,
    big_text: str = "",
    small_text: str = "",
    color_bgr: tuple[int, int, int] = COLOR_GREEN,
) -> np.ndarray:
    """在画面底部绘制半透明黑底字幕条（两行文字，居中，支持中文）。

    参数:
        frame_bgr: 原始 BGR 帧（in-place 修改并返回）
        big_text: 第一行大字（主要信息，如倒计时、状态、结果）
        small_text: 第二行小字（辅助信息，如"可以转动物体"）
        color_bgr: 文字颜色（BGR），默认绿色，unknown 用红色

    返回:
        修改后的 frame_bgr（同一对象）
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return frame_bgr

    h, w = frame_bgr.shape[:2]
    bar_h = max(40, int(h * SUBTITLE_HEIGHT_RATIO))
    bar_y = h - bar_h

    # ---- 1. 绘制半透明黑色背景 ----
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (0, bar_y), (w, h), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, SUBTITLE_BG_ALPHA, frame_bgr, 1 - SUBTITLE_BG_ALPHA, 0, frame_bgr)

    # ---- 2. 用 PIL 绘制中文文字 ----
    font_path = _find_cn_font()
    if not font_path or not big_text:
        # 没有中文字体或没有大字 → 退化到 OpenCV 英文绘制（至少能显示数字）
        if big_text:
            big_size = int(bar_h * 0.45)
            cv2.putText(
                frame_bgr, big_text,
                (w // 2 - len(big_text) * big_size // 4, bar_y + int(bar_h * 0.55)),
                cv2.FONT_HERSHEY_SIMPLEX, big_size / 30,
                color_bgr, 2, cv2.LINE_AA,
            )
        return frame_bgr

    # 转换 BGR → RGB → PIL Image
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_img)

    # 字号：大字约占字幕条高度的 45%，小字约 25%
    big_font_size = max(12, int(bar_h * 0.45))
    small_font_size = max(10, int(bar_h * 0.25))

    try:
        big_font = ImageFont.truetype(font_path, big_font_size)
        small_font = ImageFont.truetype(font_path, small_font_size)
    except (OSError, IOError):
        # 字体加载失败，退化到默认字体
        big_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    color_rgb = _rgb_to_bgr(color_bgr)  # BGR → RGB 给 PIL 用

    # ---- 第一行（大字）----
    # 大字垂直居中于字幕条上半部分（中心约在 bar_y + bar_h * 0.30）
    if big_text:
        bbox = draw.textbbox((0, 0), big_text, font=big_font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = (w - text_w) // 2
        center_y = bar_y + int(bar_h * 0.30)
        y = center_y - text_h // 2 - bbox[1]  # 减去 textbbox 的 top 偏移
        draw.text((x, y), big_text, font=big_font, fill=color_rgb)

    # ---- 第二行（小字）----
    # 小字垂直居中于字幕条下半部分（中心约在 bar_y + bar_h * 0.72）
    if small_text:
        bbox = draw.textbbox((0, 0), small_text, font=small_font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = (w - text_w) // 2
        center_y = bar_y + int(bar_h * 0.72)
        y = center_y - text_h // 2 - bbox[1]
        draw.text((x, y), small_text, font=small_font, fill=color_rgb)

    # 转回 BGR
    frame_bgr[:, :, :] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return frame_bgr
