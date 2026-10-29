"""core/video.py

视频识别的可测试工具层。
- 按固定时间间隔抽帧（OpenCV VideoCapture，CPU 可运行）。
- 逐帧识别通过注入的 frame_fn 完成，本模块不依赖 torch / extractor / GUI，
  便于自动测试（测试传入假的 frame_fn 即可）。
- 多帧结果聚合为整段视频的 label / score：多数投票，票数相同取平均分高者。
- 视频与图片共用同一特征库；本模块不做录入，只做识别。

显示层与主循环在 app.py 的「视频识别」Tab，本模块只含纯逻辑。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

# 单帧识别结果：(label, score, topk)，与 Recognizer.predict_feature 返回一致
FrameResult = tuple[str, float, list[dict[str, Any]]]
# 帧 -> 识别结果 的回调；GUI 层封装 core.webcam.recognize_frame 传入
FrameFn = Callable[[np.ndarray], FrameResult]

DEFAULT_FRAME_INTERVAL = 1.0
FALLBACK_FPS = 25.0  # 部分容器读不到 fps 时的兜底值


def sample_frames(
    video_path: str | Path,
    interval_sec: float = DEFAULT_FRAME_INTERVAL,
    max_frames: int = 0,
) -> list[tuple[float, np.ndarray]]:
    """按时间间隔抽帧，返回 [(timestamp_sec, frame_bgr), ...]，按时间升序。

    - interval_sec <= 0 报错；max_frames <= 0 表示不限制。
    - 视频不存在抛 FileNotFoundError；无法打开抛 ValueError。
    - fps 读不到时按 FALLBACK_FPS 处理。
    """
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"视频文件不存在: {path}")
    if interval_sec <= 0:
        raise ValueError("interval_sec 必须 > 0")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"无法打开视频: {path}")

    frames: list[tuple[float, np.ndarray]] = []
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1e-3:
            fps = FALLBACK_FPS
        step = max(1, int(round(fps * interval_sec)))
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if idx % step == 0:
                frames.append((idx / fps, frame))
                if max_frames > 0 and len(frames) >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()
    return frames


def aggregate_results(
    frame_results: list[FrameResult],
    unknown_label: str = "unknown",
) -> tuple[str, float, dict[str, int]]:
    """把逐帧 (label, score, topk) 聚合为整段视频结果。

    规则：对所有帧的预测 label 多数投票（unknown 也算一票）；
    票数相同取平均 score 高者；最终 score 为获胜 label 所有帧的均值。
    空结果返回 (unknown_label, 0.0, {})。

    返回 (label, score, votes)，votes 为 {label: 票数}。
    """
    if not frame_results:
        return unknown_label, 0.0, {}

    scores_by_label: dict[str, list[float]] = {}
    for label, score, _ in frame_results:
        scores_by_label.setdefault(str(label), []).append(float(score))

    best_label = max(
        scores_by_label,
        key=lambda l: (
            len(scores_by_label[l]),
            sum(scores_by_label[l]) / len(scores_by_label[l]),
        ),
    )
    best_scores = scores_by_label[best_label]
    best_score = sum(best_scores) / len(best_scores)
    votes = {l: len(s) for l, s in scores_by_label.items()}
    return best_label, best_score, votes


def recognize_video(
    video_path: str | Path,
    frame_fn: FrameFn,
    interval_sec: float = DEFAULT_FRAME_INTERVAL,
    max_frames: int = 0,
    unknown_label: str = "unknown",
) -> dict[str, Any]:
    """抽帧 -> 逐帧识别 -> 聚合，返回整段视频的识别结果。

    frame_fn(frame_bgr) -> (label, score, topk)，由调用方注入；
    单帧识别异常时该帧记为 unknown（score 0.0），不中断整段识别。

    返回 dict:
        label / score     聚合后的整段结果
        votes             {label: 票数}
        frames            [{"time": 秒, "label": str, "score": float}, ...] 逐帧明细
        n_sampled         实际抽到的帧数
    """
    sampled = sample_frames(video_path, interval_sec=interval_sec, max_frames=max_frames)

    results: list[FrameResult] = []
    frames_detail: list[dict[str, Any]] = []
    for ts, frame in sampled:
        try:
            label, score, _ = frame_fn(frame)
        except Exception:
            label, score = unknown_label, 0.0
        results.append((label, score, []))
        frames_detail.append({"time": round(ts, 3), "label": label, "score": float(score)})

    label, score, votes = aggregate_results(results, unknown_label=unknown_label)
    return {
        "label": label,
        "score": float(score),
        "votes": votes,
        "frames": frames_detail,
        "n_sampled": len(sampled),
    }
