"""tests/test_video.py

core/video.py 的纯逻辑测试。用 OpenCV 合成假视频，注入假的 frame_fn，
不依赖真实模型、摄像头或麦克风。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))

import video as vc  # noqa: E402


def make_fake_video(path: Path, fps: int = 25, seconds: int = 2) -> None:
    """写一段纯色帧（红→绿→蓝循环）的 mp4 假视频。"""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (64, 64))
    if not writer.isOpened():
        pytest.skip("当前环境不支持 VideoWriter 写 mp4")
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR
    for _ in range(fps * seconds):
        color = colors[_ % len(colors)]
        frame = np.full((64, 64, 3), color, dtype=np.uint8)
        writer.write(frame)
    writer.release()


# ------------------------------------------------------------------
# sample_frames
# ------------------------------------------------------------------
def test_sample_frames_basic(tmp_path: Path) -> None:
    vid = tmp_path / "basic.mp4"
    make_fake_video(vid, fps=10, seconds=2)  # 20 帧
    frames = vc.sample_frames(vid, interval_sec=1.0)  # 每 1s 取 1 帧，共 2 帧
    assert len(frames) == 2
    # 时间戳近似为 0.0s 和 1.0s
    assert abs(frames[0][0] - 0.0) < 0.05
    assert abs(frames[1][0] - 1.0) < 0.05
    assert all(isinstance(f, np.ndarray) and f.shape == (64, 64, 3) for _, f in frames)


def test_sample_frames_max_frames(tmp_path: Path) -> None:
    vid = tmp_path / "max.mp4"
    make_fake_video(vid, fps=10, seconds=2)
    frames = vc.sample_frames(vid, interval_sec=0.5, max_frames=2)
    assert len(frames) == 2


def test_sample_frames_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        vc.sample_frames("not_exist.mp4")


def test_sample_frames_invalid_interval(tmp_path: Path) -> None:
    vid = tmp_path / "dummy.mp4"
    make_fake_video(vid, fps=10, seconds=1)
    with pytest.raises(ValueError):
        vc.sample_frames(vid, interval_sec=0)


# ------------------------------------------------------------------
# aggregate_results
# ------------------------------------------------------------------
def test_aggregate_majority() -> None:
    # 5 帧 apple，3 帧 banana
    results = [
        ("apple", 0.9, []),
        ("apple", 0.8, []),
        ("apple", 0.85, []),
        ("apple", 0.82, []),
        ("apple", 0.88, []),
        ("banana", 0.7, []),
        ("banana", 0.72, []),
        ("banana", 0.71, []),
    ]
    label, score, votes = vc.aggregate_results(results)
    assert label == "apple"
    assert pytest.approx(score, abs=0.01) == 0.85
    assert votes == {"apple": 5, "banana": 3}


def test_aggregate_tie_breaker() -> None:
    # apple 2 帧 avg 0.5，banana 2 帧 avg 0.8 -> banana 赢（平局看均值）
    results = [
        ("apple", 0.45, []),
        ("apple", 0.55, []),
        ("banana", 0.78, []),
        ("banana", 0.82, []),
    ]
    label, score, votes = vc.aggregate_results(results)
    assert label == "banana"
    assert pytest.approx(score, abs=0.01) == 0.80
    assert votes == {"apple": 2, "banana": 2}


def test_aggregate_empty() -> None:
    label, score, votes = vc.aggregate_results([], unknown_label="unknown")
    assert label == "unknown"
    assert score == 0.0
    assert votes == {}


# ------------------------------------------------------------------
# recognize_video（注入假 frame_fn）
# ------------------------------------------------------------------
def fake_frame_fn(frame_bgr: np.ndarray) -> tuple[str, float, list]:
    """假的单帧识别：根据帧的平均 B 值决定标签。"""
    avg_b = float(np.mean(frame_bgr[:, :, 0]))
    if avg_b > 200:
        return "blue", 0.85, []
    if avg_b > 100:
        return "green", 0.80, []
    return "red", 0.75, []


def test_recognize_video_full(tmp_path: Path) -> None:
    vid = tmp_path / "full.mp4"
    # 写 3s@10fps=30 帧：红→绿→蓝循环，每色 10 帧
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(vid), fourcc, 10, (64, 64))
    if not writer.isOpened():
        pytest.skip("VideoWriter 不可用")
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR
    for _ in range(30):
        frame = np.full((64, 64, 3), colors[_ % 3], dtype=np.uint8)
        writer.write(frame)
    writer.release()

    # 每 1s（10 帧）抽 1 帧，共 3 帧，分别对应红→绿→蓝
    result = vc.recognize_video(
        vid, frame_fn=fake_frame_fn, interval_sec=1.0, max_frames=0
    )

    assert result["n_sampled"] == 3
    assert len(result["frames"]) == 3
    # 投票 3 个各 1 帧，分数平局（都是 1 帧），按分数 0.85 > 0.80 > 0.75 -> blue 赢
    assert result["label"] == "blue"
    assert pytest.approx(result["score"], abs=0.01) == 0.85
    assert result["votes"] == {"red": 1, "green": 1, "blue": 1}


def test_recognize_video_frame_error(tmp_path: Path) -> None:
    """单帧识别异常时整段不中断，该帧记为 unknown。"""
    vid = tmp_path / "err.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(vid), fourcc, 10, (64, 64))
    if not writer.isOpened():
        pytest.skip("VideoWriter 不可用")
    for _ in range(10):
        writer.write(np.full((64, 64, 3), (0, 0, 255), dtype=np.uint8))
    writer.release()

    error_every = [0]

    def flaky_fn(frame_bgr: np.ndarray) -> tuple[str, float, list]:
        error_every[0] += 1
        if error_every[0] % 2 == 0:
            raise RuntimeError("boom")
        return "red", 0.7, []

    result = vc.recognize_video(vid, frame_fn=flaky_fn, interval_sec=0.5)
    # 10 帧每 0.5s（5 帧间隔）-> 抽 2 帧（0s 和 1s）
    assert result["n_sampled"] == 2
    assert result["votes"] == {"red": 1, "unknown": 1}
    # 票数 red 1 == unknown 1，平局按分数：red 0.7 > unknown 0.0
    assert result["label"] == "red"


# ------------------------------------------------------------------
# 边界
# ------------------------------------------------------------------
def test_sample_frames_zero_length(tmp_path: Path) -> None:
    """0 帧视频应返回空列表。"""
    vid = tmp_path / "empty.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(vid), fourcc, 25, (64, 64))
    if not writer.isOpened():
        pytest.skip("VideoWriter 不可用")
    writer.release()
    frames = vc.sample_frames(vid, interval_sec=1.0)
    assert frames == []
