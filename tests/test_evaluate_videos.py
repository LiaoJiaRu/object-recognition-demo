"""tests/test_evaluate_videos.py

scripts/evaluate_videos.py 的纯逻辑测试。
合成 mp4 + 注入假 frame_fn，不依赖 torch / 模型权重 / 摄像头。

注意假帧函数的通道语义：cv2 读出的是 BGR，绿色 (0,255,0) 的 B 通道是 0，
所以用「B 通道均值」只能区分蓝与非蓝（test_video.py 的历史教训）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_videos import evaluate_videos  # noqa: E402


def make_solid_video(path: Path, bgr: tuple, fps: int = 10, seconds: int = 2) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (64, 64))
    if not writer.isOpened():
        pytest.skip("当前环境 VideoWriter 无法写 mp4（缺编码器）")
    for _ in range(fps * seconds):
        writer.write(np.full((64, 64, 3), bgr, dtype=np.uint8))
    writer.release()


def make_cfg(tmp_path: Path) -> dict:
    return {
        "data_root": str(tmp_path),
        "video_test_dir": "test_videos",
        "frame_interval": 1.0,
        "max_frames": 0,
        "unknown_label": "unknown",
    }


def fake_frame_fn(frame_bgr: np.ndarray):
    """按 B 通道均值把帧分成 apple(蓝) / banana(非蓝)。"""
    avg_b = float(np.mean(frame_bgr[:, :, 0]))
    if avg_b > 200:
        return "apple", 0.9, []
    return "banana", 0.8, []


def test_evaluate_videos_report(tmp_path: Path) -> None:
    (tmp_path / "test_videos" / "apple").mkdir(parents=True)
    (tmp_path / "test_videos" / "banana").mkdir(parents=True)
    make_solid_video(tmp_path / "test_videos" / "apple" / "a.mp4", (255, 0, 0))
    make_solid_video(tmp_path / "test_videos" / "banana" / "b.mp4", (0, 255, 0))

    report = evaluate_videos(make_cfg(tmp_path), frame_fn=fake_frame_fn)

    assert report["video_total"] == 2
    assert report["video_correct"] == 2
    assert report["video_top1_acc"] == 1.0
    assert report["per_class"]["apple"] == {"total": 1, "correct": 1}
    assert all(d["ok"] for d in report["details"])
    assert all(d["n_frames"] >= 2 for d in report["details"])


def test_evaluate_videos_counts_read_error_as_wrong(tmp_path: Path) -> None:
    (tmp_path / "test_videos" / "apple").mkdir(parents=True)
    # 0 字节伪视频：打不开时应记 read_error 并计为错误，而不是让整个评估中断
    (tmp_path / "test_videos" / "apple" / "bad.mp4").write_bytes(b"not a video")
    make_solid_video(tmp_path / "test_videos" / "apple" / "good.mp4", (255, 0, 0))

    report = evaluate_videos(make_cfg(tmp_path), frame_fn=fake_frame_fn)

    assert report["video_total"] == 2
    assert report["video_correct"] == 1
    bad = next(d for d in report["details"] if d["pred"] == "read_error")
    assert bad["error"]


def test_evaluate_videos_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        evaluate_videos(make_cfg(tmp_path))


def test_evaluate_videos_no_videos(tmp_path: Path) -> None:
    (tmp_path / "test_videos" / "apple").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        evaluate_videos(make_cfg(tmp_path), frame_fn=fake_frame_fn)
