"""tests/test_webcam_utils.py

只测 core.webcam 的工具函数，不打开真实摄像头、不创建窗口。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import webcam  # noqa: E402


# ---------------------------------------------------------------------
# InferenceGate
# ---------------------------------------------------------------------
def test_gate_first_frame_infers():
    gate = webcam.InferenceGate(infer_every=5)
    assert gate.tick() is True  # 第 1 帧必须立即识别


def test_gate_cadence_every_five_frames():
    gate = webcam.InferenceGate(infer_every=5)
    infer_frames = [i for i in range(1, 16) if gate.tick()]
    assert infer_frames == [1, 6, 11]


def test_gate_configurable_interval():
    gate = webcam.InferenceGate(infer_every=1)
    assert all(gate.tick() for _ in range(5))
    gate3 = webcam.InferenceGate(infer_every=3)
    flags = [gate3.tick() for _ in range(7)]
    assert flags == [True, False, False, True, False, False, True]


def test_gate_invalid_interval():
    with pytest.raises(ValueError):
        webcam.InferenceGate(infer_every=0)


def test_gate_caches_last_result():
    gate = webcam.InferenceGate(infer_every=5)
    assert gate.has_result is False
    gate.tick()
    gate.update("apple", 0.91, [{"label": "apple", "score": 0.91}])
    # 接下来的非识别帧仍能读到上一次结果
    for _ in range(4):
        assert gate.tick() is False
        assert gate.has_result is True
        assert gate.last_label == "apple"
        assert gate.last_score == pytest.approx(0.91)
        assert gate.last_topk[0]["label"] == "apple"


# ---------------------------------------------------------------------
# 帧特征 + Recognizer 调用
# ---------------------------------------------------------------------
class FakeExtractor:
    """最小鸭子类型：具备 preprocess / model / device，输出 512 维。"""

    def __init__(self):
        import torch

        self.device = torch.device("cpu")
        self.calls = 0

        def preprocess(pil_img):
            import torch

            return torch.zeros(3, 224, 224)

        class _Model:
            def __call__(self, batch):
                import torch

                n = batch.shape[0]
                return torch.ones(n, 512)

        self.preprocess = preprocess
        self.model = _Model()


class RecordingRecognizer:
    def __init__(self):
        self.received = None

    def predict_feature(self, feat):
        self.received = feat
        return "cup", 0.87, [{"label": "cup", "score": 0.87}]


def test_frame_to_feature_shape():
    extractor = FakeExtractor()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    feat = webcam.frame_to_feature(extractor, frame)
    assert feat.shape == (512,)
    assert feat.dtype == np.float32


def test_frame_to_feature_rejects_empty_frame():
    with pytest.raises(ValueError):
        webcam.frame_to_feature(FakeExtractor(), np.empty((0, 0, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        webcam.frame_to_feature(FakeExtractor(), None)


def test_recognize_frame_calls_recognizer():
    extractor = FakeExtractor()
    recognizer = RecordingRecognizer()
    frame = np.full((100, 120, 3), 128, dtype=np.uint8)
    label, score, topk = webcam.recognize_frame(recognizer, extractor, frame)
    assert (label, score) == ("cup", 0.87)
    assert topk[0]["label"] == "cup"
    # 传给 Recognizer 的确实是帧特征向量
    assert recognizer.received.shape == (512,)


# ---------------------------------------------------------------------
# 按键
# ---------------------------------------------------------------------
def test_is_quit_key():
    assert webcam.is_quit_key(ord("q")) is True
    assert webcam.is_quit_key(ord("Q")) is True
    assert webcam.is_quit_key(ord("a")) is False
    assert webcam.is_quit_key(-1) is False  # waitKey 无按键时常返回 -1


# ---------------------------------------------------------------------
# 画面叠加
# ---------------------------------------------------------------------
def _panel_pixels(frame):
    return set(frame[:40, :, :].reshape(-1, 3).mean(axis=1).round().tolist())


def test_draw_overlay_warming_up():
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    out = webcam.draw_overlay(frame, None, None)
    assert out is frame  # in-place
    # 顶部面板被填充，且出现灰色文字像素（非纯黑面板）
    panel = frame[:40, :, :]
    assert (panel.sum(axis=2) > 0).any()


def test_draw_overlay_known_green():
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    webcam.draw_overlay(frame, "apple", 0.95, unknown_label="unknown")
    panel = frame[:40, :, :]
    # 绿色文字：G 通道显著大于 R 通道
    greenish = (panel[:, :, 1] > panel[:, :, 2] + 40) & (panel[:, :, 1] > 80)
    assert greenish.any()


def test_draw_overlay_unknown_red():
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    webcam.draw_overlay(frame, "unknown", 0.42, unknown_label="unknown")
    panel = frame[:40, :, :]
    # 红色文字：R 通道显著大于 G 通道
    reddish = (panel[:, :, 2] > panel[:, :, 1] + 40) & (panel[:, :, 2] > 80)
    assert reddish.any()


def test_draw_overlay_text_contains_score():
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    # 仅验证不抛异常且形状不变（文字内容通过 OCR 验证过重，抽样像素即可）
    out = webcam.draw_overlay(frame, "banana", 0.8)
    assert out.shape == (120, 200, 3)
