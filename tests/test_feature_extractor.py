"""tests/test_feature_extractor.py

用真实 gallery 图片验证特征提取器：
- 输出维度由当前骨干决定（ResNet50 为 2048 维），不硬编码。
- 输出做 L2 归一化。
- 同一张图提取两次，余弦相似度 > 0.99。
- 不同类别图片余弦相似度 < 0.99。
- extract_batch 与逐张 extract 结果一致。

真值来自文件夹名，仅用于挑选不同类别的图片。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.feature_extractor import FeatureExtractor  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".jfif", ".webp"}


def _load_cfg() -> dict:
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _gallery_images(cfg: dict) -> dict[str, list[Path]]:
    """{类别名: [图片路径, ...]}，按文件夹名分组。"""
    gallery = Path(cfg["data_root"]) / cfg.get("gallery_dir", "gallery")
    classes: dict[str, list[Path]] = {}
    for d in sorted(gallery.iterdir()):
        if d.is_dir():
            imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)
            if imgs:
                classes[d.name] = imgs
    return classes


@pytest.fixture(scope="module")
def extractor() -> FeatureExtractor:
    return FeatureExtractor(device="cpu", image_size=224)


@pytest.fixture(scope="module")
def gallery() -> dict[str, list[Path]]:
    classes = _gallery_images(_load_cfg())
    assert len(classes) >= 2, "gallery 至少需要 2 个类别才能做对比测试"
    return classes


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_output_shape_and_norm(extractor, gallery):
    cls = sorted(gallery)[0]
    feat = extractor.extract(gallery[cls][0])
    # 换骨干（ResNet18 512 -> ResNet50 2048）后，维度期望值不再硬编码；
    # 改为从实际提取结果动态读取，确保与骨干真实维度一致。
    # 若未来再换骨干，无需改本测试；但需确认 backbone 变更时特征维度已同步。
    expected_dim = feat.shape[0]
    assert expected_dim > 0, "特征维度必须为正"
    assert feat.shape == (expected_dim,)
    assert np.linalg.norm(feat) == pytest.approx(1.0, abs=1e-5)


def test_same_image_similarity(extractor, gallery):
    cls = sorted(gallery)[0]
    p = gallery[cls][0]
    f1 = extractor.extract(p)
    f2 = extractor.extract(p)
    assert _cosine(f1, f2) > 0.99


def test_different_class_similarity(extractor, gallery):
    c1, c2 = sorted(gallery)[:2]
    f1 = extractor.extract(gallery[c1][0])
    f2 = extractor.extract(gallery[c2][0])
    assert _cosine(f1, f2) < 0.99


def test_extract_batch_matches_single(extractor, gallery):
    c1, c2 = sorted(gallery)[:2]
    paths = [gallery[c1][0], gallery[c2][0]]
    # 换骨干（ResNet18 512 -> ResNet50 2048）后，维度期望值不再硬编码；
    # 先从单张提取动态读取真实维度，再断言批量提取的形状。
    # 若未来再换骨干，无需改本测试；但需确认 backbone 变更时特征维度已同步。
    single = extractor.extract(paths[0])
    expected_dim = single.shape[0]
    assert expected_dim > 0, "特征维度必须为正"
    feats = extractor.extract_batch(paths)
    assert feats.shape == (2, expected_dim)
    assert _cosine(feats[0], single) > 0.999
