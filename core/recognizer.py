"""core/recognizer.py

识别器与未知判断（阶段 5）。
- 基于特征库中每个图片特征的余弦相似度做最近邻 top-k 检索。
- 最高相似度 < threshold（来自 config.yaml）时输出 unknown_label。
- 特征库为空时一律输出 unknown_label。
- 不涉及 GUI 与语音。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import numpy as np

PathLike = Union[str, Path]


class Recognizer:
    """余弦相似度 top-k 识别器。

    构造参数:
        gallery: 已加载的 FeatureGallery（需提供 get_all()）。
        extractor: 具备 extract(path) -> np.ndarray 的对象；仅 predict(图片) 时需要。
        cfg: 配置字典，读取 threshold / topk / unknown_label，不硬编码。
    """

    def __init__(self, gallery, extractor=None, cfg: dict | None = None) -> None:
        cfg = cfg or {}
        self.gallery = gallery
        self.extractor = extractor
        self.threshold = float(cfg.get("threshold", 0.75))
        self.topk = int(cfg.get("topk", 5))
        self.unknown_label = cfg.get("unknown_label", "unknown")

    def _gallery_data(self) -> tuple[np.ndarray, np.ndarray]:
        """返回 (features(N,D) L2 归一化, labels(N,))；空库时 features 形状 (0,0)。"""
        feats, labels, _ = self.gallery.get_all()
        feats = np.asarray(feats, dtype=np.float32)
        if feats.size == 0:
            return np.empty((0, 0), dtype=np.float32), np.asarray([], dtype=object)
        return feats, np.asarray(labels, dtype=object)

    @staticmethod
    def _normalize(feature: np.ndarray) -> np.ndarray:
        v = np.asarray(feature, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(v))
        if norm < 1e-12:
            raise ValueError("特征向量为零向量，无法计算余弦相似度")
        return v / norm

    def predict_feature(self, feature: np.ndarray) -> tuple[str, float, list[dict[str, Any]]]:
        """对单个特征向量做识别。

        返回 (label, score, topk)：
        - label: 最高相似度 >= threshold 时为其类别名，否则为 unknown_label；
        - score: 最高相似度（float，unknown 时也保留该数值）；
        - topk: [{"label": str, "score": float}, ...]，按相似度降序。
        """
        feats, labels = self._gallery_data()

        # 空特征库：无法识别，一律 unknown；score 记为 0.0
        if feats.shape[0] == 0:
            return self.unknown_label, 0.0, []

        q = self._normalize(feature)
        # 特征库特征已 L2 归一化，点积即余弦相似度
        sims = feats @ q
        k = min(self.topk, feats.shape[0])
        idx = np.argsort(-sims)[:k]

        topk = [
            {"label": str(labels[i]), "score": float(sims[i])}
            for i in idx
        ]
        best = topk[0]
        if best["score"] < self.threshold:
            return self.unknown_label, best["score"], topk
        return best["label"], best["score"], topk

    def predict(self, image_path: PathLike) -> tuple[str, float, list[dict[str, Any]]]:
        """对图片文件做识别：先提特征，再走 predict_feature。"""
        if self.extractor is None:
            raise RuntimeError("未提供 extractor，无法从图片提取特征")
        feature = self.extractor.extract(str(image_path))
        return self.predict_feature(feature)
