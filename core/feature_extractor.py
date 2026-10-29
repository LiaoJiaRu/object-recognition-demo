"""core/feature_extractor.py

基于 ImageNet 预训练 ResNet50 的特征提取器（阶段 3）。
- model.fc = nn.Identity()，去掉最后全连接层，输出 2048 维特征。
- 输出做 L2 归一化，之后余弦相似度可直接用点积计算。

本模块只负责特征提取；识别与特征库在后续阶段实现。
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

PathLike = Union[str, Path]


class FeatureExtractor:
    """ResNet50（去 FC）特征提取器，输出 L2 归一化的 2048 维 numpy 向量。"""

    def __init__(self, device: str = "cpu", image_size: int = 224) -> None:
        self.device = torch.device(device)
        self.image_size = image_size

        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Identity()
        self.model = model.eval().to(self.device)

        # ImageNet 标准预处理（整图缩放到短边 256 后中心裁剪到 image_size）
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def _load_tensor(self, image_path: PathLike) -> torch.Tensor:
        img = Image.open(image_path).convert("RGB")
        return self.preprocess(img)

    @staticmethod
    def _l2_normalize(feats: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        return (feats / norms).astype(np.float32)

    @torch.no_grad()
    def extract(self, image_path: PathLike) -> np.ndarray:
        """提取单张图片特征，返回 shape=(2048,) 的 L2 归一化向量。"""
        return self.extract_batch([image_path])[0]

    @torch.no_grad()
    def extract_batch(self, paths: Sequence[PathLike]) -> np.ndarray:
        """批量提取特征，返回 shape=(N, 2048) 的 L2 归一化矩阵。"""
        tensors = [self._load_tensor(p) for p in paths]
        batch = torch.stack(tensors).to(self.device)
        feats = self.model(batch).cpu().numpy()
        return self._l2_normalize(feats)
