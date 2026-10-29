"""core/gallery.py

注册特征库（阶段 4）。
- 特征向量保存到 data_root/<feature_dir>/<feature_file>（gallery.npz）。
- 标签与图片路径保存到 data_root/<feature_dir>/<labels_file>（labels.json）。
- 同一个类别名可保存多张图片的多个特征。
- save() 落盘后，新实例通过 load() 可完整恢复，支持程序重启。

路径全部来自 config.yaml 对应的 cfg 字典，本模块不硬编码数据路径。
本模块只做特征库的增删存取，不实现识别器。
"""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Sequence, Union

import numpy as np

PathLike = Union[str, Path]


class FeatureGallery:
    """类别 -> 多张图片特征 的持久化特征库。

    构造参数:
        cfg: 配置字典，需包含 data_root，可含 feature_dir/feature_file/labels_file。
        extractor: 具备 extract_batch(paths) -> np.ndarray (N, D) 的对象，
                   通常是 core.feature_extractor.FeatureExtractor；仅 add 时需要。
    """

    def __init__(self, cfg: dict, extractor=None) -> None:
        self.cfg = cfg
        self.extractor = extractor

        data_root = Path(cfg["data_root"])
        self.feature_dir = data_root / cfg.get("feature_dir", "features")
        self.feature_path = self.feature_dir / cfg.get("feature_file", "gallery.npz")
        self.labels_path = self.feature_dir / cfg.get("labels_file", "labels.json")

        self._features: list[np.ndarray] = []
        self._labels: list[str] = []
        self._image_paths: list[str] = []

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    def add(self, name: str, image_paths: Sequence[PathLike]) -> int:
        """为类别 name 提取并追加多张图片的特征，返回新增特征条数。"""
        if not name:
            raise ValueError("类别名 name 不能为空")
        image_paths = list(image_paths)
        if not image_paths:
            return 0
        if self.extractor is None:
            raise RuntimeError("未提供 extractor，无法在线提取特征")

        feats = self.extractor.extract_batch([str(p) for p in image_paths])
        feats = np.asarray(feats, dtype=np.float32)
        if feats.ndim != 2 or feats.shape[0] != len(image_paths):
            raise ValueError(
                f"extractor.extract_batch 返回形状异常: {feats.shape}，"
                f"期望 ({len(image_paths)}, D)"
            )

        for path, vec in zip(image_paths, feats):
            self._features.append(vec.astype(np.float32, copy=False))
            self._labels.append(name)
            self._image_paths.append(str(path))
        return len(image_paths)

    def delete(self, name: str) -> int:
        """删除类别 name 的全部特征，返回删除条数；不存在时返回 0。"""
        kept = [
            (f, l, p)
            for f, l, p in zip(self._features, self._labels, self._image_paths)
            if l != name
        ]
        removed = len(self._labels) - len(kept)
        if removed:
            self._features = [x[0] for x in kept]
            self._labels = [x[1] for x in kept]
            self._image_paths = [x[2] for x in kept]
        return removed

    # ------------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------------
    def list(self) -> list[str]:
        """返回库中已注册的类别名（按首次加入顺序去重）。"""
        seen: dict[str, None] = {}
        for label in self._labels:
            seen.setdefault(label, None)
        return list(seen.keys())

    def counts(self) -> dict[str, int]:
        """返回每个类别名对应的特征条数。"""
        result: dict[str, int] = {}
        for label in self._labels:
            result[label] = result.get(label, 0) + 1
        return result

    def get_all(self) -> tuple[np.ndarray, list[str], list[str]]:
        """返回 (features(N, D), labels(N,), image_paths(N,))。空库时 features 形状 (0, 0)。"""
        if not self._features:
            empty = np.empty((0, 0), dtype=np.float32)
            return empty, [], []
        feats = np.stack(self._features).astype(np.float32, copy=False)
        return feats, list(self._labels), list(self._image_paths)

    def __len__(self) -> int:
        return len(self._labels)

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    def save(self) -> None:
        """将特征与标签落盘（npz + json 成对写入）。

        先写临时文件再原子替换（os.replace），保证读者不会看到写了一半的文件；
        替换前若目标文件带只读属性（Windows 下复制/同步工具常误设），自动解除，
        解除失败或仍被独占占用时抛出带排查提示的 PermissionError。
        """
        self.feature_dir.mkdir(parents=True, exist_ok=True)
        feats, labels, image_paths = self.get_all()

        npz_tmp = self.feature_path.with_name(self.feature_path.name + ".tmp")
        json_tmp = self.labels_path.with_name(self.labels_path.name + ".tmp")
        try:
            for dst in (self.feature_path, self.labels_path):
                if dst.exists():
                    try:
                        os.chmod(dst, stat.S_IREAD | stat.S_IWRITE)
                    except OSError:
                        pass  # 解除失败时交给 os.replace 抛出可诊断的错误
            with open(npz_tmp, "wb") as f:
                np.savez(f, features=feats)
            with open(json_tmp, "w", encoding="utf-8") as f:
                json.dump(
                    {"labels": labels, "image_paths": image_paths},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            os.replace(npz_tmp, self.feature_path)
            os.replace(json_tmp, self.labels_path)
        except PermissionError as exc:
            for p in (npz_tmp, json_tmp):
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    pass
            raise PermissionError(
                f"无法写入特征库 {self.feature_path}：文件被只读属性或"
                f"其他程序占用（常见：另一个 app.py 实例、IDE 预览了 npz、"
                f"杀毒软件/同步盘正在扫描）。请解除占用后重试。原始错误: {exc}"
            ) from exc

    def load(self) -> "FeatureGallery":
        """从磁盘恢复特征库状态（要求 npz 与 json 成对存在）。"""
        if not self.feature_path.exists() or not self.labels_path.exists():
            raise FileNotFoundError(
                f"特征库文件缺失: {self.feature_path} / {self.labels_path}"
            )

        with np.load(self.feature_path) as data:
            feats = data["features"]
        with open(self.labels_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        labels = meta["labels"]
        image_paths = meta["image_paths"]

        if feats.shape[0] != len(labels) or len(labels) != len(image_paths):
            raise ValueError(
                f"特征库内容不一致: features={feats.shape[0]} 条, "
                f"labels={len(labels)} 条, image_paths={len(image_paths)} 条"
            )

        self._features = [row.astype(np.float32, copy=False) for row in feats]
        self._labels = list(labels)
        self._image_paths = list(image_paths)
        return self
