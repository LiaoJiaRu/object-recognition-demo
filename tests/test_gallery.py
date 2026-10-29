"""tests/test_gallery.py

用 tmp_path + 伪特征提取器验证 FeatureGallery，不依赖真实数据与 torch。
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.gallery import FeatureGallery  # noqa: E402


class FakeExtractor:
    """按图片路径生成确定性的 512 维 L2 归一化向量。"""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def extract_batch(self, paths):
        vecs = []
        for p in paths:
            seed = int.from_bytes(hashlib.md5(str(p).encode("utf-8")).digest()[:4], "little")
            rng = np.random.RandomState(seed)
            v = rng.randn(self.dim).astype(np.float32)
            vecs.append(v / np.linalg.norm(v))
        return np.stack(vecs)


def make_cfg(data_root: Path) -> dict:
    return {
        "data_root": str(data_root),
        "feature_dir": "features",
        "feature_file": "gallery.npz",
        "labels_file": "labels.json",
    }


def make_gallery(data_root: Path) -> FeatureGallery:
    return FeatureGallery(make_cfg(data_root), extractor=FakeExtractor())


def test_add_and_get_all(tmp_path: Path) -> None:
    gallery = make_gallery(tmp_path)
    n = gallery.add("apple", [tmp_path / f"a{i}.jpg" for i in range(3)])
    assert n == 3
    feats, labels, paths = gallery.get_all()
    assert feats.shape == (3, 512)
    assert labels == ["apple", "apple", "apple"]
    assert len(paths) == 3
    # 特征应为 L2 归一化
    np.testing.assert_allclose(np.linalg.norm(feats, axis=1), 1.0, rtol=1e-5)


def test_same_name_multiple_images_accumulate(tmp_path: Path) -> None:
    gallery = make_gallery(tmp_path)
    gallery.add("cup", [tmp_path / "c1.jpg", tmp_path / "c2.jpg"])
    gallery.add("cup", [tmp_path / "c3.jpg"])
    gallery.add("banana", [tmp_path / "b1.jpg"])
    feats, labels, _ = gallery.get_all()
    assert feats.shape[0] == 4
    assert labels.count("cup") == 3
    assert gallery.counts() == {"cup": 3, "banana": 1}
    assert gallery.list() == ["cup", "banana"]
    assert len(gallery) == 4


def test_save_and_load_restart(tmp_path: Path) -> None:
    gallery = make_gallery(tmp_path)
    gallery.add("apple", [tmp_path / "a1.jpg", tmp_path / "a2.jpg"])
    gallery.add("cup", [tmp_path / "c1.jpg"])
    gallery.save()

    # 文件落在 data_root/features 下，文件名来自配置
    assert (tmp_path / "features" / "gallery.npz").exists()
    assert (tmp_path / "features" / "labels.json").exists()

    # 模拟程序重启：新实例 load 恢复
    restarted = FeatureGallery(make_cfg(tmp_path), extractor=FakeExtractor())
    restarted.load()
    feats1, labels1, paths1 = gallery.get_all()
    feats2, labels2, paths2 = restarted.get_all()
    assert labels2 == labels1
    assert paths2 == paths1
    np.testing.assert_allclose(feats1, feats2, rtol=1e-6)
    assert restarted.list() == ["apple", "cup"]
    assert restarted.counts() == {"apple": 2, "cup": 1}


def test_delete_removes_only_named_class(tmp_path: Path) -> None:
    gallery = make_gallery(tmp_path)
    gallery.add("apple", [tmp_path / "a1.jpg"])
    gallery.add("cup", [tmp_path / "c1.jpg", tmp_path / "c2.jpg"])
    removed = gallery.delete("cup")
    assert removed == 2
    feats, labels, _ = gallery.get_all()
    assert feats.shape == (1, 512)
    assert labels == ["apple"]
    # 删除不存在的类别返回 0，不报错
    assert gallery.delete("ghost") == 0


def test_delete_persists_after_reload(tmp_path: Path) -> None:
    gallery = make_gallery(tmp_path)
    gallery.add("apple", [tmp_path / "a1.jpg"])
    gallery.add("cup", [tmp_path / "c1.jpg"])
    gallery.delete("apple")
    gallery.save()

    restarted = FeatureGallery(make_cfg(tmp_path)).load()
    assert restarted.list() == ["cup"]


def test_load_missing_files_raises(tmp_path: Path) -> None:
    gallery = FeatureGallery(make_cfg(tmp_path))
    with pytest.raises(FileNotFoundError):
        gallery.load()


def test_add_without_extractor_raises(tmp_path: Path) -> None:
    gallery = FeatureGallery(make_cfg(tmp_path))
    with pytest.raises(RuntimeError):
        gallery.add("apple", [tmp_path / "a1.jpg"])


def test_add_empty_paths_is_noop(tmp_path: Path) -> None:
    gallery = make_gallery(tmp_path)
    assert gallery.add("apple", []) == 0
    feats, labels, _ = gallery.get_all()
    assert feats.shape == (0, 0)
    assert labels == []


def test_empty_gallery_save_load_roundtrip(tmp_path: Path) -> None:
    gallery = make_gallery(tmp_path)
    gallery.save()
    restarted = FeatureGallery(make_cfg(tmp_path)).load()
    feats, labels, paths = restarted.get_all()
    assert feats.shape[0] == 0
    assert labels == []
    assert paths == []


def test_save_twice_overwrite_then_reload(tmp_path: Path) -> None:
    """回归：对同一个 gallery.npz 连续覆盖保存（临时文件+原子替换）。

    背景：Windows 下目标文件曾被误设只读属性，np.savez 直接覆盖写报
    PermissionError [Errno 13]，导致录入只落盘了第一批类别。
    """
    gallery = make_gallery(tmp_path)
    gallery.add("apple", [tmp_path / "a1.jpg", tmp_path / "a2.jpg"])
    gallery.save()
    gallery.add("banana", [tmp_path / "b1.jpg"])
    gallery.save()  # 第二次保存覆盖同一 gallery.npz / labels.json

    restarted = FeatureGallery(make_cfg(tmp_path), extractor=FakeExtractor()).load()
    assert restarted.counts() == {"apple": 2, "banana": 1}
    # 保存失败时不允许留下残留临时文件
    assert not (tmp_path / "features" / "gallery.npz.tmp").exists()
    assert not (tmp_path / "features" / "labels.json.tmp").exists()


def test_save_over_readonly_file(tmp_path: Path) -> None:
    """回归：目标文件带只读属性时 save 应自动解除并成功写入。"""
    gallery = make_gallery(tmp_path)
    gallery.add("apple", [tmp_path / "a1.jpg"])
    gallery.save()

    npz = tmp_path / "features" / "gallery.npz"
    npz.chmod(npz.stat().st_mode & ~0o222)  # 置为只读（模拟 Windows 只读属性）

    gallery.add("banana", [tmp_path / "b1.jpg"])
    gallery.save()  # 只读文件不应阻塞覆盖写

    restarted = FeatureGallery(make_cfg(tmp_path), extractor=FakeExtractor()).load()
    assert restarted.counts() == {"apple": 1, "banana": 1}
