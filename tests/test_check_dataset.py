"""tests/test_check_dataset.py

用 tmp_path 构造合成数据，验证 check_dataset 的检查逻辑。
不依赖真实数据根目录或麦克风/网络。
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

# 让测试能导入 scripts/check_dataset.py
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_dataset as cd  # noqa: E402


def make_cfg(data_root: Path) -> dict:
    return {
        "data_root": str(data_root),
        "gallery_dir": "gallery",
        "test_dir": "test",
        "report_dir": "reports",
        "unknown_label": "unknown",
    }


def add_imgs(d: Path, n: int, ext: str = ".jpg") -> None:
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (d / f"img_{i}{ext}").write_bytes(b"")


def build_valid(root: Path) -> None:
    """构造一份通过全部检查的合成数据集。"""
    classes = ["c1", "c2", "c3", "c4", "c5"]
    for c in classes:
        add_imgs(root / "gallery" / c, 5)
        add_imgs(root / "test" / c, 1)
    add_imgs(root / "test" / "unknown", 3)


def test_valid_dataset(tmp_path: Path) -> None:
    build_valid(tmp_path)
    ok, stats, errors = cd.check_dataset(make_cfg(tmp_path))
    assert ok, errors
    assert stats["gallery"]["num_classes"] == 5
    assert stats["unknown"]["num_images"] == 3
    assert len(stats["test"]["registered_classes"]) == 5


def test_gallery_too_few_classes(tmp_path: Path) -> None:
    build_valid(tmp_path)
    shutil.rmtree(tmp_path / "gallery" / "c5")
    ok, stats, errors = cd.check_dataset(make_cfg(tmp_path))
    assert not ok
    assert any("gallery 类别数 4" in e for e in errors)


def test_gallery_class_too_few_images(tmp_path: Path) -> None:
    build_valid(tmp_path)
    # c1: 5 -> 4
    (list((tmp_path / "gallery" / "c1").glob("*"))[0]).unlink()
    ok, stats, errors = cd.check_dataset(make_cfg(tmp_path))
    assert not ok
    assert any("c1' 图片数 4 < 5" in e for e in errors)


def test_registered_test_too_few(tmp_path: Path) -> None:
    build_valid(tmp_path)
    shutil.rmtree(tmp_path / "test" / "c5")
    ok, stats, errors = cd.check_dataset(make_cfg(tmp_path))
    assert not ok
    assert any("已注册类别数 4" in e for e in errors)


def test_unknown_too_few(tmp_path: Path) -> None:
    build_valid(tmp_path)
    # unknown: 3 -> 2
    (list((tmp_path / "test" / "unknown").glob("*"))[0]).unlink()
    ok, stats, errors = cd.check_dataset(make_cfg(tmp_path))
    assert not ok
    assert any("unknown 图片数 2 < 3" in e for e in errors)


def test_unknown_missing_dir(tmp_path: Path) -> None:
    build_valid(tmp_path)
    shutil.rmtree(tmp_path / "test" / "unknown")
    ok, stats, errors = cd.check_dataset(make_cfg(tmp_path))
    assert not ok
    assert any("unknown 目录不存在" in e for e in errors)


def test_extension_filter(tmp_path: Path) -> None:
    build_valid(tmp_path)
    # 非图片文件应被忽略，不影响计数
    (tmp_path / "gallery" / "c1" / "readme.txt").write_text("x")
    (tmp_path / "gallery" / "c1" / "notes.md").write_text("x")
    ok, stats, errors = cd.check_dataset(make_cfg(tmp_path))
    assert ok, errors
    assert stats["gallery"]["per_class"]["c1"] == 5


def test_extension_variants(tmp_path: Path) -> None:
    """.jpg/.jpeg/.png 都应被计入。"""
    build_valid(tmp_path)
    d = tmp_path / "gallery" / "c1"
    # 删掉默认 5 张，换成混合后缀
    for f in list(d.glob("*")):
        f.unlink()
    (d / "a.jpg").write_bytes(b"")
    (d / "b.JPEG").write_bytes(b"")  # 大写后缀
    (d / "c.png").write_bytes(b"")
    (d / "d.jpeg").write_bytes(b"")
    (d / "e.bmp").write_bytes(b"")  # 不支持，应忽略 -> 计数 4 < 5 触发失败
    ok, stats, errors = cd.check_dataset(make_cfg(tmp_path))
    assert not ok
    assert stats["gallery"]["per_class"]["c1"] == 4


def test_unknown_recursive(tmp_path: Path) -> None:
    """test/unknown 子文件夹内的图片也应计入（递归）。"""
    build_valid(tmp_path)
    unk = tmp_path / "test" / "unknown"
    sub = unk / "sub"
    sub.mkdir()
    for f in list(unk.glob("*.jpg")):
        f.rename(sub / f.name)
    ok, stats, errors = cd.check_dataset(make_cfg(tmp_path))
    assert ok, errors
    assert stats["unknown"]["num_images"] == 3


def test_unregistered_test_is_warning(tmp_path: Path) -> None:
    """test 中存在未注册类别应为警告，不改变退出码。"""
    build_valid(tmp_path)
    add_imgs(tmp_path / "test" / "ghost", 2)  # 不在 gallery
    ok, stats, errors = cd.check_dataset(make_cfg(tmp_path))
    assert ok, errors
    assert any("未注册类别" in w for w in stats["warnings"])
