"""scripts/build_gallery.py

离线构建持久化特征库（gallery.npz + labels.json）。

- 从 config.yaml 读取 data_root / gallery_dir / feature_dir / feature_file / labels_file。
- 扫描 gallery_dir 下所有类别文件夹，用 core.feature_extractor 提特征。
- 调用 core.gallery.FeatureGallery.save() 写入（原子替换 + 自动解除只读属性，沿用既有逻辑）。
- 打印类别数、每类图片数、总特征数、输出路径。

用法:
    python scripts/build_gallery.py [--config config.yaml]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.feature_extractor import FeatureExtractor  # noqa: E402
from core.gallery import FeatureGallery  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".jfif", ".webp"}
DEFAULT_CONFIG_PATH = ROOT / "config.yaml"


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build(cfg: dict) -> tuple[FeatureGallery, Path]:
    """扫描 gallery_dir，提特征，save()，返回 (gallery, 输出 npz 路径)。"""
    gallery_path = Path(cfg["data_root"]) / cfg.get("gallery_dir", "gallery")
    if not gallery_path.exists():
        raise FileNotFoundError(f"gallery 目录不存在: {gallery_path}")

    extractor = FeatureExtractor(
        device=cfg.get("device", "cpu"),
        image_size=cfg.get("image_size", 224),
    )
    gallery = FeatureGallery(cfg, extractor=extractor)

    class_dirs = sorted(p for p in gallery_path.iterdir() if p.is_dir())
    if not class_dirs:
        raise RuntimeError(f"gallery 目录下没有任何类别文件夹: {gallery_path}")

    per_class: dict[str, int] = {}
    for cls_dir in class_dirs:
        imgs = sorted(
            p for p in cls_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        )
        if not imgs:
            print(f"[skip] {cls_dir.name}: 无 .jpg/.jpeg/.png 图片")
            continue
        gallery.add(cls_dir.name, imgs)
        per_class[cls_dir.name] = len(imgs)

    gallery.save()
    return gallery, gallery.feature_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="离线构建持久化特征库")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="config.yaml 路径")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    gallery, out_path = build(cfg)

    total = len(gallery)
    print(f"[build] 输出: {out_path}")
    print(f"[build] 类别数: {len(gallery.list())}")
    counts = gallery.counts()
    for cls in sorted(counts.keys()):
        print(f"  - {cls}: {counts[cls]}")
    print(f"[build] 总特征数: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
