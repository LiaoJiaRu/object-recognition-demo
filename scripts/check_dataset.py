"""数据集组织校验脚本。

从 config.yaml 读取 data_root，检查 gallery / test / test-unknown 的组织是否符合要求。
- 真值永远来自文件夹名；test/<unknown_label>/ 下所有图片（含子文件夹）一律视为 unknown。
- data_root/reports/data_index.json 仅为记录快照，不作为真值来源。

退出码：检查通过返回 0，否则返回 1。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".jfif", ".webp"}
DEFAULT_CONFIG_PATH = "config.yaml"

# 检查阈值
GALLERY_MIN_CLASSES = 5
GALLERY_MIN_PER_CLASS = 5
TEST_MIN_REGISTERED_CLASSES = 5
TEST_MIN_PER_CLASS = 1
UNKNOWN_MIN_IMAGES = 3


def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    """读取 config.yaml（扁平字段）。"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _list_images(directory: Path, recursive: bool = False) -> list[Path]:
    """列出目录下支持的图片文件。recursive=True 时包含子目录（用于 unknown）。"""
    if not directory.exists():
        return []
    it = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in IMG_EXTS)


def check_dataset(cfg: dict) -> tuple[bool, dict, list[str]]:
    """纯检查函数，不写文件。

    返回 (ok, stats, errors)。ok 为 True 表示全部硬性检查通过。
    """
    errors: list[str] = []
    warnings: list[str] = []

    data_root = Path(cfg["data_root"])
    gallery_dir = cfg.get("gallery_dir", "gallery")
    test_dir = cfg.get("test_dir", "test")
    unknown_label = cfg.get("unknown_label", "unknown")

    gallery_path = data_root / gallery_dir
    test_path = data_root / test_dir
    unknown_path = test_path / unknown_label

    # --- gallery ---
    if not gallery_path.exists():
        errors.append(f"gallery 目录不存在: {gallery_path}")
        gallery_classes: list[str] = []
        gallery_counts: dict[str, int] = {}
    else:
        gallery_classes = sorted(p.name for p in gallery_path.iterdir() if p.is_dir())
        gallery_counts = {c: len(_list_images(gallery_path / c)) for c in gallery_classes}

    # --- test（非 unknown）---
    if not test_path.exists():
        errors.append(f"test 目录不存在: {test_path}")
        test_classes_all: list[str] = []
        test_counts: dict[str, int] = {}
        registered_test: list[str] = []
        unregistered_test: list[str] = []
    else:
        test_classes_all = sorted(
            p.name for p in test_path.iterdir() if p.is_dir() and p.name != unknown_label
        )
        test_counts = {c: len(_list_images(test_path / c)) for c in test_classes_all}
        registered_test = [c for c in test_classes_all if c in gallery_classes]
        unregistered_test = [c for c in test_classes_all if c not in gallery_classes]

    # --- unknown（递归，子文件夹也算）---
    unknown_imgs = _list_images(unknown_path, recursive=True)
    if not unknown_path.exists():
        errors.append(f"test/unknown 目录不存在: {unknown_path}")

    # --- 硬性检查 ---
    if len(gallery_classes) < GALLERY_MIN_CLASSES:
        errors.append(
            f"gallery 类别数 {len(gallery_classes)} < {GALLERY_MIN_CLASSES}"
        )
    for c, n in gallery_counts.items():
        if n < GALLERY_MIN_PER_CLASS:
            errors.append(
                f"gallery 类别 '{c}' 图片数 {n} < {GALLERY_MIN_PER_CLASS}"
            )
    if len(registered_test) < TEST_MIN_REGISTERED_CLASSES:
        errors.append(
            f"test 已注册类别数 {len(registered_test)} < {TEST_MIN_REGISTERED_CLASSES}"
        )
    for c in registered_test:
        if test_counts.get(c, 0) < TEST_MIN_PER_CLASS:
            errors.append(f"test 已注册类别 '{c}' 图片数 < {TEST_MIN_PER_CLASS}")
    if len(unknown_imgs) < UNKNOWN_MIN_IMAGES:
        errors.append(
            f"test/unknown 图片数 {len(unknown_imgs)} < {UNKNOWN_MIN_IMAGES}"
        )

    # --- 软警告（不改变退出码）---
    if unregistered_test:
        warnings.append(
            f"test 存在未注册类别（不在 gallery，评估时会被判 unknown）: {unregistered_test}"
        )

    stats = {
        "data_root": str(data_root),
        "gallery": {
            "num_classes": len(gallery_classes),
            "per_class": gallery_counts,
        },
        "test": {
            "num_classes": len(test_classes_all),
            "per_class": test_counts,
            "registered_classes": registered_test,
            "unregistered_classes": unregistered_test,
        },
        "unknown": {"num_images": len(unknown_imgs)},
        "thresholds": {
            "gallery_min_classes": GALLERY_MIN_CLASSES,
            "gallery_min_per_class": GALLERY_MIN_PER_CLASS,
            "test_min_registered_classes": TEST_MIN_REGISTERED_CLASSES,
            "test_min_per_class": TEST_MIN_PER_CLASS,
            "unknown_min_images": UNKNOWN_MIN_IMAGES,
        },
        "warnings": warnings,
        "note": "真值来自文件夹名；本文件仅为记录，不作为真值来源。",
    }

    return (len(errors) == 0), stats, errors


def write_index(cfg: dict, stats: dict) -> Path | None:
    """将统计快照写入 data_root/<report_dir>/data_index.json（仅记录）。"""
    data_root = Path(cfg["data_root"])
    if not data_root.exists():
        return None
    report_dir = data_root / cfg.get("report_dir", "reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    out = report_dir / "data_index.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return out


def print_stats(stats: dict) -> None:
    print(f"data_root: {stats['data_root']}")
    g = stats["gallery"]
    print(f"[gallery] 类别数={g['num_classes']}")
    for c, n in g["per_class"].items():
        print(f"  - {c}: {n}")
    t = stats["test"]
    print(
        f"[test] 类别数={t['num_classes']} 已注册类别数={len(t['registered_classes'])}"
    )
    for c, n in t["per_class"].items():
        mark = "" if c in t["registered_classes"] else "  (未注册)"
        print(f"  - {c}: {n}{mark}")
    print(f"[unknown] 图片数={stats['unknown']['num_images']}")
    for w in stats.get("warnings", []):
        print(f"[WARN] {w}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="校验数据集组织")
    ap.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="config.yaml 路径")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    ok, stats, errors = check_dataset(cfg)
    print_stats(stats)

    idx = write_index(cfg, stats)
    if idx is not None:
        print(f"[record] {idx}")
    else:
        print("[record] data_root 不存在，跳过 data_index.json 写入")

    if errors:
        print("\n[FAIL] 检查未通过:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("\n[OK] 数据集检查通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
