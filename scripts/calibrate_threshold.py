"""scripts/calibrate_threshold.py

阈值标定与可行性判断（只读脚本）。
- 复用持久化特征库 data_root/<feature_dir>/gallery.npz + labels.json，不现场重建；
  库不存在或为空时直接报错，提示先跑 scripts/build_gallery.py。
- 对每张测试图只提一次特征，缓存 top-1 相似度与 top-1 标签（避免逐阈值重复提特征）。
- 扫描 threshold t 从 0.55 到 0.85（步长 0.01），统计：
    registered_acc = (真值≠unknown 且 pred==真值) / (真值≠unknown 总数)
    unknown_recall = (真值==unknown 且 pred==unknown) / (真值==unknown 总数)
- 打印表格、可行区间（两指标同时 >= 0.90）、当前 threshold 行、实际使用的 IMG_EXTS。
- 只写 data_root/<report_dir>/threshold_calibration.json，不写 gallery.npz / labels.json。

用法:
    python scripts/calibrate_threshold.py [--config config.yaml]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.feature_extractor import FeatureExtractor  # noqa: E402
from core.gallery import FeatureGallery  # noqa: E402

# 与第 1.5 步同步后的全项目统一图片扩展名（5 元素，不含 .bmp）
IMG_EXTS = {".jpg", ".jpeg", ".png", ".jfif", ".webp"}
DEFAULT_CONFIG_PATH = ROOT / "config.yaml"

T_MIN, T_MAX, T_STEP = 0.55, 0.85, 0.01
MIN_ACC = 0.90
CALIB_FILE = "threshold_calibration.json"


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_truth(image_path: Path, test_path: Path, unknown_label: str) -> str:
    """真值规则与 evaluate.py 保持一致：路径含 unknown 段 -> unknown；否则父文件夹名。"""
    rel = Path(image_path).resolve().relative_to(Path(test_path).resolve())
    if unknown_label in rel.parts[:-1]:
        return unknown_label
    return Path(image_path).parent.name


def iter_test_images(test_path: Path) -> list[Path]:
    """递归列出 test 下全部图片（按 IMG_EXTS 过滤），按路径排序。"""
    return [
        p for p in sorted(test_path.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ]


def load_persistent_gallery(cfg: dict) -> tuple[FeatureGallery, FeatureExtractor]:
    """加载持久化特征库；不存在或为空则报错。"""
    extractor = FeatureExtractor(
        device=cfg.get("device", "cpu"),
        image_size=cfg.get("image_size", 224),
    )
    gallery = FeatureGallery(cfg, extractor=extractor)
    feat_path = gallery.feature_path
    labels_path = gallery.labels_path
    if not (feat_path.exists() and labels_path.exists()):
        raise FileNotFoundError(
            f"持久化特征库不存在（需同时有 {feat_path} 和 {labels_path}）；"
            f"请先运行: python scripts/build_gallery.py"
        )
    gallery.load()
    if len(gallery) == 0:
        raise RuntimeError(
            f"持久化特征库为空（{feat_path}）；请先运行: python scripts/build_gallery.py"
        )
    return gallery, extractor


def collect_scores(cfg: dict) -> tuple[list[dict], int]:
    """对每张测试图提一次特征，记录 (truth, top1_label, score)。"""
    gallery, extractor = load_persistent_gallery(cfg)

    feats, labels, _ = gallery.get_all()
    feats = np.asarray(feats, dtype=np.float32)
    labels = np.asarray(labels, dtype=object)
    if feats.size == 0:
        raise RuntimeError("特征库无特征，无法标定")

    data_root = Path(cfg["data_root"])
    test_path = data_root / cfg.get("test_dir", "test")
    unknown_label = cfg.get("unknown_label", "unknown")
    if not test_path.exists():
        raise FileNotFoundError(f"test 目录不存在: {test_path}")

    records = []
    for img in iter_test_images(test_path):
        feature = extractor.extract(str(img))
        v = np.asarray(feature, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(v))
        if norm < 1e-12:
            raise ValueError(f"特征为零向量: {img}")
        sims = feats @ (v / norm)
        i = int(np.argmax(sims))
        records.append(
            {
                "path": str(img),
                "truth": get_truth(img, test_path, unknown_label),
                "top1_label": str(labels[i]),
                "score": float(sims[i]),
            }
        )
    return records, len(feats)


def scan_thresholds(records: list[dict], unknown_label: str) -> list[dict]:
    """按阈值扫描统计，特征只提过一次。"""
    steps = int(round((T_MAX - T_MIN) / T_STEP)) + 1
    rows = []
    for i in range(steps):
        t = round(T_MIN + i * T_STEP, 2)
        reg_total = reg_correct = unk_total = unk_correct = 0
        for r in records:
            pred = unknown_label if r["score"] < t else r["top1_label"]
            if r["truth"] == unknown_label:
                unk_total += 1
                if pred == unknown_label:
                    unk_correct += 1
            else:
                reg_total += 1
                if pred == r["truth"]:
                    reg_correct += 1
        rows.append(
            {
                "threshold": t,
                "registered_acc": reg_correct / reg_total if reg_total else 0.0,
                "unknown_recall": unk_correct / unk_total if unk_total else 0.0,
                "registered_correct": reg_correct,
                "registered_total": reg_total,
                "unknown_correct": unk_correct,
                "unknown_total": unk_total,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="阈值标定与可行性判断（只读）")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="config.yaml 路径")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    unknown_label = cfg.get("unknown_label", "unknown")
    current_threshold = round(float(cfg.get("threshold", 0.75)), 2)

    records, n_features = collect_scores(cfg)
    rows = scan_thresholds(records, unknown_label)

    feasible = [r["threshold"] for r in rows
                if r["registered_acc"] >= MIN_ACC and r["unknown_recall"] >= MIN_ACC]

    print(f"IMG_EXTS = {sorted(IMG_EXTS)}")
    print(f"data_root: {cfg['data_root']}")
    print(f"持久化特征库: {n_features} 条特征；测试图: {len(records)} 张")
    print(f"扫描阈值: {T_MIN}~{T_MAX} 步长 {T_STEP}；同时达标线: {MIN_ACC}")
    print("")
    print(f"{'threshold':>9} | {'registered_acc':>14} | {'unknown_recall':>14} | "
          f"{'reg_correct/total':>17} | {'unk_correct/total':>17}")
    print("-" * 92)
    current_row = None
    for r in rows:
        print(f"{r['threshold']:>9.2f} | {r['registered_acc']:>14.4f} | {r['unknown_recall']:>14.4f} | "
              f"{r['registered_correct']:>7}/{r['registered_total']:<9} | "
              f"{r['unknown_correct']:>7}/{r['unknown_total']:<9}")
        if abs(r["threshold"] - current_threshold) < 1e-9:
            current_row = r
    print("-" * 92)

    if feasible:
        t_lo, t_hi = min(feasible), max(feasible)
        contiguous = (len(feasible) == int(round((t_hi - t_lo) / T_STEP)) + 1)
        print(f"FEASIBLE INTERVAL: [{t_lo:.2f}, {t_hi:.2f}]"
              f"（同时 >= {MIN_ACC} 的阈值共 {len(feasible)} 个"
              f"{'，连续' if contiguous else '，不连续：' + str(feasible)}）")
    else:
        print(f"NO FEASIBLE INTERVAL（不存在使 registered_acc 与 unknown_recall 同时 >= {MIN_ACC} 的阈值）")

    print("")
    if current_row is not None:
        print(f"当前 threshold={current_threshold:.2f}: "
              f"registered_acc={current_row['registered_acc']:.4f} "
              f"({current_row['registered_correct']}/{current_row['registered_total']}), "
              f"unknown_recall={current_row['unknown_recall']:.4f} "
              f"({current_row['unknown_correct']}/{current_row['unknown_total']})")

    report = {
        "data_root": str(cfg["data_root"]),
        "gallery_feature_path": str(Path(cfg["data_root"]) / cfg.get("feature_dir", "features")
                                    / cfg.get("feature_file", "gallery.npz")),
        "img_exts": sorted(IMG_EXTS),
        "topk": int(cfg.get("topk", 5)),
        "unknown_label": unknown_label,
        "min_acc": MIN_ACC,
        "t_min": T_MIN,
        "t_max": T_MAX,
        "t_step": T_STEP,
        "feature_count": n_features,
        "test_image_count": len(records),
        "current_threshold": current_threshold,
        "current": current_row,
        "scan": rows,
        "feasible": {
            "interval": [min(feasible), max(feasible)] if feasible else None,
            "thresholds": feasible,
        },
        "records": records,
    }

    report_dir = Path(cfg["data_root"]) / cfg.get("report_dir", "reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    out = report_dir / CALIB_FILE
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[report] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
