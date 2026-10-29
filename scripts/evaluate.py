"""scripts/evaluate.py

自动评估脚本（阶段 6）。
- 从 config.yaml 读取 data_root / threshold / topk 等全部路径与参数。
- 遍历 data_root/<test_dir> 下所有图片（递归）。
- 真值规则（永远来自文件夹名，不读 data_index.json）：
  * 路径中包含 unknown 文件夹（含其子文件夹）-> "unknown"；
  * 否则取图片的直接父文件夹名。
- 用 gallery 注册图现场构建特征库，调用 Recognizer 预测。
- 输出 data_root/<report_dir>/eval_report.json：
  registered_acc / unknown_recall / unknown_false_accept / details。

用法:
    python scripts/evaluate.py [--config config.yaml]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.feature_extractor import FeatureExtractor  # noqa: E402
from core.gallery import FeatureGallery  # noqa: E402
from core.recognizer import Recognizer  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".jfif", ".webp"}
DEFAULT_CONFIG_PATH = ROOT / "config.yaml"


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_truth(image_path: Path, test_path: Path, unknown_label: str = "unknown") -> str:
    """按文件夹名返回真值。

    路径中任意一段等于 unknown_label -> unknown（覆盖 unknown 下嵌套子文件夹）；
    否则取直接父文件夹名。
    """
    rel = Path(image_path).resolve().relative_to(Path(test_path).resolve())
    if unknown_label in rel.parts[:-1]:
        return unknown_label
    return Path(image_path).parent.name


def iter_test_images(test_path: Path, unknown_label: str = "unknown") -> list[tuple[Path, str]]:
    """递归列出 test 下全部图片及其真值，返回 [(path, truth), ...]，按路径排序。"""
    items = []
    for p in sorted(test_path.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            items.append((p, get_truth(p, test_path, unknown_label)))
    return items


def build_gallery(cfg: dict, extractor: FeatureExtractor) -> FeatureGallery:
    """构建评估用特征库：优先加载持久化 gallery.npz，不存在或为空时回退到现场扫描 gallery_dir。"""
    gallery = FeatureGallery(cfg, extractor=extractor)

    # --- 优先尝试加载持久化特征库 ---
    feat_path = gallery.feature_path
    labels_path = gallery.labels_path
    if feat_path.exists() and labels_path.exists():
        try:
            gallery.load()
        except Exception as exc:
            print(f"[warn] 持久化特征库损坏，回退现场重建: {exc}")
        else:
            if len(gallery) > 0:
                print(f"[info] 加载持久化特征库: {feat_path} ({len(gallery)} 条)")
                return gallery
            print(f"[warn] 持久化特征库为空（labels.json 条目={len(gallery)}），回退现场重建")
    else:
        print(f"[warn] 未找到持久化特征库 (需同时存在 {feat_path} 和 {labels_path})，现场重建")

    # --- 回退: 现场扫描 gallery_dir ---
    gallery_path = Path(cfg["data_root"]) / cfg.get("gallery_dir", "gallery")
    if not gallery_path.exists():
        raise FileNotFoundError(f"gallery 目录不存在: {gallery_path}")
    for cls_dir in sorted(p for p in gallery_path.iterdir() if p.is_dir()):
        imgs = sorted(p for p in cls_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        if imgs:
            gallery.add(cls_dir.name, imgs)
    return gallery


def evaluate(cfg: dict, recognizer: Recognizer | None = None) -> dict:
    """执行评估并返回报告字典。recognizer 为 None 时现场构建模型与特征库。"""
    if recognizer is None:
        extractor = FeatureExtractor(
            device=cfg.get("device", "cpu"),
            image_size=cfg.get("image_size", 224),
        )
        gallery = build_gallery(cfg, extractor)
        recognizer = Recognizer(gallery, extractor=extractor, cfg=cfg)

    data_root = Path(cfg["data_root"])
    test_path = data_root / cfg.get("test_dir", "test")
    unknown_label = cfg.get("unknown_label", "unknown")
    if not test_path.exists():
        raise FileNotFoundError(f"test 目录不存在: {test_path}")

    details = []
    reg_total = reg_correct = 0
    unk_total = unk_correct = 0

    for image_path, truth in iter_test_images(test_path, unknown_label):
        pred, score, _ = recognizer.predict(str(image_path))
        ok = pred == truth
        details.append(
            {
                "path": str(image_path),
                "truth": truth,
                "pred": pred,
                "score": round(float(score), 6),
                "ok": bool(ok),
            }
        )
        if truth == unknown_label:
            unk_total += 1
            if pred == unknown_label:
                unk_correct += 1
        else:
            reg_total += 1
            if ok:
                reg_correct += 1

    registered_acc = reg_correct / reg_total if reg_total else 0.0
    unknown_recall = unk_correct / unk_total if unk_total else 0.0
    unk_false_accept = (
        (unk_total - unk_correct) / unk_total if unk_total else 0.0
    )

    return {
        "threshold": float(cfg.get("threshold", 0.75)),
        "topk": int(cfg.get("topk", 5)),
        "data_root": str(data_root),
        "registered_acc": registered_acc,
        "unknown_recall": unknown_recall,
        "unknown_false_accept": unk_false_accept,
        "counts": {
            "registered_total": reg_total,
            "registered_correct": reg_correct,
            "unknown_total": unk_total,
            "unknown_correct": unk_correct,
            "unknown_false_accept_count": unk_total - unk_correct,
            "total": reg_total + unk_total,
        },
        "details": details,
    }


def write_report(cfg: dict, report: dict) -> Path:
    report_dir = Path(cfg["data_root"]) / cfg.get("report_dir", "reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    out = report_dir / cfg.get("report_file", "eval_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="物体识别自动评估")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="config.yaml 路径")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    report = evaluate(cfg)
    out = write_report(cfg, report)

    c = report["counts"]
    print(f"data_root: {report['data_root']}")
    print(f"threshold: {report['threshold']}  topk: {report['topk']}")
    print(
        f"registered_acc      = {report['registered_acc']:.4f} "
        f"({c['registered_correct']}/{c['registered_total']})"
    )
    print(
        f"unknown_recall      = {report['unknown_recall']:.4f} "
        f"({c['unknown_correct']}/{c['unknown_total']})"
    )
    print(
        f"unknown_false_accept = {report['unknown_false_accept']:.4f} "
        f"({c['unknown_false_accept_count']}/{c['unknown_total']})"
    )
    print(f"[report] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
