"""scripts/evaluate_videos.py

视频测试集自动评估（视频扩展，消费 config.yaml 的视频配置项）。

- 配置项来源（此前无代码消费，本脚本使其生效）：
  * video_test_dir  视频测试目录：<data_root>/<video_test_dir>/<类别名>/*.mp4
  * frame_interval  抽帧间隔（秒）
  * max_frames      单个视频最多抽帧数（0 表示不限制）
- 每个视频：core.video.recognize_video 抽帧
  -> core.webcam.recognize_frame 逐帧识别
  -> core.video.aggregate_results 多数投票，得到整段视频的 label；
- 真值规则：视频所在子文件夹名即类别名（与图片 test 目录一致）；
- 输出：<data_root>/<report_dir>/video_eval_report.json，
  含视频级 top-1 准确率、逐类统计与逐视频明细。

用法:
    python scripts/evaluate_videos.py [--config config.yaml]
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
from core.video import recognize_video  # noqa: E402
from core.webcam import recognize_frame  # noqa: E402

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_frame_fn(recognizer: Recognizer, extractor: FeatureExtractor):
    """构造单帧识别回调：BGR 帧 -> (label, score, topk)。"""
    return lambda frame_bgr: recognize_frame(recognizer, extractor, frame_bgr)


def evaluate_videos(cfg: dict, frame_fn=None) -> dict:
    """评估 video_test_dir 下全部视频，返回报告字典。

    frame_fn 为 None 时用真实模型链路现场构建（ResNet18 特征库 + 检索器）；
    测试可注入假 frame_fn 以脱离 torch / 模型权重依赖。
    """
    video_root = Path(cfg["data_root"]) / cfg.get("video_test_dir", "test_videos")
    if not video_root.is_dir():
        raise FileNotFoundError(f"视频测试目录不存在: {video_root}")

    interval = float(cfg.get("frame_interval", 1.0))
    max_frames = int(cfg.get("max_frames", 0))
    unknown_label = cfg.get("unknown_label", "unknown")

    if frame_fn is None:
        extractor = FeatureExtractor(
            device=cfg.get("device", "cpu"),
            image_size=cfg.get("image_size", 224),
        )
        gallery = FeatureGallery(cfg, extractor=extractor)
        for class_dir in sorted(p for p in (Path(cfg["data_root"]) / "gallery").iterdir() if p.is_dir()):
            images = sorted(
                p for p in class_dir.iterdir()
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".jfif", ".webp"}
            )
            if images:
                gallery.add(class_dir.name, images)
        if len(gallery) == 0:
            raise RuntimeError("gallery 目录下没有可用注册图，无法构建特征库")
        recognizer = Recognizer(gallery, extractor=extractor, cfg=cfg)
        frame_fn = make_frame_fn(recognizer, extractor)

    details: list[dict] = []
    for class_dir in sorted(p for p in video_root.iterdir() if p.is_dir()):
        truth = class_dir.name
        videos = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS
        )
        for video_path in videos:
            try:
                result = recognize_video(
                    video_path,
                    frame_fn=frame_fn,
                    interval_sec=interval,
                    max_frames=max_frames,
                )
            except (FileNotFoundError, ValueError) as exc:
                # 打不开 / 0 帧 / 损坏视频：记为读取失败而不是中断整体评估
                details.append({
                    "video": str(video_path), "truth": truth,
                    "pred": "read_error", "score": 0.0, "ok": False,
                    "n_frames": 0, "error": str(exc),
                })
                continue
            details.append({
                "video": str(video_path),
                "truth": truth,
                "pred": result["label"],
                "score": float(result["score"]),
                "votes": result["votes"],
                "n_frames": result["n_sampled"],
                "ok": result["label"] == truth,
            })

    if not details:
        raise FileNotFoundError(
            f"{video_root} 下没有找到任何视频"
            f"（支持 {', '.join(sorted(VIDEO_EXTS))}，按 <类别名>/<视频> 组织）"
        )

    total = len(details)
    correct = sum(1 for d in details if d["ok"])
    per_class: dict[str, dict] = {}
    for d in details:
        c = per_class.setdefault(d["truth"], {"total": 0, "correct": 0})
        c["total"] += 1
        c["correct"] += 1 if d["ok"] else 0

    return {
        "data_root": str(Path(cfg["data_root"])),
        "video_test_dir": str(video_root),
        "frame_interval": interval,
        "max_frames": max_frames,
        "unknown_label": unknown_label,
        "video_total": total,
        "video_correct": correct,
        "video_top1_acc": correct / total,
        "per_class": per_class,
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="视频测试集自动评估")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    report = evaluate_videos(cfg)

    report_path = (
        Path(cfg["data_root"]) / cfg.get("report_dir", "reports") / "video_eval_report.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"video_test_dir : {report['video_test_dir']}")
    print(
        f"video_top1_acc = {report['video_top1_acc']:.4f} "
        f"({report['video_correct']}/{report['video_total']})"
    )
    for name, c in report["per_class"].items():
        print(f"  - {name}: {c['correct']}/{c['total']}")
    print(f"[report] {report_path}")


if __name__ == "__main__":
    main()
