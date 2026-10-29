"""webcam_app.py

摄像头实时识别（阶段 10，加分项）。
- OpenCV 单线程同步采集（避免多线程竞争摄像头导致黑屏）。
- 每 N 帧（默认 5）识别一次，降低 CPU 压力；其余帧沿用上一次结果。
- 画面左上角显示 label / score，unknown 为红色、已知类别为绿色。
- 按 q 退出。
- 识别仍调用 core.webcam.recognize_frame -> Recognizer.predict_feature。

用法:
    python webcam_app.py [--camera 0] [--every 5] [--config config.yaml]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.feature_extractor import FeatureExtractor  # noqa: E402
from core.gallery import FeatureGallery  # noqa: E402
from core.recognizer import Recognizer  # noqa: E402
from core.webcam import (  # noqa: E402
    DEFAULT_INFER_EVERY,
    InferenceGate,
    draw_overlay,
    is_quit_key,
    recognize_frame,
)

DEFAULT_CONFIG_PATH = ROOT / "config.yaml"


def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_recognizer(cfg: dict) -> Recognizer:
    """构建识别栈；特征库文件存在则加载，空库时识别结果一律 unknown。"""
    extractor = FeatureExtractor(
        device=cfg.get("device", "cpu"),
        image_size=cfg.get("image_size", 224),
    )
    gallery = FeatureGallery(cfg, extractor=extractor)
    if gallery.feature_path.exists() and gallery.labels_path.exists():
        gallery.load()
    if len(gallery) == 0:
        print(
            "[WARN] 特征库为空，所有画面都将识别为 unknown。"
            "可先运行: python app_cli.py enroll --name <类别> --dir <图片目录>"
        )
    else:
        print(f"[info] 特征库已加载: {len(gallery)} 条特征, 类别 {gallery.list()}")
    return Recognizer(gallery, extractor=extractor, cfg=cfg), extractor


def run(camera_index: int, infer_every: int, cfg: dict) -> int:
    # 先尝试打开摄像头，无摄像头时快速失败，不白等模型加载
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # Windows 下 DSHOW 启动更快
    if not cap.isOpened():
        print(f"[ERROR] 无法打开摄像头 (index={camera_index})")
        cap.release()
        return 1

    recognizer, extractor = build_recognizer(cfg)
    unknown_label = cfg.get("unknown_label", "unknown")
    gate = InferenceGate(infer_every=infer_every)
    window_name = "object recognition (press q to quit)"
    print(f"[info] 摄像头已打开，每 {infer_every} 帧识别一次，按 q 退出。")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            if gate.tick():
                try:
                    label, score, _ = recognize_frame(recognizer, extractor, frame)
                    gate.update(label, score)
                except Exception as exc:  # 单帧识别失败不影响采集与显示
                    print(f"[WARN] 本帧识别失败，沿用上一次结果: {exc}")

            draw_overlay(frame, gate.last_label, gate.last_score, unknown_label)
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if is_quit_key(key):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="摄像头实时物体识别")
    ap.add_argument("--camera", type=int, default=0, help="摄像头索引（默认 0）")
    ap.add_argument("--every", type=int, default=DEFAULT_INFER_EVERY,
                    help=f"每 N 帧识别一次（默认 {DEFAULT_INFER_EVERY}）")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="config.yaml 路径")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    return run(args.camera, args.every, cfg)


if __name__ == "__main__":
    sys.exit(main())
