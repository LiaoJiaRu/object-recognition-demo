"""app_cli.py

命令行核心闭环（阶段 7）。不依赖 GUI / 语音。

子命令:
    enroll    --name <类别> --dir <图片目录>   提取特征并持久化到特征库
    recognize --image <图片路径>               输出 label 和 score
    list                                      列出已注册类别及特征条数
    delete    --name <类别>                   删除该类别全部特征

路径规则:
    - data_root / feature_dir / 各文件名默认从 config.yaml 读取；
    - 全局参数 --config 可指定配置文件，--data-root 可临时覆盖 data_root。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.feature_extractor import FeatureExtractor  # noqa: E402
from core.gallery import FeatureGallery  # noqa: E402
from core.recognizer import Recognizer  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".jfif", ".webp"}
DEFAULT_CONFIG_PATH = ROOT / "config.yaml"


def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_images(directory: Path) -> list[Path]:
    """列出目录下（不递归）支持的图片，按文件名排序。"""
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def make_extractor(cfg: dict) -> FeatureExtractor:
    return FeatureExtractor(
        device=cfg.get("device", "cpu"),
        image_size=cfg.get("image_size", 224),
    )


def load_or_create_gallery(cfg: dict, extractor=None) -> FeatureGallery:
    """特征库文件存在则加载（恢复重启状态），否则返回空库。"""
    gallery = FeatureGallery(cfg, extractor=extractor)
    if gallery.feature_path.exists() and gallery.labels_path.exists():
        gallery.load()
    return gallery


# ---------------------------------------------------------------------
# 子命令
# ---------------------------------------------------------------------
def cmd_enroll(args: argparse.Namespace, cfg: dict) -> int:
    image_dir = Path(args.dir)
    images = list_images(image_dir)
    if not images:
        print(f"[ERROR] 目录不存在或其中没有图片: {image_dir}")
        return 1

    extractor = make_extractor(cfg)
    gallery = load_or_create_gallery(cfg, extractor=extractor)
    added = gallery.add(args.name, images)
    gallery.save()

    counts = gallery.counts()
    print(
        f"已录入 '{args.name}': 新增 {added} 张，"
        f"该类别累计 {counts.get(args.name, 0)} 条特征"
    )
    print(f"[saved] {gallery.feature_path}")
    print(f"[saved] {gallery.labels_path}")
    return 0


def cmd_recognize(args: argparse.Namespace, cfg: dict) -> int:
    image_path = Path(args.image)
    if not image_path.is_file():
        print(f"[ERROR] 图片不存在: {image_path}")
        return 1

    extractor = make_extractor(cfg)
    gallery = load_or_create_gallery(cfg, extractor=extractor)
    recognizer = Recognizer(gallery, extractor=extractor, cfg=cfg)

    label, score, topk = recognizer.predict(str(image_path))
    print(f"image: {image_path}")
    print(f"label: {label}")
    print(f"score: {score:.4f}")
    if getattr(args, "verbose", False):
        print("topk:")
        for item in topk:
            print(f"  - {item['label']}: {item['score']:.4f}")
    return 0


def cmd_list(args: argparse.Namespace, cfg: dict) -> int:
    gallery = load_or_create_gallery(cfg)
    names = gallery.list()
    if not names:
        print("特征库为空（尚未录入任何类别）")
        print(f"[path] {gallery.feature_path}")
        return 0

    counts = gallery.counts()
    print(f"特征库路径: {gallery.feature_path}")
    print(f"共 {len(names)} 个类别, {len(gallery)} 条特征:")
    for name in names:
        print(f"  - {name}: {counts[name]}")
    return 0


def cmd_delete(args: argparse.Namespace, cfg: dict) -> int:
    gallery = load_or_create_gallery(cfg)
    removed = gallery.delete(args.name)
    if removed == 0:
        print(f"[WARN] 特征库中不存在类别 '{args.name}'，未做改动")
        return 1
    gallery.save()
    print(f"已删除类别 '{args.name}' 的 {removed} 条特征，剩余 {len(gallery)} 条")
    print(f"[saved] {gallery.feature_path}")
    return 0


# ---------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="物体识别命令行工具")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="config.yaml 路径（默认项目根目录 config.yaml）",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="临时覆盖 config.yaml 中的 data_root",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_enroll = sub.add_parser("enroll", help="录入一个类别的图片")
    p_enroll.add_argument("--name", required=True, help="类别名")
    p_enroll.add_argument("--dir", required=True, help="图片目录")
    p_enroll.set_defaults(func=cmd_enroll)

    p_rec = sub.add_parser("recognize", help="识别单张图片")
    p_rec.add_argument("--image", required=True, help="图片路径")
    p_rec.add_argument("-v", "--verbose", action="store_true", help="打印 top-k 明细")
    p_rec.set_defaults(func=cmd_recognize)

    p_list = sub.add_parser("list", help="列出已注册类别")
    p_list.set_defaults(func=cmd_list)

    p_del = sub.add_parser("delete", help="删除一个类别")
    p_del.add_argument("--name", required=True, help="类别名")
    p_del.set_defaults(func=cmd_delete)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.data_root is not None:
        cfg["data_root"] = args.data_root

    return args.func(args, cfg)


if __name__ == "__main__":
    sys.exit(main())
