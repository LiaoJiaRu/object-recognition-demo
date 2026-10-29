"""scripts/verify_backbone.py

第 D-验证步：独立验证候选骨干能否在当前 venv 正常加载 / 前向 / 输出可用特征。

约束：
- 只读：不写 gallery.npz / labels.json，不改任何现有文件。
- 缺依赖时如实记录 missing_dependency 并给出安装命令建议，但不自动安装、不抛异常。
- 如产生缓存权重，落在各库默认缓存目录（torch.hub 缓存 / open_clip 缓存）。
- 不并入现有识别链路，只做事实验证。

用法：
    python scripts/verify_backbone.py --backbone all
    python scripts/verify_backbone.py --backbone resnet50
    python scripts/verify_backbone.py --backbone all --image <path/to.jpg> --runs 3
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CANDIDATES = ("resnet18", "resnet50", "dinov2_vits14", "clip_vitb32")

# 候选 -> 需要的 python 模块名（用于 import 检查）
REQUIRES: dict[str, tuple[str, ...]] = {
    "resnet18": ("torch", "torchvision"),
    "resnet50": ("torch", "torchvision"),
    "dinov2_vits14": ("torch", "torchvision"),
    "clip_vitb32": ("torch", "open_clip"),
}

# 缺失模块 -> pip 安装建议
INSTALL_HINTS: dict[str, str] = {
    "open_clip": "venv\\Scripts\\python.exe -m pip install open_clip_torch",
    "torch": "venv\\Scripts\\python.exe -m pip install torch==2.2.2 torchvision==0.17.2",
    "torchvision": "venv\\Scripts\\python.exe -m pip install torchvision==0.17.2",
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".jfif", ".webp"}

# 加载失败信息中若命中以下关键词，判定为"环境受限（网络/权重不可达）"而非代码缺陷
_NET_MARKERS = (
    "no internet connection",
    "could not be found in the cache",
    "failed to download",
    "cannot find the requested files",
    "connection",
    "timed out",
    "timeout",
    "urlopen",
    "url error",
    "temporary failure",
    "network",
    "internet connection is on",
)


def _looks_network(msg: str) -> bool:
    m = (msg or "").lower()
    return any(k in m for k in _NET_MARKERS)


# ---------------------------------------------------------------------
# 依赖与图片定位（只读）
# ---------------------------------------------------------------------
def _has(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except Exception:
        return False


def _load_cfg() -> dict:
    import yaml

    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _default_image(cfg: dict) -> Path | None:
    """从 data_root/test 下取第一张真实图片（只读，不写入）。"""
    data_root = Path(cfg["data_root"])
    test_dir = data_root / cfg.get("test_dir", "test")
    if not test_dir.exists():
        return None
    for p in sorted(test_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            return p
    return None


def _missing_deps(name: str) -> str | None:
    for mod in REQUIRES.get(name, ()):
        if not _has(mod):
            return mod
    return None


# ---------------------------------------------------------------------
# 各候选的加载器：返回 (model, preprocess, feature_fn, weight_source)
# ---------------------------------------------------------------------
def _load_resnet(arch: str):
    import torch
    from torchvision import models

    if arch == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    else:
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        model = models.resnet50(weights=weights)
    model.fc = torch.nn.Identity()  # 去 FC → 池化特征
    model.eval()
    preprocess = weights.transforms()
    return model, preprocess, (lambda m, x: m(x)), f"torchvision:{weights.name}"


def _load_dino():
    import torch
    from torchvision import transforms

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False)
    model.eval()
    preprocess = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return (
        model,
        preprocess,
        (lambda m, x: m(x)),
        "torch.hub:facebookresearch/dinov2:dinov2_vits14",
    )


def _load_clip():
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model.eval()
    return (
        model,
        preprocess,
        (lambda m, x: m.encode_image(x)),
        "open_clip:ViT-B-32:openai",
    )


def _loader_for(name: str) -> Callable[[], tuple]:
    if name == "resnet18":
        return lambda: _load_resnet("resnet18")
    if name == "resnet50":
        return lambda: _load_resnet("resnet50")
    if name == "dinov2_vits14":
        return _load_dino
    if name == "clip_vitb32":
        return _load_clip
    raise ValueError(f"未知候选: {name}")


def _rss_mb() -> float | None:
    try:
        import os

        import psutil

        return round(psutil.Process(os.getpid()).memory_info().rss / 1e6, 1)
    except Exception:
        return None


# ---------------------------------------------------------------------
# 核心检查
# ---------------------------------------------------------------------
def check_backbone(
    name: str, image_path: str | Path | None = None, runs: int = 3
) -> dict[str, Any]:
    """对单个候选骨干做结构化验证，任何失败都记录到返回值，不抛异常。

    返回 dict 键：
        name, requires, missing_dependency, install_hint,
        import_ok, load_ok, weight_source, load_seconds,
        image, output_dim, dtype, has_nan, has_inf, l2_normalizable,
        forward_ms_median, forward_ms_all, rss_mb, error
    """
    res: dict[str, Any] = {
        "name": name,
        "requires": list(REQUIRES.get(name, ())),
        "missing_dependency": None,
        "install_hint": None,
        "env_limited": False,
        "env_reason": None,
        "import_ok": False,
        "load_ok": False,
        "weight_source": None,
        "load_seconds": None,
        "image": None,
        "output_dim": None,
        "dtype": None,
        "has_nan": None,
        "has_inf": None,
        "l2_normalizable": None,
        "forward_ms_median": None,
        "forward_ms_all": None,
        "rss_mb": None,
        "error": None,
    }

    if name not in CANDIDATES:
        res["error"] = f"未知候选: {name}"
        return res

    # a) 依赖检查（缺失即如实记录，不抛异常）
    missing = _missing_deps(name)
    if missing is not None:
        res["missing_dependency"] = missing
        res["install_hint"] = INSTALL_HINTS.get(missing, f"pip install {missing}")
        res["env_limited"] = True
        res["env_reason"] = f"缺少依赖模块 {missing}"
        return res
    res["import_ok"] = True

    # b) 加载权重（记录来源与耗时）
    try:
        t0 = time.perf_counter()
        model, preprocess, feat_fn, source = _loader_for(name)()
        res["load_seconds"] = round(time.perf_counter() - t0, 2)
        res["weight_source"] = source
        res["load_ok"] = True
    except Exception as e:  # noqa: BLE001 - 需如实上报任何加载失败
        msg = f"{type(e).__name__}: {e}"
        res["error"] = f"[load] {msg}"
        if _looks_network(msg):
            res["env_limited"] = True
            res["env_reason"] = f"权重/仓库网络不可达：{msg}"
        return res

    # c)+d) 前向 + 特征可用性
    try:
        import torch
        from PIL import Image

        if image_path is None:
            img_path = _default_image(_load_cfg())
        else:
            img_path = Path(image_path)
        if img_path is None or not Path(img_path).exists():
            res["error"] = "[image] 找不到可用于前向的真实图片"
            return res
        res["image"] = str(img_path)

        pil = Image.open(img_path).convert("RGB")
        with torch.no_grad():
            x = preprocess(pil).unsqueeze(0)
            out = feat_fn(model, x)
            feat = out.detach().cpu().numpy().reshape(-1).astype(np.float32)

        res["output_dim"] = int(feat.size)
        res["dtype"] = "float32"
        res["has_nan"] = bool(np.isnan(feat).any())
        res["has_inf"] = bool(np.isinf(feat).any())
        norm = float(np.linalg.norm(feat))
        res["l2_normalizable"] = bool(np.isfinite(norm) and norm > 1e-12)

        # e) 单张前向 CPU 耗时：跑 runs 次取中位数
        times: list[float] = []
        with torch.no_grad():
            for _ in range(max(1, runs)):
                t = time.perf_counter()
                feat_fn(model, x)
                times.append((time.perf_counter() - t) * 1000.0)
        res["forward_ms_all"] = [round(t, 2) for t in times]
        res["forward_ms_median"] = round(float(np.median(times)), 2)

        # f) 进程 RSS（可选）
        res["rss_mb"] = _rss_mb()
    except Exception as e:  # noqa: BLE001
        res["error"] = f"[forward] {type(e).__name__}: {e}"

    return res


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "是" if v else "否"
    return str(v)


def _print_summary(results: list[dict[str, Any]]) -> None:
    header = (
        f"{'候选':<16}{'依赖':<8}{'加载':<8}{'维度':<8}"
        f"{'前向中位/ms':<14}{'NaN':<7}{'Inf':<7}{'L2可归一化':<12}备注"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        dep = "缺失" if r["missing_dependency"] else "具备"
        if r["load_ok"]:
            load = "成功"
        elif r["env_limited"]:
            load = "跳过"
        else:
            load = "失败"
        note = r["weight_source"] or (r["error"] or "")
        if r["missing_dependency"]:
            note = f"缺失 {r['missing_dependency']}；安装: {r['install_hint']}"
        elif r["env_limited"] and not r["load_ok"]:
            note = f"环境受限：{r['env_reason']}"
        print(
            f"{r['name']:<16}{dep:<8}{load:<8}{_fmt(r['output_dim']):<8}"
            f"{_fmt(r['forward_ms_median']):<14}{_fmt(r['has_nan']):<7}"
            f"{_fmt(r['has_inf']):<7}{_fmt(r['l2_normalizable']):<12}{note}"
        )


def main(argv: list[str] | None = None) -> list[dict[str, Any]]:
    ap = argparse.ArgumentParser(description="验证候选骨干能否跑通（只读）")
    ap.add_argument(
        "--backbone",
        default="all",
        choices=(*CANDIDATES, "all"),
        help="要验证的骨干；all 表示依次跑全部候选",
    )
    ap.add_argument("--image", default=None, help="用于前向的真实图片路径")
    ap.add_argument("--runs", type=int, default=3, help="单张前向计时次数（取中位数）")
    args = ap.parse_args(argv)

    names = list(CANDIDATES) if args.backbone == "all" else [args.backbone]

    print(f"python: {sys.executable}")
    print(f"backbone: {args.backbone}    runs: {args.runs}")
    print()

    results = [check_backbone(n, args.image, args.runs) for n in names]

    for r in results:
        print(f"[{r['name']}]")
        print(f"  requires           : {r['requires']}")
        print(f"  missing_dependency : {r['missing_dependency']}")
        print(f"  env_limited        : {r['env_limited']}")
        if r["env_limited"]:
            print(f"  env_reason         : {r['env_reason']}")
        if r["missing_dependency"]:
            print(f"  install_hint       : {r['install_hint']}")
        print(f"  import_ok          : {r['import_ok']}")
        print(f"  load_ok            : {r['load_ok']}")
        print(f"  weight_source      : {r['weight_source']}")
        print(f"  load_seconds       : {r['load_seconds']}")
        print(f"  image              : {r['image']}")
        print(f"  output_dim         : {r['output_dim']}")
        print(f"  has_nan / has_inf  : {r['has_nan']} / {r['has_inf']}")
        print(f"  l2_normalizable    : {r['l2_normalizable']}")
        print(f"  forward_ms (all)   : {r['forward_ms_all']}")
        print(f"  forward_ms_median  : {r['forward_ms_median']}")
        print(f"  rss_mb             : {r['rss_mb']}")
        print(f"  error              : {r['error']}")
        print()

    print("==== 汇总表 ====")
    _print_summary(results)
    return results


if __name__ == "__main__":
    main()
