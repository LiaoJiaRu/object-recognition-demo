"""tests/test_cli.py

用 tmp_path 下的合成图片跑 app_cli.main(argv) 闭环：
enroll -> list -> recognize -> delete，并验证 --data-root 覆盖与持久化。
不依赖 GUI、麦克风，也不污染真实数据目录。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app_cli  # noqa: E402


def write_config(data_root: Path, threshold: float = 0.75) -> Path:
    cfg_path = data_root / "config.yaml"
    cfg = {
        "data_root": str(data_root),
        "gallery_dir": "gallery",
        "test_dir": "test",
        "feature_dir": "features",
        "report_dir": "reports",
        "feature_file": "gallery.npz",
        "labels_file": "labels.json",
        "threshold": threshold,
        "topk": 5,
        "unknown_label": "unknown",
        "device": "cpu",
        "image_size": 224,
    }
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return cfg_path


def make_noise_image(path: Path, seed: int) -> None:
    rng = np.random.RandomState(seed)
    arr = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def make_solid_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color).save(path)


@pytest.fixture()
def env(tmp_path: Path) -> dict:
    cfg_path = write_config(tmp_path)
    apple_dir = tmp_path / "gallery" / "apple"
    make_noise_image(apple_dir / "a1.png", seed=1)
    make_noise_image(apple_dir / "a2.png", seed=2)
    banana_dir = tmp_path / "gallery" / "banana"
    make_noise_image(banana_dir / "b1.png", seed=10)
    unknown_img = tmp_path / "probe.png"
    make_solid_image(unknown_img, (0, 200, 0))
    return {"cfg": str(cfg_path), "root": tmp_path, "apple_dir": apple_dir,
            "apple_img": apple_dir / "a1.png", "banana_dir": banana_dir,
            "unknown_img": unknown_img}


def test_enroll_and_persisted_list(env, capsys):
    rc = app_cli.main(["--config", env["cfg"], "enroll",
                       "--name", "apple", "--dir", str(env["apple_dir"])])
    assert rc == 0
    out = capsys.readouterr().out
    assert "新增 2" in out
    assert (env["root"] / "features" / "gallery.npz").exists()
    assert (env["root"] / "features" / "labels.json").exists()

    # 再次启动进程（新 main 调用），list 应能从磁盘恢复
    rc = app_cli.main(["--config", env["cfg"], "list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "apple: 2" in out
    assert "1 个类别" in out


def test_enroll_multiple_classes_and_counts(env, capsys):
    app_cli.main(["--config", env["cfg"], "enroll",
                  "--name", "apple", "--dir", str(env["apple_dir"])])
    capsys.readouterr()
    app_cli.main(["--config", env["cfg"], "enroll",
                  "--name", "banana", "--dir", str(env["banana_dir"])])
    rc = app_cli.main(["--config", env["cfg"], "list"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "2 个类别" in out
    assert "apple: 2" in out
    assert "banana: 1" in out


def test_recognize_registered_image(env, capsys):
    app_cli.main(["--config", env["cfg"], "enroll",
                  "--name", "apple", "--dir", str(env["apple_dir"])])
    capsys.readouterr()
    rc = app_cli.main(["--config", env["cfg"], "recognize",
                       "--image", str(env["apple_img"])])
    out = capsys.readouterr().out
    assert rc == 0
    assert "label: apple" in out
    assert "score: 1.0000" in out  # 与注册图同一张


def test_recognize_unknown_image(env, capsys):
    app_cli.main(["--config", env["cfg"], "enroll",
                  "--name", "apple", "--dir", str(env["apple_dir"])])
    capsys.readouterr()
    rc = app_cli.main(["--config", env["cfg"], "recognize",
                       "--image", str(env["unknown_img"])])
    out = capsys.readouterr().out
    assert rc == 0
    assert "label: unknown" in out


def test_recognize_verbose_topk(env, capsys):
    app_cli.main(["--config", env["cfg"], "enroll",
                  "--name", "apple", "--dir", str(env["apple_dir"])])
    capsys.readouterr()
    rc = app_cli.main(["--config", env["cfg"], "recognize",
                       "--image", str(env["apple_img"]), "--verbose"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "topk:" in out


def test_delete_class(env, capsys):
    app_cli.main(["--config", env["cfg"], "enroll",
                  "--name", "apple", "--dir", str(env["apple_dir"])])
    app_cli.main(["--config", env["cfg"], "enroll",
                  "--name", "banana", "--dir", str(env["banana_dir"])])
    capsys.readouterr()

    rc = app_cli.main(["--config", env["cfg"], "delete", "--name", "apple"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "已删除类别 'apple' 的 2 条" in out

    # 重启后确认删除已持久化
    rc = app_cli.main(["--config", env["cfg"], "list"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "apple" not in out
    assert "banana: 1" in out


def test_delete_missing_returns_nonzero(env, capsys):
    app_cli.main(["--config", env["cfg"], "enroll",
                  "--name", "apple", "--dir", str(env["apple_dir"])])
    capsys.readouterr()
    rc = app_cli.main(["--config", env["cfg"], "delete", "--name", "ghost"])
    assert rc == 1
    assert "不存在" in capsys.readouterr().out


def test_enroll_missing_dir_returns_nonzero(env, capsys):
    rc = app_cli.main(["--config", env["cfg"], "enroll",
                       "--name", "x", "--dir", str(env["root"] / "nope")])
    assert rc == 1
    assert "没有图片" in capsys.readouterr().out


def test_recognize_missing_image_returns_nonzero(env, capsys):
    app_cli.main(["--config", env["cfg"], "enroll",
                  "--name", "apple", "--dir", str(env["apple_dir"])])
    capsys.readouterr()
    rc = app_cli.main(["--config", env["cfg"], "recognize",
                       "--image", str(env["root"] / "missing.png")])
    assert rc == 1


def test_list_empty_gallery(env, capsys):
    rc = app_cli.main(["--config", env["cfg"], "list"])
    assert rc == 0
    assert "特征库为空" in capsys.readouterr().out


def test_data_root_override(tmp_path: Path, capsys):
    cfg_root = tmp_path / "cfgroot"
    override_root = tmp_path / "dataroot_override"
    cfg_root.mkdir()
    cfg_path = write_config(cfg_root)
    img_dir = override_root / "gallery" / "apple"
    make_noise_image(img_dir / "a1.png", seed=5)

    rc = app_cli.main([
        "--config", str(cfg_path),
        "--data-root", str(override_root),
        "enroll", "--name", "apple", "--dir", str(img_dir),
    ])
    assert rc == 0
    capsys.readouterr()

    # 特征必须写到覆盖后的 data_root，而不是配置文件里的 data_root
    assert (override_root / "features" / "gallery.npz").exists()
    assert not (cfg_root / "features").exists()

    rc = app_cli.main(["--config", str(cfg_path),
                       "--data-root", str(override_root), "list"])
    assert rc == 0
    assert "apple: 1" in capsys.readouterr().out


def test_duplicate_enroll_accumulates(env, capsys):
    app_cli.main(["--config", env["cfg"], "enroll",
                  "--name", "apple", "--dir", str(env["apple_dir"])])
    capsys.readouterr()
    # 同名再次录入，特征条数应累加
    rc = app_cli.main(["--config", env["cfg"], "enroll",
                       "--name", "apple", "--dir", str(env["apple_dir"])])
    assert rc == 0
    assert "累计 4" in capsys.readouterr().out
    rc = app_cli.main(["--config", env["cfg"], "list"])
    assert "apple: 4" in capsys.readouterr().out
