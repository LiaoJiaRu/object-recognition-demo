"""tests/backbones/test_backbone_candidates.py

第 D-验证步：对候选骨干做"能否跑通"的断言，不评价效果好坏。

断言原则：
- 环境不具备依赖的候选：断言 missing_dependency 被如实报告，并用 pytest.skip 跳过（不算失败）。
- 依赖具备的候选：断言 load_ok、输出维度 > 0、无 NaN/Inf、可 L2 归一化。

不访问网络（权重需下载时由 scripts/verify_backbone.py 预先完成，测试在已缓存权重后离线通过）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.verify_backbone import CANDIDATES, check_backbone  # noqa: E402


@pytest.mark.parametrize("name", CANDIDATES)
def test_backbone_candidate_runs(name: str) -> None:
    res = check_backbone(name)

    # 依赖缺失：必须如实报告，并 skip（不失败）
    if res["missing_dependency"]:
        assert res["env_limited"] is True, f"{name} 缺依赖却未标记 env_limited"
        pytest.skip(
            f"{name}: 缺依赖 {res['missing_dependency']}；安装建议 {res['install_hint']}"
        )

    # 环境受限（依赖齐备但权重/仓库网络不可达）：skip（不失败），并保留原因
    if res["env_limited"]:
        pytest.skip(f"{name}: 环境受限，无法在本机验证 —— {res['env_reason']}")

    # 依赖具备且环境可用：必须真的跑通
    assert res["import_ok"] is True, f"{name} 依赖齐备但 import_ok=False"
    assert res["load_ok"] is True, f"{name} 权重加载失败: {res['error']}"
    assert res["output_dim"] is not None and res["output_dim"] > 0, (
        f"{name} 输出维度非法: {res['output_dim']}"
    )
    assert res["has_nan"] is False, f"{name} 输出含 NaN"
    assert res["has_inf"] is False, f"{name} 输出含 Inf"
    assert res["l2_normalizable"] is True, f"{name} 输出无法 L2 归一化"
