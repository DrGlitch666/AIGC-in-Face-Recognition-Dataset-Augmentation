"""配置加载器的测试。

原则：**用假数据验证真逻辑**——这里不需要 GPU、不需要数据集，
任何一台机器上跑 `pytest -q` 都应该全绿。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aigcfr.utils.config import (
    ConfigError,
    config_hash,
    decide_profile,
    deep_merge,
    guard_local,
    load_config,
)

EXP = "configs/exp/e1-real-only-seed0.yaml"


# ---------------------------------------------------------------- 合并逻辑

def test_deep_merge_overrides_only_target_keys():
    base = {"a": 1, "nested": {"x": 1, "y": 2}}
    override = {"nested": {"y": 99}, "b": 3}
    out = deep_merge(base, override)
    assert out == {"a": 1, "b": 3, "nested": {"x": 1, "y": 99}}
    # 不能改坏入参（否则第二次加载会被污染）
    assert base["nested"]["y"] == 2


# ---------------------------------------------------------------- 档位判定

@pytest.mark.parametrize(
    ("vram", "expected"),
    [(None, "cpu"), (4.0, "cpu"), (5.9, "cpu"), (6.0, "b"), (8.0, "b"), (11.9, "b"), (12.0, "a"), (24.0, "a")],
)
def test_decide_profile(vram, expected):
    assert decide_profile(vram) == expected


# ---------------------------------------------------------------- 保护规则

def test_local_may_only_hold_machine_private_keys():
    # 允许：路径与运行参数
    guard_local({"paths": {"data_root": "F:/aigcfr/data"}, "runtime": {"num_workers": 2}})
    guard_local({"cache": {"root": "F:/aigcfr/_cache"}, "env": {"HF_ENDPOINT": "https://hf-mirror.com"}})


@pytest.mark.parametrize("bad_cfg", [
    {"train": {"batch_size": 999}},
    {"model": {"emb_dim": 256}},
    {"eval": {"benchmarks": ["cfp_fp"]}},
    {"seed": 123},
])
def test_local_cannot_override_science(bad_cfg):
    with pytest.raises(ConfigError, match="不允许的顶层键"):
        guard_local(bad_cfg)


def test_load_config_rejects_bad_local_file(tmp_path):
    bad = tmp_path / "local-bad.yaml"
    bad.write_text("train:\n  batch_size: 999\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(EXP, profile="b", local_path=str(bad))


# ---------------------------------------------------------------- 四层合并

def test_load_config_merges_all_layers(tmp_path):
    # Test-only local layer; no dependency on anyone's private configuration.
    local = tmp_path / "local-test.yaml"
    local.write_text("paths:\n  data_root: TEST_ONLY_DATA_ROOT\n", encoding="utf-8")
    cfg = load_config(EXP, profile="b", local_path=str(local))

    assert cfg["_meta"]["profile"] == "b"
    assert cfg["exp_id"] == "e1-real-only-seed0"  # experiment layer
    assert cfg["model"]["emb_dim"] == 512  # base layer
    # Check the scientific setting remains legal; do not freeze its exact value.
    assert isinstance(cfg["train"]["epochs"], int) and cfg["train"]["epochs"] > 0
    assert cfg["train"]["batch_size"] == 64  # profile layer
    assert cfg["train"]["grad_accum"] == 2
    assert cfg["seed"] == 0
    assert str(cfg["paths"]["data_root"]) == "TEST_ONLY_DATA_ROOT"  # local layer


def test_exp_config_overrides_base_but_keeps_siblings(tmp_path):
    """exp overrides the named field while preserving other layers and siblings."""
    exp = tmp_path / "exp-override.yaml"
    exp.write_text("train:\n  epochs: 3\n", encoding="utf-8")
    cfg = load_config(str(exp), profile="b", local_path=str(tmp_path / "absent-local.yaml"))
    assert cfg["train"]["epochs"] == 3
    assert cfg["train"]["batch_size"] == 64
    assert cfg["model"]["emb_dim"] == 512


def test_load_config_works_without_local(tmp_path):
    cfg = load_config(EXP, profile="cpu", local_path=str(tmp_path / "absent-local.yaml"))
    assert cfg["_meta"]["profile"] == "cpu"
    assert cfg["train"]["batch_size"] == 16
    assert cfg["train"]["amp"] is False


# ---------------------------------------------------------------- 哈希行为

def test_hash_ignores_machine_private_paths(tmp_path):
    """Actual changes to the private path layer must not affect scientific hashes."""
    hashes = []
    for index, data_root in enumerate(("TEST_ONLY_ROOT_A", "TEST_ONLY_ROOT_B")):
        local = tmp_path / f"local-{index}.yaml"
        local.write_text(f"paths:\n  data_root: {data_root}\n", encoding="utf-8")
        cfg = load_config(EXP, profile="b", local_path=str(local))
        # Confirm the layer was loaded, so this cannot pass by ignoring both files.
        assert str(cfg["paths"]["data_root"]) == data_root
        hashes.append(cfg["_meta"]["config_hash"])
    without_local = load_config(EXP, profile="b", local_path=str(tmp_path / "absent-local.yaml"))
    assert hashes[0] == hashes[1] == without_local["_meta"]["config_hash"]


def test_hash_changes_when_resource_setting_changes():
    """改了 batch（资源设定）哈希必须变 —— 这是"这次实验条件变了"的标记。"""
    b = load_config(EXP, profile="b")
    a = load_config(EXP, profile="a")
    assert b["_meta"]["config_hash"] != a["_meta"]["config_hash"]
    assert config_hash({"train": {"batch_size": 64}}) != config_hash({"train": {"batch_size": 128}})
