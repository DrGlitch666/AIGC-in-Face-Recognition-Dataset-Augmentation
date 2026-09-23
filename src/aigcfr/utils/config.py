"""三层配置加载：base（科学设定）+ profile（档位资源设定）+ exp（实验）+ local（机器私有）。

**为什么要分三层？** 因为项目由两人在两台配置不同的机器上开发：
  - 「科学设定」必须一致，否则实验不可比；
  - 「资源设定」必须随机器变（8 GB 显存和 CPU 不可能用同一个 batch）；
  - 「机器私有设定」只放路径，绝不能影响结果。

合并顺序（后面的覆盖前面的）：
    configs/base.yaml
      -> configs/profiles/<档位>.yaml      （档位由显存自动判定，或 --profile 指定）
      -> configs/exp/<实验>.yaml
      -> configs/local.<机器>.yaml         （只允许 paths / runtime / cache / env）

合并结果带 `_meta`（档位、来源、配置哈希、显存），要写进 run 目录供追溯。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

# src/aigcfr/utils/config.py -> 上溯 3 层就是仓库根目录
REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs"

# local 层是「白名单」：只允许这四类顶层键。
# 白名单比黑名单安全 —— 新增的科学设定字段不会被漏掉。
LOCAL_ALLOWED_TOP_KEYS = {"paths", "runtime", "cache", "env"}


class ConfigError(RuntimeError):
    """配置错误（最常见：local 层试图覆盖科学设定）。"""


# ---------------------------------------------------------------- 基础工具

def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"配置文件不存在: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"配置文件顶层必须是键值对: {path}")
    return data


def deep_merge(base: dict, override: dict) -> dict:
    """递归合并：字典递归合并，其他类型直接替换。返回新字典，不改动入参。"""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def guard_local(local_cfg: dict, local_path: object = None) -> None:
    """local 层只允许机器私有设定；碰「科学设定」直接报错。

    这是整套协作机制里最关键的一道闸：它保证没人能"悄悄"用本地覆盖改掉实验条件，
    导致另一台机器复现不出来。
    """
    bad = sorted(set(local_cfg) - LOCAL_ALLOWED_TOP_KEYS)
    if bad:
        raise ConfigError(
            f"{local_path or 'local 配置'} 里出现了不允许的顶层键: {bad}\n"
            f"  local 层只允许: {sorted(LOCAL_ALLOWED_TOP_KEYS)}\n"
            "  「科学设定」(train / model / eval / seed 等) 请改 "
            "configs/base.yaml 或 configs/exp/*.yaml，并让另一台机器也知道——\n"
            "  否则两台机器的实验将不可比。"
        )


# ---------------------------------------------------------------- 档位判定

def detect_vram_gb() -> float | None:
    """返回 GPU 显存总量（GB）；没有可用 CUDA 时返回 None。"""
    try:
        import torch  # 延迟导入：没装 torch 时也能用配置加载器
    except Exception:  # noqa: BLE001
        return None
    try:
        if not torch.cuda.is_available():
            return None
        return round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
    except Exception:  # noqa: BLE001
        return None


def decide_profile(vram_gb: float | None) -> str:
    """档位规则，与 docs/FRAMEWORK.md §4.1 保持一致。"""
    if vram_gb is None:
        return "cpu"
    if vram_gb >= 12:
        return "a"
    if vram_gb >= 6:
        return "b"
    return "cpu"


# ---------------------------------------------------------------- 配置哈希

def config_hash(cfg: dict) -> str:
    """对「科学 + 资源设定」做哈希。

    刻意**排除** paths / cache / env —— 它们只是路径，两台机器必然不同，
    如果参与哈希，两台机器的 config_hash 永远对不上，就失去了溯源意义。
    """
    payload = {k: v for k, v in cfg.items() if k not in ("paths", "cache", "env", "_meta")}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------- 主入口

def _resolve(path_like: str | Path, config_dir: Path, subdir: str | None = None) -> Path:
    """把用户给的路径解析成真实文件：先按仓库根解析，再按 configs/<subdir> 兜底。"""
    candidate = Path(path_like)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    from_root = REPO_ROOT / candidate
    if from_root.exists():
        return from_root
    if subdir:
        from_sub = config_dir / subdir / candidate.name
        if from_sub.exists():
            return from_sub
    return from_root  # 交给 _read_yaml 报"不存在"


def load_config(
    exp_path: str | Path | None = None,
    profile: str = "auto",
    local_path: str | Path | None = None,
    config_dir: str | Path | None = None,
) -> dict[str, Any]:
    """加载并合并四层配置，返回带 `_meta` 的完整配置字典。

    参数
    ----
    exp_path   : 实验配置，例如 "configs/exp/e1-real-only-seed0.yaml"（可省略）
    profile    : "auto" 或 "a" / "b" / "cpu"
    local_path : 机器私有配置，例如 "configs/local.a.yaml"（可省略）
    config_dir : 配置根目录，默认 <仓库根>/configs（测试时可指定别处）
    """
    cfg_dir = Path(config_dir) if config_dir else CONFIG_DIR
    sources: list[str] = []

    # ① base
    cfg = _read_yaml(cfg_dir / "base.yaml")
    sources.append("base.yaml")

    # ② profile
    prof_name = decide_profile(detect_vram_gb()) if profile == "auto" else str(profile)
    prof_file = cfg_dir / "profiles" / f"{prof_name}.yaml"
    cfg = deep_merge(cfg, _read_yaml(prof_file))
    sources.append(f"profiles/{prof_name}.yaml")

    # ③ 实验配置
    if exp_path:
        exp_file = _resolve(exp_path, cfg_dir, "exp")
        cfg = deep_merge(cfg, _read_yaml(exp_file))
        try:
            sources.append(str(exp_file.relative_to(REPO_ROOT)))
        except ValueError:
            sources.append(exp_file.name)

    # ④ 机器私有
    if local_path:
        loc_file = _resolve(local_path, cfg_dir)
        if loc_file.exists():
            local_cfg = _read_yaml(loc_file)
            guard_local(local_cfg, loc_file)   # ← 保护规则
            cfg = deep_merge(cfg, local_cfg)
            sources.append(loc_file.name)
        else:
            sources.append(f"(local 不存在: {loc_file})")

    cfg["_meta"] = {
        "profile": prof_name,
        "sources": sources,
        "config_hash": config_hash(cfg),
        "vram_gb": detect_vram_gb(),
    }
    return cfg


__all__ = [
    "CONFIG_DIR",
    "REPO_ROOT",
    "LOCAL_ALLOWED_TOP_KEYS",
    "ConfigError",
    "config_hash",
    "decide_profile",
    "deep_merge",
    "detect_vram_gb",
    "guard_local",
    "load_config",
]
