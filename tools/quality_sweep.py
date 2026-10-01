#!/usr/bin/env python
"""配置扫描：找出"身份相似度最高"的生成配置（一次性工具）

## 设计原则（吸取教训）

**不自己写生成逻辑**，而是通过 subprocess 调用 `scripts/generate.py` 本身
（只加开关）。这样测的就是**将来真正要跑的代码**，不会出现"扫描脚本自己跑偏、
结论和正式脚本不符"的情况 —— 之前正是这么栽过一次。

评分同样复用共享的 `identity_embedding`（与 filter_synth 一致），
检测尺寸 320（640 会造成幸存者偏差）。

## 用法

    python tools/quality_sweep.py --identities 2 --seeds 3
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.stdout.reconfigure(errors="replace")

import numpy as np
import torch  # noqa: F401

from aigcfr.eval.embed import load_app, warmup
from aigcfr.eval.faces import identity_embedding
from aigcfr.eval.verify import l2_normalize
from aigcfr.utils.config import find_local_config, load_config

# ---------------- 扫描的配置 ----------------
# 每组只改 generate.py 的开关，不改代码路径
CONFIGS = [
    ("baseline", ["--scheduler", "pndm", "--ref-mode", "single", "--steps", "30", "--guidance", "5.0"]),
    ("A_dpmpp", ["--scheduler", "dpmpp", "--ref-mode", "single", "--steps", "30", "--guidance", "5.0"]),
    ("B_meanref", ["--scheduler", "pndm", "--ref-mode", "mean", "--steps", "30", "--guidance", "5.0"]),
    ("C_steps40_cfg40", ["--scheduler", "pndm", "--ref-mode", "single", "--steps", "40", "--guidance", "4.0"]),
    ("D_realvis", ["--base", "realvis", "--scheduler", "pndm", "--ref-mode", "single", "--steps", "30", "--guidance", "5.0"]),
    ("E_all", ["--base", "sd15", "--scheduler", "dpmpp", "--ref-mode", "mean", "--steps", "40", "--guidance", "4.0"]),
]


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--identities", type=int, default=2, help="用前 N 个身份")
    ap.add_argument("--seeds", type=int, default=3, help="每个身份生成几张（不同 seed）")
    ap.add_argument("--only", nargs="*", default=None, help="只跑指定的组（默认全跑）")
    args = ap.parse_args()

    cfg = load_config(None, profile="auto", local_path=str(find_local_config()))
    data_root = Path(cfg["paths"]["data_root"])
    models_root = Path(cfg["paths"]["models_root"])

    # ---------------- 参考：真实图的身份中心 ----------------
    print("加载 buffalo_l 并算真实图身份中心 …")
    app = load_app(models_root / "insightface", det_size=320, ctx_id=0)
    warmup(app)

    real = []
    with open(REPO_ROOT / cfg["data"]["manifest"], encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                if r.get("split") == "train" and r.get("status") == "accepted":
                    real.append(r)
    by_id: dict[str, list[np.ndarray]] = {}
    for r in real:
        v = identity_embedding(app, data_root / r["path"])
        if v is not None:
            by_id.setdefault(r["identity_id"], []).append(v)
    centroids = {k: l2_normalize(np.mean(np.stack(v), axis=0)) for k, v in by_id.items()}
    print(f"  {len(centroids)} 个身份")

    results: list[tuple[str, float, int]] = []
    for name, extra in CONFIGS:
        if args.only and name not in args.only:
            continue
        out_dir = REPO_ROOT / "results" / "runs" / f"_sweep_{name}"
        manifest = out_dir / "manifest.jsonl"
        cmd = [sys.executable, str(REPO_ROOT / "scripts" / "generate.py"),
               "--identities", str(args.identities), "--per-identity", str(args.seeds),
               "--scale", "0.8", "--out", str(out_dir), "--manifest", str(manifest),
               "--exp-id", f"sweep-{name}", *extra]
        print(f"\n{'='*72}\n[{name}] {' '.join(extra)}\n{'='*72}")
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            print(f"  !! 失败 (exit {proc.returncode})")
            print("\n".join((proc.stderr or proc.stdout or "").splitlines()[-6:]))
            results.append((name, float("nan"), 0))
            continue
        tail = [l for l in proc.stdout.splitlines() if "生成" in l or "跳过" in l]
        print("  " + " | ".join(t.strip() for t in tail[-2:]))

        # ---------- 评分 ----------
        sims = []
        for rec in (json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()):
            v = identity_embedding(app, Path(rec["path"]))
            c = centroids.get(rec["identity_id"])
            if v is not None and c is not None:
                sims.append(float(l2_normalize(v) @ c))
        mean = statistics.fmean(sims) if sims else float("nan")
        print(f"  -> 可评分 {len(sims)} 张 | id_sim 均值 {mean:+.4f} | "
              f"中位数 {statistics.median(sims):+.4f}" if sims else "  -> 全部检不出脸")
        results.append((name, mean, len(sims)))

    print()
    print("=" * 72)
    print("扫描结果（id_sim = 与本人真实图中心的余弦；参考：真实同人 0.745、不同人 <0.25）")
    print("=" * 72)
    print(f"{'配置':>18} {'id_sim 均值':>12} {'可评分':>8}")
    for name, mean, n in sorted(results, key=lambda x: -(x[1] if x[1] == x[1] else -9)):
        flag = "  <<< 最佳" if results and mean == max(r[1] for r in results if r[1] == r[1]) else ""
        print(f"{name:>18} {mean:>+12.4f} {n:>8}{flag}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
