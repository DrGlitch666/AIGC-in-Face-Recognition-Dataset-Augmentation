#!/usr/bin/env python
"""W4 多种子对照分析（E1 纯真实 vs E4a 真实+合成）

读 `results/runs/<exp>/metrics.json` 与 `pairs.csv`，按**方案**（不是单次实验）汇总：

* 每种子的 accuracy 与 TAR@FAR=1e-3，以及跨种子的均值 ± 标准差
* 方案间 Welch t 检验（不同种子视为独立重复 —— 这是 TAR 这类指标的主导不确定性来源）
* 配对 bootstrap（同一批 LFW 配对、同一组重采样索引）作为评测层面的补充视角
* 按折配对 t 检验（accuracy 有 10 个配对的折，灵敏度最高）
* 结果写到 `results/runs/w4-e4-comparison/comparison.json`

用法：
    python tools/w4_compare.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.stdout.reconfigure(errors="replace")

import numpy as np  # noqa: E402

from aigcfr.eval.verify import tar_at_far  # noqa: E402

FAR = 1e-3
OUT_DIR = REPO_ROOT / "results" / "runs" / "w4-e4-comparison"
# 方案 -> (短名, 标签, 训练集说明, [各 seed 的 exp-id])
ARMS = [
    ("E1", "E1 纯真实", "real 3590",
     ["e1-real-only-seed0", "e1-real-only-seed1", "e1-real-only-seed2"]),
    ("E4a", "E4a 真实+合成(17.4%)", "real 3590 + synth 755",
     ["e4-real-plus-synth-seed0", "e4-real-plus-synth-seed1", "e4-real-plus-synth-seed2"]),
]
PROTOCOLS = [("lfw", "official"), ("lfw_filtered", "filtered")]


def pooled_pairs(exp: str, which: str) -> tuple[np.ndarray, np.ndarray]:
    """某协议下该实验的全部配对分数与标签。"""
    path = REPO_ROOT / "results" / "runs" / exp / "pairs.csv"
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8"))
            if r["set"] == which and r.get("score", "").strip()]
    scores = np.asarray([float(r["score"]) for r in rows], dtype=np.float64)
    labels = np.asarray([1 if r["label"].strip() == "1" else 0 for r in rows], dtype=np.int64)
    return scores, labels


def _tar(labels: np.ndarray, scores: np.ndarray) -> float:
    """⚠️ `tar_at_far` 的参数顺序是 **(labels, scores, far)**。

    写反了不会报错：`labels == 1` 会去筛"分数恰好等于 1.0"的样本，
    same/diff 之一为空 -> **静默返回 nan**。真实踩过，排查了很久。
    """
    t = tar_at_far(labels, scores, FAR)
    return float(t) if not isinstance(t, tuple) else float(t[0])


def paired_bootstrap(base_exp: str, other_exp: str, which: str,
                     n_boot: int = 2000, seed: int = 0) -> dict:
    """配对 bootstrap：同一组重采样索引下比较两个模型，消掉共同噪声。

    两个模型评测的是**同一批 6000 对**，是配对数据。若各自独立 bootstrap，
    会把配对带来的方差抵消全部丢掉，严重低估显著性。
    """
    sb, lb = pooled_pairs(base_exp, which)
    so, lo = pooled_pairs(other_exp, which)
    n = min(len(sb), len(so))
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        tb, to = _tar(lb[idx], sb[idx]), _tar(lo[idx], so[idx])
        if np.isfinite(tb) and np.isfinite(to):
            diffs.append(to - tb)
    arr = np.asarray(diffs)
    if arr.size == 0:
        return {"n_boot": 0}
    return {"n_boot": int(arr.size), "delta_mean": round(float(arr.mean()), 6),
            "delta_std": round(float(arr.std(ddof=0)), 6),
            "delta_p05": round(float(np.percentile(arr, 5)), 6),
            "delta_p95": round(float(np.percentile(arr, 95)), 6),
            "frac_positive": round(float((arr > 0).mean()), 4)}


def paired_ttest_folds(base_exp: str, other_exp: str, key: str) -> dict:
    """按折配对 t 检验（同一批 10 折天然配对，消掉折间共同波动）。"""

    def accs(exp: str) -> np.ndarray:
        d = json.loads((REPO_ROOT / "results" / "runs" / exp / "metrics.json")
                       .read_text(encoding="utf-8"))
        return np.asarray([x["accuracy"] for x in d["benchmarks"][key]["per_fold"]])

    d = accs(other_exp) - accs(base_exp)
    n = len(d)
    s = float(d.std(ddof=1))
    t = float(d.mean()) / (s / np.sqrt(n)) if s > 0 else float("inf")
    return {"n_folds": n, "mean_delta": round(float(d.mean()), 6), "std_delta": round(s, 6),
            "t": round(t, 4), "folds_won": int((d > 0).sum())}


def welch(a: list[float], b: list[float]) -> dict:
    """Welch t 检验（不假设方差齐），样本是各方案的种子重复。

    n=3 时自由度很小（df≈3~4），临界值很高：单侧 p<0.05 需 t>2.35，双侧需 t>3.18。
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = float(np.sqrt(va / na + vb / nb))
    t = float(b.mean() - a.mean()) / se if se > 0 else float("inf")
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return {"n_a": na, "n_b": nb, "mean_a": round(float(a.mean()), 6),
            "mean_b": round(float(b.mean()), 6),
            "delta": round(float(b.mean() - a.mean()), 6),
            "se": round(se, 6), "t": round(t, 4), "df": round(float(df), 3)}


def main() -> int:
    summary: dict = {"far": FAR, "protocol": "LFW 10-fold", "n_seeds": 3, "arms": [],
                     "tests": {}, "critical_values_n3": {"one_sided_p05": 2.353,
                                                         "two_sided_p05": 3.182}}
    arms: dict[str, dict] = {}
    for name, label, train_desc, exps in ARMS:
        rec: dict = {"name": name, "label": label, "train_set": train_desc,
                     "seeds": [], "protocols": {}}
        for exp in exps:
            mp = REPO_ROOT / "results" / "runs" / exp / "metrics.json"
            if not mp.exists():
                print(f"!! 缺 {mp}")
                return 2
            d = json.loads(mp.read_text(encoding="utf-8"))
            b = d["benchmarks"]
            row = {"exp_id": exp,
                   "best_train_acc": d.get("training", {}).get("best_train_acc"),
                   "weight_ratio": d.get("training", {}).get("final_weight_max_median_ratio")}
            for key, pname in PROTOCOLS:
                if key in b:
                    row[pname] = {"accuracy": b[key]["value"],
                                  "tar": b[key]["tar_at_far"]["1e-03"]["tar"]}
            rec["seeds"].append(row)
        for _, pname in PROTOCOLS:
            accs = [s[pname]["accuracy"] for s in rec["seeds"] if pname in s]
            tars = [s[pname]["tar"] for s in rec["seeds"] if pname in s]
            rec["protocols"][pname] = {
                "per_seed_accuracy": [round(x, 6) for x in accs],
                "per_seed_tar": [round(x, 6) for x in tars],
                "accuracy_mean": round(float(np.mean(accs)), 6),
                "accuracy_std": round(float(np.std(accs, ddof=1)), 6) if len(accs) > 1 else None,
                "tar_mean": round(float(np.mean(tars)), 6),
                "tar_std": round(float(np.std(tars, ddof=1)), 6) if len(tars) > 1 else None,
            }
        summary["arms"].append(rec)
        arms[name] = rec

    # ---------------- 终端表格 ----------------
    for _, pname in PROTOCOLS:
        print("=" * 108)
        print(f"[{pname}]  LFW 10 折   TAR@FAR={FAR:g}   3 个种子")
        print(f"{'方案':26}{'训练集':>24}{'accuracy（种子均值 ± std）':>30}{'TAR（种子均值 ± std）':>28}")
        print("-" * 108)
        for name, rec in arms.items():
            p = rec["protocols"][pname]
            print(f"{rec['label']:26}{rec['train_set']:>24}"
                  f"{p['accuracy_mean']:>18.4f} ± {p['accuracy_std']:.4f}"
                  f"{p['tar_mean']:>18.4f} ± {p['tar_std']:.4f}")
        print("-" * 108)
        a, b = arms["E1"]["protocols"][pname], arms["E4a"]["protocols"][pname]
        for metric, key in (("accuracy", "accuracy"), ("TAR", "tar")):
            w = welch(a[f"per_seed_{key}"], b[f"per_seed_{key}"])
            print(f"  {metric:9} Δ {w['delta']:+.4f}   Welch t={w['t']:+.2f} (df={w['df']:.1f})"
                  f"   n={w['n_a']}v{w['n_b']}")
            summary["tests"].setdefault(pname, {})[f"welch_{key}"] = w
        pb = paired_bootstrap(ARMS[0][3][0], ARMS[1][3][0], pname)
        tt = paired_ttest_folds(ARMS[0][3][0], ARMS[1][3][0],
                                "lfw" if pname == "official" else "lfw_filtered")
        print(f"  配对 bootstrap（seed0，同一批配对）: ΔTAR {pb.get('delta_mean'):+.4f}  "
              f"90% 区间 [{pb.get('delta_p05'):+.4f}, {pb.get('delta_p95'):+.4f}]  "
              f"P(Δ>0)={pb.get('frac_positive'):.1%}")
        print(f"  按折配对 t 检验（seed0，accuracy）: Δ {tt['mean_delta']:+.4f}  "
              f"t={tt['t']:+.2f}  赢 {tt['folds_won']}/{tt['n_folds']} 折")
        summary["tests"][pname]["paired_bootstrap_seed0"] = pb
        summary["tests"][pname]["paired_ttest_folds_seed0"] = tt
        print("=" * 108)
        print()

    summary["conclusion"] = (
        "三个种子、两种协议方向完全一致，但**两个指标的统计强度差别很大**："
        "accuracy 从 0.8812 提升到 0.8899（official，filtered 0.8727->0.8819），"
        "种子间标准差只有 0.002~0.004，Welch t=+3.35~+3.59（df≈3.8），"
        "超过 n=3 时双侧 p<0.05 的临界值 3.18 —— **显著**。"
        "TAR@FAR=1e-3 的点估计提升 +0.034~+0.039（约 +8~11%），但 t 只有 1.82，**未达显著**："
        "E4a 三个种子分别给出 0.4209/0.4587/0.4848，其中 seed0 与基线持平，"
        "种子间标准差 0.032 是 E1（0.004）的 8 倍。"
        "因此**单次运行报出的 TAR 提升会严重误导** —— 本项目最初基于单次运行得到过 '+35%'，"
        "而多种子下真实效应约为 +8%，且不显著。"
        "结论：合成数据能稳定提升 LFW accuracy（显著），对 TAR@FAR=1e-3 有正向趋势但"
        "需要更多种子或更大的评测集才能判定。"
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    print(f"已写入 {OUT_DIR / 'comparison.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
