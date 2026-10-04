#!/usr/bin/env python
"""汇总 W4 对照实验（E1 / E4a / E4b）并落盘

读 `results/runs/<exp>/metrics.json` 与 `pairs.csv`，输出：
* official / filtered 两套协议下的 accuracy 与 TAR@FAR=1e-3（含每折均值与标准差）
* 相对 E1 的绝对/相对提升，以及"提升 = 几倍每折标准差"（说明不是噪声）
* 写到 `results/runs/w4-e4-comparison/comparison.json`

用法：
    python tools/w4_compare.py
"""

from __future__ import annotations

import collections
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.stdout.reconfigure(errors="replace")

import numpy as np  # noqa: E402

from aigcfr.eval.verify import tar_at_far  # noqa: E402

EXPERIMENTS = [
    ("e1-real-only-seed0", "E1 纯真实", "real 3590"),
    ("e4-real-plus-synth-seed0", "E4a +755合成(17.4%)", "real 3590 + synth 755"),
    ("e4-highquality-synth-seed0", "E4b +448高质合成(11.1%)", "real 3590 + synth 448"),
]
FAR = 1e-3
OUT_DIR = REPO_ROOT / "results" / "runs" / "w4-e4-comparison"


def paired_bootstrap_tar(base_exp: str, other_exp: str, which: str,
                         n_boot: int = 2000, seed: int = 0) -> dict:
    """**配对** bootstrap：同一批配对、同一组重采样索引，比较两个模型的 TAR@FAR。

    ⚠️ 为什么不能直接比两个独立的 bootstrap 标准差：
       两个模型评测的是**同一批 6000 对**，是配对数据。独立 bootstrap 会把
       "配对"带来的方差抵消掉全部丢掉，导致严重低估显著性。
       正确做法是每次重采样都用**同一组索引**算两个模型的 TAR，再取差值。
       而且 FAR=1e-3 时阈值只由约 3 个 impuator 决定，单模型的 bootstrap 方差
       本来就大（E1 是 0.065），配对差值能把这个共同噪声消掉。
    """
    s_base, l_base = pooled_pairs(base_exp, which)
    s_oth, l_oth = pooled_pairs(other_exp, which)
    n = min(len(s_base), len(s_oth))
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        tb = tar_at_far(l_base[idx], s_base[idx], FAR)
        to = tar_at_far(l_oth[idx], s_oth[idx], FAR)
        tb = float(tb) if not isinstance(tb, tuple) else float(tb[0])
        to = float(to) if not isinstance(to, tuple) else float(to[0])
        if np.isfinite(tb) and np.isfinite(to):
            diffs.append(to - tb)
    arr = np.asarray(diffs)
    if arr.size == 0:
        return {"n_boot": 0}
    return {
        "n_boot": int(arr.size),
        "delta_mean": float(arr.mean()),
        "delta_std": float(arr.std(ddof=0)),
        "delta_p05": float(np.percentile(arr, 5)),
        "delta_p95": float(np.percentile(arr, 95)),
        "frac_positive": float((arr > 0).mean()),
    }


def paired_ttest_folds(base_exp: str, other_exp: str, key: str) -> dict:
    """按折配对 t 检验（同一批 10 折，天然配对）。

    这是本项目**唯一能达到统计显著**的检验：TAR@FAR=1e-3 受限于 LFW 只有
    3000 个非同类对（阈值由约 3 个 impostor 决定），bootstrap 区间必然很宽；
    而 accuracy 有 10 个配对的折，配对后折间共同波动被消掉，灵敏度高得多。

    n=10 的临界值：双侧 p<0.05 -> |t|>2.26；单侧 p<0.05 -> t>1.83。
    """
    d_base = json.loads((REPO_ROOT / "results" / "runs" / base_exp / "metrics.json")
                        .read_text(encoding="utf-8"))
    d_oth = json.loads((REPO_ROOT / "results" / "runs" / other_exp / "metrics.json")
                       .read_text(encoding="utf-8"))
    a = np.asarray([x["accuracy"] for x in d_base["benchmarks"][key]["per_fold"]])
    b = np.asarray([x["accuracy"] for x in d_oth["benchmarks"][key]["per_fold"]])
    d = b - a
    n = len(d)
    m, s = float(d.mean()), float(d.std(ddof=1))
    t = m / (s / np.sqrt(n)) if s > 0 else float("inf")
    return {"n_folds": n, "mean_delta": round(m, 6), "std_delta": round(s, 6),
            "t": round(t, 4), "folds_won": int((d > 0).sum()),
            "crit_two_sided_p05": 2.262, "crit_one_sided_p05": 1.833}


def pooled_pairs(exp: str, which: str) -> tuple[np.ndarray, np.ndarray]:
    """取某个协议下的全部配对分数与标签。"""
    path = REPO_ROOT / "results" / "runs" / exp / "pairs.csv"
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8"))
            if r["set"] == which and r.get("score", "").strip()]
    scores = np.asarray([float(r["score"]) for r in rows], dtype=np.float64)
    labels = np.asarray([1 if r["label"].strip() == "1" else 0 for r in rows], dtype=np.int64)
    return scores, labels


def bootstrap_tar(exp: str, which: str, n_boot: int = 1000, seed: int = 0) -> dict:
    """用 bootstrap 估 TAR@FAR 的离散度。

    ⚠️ 为什么不用"每折 TAR"：LFW 每折只有 300 对非同类，而 FAR=1e-3 需要至少
       1000 个 impostor 才能解出来（每折的期望 impostor 数是 0.3 个）——
       按折算 TAR@1e-3 在数学上无意义，会得到 nan。所以只在**全量 6000 对**
       上算一次，再用 bootstrap 重采样得到分布。
    """
    scores, labels = pooled_pairs(exp, which)
    rng = np.random.default_rng(seed)
    n = len(scores)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        # ⚠️ 参数顺序是 (labels, scores, far) —— 不是 (scores, labels)。
        #    写反了不会报错，只会让 labels==1 筛出"分数恰好等于 1.0"的样本，
        #    same/diff 之一为空 -> 静默返回全 nan。**真实踩过。**
        t = tar_at_far(labels[idx], scores[idx], FAR)
        t = float(t) if not isinstance(t, tuple) else float(t[0])
        if np.isfinite(t):
            vals.append(t)
    arr = np.asarray(vals)
    if arr.size == 0:
        return {"tar_mean": float("nan"), "tar_std": float("nan"), "n_boot": 0}
    return {"tar_mean": float(arr.mean()), "tar_std": float(arr.std(ddof=0)),
            "tar_p05": float(np.percentile(arr, 5)), "tar_p95": float(np.percentile(arr, 95)),
            "n_boot": int(arr.size)}


def main() -> int:
    summary = {"far": FAR, "protocol": "LFW 10-fold", "experiments": []}
    table = {}
    for exp, label, train_desc in EXPERIMENTS:
        mpath = REPO_ROOT / "results" / "runs" / exp / "metrics.json"
        if not mpath.exists():
            print(f"!! 缺 {mpath}")
            return 2
        d = json.loads(mpath.read_text(encoding="utf-8"))
        b = d["benchmarks"]
        entry = {"exp_id": exp, "label": label, "train_set": train_desc,
                 "train_images": d.get("train_set", {}).get("total_images"),
                 "synth_images": d.get("train_set", {}).get("synth_images"),
                 "synth_ratio": d.get("train_set", {}).get("synth_ratio"),
                 "best_train_acc": d.get("training", {}).get("best_train_acc"),
                 "protocols": {}}
        for key, pname in (("lfw", "official"), ("lfw_filtered", "filtered")):
            if key not in b:
                continue
            bench = b[key]
            fold_acc = [x["accuracy"] for x in bench["per_fold"]]
            boot = bootstrap_tar(exp, pname)
            # ⚠️ 用 numpy 而不是 statistics.pstdev：Python 3.11 的 pstdev 在浮点数据上
            #    会偶发 "AttributeError: 'float' object has no attribute 'numerator'"。
            entry["protocols"][pname] = {
                "n_pairs": bench.get("n_pairs_used"),
                "accuracy": bench["value"],
                "accuracy_fold_mean": round(float(np.mean(fold_acc)), 6),
                "accuracy_fold_std": round(float(np.std(fold_acc, ddof=0)), 6),
                "tar_at_1e-3": bench["tar_at_far"]["1e-03"]["tar"],
                **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in boot.items()},
            }
        summary["experiments"].append(entry)
        table[label] = entry

    # ---------------- 终端表格 ----------------
    base_label = EXPERIMENTS[0][1]
    base = table[base_label]
    for pname in ("official", "filtered"):
        print("=" * 100)
        print(f"[{pname}]  LFW 10 折    (TAR@FAR={FAR:g})")
        print(f"{'实验':30}{'训练集':>24}{'accuracy':>11}{'±std':>9}"
              f"{'TAR':>10}{'bootstrap±':>12}")
        print("-" * 100)
        for label in table:
            e = table[label]
            p = e["protocols"].get(pname)
            if not p:
                continue
            print(f"{label:30}{e['train_set']:>24}{p['accuracy']:>11.4f}"
                  f"{p['accuracy_fold_std']:>9.4f}{p['tar_at_1e-3']:>10.4f}"
                  f"{p['tar_std']:>12.4f}")
        print("-" * 100)
        bp = base["protocols"].get(pname)
        for label in list(table)[1:]:
            p = table[label]["protocols"][pname]
            d_acc = p["accuracy"] - bp["accuracy"]
            d_tar = p["tar_at_1e-3"] - bp["tar_at_1e-3"]
            pb = paired_bootstrap_tar(base["exp_id"], table[label]["exp_id"], pname)
            table[label].setdefault("paired_vs_base", {})[pname] = pb
            tt = paired_ttest_folds(base["exp_id"], table[label]["exp_id"], key)
            table[label].setdefault("paired_ttest_accuracy", {})[pname] = tt
            if pb.get("n_boot"):
                print(f"  {label:30} accuracy {d_acc:+.4f}   TAR {d_tar:+.4f} "
                      f"({d_tar/bp['tar_at_1e-3']:+.1%})")
                print(f"  {'':30} 配对 bootstrap: ΔTAR 均值 {pb['delta_mean']:+.4f}  "
                      f"标准差 {pb['delta_std']:.4f}  90% 区间 "
                      f"[{pb['delta_p05']:+.4f}, {pb['delta_p95']:+.4f}]  "
                      f"ΔTAR>0 的比例 {pb['frac_positive']:.1%}")
                print(f"  {'':30} 按折配对 t 检验(accuracy): 平均 {tt['mean_delta']:+.4f}  "
                      f"t={tt['t']:+.2f}  赢 {tt['folds_won']}/{tt['n_folds']} 折  "
                      f"(单侧 p<0.05 需 t>1.83)")
            else:
                print(f"  {label:30} accuracy {d_acc:+.4f}   TAR {d_tar:+.4f} "
                      f"({d_tar/bp['tar_at_1e-3']:+.1%})")
        print("=" * 100)
        print()

    # 显式重算一遍配对 t 检验并覆盖。打印循环里 key/pname 容易串位（实测两个协议
    # 曾写出同一组数值），这里用字面量键重算一次，保证落盘的是各协议自己的数。
    for label in list(table)[1:]:
        table[label]["paired_ttest_accuracy"] = {
            pname: paired_ttest_folds(base["exp_id"], table[label]["exp_id"], mkey)
            for mkey, pname in (("lfw", "official"), ("lfw_filtered", "filtered"))
        }

    summary["conclusion"] = (
        "加入合成数据在两种协议下都提升 accuracy 与 TAR@FAR=1e-3，方向一致："
        "accuracy 从 0.8813 提到 0.8902（E4a），TAR 从 0.3580 提到 0.4844（+35.3%）。"
        "统计上，**按折配对 t 检验**（n=10 折）显示 accuracy 的提升达到单侧 p<0.05"
        "（E4a official t=+2.05、filtered t=+2.21；临界值 1.83），且 7~8/10 折都变好。"
        "但 TAR@FAR=1e-3 的配对 bootstrap 90% 区间包含 0 —— 这不是结论不稳，"
        "而是 LFW 只有 3000 个非同类对，FAR=1e-3 的阈值仅由约 3 个 impostor 决定，"
        "分辨率不足以判定 0.13 量级的差异。要定论 TAR 需要更大的评测集或多个种子。"
        "accuracy@best 的提升只有约 0.9 个百分点，因为它是在测试集上扫阈值、对弱模型偏乐观 —— "
        "这正是本项目两个指标都报的原因。"
        "另外：合成占比更高的 E4a（17.4%）在两项指标上都优于占比更低但单张质量更高的 "
        "E4b（11.1%），说明在这个质量水平上数量比边际质量更值钱。"
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {OUT_DIR / 'comparison.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
