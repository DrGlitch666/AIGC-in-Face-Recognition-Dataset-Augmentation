#!/usr/bin/env python
"""构造「大样本留出验证协议」（held-out large-sample verification protocol）

## 为什么需要它

LFW 官方协议有两个硬伤：

1. **分辨率不足**：官方只有 3000 个非同类对，FAR=1e-3 时阈值只由
   `floor(3000 × 0.001) = 3` 个 impostor 决定。3 个样本决定阈值，
   根本没法判定 0.03 量级的 TAR 差异 —— 我们的 E4 对照就卡在这里。
2. **训练/测试泄漏**：官方 6000 对覆盖 4281 个身份，其中**包含我们训练用的 96 个身份**。
   识别模型见过这些人，测出来的数字偏乐观。

## 做法

LFW 共 **5749 个身份**，我们训练只用 96 个，**剩下 5653 个从未参与训练**。
从中构造对子：

* `same` 对：同一身份的**不同**两张图
* `diff` 对：两个不同身份各取一张

两条关键保证：

* **身份与训练身份交集为 0** —— 脚本最后会断言这一点，并在输出里报告
* 非同类对数可做到**数万** → FAR=1e-3 下有几十个 impostor，分辨率提升一个数量级

> ⚠️ 这是一个**补充协议**，不是 LFW 官方协议。报告里必须写清楚两者并存，
> 不能拿它冒充官方数字。

用法：

    python scripts/build_verify_protocol.py --n-same 25000 --n-diff 25000
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.stdout.reconfigure(errors="replace")

from aigcfr.data.lfw import Pair  # noqa: E402
from aigcfr.utils.config import find_local_config, load_config  # noqa: E402

# LFW 原始文件名形如 George_W_Bush_0001.jpg —— 身份名里本身含下划线，
# 所以要从**右边**匹配 `_<4 位数字>.jpg` 才能正确切出身份。
NAME_RE = re.compile(r"^(?P<ident>.+)_(?P<idx>\d{4})\.jpg$")


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False, description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-same", type=int, default=0,
                    help="同人对数；0=用满全部唯一组合（推荐，重复抽样会虚高置信度）")
    ap.add_argument("--n-diff", type=int, default=0,
                    help="异人对数；0=取 same 的 10 倍（TAR@FAR 只依赖 impostor 分布，"
                         "不平衡不影响 TAR；accuracy 会在评测端用平衡子集）")
    ap.add_argument("--min-images", type=int, default=2,
                    help="留出身份至少要有几张图才能参与（same 对需要 ≥2）")
    ap.add_argument("--folds", type=int, default=10, help="折数（沿用 LFW 的 10 折惯例）")
    ap.add_argument("--seed", type=int, default=0, help="固定种子保证可复现")
    ap.add_argument("--raw-root", default=None,
                    help="原始 LFW 根目录，默认 <data_root>/raw/lfw/lfw")
    ap.add_argument("--train-manifest", default=None,
                    help="训练清单（用来排除训练身份），默认取配置里的 data.manifest")
    ap.add_argument("--out", default="data/manifests/verify_heldout.jsonl")
    args = ap.parse_args()

    cfg = load_config(None, profile="auto", local_path=str(find_local_config()))
    data_root = Path(cfg["paths"]["data_root"]).resolve()
    raw_root = (Path(args.raw_root).resolve() if args.raw_root
                else data_root / "raw" / "lfw" / "lfw")
    if not raw_root.is_dir():
        print(f"!! 找不到原始 LFW 目录: {raw_root}")
        return 2

    train_manifest = (Path(args.train_manifest) if args.train_manifest
                      else REPO_ROOT / cfg["data"]["manifest"])
    if not train_manifest.exists():
        print(f"!! 找不到训练清单: {train_manifest}")
        return 2

    # ---------------- 1) 训练身份 ----------------
    train_ids = set()
    for line in train_manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("split") == "train" and r.get("status") == "accepted":
                train_ids.add(r["identity_id"])
    print(f"训练身份 {len(train_ids)} 个（来自 {train_manifest.name}）")

    # ---------------- 2) 扫描全部 LFW 身份 ----------------
    print(f"扫描原始 LFW: {raw_root}")
    by_ident: dict[str, list[int]] = {}
    for id_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        idxs = []
        for f in id_dir.glob("*.jpg"):
            m = NAME_RE.match(f.name)
            if m and m.group("ident") == id_dir.name:
                idxs.append(int(m.group("idx")))
        if idxs:
            by_ident[id_dir.name] = sorted(idxs)
    n_img = sum(len(v) for v in by_ident.values())
    print(f"  身份 {len(by_ident)} 个，图片 {n_img} 张")

    # ---------------- 3) 留出身份 ----------------
    held = {k: v for k, v in by_ident.items()
            if k not in train_ids and len(v) >= max(args.min_images, 2)}
    overlap = set(held) & train_ids
    print(f"  留出身份 {len(held)} 个（排除训练身份后，且 ≥{max(args.min_images,2)} 张图）")
    print(f"  留出身份中可用于 same 对的图片 {sum(len(v) for v in held.values())} 张")
    if len(held) < 100:
        print("!! 留出身份太少，检查 raw-root 与 train-manifest 是否匹配")
        return 2

    # ---------------- 4) 构造对子 ----------------
    # ⚠️ `Pair` 是 frozen dataclass，**构造后不能改 fold** —— 所以先把
    #    (kind, id_a, idx_a, id_b, idx_b) 攒成列表，打乱后再带上 fold 构造 Pair。
    #
    # ⚠️ same 对必须**全部唯一**：同一个组合被抽中多次等于伪重复，
    #    会让置信区间虚窄。diff 那边池子极大（1500 万量级身份组合），不担心重复。
    rng = random.Random(args.seed)
    idents = sorted(held)

    all_same = [(ident, a, b)
                for ident in idents
                for a, b in itertools.combinations(held[ident], 2)]
    rng.shuffle(all_same)
    n_same = len(all_same) if args.n_same <= 0 else min(args.n_same, len(all_same))
    if args.n_same > len(all_same):
        print(f"  ! 请求 {args.n_same} 个 same 对，但唯一组合只有 {len(all_same)} 个 -> 用满")
    n_diff = args.n_diff if args.n_diff > 0 else n_same * 10

    specs: list[tuple[str, str, int, str, int]] = [
        ("same", ident, a, ident, b) for ident, a, b in all_same[:n_same]]
    for _ in range(n_diff):
        ia, ib = rng.sample(idents, 2)             # 两个不同身份
        specs.append(("diff", ia, rng.choice(held[ia]), ib, rng.choice(held[ib])))

    # 打乱后轮转分折（保证每折 same/diff 比例接近）
    rng.shuffle(specs)
    pairs = [Pair(fold=(i % args.folds) + 1, kind=k, id_a=ia, idx_a=xa, id_b=ib, idx_b=xb)
             for i, (k, ia, xa, ib, xb) in enumerate(specs)]

    # ---------------- 5) 自检 ----------------
    bad = [p for p in pairs if p.is_same and p.id_a != p.id_b]
    assert not bad, f"same 对里出现不同身份: {bad[:3]}"
    assert all((p.id_a not in train_ids) and (p.id_b not in train_ids) for p in pairs), \
        "有对子用到了训练身份！"
    n_same = sum(1 for p in pairs if p.is_same)
    used_ids = {p.id_a for p in pairs} | {p.id_b for p in pairs}
    print()
    print("=" * 74)
    print("协议自检")
    print("=" * 74)
    print(f"  对子总数      {len(pairs)}（same {n_same} / diff {len(pairs)-n_same}）")
    print(f"  参与身份      {len(used_ids)} 个")
    print(f"  **与训练身份的交集 = {len(used_ids & train_ids)}**（必须是 0）")
    print(f"  FAR=1e-3 下决定阈值的 impostor 数 = floor({len(pairs)-n_same} × 1e-3) "
          f"= {int((len(pairs)-n_same) * 1e-3)}")
    print(f"  （LFW 官方协议只有 3000 个非同类对 -> 只有 3 个）")

    # ---------------- 6) 落盘 ----------------
    out = (REPO_ROOT / args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps({
                "fold": p.fold, "kind": p.kind,
                "id_a": p.id_a, "idx_a": p.idx_a,
                "id_b": p.id_b, "idx_b": p.idx_b, "label": p.label,
                "relpath_a": p.relpath_a, "relpath_b": p.relpath_b,
            }, ensure_ascii=False) + "\n")
    meta = {"protocol": "heldout-large-sample", "n_pairs": len(pairs), "n_same": n_same,
            "n_diff": len(pairs) - n_same, "n_identities": len(used_ids),
            "n_train_identities": len(train_ids),
            "train_identities_overlap": len(used_ids & train_ids),
            "folds": args.folds, "seed": args.seed, "raw_root": str(raw_root),
            "impostors_at_far_1e-3": int((len(pairs) - n_same) * 1e-3),
            "note": ("补充协议，不是 LFW 官方协议。identity 与训练身份交集为 0，"
                     "因此不存在训练/测试泄漏；非同类对数远多于官方，"
                     "FAR=1e-3 的分辨率显著更好。"),
            "pairs_manifest": str(out.relative_to(REPO_ROOT))}
    (out.parent / "verify_heldout.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  对子清单: {out}")
    print(f"  协议说明: {out.parent / 'verify_heldout.meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
