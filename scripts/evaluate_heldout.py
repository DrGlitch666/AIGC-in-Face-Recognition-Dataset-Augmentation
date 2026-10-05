#!/usr/bin/env python
"""在「大样本留出协议」上评测（配合 build_verify_protocol.py）

## 它回答什么问题

LFW 官方协议只有 3000 个非同类对 -> FAR=1e-3 的阈值只由 **3 个 impostor** 决定，
分辨率不足以判定 0.03 量级的 TAR 差异；而且官方 6000 对里**包含我们的训练身份**（泄漏）。
本脚本在**身份与训练集交集为 0**、且非同类对多两个数量级的协议上重测。

## 两个数字分开报（重要）

协议是**不平衡**的（same 1.1 万 / diff 11 万），这没问题 ——
**TAR@FAR 只依赖 impostor 分布**，same 的多少只影响估计精度。
但 `accuracy@best` 在不平衡集上会被多数类主导、失去意义，
所以 accuracy **只在平衡子集上算**（same 全用，diff 随机取等量）。

## 用法

    # 自研 checkpoint
    python scripts/evaluate_heldout.py --exp-id e4-real-plus-synth-seed0 \
        --ckpt F:/aigcfr/ckpt/e4-real-plus-synth-seed0/best.pth

    # 或零样本基线 buffalo_l
    python scripts/evaluate_heldout.py --exp-id e0-buffalo_l-heldout
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.stdout.reconfigure(errors="replace")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from aigcfr.data.lfw import Pair  # noqa: E402
from aigcfr.eval.embed import extract, load_app, provider_report, warmup  # noqa: E402
from aigcfr.eval.faces import build_aligned_cache  # noqa: E402
from aigcfr.eval.torch_embed import extract_aligned, load_checkpoint  # noqa: E402
from aigcfr.eval.verify import pair_scores, summarize  # noqa: E402
from aigcfr.utils.config import find_local_config, load_config  # noqa: E402

PROTOCOL_KEY = "lfw_heldout"


def load_pairs(path: Path) -> list[Pair]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out.append(Pair(fold=int(r["fold"]), kind=r["kind"], id_a=r["id_a"],
                        idx_a=int(r["idx_a"]), id_b=r["id_b"], idx_b=int(r["idx_b"])))
    return out


def balanced_subset(pairs: list[Pair], seed: int = 0) -> list[Pair]:
    """same 全用，diff 随机取等量 —— accuracy@best 在不平衡集上没有意义。"""
    same = [p for p in pairs if p.is_same]
    diff = [p for p in pairs if not p.is_same]
    rng = random.Random(seed)
    rng.shuffle(diff)
    return same + diff[: len(same)]


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False, description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--ckpt", default=None, help="自研 checkpoint；不给则评 buffalo_l（零样本）")
    ap.add_argument("--pairs", default="data/manifests/verify_heldout.jsonl")
    ap.add_argument("--raw-root", default=None, help="默认 <data_root>/raw/lfw/lfw")
    ap.add_argument("--out", default=None, help="结果目录（默认 results/runs/<exp-id>）")
    ap.add_argument("--align-cache", default=None)
    ap.add_argument("--det-size", type=int, default=None)
    ap.add_argument("--face-select", default=None, choices=["center", "score", "area"])
    ap.add_argument("--limit", type=int, default=None, help="只用前 N 对（自测用）")
    ap.add_argument("--no-cache", action="store_true",
                    help="不写 embedding 缓存（自测用；正常跑建议留缓存，重评很快）")
    args = ap.parse_args()

    cfg = load_config(None, profile="auto", local_path=str(find_local_config()))
    paths, runtime = cfg["paths"], cfg.get("runtime", {})
    data_root = Path(paths["data_root"]).resolve()
    models_root = Path(paths["models_root"])
    eval_cfg = cfg.get("eval", {})
    det_size = args.det_size or int(runtime.get("onnx_det_size", 640))
    face_select = args.face_select or eval_cfg.get("face_select", "center")

    pairs_path = (REPO_ROOT / args.pairs).resolve() if not Path(args.pairs).is_absolute() \
        else Path(args.pairs)
    if not pairs_path.exists():
        print(f"!! 找不到对子清单: {pairs_path}")
        print("   先跑：python scripts/build_verify_protocol.py")
        return 2
    pairs = load_pairs(pairs_path)
    if args.limit:
        pairs = pairs[: args.limit]
    raw_root = (Path(args.raw_root).resolve() if args.raw_root
                else data_root / "raw" / "lfw" / "lfw")
    out_dir = (Path(args.out).resolve() if args.out
               else REPO_ROOT / "results" / "runs" / args.exp_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_same = sum(1 for p in pairs if p.is_same)
    print("=" * 78)
    print(f"大样本留出协议评测   exp={args.exp_id}")
    print(f"  对子: {len(pairs)}（same {n_same} / diff {len(pairs)-n_same}）")
    print(f"  图片根目录: {raw_root}")
    print(f"  det_size={det_size}  face_select={face_select}")
    print("=" * 78)

    app = load_app(models_root / "insightface",
                   providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                   det_size=det_size, ctx_id=0)
    warmup(app)

    needed = sorted({p.relpath_a for p in pairs} | {p.relpath_b for p in pairs})
    print(f"\n[1/4] 需要 {len(needed)} 张图，做对齐（有缓存会命中）...")
    aligned_dir = (Path(args.align_cache).resolve() if args.align_cache
                   else data_root / "cache" / "aligned_lfw_heldout" / face_select)
    mapping, align_fail, align_stat = build_aligned_cache(
        app, raw_root, needed, aligned_dir, face_select=face_select)
    print(f"      对齐成功 {len(mapping)}/{len(needed)}（失败 {align_fail}）"
          f"  用时 {align_stat.get('seconds', 0):.0f}s")

    # ---------------- 提特征 ----------------
    if args.ckpt:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, classes, ckpt_meta = load_checkpoint(args.ckpt, device)
        print(f"\n[2/4] 自研模型 {ckpt_meta.get('arch')} epoch={ckpt_meta.get('epoch')} "
              f"类别数={len(classes)} device={device}")
        vectors = extract_aligned(model, mapping, device=device,
                                  batch_size=int(runtime.get("batch_size", 128)))
        model_meta = {"arch": ckpt_meta.get("arch"), "epochs": ckpt_meta.get("epoch"),
                      "ckpt": str(args.ckpt), "embedding_source": "torch"}
    else:
        print("\n[2/4] 零样本基线 buffalo_l")
        cache = None if args.no_cache else \
            data_root / "cache" / "embeddings" / "lfw_heldout_center.npz"
        emb = extract(app, raw_root, needed, cache_path=cache, face_select=face_select)
        vectors = emb.vectors
        model_meta = {"arch": "buffalo_l (w600k_r50)", "epochs": 0,
                      "embedding_source": "onnx",
                      "onnx_providers": provider_report(app).get("recognition", [])}

    print(f"      取得特征 {len(vectors)}/{len(needed)}")

    # ---------------- 评分 ----------------
    print("\n[3/4] 计算指标 ...")
    sc_full = pair_scores(pairs, vectors)
    m_full = summarize(sc_full)
    sub = balanced_subset(pairs)
    sc_bal = pair_scores(sub, vectors)
    m_bal = summarize(sc_bal)
    print(f"      [full   ] 有效 {len(sc_full.scores)}/{len(pairs)} 对  "
          f"TAR@FAR=1e-3 {m_full.get('tar_at_far', {}).get('1e-03', {}).get('tar')}")
    print(f"      [balanced] 有效 {len(sc_bal.scores)}/{len(sub)} 对  "
          f"accuracy {m_bal.get('value')}")

    # ---------------- 合并进 metrics.json ----------------
    print("\n[4/4] 写结果 ...")
    mp = out_dir / "metrics.json"
    metrics = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {"exp_id": args.exp_id}
    metrics.setdefault("benchmarks", {})[PROTOCOL_KEY] = {
        **m_bal,                                    # accuracy 用平衡子集
        "protocol": "heldout-large-sample (identities disjoint from training)",
        "n_pairs_full": len(sc_full.scores),
        "n_same_full": n_same,
        "tar_at_far_full": m_full.get("tar_at_far", {}),
        "accuracy_note": "accuracy 在平衡子集上算（same 全用 + 等量 diff）；"
                         "TAR@FAR 在全量集上算（只依赖 impostor 分布）",
        "impostors_at_far_1e-3": int((len(sc_full.labels) - int((sc_full.labels == 1).sum()))
                                     * 1e-3) if len(sc_full.labels) else 0,
        "model": model_meta,
    }
    metrics.setdefault("protocol_detail", {})[PROTOCOL_KEY] = {
        "pairs_manifest": str(pairs_path.relative_to(REPO_ROOT)),
        "n_pairs": len(pairs), "n_same": n_same, "n_diff": len(pairs) - n_same,
        "n_images_needed": len(needed), "n_aligned": len(mapping), "n_align_failed": align_fail,
        "raw_root": str(raw_root), "aligned_dir": str(aligned_dir),
        "evaluated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    mp.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    # 逐对分数落盘，供后续 bootstrap / 配对检验。
    # ⚠️ `Scores` 只返回标签/分数，**不记录哪些对被丢弃**，所以这里自己重算一遍
    #    过滤条件，保证 pairs 与 scores 一一对应（否则会错位，且不会报错）。
    import csv  # noqa: PLC0415
    kept = [p for p in pairs
            if p.relpath_a in vectors and p.relpath_b in vectors]
    if len(kept) != len(sc_full.scores):
        print(f"      ! 警告: 有效对 {len(kept)} 与分数数 {len(sc_full.scores)} 不一致")
    with open(out_dir / "pairs_heldout.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["set", "fold", "kind", "id_a", "file_a",
                                           "id_b", "file_b", "score", "label"])
        w.writeheader()
        for p, s in zip(kept, sc_full.scores):
            w.writerow({"set": "heldout", "fold": p.fold, "kind": p.kind,
                        "id_a": p.id_a, "file_a": p.relpath_a,
                        "id_b": p.id_b, "file_b": p.relpath_b,
                        "score": f"{float(s):.6f}", "label": p.label})
    print(f"      {mp}")
    print(f"      {out_dir / 'pairs_heldout.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
