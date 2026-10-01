#!/usr/bin/env python
"""合成图筛选与身份保持度量（任务卡 #11）

## 它做什么

对每张合成图算一个**身份相似度**：

    id_sim = cosine( 合成图的 embedding , 该身份**真实图**的 embedding 中心 )

这是"像不像本人"的**数字化判据** —— 比肉眼看更客观，而且能定阈值。
（用 L2 归一化后的余弦，见 `aigcfr.eval.verify.l2_normalize` 的说明。）

## 用法

    # 只看数字（不改 manifest）
    python scripts/filter_synth.py --exp-id syn-faceid --report-only

    # 按阈值写入 accepted / rejected
    python scripts/filter_synth.py --exp-id syn-faceid --threshold 0.45

## 产出

    results/runs/<exp-id>/filter_report.json     整体分布 + 逐身份统计
    results/runs/<exp-id>/synth_scores.csv       每张图的分数明细
    data/manifests/<exp-id>.jsonl                原地更新 status 与 meta.id_sim
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aigcfr.eval.faces import identity_embedding  # noqa: E402
from aigcfr.eval.verify import l2_normalize  # noqa: E402
from aigcfr.utils.config import find_local_config, load_config  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:  # noqa: BLE001
        pass


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False, description="合成图筛选 / 身份保持度量")
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--manifest", default=None, help="合成图 manifest（默认 data/manifests/<exp-id>.jsonl）")
    ap.add_argument("--train-manifest", default=None, help="真实训练集 manifest（默认取配置里的）")
    ap.add_argument("--threshold", type=float, default=None, help="id_sim 低于它就标 rejected")
    ap.add_argument("--report-only", action="store_true", help="只出报告，不改 manifest")
    # ⚠️ 默认 320 而不是 640：见下面 load_app 处的说明（实测 640 会一张脸都检不出）
    ap.add_argument("--det-size", type=int, default=320,
                    help="检测输入尺寸（默认 320；生成图/对齐图用 640 会检不出人脸）")
    args = ap.parse_args()

    local_file = find_local_config()
    if local_file is None:
        print("!! 找不到 configs/local.<machine>.yaml")
        return 2
    cfg = load_config(None, profile="auto", local_path=str(local_file))
    paths = cfg.get("paths", {})
    data_root = Path(paths["data_root"])
    models_root = Path(paths.get("models_root", data_root.parent / "models"))
    runtime = cfg.get("runtime", {})
    # ⚠️⚠️ 这里**不能**用配置里的 onnx_det_size（640）。实测（2026-10-01）：
    #
    #     det_size   112x112 对齐图   512x512 生成图
    #       160           1 张            1 张
    #       320           1 张            1 张     <- 用这个
    #       480           0 张            1 张
    #       640           0 张            0 张     <- 一张都检不出
    #
    #   insightface 会把图像**长边缩放到 det_size**：对 112x112 的对齐图，
    #   640 意味着放大 5.7 倍（人脸占满整个输入框），det_10g 反而失效。
    #   结果就是"可评分 0/N"—— 一度被误判成"生成的图崩坏"，其实是检测配置错了。
    #
    #   注意：**E0/E1 评测仍然用 640**，因为那边输入是 250x250 的 LFW 原图
    #   （人脸约占 40%），640 下检出正常且拿到了 0.9985 的准确率 —— 不要改那边。
    det_size = args.det_size

    synth_manifest = (Path(args.manifest).resolve() if args.manifest
                      else REPO_ROOT / "data" / "manifests" / f"{args.exp_id}.jsonl")
    train_manifest = (Path(args.train_manifest).resolve() if args.train_manifest
                      else REPO_ROOT / (cfg.get("data", {}).get("manifest",
                                                             "data/manifests/toy_train.jsonl")))
    out_dir = REPO_ROOT / "results" / "runs" / args.exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in (synth_manifest, train_manifest):
        if not p.exists():
            print(f"!! 找不到 manifest: {p}")
            return 2

    print("=" * 72)
    print(f"实验      : {args.exp_id}")
    print(f"合成图    : {synth_manifest}")
    print(f"真实图    : {train_manifest}")
    print(f"阈值      : {args.threshold if args.threshold is not None else '(只报告，不筛选)'}")
    print("=" * 72)

    # ---------------- 读两个 manifest ----------------
    def read_records(path: Path) -> list[dict]:
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    synth = [r for r in read_records(synth_manifest) if r.get("source") == "synth"]
    real = [r for r in read_records(train_manifest)
            if r.get("source") == "real" and r.get("status") == "accepted"]
    print(f"\n[1/4] 合成图 {len(synth)} 张 | 真实图 {len(real)} 张")

    # ---------------- 提特征（统一用 buffalo_l，与 E0/E1 评测一致）----------------
    print("\n[2/4] 提特征（buffalo_l）...")
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: F401,PLC0415  —— 必须在 onnxruntime 之前

    from aigcfr.eval.embed import assert_provider, load_app, pick_face, provider_report, warmup

    app = load_app(models_root / "insightface",
                   providers=list(runtime.get("onnx_providers",
                                              ["CUDAExecutionProvider", "CPUExecutionProvider"])),
                   det_size=det_size, ctx_id=0)
    assert_provider(app, expect_cuda=True)
    print(f"    providers = {provider_report(app)}")
    print(f"    预热 {warmup(app):.1f}s")

    def embed(relpath: str) -> np.ndarray | None:
        p = Path(relpath)
        return identity_embedding(app, p if p.is_absolute() else data_root / p)

    # 真实图：按身份求中心（这是"本人长什么样"的参考）
    real_by_id: dict[str, list[np.ndarray]] = {}
    n_real_fail = 0
    for r in real:
        vec = embed(r["path"])
        if vec is None:
            n_real_fail += 1
            continue
        real_by_id.setdefault(r["identity_id"], []).append(vec)
    centroids = {k: l2_normalize(np.mean(np.stack(v), axis=0)) for k, v in real_by_id.items()}
    print(f"    真实图成功 {sum(len(v) for v in real_by_id.values())} 张（失败 {n_real_fail}）"
          f"，覆盖 {len(centroids)} 个身份")

    # ---------------- 逐张算身份相似度 ----------------
    print("\n[3/4] 计算身份相似度 ...")
    rows: list[dict] = []
    updated: list[dict] = []
    for rec in synth:
        vec = embed(rec["path"])
        centroid = centroids.get(rec["identity_id"])
        if vec is None:
            sim, note = None, "no_face"
        elif centroid is None:
            sim, note = None, "no_real_centroid"
        else:
            sim, note = float(l2_normalize(vec) @ centroid), "ok"
        rec.setdefault("meta", {})["id_sim"] = None if sim is None else round(sim, 6)
        if args.threshold is not None:
            rec["status"] = "rejected" if (sim is None or sim < args.threshold) else "accepted"
            rec["meta"]["reject_reason"] = note if sim is None else (
                None if sim >= args.threshold else f"id_sim {sim:.3f} < {args.threshold}")
        rows.append({"image_id": rec["image_id"], "identity_id": rec["identity_id"],
                     "status": rec.get("status"), "id_sim": sim, "note": note,
                     "path": rec["path"]})
        updated.append(rec)

    ok = [r for r in rows if r["id_sim"] is not None]
    sims = [r["id_sim"] for r in ok]
    print(f"    可评分 {len(ok)}/{len(rows)} 张")

    # ---------------- 报告 ----------------
    print("\n[4/4] 写报告 ...")
    per_identity: dict[str, list[float]] = {}
    for r in ok:
        per_identity.setdefault(r["identity_id"], []).append(r["id_sim"])

    summary = {
        "n_synth": len(rows),
        "n_scored": len(ok),
        "n_failed": len(rows) - len(ok),
        "id_sim_mean": round(statistics.fmean(sims), 4) if sims else None,
        "id_sim_median": round(statistics.median(sims), 4) if sims else None,
        "id_sim_min": round(min(sims), 4) if sims else None,
        "id_sim_max": round(max(sims), 4) if sims else None,
        "id_sim_stdev": round(statistics.pstdev(sims), 4) if len(sims) > 1 else None,
        "threshold": args.threshold,
        "n_accepted": sum(1 for r in rows if r["status"] == "accepted"),
        "n_rejected": sum(1 for r in rows if r["status"] == "rejected"),
        "identities": {
            k: {"n": len(v), "mean": round(statistics.fmean(v), 4),
                "min": round(min(v), 4), "max": round(max(v), 4)}
            for k, v in sorted(per_identity.items())
        },
    }
    report = {
        "exp_id": args.exp_id,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "embedding_model": "buffalo_l (w600k_r50)",
        "reference": "center of accepted REAL images per identity",
        "summary": summary,
    }
    (out_dir / "filter_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                encoding="utf-8")

    with open(out_dir / "synth_scores.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["image_id", "identity_id", "status",
                                                "id_sim", "note", "path"])
        writer.writeheader()
        for r in rows:
            writer.writerow({**r, "id_sim": "" if r["id_sim"] is None else f"{r['id_sim']:.6f}"})

    if args.threshold is not None and not args.report_only:
        with open(synth_manifest, "w", encoding="utf-8") as fh:
            for rec in updated:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"    manifest 已更新: {synth_manifest}")

    print()
    print("=" * 72)
    print("身份相似度（id_sim = 与本人真实图中心的余弦）")
    print("=" * 72)
    if sims:
        print(f"  均值 {summary['id_sim_mean']:+.4f} | 中位数 {summary['id_sim_median']:+.4f} | "
              f"范围 [{summary['id_sim_min']:+.4f}, {summary['id_sim_max']:+.4f}]")
        print("  参考：同一人不同真实照片的相似度通常 > 0.4；不同人通常 < 0.25")
        for k, v in summary["identities"].items():
            print(f"    {k[:32]:34} n={v['n']:<4} 均值 {v['mean']:+.4f}")
    else:
        print("  （没有可评分的图）")
    if args.threshold is not None:
        print(f"\n  阈值 {args.threshold}: 通过 {summary['n_accepted']} / "
              f"拒绝 {summary['n_rejected']}")
    print(f"\n  报告: {out_dir / 'filter_report.json'}")
    print(f"  明细: {out_dir / 'synth_scores.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
