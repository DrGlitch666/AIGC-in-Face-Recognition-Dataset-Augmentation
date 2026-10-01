#!/usr/bin/env python
"""构建对齐后的训练集 + 写 manifest.jsonl（契约 A）

## 用法

    # 完整构建（LFW 中图片数 >= 15 的 96 个身份，3595 张图）
    python scripts/build_dataset.py --config configs/data/lfw.yaml --min-images 15

    # 快速自测（只做 3 个身份，输出到工作区内）
    python scripts/build_dataset.py --config configs/data/lfw.yaml --identity-limit 3 --out .smoke_aligned --manifest .smoke_aligned/manifest.jsonl

## 产出

    <data_root>/processed/aligned/<身份>/<文件名>.png    对齐后的 112x112 人脸
    data/manifests/toy_train.jsonl                       契约 A 的数据清单（入库）
    <out>/build_report.json                              本次构建统计（成功率、检测质量）

## 契约 A 的一处约定（与 docs/FRAMEWORK.md §3.3 的措辞不同，以此为准）

`path` 字段是**相对 `paths.data_root`** 的路径，**不是**相对仓库根。
原因：原始数据与对齐结果都放在仓库之外的独立盘（见 configs/local.<机器>.yaml），
两台机器的 data_root 不同，所以清单必须存相对路径才能互通。
（后续会同步更新 FRAMEWORK 的措辞。）
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aigcfr.data.align import ARCFACE_SIZE, align_by_kps, align_stats  # noqa: E402
from aigcfr.data.lfw import list_identities, select_training_identities  # noqa: E402
from aigcfr.eval.embed import assert_provider, load_app, pick_face, provider_report, warmup  # noqa: E402
from aigcfr.utils.config import find_local_config, load_config  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:  # noqa: BLE001
        pass


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False, description="构建对齐训练集 + manifest")
    ap.add_argument("--config", default="configs/data/lfw.yaml")
    ap.add_argument("--min-images", type=int, default=15, help="选身份的最小图片数")
    ap.add_argument("--identity-limit", type=int, default=None, help="只做前 N 个身份（自测用）")
    ap.add_argument("--out", default=None, help="对齐结果输出目录（默认 <data_root>/processed/aligned）")
    ap.add_argument("--manifest", default=None, help="manifest 输出路径（默认 data/manifests/toy_train.jsonl）")
    ap.add_argument("--face-select", default=None, choices=["center", "score", "area"])
    ap.add_argument("--det-size", type=int, default=None)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    # ---------------- 配置 ----------------
    local_file = find_local_config()
    if local_file is None:
        print("!! 找不到 configs/local.<machine>.yaml")
        return 2
    cfg = load_config(None, profile="auto", local_path=str(local_file))
    data_root = Path(cfg["paths"]["data_root"])
    models_root = Path(cfg["paths"].get("models_root", data_root.parent / "models"))
    runtime = cfg.get("runtime", {})
    eval_cfg = cfg.get("eval", {})

    det_size = args.det_size or int(runtime.get("onnx_det_size", 640))
    face_select = args.face_select or eval_cfg.get("face_select", "center")
    providers = (["CPUExecutionProvider"] if args.cpu
                 else list(runtime.get("onnx_providers", ["CUDAExecutionProvider", "CPUExecutionProvider"])))

    out_dir = Path(args.out).resolve() if args.out else (data_root / "processed" / "aligned")
    manifest_path = (Path(args.manifest).resolve() if args.manifest
                     else REPO_ROOT / "data" / "manifests" / "toy_train.jsonl")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------------- 选身份 ----------------
    print("=" * 70)
    print(f"数据根目录 : {data_root}")
    print(f"对齐输出   : {out_dir}")
    print(f"manifest   : {manifest_path}")
    print(f"det_size={det_size}  face_select={face_select}  providers={providers}")
    print("=" * 70)

    lfw_root = data_root / "raw" / "lfw"
    img_dir = lfw_root / "lfw"
    if not img_dir.is_dir():
        print(f"!! 找不到 LFW 图片目录: {img_dir}（先跑 scripts/download_data.py）")
        return 2

    print("\n[1/3] 扫描身份 ...")
    identities = list_identities(img_dir)
    train_ids = select_training_identities(identities, min_images=args.min_images)
    if args.identity_limit:
        train_ids = train_ids[: args.identity_limit]
        print(f"  [--identity-limit {args.identity_limit}] 只做前 {len(train_ids)} 个身份")
    total_imgs = sum(len(identities[i]) for i in train_ids)
    print(f"  身份 {len(train_ids)} 个 | 图片 {total_imgs} 张")

    # ---------------- 模型 ----------------
    print("\n[2/3] 加载检测模型并逐张对齐 ...")
    app = load_app(models_root / "insightface", providers=providers, det_size=det_size,
                   ctx_id=0 if not args.cpu else -1)
    assert_provider(app, expect_cuda=not args.cpu)
    for name, provs in provider_report(app).items():
        print(f"    {name:12} {provs}")
    if not args.cpu:
        print(f"  CUDA 预热 {warmup(app):.1f}s")

    import cv2  # noqa: PLC0415

    records: list[dict] = []
    failures: list[dict] = []
    n_multi = 0
    t0 = time.time()
    done = 0
    for ident in train_ids:
        (out_dir / ident).mkdir(parents=True, exist_ok=True)
        for fname in identities[ident]:
            done += 1
            img = cv2.imread(str(img_dir / ident / fname))
            if img is None:
                failures.append({"identity": ident, "file": fname, "reason": "read_failed"})
                continue
            faces = app.get(img)
            if not faces:
                failures.append({"identity": ident, "file": fname, "reason": "no_face"})
                continue
            if len(faces) > 1:
                n_multi += 1
            face = pick_face(faces, img.shape, face_select)
            aligned = align_by_kps(img, face.kps, ARCFACE_SIZE)

            stem = Path(fname).stem
            target = out_dir / ident / f"{stem}.png"
            if not cv2.imwrite(str(target), aligned):
                failures.append({"identity": ident, "file": fname, "reason": "write_failed"})
                continue
            # ⚠️ manifest 里的 path 必须**相对 data_root**（契约 A），
            #    消费方（train.py）就是这么解析的；写成"相对对齐目录"会读不到文件（真实踩过）。
            try:
                rel = target.relative_to(data_root).as_posix()
            except ValueError:
                # --out 指到了 data_root 之外（自测场景）：退回绝对路径并提示
                rel = target.as_posix()
                if not records:
                    print(f"  [!] --out 不在 data_root 下，manifest 里将写绝对路径（仅供自测）")

            stats = align_stats(aligned)
            records.append({
                "image_id": f"lfw_{ident}_{stem.split('_')[-1]}",
                "path": rel,                     # 相对 data_root（见模块文档的说明）
                "identity_id": ident,
                "source": "real",
                "generator": None,
                "parent_image_id": None,
                "split": "train",
                "status": "accepted",
                "meta": {
                    "origin": f"raw/lfw/lfw/{ident}/{fname}",
                    "det_score": round(float(face.det_score), 4),
                    "n_faces": len(faces),
                    "face_select": face_select,
                    "align": {"size": stats["size"], "mean": round(stats["mean"], 2),
                              "std": round(stats["std"], 2)},
                    "suspicious": stats["is_suspicious"],
                },
            })
            if done % 500 == 0:
                rate = done / max(time.time() - t0, 1e-6)
                print(f"  {done}/{total_imgs}  {rate:.1f} 张/秒")

    elapsed = time.time() - t0

    # ---------------- 写 manifest ----------------
    print("\n[3/3] 写 manifest 与报告 ...")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    from collections import Counter
    fail_reasons = Counter(f["reason"] for f in failures)
    suspicious = sum(1 for r in records if r["meta"]["suspicious"])
    det_scores = [r["meta"]["det_score"] for r in records]
    import statistics

    report = {
        "built_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "config_hash": cfg["_meta"]["config_hash"],
        "source": str(img_dir),
        "out_dir": str(out_dir),
        "manifest": str(manifest_path),
        "min_images": args.min_images,
        "identity_limit": args.identity_limit,
        "det_size": det_size,
        "face_select": face_select,
        "n_identities": len(train_ids),
        "n_images_input": total_imgs,
        "n_images_aligned": len(records),
        "n_failed": len(failures),
        "failure_reasons": dict(fail_reasons),
        "n_multi_face": n_multi,
        "multi_face_ratio": round(n_multi / max(total_imgs, 1), 4),
        "det_score_mean": round(statistics.fmean(det_scores), 4) if det_scores else None,
        "det_score_min": round(min(det_scores), 4) if det_scores else None,
        "suspicious_aligned": suspicious,
        "elapsed_seconds": round(elapsed, 1),
        "images_per_second": round(len(records) / max(elapsed, 1e-6), 2),
        "sample_failures": failures[:20],
    }
    (out_dir / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"  对齐成功 {len(records)} / 失败 {len(failures)}  {dict(fail_reasons) if failures else ''}")
    print(f"  多脸图 {n_multi} 张（{report['multi_face_ratio']*100:.1f}%）| "
          f"可疑对齐 {suspicious} 张 | det_score 均值 {report['det_score_mean']}")
    print(f"  用时 {elapsed:.0f}s（{report['images_per_second']} 张/秒）")
    print(f"  manifest : {manifest_path}  （{len(records)} 行）")
    print(f"  报告     : {out_dir / 'build_report.json'}")
    print("\n完成 ✅  下一步：python scripts/train.py --exp-id e1-real-only-seed0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
