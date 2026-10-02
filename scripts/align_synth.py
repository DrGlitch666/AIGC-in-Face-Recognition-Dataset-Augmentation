#!/usr/bin/env python
"""把合成图对齐成 112x112，供训练使用

## ⚠️ 为什么必须有这一步

真实图是 `build_dataset.py` 对齐好的 **112x112 裁剪**，而生成出来的是 **512x512 原图**。
直接混进 DataLoader 会崩：

    RuntimeError: stack expects each tensor to be equal size,
                  but got [3, 112, 112] at entry 0 and [3, 512, 512] at entry 14

**但"把 512 缩到 112"是错的** —— 那样人脸只占画面一小块（512 里人脸可能只占 200px，
缩到 112 后只剩 40px），等于给训练喂垃圾。
必须和真实图走**同一套 ArcFace 对齐**（检测 5 点 -> `norm_crop2` 仿射到 112x112）。

## ⚠️ det_size 必须用 320，不能用配置里的 640

实测（见 docs/W3-GENERATION.md §3.6）：insightface 把图像**长边缩放到 det_size**，
生成的 512x512 图在 640 下**经常一张脸都检不出**（在 320 下 6/6 全部检出）。
`filter_synth.py` 也是因此用 320。

## 做法

* 对齐结果写到 `<data_root>/processed/aligned_synth/<身份>/<image_id>.png`
* 原地更新 manifest 里的 `path`，并把原图路径留存到 `meta.raw_path`（可追溯）
* 对齐参数（均值/标准差）写进 `meta.align`，与真实图 manifest 保持同构

用法：

    python scripts/align_synth.py \
        --manifests data/manifests/syn-plusv2.jsonl data/manifests/syn-plusv2-topq.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.stdout.reconfigure(errors="replace")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402  —— 必须在 onnxruntime 之前

from aigcfr.data.align import ARCFACE_SIZE, align_by_kps, align_stats  # noqa: E402
from aigcfr.eval.embed import load_app, pick_face  # noqa: E402
from aigcfr.utils.config import find_local_config, load_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False, description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifests", nargs="+",
                    default=["data/manifests/syn-plusv2.jsonl",
                             "data/manifests/syn-plusv2-topq.jsonl"],
                    help="要原地更新的 manifest（可多个，共用同一份对齐结果）")
    ap.add_argument("--out", default=None, help="对齐输出目录（默认 <data_root>/processed/aligned_synth）")
    ap.add_argument("--det-size", type=int, default=320,
                    help="检测尺寸。**必须 320**：640 对生成的 512x512 图经常检不出脸")
    ap.add_argument("--face-select", default="center", choices=["center", "score", "area"],
                    help="多脸时怎么选。center 实测最稳（area 会选到背景误检）")
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 张（自测用）")
    ap.add_argument("--no-update-manifest", action="store_true",
                    help="只对齐、不改 manifest（自测用；配合 --limit 以免写坏正式清单）")
    args = ap.parse_args()

    cfg = load_config(None, profile="auto", local_path=str(find_local_config()))
    paths = cfg["paths"]
    data_root = Path(paths["data_root"]).resolve()
    out_dir = (Path(args.out).resolve() if args.out
               else data_root / "processed" / "aligned_synth")
    out_dir.mkdir(parents=True, exist_ok=True)

    man_paths = [REPO_ROOT / m for m in args.manifests]
    for p in man_paths:
        if not p.exists():
            print(f"!! 找不到 manifest: {p}")
            return 2

    print("=" * 74)
    print("合成图对齐（512x512 原图 -> 112x112 ArcFace 裁剪）")
    print(f"  输入 manifest : {', '.join(p.name for p in man_paths)}")
    print(f"  输出目录      : {out_dir}")
    print(f"  det_size={args.det_size}  face_select={args.face_select}  size={ARCFACE_SIZE}")
    print("=" * 74)

    app = load_app(Path(paths["models_root"]) / "insightface",
                   providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                   det_size=args.det_size, ctx_id=0)

    # ---------------- 1) 收集所有唯一的 image_id -> 原图路径 ----------------
    def read(p: Path) -> list[dict]:
        return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

    all_recs: dict[str, dict] = {}
    for p in man_paths:
        for r in read(p):
            if r.get("source") == "synth":
                all_recs.setdefault(r["image_id"], r)
    if args.limit:
        all_recs = dict(list(all_recs.items())[: args.limit])
    print(f"\n[1/3] 待对齐 {len(all_recs)} 张（跨 {len(man_paths)} 个 manifest 去重）")

    # ---------------- 2) 逐张检测 + 对齐 ----------------
    print("\n[2/3] 检测并对齐 ...")
    mapping: dict[str, dict] = {}          # image_id -> {path, align, det_score, n_faces, ok}
    t0 = time.time()
    n_ok = n_noface = 0
    for i, (iid, rec) in enumerate(all_recs.items(), 1):
        raw = rec.get("meta", {}).get("raw_path") or rec["path"]
        src = Path(raw)
        src = src if src.is_absolute() else (data_root / src if (data_root / src).exists()
                                             else REPO_ROOT / src)
        img = cv2.imread(str(src))
        entry: dict = {"ok": False}
        if img is None:
            entry["reason"] = "read_failed"
        else:
            faces = app.get(img)
            entry["n_faces"] = len(faces)
            face = pick_face(faces, img.shape, args.face_select)
            if face is None:
                entry["reason"] = "no_face"
            else:
                aligned = align_by_kps(img, face.kps, ARCFACE_SIZE)
                rel_dir = rec["identity_id"]
                (out_dir / rel_dir).mkdir(parents=True, exist_ok=True)
                dst = out_dir / rel_dir / f"{iid}.png"
                cv2.imwrite(str(dst), aligned)
                # 正常情况下裁剪图在 data_root 内 -> 写相对路径（消费端的统一约定）；
                # 若 --out 指到 data_root 之外（自测），退回绝对路径。
                # ⚠️ 这里必须 try/except：`relative_to` 在跨盘符时直接抛 ValueError。
                try:
                    rel = dst.relative_to(data_root).as_posix()
                except ValueError:
                    rel = dst.as_posix()
                entry.update({"ok": True, "path": rel,
                              "det_score": float(face.det_score),
                              "align": align_stats(aligned)})
                n_ok += 1
        if not entry["ok"]:
            n_noface += 1
        mapping[iid] = entry
        if i % 100 == 0 or i == len(all_recs):
            el = time.time() - t0
            print(f"\r  {i}/{len(all_recs)}  {i/el:.1f} 张/秒  "
                  f"成功 {n_ok} 失败 {n_noface}  剩余约 {el/i*(len(all_recs)-i)/60:.1f} 分钟",
                  end="", flush=True)
    print()
    print(f"    成功 {n_ok} / 失败 {n_noface}（失败原因多为检不出脸）")

    # ---------------- 3) 原地更新 manifest ----------------
    if args.no_update_manifest:
        print("\n[3/3] 跳过 manifest 更新（--no-update-manifest）")
    else:
        print("\n[3/3] 更新 manifest ...")
        for p in man_paths:
            recs = read(p)
            n_upd = 0
            for r in recs:
                if r.get("source") != "synth":
                    continue
                m = mapping.get(r["image_id"])
                if not m or not m["ok"]:
                    continue
                # 原图路径留档，之后还能回到未对齐的版本
                r.setdefault("meta", {})
                if "raw_path" not in r["meta"]:
                    r["meta"]["raw_path"] = r["path"]
                r["path"] = m["path"]
                r["meta"]["det_score"] = round(m["det_score"], 4)
                r["meta"]["align"] = m["align"]
                r["meta"]["aligned_from"] = "512x512 generation"
                n_upd += 1
            p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n",
                         encoding="utf-8")
            print(f"    {p.name}: 更新 {n_upd} 条")

    # ---------------- 自检：对齐前后 id_sim 应基本一致 ----------------
    print("\n[自检] 抽查对齐后的人脸是否仍能被识别、且与原图同分 ...")
    from aigcfr.eval.faces import identity_embedding
    from aigcfr.eval.verify import l2_normalize
    sample = [r for r in read(man_paths[0]) if r.get("source") == "synth"
              and r["status"] == "accepted"][:8]
    diffs = []
    for r in sample:
        iid = r["image_id"]
        # 原图路径：优先取 manifest 里留档的 raw_path；--no-update-manifest 模式下没有，
        # 就从 all_recs（更新前读入的那份）里取，保证自检在两种模式下都能跑。
        raw = r.get("meta", {}).get("raw_path") or all_recs.get(iid, {}).get("path")
        if not raw:
            continue
        raw_p = Path(raw)
        raw_p = raw_p if raw_p.is_absolute() else data_root / raw_p
        v_new = identity_embedding(app, data_root / r["path"])
        v_old = identity_embedding(app, raw_p)
        if v_new is None or v_old is None:
            continue
        sim = float(l2_normalize(v_new) @ l2_normalize(v_old))
        diffs.append(sim)
    if diffs:
        print(f"    对齐图与原始生成图的特征一致性: 均值 {np.mean(diffs):.4f} "
              f"最低 {min(diffs):.4f}（越接近 1 说明对齐没有损失信息）")
    print(f"\n  对齐结果: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
