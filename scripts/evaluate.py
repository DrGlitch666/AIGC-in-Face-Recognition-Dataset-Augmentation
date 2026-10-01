#!/usr/bin/env python
"""E0：用预训练模型跑通 1:1 人脸验证评测（任务卡 #5）

这是本项目的**第一把尺子**：不训练，直接用 insightface 的 ``buffalo_l`` 在 LFW 上
跑官方 6000 对协议，用来验证「数据 → 对齐 → 提特征 → 算指标 → 落盘」整条管线是通的。
期望落在 LFW 99.8% 量级（文献：ArcFace 在 MS1MV2 上报告 99.83%）。

## 用法

    python scripts/evaluate.py --exp-id e0-buffalo_l-lfw
    python scripts/evaluate.py --exp-id e0-smoke --limit-pairs 200 --out results/runs/_smoke   # 快速自测

## 输出（契约 B：docs/FRAMEWORK.md §3.3）

    results/runs/<exp-id>/metrics.json    实验指标（网站导出的唯一依据）
    results/runs/<exp-id>/pairs.csv       每一对的分数明细（便于排查与画图）

## 两个协议（原因见 src/aigcfr/data/lfw.py 的模块文档）

    official   官方 6000 对，10 折            -> 与文献对照
    filtered   剔除含"训练身份"的对（5553 对）-> 自训练模型的主结论用这个
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aigcfr.data.lfw import (  # noqa: E402
    filter_pairs_by_identity,
    list_identities,
    parse_pairs,
    protocol_summary,
    select_training_identities,
)
from aigcfr.eval.embed import (  # noqa: E402
    Embeddings,
    assert_provider,
    extract,
    load_app,
    provider_report,
    warmup,
)
from aigcfr.eval.faces import build_aligned_cache  # noqa: E402
from aigcfr.eval.torch_embed import extract_aligned, load_checkpoint  # noqa: E402
from aigcfr.eval.verify import cosine, norm_stats, pair_scores, summarize  # noqa: E402
from aigcfr.utils.config import find_local_config, load_config  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:  # noqa: BLE001
        pass


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=10, check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description="E0：预训练模型零样本评测")
    ap.add_argument("--exp-id", required=True, help="实验 ID，决定输出目录名")
    ap.add_argument("--data-name", default="lfw")
    ap.add_argument("--min-images", type=int, default=15,
                    help="选训练身份的图片数阈值（默认 15 -> LFW 上得 96 个身份）")
    ap.add_argument("--limit-pairs", type=int, default=None, help="只用前 N 对（自测用）")
    ap.add_argument("--cache", default=None, help="embedding 缓存 .npz 路径")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--out", default=None, help="输出目录（默认 results/runs/<exp-id>）")
    ap.add_argument("--cpu", action="store_true", help="强制 CPU（自测用）")
    ap.add_argument("--det-size", type=int, default=None, help="检测尺寸（默认取配置）")
    ap.add_argument("--face-select", default=None, choices=["center", "score", "area"],
                    help="一张图检出多张脸时选哪张（默认取配置的 eval.face_select）")
    ap.add_argument("--ckpt", default=None,
                    help="自研模型 checkpoint（给定时走 PyTorch 路径；不给则走 E0 的 ONNX 路径）")
    ap.add_argument("--align-cache", default=None,
                    help="自研模型路径的对齐缓存目录（默认 <data_root>/cache/aligned_<数据名>/<选脸策略>）")
    args = ap.parse_args()

    # ---------------- 配置 ----------------
    local_file = find_local_config()
    if local_file is None:
        print("!! 找不到 configs/local.<machine>.yaml（或存在多份），无法确定数据路径")
        return 2
    cfg = load_config(None, profile="auto", local_path=str(local_file))
    paths = cfg.get("paths", {})
    data_root = Path(paths["data_root"])
    models_root = Path(paths.get("models_root", data_root.parent / "models"))
    runtime = cfg.get("runtime", {})

    providers = (["CPUExecutionProvider"] if args.cpu
                 else list(runtime.get("onnx_providers", ["CUDAExecutionProvider", "CPUExecutionProvider"])))
    det_size = args.det_size or int(runtime.get("onnx_det_size", 640))
    face_select = args.face_select or (cfg.get("eval") or {}).get("face_select", "center")

    lfw_root = data_root / "raw" / args.data_name
    out_dir = (Path(args.out) if args.out else REPO_ROOT / "results" / "runs" / args.exp_id)
    out_dir = out_dir.resolve()          # 统一成绝对路径，避免 relative_to 报错
    out_dir.mkdir(parents=True, exist_ok=True)

    # ⚠️ 缓存文件名必须带上选脸策略：换策略后旧缓存不能再复用，
    #    否则会拿"用错误策略算出的 embedding"去算指标（真实踩过）。
    cache_path = None if args.no_cache else Path(
        args.cache or (data_root / "cache" / "embeddings"
                       / f"{args.data_name}_{face_select}.npz")
    )

    print("=" * 68)
    print(f"实验      : {args.exp_id}")
    print(f"模型来源  : {'自研 checkpoint ' + str(args.ckpt) if args.ckpt else 'ONNX buffalo_l（E0 零样本）'}")
    print(f"机器配置  : {local_file.name}  (profile={cfg['_meta']['profile']})")
    print(f"数据根目录: {lfw_root}")
    print(f"输出目录  : {out_dir}")
    print(f"providers : {providers}   det_size={det_size}   face_select={face_select}")
    print(f"缓存      : {cache_path if cache_path else '(已关闭)'}")
    print("=" * 68)

    # ---------------- 协议 ----------------
    print("\n[1/5] 解析 LFW 协议 ...")
    pairs = parse_pairs(lfw_root / "pairs.txt")
    img_dir = lfw_root / args.data_name if (lfw_root / args.data_name).is_dir() else lfw_root / "lfw"
    identities = list_identities(img_dir)
    train_ids = select_training_identities(identities, min_images=args.min_images)

    if args.limit_pairs:
        # 自测用：同人/异人各取一半，否则指标会退化（全同人或全异人）
        half = max(args.limit_pairs // 2, 1)
        pairs = ([p for p in pairs if p.is_same][:half]
                 + [p for p in pairs if not p.is_same][:half])
        print(f"  [--limit-pairs {args.limit_pairs}] 同人/异人各取 {half} 对用于自测")

    kept = filter_pairs_by_identity(pairs, set(train_ids))
    print(f"  官方 6000 对 -> 解析 {len(pairs)} 对")
    print(f"  身份 {len(identities)} 个，图片 {sum(len(v) for v in identities.values())} 张")
    print(f"  训练身份（>={args.min_images} 张图）: {len(train_ids)} 个，"
          f"{sum(len(identities[i]) for i in train_ids)} 张图")
    print(f"  剔除泄漏后剩 {len(kept)} 对（损失 {len(pairs) - len(kept)} 对）")

    # ---------------- 需要处理的图片 ----------------
    print("\n[2/5] 汇总待处理图片 ...")
    needed = sorted({p.relpath_a for p in pairs} | {p.relpath_b for p in pairs})
    print(f"  去重后 {len(needed)} 张")

    # ---------------- 提特征 ----------------
    print("\n[3/5] 加载模型并提取特征 ...")
    t_load = time.time()
    # 检测模型两条路径都要用：ONNX 路径自带识别，自研模型路径只用它做检测+对齐
    app = load_app(models_root / "insightface", providers=providers, det_size=det_size,
                   ctx_id=0 if not args.cpu else -1)
    assert_provider(app, expect_cuda=not args.cpu)
    print(f"  检测模型加载完成（{time.time() - t_load:.1f}s）")
    for name, provs in provider_report(app).items():
        flag = "  " if provs and provs[0] == "CUDAExecutionProvider" else "⚠️"
        print(f"    {flag} {name:12} {provs}")

    warm_seconds = 0.0
    if not args.cpu:
        warm_seconds = warmup(app)
        print(f"  CUDA 预热完成（{warm_seconds:.1f}s，一次性开销，不计入吞吐）")

    model_meta: dict[str, object] = {}
    align_stat: dict[str, object] = {}
    if args.ckpt:
        # ---------- 自研模型路径 ----------
        import torch  # noqa: PLC0415

        device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        model, classes, ckpt_meta = load_checkpoint(args.ckpt, device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  自研模型: {ckpt_meta.get('arch')}  epoch={ckpt_meta.get('epoch')}  "
              f"类别数={len(classes)}  参数 {n_params/1e6:.1f}M  device={device}")

        aligned_dir = (Path(args.align_cache).resolve() if args.align_cache
                       else data_root / "cache" / f"aligned_{args.data_name}" / face_select)
        print(f"  对齐缓存: {aligned_dir}")
        mapping, align_fail, align_stat = build_aligned_cache(
            app, img_dir, needed, aligned_dir, face_select=face_select)
        vectors = extract_aligned(model, mapping, device=device,
                                  batch_size=int(runtime.get("batch_size", 128)))
        emb = Embeddings(vectors=vectors, failures=align_fail, provider=["PyTorch"],
                         seconds=float(align_stat.get("seconds", 0.0)))
        model_meta = {
            "arch": ckpt_meta.get("arch", "iresnet18"),
            "loss": "arcface (self-trained)",
            "epochs": ckpt_meta.get("epoch"),
            "seed": None,
            "params_total": n_params,
            "ckpt": str(args.ckpt),
            "embedding_source": "torch",
            "face_select": face_select,
            "det_size": det_size,
        }
    else:
        # ---------- E0 ONNX 路径 ----------
        emb = extract(app, img_dir, needed, cache_path=cache_path, face_select=face_select)
        model_meta = {
            "arch": "buffalo_l (w600k_r50, ResNet50 + ArcFace)",
            "loss": "arcface (pretrained)",
            "epochs": 0,
            "seed": None,
            "onnx_providers": provider_report(app).get("recognition", []),
            "det_size": det_size,
            "face_select": face_select,
            "embedding_source": "onnx",
        }
    print(f"  成功 {emb.num_ok} / 失败 {emb.num_failed}")
    if emb.failures:
        print(f"  失败原因: {emb.failure_counts()}")
    ns = norm_stats(emb.vectors)
    print(f"  L2 范数: min={ns.get('min'):.4f} max={ns.get('max'):.4f} "
          f"归一化正确={ns.get('is_normalized')}")

    # ---------------- 指标 ----------------
    print("\n[4/5] 计算指标 ...")
    sc_official = pair_scores(pairs, emb.vectors)
    sc_filtered = pair_scores(kept, emb.vectors)
    m_official = summarize(sc_official)
    m_filtered = summarize(sc_filtered)

    print(f"  [official] {m_official['n_pairs_used']}/{len(pairs)} 对可用 | "
          f"accuracy={m_official['value']:.4f} | "
          f"TAR@FAR=1e-3 {m_official['tar_at_far']['1e-03']['tar']:.4f}")
    print(f"  [filtered] {m_filtered['n_pairs_used']}/{len(kept)} 对可用 | "
          f"accuracy={m_filtered['value']:.4f} | "
          f"TAR@FAR=1e-3 {m_filtered['tar_at_far']['1e-03']['tar']:.4f}")

    # ---------------- 落盘 ----------------
    print("\n[5/5] 写结果 ...")
    import torch  # noqa: PLC0415

    # ⚠️ 必须与 train.py 写的 metrics.json **合并**：
    #    同一个 exp_id 下训练信息和评测指标要在一个文件里（契约 B），
    #    直接覆盖会把 train_set / optim / training 全丢掉（真实踩过）。
    metrics_file = out_dir / "metrics.json"
    prev: dict = {}
    if metrics_file.exists():
        try:
            prev = json.loads(metrics_file.read_text(encoding="utf-8"))
            print(f"  合并已有 metrics.json（保留训练信息："
                  f"{'有' if prev.get('train_set') else '无'} train_set）")
        except Exception as exc:  # noqa: BLE001
            print(f"  [!] 已有 metrics.json 读不出来，将新建：{exc}")

    now = datetime.now().astimezone().isoformat(timespec="seconds")
    metrics = {
        **prev,
        "exp_id": args.exp_id,
        "created_at": prev.get("created_at", now),
        "evaluated_at": now,
        "config_path": prev.get("config_path"),
        "config_hash": cfg["_meta"]["config_hash"],
        "git_commit": git_commit(),
        "train_set": prev.get("train_set"),      # E0 为 None；自研模型保留训练信息
        "model": model_meta,
        "benchmarks": {
            **prev.get("benchmarks", {}),
            "lfw": {**m_official, "protocol": "official-6000pairs-10fold",
                    "description": "官方全量 6000 对，用于与文献对照"},
            "lfw_filtered": {**m_filtered,
                             "protocol": f"official-excluding-train-identities(min_images={args.min_images})",
                             "description": "剔除含训练身份的对，自训练模型的主结论用这个",
                             "train_identities": len(train_ids),
                             "train_images": sum(len(identities[i]) for i in train_ids)},
        },
        "protocol_detail": {
            "official": protocol_summary(pairs),
            "filtered": protocol_summary(pairs, kept),
        },
        "embedding": {
            "num_used": emb.num_ok,
            "num_failed": emb.num_failed,
            "failure_reasons": emb.failure_counts(),
            "norm_stats": ns,
            "extract_seconds": round(emb.seconds, 2),
            "images_per_second": round(emb.num_ok / emb.seconds, 2) if emb.seconds > 0 else None,
            "warmup_seconds": round(warm_seconds, 2),
            "detection": emb.detection_stats() if not args.ckpt else align_stat,
            "aligned_cache": str(aligned_dir) if args.ckpt else None,
        },
        "hardware": {
            "hardware_profile": cfg["_meta"]["profile"],
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "vram_gb": cfg["_meta"].get("vram_gb"),
            "torch_version": torch.__version__,
            "python": platform.python_version(),
        },
        "artifacts": {
            **prev.get("artifacts", {}),
            "metrics": str(metrics_file.relative_to(REPO_ROOT)),
            "pairs_csv": str((out_dir / "pairs.csv").relative_to(REPO_ROOT)),
        },
        "notes": (
            "E0 零样本基线：不训练，直接用预训练 buffalo_l 验证整条评测管线。"
            "official 用于与文献对照（阈值在测试集上选，是 LFW 惯例）；"
            "filtered 剔除训练身份后更干净，后续自训练模型主结论用它。"
        ) if not args.ckpt else (
            f"自研模型评测（{args.ckpt}）。embedding 由 PyTorch 模型产生，"
            f"检测与对齐复用 buffalo_l + norm_crop2（与训练完全一致）。"
            f"filtered 协议已剔除 {len(train_ids)} 个训练身份。"
        ),
    }

    metrics_file.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    with open(out_dir / "pairs.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["set", "fold", "kind", "id_a", "file_a", "id_b", "file_b",
                         "score", "label"])
        kept_set = set(kept)          # Pair 是 frozen dataclass，可直接进 set
        for p in pairs:
            a, b = emb.vectors.get(p.relpath_a), emb.vectors.get(p.relpath_b)
            score = "" if a is None or b is None else f"{cosine(a, b):.6f}"
            tags = ["official", "filtered"] if p in kept_set else ["official"]
            for tag in tags:
                writer.writerow([tag, p.fold, p.kind, p.id_a, p.filename_a,
                                 p.id_b, p.filename_b, score, p.label])

    print(f"  {out_dir / 'metrics.json'}")
    print(f"  {out_dir / 'pairs.csv'}")
    print("\n完成 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
