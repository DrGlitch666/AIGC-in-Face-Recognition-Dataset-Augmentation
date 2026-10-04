#!/usr/bin/env python
"""训练入口：iresnet18 + ArcFace（任务卡 #6）

## 用法

    python scripts/train.py --exp-id e1-real-only-seed0
    python scripts/train.py --exp-id e1-smoke --epochs 2 --max-steps 30 --out results/runs/_smoke   # 冒烟

## 产出（契约 B：docs/FRAMEWORK.md §3.3）

    results/runs/<exp-id>/metrics.json     训练信息（benchmarks 先留空，评测后回填）
    results/runs/<exp-id>/train_log.csv    每轮 loss / 准确率 / lr / 耗时（画曲线用）
    <paths.ckpt_root>/<exp-id>/last.pth    最后一轮
    <paths.ckpt_root>/<exp-id>/best.pth    训练准确率最高的一轮

## 为什么这些超参这么定

* 优化器 SGD + momentum 0.9 + wd 5e-4、lr 0.1：ArcFace 论文的标准配法
* **余弦退火**：只有 10 个 epoch，阶梯衰减没机会生效
* ArcFace 的 `m=0.5 / s=64` 来自论文消融实验（见 L6 学习材料）
* 增广**只用水平翻转** —— 与后续"合成图实验"保持可比：如果增广拉满，
  两臂的差异会被增广掩盖，看不到合成图的作用
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from aigcfr.train.dataset import ManifestDataset  # noqa: E402
from aigcfr.train.loss import ArcFace  # noqa: E402
from aigcfr.train.model import build_model, count_parameters  # noqa: E402
from aigcfr.utils.config import find_local_config, load_config  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:  # noqa: BLE001
        pass


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, timeout=10, check=False)
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id: int) -> None:
    """给每个 DataLoader worker **单独**播种 numpy / random。

    ## 为什么需要它

    数据增强（`dataset.py`）用的是 `np.random`，而 PyTorch 的 DataLoader
    **只负责 seed torch，不管 numpy**。`num_workers>0` 时每个 worker 由 fork/spawn
    产生，于是**继承同一份 numpy RNG 状态**，跑出完全相同的随机序列 ——
    水平翻转的决策在所有 worker 间**同相位**，增强退化成"相关噪声"。

    实测（2026-10-02，E4b 数据，前 3 轮 train_acc）：

        num_workers=4（有 bug）  0.0000 / 0.0000 / 0.0000
        num_workers=0（正常）    0.0007 / 0.0421 / 0.1320

    同一份数据、同一 seed，只差 worker 数 —— 差的不是模型，是增强。

    ## ⚠️ 必须是**模块级**函数，不能定义在 main() 里

    Windows 上 multiprocessing 用 **spawn**（不是 fork），worker 相关对象要能被
    pickle。嵌套函数是局部对象，pickle 不了，会直接崩：

        AttributeError: Can't pickle local object 'main.<locals>._seed_worker'

    `torch.initial_seed()` 在 worker 内已经带上了 worker_id，所以各 worker 拿到的
    种子互不相同。
    """
    base = torch.initial_seed() % (2 ** 32)
    np.random.seed(base)
    random.seed(base)


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False, description="训练 iresnet18 + ArcFace")
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--exp", default=None, help="实验配置，如 configs/exp/e1-real-only-seed0.yaml")
    ap.add_argument("--epochs", type=int, default=None, help="覆盖配置里的轮数（冒烟用）")
    ap.add_argument("--max-steps", type=int, default=None, help="每轮最多多少步（冒烟用）")
    ap.add_argument("--out", default=None, help="输出目录（默认 results/runs/<exp-id>）")
    ap.add_argument("--pooled", action="store_true", help="用小 fc 版本（参数量 ~1/6）")
    ap.add_argument("--num-workers", type=int, default=None, help="DataLoader 进程数（默认取配置）")
    ap.add_argument("--ckpt-dir", default=None, help="checkpoint 目录（默认 <paths.ckpt_root>/<exp-id>）")
    ap.add_argument("--seed", type=int, default=None,
                    help="覆盖配置里的随机种子。跑多种子对照时用（同一配置换 seed 即为独立重复）")
    args = ap.parse_args()

    # ---------------- 配置 ----------------
    local_file = find_local_config()
    if local_file is None:
        print("!! 找不到 configs/local.<machine>.yaml")
        return 2
    cfg = load_config(args.exp, profile="auto", local_path=str(local_file))
    data_cfg, model_cfg, train_cfg = cfg.get("data", {}), cfg.get("model", {}), cfg.get("train", {})
    runtime = cfg.get("runtime", {})
    paths = cfg.get("paths", {})

    data_root = Path(paths["data_root"])
    ckpt_root = Path(paths.get("ckpt_root", data_root.parent / "ckpt"))
    # data.manifests（列表）用于 E4：「真实 + 筛选后的合成」拼在一起训练；
    # 只写 data.manifest（单个字符串）就是 E1 那种「只用真实」。
    raw_manifests = data_cfg.get("manifests") or [data_cfg.get("manifest", "data/manifests/toy_train.jsonl")]
    manifest_paths = [REPO_ROOT / m for m in raw_manifests]
    # 用哪些来源的图训练：real / synth / aug
    train_sources = tuple(train_cfg.get("sources", ["real"]))
    image_size = int(data_cfg.get("image_size", 112))
    emb_dim = int(model_cfg.get("emb_dim", 512))
    epochs = int(args.epochs or train_cfg.get("epochs", 10))
    batch_size = int(runtime.get("batch_size", 64))
    grad_accum = int(runtime.get("grad_accum", 1))
    amp_enabled = bool(runtime.get("amp", True)) and torch.cuda.is_available()
    num_workers = int(args.num_workers if args.num_workers is not None
                      else runtime.get("num_workers", 0))
    # 种子：CLI 覆盖配置。多种子对照就是"同一配置换 seed 跑几遍"，用以估计种子间方差
    # ——那是 TAR@FAR 这类指标的主导不确定性来源（LFW 本身的评测分辨率不够）。
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 0))

    out_dir = (Path(args.out) if args.out else REPO_ROOT / "results" / "runs" / args.exp_id).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.ckpt_dir).resolve() if args.ckpt_dir else (ckpt_root / args.exp_id)

    device = torch.device(runtime.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    print("=" * 72)
    print(f"实验      : {args.exp_id}")
    print(f"配置      : {local_file.name} + {args.exp or '(无实验配置)'}  hash={cfg['_meta']['config_hash']}")
    print(f"设备      : {device}   batch={batch_size} x accum={grad_accum} = 有效 {batch_size*grad_accum}")
    print(f"AMP       : {amp_enabled}   epochs={epochs}   lr={train_cfg.get('lr')}   seed={seed}")
    print("=" * 72)

    # ---------------- 数据 ----------------
    ds = ManifestDataset(manifest_paths, data_root, split="train", image_size=image_size,
                         augment=True, sources=train_sources)
    counts = ds.class_counts()
    n_imgs, n_cls = len(ds), len(ds.classes)
    by_source: dict[str, int] = {}
    for r in ds.records:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    src_txt = " + ".join(f"{k} {v}" for k, v in sorted(by_source.items()))
    print(f"\n[1/3] 数据: {n_imgs} 张图 / {n_cls} 个身份（{src_txt}）"
          f"（每身份 {min(counts.values())}~{max(counts.values())} 张，均值 {n_imgs/n_cls:.1f}）")
    if len(by_source) > 1:
        real_n, synth_n = by_source.get("real", 0), by_source.get("synth", 0)
        print(f"    合成图占比 = {synth_n}/{real_n + synth_n} = {synth_n/max(real_n+synth_n,1)*100:.1f}%")
    for p in manifest_paths:
        print(f"    manifest: {p.relative_to(REPO_ROOT)}")
    if n_cls < 2:
        print("!! 只有一个身份，无法用 ArcFace 训练")
        return 2
    # ⚠️⚠️ 每个 DataLoader worker 必须**单独** seed numpy。
    #
    #   数据增强（dataset.py）用的是 `np.random`，而 PyTorch 的 DataLoader
    #   **只负责 seed torch，不管 numpy**。num_workers>0 时每个 worker 都继承
    #   同一份 numpy RNG 状态，跑出完全相同的随机序列 —— 水平翻转的决策在所有
    #   worker 间**同相位**，增强退化成"相关噪声"。
    #   （注意：Windows 上是 spawn 不是 fork，但两边都有这个问题。）
    #
    #   实测（2026-10-02，E4b 数据，前 3 轮 train_acc）：
    #       num_workers=4（有 bug）  0.0000 / 0.0000 / 0.0000
    #       num_workers=0（正常）    0.0007 / 0.0421 / 0.1320
    #   同一份数据、同一 seed，只差 worker 数 —— 差的不是模型，是增强。
    #
    #   修法见模块级的 `_seed_worker`（**必须**放在模块级，见那里的说明）。
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                        pin_memory=(device.type == "cuda"), drop_last=False,
                        worker_init_fn=_seed_worker if num_workers > 0 else None)

    # ---------------- 模型与损失 ----------------
    model = build_model(model_cfg.get("arch", "iresnet18"), emb_dim=emb_dim,
                        input_size=image_size, dropout=float(model_cfg.get("dropout", 0.0)),
                        pooled=args.pooled).to(device)
    head = ArcFace(emb_dim, n_cls, margin=float(model_cfg.get("margin", 0.5)),
                   scale=float(model_cfg.get("scale", 64.0))).to(device)
    params = count_parameters(model)
    print(f"[2/3] 模型: {model_cfg.get('arch')}  参数 {params['total']/1e6:.1f}M"
          f"（可训练 {params['trainable']/1e6:.1f}M）"
          f"{'  [pooled]' if args.pooled else ''}")
    print(f"      损失: ArcFace(m={head.margin}, s={head.scale})  类数={n_cls}")

    optimizer = torch.optim.SGD(
        list(model.parameters()) + list(head.parameters()),
        lr=float(train_cfg.get("lr", 0.1)),
        momentum=float(train_cfg.get("momentum", 0.9)),
        weight_decay=float(train_cfg.get("weight_decay", 5e-4)),
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    # ---------------- 训练 ----------------
    print(f"[3/3] 开始训练（{len(loader)} 步/轮）...\n")
    log_rows: list[dict] = []
    best_acc = -1.0
    best_epoch: int | None = None
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        # 先记下本轮**实际使用**的 lr：scheduler.step() 在轮末执行，改的是"下一轮"的 lr。
        # （之前把打印放在 scheduler.step() 之后，日志里的 lr 标签是错位的。）
        current_lr = optimizer.param_groups[0]["lr"]
        model.train()
        head.train()
        run_loss, run_correct, run_total = 0.0, 0, 0
        t_epoch = time.time()
        optimizer.zero_grad(set_to_none=True)

        for step, (imgs, labels) in enumerate(loader, 1):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                emb = model(imgs)
                loss, logits = head(emb, labels)
                loss = loss / grad_accum
            scaler.scale(loss).backward()

            if step % grad_accum == 0 or step == len(loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            run_loss += float(loss) * grad_accum * imgs.size(0)
            run_correct += int((logits.argmax(1) == labels).sum())
            run_total += imgs.size(0)

            if args.max_steps and step >= args.max_steps:
                break

        scheduler.step()
        acc = run_correct / max(run_total, 1)
        avg_loss = run_loss / max(run_total, 1)
        dt = time.time() - t_epoch
        print(f"  epoch {epoch:>2}/{epochs}  loss={avg_loss:.4f}  train_acc={acc:.4f}  "
              f"lr={current_lr:.5f}  {dt:.1f}s")
        log_rows.append({"epoch": epoch, "loss": round(avg_loss, 6),
                         "train_acc": round(acc, 6), "lr": round(current_lr, 8),
                         "seconds": round(dt, 2)})

        state = {"model": model.state_dict(), "head": head.state_dict(),
                 "classes": ds.classes, "epoch": epoch, "config_hash": cfg["_meta"]["config_hash"],
                 "arch": model_cfg.get("arch", "iresnet18"), "emb_dim": emb_dim,
                 "image_size": image_size, "pooled": args.pooled}
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(state, ckpt_dir / "last.pth")
        # ⚠️ 只在 **acc > 0 且创新高** 时才写 best.pth。
        #
        #    之前 best_acc 初值是 -1.0，于是**第 1 轮必然被存成 best**；一旦之后 acc 恒为 0，
        #    `0 > 0` 为假，best.pth 就永远停在第 1 轮那个废模型上 —— 评测时拿到的是垃圾，
        #    却完全看不出来（真实踩过：E4b 的 best.pth 停在 epoch 1，acc=0.0000）。
        if acc > best_acc and acc > 0:
            best_acc = acc
            torch.save(state, ckpt_dir / "best.pth")
            best_epoch = epoch
        # 权重健康度：某类的范数远高于中位数 = 多数类变成"吸引子"，模型开始崩塌。
        # 这是 E4b 那次崩塌的直接证据（最大/中位数 = 3.30，正常只有 1.75~1.91），
        # 加进日志后一眼可见。
        with torch.no_grad():
            norms = head.weight.norm(dim=1)
            ratio = float(norms.max() / norms.median())
        log_rows[-1]["weight_max_median_ratio"] = round(ratio, 3)

    total_time = time.time() - t_start

    # ---------------- 落盘 ----------------
    with open(out_dir / "train_log.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["epoch", "loss", "train_acc", "lr", "seconds",
                                                "weight_max_median_ratio"])
        writer.writeheader()
        writer.writerows(log_rows)

    metrics = {
        "exp_id": args.exp_id,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "config_path": args.exp,
        "config_hash": cfg["_meta"]["config_hash"],
        "git_commit": git_commit(),
        "train_set": {
            "real_identities": n_cls,
            "real_images": by_source.get("real", 0),
            "synth_images": by_source.get("synth", 0),
            "synth_ratio": round(by_source.get("synth", 0) / max(n_imgs, 1), 4),
            "generator": data_cfg.get("synth_generator"),
            "manifests": [str(p.relative_to(REPO_ROOT)) for p in manifest_paths],
            "sources": list(train_sources),
            "total_images": n_imgs,
        },
        "model": {
            "arch": model_cfg.get("arch", "iresnet18"),
            "loss": "arcface",
            "margin": head.margin,
            "scale": head.scale,
            "emb_dim": emb_dim,
            "epochs": epochs,
            "seed": seed,
            "params_total": params["total"],
            "pooled_fc": args.pooled,
        },
        "optim": {
            "optimizer": train_cfg.get("optimizer", "sgd"),
            "lr": train_cfg.get("lr"), "momentum": train_cfg.get("momentum"),
            "weight_decay": train_cfg.get("weight_decay"), "scheduler": "cosine",
            "batch_size": batch_size, "grad_accum": grad_accum, "amp": amp_enabled,
        },
        "training": {
            "final_loss": log_rows[-1]["loss"],
            "final_train_acc": log_rows[-1]["train_acc"],
            "best_train_acc": best_acc,
            "best_epoch": best_epoch,
            "best_ckpt_written": best_epoch is not None,
            "final_weight_max_median_ratio": log_rows[-1].get("weight_max_median_ratio"),
            "elapsed_seconds": round(total_time, 1),
            "log": str((out_dir / "train_log.csv").relative_to(REPO_ROOT)),
        },
        "benchmarks": {},          # 由 scripts/evaluate.py 回填
        "hardware": {
            "hardware_profile": cfg["_meta"]["profile"],
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "vram_gb": cfg["_meta"].get("vram_gb"),
            "torch_version": torch.__version__,
            "python": platform.python_version(),
        },
        "artifacts": {
            "ckpt_last": str((ckpt_dir / "last.pth")),
            "ckpt_best": str((ckpt_dir / "best.pth")),
            "metrics": str((out_dir / "metrics.json").relative_to(REPO_ROOT)),
        },
        "notes": "",
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                                          encoding="utf-8")

    print(f"\n  用时 {total_time:.0f}s | 最终 loss {log_rows[-1]['loss']:.4f} | "
          f"训练准确率 {log_rows[-1]['train_acc']:.4f}（最好 {best_acc:.4f}）")
    print(f"  权重健康度（最大/中位数范数）: {log_rows[-1].get('weight_max_median_ratio')}"
          f"  — 远大于 2 说明多数类在变成吸引子")
    if best_epoch is None:
        # 从未写过 best.pth：整段训练 acc 恒为 0。**明确报出来**，
        # 免得评测时拿着一个不存在的文件（或更糟：上一轮遗留的旧文件）当成结果。
        print("  ⚠️  没有写出 best.pth —— 所有轮次的 train_acc 都是 0，模型没有学到东西。")
        print("      请看 train_log.csv 的 weight_max_median_ratio 列排查。")
        print(f"  checkpoint: {ckpt_dir / 'last.pth'}（仅最后一轮）")
    else:
        print(f"  checkpoint: {ckpt_dir / 'best.pth'}（第 {best_epoch} 轮，train_acc {best_acc:.4f}）")
    print(f"  metrics   : {out_dir / 'metrics.json'}")
    print("\n完成 ✅  下一步：python scripts/evaluate.py --exp-id <同一个 id> --ckpt <best.pth>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
