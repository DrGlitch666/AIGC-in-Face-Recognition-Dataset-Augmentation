#!/usr/bin/env python
"""身份保持的人脸生成（任务卡 #10）

用 SD 1.5 + IP-Adapter-FaceID 为训练集里的每个身份生成**新的**人脸图，
作为数据扩充的来源（后面由 #11 筛选，再由 #6 训练 E4）。

## 用法

    # 冒烟：2 个身份 × 2 张
    python scripts/generate.py --identities 2 --per-identity 2 --steps 20 --out results/runs/_smoke_gen

    # 正式：全部 96 个身份 × 6 张
    python scripts/generate.py --per-identity 6

## ⚠️ 三个踩过坑后固化下来的决定（不要随手改）

1. **用 base FaceID，不用 plusv2**
   `IPAdapterFaceIDPlusImageProjection.forward` 依赖 `self.clip_embeds`，
   而整个 diffusers 包里没有任何地方给它赋值 → 走标准路径必崩。
   base FaceID 只需要一个 512 维嵌入，路径完整。

2. **ID 嵌入用 antelopev2，不用 buffalo_l**
   IP-Adapter-FaceID 是在 **antelopev2 的嵌入空间**上训练的。
   用 buffalo_l 提特征属于域不匹配，身份保持会明显变差。
   （评测仍用 buffalo_l —— 与 E0 一致；两者角色不同，互不冲突。）

3. **嵌入必须拼成 `(2, 1, 512)` 传入**
   `prepare_ip_adapter_image_embeds` 对传入的 embeds 做 `.chunk(2)`，
   第一半当**负例**、第二半当**正例**（CFG）。所以 batch=1 时形状是 (2,1,512)。

## 产出

    <paths.synth_root>/<exp-id>/<身份>/<seed>.png    生成的图
    data/manifests/<exp-id>.jsonl                    契约 A 清单（source=synth）
    results/runs/<exp-id>/gen_report.json            本次生成统计
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from aigcfr.utils.config import find_local_config, load_config  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:  # noqa: BLE001
        pass

# 提示词：保持"证件照/自然光"风格，不加夸张风格词 —— 合成数据要和真实数据同分布
PROMPTS = [
    "a photo of a person's face, frontal view, natural lighting, sharp focus",
    "a close-up portrait photo of a person, looking at the camera, neutral expression",
    "a photo of a person's face, slight side angle, soft studio lighting",
    "a high quality portrait photograph of a person, plain background",
]
NEGATIVE = "blurry, low quality, distorted, deformed, cartoon, painting, watermark, text"


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False, description="SD1.5 + IP-Adapter-FaceID 生成")
    ap.add_argument("--exp-id", default="syn-faceid")
    ap.add_argument("--per-identity", type=int, default=6, help="每个身份生成几张")
    ap.add_argument("--identities", type=int, default=None, help="只用前 N 个身份（冒烟用）")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--scale", type=float, default=0.6, help="IP-Adapter 强度")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="输出目录（默认 <synth_root>/<exp-id>）")
    ap.add_argument("--manifest", default=None, help="manifest 路径（默认 data/manifests/<exp-id>.jsonl）")
    args = ap.parse_args()

    # ---------------- 配置 ----------------
    local_file = find_local_config()
    if local_file is None:
        print("!! 找不到 configs/local.<machine>.yaml")
        return 2
    cfg = load_config(None, profile="auto", local_path=str(local_file))
    paths = cfg.get("paths", {})
    models_root = Path(paths.get("models_root", Path(paths["data_root"]).parent / "models"))
    synth_root = Path(paths.get("synth_root", Path(paths["data_root"]).parent / "synth"))
    data_root = Path(paths["data_root"])

    sd_root = models_root / "sd15-faceid"
    sd15_dir = sd_root / "sd15"
    faceid_dir = sd_root / "ip-adapter-faceid"
    faceid_bin = "ip-adapter-faceid_sd15.bin"

    if not (sd15_dir / "model_index.json").exists():
        print(f"!! 找不到 SD1.5: {sd15_dir}")
        print("   先跑：python scripts/download_data.py --config configs/data/sd15_faceid.yaml")
        return 2
    if not (faceid_dir / faceid_bin).exists():
        print(f"!! 找不到 IP-Adapter-FaceID: {faceid_dir / faceid_bin}")
        return 2

    # ⚠️ 默认放在 data_root/processed/synth/ 下 —— 和真实对齐图（processed/aligned/）
    #    同一棵树，这样 manifest 里的 path 才能统一相对 data_root 解析（契约 A）。
    #    写到 synth_root 会导致 path 无法用同一条规则解析（真实踩过）。
    out_dir = (Path(args.out).resolve() if args.out
               else data_root / "processed" / "synth" / args.exp_id)
    manifest_path = (Path(args.manifest).resolve() if args.manifest
                     else REPO_ROOT / "data" / "manifests" / f"{args.exp_id}.jsonl")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print(f"实验        : {args.exp_id}")
    print(f"SD1.5       : {sd15_dir}")
    print(f"IP-Adapter  : {faceid_dir / faceid_bin}")
    print(f"输出        : {out_dir}")
    print(f"manifest    : {manifest_path}")
    print(f"每身份 {args.per_identity} 张 | steps={args.steps} | scale={args.scale} | guidance={args.guidance}")
    print("=" * 74)

    # ---------------- 读训练集（决定给哪些身份生成）----------------
    train_manifest = REPO_ROOT / (cfg.get("data", {}).get("manifest", "data/manifests/toy_train.jsonl"))
    if not train_manifest.exists():
        print(f"!! 找不到训练 manifest: {train_manifest}（先跑 scripts/build_dataset.py）")
        return 2
    by_identity: dict[str, list[dict]] = {}
    with open(train_manifest, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("split") == "train" and rec.get("status") == "accepted":
                by_identity.setdefault(rec["identity_id"], []).append(rec)
    identities = sorted(by_identity)
    if args.identities:
        identities = identities[: args.identities]
    print(f"\n[1/4] 待生成身份 {len(identities)} 个"
          f"（每个选 1 张参考图）→ 共 {len(identities) * args.per_identity} 张")

    # ---------------- 提 ID 嵌入（antelopev2！）----------------
    print("\n[2/4] 加载 antelopev2 提取 ID 嵌入 ...")
    import cv2  # noqa: PLC0415

    # ⚠️ 先自己检查模型是否就位 —— **不要让 insightface 自动下载**：
    #    它的下载器不支持断点续传也没有重试，实测拉 antelopev2.zip（360MB）
    #    时中断一次就整体抛 ChunkedEncodingError 崩掉。请用本项目的下载器。
    antelope_dir = models_root / "insightface" / "models" / "antelopev2"
    if not antelope_dir.is_dir():
        print(f"!! 找不到 antelopev2: {antelope_dir}")
        print("   请不要让 insightface 自动下载（它的下载器不支持断点续传/重试）。")
        print("   用本项目的下载器（支持续传 + 重试）：")
        print("     python scripts/download_data.py --config configs/data/antelopev2.yaml")
        return 2

    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="antelopev2", root=str(models_root / "insightface"),
                       providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                       allowed_modules=["detection", "recognition"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    rec_prov = list(app.models["recognition"].session.get_providers())
    print(f"    antelopev2 recognition provider = {rec_prov}")
    if rec_prov and rec_prov[0] != "CUDAExecutionProvider":
        print("    [!] 没走 GPU —— 检查 import torch 是否在 onnxruntime 之前")

    id_embeds: dict[str, np.ndarray] = {}
    ref_ids: dict[str, str] = {}          # 身份 -> 实际用作参考的真实图 image_id
    skipped: list[str] = []
    for ident in identities:
        # 从该身份的多张真实图里挑一张检测质量最好的当参考
        best: tuple[np.ndarray, str] | None = None
        for rec in sorted(by_identity[ident], key=lambda r: -r["meta"].get("det_score", 0)):
            img = cv2.imread(str(data_root / rec["path"]))
            if img is None:
                continue
            faces = app.get(img)
            if not faces:
                continue
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            best = (face.normed_embedding.astype(np.float32), rec["image_id"])
            break
        if best is not None:
            id_embeds[ident] = best[0]
            ref_ids[ident] = best[1]
        else:
            skipped.append(ident)
    print(f"    拿到嵌入 {len(id_embeds)} 个身份，跳过 {len(skipped)} 个")
    if not id_embeds:
        print("!! 一个嵌入都没拿到")
        return 1

    # ---------------- 加载 SD1.5 + IP-Adapter ----------------
    print("\n[3/4] 加载 SD1.5 + IP-Adapter-FaceID ...")
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        str(sd15_dir),
        torch_dtype=torch.float16,
        variant="fp16",
        safety_checker=None,               # 人脸研究不需要，且它会拦图
        requires_safety_checker=False,
    )
    pipe.load_ip_adapter(str(faceid_dir), subfolder=None, weight_name=faceid_bin,
                         image_encoder_folder=None)   # FaceID 不需要 CLIP 编码器
    pipe.set_ip_adapter_scale(args.scale)
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    print(f"    scale={args.scale}  dtype={pipe.dtype}  device={pipe.device}")

    # ---------------- 生成 ----------------
    print("\n[4/4] 开始生成 ...")
    records: list[dict] = []
    t0 = time.time()
    total = len(id_embeds) * args.per_identity
    done = 0
    for ident, emb in id_embeds.items():
        base = torch.from_numpy(emb).to(dtype=pipe.dtype, device="cuda").reshape(1, 1, -1)
        # ⚠️ CFG 的两半：**前一半是负例、后一半是正例**（prepare_ip_adapter_image_embeds 做 chunk(2)）。
        #    负例必须是**全零** —— diffusers 的 encode_image 就是
        #        uncond_image_embeds = torch.zeros_like(image_embeds)
        #    如果两半都给同一个人脸嵌入，CFG 的 (cond - uncond) 会把面部条件**抵消掉**，
        #    症状是"生成的人不像本人、甚至性别都错"（真实踩过，别改回去）。
        embeds = torch.cat([torch.zeros_like(base), base], dim=0)

        for k in range(args.per_identity):
            # ⚠️ 必须用**确定性**哈希：Python 内置 hash() 对字符串每个进程都会变
            #    （PYTHONHASHSEED 随机化），那样种子无法复现、实验结果不可复现。
            seed = args.seed + zlib.crc32(f"{ident}_{k}".encode()) % 100000
            generator = torch.Generator(device="cuda").manual_seed(seed)
            prompt = PROMPTS[k % len(PROMPTS)]
            image = pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE,
                ip_adapter_image_embeds=[embeds],
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                generator=generator,
                height=512, width=512,
            ).images[0]

            name = f"{ident}_syn{k:02d}.png"
            target = out_dir / ident / name
            target.parent.mkdir(parents=True, exist_ok=True)
            image.save(target)
            # path 统一相对 data_root；--out 指到 data_root 之外（自测）时退回绝对路径
            try:
                rel = target.relative_to(data_root).as_posix()
            except ValueError:
                rel = target.as_posix()
            records.append({
                "image_id": f"syn_{ident}_{k:02d}",
                "path": rel,
                "identity_id": ident,
                "source": "synth",
                "generator": "ipadapter-faceid-sd15",
                "parent_image_id": ref_ids[ident],
                "split": "train",
                "status": "pending",              # 由 #11 筛选后改成 accepted/rejected
                "meta": {
                    "seed": seed, "prompt": prompt, "negative_prompt": NEGATIVE,
                    "steps": args.steps, "guidance_scale": args.guidance,
                    "ip_adapter_scale": args.scale, "id_model": "antelopev2",
                    "resolution": 512,
                },
            })
            done += 1
            if done % 10 == 0 or done == total:
                rate = done / max(time.time() - t0, 1e-6)
                eta = (total - done) / max(rate, 1e-6)
                print(f"  {done}/{total}  {rate:.2f} 张/秒  剩余约 {eta/60:.1f} 分钟")

    elapsed = time.time() - t0

    with open(manifest_path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    report = {
        "exp_id": args.exp_id,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "generator": "ipadapter-faceid-sd15 (base FaceID, antelopev2 embeddings)",
        "n_identities": len(id_embeds),
        "per_identity": args.per_identity,
        "n_generated": len(records),
        "skipped_identities": skipped,
        "steps": args.steps, "guidance_scale": args.guidance, "ip_adapter_scale": args.scale,
        "resolution": 512,
        "out_dir": str(out_dir),
        "manifest": str(manifest_path),
        "elapsed_seconds": round(elapsed, 1),
        "seconds_per_image": round(elapsed / max(len(records), 1), 2),
        "id_embedding_provider": rec_prov,
    }
    (out_dir / "gen_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                             encoding="utf-8")

    print(f"\n  生成 {len(records)} 张，用时 {elapsed:.0f}s（{report['seconds_per_image']} 秒/张）")
    print(f"  manifest: {manifest_path}（status=pending，待 #11 筛选）")
    print(f"  报告    : {out_dir / 'gen_report.json'}")
    print("\n完成 ✅  下一步：python scripts/filter_synth.py（#11，待实现）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
