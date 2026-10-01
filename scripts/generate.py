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

from aigcfr.eval.faces import identity_embedding  # noqa: E402
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
    # ⚠️ 0.6 -> 0.8 是**实测扫描**的结果（2026-10-01）：
    #    scale 0.0~0.4 时身份相似度基本是 0（IP-Adapter 等于没起作用），
    #    0.6 起才明显有效，0.8 最好（id_sim 0.04 -> 0.39，差 10 倍）。
    #    而且强度不够时生成的图**连人脸都检测不到**（条件是泛化人像而非本人）。
    ap.add_argument("--scale", type=float, default=0.8, help="IP-Adapter 强度（实测 0.8 最佳，勿低于 0.6）")
    ap.add_argument("--base", default="sd15", choices=["sd15", "realvis"],
                    help="底模：sd15=原始 SD1.5；realvis=Realistic Vision V6（人脸向微调，强烈建议）")
    ap.add_argument("--vae", default="auto", choices=["auto", "none", "mse"],
                    help="auto=mse(realvis)/none(sd15)；mse=vae-ft-mse-840000（人脸微调底模配套）")
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
        # 从该身份的多张真实图里挑一张检测质量最好的当参考。
        # ⚠️ 训练图是**已对齐的 112x112**，必须直通识别模型（不能再检测）——
        #    详见 aigcfr.eval.faces.identity_embedding 的说明。
        best: tuple[np.ndarray, str] | None = None
        for rec in sorted(by_identity[ident], key=lambda r: -r["meta"].get("det_score", 0)):
            vec = identity_embedding(app, data_root / rec["path"])
            if vec is not None:
                best = (vec, rec["image_id"])
                break
        if best is not None:
            id_embeds[ident] = best[0]
            ref_ids[ident] = best[1]
        else:
            skipped.append(ident)
    print(f"    拿到嵌入 {len(id_embeds)} 个身份，跳过 {len(skipped)} 个")
    if skipped:
        print(f"    跳过的身份: {skipped[:8]}{' ...' if len(skipped) > 8 else ''}")
    if not id_embeds:
        print("!! 一个嵌入都没拿到")
        return 1

    # 自检：同一身份的多张真实图之间应该互相相似（> 0.4）。
    # 如果这里很低，说明嵌入提取本身有问题，后面生成再调参也没用。
    probe_id = next(iter(id_embeds))
    probe_sims = []
    for rec in by_identity[probe_id][1:4]:
        vec = identity_embedding(app, data_root / rec["path"])
        if vec is not None:
            probe_sims.append(float(id_embeds[probe_id] @ vec))
    if probe_sims:
        print(f"    [自检] {probe_id} 的参考图与该身份另外 {len(probe_sims)} 张真实图的"
              f"相似度: {[round(s, 3) for s in probe_sims]}")
        if max(probe_sims) < 0.3:
            print("    [!] 同身份真实图之间都不相似（<0.3）—— 嵌入提取有问题，先别急着生成")

    # ---------------- 加载 SD1.5 + IP-Adapter ----------------
    print("\n[3/4] 加载底模 + IP-Adapter-FaceID ...")
    from diffusers import StableDiffusionPipeline

    if args.base == "realvis":
        # ⚠️ 人脸向微调的底模是**单文件** .safetensors（不是 diffusers 分文件夹格式）。
        #    SD 1.5 基础模型画人脸很差，换它通常能把身份保持从"不像"拉到"像"。
        #    config 用本地那份 SD1.5 的 diffusers 配置（官方 config 仓库已下架）。
        ckpt = sd_root / "realvis" / "Realistic_Vision_V6.0_NV_B1_fp16.safetensors"
        if not ckpt.exists():
            print(f"!! 找不到底模: {ckpt}")
            print("   先跑：python scripts/download_data.py --config configs/data/sd15_realvis.yaml")
            return 2
        print(f"    底模: {ckpt.name}（人脸向微调）")
        pipe = StableDiffusionPipeline.from_single_file(
            str(ckpt), config=str(sd15_dir),
            torch_dtype=torch.float16,
            safety_checker=None, requires_safety_checker=False,
        )
    else:
        print(f"    底模: 原始 SD1.5（{sd15_dir}）")
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

    # ---------- VAE：换配套的，并且强制 fp32 ----------
    # ⚠️ 两个真实踩过的坑，合起来会让画面出现"脸上的色块、整体异常偏亮、不像人类"：
    #    ① 人脸/写实向的 SD1.5 微调（如 Realistic Vision）是用 **MSE VAE**
    #       （vae-ft-mse-840000）训练的，配 SD1.5 原版 VAE 解码会**色调错乱**；
    #    ② **VAE 必须在 fp32 下解码**：SD 管线**不读** vae.config.force_upcast
    #       （那是 SDXL 管线的逻辑），所以 pipe.vae 会以 fp16 解码 -> 数值溢出。
    #       实测把 VAE 换成 fp32 后，portrait 提示词的身份相似度从 0.21 提到 **0.43**。
    vae_choice = args.vae
    if vae_choice == "auto":
        vae_choice = "mse" if args.base == "realvis" else "none"
    if vae_choice == "mse":
        vae_dir = sd_root / "vae-mse"
        if not (vae_dir / "diffusion_pytorch_model.safetensors").exists():
            print(f"!! 找不到 MSE VAE: {vae_dir}")
            print("   先跑：python scripts/download_data.py --config configs/data/sd15_vae.yaml")
            return 2
        from diffusers import AutoencoderKL
        pipe.vae = AutoencoderKL.from_pretrained(str(vae_dir), torch_dtype=torch.float32)
        print("    VAE: sd-vae-ft-mse（配套）")
    pipe.vae.to(dtype=torch.float32)      # ← 必须 fp32，见上面的说明
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    print(f"    底模={args.base}  scale={args.scale}  VAE={vae_choice}/fp32  device={pipe.device}")

    def decode_to_pil(latents):
        """用 **fp32 VAE** 手动解码 latents。

        管线只算到 latents（output_type="latent"），跳过它自带的 fp16 解码。
        """
        latents = latents.to(dtype=pipe.vae.dtype) / pipe.vae.config.scaling_factor
        image = pipe.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        arr = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
        arr = (arr * 255).round().astype(np.uint8)
        from PIL import Image as _Image
        return _Image.fromarray(arr[0])

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
            image = decode_to_pil(pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE,
                ip_adapter_image_embeds=[embeds],
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                generator=generator,
                height=512, width=512,
                output_type="latent",          # 关键：只算到 latents，自己用 fp32 VAE 解码
            ).images)

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
