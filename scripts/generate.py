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

2. **ID 嵌入必须用 buffalo_l，不能用 antelopev2**
   网上文档（IP-Adapter-FaceID README）说要用 antelopev2，**照做会得到完全不像本人的图**。
   交叉验证（同一 seed、同一参考图、两个嵌入模型 × 两个度量空间）：

       条件嵌入        度量空间        id_sim
       buffalo_l      buffalo_l      +0.4002
       buffalo_l      antelopev2     +0.4464   <- 跨空间更高，排除"度量偏差"
       antelopev2     buffalo_l      -0.1352
       antelopev2     antelopev2     -0.0754

   buffalo_l 的嵌入能让身份真正传过去；antelopev2 的嵌入让身份条件**完全失效**。
   两者对真实图的嵌入都正常（同身份自检 0.755 / 0.768），
   所以是**这个适配器只认 buffalo_l 的嵌入分布**，与嵌入质量无关。

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
from PIL import Image  # noqa: E402

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
    # 下面四个默认值是 2026-10-01 配置扫描（tools/quality_sweep.py）选出来的 **E_all**：
    #   dpmpp + mean-ref + steps40 + guidance4.0 -> id_sim 均值 +0.3352（原配置 +0.2932，+14%）
    # 其中**贡献最大的是 --ref-mode mean**（单独就有 +0.3272）：
    # 同一身份不同真实照片的嵌入相似度只有 ~0.75，单张参考本身带噪声，平均能把它压下去。
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--guidance", type=float, default=4.0)
    # ⚠️ 0.6 -> 0.8 是**实测扫描**的结果（2026-10-01）：
    #    scale 0.0~0.4 时身份相似度基本是 0（IP-Adapter 等于没起作用），
    #    0.6 起才明显有效，0.8 最好（id_sim 0.04 -> 0.39，差 10 倍）。
    #    而且强度不够时生成的图**连人脸都检测不到**（条件是泛化人像而非本人）。
    ap.add_argument("--scale", type=float, default=0.8, help="IP-Adapter 强度（实测 0.8 最佳，勿低于 0.6）")
    ap.add_argument("--base", default="sd15", choices=["sd15", "realvis"],
                    help="底模：sd15=原始 SD1.5；realvis=Realistic Vision V6"
                         "（2026-10-01 在修好的管线下重测：id_sim 仅 0.0428，比基线低 7 倍，确认不兼容）")
    ap.add_argument("--ip-adapter", default="faceid", choices=["faceid", "faceid-plusv2"],
                    help="faceid=base（仅 ID 嵌入，实测 id_sim 0.335，**默认用这个**）；"
                         "faceid-plusv2=CLIP+ID 双路（代码已打通、形状已验证，但实测只有 0.082，"
                         "比 base 差 4 倍；上游 diffusers 无参考实现，故不推荐）")
    ap.add_argument("--scheduler", default="dpmpp", choices=["pndm", "dpmpp"],
                    help="采样器：dpmpp=DPM++ 2M Karras（默认，实测更好）；pndm=SD1.5 旧默认")
    ap.add_argument("--ref-mode", default="mean", choices=["single", "mean"],
                    help="身份嵌入：mean=前 N 张真实图平均（默认，实测贡献最大）；single=只用一张")
    ap.add_argument("--ref-count", type=int, default=5, help="--ref-mode mean 时平均几张")
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

    # ---------------- 提 ID 嵌入 ----------------
    print("\n[2/4] 提取身份嵌入 ...")
    import cv2  # noqa: PLC0415

    # ⚠️⚠️ 条件嵌入必须用 **buffalo_l**，**不是** antelopev2！
    #
    #    网上文档（IP-Adapter-FaceID 的 README）说要用 antelopev2，照做之后
    #    生成的人完全不像本人。交叉验证（2026-10-01，同一 seed、同一参考图）：
    #
    #        条件嵌入        度量空间        id_sim
    #        buffalo_l      buffalo_l      +0.4002
    #        buffalo_l      antelopev2     +0.4464   <- 跨空间更高，不是度量偏差
    #        antelopev2     buffalo_l      -0.1352
    #        antelopev2     antelopev2     -0.0754
    #
    #    buffalo_l 的嵌入能让身份真正传过去（跨空间度量也高）；
    #    antelopev2 的嵌入则让身份条件**完全失效**。
    #    两个模型对真实图的嵌入都正常（同身份自检 0.755 / 0.768），
    #    所以问题不在嵌入质量，而在**这个适配器只认 buffalo_l 的嵌入分布**。
    from aigcfr.eval.embed import load_app, warmup

    app = load_app(models_root / "insightface",
                   providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                   det_size=640, ctx_id=0)
    if not torch.cuda.is_available():
        app.prepare(ctx_id=-1, det_size=(640, 640))
    print(f"    buffalo_l provider = {list(app.models['recognition'].session.get_providers())}")
    warmup(app)

    id_embeds: dict[str, np.ndarray] = {}
    ref_ids: dict[str, str] = {}          # 身份 -> 参考说明（图片 id 或 "mean of N"）
    ref_paths: dict[str, Path] = {}       # 身份 -> 代表图（对齐裁剪）
    clip_paths: dict[str, Path] = {}      # 身份 -> 喂 CLIP 那一路的图（原始整图）
    skipped: list[str] = []
    for ident in identities:
        # ⚠️ 训练图是**已对齐的 112x112**，必须直通识别模型（不能再检测）——
        #    详见 aigcfr.eval.faces.identity_embedding 的说明。
        #
        # 两种取法：
        #   single —— 取 det_score 最高的一张
        #   mean   —— 取前 N 张求平均再归一化。同一身份不同真实照片的嵌入相似度只有 ~0.75，
        #             说明单张参考本身带噪声；平均能把这个噪声压下去（零额外计算开销）。
        cands = sorted(by_identity[ident], key=lambda r: -r["meta"].get("det_score", 0))
        vecs: list[np.ndarray] = []
        for rec in (cands[: args.ref_count] if args.ref_mode == "mean" else cands):
            vec = identity_embedding(app, data_root / rec["path"])
            if vec is not None:
                vecs.append(vec)
            if args.ref_mode == "single" and vecs:
                break
        if not vecs:
            skipped.append(ident)
            continue
        if args.ref_mode == "mean":
            mean = np.mean(np.stack(vecs), axis=0)
            id_embeds[ident] = mean / max(float(np.linalg.norm(mean)), 1e-12)
            ref_ids[ident] = f"mean of {len(vecs)}"
        else:
            id_embeds[ident] = vecs[0]
            ref_ids[ident] = cands[0]["image_id"]
        ref_paths[ident] = data_root / cands[0]["path"]
        # ⚠️ plusv2 的 CLIP 那一路要的是**整体外观**（发型、轮廓、光照、衣着），
        #    **不是**五官细节 —— 官方 demo 喂的是原始整图。
        #    喂 112x112 的紧致裁剪会让 CLIP 嵌入完全落在训练分布之外，
        #    实测身份相似度只有 0.0775（而 base FaceID 是 0.3352）。
        origin = cands[0].get("meta", {}).get("origin")
        clip_paths[ident] = (data_root / origin) if origin else ref_paths[ident]
    print(f"    拿到嵌入 {len(id_embeds)} 个身份（ref-mode={args.ref_mode}），跳过 {len(skipped)} 个")
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
    # ---------------- IP-Adapter：base FaceID 或 plusv2 ----------------
    plusv2 = args.ip_adapter == "faceid-plusv2"
    if plusv2:
        faceid_bin = "ip-adapter-faceid-plusv2_sd15.bin"
        clip_dir = faceid_dir / "image_encoder"
        if not (clip_dir / "model.safetensors").exists():
            print(f"!! plusv2 需要 CLIP 图像编码器: {clip_dir}")
            print("   先跑：python scripts/download_data.py --config configs/data/ipadapter_plusv2.yaml")
            return 2
    pipe.load_ip_adapter(
        str(faceid_dir),
        # ⚠️ subfolder **不能**是 None：diffusers 里
        #      image_encoder_subfolder = Path(subfolder, image_encoder_folder).as_posix()
        #    只要给了 image_encoder_folder，就会把 subfolder 也传进 Path()，
        #    None 会直接 TypeError: expected str, bytes or os.PathLike object, not NoneType。
        #    （真实踩过：plusv2 一开就崩）
        subfolder="" if plusv2 else None,
        weight_name=faceid_bin,
        # base FaceID 不需要 CLIP 编码器；plusv2 需要（放在 image_encoder 子目录下）
        image_encoder_folder="image_encoder" if plusv2 else None,
    )
    pipe.set_ip_adapter_scale(args.scale)
    print(f"    IP-Adapter: {'FaceID plusv2（CLIP + ID 双路）' if plusv2 else 'base FaceID（仅 ID 一路）'}")

    if plusv2:
        # ⚠️⚠️ diffusers **没有实现** FaceIDPlus 的 CLIP 这一路，必须手工接。
        #
        #    `IPAdapterFaceIDPlusImageProjection.forward(id_embeds)` 内部用的是
        #    `self.clip_embeds`，而整个 diffusers 包里**没有任何地方给它赋值**
        #    （只有 `__init__` 里的 `self.clip_embeds = None`）。
        #    更绕的是：管线 `encode_image(..., output_hidden_state=True)` 明明算出了
        #    CLIP 隐状态，却把它当作 `id_embeds` **传进了 forward**，而 forward
        #    完全不用这个参数 —— 于是整条路断掉，走标准调用必崩。
        #
        #    所以这里自己算、自己塞：
        #      * 形状必须是 **4 维** `(2, 1, 257, 1280)`：
        #          - forward 里 `clip_embeds.reshape(-1, shape[2], shape[3])` 要求 >=4 维
        #          - 最后一维 1280 = CLIP ViT-H 的隐藏维（proj_in = Linear(1280, 768)）
        #          - 257 = 256 patch + 1 CLS
        #          - 第 0 维 2 = CFG 的 [负例, 正例]，负例给全零（与 ID 那一路一致）
        from transformers import CLIPImageProcessor

        # ⚠️ h94/IP-Adapter 的 models/image_encoder/ 里**没有** preprocessor_config.json
        #    （只有 config.json 和 model.safetensors），所以 from_pretrained 会报
        #    "Can't load image processor ... preprocessor_config.json"。
        #    不用再下载 —— CLIP 的图像预处理是**公开固定常量**，直接构造即可。
        #    下面这组值与 laion/CLIP-ViT-H-14 的 preprocessor_config.json 一致
        #    （proj_in = Linear(1280, 768) 说明这个编码器是 ViT-H/14，隐藏维 1280）。
        try:
            clip_proc = CLIPImageProcessor.from_pretrained(str(clip_dir))
        except OSError:
            clip_proc = CLIPImageProcessor(
                size={"shortest_edge": 224},
                crop_size={"height": 224, "width": 224},
                resample=3,                       # PILImageResampling.BICUBIC
                image_mean=[0.48145466, 0.4578275, 0.40821073],
                image_std=[0.26862954, 0.26130258, 0.27577711],
                do_center_crop=True, do_normalize=True, do_resize=True,
                do_rescale=True, do_convert_rgb=True,
            )
            print("    CLIP 预处理: 用内置常量（仓库里没有 preprocessor_config.json）")

        def clip_hidden(pil_img):
            px = clip_proc(images=pil_img, return_tensors="pt").pixel_values
            px = px.to(device=pipe.device, dtype=pipe.dtype)
            with torch.no_grad():
                h = pipe.image_encoder(px).last_hidden_state          # (1, 257, 1280)
            return h.to(dtype=pipe.dtype)

    # 采样器：PNDM 是 SD1.5 的老默认；DPM++ 2M Karras 是社区标准，同样步数下细节更好
    if args.scheduler == "dpmpp":
        from diffusers import DPMSolverMultistepScheduler
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, use_karras_sigmas=True)
        print("    采样器: DPM++ 2M Karras")
    else:
        print("    采样器: PNDM（默认）")

    # ---------- 解码：用管线自带的标准路径 ----------
    # ⚠️ 这里曾经被我改成"手动拿 latents 再用 fp32 VAE 解码"，理由是怀疑 fp16 VAE 溢出。
    #    A/B 实测（同一 seed，三种解码）证明**三者输出完全一致**（均值 94.6、R-B 差异 35.3、
    #    id_sim 0.4002 / 0.3998 / 0.4002）—— 手动解码并不更好，反而多两个出错点。
    #    所以回到标准路径：直接让管线解码。少动一个部件，少一份风险。
    pipe = pipe.to("cuda")
    if plusv2:
        # CLIP 编码器默认会以 fp32 加载（2.5 GB 权重 = 5 GB 显存），8 GB 卡上会 OOM。
        # 手工降到 fp16（1.25 GB），与主模型一致。
        pipe.image_encoder.to(device=pipe.device, dtype=torch.float16)
    pipe.set_progress_bar_config(disable=True)
    print(f"    底模={args.base}  scale={args.scale}  device={pipe.device}")

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

        if plusv2:
            # CLIP 这一路：每个身份都要重算（不同人 → 不同隐状态），
            # 然后在整段去噪里保持不变（投影层每步都会读它）。
            ref_pil = Image.open(clip_paths[ident]).convert("RGB")
            hidden = clip_hidden(ref_pil)                       # (1, 257, 1280)
            proj = pipe.unet.encoder_hid_proj.image_projection_layers[0]
            proj.clip_embeds = torch.cat([torch.zeros_like(hidden), hidden], dim=0).unsqueeze(1)
            assert proj.clip_embeds.shape == (2, 1, 257, hidden.shape[-1]), proj.clip_embeds.shape

        for k in range(args.per_identity):
            # ⚠️ 必须用**确定性**哈希：Python 内置 hash() 对字符串每个进程都会变
            #    （PYTHONHASHSEED 随机化），那样种子无法复现、实验结果不可复现。
            seed = args.seed + zlib.crc32(f"{ident}_{k}".encode()) % 100000
            generator = torch.Generator(device="cuda").manual_seed(seed)
            prompt = PROMPTS[k % len(PROMPTS)]
            # 标准解码路径（见上面的说明：手动解码并不更好）
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
                    "ip_adapter_scale": args.scale, "id_model": "buffalo_l",
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
        "generator": "ipadapter-faceid-sd15 (base FaceID, buffalo_l embeddings)",
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
        "id_embedding_model": "buffalo_l",
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
