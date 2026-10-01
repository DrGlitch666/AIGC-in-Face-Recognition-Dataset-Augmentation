#!/usr/bin/env python
"""IP-Adapter-FaceID **plusv2** 专用生成脚本（独立实现，严格对齐官方）

## 为什么单独写这个文件

`scripts/generate.py` 走的是 base FaceID（只用人脸 ID 嵌入）。plusv2 是**两路条件**
（人脸 ID + CLIP 图像嵌入），调用约定完全不同，混在一个脚本里会互相干扰。
所以这里**从零按官方实现写**，不复用 `generate.py` 的任何生成逻辑。

## 官方配方（tencent-ailab/IP-Adapter + h94/IP-Adapter-FaceID README）

```python
app = FaceAnalysis(name="buffalo_l", ...)            # 注意：buffalo_l，不是 antelopev2
faces = app.get(image)
faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
face_image = face_align.norm_crop(image, landmark=faces[0].kps, image_size=224)

ip_model = IPAdapterFaceIDPlus(pipe, "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", ip_ckpt, device)
images = ip_model.generate(
    prompt=..., negative_prompt=...,
    face_image=face_image, faceid_embeds=faceid_embeds,
    shortcut=v2,          # ← plusv2 用 True
    s_scale=1.0,
    num_inference_steps=30, seed=2023)
```

官方 `IPAdapterFaceIDPlus.get_image_embeds` 的两路条件是：

```python
clip_image_embeds        = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
uncond_clip_image_embeds = self.image_encoder(torch.zeros_like(clip_image),
                                              output_hidden_states=True).hidden_states[-2]
image_prompt_embeds        = self.image_proj_model(faceid_embeds,
                                                   clip_image_embeds, shortcut, s_scale)
uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(faceid_embeds),
                                                   uncond_clip_image_embeds, shortcut, s_scale)
```

官方 `ProjPlusModel.forward`：

```python
x = self.norm(self.proj(id_embeds).reshape(-1, num_tokens, cross_attention_dim))
out = self.perceiver_resampler(x, clip_embeds)
if shortcut:
    out = x + scale * out        # ← 残差！保住 ID 分支
return out
```

## 与 base FaceID 的三处关键差别（都是实测踩过的坑）

1. **`shortcut` 必须为 True**。diffusers 里 `if self.shortcut: out = id_embeds + scale*out`；
   若为 False，**感知器的输出会直接覆盖掉 ID 分支**，身份信号全丢 —— 实测身份相似度
   从应有的 0.4 掉到 0.08（"适配器等于没起作用"）。
2. **CLIP 取 `hidden_states[-2]`（倒数第二层）**，不是 `last_hidden_state`。
3. **负例的 CLIP 嵌入是"对全零图像跑一遍编码器"**，不是全零张量。

另外官方明确：CLIP 那一路的输入是 **`norm_crop(..., image_size=224)` 的人脸对齐裁剪**，
不是原始整图，也不是 112 的裁剪。

## 用法

    python scripts/generate_plusv2.py --identities 2 --per-identity 2 --out results/runs/_pv2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.stdout.reconfigure(errors="replace")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402  —— 必须在 onnxruntime 之前 import
from PIL import Image  # noqa: E402

from aigcfr.utils.config import find_local_config, load_config  # noqa: E402

# ---------------------------------------------------------------- 官方默认值
PROMPTS = [
    "photo of a person, best quality, high quality",
    "a close-up portrait photo of a person, looking at the camera, best quality",
    "photo of a person in a garden, natural lighting, best quality",
    "a professional headshot photo of a person, studio lighting, best quality",
]
NEGATIVE = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(allow_abbrev=False, description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-id", default="syn-plusv2")
    ap.add_argument("--identities", type=int, default=None, help="只用前 N 个身份")
    ap.add_argument("--per-identity", type=int, default=4)
    ap.add_argument("--out", default=None, help="默认 <synth_root>/<exp-id>")
    ap.add_argument("--manifest", default=None, help="默认 data/manifests/<exp-id>.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base", default="sd15", choices=["sd15", "realvis"])
    ap.add_argument("--vae", default="mse", choices=["none", "mse"],
                    help="官方用 stabilityai/sd-vae-ft-mse（默认 mse）")
    ap.add_argument("--scheduler", default="ddim", choices=["ddim", "dpmpp", "pndm"],
                    help="官方用 DDIM（默认）")
    ap.add_argument("--steps", type=int, default=30, help="官方 30")
    ap.add_argument("--guidance", type=float, default=7.5, help="官方 7.5")
    ap.add_argument("--scale", type=float, default=1.0, help="IP-Adapter 强度，官方 1.0")
    ap.add_argument("--s-scale", type=float, default=1.0, help="shortcut 残差权重，官方 1.0")
    ap.add_argument("--no-shortcut", action="store_true",
                    help="关掉 shortcut 残差（**会丢掉身份信号**，只用于对照实验）")
    ap.add_argument("--ref-mode", default="mean", choices=["single", "mean"],
                    help="ID 嵌入取单张还是多张平均。官方用单张，但实测**多张平均明显更好**"
                         "（0.5138 -> 0.6084，且最差样本从 0.4320 提到 0.5273），故默认 mean")
    ap.add_argument("--ref-count", type=int, default=5)
    ap.add_argument("--clip-layer", type=int, default=-2,
                    help="CLIP 取第几层隐状态，官方为 -2（倒数第二层）")
    return ap.parse_args()


def build_pipeline(args, sd_root: Path):
    """按官方配方装管线：底模 + 可选 MSE VAE + 调度器 + plusv2 适配器。"""
    from diffusers import AutoencoderKL, DDIMScheduler, DPMSolverMultistepScheduler, StableDiffusionPipeline

    if args.base == "realvis":
        ckpt = sd_root / "realvis" / "Realistic_Vision_V6.0_NV_B1_fp16.safetensors"
        if not ckpt.exists():
            print(f"!! 缺底模 {ckpt}（download_data.py --config configs/data/sd15_realvis.yaml）")
            raise SystemExit(2)
        pipe = StableDiffusionPipeline.from_single_file(
            str(ckpt), config=str(sd_root / "sd15"), torch_dtype=torch.float16,
            safety_checker=None, requires_safety_checker=False)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            str(sd_root / "sd15"), torch_dtype=torch.float16, variant="fp16",
            safety_checker=None, requires_safety_checker=False)

    if args.vae == "mse":
        vae_dir = sd_root / "vae-mse"
        if not (vae_dir / "diffusion_pytorch_model.safetensors").exists():
            print(f"!! 缺 MSE VAE {vae_dir}（download_data.py --config configs/data/sd15_vae.yaml）")
            raise SystemExit(2)
        pipe.vae = AutoencoderKL.from_pretrained(str(vae_dir), torch_dtype=torch.float16)

    # 官方用的是 DDIM 这组参数
    if args.scheduler == "ddim":
        pipe.scheduler = DDIMScheduler(
            num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
            beta_schedule="scaled_linear", clip_sample=False,
            set_alpha_to_one=False, steps_offset=1)
    elif args.scheduler == "dpmpp":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, use_karras_sigmas=True)

    # ⚠️ subfolder 不能是 None：diffusers 会做 Path(subfolder, image_encoder_folder)
    pipe.load_ip_adapter(str(sd_root / "ip-adapter-faceid"), subfolder="",
                         weight_name="ip-adapter-faceid-plusv2_sd15.bin",
                         image_encoder_folder="image_encoder")
    pipe.set_ip_adapter_scale(args.scale)
    pipe = pipe.to("cuda")
    # CLIP 编码器按 fp32 加载会占 5GB 显存，8GB 卡会 OOM
    pipe.image_encoder.to(device=pipe.device, dtype=torch.float16)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def get_clip_processor(clip_dir: Path):
    """CLIP 预处理。

    ⚠️ h94/IP-Adapter 的 models/image_encoder/ 里**没有** preprocessor_config.json，
    所以 from_pretrained 会失败。CLIP 的图像预处理是公开固定常量，直接构造即可
    （与 laion/CLIP-ViT-H-14 的 preprocessor_config.json 一致）。
    """
    from transformers import CLIPImageProcessor

    try:
        return CLIPImageProcessor.from_pretrained(str(clip_dir))
    except OSError:
        return CLIPImageProcessor(
            size={"shortest_edge": 224}, crop_size={"height": 224, "width": 224},
            resample=3,                                   # BICUBIC
            image_mean=[0.48145466, 0.4578275, 0.40821073],
            image_std=[0.26862954, 0.26130258, 0.27577711],
            do_center_crop=True, do_normalize=True, do_resize=True,
            do_rescale=True, do_convert_rgb=True,
        )


@torch.inference_mode()
def clip_hidden_states(pipe, processor, pil_image, layer: int) -> torch.Tensor:
    """取 CLIP 的第 `layer` 层隐状态（官方用 -2）。"""
    px = processor(images=pil_image, return_tensors="pt").pixel_values
    px = px.to(device=pipe.device, dtype=pipe.dtype)
    return pipe.image_encoder(px, output_hidden_states=True).hidden_states[layer]


@torch.inference_mode()
def clip_hidden_states_zeros(pipe, processor, pil_image, layer: int) -> torch.Tensor:
    """**负例的 CLIP 嵌入 = 对全零图像跑一遍编码器**（官方做法，不是全零张量）。"""
    px = processor(images=pil_image, return_tensors="pt").pixel_values
    px = torch.zeros_like(px).to(device=pipe.device, dtype=pipe.dtype)
    return pipe.image_encoder(px, output_hidden_states=True).hidden_states[layer]


def main() -> int:
    args = parse_args()
    cfg = load_config(None, profile="auto", local_path=str(find_local_config()))
    paths = cfg["paths"]
    data_root = Path(paths["data_root"])
    sd_root = Path(paths["models_root"]) / "sd15-faceid"

    # ⚠️ 必须 .resolve()：`--out results/runs/xxx` 是**相对路径**，不解析的话
    #    `target.relative_to(data_root)` 必然抛 ValueError，于是退化成
    #    "相对仓库根"的相对路径，而消费端（filter_synth.py）按"相对 data_root"解析，
    #    结果全部 no_face。**这个 bug 真实踩过，害得 6 张好图被判成检不出人脸。**
    out_dir = Path(args.out) if args.out else \
        Path(paths.get("synth_root", data_root / "processed/synth")) / args.exp_id
    out_dir = out_dir.resolve()
    manifest = Path(args.manifest) if args.manifest else REPO_ROOT / "data" / "manifests" / f"{args.exp_id}.jsonl"
    manifest = manifest.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    data_root = data_root.resolve()

    print("=" * 74)
    print("IP-Adapter-FaceID plusv2 专用生成")
    print(f"  底模={args.base}  VAE={args.vae}  调度器={args.scheduler}  步数={args.steps}")
    print(f"  guidance={args.guidance}  IP scale={args.scale}  shortcut={'OFF' if args.no_shortcut else 'ON'}"
          f"  s_scale={args.s_scale}  CLIP 层={args.clip_layer}")
    print(f"  输出: {out_dir}")
    print("=" * 74)

    # ---------------- 1) 人脸：ID 嵌入 + 224 对齐裁剪 ----------------
    print("\n[1/5] buffalo_l 提取 ID 嵌入与 224 对齐人脸 ...")
    from insightface.app import FaceAnalysis
    from insightface.utils import face_align

    app = FaceAnalysis(name="buffalo_l", root=str(Path(paths["models_root"]) / "insightface"),
                       providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                       allowed_modules=["detection", "recognition"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    print(f"    provider = {list(app.models['recognition'].session.get_providers())}")

    records = [json.loads(l) for l in (REPO_ROOT / cfg["data"]["manifest"]).read_text(
        encoding="utf-8").splitlines() if l.strip()]
    real = [r for r in records if r.get("split") == "train" and r.get("status") == "accepted"]
    by_id: dict[str, list[dict]] = {}
    for r in real:
        by_id.setdefault(r["identity_id"], []).append(r)
    identities = sorted(by_id)
    if args.identities:
        identities = identities[: args.identities]

    id_embeds: dict[str, np.ndarray] = {}
    faces224: dict[str, np.ndarray] = {}
    for ident in identities:
        cands = sorted(by_id[ident], key=lambda r: -r["meta"].get("det_score", 0))
        vecs, face_img = [], None
        for rec in cands[: args.ref_count] if args.ref_mode == "mean" else cands[:1]:
            origin = rec["meta"].get("origin")
            src = (data_root / origin) if origin else (data_root / rec["path"])
            img = cv2.imread(str(src))
            if img is None:
                continue
            faces = app.get(img)
            if not faces:
                continue
            f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            vecs.append(np.asarray(f.normed_embedding, dtype=np.float32))
            if face_img is None:
                # 官方：face_align.norm_crop(image, landmark=kps, image_size=224)
                face_img = face_align.norm_crop(img, landmark=f.kps, image_size=224)
        if not vecs or face_img is None:
            continue
        if args.ref_mode == "mean":
            m = np.mean(np.stack(vecs), axis=0)
            id_embeds[ident] = m / max(float(np.linalg.norm(m)), 1e-12)
        else:
            id_embeds[ident] = vecs[0]
        faces224[ident] = face_img
    print(f"    {len(id_embeds)} 个身份（ref-mode={args.ref_mode}，{args.ref_count if args.ref_mode=='mean' else 1} 张参考）")
    if not id_embeds:
        return 1

    # ---------------- 2) 管线 ----------------
    print("\n[2/5] 装管线（底模 + plusv2 + CLIP）...")
    pipe = build_pipeline(args, sd_root)
    processor = get_clip_processor(sd_root / "ip-adapter-faceid" / "image_encoder")
    proj = pipe.unet.encoder_hid_proj.image_projection_layers[0]
    proj.shortcut = not args.no_shortcut          # ← 关键：plusv2 必须 True
    proj.shortcut_scale = args.s_scale
    print(f"    投影层 {type(proj).__name__}  shortcut={proj.shortcut}  "
          f"shortcut_scale={proj.shortcut_scale}  num_tokens={proj.num_tokens}")

    # ---------------- 3) 负例 CLIP ----------------
    print("\n[3/5] 算负例 CLIP 嵌入（对全零图像编码）...")
    sample_face = next(iter(faces224.values()))
    neg_clip = clip_hidden_states_zeros(pipe, processor, sample_face, args.clip_layer)
    print(f"    负例 CLIP {tuple(neg_clip.shape)}  均值 {neg_clip.mean().item():+.4f} "
          f"（官方做法，**不是**全零张量）")

    # ---------------- 4) 逐身份生成 ----------------
    print("\n[4/5] 生成 ...")
    records_out: list[dict] = []
    t0 = time.time()
    total = len(id_embeds) * args.per_identity
    done = 0
    for ident, emb in id_embeds.items():
        face_rgb = Image.fromarray(cv2.cvtColor(faces224[ident], cv2.COLOR_BGR2RGB))

        # ID 那一路：CFG 的 [负例(全零), 正例]
        idt = torch.from_numpy(emb).to(dtype=pipe.dtype, device=pipe.device).reshape(1, 1, -1)
        id_embeds_pair = torch.cat([torch.zeros_like(idt), idt], dim=0)

        # CLIP 那一路：官方取 hidden_states[-2]，负例来自全零图像
        clip_pos = clip_hidden_states(pipe, processor, face_rgb, args.clip_layer)
        # diffusers 要求 4 维并在 batch 维对齐 CFG：[负, 正]
        proj.clip_embeds = torch.cat([neg_clip, clip_pos], dim=0).unsqueeze(1)
        assert proj.clip_embeds.shape[0] == 2, proj.clip_embeds.shape

        for k in range(args.per_identity):
            seed = args.seed + zlib.crc32(f"{ident}_{k}".encode()) % 100000
            gen = torch.Generator(device="cuda").manual_seed(seed)
            prompt = PROMPTS[k % len(PROMPTS)]
            image = pipe(prompt=prompt, negative_prompt=NEGATIVE,
                         ip_adapter_image_embeds=[id_embeds_pair],
                         num_inference_steps=args.steps, guidance_scale=args.guidance,
                         generator=gen, height=512, width=512).images[0]
            name = f"{ident}_plusv2_{k:02d}.png"
            target = out_dir / ident / name
            target.parent.mkdir(parents=True, exist_ok=True)
            image.save(target)
            try:
                rel = target.relative_to(data_root).as_posix()
            except ValueError:
                rel = target.as_posix()
            records_out.append({
                "image_id": f"syn_{ident}_{k:02d}", "path": rel, "identity_id": ident,
                "source": "synth", "generator": "ipadapter-faceid-plusv2-sd15",
                "parent_image_id": None, "split": "train", "status": "pending",
                "meta": {"exp_id": args.exp_id, "seed": seed, "prompt": prompt,
                         "ip_scale": args.scale, "guidance": args.guidance,
                         "steps": args.steps, "shortcut": proj.shortcut,
                         "s_scale": args.s_scale, "clip_layer": args.clip_layer,
                         "base": args.base, "vae": args.vae, "scheduler": args.scheduler,
                         "ref_mode": args.ref_mode, "id_model": "buffalo_l"},
            })
            done += 1
            el = time.time() - t0
            print(f"\r  {done}/{total}  {done/el:.2f} 张/秒  剩余约 {el/done*(total-done)/60:.1f} 分钟",
                  end="", flush=True)
    print()

    # ---------------- 5) 写报告 ----------------
    print("\n[5/5] 写清单 ...")
    with open(manifest, "w", encoding="utf-8") as fh:
        for r in records_out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    report = {"exp_id": args.exp_id, "n_images": len(records_out), "n_identities": len(id_embeds),
              "generator": "ipadapter-faceid-plusv2-sd15", "base": args.base, "vae": args.vae,
              "scheduler": args.scheduler, "steps": args.steps, "guidance": args.guidance,
              "ip_scale": args.scale, "shortcut": proj.shortcut, "s_scale": args.s_scale,
              "clip_layer": args.clip_layer, "id_model": "buffalo_l", "ref_mode": args.ref_mode,
              "out_dir": str(out_dir), "manifest": str(manifest),
              "seconds": round(time.time() - t0, 1)}
    (out_dir / "gen_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    print(f"  生成 {len(records_out)} 张，用时 {time.time()-t0:.0f}s")
    print(f"  manifest: {manifest}（status=pending，下一步跑 filter_synth.py）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
