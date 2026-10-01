"""一次性诊断（用完即删）：为什么生成的人不像本人、甚至检测不到人脸。

把问题二分：
  A. 纯 SD1.5（不加载 IP-Adapter）-> 生成的"人脸"能不能被 buffalo_l 检测到？
  B. 加载 IP-Adapter + 人脸嵌入 -> 呢？

  如果 A 就失败  -> 问题在 SD1.5 管线本身（权重/dtype/配置）
  如果 A 成、B 败 -> 问题在 IP-Adapter 的集成或嵌入

同时打印参考图的实际情况（回答"是不是把不同的人当成一个人"）。
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.stdout.reconfigure(errors="replace")

import cv2
import numpy as np
import torch  # noqa: F401  —— 必须在 onnxruntime 之前

from aigcfr.eval.embed import load_app, warmup
from aigcfr.eval.faces import identity_embedding
from aigcfr.utils.config import find_local_config, load_config

OUT = REPO_ROOT / "results" / "runs" / "_diag_gen"
OUT.mkdir(parents=True, exist_ok=True)

cfg = load_config(None, profile="auto", local_path=str(find_local_config()))
paths = cfg["paths"]
data_root = Path(paths["data_root"])
models_root = Path(paths["models_root"])
sd15 = models_root / "sd15-faceid" / "sd15"
faceid = models_root / "sd15-faceid" / "ip-adapter-faceid"

# ---------- 0) 参考图与嵌入 ----------
train = REPO_ROOT / cfg["data"]["manifest"]
by_id: dict[str, list[dict]] = {}
for line in train.read_text(encoding="utf-8").splitlines():
    if line.strip():
        r = json.loads(line)
        if r.get("split") == "train" and r.get("status") == "accepted":
            by_id.setdefault(r["identity_id"], []).append(r)

ident = sorted(by_id)[0]
rec = sorted(by_id[ident], key=lambda r: -r["meta"].get("det_score", 0))[0]
ref_path = data_root / rec["path"]
print("=" * 74)
print(f"[0] 参考身份: {ident}")
print(f"    参考图  : {ref_path}")
print(f"    该身份共有 {len(by_id[ident])} 张真实图, 全库身份数 {len(by_id)}")

app = load_app(models_root / "insightface", det_size=640, ctx_id=0)
warmup(app)
img_ref = cv2.imread(str(ref_path))
print(f"    参考图尺寸: {img_ref.shape}")
ref_emb = identity_embedding(app, ref_path)
if ref_emb is None:
    print("    !! 连参考图都拿不到嵌入，后面没意义")
    raise SystemExit(1)
print(f"    嵌入: shape={ref_emb.shape} norm={np.linalg.norm(ref_emb):.4f}")

# 参考图与同身份其他图的相似度（自检：说明嵌入是否可用）
others = [r for r in by_id[ident] if r["image_id"] != rec["image_id"]][:5]
sims = []
for o in others:
    vec = identity_embedding(app, data_root / o["path"])
    if vec is not None:
        sims.append(float(ref_emb @ vec))
print(f"    与同身份另外 {len(sims)} 张真实图的相似度: "
      f"{[round(s,3) for s in sims]}  均值 {np.mean(sims):.3f}" if sims else "    (无)")

# ---------- 加载管线 ----------
print()
print("=" * 74)
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    str(sd15), torch_dtype=torch.float16, variant="fp16",
    safety_checker=None, requires_safety_checker=False)
pipe.set_progress_bar_config(disable=True)
pipe = pipe.to("cuda")
print(f"[1] 纯 SD1.5 加载完成  dtype={pipe.dtype}  device={pipe.device}")

PROMPT = "a photo of a person's face, frontal view, natural lighting, sharp focus"
NEG = "blurry, low quality, distorted, deformed, cartoon, painting, watermark, text"


def detect(path, label):
    img = cv2.imread(str(path))
    if img is None:
        print(f"    {label}: 读不到")
        return None
    h, w = img.shape[:2]
    vec = identity_embedding(app, path)      # 512x512 -> 走"检测+选脸+对齐"
    if vec is None:
        print(f"    {label}: {w}x{h} 均值{img.mean():.0f} 标准差{img.std():.0f} -> 检出 0 张脸")
        return None
    sim = float(ref_emb @ vec)
    print(f"    {label}: {w}x{h} 均值{img.mean():.0f} -> 检出人脸, 与参考的 id_sim = {sim:+.4f}")
    return sim


print()
print("=" * 74)
print("[2] A 组：纯 SD1.5（不加载 IP-Adapter）")
print("=" * 74)
sims_a = []
for seed in (1, 2, 3):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    im = pipe(prompt=PROMPT, negative_prompt=NEG, num_inference_steps=25,
              guidance_scale=5.0, generator=gen, height=512, width=512).images[0]
    p = OUT / f"a_plain_seed{seed}.png"
    im.save(p)
    s = detect(p, f"seed={seed}")
    if s is not None:
        sims_a.append(s)

# ---------- 加载 IP-Adapter ----------
print()
print("=" * 74)
print("[3] 加载 IP-Adapter-FaceID")
print("=" * 74)
pipe.load_ip_adapter(str(faceid), subfolder=None,
                     weight_name="ip-adapter-faceid_sd15.bin", image_encoder_folder=None)
pipe.set_ip_adapter_scale(0.6)
proj = getattr(pipe.unet, "encoder_hid_proj", None)
print(f"    encoder_hid_proj = {type(proj).__name__ if proj else None}")
if proj is not None:
    layers = getattr(proj, "image_projection_layers", None)
    print(f"    image_projection_layers = {[type(x).__name__ for x in layers] if layers else None}")
try:
    adapters = pipe.get_active_adapters() if hasattr(pipe, "get_active_adapters") else None
    print(f"    LoRA adapters = {list(adapters) if adapters else None}")
except Exception as exc:  # noqa: BLE001
    print(f"    (读 LoRA 信息失败: {type(exc).__name__}: {exc})")

print()
print("=" * 74)
print("[4] B 组：带 IP-Adapter + 人脸嵌入")
print("=" * 74)
base = torch.from_numpy(ref_emb).to(dtype=pipe.dtype, device="cuda").reshape(1, 1, -1)
embeds = torch.cat([torch.zeros_like(base), base], dim=0)
print(f"    嵌入张量: shape={tuple(embeds.shape)} dtype={embeds.dtype} "
      f"[前半=全零, 后半=本人]")
sims_b = []
for seed in (1, 2, 3):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    im = pipe(prompt=PROMPT, negative_prompt=NEG, ip_adapter_image_embeds=[embeds],
              num_inference_steps=25, guidance_scale=5.0, generator=gen,
              height=512, width=512).images[0]
    p = OUT / f"b_faceid_seed{seed}.png"
    im.save(p)
    s = detect(p, f"seed={seed}")
    if s is not None:
        sims_b.append(s)

print()
print("=" * 74)
print("结论")
print("=" * 74)
print(f"  A 组（纯 SD1.5）  ：检出 {len(sims_a)}/3 张, id_sim 均值 "
      f"{np.mean(sims_a):+.4f}" if sims_a else "  A 组（纯 SD1.5）  ：一张人脸都没检出")
print(f"  B 组（带 FaceID）  ：检出 {len(sims_b)}/3 张, id_sim 均值 "
      f"{np.mean(sims_b):+.4f}" if sims_b else "  B 组（带 FaceID）  ：一张人脸都没检出")
print(f"  图片在: {OUT}")
