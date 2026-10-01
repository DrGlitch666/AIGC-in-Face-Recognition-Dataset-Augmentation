"""一次性扫描（用完即删）：IP-Adapter 强度 × 提示词，找能生成"可检出人脸"的配置。"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.stdout.reconfigure(errors="replace")

import torch  # noqa: F401

from aigcfr.eval.embed import load_app, warmup
from aigcfr.eval.faces import identity_embedding
from aigcfr.utils.config import find_local_config, load_config

OUT = REPO_ROOT / "results" / "runs" / "_diag_sweep"
OUT.mkdir(parents=True, exist_ok=True)
import json

cfg = load_config(None, profile="auto", local_path=str(find_local_config()))
data_root = Path(cfg["paths"]["data_root"])
models_root = Path(cfg["paths"]["models_root"])
train = REPO_ROOT / cfg["data"]["manifest"]

records = [json.loads(l) for l in train.read_text(encoding="utf-8").splitlines() if l.strip()]
real = [r for r in records if r.get("split") == "train" and r.get("status") == "accepted"]
ident = sorted({r["identity_id"] for r in real})[0]
rec = sorted([r for r in real if r["identity_id"] == ident],
             key=lambda r: -r["meta"].get("det_score", 0))[0]

app = load_app(models_root / "insightface", det_size=640, ctx_id=0)
warmup(app)
ref_emb = identity_embedding(app, data_root / rec["path"])
print(f"参考身份 {ident}  参考图 {rec['path']}")

from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    str(models_root / "sd15-faceid" / "sd15"), torch_dtype=torch.float16, variant="fp16",
    safety_checker=None, requires_safety_checker=False)
pipe.set_progress_bar_config(disable=True)
pipe = pipe.to("cuda")
pipe.load_ip_adapter(str(models_root / "sd15-faceid" / "ip-adapter-faceid"), subfolder=None,
                     weight_name="ip-adapter-faceid_sd15.bin", image_encoder_folder=None)

base = torch.from_numpy(ref_emb).to(dtype=pipe.dtype, device="cuda").reshape(1, 1, -1)
embeds = torch.cat([torch.zeros_like(base), base], dim=0)

PROMPTS = {
    "face": "a photo of a person's face, frontal view, natural lighting, sharp focus",
    "portrait": "a portrait photo of a person, looking at the camera",
    "simple": "a photo of a face",
}
NEG = "blurry, low quality, distorted, deformed, cartoon, painting, watermark, text"

print()
print("=" * 84)
print(f"{'scale':>6} {'prompt':>10} {'检出':>5} {'id_sim':>9}  图像统计")
print("=" * 84)
results = []
for scale in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
    pipe.set_ip_adapter_scale(scale)
    for pname, prompt in PROMPTS.items():
        gen = torch.Generator(device="cuda").manual_seed(7)
        # ⚠️ 即使 scale=0 也必须传 embeds：一旦加载了 IP-Adapter，
        #    UNet 的 process_encoder_hidden_states 就会去读 added_cond_kwargs，
        #    传 None 会报 "argument of type 'NoneType' is not iterable"（真实踩过）。
        im = pipe(prompt=prompt, negative_prompt=NEG,
                  ip_adapter_image_embeds=[embeds],
                  num_inference_steps=25, guidance_scale=5.0, generator=gen,
                  height=512, width=512).images[0]
        tag = f"s{scale:.1f}_{pname}"
        p = OUT / f"{tag}.png"
        im.save(p)
        vec = identity_embedding(app, p)
        import numpy as np

        arr = np.asarray(im)
        sim = float(ref_emb @ vec) if vec is not None else None
        ok = "YES" if vec is not None else "no"
        simtxt = f"{sim:+.4f}" if sim is not None else "   -   "
        print(f"{scale:>6.1f} {pname:>10} {ok:>5} {simtxt:>9}  "
              f"均值{arr.mean():5.0f} 标准差{arr.std():5.0f}")
        results.append((scale, pname, sim))

ok_ones = [r for r in results if r[2] is not None]
print()
print(f"共 {len(results)} 张，检出人脸 {len(ok_ones)} 张")
if ok_ones:
    best = max(ok_ones, key=lambda r: r[2])
    print(f"最佳: scale={best[0]} prompt={best[1]} id_sim={best[2]:+.4f}")
print(f"图片在: {OUT}")
