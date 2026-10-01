#!/usr/bin/env python
"""生成肉眼评测用的对照图（真实参考 | base FaceID | plusv2）

产出：一张网格图，每行一个身份：
    [身份名]  [真实参考 ×N]  |  [方案A ×N]  |  [方案B ×N]
每张合成图下面标注它的 id_sim，便于把"看起来像"和"数字"对上。

⚠️ 合成图会**按检测到的人脸裁剪并放大**再贴 —— 直接贴 512×512 原图的话
   人脸只占一小块，肉眼看不出区别，对照就失去意义。

用法：
    python tools/contact_sheet.py \
        --arm "base FaceID=results/runs/_eye_base/m.jsonl" \
        --arm "plusv2=results/runs/_eye_pv2/m.jsonl" \
        --out results/runs/w3-generator-comparison/contact_sheet.png
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.stdout.reconfigure(errors="replace")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from aigcfr.eval.embed import load_app  # noqa: E402
from aigcfr.utils.config import find_local_config, load_config  # noqa: E402

CELL = 200          # 每个单元格边长
PAD = 8
LABEL_H = 26
ROW_LABEL_W = 210
HEADER_H = 34


def load_font(size: int):
    for name in ("msyh.ttc", "msyhbd.ttc", "simhei.ttf", "arial.ttf"):
        p = Path(r"C:\Windows\Fonts") / name
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size)
            except OSError:
                continue
    return ImageFont.load_default()


def face_crop(img_bgr: np.ndarray, app, size: int) -> np.ndarray | None:
    """检测人脸并裁出带边距的方形区域，缩放到 size×size。"""
    faces = app.get(img_bgr)
    if not faces:
        return None
    f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    x1, y1, x2, y2 = f.bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * 0.85          # 边距系数：人脸占画面约 60%
    h, w = img_bgr.shape[:2]
    x1 = int(max(0, cx - half)); x2 = int(min(w, cx + half))
    y1 = int(max(0, cy - half)); y2 = int(min(h, cy + half))
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--arm", action="append", required=True, metavar="名称=manifest路径",
                    help="可重复；每多一个方案就多一组列")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-identity", type=int, default=3, help="每个身份取几张")
    ap.add_argument("--real", type=int, default=3, help="每个身份展示几张真实参考图")
    args = ap.parse_args()

    cfg = load_config(None, profile="auto", local_path=str(find_local_config()))
    data_root = Path(cfg["paths"]["data_root"]).resolve()
    models_root = Path(cfg["paths"]["models_root"])
    app = load_app(models_root / "insightface", det_size=320, ctx_id=0)

    arms: list[tuple[str, Path]] = []
    for spec in args.arm:
        name, _, path = spec.partition("=")
        arms.append((name, Path(path)))

    # 每个方案：身份 -> [(裁剪后的图, id_sim)]
    data: dict[str, dict[str, list]] = {}
    scores: list[dict[str, dict[str, float]]] = []
    for name, manifest in arms:
        recs = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
        by_id: dict[str, list] = {}
        for r in recs:
            p = Path(r["path"])
            p = p if p.is_absolute() else (data_root / p if (data_root / p).exists() else REPO_ROOT / p)
            img = cv2.imread(str(p))
            if img is None:
                continue
            crop = face_crop(img, app, CELL)
            if crop is None:
                crop = cv2.resize(img, (CELL, CELL))
            by_id.setdefault(r["identity_id"], []).append(crop)
        data[name] = by_id
        # 读同目录的 id_sim（filter_synth 的 synth_scores.csv）
        sc = manifest.parent / "synth_scores.csv"
        m: dict[str, dict[str, float]] = {}
        if sc.exists():
            for row in csv.DictReader(open(sc, encoding="utf-8")):
                if row.get("id_sim"):
                    m.setdefault(row["identity_id"], {})[row["image_id"]] = float(row["id_sim"])
        scores.append(m)
        print(f"  {name}: {sum(len(v) for v in by_id.values())} 张")

    identities = sorted(set.intersection(*(set(d) for d in data.values())))[:4]

    # 真实参考图
    real_recs = [json.loads(l) for l in (REPO_ROOT / cfg["data"]["manifest"]).read_text(
        encoding="utf-8").splitlines() if l.strip()]
    real_by_id: dict[str, list] = {}
    for r in real_recs:
        if r.get("split") == "train" and r.get("status") == "accepted":
            real_by_id.setdefault(r["identity_id"], []).append(r)

    n_cols = args.real + len(arms) * args.per_identity
    width = ROW_LABEL_W + PAD + n_cols * (CELL + PAD) + PAD
    height = HEADER_H + PAD + len(identities) * (CELL + LABEL_H + PAD) + PAD
    sheet = Image.new("RGB", (width, height), (248, 248, 250))
    draw = ImageDraw.Draw(sheet)
    f_head = load_font(17)
    f_lab = load_font(13)
    f_row = load_font(15)

    # 表头
    x = ROW_LABEL_W + PAD
    draw.text((PAD, 9), "身份", font=f_head, fill=(20, 20, 20))
    draw.text((x + 4, 9), "真实参考图", font=f_head, fill=(20, 90, 20))
    x += args.real * (CELL + PAD)
    for name, _ in arms:
        draw.text((x + 4, 9), name, font=f_head, fill=(20, 40, 120))
        x += args.per_identity * (CELL + PAD)

    for ri, ident in enumerate(identities):
        y = HEADER_H + PAD + ri * (CELL + LABEL_H + PAD)
        draw.text((PAD + 4, y + CELL // 2 - 10), ident, font=f_row, fill=(20, 20, 20))
        x = ROW_LABEL_W + PAD
        refs = sorted(real_by_id.get(ident, []), key=lambda r: -r["meta"].get("det_score", 0))
        for j in range(args.real):
            if j < len(refs):
                p = data_root / refs[j]["path"]
                img = cv2.imread(str(p))
                if img is not None:
                    cell = cv2.resize(img, (CELL, CELL), interpolation=cv2.INTER_CUBIC)[:, :, ::-1]
                    sheet.paste(Image.fromarray(cell), (x, y))
            draw.rectangle([x, y, x + CELL, y + CELL], outline=(200, 200, 200))
            x += CELL + PAD
        for ai, (name, _) in enumerate(arms):
            imgs = data[name].get(ident, [])[: args.per_identity]
            smap = scores[ai].get(ident, {})
            ordered = sorted(smap.items(), key=lambda kv: kv[0])
            for j in range(args.per_identity):
                if j < len(imgs):
                    sheet.paste(Image.fromarray(imgs[j][:, :, ::-1]), (x, y))
                    # 标 id_sim
                    sim = ordered[j][1] if j < len(ordered) else None
                    if sim is not None:
                        txt = f"{sim:+.3f}"
                        col = (10, 130, 10) if sim >= 0.4 else (200, 60, 20)
                        draw.rectangle([x, y + CELL - 20, x + 58, y + CELL], fill=(255, 255, 255))
                        draw.text((x + 3, y + CELL - 18), txt, font=f_lab, fill=col)
                draw.rectangle([x, y, x + CELL, y + CELL], outline=(200, 200, 200))
                x += CELL + PAD

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"\n对照图: {out}  ({sheet.width}x{sheet.height}, {out.stat().st_size/1e6:.2f} MB)")
    print("说明：每张合成图左下角是它的 id_sim；>=0.4（绿）视为'像同一个人'，<0.4（红）不算。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
