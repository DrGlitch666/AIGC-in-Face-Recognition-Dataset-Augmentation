"""检测 + 选脸 + 对齐，并缓存对齐结果（供自研模型评测复用）。

## 为什么需要单独这一层

* **E0** 走 insightface 的 ``FaceAnalysis.get()`` —— 检测/对齐/识别一体，够用
* **自研模型（E1/E4…）** 必须自己做检测与对齐，再喂给 PyTorch 模型
* 两边**必须用同一套对齐**（``align.py`` 里的 ``norm_crop2``），否则指标不可比

## 为什么要缓存

检测是整条链路最慢的一步（7701 张图约 5 分钟），而对齐结果**与模型无关**。
把对齐后的 112x112 人脸写成 PNG 缓存一次，之后换任何模型都只需跑前向（几十秒）。
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from aigcfr.data.align import ARCFACE_SIZE, align_by_kps
from aigcfr.eval.embed import pick_face


def build_aligned_cache(
    app,
    img_dir: str | Path,
    relpaths: list[str],
    cache_dir: str | Path,
    face_select: str = "center",
    size: int = ARCFACE_SIZE,
    verbose: bool = True,
    log_every: int = 1000,
) -> tuple[dict[str, str], dict[str, str], dict[str, object]]:
    """对 ``relpaths`` 做检测+对齐，PNG 缓存到 ``cache_dir``。

    返回 ``(mapping, failures, stats)``：

    * ``mapping``  相对路径 -> 缓存 PNG 的**绝对路径字符串**
    * ``failures`` 相对路径 -> 失败原因（``read_failed`` / ``no_face``）
    * ``stats``    检测质量摘要（续跑时会命中缓存，统计只反映本次新处理的图）
    """
    import cv2  # noqa: PLC0415

    img_dir = Path(img_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    mapping: dict[str, str] = {}
    failures: dict[str, str] = {}
    face_count_hist: dict[int, int] = {}
    det_scores: list[float] = []
    todo: list[str] = []

    for rel in relpaths:
        cached = cache_dir / (rel + ".png")
        if cached.exists():
            mapping[rel] = str(cached)
        else:
            todo.append(rel)

    if verbose:
        print(f"  [align-cache] 命中 {len(mapping)} 张，待处理 {len(todo)} 张 -> {cache_dir}")

    t0 = time.time()
    for i, rel in enumerate(todo, 1):
        img = cv2.imread(str(img_dir / rel))
        if img is None:
            failures[rel] = "read_failed"
            continue
        faces = app.get(img)
        if not faces:
            failures[rel] = "no_face"
            continue
        face = pick_face(faces, img.shape, face_select)
        aligned = align_by_kps(img, face.kps, size)
        out = cache_dir / (rel + ".png")
        out.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out), aligned):
            failures[rel] = "write_failed"
            continue
        mapping[rel] = str(out)
        face_count_hist[len(faces)] = face_count_hist.get(len(faces), 0) + 1
        det_scores.append(float(face.det_score))
        if verbose and i % log_every == 0:
            rate = i / max(time.time() - t0, 1e-6)
            print(f"  [align-cache] {i}/{len(todo)}  {rate:.1f} 张/秒")

    scores = np.array(det_scores, dtype=np.float64) if det_scores else np.array([])
    total = sum(face_count_hist.values())
    stats = {
        "newly_processed": len(todo),
        "from_cache": len(relpaths) - len(todo),
        "failed": len(failures),
        "seconds": round(time.time() - t0, 1),
        "face_count_hist": {str(k): v for k, v in sorted(face_count_hist.items())},
        "multi_face_ratio": round(sum(v for k, v in face_count_hist.items() if k > 1) / total, 4) if total else None,
        "det_score_mean": round(float(scores.mean()), 4) if scores.size else None,
    }
    if verbose and todo:
        print(f"  [align-cache] 完成 {len(todo)} 张，用时 {stats['seconds']}s")
    return mapping, failures, stats
