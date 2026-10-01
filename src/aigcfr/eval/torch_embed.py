"""用**自研 PyTorch 模型**提特征（E1/E4… 的评测路径）。

## 预处理必须与训练完全一致

``src/aigcfr/train/dataset.py`` 里的约定：

    PNG(已对齐 112x112) -> cv2 读入(BGR) -> RGB -> /255 -> (x-0.5)/0.5 -> [3,112,112]

**顺序、色彩空间、归一化区间全部要一致。** 这是最容易出错、也最难发现的地方：
差异不会报错，只会让指标低几个点。所以本模块直接复用同一套常量。
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from aigcfr.train.model import build_model

# 与 train/dataset.py 保持一致（不要各写一份）
PIXEL_MEAN = 0.5
PIXEL_STD = 0.5


def load_checkpoint(ckpt_path: str | Path, device: torch.device | str = "cpu"):
    """加载训练好的 checkpoint，返回 ``(model, classes, meta)``（model 已 eval 且移到 device）。"""
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model = build_model(
        state.get("arch", "iresnet18"),
        emb_dim=int(state.get("emb_dim", 512)),
        input_size=int(state.get("image_size", 112)),
        pooled=bool(state.get("pooled", False)),
    )
    model.load_state_dict(state["model"])
    model.eval()
    model.to(device)

    meta = {k: v for k, v in state.items() if k not in ("model", "head")}
    return model, list(state.get("classes", [])), meta


def _preprocess(path: str | Path) -> torch.Tensor | None:
    """读一张已对齐的人脸 PNG -> ``[3,H,W]`` 张量（与训练一致）。"""
    import cv2  # noqa: PLC0415

    img = cv2.imread(str(path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - PIXEL_MEAN) / PIXEL_STD
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))


@torch.no_grad()
def extract_aligned(
    model,
    aligned_paths: dict[str, str],
    batch_size: int = 128,
    device: torch.device | str = "cpu",
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """对已对齐的人脸跑前向，返回 ``{相对路径: (emb_dim,) float32}``。

    embedding **不做 L2 归一化** —— 归一化在各处（评分、筛选）按需做，
    这样原始尺度信息不丢（例如将来要看 embedding 的模长分布）。
    """
    device = torch.device(device)
    rels = list(aligned_paths.keys())
    vectors: dict[str, np.ndarray] = {}
    t0 = time.time()
    batch: list[torch.Tensor] = []
    batch_rels: list[str] = []

    def flush() -> None:
        if not batch:
            return
        x = torch.stack(batch).to(device)
        out = model(x).float().cpu().numpy()
        for rel, vec in zip(batch_rels, out, strict=True):
            vectors[rel] = vec.astype(np.float32)
        batch.clear()
        batch_rels.clear()

    for rel in rels:
        tensor = _preprocess(aligned_paths[rel])
        if tensor is None:
            continue
        batch.append(tensor)
        batch_rels.append(rel)
        if len(batch) >= batch_size:
            flush()
    flush()

    if verbose:
        dt = time.time() - t0
        print(f"  [torch] {len(vectors)}/{len(rels)} 张，用时 {dt:.1f}s"
              f"（{len(vectors) / max(dt, 1e-6):.0f} 张/秒）")
    return vectors
