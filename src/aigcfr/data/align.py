"""人脸对齐：5 点关键点 → 112x112 的 ArcFace 标准模板。

## ⚠️ 对齐是整条链路里最容易出错、后果最严重的一步

对齐错了，后面所有指标都不可信（`docs/FRAMEWORK.md` §3.2 第 2 步写得很明确）。
所以本项目有一条硬规矩：

> **训练与评测必须使用同一套对齐实现。**

## 为什么用 ``norm_crop2`` 而不是 ``norm_crop``

insightface 里有两个模板函数：

* ``face_align.norm_crop(img, kps)``  —— 经典 ArcFace 模板（1998 年参考点）
* ``face_align.norm_crop2(img, kps)`` —— insightface 的 **识别模型实际使用**的那个

评测时我们调的是 ``FaceAnalysis.get()``，它内部走的是 ``norm_crop2``。
如果训练用 ``norm_crop``，两边裁出来的脸会有细微的平移/缩放差异，
模型学到的和评测看到的不是同一个分布 —— 这种错误很难发现，但会稳定地拉低指标。

**所以本模块只用 ``norm_crop2``。不要改成 ``norm_crop``。**
"""

from __future__ import annotations

import numpy as np

from insightface.utils import face_align

ARCFACE_SIZE = 112


def align_by_kps(img: np.ndarray, kps: np.ndarray, size: int = ARCFACE_SIZE) -> np.ndarray:
    """按 5 点关键点做相似变换，裁出 ``size x size`` 的对齐人脸。

    ``kps`` 是 5x2 的关键点（检测模型给出）。
    """
    if kps is None or len(kps) != 5:
        raise ValueError(f"需要 5 个关键点，收到 {None if kps is None else len(kps)} 个")
    aligned, _ = face_align.norm_crop2(img, landmark=np.asarray(kps, dtype=np.float32), image_size=size)
    return aligned


def align_stats(aligned: np.ndarray) -> dict[str, float]:
    """对齐结果的自检指标（写进 manifest / 日志，便于发现"裁歪了"）。"""
    import cv2  # noqa: PLC0415

    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    return {
        "size": int(aligned.shape[0]),
        "mean": float(gray.mean()),
        "std": float(gray.std()),
        # 亮度太低或方差太小通常意味着裁到了背景/纯色区域
        "is_suspicious": bool(gray.mean() < 40 or gray.std() < 15),
    }
