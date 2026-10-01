"""训练数据集：读 manifest.jsonl（契约 A）。

## 颜色通道约定（容易出错，写在这里）

本项目的**训练与自研模型评测统一使用 RGB**，归一化到 ``[-1, 1]``：

    像素 -> /255 -> (x - 0.5) / 0.5

注意与 E0 的差异：E0 走 insightface（ONNX），它**内部约定是 BGR**，
所以那条路径直接吃 cv2 读出来的 BGR。两条路径各自自洽即可，
但**自研模型训练与评测必须都用 RGB** —— 混用会让指标莫名下降。

## 增广

人脸识别的标准增广很轻：**随机水平翻转**为主（ArcFace 论文只用了翻转）。
颜色抖动默认关闭；合成图实验里如果发现两臂差异被增广掩盖，优先调这里而不是改模型。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ManifestDataset(Dataset):
    """按 manifest 读取对齐好的人脸，返回 ``(tensor[3,H,W], label)``。"""

    def __init__(
        self,
        manifest_path: str | Path | list[str | Path],
        data_root: str | Path,
        split: str = "train",
        image_size: int = 112,
        augment: bool = True,
        color_jitter: float = 0.0,
        sources: tuple[str, ...] | None = None,
    ):
        """``manifest_path`` 可以是单个路径，也可以是**多个**（E4 要把真实+合成拼起来）。"""
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.augment = augment
        self.color_jitter = color_jitter

        paths = [manifest_path] if isinstance(manifest_path, (str, Path)) else list(manifest_path)
        records: list[dict] = []
        for one in paths:
            path = Path(one)
            if not path.exists():
                raise FileNotFoundError(f"找不到 manifest: {path}（先跑 scripts/build_dataset.py）")
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if rec.get("split") != split or rec.get("status") != "accepted":
                        continue
                    if sources is not None and rec.get("source") not in sources:
                        continue
                    records.append(rec)
        if not records:
            raise ValueError(f"这些 manifest 里没有 split={split!r} / status='accepted' 的样本: {paths}")

        # 类别索引按身份名排序 —— 保证两台机器、多次运行的 label 顺序一致
        identities = sorted({r["identity_id"] for r in records})
        self.classes = identities
        self.class_to_idx = {name: i for i, name in enumerate(identities)}

        self.records = records
        self.labels = [self.class_to_idx[r["identity_id"]] for r in records]

    def __len__(self) -> int:
        return len(self.records)

    def class_counts(self) -> dict[str, int]:
        counts = {name: 0 for name in self.classes}
        for r in self.records:
            counts[r["identity_id"]] += 1
        return counts

    def __getitem__(self, index: int):
        import cv2  # noqa: PLC0415

        rec = self.records[index]
        img = cv2.imread(str(self.data_root / rec["path"]))
        if img is None:
            raise RuntimeError(f"读不到图片: {self.data_root / rec['path']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      # ← 统一 RGB

        if self.augment:
            if np.random.rand() < 0.5:                   # 水平翻转（人脸识别标配）
                img = img[:, ::-1]
            if self.color_jitter > 0:
                factor = 1.0 + np.random.uniform(-self.color_jitter, self.color_jitter)
                img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5                          # -> [-1, 1]
        tensor = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        return tensor, self.labels[index]
