"""ArcFace 损失（Additive Angular Margin Loss, CVPR 2019）。

## 一句话原理

把人脸分类的 logit 从「内积」改成「归一化后的余弦」，并在**正确类的角度上加一个间隔 m**：

    logit_j = s * cos(theta_j)            (j != y)
    logit_y = s * cos(theta_y + m)        (正确类：角度上多付 m)
    损失 = 交叉熵(logits, y)

含义：正确类必须在角度空间里比其它类**接近至少 m 弧度**才算答对 ——
逼着同类更紧、异类更远。`m` 控制强制分离的程度，`s` 控制 softmax 的陡峭程度。

## 超参（论文默认值，也是本项目的默认值）

    m = 0.5     论文扫过 0.1~0.7，0.5 最好；m=0 会退化成"归一化 softmax"
    s = 64      论文扫过 16~128，64 最好；太小则 softmax 太平、梯度消失

小数据集上如果 loss 迟迟不降，可以把 m 降到 0.3 试试（工程调整，非论文做法）。

## 一个必须小心的数值问题

``cos(theta + m)`` 在 theta 接近 pi 时用和角公式展开会失真
（``cos(theta+m) = cos theta * cos m - sin theta * sin m``，其中 ``sin theta`` 可能是 0）。
所以当 ``cos theta`` 落到 ``cos(pi - m)`` 以下时，改用线性近似 ``cos theta - m * sin(pi - m)``
—— 这就是下面 ``self.th`` / ``self.mm`` 的用途。**不要让 theta 跨过 pi。**
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class ArcFace(nn.Module):
    """ArcFace 头：``num_classes x emb_dim`` 的可学习类中心 + 角度间隔。"""

    def __init__(self, emb_dim: int, num_classes: int, margin: float = 0.5, scale: float = 64.0):
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"至少需要 2 个类，收到 {num_classes}")
        self.emb_dim = emb_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale

        self.weight = nn.Parameter(torch.empty(num_classes, emb_dim))
        nn.init.xavier_normal_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)               # 数值稳定边界
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, emb: torch.Tensor, label: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 ``(loss, logits)``；logits 可用于算训练准确率。"""
        cosine = F.linear(F.normalize(emb), F.normalize(self.weight))
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)     # 防止 sqrt 负数
        sine = torch.sqrt(1.0 - cosine.pow(2))
        phi = cosine * self.cos_m - sine * self.sin_m       # cos(theta + m)
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = F.one_hot(label, self.num_classes).to(cosine.dtype)
        logits = (one_hot * phi + (1.0 - one_hot) * cosine) * self.scale
        return F.cross_entropy(logits, label), logits

    def extra_repr(self) -> str:
        return (f"emb_dim={self.emb_dim}, num_classes={self.num_classes}, "
                f"margin={self.margin}, scale={self.scale}")
