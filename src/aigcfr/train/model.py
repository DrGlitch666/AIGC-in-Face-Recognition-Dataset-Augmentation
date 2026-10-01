"""iresnet18：ArcFace 使用的改进版 ResNet 主干。

    输入 (B, 3, 112, 112)  ->  输出 (B, 512) 的 embedding

## 与标准 ResNet 的区别（都来自 ArcFace 论文 / insightface 实现）

* **pre-activation**：BN 放在 conv **之前**（``conv(bn(x))``），而不是 conv 之后
* **PReLU** 而不是 ReLU
* 末端用 ``BatchNorm1d`` 而不是 L2 归一化 —— 归一化交给损失函数（ArcFace 会做），
  这里再做一次会重复且削弱梯度

## 尺寸推导（112x112 输入）

    112 --conv1(s1)--> 112 --layer1(s1)--> 112 --layer2(s2)--> 56
        --layer3(s2)--> 28 --layer4(s2)--> 14 --> flatten(512*14*14=100352) --fc--> 512

⚠️ 这个 fc 层很大（约 5100 万参数），整个模型约 6200 万参数。
   在玩具数据集（96 身份 / 3590 张图）上会过拟合 —— 这是**预期且可接受**的：
   本项目比较的是"真实 vs 真实+合成"的**相对**差异，两臂用完全相同的模型与超参。
   如果 loss 曲线明显过拟合，可以在配置里加大 dropout 或换更小的 fc（见下）。
"""

from __future__ import annotations

import torch
from torch import nn


class IBasicBlock(nn.Module):
    """iresnet 的基础残差块（pre-activation）。"""

    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-5)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-5)
        self.prelu = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-5)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(self.bn1(x))
        out = self.prelu(self.bn2(out))
        out = self.bn3(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return out + identity


class IResNet(nn.Module):
    """``layers=(2,2,2,2)`` 就是 iresnet18。"""

    def __init__(
        self,
        layers: tuple[int, ...] = (2, 2, 2, 2),
        emb_dim: int = 512,
        input_size: int = 112,
        dropout: float = 0.0,
        pooled: bool = False,
    ):
        super().__init__()
        if input_size % 8 != 0:
            raise ValueError(f"input_size 必须是 8 的倍数（三次下采样），收到 {input_size}")
        self.input_size = input_size
        self.emb_dim = emb_dim
        self.pooled = pooled

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5)
        self.prelu = nn.PReLU(64)
        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.bn2 = nn.BatchNorm2d(512, eps=1e-5)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if pooled:
            # 小 fc 版本：参数量掉到 ~1100 万，适合数据量很小的场景
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(512, emb_dim)
        else:
            self.pool = None
            feat = input_size // 8
            self.fc = nn.Linear(512 * feat * feat, emb_dim)

        self.features = nn.BatchNorm1d(emb_dim, eps=1e-5)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes, eps=1e-5),
            )
        layers = [IBasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        layers += [IBasicBlock(planes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.prelu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        # ⚠️ 池化后必须再 flatten：(B,512,1,1) 直接喂给 fc 会报
        #    "mat1 and mat2 shapes cannot be multiplied (Bx1 and 512x512)"。
        x = self.pool(x) if self.pool is not None else x
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return self.features(x)          # BN1d，不归一化（归一化由 ArcFace 做）


def build_model(arch: str = "iresnet18", emb_dim: int = 512, input_size: int = 112,
                dropout: float = 0.0, pooled: bool = False) -> IResNet:
    """按配置里的 ``model.arch`` 建模型。"""
    table = {
        "iresnet18": (2, 2, 2, 2),
        "iresnet34": (3, 4, 6, 3),
        "iresnet50": (3, 4, 14, 3),
        "iresnet100": (3, 13, 30, 3),
    }
    if arch not in table:
        raise ValueError(f"未知的 arch: {arch}（可选 {sorted(table)}）")
    return IResNet(layers=table[arch], emb_dim=emb_dim, input_size=input_size,
                   dropout=dropout, pooled=pooled)


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
