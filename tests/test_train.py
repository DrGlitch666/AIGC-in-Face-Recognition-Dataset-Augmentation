"""训练层单元测试：ArcFace 损失、iresnet 模型、训练数据集。

不需要 GPU、不需要数据集；纯张量运算（模型测试可跑在 CPU 上）。
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from aigcfr.train.loss import ArcFace
from aigcfr.train.model import build_model, count_parameters

torch.manual_seed(0)


# ---------------------------------------------------------------- ArcFace 损失

def test_arcface_logit_scale_and_shape():
    head = ArcFace(emb_dim=8, num_classes=5, margin=0.5, scale=64.0)
    emb = F.normalize(torch.randn(4, 8), dim=1)
    labels = torch.tensor([0, 1, 2, 3])
    loss, logits = head(emb, labels)
    assert logits.shape == (4, 5)
    assert loss.ndim == 0 and loss.item() > 0
    # 未加间隔时 logit 就是 s * cos(theta)，绝对值不会超过 s
    assert logits.abs().max().item() <= 64.0 * 1.5


def test_arcface_margin_zero_matches_normalized_softmax():
    """m=0 时 cos(theta+0) = cos(theta)，应退化成"归一化 softmax"。"""
    emb = F.normalize(torch.randn(6, 16), dim=1)
    labels = torch.tensor([0, 1, 2, 3, 4, 5])

    head0 = ArcFace(16, 6, margin=0.0, scale=64.0)
    head0.weight.data = F.normalize(torch.randn(6, 16), dim=1)
    loss0, _ = head0(emb, labels)

    cosine = F.linear(emb, F.normalize(head0.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
    reference = F.cross_entropy(cosine * 64.0, labels)
    assert loss0.item() == pytest.approx(reference.item(), abs=1e-5)


def test_arcface_margin_increases_loss_for_hard_cases():
    """同样的输入，加间隔后的 loss 必须更大 —— 间隔在"惩罚"正确类。"""
    emb = F.normalize(torch.randn(8, 16), dim=1)
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    w = F.normalize(torch.randn(8, 16), dim=1)

    head_m = ArcFace(16, 8, margin=0.5, scale=64.0)
    head_m.weight.data = w.clone()
    head_0 = ArcFace(16, 8, margin=0.0, scale=64.0)
    head_0.weight.data = w.clone()

    loss_m, _ = head_m(emb, labels)
    loss_0, _ = head_0(emb, labels)
    assert loss_m.item() > loss_0.item()


def test_arcface_loss_is_low_when_embedding_matches_its_center():
    """把 embedding 直接设成正确类的中心，loss 应该很小。"""
    head = ArcFace(16, 4, margin=0.5, scale=64.0)
    labels = torch.tensor([0, 1, 2, 3])
    emb = F.normalize(head.weight.detach().clone(), dim=1)   # 与类中心完全一致
    loss, _ = head(emb, labels)
    assert loss.item() < 1.0, f"完美对齐时 loss 应接近 0，实得 {loss.item()}"


def test_arcface_rejects_single_class():
    with pytest.raises(ValueError, match="至少需要 2 个类"):
        ArcFace(8, 1)


def test_arcface_extra_repr_is_informative():
    head = ArcFace(16, 7, margin=0.35, scale=32.0)
    text = head.extra_repr()
    assert "margin=0.35" in text and "scale=32.0" in text and "num_classes=7" in text


def test_arcface_stability_constants():
    """数值稳定用的 th / mm 必须与和角公式一致；这是 theta 跨过 pi 时的兜底。"""
    for margin in (0.0, 0.3, 0.5, 0.7):
        head = ArcFace(8, 4, margin=margin)
        assert head.cos_m == pytest.approx(math.cos(margin))
        assert head.sin_m == pytest.approx(math.sin(margin))
        assert head.th == pytest.approx(math.cos(math.pi - margin))
        assert head.mm == pytest.approx(math.sin(math.pi - margin) * margin)


# ---------------------------------------------------------------- 模型

def test_iresnet18_output_shape_and_normalized_bn():
    model = build_model("iresnet18", emb_dim=512, input_size=112).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 112, 112))
    assert out.shape == (2, 512)


def test_iresnet18_param_count_is_reasonable():
    """完整版约 6200 万参数（fc 占大头）；这是预期值，明显偏离说明结构被改了。"""
    model = build_model("iresnet18", emb_dim=512, input_size=112)
    params = count_parameters(model)
    assert 55e6 < params["total"] < 70e6, params
    # 末端 BN1d 的 weight 被冻结（requires_grad=False），所以只有它不计入可训练参数
    assert params["total"] - params["trainable"] == 512, params


def test_pooled_variant_is_much_smaller():
    full = count_parameters(build_model("iresnet18", pooled=False))["total"]
    small = count_parameters(build_model("iresnet18", pooled=True))["total"]
    assert small < full / 4, f"pooled 应小得多: {small} vs {full}"


def test_build_model_rejects_unknown_arch():
    with pytest.raises(ValueError, match="未知的 arch"):
        build_model("iresnet999")


def test_build_model_rejects_bad_input_size():
    with pytest.raises(ValueError, match="8 的倍数"):
        build_model("iresnet18", input_size=100)


def test_bn1d_features_layer_is_frozen():
    """末端 BN1d 的 weight 固定为 1 且不训练 —— 归一化交给 ArcFace 做。"""
    model = build_model("iresnet18")
    assert model.features.weight.requires_grad is False
    assert torch.allclose(model.features.weight, torch.ones_like(model.features.weight))


def test_model_is_differentiable_end_to_end():
    """一次完整的 前向 -> ArcFace -> 反向 必须能跑通并产生梯度。"""
    model = build_model("iresnet18", pooled=True)
    head = ArcFace(512, 5, margin=0.5, scale=64.0)
    x = torch.randn(4, 3, 112, 112)
    y = torch.tensor([0, 1, 2, 3])
    loss, _ = head(model(x), y)
    loss.backward()
    assert model.conv1.weight.grad is not None
    assert head.weight.grad is not None
    assert torch.isfinite(loss)
