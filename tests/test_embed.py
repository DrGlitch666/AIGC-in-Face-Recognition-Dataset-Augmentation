"""``aigcfr.eval.embed`` 的单元测试（不需要数据集；需要 torch/insightface 能 import）。

★ 核心是 :func:`test_pick_face_center_beats_larger_box` —— 它是 2026-10-01 那个
   "LFW 准确率少 2.6 个百分点" bug 的回归测试。
"""

from __future__ import annotations

import numpy as np
import pytest

from aigcfr.eval.embed import pick_face


class _Face:
    """最小化的 face 替身（只需要 bbox 与 det_score）。"""

    def __init__(self, bbox, score):
        self.bbox = np.array(bbox, dtype=np.float64)
        self.det_score = float(score)


IMG = (250, 250, 3)


def test_pick_face_center_beats_larger_box():
    """★ 回归测试：背景里的大误检框必须被"居中"策略排掉。

    真实场景：LFW 250x250 单人居中图，检测器偶尔在边上产生一个**更大**的框。
    旧代码取"面积最大" → 裁到背景 → 同人相似度 0.67 掉到 -0.03。
    """
    big_edge = _Face([0, 0, 210, 210], 0.62)          # 面积 44100，中心在 (105,105)，偏左上
    small_center = _Face([100, 90, 150, 165], 0.85)   # 面积 3750，中心在 (125,127.5)，很居中

    assert pick_face([big_edge, small_center], IMG, "area") is big_edge       # 旧行为
    assert pick_face([big_edge, small_center], IMG, "center") is small_center  # 修复后
    assert pick_face([big_edge, small_center], IMG, "score") is small_center


def test_pick_face_single_face_is_returned_by_all_strategies():
    only = _Face([10, 10, 60, 70], 0.9)
    for strategy in ("center", "score", "area"):
        assert pick_face([only], IMG, strategy) is only


def test_pick_face_empty_returns_none():
    for strategy in ("center", "score", "area"):
        assert pick_face([], IMG, strategy) is None


def test_pick_face_center_uses_image_center_not_origin():
    """居中判据要用图像中心：两张同样大的框，靠近中心的胜出。"""
    top_left = _Face([0, 0, 50, 50], 0.9)
    middle = _Face([100, 100, 150, 150], 0.8)
    assert pick_face([top_left, middle], IMG, "center") is middle
    # 反过来：图像中心在 (125,125)，middle 的中心是 (125,125) → 距离 0
    corner = _Face([200, 200, 250, 250], 0.9)
    assert pick_face([corner, middle], IMG, "center") is middle


def test_pick_face_unknown_strategy_raises():
    with pytest.raises(ValueError, match="未知的选脸策略"):
        pick_face([_Face([0, 0, 10, 10], 0.9)], IMG, "biggest")


def test_default_strategy_is_center():
    """默认参数必须是 center —— 防止有人"顺手"改回 area。"""
    import inspect

    sig = inspect.signature(pick_face)
    assert sig.parameters["strategy"].default == "center"
