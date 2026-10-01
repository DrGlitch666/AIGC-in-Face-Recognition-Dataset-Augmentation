"""1:1 验证协议与指标的单元测试（纯 numpy，不需要数据集、不需要 GPU）。

⚠️ 其中 :func:`test_best_threshold_handles_outlier_same_pair` 是**回归测试**：
   它复现了 2026-10-01 冒烟测试里真实出现过的 bug —— `tp` 写错导致
   "完全可分的数据算出 accuracy = 0.5"，阈值退化到最低分。
"""

from __future__ import annotations

import numpy as np
import pytest

from aigcfr.data.lfw import Pair
from aigcfr.eval.verify import (
    accuracy_at,
    best_threshold,
    cosine,
    fold_metrics,
    norm_stats,
    pair_scores,
    summarize,
    tar_at_far,
)


# ---------------------------------------------------------------- 基础

def test_cosine_range_and_identity():
    a = np.array([1.0, 0.0, 0.0])
    assert cosine(a, a) == pytest.approx(1.0)
    assert cosine(a, np.array([-1.0, 0.0, 0.0])) == pytest.approx(-1.0)
    assert cosine(a, np.array([0.0, 1.0, 0.0])) == pytest.approx(0.0)


def test_cosine_ignores_length():
    """余弦不受向量长度影响 —— 这正是它比内积适合做判据的原因。"""
    a = np.array([1.0, 2.0, 3.0])
    assert cosine(a, a * 7.5) == pytest.approx(1.0)


def test_cosine_zero_vector_is_safe():
    assert cosine(np.zeros(3), np.ones(3)) == 0.0


def test_norm_stats_detects_unnormalized():
    good = {f"a{i}": np.array([1.0, 0.0]) for i in range(3)}
    bad = {f"a{i}": np.array([3.0, 4.0]) for i in range(3)}   # 模长 5
    assert norm_stats(good)["is_normalized"] is True
    assert norm_stats(bad)["is_normalized"] is False
    assert norm_stats(bad)["max"] == pytest.approx(5.0)
    assert norm_stats({}) == {"count": 0}


# ---------------------------------------------------------------- 配对打分

def _pair(fold, kind, ia, ib):
    if kind == "same":
        return Pair(fold, "same", ia, 1, ia, 2)
    return Pair(fold, "diff", ia, 1, ib, 1)


def test_pair_scores_matches_cosine():
    v1 = np.array([1.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0], dtype=np.float32)
    vectors = {
        "A/A_0001.jpg": v1, "A/A_0002.jpg": v1.copy(),
        "B/B_0001.jpg": v2, "B/B_0002.jpg": v2.copy(),
    }
    pairs = [_pair(1, "same", "A", "A"), _pair(1, "diff", "A", "B")]
    sc = pair_scores(pairs, vectors)

    assert len(sc) == 2
    assert sc.num_same == 1 and sc.num_diff == 1
    assert sc.scores[0] == pytest.approx(1.0)
    assert sc.scores[1] == pytest.approx(0.0)
    assert list(sc.labels) == [1, 0]
    assert sc.missing == {}


def test_pair_scores_drops_pairs_with_missing_side():
    vectors = {"A/A_0001.jpg": np.ones(2, dtype=np.float32)}
    pairs = [_pair(1, "same", "A", "A"), _pair(1, "diff", "A", "B")]
    sc = pair_scores(pairs, vectors)
    assert len(sc) == 0
    assert sc.missing == {"side_b": 2}


def test_pair_scores_normalizes_input():
    """即使传进来的向量没归一化，算出来的仍是余弦。"""
    vectors = {
        "A/A_0001.jpg": np.array([10.0, 0.0], dtype=np.float32),
        "A/A_0002.jpg": np.array([0.0, 5.0], dtype=np.float32),
    }
    sc = pair_scores([_pair(1, "same", "A", "A")], vectors)
    assert sc.scores[0] == pytest.approx(0.0)


# ---------------------------------------------------------------- 阈值与准确率

def test_accuracy_at_basic():
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    assert accuracy_at(labels, scores, 0.5) == pytest.approx(1.0)
    assert accuracy_at(labels, scores, 0.0) == pytest.approx(0.5)
    assert accuracy_at(labels, scores, 0.95) == pytest.approx(0.5)


def test_best_threshold_perfectly_separated():
    labels = np.array([1] * 30 + [0] * 30)
    scores = np.concatenate([np.full(30, 0.7), np.full(30, 0.1)])
    thr, acc = best_threshold(labels, scores)
    assert acc == pytest.approx(1.0)
    assert 0.1 < thr < 0.7


def test_best_threshold_handles_outlier_same_pair():
    """★ 回归测试：复现冒烟测试里真实出现过的 bug。

    真实分数分布：30 个同人分（其中 1 对极低 -0.0054）+ 30 个异人分（最高 0.1424）。
    最优阈值应落在 0.1424 与 0.54 之间，准确率 = 59/60。
    修复前这里会返回最低分作阈值、accuracy 固定 0.5。
    """
    same = np.array([-0.0054] + [0.54 + 0.01 * i for i in range(29)])
    diff = np.array([-0.1843 + (0.1424 + 0.1843) / 29 * i for i in range(30)])
    labels = np.concatenate([np.ones(30, dtype=np.int64), np.zeros(30, dtype=np.int64)])
    scores = np.concatenate([same, diff])

    thr, acc = best_threshold(labels, scores)

    assert acc == pytest.approx(59 / 60, abs=1e-9), f"应得 59/60，实得 {acc}"
    assert 0.1424 < thr < 0.54, f"阈值应落在两簇之间，实得 {thr}"
    assert accuracy_at(labels, scores, thr) == pytest.approx(acc, abs=1e-9)


def test_best_threshold_all_same_labels_is_not_crash():
    labels = np.ones(10, dtype=np.int64)
    scores = np.linspace(0.1, 0.9, 10)
    thr, acc = best_threshold(labels, scores)
    assert acc == pytest.approx(1.0)
    assert thr < 0.1


def test_best_threshold_empty():
    thr, acc = best_threshold(np.array([]), np.array([]))
    assert np.isnan(thr) and np.isnan(acc)


# ---------------------------------------------------------------- TAR@FAR

def test_tar_at_far_perfect_separation():
    labels = np.array([1] * 100 + [0] * 100)
    scores = np.concatenate([np.full(100, 0.8), np.full(100, 0.2)])
    tar, thr, actual_far = tar_at_far(labels, scores, 1e-3)
    assert tar == pytest.approx(1.0)
    assert 0.2 <= thr <= 0.8
    assert actual_far == pytest.approx(0.0)


def test_tar_at_far_threshold_comes_from_impostors_only():
    """阈值只能由异人分布决定（因此不依赖测试集里同人/异人的比例）。"""
    labels = np.array([1] * 10 + [0] * 1000)
    rng = np.random.default_rng(0)
    scores = np.concatenate([rng.normal(0.6, 0.1, 10), rng.normal(0.0, 0.1, 1000)])
    tar, thr, actual_far = tar_at_far(labels, scores, 1e-2)
    assert actual_far == pytest.approx(0.01, abs=0.005)
    assert 0.0 < tar <= 1.0


def test_tar_at_far_missing_class():
    tar, thr, far = tar_at_far(np.ones(5, dtype=np.int64), np.ones(5), 1e-3)
    assert np.isnan(tar) and np.isnan(thr) and np.isnan(far)


# ---------------------------------------------------------------- 逐折与汇总

def test_fold_metrics():
    labels = np.array([1, 0, 1, 0])
    scores = np.array([0.9, 0.1, 0.8, 0.2])
    folds = np.array([1, 1, 2, 2])
    out = fold_metrics(labels, scores, folds, threshold=0.5)
    assert [f["fold"] for f in out] == [1, 2]
    assert all(f["accuracy"] == pytest.approx(1.0) for f in out)
    assert out[0]["n_pairs"] == 2 and out[0]["n_same"] == 1


def test_summarize_has_contract_fields():
    labels = np.array([1] * 20 + [0] * 20)
    scores = np.concatenate([np.linspace(0.6, 0.9, 20), np.linspace(-0.1, 0.2, 20)])
    folds = np.array([1 + i % 10 for i in range(40)])
    sc = pair_scores(
        [Pair(int(folds[i]), "same" if labels[i] else "diff", f"I{i}", 1,
              f"I{i}" if labels[i] else f"J{i}", 2 if labels[i] else 1) for i in range(40)],
        {**{f"I{i}/I{i}_0001.jpg": np.eye(2)[0].astype(np.float32) for i in range(40)},
         **{f"J{i}/J{i}_0001.jpg": np.eye(2)[1].astype(np.float32) for i in range(40)},
         **{f"I{i}/I{i}_0002.jpg": np.eye(2)[0].astype(np.float32) for i in range(40)}},
    )
    m = summarize(sc)
    for key in ("n_pairs_used", "n_same", "n_diff", "metric", "value", "threshold",
                "tar_at_far", "same_score", "diff_score", "per_fold",
                "accuracy_per_fold_mean", "accuracy_per_fold_std"):
        assert key in m, f"metrics.json 契约里缺字段: {key}"
    assert m["metric"] == "accuracy"
    assert "1e-03" in m["tar_at_far"] and "1e-02" in m["tar_at_far"]


def test_summarize_empty_scores():
    sc = pair_scores([], {})
    m = summarize(sc)
    assert "error" in m
