"""1:1 人脸验证协议与指标（**纯 numpy**，不依赖 torch/insightface，方便单元测试）。

## 评分

``score = cosine(emb_a, emb_b)``。embedding 本身已由模型做过 L2 归一化，
所以余弦就等于内积；本模块仍然显式除以模长，**并且提供**
:func:`norm_stats` 用来发现"忘记归一化"这个经典错误。

## 报告两类指标（都必须写清楚，否则数字没有意义）

* ``accuracy@best``：在**测试集自身**上扫出最佳阈值后的准确率。
  这是 LFW 论文的惯例做法，所以文献数字普遍偏高 —— 用它来**对照文献**。
* ``TAR@FAR=k``：先用**异人对**的分数分布定出 FAR=k 的阈值，再看同人对的通过率。
  它不依赖测试集里同人/异人的比例，**更严格、也不依赖测试集调参** —— 用它来**做主结论**。

> 参考：docs/FRAMEWORK.md 契约 B（metrics.json）
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

DEFAULT_FAR_TARGETS: tuple[float, ...] = (1e-2, 1e-3)


@dataclass
class Scores:
    """一对样本的标签与相似度分数。"""

    labels: np.ndarray            # 1 = 同人，0 = 异人
    scores: np.ndarray            # 余弦相似度
    folds: np.ndarray             # 折号（1..10）
    missing: dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.labels.size)

    @property
    def num_same(self) -> int:
        return int((self.labels == 1).sum())

    @property
    def num_diff(self) -> int:
        return int((self.labels == 0).sum())


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """两个向量的余弦相似度（显式归一化，不假设输入已归一化）。"""
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def norm_stats(vectors: dict[str, np.ndarray]) -> dict[str, float]:
    """检查 embedding 的 L2 范数分布。

    模型输出的 ``normed_embedding`` 应该都约等于 1.0。如果明显偏离 1，
    说明拿到了未归一化的特征 —— 那样算出来的**不是余弦相似度**，
    向量长的图会天然占便宜，阈值失去意义（docs/SETUP.md 坑清单的常见错误）。
    """
    if not vectors:
        return {"count": 0}
    norms = np.array([float(np.linalg.norm(v)) for v in vectors.values()])
    return {
        "count": int(norms.size),
        "min": float(norms.min()),
        "max": float(norms.max()),
        "mean": float(norms.mean()),
        "is_normalized": bool(np.allclose(norms, 1.0, atol=1e-3)),
    }


def pair_scores(pairs, vectors: dict[str, np.ndarray]) -> Scores:
    """把 ``Pair`` 列表和 embedding 字典变成标签/分数数组。

    任一侧缺失（没检测到人脸等）的**对会被丢弃并计数**，不会伪造分数。
    """
    labels: list[int] = []
    scores: list[float] = []
    folds: list[int] = []
    missing: Counter = Counter()

    keep_a: list[np.ndarray] = []
    keep_b: list[np.ndarray] = []
    for p in pairs:
        a = vectors.get(p.relpath_a)
        b = vectors.get(p.relpath_b)
        if a is None:
            missing["side_a"] += 1
        if b is None:
            missing["side_b"] += 1
        if a is None or b is None:
            continue
        labels.append(p.label)
        folds.append(p.fold)
        keep_a.append(a)
        keep_b.append(b)

    if not labels:
        return Scores(np.array([], dtype=np.int64), np.array([], dtype=np.float64),
                      np.array([], dtype=np.int64), dict(missing))

    mat_a = np.stack(keep_a).astype(np.float64)
    mat_b = np.stack(keep_b).astype(np.float64)
    mat_a /= np.maximum(np.linalg.norm(mat_a, axis=1, keepdims=True), 1e-12)
    mat_b /= np.maximum(np.linalg.norm(mat_b, axis=1, keepdims=True), 1e-12)
    scores_arr = np.einsum("ij,ij->i", mat_a, mat_b)

    return Scores(
        labels=np.array(labels, dtype=np.int64),
        scores=scores_arr,
        folds=np.array(folds, dtype=np.int64),
        missing=dict(missing),
    )


def accuracy_at(labels: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    """给定阈值下的准确率。

    判定规则全项目统一为 **``score > threshold`` 判为同人**（严格大于）。
    选严格大于而不是大于等于，是为了在分数并列时不出现"阈值等于分数本身、
    却把整簇都接受"的怪现象（:func:`tar_at_far` 上踩过这个坑）。
    """
    if labels.size == 0:
        return float("nan")
    pred = scores > threshold
    return float((pred == (labels == 1)).mean())


def best_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """扫出使准确率最高的阈值，返回 ``(threshold, accuracy)``。

    ⚠️ 阈值是在**测试集自身**上选的 —— 这是 LFW 论文的惯例，会略微高估。
    所以它只用于**对照文献**；主结论请用 :func:`tar_at_far`。

    ⚠️ 实现要点：候选阈值取**相邻不同分数之间的中点**，而不是分数本身。
    因为分数会有并列值 —— 若直接把观测值当阈值，在"最大异人分"与"最小同人分"
    之间就没有候选点，两边会被判成同一类，准确率会莫名其妙地掉到 0.5。
    （这个坑真实踩过：30 同人 + 30 异人完全可分，却算出 accuracy = 0.5000。）
    """
    if labels.size == 0:
        return float("nan"), float("nan")

    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    y = labels[order] == 1
    n = s.size

    uniq = np.unique(s)
    eps = 1e-9
    if uniq.size == 1:
        candidates = np.array([uniq[0] - eps, uniq[0] + eps])
    else:
        mid = (uniq[:-1] + uniq[1:]) / 2.0
        candidates = np.concatenate([[uniq[0] - eps], mid, [uniq[-1] + eps]])

    # 语义：score > thr 判为同人 → 负预测数量 = #(score <= thr)
    cut = np.searchsorted(s, candidates, side="right")
    cum_pos = np.concatenate([[0], np.cumsum(y)])
    cum_neg = np.concatenate([[0], np.cumsum(~y)])

    # ⚠️ 这里极易写错：tp 是「预测为正 且 实际为正」的数量，
    #    也就是 cum_pos[n] - cum_pos[cut]，**不是** (n - cut) - (...)。
    #    后者算出来的是假正例数，会让 argmax 选中"把全部样本判为同人"的垃圾阈值，
    #    准确率固定显示 0.5。（这个坑真实踩过。）
    tp = cum_pos[n] - cum_pos[cut]
    tn = cum_neg[cut]
    acc = (tp + tn) / n

    best = int(np.argmax(acc))
    return float(candidates[best]), float(acc[best])


def tar_at_far(
    labels: np.ndarray,
    scores: np.ndarray,
    far: float,
) -> tuple[float, float, float]:
    """在目标 FAR 下的 TAR，返回 ``(tar, threshold, actual_far)``。

    做法：把异人分数**降序**排列，只允许前 ``floor(far * n_impostor)`` 个被接受，
    阈值取第 ``k+1`` 大的那个分数（配合严格 ``>`` 语义）。

    ⚠️ **不要用分位数**：分数并列时 ``np.quantile`` 会返回该值本身，
    而 ``>=`` 会把整簇并列分数一次全接受 —— 全常数分数时 FAR 直接变成 1.0。
    """
    same = scores[labels == 1]
    diff = scores[labels == 0]
    if same.size == 0 or diff.size == 0:
        return float("nan"), float("nan"), float("nan")

    diff_desc = np.sort(diff)[::-1]
    allowed = int(np.floor(far * diff.size))
    thr = float(diff_desc[allowed]) if allowed < diff.size else float("-inf")

    tar = float((same > thr).mean())
    actual_far = float((diff > thr).mean())
    return tar, thr, actual_far


def fold_metrics(labels: np.ndarray, scores: np.ndarray, folds: np.ndarray,
                 threshold: float) -> list[dict[str, float]]:
    """按折报告准确率（过滤后各折对数不均衡，所以要逐折记录）。"""
    out: list[dict[str, float]] = []
    for f in sorted(set(folds.tolist())):
        mask = folds == f
        if not mask.any():
            continue
        out.append({
            "fold": int(f),
            "n_pairs": int(mask.sum()),
            "n_same": int((labels[mask] == 1).sum()),
            "n_diff": int((labels[mask] == 0).sum()),
            "accuracy": round(accuracy_at(labels[mask], scores[mask], threshold), 6),
        })
    return out


def _dist(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {}
    return {
        "mean": round(float(values.mean()), 6),
        "std": round(float(values.std()), 6),
        "min": round(float(values.min()), 6),
        "p05": round(float(np.quantile(values, 0.05)), 6),
        "p50": round(float(np.quantile(values, 0.50)), 6),
        "p95": round(float(np.quantile(values, 0.95)), 6),
        "max": round(float(values.max()), 6),
    }


def summarize(
    sc: Scores,
    far_targets: tuple[float, ...] = DEFAULT_FAR_TARGETS,
) -> dict[str, object]:
    """把 :class:`Scores` 汇总成可直接写进 metrics.json 的字典。"""
    if len(sc) == 0:
        return {"error": "no usable pairs", "missing": sc.missing}

    thr_best, acc_best = best_threshold(sc.labels, sc.scores)

    tar_block: dict[str, dict[str, float]] = {}
    for far in far_targets:
        tar, thr, actual = tar_at_far(sc.labels, sc.scores, far)
        tar_block[f"{far:.0e}"] = {
            "tar": round(tar, 6),
            "threshold": round(thr, 6),
            "actual_far": round(actual, 6),
        }

    per_fold = fold_metrics(sc.labels, sc.scores, sc.folds, thr_best)
    fold_accs = np.array([f["accuracy"] for f in per_fold], dtype=np.float64)

    same = sc.scores[sc.labels == 1]
    diff = sc.scores[sc.labels == 0]

    return {
        "n_pairs_used": len(sc),
        "n_pairs_missing": sum(sc.missing.values()),
        "n_same": sc.num_same,
        "n_diff": sc.num_diff,
        # —— 与文献可比（测试集自选阈值）——
        "metric": "accuracy",
        "value": round(acc_best, 6),
        "threshold": round(thr_best, 6),
        # —— 更严格：不用测试集调参 ——
        "tar_at_far": tar_block,
        "same_score": _dist(same),
        "diff_score": _dist(diff),
        "per_fold": per_fold,
        "accuracy_per_fold_mean": round(float(fold_accs.mean()), 6),
        "accuracy_per_fold_std": round(float(fold_accs.std()), 6),
    }
