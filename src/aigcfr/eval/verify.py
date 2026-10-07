"""1:1 人脸验证协议与指标（**纯 numpy**，不依赖 torch/insightface，方便单元测试）。

## 评分

``score = cosine(emb_a, emb_b)``。embedding 本身已由模型做过 L2 归一化，
所以余弦就等于内积；本模块仍然显式除以模长，**并且提供**
:func:`norm_stats` 用来发现"忘记归一化"这个经典错误。

## 报告两类指标（都必须写清楚，否则数字没有意义）

* ``accuracy@best``：在**测试集自身**上扫出最佳阈值后的描述性准确率。
  这不是在其他折校准阈值、再在留出折验证的交叉验证成绩，不能直接等同文献数字。
* ``TAR@FAR=k``：先用**异人对**的分数分布定出 FAR=k 的阈值，再看同人对的通过率。
  它不依赖测试集里同人/异人的比例，但阈值仍由当前评测集的异人分数校准，
  并不等同于独立校准集上的部署验证。

正的目标 FAR 若小于 ``1 / n_impostor``，按本项目约定返回 NaN，
避免把最大异人分数处的经验零误接受率冒充目标 FAR 的有分辨率估计。
这是最小分辨率保护，不代表达到这一最低样本数就有充分统计精度。

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


def l2_normalize(matrix: np.ndarray, axis: int = -1) -> np.ndarray:
    """L2 归一化（余弦相似度的前提）。

    ⚠️ **任何比较 embedding 的代码都必须先过这一步。**

    自研模型的输出**没有归一化**：末端是 ``BatchNorm1d``（每维单位方差），
    所以 512 维向量的模长约在 ``sqrt(512) ≈ 22.6`` 附近浮动（实测 9.6~32）。

    如果直接对这些向量算内积，结果是"**谁的向量更长谁得分更高**"，
    与两脸是否相似无关 —— 判据完全失效。余弦相似度按定义与长度无关，
    所以先归一化、再算内积，两者等价。
    """
    arr = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=axis, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


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

    # 必须先归一化：自研模型的 embedding 未归一化（模长 ~9.6~32），
    # 直接算内积会变成"谁的向量长谁赢"。
    mat_a = l2_normalize(np.stack(keep_a))
    mat_b = l2_normalize(np.stack(keep_b))
    scores_arr = np.einsum("ij,ij->i", mat_a, mat_b)

    return Scores(
        labels=np.array(labels, dtype=np.int64),
        scores=scores_arr,
        folds=np.array(folds, dtype=np.int64),
        missing=dict(missing),
    )


def _validated_labels_scores(labels, scores) -> tuple[np.ndarray, np.ndarray]:
    """Reject malformed inputs, including the usual swapped labels/scores call.

    Two arrays that both happen to be binary cannot be distinguished by values;
    callers must still obey the public ``(labels, scores, ...)`` signature.
    """
    y = np.asarray(labels)
    s = np.asarray(scores)
    if y.ndim != 1 or s.ndim != 1:
        raise ValueError("labels and scores must be one-dimensional arrays")
    if y.size != s.size:
        raise ValueError("labels and scores must have the same length")
    if y.dtype.kind not in "biuf" or not np.all(np.isfinite(y)):
        raise ValueError("labels must contain only finite binary values 0 or 1")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("labels must be binary values 0 or 1; check argument order (labels, scores)")
    if s.dtype.kind not in "biuf" or not np.all(np.isfinite(s)):
        raise ValueError("scores must contain only finite real numbers")
    return y, s.astype(np.float64, copy=False)


def _validated_scalar(value, name: str, allow_infinite: bool = False) -> float:
    arr = np.asarray(value)
    if arr.ndim != 0 or arr.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real scalar")
    number = float(arr)
    if np.isnan(number) or (not allow_infinite and not np.isfinite(number)):
        raise ValueError(f"{name} must be {'non-NaN' if allow_infinite else 'finite'}")
    return number


def accuracy_at(labels: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    """给定阈值下的准确率。

    判定规则全项目统一为 **``score > threshold`` 判为同人**（严格大于）。
    选严格大于而不是大于等于，是为了在分数并列时不出现"阈值等于分数本身、
    却把整簇都接受"的怪现象（:func:`tar_at_far` 上踩过这个坑）。
    """
    labels, scores = _validated_labels_scores(labels, scores)
    threshold = _validated_scalar(threshold, "threshold", allow_infinite=True)
    if labels.size == 0:
        return float("nan")
    pred = scores > threshold
    return float((pred == (labels == 1)).mean())


def best_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """扫出使准确率最高的阈值，返回 ``(threshold, accuracy)``。

    ⚠️ 阈值是在**测试集自身**上选的，结果偏乐观。
    它是描述性指标，不能直接当作独立校准或官方交叉验证成绩。

    ⚠️ 实现要点：候选阈值取**相邻不同分数之间的中点**，而不是分数本身。
    并列分数必须作为一整组切分，不能逐个拆开。对于相邻浮点数，若中点舍入为
    较高分数，则退回较低分数；严格 ``>`` 语义下仍能正确分开两组。
    """
    labels, scores = _validated_labels_scores(labels, scores)
    if labels.size == 0:
        return float("nan"), float("nan")

    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    y = labels[order] == 1
    n = s.size

    uniq = np.unique(s)
    eps = 1e-9
    below = uniq[0] - eps
    if below >= uniq[0]:
        with np.errstate(over="ignore"):
            below = np.nextafter(uniq[0], -np.inf)
    above = uniq[-1] + eps
    if uniq.size == 1:
        candidates = np.array([below, above])
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            mid = (uniq[:-1] + uniq[1:]) / 2.0
        bad = ~np.isfinite(mid)
        mid[bad] = uniq[:-1][bad] / 2.0 + uniq[1:][bad] / 2.0
        mid = np.where(mid >= uniq[1:], uniq[:-1], mid)
        candidates = np.concatenate([[below], mid, [above]])

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

    FAR 必须在 [0, 1] 内。缺少任一类别，或正目标 FAR 的样本分辨率不足
    （``far * n_impostor < 1``），返回三个 NaN。FAR=0 表示最大异人分数处
    的经验零误接受率边界，不能解释为总体 FAR 为零；FAR=1 接受所有有限分数。
    """
    labels, scores = _validated_labels_scores(labels, scores)
    far = _validated_scalar(far, "far")
    if not 0.0 <= far <= 1.0:
        raise ValueError("far must be between 0 and 1 inclusive")
    same = scores[labels == 1]
    diff = scores[labels == 0]
    if same.size == 0 or diff.size == 0:
        return float("nan"), float("nan"), float("nan")
    if 0.0 < far < 1.0 and far * diff.size < 1.0:
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

    tar_block: dict[str, dict[str, object]] = {}
    n_impostor = sc.num_diff
    for far in far_targets:
        tar, thr, actual = tar_at_far(sc.labels, sc.scores, far)
        if sc.num_same == 0 or n_impostor == 0:
            status = "missing_class"
        elif 0.0 < far < 1.0 and far * n_impostor < 1.0:
            status = "insufficient_far_resolution"
        else:
            status = "ok"
        tar_block[f"{far:.0e}"] = {
            "tar": round(tar, 6),
            "threshold": round(thr, 6),
            "actual_far": round(actual, 6),
            "n_impostor": n_impostor,
            "far_resolution": 1.0 / n_impostor if n_impostor else None,
            "status": status,
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
        # —— 描述性准确率（测试集自选阈值）——
        "metric": "accuracy",
        "value": round(acc_best, 6),
        "threshold": round(thr_best, 6),
        # —— 当前异人分数校准阈值后的 TAR ——
        "tar_at_far": tar_block,
        "same_score": _dist(same),
        "diff_score": _dist(diff),
        "per_fold": per_fold,
        "accuracy_per_fold_mean": round(float(fold_accs.mean()), 6),
        "accuracy_per_fold_std": round(float(fold_accs.std()), 6),
    }
