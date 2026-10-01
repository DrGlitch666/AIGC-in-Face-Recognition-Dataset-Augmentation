"""LFW 数据集的解析与评测协议构建（任务卡 #3 / #5 的地基）

## 目录结构

    <root>/lfw/<identity>/<identity>_%04d.jpg     5749 个身份 / 13233 张图
    <root>/pairs.txt                              官方 6000 对协议

## pairs.txt 格式（2026-10 实测确认）

    第 1 行  : 表头 "10\\t300" —— 10 折，每折 300 对同人
    之后 6000 行 **交错** 排列：每折先 300 行同人对，再 300 行异人对

        同人: name \\t idx1 \\t idx2          （3 个字段）
        异人: name1 \\t idx1 \\t name2 \\t idx2 （4 个字段）

    ⚠️ **没有折号列**！折算的是隐式分块（第 f 折 = 第 2f 与 2f+1 个 300 行块）。

## ⚠️ 身份泄漏（本项目必须处理的核心问题）

    LFW 的 5749 个身份里，**4281 个（74%）出现在官方 6000 对中**，
    而且**所有"图片数 ≥ 10"的身份都在 pairs 里**（George_W_Bush 有 530 张，也在）。

    所以「从 LFW 选身份训练 + 用官方 6000 对评测」**必然重叠**。
    本模块提供 :func:`filter_pairs_by_identity` 把含训练身份的对剔掉。

    实测代价（2026-10，机器 A）：

        训练身份（图片最多的前 K 个） | 训练图片 | 剩余对 | 损失
        ---------------------------- | -------- | ------ | ----
        40                           | 2537     | 5813   | 3.1%
        60                           | 2983     | 5713   | 4.8%
        96（>=15 张/人）             | 3595     | 5553   | 7.5%
        158                          | 4324     | 5274   | 12.1%

    结论：**剔除泄漏对是可行的**，官方对主要用的是"图少"的长尾身份。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 官方协议常量（写在 pairs.txt 表头里，这里作为校验基准）
EXPECTED_FOLDS = 10
PAIRS_PER_FOLD_PER_KIND = 300


class LfwFormatError(ValueError):
    """pairs.txt 或目录结构不符合预期。"""


@dataclass(frozen=True)
class Pair:
    """一对人脸（官方协议的一行）。"""

    fold: int          # 1..10
    kind: str          # "same" | "diff"
    id_a: str
    idx_a: int
    id_b: str
    idx_b: int

    @property
    def filename_a(self) -> str:
        return f"{self.id_a}_{self.idx_a:04d}.jpg"

    @property
    def filename_b(self) -> str:
        return f"{self.id_b}_{self.idx_b:04d}.jpg"

    @property
    def relpath_a(self) -> str:
        return f"{self.id_a}/{self.filename_a}"

    @property
    def relpath_b(self) -> str:
        return f"{self.id_b}/{self.filename_b}"

    @property
    def is_same(self) -> bool:
        return self.kind == "same"

    @property
    def label(self) -> int:
        """1 = 同人，0 = 异人。"""
        return 1 if self.is_same else 0

    def identities(self) -> tuple[str, ...]:
        return (self.id_a,) if self.is_same else (self.id_a, self.id_b)


def parse_pairs(pairs_txt: str | Path) -> list[Pair]:
    """解析官方 pairs.txt，返回 6000 个 :class:`Pair`（带折号）。

    会校验表头与行数，格式不对直接报错而不是静默产出错误协议。
    """
    path = Path(pairs_txt)
    if not path.exists():
        raise FileNotFoundError(f"找不到 pairs.txt: {path}")

    rows = [ln for ln in path.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
    if not rows:
        raise LfwFormatError("pairs.txt 是空的")

    head = rows[0].split()
    if len(head) != 2:
        raise LfwFormatError(f"表头格式不对，期望 '10\\t300'，实际 {rows[0]!r}")
    folds, per_kind = int(head[0]), int(head[1])
    if folds != EXPECTED_FOLDS or per_kind != PAIRS_PER_FOLD_PER_KIND:
        raise LfwFormatError(
            f"表头是 {folds} 折 × {per_kind} 对，与官方协议 "
            f"（{EXPECTED_FOLDS} × {PAIRS_PER_FOLD_PER_KIND}）不符"
        )

    body = rows[1:]
    block = per_kind
    expected_lines = folds * block * 2
    if len(body) != expected_lines:
        raise LfwFormatError(f"期望 {expected_lines} 行数据，实际 {len(body)} 行")

    pairs: list[Pair] = []
    for f in range(folds):
        same_chunk = body[f * 2 * block:(f * 2 + 1) * block]
        diff_chunk = body[(f * 2 + 1) * block:(f * 2 + 2) * block]

        for ln in same_chunk:
            parts = ln.split("\t")
            if len(parts) != 3:
                raise LfwFormatError(f"同人对应为 3 个字段，实际 {ln!r}")
            name, i1, i2 = parts
            pairs.append(Pair(f + 1, "same", name, int(i1), name, int(i2)))

        for ln in diff_chunk:
            parts = ln.split("\t")
            if len(parts) != 4:
                raise LfwFormatError(f"异人对应为 4 个字段，实际 {ln!r}")
            n1, i1, n2, i2 = parts
            if n1 == n2:
                raise LfwFormatError(f"异人对两侧身份相同，疑似格式错误: {ln!r}")
            pairs.append(Pair(f + 1, "diff", n1, int(i1), n2, int(i2)))

    return pairs


def pairs_identity_set(pairs: list[Pair]) -> set[str]:
    """官方协议里出现过的全部身份。"""
    ids: set[str] = set()
    for p in pairs:
        ids.add(p.id_a)
        ids.add(p.id_b)
    return ids


def filter_pairs_by_identity(pairs: list[Pair], exclude: set[str]) -> list[Pair]:
    """剔除**任一侧**身份落在 ``exclude`` 里的对（用于消除训练集泄漏）。

    返回的对会保留原折号；折内数量不再均匀，所以跨折汇总时建议整体汇总
    （或者按"剩下的对数"加权平均）。
    """
    return [p for p in pairs if p.id_a not in exclude and p.id_b not in exclude]


def list_identities(lfw_dir: str | Path) -> dict[str, list[str]]:
    """扫描 LFW 图片目录，返回 ``{身份: [文件名, ...]}``（按文件名排序）。"""
    root = Path(lfw_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"找不到 LFW 图片目录: {root}")

    out: dict[str, list[str]] = {}
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        files = sorted(f.name for f in d.iterdir() if f.suffix.lower() in IMAGE_EXT)
        if files:
            out[d.name] = files
    return out


def select_training_identities(
    identities: dict[str, list[str]],
    min_images: int = 15,
    max_identities: int | None = None,
) -> list[str]:
    """选训练身份：图片数 ≥ ``min_images``，按图片数从多到少排序。

    ``max_identities`` 可再截断（例如只要前 96 个）。
    """
    counts = {k: len(v) for k, v in identities.items()}
    picked = [k for k, n in counts.items() if n >= min_images]
    picked.sort(key=lambda k: (-counts[k], k))
    if max_identities is not None:
        picked = picked[:max_identities]
    return picked


def protocol_summary(
    pairs: list[Pair],
    kept: list[Pair] | None = None,
) -> dict[str, object]:
    """给实验记录用的协议摘要（写进 metrics.json 的 ``protocol`` 字段）。"""
    ids = pairs_identity_set(pairs)
    summary: dict[str, object] = {
        "dataset": "lfw",
        "protocol": "official-6000pairs-10fold",
        "num_pairs": len(pairs),
        "num_same": sum(1 for p in pairs if p.is_same),
        "num_diff": sum(1 for p in pairs if not p.is_same),
        "folds": len({p.fold for p in pairs}),
        "num_identities": len(ids),
    }
    if kept is not None:
        kept_ids = pairs_identity_set(kept)
        summary.update({
            "filtered": True,
            "num_pairs_kept": len(kept),
            "num_same_kept": sum(1 for p in kept if p.is_same),
            "num_diff_kept": sum(1 for p in kept if not p.is_same),
            "num_pairs_removed": len(pairs) - len(kept),
            "num_identities_kept": len(kept_ids),
        })
    return summary


def fold_counts(pairs: list[Pair]) -> dict[int, int]:
    """每折剩余对数（过滤后折会不均衡，记录用）。"""
    return dict(sorted(Counter(p.fold for p in pairs).items()))
