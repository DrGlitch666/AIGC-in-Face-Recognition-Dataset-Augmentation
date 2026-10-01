"""LFW 协议模块的单元测试。

⚠️ **不需要真实数据集**：全部用合成 fixture，所以在 B 的机器上也能跑绿。
"""

from __future__ import annotations

import pytest

from aigcfr.data.lfw import (
    EXPECTED_FOLDS,
    PAIRS_PER_FOLD_PER_KIND,
    LfwFormatError,
    Pair,
    filter_pairs_by_identity,
    fold_counts,
    list_identities,
    pairs_identity_set,
    parse_pairs,
    protocol_summary,
    select_training_identities,
)

FOLDS = EXPECTED_FOLDS
PER = PAIRS_PER_FOLD_PER_KIND


def write_pairs(path, *, folds=FOLDS, per_kind=PER, header=None, same_ids=None):
    """生成一份格式正确的合成 pairs.txt（交错布局，与官方一致）。"""
    lines = [header if header is not None else f"{folds}\t{per_kind}"]
    for f in range(folds):
        for i in range(per_kind):
            ident = f"same_f{f}_{i}" if same_ids is None else same_ids[i % len(same_ids)]
            lines.append(f"{ident}\t1\t2")
        for i in range(per_kind):
            lines.append(f"diffA_f{f}_{i}\t1\tdiffB_f{f}_{i}\t1")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def make_lfw_tree(root, spec):
    """spec: {身份: 图片数} -> 建出 lfw/<身份>/<身份>_%04d.jpg"""
    lfw = root / "lfw"
    for ident, n in spec.items():
        d = lfw / ident
        d.mkdir(parents=True, exist_ok=True)
        for i in range(1, n + 1):
            (d / f"{ident}_{i:04d}.jpg").write_bytes(b"\xff\xd8\xff")  # 假 jpg
    return lfw


# ---------------------------------------------------------------- 解析

def test_parse_pairs_counts_and_folds(tmp_path):
    pairs = parse_pairs(write_pairs(tmp_path / "pairs.txt"))
    assert len(pairs) == FOLDS * PER * 2 == 6000
    assert sum(1 for p in pairs if p.is_same) == 3000
    assert sum(1 for p in pairs if not p.is_same) == 3000
    assert sorted({p.fold for p in pairs}) == list(range(1, 11))
    # 每折恰好 600 对（300 同 + 300 异）
    assert all(v == 600 for v in fold_counts(pairs).values())


def test_parse_pairs_fold_assignment_is_interleaved(tmp_path):
    """每折 = 先 300 同人、再 300 异人（不是"先 3000 同人"）。"""
    pairs = parse_pairs(write_pairs(tmp_path / "pairs.txt"))
    fold1 = [p for p in pairs if p.fold == 1]
    assert len(fold1) == 600
    assert all(p.is_same for p in fold1[:300])
    assert all(not p.is_same for p in fold1[300:])
    # 第 2 折的第 1 对，其身份应来自第 2 折的块
    fold2_first = [p for p in pairs if p.fold == 2][0]
    assert fold2_first.id_a.startswith("same_f1_")


def test_parse_pairs_bad_header(tmp_path):
    p = write_pairs(tmp_path / "pairs.txt", header="5\t300")
    with pytest.raises(LfwFormatError, match="不符"):
        parse_pairs(p)


def test_parse_pairs_wrong_line_count(tmp_path):
    p = write_pairs(tmp_path / "pairs.txt")
    text = p.read_text(encoding="utf-8").splitlines()
    p.write_text("\n".join(text[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(LfwFormatError, match="行数据"):
        parse_pairs(p)


def test_parse_pairs_bad_field_count(tmp_path):
    p = write_pairs(tmp_path / "pairs.txt")
    text = p.read_text(encoding="utf-8").splitlines()
    text[1] = "only_two\tfields"          # 同人区第一行给 2 个字段
    p.write_text("\n".join(text) + "\n", encoding="utf-8")
    with pytest.raises(LfwFormatError, match="3 个字段"):
        parse_pairs(p)


def test_parse_pairs_diff_pair_same_identity_is_rejected(tmp_path):
    p = write_pairs(tmp_path / "pairs.txt")
    text = p.read_text(encoding="utf-8").splitlines()
    text[1 + PER] = "Alice\t1\tAlice\t2"   # 异人区第一行两侧同人
    p.write_text("\n".join(text) + "\n", encoding="utf-8")
    with pytest.raises(LfwFormatError, match="两侧身份相同"):
        parse_pairs(p)


def test_parse_pairs_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_pairs(tmp_path / "nope.txt")


# ---------------------------------------------------------------- Pair

def test_pair_paths():
    p = Pair(fold=3, kind="same", id_a="Ann", idx_a=1, id_b="Ann", idx_b=12)
    assert p.filename_a == "Ann_0001.jpg"
    assert p.filename_b == "Ann_0012.jpg"
    assert p.relpath_a == "Ann/Ann_0001.jpg"
    assert p.label == 1
    assert p.identities() == ("Ann",)

    q = Pair(1, "diff", "Ann", 1, "Bob", 2)
    assert q.label == 0
    assert set(q.identities()) == {"Ann", "Bob"}


# ---------------------------------------------------------------- 过滤

def test_filter_pairs_by_identity_removes_any_side(tmp_path):
    pairs = parse_pairs(write_pairs(tmp_path / "pairs.txt"))
    excluded = {p.id_a for p in pairs if p.fold == 1 and p.is_same}
    assert excluded, "fixture 应该产出可排除的身份"
    kept = filter_pairs_by_identity(pairs, excluded)
    assert len(kept) < len(pairs)
    assert not (pairs_identity_set(kept) & excluded)


def test_filter_with_empty_set_keeps_everything(tmp_path):
    pairs = parse_pairs(write_pairs(tmp_path / "pairs.txt"))
    assert filter_pairs_by_identity(pairs, set()) == pairs


def test_pairs_identity_set(tmp_path):
    pairs = parse_pairs(write_pairs(tmp_path / "pairs.txt"))
    ids = pairs_identity_set(pairs)
    assert "same_f0_0" in ids
    assert "diffA_f0_0" in ids and "diffB_f0_0" in ids


# ---------------------------------------------------------------- 目录扫描与身份选择

def test_list_identities(tmp_path):
    make_lfw_tree(tmp_path, {"Alice": 3, "Bob": 1, "Empty": 0})
    ids = list_identities(tmp_path / "lfw")
    assert set(ids) == {"Alice", "Bob"}          # 空目录被跳过
    assert ids["Alice"] == ["Alice_0001.jpg", "Alice_0002.jpg", "Alice_0003.jpg"]


def test_list_identities_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        list_identities(tmp_path / "nope")


def test_select_training_identities_orders_by_count_then_name(tmp_path):
    make_lfw_tree(tmp_path, {"Big": 20, "Mid": 15, "Small": 14, "TieB": 15})
    ids = list_identities(tmp_path / "lfw")
    picked = select_training_identities(ids, min_images=15)
    assert picked == ["Big", "Mid", "TieB"]      # 同数量按名字排序，Small 被排除
    assert select_training_identities(ids, min_images=15, max_identities=2) == ["Big", "Mid"]
    assert select_training_identities(ids, min_images=100) == []


# ---------------------------------------------------------------- 记录用摘要

def test_protocol_summary_full(tmp_path):
    pairs = parse_pairs(write_pairs(tmp_path / "pairs.txt"))
    s = protocol_summary(pairs)
    assert s["num_pairs"] == 6000
    assert s["num_same"] == 3000 and s["num_diff"] == 3000
    assert s["folds"] == 10
    assert "filtered" not in s


def test_protocol_summary_filtered(tmp_path):
    pairs = parse_pairs(write_pairs(tmp_path / "pairs.txt"))
    kept = filter_pairs_by_identity(pairs, {"same_f0_0"})
    s = protocol_summary(pairs, kept)
    assert s["filtered"] is True
    assert s["num_pairs_kept"] == len(kept)
    assert s["num_pairs_removed"] == 6000 - len(kept)
    assert s["num_pairs_kept"] == s["num_same_kept"] + s["num_diff_kept"]


def test_fold_counts_after_filter(tmp_path):
    """只抹掉第 1 折的"同人"侧：该折应剩 300 对（异人），其他折不受影响。

    注意 ``fold_counts`` 是**稀疏**的 —— 某折被清空时不会出现在结果里
    （所以下面用 ``.get`` 而不是 ``[]``）。
    """
    pairs = parse_pairs(write_pairs(tmp_path / "pairs.txt"))
    excluded = {p.id_a for p in pairs if p.fold == 1 and p.is_same}
    assert len(excluded) == PER, "fixture 第 1 折应有 300 个不同的同人身份"

    kept = filter_pairs_by_identity(pairs, excluded)
    counts = fold_counts(kept)
    assert counts[1] == PER            # 同人被剔，异人还在
    assert counts[2] == 600            # 其他折完全不受影响


def test_fold_counts_is_sparse_when_a_fold_is_emptied(tmp_path):
    """把一整折两侧身份都排除 -> 该折从结果里消失（而不是报 0）。"""
    pairs = parse_pairs(write_pairs(tmp_path / "pairs.txt"))
    excluded = {p.id_a for p in pairs if p.fold == 1} | {p.id_b for p in pairs if p.fold == 1}
    counts = fold_counts(filter_pairs_by_identity(pairs, excluded))
    assert 1 not in counts
    assert counts.get(1, 0) == 0
    assert counts[2] == 600
