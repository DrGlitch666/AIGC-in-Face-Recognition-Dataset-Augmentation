#!/usr/bin/env python3
"""Generate five offline SVG figures from exported site_data.json.

Usage: python tools/build_site_charts.py
Python standard library only; no GPU, matplotlib, CDN, or model loading.
"""
from __future__ import annotations

import argparse
import hashlib
from html import escape
import json
import math
from pathlib import Path
import sys


WIDTH = 1080
INK = "#172c40"
MUTED = "#637588"
GRID = "#dce4eb"
BLUE = "#326da8"
ORANGE = "#bd5126"
GOLD = "#a77818"
PROTOCOLS = ("official", "filtered", "heldout")
LABELS = {"official": "Official 官方对子", "filtered": "Filtered 剔除训练身份", "heldout": "Held-out 自定义协议"}


def finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label}: expected a finite number")
    return float(value)


class SVG:
    def __init__(self, title, subtitle, height, metadata):
        self.height = height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" viewBox="0 0 {WIDTH} {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(title)}</title><desc id="desc">{escape(subtitle)}</desc>',
            '<style>text{font-family:"Microsoft YaHei","PingFang SC","Noto Sans CJK SC",Arial,sans-serif;fill:#172c40} .muted{fill:#637588}</style>',
            '<rect width="100%" height="100%" fill="#ffffff" rx="14"/>',
            '<metadata>' + escape(json.dumps(metadata, ensure_ascii=False, allow_nan=False)) + '</metadata>',
        ]
        self.text(30, 42, title, size=25, weight=700)
        self.text(30, 73, subtitle, size=15, color=MUTED)

    def text(self, x, y, value, size=15, anchor="start", color=INK, weight=400):
        self.parts.append(f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" style="fill:{color}">{escape(str(value))}</text>')

    def line(self, x1, y1, x2, y2, color=GRID, width=1, dash=None):
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{width}"{extra}/>')

    def dot(self, x, y, color, radius=6, square=False):
        if square:
            self.parts.append(f'<rect x="{x-radius:.2f}" y="{y-radius:.2f}" width="{radius*2}" height="{radius*2}" fill="{color}"/>')
        else:
            self.parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="{color}"/>')

    def band(self, x, y, width, height, color):
        self.parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" fill="{color}"/>')

    def whisker(self, xlow, xhigh, y, color):
        self.line(xlow, y, xhigh, y, color, 3)
        self.line(xlow, y-6, xlow, y+6, color, 2)
        self.line(xhigh, y-6, xhigh, y+6, color, 2)

    def finish(self):
        return "\n".join(self.parts + ["</svg>"]) + "\n"


def axis(svg, low, high, top, bottom, left=265, right=750, tick_count=5, signed=False):
    if not high > low:
        raise ValueError("Invalid chart range")
    def project(value):
        value = finite(value, "chart coordinate")
        if value < low-1e-10 or value > high+1e-10:
            raise ValueError(f"Chart value {value} falls outside [{low}, {high}]")
        return left + (value-low)/(high-low)*(right-left)
    for index in range(tick_count+1):
        value = low + (high-low)*index/tick_count
        x = project(value)
        svg.line(x, top, x, bottom)
        text = f"{value:+.3f}" if signed else f"{value:.3f}"
        svg.text(x, bottom+25, text, size=13, anchor="middle", color=MUTED)
    svg.line(left, bottom, right, bottom, INK)
    if low <= 0 <= high:
        svg.line(project(0), top, project(0), bottom, INK, 1.5, "5 4")
    return project


def padded_range(values, minimum_span=0.001):
    lo, hi = min(values), max(values)
    span = max(hi-lo, minimum_span)
    return lo-span*0.12, hi+span*0.12


def metadata(context, chart, rows, **extra):
    return {"generator": "tools/build_site_charts.py", "chart": chart,
            "input_sha256": context["sha256"], "sources": context["sources"],
            "rows": rows, **extra}


def generator_chart(data, context):
    gen = data["generation"]
    by_key = {a["key"]: a for a in gen["arms"]}
    tuning = {a["key"]: a for a in gen["tuning_round"]["arms"]}
    selected = [
        ("base FaceID", by_key["base-faceid"]),
        ("plusv2 官方配方", by_key["plusv2"]),
        ("plusv2 多参考平均", tuning["plusv2-meanref"]),
        ("plusv2 + Realistic Vision", by_key["plusv2-realvis"]),
    ]
    rows = []
    for label, arm in selected:
        rows.append({"label": label, "key": arm["key"],
                     "mean": finite(arm["id_sim_mean"], "id_sim mean"),
                     "sd": finite(arm["id_sim_stdev"], "id_sim SD"),
                     "min": finite(arm["id_sim_min"], "id_sim min"),
                     "n_scored": arm["n_scored"]})
    refs = gen["reference_points"]
    same = finite(refs["same_person_real_photos"], "same-person reference")
    diff = finite(refs["different_people_below"], "different-person reference")
    threshold = finite(refs["same_person_threshold"], "empirical threshold")
    upper = max(same, diff, threshold, *(r["mean"]+r["sd"] for r in rows)) + 0.055
    lower = min(0.0, *(r["mean"]-r["sd"] for r in rows), *(r["min"] for r in rows))
    chart = SVG("生成器质量：更像本人，也更一致", "圆点与横线：均值 ± 样本标准差；方块：可评分图中的最小值。", 550,
                metadata(context, "generator_quality", rows, references=refs, uncertainty="sample SD, not CI"))
    chart.band(265, 128, (diff-lower)/(upper-lower)*485, 305, "#f4f7f9")
    x = axis(chart, lower, upper, 128, 433)
    for value, label, color in [(same, "同人真实参考", BLUE), (threshold, "经验门槛", GOLD)]:
        chart.line(x(value), 119, x(value), 433, color, 1.3, "4 4")
        chart.text(x(value), 111, f"{label} {value:.3f}", size=12, anchor="middle", color=color)
    for index, row in enumerate(rows):
        y = 162+index*75
        chart.text(30, y+5, row["label"], size=15, weight=600)
        chart.whisker(x(row["mean"]-row["sd"]), x(row["mean"]+row["sd"]), y, BLUE)
        chart.dot(x(row["mean"]), y, BLUE)
        chart.dot(x(row["min"]), y+16, GOLD, radius=4, square=True)
        chart.text(1025, y-1, f"{row['mean']:.4f} ± {row['sd']:.4f}", anchor="end", size=16)
        chart.text(1025, y+23, f"最小 {row['min']:.4f} · 可评分 {row['n_scored']} 张", anchor="end", size=13, color=MUTED)
    chart.text(30, 486, f"浅色区域：不同人参考 < {diff:.3f}。这些参照来自本项目，并非通用身份判定阈值。", size=14, color=MUTED)
    chart.text(30, 516, "统计包含可评分样本；检测失败不等于零分。小样本配方对照与批量训练语料应分别报告。", size=14, color=MUTED)
    return chart.finish()


def mean_chart(data, context, metric):
    arms = {a["name"]: a for a in data["recognition"]["arms"]}
    rows = []
    for protocol in PROTOCOLS:
        for arm_name in ("E1", "E4a"):
            s = arms[arm_name]["protocols"][protocol]
            rows.append({"protocol": protocol, "arm": arm_name,
                         "mean": finite(s[f"{metric}_mean"], "mean"),
                         "sd": finite(s[f"{metric}_std"], "SD"),
                         "per_seed": s[f"per_seed_{metric}"]})
    title = "Accuracy：三个协议下的均值对照" if metric == "accuracy" else "TAR：低误接受率下，提升仍有不确定性"
    far = finite(data["recognition"]["far"], "FAR")
    subtitle = f"每方案 {data['recognition']['n_seeds']} 个种子；横线为均值 ± 跨种子标准差，不是置信区间。"
    if metric == "tar":
        subtitle += f" FAR = {far:g}。"
    chart = SVG(title, subtitle, 610, metadata(context, metric+"_mean_sd", rows, uncertainty="across-seed sample SD, not CI", far=far))
    ranges = [v for r in rows for v in (r["mean"]-r["sd"], r["mean"]+r["sd"])]
    low, high = padded_range(ranges)
    if metric == "tar":
        low, high = min(0.0, low), max(0.55, high)
    x = axis(chart, low, high, 120, 473)
    chart.dot(820, 106, BLUE, 5)
    chart.text(835, 111, "E1 纯真实", size=13)
    chart.dot(937, 106, ORANGE, 5)
    chart.text(952, 111, "E4a 加合成", size=13)
    for index, protocol in enumerate(PROTOCOLS):
        base = 154+index*119
        chart.text(30, base+23, LABELS[protocol], size=14, weight=600)
        for j, arm_name in enumerate(("E1", "E4a")):
            row = rows[index*2+j]
            y = base+j*38
            color = BLUE if j == 0 else ORANGE
            chart.text(236, y+5, arm_name, size=13, anchor="end", color=color)
            chart.whisker(x(row["mean"]-row["sd"]), x(row["mean"]+row["sd"]), y, color)
            chart.dot(x(row["mean"]), y, color)
            chart.text(1025, y+5, f"{row['mean']:.4f} ± {row['sd']:.4f}", anchor="end", size=16, color=color)
        if index < 2:
            chart.line(30, base+75, 1025, base+75, "#edf1f5")
    if metric == "accuracy":
        note = "Accuracy 为 accuracy@best；数轴局部放大。显著性请查看 ΔAccuracy 的 Welch 置信区间。"
    else:
        all_nonsignificant = all(not data["derived_tests"][p]["tar"]["significant_at_0_05"] for p in PROTOCOLS)
        verdict = "当前全部协议的 TAR 均未达双侧显著。" if all_nonsignificant else "各协议的显著性请查看 ΔTAR 置信区间。"
        note = "Held-out TAR 使用全量配对；不使用 accuracy 平衡子集上的 TAR。" + verdict
    chart.text(30, 535, note, size=14, color=MUTED)
    chart.text(30, 566, "Official 包含训练身份；Filtered 为剔除训练身份的官方对子子集；Held-out 是自定义协议，非官方 LFW 数字。", size=13, color=MUTED)
    return chart.finish()


def delta_chart(data, context, metric):
    rows = []
    for protocol in PROTOCOLS:
        source = data["derived_tests"][protocol][metric]
        row = {"protocol": protocol, **source}
        for key in ("delta", "ci95_low", "ci95_high", "p_two_sided", "df", "t"):
            row[key] = finite(row[key], key)
        if not row["ci95_low"] <= row["delta"] <= row["ci95_high"]:
            raise ValueError("Confidence interval does not contain its point estimate")
        if bool(row["significant_at_0_05"]) != (row["p_two_sided"] < 0.05):
            raise ValueError("Significance flag differs from p-value")
        rows.append(row)
    name = "Accuracy" if metric == "accuracy" else "TAR"
    chart = SVG(f"Δ{name}：提升是否跨过零？", "圆点：E4a − E1；横线：跨种子 Welch 95% 置信区间；虚线：零效应。", 570,
                metadata(context, "delta_"+metric+"_ci95", rows, uncertainty="95% Welch CI using each test's df; rounded source statistics"))
    low, high = padded_range([0.0] + [v for r in rows for v in (r["ci95_low"], r["ci95_high"])])
    x = axis(chart, low, high, 122, 388, signed=True)
    for index, row in enumerate(rows):
        y = 164+index*87
        color = ORANGE if row["significant_at_0_05"] else BLUE
        chart.text(30, y+5, LABELS[row["protocol"]], size=14, weight=600)
        chart.whisker(x(row["ci95_low"]), x(row["ci95_high"]), y, color)
        chart.dot(x(row["delta"]), y, color)
        chart.text(1025, y-11, f"Δ {row['delta']:+.4f}   p={row['p_two_sided']:.6f}", anchor="end", size=14)
        chart.text(1025, y+12, f"[{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}]", anchor="end", size=14, color=MUTED)
        verdict = "双侧 p < 0.05" if row["significant_at_0_05"] else "双侧未达显著"
        chart.text(1025, y+33, f"{verdict} · df={row['df']:.3f}", anchor="end", size=12, color=color)
    chart.text(30, 449, "此区间描述跨训练种子的均值差，不是单个种子内部的配对 bootstrap 区间。", size=14, color=MUTED)
    chart.text(30, 480, "按每个检验自身的自由度计算；p 值与区间使用冻结文件中已舍入的统计量，因而为近似值。", size=14, color=MUTED)
    chart.text(30, 511, "未达显著表示现有证据不足以排除零效应；不能据此断言合成数据完全没有作用。", size=14, color=MUTED)
    return chart.finish()


def build(data, sha256):
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported site_data schema; rerun export_site_data.py")
    context = {"sha256": sha256, "sources": data["sources"]}
    # Prepare every chart before writing files, so schema errors do not leave partial updates.
    return {
        "generator_quality.svg": generator_chart(data, context),
        "accuracy_mean_sd.svg": mean_chart(data, context, "accuracy"),
        "tar_mean_sd.svg": mean_chart(data, context, "tar"),
        "tar_delta_ci95.svg": delta_chart(data, context, "tar"),
        "accuracy_delta_ci95.svg": delta_chart(data, context, "accuracy"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data", type=Path, help="Default: <root>/site/data/site_data.json")
    parser.add_argument("--output", type=Path, help="Default: <root>/site/assets/figures")
    args = parser.parse_args()
    root = args.root.resolve()
    data_path = args.data.resolve() if args.data else root / "site/data/site_data.json"
    output = args.output.resolve() if args.output else root / "site/assets/figures"
    try:
        raw = data_path.read_bytes()
        data = json.loads(raw.decode("utf-8-sig"))
        charts = build(data, hashlib.sha256(raw).hexdigest())
        output.mkdir(parents=True, exist_ok=True)
        for filename, content in charts.items():
            (output / filename).write_text(content, encoding="utf-8")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(f"Charts failed: {exc}", file=sys.stderr)
        return 1
    print("Charts OK")
    print(f"Generated {len(charts)} SVG charts in: {output}")
    for filename in charts:
        print("  " + filename)
    print("SD charts: mean +/- sample SD; delta charts: Welch 95% CI (not SD)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
