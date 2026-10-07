#!/usr/bin/env python3
"""Build an offline one-page showcase from exported data and generated SVG charts.

Run: python tools/build_site_page.py
Uses Python's standard library. Does not train models or modify experiment results.
"""
from __future__ import annotations

import argparse
import hashlib
from html import escape
import json
import math
from pathlib import Path
import shutil
import sys


CHARTS = {
    "generator_quality.svg": "生成器质量：均值、标准差与最小值",
    "accuracy_mean_sd.svg": "Accuracy：跨种子均值与标准差",
    "tar_mean_sd.svg": "TAR：跨种子均值与标准差",
    "tar_delta_ci95.svg": "TAR 增益的 Welch 95% 置信区间",
    "accuracy_delta_ci95.svg": "Accuracy 增益的 Welch 95% 置信区间",
}
PROTOCOLS = ("official", "filtered", "heldout")
PROTOCOL_LABELS = {"official": "Official · 官方对子", "filtered": "Filtered · 剔除训练身份", "heldout": "Held-out · 自定义训练身份不重叠协议"}


def n(value, label="value"):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label}: expected a finite number")
    return value


def h(value):
    return escape(str(value))


def mean_sd(summary, metric):
    return f"{n(summary[metric+'_mean']):.4f} ± {n(summary[metric+'_std']):.4f}"


def figure(name, caption=None):
    alt = CHARTS[name]
    return f'<figure class="chart"><a href="assets/figures/{name}" target="_blank" rel="noopener"><img src="assets/figures/{name}" alt="{h(alt)}" loading="lazy"></a><figcaption>{h(caption or alt)} · 点击查看大图</figcaption></figure>'


CSS = """
:root{--ink:#142a3b;--muted:#637584;--paper:#f7f8fa;--line:#dbe2e8;--orange:#b6502b;--blue:#326da8;--mint:#ecf5ef;--amber:#fff4e8}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;font-family:"Microsoft YaHei","PingFang SC",Arial,sans-serif;color:var(--ink);background:var(--paper);line-height:1.8}a{color:var(--blue);text-underline-offset:4px}button{font:inherit;cursor:pointer}button:focus-visible,a:focus-visible{outline:3px solid #de9c58;outline-offset:5px}header{position:sticky;top:0;z-index:10;background:rgba(255,255,255,.96);border-bottom:1px solid var(--line)}.nav{max-width:1200px;margin:auto;padding:15px 25px;display:flex;align-items:center;justify-content:space-between;gap:25px}.brand{font-size:15px;font-weight:800;letter-spacing:1.5px;text-decoration:none;color:var(--ink);white-space:nowrap}.brand span{color:var(--orange)}nav{display:flex;gap:23px;flex-wrap:wrap}nav a{font-size:13px;color:var(--muted);text-decoration:none}main{max-width:1200px;margin:auto;padding:0 25px}section{padding:64px 0;border-bottom:1px solid var(--line);scroll-margin-top:95px}.hero{padding:70px 0 55px}.eyebrow{color:var(--orange);font-size:12px;font-weight:700;letter-spacing:2px;margin:0 0 18px}.hero-grid{display:grid;grid-template-columns:1.35fr 1fr;gap:60px;align-items:center}h1{font-size:clamp(32px,4vw,53px);letter-spacing:-1.7px;line-height:1.28;margin:0 0 22px;max-width:720px}h1 em{font-style:normal;color:var(--orange)}.lead{font-size:17px;color:var(--muted);max-width:690px}.hero-note{margin-top:23px;border-left:3px solid var(--orange);padding-left:17px;font-size:14px;max-width:670px}.summary{background:var(--ink);border-radius:20px;padding:32px;color:#fff;box-shadow:0 20px 50px #17374a18}.summary .small{color:#b9cbd8;font-size:12px;letter-spacing:1px}.summary h2{font-size:23px;margin:10px 0 22px;line-height:1.6}.metric-line{display:flex;justify-content:space-between;align-items:center;border-top:1px solid #ffffff26;padding:18px 0;gap:15px}.metric-line:last-child{padding-bottom:0}.metric-line strong{font-size:28px;color:#f2bc94;font-variant-numeric:tabular-nums}.metric-line span{font-size:13px;max-width:210px;color:#cbd8e2}.stats{margin-top:42px;display:grid;grid-template-columns:repeat(3,1fr);gap:18px}.stat{background:#fff;border:1px solid var(--line);border-radius:12px;padding:23px}.stat strong{display:block;font-size:31px;line-height:1.4;font-variant-numeric:tabular-nums}.stat span{font-size:13px;color:var(--muted)}.section-heading{display:flex;justify-content:space-between;align-items:flex-end;gap:30px;margin-bottom:24px}.section-heading h2{margin:0;font-size:29px;letter-spacing:-.6px}.section-heading p{margin:0;color:var(--muted);font-size:14px;max-width:520px}.tag{display:inline-block;border:1px solid #d2dce3;border-radius:30px;padding:4px 11px;font-size:12px;color:var(--muted);margin:0 6px 6px 0}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}.card{background:#fff;border:1px solid var(--line);border-radius:13px;padding:25px}.card h3{font-size:18px;margin:0 0 12px}.card p{font-size:14px;color:var(--muted);margin:0}.card p+p{margin-top:10px}.method-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.method-grid .card{border-top:3px solid var(--blue)}.method-grid .step{font-size:12px;color:var(--orange);letter-spacing:1px;font-weight:700;display:block;margin-bottom:9px}.inline-result{background:var(--mint);border:1px solid #d6e8dc;padding:19px 23px;border-radius:12px;margin:22px 0}.inline-result p{font-size:14px;margin:0}.note{background:var(--amber);border:1px solid #efdcc4;border-radius:12px;padding:22px 25px;margin:20px 0}.note strong{display:block;margin-bottom:6px}.note p{font-size:14px;margin:0;color:#654c35}.chart{margin:20px 0;background:#fff;border:1px solid var(--line);border-radius:15px;padding:12px}.chart img{width:100%;height:auto;display:block}.chart figcaption{padding:8px 13px 5px;font-size:12px;color:var(--muted)}.figure-pair{display:grid;grid-template-columns:1fr;gap:0}.table-wrap{overflow-x:auto;border:1px solid var(--line);border-radius:11px;background:white}table{border-collapse:collapse;width:100%;min-width:600px;font-size:14px}th,td{padding:13px 17px;text-align:left;border-bottom:1px solid #edf0f3;font-variant-numeric:tabular-nums}th{background:#f0f4f7;color:var(--muted);font-size:12px;font-weight:600}tbody tr:last-child td{border:0}td strong{color:var(--orange)}.tabs{display:flex;gap:9px;flex-wrap:wrap;margin-bottom:18px}.tabs button{border:1px solid var(--line);background:#fff;border-radius:7px;padding:8px 14px;font-size:13px;color:var(--muted)}.tabs button[aria-pressed="true"]{background:var(--ink);color:#fff;border-color:var(--ink)}.protocol-card{margin:0 0 20px}.protocol-card h3{font-size:19px;margin:0 0 7px}.protocol-card .description{font-size:14px;color:var(--muted);margin:0 0 15px}.verdict{font-size:11px;display:inline-block;padding:2px 7px;border-radius:4px;background:#edf3f8;color:var(--blue)}.verdict.positive{background:#f9ece5;color:var(--orange)}.pvalue{display:block;color:var(--muted);font-size:12px}.gallery{background:#fff;border:1px solid var(--line);border-radius:13px;padding:12px;margin:22px 0}.gallery a{display:block}.gallery img{width:100%;height:auto;display:block}.gallery figcaption{color:var(--muted);font-size:13px;padding:13px}.gallery-meta{display:flex;gap:9px;flex-wrap:wrap}.small{font-size:13px;color:var(--muted)}details{background:white;border:1px solid var(--line);border-radius:10px;margin-top:13px;padding:16px 20px}summary{font-weight:600;cursor:pointer;font-size:15px}details p{font-size:14px;color:var(--muted)}pre{margin:13px 0;background:#132938;border-radius:8px;color:#dce6ed;padding:19px;overflow:auto;line-height:1.65;font-size:12px}code{font-family:Consolas,"Courier New",monospace;font-size:.9em}.sources{font-size:12px;padding-left:22px;color:var(--muted)}.sources li{margin:11px 0;overflow-wrap:anywhere}.hash{font-size:11px;color:#8696a2}footer{max-width:1200px;margin:auto;padding:29px 25px 40px;display:flex;justify-content:space-between;gap:20px;font-size:12px;color:var(--muted)}.footer-links{display:flex;gap:18px}.warning-list{padding-left:20px;font-size:13px;color:#654c35}noscript{display:block;font-size:13px;padding:14px;background:var(--amber)}[hidden]{display:none!important}
@media(max-width:850px){.nav{align-items:flex-start;gap:12px;flex-direction:column;padding:13px 18px}nav{gap:15px}.hero-grid{grid-template-columns:1fr;gap:25px}.hero{padding-top:42px}main{padding:0 18px}.stats,.cards{grid-template-columns:1fr}.method-grid{grid-template-columns:1fr}.section-heading{align-items:flex-start;flex-direction:column;gap:12px}.section-heading h2{font-size:25px}section{padding:43px 0;scroll-margin-top:135px}.stat{padding:18px}.stat strong{font-size:27px}.summary{padding:25px}.chart{padding:5px}.chart figcaption{font-size:11px}.table-wrap{border-radius:8px}footer{flex-direction:column}.hero-note{font-size:13px}.gallery{padding:5px}}
@media print{header,.tabs{display:none}body{background:white}main{max-width:none}.hero-grid{grid-template-columns:1fr}.summary{background:#eee;color:#111;box-shadow:none}.summary .small,.metric-line span{color:#333}.summary h2{font-size:20px}.metric-line strong{color:#111}section{padding:24px 0}.stats{margin-top:20px}.protocol-card[hidden]{display:block!important}.chart{break-inside:avoid}.card{break-inside:avoid}.chart img{max-height:420px}a{color:inherit}details{break-inside:avoid}footer{display:none}}
"""


def render(data):
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported export schema; rerun export_site_data.py")
    recognition = data["recognition"]
    arms = {a["name"]: a for a in recognition["arms"]}
    counts = data["training_counts"]
    baseline_id = arms["E1"]["seeds"][0]["exp_id"]
    baseline = data["runs"][baseline_id]
    training = baseline["train_set"]
    ids, real, synth = (n(training["real_identities"]), n(counts["real"]), n(counts["synth"]))
    ratio = n(counts["synth_fraction"])*100
    official = data["derived_tests"]["official"]
    heldout = data["derived_tests"]["heldout"]
    e1_off = arms["E1"]["protocols"]["official"]
    e4_off = arms["E4a"]["protocols"]["official"]
    seeds = int(n(recognition["n_seeds"]))
    far = n(recognition["far"])
    reproduction = data.get("cpu_reproduction")
    if reproduction:
        cpu = reproduction["protocols"]["official"]
        reproduction_html = f'''<h3>CPU E0 复现 · 已完成</h3><p>重新提取特征后，官方配对 Accuracy@best 为 <strong>{n(cpu['accuracy']):.6f}</strong>，TAR@FAR={far:g} 为 <strong>{n(cpu['tar']):.6f}</strong>。与冻结基线在记录的六位小数精度下一致；这不表示阈值或逐对分数完全相同。</p><p>记录的测试验证：<strong>{int(n(reproduction['tests_passed']))} 项通过</strong>。识别执行器：{h(', '.join(reproduction['model']['onnx_providers']))}；评测时间：{h(reproduction['evaluated_at'])}。</p><pre>{h(reproduction['command'])}</pre><p>结果保存在独立目录；此命令会重新执行 CPU 推理，查看已有证据无需重跑。</p><p><a href="data/reproduction/metrics.json" target="_blank" rel="noopener">复现指标 JSON</a> · <a href="data/reproduction/e0_cpu_reproduction.b.md" target="_blank" rel="noopener">复现报告</a> · <a href="data/reproduction/e0_cpu_reproduction.b.pytest.txt" target="_blank" rel="noopener">测试日志</a></p>'''
    else:
        reproduction_html = '''<h3>CPU E0 复现 · 尚未导入记录</h3><p>需准备项目配置、数据及预训练权重；以下命令把结果写入独立目录：</p><pre>python scripts/evaluate.py --exp-id e0-buffalo_l-lfw --cpu --no-cache --out "results/reproductions/e0-buffalo_l-lfw-b-cpu"</pre><p>运行后保存测试证据，再重新导出网页数据。完整生成与训练流程见 docs/W3-GENERATION.md。</p>'''
    gen = data["generation"]
    recipe = gen["final_recipe"]
    winner_key = gen["tuning_round"]["winner"]
    winner = next(a for a in gen["tuning_round"]["arms"] if a["key"] == winner_key)
    protocol_sections = []
    for protocol in PROTOCOLS:
        e1 = arms["E1"]["protocols"][protocol]
        e4 = arms["E4a"]["protocols"][protocol]
        descriptions = {
            "official": "使用官方配对清单，但包含训练身份；这里的 accuracy@best 在测试分数上选最佳阈值，不是可直接对照文献的官方交叉验证 LFW 成绩。",
            "filtered": "从官方配对清单剔除含训练身份的对子；是过滤子集，不是独立采集的新测试集。",
            "heldout": "使用与训练身份无交集的自定义配对协议；不是官方 LFW 数字。Accuracy 使用平衡子集，TAR 使用全量配对。",
        }
        rows = []
        for metric, label in (("accuracy", "Accuracy@best"), ("tar", f"TAR@FAR={far:g}")):
            test = data["derived_tests"][protocol][metric]
            sig = bool(test["significant_at_0_05"])
            verdict = "双侧显著" if sig else "未达双侧显著"
            rows.append(f'<tr><td>{h(label)}</td><td>{mean_sd(e1,metric)}</td><td><strong>{mean_sd(e4,metric)}</strong></td><td>{n(test["delta"]):+.4f}<span class="pvalue">p≈{n(test["p_two_sided"]):.6f}</span></td><td><span class="verdict {"positive" if sig else ""}">{verdict}</span></td></tr>')
        protocol_sections.append(f'<article class="protocol-card" data-protocol="{protocol}"><h3>{h(PROTOCOL_LABELS[protocol])}</h3><p class="description">{h(descriptions[protocol])}</p><div class="table-wrap"><table><thead><tr><th>指标</th><th>E1 · 纯真实</th><th>E4a · 真实＋合成</th><th>均值差 E4a−E1</th><th>Welch 检验</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div></article>')

    arms_table = []
    generator_arms = gen["arms"][:2] + [winner] + gen["arms"][2:]
    for arm in generator_arms:
        label = arm["description"].replace("**", "")
        generated = arm.get("n_generated")
        scored = n(arm["n_scored"])
        scored_text = f'{scored:g}/{n(generated):g}' if generated is not None else f'{scored:g}（未记录生成总数）'
        arms_table.append(f'<tr><td>{h(label)}</td><td>{h(scored_text)}</td><td>{n(arm["id_sim_mean"]):.4f}</td><td>{n(arm["id_sim_min"]):.4f}</td></tr>')

    sample_info = []
    for protocol in ("official", "heldout"):
        block = baseline["protocols"][protocol]
        used = n(block["accuracy_pairs_used"])
        missing = block.get("accuracy_pairs_missing")
        text = f'{PROTOCOL_LABELS[protocol]}：E1 首种子 accuracy 实际使用 {used:,.0f} 对'
        if missing is not None:
            text += f'，缺失 {n(missing):,.0f} 对'
        if protocol == "heldout" and block.get("tar_pairs_used") is not None:
            text += f'；TAR 全量有效配对 {n(block["tar_pairs_used"]):,.0f} 对'
        sample_info.append(f'<li>{h(text)}。</li>')
    manifest = baseline.get("heldout_manifest") or {}
    if manifest.get("n_pairs") is not None:
        sample_info.append(f'<li>Held-out 配对清单计划规模为 {n(manifest["n_pairs"]):,.0f} 对；检测或对齐失败后的实际使用规模应以上述实验记录为准。</li>')

    source_list = ''.join(f'<li><code>{h(s["path"])}</code><br><span class="hash">SHA-256：{h(s["sha256"])}</span></li>' for s in data["sources"])
    warnings = data.get("warnings") or []
    warning_html = '<div class="note"><strong>数据完整性提示</strong><ul class="warning-list">'+''.join(f'<li>{h(w)}</li>' for w in warnings)+'</ul></div>' if warnings else ''
    proto_source = f"共核对 {len(data['sources'])} 个来源文件；本页实验数字由脚本读取导出数据生成。"
    all_tar_ns = all(not data["derived_tests"][p]["tar"]["significant_at_0_05"] for p in PROTOCOLS)
    tar_conclusion = "各协议的 TAR 均未达双侧显著，不能宣称低误接受率下性能已经稳定改善。" if all_tar_ns else "TAR 的显著性因协议而异，应逐项查看下方检验。"
    heldout_verdict = "未达双侧显著" if not heldout["accuracy"]["significant_at_0_05"] else "达到双侧显著"
    pretrain_id = "e0-buffalo_l-heldout"
    pretrain = data["runs"].get(pretrain_id, {}).get("protocols", {}).get("heldout")
    if not pretrain:
        pretrain = data["runs"].get("e0-buffalo_l-lfw", {}).get("protocols", {}).get("heldout")
    reference_html = ''
    if pretrain:
        reference_html = f'<p class="small">参考上限：零样本 buffalo_l 的 held-out accuracy={n(pretrain["accuracy"]):.4f}、TAR={n(pretrain["tar"]):.4f}。这是不同训练条件的预训练参考，不是合成数据带来的增益。</p>'
    source_folds = baseline.get("protocol_detail", {}).get("official", {}).get("folds")
    fold_note = f'官方清单包含 {n(source_folds):g} 个折，但本页 accuracy@best 使用全局最佳阈值。' if source_folds is not None else '本页 accuracy@best 使用全局最佳阈值。'
    body = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="身份保持的合成人脸能否改善识别？以真实实验、两个评测口径和统计不确定性回答。"><title>AIGC Face Lab · 合成人脸数据扩充</title><style>{CSS}</style></head><body>
<header><div class="nav"><a class="brand" href="#top">AIGC <span>FACE LAB</span></a><nav aria-label="主导航"><a href="#method">方法</a><a href="#generation">生成质量</a><a href="#gallery">视觉对照</a><a href="#results">实验结果</a><a href="#lessons">踩坑与局限</a><a href="#reproduce">复现</a></nav></div></header>
<main><section class="hero" id="top"><p class="eyebrow">身份保持生成 · 数据扩充 · 人脸验证</p><div class="hero-grid"><div><h1>合成更多照片，<br>能让模型<em>认得更准吗？</em></h1><p class="lead">为已有身份生成新的人脸照片，筛选后加入训练集，与只使用真实照片的模型对照。我们关心的不只是照片“像不像”，还有识别增益能否经得起不同协议和训练种子的检验。</p><p class="hero-note"><strong>读到最后，只需记住：</strong>accuracy 的跨种子均值上升；{h(tar_conclusion)}Held-out 是自定义训练身份不重叠协议，不是官方 LFW 数字。</p><a href="#results">直接查看证据 ↓</a></div><aside class="summary"><span class="small">实验速览 / E1 对照 E4a</span><h2>均值改善，<br>证据强度因指标而异。</h2><div class="metric-line"><strong>{n(e1_off['accuracy_mean']):.4f} → {n(e4_off['accuracy_mean']):.4f}</strong><span>官方对子<br>Accuracy@best 均值</span></div><div class="metric-line"><strong>{n(official['accuracy']['delta']):+.4f}</strong><span>官方 accuracy 均值差<br>p≈{n(official['accuracy']['p_two_sided']):.4f}</span></div><div class="metric-line"><strong>{n(official['tar']['delta']):+.4f}</strong><span>官方 TAR 均值差<br>p≈{n(official['tar']['p_two_sided']):.4f}</span></div></aside></div><div class="stats"><div class="stat"><strong>{ids:g} 个身份</strong><span>训练身份 · 已有身份的数据扩充</span></div><div class="stat"><strong>{real:,.0f} ＋ {synth:,.0f}</strong><span>真实训练图 ＋ 筛选后的合成训练图</span></div><div class="stat"><strong>{ratio:.1f}% · {seeds} 个种子</strong><span>合成图占合并训练集的比例 · 重复实验</span></div></div></section>
<section id="method"><div class="section-heading"><h2>方法：从真实参考到性能验证</h2><p>身份保持不是终点；只有加入训练后的可复核增益，才能回答数据扩充是否有价值。</p></div><div class="method-grid"><article class="card"><span class="step">数据准备</span><h3>真实照片 → 检测与对齐</h3><p>选择已有身份的真实照片，按识别模型的输入方式对齐；E1 使用真实数据作为基线。</p><p>训练模型：{h(baseline['model']['arch'])}；损失：{h(baseline['model']['loss'])}。</p></article><article class="card"><span class="step">身份保持生成</span><h3>多参考嵌入 → plusv2</h3><p>对同一身份的多张参考照片提取、平均并归一化身份嵌入；结合 CLIP 结构分支生成该身份的新照片。</p><p>采用 {h(recipe['adapter'])}；底模 {h(recipe['base'])}；{n(recipe['steps']):g} 步，guidance={n(recipe['guidance']):g}。</p></article><article class="card"><span class="step">样本筛选</span><h3>质量地板 → 冗余剪枝</h3><p>筛去身份相似度不足或无法评分的图，再减少同身份内冗余。单纯保留最相似的图，可能损失为多样性引入的结构变化。</p><p>E4a 最终加入 {synth:,.0f} 张合成图。配方小样本的相似度不能当作全部训练合成图的均值。</p></article><article class="card"><span class="step">训练与验证</span><h3>E1 对照 E4a → 多协议、多种子</h3><p>同一骨干对照纯真实数据与真实＋合成数据。分别报告 Accuracy@best 和 FAR={far:g} 下的 TAR。</p><p>每个方案重复 {seeds} 个种子；展示均值、种子间波动，以及均值差的置信区间。</p></article></div></section>
<section id="generation"><div class="section-heading"><h2>生成质量：像本人，也要有下限</h2><p>id_sim 是与本人真实参考中心的余弦相似度，是本实验中的身份保持代理指标。</p></div><div class="inline-result"><p>采用方案：<strong>{h(winner['description'].replace('**',''))}</strong>。可评分 {n(winner['n_scored']):g} 张，均值 <strong>{n(winner['id_sim_mean']):.4f}</strong>，最小值 <strong>{n(winner['id_sim_min']):.4f}</strong>。</p></div>{figure('generator_quality.svg','小样本配方对照；误差线为样本标准差，方块为最小值')}<div class="table-wrap"><table><thead><tr><th>方案</th><th>可评分／生成</th><th>id_sim 均值</th><th>最小值</th></tr></thead><tbody>{''.join(arms_table)}</tbody></table></div><p class="small">参照：同人真实照片 {n(gen['reference_points']['same_person_real_photos']):.3f}；不同人参考低于 {n(gen['reference_points']['different_people_below']):.3f}。这些是本项目参照，不是普遍适用的身份识别阈值。检测失败样本不记作零分，也不应从失败率讨论中消失。</p></section>
<section id="gallery"><div class="section-heading"><h2>视觉对照：真实参考与合成照片</h2><p>先看身份是否保持，再读分数。外观更逼真不自动意味着识别训练更有用。</p></div><div class="gallery-meta"><span class="tag">左：真实参考</span><span class="tag">中：base FaceID</span><span class="tag">右：plusv2 多参考平均</span></div><figure class="gallery"><a href="assets/contact_sheet.png" target="_blank" rel="noopener" aria-label="打开完整视觉对照图"><img src="assets/contact_sheet.png" alt="按身份分行排列的真实参考、base FaceID 与最终 plusv2 配方对照；合成图下方标注可用的 id_sim 分数。" loading="lazy"></a><figcaption>来源：仓库已有的 contact_sheet.png，原图与原始分数保持不变。真实参考图没有合成图的 id_sim；未标分数的生成样本为未评分，不能理解成零分。右侧是最终多参考平均配方，不能与上表“plusv2 官方配方”的统计混用。点击可查看原图。</figcaption></figure></section>
<section id="results"><div class="section-heading"><h2>识别结果：把两个指标一起看</h2><p>每个格子为跨种子均值 ± 样本标准差；显著性使用双侧 Welch 检验，p 值由冻结统计量推导。</p></div><div class="tabs" role="group" aria-label="筛选评测协议"><button type="button" data-filter="all" aria-pressed="true">全部协议</button><button type="button" data-filter="official" aria-pressed="false">Official</button><button type="button" data-filter="filtered" aria-pressed="false">Filtered</button><button type="button" data-filter="heldout" aria-pressed="false">Held-out</button></div>{''.join(protocol_sections)}<div class="note"><strong>统计核对：不能共用一个 t 临界值</strong><p>Held-out accuracy：t={n(heldout['accuracy']['t']):.4f}、df={n(heldout['accuracy']['df']):.3f}、双侧 p≈{n(heldout['accuracy']['p_two_sided']):.6f}，{heldout_verdict}。本页按各检验自身的自由度判断，不沿用报告初稿的统一临界值；原始实验结果文件保持不变。{h(tar_conclusion)}</p></div>{figure('accuracy_mean_sd.svg')}{figure('tar_mean_sd.svg')}{figure('accuracy_delta_ci95.svg','均值差的 95% Welch 区间；与上方标准差图是不同统计量')}{figure('tar_delta_ci95.svg','均值差的 95% Welch 区间；不要误作单次运行的配对 bootstrap 区间')}{reference_html}<details><summary>评测口径与实际样本规模</summary><p>{h(fold_note)}最佳阈值是在测试分数上选取，属于偏乐观的描述性指标，不能当作独立验证阈值后的部署结果。</p><ul class="small">{''.join(sample_info)}</ul><p>这里的训练身份交集为零，指本项目的自训练身份；外部预训练模型的数据重叠尚未完整审计。Held-out 仍来自同一数据源，并不是跨数据集、跨人口分布的泛化验证。SD 描述种子间波动；Welch 区间描述跨种子均值差，两者不能混称。</p></details></section>
<section id="lessons"><div class="section-heading"><h2>踩坑记录：测量方式会改变结论</h2><p>这些经验来自本项目排查记录；实现细节应结合当前源码与依赖版本核查。</p></div><div class="cards"><article class="card"><h3>参数顺序写反，结果可能静默失效</h3><p><code>tar_at_far(labels, scores, far)</code> 的标签和分数不能交换。用边界测试防住静默返回无效结果，而不是只看程序是否报错。</p></article><article class="card"><h3>检测尺寸引入幸存者偏差</h3><p>错误检测配置会漏掉生成样本，使可评分子集发生偏移。评分口径改变时，要重新比较全部方案，并同时报告检测失败。</p></article><article class="card"><h3>高相似度筛选也可能丢掉多样性</h3><p>只按身份相似度排名保留样本，容易保留结构相近的照片。质量地板与冗余剪枝需要一起考虑。</p></article></div><details open><summary>项目结论的适用边界</summary><p>官方配对清单含训练身份；accuracy 在测试集上选择最佳阈值。自定义 held-out 提供补充证据，却不等同于官方成绩或独立采集的测试集。</p><p>种子数量有限，TAR 增益仍不确定；当前还不能确定更多合成数据是否一定更好。这里只研究有限的训练身份、生成样本及一个识别骨干，也未覆盖完整合成占比扫描和跨数据集验证。</p><p>身份相似度代理指标不能代替对生成质量、偏差及下游性能的完整评估。预训练模型若同时参与身份条件、筛选或评测，应明确度量空间依赖。</p></details></section>
<section id="reproduce"><div class="section-heading"><h2>复现：先重建展示，再核对实验</h2><p>浏览本页无需联网、构建或后端。数据和图表由仓库脚本生成，结果可追溯。</p></div><div class="method-grid"><article class="card"><h3>重建展示 · CPU 即可</h3><p>将三个脚本放入仓库的 <code>tools</code> 目录，在激活的项目环境中运行：</p><pre>python tools/export_site_data.py
python tools/build_site_charts.py
python tools/build_site_page.py</pre><p>随后双击 <code>site/index.html</code>。这是展示构建，不会重新训练或生成模型样本。</p></article><article class="card">{reproduction_html}</article></div>{warning_html}<details><summary>数据来源与自动导出记录</summary><p>{h(proto_source)}</p><p><a href="data/site_data.json" target="_blank" rel="noopener">打开完整导出数据 JSON</a> · <a href="data/site_data.js" target="_blank" rel="noopener">打开离线 JavaScript 数据文件</a></p><ol class="sources">{source_list}</ol><p>均值、标准差、点估计和检验统计量来自结果文件。页面中的比例、p 值和置信区间按记录字段计算；p 和区间使用舍入后的冻结统计量，属于近似值。</p></details></section></main>
<footer><span>AIGC Face Lab · 用可复核的实验回答数据扩充问题</span><div class="footer-links"><a href="#top">返回顶部 ↑</a><a href="data/site_data.json">查看导出数据</a></div></footer><noscript>当前浏览器关闭了 JavaScript，所有协议结果仍完整展示；仅协议筛选按钮不可用。</noscript>
<script>document.querySelectorAll('[data-filter]').forEach(function(button){{button.addEventListener('click',function(){{var filter=button.dataset.filter;document.querySelectorAll('[data-filter]').forEach(function(item){{item.setAttribute('aria-pressed',String(item===button));}});document.querySelectorAll('[data-protocol]').forEach(function(card){{card.hidden=filter!=='all'&&card.dataset.protocol!==filter;}});}});}});</script></body></html>'''
    return body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    site = root / "site"
    try:
        data_path = site / "data/site_data.json"
        data = json.loads(data_path.read_text(encoding="utf-8-sig"))
        page = render(data)
        proof_copies = []
        for proof in data.get("cpu_reproduction", {}).get("evidence", []):
            source_path = (root / proof["path"]).resolve()
            source_path.relative_to(root)
            raw = source_path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != proof["sha256"]:
                raise ValueError(f"Evidence changed: {proof['path']}; rerun export_site_data.py")
            filename = Path(proof["filename"]).name
            if filename != proof["filename"]:
                raise ValueError("Invalid evidence filename")
            proof_copies.append((site / "data/reproduction" / filename, raw))
        required = [site / "assets/figures" / name for name in CHARTS]
        required.append(site / "data/site_data.js")
        for path in required:
            if not path.is_file():
                raise ValueError(f"Missing asset: {path}; run data export and chart generation first")
        source = root / "results/runs/w3-generator-comparison/contact_sheet.png"
        if not source.is_file():
            raise ValueError(f"Missing visual comparison: {source}")
        image = site / "assets/contact_sheet.png"
        index = site / "index.html"
        page_bytes = len(page.encode("utf-8"))
        replaced = {image, index, *(p for p, _ in proof_copies)}
        current_bytes = sum(p.stat().st_size for p in site.rglob("*") if p.is_file() and p not in replaced)
        prospective_bytes = current_bytes + source.stat().st_size + page_bytes + sum(len(raw) for _, raw in proof_copies)
        if prospective_bytes >= 5_000_000:
            raise ValueError(f"Site would use {prospective_bytes:,} bytes, exceeding the <5 MB requirement")
        image.parent.mkdir(parents=True, exist_ok=True)
        for target, raw in proof_copies:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        shutil.copyfile(source, image)
        index.write_text(page, encoding="utf-8")
    except (OSError, KeyError, TypeError, ValueError, StopIteration) as exc:
        print(f"Page failed: {exc}", file=sys.stderr)
        return 1
    print("Page OK")
    print(f"Open: {index}")
    print(f"Site size: {prospective_bytes:,} bytes (< 5 MB)")
    print("Offline: no CDN, backend, fetch, or build system required")
    print("5 generated SVG charts + original visual comparison included")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
