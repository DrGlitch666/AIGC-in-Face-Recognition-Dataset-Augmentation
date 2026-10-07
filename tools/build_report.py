#!/usr/bin/env python3
"""Build the audited Markdown report from checked export data; no inference.

Run: python tools/export_site_data.py, then python tools/build_report.py.
PDF/DOCX layout is a separate delivery step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys


REFERENCES = [
    ("Huang, G. B.; Mattar, M.; Berg, T.; Learned-Miller, E. Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments. ECCV Workshop, 2008.", "https://people.cs.umass.edu/~elm/papers/Huang_eccv2008-lfw.pdf"),
    ("Huang, G. B.; Learned-Miller, E. Labeled Faces in the Wild: Updates and New Reporting Procedures. UMass Technical Report UM-CS-2014-003, 2014.", "https://people.cs.umass.edu/~elm/papers/lfw_update.pdf"),
    ("Deng, J.; Guo, J.; Xue, N.; Zafeiriou, S. ArcFace: Additive Angular Margin Loss for Deep Face Recognition. CVPR, 2019; arXiv:1801.07698v3.", "https://arxiv.org/abs/1801.07698v3"),
    ("Rombach, R.; Blattmann, A.; Lorenz, D.; Esser, P.; Ommer, B. High-Resolution Image Synthesis with Latent Diffusion Models. CVPR, 2022.", "https://arxiv.org/abs/2112.10752"),
    ("Song, J.; Meng, C.; Ermon, S. Denoising Diffusion Implicit Models. ICLR, 2021.", "https://arxiv.org/abs/2010.02502"),
    ("Ye, H.; Zhang, J.; Liu, S.; Han, X.; Yang, W. IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models. arXiv:2308.06721, 2023.", "https://arxiv.org/abs/2308.06721"),
    ("Qiu, H.; Yu, B.; Gong, D.; Li, Z.; Liu, W.; Tao, D. SynFace: Face Recognition with Synthetic Data. ICCV, 2021.", "https://arxiv.org/abs/2108.07960"),
    ("Bae, G.; de La Gorce, M.; Baltrusaitis, T.; Hewitt, C.; Chen, D.; Valentin, J.; Cipolla, R.; Shen, J. DigiFace-1M: 1 Million Digital Face Images for Face Recognition. WACV, 2023.", "https://arxiv.org/abs/2210.02579"),
    ("Kim, M.; Liu, F.; Jain, A.; Liu, X. DCFace: Synthetic Face Generation with Dual Condition Diffusion Model. CVPR, 2023.", "https://arxiv.org/abs/2304.07060"),
    ("Otroshi Shahreza, H.; Marcel, S. Unveiling Synthetic Faces: How Synthetic Datasets Can Expose Real Identities. NeurIPS Workshop on New Frontiers in Adversarial Machine Learning, 2024.", "https://arxiv.org/abs/2410.24015"),
    ("h94 / IP-Adapter authors. IP-Adapter-FaceID official model card. Implementation and research-use conditions; accessed 2026-10-07.", "https://huggingface.co/h94/IP-Adapter-FaceID"),
    ("SciPy developers. scipy.stats.ttest_ind documentation. Welch test and two-sided alternatives; accessed 2026-10-07.", "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html"),
]


def table(headers, rows):
    def cells(values):
        return "| " + " | ".join(str(v).replace("|", "\\|").replace("\n", " ") for v in values) + " |"
    return "\n".join([cells(headers), cells(["---"] * len(headers)), *(cells(r) for r in rows)])


def summary(s, metric):
    return f"{s[metric + '_mean']:.4f} ± {s[metric + '_std']:.4f}"


def validate_export(root, data):
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported site_data schema")
    for record in data["sources"]:
        path = (root / record["path"]).resolve()
        path.relative_to(root)
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Source changed: {record['path']}; rerun export_site_data.py")
    if not data.get("cpu_reproduction"):
        raise ValueError("CPU reproduction evidence missing; rerun export after saving #26 evidence")
    if not data["cpu_reproduction"]["matches_recorded_six_decimals"]:
        raise ValueError("CPU reproduction agreement was not verified")


def render(data):
    counts = data["training_counts"]
    real, synth, total = counts["real"], counts["synth"], counts["total"]
    ratio = counts["synth_fraction"] * 100
    arms = {a["name"]: a for a in data["recognition"]["arms"]}
    n_seeds = data["recognition"]["n_seeds"]
    first = data["runs"][arms["E1"]["seeds"][0]["exp_id"]]
    ids = first["train_set"]["real_identities"]
    gen = data["generation"]
    recipe = gen["final_recipe"]
    winner = next(a for a in gen["tuning_round"]["arms"] if a["key"] == gen["tuning_round"]["winner"])
    tests = data["derived_tests"]
    far = data["recognition"]["far"]
    cpu = data["cpu_reproduction"]
    held = first["heldout_manifest"]
    e4b = data["runs"].get("e4-highquality-synth-seed0")
    protocol_names = {"official": "official 官方配对", "filtered": "filtered 过滤子集", "heldout": "held-out 自定义协议"}
    result_rows, ci_rows = [], []
    for protocol, name in protocol_names.items():
        for metric, label in (("accuracy", "Accuracy@best"), ("tar", f"TAR@FAR={far:g}")):
            t = tests[protocol][metric]
            verdict = "达到名义双侧 0.05" if t["significant_at_0_05"] else "未达双侧 0.05"
            result_rows.append([name, label, summary(arms["E1"]["protocols"][protocol], metric), summary(arms["E4a"]["protocols"][protocol], metric), f"{t['delta']:+.4f}", f"{t['t']:.4f}", f"{t['df']:.3f}", f"{t['p_two_sided']:.6f}", verdict])
            ci_rows.append([name, label, f"{t['delta']:+.6f}", f"[{t['ci95_low']:+.6f}, {t['ci95_high']:+.6f}]"])
    gen_rows = []
    for a in gen["arms"][:2] + [winner] + gen["arms"][2:]:
        scored = a["n_scored"]
        population = f"{scored}/{a['n_generated']}" if "n_generated" in a else f"{scored} 张可评分"
        gen_rows.append([a["description"].replace("**", ""), population, f"{a['id_sim_mean']:.4f}", f"{a['id_sim_min']:.4f}"])
    protocol_rows = []
    for p, name in protocol_names.items():
        v = first["protocols"][p]
        planned = first["protocol_detail"]["official"]["num_pairs"] if p == "official" else first["protocol_detail"]["filtered"]["num_pairs_kept"] if p == "filtered" else held["n_pairs"]
        protocol_rows.append([name, f"{planned:,}", f"{v['accuracy_pairs_used']:,}", f"{v['tar_pairs_used']:,}", v["accuracy_pairs_missing"]])
    seed_rows = []
    differences = []
    for a, b in zip(arms["E1"]["seeds"], arms["E4a"]["seeds"]):
        d = b["heldout"]["tar"] - a["heldout"]["tar"]
        differences.append(d)
        seed_rows.append([a["exp_id"].rsplit("seed", 1)[1], f"{a['heldout']['tar']:.6f}", f"{b['heldout']['tar']:.6f}", f"{d:+.6f}"])
    boot = data["recognition"]["tests"]["heldout"].get("paired_bootstrap_seed0")
    bootstrap_note = ""
    if boot:
        bootstrap_note = f"冻结汇总另记录 seed0 的配对 bootstrap：{boot['n_boot']:,} 次重采样，Δ 的 5%–95% 分位区间 [{boot['delta_p05']:+.6f}, {boot['delta_p95']:+.6f}]，正差重采样比例 {100*boot['frac_positive']:.1f}%。这是固定一次训练结果下的重采样分布，既不是跨种子 95% Welch 区间，也不是‘真实增益为正’的后验概率。seed1/seed2 的旧稿区间未在本次冻结汇总中逐项核实，因此不继续引用。"
    cpu_rows = []
    baseline = data["runs"]["e0-buffalo_l-lfw"]["protocols"]
    for p in ("official", "filtered"):
        for metric, label in (("accuracy", "Accuracy@best"), ("tar", f"TAR@FAR={far:g}"), ("actual_far", "实际 FAR")):
            cpu_rows.append([p, label, f"{baseline[p][metric]:.6f}", f"{cpu['protocols'][p][metric]:.6f}"])
    e0held = data["runs"].get("e0-buffalo_l-heldout", {}).get("protocols", {}).get("heldout")
    reference_point = f"冻结的预训练 buffalo_l 参考：held-out Accuracy@best={e0held['accuracy']:.6f}，TAR={e0held['tar']:.6f}。该模型具有不同的外部预训练数据与训练预算，不能据此把差距唯一归因于当前训练身份数量。" if e0held else ""
    e4b_note = f"E4b 高质量子集记录 {e4b['train_set']['synth_images']} 张合成图，合并后占 {100*e4b['train_set']['synth_images']/e4b['train_set']['total_images']:.1f}%；当前只有 seed0，official accuracy={e4b['protocols']['official']['accuracy']:.6f}、TAR={e4b['protocols']['official']['tar']:.6f}。该单次结果不足以断言它优于或劣于 E4a。" if e4b else ""
    params = first["model"]["params_total"]
    optim = first.get("optim", {})
    optim_table = table(["E1 seed0 记录项", "值"], [(k, v) for k, v in optim.items() if v is not None]) if any(v is not None for v in optim.values()) else "当前导出没有优化器字段；具体优化设置须从原始运行配置和训练日志核对。"
    sources = "\n".join(f"- `{s['path']}`；SHA-256：`{s['sha256']}`。" for s in data["sources"])
    bibliography = "\n\n".join(f"[{i}] {title} [原始来源]({url})" for i, (title, url) in enumerate(REFERENCES, 1))
    method_rows = [
        ["ID 适配器", recipe["adapter"]], ["底模 / VAE", f"{recipe['base']} / {recipe['vae']}"],
        ["ID 模型 / 参考", f"{recipe['id_model']} / {recipe['ref_mode']}"],
        ["shortcut / s_scale", f"{recipe['shortcut']} / {recipe['s_scale']}"],
        ["CLIP 层 / 输入", f"{recipe['clip_layer']} / {recipe['clip_input']}"],
        ["CLIP 负例", recipe["negative_clip"]],
        ["调度器 / 步数", f"{recipe['scheduler']} / {recipe['steps']}"],
        ["guidance / IP scale", f"{recipe['guidance']} / {recipe['ip_scale']}"],
    ]
    return f'''# AIGC 合成人脸数据用于人脸识别数据扩充：实验报告

> 结果表由 `tools/build_report.py` 从核对过的 `site/data/site_data.json` 自动生成。
> 导出数据追溯到冻结的 `results/runs/` 和独立的 CPU 复现目录；原实验 JSON 保持原样。
> 均值、标准差、t、df 沿用冻结字段；双侧 p 与 Welch 95% 区间由各自 df 推导，属于舍入记录下的近似值。

## 摘要

本项目检验：为已有身份合成更多照片，能否改善小规模人脸识别模型？
E1 使用 {ids} 个身份、{real:,} 张真实训练图；E4a 加入 {synth:,} 张筛选后的合成图，占合并训练集 {ratio:.1f}%。
两方案各运行 {n_seeds} 个训练种子，并分别报告 official、filtered 与自定义 held-out 协议。
三个协议下 Accuracy@best 的跨种子均值均上升，其中 official 和 filtered 达到名义双侧 0.05 水平；
held-out accuracy 的 p≈{tests['heldout']['accuracy']['p_two_sided']:.6f}，未达到该标准。
各协议 TAR@FAR={far:g} 的均值差为正，但均未达双侧显著，当前不能确认低误接受率下的稳定增益。
独立 CPU E0 复现与冻结基线的汇总指标在记录的六位小数下一致，且保存的测试日志记录 {cpu['tests_passed']} 项通过。

## 1. 研究问题与相关工作

研究对象是已有身份的数据扩充，而非证明合成照片可以替代全部真实数据。
Q1 比较 E1 与 E4a；Q2 比较身份保持生成配方，随后观察其下游效果。
LFW 是面向非受控环境人脸验证的数据集 [1]；其评测协议强调训练数据和报告方式的可比性 [2]。
识别损失采用 ArcFace [3]；生成路线结合潜扩散模型 [4]、DDIM 采样 [5] 和 IP-Adapter [6]。
FaceID-plusv2 的具体接口与研究用途限制以作者模型卡为依据 [11]，不把普通 IP-Adapter 论文当作全部 FaceID 实现细节的出处。

SynFace 研究合成与真实数据间的域差距和类内变化 [7]；DigiFace-1M 使用数字人脸渲染并考察增强 [8]；
DCFace 通过身份与风格双条件控制变化 [9]。这些工作支持同时关注身份一致性和多样性，
但其数据规模、训练设置与协议不同，本报告不把外部成绩与当前 Accuracy@best 直接排名。

## 2. 方法与评测口径

### 2.1 数据与流水线

真实图经检测、对齐后形成 E1 训练集；同身份参考图用于身份保持生成，筛选后与真实图合并形成 E4a；
两方案使用同一识别骨干，随后按多个协议评测。

| 数据项 | 记录及来源 |
| --- | --- |
| LFW 全量 | 5,749 个身份、13,233 张图；原始数据集文献 [1] |
| E1 训练集 | {ids} 个身份、{real:,} 张真实图；E1 metrics 的 train_set |
| E4a 训练集 | {real:,} 张真实图 + {synth:,} 张合成图，共 {total:,} 张；E4a train_set |
| 合成比例 | {synth}/{total} = {ratio:.1f}%；分母是合并训练集 |
| 选择与过滤 | 评测记录中的训练身份过滤规则为 min_images=15；实际身份名单应以训练 manifest 为准 |

原图按人物目录保存；`data/raw/lfw/lfw` 为本机解压后的照片目录，归档和配对清单位于上一级。
类别失衡是当前小样本设置的局限。初稿中的最大类、最小类与比例缺少同一训练集口径的原始统计佐证，
本次不继续使用这些精确数值；不能把 E4a 合并集的比例当作 E1 真实训练集比例。

### 2.2 生成配方与筛选

最终采用配方来自 `w3-generator-comparison/comparison.json` 的 final_recipe：

{table(['设置', '冻结记录'], method_rows)}

结构参考轮换、质量地板与冗余剪枝等实现记录见 [W3-GENERATION.md](W3-GENERATION.md)。
本报告可由训练元数据核实加入的合成图数量；初稿给出的整批生成耗时、剪枝相似度以及筛选后图集的精确 id_sim 均值，
不在当前导出的冻结统计字段中，故不再作为本报告的量化证据。配方对照与筛选后训练图集是不同总体。

### 2.3 训练与实验重复

识别模型为 {first['model']['arch']}，参数量 {params:,}，损失记录为 {first['model']['loss']}。
E1 与 E4a 各有 {n_seeds} 个种子；优化细节由对应实验配置和训练日志确定，
各运行 metrics 中的 epochs 记录不是自动等价于配置的总训练轮数。

{optim_table}

{e4b_note}

### 2.4 三种协议与实际使用规模

下表取 E1 seed0 作为规模示例。计划配对数与检测后实际有效配对数分别报告：

{table(['协议', '清单配对数', 'accuracy 有效对', 'TAR 有效对', 'accuracy 缺失对'], protocol_rows)}

Official 使用官方配对清单，记录中包含项目训练身份；虽然清单带有 10 折标记，
此处汇总 Accuracy@best 在测试分数上选择最佳阈值，不能视为文献中的独立留出折验证成绩 [2]。
Filtered 删除涉及训练身份的官方对子，但仍来自同一数据源。
Held-out 是自定义、与本项目训练身份不相交的补充协议，不是官方 LFW 成绩；
它用平衡子集报告 accuracy，用全量有效配对报告 TAR，不能取平衡子集 TAR 替代全量 TAR。

TAR 阈值由当前评测负对分布确定，报告实际 FAR。阈值校准与测试未完全独立，
因此这些指标用于当前实验比较，不能直接代表部署性能。
身份交集为零指本项目训练 manifest 与评测身份，不意味着已排除外部预训练模型的数据重叠。

## 3. 实验结果

### 3.1 小样本生成配方对照

协议：{gen['protocol']}。ID 相似度评分器：{gen['scorer']}。

{table(['方案', '可评分/生成数', 'id_sim 均值', '最小值'], gen_rows)}

多参考方案在这次小样本对照中取得更高均值和最低值，支持把它作为后续生成配方。
base 的一个生成样本未被评分，比较还可能受检测筛选影响。
该均值 {winner['id_sim_mean']:.4f} 对应 {winner['n_scored']} 张配方对照样本，
不能当作 {synth} 张筛选后训练图的均值。Realistic Vision 组合在本次设置中得分更低，
这不足以证明所有参数与底模版本都不兼容。项目参照点也不是普适身份判定阈值。

![生成配方对照](../site/assets/figures/generator_quality.svg)

### 3.2 E1 与 E4a：均值、种子波动与检验

每格为 {n_seeds} 个种子的均值 ± 样本标准差（ddof=1）。以下检验沿用冻结汇总的双侧 Welch 方法 [12]：

{table(['协议', '指标', 'E1', 'E4a', 'Δ E4a−E1', 't', 'df', '近似 p', '判定'], result_rows)}

这是未作多重比较校正的名义检验；不同协议共享图片和身份，结果并不独立。
小样本对分布假设敏感；原始 Welch 检验也没有把相同数字种子自动视为有效配对设计。
因此“名义显著”需要连同数据重叠、阈值选择及重复数一起解释。

![Accuracy 均值与标准差](../site/assets/figures/accuracy_mean_sd.svg)

![TAR 均值与标准差](../site/assets/figures/tar_mean_sd.svg)

### 3.3 均值差的 95% 区间

区间 = 冻结 Δ ± t(0.975, 各检验 df) × 冻结 SE；与上方种子标准差不同。
冻结 t、df、SE 已舍入，下面 p 与区间都是近似推导值，不覆盖原实验文件。

{table(['协议', '指标', '均值差', 'Welch 95% 区间'], ci_rows)}

![Accuracy 均值差与 95% 区间](../site/assets/figures/accuracy_delta_ci95.svg)

![TAR 均值差与 95% 区间](../site/assets/figures/tar_delta_ci95.svg)

Held-out accuracy 的区间跨零；不能因为 t 较大，就沿用固定临界值宣布显著。
Welch 的临界值由每个检验的 df 决定，不存在统一的“每组 n=3 临界值”。
三个协议下 TAR 区间均跨零；均值向好与确认稳定改善是不同强度的结论。

### 3.4 Held-out TAR 的逐种子差异

{table(['种子', 'E1 TAR', 'E4a TAR', '差值'], seed_rows)}

配对点差的样本标准差为 {statistics.stdev(differences):.6f}，均值为 {statistics.mean(differences):+.6f}。
存在一个负差和两个正差，说明需要检查种子敏感性；不能据此断言真实增益已得到证明。
{bootstrap_note}

{reference_point}

## 4. 方法学讨论

### 4.1 协议改变会改变效应估计

Official TAR 均值差 {tests['official']['tar']['delta']:+.6f}，held-out 为 {tests['heldout']['tar']['delta']:+.6f}。
前者更大，且 official 已知包含训练身份；但两协议还改变了配对数量与构成，
不能把全部差异因果归于身份泄漏，也不能把两个均值差的比值当作无偏的泄漏量估计。
两个 TAR 检验都未达显著。

### 4.2 FAR 的经验分辨率

Held-out 清单计划包含 {held['n_same']:,} 个正对与 {held['n_diff']:,} 个负对；检测后有效数另见前表。
经验 FAR 的步长为 1/N_negative；负对数较少时，低 FAR 阈值受少数尾部样本影响。
增加有效负对能改善分辨率，但不会消除训练随机性、身份相关性、阈值校准偏差或域偏移。
当前证据支持优先增加训练重复并开展独立数据验证，不支持“继续增加评测数据完全无意义”。

### 4.3 指标与质量代理不能互相替代

Accuracy@best 与低 FAR 下 TAR 反映不同工作点，前者提高不能推出后者稳定提高。
ID 相似度反映特定嵌入空间中的身份一致性，不等于照片真实度、多样性、公平性或隐私安全。
条件生成与筛选共用的模型会引入度量依赖，需独立识别骨干和外部数据补充验证。

### 4.4 工程经验的证据范围

参数顺序、并列分数、空输入、缺少类别和 FAR 分辨率等问题已经加入评测器测试。
质量与冗余的权衡，以及训练稳定性排查记录，可以指导复核；
但本报告不把缺少对应完整消融日志的历史失败次数，或一次改动前后的训练 accuracy，写成已证实的因果效应。

## 5. 局限与后续验证

1. **身份重叠与同源评测。** Official 有项目训练身份重叠；filtered/held-out 缓解该问题，却仍来自 LFW。
   身份划分可以避免项目内重叠，“泄漏不可避免”不是正确表述；外部预训练重叠尚未完整审计。
2. **测试集阈值选择。** Accuracy@best 和 TAR 的阈值均使用当前评测分数，缺少完全独立的校准集。
   下一步应固定独立校准阈值，再在留出集上报告指标 [2]。
3. **训练重复不足。** 各主方案只有 {n_seeds} 个种子，TAR 尚不显著，held-out accuracy 亦未达到双侧 0.05。
   不显著不等于无效，也不能声称已确认真实增益；目前没有充分功效分析证明某个固定种子数必然够用。
4. **统计假设与多重比较。** 小样本 Welch 和多个共享数据的检验具有局限。
   需预先确定主要终点，并在更多重复下检查适当的配对或稳健分析，而非事后选择显著方法。
5. **相关配对与重采样。** 同一身份可参与多个验证对；按对 bootstrap 不等于按身份或人物簇重采样。
   固定种子的重采样区间也不包含重新训练的不确定性。
6. **比例与筛选消融不足。** 主方案合成占比为 {ratio:.1f}%；E4b 仅有一次运行，缺少完整比例扫描。
   样本数量与质量门槛同时变化，不能独立证明“越多越好”或“越相似越好”。
7. **规模与架构有限。** 当前仅 {ids} 个训练身份、一个自训练识别骨干。
   与大规模预训练模型的差距还有数据分布、模型训练和预算等因素，不能唯一归因于样本量。
8. **生成质量度量依赖。** 配方对照样本少；生成条件和筛选所用身份模型存在关联，
   应加入独立度量，并报告失败样本及不同人物的表现 [7–9]。
9. **公平性与跨域未评估。** 本项目没有完整的人口统计分组、跨数据集、姿态和年龄鲁棒性测试，
   整体指标不能替代对各群体的影响分析。
10. **隐私与使用许可。** 生成已有真实身份的照片并不自动消除隐私风险；相关研究发现合成人脸可能泄露生成器训练身份 [10]。
    FaceID 作者模型卡明确研究用途限制 [11]；本报告不把代码许可当作模型、照片或衍生数据的授权。

## 6. CPU 复现与展示重建

### 6.1 已完成的 CPU E0 验证

评测时间：{cpu['evaluated_at']}；Python：{cpu['hardware']['python']}；
识别执行器：{', '.join(cpu['model']['onnx_providers'])}；检测尺寸：{cpu['model']['det_size']}；选脸策略：{cpu['model']['face_select']}。
特征提取成功 {cpu['embedding']['num_used']:,} 张、失败 {cpu['embedding']['num_failed']} 张，
记录耗时 {cpu['embedding']['extract_seconds']:.2f} 秒。

{table(['协议', '指标', '冻结基线', 'CPU 复现'], cpu_rows)}

验收范围是现有六位小数精度的汇总指标一致性，不宣称阈值、检测框或逐对分数逐位一致。
测试日志记录 **{cpu['tests_passed']} passed**；这是保存日志的结果，不代表本脚本重新执行了测试。
证据：[CPU 复现报告](../reports/e0_cpu_reproduction.b.md)、[测试日志](../reports/e0_cpu_reproduction.b.pytest.txt)、
[指标 JSON](../{cpu['source']})、[逐对分数 CSV](../results/reproductions/e0-buffalo_l-lfw-b-cpu/pairs.csv)。

如需重新执行 CPU 推理，先准备项目环境、私有 local 配置、LFW 和模型缓存，再运行：

```powershell
{cpu['command']}
```

已有验证证据无需重跑。`--no-cache` 重新提取特征，输出放在独立复现目录，保留冻结实验结果。
下载配置没有提供可信预期校验和；本机下载摘要用于文件一致性核对，不能单独证明来源真实性。

### 6.2 重建展示与报告

```powershell
python tools/export_site_data.py
python tools/build_site_charts.py
python tools/build_report.py
python tools/build_site_page.py
```

这些脚本读取已有结果，不生成模型样本、不重新训练、不运行 CPU 特征提取。
随后双击 `site/index.html`；复现证据副本随站点提供。
GPU 生成与训练的历史配方见 [W3-GENERATION.md](W3-GENERATION.md) 及对应 configs/exp 配置，
本轮报告整理不重新执行这些实验。

## 7. 交付状态

| 交付物 | 位置 | 当前状态 |
| --- | --- | --- |
| 冻结生成与识别结果 | results/runs/ | 保留原记录 |
| 离线网站、图表与视觉对照 | site/index.html、site/assets/ | 已构建，CPU 栏目由脚本更新 |
| 评测器修复与边界测试 | src/aigcfr/eval/verify.py、tests/ | 已有合并记录 |
| CPU E0 复现证据 | results/reproductions/、reports/e0_cpu_reproduction.b.* | 已完成 |
| 报告数字与统计核对 | 本文、reports/report_numeric_audit.b.md | 本次生成 |
| 外部读者三分钟验收 | #25 验收项 | 暂未记录完成 |
| 排版导出 PDF / DOCX | reports/ | 下一步进行，本文不提前标为完成 |
| issue 索引同步 | docs/issues/00-INDEX.md | 待核对实际同步脚本与状态 |

## 8. 参考文献

以下为实际引用的原始论文、报告和官方资料。外部文献支持方法或风险讨论，不提供本项目实验数字。
作者、题名与来源已核对；实现文档访问日期为 2026-10-07。

{bibliography}

## 附录：数字与来源

训练数量与运行规模来自各 metrics 的 train_set、model、protocol_detail 与 benchmarks。
Held-out TAR 专门读取 `benchmarks.lfw_heldout.tar_at_far_full['1e-03'].tar`。
跨种子统计取 w4 comparison 的 arms[*].protocols 和 tests；配方表取 w3 comparison 的 final_recipe。
CPU 表取独立复现 metrics，与 E0 冻结记录对照；passed 数来自保存的 pytest 文本日志。
LFW 全量规模来自文献 [1]。本次没有把旧稿未核实的筛选均值或类别比例伪装成 JSON 结果。

导出来源及摘要：

{sources}
'''


def audit(data, original_digest):
    held = data["derived_tests"]["heldout"]["accuracy"]
    return f'''# B 报告数字与表述核对（#27）

原始稿保存在 [report_initial.b.md](report_initial.b.md)，SHA-256：`{original_digest}`。
更新后的报告由 tools/build_report.py 自动生成，主结果读取 site/data/site_data.json，并校验全部来源摘要。

| 原表述 | 本次处理与证据 |
| --- | --- |
| held-out accuracy 显著；固定 n=3 临界值 | 改为未达双侧显著；t={held['t']:.4f}、df={held['df']:.3f}、p≈{held['p_two_sided']:.6f}，95% 区间 [{held['ci95_low']:+.6f}, {held['ci95_high']:+.6f}] |
| 各协议 TAR 真正增益已存在 | 改为均值差为正但未显著；不把趋势视为已确认效应 |
| 配方均值与筛选后 755 张均值混用 | 配方对照只引用冻结 w3 JSON；旧稿的整批精确均值不在当前导出字段，暂不用于量化结论 |
| 最大类 538（12.4%）、最小类 23 用于真实集 | 口径未核实，不继续引用精确数；538/3590 与 538/4345 分母不同，不能混用 |
| 96 个训练身份按图片数取前 96 | 报告实际训练规模与 min_images=15 过滤记录，名单以训练 manifest 为准 |
| official-10fold 直接等于文献成绩 | 明确主指标为测试集 Accuracy@best，独立验证阈值未完成 |
| held-out 125191 即所有指标有效数 | 区分计划清单、accuracy 平衡子集与 TAR 全量有效对子 |
| 旧稿各 seed bootstrap 区间 | 当前仅逐项核实冻结汇总 seed0；其余改用可核对的逐种子点差，不混称 90%/95% 区间 |
| 所有不确定度均来自种子；再加评测无意义 | 改为分辨率与训练随机性同时存在；补充身份相关、阈值偏差和跨域限制 |
| 8 个种子必然足够、底模确认不兼容 | 删除缺乏功效分析或完整参数搜索支持的确定表述 |
| E4b 也有三个种子 | 仅记录一个 seed0，不能作为同强度对照 |
| 网站待建、CPU 复现后续完成 | 按已有证据更新，网页及生成脚本同时同步 |

未覆盖冻结 results/runs/，未重新生成或训练模型，未重新运行 CPU 推理或 pytest。
当前测试 passed 数引用既有日志；PDF/DOCX 排版和 issue 索引更新尚待后续步骤。
'''


def layout_verified(root, base_report):
    """Check both artifact bytes and the Markdown input used for their export."""
    manifest_path = root / "reports/report_exports.b.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    for name in ("AIGCFR_report.b.docx", "AIGCFR_report.b.pdf"):
        path = root / "reports" / name
        record = manifest["exports"][name]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Report export missing or changed: {name}; restore the checked export package")
    normalized = base_report.replace("\r\n", "\n").replace("\r", "\n")
    candidates = set()
    for text in (normalized, normalized.replace("\n", "\r\n")):
        for encoding in ("utf-8", "utf-8-sig"):
            candidates.add(hashlib.sha256(text.encode(encoding)).hexdigest())
    return manifest.get("source_sha256") in candidates


def index_verified(root):
    snapshot_path = root / "reports/issue_index_snapshot.b.json"
    path = root / "docs/issues/00-INDEX.md"
    if not snapshot_path.is_file() or not path.is_file():
        return False
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8-sig"))
    if (snapshot.get("repo") != "DrGlitch666/AIGC-in-Face-Recognition-Dataset-Augmentation"
            or snapshot.get("request_method") != "GET"
            or snapshot.get("pull_requests_excluded") is not True
            or snapshot.get("issue_count") != len(snapshot.get("issues", []))):
        return False
    return snapshot.get("index_sha256") == hashlib.sha256(path.read_bytes()).hexdigest()


def delivery_status(root, base_report):
    ready_layout = layout_verified(root, base_report)
    ready_index = index_verified(root)
    text = base_report
    if ready_layout:
        text = text.replace(
            "| 排版导出 PDF / DOCX | reports/ | 下一步进行，本文不提前标为完成 |",
            "| 排版导出 PDF / DOCX | reports/AIGCFR_report.b.pdf 与 .docx | 已导出，实验数字沿用本文 |",
        )
    elif (root / "reports/report_exports.b.json").is_file():
        text = text.replace("下一步进行，本文不提前标为完成", "当前报告与导出源不同，需重新排版导出")
    if ready_index:
        text = text.replace(
            "| issue 索引同步 | docs/issues/00-INDEX.md | 待核对实际同步脚本与状态 |",
            "| issue 索引同步 | docs/issues/00-INDEX.md；reports/issue_index_snapshot.b.json | 已通过 GET 读取实际 Issue；状态为快照，排除 PR |",
        )
    return text, ready_layout, ready_index




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        data = json.loads((root / "site/data/site_data.json").read_text(encoding="utf-8-sig"))
        validate_export(root, data)
        original = root / "reports/report_initial.b.md"
        if not original.is_file():
            raise ValueError("Missing preserved original report; run apply_report_update.py first")
        report, layout_ok, index_ok = delivery_status(root, render(data))
        checks = audit(data, hashlib.sha256(original.read_bytes()).hexdigest())
        status = f"当前测试 passed 数引用既有日志；PDF/DOCX：{'已校验导出源与文件摘要' if layout_ok else '待重新核对或导出'}；Issue 索引：{'已校验 GET 快照（排除 PR）' if index_ok else '待执行只读导出'}。"
        checks = checks.replace("当前测试 passed 数引用既有日志；PDF/DOCX 排版和 issue 索引更新尚待后续步骤。", status)
        (root / "docs/REPORT.md").write_text(report, encoding="utf-8")
        (root / "reports/report_numeric_audit.b.md").write_text(checks, encoding="utf-8")
    except (OSError, ValueError, TypeError, KeyError, StopIteration) as exc:
        print(f"Report failed: {exc}", file=sys.stderr)
        return 1
    print("Report OK: docs/REPORT.md")
    print("Audit OK: reports/report_numeric_audit.b.md")
    print(f"References: {len(REFERENCES)} (original papers and official sources)")
    print(f"PDF/DOCX: {'verified' if layout_ok else 'pending or stale'}; issue index: {'verified GET snapshot' if index_ok else 'pending'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
