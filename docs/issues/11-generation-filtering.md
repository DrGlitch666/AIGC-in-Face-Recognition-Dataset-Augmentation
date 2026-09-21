---
title: "[筛选] 生成质量与身份一致性门禁（漏斗）★核心"
slug: "11-generation-filtering"
labels: ["优先级:P0", "类型:筛选", "难度:中等"]
milestone: "M2 生成与筛选"
---

## 背景 / 为什么做这个

**生成数据不是越多越好，而是越干净越好。** 这是这个领域反复被验证的经验：不做筛选就把生成图混进训练集，很可能**反而让性能下降**——因为里面混着身份漂移的图（长着别人的脸却标着张三的标签）、糊图、畸变图，这些是**带毒的训练样本**。

筛选层就是数据集的**质检门禁**：每张生成图都要过几道检查，通过了才允许进入训练集，不通过的记录原因（用于分析生成器的弱点）。

这一层做得好不好，往往直接决定 {{#14-mix-ratio-matrix}} 的结论是正是负。

## 目标

一条命令把生成的原始图片变成「可用/不可用」两个清单，并给出可解释的漏斗报告。

## 任务清单

- [ ] 实现 4 类检查器，每个都可独立开关、阈值可配置：
  - [ ] **身份一致性**：生成图 embedding 与该身份参考图**中心**（identity centroid）的余弦相似度 ≥ 阈值
  - [ ] **图像质量**：Laplacian 方差（清晰度）、可选 NIQE/BRISQUE、可选人脸检测置信度
  - [ ] **重复检测**：与同身份已有图的 embedding 最近邻相似度 > 阈值 → 判为「重复，无信息量」，丢弃或降权
  - [ ] **结构有效性**：人脸检测数量必须为 1；关键点合理（不歪不斜）
- [ ] **阈值不是拍脑袋定的**，要用数据选：
  - [ ] 扫一遍阈值（例如 0.2~0.7），画「通过率 vs 阈值」曲线和「通过图平均相似度 vs 阈值」曲线
  - [ ] 选一个平衡点，并在报告里写明选择依据
  - [ ] 在**真实图**上做一次同样的打分，作为「上限参考」（真实图之间也有相似度分布，生成图应该尽量接近它）
- [ ] 写 `scripts/filter.py`：读生成 manifest → 输出 `status=accepted/rejected` + `meta.reject_reason` + `meta.id_sim` + `meta.quality`
- [ ] 生成漏斗报告 `reports/filter_report.md`：
  - [ ] 生成 N 张 → 通过 M 张（通过率）
  - [ ] 每种拒绝原因的数量与占比（饼图/条形图）
  - [ ] 被拒样本墙 `reports/figures/rejected_samples.png`（**必须人工看一遍**，确认拒绝得合理，而不是误杀）
- [ ] 抽检通过的图：随机 20 张人工确认「确实像本人」
- [ ] 输出筛选后的清单 `data/manifests/synth_e4_filtered.jsonl`（供 {{#14-mix-ratio-matrix}} 使用）
- [ ] 写 `tests/test_filter.py`：构造已知相似度/质量分的假数据，验证阈值判定逻辑

## 验收标准（Definition of Done）

- [ ] `python scripts/filter.py --config configs/filter/default.yaml` 一条命令完成，输出清单与报告
- [ ] 阈值选择有曲线图为依据（不是拍脑袋）
- [ ] 漏斗报告里能看到每种拒绝原因的数量，且**拒绝原因可解释**
- [ ] 被拒样本墙已人工检查，报告中写明「误杀率」的主观估计
- [ ] 通过率 ≥ 30%（过低说明生成器有问题，应回到 {{#10-identity-preserving-generation}} 而不是降低阈值）
- [ ] `tests/test_filter.py` 通过
- [ ] 筛选后的清单可直接被训练脚本消费（不做任何手工整理）

## 交付物

- `src/aigcfr/filter/{identity,quality,dedup,structure,funnel}.py`
- `scripts/filter.py`、`configs/filter/default.yaml`
- `data/manifests/synth_e4_filtered.jsonl`
- `reports/filter_report.md`、`reports/figures/{filter_threshold_curve,funnel,rejected_samples}.png`
- `tests/test_filter.py`

## 依赖

- 需要 {{#10-identity-preserving-generation}}（待筛选的生成图）
- 需要 {{#05-zeroshot-baseline}}（ArcFace 模型用于算身份相似度）
- 被依赖：{{#12-synth-detection-report}}、{{#14-mix-ratio-matrix}}、{{#17-site-content}}（漏斗图会在网站上展示）

## 预估工作量

1~1.5 天

## 参考

- [IQA-PyTorch (pyiqa)](https://github.com/chaofengc/IQA-PyTorch)（NIQE / BRISQUE / 美学分等无参考画质指标）
- [clean-fid](https://github.com/GaParmar/clean-fid)（FID 计算，若要做整体分布对比）
- [torchmetrics](https://github.com/Lightning-AI/torchmetrics)（KID 等分布指标）
- 关键概念：**阈值筛选会改变数据分布**，因此筛选前后都要报告「图片数、平均相似度、平均质量分」，否则无法解释实验差异来自哪里

## 附：已核实要点（框架预置，2026-09）

**指标选择（方向别搞反）**

| 指标 | 方向 | 需要参考图 | 库 / 许可 |
|---|---|---|---|
| FID | 越低越好 | 不需要（但需要真实集，经验值 ≥1 万张） | clean-fid（MIT） |
| KID | 越低越好，**无偏** | 不需要 | torchmetrics / clean-fid |
| NIQE / BRISQUE | 越低越好 | 不需要 | **pyiqa（PolyForm Noncommercial，非商业许可）** / OpenCV contrib |
| Laplacian 方差 | **越高越清晰** | 不需要 | OpenCV 一行代码 |

**两个必须避开的坑**：
1. **FID 对实现极其敏感**：同一组图，仅改变缩放滤波器就能得到 ≥6 与 ≤0.75 的差别。**两个实验组必须用同一份实现重跑**，报告时要写清用的哪个库、哪组参数。
2. **不要把「Laplacian 方差 ≈ 100」之类的民间阈值当标准**——没有任何权威来源验证过画质指标能代表生成人脸的「真实感」。**用你自己的数据选阈值（本任务要求的阈值扫描曲线就是干这个的）。**

**许可提醒**：`pyiqa` 是 PolyForm Noncommercial，学术研究可用，商用需替换实现——记进 issue #18 的 `docs/ETHICS.md`。

详细版见 [`docs/REFERENCES.md`](../REFERENCES.md) 第 4 节。

## 新手提示 / 卡住了怎么办

- 先用最简单的三个指标（身份相似度 + Laplacian 方差 + 重复检测）就能覆盖大部分问题，NIQE 这类库可以后加。
- **务必看被拒样本**：如果被拒的都是好图，说明阈值或指标选错了，这比不筛选更糟。
- 一个常见陷阱：用**训练集的 ArcFace 模型**去算「身份相似度」，等于用自己判自己。尽量用独立的预训练模型（{{#05-zeroshot-baseline}} 的那个）来打分。
- **降级方案**：只保留「身份一致性」一个筛选器，阈值定在 0.35~0.45 之间的经验值，并在报告里注明「未做多指标筛选」。
