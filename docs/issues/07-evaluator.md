---
title: "[评测] 统一评测器：1:1 / 1:N / 分桶鲁棒性 / 公平性 + 结果 schema"
slug: "07-evaluator"
labels: ["优先级:P0", "类型:评测", "难度:中等"]
milestone: "M1 基线与评测"
---

## 背景 / 为什么做这个

{{#05-zeroshot-baseline}} 里的评测代码是「为 LFW 定制的」。但项目后面要跑几十个实验、每个实验都要出一整套数字，还必须保证**所有实验用的是同一把尺子**——否则实验之间不可比。

这一步把评测泛化成一个**唯一入口**：给它一个 run 目录，它输出一份符合契约的 `metrics.json`。网站的所有图表都从这个文件生成。

## 目标

`python scripts/evaluate.py --run results/runs/<exp_id>` 一条命令产出完整评测报告。

## 任务清单

- [ ] 1:1 验证（复用并泛化 {{#05-zeroshot-baseline}} 的代码）
  - [ ] LFW（6000 对，10 折交叉验证）
  - [ ] CFP-FP（前后姿态）
  - [ ] AgeDB-30（跨年龄）
  - [ ] 可选：CALFW、CPLFW
- [ ] 1:N 闭集识别：用 toy 测试集（**身份与训练集零重叠**），算 Rank-1 / Rank-5
- [ ] 分桶评测（鲁棒性）：
  - [ ] 按质量分桶（用 Laplacian 方差或 NIQE 分三档）
  - [ ] 按姿态分桶（用关键点估计 yaw 角分三档：正脸 / 半侧 / 大侧脸）
  - [ ] 可选：按分辨率分桶（原图 vs 下采样后放大）
- [ ] 公平性评测（用 RFW 的 4 个人群分组），输出每组准确率 + **最大组间差**（max-min gap）
- [ ] 输出 `metrics.json` 严格符合 `docs/FRAMEWORK.md` §3.3 契约 B，并写 **schema 校验器**（`src/aigcfr/eval/schema.py`）
- [ ] 写 `tests/test_metrics.py`：用**手工构造的假 embedding**（已知答案）验证 Accuracy / TAR@FAR / Rank-1 的实现正确性
- [ ] 支持 TAR@FAR（例如 FAR=1e-3）：当评测对数量不足时明确标注「样本量不足以估计该 FAR」
- [ ] 生成图表：每个实验的 ROC、相似度分布、分桶柱状图 → `results/runs/<exp_id>/figures/`

## 验收标准（Definition of Done）

- [ ] `pytest tests/test_metrics.py` 全部通过（指标实现经得起手工验算）
- [ ] 一条命令对任意 run 产出 `metrics.json` + 图，**不接受手工填数字**
- [ ] schema 校验器能拦住缺字段/类型错误的结果文件（故意改坏一个文件，校验应报错）
- [ ] 用 E0（{{#05-zeroshot-baseline}}）和 E1（{{#06-arcface-training}}）各跑一次，验证结果与之前的手工结果一致
- [ ] 公平性结果里包含「最大组间差」，并写明评测集与分组来源
- [ ] 评测速度可接受（LFW 6000 对在 GPU 上应在 1 分钟内完成）

## 交付物

- `src/aigcfr/eval/{verify,identify,buckets,fairness,schema}.py`
- `scripts/evaluate.py`（完成版）
- `configs/eval/default.yaml`（定义要跑哪些 benchmark）
- `tests/test_metrics.py`
- `results/runs/*/metrics.json`（E0、E1）

## 依赖

- 需要 {{#05-zeroshot-baseline}}（打分与协议解析的第一版）
- 需要 {{#06-arcface-training}}（E1 模型拿来验证评测器）
- 被依赖：{{#08-classic-augmentation}}、{{#14-mix-ratio-matrix}}、{{#15-results-conclusion}}、{{#17-site-content}}

## 预估工作量

1~1.5 天

## 参考

- [insightface 官方仓库](https://github.com/deepinsight/insightface)（`verification.py` 是标准评测实现，建议对照阅读）
- [LFW 官方页面](http://vis-www.cs.umass.edu/lfw/)
- [RFW 官方页面](https://www.whdeng.cn/RFW/index.html)（种族偏见评测集，4 个人群分组）
- [IQA-PyTorch (pyiqa)](https://github.com/chaofengc/IQA-PyTorch)（NIQE / BRISQUE 等画质指标）

## 附：已核实要点（框架预置，2026-09）——评测协议的标准做法

照着 insightface `verification.py` 的既定做法实现，可以避免 90% 的坑：

1. **10 折协议**：`KFold(n_splits=10, shuffle=False)`；**阈值在每折的训练半上选，在该折的测试半上测**，最后报 10 折平均。
2. **距离**：在 **L2 归一化**后的 embedding 上算**平方 L2 距离**（等价于余弦距离的单调变换）。
3. **flip test**：原图与水平翻转图的 embedding **相加后重新归一化**，报告为 "Accuracy-Flip"——训练与评测都要保持一致。
4. **阈值网格**：Accuracy 用 `np.arange(0, 4, 0.01)`；**TAR@FAR=1e-3 要用更细的步长（0.001）**，否则 FAR 估不准。
5. **对齐模板**（112×112 的 5 点标准位置，`norm_crop` 用的就是这组）：
   `[(38.2946,51.6963), (73.5318,51.5014), (56.0252,71.7366), (41.5493,92.3655), (70.7299,92.2041)]`
   **真实图与合成图必须走完全相同的检测 + 对齐流程。**
6. **泄漏审计**：LFW / CALFW / CPLFW **共享身份**；MS1M/WebFace 与 LFW 也有重叠，公开的重叠名单可用于审计。**永远不要在测试集上调参。**
7. **LFW 会饱和**：好模型普遍 99.8%+，0.1% 的差异就是噪声。**务必加一个不易饱和的指标**：TinyFace（1:N Rank-1，可直接下载，约 148 MB）或你自己的 1:N 闭集测试。
8. **≥3 个随机种子，报均值±标准差**。

详细版见 [`docs/REFERENCES.md`](../REFERENCES.md) 第 7 节。

## 新手提示 / 卡住了怎么办

- **测试先行**：先写 `test_metrics.py`（用 3 个向量手工算余弦相似度，人工确定正确答案），再写实现。这是防止指标算错最省事的办法。
- 「公平性」这个词要小心：你测的是**模型在不同人群子集上的表现差异**，不是你证明了模型有偏见。措辞要克制（这也是 {{#15-results-conclusion}} 的要求）。
- RFW 需要申请或找镜像，拿不到就降级：只报告分桶结果，公平性一节写「未能获取数据集」。
- **降级方案**：先做 LFW + 1:N + 质量分桶，姿态/公平性分开补。
