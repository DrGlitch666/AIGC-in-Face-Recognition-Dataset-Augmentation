---
title: "[基线] 用预训练模型跑通 1:1 人脸验证评测（E0）"
slug: "05-zeroshot-baseline"
labels: ["优先级:P0", "类型:评测", "难度:中等"]
milestone: "M1 基线与评测"
---

## 背景 / 为什么做这个

在写任何训练代码、碰任何生成模型之前，必须先证明**评测环节本身是对的**。

做法：拿一个公开的预训练人脸识别模型（不训练），在标准评测集上跑一次验证，看结果是否落在公开报道的量级内。如果这个数字正常，说明你的对齐、取特征、算相似度、配对协议都是对的；如果偏低，那就**先修评测，别往下做**——否则后面所有「合成数据有没有用」的结论都建立在坏掉的尺子上。

这也顺便给你一个性能天花板的概念（E0），后面自训练模型（E1）会明显低于它，这是正常的。

## 目标

一条命令拿到 `results/runs/e0-zeroshot/metrics.json`，且数字落在合理区间。

## 任务清单

- [ ] 下载 insightface 官方预训练包（`buffalo_l` 或 `antelopev2`），记录包名与下载日期
- [ ] 写 `src/aigcfr/eval/embed.py`：读图 → 对齐 112×112 → 提取 512 维 embedding → **L2 归一化**
- [ ] 写 `src/aigcfr/eval/verify.py`：解析官方 pair 协议（LFW 6000 对：同人/异人各 3000）→ 算余弦相似度 → 全阈值扫 → 找最佳阈值下的 Accuracy
- [ ] 正确实现两个容易搞错的细节：
  - [ ] **10 折交叉验证**（LFW 标准做法）：在 9 折上选阈值，在第 10 折上测；报 10 折的平均与标准差
  - [ ] **flip 测试增强**：原图与水平翻转图的 embedding 取平均（可选，但要和后续实验保持一致）
- [ ] 在 LFW / CFP-FP / AgeDB-30 上各跑一次（没有的数据集见 {{#03-data-pipeline}} 的降级说明）
- [ ] 输出 `results/runs/e0-zeroshot/metrics.json`，**字段严格符合** `docs/FRAMEWORK.md` §3.3 契约 B
- [ ] 画 ROC 曲线与相似度分布直方图（同人/异人两条分布）存到 `reports/figures/`
- [ ] 写 `reports/zeroshot_report.md`：本机实测数字 vs 公开报道量级 + 差异原因分析

## 验收标准（Definition of Done）

- [ ] LFW 准确率落在公开报道量级内（同类预训练模型通常在 **99.8% 左右**，以你的实测和 {{#04-background-survey}} 整理的资料为准；差异 > 1 个百分点就要查错）
- [ ] `metrics.json` 通过 schema 校验，含 `n_pairs`、`metric`、`value`、`seed`、`config_hash`
- [ ] 同一份代码换评测集**只需改配置**，不改代码
- [ ] ROC 与分布图已落盘，图上能看到同人/异人两个分布明显分离
- [ ] 报告中明确写出「本机数字 / 公开数字 / 差异原因」

## 交付物

- `src/aigcfr/eval/{embed,verify}.py`
- `scripts/evaluate.py`（第一版）
- `configs/eval/*.yaml`
- `results/runs/e0-zeroshot/metrics.json`
- `reports/zeroshot_report.md`、`reports/figures/roc_e0.png`

## 依赖

- 需要 {{#03-data-pipeline}}（数据与 pair 协议）
- 需要 {{#01-env-setup}}（onnxruntime-gpu）
- 被依赖：{{#07-evaluator}}（本任务产出的打分代码会被它泛化）、{{#14-mix-ratio-matrix}}（E0 是矩阵里的一行）

## 预估工作量

半天到 1 天

## 参考

- [insightface 官方仓库](https://github.com/deepinsight/insightface)（模型包下载与 `FaceAnalysis` 用法）
- [ONNX Runtime CUDA Execution Provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)（确认真的在用 GPU）
- [LFW 官方页面](http://vis-www.cs.umass.edu/lfw/)（6000 对协议与 10 折划分的出处）
- 关键概念：embedding 必须**先 L2 归一化**再算余弦相似度（归一化后余弦 = 内积，很多库默认不归一化，是常见错误来源）

## 附：已核实要点（框架预置，2026-09）

**模型包**：下载地址形如 `https://github.com/deepinsight/insightface/releases/download/model-zoo/<name>.zip`

| 模型包 | 识别网络 | 大小 | LFW | CFP-FP | AgeDB-30 |
|---|---|---|---|---|---|
| `buffalo_l`（默认） | ResNet50 @ WebFace600K | 275 MB | **99.83** | **99.33** | **98.23** |
| `antelopev2` | ResNet100 @ Glint360K | 344 MB | — | — | — |
| `buffalo_s` / `buffalo_sc` | MobileFaceNet | 122 / 14.3 MB | — | — | — |

上表数字来自 insightface 官方 model zoo README，**就是你在验收标准里要对照的量级**。

- ⚠️ **许可**：insightface 的**代码**是 MIT，但**模型包仅限非商业研究用途**——写进 issue #18 的 `docs/ETHICS.md`。
- ⚠️ 官方 README 里另有一组按人群分组的数字（African / Caucasian / South Asian / East Asian），**那不是 RFW**，引用时别标错。
- ⚠️ `FaceAnalysis()` 的 provider 顺序是 CoreML → CUDA → CPU，**务必打印 providers 确认真的在用 GPU**。
- 详细版见 [`docs/REFERENCES.md`](../REFERENCES.md) 第 2 节。

## 新手提示 / 卡住了怎么办

- 先用 **10 对图片**人工验证：同一人的两张图相似度应明显高于不同人；这一步花 10 分钟能省一天。
- `onnxruntime` 默认可能跑在 CPU 上：打印 `providers` 确认，CPU 上跑 6000 对会慢到怀疑人生。
- **降级方案**：只做 LFW（6000 对），CFP-FP / AgeDB 标记为 TODO，不影响整体推进。
