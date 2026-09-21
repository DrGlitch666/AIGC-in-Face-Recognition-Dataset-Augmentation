---
title: "[实验] 传统数据增强对照组（E2）：翻转/裁剪/色彩/模糊/降质"
slug: "08-classic-augmentation"
labels: ["优先级:P1", "类型:实验", "难度:入门"]
milestone: "M2 生成与筛选"
---

## 背景 / 为什么做这个

这是全项目**最容易被忽略、但最重要**的对照组。

如果 E4（真实图 + AIGC 合成图）比 E1（只用真实图）好，别人会问：**「是因为 AIGC 特别好，还是仅仅因为你喂了更多图片？」**

E2 就是回答这个问题的：用**最土的传统增强**（翻转、随机裁剪、色彩抖动、模糊、压缩、降分辨率）把训练集扩充到**和 E4 完全相同的图片总数**。如果 E2 和 E4 效果差不多，那就说明「多喂图」本身就有效，AIGC 的独特价值需要重新论证——**这本身就是一个有价值的结论**。

顺便，E2 也是「筛选层」（{{#11-generation-filtering}}）的对照组：传统增强不需要任何质量筛选，如果 AIGC 加了筛选还比不过它，说明生成质量是瓶颈。

## 目标

产出 E2 的完整数字，并且**图片总数与 E4 严格对齐**（这是实验有效性的前提）。

## 任务清单

- [ ] 写 `src/aigcfr/generate/classic.py`：实现增强算子
  - [ ] 几何：水平翻转、随机小角度旋转、随机缩放裁剪
  - [ ] 光度：亮度/对比度/饱和度抖动、高斯模糊、高斯噪声
  - [ ] 降质：JPEG 压缩、下采样再上采样（模拟低分辨率监控画面）
  - [ ] 组合增强（随机选 2~3 种叠加）
- [ ] 实现统一接口，与后续生成器一致（见 `docs/FRAMEWORK.md` §6 的 `FaceGenerator`）：入参为「参考图 + 每身份生成数量」，出参写入 manifest（`source=aug`）
- [ ] 写 `configs/gen/classic.yaml`：增强类型、强度范围、每身份生成数量
- [ ] 让**每身份生成数量**可配置，并通过配置与 E4 对齐图片总数（例：20 个身份 × 每身份 10 张真实图 + 每身份 5 张增强 = 与 E4 相同规模）
- [ ] 生成 E2 训练集 manifest，跑统计确认：**身份数相同、图片总数相同、仅来源不同**
- [ ] 用 {{#06-arcface-training}} 训练 E2（≥1 seed，理想 3 seed）
- [ ] 用 {{#07-evaluator}} 评测，落 `results/runs/e2-classic-aug-seed*/metrics.json`
- [ ] 在 `docs/RESULTS.md` 表格里填一行，写一句结论（数量 vs 多样性）
- [ ] 生成对比图 `reports/figures/e2_aug_samples.png`（原图 vs 各增强算子结果，4×4）

## 验收标准（Definition of Done）

- [ ] E2 训练集的**身份数**和**图片总数**与目标实验完全一致（在报告里贴出两个数字对比表）
- [ ] 除数据来源外，训练超参与 E1/E4 **完全相同**（同一个配置文件模板，只改数据段）
- [ ] E2 的 `metrics.json` 符合契约，且已汇总进 `results/summary.csv`
- [ ] `docs/RESULTS.md` 里有一行 E2 结果 + 一句「与 E1 相比如何」
- [ ] 增强样本图已落盘（人工确认增强没有把脸糊到认不出）

## 交付物

- `src/aigcfr/generate/classic.py`
- `configs/gen/classic.yaml`、`configs/exp/e2-classic-aug-seed*.yaml`
- `results/runs/e2-classic-aug-seed*/metrics.json`
- `reports/figures/e2_aug_samples.png`
- `docs/RESULTS.md` 中对应行

## 依赖

- 需要 {{#06-arcface-training}}（训练管线）
- 需要 {{#07-evaluator}}（评测）
- 需要 {{#02-repo-skeleton}}（`FaceGenerator` 接口与配置约定）
- 与 {{#10-identity-preserving-generation}} **规模必须对齐**，建议先做 E2 再做 E4，把「图片总数」这个数字提前定下来

## 预估工作量

半天

## 参考

- 传统人脸识别增强的常见做法可参考 [insightface 数据增强实现](https://github.com/deepinsight/insightface)
- [Albumentations](https://github.com/albumentations-team/albumentations)（如果想省事，可以直接用这个库实现增强算子；注意保持「训练与评测对齐方式一致」）
- 关键概念：**类内多样性（intra-class diversity）**——同一个人的图片之间差异越大，模型越能学到泛化；E2 提供的是「像素级多样性」，E4 提供的是「语义级多样性」，这是两者最本质的区别

## 新手提示 / 卡住了怎么办

- 增强**只作用于训练集**，验证/评测集绝不做增强（除了 {{#05-zeroshot-baseline}} 提到的 flip-test）。
- 「图片总数对齐」是这个实验的灵魂。写报告时把两个数据集的行数并排贴出来。
- **降级方案**：时间紧张时，E2 只跑 1 个 seed，并在结论里注明「单 seed，未做方差估计」。
