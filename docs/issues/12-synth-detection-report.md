---
title: "[合规] 合成人脸可检测性报告：我们的图能被认出来是 AI 生成的吗？"
slug: "12-synth-detection-report"
labels: ["优先级:P2", "类型:合规", "难度:中等"]
milestone: "M2 生成与筛选"
---

## 背景 / 为什么做这个

这个项目有一个绕不开的伦理张力：

- **好的一面**：合成数据不包含真实人物的生物特征，是隐私友好的数据来源。
- **危险的一面**：同一种技术可以用来伪造他人的脸，用于欺诈、绕过身份验证。

作为做这个项目的人，**有责任把「我们的生成图有多容易被检测出来」这件事测清楚并公开**。这既是对技术的诚实评估，也是网站上一个能让项目区别于普通 demo 的章节。

同时它还有科学价值：不同生成器的「可检测性」差异，本身反映的就是**域差距**——越容易被检测，说明生成图和真实图的分布差得越远，也就越可能拖累识别模型。

## 目标

用现成的合成图检测器，量化「真实图 vs 我们的生成图」的可分性，并给出结论与建议。

## 任务清单

- [ ] 选 **2 个**现成检测器（要求一：能直接跑、不需要自己训练；要求二：至少一个不是专门针对我们用的生成器的）
  - 候选方向：HuggingFace 上的通用「AI 生成图检测」模型、频域/指纹类方法、通用假图检测框架
- [ ] 构建评测集：真实图 N 张（来自 {{#03-data-pipeline}}）+ 生成图 N 张（来自 {{#11-generation-filtering}} 筛选通过的），**数量对齐**（各 200 张级别即可）
- [ ] 跑检测，计算：
  - [ ] **AUC**（真实 vs 生成的区分度，0.5 = 分不出来，1.0 = 完美区分）
  - [ ] 在固定误报率下的检测率
  - [ ] 按生成器分组对比（如果做过多个生成器）
- [ ] 分析：把检测器打分与筛选层的指标（身份相似度、质量分）做相关性分析——**质量越高的生成图是否越难被检测？**
- [ ] 做一次**交叉生成器测试**：用 A 生成器数据训练的检测器，去测 B 生成器的图，看检测率是否下降（这体现检测器的泛化问题）
- [ ] 写 `reports/detection_report.md`：数字 + 直方图 + 结论 + 局限
- [ ] 在 `docs/ETHICS.md`（{{#18-ethics-and-release}}）里留一段「本项目的生成图可检测性」摘要与建议

## 验收标准（Definition of Done）

- [ ] 至少 2 个检测器的 AUC 结果已产出（附模型来源与版本）
- [ ] 打分分布直方图已落盘（真实图与生成图两条分布，视觉上能看出重叠程度）
- [ ] 报告里明确回答三个问题：**容易被检测吗？为什么？这意味着什么？**
- [ ] 复现命令清晰（一条命令重跑检测）
- [ ] 结论措辞克制：不写「我们的图无法被检测」这类绝对表述，也不写没有数据支撑的判断

## 交付物

- `src/aigcfr/filter/detectors.py`（检测器适配器）
- `scripts/detect_synth.py`
- `reports/detection_report.md`、`reports/figures/{det_hist,det_auc_by_generator}.png`

## 依赖

- 需要 {{#11-generation-filtering}}（筛选后的生成图清单）
- 需要 {{#03-data-pipeline}}（真实图清单）
- 产出被 {{#18-ethics-and-release}} 引用（`docs/ETHICS.md`）

## 预估工作量

1 天（**可与 {{#14-mix-ratio-matrix}} 的长时间训练并行推进**，训练跑着的时候做这个）

## 参考

- [UniversalFakeDetect 官方仓库](https://github.com/Yuheng-Li/UniversalFakeDetect)（CLIP 特征的通用假图检测，经典基线）
- [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench)（假脸检测的综合性基准框架，内含多种检测器）
- HuggingFace 上的通用 AI 图像检测器（搜索 "AI generated image detector"，注意核对模型卡上的训练数据与适用范围）
- 关键概念：检测器在**训练时没见过的生成器**上通常会显著掉点（泛化差）——这正是「交叉生成器测试」要验证的

## 附：已核实要点（框架预置，2026-09）

**可用的检测器（按推荐顺序）**

| 工具 | 方法 | 备注 |
|---|---|---|
| [UniversalFakeDetect](https://github.com/Yuheng-Li/UniversalFakeDetect)（CVPR'23） | 冻结 CLIP ViT-L/14 + 线性探针 | **首选**，跨生成器泛化的经典基线 |
| [NPR](https://github.com/chuangchuangtan/NPR-DeepfakeDetection)（CVPR'24） | 上采样伪影 | 在不同基准上分数差异很大（GenImage 90.1 vs SIDBench 52.15），**注意评测设置** |
| [Synthbuster](https://github.com/qbammey/synthbuster) | 傅里叶伪影 + 随机森林（**非深度网络**） | 方法正交，适合当第二个检测器 |
| [ClipBased-SyntheticImageDetection](https://github.com/grip-unina/ClipBased-SyntheticImageDetection) | CLIP 特征 | **Apache-2.0**，许可宽松 |
| [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) | 检测框架 | **CC BY-NC 4.0**；注意其范围是**被篡改的脸**，不是从零生成的假脸 |

**⚠️ 关于 HuggingFace 上那些「AI 图像检测器」**：准确率都是**自报的**，训练数据与适用范围常不明确（有的模型卡自己就写明「不是 deepfake 检测器」）。**可以做交叉参考，但不要用它们的数字支撑你的科学结论。**

**⚠️ 不存在权威的「合成人脸检测」基准**：这个领域事实上的评测协议来自 CNNDetection / UniversalFakeDetect 系列（ProGAN/StyleGAN 系 + FFHQ/LSUN）。所以你的报告要**写清用的是哪个检测器、在什么数据上、什么预处理**，否则数字无法解释。

详细版见 [`docs/REFERENCES.md`](../REFERENCES.md) 第 4 节。

## 新手提示 / 卡住了怎么办

- 不要自己训检测器（那是另一个项目的工作量），**只用现成权重推理**。
- 真实图与生成图的数量、分辨率、是否经过 JPEG 压缩都要对齐，否则检测器可能只是学到了「压缩痕迹」这个无关特征。
- 如果某个检测器的 AUC 是 0.5 左右（完全分不出来），**先怀疑自己的评测设置有 bug**，再考虑「生成质量真的很高」这个解释。
- **降级方案**：只跑 1 个检测器，在报告里注明「单检测器结果，参考价值有限」。
