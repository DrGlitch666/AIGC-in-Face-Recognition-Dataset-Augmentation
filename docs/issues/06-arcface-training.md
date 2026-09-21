---
title: "[训练] 实现 ArcFace 训练管线（E1：只用真实数据）"
slug: "06-arcface-training"
labels: ["优先级:P0", "类型:训练", "难度:进阶"]
milestone: "M1 基线与评测"
---

## 背景 / 为什么做这个

E0 用的是别人的预训练模型，你无法控制它用什么数据训练的，也就**无法做「加合成数据」的对照实验**。所以必须自己训一个模型：数据完全可控、超参完全可控、每次改动都能重跑。

这一步的产出 E1（real-only）是整个项目**最重要的基线**——后面所有「合成数据有用/没用」的结论都是相对它说的。

技术路线：`iresnet`（InsightFace 风格的 ResNet）+ **ArcFace 损失**（加性角度间隔），输入 112×112 对齐人脸，输出 512 维归一化 embedding。

## 目标

在 toy 数据集上训出一个「能正常收敛、指标可复现」的模型，且全程不 OOM。

## 任务清单

- [ ] 写 `src/aigcfr/train/model.py`：`iresnet`（建议 18/34/50 三档，默认 34 或 50，按显存选）
- [ ] 写 `src/aigcfr/train/losses.py`：ArcFace（`margin=0.5`、`scale=64` 起步），实现后**单元测试角度间隔是否正确**
- [ ] 写 `src/aigcfr/train/dataset.py`：读 manifest → 返回 (图像, 身份标签)；训练增强（随机翻转、小角度旋转、颜色抖动）；验证集不做增强
- [ ] 写 `scripts/train.py`：
  - [ ] 读 `configs/exp/*.yaml`，**所有超参来自配置，不许写死在代码里**
  - [ ] 固定随机种子（python / numpy / torch / cudnn）
  - [ ] AMP 混合精度 + 梯度裁剪（8 GB 显存的必需品）
  - [ ] 分批大小自适应：OOM 时自动梯度累积（或者至少给出清晰的报错提示）
  - [ ] 日志：每个 epoch 的 loss / lr / 验证准确率 → CSV + TensorBoard 二选一（推荐先 CSV，简单）
  - [ ] 保存 `best.pth` 与 `last.pth` 到 `results/runs/<exp_id>/`
  - [ ] 记录训练耗时与显存峰值到日志（RQ5 需要这个数据）
- [ ] 写 `configs/exp/e1-real-only-seed0.yaml`（以及 seed1/seed2）
- [ ] 跑 E1：至少 3 个 seed，每个 seed 用 {{#07-evaluator}} 评测并落 `metrics.json`
- [ ] 写 `tests/test_losses.py`：验证 ArcFace 的 logit 与角度间隔符合预期
- [ ] 写 `reports/training_notes.md`：显存-批量大小-速度 的实测对照表

## 验收标准（Definition of Done）

- [ ] `python scripts/train.py --config configs/exp/e1-real-only-seed0.yaml` 一条命令完成训练并落盘 ckpt
- [ ] 训练过程 loss 稳定下降，验证准确率**明显高于随机**（toy 身份数少，闭集准确率应能到较高水平）
- [ ] **同 seed 重跑两次，最终指标差异接近 0**（证明随机性被控制住了）
- [ ] 3 个 seed 的 E1 指标已产出，报告为 **均值 ± 标准差**
- [ ] 显存峰值 < 8 GB（记录在 `reports/training_notes.md`）
- [ ] `--dry-run` 能先打印将要执行的数据量/超参/输出路径

## 交付物

- `src/aigcfr/train/{model,losses,dataset,trainer}.py`
- `scripts/train.py`
- `configs/exp/e1-real-only-seed{0,1,2}.yaml`
- `results/runs/e1-real-only-seed*/{metrics.json,best.pth,log.csv}`
- `reports/training_notes.md`、`tests/test_losses.py`

## 依赖

- 需要 {{#03-data-pipeline}}（manifest 与 toy 训练集）
- 需要 {{#01-env-setup}}（PyTorch + CUDA）
- 被依赖：{{#08-classic-augmentation}}、{{#14-mix-ratio-matrix}}、{{#15-results-conclusion}}

## 预估工作量

1.5~2.5 天（训练是耗时项，可以睡前启动）

## 参考

- [insightface 官方仓库](https://github.com/deepinsight/insightface)（ArcFace 论文作者的实现，`recognition/arcface_torch` 目录是标准参考；**本任务不建议直接用它整套训练框架**，太重，读它的模型和 loss 即可）
- [PyTorch AMP 文档](https://pytorch.org/docs/stable/amp.html)
- ArcFace 原论文：*ArcFace: Additive Angular Margin Loss for Deep Face Recognition*（CVPR 2019）
- 关键细节：`scale=64, margin=0.5` 是常用起点；embedding 必须 L2 归一化后再乘 scale

## 附：已核实要点（框架预置，2026-09）

- **⚠️ 路径更正**：网上大量教程引用的 `src/insightface/recognition` 与 `train_rec.py` **在当前 master 上已不存在**（该路径 404）。官方 **PyTorch** 训练入口是 [`recognition/arcface_torch/train_v2.py`](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch)；MXNet 版本官方已自称「历史实现」且 MXNet 早已停止维护——**不要走 MXNet 路线**。
- `face.evoLVe.PyTorch` 虽然自带 LFW/CFP-FP/AgeDB 评测，但 README 写明只支持 **Linux/macOS**，Windows 上别选它。
- **8 GB 显存可行性（自测口径）**：iResNet-50 @112×112、batch 64~128 + AMP 可以跑。**身份数比图片总数更吃显存**——本项目 toy 规模（几十~几百身份）远低于压力线（约 1 万身份以下普通全连接头都很小）。
- **参考数字（IR-50 在 MS1M 上训练）**：LFW 99.78 / CFP-FP 98.14 / AgeDB 97.53。你的 toy 规模（几十个身份）**一定远低于这个水平，这是正常的**——本项目只做**内部相对比较**，不与论文比绝对值。
- 详细版见 [`docs/REFERENCES.md`](../REFERENCES.md) 第 1 节。

## 新手提示 / 卡住了怎么办

- **先跑通再调优**：先用 10 个身份、iresnet18、5 个 epoch 确认整条链路能跑完并落盘，再去加大规模。
- 常见症状与原因：
  - loss 不下降 → 学习率太大 / 标签错位 / 忘记归一化
  - 准确率 100% → 训练集和验证集身份重叠（数据切分错了，见 {{#03-data-pipeline}}）
  - `CUDA out of memory` → 降 batch、开 AMP、用 iresnet18、开梯度检查点
- **降级方案**：改成 iresnet18 + 输入 96×96，先要「能跑 + 可复现」，精度不是这一阶段的目标。
