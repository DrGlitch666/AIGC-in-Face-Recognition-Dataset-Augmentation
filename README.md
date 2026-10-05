# AIGC in Face Recognition Dataset Augmentation

> 用身份保持的生成模型为**已有身份**合成更多照片，检验它能否提升人脸识别性能。

**一句话结论**：加入 755 张合成图（占训练集 17.4%）使 LFW **accuracy 从 0.8812 提升到 0.8899**
（Welch t=3.59，三个协议三个种子方向一致）；**TAR@FAR=1e-3 呈正向趋势但未达显著**
（+0.018~+0.039，t=1.5~1.8，种子方差大于效应量）。

---

## 📊 核心结果

### 主对照（同模型、同损失、同轮数、同增强，**只差训练数据**；3 个随机种子）

| 方案 | 训练集 | accuracy (official) | **TAR@FAR=1e-3** |
|---|---|---|---|
| **E1** | real 3590 | 0.8812 ± 0.0032 | 0.4208 ± 0.0040 |
| **E4a** | + 合成 755（**17.4%**）| **0.8899 ± 0.0026** | 0.4548 ± 0.0321 |
| E4b | + 合成 448（11.1%）| 0.8875 | 0.4597 |

**同一对照在无泄漏协议上的复现**（LFW 5653 个未见身份 / 125,191 对）：

| 方案 | accuracy | TAR@FAR=1e-3 |
|---|---|---|
| E1 | 0.8693 ± 0.0035 | 0.3108 ± 0.0184 |
| **E4a** | **0.8775 ± 0.0005**（t=3.98）| 0.3289 ± 0.0095 |

### 生成质量（`id_sim` = 与本人真实图中心的余弦）

> 参照：**同一个人的两张真实照片 = 0.745**；**不同的人 < 0.25**

| 方案 | 均值 | 最小值 |
|---|---|---|
| base FaceID | +0.3891 | +0.2035 |
| plusv2 官方配方 | +0.5138 | +0.4320 |
| **plusv2 + 多参考平均（采用）** | **+0.6084** | **+0.5273** |
| plusv2 + Realistic Vision V6 底模 | +0.3999 | +0.2836（出局）|

视觉对照（真实参考 ┃ base FaceID ┃ plusv2）：
[`results/runs/w3-generator-comparison/contact_sheet.png`](results/runs/w3-generator-comparison/contact_sheet.png)

---

## 📁 关键文档

| 文档 | 内容 |
|---|---|
| [`docs/REPORT.md`](docs/REPORT.md) | ⭐ **实验报告**：方法 / 结果 / 结论 / 局限 / 复现 |
| [`docs/W3-GENERATION.md`](docs/W3-GENERATION.md) | **生成配方 + 11 条踩坑记录**（本项目最实用的文档） |
| [`docs/PERSONAL_PLAN.md`](docs/PERSONAL_PLAN.md) | 执行方案（A/B 两人版）：六周排期、分工、砍掉清单 |
| [`docs/FRAMEWORK.md`](docs/FRAMEWORK.md) | 项目框架：架构、数据契约、硬件档位与配置分层、实验设计 |
| [`docs/WORKFLOW.md`](docs/WORKFLOW.md) | 工作流：Sprint 路线图、Issue 流程、实验记录纪律 |
| [`docs/LEARNING.md`](docs/LEARNING.md) | 学习路线 L0~L6（含自测题） |
| [`docs/SETUP.md`](docs/SETUP.md) | 环境搭建 + 10 条已踩过的坑 |
| [`docs/MIRRORS.md`](docs/MIRRORS.md) | 受限网络下的镜像路线与**实测速度表** |

---

## 🚀 复现

```bash
# 1) 环境
conda env create -f environment.yml    # 或按 docs/SETUP.md 手动装
conda activate aigcfr

# 2) 数据：LFW 下载 + 对齐（96 身份 / 3590 张）
python scripts/download_data.py --config configs/data/lfw.yaml
python scripts/build_dataset.py

# 3) 基线 E0（零样本 buffalo_l）
python scripts/evaluate.py --exp-id e0-buffalo_l-lfw

# 4) 生成（先下 plusv2 权重，约 2.5 GB）
python scripts/download_data.py --config configs/data/ipadapter_plusv2.yaml
python scripts/generate_plusv2.py --per-identity 10 --max-refs 2

# 5) 对齐 + 筛选（960 → 755 张）
python scripts/align_synth.py
python scripts/filter_synth.py --exp-id syn-plusv2 --threshold 0.45 --keep-per-identity 8

# 6) 训练 E1 与 E4a（各 3 个种子）
python scripts/train.py --exp-id e1-real-only-seed0 --exp configs/exp/e1-real-only-seed0.yaml --seed 0
python scripts/train.py --exp-id e4-real-plus-synth-seed0 --exp configs/exp/e4-real-plus-synth.yaml --seed 0

# 7) 评测（官方协议 + 无泄漏大样本协议）
python scripts/evaluate.py --exp-id e4-real-plus-synth-seed0 --ckpt F:/aigcfr/ckpt/e4-real-plus-synth-seed0/best.pth
python scripts/build_verify_protocol.py
python scripts/evaluate_heldout.py --exp-id e4-real-plus-synth-seed0 --ckpt F:/aigcfr/ckpt/e4-real-plus-synth-seed0/best.pth

# 8) 汇总统计
python tools/w4_compare.py
```

---

## 🧭 当前状态（截至最近一次更新）

| 阶段 | 状态 |
|---|---|
| W1 环境与骨架 | ✅ |
| W2 数据管线 + E0/E1 基线 | ✅ |
| W3 合成数据生成（plusv2，`id_sim` 0.6154）| ✅ |
| W4 E4 对照 + 多种子统计 | ✅ |
| W5 展示网站 | ⬜ **由 B 负责**（Issue [#25]） |
| W6 报告排版与交付 | ⬜ **由 B 负责**（Issue [#27]） |

**Issue 状态**：22 张原始任务卡已按现实情况同步（16 张完成关闭），
并新增 3 张 B 的任务卡。见 [Issues](https://github.com/DrGlitch666/AIGC-in-Face-Recognition-Dataset-Augmentation/issues)。

---

## ⚠️ 已知局限（详见 [`docs/REPORT.md`](docs/REPORT.md) §5）

1. **训练/测试泄漏不可避免**：LFW 官方 6000 对覆盖 4281 个身份，**含全部 96 个训练身份**。
   已用无泄漏协议量化 —— 官方协议把 TAR 效应高估了近一倍（+0.034 vs +0.018）。
2. **TAR@FAR=1e-3 未达显著**，且种子方差大于效应量。
3. 合成占比只测了 17.4% / 11.1%，未做占比扫描。
4. 只测了 iresnet18 一个架构。
5. 数据规模小（96 身份 / 3590 张）—— 自研模型 TAR 0.33 vs 零样本 buffalo_l **0.9989**。

---

## 🙏 致谢与许可

本项目为非商业研究用途。注意 **insightface 预训练模型仅限非商业研究**；
生成所用底模与适配器的许可请见 [`docs/REPORT.md`](docs/REPORT.md) 与相关 Issue。
