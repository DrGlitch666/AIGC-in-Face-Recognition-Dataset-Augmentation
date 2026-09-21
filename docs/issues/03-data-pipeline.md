---
title: "[数据] 下载评测数据集 + 构建 toy 训练集 + 冻结 manifest 契约"
slug: "03-data-pipeline"
labels: ["优先级:P0", "类型:数据", "难度:中等"]
milestone: "M0 脚手架与地基"
---

## 背景 / 为什么做这个

整个项目的数据量很小（8 GB 显存 + 业余时间），所以**数据的组织方式比数据量重要得多**。

这一步要做三件事：

1. 拿到**公开可下载**的评测集，用于最终打分（LFW / CFP-FP / AgeDB-30，可选 CALFW / CPLFW / RFW）。
2. 从 LFW 里切出一个**小型闭集训练集**（几十个身份、每身份十几到几十张图），用于真正「训练」一个模型。
3. 把一切都写进统一的 `manifest.jsonl`——这是整个项目的**数据契约**，后面所有层（生成/筛选/训练/评测）都只认它。

> ⚠️ **合规红线**：原始人脸数据**绝不提交到 Git 仓库**。只提交 manifest（文本清单）和统计报告。

## 目标

一条命令能从零把数据准备好，且每个 split 的身份与图片数量都有据可查。

## 任务清单

- [ ] 调研并确定下载源：LFW、CFP-FP、AgeDB-30（每个至少准备 2 个可用来源，记录在配置里）
- [ ] 写 `scripts/download_data.py`：按 `configs/data/*.yaml` 下载并解压，带断点续传与校验（文件数/大小）
- [ ] 写 `src/aigcfr/data/align.py`：5 点关键点检测 + 相似变换裁到 **112×112**（ArcFace 模板），训练与评测**必须用同一套**
- [ ] 写 `src/aigcfr/data/manifest.py`：读写与校验 `manifest.jsonl`（字段定义见 `docs/FRAMEWORK.md` §3.3 契约 A）
- [ ] 写 `scripts/build_dataset.py`：
  - 构建 toy 训练集：从 LFW 选**身份图片数 ≥ 15** 的身份，按身份切 train/val（**按身份切，不能按图片切**）
  - 构建 toy 测试集：**与训练身份完全不重叠**的一批身份
  - 构建评测集清单：LFW 6000 对、CFP-FP、AgeDB-30 的官方 pair 列表解析成同一个 manifest 格式
- [ ] 写 `tests/test_manifest_schema.py`：字段完整性、`identity_id` 唯一性、`source` 枚举、路径存在性
- [ ] 生成统计报告 `reports/data_stats.md`：各 split 的身份数/图片数/每身份图片数分布
- [ ] 生成抽样可视化 `reports/figures/data_samples.png`：4×4 网格（含身份标注），**人工肉眼确认对齐是否正确**

## 验收标准（Definition of Done）

- [ ] `python scripts/download_data.py --config configs/data/all.yaml` 一条命令拿到全部数据
- [ ] manifest 通过 `pytest tests/test_manifest_schema.py`
- [ ] `reports/data_stats.md` 里有明确的数字（身份数、图片数、每身份最少/最多/中位图片数）
- [ ] 抽样网格图里人脸**居中、眼位对齐、无严重旋转**（对齐错了后面全错）
- [ ] 训练身份与评测身份**零重叠**（在报告里写一句确认）
- [ ] `git status` 里不出现任何图片文件

## 交付物

- `scripts/download_data.py`、`scripts/build_dataset.py`
- `src/aigcfr/data/{align,manifest,datasets}.py`
- `configs/data/{lfw,cfp_fp,agedb,toy_train}.yaml`
- `data/manifests/*.jsonl`（清单进 Git，图片不进）
- `reports/data_stats.md`、`reports/figures/data_samples.png`
- `tests/test_manifest_schema.py`

## 依赖

- 需要 {{#01-env-setup}}（opencv / insightface 用于关键点检测）
- 需要 {{#02-repo-skeleton}}（目录与配置约定）
- 被依赖：{{#05-zeroshot-baseline}}、{{#06-arcface-training}}、{{#10-identity-preserving-generation}}

## 预估工作量

1~1.5 天

## 参考

- [LFW 官方页面](http://vis-www.cs.umass.edu/lfw/)（含 6000 对验证协议说明）
- [CFP 官方页面](http://www.cfpw.io/)（CFP-FP 前后姿态对）
- [AgeDB 官方页面](https://ibug.doc.ic.ac.uk/resources/agedb/)（跨年龄验证）
- [insightface 人脸对齐实现](https://github.com/deepinsight/insightface)（`insightface/utils/face_align.py` 的 `norm_crop` 是标准做法）
- 数据集的可再分发条款各不相同，**先读许可再决定能不能公开图片**

## 附：已核实要点（框架预置，2026-09）

**可得性（这一列决定了你的计划能排多满）**

| 数据集 | 2026 年免费直接下载？ | 备注 |
|---|---|---|
| LFW | ⚠️ 官方站点不稳定，torchvision 已不再自动下载 | **准备 Kaggle 等镜像并预留时间** |
| CFP-FP | ✅ 可直接下载 | 主线之一 |
| AgeDB-30 | ⚠️ 压缩包**有密码，需邮件向作者索取** | 仅限非商业研究 |
| CALFW / CPLFW | ✅ Google Drive / 百度网盘 | 与 LFW **共享同一批身份**，不是独立测试 |
| RFW | ❌ **需邮件申请** | 评测代码免费，数据要申请 |
| TinyFace | ✅ Google Drive（约 148 MB） | **1:N Rank-1**，不易饱和，建议加入评测集 |
| IJB-C | ❌ **NIST 已于 2023-03-14 停止 IJB-A/B/C 分发** | **不要写进计划** |

- **绝不能提交到仓库**：CelebA、MS-Celeb-1M（已撤回）、VGGFace2（已撤回）、CASIA-WebFace、AgeDB、RFW 的**图片**；**对齐/裁剪后的衍生图同样受限**。只提交下载脚本 + manifest。
- `faces_emore` 归档里带有现成的 `lfw.bin` / `cfp_fp.bin` / `agedb_30.bin` 评测包，可用于交叉验证你自己的评测实现是否正确。
- 详细版见 [`docs/REFERENCES.md`](../REFERENCES.md) 第 3 节与第 8 节。

## 新手提示 / 卡住了怎么办

- 关键点检测先用 insightface 的 `FaceAnalysis` 现成能力，**不要**自己训检测器。
- 检测不到人脸的图直接丢弃并记录数量，不要硬裁。
- **降级方案**：如果 CFP-FP / AgeDB 下载受阻，只保留 LFW（6000 对协议足够支撑整个项目），在 `reports/data_stats.md` 里记一句「因下载问题降级」。
- LFW 里每个身份的图片数极不均衡（名人多的有几百张），选身份时先看分布再定阈值。
