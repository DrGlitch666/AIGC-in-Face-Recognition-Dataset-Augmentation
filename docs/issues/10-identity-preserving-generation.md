---
title: "[生成] 生成管线 B ★核心：身份保持生成（同一个人，不同姿态/光照）"
slug: "10-identity-preserving-generation"
labels: ["优先级:P0", "类型:生成", "难度:进阶"]
milestone: "M2 生成与筛选"
---

## 背景 / 为什么做这个

**这是整个项目的技术核心。**

前面 E3 生成的是「随机的人脸」，它们和训练集里的任何身份都没有关系，对识别模型的帮助很可能是间接的（充当正则化）。而人脸识别数据集扩充真正想要的，是**同一个人的更多变化**：

> 给模型看「张三的正脸」，让它能认出「张三的侧脸、逆光的张三、戴口罩的张三、十年后的张三」。

这就是**身份保持生成（identity-preserving generation）**：把某个身份的特征（通常是从参考图提取的 ArcFace embedding）作为条件注入扩散模型，生成**同一个人**的新图像。

一旦这一层跑通，你就拥有了「按需造数据」的能力——可以定向补足数据集里缺失的姿态、光照、年龄段，而这正是 AIGC 相比传统增强的本质区别。

## 目标

对每个真实身份生成 K 张「明显是同一个人、但姿态/光照/表情不同」的图像，并量化身份保持程度。

## 任务清单

- [ ] 从 {{#04-background-survey}} 的表 2 里挑 **1 个主选 + 1 个备选**生成器，并记录选择理由（必须确认权重可下载、许可可接受、8 GB 可跑）
  - 候选方向（按显存友好度大致排序）：基于 IP-Adapter 的人脸身份适配、ArcFace-embedding 条件生成、InstantID 类方案、DCFace/IDiff-Face 类专业合成数据集生成器
  - **建议**：先试「SD1.5 + 人脸身份适配器」这条线（显存最友好），跑通后再评估是否升级到 SDXL 方案
- [ ] 实现 `src/aigcfr/generate/<你的选择>.py`，继承 {{#09-unconditional-generation}} 定义的 `FaceGenerator` 接口
- [ ] 身份条件的两种参考方式都要试并对比：
  - [ ] 单张参考图（选该身份质量最好的那张）
  - [ ] 多张参考图平均（identity centroid，通常更稳）
- [ ] 构建 prompt 模板库：同一身份 × 多个属性描述（姿态：正面/侧面/俯视；光照：室内/逆光/夜间；表情：微笑/严肃；年龄：年轻/中年）
- [ ] 多样性来源要写清楚：seed 变化 + prompt 变化 +（可选）ControlNet 姿态控制
- [ ] 生成规模：**每个身份 ≥ 5 张 × ≥ 20 个身份**（= ≥100 张）；先用 3 个身份 × 5 张试跑，人工确认后再放量
- [ ] 写 `scripts/generate.py` 的身份条件分支，产出 `data/manifests/synth_e4.jsonl`
  - [ ] `parent_image_id` 必须指向参考图 → 保证可追溯
  - [ ] `meta` 里记录 prompt、seed、参考图数量
- [ ] 量化身份保持程度：用 {{#05-zeroshot-baseline}} 的 ArcFace 模型算「生成图 vs 该身份参考图中心」的余弦相似度，画分布直方图
- [ ] 做**人工评测**：随机抽 30 张生成图，人工判断「是否像参考图里的人」（是/否），给出通过率（这是最诚实的评估）
- [ ] 记录：显存峰值、每张图耗时、总耗时（RQ5 需要）

## 验收标准（Definition of Done）

- [ ] 生成图**肉眼可辨认为同一身份**，人工抽检通过率 ≥ 70%（低于此值不要进入 {{#14-mix-ratio-matrix}}，先改生成）
- [ ] 身份相似度分布图已落盘，并给出均值/中位数/10% 分位数
- [ ] 生成规模的 manifest 完整：每张图都能追到参考图、prompt、seed、生成器版本
- [ ] **不同生成图之间有明显差异**（不是同一张脸复制 5 遍）——同时报告「身份内多样性」指标（例如生成图两两相似度的均值，越低越多样）
- [ ] 显存峰值 < 8 GB，或明确记录「使用了 CPU offload，速度代价是 X」
- [ ] 报告 `reports/identity_gen_report.md`：选型理由、参数、成功率、失败模式（哪些属性生不出来）

## 交付物

- `src/aigcfr/generate/<生成器>.py` + `configs/gen/<生成器>.yaml`
- `data/manifests/synth_e4.jsonl`
- `reports/identity_gen_report.md`
- `reports/figures/identity_sim_dist.png`、`reports/figures/identity_grid.png`（每个身份一行：参考图 + 生成图）
- `reports/figures/human_eval_sheet.md`（人工评测记录）

## 依赖

- 需要 {{#09-unconditional-generation}}（生成器接口、显存管理与批量脚本）
- 需要 {{#04-background-survey}}（选型依据：权重可得性与显存需求）
- 需要 {{#03-data-pipeline}}（参考图来自对齐后的真实数据）
- 被依赖：{{#11-generation-filtering}}（筛选的对象）、{{#14-mix-ratio-matrix}}（E4 是核心实验组）

## 预估工作量

2~4 天（**这是全项目最不确定的一步，预留缓冲**）

## 参考

- [Diffusers 官方文档](https://huggingface.co/docs/diffusers)（IP-Adapter / ControlNet 的使用方式）
- [IP-Adapter 官方仓库](https://github.com/tencent-ailab/IP-Adapter)（人脸身份适配的常用实现，含 FaceID 变体）
- [InstantID 官方仓库](https://github.com/instantX-research/InstantID)（SDXL 路线，单图身份保持，显存要求更高）
- [Arc2Face 官方仓库](https://github.com/foivospar/Arc2Face)（直接用 ArcFace embedding 作为条件，概念上最贴合本项目）
- [DCFace 官方仓库](https://github.com/mk-minchul/dcface)（双条件合成人脸数据集生成，偏「造数据集」而非「造单张图」）
- [DigiFace-1M](https://github.com/microsoft/DigiFace1M)（微软公开的合成人脸数据集）
- 具体可用的权重清单与显存实测，以 {{#04-background-survey}} 的 `docs/BACKGROUND.md` 表 2 为准

## 附：已核实要点（框架预置，2026-09）——**这一节直接改变本任务的执行顺序**

### A. 双轨制：先 B 轨，再 A 轨（**强烈建议**）

| 轨道 | 做法 | 成本 | 作用 |
|---|---|---|---|
| **B 轨（先做）** | **直接下载已公开的合成人脸数据集** | 几乎为零（只花下载时间，不占显存） | 立刻跑通「合成数据 → 训练 → 评测」的因果链，保证项目一定有结论 |
| **A 轨（再做）** | 本地跑身份条件生成（本任务的主体） | 数天 + 显存压力 | 这是项目的 AIGC 主线：验证「按需生成」的价值 |

**理由**：A 轨是全项目最不确定的一步。先做 B 轨，即使 A 轨最终失败，你的项目仍然有完整可交付的结论；而且 B 轨的数据还能当作 A 轨的对照组。

**可下载的现成合成数据集（含直接链接，见 [`REFERENCES.md`](../REFERENCES.md) 第 3、5 节）**：
- [DigiFace-1M](https://github.com/microsoft/DigiFace1M)：122 万张 / 11 万身份（**首选**）
- [DCFace](https://github.com/mk-minchul/dcface)：权重与数据集在 Google Drive
- IDiff-Face：10K 身份 × 50 张（CC BY-NC-SA 4.0）
- [CemiFace](https://github.com/szlbiubiubiu/CemiFace)、Digi2Real（20K 身份，Zenodo）、HSFace10k（HuggingFace）

### B. 生成器选型（8 GB 显存实测结论）

| 方案 | 8 GB 可行性 | 说明 |
|---|---|---|
| **Arc2Face**（SD1.5，512²） | ✅ **主选** | 只以 ArcFace embedding 为条件，概念最贴合本项目；**代码 MIT** |
| **IP-Adapter-FaceID（SD1.5）** | ✅ **备选** | 生态成熟，显存友好 |
| IDiff-Face（128² 隐扩散） | ✅ 可跑 | 注意 CC BY-NC-SA 4.0 |
| DCFace（112~128²） | ✅ 可跑 | 偏「造数据集」 |
| Vec2Face / HSFace（GAN） | ✅ 可跑 | 代码 MIT |
| InstantID、IP-Adapter-FaceID-**SDXL** | ⚠️ 需降级 | 必须 `enable_model_cpu_offload()` + VAE 分块，速度明显变慢 |
| **PhotoMaker** | ❌ | 官方 README 写明**最低 11 GB 显存** |
| **PuLID-FLUX** | ❌ | 需 11~16 GB，且继承 FLUX.1-dev 的非商业许可 |

### C. ⚠️ 许可陷阱（必须写进 issue #18）

Arc2Face / InstantID / IP-Adapter-FaceID / PuLID **都依赖 insightface 的人脸模型**（`antelopev2` / `buffalo_l`），而这些**模型包仅限非商业研究用途**；InstantID 与 IP-Adapter-FaceID 的**权重本身也是 research-only**。

> **「代码是 MIT」不等于「整条链路可商用」。** 本项目定位非商业研究，可以放心使用，但必须在文档里如实声明。

## 附：两人两机与本任务的衔接

- **本任务是"吃显存"的重活，默认落在有独显的一方**（A/B 档）。弱机器（C 档）在这一阶段承担：**B 轨现成合成数据的接入、筛选（{{#11-generation-filtering}}）、生成质量的人工抽检、以及网站骨架**——这些工作量并不少。
- **A 轨（本地生成）和 B 轨（下载现成数据）可以同时开工**：
  - 弱机器先跑 B 轨，**当天就能拿到可用于训练的合成数据**，让 {{#14-mix-ratio-matrix}} 的管线提前打通；
  - 强机器同时试 A 轨。这样即使 A 轨失败，项目也不阻塞（双轨制的意义所在）。
- **生成图怎么传给另一台机器**：推荐"**谁生成谁训练**"，避免几十 GB 的传输；若要共享，只用局域网/移动硬盘，**绝不要提交进 Git**（`.gitignore` 已排除）。
- 生成的 `manifest` 片段**要入库**（含 `generator`、`seed`、`prompt`、`parent_image_id`），这样另一台机器即使没有图片，也能从清单看出"这批数据是怎么来的"。
- 本任务的动手环节与 [`LEARNING.md`](../LEARNING.md) **L3**（身份相似度热力图、保真 vs 多样性的取舍）合并完成。

## 新手提示 / 卡住了怎么办

- **先做最小验证**：1 个身份、1 张参考图、生成 4 张，看像不像。这一步不到 30 分钟，能决定后面几天往哪个方向走。
- 常见的失败模式：
  - 生成的脸「像但不是我」→ 身份条件权重太低 / 参考图质量差 / 参考图本身没对齐
  - 5 张图长得一模一样 → 多样性不足，加大 seed 与 prompt 的变化，或降低身份条件强度
  - 画面崩坏、多人脸 → 加负向 prompt、开人脸修复、提高步数
- 伦理红线：**只用公开研究数据集里的身份做参考**，不拿名人或他人的私人照片来「克隆身份」。所有生成图必须在 manifest 和网站上标注为 AI 生成。
- **降级方案（按顺序尝试）**：
  1. 换更省显存的配置（SD1.5 而非 SDXL；384×384 而非 512×512）
  2. 降规模：10 个身份 × 3 张，仍可支撑 E4 与消融
  3. 改用「下载公开合成人脸数据集」+「传统增强模拟姿态/光照变化」的组合，并把 E4 明确标注为「未能使用身份条件生成」
  4. 把身份保持生成标记为「未完成」，项目仍可用 E1/E2/E3 出结论 —— **不要因为这一步卡住就放弃整个项目**
