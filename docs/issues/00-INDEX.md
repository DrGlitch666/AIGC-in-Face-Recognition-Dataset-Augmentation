# Issue 索引（自动生成，请勿手工修改）

> 由 `tools/sync_issues.py --export-index` 于 2026-10-07T11:44:16+00:00 读取 GitHub 生成，共 25 条 Issue：开放 8、关闭 17；已排除 Pull Request。
> 重新生成：`python tools/sync_issues.py --export-index`。仅发送 GET 请求，不修改 GitHub。
> 读取来源：GitHub REST issue list。搜索来源可能存在索引延迟；完整性按返回总数核对。
> 状态为读取时的快照；Issue 关闭状态与本地验收记录分别保留。

| # | 标题 | 状态 | 里程碑 | 标签 | 任务卡 |
|---|---|---|---|---|---|
| [#1](https://github.com/DrGlitch666/AIGC-in-Face-Recognition-Dataset-Augmentation/issues/1) | [环境] 跨机器可复现环境 + 硬件档位自检（每台机器各做一次） | 关闭 | M0 脚手架与地基 | good first issue 优先级:P0 类型:环境 难度:入门 | [`01-env-setup.md`](01-env-setup.md) |
| [#2](https://github.com/DrGlitch666/AIGC-in-Face-Recognition-Dataset-Augmentation/issues/2) | [脚手架] 建立仓库目录结构、配置系统与工程规范 | 关闭 | M0 脚手架与地基 | 优先级:P0 类型:环境 难度:入门 | [`02-repo-skeleton.md`](02-repo-skeleton.md) |
| [#3](https://github.com/DrGlitch666/AIGC-in-Face-Recognition-Dataset-Augmentation/issues/3) | [数据] 下载评测数据集 + 构建 toy 训练集 + 冻结 manifest 契约 | 关闭 | M0 脚手架与地基 | 优先级:P0 类型:数据 难度:中等 | [`03-data-pipeline.md`](03-data-pipeline.md) |
| [#4](https://github.com/DrGlitch666/AIGC-in-Face-Recognition-Dataset-Augmentation/issues/4) | [文档] 背景知识地图与文献综述 docs/BACKGROUND.md | 开放 | M0 脚手架与地基 | good first issue 优先级:P0 难度:入门 类型:文档 | [`04-background-survey.md`](04-background-survey.md) |
| [#5](https://github.com/DrGlitch666/AIGC-in-Face-Recognition-Dataset-Augmentation/issues/5) | [基线] 用预训练模型跑通 1:1 人脸验证评测（E0） | 关闭 | M1 基线与评测 | 优先级:P0 难度:中等 类型:评测 | [`05-zeroshot-baseline.md`](05-zeroshot-baseline.md) |
| [#6](https://github.com/DrGlitch666/AIGC-in-Face-Recognition-Dataset-Augmentation/issues/6) | [训练] 实现 ArcFace 训练管线（E1：只用真实数据） | 关闭 | M1 基线与评测 | 优先级:P0 类型:训练 难度:进阶 | [`06-arcface-training.md`](06-arcface-training.md) |
| [#7](https://github.com/DrGlitch666/AIGC-in-Face-Recognition-Dataset-Augmentation/issues/7) | [评测] 统一评测器：1:1 / 1:N / 分桶鲁棒性 / 公平性 + 结果 schema | 开放 | M1 基线与评测 | 优先级:P0 难度:中等 类型:评测 | [`07-evaluator.md`](07-evaluator.md) |
| [#8](https://github.com/DrGlitch666/AIGC-in-Face-Recognition-Dataset-Augmentation/issues/8) | [实验] 传统数据增强对照组（E2）：翻转/裁剪/色彩/模糊/降质 | 关闭 | M2 生成与筛选 | 难度:入门 优先级:P1 类型:实验 | [`08-classic-augmentation.md`](08-classic-augmentation.md) |
| [#9](https://github.com/DrGlitch666/AIGC-in-Face-Recognition-Dataset-Augmentation/issues/9) | [生成] 生成管线 A：无条件生成，打通「生成→入库→训练→评测」闭环（E3） | 关闭 | M2 生成与筛选 | 难度:中等 优先级:P1 类型:生成 | [`09-unconditional-generation.md`](09-unconditional-generation.md) |
| [#10](https://github.com/DrGlitch666/AIGC-in-Face-Recognition-Dataset-Augmentation/issues/10) | [生成] 生成管线 B ★核心：身份保持生成（同一个人，不同姿态/光照） | 关闭 | M2 生成与筛选 | 优先级:P0 难度:进阶 类型:生成 | [`10-identity-preserving-generation.md`](10-identity-preserving-generation.md) |
| [#11](https://github.com/DrGlitch666/AIGC-in-Face-Recognition-Dataset-Augmentation/issues/11) | [筛选] 生成质量与身份一致性门禁（漏斗）★核心 | 关闭 | M2 生成与筛选 | 优先级:P0 难度:中等 类型:筛选 | [`11-generation-filtering.md`](11-generation-filtering.md) |
| [#12](https://github.com/DrGlitch666/AIGC-in-Face-Recognition-Dataset-Augmentation/issues/12) | [合规] 合成人脸可检测性报告：我们的图能被认出来是 AI 生成的吗？ | 开放 | M2 生成与筛选 | 难度:中等 优先级:P2 类型:合规 | [`12-synth-detection-report.md`](12-synth-detection-report.md) |
| [#13](https://github.com/DrGlitch666/AIGC-in-Face-Recognition-Dataset-Augmentation/issues/13) | [生成] 生成管线 C（加分）：属性可控与跨域定向生成 | 关闭 | M2 生成与筛选 | 难度:进阶 类型:生成 优先级:P2 | [`13-attribute-controlled-generation.md`](13-attribute-controlled-generation.md) |
| [#14](https://github.com/DrGlitch666/AIGC-in-Face-Recognition-Dataset-Augmentation/issues/14) | [实验] ★核心：混合比例消融实验矩阵（E5）与批量执行 | 开放 | M3 实验矩阵与结论 | 优先级:P0 难度:进阶 类型:实验 | [`14-mix-ratio-matrix.md`](14-mix-ratio-matrix.md) |
| [#15](https://github.com/DrGlitch666/AIGC-in-Face-Recognition-Dataset-Augmentation/issues/15) | [文档] 结果汇总、结论与局限 docs/RESULTS.md | 关闭 | M3 实验矩阵与结论 | 优先级:P0 难度:中等 类型:文档 | [`15-results-conclusion.md`](15-results-conclusion.md) |
| [#16](https://github.com/DrGlitch666/AIGC-in-Face-Recognition-Dataset-Augmentation/issues/16) | [网站] 静态展示站骨架：Vite + React + ECharts + 数据契约 | 关闭 | M4 网站与发布 | 难度:中等 优先级:P1 类型:网站 | [`16-site-skeleton.md`](16-site-skeleton.md) |
| [#17](https://github.com/DrGlitch666/AIGC-in-Face-Recognition-Dataset-Augmentation/issues/17) | [网站] 展示内容页：方法图、图库对比、指标看板、公平性与失败案例 | 关闭 | M4 网站与发布 | 难度:中等 优先级:P1 类型:网站 | [`17-site-content.md`](17-site-content.md) |
| [#18](https://github.com/DrGlitch666/AIGC-in-Face-Recognition-Dataset-Augmentation/issues/18) | [合规] 伦理与许可文档 + README/LICENSE + v0.1 发布 | 开放 | M4 网站与发布 | 难度:入门 优先级:P1 类型:合规 | [`18-ethics-and-release.md`](18-ethics-and-release.md) |
| [#19](https://github.com/DrGlitch666/AIGC-in-Face-Recognition-Dataset-Augmentation/issues/19) | [协作] 双人协作规范：分工认领、PR 评审、数据与产物同步 | 关闭 | M0 脚手架与地基 | good first issue 优先级:P0 类型:环境 难度:入门 | [`19-collaboration-protocol.md`](19-collaboration-protocol.md) |
| [#20](https://github.com/DrGlitch666/AIGC-in-Face-Recognition-Dataset-Augmentation/issues/20) | [环境] 硬件档位判定、配置分层与自适应降级（A/B/C profile） | 关闭 | M0 脚手架与地基 | 优先级:P0 类型:环境 难度:中等 | [`20-hardware-profiles.md`](20-hardware-profiles.md) |
| [#21](https://github.com/DrGlitch666/AIGC-in-Face-Recognition-Dataset-Augmentation/issues/21) | [学习] 按 LEARNING.md 走完六阶段学习线（含自测与每周互讲） | 开放 | M0 脚手架与地基 | good first issue 优先级:P0 难度:入门 类型:文档 | [`21-learning-path.md`](21-learning-path.md) |
| [#22](https://github.com/DrGlitch666/AIGC-in-Face-Recognition-Dataset-Augmentation/issues/22) | [环境] 受限网络下的资源获取：连通性预检、镜像路线与离线缓存 | 关闭 | M0 脚手架与地基 | good first issue 优先级:P0 类型:环境 难度:入门 | [`22-offline-mirrors.md`](22-offline-mirrors.md) |
| [#25](https://github.com/DrGlitch666/AIGC-in-Face-Recognition-Dataset-Augmentation/issues/25) | [B-1] 展示网站：静态站 + 内容页（合并原 #16 / #17） | 开放 | — | 优先级:P0 难度:中等 类型:网站 | —（见 GitHub 任务卡） |
| [#26](https://github.com/DrGlitch666/AIGC-in-Face-Recognition-Dataset-Augmentation/issues/26) | [B-2] 评测器补强：边界单元测试 + 跨机（CPU）复现验证 | 关闭 | — | 优先级:P0 难度:中等 类型:评测 | —（见 GitHub 任务卡） |
| [#27](https://github.com/DrGlitch666/AIGC-in-Face-Recognition-Dataset-Augmentation/issues/27) | [B-3] 报告排版 + 局限章节核对 + 文献补充 | 开放 | — | 优先级:P0 难度:入门 类型:文档 | —（见 GitHub 任务卡） |
