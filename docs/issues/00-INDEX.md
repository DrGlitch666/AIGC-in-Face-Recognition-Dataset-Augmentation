# Issue 索引（自动生成，请勿手工修改）

> 由 `tools/file_issues.mjs` 于 2026-09-21T05:05:27.205Z 生成，共 18 条。
> 重新生成：`GH_TOKEN=<token> node tools/file_issues.mjs`（已存在的 Issue 会自动跳过）。

| # | 标题 | 里程碑 | 标签 | 任务卡 |
|---|---|---|---|---|
| [#1](https://github.com/DrGlitch666/AIGC-in-Face-Recognition-Dataset-Augmentation/issues/1) | [环境] 搭建 Python 3.11 + PyTorch(CUDA) 环境，产出 GPU 自检报告 | M0 脚手架与地基 | 优先级:P0 类型:环境 难度:入门 good first issue | [`01-env-setup.md`](01-env-setup.md) |
| [#2](https://github.com/DrGlitch666/AIGC-in-Face-Recognition-Dataset-Augmentation/issues/2) | [脚手架] 建立仓库目录结构、配置系统与工程规范 | M0 脚手架与地基 | 优先级:P0 类型:环境 难度:入门 | [`02-repo-skeleton.md`](02-repo-skeleton.md) |
| [#3](https://github.com/DrGlitch666/AIGC-in-Face-Recognition-Dataset-Augmentation/issues/3) | [数据] 下载评测数据集 + 构建 toy 训练集 + 冻结 manifest 契约 | M0 脚手架与地基 | 优先级:P0 类型:数据 难度:中等 | [`03-data-pipeline.md`](03-data-pipeline.md) |
| [#4](https://github.com/DrGlitch666/AIGC-in-Face-Recognition-Dataset-Augmentation/issues/4) | [文档] 背景知识地图与文献综述 docs/BACKGROUND.md | M0 脚手架与地基 | 优先级:P0 类型:文档 难度:入门 good first issue | [`04-background-survey.md`](04-background-survey.md) |
| [#5](https://github.com/DrGlitch666/AIGC-in-Face-Recognition-Dataset-Augmentation/issues/5) | [基线] 用预训练模型跑通 1:1 人脸验证评测（E0） | M1 基线与评测 | 优先级:P0 类型:评测 难度:中等 | [`05-zeroshot-baseline.md`](05-zeroshot-baseline.md) |
| [#6](https://github.com/DrGlitch666/AIGC-in-Face-Recognition-Dataset-Augmentation/issues/6) | [训练] 实现 ArcFace 训练管线（E1：只用真实数据） | M1 基线与评测 | 优先级:P0 类型:训练 难度:进阶 | [`06-arcface-training.md`](06-arcface-training.md) |
| [#7](https://github.com/DrGlitch666/AIGC-in-Face-Recognition-Dataset-Augmentation/issues/7) | [评测] 统一评测器：1:1 / 1:N / 分桶鲁棒性 / 公平性 + 结果 schema | M1 基线与评测 | 优先级:P0 类型:评测 难度:中等 | [`07-evaluator.md`](07-evaluator.md) |
| [#8](https://github.com/DrGlitch666/AIGC-in-Face-Recognition-Dataset-Augmentation/issues/8) | [实验] 传统数据增强对照组（E2）：翻转/裁剪/色彩/模糊/降质 | M2 生成与筛选 | 优先级:P1 类型:实验 难度:入门 | [`08-classic-augmentation.md`](08-classic-augmentation.md) |
| [#9](https://github.com/DrGlitch666/AIGC-in-Face-Recognition-Dataset-Augmentation/issues/9) | [生成] 生成管线 A：无条件生成，打通「生成→入库→训练→评测」闭环（E3） | M2 生成与筛选 | 优先级:P1 类型:生成 难度:中等 | [`09-unconditional-generation.md`](09-unconditional-generation.md) |
| [#10](https://github.com/DrGlitch666/AIGC-in-Face-Recognition-Dataset-Augmentation/issues/10) | [生成] 生成管线 B ★核心：身份保持生成（同一个人，不同姿态/光照） | M2 生成与筛选 | 优先级:P0 类型:生成 难度:进阶 | [`10-identity-preserving-generation.md`](10-identity-preserving-generation.md) |
| [#11](https://github.com/DrGlitch666/AIGC-in-Face-Recognition-Dataset-Augmentation/issues/11) | [筛选] 生成质量与身份一致性门禁（漏斗）★核心 | M2 生成与筛选 | 优先级:P0 类型:筛选 难度:中等 | [`11-generation-filtering.md`](11-generation-filtering.md) |
| [#12](https://github.com/DrGlitch666/AIGC-in-Face-Recognition-Dataset-Augmentation/issues/12) | [合规] 合成人脸可检测性报告：我们的图能被认出来是 AI 生成的吗？ | M2 生成与筛选 | 优先级:P2 类型:合规 难度:中等 | [`12-synth-detection-report.md`](12-synth-detection-report.md) |
| [#13](https://github.com/DrGlitch666/AIGC-in-Face-Recognition-Dataset-Augmentation/issues/13) | [生成] 生成管线 C（加分）：属性可控与跨域定向生成 | M2 生成与筛选 | 优先级:P2 类型:生成 难度:进阶 | [`13-attribute-controlled-generation.md`](13-attribute-controlled-generation.md) |
| [#14](https://github.com/DrGlitch666/AIGC-in-Face-Recognition-Dataset-Augmentation/issues/14) | [实验] ★核心：混合比例消融实验矩阵（E5）与批量执行 | M3 实验矩阵与结论 | 优先级:P0 类型:实验 难度:进阶 | [`14-mix-ratio-matrix.md`](14-mix-ratio-matrix.md) |
| [#15](https://github.com/DrGlitch666/AIGC-in-Face-Recognition-Dataset-Augmentation/issues/15) | [文档] 结果汇总、结论与局限 docs/RESULTS.md | M3 实验矩阵与结论 | 优先级:P0 类型:文档 难度:中等 | [`15-results-conclusion.md`](15-results-conclusion.md) |
| [#16](https://github.com/DrGlitch666/AIGC-in-Face-Recognition-Dataset-Augmentation/issues/16) | [网站] 静态展示站骨架：Vite + React + ECharts + 数据契约 | M4 网站与发布 | 优先级:P1 类型:网站 难度:中等 | [`16-site-skeleton.md`](16-site-skeleton.md) |
| [#17](https://github.com/DrGlitch666/AIGC-in-Face-Recognition-Dataset-Augmentation/issues/17) | [网站] 展示内容页：方法图、图库对比、指标看板、公平性与失败案例 | M4 网站与发布 | 优先级:P1 类型:网站 难度:中等 | [`17-site-content.md`](17-site-content.md) |
| [#18](https://github.com/DrGlitch666/AIGC-in-Face-Recognition-Dataset-Augmentation/issues/18) | [合规] 伦理与许可文档 + README/LICENSE + v0.1 发布 | M4 网站与发布 | 优先级:P1 类型:合规 难度:入门 | [`18-ethics-and-release.md`](18-ethics-and-release.md) |
