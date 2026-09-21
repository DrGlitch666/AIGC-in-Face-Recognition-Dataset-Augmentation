---
title: "[文档] 背景知识地图与文献综述 docs/BACKGROUND.md"
slug: "04-background-survey"
labels: ["优先级:P0", "类型:文档", "难度:入门", "good first issue"]
milestone: "M0 脚手架与地基"
---

## 背景 / 为什么做这个

这是「学习线」的核心任务，也是唯一一个**不需要写代码、可以在等下载/等训练时做**的任务。

项目最大的风险不是技术，而是**你说不清自己在做什么**。读文献不是为了引用好看，而是为了回答四个问题：

1. 这个领域已经做到哪一步了？（别人用合成数据训练 FR，最好能做到什么程度？）
2. 有哪些**现成的工具/权重**可以直接用？（能省掉几周的工作量）
3. 评价这件事的**标准做法**是什么？（哪些数据集、哪些指标、怎么比才公平）
4. 常见的坑和争议是什么？（合成数据真的有用吗？有没有反例？）

## 目标

产出一份**自己写的**、能作为项目「共同语言」的背景文档；读完能对着白板讲 5 分钟。

## 任务清单

- [ ] 读 **15 篇左右**核心论文（建议配比：3 篇综述 + 6 篇合成数据集/生成方法 + 3 篇身份条件生成 + 3 篇评测/公平性/域差距分析）
- [ ] 写 `docs/BACKGROUND.md`，包含：
  - [ ] **术语表**：把 `docs/FRAMEWORK.md` §10 的术语表补全成你自己的一句话解释（不许复制粘贴）
  - [ ] **15 篇论文摘要卡**：每篇 100~200 字，固定四段式——`方法一句话 / 用了什么数据 / 关键结论（带数字）/ 与本项目的关系`
  - [ ] **表 1｜数据集对照表**：名称 / 规模 / 真实还是合成 / 许可 / 能否公开 / 本项目是否使用
  - [ ] **表 2｜生成方法对照表**：名称 / 年份 / 范式（GAN/扩散）/ 是否身份条件 / 权重是否可得 / 显存需求 / 是否可在 8 GB 上跑
  - [ ] **表 3｜评测指标对照表**：指标名 / 衡量什么 / 怎么算 / 常见陷阱
  - [ ] **争议与反例**：至少 3 条「合成数据没那么有用 / 有副作用」的证据（域差距、人群偏向、可检测性）
- [ ] 把 15 篇论文的链接整理成 `docs/references.md`（或 BACKGROUND 附录）
- [ ] 在 Issue 评论里贴出你的「三个最大收获」

## 验收标准（Definition of Done）

- [ ] `docs/BACKGROUND.md` 里 15 篇摘要齐全，每篇都有可点击的一手链接（论文或官方仓库）
- [ ] 三张表格填满，其中**「本项目是否使用 / 是否可在 8 GB 上跑」两列必须有明确判断**
- [ ] 能用文档里的内容直接回答 `docs/FRAMEWORK.md` §2 的 RQ1~RQ5（写一段映射说明）
- [ ] 没有整段英文原文粘贴，全部是自己的中文归纳

## 交付物

- `docs/BACKGROUND.md`
- `docs/references.md`

## 依赖

- 无（可与其他任务并行；建议在等数据下载、等模型训练时推进）
- 产出被依赖：{{#10-identity-preserving-generation}}（选生成器时看表 2）、{{#15-results-conclusion}}（写结论时对照别人的数字）

## 预估工作量

6~10 小时（可分多次完成，每次 1~2 篇）

## 参考

**起点（先把这几篇读完，再顺着引用往下滚）：**
- [insightface 官方仓库](https://github.com/deepinsight/insightface)——ArcFace 与其生态的实现都在这里
- [Awesome-Face-Forgery-Generation-and-Detection](https://github.com/clpeng/Awesome-Face-Forgery-Generation-and-Detection)——按主题找论文的索引
- 关键词组合（用于检索）：`synthetic face dataset face recognition`、`identity-preserving face generation`、`synthetic-to-real domain gap face recognition`、`demographic bias synthetic faces`
- 检索入口：[arXiv](https://arxiv.org/list/cs.CV/recent)、[Papers with Code](https://paperswithcode.com/)、[Semantic Scholar](https://www.semanticscholar.org/)

## 附：学习线与本任务的关系（怎么把"读论文"变成真的学会）

本任务负责**论文产出**（`docs/BACKGROUND.md`）；**学习过程与自测**由 {{#21-learning-path}} 负责，两者互补、共用同一份阅读清单。

- **阅读顺序**、**每篇该抓什么**、**配套的动手练习与自测题**，见 [`LEARNING.md`](../LEARNING.md) 的 **L1~L6** 各阶段；本任务的 15 篇摘要卡对应 L6。
- 建议的**分工做法（两人时很有效）**：
  - 每人负责一半论文，各写摘要卡，然后**互相讲给对方听**（每人每周 15 分钟）。
  - 讲不清楚的地方 = 没读懂的地方，回去重读那一节。
- **摘要卡四段式**（`方法一句话 / 用了什么数据 / 关键结论（带数字）/ 与本项目的关系`）是硬要求，它同时服务于三处：本任务的 `BACKGROUND.md`、{{#10-identity-preserving-generation}} 的选型依据、{{#15-results-conclusion}} 的文献对照。
- 遇到"读不懂但很重要"的论文，**不要卡住**：先记一句"这里没懂 + 为什么重要"，在本 Issue 下建一条评论，继续往下读；很多概念会在 L2/L3 动手之后自然就懂了。

## 新手提示 / 卡住了怎么办

- **摘要卡四段式**是硬要求：读完一篇立刻写，不要攒着最后写，否则等于没读。
- 读不懂数学推导就跳过，先抓「输入是什么、输出是什么、怎么评测、结论数字是多少」。
- 表格里凡是「权重是否可得」这一列，**一定要点开链接确认**，很多论文没有公开权重——这直接决定 {{#10-identity-preserving-generation}} 能不能做。
- 时间不够时的降级：先把 3 篇综述读透 + 填完表 2（生成方法对照表），剩下的摘要分次补。
