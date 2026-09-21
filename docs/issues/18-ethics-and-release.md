---
title: "[合规] 伦理与许可文档 + README/LICENSE + v0.1 发布"
slug: "18-ethics-and-release"
labels: ["优先级:P1", "类型:合规", "难度:入门"]
milestone: "M4 网站与发布"
---

## 背景 / 为什么做这个

生成人脸这件事有明确的双刃剑属性：**它既能保护隐私（不需要真人数据），也能被用来伪造他人身份**。一个公开的项目如果对这一点只字不提，是不负责任的；而对做项目的人来说，把法律与伦理边界写清楚，也是**保护自己**。

同时，这一步完成项目的「交付形态」：能被别人看懂、能引用、能复现、有明确的使用边界。

## 目标

一份可公开的伦理与许可文档 + 一个完整的仓库门面 + 一个打好 tag 的 v0.1 版本。

## 任务清单

- [ ] 写 `docs/ETHICS.md`，包含以下小节：
  - [ ] **项目声明**：研究用途、明确不用于监控/身份核验/冒充他人
  - [ ] **数据许可表**：逐个数据集列出「是否使用 / 能否本地留存 / 能否提交到仓库 / 引用来源」
  - [ ] **合成数据声明**：所有生成图的来源模型、许可证、生成参数、以及「本仓库只提交我们生成的图，不提交任何真实人脸数据」
  - [ ] **法律边界要点**（简要，带权威链接）：
    - [ ] GDPR 第 9 条：用于唯一识别自然人的生物特征数据属于特殊类别，处理受严格限制
    - [ ] 欧盟 AI Act 第 5 条：禁止基于生物特征推断种族/政治倾向/宗教/性取向等的分类系统；禁止公共场所实时远程生物识别（执法例外极窄）
    - [ ] 美国伊利诺伊州 BIPA：收集生物标识前需书面告知与书面同意，且有私人诉讼权
    - [ ] 中国《个人信息保护法》：生物识别信息属敏感个人信息，需**单独同意**（第 28/29 条）
  - [ ] **双刃剑风险与对策**：
    - [ ] 身份一致的生成器可用于制作**变形攻击（morphing attack）**样本，这是真实存在的滥用路径
    - [ ] 合成数据集**可能泄漏**真实训练样本（成员推断攻击已被证实），所以不能简单宣称「合成 = 隐私安全」
    - [ ] 对策：公开生成参数与来源、保留可检测性报告（{{#12-synth-detection-report}}）、不使用真实人物照片做身份克隆
  - [ ] **Do / Don't 清单**（照 `docs/FRAMEWORK.md` §8 展开）
  - [ ] **合规自查记录**：勾选确认「仓库内无真实人脸数据」「所有生成图已标注」「已引用各数据集许可」
- [ ] 数据许可要点（写作时直接采用以下已核实结论）：
  - [ ] **LFW**：研究用途可自由获取，可引用与链接
  - [ ] **CelebA**：条款明确禁止再分发，**绝不能提交图片**
  - [ ] **MS-Celeb-1M / MS1M**：微软已于 2019 年**撤回**，无有效许可，第三方镜像不可作为许可依据
  - [ ] **VGGFace2**：已撤回，事实上不可获取
  - [ ] **CASIA-WebFace**：需点击同意，非商业研究，禁止再分发
  - [ ] **AgeDB**：仅限非商业研究
  - [ ] **RFW**：条款明确写明「不得复制或分发」，需要密码获取
  - [ ] **对齐/裁剪后的版本属于衍生数据，继承同样的限制**
- [ ] 更新 `README.md`：项目简介、**结论摘要（3~5 句 + 一张图）**、目录结构、快速开始、数据获取、复现步骤、致谢、引用方式、免责声明
- [ ] 写 `LICENSE`（代码许可证，如 MIT / Apache-2.0）+ 在 README 中注明「**代码许可证不覆盖数据集与第三方模型权重**」
- [ ] 写 `CITATION.cff`（让别人能引用你的项目）
- [ ] **合规自查**（逐条执行并记录）：
  - [ ] `git log --stat --all | grep -iE '\.(jpg|jpeg|png|npy|rec|bin|pth|onnx)'` 无输出（或只有自己生成的、已声明许可的图）
  - [ ] 仓库总大小合理（`git count-objects -vH`）
  - [ ] 生成图目录里有 `NOTICE` / `README` 说明其来源与许可
- [ ] 发布 `v0.1.0`：打 tag + Release 说明（包含：做了什么、结果摘要、已知限制、如何复现）
- [ ] 更新 `docs/RESULTS.md` 与网站 `/ethics` 页，与本文件口径一致

## 验收标准（Definition of Done）

- [ ] `docs/ETHICS.md` 覆盖上述所有小节，每个法律要点带权威链接
- [ ] 仓库里**没有任何真实人脸数据集文件或对齐后的衍生文件**
- [ ] 所有发布的生成图都有明确来源与许可说明
- [ ] `README.md` 含结论摘要与复现命令，且结论与 `docs/RESULTS.md` 一致
- [ ] `LICENSE` + `CITATION.cff` 存在
- [ ] `v0.1.0` tag 与 Release 已发布，说明里含已知限制
- [ ] 合规自查的每条命令都执行过并把输出贴在 Issue 评论里

## 交付物

- `docs/ETHICS.md`
- 更新后的 `README.md`、`LICENSE`、`CITATION.cff`
- `v0.1.0` tag + GitHub Release
- 合规自查记录（Issue 评论）

## 依赖

- 需要 {{#12-synth-detection-report}}（可检测性结论）
- 需要 {{#15-results-conclusion}}（README 的结论摘要）
- 建议在 {{#17-site-content}} 之前完成，网站的 `/ethics` 页直接复用本文件

## 预估工作量

半天到 1 天

## 参考（务必用一手来源核对）

**法律与政策**
- GDPR 第 9 条（特殊类别数据）：<https://gdpr-info.eu/art-9-gdpr/>
- 欧盟 AI Act 第 5 条（禁止类实践）：<https://artificialintelligenceact.eu/article/5/>
- 欧盟 AI Act 实施时间线（禁止类 2025-02-02 起；高风险主要义务 2026-08-02 起）：<https://artificialintelligenceact.eu/implementation-timeline/>
- 伊利诺伊州 BIPA（740 ILCS 14）：<https://www.ilga.gov/legislation/ilcs/fulltext?DocName=074000140K1>
- 中国《个人信息保护法》官方英文版：<https://en.spp.gov.cn/2021-12/29/c_948419_2.htm>
- GitHub 合成媒体与 AI 工具政策：<https://docs.github.com/en/site-policy/acceptable-use-policies/github-synthetic-media-and-ai-tools>

**数据集许可**
- [CelebA 官方协议](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)（明确禁止再分发）
- [AgeDB / ibug 页面](https://ibug.doc.ic.ac.uk/resources/agedb/)（非商业研究）
- [RFW 官方页面](http://whdeng.cn/RFW/testing.html)（禁止复制与分发）
- [MS-Celeb-1M 撤回相关报道](https://www.bbc.com/news/technology-48555149)

**风险证据（写文档时引用，不要说空话）**
- 合成数据集可被成员推断攻击泄漏真实样本：[arXiv:2410.24015](https://arxiv.org/abs/2410.24015)
- 身份一致生成器可被用于变形攻击（Arc2Morph 用 Arc2Face 造护照级 morph）：[arXiv:2602.16569](https://arxiv.org/abs/2602.16569)
- 生成器继承其训练数据的人口统计分布：[arXiv:2311.03970](https://arxiv.org/abs/2311.03970)
- 合成图上的偏见审计可能得出误导结论：[arXiv:2608.23593](https://arxiv.org/abs/2608.23593)

## 附：已核实要点（框架预置，2026-09）——**模型许可陷阱清单**

写「合成数据声明」和「本项目使用的第三方组件」两节时，逐个核对下表：

| 组件 | 许可要点 |
|---|---|
| insightface **代码** | MIT |
| insightface **模型包**（`buffalo_l` / `antelopev2` 等） | **仅限非商业研究用途** |
| Arc2Face | 代码 MIT，但**依赖 insightface 人脸模型** → 继承上面的非商业限制 |
| InstantID / IP-Adapter-FaceID | 权重本身为 **research-only**，且同样依赖 insightface 模型 |
| PuLID-FLUX | 需 11~16 GB 显存；继承 **FLUX.1-dev 的非商业许可** |
| pyiqa（画质指标库） | **PolyForm Noncommercial 1.0.0** |
| DeepfakeBench | **CC BY-NC 4.0** |
| IDiff-Face 数据集 | **CC BY-NC-SA 4.0** |
| DigiFace-1M | R-UDA，非商业 |

> **一句话写进文档**：本项目为**非商业研究**用途；所依赖的多个模型包与数据集**仅限非商业研究**；**「代码是 MIT」不等于「整条链路可商用」**。

另外，写「合成数据声明」时请一并说明**双刃剑风险**（已在正文给出可引用的证据：morphing 攻击、成员推断泄漏），并说明本项目的对策（公开生成参数、保留可检测性报告、不用真人照片做身份克隆）。

## 新手提示 / 卡住了怎么办

- 写法律部分时**只用一手来源**（法条原文、官方数据集页面），引用二手博客容易出错。
- 措辞原则：**说清楚适用范围**。不写「本项目完全合规」，而写「本项目的数据来源与处理方式如下，法律适用取决于使用者所在司法辖区」。
- 「我们没有做任何违法的事」这句话不要写；写「我们做了什么、没做什么、依据是什么」。
- **降级方案**：时间不够时，最低限度是 `docs/ETHICS.md` 的「数据许可表 + 合成数据声明 + Do/Don't」三节 + README 更新；Release 说明可以简化但不能没有。

