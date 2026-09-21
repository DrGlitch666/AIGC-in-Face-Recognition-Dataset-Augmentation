---
title: "[协作] 双人协作规范：分工认领、PR 评审、数据与产物同步"
slug: "19-collaboration-protocol"
labels: ["优先级:P0", "类型:环境", "难度:入门", "good first issue"]
milestone: "M0 脚手架与地基"
---

## 背景 / 为什么做这个

**项目由两人在两台配置不同的机器上开发。** 这时最大的风险不是"人手不够"，而是**做出来的东西对不上**：

- 一个人本地能跑的脚本，另一个人跑不起来（路径、依赖、CUDA 版本不同）；
- 两人各自跑了一半的对比实验，合起来发现不可比（batch 不同、精度不同）；
- 同一个文件两人同时改，冲突花的时间比写代码还多；
- 数据/权重各存一份，谁也不知道对方用的是哪个版本。

这些问题的解药不是"多沟通"，而是**把约定写下来 + 把不确定性压到契约和档位里**。本任务负责"写下来"这一半（"压到契约里"由 {{#20-hardware-profiles}} 负责）。

> 💡 **两人项目不需要重型流程**：不要上 Git Flow、不要引入 code review 工具、不要开每日站会。**一个 PR 模板 + 一张分工表 + 每周 15 分钟**就够了。

## 目标

一份两人都认账、且**第一个 Sprint 就真的在用**的协作规范。

## 任务清单

- [ ] **认领职责域**：按 [`FRAMEWORK.md`](../FRAMEWORK.md) §11.1 的表格，两人各自认领（在 Issue 评论里公开写下来）：
      - 强机一方：A 轨生成 + 训练 + 矩阵跑批
      - 弱机一方：数据层 + B 轨 + 评测 + 筛选 + 网站 + 文档
      - 说明：**认领不等于独占**，只是"这块出问题找谁"
- [ ] 写 `docs/COLLABORATION.md`（或确认并入 `FRAMEWORK.md` §11），内容至少包含：
  - [ ] 职责域与认领结果（谁负责什么）
  - [ ] 分支命名与 commit 规范（引用 [`WORKFLOW.md`](../WORKFLOW.md) §1.2、§1.3）
  - [ ] **一台机器一个 profile** 的原则与 `configs/local.<machine>.yaml` 的使用方式
  - [ ] 冲突热点与对策表（`summary.csv` / `site/public/data/*` 只由脚本生成；`matrix.yaml` 一人维护）
  - [ ] 每周 15 分钟同步的固定议程
- [ ] 加 `.github/pull_request_template.md`，必填字段：
  - [ ] 关联 Issue（`Closes #N`）
  - [ ] **在哪台机器/哪个档位上验证过**
  - [ ] 复现命令（一行）
  - [ ] 是否改动了"科学设定"（base.yaml）——如果是，必须另一人确认
- [ ] 加 `.github/ISSUE_TEMPLATE/task.md`（任务卡模板，字段与 `docs/issues/` 保持一致）
- [ ] 确认 `.gitignore` 覆盖两人协作的私有文件：`configs/local.*.yaml`、`data/`、`*.pth`、`*.onnx`、`outputs/`、`site/node_modules/`
- [ ] **数据与权重同步方案**（三个人以上才需要 DVC，两人用最土的办法即可）：
  - [ ] manifest（`data/manifests/*.jsonl`）**进仓库**——它是数据的"索引"，两人靠它对齐
  - [ ] 原始数据与生成图**不进仓库**，靠 `scripts/download_data.py` 各自下载
  - [ ] 产出 `data/manifests/checksums.sha256`（或 `reports/data_manifest_lock.json`）：记录每个数据集的**文件数 + 校验和**，另一台机器跑一条命令就能确认"我们的数据是同一份"
  - [ ] 生成图/权重的传递方式二选一并在文档写清：局域网共享 / 网盘 / 谁生成谁训练（推荐后者，最省事）
- [ ] （可选但推荐）在 GitHub 仓库设置里开启分支保护：`main` 要求 PR + 1 个 approve
- [ ] 把"新机器 onboarding 清单"（[`WORKFLOW.md`](../WORKFLOW.md) §7.4）实际走一遍，记录卡住的地方

## 验收标准（Definition of Done）

- [ ] 职责域已在 Issue 评论里公开认领，且写进了仓库文档
- [ ] PR 模板存在，并且**已经真的用它提过一次 PR**（模板字段填写完整）
- [ ] `.gitignore` 验证生效：`git check-ignore -v configs/local.test.yaml` 有输出
- [ ] `data/manifests/checksums.sha256` 存在，另一台机器能据此验证数据一致性
- [ ] 两人都确认：每周同步议程与分支规范已读且同意
- [ ] onboarding 清单被第二个人实际走通一次（这是最真实的验收）

## 交付物

- `docs/COLLABORATION.md`（或 `FRAMEWORK.md` §11 的补充）
- `.github/pull_request_template.md`、`.github/ISSUE_TEMPLATE/task.md`
- 更新后的 `.gitignore`
- `data/manifests/checksums.sha256`
- Issue 评论里的职责认领记录

## 依赖

- 无（M0 起点，与 {{#01-env-setup}}、{{#20-hardware-profiles}} **并行推进**）
- 被依赖：全部任务（它是"怎么一起干活"的约定）

## 预估工作量

2~3 小时

## 参考

- [GitHub PR 模板文档](https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/about-issue-and-pull-request-templates)
- [GitHub 分支保护规则](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches)
- [`FRAMEWORK.md`](../FRAMEWORK.md) §11（双人协作规范）、§4.3（配置分层）
- [`WORKFLOW.md`](../WORKFLOW.md) §7（双人协作与每周节奏）

## 新手提示 / 卡住了怎么办

- **不要为了"规范"而规范**：每条规则都要能回答"不加这条会出什么事"。答不上来的就删掉。
- 最容易漏的是**数据版本对齐**——建议第一周就把 `checksums.sha256` 建起来，后面省很多"你那边数字怎么和我不一样"的扯皮。
- 如果两人时间完全错开（一个白天一个晚上），把"每周 15 分钟同步"改成**异步**：各自在 Issue 里更新「本周做了 / 下周计划 / 卡点」，另一个人 24 小时内回复。
