# 工作流（Workflow）

> 本文档回答：**按什么顺序做、每步怎么验收、卡住了怎么降级。**
> 架构与设计依据见 [`FRAMEWORK.md`](FRAMEWORK.md)；具体任务卡见 [`issues/`](issues/)。

---

## 0. 总览

### 0.1 一句话工作流

```
挑一个 Issue → 建分支 → 按验收标准做完 → 跑通复现命令 → 提交 PR（Closes #N）→ 合并 → 更新看板
```

### 0.2 六个 Sprint（总预算 4~6 周业余时间）

| Sprint | 名称 | 工期 | 对应 Milestone | 出口条件（必须全绿才进入下一阶段） | 主导（按档位） |
|---|---|---|---|---|---|
| **S0** | 地基 | 2~3 天 | M0 脚手架与地基 | 环境报告生成、**档位与分工定稿**、数据落盘、manifest 契约冻结、背景文档成型 | 两人并行 |
| **S1** | 基线与评测 | 3~5 天 | M1 基线与评测 | 有 E0/E1 两组数字，且**能一条命令重跑**（两台机器都能跑） | 强机训练 / 弱机评测 |
| **S2** | 生成与筛选 | 5~7 天 | M2 生成与筛选 | 生成→筛选→入库闭环跑通，图库里能看到「同人不同姿态」 | 强机生成 A 轨 / 弱机 B 轨与筛选 |
| **S3** | 实验与结论 | 7~10 天 | M3 实验矩阵与结论 | E2~E5 矩阵跑完（≥1 seed，**同组实验同机跑完**），结论与局限写完 | 强机跑批 / 弱机汇总分析 |
| **S4** | 网站 | 4~6 天 | M4 网站与发布 | 网站能本地打开，所有图表来自真实 `metrics.json` | 任意机器（建议弱机主导） |
| **S5** | 收尾发布 | 2~3 天 | M4 网站与发布 | 伦理文档、README、Release v0.1、补 seed | 两人分工 |

> ⏱️ **时间失控的唯一原因**通常是：在某一层追求完美。每个 Sprint 结尾必须停下来做一次「是否降级」的判断（见 §5）。
>
> 👥 **两条线并行**：Sprint 是**工程线**的节奏；**学习线**（[`LEARNING.md`](LEARNING.md) 的 L0~L6）贯穿全程，每个 Sprint 开始前集中学对应阶段。学习线的产出物直接回填到对应 Issue，不做"纯学习"的无效功。

### 0.3 任务依赖图

```mermaid
flowchart LR
    subgraph M0[协作地基]
        I19[#19 协作规范]
        I20[#20 硬件档案与配置分层]
        I21[#21 学习路线]
    end
    I1[#1 环境与档位判定] --> I2[#2 脚手架]
    I19 --> I2
    I20 --> I2
    I2 --> I3[#3 数据管线]
    I20 --> I3
    I3 --> I5[#5 零样本基线]
    I3 --> I6[#6 训练管线]
    I5 --> I7[#7 评测器]
    I6 --> I7
    I7 --> I8[#8 传统增强对照]
    I2 --> I9[#9 无条件生成]
    I3 --> I10[#10 身份保持生成]
    I9 --> I10
    I10 --> I11[#11 筛选门禁]
    I11 --> I12[#12 合成检测报告]
    I11 --> I13[#13 属性可控生成]
    I11 --> I14[#14 消融实验矩阵]
    I7 --> I14
    I14 --> I15[#15 结论与局限]
    I15 --> I17[#17 网站内容页]
    I2 --> I16[#16 网站骨架]
    I16 --> I17
    I4[#4 背景文献] --> I10
    I21 --> I4
    I18[#18 伦理与发布] --> I17
```

**两人并行安排**（按档位分工，不按"谁有空"分工）：

| 时间 | 强机一方（A/B 档） | 弱机一方（B/C 档） | 交汇点 |
|---|---|---|---|
| S0 | #1 环境与档位判定、#20 配置分层 | #3 数据管线、#19 协作规范、#21 学习路线 | 档位与契约定稿 |
| S1 | #6 训练管线 | #5 零样本基线、#7 评测器 | E0/E1 数字对齐 |
| S2 | #10 A 轨生成 | **B 轨下载现成合成数据**、#11 筛选、#9 无条件生成 | 生成清单合并 |
| S3 | #14 矩阵跑批（整组同机） | #12 检测报告、#15 结论初稿、#16 网站骨架 | 结果汇总 |
| S4 | 补 seed、性能优化 | #17 网站内容、#18 伦理与发布 | 网站验收 |
| 全程 | 各自推进 #4 论文精读（每周 2 篇，见 #21） | 同左 | 每周互讲 15 分钟 |

> 一个人做也不是不行——那就把"弱机一方的任务"当作训练等待期间的填充任务，节奏见 §7。

---

## 1. 标准工作流（每个 Issue 都照这个走）

### 1.1 六步循环

| 步骤 | 动作 | 产出 |
|---|---|---|
| 1. 选卡 | 从看板挑一个「没有阻塞依赖」的 Issue，把它拖到 `In Progress` | — |
| 2. 建分支 | `git checkout -b feat/12-synth-detector` | 分支 |
| 3. 实现 | 按 Issue 的「任务清单」逐条打勾 | 代码/文档 |
| 4. 自验 | 按 Issue 的「验收标准」逐条验证，**跑一遍复现命令** | 证据（截图/输出/JSON） |
| 5. 提交 | `git commit -m "feat(filter): add synth detector report (#12)"`；PR 描述里写 `Closes #12` | PR |
| 6. 收尾 | 合并后：更新 `docs/`、把结果写进 `results/`，Issue 自动关闭 | 更新的文档 |

### 1.2 分支命名

```
feat/<issue号>-<短描述>     新功能      feat/10-identity-preserving-gen
exp/<issue号>-<短描述>      实验        exp/14-mix-ratio-matrix
docs/<issue号>-<短描述>     文档        docs/4-background-survey
fix/<短描述>                修 bug      fix/manifest-path-normalize
chore/<短描述>              杂务        chore/add-gitignore-data
```

### 1.3 Commit 规范（够用版）

```
<类型>(<范围>): <做了什么> (#<issue号>)
类型：feat | fix | docs | exp | chore | test | refactor
范围：data | generate | filter | train | eval | site | docs
示例：exp(train): add real+synth mix ratio 0.25 config (#14)
```

### 1.4 每个 Issue 完成的定义（DoD）

一个 Issue 只有在下面 5 条全部满足时才算完成：

- [ ] Issue 里的「任务清单」全部勾选，或明确写明哪几条被有意跳过及原因
- [ ] 「验收标准」逐条有证据（命令输出 / 文件路径 / 截图）
- [ ] **有可复现入口**：一条命令能从原始状态重跑（脚本或 `scripts/` 里的入口）
- [ ] 产物已落盘到约定路径（`results/`、`docs/`、`configs/`），而不是留在临时文件里
- [ ] 相关文档（README / FRAMEWORK / RESULTS）已同步更新

> 🚫 **反模式**：Jupyter Notebook 里跑出结果，但不落盘、不写脚本、不留配置。这样一周后你自己都复现不出来。

---

## 2. 命令约定（统一入口，别记一堆参数）

所有操作尽量走统一入口，便于写进文档和 CI：

```bash
# 环境自检（同时判定本机属于 A / B / C 哪一档，写入 reports/env_report.md）
python scripts/check_env.py

# 数据：下载 / 对齐 / 建 manifest
python scripts/download_data.py  --config configs/data/lfw.yaml
python scripts/build_dataset.py  --config configs/data/toy_train.yaml

# 生成 / 筛选
python scripts/generate.py       --config configs/gen/ipadapter_faceid.yaml
python scripts/filter.py         --config configs/filter/default.yaml

# 训练 / 评测
python scripts/train.py          --config configs/exp/e4-ipadapter-r25-seed0.yaml
python scripts/evaluate.py       --run results/runs/e4-ipadapter-r25-seed0

# 汇总 / 导出给网站
python scripts/summarize.py      --runs results/runs --out results/summary.csv
python scripts/export_site_data.py --results results --out site/public/data
```

**配置三层**（详见 [`FRAMEWORK.md`](FRAMEWORK.md) §4.3 与 {{#20-hardware-profiles}}）：

```bash
# 每次运行都是「科学设定 + 档位资源设定 + 机器私有设定」三层叠加
python scripts/train.py --config configs/exp/e4-ipadapter-r25-seed0.yaml \
                        --profile b \
                        --local   configs/local.my-laptop.yaml
```

- `--profile auto`（默认）会**读取 env_report.md 的档位判定**自动选择 a/b/cpu。
- `--local` **可以省略**；文件已被 `.gitignore` 忽略，每人一份、互不影响。
- 脚本必须把**实际生效的 config/profile/local 三者合并结果与哈希**写进 `results/runs/<exp_id>/config.resolved.yaml`——这是跨机器协作能追溯的前提。

**约定**：每个脚本都必须支持 `--config`、`--profile`、`--local` 和 `--dry-run`（先打印将要做什么，不实际执行）。这样跑批之前你能先确认，也方便另一台机器复现。

---

## 3. 实验工作流（这是项目的核心节奏）

### 3.1 加一个实验的标准流程

1. **写配置**：复制 `configs/exp/_template.yaml`，填写数据、模型、训练、评测四段。`exp_id` 遵循 `{阶段}-{生成器}-{比例}-{seed}`。
2. **注册到矩阵**：在 `configs/exp/matrix.yaml` 里加一行（用于 #14 的批量执行）。
3. **小规模试跑**：`--dry-run` → 用 2 个身份 1 个 epoch 试跑，确认管线通。
4. **正式跑**：训练 → 评测 → `results/runs/<exp_id>/metrics.json` 落盘。
5. **登记**：`python scripts/summarize.py` 更新 `results/summary.csv`。
6. **一句话结论**：写进 `docs/RESULTS.md` 对应表格行（**只写事实，不写感受**）。

### 3.2 结果记录纪律

| 规则 | 原因 |
|---|---|
| 每个实验至少 3 个 seed，报均值±标准差 | 单次结果在小数据集上噪声极大 |
| 失败的实验也要保留记录（`status: failed` + 原因） | 「试过但没用」本身是结论，也避免重复踩坑 |
| 不删掉「难看」的结果 | 选择性报告 = 学术不端 |
| 每次跑完记录 GPU 型号与耗时 | 用于 RQ5（最低可行配方） |

### 3.3 每周固定动作（周日 30 分钟）

- [ ] 更新 `results/summary.csv`，把本周新增的 run 汇总
- [ ] 更新 `docs/RESULTS.md` 的表格与「本周发现」
- [ ] 看板回顾：这周哪个 Issue 卡住了？卡在**技术**还是**范围**？写下周的唯一优先事项
- [ ] 检查 `data/` 没被误提交（`git status` 里不该出现数据文件）

---

### 3.4 跨机器可比性（两台机器协作的硬规则）

两台机器配置不同（显存、CUDA 版本、甚至有没有独显），这会从三个地方污染实验结论：

| 污染源 | 后果 | 规则 |
|---|---|---|
| **batch size 不同** | 小数据集 + 归一化层下，batch 会真实影响结果 | 同一组对比实验**必须同机同 profile 跑完**；跨机时必须固定 batch 并在报告标注 |
| **数值精度不同**（AMP/CPU vs GPU） | 指标出现 0.1%~0.5% 的系统性偏移 | 对比实验统一精度；CPU 结果单独标注「CPU 档位」 |
| **依赖版本不同** | 预处理/评测实现行为差异 | Python 次版本统一；`environment.yml` 通用层入库；评测代码必须有跨机器通过的测试 |

**三条执行规则**：
1. **按"整组实验"分配机器，不按 seed 拆分**——例如「E5 的 15 个 run」整组交给同一台机器；另一台机器去做别的实验或别的层次。
2. **每个 run 必须记录硬件上下文**：`metrics.json` 里的 `hardware.profile / gpu_name / batch_size / amp / torch_version`（契约 B 已含），没记录的 run 不参与跨机比较。
3. **汇总表里分组呈现**：`results/summary.csv` 增加 `profile` 列；如果同一实验横跨两台机器，图表上要能区分（或用不同标记）。

> 💡 **实用建议**：本项目规模很小（几十个身份），训练一次的耗时通常可以接受——所以**优先选择"同机跑完"**，而不是费劲去对齐跨机器差异。

---

## 4. 看板与标签使用

| 标签 | 含义 | 怎么用 |
|---|---|---|
| `优先级:P0` | 不做完项目就断了 | 永远优先做 P0 |
| `优先级:P1` | 影响结论完整性 | 主干做完再做 |
| `优先级:P2` | 加分项 | 时间富余才做，可放弃 |
| `难度:入门` / `难度:中等` / `难度:进阶` | 预估难度 | 状态差/时间碎的时候挑「入门」 |
| `类型:*` | 环境/数据/生成/评测/实验/网站/文档 | 用于按类型筛选批量处理 |
| `good first issue` | 新手起手卡 | **第一个动的 Issue 从这里挑** |

**里程碑 = Sprint**。每个 Milestone 下的 Issue 全部关闭才算该阶段结束。

> 建议视图：Board 按 `Status` 分组（Todo / In Progress / Blocked / Done），再加一个筛选器 `label:优先级:P0`。

---

## 5. 决策卡点（Go / No-Go）与降级策略

每个 Sprint 结束时，用 5 分钟做一次判断。**降级不是失败，烂尾才是。**

### S0 出口判断
- 环境自检报告里有 GPU / CPU 信息，并**明确写出本机档位（A/B/C）**？→ 否则**先解决再往下走**（这是唯一不能绕过的卡点）
- 两台机器的档位都判定完，且 `docs/` 里写清了谁负责哪一域？→ 否则先做 {{#19-collaboration-protocol}} 与 {{#20-hardware-profiles}}
- 数据集下载成功且 manifest 能校验通过？→ 否则降级为「只用 LFW 一个数据集」

### S1 出口判断
- E0（预训练零样本）数字是否接近公开报道（LFW 通常 99%+）？
  - 是 → 评测管线可信，继续。
  - 否 → **停下修评测管线**，不要继续做生成。管线错了后面全白做。
- E1（real-only 自训练）数字是否明显低于 E0？→ 正常（数据少），记录下来即可。

### S2 出口判断
- 生成图**人工看**是否「像同一个人」？
  - 像 → 继续放量生成。
  - 不像/糊 → 调 prompt、换参考图、降分辨率要求；仍不行 → 触发**降级方案 A**：改用公开合成数据集 + 无条件生成，把 E4 标记为「未能验证」。
- 筛选通过率是否 ≥ 30%？→ 过低说明生成质量差，先修生成而不是放低阈值。

### S3 出口判断
- E2/E3/E4 相对 E1 的差异是否**大于 seed 噪声**？
  - 是 → 得到有效结论，进入写作。
  - 否 → 结论写成「在本规模下未观察到显著增益」+ 分析原因（这是**合格的负结果**，不要硬凑）。
- 时间是否已超预算 30%？→ 立刻执行**降级方案 B**：只保留 E1/E4 两组 × 1 seed，砍掉 E3/E5/E6。

### S4 出口判断
- 网站是否**不依赖本地服务**就能打开（纯静态）？
- 所有数字是否都能追到 `metrics.json`？（不许手写数字到网页里）

### 三档交付（提前决定，事到临头不纠结）

| 层级 | 包含 | 适用 |
|---|---|---|
| **最小可交付** | E0/E1/E4 + 单 seed + 6 个基础页面 | 时间只剩一半 |
| **标准交付** | 上面 + E2/E3 + 3 seed + 公平性/失败案例页 + 伦理文档 | 默认目标 |
| **加分交付** | 上面 + E5 比例曲线 + E6 属性可控 + 合成检测报告 + CI + 在线 Pages | 时间富余 |

---

## 6. 常见坑清单（跨平台 + 跨机器 + 新手）

| 坑 | 症状 | 处理 |
|---|---|---|
| Python 版本不一致 | 两人依赖解析结果不同、行为有细微差异 | 约定同一个次版本（都 3.11 或都 3.12）；写进 `environment.yml` |
| 照抄别人的 torch 安装命令 | `no kernel image is available` / 装成 CPU 版 | 按**自己机器**的 CUDA 版本选轮子（新卡 RTX 50 系需 cu128 + torch ≥ 2.7）；先 `torch.cuda.get_arch_list()` 确认 |
| 跨平台差异 | macOS/Linux 上 ONNX/OpenCV 行为不同 | ONNX Runtime 按平台选 EP（CUDA / CoreML / CPU）；路径统一 `pathlib` |
| 显存不够 | `CUDA out of memory` | 由**档位**决定 batch/分辨率（{{#20-hardware-profiles}}）；fp16、CPU offload、生成与训练错峰 |
| Windows 路径 | 中文/空格路径导致库报错 | 项目路径保持英文；Python 里统一用 `pathlib` |
| 数据集路径写死 | 换机器跑不了 | 全部走 YAML 配置；机器私有路径只写在 `configs/local.<machine>.yaml`（不入库） |
| 对齐方式不一致 | 评测数字莫名偏低 | 训练与评测必须用**同一套** 5 点对齐模板（112×112） |
| 数据泄漏 | 数字异常高（比论文还高） | 检查训练身份与评测身份是否重叠；toy 测试集必须身份隔离 |
| 只测一次就下结论 | 结论翻车 | 每个配置 ≥ 3 seed，看均值±方差 |
| 跨机器拼实验 | 两半结果对不上、无法解释 | 同一组实验同机跑完（§3.4）；`summary.csv` 带 `profile` 列 |
| 把数据提交进 Git | 仓库巨大 / 隐私问题 | `.gitignore` 写死 `data/`、`*.pth`、`*.onnx`、`configs/local.*.yaml`、`outputs/` |
| 两人同改一个文件 | 频繁冲突、互相覆盖 | 认领目录；`summary.csv`/`data/*.json` 只由脚本生成 |
| 长时间盯着一个坑 | 一天就没了 | **卡 2 小时规则**：卡满 2 小时 → 记录到 Issue 评论 → 换任务或走降级方案 |

### 卡住时的处理顺序

```
报错 → 完整读一遍错误信息 → 搜错误原文（带库名版本） → 看官方 issue 区
  → 2 小时还没解决 → 在该 Issue 下写评论（贴报错+已试过什么）→ 换一个不阻塞的任务
  → 连续 2 天卡同一处 → 执行该 Issue 的降级方案，把它拆小或换实现
```

---

## 7. 双人协作与每周节奏

### 7.1 分工与三条硬规则

分工原则（谁负责哪一域）见 [`FRAMEWORK.md`](FRAMEWORK.md) §11.1；这里只列**每天都要遵守**的三条：

1. **开工先 `git pull`，收工前 `git push`**——两个人在不同机器上，未推送的工作对另一个人等于不存在。
2. **一个 Issue 一个分支一个人**——不要两人同时做同一张卡；如果必须协作，拆成两张子卡。
3. **合并前在另一台机器上验证过一次**（至少 CI/测试通过），这是防止"只在我机器上能跑"的唯一办法。

### 7.2 每周节奏（业余时间版）

| 时段 | 谁 | 内容 | 时长 |
|---|---|---|---|
| 周中碎片时间 | 两人 | 「入门」难度任务：写文档、下数据、整理结果表、读 1 篇论文 | 30~60 min/次 |
| 周中整块时间 | 强机一方 | 「进阶」任务：调生成、跑训练 | 2~3 h |
| 周中整块时间 | 弱机一方 | 数据/评测/筛选/网站/文档，或 B 轨现成合成数据的接入 | 2~3 h |
| 周末 | 强机一方 | 启动跑批实验（训练很慢，让它自己跑） | 半天 |
| 周末 | 两人 | 周回顾 30 min：结果汇总 + 下周分工 + 是否降级 | 30 min |

**关键**：把「等待型任务」（下载、生成、训练）安排在离开电脑前启动，回来收结果。两台机器同时挂机，吞吐量翻倍。

### 7.3 每周 15 分钟同步（固定议程，别跑题）

1. **上周完成**：关了哪几张卡（对应 Issue 编号）。
2. **本周计划**：各自认领哪两张卡。
3. **卡点**：卡在哪、卡了多久、要不要执行降级方案。
4. **是否需要换分工**：某台机器被长时间占用时，另一人接手它的下一张卡。
5. **学习线**：互相讲 15 分钟（见 [`LEARNING.md`](LEARNING.md) §0）。

**所有结论写进 Issue 评论**，不要只留在聊天里。

### 7.4 新机器 / 新人 onboarding 清单（半天内完成）

- [ ] 按 {{#01-env-setup}} 建环境，产出本机的 `reports/env_report.md` 并判定档位（A/B/C）
- [ ] 按 {{#20-hardware-profiles}} 建好自己的 `configs/local.<machine>.yaml`（**不要提交**）
- [ ] `pytest -q` 全绿（契约测试是跨机器一致性的安全网）
- [ ] 能跑通两条最小命令：`scripts/evaluate.py --run <别人的 run>`、`scripts/summarize.py`
- [ ] 读一遍 [`FRAMEWORK.md`](FRAMEWORK.md) §3.3（契约）、§4.3（配置分层）、§11（协作）
- [ ] 在 {{#19-collaboration-protocol}} 下评论认领自己的职责域
- [ ] 从 `good first issue` 里挑第一张卡（推荐 {{#03-data-pipeline}} 或 {{#21-learning-path}}）

---

## 8. 下一步（现在就做这几件事）

1. **两人各自**跑 {{#01-env-setup}}，把 `reports/env_report.md` 提交上去，确认各自的档位（A/B/C）。
2. 按 {{#19-collaboration-protocol}} 在 Issue 里**认领职责域**，并把分工写进仓库文档。
3. 按 {{#20-hardware-profiles}} 建配置分层的骨架（`base.yaml` + `profiles/*` + `.gitignore` 里的 `local.*`）。
4. 按 {{#02-repo-skeleton}} 把仓库骨架建起来（目录 + `.gitignore` + 配置文件模板）。
5. 打开 [`LEARNING.md`](LEARNING.md) 走 **L0**（2~3 小时，两人各做一遍），然后按 {{#21-learning-path}} 排学习节奏。
6. 在 GitHub 上把 Milestone M0 建成看板，把 M0 的 7 张卡拖进 Todo。

> 记住附录 A（MVP 闭环）的定义：**先用最小成本把整条线走通一次**，再回来加厚。你已经有了完整的任务拆解，接下来只需要每天推进一张卡。
