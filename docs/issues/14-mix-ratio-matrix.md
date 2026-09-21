---
title: "[实验] ★核心：混合比例消融实验矩阵（E5）与批量执行"
slug: "14-mix-ratio-matrix"
labels: ["优先级:P0", "类型:实验", "难度:进阶"]
milestone: "M3 实验矩阵与结论"
---

## 背景 / 为什么做这个

前面每个实验都只跑一个配置。但项目的核心问题——**「合成数据加多少最合适」**——只有一个办法能回答：**扫一遍比例，看曲线**。

这一步把项目从「跑了几个 demo」变成「做了一组实验」。它的产出是网站上最有说服力的那张图：**合成数据占比 vs 识别性能**。

同时它也是工程量最大的一步，因为要跑很多次训练。所以本任务的重点是**自动化与可恢复性**，不是算法。

## 目标

一条命令跑完整个实验矩阵，中途断了能续跑，最终产出一张可直接放进网站的汇总表 + 曲线图。

## 实验矩阵设计（默认配置）

| 维度 | 取值 | 说明 |
|---|---|---|
| 合成数据占比 | 0% / 10% / 25% / 50% / 100% | 0% 即 E1（real-only），100% 即纯合成 |
| 数据来源 | E2 传统增强 / E3 无条件生成 / E4 身份保持生成 | 至少跑 E4；有时间再加 E2、E3 |
| 随机种子 | 0 / 1 / 2 | **最少 1 个，目标 3 个** |
| 评测 | LFW + CFP-FP + AgeDB（+ 1:N + 分桶） | 与前面所有实验保持一致 |

**最小可交付矩阵**：E4 × 比例 {0, 25, 50} × seed {0} = 3 次训练。
**标准矩阵**：E4 × {0,10,25,50,100} × seed {0,1,2} = 15 次训练。
**加分矩阵**：再加 E2/E3 各 15 次。

## 任务清单

- [ ] 设计 `configs/exp/matrix.yaml`：把矩阵定义成「维度 × 取值」的笛卡尔积，而不是手写 15 个配置
- [ ] 实现配置生成器 `src/aigcfr/utils/matrix.py`：从 matrix.yaml 展开出所有 `exp_id` 与对应配置文件
- [ ] 实现批量执行 `scripts/run_matrix.py`：
  - [ ] **断点续跑**：已存在 `metrics.json` 的组合直接跳过（这是最重要的功能）
  - [ ] `--dry-run`：先打印将要跑哪些组合、预计耗时
  - [ ] `--only <pattern>`：只跑匹配的 subset（例如只跑 seed0）
  - [ ] 失败隔离：某个组合崩溃不影响后续组合，失败原因写入日志
  - [ ] 进度输出：`[7/15] exp_id ... done in 42min`（长时间运行必须能看出进度）
- [ ] **严格控制变量**（这是实验有效性的生命线）：
  - [ ] 除「数据组成」外，所有超参完全相同（用同一份模板生成配置）
  - [ ] 明确并记录「固定 epoch」还是「固定总迭代数」——两者对「合成数据越多、看到的总图数越多」这件事的处理不同，**必须在 `docs/RESULTS.md` 里写明选了哪种**
  - [ ] 身份数、评测集、评测代码完全一致
  - [ ] 记录每个组合的实际训练图片数（存在 `metrics.json` 的 `train_set` 里）
- [ ] 汇总 `scripts/summarize.py`：把所有 `metrics.json` 聚合成 `results/summary.csv`（每行一个 run，每列一个指标）
- [ ] 统计汇总：同一配置跨 seed 求 **均值 ± 标准差**（脚本自动算，不要手算）
- [ ] 画图（存 `reports/figures/`）：
  - [ ] **比例 vs 指标曲线**（带误差带）——项目的核心图
  - [ ] 各实验组横向对比柱状图（E1/E2/E3/E4）
  - [ ] 分桶/公平性对比图
- [ ] 写 `docs/RESULTS.md` 的结果表（每个 RQ 一行结论）
- [ ] 记录每次训练耗时与显存峰值，汇总成 `reports/compute_budget.md`（回答 RQ5：最低可行配方）
- [ ] 抽查验证：随机挑 1 个 run，从配置开始完整重跑一次，确认 `metrics.json` 可复现

## 验收标准（Definition of Done）

- [ ] 最小可交付矩阵（≥3 个比例 × ≥1 seed）全部完成，每个组合都有符合契约的 `metrics.json`
- [ ] `python scripts/run_matrix.py --config configs/exp/matrix.yaml` 中断后再执行**不会重复跑已完成项**
- [ ] `results/summary.csv` 由脚本生成，包含均值与标准差列
- [ ] 核心曲线图已落盘，横轴是合成占比，纵轴是各 benchmark 指标，带误差带
- [ ] 每个数字都能追溯到 `results/runs/<exp_id>/`，且该目录里有配置副本
- [ ] `reports/compute_budget.md` 有「训练一次要多久 / 显存峰值多少」的实测数据
- [ ] **变量控制检查表**已填写（贴在本 Issue 评论里）：逐项确认除数据组成外无其他差异

## 交付物

- `configs/exp/matrix.yaml`、`src/aigcfr/utils/matrix.py`
- `scripts/run_matrix.py`、`scripts/summarize.py`
- `results/runs/e5-*/metrics.json`（多个）、`results/summary.csv`
- `reports/figures/mix_ratio_curve.png`、`reports/compute_budget.md`
- `docs/RESULTS.md` 结果表

## 依赖

- 需要 {{#06-arcface-training}}、{{#07-evaluator}}（训练与评测入口）
- 需要 {{#08-classic-augmentation}}、{{#10-identity-preserving-generation}}、{{#11-generation-filtering}}（数据来源）
- 被依赖：{{#15-results-conclusion}}、{{#17-site-content}}

## 预估工作量

工程 1 天 + **机器时间约 1~3 天**（可以整晚/整天挂着跑）

## 参考

- **FRCSyn Challenge** 的方法学最值得抄：它要求参赛者对**同一个系统训练两次**（一次只用授权的真实数据、一次只用合成数据）来做配对比较——这正是本任务「控制变量」的极端版本：[CVPRW 2024](https://openaccess.thecvf.com/content/CVPR2024W/FRCSyn/html/Deandres-Tame_Second_Edition_FRCSyn_Challenge_at_CVPR_2024_Face_Recognition_Challenge_CVPRW_2024_paper.html) · [arXiv:2404.10378](http://arxiv.org/abs/2404.10378)
- 合成数据识别竞赛（**SDFR**，FG 2024）：含 7 个基准的评测协议与公平性评估：[arXiv:2404.04580](https://arxiv.org/abs/2404.04580)
- 混合/预训练策略的经典结果：DigiFace-1M 报告「先在合成数据上预训练，再在少量真实数据上微调，可让真实数据需求下降约 75%」：[arXiv:2210.02579](https://arxiv.org/abs/2210.02579)
- 「Real Gap」的量化方式参考 VariFace：[arXiv:2412.06235](https://arxiv.org/abs/2412.06235)
- 综述（用来对照自己的实验设置是否主流）：*Synthetic data for face recognition: Current state and future prospects*，[Image and Vision Computing 2023](https://www.sciencedirect.com/science/article/abs/pii/S0262885623000628)

## 附：已核实要点（框架预置，2026-09）——**这三条决定实验是否成立**

1. **必须明确「预算口径」**：加合成数据有两种公平比较方式，**它们回答的是不同问题，结论可能相反**——
   - **等总图片数**（real + synth 总数 = 纯 real 的总数）→ 回答「合成数据能否**替代**真实数据」
   - **等真实图片数**（在 real 之上**追加** synth）→ 回答「追加合成数据能否带来**增益**」

   **最好两种都报**，至少要在配置与文档里写明用的是哪种。这是本任务最容易被审稿人/老师抓住的点。
2. **⚠️ 只跑 LFW 会掩盖问题**：合成数据实验对「每身份图片数」和「身份总数」极其敏感——已有工作报告在身份数从 2 万增到 5 万时，**IJB-C 从 75.80 掉到 37.17，而 LFW 几乎不动**。所以务必加一个**不易饱和**的指标（TinyFace 1:N Rank-1，或你自己的 1:N 闭集测试）。
3. **FRCSyn 的配对协议值得直接照搬**：要求用**同一套系统训练两次**（一次只用授权的真实数据、一次只用合成数据）来做配对比较——这是「控制变量」的极端版本，也是本领域公认的做法。

另外：**≥3 个随机种子、报均值±标准差**；LFW 上 0.1% 的差异就是噪声，不要拿它下结论。

详细版见 [`docs/REFERENCES.md`](../REFERENCES.md) 第 7 节。

## 附：两人两机的执行约定（**跑批阶段的头号风险**）

- **按"整组实验"分配机器，绝不按 seed 拆分。** 例如「E4 × 5 个比例 × 3 seed = 15 个 run」整组交给同一台机器。理由：batch/精度/硬件不同会让结果出现 0.1%~0.5% 的系统性偏移，足以把你的曲线结论搞反（详见 [`WORKFLOW.md`](../WORKFLOW.md) §3.4）。
- **`summary.csv` 必须带 `profile` 列**（来自 `metrics.json` 的 `hardware` 块）。若同一张图里混了不同档位的结果，图表要能区分（不同标记/分面）。
- **分工建议**：强机一方负责 `run_matrix.py` 的执行与监控；弱机一方负责 `summarize.py`、图表与 `docs/RESULTS.md` 初稿。两人**不要同时改 `matrix.yaml`**（约定一人维护）。
- **跑批期间另一台机器不要闲着**：弱机可以并行做 {{#12-synth-detection-report}}、{{#16-site-skeleton}}，或推进 [`LEARNING.md`](../LEARNING.md) L6 的论文精读。
- **每个 run 落盘时必须包含**：`config.resolved.yaml`（三层合并后的最终配置）、`metrics.json`（含 hardware 块）、日志。缺任何一项，另一台机器就无法判断这个结果能不能用。

## 新手提示 / 卡住了怎么办

- **先把批量脚本调通再放量**：用 1 个 epoch、2 个身份把 15 个组合全"跑"一遍（每个几秒），确认目录、命名、跳过逻辑都对，再改成真实配置。
- 训练时间长，务必：记录开始时间、每完成一个组合就写一行日志、让机器别休眠（Windows 电源设置改成「从不睡眠」）。
- 如果发现「比例越高越好」或「越高越差」的**单调趋势**很漂亮，先怀疑是不是变量没控制住（例如 epoch 固定导致高比例组看到了更多数据）。**这是最常见的实验设计错误**，务必自查。
- **降级方案**：时间不够就砍到 3 个比例 × 1 seed，并在结论里注明「单 seed，未估计方差」；宁可少跑也不要伪造或外推数据。

