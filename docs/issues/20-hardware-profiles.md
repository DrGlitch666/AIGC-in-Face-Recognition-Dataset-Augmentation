---
title: "[环境] 硬件档位判定、配置分层与自适应降级（A/B/C profile）"
slug: "20-hardware-profiles"
labels: ["优先级:P0", "类型:环境", "难度:中等"]
milestone: "M0 脚手架与地基"
---

## 背景 / 为什么做这个

**两人两台机器，配置不同**（显存大小、CUDA 版本、有没有独显、操作系统可能都不同）。

如果代码里写死 `batch_size=128`、`device='cuda'`、`resize=512`，那么：
- 弱机器跑不动 → 只能改代码 → 改完就和强机器的版本**不是同一份代码**了；
- 强机器为了"迁就"弱机器而用小 batch → 自己机器白买了；
- 两人各跑一半实验 → 因为 batch 不同，结果不可比（小数据集上 batch 会真实影响指标）。

**正确做法：按"能力档位"配置，不按机器型号写代码。** 本任务就是把这件事落到工程上。

> 档位定义（A 大显存 / B 小显存独显 / C 无独显或 CPU）见 [`FRAMEWORK.md`](../FRAMEWORK.md) §4.1 与 §4.3。

## 目标

- 每台机器一句命令判定自己的档位；
- 同一份代码 + 同一份"科学设定"，在两台机器上各自以合适的资源参数运行；
- 显存不足时**自动降级**而不是崩溃；
- 每个结果文件里都能查到"这是在什么硬件上跑的"。

## 任务清单

- [ ] **档位判定**（与 {{#01-env-setup}} 合并产出）：`scripts/check_env.py` 除版本信息外，还要输出：
  - [ ] 是否有可用 CUDA、GPU 名称与显存总量
  - [ ] 判定结果 `profile: a|b|cpu`（判定规则写在配置里，不写死在代码里）
  - [ ] 写入 `reports/env_report.md`，供别人查看
- [ ] **配置三层骨架**：
  - [ ] `configs/base.yaml`：**科学设定**——数据来源、manifest 路径（相对）、模型结构、loss、epoch、评测集、seed
  - [ ] `configs/profiles/a.yaml` / `b.yaml` / `cpu.yaml`：**资源设定**——batch_size、grad_accum、image_size、amp、num_workers、offload、生成分辨率
  - [ ] `configs/local.template.yaml`：**机器私有**模板（路径、缓存目录、线程数、显存上限），实际文件命名为 `configs/local.<machine>.yaml`
  - [ ] `.gitignore` 加入 `configs/local.*.yaml`（保留 `local.template.yaml`）
- [ ] **配置加载器** `src/aigcfr/utils/config.py`：
  - [ ] 支持 `--config`、`--profile {auto,a,b,cpu}`、`--local <path>`
  - [ ] 三层**深度合并**（后者覆盖前者），并打印合并来源
  - [ ] **保护规则**：若 `local.*.yaml` 试图覆盖 `base.yaml` 的**科学设定**键（`train.*` / `model.*` / `eval.*` / `seed`），直接**报错退出**并提示原因
  - [ ] 输出 `results/runs/<exp_id>/config.resolved.yaml`（合并后的最终配置）+ `config_hash`
- [ ] **硬件上下文记录**：所有落盘的 `metrics.json` 增加 `hardware` 块
      （`profile`、`gpu_name`、`vram_total`、`batch_size`、`grad_accum`、`amp`、`torch_version`、`platform`）
- [ ] **自适应降级**（生成与训练各自实现）：
  - [ ] 捕获 OOM → 依次降级：`batch 减半 → 开/加大梯度累积 → 降分辨率 → 开 CPU offload → 明确报错并给出建议`
  - [ ] 每次降级都**打印日志并写进 run 目录**（不能静默改变实验条件）
  - [ ] 降级后如果 batch 与对比实验不同，**在 `metrics.json` 里标注 `degraded: true`**
- [ ] **测试** `tests/test_config.py`：
  - [ ] 合并优先级正确（local > profile > base）
  - [ ] 科学设定被 local 覆盖时抛错
  - [ ] `--profile auto` 在无 GPU 环境下落到 `cpu.yaml`
- [ ] 写 `docs/CONFIG.md`（或把最终规则补进 `FRAMEWORK.md` §4.3）：一张图说明三层关系 + 一个"我要改 X 该改哪个文件"的对照表

## 验收标准（Definition of Done）

- [ ] 两台机器各跑一次 `python scripts/check_env.py`，都能得到明确的 `profile` 判定，并写入各自的 `reports/env_report.md`
- [ ] **同一条命令在两台机器上都能跑通**（无需改代码）：
      `python scripts/train.py --config configs/exp/e1-real-only-seed0.yaml --profile auto`
- [ ] `configs/local.*.yaml` 不在 `git status` 里出现；`local.template.yaml` 在
- [ ] 故意在 `local.*.yaml` 里写 `train.batch_size`，运行时应**报错**而不是静默生效
- [ ] 人为把 batch 调大到 OOM，观察**自动降级**日志，且 `metrics.json` 里 `degraded: true`
- [ ] `metrics.json` 的 `hardware` 块字段齐全（缺字段的 run 不能参与跨机比较）
- [ ] `pytest tests/test_config.py` 全绿

## 交付物

- `configs/base.yaml`、`configs/profiles/{a,b,cpu}.yaml`、`configs/local.template.yaml`
- `src/aigcfr/utils/config.py`（三层合并 + 保护规则）、`src/aigcfr/utils/hardware.py`（档位判定与降级）
- `scripts/check_env.py`（档位判定部分，与 {{#01-env-setup}} 合并）
- `docs/CONFIG.md`
- `tests/test_config.py`
- 两台机器的 `reports/env_report.md`

## 依赖

- 需要 {{#02-repo-skeleton}}（配置系统与目录约定），可与之合并推进
- 与 {{#01-env-setup}}（提供硬件信息与档位判定）、{{#19-collaboration-protocol}}（谁用哪个 profile）并行
- 被依赖：{{#03-data-pipeline}}、{{#05-zeroshot-baseline}}、{{#06-arcface-training}}、{{#09-unconditional-generation}}、{{#10-identity-preserving-generation}}、{{#14-mix-ratio-matrix}}

## 预估工作量

半天到 1 天

## 参考

- [`FRAMEWORK.md`](../FRAMEWORK.md) §4.1（硬件分层）、§4.2（环境泛化规则）、§4.3（配置分层）
- [`WORKFLOW.md`](../WORKFLOW.md) §2（命令约定）、§3.4（跨机器可比性）
- [PyYAML 文档](https://pyyaml.org/wiki/PyYAMLDocumentation)（自己写 20 行深度合并即可，**不必引入 Hydra**）
- 若确实想要成熟方案：[Hydra / OmegaConf](https://hydra.cc/docs/intro/)（支持配置组合与覆盖，但学习成本更高；两人项目通常不需要）

## 新手提示 / 卡住了怎么办

- **"保护规则"是这个任务最有价值的部分**：它防止某个人无意中用自己的本地覆盖改了实验条件，导致两人结果对不上。
- 降级逻辑要**先探测、再执行**：可以用 `torch.cuda.mem_get_info()` 提前判断，比"试了再说"更快。
- 不要试图让两台机器**跑出完全一样的数字**（浮点非确定性 + 不同硬件本来就做不到）。目标是：**同机同配置可复现，跨机结果可比且可解释**。
- **降级方案**：如果时间紧，最低限度只做两件事——① `base.yaml` + `profiles/` 分层（不需要 local 层）② `metrics.json` 记录硬件与 batch。这两件就能支撑跨机器协作；自动降级可以后补。
