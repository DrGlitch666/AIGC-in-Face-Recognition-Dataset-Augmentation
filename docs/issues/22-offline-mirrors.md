---
title: "[环境] 受限网络下的资源获取：连通性预检、镜像路线与离线缓存"
slug: "22-offline-mirrors"
labels: ["优先级:P0", "类型:环境", "难度:入门", "good first issue"]
milestone: "M0 脚手架与地基"
---

## 背景 / 为什么做这个

**两台机器访问境外站点都不稳定**（HuggingFace / Google Drive / GitHub Releases 时通时不通）。这不是一个小麻烦，它会直接影响：

- 能不能下载**现成合成数据集**（B 轨，本项目的保底路线）；
- 能不能拿到**身份条件生成的模型权重**（A 轨，本项目的核心亮点）；
- 能不能拿到**识别模型**（连基线都跑不起来）；
- 甚至能不能装对 **PyTorch 的 cu128 轮子**（RTX 50 系必须）。

如果不在第一天把这件事摸清楚，最常见的结局是：**前三周都耗在"下载失败—换地址—再失败"上**。

## 目标

- 第一天就知道**每类资源该走哪条路线**；
- 有一条**连不上时立刻降级**的止损路径；
- 大文件只在本地缓存，**不进 Git、不互相传**。

## 任务清单

- [ ] 用一个**一次性探测命令**（脚本放项目之外，如 `F:\aigcfr\_cache\tools`，**不进仓库**）对 [`docs/MIRRORS.md`](../MIRRORS.md) §3 列出的端点做**超时 5 秒**的探测
- [ ] **两台机器各跑一次**，把结果（一张表 + 一句结论）回填到 `docs/MIRRORS.md` §4，并贴到本 Issue 评论
- [ ] 按 [`docs/MIRRORS.md`](../MIRRORS.md) §2 配置三件事：
  - [ ] pip 全局源（清华 tuna）
  - [ ] conda `.condarc`（清华）
  - [ ] `HF_ENDPOINT=https://hf-mirror.com`（**用户级环境变量**，重开终端后验证生效）
- [ ] **验证 `HF_ENDPOINT` 真的生效**：用 `huggingface_hub` 下载一个几 MB 的小模型/配置，确认走的是镜像
- [ ] 确认 PyTorch **cu128** 轮子能装（这一步和 {{#01-env-setup}} 合并；镜像上没有就试官方索引）
- [ ] 确认 insightface 模型包（`buffalo_l`，275 MB）能拿到（GitHub 加速前缀 / HF 镜像 / 同学拷贝，**三条路任选一条走通即可**）
- [ ] 建立统一缓存目录约定（如 `D:\aigcfr-cache\{hf,models,datasets,weights}`），写进各自的 `configs/local.<machine>.yaml`
- [ ] 回填 [`docs/MIRRORS.md`](../docs/MIRRORS.md) §4 的「实测结果」表（每条路线一行，成功或失败都记）
- [ ] **确定 B 轨数据集**：从 Zenodo（优先）/ hf-mirror 里选定 1~2 个**实际可下载**的合成数据集，记录大小与许可证到 {{#18-ethics-and-release}} 的许可表
- [ ] **确定 A 轨模型来源**：ModelScope（优先）或 hf-mirror，写明模型名与版本
- [ ] 写一句**止损决策**：若某类资源三条路线都不通，改用什么方案（见 [`docs/MIRRORS.md`](../docs/MIRRORS.md) §5）

## 验收标准（Definition of Done）

- [ ] `docs/MIRRORS.md` §4 实测表已回填 + 两台机器各有一条 Issue 评论，结论明确可执行
- [ ] `HF_ENDPOINT` 生效已验证（有证据：下载日志或缓存目录里出现了文件）
- [ ] PyTorch 装好且 `torch.cuda.get_arch_list()` 含 `sm_120`（你这台机器）
- [ ] `buffalo_l` 已下载并能跑一次人脸比对（≥1 对图片）
- [ ] **B 轨至少选定 1 个能下载的合成数据集**，并写出预估大小与许可
- [ ] `docs/MIRRORS.md` §4 实测表已回填，不留「待测」
- [ ] 缓存目录约定已写进两份 `configs/local.*.yaml`

## 交付物

- `docs/MIRRORS.md` §4 的实测结果表（回填）
- Issue 评论：两台机器各一份连通性结论
- 回填后的 `docs/MIRRORS.md` §4
- 各自的 `configs/local.<machine>.yaml` 中的缓存路径配置

## 依赖

- 与 {{#01-env-setup}}（环境安装）、{{#20-hardware-profiles}}（机器私有配置）**合并推进**，三者是同一批工作
- 直接决定 {{#03-data-pipeline}}（数据来源）与 {{#10-identity-preserving-generation}}（模型来源）的选型
- 被依赖：{{#06-arcface-training}}（要下预训练模型）、{{#09-unconditional-generation}}、{{#10-identity-preserving-generation}}

## 预估工作量

**半天**（其中大部分是等待下载，可以并行做别的事）

## 参考

- [`docs/MIRRORS.md`](../MIRRORS.md)——**本任务的主文档**（候选路线、配置命令、预检端点、止损路线）
- [hf-mirror 站点](https://hf-mirror.com)（HF 镜像，配合 `HF_ENDPOINT` 使用）
- [ModelScope 魔搭](https://www.modelscope.cn)（国内模型社区）
- [Zenodo](https://zenodo.org)（开源数据集托管，合成人脸数据集多在此）
- [清华 PyPI 镜像帮助](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)
- [清华 Anaconda 镜像帮助](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)

## 新手提示 / 卡住了怎么办

- **不要在一个地址上反复重试超过 3 次**——立刻换下一条候选路线，并把失败记进实测表。
- `HF_ENDPOINT` 是最有价值的一条：设好之后 `diffusers` 的 `from_pretrained("...")` **一行代码都不用改**就走镜像。
- 大文件下载**放在离开电脑前启动**（`hf download` / `modelscope download` 支持断点续传）。
- **止损原则**：一个 275 MB 的模型包不值得花两天——找同学拷、用移动硬盘、或直接换替代方案。
- ⚠️ **不要用来路不明的"整合包/破解版"**：模型与数据的许可正是本项目要讨论的内容，用它就等于自己打自己的脸。
- **降级方案**：三条路都不通时，走 [`docs/MIRRORS.md`](../MIRRORS.md) §5——只用 LFW + 传统增强先把全流程跑通，把"未能使用 X"如实写进局限。
