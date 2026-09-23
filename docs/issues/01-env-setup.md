---
title: "[环境] 跨机器可复现环境 + 硬件档位自检（每台机器各做一次）"
slug: "01-env-setup"
labels: ["优先级:P0", "类型:环境", "难度:入门", "good first issue"]
milestone: "M0 脚手架与地基"
---

## 背景 / 为什么做这个

这是整个项目的第一个卡点，也是最容易浪费一天的地方。**而且你们是两台配置不同的机器**，所以这里的目标不是"把我的机器装好"，而是：

> **两台机器各自装好，并且能证明"我们跑的是同一套代码、同一套依赖约定"。**

已知的常见陷阱（不同机器踩的坑不一样，按自己情况对号入座）：

| 陷阱 | 谁会踩 | 表现 |
|---|---|---|
| Python 版本太新 | 系统自带新版 Python 的人 | `pip install torch` 找不到 wheel / 触发本地编译失败 |
| 显卡架构太新 | RTX 50 系（Blackwell, `sm_120`） | `CUDA error: no kernel image is available` |
| 显卡较老 / CUDA 版本旧 | 老卡机器 | 装了新轮子反而跑不了，需要匹配自己驱动的 CUDA 版本 |
| 没有独显 / macOS | 用 CPU 或 Apple Silicon 的人 | 需要 CPU 轮子、并按平台选择 ONNX Runtime 的 EP |
| 装成 CPU 版 torch | 照抄别人命令的人 | `torch.cuda.is_available()` 为 `False`，训练慢几十倍 |

所以本任务的产出是**每台机器一条环境自检记录 + 档位判定**（贴在本 Issue 评论里，**不落仓库文件**），让后面的所有任务都能"按档位运行"（见 {{#20-hardware-profiles}}）。

## 目标

- 每台机器都有一个可复现、可销毁重建的 Python 环境。
- 一条（一次性）命令输出「这台机器能跑什么」的明确结论，并给出**档位判定（A / B / C）**，结果贴到本 Issue 评论。

## 任务清单

- [ ] 安装 Miniconda（或 Miniforge/mamba）；**不要**直接用系统自带 Python
- [ ] 创建环境：`conda create -n aigcfr python=3.11 -y`
      ⚠️ **两人必须约定同一个次版本**（都 3.11 或都 3.12），否则依赖解析结果会不同
- [ ] 安装 PyTorch——**按自己机器的 CUDA 版本选轮子**（不是所有人都一样）：
  - [ ] 有 NVIDIA 显卡：按驱动支持的最高 CUDA 版本选（新卡 RTX 50 系需 **cu128 且 PyTorch ≥ 2.7**）
  - [ ] 无独显 / macOS：装 CPU 轮子（或用 Apple Silicon 的默认轮子）
  - [ ] 装完确认 `torch.cuda.is_available()`；有卡的话再确认 `torch.cuda.get_arch_list()` 包含你显卡的架构（如 `sm_120` / `sm_89` / `sm_86`）
- [ ] 安装其余依赖：`insightface`、`onnxruntime-gpu`（或按平台用 `onnxruntime` / CoreML EP）、`opencv-python`、`numpy`、`pillow`、`tqdm`、`pyyaml`、`pytest`、`matplotlib`
      ⚠️ **不要同时装 `onnxruntime` 和 `onnxruntime-gpu`**（会冲突，且常常静默退回 CPU）
- [ ] 做一次**环境自检**（一次性命令即可，**不写成仓库里的脚本**）：
  - [ ] 版本信息：Python / torch / torchvision / CUDA / cuDNN / onnxruntime / 关键库
  - [ ] 硬件信息：OS 与平台、GPU 名称与**显存总量**、CUDA 可用性、ONNX Runtime 实际 providers
  - [ ] **一次真实的张量运算**（如 1024×1024 矩阵乘）确认真的能算，而不是只报告"可用"
  - [ ] **档位判定**：直接调用 `src/aigcfr/utils/config.py` 的 `detect_vram_gb()` / `decide_profile()`
- [ ] 写 `docs/SETUP.md`：从零到能跑的完整步骤，**并标注哪些步骤是"每台机器不同"的**
- [ ] 冻结依赖：导出 `environment.yml`（**通用层，入库**）；如有机器私有依赖，写进 `environment.local.yml`（**不入库**）
- [ ] 把自检输出 + 档位判定**贴到本 Issue 评论**（两台机器各一条）
- [ ] ⚠️ **关于过程产物**：配环境用的自检脚本与报告属于**一次性脚手架，不进仓库**（仓库只放项目本身）；需要时放到项目之外，例如 `F:\aigcfr\_cache\tools`

## 验收标准（Definition of Done）

- [ ] 自检输出里能看到 GPU 名称（或明确写出"无可用 GPU，判定为 cpu 档"），且**已贴到本 Issue 评论**
- [ ] **两台机器各有一条**自检评论，每条包含：版本表、硬件表、**档位判定**、最终结论（可用 / 降级可用 / 不可用）
- [ ] `docs/SETUP.md` 里写清了每个包**从哪个源**装的，并明确标出"因机器而异"的步骤
- [ ] `environment.yml` 入库且能被另一台机器成功创建环境（**通用层不依赖任何一台机器的私有配置**）
- [ ] 两台机器都能跑通同一条自检命令（不需要改代码）
- [ ] 两人互相确认：各自档位是什么、后面谁负责哪部分（衔接 {{#19-collaboration-protocol}}）

## 交付物

- Issue 评论：两台机器各一条环境自检记录 + 档位判定（**不新增仓库文件**）
- `docs/SETUP.md`
- `environment.yml`（+ 私有的 `environment.local.yml`，不入库）

## 依赖

- 无（这是项目的起点，其他所有任务的先决条件）
- 与 {{#20-hardware-profiles}}（配置分层）、{{#19-collaboration-protocol}}（分工）**并行推进**，三者共同构成 M0 的地基
- 被依赖：全部任务

## 预估工作量

每台机器 2~4 小时（大部分时间在下载）

## 参考

- [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)（**务必按自己机器的 CUDA 版本选命令**）
- [insightface 官方仓库](https://github.com/deepinsight/insightface)（v2.0 起普通安装不需要 C++ 编译器）
- [ONNX Runtime Execution Providers](https://onnxruntime.ai/docs/execution-providers/)（CUDA / CoreML / CPU 按平台选）
- [Conda 环境导出文档](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#exporting-the-environment-yml-file)
- 架构代号速查：`sm_120` = Blackwell（RTX 50 系）；`sm_89` = Ada（RTX 40 系）；`sm_86` = Ampere（RTX 30 系）

## 新手提示 / 卡住了怎么办

- 如果 `torch.cuda.is_available()` 是 `False`：先看驱动版本，再看是不是装成了 CPU 版（`pip list` 里 `torch` 带 `+cpu` 后缀就是装错了）。
- 如果报 `no kernel image is available`：装错了轮子的 CUDA 版本，按**自己显卡的架构**重装（新卡要 cu128+）。
- **注意力实现统一用 PyTorch 自带的 SDPA**：Windows 上没有 FlashAttention，xformers 对 Blackwell 的支持也很晚才到，别按旧教程去装。
- **降级方案**：实在装不上 GPU 版，就先装 CPU 版把整条代码链路跑通（生成和训练会慢几十倍，只用于验证代码正确性），把环境问题单独记成 Issue 评论，**不要为此卡住整个项目**——C 档机器照样能承担数据、评测、筛选、网站与文档（见 [`FRAMEWORK.md`](../FRAMEWORK.md) §11.1）。
- ⚠️ **不要把 `environment.local.yml` 或任何含绝对路径的文件提交上去**——那是另一台机器跑不起来的最常见原因。
