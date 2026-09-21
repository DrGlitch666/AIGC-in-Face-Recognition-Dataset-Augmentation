---
title: "[环境] 搭建 Python 3.11 + PyTorch(CUDA) 环境，产出 GPU 自检报告"
slug: "01-env-setup"
labels: ["优先级:P0", "类型:环境", "难度:入门", "good first issue"]
milestone: "M0 脚手架与地基"
---

## 背景 / 为什么做这个

这是整个项目的第一个卡点，也是最容易浪费一天的地方。两个已知的硬件/版本陷阱：

1. **本机 GPU 是 NVIDIA RTX 5060 Laptop（Blackwell 架构，compute capability `sm_120`）**。用旧 CUDA 版本编译的 PyTorch 会直接报 `CUDA error: no kernel image is available for execution on the device`。必须使用支持 `sm_120` 的构建（CUDA 12.8 / cu128 及以上的 wheel）。
2. **系统默认 Python 是 3.14**，很多深度学习库在这个版本上还没有预编译 wheel，`pip install torch` 可能直接失败或触发本地编译。

所以这一步的产出不是「装好了」，而是**一份别人（和三个月后的你）能照着复现的环境说明 + 一份机器实测报告**。

## 目标

- 一个可复现、可销毁重建的 Python 环境。
- 一条命令 `python scripts/check_env.py` 输出「这台机器能不能跑本项目」的明确结论。

## 任务清单

- [ ] 安装 Miniconda（或 Miniforge/mamba），**不要**用系统 Python 3.14 直接装
- [ ] 创建环境：`conda create -n aigcfr python=3.11 -y`
- [ ] 安装 PyTorch（**优先 cu128 或更新**的官方轮子），并验证 `torch.cuda.is_available()`
- [ ] 验证架构支持：`sm_120` 是否出现在 `torch.cuda.get_arch_list()`
- [ ] 安装其余依赖：`insightface`、`onnxruntime-gpu`、`opencv-python`、`numpy`、`pillow`、`tqdm`、`pyyaml`、`pytest`、`matplotlib`
- [ ] 编写 `scripts/check_env.py`：打印 Python / torch / torchvision / CUDA / cuDNN 版本、GPU 名称与显存、`onnxruntime` 可用 providers、关键库版本
- [ ] 在脚本里做一次**真实的张量运算**（如 1024×1024 矩阵乘）确认 GPU 真的能算，而不是只报告「可用」
- [ ] 生成 `reports/env_report.md`：版本表 + 可用性判定 + 已知问题
- [ ] 写 `docs/SETUP.md`：从零到能跑的完整步骤（含踩坑记录）
- [ ] 冻结依赖：导出 `environment.yml` 或 `requirements.txt`

## 验收标准（Definition of Done）

- [ ] `python scripts/check_env.py` 退出码为 0，输出里能看到 GPU 名称
- [ ] `reports/env_report.md` 存在，且包含：版本表、GPU 名称与显存、`sm_120` 是否被支持、最终判定（可用 / 降级可用 / 不可用）
- [ ] `docs/SETUP.md` 里写清了每个包**从哪个源**装的（普通 PyPI 还是 PyTorch 专用索引）
- [ ] **重建验证**：删掉环境重新按 `docs/SETUP.md` 装一遍能成功（这一步能提前发现你漏记的步骤）

## 交付物

- `scripts/check_env.py`
- `reports/env_report.md`
- `docs/SETUP.md`
- `environment.yml` / `requirements.txt`

## 依赖

- 无（这是项目的起点，其他所有任务的先决条件）
- 被依赖：{{#02-repo-skeleton}}、{{#03-data-pipeline}}、后续全部任务

## 预估工作量

2~4 小时（大部分时间在下载）

## 参考

- [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)（务必选 CUDA 12.8+ 对应的命令）
- [insightface 官方仓库](https://github.com/deepinsight/insightface)
- [ONNX Runtime CUDA Execution Provider 文档](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- 架构代号速查：`sm_120` = Blackwell（RTX 50 系）；`sm_89` = Ada（RTX 40 系）

## 附：已核实要点（框架预置，2026-09）

- **PyTorch 版本下限**：RTX 50 系（sm_120）需要 **PyTorch ≥ 2.7.0** 的 **cu128** 轮子；装更低版本会直接报 `no kernel image is available`。对应索引：`https://download.pytorch.org/whl/cu128`
- **不要按旧教程装 xformers**：xformers 直到 0.0.33 才支持 Blackwell；Windows 上没有 FlashAttention。**统一用 PyTorch 自带的 SDPA 注意力**即可。
- **insightface 安装**：v2.0 起 face3d 扩展默认不编译，**普通安装不需要 C++ 编译器**（老教程里「Windows 装 insightface 要 VS build tools」的说法已过时）。
- **ONNX Runtime**：用 GPU 就装 `onnxruntime-gpu`，并且**不要和 `onnxruntime` 同时装**（会冲突，且常常静默退回 CPU）。
- 详细清单见 [`docs/REFERENCES.md`](../REFERENCES.md) 第 1~2 节。

## 新手提示 / 卡住了怎么办

- 如果 `torch.cuda.is_available()` 是 `False`：先看显卡驱动版本，再看装的是不是 CPU 版 torch（`pip list` 里 `torch` 带 `+cpu` 后缀就是装错了）。
- 如果报 `no kernel image is available`：说明装的不是 `sm_120` 支持的构建，换 cu128 及以上重装。
- **降级方案**：实在装不上 GPU 版，就先装 CPU 版把整条代码链路跑通（生成和训练会慢几十倍，只用于验证代码正确性），把环境问题单独记成 Issue 评论，不要为此卡住整个项目。
