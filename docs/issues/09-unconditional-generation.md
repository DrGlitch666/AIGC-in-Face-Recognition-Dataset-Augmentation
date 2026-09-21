---
title: "[生成] 生成管线 A：无条件生成，打通「生成→入库→训练→评测」闭环（E3）"
slug: "09-unconditional-generation"
labels: ["优先级:P1", "类型:生成", "难度:中等"]
milestone: "M2 生成与筛选"
---

## 背景 / 为什么做这个

在碰「身份保持」这种难度更高的东西之前，先用**最简单的生成方式**（文生图，随机人脸，不指定是谁）把整条流水线打通：

```
生成脚本 → 图片落盘 → 写 manifest → 训练 → 评测 → 结果 JSON → 图表
```

这样做的价值有两个：

1. **技术上**：把生成器接口、显存管理、manifest 登记、批量落盘这些容易出问题的环节先跑通，后面换更强的生成器只需要替换一个类。
2. **科学上**：E3 本身就是一个重要的对照组——**「不控制身份的合成人脸」到底有没有用？** 它可能就是一堆和任何真实身份都对不上的脸，只在起正则化作用。这个结果本身就有意思。

## 目标

一条命令生成 ≥200 张 512×512 的人脸图，完整登记进 manifest，并跑完 E3 实验。

## 任务清单

- [ ] 定义并实现**统一的生成器接口** `FaceGenerator`（`docs/FRAMEWORK.md` §6），所有生成器都必须实现它：
      `generate(identity_refs, n, cfg) -> list[GeneratedImage]`
- [ ] 选一个基础模型（建议 **SD1.5** 或 **SDXL-Turbo**，8 GB 显存友好），用 `diffusers` 加载
- [ ] 写 prompt 模板与负向 prompt（人脸质量、写实风格相关），配置化，不写死在代码里
- [ ] 写 `scripts/generate.py`：读 `configs/gen/*.yaml` → 批量生成 → 断点续跑（已存在的图跳过）→ 写 manifest
- [ ] 显存管理：fp16、`enable_model_cpu_offload()`、必要时 `enable_attention_slicing()`；记录显存峰值
- [ ] manifest 登记规则：
  - [ ] `source=synth`、`generator=<模型标识>`、`parent_image_id=null`（无条件生成没有参考图）
  - [ ] `identity_id`：无条件生成没有真实身份，**每张图分配一个独立伪身份**（如 `syn_0001`），并在报告里说明这个设定及其影响
- [ ] 生成 ≥200 张，人工抽检 20 张（是否是人脸、是否畸形、多样性如何）
- [ ] 产出图库网格 `reports/figures/e3_uncond_samples.png`
- [ ] 用 {{#06-arcface-training}} 训练 E3 并评测（注意：伪身份会造成大量「单样本身份」，训练配置里要说明如何处理，例如作为额外类别或按比例混合）
- [ ] `results/runs/e3-uncond-seed*/metrics.json` 落盘并汇总

## 验收标准（Definition of Done）

- [ ] `python scripts/generate.py --config configs/gen/sd15_uncond.yaml` 一条命令完成，中断后重跑不重复生成
- [ ] 生成图片数量与 manifest 行数一致，**每张图都有 `generator` 与 `seed` 记录**（不记 seed 就无法复现）
- [ ] 显存峰值 < 8 GB，生成速度已实测并记录（张/分钟）
- [ ] 人工抽检结论写在报告里，畸形/失败样本单独存一份（不要偷偷删掉）
- [ ] E3 的 `metrics.json` 符合契约并进入 `results/summary.csv`
- [ ] `docs/RESULTS.md` 增加一行 E3 结果与一句话结论

## 交付物

- `src/aigcfr/generate/base.py`（接口）、`src/aigcfr/generate/diffusers_uncond.py`
- `scripts/generate.py`
- `configs/gen/sd15_uncond.yaml`
- `data/manifests/synth_e3.jsonl`、`results/runs/e3-uncond-seed*/metrics.json`
- `reports/figures/e3_uncond_samples.png`、`reports/generation_notes.md`

## 依赖

- 需要 {{#02-repo-skeleton}}（接口约定与配置系统）
- 需要 {{#01-env-setup}}（PyTorch + CUDA）
- 需要 {{#06-arcface-training}}、{{#07-evaluator}}（闭环的后半段）
- 被依赖：{{#10-identity-preserving-generation}}（复用同一个接口与脚本）

## 预估工作量

1~1.5 天（首次下载模型权重较慢）

## 参考

- [Diffusers 官方文档](https://huggingface.co/docs/diffusers)（`StableDiffusionPipeline` 用法、显存优化那一节必读）
- [SDXL-Turbo 模型卡](https://huggingface.co/stabilityai/sdxl-turbo)（少步数生成，速度快，适合 8 GB）
- 注意各基础模型的**许可证**（例如部分模型对商用有限制），把结论记进 {{#18-ethics-and-release}} 的 `docs/ETHICS.md`

## 附：已核实要点（框架预置，2026-09）

- **基础模型选择**：SD1.5 / SDXL-Turbo 在 8 GB 上都可跑，但**必须开 fp16 + batch=1**，SDXL 级别还需 `enable_model_cpu_offload()`。
- **不要装 xformers**：Windows 上没有 FlashAttention，xformers 对 Blackwell（sm_120）的支持要到 0.0.33；**用 PyTorch 自带的 SDPA**。
- **许可要记一笔**：不同基础模型的许可差异很大（部分限制商用）。本项目是非商业研究，但仍要在 `docs/ETHICS.md`（issue #18）里写清用了哪个模型、哪个版本、什么许可。
- **注意区分两个 E3 的形态**：本任务生成的是「随机人脸」（无身份条件）；如果你想先快速验证「合成数据有没有用」这条路，**可以先跳过本任务**，直接下载现成的合成数据集（见 issue #10 的 B 轨），因为那样几乎零成本。
- 详细版见 [`docs/REFERENCES.md`](../REFERENCES.md) 第 5 节。

## 新手提示 / 卡住了怎么办

- **先跑 4 张，再跑 200 张**。第一次生成一定会有显存或路径问题，不要一上来就批量。
- `CUDA out of memory` 的排查顺序：降分辨率(512→384) → 开 fp16 → 开 cpu offload → 换更小的模型 → 降 batch 到 1。
- 生成很慢（例如 1 张 30 秒）是正常的，200 张约 1.5 小时，**放在离开电脑前跑**。
- **降级方案**：跑不动扩散模型时，改用 GAN 系（StyleGAN2-ADA 随机采样）或直接下载公开的合成人脸数据集，把 E3 标注为「使用公开合成数据」。
