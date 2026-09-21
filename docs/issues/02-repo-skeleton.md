---
title: "[脚手架] 建立仓库目录结构、配置系统与工程规范"
slug: "02-repo-skeleton"
labels: ["优先级:P0", "类型:环境", "难度:入门"]
milestone: "M0 脚手架与地基"
---

## 背景 / 为什么做这个

后面十几个 Issue 会往仓库里塞代码、配置和结果。如果现在不把目录和约定定下来，两周后你会面对一堆散落的脚本、找不到哪个配置对应哪次实验。

这一步的产出是**骨架和规矩**：目录结构、`.gitignore`（防止把数据和大模型权重提交上去）、配置模板、以及一个能跑起来的 `pytest`。

## 目标

仓库一眼能看懂，`git status` 永远干净，任何实验都能「一个配置文件 + 一条命令」表达。

## 任务清单

- [ ] 按 [`docs/FRAMEWORK.md` 第 6 节](../FRAMEWORK.md) 建立目录：`configs/ data/ docs/ notebooks/ results/ scripts/ src/aigcfr/ site/ tests/ tools/`
- [ ] 写 `.gitignore`，**必须包含**：`data/`、`*.pth`、`*.onnx`、`*.ckpt`、`*.safetensors`、`outputs/`、`__pycache__/`、`.ipynb_checkpoints/`、`site/node_modules/`、`site/dist/`
- [ ] 在每个空目录放 `.gitkeep`（`data/` 除外，它整个被忽略）
- [ ] 写 `configs/_template.yaml`：data / gen / filter / train / eval 五段齐全，每个字段带注释
- [ ] 写 `configs/exp/_template.yaml`：实验配置模板，含 `exp_id`、`seed`、`mix_ratio`、`benchmarks` 字段
- [ ] 写 `configs/exp/matrix.yaml`：实验矩阵占位（供 {{#14-mix-ratio-matrix}} 使用）
- [ ] 建立 `src/aigcfr/` 包结构：`data/ generate/ filter/ train/ eval/ report/ utils/`，每个含 `__init__.py`
- [ ] 写 `src/aigcfr/utils/config.py`：加载 YAML + 计算配置哈希（供结果溯源用）
- [ ] 写 `tests/test_config.py`：一个能通过的测试（例如加载模板配置不报错）
- [ ] 更新 `README.md`：项目一句话简介 + 快速开始占位 + 目录说明
- [ ] 加 `LICENSE`（代码建议 MIT 或 Apache-2.0；**注意：这不覆盖数据集与第三方模型权重的许可**）
- [ ] 写 `scripts/` 下的占位入口（先只有 `--help` 和 `--config` 参数解析）

## 验收标准（Definition of Done）

- [ ] 目录树与 `docs/FRAMEWORK.md` 第 6 节一致
- [ ] `git status` 干净；在 `data/` 里随便放个文件后 `git status` **不应该**显示它
- [ ] `pytest -q` 至少 1 个测试通过
- [ ] `python scripts/train.py --help` 能正常输出帮助信息
- [ ] `README.md` 里有目录说明表

## 交付物

- 目录结构 + `.gitignore` + `LICENSE`
- `configs/_template.yaml`、`configs/exp/_template.yaml`、`configs/exp/matrix.yaml`
- `src/aigcfr/utils/config.py`
- `tests/test_config.py`
- 更新后的 `README.md`

## 依赖

- 建议先完成 {{#01-env-setup}}（需要 Python 环境跑 `pytest`），但目录与配置文件可以先写，两件事可以交替进行
- 被依赖：{{#03-data-pipeline}}、{{#09-unconditional-generation}}、{{#16-site-skeleton}} 及后续全部任务

## 预估工作量

2~4 小时

## 参考

- [`docs/FRAMEWORK.md`](../FRAMEWORK.md) §3.3 层间契约、§6 仓库结构
- [Python 官方 `.gitignore` 模板](https://github.com/github/gitignore/blob/main/Python.gitignore)
- [choosealicense.com](https://choosealicense.com/) 选许可证

## 新手提示 / 卡住了怎么办

- 目录先建空壳就行，**不要**在这一步写业务逻辑。
- `.gitignore` 里的 `data/` 一定要确认生效：`git check-ignore -v data/test.png` 应该输出匹配到的规则。
- 配置模板宁可字段多写几个带注释的占位，也不要后面临时加，否则每个实验的配置格式都不一样，汇总脚本就没法写了。
