# 环境搭建（SETUP）

> **目标**：任意一台机器，按本文档从零装到 `pytest` 全绿、并知道自己是 A / B / C 哪一档。
> **本文档是"实际走通"的记录**，不是设想——所有命令都在 A 的机器（Windows + RTX 5060 Laptop 8 GB）上验证过；
> 每个"已知坑"都是真实踩过的。
>
> 相关文档：[`FRAMEWORK.md`](FRAMEWORK.md) §4（技术选型与档位）、[`MIRRORS.md`](MIRRORS.md)（受限网络）、
> [`WORKFLOW.md`](WORKFLOW.md) §2（命令约定）。

---

## 0. 先确定你的机器属于哪一档

| 档位 | 典型配置 | 能跑什么 | 本文档的走法 |
|---|---|---|---|
| **A** | ≥ 12 GB 显存独显 | 全部（含 SDXL 路线） | §4a + §5a |
| **B** | 6~12 GB 显存独显（如 RTX 5060 Laptop 8 GB） | 本地生成（SD1.5）+ 全部训练 | §4a + §5a |
| **C** | 无独显 / macOS / 集显 | 数据、评测（小规模）、筛选、网站、文档 | §4b + §5b |

**磁盘规划**（大文件一律放非系统盘，路径只写在 `configs/local.<machine>.yaml` 里）：

```
<缓存盘>/
├─ data/     数据集（原始 / 中间 / 处理后）
├─ models/   预训练权重（buffalo_l、SD1.5 等）
├─ hf/       HuggingFace 缓存
├─ ckpt/     训练 checkpoint
├─ synth/    合成的图
└─ _cache/   下载的压缩包与临时解压（放一次性脚本也放这里）
```

---

## 1. 装 Miniconda

**为什么不用系统自带的 Python**：系统 Python 往往是 3.13/3.14，很多深度学习库还没有对应的 wheel，
`pip install torch` 会直接失败或触发本地编译。

1. 下载（境内推荐清华镜像，126 MB，实测可用）：
   `https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Windows-x86_64.exe`
2. 安装时的 **4 个关键选项**：

| 安装界面 | 选什么 | 为什么 |
|---|---|---|
| Install for | **Just Me** | 选 All Users 需要管理员权限，且路径会带空格 |
| Destination Folder | 改成 **`D:\miniconda3`** | 装非系统盘；**路径里不能有空格和中文**（深度学习工具链对空格路径极敏感） |
| Add Miniconda3 to my PATH | ☐ **不要勾** | 系统里已有 Python，加 PATH 会打架。用「Anaconda Prompt」即可 |
| Clear the package cache | ☐ 不勾 | — |

3. 装完在「开始」菜单里找 **`Anaconda Prompt (miniconda3)`**。

   > ⚠️ **名字是 Miniconda 的安装程序自己起的**（历史遗留命名），看到 "Anaconda" 不代表装错了。
   > 也有 `Anaconda Powershell Prompt (miniconda3)`，两个都能用。

4. 验证：打开该快捷方式，输入 `conda --version`，能打印版本号即可。

**已知坑**：
- 在**普通 PowerShell** 里输 `conda` 会提示"无法识别"——**这是正常的**（我们故意没加 PATH）。
- 首次 `conda create` 可能报 `CondaToSNonInteractiveError: Terms of Service have not been accepted`。
  这是 Anaconda 从 2024 年起对 `defaults` 频道加的服务条款（免费范围覆盖个人与高校研究，**接受 ≠ 付费**）。
  按报错里给出的三条命令执行一遍即可：

  ```cmd
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2
  ```

---

## 2. 配镜像（两人两台机器都做）

> 受限网络下这一步是**省时间的关键**。镜像实测结果记录在 [`MIRRORS.md`](MIRRORS.md) §4。

```cmd
python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
```

```cmd
setx HF_ENDPOINT "https://hf-mirror.com"
```

- `pip config` 写的是**用户级**配置，对所有 conda 环境生效。
- `setx` 设的是**用户级环境变量**，**需要重开终端才生效**（VS Code 里要重启终端窗口）。
- `HF_ENDPOINT` 对 `huggingface_hub` / `diffusers` / `transformers` **透明生效**，代码一行都不用改。

---

## 3. 建 conda 环境

```cmd
conda create -n aigcfr python=3.11 -y
conda activate aigcfr
```

**激活成功的标志**：命令提示符前面出现 **`(aigcfr)`**（而不是 `(base)`）。

> ⚠️ **两人必须用同一个 Python 次版本**（都 3.11 或都 3.12），否则依赖解析结果会不同。

---

## 4a. 装 PyTorch（**有 NVIDIA 显卡**：A / B 档）

```cmd
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

> ⚠️ **`--index-url` 不能省**：它临时覆盖清华源，直接从 PyTorch 官方索引取 **cu128** 构建。
> RTX 50 系（Blackwell，`sm_120`）**必须 PyTorch ≥ 2.7 + cu128**，否则会报
> `CUDA error: no kernel image is available for execution on the device`。
> 老卡（RTX 30/40 系）用 cu126/cu121 也可以，规则是"按自己驱动支持的 CUDA 版本选"。

**验证**（这一步不过，后面全部免谈）：

```cmd
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0)); print(torch.cuda.get_arch_list())"
```

必须看到 `True`、正确的 GPU 名称、以及架构列表里**含 `sm_120`**（RTX 50 系）。

## 4b. 装 PyTorch（**无独显 / macOS**：C 档）

```cmd
python -m pip install torch torchvision
```

（走默认源即可，装到的是 CPU 轮子。macOS 上会装到 MPS 可用的轮子。）

---

## 5a. 装其余依赖（**有 NVIDIA 显卡**）

```cmd
python -m pip install -r requirements.txt
```

**然后检查 onnxruntime**（这一步很容易出错，必查）：

```cmd
python -m pip list | findstr /I onnxruntime
```

| 输出 | 处理 |
|---|---|
| 只有 `onnxruntime-gpu` | ✅ 正确 |
| 出现 `onnxruntime`（无 `-gpu`）或两个都有 | ❌ 会冲突，执行下面的修复 |

```cmd
python -m pip uninstall -y onnxruntime onnxruntime-gpu
python -m pip install onnxruntime-gpu==1.26.0
```

> **为什么要"先卸两个再装一个"**：这两个包**共用同一个 `onnxruntime` 目录**，
> 只卸其中一个会留下残缺文件；同时装两个则会静默退回 CPU。

**验证 CUDA 执行器真的能用**（关键，且必须 `import torch` 在前）：

```cmd
python -c "import torch, onnxruntime as ort, glob; m=sorted(glob.glob('F:/aigcfr/models/insightface/models/buffalo_l/*.onnx'))[0]; s=ort.InferenceSession(m, providers=['CUDAExecutionProvider','CPUExecutionProvider']); print('ORT', ort.__version__, '| session EP =', s.get_providers())"
```

期望：`session EP = ['CUDAExecutionProvider', 'CPUExecutionProvider']`。

> ⚠️⚠️ **本项目最重要的一条代码约束**：
> **创建 ORT 会话之前必须先 `import torch`。**
> torch 导入时会把自带的 CUDA DLL 目录注册进 Windows 的 DLL 搜索路径，ORT 才能加载 CUDA EP。
> **顺序反了不会报错**——它只是悄悄退回 CPU，你只会觉得"怎么这么慢"。
> 详见下面的「已知坑 K1」。

> 💡 **C 档可以跳过**「生成」那一组依赖（`diffusers` / `transformers` / `accelerate` / `safetensors`），
> 它们只在 W3/W4 生成合成图时用得到。

## 5b. 装其余依赖（**无独显**）

```cmd
python -m pip install onnxruntime insightface opencv-python numpy Pillow tqdm PyYAML matplotlib pytest
```

> CPU 机器用 `onnxruntime`（不加 `-gpu`），**绝不能同时装两个**。
> 同样不需要「生成」那一组依赖。

---

## 6. 拿人脸模型包 `buffalo_l`

```cmd
python -c "from insightface.app import FaceAnalysis; app=FaceAnalysis(name='buffalo_l', root='F:/aigcfr/models/insightface', providers=['CUDAExecutionProvider','CPUExecutionProvider']); app.prepare(ctx_id=0, det_size=(640,640)); print('buffalo_l OK'); print('EP =', app.models['recognition'].session.get_providers())"
```

- 首次运行会**自动下载约 275 MB** 到 `<root>\models\buffalo_l\`。
- 无独显的机器把 `providers` 换成 `['CPUExecutionProvider']`、`ctx_id` 换成 `-1`。

**如果自动下载失败**（走的是 GitHub Releases，境内可能不通），按顺序试三条路：

| 路线 | 做法 |
|---|---|
| ① HF 镜像 | 从 hf-mirror 上的 insightface 镜像仓库取（搜 `deepghs/insightface` 这类镜像） |
| ② GitHub 加速前缀 | 下载 `https://github.com/deepinsight/insightface/releases/download/model-zoo/buffalo_l.zip`（前缀形式如 `https://ghfast.top/<URL>`，需自测） |
| ③ 人工拷贝 | **只有 275 MB，找同学拷一份，不值得花两天** |

解压后确认这些文件在 `<root>\models\buffalo_l\` 下：
`det_10g.onnx`、`w600k_r50.onnx`、`2d106det.onnx`、`1k3d68.onnx`、`genderage.onnx`

---

## 7. 验证：一条命令判定档位

```cmd
python -c "import sys; sys.path.insert(0,'src'); from aigcfr.utils.config import detect_vram_gb as v, decide_profile as d; x=v(); print('vram_gb =', x, '| profile =', d(x))"
```

期望：A/B 档机器打印出显存与 `profile = a/b`；C 档打印 `vram_gb = None | profile = cpu`。

再跑一次配置层测试（不需要 GPU、不需要数据集，任何机器都应全绿）：

```cmd
python -m pytest -q
```

期望：`19 passed`。

**把这两条的输出贴到 Issue #1 的评论里**（两台机器各一条）——这就是本项目的"环境自检记录"。

---

## 8. 冻结与复现

```cmd
python -m pip freeze > requirements-lock.<machine>.txt
```

例如 A 的机器生成 `requirements-lock.a.txt`，B 的机器生成 `requirements-lock.b.txt`。

| 文件 | 是否入库 | 说明 |
|---|---|---|
| `requirements.txt` | ✅ | **人写的范围**，说明每个依赖为什么在；两台机器共用 |
| `requirements-lock.<machine>.txt` | ✅ | **机器生成的确切版本**，用于复现；**每台机器一份** |
| `environment.local.yml` | ❌ | 机器私有依赖（已被 .gitignore 忽略） |

> ⚠️ **为什么锁定文件要按机器分开**：A 的锁定文件里是 `torch==2.11.0+cu128`，
> 这个 `+cu128` 本地版本号**只存在于 PyTorch 专用索引**，PyPI 上没有。
> 如果 B（CPU 机器）直接拿 A 的锁定文件安装，会报
> `ERROR: No matching distribution found for torch==2.11.0+cu128`。
> 所以**各人用各人的锁定文件**。

**从零复现（W6 会用到）**——注意 torch 那一步必须单独指定索引：

```cmd
conda env remove -n aigcfr -y
conda create -n aigcfr python=3.11 -y
conda activate aigcfr
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-lock.<你的机器>.txt
python -m pytest -q
```

> 无独显的机器把第 4 行换成 `python -m pip install torch torchvision`（去掉 `--index-url`），
> 并使用自己的 `requirements-lock.b.txt`。
>
> `pip install -r` 对已满足的包会直接报 `Requirement already satisfied`，不会再去索引里找，
> 所以"先装 torch、再装锁定文件"这个顺序是安全的。

---

## 9. 已知坑清单（**全部是我们实际踩过的**）

| 编号 | 坑 | 症状 | 处理 |
|---|---|---|---|
| **K1** | **onnxruntime 版本与 torch 的 CUDA 版本不匹配** | 满屏 `Failed to load cublasLt64_13.dll` / `cudart64_13.dll`，最后 `providers = ['CPUExecutionProvider']` | PyPI 上的 `onnxruntime-gpu` **从 1.27 起默认改成 CUDA 13 构建**；钉 **`==1.26.0`**（= CUDA 12.8 + cuDNN 9，与 torch cu128 一致） |
| **K2** | **`import torch` 写在 ORT 后面** | 不报错，但推理速度只有 CPU 水平 | 任何创建 ORT 会话的脚本，**`import torch` 必须在最前面**；或显式调用 `onnxruntime.preload_dlls()` |
| **K3** | 同时装了 `onnxruntime` 和 `onnxruntime-gpu` | 静默退回 CPU / DLL 冲突 | 两个都卸掉，只装一个 |
| **K4** | 显卡架构太新 | `CUDA error: no kernel image is available` | RTX 50 系必须 PyTorch ≥ 2.7 + **cu128** |
| **K5** | 装成了 CPU 版 torch | `torch.cuda.is_available()` 为 `False` | `pip list` 看是否有 `+cpu` 后缀，重装 cu128 |
| **K6** | conda 的 Terms of Service 阻断 | `CondaToSNonInteractiveError` | 见 §1 的三条 `conda tos accept` |
| **K7** | Windows 控制台 GBK 编码 | 脚本打印 emoji 时 `UnicodeEncodeError` 崩溃 | 脚本开头加 `sys.stdout.reconfigure(errors="replace")`；控制台输出只用 ASCII 标记 |
| **K8** | 路径含空格或中文 | 各种库报奇怪的错 | 环境装到 `D:\miniconda3`，项目路径保持英文 |
| **K9** | `HF_ENDPOINT` 不生效 | 模型下载卡死或失败 | `setx` 之后**必须重开终端** |
| **K10** | 把数据/权重提交进仓库 | 仓库巨大、隐私风险 | `.gitignore` 已排除 `data/`、`*.pth`、`*.onnx`、`configs/local.*.yaml` |

---

## 10. 两台机器的差异一览

| 项目 | A（RTX 5060 Laptop 8 GB） | B（Windows 无独显） |
|---|---|---|
| 档位 | **b** | **cpu** |
| torch | `--index-url .../cu128` | 默认源（CPU 轮子） |
| onnxruntime | `onnxruntime-gpu==1.26.0` | `onnxruntime`（CPU） |
| 生成依赖 | 需要（W3/W4 用） | 不需要 |
| `check_env` 里 `ctx_id` | `0` | `-1` |
| `configs/local.<machine>.yaml` | 缓存放 F 盘 | 缓存放自己的盘 |
| 能跑 | 生成 + 全部训练 + 评测 | 数据、评测（小规模）、筛选、网站、文档 |

> 两边的**代码完全一样**，差异只体现在 `configs/profiles/<档位>.yaml` 与
> `configs/local.<machine>.yaml` 两个文件里——这就是把"机器差异"关进配置的好处（见 [`FRAMEWORK.md`](FRAMEWORK.md) §4.3）。
