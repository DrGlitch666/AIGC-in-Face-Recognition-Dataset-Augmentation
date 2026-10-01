# 网络受限环境的资源获取方案（镜像与离线缓存）

> **适用前提**：本项目两台机器**访问境外站点都不稳定**（HuggingFace / Google Drive / GitHub Releases 时通时不通）。
> 本文档给出**每一类资源的多条候选路线**，并要求**先自测、再下载**。
>
> ⚠️ **镜像站会变化**：下面所有 URL 都必须在 #22 里用脚本**实测一遍**，把结果回填到本文档的「实测结果」表。**没实测过的路线不要写进日志。**

---

## 0. 两条铁律

1. **先跑连通性预检，再决定下载策略**——不要在一个连不上的地址上反复重试浪费一晚上。
2. **下载一次，两台机器各自留缓存**——大文件不通过 Git，也不用互相传（见 #19）。

---

## 1. 资源 → 候选路线对照表

| 资源类型 | 具体要下什么 | 候选路线（按推荐顺序） | 备注 |
|---|---|---|---|
| **Python 包** | torch / diffusers / insightface / opencv ... | ① 清华 PyPI ② 阿里云 PyPI ③ 官方 PyPI（可能慢） | 先把 pip 全局源换掉，后面所有安装都受益 |
| **conda 包** | python=3.11、numpy 等 | ① 清华 anaconda 镜像 ② 官方 conda-forge | 写进 `.condarc` |
| **PyTorch 轮子** | torch ≥2.7 + **cu128**（RTX 50 系必须） | ① 阿里云 pytorch-wheels 镜像 ② 上海交大 pytorch-wheels 镜像 ③ 官方 `download.pytorch.org/whl/cu128`（**有时可直连**） | **cu128 是关键**：镜像上如果没有 cu128，就试官方索引；再不行退 cu126+老卡方案不适用于 50 系 |
| **HF 模型/数据集** | SD1.5、IP-Adapter、Arc2Face 权重、DigiFace-1M 镜像 | ① **`HF_ENDPOINT=https://hf-mirror.com`**（对 `huggingface_hub` / `diffusers` / `transformers` 透明生效） ② 手动 `git clone https://hf-mirror.com/<repo>` | 这是**最重要的一条**：设好环境变量后，原代码几乎不用改 |
| **国内模型社区** | SD1.5、SDXL、部分 IP-Adapter 类模型 | **ModelScope（魔搭）**：用 `modelscope` SDK 或 `git clone https://www.modelscope.cn/<org>/<model>.git` | 搜**同名模型**；找不到就用 hf-mirror |
| **insightface 模型包** | `buffalo_l` / `antelopev2`（275/344 MB） | ① GitHub Releases 加速前缀 ② HF 上的镜像仓库（经 hf-mirror） ③ 找同学/实验室已有文件拷贝 | 只有 275 MB，**用移动硬盘拷也行**，不要在这上面卡住 |
| **评测数据集** | LFW / CFP-FP / CALFW / CPLFW / TinyFace | ① 数据集官网 ② 国内网盘转载（百度网盘常见） ③ 实验室已有 | 官方站点本身就不稳定，**备好镜像** |
| **合成数据集（B 轨核心）** | Digi2Real / HyperFace / BIF-Face / DigiFace-1M | ① **Zenodo**（通常在境内可访问，只是慢） ② hf-mirror 上的镜像仓库 ③ GitHub Releases 加速前缀 | **优先 Zenodo**：它是 CERN 的，比 HF/Drive 稳 |
| **Google Drive 上的数据** | DCFace、IDiff-Face 数据集 | ❌ **默认放弃** | GDrive 在受限网络下基本不可用；改用 Zenodo 上的替代数据集 |
| **论文** | arXiv PDF | ① arXiv ② **ar5iv**（HTML 版，通常更快） ③ 实验室校园网 | 读论文不需要下载，直接看 HTML 更省事 |

> 💡 **战略结论**：在受限网络下，**B 轨（下载现成合成数据）优先选 Zenodo 上的数据集**；A 轨（本地生成）的模型**优先从 ModelScope 拿**，其次 hf-mirror。这条决定了 #10 的选型顺序。

---

## 2. 具体配置（Windows PowerShell，两台机器都做一遍）

```powershell
# ① Python 包：换成国内源
python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# ② conda：写 .condarc（清华）
@"
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
"@ | Set-Content "$env:USERPROFILE\.condarc" -Encoding utf8

# ③ HuggingFace 走镜像（对 diffusers / huggingface_hub 透明生效）
#    临时（当前会话）
$env:HF_ENDPOINT = "https://hf-mirror.com"
#    永久（用户级环境变量，重开终端生效）
setx HF_ENDPOINT "https://hf-mirror.com"

# ④ 大文件统一缓存目录（两台机器各自设自己的路径，写进 configs/local.<machine>.yaml）
#    例：D:\aigcfr-cache\{hf,models,datasets,weights}
```

**注意**：`HF_ENDPOINT` 必须**在启动 Python 之前**设置好才生效；用 VS Code 的话要重启终端。

---

## 3. 连通性预检（**第一天就做，10 分钟**）

用一个**一次性探测脚本**（放在项目之外，例如 `F:\aigcfr\_cache\tools`，**不进仓库**）对下面每个端点做一次 **5 秒超时**的探测，输出一张「通 / 不通 / 慢」的表：

| 端点 | 用途 |
|---|---|
| `https://pypi.tuna.tsinghua.edu.cn/simple/` | pip 源 |
| `https://mirrors.aliyun.com/pytorch-wheels/` | PyTorch 轮子镜像 |
| `https://download.pytorch.org/whl/cu128` | 官方 PyTorch 索引（cu128 兜底） |
| `https://hf-mirror.com` | HF 模型与数据集 |
| `https://www.modelscope.cn` | 国内模型社区 |
| `https://zenodo.org` | 合成数据集（B 轨） |
| `https://github.com` / `https://objects.githubusercontent.com` | 代码与 Releases（insightface 模型包） |
| `https://arxiv.org` / `https://ar5iv.labs.arxiv.org` | 论文 |
| `http://vis-www.cs.umass.edu/lfw/` | LFW 官方（大概率不通，用于确认需要镜像） |

**产出**：一张表 + 一句结论 —— **回填到本文档 §4 的实测结果表**，并在 Issue #22 评论里贴一份（例如「HF 走 hf-mirror 可用，GitHub 需加速，GDrive 放弃」）。
这份报告**两台机器各一份**，直接决定后面每一步用哪条路线。

---

## 4. 实测结果（**由 #22 回填**）

### 4.1 原始探测（8 个端点，单点超时 5s）

| 端点 | 用途 | 结果 | 耗时 |
|---|---|---|---|
| `pypi.tuna.tsinghua.edu.cn/simple/` | pip 源 | ✅ 200 | 635 ms |
| `mirrors.aliyun.com/pytorch-wheels/` | PyTorch 轮子镜像 | ✅ 200 | 173 ms |
| `download.pytorch.org/whl/cu128` | PyTorch 官方索引 | ✅ 200 | 1411 ms |
| `hf-mirror.com` | HF 模型/数据集 | ✅ 200 | 421 ms |
| `www.modelscope.cn` | 国内模型社区 | ✅ 200 | 662 ms |
| `zenodo.org` | 合成数据集托管 | ✅ 200 | 747 ms |
| `github.com` | 代码与 Releases | ✅ 200 | 255 ms |
| `arxiv.org` | 论文 | ✅ 200 | 338 ms |

**结论**：**8/8 可访问**，无需启用 §5 的止损路线。

### 4.2 按资源的实际使用结果

| 资源 | 实际走的路线 | 结果 | 实测人 | 日期 |
|---|---|---|---|---|
| pip 源 | 清华 tuna | ✅ 实测可用（注：裸浏览器 UA 探测会返回 403，但 pip 本身正常，**不要被这个 403 误导**） | A | 2026-09-23 |
| PyTorch cu128 | 官方索引 `download.pytorch.org/whl/cu128` | ✅ 装上 `torch 2.11.0+cu128`，`sm_120` 在架构列表里 | A | 2026-09-23 |
| HF 模型 | hf-mirror（配合 `HF_ENDPOINT`） | ✅ 连通；尚未实际下载大模型（W3/W4 才用） | A | 2026-09-23 |
| insightface 模型包 | insightface 自动下载（走 GitHub Releases） | ✅ `buffalo_l` 下载成功，无需加速前缀 | A | 2026-09-23 |
| 合成数据集 | Zenodo | ✅ 连通；尚未实际下载（W3 才用） | A | 2026-09-23 |
| conda 包 | 官方源 `repo.anaconda.com` | ✅ 未配镜像也能用（282 ms），**因此本项目不配 conda 镜像，少一个出错点** | A | 2026-09-23 |

> 每次成功或失败都记一行。**这是你们项目里最容易被忽略、但最省时间的一份文档。**
>
> ⚠️ 注意：这台机器的网络**比预期好得多**（境外源基本都能通）。B 的机器**未必一样**，
> 所以 B 必须自己跑一遍 §3 的探测并在这里补一行 —— **不要假设两台机器的网络环境相同**。

### 4.3 B 机器实测（Windows / 无独显，2026-09-24）

| 端点 | 用途 | 结果 | 耗时 |
|---|---|---|---|
| `pypi.tuna.tsinghua.edu.cn/simple/` | pip 源 | ✅ 200（本次较慢） | 35447 ms |
| `mirrors.aliyun.com/pytorch-wheels/` | PyTorch 轮子镜像 | ✅ 200 | 3467 ms |
| `download.pytorch.org/whl/cu128` | PyTorch 官方索引 | ✅ 200 | 771 ms |
| `hf-mirror.com` | HF 模型/数据集 | ✅ 200 | 1063 ms |
| `www.modelscope.cn` | 国内模型社区 | ✅ 200 | 3394 ms |
| `zenodo.org` | 合成数据集托管 | ✅ 200 | 2128 ms |
| `github.com` | 代码与 Releases | ✅ 200 | 4313 ms |
| `arxiv.org` | 论文 | ✅ 200 | 1172 ms |

**B 机结论**：8/8 端点均可访问，当前无需启用 §5 止损路线。清华 PyPI 本次 HTTP 探测耗时异常偏高，但 pip 已成功配置为清华源，后续以实际 `pip install` 表现为准。HF 使用 `HF_ENDPOINT=https://hf-mirror.com`。

### 4.4 GitHub Releases 下载测速（2026-10-01，A 机器）

> **背景**：`antelopev2.zip`（344 MB）从 GitHub Releases 直连**慢到不可用**，
> 于是对每条候选路线做 3 MB 探针测速。结论对所有 GitHub Releases 资源都适用。

| 路线 | 实测速度 | 344 MB 预计耗时 | 结论 |
|---|---|---|---|
| **`gh-proxy.com/<原始URL>`** | **2.004 MB/s** | **≈ 2.9 分钟** | ✅ **首选** |
| `hf-mirror.com/`（第三方镜像仓库） | 1.050 MB/s | ≈ 5.5 分钟 | ✅ 回退 |
| `ghfast.top/<原始URL>` | 0.469 MB/s | ≈ 12 分钟 | ⚠️ 可用但慢 |
| `ghproxy.net/<原始URL>` | 0.226 MB/s | ≈ 25 分钟 | ⚠️ 慢 |
| **GitHub 直连** | **0.058 MB/s** | **≈ 99 分钟** | ❌ **不要用** |
| `github.moeyy.xyz` | — | — | ❌ DNS 解析失败 |
| `hub.gitmirror.com` | — | — | ❌ DNS 解析失败 |
| `gh.llkk.cc` | — | — | ❌ 超时 |

**规律**：直连与代理差 **34 倍**。凡是 GitHub Releases 上的资源（insightface 模型包等），
**一律走 `gh-proxy.com` 前缀**，不要直连。

> ⚠️ 代理域名会失效（本次就有 2 个 DNS 解析不了、1 个超时）。
> 所以配置里要**多列几个镜像** —— `scripts/download_data.py` 现在会在
> 某个镜像连续失败后**自动换下一个**。

### 4.5 归档压缩包的存放约定（写下来避免再踩）

| 情况 | 例子 | 解压到 |
|---|---|---|
| 压缩包**自带一层顶层目录** | `lfw.tgz` 里就是 `lfw/` | 目标目录本身 |
| 压缩包是**散装文件** | `antelopev2.zip` 里直接是 `*.onnx` | 目标目录下**建同名子目录** `antelopev2/` |

第二条很关键：insightface 期望 `<root>/models/antelopev2/*.onnx`。
散着解压会把 onnx 撒到 `<root>/models/` 下，模型就找不到了。
`download_data.py` 会**自动判断**这两种情况。

---

## 5. 如果连镜像都不通（止损路线）

按顺序尝试，**不要在同一层反复挣扎**：

1. **降低分辨率需求**：只用 LFW（13k 张，几十 MB 级）+ 传统增强，先把「数据 → 训练 → 评测 → 网站」跑通，合成数据用**自己用最小模型现场生成**的少量图（例如 SD1.5 从 ModelScope 拿不到就换更小的模型）。
2. **借道**：实验室/学校机器、同学已下载的文件、移动硬盘拷贝——**一个 275 MB 的模型包不值得花两天**。
3. **改用 CPU 可行的小模型**做识别基线（评测仍然成立，只是慢）。
4. **明确记录降级**：在 `docs/RESULTS.md` 的局限里写清「因网络限制，未使用 X」，这比假装做过要好得多。

> ⚠️ **绝对不要**为了绕过限制去使用来源不明的"破解/整合包"——模型与数据的许可本身就是本项目要讨论的内容（见 #18）。
