#!/usr/bin/env python
"""把 GitHub 上的所有 issue 同步到项目真实现状，并给 B 提新任务。

用法（需要 GH_TOKEN 环境变量，从 Windows 凭据管理器取）：
    python tools/sync_issues.py --dry-run     # 只看要做什么
    python tools/sync_issues.py               # 真正执行

设计说明：注释正文用三引号字符串写，避免反引号转义问题（上一版 JS 就栽在这里）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

REPO = "DrGlitch666/AIGC-in-Face-Recognition-Dataset-Augmentation"
API = f"https://api.github.com/repos/{REPO}"

# ---------------------------------------------------------------- issue 现状同步
UPDATES: dict[int, dict] = {}

UPDATES[1] = {"close": True, "comment": """**✅ 已完成（A 机器）**

RTX 5060 Laptop 8GB + Miniconda `aigcfr`（Python 3.11.16）验证通过：
torch 2.11.0+cu128（sm_120）、onnxruntime-gpu 1.26.0、insightface 2.0、diffusers 0.40.0、peft。

踩坑已固化：ORT 1.30(CUDA 13) 与 torch cu128 冲突 → 钉 1.26.0；必须 `import torch` 在 onnxruntime 之前。

配置分层已落地：`base.yaml` → `profiles/{a,b,cpu}.yaml` → `configs/exp/*.yaml` → `configs/local.<machine>.yaml`
（local 只允许改 paths/runtime/cache/env）。

**B 机器（C 档 / CPU）尚未做自检** —— 见新任务 #24。"""}

UPDATES[2] = {"close": True, "comment": """**✅ 已完成**

仓库目录结构、`src/aigcfr/` 包、4 层配置合并 + `config_hash`、`tools/file_issues.mjs`
（CRLF 安全的 front-matter 解析）、`.gitignore` 全部就位。

单元测试 60 个通过（config / lfw / verify / embed / train）。"""}

UPDATES[3] = {"close": True, "comment": """**✅ 已完成**

* LFW 已下载（5749 身份 / 13233 张），走 hf-mirror。
* `scripts/build_dataset.py` 产出 **96 身份 / 3590 张** 112×112 ArcFace 对齐图。
* manifest 契约冻结：`data/manifests/toy_train.jsonl`，`path` **相对 `paths.data_root`**。
* 校验测试与对齐抽样图在 `tests/`。

**已知事实（写进报告局限）**：LFW 官方 6000 对覆盖 4281 个身份，**包含全部 96 个训练身份**，
训练/测试泄漏不可避免。已用无泄漏协议量化，见 #7。"""}

UPDATES[5] = {"close": True, "comment": """**✅ 已完成**

E0（零样本 buffalo_l）：LFW 官方协议 **accuracy 0.9985**。

后续用无泄漏协议复测：**TAR@FAR=1e-3 = 0.9989、accuracy = 0.9993**
（协议见 #7 与 `scripts/build_verify_protocol.py`）。"""}

UPDATES[6] = {"close": True, "comment": """**✅ 已完成并加固**

iresnet18(62.6M) + ArcFace(m=0.5, s=64)，SGD lr=0.1 余弦退火。
**E1（纯真实 3590 张，3 种子）：accuracy 0.8812 ± 0.0032**（official 协议）。

**训练不稳定性定位与修复**（本轮最重要的工程发现）：
5 次训练崩了 2 次 —— 某一类的分类头权重爆炸成为"吸引子"，`loss` 卡在 ~16、
`train_acc` 恒为 0，**同数据同配置，只差随机流**。
加入 **梯度裁剪 5.0 + LR warmup 3 轮**后 **6 次零崩**，且收敛更好（0.9880 → 0.9992）。
守门指标：分类头权重范数的「最大/中位数」比值（健康 1.37~1.45，崩塌 3.30）。

另修复两处：
* `DataLoader` 未按 worker 播种 numpy → 增强同相位（`worker_init_fn`，
  且**必须放模块级**，否则 Windows 的 spawn 无法 pickle）
* `best.pth` 原先在第 1 轮（acc=0）就会被写出，已改为 `acc>0 且创新高`"""}

UPDATES[7] = {"comment": """**⚠️ 大部分完成，剩 1:N 与分桶**

**已实现并验证**：
* 1:1 验证（LFW 10 折 + filtered 子集），`metrics.json` schema 冻结
* `tar_at_far` / `best_threshold` / `fold_metrics`，并修掉两个**不报错但会把指标算错**的坑：
  并列分数用分位数会让 FAR 变成 1.0（改为"第 k 大 impostor + 严格 >"）；
  `best_threshold` 误用假正例会让 accuracy 固定为 0.5
* **新增无泄漏大样本协议**（`scripts/build_verify_protocol.py` + `scripts/evaluate_heldout.py`）：
  用 LFW 的 **5653 个未见身份**构造 **125,191 对**（11,381 same / 113,810 diff），
  **身份与训练集交集 = 0**；FAR=1e-3 的 impostor 从官方协议的 **3 个**提到 **113 个**，
  TAR 的 bootstrap 标准误从 ~0.065 降到 ~0.005

**未做**：1:N 闭集检索、质量/姿态分桶（原计划依赖 B 轨数据，B 轨已砍）。

**B 的补强任务见新开卡片**（边界单元测试 + 跨机复现）。"""}

UPDATES[8] = {"close": True, "comment": """**❌ 未做，已从范围中移除**

理由：项目主线结论（合成数据是否有效）不依赖 E2 对照组，而 60 人时预算不足以再开一条对照。
在报告"未来工作"中说明。"""}

UPDATES[9] = {"close": True, "comment": """**❌ 未做，已从范围中移除**

理由：其要回答的"合成数据有没有用"已由 A 轨的身份保持生成（#10 / #14）直接回答，
重复投入收益低。报告中引用文献结论。"""}

UPDATES[10] = {"close": True, "comment": """**✅ 已完成（本项目技术含量最高的一块）**

**最终配方**（`scripts/generate_plusv2.py`，96 身份 × 10 张 = 960 张 / 58 分钟）：

| 项 | 值 |
|---|---|
| 适配器 | **IP-Adapter-FaceID-plusv2** |
| **`shortcut`** | **True** —— 为 False 时感知器输出会**覆盖** ID 分支，身份信号全丢 |
| CLIP 分支 | `hidden_states[-2]`；负例 = **对全零图像编码**；输入 `norm_crop(..., 224)` |
| ID 分支 | `buffalo_l` 的 `normed_embedding`，**多参考图平均** |
| 结构参考 | 每身份最多 2 张（质量闸门 + 主动挑差异最大者），**轮换使用** |

**质量对照**（`id_sim` = 与本人真实图中心的余弦；参照：**同人真实照片 0.745 / 不同人 <0.25**）：

| 方案 | 均值 | 最小值 |
|---|---|---|
| base FaceID | +0.3891 | +0.2035 |
| plusv2 官方配方 | +0.5138 | +0.4320 |
| **+ 多参考平均（采用）** | **+0.6084** | **+0.5273** |
| + Realistic Vision V6 底模 | +0.3999 | +0.2836（出局）|

**11 条踩坑记录**已固化为 `docs/W3-GENERATION.md`，包括：
检测尺寸的幸存者偏差、`peft` 缺失导致 FaceID LoRA 被静默跳过、
ID 嵌入必须用 buffalo_l 而非 antelopev2（交叉验证排除了度量偏差）、
diffusers 对 plusv2 是未完成实现（`clip_embeds` 从无赋值）。

视觉对照图：`results/runs/w3-generator-comparison/contact_sheet.png`。"""}

UPDATES[11] = {"close": True, "comment": """**✅ 已完成**

`scripts/filter_synth.py`：**质量地板 0.45 + 冗余剪枝**（迭代删除与同身份其余图最相似的那张），
每身份保留 ≤8 张。**960 → 755 张**，均值 `id_sim` **+0.6154**，最小 +0.4505，
**88/96 个身份保留满 8 张**。

**关键设计发现**：**不能按 `id_sim` 排名取前 N** —— 实测这样选出的 8 张比不剪枝的 10 张
**还要同质**（CLIP 结构相似度 0.7049 vs 0.7003），因为被删的恰是"为多样性特意挑的第二参考图"
所生成的图。冗余剪枝则降到 0.6846。

配套新增 `scripts/align_synth.py`：合成图是 512×512 原图、真实图是 112×112 对齐裁剪，
必须走**同一套 ArcFace 对齐**（且 **det_size 要用 320** —— 640 对生成图经常一张脸都检不出）。
对齐前后特征一致性 **1.0000**，零信息损失。"""}

UPDATES[14] = {"comment": """**⚠️ 部分完成：三点对照做了，占比扫描未做**

**已完成的三方对照**（同模型、同损失、同轮数、同增强，**只差训练数据**；3 个随机种子）：

| 方案 | 训练集 | accuracy (official) | TAR@FAR=1e-3 |
|---|---|---|---|
| E1 | real 3590 | 0.8812 ± 0.0032 | 0.4208 ± 0.0040 |
| **E4a** | + synth 755（**17.4%**）| **0.8899 ± 0.0026** | 0.4548 ± 0.0321 |
| E4b | + synth 448（11.1%）| 0.8875 ± 0.0191 | 0.4597 |

**结论**：

* ✅ **accuracy 显著提升**：Δ +0.0082~+0.0093，Welch **t = 3.35~3.98**，
  **三个协议、三个种子方向完全一致**（official / filtered / held-out 无泄漏）
* ⚠️ **TAR@FAR=1e-3 未达显著**：Δ +0.018~+0.039，t = 1.52~1.82。
  逐种子配对 bootstrap（无泄漏协议、125k 对）显示 **2 个种子强阳性（P>99.9%）、
  1 个种子无效应**；**种子间标准差 0.026 > 效应量 0.018**
* **合成占比更高者更优**：E4a(17.4%) > E4b(11.1%)，
  说明此质量水平下**数量比边际质量更值钱**

**同时也量化了官方协议的问题**：官方协议 TAR 增益 +0.034，无泄漏协议只有 +0.018
—— **官方 6000 对含全部 96 个训练身份，泄漏让效应被高估近一倍**。

**未做**：占比扫描（17.4% / 40% / 60%）。按当前种子方差外推，
TAR 要达到双侧显著约需 **8 个种子/方案**。

统计与原始数据：`results/runs/w4-e4-comparison/comparison.json`；汇总脚本 `tools/w4_compare.py`。"""}

UPDATES[15] = {"close": True, "comment": """**✅ 已完成**

`docs/REPORT.md`：方法 / 结果 / 结论 / 局限 / 复现 / 交付物清单。
所有数字可在 `results/runs/*/metrics.json` 与 `results/runs/w4-e4-comparison/comparison.json`
中逐项核对。

B 的排版与文献补充见新开卡片。"""}

UPDATES[19] = {"close": True, "comment": "**✅ 已完成**（协作规范、PR 模板、Issue 卡片体系）。由 B 交付。"}

UPDATES[20] = {"close": True, "comment": """**✅ 已完成**

4 层配置合并 + `config_hash` + 硬件档位自动判定（`detect_vram_gb` / `decide_profile`）
+ `local.*.yaml` 白名单与 `guard_local`。"""}

UPDATES[22] = {"close": True, "comment": "**✅ 已完成**（由 B 交付并 closed）。镜像实测速度表见 `docs/MIRRORS.md` §4.4。"}

UPDATES[13] = {"close": True, "comment": """**❌ 未做，已从范围中移除**

属性可控生成依赖 ControlNet，在 8 GB 显存 + 受限网络的条件下成功率低。
报告"未来工作"中说明。"""}

UPDATES[16] = {"close": True, "comment": """**🔁 已被取代**

本站的现实形态与原计划有出入（不必用 Vite + React + ECharts）。
**新的任务卡见 [B-1]**，其中包含完整的验收标准与当前的最新结论数字。"""}

UPDATES[17] = {"close": True, "comment": """**🔁 已被取代**

内容要求已并入 **[B-1]**（含"必须如实展示 TAR 未达显著"、
"每张合成图标注 `id_sim`"、"数字必须脚本生成禁止手抄"等硬性要求）。

原卡里的"公平性"部分建议降级为"未来工作"—— RFW 已在计划中砍掉。"""}

# ---------------------------------------------------------------- 给 B 的新任务
NEW_ISSUES = [
    {
        "title": "[B-1] 展示网站：静态站 + 内容页（合并原 #16 / #17）",
        "labels": ["类型:网站", "优先级:P0", "难度:中等"],
        "body": """## 背景（现状已变，以本卡为准）

A 轨（自生成合成数据）**已全部跑完**，主结论已经成立：

* ✅ **合成数据显著提升 LFW accuracy**：0.8812 → 0.8899（Δ+0.0086，Welch t=3.59）；
  无泄漏协议 0.8693 → 0.8775（t=3.98）。**三个协议、三个种子方向一致。**
* ⚠️ **TAR@FAR=1e-3 呈正向趋势但未达显著**（Δ+0.018~+0.039，t=1.52~1.82）。
* 生成质量：合成图与本人真实图中心的余弦（`id_sim`）均值 **0.6154**
  （参照：同人真实照片 0.745、不同人 <0.25）。

**原 #16 里的 "Vite + React + ECharts" 不强制** —— 用最省事、最不容易坏的方式即可，
**纯静态 HTML/CSS + 少量 JS + matplotlib 预生成 PNG 完全可以接受**。
关键是能双击打开、能被检索、数字不会写错。

## 硬性要求

1. **≤5 个页面**，`index.html` **双击即可打开**（无构建步骤、无后端、无 CDN 依赖）。
2. **所有数字必须由脚本从 `results/runs/*/metrics.json` 与
   `results/runs/w4-e4-comparison/comparison.json` 生成，禁止手抄。**
   写一个 `tools/export_site_data.py`，页面从这里取数。
   **理由：手抄必然出错，而报告里错一个数字就全盘失信。**
3. 内容至少覆盖：
   * 方法与流水线（数据 → 生成 → 筛选 → 训练 → 评测）
   * **生成器对照**（base +0.3891 → plusv2 +0.5138 → +多参考 **+0.6084**；
     Realistic Vision 出局 +0.0428）
   * **真 / 合成视觉对照图库**，每张合成图**标注它的 `id_sim`**
     （现成素材：`results/runs/w3-generator-comparison/contact_sheet.png`）
   * **E4 结论**：三方对照表 + **带误差棒的图** + 显著性说明
   * **必须如实展示"TAR 未达显著"**，不要只放 accuracy
   * 踩坑记录（`docs/W3-GENERATION.md` 有 11 条，挑最有代表性的）
   * 复现指南
4. **图表 ≥4 张**，全部脚本生成：生成器对比、IP-Adapter 强度扫描、
   E4 对照（带误差棒）、`id_sim` 分布直方图。
5. 总大小 **< 5 MB**（图片压缩）；中文排版正常，无乱码。
6. **两个评测协议都要展示**，并注明"held-out 协议身份与训练集无交集"。

## 验收标准

- [ ] `index.html` 双击可开，无控制台报错
- [ ] 找一个**没参与项目的人**打开，**3 分钟内能说出方法与结论**（说不出来算没完成）
- [ ] 页面上每个数字都能在 `comparison.json` / `metrics.json` 里找到
- [ ] 图表由脚本生成、可重跑
- [ ] 如实包含"TAR 未达显著"的说明
- [ ] GitHub Pages 可访问

## 依赖

数据已全部就绪，**不需要跑任何 GPU 任务**。""",
    },
    {
        "title": "[B-2] 评测器补强：边界单元测试 + 跨机（CPU）复现验证",
        "labels": ["类型:评测", "优先级:P0", "难度:中等"],
        "body": """## 背景

评测器是我们踩坑最多的地方，已经修掉几个**不报错但会把指标算错**的 bug。
你是"尺子"的负责人，需要把这些坑用测试钉死。

## 任务

### 1. 边界单元测试（必须覆盖）

* **`tar_at_far` 的参数顺序是 `(labels, scores, far)`** —— 写反不会报错，
  只会让 `labels==1` 去筛"分数恰好等于 1.0"的样本，**静默返回 `nan`**。
  **必须专门加一条防回归测试。**
* **并列分数**：全常数分数时 FAR 不能变成 1.0（曾用分位数实现，错在这里）。
* **`best_threshold`**：不能把假正例当真正例统计（曾导致 accuracy 固定为 0.5）。
* 空集、单样本、`FAR` 大于可用 impostor 数。
* 注意：**每折只有 300 个非同类对，FAR=1e-3 在单折内不可解**（期望 impostor 数 0.3），
  所以按折算 TAR 会得到 `nan` —— 测试里应当断言这一点，而不是当成 bug。

### 2. 跨机复现验证

在你的机器（**CPU，C 档**）上：

```bash
python scripts/evaluate.py --exp-id e0-buffalo_l-lfw --cpu
```

**验收**：LFW 官方协议 accuracy 与仓库里记录的 **0.9985 一致到小数点后 3 位**。
允许慢，**不允许不一样**。不一致就说明评测器有环境依赖，必须查清。

### 3. manifest 契约测试

故意塞一条坏记录（路径不存在 / 缺字段 / 身份不在类别表），
**测试必须失败**，不能静默跳过。

## 验收标准

- [ ] 上列边界测试全部存在且通过
- [ ] `python -m pytest tests/ -q` 全绿
- [ ] CPU 复现 E0 的数字与记录一致（附上你机器上的输出）
- [ ] 新增测试有注释说明"为什么这条要防"（参考 `src/aigcfr/eval/verify.py` 的说明风格）""",
    },
    {
        "title": "[B-3] 报告排版 + 局限章节核对 + 文献补充",
        "labels": ["类型:文档", "优先级:P0", "难度:入门"],
        "body": """## 背景

A 已完成 `docs/REPORT.md`（方法 / 结果 / 结论 / 局限 / 复现）。
你的任务是**排版、核对与文献**，不是重写。

## 任务

1. **事实核对**：报告里每一个数字，逐个到 `results/runs/*/metrics.json` 与
   `results/runs/w4-e4-comparison/comparison.json` 里核对。
   **发现对不上的地方直接开 Issue。**
2. **局限章节必须包含**（报告里已写，请确认无遗漏、表述准确）：
   * 训练/测试泄漏不可避免（官方 6000 对含全部 96 个训练身份）
   * **TAR@FAR=1e-3 未达显著**，且种子方差大于效应量
   * 合成占比只测了 17.4% / 11.1%
   * 只测了 iresnet18 一个架构
   * 数据规模小（96 身份 / 3590 张）—— 自研模型 TAR 0.33 vs 零样本 buffalo_l 0.9989
3. **文献补充 ≥8 篇**（格式统一）：ArcFace、IP-Adapter / IP-Adapter-FaceID、
   合成数据用于识别的既有工作、合成人脸可检测性。
4. **图表整理**：确保报告引用的图都由脚本生成、路径正确。
5. 导出 PDF / DOCX 版本。

## 验收标准

- [ ] 报告里每个数字都能溯源到结果文件
- [ ] 局限章节 5 条齐全
- [ ] 文献 ≥8 篇、格式统一
- [ ] 有 PDF/DOCX 导出""",
    },
]

# ---------------------------------------------------------------- API


def api(path: str, method: str = "GET", body: dict | None = None):
    token = os.environ.get("GH_TOKEN")
    if not token:
        raise SystemExit("!! 需要 GH_TOKEN 环境变量")
    req = urllib.request.Request(
        f"https://api.github.com/repos/{REPO}{path}",
        data=json.dumps(body).encode("utf-8") if body else None,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "aigcfr-sync",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read().decode("utf-8")
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{method} {path} -> {e.code} {e.read().decode('utf-8')[:200]}") from e


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only-issues", action="store_true", help="只同步 issue，不建新卡片")
    ap.add_argument("--only-new", action="store_true", help="只建新卡片")
    args = ap.parse_args()

    if args.dry_run:
        print("将同步的 issue：")
        for n, u in sorted(UPDATES.items()):
            print(f"  #{n}  {'评论 + 关闭' if u.get('close') else '仅评论'}")
        print("\n将新建：")
        for i in NEW_ISSUES:
            print(f"  {i['title']}")
        return 0

    if not args.only_new:
        for n, u in sorted(UPDATES.items()):
            try:
                api(f"/issues/{n}/comments", "POST", {"body": u["comment"]})
                if u.get("close"):
                    api(f"/issues/{n}", "PATCH", {"state": "closed", "state_reason": "completed"})
                print(f"  #{n} {'评论+关闭' if u.get('close') else '评论'}")
            except RuntimeError as e:
                print(f"  #{n} 失败: {e}")

    if not args.only_issues:
        for iss in NEW_ISSUES:
            try:
                r = api("/issues", "POST", iss)
                print(f"  新建 #{r['number']}  {iss['title']}")
            except RuntimeError as e:
                print(f"  新建失败 {iss['title']}: {e}")
    print("完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
