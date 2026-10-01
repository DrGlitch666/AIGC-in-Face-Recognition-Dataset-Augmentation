# W3 合成数据生成：最终配方与实测结论

> 本文固化 W3（数据生成）阶段的结果。所有数字都来自本项目实测，
> 原始数据在 `results/runs/w3-generator-comparison/`。
> 对应任务卡 **#10（生成）**、**#11（筛选）**。

---

## 1. 最终配方（已设为 `scripts/generate_plusv2.py` 的默认）

| 项 | 值 | 说明 |
|---|---|---|
| 适配器 | `ip-adapter-faceid-plusv2_sd15.bin` | **不是** base FaceID |
| **`shortcut`** | **`True`**（`s_scale=1.0`） | 见 §3.1，**最关键** |
| CLIP 层 | **`hidden_states[-2]`** | 不是 `last_hidden_state` |
| 负例 CLIP | **对全零图像跑编码器** | 不是全零张量 |
| CLIP 输入图 | **`face_align.norm_crop(img, kps, image_size=224)`** | 224 的人脸对齐裁剪 |
| ID 嵌入 | `buffalo_l` 的 `normed_embedding`，**多张参考图平均** | 见 §3.4 |
| ID 模型 | **`buffalo_l`** | **不是** antelopev2，见 §3.2 |
| 底模 | SD1.5（原版） | Realistic Vision 更差，见 §3.3 |
| VAE | `sd-vae-ft-mse` | 官方配方 |
| 调度器 | DDIM（`beta_start 0.00085 / beta_end 0.012 / scaled_linear / clip_sample=False / set_alpha_to_one=False / steps_offset=1`） | 官方配方 |
| 步数 / guidance / IP scale | 30 / 7.5 / 1.0 | 官方配方，实测最优 |
| 依赖 | **必须装 `peft`** | 见 §3.5 |

### 运行命令

```bash
# 生成（默认即为最优配方）
python scripts/generate_plusv2.py --per-identity 6

# 筛选（按 id_sim 阈值写入 accepted / rejected）
python scripts/filter_synth.py --exp-id syn-plusv2 --threshold 0.4
```

---

## 2. 实测效果（4 身份 × 3 seed，配对比较：同身份、同种子、同评分器）

| 方案 | 可评分 | id_sim 均值 | 中位数 | **最小值** | 标准差 | **> 0.4 的比例** |
|---|---|---|---|---|---|---|
| base FaceID | 11/12 | +0.3891 | +0.3685 | +0.2035 | 0.1121 | 5/11 |
| plusv2（官方配方） | 12/12 | +0.5138 | +0.5245 | +0.4320 | 0.0496 | 12/12 |
| **plusv2 + 多参考平均（最终）** | **12/12** | **+0.6084** | +0.6057 | **+0.5273** | 0.0530 | **12/12** |
| plusv2 + Realistic Vision V6 | 12/12 | +0.3999 | +0.3987 | +0.2836 | 0.0496 | 5/12 |
| plusv2 + IP scale 1.2 | 12/12 | +0.5223 | +0.5239 | +0.4248 | 0.0514 | 12/12 |
| plusv2 + guidance 5.0 | 12/12 | +0.5095 | +0.5178 | +0.3947 | 0.0719 | 10/12 |

### 参照点（`id_sim` = 与本人真实图中心的余弦）

| | 值 |
|---|---|
| **同一个人**的两张真实照片 | **0.745** |
| **我们的合成图（最终配方）** | **0.6084**（最高单张 0.7045） |
| 经验门槛"像同一个人" | ~0.40 |
| **不同的人** | **< 0.25** |

**结论**：最终配方把身份相似度从 0.3891 提到 **0.6084（+56%）**，
**最差样本从 0.2035 提到 0.5273** —— 对训练数据而言，
**最差样本的质量比均值更关键**，因为差样本等于标签噪声。

---

## 3. 排查过程中确认的坑（按影响排序）

### 3.1 ⭐ `shortcut` 必须为 True（最致命）

官方 `ProjPlusModel.forward`：

```python
out = self.perceiver_resampler(x, clip_embeds)
if shortcut:
    out = x + scale * out          # ← 残差，保住 ID 分支
return out
```

diffusers 里对应 `if self.shortcut: out = id_embeds + self.shortcut_scale * out`，
而它默认是 `False` —— **不设它，感知器的输出会直接覆盖掉 ID 分支**，
身份信号被完全丢弃，只剩 CLIP 的外观信息。

**症状**：`id_sim` 掉到 0.08，与"适配器根本没起作用"（scale=0 时约 0.03~0.08）无法区分。
官方 README 的调用是 `shortcut=v2`（plusv2 时为 True）。

### 3.2 ⭐ ID 嵌入必须用 `buffalo_l`，不能用 `antelopev2`

网上文档（IP-Adapter-FaceID README）说用 antelopev2，**照做会得到完全不像本人的图**。
交叉验证（同一 seed、同一参考图、两个嵌入模型 × 两个度量空间）：

| 条件嵌入 | 度量空间 | id_sim |
|---|---|---|
| buffalo_l | buffalo_l | +0.4002 |
| **buffalo_l** | **antelopev2** | **+0.4464** ← 跨空间反而更高，排除"度量偏差" |
| antelopev2 | buffalo_l | -0.1352 |
| antelopev2 | antelopev2 | -0.0754 |

两个模型对真实图的嵌入都正常（同身份自检 0.755 / 0.768），
所以是**这个适配器只认 buffalo_l 的嵌入分布**，与嵌入质量无关。
官方 README 用的正是 `FaceAnalysis(name="buffalo_l")`。

### 3.3 `Realistic Vision` 底模不兼容

官方 demo 用的是 `Realistic_Vision_V4.0_noVAE`，我们试了 V6.0 —— **在 base FaceID 和
plusv2 两种适配器下都明显更差**（0.0428 / 0.3999）。放弃。
底模保持原版 SD1.5。

### 3.4 多参考图平均嵌入（+18%）

同一身份不同真实照片的嵌入相似度只有 ~0.75，说明**单张参考本身带噪声**。
取前 N 张求平均再归一化，把均值从 0.5138 提到 0.6084、最差样本从 0.4320 提到 0.5273，
自检相似度也从 0.72~0.75 升到 0.79~0.87。

### 3.5 `peft` 不能漏

缺它时 diffusers 只打印一行
`PEFT backend is required to load these weights.`，然后**静默跳过 FaceID 的配套 LoRA**，
表现为"生成的人不像本人"。已写进 `requirements.txt`。

### 3.6 检测尺寸：生成图/对齐图要用 320，不能用 640

```
 det_size   112x112 对齐图   512x512 生成图
   160           1 张            1 张
   320           1 张            1 张     <- 用这个
   480           0 张            1 张
   640           0 张            0 张     <- 一张都检不出
```

insightface 把图像**长边缩放到 det_size**：对 112×112 的对齐图，640 意味着放大 5.7 倍
（人脸占满整个输入框），det_10g 反而失效。**这曾导致"可评分 0/N"**，
并被误判成"生成的图崩坏"。

> ⚠️ **E0/E1 评测仍然用 640**：那边输入是 250×250 的 LFW 原图（人脸约占 40%），
> 640 下检出正常且拿到了 0.9985 的准确率。**不要改那边**，改了会破坏可比性。

### 3.7 `det_size=640` 会带来幸存者偏差（测量教训）

在 640 下只有"碰巧能检出"的图会被评分，而**能检出的恰恰是更通用、更不像本人的那些**。
据此得出的"改进"结论全部不可信。**换到 320 后，同一批图的分数才单调、可解释。**
引申规则：**指标口径变了，之前所有对比都要重测**。

### 3.8 CFG 的两半：[负例(全零), 正例]

diffusers 的 `prepare_ip_adapter_image_embeds` 对传入张量做 `.chunk(2)`，
**第一半当负例、第二半当正例**。所以 batch=1 时形状是 `(2, 1, D)`，
**负例必须是全零**（官方 `encode_image` 即 `uncond = zeros_like(image_embeds)`）。
两边都给同一个人脸嵌入会让 CFG 的 `(cond - uncond)` **抵消掉面部条件**。

### 3.9 `subfolder` 与 `preprocessor_config.json`

* `pipe.load_ip_adapter(..., image_encoder_folder="image_encoder")` 时
  **`subfolder` 不能是 `None`** —— diffusers 会做 `Path(subfolder, image_encoder_folder)`，
  `None` 直接 `TypeError`。传 `""` 即可。
* `h94/IP-Adapter` 的 `models/image_encoder/` **没有 `preprocessor_config.json`**，
  `CLIPImageProcessor.from_pretrained` 会失败。CLIP 的图像预处理是公开固定常量，
  代码里直接构造（与 `laion/CLIP-ViT-H-14` 一致）。
* plusv2 的权重在 **`h94/IP-Adapter-FaceID`**，CLIP 编码器在 **`h94/IP-Adapter`**
  —— **两个不同仓库**，指向同一个会 404。

### 3.10 manifest 的 `path` 约定

`path` 是**相对 `paths.data_root`**，或**绝对路径**。
生成脚本必须先把 `--out` **`.resolve()`** 再算相对路径 ——
否则 `relative_to(data_root)` 抛异常、退化成"相对仓库根"的相对路径，
而消费端按"相对 data_root"解析，**好图会全部被判成 `no_face`**（真实踩过）。
`filter_synth.py` 已加回退（data_root 找不到就试仓库根）。

### 3.11 `diffusers` 对 plusv2 是**未完成**实现

`IPAdapterFaceIDPlusImageProjection.forward(id_embeds)` 内部用 `self.clip_embeds`，
而整个 diffusers 包里**只有 `__init__` 里的 `self.clip_embeds = None`**，
没有任何地方赋值。更绕的是：管线 `encode_image(..., output_hidden_state=True)`
明明算出了 CLIP 隐状态，却把它当作 `id_embeds` **传进 forward**，而 forward 完全不用该参数。

**所以必须自己算、自己塞**：

```python
proj.clip_embeds = torch.cat([neg_clip, pos_clip], dim=0).unsqueeze(1)   # (2, 1, 257, 1280)
proj.shortcut = True
proj.shortcut_scale = 1.0
```

形状必须是 **4 维**：forward 里 `clip_embeds.reshape(-1, shape[2], shape[3])` 要求 ≥4 维；
最后一维 **1280** = CLIP ViT-H 隐藏维；**257** = 256 patch + 1 CLS；
第 0 维 **2** = CFG 的 [负例, 正例]。

---

## 4. 复现方式

```bash
# 1) 下载模型（约 2.5 GB）
python scripts/download_data.py --config configs/data/ipadapter_plusv2.yaml

# 2) 生成
python scripts/generate_plusv2.py --per-identity 6

# 3) 筛选
python scripts/filter_synth.py --exp-id syn-plusv2 --threshold 0.4
```

对照数据：`results/runs/w3-generator-comparison/`
* `comparison.json` —— 各方案汇总 + `final_recipe` 块
* `synth_scores_*.csv` —— 逐张 `id_sim`

> 生成是**确定性**的：种子 = `seed + crc32("<身份>_<序号>") % 100000`，
> 提示词按 `k % 4` 轮转。同一条命令必然产出同一批图。
