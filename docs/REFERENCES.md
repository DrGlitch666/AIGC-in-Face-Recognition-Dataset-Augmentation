# 预置参考资料（Seed References）

> 这份清单是**项目框架预置的起点**，用来让你不必从一堆搜索结果里自己筛选。
> 它不是 [issue #4](../docs/issues/04-background-survey.md) 要产出的 `docs/BACKGROUND.md`——那份需要你自己读、自己总结。
>
> **可靠性说明**：下列条目在框架搭建时做过核实（官方仓库 / 论文 / 数据集页面 / 法条），但**链接会失效、数字会过时**。凡是「需要你自己确认」的地方都已标注。数字请以你本机实测为准。

---

## 1. 训练与评测工具箱（选型结论）

| 工具 | 现状 | Windows + 8 GB 是否可用 |
|---|---|---|
| [insightface `recognition/arcface_torch`](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch) | 官方 PyTorch 训练实现，入口 `train_v2.py` | ✅ 纯 PyTorch/CUDA |
| insightface `recognition/arcface_mxnet` | 官方 README 自称「历史实现」，需 `mxnet-cu100` | ❌ MXNet 已停止维护 |
| [face.evoLVe.PyTorch](https://github.com/zhanglaplace/face.evoLVe.PyTorch) | MIT 代码，IR-50 + ArcFace，自带 LFW/CFP-FP/AgeDB/CALFW/CPLFW 评测 | ❌ README 写明只支持 Linux/macOS |
| [facenet-pytorch](https://pypi.org/project/facenet-pytorch/) | MTCNN + InceptionResnetV1，偏推理 | ✅ 但**没有可训练的 ArcFace 头**，不适合本项目 |

**⚠️ 重要更正**：网上大量教程引用的 `src/insightface/recognition` 目录与 `train_rec.py` **在当前 master 上已不存在**（该路径 404）。代码已移到 `recognition/` 下。不要照抄老教程的路径。

**8 GB 显存可行性**（实测口径，需自行验证）：iResNet-50 @112×112、batch 64~128 + AMP 可以跑；身份数（identities）比图片总数更影响显存——**身份数控制在 1 万以内时普通全连接头很小**，本项目 toy 规模（几十到几百身份）完全不是问题。

**RecordIO `.rec` 格式**：MXNet 时代的数据格式（`im2rec` 生成 `train.rec`/`train.idx`）。`faces_emore` 归档里同时带有现成的 `lfw.bin`/`cfp_fp.bin`/`agedb_30.bin` 评测包，这也是 `arcface_torch` 评测文件的来源。**本项目不强制使用 `.rec`**，用 manifest 直接读图即可（issue #3 的契约）。

---

## 2. 零样本基线：模型包与期望数字

模型包下载地址形如：`https://github.com/deepinsight/insightface/releases/download/model-zoo/<name>.zip`

| 模型包 | 识别网络 | 大小 |
|---|---|---|
| `buffalo_l`（默认） | ResNet50 @ WebFace600K | 275 MB |
| `antelopev2` | ResNet100 @ Glint360K | 344 MB |
| `buffalo_s` / `buffalo_sc` | MobileFaceNet @ WebFace600K | 122 MB / 14.3 MB |

**⚠️ 许可**：insightface 的**代码**是 MIT，但**模型包仅限非商业研究用途**（见其 `model_zoo/README`）。写进 `docs/ETHICS.md`（issue #18）。

**`buffalo_l` 官方报告精度**（用于 issue #5 的自检对照）：

| 基准 | LFW | CFP-FP | AgeDB-30 | IJB-C @FAR=1e-4 |
|---|---|---|---|---|
| buffalo_l | 99.83 | 99.33 | 98.23 | 97.25 |

> ⚠️ 官方 README 里另有一组按人群分组的 IFRT 风格数字（African / Caucasian / South Asian / East Asian），**那不是 RFW**，引用时不要标成 RFW。

**运行时注意**：`pip install insightface` 后默认装 `onnxruntime`；要用 GPU 请换 `onnxruntime-gpu`，并且**不要两个同时装**（会冲突）。`FaceAnalysis()` 的 provider 顺序是 CoreML → CUDA → CPU，所以务必打印 providers 确认真的在用 GPU。

---

## 3. 评测数据集：可用性与获取难度（**这一节直接影响计划**）

| 数据集 | 规模 | 协议 / 指标 | 2026 年还能免费直接下载吗 |
|---|---|---|---|
| **LFW** | 13,233 图 / 5,749 人 | 6,000 对，**10 折**，Accuracy | ⚠️ 官方站点不稳定；torchvision 已不再自动下载。**准备 Kaggle 等镜像**，并预留时间 |
| **CFP-FP** | 7,000 图 / 7,000 对 | 10 折，Accuracy | ✅ [cfpw.io](http://www.cfpw.io/) 可直接下载 |
| **AgeDB-30** | 12,240 图 / 6,000 对 | 10 折，Accuracy | ⚠️ 压缩包**有密码，需邮件向作者索取**；仅限非商业研究 |
| **CALFW** | 6,000 对 | 10 折，Accuracy | ✅ Google Drive / 百度网盘 |
| **CPLFW** | 6,000 对 | 10 折，Accuracy | ✅ Google Drive / 百度网盘 |
| **RFW**（4 人群） | 每组 6,000 对 | 分组 Accuracy + 均值/标准差 | ❌ **需邮件申请**；评测代码免费 |
| **TinyFace** | 169,403 张低分辨率图 / 5,139 人 | **1:N** → Rank-1 / CMC | ✅ Google Drive，约 148 MB |
| **IJB-C** | 138k 图 + 11k 视频 | TAR@FAR=1e-4/1e-5，TPIR@FPIR | ❌ **NIST 已于 2023-03-14 停止 IJB-A/B/C 的分发，无法获取** |

**结论（已写进 issue #3 的降级方案）**：
- 主线用 **LFW + CFP-FP**（最容易拿到），CALFW/CPLFW 作为可选。
- **AgeDB / RFW 都要发邮件申请**，把它们当「加分项」而不是必经之路。
- **IJB-C 不要写进计划**——拿不到。
- ⚠️ **LFW / CALFW / CPLFW 复用同一批身份**，不是三个独立测试；引用时要说明这一点。
- LFW 系列在好模型上早已饱和（99.8%+），**只看 LFW 会掩盖问题**——所以 issue #14 建议增加 TinyFace（1:N Rank-1）作为不易饱和的指标。

**可从公开渠道下载的合成数据**：[DigiFace-1M](https://microsoft.github.io/DigiFace1M/)（122 万张合成图 / 11 万身份）——这是 issue #10 的重要兜底方案：**不用自己生成也能做「合成数据有没有用」的实验**。

---

## 4. 合成人脸检测器与画质指标

### 检测器（用于 issue #12）

| 工具 | 方法 | 说明 |
|---|---|---|
| [UniversalFakeDetect](https://github.com/Yuheng-Li/UniversalFakeDetect)（CVPR'23） | 冻结 CLIP ViT-L/14 + 线性探针 | 跨生成器泛化的经典基线，[arXiv:2302.10174](https://arxiv.org/abs/2302.10174) |
| [NPR](https://github.com/chuangchuangtan/NPR-DeepfakeDetection)（CVPR'24） | 上采样伪影 | 不同基准分数差异大，注意评测设置 |
| [Synthbuster](https://github.com/qbammey/synthbuster) | 傅里叶伪影 + 随机森林（**非深度网络**） | 方法正交，适合作为第二个检测器 |
| [CNNDetection](https://github.com/PeterWang512/CNNDetection) | 经典基线 | 事实上的评测协议来源（ProGAN/StyleGAN 系） |
| [ClipBased-SyntheticImageDetection](https://github.com/grip-unina/ClipBased-SyntheticImageDetection)（Apache-2.0） | CLIP 特征 | 许可宽松 |
| [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench)（**CC BY-NC 4.0**） | 检测框架（v2 含 36 个检测器） | 注意：它的范围是**被篡改的脸**，不是从零生成的假脸 |

**⚠️ HuggingFace 上的「AI 图像检测器」**（如 Organika/sdxl-detector、umm-maybe/AI-image-detector 等）：准确率是**自报的**，训练数据与适用范围往往不明确，其模型卡甚至写明「不是 deepfake 检测器」。**可以用来做交叉参考，但不要用它们的数字支撑科学结论。**

### 画质 / 分布指标（用于 issue #11）

| 指标 | 方向 | 需要参考图吗 | 库 |
|---|---|---|---|
| FID | 越低越好 | 不需要（但要有真实集，经验值 ≥1 万张） | [clean-fid](https://github.com/GaParmar/clean-fid)（MIT）、pytorch-fid |
| KID | 越低越好，**无偏** | 不需要 | torchmetrics / clean-fid |
| NIQE / BRISQUE | 越低越好 | **不需要** | [pyiqa](https://github.com/chaofengc/IQA-PyTorch)、OpenCV contrib |
| Laplacian 方差 | 越高越清晰 | 不需要 | OpenCV 一行代码 |
| LPIPS / PSNR / SSIM | — | **需要** | pyiqa / torchmetrics |

**⚠️ 两个必须知道的坑**：
1. **FID 对实现极其敏感**：同一组图，仅改变缩放滤波器就能得到 ≥6 与 ≤0.75 的差别（[clean-fid 论文, arXiv:2104.11222](https://arxiv.org/abs/2104.11222)）。所以**两个实验组必须用同一份实现重跑**，并且报告 FID 时要写清用哪个库。
2. **pyiqa 的许可是 PolyForm Noncommercial 1.0.0**（非商业）。学术研究没问题，但如果这个项目以后要商用，需要换实现——写进 `docs/ETHICS.md`。

---

## 5. 生成方法一览（用于 issue #9 / #10 / #13）

**身份条件生成（无条件 → 身份保持）**

| 方法 | 范式 | 说明 |
|---|---|---|
| [Arc2Face](https://github.com/foivospar/Arc2Face)（ECCV'24 Oral） | 扩散，**只以 ArcFace embedding 为条件**（无文本） | 概念上最贴合本项目；[arXiv:2403.11641](https://arxiv.org/abs/2403.11641)。⚠️ 训练用了 WebFace42M，**生成器的许可不随生成图转移给你** |
| [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter)（含 FaceID 变体） | SD1.5/SDXL + 适配器 | 显存最友好，**建议作为主选** |
| [InstantID](https://github.com/instantX-research/InstantID) | SDXL + IdentityNet | 单图身份保持效果好，但 SDXL 路线显存要求更高 |
| [PhotoMaker](https://arxiv.org/abs/2312.04461) / [PuLID](https://arxiv.org/abs/2404.16022) / [IDAdapter](https://arxiv.org/abs/2403.13535) | 个性化生成 | 备选方案 |
| [DCFace](https://github.com/mk-minchul/dcface)（CVPR'23） | 双条件扩散（ID + 风格） | 偏「造数据集」，[arXiv:2304.07060](https://arxiv.org/abs/2304.07060) |
| [IDiff-Face](https://arxiv.org/abs/2308.04995)（ICCV'23） | 身份条件隐扩散 | LFW 98.00%（真实数据上限 99.82%） |
| [ID³](https://arxiv.org/abs/2409.17576)（NeurIPS'24） | ID 保持 + 多样化 | 针对「像但不多样」的问题 |

---

### 可下载的现成合成人脸数据集（**B 轨的核心资源**）

> 这是「不训练生成器、不占显存」就能做实验的路径。**建议先跑通这条，再考虑本地生成。**

| 数据集 | 规模（身份 × 图） | 下载 | 许可 | 需要申请？ |
|---|---|---|---|---|
| **DigiFace-1M**（WACV'23） | 10K×72 + 100K×5 ≈ **122 万张** | [官方 8 个直链 zip](https://github.com/microsoft/DigiFace1M) | R-UDA 非商业 | ❌ 不需要 |
| DCFace（CVPR'23） | 0.5M / 1.2M | [GDrive](https://github.com/mk-minchul/dcface) | 未声明（需自查） | ❌ |
| IDiff-Face（ICCV'23） | 10K × 50 = 0.5M | [GDrive + 表单](https://github.com/fdbtrs/IDiff-Face) | CC BY-NC-SA 4.0 | ✅ 需表单 |
| CemiFace（NeurIPS'24） | 0.5M / 1.0M / 1.2M | [OneDrive](https://github.com/szlbiubiubiu/CemiFace) | 未声明 | ❌ |
| Digi2Real（WACV'25） | 20K 身份，7.7 GB | [Zenodo](https://zenodo.org/records/14066541) | 非商业研究 | ❌ |
| HSFace/Vec2Face（ICLR'25） | 10K→300K 身份 | [HuggingFace](https://huggingface.co/datasets/BooBooWu/Vec2Face) | 未声明 | ❌ |
| HyperFace（ICLR'25） | ≤50K 身份 / 320 万张 | [Zenodo](https://zenodo.org/records/15087238) | CC BY-SA 4.0 | ❌ |
| BIF-Face（ICML'25） | 10K（17.4 GB）/ 50K（87.4 GB） | [Zenodo](https://zenodo.org/records/19878174) | CC BY-NC-SA 4.0 | ❌ |
| SFace（IJCB'22） | 约 0.5~1.9M | [仓库](https://github.com/fdbtrs/SFace-Privacy-friendly-and-Accurate-Face-Recognition-using-Synthetic-Data) | CC BY-NC-SA 4.0 | ✅ 需表单 |

**⚠️ 三个下载陷阱**：HuggingFace 上 `FadiBoutros/IDiff-Face` 是**空仓库**；**ID³ 的数据集从未发布**（其仓库只是占位）；**FRCSyn-2024 提供的 DCFace/GANDiffFace 只发给注册的 CodaLab 参赛者**。
**⚠️ SFHQ（StyleGAN2，425K 张，CC0）虽然完全开放，但没有身份标签**，不能当作 FR 的类别使用。

### 合成数据能达到什么水平（5 个基准的平均：LFW/CFP-FP/CPLFW/AgeDB/CALFW）

| 训练数据 | 平均准确率 |
|---|---|
| SynFace（2021） | 74.75 |
| SFace（2022） | 77.71 |
| DigiFace-1M（2023） | 83.45 |
| IDiff-Face（2023） | 88.20 |
| DCFace（2023） | 89.56 |
| Arc2Face（2024） | 91.73 |
| HSFace-10K / 300K（2025） | 92.00 / 93.52 |
| **纯真实数据 CASIA-WebFace（参照天花板）** | **94.79** |

> **怎么用这张表**：① 它告诉你**差距正在缩小但尚未消失**（最好约 93.5 vs 真实 94.8）；② 差距**主要集中在姿态困难的两个基准（CFP-FP / CPLFW）**上——这正是你项目里「分桶评测」要盯的地方；③ 本项目规模远小于这些工作，绝对值会低很多，**我们只比内部相对差异**。

## 6. 核心论文清单（issue #4 的起点）

### 综述与竞赛（先读这几篇建立全局观）
- *Synthetic data for face recognition: Current state and future prospects*（Image & Vision Computing 2023）——[链接](https://www.sciencedirect.com/science/article/abs/pii/S0262885623000628)
- **SDFR Competition**（FG 2024）：合成数据人脸识别的竞赛报告，含 7 个基准与公平性评估——[arXiv:2404.04580](https://arxiv.org/abs/2404.04580)
- **FRCSyn Challenge**（CVPRW 2024）：定义了「训练两次做配对比较」的标准协议——[CVF](https://openaccess.thecvf.com/content/CVPR2024W/FRCSyn/html/Deandres-Tame_Second_Edition_FRCSyn_Challenge_at_CVPR_2024_Face_Recognition_Challenge_CVPRW_2024_paper.html) · [arXiv:2404.10378](http://arxiv.org/abs/2404.10378)

### 合成人脸数据集
- **SynFace**（ICCV'21）：首次系统研究合成-真实差距，指出**根因是类内变化不足 + 域差距**——[arXiv:2108.07960](https://arxiv.org/abs/2108.07960)
- **DigiFace-1M**（WACV'23）：100 万张渲染人脸，**激进的数据增强是缩小域差距的最大杠杆**——[arXiv:2210.02579](https://arxiv.org/abs/2210.02579)
- **SFace**（2022）：类条件 GAN + 隐私审计——[arXiv:2206.10520](https://arxiv.org/abs/2206.10520)
- **DCFace**（CVPR'23）—[arXiv:2304.07060](https://arxiv.org/abs/2304.07060)｜**IDiff-Face**（ICCV'23）—[arXiv:2308.04995](https://arxiv.org/abs/2308.04995)
- **ID³**（NeurIPS'24）—[arXiv:2409.17576](https://arxiv.org/abs/2409.17576)｜**CemiFace**（NeurIPS'24，**中等相似度的样本反而更有训练价值**）—[arXiv:2409.18876](https://arxiv.org/abs/2409.18876)
- **Digi2Real**（WACV'25）—[arXiv:2411.02188](https://arxiv.org/abs/2411.02188)｜**VariFace**（公平性+多样性引导）—[arXiv:2412.06235](https://arxiv.org/abs/2412.06235)
- **Vec2Face+/VFace**（2025）：首批在某些测试集上超过 CASIA-WebFace 的合成数据集，但**合成数据训练的模型更有偏**、孪生人脸验证几乎全失败——[arXiv:2507.17192](https://arxiv.org/abs/2507.17192)

### 域差距、混合比例、人群偏向（**写结论时必读**）
- **生成器会继承其训练数据的人口统计分布**（性别/种族/年龄/头部姿态）：Huber et al., WACV 2024 — [arXiv:2311.03970](https://arxiv.org/abs/2311.03970)
- GAN 过采样「20~29 岁白人面孔」，用合成数据微调会造成按种族的差异化影响——[arXiv:2208.13061](https://arxiv.org/abs/2208.13061)
- 合成数据流程对 **Black 与 East Asian** 子群准确率更低（48k 对，55.5 万人工标注）——[arXiv:2308.05441](https://arxiv.org/abs/2308.05441)
- **只在合成图上做偏见审计会得出误导性结论**（同一指标在真实图上结论反转）——[arXiv:2608.23593](https://arxiv.org/abs/2608.23593)
- **合成数据集可能泄漏真实训练样本**（成员推断攻击）——[arXiv:2410.24015](https://arxiv.org/abs/2410.24015) · [arXiv:2607.29144](https://arxiv.org/abs/2607.29144)
- **身份条件扩散模型普遍「类内变化不足」**（IDperturb, CVPR 2026 明确指出这一点，并指出 DCFace 的表情熵与姿态变化偏低）——[CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Boutros_IDperturb_Enhancing_Variation_in_Synthetic_Face_Generation_via_Angular_Perturbations_CVPR_2026_paper.html)
- **混合使用的效果与公平性代价**：扩散生成的合成数据**无论单独使用还是与部分真实数据混合都有帮助**，但**公平性分数「未改善甚至变差」**——[arXiv:2409.02867](https://arxiv.org/abs/2409.02867)。这条对你的 RQ4 极重要：**性能提升与公平性改善是两件事，要分开报。**
- **混合比例的实测证据，以及一个空白点**（对你的 RQ1/E5 很关键）：SynFace 的 LFW 随真实数据增加而上升——91.97（纯合成 50 万）→ 95.05（+2 万真实）→ **97.65（+4 万真实，真实占比仅约 7.4%）**；Boutros et al.（Image & Vision Computing 2023）报告 50 万合成 + 4 万真实把 DigiFace-1M 从 88.07 提到 **99.05**（纯真实 CASIA 50 万为 99.55）。**但系统的「10%/25%/50% 比例扫描」在文献中找不到直接对应物**——这正是你的 E5 可以填的空。注意这些数字来自比本项目大 3 个数量级的数据规模，**趋势可参考，绝对值不可比**。
- **身份泄漏的定量证据（最有分量的可引用风险）**：对 6 个主流合成数据集（DCFace、IDiff-Face、GANDiffFace、IDNet、SFace 等）做成员推断攻击，发现**几乎所有检索到的配对相似度都超过 IJB-C 上 FAR=0.01% 的匹配阈值**，且 DCFace 的阈值清洗「不足以防止身份泄漏」（Otroshi Shahreza & Marcel, NeurIPS 2024）——[arXiv:2410.24015](https://arxiv.org/abs/2410.24015)。**所以「合成 = 隐私安全」是不成立的假设**，写进你的伦理文档。
- 生成器会**继承其训练数据的人口统计分布**：[Huber et al., WACV 2024, arXiv:2311.03970](https://arxiv.org/abs/2311.03970)
- **Vec2Face 曾在 2025-04 重新上传数据集**，原因是早期版本「存在身份泄漏」——引用现成合成数据集时要留意版本与勘误。[仓库](https://github.com/HaiyuWu/Vec2Face)

---

## 7. 评测协议的实现细节（**最容易做错的地方**）

1. **对齐**：5 个关键点 → Umeyama 相似变换 → 112×112 的 ArcFace 模板
   `[(38.2946,51.6963),(73.5318,51.5014),(56.0252,71.7366),(41.5493,92.3655),(70.7299,92.2041)]`
   （参考 insightface 的 `norm_crop`；128×128 需按比例缩放并平移 x）
   **真实图与合成图必须走完全相同的检测+对齐流程。**
2. **评测循环**（insightface `verification.py` 的标准做法）：
   - 112×112 输入，`KFold(n_splits=10, shuffle=False)`
   - 在 **L2 归一化**后的 embedding 上算距离（常用平方 L2 距离）
   - **阈值在每折的训练半上选，在测试半上测**
   - **flip test**：原图与翻转图的 embedding 相加后重新归一化，报告为 "Accuracy-Flip"
3. **归一化**：余弦/内积比较前必须 L2 归一化，否则算的根本不是余弦相似度。
4. **泄漏**：LFW / CALFW / CPLFW 共享身份；MS1M/WebFace 与 LFW 也有重叠——用公开的重叠名单做审计，**永远不要在测试集上调参**。
5. **公平比较的两个预算口径**（issue #14 必须明确选一个并写清楚）：
   - **等总图片数**（real+synth 的总数 = 纯 real 的总数）→ 回答「合成图能否**替代**真实图」
   - **等真实图片数**（在 real 基础上**追加** synthetic）→ 回答「追加合成数据能否**增益**」
   两者问题不同，结论可能相反，**最好都报**。
6. **合成数据实验对「每身份图片数」和「身份总数」极其敏感**：Digi2Real 报告 IJB-C 在身份数从 2 万增到 5 万时从 75.80 掉到 37.17，而 **LFW 几乎不动**——所以**只跑 LFW 会掩盖问题**，务必加上不易饱和的指标（TinyFace Rank-1 或你自己的 1:N 闭集）。
7. **≥3 个随机种子，报告均值±标准差**；LFW 上 0.1% 的差异就是噪声。

---

## 8. 法律、许可与伦理（用于 issue #18）

### 数据许可：哪些**绝对不能**提交到仓库
| 数据集 | 关键条款 | 能否提交 |
|---|---|---|
| **LFW** | 研究用途可自由获取，需引用 | ✅ 可引用与链接（仍建议只放下载脚本） |
| **CelebA** | 明文**禁止**复制、发布、分发任何部分 | ❌ **绝对不能** |
| **MS-Celeb-1M / MS1M** | 微软 2019 年**已撤回**，无有效许可 | ❌ 绝对不能（镜像也不构成许可依据） |
| **VGGFace2** | 已撤回，事实上不可获取 | ❌ 绝对不能 |
| **CASIA-WebFace** | 点击同意，非商业研究，禁止再分发 | ❌ 不能 |
| **AgeDB** | 仅限非商业研究 | ❌ 不能 |
| **RFW** | 条款明文「不得复制或分发」 | ❌ 不能 |

> **对齐/裁剪后的图是衍生数据，继承相同限制。** 仓库里只放：代码、配置、随机种子、校验和、**你自己生成的合成图**、以及真实数据集的**下载脚本**。

### 法律要点
- **GDPR 第 9 条**：用于唯一识别自然人的生物特征数据属特殊类别，原则上禁止处理，除非有例外（明示同意，或第 89(1) 条下的科研目的+保障措施）— [原文](https://gdpr-info.eu/art-9-gdpr/)
- **欧盟 AI Act 第 5 条**：禁止基于生物特征推断种族/政治倾向/宗教/性取向的分类系统；禁止公共场所实时远程生物识别（2025-02-02 起生效）— [第 5 条](https://artificialintelligenceact.eu/article/5/) · [时间线](https://artificialintelligenceact.eu/implementation-timeline/)
- **美国伊利诺伊州 BIPA**（740 ILCS 14）：收集生物标识前需书面告知 + 书面同意，有**私人诉讼权** — [法条](https://www.ilga.gov/legislation/ilcs/fulltext?DocName=074000140K1)
- **中国《个人信息保护法》**：生物识别信息属敏感个人信息（第 28 条），需**单独同意**（第 29 条）— [官方英文版](https://en.spp.gov.cn/2021-12/29/c_948419_2.htm)
- **GitHub 合成媒体与 AI 工具政策**：禁止用于生成 CSAM、暴力极端主义宣传、非自愿私密影像等 — [政策原文](https://docs.github.com/en/site-policy/acceptable-use-policies/github-synthetic-media-and-ai-tools)

### 双刃剑风险（要写进伦理文档，不要说空话）
- 身份一致的生成器可直接用于制作**变形攻击（morphing）**样本，已有工作用 Arc2Face 造出护照级 morph — [arXiv:2602.16569](https://arxiv.org/abs/2602.16569)
- 合成数据集**可能泄漏**真实训练样本，所以「合成 = 隐私安全」是不安全的假设 — [arXiv:2410.24015](https://arxiv.org/abs/2410.24015)
- 因此：公开生成参数与来源、保留可检测性报告（issue #12）、不用真人照片做身份克隆、不做「无法被检测」的生成器

---

## 9. 怎么用这份清单

1. **issue #4**：不要直接抄这里的摘要——挑出 15 篇自己读，用你自己的话写 `docs/BACKGROUND.md`。这份清单只是选片单。
2. **遇到具体任务时**：回到对应章节（例如做 #11 时看第 4 节，做 #14 时看第 7 节第 5、6 条）。
3. **发现链接失效或数字不对**：直接在对应 Issue 下评论记录，并更新本文件——**这也是项目产出的一部分**。
