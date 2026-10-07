# B 机器 CPU E0 复现记录（#26）

CPU 上重新提取 LFW 人脸特征后，官方配对的 accuracy 与原记录均为 **0.998496**；在现有记录的六位小数精度下相同。

## 来源与命令

- 原始结果：[metrics.json](../results/runs/e0-buffalo_l-lfw/metrics.json)
- CPU 复现结果：[metrics.json](../results/reproductions/e0-buffalo_l-lfw-b-cpu/metrics.json)
- CPU 逐对分数：[pairs.csv](../results/reproductions/e0-buffalo_l-lfw-b-cpu/pairs.csv)
- 复现评测时间：`2026-10-07T16:41:31+08:00`
- 评测结果记录的 HEAD：`66181b6`；本次评测包含尚未提交的 #26 修复。
  `evaluate.py` 只记录 HEAD，没有记录工作区是否存在未提交改动；实际代码改动由本次 PR 提供。

```powershell
python scripts/evaluate.py --exp-id e0-buffalo_l-lfw --cpu --no-cache --out "results/reproductions/e0-buffalo_l-lfw-b-cpu"
```

`--no-cache` 关闭 embedding 缓存读写，重新执行特征提取。输出保存到独立的 `results/reproductions/` 目录，原 `results/runs/` 中的冻结结果保持原样。

## 环境与口径

| 项目 | CPU 复现记录 |
|---|---|
| 模型 | buffalo_l (w600k_r50, ResNet50 + ArcFace) |
| 识别执行器 | CPUExecutionProvider |
| 机器档位 | cpu |
| GPU | None |
| Python | 3.11.16 |
| PyTorch | 2.14.0+cpu |
| 检测尺寸 | 640 |
| 选脸策略 | center |
| 配置哈希 | b788b96322b2 |
| 成功 / 失败图片数 | 7690 / 11 |
| 特征提取耗时 | 6906.79 秒（约 115.1 分钟） |

## 对照结果

| 口径与指标 | 原记录 | CPU 复现 | CPU − 原记录 |
|---|---:|---:|---:|
| official / accuracy | 0.998496 | 0.998496 | +0.000000 |
| official / TAR@FAR=1e-3 | 0.996989 | 0.996989 | +0.000000 |
| official / actual FAR@1e-3 | 0.000668 | 0.000668 | +0.000000 |
| official / 可用配对 | 5983 | 5983 | +0 |
| official / 缺失配对 | 17 | 17 | +0 |
| filtered / accuracy | 0.998375 | 0.998375 | +0.000000 |
| filtered / TAR@FAR=1e-3 | 0.996570 | 0.996570 | +0.000000 |
| filtered / actual FAR@1e-3 | 0.000686 | 0.000686 | +0.000000 |
| filtered / 可用配对 | 5539 | 5539 | +0 |
| filtered / 缺失配对 | 14 | 14 | +0 |

相同的汇总成绩不代表检测输出、相似度或阈值逐位相同。例如官方 accuracy 阈值由 `0.249334` 变为 `0.249185`；本记录的验收范围是跨机器汇总指标一致性。

`official` 表示使用官方配对数据。当前 `summarize()` 的 accuracy 是在同一批评测对上选择最佳阈值的描述性指标，并非独立校准或留出折验证成绩。

## 数据完整性记录

来自本机 `data/raw/lfw/checksums.sha256`：

```text
055f7d9c632d7370e6fb4afc7468d40f970c34a80d4c6f50ffec63f5a8d536c0  lfw.tgz
ea42330c62c92989f9d7c03237ed5d591365e89b3e649747777b70e692dc1592  pairs.txt
```

下载配置未提供预期 SHA-256；这些是本机下载后的摘要，可用于跨机核对，不能单独作为已对照可信来源校验值的证明。

## 测试验证

```powershell
python -m pytest -q -o addopts=
```

pytest 退出码：`0`（0 表示通过）。
完整输出：[e0_cpu_reproduction.b.pytest.txt](e0_cpu_reproduction.b.pytest.txt)

评测器测试覆盖参数顺序、二元标签与分数合法性、并列分数、空集、缺少类别、最优阈值和 FAR 样本分辨率。按项目约定，300 个异人对时 TAR@FAR=1e-3 返回 NaN。
配置测试使用 pytest 临时配置文件，验证四层合并及私有路径不影响哈希，避免依赖 A 机器的 `configs/local.a.yaml`。
