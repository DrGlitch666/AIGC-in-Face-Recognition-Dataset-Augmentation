"""人脸 embedding 提取（检测 + 对齐 + 识别，由 insightface 的 buffalo_l 一体完成）。

## ⚠️⚠️ 本项目最重要的一条环境约束

**创建 ONNX Runtime 会话之前必须先 ``import torch``。**

torch 导入时会把自带的 CUDA DLL 目录注册进 Windows 的 DLL 搜索路径，ORT 才能加载
``CUDAExecutionProvider``。顺序反了**不会报错** —— 它只是静默退回 CPU，你只会觉得
"怎么这么慢"。（详见 docs/SETUP.md §9 的坑 K1 / K2）

因此本模块把 ``import torch`` 放在**模块顶部**，并在会话建好后调用
:func:`assert_provider` 做**显式断言**：如果机器上 CUDA 可用、却没拿到 CUDA provider，
直接抛错而不是继续跑。**把静默失败变成显式失败**，这是本项目的一条硬规矩。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch  # noqa: F401  —— ★ 必须在 onnxruntime 之前导入，不要删、不要挪到函数里

from insightface.app import FaceAnalysis

EMB_DIM = 512


@dataclass
class Embeddings:
    """一次提特征的完整结果。"""

    vectors: dict[str, np.ndarray] = field(default_factory=dict)   # 相对路径 -> (512,)
    failures: dict[str, str] = field(default_factory=dict)         # 相对路径 -> 失败原因
    provider: list[str] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def num_ok(self) -> int:
        return len(self.vectors)

    @property
    def num_failed(self) -> int:
        return len(self.failures)

    def failure_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for reason in self.failures.values():
            out[reason] = out.get(reason, 0) + 1
        return dict(sorted(out.items()))


def load_app(
    model_root: str | Path,
    providers: list[str] | None = None,
    det_size: int = 640,
    ctx_id: int = 0,
    allowed_modules: list[str] | None = None,
) -> FaceAnalysis:
    """加载 buffalo_l（默认 GPU 优先）。

    ⚠️ ``allowed_modules`` 默认只留 **detection + recognition**。
    buffalo_l 包里还有 ``1k3d68``（140 MB 的 3D 关键点）、``2d106det``、``genderage``，
    而 ``FaceAnalysis.get()`` 默认会**全部跑一遍** —— 人脸验证根本用不到，
    却会让吞吐量掉到 1/10（实测：1.2 张/秒 → 裁掉后快一个数量级）。
    """
    providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
    kwargs = {}
    if allowed_modules is not None:
        kwargs["allowed_modules"] = allowed_modules
    else:
        kwargs["allowed_modules"] = ["detection", "recognition"]
    app = FaceAnalysis(name="buffalo_l", root=str(model_root), providers=providers, **kwargs)
    app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))
    return app


def provider_report(app: FaceAnalysis) -> dict[str, list[str]]:
    """列出 **每一个子模型** 实际用的 provider。

    ⚠️ 只看识别模型是不够的：人脸检测（det_10g）在每个模型里都跑，而且往往是最慢的
    一步。如果检测落在 CPU 上，整条流水线会慢十倍，但识别那行日志看起来一切正常。
    """
    out: dict[str, list[str]] = {}
    for name, model in getattr(app, "models", {}).items():
        try:
            out[name] = list(model.session.get_providers())
        except Exception:  # noqa: BLE001
            out[name] = []
    return out


def assert_provider(app: FaceAnalysis, expect_cuda: bool | None = None) -> list[str]:
    """检查**所有**子模型都拿到了期望的 provider，否则报错而不是静默将就。

    ``expect_cuda=None`` 时自动判断：只要 torch 报告 CUDA 可用就要求用上 GPU。
    """
    report = provider_report(app)
    if not report.get("recognition"):
        raise RuntimeError("读不到 ONNX Runtime 的 provider（recognition 模型缺失？）")

    if expect_cuda is None:
        expect_cuda = bool(torch.cuda.is_available())

    actual = report["recognition"]
    if expect_cuda:
        on_cpu = [name for name, provs in report.items()
                  if not provs or provs[0] != "CUDAExecutionProvider"]
        if on_cpu:
            raise RuntimeError(
                f"CUDA 可用，但这些子模型没用上 GPU: {on_cpu}"
                f"（provider 明细 = {report}）。最常见的两个原因：\n"
                "  1) 本模块顶部的 `import torch` 被删掉或被挪到会话创建之后 —— 不要改；\n"
                "  2) onnxruntime-gpu 版本与 torch 的 CUDA 版本不匹配"
                "（本项目钉 onnxruntime-gpu==1.26.0，见 docs/SETUP.md 坑 K1）。\n"
                "如果你确实想用 CPU 跑，请显式传 expect_cuda=False。"
            )
    return actual


def warmup(app: FaceAnalysis, size: int = 250) -> float:
    """跑一次空推理，把 CUDA 上下文 / cuDNN 调优的一次性开销先付掉。

    实测（RTX 5060 Laptop）：不预热时首张图要 ~60s，整批吞吐看起来只有 1.5 张/秒；
    预热后稳定在 **~39 张/秒**（25 ms/张）。差别全在那个一次性开销上，不是模型慢。

    返回预热耗时（秒），建议记录进 metrics.json —— 否则吞吐数字会误导人。
    """
    import cv2  # noqa: PLC0415

    img = np.zeros((size, size, 3), dtype=np.uint8)
    t0 = time.time()
    app.get(img)
    return time.time() - t0


def extract(
    app: FaceAnalysis,
    root: str | Path,
    relpaths: list[str],
    cache_path: str | Path | None = None,
    verbose: bool = True,
    log_every: int = 500,
) -> Embeddings:
    """对一批图片提特征。

    * ``root``：图片根目录；``relpaths`` 是相对它的路径（例如 ``Colin_Powell/Colin_Powell_0001.jpg``）
    * ``cache_path``：``.npz`` 缓存；已缓存的图直接复用（重复实验省时间）
    * 检测不到人脸 / 读图失败的图会被**记录**在 ``failures`` 里，而不是让整个评测崩掉
    """
    import cv2  # 延迟导入：让本模块在没有 opencv 的环境里仍可被 import（便于单测）

    root = Path(root)
    cache = Path(cache_path) if cache_path else None

    # ---------- 读缓存 ----------
    vectors: dict[str, np.ndarray] = {}
    if cache and cache.exists():
        with np.load(cache, allow_pickle=False) as z:
            for key in z.files:
                vectors[key] = z[key].astype(np.float32)
        if verbose:
            print(f"  [cache] 命中 {len(vectors)} 张：{cache}")

    todo = [r for r in relpaths if r not in vectors]
    if verbose:
        print(f"  待处理 {len(todo)} 张（共 {len(relpaths)} 张）")

    failures: dict[str, str] = {}
    t0 = time.time()
    for i, rel in enumerate(todo, 1):
        img = cv2.imread(str(root / rel))
        if img is None:
            failures[rel] = "read_failed"
            continue
        faces = app.get(img)
        if not faces:
            failures[rel] = "no_face"
            continue
        # 多人脸时取面积最大的
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        vectors[rel] = np.asarray(face.normed_embedding, dtype=np.float32)
        if verbose and i % log_every == 0:
            rate = i / max(time.time() - t0, 1e-6)
            print(f"  {i}/{len(todo)}  {rate:.1f} 张/秒")

    elapsed = time.time() - t0
    if verbose and todo:
        print(f"  完成 {len(todo)} 张，用时 {elapsed:.1f}s（{len(todo) / max(elapsed, 1e-6):.1f} 张/秒）")

    # ---------- 写缓存 ----------
    if cache and todo:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **vectors)
        if verbose:
            print(f"  [cache] 已写入 {len(vectors)} 张：{cache}")

    try:
        provider = list(app.models["recognition"].session.get_providers())
    except Exception:  # noqa: BLE001
        provider = []

    return Embeddings(vectors=vectors, failures=failures, provider=provider, seconds=elapsed)
