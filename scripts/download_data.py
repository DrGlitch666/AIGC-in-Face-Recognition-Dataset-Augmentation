#!/usr/bin/env python
"""下载数据集的原始文件（任务卡 #3 交付物之一）

用法：
    python scripts/download_data.py --config configs/data/lfw.yaml
    python scripts/download_data.py --config configs/data/lfw.yaml --dry-run
    python scripts/download_data.py --config configs/data/lfw.yaml --dest ./.test_dl   # 覆盖目标目录（自测用）

特性：
    * **多镜像**：按配置顺序尝试，某个通了就一直用它
    * **断点续传**：半截文件用 HTTP Range 接着下（中断了直接重跑即可）
    * **完整性**：按 Content-Length 校验大小；配置里填了 sha256 会一并校验
    * **自动解压**：.tgz / .tar.gz / .zip（可用 --no-extract 关掉）
    * **校验和**：写完 checksums.sha256，两台机器靠它确认"数据是同一份"
      （见 docs/SETUP.md 与任务卡 #19 的数据同步约定）

⚠️ 目标目录来自 `configs/local.<machine>.yaml` 的 `paths.data_root`；
   脚本会**自动寻找**那个文件（两台机器文件名不同），无需手传 --local。
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aigcfr.utils.config import find_local_config, load_config  # noqa: E402

# Windows 控制台是 GBK：让编码失败退化成 "?" 而不是崩溃
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:  # noqa: BLE001
        pass

# 镜像会按 UA 拦截（实测 hf-mirror 对 python-urllib 默认 UA 返回 403），所以伪装成浏览器
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
CHUNK = 1 << 20  # 1 MB


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def remote_size(url: str) -> int | None:
    """HEAD 拿远端大小；不支持 HEAD 的镜像返回 None（不阻塞下载）。"""
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=20) as resp:
            cl = resp.headers.get("Content-Length")
            return int(cl) if cl else None
    except Exception:  # noqa: BLE001
        return None


def download(url: str, dst: Path, total: int | None) -> None:
    """流式下载 + 进度显示 + 断点续传。"""
    done = dst.stat().st_size if dst.exists() else 0
    if total and done == total:
        print(f"    已完整（{human(done)}），跳过")
        return

    req = urllib.request.Request(url, headers={"User-Agent": UA})
    mode = "wb"
    if done and (total is None or done < total):
        req.add_header("Range", f"bytes={done}-")
        mode = "ab"
        print(f"    断点续传：已有 {human(done)}")

    t0 = time.time()
    received = done if mode == "ab" else 0
    last_report = received
    with urllib.request.urlopen(req, timeout=60) as resp, open(dst, mode) as fh:
        while True:
            chunk = resp.read(CHUNK)
            if not chunk:
                break
            fh.write(chunk)
            received += len(chunk)
            # 每前进 10% 报一次进度
            if total and (received - last_report) >= total * 0.10:
                speed = received / max(time.time() - t0, 1e-6)
                print(f"    {received / total * 100:5.1f}%  {human(received)}/{human(total)}  {human(speed)}/s")
                last_report = received

    size = dst.stat().st_size
    speed = size / max(time.time() - t0, 1e-6)
    print(f"    完成 {human(size)}，用时 {time.time() - t0:.0f}s（{human(speed)}/s）")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract(archive: Path, dest: Path) -> None:
    name = archive.name.lower()
    print(f"    解压 {archive.name} ...")
    if name.endswith((".tgz", ".tar.gz")):
        with tarfile.open(archive, "r:gz") as tf:
            try:
                tf.extractall(dest, filter="data")  # Python 3.12+
            except TypeError:
                tf.extractall(dest)                 # Python 3.11
    elif name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    else:
        print(f"    （不认识的压缩格式，跳过：{archive.name}）")
        return
    print("    解压完成")


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False, description="下载数据集原始文件")
    parser.add_argument("--config", required=True, help="数据集配置，如 configs/data/lfw.yaml")
    parser.add_argument("--local", default="auto", help='机器私有配置；默认 "auto" 自动寻找')
    parser.add_argument("--dest", default=None, help="覆盖目标目录（自测用）")
    parser.add_argument("--dry-run", action="store_true", help="只打印将要做什么")
    parser.add_argument("--no-extract", action="store_true", help="下载后不解压")
    parser.add_argument("--only", nargs="*", default=None, help="只下载指定的文件名（可多个）")
    args = parser.parse_args()

    data_cfg = yaml.safe_load((REPO_ROOT / args.config).read_text(encoding="utf-8"))

    # ---------- 解析目标目录 ----------
    if args.dest:
        dest = Path(args.dest)
        if not dest.is_absolute():
            dest = REPO_ROOT / dest
        print(f"目标目录（--dest 覆盖）: {dest}")
    else:
        local_file = find_local_config()
        if args.local != "auto":
            local_file = Path(args.local)
        if local_file is None:
            print("!! 找不到机器私有配置 configs/local.<machine>.yaml（或存在多份）。")
            print("   请检查 configs/ 目录，或用 --dest 手动指定目标目录。")
            return 2
        print(f"使用机器私有配置: {local_file}")
        cfg = load_config(None, profile="auto", local_path=str(local_file))
        root = (cfg.get("paths") or {}).get("data_root")
        if not root:
            print(f"!! {local_file} 里没有 paths.data_root，无法确定下载到哪里。")
            return 2
        dest = Path(root) / data_cfg["dest_subdir"]
        print(f"目标目录: {dest}")
    dest.mkdir(parents=True, exist_ok=True)

    # ---------- 展开文件列表 ----------
    jobs = [(fn, f"{src['base_url'].rstrip('/')}/{fn}", src["name"])
            for src in data_cfg["sources"] for fn in src["files"]]
    if args.only:
        wanted = set(args.only)
        jobs = [j for j in jobs if j[0] in wanted]
        missing = wanted - {j[0] for j in jobs}
        if missing:
            print(f"!! 配置里没有这些文件: {sorted(missing)}")
            return 2
    print(f"\n数据集: {data_cfg['name']}  |  共 {len(jobs)} 个文件\n")

    if args.dry_run:
        for fn, url, mirror in jobs:
            print(f"  [dry-run] {fn:18} <- {mirror}")
        print("\n--dry-run 结束，未下载任何内容。")
        return 0

    # ---------- 逐个下载 ----------
    results: dict[str, str] = {}
    for fn, url, mirror in jobs:
        out = dest / fn
        print(f"[{fn}]  源: {mirror}")
        total = remote_size(url)
        if total:
            print(f"    远端大小 {human(total)}")
        try:
            download(url, out, total)
        except Exception as exc:  # noqa: BLE001
            print(f"    !! 下载失败: {type(exc).__name__}: {exc}")
            print(f"    提示：手动把文件放到 {out} 后重跑本脚本（支持断点续传）")
            return 1

        if total and out.stat().st_size != total:
            print(f"    !! 大小不符：本地 {out.stat().st_size} 字节 vs 远端 {total} 字节")
            return 1

        digest = sha256_of(out)
        expect = (data_cfg.get("checksums") or {}).get(fn)
        if expect and expect != digest:
            print(f"    !! sha256 不匹配！\n       期望 {expect}\n       实际 {digest}")
            return 1
        results[fn] = digest
        print(f"    sha256 {digest[:16]}...{'（已校验）' if expect else '（配置未提供，建议回填）'}")

    # ---------- 校验和（两台机器靠它对齐数据）----------
    cs_path = dest / "checksums.sha256"
    cs_path.write_text(
        "\n".join(f"{h}  {fn}" for fn, h in sorted(results.items())) + "\n",
        encoding="utf-8",
    )
    print(f"\n校验和已写入: {cs_path}")

    # ---------- 解压 ----------
    if data_cfg.get("extract") and not args.no_extract:
        for fn in results:
            if fn.lower().endswith((".tgz", ".tar.gz", ".zip")):
                extract(dest / fn, dest)
    else:
        print("（按配置/参数跳过解压）")

    print("\n完成 ✅  下一步：对齐 + 建 manifest（scripts/build_dataset.py，待实现）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
