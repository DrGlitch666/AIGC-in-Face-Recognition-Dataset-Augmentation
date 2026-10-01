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
    """拿远端文件大小。

    ⚠️ 不能只靠 HEAD：实测有镜像对某些路径 **HEAD 直接失败**（返回 None），
    而"已完整就跳过"的判断依赖这个大小 —— 大小拿不到就会对着完整文件发无效 Range，
    接着吃 416。所以这里加一条兜底：GET 首字节，从 ``Content-Range`` 解析总大小。
    （只读响应头，不读 body，所以不会真的把文件拖下来。）
    """
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=20) as resp:
            cl = resp.headers.get("Content-Length")
            if cl:
                return int(cl)
    except Exception:  # noqa: BLE001
        pass

    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            content_range = resp.headers.get("Content-Range")     # 形如 "bytes 0-0/123456"
            if content_range and "/" in content_range:
                total = content_range.rsplit("/", 1)[1].strip()
                if total.isdigit():
                    return int(total)
            cl = resp.headers.get("Content-Length")
            if cl and resp.status == 200:
                return int(cl)
    except Exception:  # noqa: BLE001
        pass
    return None


def download(url: str, dst: Path, total: int | None) -> None:
    """流式下载 + 进度显示 + 断点续传。

    ⚠️ 关于 HTTP 416（真实踩过）：
    "文件已经完整但拿不到远端大小"时会走进续传分支，发出 ``Range: bytes=<完整大小>-``，
    服务器回 **416 Requested Range Not Satisfiable**，整批下载就中断了。
    原因是我们靠 HEAD 请求拿大小，而 HEAD **可能失败**（返回 None），
    这时"已完成就跳过"的判断失效。

    所以这里对 416 做兜底：**丢掉 Range 完整重下**。
    这样既覆盖"文件其实已完整"，也覆盖"本地残片比远端还大"这类情况。
    """
    dst.parent.mkdir(parents=True, exist_ok=True)   # 文件名可能带子目录（如 unet/xxx.bin）
    done = dst.stat().st_size if dst.exists() else 0
    if total and done == total:
        print(f"    已完整（{human(done)}），跳过")
        return

    for use_range in (True, False):
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        mode, received = "wb", 0
        if use_range and done:
            req.add_header("Range", f"bytes={done}-")
            mode, received = "ab", done
            print(f"    断点续传：已有 {human(done)}")

        t0 = time.time()
        last_report = received
        try:
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
                        print(f"    {received / total * 100:5.1f}%  "
                              f"{human(received)}/{human(total)}  {human(speed)}/s")
                        last_report = received
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and use_range:
                print("    [!] HTTP 416（范围无效，通常意味着本地文件已完整或比远端大）"
                      "，改为丢弃 Range 完整重下")
                continue
            raise

        size = dst.stat().st_size
        speed = size / max(time.time() - t0, 1e-6)
        print(f"    完成 {human(size)}，用时 {time.time() - t0:.0f}s（{human(speed)}/s）")
        return


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _archive_top_levels(archive: Path) -> set[str]:
    """列出压缩包里的**顶层名字**，用来判断它是"自带一层目录"还是"散着一堆文件"。"""
    name = archive.name.lower()
    tops: set[str] = set()
    try:
        if name.endswith((".tgz", ".tar.gz")):
            with tarfile.open(archive, "r:gz") as tf:
                for member in tf.getmembers():
                    parts = Path(member.name).parts
                    if parts:
                        tops.add(parts[0])
        elif name.endswith(".zip"):
            with zipfile.ZipFile(archive) as zf:
                for entry in zf.namelist():
                    parts = Path(entry).parts
                    if parts:
                        tops.add(parts[0])
    except Exception:  # noqa: BLE001
        return set()
    return tops


def extract(archive: Path, dest: Path, into: str | None = None) -> Path:
    """解压到 ``dest``，返回实际解压到的目录。

    ⚠️ 目标目录规则（真实踩过坑，别随手改）：

    * 压缩包**自带一层顶层目录**（如 ``lfw.tgz`` 里就是 ``lfw/``）→ 直接解到 ``dest``
    * 压缩包是**散着的文件**（如 ``antelopev2.zip`` 里直接是 ``.onnx``）→ 解到 ``dest/<压缩包名>/``
      因为消费方 insightface 期望 ``<root>/models/antelopev2/*.onnx``，
      散着解到 ``dest`` 会把 onnx 文件撒得到处都是
    * 也可以用 ``into`` 显式指定子目录名
    """
    name = archive.name.lower()
    if not name.endswith((".tgz", ".tar.gz", ".zip")):
        print(f"    （不认识的压缩格式，跳过：{archive.name}）")
        return dest

    if into is None:
        tops = _archive_top_levels(archive)
        # 只有一个顶层项且它不是文件（即"带目录的包"）→ 解到 dest；否则建一个同名目录
        into = "" if len(tops) == 1 else archive.name.rsplit(".", 1)[0].replace(".tar", "")
    target = dest / into if into else dest
    target.mkdir(parents=True, exist_ok=True)
    print(f"    解压 {archive.name} -> {target}")

    if name.endswith((".tgz", ".tar.gz")):
        with tarfile.open(archive, "r:gz") as tf:
            try:
                tf.extractall(target, filter="data")   # Python 3.12+
            except TypeError:
                tf.extractall(target)                  # Python 3.11
    else:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target)
    print("    解压完成")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False, description="下载数据集原始文件")
    parser.add_argument("--config", required=True, help="数据集配置，如 configs/data/lfw.yaml")
    parser.add_argument("--local", default="auto", help='机器私有配置；默认 "auto" 自动寻找')
    parser.add_argument("--dest", default=None, help="覆盖目标目录（自测用）")
    parser.add_argument("--dry-run", action="store_true", help="只打印将要做什么")
    parser.add_argument("--no-extract", action="store_true", help="下载后不解压")
    parser.add_argument("--only", nargs="*", default=None, help="只下载指定的文件名（可多个）")
    parser.add_argument("--retries", type=int, default=4,
                        help="单个文件的中断重试次数（默认 4，续传接着下）")
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
        paths = cfg.get("paths") or {}
        # dest_root 决定下到哪个根目录：data（默认）或 models。
        # 数据集下到 data_root，模型权重下到 models_root（SD1.5 之类的权重不该混进 data）。
        root_key = {"data": "data_root", "models": "models_root"}.get(
            data_cfg.get("dest_root", "data"), "data_root")
        root = paths.get(root_key)
        if not root:
            print(f"!! {local_file} 里没有 paths.{root_key}，无法确定下载到哪里。")
            return 2
        dest = Path(root) / data_cfg["dest_subdir"]
        print(f"目标目录: {dest}   （dest_root={data_cfg.get('dest_root', 'data')}）")

    # ⚠️ 建目录必须放在 --dry-run 之后：dry-run 应当**零副作用**
    #    （之前的顺序会先建目录再 dry-run，目录不可写时 dry-run 也会失败）

    # ---------- 展开文件列表 ----------
    # 每个源可以有自己的 dest_subdir（例如 SD1.5 与 IP-Adapter 分开放），
    # 没写就放在数据集根目录下。rel 是相对 dest 的落盘路径。
    jobs: list[tuple[str, str, str, str]] = []
    for src in data_cfg["sources"]:
        sub = (src.get("dest_subdir") or "").strip("/")
        for fn in src["files"]:
            rel = f"{sub}/{fn}" if sub else fn
            jobs.append((fn, f"{src['base_url'].rstrip('/')}/{fn}", src["name"], rel))

    if args.only:
        wanted = set(args.only)
        jobs = [j for j in jobs if j[0] in wanted]
        missing = wanted - {j[0] for j in jobs}
        if missing:
            print(f"!! 配置里没有这些文件: {sorted(missing)}")
            return 2
    # 同一个文件可能配了多个镜像（自动回退用），按文件名分组
    by_name: dict[str, list[tuple[str, str, str, str]]] = {}
    for job in jobs:
        by_name.setdefault(job[0], []).append(job)

    # 估算总量：同一个文件可能配了多个镜像，**只算一次**（否则会重复计数）
    total_bytes = 0
    counted: set[str] = set()
    for fn, url, _mirror, _rel in jobs:
        if fn in counted:
            continue
        counted.add(fn)
        total_bytes += remote_size(url) or 0
    print(f"\n数据集: {data_cfg['name']}  |  共 {len(by_name)} 个文件"
          f"{f'，约 {human(total_bytes)}' if total_bytes else ''}\n")

    if args.dry_run:
        for fn, url, mirror, rel in jobs:
            size = remote_size(url)
            print(f"  [dry-run] {rel:58} {human(size) if size else '?':>10}  <- {mirror}")
        print(f"\n--dry-run 结束，未下载任何内容。预计总量 {human(total_bytes)}")
        return 0

    dest.mkdir(parents=True, exist_ok=True)

    # ---------- 逐个文件下载（同一文件可配多个镜像，失败自动换下一个）----------
    results: dict[str, str] = {}
    for fn, candidates in by_name.items():
        success = False
        for _, url, mirror, rel in candidates:
            out = dest / rel
            print(f"[{rel}]  源: {mirror}")
            total = remote_size(url)
            if total:
                print(f"    远端大小 {human(total)}")

            # ⚠️ 必须重试：镜像连接会中途断掉（实测 antelopev2 下到 115MB 断了），
            #    而 download() 支持断点续传，所以重试就能接着下，不用从头再来。
            attempt = 0
            failed = False
            while True:
                attempt += 1
                try:
                    download(url, out, total)
                    break
                except Exception as exc:  # noqa: BLE001
                    if attempt >= args.retries:
                        print(f"    !! 本镜像失败（尝试 {attempt} 次）: {type(exc).__name__}: {exc}")
                        failed = True
                        break
                    wait = min(5 * attempt, 30)
                    done_bytes = out.stat().st_size if out.exists() else 0
                    print(f"    [!] 第 {attempt} 次中断（{type(exc).__name__}），"
                          f"已有 {human(done_bytes)}；{wait}s 后续传重试 ...")
                    time.sleep(wait)
            if failed:
                continue          # 换下一个镜像

            if total and out.stat().st_size != total:
                print(f"    !! 大小不符：本地 {out.stat().st_size} 字节 vs 远端 {total} 字节")
                out.unlink(missing_ok=True)      # 残包会污染下一个镜像的续传
                continue

            digest = sha256_of(out)
            expect = (data_cfg.get("checksums") or {}).get(fn)
            if expect and expect != digest:
                print(f"    !! sha256 不匹配！\n       期望 {expect}\n       实际 {digest}")
                return 1
            results[fn] = digest
            print(f"    sha256 {digest[:16]}...{'（已校验）' if expect else '（配置未提供，建议回填）'}")
            success = True
            break

        if not success:
            print(f"!! {fn} 的所有镜像都失败了。提示：配置里再加一个镜像，或直接重跑本脚本续传。")
            return 1

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
