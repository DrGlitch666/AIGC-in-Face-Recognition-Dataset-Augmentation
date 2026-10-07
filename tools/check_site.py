#!/usr/bin/env python
"""发布前自检：确认 site/ 里的每个本地引用都能解析到真实文件。

## 为什么需要

网站是**双击可开**的静态站，所有资源走相对路径。一旦路径写错，
本地可能因为缓存看着正常，**线上就是 404** —— 而 GitHub Pages 部署一次要等一两分钟，
靠肉眼点链接检查很不可靠。这个脚本把检查自动化，并接到 CI 上：
**引用坏了，部署直接失败**，不会把破站发上去。

只检查**本地引用**：
* `#anchor` 忽略（页内跳转）
* `http://` / `https://` / `//` 开头的外部引用**报错** ——
  本项目要求完全离线自足，出现外链就是引入了运行时依赖
* 其余按相对路径解析，必须存在

用法：
    python tools/check_site.py site
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# ⚠️ Windows 控制台默认 GBK，输出非 GBK 字符（emoji 等）会直接抛
#    UnicodeEncodeError 把脚本干掉。必须显式放宽 errors。
sys.stdout.reconfigure(errors="replace")

REF_RE = re.compile(r"""(?:src|href)\s*=\s*["']([^"']+)["']""", re.IGNORECASE)


def check(site_root: Path) -> list[str]:
    problems: list[str] = []
    entries = sorted(site_root.rglob("*.html"))
    if not entries:
        return [f"{site_root} 下没有 HTML 文件"]

    for html in entries:
        text = html.read_text(encoding="utf-8", errors="replace")
        refs = REF_RE.findall(text)
        for ref in refs:
            if ref.startswith("#"):
                continue
            if ref.startswith(("http://", "https://", "//")):
                problems.append(f"{html.name}: 发现外部引用（要求离线自足）: {ref}")
                continue
            if ref.startswith(("mailto:", "data:", "javascript:")):
                continue
            target = (html.parent / ref.split("#", 1)[0].split("?", 1)[0]).resolve()
            if not target.exists():
                problems.append(f"{html.name}: 引用不存在: {ref}")
        # 顺手统计一下，让人看到检查范围
        n_ok = sum(1 for r in refs
                   if not r.startswith(("#", "http://", "https://", "//", "mailto:", "data:", "javascript:"))
                   and (html.parent / r.split("#", 1)[0]).resolve().exists())
        print(f"  {html.relative_to(site_root)}: {len(refs)} 个引用，{n_ok} 个本地文件均存在")

    total = sum(f.stat().st_size for f in site_root.rglob("*") if f.is_file())
    print(f"  站点总体积: {total / 1024 / 1024:.2f} MB")
    if total > 5 * 1024 * 1024:
        problems.append(f"总体积 {total / 1024 / 1024:.2f} MB 超过 5 MB 上限")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False, description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("site_root", nargs="?", default="site")
    args = ap.parse_args()
    root = Path(args.site_root).resolve()
    if not root.is_dir():
        print(f"!! 目录不存在: {root}")
        return 2

    print(f"检查 {root}")
    problems = check(root)
    if problems:
        print("\n[FAIL] 发现问题：")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\n[PASS] 全部引用可解析、无外部依赖、体积达标")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
