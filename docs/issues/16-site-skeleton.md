---
title: "[网站] 静态展示站骨架：Vite + React + ECharts + 数据契约"
slug: "16-site-skeleton"
labels: ["优先级:P1", "类型:网站", "难度:中等"]
milestone: "M4 网站与发布"
---

## 背景 / 为什么做这个

项目的最终形态是一个**别人打开就能看懂**的展示网站。它必须满足三条硬要求：

1. **纯静态**：不需要 GPU、不需要后端、不需要数据库——打开就是网页。
2. **数据驱动**：所有数字来自 `results/` 导出的 JSON，**不允许把数字手写进网页**（否则改了实验网页就对不上）。
3. **可以提前做**：网站只依赖**数据契约**（`docs/FRAMEWORK.md` §3.3 契约 B），不依赖实验是否跑完。所以**现在就搭骨架、用假数据填**，等 {{#14-mix-ratio-matrix}} 出结果直接替换 JSON。

这也是**并行开发的最大机会**：实验在后台跑训练的时候，正好写网站。

## 目标

`npm run dev` 能打开一个有布局、有路由、有至少一个真实图表的站点；`npm run build` 产出可直接部署的静态文件。

## 任务清单

- [ ] 初始化 `site/`：Vite + React + TypeScript
- [ ] 装图表库（ECharts 或 Recharts，二选一）与路由（react-router）
- [ ] 建立页面骨架与路由（内容留空，见 {{#17-site-content}}）：
      `/`（首页）、`/method`（方法）、`/gallery`（图库）、`/metrics`（指标）、`/fairness`（公平性与分桶）、`/failures`（失败案例）、`/reproduce`（复现指南）、`/ethics`（伦理与合规）
- [ ] 建全局布局：顶部导航 + 侧边目录 + 内容区 + 页脚（含「本页图片由 AI 生成」声明）
- [ ] 定义 `site/public/data/` 的数据文件结构（与契约 B 对齐）：
  - [ ] `runs.json`：所有 run 的指标汇总（由 `summary.csv` 转出）
  - [ ] `mix_ratio.json`：比例曲线的点集
  - [ ] `gallery.json`：图库条目（图片路径 + 身份 + 来源 + 相似度）
  - [ ] `meta.json`：项目元信息（生成时间、实验配置哈希、数据规模）
- [ ] 写 `site/src/lib/data.ts`：统一的数据加载层，**带类型定义 + 运行时校验**（数据缺字段时给出明确报错，而不是白屏）
- [ ] 先用 **mock 数据**把至少 2 个图表渲染出来（柱状图 + 曲线图）
- [ ] 图片资源策略（重要，避免仓库爆炸）：
  - [ ] 生成图统一转成 **webp 缩略图**（长边 512），原图不入库
  - [ ] 缩略图总大小控制在合理范围（例如 < 50 MB），在 `docs/` 里记录总量
  - [ ] 懒加载 + 分页/虚拟滚动（图库可能有几百张）
- [ ] 写 `scripts/export_site_data.py`：读 `results/` → 生成 `site/public/data/*.json`（真实数据，接口与 mock 一致）
- [ ] 部署配置：
  - [ ] 加 `.github/workflows/deploy-pages.yml`：push 到 main 自动 build 并部署
  - [ ] **注意本仓库当前是 private**：GitHub Pages 对私有仓库需要付费计划；**备选方案**：① 把仓库转为 public（推荐，项目本身就是要展示的）② 用 Cloudflare Pages / Vercel / Netlify（都支持私有仓库免费部署）③ 只在本地 `npm run preview` 展示
  - [ ] 在 `site/README.md` 里写清三条路径的具体步骤
- [ ] `npm run build` 产物体积检查，首屏加载 < 3 秒（本地测量）

## 验收标准（Definition of Done）

- [ ] `npm run dev` 本地打开，8 个路由都能访问（内容可以是占位）
- [ ] `npm run build` 成功，`dist/` 是纯静态产物
- [ ] 至少 2 个图表用 **ECharts/Recharts 真实渲染**出来（先喂 mock 数据）
- [ ] 数据加载层对缺失字段有明确报错（故意删掉一个字段，页面应提示而不是白屏）
- [ ] `python scripts/export_site_data.py --results results --out site/public/data` 能跑通，产出 4 个 JSON
- [ ] 仓库里没有任何原始全尺寸生成图（只有 webp 缩略图），并记录了缩略图总量
- [ ] 部署方案已选定并跑通至少一种（Pages / Cloudflare / 本地 preview）

## 交付物

- `site/`（Vite + React + TS 工程）
- `site/public/data/*.json`（mock → 真实）
- `site/src/lib/data.ts`
- `scripts/export_site_data.py`
- `.github/workflows/deploy-pages.yml`
- `site/README.md`

## 依赖

- 需要 {{#02-repo-skeleton}}（目录与配置约定）
- 只依赖**数据契约**，不依赖实验完成 → **可与 {{#14-mix-ratio-matrix}} 并行推进**
- 被依赖：{{#17-site-content}}

## 预估工作量

1.5~2 天

## 参考

- [Vite 官方文档](https://vite.dev/guide/)
- [Apache ECharts](https://echarts.apache.org/zh/index.html)（中文文档齐全，图表类型多）
- [GitHub Pages 部署文档](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)（含 Actions 部署方式）
- [GitHub Pages 可用性说明](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages)（私有仓库的限制在这里）
- [Cloudflare Pages](https://developers.cloudflare.com/pages/) / [Vercel](https://vercel.com/docs)（私有仓库免费部署的替代方案）

## 新手提示 / 卡住了怎么办

- **不要在这一步追求好看**。骨架 + 假数据 + 能构建部署，就算完成。视觉打磨放到 {{#17-site-content}}。
- 数据类型定义（TypeScript interface）要与 `docs/FRAMEWORK.md` 契约 B **逐字段对齐**，这是后面不返工的关键。
- 图片是静态站最大的坑：几百张 512×512 PNG 就能上百 MB。先做缩略图流水线，再往里加图。
- **降级方案**：如果 React 工具链让你痛苦，可以退化成「单个 HTML + ECharts CDN + 手写数据 JSON」——**能展示结果比技术栈先进重要得多**。
