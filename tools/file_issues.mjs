#!/usr/bin/env node
/**
 * 把 docs/issues/*.md 里的任务卡批量提交到 GitHub Issues。
 *
 * 特点：
 *  - 零依赖（只用 Node 内置模块），不需要 npm install
 *  - 幂等：同标题的 Issue 已存在则跳过，可安全重复执行
 *  - 两遍提交：先建 Issue，再把正文里的 {{#slug}} 占位符替换成真实的 #编号
 *  - 自动创建标签与里程碑，已存在的自动跳过
 *
 * 用法（PowerShell / bash 通用）：
 *   GH_TOKEN=<你的 token> node tools/file_issues.mjs --dry-run   # 先看要建什么
 *   GH_TOKEN=<你的 token> node tools/file_issues.mjs             # 首次批量创建
 *   GH_TOKEN=<你的 token> node tools/file_issues.mjs --update    # 改了任务卡后同步到已有 Issue
 *
 * 可选环境变量：
 *   GH_REPO   目标仓库，默认 DrGlitch666/AIGC-in-Face-Recognition-Dataset-Augmentation
 *   ISSUES_DIR 任务卡目录，默认 <cwd>/docs/issues
 *
 * Token 权限：classic PAT 需要 repo 权限；fine-grained PAT 需要 Issues: Read and write。
 */

import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';

const REPO = process.env.GH_REPO || 'DrGlitch666/AIGC-in-Face-Recognition-Dataset-Augmentation';
const TOKEN = process.env.GH_TOKEN || process.env.GITHUB_TOKEN || '';
const ISSUES_DIR = process.env.ISSUES_DIR || path.join(process.cwd(), 'docs', 'issues');
const API = 'https://api.github.com';
const DRY_RUN = process.argv.includes('--dry-run');
/** --update：把本地任务卡的改动同步到已存在的 Issue（按标题匹配），而不是只创建新的。 */
const UPDATE = process.argv.includes('--update');

/** 标签目录：颜色与说明。名字用中文，方便在 GitHub 上按类型筛选。 */
const LABEL_CATALOG = {
  '优先级:P0': { color: 'b60205', description: '不做完项目就断了，最高优先级' },
  '优先级:P1': { color: 'd93f0b', description: '影响结论完整性' },
  '优先级:P2': { color: 'fbca04', description: '加分项，时间富余才做' },
  '类型:环境': { color: '0e8a16', description: '开发环境与工程脚手架' },
  '类型:数据': { color: '1d76db', description: '数据集下载、对齐、清单' },
  '类型:生成': { color: '5319e7', description: 'AIGC 生成管线' },
  '类型:筛选': { color: '7057ff', description: '生成质量与身份一致性筛选' },
  '类型:训练': { color: '006b75', description: '识别模型训练' },
  '类型:评测': { color: '0052cc', description: '评测协议与指标' },
  '类型:实验': { color: 'c2e0c6', description: '对照实验与消融' },
  '类型:网站': { color: 'ff7619', description: '静态展示网站' },
  '类型:文档': { color: '0075ca', description: '文档与综述' },
  '类型:合规': { color: 'd4c5f9', description: '伦理、许可、发布' },
  '难度:入门': { color: 'c5def5', description: '零基础可以直接上手' },
  '难度:中等': { color: 'bfd4f2', description: '需要查资料与调试' },
  '难度:进阶': { color: 'a2b1c6', description: '需要一定经验，预留缓冲时间' },
};
const FALLBACK_LABEL = { color: 'ededed', description: '' };

// ---------------------------------------------------------------- 工具函数

function die(msg) {
  console.error(`\n[错误] ${msg}\n`);
  process.exit(1);
}

async function api(method, endpoint, body) {
  const res = await fetch(`${API}${endpoint}`, {
    method,
    headers: {
      'User-Agent': 'aigcfr-issue-filer',
      Authorization: `Bearer ${TOKEN}`,
      Accept: 'application/vnd.github+json',
      'X-GitHub-Api-Version': '2022-11-28',
      ...(body ? { 'Content-Type': 'application/json' } : {}),
    },
    ...(body ? { body: JSON.stringify(body) } : {}),
  });
  const text = await res.text();
  let json = null;
  try { json = text ? JSON.parse(text) : null; } catch { /* 非 JSON 响应 */ }
  return { ok: res.ok, status: res.status, json, text };
}

/** 解析 YAML front-matter（只支持本项目用到的简单形式，避免引入依赖）。 */
function parseFrontMatter(raw) {
  if (!raw.startsWith('---')) return { data: {}, body: raw };
  const end = raw.indexOf('\n---', 3);
  if (end === -1) return { data: {}, body: raw };
  const fm = raw.slice(3, end);
  const body = raw.slice(raw.indexOf('\n', end + 1) + 1).replace(/^\s*\n/, '');
  const data = {};
  for (const line of fm.split(/\r?\n/)) {
    const m = line.match(/^([A-Za-z_][\w-]*):\s*(.*)$/);
    if (!m) continue;
    const key = m[1];
    const value = m[2].trim();
    if (value.startsWith('[') && value.endsWith(']')) {
      data[key] = value.slice(1, -1).split(',')
        .map((s) => s.trim().replace(/^["']|["']$/g, ''))
        .filter(Boolean);
    } else {
      data[key] = value.replace(/^["']|["']$/g, '');
    }
  }
  return { data, body };
}

function loadTaskCards() {
  if (!fs.existsSync(ISSUES_DIR)) die(`找不到任务卡目录：${ISSUES_DIR}`);
  const files = fs.readdirSync(ISSUES_DIR)
    // 任务卡命名形如 01-xxx.md；排除自动生成的 00-INDEX.md / 00-index.json
    .filter((f) => f.endsWith('.md') && /^\d/.test(f) && !f.startsWith('00-'))
    .sort();
  if (files.length === 0) die(`${ISSUES_DIR} 下没有形如 01-xxx.md 的任务卡`);

  return files.map((file) => {
    const { data, body } = parseFrontMatter(fs.readFileSync(path.join(ISSUES_DIR, file), 'utf8'));
    if (!data.title) die(`${file} 缺少 front-matter 的 title 字段`);
    if (!data.slug) die(`${file} 缺少 front-matter 的 slug 字段`);
    return {
      file,
      slug: data.slug,
      title: data.title,
      labels: data.labels || [],
      milestone: data.milestone || null,
      body: body.trim(),
    };
  });
}

// ---------------------------------------------------------------- 主流程

async function main() {
  if (!TOKEN) die('缺少 GH_TOKEN 环境变量。示例：$env:GH_TOKEN="ghp_xxx"; node tools/file_issues.mjs');
  const cards = loadTaskCards();

  console.log(`仓库      : ${REPO}`);
  console.log(`任务卡目录: ${ISSUES_DIR}`);
  console.log(`任务卡数量: ${cards.length}`);
  console.log(`模式      : ${DRY_RUN ? 'DRY-RUN（只打印，不提交）' : UPDATE ? 'UPDATE（同步已存在的 Issue）' : '正式创建'}`);
  console.log('');

  if (DRY_RUN) {
    for (const [i, c] of cards.entries()) {
      console.log(`${String(i + 1).padStart(2, '0')}. ${c.title}`);
      console.log(`    标签: ${c.labels.join(' / ') || '(无)'}   里程碑: ${c.milestone || '(无)'}`);
      console.log(`    正文: ${c.body.length} 字符   文件: ${c.file}`);
    }
    console.log('\nDRY-RUN 结束，未做任何修改。');
    return;
  }

  // 0) 校验 token 与仓库
  const me = await api('GET', '/user');
  if (!me.ok) die(`Token 校验失败（HTTP ${me.status}）：${me.text.slice(0, 200)}`);
  console.log(`已认证用户: ${me.json.login}`);

  const repo = await api('GET', `/repos/${REPO}`);
  if (!repo.ok) die(`无法访问仓库 ${REPO}（HTTP ${repo.status}）`);
  if (!repo.json.has_issues) die(`仓库 ${REPO} 没有开启 Issues 功能`);
  console.log(`仓库权限  : ${JSON.stringify(repo.json.permissions)}`);

  // 1) 标签
  const wanted = new Set(cards.flatMap((c) => c.labels));
  for (const name of wanted) {
    const meta = LABEL_CATALOG[name] || FALLBACK_LABEL;
    const r = await api('POST', `/repos/${REPO}/labels`, { name, color: meta.color, description: meta.description });
    if (r.ok) console.log(`  + 标签 ${name}`);
    else if (r.status === 422) console.log(`  = 标签 ${name}（已存在）`);
    else console.log(`  ! 标签 ${name} 创建失败 HTTP ${r.status}: ${r.text.slice(0, 120)}`);
  }

  // 2) 里程碑
  const existingMs = await api('GET', `/repos/${REPO}/milestones?state=all&per_page=100`);
  const msMap = new Map((existingMs.json || []).map((m) => [m.title, m.number]));
  for (const title of [...new Set(cards.map((c) => c.milestone).filter(Boolean))]) {
    if (msMap.has(title)) { console.log(`  = 里程碑 ${title}`); continue; }
    const r = await api('POST', `/repos/${REPO}/milestones`, { title });
    if (r.ok) { msMap.set(title, r.json.number); console.log(`  + 里程碑 ${title}`); }
    else console.log(`  ! 里程碑 ${title} 创建失败 HTTP ${r.status}: ${r.text.slice(0, 120)}`);
  }

  // 3) 现有 Issue（幂等：同标题跳过）
  const existing = await api('GET', `/repos/${REPO}/issues?state=all&per_page=100`);
  const byTitle = new Map((existing.json || []).filter((i) => !i.pull_request).map((i) => [i.title, i.number]));

  // 4) 建 Issue（正文先保留 {{#slug}} 占位符）
  const created = [];
  for (const card of cards) {
    if (byTitle.has(card.title)) {
      const n = byTitle.get(card.title);
      console.log(`  = #${n} ${card.title}（已存在${UPDATE ? '，稍后同步内容' : '，跳过'}）`);
      created.push({ ...card, number: n, existing: true, url: `${API.replace('api.', '')}/${REPO}/issues/${n}` });
      continue;
    }
    const payload = {
      title: card.title,
      body: card.body,
      labels: card.labels,
      ...(card.milestone && msMap.has(card.milestone) ? { milestone: msMap.get(card.milestone) } : {}),
    };
    const r = await api('POST', `/repos/${REPO}/issues`, payload);
    if (!r.ok) { console.log(`  ! 创建失败 HTTP ${r.status} ${card.title}: ${r.text.slice(0, 200)}`); continue; }
    console.log(`  + #${r.json.number} ${card.title}`);
    created.push({ ...card, number: r.json.number, url: r.json.html_url });
  }

  // 5) 第二遍：把 {{#slug}} 替换成真实编号（--update 时同时同步标题/标签/里程碑）
  const slugToNumber = new Map(created.map((c) => [c.slug, c.number]));
  console.log('\n解析依赖占位符……');
  for (const card of created) {
    if (card.existing && !UPDATE) continue;
    const unresolved = [];
    const resolved = card.body.replace(/\{\{#([0-9a-z-]+)\}\}/g, (whole, slug) => {
      if (slugToNumber.has(slug)) return `#${slugToNumber.get(slug)}`;
      unresolved.push(slug);
      return whole;
    });
    if (unresolved.length) console.log(`  ! #${card.number} 有未解析的依赖：${unresolved.join(', ')}`);
    if (!UPDATE && resolved === card.body) continue;
    const payload = UPDATE
      ? {
          title: card.title,
          body: resolved,
          labels: card.labels,
          ...(card.milestone && msMap.has(card.milestone) ? { milestone: msMap.get(card.milestone) } : {}),
        }
      : { body: resolved };
    const r = await api('PATCH', `/repos/${REPO}/issues/${card.number}`, payload);
    if (r.ok) console.log(`  ~ #${card.number} 已同步（${card.slug}）`);
    else console.log(`  ! #${card.number} 更新失败 HTTP ${r.status}`);
  }

  // 6) 写索引
  const index = {
    repo: REPO,
    generated_at: new Date().toISOString(),
    issues_dir: path.relative(process.cwd(), ISSUES_DIR).replace(/\\/g, '/'),
    count: created.length,
    issues: created.map((c) => ({
      number: c.number, slug: c.slug, title: c.title,
      milestone: c.milestone, labels: c.labels, file: c.file, url: c.url,
    })),
  };
  fs.writeFileSync(path.join(ISSUES_DIR, '00-index.json'), `${JSON.stringify(index, null, 2)}\n`, 'utf8');

  const lines = [
    '# Issue 索引（自动生成，请勿手工修改）',
    '',
    `> 由 \`tools/file_issues.mjs\` 于 ${index.generated_at} 生成，共 ${index.count} 条。`,
    '> 重新生成：`GH_TOKEN=<token> node tools/file_issues.mjs`（已存在的 Issue 会自动跳过）。',
    '',
    '| # | 标题 | 里程碑 | 标签 | 任务卡 |',
    '|---|---|---|---|---|',
    ...index.issues.map((i) => `| [#${i.number}](${i.url}) | ${i.title} | ${i.milestone || '-'} | ${i.labels.join(' ')} | [\`${i.file}\`](${i.file}) |`),
    '',
  ];
  fs.writeFileSync(path.join(ISSUES_DIR, '00-INDEX.md'), lines.join('\n'), 'utf8');

  console.log(`\n完成：创建/确认 ${created.length} 条 Issue。`);
  console.log(`索引已写入：${path.join(index.issues_dir, '00-INDEX.md')} 与 00-index.json`);
}

main().catch((err) => die(err.stack || String(err)));
