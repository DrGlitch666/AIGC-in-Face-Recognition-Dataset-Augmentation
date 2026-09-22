#!/usr/bin/env node
/**
 * 把「文档正文」里的 {{#NN-slug}} 占位符解析成可点击的 Issue 编号（如 #22）。
 *
 * 背景：任务卡（docs/issues/*.md）里用 `{{#22-offline-mirrors}}` 这种占位符引用别的任务卡，
 * 由 tools/file_issues.mjs 在提交到 GitHub 时替换成真实编号。
 * 但**发布在仓库里的文档**（FRAMEWORK / WORKFLOW / LEARNING / PERSONAL_PLAN / MIRRORS / README）
 * 不应该出现占位符——它们需要直接可读。
 *
 * 约定：任务卡的 slug 以编号开头（`{issue编号}-{短名}`），所以 `{{#22-offline-mirrors}}` → `#22`。
 *
 * 用法：
 *   node tools/resolve_doc_refs.mjs           # 就地替换
 *   node tools/resolve_doc_refs.mjs --check   # 只检查，不修改（有残留则退出码 1）
 *
 * 注意：**不会**处理 docs/issues/*.md（那里的占位符必须保留给 file_issues.mjs 解析）。
 */

import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';

const ROOT = process.cwd();
const DOCS = [
  'README.md',
  'docs/FRAMEWORK.md',
  'docs/WORKFLOW.md',
  'docs/LEARNING.md',
  'docs/PERSONAL_PLAN.md',
  'docs/MIRRORS.md',
  'docs/REFERENCES.md',
];
const CHECK_ONLY = process.argv.includes('--check');
const PATTERN = /\{\{#(\d+)-[a-z0-9-]+\}\}/g;

let totalResolved = 0;
let totalRemaining = 0;

for (const rel of DOCS) {
  const file = path.join(ROOT, rel);
  if (!fs.existsSync(file)) continue;
  const before = fs.readFileSync(file, 'utf8');
  const after = before.replace(PATTERN, (_, num) => `#${Number(num)}`);
  const hits = (before.match(/\{\{#/g) || []).length;

  if (CHECK_ONLY) {
    const left = (after.match(/\{\{#/g) || []).length;
    if (left > 0) {
      console.log(`✗ ${rel} 仍有 ${left} 处占位符未解析（注意：只能解析 {{#数字-短名}} 形式）`);
      totalRemaining += left;
    }
    continue;
  }

  if (hits > 0) {
    fs.writeFileSync(file, after, 'utf8');
    console.log(`✓ ${rel}：解析 ${hits} 处引用`);
    totalResolved += hits;
  }
}

if (CHECK_ONLY) {
  console.log(totalRemaining === 0 ? '✓ 所有文档均无残留占位符' : `✗ 共 ${totalRemaining} 处残留`);
  process.exit(totalRemaining === 0 ? 0 : 1);
}
console.log(`完成：共解析 ${totalResolved} 处引用。`);
