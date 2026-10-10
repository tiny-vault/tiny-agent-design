#!/usr/bin/env node
// Keep Markdown unwrapped: one paragraph (or list item) = one line.
//
//   node tools/unwrap-md.mjs                 dry run over every .md file in the repo
//   node tools/unwrap-md.mjs --write         rewrite them
//   node tools/unwrap-md.mjs --check         exit 1 if any file needs unwrapping (for CI)
//   node tools/unwrap-md.mjs --write a.md b.md   only these files
//
// Why: viewers that keep each source line break draw wrapped text as a narrow
// column. Long lines are wrapped on screen by the editor and the browser.
//
// Left untouched: fenced code, indented code, tables, headings, HTML blocks,
// blockquotes, horizontal rules, link definitions, directives, hard line breaks.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const SKIP = new Set(["node_modules", ".venv", ".git", "_build", "test-results", "playwright-report", ".pytest_cache", "dist", "__pycache__", "generated"]);

const argv = process.argv.slice(2);
const write = argv.includes("--write");
const check = argv.includes("--check");
const given = argv.filter((a) => !a.startsWith("--"));

function findMarkdown(dir, out = []) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (SKIP.has(entry.name)) continue;
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) findMarkdown(full, out);
    else if (entry.name.endsWith(".md")) out.push(full);
  }
  return out;
}

const FENCE = /^\s*(```|~~~)/;
const LIST_ITEM = /^(\s*)([-*+]|\d+[.)])\s+/;
const STRUCTURAL = [/^#{1,6}(\s|$)/, /^\s*\|/, /^>/, /^\s*(-{3,}|\*{3,}|_{3,})\s*$/, /^\s*\[[^\]]+\]:\s/, /^:{3,}/, /^\$\$/, /^(=+|-+)\s*$/];
const hardBreak = (line) => / {2}$/.test(line) || /\\$/.test(line);

export function unwrap(text) {
  const eol = text.includes("\r\n") ? "\r\n" : "\n";
  const lines = text.split(/\r?\n/);
  const out = [];
  let cur = null;
  let inFence = false;
  let inHtml = false;
  const flush = () => {
    if (cur !== null) out.push(hardBreak(cur) ? cur : cur.replace(/\s+$/, ""));
    cur = null;
  };

  lines.forEach((line, i) => {
    if (inFence) {
      out.push(line);
      if (FENCE.test(line)) inFence = false;
      return;
    }
    if (FENCE.test(line)) {
      flush();
      inHtml = false;
      inFence = true;
      out.push(line);
      return;
    }
    if (line.trim() === "") {
      flush();
      inHtml = false;
      out.push(line);
      return;
    }
    if (inHtml) {
      out.push(line);
      return;
    }
    if (i === 0 && line.trim() === "---") {
      out.push(line); // front matter
      return;
    }
    if (/^\s*</.test(line) && cur === null) {
      inHtml = true;
      out.push(line);
      return;
    }
    if (STRUCTURAL.some((re) => re.test(line))) {
      flush();
      out.push(line);
      return;
    }
    if (LIST_ITEM.test(line)) {
      flush();
      cur = line;
      return;
    }
    if (cur === null && /^( {4}|\t)/.test(line)) {
      out.push(line); // indented code block
      return;
    }
    if (cur === null) {
      cur = line;
      return;
    }
    if (hardBreak(cur)) {
      out.push(cur);
      cur = line;
      return;
    }
    cur = `${cur.replace(/\s+$/, "")} ${line.trim()}`;
  });
  flush();
  return out.join(eol);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const files = given.length ? given.map((f) => path.resolve(f)) : findMarkdown(ROOT);
  const dirty = [];
  for (const file of files) {
    const before = fs.readFileSync(file, "utf8");
    const after = unwrap(before);
    if (after === before) continue;
    dirty.push(path.relative(ROOT, file).replaceAll("\\", "/"));
    if (write) fs.writeFileSync(file, after);
  }
  for (const f of dirty) console.log(`${write ? "unwrapped" : "needs unwrapping"}  ${f}`);
  if (!dirty.length) console.log("Markdown is already unwrapped.");
  if (check && dirty.length) {
    console.error(`\n${dirty.length} file(s) have wrapped paragraphs. Run: node tools/unwrap-md.mjs --write`);
    process.exitCode = 1;
  }
}
