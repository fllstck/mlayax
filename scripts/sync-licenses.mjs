#!/usr/bin/env node
/**
 * Keep the per-package LICENSE/NOTICE/licences copies in sync with the repository root.
 *
 * npm does not hoist these, and `@fllstck/mlayax-darwin-arm64` must ship them without any
 * install or prepack script (TASKS.md §5), so the copies are committed. This script is how we
 * avoid them drifting.
 *
 * Usage:
 *   node scripts/sync-licenses.mjs           # write the copies
 *   node scripts/sync-licenses.mjs --check   # exit 1 if any copy is stale (CI)
 */

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const check = process.argv.includes("--check");

const SOURCES = ["LICENSE", "NOTICE"];
const PACKAGES = ["packages/mlayax", "packages/mlayax-darwin-arm64"];

/**
 * Third-party texts, copied as a directory: root `licenses/<file>` -> `<pkg>/licenses/<file>`.
 *
 * Both packages carry them, not just the platform package. The façade vendors node-mlx's JavaScript
 * layer and redistributes it, so it has its own notice obligation; giving it the platform package's
 * texts is the cheap, unarguable way to satisfy both.
 */
const LICENCE_DIR = "licenses";
const LICENCE_TEXTS = readdirSync(path.join(root, LICENCE_DIR)).filter((f) => f.endsWith(".txt"));

let stale = 0;

function sync(dest, canonical, label) {
  let current;
  try {
    current = readFileSync(dest, "utf8");
  } catch {
    current = null;
  }
  if (current === canonical) return;

  if (check) {
    console.error(`stale: ${label}`);
    stale += 1;
  } else {
    mkdirSync(path.dirname(dest), { recursive: true });
    writeFileSync(dest, canonical);
    console.log(`wrote: ${label}`);
  }
}

for (const file of SOURCES) {
  const canonical = readFileSync(path.join(root, file), "utf8");
  for (const pkg of PACKAGES) {
    sync(path.join(root, pkg, file), canonical, `${pkg}/${file}`);
  }
}

for (const file of LICENCE_TEXTS) {
  const canonical = readFileSync(path.join(root, LICENCE_DIR, file), "utf8");
  for (const pkg of PACKAGES) {
    sync(path.join(root, pkg, LICENCE_DIR, file), canonical, `${pkg}/${LICENCE_DIR}/${file}`);
  }
}

if (check && stale > 0) {
  console.error(`\n${stale} stale licence file(s) — run: node scripts/sync-licenses.mjs`);
  process.exit(1);
}
if (check) console.log("licence copies are in sync");
