#!/usr/bin/env node
/**
 * Keep the per-package LICENSE/NOTICE copies in sync with the repository root.
 *
 * npm does not hoist these, and `@fllstck/mlayax-darwin-arm64` must ship them without any
 * install or prepack script (TASKS.md §5), so the copies are committed. This script is how we
 * avoid them drifting.
 *
 * Usage:
 *   node scripts/sync-licenses.mjs           # write the copies
 *   node scripts/sync-licenses.mjs --check   # exit 1 if any copy is stale (CI)
 */

import { readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const check = process.argv.includes("--check");

const SOURCES = ["LICENSE", "NOTICE"];
const PACKAGES = ["packages/mlayax", "packages/mlayax-darwin-arm64"];

let stale = 0;

for (const file of SOURCES) {
  const canonical = readFileSync(path.join(root, file), "utf8");
  for (const pkg of PACKAGES) {
    const dest = path.join(root, pkg, file);
    let current;
    try {
      current = readFileSync(dest, "utf8");
    } catch {
      current = null;
    }

    if (current === canonical) continue;

    if (check) {
      console.error(`stale: ${pkg}/${file}`);
      stale += 1;
    } else {
      writeFileSync(dest, canonical);
      console.log(`wrote: ${pkg}/${file}`);
    }
  }
}

if (check && stale > 0) {
  console.error(`\n${stale} stale licence file(s) — run: node scripts/sync-licenses.mjs`);
  process.exit(1);
}
if (check) console.log("licence copies are in sync");
