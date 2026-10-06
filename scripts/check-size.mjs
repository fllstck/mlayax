#!/usr/bin/env node
/**
 * Size gate for the published tarballs (TASKS.md §5 phase 1, §7).
 *
 * Runs `npm pack --dry-run --json` in each published package and asserts the
 * payload stays inside its budget. Also asserts the tarball contains only
 * intended files, which is how we notice a stray `files` entry.
 *
 * Usage: node scripts/check-size.mjs
 */

import { execFileSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

/** @type {{ dir: string, name: string, budgetBytes: number, kind: "unpacked" | "tarball", allow: RegExp[] }[]} */
const TARGETS = [
  {
    dir: "packages/mlayax",
    name: "@fllstck/mlayax",
    // TASKS.md §0: TypeScript/JS only, < 300 KB.
    budgetBytes: 300 * 1024,
    kind: "unpacked",
    allow: [/^package\.json$/, /^README\.md$/, /^LICENSE$/, /^NOTICE$/, /^dist\//],
  },
  {
    dir: "packages/mlayax-darwin-arm64",
    name: "@fllstck/mlayax-darwin-arm64",
    // TASKS.md §5: tarball <= ~120 MiB compressed (npm 413s near 200 MB).
    budgetBytes: 120 * 1024 * 1024,
    kind: "tarball",
    allow: [
      /^package\.json$/,
      /^README\.md$/,
      /^LICENSE$/,
      /^NOTICE$/,
      /^VERSION$/,
      /^SHA256SUMS$/,
      /^lib\//,
      /^licenses\//,
    ],
  },
];

function pack(dir) {
  const out = execFileSync("npm", ["pack", "--dry-run", "--json"], {
    cwd: path.join(root, dir),
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
  });
  const [entry] = JSON.parse(out);
  if (!entry) throw new Error(`npm pack produced no entry for ${dir}`);
  return entry;
}

function human(bytes) {
  const mib = bytes / 1024 / 1024;
  if (mib >= 1) return `${mib.toFixed(2)} MiB`;
  return `${(bytes / 1024).toFixed(1)} KiB`;
}

let failed = false;

for (const target of TARGETS) {
  const entry = pack(target.dir);
  const measured = target.kind === "tarball" ? entry.size : entry.unpackedSize;
  const over = measured > target.budgetBytes;

  const unexpected = entry.files
    .map((f) => f.path)
    .filter((p) => !target.allow.some((re) => re.test(p)));

  const status = over || unexpected.length > 0 ? "FAIL" : "ok  ";
  console.log(
    `${status} ${target.name.padEnd(32)} ${human(measured)} / ${human(target.budgetBytes)} ` +
      `(${entry.entryCount} files, ${target.kind})`,
  );

  if (over) {
    console.error(`     over budget by ${human(measured - target.budgetBytes)}`);
    failed = true;
  }
  for (const p of unexpected) {
    console.error(`     unexpected file: ${p}`);
    failed = true;
  }
}

if (failed) {
  console.error("\nsize check failed");
  process.exit(1);
}
console.log("\nsize check passed");
