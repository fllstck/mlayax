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

/** @type {{ dir: string, name: string, budgetBytes: number, kind: "unpacked" | "tarball", allow: RegExp[], forbid?: { pattern: RegExp, why: string }[] }[]} */
const TARGETS = [
  {
    dir: "packages/mlayax",
    name: "@fllstck/mlayax",
    // §0 said "< 300 KB"; §10.4 records the history — two raises, both caused by comments, and the
    // diagnosis that the budget was in effect a cap on how much the code is explained.
    //
    // Resolved 2026-10-07 (Phase 8) by removing the cause rather than the cap: 113.4 KiB of that
    // budget was `*.map` files, and in the *installed* package they resolve to nothing. They carry
    // no `sourcesContent`, and their `sources` point at `../../src/**` — which the `files` allowlist
    // does not ship. So the payload was 237.5 KiB once the maps went, and is 252.9 KiB now that the
    // READMEs and the JSDoc fixes are written — inside §0's 300 KiB, with ~47 KiB of real headroom.
    // The maps are still gone; what grew was documentation, which is the trade §10.4 argues for. To reverse: drop `"!dist/**/*.map"` from the package's `files`
    // and raise this back to 360 KiB — but only together with actually shipping the sources, or the
    // maps stay dead weight.
    budgetBytes: 300 * 1024,
    kind: "unpacked",
    allow: [
      /^package\.json$/,
      /^README\.md$/,
      /^LICENSE$/,
      /^NOTICE$/,
      /^licenses\//,
      /^dist\//,
      /^vendor\//,
    ],
    forbid: [
      {
        pattern: /\.map$/,
        why:
          "source maps are emitted for local use but not published: their `sources` point at " +
          "`src/**`, which the tarball does not contain, and they carry no `sourcesContent`, so a " +
          "consumer cannot resolve them (TASKS.md §10.12)",
      },
    ],
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

  // `allow` catches files nobody meant to ship. `forbid` catches files that were removed on purpose
  // and drifted back in — a decision nothing else would notice, since the budget alone would absorb
  // them silently.
  const forbidden = [];
  for (const rule of target.forbid ?? []) {
    for (const file of entry.files) {
      if (rule.pattern.test(file.path)) forbidden.push({ path: file.path, why: rule.why });
    }
  }

  const status = over || unexpected.length > 0 || forbidden.length > 0 ? "FAIL" : "ok  ";
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
  if (forbidden.length > 0) {
    console.error(
      `     forbidden: ${forbidden.length} file(s) starting with ${forbidden[0].path}\n` +
        `       ${forbidden[0].why}`,
    );
    failed = true;
  }
}

if (failed) {
  console.error("\nsize check failed");
  process.exit(1);
}
console.log("\nsize check passed");
