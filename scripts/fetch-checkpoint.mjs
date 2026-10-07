#!/usr/bin/env node
/**
 * Fetch the real checkpoint into the Hugging Face cache and print its path.
 *
 * This exists for the one gate that cannot run against the committed tiny fixture: the benchmark
 * (TASKS.md §6 — `npm run bench:check`, which needs `MLAYAX_MODEL_DIR`). It is also the only
 * end-to-end exercise of the Hub fetcher against the real API outside the opt-in live test, and the
 * only place the *cold download* path of Phase 3b runs the way a user experiences it.
 *
 * It is a script rather than a test so CI can wire it into a cache step:
 *
 *   MLAYAX_MODEL_DIR="$(node scripts/fetch-checkpoint.mjs)"   # path only, on stdout
 *
 * Stdout carries the snapshot directory and nothing else — everything a human wants to read (the
 * resolved revision, per-file progress, "already cached") goes to stderr, so the substitution above
 * cannot pick up a progress line. Pass `--offline` to require a warm cache instead of downloading.
 *
 * Needs `npm run build` first: it imports the built package, like `bench/bench.ts` does.
 *
 * Usage:
 *   node scripts/fetch-checkpoint.mjs [options]
 *
 *     --repo ID       repository id (default: aac6fef/laya-mlx, the checkpoint §2 measured)
 *     --revision REV  branch, tag, or 40-hex commit (default: the repo's default branch)
 *     --cache DIR     cache root, i.e. what HF_HUB_CACHE points at (default: the Python default)
 *     --offline       never touch the network; fail if the cache is cold
 *     --endpoint URL  Hub base URL (default: https://huggingface.co)
 */

import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, "..");
const DIST_ENTRY = path.join(repoRoot, "packages", "mlayax", "dist", "index.js");

const DEFAULT_REPO = "aac6fef/laya-mlx";

const USAGE = [
  "Usage: node scripts/fetch-checkpoint.mjs [options]",
  "",
  "  --repo ID       repository id (default: aac6fef/laya-mlx, the checkpoint §2 measured)",
  "  --revision REV  branch, tag, or 40-hex commit (default: the repo's default branch)",
  "  --cache DIR     cache root, i.e. what HF_HUB_CACHE points at (default: the Python default)",
  "  --offline       never touch the network; fail if the cache is cold",
  "  --endpoint URL  Hub base URL (default: https://huggingface.co)",
].join("\n");

function usage(message) {
  if (message !== undefined) console.error(`fetch-checkpoint: ${message}\n`);
  console.error(USAGE);
  process.exit(message === undefined ? 0 : 2);
}

const options = {
  repo: DEFAULT_REPO,
  revision: undefined,
  cache: undefined,
  offline: false,
  endpoint: undefined,
};
const argv = process.argv.slice(2);
for (let i = 0; i < argv.length; i += 1) {
  const arg = argv[i];
  const value = () => {
    const next = argv[i + 1];
    if (next === undefined) usage(`${arg} needs a value`);
    i += 1;
    return next;
  };
  if (arg === "--repo") options.repo = value();
  else if (arg === "--revision") options.revision = value();
  else if (arg === "--cache") options.cache = path.resolve(value());
  else if (arg === "--offline") options.offline = true;
  else if (arg === "--endpoint") options.endpoint = value();
  else if (arg === "-h" || arg === "--help") usage();
  else usage(`unknown option ${arg}`);
}

if (!existsSync(DIST_ENTRY)) {
  usage(`no built package at ${DIST_ENTRY} — run \`npm run build\` first`);
}

const { downloadModel, resolveCachedModel } = await import(pathToFileURL(DIST_ENTRY).href);

const hubOptions = {
  ...(options.revision === undefined ? {} : { revision: options.revision }),
  ...(options.cache === undefined ? {} : { cacheDir: options.cache }),
  ...(options.endpoint === undefined ? {} : { endpoint: options.endpoint }),
  offline: options.offline,
};

if (options.offline) {
  const cached = resolveCachedModel(options.repo, hubOptions);
  if (cached === null) {
    console.error(
      `fetch-checkpoint: no cached snapshot of ${options.repo} (and --offline was given)`,
    );
    process.exit(1);
  }
  console.error(`fetch-checkpoint: using the cached snapshot at ${cached.revision}`);
  console.log(cached.path);
  process.exit(0);
}

let lastPhase = "";
const resolved = await downloadModel(options.repo, {
  ...hubOptions,
  onProgress: (event) => {
    // One line per phase per file, on stderr, so a long download is not silent but stdout stays clean.
    const file = event.file === undefined ? "" : ` ${event.file}`;
    const key = `${event.phase}${file}${event.filesTotal}`;
    if (key === lastPhase || event.phase === "done") return;
    lastPhase = key;
    console.error(
      `fetch-checkpoint: ${event.phase}${file} ${event.bytesDone}/${event.bytesTotal} bytes`,
    );
  },
});
console.error(
  `fetch-checkpoint: ${resolved.fromCache ? "warm cache" : "downloaded"} ${options.repo}@` +
    `${resolved.revision} (${resolved.files} files) -> ${resolved.path}`,
);
console.log(resolved.path);
