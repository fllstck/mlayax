#!/usr/bin/env node

/**
 * Parity smoke test against an *installed* copy of the package — TASKS.md §6, the `pack` job.
 *
 * Every other parity test imports the workspace source (`packages/mlayax/src/index.js`), so none of
 * them would notice a packaging mistake: a missing `dist/` file, a `files` allowlist that dropped
 * `vendor/`, an `exports` map that resolves to nothing. This script exists to be run from a scratch
 * project where the only `@fllstck/mlayax` on disk is the one that came out of a tarball, and it
 * resolves the package from *there*:
 *
 *   node scripts/packed-parity.mjs --project /tmp/clean-room
 *
 * Resolution is `createRequire(project)` rather than a relative import on purpose — a bare
 * `import("@fllstck/mlayax")` inside this file would resolve against this repository's own
 * `node_modules`, which is exactly the copy we are trying not to test.
 *
 * The workload is the committed tiny fixture (no download, no checkpoint), judged with §2's
 * tolerances: fp32 **bit-exact** (Δ 0 on every numeric field), fp16 within 4e-4. It prints the
 * resolved entry point and the addon it loaded, so a green run is evidence about which files were
 * exercised and not merely that nothing threw.
 *
 * Usage:
 *   node scripts/packed-parity.mjs [options]
 *
 *     --project DIR     where to resolve @fllstck/mlayax from (default: cwd)
 *     --fixture DIR     tiny checkpoint directory (default: test/fixtures/tiny)
 *     --reference FILE  reference payload (default: test/fixtures/ref/tiny-fp32.json)
 *     --dtype NAME      float32 | float16 (default: from the reference payload's `dtype`)
 */

import { execFileSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, "..");

/** §2's tolerances. fp32 is bit-exact; fp16 is allowed to drift by 4e-4. */
const TOLERANCE = { float32: 0, float16: 4e-4 };

/**
 * The one field that moves off the reference GPU, and by how much (TASKS.md §10.14).
 *
 * This script is a CI gate on a macOS runner, whose GPU family is not the M5 the fixture's Δ 0 was
 * measured against: there `action.act_probability` moved by 3.0000000000002247e-4 while `probabilities`,
 * `choice` and `confidence` stayed exact. Same policy as `test/helpers/machine.ts`, and it is only
 * consulted away from the reference machine, so this stays a strict bit-exact check where the claim was
 * measured. The bound carries one step of headroom above the measurement: a bound set exactly at it
 * fails intermittently, and a flaky gate is worse than a slightly loose one.
 */
const ACTION_PROBABILITY_FIELD = "action.act_probability";
const ACTION_PROBABILITY_DRIFT = 4e-4;
const REFERENCE_CPU = "Apple M5";

function isReferenceMachine() {
  try {
    const cpu = execFileSync("sysctl", ["-n", "machdep.cpu.brand_string"], {
      encoding: "utf8",
    }).trim();
    return cpu === REFERENCE_CPU;
  } catch {
    return false;
  }
}

function driftForPath(path, strict) {
  if (strict) return 0;
  return path === ACTION_PROBABILITY_FIELD || path.endsWith(`.${ACTION_PROBABILITY_FIELD}`)
    ? ACTION_PROBABILITY_DRIFT
    : 0;
}

/** `load()` takes `"float16"`/`"float32"`; the reference payloads record the same spelling. */
const DTYPE_ALIASES = {
  fp16: "float16",
  fp32: "float32",
  float16: "float16",
  float32: "float32",
};

const USAGE = [
  "Usage: node scripts/packed-parity.mjs [options]",
  "",
  "  --project DIR     where to resolve @fllstck/mlayax from (default: cwd)",
  "  --fixture DIR     tiny checkpoint directory (default: test/fixtures/tiny)",
  "  --reference FILE  reference payload (default: test/fixtures/ref/tiny-fp32.json)",
  "  --dtype NAME      float32 | float16 (default: from the reference payload's `dtype`)",
].join("\n");

function usage(message) {
  if (message !== undefined) console.error(`packed-parity: ${message}\n`);
  console.error(USAGE);
  process.exit(message === undefined ? 0 : 2);
}

function parseArgs(argv) {
  const options = {
    project: process.cwd(),
    fixture: path.join(repoRoot, "test", "fixtures", "tiny"),
    reference: path.join(repoRoot, "test", "fixtures", "ref", "tiny-fp32.json"),
    dtype: null,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    const value = () => {
      const next = argv[i + 1];
      if (next === undefined) usage(`${arg} needs a value`);
      i += 1;
      return next;
    };
    if (arg === "--project") options.project = path.resolve(value());
    else if (arg === "--fixture") options.fixture = path.resolve(value());
    else if (arg === "--reference") options.reference = path.resolve(value());
    else if (arg === "--dtype") options.dtype = value();
    else if (arg === "-h" || arg === "--help") usage();
    else usage(`unknown option ${arg}`);
  }
  if (options.dtype !== null && DTYPE_ALIASES[options.dtype] === undefined) {
    usage(`unknown --dtype ${options.dtype} (expected float32 or float16)`);
  }
  return options;
}

/**
 * Resolve the package the way a consumer's process would, then import it.
 *
 * The resolved path is returned too: "which copy did this actually test" is the whole point of the
 * script, and printing it is what turns a green job into evidence. (`import.meta.resolve`'s second
 * — parent — argument is deliberately not used: without `--experimental-import-meta-resolve` Node
 * ignores it and silently resolves against *this* file's directory, i.e. against the repository we
 * are trying not to test. That mistake was caught only because this script prints its source.)
 *
 * Resolution is therefore two explicit steps, both the consumer-facing ones:
 *
 *   1. find the installed package via `require.resolve("@fllstck/mlayax/package.json")` from the
 *      project directory — the same lookup `vendor/node-mlx/native-binding.cjs` does to find the
 *      platform package, and it needs the manifest's `./package.json` export to exist;
 *   2. follow that manifest's own entry point (`exports["."].import`), which is also how `publint`
 *      and `attw` are told what to check.
 *
 * A CommonJS `require.resolve("@fllstck/mlayax")` is *not* used and would fail: the package is
 * deliberately ESM-only (`exports: { ".": { import } }`, so `attw` runs with `--profile esm-only`).
 */
async function importPackageFrom(project) {
  const require_ = createRequire(path.join(project, "packed-parity-resolver.cjs"));

  let manifestPath;
  try {
    manifestPath = require_.resolve("@fllstck/mlayax/package.json");
  } catch (error) {
    console.error(
      `packed-parity: could not find @fllstck/mlayax from ${project}.\n` +
        "  Install the tarballs there first (npm i ./fllstck-mlayax-0.1.0.tgz …), or pass --project.\n" +
        `  ${error instanceof Error ? error.message : String(error)}`,
    );
    process.exit(1);
  }

  const packageDir = path.dirname(manifestPath);
  const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
  const dot = manifest.exports?.["."];
  const entryPoint =
    (typeof dot === "string" ? dot : (dot?.import ?? dot?.default)) ?? manifest.main ?? "index.js";
  const entry = path.resolve(packageDir, typeof entryPoint === "string" ? entryPoint : "index.js");

  if (!existsSync(entry)) {
    console.error(
      `packed-parity: ${path.relative(project, entry)} is what ${manifest.name}'s manifest points at, ` +
        "but it is not on disk — the tarball's `files` allowlist dropped it.",
    );
    process.exit(1);
  }

  const module = await import(pathToFileURL(entry).href);
  return { module, entry, packageDir };
}

/**
 * Compare `got` against `want` recursively — a plain-JS port of `test/helpers/scorecard.ts`, kept
 * separate because this script must not import anything TypeScript or repository-relative: it runs
 * against a scratch install, not against this checkout.
 */
function compare(want, got, tolerance, drift = () => 0) {
  const result = { compared: 0, exact: 0, maxDelta: 0, mismatches: [] };
  const walk = (a, b, level) => {
    if (typeof a === "number") {
      result.compared += 1;
      const allowed = tolerance + drift(level);
      if (typeof b !== "number") {
        result.mismatches.push(`${level}: want number ${a}, got ${JSON.stringify(b)}`);
        return;
      }
      const delta = Math.abs(a - b);
      if (delta > result.maxDelta) result.maxDelta = delta;
      if (delta === 0) result.exact += 1;
      else if (delta > allowed) {
        result.mismatches.push(`${level}: want ${a} got ${b} (Δ${delta.toExponential(2)})`);
      }
      return;
    }
    if (Array.isArray(a)) {
      result.compared += 1;
      if (JSON.stringify(a) === JSON.stringify(b)) result.exact += 1;
      else result.mismatches.push(`${level}: want ${JSON.stringify(a)} got ${JSON.stringify(b)}`);
      return;
    }
    if (a !== null && typeof a === "object") {
      for (const [key, value] of Object.entries(a)) {
        walk(value, b?.[key], level === "" ? key : `${level}.${key}`);
      }
      return;
    }
    result.compared += 1;
    if (JSON.stringify(a) === JSON.stringify(b)) result.exact += 1;
    else result.mismatches.push(`${level}: want ${JSON.stringify(a)} got ${JSON.stringify(b)}`);
  };
  walk(want, got, "");
  return result;
}

const options = parseArgs(process.argv.slice(2));

if (!existsSync(path.join(options.fixture, "model.safetensors"))) {
  console.error(
    `packed-parity: no tiny checkpoint at ${options.fixture} (model.safetensors missing)`,
  );
  process.exit(2);
}
if (!existsSync(options.reference)) {
  console.error(`packed-parity: no reference payload at ${options.reference}`);
  process.exit(2);
}

const reference = JSON.parse(readFileSync(options.reference, "utf8"));
const strict = isReferenceMachine();
// An unrecognised `dtype` in the payload is not silently trusted: fp32 is the strict case, so that
// is the safe fallback, and the reason is said out loud rather than inferred from a passing run.
const recorded = DTYPE_ALIASES[reference.dtype];
if (options.dtype === null && recorded === undefined) {
  console.error(
    `packed-parity: reference payload declares an unknown dtype ${JSON.stringify(reference.dtype)}; ` +
      "judging as float32 (bit-exact)",
  );
}
const dtype = options.dtype === null ? (recorded ?? "float32") : DTYPE_ALIASES[options.dtype];
const tolerance = TOLERANCE[dtype];

const { module: mlayax, entry, packageDir } = await importPackageFrom(options.project);
const { load, resolveNativeAddonPath, VERSION } = mlayax;

console.log(`packed parity  entry=${entry}`);
console.log(`  package ${packageDir}`);
console.log(
  `  version ${VERSION}  dtype ${dtype}  tolerance ${tolerance}  cases ${reference.cases.length}`,
);
try {
  console.log(`  addon   ${resolveNativeAddonPath()}`);
} catch (error) {
  console.error(`  addon: ${error instanceof Error ? error.message : String(error)}`);
  process.exit(1);
}

const agent = load(options.fixture, { dtype });
let failed = 0;

for (const testCase of reference.cases) {
  const prediction = await agent.predict(testCase.state, testCase.questions);
  const result = compare(testCase.answers, prediction.answers, tolerance, (path) =>
    driftForPath(path, strict),
  );
  const label = `${testCase.label ?? "(unlabelled)"} (${dtype})`;
  const line =
    `  ${result.mismatches.length === 0 ? "ok  " : "FAIL"} ${label}: ` +
    `${result.compared} fields, ${result.exact} exact, max Δ ${result.maxDelta.toExponential(2)}`;
  console.log(line);
  for (const mismatch of result.mismatches.slice(0, 20)) console.log(`       ${mismatch}`);
  if (result.mismatches.length > 20) console.log(`       … ${result.mismatches.length - 20} more`);
  // "Δ 0" is a stronger claim than "nothing exceeded the tolerance", so fp32 asserts it directly —
  // on the machine it was measured on. Elsewhere only the action head may move, by the measured
  // drift, and anything else drifting has already shown up as a mismatch.
  const exactness = strict ? 0 : ACTION_PROBABILITY_DRIFT;
  if (tolerance === 0 && result.maxDelta > exactness) {
    console.log(
      `       fp32 must be bit-exact${strict ? "" : " up to the measured action-head drift"}: ` +
        `${result.compared - result.exact} field(s) moved, max Δ ${result.maxDelta.toExponential(2)}`,
    );
    failed += 1;
  } else if (result.mismatches.length > 0) {
    failed += 1;
  }
}

if (failed > 0) {
  console.error(`\npacked parity FAILED: ${failed} case(s)`);
  process.exit(1);
}
console.log("\npacked parity passed");
