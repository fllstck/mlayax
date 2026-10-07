/**
 * The bench gate — TASKS.md §6.
 *
 *   MLAYAX_MODEL_DIR=/path/to/english-mlx npm run bench:check
 *
 * Runs `bench/bench.ts` in a child process and compares its p50s against `bench/baseline.json`:
 * within 1.3x is fine, above 2x fails. The child process is deliberate — a measurement taken in this
 * process would inherit its JIT state and its resident MLX, neither of which a number in §2 had to
 * contend with.
 *
 * Two things this gate refuses to do quietly:
 *
 * - **Compare across hardware.** 1.3x is a claim about *comparable* hardware (§2). The reference
 *   machine is recorded in the baseline, and if this is not it, the numbers are reported as
 *   incomparable and no verdict is issued — a passing or failing verdict from a different CPU would
 *   be a lie either way.
 * - **Pass without measuring.** No checkpoint, no build, or not macOS: it exits non-zero and says
 *   which, rather than reporting success. This is an opt-in gate, so a silent pass is indistinguishable
 *   from an unchecked one.
 *
 * The RSS ceiling is checked too: §2 puts one resident model at ~0.95–1.0 GiB, and a leak in the
 * forward path shows up here long before it shows up as a wrong answer.
 */

import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, "..");

interface Baseline {
  referenceMachine: { cpu: string; os: string };
  method: { iterations: number };
  source: string;
  cases: Record<string, { baselineMs: number; what: string }>;
  tolerance: { warnRatio: number; failRatio: number };
  rssMiB: { baseline: number; ceiling: number };
}

interface BenchResult {
  runtime: string;
  loadMs: number;
  oneQuestion: { p50: number; min: number };
  threeQuestions: { p50: number; min: number };
  sixteenRows: { p50: number; min: number; qps: number };
  rssMiB: number;
}

/**
 * The baseline to judge against.
 *
 * `MLAYAX_BENCH_BASELINE` overrides the path, which exists for two reasons: it lets a candidate
 * baseline be evaluated before it is committed (the "refresh deliberately" workflow in §6 wants the
 * ratios *before* the number lands), and it is what makes this gate's own thresholds testable —
 * pointing at a stricter baseline is how the >1.3x and >2x branches get exercised without editing
 * the real file or waiting for a genuine 3x regression.
 */
const baselinePath = process.env.MLAYAX_BENCH_BASELINE ?? path.join(here, "baseline.json");
const baseline = JSON.parse(readFileSync(baselinePath, "utf8")) as Baseline;

/** The measured p50 for a baseline case key, or `null` when the harness did not report it. */
function measuredFor(result: BenchResult, key: string): number | null {
  if (key === "oneQuestion") return result.oneQuestion.p50;
  if (key === "threeQuestions") return result.threeQuestions.p50;
  if (key === "sixteenRows") return result.sixteenRows.p50;
  return null;
}

function fail(message: string): never {
  console.error(`\nbench gate failed: ${message}`);
  process.exit(1);
}

// ---------------------------------------------------------------------------------------------
// Preconditions. Each one is a refusal to judge, said out loud.
// ---------------------------------------------------------------------------------------------
if (process.platform !== "darwin") {
  fail(
    `this gate is macOS-only (TASKS.md §6); running on ${process.platform}. The MLX payload is ` +
      `Apple-Silicon only, so there is nothing here to measure.`,
  );
}

const modelDir = process.env.MLAYAX_MODEL_DIR;
if (modelDir === undefined || modelDir === "") {
  fail(
    "MLAYAX_MODEL_DIR is not set. The gate measures the real checkpoint on purpose: a synthetic " +
      "fixture has a different sequence length and a different number of forward passes, so its " +
      "latency cannot be compared to the §2 table. Set it to the checkpoint directory and re-run.",
  );
}

const benchEntry = path.join(here, "bench.ts");
const resultJson = ((): BenchResult => {
  try {
    const stdout = execFileSync(
      process.execPath,
      [benchEntry, "--json", "--iterations", String(baseline.method.iterations)],
      { cwd: repoRoot, encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] },
    );
    return JSON.parse(stdout.trim().split("\n").at(-1) ?? "") as BenchResult;
  } catch (error) {
    const stderr = (error as { stderr?: string }).stderr?.trim();
    return fail(
      `the harness did not produce a measurement.\n${stderr === undefined || stderr === "" ? "" : `\n${stderr}\n`}  ` +
        `Note that bench/bench.ts imports the built package, so run \`npm run build\` first ` +
        `(\`npm run bench:check\` does it for you).`,
    );
  }
})();

// ---------------------------------------------------------------------------------------------
// Is this the machine the baseline describes?
// ---------------------------------------------------------------------------------------------
function sysctl(key: string): string | null {
  try {
    return execFileSync("sysctl", ["-n", key], { encoding: "utf8" }).trim();
  } catch {
    return null;
  }
}

function osVersion(): string | null {
  try {
    return execFileSync("sw_vers", ["-productVersion"], { encoding: "utf8" }).trim();
  } catch {
    return null;
  }
}

const cpu = sysctl("machdep.cpu.brand_string") ?? "unknown";
const os = osVersion() ?? "unknown";
const comparable = cpu === baseline.referenceMachine.cpu;

console.log(`bench gate  runtime=${resultJson.runtime}  load=${resultJson.loadMs}ms  rss=${resultJson.rssMiB} MiB`);
console.log(`  machine: ${cpu}, macOS ${os}`);
if (!comparable) {
  console.log(
    `  note: the baseline is ${baseline.referenceMachine.cpu} on macOS ` +
      `${baseline.referenceMachine.os}. This is a different CPU, so the ratios below are reported ` +
      `for information and are NOT a verdict.`,
  );
}
console.log(`  baseline: ${baseline.source}`);
if (baselinePath !== path.join(here, "baseline.json")) {
  console.log(`  (from ${baselinePath}, via MLAYAX_BENCH_BASELINE)`);
}
console.log("");

let breaches = 0;
let warnings = 0;

for (const [key, entry] of Object.entries(baseline.cases)) {
  const measured = measuredFor(resultJson, key);
  if (measured === null) fail(`the harness reported no p50 for "${key}"`);

  const ratio = measured / entry.baselineMs;
  const verdict =
    ratio > baseline.tolerance.failRatio
      ? `FAIL  > ${baseline.tolerance.failRatio}x`
      : ratio > baseline.tolerance.warnRatio
        ? `warn  > ${baseline.tolerance.warnRatio}x`
        : "ok";
  if (ratio > baseline.tolerance.failRatio) breaches += 1;
  else if (ratio > baseline.tolerance.warnRatio) warnings += 1;

  console.log(
    `  ${verdict.padEnd(12)} ${key.padEnd(15)} ${measured.toFixed(1).padStart(6)} ms  ` +
      `baseline ${entry.baselineMs.toFixed(1)} ms  ${ratio.toFixed(2)}x   ${entry.what}`,
  );
}

const rssOverCeiling = resultJson.rssMiB > baseline.rssMiB.ceiling;
console.log(
  `  ${(rssOverCeiling ? "FAIL" : "ok").padEnd(12)} ${"rss".padEnd(15)} ` +
    `${String(resultJson.rssMiB).padStart(6)} MiB  ceiling ${baseline.rssMiB.ceiling} MiB   ` +
    `(§2: ${baseline.rssMiB.baseline} MiB for one resident model)`,
);
if (rssOverCeiling) breaches += 1;

console.log("");
console.log(
  `  16-row throughput: ${resultJson.sixteenRows.qps.toFixed(0)} q/s ` +
    `(§2: ~271 q/s, plateau ~312 q/s at 8 rows and above)`,
);

if (!comparable) {
  // Deliberately still exit 0: this is not a pass, it is a "no verdict" — but a hard failure here
  // would make the gate unusable on any machine other than the reference one, which is worse.
  console.log("\nbench gate: inconclusive (different CPU from the baseline). Reported, not judged.");
  process.exit(0);
}

if (breaches > 0) {
  console.log(
    `\nbench gate: FAILED (${breaches} over ${baseline.tolerance.failRatio}x, ${warnings} over ` +
      `${baseline.tolerance.warnRatio}x). See TASKS.md §6: a regression here is either a kernel ` +
      `regression or the wrong MLX build class, and §8.7 is how you tell them apart.`,
  );
  process.exit(1);
}

console.log(
  warnings > 0
    ? `\nbench gate: pass, with ${warnings} case(s) above ${baseline.tolerance.warnRatio}x — ` +
        `within the 2x hard limit, but worth a look before a release.`
    : `\nbench gate: pass (all cases within ${baseline.tolerance.warnRatio}x).`,
);