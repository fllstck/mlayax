/**
 * Which machine the bit-exact claims were measured on, and what "exact" means elsewhere.
 *
 * TASKS.md §10.14. The parity suite pins fp32 **bit-exactness** against the Python reference, and that
 * is a claim about a *machine* as much as about the code: MLX lowers a kernel per GPU family, and a
 * different accumulator order is a different sum. Measured on GitHub's macOS runner (`Apple M2 Pro`)
 * versus the reference machine (`Apple M5`):
 *
 * - the tiny fixture's `action.act_probability` moved by **3.0e-4** — three steps at 4 decimals;
 * - `probabilities`, `choice`, `score`, `confidence` and `answer_confidence` were **exact**;
 * - the real checkpoint's parity test is opt-in and runs on the reference machine, where it is Δ 0.
 *
 * So the strong claim is asserted *on the reference machine*, and everywhere else exactly one field is
 * allowed to move, by a bound that was measured rather than guessed. Nothing is skipped: on another
 * machine the suite still fails if anything but the action head drifts, and it reports the delta it did
 * see, which is the part a regression would change.
 *
 * `isReferenceMachine` takes the CPU as an argument so both branches are testable without swapping
 * hardware — the same shape as `selectBinding` in `vendor/node-mlx/native-binding.cjs`.
 *
 * The machine name is shared with `bench/baseline.json`, which refuses to issue a bench *verdict* off
 * the reference CPU for the same reason. One difference is deliberate: the bench gate refuses to judge,
 * while these tests judge with a measured allowance, because the tiny fixture still proves the
 * *decisions* on any GPU and that is worth having in CI.
 */

import { execFileSync } from "node:child_process";

/** The field that moves, as a path suffix: `<question>.action.act_probability`. */
export const ACTION_PROBABILITY_FIELD = "action.act_probability";

/** The machine in `bench/baseline.json`, whose kernel lowering the numbers were measured against. */
export const REFERENCE_CPU = "Apple M5";

/**
 * The one field class that moves off the reference GPU, and the bound it is allowed.
 *
 * Measured on GitHub's macOS runner: `3.0000000000002247e-4` in `action.act_probability` — three steps
 * at 4 decimals, and *slightly above* 3e-4 in binary, which is why the bound is one step of headroom
 * rather than the measured number: a bound set exactly at the measurement fails intermittently, and a
 * flaky gate is worse than a slightly loose one. The field is the escalation hint, not the answer.
 */
export const ACTION_PROBABILITY_DRIFT = 4e-4;

/** `sysctl -n machdep.cpu.brand_string`, or `null` off macOS. */
export function cpuBrand(): string | null {
  try {
    return execFileSync("sysctl", ["-n", "machdep.cpu.brand_string"], { encoding: "utf8" }).trim();
  } catch {
    return null;
  }
}

/** True when `cpu` is the machine the bit-exact numbers were measured on. */
export function isReferenceMachine(cpu: string | null = cpuBrand()): boolean {
  return cpu === REFERENCE_CPU;
}

/**
 * The extra tolerance a field gets on this machine, on top of the caller's.
 *
 * Pure, and deliberately narrow: `0` for every field except the action head on a machine other than
 * the reference one. A caller that passes `tolerance = 0` therefore still asserts that everything the
 * product reports is exact — the action head is the one exception, and it is the escalation hint rather
 * than the answer.
 */
export function driftForField(path: string, cpu: string | null = cpuBrand()): number {
  if (isReferenceMachine(cpu)) return 0;
  return path === ACTION_PROBABILITY_FIELD || path.endsWith(`.${ACTION_PROBABILITY_FIELD}`)
    ? ACTION_PROBABILITY_DRIFT
    : 0;
}

/** The `compare` option this machine implies, so callers do not each re-derive it. */
export function driftPolicy(cpu: string | null = cpuBrand()): (path: string) => number {
  return (path) => driftForField(path, cpu);
}

/** A short "which machine was this" note for a failure message: `Apple M5` / `Apple M2 Pro`. */
export function machineNote(cpu: string | null = cpuBrand()): string {
  const name = cpu ?? "an unknown CPU";
  return isReferenceMachine(cpu)
    ? `${name} (the reference machine)`
    : `${name} (not the reference machine — see TASKS.md §10.14)`;
}
