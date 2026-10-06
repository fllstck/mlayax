/**
 * Field-by-field comparison against the Python reference payloads.
 *
 * A typed port of the spike's `src/compare.ts`. The tolerance policy is the one TASKS.md pins: fp32
 * must be **bit-exact** against `laya_mlx` (Δ 0 on every one of the 61 numeric fields per case), and
 * fp16 within `4e-4`. Strings (the chosen `choice` label) must match exactly in both.
 *
 * Reporting every mismatch rather than the first is deliberate: when a port drifts, the *pattern* of
 * which fields drifted is the diagnosis, and seeing only the first one hides it.
 */

import { readFileSync } from "node:fs";

/** One reference case: an input pair and the answers `laya_mlx` produced for it. */
export interface RefCase {
  label: string;
  state: unknown;
  questions: Record<string, unknown>;
  answers: Record<string, unknown>;
}

/** A reference payload file. */
export interface Reference {
  dtype: string;
  cases: RefCase[];
}

/** Read a reference payload written by `tools/reference/laya_ref.py`. */
export function loadReference(path: string): Reference {
  return JSON.parse(readFileSync(path, "utf8")) as Reference;
}

/** The outcome of comparing one payload against a reference case. */
export interface Comparison {
  compared: number;
  exact: number;
  maxDelta: number;
  /** Human-readable mismatches, worst first only in the sense of "all of them, capped". */
  mismatches: string[];
  ok: boolean;
}

/**
 * Compare `got` against `want` recursively.
 *
 * Numbers are compared with `tolerance`; everything else is compared by JSON equality, because the
 * only non-numeric fields in the answer payload are the question type and the chosen label, and both
 * must match exactly.
 */
export function compare(want: unknown, got: unknown, tolerance: number, path = ""): Comparison {
  const result: Comparison = { compared: 0, exact: 0, maxDelta: 0, mismatches: [], ok: true };
  walk(want, got, tolerance, path, result);
  result.ok = result.mismatches.length === 0;
  return result;
}

function walk(
  want: unknown,
  got: unknown,
  tolerance: number,
  path: string,
  result: Comparison,
): void {
  if (typeof want === "number") {
    result.compared += 1;
    if (typeof got !== "number") {
      result.mismatches.push(`${path}: want number ${want}, got ${JSON.stringify(got)}`);
      return;
    }
    const delta = Math.abs(want - got);
    if (delta > result.maxDelta) result.maxDelta = delta;
    if (delta === 0) result.exact += 1;
    else if (delta > tolerance) {
      result.mismatches.push(`${path}: want ${want} got ${got} (Δ${delta.toExponential(2)})`);
    }
    return;
  }

  if (Array.isArray(want)) {
    // Arrays are not used in the payload today; compare structurally if they ever are.
    result.compared += 1;
    if (JSON.stringify(want) !== JSON.stringify(got)) {
      result.mismatches.push(`${path}: want ${JSON.stringify(want)} got ${JSON.stringify(got)}`);
    } else {
      result.exact += 1;
    }
    return;
  }

  if (want !== null && typeof want === "object") {
    const gotRecord = (got ?? {}) as Record<string, unknown>;
    for (const [key, value] of Object.entries(want as Record<string, unknown>)) {
      walk(value, gotRecord[key], tolerance, path === "" ? key : `${path}.${key}`, result);
    }
    return;
  }

  result.compared += 1;
  if (JSON.stringify(want) === JSON.stringify(got)) result.exact += 1;
  else result.mismatches.push(`${path}: want ${JSON.stringify(want)} got ${JSON.stringify(got)}`);
}

/** Format a comparison for a test failure message. */
export function describeComparison(label: string, comparison: Comparison): string {
  const head =
    `${label}: ${comparison.compared} fields compared, ${comparison.exact} exact, ` +
    `max Δ ${comparison.maxDelta.toExponential(2)}`;
  if (comparison.ok) return head;
  return [head, ...comparison.mismatches.slice(0, 20).map((line) => `  ${line}`)].join("\n");
}
