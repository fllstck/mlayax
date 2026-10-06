/**
 * Calibration and confidence maths.
 *
 * A port of the `laya_mlx.common` calibration helpers. These are pure functions of a probability
 * vector, kept out of the tensor path on purpose: the same numbers must be produced by the ONNX and
 * MLX runtimes, and they are the part of the pipeline a caller can reason about without a GPU.
 */

import { QTYPE_NAMES, type QuestionTypeId } from "./questions.js";

/**
 * Fitted temperatures below this sharpen the logits instead of softening them.
 *
 * Not cosmetic: the shipped `choice:11+` bucket is `0.1006`, which multiplies the logits ~10x and
 * publishes a 0.24 top probability as 0.99. No honest calibration needs to sharpen that hard, so a
 * temperature below the floor is clamped rather than applied. See `TEMP_MAX` for the other end.
 */
export const TEMP_MIN = 0.5;

/** Fitted temperatures above this soften the logits into near-uniformity. */
export const TEMP_MAX = 5.0;

/**
 * Coerce a temperature-like value to a number, or `null` if Python's `float()` would raise.
 *
 * Numeric strings are accepted (as `float("1.5")` is); booleans are not, because `float(True)`
 * succeeding is an accident of Python's type hierarchy that upstream explicitly guards against.
 * Exported because the load-time validation ("Calibration temperatures must be finite and
 * positive") needs the same coercion the clamp does — two spellings of it would drift.
 */
export function coerceTemperature(value: unknown): number | null {
  if (typeof value === "boolean") return null;
  if (typeof value === "number") return value;
  if (typeof value === "string" && value.trim() !== "") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

/**
 * Coerce a temperature into `[lo, hi]`, falling back to `1.0` when it is not a usable number.
 *
 * The fallback matters as much as the clamp: an unset temperature must mean "no scaling", not
 * "scale by zero" or "scale by NaN". `null`, `undefined`, booleans, empty strings, non-numeric
 * strings, arrays, objects, `NaN` and `±Infinity` all fall back to `1.0`.
 */
export function clampTemperature(t: unknown, lo = TEMP_MIN, hi = TEMP_MAX): number {
  const value = coerceTemperature(t);
  if (value === null || !Number.isFinite(value)) return 1.0;
  return Math.min(hi, Math.max(lo, value));
}

/**
 * Format with C's `%.<precision>g` semantics, which is what upstream's warning text uses.
 *
 * `%g` keeps `precision` significant digits, switches to exponent form when the decimal exponent is
 * `< -4` or `>= precision`, and strips trailing zeros. JavaScript has no equivalent, so the
 * exponent is recovered from `toExponential` and used to choose between positional and exponential
 * output.
 */
export function formatSignificant(value: number, precision = 4): string {
  if (!Number.isFinite(value)) return String(value);
  const exponent = Number(value.toExponential(0).split("e")[1] ?? "0");
  if (exponent < -4 || exponent >= precision) {
    const [mantissa = "0", rawExponent = "0"] = value.toExponential(precision - 1).split("e");
    const sign = rawExponent.startsWith("-") ? "-" : "+";
    const digits = rawExponent.replace(/^[+-]/, "").padStart(2, "0");
    return `${stripTrailingZeros(mantissa)}e${sign}${digits}`;
  }
  const decimals = Math.max(0, precision - 1 - exponent);
  return stripTrailingZeros(value.toFixed(decimals));
}

function stripTrailingZeros(text: string): string {
  if (!text.includes(".")) return text;
  return text.replace(/0+$/, "").replace(/\.$/, "");
}

/**
 * The checkpoint-supplied temperatures whose clamping changed the value, formatted as `name=value`.
 *
 * Order is part of the contract: per-bucket entries first in insertion order, then the positional
 * `temperature[i]` entries — the order the reference warning lists them in. Values that are not
 * numbers at all are skipped: they are a load-time error, not a clamp (upstream rejects them with
 * "Calibration temperatures must be finite and positive").
 */
export function rejectedTemperatureEntries(
  temperature: readonly unknown[],
  temperatureByOptions: Readonly<Record<string, unknown>>,
): string[] {
  const rejected: string[] = [];
  for (const [bucket, value] of Object.entries(temperatureByOptions)) {
    const coerced = coerceTemperature(value);
    if (coerced === null) continue;
    if (clampTemperature(coerced) !== coerced) {
      rejected.push(`${bucket}=${formatSignificant(coerced, 4)}`);
    }
  }
  temperature.forEach((value, index) => {
    const coerced = coerceTemperature(value);
    if (coerced === null) return;
    if (clampTemperature(coerced) !== coerced) {
      rejected.push(`temperature[${index}]=${formatSignificant(coerced, 4)}`);
    }
  });
  return rejected;
}

/**
 * The verbatim load-time warning for clamped temperatures, or `null` when nothing was clamped.
 *
 * Kept as an exact string because it is user-facing and explains a real hazard: a checkpoint whose
 * `choice:11+` temperature is `0.1006` publishes a coin flip as a near-certainty, so confidence
 * from that bucket must not be trusted after clamping.
 */
export function temperatureClampWarning(rejected: readonly string[]): string | null {
  if (rejected.length === 0) return null;
  return (
    "laya-mlx: this checkpoint ships temperatures outside " +
    `[${formatSignificant(TEMP_MIN, 6)}, ${formatSignificant(TEMP_MAX, 6)}] which would ` +
    `distort confidence; clamping ${rejected.join(", ")}. Treat confidence from the affected ` +
    "buckets as uncalibrated."
  );
}

/**
 * The calibration bucket key for a question: the type and the option count.
 *
 * Option counts are bucketed rather than exact because the checkpoint fits one temperature per
 * bucket (`choice:2`, `choice:3-5`, `choice:6-10`, `choice:11+`, `score:3-5`, `noul:2`), so a
 * 13-option question uses the same temperature as a 40-option one.
 */
export function tempBucket(qtype: QuestionTypeId, k: number): string {
  const size = k <= 2 ? "2" : k <= 5 ? "3-5" : k <= 10 ? "6-10" : "11+";
  return `${QTYPE_NAMES[qtype]}:${size}`;
}

/**
 * Normalized Shannon entropy confidence: `1 - H(p) / log(k)`.
 *
 * `1` means "all mass on one option", `0` means uniform. The `1e-12` floor inside the log keeps a
 * zero probability contributing zero instead of `NaN`; the result is clamped because floating
 * point can push it a hair outside `[0, 1]`.
 *
 * Note this is **not** the same quantity as {@link answerConfidence}, and the two must not be
 * compared against a shared threshold: this one is an entropy measure on `[0, 1]`, that one is the
 * raw top probability.
 */
export function confidenceFromProbs(p: number[], k: number): number {
  if (k < 2) return 1.0;
  const head = p.slice(0, k);
  const entropy = -head.reduce(
    (total, value) => total + value * Math.log(Math.min(Math.max(value, 1e-12), 1.0)),
    0,
  );
  return Math.min(1, Math.max(0, 1 - entropy / Math.log(k)));
}

/**
 * The probability mass on the reported answer: `max(p)` over the first `k` options.
 *
 * This is the quantity temperature scaling fits, and the quantity upstream's calibration figures
 * are computed on. It carries no accuracy guarantee by itself — the shipped checkpoints are
 * over-confident — so a caller gating on it must fit and validate a temperature first.
 */
export function answerConfidence(p: number[], k: number): number {
  if (k < 1) return 1.0;
  const head = p.slice(0, k);
  if (head.length === 0) return 0;
  return Math.min(1, Math.max(0, Math.max(...head)));
}

/** Decimal places in every published answer field. */
const ANSWER_DECIMALS = 4;

/**
 * Round to 4 decimals the way Python's `round(x, 4)` does.
 *
 * Python rounds the **exact** double, correctly rounded to 4 decimal places with ties to even.
 * `Number.prototype.toFixed` is specified the same way ("as close to zero as possible", so ties are
 * effectively unreachable for a binary double at 4 decimals), so this matches CPython on every value
 * tested — verified against `round(v, 4)` on 8 017 values, including near-tie cases.
 *
 * This replaces a tolerance-based formulation (`|x*1e4 - floor(x*1e4) - 0.5| < 1e-9` then round to
 * even) that treated any value merely *near* a half as an exact tie. That is wrong whenever the
 * double sits just above the half: it returned `0` for `0.00005` where Python returns `0.0001`, and
 * disagreed with Python on 120 of those 8 017 values. An answer's confidence is exactly the kind of
 * small number that lands there.
 *
 * Signed zero is preserved: Python's `round(-0.00004, 4)` is `-0.0`.
 */
export function round4(x: number): number {
  if (!Number.isFinite(x)) return x;
  const rounded = Number(x.toFixed(ANSWER_DECIMALS));
  if (rounded === 0 && (x < 0 || Object.is(x, -0))) return -0;
  return rounded;
}

/** Numerically stable softmax. Not used on the answer path — kept for callers scoring raw logits. */
export function softmax(z: number[]): number[] {
  const max = Math.max(...z);
  const exps = z.map((value) => Math.exp(value - max));
  const total = exps.reduce((a, b) => a + b, 0);
  return exps.map((value) => value / total);
}
