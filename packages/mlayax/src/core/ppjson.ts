/**
 * Python-compatible JSON rendering.
 *
 * A port of the `json.dumps(..., ensure_ascii=False)` behaviour that `laya_mlx` relies on when it
 * turns structured state and criteria into prompt text. This matters more than it looks: the
 * rendered string is tokenized, so any difference in a separator, a float's spelling, or the way
 * `None` is written changes the token ids and therefore the answer.
 *
 * Two places where JavaScript's `JSON.stringify` is *not* a drop-in for Python's `json.dumps`:
 *
 * 1. **Separators.** `JSON.stringify` is compact (`{"a":1,"b":2}`); Python's default is
 *    `{"a": 1, "b": 2}`. Handled below by building the string ourselves.
 * 2. **Floats.** JS `String(2.0)` is `"2"`, Python's `repr(2.0)` is `"2.0"`. JS also switches to
 *    exponent notation at different thresholds (`1e-7` / `1e21`) than Python (`1e-05` / `1e+16`),
 *    and pads the exponent differently. See {@link pyFloatRepr}.
 */

/**
 * Render a float the way Python's `repr` does, so `2.0` stays `"2.0"` and `1e-5` becomes `"1e-05"`.
 *
 * Both languages print the shortest round-tripping decimal, so the digits already agree; what
 * differs is the *format*:
 *
 * | value | Python `repr` | JS `String` | here |
 * |---|---|---|---|
 * | `2.0` | `2.0` | `2` | `2.0` |
 * | `1e-5` | `1e-05` | `0.00001` | `1e-05` |
 * | `1e16` | `1e+16` | `10000000000000000` | `1e+16` |
 * | `1.5e-8` | `1.5e-08` | `1.5e-8` | `1.5e-08` |
 *
 * Python uses positional notation while the decimal exponent is in `[-4, 16)` and exponential
 * outside it — which is exactly what the branch below does, since JS only disagrees about the
 * *thresholds* and the exponent's zero padding, not about the digits.
 *
 * Non-finite values keep Python's `json.dumps` spelling (`NaN`, `Infinity`, `-Infinity`) rather
 * than JS's `null`, because Python does not silently turn them into `null`.
 *
 * Prefer {@link pyNumberRepr} for JSON serialization, since it also has to guess about
 * {@link pyNumberRepr int-vs-float}.
 */
export function pyFloatRepr(value: number): string {
  if (Number.isNaN(value)) return "NaN";
  if (value === Number.POSITIVE_INFINITY) return "Infinity";
  if (value === Number.NEGATIVE_INFINITY) return "-Infinity";
  if (Object.is(value, -0)) return "-0.0";

  // Integral values below the exponential threshold are printed with a trailing ".0".
  if (Number.isInteger(value) && Math.abs(value) < 1e16) return `${value}.0`;

  const [mantissa = "0", exponentText = "0"] = value.toExponential().split("e");
  const exponent = Number(exponentText);
  if (exponent < -4 || exponent >= 16) {
    const sign = exponent < 0 ? "-" : "+";
    const digits = String(Math.abs(exponent)).padStart(2, "0");
    return `${mantissa}e${sign}${digits}`;
  }

  // Inside [-4, 16) JS and Python agree on the positional spelling.
  return String(value);
}

/**
 * Render a JSON number, guessing Python's `int` vs `float` from the value alone.
 *
 * **This is the one place the port cannot be exact.** Python's `json` has two number types and
 * prints them differently — `json.dumps(2)` is `"2"` while `json.dumps(2.0)` is `"2.0"` — but
 * JavaScript has a single `number`, and `JSON.parse` throws away whether the source literal had a
 * decimal point. So an integral value is rendered the way Python renders an **`int`** (`"2"`),
 * and a fractional value the way Python renders a **`float`** (`"2.5"`).
 *
 * The residual ambiguity is an integral value that arrived as a float literal (`{"amount": 2.0}`
 * in a request body renders as `2`, where Python would say `2.0`). Integers are the common case in
 * rubric criteria and structured state, and rendering them with a spurious `.0` would be wrong far
 * more often — so the choice is deliberate, not an oversight. Callers who need exactness for
 * integral floats can send them as strings.
 */
export function pyNumberRepr(value: number): string {
  if (Number.isInteger(value) && Math.abs(value) < 1e21) return String(value);
  return pyFloatRepr(value);
}

/**
 * `json.dumps(value, ensure_ascii=False)` with Python's default `", "` / `": "` separators.
 *
 * `undefined` is treated as `null`, matching how a missing property reaches us from a JSON body.
 * Object keys keep their insertion order, as `dict` does.
 */
export function pyJson(value: unknown): string {
  if (value === null || value === undefined) return "null";
  if (typeof value === "string") return JSON.stringify(value);
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number") return pyNumberRepr(value);
  if (Array.isArray(value)) return `[${value.map(pyJson).join(", ")}]`;
  if (typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>);
    return `{${entries.map(([k, v]) => `${JSON.stringify(k)}: ${pyJson(v)}`).join(", ")}}`;
  }
  // Functions, symbols and nothing else Python would accept. `bigint` throws in JSON.stringify,
  // which mirrors Python's TypeError on a non-serializable value.
  return JSON.stringify(value) ?? "null";
}

/**
 * The state as prompt text: strings pass through untouched, anything structured becomes JSON.
 *
 * Unchanged from `laya_mlx.common.serialize_state`.
 */
export function serializeState(state: unknown): string {
  return typeof state === "string" ? state : pyJson(state);
}

/**
 * Render one criterion value as text.
 *
 * Strings pass through; anything structured becomes compact JSON, so a rubric reads as JSON rather
 * than a JavaScript object literal. Without this a dict-valued criterion leaked `[object Object]`
 * (or `{'desc': ...}` in the Python original) into the prompt.
 */
export function renderCriterion(value: unknown): string {
  return typeof value === "string" ? value : pyJson(value);
}
