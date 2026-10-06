/**
 * Question normalisation and option rendering.
 *
 * A port of `laya_mlx.common`'s `render_options` / `resolve_noul_labels` and of
 * `Agent._to_internal`. Validation is deliberately as strict as upstream's: a question that
 * upstream rejects should be rejected here too, with a message that names the problem, rather than
 * producing a slightly different prompt and a slightly different answer.
 *
 * The rendering rules are load-bearing, and two of them are traps:
 *
 * - A choice label with no description renders as the **label alone** (`billing`), never as
 *   `billing: null`. The `: null` form is a bug we hit; `dict.fromkeys` produces `null` values, so
 *   it is the default path for list-shaped criteria, not a corner case.
 * - `0` and `false` are **legitimate criterion descriptions** and must still render as
 *   `label: 0` / `label: false`. Only `null`, `undefined` and `""` mean "no description". Testing
 *   truthiness here instead of explicit nullish/empty checks is how that gets broken.
 */

import { pyJson, renderCriterion } from "./ppjson.js";

/** The three question kinds the model is trained on. */
export type QuestionKind = "choice" | "score" | "noul";

/** `QTYPES` from upstream: the integer the model is fed for each question kind. */
export const QTYPES = { choice: 0, score: 1, noul: 2 } as const satisfies Record<
  QuestionKind,
  number
>;

/** The numeric question type id, `0 | 1 | 2`. */
export type QuestionTypeId = (typeof QTYPES)[QuestionKind];

/** Reverse of {@link QTYPES}, for prompt-independent labels such as `temp_bucket` keys. */
export const QTYPE_NAMES: Record<QuestionTypeId, QuestionKind> = {
  0: "choice",
  1: "score",
  2: "noul",
};

/** Display text for the two `noul` options. */
export interface NoulLabels {
  readonly false: string;
  readonly true: string;
}

/**
 * A question after normalisation. `crit` is `Record<string, unknown>` for `choice`, `unknown[]`
 * for `score`, and `Record<string, unknown> | null` for `noul`; it is left as `unknown` because
 * every consumer narrows on `t` first, and a dishonest union here would only need casting.
 */
export interface InternalQuestion {
  readonly t: QuestionKind;
  readonly ins: string;
  readonly crit: unknown;
  readonly labels?: NoulLabels;
}

const DEFAULT_NOUL_LABELS: NoulLabels = { false: "false", true: "true" };

const NOUL_LABEL_ERROR =
  "noul labels must map exactly 'false' and 'true' to distinct non-empty strings";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Validate a `noul` label pair and return it trimmed, in `[false, true]` semantic order.
 *
 * Both labels are trimmed, must be distinct and non-empty, and the map must carry exactly the two
 * keys `false` and `true` — no more, no fewer, and not the strings `"false"`/`"true"` used as
 * values for the wrong key.
 */
export function resolveNoulLabels(labels?: NoulLabels | undefined): readonly [string, string] {
  const raw: Record<string, unknown> =
    labels === undefined ? { ...DEFAULT_NOUL_LABELS } : { ...labels };
  if (Object.keys(raw).sort().join(",") !== "false,true") throw new Error(NOUL_LABEL_ERROR);
  const rawFalse = raw.false;
  const rawTrue = raw.true;
  if (typeof rawFalse !== "string" || typeof rawTrue !== "string")
    throw new Error(NOUL_LABEL_ERROR);
  const falseLabel = rawFalse.trim();
  const trueLabel = rawTrue.trim();
  if (!falseLabel || !trueLabel || falseLabel === trueLabel) throw new Error(NOUL_LABEL_ERROR);
  return [falseLabel, trueLabel] as const;
}

/**
 * Render the option texts in label-index order.
 *
 * For `noul` the order is always semantic `[false, true]`, whatever order the criteria object
 * happens to have — the marker order is what the model's probability vector is read against.
 */
export function renderOptions(q: InternalQuestion): string[] {
  if (q.t !== "noul" && q.labels !== undefined) {
    throw new Error("labels is only supported for noul questions");
  }

  if (q.t === "choice") {
    if (!isRecord(q.crit) || Object.keys(q.crit).length === 0) {
      throw new Error("Choice criteria must be a nonempty dictionary or list");
    }
    return Object.entries(q.crit).map(([label, description]) =>
      description === null || description === undefined || description === ""
        ? label
        : `${label}: ${renderCriterion(description)}`,
    );
  }

  if (q.t === "score") {
    if (!Array.isArray(q.crit) || q.crit.length === 0) {
      throw new Error("Score criteria must be a nonempty list");
    }
    return q.crit.map((criterion, level) => `level ${level}: ${renderCriterion(criterion)}`);
  }

  const crit: Record<string, unknown> = isRecord(q.crit) ? q.crit : {};
  const [falseLabel, trueLabel] = resolveNoulLabels(q.labels);
  const falseCrit = crit.false;
  const trueCrit = crit.true;
  return [
    `${falseLabel}: ${
      falseCrit === null || falseCrit === undefined || falseCrit === ""
        ? "no, the statement does not hold"
        : renderCriterion(falseCrit)
    }`,
    `${trueLabel}: ${
      trueCrit === null || trueCrit === undefined || trueCrit === ""
        ? "yes, the statement holds"
        : renderCriterion(trueCrit)
    }`,
  ];
}

/**
 * Normalise a public question definition into the internal form (`Agent._to_internal`).
 *
 * Accepts `unknown` and validates, so a malformed request body fails here with a described problem
 * instead of somewhere inside the model call.
 */
export function toInternal(definition: unknown): InternalQuestion {
  if (!isRecord(definition)) throw new Error("Each question must be a dictionary");

  const kind = definition.type;
  if (kind !== "choice" && kind !== "score" && kind !== "noul") {
    throw new Error(
      `Unknown question type ${JSON.stringify(kind)}; expected choice, score, or noul`,
    );
  }

  if (!Object.hasOwn(definition, "instructions")) {
    throw new Error("Question is missing instructions");
  }
  const rawInstructions = definition.instructions;
  if (rawInstructions === null || rawInstructions === undefined) {
    throw new Error("instructions must not be empty or None");
  }
  // Empty first, then blank: upstream distinguishes "" (empty) from "   " (blank).
  if (typeof rawInstructions === "string" && rawInstructions === "") {
    throw new Error("instructions must not be empty or None");
  }
  if (Array.isArray(rawInstructions) && rawInstructions.length === 0) {
    throw new Error("instructions must not be empty or None");
  }
  if (isRecord(rawInstructions) && Object.keys(rawInstructions).length === 0) {
    throw new Error("instructions must not be empty or None");
  }
  if (typeof rawInstructions === "string" && rawInstructions.trim() === "") {
    throw new Error("instructions must not be blank");
  }
  const instructionsSupported =
    typeof rawInstructions === "string" ||
    typeof rawInstructions === "number" ||
    typeof rawInstructions === "boolean" ||
    Array.isArray(rawInstructions) ||
    isRecord(rawInstructions);
  if (!instructionsSupported) {
    throw new Error("instructions must be JSON-serializable text or structured data");
  }

  const criteria = normaliseCriteria(kind, definition.criteria);

  const internal: {
    t: QuestionKind;
    ins: string;
    crit: unknown;
    labels?: NoulLabels;
  } = {
    t: kind,
    ins: typeof rawInstructions === "string" ? rawInstructions : pyJson(rawInstructions),
    crit: criteria,
  };

  if (Object.hasOwn(definition, "labels")) {
    if (kind !== "noul") throw new Error("labels is only supported for noul questions");
    const labels = definition.labels;
    if (!isRecord(labels)) throw new Error(NOUL_LABEL_ERROR);
    resolveNoulLabels(labels as unknown as NoulLabels);
    internal.labels = labels as unknown as NoulLabels;
  }

  return internal;
}

function normaliseCriteria(kind: QuestionKind, criteria: unknown): unknown {
  if (kind === "choice") {
    if (Array.isArray(criteria)) {
      if (!criteria.every((label) => typeof label === "string")) {
        throw new Error("Choice labels must be strings");
      }
      if (new Set(criteria).size !== criteria.length) {
        throw new Error("Choice labels must be unique");
      }
      // `dict.fromkeys`: no description. Rendered as the bare label.
      return Object.fromEntries(criteria.map((label) => [label, null]));
    }
    if (!isRecord(criteria) || Object.keys(criteria).length === 0) {
      throw new Error("Choice criteria must be a nonempty dictionary or list");
    }
    return criteria;
  }

  if (kind === "score") {
    if (!Array.isArray(criteria) || criteria.length === 0) {
      throw new Error("Score criteria must be a nonempty list");
    }
    if (criteria.some((level) => level === null || level === undefined)) {
      throw new Error("Score criteria must not contain a null level");
    }
    return criteria;
  }

  if (criteria === null || criteria === undefined) return null;
  if (!isRecord(criteria)) {
    throw new Error("Noul criteria must be a dictionary with false/true descriptions");
  }
  const lowered = Object.fromEntries(
    Object.entries(criteria).map(([key, value]) => [key.toLowerCase(), value]),
  );
  for (const key of Object.keys(lowered)) {
    if (key !== "false" && key !== "true") {
      throw new Error(
        "Noul criteria must be keyed only 'false'/'true'; use labels to change display text",
      );
    }
  }
  return lowered;
}
