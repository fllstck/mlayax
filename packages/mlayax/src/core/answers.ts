/**
 * Answer shaping: the public answer object for one question.
 *
 * A port of the per-row block of `laya_mlx`'s `Agent.system_one`. Field names, order and rounding
 * are part of the published contract — a caller parsing answers, and the parity harness diffing
 * against the Python reference, both depend on them.
 */

import { answerConfidence, confidenceFromProbs, round4 } from "./calibration.js";
import type { InternalQuestion, QuestionKind } from "./questions.js";
import type { PrefixStats } from "./sequence.js";

/** Fields every answer carries. */
export interface AnswerBase {
  /** The question kind, echoed back. */
  type: QuestionKind;
  /**
   * Normalized entropy confidence on `[0, 1]` — for `choice` and `score`. For `noul` this is
   * instead `max(p, 1 - p)`, the mass on the reported side, because a two-option entropy score
   * would read as certainty for a coin flip.
   */
  confidence: number;
  /** Probability mass on the reported answer (`max(p)`). A different quantity from `confidence`. */
  answer_confidence: number;
  action: {
    /** Probability that the decision should be escalated rather than auto-applied. */
    act_probability: number;
  };
}

/** Answer to a `choice` question. */
export interface ChoiceAnswer extends AnswerBase {
  type: "choice";
  /** The winning label. */
  choice: string;
  /** Probability per label, keyed by label, in criteria order. */
  probabilities: Record<string, number>;
}

/** Answer to a `score` question. */
export interface ScoreAnswer extends AnswerBase {
  type: "score";
  /** Expected level: `sum(i * p_i)`, so a score question reads as a graded number. */
  score: number;
  /** Level index (`"0"`, `"1"`, …) to the criterion text, for display. */
  legend: Record<string, unknown>;
  /** Probability per level, keyed by the level index as a string. */
  probabilities: Record<string, number>;
}

/** Answer to a `noul` question. */
export interface NoulAnswer extends AnswerBase {
  type: "noul";
  /** Probability of the `true` side. */
  noul: number;
}

/** The answer to one question, discriminated on `type`. */
export type Answer = ChoiceAnswer | ScoreAnswer | NoulAnswer;

/**
 * Shape one calibrated probability vector into the public answer.
 *
 * `p` is the temperature-scaled softmax over the question's `k` options and must have at least `k`
 * entries; `actProb` is the escalation head's probability for this row.
 *
 * Ties on `choice` go to the **lowest index**, matching `argmax` in the reference (which returns
 * the first maximum) — not to the last one, as a strict `>` comparison written the other way round
 * would.
 */
export function shapeAnswer(q: InternalQuestion, p: number[], actProb: number): Answer {
  const k = p.length;
  const base = {
    type: q.t,
    confidence:
      q.t === "noul"
        ? round4(Math.max(p[1] ?? 0, 1 - (p[1] ?? 0)))
        : round4(confidenceFromProbs(p, k)),
    answer_confidence: round4(answerConfidence(p, k)),
    action: { act_probability: round4(actProb) },
  };

  if (q.t === "choice") {
    const labels = Object.keys((q.crit ?? {}) as Record<string, unknown>);
    let best = 0;
    for (let index = 1; index < p.length; index++) {
      if ((p[index] ?? 0) > (p[best] ?? 0)) best = index;
    }
    return {
      ...base,
      type: "choice",
      choice: labels[best] ?? String(best),
      probabilities: Object.fromEntries(labels.map((label, i) => [label, round4(p[i] ?? 0)])),
    };
  }

  if (q.t === "score") {
    const levels = Array.from({ length: k }, (_, i) => String(i));
    const criteria = Array.isArray(q.crit) ? q.crit : [];
    return {
      ...base,
      type: "score",
      score: round4(p.reduce((total, value, i) => total + i * value, 0)),
      legend: Object.fromEntries(levels.map((level, i) => [level, criteria[i]])),
      probabilities: Object.fromEntries(levels.map((level, i) => [level, round4(p[i] ?? 0)])),
    };
  }

  return { ...base, type: "noul", noul: round4(p[1] ?? 0) };
}

/** One question's option count vs its distinct rendered token spans. */
export interface CollapsedOptionReport {
  /** Options the question defines — **not** the markers that survived truncation. */
  total: number;
  /** Options whose token spans differ; `< total` means duplicates collapsed. */
  distinct: number;
  /** Per-option cap that was applied, or `null`. */
  tokens_per_option: number | null;
}

/**
 * The questions whose options no longer have a distinct token span each.
 *
 * Reported only when options actually collapsed, so a caller can tell a genuinely duplicated
 * rubric from a prompt that was silently collapsed to fit the head budget.
 *
 * `total` counts the options the question **defines**, not the markers that survived: a report
 * counted from the markers would claim 43 of 58 options made it into a prompt where 28 never did.
 */
export function collapsedOptions(
  questionIds: readonly string[],
  items: readonly { options?: PrefixStats | undefined }[],
): Record<string, CollapsedOptionReport> {
  const out: Record<string, CollapsedOptionReport> = {};
  questionIds.forEach((qid, index) => {
    const stats = items[index]?.options;
    if (stats === undefined || stats.options_distinct >= stats.options) return;
    out[qid] = {
      total: stats.options,
      distinct: stats.options_distinct,
      tokens_per_option: stats.tokens_per_option,
    };
  });
  return out;
}
