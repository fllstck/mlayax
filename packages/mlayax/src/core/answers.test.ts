import { describe, expect, it } from "vitest";
import { collapsedOptions, shapeAnswer } from "./answers.js";
import { toInternal } from "./questions.js";
import type { PrefixStats } from "./sequence.js";

const choice = (criteria: string[]) =>
  toInternal({ type: "choice", instructions: "Which team?", criteria });

describe("shapeAnswer", () => {
  it("shapes a choice answer with the winning label and per-label probabilities", () => {
    const answer = shapeAnswer(
      choice(["billing", "technical", "sales"]),
      [0.9585, 0.0227, 0.0186],
      0.5,
    );
    expect(answer).toEqual({
      type: "choice",
      confidence: 0.8173,
      answer_confidence: 0.9585,
      action: { act_probability: 0.5 },
      choice: "billing",
      probabilities: { billing: 0.9585, technical: 0.0227, sales: 0.0186 },
    });
  });

  it("shapes a score answer as an expected level, with a legend", () => {
    const q = toInternal({
      type: "score",
      instructions: "How urgent?",
      criteria: ["low", "mid", "high"],
    });
    const answer = shapeAnswer(q, [0.1534, 0.3343, 0.5123], 0.25);
    expect(answer).toEqual({
      type: "score",
      confidence: 0.0929,
      answer_confidence: 0.5123,
      action: { act_probability: 0.25 },
      score: 1.3589,
      legend: { 0: "low", 1: "mid", 2: "high" },
      probabilities: { 0: 0.1534, 1: 0.3343, 2: 0.5123 },
    });
  });

  it("shapes a noul answer as the probability of the true side", () => {
    const q = toInternal({ type: "noul", instructions: "Does the customer ask for money back?" });
    const answer = shapeAnswer(q, [0.1781, 0.8219], 0.5);
    expect(answer).toEqual({
      type: "noul",
      // For noul, `confidence` is the mass on the reported side, not an entropy score: a two-option
      // entropy measure would read as certainty for a coin flip.
      confidence: 0.8219,
      answer_confidence: 0.8219,
      action: { act_probability: 0.5 },
      noul: 0.8219,
    });
  });

  it("reports entropy confidence for choice and score, not the top probability", () => {
    const answer = shapeAnswer(choice(["a", "b", "c"]), [1 / 3, 1 / 3, 1 / 3], 0);
    expect(answer.confidence).toBe(0);
    expect(answer.answer_confidence).toBe(0.3333);
  });

  it("breaks probability ties towards the lowest index, as argmax does", () => {
    const answer = shapeAnswer(choice(["first", "second", "third"]), [0.5, 0.5, 0], 0);
    expect(answer.type === "choice" && answer.choice).toBe("first");
  });

  it("returns no scaling when k is 1, because there is nothing to be uncertain about", () => {
    const answer = shapeAnswer(choice(["only"]), [1], 0);
    expect(answer.confidence).toBe(1);
    expect(answer.answer_confidence).toBe(1);
  });

  it("rounds every published field to four decimals", () => {
    const answer = shapeAnswer(choice(["a", "b"]), [0.123456789, 0.876543211], 0.11111111);
    expect(answer.answer_confidence).toBe(0.8765);
    expect(answer.action.act_probability).toBe(0.1111);
    expect(answer.type === "choice" && answer.probabilities.a).toBe(0.1235);
    expect(answer.type === "choice" && answer.probabilities.b).toBe(0.8765);
  });

  it("keeps the published field order, which is part of the contract", () => {
    const answer = shapeAnswer(choice(["a", "b"]), [0.6, 0.4], 0.1);
    expect(Object.keys(answer)).toEqual([
      "type",
      "confidence",
      "answer_confidence",
      "action",
      "choice",
      "probabilities",
    ]);
  });

  it("keeps the noul field order too", () => {
    const answer = shapeAnswer(toInternal({ type: "noul", instructions: "?" }), [0.4, 0.6], 0.1);
    expect(Object.keys(answer)).toEqual([
      "type",
      "confidence",
      "answer_confidence",
      "action",
      "noul",
    ]);
  });

  it("scores a single-level question at level zero rather than NaN", () => {
    const q = toInternal({ type: "score", instructions: "?", criteria: ["only"] });
    const answer = shapeAnswer(q, [1], 0);
    expect(answer.type === "score" && answer.score).toBe(0);
  });

  describe("defensive paths when the probability vector is short", () => {
    // The model should always return one probability per option, and the sequence builder rejects a
    // question whose markers were dropped. These paths exist so that a short vector degrades to 0
    // rather than to NaN or a crash, and they are worth pinning: NaN in `confidence` would silently
    // poison a caller's gate.

    it("reports 0 for labels with no probability, rather than omitting them", () => {
      const answer = shapeAnswer(choice(["a", "b", "c"]), [0.7], 0.1);
      expect(answer.type === "choice" && answer.probabilities).toEqual({
        a: 0.7,
        b: 0,
        c: 0,
      });
      expect(answer.type === "choice" && answer.choice).toBe("a");
    });

    it("picks the first label when the vector is empty", () => {
      const answer = shapeAnswer(choice(["a", "b"]), [], 0);
      expect(answer.type === "choice" && answer.choice).toBe("a");
      // With no options there is nothing to be uncertain about, so both confidences report 1 —
      // the same convention as a one-option question.
      expect(answer.confidence).toBe(1);
      expect(answer.answer_confidence).toBe(1);
      expect(answer.type === "choice" && answer.probabilities).toEqual({ a: 0, b: 0 });
    });

    it("falls back to the index when a label is missing", () => {
      // Only reachable if `crit` is not a record, which `toInternal` already prevents; the point is
      // that the answer path cannot throw on it.
      const q = { t: "choice" as const, ins: "?", crit: null };
      const answer = shapeAnswer(q, [0.3, 0.7], 0);
      expect(answer.type === "choice" && answer.choice).toBe("1");
    });

    it("keys score levels off the probability vector, not the criteria list", () => {
      // `k` is `p.length`, so a short vector yields fewer levels rather than an extra level at 0.
      // The criteria list only supplies the legend text.
      const q = toInternal({ type: "score", instructions: "?", criteria: ["low", "mid", "high"] });
      const answer = shapeAnswer(q, [0.5, 0.5], 0);
      expect(answer.type === "score" && answer.probabilities).toStrictEqual({ 0: 0.5, 1: 0.5 });
      expect(answer.type === "score" && answer.legend).toStrictEqual({ 0: "low", 1: "mid" });
      expect(answer.type === "score" && answer.score).toBe(0.5);
    });

    it("survives a score question whose criteria are not an array", () => {
      const q = { t: "score" as const, ins: "?", crit: "nonsense" };
      const answer = shapeAnswer(q, [1, 0], 0);
      // toStrictEqual, not toEqual: the point is that the keys exist and are undefined rather than
      // being absent from the legend, so the level indices stay aligned with the probabilities.
      expect(answer.type === "score" && answer.legend).toStrictEqual({
        0: undefined,
        1: undefined,
      });
    });

    it("reports 0 for a noul vector with no true side", () => {
      const q = toInternal({ type: "noul", instructions: "?" });
      const answer = shapeAnswer(q, [], 0);
      expect(answer.type === "noul" && answer.noul).toBe(0);
      expect(answer.confidence).toBe(1);
    });
  });
});

describe("collapsedOptions", () => {
  const stats = (options: number, options_distinct: number, per: number | null): PrefixStats => ({
    options,
    options_distinct,
    tokens_per_option: per,
  });

  it("reports only questions whose options actually collapsed", () => {
    expect(
      collapsedOptions(
        ["a", "b", "c"],
        [
          { options: stats(3, 3, null) },
          { options: stats(12, 12, 4) },
          { options: stats(4, 2, 5) },
        ],
      ),
    ).toEqual({ c: { total: 4, distinct: 2, tokens_per_option: 5 } });
  });

  it("reports the options the question defines, not the markers that survived", () => {
    // A report counted from the markers would claim 12 of 43 options made it into a prompt where
    // 28 never did.
    const report = collapsedOptions(["many"], [{ options: stats(43, 43, 8) }]);
    expect(report).toEqual({});
    expect(collapsedOptions(["many"], [{ options: stats(43, 12, 8) }]).many?.total).toBe(43);
  });

  it("is empty when nothing collapsed, and tolerates missing per-question stats", () => {
    expect(collapsedOptions(["a"], [{ options: stats(3, 3, null) }])).toEqual({});
    expect(collapsedOptions(["a", "b"], [{ options: stats(3, 3, null) }])).toEqual({});
    expect(collapsedOptions(["a"], [{}])).toEqual({});
  });
});
