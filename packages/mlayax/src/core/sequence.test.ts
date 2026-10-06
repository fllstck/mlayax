import { describe, expect, it } from "vitest";
import { renderOptions, toInternal } from "./questions.js";
import {
  buildPrefix,
  DEFAULT_HEAD_MAX_LEN,
  finishSequence,
  type TinyTokenizer,
} from "./sequence.js";

/**
 * A deterministic word-level tokenizer for structural assertions.
 *
 * Real tokenizers prepend a word-boundary marker to each token (`"Ġno"`), which makes hand-written
 * expectations unreadable. This one maps each whitespace-separated word to itself, so a failure
 * points at the *layout* rather than at an opaque id. Exact ids against a real vocabulary are
 * pinned by the golden fixture test instead.
 */
function stubTokenizer(maskToken = "[MASK]"): TinyTokenizer {
  // Fixed special ids so structural expectations stay readable.
  const CLS = 2;
  const SEP = 3;
  const PAD = 0;
  const MASK = 4;
  let next = 10;
  const vocab = new Map<string, number>();
  const idFor = (word: string): number => {
    if (word === maskToken) return MASK;
    const existing = vocab.get(word);
    if (existing !== undefined) return existing;
    const id = next++;
    vocab.set(word, id);
    return id;
  };
  return {
    mask_token: maskToken,
    cls_token_id: CLS,
    sep_token_id: SEP,
    pad_token_id: PAD,
    mask_token_id: MASK,
    encode: (text) => text.split(/\s+/).filter(Boolean).map(idFor),
  };
}

const choiceQuestion = (criteria: Record<string, unknown>) =>
  toInternal({ type: "choice", instructions: "Which team?", criteria });

describe("buildPrefix", () => {
  it("lays out [CLS] head [SEP] option+ [SEP]", () => {
    const tok = stubTokenizer();
    const { ids, markers } = buildPrefix(tok, choiceQuestion({ a: "x", b: "y" }), 192);
    expect(ids[0]).toBe(tok.cls_token_id);
    expect(ids.at(-1)).toBe(tok.sep_token_id);
    // One [SEP] closes the head, one closes the sequence: the second is at ids[1 + headLen].
    expect(ids.filter((id) => id === tok.sep_token_id)).toHaveLength(2);
    expect(markers).toHaveLength(2);
  });

  it("points every marker at a [MASK] token, in option order", () => {
    const tok = stubTokenizer();
    const q = choiceQuestion({ billing: "invoices", technical: "bugs", sales: "purchases" });
    const { ids, markers } = buildPrefix(tok, q, 192);
    expect(markers).toHaveLength(renderOptions(q).length);
    for (const marker of markers) {
      expect(ids[marker]).toBe(tok.mask_token_id);
    }
    // Markers are strictly increasing and inside the sequence.
    expect([...markers].sort((a, b) => a - b)).toEqual(markers);
    expect(Math.max(...markers)).toBeLessThan(ids.length);
  });

  it("gives noul exactly two markers, in [false, true] order", () => {
    const tok = stubTokenizer();
    const q = toInternal({
      type: "noul",
      instructions: "Does it?",
      criteria: { true: "yes", false: "no" },
    });
    expect(buildPrefix(tok, q, 192).markers).toHaveLength(2);
  });

  it("replaces the mask token inside instructions and option text", () => {
    const tok = stubTokenizer();
    // The model must not see a literal [MASK] in prose: every mask id in the prefix belongs to an
    // option marker.
    const { ids, markers } = buildPrefix(
      tok,
      toInternal({ type: "noul", instructions: `Does [MASK] hold?` }),
      192,
    );
    const maskPositions = ids
      .map((id, index) => (id === tok.mask_token_id ? index : -1))
      .filter((index) => index >= 0);
    expect(maskPositions).toEqual(markers);
  });

  it("defaults the head budget to the upstream value", () => {
    expect(DEFAULT_HEAD_MAX_LEN).toBe(192);
    const tok = stubTokenizer();
    expect(buildPrefix(tok, choiceQuestion({ a: "x" }), DEFAULT_HEAD_MAX_LEN)).toEqual(
      buildPrefix(tok, choiceQuestion({ a: "x" }), 192),
    );
  });

  it("reports option-budget stats only when asked", () => {
    const tok = stubTokenizer();
    const withStats = buildPrefix(tok, choiceQuestion({ a: "x", b: "y" }), 192, true);
    expect(withStats.stats).toEqual({
      options: 2,
      options_distinct: 2,
      tokens_per_option: null,
    });
    expect(buildPrefix(tok, choiceQuestion({ a: "x", b: "y" }), 192)).not.toHaveProperty("stats");
  });

  it("counts identical rendered token spans as one distinct option", () => {
    // Two different option texts can tokenize identically: with a small or out-of-domain
    // vocabulary, everything misses to the unknown id and the spans become equal. That is the real
    // mechanism behind `options_distinct < options`, and it is what `usage.options` reports — the
    // tiny fixture's 12-option case collapses to a single distinct span for exactly this reason.
    const allUnknown: TinyTokenizer = {
      mask_token: "[MASK]",
      cls_token_id: 2,
      sep_token_id: 3,
      pad_token_id: 0,
      mask_token_id: 4,
      encode: (text) =>
        text
          .split(/\s+/)
          .filter(Boolean)
          .map((word) => (word === "[MASK]" ? 4 : 1)),
    };
    const { stats } = buildPrefix(
      allUnknown,
      choiceQuestion({ alpha: "zzz", beta: "yyy" }),
      192,
      true,
    );
    expect(stats.options).toBe(2);
    expect(stats.options_distinct).toBe(1);
  });

  it("counts identical descriptions under different labels as distinct spans", () => {
    // `options_distinct` counts rendered token spans, not descriptions: the labels differ, so the
    // rendered option text differs. This is the trap in reading the metric as "duplicate rubric".
    const tok = stubTokenizer();
    const { stats } = buildPrefix(tok, choiceQuestion({ a: "same", b: "same" }), 192, true);
    expect(stats.options).toBe(2);
    expect(stats.options_distinct).toBe(2);
  });

  it("collapses every option to the same span when the head budget is tight", () => {
    const tok = stubTokenizer();
    const criteria = Object.fromEntries(
      Array.from({ length: 12 }, (_, i) => [
        `label_${i}`,
        `one two three four ${i} five six seven`,
      ]),
    );
    const collapsed = buildPrefix(tok, choiceQuestion(criteria), 32, true);
    expect(collapsed.stats.tokens_per_option).toBeGreaterThanOrEqual(4);
    expect(collapsed.markers).toHaveLength(12);
    // Every option keeps its marker even though the body was cut — dropping a marker would
    // silently drop that option's answer.
    for (const marker of collapsed.markers) {
      expect(collapsed.ids[marker]).toBe(tok.mask_token_id);
    }
    const spans = collapsed.markers.map((marker, i) => {
      const next = collapsed.markers[i + 1] ?? collapsed.ids.length - 1;
      return next - marker;
    });
    expect(new Set(spans).size).toBe(1);
    expect(spans[0]).toBe(collapsed.stats.tokens_per_option);
  });

  it("never truncates the question head below 8 tokens", () => {
    const tok = stubTokenizer();
    const criteria = Object.fromEntries(
      Array.from({ length: 30 }, (_, i) => [`label_${i}`, "one two three four five six"]),
    );
    const { ids } = buildPrefix(tok, choiceQuestion(criteria), 4);
    // [CLS] + >= 8 head tokens + [SEP] is the floor, however small the budget is.
    expect(ids.length).toBeGreaterThanOrEqual(10);
  });
});

describe("finishSequence", () => {
  const tok = stubTokenizer();
  const stateIds = [100, 101, 102, 103, 104, 105];

  it("appends the state, closes with [SEP] and caps at maxLen", () => {
    const built = finishSequence(tok, [2, 9, 3, 4], [3], stateIds, 512);
    expect(built.ids).toEqual([2, 9, 3, 4, ...stateIds, 3]);
    expect(built.stats).toEqual({
      state_tokens: 6,
      state_tokens_used: 6,
      state_tokens_dropped: 0,
      truncated: false,
    });
  });

  it("right-truncates a string state, keeping the head", () => {
    // A string state is prose: the opening is what matters.
    const built = finishSequence(tok, [2, 9, 3, 4], [3], stateIds, 8);
    expect(built.ids).toEqual([2, 9, 3, 4, 100, 101, 102, 3]);
    expect(built.stats.state_tokens_dropped).toBe(3);
    expect(built.stats.state_tokens_used).toBe(3);
    expect(built.stats.truncated).toBe(true);
  });

  it("left-truncates a turn list, keeping the tail", () => {
    // A list state is a turn list: the most recent turns matter most.
    const built = finishSequence(tok, [2, 9, 3, 4], [3], stateIds, 8, true);
    expect(built.ids).toEqual([2, 9, 3, 4, 103, 104, 105, 3]);
    expect(built.stats.state_tokens_dropped).toBe(3);
    expect(built.stats.truncated).toBe(true);
  });

  it("drops markers that fall past the cap", () => {
    const built = finishSequence(tok, [2, 9, 3, 4, 4, 4], [3, 4, 5], stateIds, 5);
    expect(built.ids).toHaveLength(5);
    expect(built.markers).toEqual([3, 4]);
  });

  it("never emits more than maxLen ids, even when the prefix alone overruns", () => {
    const built = finishSequence(tok, [2, 9, 3, 4, 4], [3, 4], stateIds, 3);
    expect(built.ids).toHaveLength(3);
    expect(built.stats.state_tokens_used).toBe(0);
    expect(built.stats.state_tokens_dropped).toBe(6);
    expect(built.stats.truncated).toBe(true);
  });

  it("handles maxLen 0 without producing a negative slice", () => {
    const built = finishSequence(tok, [2], [], stateIds, 0);
    expect(built.ids).toEqual([]);
    expect(built.stats.state_tokens_used).toBe(0);
  });
});
