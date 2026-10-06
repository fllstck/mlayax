import { describe, expect, it } from "vitest";
import { QTYPE_NAMES, QTYPES, renderOptions, resolveNoulLabels, toInternal } from "./questions.js";

const choice = (criteria: unknown, instructions = "Which one?") =>
  toInternal({ type: "choice", instructions, criteria });

describe("toInternal", () => {
  it("maps the three question kinds to their model ids", () => {
    expect(QTYPES).toEqual({ choice: 0, score: 1, noul: 2 });
    expect(QTYPE_NAMES).toEqual({ 0: "choice", 1: "score", 2: "noul" });
    expect(choice({ a: "x", b: "y" }).t).toBe("choice");
    expect(toInternal({ type: "score", instructions: "How?", criteria: ["a"] }).t).toBe("score");
    expect(toInternal({ type: "noul", instructions: "Does it?" }).t).toBe("noul");
  });

  it("keeps a string instruction verbatim and JSON-encodes a structured one", () => {
    expect(choice({ a: "x" }, "Which team?").ins).toBe("Which team?");
    expect(
      toInternal({ type: "choice", instructions: { ask: "Which?" }, criteria: { a: "x" } }).ins,
    ).toBe('{"ask": "Which?"}');
    expect(
      toInternal({ type: "choice", instructions: ["one", "two"], criteria: { a: "x" } }).ins,
    ).toBe('["one", "two"]');
  });

  it("turns list criteria into labels with no description (dict.fromkeys)", () => {
    const q = choice(["alpha", "beta", "gamma"]);
    expect(q.crit).toEqual({ alpha: null, beta: null, gamma: null });
    expect(renderOptions(q)).toEqual(["alpha", "beta", "gamma"]);
  });

  it("lowercases noul criteria keys but not choice or score", () => {
    expect(
      toInternal({ type: "noul", instructions: "?", criteria: { TRUE: "yes", False: "no" } }).crit,
    ).toEqual({ true: "yes", false: "no" });
    expect(choice({ Upper: "x" }).crit).toEqual({ Upper: "x" });
  });

  it("leaves noul criteria as null when omitted", () => {
    expect(toInternal({ type: "noul", instructions: "?" }).crit).toBeNull();
  });

  it("rejects malformed questions", () => {
    expect(() => toInternal(null)).toThrow(/must be a dictionary/);
    expect(() => toInternal("nope")).toThrow(/must be a dictionary/);
    expect(() => toInternal({ instructions: "x" })).toThrow(/Unknown question type/);
    expect(() => toInternal({ type: "rate", instructions: "x" })).toThrow(/Unknown question type/);
    expect(() => toInternal({ type: "choice", criteria: { a: "x" } })).toThrow(
      /missing instructions/,
    );
  });

  it("rejects empty and blank instructions, as upstream does", () => {
    expect(() => toInternal({ type: "choice", instructions: null, criteria: { a: "x" } })).toThrow(
      /must not be empty/,
    );
    expect(() => toInternal({ type: "choice", instructions: "", criteria: { a: "x" } })).toThrow(
      /must not be empty/,
    );
    expect(() => toInternal({ type: "choice", instructions: "   ", criteria: { a: "x" } })).toThrow(
      /must not be blank/,
    );
    expect(() => toInternal({ type: "choice", instructions: [], criteria: { a: "x" } })).toThrow(
      /must not be empty/,
    );
    expect(() => toInternal({ type: "choice", instructions: {}, criteria: { a: "x" } })).toThrow(
      /must not be empty/,
    );
  });

  it("rejects criteria that upstream rejects", () => {
    expect(() => choice(undefined)).toThrow(/nonempty dictionary or list/);
    expect(() => choice({})).toThrow(/nonempty dictionary or list/);
    expect(() => choice("billing")).toThrow(/nonempty dictionary or list/);
    expect(() => choice(["a", "a"])).toThrow(/must be unique/);
    expect(() => choice([1, 2])).toThrow(/must be strings/);
    expect(() => toInternal({ type: "score", instructions: "?", criteria: [] })).toThrow(
      /nonempty list/,
    );
    expect(() => toInternal({ type: "score", instructions: "?", criteria: "one" })).toThrow(
      /nonempty list/,
    );
    expect(() => toInternal({ type: "score", instructions: "?", criteria: [null] })).toThrow(
      /null level/,
    );
    expect(() => toInternal({ type: "noul", instructions: "?", criteria: { maybe: "?" } })).toThrow(
      /keyed only/,
    );
    expect(() => toInternal({ type: "noul", instructions: "?", criteria: ["a"] })).toThrow(
      /dictionary with false\/true/,
    );
  });

  it("stops labels being smuggled onto a non-noul question", () => {
    expect(() =>
      toInternal({
        type: "choice",
        instructions: "?",
        criteria: { a: "x" },
        labels: { false: "no", true: "yes" },
      }),
    ).toThrow(/labels is only supported for noul/);
  });
});

describe("renderOptions", () => {
  it("renders choice criteria as 'label: description'", () => {
    expect(choice({ billing: "invoices, payments", technical: "bugs" })).toBeDefined();
    expect(renderOptions(choice({ billing: "invoices, payments", technical: "bugs" }))).toEqual([
      "billing: invoices, payments",
      "technical: bugs",
    ]);
  });

  it("renders an undescribed label as the label alone, never 'label: null'", () => {
    // The bug we hit: `dict.fromkeys` gives null values, so this is the default path for
    // list-shaped criteria, not a corner case.
    expect(renderOptions(choice(["alpha", "beta"]))).toEqual(["alpha", "beta"]);
    expect(renderOptions(choice({ described: "yes", bare: null }))).toEqual([
      "described: yes",
      "bare",
    ]);
    expect(renderOptions(choice({ bare: undefined }))).toEqual(["bare"]);
    expect(renderOptions(choice({ bare: "" }))).toEqual(["bare"]);
    for (const rendered of renderOptions(choice({ bare: null }))) {
      expect(rendered).not.toContain("null");
    }
  });

  it("treats 0 and false as legitimate descriptions, not as 'no description'", () => {
    // A truthiness check here would render these as bare labels and change the prompt.
    expect(renderOptions(choice({ one: 0, four: false }))).toEqual(["one: 0", "four: false"]);
    expect(renderOptions(choice({ zero: 0, counted: 12, blank: "" }))).toEqual([
      "zero: 0",
      "counted: 12",
      "blank",
    ]);
  });

  it("JSON-encodes structured descriptions instead of leaking an object literal", () => {
    expect(renderOptions(choice({ rich: { desc: "scam" }, list: ["a", 1] }))).toEqual([
      'rich: {"desc": "scam"}',
      'list: ["a", 1]',
    ]);
  });

  it("numbers score levels from zero", () => {
    expect(
      renderOptions(
        toInternal({ type: "score", instructions: "?", criteria: ["calm", "furious"] }),
      ),
    ).toEqual(["level 0: calm", "level 1: furious"]);
    expect(
      renderOptions(
        toInternal({ type: "score", instructions: "?", criteria: ["calm", { desc: "very" }] }),
      ),
    ).toEqual(["level 0: calm", 'level 1: {"desc": "very"}']);
  });

  it("always renders noul in [false, true] order, whatever the criteria order", () => {
    expect(renderOptions(toInternal({ type: "noul", instructions: "?" }))).toEqual([
      "false: no, the statement does not hold",
      "true: yes, the statement holds",
    ]);
    expect(
      renderOptions(
        toInternal({
          type: "noul",
          instructions: "?",
          criteria: { true: "yes it holds", false: "no it does not" },
        }),
      ),
    ).toEqual(["false: no it does not", "true: yes it holds"]);
    // Insertion order must not leak into the marker order: the probability vector is read against
    // [false, true].
    expect(
      renderOptions(
        toInternal({
          type: "noul",
          instructions: "?",
          criteria: { true: { desc: "scam" }, false: "legitimate" },
        }),
      ),
    ).toEqual(["false: legitimate", 'true: {"desc": "scam"}']);
  });

  it("uses the default noul descriptions when criteria are absent or empty", () => {
    expect(
      renderOptions(toInternal({ type: "noul", instructions: "?", criteria: { true: "" } })),
    ).toEqual(["false: no, the statement does not hold", "true: yes, the statement holds"]);
    expect(renderOptions(toInternal({ type: "noul", instructions: "?", criteria: {} }))).toEqual([
      "false: no, the statement does not hold",
      "true: yes, the statement holds",
    ]);
  });
});

describe("custom noul labels", () => {
  it("replaces the false/true keys in the prompt", () => {
    // This is exactly the `@johnhenry/laya` 0.3.2 bug: their prompt used the raw `false`/`true`
    // keys instead of the labels, which moved the answer (noul 0.766 -> 0.466).
    const q = toInternal({
      type: "noul",
      instructions: "Is this message a scam?",
      criteria: { true: { desc: "scam" }, false: "legitimate" },
      labels: { true: "yes", false: "no" },
    });
    expect(renderOptions(q)).toEqual(["no: legitimate", 'yes: {"desc": "scam"}']);
    const rendered = renderOptions(q);
    expect(rendered.join(" ")).not.toContain("false");
    expect(rendered.join(" ")).not.toContain("true");
  });

  it("trims labels and keeps them distinct", () => {
    expect(resolveNoulLabels({ false: "  no  ", true: " yes " })).toEqual(["no", "yes"]);
    expect(resolveNoulLabels(undefined)).toEqual(["false", "true"]);
  });

  it("rejects label maps that are not exactly false/true, or are not distinct", () => {
    expect(() => resolveNoulLabels({ false: "no" } as never)).toThrow(/exactly/);
    expect(() => resolveNoulLabels({ false: "x", true: "y", other: "z" } as never)).toThrow(
      /exactly/,
    );
    expect(() => resolveNoulLabels({ false: "same", true: "same" })).toThrow(/exactly/);
    expect(() => resolveNoulLabels({ false: "  ", true: "yes" })).toThrow(/exactly/);
    expect(() => resolveNoulLabels({ false: 1, true: 2 } as never)).toThrow(/exactly/);
  });
});
