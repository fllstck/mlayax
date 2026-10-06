import { describe, expect, it } from "vitest";
import { pyFloatRepr, pyJson, pyNumberRepr, renderCriterion, serializeState } from "./ppjson.js";

describe("pyJson", () => {
  it("uses Python's default separators, not JSON.stringify's compact form", () => {
    // The whole reason this module exists. A compact rendering changes the prompt's token ids.
    expect(pyJson({ a: 1, b: 2 })).toBe('{"a": 1, "b": 2}');
    expect(pyJson([1, 2, 3])).toBe("[1, 2, 3]");
    expect(pyJson({ desc: "scam" })).toBe('{"desc": "scam"}');
    expect(pyJson({ a: "x", b: [1, { c: 2 }] })).toBe('{"a": "x", "b": [1, {"c": 2}]}');
  });

  it("renders scalars the way json.dumps does", () => {
    expect(pyJson(null)).toBe("null");
    expect(pyJson(undefined)).toBe("null");
    expect(pyJson(true)).toBe("true");
    expect(pyJson(false)).toBe("false");
    expect(pyJson("plain")).toBe('"plain"');
    expect(pyJson({})).toBe("{}");
    expect(pyJson([])).toBe("[]");
  });

  it("keeps non-ASCII characters raw (ensure_ascii=False)", () => {
    expect(pyJson("café")).toBe('"café"');
    expect(pyJson("→ 世界")).toBe('"→ 世界"');
    expect(pyJson({ note: "naïve — 250 €" })).toBe('{"note": "naïve — 250 €"}');
    // Escapes that both languages agree on.
    expect(pyJson("a\nb\tc")).toBe('"a\\nb\\tc"');
    expect(pyJson('say "hi"')).toBe('"say \\"hi\\""');
  });

  it("keeps insertion order, as a Python dict does", () => {
    expect(pyJson({ z: 1, a: 2, m: 3 })).toBe('{"z": 1, "a": 2, "m": 3}');
  });

  it("renders floats as Python repr does", () => {
    // Every expectation on this list was produced by CPython's repr(), including the awkward ones:
    // integral floats keep ".0", the exponent threshold is [-4, 16), and the exponent is padded to
    // two digits. JS String() gets 5 of these wrong (2.0, 1e-5, 1e-7, 1e16, 5e-324).
    const table: [number, string][] = [
      [0.0, "0.0"],
      [-0.0, "-0.0"],
      [2.0, "2.0"],
      [1.5, "1.5"],
      [0.5, "0.5"],
      [0.0001, "0.0001"],
      [0.00001, "1e-05"],
      [1e-7, "1e-07"],
      [1.5e-8, "1.5e-08"],
      [1e15, "1000000000000000.0"],
      [1e16, "1e+16"],
      [1.2345e17, "1.2345e+17"],
      [1e21, "1e+21"],
      [123.456, "123.456"],
      [1234567890123456.0, "1234567890123456.0"],
      [0.1, "0.1"],
      [Math.PI, "3.141592653589793"],
      [-2.5, "-2.5"],
      [-1e-5, "-1e-05"],
      [100.0, "100.0"],
      [1e100, "1e+100"],
      [5e-324, "5e-324"],
    ];
    for (const [value, expected] of table) {
      expect(pyFloatRepr(value), `pyFloatRepr(${value})`).toBe(expected);
    }
  });

  it("spells non-finite numbers the way json.dumps does, not as null", () => {
    expect(pyFloatRepr(Number.NaN)).toBe("NaN");
    expect(pyFloatRepr(Number.POSITIVE_INFINITY)).toBe("Infinity");
    expect(pyFloatRepr(Number.NEGATIVE_INFINITY)).toBe("-Infinity");
    expect(pyJson(Number.POSITIVE_INFINITY)).toBe("Infinity");
  });

  it("nests floats and structures through one code path", () => {
    expect(pyJson({ tags: ["a", 1e-5], ratio: 2.5 })).toBe('{"tags": ["a", 1e-05], "ratio": 2.5}');
  });
});

describe("pyNumberRepr", () => {
  it("renders integral values the way Python renders an int", () => {
    expect(pyNumberRepr(0)).toBe("0");
    expect(pyNumberRepr(-0)).toBe("0");
    expect(pyNumberRepr(2)).toBe("2");
    expect(pyNumberRepr(-17)).toBe("-17");
    expect(pyNumberRepr(10 ** 20)).toBe("100000000000000000000");
  });

  it("renders fractional values the way Python renders a float", () => {
    expect(pyNumberRepr(2.5)).toBe("2.5");
    expect(pyNumberRepr(1e-5)).toBe("1e-05");
    expect(pyNumberRepr(-0.5)).toBe("-0.5");
  });

  it("documents the int-vs-float ambiguity JS cannot resolve", () => {
    // Python: json.dumps(2) == "2", json.dumps(2.0) == "2.0". JS JSON.parse gives 2 for both.
    // We choose the int spelling, because rubric criteria are usually integers and a spurious
    // ".0" would be wrong far more often than a missing one. This test pins the deliberate
    // lossy case so the behaviour is a decision rather than a latent bug.
    expect(pyNumberRepr(JSON.parse("2.0"))).toBe("2");
    expect(pyNumberRepr(JSON.parse("2"))).toBe("2");
    expect(pyFloatRepr(2.0)).toBe("2.0");
  });
});

describe("serializeState", () => {
  it("passes strings through untouched", () => {
    expect(serializeState("I was billed twice.")).toBe("I was billed twice.");
    expect(serializeState("  spaced  ")).toBe("  spaced  ");
    expect(serializeState("")).toBe("");
  });

  it("renders structured state as Python-flavoured JSON", () => {
    expect(serializeState({ from: "hello@world", subject: "charge on invoice" })).toBe(
      '{"from": "hello@world", "subject": "charge on invoice"}',
    );
    expect(serializeState(["a", "b"])).toBe('["a", "b"]');
    // A list of turns is a legitimate state shape; it is what turns on left-truncation.
    expect(serializeState([{ role: "user", text: "hi" }])).toBe('[{"role": "user", "text": "hi"}]');
  });

  it("treats null and undefined state as null, matching Python None", () => {
    expect(serializeState(null)).toBe("null");
    expect(serializeState(undefined)).toBe("null");
  });
});

describe("renderCriterion", () => {
  it("passes strings through and JSON-encodes anything structured", () => {
    expect(renderCriterion("bugs and outages")).toBe("bugs and outages");
    expect(renderCriterion({ desc: "scam" })).toBe('{"desc": "scam"}');
    expect(renderCriterion(["a", "b"])).toBe('["a", "b"]');
    expect(renderCriterion(0)).toBe("0");
    expect(renderCriterion(2.5)).toBe("2.5");
    expect(renderCriterion(false)).toBe("false");
  });
});
