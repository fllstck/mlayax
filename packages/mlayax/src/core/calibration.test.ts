import { describe, expect, it } from "vitest";
import {
  answerConfidence,
  clampTemperature,
  confidenceFromProbs,
  formatSignificant,
  rejectedTemperatureEntries,
  round4,
  softmax,
  TEMP_MAX,
  TEMP_MIN,
  tempBucket,
  temperatureClampWarning,
} from "./calibration.js";

describe("clampTemperature", () => {
  it("keeps values inside [0.5, 5]", () => {
    expect(TEMP_MIN).toBe(0.5);
    expect(TEMP_MAX).toBe(5.0);
    expect(clampTemperature(1.0)).toBe(1.0);
    expect(clampTemperature(0.5)).toBe(0.5);
    expect(clampTemperature(5.0)).toBe(5.0);
    expect(clampTemperature(0.7)).toBe(0.7);
  });

  it("clamps a sharpening temperature up to the floor", () => {
    // The shipped `choice:11+` bucket is 0.1006, a ~10x sharpener. Applying it would publish a
    // 0.24 top probability as 0.99, so it is clamped rather than used.
    expect(clampTemperature(0.10058280825614929)).toBe(TEMP_MIN);
    expect(clampTemperature(0.1006)).toBe(TEMP_MIN);
    expect(clampTemperature(0)).toBe(TEMP_MIN);
    expect(clampTemperature(-1)).toBe(TEMP_MIN);
  });

  it("clamps a softening temperature down to the ceiling", () => {
    expect(clampTemperature(12)).toBe(TEMP_MAX);
    expect(clampTemperature(Number.MAX_VALUE)).toBe(TEMP_MAX);
  });

  it("falls back to 1.0 for anything that is not a usable number", () => {
    // An unset temperature means "no scaling", not "scale by zero" or "scale by NaN".
    for (const unusable of [
      null,
      undefined,
      true,
      false,
      "",
      "  ",
      "not a number",
      {},
      [],
      Number.NaN,
      Number.POSITIVE_INFINITY,
      Number.NEGATIVE_INFINITY,
    ]) {
      expect(clampTemperature(unusable), `clampTemperature(${JSON.stringify(unusable)})`).toBe(1.0);
    }
  });

  it("accepts numeric strings, as Python's float() does", () => {
    expect(clampTemperature("1.5")).toBe(1.5);
    expect(clampTemperature(" 2 ")).toBe(2);
    expect(clampTemperature("0.1")).toBe(TEMP_MIN);
  });

  it("honours custom bounds", () => {
    expect(clampTemperature(3, 1, 2)).toBe(2);
    expect(clampTemperature(0, 1, 2)).toBe(1);
  });
});

describe("rejectedTemperatureEntries", () => {
  it("lists per-bucket entries first, then the positional ones", () => {
    expect(
      rejectedTemperatureEntries([1.6, 1.25, 1.98], {
        "choice:2": 1.9,
        "choice:3-5": 1.76,
        "choice:11+": 0.1006,
        "noul:2": 1.98,
      }),
    ).toEqual(["choice:11+=0.1006"]);
  });

  it("formats each entry with %.4g, as the reference warning does", () => {
    expect(rejectedTemperatureEntries([8.123456789], {})).toEqual(["temperature[0]=8.123"]);
    expect(rejectedTemperatureEntries([0.25], { sharp: 0.10058280825614929 })).toEqual([
      "sharp=0.1006",
      "temperature[0]=0.25",
    ]);
  });

  it("is empty when nothing needed clamping", () => {
    expect(rejectedTemperatureEntries([1.6, 1.25, 1.98], { "choice:2": 1.9 })).toEqual([]);
  });

  it("skips values that are not numbers at all", () => {
    // Those are a load-time error ("Calibration temperatures must be finite and positive"), not a
    // clamp, and formatting them as a clamp would hide the real problem.
    expect(rejectedTemperatureEntries(["nope"], { bad: "not a number" })).toEqual([]);
  });
});

describe("temperatureClampWarning", () => {
  it("reproduces the reference wording verbatim", () => {
    expect(temperatureClampWarning(["choice:11+=0.1006"])).toBe(
      "laya-mlx: this checkpoint ships temperatures outside [0.5, 5] which would distort " +
        "confidence; clamping choice:11+=0.1006. Treat confidence from the affected buckets as " +
        "uncalibrated.",
    );
  });

  it("joins multiple entries with a comma and a space", () => {
    expect(temperatureClampWarning(["a=1", "b=2"])).toBe(
      "laya-mlx: this checkpoint ships temperatures outside [0.5, 5] which would distort " +
        "confidence; clamping a=1, b=2. Treat confidence from the affected buckets as uncalibrated.",
    );
  });

  it("returns null when nothing was clamped", () => {
    expect(temperatureClampWarning([])).toBeNull();
  });
});

describe("formatSignificant", () => {
  it("matches C's %.4g", () => {
    // Every expectation here was produced by CPython's `"%.4g" % v`.
    const table: [number, string][] = [
      [0.10058280825614929, "0.1006"],
      [1.6369030475616455, "1.637"],
      [1.2514300346374512, "1.251"],
      [1.983399510383606, "1.983"],
      [1.9, "1.9"],
      [1.76, "1.76"],
      [1.6, "1.6"],
      [1.25, "1.25"],
      [5.00001, "5"],
      [0.49999, "0.5"],
      [12345.678, "1.235e+04"],
      [0.000123456, "0.0001235"],
      [0, "0"],
      [1, "1"],
      [100, "100"],
      [1e-5, "1e-05"],
      [2.5, "2.5"],
      [0.30000000000000004, "0.3"],
    ];
    for (const [value, expected] of table) {
      expect(formatSignificant(value, 4), `%.4g of ${value}`).toBe(expected);
    }
  });

  it("defaults to 4 significant digits and honours a custom precision", () => {
    expect(formatSignificant(1.6369030475616455)).toBe("1.637");
    expect(formatSignificant(1.6369030475616455, 6)).toBe("1.6369");
    expect(formatSignificant(0.5, 6)).toBe("0.5");
    expect(formatSignificant(5, 6)).toBe("5");
  });
});

describe("tempBucket", () => {
  it("buckets option counts the way the checkpoint fits them", () => {
    expect(tempBucket(0, 2)).toBe("choice:2");
    expect(tempBucket(0, 3)).toBe("choice:3-5");
    expect(tempBucket(0, 5)).toBe("choice:3-5");
    expect(tempBucket(0, 6)).toBe("choice:6-10");
    expect(tempBucket(0, 10)).toBe("choice:6-10");
    expect(tempBucket(0, 11)).toBe("choice:11+");
    expect(tempBucket(0, 40)).toBe("choice:11+");
    expect(tempBucket(1, 3)).toBe("score:3-5");
    expect(tempBucket(2, 2)).toBe("noul:2");
  });
});

describe("confidenceFromProbs", () => {
  it("is 1 for a point mass and 0 for a uniform distribution", () => {
    expect(confidenceFromProbs([1, 0, 0], 3)).toBeCloseTo(1, 12);
    expect(confidenceFromProbs([1 / 3, 1 / 3, 1 / 3], 3)).toBeCloseTo(0, 12);
    expect(confidenceFromProbs([0.25, 0.25, 0.25, 0.25], 4)).toBeCloseTo(0, 12);
  });

  it("returns 1 when there is fewer than one option to be uncertain about", () => {
    expect(confidenceFromProbs([1], 1)).toBe(1.0);
    expect(confidenceFromProbs([0.5, 0.5], 0)).toBe(1.0);
  });

  it("is symmetric in the probabilities, and insensitive to their order", () => {
    expect(confidenceFromProbs([0.8, 0.1, 0.1], 3)).toBeCloseTo(
      confidenceFromProbs([0.1, 0.1, 0.8], 3),
      12,
    );
  });

  it("reads only the first k options", () => {
    expect(confidenceFromProbs([1, 0, 0, 0, 0], 2)).toBeCloseTo(1, 12);
  });

  it("treats a zero probability as contributing nothing rather than NaN", () => {
    expect(Number.isFinite(confidenceFromProbs([1, 0], 2))).toBe(true);
    expect(Number.isFinite(confidenceFromProbs([1, 0, 0, 0, 0, 0], 6))).toBe(true);
  });

  it("stays inside [0, 1]", () => {
    for (const p of [
      [1, 0, 0],
      [0.5, 0.5],
      [1 / 3, 1 / 3, 1 / 3],
      [1, 0],
    ]) {
      const value = confidenceFromProbs(p, p.length);
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThanOrEqual(1);
    }
  });
});

describe("answerConfidence", () => {
  it("is the top probability, not the entropy measure", () => {
    expect(answerConfidence([0.9585, 0.0227, 0.0186], 3)).toBeCloseTo(0.9585, 12);
    expect(answerConfidence([0.5, 0.5], 2)).toBeCloseTo(0.5, 12);
    // The two quantities differ on the same input, which is exactly why they must not share a
    // threshold: uniform over 2 gives 0 entropy-confidence but 0.5 answer-confidence.
    expect(confidenceFromProbs([0.5, 0.5], 2)).toBeCloseTo(0, 12);
    expect(answerConfidence([0.5, 0.5], 2)).toBeCloseTo(0.5, 12);
  });

  it("clamps to [0, 1] and returns 1 for an empty option set", () => {
    expect(answerConfidence([1.4, -0.2], 2)).toBe(1);
    expect(answerConfidence([1.4, -0.2], 2)).toBe(1);
    expect(answerConfidence([-1, -2], 2)).toBe(0);
    expect(answerConfidence([0.5], 0)).toBe(1.0);
  });
});

describe("round4", () => {
  it("rounds to four decimals", () => {
    expect(round4(0.9585)).toBe(0.9585);
    expect(round4(0.95855)).toBe(0.9586);
    expect(round4(0.12344)).toBe(0.1234);
    expect(round4(0.12345)).toBe(0.1235);
    expect(round4(0.12335)).toBe(0.1234);
    expect(round4(1)).toBe(1);
    expect(round4(0)).toBe(0);
  });

  it("rounds the exact double, like CPython, including near-ties", () => {
    // Every expectation is CPython's `round(v, 4)`. These are the values that separate a
    // correct implementation from a tolerance-based one: 0.00005 is a hair *above* the half as a
    // double, 0.00015 and 2.00005 a hair below, and 0.00025/0.00035 sit either side of even.
    const table: [number, number][] = [
      [0.00005, 0.0001],
      [0.00015, 0.0001],
      [0.00025, 0.0003],
      [0.00035, 0.0003],
      [1.00005, 1.0001],
      [1.00015, 1.0002],
      [2.00005, 2],
      [-0.00005, -0.0001],
      [-1.00005, -1.0001],
    ];
    for (const [value, expected] of table) {
      expect(round4(value), `round4(${value})`).toBe(expected);
    }
  });

  it("is not the naive scaled-round, which disagrees with Python near ties", () => {
    // Pin the two failure modes the old formulation had, so a future refactor cannot quietly
    // reintroduce them.
    const naiveScaledRound = (x: number) => Math.round(x * 1e4) / 1e4;
    expect(naiveScaledRound(0.00005)).toBe(0.0001);
    expect(round4(0.00005)).toBe(0.0001);
    expect(naiveScaledRound(1.00005)).toBe(1.0001);
    // The tolerance-based version snapped this to the half and rounded to even, giving 0.
    expect(round4(0.00005)).not.toBe(0);
  });

  it("preserves signed zero, as Python's round does", () => {
    expect(Object.is(round4(-0.00004), -0)).toBe(true);
    expect(Object.is(round4(-0.00005), -0.0001)).toBe(true);
    expect(Object.is(round4(0.00004), 0)).toBe(true);
  });

  it("is stable on values the model actually emits", () => {
    for (const value of [0.9585, 0.0227, 0.0186, 0.5123, 0.768, 0.9999]) {
      expect(round4(value)).toBe(value);
    }
  });

  it("passes non-finite values through rather than inventing a number", () => {
    expect(round4(Number.NaN)).toBeNaN();
    expect(round4(Number.POSITIVE_INFINITY)).toBe(Number.POSITIVE_INFINITY);
  });
});

describe("softmax", () => {
  it("normalises a vector and preserves order", () => {
    const p = softmax([1, 2, 3]);
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    expect(p[0]).toBeLessThan(p[1] as number);
    expect(p[1]).toBeLessThan(p[2] as number);
  });

  it("is stable for large logits", () => {
    const p = softmax([1000, 1001]);
    expect(p.every(Number.isFinite)).toBe(true);
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
  });
});
