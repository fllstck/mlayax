/**
 * The load-option matrix and the malformed-checkpoint paths, against the tiny fixture.
 *
 * Two jobs, both of which were uncovered before this file (TASKS.md §10.5):
 *
 * 1. **Every option still produces the reference answers.** `attention: "manual"`, `shapeless`,
 *    `pretranspose`, `addmm`, `lengthBucket`, `tidy: false`, `cacheMasks: false` and `bfloat16` are
 *    all code paths in `agent.ts` and `model.ts` that a default load never executes. A regression in
 *    one of them changes answers without changing the default configuration's answers, so the
 *    default parity gate cannot see it.
 * 2. **A checkpoint that cannot be used is rejected, with the reason.** Temperature validation, an
 *    unsupported encoder, a scaled RoPE. These run *before* anything reaches the GPU, which is the
 *    only place a clear message is still cheap.
 *
 * The tolerances below are measured, not chosen. On this fixture, with fp32 as the baseline:
 *
 * | option | max Δ | fields differing |
 * |---|---|---|
 * | `pretranspose`, `addmm`, `lengthBucket`, `tidy: false`, `cacheMasks: false` | **0** | 0 / 45 |
 * | `manual attention`, `shapeless` | 1e-4 | 2 / 45 |
 * | `bfloat16` | 8e-4 | 30 / 45 |
 *
 * The 1e-4 cases are the same one-rounding-step effect documented in `test/parity.tiny.test.ts`: a
 * different accumulation order moves `action.act_probability` across a 4-decimal boundary. bfloat16
 * gets its own bound because it is genuinely less precise than fp16 — 8 significand bits against 10,
 * so ~2e-3 relative — and reusing the 4e-4 fp16 tolerance here would be a claim about the tolerance
 * rather than about the arithmetic.
 */

import { mkdirSync, mkdtempSync, readFileSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { afterAll, describe, expect, it } from "vitest";
import type { DTypeName, LoadOptions, MlxAgent } from "../packages/mlayax/src/index.js";
// A static import is safe and deliberate: importing the package must not load native code, which
// `test/public-surface.test.ts` asserts. It also keeps these assertions synchronous — a `load()`
// that is supposed to throw must not become a rejected promise vitest reports elsewhere.
import { load } from "../packages/mlayax/src/index.js";
import { resolveNativeAddonPath } from "../packages/mlayax/src/mlx/binding.js";
import {
  ACTION_PROBABILITY_DRIFT,
  driftPolicy,
  isReferenceMachine,
  machineNote,
} from "./helpers/machine.js";
import { compare, loadReference } from "./helpers/scorecard.js";

const FIXTURE = fileURLToPath(new URL("./fixtures/tiny", import.meta.url));
const REFERENCE = fileURLToPath(new URL("./fixtures/ref/tiny-fp32.json", import.meta.url));

/** One rounding step at 4 decimals, for the options that reorder an accumulation. */
const ONE_ROUNDING_STEP = 1e-4;
/**
 * Off the reference machine, the same field that moves in the parity test gets the same measured
 * allowance: 3 steps at 4 decimals, measured on the CI runner (TASKS.md §10.14). The field class is
 * not loosened — `driftPolicy` grants it to `action.act_probability` only.
 */
const REORDERING_DRIFT = isReferenceMachine() ? ONE_ROUNDING_STEP : ACTION_PROBABILITY_DRIFT;
/**
 * And the bound for the six options documented as *bit-exact*: zero on the reference machine, the same
 * measured action-head drift elsewhere. `differing` is counted with the same policy in force, so an
 * option that moves a *different* field still fails on any machine.
 */
const BIT_EXACT_BOUND = isReferenceMachine() ? 0 : ACTION_PROBABILITY_DRIFT;
/** bfloat16's honest bound: 2^-9 relative, observed at 8e-4 on this fixture. */
const BFLOAT16_TOLERANCE = 2e-3;

function nativeAvailable(): boolean {
  try {
    resolveNativeAddonPath();
    return true;
  } catch {
    return false;
  }
}

const tempDirs: string[] = [];
afterAll(() => {
  for (const dir of tempDirs) rmSync(dir, { recursive: true, force: true });
});

/** Compare one option's answers against the fp32 reference, and return the worst delta. */
async function maxDeltaFor(options: LoadOptions): Promise<{ maxDelta: number; differing: number }> {
  const { load } = await import("../packages/mlayax/src/index.js");
  const agent: MlxAgent = load(FIXTURE, { dtype: "float32", ...options });
  const reference = loadReference(REFERENCE);
  let maxDelta = 0;
  let differing = 0;
  for (const testCase of reference.cases) {
    const prediction = await agent.predict(testCase.state, testCase.questions);
    const comparison = compare(testCase.answers, prediction.answers, 0, {
      driftForPath: driftPolicy(),
    });
    maxDelta = Math.max(maxDelta, comparison.maxDelta);
    differing += comparison.mismatches.length;
    expect(
      comparison.compared,
      `${testCase.label}: the option changed the payload's shape`,
    ).toBeGreaterThan(0);
  }
  return { maxDelta, differing };
}
/**
 * A model directory identical to the fixture except for the JSON it is handed.
 *
 * The patch is **merged** into the fixture's config rather than replacing it, so each test can vary
 * one key and still be looking at an otherwise valid checkpoint. Replacing it wholesale would make
 * every case fail on the first missing field rather than on the field under test — which is exactly
 * what happened the first time this was written.
 *
 * Symlinked rather than copied: `model.safetensors` is 248 KB and there are a dozen cases, and
 * `load()` reads these by path without caring how they got there.
 */
function patchedModelDir(patch: {
  agent?: Record<string, unknown>;
  encoder?: Record<string, unknown>;
}): string {
  const dir = mkdtempSync(path.join(tmpdir(), "mlayax-tiny-"));
  tempDirs.push(dir);
  const readJson = (relative: string): Record<string, unknown> =>
    JSON.parse(readFileSync(path.join(FIXTURE, relative), "utf8")) as Record<string, unknown>;

  symlinkSync(path.join(FIXTURE, "model.safetensors"), path.join(dir, "model.safetensors"));
  symlinkSync(path.join(FIXTURE, "tokenizer"), path.join(dir, "tokenizer"));
  mkdirSync(path.join(dir, "encoder"));
  writeFileSync(
    path.join(dir, "encoder", "config.json"),
    JSON.stringify({ ...readJson("encoder/config.json"), ...patch.encoder }),
  );
  writeFileSync(
    path.join(dir, "rl_agent_config.json"),
    JSON.stringify({ ...readJson("rl_agent_config.json"), ...patch.agent }),
  );
  return dir;
}

describe.skipIf(!nativeAvailable())(
  "load options that the default configuration never exercises",
  () => {
    it.each([
      ["pretranspose", { pretranspose: true }],
      ["addmm", { addmm: true }],
      ["lengthBucket", { lengthBucket: 32 }],
      ["tidy: false", { tidy: false }],
      ["cacheMasks: false", { cacheMasks: false }],
      // The eager forward: no `mx.compile`, so every op is issued from JS. This is the other branch of
      // `forwardItems`' `compiledForward ? … : …`, and it is ~12 % slower by design — the point here
      // is that it is not *different*.
      ["compile: false", { compile: false }],
    ] as const)("%s leaves every answer bit-exact", async (_label, options) => {
      // Bit-exact is the strong claim, and these six options earn it: they change how the same maths is
      // arranged (transposes held resident, a fused addmm, a padded length) without changing the order
      // of any accumulation. If one of them stops being exact, this names which.
      const { maxDelta, differing } = await maxDeltaFor(options);
      // `differing` counts fields beyond the drift this machine is allowed — the action head's, off the
      // reference machine — so this pair is "nothing unexpected moved" *and* "nothing moved far".
      expect(differing, `${differing} field(s) differed beyond the allowed drift`).toBe(0);
      expect(
        maxDelta,
        `${machineNote()}: nothing but the action head may move`,
      ).toBeLessThanOrEqual(BIT_EXACT_BOUND);
    });

    it.each([
      ["manual attention", { attention: "manual" }],
      ["shapeless", { shapeless: true }],
    ] as const)("%s agrees within one rounding step", async (_label, options) => {
      // Both take the same maths through a different kernel: manual attention accumulates attention in
      // fp32 by hand instead of using the fused Metal kernel, and the shapeless profile compiles once
      // for every shape. The result is a last-bit difference that can land either side of a 4-decimal
      // boundary — see test/parity.tiny.test.ts for the same effect in batched-vs-solo.
      const { maxDelta } = await maxDeltaFor(options);
      expect(maxDelta).toBeGreaterThan(0); // measured, not assumed: these are not bit-exact
      expect(maxDelta).toBeLessThanOrEqual(REORDERING_DRIFT);
    });

    it("bfloat16 loads and agrees within its own precision", async () => {
      const { maxDelta } = await maxDeltaFor({ dtype: "bfloat16" as DTypeName });
      expect(maxDelta).toBeLessThanOrEqual(BFLOAT16_TOLERANCE);
    });

    it("answers the reference questions with a finite action probability under every option", async () => {
      // A cheap sweep for a NaN or a `null` creeping in through an option nobody exercises by default.
      const { load } = await import("../packages/mlayax/src/index.js");
      const reference = loadReference(REFERENCE);
      const testCase = reference.cases[0];
      if (testCase === undefined) throw new Error("fixture has no cases");

      for (const options of [
        { attention: "manual" },
        { shapeless: true },
        { pretranspose: true },
        { addmm: true },
        { lengthBucket: 32 },
        { tidy: false },
        { cacheMasks: false },
      ] satisfies LoadOptions[]) {
        const agent: MlxAgent = load(FIXTURE, { dtype: "float32", ...options });
        const prediction = await agent.predict(testCase.state, testCase.questions);
        for (const [qid, answer] of Object.entries(prediction.answers)) {
          expect(
            Number.isFinite(answer.action.act_probability),
            `${qid} ${JSON.stringify(options)}`,
          ).toBe(true);
        }
      }
    });

    it("rejects an unknown dtype by name", async () => {
      const { load } = await import("../packages/mlayax/src/index.js");
      expect(() => load(FIXTURE, { dtype: "float8" as DTypeName })).toThrow(
        /Unknown dtype: float8/,
      );
    });

    it("rawLogits returns one finite logit per marker", async () => {
      // The diagnostics path used for numerical comparison against Python. It runs eagerly rather than
      // through the compiled graph, so it has its own uncovered code.
      const { load } = await import("../packages/mlayax/src/index.js");
      const agent: MlxAgent = load(FIXTURE, { dtype: "float32" });
      const reference = loadReference(REFERENCE);
      const testCase = reference.cases[0];
      if (testCase === undefined) throw new Error("fixture has no cases");
      const questionIds = Object.keys(testCase.questions);
      const firstId = questionIds[0];
      if (firstId === undefined) throw new Error("fixture has no questions");

      const logits = agent.rawLogits(testCase.state, {
        [firstId]: (testCase.questions as Record<string, unknown>)[firstId],
      });
      expect(logits.length).toBeGreaterThan(1);
      expect(logits.every(Number.isFinite)).toBe(true);
      // The largest logit is the reported choice, so the diagnostics path agrees with the answer path.
      const answer = (
        await agent.predict(testCase.state, {
          [firstId]: (testCase.questions as Record<string, unknown>)[firstId],
        })
      ).answers[firstId];
      const labels = Object.keys(
        (answer as unknown as { probabilities: Record<string, number> }).probabilities,
      );
      const argMax = labels[logits.indexOf(Math.max(...logits))];
      expect(argMax).toBe((answer as unknown as { choice: string }).choice);
    });
  },
);

describe.skipIf(!nativeAvailable())(
  "a checkpoint that cannot be used is refused, with the reason",
  () => {
    const loadSync = (dir: string): MlxAgent => load(dir);

    const withAgent = (agent: Record<string, unknown>, message: RegExp | string): void => {
      expect(() => loadSync(patchedModelDir({ agent }))).toThrow(message);
    };

    it("rejects temperatures that are not exactly three", () => {
      // Upstream's rule: three positional temperatures, one per question type.
      withAgent(
        { temperature: [1.5, 1.5] },
        /Calibration temperatures must be finite and positive/,
      );
    });

    it("rejects a non-finite temperature", () => {
      withAgent({ temperature: [1.5, Number.NaN, 1.5] }, /must be finite and positive/);
      withAgent(
        { temperature: [1.5, Number.POSITIVE_INFINITY, 1.5] },
        /must be finite and positive/,
      );
    });

    it("rejects a non-positive temperature", () => {
      withAgent({ temperature: [1.5, 0, 1.5] }, /must be finite and positive/);
      withAgent({ temperature: [1.5, -1, 1.5] }, /must be finite and positive/);
    });

    it("rejects a bad value in a per-option bucket, not just in the positional list", () => {
      withAgent({ temperature_by_options: { "choice:3-5": 0 } }, /must be finite and positive/);
    });

    it("rejects a boolean where a temperature belongs", () => {
      // `isinstance(t, bool)` is excluded explicitly in the reference: `True` is a valid number in
      // Python and would divide the logits by 1, silently disabling calibration.
      withAgent({ temperature: [true, 1.5, 1.5] }, /must be finite and positive/);
    });

    it("accepts temperatures outside the clamp window, clamping instead of failing", () => {
      // The shipped checkpoints contain one (choice:11+ at 0.1006), so this must load and warn rather
      // than throw. Asserted because "invalid" and "unusual" are easy to conflate here.
      const agent = loadSync(
        patchedModelDir({ agent: { temperature_by_options: { "choice:11+": 0.1 } } }),
      );
      expect(agent.temperatureByOptions).toBeDefined();
    });

    it("rejects an encoder that is not ModernBERT", () => {
      expect(() => loadSync(patchedModelDir({ encoder: { model_type: "bert" } }))).toThrow(
        /Unsupported encoder/,
      );
    });

    it("rejects a scaled RoPE rather than silently applying unscaled RoPE", () => {
      expect(() =>
        loadSync(
          patchedModelDir({
            encoder: { rope_parameters: { full_attention: { rope_type: "linear" } } },
          }),
        ),
      ).toThrow(/Only default \(unscaled\) ModernBERT RoPE/);
    });

    it("reports a missing checkpoint file rather than producing an empty model", () => {
      const dir = patchedModelDir({});
      rmSync(path.join(dir, "model.safetensors"));
      expect(() => loadSync(dir)).toThrow();
    });
  },
);
