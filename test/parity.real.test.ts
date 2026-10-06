/**
 * Real-checkpoint parity against the Python reference — the correctness gate of the whole port.
 *
 * Opt-in, because it needs the ~803 MiB checkpoint and the native payload:
 *
 * ```bash
 * MLAYAX_MODEL_DIR=~/Development/.../laya-ts-spike/models/english-mlx \
 * MLAYAX_NATIVE_DIR=~/Development/.../laya-ts-spike/node_modules/@frost-beta/mlx/build/Release \
 * npx vitest run --config vitest.parity.config.ts
 * ```
 *
 * Gates, from TASKS.md §2 and §6:
 *   fp32 — **bit-exact**, Δ 0 on all 61 numeric fields of all three cases;
 *   fp16 — Δ ≤ 4e-4.
 *
 * A failure here is the single most informative signal in the repository: the stage that drifted is
 * named by the field path, and `test/helpers/scorecard.ts` prints every mismatch rather than one.
 */

import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { resolveCachedModel } from "../packages/mlayax/src/hub.js";
import type { MlxAgent, Answer as MlxAnswer } from "../packages/mlayax/src/index.js";
import { resolveNativeAddonPath } from "../packages/mlayax/src/mlx/binding.js";
import { compare, describeComparison, loadReference } from "./helpers/scorecard.js";

/** The checkpoint this project targets, as a repository id. */
const REPO_ID = "aac6fef/laya-mlx";

const MODEL_DIR = process.env.MLAYAX_MODEL_DIR;
const FP32_TOLERANCE = 0;
const FP16_TOLERANCE = 4e-4;

function referencePath(dtype: "fp16" | "fp32"): string {
  return fileURLToPath(new URL(`./fixtures/ref/${dtype}.json`, import.meta.url));
}

// `describe.skipIf` keeps the default `vitest run` green without the checkpoint, so CI's unit job
// does not need the weights while this remains the same test.
describe.skipIf(MODEL_DIR === undefined || MODEL_DIR === "")(
  "real-checkpoint parity vs Python laya_mlx",
  () => {
    it("fp32 is bit-exact", async () => {
      const { load } = await import("../packages/mlayax/src/index.js");
      if (MODEL_DIR === undefined || !existsSync(MODEL_DIR)) {
        throw new Error(`MLAYAX_MODEL_DIR does not exist: ${String(MODEL_DIR)}`);
      }
      const agent: MlxAgent = load(MODEL_DIR, { dtype: "float32" });
      const reference = loadReference(referencePath("fp32"));
      for (const testCase of reference.cases) {
        const prediction = await agent.predict(testCase.state, testCase.questions);
        const comparison = compare(testCase.answers, prediction.answers, FP32_TOLERANCE);
        expect(comparison.mismatches, describeComparison(testCase.label, comparison)).toEqual([]);
        // "Δ 0" is stronger than "no mismatch beyond tolerance", so assert it directly.
        expect(comparison.maxDelta).toBe(0);
        expect(comparison.exact).toBe(comparison.compared);
      }
    });

    it("fp16 is within 4e-4", async () => {
      const { load } = await import("../packages/mlayax/src/index.js");
      if (MODEL_DIR === undefined || !existsSync(MODEL_DIR)) {
        throw new Error(`MLAYAX_MODEL_DIR does not exist: ${String(MODEL_DIR)}`);
      }
      const agent: MlxAgent = load(MODEL_DIR, { dtype: "float16" });
      const reference = loadReference(referencePath("fp16"));
      for (const testCase of reference.cases) {
        const prediction = await agent.predict(testCase.state, testCase.questions);
        const comparison = compare(testCase.answers, prediction.answers, FP16_TOLERANCE);
        expect(comparison.mismatches, describeComparison(testCase.label, comparison)).toEqual([]);
      }
    });

    it("answers the same in a batch as one at a time", async () => {
      // Rows are independent and padding is masked, so batching cannot change an answer. This is the
      // assertion TASKS.md §5 asks the service layer to make, checked here at the agent level where
      // a failure points straight at collation.
      const { load } = await import("../packages/mlayax/src/index.js");
      const agent: MlxAgent = load(MODEL_DIR as string, { dtype: "float32" });
      const reference = loadReference(referencePath("fp32"));
      const testCase = reference.cases[0];
      if (testCase === undefined) throw new Error("reference has no cases");

      const together = await agent.predict(testCase.state, testCase.questions);
      const separate: Record<string, unknown> = {};
      for (const [qid, definition] of Object.entries(testCase.questions)) {
        const solo = await agent.predict(testCase.state, { [qid]: definition });
        separate[qid] = solo.answers[qid];
      }
      const comparison = compare(together.answers, separate, 0);
      expect(comparison.mismatches, describeComparison("batched vs solo", comparison)).toEqual([]);
    });
  },
);

/**
 * Runtime hazards that need real weights but not correctness against the reference.
 *
 * Everything here is TASKS.md §8 with a non-zero cost attached: a mask disposed while still cached
 * (a bare `std::invalid_argument` on the next hit), the shapeless profile drifting from the default,
 * and `mx.tidy` wrapping an in-flight async evaluation.
 *
 * Agents are memoised per configuration: each one holds ~800 MiB of weights, so loading a second
 * copy per test would exhaust memory long before it found a bug.
 */
describe.skipIf(MODEL_DIR === undefined || MODEL_DIR === "")("runtime hazards", () => {
  const agents = new Map<string, MlxAgent>();

  async function agentFor(options: {
    dtype: "float16" | "float32";
    shapeless?: boolean;
    tidy?: boolean;
  }): Promise<MlxAgent> {
    const key = JSON.stringify(options);
    const existing = agents.get(key);
    if (existing !== undefined) return existing;
    const { load } = await import("../packages/mlayax/src/index.js");
    const agent = load(MODEL_DIR as string, options);
    agents.set(key, agent);
    return agent;
  }

  const STATE = "I was billed twice. Please refund the duplicate today.";
  const QUESTIONS = {
    department: {
      type: "choice",
      instructions: "Which team should handle this request?",
      criteria: { billing: "invoices, payments", technical: "bugs and outages" },
    },
  };

  it("hazard 4: two identical forwards in a row (the second is a cache hit)", async () => {
    const agent = await agentFor({ dtype: "float16" });
    const prepared = agent.prepare(STATE, QUESTIONS);
    const first = await agent.forwardItems(prepared.items);
    // A capture that disposed masks *after* caching them crashed here with a bare
    // `std::invalid_argument` on the second call.
    const second = await agent.forwardItems(prepared.items);
    expect(Array.from(second.logits)).toEqual(Array.from(first.logits));
  });

  it("hazard 4: a different length after a cache hit", async () => {
    const agent = await agentFor({ dtype: "float16" });
    const long = agent.prepare(STATE, QUESTIONS);
    await agent.forwardItems(long.items);
    const short = agent.prepare("refund", QUESTIONS);
    const result = await agent.forwardItems(short.items);
    expect(result.logits.every(Number.isFinite)).toBe(true);
    // And back to the first length, which must still hit its own cached entry.
    const again = await agent.forwardItems(long.items);
    expect(again.logits.every(Number.isFinite)).toBe(true);
  });

  it("hazard 5: the shapeless profile runs any length and agrees to within 1e-3", async () => {
    // The default profile uses fast.sdpa and compiles per shape; shapeless uses manual attention so
    // one graph serves every length. The trade is documented as 12-15 % slower — what must *not*
    // differ is the answer.
    const fused = await agentFor({ dtype: "float16" });
    const shapeless = await agentFor({ dtype: "float16", shapeless: true });
    expect(shapeless.shapeless).toBe(true);
    const seen = new Set<number>();
    const chosen = (prediction: { answers: Record<string, MlxAnswer> }): unknown => {
      const answer = prediction.answers.department;
      return answer?.type === "choice" ? answer.choice : undefined;
    };
    for (const state of [STATE, "The invoice is a duplicate charge and the account is locked."]) {
      const a = await fused.predict(state, QUESTIONS);
      const b = await shapeless.predict(state, QUESTIONS);
      seen.add(a.usage.state_tokens);

      // Identical numbers are *not* the contract. Manual attention accumulates in float32 to match
      // the fused kernel, but rounds differently: PORTING.md measures the shapeless profile at
      // Δ 1e-3 in fp16 and 9e-4 in fp32 against the Python reference, where the default profile is
      // bit-exact in fp32. Measured here at Δ 3e-4 between the two profiles, so 1e-3 is the honest
      // gate. What must hold absolutely is that no shape error occurs and the decision does not move.
      const numeric = compare(a.answers, b.answers, 1e-3);
      expect(
        numeric.mismatches,
        describeComparison(`shapeless vs default (${state.slice(0, 20)}…)`, numeric),
      ).toEqual([]);
      expect(chosen(b)).toBe(chosen(a));
    }
    // The two states really are different lengths, so the profile was exercised.
    expect(seen.size).toBeGreaterThan(1);
  });

  it("hazard 9: tidy off with asyncEval, repeatedly", async () => {
    // The service path: `mx.tidy` around an in-flight async evaluation is unsafe, so the batching
    // layer disposes its feeds explicitly and leaves tidy off. This asserts that stays stable.
    const agent = await agentFor({ dtype: "float16", tidy: false });
    const prepared = agent.prepare(STATE, QUESTIONS);
    for (let round = 0; round < 12; round += 1) {
      const result = await agent.forwardItems(prepared.items, { asyncEval: true, tidy: false });
      expect(result.logits.every(Number.isFinite)).toBe(true);
    }
  });

  it("reports usage, including truncation and collapsed options", async () => {
    const agent = await agentFor({ dtype: "float16" });
    const prediction = await agent.predict(STATE, QUESTIONS);
    expect(prediction.model).toBe("laya-rl-agent");
    expect(prediction.usage.output_tokens).toBe(0);
    expect(prediction.usage.state_tokens).toBeGreaterThan(0);
    expect(prediction.usage.input_tokens).toBeGreaterThan(prediction.usage.state_tokens);
    expect(prediction.usage.truncated).toBe(false);
    expect(prediction.usage.truncated_questions).toEqual([]);

    // A long state must report truncation, and the affected question must be named.
    const long = await agent.predict("billing refund duplicate charge ".repeat(200), QUESTIONS);
    expect(long.usage.truncated).toBe(true);
    expect(long.usage.truncated_questions).toEqual(["department"]);
    expect(long.usage.state_tokens_dropped).toBeGreaterThan(0);
  });
});

/**
 * The other half of Phase 3's acceptance line: `load()` from the **Hub cache**, not just a directory.
 *
 * Gated on the cache existing rather than on `MLAYAX_MODEL_DIR`, because that is the whole point — the
 * caller hands over a repository id and the runtime finds the checkpoint itself. This is the test
 * that would catch a revision not being recorded, or the cache lookup disagreeing with the directory
 * loader.
 */
const CACHED_HUB_MODEL = (() => {
  try {
    return resolveCachedModel(REPO_ID);
  } catch {
    return null;
  }
})();

/**
 * The cache existing is not enough: loading it needs the native payload too. Without this second gate
 * these tests run on a machine that has the checkpoint cached but no built addon, and fail for a
 * reason that has nothing to do with the Hub.
 */
const NATIVE_AVAILABLE = (() => {
  try {
    resolveNativeAddonPath();
    return true;
  } catch {
    return false;
  }
})();

describe.skipIf(CACHED_HUB_MODEL === null || !NATIVE_AVAILABLE)(
  "load() by repository id from the Hub cache",
  () => {
    it("resolves the cache, records the revision, and matches the reference", async () => {
      const { load } = await import("../packages/mlayax/src/index.js");
      const agent = load(REPO_ID, { dtype: "float32" });

      // Provenance is recorded, and it agrees with what the cache resolver found.
      expect(agent.revision).toBe(CACHED_HUB_MODEL?.revision);
      expect(agent.sourcePath).toBe(CACHED_HUB_MODEL?.path);

      const reference = loadReference(referencePath("fp32"));
      const first = reference.cases[0];
      if (first === undefined) throw new Error("reference has no cases");
      const prediction = await agent.predict(first.state, first.questions);
      const comparison = compare(first.answers, prediction.answers, 0);
      expect(comparison.mismatches, describeComparison("hub cache load", comparison)).toEqual([]);
      expect(comparison.maxDelta).toBe(0);
    }, 120_000);

    it("loadAsync is a no-op on a warm cache", async () => {
      const { loadAsync } = await import("../packages/mlayax/src/index.js");
      const agent = await loadAsync(REPO_ID, { dtype: "float16" });
      expect(agent.revision).toBe(CACHED_HUB_MODEL?.revision);
    }, 120_000);
  },
);
