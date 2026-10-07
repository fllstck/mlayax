/**
 * The runtime hazards that do not need the real checkpoint — TASKS.md §8.4 and §8.9.
 *
 * Both were covered only by `test/parity.real.test.ts`, which is checkpoint-gated and therefore
 * skipped on every ordinary run. Neither *needs* real weights: they are about mask lifetime and
 * async evaluation, and the tiny fixture exercises the same code with 61 k parameters instead of
 * 800 M. So they now run by default, and the real-checkpoint versions stay as the stronger
 * confirmation rather than the only one.
 *
 * Why these two are worth a file of their own: both failures are silent until they are catastrophic.
 *
 * - **§8.4** — disposing a mask that was just cached does not fail on the call that disposes it. It
 *   fails later, on the next *cache hit*, as a bare `std::invalid_argument` from inside MLX, with no
 *   JavaScript stack pointing at the cause.
 * - **§8.9** — `mx.tidy` around an in-flight `mx.asyncEval` is unsafe. Nothing detects it; it
 *   corrupts memory later. The documented safe shape (what the dropped batching service used) is
 *   `tidy: false` with the feeds disposed explicitly, which `forwardItems` does.
 *
 * `forwardItems` is the seam these reach, and it is public — TASKS.md §4 keeps it public precisely so
 * a caller can build coalescing without us shipping a service.
 */

import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import type { MlxAgent, Prepared } from "../packages/mlayax/src/index.js";
import { resolveNativeAddonPath } from "../packages/mlayax/src/mlx/binding.js";

const FIXTURE = fileURLToPath(new URL("./fixtures/tiny", import.meta.url));

const STATE = "I was billed twice. Please refund the duplicate today.";
const QUESTIONS = {
  department: {
    type: "choice",
    instructions: "Which team should handle this request?",
    criteria: { billing: "invoices, payments, refunds", technical: "bugs and outages" },
  },
};
/** A different state, so the padded length differs and the mask cache key changes. */
const SHORT_STATE = "refund";

function nativeAvailable(): boolean {
  try {
    resolveNativeAddonPath();
    return true;
  } catch {
    return false;
  }
}

describe.skipIf(!nativeAvailable())("runtime hazards, by default (tiny fixture)", () => {
  const agents = new Map<string, MlxAgent>();

  /** One agent per configuration: loading the fixture is cheap, but not free, and this keeps it tidy. */
  async function agentFor(options: Record<string, unknown> = {}): Promise<MlxAgent> {
    const key = JSON.stringify(options);
    const existing = agents.get(key);
    if (existing !== undefined) return existing;
    const { load } = await import("../packages/mlayax/src/index.js");
    const agent: MlxAgent = load(FIXTURE, { dtype: "float32", ...options });
    agents.set(key, agent);
    return agent;
  }

  describe("hazard 4: a cached mask must outlive the call that cached it", () => {
    it("two identical forwards in a row — the second is a cache hit", async () => {
      const agent = await agentFor();
      const prepared: Prepared = agent.prepare(STATE, QUESTIONS);
      const first = await agent.forwardItems(prepared.items);
      // A version that disposed masks *after* caching them returned fine here and then died on this
      // second call, from inside MLX, with nothing in the stack pointing back at the dispose.
      const second = await agent.forwardItems(prepared.items);
      expect(Array.from(second.logits)).toEqual(Array.from(first.logits));
    });

    it("a different length after a cache hit, then back to the first", async () => {
      // The cache key carries the row lengths as well as (batch, length), so a shorter state is a
      // miss rather than a corrupted hit — and returning to the first length must still find its own
      // entry, which is where a too-aggressive eviction would show up.
      const agent = await agentFor();
      const long: Prepared = agent.prepare(STATE, QUESTIONS);
      const longFirst = await agent.forwardItems(long.items);

      const short: Prepared = agent.prepare(SHORT_STATE, QUESTIONS);
      const shortResult = await agent.forwardItems(short.items);
      expect(shortResult.logits.every(Number.isFinite)).toBe(true);

      const longAgain = await agent.forwardItems(long.items);
      expect(Array.from(longAgain.logits)).toEqual(Array.from(longFirst.logits));
    });

    it("gives the same logits with the mask cache disabled", async () => {
      // The cache is an optimisation; with it off the masks are rebuilt every call. If these ever
      // disagreed, the cache would be returning a stale mask rather than failing outright.
      const cached = await agentFor();
      const uncached = await agentFor({ cacheMasks: false });
      const prepared = cached.prepare(STATE, QUESTIONS);
      const a = await cached.forwardItems(prepared.items);
      const b = await uncached.forwardItems(uncached.prepare(STATE, QUESTIONS).items);
      expect(Array.from(b.logits)).toEqual(Array.from(a.logits));
    });
  });

  describe("hazard 9: tidy around an in-flight async evaluation", () => {
    it("is stable across repeated async forwards with tidy off", async () => {
      // The documented safe shape. A crash here would be a use-after-free well after the loop, so the
      // assertion is partly that this *returns* twelve times and partly that the numbers stay finite.
      const agent = await agentFor({ tidy: false });
      const prepared = agent.prepare(STATE, QUESTIONS);
      for (let round = 0; round < 12; round += 1) {
        const result = await agent.forwardItems(prepared.items, { asyncEval: true, tidy: false });
        expect(result.logits.every(Number.isFinite), `round ${round}`).toBe(true);
      }
    });

    it("produces the same logits as the synchronous path", async () => {
      // The hazard is a memory-safety one, but if the async path also disagreed numerically it would
      // be worth knowing before anyone builds on it.
      const agent = await agentFor({ tidy: false });
      const sync = await agent.forwardItems(agent.prepare(STATE, QUESTIONS).items, { tidy: false });
      const async_ = await agent.forwardItems(agent.prepare(STATE, QUESTIONS).items, {
        asyncEval: true,
        tidy: false,
      });
      expect(Array.from(async_.logits)).toEqual(Array.from(sync.logits));
    });

    it("reports a stable resident set size over many async forwards", async () => {
      // TASKS.md §8.9 asks for "no crash and stable RSS". A leak in this path is invisible in the
      // numbers and only shows as growth, so the shape of the assertion has to be about RSS.
      const agent = await agentFor({ tidy: false });
      const prepared = agent.prepare(STATE, QUESTIONS);
      const rss = (): number => process.memoryUsage().rss / 1024 / 1024;

      for (let round = 0; round < 4; round += 1)
        await agent.forwardItems(prepared.items, { asyncEval: true, tidy: false });
      const warm = rss();
      for (let round = 0; round < 40; round += 1) {
        await agent.forwardItems(prepared.items, { asyncEval: true, tidy: false });
      }
      const after = rss();

      // A generous bound on purpose: MLX's own buffer cache grows and does not shrink, and the
      // fixture is small next to that. This catches a per-iteration leak (40 × anything of size),
      // not allocator noise.
      expect(after - warm, `rss ${warm.toFixed(0)} → ${after.toFixed(0)} MiB`).toBeLessThan(256);
    });
  });

  describe("the eager path is exercised too", () => {
    it("forwardItems works without a compiled graph", async () => {
      // `compile: false` takes the other side of `forwardItems`' `compiledForward ? … : …`, which the
      // default configuration never reaches. Covered for its numbers in test/mlx.options.test.ts;
      // covered here for the branch itself, through the seam that exposes it.
      const agent = await agentFor({ compile: false });
      const result = await agent.forwardItems(agent.prepare(STATE, QUESTIONS).items);
      expect(result.logits.every(Number.isFinite)).toBe(true);
      expect(result.count).toBeGreaterThan(0);
    });
  });
});
