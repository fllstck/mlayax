/**
 * Batch scaling: how per-token cost falls with more rows in one forward.
 * At batch 1 the forward streams the whole 842 MB of weights for 39 tokens; every extra row
 * reuses them. Also measures effective weight bandwidth and the steady-state cost per row.
 *
 *   MLAYAX_MODEL_DIR=/path/to/english-mlx node bench/batchscale.ts
 *
 * Not a gate — the gate is `bench/gate.ts` (§6). This is the harness you run when you want to know
 * *why* a gate number moved: it is what established §2's ~312 q/s plateau at 8 rows and above.
 *
 * Ported from the spike's `src/mlx/batchscale.ts`; the API calls are this package's (`load`,
 * `predict`, `loadMx`) rather than the spike's (`MlxAgent.load`, `systemOne`, `@frost-beta/mlx`).
 */

import { load } from "../packages/mlayax/dist/index.js";
import { loadMx, type MlxArray } from "../packages/mlayax/dist/mlx/index.js";

const mx = loadMx();
const runtime = typeof (globalThis as { Bun?: unknown }).Bun !== "undefined" ? "bun" : "node";

const modelDir = process.env.MLAYAX_MODEL_DIR;
if (modelDir === undefined || modelDir === "") {
  throw new Error("MLAYAX_MODEL_DIR is not set — this harness needs the real checkpoint.");
}
const agent = load(modelDir, { dtype: "float16" });

// Total bytes of 2-D weights read once per forward.
let weightBytes = 0;
for (const value of Object.values(agent.weights)) {
  const array = value as { ndim?: number; size?: number };
  if (array.ndim === 2) weightBytes += Number(array.size ?? 0) * 2;
}

const state = "I was billed twice. Please refund the duplicate today.";
const rows: Array<{ batch: number; p50: number; perRow: number; qps: number; p99: number }> = [];
for (const n of [1, 2, 4, 8, 16, 32, 48]) {
  const questions: Record<string, unknown> = {};
  for (let i = 0; i < n; i++) {
    questions[`q${i}`] = {
      type: "choice",
      instructions: `Which team should handle reason number ${i}?`,
      criteria: { billing: "invoices", technical: "bugs", sales: "purchases" },
    };
  }
  await agent.predict(state, questions); // warm (compile trace)
  const samples: number[] = [];
  for (let i = 0; i < 15; i++) {
    const t = performance.now();
    await agent.predict(state, questions);
    samples.push(performance.now() - t);
  }
  samples.sort((a, b) => a - b);
  const p50 = samples[Math.floor(samples.length / 2)] ?? Number.NaN;
  rows.push({
    batch: n,
    p50: Number(p50.toFixed(2)),
    perRow: Number((p50 / n).toFixed(2)),
    qps: Math.round((n * 1000) / p50),
    p99: Number((samples[samples.length - 1] ?? Number.NaN).toFixed(2)),
  });
}

// Effective weight bandwidth of a memory-bound op, for context.
//
// `zeros` is not on `MlxCore`: `types.ts` types only the surface the runtime calls, deliberately.
// This harness is not the runtime, so it declares the one extra op it needs rather than widening the
// shared type and eroding that invariant.
const bandwidthMx = mx as unknown as { zeros: (shape: number[], dtype: unknown) => MlxArray };
const big = bandwidthMx.zeros([64 * 1024 * 1024], mx.float16);
const t2 = performance.now();
mx.eval(mx.add(big, big));
const copyMs = performance.now() - t2;
mx.eval(big);

console.log(
  JSON.stringify(
    {
      runtime,
      weightMiB: Math.round(weightBytes / 1024 / 1024),
      bandwidthGBps: Math.round((2 * 64 * 1024 * 1024 * 2) / copyMs / 1e6),
      rows,
    },
    null,
    1,
  ),
);