/**
 * Batch scaling: how per-token cost falls with more rows in one forward.
 * At batch 1 the forward streams the whole 842 MB of weights for 39 tokens; every extra row
 * reuses them. Also measures effective weight bandwidth and the steady-state cost per row.
 *
 *   node src/mlx/batchscale.ts
 */

import mlx from "@frost-beta/mlx";
import { MlxAgent } from "./agent.ts";

const mx = (mlx as any).core;
const runtime = typeof (globalThis as any).Bun !== "undefined" ? "bun" : "node";

const agent = MlxAgent.load(process.env.LAYA_MODEL_DIR ?? "models/english-mlx", {
  dtype: "float16",
});

// Total bytes of 2-D weights read once per forward.
let weightBytes = 0;
for (const v of Object.values(agent.weights)) {
  const a: any = v;
  if (a.ndim === 2) weightBytes += Number(a.size) * 2;
}

const state = "I was billed twice. Please refund the duplicate today.";
const rows: number[] = [];
for (const n of [1, 2, 4, 8, 16, 32, 48]) {
  const questions: Record<string, any> = {};
  for (let i = 0; i < n; i++) {
    questions[`q${i}`] = {
      type: "choice",
      instructions: `Which team should handle reason number ${i}?`,
      criteria: { billing: "invoices", technical: "bugs", sales: "purchases" },
    };
  }
  await agent.systemOne(state, questions); // warm (compile trace)
  const samples: number[] = [];
  for (let i = 0; i < 15; i++) {
    const t = performance.now();
    await agent.systemOne(state, questions);
    samples.push(performance.now() - t);
  }
  samples.sort((a, b) => a - b);
  const p50 = samples[Math.floor(samples.length / 2)];
  rows.push({
    batch: n,
    p50: Number(p50.toFixed(2)),
    perRow: Number((p50 / n).toFixed(2)),
    qps: Math.round((n * 1000) / p50),
    p99: Number(samples[samples.length - 1].toFixed(2)),
  });
}

// Effective weight bandwidth of a memory-bound op, for context.
const big = mx.zeros([64 * 1024 * 1024], mx.float16);
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