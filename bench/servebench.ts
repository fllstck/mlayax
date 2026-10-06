/**
 * Load test for the batching service layer.
 *
 *   node src/mlx/servebench.ts [requests] [concurrency]     # or: bun src/mlx/servebench.ts
 *
 * A single-row forward streams ~800 MiB of weights, so batching pays when rows are short and
 * weight-bound — and much less when long states make each row compute-bound. This drives both
 * regimes and reports throughput and latency percentiles, plus a check that a batched answer is
 * identical to a solo answer.
 */

import { MlxAgent } from "./agent.ts";
import { Batcher } from "./batcher.ts";

const requests = Number(process.argv[2] ?? 400);
const concurrency = Number(process.argv[3] ?? 32);
const runtime = typeof (globalThis as any).Bun !== "undefined" ? "bun" : "node";

const agent = MlxAgent.load(process.env.LAYA_MODEL_DIR ?? "models/english-mlx", {
  dtype: "float16",
});

const QUESTIONS = {
  department: {
    type: "choice",
    instructions: "Which team should handle this request?",
    criteria: { billing: "invoices, refunds", technical: "bugs and outages", sales: "purchases" },
  },
  urgency: { type: "score", instructions: "How urgent is this?", criteria: ["low", "soon", "critical"] },
  refund: { type: "noul", instructions: "Does the customer ask for money back?" },
};

// Two traffic shapes: short states (~40 tokens, weight-bound) and long ones (~200, compute-bound).
const corpus = (repeats: number) =>
  Array.from({ length: 6 }, (_, i) => ({
    state: "Ticket: " + "we were billed twice and need the duplicate refunded today. ".repeat(repeats),
    questions: i % 2 === 0 ? QUESTIONS : { department: QUESTIONS.department },
  }));

const percentile = (sorted: number[], p: number) =>
  Number(sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * p))].toFixed(2));

async function drive(batcher: Batcher, items: any[], count: number, inFlight: number) {
  const latencies: number[] = [];
  let issued = 0;
  const started = performance.now();
  const worker = async () => {
    while (issued < count) {
      const request = items[issued++ % items.length];
      const t = performance.now();
      await batcher.submit(request.state, request.questions);
      latencies.push(performance.now() - t);
    }
  };
  await Promise.all(Array.from({ length: Math.min(inFlight, count) }, worker));
  const wall = performance.now() - started;
  latencies.sort((a, b) => a - b);
  return {
    wallMs: Number(wall.toFixed(1)),
    reqPerSec: Number(((count * 1000) / wall).toFixed(1)),
    p50: percentile(latencies, 0.5),
    p95: percentile(latencies, 0.95),
    p99: percentile(latencies, 0.99),
  };
}

const configs: Array<{ label: string; maxRows: number; windowMs: number; asyncEval: boolean }> = [
  { label: "no batching (rows=1)", maxRows: 1, windowMs: 0, asyncEval: true },
  { label: "batch 4", maxRows: 4, windowMs: 1, asyncEval: true },
  { label: "batch 16", maxRows: 16, windowMs: 1, asyncEval: true },
  { label: "batch 16, sync eval", maxRows: 16, windowMs: 1, asyncEval: false },
];

console.log(`runtime ${runtime}: ${requests} requests, concurrency ${concurrency}`);
let correctness = "";
for (const [name, repeats, note] of [
  ["short states (~40 tokens)", 1, "weight-bound"],
  ["long states (~200 tokens)", 5, "compute-bound"],
] as const) {
  const items = corpus(repeats);
  // Service latency with no queueing: one request at a time, no batching.
  const soloBatcher = new Batcher(agent, { maxRows: 1, windowMs: 0, asyncEval: true });
  await drive(soloBatcher, items, 5, 1);
  const soloSamples: number[] = [];
  for (let i = 0; i < 15; i++) {
    const t = performance.now();
    await soloBatcher.submit(items[i % items.length].state, items[i % items.length].questions);
    soloSamples.push(performance.now() - t);
  }
  soloSamples.sort((a, b) => a - b);
  console.log(
    `\n${name} [${note}]  solo latency (no queueing): ${soloSamples[7].toFixed(1)} ms`,
  );
  console.log(
    `  ${"config".padEnd(22)} ${"p50".padStart(8)} ${"p95".padStart(8)} ${"p99".padStart(8)} ${"req/s".padStart(8)} ${"fwd".padStart(6)} ${"rows/fwd".padStart(9)}`,
  );
  for (const config of configs) {
    const batcher = new Batcher(agent, {
      maxRows: config.maxRows,
      windowMs: config.windowMs,
      asyncEval: config.asyncEval,
    });
    await drive(batcher, items, 12, 4); // warm up: compile traces for this corpus
    batcher.stats.forwards = 0;
    batcher.stats.rows = 0;
    const measured = await drive(batcher, items, requests, concurrency);
    console.log(
      `  ${config.label.padEnd(22)} ${String(measured.p50).padStart(8)} ${String(measured.p95).padStart(8)} ${String(measured.p99).padStart(8)} ${String(measured.reqPerSec).padStart(8)} ${String(batcher.stats.forwards).padStart(6)} ${String(batcher.stats.meanBatchRows.toFixed(2)).padStart(9)}`,
    );
    if (repeats === 1 && config.label === "batch 16") {
      const batched = await batcher.submit(items[1].state, items[1].questions);
      const alone = await agent.systemOne(items[1].state, items[1].questions);
      correctness = JSON.stringify(batched.answers) === JSON.stringify(alone.answers) ? "yes" : "NO";
    }
  }
}
console.log(`\nbatched answer == solo answer: ${correctness}`);