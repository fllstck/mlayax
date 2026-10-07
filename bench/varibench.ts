/**
 * Variable-length traffic: what does compiled inference cost when the sequence length keeps
 * changing? Reports the cold call (trace) and the warm call for each of many distinct states.
 *
 *   MLAYAX_MODEL_DIR=/path/to/english-mlx node bench/varibench.ts [distinct-states]
 *
 * Not a gate — the gate is `bench/gate.ts` (§6). This is the harness because of which §9's `compile`
 * decision is "specialized by default, shapeless behind MLAYAX_SHAPELESS=1": it is the one that
 * measures the trade-off. Set MLAYAX_SHAPELESS=1 to compare against the shapeless profile, and
 * MLAYAX_BENCH_LENGTH_BUCKET=32 or MLAYAX_BENCH_TIDY=0 to compare the other options.
 *
 * Ported from the spike's `src/mlx/varibench.ts`; the API calls and environment variable names are
 * this package's.
 */

import { load } from "../packages/mlayax/dist/index.js";

const states = Number(process.argv[2] ?? 12);
const env = process.env;

const modelDir = env.MLAYAX_MODEL_DIR;
if (modelDir === undefined || modelDir === "") {
  throw new Error("MLAYAX_MODEL_DIR is not set — this harness needs the real checkpoint.");
}
const lengthBucket = env.MLAYAX_BENCH_LENGTH_BUCKET ? Number(env.MLAYAX_BENCH_LENGTH_BUCKET) : 0;
const agent = load(modelDir, {
  dtype: "float16",
  tidy: env.MLAYAX_BENCH_TIDY !== "0",
  lengthBucket,
});

const question = {
  department: {
    type: "choice",
    instructions: "Which team should handle this request?",
    criteria: { billing: "invoices", technical: "bugs", sales: "purchases" },
  },
};

// Distinct state lengths, so every call has a different (batch, length) shape.
const makeState = (i: number) =>
  `Ticket ${i}: ` +
  "we were billed twice for the same invoice and would like the duplicate charge refunded. ".repeat(1 + (i % 5)) +
  `reference ${i}`;

const cold: number[] = [];
const warm: number[] = [];
for (let i = 0; i < states; i++) {
  const state = makeState(i);
  let t = performance.now();
  await agent.predict(state, question);
  cold.push(performance.now() - t);
  t = performance.now();
  await agent.predict(state, question);
  warm.push(performance.now() - t);
}
const med = (xs: number[]) => {
  const s = [...xs].sort((a, b) => a - b);
  return Number((s[Math.floor(s.length / 2)] ?? Number.NaN).toFixed(2));
};
const sum = (xs: number[]) => Number(xs.reduce((a, b) => a + b, 0).toFixed(1));

console.log(
  JSON.stringify({
    config:
      [
        env.MLAYAX_SHAPELESS === "1" ? "shapeless" : "specialized",
        lengthBucket > 0 ? `bucket=${lengthBucket}` : "",
        env.MLAYAX_BENCH_TIDY === "0" ? "eager-dispose" : "",
      ]
        .filter(Boolean)
        .join("+"),
    distinctStates: states,
    coldMedian: med(cold),
    warmMedian: med(warm),
    coldTotal: sum(cold),
    warmTotal: sum(warm),
    total: Number((sum(cold) + sum(warm)).toFixed(1)),
    rssMiB: Math.round(process.memoryUsage().rss / 1024 / 1024),
    cold: cold.map((v) => Number(v.toFixed(1))),
    warm: warm.map((v) => Number(v.toFixed(1))),
  }),
);