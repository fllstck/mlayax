/**
 * Variable-length traffic: what does compiled inference cost when the sequence length keeps
 * changing? Reports the cold call (trace) and the warm call for each of many distinct states.
 *
 *   node src/mlx/varibench.ts [distinct-states]
 *
 * LAYA_COMPILE=1, LAYA_LENGTH_BUCKET=32 to compare configurations.
 */

import { MlxAgent } from "./agent.ts";

const states = Number(process.argv[2] ?? 12);
const env = process.env;

const agent = MlxAgent.load(env.LAYA_MODEL_DIR ?? "models/english-mlx", {
  dtype: "float16",
  tidy: true,
  compile: env.LAYA_COMPILE === "1",
  lengthBucket: env.LAYA_LENGTH_BUCKET ? Number(env.LAYA_LENGTH_BUCKET) : 0,
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
  await agent.systemOne(state, question);
  cold.push(performance.now() - t);
  t = performance.now();
  await agent.systemOne(state, question);
  warm.push(performance.now() - t);
}
const med = (xs: number[]) => {
  const s = [...xs].sort((a, b) => a - b);
  return Number(s[Math.floor(s.length / 2)].toFixed(2));
};
const sum = (xs: number[]) => Number(xs.reduce((a, b) => a + b, 0).toFixed(1));
const cacheKey = performance.now();

console.log(
  JSON.stringify({
    config:
      [
        env.LAYA_COMPILE === "1" ? "compile" : "eager",
        env.LAYA_LENGTH_BUCKET ? `bucket=${env.LAYA_LENGTH_BUCKET}` : "",
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