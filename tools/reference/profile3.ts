/** Does explicit disposal of intermediates (mx.tidy) close the gap to Python? */
import mlx from "@frost-beta/mlx";
import { MlxAgent } from "./agent.ts";

const mx = (mlx as any).core;
const runtime = typeof (globalThis as any).Bun !== "undefined" ? "bun" : "node";

const agent = MlxAgent.load(process.env.LAYA_MODEL_DIR ?? "models/english-mlx", {
  dtype: "float16",
});
const model: any = agent.model;
const { items } = agent.prepare("I was billed twice. Please refund the duplicate today.", {
  q0: {
    type: "choice",
    instructions: "Which team should handle reason number 0?",
    criteria: { billing: "invoices", technical: "bugs", sales: "purchases" },
  },
});
const feeds = agent.collate(items);
const { inputIds, attentionMask, markerPos, markerMask, qtype } = feeds;

const time = (name: string, fn: () => any, n = 20) => {
  for (let i = 0; i < 3; i++) mx.eval(fn());
  const s: number[] = [];
  for (let i = 0; i < n; i++) {
    const t = performance.now();
    mx.eval(fn());
    s.push(performance.now() - t);
  }
  s.sort((a, b) => a - b);
  return { name, ms: Number(s[Math.floor(s.length / 2)].toFixed(2)) };
};

const masks = agent.maskInputs(items, attentionMask);
const maskObj = { full_attention: masks.full, sliding_attention: masks.local };
const out = [
  time("encoder (plain)", () => model.encoder(inputIds, maskObj)),
  time("encoder (in mx.tidy)", () => mx.tidy(() => model.encoder(inputIds, maskObj))),
  time("forward (plain)", () => model.forward(inputIds, masks.full, masks.local, markerPos, markerMask, qtype)),
  time("forward (in mx.tidy)", () =>
    mx.tidy(() => model.forward(inputIds, masks.full, masks.local, markerPos, markerMask, qtype)),
  ),
];
console.log(JSON.stringify({ runtime, out }, null, 1));