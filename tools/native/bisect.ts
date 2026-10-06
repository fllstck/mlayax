/** Which op inside one real encoder layer breaks shapeless compilation? */
import mlx from "@frost-beta/mlx";
import { MlxAgent } from "./agent.ts";

const mx = (mlx as any).core;
const agent = MlxAgent.load(process.env.LAYA_MODEL_DIR ?? "models/english-mlx", {
  dtype: "float16",
  compile: false,
});
const model: any = agent.model;

const state = "I was billed twice and would like the duplicate charge refunded today.";
const q = {
  a: { type: "choice", instructions: "Which team?", criteria: { billing: "invoices", technical: "bugs", sales: "purchases" } },
  b: { type: "noul", instructions: "Money back?" },
};
const mk = (questions: Record<string, any>) => {
  const { items } = agent.prepare(state, questions);
  const feeds = agent.collate(items);
  return { feeds, masks: agent.maskInputs(items, feeds.attentionMask) };
};
const small = mk({ a: q.a });
const big = mk(q);

const P = "encoder.layers.0.attn";
const parts: Array<[string, (f: any, m: any) => any]> = [
  ["x", (f) => model.ln(model.w["encoder.embeddings.tok_embeddings.weight"].index(f.inputIds), "encoder.embeddings.norm", model.cfg.norm_eps)],
  ["proj (pre-split weight)", (f) => model.linSplit(f.x, P, "q")],
  ["proj+unflatten+transpose", (f) => model.toHeads(model.linSplit(f.x, P, "q"))],
  ["+rope", (f) => mx.fast.rope(model.toHeads(model.linSplit(f.x, P, "q")), model.headDim, false, 10000.0, 1.0, 0)],
  ["+sdpa(mask input)", (f, m) => {
    const qq = model.toHeads(model.linSplit(f.x, P, "q"));
    const kk = model.toHeads(model.linSplit(f.x, P, "k"));
    const vv = model.toHeads(model.linSplit(f.x, P, "v"));
    const r = (t: any) => mx.fast.rope(t, model.headDim, false, 10000.0, 1.0, 0);
    return mx.fast.scaledDotProductAttention(r(qq), r(kk), vv, model.headDim ** -0.5, m.full);
  }],
  ["whole attention op", (f, m) => model.attention(f.x, P, "full_attention", m.full)],
];

// Build the "x" input per case from the collated tensors.
const withX = (mkOne: (f: any, m: any) => any) => (f: any, m: any) => {
  const x = model.ln(model.w["encoder.embeddings.tok_embeddings.weight"].index(f.inputIds), "encoder.embeddings.norm", model.cfg.norm_eps);
  return mkOne({ ...f, x }, m);
};

for (const [name, fn] of parts) {
  const wrapped = withX(fn);
  try {
    const compiled = mx.compile(wrapped, true);
    mx.eval(compiled(small.feeds, small.masks));
    const got = compiled(big.feeds, big.masks);
    const want = wrapped(big.feeds, big.masks);
    mx.eval(got, want);
    const same = got.shape.join() === want.shape.join();
    let delta = "";
    if (same) {
      const d = mx.max(mx.abs(mx.subtract(got, want)));
      mx.eval(d);
      delta = ` max|Δ| ${(d as any).tolist()}`;
    }
    console.log(name.padEnd(26), same ? `OK ${JSON.stringify(got.shape)}${delta}` : `SHAPE MISMATCH ${JSON.stringify(got.shape)} vs ${JSON.stringify(want.shape)}`);
  } catch (error: any) {
    console.log(name.padEnd(26), `THROWS: ${String(error.message).slice(0, 64)}`);
  }
}