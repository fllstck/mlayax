/**
 * Chain probe: which shape transformations actually stay polymorphic under shapeless compilation?
 *
 * Each case is compiled once (shapeless) and then called with a *different* (b, L); the result is
 * compared shape- and value-wise against the eager computation. A case that returns the old shape
 * or wrong numbers is a construct the model must avoid.
 */

import mlx from "@frost-beta/mlx";

const mx = (mlx as any).core;
const H = 8;
const HEADS = 2;
const HD = H / HEADS;

const w = mx.multiply(mx.random.normal([H, H], mx.float16), 0.1);
const emb = mx.multiply(mx.random.normal([64, H], mx.float16), 0.1);
mx.eval(w, emb);

const cases: Array<[string, (ids: any, mask: any) => any]> = [
  ["unflatten->transpose->flatten", (ids) =>
    mx.flatten(mx.transpose(mx.unflatten(mx.matmul(emb.index(ids), mx.transpose(w)), -1, [HEADS, HD]), [0, 2, 1, 3]), -2)],
  ["reshape [-1, H]", (ids) => mx.matmul(emb.index(ids), mx.transpose(w)).reshape([-1, H])],
  ["takeAlongAxis + reshape -1", (ids) => {
    const h = mx.matmul(emb.index(ids), mx.transpose(w)); // [b, L, H]
    const idx = mx.multiply(mx.sum(mx.ones([1, 1], mx.int32), -1), 0); // [1]
    return mx.reshape(
      mx.takeAlongAxis(h, mx.expandDims(mx.expandDims(idx, -1), 0), 1),
      [-1, H],
    );
  }],
  ["sdpa with mask input", (ids, mask) => {
    const q = mx.transpose(mx.unflatten(mx.matmul(emb.index(ids), mx.transpose(w)), -1, [HEADS, HD]), [0, 2, 1, 3]);
    const out = mx.fast.scaledDotProductAttention(q, q, q, HD ** -0.5, mask);
    return mx.flatten(mx.transpose(out, [0, 2, 1, 3]), -2);
  }],
  ["layerNorm then matmul", (ids) =>
    mx.matmul(mx.fast.layerNorm(emb.index(ids), mx.ones([H], mx.float16), null, 1e-5), mx.transpose(w))],
  ["layerNorm->unflatten->transpose->flatten", (ids) =>
    mx.flatten(
      mx.transpose(
        mx.unflatten(mx.fast.layerNorm(emb.index(ids), mx.ones([H], mx.float16), null, 1e-5), -1, [HEADS, HD]),
        [0, 2, 1, 3],
      ),
      -2,
    )],
  ["rope on 4-D", (ids) => {
    const q = mx.transpose(mx.unflatten(emb.index(ids), -1, [HEADS, HD]), [0, 2, 1, 3]);
    return mx.flatten(mx.transpose(mx.fast.rope(q, HD, false, 10000.0, 1.0, 0), [0, 2, 1, 3]), -2);
  }],
  ["sdpa after rope + layerNorm", (ids, mask) => {
    const x = mx.fast.layerNorm(emb.index(ids), mx.ones([H], mx.float16), null, 1e-5);
    const q = mx.transpose(mx.unflatten(mx.matmul(x, mx.transpose(w)), -1, [HEADS, HD]), [0, 2, 1, 3]);
    const r = mx.fast.rope(q, HD, false, 10000.0, 1.0, 0);
    const out = mx.fast.scaledDotProductAttention(r, r, q, HD ** -0.5, mask);
    return mx.flatten(mx.transpose(out, [0, 2, 1, 3]), -2);
  }],
  ["sort/max/min over markers", (ids) => mx.max(mx.matmul(emb.index(ids), mx.transpose(w)), -1)],
];

const shapes: Array<[number, number]> = [
  [1, 3],
  [4, 7],
];

for (const [name, fn] of cases) {
  try {
    const compiled = mx.compile((ids: any, mask: any) => fn(ids, mask), true);
    const [b0, l0] = shapes[0];
    const [b1, l1] = shapes[1];
    const ids0 = mx.zeros([b0, l0], mx.int32);
    const ids1 = mx.zeros([b1, l1], mx.int32);
    const mask0 = mx.zeros([b0, 1, 1, l0], mx.bool);
    const mask1 = mx.zeros([b1, 1, 1, l1], mx.bool);
    mx.eval(compiled(ids0, mask0));
    const got = compiled(ids1, mask1);
    const want = fn(ids1, mask1);
    mx.eval(got, want);
    const shapeOk = got.shape.join() === want.shape.join();
    let valueOk = shapeOk;
    if (shapeOk) {
      const d = mx.max(mx.abs(mx.subtract(got, want)));
      mx.eval(d);
      valueOk = (d as any).tolist() < 1e-3;
    }
    console.log(
      name.padEnd(32),
      shapeOk ? (valueOk ? `OK ${JSON.stringify(got.shape)}` : "WRONG VALUES") : `WRONG SHAPE ${JSON.stringify(got.shape)} vs ${JSON.stringify(want.shape)}`,
    );
  } catch (error: any) {
    console.log(name.padEnd(32), `THROWS: ${String(error.message).slice(0, 60)}`);
  }
}