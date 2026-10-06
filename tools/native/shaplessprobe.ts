/**
 * Which MLX ops can infer output shapes under shapeless compilation?
 *
 * Each case is compiled once with shapeless=true and then called with a different shape; a case
 * that cannot infer shapes throws at the call, a case that infers but is wrong returns a wrong
 * shape. This is the checklist for writing a shape-polymorphic graph.
 *
 *   node src/mlx/shaplessprobe.ts
 */

import mlx from "@frost-beta/mlx";

const mx = (mlx as any).core;

const cases: Array<[string, (x: any, pos: any) => any]> = [
  ["reshape [-1, H]", (x) => x.reshape([-1, 1024])],
  ["unflatten -1 -> [8,128]", (x) => mx.unflatten(x, -1, [8, 128])],
  ["flatten -2", (x) => mx.flatten(mx.unflatten(x, -1, [8, 128]), -2)],
  ["transpose [0,2,1,3] (4-D)", (x) => mx.flatten(mx.transpose(mx.unflatten(x, -1, [8, 128]), [0, 2, 1, 3]), -2)],
  ["expandDims -1", (x) => mx.expandDims(x, -1)],
  ["slice static (0..512)", (x) => x.index("...", mx.Slice(0, 512))],
  ["index '...', 0", (x) => x.index("...", 0)],
  ["takeAlongAxis (3-D)", (x, pos) => mx.flatten(mx.takeAlongAxis(mx.expandDims(x, 1), mx.expandDims(pos, -1), 2), -2)],
  ["gather by index array (const)", (x) => mx.array([1, 2, 3, 4], mx.float16).index(mx.zeros([2, 3], mx.int32))],
  ["concatenate -1", (x) => mx.concatenate([x, x], -1)],
  ["stack -1", (x) => mx.stack([mx.max(x, -1), mx.min(x, -1)], -1)],
  ["sort -1", (x) => mx.sort(x, -1)],
  ["topk 2 -1", (x) => mx.topk(x, 2, -1)[0]],
  ["max/min of stack", (x) => mx.min(mx.topk(x, 2, -1)[0], -1)],
  ["squeeze -1", (x) => mx.squeeze(mx.max(x, -1), -1)],
  ["softmax -1", (x) => mx.softmax(x, -1)],
  ["sum -1", (x) => mx.sum(x, -1)],
  ["mean -1", (x) => mx.mean(x, -1)],
  ["where with scalar", (x) => mx.where(mx.greater(x, 0), x, -1e4)],
  ["erf / log / divide", (x) => mx.divide(mx.erf(x), mx.log(mx.maximum(x, 1e-9)))],
  ["fast.layerNorm", (x) => mx.fast.layerNorm(x, mx.ones([1024], mx.float16), null, 1e-5)],
  ["fast.rope (4-D)", (x) => mx.fast.rope(mx.unflatten(x, -1, [8, 128]), 128, false, 10000.0, 1.0, 0)],
  ["fast.sdpa (4-D)", (x) => mx.fast.scaledDotProductAttention(mx.unflatten(x, -1, [8, 128]), mx.unflatten(x, -1, [8, 128]), mx.unflatten(x, -1, [8, 128]), 1.0)],
];

const shapes: Array<[number, number]> = [
  [1, 1024],
  [3, 1024],
];

console.log("op".padEnd(26), "shapeless result");
for (const [name, fn] of cases) {
  try {
    const compiled = mx.compile((x: any, pos: any) => fn(x, pos), true);
    const a = mx.random.normal([shapes[0][0], shapes[0][1]], mx.float16);
    const posA = mx.zeros([1, 1], mx.int32);
    mx.eval(compiled(a, posA));
    // Second call with a different shape: shapeless must reuse the graph.
    const b = mx.random.normal([shapes[1][0], shapes[1][1]], mx.float16);
    const out = compiled(b, mx.zeros([3, 2], mx.int32));
    mx.eval(out);
    const expected = fn(b, mx.zeros([3, 2], mx.int32));
    mx.eval(expected);
    const same = out.shape.join() === expected.shape.join();
    console.log(name.padEnd(26), same ? `OK ${JSON.stringify(out.shape)}` : `WRONG ${JSON.stringify(out.shape)} vs ${JSON.stringify(expected.shape)}`);
  } catch (error: any) {
    console.log(name.padEnd(26), `THROWS: ${String(error.message).replace("[Primitive::output_shapes] ", "").slice(0, 48)}`);
  }
}