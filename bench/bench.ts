/**
 * Latency harness for the numbers that define "fast enough" — TASKS.md §2.
 *
 *   MLAYAX_MODEL_DIR=/path/to/english-mlx node bench/bench.ts [--iterations N] [--json]
 *
 * Three cases, matching the reference table exactly:
 *
 * | case | reference |
 * |---|---|
 * | one short `choice` question | 10.2 ms p50 |
 * | three questions (choice + score + noul) | 15.4 ms p50 |
 * | 16-row batch | 59.1 ms, ~271 q/s |
 *
 * `bench/gate.ts` runs this and compares against `bench/baseline.json`. Run it by hand when you want
 * to see the shape of the curve rather than a pass/fail.
 *
 * The input is the same sentence and the same questions the Python measurement used, because a
 * different sentence is a different sequence length and therefore a different number. Ported from the
 * spike's `src/mlx/bench.ts`, adapted to this package's API (`load` + `predict`, not
 * `MlxAgent.load` + `systemOne`) and extended to take p50 over `--iterations` samples for all three
 * cases — §2's 16-row figure was a single call, and a single call is not a number you can gate on.
 *
 * `--json` prints one object and nothing else, which is how the gate reads it. The default output is
 * for a human.
 */

import { load } from "../packages/mlayax/dist/index.js";

/** The sentence §2 was measured with. Changing it invalidates every number below. */
const STATE = "I was billed twice. Please refund the duplicate today.";

const CHOICE_DEPARTMENT = {
  type: "choice",
  instructions: "Which team should handle this request?",
  criteria: {
    billing: "invoices, payments, refunds",
    technical: "bugs and outages",
    sales: "new purchases",
  },
};

const ONE_QUESTION = { department: CHOICE_DEPARTMENT };

const THREE_QUESTIONS = {
  department: CHOICE_DEPARTMENT,
  urgency: {
    type: "score",
    instructions: "How urgent is this request?",
    criteria: ["not urgent", "soon", "critical"],
  },
  refund: { type: "noul", instructions: "Does the customer ask for money back?" },
};

const SIXTEEN_QUESTIONS = Object.fromEntries(
  Array.from({ length: 16 }, (_, i) => [
    `q${i}`,
    { type: "noul", instructions: `Is reason number ${i} the cause?` },
  ]),
);

const args = process.argv.slice(2);
const json = args.includes("--json");
const iterations = Number(
  args.includes("--iterations") ? args[args.indexOf("--iterations") + 1] : 10,
);
if (!Number.isInteger(iterations) || iterations < 1) {
  throw new Error(`--iterations must be a positive integer, got ${String(iterations)}`);
}

const modelDir = process.env.MLAYAX_MODEL_DIR;
if (modelDir === undefined || modelDir === "") {
  throw new Error(
    "MLAYAX_MODEL_DIR is not set. The bench needs the real checkpoint — a tiny fixture would not " +
      "produce numbers comparable to TASKS.md §2. Point it at a directory such as " +
      "models/english-mlx (weights + tokenizer).",
  );
}

const runtime = (() => {
  // `Bun` is a global that only exists under Bun, so it is not on the `globalThis` type.
  const bun = (globalThis as unknown as { Bun?: { version: string } }).Bun;
  return bun === undefined ? `node ${process.version}` : `bun ${bun.version}`;
})();

const t0 = performance.now();
const agent = load(modelDir, { dtype: "float16" });
const loadMs = performance.now() - t0;

/** Sample one case `iterations` times; the first call is a warm-up and is discarded. */
async function time(
  questions: Record<string, unknown>,
): Promise<{ p50: number; min: number; samples: number[] }> {
  await agent.predict(STATE, questions); // compiles the graph for this shape
  const samples: number[] = [];
  for (let i = 0; i < iterations; i += 1) {
    const t = performance.now();
    await agent.predict(STATE, questions);
    samples.push(performance.now() - t);
  }
  samples.sort((a, b) => a - b);
  const p50 = samples[Math.floor(samples.length / 2)] ?? Number.NaN;
  return { p50, min: samples[0] ?? Number.NaN, samples };
}

const one = await time(ONE_QUESTION);
const three = await time(THREE_QUESTIONS);
const sixteen = await time(SIXTEEN_QUESTIONS);

const result = {
  runtime,
  dtype: "float16",
  modelDir,
  iterations,
  loadMs: Number(loadMs.toFixed(0)),
  oneQuestion: { p50: Number(one.p50.toFixed(1)), min: Number(one.min.toFixed(1)) },
  threeQuestions: { p50: Number(three.p50.toFixed(1)), min: Number(three.min.toFixed(1)) },
  sixteenRows: {
    p50: Number(sixteen.p50.toFixed(1)),
    min: Number(sixteen.min.toFixed(1)),
    qps: Math.round((16_000 / sixteen.p50) * 1000) / 1000,
  },
  rssMiB: Number((process.memoryUsage().rss / 1024 / 1024).toFixed(0)),
};

if (json) {
  process.stdout.write(`${JSON.stringify(result)}\n`);
} else {
  const answers = (await agent.predict(STATE, ONE_QUESTION)).answers.department;
  console.log(`${runtime}  MLX fp16  load=${result.loadMs}ms  rss=${result.rssMiB} MiB`);
  console.log(`  one short question: p50 ${result.oneQuestion.p50} ms (min ${result.oneQuestion.min})`);
  console.log(`  3 questions:        p50 ${result.threeQuestions.p50} ms (min ${result.threeQuestions.min})`);
  console.log(
    `  16-row batch:       p50 ${result.sixteenRows.p50} ms (${result.sixteenRows.qps} q/s)`,
  );
  console.log(`  a department answer: ${JSON.stringify(answers)}`);
}