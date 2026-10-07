# mlayax

[![CI](https://github.com/fllstck/mlayax/actions/workflows/ci.yml/badge.svg)](https://github.com/fllstck/mlayax/actions/workflows/ci.yml)

A **Laya typed-decision runtime for Node and Bun on Apple Silicon**, in TypeScript on
[MLX](https://github.com/ml-explore/mlx). No Python, no `pyproject.toml`, no subprocess.

`@fllstck/mlayax` answers typed questions about a state — a ticket, an email, a form — and returns
calibrated probabilities instead of free text: a label for `choice`, an expected level for `score`, a
probability for `noul`, each with a confidence and an escalation probability for the action head. It is
an **independent port** of Laya's decision model, measured against the Python implementation it ports.

```bash
npm i @fllstck/mlayax     # or: bun add @fllstck/mlayax
```

```ts
import { load } from "@fllstck/mlayax";

const agent = load("aac6fef/laya-mlx"); // the Hub checkpoint, resolved from the local cache

const { answers, usage } = await agent.predict(
  "I was billed twice. Please refund the duplicate today.",
  {
    department: {
      type: "choice",
      instructions: "Which team should handle this request?",
      criteria: { billing: "invoices, payments, refunds", technical: "bugs and outages" },
    },
    refund: { type: "noul", instructions: "Does the customer ask for money back?" },
  },
);
// answers.department → { type: "choice", choice: "billing", probabilities: { billing: 0.9587, … },
//                        confidence: 0.8175, answer_confidence: 0.9587, action: { act_probability: 1 } }
```

Answering those three questions takes **9.9 ms** for one short question and **15.1 ms** for three, warm,
on an M5. The full API — question construction, every option, the answer and `usage` shapes, and the
"two MLX builds in one process" trap — is in
[`packages/mlayax/README.md`](packages/mlayax/README.md), which is also the page npm shows.

## What you get

| package | contents | size |
|---|---|---|
| [`@fllstck/mlayax`](packages/mlayax) | TypeScript only: prompt construction, calibration, answer shaping, the MLX runtime layer (node-mlx's MIT JavaScript, vendored unmodified), Hugging Face fetcher, mixing guard | 237.5 KiB unpacked |
| [`@fllstck/mlayax-darwin-arm64`](packages/mlayax-darwin-arm64) | native payload: `node_mlx.node` + `libmlx.dylib` + `libjaccl.dylib` + `mlx.metallib` + `SHA256SUMS` + `VERSION` | 64.9 MiB tarball |

The model weights are Apache-2.0 (Convai Innovations) and are **never bundled**: they are downloaded to
the Hugging Face cache on first use (≈ 807 MiB once — 803 MiB of weights plus the configs and
tokenizer) by `loadAsync`.

## Requirements

Apple Silicon (`darwin`/`arm64`); **macOS ≥ 26.2**; Node ≥ 22 or Bun ≥ 1.2. The floor is set by the
pinned MLX build, not by us — `libmlx.dylib` declares `minos 26.2`, so on an older system it is that
library which refuses to load. Intel Macs, Linux and Windows are out of scope for 0.1.0.

## Performance

The numbers this repository is judged by. One 39-token state, fp16, Apple M5, Node 24.15 and Bun
1.3.13 (they measure the same):

| metric | mlayax | Python `laya_mlx` (reference) |
|---|---:|---:|
| one short `choice` question | **9.9 ms** | 10.2 ms |
| three questions | **15.1 ms** | 15.4 ms |
| 16 rows in one forward | **59.1 ms** (271 q/s) | 58.4 ms |
| throughput plateau at ≥ 8 rows | ~312 q/s | — |
| RSS, one model resident | ~0.95–1.0 GiB | — |
| fp32 parity vs Python, per field | **bit-exact** (61/61) | — |
| fp16 parity | Δ ≤ 4e-4 (48/61 exact) | — |

The harness for all of it is [`bench/`](bench) (`npm run bench`, `npm run bench:check`), gated at
1.3x against [`bench/baseline.json`](bench/baseline.json) — warn above 1.3x, fail above 2x, and
"inconclusive" rather than a verdict when the machine is not the baseline's CPU.

## How it works

```
your code
  └── @fllstck/mlayax                 prompt construction → tokenizer → forward → calibration → answers
        ├── src/core/                 pure TypeScript: no MLX, no filesystem. Unit-tested everywhere
        ├── src/mlx/                  load()/predict() over the binding + the load-time mixing guard
        ├── src/hub.ts                Hugging Face fetch, cache-compatible with huggingface_hub
        └── vendor/node-mlx/          node-mlx's MIT JavaScript layer, vendored unmodified
              └── @fllstck/mlayax-darwin-arm64   node_mlx.node ← libmlx.dylib + mlx.metallib (pinned)
```

Three decisions define the shape of it, all argued with measurements in
[`docs/ECOSYSTEM.md`](docs/ECOSYSTEM.md):

- **A compiled binding, not FFI.** Every maintained alternative (`@johnhenry/backend-mlx`,
  `@nielspeter/mlx-ts`, `mlx-bun`) is FFI over mlx-c and issues one JS call per op, so `mlx_compile`
  cannot remove the call overhead: measured at 183.5 ms for the 16 rows this does in 59.1 ms.
- **Link a current MLX.** The popular `@frost-beta/mlx` vendors MLX 0.25 (last published 2025-04-19);
  we patch node-mlx for MLX 0.32.3 (`tools/native/node-mlx-mlx32.patch`, 10 files) rather than wait.
- **The MLX *build* is a first-class requirement.** The Python wheel's `mlx.metallib` (190 MB) is ~35 %
  faster than Homebrew's (137 MB) for the same version, so the build script refuses the slow class.

The port itself — what it took, what was measured, and three retracted claims — is
[`docs/PORTING.md`](docs/PORTING.md).

## Development

```bash
npm install
npm run verify          # biome ci && tsc -b && tsc -p tsconfig.test.json && vitest run && licences
npm run verify:release  # + publint, attw, tarball size/content gate, native payload check
npm test                # vitest, no native payload required (native tests skip themselves)
bun test                # the same suite under Bun

npm run build           # tsc -b → packages/mlayax/dist
npm run bench           # the benchmark harness (needs MLAYAX_MODEL_DIR)
npm run bench:check     # the gate: p50 within 1.3x of bench/baseline.json, RSS under the ceiling
npm run fetch:checkpoint  # download the reference checkpoint (807 MiB), print its path
npm run check:packed    # parity against *installed* tarballs (--project <clean-room>)
```

The native payload is built, not committed:

```bash
python3 -m venv .venv && .venv/bin/pip install mlx==0.32.3   # the wheel-class MLX (mode 1)
tools/native/build.sh                                        # clones node-mlx, patches, builds, verifies
tools/native/build.sh --mode source                          # builds MLX from source (needs full Xcode)
tools/native/build.sh --check                                # verify an already-staged payload
```

### CI

Eight jobs in [`.github/workflows/ci.yml`](.github/workflows/ci.yml): `lint-typecheck` (ubuntu),
`unit` (ubuntu, Node 22+24), `native` (mode-1 payload, its own assertions, the load-time safety tests,
the artifact), `unit-macos` (Node 22+24 against that artifact: tiny-fixture parity, the mixing guard
against a real dyld), `native-source` (mode 2, main only), `bun`, `pack` (publint, attw, size gate, then
a clean-room install of both tarballs and a real request through it), and an opt-in `bench`. Coverage is
**not** wired in: `npm run test:coverage` cannot pass yet, and a permanently red check is worse than no
check — see [`TASKS.md`](TASKS.md) §10.5 and §10.10.

## Documentation

| | |
|---|---|
| [`packages/mlayax/README.md`](packages/mlayax/README.md) | install, quickstart, API reference, question/answer reference, troubleshooting |
| [`docs/PORTING.md`](docs/PORTING.md) | the technical record: what the port took, every measurement, three retracted claims |
| [`docs/ECOSYSTEM.md`](docs/ECOSYSTEM.md) | the survey of maintained alternatives and why this is a fork with its own artifact |
| [`TASKS.md`](TASKS.md) | the plan, the quality gates, the hazards, and the corrections found while building it |
| [`CHANGELOG.md`](CHANGELOG.md) | 0.1.0, with the numbers and the corrections |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | development, the release gates, and the MLX bump playbook |
| [`SECURITY.md`](SECURITY.md) | scope, reporting, and what this package does and does not trust |

## Roadmap

Out of scope for 0.1.0 and listed rather than promised: WebGPU and CPU backends, an ONNX backend,
browser support, an embedding shortlist, quantized checkpoints, training, the Snake demo, and a
request-batching service (the seam is public; shipping a server is not the plan).

## Licensing

MIT for our code. MLX is MIT (Apple), node-mlx is MIT (frost-beta), and the Laya weights are
Apache-2.0 (Convai Innovations) — downloaded, never bundled. See [`NOTICE`](NOTICE). This is an
independent port, not an official Laya release.