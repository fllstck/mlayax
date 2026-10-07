# `v0.1.0` — GitHub release body

Paste everything below the rule into the GitHub release for tag `v0.1.0`. It is the CHANGELOG entry
expanded with the numbers a reader of a release wants in front of them, and it is written to be read
*without* the repository — so it repeats the compatibility matrix and links the upstream projects
rather than assuming the reader has the README open.

Checklist before pasting: confirm the tag matches `package.json` in both packages, confirm the version
line has no `-` suffix (npm's `0.1.0` is the version; `v0.1.0` is the tag), and re-run the pre-flight
in `TASKS.md` §5 Phase 9 against the tagged commit.

---

## `@fllstck/mlayax` 0.1.0

An independent TypeScript port of **Laya**'s typed-decision runtime, running on Apple's **MLX** through
a patched node-mlx native binding. No Python, no `pyproject.toml`, no subprocess.

Ask it typed questions about a state — a ticket, an email, a form — and it answers with calibrated
probabilities: a label for `choice`, an expected level for `score`, a probability for `noul`, each with
a confidence and an escalation probability for the action head.

```bash
npm i @fllstck/mlayax     # or: bun add @fllstck/mlayax
```

```ts
import { load } from "@fllstck/mlayax";

const agent = load("aac6fef/laya-mlx");

const { answers, usage } = await agent.predict("I was billed twice. Please refund the duplicate today.", {
  department: {
    type: "choice",
    instructions: "Which team should handle this request?",
    criteria: { billing: "invoices, payments, refunds", technical: "bugs and outages" },
  },
  refund: { type: "noul", instructions: "Does the customer ask for money back?" },
});

// answers.department → { type: "choice", choice: "billing",
//                        probabilities: { billing: 0.9587, technical: 0.0227 },
//                        confidence: 0.8175, answer_confidence: 0.9587,
//                        action: { act_probability: 1 } }
```

### Performance

One 39-token state, fp16, Apple M5, macOS 26.6.2, Node 24.15.0 and Bun 1.3.13 (both measure the same).
p50 over 10 samples after a warm-up call per shape, from `npm run bench`:

| case | mlayax | Python `laya_mlx` (the implementation this ports) |
|---|---:|---:|
| one short `choice` question | **10.3 ms** | 10.2 ms |
| three questions (a `choice`, a `score`, a `noul`) | **15.1 ms** | — |
| 16 rows in one forward | **53.7 ms** (298 q/s) | 58.4 ms |
| throughput plateau at ≥ 8 rows | ~312 q/s | — |
| RSS, one model resident | **987 MiB** | — |
| first call at a new sequence length | ~25–35 ms (one `mx.compile` trace) | — |

The right-hand column is the reference implementation's own measurement, and `—` means it was not
measured there: Python is 10.2 ms for one short question and 58.4 ms for 16 rows. The committed gate
baseline is this port's earlier single-call measurement (10.2 / 15.4 / 59.1 ms, 271 q/s, RSS ceiling
1400 MiB), so the released build measures **0.91x** on the 16-row case — which is what
`npm run bench:check` reports on the machine these numbers come from.

For scale: the fastest maintained TypeScript alternative that does not use a compiled binding measures
183.5 ms for the same 16 rows — 3.1x slower, because an FFI binding still issues every op from
JavaScript, so `mlx_compile` cannot remove the call overhead. The reasoning is measured, not asserted:
[`docs/ECOSYSTEM.md`](https://github.com/fllstck/mlayax/blob/main/docs/ECOSYSTEM.md).

### Correctness

Verified against the Python reference over three cases (a three-question set, a 12-option question that
exercises the clamped `choice:11+` bucket, and a case with structured state, dict-valued criteria and
custom `noul` labels), 61 numeric fields per dtype:

| dtype | result |
|---|---|
| fp32 | **bit-exact** — Δ 0 on 61/61 fields, every case |
| fp16 | Δ ≤ 4e-4 (48/61 fields exact) |

The same parity is asserted in CI against a synthetic checkpoint committed as a fixture, so it runs
without a download.

### What arrives, and what does not

| package | contents | size |
|---|---|---|
| `@fllstck/mlayax` | TypeScript only: prompt construction, calibration, answer shaping, the MLX runtime layer (node-mlx's MIT JavaScript, vendored unmodified), Hugging Face fetcher, mixing guard | 237.5 KiB unpacked |
| `@fllstck/mlayax-darwin-arm64` | `node_mlx.node` (patched for MLX 0.32.3) + `libmlx.dylib` + `libjaccl.dylib` + `mlx.metallib` (wheel class, 190 MB) + `SHA256SUMS` + `VERSION` | 68.0 MB packed, 215.5 MB unpacked |

Neither package has an install script, and neither bundles weights. The checkpoint downloads on first
use (**807 MiB** once — 803 MiB of weights plus configs and tokenizer) into the Hugging Face cache, in
the `huggingface_hub` layout, so an existing Python cache is reused rather than duplicated.

### Compatibility

| | |
|---|---|
| platform | Apple Silicon (`darwin`/`arm64`). Intel Macs, Linux and Windows are out of scope for 0.1.0 |
| macOS | **≥ 26.2** — set by the pinned MLX build, which declares `minos 26.2`, not by us |
| Node | ≥ 22 |
| Bun | ≥ 1.2 (verified on 1.3.13) |
| MLX | pinned: MLX `v0.32.3`, node-mlx `4bf8b1d` + this repository's patch |
| MLX build | wheel-class `mlx.metallib` (190 MB). The build refuses the ~35%-slower Homebrew class at load |

### Two things that will bite you, and what the library does about them

- **Two MLX runtimes in one process is fatal.** macOS resolves dynamic libraries by install name, and
  virtually every MLX distribution published for Node names itself `@rpath/libmlx.dylib`, so a foreign
  build already resident silently takes our addon's symbols. This is checked *before* loading and
  reported as two file paths instead of a mangled-symbol crash from dyld.
- **`confidence` from the checkpoint's `choice:11+` bucket is clamped** — that bucket's temperature sits
  outside the `[0.5, 5]` range the calibration clamps to. The library raises a `RuntimeWarning` rather
  than returning a number that looks calibrated and is not.

### Corrections carried into this release

Recorded so they are not re-derived; the long form is in
[`docs/PORTING.md`](https://github.com/fllstck/mlayax/blob/main/docs/PORTING.md) and
[`TASKS.md`](https://github.com/fllstck/mlayax/blob/main/TASKS.md) §10.

- **"fp16 drifts up to 3.3e-2 from fp32"** — wrong: the harness compared two different prompts. Matched
  inputs differ by ~2e-4. The figures above are from matched inputs.
- **"the 2x gap is the vendored MLX version"** — wrong. Rebuilding for MLX 0.32.3 recovered about 10 % of
  it; the rest was the choice of MLX *build* (~35 %) and a JavaScript scalar-dtype trap — `mx.array(1)` is
  float32, so a literal silently upcasts fp16 activations (~20 %, with no error).
- **`round4` was not banker's rounding** — it was a real parity bug, disagreeing with CPython on 120 of
  8 017 sampled values. Fixed and pinned.

### Known limitations

- `npm run test:coverage` does not pass: `src/mlx` branch coverage is 73.8 % against an 85 % threshold,
  and the missing branches are the forward pass's error and fallback paths. It is deliberately **not**
  wired into CI rather than being permanently red.
- WebGPU and CPU backends, an ONNX backend, browser support, an embedding shortlist, quantized
  checkpoints, training and the Snake demo are out of scope for 0.1.0.
- This is a library, not a service: there is no HTTP server and no batching service. The seam for
  caller-side coalescing (`prepare` → `forwardItems`) is public.

### About this release

Published **manually**, so the tarballs carry **no npm provenance attestation** (that needs CI + OIDC).
If provenance is wanted later, a release workflow is the way.

Independent port, not an official Laya release. Our code is MIT; MLX is MIT (Apple), node-mlx is MIT
(frost-beta), and the Laya weights are Apache-2.0 (Convai Innovations) — downloaded, never bundled.

| | |
|---|---|
| checkpoint used in the examples | [`aac6fef/laya-mlx`](https://huggingface.co/aac6fef/laya-mlx) on the Hub |
| upstream weights | [`Convai-Innovations/laya`](https://huggingface.co/Convai-Innovations/laya) (gated) |
| Python reference this ports | [`mizorewww/laya-mlx`](https://github.com/mizorewww/laya-mlx) |
| MLX | [`ml-explore/mlx`](https://github.com/ml-explore/mlx) v0.32.3 |
| binding | [`frost-beta/node-mlx`](https://github.com/frost-beta/node-mlx) @ `4bf8b1d` + `tools/native/node-mlx-mlx32.patch` |
| technical record | [`docs/PORTING.md`](https://github.com/fllstck/mlayax/blob/main/docs/PORTING.md) |
| why a fork with its own artifact | [`docs/ECOSYSTEM.md`](https://github.com/fllstck/mlayax/blob/main/docs/ECOSYSTEM.md) |