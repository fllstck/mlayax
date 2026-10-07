# @fllstck/mlayax

A **Laya typed-decision runtime for Node and Bun on Apple Silicon**, in TypeScript on MLX.
No Python, no `pyproject.toml`, no subprocess.

Give it a state (a ticket, an email, a form) and a set of typed questions, and it answers each one
with a calibrated probability — a label for `choice`, an expected level for `score`, a probability
for `noul` — plus an escalation probability for the action head. It is the Laya decision model, ported
to TypeScript and run on Apple's MLX through a native binding, at 10 ms per short question.

This package is the published façade. It contains TypeScript/JavaScript only — no native code, no
weights, no install scripts. The native MLX payload is delivered by the optional dependency
[`@fllstck/mlayax-darwin-arm64`](https://www.npmjs.com/package/@fllstck/mlayax-darwin-arm64), which npm
installs automatically on Apple Silicon. This is an **independent port**, not an official Laya
release; see [Licensing](#licensing).

> **This is a library, not a service.** There is no HTTP server and no subpath export that implies
> one. Batching belongs to the caller, who is the only one who can see their own concurrency — the
> seam for it (`prepare` → `forwardItems` → answer shaping) is public so you can build coalescing
> without this package shipping a server.

## Install

```bash
npm i @fllstck/mlayax
# or
bun add @fllstck/mlayax
```

Two packages arrive: this one (≈ 240 KiB unpacked) and the native payload (≈ 65 MiB, uncompressed
213 MiB — it is mostly `mlx.metallib`). The model weights are **not** shipped: the checkpoint is
downloaded to the Hugging Face cache on first use (**≈ 807 MiB** once — 803 MiB of weights plus the
configs and tokenizer), and reused offline afterwards.

## Requirements

| | |
|---|---|
| CPU / OS | Apple Silicon (`darwin`/`arm64`) only. Intel Macs, Linux and Windows are out of scope for 0.1.0. |
| macOS | **≥ 26.2** |
| Node | ≥ 22 |
| Bun | ≥ 1.2 (measured on 1.3.13) |
| MLX | pinned: MLX `v0.32.3`, node-mlx `4bf8b1d` + the patch in this repository |

macOS 26.2 is not our constraint: the MLX build we link declares `minos 26.2`, so on an older system
it is `libmlx.dylib` that refuses to load. Check the claim for yourself:

```bash
otool -l node_modules/@fllstck/mlayax-darwin-arm64/lib/libmlx.dylib | grep -A2 LC_BUILD_VERSION
```

## Quickstart

```ts
import { load } from "@fllstck/mlayax";

const agent = load("aac6fef/laya-mlx"); // cache hit, or a clear error telling you to use loadAsync

const { answers, usage } = await agent.predict(
  "I was billed twice. Please refund the duplicate today.",
  {
    department: {
      type: "choice",
      instructions: "Which team should handle this request?",
      criteria: {
        billing: "invoices, payments, refunds",
        technical: "bugs and outages",
        sales: "new purchases",
      },
    },
    urgency: {
      type: "score",
      instructions: "How urgent is this request?",
      criteria: ["not urgent", "soon", "critical"],
    },
    refund: {
      type: "noul",
      instructions: "Does the customer ask for money back?",
    },
  },
);
```

Real output, from the checkpoint on the Hub, fp16, on an M5 (`Node 24.15`):

```jsonc
{
  "department": {
    "type": "choice",
    "choice": "billing",                    // the winning label
    "probabilities": { "billing": 0.9587, "technical": 0.0227, "sales": 0.0186 },
    "confidence": 0.8175,                   // normalized entropy, [0, 1]
    "answer_confidence": 0.9587,            // mass on the reported answer
    "action": { "act_probability": 1 }      // escalate rather than auto-apply
  },
  "urgency": {
    "type": "score",
    "score": 1.3603,                        // expected level
    "legend": { "0": "not urgent", "1": "soon", "2": "critical" },
    "probabilities": { "0": 0.1528, "1": 0.3341, "2": 0.5131 },
    "confidence": 0.0936,
    "answer_confidence": 0.5131,
    "action": { "act_probability": 1 }
  },
  "refund": {
    "type": "noul",
    "noul": 0.8215,                         // probability of the `true` side
    "confidence": 0.8215,
    "answer_confidence": 0.8215,
    "action": { "act_probability": 1 }
  }
}
```

`usage` reports what the tokenizer did with the request:

```jsonc
{
  "input_tokens": 132,
  "output_tokens": 0,
  "state_tokens": 11,
  "state_tokens_dropped": 0,
  "truncated": false,
  "truncated_questions": []
  // "options": { "category": { "total": 12, "distinct": 12, "tokens_per_option": null } }  // only when options collapsed
}
```

Those three questions took **34.6 ms** on the first call (a new sequence length is a fresh `mx.compile`
trace) and **15.1 ms** warm (p50 of 20 samples); one `choice` question alone is **9.9 ms** warm. The
published reference numbers in [Performance](#performance) were measured the same way.

## Prompt construction

`state` is either a string or any JSON-serialisable object (`{from, subject, body}`, a form, a
record). Objects are rendered deterministically the way Python's `json` renders them, because the
prompt is the model's input: key order, floats and integral values all have to match the reference
implementation. `questions` is an object of question id → definition; the ids become the keys of the
answer object, so `predict` rejects an empty one rather than letting you correlate the response by
position.

| kind | `criteria` | answer | notes |
|---|---|---|---|
| `choice` | `{ label: "description", … }` or `{ label: {desc, …} }` | `choice` + `probabilities` | a dict, so **order is the criteria order**; each option is rendered as `label: description` |
| `score` | `["not urgent", "soon", "critical"]` | `score` + `legend` + `probabilities` | an ordered list of levels; `score` is `Σ i·pᵢ` |
| `noul` | optional `{ true: …, false: … }` | `noul` | a two-sided question ("does the customer ask for money back?"); `labels: { true: "yes", false: "no" }` overrides the rendered option text |

Limits worth knowing, all reported in `usage` rather than thrown:

- **Options are not capped at 11.** The calibration temperature depends on the option count, and the
  checkpoint's table has a `choice:11+` bucket, which is what the 12-option test case exercises.
- **Duplicate option texts collapse** onto one marker (they tokenise identically). `usage.options` then
  reports `distinct < total` so the collapsed probability can be recognised rather than puzzled over.
- **Long states are truncated** from the state side: the sequence is capped at 512 tokens and each
  question's header at 192, and state tokens are the only elastic part. `usage.truncated` and
  `usage.truncated_questions` say what was dropped.
- The number of questions in one request is not limited by the API. They are answered in `batchSize`
  chunks, and rows within a chunk are independent (padding is masked), so batching cannot change an
  answer — `test/parity.tiny.test.ts` asserts it.

## API reference

### `load(source, options?)`

Synchronously load a checkpoint. `source` is a local directory or a repository id
(`"aac6fef/laya-mlx"`). For a repository id it resolves the **local Hugging Face cache** and throws if
there is none — it never downloads, so a call that looks synchronous cannot silently fetch 807 MiB.

### `await loadAsync(source, options?)`

The same, but downloads into the cache first when the cache is cold. The resolved commit is recorded
on the agent as `revision`, so a report can name exactly which checkpoint produced an answer.

### Options

Every option has a default. The first group is load-time; the second only applies when `source` is a
repository id.

| option | default | what it does |
|---|---|---|
| `dtype` | `"float16"` | weight precision: `"float16"`, `"float32"`, `"bfloat16"`. The reference numbers are fp16; fp32 is the bit-exact-parity mode |
| `batchSize` | `16` | question rows per forward |
| `device` | `"gpu"` | `"cpu"` is for debugging — far slower |
| `compile` | `true` | wrap the forward in `mx.compile`; removes per-op JS call overhead (−12 %) |
| `tidy` | `true` | `mx.tidy` around each forward. Turn it off if you dispose your own feeds: `mx.tidy` must not wrap an in-flight `mx.asyncEval` |
| `cacheMasks` | `true` | reuse attention masks for a repeated `(batch, length, row lengths)` |
| `shapeless` | `false` | compile once for every shape. Removes cold-call traces (~25 ms) and costs 12–15 % per forward (`MLAYAX_SHAPELESS=1`) |
| `attention` | follows `shapeless` | force `"fused"` or `"manual"` attention |
| `lengthBucket` | `0` (off) | round the padded length up to a multiple of *n*: collapses traces onto few shapes, costs +30 % on a token-bound batch of 16 |
| `pretranspose`, `addmm` | `false` | measured as no effect on this workload; kept because they are the knobs you reach for first (`MLAYAX_PRETRANSPOSE=1`, `MLAYAX_ADDM=1`) |
| `wired`, `cacheLimit` | unset | `mx.setWiredLimit` / `mx.setCacheLimit`, in bytes |
| `token` | unset | Hub access token for private or gated repos. Sent to the Hub, **never** to the CDN a `resolve` URL redirects to |
| `revision` | `"main"` | branch, tag or 40-hex commit |
| `offline` | `false` | never touch the network; use the cache or fail |
| `cacheDir` | Python default | what `HF_HUB_CACHE` points at; reuses an existing Python cache |
| `endpoint` | `https://huggingface.co` | Hub base URL |
| `fetch` | `globalThis.fetch` | injectable, for a proxy or a fixture server |
| `onProgress` | unset | `{phase, file, filesDone, filesTotal, bytesDone, bytesTotal}` |

### `agent.predict(state, questions)`

Answers every question in one request. Returns `{ model: "laya-rl-agent", answers, usage }`.

### The agent

| member | what it is |
|---|---|
| `dtypeName`, `batchSize`, `tidy`, `cacheMasks`, `shapeless`, `lengthBucket` | the effective configuration, including anything taken from the environment |
| `sourcePath`, `revision` | where the weights came from, and which commit (`null` for a local directory) |
| `prepare(state, questions)` | the tokenised request **before** the forward: ids, markers, per-question diagnostics. Use it to inspect what the model will actually be asked — or to build request coalescing |
| `forwardItems(items, {asyncEval, tidy})` | one forward over prepared rows, flattened to the host. `asyncEval` keeps the event loop responsive while the GPU works |

Nothing else is required to answer a question; those two are public because the port's own tests and
benchmarks drive them, and because a caller who wants batching has no other way to reach the seam.

### Diagnostics

`resolveNativeAddonPath()` returns the addon this process would load (and throws the same described
errors a load would), which is what you want in a bug report. `isMxLoaded()` answers whether the
runtime has been loaded yet, without loading it.

## Performance

One short `choice` question, and 16 rows in a single forward, fp16, on an Apple M5 (Node 24.15 /
Bun 1.3.13 — both measure the same):

| case | time |
|---|---|
| one 39-token question (warm) | **9.9 ms** |
| three questions (a `choice`, a `score`, a `noul`) | **15.1 ms** |
| 16 rows in one forward | **59.1 ms** (271 q/s) |
| throughput plateau at ≥ 8 rows | ~312 q/s |
| RSS with one model resident | ~0.95–1.0 GiB |
| first call at a new sequence length | ~25–35 ms (one compile trace) |

For scale: the Python `laya_mlx` this ports is 10.2 ms on the same machine, and the fastest maintained
FFI alternative in TypeScript is 183.5 ms for the same 16 rows. Why, and what was measured to get
here, is in [`docs/PORTING.md`](https://github.com/fllstck/mlayax/blob/main/docs/PORTING.md) and
[`docs/ECOSYSTEM.md`](https://github.com/fllstck/mlayax/blob/main/docs/ECOSYSTEM.md).

Two caveats about the numbers. The temp buckets the checkpoint ships for `choice:11+` are outside the
`[0.5, 5]` clamp, so `confidence` from those buckets is **clamped and uncalibrated** — the library says
so with a `RuntimeWarning` rather than quietly returning a number. And the MLX *build* matters more
than the MLX *version*: the wheel-class `mlx.metallib` (190 MB) is ~35 % faster than Homebrew's
(137 MB), so `@fllstck/mlayax` ships the wheel-class build and refuses a slower one at load.

## Troubleshooting

### `Cannot find module` / `node_mlx.node` not found

The optional platform package did not install. npm skips optional dependencies on non-Apple-Silicon
platforms by design, and a `--no-optional` install skips it anywhere. Install
`@fllstck/mlayax-darwin-arm64` explicitly, or point `MLAYAX_NATIVE_DIR` at a directory holding a
built `node_mlx.node`.

### "Symbol not found", or "another libmlx.dylib is already resident"

You have two MLX runtimes in one process. macOS resolves a dynamic library by **install name**, and
virtually every MLX distribution published for Node names itself `@rpath/libmlx.dylib` — so if another
one is already loaded, dyld hands it to our addon instead of ours, and the addon dies on a symbol it
was not compiled against:

```text
dlopen(…/mlayax-darwin-arm64/lib/node_mlx.node, 0x0001): Symbol not found:
  __ZN3mlx4core10gather_qmmERKNS0_5arrayES3_S3_RKNSt3__18optionalIS1_EES6_S6_bNS5_IiEES9_
  RKNS4_12basic_stringIcNS4_11char_traitsIcEENS4_9allocatorIcEEEES8_bNS4_7variantIJ…
```

`@fllstck/mlayax` checks for this *before* loading, and reports it as two file paths instead:

```text
@fllstck/mlayax cannot load its MLX runtime: a different libmlx.dylib is already resident in this
process, and macOS resolves dynamic libraries by install name — every MLX distribution published
for Node names itself "@rpath/libmlx.dylib", so our addon would bind to that one instead of the one
we ship.

Already loaded:
  …/node_modules/@johnhenry/backend-mlx-darwin-arm64/lib/libmlx.dylib
      a different build (19.9 MB)

Ours:
  …/node_modules/@fllstck/mlayax-darwin-arm64/lib
```

The usual cause is a second MLX package in the same dependency tree: `@johnhenry/backend-mlx`,
`@nielspeter/mlx-ts`, `mlx-bun`, a `@frost-beta/mlx` that was built in place, or a Python `mlx`
wheel reached through a Python bridge (Python is visible to the guard only once its MLX is loaded).
Remove it, or keep the two in **separate processes** — once a dylib is resident, disposing its arrays
frees memory but cannot unload it, and Node never calls `dlclose`.

Under Node the check runs before the load, using the dynamic loader's own image list.
**Bun does not expose that list** — `process.report.getReport().sharedObjects` is always empty — so
there the collision can only be diagnosed from the error it produces, and the same explanation is
attached to the failure instead. Both paths produce the message above.

If you have verified that the two builds are interchangeable, `MLAYAX_ALLOW_MIXED_MLX=1` proceeds
anyway. It is rarely right: even when the symbols match, MLX looks for `mlx.metallib` next to whichever
library won, and fails with `Failed to load the default metallib` — or silently runs on a slower kernel
set, since build class predicts throughput.

The guard is deliberately precise about what counts as a conflict. A resident MLX that names itself
with an **absolute** install name (Homebrew's bottle, for example) cannot satisfy our addon's request
for `@rpath/libmlx.dylib`; it loads as a second copy, MLX still resolves correctly, and you get a
`RuntimeWarning` noting two resident runtimes rather than a failure. Only a library that can actually
take our slot is fatal, and an image whose install name cannot be read is assumed to be able to.

## Roadmap

Out of scope for 0.1.0: WebGPU and CPU backends, an ONNX backend, browser support, an embedding
shortlist, quantized checkpoints, training, and the Snake demo. The port's own record lists what was
measured and rejected, and what is still on the table — see
[`docs/PORTING.md`](https://github.com/fllstck/mlayax/blob/main/docs/PORTING.md).

## Licensing

MIT for this package. It also redistributes node-mlx's MIT JavaScript layer (unmodified) under
`vendor/`. MLX is MIT (Apple); the **Laya weights are Apache-2.0 (Convai Innovations)** and are
downloaded from the Hub, not bundled. See `NOTICE`. This is an independent port, not an official Laya
release.