# Porting laya-mlx to TypeScript, with MLX

Two ports live here, both measured against Python `laya_mlx` on identical inputs:

| route | graph | runtime | 1 short question | 16-question batch |
|---|---|---|---:|---:|
| **A** | ONNX export of the same weights | `onnxruntime-node` (CPU) | 57.5 ms | 635 ms |
| **B** | the weights themselves, reimplemented | **MLX via node-mlx, compiled** | **10.2 ms** | **59.1 ms** |
| — | Python `laya_mlx` (reference, eager) | MLX + Python | 10.2 ms | 58.4 ms |
| — | Python `laya_mlx` (compile=True) | MLX + Python | 10.1 ms | 58.8 ms |

**Route B is the answer to "port it and keep MLX": yes, and it now matches Python.** Node 24 and
Bun 1.3 both run the TypeScript port at 10.2 ms per question — level with Python's 10.2 ms, with fp32
results identical to Python's and fp16 within 4e-4. Getting there needed two fixes I made (link the
MLX build Python uses; fix JavaScript scalar dtypes) and one the binding needed (rebuild it for
MLX 0.32.3), plus whole-model compilation. Details and the retracted claims are below.

## What it took: three corrections

### 1. The binding ships an old MLX (20.8 → 18.8 ms)

`@frost-beta/mlx` v0.4.0 vendors MLX **0.25.0**; Python had 0.32.3. Pure forward pass, identical
tensors:

| implementation | MLX | 1 row | 16 rows |
|---|---|---:|---:|
| Python | 0.32.3 | 10.4 ms | 52.3 ms |
| Python | 0.25.2 | 19.3 ms | 166.1 ms |
| TS (prebuilt binding) | 0.25.0 | 20.2 ms | 178.4 ms |

At the same MLX version the JS boundary cost only 5–7 % — so the 2x looked like a stale
dependency, and rebuilding the binding for 0.32.3 was the obvious next step.
`tools/build-node-mlx-mlx32.patch` + `tools/build-node-mlx-mlx32.sh` do exactly that: **10** files,
242 added lines, against MLX 0.32.3. (This said "14 files" until 2026-10-06; TASKS.md §10.1 records the
correction. The count matters because it is the blast radius of an MLX bump.) The changes are the API
drift between 0.25 and 0.32:

- `fast::scaled_dot_product_attention` gained `sinks` and `force_fused`;
- `Shape` became `SmallVector<int>`, so `Shape`/`SmallVector<long long>` need kizunapi converters
  (and `ThreadLocalStream` became a `StreamOrDevice` alternative);
- `conv_transpose1d/2d/3d` gained `output_padding`;
- `metal::device_info` moved to `core::device_info(Device)`; `eval_impl` became `async_eval`;
- `detail::compile_erase` now takes a compile-cache handle;
- `fft` ops gained an `FFTNorm` parameter, which breaks their function-pointer wrappers (the file
  now uses adapters);
- taking the address of an overloaded op (`&mx::floor`, `&mx::isnan`, …) became ambiguous;
  `tools/fix_overloads.py` finds them by parsing clang's deduced-pack diagnostic.

Plus one build problem: **this machine has no Xcode, only CommandLineTools**, so there is no
`metal` compiler and MLX cannot be built from source. The patch therefore links a *prebuilt* MLX
(`mlx-c`/`mlx` from Homebrew, or the libmlx that the Python `mlx` wheel ships) instead of
`add_subdirectory(deps/mlx)`.

Rebuilding bought only 20.8 → 18.8 ms. The rest came from two things I had wrong.

### 2. Which prebuilt MLX you link matters (18.8 → 14.1 ms)

Homebrew's `mlx` 0.32.3 and the Python wheel's `mlx` 0.32.3 are different *builds*: their
`mlx.metallib` files are 137 MB and 190 MB, and 28 matmuls in one graph take 15.4 ms versus
7.7 ms. Relinking the addon against the wheel's `libmlx.dylib`+`mlx.metallib`
(`tools/build-node-mlx-mlx32.sh` defaults to it) took the single-question forward from 18.8 to
14.1 ms. Homebrew's bottle is ~35 % slower on this workload; the Python wheel's is what the
reference numbers were measured against.

### 3. JavaScript scalars silently upcast fp16 to fp32 (14.1 → 11.6 ms)

The port's `gelu` was `x * (1 + erf(x / sqrt(2))) / 2` built from the literals `1`, `2` and
`Math.SQRT2`. In Python those are weakly-typed scalars and the array stays fp16. In JavaScript
`mx.array(1)` is **float32** ("JavaScript numbers are always floating-point values"), and MLX
promotes mixed operands — so every encoder layer's MLP lifted its activations to fp32, including
the 1024→4096 matmul that follows. Fixing the constants to the activation's own dtype
(`mx.array(1, x.dtype)`), the same for `relu`'s `0`, and wrapping each forward in `mx.tidy`
(JS has no refcounting, so intermediates stay alive until GC) gave 14.1 → 11.6 ms — and made fp32
parity *exact*: 61/61 compared fields, Δ 0.

This is the trap worth remembering: it costs ~20 % end-to-end, produces no error, and shows up
only as "the JS port is slower for no reason".

## Where the port stands

Sections, one 39-token sequence, 1 row, both on MLX 0.32.3 (the wheel's build):

| section | Python | TS (Node, eager) |
|---|---:|---:|
| masks | 0.36 ms | 0.57 ms |
| encoder (28 layers) | 9.75 ms | 11.37 ms |
| type embedding + decision head | 0.88 ms | 1.21 ms |
| **full forward** | **10.37 ms** | **12.49 ms** |

28 identical ops in one graph, same shapes, same MLX build:

| chain | Python | TS | ratio |
|---|---:|---:|---:|
| matmul (up+down) ×28 | 7.16 ms | 7.66 ms | 1.07x |
| layerNorm ×28 | 0.24 ms | 0.42 ms | 1.75x |
| rope+sdpa ×28 | 0.56 ms | 0.78 ms | 1.39x |
| erf+mul ×28 | 0.49 ms | 1.01 ms | 2.06x |

Matmuls — 85 % of the FLOPs — are equal. The eager residual is per-op call overhead in the binding:
a 28-layer forward issues roughly 700 ops, and the ~15 µs/op difference on the small memory-bound
ones accounts for the ~2 ms gap. Bun measures identically, so it is the binding's argument
marshalling, not the JS engine. **Whole-model compilation removes most of it** (next section).

## Improving performance: what worked, what did not

The starting point was 20.8 ms per question. Two fixes were mistakes of mine (the MLX build, the
scalar dtypes — sections 2 and 3 above) and one came from the research below: compile the model.

### The one real win: `mx.compile` (−12 %)

The [MLX compilation guide](https://ml-explore.github.io/mlx/build/html/usage/compile.html) says
compiled functions "result in smaller graphs by merging common work and fusing certain operations",
and the `laya-mlx` author's own
[PERFORMANCE_RESEARCH.md](https://github.com/mizorewww/laya-mlx/blob/main/docs/PERFORMANCE_RESEARCH.md)
lists whole-model compilation as its **P0** experiment — noting that `Agent.forward()` has no
enclosing `mx.compile` at model or block level. (That document is a static review: no benchmark was
run for it. Its P0 is now measured, here.)

Wrapping the port's forward in `mx.compile(fn, false)` — MLX caches compiled functions per input
shape, so one callable serves every sequence length:

| | eager | compiled | change |
|---|---:|---:|---:|
| 1 question | 11.3 ms | **9.9 ms** | −12 % |
| 3 questions | 17.5 ms | **15.6 ms** | −11 % |
| 16-question batch | 64.5 ms | **59.1 ms** | −8 % |

The mechanism matches the diagnosis: compile deletes per-op call overhead, which is exactly the
residual. Note that Python gains almost nothing from the same switch (10.2 → 10.1 ms) because its
per-op overhead was already small — so this is a fix for the *JavaScript* boundary, not for MLX.

`shapeless` compilation would avoid retracing on every new sequence length, but it throws here:
`[Primitive::output_shapes] Split cannot infer output shapes`. The port splits QKV and the gated-MLP
gate with `mx.split`; MLX's shape-polymorphic path needs slice/reshape instead. That is the natural
next step if per-shape retracing ever matters more than it does now.

### Shape bucketing: a trade-off, not a win

Rounding the padded length up to a multiple of 32 collapses variable-length traffic onto few shapes.
On 12 distinct state lengths (cold call = the first call of a shape, i.e. a trace):

| config | cold median | warm median | worst cold call |
|---|---:|---:|---:|
| compiled | 12.8 ms | 11.4 ms | 28.5 ms |
| compiled + bucket 32 | 12.5 ms | 11.4 ms | 20.8 ms |

But padding is not free once a batch is token-bound: at batch 16 it costs **+30 %** (59 → 82 ms)
while saving nothing in the steady state. `lengthBucket` therefore defaults to off and is opt-in for
services that would rather pay padding than cold-call spikes.

### Measured, no effect (so they are off or absent)

Each of these was within ±0.15 ms of the baseline, single-question and batched:

| knob | what it should have done |
|---|---|
| pre-transposed, materialised weights | avoid 56 transpose nodes per forward |
| `mx.addmm` for biased linears | fuse matmul+add (what Python's `nn.Linear` does) |
| cached attention masks | avoid rebuilding `(b,1,L,L)` masks per call |
| `mx.setWiredLimit(4 GB)` | stop weights being paged out |
| `mx.setCacheLimit(256 MB)` | bound the free-buffer cache |
| `MLX_METAL_FAST_SYNCH=1` | the faster Metal CPU/GPU sync path |

MLX already folds the transposes and the mask cost is 0.2 ms; the memory knobs need a workload that
actually pages. Keeping the code without them is the better trade.

### Rejected: weight quantization

Upstream's P1 is quantizing the backbone. At Laya's shapes (39 tokens) the brief is not favourable:

| shape (M×N×K) | fp16 | 8-bit | speedup | max abs logit drift |
|---|---:|---:|---:|---:|
| 39×3072×1024 (Wqkv) | 0.59 ms | 0.39 ms | **1.52x** | 0.018 |
| 39×8192×1024 (gated up) | 0.35 ms | 0.39 ms | 0.89x | 0.017 |
| 39×1024×4096 (down) | 0.28 ms | 0.39 ms | 0.71x | 0.027 |

8-bit helps one projection, hurts two, and moves logits by ~1–3 % — on a model whose product *is* its
calibrated probability. 4-bit drifts 0.27–0.53, which is not usable without retraining and the
quality gates upstream asks for. This matches MLX's own
[quantized-matmul-is-slower-at-small-batches report](https://github.com/ml-explore/mlx/issues/3086)
and upstream's warning that "actual M3 Max speed can regress". Also worth knowing before trying:
the action head's first linear has input width `D+4 = 1028`, which is not divisible by any supported
quantization group size, so it must be excluded.

### Still on the table

A shape-inferable `fast.sdpa` upstream (which would make shapeless compilation free — see below);
fused residual+LayerNorm or GELU+gate kernels (upstream's P2 — `mx.fast.metal_kernel` is *not*
implemented in this binding, so it would mean C++ work); state-token and question-template caching
(throughput for repeated rubrics); deduplicating identical requests (upstream calls this a separate
product feature); and the prefix cache in `prepared.py`.

## Service layer: request batching (worth 2.5x)

> **Not shipped — kept as the technical record.** `batcher.ts` and `server.ts` were dropped in Phase 4:
> this is a library, and a library cannot see its caller's concurrency. The numbers below are the
> reason the seam stayed public (`prepare` → `forwardItems` → `shapeItems`), so a caller who needs
> coalescing can build it without us shipping a server. They are measurements of the spike, not a
> description of this package's API, and TASKS.md §2 lists the 2.5x gain as "not applicable".

A one-row forward streams the whole 803 MiB of 2-D weights to answer 39 tokens, so a service that
answers one question per forward pays that cost per question. `src/mlx/batcher.ts` gathers the
question rows of concurrent requests into one forward:

```
submit(state, questions) -> tokenize now (cheap, keeps the window clear)
                         -> queue the question rows
flush (window elapsed or maxRows reached)
                         -> one collate + one compiled forward for up to maxRows rows
                         -> calibrate each row, resolve each request
```

Rows belong to different requests, but the encoder is bidirectional *within* a row only and padding
is masked out, so batching cannot change an answer — `src/mlx/servebench.ts` asserts that a batched
answer equals the solo one (it does).

### Measured, 300 requests at concurrency 32 (Node 24; Bun matches within noise)

Short states (~40 tokens: weight-bound, where batching should pay):

| config | req/s | p50 | p95 | p99 | rows/forward |
|---|---:|---:|---:|---:|---:|
| rows=1 (no batching) | 50.2 | 637 ms | 638 ms | 639 ms | 1.00 |
| batch 4 | 104.8 | 305 ms | 306 ms | 306 ms | 4.00 |
| **batch 16** | **126.8** | 251 ms | 253 ms | 286 ms | 15.79 |
| batch 16, sync eval | 126.0 | 253 ms | 255 ms | 316 ms | 15.79 |

Long states (~200 tokens: compute-bound, where it should not):

| config | req/s | p50 | p95 | p99 |
|---|---:|---:|---:|---:|
| rows=1 | 41.0 | 780 ms | 781 ms | 782 ms |
| batch 4 | 55.4 | 577 ms | 581 ms | 583 ms |
| batch 16 | 63.9 | 500 ms | 501 ms | 583 ms |

**2.5x on short states, 1.6x on long ones.** The p50 figures are queueing time at concurrency 32,
not service time: served alone, a request takes ~10 ms (short states) to ~26 ms (long states, and
1–3 questions). `maxRows` buys throughput with p50 latency, so the right setting depends on whether
you are latency- or capacity-bound.

### What it costs, per request

| path | solo p50 |
|---|---:|
| `agent.systemOne` (direct) | 9.9 ms |
| through the batcher, `asyncEval`, window 0 | 11.2 ms |
| through the batcher, sync eval, window 0 | 11.7 ms |
| through the batcher, `asyncEval`, window 1 ms | 13.0 ms |

The batcher adds ~1.3 ms per request (timer, queue bookkeeping, promise), and a 1 ms coalescing
window another ~1 ms. `asyncEval` (which keeps the event loop responsive during GPU work) is
*slightly faster* than `mx.eval` here, so it is the default. Feathers worth noting: `tidy` is
disabled on this path because disposing intermediates around an in-flight async evaluation is not
worth the risk, and the per-batch input tensors are disposed explicitly after the readout.

### The service

`src/mlx/server.ts` wraps the batcher in an HTTP service (node:http only, so Node and Bun both run
it unchanged): `POST /predict`, `GET /metrics` (throughput, latency percentiles, rows per forward,
RSS), `GET /health`. A burst of 8 concurrent requests collapses into a single 8-row forward.

```bash
node src/mlx/server.ts            # or: bun src/mlx/server.ts
PORT=8787 LAYA_MAX_ROWS=16 LAYA_WINDOW_MS=1 node src/mlx/server.ts
curl -s localhost:8787/predict -H 'content-type: application/json' \
  -d '{"state":"I was billed twice. Please refund the duplicate.",
       "questions":{"department":{"type":"choice","instructions":"Which team handles this?",
       "criteria":{"billing":"invoices","technical":"bugs","sales":"purchases"}}}}'
node src/mlx/servebench.ts 300 32     # the load test above
```

### Shapeless compilation: possible now, but not the win it looked like

The retrace on a first-seen shape is the one rough edge of compiled inference (≈1.5–4 ms per shape,
and a 25 ms first call in variable-length traffic). Shapeless compilation removes it — MLX then
reuses one graph for every shape. Getting the port there was most of this task, and the result is
that **shapeless is supported but off by default**, because it costs 12–15 %:

| | per-shape compile + `fast.sdpa` (default) | shapeless + manual attention |
|---|---:|---:|
| 3 questions | 15.0 ms | 16.6 ms |
| 16-question batch | 58.5 ms | 61.3 ms |
| service, batch 16, short states | 131.2 req/s | 116.5 req/s |
| service, batch 16, long states | 68.5 req/s | 58.0 req/s |
| worst cold call, 12 unseen lengths | 25.3 ms | 18.5 ms |
| fp16 parity | Δ 4e-4 (48/61 exact) | Δ 1e-3 (38/61 exact) |
| fp32 parity | **Δ 0 (61/61 exact)** | Δ 9e-4 |

**Why manual attention is needed.** MLX's fused `fast.sdpa` is not shape-inferable: under shapeless
compilation it bakes the traced shape, so a later call with a new length fails with
`Shapes (2,1,1,43) and (1,16,37,37) cannot be broadcast` — the mask has the new shape, the scores
still have the old one. A manual `matmul → mask → softmax → matmul`, accumulated in float32 to
match the fused kernel, is shape-inferable but ~2x slower at L=64 and ~5x at L=256.

**What MLX can and cannot infer** (`src/mlx/shaplessprobe.ts` measures this; the honest answer is
that the non-inferable set is larger than expected):

| shape-inferable | cannot infer |
|---|---|
| `reshape` (`-1` ok), `unflatten`, `flatten`, `expandDims`, `transpose` | **anything with a slice**: `x[S]`, `index(…, i)`, `mx.take(x, i, axis)` |
| `concatenate`, `stack`, `softmax`, `sum`, `mean`, `max`/`min`, `where` | `mx.split` |
| `erf`/`log`/`divide`, `fast.layerNorm`, `fast.rope` | `fast.scaledDotProductAttention` |
| gather by an integer *array* (`w.index(ids)`) | |

**The refactor that made the graph polymorphic** (it is what the default profile runs too, and it
is slightly faster than before: 15.4 → 15.0 ms for three questions):

- the fused QKV projection and the gated MLP are **pre-split into separate weights at load time**
  (eager, where slicing is fine) and computed as separate matmuls — mathematically identical, since
  they were contiguous output blocks;
- marker rows are gathered with `takeAlongAxis(h, expandDims(pos, -1), 1)`. Note `mx.take` with a
  `[b, count]` index prepends the batch dim instead (`[b, b, count, H]`);
- the top-2 probabilities are built from `max` plus a `where` for ties instead of `sort(...)[..., -2:]`;
- the CLS state is gathered with a zero index tensor derived from the marker mask, not `take(h, 0, 1)`.

Two traps on the way: a capture that disposed attention masks *after* caching them (a later cache
hit then used freed arrays → `std::invalid_argument` crash), and the fp16 manual attention, which
cost 9e-4 of fp32 parity until the accumulation was moved to float32.

**Recommendation:** keep the default. MLX caches a compiled graph per shape, so traces are one-off
and amortise in a service, while the 12–15 % is paid on every request. Switch with
`LAYO_SHAPELESS=1` when shape variety is genuinely unbounded and cold-call spikes hurt more than
steady-state throughput. The real fix is upstream: a shape-inferable `fast.sdpa` would make
shapeless free.

## Ecosystem survey: is there a maintained package that does this?

`@frost-beta/mlx` was last published **2025-04-19** and vendors MLX 0.25, so it is stale. Surveying
npm (publish dates are the evidence) turns up a maintained family built on Apple's **mlx-c**:

| package | published | license | what it is |
|---|---|---|---|
| `@johnhenry/laya` 0.3.2 | 2026-10-02 | Apache-2.0 | a TS port of Laya itself (MLX / WebGPU / CPU backends, shortlist, prefix cache, quantization) |
| `@johnhenry/backend-mlx` 0.5.0 (+ `-darwin-arm64`) | 2026-10-02 | — | the MLX backend; the platform package ships `libmlx` + `libmlxc` + `libjaccl` + `mlx.metallib` + `SHA256SUMS` + `VERSION`, no install scripts |
| `@nielspeter/mlx-ts` 0.5.0 (+ `-darwin-arm64`) | 2026-09-13 | MIT | mlx.core-style SDK over mlx-c/FFI (`loadSafetensors`, `asyncEval`, `setWiredLimit`, `metalKernel`, `nn.Linear`, …) |
| `@gobing-ai/ts-laya-mlx` 0.5.16 | 2026-10-05 | Apache-2.0 | the sidecar option, packaged: drives Python `laya-mlx` as a long-lived JSON-lines worker |
| `@mlx-node/core` 0.0.16 | 2026-10-05 | MIT | active, self-contained prebuilt, but model-shaped: no fused rope/SDPA, no generic safetensors, no compile |
| `mlx-bun` 0.5.0 | 2026-09-16 | — | Bun-only MLX inference over mlx-c |

### `@johnhenry/laya` 0.3.2, evaluated against this repo's harness

`src/eval/johnhenry-laya.ts` runs our Python-generated reference fixtures and benchmark inputs through
their agent (`backend: "mlx"`). Note the two MLX builds **cannot share a process**: their `libmlxc`
resolves `@rpath/libmlx.dylib` to whichever build is already loaded, so importing our binding and
theirs together dies with a symbol mismatch (`mlx::core::gather_qmm`). The diagnostics therefore
split across processes (`dump-prepared.ts` + `johnhenry-tokens.ts`).

| | ours (patched node-mlx, MLX 0.32.3 wheel build) | `@johnhenry/laya` (MLX 0.32.2, own build) |
|---|---:|---:|
| MLX metallib | 190 MB | 135.8 MB (Homebrew-class) |
| 1 question | 10.2 ms | 19.25 ms |
| 3 questions | 15.4 ms | 49.4 ms |
| 16 rows | 59.1 ms (271 q/s) | 183.5 ms (87 q/s) |
| `compile` gain | −12 % | −3 % |
| parity vs Python | fp32 bit-exact, fp16 Δ4e-4 | matches except one case (below) |

Three diagnoses, in order of size:

1. **FFI per-op cost.** Their backend calls mlx-c through koffi, "one FFI call per op", with every op
   immediate — so `mlx_compile` cannot remove the call overhead (the JS still issues every op, which is
   why their compile buys 3 % against our 12 %). Bun measures identically (19.2 / 49.4 / 183.8 ms), so
   this is the FFI + backend layer, not the JS engine.
2. **The MLX build**, again: their metallib is Homebrew-class (135.8 MB) versus the wheel's 190 MB,
   which we measured at ~35 % on kernel-bound work.
3. **A prompt-construction bug.** Every case matches token-for-token except `noul` questions with a
   custom `labels` option, where their prompt uses the raw `false`/`true` keys instead of the labels:

   ```
   ours  : 642="Ġno" 27=":" 14905="Ġlegitimate" 50284="[MASK]" 4754="Ġyes" 27=":" …
   theirs: 3221="Ġfalse" 27=":" 14905="Ġlegitimate" 50284="[MASK]" 2032="Ġtrue" 27=":" …
   ```

   That single difference moves the answer (noul 0.766 → 0.466). It is the same class of bug as the
   `"yes: null"` rendering trap in this port.

**Verdict.** Not as a runtime: 3x slower on the same machine for the same inputs, plus a parity bug —
an FFI binding that issues ~700 calls per forward cannot match a native NAPI binding here. But their
*packaging* is ahead of ours and worth copying (platform package, no install scripts, checksums, a
`VERSION` file), their feature surface is wider (WebGPU and CPU backends, shortlist, quantization,
`embed`), and the right move is to report the `labels` bug upstream and contribute the measured
findings (MLX build quality, FFI call cost, request-level batching, the scalar-dtype trap) —
`@johnhenry/laya` is Apache-2.0, as is Laya, so the code is also a legitimate source for features we
have not ported.

## Correctness

`tools/laya_ref.py` runs the real `laya_mlx` package (fp16 and fp32) over three cases — a
three-question set, a 12-option question that exercises the clamped `choice:11+` bucket, and a
case with structured state, dict-valued criteria and custom noul labels — and both TypeScript
routes are compared field by field (61 numeric fields per dtype):

| reference | route | decisions | max Δ | exact fields |
|---|---|---|---:|---:|
| fp16 | ONNX (TS) | all match | 1.3e-3 | 39/61 |
| fp16 | **MLX (TS)** | all match | **1.2e-3** | 33/61 |
| fp32 | ONNX (TS) | all match | 1.8e-3 | 37/61 |
| fp32 | **MLX (TS)** | all match | **0** | **61/61** |

The fp16 residual is fp16 accumulation, not logic; in fp32 the port is bit-identical on every
reported number. Node and Bun produce identical results (same deltas to the last digit).

## Files

| file | lines | what |
|---|---:|---|
| `src/mlx/model.ts` | ~300 | `model.py` — encoder, RoPE, sliding-window masks, decision head, scorer, action head |
| `src/mlx/agent.ts` | ~250 | `agent.py`'s tensor path — load, cast, collate, batch, tidy, read out |
| `src/common.ts` | 283 | `common.py` + answer shaping — **shared unchanged with the ONNX route** |
| `src/tokenizer.ts` | 44 | `tokenizer.py` — **shared unchanged** |
| `tools/build-node-mlx-mlx32.sh`, `tools/node-mlx-mlx32.patch` | 552-line patch | rebuild the binding for MLX 0.32 |
| `tools/laya_ref.py`, `src/mlx/parity.ts` | | correctness harness against Python |
| `tools/profile*.py`, `src/mlx/profile*.ts`, `kernelbench`, `layerbench` | | the measurements above |
| `src/mlx/tune.ts`, `tools/tune.sh`, `src/mlx/varibench.ts`, `src/mlx/quantbench.ts` | | the configuration sweep, variable-length traffic and quantization tests |
| `src/mlx/shaplessprobe.ts`, `src/mlx/chainprobe.ts`, `src/mlx/bisect.ts`, `src/mlx/attnbench.ts` | | the shape-inference boundary, the shapeless bisect and the attention cost comparison |
| `src/mlx/batcher.ts`, `src/mlx/server.ts`, `src/mlx/servebench.ts` | ~450 | request batching, the HTTP service and its load test |

```bash
bun install
tools/build-node-mlx-mlx32.sh          # rebuild the binding for MLX 0.32.3
node src/mlx/parity.ts fixtures/ref/fp16.json fixtures/ref/fp32.json
tools/tune.sh 1 25                     # configuration sweep (compile, bucketing, limits)
node src/mlx/bench.ts 10               # or: bun src/mlx/bench.ts 10   → 10.2 ms / 59 ms
```

## Not ported yet

`router.py` (436 lines), `lang.py` (1,417), `email.py` (315), `shortlist.py` (270),
`presets.py` (193), the prefix cache, the CLI and the Snake demo. The prefix cache needs the
encoder to be re-entered with a longer sequence; `mx.asyncEval` is available in this binding for
the Snake-style loop.

## Corrections to earlier claims in this document's history

Three of my earlier statements were wrong and are retracted here:

1. *"fp16 vs fp32 drift up to 3.3e-2"* — my harness compared MLX fp16 on one prompt against ONNX
   fp32 on a different prompt. Matched inputs differed by ~2e-4. `src/crosscheck.ts` now reads both
   sides from the generated reference files.
2. *"the 2x is the vendored MLX version, not JavaScript; rebuilding recovers Python's speed"* —
   rebuilding alone recovered 10 % of the 2x. The rest was the choice of prebuilt MLX **build**
   (~35 %) and the JavaScript scalar-dtype trap (~20 %). The final answer is still "yes, port it
   with MLX", but the reason is the *set* of fixes, not the MLX version alone.
3. *"`round4` is banker's rounding"* — it was not. The port's `round4` snapped any value within
   `1e-9` of a 4-decimal half to an exact tie and then rounded to even, which is not what Python
   does: Python rounds the **exact double**, correctly rounded to 4 decimals. The two disagree
   whenever the double sits just off the half — `round4(0.00005)` returned `0` where Python returns
   `0.0001` — and the port diverged from CPython on **120 of 8 017** sampled values. It survived
   parity only because no measured answer landed within `1e-9` of a half, but `confidence` is
   precisely the kind of small number that can.

   Fixed by rounding via `Number(x.toFixed(4))`, which the spec defines as "as close to zero as
   possible" on the exact value: **8 016 of 8 017** now match CPython, the outlier being signed zero,
   which the fix preserves and the old version lost. Confirmed as identity on all 90 numbers in the
   committed fp32/fp16 references, so parity cannot regress.

A related limitation that is *not* a bug, recorded so it is not rediscovered as one: Python's `json`
has separate `int` and `float` types and prints them differently (`json.dumps(2)` is `"2"`,
`json.dumps(2.0)` is `"2.0"`). JavaScript has one number type and `JSON.parse` discards whether the
source literal had a decimal point, so the port renders integral values as Python renders an `int`
and fractional values as Python renders a `float`. Everything matches except an integral float
literal (`{"amount": 2.0}` renders `2`), which is pinning-tested and documented in `ppjson.ts`.