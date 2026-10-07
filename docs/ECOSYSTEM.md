# The ecosystem, and why this is a fork with its own artifact

`@fllstck/mlayax` depends on a **patched** `@frost-beta/mlx` rather than on any of the maintained
alternatives, and publishes its own native payload instead of reusing someone else's. That is a
deliberate decision with a measured argument, and this document is the argument — including what would
change it.

Short version: nothing maintained does a compiled in-process forward pass, and the one package that
implements Laya itself in TypeScript is an FFI binding, which is **3.1x slower on identical inputs**
for a reason that cannot be optimised away from JavaScript.

## What is on npm

Publish dates and licences are the evidence, taken from the registry on 2026-10-07.

| package | published | license | what it is |
|---|---|---|---|
| `@frost-beta/mlx` 0.4.0 | 2025-04-19 | MIT | the NAPI binding this repository forks; vendors MLX **0.25**, unmaintained |
| `@johnhenry/laya` 0.3.2 | 2026-10-02 | Apache-2.0 | a TypeScript port of Laya itself (MLX / WebGPU / CPU backends, shortlist, prefix cache, quantization) |
| `@johnhenry/backend-mlx` 0.5.0 (+ `-darwin-arm64`) | 2026-10-02 | — | the MLX backend over mlx-c; the platform package ships `libmlx` + `libmlxc` + `libjaccl` + `mlx.metallib` + `SHA256SUMS` + `VERSION`, no install scripts |
| `@nielspeter/mlx-ts` 0.5.0 (+ `-darwin-arm64`) | 2026-09-13 | MIT | an mlx.core-shaped SDK over mlx-c/FFI (`loadSafetensors`, `asyncEval`, `metalKernel`, `nn.Linear`, …) |
| `@gobing-ai/ts-laya-mlx` 0.5.16 | 2026-10-05 | Apache-2.0 | drives Python `laya-mlx` as a long-lived JSON-lines worker — the sidecar option, packaged |
| `@mlx-node/core` 0.0.16 | 2026-10-05 | MIT | active, self-contained prebuilt, but model-shaped: no fused `rope`/SDPA/`layerNorm`, no generic safetensors reader, no compile |
| `mlx-bun` 0.5.0 | 2026-09-16 | — | Bun-only MLX inference over mlx-c |

Two structural facts decide most of this. **Every maintained alternative is FFI over mlx-c**, and
**every one of them ships a Homebrew-class MLX build** (135–138 MB `mlx.metallib`) rather than the
Python wheel's 190 MB one.

## The comparison that matters

Same machine, same inputs, same reference fixtures: `@johnhenry/laya` (the only maintained TypeScript
Laya) against this port, both driven through this repository's harness
(`src/eval/johnhenry-laya.ts` in the spike, which also generated the token-level diff below).

| | mlayax (patched node-mlx, MLX 0.32.3 wheel build) | `@johnhenry/laya` (MLX 0.32.2, own build) |
|---|---:|---:|
| MLX `metallib` | 190 MB (wheel class) | 135.8 MB (Homebrew class) |
| 1 question | **10.2 ms** | 19.25 ms |
| 3 questions | **15.4 ms** | 49.4 ms |
| 16 rows | **59.1 ms** (271 q/s) | 183.5 ms (87 q/s) |
| gain from `mx.compile` | −12 % | −3 % |
| parity vs Python `laya_mlx` | fp32 bit-exact, fp16 Δ ≤ 4e-4 | matches except one case (below) |

Three causes, in order of size:

1. **FFI per-op cost.** Their backend calls mlx-c through koffi, one FFI call per op, and every op is
   issued immediately — so `mlx_compile` cannot remove the call overhead, because JavaScript still
   issues every op. That is why their `compile` buys 3 % where ours buys 12 %. The forward pass is
   ~700 ops; at ~15 µs of extra per-op cost on the small memory-bound ones, the ~2 ms eager gap and
   the 3x batched gap follow. Bun measures identically (19.2 / 49.4 / 183.8 ms), so this is the binding
   and the backend layer, not the JavaScript engine.
2. **The MLX build.** Their `metallib` is Homebrew-class; on kernel-bound work we measured that class
   at ~35 % slower than the wheel's, for the same MLX version. This is the one cost that is nobody's
   bug — it is a choice, and it is fixable from either side.
3. **A prompt-construction bug.** Every case matches token-for-token except `noul` questions with a
   custom `labels` option, where their prompt renders the raw `false`/`true` keys instead of the
   labels:

   ```text
   ours  : 642="Ġno" 27=":" 14905="Ġlegitimate" 50284="[MASK]" 4754="Ġyes" 27=":" …
   theirs: 3221="Ġfalse" 27=":" 14905="Ġlegitimate" 50284="[MASK]" 2032="Ġtrue" 27=":" …
   ```

   That single difference moves the answer (`noul` 0.766 → 0.466). It is the same class of trap as the
   `"yes: null"` rendering bug this port hit, which is why the prompt is treated as a contract here and
   pinned by fixtures.

**The two MLX builds cannot share a process.** Their `libmlxc` resolves `@rpath/libmlx.dylib` to
whichever build is already resident, so importing our binding and theirs together dies with a symbol
mismatch (`mlx::core::gather_qmm`). The diagnostics therefore run in separate processes. That is not a
limitation of either package so much as of macOS's install-name resolution, and it is what the
[mixing guard](../packages/mlayax/README.md#symbol-not-found-or-another-libmlx-dylib-is-already-resident)
exists to explain.

## The decision this follows

**Fork the artifact, not the package name.** This repository does not publish a general-purpose MLX
binding under its own name. It ships a patched build inside its own platform package
(`@fllstck/mlayax-darwin-arm64`), which is what a product with one model and one platform can do
honestly and what a general-purpose binding cannot.

Why not adopt one of the maintained packages instead:

| candidate | why not |
|---|---|
| `@johnhenry/laya` | the 3x above, plus the `labels` bug: a drop-in replacement would trade a working runtime for someone else's correctness |
| `@nielspeter/mlx-ts`, `mlx-bun` | same FFI ceiling, and neither is a Laya implementation — adopting them means re-porting the model, then still being FFI-bound |
| `@gobing-ai/ts-laya-mlx` | a wrapper around Python `laya-mlx`. The whole point here is no Python and no subprocess |
| `@mlx-node/core` | active and self-contained, but model-shaped: no fused `rope`/SDPA/`layerNorm`, no generic safetensors reader, no compile. Adopting it means hand-writing the performance-critical parts, which is the work this repository has already done |
| `@frost-beta/mlx` unpatched | MLX 0.25 measured **1.9x slower** than 0.32 on this workload, and last published 2025-04-19 |

The patch itself is small and offered upstream: `tools/native/node-mlx-mlx32.patch`, **10 files, 242
added lines**, mostly API drift between MLX 0.25 and 0.32 (`Shape` became `SmallVector<int>`,
`fast::scaled_dot_product_attention` gained `sinks` and `force_fused`, `eval_impl` became `async_eval`,
overloaded-op address-taking became ambiguous). If upstream revives, dropping the fork is a dependency
swap rather than a rewrite — and `CONTRIBUTING.md`'s MLX bump playbook is written with that in mind.

## What would change the decision

Stated in advance, so it is a rule rather than a post-hoc rationalisation. Any one of these, measured,
would reopen it:

1. **A compiled binding appears.** A maintained package whose forward pass issues one call per
   graph instead of one per op — a NAPI/`mlx-c` binding with a real `compile`, not FFI. That removes
   cause 1, which is 2x of the 3x.
2. **Someone ships the wheel-class MLX.** The 35 % is a build choice, not a version; a package that
   links the 190 MB `metallib` removes cause 2 today.
3. **`@johnhenry/laya` fixes the `labels` rendering** and closes the FFI gap. Then it is a genuine
   dependency swap, and this repository becomes a patch set and a benchmark.
4. **MLX grows a shape-inferable `fast.sdpa`.** That is the one thing standing between us and
   shapeless compilation, which would remove the cold-call trace (~25 ms at a new sequence length) at
   no cost. It is an upstream fix, available to every package here equally.

## What is worth copying from them

Their **packaging** is ahead of ours, and this repository adopted it: a platform package with
`os`/`cpu` constraints, no install scripts, `SHA256SUMS`, a `VERSION` file, and no weights in the
tarball. Their **feature surface** is wider too — WebGPU and CPU backends, an embedding shortlist, a
prefix cache, quantized checkpoints — and `@johnhenry/laya` is Apache-2.0, as is Laya itself, so it is
a legitimate source for features this port has not implemented. Those are on the roadmap in
[`TASKS.md`](../TASKS.md) §0, named rather than promised.

## To send upstream

Two pieces of goodwill, both named in the release checklist:

- The MLX 0.32 bump patch, to `frost-beta/node-mlx`.
- To `@johnhenry/laya`: the `labels` token diff above, plus the measured findings it sits in — MLX build
  class (35 %), FFI per-op cost (3x on batched rows), request-level batching (2.5x on short states), and
  the JavaScript scalar-dtype trap (`mx.array(1)` is float32, so a literal silently upcasts fp16
  activations, ~20 %, no error). Apache-2.0 means the surrounding code is also a fair source for
  anything this port has not built.