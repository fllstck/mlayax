# Upstream offers

Two pieces of goodwill, both part of the 0.1.0 release checklist (`TASKS.md` §5, Phase 9 —
"post-publish"). Each draft below is written to be pasted as-is, minus this preamble. Neither is a
complaint: one is a patch we already maintain and would rather not, the other is a bug report plus four
measurements that cost us real time to find.

Send them **after** publishing, so the links resolve and so the ask can point at a real release rather
than a plan. Fill in the bracketed placeholders (issue URLs, release URL).

Order of business, briefly:

| to | what | why it matters to them |
|---|---|---|
| [`frost-beta/node-mlx`](https://github.com/frost-beta/node-mlx) | the MLX 0.32.3 bump patch, `tools/native/node-mlx-mlx32.patch` (10 files, +242/−69) | their published 0.4.0 vendors MLX 0.25.0, which is **1.9x slower** on this workload than 0.32.3, and the repo looks dormant — but if it revives, we would rather drop our fork than keep it |
| [`@johnhenry/laya`](https://www.npmjs.com/package/@johnhenry/laya) | a prompt-construction bug for `noul` questions with custom `labels`, plus four measured findings | the bug moves an answer (0.766 → 0.466) with no error, and two of the findings are worth ~3x on their batched path |

---

# Draft 1 — MLX 0.32.3 bump for `frost-beta/node-mlx`

**Title:** `MLX 0.32: patch set for the API drift, plus an option to link a prebuilt MLX`

**Body:**

> Hello — I maintain [`@fllstck/mlayax`](https://github.com/fllstck/mlayax), an independent TypeScript
> port of the Laya decision model that runs on node-mlx. I rebuilt the binding against MLX **v0.32.3**
> at commit `4bf8b1d6de32ae5ec402525bff1f041d73618275` for it, and since that work is already done and
> maintained, I would rather hand it over than keep a fork alive.
>
> The patch is `tools/native/node-mlx-mlx32.patch` in our repository: **10 files, +242/−69**, mostly
> mechanical API drift between MLX 0.25 and 0.32 plus one deliberate addition described below.
>
> ### Why 0.32 matters
>
> On Laya's shapes (39-token state, one forward), same machine, same tensors:
>
> | implementation | MLX | 1 row | 16 rows |
> |---|---|---:|---:|
> | Python `laya_mlx` | 0.32.3 (wheel) | 10.4 ms | 52.3 ms |
> | Python `laya_mlx` | 0.25.2 | 19.3 ms | 166.1 ms |
> | this port, patched binding | 0.32.3 (wheel) | **10.2 ms** | **59.1 ms** |
> | this port, pre-patch binding | 0.25.0 | 20.2 ms | 178.4 ms |
>
> That is ~2x on the forward pass, and it is the difference that made a TypeScript port viable for us at
> all. One further finding that may be interesting on its own: **which prebuilt MLX you link matters as
> much as the version.** Homebrew's 0.32.3 and the Python wheel's 0.32.3 ship different builds —
> `mlx.metallib` of 137 MB and 190 MB respectively — and the wheel's is ~35 % faster on kernel-bound
> work. Linking the wheel's `libmlx.dylib`+`mlx.metallib` took a single-question forward from 18.8 to
> 14.1 ms at the same MLX version.
>
> ### What the patch changes
>
> | file | what |
> |---|---|
> | `CMakeLists.txt` | **the one intentional addition**: an `MLX_PREFIX` cache path. When set, the addon links a prebuilt MLX (`-L$MLX_PREFIX/lib -lmlx`, `mlx.metallib` copied next to the addon) instead of `add_subdirectory(deps/mlx)` |
> | `src/fast.cc` | the three `fast::scaled_dot_product_attention` call sites: two new positional arguments (`{}` for `sinks`, `false` for `force_fused`) |
> | `src/utils.cc`, `src/utils.h` | `PutIntoVector` now returns `std::vector<int>` rather than `mx::Shape`, plus a new `ToShape` converter for callers that need an `mx::Shape` |
> | `src/ops.cc` | `conv_transpose1d/2d/3d` gained `output_padding`; `StreamOrDevice` and `SmallVector` updates |
> | `src/transforms.cc` | `eval_impl` → `async_eval`; `detail::compile_erase` now takes a compile-cache handle |
> | `src/metal.cc` | `metal::device_info` moved to `core::device_info(Device)` |
> | `src/fft.cc` | every FFT op gained an `FFTNorm` parameter, which breaks the function-pointer wrappers; they now use per-op adapters |
> | `src/indexing.cc`, `src/stream.h` | `SmallVector` and `StreamOrDevice` fallout from the two above |
>
> Two of these are worth a warning to anyone else doing the bump:
>
> - **Taking the address of an overloaded op is now ambiguous.** `&mx::floor`, `&mx::isnan` and friends
>   fail to deduce with a deduced-pack diagnostic. Ours are now explicit (`static_cast`), and I wrote a
>   throwaway Python script that finds them by parsing clang's diagnostic — available if useful.
> - The `FFTNorm` parameter change is the reason `fft.cc` is the largest part of the diff: those ops used
>   to be wrapped uniformly from function pointers.
>
> ### Why the `MLX_PREFIX` option exists
>
> It is not laziness: a machine with only CommandLineTools has **no `metal` compiler**, so
> `add_subdirectory(deps/mlx)` cannot build MLX's Metal shaders at all:
>
> ```text
> xcrun --find metal
> xcrun: error: unable to find utility "metal", not a developer tool or not in PATH
> ```
>
> Linking a prebuilt MLX makes the binding buildable on those machines, and it is also what allows an
> application to pin the *fast* build class described above. If you would rather keep upstream's build
> self-contained, the CMakeLists change splits cleanly from the API-drift changes — say so and I will
> send it as two patches.
>
> ### Reproducing
>
> ```bash
> git clone https://github.com/frost-beta/node-mlx && cd node-mlx
> git checkout 4bf8b1d6de32ae5ec402525bff1f041d73618275
> git apply /path/to/node-mlx-mlx32.patch
> python3 -m venv .venv && .venv/bin/pip install mlx==0.32.3   # or: brew install mlx
> MLX_PREFIX="$PWD/.venv/lib/python3.12/site-packages/mlx" npx cmake-js build   # adjust for your Python version
> ```
>
> Our build script wraps all of that if a working example is more useful than instructions:
> [`tools/native/build.sh`](https://github.com/fllstck/mlayax/blob/main/tools/native/build.sh). It also
> verifies the result — the fused symbols the model needs, the `@loader_path` rpath, and the MLX build
> class.
>
> ### On what we publish
>
> To be clear about scope and to avoid any appearance of competing with you: we do **not** publish a
> general-purpose MLX binding. `@fllstck/mlayax-darwin-arm64` is a private payload for our own package —
> one model, one platform, one MLX version — which is a thing a product can do and a binding library
> cannot. If node-mlx revives against a current MLX, dropping this fork is a dependency swap for us, not
> a rewrite. That is the outcome I am hoping for.
>
> Either way, thank you for the binding — the NAPI/compiled approach is what makes the numbers above
> possible; every FFI alternative we measured is 3x slower on batched rows.

---

# Draft 2 — `noul` labels bug and measurements for `@johnhenry/laya`

Send as a GitHub issue on their repository if it has one, and as an npm-facing message otherwise. If a
public issue would be awkward, send the same text privately and skip the "reproduction" heading.

**Title:** `noul` questions ignore `labels` when rendering criteria (answer moves), plus four measurements

**Body:**

> Hello — I maintain another TypeScript port of Laya (compiled node-mlx binding, Apple Silicon) and I
> evaluated `@johnhenry/laya` **0.3.2** against it with your `backend: "mlx"` on the same machine,
> checkpoint and inputs. Your packaging is ahead of ours and I have copied parts of it (see the last
> section). Two things are worth your time: a prompt-construction bug, and four measurements.
>
> ### 1. Bug: a `noul` question with custom `labels` renders the raw keys
>
> Reproduced with a question of the form:
>
> ```json
> {
>   "type": "noul",
>   "instructions": "Is this message a scam?",
>   "criteria": { "true": { "desc": "scam" }, "false": "legitimate" },
>   "labels": { "true": "yes", "false": "no" }
> }
> ```
>
> Comparing the two prompts token-by-token, everything matches except the option labels, where the raw
> `false`/`true` keys are rendered instead of the labels:
>
> ```text
> ours  : 642="Ġno" 27=":" 14905="Ġlegitimate" 50284="[MASK]" 4754="Ġyes" 27=":" …
> theirs: 3221="Ġfalse" 27=":" 14905="Ġlegitimate" 50284="[MASK]" 2032="Ġtrue" 27=":" …
> ```
>
> It is not cosmetic: on the reference fixture it moves the answer from `noul 0.766` to `noul 0.466`
> (same checkpoint, same state, same criteria). Every other case we compared — `choice` with dict
> criteria, `score` with a list, structured state — matches exactly, so this looks isolated to the
> `labels` option.
>
> I hit the same class of trap in my own prompt construction (rendering `"yes: null"` instead of the
> masked/expected text), which is why I now pin prompt rendering with fixtures generated by the Python
> implementation. Happy to share the 15-line comparison script that produced the dump above.
>
> ### 2. Measurement: the MLX build class is worth ~35 %
>
> Your `-darwin-arm64` package ships a 135.8 MB `mlx.metallib`; the Python wheel for the same MLX
> version ships 190 MB. On kernel-bound work (16 rows, one forward) the difference is ~35 %:
>
> | | 1 question | 3 questions | 16 rows |
> |---|---:|---:|---:|
> | this port (patched node-mlx, MLX 0.32.3 wheel build) | **10.2 ms** | **15.4 ms** | **59.1 ms** (271 q/s) |
> | `@johnhenry/laya` 0.3.2, `backend: "mlx"` (MLX 0.32.2, own build) | 19.25 ms | 49.4 ms | 183.5 ms (87 q/s) |
>
> Worth checking whether your build selects a smaller Metal target set or lacks `-O3` on the shaders —
> it is the cheapest of the three causes below to fix.
>
> ### 3. Measurement: the FFI call cost is the largest cost, and it is not optimisable from JavaScript
>
> From the README I understand the backend calls mlx-c through koffi, one call per op, with ops issued
> eagerly. `mlx_compile` therefore cannot remove the per-op overhead: the JavaScript side still issues
> every op. Our numbers show exactly that signature:
>
> | | this port | yours |
> |---|---:|---:|
> | gain from `compile` | −12 % | −3 % |
> | 16 rows | 59.1 ms | 183.5 ms |
>
> A 28-layer forward is ~700 ops, and ~15 µs of extra per-op cost on the small memory-bound ones
> accounts for the ~2 ms eager gap we measure at one row. Bun measures identically (19.2 / 49.4 /
> 183.8 ms), so it is the binding and backend layer rather than the JavaScript engine. Nothing here is a
> bug — it is the ceiling of the approach, and the reason we kept a compiled binding. If a NAPI path is
> ever on the table, our patch set for MLX 0.32 on node-mlx is public and I would rather hand it over
> than maintain it (linked below).
>
> ### 4. Measurement: request-level batching is worth 2.5x on short states
>
> We built a batcher for exactly this workload before deciding not to ship a server, and measured it at
> concurrency 32 over 300 requests, short states (~40 tokens, weight-bound):
>
> | rows per forward | req/s | p50 |
> |---|---:|---:|
> | 1 (no batching) | 50.2 | 637 ms |
> | 4 | 104.8 | 305 ms |
> | **16** | **126.8** | **251 ms** |
>
> 1.6x on long states (~200 tokens, compute-bound: 41.0 → 63.9 req/s). Rows from different requests are
> safe to co-forward because the encoder is bidirectional within a row only and padding is masked —
> we assert batched == solo. On your feature surface (shortlist, prefix cache), a batcher would compose
> with both, and a one-row forward at 39 tokens is entirely weight-bandwidth-bound, so the gain is
> structural rather than workload-specific.
>
> ### 5. Measurement: JavaScript scalars silently upcast fp16
>
> If any of your kernels build constants the way a Python port naturally would — `x * (1 + erf(x / 2))`
> with plain-number literals — be aware that `mx.array(1)` is **float32** in JavaScript, so MLX promotes
> the whole expression and every encoder MLP lifts its activations to fp32. It cost us ~20 % end to end
> with no error of any kind. The fix is to build the literals in the activation's dtype
> (`mx.array(1, x.dtype)`), or to wrap the forward in `mx.tidy` so intermediates do not accumulate; both
> together took one question from 14.1 to 11.6 ms in our port.
>
> ### 6. Interop warning worth a README line
>
> Two MLX builds cannot share a process. Your `libmlxc` resolves `@rpath/libmlx.dylib` to whichever
> build is already resident, and every MLX distribution published for Node names itself
> `@rpath/libmlx.dylib` — so importing your backend and another MLX binding in one process dies with a
> symbol mismatch:
>
> ```text
> dlopen(…): Symbol not found: __ZN3mlx4core10gather_qmmERKNS0_5arrayES3_S3_…
> ```
>
> Documenting that (and that the workaround is a worker process / `dispose()` on the foreign arrays
> before loading the second binding) would save the next person the afternoon it cost us. We added a
> load-time guard that reports both library paths instead of letting dyld produce the symbol dump.
>
> ### What we copied from you
>
> Credit where it is due: the platform-package layout, `os`/`cpu` constraints so npm skips it elsewhere,
> no install scripts, `SHA256SUMS`, and a `VERSION` file recording provenance are all things your
> package does and ours did not. We adopted them. Your WebGPU and CPU backends, shortlist, prefix cache
> and quantization work are also a genuine argument for your package being the right choice for people
> who need those; our fork exists for one workload and one platform.
>
> Links, in case they are useful:
>
> - the MLX 0.32 patch set for node-mlx: https://github.com/fllstck/mlayax/blob/main/tools/native/node-mlx-mlx32.patch
> - the ecosystem comparison with the full numbers: https://github.com/fllstck/mlayax/blob/main/docs/ECOSYSTEM.md
> - [release URL]
>
> Thanks for the package — the two MLX-in-TypeScript efforts measuring the same things is how this list
> exists at all.