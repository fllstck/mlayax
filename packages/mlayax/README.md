# @fllstck/mlayax

A **Laya typed-decision runtime for Node and Bun on Apple Silicon**, in TypeScript on MLX.
No Python, no `pyproject.toml`, no subprocess.

This package is the published façade. It contains TypeScript/JavaScript only — no native code,
no weights, no install scripts. The native MLX payload is delivered by the optional dependency
[`@fllstck/mlayax-darwin-arm64`](https://www.npmjs.com/package/@fllstck/mlayax-darwin-arm64),
which npm installs automatically on Apple Silicon.

> **Status: 0.1.0 is in development.** The public surface (`load`, `predict`) is implemented and
> benchmarked; the docs, CI and release gates are still landing. This is a library, not a service —
> there is no HTTP server and no subpath export that implies one. See `TASKS.md` in the repository.

## Install

```bash
npm i @fllstck/mlayax
# or
bun add @fllstck/mlayax
```

## Requirements

- Apple Silicon (`darwin` / `arm64`) only. Intel Macs, Linux and Windows are out of scope for 0.1.0.
- **macOS ≥ 26.2.** Not our constraint: the MLX build we link declares `minos 26.2`, so on an older
  system it is `libmlx.dylib` that refuses to load. Check with
  `otool -l node_modules/@fllstck/mlayax-darwin-arm64/lib/libmlx.dylib | grep -A2 LC_BUILD_VERSION`.
- Node ≥ 22, or Bun ≥ 1.2.

## Troubleshooting

### "Symbol not found", or "another libmlx.dylib is already resident"

You have two MLX runtimes in one process. macOS resolves a dynamic library by **install name**, and
virtually every MLX distribution published for Node names itself `@rpath/libmlx.dylib` — so if
another one is already loaded, dyld hands it to our addon instead of ours, and the addon dies on a
symbol it was not compiled against:

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
Remove it, or load it in a separate worker process.

Under Node the check runs before the load, using the dynamic loader's own image list.
**Bun does not expose that list** — `process.report.getReport().sharedObjects` is always empty — so
there the collision can only be diagnosed from the error it produces, and the same explanation is
attached to the failure instead. Both paths produce the message above.

If you have verified that the two builds are interchangeable, `MLAYAX_ALLOW_MIXED_MLX=1` proceeds
anyway. It is rarely right: even when the symbols match, MLX looks for `mlx.metallib` next to
whichever library won, and fails with `Failed to load the default metallib` — or silently runs on a
slower kernel set, since build class predicts throughput (see `TASKS.md` §2).

The guard is deliberately precise about what counts as a conflict. A resident MLX that names itself
with an **absolute** install name (Homebrew's bottle, for example) cannot satisfy our addon's request
for `@rpath/libmlx.dylib`; it loads as a second copy, MLX still resolves correctly, and you get a
`RuntimeWarning` noting two resident runtimes rather than a failure. Only a library that can actually
take our slot is fatal, and an image whose install name cannot be read is assumed to be able to.

## Roadmap

WebGPU/CPU backends, ONNX backend, browser support, an embedding shortlist, quantized
checkpoints, training and the Snake demo are all explicitly out of scope for 0.1.0.

## Licensing

MIT for this package. It also redistributes node-mlx's MIT JavaScript layer (unmodified) under
`vendor/`. MLX is MIT (Apple); the Laya weights are Apache-2.0 (Convai Innovations) and are
downloaded, not bundled. See `NOTICE`. This is an independent port, not an official Laya release.