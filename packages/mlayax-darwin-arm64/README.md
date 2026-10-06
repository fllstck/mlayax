# @fllstck/mlayax-darwin-arm64

Native MLX payload for [`@fllstck/mlayax`](https://www.npmjs.com/package/@fllstck/mlayax).
Apple Silicon only.

This package ships **no JavaScript API and no install scripts**. It exists so the façade can
resolve a known-good native artifact without building anything on the user's machine:

| file | what |
|---|---|
| `lib/node_mlx.node` | the node-mlx addon, patched for the pinned MLX |
| `lib/libmlx.dylib` | MLX core |
| `lib/libjaccl.dylib` | MLX collective-comms library (a link-time dependency) |
| `lib/mlx.metallib` | the compiled Metal kernels — the build class matters for throughput |
| `SHA256SUMS` | checksums for every file above |
| `VERSION` | node-mlx commit, MLX tag, build mode, metallib/addon hashes, build timestamp |

## Requirements

- `os: darwin`, `cpu: arm64` — npm will skip this package elsewhere, which is why it is an
  *optional* dependency of the façade
- macOS ≥ 14
- Node ≥ 22, or Bun ≥ 1.2

## Warning: do not mix MLX builds in one process

Two different `libmlx` builds cannot coexist in a single process. A foreign `libmlx` already
resident will silently satisfy `@rpath/libmlx.dylib`, and the process then dies with a symbol
mismatch. `@fllstck/mlayax` checks for this at load time and fails with a clear message; until
then, keep other MLX packages in separate processes or `dispose()` them first.

## Licensing

MIT for this package's own files. It redistributes MLX (MIT, Apple) and node-mlx (MIT,
frost-beta); the full texts are in `licenses/`. See `NOTICE`. This is an independent port, not
an official Laya release.

The model weights are **not** here: they are Apache-2.0 (Convai Innovations) and are downloaded
to the Hugging Face cache on first use.