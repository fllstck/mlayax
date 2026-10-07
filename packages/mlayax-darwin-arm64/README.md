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
| `VERSION` | node-mlx commit, MLX tag, build mode, metallib/addon hashes, `min_macos`, build timestamp |

## Requirements

- `os: darwin`, `cpu: arm64` — npm will skip this package elsewhere, which is why it is an
  *optional* dependency of the façade
- macOS ≥ 26.2. The three MLX files are copied verbatim from a pinned prebuilt MLX, and that build
  declares `minos 26.2`, so it is `libmlx.dylib` that refuses to load on anything older. `VERSION`
  records the floor as `min_macos`, and the build refuses to publish a payload whose declared floor
  sits below what the shipped dylib needs.
- Node ≥ 22, or Bun ≥ 1.2

## Verifying the payload

```bash
cd node_modules/@fllstck/mlayax-darwin-arm64/lib && shasum -a 256 -c ../SHA256SUMS
cat ../VERSION                          # node-mlx commit, MLX tag, build class, floor, addon hash
otool -l libmlx.dylib | grep -A2 LC_BUILD_VERSION   # the macOS floor, from the binary itself
```

`VERSION` records both the pre-rpath and shipped addon hashes, so a rebuild on another machine can be
compared against the published one. `tools/native/build.sh --check` in the repository runs exactly
these assertions, plus the fused-symbol and rpath checks.

## Warning: do not mix MLX builds in one process

Two different `libmlx` builds cannot coexist in a single process. macOS resolves a dynamic library by
install name, and virtually every MLX distribution published for Node names itself
`@rpath/libmlx.dylib`, so a foreign build already resident silently satisfies our addon's request for
it. The process then dies on a symbol mismatch — measured, with
`@johnhenry/backend-mlx-darwin-arm64` resident first:

```text
dlopen(…/mlayax-darwin-arm64/lib/node_mlx.node, 0x0001): Symbol not found:
  __ZN3mlx4core10gather_qmmERKNS0_5arrayES3_S3_RKNSt3__18optionalIS1_EES6_S6_bNS5_IiEES9_…
```

`@fllstck/mlayax` detects this before loading and names both paths instead; see the façade README's
troubleshooting section. The failure is not recoverable *in* that process — unloading the addon does
not unload the dylib — so the fix is to remove the competing package or move it to a worker process.
A resident MLX with an absolute install name (Homebrew's, for example) cannot take our slot, and is
reported as a note rather than an error.

## Licensing

MIT for this package's own files. It redistributes MLX (MIT, Apple) and node-mlx (MIT,
frost-beta); the full texts are in `licenses/`. See `NOTICE`. This is an independent port, not
an official Laya release.

The model weights are **not** here: they are Apache-2.0 (Convai Innovations) and are downloaded
to the Hugging Face cache on first use.