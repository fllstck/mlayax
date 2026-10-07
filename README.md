# @fllstck/mlayax

A **Laya typed-decision runtime for Node and Bun on Apple Silicon**, in TypeScript on MLX.
No Python, no `pyproject.toml`, no subprocess.

> **Status: 0.1.0 is in development.** This repository is being built out phase by phase —
> see [`TASKS.md`](https://github.com/fllstck/mlayax/blob/main/TASKS.md) for the plan and the
> measured performance targets. The package is not yet installable from the registry.

## What it will be

Two packages, published together:

| package | contents |
|---|---|
| `@fllstck/mlayax` | TypeScript only: prompt construction, calibration, answer shaping, MLX runtime layer (vendored JS), Hugging Face fetcher, mixing guard |
| `@fllstck/mlayax-darwin-arm64` | native payload: `node_mlx.node`, `libmlx.dylib`, `libjaccl.dylib`, `mlx.metallib`, `SHA256SUMS`, `VERSION` |

The model weights (Apache-2.0, Convai Innovations) are **never bundled** — they are downloaded to
the Hugging Face cache on first use.

## Requirements

- Apple Silicon (`darwin` / `arm64`); Intel Macs and Windows/Linux are out of scope for 0.1.0
- macOS ≥ 26.2 — set by the pinned MLX build, which declares `minos 26.2` (see `TASKS.md` §10.2)
- Node ≥ 22, or Bun ≥ 1.2
- The MLX build is pinned — see the compatibility table (lands with Phase 8)

## Licensing

Our code is MIT. MLX is MIT (Apple), node-mlx is MIT (frost-beta), the Laya weights are
Apache-2.0 (Convai Innovations). See [`NOTICE`](NOTICE). This is an independent port, not an
official Laya release.

## Development

```bash
npm install
npm run verify          # biome ci && tsc -b && vitest run
npm run verify:release  # + publint, attw, tarball size gate
```