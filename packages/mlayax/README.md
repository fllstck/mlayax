# @fllstck/mlayax

A **Laya typed-decision runtime for Node and Bun on Apple Silicon**, in TypeScript on MLX.
No Python, no `pyproject.toml`, no subprocess.

This package is the published façade. It contains TypeScript/JavaScript only — no native code,
no weights, no install scripts. The native MLX payload is delivered by the optional dependency
[`@fllstck/mlayax-darwin-arm64`](https://www.npmjs.com/package/@fllstck/mlayax-darwin-arm64),
which npm installs automatically on Apple Silicon.

> **Status: 0.1.0 is in development.** The public surface (`load`, `predict`, the HTTP service)
> lands in phases 2–4; see `TASKS.md` in the repository.

## Install

```bash
npm i @fllstck/mlayax
# or
bun add @fllstck/mlayax
```

## Requirements

- Apple Silicon (`darwin` / `arm64`) only
- macOS ≥ 14
- Node ≥ 22, or Bun ≥ 1.2

## Roadmap

WebGPU/CPU backends, ONNX backend, browser support, an embedding shortlist, quantized
checkpoints, training and the Snake demo are all explicitly out of scope for 0.1.0.

## Licensing

MIT for this package. It also redistributes node-mlx's MIT JavaScript layer (unmodified) under
`vendor/`. MLX is MIT (Apple); the Laya weights are Apache-2.0 (Convai Innovations) and are
downloaded, not bundled. See `NOTICE`. This is an independent port, not an official Laya release.