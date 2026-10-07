# Changelog

All notable changes to `@fllstck/mlayax` and `@fllstck/mlayax-darwin-arm64`.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the versions are the
versions of both packages, published together. Performance is part of the contract here, so a number
that changes is a change worth an entry — §6 of [`TASKS.md`](TASKS.md) is the gate that enforces it.

## [0.1.0] — 2026-10-07

Published manually to npm, platform package first, so the tarballs carry **no npm provenance
attestation** — that needs CI + OIDC. The repository tag `v0.1.0` matches `package.json` in both
packages.

Verified from a clean machine's point of view after publishing, in empty directories with an empty npm
cache: the published bytes match what was built (sha1 against the registry's own metadata, sha256
against `SHA256SUMS`), both packages arrive, and `predict()` reproduces the quickstart's documented
numbers under Node 22.23.3, Node 24.15.0 and Bun 1.3.13.

**First release.** An independent TypeScript port of Laya's typed-decision runtime, running on Apple's
MLX through a patched node-mlx binding. No Python, no subprocess, no install scripts, no bundled
weights.

### Added

- **`@fllstck/mlayax`** — the façade: prompt construction, calibration, answer shaping, the Hugging
  Face fetcher, and the MLX runtime layer (node-mlx's MIT JavaScript, vendored unmodified).
  `load()` / `loadAsync()` / `predict()`, with `prepare()` and `forwardItems()` public as the seam for
  caller-side batching. 252.9 KiB unpacked, ESM-only.
- **`@fllstck/mlayax-darwin-arm64`** — the native payload: `node_mlx.node` (patched for MLX 0.32.3),
  `libmlx.dylib`, `libjaccl.dylib`, `mlx.metallib` (wheel-class, 190 MB), `SHA256SUMS`, `VERSION`.
  64.9 MiB as a tarball. No install scripts; `os: darwin`, `cpu: arm64`.
- **A load-time mixing guard.** macOS resolves dynamic libraries by install name, and every MLX
  distribution published for Node names itself `@rpath/libmlx.dylib` — so a second MLX in the process
  silently steals our addon's symbols. The guard reports the two file paths instead of letting dyld
  produce a mangled-symbol crash, and distinguishes a fatal collision from a harmless second copy.
- **Parity and hazard tests** against the Python reference: fp32 bit-exact (61/61 fields per case),
  fp16 within 4e-4, a synthetic checkpoint committed as a fixture so the default suite needs no
  download, plus tests for nine measured hazards (cached masks, `tidy` around `asyncEval`, marker
  gathering, fp16 scalar upcasting, and the rest).
- **A benchmark gate** (`npm run bench:check`) with a committed baseline: warn above 1.3x, fail above
  2x, and an explicit "inconclusive" on a machine that is not the baseline's CPU.
- **CI** — eight jobs, including a mode-1 native build whose payload is consumed by the macOS test
  jobs and by a clean-room install of both tarballs.

### Performance

Reference: one 39-token state, fp16, Apple M5, macOS 26.6.2, Node 24.15.0 / Bun 1.3.13.

| metric | value |
|---|---|
| one short `choice` question | 10.3 ms (warm p50) |
| three questions | 15.1 ms |
| 16 rows in one forward | 53.7 ms (298 q/s) |
| throughput plateau at ≥ 8 rows | ~312 q/s |
| first call at a new sequence length | ~25–35 ms (one compile trace) |
| RSS, one model resident | 987 MiB |
| fp32 parity vs Python `laya_mlx` | bit-exact, Δ 0 on 61/61 fields |
| fp16 parity | Δ ≤ 4e-4, 48/61 fields exact |

For reference, the Python implementation this ports measures 10.2 ms for one short question and
58.4 ms for 16 rows on the same machine, and the fastest maintained TypeScript alternative that does
not use a compiled binding measures 183.5 ms for the same 16 rows. Every figure here is produced by
`bench/` and gated at 1.3x against `bench/baseline.json` (which holds §2's earlier single-call
measurement, 10.2 / 15.4 / 59.1 ms — hence this build's 0.91x on the 16-row case).

### Corrections carried into 0.1.0

Recorded so they are not re-derived. The long form of each is in
[`docs/PORTING.md`](docs/PORTING.md)'s corrections section and [`TASKS.md`](TASKS.md) §10.

- **"fp16 drifts up to 3.3e-2 from fp32"** — wrong. The harness compared MLX fp16 on one prompt
  against ONNX fp32 on a *different* prompt; matched inputs differ by ~2e-4. (The fp16 figures in this
  changelog are from matched inputs, and are checked in as fixtures.)
- **"the 2x gap is the vendored MLX version, not JavaScript"** — wrong, and wrong in an expensive way.
  Rebuilding the binding for MLX 0.32.3 recovered about 10 % of the 2x. The rest was the choice of
  prebuilt MLX **build** (the wheel's 190 MB metallib versus Homebrew's 137 MB, ~35 % on kernel-bound
  work) and a JavaScript scalar-dtype trap: `mx.array(1)` is float32, so mixed operands silently
  upcast fp16 activations to fp32 — about 20 %, with no error, and the only symptom "the JS port is
  slower for no reason". The answer is still "port it with MLX"; the reason is the set of fixes.
- **`round4` was not banker's rounding** — it was a real parity bug. It snapped values within 1e-9 of a
  4-decimal half to an exact tie and rounded to even, disagreeing with CPython on 120 of 8 017 sampled
  values (`round4(0.00005)` returned `0` where Python returns `0.0001`). Now rounds the exact double
  the way the spec defines it; 8 016 of 8 017 match, the outlier being signed zero, which the fix
  preserves. See §10.7.
- **Source maps are no longer published.** They were 113.4 KiB of the 360 KiB budget and, in the
  installed package, they resolve to nothing: their `sources` point at `src/**`, which the tarball does
  not carry, and they ship no `sourcesContent`. The payload dropped from 350.9 KiB to 237.5 KiB and the
  budget returned to §0's original 300 KiB; writing these docs and the JSDoc fixes in the same phase
  brought it back up to 252.9 KiB, still inside the budget with the maps still gone. They are still emitted for local use; the size gate refuses
  to let them back into the tarball without the sources. See §10.4 and §10.12.
- **Two gates asserted things that could not be true.** `build.sh` compared a from-source payload
  against the wheel's provenance hashes, so every mode-2 build exited non-zero after building
  correctly; and the Bun half of the mixing tests required an image list Bun does not expose (its
  `sharedObjects` is always empty, so the obvious capability check lies). Both are fixed, and the Bun
  degradation is now asserted rather than assumed. See §10.11.

### Known limitations

- Apple Silicon only; macOS ≥ 26.2 (the pinned `libmlx.dylib` declares `minos 26.2`).
- `confidence` from the checkpoint's `choice:11+` bucket is clamped — that bucket's temperature is
  outside the `[0.5, 5]` range the calibration clamps to, and the library raises a `RuntimeWarning` rather
  than returning a number that looks calibrated and is not.
- One MLX runtime per process. Loading another one first is fatal, by dyld's rules rather than ours.
- `npm run test:coverage` does not pass: `src/mlx` branch coverage is 73.8 % against an 85 % threshold,
  and the missing branches are the forward pass's error and fallback paths. It is not wired into CI;
  see §10.5.

[0.1.0]: https://github.com/fllstck/mlayax/releases/tag/v0.1.0