# Security

## Scope

This policy covers the two published packages:

- **`@fllstck/mlayax`** — TypeScript/JavaScript. Prompt construction, calibration, answer shaping, the
  Hugging Face fetcher, and node-mlx's vendored JavaScript layer.
- **`@fllstck/mlayax-darwin-arm64`** — native code: `node_mlx.node`, `libmlx.dylib`, `libjaccl.dylib`,
  `mlx.metallib`. A memory-safety bug here is upstream's (MLX is Apple's, node-mlx is frost-beta's), but
  it is in scope for *this* package when the cause is our patch, our build flags, or the way we stage
  and load the payload.

Out of scope: the model weights (Apache-2.0, Convai Innovations, downloaded from the Hugging Face Hub
and not redistributed here), the correctness of the model's answers, and anything that requires an
attacker to already control your `node` command line or your environment (at that point they can set
`NODE_OPTIONS` and load arbitrary code).

## Supported versions

0.1.x. Pre-1.0, only the latest published version is supported; fixes land as a new patch release.

## Reporting

Use GitHub's **private vulnerability reporting** on this repository — the *Report a vulnerability*
button on the Security tab — rather than a public issue. If you cannot use it, open a public issue that
says only that you have a security report and how to reach you; a private thread will follow before any
details are discussed.

Useful reports come with: the version of both packages (`npm ls @fllstck/mlayax`), the Node or Bun
version, `resolveNativeAddonPath()`'s output, and a reproduction. There is no bug bounty, and no SLA —
but this is a small project with a short path from report to release, and acknowledgement is normally
within a few days.

## What this package does with trust

Worth knowing before you deploy it, and worth knowing when you evaluate a report.

- **No install-time code.** Neither package has `preinstall`, `install`, `postinstall` or a
  `binding.gyp`. `npm i @fllstck/mlayax` downloads files and nothing else.
- **No network at import.** Importing the package loads no native code and makes no request. The
  addon is loaded on the first `load()`/`predict()`, and a repository id is only downloaded by
  `loadAsync()`.
- **Weights are data, not code.** The checkpoint is safetensors, a format of length-prefixed tensors
  plus JSON configs — not `pickle`. A hostile checkpoint can exhaust memory or crash the native
  runtime (a denial of service, in scope), but loading one cannot execute Python, as a `torch.load`-style
  pickled checkpoint could.
- **Provenance is checkable.** The payload ships `SHA256SUMS` and a `VERSION` recording the node-mlx
  commit, the MLX tag, the build class and the macOS floor; `tools/native/build.sh --check` re-verifies
  all of it. Where the Hub repository publishes a `SHA256SUMS`, the fetcher verifies each file against
  it and refuses a mismatch, and the resolved commit is recorded on the agent as `revision`.
- **Hub tokens are not leaked, and not stored.** An access token is sent only to the configured
  endpoint and never to the CDN a `resolve` URL redirects to — a test asserts the CDN request carries
  no `Authorization` header. Nothing writes a token to disk.
- **Repository paths are validated.** A file from the Hub's tree listing is written only inside the
  snapshot directory: a path that would escape it (`../…`) is refused rather than followed
  (`snapshotPathFor`, the "zip slip" class). The listing is trusted over TLS, but `endpoint` exists so
  a mirror or proxy can answer instead, and the guard does not depend on who answered.

## Environment variables that change behaviour

These are deliberate escape hatches. Anyone who can set your environment can change what gets loaded,
so treat them as part of your deployment's trust boundary.

| variable | effect |
|---|---|
| `MLAYAX_NATIVE_DIR` | load `node_mlx.node` from this directory instead of the platform package — an arbitrary native library path |
| `MLAYAX_ALLOW_MIXED_MLX=1` | **waives the mixing guard** and proceeds with a foreign `libmlx.dylib` already resident. Documented as rarely correct; the result is undefined behaviour inside MLX |
| `MLAYAX_SHAPELESS`, `MLAYAX_PRETRANSPOSE`, `MLAYAX_ADDM` | performance switches only (`=1` enables) |
| `HF_HOME`, `HF_HUB_CACHE` | where checkpoints are cached, and therefore where a checkpoint is read from |

## Known sharp edges (not vulnerabilities)

- **One MLX runtime per process.** A second one steals our addon's symbols, because macOS resolves
  dynamic libraries by install name. This is dyld's rule, not a bug we can fix; the guard turns the
  crash into an explanation. Loading untrusted MLX builds in the same process as this one is a bad idea
  for the same reason, even with the waiver set.
- **Model output is not a security decision.** The library returns calibrated probabilities plus an
  escalation probability for exactly this reason; a hostile input can still produce a confident wrong
  answer. Do not auto-apply an action on `answer_confidence` alone without your own checks.
- **`confidence` from the `choice:11+` bucket is clamped** and therefore uncalibrated (the checkpoint's
  temperature for that bucket is outside the `[0.5, 5]` range the calibration clamps to). The library
  raises a `RuntimeWarning` when it happens rather than returning a number that looks calibrated.
- **The macOS floor is 26.2**, set by the pinned MLX build. On an older system the failure is a dyld
  error naming `libmlx.dylib`, which is confusing but not a vulnerability.