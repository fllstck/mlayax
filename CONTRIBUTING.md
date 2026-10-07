# Contributing

Thanks for looking. This repository is small, and almost everything in it exists because a number was
measured — so the bar for a change is not "it looks right" but "here is the measurement, and here is the
gate that keeps it right". [`TASKS.md`](TASKS.md) is the plan and the record; its §6 lists the quality
gates and its §10 the corrections found while building, including the wrong turns.

## Setup

```bash
npm install
npm run build    # tsc -b → packages/mlayax/dist (dist/ is what the tests and bench import)
npm test         # vitest; native-dependent tests skip themselves without a payload
bun test         # the same suite under Bun
npm run verify   # biome ci && tsc -b && tsc -p tsconfig.test.json && vitest run && licences
```

The native payload is **not** committed (it is ~206 MiB, mostly `mlx.metallib`). Tests that need it skip
themselves when it is absent, so the portable suite runs anywhere. To build it:

```bash
python3 -m venv .venv && .venv/bin/pip install mlx==0.32.3   # the wheel-class MLX (mode 1)
tools/native/build.sh                                        # clone node-mlx, patch, build, verify
tools/native/build.sh --mode source                          # build MLX from source (needs full Xcode)
tools/native/build.sh --check                                # verify an already-staged payload
```

`build.sh` refuses to stage a payload it cannot account for: it checks the three MLX files against the
published wheel hashes (mode 1), the `mlx.metallib` size class, the fused symbols the model needs, the
`@loader_path` rpath, the declared macOS floor against what `libmlx.dylib` actually requires, and
`SHA256SUMS` against the files on disk.

## The gates, and what they need

| command | needs | what it means |
|---|---|---|
| `npm run verify` | nothing | lint, types, dead code, tests, licence copies in sync. The commit gate |
| `npm run verify:release` | a staged payload | the above plus `publint`, `attw`, the tarball size/content gate, and `build.sh --check` |
| `npm test` | nothing (native tests skip) | the portable suite |
| `MLAYAX_MODEL_DIR=… npm run bench:check` | the real checkpoint | p50 within 1.3x of `bench/baseline.json`, RSS under the ceiling. Warn above 1.3x, fail above 2x, and **inconclusive** on a machine that is not the baseline's CPU |
| `npm run check:packed -- --project DIR` | tarballs installed in `DIR` | parity against an *installed* copy: the only test that notices a `files` allowlist mistake or an `exports` map that resolves to nothing |
| `npm run test:coverage` | nothing | **does not pass**: `src/mlx` branch coverage is 73.8 % against an 85 % threshold, and the missing branches are the forward pass's error paths. Contributions that add tests there are the way to fix it (TASKS.md §10.5) |

CI runs these as eight jobs; [`.github/workflows/ci.yml`](.github/workflows/ci.yml) explains each one
and why the macOS jobs pin `macos-26`.

## House rules

- **Numbers live in two READMEs, and they move together.** The performance table appears in
  [`README.md`](README.md) and [`packages/mlayax/README.md`](packages/mlayax/README.md); a change to one
  is a change to both. The same goes for the compatibility table.
- **A changed number is a changelog entry.** `bench/baseline.json` is refreshed deliberately, never to
  match whatever the code now does — a baseline that drifts with the code is not a gate.
- **Corrections get recorded, not fixed quietly.** When a claim in `TASKS.md`, `docs/PORTING.md` or this
  file turns out to be wrong, it is retracted in place with the measurement that falsified it (§10 has
  nine of them). That is the most useful part of the repository for whoever comes next.
- **The prompt is the contract.** `src/core` renders exactly what the Python reference renders —
  key order, float formatting, integral values, criterion separators. If you change how a question or a
  state is rendered, regenerate the fixtures with `tools/reference/laya_ref.py` and re-run parity;
  do not adjust the expectations to match.
- **Comments are shipped.** They are preserved into `dist/`, and the façade budget is a cap on how much
  the code explains — which is why the source maps were dropped instead of the explanations.
- **Conventional commits** (`feat(mlx):`, `fix(core):`, `docs(tasks):`, `chore(repo):`), imperative,
  with the reason in the body when it is not obvious.

## The MLX bump playbook

Everything here is pinned on purpose: a bump changes the numbers in `bench/baseline.json` and the
`SHA256SUMS` in the payload, and both are things users rely on. Do it deliberately, in this order.

1. **Decide the target.** The MLX tag comes from `ml-explore/mlx`; the binding comes from
   `frost-beta/node-mlx`. Check whether upstream has revived first — if `@frost-beta/mlx` ships a build
   against a current MLX, dropping this fork is a dependency swap rather than a patch to maintain.
2. **Re-apply the patch.** `tools/native/node-mlx-mlx32.patch` (10 files, 242 added lines) is the API
   drift between MLX 0.25 and 0.32. For a new version, work in a scratch checkout and use the kit that
   derived it the first time:
   - `tools/native/fix_overloads.py` — finds `&mx::op` overloads that became ambiguous, by parsing
     clang's deduced-pack diagnostic. Takes `MLX_PREFIX`.
   - `tools/native/shaplessprobe.ts` — measures which ops MLX can still infer shapes for.
   - `tools/native/chainprobe.ts`, `tools/native/bisect.ts` — bisect a failing chain / a bad op.
3. **Build and verify.** `tools/native/build.sh` (mode 1) — it will refuse a slow MLX class, a missing
   fused symbol, a stale rpath, or a declared macOS floor that disagrees with the dylib. Then
   `npm run verify:release`.
4. **Re-measure.** `npm run bench` before and after; then, if the numbers genuinely moved, update
   `bench/baseline.json` **and** both README tables **and** the changelog. Parity first: `npm test`
   (tiny fixture) and `MLAYAX_MODEL_DIR=… npx vitest run test/parity.real.test.ts`.
5. **Update every pin.** A bump touches more than `build.sh`; these are all of them:

   | pin | where |
   |---|---|
   | node-mlx commit, MLX tag | `tools/native/build.sh`, `.github/workflows/ci.yml` (env) |
   | wheel-class `metallib`/`libmlx`/`libjaccl` sha256 + byte sizes | `tools/native/build.sh`, `test/payload.test.ts` |
   | §2's addon reference hash | `tools/native/build.sh` (`ADDON_SHA256_REFERENCE`) — re-derive, and only claim it is reproduced if a build at `MACOSX_DEPLOYMENT_TARGET=26.0` actually reproduces it (§10.3) |
   | `VERSION` + `SHA256SUMS` | written by `build.sh`; never hand-edited |
   | provenance in prose | `NOTICE`, `licenses/README.md`, `packages/*/NOTICE` |
   | vendored layer note | `packages/mlayax/vendor/node-mlx/PATCHES.md` (and `core.cjs`, if upstream's `dist/core.js` moved) |
   | compatibility tables | both READMEs |

6. **Send it upstream.** The patch is offered to `frost-beta/node-mlx` as goodwill; a bump is a good
   moment to refresh that offer, and to re-run the comparison in `docs/ECOSYSTEM.md` (the alternatives
   are actively maintained, and one of them catching up would change the argument).

The same discipline applies to the model checkpoint: `tools/reference/laya_ref.py` regenerates the
reference payloads, and `tools/reference/make_tiny_checkpoint.py` the committed fixture. A checkpoint
bump means regenerating both, re-running parity, and updating `VERSION`'s `mlx_tag`-adjacent fields in
the payload if the encoder changed.

## Licences and notices

`licenses/` holds the full texts; `NOTICE` summarises who is covered by what.
`scripts/sync-licenses.mjs` copies `licenses/` into each package, and `npm run verify` fails if the
copies drift — so change the canonical file (`npm run sync:licenses`) rather than the copies. The Laya
weights are Apache-2.0 (Convai Innovations) and are never bundled; nothing may add an install script,
a `postinstall` hook, or a network call at import time.

## Releases

Manual, in the order given in [`TASKS.md`](TASKS.md) §5 Phase 9: platform package first, then the
façade, then the registry-install check on both Node and Bun. `npm run verify:release` and
`npm run bench:check` are pre-flight, not optional.

Two things about npm that cost a release an afternoon, both worth knowing before you start:

- **A brand-new package name goes through a staged release.** After `npm publish` returns `202`, the
  packument can advertise `latest: <your version>` while the name is still held by a `0.0.0-stage`
  placeholder and the real tarball answers `404`. Nothing is broken and nothing needs re-publishing;
  the version appears when the review completes. Check the *tarball*, not the metadata:

  ```bash
  curl -sIL -o /dev/null -w '%{http_code}\n' https://registry.npmjs.org/@fllstck/mlayax/-/mlayax-0.1.0.tgz
  ```

- **Never verify a publish from the machine that published it.** `npm i` there succeeds from the local
  cache, which is holding the tarball you just uploaded — so a green install can sit on top of a
  release the registry is not serving yet. A scratch directory with an empty cache is the honest test:

  ```bash
  npm i <name> --cache /tmp/fresh-cache --prefer-online --no-audit
  ```

Also expect the upload of the 68 MB platform package to be slow (tens of seconds) and to need npm's
browser 2FA step; an `ERR_SSL_SSL/TLS_ALERT_BAD_RECORD_MAC` mid-upload is npm retrying, not a lost
release.

## Security

See [`SECURITY.md`](SECURITY.md) — that file also lists the environment variables that deliberately
change load-time behaviour (`MLAYAX_NATIVE_DIR`, `MLAYAX_ALLOW_MIXED_MLX`), which is worth reading before
you make a change that reads the environment.

By contributing you agree your work is licensed under the MIT licence in [`LICENSE`](LICENSE).