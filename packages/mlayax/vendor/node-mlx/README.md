# vendored node-mlx

Upstream: [`frost-beta/node-mlx`](https://github.com/frost-beta/node-mlx) @ `4bf8b1d6de32ae5ec402525bff1f041d73618275`
Package: `@frost-beta/mlx` 0.4.0 · Licence: MIT (`./LICENSE`, © frost-beta)

## What is here, and what is deliberately not

| file | origin |
|---|---|
| `core.cjs` | upstream `dist/core.js`, **one line patched** (see `PATCHES.md`) |
| `native-binding.cjs` | **ours** — resolves the native addon from the platform package |
| `LICENSE` | upstream |
| `PATCHES.md` | the exact diff against upstream |

Only `core.js` is vendored. Upstream's `dist/` also contains `nn/`, `optimizers/` and `utils.js`
(~400 KB, ~50 files) which the runtime never imports — the MLX ops we need are on the addon itself,
and the one `nn` value the port used to thread through (`DecisionModel`'s constructor argument) was
never read. Shipping 400 KB of dead JavaScript in the façade would be the opposite of what the size
budget is for. (The budget number lives in `scripts/check-size.mjs` and moves as the package grows — it
is a guard against dead weight, not a target, so this file deliberately does not repeat it. TASKS.md
§10.4 has the measurements.)

If a future phase needs `nn` (for example a slimmed wrapper that uses `nn.Linear`), vendor it then
and add it to this table.

## Why vendor at all

The alternative is a hand-written wrapper over the patched `.node`. Upstream's core layer is proven:
every performance number in `docs/PORTING.md` was measured through it, so vendoring keeps that
property and reduces the diff to one line. A slim wrapper remains the follow-up that shrinks the
surface further (TASKS.md §9).

## Why `.cjs`

`@fllstck/mlayax` is ESM (`"type": "module"`), so a `.js` file inside it would be parsed as ESM and
`require` would not exist. The `.cjs` extension makes CommonJS unambiguous without a nested
`package.json` marker. Our TypeScript entry points load it through `createRequire`, so the module
system boundary is explicit and confined to `src/mlx/binding.ts`.

## Licence

MIT, © frost-beta. Redistributed under the terms in `./LICENSE`; see the repository root `NOTICE`.
`core.cjs` is unmodified apart from the one patched line, and this file is not upstream's.