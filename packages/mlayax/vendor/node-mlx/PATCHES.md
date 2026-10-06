# Patch applied to the vendored node-mlx core layer

Upstream source: `@frost-beta/mlx` 0.4.0 → `dist/core.js`
(commit `4bf8b1d6de32ae5ec402525bff1f041d73618275`)

Exactly one line differs. Everything else in `core.cjs` is upstream verbatim, including the
`__importDefault` helper, the `Complex` helper and the `stream` implementation.

## The patch

Upstream hardcodes where the addon lives:

```js
const node_mlx_node_1 = __importDefault(require("../build/Release/node_mlx.node"));
```

That assumes the addon was compiled in place by `@frost-beta/mlx`'s own `install.js`. We do not build
anything on the user's machine — the addon ships prebuilt in `@fllstck/mlayax-darwin-arm64` — so the
path is resolved instead:

```js
const node_mlx_node_1 = __importDefault(
  require(require("./native-binding.cjs").resolveNativeBinding()),
);
```

`native-binding.cjs` is ours (see `./README.md`) and resolves, in order:

1. `$MLAYAX_NATIVE_DIR/node_mlx.node`, when that environment variable is set;
2. `@fllstck/mlayax-darwin-arm64/lib/node_mlx.node`, resolved from this file's location.

It throws a described error naming every path it tried when neither exists, and rejects non-darwin /
non-arm64 platforms up front.

## Why not a literal path

We could have kept upstream's line untouched and shipped a copy of the addon at
`vendor/node-mlx/build/Release/`, but that would duplicate a 2.4 MB binary into the façade package
whose entire job is to be small, and would defeat the point of the platform package.

A symlink at that path would keep the line untouched, but npm does not reliably preserve symlinks
across `npm pack` and registry install, so the published artefact would behave differently from the
working tree.

## Verifying the diff by hand

```bash
curl -L https://registry.npmjs.org/@frost-beta/mlx/-/mlx-0.4.0.tgz | tar -xzO package/dist/core.js \
  | diff - core.cjs
```

Expect exactly the one `require` line above, plus this file's header comment.