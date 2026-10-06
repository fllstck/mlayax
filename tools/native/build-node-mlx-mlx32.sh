#!/usr/bin/env bash
# Build @frost-beta/mlx (node-mlx) against MLX 0.32.x and install it into this project.
#
# Upstream node-mlx v0.4.0 vendors MLX 0.25.0. This script applies tools/node-mlx-mlx32.patch
# (the changes needed for the newer MLX API) and links against a *prebuilt* MLX instead of
# building MLX from source, which would need the full Xcode Metal toolchain (a CommandLineTools
# install has no `metal` compiler).
#
# Usage: tools/build-node-mlx-mlx32.sh [mlx-prefix]
#
# The MLX prefix must contain include/mlx/*.h, lib/libmlx.dylib and lib/mlx.metallib. Defaults to
# the MLX that the project's Python environment installed, because that build measured ~35% faster
# than Homebrew's bottle for this workload (see PORTING.md).
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
MLX_PREFIX="${1:-}"
if [[ -z "$MLX_PREFIX" ]]; then
  for candidate in "$HERE/../.venv/lib/python3.12/site-packages/mlx" "$(brew --prefix mlx 2>/dev/null)"; do
    if [[ -f "$candidate/lib/libmlx.dylib" ]]; then MLX_PREFIX="$candidate"; break; fi
  done
fi
if [[ -z "$MLX_PREFIX" || ! -f "$MLX_PREFIX/lib/libmlx.dylib" ]]; then
  echo "No prebuilt MLX found. Install one (brew install mlx, or a Python 'mlx' package) and pass its prefix." >&2
  exit 1
fi
echo "using MLX from $MLX_PREFIX"

MLX_REV="${MLX_REV:-v0.32.3}"
WORK="${WORK:-/tmp/node-mlx-build}"
if [[ ! -d "$WORK/.git" ]]; then
  git clone --depth 1 --recurse-submodules --shallow-submodules \
    https://github.com/frost-beta/node-mlx "$WORK"
fi
cd "$WORK"
git checkout -- . 2>/dev/null || true
git -C deps/mlx fetch --depth 1 origin tag "$MLX_REV" 2>/dev/null || true
git -C deps/mlx checkout -q "$MLX_REV"
git apply --3way "$HERE/tools/node-mlx-mlx32.patch"

[[ -d node_modules ]] || npm install --ignore-scripts --no-audit --no-fund
rm -rf build
MLX_PREFIX="$MLX_PREFIX" npx cmake-js build --prefer-make --parallel "$(sysctl -n hw.ncpu)"

DEST="$HERE/node_modules/@frost-beta/mlx/build/Release"
mkdir -p "$DEST"
[[ -f "$DEST/node_mlx.node" ]] && cp "$DEST/node_mlx.node" "$DEST/node_mlx.node.orig" || true
cp build/Release/node_mlx.node "$DEST/node_mlx.node"
echo "installed $DEST/node_mlx.node (built against MLX $MLX_REV)"
echo "verify with: node src/mlx/parity.ts fixtures/ref/fp16.json fixtures/ref/fp32.json"