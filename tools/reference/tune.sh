#!/usr/bin/env bash
# Configuration sweep for the MLX/TypeScript port. Prints one JSON line per configuration.
set -euo pipefail
cd "$(dirname "$0")/.."
ROWS="${1:-1}"
ITERS="${2:-25}"

run() {
  local label="$1"; shift
  local out
  out=$(env "$@" node src/mlx/tune.ts "$ROWS" "$ITERS" 2>&1 | tail -1)
  echo "$out" | python3 -c "
import json,sys
line = sys.stdin.read().strip()
try:
    d = json.loads(line)
    print(f\"{d['config']:38} p50 {d['p50']:7.2f} ms   min {d['min']:7.2f}   new-shape {d['newShapeMs']:7.2f}   rss {d['rssMiB']} MiB\")
except Exception:
    print('FAILED:', line[:200])
"
}

echo "rows=$ROWS iterations=$ITERS"
run baseline ANALOG=1
run tidy-off LAYA_TIDY=0
run compile LAYA_COMPILE=1
run compile+pretranspose LAYA_COMPILE=1 LAYA_PRETRANSPOSE=1
run compile+addmm LAYA_COMPILE=1 LAYA_ADDM=1
run compile+pretranspose+addmm LAYA_COMPILE=1 LAYA_PRETRANSPOSE=1 LAYA_ADDM=1
run compile+fastsynch LAYA_COMPILE=1 MLX_METAL_FAST_SYNCH=1
run compile+wired LAYA_COMPILE=1 LAYA_WIRED_MB=4096
run compile+cachelimit LAYA_COMPILE=1 LAYA_CACHE_MB=256
run shapeless LAYA_COMPILE=1 LAYA_SHAPELESS=1
run shapeless+fastsynch LAYA_COMPILE=1 LAYA_SHAPELESS=1 MLX_METAL_FAST_SYNCH=1