#!/usr/bin/env bash
#
# Build the patched node-mlx addon and stage the native payload for
# @fllstck/mlayax-darwin-arm64. This is TASKS.md §5.
#
# What it produces, in packages/mlayax-darwin-arm64/:
#
#   lib/node_mlx.node   lib/libmlx.dylib   lib/libjaccl.dylib   lib/mlx.metallib
#   VERSION   SHA256SUMS
#
# Two build modes:
#
#   prebuilt (default)  Link a *prebuilt* MLX (a Python `mlx` wheel or Homebrew). Works with
#                       CommandLineTools only, because no Metal compiler is needed. The wheel-class
#                       build is required: Homebrew's is ~35 % slower (137 MB metallib vs 190 MB).
#   source              Build MLX from deps/mlx. Needs the full Xcode Metal toolchain. Intended for
#                       the macOS CI runner as a provenance experiment; gated on the benchmark.
#
# Usage:
#   tools/native/build.sh [options]
#
#     --mode prebuilt|source   build mode (default: prebuilt)
#     --mlx-prefix DIR         prebuilt MLX prefix (default: probe the wheel, then brew)
#     --deployment-target VER  MACOSX_DEPLOYMENT_TARGET (default: what the linked libmlx requires,
#                              which is the honest floor; pass 26.0 to reproduce §2's addon hash
#                              exactly, which turns that check into a hard gate)
#     --work DIR               scratch checkout (default: /tmp/mlayax-node-mlx)
#     --out DIR                package dir to stage into (default: packages/mlayax-darwin-arm64)
#     --check                  verify an already-staged payload, build nothing
#     -h, --help               this text
#
# The `--check` mode is the same set of assertions the release gate runs, so it is worth wiring
# into CI: it validates hashes, the MLX size class, the fused symbols, and the rpath rewrite.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

# ---------------------------------------------------------------------------------------------
# Pinned provenance. TASKS.md §2. Change these only deliberately, and re-derive the patch first.
# ---------------------------------------------------------------------------------------------
NODE_MLX_REPO="https://github.com/frost-beta/node-mlx"
NODE_MLX_COMMIT="4bf8b1d6de32ae5ec402525bff1f041d73618275"
MLX_TAG="v0.32.3"

# The wheel-class MLX build this artifact was measured against. Copied, not built, so these are
# hard gates: a mismatch means the wrong MLX was linked and every number in §2 is void.
WHEEL_METALLIB_SHA256="95974b7de464ec2be7833a914ca5a53d283a70789470fd84218738fc8fcea3e1"
WHEEL_LIBMLX_SHA256="ebc8a5f5465afb79d3d9c8f5b0929de1a9b277b7179db8672f92da77ae110a1e"
WHEEL_LIBJACCL_SHA256="949e901df832d600a71ff2bc5f3738d55b010668102f36e071e1890807a965cd"

# The addon is compiled here, so its hash is a reproduction signal, not a gate — unless the build
# matches the conditions it was measured under, in which case it becomes a gate. Confirmed
# 2026-10-06: with node-mlx 4bf8b1d + MLX 0.32.3 (wheel) + this patch + AppleClang 21, a build with
# MACOSX_DEPLOYMENT_TARGET=26.0 reproduces this hash byte-for-byte. The shipped default is not 26.0:
# see SPIKE_DEPLOYMENT_TARGET below.
ADDON_SHA256_REFERENCE="d684eaccf655d66d1d1fe03d5de8ee9e1e1b860f7da06e02a7a54984178b9ecd"

# §2's addon hash was measured with the toolchain default deployment target. We deliberately do not
# ship that default: the wheel-class libmlx.dylib is built for macOS 26.2, so an addon advertising
# 26.0 would be lying about its floor (ld even warns about it), and a user on macOS 26.1 would get a
# confusing dyld failure instead of a clear "requires macOS 26.2". Building at the linked dylib's
# own minos removes the warning and makes the floor true — at the cost of one differing hash, which
# is why the reproduction check is conditional.
SPIKE_DEPLOYMENT_TARGET="26.0"

# Size classes. §2/§8.7: the metallib size class predicts throughput.
METALLIB_MIN_BYTES=$((180 * 1000 * 1000)) # wheel-class floor; brew's is ~137 MB
SIZE_CLASS_WHEEL="wheel-class"

# The fused ops the model needs. These are C++ symbols, so they are mangled — TASKS.md §5 calls them
# mlx_fast_layer_norm / mlx_fast_rope / mlx_fast_scaled_dot_product_attention, which is the mlx-c
# spelling. We link the C++ libmlx directly, so match the mangled prefix `_ZN3mlx4core4fast`.
FUSED_SYMBOL_PREFIX="_ZN3mlx4core4fast"
FUSED_SYMBOL_COUNT_MIN=4 # layer_norm ×1, rope ×2 (two overloads), sdpa ×1

# ---------------------------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------------------------
MODE="prebuilt"
MLX_PREFIX_ARG=""
DEPLOYMENT_TARGET_ARG=""
WORK="${WORK:-/tmp/mlayax-node-mlx}"
OUT="$REPO/packages/mlayax-darwin-arm64"
CHECK_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="${2:?--mode needs a value}"; shift 2 ;;
    --mlx-prefix) MLX_PREFIX_ARG="${2:?--mlx-prefix needs a value}"; shift 2 ;;
    --deployment-target) DEPLOYMENT_TARGET_ARG="${2:?--deployment-target needs a value}"; shift 2 ;;
    --work) WORK="${2:?--work needs a value}"; shift 2 ;;
    --out) OUT="${2:?--out needs a value}"; shift 2 ;;
    --check) CHECK_ONLY=1; shift ;;
    -h|--help) sed -n '2,29p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1 (try --help)" >&2; exit 2 ;;
  esac
done

log()  { printf '\033[1m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[33mwarn:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

sha256_of() { shasum -a 256 "$1" | awk '{print $1}'; }
size_of()   { stat -f%z "$1"; }
ver_field() { grep "^$2=" "$1/VERSION" 2>/dev/null | cut -d= -f2-; }

# `otool -l` prints rpaths as `         path /some/dir (offset 12)`; strip the offset suffix or
# install_name_tool -delete_rpath will be handed a path that cannot match. (This exact mistake left
# the absolute venv rpath in the first build.)
rpaths_of() {
  otool -l "$1" | awk '/LC_RPATH/{getline; getline; sub(/^ *path /,""); sub(/ \(offset [0-9]+\)$/,""); print}'
}

# The minimum macOS the binary claims to run on. LC_BUILD_VERSION is the modern form;
# LC_VERSION_MIN_MACOSX is the pre-10.14 one.
minos_of() {
  otool -l "$1" | awk '/LC_BUILD_VERSION|LC_VERSION_MIN_MACOSX/{f=1} f&&/(minos|version)/{print $2; exit}'
}

# ---------------------------------------------------------------------------------------------
# Assertions shared by build and --check
# ---------------------------------------------------------------------------------------------
assert_no_absolute_rpath() {
  # §5: "rpath must be @loader_path-relative (the spike's build currently points at a Python venv —
  # this must not survive)." The general form of that hazard is any absolute rpath.
  local file="$1"
  local rpaths bad=0
  rpaths="$(rpaths_of "$file")"
  while IFS= read -r rp; do
    [[ -z "$rp" ]] && continue
    case "$rp" in
      @loader_path*|@executable_path*) log "  rpath ok: $rp ($(basename "$file"))" ;;
      *) warn "  ABSOLUTE rpath in $(basename "$file"): $rp"; bad=1 ;;
    esac
  done <<< "$rpaths"
  return $bad
}

assert_no_venv_anywhere() {
  # A belt-and-braces substring scan: an rpath is the known vector, but a build path could also be
  # baked into a debug section. Either one is a leak of this machine's layout into the tarball.
  local file="$1"
  if strings -a "$file" | grep -qiE '/\.venv/|site-packages|python3\.[0-9]+'; then
    warn "  $(basename "$file") contains a Python venv path string"
    return 1
  fi
  return 0
}

assert_fused_symbols() {
  local addon="$1" dylib="$2" n
  n="$(nm -u "$addon" | grep -c "$FUSED_SYMBOL_PREFIX" || true)"
  log "  addon imports $n fused symbols ($FUSED_SYMBOL_PREFIX*)"
  [[ "$n" -ge "$FUSED_SYMBOL_COUNT_MIN" ]] || { warn "  addon imports only $n < $FUSED_SYMBOL_COUNT_MIN"; return 1; }
  n="$(nm -gU "$dylib" | grep -c "$FUSED_SYMBOL_PREFIX" || true)"
  log "  libmlx defines $n fused symbols"
  [[ "$n" -ge "$FUSED_SYMBOL_COUNT_MIN" ]] || { warn "  libmlx defines only $n < $FUSED_SYMBOL_COUNT_MIN"; return 1; }
  return 0
}

assert_size_class() {
  local metallib="$1" bytes
  bytes="$(size_of "$metallib")"
  log "  metallib: $bytes bytes ($((bytes / 1000 / 1000)) MB raw)"
  if [[ "$bytes" -lt "$METALLIB_MIN_BYTES" ]]; then
    warn "  metallib is below the wheel-class floor of $METALLIB_MIN_BYTES bytes."
    warn "  This is the ~35 %-slower build class (Homebrew / @johnhenry). Refusing."
    return 1
  fi
  return 0
}

check_payload() {
  local pkg="$1"
  local lib="$pkg/lib"
  local ok=0
  log "verifying payload in $pkg"

  for f in node_mlx.node libmlx.dylib libjaccl.dylib mlx.metallib; do
    [[ -f "$lib/$f" ]] || { warn "  missing $lib/$f"; ok=1; }
  done
  [[ "$ok" -eq 0 ]] || return 1

  log "sha256"
  local want actual
  for pair in "mlx.metallib:$WHEEL_METALLIB_SHA256" "libmlx.dylib:$WHEEL_LIBMLX_SHA256" "libjaccl.dylib:$WHEEL_LIBJACCL_SHA256"; do
    want="${pair##*:}"; actual="$(sha256_of "$lib/${pair%%:*}")"
    if [[ "$actual" == "$want" ]]; then
      log "  ${pair%%:*}: ok (${actual:0:12}…)"
    else
      warn "  ${pair%%:*}: MISMATCH"
      warn "    expected ${want:0:12}… (wheel-class)"
      warn "    got      ${actual:0:12}…"
      ok=1
    fi
  done
  actual="$(sha256_of "$lib/node_mlx.node")"
  # The addon is the one file we compile and then modify (rpath rewrite + adhoc re-sign), so its
  # shipped hash can never equal a raw-build hash. VERSION records the pre-surgery hash; that is the
  # one comparable to §2's reference.
  local pre s2
  pre="$(ver_field "$pkg" addon_sha256_pre_rpath)"
  s2="$(ver_field "$pkg" s2_addon_reproduced)"
  if [[ "$actual" == "$ADDON_SHA256_REFERENCE" ]]; then
    log "  node_mlx.node: ok (reproduced §2 ${actual:0:12}…, unmodified)"
  elif [[ "$pre" == "$ADDON_SHA256_REFERENCE" ]]; then
    log "  node_mlx.node: shipped ${actual:0:12}…; pre-rpath hash reproduced §2 exactly"
  elif [[ "$s2" == "yes" ]]; then
    log "  node_mlx.node: shipped ${actual:0:12}… (VERSION claims §2 was reproduced)"
  else
    warn "  node_mlx.node: shipped ${actual:0:12}…, pre-rpath ${pre:0:12}…"
    warn "  != §2 reference ${ADDON_SHA256_REFERENCE:0:12}… — see s2_addon_reproduced in VERSION for why"
  fi

  log "size class"; assert_size_class "$lib/mlx.metallib" || ok=1
  log "fused symbols"; assert_fused_symbols "$lib/node_mlx.node" "$lib/libmlx.dylib" || ok=1

  log "rpath"
  for f in node_mlx.node libmlx.dylib libjaccl.dylib; do
    assert_no_absolute_rpath "$lib/$f" || ok=1
  done
  local addon_rpaths
  addon_rpaths="$(rpaths_of "$lib/node_mlx.node")"
  if ! grep -qx '@loader_path' <<< "$addon_rpaths"; then
    warn "  node_mlx.node has no @loader_path rpath — it cannot find libmlx.dylib when packaged"
    ok=1
  fi

  log "runtime floor (recorded for the README compatibility table)"
  local minos declared
  minos="$(minos_of "$lib/libmlx.dylib")"
  declared="$(ver_field "$pkg" min_macos)"
  log "  libmlx.dylib requires macOS $minos; VERSION declares min_macos=$declared"
  if [[ -z "$declared" ]]; then
    warn "  VERSION has no min_macos field"
    ok=1
  elif [[ "$minos" != "$declared" ]]; then
    warn "  declared floor $declared is not what libmlx.dylib actually needs ($minos)"
    ok=1
  fi

  log "path hygiene"
  for f in node_mlx.node libmlx.dylib libjaccl.dylib; do
    assert_no_venv_anywhere "$lib/$f" || ok=1
  done

  [[ "$ok" -eq 0 ]] || { warn "payload verification FAILED"; return 1; }
  log "payload verification passed"
  return 0
}

# ---------------------------------------------------------------------------------------------
# --check
# ---------------------------------------------------------------------------------------------
if [[ "$CHECK_ONLY" -eq 1 ]]; then
  check_payload "$OUT"
  exit $?
fi

# ---------------------------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------------------------
[[ "$(uname -s)" == "Darwin" ]] || die "macOS only (this artifact is darwin/arm64)"
[[ "$(uname -m)" == "arm64" ]]  || die "arm64 only (Intel Macs are out of scope for 0.1.0)"
# Validate --mode explicitly: the build branches on `prebuilt` vs *everything else*, so a typo would
# otherwise be silently treated as `source`.
case "$MODE" in
  prebuilt|source) ;;
  *) die "unknown --mode '$MODE' — expected 'prebuilt' or 'source'" ;;
esac
command -v node >/dev/null 2>&1 || die "node not found"
command -v npx  >/dev/null 2>&1 || die "npx not found"; # cmake-js comes from node-mlx's own devDependencies, checked after npm install

# ---------------------------------------------------------------------------------------------
# Resolve the prebuilt MLX (mode: prebuilt)
# ---------------------------------------------------------------------------------------------
resolve_mlx_prefix() {
  local candidates=() c pat base
  [[ -n "$MLX_PREFIX_ARG" ]] && candidates+=("$MLX_PREFIX_ARG")
  [[ -n "${MLX_PREFIX:-}" ]] && candidates+=("$MLX_PREFIX")
  # Look for a Python `mlx` wheel before Homebrew: the wheel is the fast class and the one §2 was
  # measured on, and it is what a `pip install mlx==0.32.3` next to this repo produces. Explicit
  # globs rather than a deep `find` so the probe is instant and its search space is predictable.
  for base in "$REPO" "$REPO/.." "$HOME"; do
    for pat in \
      "$base"/.venv/lib/python3.*/site-packages/mlx \
      "$base"/*/.venv/lib/python3.*/site-packages/mlx \
      "$base"/*/*/.venv/lib/python3.*/site-packages/mlx \
      "$base"/.venv/lib/python3.*/site-packages/mlx/lib/cmake ; do
      for c in $pat; do [[ -d "$c" ]] && candidates+=("$c"); done
    done
  done
  # Homebrew last: correct version, ~35 % slower on this workload, caught by the size-class gate.
  while IFS= read -r c; do candidates+=("$c"); done < <(brew --prefix mlx 2>/dev/null || true)
  candidates+=("/opt/homebrew/opt/mlx")

  for c in "${candidates[@]}"; do
    [[ -z "$c" ]] && continue
    if [[ -f "$c/lib/libmlx.dylib" && -f "$c/lib/mlx.metallib" && -d "$c/include/mlx" ]]; then
      echo "$c"; return 0
    fi
  done
  return 1
}

if [[ "$MODE" == "prebuilt" ]]; then
  MLX_PREFIX="$(resolve_mlx_prefix || true)"
  [[ -n "$MLX_PREFIX" ]] || die "no usable prebuilt MLX found.
  Install one of:
    - python:  python3 -m venv .venv && .venv/bin/pip install mlx==0.32.3   (the fast class)
    - homebrew: brew install mlx                                           (the SLOW class — warn)
  then re-run, or pass --mlx-prefix DIR."

  MLX_METAL_SHA="$(sha256_of "$MLX_PREFIX/lib/mlx.metallib")"
  MLX_METAL_BYTES="$(size_of "$MLX_PREFIX/lib/mlx.metallib")"
  MLX_MINOS="$(minos_of "$MLX_PREFIX/lib/libmlx.dylib")"
  log "using prebuilt MLX at $MLX_PREFIX"
  log "  metallib $MLX_METAL_BYTES bytes, sha256 ${MLX_METAL_SHA:0:12}…, requires macOS $MLX_MINOS"

  if [[ "$MLX_METAL_SHA" != "$WHEEL_METALLIB_SHA256" ]]; then
    if [[ "$MLX_METAL_BYTES" -lt "$METALLIB_MIN_BYTES" ]]; then
      die "the MLX at $MLX_PREFIX is the SLOW build class and is not the one §2 was measured on.
  Its mlx.metallib is $MLX_METAL_BYTES bytes; the wheel-class build is ~190 MB and ~35 % faster.
  Point --mlx-prefix at a Python mlx wheel:
    python3 -m venv .venv && .venv/bin/pip install mlx==0.32.3
  or, if a slower artifact is a deliberate decision, lower METALLIB_MIN_BYTES and note it in
  CHANGELOG.md."
    fi
    warn "metallib sha256 is not the pinned wheel hash, but the size class is right — continuing"
  fi
else
  command -v xcrun >/dev/null 2>&1 && xcrun --find metal >/dev/null 2>&1 \
    || die "mode=source needs the full Xcode Metal toolchain; 'xcrun --find metal' fails here.
  This machine has CommandLineTools only, so mode=prebuilt is the local option."
  log "mode=source: MLX will be built from deps/mlx ($MLX_TAG) by CMake"
fi

# ---------------------------------------------------------------------------------------------
# Fetch node-mlx at the pinned commit
# ---------------------------------------------------------------------------------------------
log "scratch checkout: $WORK"
mkdir -p "$WORK"
if [[ ! -d "$WORK/.git" ]]; then
  git init -q "$WORK"
  git -C "$WORK" remote add origin "$NODE_MLX_REPO" 2>/dev/null || git -C "$WORK" remote set-url origin "$NODE_MLX_REPO"
fi
# Fetch the exact commit rather than cloning HEAD: --depth 1 of a moving default branch is how a
# "pinned" build silently drifts.
if ! git -C "$WORK" cat-file -e "$NODE_MLX_COMMIT^{commit}" 2>/dev/null; then
  log "fetching node-mlx @ ${NODE_MLX_COMMIT:0:12}"
  git -C "$WORK" fetch -q --depth 1 origin "$NODE_MLX_COMMIT"
fi
git -C "$WORK" checkout -q --detach "$NODE_MLX_COMMIT"
git -C "$WORK" submodule update -q --init --recursive --depth 1
log "  HEAD = $(git -C "$WORK" rev-parse HEAD)"

# ---------------------------------------------------------------------------------------------
# Pin deps/mlx at the tag, then apply the patch
# ---------------------------------------------------------------------------------------------
log "pinning deps/mlx to $MLX_TAG"
git -C "$WORK/deps/mlx" fetch -q --depth 1 origin "refs/tags/$MLX_TAG:refs/tags/$MLX_TAG" 2>/dev/null || true
git -C "$WORK/deps/mlx" checkout -q "$MLX_TAG"
log "  deps/mlx = $(git -C "$WORK/deps/mlx" describe --tags --always)"

log "applying tools/native/node-mlx-mlx32.patch"
git -C "$WORK" checkout -q -- .
# --3way so a re-run over an already-patched tree is a clean no-op rather than a hard failure.
if git -C "$WORK" apply --check "$HERE/node-mlx-mlx32.patch" 2>/dev/null; then
  git -C "$WORK" apply "$HERE/node-mlx-mlx32.patch"
  log "  patch applied"
elif git -C "$WORK" apply --check --3way "$HERE/node-mlx-mlx32.patch" 2>/dev/null; then
  git -C "$WORK" apply --3way "$HERE/node-mlx-mlx32.patch"
  log "  patch applied (3-way)"
else
  warn "patch did not apply cleanly — checking whether it is already applied"
  if grep -q "MLX_USE_PREBUILT" "$WORK/CMakeLists.txt"; then
    log "  tree already carries the patch; continuing"
  else
    die "patch failed to apply to $WORK — re-derive it (see CONTRIBUTING.md, MLX bump playbook)"
  fi
fi

# ---------------------------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------------------------
if [[ ! -d "$WORK/node_modules" ]]; then
  log "npm install --ignore-scripts"
  (cd "$WORK" && npm install --ignore-scripts --no-audit --no-fund)
fi

rm -rf "$WORK/build"
CMAKE_JS="$WORK/node_modules/.bin/cmake-js"
[[ -x "$CMAKE_JS" ]] || die "cmake-js missing from $WORK/node_modules — node-mlx's devDependencies did not install"
# In source mode pass a bogus MLX_PREFIX: the patched CMakeLists falls back to `brew --prefix mlx`
# when MLX_PREFIX is empty, which would silently build mode 2 against Homebrew's slow class. A
# non-existent prefix keeps MLX_USE_PREBUILT FALSE and takes the add_subdirectory(deps/mlx) branch.
if [[ "$MODE" == "prebuilt" ]]; then
  CMAKE_MLX_PREFIX="$MLX_PREFIX"
  # Default to the floor the prebuilt MLX itself requires. Without this, ld warns that we are
  # "building for macOS-X, but linking with dylib ... built for newer version Y" and the artifact's
  # advertised floor is a lie: libmlx.dylib is what actually refuses to load on an older OS.
  CMAKE_OSX_TARGET="${DEPLOYMENT_TARGET_ARG:-$MLX_MINOS}"
  if [[ "$CMAKE_OSX_TARGET" != "$MLX_MINOS" ]]; then
    warn "  --deployment-target $CMAKE_OSX_TARGET understates the floor: the linked libmlx.dylib needs macOS $MLX_MINOS"
  fi
else
  CMAKE_MLX_PREFIX="/nonexistent-mode-source"
  CMAKE_OSX_TARGET="${DEPLOYMENT_TARGET_ARG:-${MACOSX_DEPLOYMENT_TARGET:-14.0}}"
fi
log "building for macOS >= $CMAKE_OSX_TARGET with $(sysctl -n hw.ncpu) cores"
(cd "$WORK" && MLX_PREFIX="$CMAKE_MLX_PREFIX" \
  "$CMAKE_JS" build --prefer-make --parallel "$(sysctl -n hw.ncpu)" \
  --CDCMAKE_OSX_DEPLOYMENT_TARGET="$CMAKE_OSX_TARGET") \
  || die "cmake-js build failed"

BUILT="$WORK/build/Release/node_mlx.node"
[[ -f "$BUILT" ]] || die "build produced no $BUILT"
# Record the raw hash before the rpath rewrite: this is the number §2 pins, and the only way to
# check that a rebuild on another machine or toolchain produced the same code.
ADDON_SHA_PRE_RPATH="$(sha256_of "$BUILT")"
log "addon built: $(size_of "$BUILT") bytes, sha256 (pre-rpath) ${ADDON_SHA_PRE_RPATH:0:12}…"
if [[ "$ADDON_SHA_PRE_RPATH" == "$ADDON_SHA256_REFERENCE" ]]; then
  log "  reproduces the §2 reference hash exactly (${ADDON_SHA256_REFERENCE:0:12}…)"
elif [[ "$CMAKE_OSX_TARGET" == "$SPIKE_DEPLOYMENT_TARGET" ]]; then
  die "the deployment target matches §2's measurement condition ($SPIKE_DEPLOYMENT_TARGET) but the
  addon hash does not match ${ADDON_SHA256_REFERENCE:0:12}… — the patch, the MLX build, or the
  compiler has drifted. Refusing to ship an unexplained binary."
else
  warn "  addon hash differs from §2's ${ADDON_SHA256_REFERENCE:0:12}… — expected, because the
  deployment target is $CMAKE_OSX_TARGET and not $SPIKE_DEPLOYMENT_TARGET. To prove the patch still
  reproduces the measured binary byte-for-byte, re-run with --deployment-target $SPIKE_DEPLOYMENT_TARGET."
fi

# ---------------------------------------------------------------------------------------------
# Stage into the package
# ---------------------------------------------------------------------------------------------
LIB="$OUT/lib"
mkdir -p "$LIB"
log "staging into $LIB"
cp "$BUILT" "$LIB/node_mlx.node"
if [[ "$MODE" == "prebuilt" ]]; then
  cp "$MLX_PREFIX/lib/libmlx.dylib"   "$LIB/libmlx.dylib"
  cp "$MLX_PREFIX/lib/libjaccl.dylib" "$LIB/libjaccl.dylib"
  cp "$MLX_PREFIX/lib/mlx.metallib"   "$LIB/mlx.metallib"
else
  # deps/mlx's own build output. The metallib lands next to the dylibs it produced.
  for f in libmlx.dylib libjaccl.dylib mlx.metallib; do
    src="$(find "$WORK/build" "$WORK/deps/mlx/build" -name "$f" -print -quit 2>/dev/null || true)"
    [[ -n "$src" ]] || die "source build produced no $f"
    cp "$src" "$LIB/$f"
  done
  dest_size="$(size_of "$LIB/mlx.metallib")"
  log "  source-built metallib: $dest_size bytes"
  if [[ "$dest_size" -lt "$METALLIB_MIN_BYTES" ]]; then
    warn "the source-built metallib is below the wheel-class floor. §5 mode 2 is only accepted if it"
    warn "is within ~5 % of mode 1 on bench/ — run 'npm run bench:check' before shipping this."
  fi
fi

# ---------------------------------------------------------------------------------------------
# rpath: rewrite to @loader_path (§5). The spike's addon carries the absolute venv path.
# ---------------------------------------------------------------------------------------------
log "rewriting rpaths to @loader_path"
rewrite_rpaths() {
  local file="$1"
  local rp changed=0
  while IFS= read -r rp; do
    [[ -z "$rp" ]] && continue
    case "$rp" in
      @loader_path*|@executable_path*) continue ;;
      *) install_name_tool -delete_rpath "$rp" "$file" && { log "  - $rp ($(basename "$file"))"; changed=1; } ;;
    esac
  done <<< "$(rpaths_of "$file")"

  if ! rpaths_of "$file" | grep -qx '@loader_path'; then
    install_name_tool -add_rpath @loader_path "$file" && { log "  + @loader_path ($(basename "$file"))"; changed=1; }
  fi

  if [[ "$changed" -eq 1 ]]; then
    # install_name_tool invalidates the (adhoc, linker-signed) signature, and arm64 macOS refuses to
    # load a binary with a broken signature. Re-sign adhoc.
    if command -v codesign >/dev/null 2>&1; then
      codesign --force --sign - "$file" >/dev/null 2>&1 \
        && log "  re-signed adhoc ($(basename "$file"))" \
        || warn "  codesign failed on $(basename "$file") — the addon may not load"
    fi
  fi
}
# Only the addon is touched. The wheel's libmlx.dylib / libjaccl.dylib already carry no absolute
# rpath: their install names and dependencies are @rpath-relative and resolve through the addon's
# LC_RPATH. Leaving them byte-identical to the wheel is what keeps §2's provenance hashes
# meaningful — SHA256SUMS then asserts "exactly the build that was benchmarked", not "a locally
# re-signed variant of it".
rewrite_rpaths "$LIB/node_mlx.node"

# ---------------------------------------------------------------------------------------------
# VERSION + SHA256SUMS
# ---------------------------------------------------------------------------------------------
PKG_VERSION="$(node -p "require('$OUT/package.json').version" 2>/dev/null || echo "0.0.0")"
BUILT_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

log "writing VERSION"
cat > "$OUT/VERSION" <<EOF
package=@fllstck/mlayax-darwin-arm64
version=$PKG_VERSION
node_mlx_commit=$NODE_MLX_COMMIT
mlx_tag=$MLX_TAG
mlx_build_class=$( [[ "$MODE" == "prebuilt" ]] && echo "$SIZE_CLASS_WHEEL" || echo "built-from-source" )
build_mode=$MODE
# The three MLX files are copied verbatim from the prebuilt MLX, so these are the §2 provenance
# hashes. The addon is compiled here and then rpath-rewritten + re-signed, hence two hashes: the
# pre-rpath one is the §2 reproduction signal, the plain one is what ships.
metallib_sha256=$(sha256_of "$LIB/mlx.metallib")
libmlx_sha256=$(sha256_of "$LIB/libmlx.dylib")
libjaccl_sha256=$(sha256_of "$LIB/libjaccl.dylib")
addon_sha256=$(sha256_of "$LIB/node_mlx.node")
addon_sha256_pre_rpath=$ADDON_SHA_PRE_RPATH
addon_bytes=$(size_of "$LIB/node_mlx.node")
metallib_bytes=$(size_of "$LIB/mlx.metallib")
rpath=@loader_path
# The floor is set by the MLX build linked in, not by our own build settings. §7 said "macOS >= 14";
# the wheel-class libmlx.dylib says minos 26.2, so that is what the README must state.
min_macos=$CMAKE_OSX_TARGET
mlx_libmlx_minos=${MLX_MINOS:-built-from-source}
s2_addon_sha256_reference=$ADDON_SHA256_REFERENCE
s2_addon_reproduced=$( [[ "$ADDON_SHA_PRE_RPATH" == "$ADDON_SHA256_REFERENCE" ]] \
  && echo yes \
  || echo "no (run with --deployment-target $SPIKE_DEPLOYMENT_TARGET to reproduce section 2 exactly)" )
built_at=$BUILT_AT
built_on=$(sw_vers -productVersion)/$(uname -m)
EOF

log "writing SHA256SUMS"
(cd "$LIB" && shasum -a 256 node_mlx.node libmlx.dylib libjaccl.dylib mlx.metallib > "$OUT/SHA256SUMS")

log "package size: $(du -sh "$OUT" | awk '{print $1}') ($(du -sk "$OUT" | awk '{print $1 * 1024}') bytes uncompressed)"
log "done"

# ---------------------------------------------------------------------------------------------
# Verify what we just staged — the same assertions --check runs, so a bad build fails here.
# ---------------------------------------------------------------------------------------------
check_payload "$OUT"