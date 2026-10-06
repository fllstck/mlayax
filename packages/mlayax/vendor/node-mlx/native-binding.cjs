"use strict";
/**
 * Resolve the native MLX addon for the vendored node-mlx core layer.
 *
 * This file is OURS, not upstream's — it exists because upstream's `dist/core.js` hardcodes
 * `require("../build/Release/node_mlx.node")`, which only works when the addon was compiled in
 * place by an install script. `core.cjs`'s one patched line calls into this instead.
 *
 * It is CommonJS rather than TypeScript because it is required synchronously from a `.cjs` file at
 * module load, before any ESM entry point has run. The typed public surface is `src/mlx/binding.ts`.
 *
 * The decision is split out as {@link selectBinding}, a pure function of
 * `(platform, arch, override, platformLibDir, exists)`, so every branch — including the ones that
 * cannot happen on an Apple Silicon machine — is unit-testable on any platform.
 */

const { existsSync } = require("node:fs");
const { createRequire } = require("node:module");
const path = require("node:path");

const PLATFORM_PACKAGE = "@fllstck/mlayax-darwin-arm64";
const ADDON = "node_mlx.node";
const NATIVE_DIR_ENV = "MLAYAX_NATIVE_DIR";

const PLATFORM_HINT =
  "There is no CPU, WebGPU or Linux backend in 0.1.0. On an Apple Silicon machine, check that you " +
  "are not running Node under Rosetta (`node -p process.arch` should print arm64).";

/**
 * Pick the addon to load, or explain why there is none.
 *
 * Pure: no filesystem or environment access. `exists` is injected so the "tried these paths" message
 * can be asserted without creating directories.
 *
 * @param {{platform: string, arch: string, override: string|undefined, platformLibDir: string|null,
 *          exists: (file: string) => boolean}} input
 * @returns {string} absolute path to `node_mlx.node`
 */
function selectBinding(input) {
  const { platform, arch, override, platformLibDir, exists } = input;

  if (platform !== "darwin" || arch !== "arm64") {
    throw new Error(
      `@fllstck/mlayax requires Apple Silicon (darwin/arm64); this process is ${platform}/${arch}.\n` +
        PLATFORM_HINT,
    );
  }

  /** Documentation order: an explicit override wins, then the platform package. */
  const candidates = [];
  if (override) candidates.push({ dir: override, source: NATIVE_DIR_ENV });
  if (platformLibDir) candidates.push({ dir: platformLibDir, source: `${PLATFORM_PACKAGE}/lib` });

  for (const candidate of candidates) {
    const file = path.join(candidate.dir, ADDON);
    if (exists(file)) return file;
  }

  const tried = candidates.map((c) => `  - ${path.join(c.dir, ADDON)}  (from ${c.source})`);
  throw new Error(
    `Could not find ${ADDON} for ${platform}/${arch}.\n` +
      (tried.length > 0 ? `Tried:\n${tried.join("\n")}\n` : "No candidate locations were available.\n") +
      `Install ${PLATFORM_PACKAGE} (npm does it automatically as an optional dependency on Apple ` +
      `Silicon), or set ${NATIVE_DIR_ENV} to a directory containing a built addon.`,
  );
}

/** Memoised: `require` caches the addon anyway, but this keeps the error path from re-probing. */
let resolved = null;

function platformPackageLib() {
  // Resolve from this file's location so hoisted and nested node_modules layouts both work.
  const require_ = createRequire(__filename);
  let packageJson;
  try {
    packageJson = require_.resolve(`${PLATFORM_PACKAGE}/package.json`);
  } catch {
    return null;
  }
  return path.join(path.dirname(packageJson), "lib");
}

/** Resolve the addon path for this process, memoised. */
function resolveNativeBinding() {
  if (resolved !== null) return resolved;
  resolved = selectBinding({
    platform: process.platform,
    arch: process.arch,
    override: process.env[NATIVE_DIR_ENV],
    platformLibDir: platformPackageLib(),
    exists: existsSync,
  });
  return resolved;
}

module.exports = { resolveNativeBinding, selectBinding, NATIVE_DIR_ENV, ADDON, PLATFORM_PACKAGE };