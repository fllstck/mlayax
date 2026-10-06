/**
 * Loading the native MLX addon — the one place the process touches native code.
 *
 * Two properties matter here, and both are load-time requirements rather than nice-to-haves:
 *
 * 1. **Lazy.** Nothing loads the addon until `loadMx()` is called. `import "@fllstck/mlayax"` must
 *    work on a Linux CI runner so the pure-TypeScript `core` layer can be unit-tested there; a
 *    top-level `require` would make the package unimportable off Apple Silicon.
 * 2. **Fail fast and specifically.** A wrong platform, a missing payload, or a payload built for a
 *    different MLX must say so at load, not surface later as a shape error or a symbol mismatch.
 *
 * Not handled here, because it can only be tested once the platform package exists (phase 5): the
 * mixing guard for a second `libmlx` already resident in the process, and the verification of
 * `SHA256SUMS`/`VERSION` against the shipped payload.
 */

import { createRequire } from "node:module";
import type { MlxCore, MlxModule } from "./types.js";

const require = createRequire(import.meta.url);

/** Where the vendored core layer lives, relative to this module (works from `src/` and `dist/`). */
const VENDORED_CORE = "../../vendor/node-mlx/core.cjs";

let cached: MlxModule | null = null;

/**
 * Load the native MLX core, memoised.
 *
 * The addon itself is cached by Node's module loader, so repeated calls are cheap; this memo exists
 * so the resolution and the platform guard run once.
 */
export function loadMxModule(): MlxModule {
  if (cached !== null) return cached;
  const loaded = require(VENDORED_CORE) as MlxModule;
  if (typeof loaded?.core !== "object" || loaded.core === null) {
    throw new Error(
      `The vendored MLX core layer at ${VENDORED_CORE} loaded but did not expose \`core\`. ` +
        "This means the package is corrupt or partially installed.",
    );
  }
  cached = loaded;
  return cached;
}

/** The MLX core namespace. Prefer this over {@link loadMxModule} unless you need the whole module. */
export function loadMx(): MlxCore {
  return loadMxModule().core;
}

/** True when `loadMx()` has already run, without triggering a load. */
export function isMxLoaded(): boolean {
  return cached !== null;
}

/** Drop the memo (the addon stays loaded — Node never unloads a native module). For tests only. */
export function resetMxCacheForTests(): void {
  cached = null;
}

/**
 * The path of the addon this process would load.
 *
 * Exposed for diagnostics and for the load-time tests, which need to assert the resolution order
 * without performing a load. Throws the same described errors as a real load.
 */
export function resolveNativeAddonPath(): string {
  const { resolveNativeBinding } = require("../../vendor/node-mlx/native-binding.cjs") as {
    resolveNativeBinding: () => string;
  };
  return resolveNativeBinding();
}
