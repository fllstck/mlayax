/**
 * Loading the native MLX addon — the one place the process touches native code.
 *
 * Three properties matter here, and all three are load-time requirements rather than nice-to-haves:
 *
 * 1. **Lazy.** Nothing loads the addon until `loadMx()` is called. `import "@fllstck/mlayax"` must
 *    work on a Linux CI runner so the pure-TypeScript `core` layer can be unit-tested there; a
 *    top-level `require` would make the package unimportable off Apple Silicon.
 * 2. **Fail fast and specifically.** A wrong platform, a missing payload, a payload built for a
 *    different MLX, or a second `libmlx` already resident must say so at load, not surface later as
 *    a shape error or a mangled dyld symbol.
 * 3. **Order matters.** The mixing guard has to run *before* the addon is loaded: once dyld has
 *    bound the wrong `libmlx`, the process cannot recover. See {@link ./mixing.js}.
 */

import { createRequire } from "node:module";
import path from "node:path";
import {
  ALLOW_MIXED_ENV,
  bytesOf,
  checkForMixedMlx,
  describeMlxLoadFailure,
  MLX_LIBRARY_NAMES,
  readOf,
  realpathOf,
  residentSharedObjects,
  sha256Of,
} from "./mixing.js";
import type { MlxCore, MlxModule } from "./types.js";

const require = createRequire(import.meta.url);

/** Where the vendored core layer lives, relative to this module (works from `src/` and `dist/`). */
const VENDORED_CORE = "../../vendor/node-mlx/core.cjs";

let cached: MlxModule | null = null;

/**
 * Check for a resident MLX build before letting dyld bind ours.
 *
 * Returns early on runtimes that do not expose the image list — the load-time error translation is
 * the only defence available there, and it is not needed when the list is empty. The policy itself
 * (`checkForMixedMlx`) and every filesystem read it needs live in `mixing.ts`, where they are
 * unit-tested against real files; this function only supplies the process this is running in.
 */
function guardAgainstMixedMlx(libDir: string): void {
  const resident = residentSharedObjects();
  if (resident.length === 0) return;

  const ours: Record<string, string> = {};
  for (const name of MLX_LIBRARY_NAMES) ours[name] = path.join(libDir, name);

  checkForMixedMlx({
    libDir,
    resident,
    ours,
    bytesOf,
    sha256Of,
    readOf,
    realpathOf,
    allowMixed: process.env[ALLOW_MIXED_ENV] === "1",
    warn: (message) => process.emitWarning(message, "RuntimeWarning"),
  });
}

/**
 * Load the native MLX core, memoised.
 *
 * The addon itself is cached by Node's module loader, so repeated calls are cheap; this memo exists
 * so the resolution and the guards run once.
 */
export function loadMxModule(): MlxModule {
  if (cached !== null) return cached;

  // Resolve first: this is what throws the described "wrong platform / missing payload" error, and
  // it is also the directory the mixing guard compares against.
  const libDir = path.dirname(resolveNativeAddonPath());
  guardAgainstMixedMlx(libDir);

  let loaded: MlxModule;
  try {
    loaded = require(VENDORED_CORE) as MlxModule;
  } catch (error) {
    // Reached on Bun (no image list to check) and on Node if a foreign library appeared in between.
    throw describeMlxLoadFailure(error, libDir);
  }

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
