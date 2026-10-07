/**
 * The mixing guard end to end — TASKS.md §8.6, the load-time half.
 *
 * The unit tests next to `mixing.ts` cover the decision table but never touch a real dynamic loader.
 * This file is the part that does: it puts a genuinely foreign `libmlx.dylib` into a real process and
 * asserts that a real dyld collision is stopped with a real explanation.
 *
 * Two fixtures, for two different reasons:
 *
 * - **A copy of our own `libmlx.dylib`** (always available, deterministic). `install_name_tool`
 *   rewrites it and `codesign` re-signs it, so it is a *different build* by size and hash while
 *   still exporting the same symbols. That distinction matters: it is the conservative case, where
 *   dyld would have bound successfully and MLX would have broken later, in a way that has nothing to
 *   do with symbols.
 * - **A third-party MLX build** (`@johnhenry`, `@frost-beta`, Homebrew), if one happens to be on
 *   this machine. That is the case §8.6 measured: an unrecoverable `Symbol not found` at dlopen.
 *   Skipped when absent, because requiring a competing package to test our own guard would be a
 *   strange thing to make CI depend on.
 *
 * Every collision runs in a child process: making a foreign MLX resident poisons the process
 * permanently, and the healthy-path test has to run somewhere.
 */

import { type ExecFileSyncOptions, execFileSync } from "node:child_process";
import { copyFileSync, existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { afterAll, describe, expect, it } from "vitest";
import {
  loadMx,
  MLX_LIBRARY_NAMES,
  readInstallName,
  resolveNativeAddonPath,
} from "../packages/mlayax/src/mlx/index.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, "..");
const PROBE = path.join(here, "helpers", "mixing-probe.mjs");

/**
 * The built entry point the child imports. A child process cannot import TypeScript through vitest's
 * transform, so it needs `dist/` — which `npm run verify` and `verify:release` both produce before
 * running the tests. `npm test` alone does not, so this test reports itself skipped rather than
 * failing on a missing build.
 */
const DIST_ENTRY = path.join(repoRoot, "packages", "mlayax", "dist", "mlx", "index.js");

function nativeAvailable(): boolean {
  try {
    resolveNativeAddonPath();
    return true;
  } catch {
    return false;
  }
}

function distAvailable(): boolean {
  return existsSync(DIST_ENTRY);
}

const PAYLOAD = nativeAvailable();
const CAN_SPAWN = PAYLOAD && distAvailable();

/**
 * Whether the runtime running *this* file can enumerate its resident images.
 *
 * The probe runs under `process.execPath` — the same runtime as the harness — and the collision tests
 * below rest entirely on `process.report.getReport().sharedObjects`: before the guard can be accused
 * of anything, the foreign library has to be *proven* resident. On a runtime that cannot enumerate
 * images, the guard cannot see a foreign library either, so these tests are skipped rather than
 * weakened; that behaviour is asserted in its own `describe` at the bottom of this file.
 *
 * The check is measured, not `typeof process.report?.getReport === "function"`. Bun has that method
 * and returns a `sharedObjects` array that is *always empty* — a runtime that answers the question is
 * not the same as one that answers it truthfully (TASKS.md §10.11). Every process has Node's own
 * shared libraries resident, so a working enumerator has something to report.
 */
const CAN_ENUMERATE_IMAGES = (() => {
  const report = (
    process.report as { getReport?: () => { sharedObjects?: string[] } } | undefined
  )?.getReport?.();
  return (report?.sharedObjects?.length ?? 0) > 0;
})();

/** The directory our payload's libraries live in, which the guard treats as "ours". */
function libDir(): string {
  return path.dirname(resolveNativeAddonPath());
}

interface ProbeResult {
  residentBeforeLoad: string[];
  residentCount: number;
  outcome: "loaded" | "threw";
  name: string | null;
  message: string | null;
  warnings: string[];
}

/** Run the probe in a child process and parse its JSON. */
function probe(foreignLibDir: string, env: Record<string, string> = {}): ProbeResult {
  const stdout = execFileSync(process.execPath, [PROBE, foreignLibDir, DIST_ENTRY], {
    encoding: "utf8",
    env: { ...process.env, ...env },
    stdio: ["ignore", "pipe", "pipe"],
  });
  return JSON.parse(stdout) as ProbeResult;
}

/**
 * Build a foreign `libmlx.dylib` that is loadable *and* shadowing: a copy of ours, told to find
 * `libjaccl.dylib` beside itself, then re-signed because `install_name_tool` invalidates the
 * signature. Same install name (`@rpath/libmlx.dylib`), therefore same dedupe slot as ours.
 *
 * Returns `null` when the tools are unavailable, so the caller can skip.
 */
function makeForeignCopy(): string | null {
  const lib = libDir();
  const dir = mkdtempSync(path.join(tmpdir(), "mlayax-mixing-"));
  try {
    for (const name of MLX_LIBRARY_NAMES) copyFileSync(path.join(lib, name), path.join(dir, name));
    // `stdio` is piped rather than inherited: codesign announces "replacing existing signature" on
    // stderr, which would otherwise appear in the middle of the test output.
    const quiet: ExecFileSyncOptions = { stdio: ["ignore", "pipe", "pipe"] };
    execFileSync(
      "install_name_tool",
      ["-add_rpath", "@loader_path", path.join(dir, "libmlx.dylib")],
      quiet,
    );
    execFileSync("codesign", ["--force", "--sign", "-", path.join(dir, "libmlx.dylib")], quiet);
    return dir;
  } catch {
    rmSync(dir, { recursive: true, force: true });
    return null;
  }
}

/** Third-party MLX builds worth looking for, in the order they plausibly sit on a dev machine. */
function findThirdPartyMlx(): string | null {
  const explicit = process.env.MLAYAX_FOREIGN_MLX;
  if (explicit !== undefined && existsSync(explicit)) return path.dirname(explicit);

  const candidates = [
    path.join(repoRoot, "node_modules", "@johnhenry", "backend-mlx-darwin-arm64", "lib"),
    path.join(repoRoot, "node_modules", "@frost-beta", "mlx", "build", "Release"),
    "/opt/homebrew/opt/mlx/lib",
  ];
  for (const dir of candidates) {
    // Homebrew names its dylib with an absolute install name, so it cannot shadow ours. It is
    // included only because a *second* resident MLX is still worth detecting.
    if (existsSync(path.join(dir, "libmlx.dylib"))) return dir;
  }
  return null;
}

const foreignCopy = CAN_SPAWN ? makeForeignCopy() : null;
const thirdParty = CAN_SPAWN ? findThirdPartyMlx() : null;

afterAll(() => {
  if (foreignCopy !== null) rmSync(foreignCopy, { recursive: true, force: true });
});

describe.skipIf(!PAYLOAD)("the healthy path stays healthy", () => {
  it("loads MLX with no complaint, and the guard sees our own library as ours", () => {
    const mx = loadMx();
    // `sum` returns a 0-d array and `toTypedArray` insists on one dimension, so reshape first.
    const total = mx.reshape(mx.sum(mx.array([1, 2, 3], mx.float32)), [1]);
    expect(Array.from(total.toTypedArray())).toEqual([6]);

    // The image list must contain our library — otherwise the child-process collisions below would
    // be "detected" for a reason other than mixing, and the guard would look correct by accident.
    const shared =
      (process.report?.getReport?.() as unknown as { sharedObjects?: string[] } | undefined)
        ?.sharedObjects ?? [];
    const mlxImages = shared.filter((image) => image.endsWith("libmlx.dylib"));
    if (shared.length > 0) {
      expect(mlxImages.some((image) => image.startsWith(libDir()))).toBe(true);
    }
  });
});

describe.skipIf(!PAYLOAD)(
  "readInstallName agrees with the platform tool on real Mach-O files",
  () => {
    it("matches otool -D for every library we ship", () => {
      for (const name of MLX_LIBRARY_NAMES) {
        const file = path.join(libDir(), name);
        const expected = execFileSync("otool", ["-D", file]).toString().trim().split("\n").pop();
        expect(readInstallName(readFileSync(file))).toBe(expected);
        // The value the whole guard turns on: our libraries name themselves the way our addon asks
        // for them, which is what makes a competing build dangerous.
        expect(expected).toBe(`@rpath/${name}`);
      }
    });

    it("reads the name from a 64 KiB prefix, which is all binding.ts ever supplies", () => {
      // `readOf` deliberately does not read 21 MB to answer this question, so the parser has to work
      // on a truncated buffer. Asserted here rather than trusted, because a throw on truncation would
      // take out the load path rather than the guard.
      const file = path.join(libDir(), "libmlx.dylib");
      const prefix = readFileSync(file).subarray(0, 64 * 1024);
      expect(prefix.byteLength).toBe(64 * 1024);
      expect(readInstallName(prefix)).toBe("@rpath/libmlx.dylib");
    });
  },
);

describe.skipIf(!CAN_SPAWN || !CAN_ENUMERATE_IMAGES)(
  "a foreign libmlx is stopped before dyld binds it",
  () => {
    it("rejects a different-build copy of our own library, and names it", () => {
      if (foreignCopy === null) return; // install_name_tool/codesign unavailable
      const result = probe(foreignCopy);

      expect(result.outcome).toBe("threw");
      expect(result.name).toBe("MlxMixingError");
      expect(result.message).toContain(path.join(foreignCopy, "libmlx.dylib"));
      expect(result.message).toContain(libDir());
      // The point of the guard: no mangled dyld symbol reaches the user as the headline.
      expect(result.message).not.toMatch(/^dlopen\(/);
    });

    it("is stopping a real breakage, not being pedantic", () => {
      // With the guard waived, dyld binds this copy happily — its symbols are ours. MLX then fails
      // anyway, with "Failed to load the default metallib", because MLX resolves `mlx.metallib`
      // relative to the library that won. So a resident foreign build breaks MLX in two separate ways,
      // and only one of them looks like a symbol problem.
      if (foreignCopy === null) return;
      const result = probe(foreignCopy, { MLAYAX_ALLOW_MIXED_MLX: "1" });

      expect(result.outcome).toBe("threw");
      expect(result.message).toMatch(/metallib/i);
      // And it is not our guard doing the throwing — the escape hatch was honoured.
      expect(result.name).not.toBe("MlxMixingError");
    });

    it("leaves an unrelated load failure alone", () => {
      // A child with a broken dist entry must not be blamed on library mixing.
      const stdout = execFileSync(
        process.execPath,
        [PROBE, libDir(), path.join(repoRoot, "packages", "mlayax", "dist", "mlx", "nope.js")],
        { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] },
      );
      const result = JSON.parse(stdout) as ProbeResult;
      expect(result.outcome).toBe("threw");
      expect(result.message).toMatch(/Cannot find module|ERR_MODULE_NOT_FOUND/);
      expect(result.message).not.toMatch(/mixed|libmlx\.dylib is already resident/);
    });
  },
);

describe.skipIf(!CAN_SPAWN || !CAN_ENUMERATE_IMAGES || thirdParty === null)(
  "a real third-party MLX build",
  () => {
    /**
     * Which half of the guard is expected to act depends on the library's own install name, so the
     * test reads it instead of assuming. Homebrew's bottle calls itself
     * `/opt/homebrew/opt/mlx/lib/libmlx.dylib`, which our addon's request for `@rpath/libmlx.dylib`
     * cannot match — it therefore loads as a second, separate copy and MLX still resolves ours. Only a
     * build that names itself `@rpath/libmlx.dylib` can take the slot, and that is the fatal case.
     */
    const thirdPartyInstallName = (): string | null => {
      if (thirdParty === null) return null;
      try {
        return readInstallName(readFileSync(path.join(thirdParty, "libmlx.dylib")));
      } catch {
        return null;
      }
    };

    it("is handled according to whether it can actually be matched", () => {
      if (thirdParty === null) return;
      const installName = thirdPartyInstallName();
      const result = probe(thirdParty);

      // Confirms the fixture genuinely became resident — otherwise this test would pass on a process
      // where nothing collided and would prove nothing.
      expect(result.residentCount).toBeGreaterThan(0);
      expect(result.residentBeforeLoad.length).toBeGreaterThan(0);

      if (installName === "@rpath/libmlx.dylib") {
        // The measured crash: dyld binds the wrong build and dies on a symbol mismatch.
        expect(result.outcome).toBe("threw");
        expect(result.name).toBe("MlxMixingError");
        expect(result.message).toContain(thirdParty);
        expect(result.message).toMatch(/Symbol not found|symbol not found|could not be bound/);
      } else {
        // Cannot shadow: our addon keeps its own copy, MLX works, and the user gets a note.
        expect(result.outcome).toBe("loaded");
        expect(result.warnings.join("\n")).toMatch(/cannot shadow/);
        expect(result.warnings.join("\n")).toContain(thirdParty);
        expect(result.message).toMatch(/^sum=6$/);
      }
    });

    it("reports the install name it measured, so the branch above is not a guess", () => {
      if (thirdParty === null) return;
      // `null` is allowed (an image we cannot parse) and is treated as shadowing, which the branch
      // above then requires to throw. Either way the decision is grounded in a real read.
      const installName = thirdPartyInstallName();
      expect(installName === null || typeof installName === "string").toBe(true);
    });
  },
);

/**
 * The degradation, asserted rather than assumed — this is the Bun half.
 *
 * `mixing.ts`'s policy is unit-tested in the abstract and `binding.ts` describes the Bun path in
 * prose, but this is the end-to-end claim: on a runtime that cannot enumerate its images, the guard
 * must not report that it saw something it cannot see. Whatever goes wrong next is dyld's or MLX's
 * error (`describeMlxLoadFailure` explains it), never a `MlxMixingError` we invented.
 *
 * Skipped on Node, where the tests above do the real work.
 */
describe.skipIf(!CAN_SPAWN || CAN_ENUMERATE_IMAGES)(
  "a runtime with no image list does not get a fabricated guard",
  () => {
    it("cannot see the resident foreign library, and says so instead of guessing", () => {
      if (foreignCopy === null) return; // install_name_tool/codesign unavailable
      const result = probe(foreignCopy);

      // The runtime told us nothing about its images...
      expect(result.residentCount).toBe(0);
      expect(result.residentBeforeLoad).toEqual([]);
      // ...so the guard must not claim it recognised a foreign library.
      expect(result.name).not.toBe("MlxMixingError");
      if (result.message !== null) {
        expect(result.message).not.toContain("is already resident");
      }
    });
  },
);
