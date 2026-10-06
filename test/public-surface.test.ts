/**
 * Guards on the *published* surface, checked against the built `dist/`.
 *
 * This is a library, so its API is the product — which makes three properties worth asserting rather
 * than promising:
 *
 * 1. **No `any` escapes.** A shipped `.d.ts` with `any` in it silently disables strict checking for
 *    every consumer, and it is exactly the kind of thing that reappears during a later refactor.
 * 2. **No service API.** Phase 4 was dropped: this is imported from TypeScript, not deployed as an
 *    HTTP server. Nothing named `server`, `http`, `batcher`, or `metrics` belongs in the exports.
 * 3. **Importing the package loads no native code.** `import { load } from "@fllstck/mlayax"` has to
 *    work on a Linux CI runner with no Apple Silicon payload, so the addon must not be touched until
 *    a load function is called.
 *
 * Skips when `dist/` is absent, so running `vitest run` before a build does not produce a false
 * failure. `npm run verify` builds first, so CI always exercises it.
 */

import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { describe, expect, it } from "vitest";

const PACKAGE_ROOT = fileURLToPath(new URL("../packages/mlayax", import.meta.url));
const DIST = join(PACKAGE_ROOT, "dist");

function distributionExists(): boolean {
  try {
    return statSync(join(DIST, "index.js")).isFile();
  } catch {
    return false;
  }
}

/** Every `.d.ts` under `dist/`, recursively. */
function declarationFiles(dir = DIST): string[] {
  const found: string[] = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) found.push(...declarationFiles(full));
    else if (entry.name.endsWith(".d.ts")) found.push(full);
  }
  return found;
}

describe.skipIf(!distributionExists())("published surface (built dist/)", () => {
  it("ships no `any` in its public types", () => {
    const offenders: string[] = [];
    for (const file of declarationFiles()) {
      const text = readFileSync(file, "utf8");
      text.split("\n").forEach((line, index) => {
        // `: any`, `<any>`, `any[]`, `any>` — but not a word like `anything` or a `.d.ts.map` path.
        if (/(?::\s*any\b|<any>|any\[\]|\bany\s*>)/.test(line)) {
          offenders.push(`${file.slice(PACKAGE_ROOT.length + 1)}:${index + 1}: ${line.trim()}`);
        }
      });
    }
    expect(offenders, `\`any\` leaked into the published types:\n${offenders.join("\n")}`).toEqual(
      [],
    );
  });

  it("exposes no service API", () => {
    // Enforces the Phase 4 decision. A server or a metrics endpoint reappearing here would mean the
    // scope crept back without the decision being revisited.
    const banned = /\b(server|http|batcher|metrics|listen|endpoint)\b/i;
    const exported = new Set<string>();
    for (const file of declarationFiles()) {
      for (const match of readFileSync(file, "utf8").matchAll(
        /^export (?:declare )?(?:abstract )?(?:class|function|const|interface|type|enum) (\w+)/gm,
      )) {
        const name = match[1];
        if (name !== undefined) exported.add(name);
      }
    }
    expect(exported.size).toBeGreaterThan(20);
    // `endpoint` is allowed: the Hub fetcher legitimately takes a Hub base URL.
    const offending = [...exported].filter((name) => banned.test(name) && name !== "endpoint");
    expect(offending, `service-shaped exports found: ${offending.join(", ")}`).toEqual([]);
  });

  it("does not load native code merely by being imported", async () => {
    // The guarantee that lets the `core` half be unit-tested on Linux CI. `isMxLoaded()` reports
    // whether the addon has been required; importing everything must leave it false.
    const built = (await import(pathToFileURL(join(DIST, "index.js")).href)) as {
      isMxLoaded: () => boolean;
      VERSION: string;
    };
    expect(built.isMxLoaded()).toBe(false);
    expect(built.VERSION).toBe("0.1.0");
  });

  it("resolves a native addon path without loading it", async () => {
    const built = (await import(pathToFileURL(join(DIST, "index.js")).href)) as {
      resolveNativeAddonPath: () => string;
      isMxLoaded: () => boolean;
    };
    try {
      expect(built.resolveNativeAddonPath().endsWith("node_mlx.node")).toBe(true);
    } catch (error) {
      // No payload on this machine (Linux CI, or before phase 5). The message must still be the
      // described one — a bare ENOENT here would be the failure mode this asserts against.
      expect((error as Error).message).toMatch(
        /Could not find node_mlx\.node|requires Apple Silicon/,
      );
    }
    expect(built.isMxLoaded()).toBe(false);
  });
});
