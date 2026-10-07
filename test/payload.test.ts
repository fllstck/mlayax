/**
 * The shipped native payload — TASKS.md §8.7.
 *
 * "**MLX build quality is a first-class requirement, not a detail.**" The compressed Metallib size is
 * the observable that predicts throughput: §2 measured the wheel-class build at 190 MB and Homebrew's
 * at 137 MB for the *same MLX version*, and the slow one was **35 % slower** on this workload. A
 * payload built against the wrong MLX therefore passes every correctness test while quietly costing a
 * third of the performance — and performance is the entire reason this project vendors a compiled
 * binding instead of using an FFI one (§0).
 *
 * So this file asserts the build class from the bytes, not from a claim about them.
 *
 * Scope note: this is the half of payload verification that is pure file reads — sizes, hashes, and
 * `VERSION`/`SHA256SUMS` consistency. The binary-level checks (rpath is `@loader_path`-relative, the
 * macOS floor, the fused symbols) live in `tools/native/build.sh --check`, which
 * `npm run verify:release` runs. They are complements, not duplicates: that one knows how to read a
 * Mach-O header, this one knows what §2 pinned.
 *
 * The hashes below are written out rather than imported from `tools/native/build.sh`. That is
 * deliberate — a test that reads its expectations from the thing it is testing cannot catch that
 * thing being edited. These are §2's numbers, transcribed.
 */

import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { resolveNativeAddonPath } from "../packages/mlayax/src/mlx/index.js";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const PAYLOAD_DIR = path.join(repoRoot, "packages", "mlayax-darwin-arm64");
const LIB_DIR = path.join(PAYLOAD_DIR, "lib");

/** §2's pinned provenance. The wheel-class MLX, byte-for-byte. */
const WHEEL_CLASS = {
  "mlx.metallib": {
    sha256: "95974b7de464ec2be7833a914ca5a53d283a70789470fd84218738fc8fcea3e1",
    bytes: 190_319_536,
    why: "the compiled Metal kernels — the size class that predicts throughput",
  },
  "libmlx.dylib": {
    sha256: "ebc8a5f5465afb79d3d9c8f5b0929de1a9b277b7179db8672f92da77ae110a1e",
    bytes: 21_108_048,
    why: "MLX core; copied verbatim so the hash still means 'the build that was measured'",
  },
  "libjaccl.dylib": {
    sha256: "949e901df832d600a71ff2bc5f3738d55b010668102f36e071e1890807a965cd",
    bytes: 1_555_552,
    why: "the collective-comms library libmlx links against",
  },
} as const;

/** The addon is compiled per-machine (rpath rewrite + re-sign), so it has no fixed hash to pin. */
const ADDON = "node_mlx.node";

/** Any metallib below this is the slow class: Homebrew's bottle is ~137 MB, `@johnhenry`'s ~135 MB. */
const WHEEL_CLASS_FLOOR_BYTES = 180 * 1000 * 1000;

function payloadPresent(): boolean {
  try {
    resolveNativeAddonPath();
    return existsSync(path.join(LIB_DIR, "mlx.metallib"));
  } catch {
    return false;
  }
}

const sha256 = (file: string): string =>
  createHash("sha256").update(readFileSync(file)).digest("hex");

/** `VERSION` is `key=value` with `#` comments; the build script writes both. */
function readVersion(): Record<string, string> {
  const fields: Record<string, string> = {};
  for (const line of readFileSync(path.join(PAYLOAD_DIR, "VERSION"), "utf8").split("\n")) {
    if (line === "" || line.startsWith("#")) continue;
    const at = line.indexOf("=");
    if (at > 0) fields[line.slice(0, at)] = line.slice(at + 1);
  }
  return fields;
}

/** `SHA256SUMS`, in the `shasum -a 256` format the build script writes. */
function readSums(): Record<string, string> {
  const sums: Record<string, string> = {};
  for (const line of readFileSync(path.join(PAYLOAD_DIR, "SHA256SUMS"), "utf8")
    .trim()
    .split("\n")) {
    const [hash, name] = line.trim().split(/\s+/);
    if (hash !== undefined && name !== undefined) sums[name] = hash;
  }
  return sums;
}

describe.skipIf(!payloadPresent())("the shipped MLX payload (§8.7)", () => {
  it("is the wheel-class build, not a slower one", () => {
    const metallib = path.join(LIB_DIR, "mlx.metallib");
    const bytes = readFileSync(metallib).byteLength;

    // Size first, because it is the cheap discriminator and the one §2's argument rests on.
    expect(bytes).toBeGreaterThanOrEqual(WHEEL_CLASS_FLOOR_BYTES);
    expect(bytes).toBe(WHEEL_CLASS["mlx.metallib"].bytes);
    // Then the hash, because a same-sized rebuild is still a different build.
    expect(sha256(metallib)).toBe(WHEEL_CLASS["mlx.metallib"].sha256);
  });

  it("keeps the MLX dylibs byte-identical to the wheel, so §2's hashes still mean something", () => {
    // `build.sh` only rewrites the addon's rpath. If these two ever stop matching, `SHA256SUMS`
    // stops being a statement about upstream provenance and becomes a statement about our local
    // surgery — which is worth less.
    for (const [name, expected] of Object.entries(WHEEL_CLASS)) {
      if (name === "mlx.metallib") continue;
      const file = path.join(LIB_DIR, name);
      expect(readFileSync(file).byteLength, `${name} size`).toBe(expected.bytes);
      expect(sha256(file), `${name} sha256`).toBe(expected.sha256);
    }
  });

  it("writes down the build class it actually shipped", () => {
    // Catches the specific mistake of regenerating the payload against a different MLX and leaving
    // the declared class alone. The claim and the bytes have to agree in both directions.
    const version = readVersion();
    const metallibSha = sha256(path.join(LIB_DIR, "mlx.metallib"));
    if (metallibSha === WHEEL_CLASS["mlx.metallib"].sha256) {
      expect(version.mlx_build_class).toBe("wheel-class");
      expect(version.build_mode).toBe("prebuilt");
    } else {
      expect(version.mlx_build_class).not.toBe("wheel-class");
    }
  });

  it("VERSION describes the bytes on disk, not an earlier build", () => {
    // The failure this prevents: someone rebuilds, VERSION is only written on the success path, and a
    // stale VERSION ships alongside a new binary. Every hash in it is checked against the file.
    const version = readVersion();
    expect(version.node_mlx_commit).toBe("4bf8b1d6de32ae5ec402525bff1f041d73618275");
    expect(version.mlx_tag).toBe("v0.32.3");
    expect(version.rpath).toBe("@loader_path");

    for (const [name, key] of [
      ["mlx.metallib", "metallib_sha256"],
      ["libmlx.dylib", "libmlx_sha256"],
      ["libjaccl.dylib", "libjaccl_sha256"],
    ] as const) {
      expect(version[key], `VERSION ${key}`).toBe(sha256(path.join(LIB_DIR, name)));
    }
    expect(Number(version.metallib_bytes)).toBe(
      readFileSync(path.join(LIB_DIR, "mlx.metallib")).byteLength,
    );
    expect(Number(version.addon_bytes)).toBe(readFileSync(path.join(LIB_DIR, ADDON)).byteLength);
    expect(version.addon_sha256).toBe(sha256(path.join(LIB_DIR, ADDON)));
  });

  it("declares a macOS floor that matches what the linked MLX needs", () => {
    // §7 used to say "macOS >= 14"; the wheel-class libmlx.dylib declares minos 26.2, so 14 was a
    // promise the artifact could not keep (TASKS.md §10.2). `build.sh --check` reads that floor off
    // the binary; here we assert the declaration is internally consistent and is not the old number.
    const version = readVersion();
    expect(version.min_macos).toBe(version.mlx_libmlx_minos);
    expect(version.min_macos).not.toBe("14");
    expect(Number.parseFloat(version.min_macos ?? "")).toBeGreaterThanOrEqual(26);
  });

  it("SHA256SUMS covers every shipped file, and every entry verifies", () => {
    const sums = readSums();
    const shipped = Object.keys(sums).sort();
    // The four files §3's layout promises, and nothing else — a missing entry is as much a failure as
    // a wrong one, since the file is what a user is told to check against.
    expect(shipped).toEqual(["libjaccl.dylib", "libmlx.dylib", "mlx.metallib", ADDON]);

    for (const [name, expected] of Object.entries(sums)) {
      expect(sha256(path.join(LIB_DIR, name)), name).toBe(expected);
    }
  });

  it("agrees with the addon the runtime actually resolves", () => {
    // Ties the payload to the loader: if `resolveNativeAddonPath` and this file disagreed about where
    // the addon lives, every other assertion here could pass while the runtime loaded something else.
    expect(path.dirname(resolveNativeAddonPath())).toBe(LIB_DIR);
  });
});
