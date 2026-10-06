/**
 * The mixing guard's decision table — TASKS.md §8.6.
 *
 * Every filesystem read is injected, so this runs on Linux CI with no payload and no Apple Silicon.
 * That matters more here than usual: the integration half of this hazard (a real foreign `libmlx`
 * resident in a real process) can only be reproduced on a Mac with a second MLX package installed,
 * so it is the *only* place these branches get exercised on most machines.
 *
 * The branches are not hypothetical. Each one is a state this machine reached while the guard was
 * being written: `@johnhenry/backend-mlx-darwin-arm64`'s `libmlx.dylib` was the fixture that
 * reproduced the crash, and a stray copy of our own library reproduced the false-positive risk.
 */

import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import {
  ALLOW_MIXED_ENV,
  assertNoMixedMlx,
  bytesOf,
  checkForMixedMlx,
  describeMlxLoadFailure,
  HEADER_PREFIX_BYTES,
  hasMixingReport,
  inspectResidentMlx,
  MLX_LIBRARY_NAMES,
  MlxMixingError,
  mixingError,
  mixingWarning,
  readInstallName,
  readOf,
  realpathOf,
  residentSharedObjects,
  sha256Of,
} from "./mixing.js";

const LIB_DIR = "/pkg/mlayax-darwin-arm64/lib";
const OURS = {
  "libmlx.dylib": path.join(LIB_DIR, "libmlx.dylib"),
  "libjaccl.dylib": path.join(LIB_DIR, "libjaccl.dylib"),
};

/** A fake filesystem: path → size, plus sha256 and raw bytes for the paths that need them. */
function fs(
  sizes: Record<string, number>,
  hashes: Record<string, string> = {},
  installNames: Record<string, string> = {},
) {
  return {
    ours: OURS,
    bytesOf: (file: string): number | null => sizes[file] ?? null,
    sha256Of: (file: string): string | null => hashes[file] ?? null,
    // Default `null` means "could not read the install name", which the guard must treat as
    // shadowing — so every test that does not care still exercises the conservative branch.
    readOf: (file: string): Uint8Array | null =>
      installNames[file] === undefined ? null : macho64(installNames[file]),
  };
}

/**
 * A minimal 64-bit Mach-O containing one `LC_ID_DYLIB`. Built rather than checked in, so the parser
 * is tested against a file whose every byte is accounted for — the real binaries are covered
 * separately in `test/mlx.mixing.test.ts`, against `otool -D`.
 */
function macho64(installName: string, options: { omitIdDylib?: boolean } = {}): Uint8Array {
  const nameBytes = new TextEncoder().encode(`${installName}\0`);
  // struct dylib_command: cmd, cmdsize, name offset, timestamp, current, compatibility = 24 bytes.
  const cmdSize = Math.ceil((24 + nameBytes.length) / 8) * 8;
  const headerSize = 32;
  const bytes = new Uint8Array(headerSize + (options.omitIdDylib === true ? 8 : cmdSize));
  const view = new DataView(bytes.buffer);
  view.setUint32(0, 0xfeedfacf, true); // MH_MAGIC_64, little-endian on disk
  view.setUint32(4, 0x0100_000c, true); // CPU_TYPE_ARM64
  view.setUint32(12, 6, true); // MH_DYLIB
  view.setUint32(16, 1, true); // ncmds
  if (options.omitIdDylib === true) {
    // A load command that is not LC_ID_DYLIB — e.g. LC_LOAD_DYLIB, as in a bundle.
    view.setUint32(20, 8, true);
    view.setUint32(headerSize, 0x0c, true);
    view.setUint32(headerSize + 4, 8, true);
    return bytes;
  }
  view.setUint32(20, cmdSize, true); // sizeofcmds
  view.setUint32(headerSize, 0x0d, true); // LC_ID_DYLIB
  view.setUint32(headerSize + 4, cmdSize, true);
  view.setUint32(headerSize + 8, 24, true); // name offset, relative to the command
  bytes.set(nameBytes, headerSize + 24);
  return bytes;
}

const inspect = (
  resident: readonly string[],
  sizes: Record<string, number>,
  hashes?: Record<string, string>,
  installNames?: Record<string, string>,
) => inspectResidentMlx({ libDir: LIB_DIR, resident, ...fs(sizes, hashes, installNames) });

describe("what counts as a foreign MLX", () => {
  it("is clear when nothing is resident", () => {
    const report = inspect([], {});
    expect(report).toEqual({ libDir: LIB_DIR, foreign: [], fatal: false });
    expect(hasMixingReport(report)).toBe(false);
  });

  it("ignores libraries we do not ship", () => {
    // A resident libonnxruntime is not our business, and treating it as one would make the guard
    // useless in exactly the process that has several native runtimes loaded.
    const report = inspect(["/other/libonnxruntime.dylib", "/other/onnxruntime_binding.node"], {});
    expect(report.foreign).toEqual([]);
    expect(hasMixingReport(report)).toBe(false);
  });

  it("is clear when our own library is the resident one", () => {
    const report = inspect([path.join(LIB_DIR, "libmlx.dylib")], {});
    expect(report.fatal).toBe(false);
    expect(report.foreign).toEqual([]);
  });

  it("does not mistake a symlinked route to our own library for a foreign build", () => {
    // Node reports images at the path dyld recorded, which is routinely a symlinked store path
    // (pnpm). Without realpath both sides, this is the false positive that breaks every install.
    const real = "/real/store/mlayax-darwin-arm64/lib/libmlx.dylib";
    const report = inspectResidentMlx({
      libDir: "/node_modules/.pnpm/mlayax-darwin-arm64@0.1.0/lib",
      resident: [real],
      ours: { "libmlx.dylib": path.join(LIB_DIR, "libmlx.dylib") },
      bytesOf: () => 100,
      sha256Of: () => "same",
      // Unreadable install name: the conservative branch, which is what this case is not about.
      readOf: () => null,
      realpathOf: (file) =>
        file.startsWith("/node_modules/.pnpm") || file === real
          ? "/real/store/mlayax-darwin-arm64/lib" +
            (file.endsWith("libmlx.dylib") ? "/libmlx.dylib" : "")
          : null,
    });
    expect(report.foreign).toEqual([]);
    expect(report.fatal).toBe(false);
  });
});

describe("a different build is fatal", () => {
  it("flags a different size as fatal and reports the size", () => {
    const foreign = "/other/@johnhenry/lib/libmlx.dylib";
    const ours = path.join(LIB_DIR, "libmlx.dylib");
    const report = inspect([foreign], { [foreign]: 20_900_000, [ours]: 21_100_000 }, undefined, {
      [foreign]: "@rpath/libmlx.dylib",
    });
    expect(report.fatal).toBe(true);
    expect(report.foreign).toEqual([
      {
        path: foreign,
        bytes: 20_900_000,
        installName: "@rpath/libmlx.dylib",
        identical: false,
        shadows: true,
      },
    ]);
  });

  it("flags an equal size with a different hash as fatal", () => {
    // Size alone is not identity: two builds of the same MLX version differ here (§2 measures
    // 21.1 MB wheel vs 21.9 MB johnhenry), and a rebuild can easily land on the same length.
    const foreign = "/other/lib/libmlx.dylib";
    const ours = path.join(LIB_DIR, "libmlx.dylib");
    const report = inspect(
      [foreign],
      { [foreign]: 1000, [ours]: 1000 },
      { [foreign]: "aaaa", [ours]: "bbbb" },
    );
    expect(report.fatal).toBe(true);
    expect(report.foreign[0]?.identical).toBe(false);
  });

  it("treats an unreadable resident library as fatal rather than assuming it is ours", () => {
    const report = inspect(["/gone/libmlx.dylib"], {});
    expect(report.fatal).toBe(true);
    expect(report.foreign[0]?.bytes).toBeNull();
  });

  it("applies the same rule to libjaccl", () => {
    // Our libmlx depends on @rpath/libjaccl.dylib and has no LC_RPATH of its own, so a foreign
    // jaccl is shadowed by exactly the same mechanism.
    const foreign = "/other/lib/libjaccl.dylib";
    const ours = path.join(LIB_DIR, "libjaccl.dylib");
    const report = inspect([foreign], { [foreign]: 1, [ours]: 2 }, undefined, {
      [foreign]: "@rpath/libjaccl.dylib",
    });
    expect(report.fatal).toBe(true);
    expect(report.foreign[0]?.path).toBe(foreign);
  });

  it("names only the foreign library when ours is also resident", () => {
    const foreign = "/other/lib/libmlx.dylib";
    const report = inspect([path.join(LIB_DIR, "libmlx.dylib"), foreign], {
      [foreign]: 10,
      [path.join(LIB_DIR, "libmlx.dylib")]: 20,
    });
    expect(report.foreign.map((l) => l.path)).toEqual([foreign]);
  });
});

describe("only a library that would actually be matched is fatal", () => {
  it("lets a resident build with an absolute install name pass", () => {
    // Homebrew's bottle names itself /opt/homebrew/…/libmlx.dylib. dyld is asked for
    // "@rpath/libmlx.dylib", which that cannot satisfy, so it loads as a second copy and our addon
    // keeps its own. Failing here would block a configuration that works.
    const brew = "/opt/homebrew/opt/mlx/lib/libmlx.dylib";
    const ours = path.join(LIB_DIR, "libmlx.dylib");
    const report = inspect([brew], { [brew]: 15_000_000, [ours]: 21_100_000 }, undefined, {
      [brew]: "/opt/homebrew/Cellar/mlx/0.32.3/lib/libmlx.dylib",
    });
    expect(report.fatal).toBe(false);
    expect(hasMixingReport(report)).toBe(true);
    expect(report.foreign[0]?.shadows).toBe(false);
    expect(report.foreign[0]?.installName).toBe("/opt/homebrew/Cellar/mlx/0.32.3/lib/libmlx.dylib");
  });

  it("still explains the non-fatal case differently, so the note is not misleading", () => {
    const brew = "/opt/homebrew/opt/mlx/lib/libmlx.dylib";
    const ours = path.join(LIB_DIR, "libmlx.dylib");
    const warning = mixingWarning(
      inspect([brew], { [brew]: 1, [ours]: 2 }, undefined, {
        [brew]: "/opt/homebrew/opt/mlx/lib/libmlx.dylib",
      }),
    );
    expect(warning).toContain(brew);
    expect(warning).toMatch(/cannot shadow/);
    expect(warning).toMatch(/two Metal device caches|double the RSS/);
    expect(warning).not.toMatch(/nothing is broken/);
  });

  it("assumes the worst when the install name cannot be read", () => {
    // An unreadable or unrecognised image could be anything, including a shadowing build. The one
    // thing the guard must not do is talk itself out of a real collision.
    const foreign = "/weird/libmlx.dylib";
    const ours = path.join(LIB_DIR, "libmlx.dylib");
    const report = inspect([foreign], { [foreign]: 1, [ours]: 2 });
    expect(report.foreign[0]?.installName).toBeNull();
    expect(report.foreign[0]?.shadows).toBe(true);
    expect(report.fatal).toBe(true);
  });

  it("does not fail when a shadowing image is byte-identical", () => {
    // Same install name and same build is a plain dedupe: dyld hands over equivalent code.
    const copy = "/tmp/copy/libmlx.dylib";
    const ours = path.join(LIB_DIR, "libmlx.dylib");
    const report = inspect(
      [copy],
      { [copy]: 1000, [ours]: 1000 },
      { [copy]: "h", [ours]: "h" },
      { [copy]: "@rpath/libmlx.dylib" },
    );
    expect(report.foreign[0]?.shadows).toBe(true);
    expect(report.fatal).toBe(false);
  });
});

describe("an identical duplicate is legal", () => {
  it("is reported but not fatal when size and hash agree", () => {
    const copy = "/tmp/copy/libmlx.dylib";
    const ours = path.join(LIB_DIR, "libmlx.dylib");
    const report = inspect(
      [copy],
      { [copy]: 1000, [ours]: 1000 },
      { [copy]: "same-bytes", [ours]: "same-bytes" },
    );
    expect(report.fatal).toBe(false);
    expect(hasMixingReport(report)).toBe(true);
    expect(report.foreign[0]?.identical).toBe(true);
  });

  it("still says something, because two identical MLX installs is a mess to debug", () => {
    const copy = "/tmp/copy/libmlx.dylib";
    const ours = path.join(LIB_DIR, "libmlx.dylib");
    const report = inspect([copy], { [copy]: 1000, [ours]: 1000 }, { [copy]: "h", [ours]: "h" });
    const warning = mixingWarning(report);
    expect(warning).toContain(copy);
    expect(warning).toContain(LIB_DIR);
    expect(warning).toMatch(/byte-identical/);
    expect(warning).toMatch(/nothing is broken/);
  });
});

describe("assertNoMixedMlx", () => {
  const fatalReport = (): ReturnType<typeof inspect> =>
    inspect(["/other/lib/libmlx.dylib"], {
      "/other/lib/libmlx.dylib": 1,
      [path.join(LIB_DIR, "libmlx.dylib")]: 2,
    });

  it("throws an MlxMixingError carrying the report", () => {
    let caught: unknown;
    try {
      assertNoMixedMlx(fatalReport(), false);
    } catch (error) {
      caught = error;
    }
    expect(caught).toBeInstanceOf(MlxMixingError);
    expect((caught as MlxMixingError).name).toBe("MlxMixingError");
    expect((caught as MlxMixingError).report.fatal).toBe(true);
  });

  it("returns the report for the non-fatal case so the caller can warn", () => {
    const copy = "/tmp/copy/libmlx.dylib";
    const ours = path.join(LIB_DIR, "libmlx.dylib");
    const report = inspect([copy], { [copy]: 5, [ours]: 5 }, { [copy]: "h", [ours]: "h" });
    expect(assertNoMixedMlx(report, false)).toBe(report);
  });

  it("lets a deliberate caller through with the escape hatch", () => {
    expect(assertNoMixedMlx(fatalReport(), true).fatal).toBe(true);
  });
});

describe("readInstallName — the one thing that decides fatal from footnote", () => {
  it("reads LC_ID_DYLIB from a thin 64-bit image", () => {
    expect(readInstallName(macho64("@rpath/libmlx.dylib"))).toBe("@rpath/libmlx.dylib");
    expect(readInstallName(macho64("/opt/homebrew/opt/mlx/lib/libmlx.dylib"))).toBe(
      "/opt/homebrew/opt/mlx/lib/libmlx.dylib",
    );
  });

  it("returns null for an image with no LC_ID_DYLIB, so the caller stays conservative", () => {
    expect(readInstallName(macho64("@rpath/x.dylib", { omitIdDylib: true }))).toBeNull();
  });

  it("returns null rather than guessing on anything it does not recognise", () => {
    for (const junk of [
      new Uint8Array(),
      new Uint8Array([0x00, 0x01, 0x02, 0x03]),
      new Uint8Array(64).fill(0xff),
      new TextEncoder().encode("#!/bin/sh\necho not a mach-o\n"),
    ]) {
      expect(readInstallName(junk)).toBeNull();
    }
  });

  it("returns null for a truncated header chain instead of reading out of bounds", () => {
    // A prefix read (binding.ts reads 64 KiB) can cut the load-command table short on a big binary.
    const full = macho64("@rpath/libmlx.dylib");
    for (const cut of [8, 20, 32, 40, 48]) {
      expect(() => readInstallName(full.subarray(0, cut))).not.toThrow();
    }
    expect(readInstallName(full.subarray(0, 40))).toBeNull();
  });

  it("picks the arm64 slice of a fat binary", () => {
    const x64 = macho64("@rpath/libmlx.dylib");
    const arm = macho64("@rpath/libmlx.dylib");
    // Rewrite the arm64 slice's install name so the two are distinguishable.
    const armName = new TextEncoder().encode("@rpath/arm64.dylib\0");
    arm.fill(0, 56, arm.byteLength);
    arm.set(armName, 56);

    const entries = [
      { cputype: 0x0100_0007, bytes: x64 }, // CPU_TYPE_X86_64
      { cputype: 0x0100_000c, bytes: arm }, // CPU_TYPE_ARM64
    ];
    const header = 8 + entries.length * 20;
    const total = header + entries.reduce((n, e) => n + e.bytes.byteLength, 0);
    const fat = new Uint8Array(total);
    const view = new DataView(fat.buffer);
    view.setUint32(0, 0xcafebabe, false);
    view.setUint32(4, entries.length, false);
    let cursor = header;
    entries.forEach((entry, i) => {
      const at = 8 + i * 20;
      view.setUint32(at, entry.cputype, false);
      view.setUint32(at + 8, cursor, false);
      view.setUint32(at + 12, entry.bytes.byteLength, false);
      fat.set(entry.bytes, cursor);
      cursor += entry.bytes.byteLength;
    });

    expect(readInstallName(fat)).toBe("@rpath/arm64.dylib");
  });

  it("returns null for a fat binary with no arm64 slice", () => {
    const fat = new Uint8Array(8 + 20 + 8);
    const view = new DataView(fat.buffer);
    view.setUint32(0, 0xcafebabe, false);
    view.setUint32(4, 1, false);
    view.setUint32(8, 0x0100_0007, false); // x86_64 only
    expect(readInstallName(fat)).toBeNull();
  });
});

describe("the message has to be actionable, because dyld's is not", () => {
  const message = mixingError(
    inspect(["/other/lib/libmlx.dylib"], {
      "/other/lib/libmlx.dylib": 20_900_000,
      [path.join(LIB_DIR, "libmlx.dylib")]: 21_100_000,
    }),
  ).message;

  it("names both paths, so the user can see which package won", () => {
    expect(message).toContain("/other/lib/libmlx.dylib");
    expect(message).toContain(LIB_DIR);
  });

  it("states the build difference it measured", () => {
    expect(message).toMatch(/a different build \(19\.9 MB\)/);
  });

  it("names the symptom, so the message connects to the crash it prevents", () => {
    expect(message).toMatch(/Symbol not found/);
    expect(message).toContain("__ZN3mlx4core10gather_qmmE");
  });

  it("also names the failure that happens when the symbols do match", () => {
    // The other half of the hazard, measured with a re-signed copy: dyld binds happily and MLX then
    // cannot find the metallib, because it looks next to whichever library won.
    expect(message).toMatch(/metallib/);
  });

  it("separates the cause from libraries that are merely resident", () => {
    const ours = path.join(LIB_DIR, "libmlx.dylib");
    const messageWithBystander = mixingError(
      inspect(
        ["/other/@johnhenry/lib/libmlx.dylib", "/opt/homebrew/opt/mlx/lib/libmlx.dylib"],
        {
          "/other/@johnhenry/lib/libmlx.dylib": 20_900_000,
          "/opt/homebrew/opt/mlx/lib/libmlx.dylib": 15_000_000,
          [ours]: 21_100_000,
        },
        undefined,
        {
          "/other/@johnhenry/lib/libmlx.dylib": "@rpath/libmlx.dylib",
          "/opt/homebrew/opt/mlx/lib/libmlx.dylib": "/opt/homebrew/opt/mlx/lib/libmlx.dylib",
        },
      ),
    ).message;
    expect(messageWithBystander).toMatch(/Also resident, and not the cause:/);
    expect(messageWithBystander).toContain("/opt/homebrew/opt/mlx/lib/libmlx.dylib");
  });

  it("suggests the packages that actually cause this", () => {
    for (const culprit of [
      "@johnhenry/backend-mlx",
      "@nielspeter/mlx-ts",
      "mlx-bun",
      "@frost-beta/mlx",
    ]) {
      expect(message).toContain(culprit);
    }
  });

  it("documents the escape hatch by its real environment variable name", () => {
    expect(ALLOW_MIXED_ENV).toBe("MLAYAX_ALLOW_MIXED_MLX");
    expect(message).toContain(`MLAYAX_ALLOW_MIXED_MLX=1`);
  });
});

describe("describeMlxLoadFailure — the Bun path", () => {
  const LIB = "/pkg/mlayax-darwin-arm64/lib";
  // Verbatim from the reproduction on 2026-10-06, trimmed at the same point dyld trims it.
  const REAL_DYLD_ERROR = new Error(
    "dlopen(/pkg/mlayax-darwin-arm64/lib/node_mlx.node, 0x0001): Symbol not found: " +
      "__ZN3mlx4core10gather_qmmERKNS0_5arrayES3_S3_RKNSt3__18optionalIS1_EES6_S6_bNS5_IiEES9_" +
      "RKNS4_12basic_stringIcNS4_11char_traitsIcEENS4_9allocatorIcEEEES8_bNS4_7variantIJNS4_9" +
      "monostateENS0_6StreamENS0_17ThreadLocalStreamENS0_6DeviceENSM_10DeviceTypeEEEE",
  );

  it("translates the real dyld error and keeps the original as the cause", () => {
    const translated = describeMlxLoadFailure(REAL_DYLD_ERROR, LIB);
    expect(translated).not.toBe(REAL_DYLD_ERROR);
    expect(translated.message).toMatch(/could not be bound/);
    expect(translated.message).toContain("__ZN3mlx4core10gather_qmmE");
    expect(translated.message).toContain(LIB);
    expect(translated.message).toContain(REAL_DYLD_ERROR.message);
    expect(translated.name).toBe("MlxMixingError");
    expect(translated.cause).toBe(REAL_DYLD_ERROR);
  });

  it("handles the lower-case spelling newer macOS uses", () => {
    const error = new Error(
      "dlopen(/x/node_mlx.node): symbol not found in flat namespace '_x' _ZN3mlx",
    );
    expect(describeMlxLoadFailure(error, LIB).message).toMatch(/could not be bound/);
  });

  it("handles an incompatible library version, which is the same hazard", () => {
    const error = new Error("Incompatible library version: libmlx.dylib requires version 1.0.0");
    expect(describeMlxLoadFailure(error, LIB).message).toMatch(/could not be bound/);
  });

  it("leaves an unrelated failure completely alone", () => {
    // A missing addon file must keep its own ENOENT, not be blamed on library mixing.
    const unrelated = new Error("Cannot find module '/pkg/lib/node_mlx.node'");
    expect(describeMlxLoadFailure(unrelated, LIB)).toBe(unrelated);
  });

  it("wraps a non-Error throw", () => {
    const translated = describeMlxLoadFailure("Symbol not found: _ZN3mlx4core", LIB);
    expect(translated.message).toMatch(/could not be bound/);
  });

  it("explains why this path exists at all", () => {
    // The message has to say that Bun cannot pre-check, or a reader will wonder why the earlier,
    // better error did not fire.
    expect(describeMlxLoadFailure(REAL_DYLD_ERROR, LIB).message).toMatch(/process\.report/);
  });
});

describe("checkForMixedMlx — the whole load-time policy", () => {
  const warningSink = (): { messages: string[]; warn: (message: string) => void } => {
    const messages: string[] = [];
    return { messages, warn: (message) => messages.push(message) };
  };

  const input = (resident: readonly string[], sizes: Record<string, number>, extra = {}) => ({
    libDir: LIB_DIR,
    resident,
    ...fs(sizes),
    ...extra,
  });

  it("stays completely quiet when nothing is resident", () => {
    const sink = warningSink();
    const report = checkForMixedMlx({ ...input([], {}), allowMixed: false, warn: sink.warn });
    expect(report.foreign).toEqual([]);
    expect(sink.messages).toEqual([]);
  });

  it("stays quiet when the only resident MLX is our own", () => {
    const sink = warningSink();
    checkForMixedMlx({
      ...input([path.join(LIB_DIR, "libmlx.dylib")], {}),
      allowMixed: false,
      warn: sink.warn,
    });
    expect(sink.messages).toEqual([]);
  });

  it("warns, without throwing, for a mix it can survive", () => {
    const brew = "/opt/homebrew/opt/mlx/lib/libmlx.dylib";
    const ours = path.join(LIB_DIR, "libmlx.dylib");
    const sink = warningSink();
    const report = checkForMixedMlx({
      libDir: LIB_DIR,
      resident: [brew],
      ...fs(
        { [brew]: 15_000_000, [ours]: 21_100_000 },
        {},
        { [brew]: "/opt/homebrew/opt/mlx/lib/libmlx.dylib" },
      ),
      allowMixed: false,
      warn: sink.warn,
    });
    expect(report.fatal).toBe(false);
    expect(sink.messages).toHaveLength(1);
    expect(sink.messages[0]).toMatch(/cannot shadow/);
  });

  it("throws for a mix it cannot survive, and does not warn as well", () => {
    const sink = warningSink();
    expect(() =>
      checkForMixedMlx({
        ...input(["/other/lib/libmlx.dylib"], {
          "/other/lib/libmlx.dylib": 1,
          [path.join(LIB_DIR, "libmlx.dylib")]: 2,
        }),
        allowMixed: false,
        warn: sink.warn,
      }),
    ).toThrow(MlxMixingError);
    // Throwing ends the load, so a warning would be noise nobody reads.
    expect(sink.messages).toEqual([]);
  });

  it("proceeds but still warns once the escape hatch is set", () => {
    // Waiving the check must not mean silence: the user should be able to see what they waived.
    const sink = warningSink();
    const report = checkForMixedMlx({
      ...input(["/other/lib/libmlx.dylib"], {
        "/other/lib/libmlx.dylib": 1,
        [path.join(LIB_DIR, "libmlx.dylib")]: 2,
      }),
      allowMixed: true,
      warn: sink.warn,
    });
    expect(report.fatal).toBe(true);
    expect(sink.messages).toHaveLength(1);
    expect(sink.messages[0]).toContain(ALLOW_MIXED_ENV);
    expect(sink.messages[0]).toMatch(/It is usually right to unset it/);
  });

  it("does not claim a shadowing build 'takes the slot' when it cannot", () => {
    // The two non-identical cases must read differently, or the note would be actively misleading.
    const brew = "/opt/homebrew/opt/mlx/lib/libmlx.dylib";
    const ours = path.join(LIB_DIR, "libmlx.dylib");
    const sink = warningSink();
    checkForMixedMlx({
      libDir: LIB_DIR,
      resident: [brew],
      ...fs(
        { [brew]: 15_000_000, [ours]: 21_100_000 },
        {},
        { [brew]: "/opt/homebrew/opt/mlx/lib/libmlx.dylib" },
      ),
      allowMixed: false,
      warn: sink.warn,
    });
    expect(sink.messages[0]).not.toContain("takes the slot");
    expect(sink.messages[0]).toContain("/opt/homebrew/opt/mlx/lib/libmlx.dylib");
  });
});

describe("the real I/O adapters, against real files", () => {
  const dir = mkdtempSync(path.join(tmpdir(), "mlayax-adapters-"));
  afterAll(() => rmSync(dir, { recursive: true, force: true }));

  const missing = path.join(dir, "absent");
  const present = path.join(dir, "present.bin");
  writeFileSync(present, Buffer.alloc(100, 7));
  const big = path.join(dir, "big.bin");
  writeFileSync(big, Buffer.alloc(HEADER_PREFIX_BYTES + 500, 3));

  it("returns null rather than throwing when a file cannot be read", () => {
    // Every one of these is reachable in the field: a loader path whose library was moved, a
    // permission problem, a deleted temp directory. The guard must survive them, not crash on them.
    expect(bytesOf(missing)).toBeNull();
    expect(sha256Of(missing)).toBeNull();
    expect(realpathOf(missing)).toBeNull();
    expect(readOf(missing)).toBeNull();
  });

  it("reports a real size and a real digest", () => {
    expect(bytesOf(present)).toBe(100);
    expect(sha256Of(present)).toMatch(/^[0-9a-f]{64}$/);
    expect(realpathOf(present)).not.toBeNull();
  });

  it("reads only a bounded prefix, however large the library is", () => {
    // The whole point: answering "what does it call itself" must not read 21 MB.
    expect((readOf(big) as Uint8Array).byteLength).toBe(HEADER_PREFIX_BYTES);
    expect((readOf(present) as Uint8Array).byteLength).toBe(100);
  });

  it("gives up quietly when the runtime will not enumerate its images", () => {
    // Bun's shape, and the disabled-report case. `describeMlxLoadFailure` is what covers Bun.
    expect(residentSharedObjects({})).toEqual([]);
    expect(residentSharedObjects({ report: {} })).toEqual([]);
    expect(
      residentSharedObjects({ report: { getReport: () => ({ sharedObjects: undefined }) } }),
    ).toEqual([]);
    expect(
      residentSharedObjects({ report: { getReport: () => ({ sharedObjects: "nope" }) } }),
    ).toEqual([]);
  });

  it("swallows a throwing report instead of failing the load", () => {
    // `process.report` throws or is absent in some sandboxes; the guard must degrade to silence.
    const throwing = {
      report: {
        getReport: () => {
          throw new Error("report generation is disabled");
        },
      },
    };
    expect(residentSharedObjects(throwing)).toEqual([]);
    expect(
      residentSharedObjects({
        report: {
          getReport: (): never => {
            throw new Error("no report");
          },
        },
      }),
    ).toEqual([]);
  });

  it("keeps only the strings from a real report", () => {
    const fake = {
      report: {
        getReport: () => ({ sharedObjects: ["/a/libmlx.dylib", 42, null, "/b/lib.dylib"] }),
      },
    };
    expect(residentSharedObjects(fake)).toEqual(["/a/libmlx.dylib", "/b/lib.dylib"]);
  });

  it("enumerates this process's images on Node", () => {
    // Not asserting a count — the point is that the real call works in the runtime running the
    // suite, which is what makes the child-process tests meaningful rather than vacuous.
    expect(Array.isArray(residentSharedObjects())).toBe(true);
  });
});

describe("the set of libraries guarded", () => {
  it("covers every MLX dylib the payload ships", () => {
    expect([...MLX_LIBRARY_NAMES].sort()).toEqual(["libjaccl.dylib", "libmlx.dylib"]);
  });
});
