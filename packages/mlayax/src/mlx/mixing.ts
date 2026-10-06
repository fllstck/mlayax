/**
 * The mixing guard — TASKS.md §8.6.
 *
 * macOS resolves a dependent library by **install name**, not by path. Our addon asks for
 * `@rpath/libmlx.dylib`, and every MLX distribution — ours, Homebrew's, `@johnhenry`'s — ships a
 * dylib whose install name is exactly that string. So if another one is already resident in the
 * process, dyld hands it over, and our addon binds against a build it was never compiled against.
 *
 * This is not a wrong-answer bug; it is a load-time failure. Measured against the shipped payload
 * (2026-10-06), with `@johnhenry/backend-mlx-darwin-arm64`'s `libmlx.dylib` resident first:
 *
 * ```text
 * dlopen(…/mlayax-darwin-arm64/lib/node_mlx.node, 0x0001): Symbol not found:
 *   __ZN3mlx4core10gather_qmmERKNS0_5arrayES3_S3_RKNSt3__18optionalIS1_EES6_S6_bNS5_IiEES9_RKNS4_12basic_stringIcNS4_11char_traitsIcEENS4_9allocatorIcEEEES8_bNS4_7variantIJ…
 * ```
 *
 * A reader cannot act on that. The whole job of this module is to turn it into a sentence naming two
 * file paths and the package that caused it.
 *
 * **The check must run before the load**, because afterwards the process is already poisoned: dyld
 * has bound the symbols and the only recovery is a different process. Node exposes the dynamic
 * loader's image list through `process.report.getReport().sharedObjects`; Bun's report always has an
 * empty `sharedObjects`, so on Bun the guard cannot run and {@link describeMlxLoadFailure} translates
 * the eventual error instead. See `binding.ts` for how the two are combined.
 *
 * "Different build" is decided the way §2 decides MLX build quality: size first, then sha256. A
 * byte-identical duplicate is deliberately *not* fatal — dyld deduplicating the same build is
 * harmless — so it warns where a foreign build throws.
 */

import { createHash } from "node:crypto";
import { closeSync, openSync, readFileSync, readSync, realpathSync, statSync } from "node:fs";
import path from "node:path";

/** The MLX dynamic libraries our payload ships, and the ones a competing package would shadow. */
export const MLX_LIBRARY_NAMES: readonly string[] = ["libmlx.dylib", "libjaccl.dylib"];

/**
 * The install name our addon asks the loader for.
 *
 * `otool -L …/lib/node_mlx.node` shows a single relevant entry, `@rpath/libmlx.dylib`, and no
 * `libjaccl` at all — jaccl is reached through `libmlx`'s own dependency entry. Both name themselves
 * `@rpath/<basename>`, which is why one comparison covers both.
 */
function installNameFor(basename: string): string {
  return `@rpath/${basename}`;
}

/** Mach-O constants, from `<mach-o/loader.h>`. */
const MH_MAGIC_64 = 0xfeedfacf;
const MH_MAGIC = 0xfeedface;
const FAT_MAGIC = 0xcafebabe;
const FAT_MAGIC_64 = 0xcafebabf;
const CPU_TYPE_ARM64 = 0x0100_000c;
const LC_ID_DYLIB = 0x0d;

/**
 * Read a dynamic library's own install name from `LC_ID_DYLIB`.
 *
 * This is what decides whether a resident library is dangerous. `@rpath/libmlx.dylib` is matched by
 * install name, so a resident build that calls itself that **will** satisfy our addon's request for
 * it. A build that names itself `/opt/homebrew/opt/mlx/lib/libmlx.dylib` (which is what Homebrew's
 * bottle does) cannot be matched and simply loads as a second, separate copy.
 *
 * Returns `null` for anything not understood — a bundle with no `LC_ID_DYLIB`, a 32-bit or fat
 * binary, a truncated file — and callers must treat `null` as "assume it can shadow", since the
 * point is never to talk ourselves out of a real collision.
 *
 * Never throws. The caller reads a bounded prefix of the file (see `binding.ts`), so a cut in the
 * middle of the load-command table is an expected input rather than an exceptional one — and a parse
 * error escaping here would turn the guard into the crash it exists to prevent.
 */
export function readInstallName(image: Uint8Array): string | null {
  try {
    return parseInstallName(image);
  } catch {
    // Bounds are checked in `parseInstallName`; this is for anything an unusual image can still reach.
    return null;
  }
}

function parseInstallName(image: Uint8Array): string | null {
  const view = new DataView(image.buffer, image.byteOffset, image.byteLength);
  const readName = (commandOffset: number): string | null => {
    // struct dylib_command { cmd, cmdsize, struct dylib { union lc_str name; … } }
    if (commandOffset + 12 > image.byteLength) return null;
    const nameOffset = view.getUint32(commandOffset + 8, true);
    const start = commandOffset + nameOffset;
    if (start >= image.byteLength) return null;
    let end = start;
    while (end < image.byteLength && image[end] !== 0) end += 1;
    return end === start ? null : new TextDecoder().decode(image.subarray(start, end));
  };

  const walk = (sliceOffset: number, sliceLength: number): string | null => {
    if (sliceLength < 32) return null;
    // A thin Mach-O is stored in the target's byte order, which for every platform this package
    // supports (arm64, and x86_64 before it) is little-endian. Fat headers below are big-endian by
    // specification, which is why the two reads differ.
    const magic = view.getUint32(sliceOffset, true);
    const is64 = magic === MH_MAGIC_64;
    if (!is64 && magic !== MH_MAGIC) return null;
    const headerSize = is64 ? 32 : 28;
    if (sliceOffset + headerSize > image.byteLength) return null;
    const ncmds = view.getUint32(sliceOffset + 16, true);
    // The slice can end before the buffer does (fat binaries), so bound by whichever is smaller.
    const limit = Math.min(sliceOffset + sliceLength, image.byteLength);
    let cursor = sliceOffset + headerSize;
    for (let i = 0; i < ncmds; i += 1) {
      if (cursor + 8 > limit) return null;
      const cmd = view.getUint32(cursor, true);
      const cmdsize = view.getUint32(cursor + 4, true);
      if (cmdsize < 8) return null;
      if (cmd === LC_ID_DYLIB) return readName(cursor);
      cursor += cmdsize;
    }
    return null;
  };

  if (image.byteLength < 8) return null;
  const outer = view.getUint32(0, false);
  if (outer !== FAT_MAGIC && outer !== FAT_MAGIC_64) return walk(0, image.byteLength);
  // Fat binary: pick the arm64 slice. Fat headers are big-endian; entries are 20 bytes (32) or
  // 32 bytes (64).
  const entrySize = outer === FAT_MAGIC_64 ? 32 : 20;
  const nfat = view.getUint32(4, false);
  for (let i = 0; i < nfat; i += 1) {
    const entry = 8 + i * entrySize;
    if (entry + entrySize > image.byteLength) return null;
    if (view.getUint32(entry, false) !== CPU_TYPE_ARM64) continue;
    const offset =
      outer === FAT_MAGIC_64
        ? view.getBigUint64(entry + 8, false)
        : view.getUint32(entry + 8, false);
    const size =
      outer === FAT_MAGIC_64
        ? view.getBigUint64(entry + 16, false)
        : view.getUint32(entry + 12, false);
    return walk(Number(offset), Number(size));
  }
  return null;
}

/** Opt out of the hard failure, for a caller who has deliberately arranged two builds. */
export const ALLOW_MIXED_ENV = "MLAYAX_ALLOW_MIXED_MLX";

/* -------------------------------------------------------------------------------------------- *
 * The real I/O adapters.
 *
 * They live here rather than in `binding.ts` so that every branch that reads the filesystem can be
 * exercised by a unit test with a real temp file — deleting the file to hit a `null`, truncating it
 * to hit the parser's bounds checks. The load-time policy is too easy to get subtly wrong to leave a
 * third of it reachable only on a machine with a poisoned process.
 * -------------------------------------------------------------------------------------------- */

/** Bytes of a file, or `null` when it cannot be read (a stale loader path, a permission problem). */
export function bytesOf(file: string): number | null {
  try {
    return statSync(file).size;
  } catch {
    return null;
  }
}

/** Lower-case hex sha256 of a file, or `null` when it cannot be read. */
export function sha256Of(file: string): string | null {
  try {
    return createHash("sha256").update(readFileSync(file)).digest("hex");
  } catch {
    return null;
  }
}

/** The file's canonical path, or `null` when it cannot be resolved. */
export function realpathOf(file: string): string | null {
  try {
    return realpathSync(file);
  } catch {
    return null;
  }
}

/**
 * How much of a library is read to answer "what does it call itself".
 *
 * Only the Mach-O header and its load commands are needed, and they are always at the front — so this
 * is a bounded prefix rather than the whole 21 MB, which would make every load pay for a hash-sized
 * read. `readInstallName` returns `null` on a truncated buffer and the guard treats that as
 * "assume it can shadow".
 */
export const HEADER_PREFIX_BYTES = 64 * 1024;

/** The first {@link HEADER_PREFIX_BYTES} of a file, or `null` when it cannot be opened. */
export function readOf(file: string): Uint8Array | null {
  try {
    const handle = openSync(file, "r");
    try {
      const buffer = Buffer.alloc(HEADER_PREFIX_BYTES);
      const read = readSync(handle, buffer, 0, HEADER_PREFIX_BYTES, 0);
      return new Uint8Array(buffer.buffer, buffer.byteOffset, read);
    } finally {
      closeSync(handle);
    }
  } catch {
    return null;
  }
}

/** The image list a runtime is willing to expose, structurally typed so a test can pass a fake. */
export interface SharedObjectReporter {
  report?: { getReport?: (() => { sharedObjects?: unknown }) | undefined } | undefined;
}

/**
 * The images the dynamic loader has resident, or `[]` when the runtime will not say.
 *
 * Node keeps the list in the diagnostic report. **Bun's report exists but always carries an empty
 * `sharedObjects`**, which is the reason the guard cannot run there and
 * {@link describeMlxLoadFailure} has to exist. The whole read is wrapped because `process.report` is
 * documented as unavailable when report generation is disabled.
 */
export function residentSharedObjects(proc: SharedObjectReporter = process): readonly string[] {
  try {
    const objects = proc.report?.getReport?.()?.sharedObjects;
    return Array.isArray(objects) ? objects.filter((s): s is string => typeof s === "string") : [];
  } catch {
    return [];
  }
}

/** A resident MLX image in the process that is not the one our lib directory provides. */
export interface ForeignMlxLibrary {
  /** The path the dynamic loader reports, i.e. the file that actually won. */
  path: string;
  /** Bytes of that file, or `null` when it could not be read. */
  bytes: number | null;
  /** Its own `LC_ID_DYLIB`, or `null` when it could not be read. */
  installName: string | null;
  /** True when it is byte-identical to the library we ship under the same basename. */
  identical: boolean;
  /**
   * True when dyld would satisfy our addon's request with this image instead of ours.
   *
   * This is the whole difference between a fatal state and a footnote: `@rpath/libmlx.dylib` is
   * matched by name, so a resident build calling itself that wins. An absolute install name cannot.
   */
  shadows: boolean;
}

/** What {@link inspectResidentMlx} concluded. */
export interface MixingReport {
  /** Our library directory, carried for the error message. */
  libDir: string;
  /** Resident MLX images outside {@link libDir}. Empty means nothing to say. */
  foreign: ForeignMlxLibrary[];
  /** True when a resident *different build* would be matched instead of ours. Not survivable. */
  fatal: boolean;
}

/** Thrown when a foreign MLX build is already resident. */
export class MlxMixingError extends Error {
  override readonly name = "MlxMixingError";

  constructor(
    message: string,
    readonly report: MixingReport,
  ) {
    super(message);
  }
}

/**
 * Is `file` inside `dir`?
 *
 * Both sides go through `realpathOf` first: Node reports images at the path dyld recorded, which is
 * routinely a symlinked route (a pnpm store, a `node_modules/.bin` shim), so a pure string
 * comparison against our lib directory produces false alarms.
 */
function isInsideDir(
  file: string,
  dir: string,
  realpathOf: (file: string) => string | null,
): boolean {
  const resolvedFile = realpathOf(file) ?? path.resolve(file);
  const resolvedDir = realpathOf(dir) ?? path.resolve(dir);
  if (resolvedFile === resolvedDir) return true;
  return resolvedFile.startsWith(
    resolvedDir.endsWith(path.sep) ? resolvedDir : resolvedDir + path.sep,
  );
}

/** The inputs to the decision, all injected so the logic is testable without a native payload. */
export interface InspectResidentMlxInput {
  /** The directory our addon resolves from — anything resident outside it is a candidate. */
  libDir: string;
  /** Absolute paths of every image the dynamic loader has resident, in load order. */
  resident: readonly string[];
  /** Basename → absolute path of the library we ship, e.g. `libmlx.dylib` → `…/lib/libmlx.dylib`. */
  ours: Readonly<Record<string, string>>;
  bytesOf: (file: string) => number | null;
  sha256Of: (file: string) => string | null;
  /** Raw first bytes of a file, or `null` when unreadable. Used only for `LC_ID_DYLIB`. */
  readOf: (file: string) => Uint8Array | null;
  realpathOf?: (file: string) => string | null;
}

/**
 * Decide whether the process already has an MLX build that would shadow ours.
 *
 * Pure with respect to the filesystem — every read is injected — so the whole decision table is
 * unit-tested on any platform, in the same style as `selectBinding`.
 */
export function inspectResidentMlx(input: InspectResidentMlxInput): MixingReport {
  const { libDir, resident, ours, bytesOf, sha256Of, readOf } = input;
  const realpathOf = input.realpathOf ?? ((): null => null);
  const foreign: ForeignMlxLibrary[] = [];

  for (const image of resident) {
    const basename = path.basename(image);
    const ourFile = ours[basename];
    // Not one of the libraries we ship: a foreign `libonnxruntime.dylib` is not our problem.
    if (ourFile === undefined) continue;
    // Correctly bound to our own copy — this is the healthy case.
    if (isInsideDir(image, libDir, realpathOf)) continue;

    const residentBytes = bytesOf(image);
    const ourBytes = bytesOf(ourFile);
    let identical = false;
    if (residentBytes !== null && ourBytes !== null && residentBytes === ourBytes) {
      // Same size is suggestive; only the hash settles it. Reached rarely, so paying for a 21 MB
      // read here is fine — and it is what keeps a legitimate duplicate from failing the load.
      const residentSha = sha256Of(image);
      identical = residentSha !== null && residentSha === sha256Of(ourFile);
    }

    const raw = readOf(image);
    const installName = raw === null ? null : readInstallName(raw);
    // `null` means we could not tell, and the safe reading of "could not tell" is "assume it wins".
    const shadows = installName === null || installName === installNameFor(basename);

    foreign.push({ path: image, bytes: residentBytes, installName, identical, shadows });
  }

  // Only a different build that would actually be matched is fatal. A resident build with an
  // absolute install name loads alongside ours and cannot hijack the binding, so blocking the load
  // over it would be crying wolf — and it would block a working configuration.
  return {
    libDir,
    foreign,
    fatal: foreign.some((library) => library.shadows && !library.identical),
  };
}

/** True when a report is worth telling the user about at all. */
export function hasMixingReport(report: MixingReport): boolean {
  return report.foreign.length > 0;
}

function formatBytes(bytes: number | null): string {
  return bytes === null ? "unreadable" : `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

/** The warning text for a foreign MLX that is legal but worth knowing about. */
export function mixingWarning(report: MixingReport): string {
  const lines = report.foreign.map((library) => {
    const names = library.installName === null ? "an unreadable install name" : library.installName;
    const why = library.identical
      ? "byte-identical to ours, so the bindings are correct"
      : library.shadows
        ? `a different build (${formatBytes(library.bytes)}) naming itself ${names}, which is the ` +
          `name our addon asks for — it takes the slot`
        : `a different build (${formatBytes(library.bytes)}), but it names itself ${names}, so it ` +
          `cannot take our slot`;
    return `  ${library.path}\n      ${why}`;
  });

  const shadowing = report.foreign.some((library) => library.shadows);
  const sameBuild = report.foreign.every((library) => library.identical);
  const tail =
    shadowing && !sameBuild
      ? `You are seeing this because ${ALLOW_MIXED_ENV} is set, so the check that would have ` +
        `refused this load was waived. It is usually right to unset it.`
      : shadowing
        ? `It is compatible with the one in ${report.libDir}, so nothing is broken — but two MLX ` +
          `installs in one dependency tree is a state worth cleaning up.`
        : `It cannot shadow the one in ${report.libDir}, so MLX resolves correctly; this is only a ` +
          `note that two MLX runtimes are resident in the same process (two Metal device caches, ` +
          `roughly double the RSS).`;

  return (
    `mlayax: another MLX dynamic library is already loaded in this process:\n${lines.join("\n")}\n` +
    tail
  );
}

/** The error for a resident *different* build that would be matched instead of ours. */
export function mixingError(report: MixingReport): MlxMixingError {
  // Only the shadowing ones are the reason we are refusing; a non-shadowing resident build is
  // mentioned as context so the message explains the whole state of the process.
  const shadowing = report.foreign.filter((library) => library.shadows);
  const other = report.foreign.filter((library) => !library.shadows);

  const detail = shadowing
    .map((library) => `  ${library.path}\n      a different build (${formatBytes(library.bytes)})`)
    .join("\n");

  const alsoResident =
    other.length === 0
      ? ""
      : `\nAlso resident, and not the cause:\n${other
          .map((library) => `  ${library.path}`)
          .join("\n")}\n`;

  const message =
    `@fllstck/mlayax cannot load its MLX runtime: a different libmlx.dylib is already resident in ` +
    `this process, and macOS resolves dynamic libraries by install name — every MLX distribution ` +
    `published for Node names itself "@rpath/libmlx.dylib", so our addon would bind to that one ` +
    `instead of the one we ship.\n\n` +
    `Already loaded:\n${detail}\n${alsoResident}\n` +
    `Ours:\n  ${report.libDir}\n\n` +
    `The symptom if you continue is a dyld "Symbol not found" crash naming a mangled C++ symbol ` +
    `such as __ZN3mlx4core10gather_qmmE…, which is why this is checked first. Where the symbols do ` +
    `happen to match, MLX instead fails at run time with "Failed to load the default metallib", ` +
    `because it looks for mlx.metallib next to whichever library won. Either way it is not ` +
    `recoverable in this process.\n\n` +
    `Fix: load only one MLX runtime per process. The usual cause is a second MLX package in the ` +
    `same dependency tree — "@johnhenry/backend-mlx", "@nielspeter/mlx-ts", "mlx-bun", a ` +
    `"@frost-beta/mlx" that was built in place, or a Python mlx wheel loaded through a Python ` +
    `bridge. Remove it, or load it in a separate worker process.\n\n` +
    `To proceed anyway — only correct if you have verified the builds are interchangeable — set ` +
    `${ALLOW_MIXED_ENV}=1.`;

  return new MlxMixingError(message, report);
}

/**
 * Throw if the process cannot host our MLX runtime. Returns the report either way, so the caller can
 * warn about the non-fatal case.
 */
export function assertNoMixedMlx(report: MixingReport, allowMixed: boolean): MixingReport {
  if (report.fatal && !allowMixed) throw mixingError(report);
  return report;
}

/**
 * Inspect the process and act: throw for a mix we cannot survive, warn about one we can, stay quiet
 * when there is nothing to say.
 *
 * This is the whole load-time policy in one place, kept out of `binding.ts` so it is reachable from a
 * test without a native payload or a poisoned process. `binding.ts` supplies the filesystem and the
 * process's image list; everything decided from them is decided here.
 *
 * The escape hatch deliberately still warns: a caller who waived the check should see what they
 * waived, not get silence.
 */
export function checkForMixedMlx(
  input: InspectResidentMlxInput & { allowMixed: boolean; warn: (message: string) => void },
): MixingReport {
  const report = inspectResidentMlx(input);
  if (!hasMixingReport(report)) return report;
  if (report.fatal) assertNoMixedMlx(report, input.allowMixed);
  input.warn(mixingWarning(report));
  return report;
}

/**
 * Turn a dyld failure from the addon load into something actionable.
 *
 * This is the Bun path: its `process.report` always reports zero shared objects, so the pre-load
 * check has nothing to read and the collision can only be diagnosed from the error it produces. It
 * also covers Node if a foreign library appeared between the check and the load.
 *
 * Returns the original error unchanged when it does not look like an MLX symbol mismatch — a
 * genuinely missing addon file must keep its own message.
 */
export function describeMlxLoadFailure(error: unknown, libDir: string): Error {
  const original = error instanceof Error ? error : new Error(String(error));
  const text = original.message;

  // dyld spells these two ways depending on macOS version, and leaks the mangled symbol into both.
  const isDyldMismatch =
    /Symbol not found|symbol not found/i.test(text) || /Incompatible library version/i.test(text);
  const isMlxSymbol = /_ZN3mlx|libmlx|libjaccl/i.test(text);
  if (!isDyldMismatch || !isMlxSymbol) return original;

  const symbol = /((?:_ZN3mlx|___Z|_)\w+)/.exec(text)?.[1];
  const cause =
    `the MLX runtime in ${libDir} could not be bound against the libmlx.dylib that was already ` +
    `resident in this process`;

  const message =
    `@fllstck/mlayax failed to load its MLX runtime: ${cause}.\n\n` +
    (symbol === undefined ? "" : `dyld reported a missing symbol:\n  ${symbol}\n\n`) +
    `macOS resolves dynamic libraries by install name, and every MLX distribution calls itself ` +
    `"@rpath/libmlx.dylib", so a second MLX in the same process shadows ours rather than ` +
    `coexisting with it.\n\n` +
    `Fix: load only one MLX runtime per process. Look for another MLX package in the dependency ` +
    `tree ("@johnhenry/backend-mlx", "@nielspeter/mlx-ts", "mlx-bun", a "@frost-beta/mlx" built in ` +
    `place, or a Python mlx wheel reached through a bridge) or verify the payload with ` +
    `\`node -e "…"\` / \`otool -L\` — see the README's "Two MLX builds in one process" section.\n\n` +
    `Under Node this is detected before the load, with both paths named. Bun's process.report ` +
    `exposes no loaded-image list, so the diagnosis falls back to this message.\n\n` +
    `Original error: ${text}`;

  const wrapped = new Error(message, { cause: original });
  wrapped.name = original.name === "Error" ? "MlxMixingError" : original.name;
  return wrapped;
}
