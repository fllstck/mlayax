/**
 * Hugging Face Hub fetcher: resolve a repo to a local snapshot, downloading only what is missing.
 *
 * Weights are ~803 MiB and Apache-2.0, so they are never bundled — the first `load()` downloads
 * them once into the Hugging Face cache and every later call is a cache hit.
 *
 * ## What "compatible with the Python cache" means here
 *
 * Precisely this, and no more:
 *
 * - **Reading** reuses whatever `huggingface_hub` already wrote. We resolve `refs/<revision>` to a
 *   commit, then walk `snapshots/<commit>/`, following symlinks. Nothing about `blobs/` matters on
 *   the read path, which is what makes an existing cache usable as-is — nobody should re-download
 *   803 MiB because they switched language.
 * - **Writing** produces the same canonical per-repo shape: `refs/<revision>`, `blobs/<name>`, and
 *   `snapshots/<commit>/<path>` as a relative symlink to `../../blobs/<name>`. Blobs are named by the
 *   git blob sha1 for ordinary files and by the LFS `sha256` for large ones, which is what upstream
 *   uses.
 * - **Not** reproduced: recent `huggingface_hub` stores LFS payloads in a *shared* store at
 *   `hub/blobs/<xx>/<sha256>`, with the per-repo `blobs/<sha256>` being a symlink into it. We write a
 *   regular file at the per-repo path instead. A `snapshots/` entry resolves either way, so both
 *   readers work; we simply give up cross-repo deduplication. Deliberate, and the reason the
 *   "compatible" claim above is spelled out rather than implied.
 *
 * ## Failure modes worth knowing
 *
 * `offline: true` never touches the network and fails with a message naming the cache directory and
 * the revision it looked for. Downloads land in `<blob>.incomplete` and are renamed into place, so an
 * interrupted download never leaves a blob that a later run would treat as complete — the failure
 * that produces a corrupt safetensors error three layers away from its cause.
 */

import { createHash } from "node:crypto";
import {
  createWriteStream,
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  renameSync,
  statSync,
  symlinkSync,
  unlinkSync,
  writeFileSync,
} from "node:fs";
import { homedir } from "node:os";
import { dirname, join, relative, resolve } from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";

/** Default endpoint. Overridden in tests to point at a local fixture server. */
export const DEFAULT_ENDPOINT = "https://huggingface.co";

/** Environment variables consulted for the cache location, in upstream's precedence order. */
export const CACHE_ENV_VARS = {
  hubCache: "HF_HUB_CACHE",
  legacyHubCache: "HUGGINGFACE_HUB_CACHE",
  home: "HF_HOME",
  xdgCache: "XDG_CACHE_HOME",
} as const;

/** Options accepted by every entry point in this module. */
export interface HubOptions {
  /**
   * Access token for private or gated repos. Sent only to the endpoint, **never** to the CDN the
   * `resolve` URL redirects to — see `fetchFile`.
   */
  token?: string;
  /** Branch, tag, or 40-hex commit. Default `"main"`. */
  revision?: string;
  /** Never touch the network; use the cache or fail. */
  offline?: boolean;
  /** Called as resolution, download and verification progress. */
  onProgress?: (event: HubProgress) => void;
  /** Fetch implementation. Injectable so tests can use a fixture server or a proxy. */
  fetch?: typeof globalThis.fetch;
  /** Cache root, i.e. what `HF_HUB_CACHE` points at. Defaults to the Python default. */
  cacheDir?: string;
  /** Hub base URL. Defaults to `https://huggingface.co`. */
  endpoint?: string;
  /**
   * Re-check the revision upstream instead of trusting a cached snapshot.
   *
   * By default a complete cached snapshot is used without any network access, including for a branch
   * name — `refs/<revision>` is what makes that possible, and trusting it is the entire point of the
   * ref file. Set this to pick up new commits on a branch you have already cached.
   */
  force?: boolean;
}

/** A progress report. Counts are cumulative over the whole operation. */
export interface HubProgress {
  phase: "resolving" | "listing" | "downloading" | "done";
  /** File being worked on, when the phase is per-file. */
  file?: string;
  filesDone: number;
  filesTotal: number;
  bytesDone: number;
  bytesTotal: number;
}

/** A repo resolved to a local snapshot directory. */
export interface ResolvedModel {
  /** The repo id as given, e.g. `"aac6fef/laya-mlx"`. */
  repoId: string;
  /** The resolved commit sha. Safe to record: the snapshot directory is named after it. */
  revision: string;
  /** Absolute path of the snapshot directory. */
  path: string;
  /** True when nothing was downloaded. */
  fromCache: boolean;
  /** Number of files in the snapshot. */
  files: number;
}

/** One entry of the repo tree, as `GET /api/models/{repo}/tree/{rev}?recursive=true` returns it. */
interface TreeEntry {
  type: "file" | "directory";
  path: string;
  size?: number;
  /** Git blob sha1 for ordinary files. */
  oid?: string;
  /** Present for git-LFS payloads; `oid` is the sha256 of the content. */
  lfs?: { oid: string; size: number };
}

/** Files the runtime needs before a snapshot is worth handing back. */
const REQUIRED_ENTRIES = [
  "rl_agent_config.json",
  "encoder/config.json",
  "model.safetensors",
  "tokenizer/tokenizer.json",
  "tokenizer/tokenizer_config.json",
];

const COMMIT_SHA = /^[0-9a-f]{40}$/;

/**
 * The cache root, following upstream's precedence: `HF_HUB_CACHE`, then `HUGGINGFACE_HUB_CACHE`,
 * then `HF_HOME` + `/hub`, then `$XDG_CACHE_HOME/huggingface/hub`, then `~/.cache/huggingface/hub`.
 *
 * Exported with an injectable environment so the precedence itself is testable without mutating
 * `process.env` globally.
 */
export function defaultCacheDir(env: Record<string, string | undefined> = process.env): string {
  if (env[CACHE_ENV_VARS.hubCache]) return resolve(env[CACHE_ENV_VARS.hubCache] as string);
  if (env[CACHE_ENV_VARS.legacyHubCache]) {
    return resolve(env[CACHE_ENV_VARS.legacyHubCache] as string);
  }
  const home = env[CACHE_ENV_VARS.home];
  if (home) return join(resolve(home), "hub");
  const xdg = env[CACHE_ENV_VARS.xdgCache] ?? join(homedir(), ".cache");
  return join(xdg, "huggingface", "hub");
}

/**
 * A repo id to its cache directory name: `aac6fef/laya-mlx` → `models--aac6fef--laya-mlx`.
 *
 * Rejects anything that is not exactly `owner/name`, so a URL or a bare name fails here rather than
 * producing a cache directory nobody will ever find again.
 */
export function repoFolderName(repoId: string): string {
  const parts = repoId.split("/");
  if (parts.length !== 2 || parts.some((part) => part.trim() === "")) {
    throw new Error(
      `Invalid repository id ${JSON.stringify(repoId)}: expected "owner/name". ` +
        "Pass a local directory path instead if the model is already on disk.",
    );
  }
  if (repoId.includes("..")) {
    throw new Error(`Invalid repository id ${JSON.stringify(repoId)}: it may not contain ".."`);
  }
  return `models--${parts[0]}--${parts[1]}`;
}

/**
 * Resolve a locally cached snapshot without any network access, or `null` if there is none.
 *
 * Revision resolution, in order: an explicit commit sha names its snapshot directory directly; a
 * branch or tag goes through `refs/<revision>`, which is what both implementations write. This is
 * the sync path behind `load(repoId)`, which is why it may not fall back to the network.
 */
export function resolveCachedModel(repoId: string, options: HubOptions = {}): ResolvedModel | null {
  const cacheDir = options.cacheDir ?? defaultCacheDir();
  const repoDir = join(cacheDir, repoFolderName(repoId));
  if (!existsSync(repoDir)) return null;

  const revision = options.revision ?? "main";
  const candidates: string[] = [];
  if (COMMIT_SHA.test(revision)) candidates.push(revision);
  else candidates.push(revision);
  const refFile = join(repoDir, "refs", revision);
  if (existsSync(refFile)) {
    const pinned = readFileSync(refFile, "utf8").trim();
    if (pinned !== "") candidates.unshift(pinned);
  }

  for (const candidate of candidates) {
    const snapshot = join(repoDir, "snapshots", candidate);
    if (existsSync(snapshot) && isDirectory(snapshot)) {
      return {
        repoId,
        revision: candidate,
        path: snapshot,
        fromCache: true,
        files: countFiles(snapshot),
      };
    }
  }
  return null;
}

/**
 * Resolve a repo to a local snapshot, downloading whatever is missing.
 *
 * With `offline: true`, a cache miss throws instead of fetching; the message names the cache
 * directory and the revision so the user can tell whether their `HF_HOME` is the one they expected.
 */
export async function downloadModel(
  repoId: string,
  options: HubOptions = {},
): Promise<ResolvedModel> {
  const folder = repoFolderName(repoId);
  const cacheDir = options.cacheDir ?? defaultCacheDir();
  const repoDir = join(cacheDir, folder);
  const endpoint = (options.endpoint ?? DEFAULT_ENDPOINT).replace(/\/+$/, "");
  const revision = options.revision ?? "main";

  // A complete cached snapshot short-circuits **before any network use**, including for a branch name.
  // `refs/<revision>` is what makes that possible, and trusting it is the point of the ref file:
  // re-resolving `main` on every call would add a round trip to a path whose whole purpose is to
  // avoid one. `force: true` opts back into the upstream check.
  const cachedSnapshot = resolveCachedModel(repoId, { cacheDir, revision });
  const cacheIsComplete =
    options.force !== true && cachedSnapshot !== null && hasRequiredEntries(cachedSnapshot.path);

  if (cacheIsComplete && cachedSnapshot !== null) {
    report(options, {
      phase: "done",
      filesDone: cachedSnapshot.files,
      filesTotal: cachedSnapshot.files,
      bytesDone: 0,
      bytesTotal: 0,
    });
    return cachedSnapshot;
  }

  if (options.offline === true) {
    const detail =
      cachedSnapshot === null
        ? `there is no snapshot for revision ${revision}`
        : `${cachedSnapshot.path} is missing ` +
          REQUIRED_ENTRIES.filter((entry) => !existsSync(join(cachedSnapshot.path, entry))).join(
            ", ",
          );
    throw new Error(
      `offline: ${repoId}@${revision} is not usable from the cache at ${cacheDir} — ${detail}.\n` +
        "Either run without `offline` once to populate it, or point HF_HOME (or `cacheDir`) at the " +
        "cache you expected to use.",
    );
  }

  report(options, {
    phase: "resolving",
    filesDone: 0,
    filesTotal: 0,
    bytesDone: 0,
    bytesTotal: 0,
  });

  // A commit sha needs no lookup; a branch or tag does.
  const commit = COMMIT_SHA.test(revision)
    ? revision
    : await resolveRevision(endpoint, repoId, revision, options);

  const snapshotDir = join(repoDir, "snapshots", commit);
  if (options.force !== true && existsSync(snapshotDir) && hasRequiredEntries(snapshotDir)) {
    report(options, {
      phase: "done",
      filesDone: 0,
      filesTotal: 0,
      bytesDone: 0,
      bytesTotal: 0,
    });
    return {
      repoId,
      revision: commit,
      path: snapshotDir,
      fromCache: true,
      files: countFiles(snapshotDir),
    };
  }

  report(options, { phase: "listing", filesDone: 0, filesTotal: 0, bytesDone: 0, bytesTotal: 0 });
  const files = await listFiles(endpoint, repoId, commit, options);

  // Download `SHA256SUMS` first when the repo publishes one, so verification data is in hand before
  // the payload arrives. Repos that do not are simply not verified beyond their blob name.
  const sumsEntry = files.find((entry) => entry.path === "SHA256SUMS");
  const expectedSums = new Map<string, string>();
  if (sumsEntry !== undefined) {
    const raw = await fetchText(endpoint, repoId, commit, "SHA256SUMS", options);
    // `parseSha256Sums` returns `[path, digest]`. Getting this order backwards silently produces an
    // empty lookup, i.e. verification that never runs — which is exactly how it failed the first time.
    for (const [path, digest] of parseSha256Sums(raw)) expectedSums.set(path, digest);
  }

  const bytesTotal = files.reduce((total, entry) => total + (entry.size ?? 0), 0);
  let bytesDone = 0;
  let filesDone = 0;
  let downloaded = 0;
  const total = files.length;

  for (const entry of files) {
    if (existsSync(join(snapshotDir, entry.path))) {
      filesDone += 1;
      bytesDone += entry.size ?? 0;
      report(options, {
        phase: "downloading",
        file: entry.path,
        filesDone,
        filesTotal: total,
        bytesDone,
        bytesTotal,
      });
      continue;
    }
    await downloadFile(
      {
        endpoint,
        repoId,
        commit,
        entry,
        repoDir,
        snapshotDir,
        // Verified before the blob is published, not after. A post-hoc sweep would leave a
        // complete-looking snapshot behind on failure, and the next run would accept it without
        // checking — a tampered download would be blessed by the run that caught it.
        expectedSha256: expectedSums.get(entry.path),
      },
      options,
      (chunkBytes) => {
        bytesDone += chunkBytes;
        report(options, {
          phase: "downloading",
          file: entry.path,
          filesDone,
          filesTotal: total,
          bytesDone,
          bytesTotal,
        });
      },
    );
    filesDone += 1;
    downloaded += 1;
  }

  const missing = REQUIRED_ENTRIES.filter((entry) => !existsSync(join(snapshotDir, entry)));
  if (missing.length > 0) {
    throw new Error(
      `${repoId}@${commit} does not look like a Laya checkpoint: missing ${missing.join(", ")}.`,
    );
  }

  // Record the revision so a later offline run asking for the same branch resolves without the
  // network. Upstream keys this file by revision name, not by commit.
  mkdirSync(join(repoDir, "refs"), { recursive: true });
  writeFileSync(join(repoDir, "refs", revision), `${commit}\n`);

  report(options, {
    phase: "done",
    filesDone: total,
    filesTotal: total,
    bytesDone: bytesTotal,
    bytesTotal,
  });
  return {
    repoId,
    revision: commit,
    path: snapshotDir,
    fromCache: downloaded === 0,
    files: total,
  };
}

// ---- internals ---------------------------------------------------------

function report(options: HubOptions, event: HubProgress): void {
  options.onProgress?.(event);
}

function isDirectory(path: string): boolean {
  try {
    return statSync(path).isDirectory();
  } catch {
    return false;
  }
}

function countFiles(dir: string): number {
  if (!isDirectory(dir)) return 0;
  let count = 0;
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    if (entry.isDirectory()) count += countFiles(join(dir, entry.name));
    else count += 1;
  }
  return count;
}

function hasRequiredEntries(snapshotDir: string): boolean {
  return REQUIRED_ENTRIES.every((entry) => existsSync(join(snapshotDir, entry)));
}

/** `HEAD`-free revision lookup: the model endpoint returns the commit in `sha`. */
async function resolveRevision(
  endpoint: string,
  repoId: string,
  revision: string,
  options: HubOptions,
): Promise<string> {
  const url = `${endpoint}/api/models/${repoId}/revision/${encodeURIComponent(revision)}`;
  const response = await request(options, url, { accept: "application/json" });
  if (!response.ok) {
    throw new Error(
      `Could not resolve ${repoId}@${revision}: ${response.status} ${response.statusText} (${url}).\n` +
        "Check the repository id and revision, and set `token` for a private or gated repo.",
    );
  }
  const body = (await response.json()) as { sha?: unknown };
  if (typeof body.sha !== "string" || !COMMIT_SHA.test(body.sha)) {
    throw new Error(`The Hub returned no commit sha for ${repoId}@${revision}`);
  }
  return body.sha;
}

/** The repo tree, following `Link: rel=next` pagination. */
async function listFiles(
  endpoint: string,
  repoId: string,
  commit: string,
  options: HubOptions,
): Promise<TreeEntry[]> {
  const entries: TreeEntry[] = [];
  let url: string | null = `${endpoint}/api/models/${repoId}/tree/${commit}?recursive=true`;
  while (url !== null) {
    const response: Response = await request(options, url, { accept: "application/json" });
    if (!response.ok) {
      throw new Error(
        `Could not list ${repoId}@${commit}: ${response.status} ${response.statusText} (${url})`,
      );
    }
    const page = (await response.json()) as TreeEntry[];
    if (!Array.isArray(page)) throw new Error(`Unexpected tree response for ${repoId}@${commit}`);
    entries.push(...page.filter((entry) => entry.type === "file"));
    url = nextLink(response.headers.get("link"), endpoint);
  }
  if (entries.length === 0) {
    throw new Error(`${repoId}@${commit} has no files`);
  }
  return entries;
}

/**
 * Parse a `Link` header for the `rel="next"` URL. Returns `null` when there is none.
 *
 * The URL test is a scheme check, not `path.isAbsolute`: `isAbsolute("https://x/y")` is `false`,
 * because it asks whether the string is an absolute *filesystem* path on this platform. Using it here
 * silently prefixed the endpoint onto an already-absolute URL, producing
 * `https://huggingface.cohttps://huggingface.co/api/...` on every paginated listing.
 */
export function nextLink(header: string | null, endpoint: string): string | null {
  if (header === null) return null;
  for (const part of header.split(",")) {
    const match = /<([^>]+)>\s*;\s*rel="next"/.exec(part);
    const target = match?.[1];
    if (target === undefined) continue;
    return /^[a-z][a-z0-9+.-]*:\/\//i.test(target) ? target : `${endpoint}${target}`;
  }
  return null;
}

/**
 * Parse a `sha256sum`-style manifest: `<64 hex>  <path>`, with `*` marking binary mode.
 *
 * Blank lines and comments are ignored. Anything else is treated as a malformed manifest and
 * skipped rather than guessed at — a wrong digest is worse than no digest, because it turns a
 * verification failure into a confusing one.
 */
export function parseSha256Sums(text: string): [path: string, digest: string][] {
  const out: [string, string][] = [];
  for (const line of text.split("\n")) {
    const trimmed = line.trim();
    if (trimmed === "" || trimmed.startsWith("#")) continue;
    const match = /^([0-9a-f]{64})\s+\*?(.+)$/.exec(trimmed);
    if (match?.[1] === undefined || match[2] === undefined) continue;
    out.push([match[2].trim(), match[1].toLowerCase()]);
  }
  return out;
}

function verifySha256Sums(entryPath: string, expected: string, actual: string): void {
  if (expected === actual) return;
  throw new Error(
    `${entryPath}: SHA256SUMS says ${expected}, downloaded ${actual}. ` +
      "The transfer is corrupt or the manifest is stale; nothing was cached.",
  );
}

async function fetchText(
  endpoint: string,
  repoId: string,
  commit: string,
  path: string,
  options: HubOptions,
): Promise<string> {
  const response = await fetchFile(endpoint, repoId, commit, path, options);
  if (!response.ok) {
    throw new Error(`Could not fetch ${path}: ${response.status} ${response.statusText}`);
  }
  return await response.text();
}

/**
 * `GET /{repo}/resolve/{commit}/{path}`, returning the **CDN** response.
 *
 * The resolve URL answers with a redirect to a signed CDN URL. `redirect: "manual"` lets the token be
 * dropped before following it, so an access token is never handed to the storage provider. If the
 * fetch implementation does not expose the redirect (a browser-style opaque response), this falls
 * back to following automatically and notes the caveat in the returned response's header, rather
 * than silently sending credentials somewhere unexpected.
 */
async function fetchFile(
  endpoint: string,
  repoId: string,
  commit: string,
  path: string,
  options: HubOptions,
): Promise<Response> {
  const url = `${endpoint}/${repoId}/resolve/${commit}/${path.split("/").map(encodeURIComponent).join("/")}`;
  const doFetch = options.fetch ?? globalThis.fetch;

  const manual = await request(options, url, { redirect: "manual" });
  if (manual.status >= 300 && manual.status < 400) {
    const location = manual.headers.get("location");
    if (location !== null) {
      // The Hub answers with a **relative** Location for its own resolve cache
      // (`/api/resolve-cache/models/...`) and an absolute one for the LFS CDN, so it must be
      // resolved against the request URL rather than handed to fetch as-is. Passing a relative value
      // through fails with `Failed to parse URL from /api/resolve-cache/...`.
      const target = new URL(location, url).toString();
      // Anonymous: the target is pre-signed or same-origin, so it needs no credentials.
      return await doFetch(target);
    }
  }
  if (manual.type === "opaqueredirect" || manual.status === 0) {
    // No way to read the redirect: follow it, accepting that the token travels with the request.
    return await request(options, url, { redirect: "follow" });
  }
  return manual;
}

/** A request to the endpoint itself, with auth and a user agent attached. */
async function request(
  options: HubOptions,
  url: string,
  init: { accept?: string; redirect?: "follow" | "error" | "manual" } = {},
): Promise<Response> {
  const doFetch = options.fetch ?? globalThis.fetch;
  const headers: Record<string, string> = { "user-agent": userAgent() };
  if (init.accept !== undefined) headers.accept = init.accept;
  if (options.token !== undefined && options.token !== "") {
    headers.authorization = `Bearer ${options.token}`;
  }
  return await doFetch(url, { headers, redirect: init.redirect ?? "follow" });
}

/** Identify the client honestly; the Hub rate-limits anonymous traffic per user agent. */
function userAgent(): string {
  return `@fllstck/mlayax (node ${process.version})`;
}

/**
 * Download one file into `blobs/`, then link it into the snapshot.
 *
 * The blob is named by the LFS `sha256` for large files and by the git blob sha1 otherwise, matching
 * upstream. Writes are atomic: bytes go to `<blob>.incomplete` and are renamed into place, and the
 * content hash is computed while streaming so a corrupt transfer is rejected before the rename.
 */
async function downloadFile(
  context: {
    endpoint: string;
    repoId: string;
    commit: string;
    entry: TreeEntry;
    repoDir: string;
    snapshotDir: string;
    /** Digest from the repo's own `SHA256SUMS`, when it publishes one. */
    expectedSha256: string | undefined;
  },
  options: HubOptions,
  onBytes: (bytes: number) => void,
): Promise<string> {
  const { entry, repoDir, snapshotDir } = context;
  const blobName =
    entry.lfs?.oid ?? entry.oid ?? createHash("sha1").update(entry.path).digest("hex");
  const blobPath = join(repoDir, "blobs", blobName);
  const target = join(snapshotDir, entry.path);

  mkdirSync(dirname(blobPath), { recursive: true });
  mkdirSync(dirname(target), { recursive: true });

  const response = await fetchFile(
    context.endpoint,
    context.repoId,
    context.commit,
    entry.path,
    options,
  );
  if (!response.ok || response.body === null) {
    throw new Error(
      `Could not download ${entry.path} from ${context.repoId}@${context.commit}: ` +
        `${response.status} ${response.statusText}`,
    );
  }

  const incomplete = `${blobPath}.incomplete`;
  const hash = createHash("sha256");
  let bytes = 0;
  try {
    await pipeline(
      Readable.fromWeb(response.body as Parameters<typeof Readable.fromWeb>[0]),
      async function* (source: AsyncIterable<Uint8Array>) {
        for await (const chunk of source) {
          hash.update(chunk);
          bytes += chunk.length;
          onBytes(chunk.length);
          yield chunk;
        }
      },
      createWriteStream(incomplete),
    );

    // Verify before publishing. An LFS blob's name *is* the sha256 of its content, so a truncated or
    // mangled transfer fails here rather than as a safetensors parse error three layers away — and,
    // because the check precedes the rename, a failed verification leaves the snapshot incomplete,
    // so the next run retries rather than accepting it.
    const digest = hash.digest("hex");
    if (entry.lfs !== undefined && digest !== entry.lfs.oid) {
      throw new Error(
        `${entry.path}: expected sha256 ${entry.lfs.oid}, got ${digest}. ` +
          "The download is corrupt; it has not been cached.",
      );
    }
    if (context.expectedSha256 !== undefined) {
      verifySha256Sums(entry.path, context.expectedSha256, digest);
    }
    if (entry.size !== undefined && bytes !== entry.size) {
      throw new Error(
        `${entry.path}: expected ${entry.size} bytes, got ${bytes}. The download is incomplete.`,
      );
    }

    renameSync(incomplete, blobPath);
  } catch (error) {
    // A half-written file must not survive as `<blob>.incomplete` for a later run to trip over.
    try {
      unlinkSync(incomplete);
    } catch {
      // Already gone, or never created.
    }
    throw error;
  }
  linkIntoSnapshot(blobPath, target);
  return blobPath;
}

/**
 * Point `snapshots/<commit>/<path>` at the blob with a *relative* symlink.
 *
 * Relative matters: the cache is relocatable (and `HF_HOME` gets moved between machines), so an
 * absolute target would break every snapshot the moment the cache directory moved.
 *
 * The link is computed from the *file's own directory*, not from the snapshot root — a nested entry
 * like `encoder/config.json` sits one level deeper, so a fixed `../../blobs/<name>` would resolve to
 * `snapshots/blobs/<name>` and dangle. That was a real bug here, caught by the "missing
 * encoder/config.json" assertion rather than by anything in the download path itself.
 */
function linkIntoSnapshot(blobPath: string, target: string): void {
  const linkTarget = relative(dirname(target), blobPath);
  // Idempotent: a stale or broken entry is replaced rather than tripping symlinkSync's EEXIST.
  try {
    unlinkSync(target);
  } catch {
    // Nothing to replace.
  }
  symlinkSync(linkTarget, target, "file");
}
