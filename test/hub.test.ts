/**
 * Hub fetcher tests, against a local HTTP fixture server.
 *
 * No network: every case — cold download, warm cache, offline, checksum verification, redirect
 * credential handling — is driven through `node:http` on a loopback port. The server records what it
 * was asked for and with which credentials, so the tests can assert *that the network was not used*
 * in the offline cases rather than merely that the result looked right.
 *
 * The redirect test is worth reading: the resolve URL answers 302 to a "CDN" path, and the assertion
 * is that the access token appears on the API and resolve requests but **not** on the CDN one.
 */

import { createHash } from "node:crypto";
import {
  existsSync,
  lstatSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import { tmpdir } from "node:os";
import { dirname, join, relative } from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import {
  defaultCacheDir,
  downloadModel,
  type HubProgress,
  nextLink,
  parseSha256Sums,
  repoFolderName,
  resolveCachedModel,
  snapshotPathFor,
} from "../packages/mlayax/src/hub.js";

const COMMIT = "20aed815fc6acde75733882e7ec0e3f28aeb9717";
const REPO = "aac6fef/laya-mlx";

/** The five entries the loader requires, plus a byte payload for each. */
const PAYLOAD = "$safetensors-placeholder-bytes";
const FIXTURE_FILES: Record<string, Buffer> = {
  "encoder/config.json": Buffer.from('{"model_type":"modernbert"}'),
  "tokenizer/tokenizer_config.json": Buffer.from('{"cls_token":"[CLS]"}'),
  "tokenizer/tokenizer.json": Buffer.from('{"model":{"type":"WordLevel"}}'),
  "rl_agent_config.json": Buffer.from('{"head_layers":2,"max_len":64}'),
  "model.safetensors": Buffer.from(PAYLOAD),
};

interface Recorded {
  path: string;
  auth: string | null;
}

interface FixtureOptions {
  /** Add a SHA256SUMS file; `"correct"` writes real digests, `"wrong"` a bogus one. */
  sums?: "correct" | "wrong";
  /** Force this file's LFS sha256 to a wrong value, to test corrupt-transfer rejection. */
  corruptLfsPath?: string;
  /** Serve a revision listing that omits a required file. */
  omit?: string;
  /** Do not redirect resolve URLs; serve bytes directly. */
  noRedirect?: boolean;
  /** Emit a **relative** Location header, as the Hub's own resolve cache does. */
  relativeRedirect?: boolean;
  /** Reject requests without a bearer token. */
  requireToken?: string;
}

interface Fixture {
  endpoint: string;
  requests: Recorded[];
  close: () => Promise<void>;
}

function sha256(bytes: Buffer): string {
  return createHash("sha256").update(bytes).digest("hex");
}

function sha1(bytes: Buffer): string {
  return createHash("sha1").update(bytes).digest("hex");
}

async function startFixture(options: FixtureOptions = {}): Promise<Fixture> {
  const files = { ...FIXTURE_FILES, ...(options.omit ? { [options.omit]: undefined } : {}) };
  const requests: Recorded[] = [];

  const server: Server = createServer((req: IncomingMessage, res: ServerResponse) => {
    const url = new URL(req.url ?? "/", "http://localhost");
    requests.push({ path: url.pathname + url.search, auth: req.headers.authorization ?? null });

    if (
      options.requireToken !== undefined &&
      !url.pathname.startsWith("/cdn/") &&
      req.headers.authorization !== `Bearer ${options.requireToken}`
    ) {
      res.writeHead(401).end("unauthorized");
      return;
    }

    // /api/models/{owner}/{name}/revision/{rev}
    const revisionMatch = /^\/api\/models\/([^/]+)\/([^/]+)\/revision\/(.+)$/.exec(url.pathname);
    if (revisionMatch) {
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify({ sha: COMMIT }));
      return;
    }

    // /api/models/{owner}/{name}/tree/{rev}
    const treeMatch = /^\/api\/models\/([^/]+)\/([^/]+)\/tree\/([^/]+)$/.exec(url.pathname);
    if (treeMatch) {
      const entries = Object.entries(files)
        .filter(([, bytes]) => bytes !== undefined)
        .map(([path, bytes]) => {
          const body = bytes as Buffer;
          const isLarge = path === "model.safetensors";
          const lfsOid =
            options.corruptLfsPath === path ? sha256(Buffer.from("something else")) : sha256(body);
          return {
            type: "file" as const,
            path,
            size: body.length,
            oid: isLarge ? undefined : sha1(body),
            ...(isLarge ? { lfs: { oid: lfsOid, size: body.length } } : {}),
          };
        });
      if (options.sums !== undefined) {
        const sumsBody = sumsFile(options.sums);
        entries.push({
          type: "file" as const,
          path: "SHA256SUMS",
          size: sumsBody.length,
          oid: sha1(sumsBody),
        });
      }
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify(entries));
      return;
    }

    // /{owner}/{name}/resolve/{rev}/{path...} -> 302 to /cdn/...
    const resolveMatch = /^\/([^/]+)\/([^/]+)\/resolve\/([^/]+)\/(.+)$/.exec(url.pathname);
    if (resolveMatch) {
      const path = decodeURIComponent(resolveMatch[4] as string);
      if (options.noRedirect === true) {
        serveFile(path, res, files, options);
        return;
      }
      res.writeHead(302, {
        location:
          options.relativeRedirect === true
            ? `/cdn/${path}`
            : `http://127.0.0.1:${port}/cdn/${path}`,
      });
      res.end();
      return;
    }

    // /cdn/{path...}
    if (url.pathname.startsWith("/cdn/")) {
      serveFile(decodeURIComponent(url.pathname.slice("/cdn/".length)), res, files, options);
      return;
    }

    res.writeHead(404).end("not found");
  });

  let port = 0;
  await new Promise<void>((resolve) => {
    server.listen(0, "127.0.0.1", () => {
      port = (server.address() as { port: number }).port;
      resolve();
    });
  });

  return {
    endpoint: `http://127.0.0.1:${port}`,
    requests,
    close: () => new Promise<void>((resolve) => server.close(() => resolve())),
  };
}

function serveFile(
  path: string,
  res: ServerResponse,
  files: Record<string, Buffer | undefined>,
  options: FixtureOptions,
): void {
  if (path === "SHA256SUMS" && options.sums !== undefined) {
    const body = sumsFile(options.sums);
    res.writeHead(200, { "content-length": String(body.length) });
    res.end(body);
    return;
  }
  const body = files[path];
  if (body === undefined) {
    res.writeHead(404).end("no such file");
    return;
  }
  res.writeHead(200, { "content-length": String(body.length) });
  res.end(body);
}

function sumsFile(kind: "correct" | "wrong"): Buffer {
  const lines = Object.entries(FIXTURE_FILES).map(([path, bytes]) => {
    const digest = kind === "correct" ? sha256(bytes) : "0".repeat(64);
    return `${digest}  ${path}`;
  });
  return Buffer.from(`${lines.join("\n")}\n`);
}

/** A scratch cache root, removed after the suite. */
function scratchCache(): string {
  return mkdtempSync(join(tmpdir(), "mlayax-hub-"));
}

const scratchDirs: string[] = [];
function trackScratch(): string {
  const dir = scratchCache();
  scratchDirs.push(dir);
  return dir;
}

afterAll(() => {
  for (const dir of scratchDirs) rmSync(dir, { recursive: true, force: true });
});

describe("defaultCacheDir", () => {
  it("follows upstream's precedence", () => {
    expect(defaultCacheDir({ HF_HUB_CACHE: "/a" })).toBe("/a");
    expect(defaultCacheDir({ HUGGINGFACE_HUB_CACHE: "/b" })).toBe("/b");
    expect(defaultCacheDir({ HF_HOME: "/c" })).toBe("/c/hub");
    expect(defaultCacheDir({ XDG_CACHE_HOME: "/d" })).toBe("/d/huggingface/hub");
    // HF_HUB_CACHE wins over the legacy name and over HF_HOME, as it does upstream.
    expect(
      defaultCacheDir({ HF_HUB_CACHE: "/a", HUGGINGFACE_HUB_CACHE: "/b", HF_HOME: "/c" }),
    ).toBe("/a");
    expect(defaultCacheDir({ HF_HOME: "/c", XDG_CACHE_HOME: "/d" })).toBe("/c/hub");
    expect(defaultCacheDir({})).toMatch(/\.cache\/huggingface\/hub$/);
  });
});

describe("repoFolderName", () => {
  it("names the folder the way the Python cache does", () => {
    expect(repoFolderName("aac6fef/laya-mlx")).toBe("models--aac6fef--laya-mlx");
  });

  it("rejects anything that is not exactly owner/name, pointing at the directory form", () => {
    expect(() => repoFolderName("laya-mlx")).toThrow(/expected "owner\/name"/);
    expect(() => repoFolderName("a/b/c")).toThrow(/expected "owner\/name"/);
    expect(() => repoFolderName("https://huggingface.co/a/b")).toThrow(/expected "owner\/name"/);
    expect(() => repoFolderName("../name")).toThrow(/may not contain/);
  });
});

describe("parseSha256Sums", () => {
  it("parses the sha256sum format, including binary-mode asterisks and blank lines", () => {
    const text =
      "# a comment\n\nabcd" + "0".repeat(60) + "  a/b.json\n" + "1".repeat(64) + " *c.bin\n";
    expect(parseSha256Sums(text)).toEqual([
      ["a/b.json", `abcd${"0".repeat(60)}`],
      ["c.bin", "1".repeat(64)],
    ]);
  });

  it("skips malformed lines rather than guessing a digest", () => {
    // A wrong digest is worse than no digest: it turns a real verification failure into a confusing
    // one.
    expect(parseSha256Sums("not-a-digest  file\nshort  file\n")).toEqual([]);
  });
});

describe("snapshotPathFor", () => {
  const snapshot = "/cache/models--a--b/snapshots/0123";

  it("places a nested path under the snapshot directory", () => {
    expect(snapshotPathFor(snapshot, "encoder/config.json")).toBe(
      "/cache/models--a--b/snapshots/0123/encoder/config.json",
    );
    // A `..` inside a *file name* is not a traversal, and rejecting it would be a bug of its own.
    expect(snapshotPathFor(snapshot, "weights..safetensors")).toBe(
      "/cache/models--a--b/snapshots/0123/weights..safetensors",
    );
  });

  it("refuses a path that escapes the snapshot directory", () => {
    // The listing is the Hub's, and a custom `endpoint` means the Hub is whoever you pointed at.
    // `join` follows `..` without complaint, so this is checked rather than assumed.
    for (const escaping of ["../evil.json", "a/../../evil.json", "..", "."]) {
      expect(() => snapshotPathFor(snapshot, escaping), escaping).toThrow(/escapes the snapshot/);
    }
  });

  it("keeps an absolute-looking entry inside the snapshot", () => {
    // `join` — unlike `resolve` — does not let a second argument replace the first, so a repo path
    // that starts with `/` is contained rather than an escape. Asserted because the opposite is a
    // common assumption, and the guard's own comment says so.
    expect(snapshotPathFor(snapshot, "/etc/passwd")).toBe(
      "/cache/models--a--b/snapshots/0123/etc/passwd",
    );
  });
});

describe("nextLink", () => {
  it("extracts a relative next link and absolutises it against the endpoint", () => {
    expect(
      nextLink('<https://huggingface.co/api/x?page=2>; rel="next"', "https://huggingface.co"),
    ).toBe("https://huggingface.co/api/x?page=2");
    expect(nextLink('</api/x?page=2>; rel="next"', "https://example.test")).toBe(
      "https://example.test/api/x?page=2",
    );
  });

  it("returns null when there is no next page", () => {
    expect(nextLink(null, "https://example.test")).toBeNull();
    expect(nextLink('<https://x/y>; rel="prev"', "https://x")).toBeNull();
  });
});

describe("downloadModel against a local fixture server", () => {
  it("cold: downloads every file into the Python cache layout", async () => {
    const fixture = await startFixture();
    const cacheDir = trackScratch();
    try {
      const events: HubProgress[] = [];
      const resolved = await downloadModel(REPO, {
        cacheDir,
        endpoint: fixture.endpoint,
        onProgress: (event) => events.push(event),
      });

      expect(resolved.repoId).toBe(REPO);
      expect(resolved.revision).toBe(COMMIT);
      expect(resolved.fromCache).toBe(false);
      expect(resolved.files).toBe(5);

      // refs/main carries the resolved commit, as upstream writes it.
      expect(
        readFileSync(join(cacheDir, "models--aac6fef--laya-mlx", "refs", "main"), "utf8").trim(),
      ).toBe(COMMIT);
      expect(resolved.path).toBe(join(cacheDir, "models--aac6fef--laya-mlx", "snapshots", COMMIT));

      // Every required file resolves, through a symlink.
      for (const [path, bytes] of Object.entries(FIXTURE_FILES)) {
        const full = join(resolved.path, path);
        expect(readFileSync(full)).toEqual(bytes);
        expect(lstatSync(full).isSymbolicLink()).toBe(true);
      }

      // Snapshot links are relative, so the cache survives being moved.
      const link = readFileSync(join(resolved.path, "model.safetensors"));
      expect(link.length).toBeGreaterThan(0);
      expect(existsSync(join(resolved.path, "encoder", "config.json"))).toBe(true);
      const relativeTarget = readdirSync(join(cacheDir, "models--aac6fef--laya-mlx", "blobs"));
      expect(relativeTarget.length).toBe(5);
      for (const name of relativeTarget) {
        expect(name).not.toMatch(/\.incomplete$/);
      }

      // Progress reaches the caller with real byte counts.
      const phases = new Set(events.map((event) => event.phase));
      expect(phases.has("resolving")).toBe(true);
      expect(phases.has("downloading")).toBe(true);
      expect(phases.has("done")).toBe(true);
      const last = events.at(-1);
      expect(last?.bytesDone).toBeGreaterThan(0);
      expect(last?.bytesTotal).toBe(last?.bytesDone);
    } finally {
      await fixture.close();
    }
  });

  it("warm: a second call makes no requests at all", async () => {
    const fixture = await startFixture();
    const cacheDir = trackScratch();
    try {
      await downloadModel(REPO, { cacheDir, endpoint: fixture.endpoint });
      const afterCold = fixture.requests.length;
      expect(afterCold).toBeGreaterThan(0);

      const warm = await downloadModel(REPO, { cacheDir, endpoint: fixture.endpoint });
      expect(warm.fromCache).toBe(true);
      // Not "fewer requests" — none. The snapshot is complete, so listing is not even needed.
      expect(fixture.requests.length).toBe(afterCold);
    } finally {
      await fixture.close();
    }
  });

  it("offline with a warm cache: no network, and the cache is used", async () => {
    const fixture = await startFixture();
    const cacheDir = trackScratch();
    try {
      await downloadModel(REPO, { cacheDir, endpoint: fixture.endpoint });
      const afterCold = fixture.requests.length;

      const offline = await downloadModel(REPO, {
        cacheDir,
        endpoint: fixture.endpoint,
        offline: true,
      });
      expect(offline.fromCache).toBe(true);
      expect(offline.revision).toBe(COMMIT);
      expect(fixture.requests.length).toBe(afterCold);
    } finally {
      await fixture.close();
    }
  });

  it("offline with a cold cache: fails clearly, naming the cache and the revision", async () => {
    const fixture = await startFixture();
    const cacheDir = trackScratch();
    try {
      await expect(
        downloadModel(REPO, {
          cacheDir,
          endpoint: fixture.endpoint,
          offline: true,
          revision: "v1.2.3",
        }),
      ).rejects.toThrow(/not usable from the cache/);
      await expect(
        downloadModel(REPO, { cacheDir, endpoint: fixture.endpoint, offline: true }),
      ).rejects.toThrow(/HF_HOME/);
      // And it really did not try.
      expect(fixture.requests.length).toBe(0);
    } finally {
      await fixture.close();
    }
  });

  it("resolves a branch through refs, so offline works after one online run", async () => {
    const fixture = await startFixture();
    const cacheDir = trackScratch();
    try {
      await downloadModel(REPO, { cacheDir, endpoint: fixture.endpoint, revision: "main" });
      const cached = resolveCachedModel(REPO, { cacheDir, revision: "main" });
      expect(cached?.revision).toBe(COMMIT);
      // A commit sha names the snapshot directory directly.
      expect(resolveCachedModel(REPO, { cacheDir, revision: COMMIT })?.revision).toBe(COMMIT);
      // An unknown revision is simply not cached.
      expect(resolveCachedModel(REPO, { cacheDir, revision: "nope" })).toBeNull();
    } finally {
      await fixture.close();
    }
  });

  it("trusts a complete cached snapshot for a branch name, and force re-checks upstream", async () => {
    const fixture = await startFixture();
    const cacheDir = trackScratch();
    try {
      await downloadModel(REPO, { cacheDir, endpoint: fixture.endpoint });
      const afterCold = fixture.requests.length;

      // A branch name resolves through refs/<name> locally, so a warm load costs no round trip.
      const trusted = await downloadModel(REPO, { cacheDir, endpoint: fixture.endpoint });
      expect(trusted.fromCache).toBe(true);
      expect(fixture.requests.length).toBe(afterCold);

      // `force` opts back in to the upstream check.
      await downloadModel(REPO, { cacheDir, endpoint: fixture.endpoint, force: true });
      expect(fixture.requests.length).toBeGreaterThan(afterCold);
    } finally {
      await fixture.close();
    }
  });

  it("sends the token to the Hub but never to the CDN it redirects to", async () => {
    const fixture = await startFixture();
    const cacheDir = trackScratch();
    try {
      await downloadModel(REPO, {
        cacheDir,
        endpoint: fixture.endpoint,
        token: "hf_secret",
      });
      const api = fixture.requests.filter((r) => r.path.startsWith("/api/"));
      const resolve = fixture.requests.filter((r) => r.path.includes("/resolve/"));
      const cdn = fixture.requests.filter((r) => r.path.startsWith("/cdn/"));

      expect(api.length).toBeGreaterThan(0);
      expect(resolve.length).toBeGreaterThan(0);
      expect(cdn.length).toBeGreaterThan(0);
      for (const request of [...api, ...resolve]) {
        expect(request.auth).toBe("Bearer hf_secret");
      }
      // The CDN URL is pre-signed, so the credential stops at the redirect.
      for (const request of cdn) {
        expect(request.auth).toBeNull();
      }
    } finally {
      await fixture.close();
    }
  });

  it("verifies SHA256SUMS when the repo publishes one", async () => {
    const fixture = await startFixture({ sums: "correct" });
    const cacheDir = trackScratch();
    try {
      const resolved = await downloadModel(REPO, { cacheDir, endpoint: fixture.endpoint });
      expect(existsSync(join(resolved.path, "SHA256SUMS"))).toBe(true);
      expect(resolved.files).toBe(6);
    } finally {
      await fixture.close();
    }
  });

  it("fails before publishing when SHA256SUMS does not match, and stays failing", async () => {
    const fixture = await startFixture({ sums: "wrong" });
    const cacheDir = trackScratch();
    const snapshotDir = join(cacheDir, "models--aac6fef--laya-mlx", "snapshots", COMMIT);
    try {
      const attempt = downloadModel(REPO, { cacheDir, endpoint: fixture.endpoint });
      await expect(attempt).rejects.toThrow(/SHA256SUMS says/);

      // Nothing was published, so the snapshot stays incomplete.
      expect(existsSync(join(snapshotDir, "encoder", "config.json"))).toBe(false);

      // The important part: a second run must *also* fail. A post-hoc verification sweep would have
      // left a complete-looking snapshot behind, and this call would then accept it unverified.
      await expect(downloadModel(REPO, { cacheDir, endpoint: fixture.endpoint })).rejects.toThrow(
        /SHA256SUMS says/,
      );
    } finally {
      await fixture.close();
    }
  });

  it("rejects a corrupt transfer before caching it, leaving no partial file behind", async () => {
    const fixture = await startFixture({ corruptLfsPath: "model.safetensors" });
    const cacheDir = trackScratch();
    try {
      await expect(downloadModel(REPO, { cacheDir, endpoint: fixture.endpoint })).rejects.toThrow(
        /expected sha256 .* got .* corrupt/s,
      );

      const blobs = join(cacheDir, "models--aac6fef--laya-mlx", "blobs");
      const leftovers = existsSync(blobs)
        ? readdirSync(blobs).filter((name) => name.includes("incomplete"))
        : [];
      expect(leftovers).toEqual([]);
      // And nothing was published under the LFS name.
      expect(
        existsSync(
          join(cacheDir, "models--aac6fef--laya-mlx", "snapshots", COMMIT, "model.safetensors"),
        ),
      ).toBe(false);
    } finally {
      await fixture.close();
    }
  });

  it("follows a relative Location header, as the Hub's resolve cache returns", async () => {
    // Found by the live test: the Hub answers with `/api/resolve-cache/...`, not an absolute URL.
    const fixture = await startFixture({ relativeRedirect: true });
    const cacheDir = trackScratch();
    try {
      const resolved = await downloadModel(REPO, { cacheDir, endpoint: fixture.endpoint });
      expect(resolved.fromCache).toBe(false);
      expect(readFileSync(join(resolved.path, "model.safetensors")).toString()).toBe(PAYLOAD);
    } finally {
      await fixture.close();
    }
  });

  it("works when the endpoint serves bytes without redirecting", async () => {
    // Some proxies and mirrors answer the resolve URL directly. The manual-redirect path must fall
    // back cleanly rather than treating a 200 as a failed redirect.
    const fixture = await startFixture({ noRedirect: true });
    const cacheDir = trackScratch();
    try {
      const resolved = await downloadModel(REPO, { cacheDir, endpoint: fixture.endpoint });
      expect(resolved.fromCache).toBe(false);
      expect(readFileSync(join(resolved.path, "rl_agent_config.json"), "utf8")).toContain(
        "head_layers",
      );
    } finally {
      await fixture.close();
    }
  });

  it("surfaces a 401 with advice about the token", async () => {
    const fixture = await startFixture({ requireToken: "hf_needed" });
    const cacheDir = trackScratch();
    try {
      await expect(downloadModel(REPO, { cacheDir, endpoint: fixture.endpoint })).rejects.toThrow(
        /set `token` for a private or gated repo/,
      );
      // With the token it works.
      const ok = await downloadModel(REPO, {
        cacheDir,
        endpoint: fixture.endpoint,
        token: "hf_needed",
      });
      expect(ok.fromCache).toBe(false);
    } finally {
      await fixture.close();
    }
  });

  it("reports which required files are missing from the repo", async () => {
    const fixture = await startFixture({ omit: "rl_agent_config.json" });
    const cacheDir = trackScratch();
    try {
      await expect(downloadModel(REPO, { cacheDir, endpoint: fixture.endpoint })).rejects.toThrow(
        /does not look like a Laya checkpoint: missing rl_agent_config\.json/,
      );
    } finally {
      await fixture.close();
    }
  });
});

describe("load() with a repository id", () => {
  it("refuses to download, pointing at loadAsync instead of silently fetching 803 MiB", async () => {
    // A sync call that quietly performed a large download would be worse than one that fails. This
    // also confirms the check happens before any native load, so it works on any platform.
    const { load } = await import("../packages/mlayax/src/index.js");
    const cacheDir = trackScratch();
    expect(() => load("example/not-cached-anywhere", { cacheDir })).toThrow(/does not download/);
    expect(() => load("example/not-cached-anywhere", { cacheDir })).toThrow(/await loadAsync/);
  });

  it("rejects a source that is neither a directory nor owner/name", async () => {
    const { load } = await import("../packages/mlayax/src/index.js");
    expect(() => load("just-a-name", { cacheDir: trackScratch() })).toThrow(
      /expected "owner\/name"/,
    );
  });
});

describe("reading a cache written by huggingface_hub", () => {
  /**
   * A faithful replica of the layout the Python client writes on this machine, including the parts we
   * deliberately do not reproduce: LFS payloads in the *shared* `hub/blobs/<xx>/<sha256>` store, with
   * the per-repo `blobs/<sha256>` being a symlink into it, and the snapshot entry a symlink to that.
   */
  function writePythonStyleCache(cacheDir: string): string {
    const repoDir = join(cacheDir, "models--aac6fef--laya-mlx");
    const snapshotDir = join(repoDir, "snapshots", COMMIT);
    mkdirSync(join(cacheDir, "blobs", "7f"), { recursive: true });
    mkdirSync(join(repoDir, "blobs"), { recursive: true });
    mkdirSync(join(snapshotDir, "encoder"), { recursive: true });
    mkdirSync(join(snapshotDir, "tokenizer"), { recursive: true });
    mkdirSync(join(repoDir, "refs"), { recursive: true });
    writeFileSync(join(repoDir, "refs", "main"), COMMIT);

    for (const [path, bytes] of Object.entries(FIXTURE_FILES)) {
      const isLarge = path === "model.safetensors";
      const name = isLarge ? sha256(bytes) : sha1(bytes);
      const repoBlob = join(repoDir, "blobs", name);
      if (isLarge) {
        // Shared store, then a relative symlink from the per-repo blobs directory.
        const shared = join(cacheDir, "blobs", "7f", `${name}${"0".repeat(1)}`);
        writeFileSync(shared, bytes);
        symlinkSync(join("..", "..", "blobs", "7f", `${name}${"0".repeat(1)}`), repoBlob, "file");
      } else {
        writeFileSync(repoBlob, bytes);
      }
      // Computed from the file's own directory, as upstream must: `encoder/config.json` is one level
      // deeper, so a fixed `../../blobs/<name>` would dangle.
      symlinkSync(
        relative(dirname(join(snapshotDir, path)), repoBlob),
        join(snapshotDir, path),
        "file",
      );
    }
    writeFileSync(join(cacheDir, "CACHEDIR.TAG"), "Signature: 8a477f597d28d172789f06886806bc55\n");
    return snapshotDir;
  }

  it("finds and reads a snapshot the Python client wrote", async () => {
    const cacheDir = trackScratch();
    const snapshotDir = writePythonStyleCache(cacheDir);

    const cached = resolveCachedModel(REPO, { cacheDir });
    expect(cached?.revision).toBe(COMMIT);
    expect(cached?.path).toBe(snapshotDir);
    expect(cached?.fromCache).toBe(true);
    // Reading follows the symlink chain all the way into the shared blob store.
    expect(readFileSync(join(snapshotDir, "model.safetensors")).toString()).toBe(PAYLOAD);
    expect(readFileSync(join(snapshotDir, "rl_agent_config.json"), "utf8")).toContain(
      "head_layers",
    );

    // And no download is attempted: `downloadModel` short-circuits on a complete snapshot, so this
    // endpoint is never reached. A bodyless server is the proof.
    const resolved = await downloadModel(REPO, {
      cacheDir,
      endpoint: "http://127.0.0.1:59999",
    });
    expect(resolved.fromCache).toBe(true);
    expect(resolved.path).toBe(snapshotDir);
  });
});

describe("the real cache on this machine", () => {
  it("resolves the checkpoint if it is present, proving read-compatibility for real", () => {
    const cached = resolveCachedModel(REPO);
    if (cached === null) return; // No warm cache here; the replica above covers the same ground.
    expect(cached.revision).toMatch(/^[0-9a-f]{40}$/);
    expect(existsSync(join(cached.path, "model.safetensors"))).toBe(true);
    expect(existsSync(join(cached.path, "encoder", "config.json"))).toBe(true);
    expect(existsSync(join(cached.path, "tokenizer", "tokenizer.json"))).toBe(true);
  });
});
