/**
 * Live Hugging Face API smoke test — opt-in, because CI must not depend on the network.
 *
 * ```bash
 * MLAYAX_LIVE_HUB=1 npx vitest run test/hub.live.test.ts
 * ```
 *
 * This exists for one reason: the fixture server in `hub.test.ts` can only confirm that the client
 * agrees with *my* idea of the API. The endpoint paths, the `sha` field on the revision response, and
 * the `tree` entry shape (`oid` for ordinary files, `lfs.oid` for large ones) are assumptions that
 * only the real Hub can falsify.
 *
 * It deliberately uses `force: true` on an already-populated cache, so it costs two HTTP requests
 * and downloads nothing — the point is the request/response shapes, not the payload.
 */

import { existsSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  DEFAULT_ENDPOINT,
  defaultCacheDir,
  downloadModel,
  resolveCachedModel,
} from "../packages/mlayax/src/hub.js";

const REPO = "aac6fef/laya-mlx";

describe.skipIf(process.env.MLAYAX_LIVE_HUB !== "1")("live Hugging Face API", () => {
  it("resolves a branch and re-lists the tree against the real endpoint", async () => {
    const cached = resolveCachedModel(REPO);
    if (cached === null) {
      throw new Error(
        "This test needs a populated cache; run it once without MLAYAX_LIVE_HUB to download.",
      );
    }
    const before = statSync(join(cached.path, "model.safetensors")).mtimeMs;

    const resolved = await downloadModel(REPO, { force: true });

    // The revision response really does carry `sha`, and it is a commit.
    expect(resolved.revision).toMatch(/^[0-9a-f]{40}$/);
    expect(resolved.path).toContain(resolved.revision);

    // `refs/main` is what the cache-free resolution path relies on.
    const ref = join(defaultCacheDir(), "models--aac6fef--laya-mlx", "refs", "main");
    expect(readFileSync(ref, "utf8").trim()).toBe(resolved.revision);

    // The listing named real files and all of them resolve.
    for (const entry of [
      "model.safetensors",
      "rl_agent_config.json",
      "encoder/config.json",
      "tokenizer/tokenizer.json",
      "tokenizer/tokenizer_config.json",
    ]) {
      expect(existsSync(join(resolved.path, entry)), entry).toBe(true);
    }

    // The 803 MiB payload was *not* re-fetched. Asserting `fromCache === true` would be wrong: a
    // cache written by `huggingface_hub` need not contain every file the tree lists — it omits
    // `.gitattributes` here — so the first run after switching languages legitimately fetches a few
    // small files. The mtime of the big one is the assertion that matters.
    expect(statSync(join(resolved.path, "model.safetensors")).mtimeMs).toBe(before);
  }, 60_000);

  it("resolves a commit sha without a revision lookup", async () => {
    const known = resolveCachedModel(REPO);
    if (known === null) return; // Nothing cached on this machine.
    const bySha = await downloadModel(REPO, { revision: known.revision, force: true });
    expect(bySha.revision).toBe(known.revision);
    expect(bySha.path).toBe(known.path);
  }, 60_000);

  it("the default endpoint is the public Hub", () => {
    expect(DEFAULT_ENDPOINT).toBe("https://huggingface.co");
  });
});
