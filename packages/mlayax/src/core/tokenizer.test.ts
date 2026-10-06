import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterAll, describe, expect, it } from "vitest";
import { loadTokenizer } from "./tokenizer.js";

const TINY = fileURLToPath(new URL("../../../../test/fixtures/tiny/tokenizer", import.meta.url));

/**
 * A word-level vocabulary with no special tokens at all, for exercising the "config points at a
 * token the vocabulary lacks" path. The tiny fixture cannot reach it: its config resolves, and its
 * fallback names all exist.
 */
const BARE_VOCAB = {
  version: "1.0",
  truncation: null,
  padding: null,
  added_tokens: [],
  normalizer: null,
  pre_tokenizer: { type: "Whitespace" },
  post_processor: null,
  decoder: null,
  model: { type: "WordLevel", vocab: { a: 0, b: 1 }, unk_token: "a" },
};

function fixtureJson(): string {
  return readFileSync(join(TINY, "tokenizer.json"), "utf8");
}

describe("loadTokenizer", () => {
  it("resolves the four special tokens and their ids from the fixture vocabulary", () => {
    const tok = loadTokenizer(TINY);
    expect(tok.mask_token).toBe("[MASK]");
    // These ids are the vocabulary's own, read back from the backend — not assumed from an offset.
    expect(tok.pad_token_id).toBe(0);
    expect(tok.cls_token_id).toBe(2);
    expect(tok.sep_token_id).toBe(3);
    expect(tok.mask_token_id).toBe(4);
  });

  it("encodes without adding special tokens", () => {
    // Upstream always passes add_special_tokens=False: the sequence builder places
    // [CLS]/[SEP]/[MASK] itself, so an added [CLS] would shift every marker by one.
    const tok = loadTokenizer(TINY);
    const ids = tok.encode("hello world");
    expect(ids).toEqual([5, 6]);
    expect(ids).not.toContain(tok.cls_token_id);
    expect(ids).not.toContain(tok.sep_token_id);
    expect(tok.encode("hello")).toEqual([5]);
  });

  it("keeps the leading space of a word, so ' world' and 'world' are distinguishable", () => {
    // Word-boundary information is part of the token, which is why build_prefix prepends " " to
    // option text rather than relying on the previous token.
    const tok = loadTokenizer(TINY);
    expect(tok.encode(" world")).toEqual([6]);
  });

  it("maps an unknown word to the unknown id rather than dropping it", () => {
    const tok = loadTokenizer(TINY);
    expect(tok.encode("zzzz")).toEqual([1]);
  });

  it("splits punctuation into its own tokens, as the checkpoint's pre-tokenizer does", () => {
    const tok = loadTokenizer(TINY);
    // "zzzz-not-in-vocab" -> zzzz | - | not | - | in | - | vocab
    expect(tok.encode("zzzz-not-in-vocab")).toEqual([1, 1, 30, 1, 47, 1, 1]);
  });

  it("returns no ids for empty or whitespace-only text", () => {
    const tok = loadTokenizer(TINY);
    expect(tok.encode("")).toEqual([]);
    expect(tok.encode("   ")).toEqual([]);
  });

  it("is deterministic across loads", () => {
    expect(loadTokenizer(TINY).encode("refund the invoice")).toEqual([7, 12, 11]);
    expect(loadTokenizer(TINY).encode("refund the invoice")).toEqual(
      loadTokenizer(TINY).encode("refund the invoice"),
    );
  });
});

describe("loadTokenizer special-token resolution", () => {
  const scratch = mkdtempSync(join(tmpdir(), "mlayax-tok-"));
  afterAll(() => rmSync(scratch, { recursive: true, force: true }));

  function writeTokenizer(name: string, tokenizerJson: unknown, config: unknown): string {
    const dir = join(scratch, name);
    mkdirSync(dir, { recursive: true });
    writeFileSync(join(dir, "tokenizer.json"), JSON.stringify(tokenizerJson));
    writeFileSync(join(dir, "tokenizer_config.json"), JSON.stringify(config));
    return dir;
  }

  it("names the missing token when the vocabulary has none of the candidates", () => {
    // A silently wrong [MASK] id would produce plausible-looking but wrong prompts, so this must
    // fail loudly and say which token is at fault.
    const dir = writeTokenizer("bare", BARE_VOCAB, {
      cls_token: "[CLS]",
      sep_token: "[SEP]",
      pad_token: "[PAD]",
      mask_token: "[MASK]",
    });
    expect(() => loadTokenizer(dir)).toThrow(/missing a valid mask token/);
  });

  it("falls back to conventional token names when the config declares none", () => {
    const dir = writeTokenizer("fallback", JSON.parse(fixtureJson()), {});
    const tok = loadTokenizer(dir);
    expect(tok.mask_token).toBe("[MASK]");
    expect(tok.mask_token_id).toBe(4);
    expect(tok.cls_token_id).toBe(2);
  });

  it("prefers the config's name over the fallback when the vocabulary has it", () => {
    const dir = writeTokenizer("preferred", JSON.parse(fixtureJson()), {
      cls_token: "[CLS]",
      sep_token: "[SEP]",
      pad_token: "[PAD]",
      mask_token: "[MASK]",
    });
    expect(loadTokenizer(dir).mask_token).toBe("[MASK]");
  });

  it("accepts a config that records a special token as an object", () => {
    // Some checkpoints write `{"content": "[MASK]", "lstrip": false}`.
    const dir = writeTokenizer("objectform", JSON.parse(fixtureJson()), {
      cls_token: { content: "[CLS]" },
      sep_token: { content: "[SEP]" },
      pad_token: { content: "[PAD]" },
      mask_token: { content: "[MASK]" },
    });
    expect(loadTokenizer(dir).mask_token_id).toBe(4);
  });
});
