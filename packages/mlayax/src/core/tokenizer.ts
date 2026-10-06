/**
 * Tokenizer loading, over `@huggingface/tokenizers`.
 *
 * A port of `laya_mlx.tokenizer`. The only surface the rest of the runtime needs is the four
 * special tokens with their ids, and `encode` with **no special tokens added** — upstream always
 * passes `add_special_tokens=False`, so callers place `[CLS]`/`[SEP]`/`[MASK]` themselves.
 *
 * Special tokens are resolved by *name* from `tokenizer_config.json` first, then from a list of
 * conventional fallbacks. That indirection is not decoration: the fallback list is what makes a
 * ModernBERT-family checkpoint and a differently-named tiny fixture work through one code path, and
 * resolving the id from the backend (rather than trusting an index) is what catches a config that
 * names a token the vocabulary does not contain.
 */

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { Tokenizer } from "@huggingface/tokenizers";
import type { TinyTokenizer } from "./sequence.js";

/** A special token as recorded in `tokenizer_config.json`. */
interface SpecialTokenEntry {
  content?: unknown;
}

interface TokenizerConfig {
  cls_token?: unknown;
  sep_token?: unknown;
  pad_token?: unknown;
  mask_token?: unknown;
  [key: string]: unknown;
}

/** The JSON shape of `tokenizer.json`, passed through to the backend untyped. */
type TokenizerJson = ConstructorParameters<typeof Tokenizer>[0];

function readJson<T>(path: string): T {
  return JSON.parse(readFileSync(path, "utf8")) as T;
}

/**
 * Load a tokenizer from a checkpoint directory.
 *
 * Throws if any of the four special tokens is absent from the vocabulary, naming the token — a
 * silently wrong `[MASK]` id would produce plausible-looking but wrong prompts.
 */
export function loadTokenizer(dir: string): TinyTokenizer {
  const tokenizerJson = readJson<TokenizerJson>(join(dir, "tokenizer.json"));
  const config = readJson<TokenizerConfig>(join(dir, "tokenizer_config.json"));
  const backend = new Tokenizer(tokenizerJson, config);

  const resolve = (name: string, fallbacks: readonly string[]): [token: string, id: number] => {
    let value = config[`${name}_token`];
    if (typeof value === "object" && value !== null) {
      value = (value as SpecialTokenEntry).content;
    }
    const candidates = [value, ...fallbacks].filter(
      (candidate): candidate is string => typeof candidate === "string" && candidate.length > 0,
    );
    for (const candidate of candidates) {
      const id = backend.token_to_id(candidate);
      if (id !== undefined) return [candidate, id];
    }
    throw new Error(`Tokenizer is missing a valid ${name} token`);
  };

  const [maskToken, maskId] = resolve("mask", ["[MASK]", "<mask>"]);
  const [, clsId] = resolve("cls", ["[CLS]", "<s>", "<cls>", "<bos>"]);
  const [, sepId] = resolve("sep", ["[SEP]", "</s>", "<sep>", "<eos>"]);
  const [, padId] = resolve("pad", ["[PAD]", "<pad>"]);

  return {
    mask_token: maskToken,
    mask_token_id: maskId,
    cls_token_id: clsId,
    sep_token_id: sepId,
    pad_token_id: padId,
    encode: (text: string) => backend.encode(text, { add_special_tokens: false }).ids,
  };
}
