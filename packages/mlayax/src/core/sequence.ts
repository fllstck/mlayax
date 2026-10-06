/**
 * Sequence construction: turn a question plus a state into the token sequence the model reads.
 *
 * A port of `laya_mlx.common.build_prefix` / `finish_sequence`.
 *
 * The layout the model expects, and why the pieces are ordered this way:
 *
 * ```
 * [CLS] <"<type> question: <instructions>"> [SEP] <option 0> <option 1> ... [SEP]
 *   ^                                         ^
 *   every option starts with a [MASK] token, and `markers[i]` records where option i begins —
 *   the decision head reads its logit from that position.
 * ```
 *
 * Two budgets, both of which are truncation traps:
 *
 * - `headMaxLen` caps the *question* part. If the options alone overrun it, every option is cut to
 *   the same `tokens_per_option` so they stay comparable, and the [`[MASK]` marker is kept even
 *   when the body is truncated away — losing a marker would silently drop that option's answer.
 * - `maxLen` caps the whole sequence. State tokens are the only elastic part.
 *
 * Truncation direction follows the state's shape, not a flag the caller picks: a **string** state
 * is prose, so its tail is cut (`truncateLeft: false`); a **list** state is a turn list whose most
 * recent turns matter most, so its head is cut (`truncateLeft: true`).
 */

import type { InternalQuestion } from "./questions.js";
import { renderOptions } from "./questions.js";

/**
 * The only tokenizer surface the sequence builder needs.
 *
 * Field names keep upstream's snake_case deliberately: this is a structural interface that the
 * real tokenizer wrapper, the test fixtures and the golden fixture all satisfy, and renaming them
 * would make the port harder to diff against `laya_mlx`.
 */
export interface TinyTokenizer {
  readonly mask_token: string;
  readonly cls_token_id: number;
  readonly sep_token_id: number;
  readonly pad_token_id: number;
  readonly mask_token_id: number;
  /** Encode without adding special tokens (upstream always passes `add_special_tokens=False`). */
  encode(text: string): number[];
}

/** Per-question token-budget diagnostics, surfaced in `usage` as the collapsed-options report. */
export interface PrefixStats {
  /** Options the question defines. */
  options: number;
  /** Options whose *rendered* token spans differ — fewer means duplicates collapsed. */
  options_distinct: number;
  /** Per-option cap that was applied, or `null` if the options fit. */
  tokens_per_option: number | null;
}

/** The question-only prefix, plus where each option begins. */
export interface BuiltPrefix {
  ids: number[];
  markers: number[];
}

/** {@link BuiltPrefix} with the budget diagnostics. */
export interface BuiltPrefixWithStats extends BuiltPrefix {
  stats: PrefixStats;
}

/** Truncation report for one sequence, surfaced in `usage`. */
export interface SequenceStats {
  state_tokens: number;
  state_tokens_used: number;
  state_tokens_dropped: number;
  truncated: boolean;
}

/** A complete sequence, with the markers that survived the length cap. */
export interface BuiltSequence {
  ids: number[];
  markers: number[];
  stats: SequenceStats;
}

/** The upstream default head budget; the checkpoint's `head_max_len` overrides it. */
export const DEFAULT_HEAD_MAX_LEN = 192;

/** Per-option token cap before collapsing kicks in. */
const OPTION_TOKEN_CAP = 48;

/** Below this much head budget left over, options start being collapsed. */
const OPTION_BUDGET_FLOOR = 16;

/** Never truncate the question head below this many tokens, however tight the option budget is. */
const MIN_HEAD_TOKENS = 8;

/** Smallest collapsed per-option span. Keeping the marker plus a few tokens preserves the shape. */
const MIN_TOKENS_PER_OPTION = 4;

/**
 * Build the question-only prefix.
 *
 * Option bodies are capped at {@link OPTION_TOKEN_CAP} tokens each first. If what remains of the
 * head budget is under {@link OPTION_BUDGET_FLOOR}, every option is instead collapsed to the same
 * `tokens_per_option` — computed from the *remaining* budget after reserving room for the head —
 * which is what makes many-option questions (`choice:11+`) still answerable.
 */
export function buildPrefix(
  tok: TinyTokenizer,
  q: InternalQuestion,
  headMaxLen?: number,
): BuiltPrefix;
export function buildPrefix(
  tok: TinyTokenizer,
  q: InternalQuestion,
  headMaxLen: number,
  returnStats: true,
): BuiltPrefixWithStats;
export function buildPrefix(
  tok: TinyTokenizer,
  q: InternalQuestion,
  headMaxLen: number,
  returnStats: boolean,
): BuiltPrefix | BuiltPrefixWithStats;
export function buildPrefix(
  tok: TinyTokenizer,
  q: InternalQuestion,
  headMaxLen = DEFAULT_HEAD_MAX_LEN,
  returnStats = false,
): BuiltPrefix | BuiltPrefixWithStats {
  const options = renderOptions(q);
  const instructions = String(q.ins).replaceAll(tok.mask_token, " ");
  const headIds = tok.encode(`${q.t} question: ${instructions}`);

  let optionIds = options.map((option) => [
    tok.mask_token_id,
    ...tok.encode(` ${option.replaceAll(tok.mask_token, " ")}`).slice(0, OPTION_TOKEN_CAP),
  ]);

  let optionBudget = headMaxLen - optionIds.reduce((total, ids) => total + ids.length, 0);
  let tokensPerOption: number | null = null;
  if (optionBudget < OPTION_BUDGET_FLOOR) {
    const collapsed = Math.max(
      MIN_TOKENS_PER_OPTION,
      Math.floor((headMaxLen - OPTION_BUDGET_FLOOR) / Math.max(1, optionIds.length)),
    );
    tokensPerOption = collapsed;
    optionIds = optionIds.map((ids) => ids.slice(0, collapsed));
    optionBudget = headMaxLen - optionIds.reduce((total, ids) => total + ids.length, 0);
  }

  const keptHead = headIds.slice(0, Math.max(MIN_HEAD_TOKENS, optionBudget));
  const ids = [tok.cls_token_id, ...keptHead, tok.sep_token_id];
  const markers: number[] = [];
  for (const option of optionIds) {
    markers.push(ids.length);
    ids.push(...option);
  }
  ids.push(tok.sep_token_id);

  if (returnStats) {
    return {
      ids,
      markers,
      stats: {
        options: optionIds.length,
        options_distinct: new Set(optionIds.map((o) => o.join(","))).size,
        tokens_per_option: tokensPerOption,
      },
    };
  }
  return { ids, markers };
}

/**
 * Append the state slice to a prefix and clamp the whole sequence to `maxLen`.
 *
 * Markers are filtered against `maxLen` rather than dropped blindly: an option whose marker lands
 * past the cap cannot be read out, so it must not be counted. Callers compare the surviving marker
 * count against the option count and reject the question if any option was lost entirely — silently
 * answering with fewer options would misalign the probability vector.
 */
export function finishSequence(
  tok: TinyTokenizer,
  prefix: number[],
  markers: number[],
  stateIds: number[],
  maxLen: number,
  truncateLeft = false,
): BuiltSequence {
  const room = Math.max(0, maxLen - prefix.length - 1);
  const kept = truncateLeft
    ? stateIds.slice(Math.max(0, stateIds.length - room))
    : stateIds.slice(0, room);
  const ids = [...prefix, ...kept, tok.sep_token_id].slice(0, maxLen);
  return {
    ids,
    markers: markers.filter((marker) => marker < maxLen),
    stats: {
      state_tokens: stateIds.length,
      state_tokens_used: kept.length,
      state_tokens_dropped: stateIds.length - kept.length,
      truncated: kept.length < stateIds.length,
    },
  };
}
