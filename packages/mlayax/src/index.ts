/**
 * @fllstck/mlayax — a Laya typed-decision runtime for Node and Bun on Apple Silicon.
 *
 * TypeScript on MLX. The native binding ships in `@fllstck/mlayax-darwin-arm64`; this package carries
 * no weights and no install scripts.
 *
 * ```ts
 * import { load } from "@fllstck/mlayax";
 *
 * const agent = load("/path/to/laya-mlx");
 * const { answers, usage } = await agent.predict("I was billed twice. Please refund it.", {
 *   department: {
 *     type: "choice",
 *     instructions: "Which team should handle this?",
 *     criteria: { billing: "invoices, payments", technical: "bugs and outages" },
 *   },
 * });
 * ```
 *
 * The surface is built up phase by phase:
 *
 * - **core** — prompt construction, calibration, answer shaping, tokenizer. Pure TypeScript with no
 *   native dependency, so it imports and unit-tests anywhere, including Linux CI.
 * - **mlx** — `load()` and `predict()` over the native binding. Imports safely everywhere, but only
 *   *loads* the addon when you call into it, and fails with a described error off Apple Silicon.
 * - **hub** — the Hugging Face fetcher: resolve a repository id against the local cache, or download
 *   a checkpoint into it with `loadAsync()`.
 *
 * There is no service layer: batching belongs to the caller, who is the only one who can see their
 * own concurrency. The seam for it is public (`prepare` → `forwardItems` → answer shaping) so a
 * caller who needs coalescing can build it without this package shipping a server.
 *
 * See TASKS.md at the repository root for the phase plan.
 */

export * from "./core/index.js";
export * from "./hub.js";
export * from "./mlx/index.js";

/** The published version of this package. Keep in lockstep with `package.json`. */
export const VERSION = "0.1.0" as const;
