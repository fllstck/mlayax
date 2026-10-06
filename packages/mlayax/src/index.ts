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
 * - **service** — request batching and the HTTP server (next).
 *
 * See TASKS.md at the repository root for the phase plan.
 */

export * from "./core/index.js";
export * from "./hub.js";
export * from "./mlx/index.js";

/** The published version of this package. Keep in lockstep with `package.json`. */
export const VERSION = "0.1.0" as const;
