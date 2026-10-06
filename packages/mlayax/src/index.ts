/**
 * @fllstck/mlayax — a Laya typed-decision runtime for Node and Bun on Apple Silicon.
 *
 * TypeScript on MLX. The native binding ships in `@fllstck/mlayax-darwin-arm64`; this package
 * carries no weights and no install scripts.
 *
 * The public surface is built up phase by phase:
 *
 * - **core** (done) — prompt construction, calibration, answer shaping, tokenizer. Pure TypeScript,
 *   no native code, works anywhere.
 * - **mlx** (next) — `load()` and `predict()` over the native binding.
 * - **service** — request batching and the HTTP server.
 *
 * See TASKS.md at the repository root for the phase plan.
 */

export * from "./core/index.js";

/** The published version of this package. Keep in lockstep with `package.json`. */
export const VERSION = "0.1.0" as const;
