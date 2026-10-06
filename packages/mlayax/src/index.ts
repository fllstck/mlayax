/**
 * @fllstck/mlayax — a Laya typed-decision runtime for Node and Bun on Apple Silicon.
 *
 * TypeScript on MLX. The native binding ships in `@fllstck/mlayax-darwin-arm64`; this package
 * carries no weights and no install scripts.
 *
 * Phase 1 scaffold: the public surface (`load`, `predict`, the service) lands in phases 2–4.
 * See TASKS.md at the repository root for the phase plan.
 */

/** The published version of this package. Keep in lockstep with `package.json`. */
export const VERSION = "0.1.0" as const;
