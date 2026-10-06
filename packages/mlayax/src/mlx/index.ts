/**
 * `@fllstck/mlayax` MLX runtime — the part that touches native code.
 *
 * Importing this module is safe on any platform: nothing loads the addon until {@link loadMx} or
 * {@link load} is called. That is deliberate, so the pure-TypeScript `core` layer can be imported
 * and unit-tested on Linux CI where no Apple Silicon payload exists.
 *
 * ```ts
 * import { load } from "@fllstck/mlayax";
 *
 * const agent = load("/path/to/laya-mlx");
 * const { answers, usage } = await agent.predict("I was billed twice.", {
 *   department: {
 *     type: "choice",
 *     instructions: "Which team should handle this?",
 *     criteria: { billing: "invoices", technical: "bugs" },
 *   },
 * });
 * ```
 */

export {
  type DTypeName,
  type ForwardResult,
  type LoadOptions,
  load,
  loadAsync,
  MlxAgent,
  type Prediction,
  type Prepared,
  type PreparedItem,
  type Usage,
} from "./agent.js";
export {
  isMxLoaded,
  loadMx,
  loadMxModule,
  resetMxCacheForTests,
  resolveNativeAddonPath,
} from "./binding.js";
export {
  ALLOW_MIXED_ENV,
  assertNoMixedMlx,
  bytesOf,
  checkForMixedMlx,
  describeMlxLoadFailure,
  type ForeignMlxLibrary,
  HEADER_PREFIX_BYTES,
  hasMixingReport,
  inspectResidentMlx,
  type MixingReport,
  MLX_LIBRARY_NAMES,
  MlxMixingError,
  mixingError,
  mixingWarning,
  readInstallName,
  readOf,
  realpathOf,
  residentSharedObjects,
  type SharedObjectReporter,
  sha256Of,
} from "./mixing.js";

export {
  type AgentConfig,
  attentionMasks,
  DecisionModel,
  type EncoderConfig,
  encoderConfigFrom,
  type ModelOptions,
  ropeBase,
  sanitizeWeights,
  type Weights,
} from "./model.js";
export type {
  MlxArray,
  MlxCore,
  MlxDevice,
  MlxDType,
  MlxFast,
  MlxModule,
  MlxOperand,
  MlxShape,
  MlxSlice,
} from "./types.js";
