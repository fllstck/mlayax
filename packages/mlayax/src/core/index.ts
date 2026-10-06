/**
 * `@fllstck/mlayax` core — pure TypeScript, no native code.
 *
 * Prompt construction, calibration and answer shaping. Nothing here touches MLX, a GPU or the file
 * system, so it is unit-testable on any platform and is the part of the pipeline that actually
 * decides what the model is asked. The MLX runtime lives in `../mlx` and the service in
 * `../service`.
 */

export type {
  Answer,
  AnswerBase,
  ChoiceAnswer,
  CollapsedOptionReport,
  NoulAnswer,
  ScoreAnswer,
} from "./answers.js";
export { collapsedOptions, shapeAnswer } from "./answers.js";
export {
  answerConfidence,
  clampTemperature,
  confidenceFromProbs,
  formatSignificant,
  rejectedTemperatureEntries,
  round4,
  softmax,
  TEMP_MAX,
  TEMP_MIN,
  tempBucket,
  temperatureClampWarning,
} from "./calibration.js";
export {
  pyFloatRepr,
  pyJson,
  pyNumberRepr,
  renderCriterion,
  serializeState,
} from "./ppjson.js";
export type {
  InternalQuestion,
  NoulLabels,
  QuestionKind,
  QuestionTypeId,
} from "./questions.js";
export {
  QTYPE_NAMES,
  QTYPES,
  renderOptions,
  resolveNoulLabels,
  toInternal,
} from "./questions.js";
export type {
  BuiltPrefix,
  BuiltPrefixWithStats,
  BuiltSequence,
  PrefixStats,
  SequenceStats,
  TinyTokenizer,
} from "./sequence.js";
export { buildPrefix, DEFAULT_HEAD_MAX_LEN, finishSequence } from "./sequence.js";

export { loadTokenizer } from "./tokenizer.js";
