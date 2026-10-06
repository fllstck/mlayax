/**
 * The Laya agent on MLX: load a checkpoint, prepare a request, run one forward, read out answers.
 *
 * A port of `laya_mlx/agent.py`'s tensor path. Everything that is not tensor work — sequence
 * construction, truncation, calibration, answer shaping — comes from `../core`, shared with any
 * other backend; the two differ only in how they execute the graph.
 *
 * The temperature story is worth reading before changing anything here. The shipped checkpoints
 * carry fitted temperatures *outside* the usable range: `choice:11+` is `0.1006`, a ~10x sharpener
 * that publishes a 0.24 top probability as 0.99. `clampTemperature` confines every temperature to
 * `[0.5, 5]`, and the checkpoint's own values are kept aside in `temperatureRaw` for inspection so
 * that clamping is visible rather than silent. The warning emitted at load says which buckets were
 * affected and that their confidence is therefore uncalibrated.
 */

import { readFileSync } from "node:fs";
import { join } from "node:path";
import {
  type Answer,
  buildPrefix,
  type CollapsedOptionReport,
  clampTemperature,
  coerceTemperature,
  collapsedOptions,
  finishSequence,
  type InternalQuestion,
  type PrefixStats,
  QTYPES,
  type QuestionTypeId,
  rejectedTemperatureEntries,
  renderOptions,
  type SequenceStats,
  serializeState,
  shapeAnswer,
  softmax,
  tempBucket,
  temperatureClampWarning,
  toInternal,
} from "../core/index.js";
import { loadTokenizer } from "../core/tokenizer.js";
import { loadMx } from "./binding.js";
import {
  type AgentConfig,
  attentionMasks,
  DecisionModel,
  type EncoderConfig,
  encoderConfigFrom,
  sanitizeWeights,
  type Weights,
} from "./model.js";
import type { MlxArray, MlxCore, MlxDType } from "./types.js";

/** Selectable weight precision. `bfloat16` is accepted for symmetry; the checkpoints ship fp16. */
export type DTypeName = "float32" | "float16" | "bfloat16";

/** Load-time options. Every one has a documented default. */
export interface LoadOptions {
  /** Weight precision. Default `float16`, which is what the reference numbers were measured at. */
  dtype?: DTypeName;
  /** Rows per forward. Default 16. */
  batchSize?: number;
  /** Run on CPU instead of the GPU. Present for debugging; it is far slower. */
  device?: "gpu" | "cpu";
  /**
   * Dispose intermediate tensors after each forward (`mx.tidy`). Default on for the synchronous
   * path. The batching service turns it **off** and disposes its feeds explicitly, because
   * `mx.tidy` around an in-flight `mx.asyncEval` is unsafe.
   */
  tidy?: boolean;
  /** Wrap the forward in `mx.compile`. Default on; it removes the per-op JS call overhead (-12 %). */
  compile?: boolean;
  /**
   * Compile once for every shape. Off by default because it forces manual attention and costs
   * 12-15 % per forward, in exchange for removing one-off traces when traffic changes length.
   * `MLAYAX_SHAPELESS=1` turns it on.
   */
  shapeless?: boolean;
  /** Force manual or fused attention independently of `shapeless`. */
  attention?: "manual" | "fused";
  /**
   * Round the padded sequence length up to a multiple of this. Off by default: it collapses compile
   * retraces onto few shapes but costs +30 % on a token-bound batch of 16.
   */
  lengthBucket?: number;
  /** Hold weight transposes resident. Measured as no effect; off by default. `MLAYAX_PRETRANSPOSE=1`. */
  pretranspose?: boolean;
  /** Fused `addmm` for biased linears. Measured as no effect; off by default. `MLAYAX_ADDM=1`. */
  addmm?: boolean;
  /** Reuse attention masks across calls with the same `(batch, length, row lengths)`. Default on. */
  cacheMasks?: boolean;
  /** `mx.setWiredLimit(bytes)` — keep allocations resident so they are not paged out. */
  wired?: number;
  /** `mx.setCacheLimit(bytes)` — bound MLX's free-buffer cache. */
  cacheLimit?: number;
}

/** Per-question bookkeeping carried from `prepare` to the `usage` report. */
export interface PreparedItem {
  ids: number[];
  markers: number[];
  qtype: QuestionTypeId;
  /** Option-budget diagnostics; `<options>` distinct means options collapsed. */
  options: PrefixStats;
  /** Truncation diagnostics for this question's sequence. */
  state_stats: SequenceStats;
}

/** A request after preparation: one row per question, plus the questions and their ids. */
export interface Prepared {
  items: PreparedItem[];
  internal: InternalQuestion[];
  ids: string[];
}

/** Raw forward output, flattened on the host. */
export interface ForwardResult {
  logits: Float32Array;
  action: Float32Array;
  /** Marker slots per row — the stride of `logits`. */
  count: number;
  /** Action classes per row — the stride of `action`. */
  nAct: number;
}

/** Token accounting for one request, mirroring upstream's `usage`. */
export interface Usage {
  input_tokens: number;
  output_tokens: number;
  state_tokens: number;
  state_tokens_dropped: number;
  truncated: boolean;
  truncated_questions: string[];
  options?: Record<string, CollapsedOptionReport>;
}

/** The full response shape. */
export interface Prediction {
  model: "laya-rl-agent";
  answers: Record<string, Answer>;
  usage: Usage;
}

/** An MLX array carrying its own shape, as reported by the native binding. */
type MlxArrayLike = MlxArray;

/** Env knobs, all opt-in and all documented in the README's options table. */
const ENV = {
  shapeless: "MLAYAX_SHAPELESS",
  pretranspose: "MLAYAX_PRETRANSPOSE",
  addmm: "MLAYAX_ADDM",
} as const;

function envFlag(name: string): boolean {
  return process.env[name] === "1";
}

/**
 * Validate a question id the way upstream does: a nonempty string or an integer.
 *
 * Question ids become the keys of the answer object, so an empty or duplicate-ish id makes the
 * response hard to correlate; upstream rejects it rather than guessing.
 */
function validateQuestionId(qid: string): void {
  if (typeof qid !== "string" || qid.trim() === "") {
    throw new Error("Question id must be a nonempty string or integer");
  }
}

export class MlxAgent {
  readonly dtypeName: DTypeName;
  readonly cfg: AgentConfig;
  readonly encoderCfg: EncoderConfig;
  readonly model: DecisionModel;
  readonly tokenizer: ReturnType<typeof loadTokenizer>;
  readonly weights: Weights;
  readonly batchSize: number;
  readonly tidy: boolean;
  readonly cacheMasks: boolean;
  readonly shapeless: boolean;
  /** Padded-length bucket, or 0 for none. */
  readonly lengthBucket: number;

  /** The checkpoint's own temperatures, unclamped, kept for inspection. */
  readonly temperatureRaw: number[];
  readonly temperatureByOptionsRaw: Record<string, unknown>;
  /** The temperatures actually applied (clamped). */
  readonly temperature: number[];
  readonly temperatureByOptions: Record<string, number>;

  private readonly maskCache = new Map<string, { full: MlxArrayLike; local: MlxArrayLike }>();
  /** Shape-specialised compiled forward, or `null` when running eagerly. */
  private readonly compiledForward:
    | ((
        inputIds: MlxArrayLike,
        fullMask: MlxArrayLike,
        localMask: MlxArrayLike,
        markerPos: MlxArrayLike,
        markerMask: MlxArrayLike,
        qtype: MlxArrayLike,
      ) => { logits: MlxArrayLike; action: MlxArrayLike })
    | null;
  private readonly mx: MlxCore;

  private constructor(init: {
    mx: MlxCore;
    cfg: AgentConfig;
    encoderCfg: EncoderConfig;
    model: DecisionModel;
    tokenizer: ReturnType<typeof loadTokenizer>;
    weights: Weights;
    dtypeName: DTypeName;
    batchSize: number;
    tidy: boolean;
    cacheMasks: boolean;
    shapeless: boolean;
    lengthBucket: number;
    compiledForward: MlxAgent["compiledForward"];
  }) {
    this.mx = init.mx;
    this.cfg = init.cfg;
    this.encoderCfg = init.encoderCfg;
    this.model = init.model;
    this.tokenizer = init.tokenizer;
    this.weights = init.weights;
    this.dtypeName = init.dtypeName;
    this.batchSize = init.batchSize;
    this.tidy = init.tidy;
    this.cacheMasks = init.cacheMasks;
    this.shapeless = init.shapeless;
    this.lengthBucket = init.lengthBucket;
    this.compiledForward = init.compiledForward;

    this.temperatureRaw = (init.cfg.temperature ?? [1, 1, 1]).map(Number);
    this.temperatureByOptionsRaw = init.cfg.temperature_by_options ?? {};
    this.temperature = this.temperatureRaw.map((t) => clampTemperature(t));
    this.temperatureByOptions = Object.fromEntries(
      Object.entries(this.temperatureByOptionsRaw).map(([bucket, value]) => [
        bucket,
        clampTemperature(value),
      ]),
    );
  }

  /**
   * Load a checkpoint from a directory.
   *
   * The directory must contain `rl_agent_config.json`, `encoder/config.json`, `model.safetensors`
   * and `tokenizer/`. Nothing is downloaded here — that is the Hub fetcher's job (phase 3b).
   */
  static load(modelDir: string, options: LoadOptions = {}): MlxAgent {
    const mx = loadMx();
    const dtypeName = options.dtype ?? "float16";
    if (options.device === "cpu") mx.setDefaultDevice(mx.cpu);

    const cfg = JSON.parse(
      readFileSync(join(modelDir, "rl_agent_config.json"), "utf8"),
    ) as AgentConfig;
    const encoderCfg = encoderConfigFrom(
      JSON.parse(readFileSync(join(modelDir, "encoder/config.json"), "utf8")) as Record<
        string,
        unknown
      >,
    );
    if (!cfg.encoder || !cfg.head_layers) {
      throw new Error("Laya config must specify encoder and head_layers");
    }
    assertUsableTemperatures(cfg);

    const tokenizer = loadTokenizer(join(modelDir, "tokenizer"));

    const sanitized = sanitizeWeights(mx.load(join(modelDir, "model.safetensors")));
    const dtype = dtypeFor(mx, dtypeName);
    const weights: Weights = {};
    for (const [name, value] of Object.entries(sanitized)) {
      weights[name] = value.astype(dtype);
    }

    // Default: per-shape compilation. MLX caches a compiled graph per input shape (a new shape
    // traces once, ~1.5-4 ms), while shapeless forces manual attention and costs 12-15 % per
    // forward. Set MLAYAX_SHAPELESS=1 when shape variety is unbounded and the one-off traces hurt
    // more than steady-state throughput.
    const shapeless = options.shapeless ?? envFlag(ENV.shapeless);
    const manualAttention =
      options.attention === "manual" ? true : options.attention === "fused" ? false : shapeless;

    const model = new DecisionModel(mx, encoderCfg, weights, cfg, dtype, {
      pretranspose: options.pretranspose ?? envFlag(ENV.pretranspose),
      addmm: options.addmm ?? envFlag(ENV.addmm),
      manualAttention,
    });
    model.prepareWeights();
    mx.eval(...Object.values(weights));
    if (options.wired !== undefined) mx.setWiredLimit(options.wired);
    if (options.cacheLimit !== undefined) mx.setCacheLimit(options.cacheLimit);

    // MLX caches compiled callables per input shape, so one compiled function serves every sequence
    // length: each new shape traces once. Weights are frozen before compiling.
    const compile = options.compile ?? true;
    const compiledForward = compile
      ? mx.compile(
          (
            inputIds: MlxArrayLike,
            fullMask: MlxArrayLike,
            localMask: MlxArrayLike,
            markerPos: MlxArrayLike,
            markerMask: MlxArrayLike,
            qtype: MlxArrayLike,
          ) => model.forward(inputIds, fullMask, localMask, markerPos, markerMask, qtype),
          shapeless,
        )
      : null;

    const agent = new MlxAgent({
      mx,
      cfg,
      encoderCfg,
      model,
      tokenizer,
      weights,
      dtypeName,
      batchSize: options.batchSize ?? 16,
      tidy: options.tidy ?? true,
      cacheMasks: options.cacheMasks ?? true,
      shapeless,
      lengthBucket: options.lengthBucket ?? 0,
      compiledForward,
    });

    const rejected = rejectedTemperatureEntries(
      agent.temperatureRaw,
      agent.temperatureByOptionsRaw,
    );
    const warning = temperatureClampWarning(rejected);
    if (warning !== null) process.emitWarning(warning, "RuntimeWarning");

    return agent;
  }

  // ---- preparation -----------------------------------------------------

  /**
   * Turn a request into one token sequence per question, with option markers.
   *
   * Truncation direction follows the *shape of the state*: a string is prose and keeps its head
   * (`truncateLeft: false`), while a list is a turn list and keeps its tail (`true`).
   */
  prepare(state: unknown, questions: Record<string, unknown>): Prepared {
    if (state === null || state === undefined) throw new Error("state must not be None");
    if (typeof questions !== "object" || questions === null || Array.isArray(questions)) {
      throw new Error("questions must be a dictionary keyed by question id");
    }
    const ids = Object.keys(questions);
    if (ids.length === 0) return { items: [], internal: [], ids };

    const maxLen = this.cfg.max_len ?? 512;
    const headMaxLen = this.cfg.head_max_len ?? 192;
    const truncateLeft = Array.isArray(state);
    const stateIds = this.tokenizer.encode(
      serializeState(state).replaceAll(this.tokenizer.mask_token, " "),
    );

    const items: PreparedItem[] = [];
    const internal: InternalQuestion[] = [];
    for (const qid of ids) {
      validateQuestionId(qid);
      const question = toInternal(questions[qid]);
      const prefix = buildPrefix(this.tokenizer, question, headMaxLen, true);
      const built = finishSequence(
        this.tokenizer,
        prefix.ids,
        prefix.markers,
        stateIds,
        maxLen,
        truncateLeft,
      );
      if (built.markers.length !== renderOptions(question).length) {
        throw new Error(`Question ${qid} has too many options for the token budget`);
      }
      items.push({
        ids: built.ids,
        markers: built.markers,
        qtype: QTYPES[question.t],
        options: prefix.stats,
        state_stats: built.stats,
      });
      internal.push(question);
    }
    return { items, internal, ids };
  }

  // ---- collation -------------------------------------------------------

  /** Pad a batch into the five input tensors the graph takes. */
  collate(items: PreparedItem[]): {
    inputIds: MlxArrayLike;
    attentionMask: MlxArrayLike;
    markerPos: MlxArrayLike;
    markerMask: MlxArrayLike;
    qtype: MlxArrayLike;
    length: number;
    count: number;
  } {
    const { mx } = this;
    const n = items.length;
    let length = Math.max(...items.map((item) => item.ids.length));
    // Shape bucketing. A compiled graph is retraced for every new (batch, length, markers) shape,
    // a few milliseconds each; rounding the padded length collapses real traffic onto few shapes.
    // Padding is masked out, so it is semantically neutral — but it is not free once the batch is
    // token-bound, which is why the default is off.
    if (this.lengthBucket > 0) {
      length = Math.ceil(length / this.lengthBucket) * this.lengthBucket;
    }
    const count = Math.max(2, ...items.map((item) => item.markers.length));
    const padId = this.tokenizer.pad_token_id;

    const inputIds = new Array<number>(n * length).fill(padId);
    const attention = new Array<number>(n * length).fill(0);
    const markerPos = new Array<number>(n * count).fill(0);
    const markerMask = new Array<number>(n * count).fill(0);
    const qtype = new Array<number>(n).fill(0);

    items.forEach((item, row) => {
      item.ids.forEach((id, column) => {
        inputIds[row * length + column] = id;
        attention[row * length + column] = 1;
      });
      item.markers.forEach((marker, column) => {
        markerPos[row * count + column] = marker;
        markerMask[row * count + column] = 1;
      });
      qtype[row] = item.qtype;
    });

    return {
      inputIds: this.tensor(inputIds, mx.int32, [n, length]),
      attentionMask: this.tensor(attention, mx.bool, [n, length]),
      markerPos: this.tensor(markerPos, mx.int32, [n, count]),
      markerMask: this.tensor(markerMask, mx.bool, [n, count]),
      qtype: this.tensor(qtype, mx.int32, [n]),
      length,
      count,
    };
  }

  private tensor(values: readonly number[], dtype: MlxDType, shape?: number[]): MlxArrayLike {
    const array = this.mx.array(values, dtype);
    return shape === undefined ? array : array.reshape(shape);
  }

  // ---- forward ---------------------------------------------------------

  /**
   * One forward over already-prepared items, flattened to the host.
   *
   * `asyncEval` keeps the event loop responsive while the GPU works, which is what a server wants.
   * Note the asymmetry with `tidy`: `mx.tidy` disposes intermediates when its callback returns, so
   * it must not wrap an in-flight async evaluation — the service path turns it off and disposes its
   * feeds explicitly.
   */
  async forwardItems(
    items: PreparedItem[],
    opts: { asyncEval?: boolean; tidy?: boolean } = {},
  ): Promise<ForwardResult> {
    const { mx } = this;
    const feeds = this.collate(items);
    const masks = this.masksFor(feeds.attentionMask, items);
    const run = (): { logits: MlxArrayLike; action: MlxArrayLike } =>
      this.compiledForward
        ? this.compiledForward(
            feeds.inputIds,
            masks.full,
            masks.local,
            feeds.markerPos,
            feeds.markerMask,
            feeds.qtype,
          )
        : this.model.forward(
            feeds.inputIds,
            masks.full,
            masks.local,
            feeds.markerPos,
            feeds.markerMask,
            feeds.qtype,
          );

    // Node frees an MLX array only when its JS handle is collected, so an eager graph keeps every
    // intermediate alive; mx.tidy disposes them as soon as the forward returns.
    const useTidy = opts.tidy ?? this.tidy;
    const { logits, action } = useTidy ? mx.tidy(run) : run();
    if (opts.asyncEval) await mx.asyncEval(logits, action);
    else mx.eval(logits, action);

    const result: ForwardResult = {
      logits: logits.reshape([-1]).toTypedArray() as Float32Array,
      action: action.reshape([-1]).toTypedArray() as Float32Array,
      count: feeds.count,
      nAct: action.shape[1] as number,
    };
    // The per-batch input tensors are scratch; the results are already copied to the host.
    // Masks are *not* disposed: they may be cached, and disposing a just-cached mask crashes the
    // next cache hit with a bare std::invalid_argument.
    mx.dispose(feeds.inputIds, feeds.attentionMask, feeds.markerPos, feeds.markerMask, feeds.qtype);
    return result;
  }

  /**
   * Attention masks for a batch, built outside the compiled region.
   *
   * The sliding-window mask is an `L x L` boolean built from `arange(L)`, which cannot be
   * shape-polymorphic, so the compiled graph receives it as an input. The cache key carries the row
   * lengths as well as `(batch, length)`, because the padding layout changes the mask for a given
   * padded shape.
   */
  maskInputs(
    attentionMask: MlxArrayLike,
    items: PreparedItem[],
  ): { full: MlxArrayLike; local: MlxArrayLike } {
    return this.masksFor(attentionMask, items);
  }

  private masksFor(
    attentionMask: MlxArrayLike,
    items: PreparedItem[],
  ): { full: MlxArrayLike; local: MlxArrayLike } {
    const [b, L] = attentionMask.shape as [number, number];
    const key = `${b}:${L}:${items.map((item) => item.ids.length).join(",")}`;
    if (this.cacheMasks) {
      const hit = this.maskCache.get(key);
      if (hit !== undefined) return hit;
    }
    const masks = attentionMasks(this.mx, attentionMask, this.model.cfg.local_attention);
    const entry = { full: masks.full_attention, local: masks.sliding_attention };
    if (this.cacheMasks) {
      // Eviction drops the reference and lets GC reclaim it. It must not `dispose()`: a mask that
      // is still serving the in-flight forward would be freed underneath it.
      if (this.maskCache.size > 64) this.maskCache.clear();
      this.maskCache.set(key, entry);
    }
    return entry;
  }

  // ---- readout ---------------------------------------------------------

  /** Calibrate raw logits and action rows into the public answer payload. */
  shapeItems(
    items: PreparedItem[],
    internal: InternalQuestion[],
    ids: string[],
    raw: ForwardResult,
  ): Record<string, Answer> {
    const answers: Record<string, Answer> = {};
    items.forEach((item, row) => {
      const question = internal[row];
      if (question === undefined) throw new Error(`Missing internal question for row ${row}`);
      const k = item.markers.length;
      const scale =
        this.temperatureByOptions[tempBucket(item.qtype, k)] ?? this.temperature[item.qtype] ?? 1;
      const logits = Array.from(raw.logits.subarray(row * raw.count, row * raw.count + k));
      // Temperature first, then softmax: a scale below 1 sharpens, above 1 softens.
      const p = softmax(logits.map((value) => value / scale));
      const actionRow = Array.from(raw.action.subarray(row * raw.nAct, row * raw.nAct + raw.nAct));
      const actProbability = softmax(actionRow)[0] ?? 0;
      const qid = ids[row];
      if (qid === undefined) throw new Error(`Missing question id for row ${row}`);
      answers[qid] = shapeAnswer(question, p, actProbability);
    });
    return answers;
  }

  /** Compose upstream's `usage` object from the per-question preparation diagnostics. */
  private usageFor(prepared: Prepared): Usage {
    const stats = prepared.items.map((item) => item.state_stats);
    const dropped = stats.reduce((worst, stat) => Math.max(worst, stat.state_tokens_dropped), 0);
    const usage: Usage = {
      input_tokens: prepared.items.reduce((total, item) => total + item.ids.length, 0),
      output_tokens: 0,
      state_tokens: stats[0]?.state_tokens ?? 0,
      state_tokens_dropped: dropped,
      truncated: dropped > 0,
      truncated_questions: prepared.ids.filter((_, index) => stats[index]?.truncated === true),
    };
    const collapsed = collapsedOptions(prepared.ids, prepared.items);
    if (Object.keys(collapsed).length > 0) usage.options = collapsed;
    return usage;
  }

  /**
   * Answer every question in one request.
   *
   * Questions are answered in `batchSize` chunks; rows within a chunk are independent and padding is
   * masked, so a batched answer equals the solo answer (the batching test asserts it).
   */
  async predict(state: unknown, questions: Record<string, unknown>): Promise<Prediction> {
    const prepared = this.prepare(state, questions);
    const answers: Record<string, Answer> = {};
    for (let start = 0; start < prepared.items.length; start += this.batchSize) {
      const chunk = prepared.items.slice(start, start + this.batchSize);
      const raw = await this.forwardItems(chunk);
      Object.assign(
        answers,
        this.shapeItems(chunk, prepared.internal.slice(start), prepared.ids.slice(start), raw),
      );
    }
    return { model: "laya-rl-agent", answers, usage: this.usageFor(prepared) };
  }

  /**
   * Raw pre-temperature logits for the first question, for numerical comparison against the Python
   * reference. Runs eagerly: `mx.compile` may fuse or reorder, and this is a diagnostics path.
   */
  rawLogits(state: unknown, questions: Record<string, unknown>): number[] {
    const prepared = this.prepare(state, questions);
    const first = prepared.items[0];
    if (first === undefined) throw new Error("questions must not be empty");
    const feeds = this.collate([first]);
    const masks = this.masksFor(feeds.attentionMask, [first]);
    const { logits } = this.model.forward(
      feeds.inputIds,
      masks.full,
      masks.local,
      feeds.markerPos,
      feeds.markerMask,
      feeds.qtype,
    );
    this.mx.eval(logits);
    const k = first.markers.length;
    return Array.from(logits.reshape([-1]).toTypedArray() as Float32Array).slice(0, k);
  }
}

function dtypeFor(mx: MlxCore, name: DTypeName): MlxDType {
  if (name === "float32") return mx.float32;
  if (name === "float16") return mx.float16;
  if (name === "bfloat16") return mx.bfloat16;
  throw new Error(`Unknown dtype: ${String(name)}`);
}

/**
 * Reject a checkpoint whose calibration is unusable, before anything is loaded onto the GPU.
 *
 * Upstream's rule: exactly three positional temperatures, and every declared temperature finite and
 * positive. Values outside `[0.5, 5]` are *not* an error — they are clamped and warned about,
 * because the shipped checkpoints contain one.
 */
function assertUsableTemperatures(cfg: AgentConfig): void {
  const temperature = cfg.temperature ?? [1.0, 1.0, 1.0];
  if (!Array.isArray(temperature) || temperature.length !== 3) {
    throw new Error("Calibration temperatures must be finite and positive");
  }
  for (const value of [...temperature, ...Object.values(cfg.temperature_by_options ?? {})]) {
    const parsed = coerceTemperature(value);
    if (parsed === null || parsed <= 0) {
      throw new Error("Calibration temperatures must be finite and positive");
    }
  }
}

/** Load a checkpoint. Convenience wrapper around {@link MlxAgent.load}. */
export function load(modelDir: string, options: LoadOptions = {}): MlxAgent {
  return MlxAgent.load(modelDir, options);
}
