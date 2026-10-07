/**
 * ModernBERT encoder plus Laya's decision heads, on MLX.
 *
 * A port of `laya_mlx/model.py`. Every kernel the Python version calls
 * (`fast.layer_norm`, `fast.rope`, `fast.scaled_dot_product_attention`) has a one-to-one
 * counterpart in node-mlx, so this is the same execution path: Metal, fp16, unified memory.
 *
 * Four behaviours here were each learned the hard way. They are load-bearing, not style:
 *
 * 1. **Every scalar is created in the activation's own dtype.** In Python a literal like `1` or
 *    `sqrt(2)` is weakly typed and leaves an fp16 array fp16. In JavaScript numbers are always
 *    float64/float32, so `mx.array(1)` promotes, and MLX lifts *the rest of the graph* with it —
 *    measured at ~20 % end to end, with no error and no wrong answer, just a slower forward. A bare
 *    `0` in `mx.maximum(intArray, 0)` is worse: it returns float32 and the following gather rejects
 *    it outright.
 * 2. **Weights are pre-split at load time** (`prepareWeights`). MLX's `Slice` primitive cannot infer
 *    output shapes, so any slicing inside the graph — including `mx.split` and `take` with an
 *    integer — makes the graph non-shape-polymorphic. Splitting the fused QKV and gated-MLP weights
 *    eagerly is mathematically identical (they were contiguous output blocks) and is what makes
 *    `mx.compile(shapeless)` possible at all.
 * 3. **Masks are inputs, not constructed here** (see `attentionMasks`). `arange(L)` and an `L x L`
 *    comparison cannot be shape-polymorphic, so the caller builds them outside the compiled region
 *    and caches them.
 * 4. **Markers are gathered with `takeAlongAxis`.** `mx.take` with a `[b, count]` index *prepends*
 *    the batch dimension and yields `[b, b, count, H]`, which silently broadcasts instead of failing.
 *
 * Each of these is pinned by a test in `test/mlx.binding.test.ts` that runs against the real binding
 * without needing the checkpoint.
 *
 * The port runs both a fused-attention profile (default) and a manual one; see `manualAttention`.
 */

import type { MlxArray, MlxCore, MlxDType } from "./types.js";

/** The encoder half of the checkpoint config (`encoder/config.json`). */
export interface EncoderConfig {
  vocab_size: number;
  hidden_size: number;
  intermediate_size: number;
  num_hidden_layers: number;
  num_attention_heads: number;
  norm_eps: number;
  hidden_activation: string;
  local_attention: number;
  global_attn_every_n_layers: number;
  global_rope_theta: number;
  local_rope_theta: number;
  layer_types: string[];
  rope_parameters: Record<string, { rope_type?: string; rope_theta?: number }>;
}

/** Checkpoint parameters, keyed by upstream's MLX module paths. */
export type Weights = Record<string, MlxArray>;

/** The Laya agent config (`rl_agent_config.json`); only the fields the model reads are named. */
export interface AgentConfig {
  head_layers?: number;
  max_len?: number;
  head_max_len?: number;
  temperature?: number[];
  temperature_by_options?: Record<string, unknown>;
  [key: string]: unknown;
}

/** Arrangement options. None of these change the maths, only how it is scheduled. */
export interface ModelOptions {
  /** Hold transposes resident instead of emitting a transpose node per call. */
  pretranspose?: boolean;
  /** Use the fused `addmm` for biased linears. */
  addmm?: boolean;
  /** Manual attention instead of `fast.sdpa`. Required for shapeless compilation. */
  manualAttention?: boolean;
}

/** Parse and validate the encoder config. */
export function encoderConfigFrom(json: Record<string, unknown>): EncoderConfig {
  if (json.model_type !== "modernbert") {
    throw new Error(`Unsupported encoder: ${String(json.model_type)}`);
  }
  if (json.hidden_activation !== "gelu") {
    throw new Error(`Unsupported encoder activation: ${String(json.hidden_activation)}`);
  }
  const num = (key: string): number => json[key] as number;
  const cfg: EncoderConfig = {
    vocab_size: num("vocab_size"),
    hidden_size: num("hidden_size"),
    intermediate_size: num("intermediate_size"),
    num_hidden_layers: num("num_hidden_layers"),
    num_attention_heads: num("num_attention_heads"),
    norm_eps: (json.norm_eps as number | undefined) ?? 1e-5,
    hidden_activation: json.hidden_activation,
    local_attention: (json.local_attention as number | undefined) ?? 128,
    global_attn_every_n_layers: (json.global_attn_every_n_layers as number | undefined) ?? 3,
    global_rope_theta: (json.global_rope_theta as number | undefined) ?? 160000.0,
    local_rope_theta: (json.local_rope_theta as number | undefined) ?? 10000.0,
    layer_types: (json.layer_types as string[] | undefined) ?? [],
    rope_parameters: (json.rope_parameters as EncoderConfig["rope_parameters"] | undefined) ?? {},
  };

  const headDim = cfg.hidden_size / cfg.num_attention_heads;
  if (!Number.isInteger(headDim) || headDim % 2) {
    throw new Error("ModernBERT requires an even, integral attention head dimension");
  }
  if (cfg.layer_types.length === 0) {
    cfg.layer_types = Array.from({ length: cfg.num_hidden_layers }, (_, i) =>
      i % cfg.global_attn_every_n_layers === 0 ? "full_attention" : "sliding_attention",
    );
  }
  for (const kind of new Set(cfg.layer_types)) {
    const params = cfg.rope_parameters[kind] ?? {};
    if ((params.rope_type ?? "default") !== "default") {
      throw new Error("Only default (unscaled) ModernBERT RoPE is supported");
    }
  }
  return cfg;
}

/** The RoPE base frequency for a layer kind. */
export function ropeBase(cfg: EncoderConfig, kind: string): number {
  const fallback = kind === "full_attention" ? cfg.global_rope_theta : cfg.local_rope_theta;
  return Number(cfg.rope_parameters[kind]?.rope_theta ?? fallback);
}

/**
 * Map upstream PyTorch parameter names onto the MLX module layout (`model.py sanitize_weights`).
 *
 * Two renames: the merged `in_proj` (an MLX *module* attribute in the head) and the flat
 * `scorer.*` / `act_head.*` names, which become `.layers.N` because the heads are sequenced
 * modules. A collision afterwards means the checkpoint genuinely has two parameters claiming the
 * same slot, which would silently drop one of them.
 */
export function sanitizeWeights(weights: Weights): Weights {
  const result: Weights = {};
  for (const [original, value] of Object.entries(weights)) {
    let name = original;
    name = name.replace(".in_proj_weight", ".in_proj.weight");
    name = name.replace(".in_proj_bias", ".in_proj.bias");
    for (const prefix of ["scorer", "act_head"]) {
      if (name.startsWith(`${prefix}.`) && !name.startsWith(`${prefix}.layers.`)) {
        name = `${prefix}.layers.${name.slice(prefix.length + 1)}`;
      }
    }
    if (name in result) throw new Error(`Duplicate checkpoint parameter: ${name}`);
    result[name] = value;
  }
  return result;
}

/**
 * Boolean key masks: a full mask, and a sliding-window mask with inclusive local distance
 * `<= window // 2`.
 *
 * Built outside the compiled region on purpose (see the file header). The local mask keeps an
 * attention edge when either the position pair is within the window *or* the key is padding — the
 * second half of that `or` is what stops padding from being masked into the *query* side, which
 * would make padded rows attend to nothing and produce NaN downstream.
 */
export function attentionMasks(
  mx: MlxCore,
  attentionMask: MlxArray,
  window: number,
): { full_attention: MlxArray; sliding_attention: MlxArray } {
  const valid = attentionMask.astype(mx.bool);
  const [b, L] = valid.shape as [number, number];
  const full = valid.reshape([b, 1, 1, L]);
  const positions = mx.arange(L);
  const distance = mx.abs(mx.subtract(positions.reshape([L, 1]), positions.reshape([1, L])));
  const local = mx.lessEqual(distance, mx.array(Math.floor(window / 2), mx.int32));
  const localMask = mx.logicalAnd(
    mx.logicalOr(local.reshape([1, 1, L, L]), mx.logicalNot(valid.reshape([b, 1, L, 1]))),
    full,
  );
  return { full_attention: full, sliding_attention: localMask };
}

/**
 * GELU as `x * (1 + erf(x / sqrt(2))) / 2`, with every constant in the activation's own dtype.
 *
 * Module-level and exported so the dtype invariant can be tested without loading a checkpoint — see
 * hazard 1 in the file header. node-mlx's `nn.gelu` is an `mx.compile` wrapper, so the plain ops are
 * used instead.
 *
 * In Python these literals are weakly typed and leave an fp16 array fp16. In JavaScript
 * `mx.array(1)` is float32, and MLX promotes mixed operands, so bare `1` / `2` / `sqrt(2)` would lift
 * every fp16 activation to fp32 for the rest of the graph: measured at ~20 % end to end, with no
 * error and no wrong answer.
 */
export function gelu(mx: MlxCore, x: MlxArray): MlxArray {
  const dt = x.dtype;
  const one = mx.array(1, dt);
  const two = mx.array(2, dt);
  const sqrt2 = mx.array(Math.SQRT2, dt);
  return mx.multiply(x, mx.divide(mx.add(one, mx.erf(mx.divide(x, sqrt2))), two));
}

/**
 * ReLU with the zero in the activation's dtype.
 *
 * The upcast is real and silent: `mx.maximum(intArray, 0)` returns float32, and would lift the rest of
 * the graph (~20 % cost, no error).
 *
 * This docstring used to add that "the gather that follows rejects a float32 index outright". That is
 * true of `mx.take` and **false of `mx.takeAlongAxis`**, which is what the marker path below actually
 * calls — it casts float32 indices, eagerly and under `mx.compile`. Verified in
 * `test/mlx.binding.test.ts` (hazard 2). So the int32 form is about not promoting, not about not
 * crashing.
 */
export function relu(mx: MlxCore, x: MlxArray): MlxArray {
  return mx.maximum(x, mx.array(0, x.dtype));
}

export class DecisionModel {
  readonly mx: MlxCore;
  readonly cfg: EncoderConfig;
  readonly w: Weights;
  readonly agentCfg: AgentConfig;
  readonly dtype: MlxDType;
  /** Options that only change how the same maths is arranged. */
  readonly opts: Required<ModelOptions>;

  /**
   * Weights sliced into q/k/v and value/gate halves at load time (see `prepareWeights`).
   *
   * Declared as a definite-assignment field rather than optional so the hot paths do not need a
   * null check: `load()` always calls `prepareWeights()` before any forward.
   */
  private split: Weights = {};
  private transposed: Weights | null = null;

  constructor(
    mx: MlxCore,
    cfg: EncoderConfig,
    weights: Weights,
    agentCfg: AgentConfig,
    dtype: MlxDType,
    opts: ModelOptions = {},
  ) {
    this.mx = mx;
    this.cfg = cfg;
    this.w = weights;
    this.agentCfg = agentCfg;
    this.dtype = dtype;
    this.opts = {
      pretranspose: opts.pretranspose ?? false,
      addmm: opts.addmm ?? false,
      manualAttention: opts.manualAttention ?? false,
    };
  }

  // ---- weight access ---------------------------------------------------

  private weight(name: string): MlxArray {
    const value = this.w[name];
    if (value === undefined) throw new Error(`Missing checkpoint parameter: ${name}`);
    return value;
  }

  private optionalWeight(name: string): MlxArray | null {
    return this.w[name] ?? null;
  }

  private splitWeight(name: string): MlxArray {
    const value = this.split[name];
    if (value === undefined) throw new Error(`Missing pre-split weight: ${name}`);
    return value;
  }

  /**
   * Prepare constant weights once, at load time — eagerly, where slicing is still allowed.
   *
   * See hazard 2 in the file header: this is what lets the graph itself stay free of slices.
   */
  prepareWeights(): void {
    const { mx } = this;
    const H = this.cfg.hidden_size;
    const inter = this.inter;
    const cut = (w: MlxArray, lo: number, hi: number): MlxArray => {
      const part = w.index(mx.Slice(lo, hi));
      mx.eval(part);
      return part;
    };

    const split: Weights = {};
    for (let i = 0; i < this.cfg.num_hidden_layers; i++) {
      const attn = `encoder.layers.${i}.attn`;
      const qkv = this.weight(`${attn}.Wqkv.weight`);
      split[`${attn}.Wq`] = cut(qkv, 0, H);
      split[`${attn}.Wk`] = cut(qkv, H, 2 * H);
      split[`${attn}.Wv`] = cut(qkv, 2 * H, 3 * H);

      const mlp = `encoder.layers.${i}.mlp`;
      const wi = this.weight(`${mlp}.Wi.weight`);
      split[`${mlp}.Wvalue`] = cut(wi, 0, inter);
      split[`${mlp}.Wgate`] = cut(wi, inter, 2 * inter);
    }

    const headLayers = this.agentCfg.head_layers ?? 2;
    for (let i = 0; i < headLayers; i++) {
      const prefix = `head.layers.${i}.self_attn`;
      const weight = this.weight(`${prefix}.in_proj.weight`);
      split[`${prefix}.Wq`] = cut(weight, 0, H);
      split[`${prefix}.Wk`] = cut(weight, H, 2 * H);
      split[`${prefix}.Wv`] = cut(weight, 2 * H, 3 * H);
      const bias = this.optionalWeight(`${prefix}.in_proj.bias`);
      if (bias !== null) {
        split[`${prefix}.bq`] = cut(bias, 0, H);
        split[`${prefix}.bk`] = cut(bias, H, 2 * H);
        split[`${prefix}.bv`] = cut(bias, 2 * H, 3 * H);
      }
    }
    this.split = split;

    if (!this.opts.pretranspose) return;
    const transposed: Weights = {};
    for (const [name, value] of Object.entries(this.w)) {
      if (name.endsWith(".weight") && value.ndim === 2) {
        const t = mx.transpose(value);
        mx.eval(t);
        transposed[name] = t;
      }
    }
    this.transposed = transposed;
  }

  // ---- helpers ---------------------------------------------------------

  private lin(x: MlxArray, prefix: string): MlxArray {
    const bias = this.optionalWeight(`${prefix}.bias`);
    const stored = this.transposed?.[`${prefix}.weight`];
    const w = stored ?? this.mx.transpose(this.weight(`${prefix}.weight`));
    if (bias !== null && this.opts.addmm) {
      // Same maths as matmul + add, but MLX fuses the pair (what nn.Linear does for biased layers).
      return this.mx.addmm(bias, x, w);
    }
    const y = this.mx.matmul(x, w);
    return bias === null ? y : this.mx.add(y, bias);
  }

  private ln(x: MlxArray, prefix: string, eps: number): MlxArray {
    return this.mx.fast.layerNorm(
      x,
      this.weight(`${prefix}.weight`),
      this.optionalWeight(`${prefix}.bias`),
      eps,
    );
  }

  /** A linear layer over one of the pre-split constant weights — no slicing in the graph. */
  private linSplit(x: MlxArray, prefix: string, part: string): MlxArray {
    const w = this.splitWeight(`${prefix}.W${part}`);
    const bias = this.split[`${prefix}.b${part}`] ?? null;
    const y = this.mx.matmul(x, this.mx.transpose(w));
    return bias === null ? y : this.mx.add(y, bias);
  }

  // node-mlx's nn.gelu / nn.relu are mx.compile wrappers, so the module-level plain-op versions are
  // used instead. Every constant carries the activation's dtype — see hazards 1 and 2.
  private gelu(x: MlxArray): MlxArray {
    return gelu(this.mx, x);
  }

  private relu(x: MlxArray): MlxArray {
    return relu(this.mx, x);
  }

  private get heads(): number {
    return this.cfg.num_attention_heads;
  }

  private get headDim(): number {
    return this.cfg.hidden_size / this.cfg.num_attention_heads;
  }

  private get inter(): number {
    return this.cfg.intermediate_size;
  }

  // ---- encoder ---------------------------------------------------------

  /**
   * One attention projection `[b, L, H] -> [b, heads, L, headDim]`.
   *
   * `unflatten` + `transpose` only: an axes permutation is static, so this stays shape-polymorphic.
   * A `reshape` here would need `b`, which is symbolic under shapeless compilation.
   */
  private toHeads(x: MlxArray): MlxArray {
    return this.mx.transpose(this.mx.unflatten(x, -1, [this.heads, this.headDim]), [0, 2, 1, 3]);
  }

  /**
   * Attention: the fused kernel by default, or a shape-polymorphic manual form when the graph is
   * compiled shapeless.
   *
   * MLX's `fast.sdpa` bakes the traced shape, so a new mask shape cannot broadcast against stale
   * scores — it fails with `Shapes (2,1,1,43) and (1,16,37,37) cannot be broadcast`. The manual form
   * costs ~2x at L=64 and ~5x at L=256 in isolation, but at Laya's short decision lengths the two
   * are even, and only the manual form allows one graph for every length.
   */
  private attend(q: MlxArray, k: MlxArray, v: MlxArray, scale: number, mask: MlxArray): MlxArray {
    const { mx } = this;
    if (!this.opts.manualAttention) {
      return mx.fast.scaledDotProductAttention(q, k, v, scale, mask);
    }
    // MLX's fused kernel accumulates in float32 whatever the input precision, so the manual version
    // has to as well or the port stops matching Python: fp32 parity went from exact to 9e-4 when
    // this ran in fp16.
    const dt = q.dtype;
    const q32 = q.astype(mx.float32);
    const scores = mx.multiply(
      mx.matmul(q32, mx.transpose(k.astype(mx.float32), [0, 1, 3, 2])),
      mx.array(scale, mx.float32),
    );
    const masked = mx.where(mask, scores, mx.array(-1e4, mx.float32));
    const p = mx.softmax(masked, -1).astype(dt);
    return mx.matmul(p, v);
  }

  private attention(x: MlxArray, prefix: string, kind: string, mask: MlxArray): MlxArray {
    const { mx } = this;
    const q = this.toHeads(this.linSplit(x, prefix, "q"));
    const k = this.toHeads(this.linSplit(x, prefix, "k"));
    const v = this.toHeads(this.linSplit(x, prefix, "v"));
    const base = ropeBase(this.cfg, kind);
    const rope = (t: MlxArray): MlxArray => mx.fast.rope(t, this.headDim, false, base, 1.0, 0);
    const out = this.attend(rope(q), rope(k), v, this.headDim ** -0.5, mask);
    // [b, heads, L, headDim] -> [b, L, H] by merging the trailing dims (static, polymorphic).
    return this.lin(mx.flatten(mx.transpose(out, [0, 2, 1, 3]), -2), `${prefix}.Wo`);
  }

  private mlp(x: MlxArray, prefix: string): MlxArray {
    const { mx } = this;
    // The gated up-projection is pre-split into its value and gate halves at load time.
    const value = this.linSplit(x, prefix, "value");
    const gate = this.linSplit(x, prefix, "gate");
    return this.lin(mx.multiply(this.gelu(value), gate), `${prefix}.Wo`);
  }

  private encoderLayer(x: MlxArray, index: number, mask: MlxArray): MlxArray {
    const { mx } = this;
    const prefix = `encoder.layers.${index}`;
    const kind = this.cfg.layer_types[index] ?? "full_attention";
    // Layer 0 skips attn_norm: ModernBERT's embedding already ends in a LayerNorm, so normalising
    // again would apply it twice.
    const normed = index === 0 ? x : this.ln(x, `${prefix}.attn_norm`, this.cfg.norm_eps);
    x = mx.add(x, this.attention(normed, `${prefix}.attn`, kind, mask));
    return mx.add(
      x,
      this.mlp(this.ln(x, `${prefix}.mlp_norm`, this.cfg.norm_eps), `${prefix}.mlp`),
    );
  }

  private encoder(
    inputIds: MlxArray,
    masks: { full_attention: MlxArray; sliding_attention: MlxArray },
  ): MlxArray {
    let x = this.ln(
      this.weight("encoder.embeddings.tok_embeddings.weight").index(inputIds),
      "encoder.embeddings.norm",
      this.cfg.norm_eps,
    );
    // Masks arrive as inputs rather than being built here — see hazard 3.
    for (let i = 0; i < this.cfg.num_hidden_layers; i++) {
      x = this.encoderLayer(x, i, masks[this.cfg.layer_types[i] as "full_attention"]);
    }
    return this.ln(x, "encoder.final_norm", this.cfg.norm_eps);
  }

  // ---- decision head ---------------------------------------------------

  /**
   * The decision head's own attention block, whose head count is derived from the width rather than
   * taken from the encoder config (`max(1, dims // 64)`).
   */
  private headAttention(x: MlxArray, prefix: string, mask: MlxArray): MlxArray {
    const { mx } = this;
    const dims = this.cfg.hidden_size;
    const heads = Math.max(1, Math.floor(dims / 64));
    if (dims % heads)
      throw new Error("Decision head dimensions must be divisible by its head count");
    const headDim = dims / heads;
    const toHeads = (t: MlxArray): MlxArray =>
      mx.transpose(mx.unflatten(t, -1, [heads, headDim]), [0, 2, 1, 3]);
    const q = toHeads(this.linSplit(x, prefix, "q"));
    const k = toHeads(this.linSplit(x, prefix, "k"));
    const v = toHeads(this.linSplit(x, prefix, "v"));
    const out = this.attend(q, k, v, headDim ** -0.5, mask);
    return this.lin(mx.flatten(mx.transpose(out, [0, 2, 1, 3]), -2), `${prefix}.out_proj`);
  }

  private headLayer(x: MlxArray, prefix: string, mask: MlxArray): MlxArray {
    const { mx } = this;
    x = mx.add(
      x,
      this.headAttention(this.ln(x, `${prefix}.norm1`, 1e-5), `${prefix}.self_attn`, mask),
    );
    // ReLU, not GELU: this is a PyTorch TransformerEncoderLayer, whose default is ReLU, even though
    // the encoder and the scoring head use GELU.
    const hidden = this.relu(this.lin(this.ln(x, `${prefix}.norm2`, 1e-5), `${prefix}.linear1`));
    return mx.add(x, this.lin(hidden, `${prefix}.linear2`));
  }

  private head(x: MlxArray, mask: MlxArray): MlxArray {
    const count = this.agentCfg.head_layers ?? 2;
    for (let i = 0; i < count; i++) {
      x = this.headLayer(x, `head.layers.${i}`, mask);
    }
    return x;
  }

  private scorer(x: MlxArray): MlxArray {
    let h = this.ln(x, "scorer.layers.0", 1e-5);
    h = this.gelu(this.lin(h, "scorer.layers.1"));
    return this.lin(h, "scorer.layers.3");
  }

  private actHead(x: MlxArray): MlxArray {
    const h = this.gelu(this.lin(x, "act_head.layers.0"));
    return this.lin(h, "act_head.layers.2");
  }

  /**
   * `DecisionModel.__call__`: marker logits `[b, markers]` and action logits `[b, n_act]`.
   *
   * Everything here is shape-polymorphic — no slices, no host-side shapes, no `mx.split` — so one
   * shapeless-compiled graph serves every sequence length and marker count.
   */
  forward(
    inputIds: MlxArray,
    fullMask: MlxArray,
    localMask: MlxArray,
    markerPos: MlxArray,
    markerMask: MlxArray,
    qtype: MlxArray,
  ): { logits: MlxArray; action: MlxArray } {
    const { mx } = this;

    let h = this.encoder(inputIds, { full_attention: fullMask, sliding_attention: localMask });
    // expandDims rather than reshape([b, 1, H]): `b` is symbolic.
    h = mx.add(h, mx.expandDims(this.weight("type_emb.weight").index(qtype), 1));
    h = this.head(h, fullMask);

    // h[b, marker, :] via takeAlongAxis, whose output follows the indices — no host-side b/count.
    // `mx.take` with a [b, count] index *prepends* the batch dim, giving [b, b, count, H], which
    // broadcasts silently instead of failing. Verified in `test/mlx.binding.test.ts`.
    //
    // The maximum against an int32 zero keeps the indices integral. Note this is a dtype correctness
    // choice, not a workaround: an earlier comment in this file claimed a bare `0` would make the
    // gather *reject* the float32 indices. That is true of `mx.take` but not of `takeAlongAxis`,
    // which is what this call uses and which accepts float32 indices both eagerly and under
    // `mx.compile` — measured, and pinned in `test/mlx.binding.test.ts` (hazard 2). The int32 form is
    // still right (the indices are integers, and the promotion is avoidable work), but it was never
    // the difference between working and crashing.
    const markers = mx.takeAlongAxis(
      h,
      mx.expandDims(mx.maximum(markerPos, mx.array(0, mx.int32)), -1),
      1,
    );

    let logits = this.scorer(markers).squeeze(-1).astype(mx.float32);
    logits = mx.where(markerMask, logits, -1e4);

    const p = mx.softmax(logits, -1);
    const k = mx.maximum(mx.sum(markerMask.astype(mx.float32), -1), 2).astype(mx.float32);
    const entropy = mx.divide(
      mx.negative(mx.sum(mx.multiply(p, mx.log(mx.maximum(p, 1e-9))), -1)),
      mx.log(k),
    );

    // The two largest probabilities, without slicing or mx.topk (Python sorts and reads the last
    // two). Ties collapse to a zero gap, exactly like the sorted form. A `where` is used rather
    // than a sort because sorting is not shape-inferable.
    const largest = mx.max(p, -1);
    const tied = mx.greaterEqual(
      mx.sum(mx.equal(p, mx.expandDims(largest, -1)).astype(mx.float32), -1),
      2,
    );
    const runnerUp = mx.where(
      tied,
      largest,
      mx.max(mx.where(mx.less(p, mx.expandDims(largest, -1)), p, mx.array(-1e4, p.dtype)), -1),
    );
    const features = mx.stack(
      [largest, mx.subtract(largest, runnerUp), entropy, mx.divide(k, 255.0)],
      -1,
    );

    // h[:, 0] — the CLS state — gathered with a zero index tensor derived from the marker mask:
    // sum(markerMask) -> [b], times zero -> zeros, expandDims -> [b, 1]. No slicing, no host shapes.
    const firstIndex = mx.expandDims(mx.multiply(mx.sum(markerMask.astype(mx.int32), -1), 0), -1);
    const pooled = mx.concatenate(
      [
        mx
          .reshape(mx.takeAlongAxis(h, mx.expandDims(firstIndex, -1), 1), [
            -1,
            this.cfg.hidden_size,
          ])
          .astype(mx.float32),
        features,
      ],
      -1,
    );
    const action = this.actHead(pooled.astype(this.weight("act_head.layers.0.weight").dtype));
    return { logits, action: action.astype(mx.float32) };
  }
}
