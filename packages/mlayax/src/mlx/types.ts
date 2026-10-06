/**
 * The slice of MLX's API this runtime uses, typed.
 *
 * Deliberately not `any`. The native addon is untyped JavaScript, so the port used to thread it
 * through as `any`; that hides real mistakes — a misspelled `mx.takeAlongAxes` or an `axis` that
 * does not exist only fails at runtime, on the one code path that has a GPU. Naming the surface
 * costs ~150 lines and turns those into compile errors.
 *
 * Every method here corresponds to a call the port actually makes. It is not an attempt to type
 * MLX; anything not listed is intentionally absent rather than accidentally forgotten.
 *
 * Dtypes are opaque handles compared by identity and passed back into `astype`/`array`, so they are
 * typed as an opaque `MlxDType` rather than a string union — getting one wrong is a correctness bug
 * (see the fp16 upcast trap in `model.ts`), not a formatting one.
 */

/** An opaque MLX dtype handle (`mx.float16`, `mx.int32`, …). */
export type MlxDType = object;

/** An opaque MLX device (`mx.gpu`, `mx.cpu`). */
export type MlxDevice = object;

/** A shape, as MLX reports it. */
export type MlxShape = number[];

/** Any MLX scalar-or-array argument: an array, a JS number, or a boolean. */
export type MlxOperand = MlxArray | number | boolean;

/**
 * An MLX array handle.
 *
 * `shape` and `dtype` are read directly off the native object; reading them forces the array to be
 * materialised, which is why the graph code passes shapes as separate tensors instead.
 */
export interface MlxArray {
  readonly shape: MlxShape;
  readonly ndim: number;
  readonly size: number;
  readonly dtype: MlxDType;

  astype(dtype: MlxDType): MlxArray;
  reshape(shape: readonly number[]): MlxArray;
  squeeze(axis?: number): MlxArray;
  /** `x[S]` / `x[i]` — the arbitrary-index form, used only at load time (it is not shape-inferable). */
  index(...indices: unknown[]): MlxArray;
  /** Copy to host memory. The dtype follows the array's dtype (fp32 results here). */
  toTypedArray(): Float32Array | Int32Array;
  toString(): string;
}

/** A `mx.Slice` specification, for `x.index(mx.Slice(lo, hi))`. */
export type MlxSlice = unknown;

/** The fused kernels under `mx.fast`. */
export interface MlxFast {
  layerNorm(x: MlxArray, weight: MlxArray, bias: MlxArray | null, eps: number): MlxArray;
  /** `fast.rope(x, dims, traditional, base, scale, offset)`. */
  rope(
    x: MlxArray,
    dims: number,
    traditional: boolean,
    base: number,
    scale: number,
    offset: number,
  ): MlxArray;
  /**
   * The fused attention kernel.
   *
   * Not shape-inferable: under shapeless compilation it bakes the traced shape, so a later call
   * with a new length fails to broadcast its mask against stale scores. Hence
   * `DecisionModel`'s `manualAttention` option.
   */
  scaledDotProductAttention(
    q: MlxArray,
    k: MlxArray,
    v: MlxArray,
    scale: number,
    mask: MlxArray,
  ): MlxArray;
}

/** The MLX core namespace: the functions the runtime calls. */
export interface MlxCore {
  // ---- dtypes and devices ----
  readonly bool: MlxDType;
  readonly int32: MlxDType;
  readonly float32: MlxDType;
  readonly float16: MlxDType;
  readonly bfloat16: MlxDType;
  readonly cpu: MlxDevice;
  readonly gpu: MlxDevice;
  setDefaultDevice(device: MlxDevice): void;
  setWiredLimit(bytes: number): void;
  setCacheLimit(bytes: number): void;
  defaultStream(device: MlxDevice): unknown;
  setDefaultStream(stream: unknown): void;

  // ---- construction and IO ----
  array(values: MlxOperand | readonly number[], dtype?: MlxDType): MlxArray;
  arange(stop: number): MlxArray;
  load(path: string): Record<string, MlxArray>;
  Slice(start: number, stop: number): MlxSlice;
  random?: unknown;

  // ---- evaluation ----
  eval(...arrays: MlxArray[]): void;
  asyncEval(...arrays: MlxArray[]): Promise<void>;
  /** Runs `fn` and disposes the intermediates it creates when it returns. */
  tidy<T>(fn: () => T): T;
  dispose(...arrays: MlxArray[]): void;
  /** Compiles `fn`; `shapeless` reuses one graph for every shape (see `DecisionModel`). */
  compile<T extends (...args: never[]) => unknown>(fn: T, shapeless?: boolean): T;

  // ---- elementwise ----
  add(a: MlxOperand, b: MlxOperand): MlxArray;
  subtract(a: MlxOperand, b: MlxOperand): MlxArray;
  multiply(a: MlxOperand, b: MlxOperand): MlxArray;
  divide(a: MlxOperand, b: MlxOperand): MlxArray;
  negative(a: MlxArray): MlxArray;
  abs(a: MlxArray): MlxArray;
  maximum(a: MlxOperand, b: MlxOperand): MlxArray;
  log(a: MlxArray): MlxArray;
  erf(a: MlxArray): MlxArray;

  // ---- comparison and selection ----
  less(a: MlxOperand, b: MlxOperand): MlxArray;
  lessEqual(a: MlxOperand, b: MlxOperand): MlxArray;
  greaterEqual(a: MlxOperand, b: MlxOperand): MlxArray;
  equal(a: MlxOperand, b: MlxOperand): MlxArray;
  logicalAnd(a: MlxArray, b: MlxArray): MlxArray;
  logicalOr(a: MlxArray, b: MlxArray): MlxArray;
  logicalNot(a: MlxArray): MlxArray;
  where(condition: MlxArray, onTrue: MlxOperand, onFalse: MlxOperand): MlxArray;

  // ---- reductions ----
  max(a: MlxArray, axis?: number): MlxArray;
  sum(a: MlxArray, axis?: number): MlxArray;
  softmax(a: MlxArray, axis: number): MlxArray;

  // ---- shape manipulation ----
  transpose(a: MlxArray, axes?: readonly number[]): MlxArray;
  reshape(a: MlxArray, shape: readonly number[]): MlxArray;
  unflatten(a: MlxArray, axis: number, shape: readonly number[]): MlxArray;
  flatten(a: MlxArray, startAxis?: number, endAxis?: number): MlxArray;
  expandDims(a: MlxArray, axis: number): MlxArray;
  concatenate(parts: readonly MlxArray[], axis: number): MlxArray;
  stack(parts: readonly MlxArray[], axis: number): MlxArray;
  takeAlongAxis(a: MlxArray, indices: MlxArray, axis: number): MlxArray;
  /**
   * Present only so the port can *demonstrate* why it is not used for markers: with a `[b, count]`
   * index it prepends a batch dimension instead of replacing the axis. See `test/mlx.binding.test.ts`.
   */
  take(a: MlxArray, indices: MlxArray, axis: number): MlxArray;

  // ---- linear algebra ----
  matmul(a: MlxArray, b: MlxArray): MlxArray;
  addmm(bias: MlxArray, a: MlxArray, b: MlxArray): MlxArray;

  // ---- fused kernels ----
  readonly fast: MlxFast;
}

/** The shape of the vendored `core.cjs` module. */
export interface MlxModule {
  core: MlxCore;
  mx: MlxCore;
  /** Present on the addon; unused by the runtime. */
  default?: unknown;
}
