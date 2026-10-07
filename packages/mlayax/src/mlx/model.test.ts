/**
 * Config parsing and checkpoint-name mapping — the validation half of `model.ts`.
 *
 * Everything here is pure: `model.ts` imports only types, and these functions take plain data. So
 * this file runs on any platform with no payload, which matters because these are exactly the
 * branches that protect the load path — a checkpoint with an odd head dimension, an unsupported
 * activation, a renamed parameter that collides with an existing one. Before this, every one of them
 * was uncovered by the default suite (TASKS.md §10.5), which is a poor place for a guard to be
 * untested: a corrupt checkpoint is the case where you least want the error to be wrong.
 *
 * `attentionMasks` and the dtype helpers need real MLX and live in `test/mlx.options.test.ts`.
 */

import { describe, expect, it } from "vitest";
import type { Weights } from "./model.js";
import { encoderConfigFrom, ropeBase, sanitizeWeights } from "./model.js";

/** A minimal config that passes, so each test can vary one field. */
function baseConfig(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    model_type: "modernbert",
    hidden_activation: "gelu",
    vocab_size: 128,
    hidden_size: 32,
    intermediate_size: 64,
    num_hidden_layers: 2,
    num_attention_heads: 2,
    ...overrides,
  };
}

describe("encoderConfigFrom", () => {
  it("accepts a valid config and applies the documented defaults", () => {
    const cfg = encoderConfigFrom(baseConfig());
    expect(cfg.norm_eps).toBe(1e-5);
    expect(cfg.local_attention).toBe(128);
    expect(cfg.global_attn_every_n_layers).toBe(3);
    expect(cfg.global_rope_theta).toBe(160000);
    expect(cfg.local_rope_theta).toBe(10000);
    expect(cfg.rope_parameters).toEqual({});
    expect(cfg.hidden_activation).toBe("gelu");
  });

  it("respects explicit values instead of the defaults", () => {
    const cfg = encoderConfigFrom(
      baseConfig({
        norm_eps: 1e-6,
        local_attention: 64,
        global_attn_every_n_layers: 2,
        global_rope_theta: 1e6,
        local_rope_theta: 500,
      }),
    );
    expect(cfg.norm_eps).toBe(1e-6);
    expect(cfg.local_attention).toBe(64);
    expect(cfg.global_attn_every_n_layers).toBe(2);
    expect(cfg.global_rope_theta).toBe(1e6);
    // `local_rope_theta` is read only through `ropeBase`, since the config keeps both in
    // `rope_parameters`; asserting it here keeps the field honest.
    expect(cfg.local_rope_theta).toBe(500);
  });

  it("derives layer_types from global_attn_every_n_layers when the config omits them", () => {
    // The ModernBERT rule: every nth layer is full attention, the rest slide. Getting this backwards
    // would still run — with the wrong attention window in most layers — which is why it is asserted
    // as a sequence rather than a length.
    const cfg = encoderConfigFrom(
      baseConfig({ num_hidden_layers: 6, global_attn_every_n_layers: 3 }),
    );
    expect(cfg.layer_types).toEqual([
      "full_attention",
      "sliding_attention",
      "sliding_attention",
      "full_attention",
      "sliding_attention",
      "sliding_attention",
    ]);
  });

  it("keeps the layer_types the checkpoint supplies", () => {
    const supplied = ["sliding_attention", "full_attention"];
    expect(encoderConfigFrom(baseConfig({ layer_types: supplied })).layer_types).toEqual(supplied);
  });

  it("rejects anything that is not ModernBERT, naming what it got", () => {
    expect(() => encoderConfigFrom(baseConfig({ model_type: "bert" }))).toThrow(
      /Unsupported encoder: bert/,
    );
    expect(() => encoderConfigFrom(baseConfig({ model_type: undefined }))).toThrow(
      /Unsupported encoder: undefined/,
    );
  });

  it("rejects an activation the port has not implemented", () => {
    // `gelu` only. The alternative is `relu`/`silu` in the MLP, which would silently be the wrong
    // function rather than a load error.
    expect(() => encoderConfigFrom(baseConfig({ hidden_activation: "silu" }))).toThrow(
      /Unsupported encoder activation: silu/,
    );
  });

  it("rejects a head dimension that is not a whole number", () => {
    expect(() =>
      encoderConfigFrom(baseConfig({ hidden_size: 33, num_attention_heads: 2 })),
    ).toThrow(/even, integral attention head dimension/);
  });

  it("rejects an odd head dimension, which RoPE cannot split into pairs", () => {
    // 6 / 2 = 3: integral, and odd, so it passes the first clause and must be caught by the second.
    expect(() => encoderConfigFrom(baseConfig({ hidden_size: 6, num_attention_heads: 2 }))).toThrow(
      /even, integral attention head dimension/,
    );
  });

  it("accepts every even integral head dimension it should", () => {
    for (const [hidden, heads] of [
      [32, 2],
      [64, 4],
      [16, 1],
      [48, 6],
    ] as const) {
      expect(() =>
        encoderConfigFrom(baseConfig({ hidden_size: hidden, num_attention_heads: heads })),
      ).not.toThrow();
    }
  });

  it("rejects a scaled RoPE, which the port does not implement", () => {
    // `rope_type` other than "default" means a length-extrapolation scheme. Loading it as plain RoPE
    // would produce plausible answers with the wrong positional encoding.
    expect(() =>
      encoderConfigFrom(
        baseConfig({
          rope_parameters: { full_attention: { rope_type: "linear", rope_theta: 160000 } },
        }),
      ),
    ).toThrow(/Only default \(unscaled\) ModernBERT RoPE/);
  });

  it("checks every layer kind, not just the ones with parameters", () => {
    // A config naming a kind the port does not know would otherwise reach the attention selection.
    expect(() =>
      encoderConfigFrom(
        baseConfig({
          layer_types: ["full_attention", "exotic_attention"],
          rope_parameters: { exotic_attention: { rope_type: "yarn" } },
        }),
      ),
    ).toThrow(/Only default \(unscaled\) ModernBERT RoPE/);
  });

  it("accepts an explicit default rope_type, which is what the shipped configs use", () => {
    const cfg = encoderConfigFrom(
      baseConfig({
        rope_parameters: {
          full_attention: { rope_type: "default", rope_theta: 160000 },
          sliding_attention: { rope_type: "default", rope_theta: 10000 },
        },
      }),
    );
    expect(cfg.layer_types.length).toBe(2);
  });
});

describe("ropeBase", () => {
  const cfg = encoderConfigFrom(baseConfig());

  it("prefers the per-kind rope_parameters value", () => {
    const configured = encoderConfigFrom(
      baseConfig({
        rope_parameters: {
          full_attention: { rope_theta: 12345 },
          sliding_attention: { rope_theta: 678 },
        },
      }),
    );
    expect(ropeBase(configured, "full_attention")).toBe(12345);
    expect(ropeBase(configured, "sliding_attention")).toBe(678);
  });

  it("falls back per kind when rope_parameters is empty", () => {
    expect(ropeBase(cfg, "full_attention")).toBe(160000);
    expect(ropeBase(cfg, "sliding_attention")).toBe(10000);
  });

  it("falls back to the local theta for a kind it does not recognise", () => {
    // The fallback is `kind === "full_attention" ? global : local`, so anything unexpected gets the
    // local theta rather than `undefined` — a NaN RoPE base would be much harder to diagnose.
    expect(ropeBase(cfg, "exotic_attention")).toBe(10000);
  });
});

describe("sanitizeWeights", () => {
  /** The functions only rewrite names, so plain values stand in for MLX arrays. */
  const weights = (names: string[]): Weights =>
    Object.fromEntries(names.map((name) => [name, { fake: name }])) as unknown as Weights;

  it("maps the upstream in_proj names onto the MLX module layout", () => {
    const out = sanitizeWeights(weights(["head.layers.0.attn.in_proj_weight"]));
    expect(Object.keys(out)).toEqual(["head.layers.0.attn.in_proj.weight"]);
  });

  it("maps in_proj_bias the same way", () => {
    const out = sanitizeWeights(weights(["head.layers.0.attn.in_proj_bias"]));
    expect(Object.keys(out)).toEqual(["head.layers.0.attn.in_proj.bias"]);
  });

  it("gives the flat scorer and act_head parameters their `.layers` segment", () => {
    const out = sanitizeWeights(
      weights(["scorer.1.weight", "scorer.3.weight", "act_head.0.weight", "act_head.2.bias"]),
    );
    expect(Object.keys(out).sort()).toEqual([
      "act_head.layers.0.weight",
      "act_head.layers.2.bias",
      "scorer.layers.1.weight",
      "scorer.layers.3.weight",
    ]);
  });

  it("leaves an already-sequenced head name alone, so a re-run is idempotent", () => {
    // The check is `startsWith(prefix) && !startsWith(prefix + ".layers")`, so this must not become
    // `scorer.layers.layers.0.weight`.
    const once = sanitizeWeights(weights(["scorer.layers.0.weight"]));
    expect(Object.keys(once)).toEqual(["scorer.layers.0.weight"]);
    expect(Object.keys(sanitizeWeights(once))).toEqual(["scorer.layers.0.weight"]);
  });

  it("does not touch a prefix that merely starts with the same letters", () => {
    // `scorer_aux.weight` is a different parameter; a loose `includes`/prefix test would rename it.
    const out = sanitizeWeights(weights(["scorer_aux.weight", "encoder.layers.0.mlp.weight"]));
    expect(Object.keys(out).sort()).toEqual(["encoder.layers.0.mlp.weight", "scorer_aux.weight"]);
  });

  it("keeps every value, and keeps them in mapping order", () => {
    const input = weights(["a.weight", "b.weight"]);
    const out = sanitizeWeights(input);
    expect(Object.values(out)).toEqual(Object.values(input));
    expect(Object.keys(out)).toEqual(Object.keys(input));
  });

  it("throws when two parameters collide after renaming", () => {
    // The real hazard: `scorer.0.weight` and `scorer.layers.0.weight` both become the same name, and
    // without this check one of them would be silently dropped at load.
    expect(() => sanitizeWeights(weights(["scorer.0.weight", "scorer.layers.0.weight"]))).toThrow(
      /Duplicate checkpoint parameter: scorer\.layers\.0\.weight/,
    );
  });

  it("does not throw for two parameters that stay distinct", () => {
    expect(() => sanitizeWeights(weights(["scorer.0.weight", "scorer.0.bias"]))).not.toThrow();
  });

  it("handles an empty checkpoint without inventing anything", () => {
    expect(sanitizeWeights(weights([]))).toEqual({});
  });
});
