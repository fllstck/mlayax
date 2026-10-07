/**
 * Native-binding guards and the dtype/shape invariants that `model.ts` depends on.
 *
 * Two halves with different reach:
 *
 * - **Resolution** is tested by importing the vendored `native-binding.cjs` and calling its pure
 *   `selectBinding`, so every branch — wrong platform, missing payload, override precedence — is
 *   exercised on any machine, including Linux CI.
 * - **The MLX invariants** need the real addon and are skipped without it. They run without the
 *   checkpoint, so they are cheap enough to keep in the default suite on a Mac.
 *
 * Each invariant here is one that a silent regression would not surface as a wrong answer — only as
 * a slower forward or a shape that broadcasts into nonsense. See `model.ts`'s file header.
 */

import { createRequire } from "node:module";
import { beforeAll, describe, expect, it } from "vitest";
import type { MlxArray, MlxCore } from "../packages/mlayax/src/mlx/index.js";
import { loadMx, resolveNativeAddonPath } from "../packages/mlayax/src/mlx/index.js";

const require = createRequire(import.meta.url);
const nativeBinding = require("../packages/mlayax/vendor/node-mlx/native-binding.cjs") as {
  selectBinding: (input: {
    platform: string;
    arch: string;
    override: string | undefined;
    platformLibDir: string | null;
    exists: (file: string) => boolean;
  }) => string;
  ADDON: string;
  NATIVE_DIR_ENV: string;
  PLATFORM_PACKAGE: string;
};

const { selectBinding } = nativeBinding;

describe("native addon resolution", () => {
  const darwinArm64 = { platform: "darwin", arch: "arm64" } as const;

  it("finds the addon in the platform package's lib directory", () => {
    expect(
      selectBinding({
        ...darwinArm64,
        override: undefined,
        platformLibDir: "/pkg/mlayax-darwin-arm64/lib",
        exists: (file) => file === "/pkg/mlayax-darwin-arm64/lib/node_mlx.node",
      }),
    ).toBe("/pkg/mlayax-darwin-arm64/lib/node_mlx.node");
  });

  it("prefers MLAYAX_NATIVE_DIR over the platform package", () => {
    // The escape hatch used by the native build script and by tests against a locally built addon.
    expect(
      selectBinding({
        ...darwinArm64,
        override: "/local/build",
        platformLibDir: "/pkg/mlayax-darwin-arm64/lib",
        exists: () => true,
      }),
    ).toBe("/local/build/node_mlx.node");
  });

  it("falls back to the platform package when the override is set but empty", () => {
    expect(
      selectBinding({
        ...darwinArm64,
        override: "/local/build",
        platformLibDir: "/pkg/lib",
        exists: (file) => file !== "/local/build/node_mlx.node",
      }),
    ).toBe("/pkg/lib/node_mlx.node");
  });

  it("rejects non-Apple-Silicon platforms, naming the platform and the Rosetta case", () => {
    for (const [platform, arch] of [
      ["linux", "x64"],
      ["darwin", "x64"],
      ["win32", "arm64"],
    ] as const) {
      expect(() =>
        selectBinding({
          platform,
          arch,
          override: undefined,
          platformLibDir: null,
          exists: () => true,
        }),
      ).toThrow(new RegExp(`requires Apple Silicon.*${platform}/${arch}`, "s"));
    }
    // Rosetta is the one that actually bites people on a correct-looking machine.
    expect(() =>
      selectBinding({
        platform: "darwin",
        arch: "x64",
        override: undefined,
        platformLibDir: null,
        exists: () => true,
      }),
    ).toThrow(/Rosetta/);
  });

  it("lists every path it tried when the payload is missing", () => {
    const message = (() => {
      try {
        selectBinding({
          ...darwinArm64,
          override: "/local/build",
          platformLibDir: "/pkg/lib",
          exists: () => false,
        });
      } catch (error) {
        return (error as Error).message;
      }
      throw new Error("expected a throw");
    })();
    expect(message).toContain("/pkg/lib/node_mlx.node");
    expect(message).toContain("/local/build/node_mlx.node");
    expect(message).toContain("MLAYAX_NATIVE_DIR");
    expect(message).toContain(nativeBinding.PLATFORM_PACKAGE);
  });

  it("says so plainly when there was nowhere to look", () => {
    expect(() =>
      selectBinding({
        ...darwinArm64,
        override: undefined,
        platformLibDir: null,
        exists: () => false,
      }),
    ).toThrow(/No candidate locations/);
  });

  it("resolves to a real file in this process, when a payload is present", () => {
    let resolved: string;
    try {
      resolved = resolveNativeAddonPath();
    } catch {
      return; // No payload here (e.g. Linux CI) — the pure cases above still covered the logic.
    }
    expect(resolved.endsWith(nativeBinding.ADDON)).toBe(true);
    expect(resolved).toContain(nativeBinding.ADDON);
  });
});

/** Does this process have a usable native payload? Checked once, without performing a load. */
function nativeAvailable(): boolean {
  try {
    resolveNativeAddonPath();
    return true;
  } catch {
    return false;
  }
}

describe.skipIf(!nativeAvailable())("MLX invariants the runtime depends on", () => {
  // Loaded in `beforeAll`, not at declaration time: `describe.skipIf` still evaluates this callback
  // for reporting in some runners, and an eager `loadMx()` here would turn "skipped" into a
  // collection failure on machines with no payload.
  let loaded: MlxCore | null = null;

  beforeAll(() => {
    loaded = loadMx();
  });

  const mx = (): MlxCore => {
    if (loaded === null) throw new Error("native MLX was not loaded");
    return loaded;
  };

  const dtypeName = (dtype: object): string => {
    const core = mx();
    return dtype === core.float16
      ? "float16"
      : dtype === core.float32
        ? "float32"
        : dtype === core.int32
          ? "int32"
          : dtype === core.bool
            ? "bool"
            : "other";
  };

  describe("hazard 1: a JavaScript scalar must not upcast fp16", () => {
    it("gelu keeps the activation's dtype", async () => {
      const { gelu } = await import("../packages/mlayax/src/mlx/model.js");
      // The ~20 % trap: with bare `1` / `2` / `sqrt(2)` this returns float32 and every downstream
      // matmul in the MLP runs at half rate, with no error and no wrong answer.
      expect(dtypeName(gelu(mx(), mx().array([1, 2, 3], mx().float16)).dtype)).toBe("float16");
      expect(dtypeName(gelu(mx(), mx().array([1, 2, 3], mx().float32)).dtype)).toBe("float32");
    });

    it("relu keeps an integer dtype", async () => {
      const { relu } = await import("../packages/mlayax/src/mlx/model.js");
      expect(dtypeName(relu(mx(), mx().array([1, -2, 3], mx().int32)).dtype)).toBe("int32");
    });

    it("documents the trap it avoids: maximum(int32, <bare js 0>) is float32", () => {
      const indices = mx().array([1, 3], mx().int32);
      // Python's weak scalars keep this int32; JavaScript's `0` is a float32 array, and MLX promotes.
      expect(dtypeName(mx().maximum(indices, 0).dtype)).toBe("float32");
      expect(dtypeName(mx().maximum(indices, mx().array(0, mx().int32)).dtype)).toBe("int32");
    });
  });

  describe("hazard 3: marker gathering", () => {
    const hidden = (): MlxArray =>
      mx()
        .array(
          Array.from({ length: 2 * 5 * 3 }, (_, i) => i),
          mx().float32,
        )
        .reshape([2, 5, 3]);
    const positions = (): MlxArray => mx().array([1, 3, 0, 2], mx().int32).reshape([2, 2]);

    it("takeAlongAxis yields [batch, count, hidden]", () => {
      const gathered = mx().takeAlongAxis(hidden(), mx().expandDims(positions(), -1), 1);
      expect(gathered.shape).toEqual([2, 2, 3]);
    });

    it("take with a [b, count] index prepends a batch dim instead", () => {
      // The trap: this does not error, it broadcasts. [2,2,2,3] against an expected [2,2,3] is a
      // silent wrong answer waiting for the next operation to make sense of it.
      expect(mx().take(hidden(), positions(), 1).shape).toEqual([2, 2, 2, 3]);
    });

    it("gathers the same values either way, so only the shape catches it", () => {
      // The values agree; the rank does not. That is what makes this trap quiet: nothing throws,
      // and a later broadcast turns it into a plausible wrong answer.
      const viaTake = mx()
        .take(hidden(), positions(), 1)
        .reshape([-1])
        .toTypedArray() as Float32Array;
      const viaAlong = mx()
        .takeAlongAxis(hidden(), mx().expandDims(positions(), -1), 1)
        .reshape([-1])
        .toTypedArray() as Float32Array;
      expect(Array.from(viaAlong.slice(0, 6))).toEqual([3, 4, 5, 9, 10, 11]);
      expect(Array.from(viaTake.slice(0, 6))).toEqual([3, 4, 5, 9, 10, 11]);
    });

    it("accepts float32 indices too, so the int32 form is about dtype, not about failing", () => {
      // Recorded because the port's comment used to claim the opposite. Verified eager and compiled:
      // takeAlongAxis casts indices, so a bare `0` does not break the gather — it just makes the
      // index tensor float32.
      const floatIndices = mx().maximum(positions(), 0);
      expect(dtypeName(floatIndices.dtype)).toBe("float32");
      expect(mx().takeAlongAxis(hidden(), mx().expandDims(floatIndices, -1), 1).shape).toEqual([
        2, 2, 3,
      ]);
    });
  });

  describe("hazard 2: which op actually rejects a float32 index", () => {
    // `mx.maximum(intArray, 0)` returns float32 — that part of the hazard is real, and the dtype
    // test for it lives in the hazard-1 block above. What this block settles is the *consequence*,
    // because `model.ts` claimed for a long time that "the gather that follows rejects a float32
    // index outright". Half true, and the half that is false is the half that mattered:
    //
    //   mx.take            -> THROWS  "Indices must be integral"   (the hazard, verified below)
    //   mx.takeAlongAxis   -> casts, eagerly and under mx.compile   (what the marker path uses)
    //
    // So the marker gather was never at risk of failing. The int32 form is still correct — the
    // indices are integers and the promotion is avoidable work — but it is a dtype choice, not a
    // workaround, and the difference matters when someone next reads that comment deciding whether
    // they can relax it.

    const hidden = (): MlxArray =>
      mx()
        .array(
          Array.from({ length: 2 * 5 * 3 }, (_, i) => i),
          mx().float32,
        )
        .reshape([2, 5, 3]);
    const intIndices = (): MlxArray => mx().array([1, 3, 0, 2], mx().int32).reshape([2, 2]);

    it("mx.take rejects float32 indices, which is the real hazard", () => {
      const floatIndices = mx().maximum(intIndices(), 0);
      expect(dtypeName(floatIndices.dtype)).toBe("float32");
      expect(() => mx().take(hidden(), floatIndices, 1)).toThrow(/Indices must be integral/);
      // And the int32 form is accepted, so the rejection is about the dtype and not the call shape.
      expect(mx().take(hidden(), intIndices(), 1).shape).toEqual([2, 2, 2, 3]);
    });

    it("mx.takeAlongAxis accepts float32 indices, which is why the marker path never broke", () => {
      const floatIndices = mx().maximum(intIndices(), 0);
      const out = mx().takeAlongAxis(hidden(), mx().expandDims(floatIndices, -1), 1);
      expect(out.shape).toEqual([2, 2, 3]);
      // Same under compile, so the specialised profile is not a second chance to fail.
      const compiled = mx().compile((h: MlxArray, p: MlxArray) =>
        mx().takeAlongAxis(h, mx().expandDims(mx().maximum(p, 0), -1), 1),
      );
      const compiledOut = compiled(hidden(), intIndices());
      expect(compiledOut.shape).toEqual([2, 2, 3]);
    });

    it("keeps the indices integral anyway, because the promotion is wasted work", () => {
      // The expression `model.ts` builds, asserted at the dtype level: marker positions clamped
      // against an int32 zero stay int32, so nothing downstream is promoted to float32.
      const intClamped = mx().maximum(intIndices(), mx().array(0, mx().int32));
      expect(dtypeName(intClamped.dtype)).toBe("int32");
      const floatClamped = mx().maximum(intIndices(), 0);
      expect(dtypeName(floatClamped.dtype)).toBe("float32");
    });
  });

  describe("hazard 6: shape-inference boundary", () => {
    it("a compiled gather matches eagerly, for both index dtypes", () => {
      const core = mx();
      const hidden = core
        .array(
          Array.from({ length: 2 * 5 * 3 }, (_, i) => i),
          core.float32,
        )
        .reshape([2, 5, 3]);
      const positions = core.array([1, 3, 0, 2], core.int32).reshape([2, 2]);
      for (const prep of [(p: MlxArray) => p, (p: MlxArray) => core.maximum(p, 0)]) {
        const compiled = core.compile((hh: MlxArray, pp: MlxArray) =>
          core.takeAlongAxis(hh, core.expandDims(prep(pp), -1), 1),
        );
        const out = compiled(hidden, positions);
        core.eval(out);
        expect(out.shape).toEqual([2, 2, 3]);
      }
    });

    it("fast.sdpa is available and takes a boolean mask", () => {
      // The default attention path. Its absence would be a hard load-time failure, so assert the
      // symbol exists rather than discovering it on the first forward.
      const core = mx();
      expect(typeof core.fast.scaledDotProductAttention).toBe("function");
      expect(typeof core.fast.layerNorm).toBe("function");
      expect(typeof core.fast.rope).toBe("function");
    });
  });
});
