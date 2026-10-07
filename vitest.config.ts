import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    // Phase 1 ships the toolchain before any test files exist.
    passWithNoTests: true,
    /**
     * 60 s, not vitest's 5 s default, and deliberately loose.
     *
     * A native test's first forward at a new sequence length JIT-compiles a Metal graph. On the
     * reference machine that is milliseconds; on GitHub's macOS runner it exceeded 5 s *twice*, and the
     * same four tests presented once as hard failures (the tiny fixture's boundary-sensitive action
     * head, §10.14) and once as timeouts — runner hardware varies. This is a correctness suite, so its
     * timeouts should not encode a performance claim: speed is measured by `bench/gate.ts` on the
     * reference machine, against the committed baseline.
     *
     * `bun test` does not read this file, which is why the CI bun job passes `--timeout` explicitly.
     */
    testTimeout: 60_000,
    include: ["packages/*/src/**/*.test.ts", "test/**/*.test.ts"],
    exclude: ["**/node_modules/**", "**/dist/**", "**/vendor/**"],
    coverage: {
      provider: "v8",
      reporter: ["text", "lcov"],
      include: ["packages/mlayax/src/core/**", "packages/mlayax/src/mlx/**"],
      exclude: ["**/*.test.ts", "**/*.d.ts", "packages/mlayax/src/**/index.ts"],
      // §6 quality gates (>= 85% core/mlx), with the tighter core bar Phase 1 asks for (>= 90%).
      thresholds: {
        "packages/mlayax/src/core/**": {
          lines: 90,
          branches: 90,
          functions: 90,
          statements: 90,
        },
        "packages/mlayax/src/mlx/**": {
          lines: 85,
          branches: 85,
          functions: 85,
          statements: 85,
        },
      },
    },
  },
});
