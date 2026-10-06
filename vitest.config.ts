import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    // Phase 1 ships the toolchain before any test files exist.
    passWithNoTests: true,
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
