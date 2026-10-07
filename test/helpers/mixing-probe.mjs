/**
 * Child-process probe for the mixing guard (TASKS.md §8.6).
 *
 * This has to be a *separate process*. Making a foreign `libmlx` resident is irreversible: dyld has
 * already resolved the install name by the time anything can be inspected, so a vitest worker that
 * reproduced the collision could not go on to test anything else — including the healthy path.
 *
 * Usage:
 *   node test/helpers/mixing-probe.mjs <foreignLibDir> <distEntry>
 *
 * `<foreignLibDir>` holds the `libmlx.dylib` to make resident first. It is loaded through
 * `process.dlopen`, which is the only dlopen Node exposes. Loading a plain dylib (rather than an
 * addon) always throws — "Module did not self-register" — *after* mapping the image, which is
 * exactly the state we need. The throw is swallowed on purpose.
 *
 * Prints one JSON object on stdout, so the test can assert on the structure rather than on prose.
 */

const [foreignLibDir, distEntry] = process.argv.slice(2);

const result = {
  /** The `libmlx.dylib` images this process has resident, at the moment of the load attempt. */
  residentBeforeLoad: [],
  /** What `process.report` was willing to tell us — empty on Bun, which is the whole reason
   * `describeMlxLoadFailure` exists. */
  residentCount: 0,
  /** "loaded" | "threw" */
  outcome: "loaded",
  name: null,
  message: null,
  /** Any `RuntimeWarning`s raised, e.g. the "another MLX is resident but cannot shadow ours" note. */
  warnings: [],
};

process.on("warning", (warning) => {
  result.warnings.push(`${warning.name}: ${warning.message}`);
});

try {
  process.dlopen({ exports: {} }, `${foreignLibDir}/libmlx.dylib`);
} catch {
  // Expected: "Module did not self-register". The image is mapped and resident anyway.
}

// `process.report` is Node-only: Bun has no image list at all, which is exactly the case
// `describeMlxLoadFailure` exists for. The probe has to survive that rather than throw here, and
// reporting zero images is the honest answer when the runtime will not enumerate them.
const sharedObjects = process.report?.getReport?.()?.sharedObjects ?? [];
result.residentCount = sharedObjects.length;
result.residentBeforeLoad = sharedObjects.filter((s) => s.endsWith("libmlx.dylib"));

try {
  const { loadMx } = await import(distEntry);
  const mx = loadMx();
  // Reaching here means the guard let the load through. Prove MLX is actually usable, or the probe
  // would pass for a process whose MLX is broken in some other way.
  const sum = mx.array([1, 2, 3], mx.float32).sum();
  mx.eval(sum);
  result.outcome = "loaded";
  result.message = `sum=${sum.item()}`;
} catch (error) {
  result.outcome = "threw";
  result.name = error?.name ?? null;
  result.message = error?.message ?? String(error);
}

// The warning is emitted on `nextTick`, so let it land before reporting.
await new Promise((resolve) => setTimeout(resolve, 50));
process.stdout.write(`${JSON.stringify(result)}\n`);
