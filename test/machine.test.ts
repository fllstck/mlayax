/**
 * The numeric policy that lets the parity suite run on a GPU that is not the reference one.
 *
 * TASKS.md §10.14: the tiny fixture's `action.act_probability` was measured at 3e-4 on GitHub's macOS
 * runner while every field the product reports stayed exact on the same run. That is a property of the
 * kernel lowering, not of this code, so the tests assert bit-exactness on the reference machine and a
 * measured allowance elsewhere.
 *
 * These tests exist because that allowance is the difference between a gate and a rubber stamp: the
 * interesting failure is not "the action head moved", it is "something that is not the action head
 * moved", and that has to stay a failure on every machine. Both branches are exercised here without
 * swapping hardware.
 */

import { describe, expect, it } from "vitest";
import {
  ACTION_PROBABILITY_DRIFT,
  cpuBrand,
  driftForField,
  driftPolicy,
  isReferenceMachine,
  machineNote,
  REFERENCE_CPU,
} from "./helpers/machine.js";
import { compare } from "./helpers/scorecard.js";

describe("the reference-machine policy", () => {
  it("recognises the machine the numbers were measured on, and only that one", () => {
    expect(isReferenceMachine(REFERENCE_CPU)).toBe(true);
    expect(isReferenceMachine(`${REFERENCE_CPU} Max`)).toBe(false);
    expect(isReferenceMachine("Apple M2 Pro")).toBe(false);
    expect(isReferenceMachine("Apple M1")).toBe(false);
    expect(isReferenceMachine(null)).toBe(false);
  });

  it("grants no drift at all on the reference machine", () => {
    expect(driftForField("department.action.act_probability", REFERENCE_CPU)).toBe(0);
    expect(driftForField("department.probabilities.billing", REFERENCE_CPU)).toBe(0);
    expect(driftForField("urgency.score", REFERENCE_CPU)).toBe(0);
  });

  it("grants the measured drift to the action head, and to nothing else, elsewhere", () => {
    const elsewhere = "Apple M2 Pro";
    expect(driftForField("department.action.act_probability", elsewhere)).toBe(
      ACTION_PROBABILITY_DRIFT,
    );
    // A nested path ends with the field name; the `endsWith` arm covers ids that contain dots.
    expect(driftForField("action.act_probability", elsewhere)).toBe(ACTION_PROBABILITY_DRIFT);
    // Everything else is still expected to be bit-exact on any GPU — this is the assertion that makes
    // the allowance narrow rather than a blanket tolerance.
    for (const strictPath of [
      "department.probabilities.billing",
      "department.confidence",
      "department.answer_confidence",
      "urgency.score",
      "urgency.legend.0",
      "refund.noul",
      "action.act_probabilities", // a near-miss name must not match
    ]) {
      expect(driftForField(strictPath, elsewhere), strictPath).toBe(0);
    }
  });

  it("is the policy `compare` is handed, not a per-test decision", () => {
    const policy = driftPolicy("Apple M2 Pro");
    expect(policy("urgency.action.act_probability")).toBe(ACTION_PROBABILITY_DRIFT);
    expect(policy("urgency.score")).toBe(0);
    expect(driftPolicy(REFERENCE_CPU)("urgency.action.act_probability")).toBe(0);
  });

  it("names the machine in a failure message, so the log says which GPU it was", () => {
    expect(machineNote(REFERENCE_CPU)).toContain("reference machine");
    expect(machineNote("Apple M2 Pro")).toContain("not the reference machine");
  });

  it("accepts exactly the drift CI measured, in the field it was measured in", () => {
    // The real number from the runner, not a rounded stand-in: `0.00030000000000002247` is *above* 3e-4
    // in binary, and a bound of 3e-4 would have failed there a second time. This test is the reason the
    // bound carries one step of headroom.
    const MEASURED = 0.00030000000000002247;
    const want = { department: { action: { act_probability: 0.495 }, confidence: 0.8175 } };
    const got = {
      department: {
        action: { act_probability: 0.495 - MEASURED },
        confidence: 0.8175,
      },
    };

    const elsewhere = compare(want, got, 0, { driftForPath: driftPolicy("Apple M2 Pro") });
    expect(elsewhere.mismatches).toEqual([]);
    expect(elsewhere.maxDelta).toBeLessThanOrEqual(ACTION_PROBABILITY_DRIFT);

    // The same payload on the reference machine is a failure, because there the claim is Δ 0.
    const reference = compare(want, got, 0, { driftForPath: driftPolicy(REFERENCE_CPU) });
    expect(reference.mismatches).toHaveLength(1);
    expect(reference.mismatches[0]).toContain("department.action.act_probability");

    // And the same 3e-4 in any other field is a failure on any machine — the allowance is a property of
    // one field, not a blanket tolerance for the fixture.
    const wrongField = compare(
      { department: { confidence: 0.8175 } },
      { department: { confidence: 0.8175 - MEASURED } },
      0,
      { driftForPath: driftPolicy("Apple M2 Pro") },
    );
    expect(wrongField.mismatches).toHaveLength(1);
    expect(wrongField.mismatches[0]).toContain("department.confidence");

    // Above the bound it fails everywhere, including off-reference: the allowance is bounded, not open.
    const tooFar = compare(
      { department: { action: { act_probability: 0.495 } } },
      { department: { action: { act_probability: 0.495 - 5e-4 } } },
      0,
      { driftForPath: driftPolicy("Apple M2 Pro") },
    );
    expect(tooFar.mismatches).toHaveLength(1);
  });

  it("reads the CPU on this machine (an Apple Silicon Mac, which is the only supported one)", () => {
    // Not asserting a specific chip: this runs on contributor machines too, and the point is only that
    // the probe works — `null` is the "not macOS" answer, and then nothing is the reference machine.
    const cpu = cpuBrand();
    expect(cpu === null || typeof cpu === "string").toBe(true);
    expect(isReferenceMachine()).toBe(cpu === REFERENCE_CPU);
  });
});
