"""Python MLX benchmark on the same inputs as src/mlx/bench.ts.

    uv run python tools/laya_bench.py <model-dir> [iterations]
"""

import json
import os
import resource
import sys
import time

import mlx.core as mx
import laya_mlx as laya

model_dir = sys.argv[1]
iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 10

STATE = "I was billed twice. Please refund the duplicate today."
ONE = {
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this request?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs and outages",
            "sales": "new purchases",
        },
    }
}
THREE = {
    **ONE,
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical"],
    },
    "refund": {"type": "noul", "instructions": "Does the customer ask for money back?"},
}
MANY = {f"q{i}": {"type": "noul", "instructions": f"Is reason number {i} the cause?"} for i in range(16)}

t0 = time.perf_counter()
agent = laya.load(model_dir, dtype="float16", compile=os.environ.get("LAYA_COMPILE") == "1")
load_ms = (time.perf_counter() - t0) * 1000

first = agent.predict(STATE, THREE)


def measure(questions, n=iterations):
    samples = []
    for _ in range(n):
        t = time.perf_counter()
        agent.predict(STATE, questions)
        samples.append((time.perf_counter() - t) * 1000)
    samples.sort()
    return samples


single = measure(ONE)
triple = measure(THREE)
t = time.perf_counter()
agent.predict(STATE, MANY)
batch_ms = (time.perf_counter() - t) * 1000

rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
print(f"python {sys.version.split()[0]}  MLX fp16  load={load_ms:.0f}ms (cached checkpoint)")
print(f"  one short question: p50 {single[len(single) // 2]:.1f} ms (min {single[0]:.1f})")
print(f"  3 questions:        p50 {triple[len(triple) // 2]:.1f} ms (min {triple[0]:.1f})")
print(f"  16-question batch:  {batch_ms:.1f} ms ({16000 / batch_ms:.0f} q/s)")
print(f"  peak MLX memory:    {mx.get_peak_memory() / 1024 / 1024:.0f} MiB   rss: {rss:.0f} MiB")
print(
    "  answers: "
    + first["answers"]["department"]["choice"]
    + " "
    + json.dumps(first["answers"]["department"]["probabilities"])
)