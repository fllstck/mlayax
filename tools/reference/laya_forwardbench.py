"""Pure forward-pass cost in Python MLX, with identical tensors.

    uv run python tools/laya_forwardbench.py <model-dir> [rows] [iterations]
"""

import json
import sys
import time

import mlx.core as mx
import laya_mlx as laya
from laya_mlx.agent import collate_items

model_dir = sys.argv[1]
rows = int(sys.argv[2]) if len(sys.argv) > 2 else 1
iterations = int(sys.argv[3]) if len(sys.argv) > 3 else 20

agent = laya.load(model_dir, dtype="float16")
questions = {
    f"q{i}": {
        "type": "choice",
        "instructions": f"Which team should handle reason number {i}?",
        "criteria": {"billing": "invoices", "technical": "bugs", "sales": "purchases"},
    }
    for i in range(rows)
}
state = "I was billed twice. Please refund the duplicate today."
items, _internal = agent.prepare(state, questions)
batch = collate_items(items, agent.tok.pad_token_id, max_length=agent.cfg.get("max_len", 512))

b, L = batch["input_ids"].shape
for _ in range(3):
    agent.forward(batch)
samples = []
for _ in range(iterations):
    t = time.perf_counter()
    agent.forward(batch)
    samples.append((time.perf_counter() - t) * 1000)
samples.sort()

print(
    json.dumps(
        {
            "runtime": "python",
            "rows": int(b),
            "seqLen": int(L),
            "markers": int(batch["marker_pos"].shape[1]),
            "p50": round(samples[len(samples) // 2], 2),
            "min": round(samples[0], 2),
        }
    )
)