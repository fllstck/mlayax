"""Section-by-section timing of the Python model, matching src/mlx/profile.ts."""

import json
import sys
import time

import mlx.core as mx
import laya_mlx as laya
from laya_mlx.agent import collate_items
from laya_mlx.model import attention_masks

model_dir = sys.argv[1]
rows = int(sys.argv[2]) if len(sys.argv) > 2 else 1
agent = laya.load(model_dir, dtype="float16")
model = agent.model

questions = {
    f"q{i}": {
        "type": "choice",
        "instructions": f"Which team should handle reason number {i}?",
        "criteria": {"billing": "invoices", "technical": "bugs", "sales": "purchases"},
    }
    for i in range(rows)
}
items, internal = agent.prepare("I was billed twice. Please refund the duplicate today.", questions)
batch = collate_items(items, agent.tok.pad_token_id, max_length=agent.cfg.get("max_len", 512))
tensors = {k: mx.array(v) for k, v in batch.items() if k in
           ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")}
input_ids = tensors["input_ids"]
attention_mask = tensors["attention_mask"]
marker_pos = tensors["marker_pos"]
marker_mask = tensors["marker_mask"]
qtype = tensors["qtype"]
b, L = input_ids.shape


def timed(name, fn):
    for _ in range(3):
        mx.eval(fn())
    samples = []
    for _ in range(15):
        t = time.perf_counter()
        mx.eval(fn())
        samples.append((time.perf_counter() - t) * 1000)
    samples.sort()
    return {"name": name, "ms": round(samples[len(samples) // 2], 2)}


masks = attention_masks(attention_mask, model.encoder.config.local_attention)
out = [
    timed("masks", lambda: attention_masks(attention_mask, model.encoder.config.local_attention)),
    timed("encoder", lambda: model.encoder(input_ids, attention_mask)),
]
h = model.encoder(input_ids, attention_mask)
mx.eval(h)


def type_head():
    t = h + model.type_emb(qtype)[:, None, :]
    return model.head(t, attention_mask[:, None, None, :].astype(mx.bool_))


out.append(timed("type_emb + head", type_head))
out.append(timed("1 layer (full attn)", lambda: model.encoder.layers[0](h, masks["full_attention"])))
out.append(timed("1 layer (sliding)", lambda: model.encoder.layers[1](h, masks["sliding_attention"])))
out.append(timed("full forward", lambda: model(input_ids, attention_mask, marker_pos, marker_mask, qtype)))
print(json.dumps({"runtime": "python", "rows": int(b), "L": int(L), "out": out}))