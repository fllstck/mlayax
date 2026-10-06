"""Attention/MLP internals in Python, matching src/mlx/profile2.ts."""

import json
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import laya_mlx as laya
from laya_mlx.agent import collate_items
from laya_mlx.model import attention_masks

model_dir = sys.argv[1]
agent = laya.load(model_dir, dtype="float16")
model = agent.model
items, _ = agent.prepare(
    "I was billed twice. Please refund the duplicate today.",
    {
        "q0": {
            "type": "choice",
            "instructions": "Which team should handle reason number 0?",
            "criteria": {"billing": "invoices", "technical": "bugs", "sales": "purchases"},
        }
    },
)
batch = collate_items(items, agent.tok.pad_token_id, max_length=agent.cfg.get("max_len", 512))
input_ids = mx.array(batch["input_ids"])
attention_mask = mx.array(batch["attention_mask"])
b, L = input_ids.shape
cfg = model.encoder.config
heads = cfg.num_attention_heads
head_dim = cfg.hidden_size // heads
layer = model.encoder.layers[0]

x = model.encoder(input_ids, attention_mask)
mx.eval(x)
mask = attention_masks(attention_mask, cfg.local_attention)["full_attention"]


def timed(name, fn, n=20):
    for _ in range(3):
        mx.eval(fn())
    s = []
    for _ in range(n):
        t = time.perf_counter()
        mx.eval(fn())
        s.append((time.perf_counter() - t) * 1000)
    s.sort()
    return {"name": name, "ms": round(s[len(s) // 2], 3)}


out = [
    timed("layerNorm", lambda: mx.fast.layer_norm(x, layer.mlp_norm.weight, None, 1e-5)),
    timed("linear (Wqkv)", lambda: mx.matmul(x, layer.attn.Wqkv.weight.T)),
]


def reshape_split_transpose():
    qkv = mx.matmul(x, layer.attn.Wqkv.weight.T).reshape(b, L, 3, heads, head_dim)
    return [qkv[:, :, i].transpose(0, 2, 1, 3) for i in range(3)]


def reshape_index_transpose():
    qkv = mx.matmul(x, layer.attn.Wqkv.weight.T).reshape(b, L, 3, heads, head_dim)
    return [mx.split(qkv, 3, axis=2)[i].reshape(b, L, heads, head_dim).transpose(0, 2, 1, 3) for i in range(3)]


out += [
    timed("reshape+slice+transpose", reshape_split_transpose),
    timed("reshape+split+transpose", reshape_index_transpose),
    timed("mlp", lambda: layer.mlp(layer.mlp_norm(x))),
]


def mlp_parts():
    value, gate = mx.split(mx.matmul(layer.mlp_norm(x), layer.mlp.Wi.weight.T), 2, axis=-1)
    return nn.gelu(value) * gate


out += [
    timed("mlp split+gelu+mul", mlp_parts),
    timed("full layer", lambda: layer(x, mask)),
    timed("encoder", lambda: model.encoder(input_ids, attention_mask)),
]
print(json.dumps({"runtime": "python", "out": out}))