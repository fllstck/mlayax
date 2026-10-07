#!/usr/bin/env python
"""Generate the tiny synthetic Laya checkpoint that the default test suite runs against.

TASKS.md §6 asks for parity that runs *by default in CI, with no download*. The real checkpoint is
803 MiB and needs the network, so the default suite had nothing that exercised a forward pass at all:
`test/fixtures/tiny/` held a tokenizer and nothing to run it on. This produces the missing half.

Usage:

    PYTHONPATH=... python tools/reference/make_tiny_checkpoint.py <out-dir> [--seed 0]

For example, against the spike's virtualenv:

    ~/Development/styleguide-checker/laya-mlx-playground/.venv/bin/python \\
        tools/reference/make_tiny_checkpoint.py test/fixtures/tiny

Output, committed to the repository:

    <out-dir>/encoder/config.json     a ModernBERT config shrunk to `hidden_size` 32, 2 layers
    <out-dir>/rl_agent_config.json    head_layers 2, max_len 64, head_max_len 32, clean temperatures
    <out-dir>/model.safetensors       ~200 KB of seeded random weights
    <out-dir>/tokenizer/              *not written here* — the 92-word fixture tokenizer already exists

Then regenerate the expectations, which is what makes this a parity fixture rather than just a
checkpoint:

    <venv>/bin/python tools/reference/laya_ref.py test/fixtures/tiny float32 test/fixtures/ref/tiny-fp32.json
    <venv>/bin/python tools/reference/laya_ref.py test/fixtures/tiny float16 test/fixtures/ref/tiny-fp16.json

Note the dtype spellings: the *files* are named `fp16`/`fp32` after the existing pair, but
`laya_ref.py` passes the argument straight to `laya_mlx`, which wants `float32`/`float16`.

Both output directories are excluded from Biome (`biome.json`'s `files.includes`), because everything
in them is generated. `laya_ref.py` writes with `indent=1` and Biome wants `indent=2`, so including
them means every regeneration produces a spurious formatting failure; excluding them keeps
`npm run verify` green after a refresh.

Three deliberate properties, each of which is a decision rather than an implementation detail:

1. **The configs are literals here, not read from the real checkpoint.** They are the real
   checkpoint's keys with the dimensions shrunk (1024→32 hidden, 28→2 layers, 16→2 heads,
   2624→64 intermediate, 50368→128 vocab). Reading the real one would make this script unrunnable
   without a 3 GiB download, and the whole point is that anyone can regenerate the fixture.
2. **`max_len` 64 / `head_max_len` 32**, matching `make_prepare_fixture.py`. The real values are
   512/192, which measure the same thing with more padding and less truncation.
3. **Temperatures are all 1.5**, inside the `[0.5, 5]` clamp window. That matters: the real
   checkpoint ships `choice:11+` at 0.1006, which fires a `RuntimeWarning` on every load, and a
   fixture whose every run emits a warning trains people to ignore warnings. It is still not the
   identity, so the calibration path — bucket lookup, clamping, division — is genuinely exercised.

Weights are random, seeded from `--seed`. They are not *meaningful*, and they are not supposed to be:
this fixture tests that two implementations of the same architecture agree bit-for-bit, which random
weights test as well as trained ones and at 200 KB instead of 803 MiB. What it cannot test is whether
the weights are interpreted the way the *upstream authors* intended — that is what the real-checkpoint
parity in `test/parity.real.test.ts` is for, and why both exist.
"""

import argparse
import json
import pathlib
import sys

import mlx.core as mx
from mlx.utils import tree_flatten

from laya_mlx.model import DecisionModel, EncoderConfig

# The real checkpoint's `encoder/config.json`, shrunk. Keys that do not affect the weight tree
# (architectures, transformers_version, classifier_*, sparse_*) are kept verbatim so the fixture
# exercises the same config parsing, and the ids are set below the tokenizer's ceiling.
TINY_ENCODER_CONFIG = {
    "architectures": ["ModernBertForMaskedLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 2,
    "classifier_activation": "gelu",
    "classifier_bias": False,
    "classifier_dropout": 0.0,
    "classifier_pooling": "mean",
    "cls_token_id": 2,
    "decoder_bias": True,
    "deterministic_flash_attn": False,
    "dtype": "float32",
    "embedding_dropout": 0.0,
    "eos_token_id": 3,
    "global_attn_every_n_layers": 3,
    "gradient_checkpointing": False,
    "hidden_activation": "gelu",
    "hidden_size": 32,
    "initializer_cutoff_factor": 2.0,
    "initializer_range": 0.02,
    "intermediate_size": 64,
    "layer_norm_eps": 1e-05,
    # 2 layers, and `global_attn_every_n_layers` 3 puts a full-attention layer first — so the two
    # layers differ, which is the point of keeping the sliding/full distinction in a 2-layer model.
    "layer_types": ["full_attention", "sliding_attention"],
    "local_attention": 128,
    "max_position_embeddings": 8192,
    "mlp_bias": False,
    "mlp_dropout": 0.0,
    "model_type": "modernbert",
    "norm_bias": False,
    "norm_eps": 1e-05,
    # 2 heads of dimension 16: `hidden_size % num_attention_heads == 0` and an even head dim are both
    # enforced by `EncoderConfig.from_dict`, so this pair is the smallest valid choice.
    "num_attention_heads": 2,
    "num_hidden_layers": 2,
    "pad_token_id": 0,
    "position_embedding_type": "absolute",
    "repad_logits_with_grad": False,
    "rope_parameters": {
        "full_attention": {"rope_theta": 160000.0, "rope_type": "default"},
        "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
    },
    "sep_token_id": 3,
    "sparse_pred_ignore_index": -100,
    "sparse_prediction": False,
    "tie_word_embeddings": True,
    "transformers_version": "5.0.0",
    # The fixture tokenizer's ids run 0..91; 128 leaves headroom and keeps the embedding a power of two.
    "vocab_size": 128,
}

# The real checkpoint's `rl_agent_config.json`, shrunk. `head_layers` 2 matches the real shape rather
# than being cut to 1: a single-layer head would not exercise the loop, and the point of a parity
# fixture is to run the same architecture.
TINY_AGENT_CONFIG = {
    "encoder": "answerdotai/ModernBERT-large",
    "head_layers": 2,
    "max_len": 64,
    "head_max_len": 32,
    "max_prefixes": 6,
    "act_costs": {"escalate": 0.5},
    "cost_wrong_act": 3.0,
    "amp_dtype": "bf16",
    "model_name": "rl-agent",
    "temperature": [1.5, 1.5, 1.5],
    "temperature_by_options": {
        "choice:3-5": 1.5,
        "choice:6-10": 1.5,
        "score:3-5": 1.5,
        "noul:2": 1.5,
        "choice:11+": 1.5,
        "choice:2": 1.5,
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dir", type=pathlib.Path, help="the fixture directory, e.g. test/fixtures/tiny")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for the random weights")
    args = parser.parse_args()

    out_dir: pathlib.Path = args.out_dir
    if not (out_dir / "tokenizer" / "tokenizer.json").is_file():
        print(
            f"error: no tokenizer at {out_dir}/tokenizer. This script does not create one — the "
            f"92-word fixture tokenizer is committed separately (see make_prepare_fixture.py).",
            file=sys.stderr,
        )
        return 1

    # Validate before writing anything: `from_dict` enforces the divisibility and layer_types rules,
    # and it is much better to fail here than to produce a checkpoint the runtime cannot load.
    encoder_cfg = EncoderConfig.from_dict(TINY_ENCODER_CONFIG)

    mx.random.seed(args.seed)
    model = DecisionModel(encoder_cfg, TINY_AGENT_CONFIG)
    mx.eval(model.parameters())
    weights = dict(tree_flatten(model.parameters()))

    total = sum(v.size for v in weights.values())
    print(f"weights: {len(weights)} tensors, {total} parameters")

    (out_dir / "encoder").mkdir(parents=True, exist_ok=True)
    (out_dir / "encoder" / "config.json").write_text(
        json.dumps(TINY_ENCODER_CONFIG, indent=2) + "\n"
    )
    (out_dir / "rl_agent_config.json").write_text(json.dumps(TINY_AGENT_CONFIG, indent=2) + "\n")
    mx.save_safetensors(str(out_dir / "model.safetensors"), weights)

    size = (out_dir / "model.safetensors").stat().st_size
    print(f"wrote {out_dir}/model.safetensors ({size} bytes)")
    print("now regenerate the expectations, or the fixture will not match:")
    print(f"  tools/reference/laya_ref.py {out_dir} fp32 test/fixtures/ref/tiny-fp32.json")
    print(f"  tools/reference/laya_ref.py {out_dir} fp16 test/fixtures/ref/tiny-fp16.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())