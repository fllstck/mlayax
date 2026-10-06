#!/usr/bin/env python
"""Generate the golden `prepare()` fixture from the real Python reference.

This is the Phase 2 gate: our TypeScript prompt construction must produce byte-identical token
sequences to `laya_mlx` itself, not merely to a previous TypeScript port. It deliberately uses the
**tiny** fixture tokenizer (92-word WordLevel vocab, `max_len` 64, `head_max_len` 32) rather than
the English checkpoint, so the checked-in expectation is a few KB, needs no download, and is tight
enough that the truncation and option-collapsing paths actually fire.

Usage:

    PYTHONPATH=... python tools/reference/make_prepare_fixture.py <tokenizer-dir> <out.json>

For example, against the spike's virtualenv:

    ~/Development/styleguide-checker/laya-mlx-playground/.venv/bin/python \\
        tools/reference/make_prepare_fixture.py \\
        test/fixtures/tiny/tokenizer test/fixtures/ref/prepare.json

The output is committed. Regenerate it only when the reference or the fixture cases change, and say
so in the commit message.
"""

import json
import sys

from laya_mlx.agent import Agent
from laya_mlx.common import build_sequence, render_options
from laya_mlx.tokenizer import Tokenizer

# The tiny checkpoint's own budget. Small on purpose: it makes truncation reachable.
MAX_LEN = 64
HEAD_MAX_LEN = 32

# The three cases the parity references use (`tools/reference/laya_ref.py`), plus cases that exist
# to pin down the paths the three do not reach: state truncation, left-truncation of a turn list,
# and option collapsing under a tight head budget.
CASES = [
    {
        "label": "three questions",
        "state": "I was billed twice. Please refund the duplicate today.",
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which team should handle this request?",
                "criteria": {
                    "billing": "invoices, payments, refunds",
                    "technical": "bugs and outages",
                    "sales": "new purchases",
                },
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this request?",
                "criteria": ["not urgent", "soon", "critical"],
            },
            "refund": {
                "type": "noul",
                "instructions": "Does the customer ask for money back?",
            },
        },
    },
    {
        "label": "12 options (clamped 11+ bucket)",
        "state": "The widget arrived scratched and the manual is missing.",
        "questions": {
            "category": {
                "type": "choice",
                "instructions": "Which category fits best?",
                "criteria": {f"label_{i}": f"description number {i}" for i in range(12)},
            },
        },
    },
    {
        "label": "structured state and criteria",
        "state": {
            "from": "hello@world",
            "subject": "charge on invoice",
            "body": "We were billed twice in March. Please refund the duplicate today.",
        },
        "questions": {
            "phish": {
                "type": "noul",
                "instructions": "Is this message a scam?",
                "criteria": {"true": {"desc": "scam"}, "false": "legitimate"},
                "labels": {"true": "yes", "false": "no"},
            },
            "tone": {
                "type": "score",
                "instructions": "How does this read?",
                "criteria": ["calm", "annoyed", "furious", "threatening", "legal action"],
            },
        },
    },
    {
        # Undescribed choice labels render as the label alone (never "label: null"), and list
        # criteria go through dict.fromkeys. Python stringifies the label, so int-looking labels
        # stay stable.
        "label": "list criteria render as bare labels",
        "state": "cancel my plan",
        "questions": {
            "kind": {
                "type": "choice",
                "instructions": "Which one?",
                "criteria": ["alpha", "beta", "gamma"],
            },
            "mixed": {
                "type": "choice",
                "instructions": "Which one?",
                "criteria": {"one": None, "two": "", "three": 0, "four": False},
            },
        },
    },
    {
        # Long enough to overrun max_len=64, so state_tokens_dropped > 0 and truncated is true.
        "label": "state truncation (right)",
        "state": (
            "We were billed twice. Please refund the duplicate today. "
            "The account is locked. The invoice is a duplicate charge. "
            "Please unlock the account and cancel the duplicate charge. "
            "We were billed twice again. The plan is not the one we asked for. "
            "Please refund the duplicate charge on the invoice today."
        ),
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which team should handle this request?",
                "criteria": {"billing": None, "technical": None, "sales": None},
            }
        },
    },
    {
        # A list state is a turn list: truncation keeps the *tail* (truncate_left=True).
        "label": "turn list truncation (left)",
        "state": [
            "hello",
            "world",
            "we were billed twice",
            "please refund the duplicate",
            "the account is locked",
            "the invoice is a duplicate charge",
            "please unlock the account",
            "and cancel the duplicate charge today",
        ],
        "questions": {
            "refund": {
                "type": "noul",
                "instructions": "Does the customer ask for money back?",
            }
        },
    },
    {
        # Twelve options against head_max_len=32 does not fit, so options collapse: every option
        # keeps the [MASK] marker and a truncated body, and `tokens_per_option` is reported.
        "label": "options collapse under a tight head budget",
        "state": "the body is locked",
        "questions": {
            "category": {
                "type": "choice",
                "instructions": "Which category fits best?",
                "criteria": {
                    "alpha": "one two three four five six seven",
                    "beta": "one two three four five six eight",
                    "gamma": "one two three four five six nine",
                    "delta": "one two three four five seven ten",
                    "mid": "one two three four five six eleven",
                    "other": "one two three four five six twelve",
                    "billing": "one two three four five six thirteen",
                    "technical": "one two three four five six fourteen",
                    "sales": "one two three four five six fifteen",
                    "locked": "one two three four five six sixteen",
                    "unlock": "one two three four five six seventeen",
                    "refund": "one two three four five six eighteen",
                },
            }
        },
    },
]


def main() -> int:
    tokenizer_dir, out_path = sys.argv[1], sys.argv[2]
    tok = Tokenizer(tokenizer_dir)

    payload = {
        "generator": "tools/reference/make_prepare_fixture.py",
        "reference": "laya_mlx",
        "tokenizer": tokenizer_dir,
        "max_len": MAX_LEN,
        "head_max_len": HEAD_MAX_LEN,
        "cases": [],
    }

    for case in CASES:
        state, questions = case["state"], case["questions"]
        truncate_left = isinstance(state, list)
        state_ids = tok(
            _serialize_state(state).replace(tok.mask_token, " "), add_special_tokens=False
        )["input_ids"]

        items = {}
        for qid, definition in questions.items():
            q = Agent._to_internal(definition)
            ids, markers, stats, state_stats = build_sequence(
                tok,
                state,
                q,
                MAX_LEN,
                HEAD_MAX_LEN,
                truncate_left=truncate_left,
                state_ids=state_ids,
                return_stats=True,
                return_truncation_stats=True,
            )
            if len(markers) != len(render_options(q)):
                raise SystemExit(f"{case['label']}/{qid}: too many options for the token budget")
            items[qid] = {
                "qtype": {"choice": 0, "score": 1, "noul": 2}[q["t"]],
                "ids": ids,
                "markers": markers,
                "options": stats,
                "state_stats": state_stats,
                "rendered_options": render_options(q),
            }
        payload["cases"].append(
            {
                "label": case["label"],
                "state": state,
                "questions": questions,
                "items": items,
            }
        )

    with open(out_path, "w") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)
        handle.write("\n")

    for case in payload["cases"]:
        summary = ", ".join(
            f"{qid}={len(item['ids'])}tok/{len(item['markers'])}mk"
            + (f"/dropped{item['state_stats']['state_tokens_dropped']}"
               if item["state_stats"]["state_tokens_dropped"]
               else "")
            + (f"/per={item['options']['tokens_per_option']}"
               if item["options"]["tokens_per_option"] is not None
               else "")
            for qid, item in case["items"].items()
        )
        print(f"  {case['label']}: {summary}")

    print(f"wrote {out_path} ({len(payload['cases'])} cases)")
    return 0


def _serialize_state(state) -> str:
    """Mirror laya_mlx.common.serialize_state without importing numpy-dependent modules twice."""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


if __name__ == "__main__":
    raise SystemExit(main())