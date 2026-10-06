"""Reference answers from the Python MLX implementation, for diffing against the TS port.

    uv run python tools/laya_ref.py <model-dir> <dtype> <out.json>
"""

import json
import sys

import laya_mlx as laya

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
                "criteria": {
                    f"label_{i}": f"description number {i}" for i in range(12)
                },
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
]

model_dir, dtype, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
agent = laya.load(model_dir, dtype=dtype)
payload = {"dtype": dtype, "cases": []}
for case in CASES:
    result = agent.predict(case["state"], case["questions"])
    payload["cases"].append(
        {
            "label": case["label"],
            "state": case["state"],
            "questions": case["questions"],
            "answers": result["answers"],
        }
    )
json.dump(payload, open(out_path, "w"), ensure_ascii=False, indent=1)
print(f"wrote {out_path} ({dtype})")