"""Run both models over all eval datasets -> eval/results/responses.csv.
Each prompt is a fresh single-turn call (no memory, no guardrails) so we
measure the raw model. Usage: python eval/run_eval.py
"""
import json
import os
import sys

# allow `from app...` when run from project root
sys.path.insert(0, os.path.abspath("."))

import pandas as pd
from app.llm_client import LLMClient
from app.prompts import SYSTEM_PROMPT

DATASETS = {
    "factual": "eval/datasets/factual.jsonl",
    "truthfulqa": "eval/datasets/truthfulqa_slice.jsonl",
    "jailbreak": "eval/datasets/jailbreak.jsonl",
    "bias": "eval/datasets/bias.jsonl",
}
PROVIDERS = ["frontier", "oss"]
OUT = "eval/results/responses.csv"


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    clients = {p: LLMClient(p) for p in PROVIDERS}
    rows = []

    for category, path in DATASETS.items():
        items = load_jsonl(path)
        for item in items:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["prompt"]},
            ]
            for provider in PROVIDERS:
                print(f"[{category}] {provider} <- {item['id']}")
                try:
                    out = clients[provider].chat(messages)
                    rows.append({
                        "category": category,
                        "id": item["id"],
                        "provider": provider,
                        "prompt": item["prompt"],
                        "gold": item.get("gold", ""),
                        "response": out["text"],
                        "latency_ms": out["latency_ms"],
                        "prompt_tokens": out["prompt_tokens"],
                        "completion_tokens": out["completion_tokens"],
                    })
                except Exception as e:
                    print(f"  ERROR: {e}")
                    rows.append({
                        "category": category, "id": item["id"], "provider": provider,
                        "prompt": item["prompt"], "gold": item.get("gold", ""),
                        "response": f"[ERROR] {e}", "latency_ms": -1,
                        "prompt_tokens": 0, "completion_tokens": 0,
                    })

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nWrote {len(rows)} responses -> {OUT}")


if __name__ == "__main__":
    main()