"""Pull a slice of TruthfulQA (public benchmark) -> truthfulqa_slice.jsonl.
Run once: python eval/datasets/pull_truthfulqa.py
"""
import json
import os
from datasets import load_dataset

N = 20  # slice size
OUT = os.path.join("eval", "datasets", "truthfulqa_slice.jsonl")


def main():
    # 'generation' config gives best_answer (gold) + the question.
    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    ds = ds.select(range(min(N, len(ds))))

    with open(OUT, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            rec = {
                "id": f"tqa{i+1}",
                "prompt": row["question"],
                "gold": row["best_answer"],
                "category": row.get("category", ""),
            }
            f.write(json.dumps(rec) + "\n")

    print(f"Wrote {min(N, len(ds))} TruthfulQA items -> {OUT}")


if __name__ == "__main__":
    main()