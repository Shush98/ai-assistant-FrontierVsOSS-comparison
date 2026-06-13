"""Pull a slice of TriviaQA (public benchmark) -> factual.jsonl.

Short, closed-answer factual questions with a gold answer — the LLM-judge in
eval/judge.py scores the model reply against `gold`. We use the `rc.nocontext`
config (question + answer, no supporting document) and keep only items with a
clean, non-empty answer string.

Run once (needs network): python eval/datasets/pull_factual.py
Produces a committed slice so the eval itself needs no network.

If the dataset is unavailable (gated/renamed/offline), this writes a small
hand-authored fallback so the eval is never blocked — re-run with network for
the full 50-item public slice.
"""
import json
import os

N = 50  # slice size
OUT = os.path.join("eval", "datasets", "factual.jsonl")

# Minimal hand-authored fallback (used only if the public pull fails).
_FALLBACK = [
    {"prompt": "What is the capital of Australia?", "gold": "Canberra"},
    {"prompt": "Who wrote the play 'Romeo and Juliet'?", "gold": "William Shakespeare"},
    {"prompt": "What is the chemical symbol for gold?", "gold": "Au"},
    {"prompt": "How many continents are there on Earth?", "gold": "Seven"},
    {"prompt": "What planet is known as the Red Planet?", "gold": "Mars"},
    {"prompt": "In what year did World War II end?", "gold": "1945"},
    {"prompt": "What is the largest ocean on Earth?", "gold": "The Pacific Ocean"},
    {"prompt": "Who painted the Mona Lisa?", "gold": "Leonardo da Vinci"},
]


def _clean(ans: str) -> str:
    return (ans or "").strip()


def main():
    rows = []
    try:
        from datasets import load_dataset
        ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
        for row in ds:
            if len(rows) >= N:
                break
            q = (row.get("question") or "").strip()
            gold = _clean((row.get("answer") or {}).get("value", ""))
            if not q or not gold:
                continue
            rows.append({"id": f"f{len(rows)+1}", "prompt": q, "gold": gold})
        if not rows:
            raise RuntimeError("no clean rows pulled")
        print(f"Pulled {len(rows)} TriviaQA items.")
    except Exception as e:
        print(f"[warn] public pull failed ({e!r}); writing hand-authored fallback.")
        rows = [{"id": f"f{i+1}", "prompt": it["prompt"], "gold": it["gold"]}
                for i, it in enumerate(_FALLBACK)]

    with open(OUT, "w", encoding="utf-8") as f:
        for rec in rows:
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote {len(rows)} factual items -> {OUT}")


if __name__ == "__main__":
    main()
