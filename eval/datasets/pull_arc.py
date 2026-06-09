"""Pull slices of ARC (AI2 Reasoning Challenge, public benchmark) -> JSONL.

ARC is grade-school science, 4-choice multiple-choice. We pull both configs:
  ARC-Challenge (the canonical hard set) and ARC-Easy.
Only clean 4-choice items are kept, and ARC's occasional numeric answer keys
(1..4) are normalized to letters (A..D) for uniform deterministic scoring.

Run once (needs network): python eval/datasets/pull_arc.py
Produces committed slices so the eval itself needs no network (like the
TruthfulQA slice).
"""
import json
import os

from datasets import load_dataset

N = 25  # slice size per config
CONFIGS = {
    "ARC-Challenge": "arc_challenge.jsonl",
    "ARC-Easy": "arc_easy.jsonl",
}
OUT_DIR = os.path.join("eval", "datasets")
LETTERS = ["A", "B", "C", "D"]
# ARC labels come as either letters (A-D) or numbers (1-4); normalize both.
_NUM_TO_LETTER = {"1": "A", "2": "B", "3": "C", "4": "D"}


def _to_letter(label: str) -> str | None:
    label = str(label).strip()
    if label in _NUM_TO_LETTER:
        return _NUM_TO_LETTER[label]
    if label in LETTERS:
        return label
    return None  # unexpected label -> caller skips the item


def pull_config(config: str, out_name: str) -> int:
    ds = load_dataset("allenai/ai2_arc", config, split="test")
    rows, written = [], 0
    prefix = "c" if "Challenge" in config else "e"

    for row in ds:
        if written >= N:
            break
        labels = row["choices"]["label"]
        texts = row["choices"]["text"]
        # Keep only clean 4-choice items.
        if len(labels) != 4 or len(texts) != 4:
            continue
        letter_labels = [_to_letter(l) for l in labels]
        gold = _to_letter(row["answerKey"])
        if None in letter_labels or gold is None:
            continue
        # Map each choice to its normalized letter (preserve display order A..D).
        choices = {letter: text for letter, text in zip(letter_labels, texts)}
        if set(choices) != set(LETTERS):  # must be exactly A,B,C,D
            continue

        written += 1
        rows.append({
            "id": f"arc_{prefix}{written}",
            "config": config,
            "question": row["question"],
            "choices": {L: choices[L] for L in LETTERS},  # ordered A..D
            "gold": gold,
        })

    out_path = os.path.join(OUT_DIR, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in rows:
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote {written} {config} items -> {out_path}")
    return written


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for config, out_name in CONFIGS.items():
        pull_config(config, out_name)


if __name__ == "__main__":
    main()
