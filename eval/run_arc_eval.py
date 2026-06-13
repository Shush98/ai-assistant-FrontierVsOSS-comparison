"""ARC benchmark eval — SEPARATE and INDEPENDENT from the other evals.

ARC (AI2 Reasoning Challenge) is a recognized public benchmark: grade-school
science, 4-choice multiple-choice. We present each question with lettered
choices, ask the model to answer with ONLY a letter, parse the chosen letter,
and compare to the gold — DETERMINISTIC scoring, no LLM judge. Runs both
configs (ARC-Challenge + ARC-Easy) for both models.

Methodology note: published ARC leaderboard numbers often use log-probability
scoring over the choices; we score the model's GENERATED letter (the realistic
chat-API setting, and the only option for the OSS HTTP endpoint). Absolute values
may therefore differ from leaderboards, but the frontier-vs-OSS comparison stays
valid because both models are scored identically.

Outputs (ARC-specific paths; no collision with other evals):
  eval/results/arc_responses.csv   per item
  eval/results/arc_metrics.csv     accuracy + format-failure rate by provider x config
  eval/results/charts/arc_accuracy.png

Usage: python eval/run_arc_eval.py
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.abspath("."))  # allow `from app...` from project root

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from app.llm_client import LLMClient
from app.prompts import SYSTEM_PROMPT

DATASETS = {
    "ARC-Challenge": "eval/datasets/arc_challenge.jsonl",
    "ARC-Easy": "eval/datasets/arc_easy.jsonl",
}
PROVIDERS = ["frontier", "oss"]
OUT_CSV = "eval/results/arc_responses.csv"
OUT_METRICS = "eval/results/arc_metrics.csv"
CHART_DIR = "eval/results/charts"
CHART = os.path.join(CHART_DIR, "arc_accuracy.png")

COLORS = {"frontier": "#2563eb", "oss": "#16a34a"}
LABELS = {"frontier": "Frontier (GPT)", "oss": "Open Source (Qwen)"}

# First standalone A/B/C/D in the reply (tolerant of "The answer is C.", "(C)", "c").
_LETTER_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_letter(text: str):
    """Return the chosen letter A/B/C/D, or None if the reply has no clear letter
    (a real failure mode — the model didn't answer in the required format)."""
    m = _LETTER_RE.search(text or "")
    return m.group(1).upper() if m else None


def build_prompt(item):
    choices = "\n".join(f"{L}. {item['choices'][L]}" for L in ["A", "B", "C", "D"])
    user = (
        "Answer the following multiple-choice question. "
        "Reply with ONLY the letter (A, B, C, or D).\n\n"
        f"Question: {item['question']}\n{choices}\nAnswer:"
    )
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]


def run():
    clients = {p: LLMClient(p) for p in PROVIDERS}
    rows = []
    for config, path in DATASETS.items():
        items = load_jsonl(path)
        for item in items:
            messages = build_prompt(item)
            for provider in PROVIDERS:
                print(f"[arc:{config}] {provider} <- {item['id']}")
                try:
                    # Tools OFF: pure reasoning, no tool assistance.
                    out = clients[provider].chat(messages, tools_enabled=False)
                    parsed = parse_letter(out["text"])
                    rows.append({
                        "config": config,
                        "id": item["id"],
                        "provider": provider,
                        "gold": item["gold"],
                        "parsed": parsed,
                        "correct": int(parsed == item["gold"]),
                        "unparsed": int(parsed is None),
                        "response": out["text"],
                        "latency_ms": out["latency_ms"],
                    })
                except Exception as e:
                    print(f"  ERROR: {e}")
                    rows.append({
                        "config": config, "id": item["id"], "provider": provider,
                        "gold": item["gold"], "parsed": None, "correct": 0,
                        "unparsed": 1, "response": f"[ERROR] {e}", "latency_ms": -1,
                    })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\nWrote {len(df)} ARC rows -> {OUT_CSV}")
    return df


def summarize_and_chart(df):
    acc = df.groupby(["config", "provider"])["correct"].mean().unstack("provider")
    acc = acc.reindex(columns=PROVIDERS)
    fail = df.groupby(["config", "provider"])["unparsed"].mean().unstack("provider")
    fail = fail.reindex(columns=PROVIDERS)

    print("\n=== ARC accuracy (provider x config) ===\n")
    print(acc.round(3).to_string())
    print("\n=== Format-failure rate (unparseable replies) ===\n")
    print(fail.round(3).to_string())

    # Combined metrics CSV: accuracy + format-failure, side by side.
    metrics = acc.round(3).add_suffix("_accuracy").join(
        fail.round(3).add_suffix("_format_fail"))
    metrics.to_csv(OUT_METRICS)
    print(f"\nSaved metrics -> {OUT_METRICS}")

    # Chart: grouped bars, accuracy per (config, provider).
    os.makedirs(CHART_DIR, exist_ok=True)
    ax = acc.plot(kind="bar", color=[COLORS[p] for p in acc.columns],
                  figsize=(8, 5), rot=0,
                  label=[LABELS[p] for p in acc.columns])  # label each bar series directly
    ax.set_title("ARC Accuracy (higher = better)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", padding=2, fontsize=8)
    # Add the random-guess baseline AFTER the bars; legend only labels the bar
    # series (handles taken from the containers, so the baseline line can't shift
    # the label mapping).
    ax.axhline(0.25, color="#9ca3af", linestyle="--", linewidth=1)
    ax.text(ax.get_xlim()[1], 0.26, "random (0.25)", ha="right", va="bottom",
            fontsize=8, color="#6b7280")
    ax.legend(handles=list(ax.containers),
              labels=[LABELS[p] for p in acc.columns])
    plt.tight_layout()
    plt.savefig(CHART, dpi=130)
    plt.close()
    print(f"Chart saved -> {CHART}")


def main():
    df = run()
    summarize_and_chart(df)


if __name__ == "__main__":
    main()
