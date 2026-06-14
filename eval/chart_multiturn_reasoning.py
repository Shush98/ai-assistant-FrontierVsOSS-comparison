"""Additive chart: multi-turn hallucination on RECALL vs REASONING probes.

Reads the rows already produced by run_multiturn_eval.py (eval/results/
multiturn_responses.csv) and renders a grouped bar chart of hallucination rate
(1 - mean score) split by probe type (recall vs reasoning) for each provider.
This highlights that the OSS model recalls planted facts fairly well but
hallucinates much more when it has to REASON over them.

Standalone — does NOT re-run the eval or touch run_multiturn_eval.py.

Usage: python eval/chart_multiturn_reasoning.py
"""
import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

IN = "eval/results/multiturn_responses.csv"
CHART_DIR = "eval/results/charts"
CHART = os.path.join(CHART_DIR, "multiturn_reasoning.png")

# Match run_multiturn_eval.py so both charts read consistently.
PROVIDERS = ["frontier", "oss"]
COLORS = {"frontier": "#2563eb", "oss": "#16a34a"}
LABELS = {"frontier": "Frontier (GPT)", "oss": "Open Source (Qwen)"}
KIND_LABELS = {False: "Recall\n(restate the fact)", True: "Reasoning\n(compute over the fact)"}


def main():
    df = pd.read_csv(IN)

    # Hallucination rate = 1 - mean(score), grouped by probe type x provider.
    halluc = (1 - df.groupby(["reasoning", "provider"])["score"].mean()).unstack("provider")
    halluc = halluc.reindex(columns=PROVIDERS)
    counts = df.groupby(["reasoning", "provider"]).size().unstack("provider").reindex(columns=PROVIDERS)

    kinds = [k for k in (False, True) if k in halluc.index]  # recall first, then reasoning

    print("=== Multi-turn hallucination rate: recall vs reasoning ===\n")
    print(halluc.round(3).rename(index={False: "recall", True: "reasoning"}).to_string())

    os.makedirs(CHART_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    x = range(len(kinds))
    width = 0.36

    for i, provider in enumerate(PROVIDERS):
        offsets = [xi + (i - 0.5) * width for xi in x]
        vals = [halluc.loc[k, provider] for k in kinds]
        bars = ax.bar(offsets, vals, width, color=COLORS[provider], label=LABELS[provider])
        # value + n labels on top of each bar
        for off, v, k in zip(offsets, vals, kinds):
            n = int(counts.loc[k, provider])
            ax.text(off, v + 0.02, f"{v:.0%}\n(n={n})", ha="center", va="bottom",
                    fontsize=9, color=COLORS[provider])

    ax.set_title("Multi-turn Hallucination: Recall vs. Reasoning Probes")
    ax.set_ylabel("Hallucination rate (lower = better)")
    ax.set_xticks(list(x))
    ax.set_xticklabels([KIND_LABELS[k] for k in kinds])
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(CHART, dpi=130)
    plt.close()
    print(f"\nChart saved -> {CHART}")


if __name__ == "__main__":
    main()
