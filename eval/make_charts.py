"""Aggregate judged.csv -> metrics + infographic charts.
Usage: python eval/make_charts.py
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed (headless)
import matplotlib.pyplot as plt

IN = "eval/results/judged.csv"
CHART_DIR = "eval/results/charts"
PROVIDERS = ["frontier", "oss"]
COLORS = {"frontier": "#2563eb", "oss": "#16a34a"}
LABELS = {"frontier": "Frontier (GPT)", "oss": "Open Source (Qwen)"}


def main():
    df = pd.read_csv(IN).fillna("")
    os.makedirs(CHART_DIR, exist_ok=True)

    # score=1 = good. Mean score per (category, provider) = success rate.
    pivot = df.groupby(["category", "provider"])["score"].mean().unstack("provider")
    pivot = pivot.reindex(columns=PROVIDERS)

    # Headline metrics (as rates, 0-1).
    # Hallucination rate = 1 - factual/truthfulqa correctness.
    fact = df[df.category.isin(["factual", "truthfulqa"])]
    halluc = 1 - fact.groupby("provider")["score"].mean()
    safety = df[df.category == "jailbreak"].groupby("provider")["score"].mean()  # refusal rate
    fairness = df[df.category == "bias"].groupby("provider")["score"].mean()      # fair rate

    summary = pd.DataFrame({
        "hallucination_rate": halluc,
        "safety_resistance_rate": safety,
        "bias_fairness_rate": fairness,
    }).reindex(PROVIDERS).round(3)

    print("\n=== Headline Metrics (per provider) ===\n")
    print(summary.to_string())
    summary.to_csv(os.path.join(CHART_DIR, "..", "metrics_summary.csv"))

    # --- Chart 1: per-category success rate (grouped bars) ---
    ax = pivot.plot(kind="bar", color=[COLORS[p] for p in pivot.columns],
                    figsize=(8, 5), rot=0)
    ax.set_title("Success Rate by Category (higher = better)")
    ax.set_ylabel("Mean score (1 = good)")
    ax.set_ylim(0, 1.05)
    ax.legend([LABELS[c] for c in pivot.columns])
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", padding=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, "success_by_category.png"), dpi=130)
    plt.close()

    # --- Chart 2: headline metrics (grouped bars) ---
    ax = summary.T.plot(kind="bar", color=[COLORS[p] for p in summary.index],
                        figsize=(8, 5), rot=15)
    ax.set_title("Headline Safety / Quality Metrics")
    ax.set_ylabel("Rate (0-1)")
    ax.set_ylim(0, 1.05)
    ax.legend([LABELS[p] for p in summary.index])
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", padding=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, "headline_metrics.png"), dpi=130)
    plt.close()

    # --- Chart 3: avg latency per provider ---
    valid = df[df.latency_ms >= 0]
    lat = valid.groupby("provider")["latency_ms"].mean().reindex(PROVIDERS)
    ax = lat.plot(kind="bar", color=[COLORS[p] for p in lat.index], figsize=(6, 5), rot=0)
    ax.set_title("Average Latency (lower = better)")
    ax.set_ylabel("ms")
    ax.set_xticklabels([LABELS[p] for p in lat.index])
    ax.bar_label(ax.containers[0], fmt="%.0f", padding=2)
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, "latency.png"), dpi=130)
    plt.close()

    print(f"\nCharts saved -> {CHART_DIR}")


if __name__ == "__main__":
    main()