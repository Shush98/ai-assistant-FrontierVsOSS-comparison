import json
import os
import pandas as pd

LOG_FILE = os.path.join("logs", "requests.jsonl")


def main():
    if not os.path.exists(LOG_FILE):
        print("No logs yet. Use the app first.")
        return
    rows = [json.loads(l) for l in open(LOG_FILE, encoding="utf-8")]
    df = pd.DataFrame(rows)

    aggs = dict(
        requests=("provider", "count"),
        avg_latency_ms=("latency_ms", "mean"),
        p95_latency_ms=("latency_ms", lambda s: s.quantile(0.95)),
        avg_prompt_tok=("prompt_tokens", "mean"),
        avg_completion_tok=("completion_tokens", "mean"),
        total_cost_usd=("cost_usd", "sum"),
        avg_cost_usd=("cost_usd", "mean"),
    )
    # True-model-latency vs transport-overhead split (OSS-only; frontier logs
    # these as null, older logs may lack the columns entirely → guard).
    if "server_ms" in df.columns:
        aggs["avg_server_ms"] = ("server_ms", "mean")        # true inference
    if "overhead_ms" in df.columns:
        aggs["avg_overhead_ms"] = ("overhead_ms", "mean")    # gradio/network/queue

    table = df.groupby("provider").agg(**aggs)

    # Latency/token columns: 1 decimal is plenty.
    round_cols = ["avg_latency_ms", "p95_latency_ms", "avg_prompt_tok", "avg_completion_tok",
                  "avg_server_ms", "avg_overhead_ms"]
    for c in round_cols:
        if c in table.columns:
            table[c] = table[c].round(1)
    # Readable per-1k-requests projection from avg cost.
    table["cost_per_1k_req_usd"] = (table["avg_cost_usd"] * 1000).round(4)
    # Cost is sub-cent: format ONLY cost cols as fixed 8-decimal strings so they
    # don't collapse to 0 / sci-notation. Other columns keep normal formatting.
    for c in ["total_cost_usd", "avg_cost_usd"]:
        table[c] = table[c].map(lambda v: f"{v:.8f}")

    print("\n=== Cost + Latency Table ===\n")
    print(table.to_string())
    out = os.path.join("eval", "results", "cost_latency_table.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    table.to_csv(out)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()