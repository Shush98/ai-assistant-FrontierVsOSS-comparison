"""Tool-calling evaluation — SEPARATE and INDEPENDENT from run_eval.py / judge.py.

Measures how reliably each model accomplishes a task that requires calling a tool.
Unlike the other evals, scoring is DETERMINISTIC (no LLM judge): each task declares
the tool that should be used and a substring the correct answer must contain. A task
"fails" if the wrong tool (or no tool) was used, OR the expected value is missing from
the final reply.

This exercises the full native tool-calling loop in LLMClient (frontier function-calling
and OSS Qwen <tool_call> template) — i.e. the same path the live app uses.

Outputs (all tool-eval-specific, no collision with the other evals):
  eval/results/tool_responses.csv     per-task results
  eval/results/tool_metrics.csv       success/failure rate per provider
  eval/results/charts/tool_calling.png

Usage: python eval/run_tool_eval.py
"""
import json
import os
import sys
import uuid

sys.path.insert(0, os.path.abspath("."))  # allow `from app...` from project root

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from app import memory
from app.llm_client import LLMClient

DATASET = "eval/datasets/tool_calling.jsonl"
PROVIDERS = ["frontier", "oss"]
OUT_CSV = "eval/results/tool_responses.csv"
OUT_METRICS = "eval/results/tool_metrics.csv"
CHART_DIR = "eval/results/charts"
CHART = os.path.join(CHART_DIR, "tool_calling.png")

COLORS = {"frontier": "#2563eb", "oss": "#16a34a"}
LABELS = {"frontier": "Frontier (GPT)", "oss": "Open Source (Qwen)"}


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def score_task(item, out):
    """Return (success: int, used_correct_tool: bool, answer_ok: bool).
    success = the assistant accomplished the task = right tool AND right answer."""
    used_correct_tool = (out.get("tool_used") == item["expected_tool"])
    answer = (out.get("text") or "").lower()
    answer_ok = item["expected_substring"].lower() in answer
    success = int(used_correct_tool and answer_ok)
    return success, used_correct_tool, answer_ok


def run():
    items = load_jsonl(DATASET)
    clients = {p: LLMClient(p) for p in PROVIDERS}
    rows = []

    for item in items:
        for provider in PROVIDERS:
            # Fresh session per (task, provider) so tasks never contaminate each
            # other (esp. memory tools); mirrors an independent conversation.
            session_id = f"tooleval-{uuid.uuid4().hex[:8]}"
            print(f"[tool_calling] {provider} <- {item['id']}")
            try:
                messages = memory.build_messages(session_id, provider, item["prompt"])
                out = clients[provider].chat(
                    messages, session_id=session_id, provider=provider
                )
                success, tool_ok, answer_ok = score_task(item, out)
                rows.append({
                    "id": item["id"],
                    "provider": provider,
                    "prompt": item["prompt"],
                    "expected_tool": item["expected_tool"],
                    "tool_used": out.get("tool_used"),
                    "expected_substring": item["expected_substring"],
                    "response": out["text"],
                    "used_correct_tool": tool_ok,
                    "answer_ok": answer_ok,
                    "success": success,
                    "latency_ms": out["latency_ms"],
                })
            except Exception as e:
                print(f"  ERROR: {e}")
                rows.append({
                    "id": item["id"], "provider": provider, "prompt": item["prompt"],
                    "expected_tool": item["expected_tool"], "tool_used": None,
                    "expected_substring": item["expected_substring"],
                    "response": f"[ERROR] {e}", "used_correct_tool": False,
                    "answer_ok": False, "success": 0, "latency_ms": -1,
                })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\nWrote {len(df)} tool-eval rows -> {OUT_CSV}")
    return df


def summarize_and_chart(df):
    n = df.groupby("provider")["id"].count()
    succ = df.groupby("provider")["success"].sum()
    tool_ok = df.groupby("provider")["used_correct_tool"].sum()

    summary = pd.DataFrame({
        "tasks": n,
        "successes": succ,
        "failures": n - succ,
        "success_rate": (succ / n).round(3),
        "correct_tool_rate": (tool_ok / n).round(3),
    }).reindex(PROVIDERS)

    print("\n=== Tool-Calling Eval (per provider) ===\n")
    print(summary.to_string())
    summary.to_csv(OUT_METRICS)
    print(f"\nSaved metrics -> {OUT_METRICS}")

    # Chart: success vs failure counts per provider (stacked).
    os.makedirs(CHART_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    provs = [p for p in PROVIDERS if p in summary.index]
    succ_vals = [summary.loc[p, "successes"] for p in provs]
    fail_vals = [summary.loc[p, "failures"] for p in provs]
    x = range(len(provs))
    ax.bar(x, succ_vals, color=[COLORS[p] for p in provs], label="succeeded")
    ax.bar(x, fail_vals, bottom=succ_vals, color="#d1d5db", label="failed")
    ax.set_xticks(list(x))
    ax.set_xticklabels([LABELS[p] for p in provs])
    ax.set_ylabel("tasks")
    ax.set_title("Tool-Calling Task Success (out of %d)" % int(summary["tasks"].iloc[0]))
    for i, p in enumerate(provs):
        ax.text(i, succ_vals[i] / 2, str(int(succ_vals[i])), ha="center", va="center",
                color="white", fontweight="bold")
        if fail_vals[i]:
            ax.text(i, succ_vals[i] + fail_vals[i] / 2, str(int(fail_vals[i])),
                    ha="center", va="center", color="#374151")
    ax.legend()
    plt.tight_layout()
    plt.savefig(CHART, dpi=130)
    plt.close()
    print(f"Chart saved -> {CHART}")


def main():
    df = run()
    summarize_and_chart(df)


if __name__ == "__main__":
    main()
