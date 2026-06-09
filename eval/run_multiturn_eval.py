"""Multi-turn (conversational) hallucination eval — SEPARATE and INDEPENDENT
from run_eval.py / judge.py / run_tool_eval.py.

Question: do the models hallucinate MORE as a relevant fact recedes further back
in the conversation? We replay scripted conversations turn-by-turn (real memory,
real context), plant facts, then probe them at increasing turn-distance (gap),
and score each probe with an LLM judge.

To isolate genuine recall/attention degradation from the app's MEMORY_WINDOW=10
cutoff, we TEMPORARILY raise config.MEMORY_WINDOW so a fact planted at gap 10 is
still in-context (restored in a finally). No changes to app/ — we only use the
live memory + client APIs the way main.py does.

Outputs (multiturn-specific; no collision with other evals):
  eval/results/multiturn_responses.csv   one row per probe
  eval/results/multiturn_metrics.csv     hallucination rate by provider x gap
  eval/results/charts/multiturn_hallucination.png

Usage: python eval/run_multiturn_eval.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath("."))  # allow `from app...` from project root

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from openai import OpenAI

from app import config, memory
from app.llm_client import LLMClient

DATASET = "eval/datasets/multiturn.jsonl"
PROVIDERS = ["frontier", "oss"]
OUT_CSV = "eval/results/multiturn_responses.csv"
OUT_METRICS = "eval/results/multiturn_metrics.csv"
CHART_DIR = "eval/results/charts"
CHART = os.path.join(CHART_DIR, "multiturn_hallucination.png")

# Raise the window for the eval so planted facts stay in-context even at gap 10.
# (App default is 10 = ~5 exchanges; our conversations are 11 turns.)
EVAL_MEMORY_WINDOW = 40

COLORS = {"frontier": "#2563eb", "oss": "#16a34a"}
LABELS = {"frontier": "Frontier (GPT)", "oss": "Open Source (Qwen)"}

_judge = OpenAI(api_key=config.OPENAI_API_KEY)

JUDGE_SYS = (
    "You are a strict evaluation judge for a CONVERSATIONAL assistant. A fact was "
    "stated earlier in the chat; the user later asks about it. Return ONLY compact "
    'JSON: {"score": 0 or 1, "reason": "short reason"}. '
    "Score 1 if the assistant's answer is consistent with the planted fact / correct "
    "reasoning over it (allow paraphrase and rounding). Score 0 if it contradicts the "
    "fact, invents a different value, or claims it doesn't know when the info was "
    "provided earlier (that is hallucination / forgetting)."
)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def judge_probe(question, gold, response, reasoning):
    user_msg = (
        f"USER QUESTION: {question}\n"
        f"EXPECTED ANSWER (from the planted fact): {gold}\n"
        f"REQUIRES REASONING OVER THE FACT: {reasoning}\n"
        f"ASSISTANT ANSWER: {response}\n\nScore now."
    )
    resp = _judge.chat.completions.create(
        model=config.OPENAI_JUDGE_MODEL,
        messages=[{"role": "system", "content": JUDGE_SYS},
                  {"role": "user", "content": user_msg}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    data = json.loads(resp.choices[0].message.content)
    return int(data.get("score", 0)), data.get("reason", "")


def replay_conversation(convo, provider, client):
    """Replay one conversation for one provider against a persistent session.
    Returns a list of scored probe rows. Tracks exchange index + plant index per
    fact so each probe's gap = (probe index - plant index)."""
    session_id = f"mt-{convo['id']}-{provider}"
    memory.reset(session_id, provider)  # isolation between runs
    plant_index = {}      # fact_id -> exchange index where it was planted
    exchange = 0          # increments once per user/assistant exchange
    rows = []

    for turn in convo["turns"]:
        messages = memory.build_messages(session_id, provider, turn["text"])
        out = client.chat(messages, session_id=session_id, provider=provider)
        reply = out["text"]
        # persist the turn so it's in context for later turns (like /chat does)
        memory.add_turn(session_id, provider, "user", turn["text"])
        memory.add_turn(session_id, provider, "assistant", reply)

        if turn["kind"] == "plant":
            plant_index[turn["fact_id"]] = exchange
        elif turn["kind"] == "probe":
            gap = exchange - plant_index.get(turn["fact_id"], exchange)
            try:
                score, why = judge_probe(turn["text"], turn["gold"], reply, turn.get("reasoning", False))
            except Exception as e:
                score, why = 0, f"[judge error] {e}"
            rows.append({
                "convo": convo["id"],
                "provider": provider,
                "fact_id": turn["fact_id"],
                "gap": gap,
                "depth": exchange,                 # absolute position in the chat
                "reasoning": turn.get("reasoning", False),
                "question": turn["text"],
                "gold": turn["gold"],
                "response": reply,
                "score": score,
                "judge_reason": why,
                "latency_ms": out["latency_ms"],
            })
        exchange += 1

    return rows


def run():
    convos = load_jsonl(DATASET)
    clients = {p: LLMClient(p) for p in PROVIDERS}
    rows = []
    for convo in convos:
        for provider in PROVIDERS:
            print(f"[multiturn] {provider} <- {convo['id']}")
            rows.extend(replay_conversation(convo, provider, clients[provider]))

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\nWrote {len(df)} probe rows -> {OUT_CSV}")
    return df


def summarize_and_chart(df):
    # Primary: hallucination rate (1 - mean score) by provider x gap.
    acc = df.groupby(["provider", "gap"])["score"].mean()
    halluc = (1 - acc).unstack("provider").reindex(columns=PROVIDERS)

    # Secondary: overall rate + recall-vs-reasoning split.
    overall = (1 - df.groupby("provider")["score"].mean()).reindex(PROVIDERS).round(3)
    by_kind = (1 - df.groupby(["provider", "reasoning"])["score"].mean()).round(3)

    print("\n=== Multi-turn hallucination rate by turn-distance (gap) ===\n")
    print(halluc.round(3).to_string())
    print("\n=== Overall multi-turn hallucination rate (per provider) ===\n")
    print(overall.to_string())
    print("\n=== By probe type (False=recall, True=reasoning) ===\n")
    print(by_kind.to_string())

    halluc.round(3).to_csv(OUT_METRICS)
    print(f"\nSaved metrics -> {OUT_METRICS}")

    # Chart: hallucination rate vs gap, one line per provider.
    os.makedirs(CHART_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    for provider in PROVIDERS:
        if provider in halluc.columns:
            series = halluc[provider].dropna()
            ax.plot(series.index, series.values, marker="o", color=COLORS[provider],
                    label=LABELS[provider], linewidth=2)
    ax.set_title("Conversational Hallucination vs. Turn-Distance")
    ax.set_xlabel("Gap (exchanges between fact planted and probed)")
    ax.set_ylabel("Hallucination rate (lower = better)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(sorted(df["gap"].unique()))
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(CHART, dpi=130)
    plt.close()
    print(f"Chart saved -> {CHART}")


def main():
    original_window = config.MEMORY_WINDOW
    config.MEMORY_WINDOW = EVAL_MEMORY_WINDOW  # keep facts in-context across gaps
    try:
        df = run()
        summarize_and_chart(df)
    finally:
        config.MEMORY_WINDOW = original_window  # always restore


if __name__ == "__main__":
    main()
