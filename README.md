# AI Personal Assistant — OSS vs Frontier Comparison

Two functionally identical personal assistants — one backed by an **open-source model**
(Qwen2.5-0.5B-Instruct, self-deployed on Hugging Face Spaces) and one by a **frontier API**
(OpenAI GPT) — with a head-to-head evaluation on hallucination, bias, and safety.

## Features
- **Two independent assistants** behind one abstraction (`LLMClient`), swappable by provider.
- **Multi-turn chat** with **short-term memory** (sliding window), scoped **per model** (independent histories).
- **`/context`** command — inspect exactly what each model sees (system prompt + memory + token estimate).
- **Dual side-by-side UI** — drive both models, each with its own input, `/context`, and reset.
- **Guardrails** — input blocklist + output moderation, toggleable.
- **Robustness / graceful degradation** — never presents a blank/`None`/`"null"` reply (safe-output fallback); a provider/API/Space failure returns a friendly "temporarily unavailable" message (HTTP 200, rendered as a normal bubble) instead of a raw 500; the global error handler logs full details but shows the user only a generic message (no leaked internals). All backend-side, so identical for both models. Proven by `eval/safety_check.py`.
- **Native tool-calling on BOTH models** — `calculator`, `current_datetime`, `unit_convert`, `get_weather` (current weather by city via Open-Meteo, no API key). Frontier uses OpenAI function-calling; OSS uses Qwen2.5's trained `<tool_call>` template (`apply_chat_template(tools=...)`). One shared tool registry + one backend agentic loop, so both models get the same tools through the same mechanism — only the provider API differs. A header **Tools toggle** (default ON) gates tool use for **both** models identically; turning it OFF skips tool schemas (shorter prompts) and, on OSS, the second round-trip — useful as a speed lever. (OSS calls tools *less reliably* by model size — a fair comparison datapoint, not a parity break.)
- **Long-term memory via `/remember` (personalization)** — the user types `/remember <fact>` (and `/recall` to view); the backend stores it **deterministically** and injects it into the system prompt each turn. Because saving is a backend command (not a model tool-call), it behaves **identically and independently on both models** — Qwen and frontier save the same way. Scoped per `(session, provider)`; in-memory.
- **Identical feature set on both models** — same system prompt, short-term memory, tools, and chat flow. The **only** difference is the provider API.
- **Observability** — structured per-request logging (latency, tokens, cost, guardrail decisions). For OSS, latency is split into **true model inference** (`server_ms`, timed inside the Space) vs **transport overhead** (`overhead_ms` = wall-clock − server_ms: gradio_client handshake + network + queue), shown in the UI and the cost/latency table. (Frontier can't expose this — the OpenAI call is opaque — so the split is OSS-only.)
- **Evaluation harness** — custom + public-benchmark (TruthfulQA) datasets, LLM-as-judge, auto charts.

## Architecture
Frontend (dual panel) ──HTTP──> FastAPI backend ──> LLMClient ──┬─> OpenAI API (frontier)
│ memory (per provider)        └─> HF Space (Qwen, OSS)
│ guardrails
└ observability


- **Backend + frontend:** FastAPI serving a static dual-panel UI. Lightweight (only HTTP orchestration).
- **OSS model:** deployed to a free Hugging Face Space (Gradio, CPU). Called over HTTP — laptop never loads the model.
- **Memory keyed by `(session_id, provider)`** → each model is a fully independent assistant; clean comparison.

### Key architecture decisions
- **OSS via HF Space, not local.** Dev hardware (8GB RAM, no GPU) can't run an LLM well. Hosting on a free Space satisfies the "deploy OSS publicly" goal and keeps the laptop free.
- **Single `LLMClient` abstraction.** Both providers return a normalized `{text, latency_ms, prompt_tokens, completion_tokens}`, so app + eval are provider-agnostic.
- **Stateless OSS Space.** The Space holds no conversation state; the backend owns all memory. Prevents hidden cross-session contamination.
- **Independent per-model memory.** Avoids one model's context leaking into the other — essential for a fair comparison.

## Setup
1. **Python 3.10** + venv:
   ```bash
   py -3.10 -m venv venv
   .\venv\Scripts\Activate.ps1   # Windows
   pip install -r requirements.txt
Env: copy .env.example -> .env, fill OPENAI_API_KEY and HF_SPACE_URL.
Run backend + UI:

uvicorn app.main:app --reload
Open http://127.0.0.1:8000/
Deploy OSS model: see deploy/hf_space/ — create a Gradio Space, upload those files.
Evaluation

python eval/datasets/pull_truthfulqa.py   # pull public benchmark slice
python eval/run_eval.py                    # both models over all datasets
python eval/judge.py                       # LLM-as-judge scoring
python eval/make_charts.py                 # metrics + infographic charts
python eval/cost_latency_table.py          # cost + latency table
Outputs: eval/results/ (CSVs + charts/*.png).

Tool-calling eval (separate + independent — deterministic, no LLM judge):
python eval/run_tool_eval.py               # 10 tool tasks per model; counts failures
Outputs: eval/results/tool_responses.csv, tool_metrics.csv, charts/tool_calling.png.
Frontier needs OPENAI_API_KEY; OSS needs the (redeployed) HF Space warm.

Multi-turn hallucination eval (separate + independent — LLM-as-judge):
python eval/run_multiturn_eval.py          # plant facts, probe at gaps 1/5/10; hallucination vs turn-distance
Outputs: eval/results/multiturn_responses.csv, multiturn_metrics.csv, charts/multiturn_hallucination.png.
Replays scripted conversations turn-by-turn; temporarily raises MEMORY_WINDOW so facts stay
in-context (isolates recall degradation from the window cutoff). Needs OPENAI_API_KEY + warm Space.

Safety / robustness smoke-test (no API key / network needed — uses TestClient):
python eval/safety_check.py                 # asserts blocklist, safe-output, graceful degradation, no leaks
Exits non-zero on any failure (can gate a demo).

ARC standard-benchmark eval (separate + independent — deterministic letter-match, no judge):
python eval/datasets/pull_arc.py           # one-time: pull ARC-Challenge + ARC-Easy slices (~25 each)
python eval/run_arc_eval.py                # 4-choice MC accuracy per model x config
Outputs: eval/results/arc_responses.csv, arc_metrics.csv, charts/arc_accuracy.png.
Reports accuracy + format-failure rate per (provider x config). Needs OPENAI_API_KEY + warm Space.
Note: scored on the model's *generated* letter (chat-API setting), so absolute values may differ
from log-prob leaderboard numbers; the frontier-vs-OSS comparison is valid (both scored identically).

Results (summary)
Metric	Frontier (GPT)	OSS (Qwen-0.5B)
Hallucination rate (lower better)	0.14	0.68
Jailbreak resistance (higher better)	1.00	0.50
Bias fairness (higher better)	1.00	0.50
Frontier outperforms on all three axes; the guardrail layer is what makes the OSS assistant viable.

Tradeoffs made
Qwen-0.5B is tiny — chosen to fit free CPU hosting; it hallucinates more and resists jailbreaks less. The comparison makes this explicit.
In-memory store — simple dict, lost on restart. Fine for the assignment; production would use Redis/DB.
Approximate token counts for OSS — the Space returns no usage; estimated at ~4 chars/token.
Small eval sets (~8/category + 20 TruthfulQA) — keeps cost/runtime low; rates are indicative not statistically tight.
Moderation fails open — if the moderation API errors, chat continues (availability over strictness).
What I'd improve with more time
Larger, balanced eval sets + multiple judge models for agreement.
Real long-term memory (fact extraction beyond the window) + tool use.
Persistent memory store (Redis) and per-user sessions.
Hosted dashboard for observability (Langfuse/Phoenix) instead of JSONL.
Stream responses; warm/keep-alive the OSS Space to cut cold-start latency.
Deployment
Backend + UI: Railway (FastAPI from repo; set env vars in dashboard; start uvicorn app.main:app --host 0.0.0.0 --port $PORT).
OSS model: Hugging Face Space (public).