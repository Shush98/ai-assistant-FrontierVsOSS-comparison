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
- **Observability** — structured per-request logging (latency, tokens, cost, guardrail decisions).
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