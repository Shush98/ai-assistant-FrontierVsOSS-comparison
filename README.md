# AI Personal Assistant — OSS vs Frontier Comparison

Two **functionally identical** personal assistants — one backed by an **open-source model**
(Qwen2.5-0.5B-Instruct, self-deployed on a Hugging Face Space) and one by a **frontier API**
(OpenAI GPT-4o-mini) — behind a single abstraction, with a head-to-head evaluation across
quality, safety, reasoning, tool-calling, and conversational memory.

The two assistants share the **same** system prompt, memory, tools, guardrails, and chat flow.
**The only difference is the provider API** — so any difference in behavior is attributable to
the model, not the harness.

---

## Features

- **Dual side-by-side UI** — drive both models at once; each panel has its own input, `/context`,
  and reset. A header **Tools** toggle (default ON) gates tool use for both models identically.
- **Multi-turn chat** with short-term sliding-window memory, scoped **per `(session, provider)`**
  so each model keeps a fully independent history.
- **Native tool-calling on BOTH models** — `calculator`, `unit_convert`, `current_datetime`,
  `get_weather` (Open-Meteo, no API key). Frontier uses OpenAI function-calling; OSS uses
  Qwen2.5's trained `<tool_call>` template. One shared registry + one backend agentic loop, so
  both get the same tools through the same mechanism.
- **Long-term memory via `/remember`** — `/remember <fact>` (and `/recall`) saves user facts
  **deterministically in the backend** and injects them into the system prompt. Because saving is
  a command (not a model decision), it behaves identically and independently on both models.
- **Guardrails** — input blocklist + OpenAI output moderation (toggleable via env).
- **Robustness / graceful degradation** — never shows a blank/`null` reply; a provider/API/Space
  failure returns a friendly "temporarily unavailable" message (HTTP 200) instead of a raw 500;
  the global handler logs full detail but never leaks internals. Proven by `eval/safety_check.py`.
- **Observability** — structured per-request JSONL logging (latency, tokens, cost, guardrail
  decisions). For OSS, latency splits into **true model inference** (`server_ms`, timed inside the
  Space) vs **transport overhead** (network + Space wake-up).
- **Evaluation harness** — five independent suites (quality/safety, ARC, tool-calling, multi-turn
  hallucination, cost/latency) with auto-generated charts and a one-command PDF report.

---

## Architecture

```
Frontend (dual side-by-side panel, static HTML/JS)
        │  HTTP (JSON)
        ▼
FastAPI backend (app/)
   ├── llm_client.py    one interface, two providers (frontier | oss)
   ├── memory.py        per-(session, provider) short-term memory (sliding window) + facts
   ├── commands.py      deterministic /remember and /recall
   ├── tools.py         shared tool registry + JSON-Schema (native tool-calling, both models)
   ├── prompts.py       shared system prompt
   ├── guardrails.py    input blocklist + output moderation (toggleable)
   ├── safety.py        safe-output + graceful-degradation helpers
   ├── observability.py per-request JSONL logging + cost estimation
   └── config.py        env-driven settings
        │                              │
        ▼ provider=frontier            ▼ provider=oss
   OpenAI API (gpt-4o-mini)      HF Space (Qwen2.5-0.5B GGUF, FastAPI/Docker, CPU)
```

- **Backend + frontend:** FastAPI serving a static dual-panel UI — lightweight HTTP orchestration.
- **OSS model:** deployed to a free Hugging Face Space (FastAPI Docker Space; 4-bit GGUF via
  llama.cpp). Called with one plain JSON POST, so the dev laptop never loads the model.
- **Memory keyed by `(session_id, provider)`** → each model is a fully independent assistant.

### Key architecture decisions

| Decision | Why |
|---|---|
| **OSS via HF Space, not local** | Dev hardware (8 GB RAM, no GPU) can't run an LLM well. A free public Space satisfies the "deploy OSS publicly" goal and keeps the laptop free. |
| **Single `LLMClient` abstraction** | Both providers return a normalized `{text, latency_ms, prompt_tokens, completion_tokens, …}`, so the app and every eval are provider-agnostic and the comparison stays clean. |
| **Strict feature parity** | Same prompt, memory, tools, and guardrails on both models. Any feature one model couldn't do reliably is implemented **backend-side** (e.g. `/remember`) so behavior is identical — the comparison isolates the model. |
| **Stateless OSS Space** | The Space holds no conversation state and runs no tools — it's a pure text generator. The backend owns memory and the agentic tool loop. Prevents hidden cross-session contamination. |
| **Independent per-model memory** | Keyed by `(session, provider)` so one model's context never leaks into the other. |
| **Native tool-calling on both** | Frontier via OpenAI function-calling; OSS via Qwen2.5's trained `<tool_call>` template — same registry, one backend loop. OSS is *less reliable* by model size, which is itself a fair comparison datapoint, not a parity break. |
| **Long-term memory as a command, not a tool** | A 0.5B model can't reliably *decide* to call a memory tool, so `/remember` is deterministic backend logic — identical and reliable on both models. |
| **Guardrails toggleable via env** | Lets the eval measure on-vs-off and show the guardrail layer is what makes OSS deployable. |
| **Fail-open everywhere** | Moderation, observability, and tool execution never crash a request — availability over strictness. |

---

## Setup

**1. Python 3.10 + virtualenv** (3.10 — newer versions lack wheels for parts of the ML/eval stack):

```bash
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1     # Windows PowerShell  (use: source venv/bin/activate on macOS/Linux)
pip install -r requirements.txt -r requirements-eval.txt   # runtime + eval/report deps
```

(Railway installs only `requirements.txt` — the eval/report stack stays local.)

**2. Secrets** — copy the example env and fill it in:

```bash
copy .env.example .env          # then edit .env
```

| Var | Purpose |
|---|---|
| `OPENAI_API_KEY` | Frontier model + moderation + LLM-judge (required) |
| `HF_SPACE_URL` | Your deployed OSS Space URL, e.g. `https://<user>-<space>.hf.space` |
| `OPENAI_MODEL` / `OPENAI_JUDGE_MODEL` | Defaults `gpt-4o-mini` / `gpt-4o` |
| `GUARDRAILS_ENABLED` | `true`/`false` (default true) |
| `MEMORY_WINDOW` | Short-term window size (default 10) |

**3. Run the backend + UI:**

```bash
uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000/>.

**4. Deploy the OSS model** — create a Gradio Space on Hugging Face and upload the files in
[`deploy/hf_space/`](deploy/hf_space/) (`app.py`, `requirements.txt`, `README.md`). Put the live
URL in `HF_SPACE_URL`. The Space sleeps when idle, so **warm it** (open its URL once) before a demo.

---

## Evaluation

All eval suites are **separate and independent** (own datasets, runners, and output paths under
`eval/results/`). Frontier needs `OPENAI_API_KEY`; OSS needs the Space warm.

```bash
# 1 · Quality & safety (custom factual/jailbreak/bias + public TruthfulQA, LLM-as-judge)
python eval/datasets/pull_truthfulqa.py     # one-time: pull the public benchmark slice
python eval/run_eval.py                     # run both models over all datasets
python eval/judge.py                        # LLM-as-judge scoring (GPT-4o, temp 0)
python eval/make_charts.py                  # metrics + infographic charts

# 2 · ARC standard benchmark (4-choice MC, deterministic letter-match — no judge)
python eval/datasets/pull_arc.py            # one-time: pull ARC-Challenge + ARC-Easy slices
python eval/run_arc_eval.py

# 3 · Tool-calling (20 tasks, deterministic: right tool AND right answer)
python eval/run_tool_eval.py

# 4 · Multi-turn hallucination (recall vs turn-distance, LLM-as-judge)
python eval/run_multiturn_eval.py

# 5 · Cost & latency (aggregates logs/requests.jsonl)
python eval/cost_latency_table.py

# Safety / robustness smoke-test (no API key or network — uses FastAPI TestClient)
python eval/safety_check.py                 # exits non-zero on any failure

# One-page PDF report bundling every suite (reads the CSVs above)
python report/build_report.py               # writes report/evaluation_report.html
#   then: open the HTML in a browser → Print → Save as PDF
#   or:   pip install playwright && playwright install chromium && python report/build_report.py --pdf
```

Outputs land in `eval/results/` (CSVs + `charts/*.png`); the bundled report is at
[`report/evaluation_report.pdf`](report/evaluation_report.pdf).

> **Note on ARC scoring:** we score the model's *generated* letter (the realistic chat-API
> setting, and the only option for the OSS HTTP endpoint), so absolute values may differ from
> log-probability leaderboard numbers. The frontier-vs-OSS comparison stays valid since both are
> scored identically.

---

## Results (summary)

| Metric | Frontier (GPT-4o-mini) | OSS (Qwen-0.5B) | Winner |
|---|---|---|---|
| Hallucination rate *(lower better)* | **0.14** | 0.68 | Frontier |
| Jailbreak resistance *(higher better)* | **1.00** | 0.50 | Frontier |
| Bias fairness *(higher better)* | **1.00** | 0.50 | Frontier |
| ARC-Challenge accuracy | **0.92** | 0.24 *(≈ random)* | Frontier |
| ARC-Easy accuracy | **0.92** | 0.64 | Frontier |
| Tool-calling success (20 tasks) | 0.90 | 0.90 | **Tie** |
| Multi-turn hallucination @ gap 10 | **0%** | 80% | Frontier |
| Avg latency | **~2.2 s** | ~23 s | Frontier |
| Cost per 1k requests | $0.05 | **$0.00 / token*** | OSS |

\* OSS has no per-token billing (self-hosted on free CPU); its real cost is latency +
infrastructure.

**Takeaways:** Frontier wins every *quality* axis, hallucinating ~5× less and reasoning far
better. But the tiny OSS model **matched frontier on well-scoped tool tasks**, and wins
decisively on per-token cost — so it's a genuine fit for narrow, deterministic, cost-sensitive
workloads, *with guardrails*. Full analysis + infographics in
[`report/evaluation_report.pdf`](report/evaluation_report.pdf).

---

## Tradeoffs made

- **Qwen-0.5B is tiny** — chosen to fit free CPU hosting. It hallucinates more, reasons worse, and
  calls tools less reliably. The comparison makes this gap explicit rather than hiding it.
- **OSS tool-calling is less reliable than frontier** — a 0.5B model emits valid `<tool_call>`
  blocks imperfectly. This is a fair datapoint, not a bug; the backend fails open to plain text.
- **In-memory store** — a simple dict, lost on restart. Fine for the assignment; production would
  use Redis/DB and per-user sessions.
- **OSS token counts are approximated** (~4 chars/token) — the Space returns no usage data.
- **Small eval sets** (~8–25 per suite) — keeps cost/runtime low; rates are *indicative*, not
  statistically tight.
- **Moderation fails open** — if the moderation API errors, chat continues (availability over
  strictness).
- **OSS latency is bounded by free CPU hosting** — no GPU. Mitigated by a 4-bit GGUF (llama.cpp)
  on the Space, plain-JSON transport (no gradio_client handshake), capped `max_new_tokens`, the
  Tools toggle, and a keep-warm ping (`.github/workflows/keep-warm.yml`) against cold starts.

---

## What I'd improve with more time

- **Larger, balanced eval sets** + multiple judge models for inter-judge agreement.
- **A bigger OSS model** (Qwen 1.5–7B) to close the quality/reasoning gap while keeping the
  $0-per-token advantage.
- **Persistent memory** (Redis/DB) and real long-term fact extraction (auto-detect facts, not
  just `/remember`), with a parity-safe update/overwrite rule for contradictory facts.
- **Retrieval grounding** for the OSS model to cut hallucination on factual queries.
- **GPU / warm host for the Space** to eliminate cold-starts and the ~10× latency gap; stream
  responses for better UX.
- **Hosted observability dashboard** (Langfuse / Phoenix) instead of JSONL, with the
  inference-vs-overhead latency split charted over time.

---

## Deployment

- **Backend + UI → Railway** — deploys the FastAPI app from the repo (`Procfile` + `runtime.txt`).
  Set all env vars in the Railway dashboard; start command:
  `uvicorn app.main:app --host 0.0.0.0 --port $PORT`. App sleeps when idle → warm it before demos.
- **OSS model → Hugging Face Space** (public, free CPU).
- **Secrets:** `.env` is gitignored; only `.env.example` (placeholders) is committed.

See [`PROJECT_NOTES.md`](PROJECT_NOTES.md) for the full build log and [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md)
for the chronological list of issues hit and fixed.
