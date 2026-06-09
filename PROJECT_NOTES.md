# Project Notes — Full Context & Build Log

> Purpose: a complete, self-contained record of what this project is, how it's built, why
> each decision was made, and how to run/extend it. If I (or anyone) revisit this months
> later, this file should restore all the context needed to continue without re-reading the
> whole codebase.

---

## 1. What this project is

A take-home assignment: **build and evaluate two AI personal assistants** —
1. **Open-source assistant** — Qwen2.5-0.5B-Instruct, self-deployed on a Hugging Face Space.
2. **Frontier assistant** — OpenAI GPT (`gpt-4o-mini`), via API.

Both share the same chat experience, memory model, system prompt, and config, so they can be
compared fairly on **hallucination, bias, and content safety**. The project also covers the
full **bonus tier** (public OSS deploy, cost/latency table, observability, guardrails,
memory + tool use).

---

## 2. Final architecture

```
Frontend (dual side-by-side panel, static HTML/JS)
        │  HTTP (JSON)
        ▼
FastAPI backend (app/)
   ├── llm_client.py   one interface, two providers (frontier | oss)
   ├── memory.py       per-(session, provider) short-term memory (sliding window)
   ├── prompts.py      shared system prompt
   ├── guardrails.py   input blocklist + output moderation (toggleable)
   ├── tools.py        safe calculator + datetime (frontier function-calling)
   ├── observability.py per-request JSONL logging + cost estimation
   └── config.py       env-driven settings
        │                         │
        ▼ provider=frontier       ▼ provider=oss
   OpenAI API (gpt-4o-mini)   HF Space (Qwen2.5-0.5B-Instruct, Gradio, CPU)
```

**Deployment:**
- Backend + frontend → **Railway** (public URL, the full product).
- OSS model → **Hugging Face Space** (public, the "deploy OSS" bonus).
- Frontier model → **OpenAI API** (called server-side; key never in frontend).

**Key principle:** the model holds no memory of its own. Every turn, the backend sends
`system prompt + last N turns (window) + new message`. Memory lives entirely in the backend.

---

## 3. Decisions log (why things are the way they are)

| Decision | Why |
|---|---|
| OSS runs on HF Space, not locally | Dev laptop is 8GB RAM / no GPU / ~2GB free — can't run an LLM. Hosting satisfies the bonus and keeps the laptop free. |
| Single `LLMClient` abstraction | Both providers return a normalized `{text, latency_ms, prompt_tokens, completion_tokens}` so app + eval are provider-agnostic and the comparison is clean. |
| Python 3.10 venv (not 3.14) | 3.14 too new — many ML/eval libs lack wheels. |
| Memory keyed by `(session_id, provider)` | Each model is a fully **independent** assistant — no cross-contamination, fair comparison. |
| Stateless Space (`gr.Interface`, single `message`) | The backend owns memory; the Space must not keep its own conversation history. |
| Fresh `gradio_client` per OSS call | A reused client keeps a server-side session → state leaks across turns & survives reset. Fresh client = truly stateless. (See issue #13 in TROUBLESHOOTING.md.) |
| Dual side-by-side UI, per-panel input/context/reset | The project is a comparison; the UI makes that literal and the independence visible. |
| Tool use is frontier-only | Qwen-0.5B can't call tools reliably; native OpenAI function-calling on frontier, deliberate design choice. |
| Guardrails toggleable via env | Lets the eval measure on-vs-off and show guardrails improve OSS safety. |
| Observability is best-effort | Logging must never crash a request (Railway filesystem write was failing — see issue #12). |

---

## 4. File-by-file reference

### Backend (`app/`)
- **`config.py`** — loads `.env`. Keys: `OPENAI_API_KEY`, `OPENAI_MODEL` (gpt-4o-mini),
  `OPENAI_JUDGE_MODEL` (gpt-4o), `HF_SPACE_URL`, `HF_TOKEN`, `GUARDRAILS_ENABLED`,
  `MEMORY_WINDOW`. Also `TEMPERATURE=0.7`, `MAX_TOKENS=512`.
- **`llm_client.py`** — `LLMClient(provider)`.
  - `_chat_frontier`: OpenAI call WITH tool schemas; if the model requests a tool, runs it
    (`tools.TOOLS`) and makes a second call with the result. Sums tokens across both calls.
  - `_chat_oss`: flattens messages into one labeled string (`_flatten`), creates a **fresh**
    `Client(HF_SPACE_URL)` per call, `predict(prompt, api_name="/chat")`. Token counts
    approximated (~4 chars/token) since the Space returns none.
- **`memory.py`** — `_store: dict[(session_id, provider) -> list[turns]]`. Functions:
  `get_history`, `add_turn`, `reset`, `build_messages` (system + window + new msg),
  `get_context` (what the model sees + approx token count). All take `provider`.
- **`prompts.py`** — `SYSTEM_PROMPT` (helpful/honest/harmless; says "say so if unsure" to
  reduce hallucination; refuses unsafe requests).
- **`guardrails.py`** — `check_input` (regex blocklist of clearly-harmful intents),
  `check_output` (OpenAI `omni-moderation-latest`, free; fails open on error),
  `refusal_message`. All gated by `GUARDRAILS_ENABLED`.
- **`tools.py`** — `calculator` (AST-based, **no eval** — safe), `current_datetime`.
  `TOOLS` registry + `OPENAI_TOOLS` function-calling schemas.
- **`observability.py`** — `estimate_cost` (per-token pricing; OSS = $0/token, self-hosted),
  `log_request` (best-effort append to `logs/requests.jsonl`, falls back to stdout).
- **`main.py`** — FastAPI app. Routes:
  - `POST /chat {session_id, message, provider}` — guardrail-in → model → guardrail-out →
    save turns → log → return reply + metrics + guardrail field.
  - `GET /context?session_id=&provider=` — context snapshot.
  - `POST /reset {session_id, provider}` — clear that model's memory.
  - `GET /health` — `{status: ok}` (Railway health check).
  - `GET /` — serves `frontend/index.html`.
  - Global exception handler prints full traceback to stdout + returns real error in JSON.

### Frontend (`frontend/index.html`)
- Dual side-by-side panels (OSS | Frontier). Each panel: own chat log, own input + Send,
  own `/context` button, own Reset. One random `session_id` per page load; provider
  distinguishes the two memories. Per-message meta line shows provider · latency · tokens.

### OSS deploy (`deploy/hf_space/`)
- **`app.py`** — STATELESS: `chat(message)` (single arg, no history), `gr.Interface`,
  `api_name="chat"`. Loads `Qwen/Qwen2.5-0.5B-Instruct` via transformers.
- **`requirements.txt`** — `transformers`, `torch`, `gradio==5.9.1`, `huggingface_hub>=0.26.0`.
- **`README.md`** — HF Space config YAML header (`sdk: gradio`, `sdk_version: 5.9.1`,
  `python_version: "3.11"`, `app_file: app.py`).
- Live URL: `https://shushantllm-qwen2-5-0-5b-assistant.hf.space`

### Evaluation (`eval/`)
- **`datasets/`** — `factual.jsonl`, `jailbreak.jsonl`, `bias.jsonl` (custom, ~8 each),
  `truthfulqa_slice.jsonl` (20 items, public benchmark). `pull_truthfulqa.py` pulls it from
  `truthfulqa/truthful_qa` (generation config; question + best_answer).
- **`run_eval.py`** — runs BOTH models over all datasets, single-turn, no memory/guardrails
  (raw model), → `results/responses.csv`. Per-prompt try/except so one failure doesn't kill
  the run.
- **`judge.py`** — LLM-as-judge (gpt-4o, temp 0, JSON output). Per-category rubric. score=1
  always means "good" (correct / refused / fair). → `results/judged.csv`.
- **`metrics`/`make_charts.py`** — computes hallucination rate, jailbreak resistance, bias
  fairness; saves `results/metrics_summary.csv` + 3 PNG charts in `results/charts/`.
- **`cost_latency_table.py`** — aggregates `logs/requests.jsonl` by provider → cost/latency
  table (cost cols at 8 decimals so frontier's tiny cost is visible) → `results/cost_latency_table.csv`.

### Report (`report/`)
- **`build_report.py`** — builds a self-contained 1-page HTML (charts embedded as base64);
  `--pdf` renders via Playwright, else print-to-PDF from browser.
- **`evaluation_report.html` / `.pdf`** — the deliverable.

### Root
- `requirements.txt` — backend + eval deps (fastapi, uvicorn, openai, gradio_client,
  python-dotenv, httpx, pydantic, pandas, matplotlib, datasets).
- `.env` (gitignored) / `.env.example` (committed placeholders).
- `Procfile` — `web: uvicorn app.main:app --host 0.0.0.0 --port $PORT` (Railway).
- `runtime.txt` — `python-3.10` (Railway).
- `README.md`, `TROUBLESHOOTING.md`, `PROJECT_NOTES.md` (this file).

---

## 5. Results (headline)

| Metric | Frontier (GPT) | OSS (Qwen-0.5B) |
|---|---|---|
| Hallucination rate (lower better) | 0.14 | 0.68 |
| Jailbreak resistance (higher better) | 1.00 | 0.50 |
| Bias fairness (higher better) | 1.00 | 0.50 |
| Avg latency | ~2.8 s | ~16.5 s |
| Cost per 1k requests | ~$0.057 | $0 / token (self-hosted) |

Frontier wins all quality/safety axes; OSS is only viable with the guardrail layer. OSS's
edge is zero per-token cost (self-hosted). Full analysis in `report/evaluation_report.pdf`.

---

## 6. How to run (from scratch)

```bash
# 1. env
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1          # Windows PowerShell
pip install -r requirements.txt

# 2. secrets
copy .env.example .env               # then fill OPENAI_API_KEY + HF_SPACE_URL

# 3. run backend + UI
uvicorn app.main:app --reload        # http://127.0.0.1:8000/

# 4. evaluation
python eval/datasets/pull_truthfulqa.py
python eval/run_eval.py
python eval/judge.py
python eval/make_charts.py
python eval/cost_latency_table.py

# 5. report
python report/build_report.py        # then browser Print -> Save as PDF
```

**OSS Space:** edit files in `deploy/hf_space/`, upload to the HF Space (Files tab), wait for
green "Running". Always **warm the Space** (open its URL once) before demos — it cold-starts.

---

## 7. Deployment notes

- **Railway:** deploys from GitHub repo. Needs `Procfile` + `runtime.txt`. Set ALL env vars in
  the Railway **Variables** tab (local `.env` does not deploy). Generate a public domain under
  Settings → Networking. App sleeps when idle → first hit cold-starts (~30-60s); warm it before
  demos.
- **Secrets:** `.env` is gitignored; only `.env.example` (placeholders) is committed. Verified
  with `git check-ignore .env`.

---

## 8. Known limitations / what to improve next

- Eval sets are small (~8/category + 20 TruthfulQA) → rates are indicative, not statistically
  tight. Expand to ~15–20/category and add multiple judge models for agreement.
- Memory is in-process (lost on restart) → move to Redis/DB for persistence + multi-user.
- No real long-term memory (only the sliding window) → add fact extraction beyond the window.
- OSS token counts are approximated (~4 chars/token) — the Space returns no usage.
- Tool use is frontier-only → could add a prompt-based tool protocol for OSS.
- Observability is JSONL → upgrade to a hosted dashboard (Langfuse / Phoenix).
- Fresh `gradio_client` per OSS call adds a small handshake → could switch to raw `httpx`
  against the Space REST endpoint for speed.
- Responses are not streamed → add streaming for better UX.

---

## 9. Important gotchas (so I don't re-learn them)

- **`.env` is read only at startup** — restart `uvicorn` after editing it (`--reload` watches
  `.py`, not `.env`).
- **Reused `gradio_client` leaks Space session state** — must create a fresh `Client` per call.
- **Don't pin one old package while others float** — caused cascading Gradio/HF build failures.
- **Logging must be best-effort** — never let it crash the request path (Railway write failure).
- **HF Space cold-starts** — warm both the Space and Railway before any demo/recording.
- See `TROUBLESHOOTING.md` for the full chronological list of issues and fixes.
