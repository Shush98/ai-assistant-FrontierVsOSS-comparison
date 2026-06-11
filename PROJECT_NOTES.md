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
   ├── tools.py        shared tool registry + JSON-Schema (native tool-calling, both models)
   ├── observability.py per-request JSONL logging + cost estimation
   └── config.py       env-driven settings
        │                         │
        ▼ provider=frontier       ▼ provider=oss
   OpenAI API (gpt-4o-mini)   HF Space (Qwen2.5-0.5B GGUF, FastAPI/Docker, CPU)
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
| Stateless Space (FastAPI Docker, structured `messages`+`tools`) | The Space takes the full conversation + tool schemas as JSON (`POST /chat`), builds the prompt with `apply_chat_template(tools=...)`, returns RAW text. It owns NO memory and runs NO tools — the backend owns both. |
| FastAPI Docker Space + httpx (replaced Gradio + gradio_client) | gradio_client added a 1–3s fresh-connection handshake per call (fresh client was itself a workaround for Gradio session leakage, issue #13) and deserialized inconsistently across environments (issue #14-style drift). Plain JSON over a shared keep-alive `httpx.Client` removes the handshake, the queue/SSE hop, and the whole normalization layer. |
| GGUF (Q4_K_M) via llama.cpp on the Space (replaced transformers) | `torch_dtype="auto"` loaded bf16 on a CPU without fast bf16 paths; a 4-bit GGUF is several times faster on the free 2-vCPU tier. Prompt is still built by the HF tokenizer's Qwen tool template, so the `<tool_call>` contract is unchanged. Generation params (`max_new_tokens`, `temperature`) are now backend-passed per request. |
| Dual side-by-side UI, per-panel input/context/reset | The project is a comparison; the UI makes that literal and the independence visible. |
| Native tool-calling on BOTH (parity) | Earlier an invented `TOOL:` text protocol failed on OSS. The fix: use each model's NATIVE tool-calling — OpenAI function-calling for frontier, and Qwen2.5's *trained* `<tool_call>` template (`apply_chat_template(tools=...)`) for OSS. One shared registry (`tools.TOOL_SCHEMAS` / `run_tool`), one backend agentic loop. Same tools, same mechanism; only the provider API differs. OSS is less reliable by model size — a fair comparison datapoint, not a parity break. |
| Backend owns the tool loop; Space stays stateless | The HF Space just runs `apply_chat_template(messages, tools=...)` and returns RAW text (incl. any `<tool_call>`). The backend parses it, runs the tool from the single registry, and calls the Space again with the result. Keeps tools in one place and the Space a dumb, stateless generator. |
| Long-term memory = `/remember` + `/recall` commands | Replaced the earlier model-driven `remember_fact`/`recall_facts` TOOLS (Qwen-0.5B couldn't reliably decide to call them → memory behaved differently per model, breaking parity). Now the USER types `/remember <fact>` / `/recall`; the backend (`app/commands.py`) stores facts per-`(session,provider)` deterministically and injects them into the system prompt. Saving is identical & independent on both models; reading needs no tool call. In-memory. The 3 utility tools (calculator/datetime/unit_convert) remain native tool-calls. |
| Guardrails toggleable via env | Lets the eval measure on-vs-off and show guardrails improve OSS safety. |
| Observability is best-effort | Logging must never crash a request (Railway filesystem write was failing — see issue #12). |

---

## 4. File-by-file reference

### Backend (`app/`)
- **`config.py`** — loads `.env`. Keys: `OPENAI_API_KEY`, `OPENAI_MODEL` (gpt-4o-mini),
  `OPENAI_JUDGE_MODEL` (gpt-4o), `HF_SPACE_URL`, `HF_TOKEN`, `GUARDRAILS_ENABLED`,
  `MEMORY_WINDOW`. Also `TEMPERATURE=0.7`, `MAX_TOKENS=512`.
- **`llm_client.py`** — `LLMClient(provider)`. `chat(messages, session_id, provider,
    tools_enabled=True)` — native tool-calling on both, one backend loop, one registry. Capped
    at 1 tool round. `tools_enabled` (UI toggle, defaults True so eval callers are unaffected)
    gates tools for BOTH: frontier omits the `tools=` arg; OSS sends empty schemas AND skips the
    parse + 2nd round-trip → faster.
  - `_chat_frontier`: OpenAI call WITH `tools=tools.TOOL_SCHEMAS`; on `message.tool_calls`,
    runs each via `tools.run_tool`, appends `{role:"tool", tool_call_id, content}`, calls
    again. Sums tokens across both calls.
  - `_chat_oss`: POSTs `{messages, tools, max_new_tokens, temperature}` to the Space's
    `/chat` over a module-level shared keep-alive `httpx.Client` (no per-call handshake).
    Parses `<tool_call>{...}</tool_call>` (`_parse_tool_call`); on a hit, runs the tool and
    re-calls the Space with the result appended (assistant tool_calls + `{role:"tool", name,
    content}`). Fail-open to raw text. Tokens approximated (~4 chars/token). `_predict`
    returns `(text, server_ms)`; `_chat_oss` SUMS `server_ms` across both Space calls and
    reports `server_ms` (true inference) + `overhead_ms = wall-clock − server_ms`
    (transport). Frontier returns these as `None` (OpenAI is opaque) — split is OSS-only.
- **`tools.py`** — shared registry. `calculator` (AST, **no eval**), `current_datetime`
  (`zoneinfo`), `unit_convert` (length/weight/temp lookup), `get_weather` (current weather by
  city via **Open-Meteo** — free, no API key; geocode → forecast over `httpx`, WMO code →
  condition text, fail-open error strings, 8s timeout). `TOOLS` (name→callable), `TOOL_SCHEMAS`
  (JSON-Schema, sent to BOTH OpenAI and the Qwen template), `run_tool(name, args, session_id,
  provider)` (never-raises dispatch). NOTE: memory is NOT a tool — see `commands.py`.
- **`commands.py`** — deterministic slash-commands. `handle_command(session_id, provider,
  message)` → `/remember <fact>` (calls `memory.add_fact`) and `/recall` (reads
  `memory.get_facts`); returns the reply string, or `None` for non-commands (so chat proceeds).
  Handled in `main.py` BEFORE any model call → instant, free, identical & independent per
  provider.
- **`memory.py`** — `_store` (short-term turns) + `_facts` (long-term, both per
  `(session_id, provider)`). Functions: `get_history`, `add_turn`, `add_fact`/`get_facts`,
  `reset` (clears both), `build_messages` (system + injected facts + window + new msg),
  `get_context` (+ `long_term_facts`). All take `provider`.
- **`prompts.py`** — `SYSTEM_PROMPT` (helpful/honest/harmless; says "say so if unsure" to
  reduce hallucination; refuses unsafe requests).
- **`guardrails.py`** — `check_input` (regex blocklist of clearly-harmful intents),
  `check_output` (OpenAI `omni-moderation-latest`, free; fails open on error),
  `refusal_message`. All gated by `GUARDRAILS_ENABLED`.
- **`safety.py`** — robustness/graceful-degradation helpers. `safe_reply(text)` normalizes
  `None`/empty/`"null"`/`"None"` → `EMPTY_REPLY_FALLBACK` so the UI never shows a blank.
  `PROVIDER_ERROR_FALLBACK` (provider/API/Space down) and `GENERIC_ERROR` (last-resort handler)
  constants. Backend-side → identical for both models.
- **`observability.py`** — `estimate_cost` (per-token pricing; OSS = $0/token, self-hosted),
  `log_request` (best-effort append to `logs/requests.jsonl`, falls back to stdout).
- **`main.py`** — FastAPI app. Routes:
  - `POST /chat {session_id, message, provider}` — guardrail-in → /remember /recall command →
    model (wrapped in try/except → graceful degradation, HTTP 200 + `PROVIDER_ERROR_FALLBACK`
    + `error:true` on failure, reply NOT saved) → `safety.safe_reply` (no blank/None) →
    guardrail-out → save turns → log → return reply + metrics + guardrail + `tool_used`.
  - `GET /context?session_id=&provider=` — context snapshot.
  - `POST /reset {session_id, provider}` — clear that model's memory.
  - `GET /health` — `{status: ok}` (Railway health check).
  - `GET /` — serves `frontend/index.html`.
  - Global exception handler (last-resort): prints full traceback to stdout, returns ONLY a
    generic message (`safety.GENERIC_ERROR`) — no internals leaked to the client.

### Frontend (`frontend/index.html`)
- Dual side-by-side panels (OSS | Frontier). Each panel: own chat log, own input + Send,
  own `/context` button, own Reset. One random `session_id` per page load; provider
  distinguishes the two memories. Per-message meta line shows provider · latency · tokens.

### OSS deploy (`deploy/hf_space/`)
- **`app.py`** — STATELESS FastAPI endpoint: `POST /chat {messages, tools, max_new_tokens,
  temperature}`. Builds the prompt with the HF tokenizer's `apply_chat_template(messages,
  tools=...)` (Qwen's trained tool template — `<tool_call>` contract unchanged), generates
  with **llama.cpp on a 4-bit GGUF** (`Qwen/Qwen2.5-0.5B-Instruct-GGUF`, Q4_K_M, `n_ctx=4096`,
  generation behind a lock — llama.cpp isn't thread-safe). Returns **`{text, server_ms}`**:
  raw generated text + true inference time. Empty `tools` (toggle off) → no tool block →
  shorter prefill. `temperature=0` → greedy (for reproducible evals). `GET /health` for the
  keep-warm ping. Generation defaults (128 tokens, temp 0.7) preserved; backend passes
  `config.OSS_MAX_NEW_TOKENS` / `config.TEMPERATURE` per request.
- **`Dockerfile`** — Docker SDK Space (free tier): python:3.11-slim, uid-1000 user,
  llama-cpp-python from the prebuilt CPU wheel index (build tools as fallback), and the GGUF
  + tokenizer **baked into the image at build time** so cold starts skip the ~400MB download.
- **`requirements.txt`** — `fastapi`, `uvicorn`, `llama-cpp-python`, `transformers>=4.45.0`
  (tokenizer/template only — **no torch**), `huggingface_hub`.
- **`README.md`** — HF Space config YAML header (`sdk: docker`, `app_port: 7860`) + API contract.
- Live URL: `https://shushantllm-qwen2-5-0-5b-assistant.hf.space`

### Evaluation (`eval/`)
- **`datasets/`** — `factual.jsonl`, `jailbreak.jsonl`, `bias.jsonl` (custom, ~8 each),
  `truthfulqa_slice.jsonl` (20 items, public benchmark). `pull_truthfulqa.py` pulls it from
  `truthfulqa/truthful_qa` (generation config; question + best_answer). Also `tool_calling.jsonl`
  (10 tool tasks), `multiturn.jsonl` (5 conversations), and `arc_challenge.jsonl` /
  `arc_easy.jsonl` (~25 each, public ARC benchmark via `pull_arc.py`).
- **`run_eval.py`** — runs BOTH models over all datasets, single-turn, no memory/guardrails
  (raw model), → `results/responses.csv`. Per-prompt try/except so one failure doesn't kill
  the run.
- **`judge.py`** — LLM-as-judge (gpt-4o, temp 0, JSON output). Per-category rubric. score=1
  always means "good" (correct / refused / fair). → `results/judged.csv`.
- **`metrics`/`make_charts.py`** — computes hallucination rate, jailbreak resistance, bias
  fairness; saves `results/metrics_summary.csv` + 3 PNG charts in `results/charts/`.
- **`cost_latency_table.py`** — aggregates `logs/requests.jsonl` by provider → cost/latency
  table (cost cols at 8 decimals so frontier's tiny cost is visible) → `results/cost_latency_table.csv`.
  Now also reports `avg_server_ms` (true OSS inference) vs `avg_overhead_ms` (transport) when
  present; frontier/old-log rows show NaN (guarded). For OSS, `avg_latency_ms ≈ avg_server_ms +
  avg_overhead_ms`, making the gradio_client overhead explicit.
- **`run_tool_eval.py`** (SEPARATE + INDEPENDENT from the above) — tool-calling eval. Runs
  `datasets/tool_calling.jsonl` (10 tasks needing calculator/unit_convert) through
  the FULL `LLMClient.chat` tool loop for both models. **Deterministic scoring, no LLM judge**:
  a task succeeds only if the expected tool was used AND the expected substring is in the reply;
  otherwise it's a failure. Fresh session per (task, provider) so tasks don't contaminate each
  other. Self-contained (runs + scores + charts in one file). Outputs to tool-eval-specific
  paths so it never collides with the other evals: `results/tool_responses.csv`,
  `results/tool_metrics.csv`, `results/charts/tool_calling.png`.
- **`pull_arc.py`** / **`run_arc_eval.py`** (SEPARATE + INDEPENDENT) — ARC standard-benchmark
  eval. `pull_arc.py` pulls ~25-item slices of **ARC-Challenge** and **ARC-Easy** from
  `allenai/ai2_arc` (filters to clean 4-choice items, normalizes numeric answer keys → A–D),
  writing committed `datasets/arc_challenge.jsonl` / `arc_easy.jsonl`. `run_arc_eval.py` presents
  each question with lettered choices, asks for ONLY a letter, and scores **deterministically**
  (`parse_letter` → first A/B/C/D in the reply; no LLM judge). Metric: **accuracy per
  (provider × config)** + a **format-failure rate** (unparseable replies). Outputs:
  `results/arc_responses.csv`, `results/arc_metrics.csv`, `results/charts/arc_accuracy.png`
  (grouped bars, with a random-guess 0.25 baseline). Caveat: scored on the *generated* letter
  (chat-API setting, and the only option for the OSS HTTP endpoint), so absolute values may
  differ from log-prob leaderboard numbers — but the frontier-vs-OSS comparison is valid since
  both use identical scoring.
- **`run_multiturn_eval.py`** (SEPARATE + INDEPENDENT) — conversational hallucination eval.
  Replays `datasets/multiturn.jsonl` (5 scripted conversations: 1 plant + 3 probes + 7 filler
  each) turn-by-turn against a persistent session, so earlier facts genuinely sit in context.
  Each fact is probed at **gaps 1/5/10** (exchanges since it was planted); `gap = probe_index −
  plant_index`. Probes are scored by an **LLM judge** (own rubric, reuses the `judge.py`
  pattern). **Temporarily raises `config.MEMORY_WINDOW` to 40** (restored in `finally`) so a
  gap-10 fact stays in-window — this isolates real recall/attention degradation from the
  hard window cutoff. Metric: **hallucination rate (1 − mean score) vs gap**, per provider
  (plus overall + recall-vs-reasoning split). Outputs: `results/multiturn_responses.csv`,
  `results/multiturn_metrics.csv`, `results/charts/multiturn_hallucination.png`.
- **`safety_check.py`** (SEPARATE + INDEPENDENT) — safety/robustness smoke-test. Uses FastAPI
  `TestClient` (no live server/API key/network) to exercise the REAL `/chat` path with the
  provider client monkeypatched. Asserts: input blocklist blocks harmful / allows benign;
  `safety.safe_reply` normalizes blank/None/"null"; a provider failure → HTTP 200 +
  `PROVIDER_ERROR_FALLBACK` + `error:true` with NO assistant turn saved; empty model output →
  `EMPTY_REPLY_FALLBACK` (and IS saved); the global handler returns `GENERIC_ERROR` without
  leaking internals. Prints PASS/FAIL per check; exits non-zero on any failure (can gate a demo).

### Report (`report/`)
- **`build_report.py`** — builds a self-contained 1-page HTML (charts embedded as base64);
  `--pdf` renders via Playwright, else print-to-PDF from browser.
- **`evaluation_report.html` / `.pdf`** — the deliverable.

### Root
- `requirements.txt` — RUNTIME deps only (fastapi, uvicorn, openai, python-dotenv, httpx,
  pydantic) — this is all Railway installs. `gradio_client` removed with the FastAPI Space.
- `requirements-eval.txt` — eval/report deps (pandas, matplotlib, datasets), installed locally
  alongside requirements.txt; keeps the Railway build small and fast.
- `.github/workflows/keep-warm.yml` — cron (every 30 min) pings the Space `/health` (and the
  Railway app if the `APP_URL` repo variable is set) so neither sleeps; `workflow_dispatch`
  for a manual pre-demo warm-up.
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
pip install -r requirements.txt -r requirements-eval.txt   # runtime + eval/report deps

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
- Tools + long-term memory work on BOTH models via native tool-calling (frontier
  function-calling; OSS via Qwen's `<tool_call>` template + backend loop). OSS calls tools
  *less reliably* due to its 0.5B size — that's an expected, fair comparison datapoint. A
  larger OSS model would close most of the gap.
- Long-term memory uses deterministic append (no LLM reconciler — that would be frontier-only
  and break parity), so contradictory facts (e.g. a name change) accumulate. A parity-safe
  fix (deterministic key-overwrite, or model-issued correction) is future work.
- OSS token counts are approximated (~4 chars/token) — the Space returns no usage.
- OSS latency (free CPU Space, no GPU). Mitigations now in place: 4-bit GGUF via llama.cpp
  (replaces transformers bf16-on-CPU), plain-JSON FastAPI transport over a shared keep-alive
  httpx client (replaces the 1–3s/call gradio_client handshake), `max_new_tokens=128`
  (env-tunable via `OSS_MAX_NEW_TOKENS`), the Tools toggle (off → shorter prompt + no 2nd
  round-trip), model baked into the Docker image (cold start skips the download), and a
  keep-warm cron pinging `/health`. Remaining ceiling: raw 2-vCPU speed (a GPU/paid tier
  would be the real fix).
- Observability is JSONL → upgrade to a hosted dashboard (Langfuse / Phoenix).
- Responses are not streamed → add streaming for better UX.

---

## 9. Important gotchas (so I don't re-learn them)

- **`.env` is read only at startup** — restart `uvicorn` after editing it (`--reload` watches
  `.py`, not `.env`).
- **(Historical — Gradio era) reused `gradio_client` leaked Space session state** — was worked
  around with a fresh `Client` per call; made moot by the FastAPI Docker Space (plain JSON, no
  client-side sessions).
- **Don't pin one old package while others float** — caused cascading Gradio/HF build failures.
- **Logging must be best-effort** — never let it crash the request path (Railway write failure).
- **HF Space cold-starts** — warm both the Space and Railway before any demo/recording.
- See `TROUBLESHOOTING.md` for the full chronological list of issues and fixes.
