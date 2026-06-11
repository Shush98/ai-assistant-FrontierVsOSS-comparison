# Plan: Simplify the Codebase + Cut OSS Latency

> Status: plan only — no code changed yet.
> Constraint: OSS model must stay on a **free** Hugging Face Space (CPU basic, 2 vCPU). Backend stays on Railway.

**TL;DR:** The single biggest win for both goals is the same change — replace the Gradio Space with a plain FastAPI **Docker Space** (free tier supports this). That kills the 1–3s `gradio_client` handshake the `overhead_ms` metric already measures, and it deletes the entire defensive-normalization layer in `llm_client.py` plus the `gradio_client` dependency. The second biggest latency win is on the Space itself: `torch_dtype="auto"` loads Qwen in **bfloat16 on a CPU** (slow emulated path), and a 4-bit GGUF via llama.cpp would be ~3–6× faster generation. Realistically, the ~16.5s OSS average could come down to roughly **3–6s** with no functionality lost.

---

## Part 1 — OSS latency (HF Spaces free tier only)

The logs already split `server_ms` (true inference) from `overhead_ms` (transport), so every step below is verifiable before/after.

### L1. Replace the Gradio Space with a FastAPI Docker Space — biggest overhead win

- **Today:** every OSS call builds a fresh `Client(HF_SPACE_URL)` (`app/llm_client.py:167`) — a config-fetch + handshake + Gradio queue/SSE round-trip, **twice** when a tool fires. This is the documented 1–3s/call overhead (PROJECT_NOTES §8), and the fresh-client-per-call workaround exists only because of Gradio session leakage (TROUBLESHOOTING issue #13).
- **Plan:** rewrite `deploy/hf_space/app.py` as FastAPI: `POST /chat {messages, tools, max_new_tokens, temperature}` → `{text, server_ms}`, plus `GET /health`. Free Spaces support `sdk: docker` — needs a small Dockerfile. The backend's `_predict` becomes one `httpx.post()` on a **shared keep-alive client** (httpx is already a dependency).
- **Why this also simplifies:** a JSON API always returns a real dict, so `_normalize_space_result` (35 lines of "dict vs stringified-dict vs bare string vs list-wrapped"), the statefulness workaround, and `gradio_client==2.5.0` all disappear. A whole class of past bugs (TROUBLESHOOTING #6, #13, the Railway drift) becomes impossible.
- **Expected:** `overhead_ms` drops from ~1.5–3s+ per call to ~0.2–0.5s network RTT.

### L2. Speed up inference itself (`server_ms` — the dominant cost)

- **Quick check first:** `deploy/hf_space/app.py:9` uses `torch_dtype="auto"`, which picks up Qwen2.5's config dtype = **bfloat16**. Free-tier CPUs generally lack AMX, so bf16 math runs on slow fallback kernels — plain `float32` is often *faster* on CPU. One-line test, measure `server_ms`.
- **Real win:** switch generation to **llama-cpp-python + the official `Qwen/Qwen2.5-0.5B-Instruct-GGUF` (Q4_K_M, ~0.4GB)**. Typically 3–6× faster than transformers fp32 on 2 vCPU. Tool-calling parity is preserved by keeping `AutoTokenizer.apply_chat_template(tools=...)` to build the exact same Qwen prompt string, then feeding that text to llama.cpp for generation — the `<tool_call>` contract the backend parses doesn't change at all.
- Make `temperature` / `do_sample` / `max_new_tokens` backend-passed params instead of hardcoded — this simultaneously fixes **B1** (max-tokens parity break) and **B2** (non-deterministic evals) from `vulnerabilities to fix.md`. Batch all Space changes into one redeploy since redeploys are slow.

### L3. Reduce tokens processed per call

- Trim the verbose `TOOL_SCHEMAS` descriptions (`app/tools.py:214-268`) — they're prefilled on *every* OSS call, and CPU prefill isn't free. Modest but free.
- `MEMORY_WINDOW=10` and `max_new_tokens=128` are already sensible; leave them.

### L4. Kill cold starts

- Add a keep-warm ping: a GitHub Actions cron (or free UptimeRobot) hitting the Space's `/health` every ~30 min. Also fixes the C1 demo risk in `vulnerabilities to fix.md`.

### L5. Optional / later: streaming

- SSE streaming from the FastAPI Space through the backend would improve *perceived* latency a lot, but it touches the Space, backend, and frontend. Defer until L1–L4 land.

---

## Part 2 — Simplification (behavior-preserving, by file)

| File | Change | Est. reduction |
|---|---|---|
| `app/llm_client.py` | L1 deletes `_normalize_space_result` + the fresh-client workaround; replace the two regexes + 40-line `_loads_first_object` brace-balancer with `text.split("<tool_call>")` + stdlib `json.JSONDecoder().raw_decode()` (handles nested braces, trailing junk, *and* missing closing tag natively); factor the duplicated `completions.create` call in `_chat_frontier` | ~274 → ~150 lines |
| `app/main.py` | The guardrail-block, command, and error branches each build a near-identical zero-metric response + `log_request` dict. One small `_short_circuit(reply, ...)` helper covers all three | ~40 lines |
| `requirements.txt` | Drop `gradio_client` (after L1); move eval-only deps (`pandas`, `matplotlib`, `datasets`) to `requirements-eval.txt` — smaller Railway image → faster deploys and cold starts | — |
| `app/config.py` | `HF_TOKEN` is loaded but never used anywhere — remove | 1 line + .env noise |
| `frontend/index.html` | Drop the duplicated `REFUSAL_TEXT` string (line 216) — `data.guardrail` is already in the `/chat` response and identifies guardrail refusals; also drop the unused `opts` param on `send()` | small |
| `app/memory.py` | `get_context` re-assembles what `build_messages` builds — share one helper | minor |

**Deliberately not touching:** `safety.py` / `guardrails.py` (tiny, distinct purposes), `tools.py` registry layout (the schema/function split is fine), `commands.py`, `analysis.py`, the eval suite (all eval scripts go through `LLMClient.chat` with an unchanged signature, so they need **zero** edits — verified all five `run_*` scripts import it).

---

## Part 3 — Sequencing & verification

1. **Space first** (L1 + L2 + params, one redeploy): new FastAPI Docker Space, GGUF model, parameterized generation. Point the local backend at it.
2. **Backend transport** (`_predict` → httpx) — must land together with step 1 since the contract changes.
3. **Verify:** `python eval/safety_check.py` (offline, exercises the real `/chat` path) → `python eval/run_tool_eval.py` (full tool loop, both providers — proves `<tool_call>` parsing still works) → arena smoke test → compare `server_ms` / `overhead_ms` in `logs/requests.jsonl` before vs after for hard numbers.
4. **Then the pure refactors** (Part 2 table), re-running safety_check + tool eval after each.
5. **Keep-warm cron last.**

### Housekeeping notes

- `deploy/hf_space/` in the repo has drifted from what PROJECT_NOTES describes (README header says gradio `4.44.0`, notes say `5.9.1`; requirements differ) — the rewrite is a good moment to make the repo copy the single source of truth again.
- `vulnerabilities to fix.md` is already queued: fold its Tier-1 Space-touching items (B1 max-tokens parity, B2 greedy eval params) into the same Space redeploy rather than redeploying twice.
