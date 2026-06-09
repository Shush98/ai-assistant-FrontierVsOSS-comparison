# Issues I Ran Into (and How I Fixed Them)

This is a running log of the real problems I hit while building this project, and how
I worked through each one. I kept it because the debugging is half the story — a lot of
these were environment/deployment issues that aren't obvious until you hit them.

---

## 1. My laptop couldn't run an LLM locally

**Problem:** My dev machine is an Intel i5-7200U with 8GB RAM (only ~2GB free) and no GPU.
Running any real LLM locally for inference would either swap to disk or be unusably slow
(seconds per token for anything above ~1.5B params).

**Fix:** I turned the constraint into the architecture. Instead of running the open-source
model locally, I deployed **Qwen2.5-0.5B-Instruct to a free Hugging Face Space** and call it
over HTTP — exactly like I call the OpenAI API. My laptop only runs the lightweight FastAPI
backend + frontend, which just orchestrate HTTP calls. This also satisfied the "deploy OSS
publicly" bonus.

---

## 2. Python 3.14 was too new for the ML/eval stack

**Problem:** My default Python was 3.14. A lot of libraries (especially `datasets` and the
ML ecosystem) don't have wheels for it yet, so installs would fail later.

**Fix:** I created a virtual environment pinned to **Python 3.10** (`py -3.10 -m venv venv`),
which I already had installed, and did all the work inside it. Kept the global Python untouched.

---

## 3. HF Space build error — `ModuleNotFoundError: No module named 'audioop'`

**Problem:** First Space build failed. The Space was running on **Python 3.13**, where the
`audioop` module was removed from the standard library. The old Gradio version I pinned pulled
in `pydub`, which needs `audioop` → crash on import.

**Fix:** Pinned the Space to **Python 3.11** in the `README.md` YAML header
(`python_version: "3.11"`), where `audioop` still exists in stdlib.

---

## 4. HF Space build error — `audioop-lts` not found

**Problem:** I'd added `audioop-lts` (a backport) to the Space requirements as a safety net,
but after switching the Space to Python 3.11 it failed: `audioop-lts` only has wheels for
Python 3.13+.

**Fix:** Removed `audioop-lts`. On 3.11 the real `audioop` is in stdlib, so the backport was
unnecessary (and now actively breaking the build).

---

## 5. HF Space runtime error — `cannot import name 'HfFolder' from 'huggingface_hub'`

**Problem:** I was pinning an old Gradio (4.44.1), but `transformers` pulled a newer
`huggingface_hub` that had removed `HfFolder`. The old Gradio expected it → import crash.
Basically I was fighting version drift by pinning one old package while everything else floated.

**Fix:** Stopped pinning the old version. Moved to a **modern Gradio (5.x)** built for the
current `huggingface_hub`, and matched the `sdk_version` in the README header. Build went green.

---

## 6. `gradio_client` couldn't connect — `401 Unauthorized` / `RepositoryNotFoundError`

**Problem:** When I tried to probe the Space API with `Client("username/space-name")`, it threw
a 401 and "Repository Not Found", even though the Space was public.

**Fix:** Connected using the **direct app URL** instead of the repo id:
`Client("https://shushantllm-qwen2-5-0-5b-assistant.hf.space")`. This hits the running app
directly and skips the `huggingface.co/api/spaces/...` lookup that was 401-ing.

---

## 7. OSS chat returned `500` / "Internal Server Error... is not valid JSON"

**Problem:** Switching to the OSS model in the UI threw a 500. The frontend just showed an
"is not valid JSON" error because it got an HTML error page instead of JSON.

**Fix:** The real error was in the server logs: `HF_SPACE_URL` was an **empty string** because
I'd started the server before saving the URL into `.env`. Since `.env` is only read at startup
(not by `--reload`), I had to fully restart `uvicorn` after editing it. Verified with
`python -c "from app import config; print(repr(config.HF_SPACE_URL))"`.

---

## 8. The two models seemed to "share" memory (the confusing one)

**Problem:** I told the OSS model my name, then the frontier model could answer it — and after
the context window slid, the behavior got weird and inconsistent between the two models. It
looked like memory was leaking between them.

**Root causes (there were two):**
1. **Memory was keyed only by `session_id`**, so both models read/wrote the *same* history.
2. **The HF Space kept its own hidden conversation history** (Gradio `ChatInterface` tracks
   `history` server-side), so the OSS model "remembered" things even after they slid out of my
   backend's window.

**Fix:**
1. Re-keyed memory by **`(session_id, provider)`** so each model has a fully independent
   conversation. Updated `/chat`, `/context`, and `/reset` to be provider-aware.
2. Made the Space **stateless** — rewrote it to take a single `message` and hold no history of
   its own (`gr.Interface` instead of `gr.ChatInterface`). Now my backend owns all memory.
3. Redesigned the UI into a **dual side-by-side layout** so each model is visibly its own
   independent assistant with its own input, `/context`, and reset.

This was the most important architectural correction — without it the comparison would have
been contaminated.

---

## 9. Frontier "forgot" a name that was visible in `/context`

**Problem:** Even when a name appeared in the `/context` output, the frontier model sometimes
said it didn't know my name.

**What was actually happening:** The name was present only in an *assistant* turn (the model's
own prior claim), not in a *user* turn where I actually stated it. The frontier model is
provenance-aware — it won't treat its own earlier output as a fact I provided, especially with a
system prompt telling it to say "I don't know" when unsure. Combined with a small window, this
produced the "forgetting." This turned out to be **correct, honest behavior**, not a bug, and
it became a nice data point for the eval (frontier is cautious; the tiny OSS model confabulates).

---

## 10. Cost in the cost/latency table showed `$0` for both models

**Problem:** The frontier model obviously has a (tiny) per-token cost, but the table rounded it
to `$0.000` because I'd rounded all columns to 3 decimals.

**Fix:** Formatted **only the cost columns** to 8 decimals (as fixed strings so they don't
collapse to 0 or scientific notation), kept latency/token columns at 1 decimal, and added a
readable `cost_per_1k_req_usd` projection. (First attempt used a global float format which
accidentally bloated *all* columns — fixed by formatting just the cost columns.)

---

## 11. TruthfulQA download failed — `Invalid HF URI ... Repository id must be 'namespace/name'`

**Problem:** `load_dataset("truthful_qa", "generation", ...)` failed because the un-namespaced
dataset id is deprecated and my `datasets`/`huggingface_hub` versions rejected it.

**Fix:** Used the canonical namespaced id: **`truthfulqa/truthful_qa`**. (The symlink warning on
Windows is harmless and can be ignored.)

---

## 12. Railway deploy — `500` on chat for BOTH models, but `/health` was fine

**Problem:** After deploying the backend + UI to Railway, every chat returned 500 — on both the
frontier and OSS panels — even though `/health` returned `ok` and there was no obvious error log.
Because both providers failed, I knew it had to be **shared code**, not a provider-specific issue.

**Fix:** The shared culprit was **observability logging** — `log_request` did file I/O with no
error handling, and writing `logs/` in the container was failing, which crashed every `/chat`.
I made logging **best-effort** (wrapped in try/except, falls back to stdout) so it can never
crash a request. I also added a global exception handler that prints the full traceback to the
platform log viewer and returns the real error message, so the next failure would be visible
instead of a bare 500. After redeploying, both models worked.

---

## 13. OSS model still "remembered" my name AFTER I clicked Reset

**Problem:** Even after resetting the OSS panel, the open-source model still knew my name. The
frontier model forgot correctly after reset, but OSS didn't. I thought I'd already made the
Space stateless back in issue #8, so this was confusing.

**How I tracked it down:** I checked everything systematically instead of guessing:
1. The local `deploy/hf_space/app.py` — stateless (single `message` arg, `gr.Interface`). ✅
2. Probed the **live** Space API — `predict(message)`, no `history` param. So the live Space
   matched the stateless file. ✅
3. `app/memory.py` — `reset()` correctly pops the `(session_id, provider)` key, and both
   providers share the same store, so memory wasn't the issue. ✅

Then I ran two **independent** `predict` calls against the live Space and reproduced it:
- With a **reused** `Client` object: call 2 ("what is my name?") knew "Zephyrina". ❌
- With a **fresh** `Client` per call: call 2 did NOT know the name. ✅

**Root cause:** The Space and my `chat` function are genuinely stateless, BUT `gradio_client`
keeps a **server-side session** when you reuse the same `Client` connection. My `LLMClient`
created the `Client` once in `__init__` and reused it for every OSS call, so Gradio's session
layer accumulated state across turns — and that state survived my backend's reset. The
stateless-looking API signature hid this; the session caching happens underneath it.

**Fix:** Create a **fresh `Client(HF_SPACE_URL)` per request** in `_chat_oss` instead of caching
one. No persistent session → no server-side state → the Space is truly stateless and reset works.
Verified end-to-end through the UI: after reset, OSS no longer knows the name. (Tradeoff: a small
per-call handshake; acceptable since correctness matters more and OSS is already slow. A faster
option would be raw `httpx` to the Space's REST endpoint.)

---

## Lessons I took away

- **Version pinning cuts both ways.** Pinning one old package while everything else floats
  caused more breakage than it prevented (issues #3–#5). Match versions across the stack.
- **`.env` is read at startup** — always restart the server after editing it (#7).
- **Never let logging/observability crash the request path** — make it best-effort (#12).
- **Design for the comparison from the start** — independent per-model state is essential, and
  hidden state (like the Space's own history) will silently contaminate results (#8).
- **Surface the real error early.** Adding a traceback-printing exception handler turned a
  guessing game into a one-line fix (#12).
- **A "stateless" API can still be stateful through the client.** A reused `gradio_client`
  connection keeps a hidden server-side session; reproduce with controlled experiments
  (fresh vs reused client) rather than trusting the API signature (#13).
