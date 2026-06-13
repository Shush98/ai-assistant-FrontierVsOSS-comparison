"""STATELESS OSS inference endpoint — FastAPI on a Docker Space.

Replaces the earlier Gradio app. Same role, faster transport + inference:
  - Plain JSON HTTP (POST /chat) instead of gradio_client's handshake + queue/SSE,
    so the backend calls it with one keep-alive httpx POST (~0.2s overhead vs ~1-3s).
  - Generation runs on a 4-bit GGUF (llama.cpp) instead of transformers bf16-on-CPU,
    which is several times faster on the free 2-vCPU tier.

Tool-calling parity is unchanged: the prompt is still built with the HF tokenizer's
`apply_chat_template(tools=...)` (Qwen2.5's trained tool template), so the model
emits the same `<tool_call>{...}</tool_call>` blocks the backend already parses.
llama.cpp only replaces the *generation* backend, not the prompt format.

The Space holds NO state and runs NO tools — the backend owns the conversation
and the tool loop, and sends the full message list every call.
"""
import glob
import os
import threading
import time

from fastapi import FastAPI
from huggingface_hub import snapshot_download
from llama_cpp import Llama
from pydantic import BaseModel
from transformers import AutoTokenizer

TOKENIZER_REPO = "Qwen/Qwen2.5-0.5B-Instruct"        # template source (tokenizer only, no torch)
GGUF_REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"         # official Qwen quantized weights
GGUF_PATTERN = "*q4_k_m.gguf"                         # 4-bit, ~0.4GB

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_REPO)

# Both downloads are baked into the Docker image at build time (see Dockerfile),
# so these resolve from the local cache and cold starts skip the ~400MB pull.
_model_dir = snapshot_download(GGUF_REPO, allow_patterns=[GGUF_PATTERN])
_gguf_path = glob.glob(os.path.join(_model_dir, GGUF_PATTERN))[0]

llm = Llama(
    model_path=_gguf_path,
    n_ctx=4096,        # system + 10-turn window + tool schemas fit comfortably
    # Free CPU tier = 2 vCPUs. DON'T use os.cpu_count(): on HF's shared hosts it
    # reports the host's logical cores (8-32), not this container's cgroup quota,
    # so llama.cpp oversubscribes 2 usable cores -> thread contention -> SLOWER.
    # Pin to the actual vCPU count (env-overridable if you move to a bigger tier).
    n_threads=int(os.getenv("OSS_THREADS", "2")),
    verbose=False,
)
# llama.cpp contexts are not thread-safe; serialize generation (FastAPI runs sync
# endpoints in a threadpool, so concurrent requests are possible).
_gen_lock = threading.Lock()

app = FastAPI(title="Qwen2.5-0.5B-Instruct (GGUF)")


class ChatBody(BaseModel):
    messages: list[dict]
    tools: list[dict] = []
    # Defaults preserve the old hardcoded behavior; the backend can override both
    # (e.g. greedy temp=0 for reproducible evals, longer caps for parity tests).
    max_new_tokens: int = 128
    temperature: float = 0.7


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(body: ChatBody):
    """Returns {"text", "server_ms"}: `text` is the RAW generated text (may contain
    a <tool_call> block); `server_ms` is true inference time measured here, so the
    backend can split model compute from network overhead."""
    t0 = time.time()
    prompt = tokenizer.apply_chat_template(
        body.messages,
        tools=body.tools or None,   # empty list -> no tool block -> shorter prefill
        add_generation_prompt=True,
        tokenize=False,
    )
    with _gen_lock:
        out = llm(
            prompt,
            max_tokens=body.max_new_tokens,
            temperature=body.temperature,   # 0 -> greedy in llama.cpp
            stop=["<|im_end|>"],
        )
    text = out["choices"][0]["text"]
    return {"text": text, "server_ms": int((time.time() - t0) * 1000)}
