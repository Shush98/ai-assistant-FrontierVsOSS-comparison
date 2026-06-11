---
title: Qwen2.5 0.5B Instruct
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Qwen2.5-0.5B-Instruct (GGUF, FastAPI)

OSS personal assistant model. Deployed for AI-assistant comparison project.

FastAPI on a Docker Space (replaces the earlier Gradio app — no gradio_client
handshake/queue, plain JSON HTTP). Generation runs a 4-bit GGUF
(`Qwen/Qwen2.5-0.5B-Instruct-GGUF`, Q4_K_M) via llama.cpp — several times faster
than transformers on the free CPU tier. The prompt is still built with the HF
tokenizer's `apply_chat_template(tools=...)`, so Qwen's trained tool template
(and the `<tool_call>` output contract) is unchanged.

## API contract

`POST /chat` is **stateless**. JSON body:

```json
{
  "messages": [{"role": "user", "content": "..."}],
  "tools": [],
  "max_new_tokens": 128,
  "temperature": 0.7
}
```

- `messages` — the full conversation (`{role, content, ...}` dicts; assistant
  `tool_calls` and `role:"tool"` results included on tool round-trips).
- `tools` — tool schemas (OpenAI function-calling shape). Empty list → no tool
  block in the prompt → shorter prefill.
- `max_new_tokens` / `temperature` — generation params (temperature `0` = greedy,
  for reproducible evals). Defaults preserve the original behavior.

Returns `{"text": ..., "server_ms": ...}`: `text` is the **raw** generated text
(may include a `<tool_call>{"name":..,"arguments":{..}}</tool_call>` block);
`server_ms` is the true inference latency timed inside the Space, so the backend
can separate model compute from network overhead. The Space does not execute
tools or keep state — the backend parses tool calls, runs them, and calls
`/chat` again with the result appended.

`GET /health` → `{"status": "ok"}` (used by the keep-warm ping).
