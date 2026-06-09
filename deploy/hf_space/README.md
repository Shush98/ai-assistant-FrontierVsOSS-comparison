---
title: Qwen2.5 0.5B Instruct
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# Qwen2.5-0.5B-Instruct

OSS personal assistant model. Deployed for AI-assistant comparison project.

## API contract

`/chat` is **stateless** and takes two JSON-string args:
1. `messages_json` — the full conversation as a JSON list of `{role, content, ...}` dicts.
2. `tools_json` — a JSON list of tool schemas (OpenAI function-calling shape).

It runs `apply_chat_template(messages, tools=...)` and returns
`{"text": ..., "server_ms": ...}`: `text` is the **raw** generated text (may include a
`<tool_call>{"name":..,"arguments":{..}}</tool_call>` block); `server_ms` is the true
inference latency timed inside the Space, so the backend can separate model compute from
`gradio_client`/network overhead. The Space does not execute tools or keep state — the
backend parses tool calls, runs them, and calls `/chat` again with the result appended.