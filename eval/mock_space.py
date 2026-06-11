"""Mock HF Space — same /chat contract as deploy/hf_space/app.py, but NO model.

Lets the backend's OSS path (httpx transport, generation params, tool loop,
server_ms/overhead_ms split) be verified locally without llama.cpp or the real
Space:

    # terminal 1 (project root):
    venv\\Scripts\\python.exe -m uvicorn eval.mock_space:app --port 7861

    # terminal 2:
    $env:HF_SPACE_URL = "http://127.0.0.1:7861"
    venv\\Scripts\\python.exe eval\\local_oss_smoke.py        # or uvicorn app.main:app

Behavior (deterministic):
  - role:"tool" result present in messages  -> answers using that result
    (simulates the model composing a final answer — 2nd round of the tool loop).
  - tools offered AND last user msg contains "calculate" -> emits a Qwen-style
    <tool_call> calculator block (1st round of the tool loop).
  - otherwise -> canned reply that echoes the received generation params, so
    param threading from config is visible.
"""
import time

from fastapi import FastAPI
from pydantic import BaseModel


class ChatBody(BaseModel):
    messages: list[dict]
    tools: list[dict] = []
    max_new_tokens: int = 128
    temperature: float = 0.7


app = FastAPI(title="mock space")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(body: ChatBody):
    t0 = time.time()
    last = body.messages[-1] if body.messages else {}
    tool_results = [m for m in body.messages if m.get("role") == "tool"]

    if tool_results:
        text = f"The result is {tool_results[-1].get('content')}."
    elif body.tools and "calculate" in str(last.get("content", "")).lower():
        text = '<tool_call>{"name": "calculator", "arguments": {"expression": "17*23"}}</tool_call>'
    else:
        text = (
            f"mock reply (tools={'on' if body.tools else 'off'}, "
            f"max_new_tokens={body.max_new_tokens}, temperature={body.temperature})"
        )
    return {"text": text, "server_ms": int((time.time() - t0) * 1000)}
