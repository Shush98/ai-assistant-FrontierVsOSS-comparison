from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import memory
from app.llm_client import LLMClient

app = FastAPI(title="AI Personal Assistant")

# reuse clients (don't rebuild per request)
_clients: dict[str, LLMClient] = {}


def get_client(provider: str) -> LLMClient:
    if provider not in _clients:
        _clients[provider] = LLMClient(provider)
    return _clients[provider]


class ChatRequest(BaseModel):
    session_id: str
    message: str
    provider: str = "frontier"


class ResetRequest(BaseModel):
    session_id: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    messages = memory.build_messages(req.session_id, req.message)
    out = get_client(req.provider).chat(messages)
    memory.add_turn(req.session_id, "user", req.message)
    memory.add_turn(req.session_id, "assistant", out["text"])
    return {
        "reply": out["text"],
        "provider": req.provider,
        "latency_ms": out["latency_ms"],
        "prompt_tokens": out["prompt_tokens"],
        "completion_tokens": out["completion_tokens"],
    }


@app.get("/context")
def context(session_id: str):
    return memory.get_context(session_id)


@app.post("/reset")
def reset(req: ResetRequest):
    memory.reset(req.session_id)
    return {"status": "reset", "session_id": req.session_id}


# serve frontend
@app.get("/")
def index():
    return FileResponse("frontend/index.html")