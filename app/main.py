import traceback

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import memory, guardrails, observability
from app.llm_client import LLMClient

app = FastAPI(title="AI Personal Assistant")


@app.exception_handler(Exception)
async def all_errors(request: Request, exc: Exception):
    # Print full traceback to stdout so it shows in the platform log viewer,
    # and return the message so the UI surfaces the real cause.
    traceback.print_exc()
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})

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
    provider: str = "frontier"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    # Guardrail 1: input filter (before the model).
    gate_in = guardrails.check_input(req.message)
    if gate_in["blocked"]:
        reply = guardrails.refusal_message()
        memory.add_turn(req.session_id, req.provider, "user", req.message)
        memory.add_turn(req.session_id, req.provider, "assistant", reply)
        observability.log_request({
            "provider": req.provider, "session_id": req.session_id,
            "latency_ms": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "cost_usd": 0.0, "guardrail": gate_in["reason"], "blocked": True,
        })
        return {
            "reply": reply, "provider": req.provider,
            "latency_ms": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "cost_usd": 0.0, "guardrail": gate_in["reason"],
        }

    messages = memory.build_messages(req.session_id, req.provider, req.message)
    out = get_client(req.provider).chat(messages)

    # Guardrail 2: output filter (after the model).
    gate_out = guardrails.check_output(out["text"])
    reply = guardrails.refusal_message() if gate_out["blocked"] else out["text"]

    memory.add_turn(req.session_id, req.provider, "user", req.message)
    memory.add_turn(req.session_id, req.provider, "assistant", reply)

    cost = observability.estimate_cost(
        req.provider, out["prompt_tokens"], out["completion_tokens"]
    )
    observability.log_request({
        "provider": req.provider,
        "session_id": req.session_id,
        "latency_ms": out["latency_ms"],
        "prompt_tokens": out["prompt_tokens"],
        "completion_tokens": out["completion_tokens"],
        "cost_usd": cost,
        "guardrail": gate_out["reason"],
        "blocked": False,
    })
    return {
        "reply": reply,
        "provider": req.provider,
        "latency_ms": out["latency_ms"],
        "prompt_tokens": out["prompt_tokens"],
        "completion_tokens": out["completion_tokens"],
        "cost_usd": cost,
        "guardrail": gate_out["reason"],
    }


@app.get("/context")
def context(session_id: str, provider: str = "frontier"):
    return memory.get_context(session_id, provider)


@app.post("/reset")
def reset(req: ResetRequest):
    memory.reset(req.session_id, req.provider)
    return {"status": "reset", "session_id": req.session_id, "provider": req.provider}


# serve frontend
@app.get("/")
def index():
    return FileResponse("frontend/index.html")