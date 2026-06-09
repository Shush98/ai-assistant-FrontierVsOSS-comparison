import traceback

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import memory, guardrails, observability, commands, safety
from app.llm_client import LLMClient

app = FastAPI(title="AI Personal Assistant")


@app.exception_handler(Exception)
async def all_errors(request: Request, exc: Exception):
    # Last-resort net. Log the full traceback for debugging, but return only a
    # generic message so internals (API errors, keys, paths, stack) never leak
    # to the client. The /chat path degrades gracefully before reaching here.
    traceback.print_exc()
    return JSONResponse(status_code=500, content={"detail": safety.GENERIC_ERROR})

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
    tools_enabled: bool = True  # UI toggle; gates tools for both models identically


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

    # Deterministic slash-commands (e.g. /remember, /recall). Handled in the
    # backend with NO model/Space call, so they behave identically and
    # independently for both providers. Scoped per (session, provider).
    cmd_reply = commands.handle_command(req.session_id, req.provider, req.message)
    if cmd_reply is not None:
        memory.add_turn(req.session_id, req.provider, "user", req.message)
        memory.add_turn(req.session_id, req.provider, "assistant", cmd_reply)
        observability.log_request({
            "provider": req.provider, "session_id": req.session_id,
            "latency_ms": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "cost_usd": 0.0, "guardrail": None, "blocked": False, "command": True,
        })
        return {
            "reply": cmd_reply, "provider": req.provider,
            "latency_ms": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "cost_usd": 0.0, "guardrail": None, "tool_used": None,
        }

    messages = memory.build_messages(req.session_id, req.provider, req.message)

    # Graceful degradation: if the provider/API/Space fails, don't 500 — log the
    # real error and return a friendly message (HTTP 200) the UI shows as a normal
    # bot bubble. This failure reply is NOT saved as a turn (it isn't a real answer).
    try:
        out = get_client(req.provider).chat(
            messages, session_id=req.session_id, provider=req.provider,
            tools_enabled=req.tools_enabled,
        )
    except Exception as exc:
        traceback.print_exc()
        observability.log_request({
            "provider": req.provider, "session_id": req.session_id,
            "latency_ms": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "cost_usd": 0.0, "guardrail": None, "blocked": False,
            "error": str(exc),
        })
        return {
            "reply": safety.PROVIDER_ERROR_FALLBACK, "provider": req.provider,
            "latency_ms": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "cost_usd": 0.0, "guardrail": None, "tool_used": None, "error": True,
        }

    # Safe-output: never present None/empty/"null" to the user (both providers).
    out["text"] = safety.safe_reply(out["text"])

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
        "server_ms": out.get("server_ms"),       # true OSS inference (None for frontier)
        "overhead_ms": out.get("overhead_ms"),   # transport overhead (None for frontier)
    })
    return {
        "reply": reply,
        "provider": req.provider,
        "latency_ms": out["latency_ms"],
        "prompt_tokens": out["prompt_tokens"],
        "completion_tokens": out["completion_tokens"],
        "server_ms": out.get("server_ms"),
        "overhead_ms": out.get("overhead_ms"),
        "cost_usd": cost,
        "guardrail": gate_out["reason"],
        "tool_used": out.get("tool_used"),
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