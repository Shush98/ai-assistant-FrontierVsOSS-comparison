from app import config
from app.prompts import SYSTEM_PROMPT

# Memory is scoped by (session_id, provider) so each model keeps a fully
# INDEPENDENT conversation. Same session, two separate histories — this keeps
# the OSS-vs-frontier comparison clean (no cross-contamination).
# key -> list of {"role","content"} (user/assistant turns only, no system)
_store: dict[tuple[str, str], list[dict]] = {}


def _key(session_id: str, provider: str) -> tuple[str, str]:
    return (session_id, provider)


def get_history(session_id: str, provider: str) -> list[dict]:
    return _store.get(_key(session_id, provider), [])


def add_turn(session_id: str, provider: str, role: str, content: str) -> None:
    _store.setdefault(_key(session_id, provider), []).append(
        {"role": role, "content": content}
    )


def reset(session_id: str, provider: str) -> None:
    _store.pop(_key(session_id, provider), None)


def build_messages(session_id: str, provider: str, user_message: str) -> list[dict]:
    """System prompt + last N turns (sliding window) + new user message."""
    history = get_history(session_id, provider)
    window = history[-config.MEMORY_WINDOW:]  # short-term memory
    return [{"role": "system", "content": SYSTEM_PROMPT}, *window,
            {"role": "user", "content": user_message}]


def get_context(session_id: str, provider: str) -> dict:
    """For /context command: what the model actually sees (per provider)."""
    history = get_history(session_id, provider)
    window = history[-config.MEMORY_WINDOW:]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *window]
    approx_tokens = sum(len(m["content"]) for m in messages) // 4  # rough: ~4 chars/token
    return {
        "provider": provider,
        "system_prompt": SYSTEM_PROMPT,
        "memory_window": config.MEMORY_WINDOW,
        "turns_in_memory": len(window),
        "total_turns": len(history),
        "approx_tokens": approx_tokens,
        "messages": messages,
    }