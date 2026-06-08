from app import config
from app.prompts import SYSTEM_PROMPT

# session_id -> list of {"role","content"} (user/assistant turns only, no system)
_store: dict[str, list[dict]] = {}


def get_history(session_id: str) -> list[dict]:
    return _store.get(session_id, [])


def add_turn(session_id: str, role: str, content: str) -> None:
    _store.setdefault(session_id, []).append({"role": role, "content": content})


def reset(session_id: str) -> None:
    _store.pop(session_id, None)


def build_messages(session_id: str, user_message: str) -> list[dict]:
    """System prompt + last N turns (sliding window) + new user message."""
    history = get_history(session_id)
    window = history[-config.MEMORY_WINDOW:]  # short-term memory
    return [{"role": "system", "content": SYSTEM_PROMPT}, *window,
            {"role": "user", "content": user_message}]


def get_context(session_id: str) -> dict:
    """For /context command: what the model actually sees."""
    history = get_history(session_id)
    window = history[-config.MEMORY_WINDOW:]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *window]
    approx_tokens = sum(len(m["content"]) for m in messages) // 4  # rough: ~4 chars/token
    return {
        "system_prompt": SYSTEM_PROMPT,
        "memory_window": config.MEMORY_WINDOW,
        "turns_in_memory": len(window),
        "total_turns": len(history),
        "approx_tokens": approx_tokens,
        "messages": messages,
    }