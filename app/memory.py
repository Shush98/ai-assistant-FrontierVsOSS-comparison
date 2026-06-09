from app import config
from app.prompts import SYSTEM_PROMPT

# Short-term memory is scoped by (session_id, provider) so each model keeps a
# fully INDEPENDENT conversation. Same session, two separate histories — this
# keeps the OSS-vs-frontier comparison clean (no cross-contamination).
# key -> list of {"role","content"} (user/assistant turns only, no system)
_store: dict[tuple[str, str], list[dict]] = {}

# Long-term memory: durable facts about the user, same keying so the two models
# stay independent. In-process (lost on restart) — consistent with _store.
# key -> list of fact strings
_facts: dict[tuple[str, str], list[str]] = {}


def _key(session_id: str, provider: str) -> tuple[str, str]:
    return (session_id, provider)


def get_history(session_id: str, provider: str) -> list[dict]:
    return _store.get(_key(session_id, provider), [])


def add_turn(session_id: str, provider: str, role: str, content: str) -> None:
    _store.setdefault(_key(session_id, provider), []).append(
        {"role": role, "content": content}
    )


# --- long-term memory ---
def add_fact(session_id: str, provider: str, fact: str) -> None:
    """Append a fact (deterministic; de-dups exact repeats). Kept simple so it
    behaves identically for both providers — no model-driven reconciliation."""
    fact = (fact or "").strip()
    if not fact:
        return
    facts = _facts.setdefault(_key(session_id, provider), [])
    if fact not in facts:
        facts.append(fact)


def get_facts(session_id: str, provider: str) -> list[str]:
    return _facts.get(_key(session_id, provider), [])


def reset(session_id: str, provider: str) -> None:
    _store.pop(_key(session_id, provider), None)
    _facts.pop(_key(session_id, provider), None)  # clear long-term memory too


def _system_content(session_id: str, provider: str) -> str:
    """System prompt, plus a 'known facts' block when long-term memory exists.
    Injected for BOTH models, so memory works as plain context even when the
    model doesn't call recall_facts."""
    facts = get_facts(session_id, provider)
    if not facts:
        return SYSTEM_PROMPT
    facts_block = "\n".join(f"- {f}" for f in facts)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Known facts about the user (use them to personalize your replies):\n{facts_block}"
    )


def build_messages(session_id: str, provider: str, user_message: str) -> list[dict]:
    """System prompt (+ long-term facts) + last N turns + new user message."""
    history = get_history(session_id, provider)
    window = history[-config.MEMORY_WINDOW:]  # short-term memory
    system = _system_content(session_id, provider)
    return [{"role": "system", "content": system}, *window,
            {"role": "user", "content": user_message}]


def get_context(session_id: str, provider: str) -> dict:
    """For /context command: what the model actually sees (per provider)."""
    history = get_history(session_id, provider)
    window = history[-config.MEMORY_WINDOW:]
    system = _system_content(session_id, provider)
    messages = [{"role": "system", "content": system}, *window]
    approx_tokens = sum(len(m["content"]) for m in messages) // 4  # rough: ~4 chars/token
    return {
        "provider": provider,
        "system_prompt": SYSTEM_PROMPT,
        "long_term_facts": get_facts(session_id, provider),
        "memory_window": config.MEMORY_WINDOW,
        "turns_in_memory": len(window),
        "total_turns": len(history),
        "approx_tokens": approx_tokens,
        "messages": messages,
    }
