import json

from app import config, kv
from app.prompts import SYSTEM_PROMPT

# Memory is scoped by (session_id, provider) so each model keeps a fully
# INDEPENDENT conversation. Same session, two separate histories — this keeps
# the OSS-vs-frontier comparison clean (no cross-contamination). One record per
# key holds both tiers:
#   turns -> [{"role","content"}]  short-term, user/assistant only (no system)
#   facts -> [str]                 long-term, durable facts about the user
# Backed by app.kv: an in-process dict locally, Upstash/Vercel KV on serverless
# (where a per-request function instance has no memory of the last request).
_local: dict[str, str] = {}


def _key(session_id: str, provider: str) -> str:
    return f"mem:{session_id}:{provider}"


def _load(session_id: str, provider: str) -> dict:
    """Read the record. Best-effort: a KV outage degrades to an empty memory
    rather than failing the whole chat request."""
    key = _key(session_id, provider)
    try:
        raw = kv.get(key) if kv.enabled else _local.get(key)
    except Exception as e:
        print(f"[memory] load failed ({e}); continuing with empty memory")
        raw = None
    if not raw:
        return {"turns": [], "facts": []}
    try:
        rec = json.loads(raw)
    except json.JSONDecodeError:
        return {"turns": [], "facts": []}
    return {"turns": rec.get("turns", []), "facts": rec.get("facts", [])}


def _save(session_id: str, provider: str, rec: dict) -> None:
    # ponytail: read-modify-write, no locking. Fine for per-session chat (turns
    # are serialized by the user); add WATCH/Lua if sessions ever go concurrent.
    key = _key(session_id, provider)
    raw = json.dumps(rec)
    try:
        kv.set(key, raw) if kv.enabled else _local.__setitem__(key, raw)
    except Exception as e:
        print(f"[memory] save failed ({e}); this turn will not be remembered")


def get_history(session_id: str, provider: str) -> list[dict]:
    return _load(session_id, provider)["turns"]


def add_turn(session_id: str, provider: str, role: str, content: str) -> None:
    rec = _load(session_id, provider)
    rec["turns"].append({"role": role, "content": content})
    _save(session_id, provider, rec)


# --- long-term memory ---
def add_fact(session_id: str, provider: str, fact: str) -> None:
    """Append a fact (deterministic; de-dups exact repeats). Kept simple so it
    behaves identically for both providers — no model-driven reconciliation."""
    fact = (fact or "").strip()
    if not fact:
        return
    rec = _load(session_id, provider)
    if fact not in rec["facts"]:
        rec["facts"].append(fact)
        _save(session_id, provider, rec)


def get_facts(session_id: str, provider: str) -> list[str]:
    return _load(session_id, provider)["facts"]


def reset(session_id: str, provider: str) -> None:
    key = _key(session_id, provider)
    try:
        kv.delete(key) if kv.enabled else _local.pop(key, None)
    except Exception as e:
        print(f"[memory] reset failed ({e})")


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
