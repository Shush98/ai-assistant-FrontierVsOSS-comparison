"""Deterministic slash-commands for the assistant.

Long-term memory is handled here (NOT as a model tool) so it behaves identically
and independently on both providers: the user explicitly types the command, and
the backend stores/reads facts. The model never has to decide to call anything —
it just sees the saved facts injected into its system prompt (see
memory._system_content). Facts are scoped by (session_id, provider), so each
model's memory stays fully independent.

  /remember <text>  -> save a durable fact for this (session, provider)
  /recall           -> list saved facts for this (session, provider)

handle_command returns the reply string for a recognized command, or None if the
message is not a command (so normal chat proceeds).
"""
from app import memory


def handle_command(session_id: str, provider: str, message: str) -> str | None:
    text = (message or "").strip()
    if not text.startswith("/"):
        return None

    # Split into "/word" + remainder; command keyword is case-insensitive.
    head, _, rest = text.partition(" ")
    cmd = head.lower()
    payload = rest.strip()

    if cmd == "/remember":
        if not payload:
            return "Usage: /remember <something to remember>. Example: /remember my name is Sam."
        memory.add_fact(session_id, provider, payload)
        return f"Saved. I'll remember: {payload}"

    if cmd == "/recall":
        facts = memory.get_facts(session_id, provider)
        if not facts:
            return "Nothing saved yet. Use /remember <fact> to add something."
        return "Here's what I remember:\n- " + "\n- ".join(facts)

    return None  # unrecognized "/..." -> let it fall through to normal chat
