"""Safe-output + graceful-degradation helpers.

These keep the assistant from "screwing up while presenting": the user should
never see a blank reply, the literal string "None"/"null", or a raw internal
error. The logic is backend-side and provider-agnostic, so it behaves identically
for both models.
"""

# Shown when the model returns nothing usable (None / empty / "null").
EMPTY_REPLY_FALLBACK = (
    "I'm sorry — I couldn't generate a response to that. Could you try rephrasing?"
)

# Shown when a provider/API/Space call fails (down, timeout, bad key, rate-limit).
PROVIDER_ERROR_FALLBACK = (
    "⚠️ The assistant is temporarily unavailable right now. Please try again in a moment."
)

# Shown by the last-resort global exception handler (no internals leaked).
GENERIC_ERROR = "Something went wrong while handling your request. Please try again."

# Small models sometimes emit these literally instead of real content.
_EMPTY_SENTINELS = {"", "none", "null", "n/a", "undefined"}


def safe_reply(text) -> str:
    """Normalize a model reply so the UI never shows a blank/None/'null'.
    Returns the original text if it's a real answer, else EMPTY_REPLY_FALLBACK."""
    if text is None:
        return EMPTY_REPLY_FALLBACK
    cleaned = str(text).strip()
    if cleaned.lower() in _EMPTY_SENTINELS:
        return EMPTY_REPLY_FALLBACK
    return cleaned
