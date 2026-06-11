import re
from openai import OpenAI
from app import config

# Deterministic input blocklist — categories of clearly unsafe intent.
# Keep tight: catch obvious harm, avoid over-blocking benign queries.
_BLOCK_PATTERNS = [
    # Weapons/illicit synthesis: any making/acquiring intent + a harmful subject, not just
    # the "how to..." framing (catches "I want to make a bomb", "give me a recipe for napalm").
    r"\b(make|making|build|building|create|creating|synthesize|synthesise|synthesizing|"
    r"manufacture|manufacturing|produce|producing|cook|cooking|brew|recipe|instructions?) "
    r"(for |a |an |some |the )*"
    r"(bomb|explosive|ied|grenade|napalm|meth|methamphetamine|nerve agent|bioweapon|"
    r"biological weapon|chemical weapon|nuclear weapon|dirty bomb|pipe bomb|pathogen)\b",
    r"\b(kill|murder|poison|hurt)\s+(someone|a person|my)\b",
    r"\bhow (to|do i).*(hack|ddos|breach).*(without consent|someone'?s account)\b",
    r"\b(child|minor).*(sexual|porn|explicit)\b",
    r"\bhow (to|do i).*(suicide|kill myself|end my life)\b",
]

_REFUSAL = (
    "I can't help with that request. If you're in distress, please reach out to a "
    "local emergency line or a mental-health professional."
)

_compiled = [re.compile(p, re.IGNORECASE) for p in _BLOCK_PATTERNS]

# lazy OpenAI client for moderation (output check)
_mod_client = None


def _moderation_client():
    global _mod_client
    if _mod_client is None:
        _mod_client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _mod_client


def check_input(text: str) -> dict:
    """Return {'blocked': bool, 'reason': str|None}. Runs before the model."""
    if not config.GUARDRAILS_ENABLED:
        return {"blocked": False, "reason": None}
    for pat in _compiled:
        if pat.search(text):
            return {"blocked": True, "reason": "input_blocklist"}
    return {"blocked": False, "reason": None}


def check_output(text: str) -> dict:
    """Return {'blocked': bool, 'reason': str|None}. Runs after the model."""
    if not config.GUARDRAILS_ENABLED:
        return {"blocked": False, "reason": None}
    try:
        resp = _moderation_client().moderations.create(
            model="omni-moderation-latest", input=text
        )
        result = resp.results[0]
        if result.flagged:
            cats = [k for k, v in result.categories.__dict__.items() if v]
            return {"blocked": True, "reason": "moderation:" + ",".join(cats)}
    except Exception:
        # Fail-open on moderation API error (don't break chat); log upstream.
        pass
    return {"blocked": False, "reason": None}


def refusal_message() -> str:
    return _REFUSAL