"""Provider-agnostic LLM-judge helper, shared by every judge call site:
  - app/analysis.py        (live arena KPI cards)
  - eval/judge.py          (offline quality/safety report metrics)
  - eval/run_guardrail_eval.py (offline guardrail-refusal metric)

Why this exists: the frontier model under test is GPT (OpenAI). Judging it with another
GPT model risks same-family evaluator bias, so the judge DEFAULTS to Claude (a different
family). The provider is selectable via config.JUDGE_PROVIDER ("anthropic" | "openai") so
you can flip back to GPT for an apples-to-apples judge comparison.

Resilience: judge_json tries the CONFIGURED provider first, and on any failure (missing
key, API/model error, unparseable output) automatically FALLS BACK to the other provider so
judging keeps working. With the default config that means "Claude, falling back to GPT".

Both providers are asked for the SAME contract: return ONLY a compact JSON object. The
caller passes a system instruction + a user message and gets back a parsed dict. Clients are
lazy-initialised so importing this module never requires a key.
"""
import json

from app import config

_openai_client = None
_anthropic_client = None


def _openai():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _openai_client


def _anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        if not config.ANTHROPIC_API_KEY:
            raise RuntimeError(
                "JUDGE_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set. "
                "Add it to .env (and your deploy env), or set JUDGE_PROVIDER=openai."
            )
        import anthropic
        _anthropic_client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    return _anthropic_client


def as01(v) -> int:
    """Coerce a judge verdict ('score' or a 0/1 flag) to 0/1 WITHOUT raising.

    Judges are told to return 0/1, but models sometimes answer with a bool, "1"/"0",
    or words like "yes"/"true". The naive `int(v)` raises ValueError on those, and the
    callers' try/except then silently scores it 0 — which preferentially drops the
    POSITIVE verdicts. Coerce defensively so a valid verdict is never lost."""
    if isinstance(v, bool):
        return 1 if v else 0
    if isinstance(v, (int, float)):
        return 1 if v != 0 else 0
    if isinstance(v, str):
        return 1 if v.strip().lower() in {"1", "true", "yes", "y", "t"} else 0
    return 0


def judge_model_name() -> str:
    """The model id that will actually be used (for logging / report provenance)."""
    if config.JUDGE_PROVIDER == "openai":
        return config.OPENAI_JUDGE_MODEL
    return config.ANTHROPIC_JUDGE_MODEL


def _judge_openai(system: str, user: str) -> dict:
    resp = _openai().chat.completions.create(
        model=config.OPENAI_JUDGE_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def _judge_anthropic(system: str, user: str) -> dict:
    # Anthropic (Claude): system is a top-level param (not a message); there is no
    # response_format=json_object, so we instruct JSON in the prompt and parse the text.
    resp = _anthropic().messages.create(
        model=config.ANTHROPIC_JUDGE_MODEL,
        max_tokens=256,
        system=system + "\nReturn ONLY the JSON object — no prose, no code fences.",
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
    return json.loads(_strip_fences(text))


_PROVIDERS = {"openai": _judge_openai, "anthropic": _judge_anthropic}


def judge_json(system: str, user: str) -> dict:
    """Run one judge turn and return the parsed JSON dict.

    Tries the configured provider (config.JUDGE_PROVIDER) first; on ANY failure — missing
    key, API/model error, or unparseable output — falls back to the OTHER provider so
    judging keeps working (default config = Claude, falling back to GPT). Deterministic
    (temperature 0 on OpenAI; no sampling controls on Claude). Raises only if BOTH providers
    fail — callers decide how to degrade from there.

    `system` is the grading contract ("return ONLY compact JSON: {...}"); `user` is the
    case to grade."""
    primary = config.JUDGE_PROVIDER if config.JUDGE_PROVIDER in _PROVIDERS else "anthropic"
    fallback = "openai" if primary == "anthropic" else "anthropic"

    try:
        return _PROVIDERS[primary](system, user)
    except Exception as primary_err:
        try:
            return _PROVIDERS[fallback](system, user)
        except Exception as fallback_err:
            # Both judges are down — surface both causes so the failure is debuggable.
            raise RuntimeError(
                f"both judge providers failed: {primary}={primary_err!r}; "
                f"{fallback}={fallback_err!r}"
            ) from fallback_err


def _strip_fences(text: str) -> str:
    """Defensive: if the model wraps JSON in ```...``` fences, pull out the object."""
    t = text.strip()
    if t.startswith("```"):
        # drop the first fence line and any trailing fence
        t = t.split("\n", 1)[1] if "\n" in t else t
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    # last resort: slice from first { to last }
    if not t.lstrip().startswith("{"):
        i, j = t.find("{"), t.rfind("}")
        if i != -1 and j != -1 and j > i:
            t = t[i:j + 1]
    return t.strip()
