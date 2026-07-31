"""Shared state for stateless hosts (Vercel), with a local fallback.

Vercel runs each request in a fresh function instance with a read-only
filesystem, so the in-process dicts in app/memory.py and the JSONL file in
app/observability.py do not survive between requests — conversation memory and
the metrics graph would silently reset on every call.

When Upstash / Vercel KV credentials are present we route that state through
its Redis REST API using httpx (already a dependency — no new package). When
they aren't (local dev, Railway, tests) `enabled` is False and callers keep
their original in-process / file behavior, so nothing changes off Vercel.
"""
import httpx

from app import config

enabled = bool(config.KV_REST_API_URL and config.KV_REST_API_TOKEN)

_http = httpx.Client(
    timeout=5.0,
    headers={"Authorization": f"Bearer {config.KV_REST_API_TOKEN}"},
) if enabled else None


def _cmd(*args):
    """Run one Redis command over the REST API. Raises on transport/HTTP error;
    callers decide whether to degrade."""
    resp = _http.post(config.KV_REST_API_URL, json=[str(a) for a in args])
    resp.raise_for_status()
    return resp.json().get("result")


def get(key: str) -> str | None:
    return _cmd("GET", key)


def set(key: str, value: str) -> None:  # noqa: A001 - mirrors the Redis verb
    _cmd("SET", key, value)


def delete(key: str) -> None:
    _cmd("DEL", key)


def rpush_capped(key: str, value: str, cap: int) -> None:
    """Append to a list and trim it to the newest `cap` entries."""
    _cmd("RPUSH", key, value)
    _cmd("LTRIM", key, -cap, -1)


def lrange(key: str) -> list[str]:
    return _cmd("LRANGE", key, 0, -1) or []
