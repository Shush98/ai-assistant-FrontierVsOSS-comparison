import json
import os
from datetime import datetime, timezone

from app import kv

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "requests.jsonl")

# Serverless hosts have a read-only, per-instance filesystem, so the JSONL file
# is replaced by a capped Redis list there (see app/kv.py). Cap keeps the KV free
# tier and the /metrics payload bounded; the chart only plots recent history.
LOG_KEY = "obs:requests"
LOG_CAP = 500

# OpenAI gpt-4o-mini pricing (USD per 1M tokens). Update if model changes.
PRICING = {
    "frontier": {"input": 0.15, "output": 0.60},  # gpt-4o-mini
    "oss": {"input": 0.0, "output": 0.0},          # self-hosted HF Space = no per-token cost
}


def estimate_cost(provider: str, prompt_tokens: int, completion_tokens: int) -> float:
    p = PRICING.get(provider, {"input": 0.0, "output": 0.0})
    return round(
        (prompt_tokens * p["input"] + completion_tokens * p["output"]) / 1_000_000, 8
    )


def log_request(record: dict) -> None:
    """Append one structured event as a JSON line. Never crash the request:
    logging is best-effort (some hosts have read-only/ephemeral filesystems)."""
    record["timestamp"] = datetime.now(timezone.utc).isoformat()
    line = json.dumps(record)
    try:
        if kv.enabled:
            kv.rpush_capped(LOG_KEY, line, LOG_CAP)
        else:
            os.makedirs(LOG_DIR, exist_ok=True)
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception as e:
        # Fall back to stdout (captured by the platform's log viewer).
        print(f"[observability] log write failed ({e}); record={line}")


def read_log() -> list[dict]:
    """Every logged event, oldest→newest, from whichever backend log_request
    used. Best-effort: a missing/partial/unreachable log yields what it can."""
    try:
        if kv.enabled:
            lines = kv.lrange(LOG_KEY)
        else:
            with open(LOG_FILE, encoding="utf-8") as f:
                lines = f.readlines()
    except Exception as e:
        print(f"[observability] log read failed ({e})")
        return []

    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out