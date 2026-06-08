import json
import os
from datetime import datetime, timezone

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "requests.jsonl")

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
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        # Fall back to stdout (captured by the platform's log viewer).
        print(f"[observability] file log failed ({e}); record={json.dumps(record)}")