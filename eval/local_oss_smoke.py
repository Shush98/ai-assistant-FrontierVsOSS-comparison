"""Smoke-test the OSS path of LLMClient against whatever HF_SPACE_URL points to.

Run against the local mock (deterministic — see eval/mock_space.py) to verify
the backend plumbing, or against the real Space after a deploy:

    $env:HF_SPACE_URL = "http://127.0.0.1:7861"   # mock; omit to use .env (real Space)
    venv\\Scripts\\python.exe eval\\local_oss_smoke.py

Checks: tools-off chat, the full tool round-trip (parse <tool_call> ->
run_tool -> second Space call -> final answer), and the server_ms/overhead_ms
latency split. NOTE: against the REAL Space the tool-call check depends on the
0.5B model actually emitting <tool_call> — a miss there is model reliability,
not a transport bug (re-run or test via the UI).
"""
import os
import sys

sys.path.insert(0, os.path.abspath("."))  # allow `from app...` from project root
os.environ.setdefault("OPENAI_API_KEY", "test")  # OSS path makes no OpenAI calls

from app import config
from app.llm_client import LLMClient

_failed = 0


def check(name, cond, extra=""):
    global _failed
    if not cond:
        _failed += 1
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   [{extra}]" if extra else ""))


print(f"Space under test: {config.HF_SPACE_URL}")
client = LLMClient("oss")

print("\n[tools off — single call, no tool block]")
out = client.chat(
    [{"role": "system", "content": "You are helpful."},
     {"role": "user", "content": "hi"}],
    session_id="smoke", provider="oss", tools_enabled=False,
)
check("got a reply", bool(out["text"].strip()), out["text"][:70])
check("no tool used", out["tool_used"] is None)
check("latency_ms > 0", out["latency_ms"] > 0, f"{out['latency_ms']}ms")
check("server_ms reported", out["server_ms"] is not None, f"{out['server_ms']}ms")

print("\n[tools on — full tool round-trip]")
out = client.chat(
    [{"role": "system", "content": "You are helpful."},
     {"role": "user", "content": "calculate 17*23"}],
    session_id="smoke", provider="oss", tools_enabled=True,
)
check("calculator was called", out["tool_used"] == "calculator", str(out["tool_used"]))
check("final reply contains 391", "391" in out["text"], out["text"][:70])
check("server/overhead split present", out["overhead_ms"] is not None,
      f"server={out['server_ms']}ms overhead={out['overhead_ms']}ms")

print(f"\n=== OSS smoke: {'ALL PASS' if _failed == 0 else str(_failed) + ' FAILED'} ===")
sys.exit(1 if _failed else 0)
