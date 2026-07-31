"""Self-check for the KV-backed memory + request log (run: python test_memory.py).

Covers what the Vercel move actually changed: memory and observability now
serialize through app.kv instead of living in module dicts / a JSONL file. Both
backends must behave identically, and a KV outage must degrade instead of raise.
"""
from app import kv, memory, observability


def _assert_memory_behaves(label: str):
    memory.reset("s1", "frontier")
    memory.reset("s1", "oss")

    memory.add_turn("s1", "frontier", "user", "hi")
    memory.add_turn("s1", "frontier", "assistant", "hello")
    memory.add_fact("s1", "frontier", "likes tea")
    memory.add_fact("s1", "frontier", "likes tea")  # de-dup

    assert memory.get_history("s1", "frontier") == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ], label
    assert memory.get_facts("s1", "frontier") == ["likes tea"], label

    # providers stay independent within one session
    assert memory.get_history("s1", "oss") == [], label
    assert memory.get_facts("s1", "oss") == [], label

    # facts reach the model via the system prompt
    assert "likes tea" in memory.build_messages("s1", "frontier", "q")[0]["content"], label

    memory.reset("s1", "frontier")
    assert memory.get_history("s1", "frontier") == [], label
    assert memory.get_facts("s1", "frontier") == [], label
    print(f"  ok: {label}")


def main():
    assert not kv.enabled, "run without KV env vars set"
    _assert_memory_behaves("in-process backend")

    # Same assertions against a fake Upstash, exercising the JSON round-trip.
    fake: dict[str, str] = {}
    kv.enabled = True
    kv.get, kv.set, kv.delete = fake.get, fake.__setitem__, lambda k: fake.pop(k, None)
    _assert_memory_behaves("kv backend")
    assert fake == {}, "reset should delete the key"

    # A KV outage degrades to empty memory rather than failing the request.
    def boom(*_a, **_k):
        raise RuntimeError("upstash down")

    kv.get = kv.set = boom
    assert memory.get_history("s1", "frontier") == []
    memory.add_turn("s1", "frontier", "user", "hi")  # must not raise
    print("  ok: kv outage degrades")

    # Log round-trip through the capped list.
    log: list[str] = []
    kv.rpush_capped = lambda k, v, cap: log.append(v)
    kv.lrange = lambda k: log
    observability.log_request({"provider": "oss", "latency_ms": 42})
    rows = observability.read_log()
    assert len(rows) == 1 and rows[0]["latency_ms"] == 42 and rows[0]["timestamp"]
    print("  ok: request log round-trip")

    print("all good")


if __name__ == "__main__":
    main()
