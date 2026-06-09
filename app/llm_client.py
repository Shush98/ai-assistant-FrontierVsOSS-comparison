import json
import re
import time
from openai import OpenAI
from gradio_client import Client
from app import config
from app import tools

# Qwen2.5 emits tool calls as Hermes-style XML: <tool_call>{"name":..,"arguments":{..}}</tool_call>
# Greedy {..} so NESTED argument objects (e.g. get_weather {"city":"Paris"}) aren't truncated at
# the first '}'; anchored by the closing tag.
_OSS_TOOL_RE = re.compile(r"<tool_call>\s*(\{.*\})\s*</tool_call>", re.DOTALL)
# Fallback for when the model omits the closing tag: grab the JSON object after <tool_call>.
_OSS_TOOL_OPEN_RE = re.compile(r"<tool_call>\s*(\{.*)", re.DOTALL)


class LLMClient:
    """One interface, two providers: 'frontier' (OpenAI) or 'oss' (HF Space).

    Both providers use NATIVE tool-calling over the SAME registry
    (tools.TOOL_SCHEMAS / tools.run_tool), driven by one backend loop, so the
    comparison stays at parity. The only difference is the provider API:
      - frontier: OpenAI function-calling (structured `tool_calls`).
      - oss: Qwen2.5's trained `tools=` template; the model emits a
        `<tool_call>` block that the backend parses (the Space stays a stateless
        text generator — it does NOT run tools).
    """

    def __init__(self, provider: str):
        if provider not in ("frontier", "oss"):
            raise ValueError(f"unknown provider: {provider}")
        self.provider = provider
        if provider == "frontier":
            self._openai = OpenAI(api_key=config.OPENAI_API_KEY)
        # OSS: do NOT cache a Client. A reused gradio_client connection keeps a
        # server-side session, which makes the Space accumulate state across
        # calls. We create a fresh, sessionless Client per request instead.

    def chat(self, messages: list[dict], session_id: str = "", provider: str = "",
             tools_enabled: bool = True) -> dict:
        """messages = [{'role':..., 'content':...}]. Returns normalized dict.
        session_id/provider are threaded into tool execution so memory tools
        can be scoped. tools_enabled gates tool-calling for BOTH providers
        identically (UI toggle): off = no tools offered → faster, simpler."""
        if self.provider == "frontier":
            return self._chat_frontier(messages, session_id, provider, tools_enabled)
        return self._chat_oss(messages, session_id, provider, tools_enabled)

    # --- frontier (OpenAI native function-calling) ---
    def _chat_frontier(self, messages: list[dict], session_id: str, provider: str,
                       tools_enabled: bool = True) -> dict:
        start = time.time()
        prompt_tok = completion_tok = 0
        tool_used = None

        # When tools are off, omit the `tools=` arg entirely → the model can't
        # request a tool, so it answers directly (and the tool branch is skipped).
        create_kwargs = {"tools": tools.TOOL_SCHEMAS} if tools_enabled else {}
        resp = self._openai.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=messages,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
            **create_kwargs,
        )
        prompt_tok += resp.usage.prompt_tokens
        completion_tok += resp.usage.completion_tokens
        msg = resp.choices[0].message

        # One tool round (bounds latency; matches the OSS path).
        if msg.tool_calls:
            messages = messages + [msg]
            for call in msg.tool_calls:
                args = json.loads(call.function.arguments or "{}")
                result = tools.run_tool(
                    call.function.name, args,
                    session_id=session_id, provider=provider,
                )
                tool_used = call.function.name
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": str(result),
                })
            resp = self._openai.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=messages,
                temperature=config.TEMPERATURE,
                max_tokens=config.MAX_TOKENS,
            )
            prompt_tok += resp.usage.prompt_tokens
            completion_tok += resp.usage.completion_tokens
            msg = resp.choices[0].message

        return {
            "text": msg.content,
            "latency_ms": int((time.time() - start) * 1000),
            "prompt_tokens": prompt_tok,
            "completion_tokens": completion_tok,
            "tool_used": tool_used,
            # OpenAI doesn't expose inference time separately from the API call,
            # so the model/overhead split is OSS-only.
            "server_ms": None,
            "overhead_ms": None,
        }

    # --- oss (Qwen2.5 native tool template; backend runs the loop) ---
    def _chat_oss(self, messages: list[dict], session_id: str, provider: str,
                  tools_enabled: bool = True) -> dict:
        start = time.time()
        tool_used = None
        server_ms_total = 0      # true inference time, summed across Space calls
        server_seen = False      # did the Space report server_ms at all?

        text, server_ms = self._predict(messages, tools_enabled)
        if server_ms is not None:
            server_ms_total += server_ms
            server_seen = True
        # When tools are off, no schemas were offered → don't parse or do a 2nd
        # round-trip; the single Space call IS the answer (faster).
        name, args = self._parse_tool_call(text) if tools_enabled else (None, {})

        # One tool round-trip. Fail-open: malformed/unknown/no tool-call → the
        # raw text IS the answer.
        if name and name in tools.TOOLS:
            result = tools.run_tool(name, args, session_id=session_id, provider=provider)
            tool_used = name
            messages = messages + [
                # Qwen tool-result convention: assistant tool_calls, then role "tool" with name.
                {"role": "assistant", "content": "",
                 "tool_calls": [{"type": "function",
                                 "function": {"name": name, "arguments": json.dumps(args)}}]},
                {"role": "tool", "name": name, "content": str(result)},
            ]
            text, server_ms = self._predict(messages, tools_enabled)
            if server_ms is not None:
                server_ms_total += server_ms
                server_seen = True

        latency_ms = int((time.time() - start) * 1000)
        # True model latency vs transport overhead (handshake + network + queue).
        # If the Space didn't report server_ms (old/mid-redeploy), leave both None.
        server_out = server_ms_total if server_seen else None
        overhead_ms = max(latency_ms - server_ms_total, 0) if server_seen else None

        # Space gives no token counts; approximate (~4 chars/token).
        prompt_tokens = sum(len(str(m.get("content", ""))) for m in messages) // 4
        completion_tokens = len(str(text)) // 4
        return {
            "text": str(text),
            "latency_ms": latency_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "tool_used": tool_used,
            "server_ms": server_out,
            "overhead_ms": overhead_ms,
        }

    def _predict(self, messages: list[dict], tools_enabled: bool = True):
        """Call the Space. Returns (text, server_ms). server_ms is the Space's
        self-reported inference time, or None if the Space returned a bare string
        (old / mid-redeploy) — defensive so the backend never breaks."""
        # Fresh Client per call => no persistent session => Space stays stateless.
        # When tools are off, send an empty list → the Space injects NO tool block
        # → shorter prompt → less CPU prefill.
        schemas = tools.TOOL_SCHEMAS if tools_enabled else []
        result = Client(config.HF_SPACE_URL).predict(
            json.dumps(messages),
            json.dumps(schemas),
            api_name="/chat",
        )
        return self._normalize_space_result(result)

    @staticmethod
    def _normalize_space_result(result):
        """Normalize whatever gradio_client hands back into (text, server_ms).

        The Space returns JSON {text, server_ms}, but different gradio_client
        versions deserialize it differently across environments (e.g. a real dict
        locally, but a STRINGIFIED dict on Railway). If we don't normalize, the
        whole `{"text": "...<tool_call>...", "server_ms": N}` wrapper ends up as
        the 'text' and the tool-call regex can't find the block → tools silently
        stop firing in that environment. Handle every plausible shape so OSS
        behaves identically everywhere."""
        # Gradio sometimes wraps a single output in a list/tuple.
        if isinstance(result, (list, tuple)):
            result = result[0] if result else ""

        if isinstance(result, dict):
            return str(result.get("text", "")), result.get("server_ms")

        if isinstance(result, str):
            s = result.strip()
            # The Railway failure case: the JSON dict came back as a string.
            if s.startswith("{"):
                try:
                    data = json.loads(s)
                    if isinstance(data, dict) and "text" in data:
                        return str(data.get("text", "")), data.get("server_ms")
                except (json.JSONDecodeError, ValueError):
                    pass  # not JSON → treat as plain text below
            # Old bare-string contract (pre-server_ms Space) or plain text.
            return result, None

        # Unknown shape — never crash; stringify and move on.
        print(f"[oss] unexpected space result type={type(result).__name__}")
        return str(result), None

    @staticmethod
    def _parse_tool_call(text: str):
        """Return (name, args_dict) from a Qwen <tool_call>{...}</tool_call> block,
        else (None, {}). Tolerant of nested-arg JSON and a missing closing tag;
        fail-open (any parse problem → no tool call)."""
        text = text or ""
        # Preferred: a properly closed tag. Try the full greedy span first, then
        # fall back to extracting the first balanced {...} object inside it.
        m = _OSS_TOOL_RE.search(text)
        if not m:
            m = _OSS_TOOL_OPEN_RE.search(text)  # closing tag missing
        if not m:
            return None, {}

        candidate = m.group(1)
        data = LLMClient._loads_first_object(candidate)
        if data is None:
            return None, {}
        name = data.get("name")
        args = data.get("arguments", {})
        if not isinstance(args, dict):
            args = {}
        return name, args

    @staticmethod
    def _loads_first_object(s: str):
        """Parse the first complete JSON object in `s` (handles trailing junk and
        nested braces via brace-balancing). Returns the dict or None."""
        s = s.strip()
        # Fast path: the whole thing is valid JSON.
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else None
        except (json.JSONDecodeError, ValueError):
            pass
        # Brace-balance to find the first complete {...}, ignoring braces in strings.
        depth = 0
        in_str = False
        esc = False
        start = s.find("{")
        if start == -1:
            return None
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(s[start:i + 1])
                        return obj if isinstance(obj, dict) else None
                    except (json.JSONDecodeError, ValueError):
                        return None
        return None
