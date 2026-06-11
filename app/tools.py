"""Safe tool implementations + JSON-Schema definitions for native tool-calling.

ONE registry, used by BOTH providers so the comparison stays at parity:
  - frontier (OpenAI): TOOL_SCHEMAS passed as `tools=` to chat.completions; the
    model returns structured `tool_calls`.
  - oss (Qwen2.5 on HF Space): the SAME TOOL_SCHEMAS are passed to
    `apply_chat_template(tools=...)`; the model emits a `<tool_call>{...}</tool_call>`
    block which the backend parses.

Both paths execute through run_tool(), the single never-raises dispatch.
run_tool still accepts (session_id, provider) context for symmetry, though the
current utility tools don't use it.

Note: long-term memory is NOT a tool. It's handled deterministically by the
`/remember` and `/recall` slash commands (see app/commands.py), so saving facts
behaves identically and independently on both models instead of depending on a
model deciding to call a tool.
"""
import ast
import operator
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

# --- safe calculator (no eval) ---
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("non-numeric")
    if isinstance(node, ast.BinOp):
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


def calculator(expression: str, **_ctx) -> str:
    """Evaluate a basic arithmetic expression safely."""
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_safe_eval(tree.body))
    except Exception as e:
        return f"calculator error: {e}"


def current_datetime(timezone: str = "UTC", **_ctx) -> str:
    """Return current date and time in the given IANA timezone (default UTC)."""
    try:
        tz = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError):
        return f"current_datetime error: unknown timezone '{timezone}'"
    return datetime.now(tz).strftime(f"%Y-%m-%d %H:%M:%S {timezone}")


# --- unit conversion (lookup table, no network) ---
# Everything reduces to a base unit per category, then scales to the target.
_UNITS = {
    # length -> meters
    "m": ("length", 1.0), "meter": ("length", 1.0), "meters": ("length", 1.0),
    "km": ("length", 1000.0), "kilometer": ("length", 1000.0), "kilometers": ("length", 1000.0),
    "cm": ("length", 0.01), "mm": ("length", 0.001),
    "mi": ("length", 1609.344), "mile": ("length", 1609.344), "miles": ("length", 1609.344),
    "ft": ("length", 0.3048), "foot": ("length", 0.3048), "feet": ("length", 0.3048),
    "in": ("length", 0.0254), "inch": ("length", 0.0254), "inches": ("length", 0.0254),
    "yd": ("length", 0.9144), "yard": ("length", 0.9144), "yards": ("length", 0.9144),
    # weight -> grams
    "g": ("weight", 1.0), "gram": ("weight", 1.0), "grams": ("weight", 1.0),
    "kg": ("weight", 1000.0), "kilogram": ("weight", 1000.0), "kilograms": ("weight", 1000.0),
    "mg": ("weight", 0.001),
    "lb": ("weight", 453.59237), "lbs": ("weight", 453.59237),
    "pound": ("weight", 453.59237), "pounds": ("weight", 453.59237),
    "oz": ("weight", 28.349523125), "ounce": ("weight", 28.349523125), "ounces": ("weight", 28.349523125),
}
_TEMP = {"c", "celsius", "f", "fahrenheit", "k", "kelvin"}


def _to_celsius(value: float, unit: str) -> float:
    u = unit.lower()
    if u in ("c", "celsius"):
        return value
    if u in ("f", "fahrenheit"):
        return (value - 32.0) * 5.0 / 9.0
    return value - 273.15  # kelvin


def _from_celsius(value: float, unit: str) -> float:
    u = unit.lower()
    if u in ("c", "celsius"):
        return value
    if u in ("f", "fahrenheit"):
        return value * 9.0 / 5.0 + 32.0
    return value + 273.15  # kelvin


def unit_convert(value: float, from_unit: str, to_unit: str, **_ctx) -> str:
    """Convert a value between common length, weight, or temperature units."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return f"unit_convert error: '{value}' is not a number"
    f, t = from_unit.strip().lower(), to_unit.strip().lower()

    if f in _TEMP or t in _TEMP:
        if not (f in _TEMP and t in _TEMP):
            return "unit_convert error: cannot mix temperature with other units"
        result = _from_celsius(_to_celsius(value, f), t)
        return f"{value} {from_unit} = {round(result, 4)} {to_unit}"

    if f not in _UNITS or t not in _UNITS:
        unknown = f if f not in _UNITS else t
        return f"unit_convert error: unknown unit '{unknown}'"
    cat_f, scale_f = _UNITS[f]
    cat_t, scale_t = _UNITS[t]
    if cat_f != cat_t:
        return f"unit_convert error: cannot convert {cat_f} to {cat_t}"
    result = value * scale_f / scale_t
    return f"{value} {from_unit} = {round(result, 6)} {to_unit}"


# --- current weather (Open-Meteo: free, no API key) ---
_HTTP_TIMEOUT = 8  # seconds; bounds the network latency inside a tool round-trip
_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# WMO weather interpretation codes -> short text condition.
_WMO_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog",
    51: "light drizzle", 53: "drizzle", 55: "dense drizzle",
    56: "freezing drizzle", 57: "dense freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    66: "freezing rain", 67: "heavy freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "rain showers", 81: "moderate rain showers", 82: "violent rain showers",
    85: "snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with heavy hail",
}


def get_weather(city: str, *, session_id: str = "", provider: str = "", **_ctx) -> str:
    """Return the CURRENT weather for a city via Open-Meteo (no API key)."""
    city = (city or "").strip()
    if not city:
        return "weather error: no city provided"
    try:
        # 1) Geocode city -> coordinates.
        geo = httpx.get(_GEOCODE_URL, params={
            "name": city, "count": 1, "language": "en", "format": "json",
        }, timeout=_HTTP_TIMEOUT)
        geo.raise_for_status()
        results = (geo.json() or {}).get("results") or []
        if not results:
            return f"weather error: could not find city '{city}'"
        place = results[0]
        lat, lon = place["latitude"], place["longitude"]
        label = place.get("name", city)
        country = place.get("country", "")

        # 2) Current weather at those coordinates.
        wx = httpx.get(_FORECAST_URL, params={
            "latitude": lat, "longitude": lon,
            "current": "temperature_2m,apparent_temperature,weather_code",
        }, timeout=_HTTP_TIMEOUT)
        wx.raise_for_status()
        cur = (wx.json() or {}).get("current") or {}
        temp = cur.get("temperature_2m")
        feels = cur.get("apparent_temperature")
        condition = _WMO_CODES.get(cur.get("weather_code"), "unknown conditions")
        if temp is None:
            return "weather error: weather service returned no data"

        where = f"{label}, {country}" if country else label
        feels_part = f" (feels like {feels}°C)" if feels is not None else ""
        return f"{where}: {temp}°C, {condition}{feels_part}"
    except (httpx.HTTPError, KeyError, ValueError):
        return "weather error: weather service unavailable"


# registry: name -> callable
TOOLS = {
    "calculator": calculator,
    "current_datetime": current_datetime,
    "unit_convert": unit_convert,
    "get_weather": get_weather,
}


def run_tool(name: str, args: dict, *, session_id: str = "", provider: str = "") -> str:
    """Single execution path for both providers. Threads memory context in via
    kwargs; tools that don't need it ignore the extras. Never raises."""
    fn = TOOLS.get(name)
    if fn is None:
        return f"unknown tool {name}"
    try:
        return str(fn(**(args or {}), session_id=session_id, provider=provider))
    except TypeError as e:
        return f"{name} error: bad arguments ({e})"
    except Exception as e:
        return f"{name} error: {e}"


# JSON-Schema tool definitions — the OpenAI function-calling shape. The SAME list
# is sent to the HF Space and passed to apply_chat_template(tools=...); Qwen2.5
# accepts JSON-Schema tool defs, so the two providers never drift.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a basic arithmetic expression (e.g. '17*23').",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "current_datetime",
            "description": "Get the current date and time in an IANA timezone (default UTC).",
            "parameters": {
                "type": "object",
                "properties": {"timezone": {"type": "string", "description": "e.g. 'America/New_York'"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "unit_convert",
            "description": "Convert a value between common length, weight, or temperature units.",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number"},
                    "from_unit": {"type": "string", "description": "e.g. km, lb, celsius"},
                    "to_unit": {"type": "string", "description": "e.g. mi, kg, fahrenheit"},
                },
                "required": ["value", "from_unit", "to_unit"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the CURRENT weather (temperature, conditions) for a city or place.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "e.g. 'Paris' or 'Tokyo, Japan'"},
                },
                "required": ["city"],
            },
        },
    },
]
