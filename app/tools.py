"""Safe tool implementations + schemas for function calling."""
import ast
import operator
from datetime import datetime, timezone

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


def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression safely."""
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_safe_eval(tree.body))
    except Exception as e:
        return f"calculator error: {e}"


def current_datetime(_: str = "") -> str:
    """Return current UTC date and time."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# registry: name -> callable
TOOLS = {
    "calculator": calculator,
    "current_datetime": current_datetime,
}

# OpenAI function-calling schemas
OPENAI_TOOLS = [
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
            "description": "Get the current UTC date and time. No arguments needed.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]