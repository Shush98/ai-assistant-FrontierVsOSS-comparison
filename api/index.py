"""Vercel entrypoint. The Python runtime serves the ASGI object named `app`,
so this just re-exports the FastAPI app; vercel.json rewrites every path here.
Local dev is unchanged: `uvicorn app.main:app --reload`.
"""
from app.main import app  # noqa: F401
