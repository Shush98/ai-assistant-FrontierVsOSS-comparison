"""Vercel entrypoint. The Python runtime serves the ASGI object named `app`,
so this re-exports the FastAPI app; vercel.json rewrites every path here.
Local dev is unchanged: `uvicorn app.main:app --reload`.

The rewrite hands the function ITS OWN path, not the browser's, so the app sees
`/api/index/health` instead of `/health` (and plain `/api/index` for the UI).
vercel.json forwards the original path as a suffix and the middleware below
strips the prefix back off, so every route matches exactly as it does locally.
"""
from app.main import app

PREFIX = "/api/index"


class StripVercelPrefix:
    """Pure-ASGI middleware: remove the function's mount path from the request.

    No-op when the prefix isn't there, so this stays correct if Vercel ever
    starts passing the original path through (and off Vercel entirely).
    """

    def __init__(self, app, prefix: str = PREFIX):
        self.app = app
        self.prefix = prefix

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            path = scope.get("path", "")
            if path == self.prefix or path.startswith(self.prefix + "/"):
                scope = dict(scope, path=path[len(self.prefix):] or "/")
            # ponytail: drop once the deploy is confirmed — it's here so Vercel's
            # runtime log shows the real path if routing still misbehaves.
            print(f"[vercel] {scope.get('method')} raw={path} -> {scope.get('path')}")
        await self.app(scope, receive, send)


app.add_middleware(StripVercelPrefix)
