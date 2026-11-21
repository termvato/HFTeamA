"""Start the Orchestra API on a background thread of the demo process."""

from __future__ import annotations

import threading
from typing import Any

LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def start(
    hub: Any, rig: Any, host: str = "127.0.0.1", port: int = 8060, token: str | None = None
) -> Any:
    """Serve the API and the screen at http://host:port/. Returns the uvicorn server.

    Bound to this laptop by default. Listening on the network needs an operator token, because
    the API can start the run and stop arms.
    """
    try:
        import uvicorn

        from hfteama.orchestra.api import create_app
    except ImportError as e:
        raise SystemExit(
            f'--orchestra needs FastAPI and uvicorn: pip install -e ".[orchestra]" ({e})'
        ) from e
    if host not in LOOPBACK and not token:
        raise SystemExit("--orchestra-host other than 127.0.0.1 needs --orchestra-token")
    config = uvicorn.Config(create_app(hub, rig, token), host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, name="orchestra", daemon=True).start()
    return server
