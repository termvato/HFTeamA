"""Serve the Orchestra dashboard (API + screen) from the demo or runner process."""

from __future__ import annotations

import threading
from typing import Any

LOOPBACK = {"127.0.0.1", "localhost", "::1"}
PORT = 8050  # the one dashboard; the old Dash screen's port, so bookmarks keep working


def start(
    hub: Any,
    rig: Any,
    host: str = "127.0.0.1",
    port: int = PORT,
    token: str | None = None,
    block: bool = False,
) -> Any:
    """Serve the API and the screen at http://host:port/. Returns the uvicorn server.

    `block=False` runs it on a daemon thread (the runner keeps its own loop); `block=True` runs it
    here until Ctrl-C (the demo runner). Bound to this laptop by default. Listening on the network
    needs an operator token, because the API can start the run and stop arms.
    """
    try:
        import uvicorn

        from hfteama.orchestra.api import create_app
    except ImportError as e:
        raise SystemExit(
            f'the dashboard needs FastAPI and uvicorn: pip install -e ".[view]" ({e})'
        ) from e
    if host not in LOOPBACK and not token:
        raise SystemExit("--host other than 127.0.0.1 needs --token (commands can move arms)")
    config = uvicorn.Config(create_app(hub, rig, token), host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    if block:
        server.run()
    else:
        threading.Thread(target=server.run, name="orchestra", daemon=True).start()
    return server
