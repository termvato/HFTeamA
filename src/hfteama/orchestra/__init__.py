"""Orchestra: the dashboard, API and screen (docs/ORCHESTRA_DASHBOARD.md).

`python -m hfteama.demo` serves it on http://127.0.0.1:8050 (and `python -m hfteama.runner
--view`). Needs the `view` extra (FastAPI, uvicorn).
"""

from hfteama.orchestra.hub import StateHub

__all__ = ["StateHub"]
