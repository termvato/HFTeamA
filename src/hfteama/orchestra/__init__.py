"""Orchestra: operator dashboard API and screen (docs/ORCHESTRA_DASHBOARD.md).

`python -m hfteama.demo ... --orchestra` starts it on http://127.0.0.1:8060 next to the Dash
screen. Needs the `orchestra` extra (FastAPI, uvicorn).
"""

from hfteama.orchestra.hub import StateHub

__all__ = ["StateHub"]
