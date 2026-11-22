"""Orchestra API (docs/ORCHESTRA_DASHBOARD.md, section 5): REST + Server-Sent Events + commands.

Runs inside `python -m hfteama.demo` (or the runner with `--view`), next to the executor, and serves
the screen in `static/` at `/`. Reads go through `StateHub`; commands go through the demo's `Rig`
and are written to `commands.jsonl` next to the event log.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import secrets
import time
from collections.abc import AsyncIterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from hfteama.orchestra.hub import StateHub

STATIC = Path(__file__).parent / "static"
STATE_PERIOD_S = 0.25  # a state event at most this often while something changes
KEEPALIVE_S = 1.0  # and at least this often, so the screen can tell a live link from a dead one


def create_app(hub: StateHub, rig: Any, token: str | None = None) -> FastAPI:
    app = FastAPI(title="Orchestra", docs_url="/api/docs", openapi_url="/api/openapi.json")
    audit = hub.log.path.parent / "commands.jsonl"
    ids = itertools.count(1)
    real = hub.backend in ("real", "sorter")  # sorter = the real arms through Hardware's IK
    commands = {
        "go": True,
        "pause": True,
        "stop_all": True,
        "plug": True,
        "remove": True,
        # dropping an arm without parking it is a test: never offered on real arms
        "simulate_unplug": not real,
        "inject_failure": hub.backend == "fake",
    }

    def answer(
        name: str, arm: str | None, ok: bool, message: str, status: int = 200
    ) -> JSONResponse:
        cid = f"c{next(ids)}"
        with audit.open("a") as f:
            f.write(
                json.dumps(
                    {
                        "t": time.time(),
                        "id": cid,
                        "name": name,
                        "arm": arm,
                        "accepted": ok,
                        "message": message,
                    }
                )
                + "\n"
            )
        if ok:
            hub.note(message)
        return JSONResponse(
            {"accepted": ok, "message": message, "command_id": cid},
            status_code=status if not ok else 200,
        )

    def refuse(name: str, arm: str | None, message: str, status: int = 409) -> JSONResponse:
        return answer(name, arm, False, message, status)

    @app.middleware("http")
    async def operator_token(request: Request, call_next):
        if token and request.method == "POST" and request.url.path.startswith("/api/"):
            if not secrets.compare_digest(request.headers.get("X-Orchestra-Token", ""), token):
                return JSONResponse(
                    {"accepted": False, "message": "operator token missing or wrong"},
                    status_code=403,
                )
        return await call_next(request)

    # ------------------------------------------------------------------ read

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        thread = getattr(hub.ex, "_thread", None)
        return {
            "ok": True,
            "executor_alive": bool(thread and thread.is_alive()),
            "log_lines": len(hub.events()),
        }

    @app.get("/api/config")
    def config() -> dict[str, Any]:
        return hub.config(commands)

    @app.get("/api/state")
    def state() -> dict[str, Any]:
        return hub.state()

    @app.get("/api/tasks")
    def tasks() -> list[dict[str, Any]]:
        return hub.tasks()

    @app.get("/api/events")
    def events(after: int = 0) -> list[dict[str, Any]]:
        return [{"line": n, **asdict(e)} for n, e in hub.events(after)]

    @app.get("/api/stream")
    async def stream(request: Request) -> StreamingResponse:
        try:
            after = int(request.headers.get("last-event-id", "0"))
        except ValueError:
            after = 0

        async def gen() -> AsyncIterator[str]:
            nonlocal after
            yield f"event: hello\ndata: {json.dumps(hub.config(commands))}\n\n"
            last_state, last_sent = "", 0.0
            while not await request.is_disconnected():
                for n, e in hub.events(after):
                    after = n
                    yield f"id: {n}\nevent: event\ndata: {json.dumps(asdict(e))}\n\n"
                body = json.dumps(hub.state())
                now = time.monotonic()
                if body != last_state or now - last_sent >= KEEPALIVE_S:
                    yield f"event: state\ndata: {body}\n\n"
                    last_state, last_sent = body, now
                await asyncio.sleep(STATE_PERIOD_S)

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ------------------------------------------------------------------ commands

    async def body(request: Request) -> dict[str, Any]:
        try:
            data = await request.json()
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def joined() -> set[str]:
        return set(hub.ex.snapshot()["arms"])

    @app.post("/api/run/go")
    def go() -> JSONResponse:
        snap = hub.ex.snapshot()
        if snap["started"]:
            return refuse("go", None, "The run has already started")
        if not snap["arms"]:
            return refuse("go", None, "Plug in an arm first")
        rig.go()
        return answer("go", None, True, "Go")

    @app.post("/api/run/pause")
    def pause() -> JSONResponse:
        snap = hub.ex.snapshot()
        if not snap["started"] or hub.ex.finished:
            return refuse("pause", None, "Not running")
        hub.ex.pause()
        return answer("pause", None, True, "Paused: no new steps start; running steps finish")

    @app.post("/api/run/resume")
    def resume() -> JSONResponse:
        if not hub.ex.paused:
            return refuse("resume", None, "Not paused")
        hub.ex.resume()
        return answer("resume", None, True, "Resumed")

    def camera_feed() -> Any:
        try:
            from hfteama.perception.camera import active_feed
        except ImportError:  # no Pillow / camera extra: no feed
            return None
        return active_feed()

    @app.get("/api/camera")
    def camera_info() -> JSONResponse:
        cam = camera_feed()
        if cam is None:
            return JSONResponse({"available": False, "device": "", "age_s": None, "error": ""})
        return JSONResponse(
            {"available": True, "device": cam.device, "age_s": cam.age, "error": cam.error}
        )

    @app.get("/api/camera.jpg")
    def camera_jpg() -> Response:
        cam = camera_feed()
        data = cam.jpeg() if cam is not None else None
        if data is None:
            return JSONResponse({"error": "no camera frame"}, status_code=404)
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/camera/stream")
    async def camera_stream() -> Response:
        """MJPEG (multipart/x-mixed-replace): an <img> shows it live, about 4 frames a second."""
        cam = camera_feed()
        if cam is None:
            return JSONResponse({"error": "no camera"}, status_code=404)

        async def frames() -> AsyncIterator[bytes]:
            last = b""
            while True:
                data = await asyncio.to_thread(cam.jpeg)
                if data and data is not last:
                    last = data
                    yield (
                        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(data)).encode()
                        + b"\r\n\r\n"
                        + data
                        + b"\r\n"
                    )
                await asyncio.sleep(0.25)

        return StreamingResponse(
            frames(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/api/run/stop-all")
    async def stop_all(request: Request) -> JSONResponse:
        abort = bool((await body(request)).get("abort"))
        arms = sorted(joined())
        if not arms:
            return refuse("stop_all", None, "No arm is in the run")
        for a in arms:
            rig.remove(a, abort=abort)
        return answer("stop_all", None, True, f"Slow stop for {', '.join(arms)}")

    def known(arm: str) -> JSONResponse | None:
        if arm not in rig.known:
            return refuse("arm", arm, f"Unknown arm {arm}", 404)
        return None

    @app.post("/api/arms/{arm}/plug")
    def plug(arm: str) -> JSONResponse:
        if (bad := known(arm)) is not None:
            return bad
        if arm in joined():
            return refuse("plug", arm, f"{arm} is already in the run")
        if hub.watching and arm not in hub.watcher.present():
            return refuse(
                "plug", arm, f"{arm}: USB serial not seen. Plug its cable in; it joins by itself"
            )
        if not rig.plug(arm):
            why = dict(hub.watcher.errors).get(arm, "it could not start")
            return refuse("plug", arm, f"{arm} did not join: {why}")
        return answer("plug", arm, True, f"{arm} plugged in from the screen")

    @app.post("/api/arms/{arm}/remove")
    async def remove(arm: str, request: Request) -> JSONResponse:
        if (bad := known(arm)) is not None:
            return bad
        if arm not in joined():
            return refuse("remove", arm, f"{arm} is not in the run")
        abort = bool((await body(request)).get("abort"))
        rig.remove(arm, abort=abort)
        return answer(
            "remove", arm, True, f"{arm}: {'step aborted, ' if abort else ''}slow stop, then park"
        )

    @app.post("/api/arms/{arm}/simulate-unplug")
    def simulate_unplug(arm: str) -> JSONResponse:
        if not commands["simulate_unplug"]:
            return refuse(
                "simulate_unplug", arm, "Not offered on real arms: pull the cable or use Remove"
            )
        if (bad := known(arm)) is not None:
            return bad
        if arm not in joined():
            return refuse("simulate_unplug", arm, f"{arm} is not in the run")
        rig.down(arm)
        return answer("simulate_unplug", arm, True, f"{arm}: simulated unplug")

    @app.post("/api/arms/{arm}/inject-failure")
    def inject_failure(arm: str) -> JSONResponse:
        if not commands["inject_failure"]:
            return refuse("inject_failure", arm, "Only fake arms can be told to fail")
        if (bad := known(arm)) is not None:
            return bad
        if arm not in joined():
            return refuse("inject_failure", arm, f"{arm} is not in the run")
        rig.fail(arm)
        return answer("inject_failure", arm, True, f"{arm}: its next step will fail (grasp)")

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app
