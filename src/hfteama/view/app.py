"""Live screen (backlog #17): countdown, estimate with what-if, the limit in words, plan vs actual.

The screen reads only the event log (hfteama.view.state.build). Buttons call an optional
`controls` object: Go starts the executor; the test panel adds, removes, fails or kills fake arms
(simulated arms are not part of the live product, only of development).
"""

from __future__ import annotations

from typing import Any, Protocol

import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html

from hfteama.clock import Clock
from hfteama.events import EventLog
from hfteama.view.state import Screen, build, clock_text

INK = "#17202a"
MUTED = "#5d6b78"
LINE = "#d5dde3"
ACCENT = "#1c6e8c"
OK = "#2e7d4f"
FAIL = "#c0392b"
TIMEOUT = "#d4880f"
RUNNING = "#4aa3c7"
PLAN = "rgba(28,110,140,0.18)"

CSS = f"""
body {{ margin: 0; background: #f4f6f7; color: {INK};
  font-family: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif; }}
.wrap {{ max-width: 1180px; margin: 0 auto; padding: 20px 24px 40px; display: grid; gap: 16px; }}
.top {{ display: flex; align-items: baseline; justify-content: space-between; gap: 12px;
  flex-wrap: wrap; }}
.top h1 {{ margin: 0; font-size: 22px; font-weight: 700; letter-spacing: .01em; }}
.pill {{ font: 600 12px/1 ui-monospace, monospace; text-transform: uppercase; letter-spacing: .08em;
  padding: 6px 10px; border-radius: 999px; border: 1.5px solid {ACCENT}; color: {ACCENT}; }}
.hero {{ display: grid; grid-template-columns: minmax(240px, 1fr) 2fr; gap: 24px;
  background: #fff; border: 1px solid {LINE}; border-radius: 12px; padding: 20px 24px; }}
.big {{ font: 700 72px/1 ui-monospace, "IBM Plex Mono", monospace;
  font-variant-numeric: tabular-nums; }}
.biglabel {{ color: {MUTED}; font-size: 14px; margin-top: 6px; }}
.limit {{ font-size: 20px; font-weight: 600; }}
.note {{ color: {MUTED}; font-size: 15px; margin-top: 6px; }}
.whatif {{ display: flex; gap: 10px; flex-wrap: wrap; margin-top: 14px; }}
.wi {{ border: 1px solid {LINE}; border-radius: 8px; padding: 8px 12px; min-width: 88px; }}
.wi b {{ display: block; font: 600 18px ui-monospace, monospace; }}
.wi span {{ color: {MUTED}; font-size: 12px; }}
.wi.cur {{ border-color: {ACCENT}; box-shadow: inset 0 0 0 1px {ACCENT}; }}
.row {{ display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }}
.go {{ font: 700 18px system-ui; padding: 12px 36px; border-radius: 10px; border: 0;
  background: {ACCENT}; color: #fff; cursor: pointer; }}
.go:disabled {{ background: #9fb7c2; cursor: default; }}
.progress {{ color: {MUTED}; font: 500 14px ui-monospace, monospace; }}
.panel {{ background: #fff; border: 1px solid {LINE}; border-radius: 12px; padding: 12px 16px; }}
.feed {{ margin: 0; padding-left: 18px; color: {INK}; font-size: 14px; line-height: 1.7; }}
.test button {{ margin-right: 6px; padding: 6px 10px; border-radius: 6px; border: 1px solid {LINE};
  background: #fff; cursor: pointer; }}
.test summary {{ cursor: pointer; color: {MUTED}; font-size: 13px; }}
@media (max-width: 760px) {{ .hero {{ grid-template-columns: 1fr; }} .big {{ font-size: 56px; }} }}
"""


class Controls(Protocol):
    def go(self) -> None: ...
    def started(self) -> bool: ...


def create_app(log: EventLog, controls: Any | None = None, clock: Clock | None = None) -> Dash:
    clock = clock or Clock()
    app = Dash(__name__, title="Plug and produce")
    app.index_string = app.index_string.replace("</head>", f"<style>{CSS}</style></head>")
    test_panel = html.Div()
    if controls is not None and hasattr(controls, "add"):
        test_panel = html.Details(
            className="panel test",
            children=[
                html.Summary("Arm controls (manual plug-in, stage fallback and tests)"),
                html.Div(
                    className="row",
                    style={"marginTop": "10px"},
                    children=[
                        html.Button("Plug in next arm", id="t-add"),
                        dcc.Dropdown(id="t-arm", placeholder="arm", style={"width": "160px"}),
                        html.Button("Remove (slow stop)", id="t-remove"),
                        html.Button("Fail its next step", id="t-fail"),
                        html.Button("Unplug (arm dies)", id="t-down"),
                        html.Span(id="t-msg", className="progress"),
                    ],
                ),
            ],
        )
    app.layout = html.Div(
        className="wrap",
        children=[
            html.Div(
                className="top",
                children=[html.H1("Plug and produce"), html.Span(id="phase", className="pill")],
            ),
            html.Div(
                className="hero",
                children=[
                    html.Div(
                        [
                            html.Div(id="big", className="big"),
                            html.Div(id="biglabel", className="biglabel"),
                        ]
                    ),
                    html.Div(
                        [
                            html.Div(id="limit", className="limit"),
                            html.Div(id="unblocked", className="note"),
                            html.Div(id="whatif", className="whatif"),
                        ]
                    ),
                ],
            ),
            html.Div(
                className="row",
                children=[
                    html.Button("Go", id="go", className="go"),
                    html.Span(id="progress", className="progress"),
                ],
            ),
            html.Div(
                className="panel",
                children=[dcc.Graph(id="gantt", config={"displayModeBar": False})],
            ),
            html.Div(className="panel", children=[html.Ul(id="feed", className="feed")]),
            test_panel,
            dcc.Interval(id="tick", interval=500),
        ],
    )

    @app.callback(
        Output("phase", "children"),
        Output("big", "children"),
        Output("biglabel", "children"),
        Output("limit", "children"),
        Output("unblocked", "children"),
        Output("whatif", "children"),
        Output("progress", "children"),
        Output("gantt", "figure"),
        Output("feed", "children"),
        Output("go", "disabled"),
        Input("tick", "n_intervals"),
    )
    def render(_: int | None):
        sc = build(log.read(), clock_now=clock.time())
        n = len(sc.arms)
        if sc.phase == "running":
            big, label = clock_text(sc.remaining), f"left with {n} arm{'s' * (n != 1)}"
        elif sc.phase == "finished":
            big, label = clock_text(sc.finish), "total time"
        else:
            big = clock_text(sc.estimate)
            label = f"estimated with {n} arm{'s' * (n != 1)}" if n else "plug in an arm"
        whatif = [
            html.Div(
                className="wi cur" if k == n else "wi",
                children=[html.B(clock_text(v)), html.Span(f"{k} arm{'s' * (k != 1)}")],
            )
            for k, v in sorted(sc.whatif.items())
        ]
        progress = f"{sc.done} / {sc.total} steps" if sc.total else ""
        if sc.blocked:
            progress += f" · {len(sc.blocked)} waiting for an arm that can reach them"
        feed = [html.Li(x) for x in sc.feed] or [html.Li("Waiting for Go")]
        go_disabled = sc.phase != "estimate" or n == 0
        return (
            {"empty": "no arms", "estimate": "ready", "running": "running", "finished": "done"}[
                sc.phase
            ],
            big,
            label,
            sc.limit,
            sc.unblocked_note,
            whatif,
            progress,
            gantt(sc),
            feed,
            go_disabled,
        )

    @app.callback(Output("go", "n_clicks"), Input("go", "n_clicks"), prevent_initial_call=True)
    def on_go(clicks: int | None):
        if controls is not None:
            controls.go()
        return clicks

    if controls is not None and hasattr(controls, "add"):

        @app.callback(
            Output("t-msg", "children"),
            Output("t-arm", "options"),
            Input("t-add", "n_clicks"),
            Input("t-remove", "n_clicks"),
            Input("t-fail", "n_clicks"),
            Input("t-down", "n_clicks"),
            Input("tick", "n_intervals"),
            State("t-arm", "value"),
        )
        def on_test(_a, _r, _f, _d, _t, arm: str | None):
            msg = ""
            trigger = ctx.triggered_id
            if trigger == "t-add":
                msg = f"added {controls.add()}"
            elif trigger in ("t-remove", "t-fail", "t-down"):
                if not arm:
                    msg = "pick an arm first"
                else:
                    {"t-remove": controls.remove, "t-fail": controls.fail, "t-down": controls.down}[
                        trigger
                    ](arm)
                    msg = f"{trigger[2:]} {arm}"
            return msg, controls.arm_ids()

    return app


def gantt(sc: Screen) -> go.Figure:
    lanes = sorted({b.arm for b in [*sc.plan, *sc.actual]} | set(sc.arms))
    labels = {a: f"{a} · {sc.arms.get(a, 'gone').upper()}" for a in lanes}
    fig = go.Figure()

    def bars(items, color, width, name, opacity=1.0):
        if not items:
            return
        fig.add_bar(
            y=[labels[b.arm] for b in items],
            x=[max(b.end - b.start, 0.05) for b in items],
            base=[b.start for b in items],
            orientation="h",
            marker={"color": color, "line": {"width": 0}},
            width=width,
            opacity=opacity,
            name=name,
            text=[b.task for b in items],
            textposition="inside",
            insidetextanchor="middle",
            hovertemplate="%{text}<br>%{base:.1f}s to %{x:.1f}s long<extra>" + name + "</extra>",
        )

    bars(sc.plan, PLAN, 0.8, "plan")
    bars([b for b in sc.actual if b.outcome == "ok"], OK, 0.45, "done")
    bars([b for b in sc.actual if b.kind == "running"], RUNNING, 0.45, "running")
    bars([b for b in sc.actual if b.outcome == "failed"], FAIL, 0.45, "failed")
    bars([b for b in sc.actual if b.outcome == "timeout"], TIMEOUT, 0.45, "timeout")
    if sc.phase == "running":
        fig.add_vline(x=sc.now, line={"color": INK, "width": 1.5})
    if sc.finish is not None:
        fig.add_vline(x=sc.finish, line={"color": ACCENT, "width": 1.5, "dash": "dash"})
    fig.update_layout(
        barmode="overlay",
        height=120 + 46 * max(1, len(lanes)),
        margin={"l": 10, "r": 10, "t": 10, "b": 30},
        paper_bgcolor="#fff",
        plot_bgcolor="#fff",
        font={"family": "IBM Plex Sans, system-ui, sans-serif", "color": INK, "size": 12},
        legend={"orientation": "h", "y": -0.25},
        xaxis={"title": "seconds", "gridcolor": LINE, "zeroline": False, "rangemode": "tozero"},
        yaxis={"autorange": "reversed"},
        uniformtext={"minsize": 8, "mode": "hide"},
    )
    return fig
