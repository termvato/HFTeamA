"""Scaling benchmark (backlog #15): finish time against the number of arms.

For each task (sort12, assemble3) and N = 1..8 arms it reports:
- planned finish (HEFT) and the lower bound with its binding term (arms, order or a shared zone);
- the ideal line, one-arm time / N;
- measured finish from real executor runs on fake arms (noise, a few trials, N = 1..4 only);
- speed-up = one-arm time / finish, planned and measured.

Outputs: a Markdown table on stdout, scaling.csv, scaling.html (self-contained interactive Plotly
chart) and the executor event logs, all under --out (default runs/scaling/, git-ignored).

Run: python scripts/scaling_benchmark.py [--trials 3] [--measure-max 4] [--no-measure]
"""

from __future__ import annotations

import argparse
import csv
import html
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from hfteama.arms.fake import FakeArm
from hfteama.clock import Clock
from hfteama.events import EventLog
from hfteama.executor import Executor
from hfteama.scheduler.problem import Problem
from hfteama.scheduler.solve import solve
from hfteama.taskgraph import TaskGraph, load

ROOT = Path(__file__).resolve().parent.parent
TASKS = {
    "sort12": ROOT / "tasks" / "sort" / "sort12.json",
    "assemble3": ROOT / "tasks" / "assemble" / "assemble3.json",
}

# Reference palette, light mode (series 1..3) plus a neutral for the ideal line.
COLORS = {"measured": "#2a78d6", "planned": "#eb6834", "bound": "#1baf7a", "ideal": "#8a8984"}


@dataclass
class Row:
    task: str
    n: int
    planned: float
    bound: float
    binding: str
    ideal: float
    planned_speedup: float
    measured: list[float] = field(default_factory=list)
    measured_speedup: float | None = None

    @property
    def mean(self) -> float | None:
        return statistics.fmean(self.measured) if self.measured else None


def arm_ids(n: int) -> list[str]:
    return [f"arm_{i}" for i in range(1, n + 1)]


def binding_text(binding: str) -> str:
    """Plain words for the limit, as shown on the chart."""
    if binding == "arms":
        return "arm count (more arms help)"
    if binding == "order":
        return "step order is the bottleneck"
    if binding.startswith("resource:"):
        return binding.removeprefix("resource:").replace("_", " ") + " is the bottleneck"
    return binding or "-"


def plan_rows(name: str, graph: TaskGraph, counts: range | list[int]) -> list[Row]:
    """Planning part only (fast, no executor): HEFT finish, bound and binding per arm count."""
    rows: list[Row] = []
    one = solve(Problem(graph, arm_ids(1)), "heft").makespan
    for n in counts:
        plan = solve(Problem(graph, arm_ids(n)), "heft")
        rows.append(
            Row(
                task=name,
                n=n,
                planned=plan.makespan,
                bound=plan.lower_bound,
                binding=plan.binding,
                ideal=one / n,
                planned_speedup=one / plan.makespan,
            )
        )
    return rows


def run_once(graph: TaskGraph, n: int, trial: int, scale: float, noise: float, log: Path) -> float:
    """One batch on n fake arms through the real executor. Returns the finish in run seconds."""
    log.unlink(missing_ok=True)
    clock = Clock(scale=scale)
    events = EventLog(log)
    ex = Executor(graph, events, clock=clock)
    for i, a in enumerate(arm_ids(n)):
        ex.add_arm(
            FakeArm(
                a,
                clock=clock,
                duration_fn=lambda t, a=a: t.duration_on(a),
                noise=noise,
                seed=1000 * trial + i,
            )
        )
    budget = 3.0 * sum(t.duration_on("arm_1") for t in graph.tasks.values()) / scale + 10.0
    ex.start()
    if not ex.wait(budget):
        ex.stop()
        raise RuntimeError(f"{graph.name}, {n} arms, trial {trial}: not done in {budget:.0f} s")
    done = [e for e in events.read() if e.kind == "replan" and e.data.get("reason") == "finished"]
    return float(done[-1].data["finish"])


def measure(rows: list[Row], graph: TaskGraph, args: argparse.Namespace, logs: Path) -> None:
    for r in rows:
        if r.n > args.measure_max:
            continue
        for k in range(args.trials):
            log = logs / f"{r.task}_n{r.n}_t{k + 1}.jsonl"
            r.measured.append(run_once(graph, r.n, k + 1, args.scale, args.noise, log))
    one = next((r.mean for r in rows if r.n == 1 and r.measured), None)
    for r in rows:
        if one is not None and r.mean:
            r.measured_speedup = one / r.mean


def _f(x: float | None, fmt: str = "{:.1f}") -> str:
    return "-" if x is None else fmt.format(x)


def markdown(rows: list[Row]) -> str:
    head = (
        "| task | arms | planned (s) | lower bound (s) | limit | ideal 1/N (s) "
        "| measured mean (s) | measured min-max (s) | planned speed-up | measured speed-up |"
    )
    out = [head, "|" + "---|" * 10]
    for r in rows:
        spread = f"{min(r.measured):.1f}-{max(r.measured):.1f}" if r.measured else "-"
        out.append(
            f"| {r.task} | {r.n} | {r.planned:.1f} | {r.bound:.1f} | {r.binding} | {r.ideal:.1f} "
            f"| {_f(r.mean)} | {spread} | {r.planned_speedup:.2f}x "
            f"| {_f(r.measured_speedup, '{:.2f}x')} |"
        )
    return "\n".join(out)


def write_csv(rows: list[Row], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "task",
                "arms",
                "planned_s",
                "lower_bound_s",
                "binding",
                "ideal_s",
                "measured_mean_s",
                "measured_min_s",
                "measured_max_s",
                "trials",
                "planned_speedup",
                "measured_speedup",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.task,
                    r.n,
                    f"{r.planned:.3f}",
                    f"{r.bound:.3f}",
                    r.binding,
                    f"{r.ideal:.3f}",
                    _f(r.mean, "{:.3f}") if r.measured else "",
                    f"{min(r.measured):.3f}" if r.measured else "",
                    f"{max(r.measured):.3f}" if r.measured else "",
                    len(r.measured),
                    f"{r.planned_speedup:.4f}",
                    "" if r.measured_speedup is None else f"{r.measured_speedup:.4f}",
                ]
            )


def write_html(by_task: dict[str, list[Row]], path: Path, meta: str) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    names = list(by_task)
    fig = make_subplots(rows=1, cols=len(names), subplot_titles=names, horizontal_spacing=0.08)
    for col, name in enumerate(names, start=1):
        rows = by_task[name]
        ns = [r.n for r in rows]
        first = col == 1

        def line(y, key, label, dash=None, hover=None, _first=first, _ns=ns, _col=col):
            fig.add_trace(
                go.Scatter(
                    x=_ns,
                    y=y,
                    name=label,
                    legendgroup=key,
                    showlegend=_first,
                    mode="lines+markers",
                    line={"color": COLORS[key], "width": 2, "dash": dash},
                    marker={"size": 8 if key != "ideal" else 6},
                    customdata=hover,
                    hovertemplate=f"{label}: %{{y:.1f}} s"
                    + ("<br>%{customdata}" if hover else "")
                    + "<extra></extra>",
                ),
                row=1,
                col=_col,
            )

        line([r.ideal for r in rows], "ideal", "ideal (one-arm / N)", dash="dot")
        line(
            [r.bound for r in rows],
            "bound",
            "lower bound",
            dash="dash",
            hover=[f"limit: {binding_text(r.binding)}" for r in rows],
        )
        line(
            [r.planned for r in rows],
            "planned",
            "planned (HEFT)",
            hover=[f"speed-up {r.planned_speedup:.2f}x" for r in rows],
        )
        meas = [r for r in rows if r.measured]
        if meas:
            fig.add_trace(
                go.Scatter(
                    x=[r.n for r in meas],
                    y=[r.mean for r in meas],
                    name="measured (fake arms, mean, min-max)",
                    legendgroup="measured",
                    showlegend=first,
                    mode="markers",
                    marker={
                        "color": COLORS["measured"],
                        "size": 10,
                        "line": {"color": "#fcfcfb", "width": 2},
                    },
                    error_y={
                        "type": "data",
                        "symmetric": False,
                        "array": [max(r.measured) - r.mean for r in meas],
                        "arrayminus": [r.mean - min(r.measured) for r in meas],
                        "color": COLORS["measured"],
                        "thickness": 2,
                        "width": 6,
                    },
                    customdata=[
                        [min(r.measured), max(r.measured), r.measured_speedup or 0.0] for r in meas
                    ],
                    hovertemplate="measured: %{y:.1f} s (%{customdata[0]:.1f}-%{customdata[1]:.1f})"
                    "<br>speed-up %{customdata[2]:.2f}x<extra></extra>",
                ),
                row=1,
                col=col,
            )
        # Mark where the limit changes, e.g. "input bin is the bottleneck".
        for prev, r in zip(rows, rows[1:], strict=False):
            if r.binding != prev.binding:
                fig.add_vline(
                    x=r.n - 0.5,
                    line={"color": "#b5b4ad", "width": 1, "dash": "dot"},
                    row=1,
                    col=col,
                )
                fig.add_annotation(
                    x=r.n - 0.5,
                    y=max(x.planned for x in rows),
                    text=f"from {r.n} arms:<br>{binding_text(r.binding)}",
                    showarrow=False,
                    xanchor="left",
                    yanchor="top",
                    align="left",
                    font={"size": 12, "color": "#52514e"},
                    row=1,
                    col=col,
                )
        fig.update_xaxes(title_text="arms", dtick=1, row=1, col=col)
        fig.update_yaxes(title_text="finish time (s)" if first else None, rangemode="tozero")
    fig.update_layout(
        template="plotly_white",
        height=520,
        margin={"l": 60, "r": 20, "t": 60, "b": 90},
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.18, "x": 0},
        font={"family": "system-ui, sans-serif", "color": "#0b0b0b"},
        paper_bgcolor="#fcfcfb",
        plot_bgcolor="#fcfcfb",
    )
    chart = fig.to_html(full_html=False, include_plotlyjs=True)
    table = _html_table([r for rows in by_task.values() for r in rows])
    path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Scaling benchmark</title><style>"
        "body{font-family:system-ui,sans-serif;background:#fcfcfb;color:#0b0b0b;"
        "margin:0 auto;padding:16px;max-width:1200px}"
        "p{color:#52514e}table{border-collapse:collapse;font-size:14px;margin-top:12px}"
        "th,td{padding:4px 10px;border-bottom:1px solid #e4e3df;text-align:right}"
        "th:first-child,td:first-child,td:nth-child(5){text-align:left}"
        ".wrap{overflow-x:auto}</style></head><body>"
        "<h1>Finish time vs number of arms</h1>"
        f"<p>{html.escape(meta)}</p>{chart}<div class='wrap'>{table}</div></body></html>"
    )


def _html_table(rows: list[Row]) -> str:
    head = (
        "task", "arms", "planned (s)", "lower bound (s)", "limit", "ideal (s)",
        "measured (s)", "min-max (s)", "planned speed-up", "measured speed-up",
    )  # fmt: skip
    body = []
    for r in rows:
        cells = (
            r.task,
            str(r.n),
            f"{r.planned:.1f}",
            f"{r.bound:.1f}",
            binding_text(r.binding),
            f"{r.ideal:.1f}",
            _f(r.mean),
            f"{min(r.measured):.1f}-{max(r.measured):.1f}" if r.measured else "-",
            f"{r.planned_speedup:.2f}x",
            _f(r.measured_speedup, "{:.2f}x"),
        )
        body.append("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in cells) + "</tr>")
    th = "".join(f"<th>{h}</th>" for h in head)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "scaling")
    ap.add_argument("--max-arms", type=int, default=8)
    ap.add_argument("--measure-max", type=int, default=4, help="measure on fake arms up to N")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--noise", type=float, default=0.1, help="fake-arm duration noise (+-)")
    ap.add_argument("--scale", type=float, default=25.0, help="clock speed-up for fake arms")
    ap.add_argument("--no-measure", action="store_true", help="planning only, no executor runs")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    logs = args.out / "logs"
    logs.mkdir(exist_ok=True)
    by_task: dict[str, list[Row]] = {}
    t0 = time.perf_counter()
    for name, path in TASKS.items():
        graph = load(path)
        rows = plan_rows(name, graph, range(1, args.max_arms + 1))
        if not args.no_measure:
            measure(rows, graph, args, logs)
        by_task[name] = rows
    all_rows = [r for rows in by_task.values() for r in rows]

    meta = (
        f"Planned: HEFT. Measured: executor on fake arms, noise +-{args.noise:.0%}, "
        f"{args.trials} trials for N = 1..{args.measure_max}, clock x{args.scale:g}. "
        "Durations are the task files' placeholders. Speed-up = one-arm time / finish."
    )
    if args.no_measure:
        meta = "Planned: HEFT. No executor runs. Speed-up = one-arm time / finish."
    print(markdown(all_rows))
    print()
    print(meta)
    write_csv(all_rows, args.out / "scaling.csv")
    write_html(by_task, args.out / "scaling.html", meta)
    print(f"\nWrote {args.out / 'scaling.csv'} and {args.out / 'scaling.html'}", end="")
    print(f" in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
