"""Live view for judges (Plotly/Dash) and engineers (Rerun). Reads only the event log.

Owner: Operations (#21, backlog #17).

- Gantt per arm: planned bars faint, actual bars solid, zone waits hatched, failures red,
  every lane labelled REAL or SIM.
- Big numbers: predicted finish time, lower bound, and the binding limit in words
  (e.g. "more arms won't help: the shared input bin is the bottleneck").
- Buttons for the demo: add sim arm, remove arm (slow stop), inject failure.
- Scaling plot: finish time vs N = 1..8 (sim sweep), real 1/2/3-arm points, ideal 1/N line.

Done when: replaying a recorded event log draws the full Gantt and the numbers update on re-plan.
"""
