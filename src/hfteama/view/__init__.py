"""The event log turned into what the dashboard shows (`state.build`). Owner: Operations (#17).

The screen itself is `hfteama.orchestra` (one dashboard since 22 Nov). This package keeps the pure
functions it reads the run with: phase, countdown, estimate and what-if, the limit in words, plan
against actual, the re-planning moment and the table map, all from `runs/<time>/events.jsonl`.
"""
