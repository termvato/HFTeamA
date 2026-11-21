# Tasks

The orchestrator is task-agnostic. A task is a folder here; the core (`src/hfteama/`) never changes
to add one.

## A task folder

```
tasks/<name>/
  <batch>.json   task graph: steps, order (deps), shared zones, which arms can reach, durations
  poses.json     named poses per arm (taught joint vectors or table-frame poses)   [when on real arms]
  skills.py      only if the task needs a skill the core does not have             [optional]
  README.md      what it is, setup on the table, what limits its speed-up
```

## Adding a task in 4 steps

1. Split the job into steps an arm does alone (or two arms together: `needs_arms: 2`).
2. Write the graph: `deps` for order, `zones` for shared table regions (a bin, a fixture, a tray),
   `same_arm_as` when an arm must keep holding the object, `allowed_arms` when reach differs.
3. Reuse generic skills (`pick`, `place`, `pick_place`, `inspect`, `park`); add new ones with
   `@skill("name")` from `hfteama.arms.skills`.
4. Run it on fake arms first (no hardware), then MuJoCo, then real arms.

## What makes a task speed up with more arms

The lower bound `max(work / N, longest chain, busiest zone)` tells you before you build:
- many independent units (sorting, kitting, packing): scales until a shared zone saturates.
- long ordered chains (assembly): capped by the chain, whatever N is.

## Included

| Folder | Batch | Shape | Limit at 3+ arms |
|---|---|---|---|
| `sort/` | `sort12.json` | 12 independent pick then place, one shared input bin | input bin |
| `assemble/` | `assemble3.json` | 3 products, 5 ordered steps each, shared output tray | step order (at 4+ arms) |

Durations are placeholders until measured on the real arms.
